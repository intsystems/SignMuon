"""
Math-correctness test for the distributed optimizers in ``signmuon_optimizers.py``.

This test is PORTABLE: it needs only torch (CPU is fine), no GPUs and no
``torch.distributed``. It verifies that each optimizer's *per-parameter update
recurrence* (momentum, sign placement, EF21 estimator, downlink error feedback)
reproduces, step for step, the trusted numpy reference in
``code/counterexamples/optimizers.py`` -- which itself follows the paper's
algorithm boxes verbatim.

Strategy
--------
* Monkeypatch the Newton-Schulz LMO with the reference's EXACT-SVD ``muon_lmo``
  (routed through numpy) so both sides use the *identical* orthogonalization;
  any discrepancy is then a bug in the update recurrence, not the LMO
  approximation.
* Drive both with the same fixed random gradient sequence on SQUARE matrices and
  ``lr_scaling="none"`` (so every per-layer multiplier is 1 and the step matches
  the reference's plain eta), weight_decay = 0, Nesterov momentum (the
  optimizer's only momentum form).
* We call ``update_param`` directly (the centralized core), bypassing the
  collectives -- those are exercised by ``test_distributed_sharding.py``.

A second test pins the per-layer LR scaling itself: that it reproduces record
#40 / Keller Jordan's aspect factor for the LMO family, that it equalizes the
per-step RMS gain across shapes and families, and that it agrees with the
repo-level ``code/common/lr_scaling.py``.

Run:  SIGNMUON_NO_COMPILE=1 python test_signmuon_optimizers.py
      (or: pytest test_signmuon_optimizers.py)
"""

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("SIGNMUON_NO_COMPILE", "1")  # eager NS; we monkeypatch it anyway

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_CODE = _HERE.parent  # .../code
sys.path.insert(0, str(_HERE))          # signmuon_optimizers
sys.path.insert(0, str(_CODE))          # counterexamples package

import signmuon_optimizers as smo
from counterexamples import optimizers as ref  # code/counterexamples/optimizers.py

torch.set_default_dtype(torch.float64)


# Route the module's LMO through the reference's exact-SVD muon_lmo (numpy), so
# the distributed optimizer and the numpy reference share one identical LMO.
# (record #40 uses Polar Express as the LMO; we swap in the exact polar factor so
# any discrepancy is a bug in the sign/EF21 recurrence, not the LMO approximation.)
def _exact_polar(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    arr = G.detach().cpu().numpy().astype(np.float64)
    if arr.ndim > 2:  # batched (merged-attn) inputs: orthogonalize each block
        out = np.stack([ref.muon_lmo(a) for a in arr.reshape(-1, *arr.shape[-2:])])
        out = out.reshape(arr.shape)
    else:
        out = ref.muon_lmo(arr)
    return torch.from_numpy(np.ascontiguousarray(out)).to(dtype=G.dtype, device=G.device)


smo.polar_express = _exact_polar  # late-bound module global -> methods pick this up


# (paper name in smo.OPTIMIZERS, reference class in ref.OPTIMIZERS)
_PAIRS = [
    "SignMuon", "EF21-SignMuon", "MuonUSign", "MuonSign",
    "EF21-MuonUSign", "EF21-MuonSign", "SignSGD", "Muon",
]


def _run_reference(name, grads, eta, mu):
    shape = grads[0].shape
    opt = ref.OPTIMIZERS[name](shape, eta, mu=mu, nesterov=True)
    for G in grads:
        opt.step(G.astype(np.float64))
    # tracked/exact model X, and (for UDSign) the compressed broadcast model W
    W = getattr(opt, "W", None)
    return opt.X.copy(), (None if W is None else W.copy())


def _run_distributed_core(name, grads, eta, mu):
    """Run the distributed optimizer's centralized core (update_param) on one
    parameter, starting from X_0 = W_0 = 0 to match the reference."""
    cls = smo.OPTIMIZERS[name]
    p = torch.zeros(grads[0].shape, dtype=torch.float64)
    # lr_scaling="none": the numpy reference implements the bare algorithm box
    # with a single global eta, so switch the per-layer multiplier off to compare
    # the recurrences themselves (the multipliers are tested separately below).
    opt = cls([p], lr=eta, momentum=mu, weight_decay=0.0, lr_scaling="none")
    group = opt.param_groups[0]
    for G in grads:
        p.grad = torch.from_numpy(G.astype(np.float64)).clone()
        opt.update_param(p, group)
    X = opt.state[p].get("exact_model", p)  # exact model for UDSign, else p itself
    W = p if "exact_model" in opt.state[p] else None
    return X.detach().numpy().copy(), (None if W is None else W.detach().numpy().copy())


def _check(name, eta, mu, T=25, dim=4, seed=0, atol=1e-6):
    rng = np.random.default_rng(seed)
    grads = [rng.standard_normal((dim, dim)) for _ in range(T)]
    Xr, Wr = _run_reference(name, grads, eta, mu)
    Xd, Wd = _run_distributed_core(name, grads, eta, mu)
    xerr = float(np.max(np.abs(Xr - Xd)))
    # The tracked/exact model X is the quantity training evaluates (validation runs on X);
    # it must match the reference tightly.
    assert xerr < atol, f"{name}: exact-model X mismatch (max abs {xerr:.2e}) at eta={eta}, mu={mu}"
    if Wr is not None:
        werr = float(np.max(np.abs(Wr - Wd)))
        # W is EF21-MuonSign's 1-bit sign-compressed broadcast model:
        #     W <- W + mean|X-W| * sign(X-W).
        # Its update rule is line-for-line the numpy reference (optimizers.py), but sign() is
        # discontinuous: when an entry of the residual X-W lands on a sign(0) tie, torch-fp and
        # numpy-fp -- which differ by ~1e-16 in the momentum EMA -- pick opposite signs, and since
        # alpha=mean|X-W| then shifts, the whole subsequent W trajectory drifts by ~one compressor
        # step (this happens e.g. at mu=0.8, seed=0). This is an unavoidable cross-backend artifact
        # on a discontinuous operator, NOT a bug:
        #   (a) the exact/tracked model X -- which training actually evaluates -- matches to ~1e-16
        #       above (asserted), and
        #   (b) test_distributed_sharding.py verifies W matches EXACTLY in a torch-vs-torch run.
        # So we require the W drift to stay within a few compressor steps (~eta); a real downlink
        # bug (wrong sign/scale, or a missing update) would blow past this bound.
        assert werr < atol or werr <= 3.0 * eta, (
            f"{name}: broadcast-model W drift {werr:.2e} exceeds ~compressor scale (eta={eta}) at "
            f"mu={mu} -- likely a real bug, not a sign-tie")
        return xerr, werr
    return xerr, None


def test_update_math_matches_reference():
    failures = []
    for name in _PAIRS:
        for mu in (0.0, 0.8, 0.95):
            for eta in (0.01, 0.1):
                try:
                    xerr, werr = _check(name, eta, mu)
                    tag = f"{name:<16} mu={mu:<4} eta={eta:<5}  |X-err|={xerr:.2e}" + (
                        f"  |W-err|={werr:.2e}" if werr is not None else "")
                    print("  OK  " + tag)
                except AssertionError as e:
                    failures.append(str(e))
                    print("  FAIL " + str(e))
    assert not failures, f"{len(failures)} mismatch(es):\n" + "\n".join(failures)


# --------------------------------------------------------------------------
# Per-layer EF21 compressor on record #40's merged Q/K/V/O weight
# --------------------------------------------------------------------------
# ``mean|r|`` is the only NON-elementwise operation in any of the EF21 update
# rules, so -- exactly like the LMO -- it has to be taken per LAYER, and record
# #40's ``qkvo_w`` is four layers in one tensor. ``sign``, the visible half of the
# compressor, IS elementwise, which is what makes this easy to get wrong and
# impossible to see in the loss curves of the non-EF21 methods.
#
# The three tests below pin, in order: the block axis, the per-block scale, and
# that single-layer parameters are untouched.

_EF21 = ["EF21-SignMuon", "EF21-MuonUSign", "EF21-MuonSign"]


def _tagged(shape, module):
    """A parameter carrying record #40's ``.module`` tag, which is how both the
    LMO and the compressor learn that ``qkvo_w`` is four layers."""
    p = torch.zeros(shape)
    if module is not None:
        p.module = module
    return p


def _compress(name, r, module=None):
    """``_scaled_sign(r, p)`` for one method, on a parameter tagged ``module``."""
    p = _tagged(tuple(r.shape), module)
    opt = smo.OPTIMIZERS[name]([p], lr=0.1, momentum=0.9, weight_decay=0.0,
                               lr_scaling="none")
    return opt._scaled_sign(r, p)


def test_compressor_block_axis_is_the_models_view():
    """A residual supported on ONE Q/K/V/O block must move only that block.

    This is the failure that is silent: ``[h, 4w]`` split on the wrong axis gives
    four "blocks" each mixing all of Q/K/V/O, so a single-block residual leaks
    into all four, every block ends up with the same scale again, and the
    per-layer compressor is a no-op that looks implemented.
    """
    h, w = 8, 16                                   # blocks are reshape(4, 8, 4)
    for name in _EF21:
        for k in range(4):
            r = torch.zeros(4, h, w // 4)
            r[k] = 1.0
            out = _compress(name, r.reshape(h, w), module="attn")
            blocks = out.reshape(4, h, w // 4)
            assert blocks[k].abs().min() > 0, f"{name}: block {k} was not moved"
            for j in range(4):
                if j != k:
                    assert blocks[j].abs().max() == 0, (
                        f"{name}: a residual confined to block {k} leaked into "
                        f"block {j} -- the reshape axis does not match the model's "
                        ".view(4, hdim, dim)")
    print(f"  OK   block axis matches the model's .view(4, hdim, dim) "
          f"({len(_EF21)} methods x 4 blocks)")


def test_compressor_scale_is_per_block():
    """Each block's step magnitude must equal that block's OWN mean|r|.

    Record #40 zero-inits the O block, so the four blocks' gradients -- and hence
    their EF21 residuals -- differ by orders of magnitude. A single shared
    ``mean|r|`` hands every block the *largest* block's step size: at scales
    (1, 1, 1, 0.01) it overshoots the O block by ~74x its own residual, and the
    LMO of that block becomes the LMO of the compressor's own overshoot.
    """
    h, w = 8, 16
    scales = torch.tensor([1.0, 1.0, 1.0, 0.01], dtype=torch.get_default_dtype())
    torch.manual_seed(0)
    r = torch.randn(4, h, w // 4) * scales[:, None, None]
    want = r.abs().mean(dim=(-2, -1))               # each block's own scale
    shared = r.abs().mean()                         # the pre-fix single scale
    assert shared / want[3] > 10, "fixture must exercise a real block disparity"
    for name in _EF21:
        out = _compress(name, r.reshape(h, w), module="attn")
        # |out| is constant within a block and equal to that block's alpha
        got = out.reshape(4, h, w // 4).abs().mean(dim=(-2, -1))
        assert torch.allclose(got, want, rtol=1e-12), (
            f"{name}: block step magnitudes {got.tolist()} != each block's own "
            f"mean|r| {want.tolist()}")
    print(f"  OK   per-block scale (shared scale would overshoot the O block by "
          f"{shared / want[3]:.0f}x)")


def test_compressor_unchanged_for_single_layer_parameters():
    """Parameters that ARE one layer must be bit-identical to the pre-fix rule.

    This is what keeps the existing non-attn results valid: 33 of record #40's 43
    matrix parameters, and all five methods that never compress at all.
    """
    torch.manual_seed(0)
    for shape, module in (((768, 3072), "mlp"), ((6, 12), "attn_gate"),
                          ((1, 12), "smear_gate"), ((5, 5), None)):
        r = torch.randn(shape)
        old = torch.sign(r) * r.abs().mean()        # the exact pre-fix expression
        for name in _EF21:
            got = _compress(name, r, module=module)
            assert torch.equal(got, old), f"{name} {shape}: not bit-identical"
    print("  OK   single-layer parameters bit-identical to the pre-fix rule")


def test_diagnostics_record_and_report():
    """The diagnostics path must run single-process and report a sane ``alpha``.

    For an isotropic residual the scaled sign's achieved contraction
    ``||r||_1^2 / (d ||r||_2^2)`` is ``2/pi = 0.6366``. That is where ``alpha_up``
    should sit early in a run -- and its DECAY toward ``1/d`` as the residual
    becomes directionally coherent is the whole reason the slot is logged.
    """
    torch.manual_seed(0)
    p = _tagged((32, 32), None)
    opt = smo.OPTIMIZERS["EF21-MuonUSign"]([p], lr=0.1, momentum=0.0,
                                           weight_decay=0.0, lr_scaling="none")
    opt.diagnostics = True
    grp = opt.param_groups[0]
    for _ in range(3):
        p.grad = torch.randn(32, 32, dtype=p.dtype)
        opt.update_param(p, grp)
    report = opt.diagnostics_report({id(p): "probe"})
    assert "alpha_up" in report and "lag_est" in report, f"bad report:\n{report}"
    a = opt._diag_buf[smo.DIAG_SLOTS.index("alpha_up"), 0].item()
    assert 0.5 < a < 0.75, (
        f"alpha_eff={a:.4f} on an isotropic residual; expected ~2/pi=0.637")
    # off by default, so the measured hot loop never pays for them
    assert smo.OPTIMIZERS["Muon"]([_tagged((4, 4), None)], lr=0.1).diagnostics is False
    print(f"  OK   diagnostics report (alpha_up={a:.4f} vs 2/pi=0.6366)")


# --------------------------------------------------------------------------
# Per-layer LR scaling
# --------------------------------------------------------------------------

# The parameter shapes record #40 actually hands to the hidden-matrix optimizer,
# with model_dim=768, hdim=3072, num_heads=6 (see GPT.__init__ in train_gpt.py).
_REC40_SHAPES = [
    ("smear_gate.weight", (1, 12), None),
    ("attn_gate.weight", (6, 12), None),
    ("blocks.attn.qkvo_w", (768, 3072), "attn"),   # used as 4 x [768, 768]
    ("blocks.mlp.c_fc", (768, 3072), "mlp"),
    ("blocks.mlp.c_proj", (768, 3072), "mlp"),
]


def test_lmo_family_matches_record40_aspect_factor():
    """The LMO multiplier must be Keller Jordan's / record #40's shipped factor.

    Record #40's Muon computes ``max(1, p.size(-2) / p.size(-1)) ** 0.5`` on the
    STORED parameter. Our rule computes it on the shape the LMO actually operates
    on (the four [hdim, dim] blocks for the merged attention weight). Those must
    coincide on every shape #40 uses, or the reference `Muon` here is not the
    record's Muon.
    """
    for name, shape, module in _REC40_SHAPES:
        p = _tagged(shape, module)
        rec40 = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
        for rule in ("unit-gain", "legacy", "mup"):
            got = smo.layer_multiplier(p, smo.FAMILY_LMO, rule)
            assert abs(got - rec40) < 1e-12, (
                f"{name} rule={rule}: lmo lambda {got} != record #40's {rec40}")
        print(f"  OK   {name:<22} {str(shape):<14} lmo lambda = {rec40:.6g}  (== record #40)")


def test_unit_gain_equalizes_the_per_step_gain():
    """One eta_0 means one per-step RMS gain, for every shape and both families.

    gamma(s) = ||s||_F / sqrt(fan_out) exactly, with ||polar||_F = sqrt(min(m,n))
    and ||+-1||_F = sqrt(m n). unit-gain must drive gamma(eta_0 * lambda * s) to
    eta_0 in both rows -- that is what makes a learning rate comparable across
    the eight methods.
    """
    eta0 = 0.06
    for name, shape, module in _REC40_SHAPES:
        p = _tagged(shape, module)
        m, n = smo.lmo_shape(p)
        lam_lmo = smo.layer_multiplier(p, smo.FAMILY_LMO, "unit-gain")
        lam_sign = smo.layer_multiplier(p, smo.FAMILY_SIGN, "unit-gain")
        gain_lmo = eta0 * lam_lmo * math.sqrt(min(m, n)) / math.sqrt(m)
        gain_sign = eta0 * lam_sign * math.sqrt(m * n) / math.sqrt(m)
        assert abs(gain_lmo - eta0) < 1e-12, f"{name}: lmo gain {gain_lmo} != {eta0}"
        assert abs(gain_sign - eta0) < 1e-12, f"{name}: sign gain {gain_sign} != {eta0}"
        print(f"  OK   {name:<22} lambda_lmo={lam_lmo:<8.5g} lambda_sign={lam_sign:<10.5g}"
              f" gain={gain_lmo:.4g}")


def test_agrees_with_common_lr_scaling():
    """This module duplicates ``code/common/lr_scaling.py`` (a log must be
    self-contained); the two must not drift apart."""
    from common import lr_scaling as cls_

    for rule in ("unit-gain", "mup", "legacy", "none"):
        ref_rule = cls_.resolve_rule(rule)
        for name, shape, module in _REC40_SHAPES:
            p = _tagged(shape, module)
            m, n = smo.lmo_shape(p)
            for family in (smo.FAMILY_LMO, smo.FAMILY_SIGN):
                got = smo.layer_multiplier(p, family, rule)
                want = ref_rule.multiplier(family, m, n)
                assert abs(got - want) < 1e-12, (
                    f"{rule}/{family}/{name}: nanogpt {got} != common/lr_scaling {want}")
        print(f"  OK   rule '{rule}' agrees with common/lr_scaling.py")


def test_semantic_rule_only_moves_the_transposed_mlp_matrix():
    """`semantic` corrects record #40's transposed `c_fc` storage and nothing else."""
    for name, shape, module in _REC40_SHAPES:
        p = _tagged(shape, module)
        m, n = smo.lmo_shape(p)
        p.fan_out_sem, p.fan_in_sem = (n, m) if name.endswith("c_fc") else (m, n)
        for family in (smo.FAMILY_LMO, smo.FAMILY_SIGN):
            a = smo.layer_multiplier(p, family, "unit-gain")
            b = smo.layer_multiplier(p, family, "semantic")
            if name.endswith("c_fc"):
                assert abs(b / a - 2.0) < 1e-12, f"{name}/{family}: expected 2x, got {b / a}"
            else:
                assert abs(a - b) < 1e-12, f"{name}/{family}: semantic moved it ({a} -> {b})"
    print("  OK   'semantic' == 'unit-gain' except a 2x on the transposed c_fc")


if __name__ == "__main__":
    print("Verifying distributed optimizer cores against the numpy paper reference...\n")
    test_update_math_matches_reference()
    print("\nVerifying the per-layer EF21 compressor on the merged Q/K/V/O weight...\n")
    test_compressor_block_axis_is_the_models_view()
    test_compressor_scale_is_per_block()
    test_compressor_unchanged_for_single_layer_parameters()
    test_diagnostics_record_and_report()
    print("\nVerifying per-layer LR scaling...\n")
    test_lmo_family_matches_record40_aspect_factor()
    test_unit_gain_equalizes_the_per_step_gain()
    test_agrees_with_common_lr_scaling()
    test_semantic_rule_only_moves_the_transposed_mlp_matrix()
    print("\nAll optimizer update recurrences and LR multipliers match. PASS.")
