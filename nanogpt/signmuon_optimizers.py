"""
Distributed (data-parallel, NOT federated) implementations of the six paper
optimizers plus SignSGD and a reference Muon, packaged for the modded-nanogpt
speedrun.  This build targets **record #40** (2025-10-04, the last record before
NorMuon), whose hidden-matrix optimizer is still a *clean, separable* Muon:

    momentum (Nesterov EMA)  ->  Polar Express orthogonalization (LMO)  ->  step

so the paper's sign/EF21 variants inject exactly at the LMO, unchanged in spirit
from the earlier classic-record port.

    SignMuon         sign AFTER the LMO          X <- X - eta * sign(PE(M))
    EF21-SignMuon    EF21 on the LMO direction   X <- X - eta * d_est,  d_est ~ PE(M)
    MuonUSign        sign BEFORE the LMO         X <- X - eta * PE(sign(M))
    MuonSign         sign BEFORE and AFTER LMO   X <- X - eta * sign(PE(sign(M)))
    EF21-MuonUSign   EF21 on the momentum        X <- X - eta * PE(g_est), g_est ~ M
    EF21-MuonSign    bidirectional EF21          exact X step + sign-compressed broadcast W
    SignSGD          sign of the momentum        X <- X - eta * sign(M)
    Muon             reference (no compression)  X <- X - eta * PE(M)

Every method's learning rate is per-layer scaled so that one ``eta_0`` means the
same thing (the per-step RMS gain) for all eight -- see "Per-layer learning-rate
scaling" below.  For the LMO family that scaling IS record #40's own aspect
factor, so ``Muon`` here is the record verbatim.

Here ``M`` is the (Nesterov) heavy-ball momentum of the *averaged* gradient and
``PE`` is the Muon Polar-Express orthogonalization (approximate polar factor).
The math for every method is verbatim the centralized algorithm boxes of the
paper and their numpy reference in ``code/counterexamples/optimizers.py`` --
only the *systems* layer differs.

-----------------------------------------------------------------------------
Two things that are specific to the record-#40 model
-----------------------------------------------------------------------------
1. Polar Express, not Newton-Schulz.  Record #40 replaced the Newton-Schulz
   iteration (record <=#37) with Polar Express (record #38), a different quintic
   whose ``coeffs_list`` are baked in below.  We reproduce it in **pure torch**
   (no Triton) so it (a) runs identically on A100 and H100 and (b) is unit-
   testable on CPU.  The Triton kernels in the upstream record are a *speed*
   optimization only; the arithmetic here is the same up to bf16 rounding.

2. Merged ``qkvo_w`` attention weight.  Record #40 stores Q/K/V/O in a single
   ``[hdim, 4*dim]`` parameter (so it batches with the ``[dim, 4*dim]`` MLP
   weights for one reduce_scatter) but the model *always* accesses it through
   ``.view(4, hdim, dim)``.  Muon must therefore orthogonalize the FOUR
   ``[hdim, dim]`` sub-matrices independently -- NOT the merged matrix.  We
   detect this via the ``.module == "attn"`` tag the model attaches to the
   parameter and reshape around the two operations that are NOT elementwise:

     * the LMO itself (:meth:`_DistributedMatrixOptimizer._lmo`), and
     * the EF21 compressor's scale ``mean|.|``
       (:meth:`_DistributedMatrixOptimizer._scaled_sign`).

   The second one matters as much as the first and is easy to miss, because
   ``sign`` -- the visible half of the compressor -- *is* elementwise.  One
   ``mean|r|`` shared across the four blocks hands every block the step size
   implied by the LARGEST, so a block whose residual sits well below that mean has
   its estimator driven by the compressor's own overshoot rather than by its own
   residual, and the LMO of that block becomes the LMO of noise.  Record #40
   zero-inits the O block (``qkvo_w.view(4,hdim,dim)[3].zero_()``), so at
   initialization no gradient reaches Q/K/V at all while O's is finite: the
   disparity is unbounded exactly where it does the most damage.  The paper's
   compressor is per-LAYER and these are four layers, so it gets four scales.

   This affects only the three EF21 methods.  The non-EF21 sign methods transmit a
   bare entrywise ``sign``, which is scale-free and therefore identical whether it
   is read as one matrix or four; ``Muon`` never compresses at all.

   Same-shaped MLP weights (``.module == "mlp"``) are one layer each and are
   orthogonalized -- and compressed -- as a single matrix, as the model uses them.

-----------------------------------------------------------------------------
Distributed vs. federated -- the part that actually changes
-----------------------------------------------------------------------------
In the FEDERATED code (``code/federated/algorithms.py``) each client keeps its
own momentum / EF21 estimator, compresses ITS OWN update, ships the compressed
message, and the server aggregates the *compressed* messages (majority vote, or
an average of scaled-signs). Compression sits on the wire between many clients.

In DISTRIBUTED data-parallel training (this file, following the modded-nanogpt
Muon) there is exactly one logical model. The ranks hold different data shards;
``dist.reduce_scatter(..., op=AVG)`` averages their gradients so that -- for each
parameter -- a single owning rank receives the *true global mean gradient*. That
owning rank then runs the ordinary CENTRALIZED update (momentum, LMO, sign, EF21
estimator, ...) and ``dist.all_gather`` puts the updated parameter back on every
rank.

Consequences that make this correct and simple:
  * There is ONE momentum buffer / EF21 estimator per parameter, living on the
    rank that owns that parameter (via ``self.state[p]``) -- never per-rank
    replicas that need reconciling. Parameters are sharded round-robin: rank r
    owns ``params[base_i + r]`` inside every ``world_size``-sized chunk.
  * The 1-bit "compression" here is a property of the update RULE (sign / EF21),
    not of the rank-to-rank transport (gradients cross the wire in full
    precision via reduce_scatter). This is the honest distributed analog of the
    centralized algorithms the paper analyzes -- see the README for the full
    argument.
  * ``EF21-MuonSign`` additionally keeps an *exact* server model ``X`` as
    optimizer state and lets the live parameters be the sign-compressed
    broadcast model ``W``; the gradient is naturally evaluated at ``W`` (the
    forward pass uses the live parameter) exactly as the downlink EF21-P scheme
    requires. ``swap_in_exact()`` / ``swap_out_exact()`` expose ``X`` for
    evaluation.

The reduce_scatter / all_gather transport is world-size agnostic (it runs on 1
GPU just as on 8): on a single process the collectives are no-ops and every
parameter is updated locally with the (grad-accumulated) mean gradient.  That is
exactly what makes the single-A100 build reproduce the 8xH100 optimizer
trajectory -- see train_gpt_a100.py.
"""

from __future__ import annotations

import math
import os

import torch
from torch import Tensor
from torch.optim import Optimizer
import torch.distributed as dist

__all__ = [
    "polar_express",
    "zeropower_via_newtonschulz5",  # backward-compatible alias for polar_express
    "Muon",
    "SignSGD",
    "SignMuon",
    "MuonUSign",
    "MuonSign",
    "EF21SignMuon",
    "EF21MuonUSign",
    "EF21MuonSign",
    "OPTIMIZERS",
    "PAPER_METHODS",
    "FAMILY_LMO",
    "FAMILY_SIGN",
    "LR_SCALING_RULES",
    "DIAG_SLOTS",
    "lmo_shape",
    "layer_multiplier",
    "describe_lr_scaling",
]

# ---------------------------------------------------------------------------
# Polar Express orthogonalization (Muon LMO), record #40's ``coeffs_list``.
# Pure torch (no Triton) so it runs on any device and is CPU-testable.
# torch.compile can be disabled with SIGNMUON_NO_COMPILE=1 (the CPU tests do
# this and then monkeypatch this function with an exact-SVD polar factor).
# ---------------------------------------------------------------------------

# Computed for num_iters=5, safety_factor=2e-2, cushion=2  (verbatim from record #40)
_POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def sign_pm1(x):
    """Elementwise sign with exact zeros mapped to independent random +-1.

    The paper's convention: every transmitted sign message is exactly one bit
    per parameter, so ``sign(0) = 0`` -- a third symbol -- is not allowed. Zeros
    are not exotic on this model: record #40 zero-initializes ``c_proj`` and the
    ``O`` block of ``qkvo_w``, and a zero output projection makes the gradient of
    everything feeding it EXACTLY zero, so at step 0 the whole ``c_fc`` and
    ``Q/K/V`` set has zero momentum and therefore zero sign.

    Differs from ``common.optimizers.sign_pm1`` in one respect: that one tests
    ``(x == 0).any()`` first and draws only when it fires, which is right for the
    CIFAR loops but is a device-to-host SYNC here -- once per sign per OWNED
    parameter, so a handful per rank per step, inside the very loop the speedrun
    clock measures. We draw unconditionally instead: one elementwise pass against
    the LMO's five matmuls. Nothing else in the training loop consumes the torch
    RNG after initialization (the data loader walks the shards in order), so
    always advancing the stream costs no reproducibility.

    ``empty_like(...).bernoulli_(0.5)`` rather than ``randint``: it inherits
    ``x``'s dtype and device without a cast, and the parameters here are both
    fp32 (the raw hidden matrices) and bf16 (the gates, which record #40 casts
    with every ``nn.Linear``) -- and float64 under the CPU tests.
    """
    s = torch.sign(x)
    r = torch.empty_like(s).bernoulli_(0.5).mul_(2).sub_(1)
    return torch.where(s == 0, r, s)


def _polar_express_eager(G: Tensor, steps: int = 5) -> Tensor:
    """Polar Express iteration approximating the orthogonal polar factor
    ``U V^T`` of ``G = U S V^T`` (the Muon LMO), record #40's coefficients.

    Runs internally in bfloat16 (as the record does) and returns a tensor in
    ``G``'s original dtype.  Supports a leading batch dimension so the four
    stacked Q/K/V/O blocks of the merged attention weight are orthogonalized in
    a single call.  ``steps`` is accepted for signature compatibility with the
    old Newton-Schulz LMO but ignored -- the number of iterations is fixed by
    ``_POLAR_EXPRESS_COEFFS``.

    Pure-torch equivalent of the upstream Triton kernels; same arithmetic up to
    bf16 rounding, which only perturbs an LMO whose spectrum is deliberately
    non-unit (~Uniform(0.68, 1.13)) anyway.
    """
    assert G.ndim >= 2, "Muon orthogonalization expects at least 2D tensors."
    in_dtype = G.dtype
    X = G.bfloat16()
    transpose = X.size(-2) > X.size(-1)
    if transpose:
        X = X.mT
    # Ensure spectral norm is at most 1 (record #40's exact normalization).
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6)
    for a, b, c in _POLAR_EXPRESS_COEFFS:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transpose:
        X = X.mT
    return X.to(in_dtype)


if os.environ.get("SIGNMUON_NO_COMPILE", "0") == "1":
    polar_express = _polar_express_eager
else:  # pragma: no cover - exercised on the GPU cluster, not in CPU tests
    # `dynamic=False` is NOT optional -- it is record #40's own comment on this
    # very function ("Must use dynamic=False or else it's much slower").  We call
    # the LMO with four distinct static shapes ([768,3072] mlp, [4,768,768] attn
    # blocks, [6,12] attn_gate, [1,12] smear_gate); without the flag dynamo marks
    # the dims dynamic after the second one and emits a much slower generic
    # kernel for all of them.  Four specializations are well under
    # `dynamo.config.recompile_limit`, which the training script raises to 64.
    polar_express = torch.compile(_polar_express_eager, dynamic=False, fullgraph=True)

# Backward-compatible alias: earlier code / tests referred to the LMO as
# ``zeropower_via_newtonschulz5``.  Record #40 uses Polar Express instead, but
# the *interface* (2D-or-batched tensor in, orthogonal factor out) is identical.
zeropower_via_newtonschulz5 = polar_express


# ---------------------------------------------------------------------------
# Per-layer learning-rate scaling (unit gain).
#
# Self-contained mirror of ``code/common/lr_scaling.py`` -- kept duplicated on
# purpose: every logged run prints this file verbatim, so a log must define its
# own learning rates without reference to the rest of the repo.
# ``test_signmuon_optimizers.py`` asserts the two implementations agree.
#
# The criterion (paper appendix "Per-Layer Step Sizes").  The RMS gain of a step
# matrix ``s in R^{m x n}`` acting on an isotropic input is exactly
# ``gamma(s) = ||s||_F / sqrt(m)``.  Requiring the same per-step gain on every
# layer gives one formula, ``lambda = sqrt(fan_out) / ||s||_F``, and the two
# families have exact Frobenius norms:
#
#     lmo  step ``U V^T``:  ||s||_F = sqrt(min(m, n))  ->  lambda = sqrt(max(1, m/n))
#     sign step ``+-1``:    ||s||_F = sqrt(m n)        ->  lambda = 1 / sqrt(n)
#
# The first line is *exactly* the aspect factor shipped in Keller Jordan's Muon
# and used verbatim by record #40 (`max(1, p.size(-2)/p.size(-1))**0.5`), so the
# reference ``Muon`` here is bit-identical to the record; the second line is the
# counterpart the sign family never had (record #40 has no sign family).  Because
# both lines equalize the same quantity, ONE learning rate ``eta_0`` -- the
# per-step RMS gain -- is directly comparable across all eight methods.
#
# Caveat, inherited from record #40: the rule reads ``(m, n) = (fan_out, fan_in)``
# off the stored tensor, and record #40 stores the MLP ``c_fc`` transposed
# (``[dim, hdim]``, used as ``x @ c_fc``) so it can share a reduce_scatter with
# the attention weight.  For ``c_fc`` the semantic fan_out/fan_in are therefore
# swapped, and both families get a 2x smaller multiplier than a
# ``[fan_out, fan_in]`` reading would give.  This is what record #40 itself does
# for Muon, so it is the default here; ``lr_scaling="semantic"`` corrects it
# (using the ``fan_out_sem`` / ``fan_in_sem`` tags the model attaches) as an
# ablation that changes the Muon baseline away from the record.
# ---------------------------------------------------------------------------

FAMILY_LMO = "lmo"      # final step is polar(.):   ||s||_F = sqrt(min(m, n))
FAMILY_SIGN = "sign"    # final step has +-1 entries: ||s||_F = sqrt(m n)


def lmo_shape(p: Tensor) -> tuple[int, int]:
    """``(m, n)`` of the matrix the LMO actually operates on.

    Record #40 stores Q/K/V/O merged as ``[hdim, 4*dim]`` but the model -- and
    therefore the LMO, see :meth:`_DistributedMatrixOptimizer._lmo` -- uses it as
    four ``[hdim, dim]`` blocks, so the per-layer multiplier must be computed on
    the block shape, not the merged one.
    """
    m, n = int(p.shape[-2]), int(p.shape[-1])
    if getattr(p, "module", None) == "attn":
        n //= 4
    return m, n


def semantic_shape(p: Tensor) -> tuple[int, int]:
    """``(fan_out, fan_in)`` of the linear map the parameter implements.

    Falls back to :func:`lmo_shape` for parameters the model did not tag.
    """
    m = getattr(p, "fan_out_sem", None)
    n = getattr(p, "fan_in_sem", None)
    if m is None or n is None:
        return lmo_shape(p)
    return int(m), int(n)


def _aspect(m: int, n: int) -> float:
    return math.sqrt(max(1.0, m / n))


def _unit_gain(family: str, m: int, n: int) -> float:
    return _aspect(m, n) if family == FAMILY_LMO else 1.0 / math.sqrt(n)


def _mup(family: str, m: int, n: int) -> float:
    return _aspect(m, n) if family == FAMILY_LMO else 1.0 / n


def _legacy(family: str, m: int, n: int) -> float:
    return _aspect(m, n) if family == FAMILY_LMO else 1.0


def _no_scaling(family: str, m: int, n: int) -> float:
    return 1.0


#: name -> (shape accessor, multiplier(family, m, n), one-line description)
LR_SCALING_RULES = {
    "unit-gain": (lmo_shape, _unit_gain,
                  "lmo: sqrt(max(1,m/n)) (== record #40 / Keller Jordan);  sign: 1/sqrt(fan_in)"),
    "semantic": (semantic_shape, _unit_gain,
                 "as unit-gain but on the SEMANTIC (fan_out, fan_in); changes Muon vs record #40"),
    "mup": (lmo_shape, _mup,
            "lmo: sqrt(max(1,m/n));  sign: 1/fan_in  (assumes accumulated sign steps align)"),
    "legacy": (lmo_shape, _legacy,
               "lmo: sqrt(max(1,m/n));  sign: 1  (one global rate for sign steps)"),
    "none": (lmo_shape, _no_scaling,
             "lambda = 1 everywhere (what Mishra et al., Algorithm 1 does)"),
}


def layer_multiplier(p: Tensor, family: str, rule: str = "unit-gain") -> float:
    """Per-layer multiplier ``lambda`` for one parameter (``eta_layer = eta_0 * lambda``)."""
    try:
        shape_of, mult, _ = LR_SCALING_RULES[rule]
    except KeyError:
        raise ValueError(
            f"unknown lr_scaling rule {rule!r}; choose from {sorted(LR_SCALING_RULES)}") from None
    m, n = shape_of(p)
    return mult(family, m, n)


def describe_lr_scaling(optimizer: "_DistributedMatrixOptimizer",
                        names: dict | None = None) -> str:
    """Table of the per-layer multipliers in force, for the run log.

    An unlogged per-layer learning rate is an unreproducible one, and the spread
    is the number a reader will want. ``names`` optionally maps ``id(param)`` to
    a human-readable parameter name.
    """
    import re

    names = names or {}
    rule = optimizer.lr_scaling
    family = optimizer.family
    # Report the (m, n) THIS RULE reads, not always lmo_shape: 'semantic' falls back
    # to the stored shape for any parameter the model forgot to tag, and printing
    # lmo_shape would hide that fallback behind numbers that look deliberate.
    shape_of = LR_SCALING_RULES[rule][0]
    lines = [f"per-layer LR scaling '{rule}' (family={family}): {LR_SCALING_RULES[rule][2]}",
             f"  {'parameter':<32}{'stored':>15}{'rule m':>8}{'rule n':>8}{'lambda':>13}{'count':>7}"]
    seen: dict[tuple, int] = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            m, n = shape_of(p)
            lam = layer_multiplier(p, family, rule)
            # collapse the per-block index so 22 identical MLP matrices are one row
            name = re.sub(r"\.\d+\.", ".*.", names.get(id(p), getattr(p, "module", "?")))
            key = (name, tuple(p.shape), m, n, round(lam, 12))
            seen[key] = seen.get(key, 0) + 1
    mults = []
    for (name, stored, m, n, lam), count in seen.items():
        mults.append(lam)
        lines.append(f"  {name:<32}{str(list(stored)):>15}{m:>8}{n:>8}{lam:>13.6g}{count:>7}")
    if mults:
        lo, hi = min(mults), max(mults)
        lines.append(f"  spread {hi / lo:.2f}x   (min {lo:.6g}, max {hi:.6g})")
        lines.append("  eta_layer = lr * lambda * p.lr_mul; for the lmo family lambda is "
                     "record #40's own aspect factor")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-parameter diagnostics published for the run log.
#
# Each is a scalar recorded on the rank that OWNS the parameter, on the single
# step per validation that the training script flags (see
# ``_DistributedMatrixOptimizer.diagnostics``), so the hot loop and its
# wall-clock are unperturbed.  ``diagnostics_report`` sums them across ranks --
# every non-owning rank contributes zero -- so the table is identical everywhere.
#
# These exist because the convergence threshold of the EF21 methods is governed
# by the compressor's contraction constant ``alpha``, and the scaled sign's
# ``alpha`` is NOT a constant: it is ``||r||_1^2 / (d ||r||_2^2)``, which equals
# ``2/pi`` for an isotropic residual and decays toward ``1/d`` as the residual
# becomes directionally coherent.  ``alpha_up`` / ``alpha_dn`` measure it.
# ---------------------------------------------------------------------------

DIAG_SLOTS = (
    "alpha_up",   # contraction achieved by the uplink (or only) scaled sign
    "alpha_dn",   # ... by the downlink EF21-P scaled sign (EF21-MuonSign)
    "lag_est",    # ||target - estimator||_F / ||target||_F  (all EF21 methods)
    "lag_XW",     # ||X - W||_F / ||W||_F  (EF21-MuonSign's server/broadcast gap)
    "gblk0",      # mean|grad| of Q  (or of the whole tensor, for non-attn params)
    "gblk1",      # mean|grad| of K
    "gblk2",      # mean|grad| of V
    "gblk3",      # mean|grad| of O
)
_DIAG_SLOT_IX = {name: i for i, name in enumerate(DIAG_SLOTS)}


# ---------------------------------------------------------------------------
# Shared distributed base: reduce_scatter -> owning-rank update -> all_gather.
# ---------------------------------------------------------------------------


class _DistributedMatrixOptimizer(Optimizer):
    """Base class handling the modded-nanogpt sharded transport.

    Subclasses implement only :meth:`update_param`, the *centralized*
    per-parameter update, and never touch the collectives. All parameters in
    one param-group share a shape so ``reduce_scatter`` / ``all_gather`` (which
    require equal shapes) are valid.
    """

    #: Step family (see the per-layer LR scaling section above). ``FAMILY_LMO``
    #: for methods whose final step is an orthogonal (LMO) direction,
    #: ``FAMILY_SIGN`` for sign-terminated steps whose entries are already ~= 1.
    #: Together with the parameter shape this fixes the per-layer multiplier.
    family = FAMILY_LMO

    def __init__(self, params, lr, weight_decay=0.0, momentum=0.95,
                 lr_scaling="unit-gain", **extra):
        params = list(params)
        if lr_scaling not in LR_SCALING_RULES:
            raise ValueError(f"unknown lr_scaling {lr_scaling!r}; "
                             f"choose from {sorted(LR_SCALING_RULES)}")
        self.lr_scaling = lr_scaling
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        defaults.update(extra)
        # one param-group per unique shape (equal-shape lists for the collectives)
        sizes = {p.shape for p in params}
        param_groups = []
        for size in sorted(sizes):  # sorted => identical group order on every rank
            param_groups.append(dict(params=[p for p in params if p.shape == size]))
        super().__init__(param_groups, defaults)
        # lambda is a pure function of (rule, family, shape); cache it so the
        # per-step cost is a dict lookup rather than a sqrt per parameter.
        self._lambda = {id(p): layer_multiplier(p, self.family, lr_scaling)
                        for g in self.param_groups for p in g["params"]}
        # --- diagnostics (see DIAG_SLOTS) ----------------------------------
        self._diag_params = [p for g in self.param_groups for p in g["params"]]
        self._diag_col = {id(p): i for i, p in enumerate(self._diag_params)}
        # Allocated EAGERLY, not on first use: `diagnostics_report` all_reduces this
        # buffer, and a rank that happened to own no parameter would otherwise reach
        # that collective with nothing allocated, bail out early, and deadlock every
        # other rank. A few hundred floats is not worth the risk.
        self._diag_buf = torch.zeros(len(DIAG_SLOTS), len(self._diag_params),
                                     dtype=torch.float32,
                                     device=self._diag_params[0].device)
        #: The training script sets this True for the one step whose statistics it
        #: wants logged (the step immediately before each validation) and False
        #: otherwise, so the ~2 extra reductions per parameter never land in the
        #: measured hot loop. Every arm therefore keeps an identical clock.
        self.diagnostics = False

    # ---- centralized math helpers (single parameter, no collectives) -------

    @staticmethod
    def _is_attn(p: Tensor) -> bool:
        """Record #40 tags the merged Q/K/V/O attention weight with
        ``p.module == "attn"``; such a ``[hdim, 4*dim]`` parameter must be
        orthogonalized as four independent ``[hdim, dim]`` blocks."""
        return getattr(p, "module", None) == "attn"

    def _lmo(self, m: Tensor, p: Tensor) -> Tensor:
        """Apply the Muon LMO (Polar Express) to ``m``.  For the merged
        attention weight, reshape ``[hdim, 4*dim] -> [4, hdim, dim]`` so the
        four heads' Q/K/V/O sub-matrices are orthogonalized independently (the
        model uses exactly this ``.view(4, hdim, dim)``), then reshape back.
        Every other operation in the update rules is elementwise, so the reshape
        is localized to just this call."""
        if self._is_attn(p):
            h, w = m.shape[-2], m.shape[-1]
            d = polar_express(m.reshape(4, h, w // 4))
            return d.reshape(m.shape)
        return polar_express(m)

    def _scaled_sign(self, r: Tensor, p: Tensor, slot: str | None = None) -> Tensor:
        """The EF21 compressor ``mean|r| * sign(r)``, with ONE scale per LAYER.

        ``mean|r|`` is the only non-elementwise operation in any of the EF21
        update rules, so it is the only one -- besides the LMO -- that has to know
        about record #40's merged ``qkvo_w``.  That tensor holds FOUR layers
        (Q/K/V/O) and both the model (``.view(4, hdim, dim)``) and :meth:`_lmo`
        read it that way, so the compressor gets four scales, not one.  See the
        module docstring, item 2, for why sharing one is damaging and why it hits
        only the EF21 methods.

        ``slot`` optionally names the DIAG_SLOTS entry under which this call's
        achieved contraction is recorded.
        """
        if self._is_attn(p):
            h, w = r.shape[-2], r.shape[-1]
            # exactly _lmo's reshape, which is exactly the model's own
            # `.view(4, hdim, dim)`: the blocks are row bands of the flat tensor,
            # NOT column slices. Getting this axis wrong yields four "blocks" that
            # each mix all of Q/K/V/O, i.e. silently no per-layer scaling at all.
            rb = r.reshape(4, h, w // 4)
            alpha = rb.abs().mean(dim=(-2, -1), keepdim=True)
            if slot is not None and self.diagnostics:
                self._record_alpha(p, rb, slot)
            return (alpha * sign_pm1(rb)).reshape(r.shape)
        if slot is not None and self.diagnostics:
            self._record_alpha(p, r, slot)
        return r.abs().mean() * sign_pm1(r)

    # ---- diagnostics (owning-rank scalars; see DIAG_SLOTS) -----------------

    def _record(self, p: Tensor, slot: str, value: Tensor) -> None:
        """Store one scalar for one owned parameter: device-to-device, no host
        sync, so this is safe to call from inside the training step."""
        self._diag_buf[_DIAG_SLOT_IX[slot], self._diag_col[id(p)]] = value

    def _record_alpha(self, p: Tensor, r: Tensor, slot: str) -> None:
        """``alpha_eff = ||r||_1^2 / (d ||r||_2^2)``: the contraction the scaled
        sign ACTUALLY achieves on this round's residual -- ``2/pi`` for an
        isotropic ``r``, decaying to ``1/d`` as ``r`` becomes coherent.  This is
        the ``alpha`` of the convergence threshold, measured rather than assumed.

        ``r`` arrives already blocked for the merged attention weight, in which
        case we report the mean of the four blocks' values.  Accumulated in fp32
        because ``r`` may be bf16 (record #40 casts every ``nn.Linear``, which
        includes the gate weights).
        """
        rf = r.float()
        d = rf.shape[-1] * rf.shape[-2]
        l1 = rf.abs().sum(dim=(-2, -1))
        l2sq = rf.pow(2).sum(dim=(-2, -1))
        self._record(p, slot, (l1.pow(2) / (d * l2sq.clamp_min(1e-30))).mean())

    def _record_lag(self, p: Tensor, slot: str, err: Tensor, ref: Tensor) -> None:
        """Relative tracking error ``||err|| / ||ref||`` (fp32-accumulated)."""
        self._record(p, slot,
                     err.float().norm() / ref.float().norm().clamp_min(1e-30))

    def _record_grad_blocks(self, p: Tensor) -> None:
        """``mean|grad|`` per Q/K/V/O block of the merged attention weight (and the
        whole-tensor value in ``gblk0`` for every other parameter).  The spread
        across the four is what decides whether one shared compressor scale would
        have been adequate -- it is the measurement, on the real model, behind the
        per-layer scaling in :meth:`_scaled_sign`."""
        g = p.grad.float()
        if self._is_attn(p):
            h, w = g.shape[-2], g.shape[-1]
            vals = g.reshape(4, h, w // 4).abs().mean(dim=(-2, -1))
            for i in range(4):
                self._record(p, f"gblk{i}", vals[i])
        else:
            self._record(p, "gblk0", g.abs().mean())

    @torch.no_grad()
    def diagnostics_report(self, names: dict | None = None) -> str:
        """Table of the diagnostics recorded on the last flagged step.

        Values live on their owning rank only, so one SUM all_reduce (every other
        rank contributes zeros) makes the table identical on every rank and the
        caller can print it from rank 0.  Rows collapse the per-block index, as
        :func:`describe_lr_scaling` does, and report the median over the identical
        layers; a slot no method touched is left blank rather than printed as 0.

        Every rank must reach this method, and every early return must happen
        *after* the collective -- see the note on eager buffer allocation in
        :meth:`__init__`.
        """
        import re
        import statistics

        buf = self._diag_buf
        if dist.is_initialized() and dist.get_world_size() > 1:
            # UNCONDITIONAL: never gate this collective on rank-local state.
            buf = buf.clone()
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        vals = buf.tolist()   # host sync: only ever called with the clock stopped
        # Decided AFTER the reduce so that every rank returns the same thing.
        if not any(any(row) for row in vals):
            return ""
        names = names or {}

        groups: dict[str, list[int]] = {}
        for p in self._diag_params:
            name = re.sub(r"\.\d+\.", ".*.",
                          names.get(id(p), getattr(p, "module", "?")))
            groups.setdefault(name, []).append(self._diag_col[id(p)])

        shown = ("alpha_up", "alpha_dn", "lag_est", "lag_XW")
        used = [s for s in shown if any(vals[_DIAG_SLOT_IX[s]][c] != 0.0
                                       for cols in groups.values() for c in cols)]
        lines = [f"diagnostics ({type(self).__name__}; medians over identical "
                 f"layers, from the step before this validation)"]
        if used:
            lines.append("  " + f"{'parameter':<32}{'count':>7}"
                         + "".join(f"{s:>11}" for s in used))
            for name, cols in groups.items():
                row = f"  {name:<32}{len(cols):>7}"
                for s in used:
                    med = statistics.median(vals[_DIAG_SLOT_IX[s]][c] for c in cols)
                    row += f"{med:>11.4g}" if med != 0.0 else f"{'-':>11}"
                lines.append(row)
        # Per-block gradient magnitudes: only the merged attention weight has
        # blocks, and their spread is the number that justifies the per-layer
        # compressor scale in _scaled_sign.
        for name, cols in groups.items():
            if not self._is_attn(self._diag_params[cols[0]]):
                continue
            blk = [statistics.median(vals[_DIAG_SLOT_IX[f"gblk{i}"]][c] for c in cols)
                   for i in range(4)]
            lo, hi = min(blk), max(blk)
            ratio = f"{hi / lo:.1f}x" if lo > 0 else "inf"
            lines.append(f"  {name} mean|grad| per Q,K,V,O block: "
                         + " ".join(f"{v:.3e}" for v in blk)
                         + f"   max/min={ratio}")
        return "\n".join(lines)

    def lambda_of(self, p: Tensor) -> float:
        """Per-layer multiplier for ``p`` (see the LR-scaling section above)."""
        lam = self._lambda.get(id(p))
        if lam is None:  # parameter added after construction (not used by the scripts)
            lam = layer_multiplier(p, self.family, self.lr_scaling)
            self._lambda[id(p)] = lam
        return lam

    def _effective_grad(self, p: Tensor, group: dict, state: dict) -> Tensor:
        """Nesterov heavy-ball momentum, identical to the upstream Muon:

            buf  <- momentum * buf + (1 - momentum) * grad      (EMA buffer)
            m    <- (1 - momentum) * grad + momentum * buf      (look-ahead)

        Returns ``m`` (aliases ``p.grad`` after an in-place lerp; callers must not
        mutate it in place).

        Takes ``p`` rather than ``p.grad`` so that this single choke point -- every
        method calls it exactly once, with the global mean gradient already in
        place -- can also publish the per-block gradient magnitudes.
        """
        if self.diagnostics:
            self._record_grad_blocks(p)
        grad = p.grad
        momentum = group["momentum"]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(grad)
        buf = state["momentum_buffer"]
        buf.lerp_(grad, 1 - momentum)
        return grad.lerp_(buf, momentum)

    def _eff_lr(self, p: Tensor, group: dict) -> float:
        # eta_layer = eta_0 * lambda(family, shape) * p.lr_mul.  For the LMO
        # family lambda is Muon's shipped aspect factor, so `Muon` here steps
        # exactly as record #40 does; the sign family gets its 1/sqrt(fan_in)
        # counterpart, which makes eta_0 mean the same thing for all eight
        # methods (the per-step RMS gain).
        return group["lr"] * self.lambda_of(p) * getattr(p, "lr_mul", 1.0)

    def _decoupled_weight_decay(self, target: Tensor, p: Tensor, group: dict) -> None:
        """AdamW-style decoupled decay applied to ``target`` (usually ``p``, but
        the *exact* model ``X`` for EF21-MuonSign)."""
        eff_wd = group["lr"] * group["weight_decay"] * getattr(p, "wd_mul", 1.0)
        if eff_wd != 0:
            target.mul_(1 - eff_wd)

    def update_param(self, p: Tensor, group: dict) -> None:  # pragma: no cover
        """In-place centralized update of one owned parameter ``p`` (whose
        ``.grad`` already holds the global mean gradient). Override in
        subclasses."""
        raise NotImplementedError

    # ---- distributed transport (identical to upstream Muon) ----------------

    @staticmethod
    def _pad_params(params: list, world_size: int) -> list:
        """``params`` padded with scratch tensors to a whole number of chunks.

        The padded slots are all_gather *outputs* that nobody reads, so scratch is
        fine -- but they must be distinct-shaped, same-dtype tensors and there must
        be exactly ``(-n) % world_size`` of them, never more (an over-long pad list
        would allocate a full parameter's worth of memory every step for nothing)."""
        n_pad = (-len(params)) % world_size
        return list(params) + [torch.empty_like(params[-1]) for _ in range(n_pad)]

    @torch.no_grad()
    def step(self):
        # Efficient sharded implementation from the modded-nanogpt record
        # (@YouJiacheng, @KonstantinWilleke, @alexrgilbert, @adricarda,
        # @tuttyfrutyee, @vdlad, @ryanyang0, @vagrawal). Each rank owns
        # params[base_i + rank] within every world_size chunk.  On a single
        # process (world_size == 1) the collectives are no-ops.
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures: list[torch.Future] = []
        all_gather_futures: list[torch.Future] = []

        # Phase 1: average each parameter's gradient onto its owning rank.
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            assert all(p.grad is not None for p in params), (
                f"{type(self).__name__}.step(): some parameters have no .grad; "
                "every parameter in a group must take part in the collective")
            # Pad the group out to a whole number of world_size-sized chunks. Only
            # the final chunk can be short, so `(-n) % world_size` padding entries
            # are enough -- and when the group divides evenly (record #40's 32
            # hidden matrices on 8 ranks) that is zero, saving a pointless
            # zeros_like of the full parameter every step. The padding must be
            # ZEROS: it is reduce_scatter input, and ranks that own nothing in the
            # last chunk must average zeros rather than uninitialized memory.
            n_pad = (-len(params)) % world_size
            grad_pad = [p.grad for p in params] + [torch.zeros_like(params[-1])] * n_pad
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    grad = params[base_i + rank].grad
                else:
                    # This rank owns nothing in the (padded) final chunk, but it
                    # must still take part in the collective. It needs a FRESH
                    # scratch buffer: reusing the previous chunk's `grad` would
                    # alias a parameter this rank *does* own and overwrite that
                    # parameter's freshly averaged gradient with padding zeros
                    # before phase 2 ever reads it.
                    grad = torch.empty_like(params[-1])
                reduce_scatter_futures.append(
                    dist.reduce_scatter(
                        grad, grad_pad[base_i:base_i + world_size],
                        op=dist.ReduceOp.AVG, async_op=True,
                    ).get_future()
                )

        # Phase 2: owning rank runs the centralized update, then broadcast back.
        idx = 0
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            params_pad = self._pad_params(params, world_size)
            for base_i in range(0, len(params), world_size):
                reduce_scatter_futures[idx].wait()
                if base_i + rank < len(params):
                    self.update_param(params[base_i + rank], group)
                idx += 1
                all_gather_futures.append(
                    dist.all_gather(
                        params_pad[base_i:base_i + world_size],
                        params_pad[base_i + rank], async_op=True,
                    ).get_future()
                )
        torch.futures.collect_all(all_gather_futures).wait()


# ---------------------------------------------------------------------------
# Reference methods
# ---------------------------------------------------------------------------


class Muon(_DistributedMatrixOptimizer):
    """Reference full-precision Muon: ``X <- X - eta * PE(M)``."""

    family = FAMILY_LMO

    def update_param(self, p, group):
        state = self.state[p]
        self._decoupled_weight_decay(p, p, group)
        m = self._effective_grad(p, group, state)
        v = self._lmo(m, p)
        p.add_(v, alpha=-self._eff_lr(p, group))


class SignSGD(_DistributedMatrixOptimizer):
    """SignSGD with momentum: ``X <- X - eta * sign(M)`` (no LMO)."""

    family = FAMILY_SIGN

    def update_param(self, p, group):
        state = self.state[p]
        self._decoupled_weight_decay(p, p, group)
        m = self._effective_grad(p, group, state)
        p.add_(sign_pm1(m), alpha=-self._eff_lr(p, group))


# ---------------------------------------------------------------------------
# Sign-around-the-LMO methods (no error feedback)
# ---------------------------------------------------------------------------


class SignMuon(_DistributedMatrixOptimizer):
    """SignMuon -- sign AFTER the LMO: ``D = PE(M);  X <- X - eta * sign(D)``.

    The Theorem-1 divergence counterexample lives here: sign-compressing the LMO
    direction can destroy the descent property.
    """

    family = FAMILY_SIGN

    def update_param(self, p, group):
        state = self.state[p]
        self._decoupled_weight_decay(p, p, group)
        m = self._effective_grad(p, group, state)
        d = self._lmo(m, p)
        p.add_(sign_pm1(d), alpha=-self._eff_lr(p, group))


class MuonUSign(_DistributedMatrixOptimizer):
    """MuonUSign -- sign BEFORE the LMO: ``s = sign(M);  X <- X - eta * PE(s)``.

    The LMO is positively homogeneous of degree zero, so compressing with the
    scaled sign ``mean|M| * sign(M)`` instead of the bare ``sign(M)`` gives the
    identical direction: the "U" (unscaled) and scaled variants of the *uplink*
    compressor coincide here.  This is NOT ``MuonSign``, which signs the LMO
    output as well (see below).
    """

    family = FAMILY_LMO

    def update_param(self, p, group):
        state = self.state[p]
        self._decoupled_weight_decay(p, p, group)
        m = self._effective_grad(p, group, state)
        d = self._lmo(sign_pm1(m), p)
        p.add_(d, alpha=-self._eff_lr(p, group))


class MuonSign(_DistributedMatrixOptimizer):
    """MuonSign -- sign BEFORE *and* AFTER the LMO:
    ``s = sign(M);  D = PE(s);  X <- X - eta * sign(D)``.
    """

    family = FAMILY_SIGN

    def update_param(self, p, group):
        state = self.state[p]
        self._decoupled_weight_decay(p, p, group)
        m = self._effective_grad(p, group, state)
        d = self._lmo(sign_pm1(m), p)
        p.add_(sign_pm1(d), alpha=-self._eff_lr(p, group))


# ---------------------------------------------------------------------------
# Error-feedback (EF21) methods
# ---------------------------------------------------------------------------


class EF21SignMuon(_DistributedMatrixOptimizer):
    """EF21-SignMuon -- EF21 error feedback on the LMO DIRECTION.

        D      = PE(M)
        delta  = D - d_est
        d_est <- d_est + mean|delta| * sign(delta)     (scaled-sign compressor)
        X     <- X - eta * d_est

    Tracks the (discontinuous) LMO direction with a contractive 1-bit estimator.

    Note that this method's target ``D`` is *blockwise orthogonal*, so its four
    merged Q/K/V/O blocks carry identical entry magnitudes and the per-layer
    compressor scale changes almost nothing here -- unlike the two methods below,
    which track the raw momentum. It uses the same operator regardless: the
    compressor is defined per layer, and one spelling is better than two.
    """

    family = FAMILY_LMO

    def update_param(self, p, group):
        state = self.state[p]
        self._decoupled_weight_decay(p, p, group)
        m = self._effective_grad(p, group, state)
        d = self._lmo(m, p)
        if "d_est" not in state:
            state["d_est"] = torch.zeros_like(p)
        d_est = state["d_est"]
        delta = d - d_est
        d_est.add_(self._scaled_sign(delta, p, slot="alpha_up"))
        if self.diagnostics:
            self._record_lag(p, "lag_est", d - d_est, d)
        p.add_(d_est, alpha=-self._eff_lr(p, group))


class EF21MuonUSign(_DistributedMatrixOptimizer):
    """EF21-MuonUSign -- EF21 error feedback on the MOMENTUM, full LMO after.

        delta  = M - g_est
        g_est <- g_est + mean|delta| * sign(delta)     (uplink, scaled-sign)
        D      = PE(g_est)
        X     <- X - eta * D

    Applying the LMO to the (asymptotically exact) gradient estimator rather than
    to a sign is what restores convergence on the Theorem-1 counterexample.

    ``mean|.|`` is per LAYER, so on the merged Q/K/V/O weight it is four scales,
    not one -- see :meth:`_DistributedMatrixOptimizer._scaled_sign`. This method
    tracks the RAW momentum, whose blocks differ by orders of magnitude, so unlike
    EF21-SignMuon it is materially affected by that choice.
    """

    family = FAMILY_LMO

    def update_param(self, p, group):
        state = self.state[p]
        self._decoupled_weight_decay(p, p, group)
        m = self._effective_grad(p, group, state)
        if "g_est" not in state:
            state["g_est"] = torch.zeros_like(p)
        g_est = state["g_est"]
        delta = m - g_est
        g_est.add_(self._scaled_sign(delta, p, slot="alpha_up"))
        if self.diagnostics:
            self._record_lag(p, "lag_est", m - g_est, m)
        d = self._lmo(g_est, p)
        p.add_(d, alpha=-self._eff_lr(p, group))


class EF21MuonSign(_DistributedMatrixOptimizer):
    """EF21-MuonSign -- bidirectional EF21 (uplink gradient + downlink model).

    Uplink (as EF21-MuonUSign) reconstructs the gradient estimator ``g_est`` and
    the server advances an EXACT model ``X`` (kept as optimizer state) with the
    Muon LMO step. A second EF21-P loop compresses the model increment onto the
    live broadcast model ``W`` (= the parameter tensor), so the next gradient is
    evaluated at ``W``:

        delta_up  = M - g_est
        g_est    <- g_est + mean|delta_up| * sign(delta_up)      (uplink)
        X        <- X - eta * PE(g_est)                          (exact step)
        delta_dn  = X - W
        W        <- W + mean|delta_dn| * sign(delta_dn)          (downlink EF21-P)

    Only ``W`` (one bit + one scalar per entry per round, conceptually) is ever
    broadcast; ``X`` never leaves its owning rank except via
    :meth:`swap_in_exact` for evaluation. Weight decay acts on the exact model.

    Both ``mean|.|`` above are per LAYER (four scales on the merged Q/K/V/O
    weight); see :meth:`_DistributedMatrixOptimizer._scaled_sign`.

    ``X`` and ``W`` are NOT interchangeable in practice. The convergence corollary
    bounds ``||grad f(X_t)||``, but the EF21-P downlink's contraction constant is
    only ``alpha ~ 1/d`` in the worst case, and on a coherent residual -- which is
    what momentum plus a slowly-turning gradient produces -- it really does
    collapse to that end of the range. ``X`` then runs many steps ahead of the
    iterate every gradient was actually taken at. The ``lag_XW`` diagnostic
    measures the gap, and the training script reports the loss at both models.
    """

    family = FAMILY_LMO

    def update_param(self, p, group):
        state = self.state[p]
        if "exact_model" not in state:
            state["exact_model"] = p.detach().clone()   # X_0 = W_0 (broadcast model)
        X = state["exact_model"]
        self._decoupled_weight_decay(X, p, group)       # decay the exact server model

        m = self._effective_grad(p, group, state)  # gradient was taken at W = p
        if "g_est" not in state:
            state["g_est"] = torch.zeros_like(p)
        g_est = state["g_est"]
        delta_up = m - g_est
        g_est.add_(self._scaled_sign(delta_up, p, slot="alpha_up"))  # uplink EF21
        if self.diagnostics:
            self._record_lag(p, "lag_est", m - g_est, m)

        d = self._lmo(g_est, p)
        X.add_(d, alpha=-self._eff_lr(p, group))        # exact server step: X <- X - eta * PE(g_est)

        delta_dn = X - p                                # downlink EF21-P on the model increment
        p.add_(self._scaled_sign(delta_dn, p, slot="alpha_dn"))
        if self.diagnostics:                            # server/broadcast gap after the round
            self._record_lag(p, "lag_XW", X - p, p)

    # -- expose the exact model X for evaluation -----------------------------

    @torch.no_grad()
    def swap_in_exact(self):
        """Broadcast the exact server model ``X`` into the live parameters (for
        eval / checkpointing), stashing the compressed model ``W``. Uses the same
        sharded all-gather as :meth:`step`."""
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        self._w_backup: dict[int, tuple[Tensor, Tensor]] = {}
        futures: list[torch.Future] = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            params_pad = self._pad_params(params, world_size)
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    st = self.state[p]
                    self._w_backup[id(p)] = (p, p.detach().clone())
                    if "exact_model" in st:
                        p.copy_(st["exact_model"])
                futures.append(
                    dist.all_gather(
                        params_pad[base_i:base_i + world_size],
                        params_pad[base_i + rank], async_op=True,
                    ).get_future()
                )
        torch.futures.collect_all(futures).wait()

    @torch.no_grad()
    def swap_out_exact(self):
        """Restore the compressed broadcast model ``W`` saved by
        :meth:`swap_in_exact`."""
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        backup = getattr(self, "_w_backup", {})
        futures: list[torch.Future] = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            params_pad = self._pad_params(params, world_size)
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    if id(p) in backup:
                        p.copy_(backup[id(p)][1])
                futures.append(
                    dist.all_gather(
                        params_pad[base_i:base_i + world_size],
                        params_pad[base_i + rank], async_op=True,
                    ).get_future()
                )
        torch.futures.collect_all(futures).wait()
        self._w_backup = {}


# ---------------------------------------------------------------------------
# Registry: paper name -> class. Used by the train scripts' --optimizer knob.
# ---------------------------------------------------------------------------

OPTIMIZERS = {
    "SignMuon":        SignMuon,
    "EF21-SignMuon":   EF21SignMuon,
    "MuonUSign":       MuonUSign,
    "MuonSign":        MuonSign,
    "EF21-MuonUSign":  EF21MuonUSign,
    "EF21-MuonSign":   EF21MuonSign,
    "SignSGD":         SignSGD,
    "Muon":            Muon,
}

#: the six methods introduced in the paper (the two above are references)
PAPER_METHODS = [
    "SignMuon", "EF21-SignMuon", "MuonUSign",
    "MuonSign", "EF21-MuonUSign", "EF21-MuonSign",
]
