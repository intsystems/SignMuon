"""Run a whole hyperparameter grid as one batched trajectory.

Why this module exists
----------------------
The synthetic problem is a quadratic on ``m x n`` matrices, and the sweeps in
``synthetic.benchmark`` spend all of their time in a loop that is *latency*
bound rather than compute bound: at ``100x100`` one optimizer step is ~50 tiny
CUDA kernels (Newton-Schulz alone is five iterations of four matmuls) plus two
blocking device-to-host syncs, while the arithmetic itself is a few
microseconds. Measured on an A4000 the driver gets ~1450 steps/second and that
figure barely moves between stages -- or, for that matter, between ``--m 64``
and ``--m 100``. Runtime is therefore ``configurations x iterations``, almost
independent of the matrix size, which is why the honest way to make the sweeps
cheaper is to stop running the configurations one after another.

Every configuration in a grid is an independent trajectory over the *same*
problem, so the grid is a batch. This module runs it as one: the iterate is a
``[B, m, n]`` stack, the learning rate and momentum are ``[B, 1, 1]`` tensors,
and one "step" advances all ``B`` runs at once. Nothing couples the slices --
``A @ X @ B`` broadcasts, Newton-Schulz normalizes per slice, ``sign`` is
elementwise -- provided the two per-tensor *reductions* in the EF21 compressor
are taken per slice as well, which is the one place a naive batching would
silently be wrong (see :func:`ef21_step`).

Three further things fall out of the rewrite:

* **No per-step syncs.** ``best_f``, ``best_gnorm``, the convergence step and
  the divergence flag are accumulated in device tensors and read back once, at
  the end. The sequential driver pulled two scalars to the host on every
  iteration, which drains the queue and serializes everything behind it.
* **Configurations leave the batch when they finish.** A run that has reached
  the target (under ``stop_at_target``), diverged, or exhausted its own budget
  is masked out immediately and dropped from the batch at the next compaction.
* **The tie-break RNG is pinned per run** (:func:`deterministic_rng`), so a
  configuration's trajectory does not depend on what was run before it.

``benchmark.run_one`` is kept as the reference implementation and is what
``--runner sequential`` uses; ``tests.test_code`` checks the two agree on every
method, over the horizon in which agreement is meaningful -- see below.

How closely it reproduces the reference loop
--------------------------------------------
A batched matmul and a single one reduce in a different order, so the two
runners see gradients differing at the last float32 bit. Measured on the
``9x7`` reference problem, that perturbation stays at ``1e-7`` for the first
~20 steps and then amplifies, because ``sign`` is discontinuous: an entry of
``M`` or of ``polar(M)`` within rounding of zero flips, and the step changes by
``O(1)``. This is the same instability the paper is about, not an artefact of
batching -- any change of GPU, BLAS or torch version perturbs a trajectory the
same way.

What that costs the reported numbers, measured at ``100x100`` over 800 steps:
``signmuon``, ``muonsign``, ``muon``, ``signsgd`` and ``sgd`` agree exactly; the
EF21 methods to ``1e-4``; ``muonusign`` and ``adam`` to ``3e-3``. The alignment
statistics agree to ``1e-5``. Everything the tables report is a statistic over a
trajectory -- a plateau level, a fitted slope, a distribution of ``rho`` -- and
those are stable; a single iterate at step 800 is not, and is not reported.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor

from common.optimizers import sign_pm1, zeropower_via_newtonschulz5

__all__ = ["run_configs", "polar_batched", "ef21_step", "BATCHED_METHODS"]

#: ``F`` above this counts as divergence, matching ``benchmark.run_one``.
DIVERGENCE_F = 1e12

#: ``torch.optim.Adam`` defaults, which ``benchmark._build`` never overrides.
ADAM_BETA1, ADAM_BETA2, ADAM_EPS = 0.9, 0.999, 1e-8

#: Methods with the shared momentum/LMO skeleton of ``_BaseMethod``.
_MATRIX_METHODS = ("signmuon", "ef21signmuon", "muonusign", "muonsign",
                   "ef21muonusign", "ef21muonsign", "muon", "signsgd")

BATCHED_METHODS = _MATRIX_METHODS + ("sgd", "adam")

#: The one method whose gradient is taken at a different point from its metrics:
#: ``p.data`` holds the broadcast model ``W``, the state holds the exact ``X``.
_BIDIRECTIONAL = "ef21muonsign"


# --------------------------------------------------------------------------
# Batched building blocks
# --------------------------------------------------------------------------


def polar_batched(Y: Tensor, dtype: Optional[torch.dtype],
                  ns_steps: int = 5, scale_aspect: bool = True) -> Tensor:
    """``polar(Y) = U V^T`` for a stack of matrices ``[B, m, n]``.

    Deliberately **not** ``common.optimizers.muon_lmo``. That function reads any
    tensor with ``ndim > 2`` as a stack of convolution filters and flattens it to
    ``[B, m*n]``, so handing it a batch of matrices would orthogonalize one
    ``B x mn`` matrix instead of ``B`` separate ones -- a silent wrong answer, not
    an error. The Newton-Schulz kernel underneath is genuinely batch-safe (it
    reduces over the trailing two dimensions throughout), so this calls it
    directly and applies Muon's aspect factor itself, matching ``muon_lmo``.
    """
    orth = zeropower_via_newtonschulz5(Y, steps=ns_steps, dtype=dtype)
    if scale_aspect:
        orth = orth * max(1.0, orth.size(-2) / orth.size(-1)) ** 0.5
    return orth


def _mean_abs(Y: Tensor) -> Tensor:
    """``alpha = mean|Y|`` per matrix in the stack, shaped ``[B, 1, 1]``."""
    return Y.abs().mean(dim=(-2, -1), keepdim=True)


def dual_norm_of(G: Tensor, kind: str) -> Tensor:
    """``||G||_*`` per slice, for the norm dual to the method's LMO ball.

    The theorems bound the *dual* norm of the gradient, and which one that is
    depends on the ball the LMO minimizes over: ``ell_infty`` for a sign step,
    so ``ell_1``; the spectral ball for an LMO step, so nuclear. SGD and Adam
    have no LMO, and Frobenius is its own dual.
    """
    if kind == "l1":
        return G.abs().sum(dim=(-2, -1))
    if kind == "nuclear":
        return torch.linalg.svdvals(G.float()).sum(dim=-1)
    if kind == "fro":
        return G.flatten(1).norm(dim=1)
    raise ValueError(f"unknown dual norm {kind!r}")


def ef21_step(target: Tensor, est: Tensor) -> Tensor:
    """In-place ``est += mean|target-est| * sign(target-est)``, **per slice**.

    The per-slice reduction is the whole difference from
    ``_EF21Mixin._ef21_update``, and it is the one thing a naive batching gets
    wrong: ``.mean()`` over a ``[B, m, n]`` tensor returns a single scalar, which
    would hand every configuration in the batch the same magnitude ``alpha_t``
    and couple runs that must stay independent. It would also not fail loudly --
    the shapes broadcast fine and the numbers stay plausible.
    """
    delta = target - est
    est.add_(_mean_abs(delta) * sign_pm1(delta))
    return est


def _direction(method: str, m_tilde: Tensor, st: Dict[str, Tensor],
               dtype: Optional[torch.dtype]) -> Tensor:
    """``d_t`` for the eight matrix methods -- the table in ``common.optimizers``.

    Transcribed from the ``_direction`` methods there rather than shared with
    them, because those take a single matrix and the EF21 pair need the per-slice
    reduction above. ``tests.test_code`` pins the two to each other.
    """
    if method == "muon":                                   # polar(M)
        return polar_batched(m_tilde, dtype)
    if method == "signsgd":                                # sign(M)
        return sign_pm1(m_tilde)
    if method == "signmuon":                               # sign AFTER the LMO
        return sign_pm1(polar_batched(m_tilde, dtype))
    if method == "muonusign":                              # sign BEFORE the LMO
        return polar_batched(sign_pm1(m_tilde), dtype)
    if method == "muonsign":                               # sign on BOTH sides
        return sign_pm1(polar_batched(sign_pm1(m_tilde), dtype))
    if method == "ef21signmuon":                           # EF21 on the LMO output
        return ef21_step(polar_batched(m_tilde, dtype), st["dir_estimator"])
    if method in ("ef21muonusign", "ef21muonsign"):        # EF21 before the LMO
        return polar_batched(ef21_step(m_tilde, st["grad_estimator"]), dtype)
    raise ValueError(f"no batched direction for {method!r}")


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def _column(values: Sequence[float], device, dtype=torch.float32) -> Tensor:
    """A per-configuration scalar, shaped ``[B, 1, 1]`` to broadcast over ``[B, m, n]``."""
    return torch.tensor(list(values), device=device, dtype=dtype).view(-1, 1, 1)


@contextmanager
def deterministic_rng(seed: int, device):
    """Pin the tie-break RNG for the duration of a run, then restore it.

    ``sign_pm1`` maps an exact zero to a random ``+-1`` drawn from the global
    torch RNG, and exact zeros do occur here -- bfloat16 ``polar(M)`` produces
    them. Without this, a run's trajectory would depend on how many draws
    happened to be made before it, so the same configuration would give
    different numbers depending on whether it was reached through a grid search
    or run on its own. Restores the caller's stream on exit, so this stays a
    local guarantee rather than a global side effect.
    """
    devices = [device] if getattr(device, "type", "") == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        yield


def run_configs(method: str, problem, configs: Sequence[Dict], **kwargs
                ) -> List[Dict]:
    """Run every configuration in ``configs`` as one batch; one record each.

    Thin wrapper: everything happens under a pinned tie-break RNG, so a run
    depends only on its own seeds and not on what was run before it.
    """
    with deterministic_rng(kwargs.get("init_seed", 42), problem.A.device):
        return _run_configs(method, problem, configs, **kwargs)


def _run_configs(
    method: str,
    problem,
    configs: Sequence[Dict],
    *,
    target_loss: float = 1e-3,
    max_iters: int = 5000,
    init_seed: int = 42,
    lmo_dtype="bfloat16",
    capture_alignment: bool = False,
    keep_history: bool = False,
    stop_at_target: bool = False,
    dual_norm: Optional[str] = None,
    compact_every: int = 200,
) -> List[Dict]:
    """Advance the batch. See :func:`run_configs`.

    ``configs`` is a list of ``{"lr", "momentum"?, "schedule"?, "max_iters"?}``.
    Each entry may carry its own iteration budget -- ``--mode floor`` needs that,
    since time-to-plateau scales like ``1/eta`` and it gives the small steps a
    proportionally longer run. Records come back in the order given, with the
    same keys :func:`benchmark.run_one` returns, so the two runners are
    interchangeable to everything downstream.

    ``dual_norm`` additionally tracks ``min_t ||grad F(X_t)||_*`` in that norm --
    the quantity the convergence theorems bound. ``"nuclear"`` costs an SVD per
    step, so it is meant for a re-run at a tuned optimum rather than for a sweep.

    ``compact_every`` steps, finished configurations are dropped from the batch.
    Detecting that costs one device-to-host sync, which is why it is not done
    every step.
    """
    if method not in BATCHED_METHODS:
        raise ValueError(f"no batched implementation of {method!r}")
    device = problem.A.device
    dtype = getattr(torch, lmo_dtype) if isinstance(lmo_dtype, str) else lmo_dtype
    bidirectional = method == _BIDIRECTIONAL
    is_adam = method == "adam"
    n_cfg = len(configs)
    if n_cfg == 0:
        return []

    schedules = [c.get("schedule", "const") for c in configs]
    budgets = [int(c.get("max_iters", max_iters)) for c in configs]
    unknown = set(schedules) - {"const", "sqrt", "linear"}
    if unknown:
        raise ValueError(f"unknown schedule(s) {sorted(unknown)}")
    # The optimizer classes reject these in their constructors, so the batched
    # path has to as well or it would quietly run a grid the reference runner
    # refuses.
    for c in configs:
        if float(c["lr"]) < 0.0:
            raise ValueError(f"Invalid learning rate: {c['lr']}")
        if not 0.0 <= float(c.get("momentum", 0.0)) < 1.0:
            raise ValueError(f"momentum must lie in [0, 1), got {c['momentum']}")
    T_max = max(budgets)

    # -- batch state. Every entry has the configuration axis first, so a
    #    compaction is one indexing pass over the whole dict.
    X0 = problem.initial_point(init_seed)
    st: Dict[str, Tensor] = {
        "W": X0.unsqueeze(0).repeat(n_cfg, 1, 1),
        "lr": _column([float(c["lr"]) for c in configs], device),
        "mom": _column([float(c.get("momentum", 0.0)) for c in configs], device),
        "budget": _column(budgets, device),
        "is_const": _column([s == "const" for s in schedules], device),
        "is_sqrt": _column([s == "sqrt" for s in schedules], device),
        "is_linear": _column([s == "linear" for s in schedules], device),
        # Where each surviving slice belongs in the full-width result buffers.
        "idx": torch.arange(n_cfg, device=device),
        "alive": torch.ones(n_cfg, dtype=torch.bool, device=device),
    }
    if is_adam:
        st["exp_avg"] = torch.zeros_like(st["W"])
        st["exp_avg_sq"] = torch.zeros_like(st["W"])
    else:
        st["momentum_buffer"] = torch.zeros_like(st["W"])
    if method == "ef21signmuon":
        st["dir_estimator"] = torch.zeros_like(st["W"])
    if method in ("ef21muonusign", "ef21muonsign"):
        st["grad_estimator"] = torch.zeros_like(st["W"])
    if bidirectional:
        st["exact_model"] = st["W"].clone()

    # -- results, full width and on the device until the very end.
    inf = float("inf")
    best_f = torch.full((n_cfg,), inf, device=device)
    best_g = torch.full((n_cfg,), inf, device=device)
    best_dual = torch.full((n_cfg,), inf, device=device)
    last_f = torch.full((n_cfg,), float("nan"), device=device)
    iters = torch.tensor(budgets, device=device, dtype=torch.long)
    reached = torch.zeros(n_cfg, dtype=torch.bool, device=device)
    diverged = torch.zeros(n_cfg, dtype=torch.bool, device=device)
    recorded = torch.zeros(n_cfg, dtype=torch.long, device=device)

    # ``rho`` and the loss curves are the only quantities we cannot reduce on the
    # fly, so they get a full [T, B] buffer written through ``idx``; NaN marks a
    # step a configuration was no longer running for.
    hist_f = (torch.full((T_max, n_cfg), float("nan"), device=device)
              if keep_history else None)
    hist_g = (torch.full((T_max, n_cfg), float("nan"), device=device)
              if keep_history else None)
    # torch.optim's SGD and Adam expose no direction hook, so the sequential
    # driver reports no rho for them; match that rather than inventing one.
    want_rho = capture_alignment and method in _MATRIX_METHODS
    hist_rho = (torch.full((T_max, n_cfg), float("nan"), device=device)
                if want_rho else None)

    start = time.time()
    for t in range(T_max):
        idx, alive = st["idx"], st["alive"]

        # 1) gradient at p.data (the broadcast model W for the bidirectional
        #    method, as its algorithm requires) and metrics at the tracked
        #    iterate. They coincide for every other method, and the closed forms
        #    are identical operations, so compute the pair only once there.
        if bidirectional:
            g_step = problem.grad(st["W"])
            f_val, g_tracked = problem.value_and_grad_batched(st["exact_model"])
        else:
            f_val, g_tracked = problem.value_and_grad_batched(st["W"])
            g_step = g_tracked
        g_norm = g_tracked.flatten(1).norm(dim=1)

        # 2) record, for the slices still running. ``fmin``, not ``minimum``:
        #    the sequential driver runs ``min()`` on Python floats, and
        #    ``min(best, nan)`` keeps ``best`` -- a NaN loss must not overwrite
        #    the best one seen before the run blew up, which ``torch.minimum``
        #    (NaN-propagating) would do.
        best_f[idx] = torch.where(alive, torch.fmin(best_f[idx], f_val),
                                  best_f[idx])
        best_g[idx] = torch.where(alive, torch.fmin(best_g[idx], g_norm),
                                  best_g[idx])
        if dual_norm is not None:
            gd = dual_norm_of(g_tracked, dual_norm)
            best_dual[idx] = torch.where(alive, torch.fmin(best_dual[idx], gd),
                                         best_dual[idx])
        last_f[idx] = torch.where(alive, f_val, last_f[idx])
        recorded[idx] = recorded[idx] + alive.long()
        if keep_history:
            hist_f[t, idx] = torch.where(alive, f_val, float("nan"))
            hist_g[t, idx] = torch.where(alive, g_norm, float("nan"))

        hit = alive & (f_val <= target_loss) & ~reached[idx]
        iters[idx] = torch.where(hit, t + 1, iters[idx])
        reached[idx] = reached[idx] | hit

        bad = alive & (~torch.isfinite(f_val) | (f_val > DIVERGENCE_F))
        diverged[idx] = diverged[idx] | bad

        # Two distinct masks, because ``run_one`` has two distinct exits.
        # ``stepping``: it breaks *before* ``optimizer.step()`` on divergence and
        # on the first target hit under ``stop_at_target``, so those slices take
        # no step and capture no rho this iteration.
        # ``st["alive"]``: running out of budget ends the loop *after* a step, so
        # the last iteration of a budget still steps and still captures rho --
        # it is only the next iteration that never happens.
        stepping = alive & ~bad
        if stop_at_target:
            stepping = stepping & ~hit
        st["alive"] = stepping & (float(t + 1) < st["budget"].view(-1))

        # 3) the step. ``eta`` is zeroed on slices that are not stepping, so
        #    their iterate freezes instead of drifting (or spreading a NaN)
        #    until the next compaction drops them.
        mult = (st["is_const"]
                + st["is_sqrt"] / math.sqrt(1.0 + t)
                + st["is_linear"] * torch.clamp(
                    1.0 - t / torch.clamp(st["budget"], min=1.0), min=0.0))
        eta = st["lr"] * mult * stepping.view(-1, 1, 1).to(st["lr"].dtype)

        if is_adam:
            exp_avg, exp_avg_sq = st["exp_avg"], st["exp_avg_sq"]
            exp_avg.mul_(ADAM_BETA1).add_(g_step, alpha=1.0 - ADAM_BETA1)
            exp_avg_sq.mul_(ADAM_BETA2).addcmul_(g_step, g_step,
                                                 value=1.0 - ADAM_BETA2)
            bias1 = 1.0 - ADAM_BETA1 ** (t + 1)
            bias2 = 1.0 - ADAM_BETA2 ** (t + 1)
            denom = exp_avg_sq.sqrt().div_(math.sqrt(bias2)).add_(ADAM_EPS)
            st["W"].sub_((eta / bias1) * exp_avg / denom)
        elif method == "sgd":
            # torch.optim.SGD's heavy-ball form, not the EMA one: dampening is 0
            # and the buffer is initialized to the first gradient, which
            # ``mu*0 + g`` reproduces.
            buf = st["momentum_buffer"]
            buf.mul_(st["mom"]).add_(g_step)
            st["W"].addcmul_(buf, eta, value=-1.0)
        else:
            # EMA momentum, matching ``_BaseMethod.step`` (nesterov is off
            # everywhere in this benchmark).
            buf = st["momentum_buffer"]
            buf.mul_(st["mom"]).add_(g_step * (1.0 - st["mom"]))
            d_t = _direction(method, buf, st, dtype)
            if want_rho:
                dn = d_t.flatten(1).norm(dim=1)
                denom = g_norm * dn
                rho = torch.where(denom > 0,
                                  (g_tracked * d_t).flatten(1).sum(1)
                                  / torch.clamp(denom, min=1e-45),
                                  float("nan"))
                # ``stepping``, not ``alive``: the sequential driver captures the
                # direction of the step it just took, so a slice that broke out
                # before stepping contributes nothing at this t. NaN marks both
                # that and the ``denom == 0`` case, which it skips as well.
                hist_rho[t, idx] = torch.where(stepping, rho, float("nan"))
            target = st["exact_model"] if bidirectional else st["W"]
            target.addcmul_(d_t, eta, value=-1.0)
            if bidirectional:
                # Downlink EF21-P: broadcast a scaled sign of the model shift.
                shift = st["exact_model"] - st["W"]
                st["W"].add_(_mean_abs(shift) * sign_pm1(shift))

        # 4) drop the finished slices. Reading the mask is the only
        #    device-to-host sync in the loop, which is why it is periodic rather
        #    than per-step -- the point of the rewrite was to stop draining the
        #    queue every iteration.
        if (t + 1) % compact_every == 0:
            keep = st["alive"]
            n_live = int(keep.sum())
            if n_live == 0:
                break
            if n_live < keep.numel():
                st = {k: v[keep] for k, v in st.items()}

    elapsed = time.time() - start

    # -- one readback
    best_f_l = best_f.tolist()
    best_g_l = best_g.tolist()
    best_dual_l = best_dual.tolist() if dual_norm is not None else None
    last_f_l = last_f.tolist()
    iters_l = iters.tolist()
    reached_l = reached.tolist()
    diverged_l = diverged.tolist()
    recorded_l = recorded.tolist()
    hist_f_l = hist_f.T.tolist() if keep_history else None
    hist_g_l = hist_g.T.tolist() if keep_history else None
    rho_stats = _rho_statistics(hist_rho) if want_rho else [None] * n_cfg

    out: List[Dict] = []
    for i, cfg in enumerate(configs):
        kwargs = {"lr": float(cfg["lr"])}
        if not is_adam:
            kwargs["momentum"] = float(cfg.get("momentum", 0.0))
        rec = {
            "method": method,
            "kwargs": kwargs,
            "schedule": schedules[i],
            "iters_to_converge": iters_l[i],
            "reached_target": bool(reached_l[i]) and iters_l[i] < budgets[i],
            "best_f": best_f_l[i],
            "best_gnorm": best_g_l[i],
            "final_loss": last_f_l[i],
            "diverged": bool(diverged_l[i]),
            # Wall time is a property of the batch, not of one configuration;
            # recorded whole so it cannot be mistaken for a per-run timing.
            "time_seconds": elapsed,
            "batch_size": n_cfg,
        }
        if best_dual_l is not None:
            rec["best_dual"] = best_dual_l[i]
            rec["dual_norm"] = dual_norm
        if rho_stats[i] is not None:
            rec["rho"] = rho_stats[i]
        if keep_history:
            n = recorded_l[i]
            rec["loss_history"] = hist_f_l[i][:n]
            rec["grad_norm_history"] = hist_g_l[i][:n]
        out.append(rec)
    return out


def _rho_statistics(hist_rho: Tensor) -> List[Optional[Dict]]:
    """Per-configuration summary of the alignment trace, skipping the NaN pad.

    Matches ``run_one``: the sequential driver only appends ``rho_t`` when the
    denominator is nonzero, so a step that produced none is absent rather than
    zero, and here it is NaN rather than present.
    """
    stats: List[Optional[Dict]] = []
    for col in hist_rho.T.tolist():
        vals = sorted(v for v in col if v == v)          # drop NaN
        if not vals:
            stats.append(None)
            continue
        k = len(vals)
        stats.append({
            "min": vals[0],
            "p01": vals[max(0, k // 100)],
            "median": vals[k // 2],
            "mean": sum(vals) / k,
            "max": vals[-1],
            "frac_negative": sum(1 for v in vals if v < 0) / k,
        })
    return stats
