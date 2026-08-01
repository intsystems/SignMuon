"""Synthetic smooth convex benchmark: F(X) = 1/2 <X-C, A(X-C)B> (Equation 9).

    python3 -m synthetic.benchmark --mode grid       --device cuda:0
    python3 -m synthetic.benchmark --mode final      --device cuda:0
    python3 -m synthetic.benchmark --mode alignment  --device cuda:0
    python3 -m synthetic.benchmark --mode horizon    --device cuda:0
    python3 -m synthetic.benchmark --mode floor      --device cuda:0
    python3 -m synthetic.benchmark --mode stability  --device cuda:0
    python3 -m synthetic.benchmark --mode kappa      --device cuda:0

Problem
-------
``A in S_++^m``, ``B in S_++^n`` are symmetric with a prescribed spectrum in a
prescribed eigenbasis, so the Hessian of ``F`` is the Kronecker product
``B (x) A`` with eigenvalues ``lambda_i(A) * lambda_j(B)``. Two consequences are
used throughout:

* the Frobenius smoothness constant is *exactly* ``L = max_ij lambda_i mu_j``
  and the strong-convexity constant is *exactly* ``sigma = min_ij lambda_i mu_j``,
  both known in closed form -- no estimation anywhere below;
* ``grad F(X) = A (X - C) B`` and ``F(X) = 1/2 <X - C, grad F(X)>``, so both are
  computed in closed form. There is no autograd graph. This matters for the
  bidirectional method, whose gradient must be taken at the broadcast model
  ``W`` while the metrics are scored at the exact model ``X``; with a closed-form
  gradient either point is one matmul away.

``--spectrum uniform`` (the default) draws eigenvalues from ``U(0,1)``, which
bounds ``L <= 1`` and leaves ``L/sigma`` to the draw. ``--spectrum logspace
--kappa K`` fixes ``L = 1`` and ``L/sigma = K`` exactly, which is what ``kappa``
sweeps. Every measurement averages over three independent draws of ``(A, B)``.

What each mode measures
-----------------------
Everything below is a reading of the descent lemma

    F(X_{t+1}) <= F(X_t) - eta <grad F(X_t), D_t> + (eta^2 L / 2) ||D_t||_F^2,

whose first term is the rate and whose second is the floor.

``alignment``
    The distribution of the normalized alignment

        rho_t := <grad F(X_t), D_t> / (||grad F(X_t)||_F ||D_t||_F)

    along the tuned trajectory. The descent lemma makes progress contingent on
    it, and the divergence theorems are constructions that drive it negative;
    this is their empirical counterpart on random instances, and the one
    measurement here that is about the methods rather than the protocol. Three
    closed forms anchor it: ``rho = 1`` for SGD, ``||G||_1/(||G||_F sqrt(mn))``
    for SignSGD, ``||G||_nuc/(||G||_F sqrt(r))`` for Muon. The six
    sign-around-the-LMO methods admit none, which is the subject of the paper.

``floor``
    The plateau ``F_inf(eta)``, ``||grad F||_inf(eta)`` of a constant step and
    their exponents in ``eta``. Every normalized step has a floor: ``||D_t||_F``
    is ``sqrt(mn)`` for a sign step and ``sqrt(r)`` for an LMO step whatever the
    gradient does, so a constant ``eta`` settles into a ball of radius
    ``eta ||S||_F`` instead of converging. Balancing the two terms of the
    descent lemma predicts a gradient floor linear in ``eta`` with coefficient
    ``L ||S||_F / 2 rho``. SignMuon and SignSGD share ``||S||_F = sqrt(mn)``
    exactly, so any gap between their floors is attributable to ``rho`` alone.

``horizon``
    The rate. Tunes ``(eta, momentum, schedule)`` separately at each budget
    ``T`` and fits ``err ~ T^-p``, ``eta* ~ T^-q``, where
    ``err := min_t ||grad F(X_t)||_*^2`` is the squared dual norm the theorems
    bound -- ``l1`` for the sign family, nuclear for the LMO family, see
    ``DUAL_NORM``. ``p = q = 1/2`` is the nonconvex L-smooth regime the theorems
    prove; ``p = q = 1`` is strongly convex, which this quadratic also is. The
    schedule is tuned per method rather than imposed.

``stability``
    Largest stable ``eta``, by bisection, with SGD as the control: its value
    must reproduce the textbook ``2/L``. Reported as a step length
    ``eta_max ||S||_F``, which would be family-independent if the operative
    trust region were the Frobenius ball; the spread measures how far it is.

``kappa``
    Conditioning is the only knob that governs the dynamics of a quadratic, and
    the uniform draw leaves it to chance. This sweeps it over decades at
    ``L = 1``.

``grid`` / ``final``
    Fixed-target protocol: fewest iterations to ``F(X) <= 1e-3``, tuned per
    method, and the loss curves at those optima. Read it as a ranking of
    accuracy floors rather than of rates -- the iteration count comes out as
    ``const/eta_max``, so the tuner is returning the largest ``eta`` whose
    plateau still fits under the target. ``floor`` and ``horizon`` separate the
    two effects; this reports the criterion itself.

How a sweep is executed
-----------------------
Every configuration in a grid is an independent trajectory over the same
problem, so ``--runner batched`` (the default) advances the whole
``(eta, momentum, schedule)`` grid at once as a ``[B, m, n]`` stack; see
``synthetic.batched``. ``--runner sequential`` is :func:`run_one` in a loop, the
reference implementation the batched one is tested against.

Two conventions that move numbers
---------------------------------
* ``--lmo-dtype`` sets the Newton-Schulz working precision. The default
  ``bfloat16`` matches the reference Muon implementation but carries ~3 decimal
  digits, so for the methods that sign the LMO *output* an entry of
  ``polar(M)`` near zero can flip; ``float32`` removes that.
* ``EF21MuonSign`` is scored on its **exact** model ``X``, not on the broadcast
  model ``W`` held in ``p.data``. ``X`` is the iterate the theory bounds.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from common.utils import results_root
from synthetic import batched
from common.optimizers import (
    EF21MuonSign,
    EF21MuonUSign,
    EF21SignMuon,
    Muon,
    MuonSign,
    MuonUSign,
    SignMuon,
    SignSGD,
)

METHOD_CLASSES = {
    "signmuon": SignMuon,
    "ef21signmuon": EF21SignMuon,
    "muonusign": MuonUSign,
    "muonsign": MuonSign,
    "ef21muonusign": EF21MuonUSign,
    "ef21muonsign": EF21MuonSign,
    "muon": Muon,
    "signsgd": SignSGD,
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
}

DEFAULT_METHODS = list(METHOD_CLASSES)

#: Methods whose step is a ``+-1`` matrix (``||s||_F = sqrt(mn)``). The rest take
#: a step of Frobenius norm ``sqrt(min(m,n))`` -- except ``sgd``/``adam``, whose
#: step length is not normalized at all.
SIGN_FAMILY = ("signmuon", "muonsign", "signsgd")
LMO_FAMILY = ("muon", "muonusign", "ef21muonusign", "ef21muonsign", "ef21signmuon")

#: Norm dual to each method's LMO ball -- the one the convergence theorems bound
#: ``min_t ||grad F(X_t)||_*^2`` in. A sign step minimizes over the ``l_infty``
#: ball, so its dual is ``l_1``; an LMO step over the spectral ball, so nuclear.
#: SGD and Adam have no LMO ball; Frobenius is its own dual.
DUAL_NORM: Dict[str, str] = {
    **{m: "l1" for m in SIGN_FAMILY},
    **{m: "nuclear" for m in LMO_FAMILY},
    "sgd": "fro", "adam": "fro",
}


# --------------------------------------------------------------------------
# Learning-rate / momentum grids
# --------------------------------------------------------------------------

# Grid syntax: ``"lo:hi:step"`` is linear, ``"lo:hi:xN"`` logarithmic with N
# points per decade; both endpoints inclusive.
#
# The grids below are logarithmic and five decades wide, because the optimal eta
# differs by three orders of magnitude across these methods -- a sign step has
# fixed length eta*sqrt(mn), an LMO step eta*sqrt(r), and SGD's scales with the
# gradient. A grid narrow enough to miss that lands its optimum on an edge,
# which is an upper bound and not a tuned value; ``tune`` flags any such row.
#
# One spec per *family* rather than per method, keyed off the same
# SIGN_FAMILY/LMO_FAMILY lists that ``step_norm`` and ``DUAL_NORM`` use. Written
# out by hand, the two drifted apart: ``muonusign`` and ``ef21signmuon`` take an
# LMO-length step (``||s||_F = sqrt(r)``, so their optimal eta is ~10x that of a
# sign step at 100x100) but had been given the sign-family grid, which censored
# ``muonusign``'s optimum at the upper edge in the grid and horizon stages and
# left ``ef21signmuon``'s smallest floor point still unsettled after 60k
# iterations. Deriving the grid from the family is what stops that recurring.
#
# Each normalized family's window ends past the largest stability edge that
# ``--mode stability`` measures for it at 100x100 -- eta_max in 0.111-0.134 for
# a sign step, 1.49-1.89 for an LMO one -- so that no stable step size falls
# outside the search. That is the point: a stable eta the grid cannot reach is a
# censored optimum waiting to happen, which is what a window stopping at 0.01
# did to MuonUSign. An optimum on the top point now means the edge of
# *stability*, a fact about the method, not a grid too narrow to hold it.
#
# The points above each edge cost almost nothing: they diverge on the first few
# steps and the runner retires a diverged trajectory instead of running it out.
def family_lr_grids(sign: str, lmo: str, sgd: str, adam: str) -> Dict[str, str]:
    """One grid spec per method, expanded from one spec per step-norm family."""
    grids = {m: sign for m in SIGN_FAMILY}
    grids.update({m: lmo for m in LMO_FAMILY})
    grids["sgd"], grids["adam"] = sgd, adam
    return grids


#: Family names accepted by ``--lr-grid`` in place of a method name.
LR_GRID_FAMILIES: Dict[str, Sequence[str]] = {
    "sign": SIGN_FAMILY, "lmo": LMO_FAMILY,
}

DEFAULT_LR_GRIDS: Dict[str, str] = family_lr_grids(
    sign="1e-5:1e+0:x6", lmo="1e-4:1e+1:x6",
    sgd="1e-3:1e+1:x6", adam="1e-3:1e+1:x6",
)

#: Momentum is bounded below by 0 and above by 1; 0.99 is here because SGD's
#: optimum reached 0.95 -- the previous top point -- at the two longest budgets.
DEFAULT_MOMENTUM_GRID = "0.0,0.5,0.9,0.95,0.99"

#: One instance per seed, results averaged. Three draws rather than one, because
#: every statement here is about a *random* instance and one draw cannot support
#: it -- the alignment distribution least of all.
DEFAULT_PROBLEM_SEEDS = (1337, 1338, 1339)


def parse_lr_grid(spec: str) -> List[float]:
    """``"lo:hi:step"`` (linear) or ``"lo:hi:xN"`` (log, N per decade)."""
    lo, hi, step = spec.split(":")
    lo, hi = float(lo), float(hi)
    if step.startswith("x"):
        per_decade = float(step[1:])
        if per_decade <= 0:
            raise ValueError(f"points per decade must be positive: {spec!r}")
        decades = math.log10(hi / lo)
        n = int(round(decades * per_decade))
        return [lo * 10.0 ** (i * decades / n) for i in range(n + 1)]
    step = float(step)
    if step <= 0:
        raise ValueError(f"grid step must be positive: {spec!r}")
    n = int(round((hi - lo) / step))
    return [round(lo + i * step, 12) for i in range(n + 1)]


def parse_momentum_grid(spec: str) -> List[float]:
    return [float(x) for x in spec.split(",") if x.strip()]


# --------------------------------------------------------------------------
# Step-size schedules
# --------------------------------------------------------------------------

#: ``eta_t = eta * SCHEDULES[name](t, T)``. Tuned per method rather than fixed:
#: the constant-step bound and the decaying-step bound are different theorems
#: and there is no reason all six methods should prefer the same one.
SCHEDULES: Dict[str, Callable[[int, int], float]] = {
    "const":  lambda t, T: 1.0,
    "sqrt":   lambda t, T: 1.0 / math.sqrt(1.0 + t),
    "linear": lambda t, T: max(0.0, 1.0 - t / max(1, T)),
}


# --------------------------------------------------------------------------
# Problem
# --------------------------------------------------------------------------


class Quadratic:
    """``F(X) = 1/2 <X-C, A(X-C)B>`` with everything about it known exactly.

    ``L`` and ``sigma`` are the extreme eigenvalues of the Hessian ``B (x) A``,
    i.e. the extreme products ``lambda_i(A) lambda_j(B)``, so no estimation is
    needed anywhere. ``F_star = 0`` at ``X = C``.
    """

    def __init__(self, m: int, n: int, device, seed: int,
                 spectrum: str = "uniform", kappa: float = 1e4,
                 basis: str = "haar", shift: float = 0.0):
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))

        # A is drawn to completion before B; the order is part of what the
        # seed pins, so changing it would silently change every instance.
        la = self._spectrum(m, spectrum, kappa, device, gen)
        self.A = self._embed(la, basis, device, gen)
        lb = self._spectrum(n, spectrum, kappa, device, gen)
        self.B = self._embed(lb, basis, device, gen)

        prod = torch.outer(la, lb)
        self.L = float(prod.max())
        self.sigma = float(prod.min())
        self.kappa = self.L / self.sigma

        self.C = (torch.randn(m, n, device=device, generator=gen) * shift
                  if shift else None)
        self.m, self.n = m, n

    @staticmethod
    def _spectrum(dim: int, kind: str, kappa: float, device, gen) -> Tensor:
        if kind == "uniform":
            # Eigenvalues ~ U(0,1): L <= 1, condition number left to the draw.
            return torch.rand(dim, device=device, generator=gen)
        if kind == "logspace":
            # Each factor gets condition number sqrt(kappa) and top eigenvalue 1,
            # so the Kronecker product has L = 1 and condition number exactly
            # kappa.
            lo = kappa ** -0.5
            return torch.logspace(math.log10(lo), 0.0, dim, device=device)
        raise ValueError(f"unknown spectrum {kind!r}")

    @staticmethod
    def _embed(eigenvalues: Tensor, basis: str, device, gen) -> Tensor:
        if basis == "diagonal":
            return torch.diag(eigenvalues)
        if basis == "haar":
            dim = eigenvalues.numel()
            Q, _ = torch.linalg.qr(
                torch.randn(dim, dim, device=device, generator=gen))
            return Q @ torch.diag(eigenvalues) @ Q.T
        raise ValueError(f"unknown basis {basis!r}")

    def grad(self, X: Tensor) -> Tensor:
        D = X if self.C is None else X - self.C
        return self.A @ D @ self.B

    def value_and_grad(self, X: Tensor) -> Tuple[float, Tensor]:
        D = X if self.C is None else X - self.C
        G = self.A @ D @ self.B
        return 0.5 * float(torch.sum(D * G)), G

    def value_and_grad_batched(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        """``(F, grad F)`` for a stack of iterates ``[B, m, n]``; ``F`` is ``[B]``.

        ``A @ X @ B`` already broadcasts over the leading axis -- the two
        matmuls are batched and no slice sees another -- so the only change from
        the scalar version is that the inner product reduces over the matrix
        axes instead of everything, and that the value stays on the device
        rather than being pulled to the host every step.
        """
        D = X if self.C is None else X - self.C
        G = self.A @ D @ self.B
        return 0.5 * torch.sum(D * G, dim=(-2, -1)), G

    def initial_point(self, seed: int, scale: float = 0.1) -> Tensor:
        gen = torch.Generator(device=self.A.device)
        gen.manual_seed(int(seed))
        return torch.randn(self.m, self.n, device=self.A.device,
                           generator=gen) * scale


# --------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------


def _build(method: str, X: Tensor, kwargs: Dict[str, float], lmo_dtype: str):
    cls = METHOD_CLASSES[method]
    opt_kwargs = dict(kwargs)
    opt_kwargs.pop("schedule", None)
    if method in ("sgd", "adam"):
        opt_kwargs.pop("nesterov", None)
        if method == "adam":
            opt_kwargs.pop("momentum", None)
    else:
        opt_kwargs["lmo_dtype"] = getattr(torch, lmo_dtype)
    return cls([X], **opt_kwargs)


def run_one(
    method: str,
    kwargs: Dict[str, float],
    problem: Quadratic,
    target_loss: float = 1e-3,
    max_iters: int = 5000,
    init_seed: int = 42,
    lmo_dtype: str = "bfloat16",
    schedule: str = "const",
    capture_alignment: bool = False,
    keep_history: bool = False,
    stop_at_target: bool = False,
    dual_norm: Optional[str] = None,
) -> Dict:
    """Run one ``(method, hyperparameter)`` configuration for ``max_iters`` steps.

    ``iters_to_converge`` records the first iteration at which
    ``F <= target_loss`` (``max_iters`` if never). ``stop_at_target`` returns
    there, which is all the fixed-target protocol needs; the sweeps leave it
    off because they read the plateau, which only exists after the target is
    passed.

    ``init_seed`` fixes ``X_0``, so every configuration starts from the same
    point and the comparison is not confounded by initialization. It also pins
    the tie-break RNG, so the run depends only on its own seeds -- see
    ``batched.deterministic_rng``.
    """
    with batched.deterministic_rng(init_seed, problem.A.device):
        return _run_one_body(method, kwargs, problem, target_loss, max_iters,
                             init_seed, lmo_dtype, schedule, capture_alignment,
                             keep_history, stop_at_target, dual_norm)


def _run_one_body(method, kwargs, problem, target_loss, max_iters, init_seed,
                  lmo_dtype, schedule, capture_alignment, keep_history,
                  stop_at_target, dual_norm) -> Dict:
    """The loop itself. See :func:`run_one`."""
    X = torch.nn.Parameter(problem.initial_point(init_seed))
    optimizer = _build(method, X, kwargs, lmo_dtype)
    lr0 = kwargs["lr"]
    sched = SCHEDULES[schedule]

    if capture_alignment and hasattr(optimizer, "capture_direction"):
        optimizer.capture_direction = True

    def tracked() -> Tensor:
        """The iterate the convergence theory bounds.

        EF21MuonSign holds the compressed broadcast model ``W`` in ``p.data`` and
        the exact model ``X`` in its state; every other method has only one.
        """
        return optimizer.state.get(X, {}).get("exact_model", X.data)

    iters_to_converge = max_iters
    best_f = math.inf
    best_gnorm = math.inf
    best_dual = math.inf
    f_hist: List[float] = []
    g_hist: List[float] = []
    rho_hist: List[float] = []
    diverged = False
    start = time.time()

    for t in range(max_iters):
        # Gradient at p.data -- which is W for the bidirectional method, exactly
        # as its algorithm requires -- and metrics at the tracked iterate.
        X.grad = problem.grad(X.data)
        f_val, g_tracked = problem.value_and_grad(tracked())
        g_norm = float(g_tracked.norm())

        f_hist.append(f_val)
        g_hist.append(g_norm)
        best_f = min(best_f, f_val)
        best_gnorm = min(best_gnorm, g_norm)
        if dual_norm is not None:
            best_dual = min(best_dual, float(
                batched.dual_norm_of(g_tracked.unsqueeze(0), dual_norm)[0]))
        if f_val <= target_loss and iters_to_converge == max_iters:
            iters_to_converge = t + 1
            if stop_at_target:
                break

        if not math.isfinite(f_val) or f_val > 1e12:
            diverged = True
            break

        for group in optimizer.param_groups:
            group["lr"] = lr0 * sched(t, max_iters)
        optimizer.step()

        if capture_alignment:
            d = optimizer.state[X].get("last_direction")
            if d is not None:
                denom = g_norm * float(d.norm())
                if denom > 0:
                    rho_hist.append(float(torch.sum(g_tracked * d)) / denom)

    out = {
        "method": method,
        "kwargs": {k: v for k, v in kwargs.items()},
        "schedule": schedule,
        "iters_to_converge": iters_to_converge,
        "reached_target": iters_to_converge < max_iters,
        "best_f": best_f,
        "best_gnorm": best_gnorm,
        "final_loss": f_hist[-1] if f_hist else float("nan"),
        "diverged": diverged,
        "time_seconds": time.time() - start,
    }
    if dual_norm is not None:
        out["best_dual"] = best_dual
        out["dual_norm"] = dual_norm
    if rho_hist:
        rho_hist.sort()
        k = len(rho_hist)
        out["rho"] = {
            "min": rho_hist[0],
            "p01": rho_hist[max(0, k // 100)],
            "median": rho_hist[k // 2],
            "mean": sum(rho_hist) / k,
            "max": rho_hist[-1],
            "frac_negative": sum(1 for r in rho_hist if r < 0) / k,
        }
    if keep_history:
        out["loss_history"] = f_hist
        out["grad_norm_history"] = g_hist
    return out


# --------------------------------------------------------------------------
# Tuning
# --------------------------------------------------------------------------

#: ``objective -> (metric key, want-small)``. ``iters`` is the fixed-target
#: criterion; ``best_f`` / ``best_gnorm`` are what the rate statements are about.
OBJECTIVES = {
    "iters": ("iters_to_converge", True),
    "best_f": ("best_f", True),
    "best_gnorm": ("best_gnorm", True),
}


def _score(metrics: Dict, objective: str) -> Tuple[float, float]:
    key, _ = OBJECTIVES[objective]
    value = metrics[key]
    if not math.isfinite(value):
        return (math.inf, math.inf)
    # Ties on the iteration count (very common -- it is an integer) are broken by
    # the loss actually reached.
    return (value, metrics["best_f"])


def run_grid(
    method: str,
    problems: Sequence[Quadratic],
    configs: Sequence[Dict],
    runner: str = "batched",
    **run_kwargs,
) -> List[List[Dict]]:
    """``[instance][config]`` run records, by whichever runner is selected.

    ``batched`` advances the whole configuration list as one ``[B, m, n]``
    trajectory (see ``synthetic.batched``); ``sequential`` is the original
    one-configuration-at-a-time loop, kept as the reference implementation. Each
    config is ``{"lr", "momentum"?, "schedule"?, "max_iters"?}`` -- the per-config
    budget is what ``--mode floor`` needs, since it gives small steps a
    proportionally longer run.
    """
    if runner == "batched":
        return [batched.run_configs(method, prob, configs, **run_kwargs)
                for prob in problems]
    if runner != "sequential":
        raise ValueError(f"unknown runner {runner!r}")

    kw = dict(run_kwargs)
    kw.pop("compact_every", None)                  # batched-only knob
    default_T = kw.pop("max_iters", 5000)
    out = []
    for prob in problems:
        runs = []
        for c in configs:
            cfg = {"lr": c["lr"]}
            if method != "adam":
                cfg["momentum"] = c.get("momentum", 0.0)
            runs.append(run_one(method, cfg, prob,
                                schedule=c.get("schedule", "const"),
                                max_iters=int(c.get("max_iters", default_T)),
                                **kw))
        out.append(runs)
    return out


def _aggregate_over_instances(per_instance: Sequence[Sequence[Dict]]) -> List[Dict]:
    """Collapse ``[instance][config]`` to one record per config."""
    return [_aggregate([runs[i] for runs in per_instance])
            for i in range(len(per_instance[0]))]


def tune(
    method: str,
    problems: Sequence[Quadratic],
    lrs: Sequence[float],
    momenta: Sequence[float],
    schedules: Sequence[str],
    objective: str = "iters",
    verbose: bool = True,
    runner: str = "batched",
    **run_kwargs,
) -> Dict:
    """Grid-search ``(lr, momentum, schedule)``, averaging over problem instances.

    Averaging is geometric on the error metrics and arithmetic on the iteration
    count. Multiple instances are the point: one draw of ``A``, ``B`` is a
    single sample of a random problem.

    The whole grid goes to the runner at once rather than one configuration per
    call, which is what lets the batched runner exist at all; the enumeration
    order below is the order the verbose log prints in, and is unchanged.
    """
    best: Optional[Dict] = None
    boundary_lr = (lrs[0], lrs[-1])
    # Only the *upper* momentum end is a censored edge. Momentum is bounded
    # below by 0 -- the optimizers reject anything else -- so an optimum at
    # mom = 0 is the true optimum sitting on a natural bound, not a grid too
    # narrow to contain it. Flagging it sends the reader to widen a grid that
    # cannot be widened, and, under the protocol that treats a boundary hit as
    # a failed measurement, would discard a perfectly good row.
    boundary_mom = tuple(m for m in (momenta[0], momenta[-1]) if m > 0.0)

    configs = [{"lr": lr, "momentum": mom, "schedule": sch}
               for lr in lrs
               for mom in (momenta if method != "adam" else [0.0])
               for sch in schedules]
    results = _aggregate_over_instances(
        run_grid(method, problems, configs, runner, **run_kwargs))

    for cfg, agg in zip(configs, results):
        if best is None or _score(agg, objective) < _score(best, objective):
            best = agg
        if verbose:
            it = agg["iters_to_converge"]
            reached = agg["reached_target"]
            print(f"    lr={cfg['lr']:<10.4g} mom={cfg['momentum']:<5}"
                  f" sch={cfg['schedule']:<7}"
                  f" iters={(f'{it:.0f}' if reached else 'none'):<8}"
                  f" best_f={agg['best_f']:.3e}"
                  f" best_gn={agg['best_gnorm']:.3e}")

    assert best is not None
    edges = []
    if best["kwargs"]["lr"] in boundary_lr and len(lrs) > 1:
        edges.append("lr")
    if (boundary_mom and best["kwargs"].get("momentum") in boundary_mom
            and len(momenta) > 1):
        edges.append("momentum")
    best["on_grid_boundary"] = edges
    if edges and verbose:
        print(f"  !! {method}: tuned {', '.join(edges)} sits on the grid "
              f"boundary -- this is an upper bound, not an optimum. Widen with "
              f"--lr-grid / --momentum-grid.")
    return best


def _aggregate(runs: Sequence[Dict]) -> Dict:
    """Combine repeats over problem instances into one record."""
    if len(runs) == 1:
        return dict(runs[0])
    out = dict(runs[0])
    out["n_instances"] = len(runs)
    out["reached_target"] = all(r["reached_target"] for r in runs)
    out["diverged"] = any(r["diverged"] for r in runs)
    out["iters_to_converge"] = sum(r["iters_to_converge"] for r in runs) / len(runs)
    for key in ("best_f", "best_gnorm", "final_loss", "best_dual"):
        if key in runs[0]:
            out[key] = _geomean([r[key] for r in runs])
    if "rho" in runs[0]:
        out["rho"] = {k: (min(r["rho"][k] for r in runs) if k in ("min", "p01")
                          else sum(r["rho"][k] for r in runs) / len(runs))
                      for k in runs[0]["rho"]}
    # Curves are averaged, not dropped: ``--mode final`` is what draws them, and
    # it runs over the same three instances as everything else. Geometric mean
    # elementwise, matching the scalars, over the common prefix -- the runs stop
    # at the target and so can differ in length by a few steps.
    for key in ("loss_history", "grad_norm_history"):
        series = [r.get(key) for r in runs]
        if any(s is None for s in series):
            out.pop(key, None)
            continue
        n = min(len(s) for s in series)
        out[key] = [_geomean([s[t] for s in series]) for t in range(n)]
    return out


def _geomean(values: Sequence[float]) -> float:
    vals = [v for v in values if math.isfinite(v) and v > 0]
    if not vals:
        return math.inf
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def loglog_fit(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float]:
    """Least-squares slope of ``log y`` on ``log x``, and its ``R^2``.

    Returned as ``(slope, r2)``; the exponent reported in the tables is
    ``-slope``. ``R^2`` is reported alongside because a clean exponent on a
    two-point fit means nothing.
    """
    pts = [(x, y) for x, y in zip(xs, ys)
           if x > 0 and y > 0 and math.isfinite(y)]
    if len(pts) < 2:
        return (float("nan"), float("nan"))
    lx = [math.log(x) for x, _ in pts]
    ly = [math.log(y) for _, y in pts]
    k = len(pts)
    mx, my = sum(lx) / k, sum(ly) / k
    sxx = sum((a - mx) ** 2 for a in lx)
    if sxx == 0:
        return (float("nan"), float("nan"))
    slope = sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sxx
    ss_res = sum((b - my - slope * (a - mx)) ** 2 for a, b in zip(lx, ly))
    ss_tot = sum((b - my) ** 2 for b in ly)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return (slope, r2)


def step_norm(method: str, m: int, n: int) -> float:
    """``||s||_F`` of one unit step for the **exact** LMO (see ``common.lr_scaling``).

    The sign family realizes ``sqrt(mn)`` exactly -- a ``+-1`` matrix has no
    approximation in it. The LMO family does not: five Newton-Schulz steps leave
    the singular values in a band around 1 rather than at 1, so the realized step
    is measurably shorter than ``sqrt(min(m,n))`` -- 0.87x at ``64x64``, 0.94x at
    ``500x500``, and 0.78x at the most rectangular ResNet-18 shape (``64x576``).
    Anything comparing an LMO method's measured floor against ``L ||s||_F^2 / 2``
    should expect that much slack, and the same factor silently rescales the
    effective learning rate of every LMO-family run relative to the theory.
    """
    if method in SIGN_FAMILY:
        return math.sqrt(m * n)
    if method in LMO_FAMILY:
        return math.sqrt(min(m, n))
    return float("nan")            # sgd / adam: not a normalized step


# --------------------------------------------------------------------------
# Closed-form alignment references
# --------------------------------------------------------------------------


def rho_reference(method: str, G: Tensor) -> Optional[float]:
    """``rho`` predicted in closed form at gradient ``G``, where one exists.

    ``SGD``: ``d = G``, so ``rho = 1``.
    ``SignSGD``: ``d = sign(G)``, so ``rho = ||G||_1 / (||G||_F sqrt(mn))``,
    which tends to ``sqrt(2/pi) ~ 0.798`` as ``G`` becomes entrywise Gaussian.
    ``Muon``: ``d = polar(G)``, so ``rho = ||G||_nuc / (||G||_F sqrt(r))``.
    The three sign-around-the-LMO methods have no closed form; that is the point.
    """
    m, n = G.shape
    fro = float(G.norm())
    if fro == 0:
        return None
    if method == "sgd":
        return 1.0
    if method == "signsgd":
        return float(G.abs().sum()) / (fro * math.sqrt(m * n))
    if method == "muon":
        s = torch.linalg.svdvals(G.float())
        r = int((s > 1e-10 * s[0]).sum())
        return float(s.sum()) / (fro * math.sqrt(r))
    return None


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------


def tuned_hyperparameters(out_root: Path, method: str
                          ) -> Optional[Tuple[float, float, str, str]]:
    """``(lr, momentum, schedule, source)`` for ``--mode final``.

    Read from this run's own ``grid.json``, so the curves are drawn at the
    optimum the grid stage actually found; ``None`` if it has not run.
    """
    path = out_root / method / "grid.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        result = json.load(f)["result"]
    return (result["kwargs"]["lr"], result["kwargs"].get("momentum", 0.0),
            result.get("schedule", "const"), str(path))


def mode_grid(args, problems, lr_grids, momenta, out_root) -> List[Dict]:
    """Fixed-target protocol: fewest iterations to the target loss.

    Every configuration runs the full budget rather than stopping at the target,
    so ``best_f`` and ``best_gnorm`` are minima over the whole trajectory and are
    comparable with the other modes; ``iters_to_converge`` is the first crossing
    either way.
    """
    summary = []
    for method in args.methods:
        schedules = args.schedules
        if args.mode == "final":
            best = tuned_hyperparameters(out_root, method)
            if best is None:
                print(f"{method}: no grid.json in {out_root} -- run --mode grid "
                      f"first. Skipping.")
                continue
            lrs, moms, schedules = [best[0]], [best[1]], [best[2]]
            print(f"--- {method}: lr={best[0]:g} momentum={best[1]:g} "
                  f"schedule={best[2]} (from {best[3]}) ---")
        else:
            lrs = parse_lr_grid(lr_grids[method])
            moms = momenta
            n_cfg = (len(lrs) * (1 if method == "adam" else len(moms))
                     * len(schedules))
            print(f"--- {method}: {n_cfg} configuration(s) x "
                  f"{len(problems)} instance(s) ---")
        common = dict(target_loss=args.target_loss, max_iters=args.max_iters,
                      init_seed=args.init_seed, lmo_dtype=args.lmo_dtype)
        best_metrics = tune(method, problems, lrs, moms, schedules,
                            objective="iters", runner=args.runner,
                            keep_history=args.save_histories
                            and args.mode == "final", **common)
        if args.mode == "grid":
            # Re-run the winner on its own before reporting it. The tuner ran it
            # inside a batch of ~30 configurations, and in bfloat16 a matmul of
            # a different batch width can round differently -- enough to move an
            # iteration count by a percent or two. `final` re-runs the optimum
            # alone, so without this the table and the figure it accompanies
            # would quote slightly different numbers for the same run.
            opt = dict(best_metrics["kwargs"])
            opt["schedule"] = best_metrics["schedule"]
            edges = best_metrics["on_grid_boundary"]
            best_metrics = _aggregate_over_instances(
                run_grid(method, problems, [opt], args.runner, **common))[0]
            best_metrics["on_grid_boundary"] = edges
        _write(out_root, method, args.mode, args, problems, best_metrics)
        summary.append(best_metrics)

    print(f"\n{'method':<16}{'iters':>10}{'best F':>14}{'||g||':>12}   hyperparameters")
    print("-" * 84)
    for m in summary:
        it = (f"{m['iters_to_converge']:.0f}" if m["reached_target"]
              else f">{args.max_iters}")
        kw = ", ".join(f"{k}={v:g}" for k, v in m["kwargs"].items())
        flag = "  [BOUNDARY]" if m.get("on_grid_boundary") else ""
        print(f"{m['method']:<16}{it:>10}{m['best_f']:>14.3e}"
              f"{m['best_gnorm']:>12.3e}   {kw}, {m['schedule']}{flag}")
    return summary


def mode_alignment(args, problems, lr_grids, momenta, out_root) -> List[Dict]:
    """Distribution of ``rho_t = <g, d> / (||g|| ||d||)`` along a tuned run."""
    prob = problems[0]
    G0 = prob.grad(prob.initial_point(args.init_seed))

    summary = []
    for method in args.methods:
        lrs = parse_lr_grid(lr_grids[method])
        print(f"--- {method} ---")
        best = tune(method, problems, lrs, momenta, args.schedules,
                    objective="best_gnorm", verbose=args.verbose,
                    runner=args.runner,
                    target_loss=args.target_loss, max_iters=args.align_iters,
                    init_seed=args.init_seed, lmo_dtype=args.lmo_dtype,
                    capture_alignment=True)
        best["rho_reference_at_X0"] = rho_reference(method, G0)
        _write(out_root, method, "alignment", args, problems, best)
        summary.append(best)

    print(f"\n{'method':<16}{'min rho':>10}{'p01':>10}{'median':>10}{'mean':>10}"
          f"{'%neg':>8}{'ref@X0,mu=0':>13}{'tuned':>18}")
    print("-" * 97)
    for m in summary:
        r = m.get("rho")
        ref = m["rho_reference_at_X0"]
        ref_s = f"{ref:.4f}" if ref is not None else "--"
        cfg = (f"eta={m['kwargs']['lr']:.3g},mu="
               f"{m['kwargs'].get('momentum', 0):g}")
        if r is None:
            # torch.optim's SGD and Adam have no capture hook. SGD's rho is 1 by
            # definition; Adam's step is not one the descent lemma covers.
            print(f"{m['method']:<16}" + f"{'--':>10}" * 4
                  + f"{'--':>8}{ref_s:>13}{cfg:>18}")
            continue
        print(f"{m['method']:<16}{r['min']:>10.4f}{r['p01']:>10.4f}"
              f"{r['median']:>10.4f}{r['mean']:>10.4f}"
              f"{100 * r['frac_negative']:>7.2f}%{ref_s:>13}{cfg:>18}")
    print("\n  rho > 0 throughout means the descent lemma bites on every step of "
          "this trajectory:\n  the divergence theorems' construction does not "
          "occur under random data.\n"
          "  The reference column is the closed form for d = compressor(grad F) at "
          "X_0 with no\n  momentum, so it is only comparable to a row whose tuned "
          "mu is 0; with momentum the\n  step is built from M_t rather than the "
          "current gradient and rho drops accordingly.")
    return summary


def mode_horizon(args, problems, lr_grids, momenta, out_root) -> List[Dict]:
    """Tune per budget ``T`` and fit ``err ~ T^-p``, ``eta* ~ T^-q``."""
    budgets = args.budgets
    summary = []
    for method in args.methods:
        lrs = parse_lr_grid(lr_grids[method])
        dual = DUAL_NORM[method]
        rows = []
        for T in budgets:
            print(f"--- {method}, T={T} ---")
            best = tune(method, problems, lrs, momenta, args.schedules,
                        objective=args.objective, verbose=args.verbose,
                        runner=args.runner, target_loss=0.0, max_iters=T,
                        init_seed=args.init_seed, lmo_dtype=args.lmo_dtype)
            # The theorems bound min_t ||grad F||_*^2 in the norm dual to this
            # method's LMO ball, not the Frobenius norm the tuner ranks on. The
            # dual norm costs an SVD per step for the LMO family, so it is
            # measured once, on a re-run of the tuned optimum, rather than
            # across the grid.
            opt = dict(best["kwargs"])
            opt["schedule"] = best["schedule"]
            at_opt = _aggregate_over_instances(run_grid(
                method, problems, [opt], args.runner, target_loss=0.0,
                max_iters=T, init_seed=args.init_seed,
                lmo_dtype=args.lmo_dtype, dual_norm=dual))[0]
            rows.append({"T": T, "lr": best["kwargs"]["lr"],
                         "momentum": best["kwargs"].get("momentum"),
                         "schedule": best["schedule"],
                         "best_f": best["best_f"],
                         "best_gnorm": best["best_gnorm"],
                         "best_dual": at_opt["best_dual"],
                         "on_grid_boundary": best["on_grid_boundary"]})
            print(f"  -> lr*={rows[-1]['lr']:.4g} sch={rows[-1]['schedule']} "
                  f"best_f={rows[-1]['best_f']:.4e} "
                  f"best_gn={rows[-1]['best_gnorm']:.4e} "
                  f"best_{dual}={rows[-1]['best_dual']:.4e}")

        p_f, r2_f = loglog_fit(budgets, [r["best_f"] for r in rows])
        p_g, r2_g = loglog_fit(budgets, [r["best_gnorm"] for r in rows])
        p_d, r2_d = loglog_fit(budgets, [r["best_dual"] for r in rows])
        q, r2_q = loglog_fit(budgets, [r["lr"] for r in rows])
        rec = {"method": method, "rows": rows, "dual_norm": dual,
               "exponent_f": -p_f, "r2_f": r2_f,
               "exponent_gnorm": -p_g, "r2_gnorm": r2_g,
               # The reported exponent is for the *squared* dual norm, which is
               # what the theorems bound; squaring doubles the slope exactly and
               # leaves R^2 untouched.
               "exponent_dual_sq": -2.0 * p_d, "r2_dual": r2_d,
               "exponent_lr": -q, "r2_lr": r2_q}
        _write(out_root, method, "horizon", args, problems, rec)
        summary.append(rec)

    print(f"\n{'method':<16}{'p (||g||_*^2)':>15}{'R2':>7}{'p (||g||_F)':>13}"
          f"{'R2':>7}{'p (F)':>9}{'R2':>7}{'q (eta*)':>10}{'R2':>7}")
    print("-" * 91)
    for r in summary:
        print(f"{r['method']:<16}{r['exponent_dual_sq']:>15.3f}{r['r2_dual']:>7.3f}"
              f"{r['exponent_gnorm']:>13.3f}{r['r2_gnorm']:>7.3f}"
              f"{r['exponent_f']:>9.3f}{r['r2_f']:>7.3f}"
              f"{r['exponent_lr']:>10.3f}{r['r2_lr']:>7.3f}")
    print("\n  p is fitted on min_t ||grad F(X_t)||_*^2, the squared dual norm "
          "the theorems\n  bound -- l1 for the sign family, nuclear for the LMO "
          "family. The Frobenius\n  column is the same trajectory in a "
          "family-independent norm, for comparison\n  across rows.")
    print("  p = q = 1/2 is the nonconvex L-smooth rate the theorems prove: the "
          "rate term\n  (F_0-F_*)/(eta T) balanced against the floor "
          "eta L ||s||_F^2 / 2. p = q = 1 is what a\n  strongly convex problem "
          "gives -- which this quadratic is, sigma > 0 -- because the\n  rate "
          "term contracts geometrically and the error is floor-limited at "
          "eta ~ 1/T.\n  The fit therefore says which regime the instance is in. "
          "SGD has no floor, so no\n  power law fits it: expect a large p, and "
          "read its q as 'eta* is stability-limited\n  at every budget'.")
    return summary


def mode_floor(args, problems, lr_grids, momenta, out_root) -> List[Dict]:
    """Plateau of a constant step, and its exponent in ``eta``.

    Balancing the two terms of the descent lemma gives
    ``g_inf = eta L ||s||_F / (2 rho)`` -- slope 1 in ``eta``, which is what the
    fit tests. The value is an upper bound rather than a prediction (the lemma
    bounds the increase), and is measured here at ``momenta[0]``.
    """
    mom = momenta[0]
    summary = []
    for method in args.methods:
        lrs = parse_lr_grid(lr_grids[method])
        lr_max = max(lrs)
        print(f"--- {method} (momentum={mom}) ---")
        # Time to reach the plateau scales like 1/eta, so a fixed budget would
        # leave the small-eta runs still descending and fake a floor that is
        # really just "how far it got". Budget is scaled to match, capped so the
        # sweep terminates -- and carried per configuration, so the whole eta
        # grid can still go to the runner in one batch.
        configs = [{"lr": lr, "momentum": mom, "schedule": "const",
                    "max_iters": min(args.floor_max_iters,
                                     int(round(args.floor_iters * lr_max / lr)))}
                   for lr in lrs]
        per_instance = run_grid(method, problems, configs, args.runner,
                                target_loss=0.0, init_seed=args.init_seed,
                                lmo_dtype=args.lmo_dtype, keep_history=True)

        rows = []
        for i, cfg in enumerate(configs):
            runs = [inst[i] for inst in per_instance]
            f_pairs = [_plateau(r["loss_history"]) for r in runs]
            g_pairs = [_plateau(r["grad_norm_history"]) for r in runs]
            f_inf = _geomean([v for v, _ in f_pairs])
            g_inf = _geomean([v for v, _ in g_pairs])
            stable = all(not r["diverged"] for r in runs)
            settled = stable and all(s for _, s in g_pairs)
            rows.append({"lr": cfg["lr"], "iters": cfg["max_iters"],
                         "f_inf": f_inf, "g_inf": g_inf,
                         "stable": stable, "settled": settled})
            note = ("" if settled else
                    ("   DIVERGED" if not stable else "   still descending"))
            print(f"    lr={cfg['lr']:<10.4g} T={cfg['max_iters']:<7} "
                  f"F_inf={f_inf:.4e} |g|_inf={g_inf:.4e}{note}")

        good = [r for r in rows if r["settled"]]
        if len(good) < 2:
            print(f"  {method}: fewer than two settled points -- no floor to fit. "
                  f"For SGD that is the correct answer (its step vanishes with "
                  f"the gradient, so it converges linearly here); for a "
                  f"normalized step it means the budget was too short -- raise "
                  f"--floor-iters.")
        sf, r2f = loglog_fit([r["lr"] for r in good], [r["f_inf"] for r in good])
        sg, r2g = loglog_fit([r["lr"] for r in good], [r["g_inf"] for r in good])
        rec = {"method": method, "momentum": mom, "rows": rows,
               "n_settled": len(good),
               "slope_f": sf, "r2_f": r2f, "slope_gnorm": sg, "r2_gnorm": r2g,
               "step_norm": step_norm(method, args.m, args.n)}
        _write(out_root, method, "floor", args, problems, rec)
        summary.append(rec)

    L = problems[0].L
    print(f"\n{'method':<16}{'settled':>9}{'d log|g|/d log eta':>20}{'R2':>7}"
          f"{'d log F/d log eta':>19}{'R2':>7}{'L||s||^2/2':>13}")
    print("-" * 91)
    for r in summary:
        s = r["step_norm"]
        pred = L * s * s / 2 if math.isfinite(s) else float("nan")
        pred_s = f"{pred:.4g}" if math.isfinite(pred) else "--"
        n = f"{r['n_settled']}/{len(r['rows'])}"
        if r["n_settled"] < 2:
            print(f"{r['method']:<16}{n:>9}{'no floor':>20}{'':>7}{'':>19}"
                  f"{'':>7}{pred_s:>13}")
            continue
        print(f"{r['method']:<16}{n:>9}{r['slope_gnorm']:>20.3f}{r['r2_gnorm']:>7.3f}"
              f"{r['slope_f']:>19.3f}{r['r2_f']:>7.3f}{pred_s:>13}")
    print("\n  The descent lemma predicts a gradient floor linear in eta "
          "(slope 1) with\n  coefficient L||s||_F / (2 rho). SignMuon and "
          "SignSGD share ||s||_F = sqrt(mn)\n  exactly, so any gap between their "
          "floors is attributable to rho alone.\n  The last column is the other "
          "term of the lemma, the eta^2 coefficient\n  L||s||_F^2 / 2, printed as "
          "a scale for F_inf rather than as that prediction.")
    return summary


def mode_stability(args, problems, lr_grids, momenta, out_root) -> List[Dict]:
    """Largest stable ``eta``, by bisection on ``log eta``."""
    mom = momenta[0]
    T = args.stability_iters
    summary = []

    def stable(method: str, lr: float) -> bool:
        cfg = {"lr": lr} if method == "adam" else {"lr": lr, "momentum": mom}
        for p in problems:
            r = run_one(method, cfg, p, schedule="const", target_loss=0.0,
                        max_iters=T, init_seed=args.init_seed,
                        lmo_dtype=args.lmo_dtype)
            f0, _ = p.value_and_grad(p.initial_point(args.init_seed))
            if r["diverged"] or not (r["best_f"] < f0):
                return False
        return True

    for method in args.methods:
        lo, hi = args.stability_lo, args.stability_hi
        if not stable(method, lo):
            print(f"--- {method}: unstable already at eta={lo:g}; widen "
                  f"--stability-lo ---")
            continue
        censored = stable(method, hi)
        if censored:
            # Not a measurement: everything below is a lower bound. Adam lands
            # here by construction -- its step is bounded by roughly lr whatever
            # the gradient does, so it oscillates rather than diverging and has
            # no stability edge of this kind.
            print(f"--- {method}: still stable at eta={hi:g}; this is a LOWER "
                  f"BOUND, widen --stability-hi to measure it ---")
            eta_max = hi
        else:
            for _ in range(args.stability_steps):
                mid = math.sqrt(lo * hi)
                if stable(method, mid):
                    lo = mid
                else:
                    hi = mid
            eta_max = lo
        s = step_norm(method, args.m, args.n)
        rec = {"method": method, "eta_max": eta_max, "step_norm": s,
               "step_length": eta_max * s, "momentum": mom, "iters": T,
               "censored": censored}
        print(f"--- {method}: eta_max = {eta_max:.4g}, "
              f"eta_max*||s||_F = {eta_max * s:.4g} ---")
        _write(out_root, method, "stability", args, problems, rec)
        summary.append(rec)

    L = problems[0].L
    print(f"\n{'method':<16}{'eta_max':>12}{'||s||_F':>10}"
          f"{'eta_max*||s||_F':>18}{'x (2/L)':>10}")
    print("-" * 66)
    for r in summary:
        eta = ("> " if r["censored"] else "") + f"{r['eta_max']:.4g}"
        if not math.isfinite(r["step_norm"]):
            # sgd / adam: the step is not normalized, so there is no ||s||_F.
            # SGD's eta_max is directly comparable to the textbook 2/L instead.
            print(f"{r['method']:<16}{eta:>12}{'--':>10}{'--':>18}"
                  f"{r['eta_max'] / (2 / L):>10.3f}")
            continue
        print(f"{r['method']:<16}{eta:>12}{r['step_norm']:>10.4g}"
              f"{r['step_length']:>18.4g}{r['step_length'] / (2 / L):>10.3f}")
    print(f"\n  L = {L:.4g}. SGD is the control: its eta_max must land on the "
          f"textbook 2/L.\n  If the operative trust region were the Frobenius "
          f"ball, eta_max*||s||_F\n  would be family-independent and near 2/L = "
          f"{2 / L:.4g}; the spread measures how far\n  off the Frobenius bound "
          f"is for each step geometry.")
    return summary


def mode_kappa(args, problems, lr_grids, momenta, out_root) -> List[Dict]:
    """The tuned comparison, swept over a controlled condition number."""
    summary = []
    probs = problems
    for method in args.methods:
        lrs = parse_lr_grid(lr_grids[method])
        rows = []
        for kappa in args.kappas:
            probs = [Quadratic(args.m, args.n, args.device, seed,
                               spectrum="logspace", kappa=kappa,
                               basis=args.basis, shift=args.shift)
                     for seed in args.problem_seeds]
            print(f"--- {method}, kappa={kappa:g} ---")
            best = tune(method, probs, lrs, momenta, args.schedules,
                        objective=args.objective, verbose=args.verbose,
                        runner=args.runner,
                        target_loss=args.target_loss, max_iters=args.max_iters,
                        init_seed=args.init_seed, lmo_dtype=args.lmo_dtype)
            rows.append({"kappa": kappa, "lr": best["kwargs"]["lr"],
                         "momentum": best["kwargs"].get("momentum"),
                         "schedule": best["schedule"],
                         "iters": best["iters_to_converge"],
                         "reached": best["reached_target"],
                         "best_f": best["best_f"],
                         "best_gnorm": best["best_gnorm"],
                         # Recorded here for the same reason as in every other
                         # tuning mode: without it a censored optimum reads as a
                         # measured one, and this sweep tunes at every kappa.
                         "on_grid_boundary": best["on_grid_boundary"]})
            flag = ("  [BOUNDARY: " + ", ".join(rows[-1]["on_grid_boundary"]) + "]"
                    if rows[-1]["on_grid_boundary"] else "")
            print(f"  -> lr*={rows[-1]['lr']:.4g} iters={rows[-1]['iters']:.0f} "
                  f"best_gn={rows[-1]['best_gnorm']:.4e}{flag}")
        slope, r2 = loglog_fit([r["kappa"] for r in rows],
                               [r["best_gnorm"] for r in rows])
        rec = {"method": method, "rows": rows, "kappas": list(args.kappas),
               "exponent_kappa": slope, "r2_kappa": r2}
        # ``probs`` is the last kappa's instance list; the per-kappa L and sigma
        # are in ``rows``, this only records the shared shape/basis metadata --
        # with the spectrum this sweep actually used, not the one on the command
        # line, which it overrides.
        _write(out_root, method, "kappa", args, probs, rec,
               problem_overrides={"spectrum": "logspace",
                                  "kappa_sweep": list(args.kappas)})
        summary.append(rec)

    header = f"{'method':<16}" + "".join(f"{k:>12.0e}" for k in args.kappas)
    print(f"\nbest ||grad F|| after {args.max_iters} tuned iterations")
    print(header)
    print("-" * len(header))
    for r in summary:
        print(f"{r['method']:<16}"
              + "".join(f"{row['best_gnorm']:>12.3e}" for row in r["rows"]))
    print(f"\n{'method':<16}{'d log||g|| / d log kappa':>26}{'R2':>7}")
    print("-" * 49)
    for r in summary:
        print(f"{r['method']:<16}{r['exponent_kappa']:>26.3f}{r['r2_kappa']:>7.3f}")
    return summary


MODES = {
    "grid": mode_grid,
    "final": mode_grid,
    "alignment": mode_alignment,
    "horizon": mode_horizon,
    "floor": mode_floor,
    "stability": mode_stability,
    "kappa": mode_kappa,
}

#: One size for every mode, so all of the reported numbers describe the same
#: instance family. Override per run with ``--m/--n``.
DEFAULT_SIZE = 100


def _plateau(history: Sequence[float], tol: float = 0.15) -> Tuple[float, bool]:
    """``(level, settled)`` for the tail of a constant-step trajectory.

    ``level`` is the mean over the last quarter. ``settled`` compares it against
    the third quarter: a run that is still descending has not reached its floor,
    and reading one off it would report "how far it got in the budget" instead.
    That distinction is the whole point of this mode, so non-settled points are
    excluded from the fit rather than silently averaged in.

    Every *normalized* step has a floor, which is all of them except SGD: an LMO
    step has fixed norm ``sqrt(r)`` and a sign step fixed norm ``sqrt(mn)``
    whatever the gradient does, so a constant ``eta`` cannot converge. Muon is
    not exempt; only plain SGD, whose step vanishes with the gradient, converges
    linearly here and never settles.
    """
    k = len(history)
    q3 = [v for v in history[k // 2: 3 * k // 4] if math.isfinite(v)]
    q4 = [v for v in history[3 * k // 4:] if math.isfinite(v)]
    if not q4 or not q3:
        return (float("inf"), False)
    m4 = sum(q4) / len(q4)
    m3 = sum(q3) / len(q3)
    if m4 <= 0:
        return (m4, False)
    return (m4, abs(m4 - m3) / m4 < tol)


def _write(out_root: Path, method: str, mode: str, args, problems, payload,
           problem_overrides: Optional[Dict] = None) -> None:
    """Write ``<method>/<mode>.json``, stamping the instance it was measured on.

    ``problem_overrides`` exists for ``kappa``, which builds its own log-spaced
    instances rather than the ones ``main`` handed down: recording ``args.spectrum``
    there would label a log-spaced sweep ``uniform``, contradicting the very
    columns beside it.
    """
    method_dir = out_root / method
    method_dir.mkdir(parents=True, exist_ok=True)
    p0 = problems[0]
    problem = {
        "m": args.m, "n": args.n, "spectrum": args.spectrum,
        "basis": args.basis, "shift": args.shift,
        "L": p0.L, "sigma": p0.sigma, "condition_number": p0.kappa,
        "target_loss": args.target_loss, "max_iters": args.max_iters,
        "problem_seeds": args.problem_seeds, "init_seed": args.init_seed,
        "lmo_dtype": args.lmo_dtype, "runner": args.runner,
    }
    problem.update(problem_overrides or {})
    body = {
        "problem": problem,
        "schedules": args.schedules,
        "result": payload,
    }
    with open(method_dir / f"{mode}.json", "w", encoding="utf-8") as f:
        json.dump(body, f, indent=2)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def get_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=list(MODES), default="grid")
    p.add_argument("--methods", nargs="*", default=DEFAULT_METHODS,
                   choices=DEFAULT_METHODS)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--verbose", action="store_true",
                   help="print every configuration, not just the tuned one")
    p.add_argument("--runner", choices=["batched", "sequential"],
                   default="batched",
                   help="'batched' advances the whole (lr, momentum, schedule) "
                        "grid as one [B, m, n] trajectory, which is 10-40x "
                        "faster because the per-step cost here is kernel launch "
                        "latency, not arithmetic; 'sequential' is the original "
                        "one-run-at-a-time loop, kept as the reference "
                        "implementation. The two agree exactly in float32 and "
                        "to bfloat16 rounding otherwise")

    g = p.add_argument_group("problem")
    g.add_argument("--m", type=int, default=None,
                   help=f"matrix rows (default {DEFAULT_SIZE}, every mode, so "
                        f"that every reported number describes one instance "
                        f"family); not a cost knob -- see synthetic/README.md")
    g.add_argument("--n", type=int, default=None,
                   help=f"matrix columns (default {DEFAULT_SIZE})")
    g.add_argument("--spectrum", choices=["uniform", "logspace"], default="uniform",
                   help="uniform: U(0,1) eigenvalues (L<=1, kappa left to the draw); "
                        "logspace: L=1 and condition number exactly --kappa")
    g.add_argument("--kappa", type=float, default=1e4,
                   help="condition number for --spectrum logspace")
    g.add_argument("--basis", choices=["haar", "diagonal"], default="haar",
                   help="eigenbasis of A and B; Muon and SGD are equivariant "
                        "under it, the sign family is not")
    g.add_argument("--shift", type=float, default=0.0,
                   help="entrywise scale of a random minimizer C; 0 puts the "
                        "minimizer at the origin, which is a special point for "
                        "sign methods")
    g.add_argument("--problem-seeds", type=int, nargs="+", default=None,
                   help=f"one instance per seed, results averaged -- geometric "
                        f"mean on the error metrics, arithmetic on the iteration "
                        f"count (default {list(DEFAULT_PROBLEM_SEEDS)}, every mode)")
    g.add_argument("--init-seed", type=int, default=42, help="seed for X_0")

    g = p.add_argument_group("run")
    g.add_argument("--target-loss", type=float, default=1e-3)
    g.add_argument("--max-iters", type=int, default=5000)
    g.add_argument("--lmo-dtype", choices=["bfloat16", "float32"], default="bfloat16")
    g.add_argument("--objective", choices=list(OBJECTIVES), default="best_gnorm",
                   help="what the tuner minimizes in horizon/kappa mode "
                        "(grid/final always use the iteration count)")

    g = p.add_argument_group("grids")
    g.add_argument("--lr-grid", nargs="*", default=[],
                   metavar="METHOD|FAMILY=lo:hi:step",
                   help="override the grid of one method, or of a whole "
                        "step-norm family ('sign', 'lmo'); 'lo:hi:step' is "
                        "linear, 'lo:hi:xN' is N points per decade")
    g.add_argument("--momentum-grid", type=str, default=None)
    g.add_argument("--schedules", nargs="+", default=None,
                   choices=list(SCHEDULES),
                   help="step-size schedules to tune over, separately per method "
                        "(default: 'const' everywhere except horizon mode, which "
                        "tunes over 'const sqrt' because nothing says the six "
                        "methods want the same schedule)")

    g = p.add_argument_group("mode-specific")
    g.add_argument("--budgets", type=int, nargs="+",
                   default=[125, 250, 500, 1000, 2000, 4000],
                   help="horizon mode: iteration budgets T")
    g.add_argument("--align-iters", type=int, default=2000,
                   help="alignment mode: trajectory length")
    g.add_argument("--floor-iters", type=int, default=3000,
                   help="floor mode: budget at the LARGEST eta in the grid; "
                        "smaller eta gets proportionally longer, since time to "
                        "plateau scales like 1/eta")
    g.add_argument("--floor-max-iters", type=int, default=60000,
                   help="floor mode: cap on that scaled budget")
    g.add_argument("--stability-iters", type=int, default=300)
    g.add_argument("--stability-lo", type=float, default=1e-6)
    g.add_argument("--stability-hi", type=float, default=1e2)
    g.add_argument("--stability-steps", type=int, default=20)
    g.add_argument("--kappas", type=float, nargs="+",
                   default=[1e1, 1e2, 1e3, 1e4, 1e5, 1e6])

    g = p.add_argument_group("output")
    g.add_argument("--out", type=str, default=None,
                   help="output directory (default: results/synthetic/)")
    g.add_argument("--save-histories", action="store_true",
                   help="store the full loss/grad-norm curves (large)")
    return p.parse_args()


def main() -> None:
    args = get_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.m is None:
        args.m = DEFAULT_SIZE
    if args.n is None:
        args.n = DEFAULT_SIZE
    if args.problem_seeds is None:
        args.problem_seeds = list(DEFAULT_PROBLEM_SEEDS)
    if args.momentum_grid is None:
        args.momentum_grid = DEFAULT_MOMENTUM_GRID
    if args.schedules is None:
        args.schedules = ["const", "sqrt"] if args.mode == "horizon" else ["const"]

    lr_grids = dict(DEFAULT_LR_GRIDS)
    for override in args.lr_grid:
        key, _, spec = override.partition("=")
        if key in LR_GRID_FAMILIES:
            lr_grids.update({m: spec for m in LR_GRID_FAMILIES[key]})
        elif key in METHOD_CLASSES:
            lr_grids[key] = spec
        else:
            raise ValueError(f"--lr-grid: unknown method or family {key!r} "
                             f"(families: {', '.join(LR_GRID_FAMILIES)})")
    momenta = parse_momentum_grid(args.momentum_grid)

    problems = [Quadratic(args.m, args.n, args.device, seed,
                          spectrum=args.spectrum, kappa=args.kappa,
                          basis=args.basis, shift=args.shift)
                for seed in args.problem_seeds]
    p0 = problems[0]

    print(f"mode={args.mode}  device={args.device}  {args.m}x{args.n}  "
          f"lmo_dtype={args.lmo_dtype}")
    print(f"problem: spectrum={args.spectrum} basis={args.basis} "
          f"shift={args.shift}  instances={len(problems)}")
    print(f"         L={p0.L:.6g}  sigma={p0.sigma:.6g}  "
          f"condition number={p0.kappa:.4g}\n")

    out_root = Path(args.out) if args.out else results_root() / "synthetic"
    out_root.mkdir(parents=True, exist_ok=True)

    MODES[args.mode](args, problems, lr_grids, momenta, out_root)
    print(f"\nResults written to {out_root.resolve()}")


if __name__ == "__main__":
    main()
