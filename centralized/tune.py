"""Equal-budget, validation-only learning-rate search.

The overnight driver (``centralized.overnight``) runs all of this for you; these
stages exist for driving one of them by hand.

    # is the optimal auxiliary rate method-independent?
    python3 -m centralized.tune --stage aux --epochs 15 --device cuda:0

    # which per-layer scaling exponent?  (not needed for the paper: the
    # --log-gain diagnostic measures it directly)
    python3 -m centralized.tune --stage alpha --epochs 15 --method signmuon

    # eta_0 per method, identical budget for all of them
    python3 -m centralized.tune --stage lr --lr-scaling unit-gain --device cuda:0

Three properties make this a defensible protocol rather than a sweep:

1. **Selection is on validation accuracy only** (``--split tune``, a fixed 45k/5k
   partition). ``best_of`` ranks on ``val_acc`` and nothing else. Test accuracy is
   still *computed and logged* every epoch -- it is simply never used to choose,
   which is the property that matters. (``federated.tune`` goes further and does
   not evaluate the test set at all under ``--split tune``.)
2. **Equal budget.** Every method gets the same number of configurations, on a grid
   that is *multiplicatively* anchored -- a shared absolute grid would be unfair
   because the families have genuinely different natural scales.
3. **Boundary check.** If a method's winner sits at an endpoint of its grid, that is
   reported as a failure rather than a result -- an optimum on the boundary is the
   second-most-common reviewer catch after test-set tuning. This module *reports*
   it; the automatic widen-and-re-run loop lives in ``centralized.overnight``
   (``extend_grid``, up to four widenings), because it needs the resumable job
   state that only the overnight driver keeps.

``--epochs`` defaults to the **reporting horizon**, 75, for the ``lr`` stage. An
earlier version tuned at 15 and reported at 75; re-running the top rates at 75
reversed the ranking for both methods checked, because each run anneals cosinally
over its own horizon and a short run therefore spends its budget on a different
schedule. The ``aux`` and ``alpha`` stages ask only whether two arms pick the
*same* value at a shared horizon, where that bias cancels, so pass
``--epochs 15`` there and keep them cheap.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from centralized.train import LMO_FAMILY
from common.lr_scaling import FAMILY_SIGN, RULES
from common.paths import scrub_text
from common.utils import results_root

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # code/, the package root

#: All ten methods, in the paper's order.
ALL_METHODS = ["signmuon", "muonusign", "muonsign", "ef21signmuon",
               "ef21muonusign", "ef21muonsign", "muon", "signsgd", "sgd", "adam"]

#: Anchor for each method's grid, in the ``legacy`` parameterization. New methods
#: are anchored at their sibling: MuonUSign<->Muon (both take a full LMO step),
#: MuonSign<->SignMuon (both take a unit-sign step), EF21-SignMuon<->EF21-MuonUSign.
LEGACY_ANCHORS: Dict[str, float] = {
    "signmuon": 1e-3,
    "muonsign": 1e-3,
    "signsgd": 1e-3,
    "muon": 1.5e-2,
    "muonusign": 1.5e-2,
    "ef21muonusign": 8e-3,
    "ef21muonsign": 7e-3,
    "ef21signmuon": 8e-3,
    "sgd": 1.5e-2,
    "adam": 1e-3,
}

#: Under a scaling rule the sign family's eta_0 is a *base* rate, larger than the
#: legacy per-layer-constant rate by roughly the reciprocal of the rule's typical
#: multiplier. These are order-of-magnitude anchors only; the coarse grid spans
#: 1.5 decades either side, so being 3x off costs nothing.
SCALED_ANCHOR_BOOST: Dict[str, float] = {
    "none": 1.0,
    "legacy": 1.0,
    "unit-gain": 34.0,          # ~sqrt(fan_in) at the median ResNet-18 layer
    "mup": 1152.0,              # ~fan_in at the median layer
    "mishra-analysis": 384.0,   # ~sqrt(m n) at the median layer
}

AUX_GRID = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3]
ALPHA_GRID = [0.0, 0.5, 1.0]


def anchor_for(method: str, rule: str) -> float:
    """Where ``method``'s grid is centred under a per-layer scaling ``rule``.

    Only the sign family's anchor moves with the rule: the LMO family carries the
    aspect factor under every rule, so its ``eta_0`` means the same thing
    throughout, and ``sgd``/``adam`` are run unscaled, as practitioners run them.
    One function rather than the same three lines in each caller -- they had
    drifted apart once already, and a mis-anchored grid is invisible until the
    tuned rate comes out at an endpoint twelve hours later.
    """
    if method not in LEGACY_ANCHORS:
        raise KeyError(f"unknown method {method!r}; known: {sorted(LEGACY_ANCHORS)}")
    cls = LMO_FAMILY.get(method)                # None for the torch baselines
    boost = (SCALED_ANCHOR_BOOST.get(rule, 1.0)
             if cls is not None and cls.family == FAMILY_SIGN else 1.0)
    return LEGACY_ANCHORS[method] * boost

# --------------------------------------------------------------------------
# The round grid
# --------------------------------------------------------------------------
#
# Every learning rate the driver tries is a 1-2-5 lattice point: a mantissa from
# ``ROUND_MANTISSAS`` times a power of ten. Three reasons this is worth pinning
# down rather than letting a log-spaced grid land wherever it lands:
#
# 1. A tuned value is quotable. The paper's tables carry ``0.03``, not
#    ``0.03231652035047826``, and a reader can re-run it from the table.
# 2. Extension stays on the same lattice, so a widened grid is still one uniform
#    sweep -- the property the old geometric extension preserved by accident of
#    its own spacing and lost as soon as two extensions used different strides.
# 3. Two methods anchored at slightly different places snap to the *same* grid,
#    so the equal-budget claim is exact rather than approximate.
#
# The lattice spacing is 2x-2.5x (mean 2.15x), i.e. three points per decade,
# which is finer than the 4.64x the previous four-point/two-decade default used.

ROUND_MANTISSAS = (1.0, 2.0, 5.0)


def lattice_value(k: int) -> float:
    """The ``k``-th 1-2-5 lattice point; ``k=0`` is ``1.0``, and ``k`` may be negative.

    Python's floor division and modulo agree in sign, so negative ``k`` walks the
    lattice downwards without a special case: ``k=-1`` is ``0.5``, ``k=-3`` is
    ``0.1``. The result is re-parsed at six significant digits so the value has a
    clean decimal repr and reaches ``centralized.main`` as ``0.02`` rather than
    ``0.020000000000000004``.
    """
    return float(f"{ROUND_MANTISSAS[k % 3] * 10.0 ** (k // 3):.6g}")


def lattice_index(x: float) -> int:
    """Index of the lattice point closest to ``x`` in log space."""
    if x <= 0:
        raise ValueError(f"learning rate must be positive; got {x}")
    approx = 3 * math.log10(x)
    target = math.log10(x)
    candidates = range(math.floor(approx) - 2, math.ceil(approx) + 3)
    return min(candidates, key=lambda k: abs(math.log10(lattice_value(k)) - target))


def round_grid(anchor: float, points: int) -> List[float]:
    """``points`` consecutive lattice values centred on the one nearest ``anchor``.

    ``points`` alone fixes the span: three lattice points per decade, so seven
    points is a little over two decades. There is deliberately no ``decades``
    argument -- the lattice sets the resolution, and letting a caller ask for both
    a span and a point count would reintroduce the off-lattice strides this exists
    to remove.
    """
    if points < 1:
        return [float(f"{anchor:.6g}")]
    k0 = lattice_index(anchor)
    half = (points - 1) // 2
    return [lattice_value(k0 - half + i) for i in range(points)]


def extend_grid(grid: Sequence[float], *, low: bool, points: int = 2) -> List[float]:
    """Continue ``grid`` ``points`` lattice steps past one endpoint.

    An optimum sitting on an endpoint is not an optimum -- the grid simply ran out
    before the objective turned over. Extension walks the same lattice, so however
    many rounds it takes, the union is still one uniform sweep and every method
    keeps the same resolution per decade searched.
    """
    g = sorted(grid)
    if not g:
        return []
    if low:
        k = lattice_index(g[0])
        return sorted([lattice_value(k - i - 1) for i in range(points)] + g)
    k = lattice_index(g[-1])
    return sorted(g + [lattice_value(k + i + 1) for i in range(points)])


# --------------------------------------------------------------------------
# Running one configuration
# --------------------------------------------------------------------------

_SUMMARY_RE = re.compile(r"val_acc\s+mean of last\s+\d+\s*:\s*([\d.]+)%")
_FINAL_RE = re.compile(r"test_acc mean of last\s+\d+\s*:\s*([\d.]+)%")


_EPOCH_TIME_RE = re.compile(r"median epoch time\s*:\s*([\d.]+)s")
_TARGET_RE = re.compile(r"epochs to [\d.]+% test acc\s*:\s*(\d+)")


def canonical_tag(tag: str, *, epochs: int, split: str = "tune",
                  seed: Optional[int] = None) -> str:
    """The unique identity of a job.

    Two runs of the same method at the same learning rate but a different epoch
    count, split or seed are *different experiments*. The identity must say so, or a
    resume keyed on the tag would treat them as one and silently mix horizons inside
    a single selection.

    Callers must use this for their bookkeeping keys as well -- ``run_one`` applying
    it internally is not enough, because the caller decides what to skip on resume
    *before* ``run_one`` is reached.
    """
    return f"{tag}_e{epochs}_{split[0]}" + ("" if seed is None else f"s{seed}")


def run_one(args, *, lr: float, lr_aux: float, lr_scaling: str,
            method: str, epochs: int, tag: str, split: str = "tune",
            seed: Optional[int] = None,
            extra: Sequence[str] = ()) -> Optional[Dict[str, float]]:
    """Launch one training run; return its metrics, or ``None`` on failure.

    Selection uses ``val_acc`` averaged over the last ``--last-k`` epochs -- the
    tail mean, not a single epoch, so the choice is not decided by one noisy
    evaluation. ``test_acc`` is parsed and recorded but is **never** used for
    selection; with ``split="tune"`` it is measured on 45k-trained models anyway.

    The job identity is canonicalized here to include the **horizon, split and
    seed**. Two runs of the same method at the same learning rate but different
    epoch counts are different experiments, and a resume that treated them as one
    would silently mix horizons inside a single selection -- so the caller cannot
    forget to disambiguate them.
    """
    last_k = min(args.last_k, max(1, epochs // 3))
    tag = canonical_tag(tag, epochs=epochs, split=split, seed=seed)

    cmd = [
        sys.executable, "-m", "centralized.main",
        "--dataset", args.dataset, "--model", args.model,
        "--optimizer", method, "--lr-scaling", lr_scaling,
        "--split", split, "--val-seed", str(args.val_seed),
        "--epochs", str(epochs), "--batch-size", str(args.batch_size),
        "--lr", repr(lr), "--lr-aux", repr(lr_aux),
        "--momentum", str(args.momentum), "--weight-decay", str(args.weight_decay),
        "--weight-decay-mode", getattr(args, "weight_decay_mode", "decoupled"),
        "--head-adamw", args.head_adamw, "--last-k", str(last_k),
        "--device", args.device,
        "--seed", str(args.seed if seed is None else seed),
        "--data", args.data, "--num-workers", str(args.num_workers),
        "--run-name", f"{'tune' if split == 'tune' else 'final'}_{tag}",
        *extra,
    ]
    if getattr(args, "nondeterministic", False):
        cmd.append("--nondeterministic")
    if getattr(args, "download", False):
        cmd.append("--download")

    log_dir = results_root() / "tuning_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{tag}.log"

    print(f"  lr={lr:<12.6g} lr_aux={lr_aux:<10.6g} -> ", end="", flush=True)
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT, text=True)
    # Rewritten in place: the child prints where it saved its run, and a traceback
    # quotes the source tree, so its stdout carries the writing machine's paths --
    # and `results/` now ships with the code. See `federated/tune.py`.
    text = scrub_text(log_path.read_text(encoding="utf-8", errors="replace"))
    log_path.write_text(text, encoding="utf-8")

    if proc.returncode != 0:
        tail = "".join(text.splitlines(keepends=True)[-3:]).strip()
        print(f"FAILED (exit {proc.returncode}) {tail[:160]}")
        return None

    run_name = f"{'tune' if split == 'tune' else 'final'}_{tag}"
    metrics = (results_root() / "centralized" / run_name /
               f"seed{args.seed if seed is None else seed}" / "metrics.json")
    out: Dict[str, float] = {"lr": lr, "lr_aux": lr_aux, "epochs": epochs,
                             "split": split, "log": scrub_text(str(log_path)),
                             "metrics": scrub_text(str(metrics))}
    m_val = _SUMMARY_RE.search(text)
    m_test = _FINAL_RE.search(text)
    m_time = _EPOCH_TIME_RE.search(text)
    m_target = _TARGET_RE.search(text)
    if m_val:
        out["val_acc"] = float(m_val.group(1))
    if m_test:
        out["test_acc"] = float(m_test.group(1))     # recorded, NEVER selected on
    if m_time:
        out["epoch_seconds"] = float(m_time.group(1))
    if m_target:
        out["epochs_to_target"] = int(m_target.group(1))

    key = "val_acc" if "val_acc" in out else "test_acc"
    if key not in out:
        print(f"no metric found (see {log_path.name})")
        return None
    print(f"{key.replace('_', ' ')} {out[key]:.2f}%"
          + (f"  [{out['epoch_seconds']:.1f}s/ep]" if "epoch_seconds" in out else ""))
    return out


def best_of(results: Sequence[Optional[Dict[str, float]]],
            key: str = "val_acc") -> Optional[Dict[str, float]]:
    """Best run by ``key`` (validation accuracy by default)."""
    valid = [r for r in results if r is not None and key in r]
    return max(valid, key=lambda r: r[key]) if valid else None


def boundary_warning(best: Dict[str, float], grid: Sequence[float], key: str = "lr") -> str:
    """Flag an optimum sitting on an endpoint of the grid."""
    if not grid or best is None:
        return ""
    lo, hi = min(grid), max(grid)
    if math.isclose(best[key], lo, rel_tol=1e-9):
        return f"  !! winner is at the LOW end of the {key} grid -- extend downward and re-run"
    if math.isclose(best[key], hi, rel_tol=1e-9):
        return f"  !! winner is at the HIGH end of the {key} grid -- extend upward and re-run"
    return ""


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------


def stage_aux(args) -> Dict:
    """Is the optimal auxiliary rate method-independent?

    The auxiliary group is AdamW on the same parameters for every method, so its
    optimum should not depend on the matrix rule. Verifying that on two anchor
    methods spanning an order of magnitude in ``eta_0`` earns the right to fix one
    ``lr_aux`` globally -- instead of a 2-D grid per method (10x the cost) or an
    unverified assertion.
    """
    anchors = args.aux_anchors or ["signmuon", "muon"]
    out: Dict[str, Dict] = {}
    for method in anchors:
        base = anchor_for(method, args.lr_scaling)
        lr_grid = round_grid(base, points=args.lr_points)
        print(f"\n=== aux stage: {method} (lr around {base:.4g}) ===")
        results = []
        for lr in lr_grid:
            for lr_aux in AUX_GRID:
                tag = f"aux_{method}_{args.lr_scaling}_lr{lr:.4g}_aux{lr_aux:.4g}"
                r = run_one(args, lr=lr, lr_aux=lr_aux, lr_scaling=args.lr_scaling,
                            method=method, epochs=args.epochs, tag=tag)
                if r:
                    results.append(r)
        best = best_of(results)
        if best is None:
            print(f"  no successful run for {method}")
            continue
        out[method] = {"best": best, "all": results}
        print(f"  best: lr={best['lr']:.4g}, lr_aux={best['lr_aux']:.4g}, "
              f"val {best['val_acc']:.2f}%")

    if len(out) >= 2:
        chosen = {m: d["best"]["lr_aux"] for m, d in out.items()}
        print("\n--- aux verdict ---")
        for m, v in chosen.items():
            print(f"  {m:<16} best lr_aux = {v:.4g}")
        if len(set(chosen.values())) == 1:
            v = next(iter(chosen.values()))
            print(f"  AGREE -> fix lr_aux = {v:.4g} for every method, and report that "
                  f"it was verified method-independent.")
        else:
            print("  DISAGREE -> lr_aux must be tuned per method; give every method "
                  "the same 2-D budget and say so.")
    return out


def stage_alpha(args) -> Dict:
    """Which per-layer scaling exponent for the sign family?

    Sweeps ``power:ALPHA`` jointly with ``eta_0`` on one representative sign-family
    method. ``alpha = 0`` is a global learning rate, ``1/2`` is the unit-gain rule,
    ``1`` is muP-with-alignment.

    Caveat worth reporting: ResNet-18 is a **weak instrument** for this. Thirteen
    of its twenty conv weight tensors have ``fan_in/fan_out = 9`` exactly and hold
    84.5% of all parameters -- and one shape alone, ``(512, 4608)`` appearing three
    times, is 63% of the model -- so ``alpha`` is only identified through the
    transition and 1x1-downsample layers. Confirm on a second architecture (or read
    the ``--log-gain`` diagnostic, which measures the exponent directly) before
    treating a small val-accuracy gap as decisive.
    """
    method = args.method or "signmuon"
    out: Dict[str, Dict] = {}
    for alpha in (args.alpha_grid or ALPHA_GRID):
        rule = f"power:{alpha:g}"
        boost = SCALED_ANCHOR_BOOST["unit-gain"] ** (2 * alpha)   # ~fan_in^alpha
        base = LEGACY_ANCHORS[method] * boost
        grid = round_grid(base, points=args.lr_points)
        print(f"\n=== alpha stage: {method}, alpha={alpha:g} (lr around {base:.4g}) ===")
        results = []
        for lr in grid:
            tag = f"alpha{alpha:g}_{method}_lr{lr:.4g}"
            r = run_one(args, lr=lr, lr_aux=args.lr_aux, lr_scaling=rule,
                        method=method, epochs=args.epochs, tag=tag)
            if r:
                results.append(r)
        best = best_of(results)
        if best is None:
            continue
        out[rule] = {"best": best, "all": results, "alpha": alpha}
        warn = boundary_warning(best, grid)
        print(f"  best: lr={best['lr']:.4g}, val {best['val_acc']:.2f}%")
        if warn:
            print(warn)

    if out:
        print("\n--- alpha verdict ---")
        ranked = sorted(out.items(), key=lambda kv: -kv[1]["best"]["val_acc"])
        for rule, d in ranked:
            print(f"  alpha={d['alpha']:<4g} val {d['best']['val_acc']:.2f}%  "
                  f"(at lr={d['best']['lr']:.4g})")
        top, second = ranked[0][1], ranked[1][1] if len(ranked) > 1 else None
        if second and abs(top["best"]["val_acc"] - second["best"]["val_acc"]) < 0.3:
            print("  gap < 0.3% -> NOT decisive on this architecture. Use the "
                  "--log-gain diagnostic, or a model with more shape diversity.")
        else:
            print(f"  -> alpha = {top['alpha']:g}")
    return out


def stage_lr(args) -> Dict:
    """Tune ``eta_0`` per method under a fixed scaling rule, equal budget for all."""
    methods = args.methods or ALL_METHODS
    out: Dict[str, Dict] = {}

    for method in methods:
        base = anchor_for(method, args.lr_scaling)
        # One sweep, not coarse-then-fine: the 1-2-5 lattice already *is* the
        # finest round resolution, so a refinement stage below 2x would have to
        # emit off-lattice values, and the neighbours of an interior winner in a
        # run of consecutive lattice points are measured by construction.
        grid = round_grid(base, points=args.lr_points)
        print(f"\n=== lr stage: {method} "
              f"({args.lr_points} pts around {base:.4g}) ===")
        results = [run_one(args, lr=lr, lr_aux=args.lr_aux, lr_scaling=args.lr_scaling,
                           method=method, epochs=args.epochs,
                           tag=f"lr_{method}_{args.lr_scaling}_{lr:.4g}")
                   for lr in grid]
        best = best_of(results)
        if best is None:
            print(f"  every run failed for {method}")
            continue
        warn = boundary_warning(best, grid)
        if warn:
            print(warn)
        out[method] = {"best": best, "all": [r for r in results if r],
                       "n_configs": len(grid), "boundary_warning": warn}
        print(f"  BEST: lr={best['lr']:.6g}, val {best['val_acc']:.2f}% "
              f"({len(grid)} configs)")

    if out:
        print(f"\n--- eta_0 per method (rule '{args.lr_scaling}', "
              f"{args.lr_points} configs each) ---")
        for method, d in out.items():
            flag = "  [BOUNDARY]" if d["boundary_warning"] else ""
            print(f"  {method:<16} eta_0 = {d['best']['lr']:<12.6g} "
                  f"val {d['best']['val_acc']:.2f}%{flag}")
        _check_family_agreement(out)
    return out


def _check_family_agreement(out: Dict[str, Dict]) -> None:
    """The scaling rule's falsifiable prediction.

    Once the shape dependence lives in ``lambda``, ``eta_0`` is shape-free, so the
    tuned ``eta_0`` should agree *within* each family. Reporting this is a result,
    not a protocol note -- and if it fails, that is a real finding about where the
    families differ.
    """
    groups: Dict[str, List[Tuple[str, float]]] = {}
    for method, d in out.items():
        cls = LMO_FAMILY.get(method)
        if cls is None:
            continue
        groups.setdefault(cls.family, []).append((method, d["best"]["lr"]))

    print("\n--- family agreement (the scaling rule's prediction) ---")
    for family, entries in groups.items():
        if len(entries) < 2:
            continue
        vals = [v for _, v in entries]
        spread = max(vals) / min(vals)
        names = ", ".join(f"{m}={v:.4g}" for m, v in entries)
        verdict = ("AGREE (within 2x, i.e. one fine-grid step)" if spread <= 2.0
                   else f"DISAGREE by {spread:.1f}x")
        print(f"  {family:<6} {names}")
        print(f"  {'':<6} spread {spread:.2f}x -> {verdict}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=["aux", "alpha", "lr"])
    p.add_argument("--methods", nargs="*", default=None,
                   help="lr stage: which methods (default: all ten)")
    p.add_argument("--method", type=str, default=None,
                   help="alpha stage: the representative sign-family method")
    p.add_argument("--aux-anchors", nargs="*", default=None,
                   help="aux stage: anchor methods (default: signmuon muon)")
    p.add_argument("--alpha-grid", nargs="*", type=float, default=None)
    p.add_argument("--lr-scaling", type=str, default="unit-gain",
                   help=", ".join(sorted(RULES)) + ", or power:ALPHA[,BETA]")

    p.add_argument("--dataset", type=str, default="cifar10")
    p.add_argument("--model", type=str, default="resnet18")
    p.add_argument("--epochs", type=int, default=75,
                   help="Training horizon. For --stage lr this must be the horizon "
                        "the paper reports (75): a shorter proxy anneals over a "
                        "different schedule and reordered the rates when it was "
                        "checked. --stage aux and --stage alpha compare two arms at "
                        "one shared horizon, where that bias cancels, so run those "
                        "with --epochs 15")
    p.add_argument("--lr-points", type=int, default=5,
                   help="Lattice points per method: 3 per decade, so 5 spans "
                        "~1.3 decades and 7 spans ~2")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr-aux", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="See centralized.main; 0 by default.")
    p.add_argument("--weight-decay-mode", type=str, default="decoupled",
                   choices=["decoupled", "coupled"],
                   help="See centralized.main; forwarded verbatim to every child run.")
    p.add_argument("--head-adamw", type=str, default="always",
                   choices=["auto", "always", "never"])
    p.add_argument("--last-k", type=int, default=5)
    p.add_argument("--val-seed", type=int, default=12345)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--data", type=str, default="./data")
    p.add_argument("--download", action="store_true", help="Download the dataset if missing")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--nondeterministic", action="store_true")
    p.add_argument("--out", type=str, default=None,
                   help="Write the stage results to this JSON (default: "
                        "results/tuning/<stage>_<rule>.json)")
    return p.parse_args()


def main() -> None:
    args = get_args()
    stage = {"aux": stage_aux, "alpha": stage_alpha, "lr": stage_lr}[args.stage]
    out = stage(args)

    path = Path(args.out) if args.out else (
        results_root() / "tuning" / f"{args.stage}_{args.lr_scaling.replace(':', '')}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"stage": args.stage, "lr_scaling": args.lr_scaling,
                   "epochs": args.epochs, "selection_metric": "val_acc (last-k mean)",
                   "results": out}, f, indent=2)
    print(f"\nWritten to {path}")
    print("Selection used validation accuracy only. (Test accuracy is logged "
          "each epoch but never enters the ranking.)")


if __name__ == "__main__":
    main()
