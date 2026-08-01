"""Aggregate multi-seed runs into mean +/- std curves and a summary table.

    python3 -m aggregate                             # every run under results/
    python3 -m aggregate --root results/federated     # one family
    python3 -m aggregate --metric test_acc --csv summary.csv

Runs are grouped by their *configuration minus the seed* (and minus fields that
cannot affect the result, such as the device or the data directory), so a sweep
of ``--seed 0 1 2 3 4`` collapses into one row. Curves are averaged pointwise on
the intersection of the recorded step indices, which is why
``utils.History`` stores the x-axis explicitly: without it, runs with different
``eval_freq`` (or forward-filled un-evaluated rounds) would be silently
misaligned.

Nothing here writes into a run directory; aggregation is a pure read.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from common.paths import results_root

# Config fields that do not affect the trajectory (or that identify the run
# rather than the experiment) and are therefore ignored when grouping.
#
# ``hardware`` is here because ``save_run`` stamps the machine into every config:
# without it, seeds run on two different GPUs land in two groups of one instead of
# one group of two, and every tool downstream reports "1 seed" and prints a blank
# std. It does perturb the trajectory at the last bit, which is the point of
# reporting the seed spread rather than a bitwise trajectory.
IGNORED_FIELDS = {"seed", "run_name", "device", "data", "datadir", "download",
                  "hardware"}


def group_key(config: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    """Hashable identity of an experiment: the config minus per-run fields."""
    return tuple(sorted((k, _hashable(v)) for k, v in config.items()
                        if k not in IGNORED_FIELDS))


def _hashable(v: Any) -> Any:
    if isinstance(v, (list, tuple)):
        return tuple(_hashable(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _hashable(x)) for k, x in v.items()))
    return v


def describe(config: Dict[str, Any]) -> str:
    """Short human label for a group."""
    parts = []
    for key in ("dataset", "model", "algorithm", "optimizer", "n_parties",
                "n_steps", "rounds", "epochs", "lr", "lr_aux", "momentum"):
        if key in config and config[key] is not None:
            parts.append(f"{key}={config[key]}")
    return " ".join(parts)


def unique_labels(groups: Dict[Tuple, List[Dict[str, Any]]]) -> Dict[Tuple, str]:
    """Map each group to a label that no other group shares.

    ``describe`` prints a fixed shortlist of config keys, so two groups that
    differ only in, say, ``lr_scaling`` collapse to the same string -- which
    silently overwrote curves when they were stored in a dict. Any label claimed
    by more than one group is extended with exactly the fields that tell those
    groups apart, so the suffix stays empty in the common case.
    """
    base: Dict[Tuple, str] = {k: describe(g[0]["config"]) for k, g in groups.items()}

    clashes: Dict[str, List[Tuple]] = {}
    for key, label in base.items():
        clashes.setdefault(label, []).append(key)

    labels: Dict[Tuple, str] = {}
    for label, keys in clashes.items():
        if len(keys) == 1:
            labels[keys[0]] = label
            continue
        configs = [groups[k][0]["config"] for k in keys]
        differing = sorted(
            field for field in set().union(*(c.keys() for c in configs))
            if field not in IGNORED_FIELDS
            and len({_hashable(c.get(field)) for c in configs}) > 1
        )
        for key, config in zip(keys, configs):
            suffix = " ".join(f"{f}={config.get(f)}" for f in differing)
            labels[key] = f"{label} {suffix}" if suffix else label

    return labels


def load_runs(roots: Sequence[Path]) -> List[Dict[str, Any]]:
    """Read every ``metrics.json`` under ``roots``."""
    runs = []
    for root in roots:
        for path in sorted(Path(root).rglob("metrics.json")):
            try:
                with open(path, encoding="utf-8") as f:
                    payload = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                print(f"  ! skipping {path}: {exc}")
                continue
            config = payload.get("config", {})
            history = payload.get("history", {})
            if "steps" not in history:
                # Pre-refactor format: one entry per round, no x-axis. Recover the
                # x-axis as 0..n-1, which is correct only when eval_freq == 1.
                length = max((len(v) for v in history.values() if isinstance(v, list)),
                             default=0)
                history = {"steps": list(range(length)), **history}
                print(f"  ~ {path}: legacy history without 'steps'; assuming eval_freq=1")
            runs.append({"path": path, "config": config, "history": history})
    return runs


def _mean_std(values: Iterable[float]) -> Tuple[float, float, int]:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = sum(vals) / n
    if n == 1:
        return mean, 0.0, 1
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)      # sample std
    return mean, math.sqrt(var), n


def aggregate_group(runs: Sequence[Dict[str, Any]], metric: str) -> Optional[Dict[str, Any]]:
    """Pointwise mean/std of ``metric`` over the runs' common step indices."""
    series = []
    for run in runs:
        hist = run["history"]
        if metric not in hist:
            continue
        series.append(dict(zip(hist["steps"], hist[metric])))
    if not series:
        return None

    common = set(series[0])
    for s in series[1:]:
        common &= set(s)
    steps = sorted(common)
    if not steps:
        return None

    means, stds, counts = [], [], []
    for step in steps:
        mean, std, n = _mean_std(s[step] for s in series)
        means.append(mean)
        stds.append(std)
        counts.append(n)

    return {"steps": steps, "mean": means, "std": stds, "n": counts[0] if counts else 0,
            "n_runs": len(series)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=str, nargs="*", default=None,
                   help="Directories to scan (default: the whole results/ tree)")
    p.add_argument("--metric", type=str, default="test_acc",
                   help="Metric to summarize (default: test_acc)")
    p.add_argument("--csv", type=str, default=None, help="Write the summary table here")
    p.add_argument("--curves", type=str, default=None,
                   help="Write the full mean/std curves to this JSON file")
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    # Default: the whole results/ tree, plus the pre-reorganization directories so
    # that runs made before the code was restructured are still picked up. The
    # results tree is resolved through `common.paths`, so a sweep redirected with
    # SIGNMUON_RESULTS is aggregated from where it actually wrote.
    roots = [Path(r) for r in args.root] if args.root else [
        results_root(), here / "saves", here / "saves_federated"]
    roots = [r for r in roots if r.exists()]
    if not roots:
        print("No saves directories found.")
        return

    runs = load_runs(roots)
    print(f"Found {len(runs)} run(s) under {', '.join(str(r) for r in roots)}\n")

    groups: Dict[Tuple, List[Dict[str, Any]]] = {}
    for run in runs:
        groups.setdefault(group_key(run["config"]), []).append(run)

    labels = unique_labels(groups)
    n_extended = sum(1 for k, g in groups.items()
                     if labels[k] != describe(g[0]["config"]))
    if n_extended:
        print(f"Note: {n_extended} group(s) shared a label with another group; "
              f"the distinguishing config fields were appended.\n")

    rows, curves = [], {}
    for key, group in sorted(groups.items(), key=lambda kv: labels[kv[0]]):
        label = labels[key]
        agg = aggregate_group(group, args.metric)
        if agg is None:
            print(f"  ! {label}: metric {args.metric!r} absent")
            continue
        # A pre-refactor ``metrics.json`` has no ``seed`` field at all, so None
        # is a real value here and must not reach sorted() next to ints.
        raw = [r["config"].get("seed") for r in group]
        known = sorted(s for s in raw if s is not None)
        n_unknown = len(raw) - len(known)
        seed_label = ",".join(str(s) for s in known)
        if n_unknown:
            seed_label = (seed_label + "," if seed_label else "") + f"{n_unknown}x?"
        final_mean, final_std = agg["mean"][-1], agg["std"][-1]
        rows.append({
            "label": label,
            "seeds": seed_label,
            # What the std is actually over: runs that carried the metric. Two
            # runs at the same seed are two runs but ONE seed, and a run missing
            # the metric contributes to neither.
            "n_runs": agg["n_runs"],
            "n_distinct_seeds": len(set(known)) + n_unknown,
            "n_runs_found": len(group),
            f"final_{args.metric}_mean": final_mean,
            f"final_{args.metric}_std": final_std,
            "final_step": agg["steps"][-1],
        })
        curves[label] = agg

    if not rows:
        print("Nothing to aggregate.")
        return

    width = max(len(r["label"]) for r in rows)
    print(f"{'experiment':<{width}}  seeds  {args.metric + ' (mean +/- std)':>26}")
    print("-" * (width + 36))
    for r in sorted(rows, key=lambda r: -r[f"final_{args.metric}_mean"]):
        mean, std = r[f"final_{args.metric}_mean"], r[f"final_{args.metric}_std"]
        disp = "  n/a   " if r["n_runs"] < 2 else f"{std:<8.4f}"
        print(f"{r['label']:<{width}}  {r['n_runs']:>5}  {mean:>14.4f} +/- {disp}")

    if len(set(r["n_runs"] for r in rows)) > 1:
        print("\nNote: groups have different run counts; std values are not comparable.")
    if any(r["n_runs"] == 1 for r in rows):
        print("Note: single-run groups show 'n/a' -- no dispersion was measured. "
              "The CSV records std = 0 for them; do not read it as agreement.")
    dup = [r for r in rows if r["n_distinct_seeds"] < r["n_runs"]]
    if dup:
        print(f"Note: {len(dup)} group(s) contain repeated seeds "
              f"({', '.join(r['label'] for r in dup[:3])}"
              f"{', ...' if len(dup) > 3 else ''}); the std understates the "
              f"true seed-to-seed spread.")
    partial = [r for r in rows if r["n_runs"] < r["n_runs_found"]]
    if partial:
        print(f"Note: {len(partial)} group(s) had runs without the metric "
              f"{args.metric!r}; those runs are excluded from mean and std.")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSummary written to {args.csv}")

    if args.curves:
        with open(args.curves, "w", encoding="utf-8") as f:
            json.dump({"metric": args.metric, "curves": curves}, f, indent=2)
        print(f"Curves written to {args.curves}")


if __name__ == "__main__":
    main()
