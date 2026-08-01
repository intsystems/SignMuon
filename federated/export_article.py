"""Pack the federated results tree into one downloadable archive.

    python3 -m federated.export_article                 # scan + write + zip
    python3 -m federated.export_article --no-archive    # leave the folder only
    python3 -m federated.export_article --phase final   # just the reported runs

This is the second half of the federated workflow. ``federated.overnight`` computes
on the GPU box and calls this at the end; you download the single
``results/federated_results.zip`` it prints, and

    python3 -m federated.plot_article --bundle results/federated_results.zip

unpacks it and redraws the figure. The run tree itself never has to leave the
machine: it is one ``model.pt`` of 2.9 MB per job, ~370 MB for a full night, and the
paper needs none of it.

What lands in the bundle
-----------------------

    SUMMARY.md            <- every table in one file; this is the one to read
    table_federated.csv   <- `tab:exp_3`, aggregated over seeds exactly as the paper
                             defines the columns. Quote from here, not by hand
    runs.csv              <- one row per run: config + derived summary metrics
    curves.csv            <- tidy per-round series, the input to every figure
    communication.csv     <- `tab:commacct`, recomputed from each run's own alphabet
    configs.json          <- the full config of every run, nothing dropped
    environment.json      <- every machine that contributed, with GPU / driver /
                             CUDA / Python / PyTorch / commit and its run count
    hardware.tex          <- the same as a LaTeX row for the reproducibility appendix
    MANIFEST.json         <- commit, argv, run counts, and what was left out
    runs/...              <- each run's `metrics.json`, minus the model weights

That last directory is the point of the layout: the plotting scripts read
``metrics.json`` trees, so ``--bundle`` is just ``--root <unpacked>/runs`` and no
figure code has to know that a bundle exists.

This is a **pure read**: nothing is written inside a run directory. It is stdlib
apart from one import of ``federated.algorithms.communication_bits``, which is
deliberate -- the bit accounting has exactly one implementation and this must not
become a second one. That import pulls in torch, so on a machine without it the
two communication files are skipped and everything else is still written.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from common.paths import repo_relative, results_root, scrub, scrub_text

# code/ -- this file is code/federated/export_article.py. Recomputed locally rather
# than imported from common.utils, which pulls in torch.
ROOT = Path(__file__).resolve().parents[1]

#: Config fields promoted to columns in ``runs.csv``. Anything not listed still
#: reaches ``configs.json`` verbatim; this is a readability choice, not a filter.
RUN_COLS = [
    "run_name", "seed", "phase", "algorithm", "variant", "scale_baselines",
    "lr_scaling", "split", "rounds",
    "n_parties", "n_steps", "batch_size", "lr", "lr_aux", "momentum",
    "weight_decay", "weight_decay_mode", "partition", "beta", "dataset", "model",
    "ns_steps", "lmo_dtype", "uplink_zeros", "mv_ties", "eval_freq", "last_k",
    "target_acc", "val_seed", "partition_seed",
]

#: Per-round series carried into ``curves.csv``. ``val_*`` and ``test_*`` both
#: appear because a tuning run records the former and a final run the latter.
CURVE_SERIES = [
    "train_loss", "test_loss", "test_acc", "val_loss", "val_acc",
    "round_seconds", "lr", "gain_spread", "mv_tie_frac", "uplink_zero_frac",
]


# --------------------------------------------------------------------------
# Phase attribution
# --------------------------------------------------------------------------


def phase_of(config: Dict[str, Any], primary_rule: Optional[str] = None) -> str:
    """Which overnight phase produced this run.

    Read off the config rather than the directory name: the federated driver
    encodes the phase in neither, so ``split`` and ``rounds`` are what distinguish
    a tuning run from a reported one. A run with weight decay on when the primary
    protocol sets it to zero is the ``wd`` ablation, and must be tested first or it
    would be reported as a headline result.

    ``primary_rule`` separates the ``rules`` ablation the same way. Its finals are
    full-50k runs at zero decay like any other, differing only in ``lr_scaling``, so
    without this they would land in the reported table -- which compares methods at
    one rule, not rules. ``scan`` infers it as the rule most of the finals used
    rather than taking it on trust.
    """
    # The driver's two timing runs are `--split tune` like any search job, so
    # without this they land in the tuning table as Muon at 0 and 40 rounds -- which
    # is what the 2026-07-30 bundle reported. `centralized.export_article` has
    # always labelled them; this is the federated twin.
    if "preflight" in str(config.get("run_name") or ""):
        return "preflight"
    if config.get("split") == "tune":
        return "lr"
    if float(config.get("weight_decay") or 0.0) > 0.0:
        return "wd"
    if primary_rule and str(config.get("lr_scaling") or "") != primary_rule:
        return "rules"
    return "final"


# --------------------------------------------------------------------------
# Series helpers -- the JSON keeps `None` for a round with no evaluation
# --------------------------------------------------------------------------


def _pairs(hist: Dict[str, Any], key: str) -> List[Tuple[int, float]]:
    steps = hist.get("steps") or []
    col = hist.get(key) or []
    return [(int(s), float(v)) for s, v in zip(steps, col) if v is not None]


def _values(hist: Dict[str, Any], key: str) -> List[float]:
    return [v for _, v in _pairs(hist, key)]


def _last(hist: Dict[str, Any], key: str) -> Optional[float]:
    vals = _values(hist, key)
    return vals[-1] if vals else None


def _tail_mean(hist: Dict[str, Any], key: str, k: int) -> Optional[float]:
    vals = _values(hist, key)
    if not vals:
        return None
    tail = vals[-max(1, k):]
    return sum(tail) / len(tail)


def _mean(hist: Dict[str, Any], key: str) -> Optional[float]:
    vals = _values(hist, key)
    return sum(vals) / len(vals) if vals else None


def _median(hist: Dict[str, Any], key: str) -> Optional[float]:
    vals = sorted(_values(hist, key))
    return vals[len(vals) // 2] if vals else None


def _steps_to(hist: Dict[str, Any], key: str, target: float) -> Optional[int]:
    for step, val in _pairs(hist, key):
        if val >= target:
            return step
    return None


def mean_std(values: Sequence[Optional[float]]) -> Tuple[Optional[float], Optional[float], int]:
    """Mean and *sample* std over seeds, and how many seeds contributed.

    One seed measures no dispersion, so the std is ``None`` rather than ``0.00`` --
    a printed zero reads as agreement between methods when it is the absence of a
    measurement.
    """
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return None, None, 0
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return mean, None, 1
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, math.sqrt(var), len(vals)


# --------------------------------------------------------------------------
# Scanning
# --------------------------------------------------------------------------


class Run:
    """One ``metrics.json``, with the derived numbers the paper quotes."""

    def __init__(self, path: Path, payload: Dict[str, Any]) -> None:
        self.path = path
        #: Where this run sits, as the bundle records it: rooted at ``results/``
        #: rather than at whatever home directory the GPU box used. Every write of
        #: a path into a shipped file goes through this, not through ``self.path``.
        self.rel = repo_relative(path.parent.as_posix())
        self.config: Dict[str, Any] = payload.get("config", {}) or {}
        self.history: Dict[str, Any] = payload.get("history", {}) or {}
        self.phase = phase_of(self.config)
        self.algorithm = str(self.config.get("algorithm") or "unknown")
        self.eval_name = "val" if self.config.get("split") == "tune" else "test"

    @property
    def variant(self) -> str:
        """The table row this run belongs to, which is not always the algorithm.

        ``--scale-baselines`` gives SGD and Adam the sign family's per-layer rule.
        That is an ablation, not the baseline the paper reports, and the two share
        an ``algorithm`` -- so keying a table row on the algorithm alone silently
        merges them and reports whichever the dict happened to yield. The rule and
        the client count matter for the same reason: a bundle can legitimately hold
        more than one of each.
        """
        parts = [self.algorithm]
        if self.config.get("scale_baselines"):
            parts.append("scaled")
        rule = str(self.config.get("lr_scaling") or "")
        if rule and rule != "unit-gain":
            parts.append(rule)
        return "+".join(parts)

    @property
    def seed(self) -> Optional[int]:
        seed = self.config.get("seed")
        if seed is not None:
            return int(seed)
        name = self.path.parent.name                    # ``seed3``
        return int(name[4:]) if name.startswith("seed") and name[4:].isdigit() else None

    @property
    def last_k(self) -> int:
        return int(self.config.get("last_k") or 5)

    @property
    def target_acc(self) -> float:
        return float(self.config.get("target_acc") or 80.0)

    @property
    def acc_key(self) -> str:
        return f"{self.eval_name}_acc"

    @property
    def n_rounds_recorded(self) -> int:
        return len(self.history.get("steps") or [])

    def summary(self) -> Dict[str, Any]:
        """The ``--- summary ---`` block of the run log, recomputed from the JSON."""
        acc = self.acc_key
        return {
            "acc_final": _last(self.history, acc),
            "acc_tail_mean": _tail_mean(self.history, acc, self.last_k),
            "rounds_to_target": _steps_to(self.history, acc, self.target_acc),
            "loss_final": _last(self.history, f"{self.eval_name}_loss"),
            "train_loss_final": _last(self.history, "train_loss"),
            "median_round_seconds": _median(self.history, "round_seconds"),
            "gain_spread_first": (_values(self.history, "gain_spread") or [None])[0],
            "gain_spread_final": _last(self.history, "gain_spread"),
            "mv_tie_frac_mean": _mean(self.history, "mv_tie_frac"),
            "uplink_zero_frac_mean": _mean(self.history, "uplink_zero_frac"),
            "rounds_recorded": self.n_rounds_recorded,
        }


def scan(root: Path) -> Tuple[List[Run], List[Tuple[Path, str]]]:
    """Every readable run under ``root``, plus what could not be read and why."""
    runs, bad = [], []
    for path in sorted(Path(root).rglob("metrics.json")):
        try:
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            bad.append((path, str(exc)))
            continue
        run = Run(path, payload)
        if not run.history.get("steps"):
            bad.append((path, "no recorded rounds"))
            continue
        runs.append(run)

    # Second pass: the reported protocol is whichever per-layer rule most of the
    # full-50k runs used, and anything else at that split is the rule ablation.
    # Inferred rather than assumed, so a tree tuned entirely under `legacy` still
    # reports a table instead of calling every run an ablation.
    tally: Dict[str, int] = {}
    for run in runs:
        if run.phase == "final":
            rule = str(run.config.get("lr_scaling") or "")
            tally[rule] = tally.get(rule, 0) + 1
    if len(tally) > 1:
        primary = max(tally, key=lambda k: (tally[k], k == "unit-gain"))
        for run in runs:
            run.phase = phase_of(run.config, primary)

    runs, stale = _drop_foreign_conventions(runs)
    return runs, bad + stale


def _sign_convention(config: Dict[str, Any]) -> Tuple[str, str]:
    """How this run mapped an exact zero, on the uplink and in the vote."""
    return (str(config.get("uplink_zeros") or "keep"),
            str(config.get("mv_ties") or "zero"))


def _drop_foreign_conventions(runs: List[Run]) -> Tuple[List[Run], List[Tuple[Path, str]]]:
    """Keep only the sign convention the reported runs were made under.

    A results tree accumulates: a night that re-runs a study leaves the previous
    night's jobs beside the new ones, and if the convention changed in between,
    the two are not comparable. They are also not distinguishable by name, so
    they land in the same tuning table under the same nominal configuration, at
    two different accuracies, which reads as a selection failure rather than as
    the two experiments it is. The convention the ``final`` runs used is the one
    the bundle reports; anything else is a leftover and is excluded here rather
    than listed beside a run it cannot be compared with.
    """
    tally: Dict[Tuple[str, str], int] = {}
    for run in runs:
        if run.phase == "final":
            key = _sign_convention(run.config)
            tally[key] = tally.get(key, 0) + 1
    if not tally:                    # no reported runs: nothing to be foreign to
        return runs, []
    primary = max(tally, key=lambda k: tally[k])

    kept, stale = [], []
    for run in runs:
        found = _sign_convention(run.config)
        if found == primary:
            kept.append(run)
        else:
            stale.append((run.path, f"sign convention uplink_zeros={found[0]}, "
                                    f"mv_ties={found[1]}; the reported runs used "
                                    f"{primary[0]}/{primary[1]}"))
    return kept, stale


# --------------------------------------------------------------------------
# The paper's table
# --------------------------------------------------------------------------


#: Row order of ``tab:exp_3``: the six placements, then the controls.
METHOD_ORDER = [
    "muon", "muonserver", "signmuon", "ef21signmuon", "muonusign",
    "ef21muonsign", "ef21muonusign", "signsgd", "muonsign", "sgd", "adam",
]


def _order_key(variant: str) -> Tuple[int, int, str]:
    """Paper order for the plain variants; ablations sort after their baseline."""
    base, _, suffix = variant.partition("+")
    try:
        return (METHOD_ORDER.index(base), bool(suffix), suffix)
    except ValueError:
        return (len(METHOD_ORDER), bool(suffix), variant)


def group_reported(runs: Sequence[Run]) -> Dict[str, List[Run]]:
    """``final``-phase runs by *variant*, one group per table row.

    Runs that differ in anything but the seed are a different experiment and are
    kept apart: if two client counts or two learning rates are present for one
    variant, the larger group wins and the smaller is reported as dropped rather
    than pooled into a meaningless mean. Grouping on the variant rather than the
    algorithm is what keeps `adam` and `adam+scaled` in separate rows -- they have
    five seeds each, so keying on the algorithm made the winner a dict-ordering
    accident, and the 2026-07-30 export reported the ablation as the baseline.
    """
    by_config: Dict[Tuple, List[Run]] = {}
    for run in runs:
        if run.phase != "final":
            continue
        key = (run.variant, run.config.get("lr"), run.config.get("rounds"),
               run.config.get("n_parties"), run.config.get("n_steps"),
               run.config.get("partition"),
               run.config.get("model"), run.config.get("dataset"))
        by_config.setdefault(key, []).append(run)

    best: Dict[str, List[Run]] = {}
    for key, group in by_config.items():
        variant = key[0]
        if variant not in best or len(group) > len(best[variant]):
            best[variant] = group
    return dict(sorted(best.items(), key=lambda kv: _order_key(kv[0])))


#: CNN2 on CIFAR-10, from ``common.models``. Used only for a run written before
#: the config recorded its own counts; pinned by the test suite.
_LEGACY_COUNTS = {("cifar10", "cnn2"): (762_560, 2_146, 3)}


def _param_counts(config: Dict[str, Any]) -> Optional[Tuple[int, int, int]]:
    """``(matrix, auxiliary, matrix layers)`` for a run, or ``None`` if unknown.

    Recorded in the config since 2026-07-30. Older runs fall back to the known
    counts for their (dataset, model) pair, and a pair with no entry returns
    ``None`` rather than a guess -- an invented parameter count would produce a
    communication table that looks authoritative and is not.
    """
    n_mat = int(config.get("n_matrix_params") or 0)
    n_aux = int(config.get("n_aux_params") or 0)
    n_lay = int(config.get("n_matrix_layers") or 0)
    if n_mat and n_lay:
        return n_mat, n_aux, n_lay
    return _LEGACY_COUNTS.get((str(config.get("dataset")), str(config.get("model"))))


def table_rows(runs: Sequence[Run]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """One row per method: accuracy, rounds-to-target, and the communication cost.

    Returns the rows and any notes worth printing -- a missing torch or an unknown
    parameter count leaves the communication columns empty rather than failing, so
    the accuracy table still lands.
    """
    notes: List[str] = []
    try:
        from federated.algorithms import communication_bits
    except ModuleNotFoundError as exc:                       # torch not installed
        communication_bits = None
        notes.append(f"communication columns skipped: {exc.name} is not installed")

    rows = []
    for variant, group in group_reported(runs).items():
        acc_mean, acc_std, n_seeds = mean_std([r.summary()["acc_tail_mean"] for r in group])
        hit = [v for v in (r.summary()["rounds_to_target"] for r in group) if v is not None]
        rnd_mean, _, _ = mean_std(hit)
        ref = group[0]
        comm: Dict[str, Any] = {}
        counts = _param_counts(ref.config)
        if communication_bits is not None and counts is not None:
            n_mat, n_aux, n_lay = counts
            comm = communication_bits(
                ref.algorithm, n_mat, n_aux,
                ref.summary()["uplink_zero_frac_mean"] or 0.0,
                n_layers=n_lay,
                uplink_zeros=str(ref.config.get("uplink_zeros") or "random"),
            )
        elif counts is None:
            notes.append(f"{variant}: no parameter counts recorded for "
                         f"{ref.config.get('dataset')}/{ref.config.get('model')}; "
                         f"communication columns left empty")
        rows.append({
            "algorithm": variant,
            "lr": ref.config.get("lr"),
            "seeds": n_seeds,
            "acc_mean": acc_mean,
            "acc_std": acc_std,
            "rounds_to_target": rnd_mean,
            "seeds_reaching_target": len(hit),
            "target_acc": ref.target_acc,
            "uplink_bits_per_param": comm.get("uplink_bits_per_param"),
            "downlink_bits_per_param": comm.get("downlink_bits_per_param"),
            "round_trip_reduction": comm.get("round_trip_reduction"),
            "gain_spread_final": ref.summary()["gain_spread_final"],
        })
    return rows, sorted(set(notes))


# --------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------


def _r(val: Optional[float], nd: int = 4) -> Optional[float]:
    return None if val is None else round(float(val), nd)


def write_runs(runs: Sequence[Run], out: Path) -> Path:
    path = out / "runs.csv"
    summary_cols = list(runs[0].summary().keys()) if runs else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path"] + RUN_COLS + summary_cols)
        for run in sorted(runs, key=lambda r: (_order_key(r.variant), r.phase,
                                               r.seed if r.seed is not None else -1)):
            cfg = dict(run.config)
            cfg["phase"] = run.phase
            cfg["seed"] = run.seed
            cfg["variant"] = run.variant
            summ = run.summary()
            w.writerow([run.rel]
                       + [cfg.get(c) for c in RUN_COLS]
                       + [_r(summ[c]) if isinstance(summ[c], float) else summ[c]
                          for c in summary_cols])
    return path


def write_curves(runs: Sequence[Run], out: Path) -> Path:
    path = out / "curves.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run_name", "algorithm", "phase", "seed", "lr", "round"]
                   + CURVE_SERIES)
        for run in sorted(runs, key=lambda r: (_order_key(r.variant), r.phase)):
            steps = run.history.get("steps") or []
            cols = {k: (run.history.get(k) or []) for k in CURVE_SERIES}
            for i, step in enumerate(steps):
                w.writerow([run.config.get("run_name"), run.variant, run.phase,
                            run.seed, run.config.get("lr"), step]
                           + [_r(cols[k][i]) if i < len(cols[k]) else None
                              for k in CURVE_SERIES])
    return path


def write_table(rows: Sequence[Dict[str, Any]], out: Path) -> Path:
    path = out / "table_federated.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["algorithm"])
        w.writeheader()
        for row in rows:
            w.writerow({k: (_r(v) if isinstance(v, float) else v) for k, v in row.items()})
    return path


def write_communication(rows: Sequence[Dict[str, Any]], out: Path) -> Path:
    """``tab:commacct``, from each run's own alphabet rather than an assumed one."""
    path = out / "communication.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "uplink_bits_per_param", "downlink_bits_per_param",
                    "round_trip_reduction"])
        for row in rows:
            w.writerow([row["algorithm"], _r(row["uplink_bits_per_param"], 4),
                        _r(row["downlink_bits_per_param"], 4),
                        _r(row["round_trip_reduction"], 2)])
    return path


def rule_rows(runs: Sequence[Run]) -> List[Dict[str, Any]]:
    """The per-layer-rule ablation: one row per (method, rule), seeds pooled.

    The reported rule's rows come from the ``final`` phase and the alternatives from
    ``rules``, so the comparison is against the same runs the paper's table quotes
    rather than against a re-run of them.
    """
    by: Dict[Tuple[str, str], List[Run]] = {}
    for run in runs:
        if run.phase not in ("final", "rules") or run.config.get("scale_baselines"):
            continue
        rule = str(run.config.get("lr_scaling") or "")
        by.setdefault((run.algorithm, rule), []).append(run)
    if len({rule for _, rule in by}) < 2:
        return []

    # Only the methods the ablation actually re-tuned. Carrying the other eight
    # along at the reported rule alone would put an eleven-method ordering beside
    # three-method ones and make every comparison of the two report a difference.
    covered = {alg for alg, _ in by}
    n_rules = len({rule for _, rule in by})
    covered = {alg for alg in covered
               if sum(1 for a, _ in by if a == alg) == n_rules}
    by = {k: v for k, v in by.items() if k[0] in covered}
    if not by:
        return []

    rows = []
    for (alg, rule), group in by.items():
        mean, std, n = mean_std([r.summary()["acc_tail_mean"] for r in group])
        rows.append({"algorithm": alg, "rule": rule, "phase": group[0].phase,
                     "lr": group[0].config.get("lr"), "seeds": n,
                     "acc_mean": mean, "acc_std": std})
    rows.sort(key=lambda r: (_order_key(r["algorithm"]), r["phase"] != "final", r["rule"]))
    return rows


def write_rules(rows: Sequence[Dict[str, Any]], out: Path) -> Path:
    path = out / "rule_ablation.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow({k: (_r(v) if isinstance(v, float) else v) for k, v in row.items()})
    return path


def write_configs(runs: Sequence[Run], out: Path) -> Path:
    path = out / "configs.json"
    # Keys are run directories and the values hold whatever the CLI was given, so
    # both can carry the absolute path -- hence the username -- of the box that
    # ran them. `scrub` cuts those back to `results/...`; see `common/paths.py`.
    payload = scrub({run.rel: run.config for run in runs})
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def write_environment(runs: Sequence[Run], out: Path) -> List[Path]:
    """Every machine that contributed, and how many runs each produced."""
    machines: Dict[str, Dict[str, Any]] = {}
    for run in runs:
        hw = run.config.get("hardware")
        if not isinstance(hw, dict):
            continue
        key = json.dumps({k: hw.get(k) for k in sorted(hw)}, sort_keys=True, default=str)
        entry = machines.setdefault(key, {"hardware": hw, "runs": 0, "algorithms": set()})
        entry["runs"] += 1
        entry["algorithms"].add(run.algorithm)

    listing = [{"hardware": e["hardware"], "runs": e["runs"],
                "algorithms": sorted(e["algorithms"])} for e in machines.values()]
    env = out / "environment.json"
    env.write_text(json.dumps(listing, indent=2, default=str), encoding="utf-8")

    lines = ["% Federated runs, one row per machine. Generated by",
             "% federated.export_article -- do not edit by hand.",
             r"\begin{tabular}{@{}llr@{}}", r"\toprule",
             r"GPU & PyTorch / CUDA & Runs \\", r"\midrule"]
    for item in listing:
        hw = item["hardware"]
        gpu = str(hw.get("gpu_name") or hw.get("gpu") or "CPU")
        stack = f"{hw.get('torch_version', '?')} / {hw.get('cuda_version', '-')}"
        lines.append(f"{gpu} & {stack} & {item['runs']} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex = out / "hardware.tex"
    tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [env, tex]


def copy_metrics(runs: Sequence[Run], out: Path, root: Path) -> int:
    """Copy each ``metrics.json`` into ``<out>/runs/``, keeping the tree shape.

    The model weights are what make the tree too big to move, and the figures do
    not need them. Keeping the directory layout is what lets ``--bundle`` be a
    plain ``--root`` on the unpacked copy.
    """
    dest_root = out / "runs"
    for run in runs:
        try:
            rel = run.path.parent.relative_to(root)
        except ValueError:
            rel = Path(run.path.parent.name)
        dest = dest_root / rel
        dest.mkdir(parents=True, exist_ok=True)
        # Rewritten, not copied: a run's own config records `--data`, which is
        # absolute whenever it was given as an absolute path, and `configs.json`
        # being scrubbed does not help a reader who opens the metrics file itself.
        (dest / "metrics.json").write_text(
            scrub_text(run.path.read_text(encoding="utf-8")), encoding="utf-8")
    return len(runs)


def copy_overnight(out: Path, driver_root: Path) -> List[str]:
    """Bring the driver's own record along: what it planned, ran and gave up."""
    if not driver_root.is_dir():
        return []
    dest = out / "overnight"
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in ("REPORT.md", "state.json", "schedule.txt"):
        src = driver_root / name
        if src.is_file():
            # Not a straight copy: `state.json` stores an absolute log and metrics
            # path per job, which begins with a home directory (`common/paths.py`).
            (dest / name).write_text(repo_relative(src.read_text(encoding="utf-8")),
                                     encoding="utf-8")
            copied.append(name)
    return copied


def git_commit() -> Dict[str, Any]:
    def run(*args: str) -> Optional[str]:
        try:
            return subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                                  timeout=10, check=True).stdout.strip()
        except Exception:                                    # noqa: BLE001
            return None
    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain"))}


# --------------------------------------------------------------------------
# SUMMARY.md
# --------------------------------------------------------------------------


def _fmt(val: Optional[float], spec: str = ".2f") -> str:
    return "--" if val is None else format(val, spec)


def _rules_section(rules: Sequence[Dict[str, Any]]) -> List[str]:
    if not rules:
        return []
    lines = ["", "## Per-layer rule ablation", "",
             "Each (method, rule) pair is tuned from scratch under that rule, so a",
             "rule is not penalized for prescribing a different `eta_0` -- it should.",
             "What would signify is the *ordering* of the methods changing.",
             "",
             "| method | rule | eta_0 | seeds | acc (%) |",
             "| :--- | :--- | ---: | ---: | ---: |"]
    for r in rules:
        std = "" if r["acc_std"] is None else f" +/- {r['acc_std']:.2f}"
        mark = "**" if r["phase"] == "final" else ""
        lines.append(f"| {r['algorithm']} | {mark}{r['rule']}{mark} | {r['lr']} | "
                     f"{r['seeds']} | {_fmt(r['acc_mean'])}{std} |")
    order = {}
    for r in rules:
        order.setdefault(r["rule"], []).append((r["acc_mean"], r["algorithm"]))
    ranked = {k: [a for _, a in sorted(v, reverse=True)] for k, v in order.items()
              if all(m is not None for m, _ in v)}
    if len(ranked) > 1:
        same = len({tuple(v) for v in ranked.values()}) == 1
        lines += ["", f"Ordering under each rule: "
                      + "; ".join(f"`{k}` {' > '.join(v)}" for k, v in ranked.items()),
                  "", ("**The ordering is the same under every rule.**" if same else
                       "**The ordering is NOT the same under every rule** -- the "
                       "sign-family comparison depends on the multiplier, and the "
                       "paper has to say so.")]
    return lines


def summary_markdown(runs: Sequence[Run], rows: Sequence[Dict[str, Any]],
                     dropped: Sequence[Tuple[Path, str]],
                     rules: Sequence[Dict[str, Any]] = ()) -> str:
    by_phase: Dict[str, int] = {}
    for run in runs:
        by_phase[run.phase] = by_phase.get(run.phase, 0) + 1

    target = rows[0]["target_acc"] if rows else 80.0
    lines = [
        "# Federated results",
        "",
        f"{len(runs)} runs: " + ", ".join(f"{n} {p}" for p, n in sorted(by_phase.items()))
        + ".",
        "",
        "## The reported table (`tab:exp_3`)",
        "",
        "Mean +/- sample std over seeds of the per-seed tail mean. A blank std means",
        "one seed, which measures no dispersion -- do not read it as agreement.",
        "",
        f"| method | eta_0 | seeds | acc (%) | rounds to {target:g}% | up (bits) | down (bits) | round trip |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        std = "" if row["acc_std"] is None else f" +/- {row['acc_std']:.2f}"
        note = ("" if row["seeds_reaching_target"] == row["seeds"]
                else f" ({row['seeds_reaching_target']}/{row['seeds']})")
        lines.append(
            f"| {row['algorithm']} | {row['lr']} | {row['seeds']} | "
            f"{_fmt(row['acc_mean'])}{std} | "
            f"{_fmt(row['rounds_to_target'], '.0f')}{note} | "
            f"{_fmt(row['uplink_bits_per_param'])} | "
            f"{_fmt(row['downlink_bits_per_param'])} | "
            f"{_fmt(row['round_trip_reduction'], '.1f')}x |")

    lines += [
        "",
        "`round trip` is the reduction against full precision with the uncompressed",
        "auxiliary group counted in, computed from each run's own sign alphabet. It is",
        "the number to quote, not the uplink-only one.",
    ]
    lines += _rules_section(rules)
    lines += [
        "",
        "## Diagnostics",
        "",
        "`gain spread` is max/min over layers of the realized `||lambda*s||_F/sqrt(fan_out)`;",
        "the per-layer rule targets 1.00x, and the sign family is 1.00x by construction.",
        "`MV ties` and `uplink zeros` are RAW rates, counted before the randomized-pm1",
        "mapping, so they are diagnostics and feed no accounting.",
        "",
        "| method | seed | gain spread (round 1 -> final) | MV ties | uplink zeros |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for run in sorted((r for r in runs if r.phase == "final"),
                      key=lambda r: (_order_key(r.variant),
                                     r.seed if r.seed is not None else -1)):
        s = run.summary()
        lines.append(
            f"| {run.variant} | {run.seed} | "
            f"{_fmt(s['gain_spread_first'])}x -> {_fmt(s['gain_spread_final'])}x | "
            f"{_fmt(s['mv_tie_frac_mean'], '.4f')} | "
            f"{_fmt(s['uplink_zero_frac_mean'], '.4f')} |")

    tuned = sorted((r for r in runs if r.phase == "lr"),
                   key=lambda r: (_order_key(r.variant), r.config.get("lr") or 0))
    if tuned:
        lines += ["", "## The learning-rate search", "",
                  "Selected on validation accuracy only; no tuning run scores a test image.",
                  "", "| method | eta_0 | val acc (%) | rounds |",
                  "| :--- | ---: | ---: | ---: |"]
        for run in tuned:
            s = run.summary()
            lines.append(f"| {run.variant} | {run.config.get('lr')} | "
                         f"{_fmt(s['acc_tail_mean'])} | {run.config.get('rounds')} |")

    if dropped:
        lines += ["", "## Not included", ""]
        lines += [f"* `{repo_relative(p.parent.as_posix())}` -- {why}"
                  for p, why in dropped]

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Archive
# --------------------------------------------------------------------------


def make_archive(out: Path) -> Optional[Path]:
    """Zip the bundle directory into one file to copy off a remote GPU box.

    ``.zip`` rather than ``.tar.gz`` so that a double-click opens it on any of the
    three platforms this project is used from, and named after the directory the
    same way ``synthetic.run_gpu`` names its own.
    """
    archive = out.parent / f"{out.name}_results.zip"
    files = sorted(p for p in out.rglob("*") if p.is_file() and p != archive)
    if not files:
        return None
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, path.relative_to(out.parent))
    return archive


def open_bundle(path: Path, unpack_to: Optional[Path] = None) -> Path:
    """Resolve a bundle argument to a directory, unpacking a ``.zip`` if needed.

    Accepts the ``.zip`` itself, the directory it unpacks to, or the ``runs/``
    subdirectory, so that ``--bundle`` does the obvious thing with whatever the
    user happens to have on hand. Shared with the plotting scripts.
    """
    path = Path(path)
    if path.is_dir():
        return path
    if path.suffix == ".zip" and path.is_file():
        dest = Path(unpack_to) if unpack_to else path.parent
        with zipfile.ZipFile(path) as zf:
            zf.extractall(dest)
            names = [n for n in zf.namelist() if n.strip("/")]
        # The archive stores paths relative to the bundle's parent, so every entry
        # shares one top-level directory; return it rather than the extraction root.
        tops = {n.split("/")[0] for n in names}
        if len(tops) == 1:
            return dest / tops.pop()
        return dest
    raise SystemExit(f"Not a bundle: {path}\n"
                     f"Pass the .zip written by 'python3 -m federated.export_article', "
                     f"or the directory it unpacks to.")


def runs_root(bundle: Path) -> Path:
    """Where the ``metrics.json`` tree lives inside a bundle."""
    if (bundle / "runs").is_dir():
        return bundle / "runs"
    return bundle


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=None,
                   help="Results tree to scan (default: results/federated)")
    p.add_argument("--out", type=Path, default=None,
                   help="Bundle directory to write (default: results/federated_export)")
    p.add_argument("--overnight", type=Path, default=None,
                   help="Driver state to copy in (default: results/federated_overnight)")
    p.add_argument("--phase", nargs="*", default=None,
                   choices=["lr", "final", "wd"],
                   help="Export only these phases (default: all)")
    p.add_argument("--no-archive", action="store_true",
                   help="Write the bundle directory but do not zip it")
    args = p.parse_args(argv)

    # Resolved without importing torch, so this runs on a login node.
    results = results_root()
    root = args.root or results / "federated"
    out = args.out or results / "federated_export"
    driver = args.overnight or results / "federated_overnight"

    if not root.is_dir():
        print(f"No results at {root.resolve()}.\n"
              f"Run 'python3 -m federated.overnight' first, or pass --root.")
        return 1

    runs, bad = scan(root)
    if args.phase:
        keep = set(args.phase)
        bad += [(r.path, f"phase {r.phase} not requested")
                for r in runs if r.phase not in keep]
        runs = [r for r in runs if r.phase in keep]
    if not runs:
        print(f"No usable runs under {root.resolve()}.")
        return 1

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    rows, notes = table_rows(runs)
    rules = rule_rows(runs)
    written = [write_runs(runs, out), write_curves(runs, out), write_configs(runs, out)]
    if rows:
        written += [write_table(rows, out), write_communication(rows, out)]
    if rules:
        written.append(write_rules(rules, out))
    written += write_environment(runs, out)
    n_metrics = copy_metrics(runs, out, root)
    driver_files = copy_overnight(out, driver)

    (out / "SUMMARY.md").write_text(summary_markdown(runs, rows, bad, rules),
                                    encoding="utf-8")

    manifest = {
        "generated_by": "federated.export_article",
        # argv carries whatever --root/--out were given, which on the GPU box are
        # absolute and rooted at a home directory.
        "argv": [repo_relative(a) for a in [Path(sys.argv[0]).name] + list(sys.argv[1:])],
        "source_tree": repo_relative(str(root.resolve())),
        "git": git_commit(),
        "runs": len(runs),
        "runs_by_phase": {ph: sum(1 for r in runs if r.phase == ph)
                          for ph in sorted({r.phase for r in runs})},
        "metrics_copied": n_metrics,
        "overnight_files": driver_files,
        "excluded": [{"path": repo_relative(p.parent.as_posix()), "reason": why}
                     for p, why in bad],
        "notes": notes,
        "files": sorted(pth.name for pth in written) + ["SUMMARY.md", "runs/"],
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nBundle  {out.resolve()}")
    print(f"Read    {(out / 'SUMMARY.md').resolve()}")
    if manifest["git"]["dirty"]:
        print("NOTE: the working tree has uncommitted changes; MANIFEST.json says so.")
    for note in notes:
        print(f"  ! {note}")
    for path, why in bad:
        print(f"  ! excluded {path.parent.as_posix()}: {why}")

    if args.no_archive:
        return 0
    archive = make_archive(out)
    if archive is None:
        print("Nothing to archive.")
        return 1
    print(f"\nDownload {archive.resolve()} "
          f"({archive.stat().st_size / 1024:.0f} KB)")
    print("In VS Code, right-click it under code/results/ and choose Download. Then:")
    print(f"  python3 -m federated.plot_article --bundle {archive.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
