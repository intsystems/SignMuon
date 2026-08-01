"""Pack the centralized results tree into one small archive to bring home.

    python3 -m centralized.export_article                  # scan + write + archive
    python3 -m centralized.export_article --no-archive     # leave the folder only
    python3 -m centralized.export_article --phase final    # just the reported runs

This is the **second of the two commands** in the centralized workflow: run
``centralized.overnight`` on the GPU box, then this, then download the single
``results/article_export.tar.gz`` it prints. Unpack it anywhere and
``centralized.plot_analysis --bundle <dir>`` redraws every figure from it -- the
run tree itself never has to leave the machine.

Why this exists
---------------
``results/centralized`` is one directory per run, each holding a ``metrics.json``
next to a ``model.pt`` of tens of megabytes (a 33-job ResNet-18 sweep is ~1.5 GB).
The article needs the former and none of the latter. This walks the tree, derives
every number the paper quotes, and writes:

    table_cifar.csv     the paper's table, aggregated over seeds exactly as it
                        defines the columns -- mean +/- sample std of per-seed
                        tail means. Quote from here, not by hand.
    runs.csv            one row per run: config + derived summary metrics
    curves.csv          tidy per-epoch series, for the figures
    gain.csv            the ``--log-gain`` series, per epoch
    gain_fits.csv       log-log slope of gain_median vs epoch (the alpha measurement)
    environment.json    every machine that contributed, with GPU / driver / CUDA /
                        Python / PyTorch / commit, and how many runs each produced
    hardware.tex        the same as a LaTeX table for the reproducibility appendix
    configs.json        the full config of every run, nothing dropped
    overnight/          whatever small files the overnight driver left behind

Everything is stdlib -- no torch, no numpy -- so it runs on a login node without
touching the training environment, and it is a **pure read**: nothing is written
inside a run directory, matching the convention ``aggregate.py`` documents.

The summary printed at the end is meant to be copy-pasteable, so the numbers can
be discussed before the bundle is moved.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from common.paths import repo_relative, results_root

# code/ -- this file is code/centralized/export_article.py.
ROOT = Path(__file__).resolve().parents[1]
#: Honours ``SIGNMUON_RESULTS``, like the driver that calls us. Resolved through
#: `common.paths` rather than `common.utils`, which pulls in torch.
RESULTS = results_root()

#: Config fields promoted to columns in ``runs.csv``. Anything not listed still
#: reaches ``configs.json`` verbatim -- this is a readability choice, not a filter.
RUN_COLS = [
    "run_name", "seed", "phase", "optimizer", "lr_scaling", "split", "epochs",
    "lr", "lr_aux", "momentum", "weight_decay", "weight_decay_mode",
    "head_adamw", "batch_size", "dataset", "model", "last_k", "val_seed",
]

CURVE_SERIES = ["train_loss", "train_acc", "test_loss", "test_acc",
                "val_loss", "val_acc", "epoch_seconds"]

GAIN_SERIES = ["gain_min", "gain_median", "gain_max"]


# --------------------------------------------------------------------------
# Phase attribution
# --------------------------------------------------------------------------

def phase_of(run_name: str) -> str:
    """Which overnight phase produced this run, read off its directory name.

    ``tune.run_one`` builds the name as ``{tune|final}_{tag}_e{epochs}_{split[0]}``
    where the tag carries the phase prefix the driver chose. Order matters below:
    ``final_wd_*`` must be tested before the bare ``final_`` fallback, or the
    weight-decay ablation would be reported as a primary result.
    """
    stem = run_name
    for prefix in ("tune_", "final_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    # ``verify`` and ``alpha`` are retired phases, kept here so that an export of a
    # tree written before they were removed still labels its runs rather than
    # dropping them into ``other``. The 2026-07-27 tree is exactly that, and it is
    # the provenance of the submitted table.
    for marker, phase in (("wd_", "wd"), ("gain_", "gain"), ("aux_", "aux"),
                          ("alpha", "alpha"), ("lr_", "lr"),
                          ("verify_", "verify"), ("preflight", "preflight")):
        if stem.startswith(marker):
            return phase
    return "final" if run_name.startswith("final_") else "other"


# --------------------------------------------------------------------------
# History helpers -- the History class lives behind a torch import, so the few
# accessors needed here are reimplemented against the raw JSON.
# --------------------------------------------------------------------------

def _pairs(hist: Dict[str, Any], key: str) -> List[Tuple[int, float]]:
    """``(step, value)`` for the recorded (non-null) points of one series.

    Series are null-padded rather than forward-filled -- epoch 0 has no
    ``epoch_seconds``, a ``full``-split run has no ``val_acc`` -- so the nulls must
    be dropped rather than read as zeros.
    """
    steps = hist.get("steps") or []
    col = hist.get(key) or []
    return [(int(s), float(v)) for s, v in zip(steps, col)
            if v is not None and isinstance(v, (int, float))]


def _last(hist: Dict[str, Any], key: str) -> Optional[float]:
    pts = _pairs(hist, key)
    return pts[-1][1] if pts else None


def _tail_mean(hist: Dict[str, Any], key: str, k: int) -> Optional[float]:
    """Mean of the final ``k`` recorded values -- the paper's primary metric."""
    vals = [v for _, v in _pairs(hist, key)]
    if not vals:
        return None
    tail = vals[-max(1, k):]
    return sum(tail) / len(tail)


def _best(hist: Dict[str, Any], key: str, mode: str = "max") -> Optional[float]:
    vals = [v for _, v in _pairs(hist, key)]
    if not vals:
        return None
    return max(vals) if mode == "max" else min(vals)


def _median(hist: Dict[str, Any], key: str) -> Optional[float]:
    vals = sorted(v for _, v in _pairs(hist, key))
    if not vals:
        return None
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else 0.5 * (vals[mid - 1] + vals[mid])


def _steps_to(hist: Dict[str, Any], key: str, target: float) -> Optional[int]:
    """First step at which ``key`` reaches ``target``. Separates speed from quality."""
    for step, val in _pairs(hist, key):
        if val >= target:
            return step
    return None


def fit_gain_slope(hist: Dict[str, Any],
                   key: str = "gain_median") -> Optional[Tuple[float, float, int]]:
    """Least-squares slope of ``log(gain)`` against ``log(epoch)``.

    Epoch 0 is excluded -- the accumulated update is identically zero there, so its
    log is undefined -- and four points are the minimum worth fitting. The overnight
    driver's report calls this same function, so the slope it prints mid-run and the
    slope in ``gain_fits.csv`` cannot drift apart.

    A slope near 1/2 means successive sign steps stay incoherent (the ``unit-gain``
    exponent); near 1 means they align (the muP exponent).
    """
    pts = [(s, v) for s, v in _pairs(hist, key) if s > 0 and v > 0]
    if len(pts) < 4:
        return None
    xs = [math.log(s) for s, _ in pts]
    ys = [math.log(v) for _, v in pts]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, r2, n


# --------------------------------------------------------------------------
# Scanning
# --------------------------------------------------------------------------

class Run:
    """One ``metrics.json``: its config, its history, and the path it came from."""

    def __init__(self, path: Path, payload: Dict[str, Any]) -> None:
        self.path = path
        self.config: Dict[str, Any] = payload.get("config") or {}
        self.history: Dict[str, Any] = payload.get("history") or {}
        # The directory name is authoritative for identity: config["run_name"] is
        # present in current runs but was not always written, and the seed is only
        # ever unambiguous from the seed<N> directory.
        self.run_name = self.config.get("run_name") or path.parent.parent.name
        self.seed = self._seed_from(path)
        self.phase = phase_of(self.run_name)

    @staticmethod
    def _seed_from(path: Path) -> Optional[int]:
        name = path.parent.name
        if name.startswith("seed"):
            try:
                return int(name[4:])
            except ValueError:
                return None
        return None

    @property
    def last_k(self) -> int:
        try:
            return max(1, int(self.config.get("last_k") or 5))
        except (TypeError, ValueError):
            return 5

    @property
    def n_epochs_recorded(self) -> int:
        return len(self.history.get("steps") or [])


def scan(root: Path) -> Tuple[List[Run], List[Tuple[Path, str]]]:
    """Every ``*/seed*/metrics.json`` under ``root``, plus whatever failed to parse.

    Unreadable runs are collected rather than raised on: a partially written
    ``metrics.json`` from an interrupted job should cost one warning line, not the
    whole export.
    """
    runs: List[Run] = []
    bad: List[Tuple[Path, str]] = []
    for path in sorted(root.glob("*/seed*/metrics.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            bad.append((path, str(exc)))
            continue
        if not isinstance(payload, dict) or "history" not in payload:
            bad.append((path, "no 'history' key -- not a metrics.json?"))
            continue
        runs.append(Run(path, payload))
    return runs, bad


# --------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------

def _r(val: Optional[float], nd: int = 4) -> Optional[float]:
    """Round a derived metric, leaving ``None`` alone (csv writes it as empty)."""
    return None if val is None else round(val, nd)


def group_reported(runs: Sequence[Run]) -> Dict[Tuple, List[Run]]:
    """Reported runs keyed by everything except the seed, seeds kept together.

    The key carries ``weight_decay`` as well as the optimizer and rate, so the
    decay ablation cannot merge into the row it is an ablation *of* -- the two run
    at the same ``eta_0`` by design, which is exactly what makes them collide under
    a looser key.
    """
    out: Dict[Tuple, List[Run]] = {}
    for run in runs:
        if run.phase not in ("final", "wd"):
            continue
        key = (run.phase, str(run.config.get("optimizer")),
               run.config.get("lr"), run.config.get("lr_scaling"),
               run.config.get("weight_decay"), run.config.get("epochs"))
        out.setdefault(key, []).append(run)
    for group in out.values():
        group.sort(key=lambda r: (r.seed is None, r.seed))
    return out


def mean_std(values: Sequence[Optional[float]]) -> Tuple[Optional[float], Optional[float], int]:
    """``(mean, sample std, n)`` over the non-``None`` entries.

    Sample std (``n-1``), and ``None`` rather than ``0.0`` at ``n = 1``: a single
    seed measured no dispersion at all, and printing zero for it reads as perfect
    agreement.
    """
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None, 0
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return mean, None, 1
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, math.sqrt(var), len(vals)


def _relpath(path: Path) -> str:
    """Path relative to ``results/`` when it lives there, else absolute.

    ``--root`` may point outside the results tree (a copied-out directory, a test
    fixture), and a bare ``relative_to`` would raise on exactly that case.
    """
    try:
        return path.relative_to(RESULTS).as_posix()
    except ValueError:
        return path.as_posix()


def write_runs(runs: Sequence[Run], out: Path, targets: Sequence[float]) -> Path:
    target_cols = [f"epochs_to_{t:g}" for t in targets]
    cols = RUN_COLS + [
        "n_epochs_recorded", "test_acc_final", "test_acc_tail", "test_acc_best",
        "val_acc_tail", "val_acc_best", "train_loss_final", "train_acc_final",
        *target_cols, "epoch_seconds_median",
        "gain_slope", "gain_r2", "gain_points", "metrics_path",
    ]
    path = out / "runs.csv"
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for run in runs:
            row: Dict[str, Any] = {c: run.config.get(c) for c in RUN_COLS}
            row.update(run_name=run.run_name, seed=run.seed, phase=run.phase)
            hist, k = run.history, run.last_k
            row.update(
                n_epochs_recorded=run.n_epochs_recorded,
                test_acc_final=_r(_last(hist, "test_acc")),
                test_acc_tail=_r(_tail_mean(hist, "test_acc", k)),
                test_acc_best=_r(_best(hist, "test_acc")),
                val_acc_tail=_r(_tail_mean(hist, "val_acc", k)),
                val_acc_best=_r(_best(hist, "val_acc")),
                train_loss_final=_r(_last(hist, "train_loss"), 6),
                train_acc_final=_r(_last(hist, "train_acc")),
                epoch_seconds_median=_r(_median(hist, "epoch_seconds"), 3),
                metrics_path=_relpath(run.path),
            )
            for target, col in zip(targets, target_cols):
                row[col] = _steps_to(hist, "test_acc", target)
            fit = fit_gain_slope(hist)
            if fit:
                slope, r2, n = fit
                row.update(gain_slope=_r(slope, 4), gain_r2=_r(r2, 4), gain_points=n)
            w.writerow(row)
    return path


def write_curves(runs: Sequence[Run], out: Path) -> Path:
    cols = ["run_name", "seed", "phase", "optimizer", "lr_scaling", "split",
            "epochs", "lr", "epoch", *CURVE_SERIES]
    path = out / "curves.csv"
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for run in runs:
            head = [run.run_name, run.seed, run.phase,
                    run.config.get("optimizer"), run.config.get("lr_scaling"),
                    run.config.get("split"), run.config.get("epochs"),
                    run.config.get("lr")]
            steps = run.history.get("steps") or []
            # Index by step so a null-padded series stays aligned with its epoch.
            byname = {s: dict(_pairs(run.history, s)) for s in CURVE_SERIES}
            for step in steps:
                w.writerow(head + [step] +
                           [byname[s].get(int(step)) for s in CURVE_SERIES])
    return path


def write_gain(runs: Sequence[Run], out: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """The ``--log-gain`` series and its fit -- the direct alpha measurement."""
    gain_runs = [r for r in runs if _pairs(r.history, "gain_median")]
    if not gain_runs:
        return None, None

    series_path = out / "gain.csv"
    with open(series_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run_name", "optimizer", "lr_scaling", "lr", "epoch",
                    *GAIN_SERIES])
        for run in gain_runs:
            head = [run.run_name, run.config.get("optimizer"),
                    run.config.get("lr_scaling"), run.config.get("lr")]
            byname = {s: dict(_pairs(run.history, s)) for s in GAIN_SERIES}
            for step in run.history.get("steps") or []:
                w.writerow(head + [step] +
                           [byname[s].get(int(step)) for s in GAIN_SERIES])

    fit_path = out / "gain_fits.csv"
    with open(fit_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run_name", "optimizer", "lr_scaling", "lr", "epochs",
                    "slope", "r2", "points", "reading"])
        for run in gain_runs:
            fit = fit_gain_slope(run.history)
            if not fit:
                continue
            slope, r2, n = fit
            reading = ("incoherent -> alpha=1/2" if slope < 0.7 else
                       "aligned -> alpha=1" if slope > 0.85 else
                       "between 1/2 and 1")
            w.writerow([run.run_name, run.config.get("optimizer"),
                        run.config.get("lr_scaling"), run.config.get("lr"),
                        run.config.get("epochs"),
                        round(slope, 4), round(r2, 4), n, reading])
    return series_path, fit_path


def table_rows(runs: Sequence[Run], targets: Sequence[float]) -> List[Dict[str, Any]]:
    """The paper's table: one row per reported configuration, aggregated over seeds.

    Every column is a **per-seed quantity first, averaged over seeds second**, which
    is how the paper defines them and is not the same as aggregating the pooled
    epochs: the tail mean of a pooled series is not the mean of the per-seed tail
    means once two seeds record different numbers of epochs.

    Sorted by test accuracy, descending, which is the order the table is printed in.
    """
    rows: List[Dict[str, Any]] = []
    for key, group in group_reported(runs).items():
        phase, opt, lr, rule, wd, epochs = key
        k = group[0].last_k
        acc, acc_sd, n = mean_std([_tail_mean(r.history, "test_acc", k) for r in group])
        row: Dict[str, Any] = {
            "phase": phase, "optimizer": opt, "lr": lr, "lr_scaling": rule,
            "weight_decay": wd, "epochs": epochs, "n_seeds": n,
            "seeds": " ".join(str(r.seed) for r in group if r.seed is not None),
            "last_k": k,
            "test_acc_mean": _r(acc, 2), "test_acc_std": _r(acc_sd, 2),
            "train_acc_mean": _r(mean_std(
                [_last(r.history, "train_acc") for r in group])[0], 2),
            "test_loss_mean": _r(mean_std(
                [_tail_mean(r.history, "test_loss", k) for r in group])[0], 4),
            "epoch_seconds_median": _r(mean_std(
                [_median(r.history, "epoch_seconds") for r in group])[0], 1),
        }
        for t in targets:
            reached = [_steps_to(r.history, "test_acc", t) for r in group]
            row[f"epochs_to_{t:g}"] = _r(mean_std(reached)[0], 1)
            # A method that never crossed the threshold in some seeds makes the
            # mean of the others an underestimate. Say how many crossed.
            row[f"n_reached_{t:g}"] = sum(1 for v in reached if v is not None)
        rows.append(row)
    rows.sort(key=lambda r: (r["phase"], -(r["test_acc_mean"] or -1e9)))
    return rows


TABLE_COLS = ["phase", "optimizer", "lr", "lr_scaling", "weight_decay", "epochs",
              "n_seeds", "seeds", "last_k", "test_acc_mean", "test_acc_std",
              "train_acc_mean", "test_loss_mean", "epoch_seconds_median"]


def write_table(runs: Sequence[Run], out: Path,
                targets: Sequence[float]) -> Optional[Path]:
    rows = table_rows(runs, targets)
    if not rows:
        return None
    cols = TABLE_COLS + [c for t in targets
                         for c in (f"epochs_to_{t:g}", f"n_reached_{t:g}")]
    path = out / "table_cifar.csv"
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


# --------------------------------------------------------------------------
# The machine
# --------------------------------------------------------------------------

def write_environment(runs: Sequence[Run], out: Path) -> List[Path]:
    """Which machines produced these runs, as JSON and as a LaTeX table row.

    The paper's reproducibility appendix has to name the exact GPU, driver, CUDA,
    Python and PyTorch. Recovering that months later from memory is precisely the
    thing that goes wrong, so every run stamps it into its own ``metrics.json``
    (``common.utils.save_run``) and the bundle carries the distinct values here.
    Anonymous by construction -- ``common.hardware`` collects no hostname, no
    username and no absolute paths.
    """
    # Imported lazily and only for its renderers: `common.hardware` reaches for
    # torch inside `describe()`, which this never calls, so the module stays
    # importable on a login node.
    from common.hardware import LATEX_FOOTER, LATEX_HEADER, as_latex_row, as_sentence

    seen: Dict[Tuple, Dict[str, Any]] = {}
    missing = 0
    for run in runs:
        hw = run.config.get("hardware")
        if not isinstance(hw, dict):
            missing += 1
            continue
        key = (hw.get("gpu"), hw.get("cpu"), hw.get("os"), hw.get("python"),
               hw.get("torch"), hw.get("cuda"), hw.get("driver"),
               hw.get("git_commit"))
        entry = seen.setdefault(key, {"hardware": hw, "n_runs": 0, "runs": []})
        entry["n_runs"] += 1
        entry["runs"].append(f"{run.run_name}/seed{run.seed}")

    payload = {
        "machines": [{"hardware": e["hardware"], "n_runs": e["n_runs"],
                      "runs": sorted(e["runs"])} for e in seen.values()],
        "runs_without_a_hardware_record": missing,
    }
    json_path = out / "environment.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    tex_path = out / "hardware.tex"
    body = [LATEX_HEADER]
    for e in seen.values():
        body.append(as_latex_row(f"Centralized CIFAR-10 (ResNet-18), "
                                 f"{e['n_runs']} runs", e["hardware"]))
    body.append(LATEX_FOOTER)
    tex_path.write_text("\n".join(body) + "\n", encoding="utf-8")

    for e in seen.values():
        print(f"  [{e['n_runs']} runs] {as_sentence(e['hardware'])}")
    if missing:
        print(f"  WARNING {missing} run(s) carry no hardware record -- they predate "
              f"common/hardware.py; re-run or fill the appendix by hand")
    return [json_path, tex_path]


def write_configs(runs: Sequence[Run], out: Path) -> Path:
    path = out / "configs.json"
    payload = {
        f"{r.run_name}/seed{r.seed}": {
            "config": r.config,
            "metrics_path": _relpath(r.path),
            "n_epochs_recorded": r.n_epochs_recorded,
            "series": sorted(k for k in r.history if k != "steps"),
        }
        for r in runs
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def copy_overnight(out: Path, root: Path,
                   max_bytes: int = 4 * 1024 * 1024) -> List[Path]:
    """Copy the overnight driver's own small artifacts (state, report) verbatim.

    Taken from ``root``'s sibling rather than from the default results tree, so a
    ``--root`` pointing at a copied-out directory picks up *that* run's report
    instead of whatever happens to sit in ``results/overnight``.

    Globbed rather than named, so a renamed report still travels; size-capped so a
    stray checkpoint in that directory cannot inflate the bundle.

    "Verbatim" except for one rewrite: the driver stores an absolute ``log`` and
    ``metrics`` path per job, which on the machines we run on begins with a home
    directory. `repo_relative` cuts those back to ``results/...`` so the bundle
    carries no username (`common/paths.py`).
    """
    src = root.parent / "overnight"
    if not src.is_dir():
        return []
    dst = out / "overnight"
    dst.mkdir(parents=True, exist_ok=True)
    copied = []
    for item in sorted(src.iterdir()):
        if not (item.is_file() and item.stat().st_size <= max_bytes):
            continue
        target = dst / item.name
        if item.suffix in (".json", ".md", ".txt", ".csv", ".tex"):
            target.write_text(repo_relative(item.read_text(encoding="utf-8")),
                              encoding="utf-8")
        else:
            shutil.copy2(item, target)
        copied.append(target)
    return copied


# --------------------------------------------------------------------------
# Console summary
# --------------------------------------------------------------------------

def _fmt(val: Optional[float], spec: str = "7.2f") -> str:
    """Format a metric, right-aligning a dash in the same width when it is absent."""
    width = int((spec.split(".")[0].lstrip("+-<>^") or "0"))
    return format("-", f">{width}") if val is None else format(val, spec)


def print_summary(runs: Sequence[Run], targets: Sequence[float]) -> None:
    """Print what the article needs, so the numbers can be read before transfer."""
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)

    counts: Dict[str, int] = {}
    for run in runs:
        counts[run.phase] = counts.get(run.phase, 0) + 1
    print("\nruns by phase: " +
          ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    rules = sorted({str(r.config.get("lr_scaling")) for r in runs
                    if r.config.get("lr_scaling")})
    print(f"lr_scaling rules present: {', '.join(rules) or '(none)'}")
    if len(rules) > 1:
        print("  !! more than one per-layer rule in this tree. They are recorded per "
              "run, so\n  !! nothing is corrupted, but a table built across them is "
              "not one comparison.")

    target = targets[0]
    for phase, title in (("final", "PAPER TABLE (full 50k, test set)"),
                         ("wd", "WEIGHT-DECAY ABLATION")):
        rows = [r for r in table_rows(runs, targets) if r["phase"] == phase]
        if not rows:
            continue
        print(f"\n--- {title} ---")
        print(f"{'optimizer':<16}{'eta_0':>10}{'n':>3}{'test acc':>16}"
              f"{'train':>8}{f'ep->{target:g}%':>10}{'s/ep':>7}")
        for row in rows:
            sd = ("" if row["test_acc_std"] is None
                  else f" +/- {row['test_acc_std']:.2f}")
            lr = row["lr"]
            print(f"{row['optimizer']:<16}"
                  f"{(f'{lr:.6g}' if isinstance(lr, (int, float)) else '-'):>10}"
                  f"{row['n_seeds']:>3}"
                  f"{_fmt(row['test_acc_mean'], '10.2f')}{sd:<6}"
                  f"{_fmt(row['train_acc_mean'], '8.2f')}"
                  f"{_fmt(row[f'epochs_to_{target:g}'], '10.1f')}"
                  f"{_fmt(row['epoch_seconds_median'], '7.1f')}")
        thin = [r["optimizer"] for r in rows if r["n_seeds"] < 2]
        if thin:
            print(f"  single-seed, no dispersion measured: {', '.join(thin)}")

    fits = [(r, fit_gain_slope(r.history)) for r in runs]
    fits = [(r, f) for r, f in fits if f]
    if fits:
        print("\n--- GAIN EXPONENT (log gain_median vs log epoch) ---")
        print(f"{'run':<34}{'rule':<12}{'slope':>8}{'R^2':>8}{'pts':>5}")
        for run, (slope, r2, n) in sorted(fits, key=lambda rf: rf[0].run_name):
            print(f"{run.run_name[:33]:<34}"
                  f"{str(run.config.get('lr_scaling')):<12}"
                  f"{slope:>8.3f}{r2:>8.3f}{n:>5}")
        print("  slope ~ 0.5 -> incoherent accumulation, supports unit-gain (alpha=1/2)")
        print("  slope ~ 1.0 -> aligned accumulation, supports mup (alpha=1)")


# --------------------------------------------------------------------------
# Manifest and archive
# --------------------------------------------------------------------------

def write_manifest(out: Path, root: Path, runs: Sequence[Run],
                   written: Sequence[Path], bad: Sequence[Tuple[Path, str]],
                   targets: Sequence[float]) -> Path:
    lines = [
        "# Centralized results export",
        "",
        f"* source: `{repo_relative(str(root))}`",
        f"* runs found: {len(runs)}",
        f"* unreadable: {len(bad)}",
        f"* targets for `epochs_to_*`: {', '.join(f'{t:g}%' for t in targets)}",
        "",
        "## Files",
        "",
        "| file | bytes | contents |",
        "| :--- | ---: | :--- |",
    ]
    blurb = {
        "table_cifar.csv": "**the paper's table**: mean +/- sample std over seeds",
        "runs.csv": "one row per run: config + derived summary metrics",
        "curves.csv": "tidy per-epoch series for the figures",
        "gain.csv": "accumulated-gain series (`--log-gain` runs only)",
        "gain_fits.csv": "log-log slope of gain_median vs epoch",
        "environment.json": "every machine that contributed: GPU, CUDA, Python, torch",
        "hardware.tex": "the same as a LaTeX row for the reproducibility appendix",
        "configs.json": "full config of every run",
    }
    for path in written:
        if path is None or not path.exists():
            continue
        rel = path.relative_to(out).as_posix()
        lines.append(f"| `{rel}` | {path.stat().st_size} | "
                     f"{blurb.get(path.name, 'copied verbatim')} |")
    if bad:
        lines += ["", "## Unreadable", ""]
        lines += [f"* `{p}` -- {e}" for p, e in bad]
    seeds_per_group = sorted({row["n_seeds"] for row in table_rows(runs, targets)})
    lines += [
        "",
        "## Notes",
        "",
        "* Accuracies are percentages. `*_tail` is the mean of the final `last_k`",
        "  recorded epochs -- the paper's primary metric -- not a single epoch.",
        "* `val_*` is empty for `split=full` runs, which have no validation set;",
        "  `test_acc` on `split=tune` runs comes from a 45k-trained model and was",
        "  never used for selection.",
        "* `epochs_to_*` is recomputed from the history here, so it does not depend",
        "  on the training logs having been kept.",
        f"* Reported configurations carry {', '.join(str(s) for s in seeds_per_group)}"
        f" seed(s). `test_acc_std` is a *sample* std and is empty at one seed --",
        "  a single seed measured no dispersion; do not read a blank as agreement.",
        "* Redraw every figure from this bundle with",
        "  `python3 -m centralized.plot_analysis --bundle <this directory>`.",
        "",
    ]
    path = out / "MANIFEST.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def make_archive(out: Path) -> Path:
    """One ``.tar.gz`` beside the folder -- a single file is easier to pull down."""
    archive = out.parent / (out.name + ".tar.gz")
    if archive.exists():
        archive.unlink()
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(out, arcname=out.name)
    return archive


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Export the centralized results tree as small CSVs.")
    p.add_argument("--root", type=Path, default=RESULTS / "centralized",
                   help="results tree to scan (default: results/centralized)")
    p.add_argument("--out", type=Path, default=RESULTS / "article_export",
                   help="output directory (default: results/article_export)")
    p.add_argument("--phase", nargs="*", default=None,
                   metavar="PHASE",
                   help="keep only these phases (final wd lr verify alpha gain); "
                        "default keeps everything")
    p.add_argument("--targets", nargs="+", type=float, default=[90.0, 93.0, 94.0],
                   help="accuracy targets for the epochs-to-target columns")
    p.add_argument("--no-archive", action="store_true",
                   help="skip the .tar.gz and leave the folder only")
    p.add_argument("--quiet", action="store_true", help="skip the console summary")
    args = p.parse_args(argv)

    if not args.root.is_dir():
        print(f"error: {args.root} does not exist -- run from code/ with "
              f"'python3 -m centralized.export_article'")
        return 1

    runs, bad = scan(args.root)
    if args.phase:
        keep = set(args.phase)
        runs = [r for r in runs if r.phase in keep]
    if not runs:
        print(f"error: no runs matched under {args.root}")
        return 1

    print(f"scanned {args.root}: {len(runs)} runs"
          + (f", {len(bad)} unreadable" if bad else ""))
    for path, err in bad:
        print(f"  WARNING unreadable {path}: {err}")

    # Rebuilt from scratch each time so a deleted run cannot linger in the bundle.
    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    written: List[Path] = [
        write_runs(runs, args.out, args.targets),
        write_curves(runs, args.out),
    ]
    table = write_table(runs, args.out, args.targets)
    if table:
        written.append(table)
    series_path, fit_path = write_gain(runs, args.out)
    written += [pp for pp in (series_path, fit_path) if pp]
    print("\nmachines:")
    written += write_environment(runs, args.out)
    written.append(write_configs(runs, args.out))
    written += copy_overnight(args.out, args.root)
    written.append(write_manifest(args.out, args.root, runs, written, bad,
                                  args.targets))

    total = sum(pp.stat().st_size for pp in written if pp.exists())
    print(f"\nwrote {len(written)} files, {total / 1024:.0f} KiB -> {args.out}")

    if not args.no_archive:
        archive = make_archive(args.out)
        print(f"archive: {archive}  ({archive.stat().st_size / 1024:.0f} KiB)")
        print("  ^ this single file is what to bring home. Unpack it to\n"
              "    results/article_export/, then:\n"
              "      python3 -m centralized.plot_analysis")

    if not args.quiet:
        print_summary(runs, args.targets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
