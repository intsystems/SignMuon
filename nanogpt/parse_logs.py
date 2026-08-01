"""Turn modded-nanogpt run logs into tidy data.

A speedrun log is a self-contained artefact: it starts with a verbatim copy of
the training script and of ``signmuon_optimizers.py`` (so the run reproduces
itself), then a JSON ``RUNMETA`` header, then one line per step. Excellent for
provenance, useless for plotting. This module extracts the numbers.

Outputs (into ``-o OUTDIR``, default ``../results/nanogpt/``, the tree
REPRODUCE.md treats as this arm's results, beside ``synthetic/`` and
``federated/``):

``runs.csv``
    One row per run: optimizer, lr, momentum, weight decay, lr-scaling rule,
    world size, whether it diverged, final/best validation loss, total training
    time, ms/step, peak memory, and the time-to-target ("speedrun") numbers.
``steps.csv``
    Long/tidy: ``run_id, optimizer, lr, step, wallclock_ms, train_loss, val_loss``.
    One row per logged step; ``train_loss`` and ``val_loss`` are sparse (a step
    has one, the other, or both). This is the frame the plots consume.
``runs.json``
    The same as ``runs.csv`` plus the full RUNMETA/RUNEND dicts.

Wall-clock note
---------------
Two clocks appear in a log and they are NOT the same thing:

* ``train_time:`` on a *val_loss* line is the authoritative cumulative training
  time, with validation excluded -- the number the speedrun reports.
* ``train_time:`` on a plain step line is ``approx_training_time_ms``, measured
  before the step's own CUDA work has necessarily completed.

Validation-point times are therefore used as anchors, and per-step times are
kept as-is for the fine-grained curve. Both are "training time", i.e. evaluation
and compilation are excluded, which is what makes loss-vs-time comparable across
optimizers.

Usage
-----
    python parse_logs.py                     # the eight canonical runs
    python parse_logs.py ../results/nanogpt/logs/Muon_lr0.06_5db64adc.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

#: Paths default to this arm's results tree, resolved against this file so
#: the scripts work from any working directory.
_HERE = Path(__file__).resolve().parent


__all__ = ["parse_log", "parse_many", "write_csv", "RunRecord"]

# step:1234/2330 val_loss:3.2812 train_time:141234ms step_avg:60.61ms
# EF21-MuonSign additionally reports the broadcast model:
# step:1234/2330 val_loss:7.8221 val_loss_W:3.3612 train_time:141234ms step_avg:60.61ms
# ``val_loss`` stays the exact server model X (the iterate the convergence
# corollary bounds), so this column means the same thing in old and new logs; the
# optional group keeps pre-existing logs parseable.
_VAL_RE = re.compile(
    r"^step:(?P<step>\d+)/(?P<total>\d+)\s+val_loss:(?P<val>[-\deE.]+|nan|inf|-inf)"
    r"(?:\s+val_loss_W:(?P<valw>[-\deE.]+|nan|inf|-inf))?\s+"
    r"train_time:(?P<ms>[\d.]+)ms")
# step:1234/2330 train_time:141234ms step_avg:60.61ms
_STEP_RE = re.compile(
    r"^step:(?P<step>\d+)/(?P<total>\d+)\s+train_time:(?P<ms>[\d.]+)ms")
# step:1233 train_loss:3.281234
_TRAIN_RE = re.compile(
    r"^step:(?P<step>\d+)\s+train_loss:(?P<loss>[-\deE.]+|nan|inf|-inf)\s*$")
# The per-validation diagnostics block written by
# ``_DistributedMatrixOptimizer.diagnostics_report``. The slot columns present
# depend on the method (a slot no method touched is omitted), so the column
# header has to be read rather than assumed. All three patterns are anchored and
# only ever applied past RUNMETA, so the verbatim source dump at the top of the
# log -- which contains these same strings inside f-strings -- cannot match.
_DIAG_HDR_RE = re.compile(r"^diagnostics \((?P<opt>\w+);")
_DIAG_COLS_RE = re.compile(r"^\s+parameter\s+count\s+(?P<cols>[\w\s]+?)\s*$")
_DIAG_ROW_RE = re.compile(r"^\s{2}(?P<name>\S+)\s+(?P<count>\d+)\s+(?P<vals>[-\d.eE+\s]+?)\s*$")
_DIAG_BLK_RE = re.compile(
    r"^\s{2}(?P<name>\S+) mean\|grad\| per Q,K,V,O block:\s+"
    r"(?P<vals>[\d.eE+-]+\s+[\d.eE+-]+\s+[\d.eE+-]+\s+[\d.eE+-]+)\s+max/min=(?P<ratio>\S+)")
_META_RE = re.compile(r"^RUNMETA (?P<json>\{.*\})\s*$")
_END_RE = re.compile(r"^RUNEND (?P<json>\{.*\})\s*$")
_DIVERGED_RE = re.compile(r"^DIVERGED step:(?P<step>\d+)/")
_PEAK_RE = re.compile(r"^peak memory allocated: (?P<mib>\d+) MiB")
# fallback for logs written before RUNMETA existed
_OPTLINE_RE = re.compile(r"^hidden-matrix optimizer: (?P<opt>\S+)\s+config=(?P<cfg>\{.*\})")
_GPU_RE = re.compile(r"\|\s+\d+\s+(?P<name>NVIDIA [A-Za-z0-9 \-]+?)\s{2,}")


def _f(x: str) -> float:
    return float(x)


class RunRecord(dict):
    """One run: scalar summary fields plus the per-step series."""

    @property
    def steps(self) -> list[dict[str, Any]]:
        return self["_steps"]


def parse_log(path: Path) -> RunRecord | None:
    """Parse one log file. Returns ``None`` if it contains no step lines
    (e.g. a run that died during compilation)."""
    meta: dict[str, Any] = {}
    end: dict[str, Any] = {}
    train_loss: dict[int, float] = {}
    val: dict[int, tuple[float, float]] = {}     # step -> (val_loss, cumulative ms)
    val_w: dict[int, float] = {}                 # step -> val_loss of W (EF21-MuonSign)
    diag: list[dict[str, Any]] = []              # per-validation diagnostics rows
    diag_cols: list[str] | None = None
    diag_step: int | None = None
    step_ms: dict[int, float] = {}               # step -> approx cumulative ms
    diverged_at: int | None = None
    peak_mib: int | None = None
    gpu: str | None = None
    total_steps: int | None = None

    seps = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            # The log opens with a verbatim dump of the training script and of
            # signmuon_optimizers.py, whose own text contains every one of these
            # patterns inside f-strings. Two things keep the source from being
            # read as data: every regex is anchored at ^ (in the source those
            # patterns are always inside an indented `print0(f"...")`), and the
            # per-step regexes stay switched off until the preamble is over --
            # marked by RUNMETA, or, in logs written before RUNMETA existed, by
            # the third "="*100 separator that closes the environment dump.
            if line.startswith("=" * 20):
                seps += 1
                continue
            if not meta:
                m = _META_RE.match(line)
                if m:
                    meta = json.loads(m.group("json"))
                    continue
                m = _OPTLINE_RE.match(line)      # pre-RUNMETA logs
                if m:
                    cfg = m.group("cfg").replace("'", '"')
                    try:
                        meta = dict(json.loads(cfg), optimizer=m.group("opt"))
                    except json.JSONDecodeError:
                        meta = dict(optimizer=m.group("opt"))
                    continue
            if gpu is None:
                g = _GPU_RE.search(line)
                if g:
                    gpu = g.group("name").strip()
                    continue
            if not meta and seps < 3:
                continue                          # still inside the source dump

            m = _TRAIN_RE.match(line)
            if m:
                train_loss[int(m.group("step"))] = _f(m.group("loss"))
                continue
            m = _VAL_RE.match(line)
            if m:
                total_steps = int(m.group("total"))
                diag_step = int(m.group("step"))
                val[diag_step] = (_f(m.group("val")), _f(m.group("ms")))
                if m.group("valw") is not None:
                    val_w[diag_step] = _f(m.group("valw"))
                continue
            # ---- diagnostics block, printed straight after its val line -----
            if _DIAG_HDR_RE.match(line):
                diag_cols = None
                continue
            m = _DIAG_COLS_RE.match(line)
            if m:
                diag_cols = m.group("cols").split()
                continue
            m = _DIAG_BLK_RE.match(line)          # before _DIAG_ROW_RE: also 2-indented
            if m and diag_step is not None:
                row = dict(step=diag_step, parameter=m.group("name"))
                row.update({f"gblk{i}": _f(v)
                            for i, v in enumerate(m.group("vals").split())})
                diag.append(row)
                continue
            if diag_cols:
                m = _DIAG_ROW_RE.match(line)
                if m and diag_step is not None:
                    vals = m.group("vals").split()
                    if len(vals) == len(diag_cols):
                        row = dict(step=diag_step, parameter=m.group("name"),
                                   count=int(m.group("count")))
                        row.update(dict(zip(diag_cols, (_f(v) for v in vals))))
                        diag.append(row)
                        continue
            m = _STEP_RE.match(line)
            if m:
                total_steps = int(m.group("total"))
                step_ms[int(m.group("step"))] = _f(m.group("ms"))
                continue
            m = _DIVERGED_RE.match(line)
            if m:
                diverged_at = int(m.group("step"))
                continue
            m = _PEAK_RE.match(line)
            if m:
                peak_mib = int(m.group("mib"))
                continue
            m = _END_RE.match(line)
            if m:
                end = json.loads(m.group("json"))

    if not (train_loss or val):
        return None

    steps = sorted(set(train_loss) | set(val) | set(step_ms))
    series = []
    for s in steps:
        v = val.get(s)
        # Prefer the validation-line clock (validation excluded, authoritative);
        # fall back to the approximate per-step clock.
        ms = v[1] if v is not None else step_ms.get(s)
        series.append(dict(step=s, wallclock_ms=ms,
                           train_loss=train_loss.get(s),
                           val_loss=None if v is None else v[0],
                           val_loss_w=val_w.get(s)))

    finite_val = {s: v for s, (v, _) in val.items() if v == v and abs(v) != float("inf")}
    finite_val_w = {s: v for s, v in val_w.items() if v == v and abs(v) != float("inf")}
    # A run whose validation loss never returns to its first measured value has
    # failed, whether or not it ever went non-finite. EF21-MuonSign at eta=0.06
    # climbs 4.39 -> 8.08 on the exact server model X while staying finite
    # throughout, which the `diverged` flag (non-finite only) reports as a clean
    # completion. Surface it as its own column rather than overloading `diverged`,
    # whose meaning other consumers already rely on.
    _probe = {s: v for s, v in finite_val.items() if s > 0}
    val_rose = (len(_probe) >= 2
                and _probe[max(_probe)] > _probe[min(_probe)])
    rec = RunRecord(
        run_id=meta.get("run_id", path.stem),
        log=str(path),
        optimizer=meta.get("optimizer"),
        family=meta.get("family"),
        lr=meta.get("lr"),
        momentum=meta.get("momentum"),
        weight_decay=meta.get("weight_decay"),
        lr_scaling=meta.get("lr_scaling"),
        # Absent from logs written before the seed knob existed (the eight runs in
        # the paper), where the initialization is whatever torch drew at process
        # start -- so `None` here means "unseeded", not "seed 0".
        seed=meta.get("seed"),
        world_size=meta.get("world_size"),
        train_steps=meta.get("train_steps", total_steps),
        tokens_per_step=meta.get("tokens_per_step"),
        gpu=gpu,
        diverged=bool(end.get("diverged", diverged_at is not None)),
        diverged_at=diverged_at,
        completed=bool(end),
        last_step=max(steps) if steps else None,
        final_val_loss=finite_val.get(max(finite_val)) if finite_val else None,
        best_val_loss=min(finite_val.values()) if finite_val else None,
        # EF21-MuonSign only: the sign-compressed broadcast model W, i.e. the
        # iterate every gradient was actually evaluated at. None for every other
        # method, where W and X are the same tensor.
        final_val_loss_w=finite_val_w.get(max(finite_val_w)) if finite_val_w else None,
        best_val_loss_w=min(finite_val_w.values()) if finite_val_w else None,
        val_rose=bool(val_rose),
        train_time_ms=end.get("train_time_ms",
                              max((m for m in step_ms.values()), default=None)),
        peak_memory_mib=end.get("peak_memory_mib", peak_mib),
    )
    n = rec["last_step"] or 0
    rec["ms_per_step"] = round(rec["train_time_ms"] / n, 3) if rec["train_time_ms"] and n else None
    rec["_steps"] = series
    rec["_diag"] = diag
    rec["_meta"] = meta
    rec["_end"] = end
    return rec


def _first_crossing(pts: list[tuple[float, float]], target: float) -> float | None:
    """First x at which the piecewise-linear curve through ``pts`` reaches
    ``target`` from above. ``None`` if it never does."""
    prev = None
    for x, v in pts:
        if v <= target:
            if prev is None:
                return float(x)
            px, pv = prev
            if pv == v:
                return float(x)
            return px + (pv - target) / (pv - v) * (x - px)
        prev = (x, v)
    return None


def time_to_loss(rec: RunRecord, target: float, key: str = "val_loss") -> float | None:
    """Training milliseconds until validation loss first reaches ``target``.

    Linearly interpolates between the two bracketing validation points, so the
    number does not depend on where the (coarse) validation grid happens to fall.
    """
    return _first_crossing([(s["wallclock_ms"], s[key]) for s in rec.steps
                            if s.get(key) is not None
                            and s["wallclock_ms"] is not None], target)


def steps_to_loss(rec: RunRecord, target: float, key: str = "val_loss") -> float | None:
    """Optimizer steps until validation loss first reaches ``target``."""
    return _first_crossing([(s["step"], s[key]) for s in rec.steps
                            if s.get(key) is not None], target)


def parse_many(paths: Iterable[Path]) -> list[RunRecord]:
    out = []
    for p in sorted(paths):
        try:
            rec = parse_log(p)
        except Exception as exc:                      # a truncated log must not kill the batch
            print(f"  !! {p.name}: {type(exc).__name__}: {exc}")
            continue
        if rec is None:
            print(f"  -- {p.name}: no step lines, skipped")
            continue
        out.append(rec)
    return out


_RUN_FIELDS = ["run_id", "optimizer", "family", "lr", "momentum", "weight_decay",
               "lr_scaling", "seed", "world_size", "gpu", "train_steps", "last_step",
               "diverged", "val_rose", "completed", "final_val_loss", "best_val_loss",
               "final_val_loss_w", "best_val_loss_w",
               "train_time_ms", "ms_per_step", "peak_memory_mib", "log"]


def _relative_log(path_str: str, outdir: Path) -> str:
    """The log's path as recorded in the outputs: relative to ``outdir`` when it
    sits under it, else the bare filename.

    Never the absolute path. These CSVs ship in the repository and in the
    double-blind bundle, and an absolute path carries the analyst's username --
    which is exactly what ``anonymize.py`` fails the build on.
    """
    p = Path(path_str)
    if not p.is_absolute():          # already rewritten; idempotent
        return p.as_posix()
    try:
        return p.resolve().relative_to(outdir.resolve()).as_posix()
    except ValueError:
        return p.name


def write_csv(records: list[RunRecord], outdir: Path, targets: list[float]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for r in records:
        r["log"] = _relative_log(r["log"], outdir)

    fields = list(_RUN_FIELDS)
    for t in targets:
        # ``_w`` columns are the same crossing measured on EF21-MuonSign's
        # broadcast model, and are empty for every other method. They exist
        # because the X column never reaches these targets for that one arm, so a
        # single set of columns would report "never" for a method that in fact
        # trains normally at the iterate its gradients are taken at.
        fields += [f"steps_to_{t:g}", f"ms_to_{t:g}",
                   f"steps_to_{t:g}_w", f"ms_to_{t:g}_w"]
    with (outdir / "runs.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in records:
            row = {k: r.get(k) for k in _RUN_FIELDS}
            for t in targets:
                row[f"steps_to_{t:g}"] = steps_to_loss(r, t)
                row[f"ms_to_{t:g}"] = time_to_loss(r, t)
                if r.get("final_val_loss_w") is not None:
                    row[f"steps_to_{t:g}_w"] = steps_to_loss(r, t, "val_loss_w")
                    row[f"ms_to_{t:g}_w"] = time_to_loss(r, t, "val_loss_w")
            w.writerow(row)

    with (outdir / "steps.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["run_id", "optimizer", "lr", "lr_scaling", "step",
                    "wallclock_ms", "train_loss", "val_loss", "val_loss_w"])
        for r in records:
            for s in r.steps:
                w.writerow([r["run_id"], r["optimizer"], r["lr"], r["lr_scaling"],
                            s["step"], s["wallclock_ms"], s["train_loss"],
                            s["val_loss"], s.get("val_loss_w")])

    # Per-validation optimizer diagnostics (compressor contraction, estimator lag,
    # server/broadcast gap, per-block gradient magnitudes). Long format: one row
    # per (run, step, parameter), so a method that never wrote a slot simply has
    # no value there rather than a misleading zero.
    diag_fields = ["run_id", "optimizer", "lr", "step", "parameter", "count",
                   "alpha_up", "alpha_dn", "lag_est", "lag_XW",
                   "gblk0", "gblk1", "gblk2", "gblk3"]
    with (outdir / "diagnostics.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=diag_fields, extrasaction="ignore")
        w.writeheader()
        for r in records:
            for row in r.get("_diag", []):
                w.writerow(dict(row, run_id=r["run_id"], optimizer=r["optimizer"],
                                lr=r["lr"]))

    with (outdir / "runs.json").open("w", encoding="utf-8") as fh:
        json.dump([{k: v for k, v in r.items() if k not in ("_steps", "_diag")}
                   for r in records], fh, indent=2, default=str)


def _collect(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            paths += sorted(p.rglob("*.txt"))
        elif p.exists():
            paths.append(p)
        else:                                          # let the shell off the hook
            paths += sorted(Path().glob(item))
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", default=[str(_HERE.parent / "results" / "nanogpt" / "logs")],
                    help="log files and/or directories of logs "
                         "(default: ../results/nanogpt/logs)")
    ap.add_argument("-o", "--outdir", default=_HERE.parent / "results" / "nanogpt", type=Path)
    ap.add_argument("--target", type=float, nargs="*", default=[3.35, 3.28],
                    help="val-loss targets for the time-to-loss columns "
                         "(default 3.35, the threshold `tab:nanogpt` reports, "
                         "and 3.28, the speedrun's own target -- which only the "
                         "uncompressed Muon arm reaches inside the budget)")
    args = ap.parse_args()

    paths = _collect(args.inputs)
    if not paths:
        raise SystemExit(f"no logs found in {args.inputs}")
    print(f"parsing {len(paths)} log file(s)...")
    records = parse_many(paths)
    if not records:
        raise SystemExit("no parsable runs")
    write_csv(records, args.outdir, args.target)

    w = max(len(str(r["optimizer"])) for r in records)
    print(f"\n{'optimizer':<{w}} {'lr':>7} {'steps':>6} {'final val':>10} "
          f"{'best val':>9} {'val(W)':>9} {'time':>9} {'ms/step':>8}  status")
    for r in sorted(records, key=lambda r: (r["final_val_loss"] is None,
                                            r["final_val_loss"] or 0)):
        t = r["train_time_ms"]
        vw = r.get("final_val_loss_w")
        if r["diverged"]:
            status = "DIVERGED"
        elif r.get("val_rose"):
            status = "VAL ROSE"        # finite throughout, but never came back down
        else:
            status = "ok" if r["completed"] else "incomplete"
        print(f"{str(r['optimizer']):<{w}} {r['lr'] or float('nan'):>7.4g} "
              f"{r['last_step'] or 0:>6} "
              f"{(r['final_val_loss'] if r['final_val_loss'] is not None else float('nan')):>10.4f} "
              f"{(r['best_val_loss'] if r['best_val_loss'] is not None else float('nan')):>9.4f} "
              f"{(f'{vw:.4f}' if vw is not None else '-'):>9} "
              f"{(t / 1000 if t else float('nan')):>8.1f}s "
              f"{(r['ms_per_step'] or float('nan')):>8.2f}  {status}")
    print(f"\nwrote {args.outdir}/runs.csv, {args.outdir}/steps.csv, {args.outdir}/runs.json")


if __name__ == "__main__":
    main()
