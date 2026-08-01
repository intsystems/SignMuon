"""One-command overnight run for the centralized ResNet-18 study.

    cd code
    python3 -m centralized.overnight --device cuda:0 --download

Watch the first ~6 minutes: it runs the CPU test suite, prints the per-layer
learning-rate table, records the machine, times two real epochs on *your* GPU,
and then prints a schedule with a finish time per phase. Once the schedule
appears you can leave it.

``--budget-hours`` defaults to 0, meaning **no deadline**: every phase runs to
completion and only Ctrl-C stops the run. ``results/overnight/REPORT.md`` is
rewritten after every phase and after every final run, so it can be read at any
time *without* stopping anything. Ctrl-C stops cleanly and writes the report; a
second Ctrl-C exits at once.

When it finishes, ``python3 -m centralized.export_article`` packs everything the
paper needs into one ``.tar.gz`` to bring home.

Design for an unattended night
------------------------------
* **Budget-aware.** Every phase is costed from the *measured* epoch time, and the
  deadline is checked before each individual job, so a bounded run stops cleanly
  instead of being killed mid-way.
* **Crash-isolated.** Each job is a subprocess; a failure is logged and the run
  continues. One diverging learning rate cannot take down the night.
* **Resumable and incremental.** State is written to
  ``results/overnight/state.json`` after every job, and ``--resume`` skips
  everything already done.
* **Priority-ordered.** The phases are ordered so that stopping early costs the
  least: the diagnostics and eta_0 first, then the headline table, then the
  ablation. Finals are **seed-major** -- every method at seed 0 before any of
  them reaches seed 1 -- so an early stop yields a complete 1-seed table rather
  than a fragmentary 3-seed one.

Every learning rate tried is a **1-2-5 lattice point** (``tune.round_grid``), so a
tuned value is quotable as ``0.02`` rather than ``0.0172354775``, grid extension
stays on the same lattice however many rounds it takes, and two methods anchored
at slightly different places search the *same* grid.

Phases
------
0. ``preflight``  CPU tests, scaling tables, hardware record, 2-epoch timing.
1. ``gain``       ``--log-gain`` runs at a CONSTANT step size: does the accumulated
                  update grow like ``sqrt(t)`` (alpha = 1/2, ``unit-gain``) or like
                  ``t`` (alpha = 1, ``mup``)? Annealing would let the accumulation
                  saturate and the fit would measure the schedule, so this phase
                  passes ``--constant-lr`` and its runs must never be compared with
                  scheduled ones.
2. ``aux``        Is the optimal auxiliary rate method-independent? Two anchor
                  methods an order of magnitude apart in eta_0, each over the same
                  ``lr_aux`` grid. If they agree, one ``lr_aux`` may be held fixed
                  for every method and *reported as verified* rather than asserted.
3. ``lr``         eta_0 per method under the chosen rule, equal budget each,
                  **at the reporting horizon** -- see below.
4. ``final``      full 50k training runs at the tuned values, seed-major.
5. ``wd``         re-run the best few methods with weight decay switched on. The
                  primary table is unregularized -- that is the setting the theorems
                  analyse, the one the nanoGPT record #40 config uses, and the one
                  Mishra et al.'s own sweep selects -- so this phase supplies the
                  regularized number and shows whether the ordering moves.

Why eta_0 is selected at the full horizon
-----------------------------------------
An earlier version of this driver ranked learning rates at a 15-epoch proxy and
re-checked the top few at 75. The check failed: on the 2026-07-27 run the ranking
*reversed* for both methods probed -- SignMuon's best moved from 0.02 to 0.2 and
Muon's from 0.05 to 0.01. That is not seed noise but a systematic artefact of the
proxy. Both runs anneal cosinally to zero over their own horizon, so a 15-epoch
run spends almost all of its budget at a decayed rate and a 75-epoch run does not;
the proxy therefore measures a different schedule, and its bias is not even in a
consistent direction across families.

So the ``lr`` phase now runs at ``--final-epochs``. It costs more than a proxy and
buys the one thing a proxy cannot: the rate reported in the table is the rate that
won at the horizon the table reports. Selection is still on ``val_acc`` from the
45k/5k split, so the test set enters nothing.

``lr_aux`` is exempt, and legitimately: the ``aux`` phase asks whether the *argmax
over lr_aux* agrees between two methods measured at the **same** horizon, and a
shared horizon bias cancels in that comparison. It runs at ``--aux-epochs``.

One rule per results tree
-------------------------
``--lr-scaling`` and ``--weight-decay-mode`` are independent flags, and nothing
stops two invocations with different settings from writing into the same
``results/centralized``. They are recorded per run, so nothing is corrupted -- but
a later reader comparing across invocations can easily attribute a weight-decay
difference to the scaling rule. If you deliberately want two arms, keep them in
separate result trees (``SIGNMUON_RESULTS``).

The per-layer exponent sweep is a separate tool, and is not needed for the paper's
numbers -- phase 1 measures the exponent directly:
``python3 -m centralized.tune --stage alpha --epochs 15``.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from argparse import Namespace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

from centralized.export_article import fit_gain_slope
from centralized.tune import (ALL_METHODS, AUX_GRID, LEGACY_ANCHORS, ROOT,
                              anchor_for, best_of, boundary_warning,
                              canonical_tag, extend_grid, round_grid, run_one)
from common.lr_scaling import FAMILY_SIGN, describe_rule, resolve_rule
from common.paths import scrub
from common.utils import results_root

OUT_DIR = results_root() / "overnight"
STATE_PATH = OUT_DIR / "state.json"
REPORT_PATH = OUT_DIR / "REPORT.md"

#: The phases, in the order they are always run. ``--phases`` selects a subset;
#: it does not reorder, because the later phases consume the earlier ones' output.
PHASES = ("gain", "aux", "lr", "final", "wd")

#: Startup cost of one subprocess (imports, CUDA init, dataset scan), seconds.
JOB_OVERHEAD_S = 35.0

_stop = {"requested": False}


def _handle_sigint(signum, frame):        # pragma: no cover - interactive
    if _stop["requested"]:
        raise KeyboardInterrupt("second interrupt -- exiting now")
    _stop["requested"] = True
    print("\n[interrupt] stopping cleanly and writing the report. "
          "Press Ctrl-C again to exit immediately.", flush=True)


def record(state: Dict, tag: str, result) -> None:
    """Record a job's outcome, then persist the state.

    A job cut short by Ctrl-C is deliberately **not** recorded: ``--resume`` skips
    any tag already present, so recording an interrupted run would silently retire
    it. Genuine failures *are* recorded, because a configuration that diverges will
    diverge again and re-running it only wastes the next night.
    """
    if result is None and _stop["requested"]:
        print(f"  (interrupted before {tag} finished -- not recorded, "
              f"--resume will retry it)")
        return
    state["jobs"][tag] = result
    save_state(state)


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------


def load_state() -> Dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"[warn] {STATE_PATH} is corrupt; starting fresh")
    return {"jobs": {}, "phases": {}, "started": None, "epoch_seconds": None}


def save_state(state: Dict) -> None:
    """Persist the driver state, with machine-specific paths taken out.

    Scrubbed on the way to disk rather than in memory: the driver reopens
    ``job["metrics"]`` to refit a diagnostic, and both drivers run from ``code/``,
    so a `results/...` path resolves on resume just as an absolute one did. Now
    that `results/` ships with the code, `state.json` is a file reviewers read.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(scrub(state), indent=2), encoding="utf-8")


def stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


class Budget:
    """Deadline bookkeeping, with a per-job feasibility check."""

    def __init__(self, hours: float):
        self.start = time.time()
        self.hours = hours
        #: ``hours <= 0`` means no deadline: every phase runs to completion and only
        #: Ctrl-C stops the run.
        self.unlimited = hours <= 0
        self.deadline = float("inf") if self.unlimited else self.start + hours * 3600.0

    def left(self) -> float:
        return float("inf") if self.unlimited else self.deadline - time.time()

    def fits(self, seconds: float) -> bool:
        return self.unlimited or self.left() > seconds

    def report(self) -> str:
        elapsed = timedelta(seconds=int(time.time() - self.start))
        if self.unlimited:
            return f"{elapsed} elapsed, no deadline (Ctrl-C to stop)"
        return f"{elapsed} elapsed, {timedelta(seconds=int(max(0.0, self.left())))} left"


# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------


def preflight(args, state: Dict) -> Optional[float]:
    """Run the tests, print the scaling table, record the machine, time two epochs.

    Returns the measured seconds per epoch, or ``None`` if the timing run failed
    (in which case nothing else should be attempted).
    """
    print("=" * 78)
    print(f"[{stamp()}] PREFLIGHT")
    print("=" * 78)

    print("\n[1/4] CPU test suite (no GPU, no downloads)")
    proc = subprocess.run([sys.executable, "-m", "tests.test_code"], cwd=ROOT,
                          capture_output=True, text=True)
    tail = proc.stdout.strip().splitlines()[-1:] or ["(no output)"]
    print(f"      {tail[0]}")
    if proc.returncode != 0:
        print("      FAILED -- fix this before running overnight. Failing tests:")
        # Print the assertion detail, not just the test name: the indented lines
        # that follow a FAIL are what say *which* file leaked, *which* value
        # disagreed. Without them a preflight failure needs a second run to
        # diagnose, which is exactly when nobody has patience for one.
        emit = False
        for line in proc.stdout.splitlines():
            if line.startswith(("FAIL", "ERROR")):
                print(f"        {line}")
                emit = True
            elif emit and (line.startswith((" ", "\t")) and line.strip()):
                print(f"        {line.rstrip()}")
            elif line.strip():
                emit = False
        if not args.force:
            # No hint about the anonymity scan here: the test that fails on one
            # prints the command itself, and that test ships only where the scanner
            # does. Naming it here would point the anonymous bundle at a module it
            # deliberately does not carry.
            print("      (re-run with --force to train anyway)")
            return None
        print("      --force given, continuing anyway")

    print(f"\n[2/4] per-layer learning-rate multipliers, rule '{args.lr_scaling}'")
    rule = resolve_rule(args.lr_scaling)
    shapes = [("conv1", (64, 3, 3, 3)), ("layer1.conv", (64, 64, 3, 3)),
              ("layer2.conv", (128, 128, 3, 3)), ("layer2.downsample", (128, 64, 1, 1)),
              ("layer3.conv", (256, 256, 3, 3)), ("layer4.conv", (512, 512, 3, 3)),
              ("layer4.downsample", (512, 256, 1, 1))]
    for family in (FAMILY_SIGN, "lmo"):
        print(describe_rule(rule, family, shapes))

    # The paper's reproducibility appendix needs the exact GPU / Python / PyTorch /
    # CUDA that produced the numbers. Capture it here, once, into the run state, so
    # it travels with the report and the export bundle instead of being recalled
    # from memory months later. Every run's metrics.json carries the same block.
    print("\n[3/4] machine")
    state["hardware"] = _hardware(args.device)
    print(f"      {_hardware_sentence(state['hardware'])}")

    print(f"\n[4/4] timing 2 real epochs of {args.model} on {args.device}")
    r = run_one(_tune_args(args), lr=LEGACY_ANCHORS["muon"], lr_aux=args.lr_aux,
                lr_scaling=args.lr_scaling, method="muon", epochs=2,
                tag="preflight_timing")
    if r is None or "epoch_seconds" not in r:
        print("      timing run FAILED -- see the log above. Aborting.")
        return None
    sec = r["epoch_seconds"]
    print(f"      measured {sec:.1f}s per epoch")
    return sec


def _hardware(device: str) -> Dict:
    try:
        from common.hardware import describe
        return describe(device)
    except Exception as exc:                                # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def _hardware_sentence(info: Optional[Dict]) -> str:
    if not info:
        return "(not recorded)"
    try:
        from common.hardware import as_sentence
        return as_sentence(info)
    except Exception:                                       # noqa: BLE001
        return json.dumps(info)


def _tune_args(args) -> Namespace:
    """The Namespace ``centralized.tune.run_one`` expects."""
    return Namespace(
        dataset=args.dataset, model=args.model, batch_size=args.batch_size,
        momentum=args.momentum, weight_decay=args.weight_decay,
        weight_decay_mode=args.weight_decay_mode,
        head_adamw=args.head_adamw, last_k=args.last_k, val_seed=args.val_seed,
        seed=args.seed, device=args.device, data=args.data,
        num_workers=args.num_workers,
        nondeterministic=not args.deterministic,
        download=False,   # preflight fetches the data once, up front
    )


def job_cost(sec_per_epoch: float, epochs: int) -> float:
    """Estimated wall-clock of one run.

    ``sec_per_epoch`` is measured on the tuning split (45k train + 15k of eval).
    A ``full``-split run trains on 50k and evaluates only the 10k test set, which
    is close enough to the same per-epoch cost; the ``+1`` is the epoch-0
    evaluation.
    """
    return JOB_OVERHEAD_S + sec_per_epoch * (epochs + 1)


# --------------------------------------------------------------------------
# Phases
# --------------------------------------------------------------------------


def phase_gain(args, state, budget, sec) -> None:
    """Measure the growth of the accumulated update's gain.

    ``sqrt(t)`` growth means successive sign steps stay incoherent (alpha = 1/2,
    ``unit-gain``); linear ``t`` growth means they align (alpha = 1, ``mup``). This
    is the one measurement that decides the exponent without reference to accuracy.
    """
    for method in args.gain_methods:
        key = canonical_tag(f"gain_{method}", epochs=args.gain_epochs)
        if key in state["jobs"]:
            continue
        cost = job_cost(sec, args.gain_epochs)
        if not budget.fits(cost) or _stop["requested"]:
            return
        print(f"[{stamp()}] gain/{method} ({args.gain_epochs} ep, "
              f"~{cost/60:.0f} min) | {budget.report()}")
        # A third of the anchor: this is a diagnostic, not a performance run, and a
        # constant (un-annealed) rate at the full anchor risks instability, which
        # would corrupt the very series we are trying to fit.
        base = anchor_for(method, args.lr_scaling) / 3.0
        r = run_one(_tune_args(args), lr=base, lr_aux=args.lr_aux,
                    lr_scaling=args.lr_scaling, method=method,
                    epochs=args.gain_epochs, tag=f"gain_{method}",
                    extra=("--log-gain", "--constant-lr"))
        record(state, key, r)


def phase_aux(args, state, budget, sec) -> Dict[str, Dict]:
    """Is the optimal auxiliary rate method-independent?

    The auxiliary group is AdamW on the same parameters (biases, BatchNorm, head)
    for every method, so its optimum should not depend on the matrix rule. Two
    anchor methods an order of magnitude apart in eta_0 either agree or they do
    not, and verifying it earns the right to fix one ``lr_aux`` globally -- instead
    of a 2-D grid per method, which is ten times the cost, or an unverified
    assertion in the paper.

    Run at ``--aux-epochs`` rather than the reporting horizon, unlike ``lr``. What
    is at issue here is whether the *argmax over lr_aux* is the same for both
    methods, and both are measured at the same horizon, so a horizon bias cancels.
    """
    out: Dict[str, Dict] = state["phases"].setdefault("aux", {})
    for method in args.aux_methods:
        entry = out.setdefault(method, {"runs": []})
        for lr in round_grid(anchor_for(method, args.lr_scaling),
                             points=args.aux_lr_points):
            for lr_aux in AUX_GRID:
                tag = f"aux_{method}_lr{lr:.4g}_aux{lr_aux:.4g}"
                key = canonical_tag(tag, epochs=args.aux_epochs)
                if key in state["jobs"]:
                    continue
                cost = job_cost(sec, args.aux_epochs)
                if not budget.fits(cost) or _stop["requested"]:
                    return out
                print(f"[{stamp()}] aux/{method} (~{cost/60:.0f} min) "
                      f"| {budget.report()}")
                r = run_one(_tune_args(args), lr=lr, lr_aux=lr_aux,
                            lr_scaling=args.lr_scaling, method=method,
                            epochs=args.aux_epochs, tag=tag)
                # Append BEFORE recording: ``record`` persists the state, and
                # ``entry`` lives inside it. The other order leaves a window in
                # which the job is marked done but its result is not in ``runs``,
                # and a resume then skips the job and selects without it.
                if r:
                    entry["runs"].append(r)
                record(state, key, r)
        best = best_of(entry["runs"])
        if best:
            entry["best"] = best
            print(f"  BEST {method}: lr_aux={best['lr_aux']:.4g} "
                  f"(at eta_0={best['lr']:.4g}), val {best['val_acc']:.2f}%")
    _aux_verdict(out)
    return out


def _aux_verdict(out: Dict[str, Dict]) -> Optional[float]:
    """Print the agreement verdict; return the agreed ``lr_aux`` or ``None``."""
    chosen = {m: d["best"]["lr_aux"] for m, d in out.items() if d.get("best")}
    if len(chosen) < 2:
        return None
    print("\n  --- aux verdict ---")
    for m, v in chosen.items():
        print(f"    {m:<16} best lr_aux = {v:.4g}")
    if len(set(chosen.values())) == 1:
        v = next(iter(chosen.values()))
        print(f"    AGREE -> one lr_aux = {v:.4g} for every method, verified "
              f"rather than assumed.")
        return v
    print("    DISAGREE -> lr_aux must be tuned per method; give every method the "
          "same 2-D budget and say so in the paper.")
    return None


def _aux_profile(out: Dict[str, Dict]) -> Dict[str, Dict[float, float]]:
    """``{method: {lr_aux: best val_acc over the eta_0 grid}}``.

    The marginal over eta_0, which is the quantity the agreement claim is about:
    two methods can differ in where their eta_0 optimum sits and still share an
    ``lr_aux`` optimum, and only the marginal shows that.
    """
    prof: Dict[str, Dict[float, float]] = {}
    for method, d in out.items():
        col: Dict[float, float] = {}
        for r in d.get("runs") or []:
            if not r or "val_acc" not in r:
                continue
            key = float(r["lr_aux"])
            col[key] = max(col.get(key, -1e9), r["val_acc"])
        if col:
            prof[method] = col
    return prof


def phase_lr(args, state, budget, sec, rule: str) -> Dict[str, Dict]:
    """Tune eta_0 per method under ``rule``, identical budget for every method.

    At ``--final-epochs``, the horizon the table reports -- see the module
    docstring for why a short proxy was retired.
    """
    out: Dict[str, Dict] = state["phases"].setdefault("lr", {})
    for method in args.methods:
        entry = out.setdefault(method, {"runs": [], "grid": []})
        runs = entry["runs"]
        # A resumed run inherits the grid an earlier round had already widened to,
        # so the extension budget is not spent twice on the same method.
        grid = entry.get("grid") or round_grid(anchor_for(method, rule),
                                               points=args.lr_points)
        entry["grid"] = grid
        for extension in range(args.lr_extend_rounds + 1):
            for lr in grid:
                tag = f"lr_{method}_{rule.replace(':', '')}_{lr:.4g}"
                jkey = canonical_tag(tag, epochs=args.final_epochs)
                if jkey in state["jobs"]:
                    continue
                cost = job_cost(sec, args.final_epochs)
                if not budget.fits(cost) or _stop["requested"]:
                    return out
                print(f"[{stamp()}] lr/{method} (~{cost/60:.0f} min) "
                      f"| {budget.report()}")
                r = run_one(_tune_args(args), lr=lr, lr_aux=args.lr_aux,
                            lr_scaling=rule, method=method,
                            epochs=args.final_epochs, tag=tag)
                # Append before recording -- see phase_aux for why the order
                # matters to a resume.
                if r:
                    runs.append(r)
                record(state, jkey, r)
            best = best_of(runs)
            if best is None:
                break
            entry["best"] = best
            warn = boundary_warning(best, grid)
            print(f"  BEST {method}: eta_0={best['lr']:.6g}  "
                  f"val {best['val_acc']:.2f}%")
            # An optimum on an endpoint is not an optimum: widen the grid and keep
            # going rather than reporting a rate the grid boundary chose for us.
            if not warn:
                entry.pop("boundary", None)
                break
            print(warn)
            entry["boundary"] = warn.strip()
            if extension == args.lr_extend_rounds:
                break
            grid = extend_grid(grid, low="LOW end" in warn,
                               points=args.lr_extend_points)
            entry["grid"] = grid
            print(f"  -> extending to [{min(grid):.4g}, {max(grid):.4g}]: "
                  f"{args.lr_extend_points} more points "
                  f"(round {extension + 1}/{args.lr_extend_rounds})")
            save_state(state)
    return out


def phase_final(args, state, budget, sec, rule: str, tuned: Dict[str, Dict],
                on_run=None) -> None:
    """Full-50k runs at the tuned eta_0, seed-major so a cut night still completes
    a whole 1-seed table."""
    for seed in args.final_seeds:
        for method in args.methods:
            best = tuned.get(method, {}).get("best")
            if not best:
                continue
            tag = f"{method}_{rule.replace(':', '')}"
            jkey = canonical_tag(tag, epochs=args.final_epochs, split="full",
                                 seed=seed)
            if jkey in state["jobs"]:
                continue
            cost = job_cost(sec, args.final_epochs)
            if not budget.fits(cost) or _stop["requested"]:
                return
            print(f"[{stamp()}] final/{method} seed {seed} ({args.final_epochs} ep, "
                  f"~{cost/60:.0f} min) | {budget.report()}")
            r = run_one(_tune_args(args), lr=best["lr"], lr_aux=args.lr_aux,
                        lr_scaling=rule, method=method, epochs=args.final_epochs,
                        tag=tag, split="full", seed=seed)
            record(state, jkey, r)
            if r:
                state["phases"].setdefault("final", {})[jkey] = r
                save_state(state)
                if on_run is not None:
                    on_run()


def phase_wd(args, state, budget, sec, rule: str, tuned: Dict[str, Dict]) -> None:
    """Re-run the best few methods at the final horizon with decay switched on.

    The primary table uses ``--weight-decay 0``, so that the experiment and the
    theorems describe the same algorithm and no coupled/decoupled question arises.
    This phase supplies the other number: whether decay changes the *ordering*, and
    what the absolute accuracy is for readers who expect a regularized ResNet-18.
    The decay is decoupled -- the only well-posed choice for a scale-invariant step.
    """
    if args.wd_ablation <= 0 or not tuned:
        return
    ranked = sorted(((d["best"]["val_acc"], m, d["best"]["lr"])
                     for m, d in tuned.items() if d.get("best")), reverse=True)
    picks = ranked[:max(0, args.wd_ablation_top)]
    out = state["phases"].setdefault("wd", {})
    print(f"  decay ablation on {[m for _, m, _ in picks]} "
          f"at wd={args.wd_ablation:g} (decoupled)")
    for _, method, lr in picks:
        seed = args.final_seeds[0]
        tag = f"wd_{method}_{rule.replace(':', '')}"
        jkey = canonical_tag(tag, epochs=args.final_epochs, split="full", seed=seed)
        if jkey in state["jobs"]:
            continue
        cost = job_cost(sec, args.final_epochs)
        if not budget.fits(cost) or _stop["requested"]:
            return
        print(f"[{stamp()}] wd/{method} (~{cost / 60:.0f} min) | {budget.report()}")
        # ``extra`` lands at the END of the child argv, so these override the values
        # the driver already put there.
        r = run_one(_tune_args(args), lr=lr, lr_aux=args.lr_aux, lr_scaling=rule,
                    method=method, epochs=args.final_epochs, tag=tag, split="full",
                    seed=seed,
                    extra=("--weight-decay", repr(args.wd_ablation),
                           "--weight-decay-mode", "decoupled"))
        record(state, jkey, r)
        if r:
            # The undecayed counterpart of this exact configuration, for the delta.
            ref_key = canonical_tag(f"{method}_{rule.replace(':', '')}",
                                    epochs=args.final_epochs, split="full", seed=seed)
            ref = (state["phases"].get("final") or {}).get(ref_key) or {}
            out[method] = {"wd": args.wd_ablation, "lr": lr,
                           "test_acc": r.get("test_acc"),
                           "test_acc_no_decay": ref.get("test_acc")}
            save_state(state)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def _fit_gain_slope(job: Dict):
    """Least-squares slope of ``log(gain_median)`` against ``log(epoch)``.

    Returns ``(slope, r_squared, n_points)`` or ``None`` if the series is missing.
    Reads the run's own ``metrics.json`` rather than re-deriving anything, and
    delegates the fit to the exporter's, so the slope in this report and the slope
    in ``gain_fits.csv`` are the same function of the same data rather than two
    implementations that agree until one is edited.
    """
    path = job.get("metrics")
    if not path or not Path(path).exists():
        return None
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return fit_gain_slope(payload.get("history") or {})


def build_report(args, state, budget, sec, rule, tuned) -> str:
    lines = [
        "# Overnight run report",
        "",
        f"* started `{state.get('started')}`, wall clock {budget.report()}",
        (f"* device `{args.device}`, {args.model} / {args.dataset}, "
         f"measured **{sec:.1f} s/epoch**") if sec else "* timing unavailable",
        f"* machine: {_hardware_sentence(state.get('hardware'))}",
        f"* scaling rule **`{rule}`**, `--head-adamw {args.head_adamw}`, "
        f"lr_aux = {args.lr_aux:g}, momentum {args.momentum:g}, "
        f"weight decay {args.weight_decay:g} ({args.weight_decay_mode})",
        f"* eta_0 selected at **{args.final_epochs} epochs**, the horizon the table "
        f"reports, on the 45k/5k split, on **val_acc** (tail mean of {args.last_k}); "
        f"the test set was not used for any decision",
        "",
    ]

    gain = {k: v for k, v in state["jobs"].items() if k.startswith("gain_") and v}
    if gain:
        lines += ["## Gain diagnostic (measures the exponent directly)", "",
                  "Growth of the accumulated update's RMS gain "
                  "`||X_t - X_0||_F / sqrt(fan_out)` against the epoch count, at a "
                  "**constant** learning rate (with annealing the accumulation "
                  "saturates and the slope would measure the schedule instead).",
                  "",
                  "A slope near **0.5** means successive sign steps stay incoherent, "
                  "supporting `unit-gain` (alpha=1/2); near **1.0** means they align, "
                  "supporting `mup` (alpha=1).", "",
                  "| run | fitted slope | R^2 | epochs used | reading |",
                  "| :--- | ---: | ---: | ---: | :--- |"]
        for k, v in sorted(gain.items()):
            fit = _fit_gain_slope(v)
            if fit is None:
                lines.append(f"| `{k}` | - | - | - | series unavailable "
                             f"(see `{Path(v['log']).name}`) |")
                continue
            slope, r2, n = fit
            reading = ("incoherent -> alpha=1/2" if slope < 0.7 else
                       "aligned -> alpha=1" if slope > 0.85 else "between 1/2 and 1")
            lines.append(f"| `{k}` | {slope:.3f} | {r2:.3f} | {n} | {reading} |")
        lines.append("")

    aux = state["phases"].get("aux") or {}
    profile = _aux_profile(aux)
    if profile:
        rates = sorted({v for col in profile.values() for v in col})
        lines += [f"## Auxiliary rate (is `lr_aux` method-independent?)", "",
                  f"Best `val_acc` over the eta_0 grid at each `lr_aux`, "
                  f"{args.aux_epochs} epochs. The comparison is between the "
                  f"**columns' argmaxes**: both methods are measured at the same "
                  f"horizon, so a horizon bias cancels here in a way it does not "
                  f"for eta_0.", "",
                  "| method | " + " | ".join(f"{v:g}" for v in rates) + " | argmax |",
                  "| :--- | " + " | ".join("---:" for _ in rates) + " | ---: |"]
        for method, col in profile.items():
            cells = [f"{col[v]:.2f}" if v in col else "-" for v in rates]
            arg = max(col, key=lambda v: col[v])
            lines.append(f"| `{method}` | " + " | ".join(cells) + f" | {arg:g} |")
        argmaxes = {max(col, key=lambda v: col[v]) for col in profile.values()}
        lines += ["",
                  (f"**AGREE** -- both methods select `lr_aux = "
                   f"{next(iter(argmaxes)):g}`, so one value may be held fixed for "
                   f"every method and reported as verified."
                   if len(argmaxes) == 1 and len(profile) > 1 else
                   "**DISAGREE** -- the optimum depends on the matrix rule, so "
                   "`lr_aux` must be tuned per method at an equal 2-D budget."),
                  ""]

    if tuned:
        lines += [f"## Tuned eta_0 ({args.final_epochs} epochs, validation)", "",
                  "| method | eta_0 | val acc | configs | note |",
                  "| :--- | ---: | ---: | ---: | :--- |"]
        for method, d in tuned.items():
            b = d.get("best")
            if not b:
                continue
            lines.append(f"| `{method}` | {b['lr']:.6g} | {b['val_acc']:.2f}% | "
                         f"{len(d.get('runs') or [])} | {d.get('boundary', '')} |")
        lines += ["", "A `BOUNDARY` note means the optimum sat on a grid endpoint "
                      "and the grid ran out of extension rounds: widen it and re-run "
                      "that method before reporting it.", ""]

    finals = state["phases"].get("final", {})
    if finals:
        lines += ["## Final runs (full 50k, test set)", "",
                  "| run | test acc (tail mean) | epochs to target |",
                  "| :--- | ---: | ---: |"]
        for tag, v in sorted(finals.items()):
            if not v:
                continue
            lines.append(f"| `{tag}` | {v.get('test_acc', float('nan')):.2f}% | "
                         f"{v.get('epochs_to_target', '-')} |")
        lines += ["", "The paper's table, aggregated over seeds exactly as it "
                      "defines them, is `table_cifar.csv` in "
                      "`python3 -m centralized.export_article`.", ""]

    wd = state["phases"].get("wd", {})
    if wd:
        lines += [f"## Weight-decay ablation (decoupled, wd = {args.wd_ablation:g})", "",
                  f"The primary table above is unregularized (`--weight-decay "
                  f"{args.weight_decay:g}`), which is the setting the theorems analyse "
                  f"and the one both reference implementations use. These runs repeat "
                  f"the best methods at seed {args.final_seeds[0]} with decoupled decay "
                  f"on the matrix parameters, at the *same* eta_0. What matters is "
                  f"whether the ordering moves, not the absolute gain -- eta_0 was not "
                  f"re-tuned under decay.", "",
                  "| method | eta_0 | with decay | no decay | delta |",
                  "| :--- | ---: | ---: | ---: | ---: |"]
        for k, v in sorted(wd.items()):
            got, ref = v.get("test_acc"), v.get("test_acc_no_decay")
            cells = [f"`{k}`", f"{v['lr']:.6g}",
                     "-" if got is None else f"{got:.2f}%",
                     "-" if ref is None else f"{ref:.2f}%",
                     "-" if (got is None or ref is None) else f"{got - ref:+.2f}"]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    done = sum(1 for v in state["jobs"].values() if v)
    failed = sum(1 for v in state["jobs"].values() if not v)
    lines += ["## What this run did NOT establish", ""]
    if not profile:
        # Only claim the auxiliary rate was verified when the phase actually ran.
        # The previous protocol asserted the check in the documentation while the
        # phase had never been executed; a conditional caveat is what keeps the two
        # from drifting apart again.
        lines.append(
            f"* **`lr_aux` was fixed at {args.lr_aux:g}**, not tuned, and not verified "
            f"to be method-independent -- the `aux` phase did not run. The auxiliary "
            f"group is AdamW on the same parameters for every method, so its optimum "
            f"should not depend on the matrix rule, but that is an argument rather "
            f"than a measurement. Add `aux` to `--phases`, or run "
            f"`python3 -m centralized.tune --stage aux --epochs {args.aux_epochs}`.")
    lines += [f"* **Momentum ({args.momentum:g}) and weight decay "
              f"({args.weight_decay:g}) were held fixed** for every method, so this is "
              f"a comparison at equal momentum, not at each method's own optimum.",
              f"* Weight decay is **{args.weight_decay_mode}**"
              + (" (`X *= 1 - lr*wd`, the LMO sees the true gradient). The coupled "
                 "convention -- `wd*X` added to the gradient, which is what "
                 "Mishra et al.'s Algorithm 1 and our own earlier numbers used -- is "
                 "*not* an alternative worth reporting as an equal: every step "
                 "direction here is scale-invariant, so coupled decay shrinks nothing "
                 "and only rotates the direction. `--weight-decay-mode coupled` "
                 "reproduces it for the appendix ablation."
                 if args.weight_decay_mode == "decoupled" else
                 " -- `wd*X` is folded into the gradient. Every step direction in this "
                 "code is scale-invariant, so this shrinks nothing and only rotates the "
                 "direction, by an amount set by the drifting, method-dependent ratio "
                 "`wd*||X||/||G||`. Prefer `--weight-decay-mode decoupled`."),
              "* **A gap smaller than the seed spread is not a result.** Add seeds with "
              "`--resume` and aggregate before claiming one.",
              "",
              "## Next steps", "",
              f"{done} jobs completed, {failed} failed.", "",
              "```bash",
              f"python3 -m centralized.overnight --device {args.device} --resume "
              f"  # continue",
              "python3 -m centralized.export_article"
              "                # pack results/article_export.tar.gz",
              "```", ""]
    if not finals:
        lines += ["No final runs completed. With the tuned eta_0 above, launch them "
                  "directly:", "",
                  "```bash",
                  f"python3 -m centralized.main --dataset {args.dataset} "
                  f"--model {args.model} --epochs {args.final_epochs} \\",
                  f"    --optimizer <method> --lr-scaling {rule} "
                  f"--head-adamw {args.head_adamw} \\",
                  f"    --lr <eta_0> --lr-aux {args.lr_aux:g} --seed 0 "
                  f"--device {args.device}",
                  "```", ""]

    return "\n".join(lines)


def refresh_report(args, state, budget, sec, rule, tuned, *,
                   echo: bool = False) -> None:
    """Rewrite ``REPORT.md`` from the current state.

    Called after every phase and after every final run, so the report on disk is
    always current: with no deadline the run may last more than a day, and you
    should be able to read what has been found without stopping it.
    """
    text = build_report(args, state, budget, sec, rule, tuned)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text, encoding="utf-8")
    if echo:
        print("\n" + "=" * 78)
        print(text)
        print("=" * 78)
    print(f"[{stamp()}] report refreshed -> {REPORT_PATH}", flush=True)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--budget-hours", type=float, default=0.0,
                   help="Wall-clock deadline; 0 (the default) means NO deadline -- every "
                        "phase runs to completion and only Ctrl-C stops it "
                        "(the report is written on the way out)")
    p.add_argument("--data", type=str, default="./data")
    p.add_argument("--download", action="store_true",
                   help="Download CIFAR-10 if missing (do this on the first run)")
    p.add_argument("--dataset", type=str, default="cifar10")
    p.add_argument("--model", type=str, default="resnet18")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="Applied to the MATRIX parameters only (the auxiliary group is "
                        "never decayed). Defaults to 0: our theorems analyse "
                        "unregularized f, the nanoGPT record we build on uses 0.0 for "
                        "every group, and all ten of Mishra et al.'s best CIFAR "
                        "configurations select 0 -- so 0 is the setting under which "
                        "theory and experiment describe the same algorithm. The `wd` "
                        "phase re-runs the top methods at 5e-4 as an ablation.")
    p.add_argument("--wd-ablation", type=float, default=5e-4,
                   help="Decay rate for the `wd` phase, which re-runs the "
                        "best --wd-ablation-top methods at the final horizon "
                        "with decay switched on. 0 disables the phase.")
    p.add_argument("--wd-ablation-top", type=int, default=3,
                   help="How many of the tuned methods to re-run with decay.")
    p.add_argument("--weight-decay-mode", type=str, default="decoupled",
                   choices=["decoupled", "coupled"],
                   help="decoupled (default): X *= 1 - lr*wd, leaving the LMO to see "
                        "the true gradient. coupled folds wd*X into the gradient, "
                        "which cannot shrink a scale-invariant step at all and only "
                        "rotates it -- kept for the appendix ablation.")
    p.add_argument("--lr-aux", type=float, default=1e-3,
                   help="Auxiliary AdamW rate, held fixed for every method. The `aux` "
                        "phase is the check that it may be.")
    p.add_argument("--head-adamw", type=str, default="always",
                   choices=["auto", "always", "never"])
    p.add_argument("--lr-scaling", type=str, default="unit-gain")
    p.add_argument("--last-k", type=int, default=5)
    p.add_argument("--val-seed", type=int, default=12345)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=2,
                   help="DataLoader workers for training (Windows spawn is costly)")
    p.add_argument("--no-export", action="store_true",
                   help="Skip the closing 'centralized.export_article' step. The "
                        "bundle is the file you download, so this is for debugging")
    p.add_argument("--deterministic", action="store_true",
                   help="Disable cuDNN autotuning: bitwise reproducible but slower. "
                        "Off by default -- with multiple seeds we measure seed "
                        "variation, not bitwise determinism")

    p.add_argument("--methods", nargs="*", default=ALL_METHODS)
    p.add_argument("--phases", nargs="*", default=list(PHASES), choices=list(PHASES),
                   help="Which phases to run. They always run in the canonical order "
                        f"{' -> '.join(PHASES)}, because each consumes the previous "
                        f"one's output; this flag selects, it does not reorder.")
    p.add_argument("--gain-epochs", type=int, default=20)
    p.add_argument("--gain-methods", nargs="*",
                   default=["signmuon", "muonsign", "signsgd", "muon"],
                   help="Methods to run the --log-gain diagnostic on. The exponent "
                        "is only open for the SIGN family -- for the LMO family "
                        "unit-gain and mup are the same multiplier -- so the sign "
                        "three are the measurement and muon is the reference.")
    p.add_argument("--aux-epochs", type=int, default=15,
                   help="Horizon for the `aux` phase. Short on purpose: the claim "
                        "being tested is that two methods pick the SAME lr_aux, and "
                        "both are measured here, so a horizon bias cancels.")
    p.add_argument("--aux-methods", nargs="*", default=["signmuon", "muon"],
                   help="Anchor methods for the `aux` phase -- pick two an order of "
                        "magnitude apart in eta_0, so agreement is informative.")
    p.add_argument("--aux-lr-points", type=int, default=3,
                   help="eta_0 lattice points per anchor method in the `aux` phase.")
    p.add_argument("--final-epochs", type=int, default=75,
                   help="The reporting horizon: the budget for the full-50k runs, and "
                        "also the horizon eta_0 is selected at. 75 matches the paper")
    p.add_argument("--final-seeds", nargs="*", type=int, default=[0, 1, 2],
                   help="Seed-major: all methods at seed 0, then seed 1, ... "
                        "so an interrupted night still leaves a complete table")
    p.add_argument("--lr-points", type=int, default=5,
                   help="Lattice points per method: 3 per decade, so 5 spans "
                        "~1.3 decades and 7 spans ~2.")
    p.add_argument("--lr-extend-rounds", type=int, default=4,
                   help="When a method's optimum lands on a grid endpoint, widen the "
                        "grid in that direction and re-tune, up to this many times. "
                        "0 restores the old behaviour of reporting the endpoint with "
                        "a warning.")
    p.add_argument("--lr-extend-points", type=int, default=2,
                   help="Points added per extension round.")
    p.add_argument("--report-only", action="store_true",
                   help="Rebuild REPORT.md from state.json and exit: runs nothing, "
                        "needs no GPU, and is safe to call while a run is in flight.")

    p.add_argument("--resume", action="store_true", help="Skip jobs already recorded")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="Continue even if the CPU test suite fails")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the schedule and exit without training")
    return p.parse_args()


def print_schedule(args, sec: float, budget: Budget) -> None:
    n_methods = len(args.methods)
    plan = [
        ("gain", len(args.gain_methods), args.gain_epochs),
        ("aux", len(args.aux_methods) * args.aux_lr_points * len(AUX_GRID),
         args.aux_epochs),
        ("lr", n_methods * args.lr_points, args.final_epochs),
        ("final", n_methods * len(args.final_seeds), args.final_epochs),
        ("wd", 0 if args.wd_ablation <= 0 else args.wd_ablation_top,
         args.final_epochs),
    ]
    # The lr phase can widen a method's grid when its optimum lands on an endpoint,
    # which is extra work the plan cannot know about in advance.
    extra_lr = n_methods * args.lr_extend_rounds * args.lr_extend_points
    plan = [(n, j, e) for n, j, e in plan if j > 0 and n in args.phases]
    print("\n" + "=" * 78)
    budget_desc = ("no deadline -- runs until Ctrl-C" if budget.unlimited
                   else f"budget {args.budget_hours:g} h")
    print(f"SCHEDULE  ({budget_desc}, measured {sec:.1f} s/epoch)")
    print("=" * 78)
    last_col = "done by" if budget.unlimited else "fits?"
    print(f"  {'phase':<8}{'jobs':>6}{'epochs':>8}{'hours':>8}{'cumulative':>12}"
          f"   {last_col}")
    cum = 0.0
    for name, n, ep in plan:
        hrs = n * job_cost(sec, ep) / 3600.0
        cum += hrs
        if budget.unlimited:
            status = (datetime.now() + timedelta(hours=cum)).strftime("%a %H:%M")
        else:
            status = "yes" if cum <= args.budget_hours else "NO -- will be cut"
        print(f"  {name:<8}{n:>6}{ep:>8}{hrs:>8.1f}{cum:>12.1f}   {status}")
    print("=" * 78)
    if extra_lr and "lr" in args.phases:
        print(f"  Plus up to {extra_lr} more lr jobs "
              f"(+{extra_lr * job_cost(sec, args.final_epochs) / 3600.0:.1f} h) if "
              f"optima land on grid endpoints:\n  the grid is then widened and that "
              f"method re-tuned, rather than reporting the endpoint.")
    if budget.unlimited:
        eta = (datetime.now() + timedelta(hours=cum)).strftime("%a %H:%M")
        print(f"  Everything runs to completion: {cum:.1f} h total, finishing around "
              f"{eta}.")
        print(f"  Ctrl-C at any point stops cleanly and writes the report. The report "
              f"on disk is\n  refreshed after every phase and after every final run, so "
              f"you can read it mid-run.")
        print(f"  Finals are seed-major: all {n_methods} methods at seed "
              f"{args.final_seeds[0]} first, then the next seed --\n  so stopping "
              f"early leaves complete tables, never fragments.")
        print(flush=True)
        return
    if cum > args.budget_hours:
        n_final = n_methods * len(args.final_seeds)
        final_h = (n_final * job_cost(sec, args.final_epochs) / 3600.0
                   if "final" in args.phases else 0.0)
        tuning_h = cum - final_h
        spare_s = (args.budget_hours - tuning_h) * 3600.0
        print(f"  Total {cum:.1f} h exceeds the {args.budget_hours:g} h budget. The run "
              f"stops cleanly at the deadline;")
        print(f"  phases are ordered so that what completes is still usable, and "
              f"--resume continues tomorrow.")
        if final_h > 0 and spare_s > 0:
            n_fit = int(spare_s // job_cost(sec, args.final_epochs))
            print(f"  Tuning alone needs {tuning_h:.1f} h, leaving {spare_s / 3600:.1f} h: "
                  f"about {n_fit} of {n_final} final runs will")
            print(f"  complete (seed-major, so whole methods finish rather than "
                  f"fragments).")
        else:
            print(f"  Tuning alone needs {tuning_h:.1f} h. Reduce --lr-points, drop a "
                  f"phase, or raise --budget-hours.")
    else:
        print(f"  Total {cum:.1f} h fits with "
              f"{args.budget_hours - cum:.1f} h to spare.")
    print(flush=True)


def main() -> None:
    args = get_args()
    signal.signal(signal.SIGINT, _handle_sigint)

    state = load_state() if args.resume or args.report_only else {"jobs": {}, "phases": {}}
    state.setdefault("jobs", {})
    state.setdefault("phases", {})
    state["started"] = state.get("started") or datetime.now().isoformat(timespec="seconds")
    budget = Budget(args.budget_hours)
    rule = args.lr_scaling

    if args.report_only:
        # Read-only: rebuild the report from whatever is on disk and leave the
        # state file alone, so this is safe to run while a job is training.
        text = build_report(args, state, budget, state.get("epoch_seconds") or 0.0,
                            rule, state["phases"].get("lr", {}))
        print(text)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(text, encoding="utf-8")
        print(f"\n[{stamp()}] rewrote {REPORT_PATH}")
        return

    if args.download:
        # Fetch once, here, rather than letting every child run pass --download:
        # a partially written archive shared by many runs is a bad failure mode.
        print(f"[{stamp()}] fetching {args.dataset} into {args.data} ...")
        from centralized.data import build_loaders
        build_loaders(args.dataset, args.data, batch_size=args.batch_size,
                      download=True, split="tune", num_workers=0)
        print(f"[{stamp()}] dataset ready")

    # Resuming across a change of convention would silently mix runs that are not
    # comparable: the cached jobs are keyed by method and rate only.
    fingerprint = {
        "weight_decay_mode": args.weight_decay_mode,
        "weight_decay": args.weight_decay,
        "momentum": args.momentum,
        "head_adamw": args.head_adamw,
        "lr_aux": args.lr_aux,
        "lr_scaling": args.lr_scaling,
        "final_epochs": args.final_epochs,
        "dataset": args.dataset,
        "model": args.model,
    }
    old = state.get("fingerprint")
    if args.resume and old and old != fingerprint:
        differs = {k: (old.get(k), v) for k, v in fingerprint.items() if old.get(k) != v}
        print("REFUSING to resume: this run's settings differ from the recorded ones,")
        print("so the cached jobs are not comparable with the new ones.")
        for k, (was, now) in differs.items():
            print(f"  {k}: recorded {was!r} -> requested {now!r}")
        print(f"\nStart a fresh sweep (drop --resume, which overwrites "
              f"{STATE_PATH.name}), or move the old results aside first. Use "
              f"--report-only to read the existing report without running anything.")
        sys.exit(1)
    state["fingerprint"] = fingerprint

    sec = state.get("epoch_seconds") if args.resume else None
    if sec is None:
        sec = preflight(args, state)
        if sec is None:
            print("Preflight failed; nothing was trained.")
            sys.exit(1)
        state["epoch_seconds"] = sec
        save_state(state)
    else:
        print(f"[{stamp()}] resuming with the previously measured {sec:.1f} s/epoch "
              f"({len(state['jobs'])} jobs already recorded)")

    print_schedule(args, sec, budget)
    if args.preflight_only or args.dry_run:
        print("Stopping here as requested (--preflight-only / --dry-run).")
        return

    tuned: Dict[str, Dict] = {}
    if budget.unlimited:
        print(f"[{stamp()}] no deadline: every phase runs to completion. Ctrl-C stops "
              f"cleanly and writes the report; the report on disk is refreshed after "
              f"every phase, so you can read it at any time without stopping.")

    def refresh(echo: bool = False) -> None:
        refresh_report(args, state, budget, sec, rule,
                       tuned or state["phases"].get("lr", {}), echo=echo)

    def banner(name: str, note: str = "") -> None:
        print(f"\n{'=' * 78}\n[{stamp()}] PHASE {name}{note}\n{'=' * 78}", flush=True)

    # A second Ctrl-C raises out of the handler; the report must still be written,
    # so the phases run inside try/finally rather than being trusted to complete.
    try:
        if "gain" in args.phases:
            banner("gain", f"  (constant LR, {args.gain_epochs} epochs)")
            phase_gain(args, state, budget, sec)
            refresh()
        if "aux" in args.phases and not _stop["requested"]:
            banner("aux", f"  (lr_aux independence, {args.aux_epochs} epochs)")
            phase_aux(args, state, budget, sec)
            refresh()
        if "lr" in args.phases and not _stop["requested"]:
            banner("lr", f"  (rule '{rule}', {args.final_epochs} epochs -- the "
                         f"reporting horizon)")
            tuned = phase_lr(args, state, budget, sec, rule)
            refresh()
        if "final" in args.phases and not _stop["requested"]:
            banner("final", f"  ({args.final_epochs} epochs, full 50k, "
                            f"seeds {args.final_seeds})")
            phase_final(args, state, budget, sec, rule,
                        tuned or state["phases"].get("lr", {}), on_run=refresh)
            refresh()
        if "wd" in args.phases and not _stop["requested"]:
            banner("wd", f"  (decay ablation, wd={args.wd_ablation:g} decoupled)")
            phase_wd(args, state, budget, sec, rule,
                     tuned or state["phases"].get("lr", {}))
    except KeyboardInterrupt:
        print(f"\n[{stamp()}] interrupted -- writing the report from what completed.")
    finally:
        save_state(state)
        refresh(echo=True)
        if not args.no_export:
            export()


def export() -> None:
    """Pack the results and print the one file to download.

    Runs even after an interrupt: a sweep that stopped at seed 2 still produced a
    table worth carrying home, and the bundle's MANIFEST says how partial it is.
    Never fatal -- the runs are already on disk, and losing the archive step should
    not make the sweep look like a failure.
    """
    from centralized import export_article
    try:
        print(f"\n{'=' * 78}\n[{stamp()}] packing the results\n{'=' * 78}", flush=True)
        export_article.main([])
    except SystemExit:
        raise
    except Exception as exc:                                 # noqa: BLE001
        print(f"[{stamp()}] could not write the bundle: {type(exc).__name__}: {exc}")
        print("The runs themselves are intact under results/centralized; "
              "run 'python3 -m centralized.export_article' by hand.")


if __name__ == "__main__":
    main()
