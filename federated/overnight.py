"""One-command overnight run for the federated CNN2 study (Table 5).

    cd code
    python3 -m federated.overnight --device cuda:0 --budget-hours 12 --download

Watch the first few minutes: it runs the CPU test suite, prints the per-layer
learning-rate table and the transported grid anchors, times a real 40-round run
on *your* GPU, and then prints a schedule with a finish time per phase. Once the
schedule appears you can leave it.

The federated twin of ``centralized/overnight.py``, and deliberately the same
shape: budget-aware, crash-isolated (each job is a subprocess), resumable
(``--resume``), priority-ordered, and with ``results/federated_overnight/REPORT.md``
rewritten after every phase so it can be read mid-run.

What is different from the centralized night
--------------------------------------------
* **Selection has somewhere to happen.** ``--split tune`` holds 5k images out of
  the 50k *before* the client partition -- the same 5k the centralized path holds
  out -- and a tuning child never evaluates the test set. The learning rates in the
  published Table 5 came from ``federated/grid.py``, which ranked configurations
  by test accuracy.
* **Anchors are transported, not guessed.** The published rates were tuned under
  the ``legacy`` rule; ``federated.tune.anchor_for`` moves each one to the chosen
  rule by the ratio of geometric-mean per-layer multipliers. On CNN2 that leaves
  the LMO family alone and moves the sign family by ~29x.
* **The schedule rebalances itself.** The cost of a round is not known until it is
  measured on the actual GPU, so ``--budget-hours`` is met by *adjusting the plan*
  rather than by being cut halfway: seeds, then the weight-decay ablation, then
  the tuning horizon give way, in that order, and the reverse when there is
  slack. ``--budget-hours 0`` disables this and runs everything.
* **No ``gain`` phase.** The accumulated-update exponent is a property of the
  architecture and the step family, not of the federation, and it is measured
  centrally on ResNet-18. CNN2 would be the *better* instrument for it -- its
  sign-family multipliers span 7.8x against ResNet-18's ~5x, and it has no layer
  shape holding 63% of the parameters -- so if the centralized measurement comes
  out ambiguous, ``--phases alpha`` is the place to repeat it.

Phases
------
1. ``lr``      eta_0 per method under the chosen rule, equal budget each, on a
               1-2-5 lattice, with the grid widened when an optimum lands on an
               endpoint.
2. ``verify``  re-run the top rates at the FINAL horizon: is the short-horizon
               ranking horizon-stable? (the assumption a short proxy makes)
3. ``final``   full-50k runs at the tuned rates, **seed-major** -- every method at
               seed 0 before any of them reaches seed 1, so an early stop leaves a
               complete 1-seed table rather than a fragmentary 3-seed one.
4. ``wd``      re-run the best few with weight decay switched on. The primary
               table is unregularized, matching the centralized protocol and the
               setting the theorems analyse.
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
from typing import Dict, List, Optional

from common.lr_scaling import describe_rule, resolve_rule
from common.paths import scrub
from common.utils import results_root
from federated.algorithms import method_family
from federated.tune import (ALL_METHODS, ROOT, add_common_args, anchor_for, best_of,
                            boundary_warning, canonical_tag, describe_anchors,
                            extend_grid, matrix_shapes, round_grid, run_one)

OUT_DIR = results_root() / "federated_overnight"
STATE_PATH = OUT_DIR / "state.json"
REPORT_PATH = OUT_DIR / "REPORT.md"

#: Every phase, in the order they are run and must be re-sorted into. **The single
#: source of this list.** It used to be spelled out in four places, and `rules` was
#: added to three of them: the fourth silently deleted the phase from any run that
#: set --budget-hours, requested on the command line or not.
PHASE_ORDER = ("lr", "verify", "final", "rules", "wd")

#: Run by default. `rules` is opt-in because it costs ~4 h and answers a question
#: about the per-layer rule rather than about the methods.
DEFAULT_PHASES = ("lr", "verify", "final", "wd")

#: How a phase is named in the report when rebalancing gives it up. Every
#: sheddable phase needs an entry, so that dropping one is always something the
#: report says in words rather than a difference the reader has to notice.
PHASE_LABELS = {
    "verify": "the horizon-stability check",
    "rules": "the per-layer rule ablation",
    "wd": "the weight-decay ablation",
}

#: Startup cost of one subprocess (imports, CUDA init, dataset upload), seconds.
JOB_OVERHEAD_S = 40.0

_stop = {"requested": False}


def _handle_sigint(signum, frame):        # pragma: no cover - interactive
    if _stop["requested"]:
        raise KeyboardInterrupt("second interrupt -- exiting now")
    _stop["requested"] = True
    print("\n[interrupt] stopping cleanly and writing the report. "
          "Press Ctrl-C again to exit immediately.", flush=True)


def stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------


def load_state() -> Dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"[warn] {STATE_PATH} is corrupt; starting fresh")
    return {"jobs": {}, "phases": {}, "started": None, "round_seconds": None}


def save_state(state: Dict) -> None:
    """Persist the driver state, with machine-specific paths taken out.

    Scrubbed on the way to disk rather than in memory: the driver reopens
    ``job["metrics"]`` to refit a diagnostic, and both drivers run from ``code/``,
    so a `results/...` path resolves on resume just as an absolute one did. Now
    that `results/` ships with the code, `state.json` is a file reviewers read.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(scrub(state), indent=2), encoding="utf-8")


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


class Budget:
    """Deadline bookkeeping, with a per-job feasibility check."""

    def __init__(self, hours: float):
        self.start = time.time()
        self.hours = hours
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


def _tune_args(args) -> Namespace:
    """The Namespace ``federated.tune.run_one`` expects."""
    return Namespace(
        dataset=args.dataset, model=args.model, n_parties=args.n_parties,
        n_steps=args.n_steps, batch_size=args.batch_size, momentum=args.momentum,
        weight_decay=args.weight_decay, partition=args.partition, beta=args.beta,
        lr_aux=args.lr_aux, lr_scaling=args.lr_scaling, last_k=args.last_k,
        eval_points=args.eval_points, target_acc=args.target_acc,
        val_seed=args.val_seed, seed=args.seed, device=args.device, data=args.data,
        loader=args.loader, mv_ties=args.mv_ties, uplink_zeros=args.uplink_zeros,
        nondeterministic=args.nondeterministic,
        download=False,        # preflight fetches the data once, up front
    )


def preflight(args) -> Optional[tuple]:
    """Tests, the scaling tables, and a measured cost model.

    Returns ``(seconds_per_round, fixed_overhead)`` or ``None`` if the timing runs
    failed. Both terms are **measured**, with two runs rather than one:

    * ``--rounds 0`` does the start-up (imports, CUDA init, dataset upload) and the
      round-0 evaluation, and nothing else, so its wall clock *is* the fixed
      overhead;
    * ``--rounds R`` adds the training, so ``(wall_R - wall_0) / R`` is the true
      marginal cost of a round, evaluation amortized in.

    Assuming a constant overhead instead is how the first version of this got a
    negative marginal cost on a fast machine, silently clamped it, and then
    reported that everything fits in no time at all.
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
            elif emit and (line.startswith((" ", "	")) and line.strip()):
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

    print(f"\n[2/4] per-layer multipliers, rule '{args.lr_scaling}'")
    rule = resolve_rule(args.lr_scaling)
    shapes = matrix_shapes(args.dataset, args.model)
    for family in ("sign", "lmo"):
        print(describe_rule(rule, family, shapes))

    print(f"\n[3/4] grid anchors")
    print(describe_anchors(args.lr_scaling, args.dataset, args.model, args.methods))

    print(f"\n[4/4] timing {args.model} ({args.n_parties} clients x {args.n_steps} "
          f"steps) on {args.device}: start-up, then {args.timing_rounds} rounds")
    lr = anchor_for("muon", args.lr_scaling, shapes)
    zero = run_one(_tune_args(args), lr=lr, lr_aux=args.lr_aux,
                   lr_scaling=args.lr_scaling, method="muon", rounds=0,
                   tag="preflight_startup")
    timed = run_one(_tune_args(args), lr=lr, lr_aux=args.lr_aux,
                    lr_scaling=args.lr_scaling, method="muon",
                    rounds=args.timing_rounds, tag="preflight_timing")
    if timed is None or "wall_seconds" not in timed:
        print("      timing run FAILED -- see the log above. Aborting.")
        return None

    overhead = zero["wall_seconds"] if zero and "wall_seconds" in zero else JOB_OVERHEAD_S
    per_round = (timed["wall_seconds"] - overhead) / args.timing_rounds
    if per_round <= 0:
        # The rounds were lost in the start-up noise; fall back to the child's own
        # per-round measurement, which excludes evaluation but is never negative.
        per_round = timed.get("round_seconds") or 1e-3
        print(f"      (the {args.timing_rounds} rounds were within the start-up noise; "
              f"falling back to the child's own round timer)")
    print(f"      start-up {overhead:.0f}s, {timed['wall_seconds']:.0f}s for "
          f"{args.timing_rounds} rounds -> {per_round:.3f}s/round marginal "
          f"({timed.get('round_seconds', float('nan')):.3f}s/round training only)")
    return per_round, overhead


#: Measured fixed cost of one job, filled in by ``preflight``. A module-level
#: value rather than a threaded argument because it is one property of one
#: machine, read by every cost estimate in this file.
_OVERHEAD = {"seconds": JOB_OVERHEAD_S}


def job_cost(sec_per_round: float, rounds: int) -> float:
    return _OVERHEAD["seconds"] + sec_per_round * rounds


# --------------------------------------------------------------------------
# Auto-balancing the plan to the budget
# --------------------------------------------------------------------------


def _lr_jobs(args):
    """``(method, scale_baselines)`` pairs.

    **Adam** is additionally run under the sign-family rule: its step is
    approximately a sign step and muP does prescribe a per-layer rate for it, so
    reporting the better of the two removes any suspicion that the baseline was
    handicapped. **SGD deliberately is not** -- its step is ``eta * m``, whose
    norm is data-dependent, so no static multiplier implements the unit-gain
    criterion and applying one would be an arbitrary rescaling.
    """
    for method in args.methods:
        yield method, False
        if method == "adam" and not args.skip_baseline_variants:
            yield method, True


def plan_cost(args, sec: float) -> Dict[str, float]:
    n_par = sum(1 for _ in _lr_jobs(args))
    n_rules = sum(1 for r in args.rule_alternatives if r != args.lr_scaling)
    n_rule_cfg = n_rules * len(args.rule_methods)
    return {
        "lr": n_par * args.lr_points * job_cost(sec, args.tune_rounds),
        # Vacuous, and skipped, once the tuning horizon is the reporting one.
        "verify": ((len(args.verify_methods) * args.verify_top
                    * job_cost(sec, args.final_rounds))
                   if args.tune_rounds < args.final_rounds else 0.0),
        "final": n_par * len(args.final_seeds) * job_cost(sec, args.final_rounds),
        "rules": (n_rule_cfg * args.lr_points * job_cost(sec, args.tune_rounds)
                  + n_rule_cfg * len(args.rule_seeds) * job_cost(sec, args.final_rounds)
                  if "rules" in args.phases else 0.0),
        "wd": (args.wd_ablation_top * job_cost(sec, args.final_rounds)
               if args.wd_ablation > 0 else 0.0),
    }


def autobalance(args, sec: float, budget: Budget) -> List[str]:
    """Fit the plan to the budget by *adjusting* it, not by being cut.

    A federated round on an unseen GPU can differ by an order of magnitude from
    the reference machine, so a fixed plan either overruns (and the finals are
    truncated, which is the expensive part) or leaves the GPU idle. The ordering
    of concessions is the same one the phase ordering encodes: give up error bars
    before the table, the ablation before the error bars, and the tuning
    resolution last -- a badly tuned rate poisons everything downstream.

    Returns a list of human-readable adjustments for the report.
    """
    notes: List[str] = []
    if budget.unlimited:
        return notes

    def total() -> float:
        c = plan_cost(args, sec)
        return sum(v for k, v in c.items() if k in args.phases)

    limit = args.budget_hours * 3600.0 * args.budget_headroom

    # --- shed work, cheapest-to-lose first ---------------------------------
    # Drops are recorded but not announced yet: the restore pass below may put a
    # phase back, and "dropped X / kept X after all" is a worse report than
    # saying nothing about X.
    dropped: set = set()
    if total() > limit and args.wd_ablation > 0 and "wd" in args.phases:
        args.phases = [p for p in args.phases if p != "wd"]
        dropped.add("wd")
    if total() > limit and "rules" in args.phases:
        args.phases = [p for p in args.phases if p != "rules"]
        dropped.add("rules")
    if total() > limit and "verify" in args.phases:
        args.phases = [p for p in args.phases if p != "verify"]
        dropped.add("verify")
    seeds_before = len(args.final_seeds)
    while total() > limit and len(args.final_seeds) > 1:
        args.final_seeds = args.final_seeds[:-1]
    tune_before, points_before = args.tune_rounds, args.lr_points
    while total() > limit and args.tune_rounds > args.min_tune_rounds:
        args.tune_rounds = max(args.min_tune_rounds, args.tune_rounds // 2)
    while total() > limit and args.lr_points > 3:
        args.lr_points -= 2

    # --- put back what the shedding over-shed ------------------------------
    # The concessions above are greedy and each is irreversible, so shedding a
    # cheap phase and *then* a costly one can leave the plan well under budget
    # with the cheap phase needlessly gone. Restore in the reverse order, keeping
    # anything that now fits: dropping the horizon-stability check to save 0.3 h
    # and finishing 1.7 h early is a bad trade.
    for name in ("verify", "rules", "wd"):
        if name in dropped:
            args.phases = args.phases + [name]
            if total() > limit:
                args.phases = [p for p in args.phases if p != name]
            else:
                dropped.discard(name)
    for name in ("wd", "rules", "verify"):
        if name in dropped:
            notes.append(f"dropped {PHASE_LABELS[name]}")

    # A shortened tuning horizon puts the proxy back, and the proxy is not
    # trustworthy here: on 2026-07-30 it picked the wrong rate for both methods it
    # was checked on. So re-enable the check that would catch it, and say plainly
    # what was given up -- "reduced the tuning horizon" reads like a resolution
    # setting, and it is not.
    if args.tune_rounds < args.final_rounds and "lr" in args.phases:
        if "verify" not in args.phases:
            args.phases = args.phases + ["verify"]
            dropped.discard("verify")
        notes.append(
            f"WARNING: tuning at {args.tune_rounds} rounds against a "
            f"{args.final_rounds}-round report, so rates are selected on a PROXY. "
            f"The 2026-07-30 run found the 400-round proxy picking the wrong rate "
            f"for both methods checked. The horizon-stability check is switched "
            f"back on; read its verdict before quoting the table.")

    # ``phases`` is consumed in order, so restore the canonical one. Every phase
    # name must appear here or it is silently dropped -- which is what happened to
    # `rules` for its first day: requested on the command line, deleted by any
    # --budget-hours at all, and never mentioned in the report.
    args.phases = [p for p in PHASE_ORDER if p in args.phases]

    # --- or spend the slack ------------------------------------------------
    while (total() + job_cost(sec, args.final_rounds) * sum(1 for _ in _lr_jobs(args))
           <= limit and len(args.final_seeds) < args.max_final_seeds):
        args.final_seeds = args.final_seeds + [max(args.final_seeds) + 1]

    # One note per dimension that actually moved, not one per loop iteration.
    if len(args.final_seeds) != seeds_before:
        verb = "raised" if len(args.final_seeds) > seeds_before else "reduced"
        notes.append(f"{verb} the final runs to {len(args.final_seeds)} seed(s) "
                     f"(from {seeds_before})")
    if args.tune_rounds != tune_before:
        notes.append(f"reduced the tuning horizon to {args.tune_rounds} rounds "
                     f"(from {tune_before})")
    if args.lr_points != points_before:
        notes.append(f"reduced the grid to {args.lr_points} points per method "
                     f"(from {points_before})")

    if total() > limit:
        # Everything shed-able has been shed and the plan still overruns, which
        # means the final runs alone do not fit. Those are deliberately NOT cut
        # automatically: --final-rounds 2000 is the paper's setting and shortening
        # it changes what is being reported, which is the author's call, not the
        # scheduler's.
        need = total() / 3600.0
        notes.append(
            f"STILL over budget: the plan needs ~{need:.1f} h against "
            f"{args.budget_hours:g} h. The run stops at the deadline and --resume "
            f"continues it; to fit in one sitting, lower --final-rounds (this "
            f"changes the reported setting), narrow --methods, or raise "
            f"--budget-hours.")
    return notes


def print_schedule(args, sec: float, budget: Budget, notes: List[str]) -> None:
    cost = plan_cost(args, sec)
    n_par = sum(1 for _ in _lr_jobs(args))
    n_rule_cfg = (sum(1 for r in args.rule_alternatives if r != args.lr_scaling)
                  * len(args.rule_methods))
    counts = {"lr": n_par * args.lr_points,
              "verify": (len(args.verify_methods) * args.verify_top
                         if args.tune_rounds < args.final_rounds else 0),
              "final": n_par * len(args.final_seeds),
              "rules": n_rule_cfg * (args.lr_points + len(args.rule_seeds)),
              "wd": args.wd_ablation_top if args.wd_ablation > 0 else 0}
    rounds = {"lr": args.tune_rounds, "verify": args.final_rounds,
              "final": args.final_rounds, "rules": args.final_rounds,
              "wd": args.final_rounds}

    print("\n" + "=" * 78)
    desc = ("no deadline -- runs until Ctrl-C" if budget.unlimited
            else f"budget {args.budget_hours:g} h")
    print(f"SCHEDULE  ({desc}, measured {sec:.3f} s/round amortized)")
    print("=" * 78)
    if notes:
        print("  Rebalanced to fit:")
        for n in notes:
            print(f"    - {n}")
        print()
    print(f"  {'phase':<8}{'jobs':>6}{'rounds':>8}{'hours':>8}{'cumulative':>12}   done by")
    cum = 0.0
    for name in PHASE_ORDER:
        if name not in args.phases or not counts[name]:
            why = ("skipped" if name != "verify" or args.tune_rounds < args.final_rounds
                   else "vacuous: tuning already runs at the reporting horizon")
            print(f"  {name:<8}{'-':>6}{'-':>8}{'-':>8}{'-':>12}   {why}")
            continue
        hrs = cost[name] / 3600.0
        cum += hrs
        eta = (datetime.now() + timedelta(hours=cum)).strftime("%a %H:%M")
        status = eta if budget.unlimited or cum <= args.budget_hours else "NO -- will be cut"
        print(f"  {name:<8}{counts[name]:>6}{rounds[name]:>8}{hrs:>8.1f}{cum:>12.1f}   {status}")
    print("=" * 78)
    print(f"  Total {cum:.1f} h. Finals are seed-major over seeds {args.final_seeds}, so "
          f"stopping early\n  leaves complete tables rather than fragments. The report on "
          f"disk is refreshed after\n  every phase and every final run; Ctrl-C stops cleanly "
          f"and writes it.")
    # Both searching phases widen a grid whose optimum lands on an endpoint, so
    # both contribute to the worst case. Counting only `lr` understated a run that
    # includes `rules` by a third.
    searching = (n_par if "lr" in args.phases else 0) + (
        n_rule_cfg if "rules" in args.phases else 0)
    extra = searching * args.lr_extend_rounds * args.lr_extend_points
    if extra:
        print(f"  Plus up to {extra} more tuning jobs "
              f"(+{extra * job_cost(sec, args.tune_rounds) / 3600:.1f} h) "
              f"if optima land on grid\n  endpoints: the grid is then widened and that method "
              f"re-tuned.")
    print(flush=True)


# --------------------------------------------------------------------------
# Phases
# --------------------------------------------------------------------------


def phase_lr(args, state, budget, sec) -> Dict[str, Dict]:
    """Tune eta_0 per method, identical budget for every method."""
    out: Dict[str, Dict] = state["phases"].setdefault("lr", {})
    shapes = matrix_shapes(args.dataset, args.model)
    rulekey = args.lr_scaling.replace(":", "")

    for method, scaled in _lr_jobs(args):
        key = f"{method}{'+scaled' if scaled else ''}"
        entry = out.setdefault(key, {"runs": [], "grid": []})
        base = anchor_for(method, args.lr_scaling, shapes, scale_baselines=scaled)
        # A resumed run inherits the grid an earlier round had already widened to,
        # so the extension budget is not spent twice on the same method.
        grid = entry.get("grid") or round_grid(base, points=args.lr_points)
        entry["grid"] = grid

        for extension in range(args.lr_extend_rounds + 1):
            for lr in grid:
                tag = f"lr_{key.replace('+', '_')}_{rulekey}_{lr:.4g}"
                jkey = canonical_tag(tag, epochs=args.tune_rounds, split="tune")
                if jkey in state["jobs"]:
                    continue
                cost = job_cost(sec, args.tune_rounds)
                if not budget.fits(cost) or _stop["requested"]:
                    return out
                print(f"[{stamp()}] lr/{key} ({args.tune_rounds} rounds, "
                      f"~{cost/60:.0f} min) | {budget.report()}")
                r = run_one(_tune_args(args), lr=lr, lr_aux=args.lr_aux,
                            lr_scaling=args.lr_scaling, method=method,
                            rounds=args.tune_rounds, tag=tag,
                            extra=("--scale-baselines",) if scaled else ())
                record(state, jkey, r)
                if r:
                    entry["runs"].append(r)
                    save_state(state)
            best = best_of(entry["runs"])
            if best is None:
                break
            entry["best"] = best
            warn = boundary_warning(best, grid)
            print(f"  BEST {key}: eta_0={best['lr']:.6g}  val {best['val_acc']:.2f}%")
            # An optimum on an endpoint is not an optimum: widen and keep going
            # rather than reporting a rate the grid boundary chose for us.
            if not warn:
                entry.pop("boundary", None)
                break
            print(warn)
            entry["boundary"] = warn.strip()
            if extension == args.lr_extend_rounds:
                break
            grid = extend_grid(grid, low="LOW end" in warn, points=args.lr_extend_points)
            entry["grid"] = grid
            print(f"  -> extending to [{min(grid):.4g}, {max(grid):.4g}] "
                  f"(round {extension + 1}/{args.lr_extend_rounds})")
            save_state(state)
    return out


def phase_verify(args, state, budget, sec, tuned: Dict[str, Dict]) -> Dict[str, Dict]:
    """Is the short-horizon ranking horizon-stable?

    Tuning at ``--tune-rounds`` and reporting at ``--final-rounds`` assumes the
    ranking of learning rates does not depend on the horizon -- and under a cosine
    schedule annealed to zero at the horizon, that assumption has real content:
    the two runs do not share a single step size. Deliberately still on the
    **tuning split**, so ``val_acc`` is the comparison metric at both horizons.

    Vacuous once the two horizons agree, which is now the default: there is no
    proxy left to check. It is kept because ``--tune-rounds`` can still be shortened
    by hand or by rebalancing, and that is exactly when the check earns its cost.
    """
    if args.tune_rounds >= args.final_rounds:
        print(f"  tuning already runs at the reporting horizon "
              f"({args.final_rounds} rounds); nothing to verify.")
        return state["phases"].setdefault("verify", {})
    out: Dict[str, Dict] = state["phases"].setdefault("verify", {})
    rulekey = args.lr_scaling.replace(":", "")
    for method in args.verify_methods:
        entry = tuned.get(method)
        if not entry or not entry.get("runs"):
            continue
        short = sorted((r for r in entry["runs"] if r and "val_acc" in r),
                       key=lambda r: -r["val_acc"])[:args.verify_top]
        if len(short) < 2:
            continue
        long_runs = out.setdefault(method, {"short": short, "long": []})["long"]
        for r in short:
            tag = f"verify_{method}_{rulekey}_{r['lr']:.4g}"
            jkey = canonical_tag(tag, epochs=args.final_rounds, split="tune")
            if jkey in state["jobs"]:
                continue
            cost = job_cost(sec, args.final_rounds)
            if not budget.fits(cost) or _stop["requested"]:
                return out
            print(f"[{stamp()}] verify/{method} at {args.final_rounds} rounds "
                  f"(~{cost/60:.0f} min) | {budget.report()}")
            res = run_one(_tune_args(args), lr=r["lr"], lr_aux=args.lr_aux,
                          lr_scaling=args.lr_scaling, method=method,
                          rounds=args.final_rounds, tag=tag)
            record(state, jkey, res)
            if res:
                long_runs.append(res)
                save_state(state)
    _verify_verdict(out, args)
    return out


def _verify_verdict(out: Dict[str, Dict], args) -> None:
    if not out:
        return
    print("\n  --- horizon-stability verdict ---")
    for method, d in out.items():
        short, long = d.get("short", []), d.get("long", [])
        if len(long) < 2:
            print(f"    {method}: not enough long-horizon runs to judge")
            continue
        s_order = [r["lr"] for r in sorted(short, key=lambda r: -r["val_acc"])]
        l_order = [r["lr"] for r in sorted(long, key=lambda r: -r["val_acc"])]
        same = abs(s_order[0] - l_order[0]) < 1e-12
        print(f"    {method}: best at {args.tune_rounds} rounds = {s_order[0]:.4g}, "
              f"at {args.final_rounds} = {l_order[0]:.4g} -> "
              f"{'STABLE' if same else 'MOVED'}")
        if not same:
            print(f"      the {args.tune_rounds}-round proxy picked the wrong rate for "
                  f"this method; prefer the long-horizon winner and treat the tuned "
                  f"table as provisional.")
        d["stable"] = bool(same)
        d["best_short"], d["best_long"] = s_order[0], l_order[0]


def phase_final(args, state, budget, sec, tuned: Dict[str, Dict], refresh) -> None:
    """Full-50k runs at the tuned eta_0, seed-major."""
    rulekey = args.lr_scaling.replace(":", "")
    for seed in args.final_seeds:
        for method, scaled in _lr_jobs(args):
            key = f"{method}{'+scaled' if scaled else ''}"
            best = tuned.get(key, {}).get("best")
            if not best:
                continue
            tag = f"{key.replace('+', '_')}_{rulekey}"
            jkey = canonical_tag(tag, epochs=args.final_rounds, split="full", seed=seed)
            if jkey in state["jobs"]:
                continue
            cost = job_cost(sec, args.final_rounds)
            if not budget.fits(cost) or _stop["requested"]:
                return
            print(f"[{stamp()}] final/{key} seed {seed} ({args.final_rounds} rounds, "
                  f"~{cost/60:.0f} min) | {budget.report()}")
            r = run_one(_tune_args(args), lr=best["lr"], lr_aux=args.lr_aux,
                        lr_scaling=args.lr_scaling, method=method,
                        rounds=args.final_rounds, tag=tag, split="full", seed=seed,
                        extra=("--scale-baselines",) if scaled else ())
            record(state, jkey, r)
            if r:
                state["phases"].setdefault("final", {})[jkey] = r
                save_state(state)
                refresh()


def phase_wd(args, state, budget, sec, tuned: Dict[str, Dict]) -> None:
    """Re-run the best few at the final horizon with decay switched on.

    The primary table uses ``--weight_decay 0``, so that the experiment and the
    theorems describe the same algorithm. This phase supplies the other number:
    whether decay changes the *ordering*, at the same eta_0 (which was not
    re-tuned under decay).
    """
    if args.wd_ablation <= 0 or not tuned:
        return
    rulekey = args.lr_scaling.replace(":", "")
    ranked = sorted(((d["best"]["val_acc"], k, d["best"]["lr"])
                     for k, d in tuned.items() if d.get("best")), reverse=True)
    picks = ranked[:max(0, args.wd_ablation_top)]
    out = state["phases"].setdefault("wd", {})
    print(f"  decay ablation on {[k for _, k, _ in picks]} at "
          f"wd={args.wd_ablation:g} (decoupled; sgd is coupled by construction, "
          f"since its step is not scale-invariant)")
    for _, key, lr in picks:
        method = key.replace("+scaled", "")
        seed = args.final_seeds[0]
        tag = f"wd_{key.replace('+', '_')}_{rulekey}"
        jkey = canonical_tag(tag, epochs=args.final_rounds, split="full", seed=seed)
        if jkey in state["jobs"]:
            continue
        cost = job_cost(sec, args.final_rounds)
        if not budget.fits(cost) or _stop["requested"]:
            return
        print(f"[{stamp()}] wd/{key} (~{cost/60:.0f} min) | {budget.report()}")
        # ``extra`` lands at the END of the child argv, so it overrides what the
        # driver already put there.
        extra = (("--scale-baselines",) if key.endswith("+scaled") else ()) + (
            "--weight_decay", repr(args.wd_ablation))
        r = run_one(_tune_args(args), lr=lr, lr_aux=args.lr_aux,
                    lr_scaling=args.lr_scaling, method=method,
                    rounds=args.final_rounds, tag=tag, split="full", seed=seed,
                    extra=extra)
        record(state, jkey, r)
        if r:
            ref_key = canonical_tag(f"{key.replace('+', '_')}_{rulekey}",
                                    epochs=args.final_rounds, split="full", seed=seed)
            ref = (state["phases"].get("final") or {}).get(ref_key) or {}
            out[key] = {"wd": args.wd_ablation, "lr": lr,
                        "test_acc": r.get("test_acc"),
                        "test_acc_no_decay": ref.get("test_acc")}
            save_state(state)


def phase_rules(args, state, budget, sec) -> Dict[str, Dict]:
    """Re-tune the sign family from scratch under the competing per-layer rules.

    The unit-gain multiplier is a heuristic, so the ordering it induces *within* the
    sign family rests on it. Two things bound the exposure without an experiment:
    the LMO family cannot move, because unit gain and muP prescribe it the identical
    factor, and all three sign methods are tuned and reported under one rule, so a
    wrong exponent rescales them alike. This settles it directly instead.

    Each (method, rule) gets its own full grid search and its own finals -- a rule
    that shifted eta_0 without being re-tuned would be a strictly worse rule by
    construction, which would prove nothing. ``--lr-scaling`` itself is not repeated
    here: the `lr` and `final` phases already ran it.
    """
    if not args.rule_methods or not args.rule_alternatives:
        return {}
    out: Dict[str, Dict] = state["phases"].setdefault("rules", {})
    shapes = matrix_shapes(args.dataset, args.model)

    for rule in args.rule_alternatives:
        if rule == args.lr_scaling:
            continue
        rulekey = rule.replace(":", "")
        for method in args.rule_methods:
            key = f"{method}@{rule}"
            entry = out.setdefault(key, {"rule": rule, "method": method,
                                         "runs": [], "finals": []})
            base = anchor_for(method, rule, shapes)
            grid = entry.get("grid") or round_grid(base, points=args.lr_points)
            entry["grid"] = grid

            # The grid is widened on a boundary here for the same reason it is in
            # `phase_lr`, and it matters more: an alternative rule whose optimum is
            # censored by its grid loses to a properly tuned `--lr-scaling`
            # automatically, which would make this ablation confirm the rule it is
            # supposed to test.
            for extension in range(args.lr_extend_rounds + 1):
                for lr in grid:
                    tag = f"rule_{method}_{rulekey}_{lr:.4g}"
                    jkey = canonical_tag(tag, epochs=args.tune_rounds, split="tune")
                    if jkey in state["jobs"]:
                        continue
                    cost = job_cost(sec, args.tune_rounds)
                    if not budget.fits(cost) or _stop["requested"]:
                        return out
                    print(f"[{stamp()}] rules/{key} lr={lr:.4g} "
                          f"(~{cost/60:.0f} min) | {budget.report()}")
                    r = run_one(_tune_args(args), lr=lr, lr_aux=args.lr_aux,
                                lr_scaling=rule, method=method,
                                rounds=args.tune_rounds, tag=tag)
                    record(state, jkey, r)
                    if r:
                        entry["runs"].append(r)
                        save_state(state)

                best = best_of(entry["runs"])
                if best is None:
                    break
                entry["best"] = best
                warn = boundary_warning(best, grid)
                print(f"  BEST {key}: eta_0={best['lr']:.6g}  val {best['val_acc']:.2f}%")
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
                print(f"  -> extending to [{min(grid):.4g}, {max(grid):.4g}] "
                      f"(round {extension + 1}/{args.lr_extend_rounds})")
                save_state(state)

    for rule in args.rule_alternatives:
        if rule == args.lr_scaling:
            continue
        rulekey = rule.replace(":", "")
        for seed in args.rule_seeds:
            for method in args.rule_methods:
                entry = out.get(f"{method}@{rule}") or {}
                if not entry.get("best"):
                    continue
                tag = f"rulefinal_{method}_{rulekey}"
                jkey = canonical_tag(tag, epochs=args.final_rounds,
                                     split="full", seed=seed)
                if jkey in state["jobs"]:
                    continue
                cost = job_cost(sec, args.final_rounds)
                if not budget.fits(cost) or _stop["requested"]:
                    return out
                print(f"[{stamp()}] rules-final/{method}@{rule} seed {seed} "
                      f"(~{cost/60:.0f} min) | {budget.report()}")
                r = run_one(_tune_args(args), lr=entry["best"]["lr"],
                            lr_aux=args.lr_aux, lr_scaling=rule, method=method,
                            rounds=args.final_rounds, tag=tag, split="full", seed=seed)
                record(state, jkey, r)
                if r:
                    entry["finals"].append(r)
                    save_state(state)
    return out


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def build_report(args, state, budget, sec, tuned, notes: List[str]) -> str:
    lines = [
        "# Federated overnight run report",
        "",
        f"* started `{state.get('started')}`, wall clock {budget.report()}",
        (f"* device `{args.device}`, {args.model}/{args.dataset}, "
         f"N={args.n_parties} clients x {args.n_steps} local steps, batch "
         f"{args.batch_size}, `{args.partition}` split, measured "
         f"**{sec:.3f} s/round** amortized" if sec else "* timing unavailable"),
        f"* scaling rule **`{args.lr_scaling}`**, lr_aux = {args.lr_aux:g}, "
        f"momentum {args.momentum:g}, weight decay {args.weight_decay:g} "
        f"(decoupled; sgd coupled)",
        f"* tuning: {args.tune_rounds} rounds on the 45k/5k split, selected on "
        f"**val_acc** (tail mean of the last {args.last_k} of {args.eval_points} "
        f"evaluations); **no test image is ever scored by a tuning run**",
        f"* finals: {args.final_rounds} rounds on the full 50k, seeds "
        f"{args.final_seeds}",
        "",
    ]
    if notes:
        lines += ["## Schedule rebalancing", "",
                  "The plan was fitted to the budget from the measured round time:", ""]
        lines += [f"* {n}" for n in notes]
        lines.append("")

    if tuned:
        lines += ["## Tuned eta_0 (validation)", "",
                  "| method | family | eta_0 | val acc | configs | note |",
                  "| :--- | :--- | ---: | ---: | ---: | :--- |"]
        for key, d in tuned.items():
            b = d.get("best")
            if not b:
                continue
            fam = method_family(key.replace("+scaled", ""),
                                key.endswith("+scaled")) or "-"
            lines.append(f"| `{key}` | {fam} | {b['lr']:.6g} | {b['val_acc']:.2f}% | "
                         f"{len(d.get('runs') or [])} | {d.get('boundary', '')} |")
        lines += ["", "A `BOUNDARY` note means the optimum sat on a grid endpoint: "
                      "extend the grid and re-run that method before reporting it.",
                  "",
                  "The rule's falsifiable prediction is that `eta_0` agrees *within* "
                  "each family once the shape dependence lives in `lambda`. Check the "
                  "`family` column against the rates.", ""]

    verify = state["phases"].get("verify", {})
    if not verify and args.tune_rounds >= args.final_rounds:
        # Said rather than left to inference: "no horizon section" and "rates were
        # selected at the horizon they are reported at" look identical in a report,
        # and only the second is a claim the paper makes.
        lines += ["## Horizon stability", "",
                  f"Not applicable: rates were selected at {args.tune_rounds} rounds, "
                  f"the same horizon the table reports, so there is no proxy to "
                  f"check.", ""]
    if verify:
        lines += ["## Horizon stability", "",
                  f"Top-{args.verify_top} rates re-run at {args.final_rounds} rounds "
                  f"(still on the tuning split, so `val_acc` is comparable):", "",
                  "| method | best @ tune | best @ final | verdict |",
                  "| :--- | ---: | ---: | :--- |"]
        for m, d in verify.items():
            if "best_short" not in d:
                continue
            lines.append(f"| `{m}` | {d['best_short']:.4g} | {d['best_long']:.4g} | "
                         f"{'stable' if d.get('stable') else '**MOVED**'} |")
        lines.append("")

    finals = state["phases"].get("final", {})
    if finals:
        lines += ["## Final runs (full 50k, test set)", "",
                  "| run | test acc (tail mean) | rounds to target | comm. saving "
                  "(round trip) | MV ties | gain spread |",
                  "| :--- | ---: | ---: | ---: | ---: | ---: |"]

        def _fmt(x, suffix="", nd=3):
            return "-" if x is None else f"{x:.{nd}f}{suffix}"

        for tag, v in sorted(finals.items()):
            if not v:
                continue
            lines.append(f"| `{tag}` | {v.get('test_acc', float('nan')):.2f}% | "
                         f"{v.get('rounds_to_target', '-')} | "
                         f"{_fmt(v.get('round_trip_reduction'), 'x', 1)} | "
                         f"{_fmt(v.get('mv_tie_frac'))} | "
                         f"{_fmt(v.get('gain_spread'), 'x', 2)} |")
        lines += ["", "Aggregate across seeds with "
                      "`python3 -m aggregate --root results/federated`.",
                  "",
                  "**comm. saving** is the *round-trip* reduction against full "
                  "precision, with the uncompressed auxiliary group counted in "
                  "(the sign alphabet is a genuine 1 bit under the randomized-zero "
                  "convention). On CNN2 it is 1.9x for the three uplink-only "
                  "methods -- MuonUSign, EF21-SignMuon and EF21-MuonUSign still "
                  "broadcast a full-precision model every round -- and 29.4x for the "
                  "four that are 1 bit in both directions (SignMuon, MuonSign, "
                  "EF21-MuonSign, SignSGD). "
                  "**Quote this number, not the uplink-only one.**",
                  "",
                  "**gain spread** is `max/min` over layers of the realized "
                  "`||lambda*s||_F/sqrt(fan_out)`. The per-layer rule targets 1.00x. "
                  "The sign family is flat by construction; the LMO family inherits "
                  "the Newton-Schulz norm error, which is shape-dependent and so is "
                  "not absorbable into a single tuned `eta_0`.",
                  "",
                  f"**MV ties** is the fraction of coordinates where the RAW "
                  f"vote came out exactly zero, counted before any tie-break. "
                  f"Under the randomized-zero convention client messages are +-1, "
                  f"so at an odd `N` the vote cannot actually tie. This "
                  f"run used **N = {args.n_parties}**"
                  + (" (odd)." if args.n_parties % 2
                     else f" (EVEN -- prefer {args.n_parties + 1})."), ""]

    rules = state["phases"].get("rules", {})
    if rules:
        import statistics
        lines += ["## Per-layer rule ablation (sign family)", "",
                  "Each (method, rule) pair is re-tuned from scratch, then run at the "
                  "final horizon. What matters is whether the *ordering* of the "
                  "methods changes with the rule, not whether `eta_0` does -- it "
                  "should, by roughly the multiplier the rule prescribes.",
                  "",
                  "| method | rule | eta_0 | val @ tune | test acc (finals) | seeds |",
                  "| :--- | :--- | ---: | ---: | ---: | ---: |"]
        def _row(method, rule, best, finals, boundary=False, primary=False):
            got = [x for x in (r.get("test_acc") for r in finals if r) if x is not None]
            label = f"**{rule}**" if primary else rule
            return "| " + " | ".join([
                f"`{method}`", label + ("  (grid boundary)" if boundary else ""),
                f"{best['lr']:.6g}" if best.get("lr") is not None else "-",
                f"{best['val_acc']:.2f}%" if best.get("val_acc") is not None else "-",
                f"{statistics.fmean(got):.2f}%" if got else "-",
                str(len(got))]) + " |"

        # Grouped by method, its own rule first, so the comparison the ablation is
        # about -- one method across three rules -- reads down the table.
        for method in args.rule_methods:
            # The `--lr-scaling` arm is not re-run here; the `lr`/`final` phases are it.
            final_runs = [v for v in (state["phases"].get("final") or {}).values()
                          if v and v.get("method") == method
                          and v.get("lr_scaling", args.lr_scaling) == args.lr_scaling]
            lines.append(_row(method, args.lr_scaling,
                              tuned.get(method, {}).get("best") or {},
                              final_runs, primary=True))
            for key, v in sorted(rules.items()):
                if v.get("method") != method:
                    continue
                lines.append(_row(method, v["rule"], v.get("best") or {},
                                  v.get("finals", []), boundary=bool(v.get("boundary"))))
        lines.append("")

    wd = state["phases"].get("wd", {})
    if wd:
        lines += [f"## Weight-decay ablation (decoupled, wd = {args.wd_ablation:g})", "",
                  "| method | eta_0 | with decay | no decay | delta |",
                  "| :--- | ---: | ---: | ---: | ---: |"]
        for k, v in sorted(wd.items()):
            got, ref = v.get("test_acc"), v.get("test_acc_no_decay")
            lines.append("| " + " | ".join([
                f"`{k}`", f"{v['lr']:.6g}",
                "-" if got is None else f"{got:.2f}%",
                "-" if ref is None else f"{ref:.2f}%",
                "-" if (got is None or ref is None) else f"{got - ref:+.2f}"]) + " |")
        lines.append("")

    done = sum(1 for v in state["jobs"].values() if v)
    failed = sum(1 for v in state["jobs"].values() if not v)
    lines += ["## What this run did NOT establish", "",
              f"* **`lr_aux` was fixed at {args.lr_aux:g}**, not tuned. The auxiliary "
              f"group is AdamW on the same parameters for every method, so its optimum "
              f"should not depend on the matrix rule -- but that is an argument, not a "
              f"measurement. Check it with `python3 -m federated.tune --stage aux`.",
              f"* **One federation scale** (N={args.n_parties}, {args.n_steps} local "
              f"steps) and one partition (`{args.partition}`). The paper's Experiment 1 "
              f"(N=3) is not covered here.",
              "* **BatchNorm running statistics are never updated from data.** Local "
              "models are discarded each round and BN runs in inference mode during "
              "accumulation, so the statistics stay at their initialization `(0, 1)` for "
              "the whole run, in training and evaluation alike. This is self-consistent "
              "but it needs a sentence in the reproducibility appendix.",
              "* **A gap smaller than the seed spread is not a result.** Add seeds with "
              "`--resume --max-final-seeds 5` and aggregate before claiming one.",
              "",
              "## Next steps", "",
              f"{done} jobs completed, {failed} failed. Resume with:", "",
              "```bash",
              f"python3 -m federated.overnight --device {args.device} --resume",
              "```", ""]
    return "\n".join(lines)


def refresh_report(args, state, budget, sec, tuned, notes, *, echo: bool = False) -> None:
    text = build_report(args, state, budget, sec, tuned, notes)
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
    p.add_argument("--budget-hours", type=float, default=12.0,
                   help="Wall-clock deadline. The plan is rebalanced to fit it from "
                        "the measured round time; 0 means NO deadline and no "
                        "rebalancing (every phase runs to completion)")
    p.add_argument("--budget-headroom", type=float, default=0.9,
                   help="Fraction of the budget the plan is fitted to, leaving room "
                        "for the round time being underestimated")
    p.add_argument("--methods", nargs="*", default=ALL_METHODS)
    p.add_argument("--phases", nargs="*", default=list(DEFAULT_PHASES),
                   choices=list(PHASE_ORDER),
                   help="`rules` is the per-layer-rule ablation and is off by "
                        "default; add it explicitly")

    p.add_argument("--tune-rounds", type=int, default=None,
                   help="Horizon for ranking rates. Defaults to --final-rounds, "
                        "i.e. rates are selected at the horizon the table reports. "
                        "A shorter value is a proxy, and the 2026-07-30 run showed "
                        "the 400-round proxy picking the wrong rate for both methods "
                        "it was checked on -- see the `verify` phase")
    p.add_argument("--min-tune-rounds", type=int, default=100,
                   help="Rebalancing will not shorten the tuning horizon below this")
    p.add_argument("--final-rounds", type=int, default=2000,
                   help="Round budget for the full-50k runs; 2000 matches the paper")
    p.add_argument("--final-seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4],
                   help="Seeds for the full-50k runs; five is what the paper's "
                        "federated table reports. Rebalancing drops seeds from the "
                        "end first if the budget cannot hold them, and says so")
    p.add_argument("--max-final-seeds", type=int, default=5,
                   help="Upper limit when the budget has slack to spend on error bars")
    p.add_argument("--lr-points", type=int, default=5)
    p.add_argument("--lr-extend-rounds", type=int, default=3)
    p.add_argument("--lr-extend-points", type=int, default=2)
    p.add_argument("--verify-methods", nargs="*", default=["signmuon", "ef21muonsign"],
                   help="Methods whose top rates are re-run at --final-rounds. "
                        "Vacuous, and skipped, when the tuning horizon already is "
                        "--final-rounds")
    p.add_argument("--verify-top", type=int, default=2)
    p.add_argument("--rule-methods", nargs="*", default=["signmuon", "muonsign", "signsgd"],
                   help="`rules` phase: which methods to re-tune. The sign family, "
                        "since the LMO family is prescribed the same multiplier by "
                        "every candidate rule and cannot move")
    p.add_argument("--rule-alternatives", nargs="*", default=["none", "mup"],
                   help="`rules` phase: the conventions to compare against "
                        "--lr-scaling, which is already covered by the `lr` phase")
    p.add_argument("--rule-seeds", nargs="*", type=int, default=[0, 1, 2],
                   help="`rules` phase: seeds per (method, rule) at the final horizon")
    p.add_argument("--wd-ablation", type=float, default=5e-4,
                   help="Decay rate for the `wd` phase; 0 disables it")
    p.add_argument("--wd-ablation-top", type=int, default=3)
    p.add_argument("--skip-baseline-variants", action="store_true",
                   help="Do NOT additionally tune Adam under the sign rule")
    p.add_argument("--timing-rounds", type=int, default=40,
                   help="Rounds used to measure the per-round cost in preflight")

    p.add_argument("--resume", action="store_true", help="Skip jobs already recorded")
    p.add_argument("--no-export", action="store_true",
                   help="Skip the closing 'federated.export_article' step. The "
                        "bundle is the file you download, so this is for debugging")
    p.add_argument("--report-only", action="store_true",
                   help="Rebuild REPORT.md from state.json and exit: runs nothing and "
                        "is safe to call while a run is in flight")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the schedule and exit without training")
    p.add_argument("--force", action="store_true",
                   help="Continue even if the CPU test suite fails")
    args = add_common_args(p).parse_args()
    # Select at the horizon we report at. The centralized driver already does this,
    # for the reason a 2026-07-30 federated run then confirmed here: under a cosine
    # schedule annealed to zero at each run's *own* horizon, a short run spends
    # nearly its whole budget at a decayed rate, and the rate that wins there is not
    # the rate that wins at 2000 rounds. It moved for both methods checked.
    if args.tune_rounds is None:
        args.tune_rounds = args.final_rounds
    return args


def main() -> None:
    args = get_args()
    signal.signal(signal.SIGINT, _handle_sigint)

    state = load_state() if args.resume or args.report_only else {"jobs": {}, "phases": {}}
    state.setdefault("jobs", {})
    state.setdefault("phases", {})
    state["started"] = state.get("started") or datetime.now().isoformat(timespec="seconds")
    budget = Budget(args.budget_hours)

    if args.report_only:
        sec = state.get("round_seconds") or 0.0
        notes = state.get("rebalance_notes", [])
        print(build_report(args, state, budget, sec, state["phases"].get("lr", {}), notes))
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(
            build_report(args, state, budget, sec, state["phases"].get("lr", {}), notes),
            encoding="utf-8")
        print(f"\n[{stamp()}] rewrote {REPORT_PATH}")
        return

    if args.download:
        # Fetch once, here, rather than letting every child pass --download: a
        # partially written archive shared by many runs is a bad failure mode.
        print(f"[{stamp()}] fetching {args.dataset} into {args.data} ...")
        from federated.data import load_raw
        load_raw(args.dataset, args.data, download=True)
        print(f"[{stamp()}] dataset ready")

    # Resuming across a change of convention would silently mix runs that are not
    # comparable: the cached jobs are keyed by method and rate only.
    # ``tune_rounds`` belongs here even though it names no final run: a final job's
    # key does not encode the rate it was launched at, so resuming across a change
    # of tuning horizon would keep finals selected by the old horizon and pair them
    # with rates selected by the new one, silently.
    fingerprint = {k: getattr(args, k) for k in
                   ("dataset", "model", "n_parties", "n_steps", "batch_size",
                    "momentum", "weight_decay", "partition", "beta", "lr_aux",
                    "lr_scaling", "final_rounds", "tune_rounds")}
    old = state.get("fingerprint")
    if args.resume and old and old != fingerprint:
        print("REFUSING to resume: this run's settings differ from the recorded ones,")
        print("so the cached jobs are not comparable with the new ones.")
        for k, v in fingerprint.items():
            if old.get(k) != v:
                print(f"  {k}: recorded {old.get(k)!r} -> requested {v!r}")
        print(f"\nStart a fresh sweep (drop --resume, which overwrites "
              f"{STATE_PATH.name}), or move the old results aside first. Use "
              f"--report-only to read the existing report without running anything.")
        sys.exit(1)
    state["fingerprint"] = fingerprint

    sec = state.get("round_seconds") if args.resume else None
    if sec is None:
        measured = preflight(args)
        if measured is None:
            print("Preflight failed; nothing was trained.")
            sys.exit(1)
        sec, _OVERHEAD["seconds"] = measured
        state["round_seconds"] = sec
        state["job_overhead_seconds"] = _OVERHEAD["seconds"]
        save_state(state)
    else:
        _OVERHEAD["seconds"] = state.get("job_overhead_seconds", JOB_OVERHEAD_S)
        print(f"[{stamp()}] resuming with the previously measured {sec:.3f} s/round "
              f"(+{_OVERHEAD['seconds']:.0f}s per job, "
              f"{len(state['jobs'])} jobs already recorded)")

    notes = autobalance(args, sec, budget)
    state["rebalance_notes"] = notes
    state["plan"] = {"tune_rounds": args.tune_rounds, "final_rounds": args.final_rounds,
                     "final_seeds": args.final_seeds, "lr_points": args.lr_points,
                     "phases": args.phases}
    save_state(state)
    print_schedule(args, sec, budget, notes)
    if args.preflight_only or args.dry_run:
        print("Stopping here as requested (--preflight-only / --dry-run).")
        return

    tuned: Dict[str, Dict] = {}

    def refresh(echo: bool = False) -> None:
        refresh_report(args, state, budget, sec,
                       tuned or state["phases"].get("lr", {}), notes, echo=echo)

    def banner(name: str, note: str = "") -> None:
        print(f"\n{'=' * 78}\n[{stamp()}] PHASE {name}{note}\n{'=' * 78}", flush=True)

    # A second Ctrl-C raises out of the handler; the report must still be written,
    # so the phases run inside try/finally rather than being trusted to complete.
    try:
        if "lr" in args.phases:
            banner("lr", f"  (rule '{args.lr_scaling}', {args.tune_rounds} rounds)")
            tuned = phase_lr(args, state, budget, sec)
            refresh()
        if "verify" in args.phases and not _stop["requested"]:
            banner("verify", f"  (horizon stability, {args.final_rounds} rounds)")
            phase_verify(args, state, budget, sec,
                         tuned or state["phases"].get("lr", {}))
            refresh()
        if "final" in args.phases and not _stop["requested"]:
            banner("final", f"  ({args.final_rounds} rounds, full 50k, "
                            f"seeds {args.final_seeds})")
            phase_final(args, state, budget, sec,
                        tuned or state["phases"].get("lr", {}), refresh)
            refresh()
        if "rules" in args.phases and not _stop["requested"]:
            banner("rules", f"  (per-layer rule ablation: "
                            f"{', '.join(args.rule_alternatives)} vs "
                            f"{args.lr_scaling}, sign family)")
            phase_rules(args, state, budget, sec)
            refresh()
        if "wd" in args.phases and not _stop["requested"]:
            banner("wd", f"  (decay ablation, wd={args.wd_ablation:g} decoupled)")
            phase_wd(args, state, budget, sec, tuned or state["phases"].get("lr", {}))
    except KeyboardInterrupt:
        print(f"\n[{stamp()}] interrupted -- writing the report from what completed.")
    finally:
        save_state(state)
        refresh(echo=True)
        if not args.no_export:
            export()


def export() -> None:
    """Bundle the results and print the one file to download.

    Runs even after an interrupt: a night that stopped at seed 3 still produced a
    table worth carrying home, and a partial bundle says how partial it is. Never
    fatal -- the runs are already on disk, and losing the archive step should not
    make the night look like a failure.
    """
    from federated import export_article
    try:
        print(f"\n{'=' * 78}\n[{stamp()}] packing the results\n{'=' * 78}", flush=True)
        export_article.main([])
    except SystemExit:
        raise
    except Exception as exc:                                 # noqa: BLE001
        print(f"[{stamp()}] could not write the bundle: {type(exc).__name__}: {exc}")
        print("The runs themselves are intact under results/federated; "
              "run 'python3 -m federated.export_article' by hand.")


if __name__ == "__main__":
    main()
