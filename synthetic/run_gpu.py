"""One command to produce every synthetic-benchmark number, on a GPU box.

Run both of these from ``code/``::

    python3 -m synthetic.run_gpu --force      # everything, ~1.6 h
    python3 -m synthetic.run_gpu --archive    # rebuild SUMMARY.md + the .zip

The first runs the CPU test suite as a preflight, then all seven stages, and
writes the archive itself; the second only rebuilds it, from whatever is
already on disk. ``--force`` is what makes the first a *rerun*: a stage whose
output is already on disk is skipped, so without it a box that has run before
skips all seven and exits in seconds looking like a success. Useful variants::

    python3 -m synthetic.run_gpu --quick             # ~2 min smoke test
    python3 -m synthetic.run_gpu --stages floor horizon
    python3 -m synthetic.run_gpu --list              # what the stages measure

Everything lands under ``results/synthetic/``::

    SUMMARY.md              <- every table in one file; this is the one to read
    MANIFEST.json           <- commit, GPU, CPU, torch/CUDA, argv, wall time per stage
    logs/<stage>.log        <- full console output, including the preflight
    <method>/<mode>.json    <- machine-readable results

and ``results/synthetic_results.zip`` holds all of it -- the one file to bring
back from a remote box, since ``results/`` is gitignored.

Each stage is a separate subprocess, so one failure does not take the others
down, and a stage whose output already exists is skipped (``--force`` to redo).
Stage order is cheapest-first.

What a stage costs
------------------
An A4000 runs ~1450 optimizer steps per second and that figure barely moves
between ``--m 20`` and ``--m 100``: at these sizes a step is ~50 tiny CUDA
kernels and the arithmetic is microseconds, so cost is set by the *number* of
steps, not their size. The batched runner advances a whole
``(eta, momentum, schedule)`` grid as one ``[B, m, n]`` trajectory for about
what a single trajectory costs, which leaves the iteration counts -- not the
grids -- as what a stage is paying for. Trim a grid because it tells you
nothing, not because it is expensive. The estimates below start from the
2026-07-29 A4000 run (68 min for all seven, the only measurement there is) and
scale the four tuning stages by the 2.04x more configurations the widened
learning-rate and momentum grids carry. That scaling is the pessimistic reading
and contradicts the paragraph above: if the runner is still launch-bound at a
batch of ~310 the stages barely move, and if it has become arithmetic-bound
they roughly double. Which one holds has not been measured. The estimates take
the expensive branch, since an estimate that runs short is worse than one that
runs long for anyone planning an overnight run, and the next full run settles
it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from common.utils import results_root, split_list_arg

CODE_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Stage:
    """One ``synthetic.benchmark`` invocation."""

    name: str
    mode: str
    why: str
    args: List[str] = field(default_factory=list)
    quick_args: List[str] = field(default_factory=list)
    estimate: str = "?"

    def argv(self, device: str, out: Path, quick: bool,
             size: Optional[tuple] = None, runner: Optional[str] = None) -> List[str]:
        base = [sys.executable, "-m", "synthetic.benchmark",
                "--mode", self.mode, "--device", device, "--out", str(out)]
        argv = base + self.args + (self.quick_args if quick else [])
        if size is not None:
            # Appended last so it beats any --m/--n baked into the stage's own
            # args (argparse keeps the final occurrence).
            argv += ["--m", str(size[0]), "--n", str(size[1])]
        if runner is not None:
            argv += ["--runner", runner]
        return argv


# Cheapest first, so the interesting measurements land before the long ones
# finish. Every stage runs at the module's 100x100 default over three problem
# draws and the same (eta, momentum) grids; where a stage sets its own, the
# comment says what the measurement needs that the default does not give.
STAGES: List[Stage] = [
    Stage(
        "stability", "stability",
        "Largest stable eta per method, and the step length eta_max*||S||_F it "
        "corresponds to. SGD is the built-in control: its eta_max must land on "
        "the textbook 2/L, and if it does not, nothing else here is trustworthy.",
        estimate="~2 min",
        quick_args=["--m", "64", "--n", "64", "--stability-iters", "150",
                    "--problem-seeds", "1337"],
    ),
    Stage(
        "alignment", "alignment",
        "Distribution of rho_t = <grad F, D_t>/(||grad F||_F ||D_t||_F) along "
        "the tuned trajectory -- the quantity the descent lemma needs positive "
        "and the divergence theorems drive negative. The one measurement here "
        "that is about the methods rather than the tuning protocol.",
        estimate="~3 min",
        quick_args=["--m", "64", "--n", "64", "--align-iters", "300",
                    "--problem-seeds", "1337", "--momentum-grid", "0.0,0.9",
                    "--lr-grid", "sign=1e-4:1e-2:x2", "lmo=1e-3:1e-1:x2",
                    "sgd=1e-2:1e0:x2", "adam=1e-2:1e0:x2"],
    ),
    Stage(
        "floor", "floor",
        "The plateau F_inf(eta), ||grad F||_inf(eta) of a constant step and "
        "their exponents in eta. The descent lemma predicts a gradient floor "
        "linear in eta with coefficient L||S||_F/2rho; SignMuon and SignSGD "
        "share ||S||_F exactly, so any gap between their floors is rho alone.",
        # Its own eta grid, one and a half decades wide: time to plateau scales
        # like 1/eta, so the budget does too, and widening the grid costs
        # superlinearly for no extra slope. The longest budget sets the wall
        # clock of this stage, which is why it is the slowest of the seven.
        # One spec per step-norm family, not per method: written out per method,
        # ``ef21signmuon`` had been given the sign-family window although its
        # step is an LMO one, and its smallest eta was then still 10x too small
        # to plateau inside 60k iterations.
        args=["--floor-iters", "3000", "--floor-max-iters", "60000",
              "--lr-grid", "sign=6e-5:2e-3:x4", "lmo=6e-4:2e-2:x4",
              "sgd=6e-3:2e-1:x4", "adam=6e-4:2e-2:x4"],
        estimate="~35 min",
        quick_args=["--m", "64", "--n", "64", "--floor-iters", "1200",
                    "--floor-max-iters", "20000", "--problem-seeds", "1337",
                    "--methods", "signmuon", "signsgd", "muonsign", "muon",
                    "--lr-grid", "sign=2e-4:2e-3:x3", "lmo=2e-3:2e-2:x3"],
    ),
    Stage(
        "horizon", "horizon",
        "Tunes (eta, momentum, schedule) separately at each budget T and fits "
        "err ~ T^-p, eta* ~ T^-q, with err = min_t ||grad F||_*^2 in the norm "
        "dual to each method's LMO ball. p = q = 1/2 is the nonconvex regime "
        "the theorems prove, p = q = 1 the strongly convex one.",
        args=["--budgets", "125", "250", "500", "1000", "2000"],
        estimate="~18 min",
        quick_args=["--m", "64", "--n", "64", "--budgets", "125", "250", "500",
                    "--problem-seeds", "1337", "--momentum-grid", "0.0,0.9",
                    "--methods", "signmuon", "signsgd", "muon", "sgd",
                    "--lr-grid", "sign=1e-4:1e-2:x3", "lmo=1e-3:1e-1:x3",
                    "sgd=1e-2:1e1:x3"],
    ),
    Stage(
        "kappa", "kappa",
        "The tuned comparison at a controlled condition number, swept over "
        "decades at L = 1. Conditioning is the only knob that governs the "
        "dynamics of a quadratic, and the uniform draw leaves it to chance.",
        args=["--max-iters", "3000"],
        estimate="~22 min",
        quick_args=["--m", "40", "--n", "40", "--max-iters", "1500",
                    "--kappas", "1e2", "1e3", "1e4", "--problem-seeds", "1337",
                    "--methods", "signmuon", "signsgd", "muon",
                    "--momentum-grid", "0.0,0.9",
                    "--lr-grid", "sign=1e-4:1e-2:x3", "lmo=1e-3:1e-1:x3"],
    ),
    Stage(
        "grid", "grid",
        "The fixed-target criterion: fewest iterations to F <= 1e-3 within "
        "5000, eta and momentum tuned per method. Grids are logarithmic and "
        "five decades wide, since the optimal eta spans three across these "
        "methods, and each normalized family's grid runs past its measured "
        "stability edge; any optimum landing on an edge is flagged [BOUNDARY] "
        "and is an upper bound rather than a tuned value.",
        estimate="~12 min",
        quick_args=["--m", "64", "--n", "64", "--max-iters", "1500",
                    "--methods", "signmuon", "signsgd", "muon", "sgd",
                    "--momentum-grid", "0.0,0.9",
                    "--lr-grid", "sign=1e-4:1e-2:x3", "lmo=1e-3:1e-1:x3",
                    "sgd=1e-2:1e1:x3"],
    ),
    Stage(
        "final", "final",
        "Re-runs the optima the grid stage found, with --save-histories, for "
        "the loss and gradient-norm curves.",
        args=["--save-histories"],
        estimate="~3 min",
        quick_args=["--m", "64", "--n", "64", "--max-iters", "1500",
                    "--methods", "signmuon", "signsgd", "muon", "sgd"],
    ),
]

BY_NAME = {s.name: s for s in STAGES}


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------


def environment(device: str) -> Dict:
    """Everything a reader needs to know the run was what it claims to be."""
    # One source of truth for the machine record, so the paper's hardware table
    # can be built from any results tree (`python3 -m common.hardware --scan`).
    from common.hardware import describe
    info: Dict = dict(describe(device))
    info["hardware"] = dict(info)

    for key, cmd in (("git_commit", ["git", "rev-parse", "HEAD"]),
                     ("git_branch", ["git", "rev-parse", "--abbrev-ref", "HEAD"]),
                     ("git_dirty", ["git", "status", "--porcelain"])):
        try:
            out = subprocess.run(cmd, cwd=CODE_ROOT, capture_output=True,
                                 text=True, timeout=30)
            info[key] = out.stdout.strip() if key != "git_dirty" else bool(out.stdout.strip())
        except Exception:                                   # noqa: BLE001
            info[key] = None
    return info


def selftest(log_dir: Path) -> bool:
    """Run the CPU test suite before committing the box to a sweep.

    The batched runner is a second implementation of all ten update rules, so
    the check that it still agrees with the reference loop is worth a minute
    before an hour of GPU time. Same preflight the two ``overnight.py`` drivers
    do.
    """
    log_path = log_dir / "selftest.log"
    print(f"\n{'=' * 78}\n[selftest]  ~1 min -- tests/test_code.py\n{'=' * 78}",
          flush=True)
    out = subprocess.run([sys.executable, "-m", "tests.test_code"],
                         cwd=CODE_ROOT, capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    log_path.write_text(out.stdout + "\n--- stderr ---\n" + out.stderr,
                        encoding="utf-8")
    # The runner prints "FAIL <name>" and the assertion message on the *next*
    # line, indented; without it a failure report says nothing at all. Warnings
    # go to stderr, so read the tail from stdout or they crowd out the summary.
    lines = out.stdout.strip().splitlines()
    shown = []
    for i, ln in enumerate(lines):
        if ln.startswith(("FAIL", "ERROR")):
            shown.append(ln)
            shown += [x for x in lines[i + 1:i + 2] if x.startswith("      ")]
    shown += [ln for ln in lines if "passed" in ln][-1:]
    # A skipped test is a check that did not run, so name it rather than let the
    # count in the summary line stand in for it.
    shown += [ln for ln in lines if ln.startswith("skipped: ")][-1:]
    print("\n".join(shown), flush=True)
    if out.returncode != 0:
        print(f"\nSELFTEST FAILED -- nothing was run. Full output in {log_path}.\n"
              f"If the failures above are all in parts this sweep does not use "
              f"(federated,\ndata loading, a missing optional dependency), "
              f"--no-selftest skips it. A failure in\nthe synthetic or optimizer "
              f"tests is not worth overriding.")
        return False
    print("[selftest] ok", flush=True)
    return True


def run_stage(stage: Stage, device: str, out: Path, log_dir: Path,
              quick: bool, size: Optional[tuple] = None,
              runner: Optional[str] = None) -> Dict:
    """Run one stage, teeing its output to console and to a log file."""
    argv = stage.argv(device, out, quick, size, runner)
    log_path = log_dir / f"{stage.name}.log"
    print(f"\n{'=' * 78}\n[{stage.name}]  {stage.estimate}\n{stage.why}\n"
          f"$ {' '.join(argv[1:])}\n{'=' * 78}", flush=True)

    start = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(argv)}\n\n")
        log.flush()
        proc = subprocess.Popen(argv, cwd=CODE_ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        code = proc.wait()
    elapsed = time.time() - start

    status = "ok" if code == 0 else f"FAILED (exit {code})"
    print(f"[{stage.name}] {status} in {elapsed / 60:.1f} min -> {log_path}",
          flush=True)
    return {"stage": stage.name, "mode": stage.mode, "argv": argv,
            "exit_code": code, "seconds": round(elapsed, 1),
            "log": str(log_path)}


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------


def _load(out: Path, mode: str) -> List[Dict]:
    """Every ``<method>/<mode>.json`` under ``out``, in METHOD_CLASSES order."""
    from synthetic.benchmark import DEFAULT_METHODS

    found = []
    for method in DEFAULT_METHODS:
        path = out / method / f"{mode}.json"
        if not path.exists():
            continue
        try:
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
        except json.JSONDecodeError:
            continue
        payload["_method"] = method
        found.append(payload)
    return found


def _problem_line(payloads: Sequence[Dict]) -> str:
    if not payloads:
        return ""
    p = payloads[0]["problem"]
    return (f"`{p['m']}x{p['n']}`, spectrum `{p['spectrum']}`, basis "
            f"`{p['basis']}`, L = {p['L']:.4g}, sigma = {p['sigma']:.4g}, "
            f"condition number = {p['condition_number']:.4g}, "
            f"lmo_dtype `{p['lmo_dtype']}`, seeds {p['problem_seeds']}, "
            f"X0 seed {p['init_seed']}, runner `{p.get('runner', '?')}`\n")


def _fmt(value, spec: str = ".4g") -> str:
    if value is None:
        return "--"
    if isinstance(value, float) and value != value:
        return "--"
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


_EDGE_LEGEND = ("A `†` marks a value censored at its grid edge: an upper bound "
                "rather than a tuned optimum.")


def _edge_flag(record: Dict, axis: str) -> str:
    """Mark a tuned value censored at a grid edge, on the axis that was censored.

    The flag was recorded by every tuning mode but rendered only in the tuned
    comparison, so a censored optimum in the per-budget or per-kappa breakdown
    read as a measured one. It goes on the cell it belongs to -- a momentum hit
    marks momentum, not the perfectly good learning rate beside it -- and is a
    dagger rather than a column because most rows do not carry it.
    """
    return "†" if axis in (record.get("on_grid_boundary") or []) else ""


def summarize(out: Path) -> str:
    """Rebuild SUMMARY.md from whatever result JSON is on disk."""
    lines: List[str] = ["# Synthetic benchmark results", ""]

    manifest_path = out / "MANIFEST.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            man = json.load(f)
        env = man.get("environment", {})
        lines += [
            f"Run {man.get('started', '?')} — "
            f"{env.get('gpu', 'CPU')}, torch {env.get('torch', '?')}, "
            f"commit `{(env.get('git_commit') or '?')[:10]}`"
            f"{' (dirty tree)' if env.get('git_dirty') else ''}", ""]
        rows = ["| stage | exit | minutes |", "| :--- | :--- | ---: |"]
        for st in man.get("stages", []):
            rows.append(f"| {st['stage']} | {st['exit_code']} | "
                        f"{st['seconds'] / 60:.1f} |")
        lines += rows + [""]

    # -- grid / final ---------------------------------------------------
    for mode, title in (("grid", "Tuned comparison (`tab:synthetic_tuned`)"),
                        ("final", "Re-run at the tuned optima "
                                  "(`fig:synthetic_main`)")):
        payloads = _load(out, mode)
        if not payloads:
            continue
        lines += [f"## {title}", "", _problem_line(payloads),
                  "| method | iterations to target | best F | min ‖∇F‖ | lr | momentum | schedule | on grid boundary |",
                  "| :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- |"]
        for p in payloads:
            r = p["result"]
            it = (f"{r['iters_to_converge']:.0f}" if r["reached_target"]
                  else f">{p['problem']['max_iters']}")
            edge = ", ".join(r.get("on_grid_boundary") or []) or "—"
            lines.append(
                f"| {p['_method']} | {it} | {_fmt(r['best_f'], '.3e')} | "
                f"{_fmt(r['best_gnorm'], '.3e')} | {_fmt(r['kwargs']['lr'])} | "
                f"{_fmt(r['kwargs'].get('momentum'))} | {r.get('schedule', '?')} | "
                f"{edge} |")
        lines.append("")

    # -- alignment ------------------------------------------------------
    payloads = _load(out, "alignment")
    if payloads:
        lines += ["## Gradient/step alignment", "",
                  "`rho_t = <grad F(X_t), d_t> / (||grad F(X_t)||_F ||d_t||_F)`. "
                  "The descent lemma needs it positive; the divergence theorems "
                  "make it negative. The reference column is the closed form for "
                  "`d = compressor(grad F)` at `X_0` with no momentum, so it is "
                  "only comparable to a row whose tuned momentum is 0. "
                  + _EDGE_LEGEND, "",
                  _problem_line(payloads),
                  "| method | min rho | 1st pct | median | mean | % negative | closed form | tuned |",
                  "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |"]
        for p in payloads:
            r = p["result"]
            rho = r.get("rho")
            cfg = (f"eta={_fmt(r['kwargs']['lr'])}{_edge_flag(r, 'lr')}, "
                   f"mu={_fmt(r['kwargs'].get('momentum', 0))}"
                   f"{_edge_flag(r, 'momentum')}")
            ref = _fmt(r.get("rho_reference_at_X0"), ".4f")
            if not rho:
                lines.append(f"| {p['_method']} | — | — | — | — | — | {ref} | {cfg} |")
                continue
            lines.append(
                f"| {p['_method']} | {rho['min']:.4f} | {rho['p01']:.4f} | "
                f"{rho['median']:.4f} | {rho['mean']:.4f} | "
                f"{100 * rho['frac_negative']:.2f}% | {ref} | {cfg} |")
        lines.append("")

    # -- floor ----------------------------------------------------------
    payloads = _load(out, "floor")
    if payloads:
        lines += ["## Accuracy floor of a constant step", "",
                  "Slope of `log(plateau)` against `log(eta)`. Balancing the two "
                  "terms of the descent lemma predicts the gradient floor is "
                  "linear in `eta` (slope 1) with coefficient "
                  "`L||s||_F / (2 rho)`; the slope is what the fit tests, the "
                  "coefficient being an upper bound rather than a prediction. "
                  "SignMuon and SignSGD share `||s||_F = sqrt(mn)` exactly, so "
                  "any gap between their floors is attributable to `rho` alone. "
                  "The last column is the lemma's *other* term, the `eta^2` "
                  "coefficient `L||s||_F^2 / 2`, printed as a scale for `F∞`.",
                  "", _problem_line(payloads),
                  "| method | settled points | d log‖∇F‖/d log η | R² | d log F/d log η | R² | L‖s‖²/2 |",
                  "| :--- | :--- | ---: | ---: | ---: | ---: | ---: |"]
        for p in payloads:
            r = p["result"]
            s = r.get("step_norm")
            pred = (p["problem"]["L"] * s * s / 2
                    if isinstance(s, (int, float)) and s == s else None)
            n = f"{r.get('n_settled', 0)}/{len(r.get('rows', []))}"
            if r.get("n_settled", 0) < 2:
                lines.append(f"| {p['_method']} | {n} | no floor | | | | "
                             f"{_fmt(pred)} |")
                continue
            lines.append(
                f"| {p['_method']} | {n} | {r['slope_gnorm']:.3f} | "
                f"{r['r2_gnorm']:.3f} | {r['slope_f']:.3f} | {r['r2_f']:.3f} | "
                f"{_fmt(pred)} |")
        lines += ["", "<details><summary>Per-η plateaus</summary>", ""]
        for p in payloads:
            lines += [f"**{p['_method']}**", "",
                      "| η | iterations | F∞ | ‖∇F‖∞ | settled |",
                      "| ---: | ---: | ---: | ---: | :--- |"]
            for row in p["result"].get("rows", []):
                lines.append(
                    f"| {_fmt(row['lr'])} | {row.get('iters', '?')} | "
                    f"{_fmt(row['f_inf'], '.4e')} | {_fmt(row['g_inf'], '.4e')} | "
                    f"{'yes' if row.get('settled') else 'no'} |")
            lines.append("")
        lines += ["</details>", ""]

    # -- horizon --------------------------------------------------------
    payloads = _load(out, "horizon")
    if payloads:
        lines += ["## Budget scaling", "",
                  "Tuned separately at each budget `T`, then fitted. `p` is "
                  "fitted on `min_t ‖∇F(X_t)‖_*²`, the squared dual norm the "
                  "theorems bound — ℓ1 for the sign family, nuclear for the LMO "
                  "family; the Frobenius column is the same trajectory in a "
                  "family-independent norm. `p = q = 1/2` is the nonconvex "
                  "L-smooth regime the theorems prove, `p = q = 1` the strongly "
                  "convex one. SGD has no floor, so no power law fits it. "
                  + _EDGE_LEGEND, "",
                  _problem_line(payloads),
                  "| method | dual | p (‖∇F‖_*²) | R² | p (‖∇F‖_F) | R² | p (F) | R² | q (η*) | R² |",
                  "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for p in payloads:
            r = p["result"]
            lines.append(
                f"| {p['_method']} | {r.get('dual_norm', '—')} | "
                f"{_fmt(r.get('exponent_dual_sq'), '.3f')} | "
                f"{_fmt(r.get('r2_dual'), '.3f')} | "
                f"{_fmt(r['exponent_gnorm'], '.3f')} | "
                f"{_fmt(r['r2_gnorm'], '.3f')} | {_fmt(r['exponent_f'], '.3f')} | "
                f"{_fmt(r['r2_f'], '.3f')} | {_fmt(r['exponent_lr'], '.3f')} | "
                f"{_fmt(r['r2_lr'], '.3f')} |")
        lines += ["", "<details><summary>Per-budget optima</summary>", ""]
        for p in payloads:
            lines += [f"**{p['_method']}**", "",
                      "| T | η* | momentum | schedule | best F | min ‖∇F‖_F | min ‖∇F‖_* |",
                      "| ---: | ---: | ---: | :--- | ---: | ---: | ---: |"]
            for row in p["result"].get("rows", []):
                lines.append(
                    f"| {row['T']} | {_fmt(row['lr'])}{_edge_flag(row, 'lr')} | "
                    f"{_fmt(row.get('momentum'))}{_edge_flag(row, 'momentum')} | "
                    f"{row.get('schedule', '?')} | "
                    f"{_fmt(row['best_f'], '.4e')} | "
                    f"{_fmt(row['best_gnorm'], '.4e')} | "
                    f"{_fmt(row.get('best_dual'), '.4e')} |")
            lines.append("")
        lines += ["</details>", ""]

    # -- stability ------------------------------------------------------
    payloads = _load(out, "stability")
    if payloads:
        L = payloads[0]["problem"]["L"]
        lines += ["## Step-size stability edge", "",
                  f"L = {L:.4g}, so `2/L` = {2 / L:.4g}. **SGD is the control** "
                  "— its `eta_max` must land on `2/L`. If the operative trust "
                  "region were the Frobenius ball, `eta_max*||s||_F` would be "
                  "family-independent and also near `2/L`; the spread measures "
                  "how far off the Frobenius bound is for each step geometry.",
                  "", _problem_line(payloads),
                  "A `>` means the search hit its ceiling without finding an "
                  "edge, so that row is a lower bound, not a measurement. Adam "
                  "lands there by construction: its step is bounded by roughly "
                  "`lr` whatever the gradient does, so it oscillates instead of "
                  "diverging.", "",
                  "| method | η_max | ‖s‖_F | η_max·‖s‖_F | ÷ (2/L) |",
                  "| :--- | ---: | ---: | ---: | ---: |"]
        for p in payloads:
            r = p["result"]
            s = r.get("step_norm")
            eta = ("&gt; " if r.get("censored") else "") + _fmt(r["eta_max"])
            if not (isinstance(s, (int, float)) and s == s):
                lines.append(f"| {p['_method']} | {eta} | — | — | "
                             f"{r['eta_max'] / (2 / L):.3f} |")
                continue
            lines.append(
                f"| {p['_method']} | {eta} | {_fmt(s)} | "
                f"{_fmt(r['step_length'])} | {r['step_length'] / (2 / L):.3f} |")
        lines.append("")

    # -- kappa ----------------------------------------------------------
    payloads = _load(out, "kappa")
    if payloads:
        kappas = payloads[0]["result"].get("kappas", []) or []
        header = " | ".join(f"κ={k:g}" for k in kappas)
        lines += ["## Condition-number sweep", "",
                  "Best `‖∇F‖` reached within the budget, tuned at each κ. "
                  "Spectra are log-spaced with `L = 1`, so κ is exact. "
                  + _EDGE_LEGEND, "",
                  _problem_line(payloads),
                  f"| method | {header} | d log‖∇F‖/d log κ | R² |",
                  "| :--- |" + " ---: |" * (len(kappas) + 2)]
        for p in payloads:
            r = p["result"]
            cells = " | ".join(_fmt(row["best_gnorm"], ".3e")
                               for row in r.get("rows", []))
            lines.append(f"| {p['_method']} | {cells} | "
                         f"{_fmt(r.get('exponent_kappa'), '.3f')} | "
                         f"{_fmt(r.get('r2_kappa'), '.3f')} |")
        lines += ["", "<details><summary>Per-κ optima</summary>", ""]
        for p in payloads:
            lines += [f"**{p['_method']}**", "",
                      "| κ | η* | momentum | schedule | iterations | min ‖∇F‖ |",
                      "| ---: | ---: | ---: | :--- | ---: | ---: |"]
            for row in p["result"].get("rows", []):
                it = (f"{row['iters']:.0f}" if row.get("reached") else "none")
                lines.append(
                    f"| {row['kappa']:g} | {_fmt(row['lr'])}{_edge_flag(row, 'lr')} | "
                    f"{_fmt(row.get('momentum'))}{_edge_flag(row, 'momentum')} | "
                    f"{row.get('schedule', '?')} | "
                    f"{it} | {_fmt(row['best_gnorm'], '.4e')} |")
            lines.append("")
        lines += ["</details>", ""]

    if len(lines) <= 2:
        lines.append("_No result files found._")
    return "\n".join(lines) + "\n"


def bundle(out: Path) -> Optional[Path]:
    """Zip the whole result tree into one file to copy off a remote GPU box.

    ``code/results/`` is gitignored, so on a remote machine there is otherwise
    no single artifact to bring back. Everything here is text, so it compresses
    to a fraction of its size even with the saved loss curves.
    """
    import zipfile

    archive = out.parent / f"{out.name}_results.zip"
    files = sorted(p for p in out.rglob("*") if p.is_file() and p != archive)
    if not files:
        return None
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, path.relative_to(out.parent))
    return archive


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def get_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default=None,
                   help="output directory (default: results/synthetic/)")
    # No argparse `choices` here: it validates each token *before* commas can be
    # split off, so `--stages grid,final` would fail on a token the user never
    # typed. Validation happens after splitting, in main().
    p.add_argument("--stages", nargs="+", default=None, metavar="NAME",
                   help="space- or comma-separated. Choices: "
                        + ", ".join(s.name for s in STAGES)
                        + ". Default: all of them")
    p.add_argument("--quick", action="store_true",
                   help="tiny sizes and coarse grids, ~2 min for the whole "
                        "pipeline; run this first")
    p.add_argument("--m", type=int, default=None, metavar="M",
                   help="Override the problem size for EVERY stage, keeping the "
                        "real grids and iteration counts, and write to a "
                        "separate tree. Not a cost knob: below ~200x200 a step "
                        "is dominated by kernel-launch latency rather than "
                        "arithmetic, so --m 20 and --m 100 take about the same "
                        "time. Default: 100.")
    p.add_argument("--n", type=int, default=None, metavar="N",
                   help="Columns; defaults to --m when only --m is given.")
    p.add_argument("--runner", choices=["batched", "sequential"], default=None,
                   help="passed to every stage. The default (batched) advances "
                        "a whole grid as one trajectory; 'sequential' is the "
                        "reference one-run-at-a-time loop and is hours slower")
    p.add_argument("--force", action="store_true",
                   help="re-run stages whose output already exists")
    p.add_argument("--no-selftest", action="store_true",
                   help="skip the tests/test_code.py preflight (not advised: it "
                        "is what checks the batched runner against the "
                        "reference loop)")
    p.add_argument("--archive", "--summarize-only", dest="archive",
                   action="store_true",
                   help="rebuild SUMMARY.md and the .zip from the JSON already "
                        "on disk, then exit; runs nothing")
    p.add_argument("--list", action="store_true", help="describe the stages and exit")
    return p.parse_args()


def already_done(stage: Stage, out: Path) -> bool:
    from synthetic.benchmark import DEFAULT_METHODS
    return any((out / m / f"{stage.mode}.json").exists() for m in DEFAULT_METHODS)


def main() -> int:
    args = get_args()
    size = None
    if args.m is not None or args.n is not None:
        m = args.m if args.m is not None else args.n
        size = (m, args.n if args.n is not None else m)
    if args.out:
        out = Path(args.out)
    else:
        # A quick pass and a non-default size each get their own tree. Sharing
        # one would leave smoke-test JSON where the reported results belong, and
        # the already-done check would then skip the real run.
        out = results_root() / ("synthetic_quick" if args.quick else "synthetic")
        if size is not None:
            out = out.with_name(f"{out.name}_{size[0]}x{size[1]}")

    if args.list:
        for s in STAGES:
            print(f"\n{s.name}  [{s.estimate}]\n  {s.why}")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    if args.archive:
        path = out / "SUMMARY.md"
        path.write_text(summarize(out), encoding="utf-8")
        archive = bundle(out)
        if archive is None:
            print(f"Nothing to archive: {out.resolve()} holds no results yet.")
            return 1
        print(f"Wrote {path.resolve()}\n"
              f"Wrote {archive.resolve()} "
              f"({archive.stat().st_size / 1024:.0f} KB)\n\n"
              f"Download that .zip -- in VS Code, right-click it in the "
              f"Explorer\nunder code/results/ and choose Download.")
        return 0

    try:
        requested = split_list_arg(args.stages, [s.name for s in STAGES], "stage")
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    names = requested or [s.name for s in STAGES]
    selected = [BY_NAME[n] for n in names]
    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # The machine record, printed verbatim because this sentence is what the
    # reproducibility appendix quotes, and stamped into MANIFEST.json so
    # `python3 -m common.hardware --scan results --latex` can build the table
    # from the results tree rather than from whichever box compiles the LaTeX.
    env = environment(args.device)
    from common.hardware import as_sentence
    print(as_sentence(env))
    if not env.get("cuda_available"):
        print("WARNING: CUDA is not available; this will run on CPU and the "
              "estimates below are far too optimistic.")
    elif not env.get("gpu_homogeneous", True):
        # Mixed-card boxes are how a run silently lands on the wrong GPU.
        print(f"NOTE: this box has {env.get('gpu_count')} GPUs of differing "
              f"models; --device {args.device} selected index "
              f"{env.get('gpu_index')} ({env.get('gpu')}).")
    if env.get("git_dirty"):
        print("NOTE: the working tree has uncommitted changes; MANIFEST.json "
              "records the commit but not those edits.")
    print(f"\nStages: {', '.join(names)}"
          f"{'   [QUICK]' if args.quick else ''}"
          f"{f'   [{size[0]}x{size[1]}]' if size else ''}"
          f"\nOutput: {out.resolve()}")

    t0 = time.time()
    if not args.no_selftest and not selftest(log_dir):
        return 2

    started = time.strftime("%Y-%m-%d %H:%M:%S")
    records: List[Dict] = []
    for stage in selected:
        if not args.force and not args.quick and already_done(stage, out):
            print(f"\n[{stage.name}] already has output; skipping (--force to redo)")
            records.append({"stage": stage.name, "mode": stage.mode,
                            "exit_code": 0, "seconds": 0.0, "skipped": True})
            continue
        records.append(run_stage(stage, args.device, out, log_dir, args.quick,
                                 size, args.runner))

        manifest = {"started": started, "quick": args.quick,
                    "size": list(size) if size else None,
                    "environment": env, "stages": records}
        with open(out / "MANIFEST.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        (out / "SUMMARY.md").write_text(summarize(out), encoding="utf-8")

    failed = [r["stage"] for r in records if r["exit_code"] != 0]
    print(f"\n{'=' * 78}\nTotal {(time.time() - t0) / 60:.1f} min. "
          f"{len(records) - len(failed)}/{len(records)} stages ok.")
    if failed:
        print(f"FAILED: {', '.join(failed)}  (see {log_dir})")

    archive = bundle(out)
    print(f"\nRead    {(out / 'SUMMARY.md').resolve()}")
    print(f"Logs    {log_dir.resolve()}")
    if archive:
        print(f"Bundle  {archive.resolve()} "
              f"({archive.stat().st_size / 1024:.0f} KB)")
        print("\nThat .zip is the one artifact to bring back: in VS Code, "
              "right-click it\nin the Explorer under code/results/ and choose "
              "Download. --archive rebuilds\nit at any time without re-running "
              "anything.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
