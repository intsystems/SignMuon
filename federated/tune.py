"""Equal-budget, validation-only learning-rate search for the federated methods.

    # eta_0 per method, identical budget for all eleven
    python3 -m federated.tune --stage lr --device cuda:0 --lr-scaling unit-gain

    # is the optimal auxiliary rate method-independent?
    python3 -m federated.tune --stage aux --device cuda:0

This is the federated twin of ``centralized/tune.py`` and shares its lattice, its
boundary check and its job bookkeeping, so the two settings search the *same*
grid and mean the same thing by ``eta_0``. Three properties make it a protocol
rather than a sweep:

1. **Selection is on validation accuracy only** (``--split tune``: 5k images held
   out of the 50k *before* the client partition, the same 5k the centralized path
   holds out). The test set is not merely unused for ranking -- with
   ``--split tune`` it is never *evaluated*, so there is no test number in the log
   to be tempted by. This replaces ``federated/grid.py``, which ranked
   configurations by the test accuracy printed each round.
2. **Equal budget** on a multiplicatively anchored 1-2-5 lattice, so a tuned rate
   is quotable as ``0.02`` and two methods anchored at different places snap to
   the same grid.
3. **Boundary check**: a winner on a grid endpoint is a failure, not a result.

Anchors
-------
The published federated learning rates (Table~5) were tuned under the ``legacy``
per-layer rule -- one global rate for the sign family, Muon's aspect factor for
the LMO family. A rule changes what ``eta_0`` *means*, so each anchor is
transported analytically:

    anchor(rule) = anchor_legacy * geomean(lambda_legacy) / geomean(lambda_rule)

with the geometric mean taken over the model's own matrix-parameter shapes. That
is exact for the quantity being anchored -- the typical per-layer rate -- and it
means adding a model or a rule needs no new constants. On CNN2 the sign family's
multiplier spans 7.8x (``conv1`` 0.115, ``fc1`` 0.0147), so this transport moves
the sign anchors by ~29x and leaves the LMO anchors untouched.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from common.lr_scaling import (RULES, describe_rule, fan_in_out, layer_multiplier,
                               resolve_rule)
from common.paths import scrub_text
from common.utils import results_root, split_param_names
from federated.algorithms import method_family
# One lattice for both settings: same helpers, same resolution, same job identity.
from centralized.tune import (best_of, boundary_warning, canonical_tag, extend_grid,
                              lattice_index, lattice_value, round_grid)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # code/, the package root

#: All eleven methods, in the paper's order.
ALL_METHODS = ["signmuon", "muonusign", "muonsign", "ef21signmuon",
               "ef21muonusign", "ef21muonsign",
               "muon", "muonserver", "signsgd", "sgd", "adam"]

#: Anchor for each method's grid, in the ``legacy`` parameterization. The four
#: values the paper reports for N=10 (Table 5) plus siblings for the methods that
#: were never run: MuonUSign<->Muon (both take a full LMO step),
#: MuonSign<->SignMuon (both take a unit-sign step),
#: EF21-SignMuon<->EF21-MuonUSign (both step along an EF21 estimate of a polar
#: factor).
LEGACY_ANCHORS: Dict[str, float] = {
    "signmuon": 1e-3,          # Table 5
    "muonsign": 1e-3,          # sibling of signmuon
    "signsgd": 1e-3,           # Table 5
    "muon": 5e-2,              # Table 5
    "muonserver": 5e-2,        # sibling of muon: same step family, server-side LMO
    "muonusign": 5e-2,         # sibling of muon
    "ef21muonusign": 2e-2,     # Table 5
    "ef21muonsign": 3.5e-2,    # Table 5
    "ef21signmuon": 2e-2,      # sibling of ef21muonusign
    "sgd": 3e-2,               # Table 5
    "adam": 1e-3,              # Table 5
}

AUX_GRID = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3]

#: Default horizon per stage, and they differ for a reason.
#:
#: ``lr`` ranks rates that are then *reported* at 2000 rounds, so it has to run
#: there too. A shorter proxy is not a noisier version of the same measurement: the
#: cosine schedule anneals to zero over each run's own horizon, so a 400-round run
#: spends nearly its whole budget at a decayed rate, and on 2026-07-30 the proxy
#: picked the wrong rate for both methods it was checked on. ``federated.overnight``
#: was fixed then; this standalone path kept the proxy for another day, which meant
#: the by-hand recipe in REPRODUCE.md and the driver ran different protocols.
#:
#: ``aux`` compares the *argmax over lr_aux* between two methods measured at the
#: same horizon, and a shared horizon bias cancels in that comparison, so it stays
#: short.
STAGE_ROUNDS = {"lr": 2000, "aux": 400}


# --------------------------------------------------------------------------
# Transporting an anchor between per-layer rules
# --------------------------------------------------------------------------


def matrix_shapes(dataset: str, model: str, n_head_tensors: int = 2
                  ) -> List[Tuple[str, Tuple[int, ...]]]:
    """``(name, shape)`` of every parameter the matrix rule applies to."""
    from federated.main import build_model                     # local: torch import
    net = build_model(dataset, model)
    names, _ = split_param_names(net, n_head_tensors)
    named = dict(net.named_parameters())
    return [(n, tuple(named[n].shape)) for n in names]


def _geomean_multiplier(rule_name: str, family: str,
                        shapes: Sequence[Tuple[str, Tuple[int, ...]]]) -> float:
    rule = resolve_rule(rule_name)
    vals = [layer_multiplier(rule, family, s) for _, s in shapes]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else 1.0


def anchor_for(method: str, rule: str,
               shapes: Sequence[Tuple[str, Tuple[int, ...]]],
               scale_baselines: bool = False) -> float:
    """The ``legacy`` anchor of ``method``, transported to ``rule``.

    Returns the legacy value unchanged when the method has no per-layer family
    (``sgd``, ``adam`` without ``--scale-baselines``) or when the rule leaves the
    typical multiplier alone.
    """
    base = LEGACY_ANCHORS[method]
    family = method_family(method, scale_baselines)
    if family is None:
        return base
    ref = _geomean_multiplier("legacy", family, shapes)
    now = _geomean_multiplier(rule, family, shapes)
    return base * ref / now


def describe_anchors(rule: str, dataset: str, model: str,
                     methods: Sequence[str] = ALL_METHODS) -> str:
    shapes = matrix_shapes(dataset, model)
    lines = [f"Grid anchors under rule '{rule}' ({model}/{dataset}, "
             f"{len(shapes)} matrix parameters)",
             f"  {'method':<16}{'family':>8}{'legacy':>12}{'anchor':>12}{'lattice':>12}"]
    for m in methods:
        fam = method_family(m) or "-"
        a = anchor_for(m, rule, shapes)
        lines.append(f"  {m:<16}{fam:>8}{LEGACY_ANCHORS[m]:>12.6g}{a:>12.6g}"
                     f"{lattice_value(lattice_index(a)):>12.6g}")

    # The per-layer multipliers themselves, which is what the anchors are derived
    # from and what makes the transport auditable. The two families get different
    # rules, so both are printed; the spread is the number that matters, because
    # it is what a single global rate would have to paper over.
    resolved = resolve_rule(rule)
    for fam in ("lmo", "sign"):
        lines.append("")
        lines.append(describe_rule(resolved, fam, shapes))

    # And the comparison the README quotes: at ONE global rate, how much longer is
    # a sign-family step than the corresponding LMO-family step, layer by layer?
    lines += ["", "  Step-length ratio at a single global rate "
                  "(||s||_F sign / ||s||_F lmo), per layer:"]
    for name, shape in shapes:
        m_out, n_in = fan_in_out(shape)
        ratio = math.sqrt(m_out * n_in) / math.sqrt(min(m_out, n_in))
        lines.append(f"    {name:<34}{ratio:>10.1f}x")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Running one configuration
# --------------------------------------------------------------------------

_VAL_RE = re.compile(r"val_acc\s+mean of last\s+\d+\s*:\s*([\d.]+)%")
_TEST_RE = re.compile(r"test_acc\s+mean of last\s+\d+\s*:\s*([\d.]+)%")
_ROUND_TIME_RE = re.compile(r"median round time\s*:\s*([\d.]+)s")
_TARGET_RE = re.compile(r"rounds to [\d.]+% \w+ acc\s*:\s*(\d+)")
_TIE_RE = re.compile(r"majority-vote tie fraction\s*:\s*([\d.]+) mean")
# The three numbers the experiment's *motivation* rests on. Parsed here so they
# reach the overnight report rather than only the individual run logs -- a
# diagnostic nobody reads is a diagnostic that does not exist.
_GAIN_RE = re.compile(r"per-layer gain spread\s*:\s*[\d.]+x at round 1, ([\d.]+)x final")
_ZERO_RE = re.compile(r"uplink zero fraction\s*:\s*([\d.]+) mean")
_ROUNDTRIP_RE = re.compile(r"vs full precision\s*:.*?([\d.]+)x round trip")


def eval_freq_for(rounds: int, points: int = 20) -> int:
    """Evaluate ``points`` times over the run.

    The reported number is the tail mean of the last ``--last-k`` evaluations, so
    the evaluation count -- not the round count -- is what has to stay comparable
    between a short tuning run and a long final one. Twenty points puts the tail
    mean over the last quarter of training at either horizon.
    """
    return max(1, rounds // max(1, points))


def run_one(args, *, lr: float, lr_aux: float, lr_scaling: str, method: str,
            rounds: int, tag: str, split: str = "tune", seed: Optional[int] = None,
            extra: Sequence[str] = ()) -> Optional[Dict[str, float]]:
    """Launch one federated run; return its metrics, or ``None`` on failure.

    Selection uses ``val_acc`` averaged over the last ``--last-k`` evaluations.
    ``test_acc`` is parsed when present, but a ``split="tune"`` child never
    evaluates the test set, so there is nothing to select on by mistake.
    """
    tag = canonical_tag(tag, epochs=rounds, split=split, seed=seed)
    freq = eval_freq_for(rounds, args.eval_points)

    cmd = [
        sys.executable, "-m", "federated.main",
        "--dataset", args.dataset, "--model", args.model,
        "--algorithm", method, "--lr-scaling", lr_scaling,
        "--split", split, "--val-seed", str(args.val_seed),
        "--rounds", str(rounds), "--n_parties", str(args.n_parties),
        "--n_steps", str(args.n_steps), "--batch_size", str(args.batch_size),
        "--lr", repr(lr), "--lr-aux", repr(lr_aux),
        "--momentum", str(args.momentum), "--weight_decay", str(args.weight_decay),
        "--partition", args.partition, "--beta", str(args.beta),
        "--last-k", str(args.last_k), "--target-acc", str(args.target_acc),
        "--eval_freq", str(freq), "--device", args.device,
        "--seed", str(args.seed if seed is None else seed),
        "--data", args.data, "--loader", args.loader,
        "--mv-ties", args.mv_ties, "--uplink-zeros", args.uplink_zeros,
        "--run_name", f"{'tune' if split == 'tune' else 'final'}_{tag}",
        *extra,
    ]
    if getattr(args, "nondeterministic", False):
        cmd.append("--nondeterministic")
    if getattr(args, "download", False):
        cmd.append("--download")

    log_dir = results_root() / "federated_tuning_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{tag}.log"

    print(f"  lr={lr:<12.6g} lr_aux={lr_aux:<10.6g} -> ", end="", flush=True)
    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT, text=True)
    wall = time.perf_counter() - t0
    # The child prints where it saved its run, and a traceback quotes the source
    # tree, so its stdout carries the writing machine's paths. These logs live
    # under `results/`, which now ships, so they are rewritten in place before
    # anything reads them -- a fresh run therefore leaves the tree clean rather
    # than accumulating files the preflight will refuse to start on.
    text = scrub_text(log_path.read_text(encoding="utf-8", errors="replace"))
    log_path.write_text(text, encoding="utf-8")

    if proc.returncode != 0:
        tail = "".join(text.splitlines(keepends=True)[-3:]).strip()
        print(f"FAILED (exit {proc.returncode}) {tail[:160]}")
        return None

    run_name = f"{'tune' if split == 'tune' else 'final'}_{tag}"
    metrics = (results_root() / "federated" / run_name /
               f"seed{args.seed if seed is None else seed}" / "metrics.json")
    out: Dict[str, float] = {"lr": lr, "lr_aux": lr_aux, "rounds": rounds,
                             "split": split, "log": scrub_text(str(log_path)),
                             "metrics": scrub_text(str(metrics)),
                             # Recorded so that a consumer of state.json does not
                             # have to parse them back out of the job key.
                             "method": method, "lr_scaling": lr_scaling,
                             "seed": args.seed if seed is None else seed,
                             # Wall clock INCLUDING process start-up and evaluation:
                             # what the overnight scheduler has to budget for.
                             "wall_seconds": wall}
    for key, pattern, cast in (("val_acc", _VAL_RE, float),
                               ("test_acc", _TEST_RE, float),
                               ("round_seconds", _ROUND_TIME_RE, float),
                               ("rounds_to_target", _TARGET_RE, int),
                               ("mv_tie_frac", _TIE_RE, float),
                               ("gain_spread", _GAIN_RE, float),
                               ("uplink_zero_frac", _ZERO_RE, float),
                               ("round_trip_reduction", _ROUNDTRIP_RE, float)):
        m = pattern.search(text)
        if m:
            out[key] = cast(m.group(1))

    key = "val_acc" if "val_acc" in out else "test_acc"
    if key not in out:
        print(f"no metric found (see {log_path.name})")
        return None
    print(f"{key.replace('_', ' ')} {out[key]:.2f}%"
          + (f"  [{out['round_seconds']:.3f}s/round]" if "round_seconds" in out else ""))
    return out


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------


def stage_lr(args) -> Dict[str, Dict]:
    """Tune ``eta_0`` per method under a fixed rule, equal budget for all."""
    methods = args.methods or ALL_METHODS
    shapes = matrix_shapes(args.dataset, args.model)
    print(describe_anchors(args.lr_scaling, args.dataset, args.model, methods))
    out: Dict[str, Dict] = {}

    for method in methods:
        base = anchor_for(method, args.lr_scaling, shapes)
        grid = round_grid(base, points=args.lr_points)
        print(f"\n=== lr stage: {method} ({args.lr_points} pts around {base:.4g}) ===")
        results = [run_one(args, lr=lr, lr_aux=args.lr_aux, lr_scaling=args.lr_scaling,
                           method=method, rounds=args.rounds,
                           tag=f"lr_{method}_{args.lr_scaling.replace(':', '')}_{lr:.4g}")
                   for lr in grid]
        results = [r for r in results if r]
        best = best_of(results)
        if best is None:
            print(f"  every run failed for {method}")
            continue

        for _ in range(args.lr_extend_rounds):
            warn = boundary_warning(best, grid)
            if not warn:
                break
            print(warn)
            grid = extend_grid(grid, low="LOW end" in warn, points=args.lr_extend_points)
            print(f"  -> extending to [{min(grid):.4g}, {max(grid):.4g}]")
            for lr in grid:
                if any(math.isclose(lr, r["lr"], rel_tol=1e-9) for r in results):
                    continue
                r = run_one(args, lr=lr, lr_aux=args.lr_aux, lr_scaling=args.lr_scaling,
                            method=method, rounds=args.rounds,
                            tag=f"lr_{method}_{args.lr_scaling.replace(':', '')}_{lr:.4g}")
                if r:
                    results.append(r)
            best = best_of(results)

        out[method] = {"best": best, "all": results, "grid": grid,
                       "boundary_warning": boundary_warning(best, grid)}
        print(f"  BEST: eta_0={best['lr']:.6g}, val {best['val_acc']:.2f}% "
              f"({len(results)} configs)")

    if out:
        print(f"\n--- eta_0 per method (rule '{args.lr_scaling}') ---")
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
        fam = method_family(method)
        if fam is None:
            continue
        groups.setdefault(fam, []).append((method, d["best"]["lr"]))

    print("\n--- family agreement (the scaling rule's prediction) ---")
    for family, entries in groups.items():
        if len(entries) < 2:
            continue
        vals = [v for _, v in entries]
        spread = max(vals) / min(vals)
        names = ", ".join(f"{m}={v:.4g}" for m, v in entries)
        verdict = ("AGREE (within 2.5x, i.e. one lattice step)" if spread <= 2.5
                   else f"DISAGREE by {spread:.1f}x")
        print(f"  {family:<6} {names}")
        print(f"  {'':<6} spread {spread:.2f}x -> {verdict}")


def stage_aux(args) -> Dict:
    """Is the optimal auxiliary rate method-independent?

    The auxiliary group is AdamW on the same parameters for every method, so its
    optimum should not depend on the matrix rule. Verifying that on two anchor
    methods spanning an order of magnitude in ``eta_0`` earns the right to fix one
    ``lr_aux`` globally instead of paying for a 2-D grid per method.
    """
    anchors = args.aux_anchors or ["signmuon", "muon"]
    shapes = matrix_shapes(args.dataset, args.model)
    out: Dict[str, Dict] = {}
    for method in anchors:
        base = anchor_for(method, args.lr_scaling, shapes)
        lr_grid = round_grid(base, points=3)
        print(f"\n=== aux stage: {method} (eta_0 around {base:.4g}) ===")
        results = []
        for lr in lr_grid:
            for lr_aux in AUX_GRID:
                tag = f"aux_{method}_{args.lr_scaling.replace(':', '')}_lr{lr:.4g}_aux{lr_aux:.4g}"
                r = run_one(args, lr=lr, lr_aux=lr_aux, lr_scaling=args.lr_scaling,
                            method=method, rounds=args.rounds, tag=tag)
                if r:
                    results.append(r)
        best = best_of(results)
        if best is None:
            print(f"  no successful run for {method}")
            continue
        out[method] = {"best": best, "all": results}
        print(f"  best: eta_0={best['lr']:.4g}, lr_aux={best['lr_aux']:.4g}, "
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


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def add_common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Arguments shared with ``federated.overnight``."""
    p.add_argument("--dataset", type=str, default="cifar10")
    p.add_argument("--model", type=str, default="cnn2")
    p.add_argument("--n_parties", type=int, default=11,
                   help="Clients. ODD by default and deliberately so: with an even "
                        "count the majority vote can tie, and the extra voter is "
                        "never decisive -- N=10 measures the same vote quality as "
                        "N=9 while leaving ~15%% of coordinates tied. See "
                        "federated/algorithms.py")
    p.add_argument("--n_steps", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--partition", type=str, default="homo",
                   choices=["homo", "noniid-labeldir"])
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--lr-aux", type=float, default=1e-3)
    p.add_argument("--lr-scaling", type=str, default="unit-gain",
                   help=", ".join(sorted(RULES)) + ", or power:ALPHA[,BETA]")
    p.add_argument("--last-k", type=int, default=5,
                   help="Tail length, in EVALUATIONS not rounds")
    p.add_argument("--eval-points", type=int, default=20,
                   help="Evaluations per run; --last-k of them form the tail mean")
    p.add_argument("--target-acc", type=float, default=80.0)
    p.add_argument("--val-seed", type=int, default=12345)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--data", type=str, default="./data_federated")
    p.add_argument("--download", action="store_true")
    p.add_argument("--loader", type=str, default="auto", choices=["auto", "gpu", "torch"])
    p.add_argument("--mv-ties", type=str, default="random", choices=["random", "zero"],
                   help="What the server does with a tied majority vote; unreachable "
                        "at an odd client count under the default uplink")
    p.add_argument("--uplink-zeros", type=str, default="random",
                   choices=["random", "positive", "keep"],
                   help="What a client does with sign(0) on the majority-vote "
                        "uplink; 'random' (default) is the paper's convention and "
                        "makes the channel a genuine 1 bit, 'keep' is the legacy "
                        "ternary behaviour, retained for the alphabet diagnostic")
    p.add_argument("--nondeterministic", action="store_true")
    return p


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True,
                   choices=["lr", "aux", "anchors", "votes"],
                   help="lr/aux: the search stages. anchors: the per-layer "
                        "multipliers and transported grid anchors. votes: the "
                        "majority-vote alignment table behind --n_parties 11.")
    p.add_argument("--methods", nargs="*", default=None)
    p.add_argument("--aux-anchors", nargs="*", default=None)
    p.add_argument("--rounds", type=int, default=None,
                   help=f"Horizon for this stage. Defaults per stage: "
                        f"{STAGE_ROUNDS['lr']} for `lr`, the horizon the table "
                        f"reports, and {STAGE_ROUNDS['aux']} for `aux`, whose "
                        f"comparison is between two arms at a shared horizon")
    p.add_argument("--lr-points", type=int, default=5,
                   help="Lattice points per method: 3 per decade, so 5 spans ~1.3 decades")
    p.add_argument("--lr-extend-rounds", type=int, default=3)
    p.add_argument("--lr-extend-points", type=int, default=2)
    p.add_argument("--out", type=str, default=None)
    args = add_common_args(p).parse_args()
    if args.rounds is None:
        args.rounds = STAGE_ROUNDS.get(args.stage, STAGE_ROUNDS["lr"])
    return args


def vote_alignment(n_values: Sequence[int] = (9, 10, 11, 15),
                   q_values: Sequence[float] = (0.0, 0.10),
                   p_correct: float = 0.65, coords: int = 400_000,
                   seed: int = 0) -> str:
    """The majority-vote alignment table that justifies ``--n_parties 11``.

    ``A = E[<truth, s_agg>] / (mn)`` -- the fraction of a full-strength descent
    step the aggregate actually delivers -- with each client correct with
    probability ``p_correct`` and silent (``sign(0) = 0``) with probability ``q``.

    This table is quoted in three places and, until now, was produced by nothing:
    it was prose. Here it is, as a command.
    """
    import torch

    g = torch.Generator().manual_seed(seed)
    lines = [f"Majority-vote alignment, {coords} coordinates, "
             f"P(client correct) = {p_correct}, truth = +1",
             f"  {'N':>4}" + "".join(f"{f'tie% (q={q})':>16}{f'A (q={q})':>14}"
                                     for q in q_values)]
    for n in n_values:
        row = f"  {n:>4}"
        for q in q_values:
            # Each client sends +1 (correct), -1 (wrong) or 0 (silent).
            u = torch.rand(coords, n, generator=g)
            s = torch.where(u < q, torch.zeros(()),
                            torch.where(u < q + (1 - q) * p_correct,
                                        torch.ones(()), -torch.ones(())))
            agg = torch.sign(s.sum(dim=1))
            ties = float((agg == 0).float().mean())
            row += f"{100 * ties:>15.2f}%{float(agg.mean()):>14.4f}"
        lines.append(row)
    lines += ["",
              "  A is monotone in N once q > 0; at q = 0 the parity pathology is",
              "  visible (N=10 buys nothing over N=9). Either way 11 beats 10 --",
              "  because of more voters, not parity. An odd N does NOT remove ties:",
              "  a silent client makes a tie possible at any count."]
    return "\n".join(lines)


def main() -> None:
    args = get_args()
    if args.stage == "anchors":
        print(describe_anchors(args.lr_scaling, args.dataset, args.model,
                               args.methods or ALL_METHODS))
        return
    if args.stage == "votes":
        print(vote_alignment())
        return

    out = {"lr": stage_lr, "aux": stage_aux}[args.stage](args)

    path = Path(args.out) if args.out else (
        results_root() / "federated_tuning"
        / f"{args.stage}_{args.lr_scaling.replace(':', '')}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"stage": args.stage, "lr_scaling": args.lr_scaling,
                   "rounds": args.rounds, "n_parties": args.n_parties,
                   "selection_metric": "val_acc (last-k mean)", "results": out},
                  f, indent=2)
    print(f"\nWritten to {path}")
    print("Selection used validation accuracy only; the test set was never loaded.")


if __name__ == "__main__":
    main()
