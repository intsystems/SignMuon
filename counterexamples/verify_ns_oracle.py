"""Do the counterexamples survive the *implemented* Newton-Schulz LMO?

Theorems 1-3 are stated for the exact Muon LMO ``polar(Y) = U V^T``. The code that
trains networks (and the reference Muon implementation) approximates it with a
fixed number of iterations of the quintic of Algorithm 1, in bfloat16. Those are
different maps, and the theorems' quantity of interest is a *sign pattern*, so the
substitution is not automatically safe: an entry of ``polar(Y)`` that is small in
magnitude can flip.

This script evaluates each theorem's descent inner product under

  * the exact oracle (rank-truncated SVD), and
  * the Newton-Schulz oracle for several step counts, in float32 and bfloat16,

and reports, for each instance's free parameter (``sigma1`` for Theorem 1, ``M``
for Theorems 2-3), the values for which *every* oracle ascends.

Run from ``code/``, the package root:

    python3 -m counterexamples.verify_ns_oracle
    python3 -m counterexamples.verify_ns_oracle --steps 5 6 8 10 20 --trajectories

Summary of what it finds (see the module docstrings in ``problems.py``):

  Theorem 1   sigma1 = 1000 (as published) FAILS at 5 steps -- cond(G) = 1001 is
              too ill-conditioned for 5 iterations to resolve the O component.
              sigma1 = 100 ascends under every oracle; exact value -3768/103.
  Theorem 2   M = 100 (as published) FAILS at 5 steps; the mismatch entry is
              present but only -0.05 deep, so M must exceed ~217. M = 500 works
              under every oracle; exact value -110.90.
  Theorem 3   M = 100 already ascends under every oracle (value -76): signing the
              oracle output discards the magnitude error that breaks Theorem 2.
"""

from __future__ import annotations

import argparse

import numpy as np

from counterexamples.optimizers import make_lmo, muon_lmo
from counterexamples.problems import muonsign_counterexample, signmuon_counterexample

DTYPES = ("float32", "bfloat16")


def oracles(steps_list):
    """``{label: lmo_callable}``, exact first."""
    out = {"exact": muon_lmo}
    for k in steps_list:
        for dt in DTYPES:
            out[f"NS{k}-{dt[:2]}{dt[-2:]}"] = make_lmo("ns", steps=k, dtype=dt)
    return out


def _table(title, header, rows, note=""):
    print(f"\n{title}")
    print("  " + header)
    print("  " + "-" * (len(header) + 2))
    for row in rows:
        print("  " + row)
    if note:
        print("  " + note)


def theorem1(steps_list, sigmas):
    """<G, sign(oracle(G))> for G = sigma1 u1 v1^T + O."""
    ora = oracles(steps_list)
    header = f"{'sigma1':>8}" + "".join(f"{k:>14}" for k in ora) + "   verdict"
    rows, all_ok = [], []
    for s in sigmas:
        G, _ = signmuon_counterexample(sigma1=s)
        vals = [float(np.sum(G * np.sign(lmo(G)))) for lmo in ora.values()]
        ok = all(v < 0 for v in vals)
        all_ok.append((s, ok))
        rows.append(f"{s:>8g}" + "".join(f"{v:>+14.3f}" for v in vals)
                    + ("   ALL ascend" if ok else "   mixed"))
    good = [s for s, ok in all_ok if ok]
    _table("Theorem 1 -- SignMuon, <G, sign(oracle(G))>  (negative = ascends)",
           header, rows,
           note=f"sigma1 values where every oracle ascends: {good}"
                f"\n  exact closed form: (-43*sigma1 + 532)/103, negative for sigma1 > 12.37")
    return good


def theorems23(steps_list, Ms):
    """<G, D> (MuonUSign) and <G, sign(D)> (MuonSign) with D = oracle(sign(G))."""
    ora = oracles(steps_list)
    _, info = muonsign_counterexample()
    S, (i, j) = info["S"], info["mismatch"]

    header = f"{'oracle':>14}{'<S,D>':>10}{'D[4,2]':>10}{'M threshold':>13}"
    rows = []
    thresholds = {}
    for label, lmo in ora.items():
        D = lmo(S)
        sd, d = float(np.sum(S * D)), float(D[i, j])
        thr = 1.0 + sd / (-d) if d < 0 else float("inf")
        thresholds[label] = thr
        rows.append(f"{label:>14}{sd:>10.4f}{d:>10.4f}"
                    + (f"{thr:>13.1f}" if np.isfinite(thr) else f"{'no mismatch':>13}"))
    worst = max(thresholds.values())
    _table("Theorems 2-3 -- the mismatch entry D[4,2] under each oracle",
           header, rows,
           note=f"MuonUSign needs M > {worst:.1f} for every oracle "
                f"(sign(G) = S is independent of M)")

    header2 = (f"{'M':>8}" + "".join(f"{k:>12}" for k in ora) + "   verdict")
    for name, transform in (("Theorem 2 -- MuonUSign, <G, oracle(sign G)>", lambda D: D),
                            ("Theorem 3 -- MuonSign, <G, sign(oracle(sign G))>", np.sign)):
        rows, good = [], []
        for M in Ms:
            G, _ = muonsign_counterexample(eps=1.0, M=M)
            assert np.array_equal(np.sign(G), S), "sign(G) must remain S"
            vals = [float(np.sum(G * transform(lmo(S)))) for lmo in ora.values()]
            ok = all(v < 0 for v in vals)
            if ok:
                good.append(M)
            rows.append(f"{M:>8g}" + "".join(f"{v:>+12.2f}" for v in vals)
                        + ("   ALL ascend" if ok else "   mixed"))
        _table(name + "  (negative = ascends)", header2, rows,
               note=f"M values where every oracle ascends: {good}")


def trajectories(steps, dtype, T, eta):
    """Run every method with the practical oracle and report the tail slope."""
    from counterexamples.optimizers import OPTIMIZERS, PAPER_METHODS, REFERENCE_METHODS
    from counterexamples.problems import make_linear_problem

    lmo = make_lmo("ns", steps=steps, dtype=dtype)
    for label, (G, _) in (("Theorem 1 instance (sigma1=100)", signmuon_counterexample(100.0)),
                          ("Theorems 2-3 instance (M=500)",
                           muonsign_counterexample(eps=1.0, M=500.0))):
        grad_fn, loss_fn = make_linear_problem(G)
        print(f"\n  {label}, LMO = Newton-Schulz({steps}, {dtype}), mu=0, T={T}")
        print(f"    {'method':<18}{'f[0]':>10}{'f[T-1]':>12}   verdict")
        for name in PAPER_METHODS + REFERENCE_METHODS:
            opt = OPTIMIZERS[name](G.shape, eta=eta, mu=0.0, nesterov=False, lmo=lmo)
            losses = []
            for _ in range(T):
                losses.append(loss_fn(opt.track_point()))
                opt.step(grad_fn(opt.grad_point()))
            rose = losses[-1] > losses[0] + 1e-12
            print(f"    {name:<18}{losses[0]:>10.3f}{losses[-1]:>12.3f}"
                  f"   {'DIVERGES (up)' if rose else 'descends'}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, nargs="*", default=[5, 6, 8, 10, 20],
                   help="Newton-Schulz step counts to test (default: 5 6 8 10 20)")
    p.add_argument("--sigmas", type=float, nargs="*",
                   default=[30, 50, 80, 100, 120, 150, 200, 1000],
                   help="sigma1 values for Theorem 1")
    p.add_argument("--Ms", type=float, nargs="*", default=[100, 250, 500, 1000],
                   help="M values for Theorems 2-3")
    p.add_argument("--trajectories", action="store_true",
                   help="Also run the full trajectories with the practical oracle")
    p.add_argument("--traj-steps", type=int, default=5)
    p.add_argument("--traj-dtype", type=str, default="bfloat16", choices=DTYPES)
    p.add_argument("--T", type=int, default=60)
    p.add_argument("--eta", type=float, default=1e-3)
    args = p.parse_args()

    print("Descent inner products under the exact vs the implemented (Newton-Schulz) LMO.")
    print("A method ascends on a linear objective exactly when its inner product is < 0.")
    theorem1(args.steps, args.sigmas)
    theorems23(args.steps, args.Ms)
    if args.trajectories:
        print("\n" + "=" * 78)
        trajectories(args.traj_steps, args.traj_dtype, args.T, args.eta)


if __name__ == "__main__":
    main()
