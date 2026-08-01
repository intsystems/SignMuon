"""How small can the MuonUSign / MuonSign counterexample be?

For MuonUSign the linear-objective increment factors through the sign pattern:
with ``S = sign(G)`` and ``D = polar(S)`` (rank-truncated SVD),

    <G, polar(sign(G))> = sum_ij |G_ij| * S_ij * D_ij,

because ``G`` vanishes off the support of ``S``.  For MuonSign the same holds
with ``sign(D_ij)`` in place of ``D_ij``.  Hence an ascent instance exists at a
given shape *iff* some sign pattern there has a mismatched entry
``S_ij * D_ij < 0`` (load that entry of ``G``); if no pattern has one, neither
method can ascend on any linear objective at that shape.

Mismatch existence is invariant under row/column sign flips, since
``polar(P1 S P2) = P1 polar(S) P2`` for +-1 diagonal ``P1, P2``.  Every +-1
pattern can be flipped so that its first row and column are all +1, which cuts
the enumeration to ``2^((m-1)(n-1))`` canonical representatives.  Ternary
patterns ({-1,0,+1}, i.e. gradients with exact zeros) are enumerated in full.

Run from ``code/``, the package root:

    python3 -m counterexamples.enumerate_minimality             # the quick tier
    python3 -m counterexamples.enumerate_minimality --full      # + 3^16 ternary 4x4 (~10 min)

Findings (quoted in the paper's Appendix, proof of Theorem 2):

  +-1 patterns (dense G):  NO mismatch at 4x4, 3xn (n<=9), 2xn (n<=12).
                           First mismatches: 4x5 (720 of 2^12 canonical
                           classes) and, among squares, 5x5 -- where the
                           deepest achievable mismatch is 1/sqrt(17), attained
                           by the published instance of Theorems 2-3.
  ternary patterns:        NO mismatch at 3x3 (3^9) and 2xn (n<=5); mismatches
                           exist from 3x4 on, and at 4x4 (2 777 088 of 3^16,
                           deepest 1/sqrt(13), most of them full-rank).  A
                           zero-entry witness is knife-edge (perturbing a zero
                           flips sign(G) discontinuously), which is why the
                           paper keeps the dense 5x5 instance.
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np

TOL = 1e-9


def polar_batch(S: np.ndarray) -> np.ndarray:
    """Rank-truncated polar factor of a stack of matrices (the paper's LMO
    selection): directions with singular value <= 1e-8 * sigma_max are dropped."""
    U, s, Vt = np.linalg.svd(S, full_matrices=False)
    keep = s > (1e-8 * s[:, :1])
    return (U * keep[:, None, :]) @ Vt


def scan_pm1(m: int, n: int) -> tuple[int, float]:
    """All +-1 patterns with first row/column fixed to +1 (canonical under
    sign flips). Returns (#mismatched classes, deepest S_ij*D_ij)."""
    free = (m - 1) * (n - 1)
    mismatched, deepest = 0, 0.0
    for bits in itertools.product([-1.0, 1.0], repeat=free):
        S = np.ones((m, n))
        S[1:, 1:] = np.asarray(bits).reshape(m - 1, n - 1)
        D = polar_batch(S[None])[0]
        v = float((D * S).min())
        if v < -TOL:
            mismatched += 1
            deepest = min(deepest, v)
    return mismatched, deepest


def scan_ternary(m: int, n: int, chunk: int = 1 << 18) -> tuple[int, float]:
    """All 3^(mn) ternary patterns, batched. Returns (#mismatched, deepest)."""
    total = 3 ** (m * n)
    mismatched, deepest = 0, 0.0
    for start in range(0, total, chunk):
        block = np.arange(start, min(start + chunk, total), dtype=np.int64)
        digits = np.empty((block.size, m * n), dtype=np.int8)
        b = block.copy()
        for k in range(m * n):
            digits[:, k] = (b % 3).astype(np.int8) - 1
            b //= 3
        S = digits.reshape(-1, m, n).astype(np.float64)
        S = S[np.abs(S).sum(axis=(1, 2)) > 0]  # drop the all-zero pattern
        prod = (polar_batch(S) * S).min(axis=(1, 2))
        bad = prod < -TOL
        mismatched += int(bad.sum())
        deepest = min(deepest, float(prod.min()))
    return mismatched, deepest


def published_5x5_is_extremal() -> None:
    """The paper's S attains the deepest mismatch any 5x5 sign matrix admits."""
    S = np.array([[-1, -1, 1, 1, 1],
                  [-1, -1, 1, -1, -1],
                  [1, -1, 1, 1, -1],
                  [1, 1, -1, -1, 1],
                  [1, 1, 1, -1, 1]], dtype=float)
    D = polar_batch(S[None])[0]
    d42 = D[3, 1]
    print(f"published 5x5 witness: D[4,2] = {d42:+.10f} = -1/sqrt(17) "
          f"(check: {d42 + 1/np.sqrt(17):+.1e}); single mismatch, full rank")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--full", action="store_true",
                    help="also enumerate the 3^16 ternary patterns at 4x4 (~10 min)")
    args = ap.parse_args()

    print("+-1 patterns, canonical under sign flips (mismatched classes / deepest):")
    shapes = [(2, n) for n in range(2, 13)] + [(3, n) for n in range(3, 10)] + \
             [(4, 4), (4, 5), (4, 6), (5, 5)]
    for m, n in shapes:
        k, d = scan_pm1(m, n)
        print(f"  {m}x{n}: {k} / {d:.6f}")

    print("ternary patterns, full enumeration:")
    ternary = [(2, 2), (2, 3), (2, 4), (2, 5), (3, 3), (3, 4), (3, 5)]
    if args.full:
        ternary.append((4, 4))
    for m, n in ternary:
        k, d = scan_ternary(m, n)
        print(f"  {m}x{n}: {k} / {d:.6f}")

    published_5x5_is_extremal()


if __name__ == "__main__":
    main()
