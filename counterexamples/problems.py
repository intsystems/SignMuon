"""
The divergence counterexamples from the paper.

Linear instances (constant gradient ``grad f(W) = G``, objective ``f = Tr(G^T W)``):

* ``signmuon_counterexample`` -- the 4x4 instance of Theorem 1 on which
  **SignMuon** (sign after the LMO) diverges.  Here ``G = 1000 * u1 v1^T + O``
  with ``O`` orthogonal, so ``LMO(G) = O`` and ``<G, sign(O)> < 0``.

* ``muonsign_counterexample`` -- the single 5x5 instance of Theorems 2 and 3,
  on which **MuonUSign** (sign before the LMO) and **MuonSign** (sign on both
  sides) both diverge.  Here
  ``sign(G) = S`` for a fixed sign matrix ``S`` whose polar factor
  ``D = LMO(S)`` satisfies ``<G, D> < 0``.

Varying-gradient instance (the universal EF21-SignMuon counterexample):

* ``ef21_signmuon_counterexample`` -- the exact 2x2 construction of the appendix
  theorem "Divergence of EF21-SignMuon" (Theorem~\ref{th:ef_div}).  This is the
  SAME function the theorem builds, for a given ``(mu, nesterov)``; the code here
  and the LaTeX proof describe one object.  EF21-SignMuon cannot be broken on a
  linear objective (its estimator converges to ``polar(G)`` and descends), so a
  *varying* gradient is required, and a plain quadratic valley diverges only at
  ``mu = 0``.  The universal instance instead *forces* a fixed sequence of LMO
  targets (a rotation, a rank-one map, then alternating reflections) regardless
  of ``(mu, variant)``: the shared scaled-sign magnitude locks the estimator into
  a period-two cycle whose diagonal averages to the WRONG sign, so ``f -> +inf``
  at the exact rate ``49/480`` per step for every ``L``, ``eta``,
  ``mu in [0, 1)`` and both momentum variants.  The function is
  ``f(W) = gamma*h(W11) + A*(Phi1(W01) + Phi2(W10)) + sum_k b_k(W)``: a linear
  divergence slope, two periodic ramps that keep the off-diagonal gradient
  alternating, and three compactly supported corrections ``b_k`` that seed the
  first three gradients (all as in the proof).  Indices are 0-based here, so
  ``W11`` is the paper's ``W_22``.  ``h`` is the slope ``-v`` levelled off above
  ``v = 1``, the bounded-below variant the theorem is stated with (Remark
  "Boundedness below"); it equals ``-v`` on the whole region the iterates visit,
  ``{W11 <= 7/10}``, so no trajectory sees the modification.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# Linear objective  f(W) = <G, W> = Tr(G^T W),  grad f(W) = G
# --------------------------------------------------------------------------


def make_linear_problem(G: np.ndarray):
    """Return ``(grad_fn, loss_fn)`` for ``f(W) = Tr(G^T W)``.

    The gradient is globally constant, so ``grad_fn`` ignores its argument; it
    still takes ``W`` so the interface matches a general (e.g. quadratic)
    objective the user may plug in later.
    """
    G = np.asarray(G, dtype=float)

    def grad_fn(W):
        return G

    def loss_fn(W):
        return float(np.sum(G * W))

    return grad_fn, loss_fn


# --------------------------------------------------------------------------
# Theorem 1 -- SignMuon counterexample (4x4)
# --------------------------------------------------------------------------


def signmuon_counterexample(sigma1: float = 1000.0):
    """Build the 4x4 gradient ``G`` on which SignMuon diverges.

    ``G = sigma1 * u1 v1^T + O`` with ``O`` orthogonal and ``u1 = O v1``, so that
    ``polar(G) = O`` for *every* ``sigma1 > 0`` while the sign pattern of ``O`` is
    anticorrelated with the dominant component. The exact descent inner product is
    the rational function

        <G, sign(polar(G))> = (-43*sigma1 + 532) / 103,

    negative for every ``sigma1 > 532/43 = 12.37``. At the paper's ``sigma1 = 1000``
    this is ``-42468/103``.

    Choosing ``sigma1``
    -------------------
    Any ``sigma1 > 12.37`` refutes descent for the **exact** oracle. Under the
    *implemented* 5-step Newton-Schulz oracle the choice matters: ``sigma1 = 1000``
    makes ``G`` so ill-conditioned (``cond = 1001``) that 5 steps cannot resolve the
    ``O`` component at all, and the practical method *descends* on this instance.
    ``sigma1 = 100`` ascends under the exact oracle **and** under Newton-Schulz for
    every step count in ``{5, 6, 8, 10, 20}`` and both float32 and bfloat16, with
    exact value ``-3768/103``. Run ``verify_ns_oracle.py`` to reproduce the table.

    Returns
    -------
    G : (4, 4) ndarray
    info : dict
        Diagnostic pieces (``O``, ``u1``, ``v1``, ``sigma1``).
    """
    O = (1.0 / 103.0) * np.array([
        [101,  20,   2,  -2],
        [-20,  97,  20, -20],
        [-2,   20,   2, 101],
        [-2,   20, -101,  -2],
    ], dtype=float)
    u1 = (1.0 / np.sqrt(309.0)) * np.array([10, -3, 10, 10], dtype=float).reshape(-1, 1)
    v1 = (1.0 / np.sqrt(309.0)) * np.array([10,  3, -10, 10], dtype=float).reshape(-1, 1)
    G = float(sigma1) * (u1 @ v1.T) + O
    return G, {"O": O, "u1": u1, "v1": v1, "sigma1": float(sigma1)}


# --------------------------------------------------------------------------
# Theorem 2 -- MuonSign / MuonUSign counterexample (5x5)
# --------------------------------------------------------------------------


def muonsign_counterexample(eps: float = 1.0, M: float = 100.0):
    """Build the 5x5 gradient ``G`` on which MuonSign / MuonUSign diverges.

    ``G = eps * S + (M - eps) * e4 e2^T`` (1-indexed), so ``sign(G) = S`` for
    every ``M > 0`` while the single entry ``G[4,2] = M`` is inflated to expose
    the lone sign mismatch between ``S`` and ``polar(S)``.

    Choosing ``M``
    --------------
    The descent inner product is ``eps*<S,D> + (M-eps)*D[4,2]`` with
    ``D`` the oracle output, so ascent needs ``M > eps*(1 + <S,D>/(-D[4,2]))``.
    The mismatch ``D[4,2] < 0`` holds for the exact oracle *and* for Newton-Schulz
    at every step count tested, but its magnitude is smaller under Newton-Schulz
    (``-0.05`` at 5 steps vs ``-0.2425`` exact), which raises the threshold:

        exact oracle          M > 42.7
        Newton-Schulz, 5      M > 216.5   <-- the paper's M = 100 is BELOW this
        Newton-Schulz, >= 6   M > 19.8

    ``M = 100`` therefore refutes descent for MuonUSign only under the exact
    oracle; ``M = 500`` refutes it under every oracle tested (exact value
    ``-110.90``). MuonSign, which signs the oracle output, ascends at ``M = 100``
    for every oracle already (value ``-76``; ``-476`` at ``M = 500``).

    Returns
    -------
    G : (5, 5) ndarray
    info : dict
        Diagnostic pieces (``S``, ``eps``, ``M``, mismatch index).
    """
    S = np.array([
        [-1, -1,  1,  1,  1],
        [-1, -1,  1, -1, -1],
        [1,  -1,  1,  1, -1],
        [1,   1, -1, -1,  1],
        [1,   1,  1, -1,  1],
    ], dtype=float)
    G = eps * S.copy()
    G[3, 1] = eps * S[3, 1] + (M - eps)   # e4 e2^T  ->  0-indexed (3, 1)
    return G, {"S": S, "eps": eps, "M": M, "mismatch": (3, 1)}


# --------------------------------------------------------------------------
# Appendix theorem -- universal EF21-SignMuon counterexample (Theorem th:ef_div)
#
# The exact construction of the proof, in normalized units (eta = 1):
#   f(W) = gamma*h(W11) + A*(Phi1(W01) + Phi2(W10)) + sum_{k=1}^3 b_k(W).
# h is the floored linear slope (-gamma slope, bounded below); Phi_i are
# antiderivatives of periodic +-1 ramps psi_i; b_k are compactly supported
# corrections pinning the first three gradients.  A and the b_k depend on
# (mu, variant); everything else is fixed.  With this f the LMO targets are
# forced to  D1 = S1 (rotation), D2 = S2 (rank one), D_t = Dbar^{(-1)^t}
# (alternating reflections) for t >= 3, independent of (mu, variant).
# --------------------------------------------------------------------------

_GAMMA = 7.0 / 12.0                      # divergence slope gamma
_S1 = np.array([[-4/5,  3/5], [-3/5, -4/5]])   # rotation  (preamble target D1)
_S2 = np.array([[ 3/5, -4/5], [  0.0, 0.0]])   # rank-one  (preamble target D2)
_MBAR_M = np.array([[0.0, -1.0], [-1.0, -_GAMMA]])   # momentum preimage of Dbar^-
_Z2 = np.array([[7/10, -7/10], [7/10, 7/10]])  # second iterate  Xtil_1
_Z3 = np.array([[7/20, -7/20], [7/20, 7/20]])  # third iterate   Xtil_2
_TAU = 1.0 / 200.0                       # ramp half-width of psi_i (subset of the
                                         # theorem's band half-width delta = 1/100)
_RBUMP = 1.0 / 50.0                      # correction radius r


class _PeriodicRamp:
    r"""A fixed ``p``-periodic ``psi`` with ``\int_0^p psi = 0``.

    ``psi`` is ``+1`` on the first half-period plateau and ``-1`` on the second,
    joined by sine ramps of half-width ``tau`` centered at ``a_up`` (the
    ``-1 -> +1`` crossing) and ``a_up + p/2``.  ``Phi`` is its periodic
    antiderivative.  ``a_up`` is chosen so the required residues land inside the
    plateaus, i.e. ``psi = +-1`` there exactly, with room to spare: each plateau
    contains the theorem's whole band ``[rho +- delta]``, ``delta = 1/100``.

    The sine joint makes this ``psi`` only ``C^1``, where the proof asks for a
    ``C^\infty`` one.  Nothing measured depends on the difference: the trajectory
    samples ``psi`` only on the plateaus, where the two agree exactly at ``+-1``,
    and the joint is smoothed purely so ``grad f`` is continuous.  The same holds
    for the ``C^1`` cutoff :func:`_chi` below, of which the proof uses only
    ``phi(0) = 1`` and ``supp phi subset [0, 1)``.
    """

    def __init__(self, p, a_up):
        self.p, self.a, self.tau = p, a_up % p, _TAU

    def psi(self, w):
        p, a, tau = self.p, self.a, self.tau
        r = (w - a) % p
        if r < tau:
            return np.sin(np.pi * r / (2 * tau))
        if r < p / 2 - tau:
            return 1.0
        if r < p / 2 + tau:
            return -np.sin(np.pi * (r - p / 2) / (2 * tau))
        if r < p - tau:
            return -1.0
        return np.sin(np.pi * (r - p) / (2 * tau))

    def Phi(self, w):
        p, tau = self.p, self.tau
        r = (w - self.a) % p
        out = 0.0
        t = min(r, tau)
        out += (2 * tau / np.pi) * (1 - np.cos(np.pi * t / (2 * tau)))
        if r <= tau:
            return out
        t = min(r, p / 2 - tau)
        out += t - tau
        if r <= p / 2 - tau:
            return out
        t = min(r, p / 2 + tau)
        out += (2 * tau / np.pi) * np.cos(np.pi * (t - p / 2) / (2 * tau))
        if r <= p / 2 + tau:
            return out
        t = min(r, p - tau)
        out += -(t - p / 2 - tau)
        if r <= p - tau:
            return out
        out += -(2 * tau / np.pi) * np.cos(np.pi * (p - r) / (2 * tau))
        return out


# W01: period p1 = 21/20, residues 131/200 (psi=+1) and 140/200 (psi=-1).
# W10: period p2 = 7/20,  residues 61/200 (psi=+1) and 0 (psi=-1).
_PS1 = _PeriodicRamp(21 / 20, a_up=30.5 / 200)
_PS2 = _PeriodicRamp(7 / 20, a_up=30.5 / 200)


def _floor_h(v):
    """Floored linear slope ``h`` and its derivative: ``h(v) = -v`` for ``v<=1``,
    smoothly constant for ``v>=2`` (so ``gamma*h`` is bounded below yet equals the
    ``-gamma`` slope on the whole visited region ``{W11 <= 7/10}``)."""
    if v <= 1.0:
        return -v, -1.0
    if v <= 2.0:
        return (-1.0 - (2 / np.pi) * np.sin(np.pi * (v - 1) / 2),
                -np.cos(np.pi * (v - 1) / 2))
    return -1.0 - 2 / np.pi, 0.0


def _chi(s):      # C^inf cutoff: 1 near 0, 0 beyond s = 1
    return np.cos(np.pi * s / 2) ** 2 if s < 1.0 else 0.0


def _chi_prime(s):
    return -(np.pi / 2) * np.sin(np.pi * s) if s < 1.0 else 0.0


def ef21_signmuon_counterexample(mu=0.0, nesterov=False):
    r"""Universal 2x2 instance on which **EF21-SignMuon** diverges (Theorem 4).

    Builds the exact function of the appendix proof for the given momentum
    coefficient ``mu`` and variant (``nesterov``).  The returned ``grad_fn`` /
    ``loss_fn`` are the honest gradient and value of a single smooth
    ``f: R^{2x2} -> R``; running EF21-SignMuon (this ``mu``, this variant, from
    ``0``) makes ``f -> +inf`` at the exact rate ``49/480`` per step, while Muon,
    EF21-MuonUSign and SignSGD descend.

    The field ``-gamma W11 + A(Phi1(W01)+Phi2(W10))`` supplies the divergence
    slope and the alternating off-diagonal gradient; three corrections seed the
    first three gradients so momentum (either variant) reproduces the forced
    target sequence.  ``A`` and the corrections depend on ``(mu, variant)``:
    momentum is a positive linear filter of the gradients and we invert it.

    Returns ``(grad_fn, loss_fn, shape, info)``.
    """
    if not (0.0 <= mu < 1.0):
        raise ValueError("mu must lie in [0, 1)")

    # --- invert the momentum filter for the transient gradients G1,G2,G3 -----
    if not nesterov:
        A = (1 + mu) / (1 - mu)
        M1, M2, M3 = _S1, _S2, _MBAR_M           # effective directions = buffer
    else:
        nu = 1 / (1 + 2 * mu)
        A = nu * (1 + mu) / (1 - mu)
        if mu == 0.0:
            M1, M2, M3 = _S1, _S2, _MBAR_M
        else:
            tau = (140 * (1 + mu) ** 2 / (1 + 2 * mu) - 44) * (1 + mu) / (117 * mu)
            H1 = np.diag([1.0, 1.0 + tau])
            M1 = _S1 @ H1 / (1 + mu)
            M2 = (_S2 + mu * M1) / (1 + mu)
            M3 = np.array([[0.0, -nu], [-nu, -_GAMMA]])
    G1 = M1 / (1 - mu)
    G2 = (M2 - mu * M1) / (1 - mu)
    G3 = (M3 - mu * M2) / (1 - mu)

    def field_grad(W):
        g = np.zeros((2, 2))
        g[0, 1] = A * _PS1.psi(W[0, 1])
        g[1, 0] = A * _PS2.psi(W[1, 0])
        g[1, 1] = _GAMMA * _floor_h(W[1, 1])[1]
        return g

    def field_val(W):
        return (_GAMMA * _floor_h(W[1, 1])[0]
                + A * (_PS1.Phi(W[0, 1]) + _PS2.Phi(W[1, 0])))

    # corrections b_k = <C_k, W - Z_k> * chi(||W - Z_k|| / r); grad at Z_k = C_k
    centers = (np.zeros((2, 2)), _Z2, _Z3)
    Cs = [Gk - field_grad(Zk) for Zk, Gk in zip(centers, (G1, G2, G3))]

    def grad_fn(W):
        g = field_grad(W)
        for Zk, Ck in zip(centers, Cs):
            D = W - Zk
            n = float(np.linalg.norm(D))
            if n < _RBUMP:
                s = n / _RBUMP
                g = g + _chi(s) * Ck
                if n > 1e-14:
                    g = g + np.vdot(Ck, D) * _chi_prime(s) / _RBUMP * D / n
        return g

    def loss_fn(W):
        v = field_val(W)
        for Zk, Ck in zip(centers, Cs):
            D = W - Zk
            n = float(np.linalg.norm(D))
            if n < _RBUMP:
                v += np.vdot(Ck, D) * _chi(n / _RBUMP)
        return float(v)

    info = {"mu": mu, "nesterov": nesterov, "A": A, "gamma": _GAMMA,
            "rate": 49.0 / 480.0}
    return grad_fn, loss_fn, (2, 2), info


# --------------------------------------------------------------------------
# Self-check: reproduce the exact descent inner products from the proofs
# --------------------------------------------------------------------------


def ef21_alternating_cycle(a=7 / 25, b=24 / 25, steps=40):
    """The *harmless* limit cycle, and why the counterexample needs a preamble.

    Runs the EF21 estimator recursion ``d <- d + mean|D-d| * sign(D-d)`` from
    ``d = 0`` on the purely alternating targets ``Dbar^-, Dbar^+, ...`` with
    ``Dbar^+- = [[a, +-b], [+-b, -a]]`` -- i.e. the tail of the counterexample
    with its two-step preamble removed.  Plain ``np.sign`` suffices here rather
    than the randomized convention of ``optimizers.sign_pm1``: every residual
    along either cycle is entrywise nonzero, as both lemmas record, so the two
    agree and the arithmetic is exact.

    It enters a period-two cycle at once whose (2,2) average is ``-a/2``: the
    SAME sign as every target value ``-a``, so the iterate does not run away.
    The divergent cycle of Lemma "Wrong-sign limit cycle" is a *different*
    period-two orbit of the same recursion, sitting in a different sign-pattern
    cell, and the preamble exists only to land the estimator in that cell.

    Returns ``(d_odd, d_even, mean22)``.
    """
    Dp = np.array([[a, b], [b, -a]])
    Dm = np.array([[a, -b], [-b, -a]])
    d = np.zeros((2, 2))
    hist = []
    for t in range(1, steps + 1):
        D = Dm if t % 2 else Dp
        R = D - d
        d = d + np.abs(R).mean() * np.sign(R)
        hist.append(d.copy())
    d_odd, d_even = hist[-2], hist[-1]        # steps -2 and -1 have opposite parity
    if steps % 2 == 1:                        # last index is odd -> swap
        d_odd, d_even = d_even, d_odd
    return d_odd, d_even, 0.5 * (d_odd[1, 1] + d_even[1, 1])


def _self_check():
    from counterexamples.optimizers import muon_lmo, sign_pm1

    print("== the sign convention ==")
    rng = np.random.default_rng(0)
    probe = np.array([[-2.0, 0.0], [0.0, 3.0]])
    print(f"  sign_pm1 of a matrix with two exact zeros -> "
          f"{sign_pm1(probe, rng).ravel().tolist()} (never 0)")
    for label, M in (("Theorem 1: O", signmuon_counterexample()[1]["O"]),
                     ("Theorem 2: S", muonsign_counterexample()[1]["S"]),
                     ("Theorem 2: polar(S)",
                      muon_lmo(muonsign_counterexample()[1]["S"]))):
        assert np.all(M != 0.0), label
        print(f"  {label:<20} has no zero entry -> its constants are "
              f"convention-free")

    print("== Theorem 1 (SignMuon, 4x4) ==")
    G1, info1 = signmuon_counterexample()
    D1 = muon_lmo(G1)                       # LMO(G) should equal O
    S1 = np.sign(D1)                        # sign of the Muon direction
    print(f"  ||LMO(G) - O||_F         = {np.linalg.norm(D1 - info1['O']):.2e}")
    print(f"  <G, sign(LMO(G))>        = {np.sum(G1 * S1):+.3f}   (paper: -412.311)")
    print(f"  <G, LMO(G)>  (Muon)      = {np.sum(G1 * D1):+.3f}   (descends, > 0)")

    print("== Theorem 2 (MuonSign / MuonUSign, 5x5) ==")
    G2, info2 = muonsign_counterexample(eps=1.0, M=100.0)
    S2 = info2["S"]
    print(f"  sign(G) == S             = {np.array_equal(np.sign(G2), S2)}")
    D2 = muon_lmo(S2)                       # polar factor of the sign matrix
    print(f"  D[4,2] (mismatch entry)  = {D2[3, 1]:+.4f}   (paper: -0.2425)")
    print(f"  <G, LMO(sign(G))>        = {np.sum(G2 * D2):+.3f}   (paper: -13.89)")
    print(f"  <G, LMO(G)>  (Muon)      = {np.sum(G2 * muon_lmo(G2)):+.3f}   (descends, > 0)")

    print("== Appendix (why the counterexample needs its preamble) ==")
    a = 7 / 25
    d_odd, d_even, m22 = ef21_alternating_cycle(a=a)
    print(f"  alternating targets alone -> 2-cycle with (2,2) mean = {m22:+.6f}")
    print(f"  predicted -a/2 = {-a/2:+.6f}; target value is {-a:+.4f}  "
          f"(same sign -> no divergence)")

    print("== Appendix (EF21-SignMuon, universal 2x2 construction) ==")
    from counterexamples.optimizers import EF21SignMuon
    settle, periods = 400, 500       # measure over whole 2-step periods
    print(f"  exact rate 49/480 = {49/480:.8f}; per-step rate over {periods} "
          f"periods after settling:")
    print(f"  {'variant':<10}" + "".join(f"mu={m:<8}" for m in
                                          (0.0, 0.25, 0.5, 0.9, 0.99)))
    for nesterov in (False, True):
        rates = []
        for mu in (0.0, 0.25, 0.5, 0.9, 0.99):
            grad_fn, loss_fn, shape, _ = ef21_signmuon_counterexample(
                mu=mu, nesterov=nesterov)
            opt = EF21SignMuon(shape, eta=1.0, mu=mu, nesterov=nesterov)
            for _ in range(settle):
                opt.step(grad_fn(opt.grad_point()))
            f0 = loss_fn(opt.track_point())
            for _ in range(2 * periods):
                opt.step(grad_fn(opt.grad_point()))
            rates.append((loss_fn(opt.track_point()) - f0) / (2 * periods))
        print(f"  {'Nesterov' if nesterov else 'standard':<10}"
              + "".join(f"{r:.6f}  " for r in rates))
    print("  every mu and both variants ascend at exactly 49/480; the iterate "
          "trajectory is identical across them.")


if __name__ == "__main__":
    _self_check()
