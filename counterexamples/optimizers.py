"""
Exact-SVD implementations of the eight optimizers studied in the paper
"SignMuon, MuonSign, and the Role of Error Feedback".

Every method follows the corresponding pseudocode box in the paper verbatim,
with three deliberate choices:

  * the Muon LMO defaults to an **exact SVD** (rank-truncated ``U V^T``), which is
    what the theorems are stated about. Pass ``lmo=make_lmo("ns", steps=5)`` to
    run the *implemented* Newton-Schulz oracle instead -- the two are NOT
    interchangeable on these instances, see :func:`newton_schulz_lmo` and
    ``verify_ns_oracle.py``;
  * momentum is the EMA / heavy-ball form of the centralized algorithm boxes
    ``M_t = mu*M_{t-1} + (1-mu)*G_t`` with an optional Nesterov look-ahead
    ``M_tilde = (1-mu)*G_t + mu*M_t``.  Both the coefficient (default
    ``mu = 0.0``) and the variant are parameters; on the linear instances neither
    can change a verdict (Proposition "reduction"), and on the EF21-SignMuon
    instance the trajectory is identical for every setting; and
  * ``sign(0)`` resolves to an independent random ``+-1``, the paper's convention,
    so every transmitted symbol is a strict bit.  Each optimizer carries its own
    generator, seeded by ``sign_seed``, so a run depends only on its own seed and
    not on what was run before it (as ``synthetic.batched`` does on the GPU side).

The exact zeros this last convention covers are real but confined: no entry of
the Theorem 1-3 instances or of their oracle outputs is zero, so every constant
those theorems quote is convention-free.  What the convention does touch is the
seven *bounded* methods on the Theorem 4 instance, whose field gradient has an
exactly-zero ``(1,1)`` entry by construction: they stay bounded either way, and
EF21-SignMuon's own trajectory is untouched, every residual along it being
strictly nonzero (Lemma "Wrong-sign limit cycle").

Interface (used by ``run_counterexamples.py``)
----------------------------------------------
Each optimizer owns its model state.  The driver loop is simply::

    opt = SignMuon(shape, eta, mu, nesterov)
    for t in range(T):
        losses.append(loss_fn(opt.track_point()))   # record, then step
        G = grad_fn(opt.grad_point())
        opt.step(G)

``grad_point()`` is where the (stochastic) gradient is evaluated and
``track_point()`` is the model whose objective value we plot.  For every method
except EF21-MuonSign these coincide with the single model ``X``; the
bidirectional method evaluates the gradient at the *compressed* broadcast model
``W`` while the tracked/exact model is ``X``.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------

# Newton-Schulz coefficients of the paper's Algorithm 1.
NS_COEFFS = (3.4445, 4.7750, 2.0315)


def bfloat16(x: np.ndarray) -> np.ndarray:
    """Round a float32 array to bfloat16 precision and back.

    NumPy has no bfloat16, so this truncates the mantissa to 8 bits with
    round-to-nearest. It is a *pessimistic* model of torch's bfloat16, which
    accumulates matmuls in float32.
    """
    a = np.asarray(x, dtype=np.float32).copy()
    u = a.view(np.uint32)
    u += 0x8000
    u &= 0xFFFF0000
    return u.view(np.float32).astype(np.float32)


def newton_schulz_lmo(Y: np.ndarray, steps: int = 5, dtype: str = "float32",
                      eps: float = 1e-7) -> np.ndarray:
    r"""The *practical* LMO: ``steps`` iterations of the quintic of Algorithm 1.

    This is what the torch code (and the reference Muon implementation) actually
    runs, as opposed to the exact polar factor :func:`muon_lmo` the theorems are
    stated about.

    The two are **not** interchangeable on the counterexamples. The Muon quintic
    is not a convergent iteration -- it is tuned to push singular values into a
    band around 1 and then oscillates inside it -- so ``newton_schulz_lmo`` differs
    from ``muon_lmo`` by an ``O(1)`` perturbation whose size and sign depend on
    ``steps``. Whenever the quantity of interest is the *sign pattern* of the
    output, an entry of the exact polar factor that is small in magnitude can
    therefore flip. Use :mod:`verify_ns_oracle` to check a given instance against
    both oracles before relying on it.
    """
    a, b, c = NS_COEFFS
    cast = bfloat16 if dtype == "bfloat16" else (lambda z: np.asarray(z, np.float32))
    X = cast(Y)
    transposed = X.shape[-2] > X.shape[-1]
    if transposed:
        X = X.T
    X = cast(X / (np.linalg.norm(X) + eps))
    for _ in range(steps):
        A = cast(X @ X.T)
        X = cast(a * X + cast(-b * A + c * cast(A @ A)) @ X)
    return (X.T if transposed else X).astype(np.float64)


def make_lmo(oracle: str = "exact", steps: int = 5, dtype: str = "float32"):
    """Return the LMO callable named by ``oracle`` (``"exact"`` or ``"ns"``)."""
    if oracle == "exact":
        return muon_lmo
    if oracle == "ns":
        return lambda Y: newton_schulz_lmo(Y, steps=steps, dtype=dtype)
    raise ValueError(f"unknown oracle {oracle!r} (expected 'exact' or 'ns')")


def muon_lmo(Y: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    r"""Muon LMO direction :math:`\mathbf D = \mathbf U \mathbf V^\top` via exact SVD.

    ``muon_lmo`` returns the (rank-truncated) orthogonal polar factor, i.e. the
    minimizer ``-A(Y)`` of the spectral-norm linear-minimization oracle.

    Only the singular directions with **nonzero** singular value are kept
    (``r = rank(Y)``).  The full ``U @ Vt`` would append an arbitrary orthonormal
    completion of the null space, which is *non-unique* on rank-deficient inputs
    (e.g. ``sign(G)`` is often low-rank) and would silently change the answer.
    ``muon_lmo`` is scale-invariant: ``muon_lmo(c*Y) == muon_lmo(Y)`` for ``c > 0``.
    """
    U, S, Vt = np.linalg.svd(Y, full_matrices=False)
    if S.size == 0:
        return np.zeros_like(Y)
    r = int(np.count_nonzero(S > tol * S[0]))
    if r == 0:
        return np.zeros_like(Y)
    return U[:, :r] @ Vt[:r, :]


def sign_pm1(Y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    r"""Elementwise sign with :math:`\operatorname{sign}(0)` drawn uniformly from
    :math:`\{\pm1\}`.

    The paper's convention, and the reason every sign channel here is a strict one
    bit rather than a ternary alphabet.  ``rng`` is the caller's generator, so the
    draws belong to one run and no run depends on the draws of another.
    """
    s = np.sign(Y)
    zeros = s == 0
    if zeros.any():
        s = np.where(zeros, rng.choice((-1.0, 1.0), size=s.shape), s)
    return s


def scaled_sign(Y: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    r"""Contractive 1-bit compressor :math:`\mathrm{mean}|\mathbf Y|\,\operatorname{sign}(\mathbf Y)`.

    The uplink compressor of the EF21 methods: one bit per entry (the sign)
    plus a single shared magnitude scalar per matrix.  ``rng`` supplies the
    ``sign(0)`` draws; the optimizers below pass their own.
    """
    sgn = np.sign(Y) if rng is None else sign_pm1(Y, rng)
    return np.mean(np.abs(Y)) * sgn


# --------------------------------------------------------------------------
# Base optimizer
# --------------------------------------------------------------------------


class Optimizer:
    """Common momentum bookkeeping shared by every method.

    Parameters
    ----------
    shape : tuple[int, int]
        Matrix parameter shape.
    eta : float
        Learning rate ``eta_t`` (constant here).
    mu : float
        Momentum coefficient ``mu``. Defaults to ``0.0``: on the linear instances
        the step is the same matrix for every ``mu`` in ``[0, 1)`` and either
        variant, and on the Theorem 4 instance so is the whole trajectory, so the
        momentum-free run is the run for all of them.
    nesterov : bool
        If ``True`` use the Nesterov look-ahead effective direction, otherwise
        the standard (default) momentum ``M_tilde = M_t``.
    sign_seed : int
        Seed of this optimizer's ``sign(0)`` generator (see :func:`sign_pm1`).
    """

    name = "Optimizer"

    def __init__(self, shape, eta: float, mu: float = 0.0, nesterov: bool = False,
                 lmo=None, sign_seed: int = 0):
        self.shape = tuple(shape)
        self.eta = float(eta)
        self.mu = float(mu)
        self.nesterov = bool(nesterov)
        # The LMO used by every method. Defaults to the exact polar factor, which
        # is what the theorems are stated about; pass ``make_lmo("ns", steps=5)``
        # to run the *implemented* Newton-Schulz oracle instead.
        self.lmo = lmo if lmo is not None else muon_lmo
        # One generator per optimizer, so the tie-break draws of one run cannot
        # shift another's trajectory.
        self.rng = np.random.default_rng(sign_seed)
        self.X = np.zeros(self.shape)   # tracked / exact model
        self.M = np.zeros(self.shape)   # momentum buffer M_t

    def sign(self, Y: np.ndarray) -> np.ndarray:
        """``sign(Y)`` under the paper's randomized convention."""
        return sign_pm1(Y, self.rng)

    # -- momentum ---------------------------------------------------------
    def _effective_direction(self, G: np.ndarray) -> np.ndarray:
        r"""Update ``M_t`` and return the effective direction ``\tilde M_t``.

        ``M_t     = mu * M_{t-1} + (1 - mu) * G_t``            (EMA momentum)
        ``M_tilde = M_t``                                      (default), or
        ``M_tilde = (1 - mu) * G_t + mu * M_t``                (Nesterov).
        """
        self.M = self.mu * self.M + (1.0 - self.mu) * G
        if self.nesterov:
            return (1.0 - self.mu) * G + self.mu * self.M
        return self.M

    # -- driver hooks -----------------------------------------------------
    def grad_point(self) -> np.ndarray:
        """Point at which the next gradient is evaluated."""
        return self.X

    def track_point(self) -> np.ndarray:
        """Model whose objective value is recorded / plotted."""
        return self.X

    def step(self, G: np.ndarray) -> None:                      # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------------------
# Reference methods
# --------------------------------------------------------------------------


class SignSGD(Optimizer):
    """Reference: sign compression of the (momentum) gradient itself."""

    name = "SignSGD"

    def step(self, G):
        M_tilde = self._effective_direction(G)
        self.X = self.X - self.eta * self.sign(M_tilde)


class Muon(Optimizer):
    """Reference: full-precision Muon, ``X <- X - eta * LMO(M_tilde)``."""

    name = "Muon"

    def step(self, G):
        M_tilde = self._effective_direction(G)
        self.X = self.X - self.eta * self.lmo(M_tilde)


# --------------------------------------------------------------------------
# Sign-around-the-LMO methods (no error feedback)
# --------------------------------------------------------------------------


class SignMuon(Optimizer):
    """Algorithm ``central_alg``: sign **after** the LMO.

    ``D = LMO(M_tilde);  X <- X - eta * sign(D)``.
    """

    name = "SignMuon"

    def step(self, G):
        M_tilde = self._effective_direction(G)
        D = self.lmo(M_tilde)
        self.X = self.X - self.eta * self.sign(D)


class MuonUSign(Optimizer):
    """Algorithm ``alg:muon_usign``: sign **before** the LMO (Theorem 2).

    ``s = sign(M_tilde);  D = LMO(s);  X <- X - eta * D``.

    Because the LMO is scale-invariant, feeding it the scaled sign instead of
    the plain sign gives the identical direction. Not to be confused with
    ``MuonSign``, which signs the LMO output as well.
    """

    name = "MuonUSign"

    def step(self, G):
        M_tilde = self._effective_direction(G)
        s = self.sign(M_tilde)
        D = self.lmo(s)
        self.X = self.X - self.eta * D


class MuonSign(Optimizer):
    """Algorithm ``alg:muon_sign``: sign before *and* after the LMO.

    ``s_up = sign(M_tilde);  D = LMO(s_up);  X <- X - eta * sign(D)``.
    """

    name = "MuonSign"

    def step(self, G):
        M_tilde = self._effective_direction(G)
        s_up = self.sign(M_tilde)
        D = self.lmo(s_up)
        s_down = self.sign(D)
        self.X = self.X - self.eta * s_down


# --------------------------------------------------------------------------
# Error-feedback (EF21) methods
# --------------------------------------------------------------------------


class EF21SignMuon(Optimizer):
    """Algorithm ``ef21_signmuon``: EF21 on the LMO *direction*.

    ``D = LMO(M_tilde)`` is tracked by a scaled-sign EF21 estimator
    ``d_est`` and the step is ``X <- X - eta * d_est``.
    """

    name = "EF21-SignMuon"

    def __init__(self, shape, eta, mu=0.0, nesterov=False, lmo=None,
                 sign_seed: int = 0):
        super().__init__(shape, eta, mu, nesterov, lmo, sign_seed)
        self.d_est = np.zeros(self.shape)   # EF21 estimator of the LMO direction

    def step(self, G):
        M_tilde = self._effective_direction(G)
        D = self.lmo(M_tilde)
        delta = D - self.d_est
        alpha = np.mean(np.abs(delta))
        self.d_est = self.d_est + alpha * self.sign(delta)
        self.X = self.X - self.eta * self.d_est


class EF21MuonUSign(Optimizer):
    """Algorithm ``central_alg_ef``: EF21 on the momentum, LMO after.

    A scaled-sign EF21 estimator ``g_est`` tracks the effective momentum
    ``M_tilde``; the step uses the full Muon LMO of the reconstructed
    estimator: ``D = LMO(g_est);  X <- X - eta * D``.
    """

    name = "EF21-MuonUSign"

    def __init__(self, shape, eta, mu=0.0, nesterov=False, lmo=None,
                 sign_seed: int = 0):
        super().__init__(shape, eta, mu, nesterov, lmo, sign_seed)
        self.g_est = np.zeros(self.shape)   # EF21 estimator of the momentum

    def step(self, G):
        M_tilde = self._effective_direction(G)
        delta = M_tilde - self.g_est
        alpha = np.mean(np.abs(delta))
        self.g_est = self.g_est + alpha * self.sign(delta)
        D = self.lmo(self.g_est)
        self.X = self.X - self.eta * D


class EF21MuonSign(Optimizer):
    """Algorithm ``central_alg_ud``: bidirectional EF21 (uplink + downlink).

    Uplink EF21 reconstructs the momentum estimator ``g_est`` (as in
    EF21-MuonUSign) and the server advances the **exact** model ``X`` with the
    Muon LMO step.  A second EF21-P loop compresses the model shift on the
    downlink into ``W``; the client evaluates its gradient at ``W`` while the
    objective is tracked on the exact server model ``X``.
    """

    name = "EF21-MuonSign"

    def __init__(self, shape, eta, mu=0.0, nesterov=False, lmo=None,
                 sign_seed: int = 0):
        super().__init__(shape, eta, mu, nesterov, lmo, sign_seed)
        self.g_est = np.zeros(self.shape)   # uplink EF21 estimator
        self.W = np.zeros(self.shape)       # compressed broadcast model

    def grad_point(self):
        # Gradient is evaluated at the compressed broadcast model.
        return self.W

    def track_point(self):
        # The exact server model carries the "true" progress.
        return self.X

    def step(self, G):
        M_tilde = self._effective_direction(G)
        # --- uplink EF21 ---
        delta_up = M_tilde - self.g_est
        alpha_up = np.mean(np.abs(delta_up))
        self.g_est = self.g_est + alpha_up * self.sign(delta_up)
        # --- exact server step ---
        D = self.lmo(self.g_est)
        self.X = self.X - self.eta * D
        # --- downlink EF21-P (compress the model shift into W) ---
        delta_dn = self.X - self.W
        alpha_dn = np.mean(np.abs(delta_dn))
        self.W = self.W + alpha_dn * self.sign(delta_dn)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

# Insertion order defines the six paper algorithms first, then the two
# references, which is how the runner iterates and plots them.
OPTIMIZERS = {
    "SignMuon":        SignMuon,
    "EF21-SignMuon":   EF21SignMuon,
    "MuonUSign":       MuonUSign,
    "MuonSign":      MuonSign,
    "EF21-MuonUSign":  EF21MuonUSign,
    "EF21-MuonSign": EF21MuonSign,
    "SignSGD":         SignSGD,
    "Muon":            Muon,
}

PAPER_METHODS = ["SignMuon", "EF21-SignMuon", "MuonUSign", "MuonSign",
                 "EF21-MuonUSign", "EF21-MuonSign"]
REFERENCE_METHODS = ["SignSGD", "Muon"]
