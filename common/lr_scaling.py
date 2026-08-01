"""Principled per-layer learning-rate scaling for the sign/LMO family.

The problem
-----------
The theory treats a single matrix, so one scalar step size suffices. A network has
layers of very different shapes, and the six methods produce step matrices from
two families whose norms scale *differently* with shape. One global learning rate
therefore cannot be right for both families and across layers simultaneously.

This is not a hack bolted onto the theory. The layer-wise LMO framework this
paper's convergence result reduces to (Gluon, Riabinin et al.; EF21-Muon,
Gruntkowska et al.) is *already* per-layer: per-layer norms ``||.||_(l)``,
smoothness constants ``L_l``, and radii. The experiments have been setting that
radius to a constant. This module instantiates it.

The criterion: unit gain
------------------------
Define the **RMS gain** of a matrix ``A in R^{m x n}`` -- how much it amplifies a
generic input in root-mean-square terms::

    gamma(A) := rms(A u) / rms(u),      u ~ N(0, I_n),   rms(v) := ||v|| / sqrt(dim)

This is exact and deterministic, because ``E||A u||^2 = ||A||_F^2``::

    gamma(A) = ||A||_F / sqrt(m).                                          (*)

No random-matrix theory, no distributional fitting. The one modelling assumption in
(*) is that the input is isotropic **and independent of** ``A``. That is the right
model for a single step, and it is where ``mup`` parts company: if the *accumulated*
update correlates with the activations, the gain can be as large as
``|A|_op * sqrt(n/m) = eta * n`` instead of ``eta * sqrt(n)``, giving ``alpha = 1``
rather than ``1/2``. So ``alpha`` in ``[1/2, 1]`` is exactly the incoherent-to-aligned
range, and the measurements below locate it. Now evaluate (*) on the two step
families, using their *exact* Frobenius norms (``||U V^T||_F^2 = tr(V V^T) = r``;
a ``+-1`` matrix has ``||s||_F = sqrt(mn)``), with ``r = min(m, n)`` generically:

===========  ==================================  =================  =========================
family       methods                             ``||s||_F``        ``gamma(eta * s)``
===========  ==================================  =================  =========================
``lmo``      Muon, MuonUSign, EF21-MuonUSign,    ``sqrt(min(m,n))`` ``eta / sqrt(max(1,m/n))``
             EF21-MuonSign, EF21-SignMuon
``sign``     SignMuon, MuonSign, SignSGD         ``sqrt(mn)``       ``eta * sqrt(n)``
===========  ==================================  =================  =========================

Any standard fan-in-scaled initialization has *shape-independent* gain: for He
normal (``sigma^2 = 2/n``), ``||X||_F = sqrt(2m)`` so ``gamma(X) = sqrt(2)``; for
PyTorch's default Kaiming-uniform Conv2d, ``gamma(X) = 1/sqrt(3)``. Either way a
constant, absorbed into ``eta_0``. So requiring the update's gain to be a fixed
fraction of the weight's is just: **make the per-step gain the same on every
layer**. By (*) that is one formula,

    lambda = sqrt(fan_out) / ||s||_F,            so that  gamma(eta_0 * lambda * s) = eta_0

exactly, for every shape and both families -- ``eta_0`` literally *is* the per-step
RMS gain. Substituting the two Frobenius norms gives the closed forms::

    lmo   family:  eta_layer = eta_0 * sqrt(max(1, m/n))     <-- Muon's shipped factor
    sign  family:  eta_layer = eta_0 / sqrt(fan_in)          <-- the missing counterpart

The first line is the strongest available validation of the criterion: it
*derives* the ``max(1, m/n)^{1/2}`` heuristic in the reference Muon
implementation, which is also why Muon's learning rate is known to transfer
across widths. The sign family has simply never been given its counterpart.

What the alternatives assume
----------------------------
``mup`` applies the same criterion to the **accumulated** update rather than to
one step, under the assumption that successive sign steps align with the
activations, so that ``gamma`` grows like ``n`` instead of ``sqrt(n)``. That yields
the classical muP/Adam rule ``eta ~ 1/fan_in``. Whether the accumulation aligns is
an empirical question -- and it is the *only* thing separating ``unit-gain`` from
``mup``. Two measurements settle it:

* single step: ``python3 -m common.lr_scaling --measure`` shows
  ``||sign(polar(M))||_op = 0.93 (sqrt m + sqrt n)`` to within 8% at every
  ResNet-18 shape, i.e. a single sign step is spectrally *incoherent*, not
  rank-one aligned.
* accumulated: train with ``--log-gain`` and check whether the realized gain of
  ``X_t - X_0`` grows like ``sqrt(t)`` (incoherent, favours ``unit-gain``) or like
  ``t`` (aligned, favours ``mup``).

``mishra-analysis`` is the normalization ``D = S / sqrt(mn)`` used in the *analysis*
of the concurrent Sign-Muon paper (Mishra et al.), which substitutes the bound
``||s||_op <= ||s||_F`` for the true operator norm. Note their **algorithm** applies
no shape factor at all (a single global ``eta``, i.e. the ``none`` preset here);
their remark that the two are "equivalent after absorbing ``sqrt(mn)`` into
``eta_t``" holds for a single matrix but not across layers of differing shape. The
``--measure`` diagnostic quantifies the cost: the Frobenius bound overstates the
operator norm by ``3.7x`` (conv1) to ``18.1x`` (layer4), so the induced spectral
trust region drifts ~5x across a ResNet-18 rather than staying flat.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple

__all__ = [
    "FAMILY_LMO",
    "FAMILY_SIGN",
    "ScalingRule",
    "RULES",
    "resolve_rule",
    "fan_in_out",
    "layer_multiplier",
    "describe_rule",
]

FAMILY_LMO = "lmo"      # step is polar(.):  ||s||_F = sqrt(min(m, n))
FAMILY_SIGN = "sign"    # step has +-1 entries: ||s||_F = sqrt(m n)


def fan_in_out(shape: Sequence[int]) -> Tuple[int, int]:
    """``(fan_out, fan_in)`` of a parameter, matching the LMO's 2-D reshape.

    A conv weight ``[c_out, c_in, kh, kw]`` flattens to ``[c_out, c_in*kh*kw]``
    exactly as ``muon_lmo`` does, so ``fan_in = c_in*kh*kw`` -- PyTorch's own
    convention. A ``Linear`` weight ``[out, in]`` passes through unchanged.
    """
    if len(shape) < 2:
        raise ValueError(
            f"per-layer scaling applies to matrix parameters; got shape {tuple(shape)}")
    m = int(shape[0])
    n = 1
    for s in shape[1:]:
        n *= int(s)
    return m, n


# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ScalingRule:
    """A per-layer multiplier ``lambda`` with ``eta_layer = eta_0 * lambda``."""

    name: str
    multiplier: Callable[[str, int, int], float]     # (family, m, n) -> lambda
    description: str
    criterion: str = ""


def _aspect(m: int, n: int) -> float:
    """``sqrt(max(1, fan_out/fan_in))`` -- the unit-gain factor for an LMO step."""
    return math.sqrt(max(1.0, m / n))


def _unit_gain(family: str, m: int, n: int) -> float:
    if family == FAMILY_LMO:
        return _aspect(m, n)
    return 1.0 / math.sqrt(n)


def _mup(family: str, m: int, n: int) -> float:
    if family == FAMILY_LMO:
        return _aspect(m, n)
    return 1.0 / n


def _legacy(family: str, m: int, n: int) -> float:
    return _aspect(m, n) if family == FAMILY_LMO else 1.0


def _none(family: str, m: int, n: int) -> float:
    return 1.0


def _mishra_analysis(family: str, m: int, n: int) -> float:
    # Unit-Frobenius step: 1/||s||_F for both families.
    return 1.0 / math.sqrt(min(m, n) if family == FAMILY_LMO else m * n)


RULES: Dict[str, ScalingRule] = {
    "unit-gain": ScalingRule(
        "unit-gain", _unit_gain,
        "lmo: eta_0*sqrt(max(1,m/n));  sign: eta_0/sqrt(fan_in)",
        "the update's RMS gain is a fixed fraction of the initialization's, "
        "per step, with no alignment assumption"),

    "mup": ScalingRule(
        "mup", _mup,
        "lmo: eta_0*sqrt(max(1,m/n));  sign: eta_0/fan_in",
        "as unit-gain but applied to the ACCUMULATED update assuming successive "
        "sign steps align with the activations (the classical muP/Adam rule)"),

    "legacy": ScalingRule(
        "legacy", _legacy,
        "lmo: eta_0*sqrt(max(1,m/n));  sign: eta_0",
        "what the paper's experiments currently do: the reference Muon aspect "
        "factor for LMO steps, a single global rate for sign steps"),

    "none": ScalingRule(
        "none", _none,
        "eta_0 for every layer and both families",
        "no per-layer scaling; this is also what the concurrent Sign-Muon "
        "implementation (Mishra et al., Algorithm 1) does"),

    "mishra-analysis": ScalingRule(
        "mishra-analysis", _mishra_analysis,
        "lmo: eta_0/sqrt(min(m,n));  sign: eta_0/sqrt(mn)",
        "unit-Frobenius step, the normalization used in the ANALYSIS of Mishra "
        "et al. (their algorithm applies no shape factor); substitutes "
        "||s||_op <= ||s||_F for the true operator norm"),
}


def power_rule(alpha: float, beta: float) -> ScalingRule:
    """``lambda_sign = m^-beta * n^-alpha``; the LMO family keeps the aspect factor.

    The presets are points in this two-parameter family, which is what turns the
    exponent into an empirically answerable question:

    ===================  =======  ======
    rule                  alpha    beta
    ===================  =======  ======
    ``none``                0       0
    ``unit-gain``          1/2      0
    ``mishra-analysis``    1/2     1/2
    ``mup``                 1       0
    ===================  =======  ======
    """
    def multiplier(family: str, m: int, n: int) -> float:
        if family == FAMILY_LMO:
            return _aspect(m, n)
        return (m ** -beta) * (n ** -alpha)

    return ScalingRule(
        f"power:{alpha:g},{beta:g}", multiplier,
        f"lmo: eta_0*sqrt(max(1,m/n));  sign: eta_0 * m^-{beta:g} * n^-{alpha:g}",
        "explicit exponents, for sweeping the family")


def resolve_rule(spec: str) -> ScalingRule:
    """Look up a preset by name, or parse ``power:ALPHA[,BETA]``."""
    key = spec.strip().lower()
    if key in RULES:
        return RULES[key]
    if key.startswith("power:"):
        parts = [p for p in key[len("power:"):].split(",") if p]
        if len(parts) == 1:
            return power_rule(float(parts[0]), 0.0)
        if len(parts) == 2:
            return power_rule(float(parts[0]), float(parts[1]))
    raise ValueError(
        f"Unknown lr-scaling rule {spec!r}. Presets: {sorted(RULES)}; "
        f"or 'power:ALPHA[,BETA]'.")


def layer_multiplier(rule: ScalingRule, family: str, shape: Sequence[int]) -> float:
    """Per-layer multiplier for one parameter."""
    if family not in (FAMILY_LMO, FAMILY_SIGN):
        raise ValueError(f"unknown family {family!r}; expected {FAMILY_LMO!r} or {FAMILY_SIGN!r}")
    m, n = fan_in_out(shape)
    return rule.multiplier(family, m, n)


def describe_rule(rule: ScalingRule, family: str,
                  shapes: Sequence[Tuple[str, Sequence[int]]]) -> str:
    """Table of the multipliers a rule assigns, for the run log.

    An unlogged per-layer learning rate is an unreproducible one, and the spread
    is the number a reader will want.
    """
    lines = [f"LR scaling '{rule.name}' (family={family}): {rule.description}"]
    if rule.criterion:
        lines.append(f"  criterion: {rule.criterion}")
    lines.append(f"  {'parameter':<36}{'fan_out':>8}{'fan_in':>8}{'lambda':>13}")
    mults = []
    for name, shape in shapes:
        m, n = fan_in_out(shape)
        lam = rule.multiplier(family, m, n)
        mults.append(lam)
        lines.append(f"  {name:<36}{m:>8}{n:>8}{lam:>13.6g}")
    if mults:
        lo, hi = min(mults), max(mults)
        lines.append(f"  spread {hi / lo:.1f}x   (min {lo:.4g}, max {hi:.4g})")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------

# Every distinct (fan_out, fan_in) in the CIFAR ResNet-18 of common/models.py.
RESNET18_SHAPES: Sequence[Tuple[str, int, int]] = (
    ("conv1", 64, 27),
    ("layer1", 64, 576),
    ("layer2.first", 128, 576),
    ("layer2", 128, 1152),
    ("layer2.downsample", 128, 64),
    ("layer3.first", 256, 1152),
    ("layer3", 256, 2304),
    ("layer3.downsample", 256, 128),
    ("layer4.first", 512, 2304),
    ("layer4", 512, 4608),
    ("layer4.downsample", 512, 256),
)


def _measure(draws: int = 5, seed: int = 0) -> None:  # pragma: no cover - diagnostic
    """Is a single sign step spectrally incoherent or aligned?

    Compares the measured ``||sign(polar(M))||_op`` against ``sqrt(m)+sqrt(n)``
    (incoherent) and ``sqrt(mn)`` (the Frobenius bound / rank-one-aligned
    extreme). NumPy only, a few seconds.
    """
    import numpy as np

    rng = np.random.default_rng(seed)

    def polar(M):
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        r = int(np.count_nonzero(S > 1e-12 * S[0]))
        return U[:, :r] @ Vt[:r, :]

    def opnorm(X):
        return float(np.linalg.svd(X, compute_uv=False)[0])

    print(f"Operator norm of a single sign step ({draws} Gaussian draws per shape)\n")
    print(f"{'layer':<20}{'m':>5}{'n':>6}{'|sign polar|op':>15}{'|sign G|op':>12}"
          f"{'sqrt(mn)':>10}{'ratio':>7}{'sqrt m+sqrt n':>15}{'ratio':>7}")
    print("-" * 97)
    r_fro, r_inc = [], []
    for name, m, n in RESNET18_SHAPES:
        a = float(np.mean([opnorm(np.sign(polar(rng.normal(size=(m, n)))))
                           for _ in range(draws)]))
        b = float(np.mean([opnorm(np.sign(rng.normal(size=(m, n)))) for _ in range(draws)]))
        fro, inc = math.sqrt(m * n), math.sqrt(m) + math.sqrt(n)
        r_fro.append(fro / a)
        r_inc.append(inc / a)
        print(f"{name:<20}{m:>5}{n:>6}{a:>15.1f}{b:>12.1f}"
              f"{fro:>10.1f}{fro / a:>7.1f}{inc:>15.1f}{inc / a:>7.2f}")
    print(f"\n  sqrt(mn)        overstates by {min(r_fro):.1f}x-{max(r_fro):.1f}x  "
          f"-- SHAPE-DEPENDENT ({max(r_fro) / min(r_fro):.1f}x drift across layers)")
    print(f"  sqrt m+sqrt n   overstates by {min(r_inc):.2f}x-{max(r_inc):.2f}x  "
          f"-- flat to {100 * (max(r_inc) / min(r_inc) - 1):.0f}%")
    print("\n  => a single sign step is spectrally INCOHERENT, not rank-one aligned.")


def _compare(rules: Sequence[str]) -> None:  # pragma: no cover - diagnostic
    """Per-layer multiplier profiles, each rule normalized to its geometric mean.

    The overall constant is absorbed by the tuned ``eta_0``, so only the *shape* of
    the profile distinguishes two rules.
    """
    print("\nSIGN-family multipliers, each rule normalized to its geometric mean")
    print("(the constant is absorbed into eta_0; only the profile is a real difference)\n")
    header = f"{'layer':<20}{'m':>5}{'n':>6}" + "".join(f"{r:>18}" for r in rules)
    print(header)
    print("-" * len(header))
    profiles = {}
    for r in rules:
        rule = resolve_rule(r)
        vals = [rule.multiplier(FAMILY_SIGN, m, n) for _, m, n in RESNET18_SHAPES]
        gm = math.exp(sum(math.log(v) for v in vals) / len(vals))
        profiles[r] = [v / gm for v in vals]
    for i, (name, m, n) in enumerate(RESNET18_SHAPES):
        print(f"{name:<20}{m:>5}{n:>6}"
              + "".join(f"{profiles[r][i]:>18.4g}" for r in rules))
    print("-" * len(header))
    print(f"{'spread (max/min)':<31}"
          + "".join(f"{max(profiles[r]) / min(profiles[r]):>17.1f}x" for r in rules))


def main() -> None:  # pragma: no cover - diagnostic
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--measure", action="store_true",
                   help="measure the single-step sign operator norm (needs numpy)")
    p.add_argument("--draws", type=int, default=5)
    p.add_argument("--compare", nargs="*", default=None,
                   help="print the multiplier profiles of these rules")
    args = p.parse_args()

    if args.measure:
        _measure(args.draws)
    if args.compare is not None:
        _compare(args.compare or ["none", "legacy", "unit-gain", "mup", "mishra-analysis"])
    if not args.measure and args.compare is None:
        for name, rule in RULES.items():
            print(f"{name:<18} {rule.description}")
            if rule.criterion:
                print(f"{'':<18} ({rule.criterion})")


if __name__ == "__main__":
    main()
