"""What each federated method is, and what one round of it costs to transmit.

Split out of ``algorithms.py`` so that the accounting can be read without a GPU.
``algorithms.py`` imports torch at module scope -- it has to, it is the training
loop -- and ``federated.export_article`` needs exactly one thing from it,
`communication_bits`, to fill the two communication columns of `tab:commacct`.
That one import made the whole export depend on torch, so the archive a reviewer
downloads could only be rebuilt on the box that produced the runs; anywhere else
the exporter wrote the table with its communication columns empty.

Nothing here touches a tensor. A method specification is a description of a
protocol -- where the LMO runs, how each channel is compressed, what the server
does with the aggregate -- and the bit accounting is arithmetic over that
description plus the parameter counts a run recorded. Both are properties of the
algorithm, not of any execution of it.

``algorithms.py`` re-exports every name below, so ``from federated.algorithms
import METHODS`` keeps working and there is still exactly one definition of each.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from common.lr_scaling import FAMILY_LMO, FAMILY_SIGN

__all__ = ["MethodSpec", "METHODS", "METHOD_ALIASES", "resolve_method",
           "compresses_downlink", "communication_bits", "method_family"]


@dataclass(frozen=True)
class MethodSpec:
    """Where the LMO runs, how each channel is compressed, and the step family.

    Attributes
    ----------
    lmo : {"none", "worker", "server"}
        ``"worker"``: each client orthogonalizes locally and transmits a
        compressed *direction*. ``"server"``: clients transmit a compressed
        *gradient* and the server applies a single LMO to the reconstruction.
    uplink : {"exact", "sign_mv", "ef21"}
        ``"sign_mv"`` sends one bit per entry and the server takes a majority
        vote; ``"ef21"`` sends the scaled sign of the EF21 residual plus one
        scalar; ``"exact"`` sends the full-precision tensor and the server
        averages.
    downlink : {"exact", "sign", "ef21p"}
        ``"sign"`` broadcasts ``sign(D)`` (both sides apply the same step, so the
        models stay identical); ``"ef21p"`` keeps the exact model ``X`` on the
        server and broadcasts a scaled sign of the model shift, so clients see a
        compressed model ``W`` and evaluate their gradients there.
    server : {"step", "adam"}
        ``"step"``: ``X <- X - lr * D``. ``"adam"``: the aggregate is used as a
        gradient for a server-side Adam.
    family : {"sign", "lmo", None}
        Which per-layer multiplier the step takes. ``sign`` when the matrix
        applied to ``X`` has +-1 entries (``||s||_F = sqrt(mn)``), ``lmo`` when it
        is a polar factor or an estimate of one (``||s||_F = sqrt(min(m,n))``).
        ``None`` for the SGD/Adam baselines, whose step norm is data-dependent so
        that no static multiplier corresponds to the unit-gain criterion -- they
        are run with one global rate unless ``scale_baselines`` is set.
    client_momentum : bool
        Whether clients maintain a momentum buffer at all.
    momentum_form : {"ema", "heavy_ball"}
        See the module docstring; only ``sgd`` needs ``"heavy_ball"``.
    """

    lmo: str = "none"
    uplink: str = "exact"
    downlink: str = "exact"
    server: str = "step"
    family: Optional[str] = None
    client_momentum: bool = True
    momentum_form: str = "ema"

    @property
    def needs_exact_model(self) -> bool:
        """True when the server must keep an exact ``X`` distinct from ``W``."""
        return self.downlink == "ef21p"


METHODS: Dict[str, MethodSpec] = {
    # --- the six paper methods -------------------------------------------
    "signmuon":      MethodSpec(lmo="worker", uplink="sign_mv", downlink="exact",
                                family=FAMILY_SIGN),
    "ef21signmuon":  MethodSpec(lmo="worker", uplink="ef21",    downlink="exact",
                                family=FAMILY_LMO),
    "muonusign":     MethodSpec(lmo="server", uplink="sign_mv", downlink="exact",
                                family=FAMILY_LMO),
    "muonsign":      MethodSpec(lmo="server", uplink="sign_mv", downlink="sign",
                                family=FAMILY_SIGN),
    "ef21muonusign": MethodSpec(lmo="server", uplink="ef21",    downlink="exact",
                                family=FAMILY_LMO),
    "ef21muonsign":  MethodSpec(lmo="server", uplink="ef21",    downlink="ef21p",
                                family=FAMILY_LMO),
    # --- references -------------------------------------------------------
    # TWO full-precision Muons, one per template, because a compressed method must
    # be measured against the uncompressed version of ITS OWN template.
    #
    # ``muon`` orthogonalizes on the worker and the server averages the resulting
    # polar factors -- the uncompressed control for SignMuon and EF21-SignMuon.
    # Averaging near-orthogonal matrices shortens the step: with per-client noise
    # comparable to the shared signal, ||mean_j polar(M_j)||_F is 0.56x
    # ||polar(mean_j M_j)||_F at N=11, and the shortfall grows with N and with
    # heterogeneity. A tuned eta_0 absorbs a *constant* handicap; this one drifts
    # over training as the gradient-to-noise ratio changes.
    #
    # ``muonserver`` sends the momentum uncompressed and applies ONE LMO on the
    # server -- the uncompressed control for MuonUSign, EF21-MuonUSign and
    # EF21-MuonSign, and the method they reduce to when the compressor is the
    # identity. Its step norm is N-independent (measured flat to 0.3% from N=1 to
    # N=21 at every noise level). Comparing the server-LMO family against ``muon``
    # instead confounds "what does the 1-bit uplink cost?" with "what does
    # averaging orthogonal matrices cost?".
    "muon":          MethodSpec(lmo="worker", uplink="exact",   downlink="exact",
                                family=FAMILY_LMO),
    "muonserver":    MethodSpec(lmo="server", uplink="exact",   downlink="exact",
                                family=FAMILY_LMO),
    "signsgd":       MethodSpec(lmo="none",   uplink="sign_mv", downlink="exact",
                                family=FAMILY_SIGN),
    "sgd":           MethodSpec(lmo="none",   uplink="exact",   downlink="exact",
                                family=None, momentum_form="heavy_ball"),
    "adam":          MethodSpec(lmo="none",   uplink="exact",   downlink="exact",
                                family=None, server="adam", client_momentum=False),
}

# Legacy CLI spellings kept working so old commands and logs still resolve.
METHOD_ALIASES = {
    "signmuon_cl": "signmuon",
    "signmuon_ef_21": "ef21muonusign",
    "signmuon_ef_ud": "ef21muonsign",
    "ef_usignmuon": "ef21muonusign",
    "ef_udsignmuon": "ef21muonsign",
    "muon_server": "muonserver",
    "muonlmoserver": "muonserver",
}


def resolve_method(name: str) -> tuple[str, MethodSpec]:
    key = name.strip().lower().replace("-", "").replace(" ", "")
    key = METHOD_ALIASES.get(name.strip().lower(), METHOD_ALIASES.get(key, key))
    if key not in METHODS:
        raise ValueError(
            f"Unknown federated method {name!r}. Available: {sorted(METHODS)}"
        )
    return key, METHODS[key]


def compresses_downlink(spec: MethodSpec) -> bool:
    """Whether the server's per-round broadcast is one bit per matrix entry.

    The criterion is not "does the server apply a compressor" but "is the object
    the server has to distribute already ``+-1``-valued", and three cases satisfy
    it:

    * ``downlink="sign"`` (MuonSign) -- the server signs the LMO output;
    * ``downlink="ef21p"`` (EF21-MuonSign) -- a scaled sign of the model shift;
    * **the majority vote itself** (SignMuon, SignSGD). ``sign(sum_j s_j)`` is
      ``+-1`` in every coordinate -- exactly, at an odd client count, under the
      randomized-zero convention -- so the server broadcasts the *vote* rather
      than the model and each client applies ``X <- X - lr*lam*s_agg`` to its own
      replica. Replicas start from a common ``X_0`` and receive identical
      updates, so they never diverge. This is the property that makes the
      original signSGD-with-majority-vote one bit in both directions
      (Bernstein et al., 2019), and it is a fact about the vote, not about any
      compressor the implementation applies.

    It fails exactly when the server-side quantity is dense: ``polar(.)`` of the
    aggregate (MuonUSign, EF21-MuonUSign) or a scaled average of signs
    (EF21-SignMuon). Those must broadcast a full-precision model.

    Note this is deliberately NOT ``spec.downlink != "exact"``: ``spec.downlink``
    says which compressor the training loop *applies*, and SignMuon's server has
    nothing to apply one to. Reading the accounting off that field is the bug this
    function exists to prevent -- and flipping the field to make the arithmetic
    come out would silently change the algorithm.
    """
    return spec.downlink in ("sign", "ef21p") or (
        spec.uplink == "sign_mv" and spec.lmo != "server")


def communication_bits(name: str, n_matrix: int, n_aux: int,
                       uplink_zero_frac: float = 0.0,
                       n_layers: int = 0,
                       uplink_zeros: str = "random") -> Dict[str, float]:
    """Bits per parameter per round, and the reduction against full precision.

    The paper's headline is a "32x reduction in transmitted data". Four things
    have to be counted for that number to mean anything, and this function counts
    all four:

    * **The uplink alphabet.** Under the paper's convention exact zeros are
      randomized to +-1 (``sign_pm1``), so a symbol is a genuine 1 bit whatever
      the raw zero rate, and the per-channel reduction is the full 32x. Pass the
      run's ``uplink_zeros`` so that this is not assumed: under the legacy
      ``keep`` the majority-vote alphabet is ternary and a symbol costs
      ``H(p0, (1-p0)/2, (1-p0)/2)`` bits, which is what ``uplink_zero_frac`` is
      for -- 1.02 to 1.16 bits at the 0.1-3.0% CNN2 actually shows, and 1.37 at
      the 10% an earlier note assumed. The EF21 channels go through ``sign_pm1``
      unconditionally, so they are binary under either setting.
    * **The auxiliary group is never compressed.** Biases, BatchNorm scales and the
      head go uncompressed in both directions. On CNN2 they are 0.28% of the
      parameters, so this costs little -- but it is what turns "1 bit per
      parameter" into "1 bit per parameter plus epsilon", and on a model with a
      larger head it would not be negligible.
    * **Error feedback carries a scalar per layer.** Both EF21 channels transmit
      ``(sign(residual), alpha)`` with one full-precision ``alpha`` per matrix
      layer -- on the uplink for every EF21 method, and again on the downlink for
      EF21-MuonSign. Pass ``n_layers`` to count it. It is 96 bits per round on
      CNN2's three matrix layers against 762k parameters, i.e. four decimal places
      in, but it is the difference between "one bit per parameter" and "one bit
      per parameter, and here is the constant".
    * **Which methods actually compress the downlink**, per
      ``compresses_downlink`` above: SignMuon, SignSGD, MuonSign and
      EF21-MuonSign do; MuonUSign, EF21-SignMuon and EF21-MuonUSign broadcast an
      uncompressed model and so are capped below 2x round trip however good their
      uplink.

    All figures are per client per round: ``up`` is what one client sends, ``down``
    what the server sends to one client.
    """
    import math

    _, spec = resolve_method(name)
    if uplink_zeros not in ("keep", "random", "positive"):
        # Not pedantry: a typo here would silently report the idealized 1 bit for a
        # run that actually transmitted a third symbol, which is the whole point of
        # taking the convention as an argument.
        raise ValueError(f"uplink_zeros must be 'keep', 'random' or 'positive', "
                         f"got {uplink_zeros!r}")
    p0 = min(max(float(uplink_zero_frac), 0.0), 1.0 - 1e-12)
    ternary = (spec.uplink == "sign_mv" and uplink_zeros == "keep" and p0 > 0.0)
    if spec.uplink == "exact":
        up_per = 32.0
    elif not ternary:
        up_per = 1.0
    else:
        up_per = -(p0 * math.log2(p0) + (1.0 - p0) * math.log2((1.0 - p0) / 2.0))
    down_per = 1.0 if compresses_downlink(spec) else 32.0

    # One full-precision scale per matrix layer, on each error-feedback channel.
    up_scalars = 32.0 * n_layers if spec.uplink == "ef21" else 0.0
    down_scalars = 32.0 * n_layers if spec.downlink == "ef21p" else 0.0

    total = n_matrix + n_aux
    up = up_per * n_matrix + 32.0 * n_aux + up_scalars
    down = down_per * n_matrix + 32.0 * n_aux + down_scalars
    base = 32.0 * total
    return {
        # Per *symbol* on the compressed channel, before the uncompressed
        # auxiliary group and the EF21 scales are averaged in: 1.0 under the
        # paper's convention, the ternary entropy under ``--uplink-zeros keep``.
        "uplink_bits_per_symbol": up_per,
        "uplink_bits_per_param": up / total if total else 0.0,
        "downlink_bits_per_param": down / total if total else 0.0,
        "uplink_reduction": base / up if up else 0.0,
        "downlink_reduction": base / down if down else 0.0,
        "round_trip_reduction": (2.0 * base) / (up + down) if up + down else 0.0,
    }


def method_family(name: str, scale_baselines: bool = False) -> Optional[str]:
    """The per-layer family a method's step belongs to.

    ``scale_baselines`` gives SGD/Adam the sign-family rule. It is off by default
    for the same reason as centrally: Adam's step is approximately a sign step so
    the rule is at least *arguable* for it, while SGD's step is ``eta * m``, whose
    norm is data-dependent -- no static multiplier implements the unit-gain
    criterion there, and applying one anyway would be an arbitrary rescaling
    dressed up as a parameterization.
    """
    _, spec = resolve_method(name)
    if spec.family is not None:
        return spec.family
    return FAMILY_SIGN if scale_baselines else None
