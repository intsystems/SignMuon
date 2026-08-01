"""
Run all eight optimizers on the two linear counterexamples and report which
methods diverge.

Usage
-----
Run from ``code/``, the package root:

    python3 -m counterexamples.run_counterexamples             # default mu=0.0
    python3 -m counterexamples.run_counterexamples --nesterov  # Nesterov momentum
    python3 -m counterexamples.run_counterexamples --mu 0.9 --T 80 --eta 2e-3

Outputs
-------
* a per-problem table of  f[0], f[T-1], the mean per-step descent inner product
  <G, d_t>, and a DIVERGES / descends verdict, and
* one three-panel figure ``counterexamples_main`` -- SignMuon (4x4), MuonUSign
  and MuonSign (5x5), EF21-SignMuon (2x2) -- written as PNG + PDF to
  ``figures/`` and as PDF to ``../aaai_article/images/counterexamples/``, which
  is what LaTeX includes.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")            # headless-safe; figures are written to disk
import matplotlib.pyplot as plt  # noqa: E402

from common.plotting import (AXIS, FS_ANNOT, FS_LABEL, FS_LEGEND,  # noqa: E402
                             SURFACE, TEXT_WIDTH, INK_2, color_of, figure_legend,
                             label_of, marker_of, style_axes, use_paper_style)
from counterexamples.optimizers import OPTIMIZERS, PAPER_METHODS, REFERENCE_METHODS  # noqa: E402
from counterexamples.problems import (  # noqa: E402
    make_linear_problem,
    signmuon_counterexample,
    muonsign_counterexample,
    ef21_signmuon_counterexample,
)

use_paper_style()

# Colour and marker come from ``common.plotting`` -- the repository's single
# method-to-style map -- so a method looks the same here as in the CIFAR,
# synthetic and federated figures.  The dash pattern is local, and it earns its
# place: this is the only figure in the paper with eight curves on one axis, and
# on these instances several of the descending ones are exactly coincident
# (SignSGD and MuonSign in the left panel; Muon and the two EF21 baselines in
# both linear panels).  Cycling four patterns interleaves the dashes of
# neighbouring methods so a covered curve still shows through the gaps.
DASHES = [(0, (4, 2)), (0, (1.6, 1.6)), (0, (5.5, 1.6, 1, 1.6)), (0, (2.6, 1.4))]

# Map the internal algorithm names to the paper's display names.  The code's
# "sign before *and* after the LMO" method (``MuonSign``) is the paper's
# ``MuonSign`` (Theorem 3), and its error-feedback counterpart ``EF21-MuonSign``
# is the paper's bidirectional ``EF21-MuonSign``.  Everything else is unchanged.
DISPLAY_NAMES = {name: label_of(name) for name in
                 PAPER_METHODS + REFERENCE_METHODS}

# Divergence tests.  For a LINEAR objective f(W)=Tr(G^T W), f decreases iff the
# per-step descent inner product <G, d_t> is positive, so the exact, eta/T-free
# test is "mean <G, d_t> persistently negative" (verdict_mode="inner").  For the
# universal EF21-SignMuon instance (periodic ramps, not a quadratic) the ascent is
# second-order: <G, d_t> stays POSITIVE while f rises, so the verdict has to be
# read off f itself (verdict_mode="period").
#
# It is read as the theorem states it -- f(X_{t+2}) - f(X_t) = c > 0 for every t
# past the transient -- rather than from a tail slope against an absolute
# tolerance.  That distinction is not cosmetic.  The bounded methods oscillate over
# a range that scales with the instance's periodic constant A(mu) (199 at
# mu = 0.99), so over any fixed window their apparent slope scales with it too:
# an absolute threshold small enough to catch the ascent at mu = 0 also reports
# every bounded method as ascending at mu = 0.9.  The per-period increment has no
# such failure mode -- a bounded trajectory cannot hold a constant positive one --
# and separates the eight methods correctly at every mu in {0, 0.25, 0.5, 0.9,
# 0.95, 0.99}, both variants, at T = 60 and at T = 400.
DIVERGE_TOL = 1e-6
#: A period-two increment counts as divergence when it is positive throughout the
#: tail and constant to this relative spread.  EF21-SignMuon holds 49/240 to
#: machine precision; the widest spread any bounded method achieves is O(1).
PERIOD_TOL = 0.05

# The figure is authored at its printed width (``TEXT_WIDTH``, a two-column
# float), so every point size below is a point on the page.


def run(opt_cls, grad_fn, loss_fn, shape, T, eta, mu, nesterov):
    """Run one optimizer; return (losses, mean per-step <G, d_t>)."""
    opt = opt_cls(shape, eta=eta, mu=mu, nesterov=nesterov)
    losses = []
    inner = []
    for _ in range(T):
        prev = opt.track_point().copy()
        losses.append(loss_fn(opt.track_point()))
        G = grad_fn(opt.grad_point())
        opt.step(G)
        # d_t reconstructed from the tracked-model shift:  X_{t+1} = X_t - eta*d_t
        d_t = (prev - opt.track_point()) / eta
        inner.append(float(np.sum(G * d_t)))
    return np.asarray(losses), float(np.mean(inner))


def _period_increment(losses, T):
    """``(min, relative spread)`` of ``f[t+2] - f[t]`` over the tail.

    Theorem 4 states the divergence as a *constant positive* period-two
    increment, and that is what this measures: a bounded trajectory cannot hold
    one, however wide its oscillation. The window starts at ``T // 2`` so the
    transient is excluded.
    """
    tail = np.asarray(losses[T // 2:])
    if tail.size < 4:
        return (0.0, float("inf"))
    d = tail[2:] - tail[:-2]
    mean = float(np.mean(d))
    if mean == 0.0:
        return (float(d.min()), float("inf"))
    return (float(d.min()), float((d.max() - d.min()) / abs(mean)))


def run_problem(title, grad_fn, loss_fn, shape, T, eta, mu, nesterov,
                verdict_mode="inner"):
    """Run every optimizer on one problem, print a table.

    ``verdict_mode`` selects the divergence test: ``"inner"`` (mean <G,d> < 0,
    for the linear objectives) or ``"period"`` (a constant positive period-two
    increment of f, for the curvature-driven EF21-SignMuon instance).

    Returns ``(results, diverging)`` where ``results`` maps method -> loss
    trajectory and ``diverging`` lists the methods that diverge.
    """
    order = PAPER_METHODS + REFERENCE_METHODS

    results, diverging = {}, []
    print(f"\n{title}")
    print(f"  dim={shape}, eta={eta}, mu={mu}, "
          f"momentum={'Nesterov' if nesterov else 'standard'}, T={T}, "
          f"test={verdict_mode}")
    print(f"  {'method':<17}{'f[0]':>9}{'f[T-1]':>12}{'mean<G,d>':>12}"
          f"{'tail slope':>12}   verdict")
    print("  " + "-" * 76)
    for name in order:
        losses, mean_inner = run(OPTIMIZERS[name], grad_fn, loss_fn,
                                 shape, T, eta, mu, nesterov)
        results[name] = losses
        # Measure over a whole number of periods: the EF21-SignMuon trajectory
        # is period-two, so mismatched endpoint parity leaks half an oscillation
        # into the slope and hides the exact rate.
        half = T // 2
        half -= (T - 1 - half) % 2
        slope = float((losses[-1] - losses[half]) / max(1, (T - 1 - half))) / eta
        if verdict_mode == "period":
            lo, spread = _period_increment(losses, T)
            is_div = lo > DIVERGE_TOL and spread < PERIOD_TOL
        else:
            is_div = mean_inner < -DIVERGE_TOL
        if is_div:
            diverging.append(name)
        verdict = "DIVERGES (up)" if is_div else "descends"
        print(f"  {name:<17}{losses[0]:>9.3f}{losses[-1]:>12.3f}"
              f"{mean_inner:>12.3f}{slope:>12.4f}   {verdict}")
    return results, diverging


def _draw_curves(ax, results, diverging, T, order):
    """Plot every trajectory on ``ax``.

    The diverging method is the point of each panel, so it is solid, heavier and
    on top. The other seven are not background: they carry the claim that every
    other method descends, so they are drawn at a weight that survives printing
    and are separated from one another by dash pattern and by staggered markers
    (``markevery`` offset) rather than by being faint.
    """
    for i, name in enumerate(order):
        losses = results[name]
        emph = name in diverging
        ax.plot(
            np.arange(len(losses)), losses, label=DISPLAY_NAMES[name],
            color=color_of(name), linewidth=2.0 if emph else 1.5,
            linestyle="-" if emph else DASHES[i % len(DASHES)],
            marker=marker_of(name) or None, markersize=3.8 if emph else 3.2,
            markeredgecolor=SURFACE, markeredgewidth=0.5,
            markevery=(2 * i, max(1, T // 6)),
            zorder=5 if emph else 3, clip_on=True,
        )
    ax.axhline(0, color=AXIS, linewidth=0.8, zorder=1)
    ax.set_xlabel("iteration", color=INK_2, fontsize=FS_LABEL)
    ax.set_xlim(-T * 0.02, T * 1.02)


def _annotate_diverging(ax, results, diverging, T, top=None):
    """Name each ascending curve directly above the curve, in its own colour.

    Directly, not with a leader line into a corner: at this panel size an arrow
    long enough to reach a free corner crosses the very curve it points at. The
    anchor is taken inside the frame, so in the third panel -- where
    EF21-SignMuon climbs out of the top -- the label sits on the visible ascent.
    """
    for i, name in enumerate(diverging):
        curve = np.asarray(results[name])[:T]
        inside = np.where(curve <= top)[0] if top is not None \
            else np.arange(len(curve))
        if not inside.size:
            continue
        # Staggered along the x-axis: two ascending curves in one panel put
        # their labels at the same height otherwise, and the shallower one lands
        # on the steeper one.
        frac = 0.55 - 0.18 * i
        k = int(inside[min(len(inside) - 1, int(frac * len(inside)))])
        # Above the curve by default -- an ascending curve leaves its upper-left
        # empty -- but below it when a steeper ascent runs overhead, which is the
        # MuonUSign/MuonSign panel.
        overhead = any(np.asarray(results[other])[k] > curve[k]
                       for other in diverging if other != name)
        ax.annotate(
            DISPLAY_NAMES[name] + r" $\uparrow$",
            xy=(k, curve[k]), xytext=(-5, -6 if overhead else 6),
            textcoords="offset points", color=color_of(name),
            fontstyle="italic", ha="right",
            va="top" if overhead else "bottom",
            fontsize=FS_ANNOT, zorder=6,
        )


def _save(fig, outfiles):
    """Write each ``(stem, want_png)`` target: PDF always, PNG only where asked.
    The local ``figures/`` dir keeps PNG+PDF (handy for quick viewing); the
    paper's image dir gets the PDF only (LaTeX embeds the vector version)."""
    msgs = []
    for stem, want_png in outfiles:
        fig.savefig(stem + ".pdf", facecolor=SURFACE)
        if want_png:
            fig.savefig(stem + ".png", dpi=200, facecolor=SURFACE)
        msgs.append(stem + (".{png,pdf}" if want_png else ".pdf"))
    plt.close(fig)
    print(f"  saved -> {', '.join(msgs)}")


def plot_panels(panels, outfiles):
    """One three-panel figure: the three counterexamples side by side.

    ``panels`` is a list of dicts with the keys ``results``, ``diverging``,
    ``T``, ``ylabel`` and optionally ``ylim``.  No panel titles are drawn --- the
    paper's subcaptions name them --- and a single legend sits under all three,
    since the eight methods are the same in every panel.
    """
    order = PAPER_METHODS + REFERENCE_METHODS
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.35), squeeze=False)

    for ax, p in zip(axes[0], panels):
        style_axes(ax)
        T = p["T"]
        _draw_curves(ax, p["results"], p["diverging"], T, order)
        if p.get("ylim"):
            ax.set_ylim(*p["ylim"])
        if p["ylabel"]:
            ax.set_ylabel(p["ylabel"], color=INK_2, fontsize=FS_LABEL)
        _annotate_diverging(ax, p["results"], p["diverging"], T,
                            top=p["ylim"][1] if p.get("ylim") else None)

    handles, labels = axes[0][0].get_legend_handles_labels()
    figure_legend(fig, handles, labels, ncol=8, fontsize=FS_LEGEND)

    fig.subplots_adjust(left=0.095, right=0.995, top=0.97, bottom=0.30,
                        wspace=0.24)
    _save(fig, outfiles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mu", type=float, default=None,
                        help="momentum coefficient; overrides every per-problem "
                             "default (default: use each problem's own mu)")
    parser.add_argument("--nesterov", action="store_true",
                        help="use Nesterov momentum instead of standard")
    parser.add_argument("--eta", type=float, default=None,
                        help="learning rate (overrides the per-problem default)")
    parser.add_argument("--T", type=int, default=None,
                        help="number of iterations (overrides the per-problem default)")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    figdir = os.path.join(here, "figures")
    os.makedirs(figdir, exist_ok=True)
    # The paper picks up the same PDFs from aaai_article/images/counterexamples/.
    images_dir = os.path.abspath(
        os.path.join(here, "..", "..", "aaai_article", "images", "counterexamples"))
    have_images = os.path.isdir(os.path.dirname(images_dir))
    if have_images:
        os.makedirs(images_dir, exist_ok=True)

    def stems(basename):
        # (stem, want_png): the local figures/ dir keeps PNG+PDF; the paper's
        # image dir gets the PDF only.
        out = [(os.path.join(figdir, basename), True)]
        if have_images:
            out.append((os.path.join(images_dir, basename), False))
        return out

    # Per-problem step size and horizon.  The two gradients live on very
    # different scales (problem 1 has sigma_1 = 1001; problem 2 has M = 100), so
    # each gets an eta that makes the divergence clearly visible without the
    # steep descenders dominating the axis.  --eta / --T override both.
    #
    # Momentum: the two LINEAR counterexamples are run momentum-free (mu = 0).
    # The gradient is the constant G, so the momentum buffer is (1 - mu^t) G and
    # six of the eight methods are momentum-invariant in exact arithmetic: the
    # five whose step is scale-invariant in its target, plus EF21-SignMuon, whose
    # EF21 target LMO((1 - mu^t) G) = LMO(G) is constant in t.  EF21-MuonUSign
    # and EF21-MuonSign are not -- their EF21 target (1 - mu^t) G does move with
    # t -- so their trajectories vary with mu, but stay bounded: no verdict
    # changes.
    G1, _ = signmuon_counterexample()
    G2, _ = muonsign_counterexample(eps=1.0, M=100.0)
    grad1, loss1 = make_linear_problem(G1)
    grad2, loss2 = make_linear_problem(G2)
    lin_ylabel = r"$f(\mathbf{W})=\mathrm{Tr}(\mathbf{G}^\top\mathbf{W})$"
    problems = [
        dict(
            title="Counterexample 1 -- SignMuon (Theorem 1)",
            grad_fn=grad1, loss_fn=loss1, shape=G1.shape, eta=1e-3, T=60,
            mu=0.0, verdict_mode="inner", ylabel=lin_ylabel,
        ),
        dict(
            title="Counterexample 2 -- MuonSign / MuonUSign (Theorems 2--3)",
            grad_fn=grad2, loss_fn=loss2, shape=G2.shape, eta=5e-3, T=60,
            # No ylabel: this panel plots the same objective as the one to its
            # left, and repeating the formula only costs panel width.
            mu=0.0, verdict_mode="inner", ylabel="",
        ),
        dict(
            title="Counterexample 3 -- EF21-SignMuon (Theorem 4)",
            # The universal construction depends on (mu, variant); it is rebuilt
            # per run below.  It diverges for EVERY mu in [0,1) and both variants
            # (the iterate trajectory is identical), so any mu witnesses it.
            #
            # This objective is bounded below, so the seven other methods sit in
            # a narrow band near the floor while EF21-SignMuon climbs away at
            # 49/480 per step.  ``ylim`` frames that band and then leaves room
            # above it: enough of the ascent is in view for its slope to read as
            # a straight line, and the band below is still thick enough to see
            # the bounded methods separate.
            builder=ef21_signmuon_counterexample, eta=1.0, T=60,
            mu=0.0, verdict_mode="period", ylim=(-1.2, 6.5),
            ylabel=r"$f(\mathbf{X}_t)$",
        ),
    ]

    panels = []
    for p in problems:
        eta = args.eta if args.eta is not None else p["eta"]
        T = args.T if args.T is not None else p["T"]
        mu = args.mu if args.mu is not None else p["mu"]
        if "builder" in p:
            grad_fn, loss_fn, shape, _ = p["builder"](mu=mu, nesterov=args.nesterov)
        else:
            grad_fn, loss_fn, shape = p["grad_fn"], p["loss_fn"], p["shape"]
        results, diverging = run_problem(
            p["title"], grad_fn, loss_fn, shape, T, eta,
            mu, args.nesterov, verdict_mode=p["verdict_mode"])
        panels.append(dict(results=results, diverging=diverging, T=T,
                           ylabel=p["ylabel"], ylim=p.get("ylim")))

    plot_panels(panels, stems("counterexamples_main"))


if __name__ == "__main__":
    main()
