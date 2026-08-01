"""Figures for the synthetic convex problem, from the JSON ``run_gpu`` writes.

    python3 -m synthetic.plot_synthetic                      # results/synthetic/
    python3 -m synthetic.plot_synthetic --results results/synthetic_20x20  # --m 20 pass
    python3 -m synthetic.plot_synthetic --figures loss gnorm

Reads ``results/synthetic*/<method>/<stage>.json`` and skips, with a message,
any figure whose stage has not been run -- so a partial run still produces the
figures it can.

Each figure is named for the ``\\label`` it belongs to rather than for a float
number (the numbers move whenever a float is added elsewhere):

=========================  =====================  =========================
figure                     stages it needs        paper
=========================  =====================  =========================
``synthetic_main``         ``final``              ``fig:synthetic_main``
``loss``, ``GN``           ``final``              (diagnostic, not in the paper)
``diagnostics``            ``floor``,             ``fig:synthetic_dynamics``
                           ``horizon``, ``kappa``
=========================  =====================  =========================

``diagnostics`` is one row of three panels under a single legend, the layout
the counterexample and federated figures use, because the panels draw the same
ten methods. Drawn one per figure with a legend beside each, the legend was
wider than the axes it belonged to.

Output goes to ``<results>/figures/`` as both PDF and PNG. Nothing is written
into ``aaai_article/``: copy a figure over deliberately once you have looked at
it.
"""

from __future__ import annotations

import argparse
import json
import math
from matplotlib import ticker
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import Dict, Sequence

from common.plotting import (AXIS, FS_ANNOT, FS_LABEL, FS_LEGEND, INK_2, MUTED,
                             SURFACE, TEXT_WIDTH, color_of, label_of,
                             order_methods,
                             figure_legend, panel_legend, save_figure,
                             style_axes, use_paper_style)
from common.paths import results_root      # not common.utils: that pulls torch

use_paper_style()

#: Width of a half-page panel: the diagnostic figures below are authored at the
#: size a ``0.48\textwidth`` subfigure prints at. A figure drawn wider is scaled
#: down by LaTeX and takes its type with it, which is how 9 pt labels turn into
#: 5 pt ones. The paper's own figure is ``fig_trajectories``, at TEXT_WIDTH.
PANEL_WIDTH = 0.48 * TEXT_WIDTH

#: Most entries a shared legend fits across ``TEXT_WIDTH`` before it overruns the
#: figure. Set from the longest label set here (``EF21-MuonUSign`` and friends);
#: shorter labels would fit more, but the cost of being conservative is one line
#: of white space and the cost of being wrong is a silently clipped label.
MAX_LEGEND_COLS = 5

#: Height of the strip under the axes, in inches: x label and tick labels, plus
#: one term per legend row. Absolute rather than a fraction of the figure,
#: because it holds type at a fixed point size -- expressed as a fraction, the
#: legend grows with the panels and eats the height it was given them for.
XAXIS_STRIP_IN, LEGEND_ROW_IN = 0.59, 0.17

#: Figure heights. Both were authored at 2.4/2.35 inches, where a panel is
#: wider than it is tall and every curve is squeezed into a strip; these
#: leave the axes about an inch taller, which is what the log decades need
#: to be readable on screen at one-to-one.
TRAJECTORY_HEIGHT, DIAGNOSTIC_HEIGHT = 3.2, 3.0

#: The arrival inset, in axes fractions of the panel it sits in. High enough
#: that the card around it -- the inset plus a gutter of tick labels on two
#: sides -- clears the plateau band, which on both panels runs at just under
#: half height and is the one thing in the upper half worth not covering.
INSET_RECT = (0.40, 0.62, 0.585, 0.365)

#: How far the inset window reaches past what it is there to show. It is a zoom
#: on the arrival, so it stops a little above the last thing that arrives -- the
#: target on the loss panel, the highest plateau on the gradient-norm one -- and
#: a little under the lowest plateau, which is about one decade in all. The
#: first version ran to ninety times the lowest plateau and 1.35 times the last
#: crossing: two decades of headroom, most of it empty, with every crossing
#: squeezed into the bottom fifth of the inset.
INSET_HEADROOM, INSET_FLOORROOM, INSET_TAIL = 2.2, 1.5, 1.18


def _bottom_fraction(fig_height: float, legend_rows: int) -> float:
    return (XAXIS_STRIP_IN + LEGEND_ROW_IN * legend_rows) / fig_height

#: ``stage -> the figures that need it``. Used to report what a partial run is
#: missing, instead of failing on the first absent file.
FIGURES = {
    "trajectories": ("final",),
    "loss": ("final",),
    "gnorm": ("final",),
    "diagnostics": ("floor", "horizon", "kappa"),
}


def load_stage(results: Path, stage: str) -> Dict[str, dict]:
    """``{method: result}`` for one stage, in the paper's method order."""
    found = {}
    for path in sorted(results.glob(f"*/{stage}.json")):
        try:
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  ! skipping {path}: {exc}")
            continue
        found[path.parent.name] = payload
    return {m: found[m] for m in order_methods(found)}


def _problem_caption(payloads: Sequence[dict]) -> str:
    if not payloads:
        return ""
    p = next(iter(payloads))["problem"]
    return (f"{p['m']}x{p['n']}, spectrum {p['spectrum']}, "
            f"L/sigma = {p['condition_number']:.3g}, lmo_dtype {p['lmo_dtype']}")


# --------------------------------------------------------------------------
# fig:synthetic_main -- loss and gradient-norm trajectories
# --------------------------------------------------------------------------


def _draw_trajectory(ax, data: Dict[str, dict], key: str, ylabel: str) -> int:
    """One curve per method, at that method's own tuned hyperparameters."""
    drawn = 0
    for method, payload in data.items():
        hist = payload["result"].get(key)
        if not hist:
            continue
        # The curve is the geometric mean over the stage's problem instances,
        # elementwise, matching how every scalar in the tables is aggregated.
        # The plateau oscillation survives it -- averaging independent draws
        # damps its amplitude but not its level, which is what the panel is for.
        ax.plot(range(len(hist)), hist, color=color_of(method), linewidth=1.6,
                label=label_of(method), zorder=3)
        drawn += 1
    ax.set_xlabel("iteration", color=INK_2, fontsize=FS_LABEL)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=FS_LABEL)
    return drawn


def _arrival_inset(ax, data: Dict[str, dict], key: str):
    """Zoom on the window where the methods arrive, not on where they end up.

    The criterion the tables report is a crossing time, so the question the
    figure has to answer is which curve reaches the target first. Over the full
    axis that happens in the first sixth of the x range, inside a fold of curves
    a millimetre wide, and the remaining five sixths are flat. The window is cut
    to that fold: from just under the lowest plateau to just over the last thing
    that arrives, and from the first entry into it to a little past the last
    recorded crossing.
    """
    series, crossings, plateaus = {}, [], []
    for method, payload in data.items():
        rec = payload["result"]
        hist = rec.get(key)
        if not hist:
            continue
        series[method] = hist
        if method in UNNORMALIZED:
            continue
        tail = sorted(hist[int(len(hist) * 0.8):])
        plateaus.append(tail[len(tail) // 2])
        if rec.get("reached_target") and rec.get("iters_to_converge"):
            crossings.append(rec["iters_to_converge"])
    settled = [p for p in plateaus if p > 0]
    if len(settled) < 2 or not crossings:
        return None

    # The criterion is a crossing of this line, so the loss window has to hold
    # it. The gradient-norm panel has no counterpart and is bounded by the
    # plateaus alone.
    target = next(iter(data.values()))["problem"].get("target_loss")
    ceiling = max(settled)
    if key == "loss_history" and target:
        ceiling = max(ceiling, target)
    lo, hi = min(settled) / INSET_FLOORROOM, ceiling * INSET_HEADROOM
    x_hi = min(max(len(h) for h in series.values()),
               int(INSET_TAIL * max(crossings)))
    # Start where the fastest curve first drops under the top of the window, so
    # the near-vertical opening plunge is left to the main panel.
    entries = [next((t for t, y in enumerate(ys) if y <= hi), None)
               for m, ys in series.items() if m not in UNNORMALIZED]
    x_lo = max(0, min(t for t in entries if t is not None) - 5)

    axins = ax.inset_axes(list(INSET_RECT), zorder=7)
    style_axes(axins, logy=True)
    # style_axes makes a panel transparent so a neighbour's labels can sit in
    # the gutter. An inset is the opposite case: it lies on top of the curves it
    # magnifies, and has to hide them to be legible. Its spines stay in the
    # house style -- left and bottom only. The frame around the whole thing is
    # the card's, drawn in _inset_card; boxing the axes as well would put a box
    # inside a box.
    axins.patch.set_alpha(1.0)
    axins.set_facecolor(SURFACE)
    for side in ("left", "bottom"):
        axins.spines[side].set_linewidth(0.5)
    if key == "loss_history" and target and lo < target < hi:
        axins.axhline(target, color=MUTED, linewidth=0.7, linestyle=(0, (3, 2)),
                      zorder=2)
    for method, ys in series.items():
        axins.plot(range(len(ys)), ys, color=color_of(method), linewidth=1.2,
                   zorder=3)
    axins.set_ylim(lo, hi)
    axins.set_xlim(x_lo, x_hi)
    axins.xaxis.set_major_locator(ticker.MaxNLocator(4, integer=True))
    # Label the powers-of-ten subdivisions only when the window is under a
    # decade, where the decade ticks would label it once or not at all. Over a
    # wider window they are clutter: eight labels per decade on a two-inch axis.
    if hi / lo < 12.0:
        axins.yaxis.set_minor_locator(
            ticker.LogLocator(base=10.0, subs=(2, 3, 5), numticks=12))
        axins.yaxis.set_minor_formatter(
            ticker.LogFormatterSciNotation(minor_thresholds=(4.0, 0.4)))
    else:
        axins.minorticks_off()
    axins.tick_params(which="both", labelsize=FS_ANNOT - 2.0, pad=1.0,
                      length=2.0, colors=MUTED)
    indicator = ax.indicate_inset_zoom(axins, edgecolor=MUTED, linewidth=0.6,
                                       alpha=0.5)
    # Matplotlib 3.10 returns an indicator object where 3.9 returns the
    # (rectangle, connectors) pair; both spell the parts the same way.
    rect = getattr(indicator, "rectangle", None) or indicator[0]
    connectors = getattr(indicator, "connectors", None) or indicator[1]
    rect.set_zorder(6)                    # over the curves it encloses
    for line in connectors:
        # Under the card, so each connector is occluded from the card's border
        # inward and appears to terminate on it. Run over the top instead and
        # two grey lines cross the magnified plot to reach its far corners.
        line.set_zorder(4)
    return axins


def _inset_card(fig, ax, axins, pad: float = 0.016) -> None:
    """The framed card an inset and its tick labels sit on.

    Two things have to happen at once. The tick labels fall outside the inset's
    own patch, so without a backing they land on whatever the main panel drew
    underneath; and a backing painted white and left unbordered reads as a hole
    torn in the panel, because the grid runs into it and stops. Bordering it
    turns the same rectangle into a card laid over the panel, which is what a
    reader should see. Sized from the drawn extent -- the only thing that knows
    how wide a 5.5 pt ``10^-3`` is -- hence a figure already laid out.
    """
    fig.canvas.draw()
    bbox = (axins.get_tightbbox(fig.canvas.get_renderer())
            .transformed(ax.transAxes.inverted()))
    ax.add_patch(Rectangle((bbox.x0 - pad, bbox.y0 - pad),
                           bbox.width + 2 * pad, bbox.height + 2 * pad,
                           transform=ax.transAxes, facecolor=SURFACE,
                           edgecolor=AXIS, linewidth=0.5, zorder=5,
                           clip_on=False))


def fig_trajectory(plt, data: Dict[str, dict], key: str, ylabel: str):
    """A single trajectory panel -- the diagnostic form, one metric at a time."""
    fig, ax = plt.subplots(figsize=(PANEL_WIDTH, 2.4))
    style_axes(ax, logy=True)
    if not _draw_trajectory(ax, data, key, ylabel):
        plt.close(fig)
        return None
    ax.set_title(_problem_caption(list(data.values())), color=MUTED,
                 fontsize=FS_ANNOT - 1.5, loc="left", pad=6)
    # Inside, not in a gutter: every curve here decays, so the upper right is
    # empty, and a gutter wide enough for "EF21-MuonUSign" would take a third of
    # a 3.4-inch panel.
    panel_legend(ax, "upper right")
    fig.subplots_adjust(left=0.155, right=0.985, top=0.90, bottom=0.185)
    return fig


def fig_trajectories(plt, data: Dict[str, dict]):
    """The paper's figure: loss and gradient norm side by side, one legend.

    Two panels rather than two separate subfigures, for the same reason the CIFAR
    and counterexample figures are laid out this way: the panels draw the same
    methods, so a shared legend under both says once what two legends would say
    twice -- and at a two-column width neither panel has to give up a third of
    its area to hold it.

    The legend wraps past ``MAX_LEGEND_COLS``. In one row, ten entries of these
    lengths overrun ``TEXT_WIDTH`` at both ends, and matplotlib centres the
    overrun rather than reporting it: the outermost labels are simply clipped
    off the page, which in the first version of this figure cost SignMuon its
    name and Adam its entry.
    """
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, TRAJECTORY_HEIGHT),
                             squeeze=False)
    specs = [("loss_history", "$F(X_t)$"),
             ("grad_norm_history", r"$\|\nabla F(X_t)\|_F$")]
    insets = []
    for ax, (key, ylabel) in zip(axes[0], specs):
        style_axes(ax, logy=True)
        if not _draw_trajectory(ax, data, key, ylabel):
            plt.close(fig)
            return None
        insets.append((ax, _arrival_inset(ax, data, key)))
    handles, labels = axes[0][0].get_legend_handles_labels()
    rows = 1 + (len(labels) - 1) // MAX_LEGEND_COLS
    ncol = -(-len(labels) // rows)          # balance the rows, don't fill-then-spill
    figure_legend(fig, handles, labels, ncol=ncol)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.965, wspace=0.20,
                        bottom=_bottom_fraction(TRAJECTORY_HEIGHT, rows))
    # After the layout is final: the card is sized from where the inset's tick
    # labels actually landed, and subplots_adjust moves them.
    for ax, axins in insets:
        if axins is not None:
            _inset_card(fig, ax, axins)
    return fig


# --------------------------------------------------------------------------
# fig:synthetic_dynamics (left) -- the accuracy floor of a constant step
# --------------------------------------------------------------------------


#: The two methods whose step is not norm-fixed. They are references here, not
#: subjects: on a quadratic with an exactly known Hessian a scaled gradient step
#: converges to machine precision, so they run far below the eight methods the
#: panels are about -- nineteen decades below at the widest, in the conditioning
#: panel at kappa = 10. Left to set the y-range, they compress those eight into
#: the top inch of the axes and force tick labels (``10^-19``) wide enough to
#: collide with the neighbouring panel.
UNNORMALIZED = ("sgd", "adam")


def _focus_ylim(ax, series: Dict[str, list], pad: float = 0.6) -> None:
    """Scale the y-axis to the norm-fixed methods, letting the references clip.

    A curve that leaves the frame is named at the edge it left by, in the muted
    annotation style. Silently clipping it would put a method in the shared
    legend that appears in no panel, which reads as a missing run rather than a
    deliberate choice of range.
    """
    focus = [y for method, ys in series.items() if method not in UNNORMALIZED
             for y in ys if y is not None and y > 0]
    if len(focus) < 2:
        return
    lo, hi = min(focus), max(focus)
    decades = math.log10(hi / lo)
    lo, hi = lo * 10 ** (-pad), hi * 10 ** (pad if decades > 0.5 else 1.0)
    ax.set_ylim(lo, hi)

    gone = [label_of(m) for m in UNNORMALIZED
            if series.get(m) and all(y < lo for y in series[m] if y and y > 0)]
    if gone:
        ax.annotate(f"{', '.join(gone)} below", xy=(0.5, 0.012),
                    xycoords="axes fraction", ha="center", va="bottom",
                    color=MUTED, fontsize=FS_ANNOT - 1.0)


def _curve(ax, method: str, xs, ys) -> None:
    """One method's series, labelled with its name and nothing else.

    The fitted exponents used to ride in the label. With ten methods that made
    the legend wider than the axes it belonged to, and the panels were squeezed
    into the strip left over. Every exponent is a column of the table these
    panels illustrate, so the label carries the name alone and the legend goes
    under the row, as in every other figure in the paper.
    """
    ax.plot(xs, ys, color=color_of(method), linewidth=1.5, marker="o",
            markersize=3.2, label=label_of(method), zorder=3)


def _panel_floor(ax, data: Dict[str, dict]) -> int:
    """``g_infty`` against ``eta``: the accuracy floor of a constant step.

    A normalized step of fixed length cannot converge at a constant rate; the
    gradient plateaus at a level linear in ``eta``, so slope 1 is the
    prediction. SGD has no floor and is absent by construction.
    """
    drawn = 0
    for method, payload in data.items():
        rows = [r for r in payload["result"]["rows"] if r.get("settled")]
        if len(rows) < 2:
            continue
        _curve(ax, method, [r["lr"] for r in rows], [r["g_inf"] for r in rows])
        drawn += 1
    ax.set_xlabel(r"step size $\eta$", color=INK_2, fontsize=FS_LABEL)
    # The infinity in ``g_inf`` is the settling, not a norm: the quantity is the
    # plateau of the *Frobenius* norm, which is how the appendix defines
    # g_infty and how Table 7 heads its column. The label used to read
    # ``floor ||grad F||_inf``, where the subscript reads as the max norm.
    ax.set_ylabel(r"$g_\infty$", color=INK_2, fontsize=FS_LABEL)
    return drawn


def _panel_horizon(ax, data: Dict[str, dict]) -> int:
    """Error against budget ``T``, with ``(eta, mu, schedule)`` retuned at each.

    Plots the quantity the theorems bound, ``min_t ||grad F(X_t)||_*^2`` in the
    norm dual to each method's LMO ball, so the exponent fitted on the curve is
    the ``p`` in the table. Retuning per budget is the point: imposing one
    schedule on every budget measures the schedule, not the method.
    """
    drawn, series, dual = 0, {}, True
    for method, payload in data.items():
        rec = payload["result"]
        rows = rec.get("rows") or []
        if len(rows) < 2:
            continue
        # Falls back to the Frobenius norm for a results tree written before the
        # dual norm was recorded.
        dual = all(r.get("best_dual") is not None for r in rows)
        ys = ([r["best_dual"] ** 2 for r in rows] if dual
              else [r["best_gnorm"] for r in rows])
        _curve(ax, method, [r["T"] for r in rows], ys)
        series[method] = ys
        drawn += 1
    _focus_ylim(ax, series)
    ax.set_xlabel("horizon $T$", color=INK_2, fontsize=FS_LABEL)
    ax.set_ylabel(r"$\min_{t\leq T}\|\nabla F(X_t)\|_*^2$" if dual
                  else r"$\min_{t\leq T}\|\nabla F(X_t)\|_F$",
                  color=INK_2, fontsize=FS_LABEL)
    return drawn


def _panel_kappa(ax, data: Dict[str, dict]) -> int:
    """Best gradient norm within the budget against ``L/sigma``, tuned at each."""
    drawn, series = 0, {}
    for method, payload in data.items():
        rows = payload["result"].get("rows") or []
        if len(rows) < 2:
            continue
        ys = [r["best_gnorm"] for r in rows]
        _curve(ax, method, [r["kappa"] for r in rows], ys)
        series[method] = ys
        drawn += 1
    _focus_ylim(ax, series)
    ax.set_xlabel(r"condition number $\kappa = L/\sigma$", color=INK_2,
                  fontsize=FS_LABEL)
    ax.set_ylabel(r"$\min_{t\leq T}\|\nabla F(X_t)\|_F$", color=INK_2,
                  fontsize=FS_LABEL)
    return drawn


#: ``stage -> (panel drawer, subcaption)``, in printed order.
DIAGNOSTIC_PANELS = (
    ("floor", _panel_floor, "floor"),
    ("horizon", _panel_horizon, "budget"),
    ("kappa", _panel_kappa, "conditioning"),
)


def fig_diagnostics(plt, staged: Dict[str, Dict[str, dict]]):
    """The three log-log diagnostics as one row, under a single legend.

    Laid out like the counterexample and federated figures: full text width, one
    legend beneath the row because the panels draw the same ten methods, and no
    per-panel legend at all. A stage that has not been run drops its panel
    rather than the whole figure.
    """
    present = [(stage, draw) for stage, draw, _ in DIAGNOSTIC_PANELS
               if staged.get(stage)]
    if not present:
        return None
    fig, axes = plt.subplots(1, len(present),
                             figsize=(TEXT_WIDTH, DIAGNOSTIC_HEIGHT),
                             squeeze=False)
    for ax, (stage, draw) in zip(axes[0], present):
        style_axes(ax, logx=True, logy=True)
        if not draw(ax, staged[stage]):
            plt.close(fig)
            return None
    # Across every panel, not just the first: SGD has no floor and so never
    # appears in the left one, and a legend taken from there alone would omit a
    # method the other two panels draw.
    handles, labels = {}, []
    for ax in axes[0]:
        for handle, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in handles:
                handles[lab], _ = handle, labels.append(lab)
    rows = 1 + (len(labels) - 1) // MAX_LEGEND_COLS
    figure_legend(fig, [handles[lab] for lab in labels], labels,
                  ncol=-(-len(labels) // rows), fontsize=FS_LEGEND)
    # A wide gutter because each panel carries its own y tick labels and its
    # own y label; at the default they land on the neighbour's axes.
    fig.subplots_adjust(left=0.075, right=0.995, top=0.975, wspace=0.42,
                        bottom=_bottom_fraction(DIAGNOSTIC_HEIGHT, rows))
    return fig


# --------------------------------------------------------------------------


def get_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=None,
                   help="Directory holding <method>/<stage>.json "
                        "(default: results/synthetic/). A `run_gpu --m 20` "
                        "small-size pass writes results/synthetic_20x20/, so "
                        "point it there to plot one.")
    p.add_argument("--out", default=None,
                   help="Where to write the figures (default: <results>/figures/)")
    p.add_argument("--figures", nargs="+", default=sorted(FIGURES),
                   choices=sorted(FIGURES), metavar="NAME",
                   help=f"default: all of {sorted(FIGURES)}")
    p.add_argument("--formats", nargs="+", default=["pdf", "png"],
                   help="default: pdf png")
    return p.parse_args()


def main() -> int:
    args = get_args()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required: pip install -r requirements.txt")
        return 1

    results = Path(args.results) if args.results else results_root() / "synthetic"
    if not results.is_dir():
        print(f"No results at {results.resolve()}.\n"
              f"Run `python3 -m synthetic.run_gpu` first (or `--m 20` for a "
              f"small pass, whose output lands in results/synthetic_20x20/).")
        return 1
    out = Path(args.out) if args.out else results / "figures"

    builders = {
        "trajectories": lambda s: fig_trajectories(plt, s["final"]),
        "loss": lambda s: fig_trajectory(plt, s["final"], "loss_history",
                                         "$F(X_t)$"),
        "gnorm": lambda s: fig_trajectory(plt, s["final"], "grad_norm_history",
                                          r"$\|\nabla F(X_t)\|_F$"),
        "diagnostics": lambda s: fig_diagnostics(plt, s),
    }
    stems = {"trajectories": "synthetic_main", "loss": "loss", "gnorm": "GN",
             "diagnostics": "diagnostics"}

    cache: Dict[str, Dict[str, dict]] = {}
    written, skipped = [], []
    for name in args.figures:
        stages = FIGURES[name]
        for stage in stages:
            if stage not in cache:
                cache[stage] = load_stage(results, stage)
        staged = {stage: cache[stage] for stage in stages}
        missing = [stage for stage in stages if not staged[stage]]
        if len(missing) == len(stages):
            skipped.append(
                f"{name}: no output for {' or '.join(missing)} under "
                f"{results.name}/ -- run "
                f"`python3 -m synthetic.run_gpu --stages {' '.join(missing)}`")
            continue
        if missing:
            # A multi-panel figure drops the panels it cannot draw rather than
            # the whole figure, but says which, so a partial run is never
            # mistaken for a complete one.
            skipped.append(f"{name}: drawn without the "
                           f"{', '.join(missing)} panel(s), which have no output")
        fig = builders[name](staged)
        if fig is None:
            note = (" (--mode final keeps trajectories only for a single problem "
                    "instance; re-run with --save-histories)"
                    if "final" in stages else " (too few usable points to plot)")
            skipped.append(f"{name}: ran but has nothing to draw{note}")
            continue
        # Every figure here is laid out at the width it prints at, so none of
        # them may be re-scaled by a tight bounding box.
        written += save_figure(fig, out, stems[name], formats=args.formats,
                               tight=False)
        plt.close(fig)

    for line in skipped:
        print(f"  ~ {line}")
    if not written:
        print("Nothing plotted.")
        return 1
    print(f"\nWrote {len(written)} file(s) to {out.resolve()}:")
    for path in written:
        print(f"    {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
