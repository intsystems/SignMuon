"""The federated figure as the article prints it: two panels, one shared legend.

    python3 -m federated.plot_article --bundle results/federated_results.zip
    python3 -m federated.plot_article --root results/federated --n-parties 11

``--bundle`` is the normal path: it takes the archive ``federated.export_article``
writes, unpacks it if it is still a ``.zip``, and draws from the ``metrics.json``
tree inside. ``--root`` reads a live results tree on the machine that trained it.

``plot_federated.py`` is the exploratory plotter -- one file per metric, legend
parked in the gutter, any metric you ask for. That layout does not survive
contact with the page: at ``0.48\\textwidth`` each panel is about 3.4 inches wide
and an eleven-entry legend beside the axes takes nearly half of it, leaving the
curves in a strip too narrow to read.

This writes the article figure instead: both panels in one full-width file, a
single legend underneath spanning both, and markers as a second channel so the
eleven methods stay distinguishable in grayscale and to a reader with a colour
vision deficiency. The output is one PDF, included with ``\\includegraphics``
rather than as two subfigures.

Deliberate departures from the exploratory plotter
--------------------------------------------------
* **Linear loss axis, clipped to the tail.** The log axis compressed the region
  where the methods actually differ into the top decade. Round 0 (loss 2.3) is
  off-scale by construction: it is the same value for every method and spending
  a third of the axis on it is what flattened everything else.
* **Markers, spaced apart per method** so that two curves within a line width of
  each other can still be told apart -- the same device Figure 1 uses.
* **Curves named at their own ends inside the inset**, in their own colour. Ten
  curves within a fifth of a nat is past what hue can separate, and the legend is
  three inches away at the foot of the figure; the label column also reads as the
  ranking the table reports. The inset window covers the crowded part of the tail
  only, so a method that has already separated does not cost resolution to the
  ones that have not.
* **No per-curve seed count in the legend.** The panel title reports it once,
  read off the runs rather than hard-coded, so a figure drawn from a partial sweep
  says how partial it is instead of claiming the paper's five.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from matplotlib.ticker import MaxNLocator

from aggregate import aggregate_group, group_key, load_runs
from common.paths import results_root
from common.plotting import (FS_ANNOT, FS_LEGEND, MUTED, SURFACE, TEXT_WIDTH,
                             color_of, label_of, marker_of, order_methods,
                             save_figure, style_axes, use_paper_style)

use_paper_style()

#: (metric, axis label, clip the y-axis to the tail?)
PANELS = [
    ("test_loss", "test cross-entropy", True),
    ("test_acc", "test accuracy (%)", True),
]

#: Markers are drawn every ``MARK_EVERY`` points, offset per method so that they
#: do not stack into a vertical line at the same x.
MARK_EVERY = 4

#: Denser inside the inset, which shows the tail alone: at the outer spacing a
#: curve can cross the whole magnified window carrying one marker or none, which
#: leaves hue as the only channel exactly where hue has stopped working.
MARK_EVERY_INSET = 3

#: Fraction of the inset's width kept clear on the right for the curve labels.
LABEL_BAND = 0.42

#: Labels are set beside the curve, where "Muon (server LMO)" would eat a third of
#: the magnified window. Only the names that do not fit are shortened.
SHORT_LABEL = {"muonserver": "Muon (srv)"}

#: Line style carries the method's *role*, so that hue is not the only channel
#: separating eleven curves. Solid is a one-bit method -- what the paper is about;
#: dashed is an uncompressed reference; dash-dot is the compressed baseline the
#: proposed methods are meant to beat. A reader can then find the comparison
#: without reading the legend at all.
REFERENCES = {"muon", "muonserver", "sgd", "adam"}
BASELINES = {"signsgd"}


def _style(alg: str) -> dict:
    if alg in REFERENCES:
        return {"linestyle": (0, (5, 2)), "linewidth": 1.2, "marker": None}
    if alg in BASELINES:
        return {"linestyle": (0, (6, 1.6, 1, 1.6)), "linewidth": 1.4}
    return {"linestyle": "-", "linewidth": 1.7}


def _tail_window(data, metric, frac: float = 0.45):
    """``(x0, x1, y0, y1, shown)`` for an inset over the converged tail.

    The tail is where every claim in the table lives and where the curves are
    within a line width of each other, so it is the only part worth magnifying.
    Bounds come from the data rather than being hard-coded, so the inset survives
    a change of horizon.

    The window covers the *crowded* part of the tail and not all of it. A method
    that has already separated in the main panel gains nothing from magnification
    and costs the ones that have not: on CNN2 Adam finishes a tenth of a nat clear
    of the nearest other method, and stretching the window to reach it halves the
    resolution left for the ten that finish inside $0.11$ of one another. The pack
    is the longest run of adjacent finishers with no gap wider than a quarter of
    the total spread; whatever falls outside it keeps its place in the main panel
    and simply does not enter the inset. ``shown`` names what did.
    """
    series = {}
    for alg, runs in data.items():
        agg = aggregate_group(runs, metric)
        if not agg or not agg["steps"]:
            continue
        cut = max(agg["steps"]) * (1.0 - frac)
        pts = [(s, m) for s, m in zip(agg["steps"], agg["mean"]) if s >= cut]
        if pts:
            series[alg] = pts
    if not series:
        return None

    finals = sorted((pts[-1][1], alg) for alg, pts in series.items())
    spread = finals[-1][0] - finals[0][0]
    limit = 0.25 * spread if spread > 0 else float("inf")
    best, run = [finals[0]], [finals[0]]
    for prev, cur in zip(finals, finals[1:]):
        run = run + [cur] if cur[0] - prev[0] <= limit else [cur]
        if len(run) > len(best):
            best = run
    shown = [alg for _, alg in best]

    xs = [s for alg in shown for s, _ in series[alg]]
    ys = [m for alg in shown for _, m in series[alg]]
    pad = 0.10 * (max(ys) - min(ys)) or 0.01
    return min(xs), max(xs), min(ys) - pad, max(ys) + pad, shown


def collect(root: Path, n_parties: int, rounds: int) -> Dict[str, List[dict]]:
    """``{algorithm: runs}``, one configuration per method.

    A bundle carries the ablations alongside the reported table: SGD and Adam
    under the sign-family rule (``scale_baselines``), the sign family re-tuned
    under the competing per-layer rules, and the weight-decay arm. Each shares an
    ``algorithm`` with a reported run, so grouping on the algorithm alone merges
    them and then leaves a size tie-break to choose. That is not hypothetical:
    Adam's two arms carry five seeds each, and the tie was being settled by which
    directory name sorted first, which drew the scaled arm under Adam's label.
    Each ablation is therefore excluded by the config field that makes it one,
    and the tie-break below is the safety net it was meant to be.
    """
    runs = load_runs([root])
    kept: Dict[str, List[dict]] = {}
    for run in runs:
        cfg = run["config"]
        if n_parties and cfg.get("n_parties") != n_parties:
            continue
        if rounds and cfg.get("rounds") != rounds:
            continue
        if cfg.get("split") != "full":          # tuning runs never enter a figure
            continue
        if cfg.get("scale_baselines"):          # the SGD/Adam per-layer-rule arm
            continue
        if str(cfg.get("lr_scaling") or "unit-gain") != "unit-gain":
            continue                            # the sign-family rule ablation
        if cfg.get("weight_decay"):             # the weight-decay ablation
            continue
        alg = cfg.get("algorithm") or cfg.get("optimizer")
        if alg:
            kept.setdefault(alg, []).append(run)

    resolved = {}
    for alg, group in kept.items():
        by_cfg: Dict[tuple, List[dict]] = {}
        for run in group:
            by_cfg.setdefault(group_key(run["config"]), []).append(run)
        best = max(by_cfg.values(), key=len)
        if len(by_cfg) > 1:
            dropped = sum(len(v) for v in by_cfg.values()) - len(best)
            print(f"  ~ {alg}: {len(by_cfg)} configurations, keeping the "
                  f"{len(best)}-seed one and ignoring {dropped} run(s)")
            if sum(len(v) == len(best) for v in by_cfg.values()) > 1:
                print(f"  ! {alg}: the kept configuration is not determined by "
                      f"seed count; check which one the figure drew")
        resolved[alg] = best
    return {a: resolved[a] for a in order_methods(resolved)}


def _draw_series(ax, data, metric, *, bands: bool, collect_handles: bool,
                 only=None, dense_marks: bool = False):
    """Draw every method onto ``ax``; return (handles, labels) if asked.

    ``only`` restricts the draw to a subset, in the paper's order, so the inset
    can carry the methods its window was built around and no others.
    """
    algs = [a for a in data if only is None or a in only]
    handles, labels = [], []
    for i, alg in enumerate(algs):
        agg = aggregate_group(data[alg], metric)
        if agg is None:
            continue
        style, color = _style(alg), color_of(alg)
        mark = style.pop("marker", marker_of(alg))
        every = ((i % MARK_EVERY_INSET, MARK_EVERY_INSET) if dense_marks
                 else (i % MARK_EVERY + 1, MARK_EVERY + 2))
        line, = ax.plot(agg["steps"], agg["mean"], color=color, zorder=3,
                        marker=mark or None, markersize=3.4, markevery=every,
                        markeredgewidth=0.0, **style)
        if bands:
            lo = [m - s for m, s in zip(agg["mean"], agg["std"])]
            hi = [m + s for m, s in zip(agg["mean"], agg["std"])]
            ax.fill_between(agg["steps"], lo, hi, color=color, alpha=0.13,
                            linewidth=0, zorder=1)
        if collect_handles:
            handles.append(line)
            labels.append(label_of(alg))
    return handles, labels


def _label_ends(ax, data, metric, shown, x_end, fontsize):
    """Name each curve at its own end, in its own colour, inside the inset.

    Ten curves within a fifth of a nat of one another is past what hue can
    separate, and the figure legend sits three inches away at the foot of the
    page: reading the inset meant matching a colour across that distance, twice
    per curve. Labelling in place removes the lookup, and the label column doubles
    as the ranking the table reports.

    Labels are placed at each curve's own height and then pushed apart to a
    minimum spacing, bottom upwards and then top downwards so that the column
    stays inside the axes; a leader is drawn for any that had to move far enough
    to be ambiguous.
    """
    y0, y1 = ax.get_ylim()
    ends = []
    for alg in shown:
        agg = aggregate_group(data[alg], metric)
        if agg and agg["mean"]:
            ends.append((agg["mean"][-1], alg))
    if not ends:
        return
    ends.sort()

    # The minimum spacing is a type size, so it has to be measured against the
    # axes as *printed*. ``get_window_extent`` is in pixels and the font size in
    # points; conflating the two silently under-spaces the column by dpi/72.
    px_per_pt = ax.get_figure().dpi / 72.0
    axis_px = max(ax.get_window_extent().height, 1e-9)
    gap = (y1 - y0) * (fontsize * px_per_pt * 1.22) / axis_px
    ys = [y for y, _ in ends]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    ceiling = y1 - 0.8 * gap                      # clear of the top spine
    if ys[-1] > ceiling:
        ys[-1] = ceiling
        for i in range(len(ys) - 2, -1, -1):
            ys[i] = min(ys[i], ys[i + 1] - gap)

    pad = (ax.get_xlim()[1] - x_end) * 0.14
    for (y_true, alg), y in zip(ends, ys):
        color = color_of(alg)
        if abs(y - y_true) > 0.35 * gap:
            ax.plot([x_end, x_end + 0.8 * pad], [y_true, y], color=color,
                    linewidth=0.5, alpha=0.75, zorder=4, solid_capstyle="butt")
        ax.text(x_end + pad, y, SHORT_LABEL.get(alg, label_of(alg)),
                color=color, fontsize=fontsize, va="center", ha="left",
                zorder=7, clip_on=False)


def _band_note(data: Dict[str, List[dict]]) -> str:
    """What the band means, read off the runs rather than assumed.

    The paper's figure is five seeds per method, but a figure drawn mid-sweep is
    not, and a hard-coded "over 5 seeds" would say otherwise. A single seed carries
    no dispersion at all, so it is named as such instead of being given a band's
    worth of authority.
    """
    counts = sorted({len(runs) for runs in data.values()})
    if counts == [1]:
        return "single seed: no dispersion measured"
    span = f"{counts[0]}" if len(counts) == 1 else f"{counts[0]}--{counts[-1]}"
    return f"band is $\\pm1$ s.d. over {span} seeds"


def draw(plt, data: Dict[str, List[dict]], out: Path, formats: List[str]) -> List[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 3.45))
    handles, labels = [], []
    insets = []

    for panel, (metric, ylabel, clip) in zip(axes, PANELS):
        style_axes(panel)
        h, l = _draw_series(panel, data, metric, bands=True,
                            collect_handles=metric == PANELS[0][0])
        if h:
            handles, labels = h, l
        panel.set_xlabel("communication round")
        panel.set_ylabel(ylabel)

        # The tail carries every number in the table and is where the curves sit
        # within a line width of each other, so it gets magnified rather than
        # described. Bands are omitted inside the inset: at this zoom they overlap
        # into a single wash and hide the very lines the inset exists to separate.
        win = _tail_window(data, metric)
        if win:
            x0, x1, y0, y1, shown = win
            # Park the inset in whichever corner the curves have vacated: a
            # decreasing loss empties the top right, a rising accuracy the bottom.
            loc = [0.31, 0.44, 0.67, 0.53] if metric.endswith("loss") \
                else [0.31, 0.04, 0.67, 0.53]
            inset = panel.inset_axes(loc, zorder=6)
            style_axes(inset)
            _draw_series(inset, data, metric, bands=False, collect_handles=False,
                         only=shown, dense_marks=True)
            # The right of the window is the label column, not data: the curves
            # still end at x1, and the axis is widened past it to hold the names.
            inset.set_xlim(x0, x1 + (x1 - x0) * LABEL_BAND / (1.0 - LABEL_BAND))
            inset.set_ylim(y0, y1)
            # style_axes leaves the patch transparent, which is right for a full
            # panel and wrong for one floating over live curves.
            inset.set_facecolor(SURFACE)
            inset.patch.set_alpha(1.0)
            inset.tick_params(labelsize=FS_ANNOT - 2.0, length=2, pad=1.5)
            inset.set_xlabel("")
            inset.set_ylabel("")
            # Ticks belong under the curves, so they are taken over the data range
            # and not over the widened axis, which would put one in the labels.
            ticks = MaxNLocator(nbins=3, integer=True).tick_values(x0, x1)
            inset.set_xticks([t for t in ticks if x0 <= t <= x1])
            inset.yaxis.set_major_locator(MaxNLocator(nbins=4))
            for spine in inset.spines.values():
                spine.set_visible(True)
                spine.set_color(MUTED)
                spine.set_linewidth(0.6)
            # Not ``indicate_inset_zoom``: it reads the rectangle off the inset's
            # limits, which now run past the data into the label column, and would
            # mark a window extending beyond the end of training.
            panel.indicate_inset((x0, y0, x1 - x0, y1 - y0), inset_ax=inset,
                                 edgecolor=MUTED, linewidth=0.6, alpha=0.5)
            insets.append((inset, metric, shown, x1))

        if clip:
            # Round 0 is the untrained model -- identical for every method, and
            # three times the final loss. Including it spends a third of the axis
            # on a point that carries no comparison and flattens the rest. The
            # range is taken over the BANDS, not the means, or the clip cuts the
            # very uncertainty it is there to show.
            lows, highs = [], []
            for runs in data.values():
                agg = aggregate_group(runs, metric)
                if not agg:
                    continue
                for s, m, sd in zip(agg["steps"], agg["mean"], agg["std"]):
                    if s > 0:
                        lows.append(m - sd)
                        highs.append(m + sd)
            if lows:
                lo_t, hi_t = min(lows), max(highs)
                pad = 0.06 * (hi_t - lo_t)
                panel.set_ylim(lo_t - pad, hi_t + pad)

    axes[0].set_title(_band_note(data), color=MUTED,
                      fontsize=FS_ANNOT - 1.0, loc="left", pad=5)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False,
               fontsize=FS_LEGEND, bbox_to_anchor=(0.5, -0.01),
               handlelength=1.9, columnspacing=1.3, handletextpad=0.5)

    # After the layout: the label spacing is set from the inset's printed height,
    # which tight_layout is still free to change up to this point.
    fig.canvas.draw()
    for inset, metric, shown, x_end in insets:
        _label_ends(inset, data, metric, shown, x_end, FS_ANNOT - 2.0)
    return save_figure(fig, out, "fig_federated_main", formats=formats)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", default=None,
                   help="Archive from 'federated.export_article' -- the .zip itself "
                        "or the directory it unpacks to. Takes precedence over --root")
    # Resolved through `common.paths`, so a run redirected with SIGNMUON_RESULTS
    # is plotted from where it actually wrote.
    p.add_argument("--root", default=str(results_root() / "federated"))
    p.add_argument("--out", default=None,
                   help="default: <root>/figures/, or <bundle>/figures/")
    p.add_argument("--n-parties", type=int, default=11)
    p.add_argument("--rounds", type=int, default=2000)
    p.add_argument("--formats", nargs="+", default=["pdf", "png"])
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.bundle:
        from federated.export_article import open_bundle, runs_root
        bundle = open_bundle(Path(args.bundle))
        root = runs_root(bundle)
        print(f"Bundle {bundle.resolve()}")
        if args.out is None:
            args.out = str(bundle / "figures")
    else:
        root = Path(args.root)
    if not root.is_dir():
        print(f"No runs at {root.resolve()}.")
        return 1
    data = collect(root, args.n_parties, args.rounds)
    if not data:
        print(f"Nothing to plot under {root.resolve()} after filtering.")
        return 1
    print(f"Plotting {len(data)} method(s): {', '.join(data)}")
    written = draw(plt, data, Path(args.out) if args.out else root / "figures",
                   args.formats)
    for path in written:
        print(f"    {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
