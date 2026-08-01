"""Paper figures for the NanoGPT experiment, in the article's plotting style.

``plot_runs.py`` is the exploratory tool (titles, every method/axis combination,
diagnostics of the port). This script emits only the figures that go into the
paper, styled to match the CIFAR/federated/synthetic figures: the same rcParams,
the same method colours and no titles (the caption carries the description), all
from ``common.plotting``.

    python parse_logs.py
    python plot_article.py     # -> ../results/nanogpt/figures/fig_nanogpt_*.pdf

Three figures:

``fig_nanogpt_main``   (main text, 1 single-column panel)
    validation loss vs optimizer step for all eight methods, with a zoomed inset
    on the tail. EF21-MuonSign is drawn at the broadcast model W; its exact server
    model X is a separate story and lives in ``fig_nanogpt_diag``.

``fig_nanogpt_appendix`` (appendix, 2 panels)
    (a) training loss vs step;  (b) validation loss vs wall-clock.

``fig_nanogpt_diag``   (appendix, 1 panel)
    the downlink measurement: the contraction the scaled sign actually achieves
    and the resulting server/broadcast gap, per layer type, on a log axis.

Colour identity is kept across the paper: every method takes its colour from
``common.plotting.color_of``, the same map the CIFAR, synthetic and federated
figures read, so a reader tracks a method by colour from one figure to the next.

These three are the only figures in the paper authored LARGER than they print --
the insets and the eight-way legends need the room, and re-laying them out at
3.3 inches would cost more than it buys. ``SCALE`` is the ratio, and
``use_paper_style(scale=SCALE)`` sizes every glyph and rule so that after LaTeX
reduces the figure the printed result matches the rest of the paper.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# This script lives one level below the package root but is documented to run
# from its own directory (`cd code/nanogpt && python plot_article.py`), which
# leaves `code/` off sys.path and `common.plotting` unimportable. Put it there.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from common.plotting import (COLUMN_WIDTH, FS_LABEL, FS_LEGEND, INK_2,
                             SERIES, SURFACE, TEXT_WIDTH, color_of,
                             use_paper_style)

# --------------------------------------------------------------------------
# The article's style, from common.plotting
# --------------------------------------------------------------------------
# Authored oversize and reduced by LaTeX: the main panel prints at \columnwidth,
# so SCALE is how much bigger the canvas is than the page. Every point size and
# rule width below is multiplied by it, which is what makes the printed result
# match the figures that are authored 1:1.
SCALE = 7.2 / COLUMN_WIDTH
use_paper_style(scale=SCALE)

#: Inset tick labels, in printed points. One point under the ordinary tick size
#: (FS_TICK = 8): an inset is read after the parent axes, so it may be smaller,
#: but not so small that the template's legibility floor is breached.
FS_INSET_TICK = 7.0

#: The canonical run per method. Explicit rather than "latest matching log":
#: ``logs/`` also holds the pre-fix EF21 runs, and a paper figure must not depend
#: on directory order. The five non-EF21 arms are unaffected by that fix (their
#: update rule contains no compressor scale) and are the original runs.
# Several methods land within 0.005 nats of each other, i.e. well inside a line
# width, so colour alone does not separate them: with a shared dash pattern the
# last-drawn curve simply hides the one beneath it. Every series therefore gets a
# DISTINCT dash pattern (so an overlapped curve shows through the gaps of the one
# on top) and a distinct marker. Colours still match the federated figures.
# ``phase`` staggers the marker POSITIONS: the methods bunch into two tight
# groups, and drawing every marker at the same x would just stack them. Phases are
# assigned so that no two members of the same group ever mark the same point --
# {Muon, SignMuon, EF21-SignMuon, MuonUSign} = 0,1,2,3 and {MuonSign,
# EF21-MuonUSign, EF21-MuonSign} = 0,1,2 -- which is what lets a reader follow an
# individual curve through a bundle it cannot be separated from.
RUNS = [
    # label,           run_id,                           colour,   dash,                marker, phase
    ("Muon",           "Muon_lr0.06_5db64adc",           color_of("Muon"), "-",                 "o", 0),
    ("SignSGD",        "SignSGD_lr0.03_1f0db2d4",        color_of("SignSGD"), "-",                 "s", 4),
    ("SignMuon",       "SignMuon_lr0.03_19f64fe1",       color_of("SignMuon"), "-",                 "^", 1),
    ("MuonUSign",      "MuonUSign_lr0.06_d9721bde",      color_of("MuonUSign"), (0, (9, 2)),         "v", 3),
    ("MuonSign",       "MuonSign_lr0.03_8ae069a3",       color_of("MuonSign"), "-",                 "D", 0),
    ("EF21-SignMuon",  "EF21-SignMuon_lr0.06_bb803ec4",  color_of("EF21-SignMuon"), (0, (6, 2.5)),       "P", 2),
    ("EF21-MuonUSign", "EF21-MuonUSign_lr0.06_2717df49", color_of("EF21-MuonUSign"), (0, (1.7, 1.7)),     "X", 1),
    ("EF21-MuonSign",  "EF21-MuonSign_lr0.06_e6770317",  color_of("EF21-MuonSign"), (0, (7, 2, 1.6, 2)), "*", 2),
]


def _legend_handles():
    """Legend proxies carrying colour, dash pattern AND marker.

    Markers are drawn only inside the insets -- on the parent axes, where the
    curves are noisy and dense, they add clutter without adding information. The
    legend is therefore built from explicit proxies rather than from the parent's
    own artists, so that the marker shown here is the one a reader will hunt for
    in the inset.
    """
    handles = [Line2D([], [], color=c, ls=ls, lw=2.8, marker=mk, markersize=8,
                      markeredgecolor=SURFACE, markeredgewidth=0.9)
               for _, _, c, ls, mk, _ in RUNS]
    labels = [lbl + (r" ($\mathbf{W}$)" if lbl in _USE_W else "")
              for lbl, *_ in RUNS]
    return handles, labels


def _mev(phase, n, tail=False):
    """``markevery`` for one series, staggered by ``phase``.

    Validation is logged only every 250 steps, so those series have ~10 points and
    the inset window holds just three: there we give each series a SINGLE marker at
    a distinct one of them (``[index]`` form). Per-step series (training loss) have
    thousands of points and take the usual ``(offset, stride)`` form.
    """
    if n > 100:                                  # per-step series
        return (phase * (9 if tail else 33), 55 if tail else 265)
    if tail:                                     # 3 visible points, 8 series
        return [max(0, n - 3 + phase % 3)]
    return (phase % 5, 5)


#: which methods carry a separate broadcast model (plot W, not X, in panel (a))
_USE_W = {"EF21-MuonSign"}

#: Layer types shown in the diagnostics panel. These are parameter groups, not
#: methods, so they take the article's three categorical slots rather than a
#: method colour -- nothing here is "the SignMuon curve".
LAYERS = [
    ("blocks.*.mlp.c_proj", r"$\mathtt{c\_proj}$", SERIES[0]),
    ("blocks.*.mlp.c_fc",   r"$\mathtt{c\_fc}$",   SERIES[1]),
    ("blocks.*.attn.qkvo_w", r"$\mathtt{qkvo\_w}$", SERIES[2]),
]

TOKENS_PER_RANK = 32768   # train_loss is a per-rank SUM over this many tokens


def _f(x):
    return float(x) if x not in (None, "", "None") else None


def load_steps(path: Path):
    out = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(path.open(encoding="utf-8")):
        d = out[r["run_id"]]
        d["step"].append(int(r["step"]))
        for k in ("train_loss", "val_loss", "val_loss_w", "wallclock_ms"):
            d[k].append(_f(r.get(k)))
    return out


def load_diag(path: Path):
    out = defaultdict(list)
    if not path.exists():
        return out
    for r in csv.DictReader(path.open(encoding="utf-8")):
        out[(r["run_id"], r["parameter"])].append(r)
    return out


def _series(d, key):
    """(x, y) over the steps where ``key`` is present."""
    xy = [(s, v) for s, v in zip(d["step"], d[key]) if v is not None]
    return np.array([p[0] for p in xy]), np.array([p[1] for p in xy])


def _ema(y, beta=0.98):
    out, m = np.empty_like(y), y[0]
    for i, v in enumerate(y):
        m = beta * m + (1 - beta) * v
        out[i] = m
    return out


#: Where the inset sits, in axes fractions. The curves run along the diagonal and
#: the legend takes the top-right corner, so the inset goes in the band between
#: them. The y-limits below carry extra headroom for exactly this reason: without
#: it there is no rectangle in these axes that is free of both.
_INSET_RECT = (0.50, 0.30, 0.47, 0.31)


def _zoom(ax, draw, xlim, ylim, rect=_INSET_RECT, nticks=3):
    """Add a zoomed inset to ``ax``, drawn by the same routine as the parent.

    Over the last few hundred steps most of the eight methods are separated by
    less than the resolution of the parent axes -- which is precisely where the
    comparison is decided -- so without this the figure shows only that they are
    close, not how they rank.

    Connector lines are suppressed: with eight overlapping series they cross the
    data and cost more legibility than the rectangle alone provides.
    """
    axins = ax.inset_axes(rect)
    draw(axins)
    axins.set_xlim(*xlim)
    axins.set_ylim(*ylim)
    axins.set_xlabel("")
    axins.set_ylabel("")
    # Sized like everything else in this file: a PRINTED point size times SCALE,
    # never a bare number. A literal 12.5 here printed at 5.8 pt after LaTeX's
    # 1/SCALE reduction -- below the template's floor and unreadable.
    axins.tick_params(axis="both", labelsize=FS_INSET_TICK * SCALE,
                      length=3.5, width=1.0)
    axins.xaxis.set_major_locator(plt.MaxNLocator(nticks))
    axins.yaxis.set_major_locator(plt.MaxNLocator(nticks))
    axins.grid(True, linestyle="--", linewidth=0.7, alpha=0.25)
    for spine in axins.spines.values():          # full box, unlike the parent
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("0.35")
    _, connectors = ax.indicate_inset_zoom(axins, edgecolor="0.35", alpha=0.9,
                                           linewidth=1.1)
    for c in connectors:
        c.set_visible(False)
    return axins


# --------------------------------------------------------------------------
# (a) validation loss vs step
# --------------------------------------------------------------------------
def _draw_val(ax, steps, lw=3.0, ms=0.0, tail=False):
    # The upstream record-#40 curve is not drawn: our Muon arm IS that optimizer,
    # so it would sit underneath Muon and cost a legend slot. The numerical
    # agreement is reported in the caption of Table 3 instead.
    for label, rid, color, ls, mk, ph in RUNS:
        if rid not in steps:
            continue
        key = "val_loss_w" if label in _USE_W else "val_loss"
        x, y = _series(steps[rid], key)
        keep = x > 0                       # step 0 is the untrained model (10.83)
        lab = label + (r" ($\mathbf{W}$)" if label in _USE_W else "")
        ax.plot(x[keep], y[keep], color=color, ls=ls, lw=lw, alpha=0.9,
                marker=mk if ms else None, markersize=ms,
                markevery=_mev(ph, int(keep.sum()), tail),
                markeredgecolor=SURFACE, markeredgewidth=0.9,
                solid_capstyle="round", zorder=3, label=lab)
    ax.axhline(3.28, color="0.3", ls=":", lw=1.6, zorder=1)


def panel_val(ax, steps):
    _draw_val(ax, steps)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Validation loss")
    ax.set_xlim(150, 2400)
    ax.set_ylim(3.24, 4.72)
    # The legend is placed below the figure by the caller, which frees the whole
    # upper-right corner for the inset. The 3.28 target line is identified in the
    # caption rather than by an in-axes label, which would land on the curves.
    # SignSGD (3.405) is deliberately outside the inset's y-range: it is the one
    # method the parent axes already separates unambiguously.
    _zoom(ax, lambda a: _draw_val(a, steps, lw=2.4, ms=9.5, tail=True), (1950, 2365),
          (3.268, 3.398), rect=(0.46, 0.42, 0.52, 0.55))
    return _legend_handles()


# --------------------------------------------------------------------------
# (b) the downlink measurement
# --------------------------------------------------------------------------
def panel_diag(ax, diag, run_id):
    for param, tex, color in LAYERS:
        rows = diag.get((run_id, param), [])
        for key, ls in (("alpha_dn", "-"), ("lag_XW", "--")):
            # Each parameter has TWO kinds of row per step -- the slot table and
            # the per-block gradient line -- so select on the slot being present
            # rather than assuming one row per step.
            pts = sorted((int(r["step"]), _f(r[key])) for r in rows
                         if _f(r.get(key)) is not None)
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, ls=ls,
                    lw=3.0, alpha=0.9, zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Downlink contraction / gap")
    ax.set_xlim(150, 2400)
    handles = [Line2D([], [], color=c, lw=3.0) for _, _, c in LAYERS]
    labels = [t for _, t, _ in LAYERS]
    handles += [Line2D([], [], color="0.35", lw=3.0, ls="-"),
                Line2D([], [], color="0.35", lw=3.0, ls="--")]
    labels += [r"$\alpha(\boldsymbol{\Delta}^{\downarrow})$",
               r"$\|\mathbf{X}-\mathbf{W}\|_F/\|\mathbf{W}\|_F$"]
    leg = ax.legend(handles, labels, loc="lower left", ncol=1,
                    fontsize=FS_LEGEND * SCALE, frameon=False,
                    handlelength=1.9, labelspacing=0.32, borderpad=0.4)
    for text in leg.get_texts():
        text.set_color(INK_2)


# --------------------------------------------------------------------------
# appendix panels
# --------------------------------------------------------------------------
def _draw_train(ax, steps, lw=2.6, ms=0.0, tail=False):
    for label, rid, color, ls, mk, ph in RUNS:
        if rid not in steps:
            continue
        x, y = _series(steps[rid], "train_loss")
        y = y / TOKENS_PER_RANK          # logged as a per-rank sum over the batch
        ax.plot(x, _ema(y), color=color, ls=ls, lw=lw, alpha=0.95, zorder=3,
                marker=mk if ms else None, markersize=ms,
                markevery=_mev(ph, len(x), tail),
                markeredgecolor=SURFACE, markeredgewidth=0.9, label=label)


def panel_train(ax, steps):
    _draw_train(ax, steps)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Training loss")
    ax.set_xlim(0, 2400)
    ax.set_ylim(3.25, 5.15)   # headroom for the legend + inset
    ax.legend(*_legend_handles(), loc="upper right", ncol=2, fontsize=FS_LEGEND * SCALE,
              handlelength=2.4, columnspacing=1.1, labelspacing=0.32, borderpad=0.4)
    _zoom(ax, lambda a: _draw_train(a, steps, lw=2.0, ms=8.5, tail=True), (2050, 2345),
          (3.288, 3.408))


def _draw_time(ax, steps, lw=3.0, ms=0.0, tail=False):
    for label, rid, color, ls, mk, ph in RUNS:
        if rid not in steps:
            continue
        key = "val_loss_w" if label in _USE_W else "val_loss"
        d = steps[rid]
        xy = [(ms, v) for ms, v, s in zip(d["wallclock_ms"], d[key], d["step"])
              if v is not None and ms is not None and s > 0]
        if not xy:
            continue
        x = np.array([p[0] for p in xy]) / 1000.0
        y = np.array([p[1] for p in xy])
        lab = label + (r" ($\mathbf{W}$)" if label in _USE_W else "")
        ax.plot(x, y, color=color, ls=ls, lw=lw, alpha=0.9, zorder=3,
                marker=mk if ms else None, markersize=ms,
                markevery=_mev(ph, len(x), tail),
                markeredgecolor=SURFACE, markeredgewidth=0.9, label=lab)
    ax.axhline(3.28, color="0.3", ls=":", lw=1.6, zorder=1)


def panel_time(ax, steps):
    _draw_time(ax, steps)
    ax.set_xlabel("Training time (s, 8$\\times$H100)")
    ax.set_ylabel("Validation loss")
    ax.set_ylim(3.24, 5.10)   # headroom for the legend + inset
    ax.legend(*_legend_handles(), loc="upper right", ncol=2, fontsize=FS_LEGEND * SCALE,
              handlelength=2.4, columnspacing=1.1, labelspacing=0.32, borderpad=0.4)
    _zoom(ax, lambda a: _draw_time(a, steps, lw=2.4, ms=9.5, tail=True), (119, 148),
          (3.268, 3.398))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=_HERE.parent / "results" / "nanogpt",
                    type=Path)
    ap.add_argument("--outdir", default=None, type=Path,
                    help="default: <results>/figures")
    args = ap.parse_args()

    args.outdir = args.outdir or args.results / "figures"
    steps = load_steps(args.results / "steps.csv")
    diag = load_diag(args.results / "diagnostics.csv")
    args.outdir.mkdir(parents=True, exist_ok=True)

    missing = [r for _, r, *_ in RUNS if r not in steps]
    if missing:
        raise SystemExit("missing runs in steps.csv:\n  " + "\n  ".join(missing))

    ef21ms = dict((lbl, rid) for lbl, rid, *_ in RUNS)["EF21-MuonSign"]

    # --- main text: ONE single-column panel. At a 7-page limit a full-width
    # two-panel figure is not affordable, and the diagnostics belong with the
    # appendix material they support. The legend goes below the axes so the
    # upper-right corner is free for the inset.
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH * SCALE, 5.9))
    handles, labels = panel_val(ax, steps)
    leg = fig.legend(handles, labels, loc="lower center", ncol=3,
                     fontsize=FS_LEGEND * SCALE, frameon=False,
                     handlelength=2.4, columnspacing=1.1, labelspacing=0.35,
                     borderpad=0.45, bbox_to_anchor=(0.5, 0.004))
    for text in leg.get_texts():
        text.set_color(INK_2)
    fig.tight_layout(rect=(0, 0.125, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(args.outdir / f"fig_nanogpt_main.{ext}")
    plt.close(fig)

    # --- appendix: supporting curves
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH * SCALE, 5.6))
    panel_train(axes[0], steps)
    panel_time(axes[1], steps)
    for a, tag in zip(axes, "ab"):
        a.text(-0.085, 1.02, f"({tag})", transform=a.transAxes,
               fontsize=FS_LABEL * SCALE, va="bottom")
    fig.tight_layout(w_pad=3.0)
    for ext in ("pdf", "png"):
        fig.savefig(args.outdir / f"fig_nanogpt_appendix.{ext}")
    plt.close(fig)

    # --- appendix: the downlink measurement
    # printed at 0.65 of the text width, so authored at 0.65 x SCALE
    fig, ax = plt.subplots(figsize=(0.65 * TEXT_WIDTH * SCALE, 5.4))
    panel_diag(ax, diag, ef21ms)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(args.outdir / f"fig_nanogpt_diag.{ext}")
    plt.close(fig)

    print(f"wrote fig_nanogpt_{{main,appendix,diag}}.{{pdf,png}} to {args.outdir}")


if __name__ == "__main__":
    main()
