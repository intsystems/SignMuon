"""Shared figure style: one palette, one method-to-colour map, one type scale.

Every plotting script in this repository imports from here, so a method keeps the
same colour whether it appears in a synthetic-benchmark figure, a federated
accuracy curve or the centralized learning-rate sweep. That consistency is the
first reason this module exists -- a reader who has learned that orange is
EF21-MuonUSign in one figure should not have to relearn it in the next.

The second reason is the paper itself. ``use_paper_style()`` sets the matplotlib
rcParams that make a figure match ``aaai2027.sty``:

* **Times text and STIX math.** The template loads ``newtxtext``, so an axis
  label in matplotlib's default sans-serif reads as a foreign object on the page.
* **TrueType outlines** (``pdf.fonttype = 42``). Matplotlib's default is Type 3,
  which AAAI forbids outright -- the style file raises an error for the ``bbm``
  package on exactly these grounds, and a Type 3 figure is the same violation
  arriving through a different door.
* **Type sizes in printed points.** Author a figure at ``TEXT_WIDTH`` or
  ``COLUMN_WIDTH`` and include it at ``width=\\textwidth`` / ``\\columnwidth``:
  LaTeX then scales by exactly 1, and a 9 pt label is 9 pt on the page, against
  the template's 10 pt body. A figure drawn oversize and scaled down is what
  produces the 6 pt tick labels that reviewers complain about.

Scripts that deliberately draw oversize pass ``use_paper_style(scale=k)``, which
multiplies every size and line width by ``k`` so the *printed* result is
unchanged.

Nothing here writes into ``aaai_article/``. The paper's figures are copied over
deliberately; a plotting run must never silently replace one. (The
counterexample scripts are the exception, and say so.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib

__all__ = [
    "SURFACE", "INK", "INK_2", "MUTED", "GRID", "AXIS", "REFERENCE", "SERIES",
    "TEXT_WIDTH", "COLUMN_WIDTH", "FS_LABEL", "FS_TICK", "FS_LEGEND", "FS_ANNOT",
    "METHOD_LABEL", "METHOD_COLOR", "METHOD_MARKER", "METHOD_ORDER",
    "use_paper_style", "label_of", "color_of", "marker_of", "order_methods",
    "style_axes", "panel_legend", "figure_legend", "legend", "save_figure",
]

# -- palette ---------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
REFERENCE = "#898781"
INK, INK_2, MUTED = "#2b2a27", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"

# -- page geometry ---------------------------------------------------------
#: ``aaai2027.sty``: ``\textwidth 7.0in``, ``\columnsep 0.375in``, two columns.
#: A ``figure*`` spans TEXT_WIDTH, a ``figure`` spans COLUMN_WIDTH; author at
#: those numbers and include at ``width=\textwidth`` / ``\columnwidth``.
TEXT_WIDTH = 7.0
COLUMN_WIDTH = (TEXT_WIDTH - 0.375) / 2      # 3.3125

#: Type scale, in points on the printed page. The body is 10 pt on 11 and
#: captions are 9 pt, so axis labels sit just under the caption and tick labels
#: one point under those -- small enough to stay out of the way, large enough to
#: read without magnification.
FS_LABEL, FS_TICK, FS_LEGEND, FS_ANNOT = 9.0, 8.0, 7.5, 7.5

#: Display names, spelled as the paper spells them. ``SignMuon`` signs AFTER the
#: LMO, ``MuonUSign`` signs BEFORE it, ``MuonSign`` signs on both sides -- the
#: names are not interchangeable and a figure legend is exactly where a reader
#: would be misled by the old convention.
METHOD_LABEL: Dict[str, str] = {
    "signmuon": "SignMuon",
    "ef21signmuon": "EF21-SignMuon",
    "muonusign": "MuonUSign",
    "muonsign": "MuonSign",
    "ef21muonusign": "EF21-MuonUSign",
    "ef21muonsign": "EF21-MuonSign",
    "muon": "Muon",
    "muonserver": "Muon (server LMO)",
    "signsgd": "SignSGD",
    "sgd": "SGD",
    "adam": "Adam",
}

#: Hue encodes the family: the sign-around-the-LMO methods are cool, the
#: error-feedback methods are warm, the uncompressed references are green/grey.
METHOD_COLOR: Dict[str, str] = {
    "signmuon": "#2a78d6",
    "muonusign": "#6f4ecf",
    "muonsign": "#0f9bd0",
    "ef21signmuon": "#d4342b",
    "ef21muonusign": "#eb6834",
    "ef21muonsign": "#c9761a",
    "muon": "#1baf7a",
    "muonserver": "#0f7a58",
    "signsgd": "#8a6d3b",
    "sgd": "#898781",
    "adam": "#52514e",
}

#: A second channel for figures that put more than three or four methods on one
#: axis, where hue alone stops separating them -- and the channel a reader with
#: a colour vision deficiency is left with. Figures with three series and a
#: reference (the CIFAR panels) do not need it.
METHOD_MARKER: Dict[str, str] = {
    "signmuon": "^",
    "ef21signmuon": "D",
    "muonusign": "s",
    "muonsign": "v",
    "ef21muonusign": "o",
    "ef21muonsign": "P",
    "muon": "",
    "muonserver": "",
    "signsgd": "*",
    "sgd": "x",
    "adam": "d",
}

#: The order the paper lists them in: six proposed methods, then the references.
METHOD_ORDER: List[str] = list(METHOD_LABEL)

#: Pre-refactor spellings that still appear in older ``metrics.json`` files and
#: in the nanoGPT logs. Resolved so an old result plots under its current name.
_ALIASES = {
    "signmuon_cl": "signmuon",
    "signmuon_ef_21": "ef21signmuon",
    "signmuon_ef_ud": "ef21muonsign",
    "ef_usignmuon": "ef21muonusign",
    "ef_udsignmuon": "ef21muonsign",
    "muon_server": "muonserver",
    "muonlmoserver": "muonserver",
}


def _canonical(method: str) -> str:
    key = str(method).strip().lower().replace("-", "").replace(" ", "")
    return _ALIASES.get(str(method).strip().lower(), key)


def label_of(method: str) -> str:
    """Display name, falling back to the raw key for anything unregistered."""
    return METHOD_LABEL.get(_canonical(method), str(method))


def color_of(method: str, fallback: str = REFERENCE) -> str:
    return METHOD_COLOR.get(_canonical(method), fallback)


def marker_of(method: str, fallback: str = "") -> str:
    """Marker for the method, or ``""`` (no marker) if it has none."""
    return METHOD_MARKER.get(_canonical(method), fallback)


def order_methods(methods: Iterable[str]) -> List[str]:
    """Sort into the paper's order; unknown names keep their relative order last."""
    known = {m: i for i, m in enumerate(METHOD_ORDER)}
    return sorted(methods,
                  key=lambda m: (known.get(_canonical(m), len(known)), str(m)))


# -- rcParams --------------------------------------------------------------


def use_paper_style(scale: float = 1.0) -> None:
    """Install the paper's rcParams. Call once, before creating any figure.

    ``scale`` multiplies every point size and line width, for a figure that is
    authored larger than it prints; at ``scale = k`` and a LaTeX reduction of
    ``1/k`` the printed result matches a figure authored at ``scale = 1``. Prefer
    authoring at the printed width and leaving this at 1.
    """
    s = float(scale)
    matplotlib.rcParams.update({
        # Times text and Times-like math, as aaai2027.sty loads newtxtext. The
        # fallbacks matter: the CI box has no Times New Roman.
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif",
                       "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": FS_TICK * s,
        "axes.labelsize": FS_LABEL * s,
        "axes.titlesize": FS_LABEL * s,
        "xtick.labelsize": FS_TICK * s,
        "ytick.labelsize": FS_TICK * s,
        "legend.fontsize": FS_LEGEND * s,
        # Ink: labels and titles in the darker grey, ticks in the muted one.
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK_2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8 * s,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6 * s,
        "grid.linestyle": "-",
        "lines.linewidth": 1.6 * s,
        "lines.markersize": 3.4 * s,
        "lines.markeredgewidth": 0.6 * s,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3 * s,
        "ytick.major.size": 3 * s,
        "xtick.major.width": 0.8 * s,
        "ytick.major.width": 0.8 * s,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.45,
        "legend.labelspacing": 0.3,
        "legend.borderaxespad": 0.3,
        "figure.facecolor": SURFACE,
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.facecolor": SURFACE,
        # TrueType, not matplotlib's default Type 3: AAAI rejects Type 3 fonts.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# -- drawing ---------------------------------------------------------------


def style_axes(ax, logx: bool = False, logy: bool = False) -> None:
    """The house axis style: hairline grid, no top/right spines, muted ticks.

    ``use_paper_style`` already sets all of this through rcParams; this stays for
    the per-axes bits an rcParam cannot express (the transparent patch, which
    keeps a neighbouring panel from painting over labels in the gutter) and for
    axes created before the rcParams were installed.
    """
    ax.patch.set_alpha(0)
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=matplotlib.rcParams["grid.linewidth"],
            zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(matplotlib.rcParams["axes.linewidth"])
    ax.tick_params(colors=MUTED,
                   labelsize=matplotlib.rcParams["xtick.labelsize"],
                   length=matplotlib.rcParams["xtick.major.size"],
                   width=matplotlib.rcParams["xtick.major.width"])
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")


def panel_legend(ax, loc: str = "best", ncol: int = 1,
                 fontsize: Optional[float] = None):
    """A compact legend in an empty corner of the panel.

    Two of the series colours sit below 3:1 against a white page, which the
    palette permits only with a visible label rather than hue alone; a legend
    entry is that label, with the swatch adjacent to the name.
    """
    leg = ax.legend(loc=loc, ncol=ncol, fontsize=fontsize, frameon=False,
                    handlelength=1.3, handletextpad=0.4, labelspacing=0.22,
                    borderpad=0.15, borderaxespad=0.3)
    if leg is not None:
        for text in leg.get_texts():
            text.set_color(INK_2)
    return leg


def figure_legend(fig, handles, labels, ncol: int, y: float = -0.015,
                  fontsize: Optional[float] = None):
    """One legend under a row of panels that all draw the same methods."""
    leg = fig.legend(handles, labels, loc="lower center", ncol=ncol,
                     fontsize=fontsize or matplotlib.rcParams["legend.fontsize"],
                     frameon=False, handlelength=1.9, handletextpad=0.4,
                     columnspacing=1.1, bbox_to_anchor=(0.5, y))
    for text in leg.get_texts():
        text.set_color(INK_2)
    return leg


def legend(ax, *, loc: str = "best", ncol: int = 1, title: Optional[str] = None,
           outside: bool = False):
    """A legend with the same muted styling as the axes.

    ``outside=True`` parks it to the right of the axes. With ten methods on one
    plot there is no interior corner that does not cover data, and a legend that
    hides the curves it labels is worse than no legend.
    """
    if outside:
        leg = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), ncol=ncol,
                        title=title, frameon=False, borderaxespad=0.0)
    else:
        leg = ax.legend(loc=loc, ncol=ncol, title=title, frameon=False)
    if leg is not None:
        for text in leg.get_texts():
            text.set_color(INK_2)
        if leg.get_title() is not None:
            leg.get_title().set_color(MUTED)
            leg.get_title().set_fontsize(matplotlib.rcParams["legend.fontsize"])
    return leg


def save_figure(fig, out_dir: Path, stem: str,
                formats: Sequence[str] = ("pdf", "png"), dpi: int = 200,
                tight: bool = True) -> List[Path]:
    """Write ``stem`` into ``out_dir`` in each format, and return the paths.

    ``tight=False`` keeps the figure exactly the size it was authored at. That
    is what a paper figure wants: a tight bounding box trims the margins, LaTeX
    scales the result back up to ``\\textwidth``, and every point size in the
    figure grows by the same unintended factor. Use it together with
    ``fig.subplots_adjust``; leave ``tight=True`` for a figure whose legend sits
    outside the axes, where the bounding box is the only thing that knows how
    wide the result is.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=dpi, facecolor=SURFACE,
                    **({"bbox_inches": "tight"} if tight else {}))
        written.append(path)
    return written
