"""
Plot the helper functions of the EF21-SignMuon counterexample (Theorem 4).

The appendix builds the objective ``f = gamma*h(W22) + A*(Phi1(W12) +
Phi2(W21)) + sum_k b_k`` from pieces a reader never sees drawn: two periodic
ramps and their antiderivatives, the floored linear slope of the boundedness
remark, and the compactly supported corrections.  This script draws them, one
panel each, so the construction can be checked by eye against the text:

* ``psi_1, Phi_1`` and ``psi_2, Phi_2`` over two periods, with the residues
  ``rho_i^+`` (odd iterates) and ``rho_i^-`` (even iterates) marked on the
  plateaus where ``psi_i = +-1`` exactly;
* the divergence term ``gamma*h(W22)`` against the unbounded ``-gamma*W22`` it
  replaces (they agree on the whole visited region ``W22 <= 7/10``);
* the correction ``b_k`` along the ray ``Z_k + s*C_k/||C_k||``, normalized,
  together with the cutoff ``phi``: zero at the center, gradient ``C_k`` there,
  support inside ``||W - Z_k||_F <= r``.

Usage (from ``code/``, the package root):

    python3 -m counterexamples.plot_ef21_construction

Writes ``ef21_construction`` as PNG+PDF to ``figures/`` and as PDF to
``../aaai_article/images/counterexamples/`` (the deliberate exception to the
"plots never write into aaai_article/" rule, like run_counterexamples.py).
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")            # headless-safe; figures are written to disk
import matplotlib.pyplot as plt  # noqa: E402

from common.plotting import (FS_ANNOT, FS_LABEL, GRID, INK_2, REFERENCE,  # noqa: E402
                             SERIES, SURFACE, TEXT_WIDTH, panel_legend,
                             style_axes, use_paper_style)
# The pieces themselves come from the module that implements the theorem, so
# the figure cannot drift from the construction it illustrates.
from counterexamples.problems import (_GAMMA, _PS1, _PS2, _RBUMP,  # noqa: E402
                                      _chi, _floor_h)

use_paper_style()

BLUE, ORANGE, GREEN = SERIES

# The residues the trajectory keeps (mod p_i): rho^+ at odd t, rho^- at even t.
RHO1 = (131 / 200, 140 / 200)
RHO2 = (61 / 200, 0.0)


def _ramp_panel(ax, ramp, idx, rho_plus, rho_minus):
    """psi_i and Phi_i over two periods, residues marked on the plateaus."""
    p = ramp.p
    x = np.linspace(0.0, 2 * p, 1601)
    ax.plot(x, [ramp.psi(v) for v in x], color=BLUE,
            label=rf"$\psi_{idx}$", zorder=3)
    ax.plot(x, [ramp.Phi(v) for v in x], color=ORANGE,
            label=rf"$\Phi_{idx}$", zorder=3)
    for rho, val, name, dy, va in ((rho_plus, 1.0, rf"$\rho_{idx}^{{+}}$", 5, "bottom"),
                                   (rho_minus, -1.0, rf"$\rho_{idx}^{{-}}$", -5, "top")):
        ax.plot([rho], [val], marker="o", color=GREEN, markersize=4.5,
                markeredgecolor=SURFACE, markeredgewidth=0.6, zorder=5,
                linestyle="none")
        ax.annotate(name, xy=(rho, val), xytext=(2, dy),
                    textcoords="offset points", ha="left", va=va,
                    fontsize=FS_ANNOT, color=INK_2, zorder=6)
    ax.set_xlim(-0.04 * p, 2 * p)
    # Headroom above the +-1 plateaus for a one-row legend.
    ax.set_ylim(-1.5, 1.95)
    ax.set_xlabel(r"$W_{12}$" if idx == 1 else r"$W_{21}$",
                  color=INK_2, fontsize=FS_LABEL)
    panel_legend(ax, loc="upper center", ncol=2, fontsize=FS_ANNOT)


def _slope_panel(ax):
    """gamma*h(W22) against the -gamma*W22 it replaces (boundedness remark)."""
    v = np.linspace(-1.0, 3.0, 801)
    ax.plot(v, -_GAMMA * v, color=REFERENCE, linestyle=(0, (4, 2)),
            label=r"$-\gamma W_{22}$", zorder=2)
    ax.plot(v, [_GAMMA * _floor_h(w)[0] for w in v], color=BLUE,
            label=r"$\gamma\,h(W_{22})$", zorder=3)
    # The whole trajectory keeps (X_t)_{22} <= 7/10, where the two agree.
    ax.axvspan(-1.0, 7 / 10, color=GRID, alpha=0.55, zorder=0)
    ax.annotate("visited\nregion", xy=(-0.62, -1.42), fontsize=FS_ANNOT,
                color=INK_2, ha="left", va="top")
    ax.set_xlim(-1.0, 3.0)
    ax.set_xlabel(r"$W_{22}$", color=INK_2, fontsize=FS_LABEL)
    panel_legend(ax, loc="upper right", fontsize=FS_ANNOT)


def _bump_panel(ax):
    """The correction b_k along Z_k + s*C_k/||C_k||_F, normalized by ||C_k||_F*r."""
    u = np.linspace(-1.6, 1.6, 801)
    ax.plot(u, [_chi(abs(s)) for s in u], color=REFERENCE,
            linestyle=(0, (4, 2)), label=r"$\phi(|s|/r)$", zorder=2)
    ax.plot(u, [s * _chi(abs(s)) for s in u], color=BLUE,
            label=r"$b_k/(\|\mathbf{C}_k\|_F\, r)$", zorder=3)
    ax.axhline(0, color=REFERENCE, linewidth=0.6, zorder=1)
    ax.set_xlim(-1.6, 1.6)
    ax.set_xlabel(r"$s/r$", color=INK_2, fontsize=FS_LABEL)
    panel_legend(ax, loc="upper left", fontsize=FS_ANNOT)


def main():
    fig, axes = plt.subplots(1, 4, figsize=(TEXT_WIDTH, 2.0))
    for ax in axes:
        style_axes(ax)
    _ramp_panel(axes[0], _PS1, 1, *RHO1)
    _ramp_panel(axes[1], _PS2, 2, *RHO2)
    _slope_panel(axes[2])
    _bump_panel(axes[3])
    fig.subplots_adjust(left=0.052, right=0.995, top=0.965, bottom=0.235,
                        wspace=0.34)

    here = os.path.dirname(os.path.abspath(__file__))
    figdir = os.path.join(here, "figures")
    os.makedirs(figdir, exist_ok=True)
    images_dir = os.path.abspath(
        os.path.join(here, "..", "..", "aaai_article", "images",
                     "counterexamples"))
    targets = [(os.path.join(figdir, "ef21_construction"), True)]
    if os.path.isdir(os.path.dirname(images_dir)):
        os.makedirs(images_dir, exist_ok=True)
        targets.append((os.path.join(images_dir, "ef21_construction"), False))
    for stem, want_png in targets:
        fig.savefig(stem + ".pdf", facecolor=SURFACE)
        if want_png:
            fig.savefig(stem + ".png", dpi=200, facecolor=SURFACE)
        print("saved ->", stem + (".{png,pdf}" if want_png else ".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
