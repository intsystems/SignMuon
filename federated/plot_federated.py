"""Federated accuracy and loss curves, with a seed band, from ``results/federated/``.

    python3 -m federated.plot_federated
    python3 -m federated.plot_federated --bundle results/federated_results.zip
    python3 -m federated.plot_federated --metrics test_acc test_loss --n-parties 11

Replaces ``notebooks/plot_federated.ipynb``, which read the pre-refactor
``saves_federated/`` layout and the retired name ``EF-UDSignMuon``.

Runs are grouped by ``aggregate.group_key`` -- the configuration *minus* the
seed -- so a multi-seed sweep becomes one curve with a +-1 std band, and two runs
that differ in any other field stay separate rather than being silently pooled.
Only evaluated points are plotted, at the steps actually recorded; nothing is
interpolated or forward-filled, so curves from different ``--eval_freq`` values
still line up pointwise.

By default every algorithm found under ``--root`` is drawn on one axis. A run
that differs from the others in more than the algorithm (a different client
count, say) is a *different experiment*, so ``--n-parties`` / ``--rounds`` filter
rather than overlay, and the caption records what was pinned.

Output goes to ``<root>/figures/`` as PDF and PNG. Nothing is written into
``aaai_article/``. This is the *exploratory* plotter: the figure the paper prints
(``fig:exp_3``) is the two-panel ``fig_federated_main.pdf`` of
``federated.plot_article``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from aggregate import aggregate_group, group_key, load_runs
from common.plotting import (FS_ANNOT, FS_LABEL, INK_2, MUTED, TEXT_WIDTH,
                             color_of, label_of, legend, order_methods,
                             save_figure, style_axes, use_paper_style)
from common.utils import results_root

use_paper_style()

#: One panel per file, at the width a two-up pair would occupy on the page.
PANEL_WIDTH = 0.48 * TEXT_WIDTH

#: Axis labels, and whether lower is better (which fixes the legend corner).
METRICS = {
    "test_acc": ("test accuracy (%)", False),
    "test_loss": ("test cross-entropy", True),
    "val_acc": ("validation accuracy (%)", False),
    "train_loss": ("train cross-entropy", True),
    "gain_spread": ("realized gain spread (max/min over layers)", True),
    # The RAW sign output, before the randomized-zero mapping. Nothing transmitted
    # is zero under the default convention, so this is a diagnostic, not a cost.
    "uplink_zero_frac": ("raw sign entries equal to zero (before the mapping)", True),
    "mv_tie_frac": ("fraction of tied majority votes", True),
}


def collect(root: Path, filters: Dict[str, object]) -> Dict[str, List[dict]]:
    """``{algorithm: runs}`` for the runs under ``root`` matching ``filters``."""
    runs = load_runs([root])
    if not runs:
        return {}
    kept, dropped = {}, 0
    for run in runs:
        cfg = run["config"]
        if any(cfg.get(k) != v for k, v in filters.items() if v is not None):
            dropped += 1
            continue
        alg = cfg.get("algorithm") or cfg.get("optimizer")
        if alg is None:
            dropped += 1
            continue
        kept.setdefault(alg, []).append(run)
    if dropped:
        print(f"  ~ {dropped} run(s) filtered out")
    # Within one algorithm, several configurations may still be present (two
    # learning rates, say). Keep the largest group and say so, rather than
    # pooling runs that are not repeats of each other.
    resolved = {}
    for alg, group in kept.items():
        by_config: Dict[tuple, List[dict]] = {}
        for run in group:
            by_config.setdefault(group_key(run["config"]), []).append(run)
        best = max(by_config.values(), key=len)
        if len(by_config) > 1:
            others = sum(len(v) for v in by_config.values()) - len(best)
            print(f"  ~ {alg}: {len(by_config)} distinct configurations; plotting "
                  f"the one with the most seeds ({len(best)} run(s)), ignoring "
                  f"{others}. Narrow it with --lr / --rounds / --n-parties.")
        resolved[alg] = best
    return {a: resolved[a] for a in order_methods(resolved)}


def fig_metric(plt, data: Dict[str, List[dict]], metric: str, xlabel: str):
    label, lower_is_better = METRICS.get(metric, (metric, False))
    # Authored at the width a 0.48\textwidth subfigure prints at, which is how
    # the paper places these; the legend sits in the gutter, so the saved
    # bounding box is wider than the axes and LaTeX still scales a little.
    fig, ax = plt.subplots(figsize=(PANEL_WIDTH, 2.4))
    style_axes(ax, logy=lower_is_better and metric.endswith("loss"))
    drawn = 0
    for alg, runs in data.items():
        agg = aggregate_group(runs, metric)
        if agg is None:
            continue
        color = color_of(alg)
        n = agg["n_runs"]
        ax.plot(agg["steps"], agg["mean"], color=color, linewidth=1.6,
                label=f"{label_of(alg)}" + (f"  (n={n})" if n > 1 else ""),
                zorder=3)
        if n > 1:
            lo = [m - s for m, s in zip(agg["mean"], agg["std"])]
            hi = [m + s for m, s in zip(agg["mean"], agg["std"])]
            ax.fill_between(agg["steps"], lo, hi, color=color, alpha=0.16,
                            linewidth=0, zorder=1)
        drawn += 1
    if not drawn:
        plt.close(fig)
        return None
    ax.set_xlabel(xlabel, color=INK_2, fontsize=FS_LABEL)
    ax.set_ylabel(label, color=INK_2, fontsize=FS_LABEL)
    ax.set_title("band is +-1 sample std over seeds", color=MUTED,
                 fontsize=FS_ANNOT - 1.5, loc="left", pad=6)
    legend(ax, outside=True)
    fig.tight_layout()
    return fig


def get_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", default=None,
                   help="Archive from 'federated.export_article' -- the .zip itself "
                        "or the directory it unpacks to. Takes precedence over --root")
    p.add_argument("--root", default=None,
                   help="Run tree to read (default: results/federated/)")
    p.add_argument("--out", default=None,
                   help="Where to write the figures (default: <root>/figures/)")
    p.add_argument("--metrics", nargs="+", default=["test_acc", "test_loss"],
                   metavar="NAME",
                   help=f"any recorded series; known ones are "
                        f"{sorted(METRICS)}")
    p.add_argument("--n-parties", type=int, default=None,
                   help="Plot only runs at this client count")
    p.add_argument("--rounds", type=int, default=None,
                   help="Plot only runs at this round budget")
    p.add_argument("--lr", type=float, default=None,
                   help="Plot only runs at this base learning rate")
    p.add_argument("--formats", nargs="+", default=["pdf", "png"])
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

    if args.bundle:
        from federated.export_article import open_bundle, runs_root
        bundle = open_bundle(Path(args.bundle))
        root = runs_root(bundle)
        print(f"Bundle {bundle.resolve()}")
        if args.out is None:
            args.out = str(bundle / "figures")
    else:
        root = Path(args.root) if args.root else results_root() / "federated"
    if not root.is_dir():
        print(f"No runs at {root.resolve()}.\n"
              f"Run `python3 -m federated.main ...` or "
              f"`python3 -m federated.overnight` first.")
        return 1
    out = Path(args.out) if args.out else root / "figures"

    data = collect(root, {"n_parties": args.n_parties, "rounds": args.rounds,
                          "lr": args.lr})
    if not data:
        print(f"Nothing to plot under {root.resolve()} after filtering.")
        return 1
    print(f"Plotting {len(data)} algorithm(s): {', '.join(data)}")

    written = []
    for metric in args.metrics:
        fig = fig_metric(plt, data, metric, xlabel="communication round")
        if fig is None:
            print(f"  ~ {metric}: not recorded by any run")
            continue
        written += save_figure(fig, out, f"federated_{metric}",
                               formats=args.formats)
        plt.close(fig)

    if not written:
        print("Nothing plotted.")
        return 1
    print(f"\nWrote {len(written)} file(s) to {out.resolve()}:")
    for path in written:
        print(f"    {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
