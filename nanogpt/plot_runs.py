"""Comparison plots for the NanoGPT speedrun runs: loss vs steps, loss vs time.

Consumes ``steps.csv`` / ``runs.csv`` as written by ``parse_logs.py``:

    python parse_logs.py    # ../results/nanogpt/logs/ -> ../results/nanogpt/*.csv
    python plot_runs.py     # ../results/nanogpt/steps.csv
                            #     -> ../results/nanogpt/figures/exploratory/

and writes, for each of {steps, time} x {val, train}, a PDF and a PNG.

What the two x-axes mean, and why both
--------------------------------------
``loss vs steps`` compares the methods as *optimizers*: same data, same number of
updates, so the difference is the update rule. ``loss vs time`` compares them as
*systems*: a method that needs an extra orthogonalization or an extra
error-feedback buffer pays for it in wall-clock. The x-axis for the second is the
speedrun's own ``train_time`` -- validation and compilation excluded -- so it is
comparable across runs and across optimizers.

Rigour notes (the concurrent Sign-Muon paper plots an "anytime best" envelope)
-----------------------------------------------------------------------------
* The default is the RAW curve, not the running minimum. A running-min envelope
  is monotone by construction and hides instability, which is exactly the
  behaviour the sign methods are being tested for. ``--anytime`` draws the
  envelope as well, dashed, for comparison with the published figures.
* Validation points are marked; the line between them is interpolation, not data.
  Train loss is per-step and therefore drawn as the real, noisy series (``--ema``
  smooths it for readability but the raw series stays visible underneath).
* Diverged runs are drawn to the step at which they diverged and flagged in the
  legend, rather than dropped -- a divergence is a result.
* Loss and perplexity are the same information (``ppl = exp(loss)``);
  ``--metric perplexity`` relabels for comparison with figures that use it. Never
  both on one figure -- one axis per chart.
* Every validation figure carries upstream record #40's own curve
  (``reference_record40.csv``, mean of its five published 8xH100 logs) as a grey
  dashed backdrop with a +/-3 sd band. ``Muon`` here IS record #40's optimizer, so
  it must lie on that line; a gap means the port is broken and nothing else on the
  figure can be trusted yet. ``--no-reference`` removes it.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import OrderedDict
from pathlib import Path

#: Paths default to this arm's results tree, resolved against this file so
#: the scripts work from any working directory.
_HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# Palette: the documented categorical order (fixed slots, never cycled), stepped
# for each surface. Eight optimizers == eight slots exactly; a ninth series would
# have to fold into "other" or be faceted rather than get a generated hue.
# --------------------------------------------------------------------------
_LIGHT = dict(
    surface="#fcfcfb", text="#0b0b0b", text2="#52514e", grid="#d9d8d4",
    series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
            "#e87ba4", "#008300", "#4a3aa7", "#e34948"])
_DARK = dict(
    surface="#1a1a19", text="#ffffff", text2="#c3c2b7", grid="#3a3a38",
    series=["#3987e5", "#d95926", "#199e70", "#c98500",
            "#d55181", "#008300", "#9085e9", "#e66767"])

# Fixed slot per optimizer: colour follows the entity, so adding or filtering a
# run never repaints the survivors. Reference methods first.
SERIES_ORDER = ["Muon", "SignSGD", "SignMuon", "MuonUSign", "MuonSign",
                "EF21-SignMuon", "EF21-MuonUSign", "EF21-MuonSign"]
# Secondary (non-colour) encoding, so identity survives greyscale and CVD.
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def _slot(name: str) -> int:
    try:
        return SERIES_ORDER.index(name)
    except ValueError:
        return len(SERIES_ORDER) + sorted([name]).index(name)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def _num(x):
    if x is None or x == "" or x == "None":
        return None
    try:
        v = float(x)
    except ValueError:
        return None
    return None if math.isnan(v) else v


def load_steps(path: Path) -> "OrderedDict[str, dict]":
    """steps.csv -> {run_id: {label, optimizer, lr, step[], ms[], train[], val[]}}."""
    runs: "OrderedDict[str, dict]" = OrderedDict()
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rid = row["run_id"]
            r = runs.setdefault(rid, dict(
                run_id=rid, optimizer=row["optimizer"] or rid, lr=_num(row["lr"]),
                lr_scaling=row.get("lr_scaling"), step=[], ms=[], train=[], val=[]))
            r["step"].append(int(row["step"]))
            r["ms"].append(_num(row["wallclock_ms"]))
            r["train"].append(_num(row["train_loss"]))
            r["val"].append(_num(row["val_loss"]))
    for r in runs.values():
        r["label"] = f"{r['optimizer']}  lr={r['lr']:g}" if r["lr"] else r["optimizer"]
    return runs


#: Upstream record #40's own validation curve, averaged over its five published
#: 8xH100 logs. Drawn behind every validation figure as the "is the port intact?"
#: baseline: the `Muon` arm is record #40's optimizer verbatim, so it must land on
#: this line. Comment lines (``#``) carry the provenance.
REFERENCE_CSV = Path(__file__).resolve().parent / "reference_record40.csv"


def load_reference(path: Path) -> dict | None:
    """reference_record40.csv -> {step[], ms[], val[], sd[]}, or None if absent."""
    if not path.exists():
        return None
    rows = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(ln for ln in fh if not ln.lstrip().startswith("#")):
            rows.append(row)
    if not rows:
        return None
    return dict(step=[int(r["step"]) for r in rows],
                ms=[_num(r["train_time_ms"]) for r in rows],
                val=[_num(r["val_loss"]) for r in rows],
                sd=[_num(r["val_loss_sd"]) for r in rows])


def load_runs_meta(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        return {row["run_id"]: row for row in csv.DictReader(fh)}


def _series(run: dict, which: str, xaxis: str):
    """(x, y) pairs for one run, dropping steps where the metric is absent."""
    xs, ys = [], []
    for st, ms, tr, va in zip(run["step"], run["ms"], run["train"], run["val"]):
        y = tr if which == "train" else va
        x = st if xaxis == "steps" else ms
        if y is None or x is None:
            continue
        xs.append(x)
        ys.append(y)
    return xs, ys


def _ema(ys, alpha):
    out, acc = [], None
    for y in ys:
        acc = y if acc is None else alpha * acc + (1 - alpha) * y
        out.append(acc)
    return out


def _running_min(ys):
    out, best = [], float("inf")
    for y in ys:
        best = min(best, y)
        out.append(best)
    return out


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------

def make_figure(runs, which, xaxis, theme, args, meta, reference=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=args.dpi)
    fig.patch.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])

    xscale = 1.0 if xaxis == "steps" else 1000.0            # ms -> s
    if xaxis == "time" and args.minutes:
        xscale = 60_000.0

    plotted, all_y = 0, []

    # Upstream record #40, drawn first so it sits behind everything and reads as
    # the backdrop rather than as a ninth competitor: neutral grey, no marker, and
    # deliberately NOT given a palette slot. `Muon` should lie on top of it; any
    # visible gap is a port bug, not an optimizer result. Its y-values are kept out
    # of `all_y` so the reference never drives the axis limits.
    if reference is not None and which == "val":
        rx = reference["step"] if xaxis == "steps" else reference["ms"]
        pts = [(x / xscale, y, sd) for x, y, sd in
               zip(rx, reference["val"], reference["sd"]) if x is not None and y is not None]
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            los = [p[1] - 3 * (p[2] or 0.0) for p in pts]
            his = [p[1] + 3 * (p[2] or 0.0) for p in pts]
            if args.metric == "perplexity":
                ys, los, his = ([math.exp(min(v, 20.0)) for v in seq]
                                for seq in (ys, los, his))
            # +/-3 sd over the record's own five runs: the band a correct port must
            # land inside, not a confidence interval on our single run.
            ax.fill_between(xs, los, his, color=theme["text2"], alpha=0.16, lw=0, zorder=1)
            ax.plot(xs, ys, color=theme["text2"], lw=1.4, ls=(0, (5, 2)), zorder=1.5,
                    label="record #40 upstream (mean of 5)")
    for rid, run in sorted(runs.items(), key=lambda kv: _slot(kv[1]["optimizer"])):
        xs, ys = _series(run, which, xaxis)
        if not xs:
            continue
        xs = [x / xscale for x in xs]
        if args.metric == "perplexity":
            ys = [math.exp(min(y, 20.0)) for y in ys]
        slot = _slot(run["optimizer"]) % len(theme["series"])
        color = theme["series"][slot]
        marker = MARKERS[slot % len(MARKERS)]
        label = run["label"]
        info = meta.get(rid, {})
        if str(info.get("diverged", "")).lower() == "true":
            label += "  (diverged)"

        if which == "train":
            # raw series stays visible; the smoothed one carries the shape
            ax.plot(xs, ys, color=color, lw=0.6, alpha=0.22, zorder=2)
            ys_draw = _ema(ys, args.ema) if args.ema > 0 else ys
            ax.plot(xs, ys_draw, color=color, lw=1.8, label=label, zorder=3,
                    solid_capstyle="round")
            all_y += ys_draw[max(1, len(ys_draw) // 20):]
        else:
            ax.plot(xs, ys, color=color, lw=1.8, label=label, zorder=3,
                    marker=marker, markersize=6, markeredgecolor=theme["surface"],
                    markeredgewidth=1.2, solid_capstyle="round")
            all_y += ys[1:] if len(ys) > 1 else ys
            if args.anytime:
                ax.plot(xs, _running_min(ys), color=color, lw=1.1, ls="--",
                        alpha=0.75, zorder=2)
        plotted += 1

    if not plotted:
        plt.close(fig)
        return None

    unit = "steps" if xaxis == "steps" else ("training time (min)" if args.minutes
                                             else "training time (s)")
    metric = ("validation" if which == "val" else "training") + " " + args.metric
    ax.set_xlabel("optimizer " + unit if xaxis == "steps" else unit,
                  color=theme["text2"], fontsize=10)
    ax.set_ylabel(metric, color=theme["text2"], fontsize=10)
    ax.set_title(f"NanoGPT speedrun (record #40): {metric} vs "
                 f"{'steps' if xaxis == 'steps' else 'wall-clock'}",
                 color=theme["text"], fontsize=11.5, loc="left", pad=10)

    lo, hi = args.ymin, args.ymax
    if all_y:
        finite = sorted(y for y in all_y if math.isfinite(y))
        if finite:
            if lo is None:
                lo = finite[0] - 0.02 * (abs(finite[0]) + 1e-9)
            if hi is None:
                # the first steps are an order of magnitude above the endgame and
                # would flatten everything worth seeing; clip to the bulk
                hi = finite[int(0.75 * (len(finite) - 1))]
    if lo is not None and hi is not None and hi > lo:
        ax.set_ylim(lo, hi)
    if args.ylog:
        ax.set_yscale("log")
    if args.xlog:
        ax.set_xscale("log")
    if args.target and which == "val":
        t = math.exp(args.target) if args.metric == "perplexity" else args.target
        ax.axhline(t, color=theme["text2"], lw=1.0, ls=":", zorder=1)
        # left edge: the curves are still far above the target there, so the
        # label cannot land on top of a mark
        ax.annotate(f"target {args.target:g}", xy=(0.008, t), xycoords=("axes fraction", "data"),
                    ha="left", va="bottom", fontsize=8.5, color=theme["text2"])

    ax.grid(True, color=theme["grid"], lw=0.7, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["grid"])
    ax.tick_params(colors=theme["text2"], labelsize=9)
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v:,.0f}" if abs(v) >= 1 else f"{v:g}"))

    # Legend below the axes: a loss curve sweeps from the top-left to the
    # bottom-right, so every in-axes corner is occupied by some run at some point.
    handles, labels = ax.get_legend_handles_labels()
    n_entries = len(labels)             # runs, plus the reference line when drawn
    ncol = 4 if n_entries > 6 else max(1, n_entries)
    rows = -(-n_entries // ncol)
    band = 0.03 + 0.045 * rows          # figure fraction reserved for the legend
    leg = fig.legend(handles, labels, frameon=False, fontsize=8.8,
                     ncol=ncol, loc="upper center", bbox_to_anchor=(0.5, band),
                     handlelength=2.2, columnspacing=1.6, handletextpad=0.6)
    for t in leg.get_texts():
        t.set_color(theme["text"])
    fig.tight_layout(rect=(0, band, 1, 1))
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps_csv", type=Path, nargs="?", default=_HERE.parent / "results" / "nanogpt" / "steps.csv")
    ap.add_argument("-o", "--outdir", type=Path, default=_HERE.parent / "results" / "nanogpt" / "figures" / "exploratory")
    ap.add_argument("--runs-csv", type=Path, default=None,
                    help="runs.csv (default: next to steps.csv); used for the "
                         "'diverged' legend flags")
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these optimizer names")
    ap.add_argument("--metric", choices=["loss", "perplexity"], default="loss")
    ap.add_argument("--ema", type=float, default=0.9,
                    help="EMA factor for the train-loss curve (0 disables)")
    ap.add_argument("--anytime", action="store_true",
                    help="also draw the running-minimum envelope (dashed)")
    ap.add_argument("--reference", type=Path, default=REFERENCE_CSV,
                    help="upstream record-#40 val curve to draw as a baseline "
                         "(default: reference_record40.csv next to this script)")
    ap.add_argument("--no-reference", dest="reference", action="store_const", const=None,
                    help="do not draw the record-#40 baseline")
    ap.add_argument("--minutes", action="store_true", help="x axis in minutes, not seconds")
    ap.add_argument("--target", type=float, default=3.28,
                    help="draw the speedrun's val-loss target (0 to omit)")
    ap.add_argument("--ymin", type=float, default=None)
    ap.add_argument("--ymax", type=float, default=None)
    ap.add_argument("--ylog", action="store_true")
    ap.add_argument("--xlog", action="store_true")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--dark", action="store_true", help="render on the dark surface")
    ap.add_argument("--both-themes", action="store_true",
                    help="render each figure on both surfaces")
    args = ap.parse_args()

    if not args.steps_csv.exists():
        raise SystemExit(f"{args.steps_csv} not found -- run parse_logs.py first")
    runs = load_steps(args.steps_csv)
    if args.only:
        keep = {o.lower() for o in args.only}
        runs = OrderedDict((k, v) for k, v in runs.items()
                           if (v["optimizer"] or "").lower() in keep)
        if not runs:
            raise SystemExit(f"no runs matching {args.only}")
    meta = load_runs_meta(args.runs_csv or args.steps_csv.with_name("runs.csv"))
    reference = load_reference(args.reference) if args.reference else None
    if args.reference and reference is None:
        print(f"  note: {args.reference} not found -- no record-#40 baseline drawn")

    args.outdir.mkdir(parents=True, exist_ok=True)
    themes = [("light", _LIGHT)]
    if args.dark:
        themes = [("dark", _DARK)]
    if args.both_themes:
        themes = [("light", _LIGHT), ("dark", _DARK)]

    made = []
    for tname, theme in themes:
        for which in ("val", "train"):
            for xaxis in ("steps", "time"):
                fig = make_figure(runs, which, xaxis, theme, args, meta, reference)
                if fig is None:
                    continue
                suffix = "" if tname == "light" else "_dark"
                stem = f"{which}_{args.metric}_vs_{xaxis}{suffix}"
                for ext in ("pdf", "png"):
                    out = args.outdir / f"{stem}.{ext}"
                    fig.savefig(out, facecolor=theme["surface"], bbox_inches="tight")
                    made.append(out)
                fig.clf()

    print(f"{len(runs)} run(s): " + ", ".join(r["label"] for r in runs.values()))
    for p in made:
        print(f"  wrote {p}")
    print("\nthe table view for these figures is runs.csv / steps.csv next to them")


if __name__ == "__main__":
    main()
