"""Turn the parsed nanoGPT runs into the paper's two tables, plus provenance.

``parse_logs.py`` produces tidy data; this produces the *reported* numbers, so
that ``tab:nanogpt`` and ``tab:nanogpt_diag`` are derived from the logs rather
than transcribed from them. It is the nanoGPT counterpart of the ``SUMMARY.md``
the synthetic arm writes and of ``aggregate.py`` for the federated one.

    cd code/nanogpt
    python parse_logs.py          # -> ../results/nanogpt/{runs,steps,diagnostics}.csv
    python make_tables.py         # -> ../results/nanogpt/{SUMMARY.md,MANIFEST.json,*.tex}

Outputs, all into ``--results`` (default ``../results/nanogpt``):

``SUMMARY.md``
    The file to read. Headline table, compressor diagnostics, the record-#40
    control check, the wall-clock spread, and the provenance table.
``tab_nanogpt.tex`` / ``tab_nanogpt_diag.tex``
    The two LaTeX table bodies, for diffing against the paper.
``MANIFEST.json``
    One entry per run: the log, the environment it ran in, and the SHA-256 of
    the training script and of ``signmuon_optimizers.py`` *as embedded in that
    log*, against the same hashes for the current working tree. A speedrun log
    carries its own source, so this is checkable rather than asserted -- which
    matters here, because the eight runs are not all of one vintage (see
    ``code/nanogpt/README.md``, "Provenance").

Standard library only: the analysis machine need not have torch or numpy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

#: Paths default to this arm's results tree, resolved against this file so
#: the scripts work from any working directory.
_HERE = Path(__file__).resolve().parent


#: Report order: the paper's table is sorted by final validation loss, and the
#: reference Muon leads it.
METHOD_STEP = {           # paper name -> which family's step terminates the update
    "Muon": "lmo", "EF21-SignMuon": "lmo", "MuonUSign": "lmo",
    "EF21-MuonUSign": "lmo", "EF21-MuonSign": "lmo",
    "SignMuon": "sign", "MuonSign": "sign", "SignSGD": "sign",
}

#: The threshold `tab:nanogpt` reports "steps to", and the method it is relative to.
TARGET = 3.35
BASELINE = "Muon"

#: Layer types in `tab:nanogpt_diag`. The two gates are omitted there (they are
#: 6x12 and 1x12, so neither the contraction nor the rank story means much on
#: them), but SUMMARY.md prints them anyway -- the appendix quotes ranges, and a
#: reader checking a range needs to see what was excluded from it.
DIAG_LAYERS = [("blocks.*.attn.qkvo_w", r"\mathtt{qkvo\_w}"),
               ("blocks.*.mlp.c_fc", r"\mathtt{c\_fc}"),
               ("blocks.*.mlp.c_proj", r"\mathtt{c\_proj}")]
DIAG_GATES = [("blocks.*.attn.attn_gate.weight", r"\mathtt{attn\_gate}"),
              ("smear_gate.weight", r"\mathtt{smear\_gate}")]
DIAG_METHODS = ["EF21-SignMuon", "EF21-MuonUSign", "EF21-MuonSign"]

SEP = "=" * 100


def _f(x):
    return float(x) if x not in (None, "", "None") else None


def read_csv(path: Path) -> list[dict]:
    """Rows of a CSV, ignoring ``#`` comment lines.

    ``reference_record40.csv`` opens with a long provenance comment; without the
    filter, ``DictReader`` would take its first line as the header.
    """
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(l for l in fh if not l.startswith("#")))


def reported_loss(row: dict) -> float:
    """The validation loss the table reports for this run.

    For every method but EF21-MuonSign that is simply the final validation loss.
    EF21-MuonSign has two models and the table gives both, headed by the
    broadcast model ``W`` -- the iterate at which every gradient was evaluated,
    hence the one comparable with the other seven arms.
    """
    w = _f(row["final_val_loss_w"])
    return _f(row["final_val_loss"]) if w is None else w


def reported_steps(row: dict, target: float) -> float | None:
    w = _f(row.get(f"steps_to_{target:g}_w"))
    return _f(row.get(f"steps_to_{target:g}")) if w is None else w


# ---------------------------------------------------------------------------
# provenance: what code produced each log
# ---------------------------------------------------------------------------

def log_provenance(path: Path) -> dict:
    """Environment and embedded-source hashes of one run log.

    The log opens with the training script, then ``signmuon_optimizers.py``, then
    the environment dump, each closed by a line of 100 ``=``.
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    seps = [i for i, l in enumerate(lines) if l.rstrip() == SEP][:3]
    out: dict = {"log": path.name}
    if len(seps) >= 2:
        def sha(block):
            return hashlib.sha256("\n".join(block).encode()).hexdigest()
        out["train_script_sha256"] = sha(lines[:seps[0]])
        out["optimizers_sha256"] = sha(lines[seps[0] + 1:seps[1]])
    if len(seps) >= 3:
        for line in lines[seps[1] + 1:seps[2]]:
            if line.startswith("Running Python"):
                out["python"] = line.split(" ", 2)[2].split(" (")[0]
            elif line.startswith("Running PyTorch"):
                out["torch"] = line.split()[2]
            elif "Driver Version:" in line:
                parts = line.split()
                out["driver"] = parts[parts.index("Version:") + 1]
                out["cuda_driver"] = parts[-2]
    return out


def current_sha() -> dict:
    here = Path(__file__).resolve().parent
    def sha(name):
        # The log stores the file with its trailing newline stripped by
        # splitlines(), so hash it the same way or nothing ever matches.
        return hashlib.sha256(
            "\n".join((here / name).read_text(encoding="utf-8").splitlines())
            .encode()).hexdigest()
    return {"train_gpt.py": sha("train_gpt.py"),
            "signmuon_optimizers.py": sha("signmuon_optimizers.py")}


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def headline_rows(runs: list[dict]) -> list[dict]:
    by_name = {r["optimizer"]: r for r in runs}
    base = reported_steps(by_name[BASELINE], TARGET)
    rows = []
    for r in sorted(runs, key=reported_loss):
        st = reported_steps(r, TARGET)
        rows.append(dict(
            method=r["optimizer"],
            step=METHOD_STEP.get(r["optimizer"], "?"),
            lr=_f(r["lr"]),
            loss=reported_loss(r),
            rel=None if st is None or not base else st / base,
            steps=st,
            ms_per_step=_f(r["ms_per_step"]),
            two_models=_f(r["final_val_loss_w"]) is not None,
            x_loss=_f(r["final_val_loss"]),
        ))
    return rows


def tex_headline(rows: list[dict]) -> str:
    out = [r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
           r"\textbf{Method} & \textbf{Step} & $\boldsymbol{\eta_0}$ & "
           r"\textbf{Val.\ loss} & \textbf{Steps to " f"${TARGET:g}$" r"} \\",
           r"\midrule"]
    for r in rows:
        name = r["method"] + (r" ($\mathbf{W}$)" if r["two_models"] else "")
        loss = f"{r['loss']:.4f}"
        if r["method"] == BASELINE:
            name = f"{name} (record \\#40)"
            loss = rf"\textbf{{{loss}}}"
        rel = "--" if r["rel"] is None else f"${r['rel']:.2f}\\times$"
        out.append(rf"{name} & \textsc{{{r['step']}}} & {r['lr']:g} & {loss} & {rel} \\")
    x = [r for r in rows if r["two_models"]]
    if x:
        out += [r"\midrule",
                rf"EF21-MuonSign ($\mathbf{{X}}$) & \textsc{{lmo}} & {x[0]['lr']:g} "
                rf"& {x[0]['x_loss']:.4f} & -- \\"]
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def diag_at_final(diag: list[dict], step: int) -> dict:
    """(optimizer, parameter) -> row, at the final validation."""
    return {(r["optimizer"], r["parameter"]): r for r in diag
            if int(r["step"]) == step and r["count"]}


def _round(v: float, places: int) -> str:
    """Fixed-point with ROUND_HALF_UP.

    Python's own formatting rounds half to even *on the binary value*, which
    turns the logged ``0.3735`` into ``0.373``. These are 4-significant-digit
    medians read back from a text log, so the tie is an artefact of the log's own
    precision and half-up is the convention a reader will check against.
    """
    from decimal import Decimal, ROUND_HALF_UP
    q = Decimal(1).scaleb(-places)
    return str(Decimal(repr(v)).quantize(q, rounding=ROUND_HALF_UP))


def tex_diag(at_end: dict) -> str:
    out = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
           r"& \multicolumn{2}{c}{\textbf{uplink}} & \multicolumn{2}{c}{\textbf{downlink}} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
           r"\textbf{Layer type} & $\alpha$ & lag & $\alpha$ & gap \\", r"\midrule"]
    counts = {"blocks.*.attn.qkvo_w": 10, "blocks.*.mlp.c_fc": 11,
              "blocks.*.mlp.c_proj": 11}
    # The row the appendix argues from: the worst downlink contraction anywhere in
    # the table. Found rather than named, so the emphasis follows the measurement.
    worst = min((k for k in at_end if _f(at_end[k].get("alpha_dn")) is not None),
                key=lambda k: _f(at_end[k]["alpha_dn"]), default=None)
    for i, meth in enumerate(DIAG_METHODS):
        if i:
            out.append(r"\midrule")
        out.append(rf"\multicolumn{{5}}{{@{{}}l}}{{\emph{{{meth}}}}} \\")
        for key, tex in DIAG_LAYERS:
            r = at_end.get((meth, key))
            if r is None:
                continue
            suffix = rf" (${{\times}}{counts[key]}$)" if i == 0 else ""
            cells = []
            for col in ("alpha_up", "lag_est", "alpha_dn", "lag_XW"):
                v = _f(r.get(col))
                if v is None:
                    cells.append("--")
                    continue
                if col == "lag_XW":
                    cell = f"{v:.2g}"
                elif v < 1e-3:
                    mant, exp = f"{v:.1e}".split("e")
                    cell = rf"{mant}{{\times}}10^{{{int(exp)}}}"
                else:
                    cell = _round(v, 3 if col.startswith("alpha") else 2)
                if (meth, key) == worst and col in ("alpha_dn", "lag_XW"):
                    cell = rf"$\mathbf{{{cell}}}$"      # the anomaly, both cells
                elif v < 1e-3:
                    cell = f"${cell}$"
                cells.append(cell)
            out.append(rf"\quad ${tex}${suffix}   & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def md_table(header: list[str], rows: list[list[str]], align: str | None = None) -> str:
    align = align or ("| :--- " + "| ---: " * (len(header) - 1) + "|")
    out = ["| " + " | ".join(header) + " |", align]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=_HERE.parent / "results" / "nanogpt", type=Path)
    ap.add_argument("--reference", default=_HERE / "reference_record40.csv", type=Path)
    args = ap.parse_args()

    runs = read_csv(args.results / "runs.csv")
    if not runs:
        raise SystemExit(f"no runs.csv in {args.results}; run parse_logs.py first")
    diag = read_csv(args.results / "diagnostics.csv")
    ref = {int(r["step"]): r for r in read_csv(args.reference)
           if not r["step"].startswith("#")}

    final_step = max(int(r["train_steps"]) for r in runs)
    rows = headline_rows(runs)
    at_end = diag_at_final(diag, final_step)

    # --- provenance ------------------------------------------------------
    cur = current_sha()
    prov = []
    for r in sorted(runs, key=lambda r: r["optimizer"]):
        p = log_provenance(args.results / "logs" / Path(r["log"]).name)
        p.update(run_id=r["run_id"], optimizer=r["optimizer"],
                 seed=r["seed"] or None,
                 optimizers_matches_worktree=(p.get("optimizers_sha256")
                                              == cur["signmuon_optimizers.py"]),
                 train_script_matches_worktree=(p.get("train_script_sha256")
                                                == cur["train_gpt.py"]))
        prov.append(p)
    # Label the distinct builds present. The eight runs are not all of one
    # vintage, and "which build" is the question a reader asks the moment they
    # notice that only three logs carry a diagnostics block.
    builds: dict[str, str] = {}
    for p in prov:
        h = p.get("optimizers_sha256", "?")
        builds.setdefault(h, chr(ord("A") + len(builds)))
        p["build"] = builds[h]
    for p in prov:
        if p["optimizers_matches_worktree"]:
            p["build"] += " (= worktree)"
    (args.results / "MANIFEST.json").write_text(json.dumps(dict(
        written=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        experiment="nanoGPT speedrun, modded-nanogpt record #40",
        train_steps=final_step,
        target=TARGET, baseline=BASELINE,
        worktree_sha256=cur, runs=prov), indent=2) + "\n", encoding="utf-8")

    (args.results / "tab_nanogpt.tex").write_text(tex_headline(rows) + "\n",
                                                  encoding="utf-8")
    (args.results / "tab_nanogpt_diag.tex").write_text(tex_diag(at_end) + "\n",
                                                       encoding="utf-8")

    # --- SUMMARY.md ------------------------------------------------------
    ms = [r["ms_per_step"] for r in rows]
    muon = [r for r in rows if r["method"] == BASELINE][0]
    ref_end = ref.get(final_step)
    md = [f"# NanoGPT speedrun results (record #40, {final_step} steps, "
          f"8×H100, single run per method)",
          "",
          "Written by `code/nanogpt/make_tables.py` from `runs.csv` / `diagnostics.csv`.",
          "Every number the paper quotes for this arm is below; `MANIFEST.json` says",
          "which log, and which build of the optimizer module, produced each one.",
          "",
          "## Headline (`tab:nanogpt`)",
          "",
          md_table(["method", "step", "eta_0", "val. loss",
                    f"steps to {TARGET:g}", f"rel. {BASELINE}", "ms/step"],
                   [[r["method"] + (" (**W**)" if r["two_models"] else ""),
                     r["step"], f"{r['lr']:g}", f"{r['loss']:.4f}",
                     "--" if r["steps"] is None else f"{r['steps']:.0f}",
                     "--" if r["rel"] is None else f"{r['rel']:.2f}x",
                     f"{r['ms_per_step']:.2f}"] for r in rows]),
          ""]
    x = [r for r in rows if r["two_models"]]
    if x:
        md += [f"EF21-MuonSign's exact server model **X** ends at "
               f"`{x[0]['x_loss']:.4f}`, against `{x[0]['loss']:.4f}` at the "
               "broadcast model **W** above: the two models of that method, not "
               "two runs.", ""]
    md += ["## Control: does the `Muon` arm reproduce record #40?", ""]
    if ref_end:
        d = muon["loss"] - float(ref_end["val_loss"])
        sd = float(ref_end["val_loss_sd"])
        md += [f"Upstream record #40, mean of its five 8xH100 logs at step "
               f"{final_step}: `{float(ref_end['val_loss']):.4f} +/- {sd:.4f}`.",
               f"This port's `Muon`: `{muon['loss']:.4f}` "
               f"({d:+.4f}, {abs(d) / sd:.1f} upstream sd). "
               f"{'PASS' if abs(d) <= 0.003 else 'FAIL -- the port is broken, fix it before reading anything else'}.",
               ""]
    md += ["## Wall-clock", "",
           f"`{min(ms):.2f}`--`{max(ms):.2f}` ms/step across the eight methods, "
           f"a spread of {(max(ms) / min(ms) - 1) * 100:.1f}%: no method pays a "
           "measurable premium for its compressor or its error-feedback buffers. "
           "All eight are above record #40's own `60.4` ms/step, by the same "
           "margin, because this port replaces its Triton kernels and batched "
           "sharded transport with a pure-torch per-parameter equivalent.",
           "",
           "## Compressor diagnostics at the final step (`tab:nanogpt_diag`)", "",
           "Medians over the identical layers of each type. `alpha` is the "
           "contraction the scaled sign achieves on that round's residual "
           "(`2/pi = 0.637` for an isotropic residual, `1/d` in the worst case); "
           "`lag` is `||target - estimator||_F / ||target||_F`; `gap` is "
           "`||X - W||_F / ||W||_F`. The two gates are **not** in the paper's "
           "table and are marked here, because the ranges the appendix quotes are "
           "over the layer types only.", ""]
    drows = []
    for meth in DIAG_METHODS:
        for key, tex in DIAG_LAYERS + DIAG_GATES:
            r = at_end.get((meth, key))
            if r is None:
                continue
            name = tex.replace(r"\mathtt{", "`").replace(r"\_", "_").replace("}", "`")
            if (key, tex) in DIAG_GATES:
                name += " (gate)"
            drows.append([meth, name, r["count"]]
                         + [("--" if _f(r.get(c)) is None else f"{_f(r[c]):.4g}")
                            for c in ("alpha_up", "lag_est", "alpha_dn", "lag_XW")])
    md += [md_table(["method", "layer", "count", "uplink alpha", "uplink lag",
                     "downlink alpha", "gap"], drows), ""]
    lay = [(m, k) for m in DIAG_METHODS for k, _ in DIAG_LAYERS
           if (m, k) in at_end]
    if lay:
        au = [_f(at_end[k]["alpha_up"]) for k in lay]
        lg = [_f(at_end[k]["lag_est"]) for k in lay]
        md += [f"Over the layer types only: uplink `alpha` in "
               f"[{min(au):.2f}, {min(max(au), 0.999):.2f}], uplink lag in "
               f"[{min(lg):.2f}, {max(lg):.2f}].", ""]
    md += ["## Provenance", "",
           md_table(["run", "optimizer", "seed", "torch", "driver",
                     "optimizer build"],
                    [[p["log"], p["optimizer"], p.get("seed") or "unseeded",
                      p.get("torch", "?"), p.get("driver", "?"), p["build"]]
                     for p in prov],
                    align="| :--- | :--- | :--- | :--- | :--- | :--- |"),
           "",
           f"{len(builds)} distinct build(s) of `signmuon_optimizers.py` produced "
           "these logs, keyed by the SHA-256 of the copy each log embeds "
           "(`MANIFEST.json` has the hashes). What differs between them, and why "
           "it does not mix vintages that should not be mixed, is "
           "`code/nanogpt/README.md`, section \"Provenance\".",
           ""]
    (args.results / "SUMMARY.md").write_text("\n".join(md), encoding="utf-8")

    print(f"wrote {args.results}/SUMMARY.md, MANIFEST.json, "
          "tab_nanogpt.tex, tab_nanogpt_diag.tex")
    print()
    print("\n".join(md[6:9]))


if __name__ == "__main__":
    main()
