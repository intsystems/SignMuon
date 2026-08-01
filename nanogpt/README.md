# SignMuon / MuonSign / EF21 optimizers on the NanoGPT speedrun (record #40)

The paper's six matrix-aware 1-bit optimizers, plus `SignSGD` and a reference
`Muon`, ported into
[modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) in the
**distributed data-parallel** setting (*not* the federated setting).

The base is **record #40** (`2025-10-04_Backout`, PR #140, 140.7 s = 2.345 min
on 8×H100), the last record before NorMuon (#41). It carries every pre-#40
improvement (Polar Express orthogonalization, Flash Attention 3, YaRN, sparse
attention gate, token smearing, dropped first attn/MLP layers, BF16
cross-entropy, the backout skip) while **Muon is still a clean, separable
optimizer** whose orthogonalization step is where the sign/EF21 variants inject.
Excluding NorMuon and what is co-tuned to it (#41 through #84) keeps every
hyperparameter tuned for plain Muon. See "Why record #40" below.

## Files

Code lives here; **every artefact of a run lives in
[`../results/nanogpt/`](../results/nanogpt/)**, beside `results/synthetic/` and
`results/federated/`. Start at
[`../results/nanogpt/SUMMARY.md`](../results/nanogpt/SUMMARY.md).

| file | what it is |
|------|------------|
| `signmuon_optimizers.py` | The eight optimizers. **Single source of truth, shared by both training scripts**, pure-torch (Polar Express LMO, no Triton), unit-testable off-GPU. |
| `train_gpt.py` | The **8×H100** script: record #40 (Flash Attention 3 + FP8) with `Muon` replaced by the `SIGNMUON_OPT=` selector. Used for the final runs. |
| `train_gpt_a100.py` | The **single-A100** build: `train_gpt.py` with the two Hopper-only pieces (FA3, FP8) swapped for Ampere-safe equivalents, each under a `# ===== [A100 DIFF #k] ...` banner. |
| `test_signmuon_optimizers.py` | Portable CPU test: each update recurrence equals the numpy paper reference (`../counterexamples/optimizers.py`); the per-layer EF21 compressor scale on the merged `qkvo_w`; the per-layer LR multipliers (record #40's aspect factor for the LMO family, unit gain for both, agreement with `../common/lr_scaling.py`). |
| `test_distributed_sharding.py` | gloo/CPU test: the sharded `step()` equals a single-process centralized run, over both padding regimes. |
| `setup_env.sh` | Builds the venv from `requirements.txt` and verifies it with a real FA3 fetch. **Start here.** |
| `run_all.sh` | Launches the eight hero runs, one per optimizer, at their starting learning rates. |
| `parse_logs.py` | Raw logs -> `runs.csv` (one row per run) + `steps.csv` (tidy per-step) + `diagnostics.csv` + `runs.json`. |
| `make_tables.py` | Those CSVs -> `SUMMARY.md`, `MANIFEST.json` and the two LaTeX table bodies, making `tab:nanogpt` and `tab:nanogpt_diag` *derived* rather than transcribed. |
| `plot_article.py` | The paper's three figures (`fig:nanogpt`, `fig:nanogpt_appendix`, `fig:nanogpt_diag`). Run IDs are pinned in the `RUNS` table (`plot_article.py`, line 91), with the reason in the comment above it. |
| `plot_runs.py` | Exploratory loss-vs-steps and loss-vs-time figures from `steps.csv`, with record #40's curve behind every validation figure. Not used by the paper. |
| `reference_record40.csv` | Record #40's published validation curve, averaged over its **five** upstream 8×H100 logs (`3.2780 ± 0.0009` at step 2330, `140.7 s`). The pass/fail line for the `Muon` control. |
| `data/cached_fineweb10B.py` | Downloads the pre-tokenized FineWeb-10B GPT-2 tokens (same stream for every record). |
| `requirements.txt` | Python deps (mirrors upstream; `torch==2.10`). |
| `train_gpt_rec40_reference.py` | The **verbatim record-#40 source** (from its run log), for provenance and diffing. Not wired to the optimizer knob. |

Outside the training scripts: `parse_logs.py` and `make_tables.py` are pure
standard library; `plot_runs.py` adds matplotlib; `plot_article.py` adds
matplotlib, numpy and `../common/plotting.py`; `data/cached_fineweb10B.py` needs
`huggingface_hub`; both tests need torch (the math test also numpy). Only
`run_all.sh` needs a GPU.

## Setup

```bash
cd code/nanogpt
bash setup_env.sh                         # venv + deps + a real FA3 fetch, verified
source .venv/bin/activate
python data/cached_fineweb10B.py 9        # ~900M train tokens + the val chunk
```

Both pins are in `requirements.txt`, and the venv itself is not optional:

* **Never `pip install` into the system python.** apt-installed packages carry
  no pip RECORD, so pip cannot remove them:
  `Cannot uninstall cryptography 41.0.7 -- no RECORD file was found`.
* **`kernels<0.13`.** 0.14+ needs `huggingface-hub>=1.10` (a major bump); 0.16+
  also `sigstore>=4`, which pulls cryptography / pyOpenSSL / rfc3161-client and
  triggers the collision above. `<0.13` fetches FA3 identically.
* **`torch==2.10.0` from PyPI, not the container's.** `kernels` downloads a
  *prebuilt* FA3 binary and the hub carries only **cu126 / cu128 / cu130**
  variants (torch 2.8-2.12, x86_64 and aarch64). An NGC container's **CUDA
  13.1** build (`2.10.0a0+...nv26.01`) would need a nonexistent `cu131` variant;
  PyPI's is `torch210-cxx11-cu128-x86_64-linux`, and cu128 runs on the newer
  driver.

Record #40 ran **torch `2.10.0.dev20250926+cu126`, Python 3.10.12, Triton 3.5.0,
CUDA 12.6**; the paper's eight runs ran **torch `2.10.0+cu128` (PyPI, CUDA
12.8), Python 3.12.3** on an 8×H100 SXM node, driver `595.71.05`, read out of
the logs by `make_tables.py` into
[`../results/nanogpt/MANIFEST.json`](../results/nanogpt/MANIFEST.json).

If FA3 will not resolve, substitute FlexAttention on the same 8 GPUs:

```bash
NPROC=8 SCRIPT=train_gpt_a100.py bash run_all.sh
```

Both substitutions sit outside the optimizer, so the sign/EF21 comparison is
unchanged and only wall-clock differs. `run_all.sh` repeats that check itself
(imports, GPU count, data shards, a real FA3 fetch) on one process before
launching and aborts if the first runs all fail; `SKIP_PREFLIGHT=1` bypasses it.
It uses one interpreter throughout (`$PYTHON -m torch.distributed.run`, not a
bare `torchrun`); set `PYTHON=/path/to/venv/bin/python` to be explicit. The
scripts read `data/fineweb10B/fineweb_{train,val}_*.bin` relative to the current
directory, or from `DATA_PATH=/abs/path`; a full run consumes ~611M tokens (2330
steps × 262144), so 7+ shards.

## Running

`SIGNMUON_OPT` selects the optimizer for the hidden matrix weights. Embeddings,
scalars and the LM head always use `DistAdam`, as in every Muon speedrun. The
`attn_gate` and `smear_gate` matrices go to the selected optimizer, as in #40;
the skip gates are a slice of the 1-D `scalars` tensor and stay with `DistAdam`.

```bash
# --- all eight, one run each, at their starting LRs (the main experiment) ---
bash run_all.sh
NANOGPT_ITERS=200 bash run_all.sh        # cheap smoke pass first, once

# --- a single method on 8×H100 (record #40, FA3 + FP8) ---
SIGNMUON_OPT=EF21-MuonUSign torchrun --standalone --nproc_per_node=8 train_gpt.py

# --- single A100 (imitates the 8×H100 run; ~8× slower wall-clock, same curves) ---
SIGNMUON_OPT=EF21-MuonUSign python train_gpt_a100.py
#   or:  NPROC=1 SCRIPT=train_gpt_a100.py bash run_all.sh
```

Valid `SIGNMUON_OPT`: `SignMuon`, `EF21-SignMuon`, `MuonUSign`, `MuonSign`,
`EF21-MuonUSign`, `EF21-MuonSign`, `SignSGD`, `Muon` (default `Muon`).

| env var | meaning |
|---|---|
| `SIGNMUON_LR`, `SIGNMUON_MOMENTUM`, `SIGNMUON_WD` | override the method's hyperparameters |
| `SIGNMUON_LR_SCALING` | per-layer rule: `unit-gain` (default), `semantic`, `mup`, `legacy`, `none` |
| `SIGNMUON_SEED` | RNG seed (default `0`); see **Provenance**, the eight logged runs predate it |
| `SIGNMUON_RUN_ID` | override the log name (default `<Opt>_lr<lr>_<hash>`) |
| `NANOGPT_ITERS`, `NANOGPT_VAL_EVERY` | shorten a run / change the validation cadence |
| `LOG_DIR`, `DATA_PATH` | where logs are written / where the `.bin` shards live |
| `SIGN_PROBE_LR`, `SKIP_PREFLIGHT` | `run_all.sh` only: extra sign-method runs at that LR / skip the environment check |

Each log records `train_gpt*.py` **and** `signmuon_optimizers.py` verbatim, so
it reproduces the optimizer definitions despite the import, plus a
machine-readable `RUNMETA {...}` / `RUNEND {...}` JSON header, the per-layer LR
multiplier table, **per-step training loss**, per-validation compressor
diagnostics and the validation points. A non-finite loss logs `DIVERGED` and
stops the run, which the tooling reports as a result rather than a crash.
`EF21-MuonSign` has two models and reports both at each validation: `val_loss`
is the exact server model `X` (the iterate the convergence corollary bounds),
`val_loss_W` the sign-compressed broadcast model `W` (the iterate every gradient
was evaluated at), ~2.2 nats apart.

### Control: reproducing record #40

`run_all.sh` runs `Muon` first because it **is** record #40's optimizer and must
land on record #40's published curve, checked in as `reference_record40.csv`
(mean of upstream's five 8×H100 logs) and drawn behind every validation figure:

| step | 250 | 500 | 1000 | 1500 | 2000 | 2330 |
|---|---|---|---|---|---|---|
| upstream val loss | 4.0898 | 3.8203 | 3.5766 | 3.4509 | 3.3307 | **3.2780 ± 0.0009** |

A `Muon` run outside roughly `3.278 ± 0.003` at step 2330 indicates a broken
**port**, not a broken optimizer. Wall-clock sits above the record's `140.7 s` /
`60.4 ms per step` because this port replaces #40's Triton kernels and batched
sharded transport with a pure-torch, per-parameter equivalent (identical
arithmetic, more kernel launches), a cost all eight methods pay equally.

## Results, and how to regenerate them

Everything a run produces is under [`../results/nanogpt/`](../results/nanogpt/):

| path | what it is |
|---|---|
| `SUMMARY.md` | **read this first.** Both paper tables, the record-#40 control check, the wall-clock spread, and the provenance table. |
| `logs/` | the eight raw speedrun logs (each embeds its own source; ~300 kB apiece) |
| `runs.csv` | one row per run: final/best val loss (and `_w` for the broadcast model), `steps_to_<target>` / `ms_to_<target>`, ms/step, peak memory, seed, GPU |
| `steps.csv` | tidy `run_id, optimizer, lr, lr_scaling, step, wallclock_ms, train_loss, val_loss, val_loss_w` |
| `diagnostics.csv` | one row per (run, validation, layer type): compressor contraction `alpha_up`/`alpha_dn`, estimator lag, server/broadcast gap, per-block `mean\|grad\|` |
| `runs.json` | `runs.csv` plus the raw `RUNMETA`/`RUNEND` dicts |
| `MANIFEST.json` | per-run provenance: environment, and the SHA-256 of the source each log embeds |
| `tab_nanogpt.tex`, `tab_nanogpt_diag.tex` | the two LaTeX table bodies, for diffing against the paper |
| `figures/` | `fig_nanogpt_{main,appendix,diag}.{pdf,png}` |

Three commands rebuild all of it from the logs, on any machine, with no GPU:

```bash
cd code/nanogpt
python parse_logs.py        # logs        -> runs.csv, steps.csv, diagnostics.csv, runs.json
python make_tables.py       # those CSVs  -> SUMMARY.md, MANIFEST.json, tab_nanogpt*.tex
python plot_article.py      # those CSVs  -> figures/fig_nanogpt_{main,appendix,diag}.*
```

`parse_logs.py` also prints the summary table. `steps_to_<target>` is linearly
interpolated between validation points, so it does not depend on where the
coarse (every-250-step) grid falls; the `_w` columns repeat the crossing on
`EF21-MuonSign`'s broadcast model, whose `X` column never reaches these targets.

The headline numbers, with the full table in `SUMMARY.md`:

| method | step | η₀ | val. loss | steps to 3.35 | rel. Muon |
|---|---|---|---|---|---|
| `Muon` (== record #40) | lmo | 0.06 | **3.2785** | 1903 | 1.00× |
| `EF21-SignMuon` | lmo | 0.06 | 3.2860 | 1949 | 1.02× |
| `SignMuon` | sign | 0.03 | 3.2881 | 1942 | 1.02× |
| `MuonUSign` | lmo | 0.06 | 3.2959 | 1990 | 1.05× |
| `EF21-MuonUSign` | lmo | 0.06 | 3.3203 | 2157 | 1.13× |
| `EF21-MuonSign` (`W`) | lmo | 0.06 | 3.3213 | 2164 | 1.14× |
| `MuonSign` | sign | 0.03 | 3.3249 | 2175 | 1.14× |
| `SignSGD` | sign | 0.03 | 3.4049 | never | n/a |
| `EF21-MuonSign` (`X`) | lmo | 0.06 | 5.5198 | never | n/a |

For exploratory work rather than the paper's figures:

```bash
python plot_runs.py                                     # -> figures/exploratory/
python plot_runs.py --both-themes --anytime --minutes
```

`plot_runs.py` writes four figures (PDF + PNG; eight with `--both-themes`):
{validation, training} loss vs {steps, wall-clock}. **Steps** compares the
methods as *optimizers* (same data, same updates); **wall-clock** as *systems*,
on the speedrun's own `train_time` clock, excluding validation and compilation.
The default is the raw curve, not the running-minimum "anytime best" envelope,
which is monotone by construction and hides the instability under test;
`--anytime` overlays it dashed, `--metric perplexity` relabels to `exp(loss)`,
`--only Muon SignMuon` restricts the figure. Its output is gitignored; the
paper's three figures are not.

## Provenance: which build produced which log

A speedrun log embeds its own source, so `make_tables.py` checks provenance
rather than asserting it: `MANIFEST.json` carries the SHA-256 of `train_gpt.py`
and of `signmuon_optimizers.py` *as recorded in each log*, and `SUMMARY.md`
labels them. There are two builds and two differences from the working tree,
none invalidating the comparison.

**Build B** produced the five non-EF21 methods (`Muon`, `SignSGD`, `SignMuon`,
`MuonUSign`, `MuonSign`): no compressor diagnostics, and the EF21 compressor's
`mean|·|` taken over the whole merged `qkvo_w` rather than per Q/K/V/O block.
**Build A** produced the three EF21 methods, re-run after that fix. The fix is
confined to `_scaled_sign`, which only the EF21 methods call, so the build-B
five were not re-run and the tables mix vintages without any number changing.
Both ran on the same node the same day, with identical torch and driver in
`MANIFEST.json`.

**The working tree differs from both builds, in two files.** In
`signmuon_optimizers.py` the sole delta is `sign_pm1`, which maps exact zeros to
an independent random ±1 as the paper's convention requires; the logged builds
used `torch.sign`, i.e. `sign(0) = 0`. It affects one place on this model:
record #40 zero-initializes `c_proj` and the `O` block of `qkvo_w`, so at step 0
the gradient of every `c_fc` and of `Q/K/V` is *exactly* zero. There the three
sign-terminated methods (`SignMuon`, `MuonSign`, `SignSGD`) now take a full
random ±1 step instead of none, and `MuonUSign` takes `PE(random ±1)`, an
orthogonal step rather than a random one; the three EF21 methods are unaffected,
their compressor scale `mean|Δ|` being zero as well. In `train_gpt.py`,
`MANIFEST.json` records `train_script_matches_worktree: false` for all eight
runs, the deltas being the `SIGNMUON_SEED` plumbing and the `LOG_DIR` default,
moved from `logs` to `../results/nanogpt/logs`; neither touches the model, the
data order or the optimizer. A re-run will therefore not reproduce these curves
bit-for-bit, but should reproduce them well within record #40's ±0.0009 seed
spread.

**The eight runs are unseeded.** Upstream modded-nanogpt seeds nothing, and
neither did this port when they were made: `MANIFEST.json` records `seed: null`
for all eight, meaning whatever torch drew at process start, not seed 0.
`SIGNMUON_SEED` (default `0`) now pins the RNG, and only the RNG: cuDNN/cuBLAS
autotuning and reduction order are left alone, since deterministic kernels would
change the wall-clock the table reports.

## How the single A100 imitates 8×H100

**The batch/step math is unchanged; this is record #40's own fewer-GPUs path.**
Record #40 sets `grad_accum_steps = 8 // world_size`, so on one GPU
(`world_size == 1`) it accumulates 8 micro-batches per optimizer step and sees
the **same global batch, the same tokens, and the same number of optimizer
steps** as the 8-rank run. The missing `1/8` factor in `reduce_scatter(op=AVG)`
(a no-op on one process) washes out because **Muon and Adam are
scale-invariant** (Polar Express normalizes the update; Adam is `m/√v`).
Upstream: *"To run experiments on fewer GPUs, simply modify `--nproc_per_node`.
This should not change the behavior of the training."* **The optimizer under
study is byte-for-byte identical on both**, the two scripts importing the same
pure-torch `signmuon_optimizers`; the residual differences are the three
`[A100 DIFF #k]` banners in `train_gpt_a100.py`, all outside the optimizer:

| # | 8×H100 (`train_gpt.py`) | single A100 (`train_gpt_a100.py`) | touches |
|---|---|---|---|
| 1 | 8 ranks | 1 rank, `grad_accum_steps=8` (built into #40) + torchrun env defaults | nothing (same batch/steps) |
| 2 | Flash Attention 3 (Hopper-only) | **FlexAttention** with a block mask reproducing #40's mask *exactly* (per-document causal + `bm_size`-token left window; documents delimited by BOS=50256; RoPE/YaRN unchanged) | attention kernel numerics only |
| 3 | FP8 lm_head matmul (Hopper-only) | **bf16** lm_head (`DISABLE_FP8=1`) | lm_head (a DistAdam param), not the methods under study |

A100 curves should therefore track the 8×H100 curves closely, differing only by
the FlexAttention-vs-FA3 kernel and the bf16-vs-FP8 lm_head. **Validate that
swap once, on a GPU:** it is the one change not verifiable on a CPU-only laptop.
The masks are mathematically equivalent (same causal + token window +
per-document structure), but long A100 sweeps should be preceded by a short A/B.
Run `train_gpt_a100.py` on a **single H100** (FlexAttention) against
`train_gpt.py` on 8×H100 (FA3), same `SIGNMUON_OPT`, and confirm the first few
hundred steps' loss curves overlap; `train_gpt_a100.py` is device-agnostic,
which is what isolates the attention kernel. An 80 GB A100 is assumed, so that
one 32 768-token micro-batch fits as on one H100 rank, and `create_block_mask`
is built eagerly per micro-batch, making A100 validation passes slower. If
`_compile=True` in `build_flex_block_mask` errors on a torch build, set it to
`False`.

## The eight optimizers

Let `M` be the (Nesterov EMA) momentum of the **averaged** gradient and `PE(·)`
the Muon Polar Express orthogonalization (record #40's LMO; approximate polar
factor `U Vᵀ`). Every rule is verbatim the centralized algorithm boxes of the
paper and their numpy reference in
[`../counterexamples/optimizers.py`](../counterexamples/optimizers.py).

| name | update | final step |
|------|--------|------------|
| `SignMuon` | `X ← X − η·sign(PE(M))` | sign |
| `EF21-SignMuon` | `d_est ← d_est + mean\|D−d_est\|·sign(D−d_est)`, `D=PE(M)`; `X ← X − η·d_est` | LMO |
| `MuonUSign` | `X ← X − η·PE(sign(M))` | LMO |
| `MuonSign` | `X ← X − η·sign(PE(sign(M)))` | sign |
| `EF21-MuonUSign` | `g_est ← g_est + mean\|M−g_est\|·sign(M−g_est)`; `X ← X − η·PE(g_est)` | LMO |
| `EF21-MuonSign` | uplink EF on `g_est` → exact `X ← X − η·PE(g_est)`; downlink EF compresses `X−W` into the broadcast model `W` | LMO |
| `SignSGD` | `X ← X − η·sign(M)` | sign |
| `Muon` | `X ← X − η·PE(M)` (reference, no compression; == record #40) | LMO |

## Per-layer learning rates

The two families produce step matrices whose norms scale *differently* with
shape, so one global `η` cannot suit both. The paper's appendix (`app:lrscale`)
fixes this with the **unit-gain** criterion: the RMS gain of a step matrix
`s ∈ R^{m×n}` on an isotropic input is `γ(s) = ‖s‖_F / √m`, so equal per-step
gain on every layer gives `λ = √fan_out / ‖s‖_F` and two closed forms:

| family | methods | `‖s‖_F` | `λ` |
|---|---|---|---|
| `lmo` | `Muon`, `MuonUSign`, `EF21-MuonUSign`, `EF21-MuonSign`, `EF21-SignMuon` | `√min(m,n)` | `√max(1, m/n)` |
| `sign` | `SignMuon`, `MuonSign`, `SignSGD` | `√(mn)` | `1/√fan_in` |

Two properties make this the right default. First, **the `lmo` line *is* record
#40**: `√max(1, m/n)` is character-for-character Keller Jordan's shipped aspect
factor `max(1, p.size(-2)/p.size(-1))**0.5`, and on every shape #40 uses (gates
`[1,12]` and `[6,12]`, attention blocks `[768,768]`, MLP matrices `[768,3072]`)
both evaluate to `1.0`, so **`Muon` here is the record verbatim**
(`test_lmo_family_matches_record40_aspect_factor` pins this). Second, **`η₀`
then means one thing for all eight methods**, the per-step RMS gain: at
`η₀ = 0.06` every method takes the same per-entry RMS step on every layer,
`1.08e-3` on the MLP matrices and `1.73e-2` on the gates, whether the step is
`PE(·)` or `± 1`. That is what makes `η₀` transferable from the reference to the
methods under study, and why the sign family starts at `0.03` rather than the
`3e-4` an unscaled implementation needs.

**Starting learning rates** (`train_gpt.py:OPTIMIZER_CONFIG`, all "round" to one
significant digit):

| methods | `η₀` | why |
|---|---|---|
| `Muon`, `MuonUSign`, `EF21-MuonUSign`, `EF21-MuonSign`, `EF21-SignMuon` | **0.06** | the reference's own value, not a guess |
| `SignMuon`, `MuonSign`, `SignSGD` | **0.03** | spectral discount on an entrywise-uniform step |

**The LMO five sit at the reference's LR on purpose.** Their final step is an
orthogonal matrix, or for the EF21 pair an error-feedback estimate of one, with
the same spectral *and* Frobenius norm as Muon's, so there is nothing to
rescale, and the paper's contrasts become matched-hyperparameter: `Muon` vs
`MuonUSign` (the cost of a 1-bit uplink) and `EF21-SignMuon` vs
`EF21-MuonUSign`/`EF21-MuonSign` (Thm 4: EF21 on the LMO *output* diverges, EF21
on the momentum does not) differ only in the rule. `EF21-SignMuon` belongs with
the LMO five because error feedback undoes the quantization: `d_est` is a
full-precision accumulator tracking `PE(M)`, so its operator norm starts at
~1.1× Muon's and decays toward 1.0. Only at Muon's own LR does its divergence
read as the *rule's* fault rather than the step size's.

**Where 0.03 comes from.** A sign step is entrywise uniform, so at equal RMS
gain it is spectrally more aggressive than an orthogonal one, and the framework
these methods are analysed in (Gluon / EF21-Muon) is a *spectral*-norm
framework. Three independent routes agree:

| route | number |
|---|---|
| spectral matching: `‖λ·sign(·)‖_op/‖λ·PE(·)‖_op` = 1.40 (mlp) / 1.86 (attn); one `η₀` must satisfy the tighter | `0.06/1.86 = 0.032` |
| Mishra et al.'s tuned nanoGPT value, mapped in: their Alg. 1 has **no** shape factor, so `η=1e-3` on `d=384` is a global unscaled LR → `η₀ ≈ 0.023`; correcting for their broken schedule (`warmup_iters=2000 > max_iters=1500`, so their LR only ramps 0→7.5e-4) and our 8.5× batch | `≈ 0.032` |
| Lion's "3–10× smaller than AdamW", decomposed: AdamW's `m/√v` is ~0.3 per entry vs a sign step's 1.0, so ~3× of that is pure norm, which unit-gain already handles; residual robustness discount 1–3× | `0.02 – 0.06` |

`1e-4` would be a **global, unscaled** LR from standard-batch, long-schedule
training. Per weight entry `η₀=0.03` means `5.4e-4` (mlp) to `1.1e-3` (attn),
*half* what Muon takes at the record's `0.06`. This codebase is far more
aggressive than standard GPT-2 training: 2330 steps, 262k tokens/step, Muon at
`0.06` (vs ~0.02 typical), Adam at `0.008` with `lr_mul=75` on the embeddings.
The LMO five are pinned by the record; the sign three are the one uncertain
number, bracketed at `0.01–0.04`. `SIGN_PROBE_LR=0.01 bash run_all.sh` adds one
run per sign method at the downside; the tuning ladder is
`0.01, 0.02, 0.03, 0.06, 0.1, 0.2`
(`SIGNMUON_LR=0.1 SIGNMUON_OPT=SignMuon ...`), and the multiplier table in force
is printed into each log.

**Two inherited storage quirks.** The rule reads `(fan_out, fan_in)` off the
*stored* tensor, and record #40 stores the MLP `c_fc` transposed (`[dim, hdim]`,
used as `x @ c_fc`) to share a `reduce_scatter` with the attention weight, so
the semantic fan-in/fan-out are swapped there and both families get a 2× smaller
multiplier than a `[fan_out, fan_in]` reading would. That is what record #40
does for Muon, so it is the default; `SIGNMUON_LR_SCALING=semantic` corrects it
(moving the Muon baseline off the record), and `none`/`legacy`/`mup` serve the
ablation. Record #40 also stores Q/K/V/O in one `[hdim, 4·dim]` parameter but
uses it as `.view(4, hdim, dim)`, so the optimizer orthogonalizes the four
`[hdim, dim]` blocks independently, detected via the model's `.module == "attn"`
tag, while same-shaped MLP weights are orthogonalized as a single matrix. This
is localized to the LMO call, every other op (momentum, sign, EF21) being
elementwise.

## Distributed ≠ federated

**Federated** (`../federated/algorithms.py`): each client keeps its own
momentum/EF21 estimator, compresses *its own* update, and the server aggregates
the *compressed* messages. **Distributed data-parallel** (here): one logical
model; `reduce_scatter(op=AVG)` gives one owning rank the true mean gradient;
that rank runs the ordinary **centralized** rule (momentum/LMO/sign/EF21 via
`self.state[p]`); `all_gather` broadcasts the updated parameter. The 1-bit
compression is thus a property of the update **rule**, reproduced exactly, not
of the wire transport. `EF21-MuonSign` keeps the exact server model `X` in state
and lets the live params be the sign-compressed broadcast model `W`;
`swap_in_exact()`/`swap_out_exact()` expose `X` for validation, and the scripts
report the loss at **both** models on the same validation tokens.

## Testing

Both tests run on CPU with only `torch` (plus `numpy` for the math test); no
GPUs.

```bash
# 1) update recurrence == numpy paper reference (../counterexamples/optimizers.py),
#    plus the per-layer LR multipliers
SIGNMUON_NO_COMPILE=1 python test_signmuon_optimizers.py

# 2) sharded step() == single-process centralized run.  Runs a PORTABLE simulation of
#    world sizes 1/2/4/8 first (no gloo, no multiprocessing -- works on Windows), then
#    the real-collectives gloo test if the platform supports it.
python test_distributed_sharding.py          # SIGNMUON_SKIP_GLOO=1 for the simulation only
```

Run **both** before renting the machine. Test (2) covers the two padding regimes
record #40's real parameter counts hit on 8 ranks: a group shorter than
`world_size`, and a group of 10 spanning two chunks the second of which is
partial. A rank owning a parameter in an early chunk but nothing in the last
needs a *fresh* scratch buffer for the padded `reduce_scatter`; reusing the
earlier one silently zeroes an already-averaged gradient, which on 8 GPUs would
freeze six of the ten `attn_gate` matrices for a whole run with no error
message. Neither test drives an input containing an exact zero, so both are
insensitive to the `sign_pm1` tie-breaking described under **Provenance**,
though the compressor tests exercise it indirectly: a residual confined to one
Q/K/V/O block leaves the other three at `mean|Δ| = 0`, where the random signs
must still be scaled away to nothing.

## Why record #40, and not the current NorMuon record

The current upstream record (#84, 1.320 min) fuses Muon into a `NorMuonAndAdam`
optimizer (NorMuon = Muon + an Adafactor-style per-neuron variance
normalization) with cautious weight decay, a batch-size/seq-len curriculum,
paired-head Muon, MTP, heavy Triton kernels, and ~15 records of hyperparameters
co-tuned to NorMuon. It offers no clean single orthogonalization step to host
the sign/EF21 rules, and swapping NorMuon out while keeping NorMuon-tuned LR/WD
would degrade results. **Record #40 is the principled cut**: the last record
*before* NorMuon, carrying every non-NorMuon architecture and kernel improvement
while keeping a separable Muon, tuned for plain Muon, whose LMO the variants
slot into unchanged. Porting onto a later record is a separate, larger effort
with real degradation risk (see the project notes).
