# Reproducing every number and figure in the paper

The command for each artifact of *SignMuon, MuonSign, and the Role of Error
Feedback* (`aaai_article/signmuon_body.tex`, `signmuon_appendix.tex`). Every
command runs **from this `code/` directory**, the Python package root.

```bash
cd code
pip install -r requirements.txt
python3 -m tests.test_code      # sanity check: CPU only, ~1 min, no downloads
```

Substitute `python` for `python3` on Windows, provided it has `torch` installed.
Add `--download` on the first run of anything that needs CIFAR-10 or MNIST.
Runtimes are for a single A100; everything writes to `results/`. The package
READMEs, linked per section, hold the rationale for each protocol. Artifacts are
named by **LaTeX label**, not by float number, which shifts whenever a float is
added; locate one with `grep -n 'label{tab:exp_3}' ../aaai_article/*.tex`.

| Paper artifact | Section |
| :--- | :--- |
| Theorems 1–3, `fig:divergence_plot` | [1. Counterexamples](#1-counterexamples-figdivergence_plot-theorems-13) |
| Theorem 4 (EF21-SignMuon), `fig:ef21_momentum` | [2. EF21-SignMuon divergence](#2-ef21-signmuon-divergence-theorem-4) |
| `tab:synthetic_tuned`, `tab:synthetic_alignment`, `tab:synthetic_dynamics`, `fig:synthetic_*` | [3. Synthetic convex problem](#3-synthetic-convex-problem) |
| `tab:cifar_main`, `tab:cifar_central`, `fig:cifar_results`, `fig:cifar_curves_appendix`, `fig:cifar_lr` | [4. Centralized CIFAR-10](#4-centralized-cifar-10-resnet-18) |
| `tab:fed_master`, `tab:exp_3`, `tab:commacct`, `fig:exp_3` | [5. Federated CIFAR-10](#5-federated-cifar-10-tabexp_3-figexp_3) |
| `tab:nanogpt`, `tab:nanogpt_diag`, `fig:nanogpt*` | [6. NanoGPT speedrun](#6-language-modelling-the-nanogpt-speedrun-tabnanogpt-fignanogpt) |
| all arms | [7. Multi-seed runs](#7-multi-seed-runs) |

Sections 3, 4 and 5 share one shape: **compute on the GPU box → download one
archive → unpack and plot anywhere.** The archives are
`results/synthetic_results.zip`, `results/article_export.tar.gz` and
`results/federated_export_results.zip`, each written by the command that
computes it. Section 6 needs none: its logs live under `results/nanogpt/`, the
one part of `results/` that git tracks. `results/` holds only what the paper
currently reports:

| Path | Arm |
| :--- | :--- |
| `article_export/`, `article_export.tar.gz`, `analysis/` | centralized (§4) |
| `federated/`, `federated_export_results.zip`, `federated_overnight/`, `federated_tuning_logs/` | federated (§5) |
| `synthetic/`, `synthetic_results.zip` | synthetic (§3) |
| `nanogpt/` | language modelling (§6), git-tracked |

Superseded runs live in `results_old/`, which nothing reads. Both trees are
gitignored, apart from `results/nanogpt/`.

> **In the anonymous supplement, `results/` holds the three archives, plus
> `nanogpt/`.** Unpack an archive where its section says to and every table and
> figure of that arm redraws without a GPU. The *unpacked* trees are not shipped,
> and neither are the raw per-run trees: an archive is the curated run set its
> exporter chose, while `results/federated/` also holds 66 runs from an earlier
> session under a superseded sign convention. `MANIFEST.md` lists the omissions.

## The two overnight drivers

Sections 4 and 5 are each one command:

```bash
python3 -m centralized.overnight --device cuda:0 --download
python3 -m federated.overnight   --device cuda:0 --budget-hours 24 --download
```

Both are **self-checking** (`tests/test_code.py` first, `--force` overrides),
**crash-isolated** (one subprocess per job), **resumable** (`--resume` retries
interrupted jobs), **budget-aware** (two real jobs are timed, a per-phase
schedule with finish times is printed, and the deadline is checked before each
job; `--budget-hours 0` disables it, the centralized default),
**priority-ordered** (diagnostics, `η₀`, headline table, ablation, finals
**seed-major**, so stopping early leaves a complete 1-seed table) and **readable
mid-run** (`REPORT.md` after every phase; Ctrl-C writes it). Each ends by
calling `centralized.export_article` or `federated.export_article`, which also
run by hand. Common flags: `--dry-run`, `--preflight-only`, `--phases lr final`,
`--lr-points 3`, `--methods signmuon ef21muonsign`, `--report-only` (rebuilds
`REPORT.md` from `state.json`, safe while a job trains). The federated driver is
deterministic by default and takes `--nondeterministic`; the centralized one is
not and takes `--deterministic`, reporting seed spread instead. Only the
federated driver *rebalances*, fitting the plan to `--budget-hours` rather than
cutting it halfway: seeds, then the weight-decay ablation, then the tuning
horizon give way in that order, the reverse when there is slack, and what is
dropped is recorded in `REPORT.md`.

---

## 1. Counterexamples (`fig:divergence_plot`, Theorems 1–3)

CPU, numpy only, **under a minute in total**.
[`counterexamples/README.md`](counterexamples/README.md) explains how the
instances are built and what each verdict column means.

```bash
python3 -m counterexamples.problems               # the theorem constants
python3 -m counterexamples.run_counterexamples    # fig:divergence_plot
python3 -m counterexamples.enumerate_minimality   # the Thm 2-3 minimality claims
python3 -m counterexamples.verify_ns_oracle --trajectories   # exact vs Newton-Schulz
```

`counterexamples.problems` prints the values quoted in Theorems 1–2 and the
Theorem 4 rate:

```
<G, sign(LMO(G))>   = -412.311   (Theorem 1: -42468/103)
<G, LMO(sign(G))>   =  -13.888   (Theorem 2: -13.89)
rate 49/480 = 0.102083 for every mu in {0, 0.25, 0.5, 0.9, 0.99}, both variants
```

Theorem 3's `<G, sign(LMO(sign(G)))> = -76` appears in `run_counterexamples`'s
Counterexample-2 table. That script and `plot_ef21_momentum` are the only two
that write into `aaai_article/`: each figure goes to `counterexamples/figures/`
as PNG and PDF, and to `../aaai_article/images/counterexamples/` as the PDF
LaTeX includes. `verify_ns_oracle` re-evaluates the instances under the
*implemented* Newton–Schulz oracle rather than the exact SVD, sweeping
`--steps`, `--sigmas` and `--Ms`; it has **no** `--mu` or `--nesterov`, since
momentum provably cannot change the trajectory (Proposition 1).

## 2. EF21-SignMuon divergence (Theorem 4)

```bash
python3 -m counterexamples.plot_ef21_momentum    # fig:ef21_momentum, ~20 s, CPU
```

Confirms the rate `49/480` per step for every `μ ∈ {0, 0.5, 0.9, 0.95, 0.99}`
and both momentum variants; `counterexamples.problems` checks
`μ ∈ {0, 0.25, 0.5, 0.9, 0.99}`. Both measure the rise over a whole number of
periods, since the trajectory is period-two and opposite-parity endpoints leak
half an oscillation into the slope; `run_counterexamples` reads its verdict off
the *period-two increment* for the same reason.

## 3. Synthetic convex problem

`F(X) = ½⟨X, AXB⟩` on `100 × 100` matrices, three independent draws of `(A, B)`,
`X₀ ~ N(0, 0.01)` entrywise, `L` and `σ` known in closed form.
[`synthetic/README.md`](synthetic/README.md) covers what each stage measures,
the problem construction, the grids and the batched runner.

```bash
python3 -m synthetic.run_gpu --force      # every stage, ~1.6 h on one GPU
python3 -m synthetic.run_gpu --archive    # rebuild SUMMARY.md + the .zip
python3 -m synthetic.plot_synthetic       # figures, from results/synthetic/
```

The first runs `tests/test_code.py` as a preflight, then all seven stages, then
writes the archive; the second only rebuilds the archive from disk. Variants:

```bash
python3 -m synthetic.run_gpu --list                 # what each stage measures
python3 -m synthetic.run_gpu --quick                # ~2 min smoke test, own tree
python3 -m synthetic.run_gpu --stages floor horizon # split the run
grep '†' results/synthetic/SUMMARY.md | grep '^|'   # the censored cells, if any
```

`run_gpu.py` takes only `--device`, `--out`, `--stages`, `--quick`, `--m`,
`--n`, `--runner`, `--force`, `--no-selftest`, `--archive` and `--list`. The
per-run knobs (`--methods`, `--init-seed`, `--problem-seeds`, `--lmo-dtype`, the
grids) belong to `synthetic/benchmark.py`, which each stage invokes with fixed
values. That `grep` lists table rows only, not the three legend lines explaining
the dagger, so `grep -c` is not the check: on the 2026-07-29 run it prints five
rows, all a *momentum* of `0.99` at the top of its grid, for SGD and SignSGD at
the largest condition numbers. **No learning rate is censored**, the condition
for reporting the tuned rates as measurements.

**`--force` is what makes the first command a rerun.** A stage counts as done
once one `<method>/<mode>.json` exists, so plain `run_gpu` on a box that has run
before skips all seven and exits in seconds looking like a success. `--force`
overwrites those files in place without clearing the directory, so files from a
run over a different method set survive; move `results/synthetic/` and its
`.zip` aside first if the previous run still backs paper numbers. `--quick`
writes to `results/synthetic_quick/` and `--m N` to `results/synthetic_NxN/`, so
no smoke test lands where reported numbers belong.
**`results/synthetic_results.zip` is the one file to bring back**: right-click
it under `code/results/` in VS Code and choose Download. `results/` is
gitignored, so nothing else leaves the box.

| Stage | Feeds |
| :--- | :--- |
| `stability` | `tab:synthetic_dynamics`. SGD is the control and must return `2/L`. **Do not skip it**: it is the only end-to-end check that the harness measures what it claims |
| `alignment` | `tab:synthetic_alignment` |
| `floor` | `tab:synthetic_dynamics`, `fig:synthetic_dynamics` (left) |
| `horizon` | `tab:synthetic_dynamics`, `fig:synthetic_dynamics` (centre) |
| `kappa` | `fig:synthetic_dynamics` (right) |
| `grid` | `tab:synthetic_tuned` |
| `final` | `fig:synthetic_main` |

| Path under `results/synthetic/` | What it is |
| :--- | :--- |
| `SUMMARY.md` | every table in one file, **the one to read** |
| `MANIFEST.json` | commit, GPU, CPU, RAM, OS, torch/CUDA/driver, argv and wall time per stage |
| `logs/<stage>.log` | full console output, including the preflight |
| `<method>/<mode>.json` | machine-readable results |

A `†` marks a tuned value at its grid edge, an upper bound rather than a
measurement; since each learning-rate window ends past its family's largest
measured stability edge, a surviving `†` puts the optimum at the edge of
stability, not outside the grid. `synthetic/README.md` explains the `p` and `q`
fits, in particular why `q ≈ 1/2` and `p ≈ 2` are the values to expect.
EF21-MuonSign is scored on `X`, the model its guarantee bounds, with the
gradient taken at the broadcast `W`.

### What is pinned, and what is not

One machine reproduces a stage exactly, and `grid` and `final` agree digit for
digit on their shared run: `benchmark.py`'s `--init-seed` fixes `X₀`,
`--problem-seeds` fixes `(A, B)`, and the tie-break RNG is forked and re-seeded
per run. `grid` re-runs its winner alone before reporting it, since in bfloat16
a matmul of a different batch width can round differently. Across machines it is
*not* bit-exact and cannot be: a different GPU or BLAS perturbs a gradient at
the last bit, `sign` is discontinuous, and an entry within rounding of zero
flips the step by `O(1)`, the instability this paper is about. Every reported
number survives, being a statistic over a trajectory rather than a single
iterate; `synthetic/README.md` gives the measured per-method agreement and the
`--lmo-dtype bfloat16` default that makes the effect visible.

## 4. Centralized CIFAR-10 (ResNet-18)

ResNet-18, 75 epochs, batch 128, momentum 0.9, cosine-annealed, three seeds.
[`centralized/README.md`](centralized/README.md) gives the protocol and its
justification: why `η₀` is selected at 75 epochs rather than at a proxy, which
`--lr-scaling` to use, and what the reported metrics mean.

### 4a. Three commands, start to finish

```bash
# 1. on the GPU box
python3 -m centralized.overnight --device cuda:0 --download

# 2. bring back the file it prints: results/article_export.tar.gz  (~355 KiB)
#    and unpack it to results/article_export/

# 3. redraw the figures  ->  results/analysis/
python3 -m centralized.plot_analysis
```

The driver calls `centralized.export_article` on finishing, and by hand it
rebuilds the archive at any time. Only that file needs bringing home:
`centralized/main.py` saves a 42.7 MB `model.pt` for every run, tuning runs
included, so a 125-job centralized night is several GB and stays on the box.
Step 3 writes `fig:cifar_results` (`cifar_main`), `fig:cifar_curves_appendix`
(`curves_75ep`), `fig:cifar_train_loss` (`train_loss_75ep`) and `fig:cifar_lr`
(`lr_sensitivity`) as PNG and PDF, and prints the two sweep spreads the
`fig:cifar_lr` caption quotes: the absolute window `[0.01, 0.05]`, and the range
within a factor of five of each method's own optimum. Copying into
`aaai_article/` is manual. At the 12.9 s/epoch measured on an RTX A4500 the
schedule reads:

```
  phase     jobs  epochs   hours  cumulative   done by
  gain         4      20     0.3         0.3   Sun 02:20
  aux         30      15     1.9         2.2   Sun 04:12
  lr          50      75    15.0        17.2   Sun 19:12
  final       30      75     9.0        26.2   Mon 04:12
  wd           3      75     0.9        27.1   Mon 05:04
```

The archive contains:

| File | What it is |
| :--- | :--- |
| `table_cifar.csv` | **`tab:cifar_main` / `tab:cifar_central`**, aggregated over seeds exactly as the paper defines them; quote from here rather than retyping |
| `runs.csv` | one row per run: config plus every derived summary metric |
| `curves.csv` | tidy per-epoch series; the figures are a function of this and `runs.csv` alone |
| `gain.csv`, `gain_fits.csv` | the `--log-gain` series and its fitted exponent |
| `environment.json`, `hardware.tex` | **exact GPU, driver, CUDA, Python, PyTorch, commit**, §4d |
| `configs.json` | the full config of every run, nothing dropped |
| `overnight/` | `REPORT.md` and `state.json` from the driver |
| `MANIFEST.md` | what each file is, and how many seeds each configuration carries |

> **The AAAI-submitted figures.** The paper reports the 2026-07-30 re-run; the figures in the frozen
> AAAI submission came from an earlier run under the 15-epoch selection protocol. To reproduce them,
> check out the commit that carried `centralized/table2_full.csv`.

### 4b. Selection happens at the reporting horizon

The `lr` phase runs at `--final-epochs`, not a short proxy. Re-running the top
three rates at 75 epochs once **reversed** the 15-epoch ranking for both methods
probed, SignMuon's optimum moving up three lattice steps and Muon's down two:
each run anneals cosinally to zero over *its own* horizon, so a proxy measures a
different schedule, not a noisier one. The cost is 15 GPU-h instead of 3.7.
Selection is `val_acc` on the 45k/5k split; the argument is in
[`centralized/README.md`](centralized/README.md).

### 4c. The stages by hand

```bash
# The horizon matters: --stage lr must run at the horizon the table reports,
# while aux and alpha compare two arms at one shared horizon and can be short.
TUNE="--dataset cifar10 --model resnet18 --head-adamw always \
      --device cuda:0 --data ./data"

# Test whether the auxiliary rate is method-independent  (~2 GPU-h)
python3 -m centralized.tune --stage aux $TUNE --epochs 15 --lr-scaling unit-gain

# eta_0 per method, 5 lattice points each, at the reporting horizon (~15 GPU-h)
python3 -m centralized.tune --stage lr $TUNE --epochs 75 --lr-scaling unit-gain \
                            --lr-aux 0.001

# Finals: 3 seeds, full 50k, fixed 75-epoch budget
for m in signmuon muonusign muonsign ef21signmuon ef21muonusign ef21muonsign \
         muon signsgd sgd adam; do
  for s in 0 1 2; do
    python3 -m centralized.main --dataset cifar10 --model resnet18 --epochs 75 \
      --optimizer $m --lr-scaling unit-gain --head-adamw always \
      --lr <tuned> --lr-aux 0.001 --seed $s --device cuda:0 --data ./data
  done
done
python3 -m centralized.export_article
```

Two diagnostics the paper's appendix quotes but no table depends on:

```bash
python3 -m common.lr_scaling --measure     # single step: incoherent, favours unit-gain
python3 -m centralized.main ... --log-gain --constant-lr    # the driver's `gain` phase
```

`--constant-lr` is required there: under a decaying schedule the accumulation
saturates and the fit reports the schedule rather than the alignment. The
per-layer exponent sweep (`--stage alpha`) feeds no paper number, since the
`gain` phase measures the exponent directly and at a single width it is largely
absorbed into `η₀` (the sweep came out flat to within 0.3%). Use the **same seed
set for every method**, so the comparison is *paired*; claim a gap only when it
exceeds the paired std, otherwise write "indistinguishable". All ten methods
appear in `tab:cifar_central`, including `ef21signmuon`, the method Theorem 4
can make diverge, present as a predicted-failure baseline and in fact topping
the table.

### 4d. Hardware and software, for the reproducibility appendix

Every run stamps its machine into its own `metrics.json`
(`common.utils.save_run` → `common.hardware.describe`): GPU model and memory,
driver, CUDA, cuDNN, CPU, RAM, OS, Python and PyTorch versions, the git commit,
and whether the tree was dirty. No hostname, username or absolute path is
recorded, so the stamp is safe under double-blind review.

```bash
python3 -m common.hardware                     # this machine, one prose sentence
python3 -m common.hardware --latex             # this machine, as a LaTeX row
python3 -m common.hardware --scan results      # every machine used, by experiment
```

Fill the checklist from these, not from whichever machine compiles the LaTeX:
the experiments ran on different GPUs, so one "computing infrastructure"
sentence would be wrong for most of the table.

## 5. Federated CIFAR-10 (`tab:exp_3`, `fig:exp_3`)

CNN2, **11 clients**, 3 local accumulation steps, 2000 rounds, batch 64 per step
(= a gradient at batch 192), momentum 0.9, weight decay 0, homogeneous split,
**five seeds**: one federation scale, all eleven methods. The protocol matches
§4: a held-out validation split, per-layer rates derived from the shape, an
equal-budget lattice with a boundary check, multi-seed finals.
[`federated/README.md`](federated/README.md) covers what is specific to
federation, including the eleven methods and their two uncompressed controls,
the randomization of exact zeros, `tab:commacct` and which methods compress the
downlink, the realized per-layer gain, frozen BatchNorm, and why `N = 11`.
**Read its compression section before quoting any ratio.**

### 5a. Three commands, start to finish

```bash
# 1. on the GPU box
python3 -m federated.overnight --device cuda:0 --budget-hours 24 --download

# 2. download the file it prints: results/federated_export_results.zip  (~285 KiB)

# 3. anywhere
python3 -m federated.plot_article --bundle results/federated_export_results.zip
```

The driver calls `federated.export_article` on finishing;
`python3 -m federated.export_article` rebuilds the archive at any time from
disk, without retraining. `--bundle` unpacks the `.zip` itself and writes
`fig:exp_3` as `fig_federated_main.pdf` (and PNG) into `<bundle>/figures/`; copy
it into `aaai_article/images/federated_images/` deliberately. The run tree stays
on the GPU box: 2.9 MB of `model.pt` per job, ~370 MB for a night.

`SUMMARY.md` inside the bundle is the file to read: `tab:exp_3`, the
communication accounting and the per-run diagnostics in one place. Beside it are
`table_federated.csv` (the table, aggregated over seeds as the paper defines the
columns; quote from here rather than retyping), `communication.csv`, `runs.csv`,
`curves.csv`, `configs.json`, `environment.json`, `hardware.tex`,
`MANIFEST.json`, and `runs/`, which holds each run's `metrics.json` in the
original tree shape, so `--bundle` is `--root` on the unpacked copy. Phases:
`lr` (η₀ per method on the lattice) → `verify` (the top rates re-run at
`--final-rounds` to test horizon stability, skipped when the tuning horizon
already equals it) → `final` (full 50k, seed-major) → `wd` (the decay ablation
at `5e-4`), plus the opt-in `rules` (the per-layer rule ablation, §5e).

**`--tune-rounds` defaults to `--final-rounds`**: rates are selected at the horizon
the table reports, as in §4b. That is not a refinement — the 2026-07-30 run's own
`verify` phase found the 400-round proxy picking the wrong rate for *both* methods
it checked (SignMuon 0.05 → 0.1, EF21-MuonSign 0.05 → 0.02), which is the same
cosine-schedule artefact that retired proxy tuning centrally. At the 0.127 s/round
measured on an A100 the schedule reads:

```
  phase     jobs  rounds   hours  cumulative   done by
  lr          60    2000     4.9         4.9   Fri 03:36
  verify       -       -       -           -   vacuous: tuning is at the reporting horizon
  final       60    2000     4.9         9.8   Fri 08:30
  rules       48    2000     3.9        13.7   Fri 12:25
  wd           3    2000     0.2        14.0   Fri 12:40
```

Sixty jobs per phase, not fifty-five: Adam is additionally tuned under the sign
rule (`--skip-baseline-variants` turns that off), so twelve tracks cover eleven
methods. Give it `--budget-hours 0` (no deadline) or two nights with `--resume`.
Under a tighter budget the driver sheds the ablations, then seeds, then the tuning
horizon, **naming each** in `REPORT.md`; a shortened horizon reintroduces the proxy,
so it switches the `verify` check back on and says so. A 3-seed table is not what
`tab:exp_3` reports. Federated-only variants:
`--partition noniid-labeldir --beta 0.5`, `--final-seeds 0 1 2`, `--wd-ablation 0`.

### 5b. The stages by hand

```bash
FED="--dataset cifar10 --model cnn2 --n_parties 11 --n_steps 3 --batch_size 64 \
     --lr-scaling unit-gain --device cuda:0 --data ./data_federated"

# The grid each method will search, and where its anchors come from. Run this first.
python3 -m federated.tune --stage anchors $FED
python3 -m federated.tune --stage votes            # the majority-vote alignment table

# Test whether the auxiliary rate is method-independent  (~30 configs, 400 rounds)
python3 -m federated.tune --stage aux $FED

# eta_0 per method, equal budget, selected on val_acc only, at 2000 rounds
python3 -m federated.tune --stage lr $FED --lr-points 5
```

Do not pass `--rounds` to `--stage lr` to make it cheaper. The horizon defaults per
stage — 2000 for `lr`, because that is where the table is reported, and 400 for
`aux`, whose comparison is between two arms at a *shared* horizon, where the bias
cancels. Shortening `lr` reintroduces the proxy that picked the wrong rate for both
methods it was checked on.

`--stage anchors` is the check that matters: it prints `λ` per layer per family
and the transported grid anchors, which every tuned rate downstream inherits.
Then the finals, five seeds, on the full 50k:

```bash
FED11="--model cnn2 --dataset cifar10 --rounds 2000 --n_parties 11 --n_steps 3 \
       --batch_size 64 --momentum 0.9 --lr-scaling unit-gain --split full \
       --data ./data_federated --device cuda:0 --eval_freq 100 --lr-aux 0.001"

for m in signmuon muonusign muonsign ef21signmuon ef21muonusign ef21muonsign \
         muon muonserver signsgd sgd adam; do
  for s in 0 1 2 3 4; do
    python3 -m federated.main $FED11 --algorithm $m --lr <tuned> --seed $s
  done
done
python3 -m aggregate --root results/federated --metric test_acc --csv fed_table.csv
```

Runs land in
`results/federated/fed_cifar10_<algorithm>_homo_cnn2_r2000_c11_s3_unit-gain/seed<s>/`.
Use the same seed set for every method, and claim a gap only when it exceeds the
paired std. Then bundle and plot as in §5a, or read a live tree directly:

```bash
python3 -m federated.export_article                                  # -> the .zip
python3 -m federated.plot_article --root results/federated --n-parties 11
python3 -m federated.plot_federated --metrics gain_spread mv_tie_frac
```

`federated.plot_federated` is the exploratory plotter, one file per metric over
any recorded series, and is not what the paper prints. Both plotters take
`--root` (a live tree) or `--bundle` (an archive, unpacked or still zipped).

> **`muonserver` is the row that makes the comparison honest.** There are two full-precision Muons,
> one per template: `muon` orthogonalizes on the worker and the server averages the polar factors,
> while `muonserver` averages first and orthogonalizes once, which is what MuonUSign, EF21-MuonUSign
> and EF21-MuonSign *become* under an identity compressor. Averaging near-orthogonal matrices
> shortens the step by up to 27% at N=11, so comparing the server-LMO family against `muon` alone
> confounds the 1-bit uplink with the averaging. It appears in `fig:exp_3` and not in `tab:exp_3`.

### 5c. Reproducing the *previously published* federated table instead

The rates published before this protocol (`icomp_article/`, `paper/main_ru.tex`)
were tuned under the `legacy` per-layer rule, one global rate for the sign
family and Muon's aspect factor for the LMO family, at `N = 10` with weight
decay `5e-4`. `--lr-scaling legacy` reproduces that convention exactly: the
shape factor is applied outside the oracle now, and under `legacy` the two
conventions coincide bit for bit
(`test_federated_legacy_rule_is_the_old_convention`).

```bash
LEG="--model cnn2 --dataset cifar10 --rounds 2000 --n_parties 10 --n_steps 3 \
     `# 10, not 11: this block reproduces the published setting verbatim` \
     --batch_size 64 --momentum 0.9 --lr-scaling legacy --weight_decay 5e-4 \
     --split full --data ./data_federated --device cuda:0 --eval_freq 100 \
     --seed 0 --lr-aux 0.001"

python3 -m federated.main $LEG --algorithm signmuon      --lr 0.001
python3 -m federated.main $LEG --algorithm muon          --lr 0.05
python3 -m federated.main $LEG --algorithm signsgd       --lr 0.001
python3 -m federated.main $LEG --algorithm sgd           --lr 0.03
python3 -m federated.main $LEG --algorithm adam          --lr 0.001
python3 -m federated.main $LEG --algorithm ef21muonusign --lr 0.02
python3 -m federated.main $LEG --algorithm ef21muonsign  --lr 0.035
```

These still will not match the printed numbers: the pre-refactor code was not
uniform across methods (`../REVIEW_NOTES.md` §4, the cosine schedule and the
routing of biases and BatchNorm differed per method). They reproduce the
*corrected* comparison under the old parameterization.

### 5d. What changed, and why

| | before | now |
| :--- | :--- | :--- |
| **Selection data** | test accuracy, read from the training log by `federated/grid.py` | `--split tune`: 5k held out of the 50k *before* the client partition, and a tuning run never scores a test image |
| **Per-layer LR** | one global rate (`legacy`) | `--lr-scaling unit-gain`, as centrally |
| **Grid** | ad-hoc `arange` per method | 1–2–5 lattice, equal budget, boundary extension |
| **Weight decay** | `5e-4`, and the auxiliary AdamW was decayed too | `0` primary (matching §4), auxiliary group never decayed, `5e-4` as an ablation |
| **Clients** | 10 (even: the vote ties) | 11 |
| **Seeds** | 1 | 5 |
| **Sign alphabet** | ternary (`sign(0) = 0`) | randomized `±1`, a strict one bit |
| **Data pipeline** | 10 `DataLoader`s, `num_workers=0` | dataset resident on the GPU, augmentation as tensor ops |

Two facts to keep stated in the paper: `lr_aux = 0.001` is used throughout and
is not tuned per method (`--stage aux` checks that it may be held fixed), and
BatchNorm running statistics are never updated from data in the federated
setting.

### 5e. The per-layer rule ablation (`rules`)

Whether the sign family's *ordering* depends on the per-layer multiplier. It is
opt-in, because it answers a question about the rule rather than about the methods:

```bash
python3 -m federated.overnight --device cuda:0 --budget-hours 0 \
    --phases lr final rules wd
```

Each (method, rule) pair is re-tuned **from scratch** under that rule, with its own
grid search and its own boundary extension, then run at the final horizon on three
seeds. Both halves matter: a rule shifts the optimal `η₀` by roughly the multiplier
it prescribes, so comparing rules at one shared rate would measure the shift rather
than the rule; and an alternative rule whose optimum is censored by its grid would
lose to a properly tuned `unit-gain` automatically, which would make the ablation
confirm the rule it exists to test.

`--lr-scaling` itself is not repeated — the `lr` and `final` phases already are it,
so the ablation compares against the same runs `tab:exp_3` quotes. Defaults:
`--rule-methods signmuon muonsign signsgd` (the LMO family cannot move, since unit
gain and µP prescribe it the identical factor), `--rule-alternatives none mup`,
`--rule-seeds 0 1 2`. The bundle gains `rule_ablation.csv` and a SUMMARY section
that states the ordering under each rule and whether it changed.

## 6. Language modelling: the nanoGPT speedrun (`tab:nanogpt`, `fig:nanogpt*`)

The one experiment that needs rented hardware, **8×H100 for ~25 minutes of
training per method, eight methods**, and the one whose *analysis* needs none:
the released logs are in the repository, so every number and figure rebuilds on
a laptop without a GPU, without torch, and without downloading FineWeb. Read
[`nanogpt/README.md`](nanogpt/README.md) before changing a setting.

### 6a. Rebuild every nanoGPT number and figure, no GPU (~5 seconds)

```bash
cd nanogpt
python parse_logs.py     # ../results/nanogpt/logs -> runs.csv, steps.csv, diagnostics.csv
python make_tables.py    # -> SUMMARY.md, MANIFEST.json, tab_nanogpt.tex, tab_nanogpt_diag.tex
python plot_article.py   # -> ../results/nanogpt/figures/fig_nanogpt_{main,appendix,diag}.pdf
```

`parse_logs.py` and `make_tables.py` are standard library only;
`plot_article.py` needs matplotlib. Then:

```bash
cat ../results/nanogpt/SUMMARY.md               # both tables, the control check, provenance
diff <(sed -n '/tabular/,/tabular/p' ../../aaai_article/signmuon_body.tex) \
     ../results/nanogpt/tab_nanogpt.tex         # the paper's table vs the derived one
```

`SUMMARY.md` carries `tab:nanogpt`, `tab:nanogpt_diag`, the record-#40 control
check (whether the `Muon` run lands on the published curve), the wall-clock
spread the appendix quotes, and a provenance table naming the build of
`signmuon_optimizers.py` behind each log. The eight runs are **not** all of one
vintage, and they predate the `SIGNMUON_SEED` knob; `nanogpt/README.md`
§"Provenance" says why that is sound.

### 6b. The runs themselves (8×H100)

```bash
cd nanogpt
bash setup_env.sh && source .venv/bin/activate   # venv + a real Flash-Attention-3 fetch
python data/cached_fineweb10B.py 9               # ~900M train tokens + the val chunk
NANOGPT_ITERS=200 bash run_all.sh                # smoke pass first: do this once
bash run_all.sh                                  # the eight hero runs (~25 min each)
```

`run_all.sh` preflights the environment once rather than eight times, runs
`Muon` first (it *is* record #40's optimizer, so it must land on that record's
published curve, and `reference_record40.csv` is the pass/fail line), and writes
logs into `../results/nanogpt/logs/`. Then rerun §6a. Two substitutions cover
the common hardware gaps, both swapping FA3 → FlexAttention and FP8 → bf16,
outside the optimizer:

```bash
NPROC=8 SCRIPT=train_gpt_a100.py bash run_all.sh   # eight GPUs, no Flash-Attention-3
NPROC=1 SCRIPT=train_gpt_a100.py bash run_all.sh   # a single A100 or H100
```

Before renting anything, run the two CPU tests (they need torch, nothing else):

```bash
SIGNMUON_NO_COMPILE=1 python test_signmuon_optimizers.py   # update rules == the paper's
python test_distributed_sharding.py                        # sharded step() == centralized
```

### 6c. What is fixed a priori, and what is not

Nothing here is tuned by us. Every hyperparameter outside the matrix optimizer
is record #40's own: the five LMO-family methods run at its `η₀ = 0.06`, the
three sign-family methods at `0.03`, the spectral discount derived in
`app:lrscale` and defended three independent ways in `train_gpt.py`'s
`OPTIMIZER_CONFIG` comment. Every contrast in `tab:nanogpt` is therefore
matched-hyperparameter. The one number with real uncertainty is that `0.03`;
`SIGN_PROBE_LR=0.01 bash run_all.sh` adds one downside probe per sign method.
Unlike §4 and §5 this experiment is **single-run**: the table quotes record
#40's own five-seed spread (`± 0.0009`) as the noise scale instead, so differences below ~0.003 are
not results.

## 7. Multi-seed runs

Any command above becomes a multi-seed sweep by varying `--seed`; the seed is
part of the output path, so nothing is overwritten:

```bash
for s in 0 1 2 3 4; do
  python3 -m federated.main $FED11 --algorithm ef21muonsign --lr 0.05 --seed $s
done

python3 -m aggregate --metric test_acc --csv summary.csv --curves curves.json
```

`aggregate.py` groups runs by configuration-minus-seed (and minus device and
data-path fields) and reports mean ± sample std; `curves.json` holds the
pointwise mean/std curves for error-band plots. It scans all of `results/` by
default (`--root results/federated` for one family), and is the **federated**
path's aggregator; the centralized one goes through
`centralized.export_article`, whose `table_cifar.csv` is the paper's table
itself. Both report mean ± sample std over seeds: three for `tab:cifar_main` and
`tab:cifar_central`, five for `tab:exp_3`. A single seed measures no dispersion,
so the tools print a blank rather than `± 0.00`; a blank is not agreement. Claim
a gap only when it exceeds the seed spread.

---

## Notes

* **Plotting is scripts, not notebooks.** The four notebooks under `notebooks/`
  read the pre-reorganization `saves*` paths and were removed on 2026-07-28.
  Their replacements read `results/`:

  | figure | script |
  | :--- | :--- |
  | `fig:divergence_plot` | `counterexamples.run_counterexamples` |
  | `fig:ef21_momentum` | `counterexamples.plot_ef21_momentum` |
  | `fig:synthetic_*` | `synthetic.plot_synthetic` |
  | `fig:cifar_*` | `centralized.plot_analysis` (§4a) |
  | `fig:exp_3` | `federated.plot_article` |
  | `fig:nanogpt*` | `nanogpt/plot_article.py` |

* **One style, in [`common/plotting.py`](common/README.md#plottingpy).** Every
  script above calls `use_paper_style()` and takes its colours from `color_of`,
  so a method keeps its colour across figures and every figure matches
  `aaai2027.sty`: Times text, STIX math, and TrueType outlines rather than the
  Type 3 fonts matplotlib emits by default and AAAI forbids. Figures are
  authored at their printed width (`TEXT_WIDTH`, `COLUMN_WIDTH`) and included at
  `width=\textwidth` or `\columnwidth`, so a 9 pt label is 9 pt on the page.
  Style new figures the same way.
* **Disk.** `SIGNMUON_RESULTS` relocates the whole `results/` tree and the
  scripts follow it, each resolving its paths through
  `common/paths.py:results_root()`; `--data` is separate and can also be pointed
  off the system disk. One `model.pt` is 2.9 MB for CNN2 and 42.7 MB for
  ResNet-18, so the ~129-job federated night is ~370 MB and the 125-job
  centralized night several GB. See [README.md](README.md).
