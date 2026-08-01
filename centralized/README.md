# `centralized/` — single-node CIFAR-10 / MNIST

ResNet-18 on CIFAR-10, 75 epochs, cosine-annealed. Produces `tab:cifar_main`,
`tab:cifar_central` and `fig:cifar_*`.

| File | What it does |
| :--- | :--- |
| [`main.py`](main.py) | One run, one config, one `metrics.json` |
| [`train.py`](train.py) | Training loop and optimizer construction |
| [`data.py`](data.py) | Loaders, including the fixed 45k/5k tuning split |
| [`tune.py`](tune.py) | Equal-budget, validation-only learning-rate search |
| [`overnight.py`](overnight.py) | **Step 1**: the whole protocol, resumable |
| [`export_article.py`](export_article.py) | **Step 2**: pack the results into one archive |
| [`plot_analysis.py`](plot_analysis.py) | **Step 3**: redraw every figure from that archive |

## Run it

```bash
cd code

# 1. on the GPU box: ~1.5 days, resumable, bundles itself at the end
python3 -m centralized.overnight --device cuda:0 --download

# 2. bring back results/article_export.tar.gz (~350 KiB) and unpack it
#    to results/article_export/

# 3. redraw the figures  ->  results/analysis/
python3 -m centralized.plot_analysis
```

`export_article` runs automatically at the end of step 1;
`python3 -m centralized.export_article` rebuilds the archive from disk at any
time. The run tree never has to move: it holds a `model.pt` per job, several GB
for a night, and the archive carries every number the paper quotes. Set
`SIGNMUON_RESULTS` to write that tree to another volume; every script here reads
the same variable.

Current results: the 2026-07-30 run, 126 runs, three seeds per reported
configuration (one for the weight-decay ablation), `η₀` selected at the reporting
horizon.

## Phases

Watch the first ~6 minutes. The driver runs the CPU test suite, prints the
per-layer learning-rate table, records GPU / driver / CUDA / Python / PyTorch,
times two real epochs, then prints a schedule with a finish time per phase.
`results/overnight/REPORT.md` is rewritten after every phase, so it can be read
mid-run; Ctrl-C stops cleanly and writes it.

Ordered so that stopping early costs the least:

| Phase | What it establishes |
| :--- | :--- |
| `gain` | Does the accumulated update grow like `√t` (incoherent → `unit-gain`) or like `t` (aligned → `mup`)? Run at a **constant** rate: under annealing the accumulation saturates and the slope measures the schedule |
| `aux` | Is the optimal `lr_aux` shared between two methods far apart in `η₀`? On the 2026-07-30 run it **disagreed** (SignMuon `0.001`, Muon `0.002`, a difference of 0.16 points), so the fixed `1e-3` is reported as a convention, not as verified |
| `lr` | `η₀` per method, 1–2–5 lattice, grid widened when an optimum lands on an endpoint — **at the reporting horizon**, see below |
| `final` | Full-50k runs at the tuned rates, **seed-major**: every method at seed 0 before any reaches seed 1, so a cut night leaves a complete table rather than fragments |
| `wd` | The best `--wd-ablation-top` methods (default 3) repeated at seed 0 with weight decay on, to test whether the *ordering* moves |

## Why `η₀` is selected at 75 epochs, not at a proxy

The driver used to rank at 15 epochs and re-check the top few at 75. On the
2026-07-27 run the ranking **reversed** for both methods probed:

| method | best @ 15 ep | best @ 75 ep |
| :--- | ---: | ---: |
| SignMuon | `0.02` (93.68) | `0.2` (94.54) |
| Muon | `0.05` (93.46) | `0.01` (94.60) |

Not seed noise: both runs anneal to zero over *their own* horizon, so the
15-epoch run spends nearly its whole budget at a decayed rate and measures a
different schedule. The bias is not one-directional either: SignMuon's optimum
moved up three lattice steps and Muon's down two, so no correction factor
patches it.

The `lr` phase therefore runs at `--final-epochs`: costlier than a proxy, but the
rate in the table is the rate that won at the horizon the table reports.
Selection remains `val_acc` on the 45k/5k split.

`lr_aux` is exempt. The `aux` phase compares the *argmax over `lr_aux`* between
two methods at the **same** horizon, so a shared horizon bias cancels; it runs at
`--aux-epochs 15`.

## Protocol

Four properties enforced by the code rather than by discipline:

1. **No test-set tuning.** Selection uses `--split tune`, a fixed 45k/5k
   partition with `--val-seed` independent of `--seed`, and reads `val_acc`
   averaged over the last `--last-k` epochs rather than one evaluation.
2. **Equal budget.** Every method starts from the same five-point grid, anchored
   *multiplicatively* (`tune.anchor_for`); the two families have different
   natural scales, so a shared absolute grid would not be equal. Property (3)
   then extends individual methods.
3. **An optimum on a grid endpoint is a failure.** The grid is widened along the
   same lattice and the method re-run, up to four times. On the shipped run this
   took Muon and EF21-MuonUSign to seven points and SignSGD to nine.
4. **Per-layer rates are derived, not tuned.** `--lr-scaling` sets
   `η_layer = η₀·λ(family, shape)`; only the shape-free `η₀` is searched.

Property (4) is falsifiable: once the shape dependence lives in `λ`, the tuned
`η₀` should agree *within* each family. `tune.py` reports that agreement.

## Which `--lr-scaling`?

| rule | sign family | LMO family | note |
| :--- | :--- | :--- | :--- |
| `legacy` | `η₀` | `η₀·√max(1,m/n)` | one global rate for the sign family |
| `none` | `η₀` | `η₀` | what the concurrent Sign-Muon implementation runs |
| **`unit-gain`** | `η₀/√fan_in` | `η₀·√max(1,m/n)` | **derived; the default** |
| `mup` | `η₀/fan_in` | `η₀·√max(1,m/n)` | assumes accumulated steps align |
| `mishra-analysis` | `η₀/√(mn)` | `η₀/√min(m,n)` | the normalization used in their *proof* |

> **ResNet-18 is a weak instrument for the exponent.** Thirteen of its twenty conv
> weight tensors have `fan_in/fan_out = 9` exactly and hold 84.5% of all
> parameters; one shape alone, `(512, 4608)` appearing three times, is 63% of the
> model. So `α` is identified only through the transition and 1×1-downsample
> layers. CNN2 in the federated setting has a 7.8× multiplier spread and no
> dominant shape, so it is the better instrument; the `gain` phase measures the
> exponent directly either way.

## Reported metrics

| metric | role |
| :--- | :--- |
| test accuracy, mean of the last `--last-k` epochs | **primary** |
| train accuracy | underfitting diagnostic |
| epochs to `--target-acc` | separates *speed* from final quality |
| median epoch time | cost of the method |

All four are printed in a `--- summary ---` block at the end of every run, and
`table_cifar.csv` in the export bundle carries them aggregated over seeds exactly
as the paper defines them, so no table row is retyped by hand.

**Test loss rises after epoch ≈40 while test accuracy keeps improving.** That is
the standard overconfidence regime once train accuracy saturates: the loss on
misclassified points grows while the decision boundary still improves. Not a bug,
and not a reason to early-stop on test.

See [`../REPRODUCE.md`](../REPRODUCE.md) §4 for the exact commands.
