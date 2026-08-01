# `common/` — the shared library

Everything more than one experiment needs. Nothing here trains anything; all five
experiment packages import from it, which is what stops them describing different
algorithms under the same name.

| File | What it holds |
| :--- | :--- |
| [`optimizers.py`](optimizers.py) | The eight methods as `torch.optim.Optimizer` subclasses, plus `muon_lmo` and the Newton–Schulz iteration |
| [`lr_scaling.py`](lr_scaling.py) | Per-layer learning-rate rules, derived rather than tuned |
| [`models.py`](models.py) | `CNN2`, `ResNet9`, `ResNet18` |
| [`utils.py`](utils.py) | Seeding, run directories, the metrics schema, parameter routing, the cosine schedule |
| [`plotting.py`](plotting.py) | The one figure style: palette, page geometry, rcParams |
| [`hardware.py`](hardware.py) | GPU, driver, CUDA, Python, torch and commit, stamped into every run |
| [`paths.py`](paths.py) | Where `results/` is, and how absolute paths are kept out of a bundle |

## `optimizers.py`

One base class with a `_direction` hook; each method is the ~5 lines that differ.
`M` is the effective momentum direction and `polar(Y) = U Vᵀ` is the Muon LMO.

| Class | `d_t` (the step is `X ← X − η·λ·d_t`) | Family |
| :--- | :--- | :--- |
| `Muon` | `polar(M)` | `lmo` |
| `SignSGD` | `sign(M)` | `sign` |
| `SignMuon` | `sign(polar(M))` | `sign` |
| `MuonUSign` | `polar(sign(M))` | `lmo` |
| `MuonSign` | `sign(polar(sign(M)))` | `sign` |
| `EF21SignMuon` | EF21 estimator of `polar(M)` | `lmo` |
| `EF21MuonUSign` | `polar(g_est)`, `g_est → M` | `lmo` |
| `EF21MuonSign` | as above, plus downlink EF21-P on `X` | `lmo` |

Three things to know before reading a step:

* **Momentum is the EMA form** `M = μM + (1−μ)G` of the algorithm boxes. It is
  trajectory-identical to the main text's heavy-ball form: the buffers differ by
  the constant `(1−μ)`, and `sign`, `polar` and the EF21 recursion are all
  positively homogeneous, so the factor cancels. Pinned by
  `test_gradient_scale_invariance`.
* **Weight decay is decoupled** (`p *= 1 − lr·wd`), not as a preference. Every
  `_direction` is positively homogeneous of degree *zero*, so folding `wd·p` into
  the gradient cannot shorten the step at all; it only rotates it, by an amount
  set by the drifting ratio `wd‖p‖/‖g‖`. `decoupled_weight_decay=False`
  reproduces the coupled convention for the ablation.
* **`EF21MuonSign` keeps two models.** `p.data` is the broadcast model `W`, so
  `p.grad` is the gradient at `W`; the server model `X`, the iterate the theory
  bounds, lives in optimizer state. `using_exact()` evaluates on it,
  `restore_exact()` installs it.

`zeropower_via_newtonschulz5` approximates the polar factor's *direction* well but
its *norm* only to a few percent, and oscillates rather than converging;
`test_newton_schulz_gets_the_direction_but_not_the_scale` measures that band
instead of asserting a tolerance someone picked. Which is why magnitude is handled
separately, by `lr_scaling`.

## `lr_scaling.py`

The two families produce step matrices whose Frobenius norms scale differently with
shape (`√min(m,n)` against `√(mn)`), so one global rate cannot be right for both,
nor across layers. The rule is *derived* from one criterion: an update's RMS gain
`γ(A) = ‖A‖_F/√fan_out` should be a fixed fraction of the initialization's.

```
λ = √fan_out / ‖s‖_F     ⟹     lmo: η₀·√max(1, m/n)      sign: η₀/√fan_in
```

The strongest evidence for the criterion is that it *re-derives* the
`√max(1, m/n)` factor already shipped in reference Muon, which is also why Muon's
learning rate transfers across widths. The sign family has never been given its
counterpart.

```bash
python3 -m common.lr_scaling             # the rules, and what each assumes
python3 -m common.lr_scaling --compare   # per-layer multiplier profiles
python3 -m common.lr_scaling --measure   # is a single sign step incoherent or aligned?
```

`--measure` settles the one modelling question the derivation leaves open. `mup`
applies the criterion to the *accumulated* update assuming successive sign steps
align with the activations; `unit-gain` assumes they do not. The measurement gives
`‖sign(polar(M))‖_op ≈ 0.93(√m + √n)`, flat to 9% across every ResNet-18 shape:
spectrally incoherent, not rank-one aligned. The complementary check is
`centralized.main --log-gain --constant-lr`, which records the accumulated update's
growth; `centralized.export_article` fits its exponent. Both flags are needed —
under a decaying schedule the accumulation saturates and the fit reports the
schedule.

## `plotting.py`

Every plotting script starts with `use_paper_style()`. Two properties follow:

* **A figure matches `aaai2027.sty`.** Times text and STIX math (the template
  loads `newtxtext`), and `pdf.fonttype = 42`; matplotlib's default emits **Type
  3** fonts, which AAAI forbids.
* **A method keeps one colour across a figure family.** `color_of` holds the
  per-method map used by the counterexample, synthetic and nanoGPT figures. The
  centralized panels are the deliberate exception: each holds three methods and
  reads as its own small comparison, so it colours by panel slot and draws Muon in
  reference gray throughout.

Sizes are printed points, so author at `TEXT_WIDTH` (7.0 in, a `figure*`) or
`COLUMN_WIDTH` (3.3125 in, a `figure`), include at `width=\textwidth` or
`\columnwidth`, and skip the tight bounding box. Trimming margins makes LaTeX
scale the figure back up, taking every point size with it. A script that must draw
oversize (`nanogpt/plot_article.py`, whose insets need the room) passes
`use_paper_style(scale=k)`.

## `utils.py`

`History` is the piece to know. It records the x-axis **explicitly** and stores
only evaluated points:

```json
{"steps": [0, 100, 200], "test_acc": [10.0, 61.2, 74.8], "test_loss": [...]}
```

The pre-refactor format wrote one entry per round and forward-filled whenever
`eval_freq > 1`, which turns unmeasured rounds into flat plateaus and makes a
cross-seed mean meaningless. `split_param_names` is the other load-bearing one:
both drivers call it, so the centralized and federated settings cannot drift apart
in which parameters count as matrices.

## `paths.py`

`results_root()` is the single definition of where runs land, honouring
`SIGNMUON_RESULTS`. `repo_relative()` strips the writing machine's prefix from the
paths a run records, so an export bundle carries no home directory; a bundle
ships with an anonymous submission. No torch import, so the exporters and plotters
can use both without one.
