# Divergence counterexamples

Exact-SVD implementations of the eight optimizers, run on the two linear
instances of Theorems 1–3 and on the universal 2×2 instance of Theorem 4
(`th:ef_div`). CPU only, no GPU and no data.

```bash
cd code                                          # the package root
python3 -m counterexamples.problems              # the exact constants        (~2 s)
python3 -m counterexamples.run_counterexamples   # fig:divergence_plot        (~5 s)
python3 -m counterexamples.plot_ef21_construction # fig:ef21_construction     (~2 s)
python3 -m counterexamples.verify_ns_oracle      # exact vs Newton-Schulz     (~1 s)
python3 -m counterexamples.enumerate_minimality  # the Thm 2-3 minimality claims (~70 s;
                                                 #   --full is ~10 min)
```

`problems`, `enumerate_minimality` and `verify_ns_oracle` need numpy alone; the
plotting scripts also need matplotlib. `run_counterexamples`,
`plot_ef21_momentum` and `plot_ef21_construction` are the only scripts in the
repository that write into `aaai_article/`: each figure goes to `figures/` as
PNG + PDF and to `aaai_article/images/counterexamples/` as the PDF LaTeX
includes.

## Files

| File | Contents |
|------|----------|
| [`optimizers.py`](optimizers.py) | `muon_lmo`, `sign_pm1`, `scaled_sign`, the 8 optimizers (`OPTIMIZERS`) |
| [`problems.py`](problems.py) | the 4×4 and 5×5 linear instances, the universal 2×2 EF21-SignMuon instance built per `(mu, variant)`, and a `_self_check` reproducing the paper's constants |
| [`run_counterexamples.py`](run_counterexamples.py) | every method on all three problems: verdict tables and `counterexamples_main` |
| [`plot_ef21_momentum.py`](plot_ef21_momentum.py) | EF21-SignMuon at `mu ∈ {0, 0.5, 0.9, 0.95, 0.99}`, both variants, against the common slope `49/480` |
| [`plot_ef21_construction.py`](plot_ef21_construction.py) | the helper functions of the Theorem 4 objective: ramps `ψᵢ, Φᵢ`, the floored slope, the correction profile |
| [`enumerate_minimality.py`](enumerate_minimality.py) | which shapes admit a sign/polar mismatch at all |
| [`verify_ns_oracle.py`](verify_ns_oracle.py) | the same instances under the *implemented* Newton–Schulz oracle, over step counts and dtypes |

## Three conventions

* **Exact SVD LMO.** `muon_lmo` is the rank-truncated polar factor `U Vᵀ` of an
  exact SVD, which is what the theorems are stated about, not the Newton–Schulz
  polynomial used in practice. Only nonzero singular directions are kept: `U @ Vᵀ`
  from a full SVD is non-unique on rank-deficient input, and `sign(G)` is often
  low-rank. `verify_ns_oracle` measures what the practical oracle does instead;
  the two are **not** interchangeable here.
* **Randomized `sign(0)`.** Zeros map to an independent random `±1`, as in the
  paper, so every sign channel is a strict bit. Each optimizer owns its generator
  (`sign_seed`, default 0), so a run depends only on its own seed. No constant in
  Theorems 1–3 is affected: none of those matrices, or their oracle outputs, has a
  zero entry. What the convention does reach is the seven bounded methods on the
  Theorem 4 instance, whose field gradient has an exactly-zero `(1,1)` entry by
  construction. They stay bounded either way, and EF21-SignMuon's own trajectory
  is untouched, every residual along it being strictly nonzero.
* **Momentum** is the EMA form `Mₜ = μ·Mₜ₋₁ + (1−μ)·Gₜ` with an optional Nesterov
  look-ahead `M̃ₜ = (1−μ)·Gₜ + μ·Mₜ`, `μ = 0` by default. The default costs
  nothing *for the verdicts*, but the invariance is narrower than it looks. On the
  linear objectives the step is the same matrix for every `μ ∈ [0,1)` and either
  variant (Proposition 1) for the six methods whose direction is scale-invariant
  in its target; EF21-MuonUSign and EF21-MuonSign are exempt, their EF21 target
  `(1−μᵗ)G` not being constant in `t`. On the Theorem 4 instance only
  EF21-SignMuon's trajectory is μ-invariant, which is the one the theorem is
  about; the others move with `μ` while staying bounded. No verdict changes at any
  `μ`.

## Methods

`D = muon_lmo(·)` is the exact polar factor, `sign` is elementwise,
`scaled_sign(Y) = mean|Y|·sign(Y)`.

| Method | Update direction `d` (`X ← X − η·d`) |
|--------|--------------------------------------|
| `SignMuon` | `sign(LMO(M̃))`, sign **after** the LMO |
| `MuonUSign` | `LMO(sign(M̃))`, sign **before** |
| `MuonSign` | `sign(LMO(sign(M̃)))`, **both** sides |
| `EF21-SignMuon` | EF21 estimator of `LMO(M̃)`, refreshed by a scaled sign of the residual |
| `EF21-MuonUSign` | `LMO(g_est)`, `g_est` a scaled-sign EF21 estimate of `M̃` |
| `EF21-MuonSign` | as above, plus an EF21-P loop compressing the downlink model shift |
| `SignSGD` (ref) | `sign(M̃)` |
| `Muon` (ref) | `LMO(M̃)` |

## How a verdict is reached

On a linear objective `f(W) = Tr(GᵀW)` the gradient is constant, so `f` decreases
**iff** the per-step inner product `⟨G, dₜ⟩` is positive. The verdict for
Counterexamples 1–2 is therefore the sign of `mean ⟨G, dₜ⟩`: the exact criterion,
free of `η` and `T` (`verdict_mode="inner"`).

Counterexample 3 needs a different test. Its ascent is second-order, driven by the
compressor overshooting rather than by a downhill step, so `⟨G, dₜ⟩` stays
positive while `f` rises. It is read as the theorem states it,
`f(X_{t+2}) − f(X_t) = c > 0` for every `t` past the transient, and a method is
called divergent when that period-two increment is positive throughout the tail
and constant to 5% (`verdict_mode="period"`). An absolute tolerance on the tail
*slope*, which this script used until 2026-07-30, does not work: the bounded
methods oscillate over a range that scales with the instance's periodic constant
`A(μ)` (199 at `μ = 0.99`), so their apparent slope over a fixed window scales
with it too, and a threshold tight enough to catch the ascent at `μ = 0` reports
every bounded method as ascending at `μ = 0.9`. The per-period increment has no
such failure mode, a bounded trajectory being unable to hold a constant positive
increment, and it separates the eight methods correctly at every
`μ ∈ {0, 0.25, 0.5, 0.9, 0.95, 0.99}`, both variants, at `T = 60` and `T = 400`.

## Results (defaults: μ = 0, standard momentum)

**Counterexample 1, SignMuon (Theorem 1, 4×4).** `G = 1000·u₁v₁ᵀ + O` with `O`
orthogonal and `Ov₁ = u₁`, so `LMO(G) = O` exactly while
`⟨G, sign(O)⟩ = −42468/103 ≈ −412.311 < 0`. Only **SignMuon** diverges; every
other method, all three EF21 ones included, descends.

**Counterexample 2, MuonUSign and MuonSign (Theorems 2–3, 5×5).** `sign(G) = S`,
and `polar(S)` disagrees with `S` at exactly one entry, the deepest mismatch any
5×5 sign matrix admits (`D₄₂ = −1/√17`). Loading that entry gives
`⟨G, LMO(sign(G))⟩ ≈ −13.888` and `⟨G, sign(LMO(sign(G)))⟩ = −76`, so
**MuonUSign** and **MuonSign** both diverge on the *same* instance. `SignMuon`,
`SignSGD`, `Muon` and all three EF21 methods descend.

In both linear cases error feedback restores convergence, matching the paper.

**Counterexample 3, EF21-SignMuon (Theorem 4, universal 2×2).**
`ef21_signmuon_counterexample(mu, nesterov)` returns the *exact* function the
appendix builds for that momentum coefficient and variant, so code and proof
describe one object. EF21-SignMuon cannot be broken on a linear objective, its
estimator converging to `polar(G)` so that the step is genuine descent, and a plain
quadratic valley breaks it only at `μ = 0`. So the universal instance instead
*forces* a fixed sequence of LMO targets whatever `(μ, variant)` is: a rotation
`S₁`, a rank-one map `S₂`, then alternating reflections `D̄^±`. The reflections
share a small diagonal while their `O(1)` off-diagonal flips sign every step, so
the shared magnitude `αₜ = mean|Δₜ| = 24/25` overshoots the diagonal: the
estimator's `(2,2)` entry locks into the period-two cycle `{−61/200, +131/200}`,
of mean `+7/40 > 0` against its target `−7/25`, and the coordinate it drives runs
off, so `f → +∞` at exactly `49/480` per step.

The function is `f(W) = γ·h(W₂₂) + A(Φ₁(W₁₂) + Φ₂(W₂₁)) + Σₖ bₖ(W)` with
`γ = 7/12`: a linear divergence slope (levelled off above `W₂₂ = 1`, which no
iterate reaches, so `f` is bounded below), two periodic ramps `Φᵢ` sustaining the
alternating off-diagonal gradient, and three compactly supported corrections `bₖ`
seeding the first three gradients. `A` and the `bₖ` depend on `(μ, variant)`,
momentum being a positive linear filter of the gradients, which we invert.
Divergence holds for **every** `L > 0`, `η > 0`, `μ ∈ [0,1)` and both variants,
the iterate trajectory being identical across them, and only **EF21-SignMuon**
diverges while `Muon`, `EF21-MuonUSign`, `SignSGD` and the rest stay bounded. It
shows that the `Θ(σ_min/L)` step-size restriction of the conditional convergence
theorem cannot be replaced by any `(L, μ)`-only rule.

Figures: the right-hand panel of `counterexamples_main`, framed so the linear
ascent is visible above the band the seven bounded methods sit in, and
`ef21_signmuon_momentum` from [`plot_ef21_momentum.py`](plot_ef21_momentum.py).
