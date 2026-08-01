# NanoGPT speedrun results (record #40, 2330 steps, 8×H100, single run per method)

Written by `code/nanogpt/make_tables.py` from `runs.csv` / `diagnostics.csv`.
Every number the paper quotes for this arm is below; `MANIFEST.json` says
which log, and which build of the optimizer module, produced each one.

## Headline (`tab:nanogpt`)

| method | step | eta_0 | val. loss | steps to 3.35 | rel. Muon | ms/step |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Muon | lmo | 0.06 | 3.2785 | 1903 | 1.00x | 61.73 |
| EF21-SignMuon | lmo | 0.06 | 3.2860 | 1949 | 1.02x | 62.00 |
| SignMuon | sign | 0.03 | 3.2881 | 1942 | 1.02x | 61.95 |
| MuonUSign | lmo | 0.06 | 3.2959 | 1990 | 1.05x | 61.65 |
| EF21-MuonUSign | lmo | 0.06 | 3.3203 | 2157 | 1.13x | 62.03 |
| EF21-MuonSign (**W**) | lmo | 0.06 | 3.3213 | 2164 | 1.14x | 62.06 |
| MuonSign | sign | 0.03 | 3.3249 | 2175 | 1.14x | 61.41 |
| SignSGD | sign | 0.03 | 3.4049 | -- | -- | 61.48 |

EF21-MuonSign's exact server model **X** ends at `5.5198`, against `3.3213` at the broadcast model **W** above: the two models of that method, not two runs.

## Control: does the `Muon` arm reproduce record #40?

Upstream record #40, mean of its five 8xH100 logs at step 2330: `3.2780 +/- 0.0009`.
This port's `Muon`: `3.2785` (+0.0005, 0.6 upstream sd). PASS.

## Wall-clock

`61.41`--`62.06` ms/step across the eight methods, a spread of 1.1%: no method pays a measurable premium for its compressor or its error-feedback buffers. All eight are above record #40's own `60.4` ms/step, by the same margin, because this port replaces its Triton kernels and batched sharded transport with a pure-torch per-parameter equivalent.

## Compressor diagnostics at the final step (`tab:nanogpt_diag`)

Medians over the identical layers of each type. `alpha` is the contraction the scaled sign achieves on that round's residual (`2/pi = 0.637` for an isotropic residual, `1/d` in the worst case); `lag` is `||target - estimator||_F / ||target||_F`; `gap` is `||X - W||_F / ||W||_F`. The two gates are **not** in the paper's table and are marked here, because the ranges the appendix quotes are over the layer types only.

| method | layer | count | uplink alpha | uplink lag | downlink alpha | gap |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| EF21-SignMuon | `qkvo_w` | 10 | 0.6399 | 0.6609 | -- | -- |
| EF21-SignMuon | `c_fc` | 11 | 0.633 | 0.6166 | -- | -- |
| EF21-SignMuon | `c_proj` | 11 | 0.6342 | 0.6212 | -- | -- |
| EF21-SignMuon | `attn_gate` (gate) | 10 | 0.6431 | 0.651 | -- | -- |
| EF21-SignMuon | `smear_gate` (gate) | 1 | 0.7196 | 0.3589 | -- | -- |
| EF21-MuonUSign | `qkvo_w` | 10 | 0.3735 | 0.8057 | -- | -- |
| EF21-MuonUSign | `c_fc` | 11 | 0.3227 | 0.8158 | -- | -- |
| EF21-MuonUSign | `c_proj` | 11 | 0.5971 | 0.6163 | -- | -- |
| EF21-MuonUSign | `attn_gate` (gate) | 10 | 0.5818 | 0.7102 | -- | -- |
| EF21-MuonUSign | `smear_gate` (gate) | 1 | 0.6528 | 0.7963 | -- | -- |
| EF21-MuonSign | `qkvo_w` | 10 | 0.4061 | 0.7947 | 0.3803 | 0.001121 |
| EF21-MuonSign | `c_fc` | 11 | 0.3547 | 0.7849 | 0.2529 | 0.001339 |
| EF21-MuonSign | `c_proj` | 11 | 0.5962 | 0.609 | 0.0001233 | 0.08274 |
| EF21-MuonSign | `attn_gate` (gate) | 10 | 0.592 | 0.5982 | 0.3507 | 0.00853 |
| EF21-MuonSign | `smear_gate` (gate) | 1 | 0.8276 | 0.2773 | 0.1651 | 0.003848 |

Over the layer types only: uplink `alpha` in [0.32, 0.64], uplink lag in [0.61, 0.82].

## Provenance

| run | optimizer | seed | torch | driver | optimizer build |
| :--- | :--- | :--- | :--- | :--- | :--- |
| EF21-MuonSign_lr0.06_e6770317.txt | EF21-MuonSign | unseeded | 2.10.0+cu128 | 595.71.05 | A |
| EF21-MuonUSign_lr0.06_2717df49.txt | EF21-MuonUSign | unseeded | 2.10.0+cu128 | 595.71.05 | A |
| EF21-SignMuon_lr0.06_bb803ec4.txt | EF21-SignMuon | unseeded | 2.10.0+cu128 | 595.71.05 | A |
| Muon_lr0.06_5db64adc.txt | Muon | unseeded | 2.10.0+cu128 | 595.71.05 | B |
| MuonSign_lr0.03_8ae069a3.txt | MuonSign | unseeded | 2.10.0+cu128 | 595.71.05 | B |
| MuonUSign_lr0.06_d9721bde.txt | MuonUSign | unseeded | 2.10.0+cu128 | 595.71.05 | B |
| SignMuon_lr0.03_19f64fe1.txt | SignMuon | unseeded | 2.10.0+cu128 | 595.71.05 | B |
| SignSGD_lr0.03_1f0db2d4.txt | SignSGD | unseeded | 2.10.0+cu128 | 595.71.05 | B |

2 distinct build(s) of `signmuon_optimizers.py` produced these logs, keyed by the SHA-256 of the copy each log embeds (`MANIFEST.json` has the hashes). What differs between them, and why it does not mix vintages that should not be mixed, is `code/nanogpt/README.md`, section "Provenance".
