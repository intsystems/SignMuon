# `tests/` — the CPU test suite

```bash
cd code
python3 -m tests.test_code      # ~1 min, no GPU, no downloads
pytest tests/test_code.py       # or under pytest
```

No GPU, no dataset, no network. Both `overnight.py` drivers run it as their first
preflight step and refuse to start a night if it fails, printing the failing
assertions rather than the test names.

[`test_code.py`](test_code.py) is the suite. The anonymity checks sit beside it in
`test_anonymize.py`, which `test_code.py` adopts when it is present, so one command
still runs everything. They are separate because that file and the `anonymize.py`
it imports are both withheld from the anonymous supplement — and a suite that a
reviewer runs should have no import that cannot resolve.

## What it is for

Most of these do not test "does the code run". They pin **claims the paper
makes**, so a change to the code cannot silently invalidate a sentence in the
paper.

| Group | The claim being pinned |
| :--- | :--- |
| Newton–Schulz | It approximates the polar factor's *direction*, and its *norm* only to a measured band, which is why magnitude is handled separately by `lr_scaling` |
| Scale invariance | Every method's iterates are unchanged under `G → cG`, which is why the paper's heavy-ball main text and its EMA algorithm boxes describe the same trajectories |
| Theorems 1–3 | The published descent inner products, recomputed in torch |
| Exact vs Newton–Schulz | The two oracles genuinely disagree on the published instances: a measured fact, pinned so it cannot drift |
| Weight decay | Coupled decay cannot shorten a scale-invariant step; it only rotates it |
| Per-layer scaling | `unit-gain` re-derives the shipped Muon aspect factor, and equalizes the per-step gain exactly |
| Federated ↔ centralized | **The load-bearing one**, below |
| Batched ↔ sequential | The synthetic sweeps' fast path reproduces `run_one` on all ten methods; the same two-implementations problem, below |
| Federated protocol | The validation split is held out before partitioning; the GPU augmentation matches torchvision; the uplink alphabet is strictly ±1 by default |
| Communication | The bit accounting follows the run's *own* alphabet, and a downlink counts as one bit exactly when the object the server distributes is already ±1 — the paper's `tab:commacct` |
| Centralized export | The paper's table aggregates per-seed tail means, in that order; every driver phase lands in the right bucket; the figures are a function of the export bundle alone |
| Anonymity | No export bundle carries an absolute path, and an unpacked bundle is excluded from both the scan and the submission |
| Plumbing | The metrics schema, multi-seed aggregation |

## The load-bearing test

`test_federated_one_client_equals_centralized` and
`test_federated_per_layer_scaling_matches_centralized`: the federated driver at
`N = 1` must reproduce the corresponding centralized optimizer **exactly**, for
all eight matrix rules, under both the `legacy` and `unit-gain` conventions.

`federated/algorithms.py` and `common/optimizers.py` are two implementations of
the same eight algorithms. Nothing but this test stops them drifting apart, and
they had drifted before it existed: in the learning-rate schedule, the routing of
biases and BatchNorm, and the weight-decay convention.

`test_batched_runner_reproduces_the_sequential_one` is the same situation again.
`synthetic/batched.py` runs a whole hyperparameter grid as one `[B, m, n]`
trajectory, which is what makes the synthetic sweeps finish in minutes rather
than hours, but it is a second implementation of all ten update rules. It is
checked against `benchmark.run_one`, over a grid containing a diverging run, a
per-config budget and all three schedules, with `stop_at_target` off and on. Run
it in float32: in bfloat16 a batched matmul can pick a different cuBLAS kernel
than a single one, and the methods that sign the LMO output are sensitive to the
last bit.

One trap, before editing these: a test whose *reference* is hand-rolled inside
the test file can pin the wrong convention. This one did. Its reference decayed
the auxiliary AdamW group, which the real centralized path never does, so it
certified an agreement that did not hold against production code. Change a
convention in `train.py`, not in the reference.

## Tests that measure rather than assert

Several derive their own tolerance instead of hard-coding one, so they cannot go
stale when the implementation changes:

* `test_coupled_decay_does_not_shrink_the_implemented_step` measures the
  Newton–Schulz norm band over the three input distributions the methods actually
  feed it, then asserts against that.
* `test_gpu_crop_and_flip_match_torchvision_distributionally` compares the
  per-row probability that an output row came from the zero padding against
  `RandomCrop(32, padding=4)` itself, over 2000 draws. A shape check would pass
  on an off-by-one in the padding; this does not.

## Adding one

Any module-level function named `test_*` is picked up by both the built-in runner
and pytest. Keep it CPU-only and under a second or two: the value of this suite is
that it runs before every night, which stops being true at ten minutes.
