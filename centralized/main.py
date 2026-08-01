"""Entry point for the centralized experiments.

    # tune: held-out validation split, no test-set peeking
    python3 -m centralized.main --dataset cifar10 --model resnet18 \
        --optimizer signmuon --lr-scaling unit-gain --split tune \
        --epochs 20 --device cuda:0 --seed 0

    # final: retrain on the full 50k at the chosen hyperparameters
    python3 -m centralized.main --dataset cifar10 --model resnet18 \
        --optimizer signmuon --lr-scaling unit-gain --head-adamw always \
        --epochs 75 --device cuda:0 --seed 0

Results go to ``results/centralized/<run_name>/seed<seed>/metrics.json``. The seed
is part of the path, so a multi-seed sweep is the same command with different
``--seed`` values and nothing gets overwritten.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, fields

from centralized.data import DEFAULT_VAL_SEED, VAL_SIZE
from centralized.train import OPTIMIZER_CHOICES, resolve_optimizer_name, train
from common.lr_scaling import RULES, resolve_rule
from common.utils import results_root, run_dir, save_run, seed_everything


@dataclass
class RunConfig:
    """What is recorded in ``metrics.json`` as the identity of this run.

    An explicit allowlist, not ``vars(args)``: ``aggregate.py`` groups runs by
    their config minus a few per-run fields, so anything recorded here that does
    not change the trajectory -- ``--num-workers``, ``--nondeterministic``,
    ``--log-gain`` -- would silently split one experiment into two groups. Every
    field name matches its argparse destination, and ``from_args`` relies on that.
    """

    dataset: str
    model: str
    optimizer: str
    epochs: int
    batch_size: int
    lr: float
    lr_aux: float
    momentum: float
    nesterov: bool
    ns_steps: int
    lmo_dtype: str
    lr_scaling: str
    scale_baselines: bool
    constant_lr: bool
    head_adamw: str
    n_head_tensors: int
    weight_decay: float
    weight_decay_mode: str
    split: str
    val_seed: int
    last_k: int
    target_acc: float
    seed: int
    device: str
    data: str
    download: bool
    run_name: str

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "RunConfig":
        missing = [f.name for f in fields(cls) if not hasattr(args, f.name)]
        if missing:
            raise AttributeError(
                f"RunConfig fields with no matching CLI argument: {missing}. "
                f"Add the argument, or drop the field -- the two are kept in step "
                f"by name.")
        return cls(**{f.name: getattr(args, f.name) for f in fields(cls)})


def get_params() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=str, required=True, choices=["mnist", "cifar10"])
    p.add_argument("--model", type=str, default="cnn2", choices=["cnn2", "resnet9", "resnet18"])
    p.add_argument("--optimizer", type=str, default="signmuon",
                   choices=OPTIMIZER_CHOICES + ["ef_usignmuon", "ef_udsignmuon"])
    p.add_argument("--data", type=str, default="./data")
    p.add_argument("--download", action="store_true", help="Download dataset if missing")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Base rate eta_0; the per-layer rate is eta_0 * lambda(shape)")
    p.add_argument("--lr-aux", type=float, default=1e-3,
                   help="Rate of the auxiliary AdamW (biases, BatchNorm, head)")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--nesterov", action="store_true")
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="Applied to the MATRIX parameters only (the auxiliary group is never decayed). Defaults to 0: our theorems analyse unregularized f, the nanoGPT record we build on uses 0.0 for every group, and all ten of Mishra et al.'s best CIFAR configurations select 0 -- so 0 is the setting under which theory and experiment describe the same algorithm. The overnight driver re-runs the top methods at 5e-4 as an ablation.")
    p.add_argument("--weight-decay-mode", type=str, default="decoupled",
                   choices=["decoupled", "coupled"],
                   help="decoupled (default): X *= 1 - lr*wd, so the LMO sees the "
                        "true gradient. This is the only well-posed choice for a "
                        "norm-constrained step: the LMO output is scale-invariant, so "
                        "folding wd*X into the gradient cannot shrink X at all -- it "
                        "only rotates the direction, by an amount set by the drifting "
                        "ratio wd*||X||/||G||. coupled: wd*X is added to the gradient, so it "
                        "passes through the LMO -- what the paper's numbers used. "
                        "decoupled: X *= 1 - lr*wd, leaving the LMO to see the "
                        "true gradient geometry (the AdamW/Muon convention, and "
                        "what the federated driver does). Decoupled decay is "
                        "applied uniformly across layers, NOT scaled by the "
                        "per-layer multiplier")

    # --- per-layer learning-rate scaling ---------------------------------
    p.add_argument("--lr-scaling", type=str, default="unit-gain",
                   help="Per-layer rule: " + ", ".join(sorted(RULES))
                        + ", or power:ALPHA[,BETA]. Default 'unit-gain' is the derived "
                          "rule (see common/lr_scaling.py); 'legacy' reproduces the "
                          "paper's published numbers")
    p.add_argument("--scale-baselines", action="store_true",
                   help="Apply the sign-family rule to the SGD/Adam baselines too. "
                        "Off by default: they are run as practitioners run them, with "
                        "one global rate. Adam's step is approximately sign-like, so "
                        "both parameterizations are worth reporting")
    p.add_argument("--ns-steps", type=int, default=5, help="Newton-Schulz iterations")
    p.add_argument("--lmo-dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"],
                   help="Working precision of the Newton-Schulz iteration")
    p.add_argument("--n-head-tensors", type=int, default=2,
                   help="How many trailing tensors count as the classification head")
    p.add_argument("--head-adamw", type=str, default="auto",
                   choices=["auto", "always", "never"],
                   help="auto: AdamW auxiliary group for the LMO methods only "
                        "(reproduces the paper); always: for every method, including "
                        "the SGD/Adam/SignSGD baselines (apples-to-apples); never: none")

    # --- protocol ---------------------------------------------------------
    p.add_argument("--split", type=str, default="full", choices=["full", "tune"],
                   help=f"tune: hold out {VAL_SIZE} training examples as validation and "
                        f"select on val_acc; full: train on everything (final runs)")
    p.add_argument("--val-seed", type=int, default=DEFAULT_VAL_SEED,
                   help="Seed of the train/val partition; keep fixed across methods "
                        "and run seeds so the split is not a confounder")
    p.add_argument("--last-k", type=int, default=5,
                   help="Primary metric is the mean test accuracy over the last k epochs")
    p.add_argument("--target-acc", type=float, default=90.0,
                   help="Report the epoch at which test accuracy first reaches this")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--constant-lr", action="store_true",
                   help="Disable the cosine schedule. Required for the --log-gain "
                        "diagnostic: with annealing the accumulated update saturates, "
                        "so its growth exponent would measure the schedule rather "
                        "than the coherence of successive steps")
    p.add_argument("--log-gain", action="store_true",
                   help="Record the realized gain of the accumulated update per layer "
                        "(the diagnostic that separates unit-gain from mup)")
    p.add_argument("--nondeterministic", action="store_true",
                   help="Allow cuDNN autotuning: faster, but not bitwise reproducible")

    p.add_argument("--run-name", type=str, default="",
                   help="Folder under results/centralized/ ; auto-generated if empty")
    return p


def main() -> None:
    args = get_params().parse_args()
    seed_everything(args.seed, deterministic=not args.nondeterministic)

    args.optimizer = resolve_optimizer_name(args.optimizer)
    resolve_rule(args.lr_scaling)                 # fail fast on an unknown rule
    if args.log_gain and not args.constant_lr:
        print("WARNING: --log-gain without --constant-lr: the cosine schedule "
              "saturates the accumulated update, so the growth exponent will "
              "measure the schedule rather than the alignment of successive "
              "steps. Pass --constant-lr.", file=sys.stderr)
    if args.dataset == "mnist":
        args.weight_decay = 0.0

    if not args.run_name:
        suffix = f"_{args.lr_scaling.replace(':', '')}"
        if args.scale_baselines:
            suffix += "_scaledbase"
        tune = "_tune" if args.split == "tune" else ""
        args.run_name = f"{args.dataset}_{args.model}_{args.optimizer}{suffix}{tune}"

    config = RunConfig.from_args(args)

    print(f"Training {args.optimizer} on {args.dataset}/{args.model} "
          f"(lr_scaling={args.lr_scaling}, split={args.split}, seed={args.seed})")
    model, history = train(args)

    out = run_dir(results_root() / "centralized", args.run_name, args.seed)
    save_run(out, config, history, model=model.to("cpu"))
    print(f"\nSaved run to: {out}")


if __name__ == "__main__":
    main()
