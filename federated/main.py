"""Entry point for the federated experiments.

    # tune: held-out validation split; no test image is ever scored
    python3 -m federated.main --model cnn2 --dataset cifar10 \
        --algorithm signmuon --lr-scaling unit-gain --split tune \
        --rounds 400 --n_parties 11 --n_steps 3 --batch_size 64 \
        --lr 0.05 --device cuda:0 --eval_freq 20 --seed 0

    # final: partition all 50k across the clients and report on the test set
    python3 -m federated.main --model cnn2 --dataset cifar10 \
        --algorithm signmuon --lr-scaling unit-gain --split full \
        --rounds 2000 --n_parties 11 --n_steps 3 --batch_size 64 \
        --lr 0.05 --device cuda:0 --eval_freq 100 --seed 0

Results go to ``results/federated/<run_name>/seed<seed>/metrics.json``. The seed is
part of the path, so a multi-seed sweep is just the same command with different
``--seed`` values and nothing gets overwritten.

The protocol matches the centralized one (``centralized/main.py``): only ``eta_0``
is tuned, the per-layer rate is derived from the shape by ``--lr-scaling``,
selection reads validation accuracy on a 45k/5k split, and the primary reported
number is the tail mean over the last ``--last-k`` evaluations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import torch

from common.lr_scaling import RULES, resolve_rule
from common.models import CNN2, ResNet9, ResNet18
from common.utils import (History, resolve_device, results_root, run_dir, save_run,
                          seed_everything, split_param_names)
from federated.algorithms import (METHODS, communication_bits, resolve_method,
                                  run_federated)
from federated.data import DEFAULT_VAL_SEED, VAL_SIZE, build_federated_data

# The eleven methods: six proposed, then the references -- which include TWO
# full-precision Muons, one per template (see algorithms.py).
ALGORITHM_CHOICES = [
    "signmuon", "ef21signmuon", "muonusign", "muonsign",
    "ef21muonusign", "ef21muonsign",
    "muon", "muonserver", "signsgd", "sgd", "adam",
    # legacy spellings, accepted so older commands keep working
    "signmuon_cl", "signmuon_ef_21", "signmuon_ef_ud", "muon_server",
]


@dataclass
class FederatedConfig:
    dataset: str
    model: str
    algorithm: str
    rounds: int
    n_parties: int
    n_steps: int
    batch_size: int
    lr: float
    lr_aux: float
    lr_scaling: str
    scale_baselines: bool
    momentum: float
    nesterov: bool
    partition: str
    beta: float
    ns_steps: int
    split: str
    val_seed: int
    partition_seed: Optional[int]
    last_k: int
    target_acc: float
    seed: int
    device: str
    datadir: str
    run_name: str
    eval_freq: int
    weight_decay: float
    weight_decay_mode: str
    eps: float
    cosine_schedule: bool
    freeze_bn_stats: bool
    lmo_dtype: str
    loader: str
    mv_ties: str
    uplink_zeros: str
    # Parameter counts, so the communication accounting can be recomputed from
    # metrics.json alone rather than by rebuilding the model. They are a property
    # of (dataset, model), but recording them is what lets `export_article` produce
    # `tab:commacct` for a model it does not construct.
    n_matrix_params: int = 0
    n_aux_params: int = 0
    n_matrix_layers: int = 0


def get_params() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # --- general ---------------------------------------------------------
    p.add_argument("--dataset", type=str, default="cifar10", choices=["mnist", "cifar10"])
    p.add_argument("--model", type=str, default="cnn2", choices=["resnet9", "cnn2", "resnet18"])
    p.add_argument("--algorithm", type=str, default="signmuon", choices=ALGORITHM_CHOICES)
    p.add_argument("--data", type=str, default="./data_federated")
    p.add_argument("--download", action="store_true", help="Download dataset if missing")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)

    # --- federated hyperparameters ---------------------------------------
    p.add_argument("--rounds", type=int, default=2000, help="Number of communication rounds")
    p.add_argument("--n_parties", type=int, default=11,
                   help="Number of clients. ODD by default: an even count lets the "
                        "majority vote tie, and the extra voter is never decisive "
                        "(N=10 gives the vote quality of N=9 with ~15%% of "
                        "coordinates tied). Use --mv-ties to choose what a tie does")
    p.add_argument("--n_steps", type=int, default=3, help="Local accumulation steps per round")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Base rate eta_0; the per-layer rate is eta_0 * lambda(shape)")
    p.add_argument("--lr-aux", type=float, default=1e-3,
                   help="Learning rate of the auxiliary AdamW (biases, BatchNorm, head)")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--nesterov", action="store_true", help="Nesterov look-ahead momentum")
    p.add_argument("--no-cosine", action="store_true",
                   help="Disable the cosine learning-rate schedule (on by default, "
                        "uniformly for every method)")
    p.add_argument("--mv-ties", type=str, default="random", choices=["random", "zero"],
                   help="What the SERVER does with a tied majority vote: random "
                        "(default) breaks the tie with a fair coin, restoring "
                        "||s||_F = sqrt(mn), which the unit-gain rule assumes; "
                        "zero lets the coordinate abstain. "
                        "Measured to be equivalent in expected descent -- a tie "
                        "carries no information either way, and at an odd client "
                        "count under the default uplink no tie can occur")
    p.add_argument("--uplink-zeros", type=str, default="random",
                   choices=["random", "positive", "keep"],
                   help="What a CLIENT does with sign(0) on the majority-vote "
                        "uplink. random (default) is the paper's convention: an "
                        "independent +-1, so the channel is a genuine 1 bit; "
                        "positive fills deterministically; keep sends the third "
                        "symbol, making the alphabet ternary (1.02-1.16 "
                        "bits/parameter at the 0.1-3.0%% zero rate CNN2 exhibits) "
                        "and is kept only for the alphabet diagnostic. Either way, "
                        "uplink_zero_frac records the raw rate before any mapping")

    # --- per-layer learning-rate scaling ---------------------------------
    p.add_argument("--lr-scaling", type=str, default="unit-gain",
                   help="Per-layer rule: " + ", ".join(sorted(RULES))
                        + ", or power:ALPHA[,BETA]. Default 'unit-gain' is the derived "
                          "rule (see common/lr_scaling.py); 'legacy' reproduces the "
                          "paper's published federated numbers")
    p.add_argument("--scale-baselines", action="store_true",
                   help="Apply the sign-family rule to the SGD/Adam baselines too")

    # --- Muon / LMO ------------------------------------------------------
    p.add_argument("--ns_steps", type=int, default=5, help="Newton-Schulz steps for the LMO")
    p.add_argument("--lmo-dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float32"],
                   help="Working precision of the Newton-Schulz iteration")

    # --- partitioning ----------------------------------------------------
    p.add_argument("--partition", type=str, default="homo", choices=["homo", "noniid-labeldir"])
    p.add_argument("--beta", type=float, default=0.5, help="Dirichlet concentration parameter")
    p.add_argument("--partition-seed", type=int, default=None,
                   help="Seed of the client split (default: the run seed). Pin it to "
                        "hold the partition fixed while varying the run seed")

    # --- protocol ---------------------------------------------------------
    p.add_argument("--split", type=str, default="full", choices=["full", "tune"],
                   help=f"tune: hold out {VAL_SIZE} training images as validation, "
                        f"partition the remaining 45k across the clients, and report "
                        f"val accuracy -- no test image is ever scored; full: partition "
                        f"all 50k and report test accuracy (final runs)")
    p.add_argument("--val-seed", type=int, default=DEFAULT_VAL_SEED,
                   help="Seed of the train/val partition; the same default as the "
                        "centralized path, so both hold out the identical 5000 images")
    p.add_argument("--last-k", type=int, default=5,
                   help="Primary metric is the mean accuracy over the last k evaluations")
    p.add_argument("--target-acc", type=float, default=80.0,
                   help="Report the round at which accuracy first reaches this")

    # --- system ----------------------------------------------------------
    p.add_argument("--eval_freq", type=int, default=100, help="Evaluate every N rounds")
    p.add_argument("--eval-batch-size", type=int, default=1000)
    p.add_argument("--loader", type=str, default="auto", choices=["auto", "gpu", "torch"],
                   help="gpu: hold the dataset on the device and augment with tensor "
                        "ops (default on CUDA, several times faster); torch: the "
                        "torchvision DataLoader path")
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers, for --loader torch only")
    p.add_argument("--buffer-device", type=str, default=None,
                   help="Where per-client momentum/EF21 buffers live between rounds "
                        "(default: the compute device)")
    p.add_argument("--run_name", type=str, default="", help="Auto-generated if empty")
    p.add_argument("--weight_decay", type=float, default=0.0,
                   help="Applied to the MATRIX parameters only, decoupled (the "
                        "auxiliary group is never decayed). Defaults to 0, matching "
                        "the centralized primary table: that is the setting the "
                        "theorems analyse. The overnight driver re-runs the best "
                        "methods at 5e-4 as an ablation")
    p.add_argument("--weight-decay-mode", type=str, default="decoupled",
                   choices=["decoupled", "coupled"],
                   help="decoupled (default): X *= 1 - lr*wd on the server, so the "
                        "LMO sees the true gradient. coupled: wd*X is folded into "
                        "the client gradient before the oracle. Every step direction "
                        "here is positively homogeneous of degree zero, so coupling "
                        "cannot change the step LENGTH -- it only rotates the "
                        "direction. Same flag and same meaning as centralized.main.")
    p.add_argument("--eps", type=float, default=1e-8, help="Epsilon for Adam/AdamW")
    p.add_argument("--live-bn-stats", action="store_true",
                   help="Normalize with BATCH statistics during local gradient "
                        "accumulation, instead of the frozen (0,1) initialization "
                        "that is the default and what the reported numbers used. "
                        "Note what this does NOT do: the local model is rebuilt "
                        "from the global one every round and never written back, "
                        "so the running statistics it accumulates are discarded "
                        "and evaluation still uses (0,1). The flag therefore "
                        "*introduces* a train/eval mismatch rather than removing "
                        "one. Making the statistics genuinely live would need them "
                        "aggregated across clients and broadcast, which is a "
                        "different algorithm and extra communication")
    p.add_argument("--nondeterministic", action="store_true",
                   help="Allow cuDNN autotuning: faster, but not bitwise reproducible")
    return p


def build_model(dataset: str, model_name: str) -> torch.nn.Module:
    out_dim = 10
    in_ch = 3 if dataset == "cifar10" else 1
    size = 32 if dataset == "cifar10" else 28

    if model_name == "resnet18":
        return ResNet18(in_channels=in_ch, num_classes=out_dim)
    if model_name == "resnet9":
        return ResNet9(num_classes=out_dim)
    return CNN2(in_channels=in_ch, input_size=size, out_dim=out_dim)


def summarize(history: History, args, eval_name: str, method: str,
              n_matrix: int, n_aux: int, n_layers: int = 0) -> None:
    """Print the reported metrics, so a log alone is enough to fill a table row.

    The line formats are the ones ``federated.tune`` parses; keep them in step.
    """
    acc_key = f"{eval_name}_acc"
    print("\n--- summary ---")
    final = history.last(acc_key)
    print(f"{acc_key} final              : {final:.2f}%")
    tail = history.last_k_mean(acc_key, args.last_k)
    if tail is not None:
        role = "tuning criterion" if eval_name == "val" else "primary metric"
        print(f"{acc_key} mean of last {args.last_k:<4}: {tail:.2f}%   <-- {role}")
    reached = history.steps_to_target(acc_key, args.target_acc)
    print(f"rounds to {args.target_acc:g}% {eval_name} acc : "
          f"{reached if reached is not None else 'not reached'}")
    secs = history.values("round_seconds")
    if secs:
        print(f"median round time           : {sorted(secs)[len(secs) // 2]:.3f}s")
    ties = history.values("mv_tie_frac")
    if ties:
        print(f"majority-vote tie fraction  : {sum(ties) / len(ties):.4f} mean, "
              f"{ties[-1]:.4f} final")
    spreads = history.values("gain_spread")
    if spreads:
        print(f"per-layer gain spread       : {spreads[0]:.2f}x at round 1, "
              f"{spreads[-1]:.2f}x final   (the rule targets 1.00x)")

    zeros = history.values("uplink_zero_frac")
    mean_z = sum(zeros) / len(zeros) if zeros else 0.0
    if zeros:
        print(f"uplink zero fraction        : {mean_z:.4f} mean, {zeros[-1]:.4f} final")

    # The run's own alphabet, not an assumed one: under the default the zeros are
    # randomized and a symbol is a genuine bit whatever `mean_z` says, while
    # `--uplink-zeros keep` really does pay the ternary entropy.
    comm = communication_bits(method, n_matrix, n_aux, mean_z, n_layers=n_layers,
                              uplink_zeros=args.uplink_zeros)
    print(f"communication               : {comm['uplink_bits_per_param']:.2f} bits/param up, "
          f"{comm['downlink_bits_per_param']:.2f} down   "
          f"({comm['uplink_bits_per_symbol']:.2f} bits/symbol on the uplink, "
          f"--uplink-zeros {args.uplink_zeros})")
    print(f"  vs full precision         : {comm['uplink_reduction']:.1f}x uplink, "
          f"{comm['downlink_reduction']:.1f}x downlink, "
          f"{comm['round_trip_reduction']:.1f}x round trip   <-- quote the round trip")


def main() -> None:
    args = get_params().parse_args()

    seed_everything(args.seed, deterministic=not args.nondeterministic)

    if args.dataset == "mnist" and args.model == "resnet9":
        raise ValueError("ResNet9 is hardcoded for 3 input channels (CIFAR); use cnn2 for MNIST.")
    if args.dataset == "mnist":
        args.weight_decay = 0.0

    method, _ = resolve_method(args.algorithm)
    resolve_rule(args.lr_scaling)                 # fail fast on an unknown rule

    if not args.run_name:
        suffix = f"_{args.lr_scaling.replace(':', '')}"
        tune = "_tune" if args.split == "tune" else ""
        args.run_name = (f"fed_{args.dataset}_{method}_{args.partition}_{args.model}"
                         f"_r{args.rounds}_c{args.n_parties}_s{args.n_steps}{suffix}{tune}")

    device = resolve_device(args.device)

    # 1) data -- the validation images are held out before the client split, so
    #    no client holds one and a tuning run never loads the test set at all.
    data = build_federated_data(
        args.dataset, args.data,
        n_parties=args.n_parties, batch_size=args.batch_size,
        partition=args.partition, beta=args.beta, split=args.split,
        seed=args.seed, partition_seed=args.partition_seed,
        val_seed=args.val_seed, device=device, loader=args.loader,
        num_workers=args.num_workers, eval_batch_size=args.eval_batch_size,
        download=args.download,
    )

    config = FederatedConfig(
        dataset=args.dataset,
        model=args.model,
        algorithm=method,               # store the canonical name, not the alias
        rounds=args.rounds,
        n_parties=args.n_parties,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_aux=args.lr_aux,
        lr_scaling=args.lr_scaling,
        scale_baselines=bool(args.scale_baselines),
        momentum=args.momentum,
        nesterov=bool(args.nesterov),
        partition=args.partition,
        beta=args.beta,
        ns_steps=args.ns_steps,
        split=args.split,
        val_seed=args.val_seed,
        partition_seed=args.partition_seed,
        last_k=args.last_k,
        target_acc=args.target_acc,
        seed=args.seed,
        device=str(device),
        datadir=args.data,
        run_name=args.run_name,
        eval_freq=args.eval_freq,
        weight_decay=args.weight_decay,
        weight_decay_mode=args.weight_decay_mode,
        eps=args.eps,
        cosine_schedule=not args.no_cosine,
        freeze_bn_stats=not args.live_bn_stats,
        lmo_dtype=args.lmo_dtype,
        loader=data.loader,
        mv_ties=args.mv_ties,
        uplink_zeros=args.uplink_zeros,
    )

    # 2) model. The parameter split is needed twice -- once here to stamp the
    #    counts into the config, once after training to print the summary -- and
    #    `split_param_names` is the single rule both the drivers use.
    global_model = build_model(args.dataset, args.model)
    matrix_names, _ = split_param_names(global_model, 2)
    named = dict(global_model.named_parameters())
    config.n_matrix_params = sum(named[n].numel() for n in matrix_names)
    config.n_aux_params = (sum(p.numel() for p in global_model.parameters())
                           - config.n_matrix_params)
    config.n_matrix_layers = len(matrix_names)

    # 3) train
    print(f"Starting {method} on {device} (split={args.split}, "
          f"lr_scaling={args.lr_scaling}, seed={args.seed})", flush=True)
    print(f"Data: {data.n_train:,} training images over {args.n_parties} clients "
          f"(shards {min(data.shard_sizes)}-{max(data.shard_sizes)}), "
          f"{sum(len(e) for e in data.eval_sets):,} {data.eval_name} images, "
          f"loader={data.loader}", flush=True)

    history = run_federated(
        method,
        global_model,
        data.train_shards,
        data.eval_sets,
        rounds=args.rounds,
        n_steps=args.n_steps,
        lr=args.lr,
        lr_aux=args.lr_aux,
        momentum=args.momentum,
        nesterov=args.nesterov,
        weight_decay=args.weight_decay,
        decoupled_weight_decay=(args.weight_decay_mode == "decoupled"),
        ns_steps=args.ns_steps,
        eval_freq=args.eval_freq,
        device=device,
        buffer_device=args.buffer_device,
        cosine_schedule=config.cosine_schedule,
        freeze_bn_stats=config.freeze_bn_stats,
        adam_eps=args.eps,
        lmo_dtype=getattr(torch, args.lmo_dtype),
        lr_scaling=args.lr_scaling,
        scale_baselines=args.scale_baselines,
        eval_name=data.eval_name,
        mv_ties=args.mv_ties,
        uplink_zeros=args.uplink_zeros,
    )

    summarize(history, args, data.eval_name, method,
              config.n_matrix_params, config.n_aux_params,
              n_layers=config.n_matrix_layers)

    out = run_dir(results_root() / "federated", args.run_name, args.seed)
    save_run(out, config, history, model=global_model.to("cpu"))
    print(f"\nSaved run to: {out}")


if __name__ == "__main__":
    main()
