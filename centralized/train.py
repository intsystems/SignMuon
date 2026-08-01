"""Centralized training for CIFAR-10 / MNIST, shared by every optimizer.

Protocol (see ``REPRODUCE.md`` for the commands)
-----------------------------------------------
* **Parameter routing.** Matrix parameters (``ndim >= 2``, excluding the
  classification head) get the LMO/sign rule; biases, BatchNorm scales and the
  head get AdamW. With ``--head-adamw always`` this holds for *every* method,
  baselines included, so the only difference between two runs is the matrix rule.
* **Per-layer learning rate.** ``eta_layer = eta_0 * lambda(family, shape)`` with
  ``lambda`` set analytically by ``common.lr_scaling`` -- derived, not tuned. Only
  ``eta_0`` is tuned, and it is a shape-free quantity. The shape factor is applied
  as a per-parameter-group multiplier, so ``scale_aspect`` is switched **off**
  inside the LMO to avoid counting the aspect ratio twice.
* **Model selection.** Tuning reads ``val_acc`` from a held-out split; the reported
  test number is the tail mean of the last ``--last-k`` epochs at a fixed epoch
  budget, never an early stop on test.

Metrics per epoch: ``train_loss/acc``, ``val_loss/acc`` (when tuning),
``test_loss/acc``, ``epoch_seconds``, and -- with ``--log-gain`` -- the realized gain
of the accumulated update, which is what distinguishes the ``unit-gain`` and
``mup`` scaling rules empirically.
"""

from __future__ import annotations

import math
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from centralized.data import DEFAULT_VAL_SEED, build_loaders
from common.lr_scaling import (FAMILY_SIGN, describe_rule, layer_multiplier,
                               resolve_rule)
from common.models import CNN2, ResNet9, ResNet18
from common.optimizers import (
    EF21MuonSign,
    EF21MuonUSign,
    EF21SignMuon,
    Muon,
    MuonSign,
    MuonUSign,
    SignMuon,
    SignSGD,
)
from common.utils import History, resolve_device, split_param_names

#: Methods implemented in ``common.optimizers``, i.e. everything following the
#: paper's matrix-parameter template. ``sgd``/``adam`` come from torch.
LMO_FAMILY = {
    "signmuon": SignMuon,
    "ef21signmuon": EF21SignMuon,
    "muonusign": MuonUSign,
    "muonsign": MuonSign,
    "ef21muonusign": EF21MuonUSign,
    "ef21muonsign": EF21MuonSign,
    "muon": Muon,
    "signsgd": SignSGD,
}

OPTIMIZER_CHOICES = list(LMO_FAMILY) + ["sgd", "adam"]

#: Methods that use the LMO. ``signsgd`` is implemented here but takes no LMO, so
#: under ``--head-adamw auto`` it is treated like the other baselines.
MATRIX_RULE_METHODS = {"signmuon", "ef21signmuon", "muonusign", "muonsign",
                       "ef21muonusign", "ef21muonsign", "muon"}

#: Pre-refactor CLI spellings, kept so old commands and logs still resolve. There
#: is deliberately no entry for the old ``MuonSign``, which used to denote the
#: sign-BEFORE method the paper now calls MuonUSign: resolving it silently would
#: change the algorithm rather than the label.
ALIASES = {"ef_usignmuon": "ef21muonusign", "ef_udsignmuon": "ef21muonsign"}


def resolve_optimizer_name(name: str) -> str:
    key = name.strip().lower().replace("-", "")
    key = ALIASES.get(key, key)
    if key not in OPTIMIZER_CHOICES:
        raise ValueError(f"Unknown optimizer {name!r}. Available: {OPTIMIZER_CHOICES}")
    return key


# --------------------------------------------------------------------------
# Epoch loops
# --------------------------------------------------------------------------


def train_epoch(model, loader, optimizers, device) -> Tuple[float, float]:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = correct = total = 0.0

    opt_main, opt_aux = optimizers
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        for opt in (opt_main, opt_aux):
            if opt is not None:
                opt.zero_grad(set_to_none=True)

        out = model(x)
        loss = criterion(out, y)
        loss.backward()

        for opt in (opt_main, opt_aux):
            if opt is not None:
                opt.step()

        total_loss += loss.item() * y.size(0)
        correct += (out.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, device) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = correct = total = 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        total_loss += criterion(out, y).item()
        correct += (out.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, opt_main=None) -> Tuple[float, float]:
    """Evaluate, exposing the *exact* model for the bidirectional EF method.

    ``EF21MuonSign`` keeps the broadcast model ``W`` in ``p.data`` and the exact
    server model ``X`` in its state; ``X`` is the iterate the convergence theory
    bounds, so metrics are computed there.
    """
    using_exact = getattr(opt_main, "using_exact", None)
    if using_exact is None:
        return eval_epoch(model, loader, device)
    with using_exact():
        return eval_epoch(model, loader, device)


# --------------------------------------------------------------------------
# Optimizer construction, with per-layer scaling
# --------------------------------------------------------------------------


def build_optimizers(model, args):
    """Return ``(opt_main, opt_aux, info)``.

    ``opt_main`` carries **one parameter group per matrix parameter**, each with
    its own ``lambda_mult`` from the scaling rule. ``CosineAnnealingLR`` scales
    every group's ``base_lr`` independently, so the schedule is unaffected.

    ``--head-adamw``:

    * ``auto`` -- the LMO methods put matrix parameters under their own rule and
      biases/BatchNorm/head under AdamW; ``sgd``, ``adam`` and ``signsgd`` apply
      their single rule to every parameter, as published.
    * ``always`` -- every method gets the AdamW auxiliary group, so the only
      difference between runs is the matrix rule. The apples-to-apples setting.
    * ``never`` -- no auxiliary group at all.
    """
    name = resolve_optimizer_name(args.optimizer)
    rule = resolve_rule(getattr(args, "lr_scaling", "unit-gain"))
    n_head = getattr(args, "n_head_tensors", 2)
    matrix_names, aux_names = split_param_names(model, n_head)

    mode = getattr(args, "head_adamw", "auto")
    split = (name in MATRIX_RULE_METHODS) if mode == "auto" else (mode == "always")

    named = dict(model.named_parameters())
    if split:
        main_names = list(matrix_names)
        aux_params = [named[n] for n in aux_names]
    else:
        main_names = [n for n, p in model.named_parameters() if p.requires_grad]
        aux_params = []

    common = dict(
        lr=args.lr,
        momentum=args.momentum,
        nesterov=args.nesterov,
        weight_decay=args.weight_decay,
        decoupled_weight_decay=getattr(args, "weight_decay_mode", "decoupled")
        == "decoupled",
        ns_steps=getattr(args, "ns_steps", 5),
        lmo_dtype=getattr(torch, getattr(args, "lmo_dtype", "bfloat16")),
        # The shape factor lives in ``lambda_mult`` below; applying it inside the
        # LMO as well would count the aspect ratio twice.
        scale_aspect=False,
    )

    info: Dict[str, object] = {"rule": rule.name, "multipliers": {}, "family": None,
                               "shapes": []}

    if name in LMO_FAMILY:
        cls = LMO_FAMILY[name]
        groups = []
        for n in main_names:
            p = named[n]
            # 1-D parameters have no matrix structure (the LMO is the identity on
            # them), so no shape rule applies. They only reach here when the
            # auxiliary group is disabled.
            lam = layer_multiplier(rule, cls.family, tuple(p.shape)) if p.ndim >= 2 else 1.0
            groups.append({"params": [p], "lambda_mult": lam, "name": n})
            info["multipliers"][n] = lam
        opt_main = cls(groups, **common)
        info["family"] = cls.family
        info["shapes"] = [(n, tuple(named[n].shape)) for n in main_names
                          if named[n].ndim >= 2]
    else:
        # torch's SGD/Adam have no ``lambda_mult``, so when the baselines are scaled
        # the multiplier is folded straight into each group's ``lr``. That is exactly
        # equivalent (CosineAnnealingLR anneals every group from its own base_lr); the
        # only difference is that ``lr`` is then already shape-adjusted in the logs.
        scale = bool(getattr(args, "scale_baselines", False))
        groups = []
        for n in main_names:
            p = named[n]
            lam = (layer_multiplier(rule, FAMILY_SIGN, tuple(p.shape))
                   if scale and p.ndim >= 2 else 1.0)
            groups.append({"params": [p], "lr": args.lr * lam, "name": n})
            info["multipliers"][n] = lam
        if scale:
            # Adam's step is approximately a sign step (|s_ij| ~ 1), so the sign-family
            # rule is the consistent choice when the baselines are scaled at all.
            info["family"] = FAMILY_SIGN
            info["shapes"] = [(n, tuple(named[n].shape)) for n in main_names
                              if named[n].ndim >= 2]
        decoupled = getattr(args, "weight_decay_mode", "decoupled") == "decoupled"
        if name == "sgd":
            # torch's SGD couples the decay (``g <- g + wd*p``), and that is the right
            # convention here: SGD's step is ``eta*m``, which is NOT scale-invariant,
            # so coupling really does minimize ``f + (wd/2)||W||^2``. There is nothing
            # to switch, and forcing decoupling on it would be the deviation.
            opt_main = torch.optim.SGD(groups, lr=args.lr, momentum=args.momentum,
                                       nesterov=args.nesterov,
                                       weight_decay=args.weight_decay)
        else:
            # Adam's step *is* approximately scale-invariant, so the same argument as
            # for the LMO methods applies: use AdamW (decoupled) unless the coupled
            # ablation was asked for. Running plain Adam against decoupled LMO methods
            # would confound the comparison with the AdamW-vs-Adam difference.
            adam_cls = torch.optim.AdamW if decoupled else torch.optim.Adam
            opt_main = adam_cls(groups, lr=args.lr, weight_decay=args.weight_decay)

    # The auxiliary group (BatchNorm scales, biases, and under ``--head-adamw always``
    # the classifier) is never decayed: decaying normalization scales is not standard
    # practice, and keeping it at 0 for every method means ``--weight-decay`` describes
    # the matrix parameters only -- the ones the shape rule and the LMO act on.
    opt_aux = (torch.optim.AdamW(aux_params, lr=args.lr_aux, weight_decay=0.0)
               if aux_params else None)
    info["n_matrix_params"] = sum(named[n].numel() for n in main_names)
    info["n_aux_params"] = sum(p.numel() for p in aux_params)
    return opt_main, opt_aux, info


def build_model(dataset: str, model_name: str) -> nn.Module:
    if dataset == "cifar10":
        if model_name == "resnet18":
            return ResNet18(in_channels=3, num_classes=10)
        if model_name == "resnet9":
            return ResNet9(num_classes=10)
        return CNN2(in_channels=3, input_size=32, out_dim=10)
    if model_name == "resnet9":
        raise ValueError("ResNet9 is hardcoded for 3 input channels (CIFAR).")
    if model_name == "resnet18":
        return ResNet18(in_channels=1, num_classes=10)
    return CNN2(in_channels=1, input_size=28, n_kernels=32, out_dim=10)


# --------------------------------------------------------------------------
# Gain diagnostics
# --------------------------------------------------------------------------


@torch.no_grad()
def accumulated_gain(model, snapshot: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Realized RMS gain of the accumulated update, per matrix parameter.

    ``gamma(A) = ||A||_F / sqrt(fan_out)`` -- exactly the quantity
    ``common.lr_scaling`` reasons about. Tracking it against the epoch count
    settles the one open modelling question in the scaling rule: growth like
    ``sqrt(t)`` means successive sign steps stay incoherent (favouring
    ``unit-gain``), growth like ``t`` means they align (favouring ``mup``).
    """
    out = {}
    for n, p in model.named_parameters():
        if n not in snapshot:
            continue
        out[n] = float((p.detach() - snapshot[n]).norm() / math.sqrt(p.shape[0]))
    return out


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


def train(args) -> Tuple[nn.Module, History]:
    """Train for ``args.epochs`` and return ``(model, History)``.

    The history is indexed by epoch; epoch 0 records the untrained model.
    """
    device = resolve_device(args.device)
    seed = getattr(args, "seed", None)
    split = getattr(args, "split", "full")

    train_dl, val_dl, test_dl = build_loaders(
        args.dataset, args.data, batch_size=args.batch_size,
        download=args.download, seed=seed, split=split,
        val_seed=getattr(args, "val_seed", DEFAULT_VAL_SEED),
        num_workers=getattr(args, "num_workers", 4),
    )

    model = build_model(args.dataset, args.model).to(device)
    opt_main, opt_aux, info = build_optimizers(model, args)

    rule = resolve_rule(getattr(args, "lr_scaling", "unit-gain"))
    if info["family"] and info["shapes"]:
        print(describe_rule(rule, info["family"], info["shapes"]))
    n_all = info["n_matrix_params"] + info["n_aux_params"]
    if n_all:
        frac = 100.0 * info["n_aux_params"] / n_all
        # The auxiliary group is transmitted uncompressed, so this fraction is what
        # turns "1 bit/param" into "1 bit/param + eps".
        print(f"Parameters: {info['n_matrix_params']:,} matrix (compressed) + "
              f"{info['n_aux_params']:,} auxiliary ({frac:.3f}% uncompressed)")

    # A constant rate is required by the --log-gain diagnostic: with annealing the
    # accumulated update saturates, so a growth exponent in t would measure the
    # schedule rather than the coherence of successive steps. Every other run
    # anneals, since the schedule shape is part of the fixed epoch budget.
    constant_lr = bool(getattr(args, "constant_lr", False))
    scheduler_main = (None if constant_lr else
                      torch.optim.lr_scheduler.CosineAnnealingLR(
                          opt_main, T_max=args.epochs))
    scheduler_aux = (None if constant_lr or opt_aux is None else
                     torch.optim.lr_scheduler.CosineAnnealingLR(
                         opt_aux, T_max=args.epochs))
    if constant_lr:
        print("Learning rate held CONSTANT (no cosine schedule)")

    log_gain = bool(getattr(args, "log_gain", False))
    snapshot = ({n: p.detach().clone() for n, p in model.named_parameters() if p.ndim >= 2}
                if log_gain else {})

    history = History()          # accuracies in percent, matching the federated driver

    def _record(epoch: int, tr: Tuple[float, float], extra: Optional[dict] = None) -> None:
        # For EF21-MuonSign only, ``tr`` and the evaluated metrics live on
        # DIFFERENT iterates, and necessarily so: the training pass must
        # differentiate at the broadcast model W (that is what the algorithm
        # sends the client), while X is the exact server iterate the convergence
        # theory bounds and the one worth reporting. Recomputing train metrics at
        # X would cost a second full pass per epoch. So `train_*` is at W and
        # `test_*`/`val_*` are at X; do not read the two curves as one trajectory
        # for that method. Every other method has W == X and the question is moot.
        loss_te, acc_te = evaluate(model, test_dl, device, opt_main)
        values = {"train_loss": tr[0], "train_acc": 100 * tr[1],
                  "test_loss": loss_te, "test_acc": 100 * acc_te}
        if val_dl is not None:
            loss_va, acc_va = evaluate(model, val_dl, device, opt_main)
            values.update(val_loss=loss_va, val_acc=100 * acc_va)
        if extra:
            values.update(extra)
        history.record(epoch, **values)
        msg = (f"Epoch {epoch}/{args.epochs} | train loss {tr[0]:.4f}, "
               f"acc {100 * tr[1]:.2f}% | test loss {loss_te:.4f}, acc {100 * acc_te:.2f}%")
        if val_dl is not None:
            msg += f" | val acc {values['val_acc']:.2f}%"
        print(msg, flush=True)

    _record(0, evaluate(model, train_dl, device, opt_main))

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        tr = train_epoch(model, train_dl, (opt_main, opt_aux), device)

        if scheduler_main is not None:
            scheduler_main.step()
        if scheduler_aux is not None:
            scheduler_aux.step()

        extra = {"epoch_seconds": time.perf_counter() - t0}
        if log_gain:
            gains = sorted(accumulated_gain(model, snapshot).values())
            if gains:
                # Summary series only, to keep metrics.json small; the per-layer
                # detail is printed once at the end.
                extra.update(gain_min=gains[0], gain_median=gains[len(gains) // 2],
                             gain_max=gains[-1])
        _record(epoch, tr, extra)

    # Leave the caller holding the exact model, not the compressed broadcast one.
    if hasattr(opt_main, "restore_exact"):
        opt_main.restore_exact()

    if log_gain:
        print("\nAccumulated-update gain, ||X_T - X_0||_F / sqrt(fan_out), per parameter:")
        for n, g in accumulated_gain(model, snapshot).items():
            print(f"  {n:<42}{g:>12.5g}")

    _summarize(history, args)
    return model, history


def _summarize(history: History, args) -> None:
    """Print the reported metrics, so a log alone is enough to fill a table row."""
    last_k = getattr(args, "last_k", 5)
    target = getattr(args, "target_acc", 90.0)
    print("\n--- summary ---")
    print(f"test_acc final              : {history.last('test_acc'):.2f}%")
    tail = history.last_k_mean("test_acc", last_k)
    if tail is not None:
        print(f"test_acc mean of last {last_k:<4}: {tail:.2f}%   <-- primary metric")
    best_val = history.argbest("val_acc", "max")
    if best_val is not None:
        print(f"test_acc @ best-val epoch {best_val:<3}: "
              f"{history.at('test_acc', best_val):.2f}%  "
              f"(val {history.at('val_acc', best_val):.2f}%)")
        print(f"val_acc  mean of last {last_k:<4}: "
              f"{history.last_k_mean('val_acc', last_k):.2f}%   <-- tuning criterion")
    else:
        # Full-split runs train on all 50k, so there is no held-out validation set
        # and no honest early-stopping number. Say so rather than quietly omitting
        # the row: the primary metric is the fixed-budget tail mean anyway.
        print("test_acc @ best-val epoch  : n/a (no validation split; "
              "--split full trains on all 50k)")
    # Underfitting diagnostic. On ResNet-18/CIFAR-10 every method reaches
    # ~100%, so this separates nothing there -- which is itself the finding, and
    # is only visible if the number is printed.
    train_tail = history.last_k_mean("train_acc", last_k)
    if train_tail is not None:
        at_w = getattr(args, "optimizer", "") == "ef21muonsign"
        note = ("<-- measured at the BROADCAST model W, not the exact model X "
                "that test/val use") if at_w else "<-- underfitting diagnostic"
        print(f"train_acc mean of last {last_k:<3}: {train_tail:.2f}%   {note}")
    # Reported, with the caveat: test cross-entropy rises late in training while
    # test accuracy keeps improving. That is the standard overconfidence regime
    # once train accuracy saturates, not a bug and not a reason to early-stop on
    # test. Accuracy is the primary metric, designated a priori.
    test_loss = history.last_k_mean("test_loss", last_k)
    if test_loss is not None:
        print(f"test_loss mean of last {last_k:<3}: {test_loss:.4f}   "
              f"(rises late; see REPRODUCE.md 4a)")
    ep = history.steps_to_target("test_acc", target)
    print(f"epochs to {target:g}% test acc  : {ep if ep is not None else 'not reached'}")
    secs = history.values("epoch_seconds")
    if secs:
        print(f"median epoch time           : {sorted(secs)[len(secs) // 2]:.2f}s")
