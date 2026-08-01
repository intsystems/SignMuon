"""Federated implementations of the paper's methods, as one parameterized driver.

The paper's Table "The six federated methods as instantiations of the two
templates" says every method is fixed by three choices: *where the Muon LMO is
evaluated*, the *uplink* compressor, and the *downlink* compressor. This module
encodes exactly that, so all eleven algorithms share one training loop and cannot
drift apart in learning-rate schedule, parameter routing, evaluation, or
weight-decay handling:

=================  ========  ===============  ==========  ========  ====================
Method             LMO       Uplink           Downlink    Family    Paper reference
=================  ========  ===============  ==========  ========  ====================
``signmuon``       worker    sign / MV        exact       sign      ``fed_workerlmo`` r1
``ef21signmuon``   worker    EF21             exact       lmo       ``fed_workerlmo`` r2
``muonusign``      server    sign / MV        exact       lmo       ``fed_serverlmo`` r3
``muonsign``       server    sign / MV        sign        sign      ``fed_serverlmo`` r4
``ef21muonusign``  server    EF21             exact       lmo       ``fed_serverlmo`` r5
``ef21muonsign``   server    EF21             EF21-P      lmo       ``fed_serverlmo`` r6
``muon``           worker    exact (average)  exact       lmo       control for rows 1-2
``muonserver``     server    exact (average)  exact       lmo       control for rows 3-6
``signsgd``        none      sign / MV        exact       sign      reference
``sgd``            none      exact (average)  exact       --        reference
``adam``           none      exact (average)  exact       --        reference (server Adam)
=================  ========  ===============  ==========  ========  ====================

There are **two** full-precision Muons, one per template, because a compressed
method must be measured against the uncompressed version of its own template; see
the ``METHODS`` table below for why they are not interchangeable.

Conventions shared by every method (this uniformity is the point of the module)
-----------------------------------------------------------------------------
* **Parameter routing.** The LMO/sign rule applies only to matrix parameters
  (``ndim >= 2``, excluding the classification head). Biases, BatchNorm scales
  and the head go to a server-side AdamW fed with the plain averaged gradient,
  as the paper specifies -- and identically for every method, so the *only*
  difference between two runs is the matrix-parameter rule. That auxiliary group
  is **never weight-decayed**, matching ``centralized.train.build_optimizers``.
* **Per-layer learning rate.** ``eta_layer = eta_0 * lambda(family, shape)`` with
  ``lambda`` set analytically by ``common.lr_scaling`` -- derived, not tuned, and
  a deterministic function of the layer shape known to server and clients alike,
  so it costs nothing in communication and does not affect the 1-bit claim. The
  factor is applied outside the oracle and ``scale_aspect`` is therefore switched
  **off** inside ``muon_lmo``, exactly as the centralized path does; under the
  ``legacy`` rule the two conventions coincide bit-for-bit.
* **Learning rate schedule.** One cosine schedule (``utils.cosine_lr``), applied
  to both the main step size and the AdamW auxiliary rate, for every method.
* **Weight decay** is applied exactly once, decoupled, on the server
  (``X *= 1 - lr*wd`` for matrix parameters, *unscaled* by the per-layer factor,
  as centrally). Clients accumulate *pure* gradients, so the LMO always sees the
  gradient geometry rather than a shrinkage-perturbed version of it.
* **Momentum** is the EMA form of the paper's boxes,
  ``M = mu*M + (1-mu)*G``, which is trajectory-identical to the heavy-ball form
  for every sign/LMO method (all are positively homogeneous in ``M``). The one
  exception is ``sgd``, whose step *is* the momentum buffer and which therefore
  keeps the heavy-ball convention of ``torch.optim.SGD``.

Every sign channel is a strict one-bit channel (the paper's convention)
-----------------------------------------------------------------------
``sign(x)`` is **zero** at ``x = 0``, and that is not a corner case here:
``polar(M)`` has an exactly-zero column wherever ``M`` does, and ``M`` does
wherever a feature was zero for the whole local batch -- which after ReLU and
MaxPool is common. Measured on CNN2 over the 2026-07-30 five-seed runs, the raw
rate on the majority-vote uplink is **0.1% to 3.0% of sign entries per round**,
and it is strongly method-dependent: SignSGD and MuonSign 1.4-3.0%, MuonUSign
0.1-1.4%, SignMuon 0.19-0.20%. Signing the LMO *output* produces far fewer exact
zeros than signing the momentum. (``uplink_zero_frac`` records this raw rate on
every run. An earlier note here said 8-17%, from a 2026-07-27 measurement that
no run since reproduces; nothing depended on it, because the randomization below
makes the cost independent of the rate.)

The paper's convention, and the default here, maps every exact zero to an
independent random ``+-1`` (``common.optimizers.sign_pm1``), on **all** sign
channels -- the majority-vote uplink, both EF21 residual channels, and the
MuonSign sign-downlink. This makes each channel exactly one bit per parameter
*whatever the zero rate*, so ``uplink_zero_frac`` is now a diagnostic and no
longer feeds the bit accounting. It costs nothing in expected descent (a zeroed
entry carries no directional information), and it keeps the scaled-sign
contraction lemma, whose identity
``||C(Y)-Y||_F^2 = ||Y||_F^2 - ||Y||_1^2/d`` holds for any tie-breaking.
With ``+-1`` client messages and an **odd** client count the majority vote
cannot tie; ``mv_tie_frac`` still measures raw ties, counted *before* any
tie-breaking, so it describes the vote and not the rule.

``uplink_zeros`` (majority-vote uplink): ``"random"`` (default, the paper's
convention), ``"positive"`` (deterministic ``+1`` fill), or ``"keep"`` (the
pre-convention ternary behaviour, for the alphabet diagnostic).

``mv_ties`` (server tie-break, reachable only at an even client count under
the default uplink): ``"random"`` (default) breaks ties with a fair coin,
``"zero"`` abstains.

How many clients?
-----------------
Alignment ``A = E[<truth, s_agg>]/(mn)`` -- the fraction of a full-strength
descent step actually delivered -- over 400k coordinates, each client correct with
probability ``0.65`` and transmitting zero with probability ``q``:

======  =============  =============  =============  =============
``N``   tie % (q=0)    ``A`` (q=0)    tie % (q=.10)  ``A`` (q=.10)
======  =============  =============  =============  =============
9        0.00           0.6573         8.38           0.6183
10      15.32           0.6558         9.23           0.6410
11       0.00           0.7029         7.22           0.6676
15       0.00           0.7747         5.54           0.7406
======  =============  =============  =============  =============

Reproduce with ``python3 -m federated.tune --stage votes``. The table is Monte
Carlo over 400k coordinates, so the last digit moves between runs; differences
below ~0.002 are noise, which is itself the point for the ``N = 9`` vs ``N = 10``
comparison.

At ``q = 0`` the classical parity pathology is visible: ``N = 10`` buys nothing
over ``N = 9`` because an even vote's extra voter is never decisive, it only
creates ties. At the zero rate CNN2 actually exhibits that effect is washed out
and alignment is simply monotone in ``N``. Either way ``N = 11`` beats ``N = 10``,
which is why it is the default -- but the reason is "more voters", not parity, and
an odd count does **not** by itself remove ties.

The tie *rule* does not matter: a tie carries no information about the true sign,
so ``"zero"`` and ``"random"`` deliver the same expected descent (0.6580 vs 0.6581
at ``N = 10``). ``"random"`` only restores ``||s||_F = sqrt(mn)``, which the
unit-gain multiplier assumes, by adding noise of the matching size.

BatchNorm
---------
``freeze_bn_stats=True`` (the default, and the behaviour of the original code)
runs every BatchNorm layer in inference mode during local gradient
accumulation. Because the local models are discarded each round, this means the
running statistics are *never updated from data*: they stay at their
initialization ``(mean 0, var 1)`` for the whole run, in training and in
evaluation alike. BatchNorm therefore acts as a fixed identity normalization
with learnable affine parameters. This is self-consistent (no train/test
statistics mismatch) and is what the reported federated numbers were produced
with, but it is worth stating explicitly in a reproducibility appendix.
"""

from __future__ import annotations

import copy
import time
from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

from common.lr_scaling import layer_multiplier, describe_rule, resolve_rule
from common.optimizers import muon_lmo, sign_pm1
from common.utils import History, cosine_lr, resolve_device, split_param_names

__all__ = ["MethodSpec", "METHODS", "evaluate_model", "run_federated",
           "resolve_method", "method_family"]


# --------------------------------------------------------------------------
# Method specifications
# --------------------------------------------------------------------------

# Re-exported, not redefined. These live in `federated.methods`, which imports no
# torch, so `federated.export_article` can compute the communication table on a
# machine that cannot run a single round. Kept importable from here because that
# is where every caller already looks for them.
from federated.methods import (  # noqa: E402  (after the torch imports, by design)
    METHOD_ALIASES,
    METHODS,
    MethodSpec,
    communication_bits,
    compresses_downlink,
    method_family,
    resolve_method,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


@contextmanager
def disable_bn_running_stats(model: nn.Module):
    """Temporarily put every BatchNorm layer in inference mode.

    Prevents the running statistics from being polluted by the small local
    batches; see the module docstring for the consequence.
    """
    flags = {}
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            flags[module] = module.training
            module.eval()
    try:
        yield
    finally:
        for module, was_training in flags.items():
            module.train(was_training)


@torch.no_grad()
def evaluate_model(model, eval_sets: Sequence, device=None, verbose: bool = False):
    """Aggregate accuracy/loss over the evaluation sets.

    Returns ``(per_set_accuracies, total_accuracy_pct, mean_loss)``. The loss is a
    true sample mean (``reduction="sum"`` divided by the sample count), so it does
    not depend on the batch size or on unequal client shard sizes.
    """
    device = resolve_device(device) if not isinstance(device, torch.device) else device
    was_training = model.training
    model.eval()
    model.to(device)

    criterion = nn.CrossEntropyLoss(reduction="sum")
    per_set: List[float] = []
    total_correct = total_samples = 0
    total_loss = 0.0

    for set_id, loader in enumerate(eval_sets):
        correct = count = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[-1]
            total_loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            count += labels.size(0)
        acc = 100.0 * correct / count if count else 0.0
        per_set.append(acc)
        if verbose:
            print(f"  set {set_id}: {acc:.2f}%")
        total_correct += correct
        total_samples += count

    model.train(was_training)
    total_acc = 100.0 * total_correct / total_samples if total_samples else 0.0
    mean_loss = total_loss / total_samples if total_samples else 0.0
    return per_set, total_acc, mean_loss


class _ClientData:
    """A ``DataLoader`` plus a *persistent* iterator across rounds.

    Re-creating the iterator every round (as the pre-refactor implementation did)
    restarts the epoch each time, so a run only ever sees the first
    ``n_steps * batch_size`` samples of each freshly shuffled permutation. Keeping
    the iterator alive sweeps the client's whole shard.

    ``federated.data`` now hands the driver objects that already expose
    ``next_batch()``; this wrapper is what makes a plain ``DataLoader`` (the tests,
    and any external caller) work unchanged.
    """

    def __init__(self, loader):
        self.loader = loader
        self._it = None

    def next_batch(self):
        if self._it is None:
            self._it = iter(self.loader)
        try:
            return next(self._it)
        except StopIteration:
            self._it = iter(self.loader)
            return next(self._it)


def _as_shard(obj):
    """Accept either a ``next_batch()`` shard or a plain ``DataLoader``."""
    return obj if hasattr(obj, "next_batch") else _ClientData(obj)


def _accumulate_gradients(
    model: nn.Module,
    data,
    n_steps: int,
    device,
    names: Optional[Iterable[str]] = None,
    freeze_bn_stats: bool = True,
) -> Dict[str, torch.Tensor]:
    """Average the gradient of the local loss over ``n_steps`` mini-batches.

    This is the paper's gradient-accumulation step,
    ``G_t = (1/n_steps) * sum_i grad f_j(X; xi_i)``. Gradients are *pure*: weight
    decay is applied on the server instead (see the module docstring).
    """
    wanted = set(names) if names is not None else None
    criterion = nn.CrossEntropyLoss()
    accumulated: Dict[str, torch.Tensor] = {}

    ctx = disable_bn_running_stats(model) if freeze_bn_stats else _nullcontext()
    with ctx:
        for _ in range(n_steps):
            x, y = data.next_batch()
            x, y = x.to(device), y.to(device).long()
            model.zero_grad(set_to_none=True)
            criterion(model(x), y).backward()
            for name, param in model.named_parameters():
                if param.grad is None or (wanted is not None and name not in wanted):
                    continue
                if name in accumulated:
                    accumulated[name] += param.grad
                else:
                    accumulated[name] = param.grad.clone()

    for name in accumulated:
        accumulated[name] /= n_steps
    return accumulated


@contextmanager
def _nullcontext():
    yield


# --------------------------------------------------------------------------
# The driver
# --------------------------------------------------------------------------


def run_federated(
    method: str,
    global_model: nn.Module,
    train_loaders: Sequence,
    test_loaders: Sequence,
    rounds: int,
    n_steps: int,
    lr: float,
    lr_aux: float = 1e-3,
    momentum: float = 0.9,
    nesterov: bool = False,
    weight_decay: float = 0.0,
    ns_steps: int = 5,
    eval_freq: int = 1,
    device=None,
    buffer_device: Optional[str] = None,
    cosine_schedule: bool = True,
    freeze_bn_stats: bool = True,
    n_head_tensors: int = 2,
    adam_eps: float = 1e-8,
    lmo_dtype: Optional[torch.dtype] = torch.bfloat16,
    # Matches ``federated.main``'s CLI default. It used to be "legacy" here,
    # so a library caller silently got the published-paper convention while
    # the documented default was the derived one.
    lr_scaling: str = "unit-gain",
    scale_baselines: bool = False,
    eval_name: str = "test",
    decoupled_weight_decay: bool = True,
    mv_ties: str = "random",
    uplink_zeros: str = "random",
    verbose: bool = True,
) -> History:
    """Train ``global_model`` with one of the eleven federated methods.

    Returns a :class:`utils.History` recording ``<eval_name>_acc`` /
    ``<eval_name>_loss`` at every evaluated round (round 0 = before training).
    Un-evaluated rounds are simply absent rather than forward-filled, so curves
    from different seeds can be averaged pointwise.

    ``eval_name`` is ``"val"`` while tuning and ``"test"`` for final runs; only one
    set is ever passed, so a tuning run cannot score the test set even by
    accident.

    ``buffer_device`` is where the per-client momentum / EF21 buffers live between
    rounds. It now defaults to the **compute device**: the previous ``"cpu"``
    default round-tripped every client's buffers across PCIe twice per round,
    which for the paper's configuration is ~240k transfers for a saving of a few
    tens of MB of VRAM. Pass ``"cpu"`` explicitly if ``N`` is large enough for that
    trade to flip.
    """
    name, spec = resolve_method(method)
    if mv_ties not in ("zero", "random"):
        raise ValueError(f"mv_ties must be 'zero' or 'random', got {mv_ties!r}")
    if uplink_zeros not in ("keep", "random", "positive"):
        raise ValueError(f"uplink_zeros must be 'keep', 'random' or 'positive', "
                         f"got {uplink_zeros!r}")
    device = resolve_device(device) if not isinstance(device, torch.device) else device
    buffer_device = device if buffer_device is None else torch.device(buffer_device)
    num_clients = len(train_loaders)

    if spec.uplink == "sign_mv" and num_clients % 2 == 0 and num_clients > 1 and verbose:
        print(f"[warn] {num_clients} clients is EVEN. With a strictly +-1 uplink an "
              f"even vote is wasted -- {num_clients} would deliver the same alignment "
              f"as {num_clients - 1} while tying on ~15% of coordinates. Prefer "
              f"--n_parties {num_clients + 1}.", flush=True)

    global_model.to(device)
    global_model.train()

    matrix_names, aux_names = split_param_names(global_model, n_head_tensors)
    params = dict(global_model.named_parameters())

    # -- per-layer learning-rate multipliers -------------------------------
    rule = resolve_rule(lr_scaling)
    family = method_family(name, scale_baselines)
    if family is None:
        lam = {n: 1.0 for n in matrix_names}
    else:
        lam = {n: layer_multiplier(rule, family, tuple(params[n].shape))
               for n in matrix_names}

    if verbose:
        print(f"Federated {name} on {device}: {num_clients} clients, {rounds} rounds, "
              f"{len(matrix_names)} matrix params (LMO={spec.lmo}, "
              f"up={spec.uplink}, down={spec.downlink}), {len(aux_names)} aux params",
              flush=True)
        if family is not None:
            print(describe_rule(rule, family,
                                [(n, tuple(params[n].shape)) for n in matrix_names]),
                  flush=True)
        else:
            print(f"LR scaling: none for '{name}' (its step norm is data-dependent, "
                  f"so no static multiplier implements the unit-gain criterion); "
                  f"pass scale_baselines=True to override", flush=True)
        n_mat = sum(params[n].numel() for n in matrix_names)
        n_aux = sum(params[n].numel() for n in aux_names)
        if n_mat + n_aux:
            print(f"Parameters: {n_mat:,} matrix (compressed) + {n_aux:,} auxiliary "
                  f"({100.0 * n_aux / (n_mat + n_aux):.3f}% uncompressed)", flush=True)

    # -- server state ------------------------------------------------------
    # AdamW on the auxiliary parameters, uniformly for every method, and never
    # decayed -- matching centralized.train.build_optimizers, so `weight_decay`
    # describes the matrix parameters alone in both settings.
    adamw = torch.optim.AdamW(
        [params[n] for n in aux_names], lr=lr_aux, weight_decay=0.0, eps=adam_eps
    ) if aux_names else None

    # Server-side Adam over the matrix parameters (the ``adam`` baseline only).
    # AdamW when the decay convention is decoupled, so the baseline is not
    # additionally handicapped by an Adam-vs-AdamW difference.
    # One group per tensor, so the per-layer multiplier ``lam[n]`` reaches the
    # baseline too: the step below ``continue``s past the ``lam`` application for
    # ``adam``, and torch owns the step. Without this, ``--scale-baselines``
    # silently did nothing for ``adam`` while the schedule and the report both
    # claimed it applied. Matches ``centralized/train.py`` ("lr": args.lr * lam).
    server_adam_cls = torch.optim.AdamW if decoupled_weight_decay else torch.optim.Adam
    server_adam = server_adam_cls(
        [{"params": [params[n]], "lr": lr * lam[n], "initial_lr": lr * lam[n]}
         for n in matrix_names],
        lr=lr, weight_decay=weight_decay, eps=adam_eps
    ) if spec.server == "adam" and matrix_names else None

    # EF21 reconstruction G_t on the server (uplink == "ef21").
    server_estimator = {n: torch.zeros_like(params[n], device=device)
                        for n in matrix_names} if spec.uplink == "ef21" else {}

    # Exact model X, distinct from the broadcast W held in ``params`` (ef21p).
    server_exact = {n: params[n].detach().clone()
                    for n in matrix_names} if spec.needs_exact_model else {}

    # -- client state ------------------------------------------------------
    def _zeros():
        return {n: torch.zeros_like(params[n], device=buffer_device) for n in matrix_names}

    client_momentum = [_zeros() for _ in range(num_clients)] if spec.client_momentum else None
    client_estimator = [_zeros() for _ in range(num_clients)] if spec.uplink == "ef21" else None
    client_data = [_as_shard(dl) for dl in train_loaders]
    # One reusable local model: cheaper than deep-copying the global model per
    # client per round, and semantically identical because the broadcast weights
    # are reloaded before every client's local work.
    local_model = copy.deepcopy(global_model).to(device)

    history = History()
    acc_key, loss_key = f"{eval_name}_acc", f"{eval_name}_loss"

    @contextmanager
    def _exact_view():
        """Expose the exact model X in ``global_model`` for evaluation."""
        if not server_exact:
            yield
            return
        backup = {n: params[n].detach().clone() for n in server_exact}
        with torch.no_grad():
            for n, X in server_exact.items():
                params[n].data.copy_(X)
        try:
            yield
        finally:
            with torch.no_grad():
                for n, w in backup.items():
                    params[n].data.copy_(w)

    def _evaluate(step: int, elapsed: Optional[float] = None,
                  extra: Optional[dict] = None) -> None:
        with _exact_view():
            _, acc, loss = evaluate_model(global_model, test_loaders, device=device)
        history.record(step, **{acc_key: acc, loss_key: loss}, **(extra or {}))
        if verbose:
            timing = f" | {elapsed:.2f}s" if elapsed is not None else ""
            print(f"Round {step}{timing} | {eval_name} acc: {acc:.2f}%, "
                  f"loss: {loss:.4f}", flush=True)

    _evaluate(0)

    for r in range(1, rounds + 1):
        t0 = time.perf_counter()
        eta = cosine_lr(1.0, r - 1, rounds) if cosine_schedule else 1.0
        current_lr, current_lr_aux = lr * eta, lr_aux * eta
        if adamw is not None:
            for g in adamw.param_groups:
                g["lr"] = current_lr_aux
        if server_adam is not None:
            # Anneal each group from its OWN base rate, so the per-layer factor
            # folded in at construction survives the cosine schedule.
            for g in server_adam.param_groups:
                g["lr"] = g["initial_lr"] * eta

        # ---------------- clients ----------------------------------------
        # The broadcast model: W (compressed) under ef21p, X otherwise. Both are
        # what ``params`` currently holds, so a plain state_dict copy is right.
        # Clients only *accumulate gradients* (no local parameter steps), so with
        # frozen BatchNorm statistics the local model is identical for every
        # client and one copy per round suffices.
        local_model.load_state_dict(global_model.state_dict())
        local_model.train()

        uplink_payloads: List[Dict[str, torch.Tensor]] = []
        uplink_scales: List[Dict[str, torch.Tensor]] = []
        aux_grads_sum: Dict[str, torch.Tensor] = {}
        # How much of the uplink is the third symbol. The "1 bit per parameter"
        # claim is about this number, so it is measured rather than asserted.
        sent_zero = sent_total = 0

        for j in range(num_clients):
            if not freeze_bn_stats and j > 0:
                # Live BatchNorm statistics *would* drift from one client to the
                # next within a round, so re-broadcast before each client.
                local_model.load_state_dict(global_model.state_dict())
            grads = _accumulate_gradients(
                local_model, client_data[j], n_steps, device,
                names=None, freeze_bn_stats=freeze_bn_stats,
            )

            for n in aux_names:
                if n in grads:
                    aux_grads_sum[n] = aux_grads_sum.get(n, 0) + grads[n]

            payload, scales = {}, {}
            for n in matrix_names:
                G = grads[n]

                # 0) coupled weight decay: fold wd*X into the gradient *before*
                #    the oracle sees it, so the geometry is perturbed rather than
                #    the step length. Matches ``common.optimizers._BaseMethod``.
                #
                #    Two ways in. As the appendix ablation, via
                #    ``--weight-decay-mode coupled``. And ALWAYS for ``sgd``:
                #    torch's SGD couples, and that is the right convention there
                #    rather than a quirk -- SGD's step is ``eta*m``, which is NOT
                #    positively homogeneous of degree zero, so coupling genuinely
                #    minimizes ``f + (wd/2)||W||^2``. Forcing decoupling on it
                #    would be the deviation, and it would silently disagree with
                #    ``centralized/train.py``, which spells this out at its own
                #    ``name == "sgd"`` branch.
                if weight_decay != 0 and (not decoupled_weight_decay
                                          or spec.momentum_form == "heavy_ball"):
                    ref = server_exact[n] if spec.needs_exact_model else params[n].data
                    G = G.add(ref, alpha=weight_decay)

                # 1) client momentum
                if spec.client_momentum:
                    buf = client_momentum[j][n].to(device)
                    if spec.momentum_form == "ema":
                        buf.mul_(momentum).add_(G, alpha=1.0 - momentum)
                        m_tilde = G.mul(1.0 - momentum).add_(buf, alpha=momentum) \
                            if nesterov else buf
                    else:
                        buf.mul_(momentum).add_(G)
                        m_tilde = G.add(buf, alpha=momentum) if nesterov else buf
                else:
                    m_tilde = G

                # 2) worker-side LMO, when the sign acts after the oracle.
                #    scale_aspect=False: the shape factor lives in ``lam``.
                target = muon_lmo(m_tilde, ns_steps=ns_steps, dtype=lmo_dtype,
                                  scale_aspect=False) \
                    if spec.lmo == "worker" else m_tilde

                # 3) uplink compression
                if spec.uplink == "ef21":
                    est = client_estimator[j][n].to(device)
                    delta = target - est
                    alpha = delta.abs().mean()
                    raw_sign = torch.sign(delta)
                    # Zero-fraction diagnostic is measured on the raw sign, BEFORE
                    # the paper's randomized-zero convention maps it to +-1.
                    sent_zero += int((raw_sign == 0).sum())
                    sign_delta = sign_pm1(delta)
                    est.add_(alpha * sign_delta)
                    sent_total += sign_delta.numel()
                    payload[n] = sign_delta.to(buffer_device)
                    scales[n] = alpha.to(buffer_device)
                    client_estimator[j][n] = est.to(buffer_device)
                elif spec.uplink == "sign_mv":
                    s = torch.sign(target)
                    zero_mask = s == 0
                    n_zero = int(zero_mask.sum())
                    sent_zero += n_zero
                    sent_total += s.numel()
                    if n_zero and uplink_zeros != "keep":
                        fill = torch.ones_like(s) if uplink_zeros == "positive" else \
                            torch.randint(0, 2, s.shape, device=device,
                                          dtype=s.dtype).mul_(2).sub_(1)
                        s = torch.where(zero_mask, fill, s)
                    payload[n] = s.to(buffer_device)
                else:
                    payload[n] = target.detach().clone().to(buffer_device)

                if spec.client_momentum:
                    # Persist the momentum *buffer*, not the (possibly Nesterov)
                    # look-ahead direction derived from it.
                    client_momentum[j][n] = buf.to(buffer_device)

            uplink_payloads.append(payload)
            uplink_scales.append(scales)

        # ---------------- server -----------------------------------------
        ties = tied = 0
        # Realized per-layer gain, gamma = ||lambda * s||_F / sqrt(fan_out). The
        # unit-gain rule exists to make this the SAME on every layer, and that is
        # checkable rather than assumable: the derivation assumes the oracle
        # returns ||polar||_F = sqrt(min(m,n)) exactly, while 5 Newton-Schulz steps
        # return 0.77-0.87 of it on CNN2's shapes, shape-dependently. A tuned eta_0
        # absorbs a constant factor; it cannot absorb a per-layer one.
        realized_gain: Dict[str, float] = {}
        with torch.no_grad():
            for n in matrix_names:
                # 1) aggregate the uplink
                if spec.uplink == "ef21":
                    agg = sum(uplink_payloads[j][n].to(device) * uplink_scales[j][n].to(device)
                              for j in range(num_clients)) / num_clients
                    server_estimator[n].add_(agg)
                    aggregate = server_estimator[n]
                elif spec.uplink == "sign_mv":
                    aggregate = torch.sign(
                        sum(uplink_payloads[j][n].to(device) for j in range(num_clients)))
                    # A zero entry is an even split. Counted BEFORE any tie-break,
                    # so mv_tie_frac measures the vote rather than the rule.
                    tie_mask = aggregate == 0
                    n_tied = int(tie_mask.sum())
                    tied += n_tied
                    ties += aggregate.numel()
                    if n_tied and mv_ties == "random":
                        # Draws from the global generator, which seed_everything
                        # has already seeded, so the run stays reproducible.
                        coin = torch.randint(0, 2, aggregate.shape, device=device,
                                             dtype=aggregate.dtype).mul_(2).sub_(1)
                        aggregate = torch.where(tie_mask, coin, aggregate)
                else:
                    aggregate = sum(uplink_payloads[j][n].to(device)
                                    for j in range(num_clients)) / num_clients

                # 2) server-side LMO, when the sign acted before the oracle
                D = muon_lmo(aggregate, ns_steps=ns_steps, dtype=lmo_dtype,
                             scale_aspect=False) \
                    if spec.lmo == "server" else aggregate

                # 3) the step, on X (ef21p) or directly on the broadcast model
                if spec.server == "adam":
                    params[n].grad = D.clone()
                    continue

                target_tensor = server_exact[n] if spec.needs_exact_model else params[n].data
                if (weight_decay != 0 and decoupled_weight_decay
                        and spec.momentum_form != "heavy_ball"):
                    # Unscaled by the per-layer factor, matching the centralized
                    # convention: decay is a property of the parameter, not of the
                    # step geometry.
                    target_tensor.mul_(1.0 - current_lr * weight_decay)

                step = sign_pm1(D) if spec.downlink == "sign" else D
                realized_gain[n] = float(lam[n] * step.norm()
                                         / (params[n].shape[0] ** 0.5))
                target_tensor.add_(step, alpha=-current_lr * lam[n])

                # 4) downlink error feedback: broadcast a scaled sign of X - W
                if spec.downlink == "ef21p":
                    shift = server_exact[n] - params[n].data
                    params[n].data.add_(shift.abs().mean() * sign_pm1(shift))

        if server_adam is not None:
            server_adam.step()
            server_adam.zero_grad(set_to_none=True)

        # ---------------- auxiliary parameters (AdamW, uncompressed) -------
        if adamw is not None:
            adamw.zero_grad(set_to_none=True)
            for n in aux_names:
                if n in aux_grads_sum:
                    params[n].grad = (aux_grads_sum[n] / num_clients).clone()
            adamw.step()

        if r == 1 and verbose and realized_gain:
            lo, hi = min(realized_gain.values()), max(realized_gain.values())
            print(f"Realized per-layer gain ||lambda*s||_F/sqrt(fan_out) at round 1 "
                  f"(the rule's own target is a flat profile):", flush=True)
            for n, g in realized_gain.items():
                print(f"  {n:<28}{g:>12.5g}", flush=True)
            # "gain spread", not "spread": describe_rule() above prints its own
            # "spread ...x" line for the MULTIPLIERS, and a log reader (or a regex)
            # that confuses the two reads a flat profile where there is not one.
            print(f"  gain spread {hi / lo:.2f}x   (min {lo:.4g}, max {hi:.4g})"
                  + ("   <- the per-layer rule is NOT equalizing the realized "
                     "step; the oracle's norm error is shape-dependent"
                     if hi / lo > 1.15 else ""), flush=True)

        elapsed = time.perf_counter() - t0
        if r % eval_freq == 0 or r == rounds:
            extra = {"round_seconds": elapsed}
            if ties:
                extra["mv_tie_frac"] = tied / ties
            if sent_total:
                extra["uplink_zero_frac"] = sent_zero / sent_total
            if realized_gain:
                vals = list(realized_gain.values())
                extra["gain_spread"] = max(vals) / min(vals) if min(vals) > 0 else None
            _evaluate(r, elapsed, extra)

    # Leave the caller holding the exact model: it is the iterate the theory
    # bounds, and the one whose accuracy is reported.
    if server_exact:
        with torch.no_grad():
            for n, X in server_exact.items():
                params[n].data.copy_(X)

    return history
