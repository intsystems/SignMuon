"""Federated CIFAR-10 / MNIST shards, with a held-out validation split.

Two things this module exists to fix.

**1. A validation split, so tuning never reads the test set.**
The centralized path holds out 5k of the 50k training images and selects every
hyperparameter on validation accuracy (``centralized/data.py``). The federated
path had no such split: ``federated/grid.py`` ranked learning rates by the
accuracy printed each round, which is *test* accuracy. With eleven methods and
~7 rates each that is ~70 peeks at the test set, and the resulting comparison is
contaminated. The protocol is now identical to the centralized one:

1. **Tune** on ``split="tune"``. The 5k validation images are held out *first*,
   the remaining 45k are partitioned across the clients, and the validation set
   stays whole and server-side -- it is the server's model that is being
   selected, so a per-client split of it would only add noise.
2. **Retrain** on ``split="full"``: all 50k partitioned across the clients, and
   the test set is touched once per (method, seed).

``val_seed`` is deliberately independent of the run seed and defaults to the same
``DEFAULT_VAL_SEED`` the centralized code uses, so the two settings hold out the
**same 5000 images** and their validation numbers are directly comparable.

**2. Speed.** The old path built one ``DataLoader`` per client over a
``Subset`` of a torchvision dataset with ``num_workers=0``, so every one of the
2000 rounds x 10 clients x 3 steps x 64 images went through PIL decode +
augment on the main thread -- and 10 clients x N workers is not a fix, it is 20
worker processes fighting over the same GPU feed. With ``loader="gpu"``
(the default on CUDA) the whole dataset is uploaded once as ``uint8`` (153 MB
for CIFAR-10 train) and cropping, flipping and normalization are tensor ops on
the device. The augmentation is the same one torchvision applies -- zero-padded
random 32x32 crop with ``padding=4``, then a random horizontal flip -- and
``tests/test_code.py`` pins that equivalence.

Partitioning
------------
``partition_seed`` (default: the run seed) drives the client split through an
explicit ``numpy`` generator rather than the global RNG, so the partition is
reproducible on its own and does not shift when unrelated code draws a random
number. Pin it to compare methods on an identical partition across seeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from centralized.data import DEFAULT_VAL_SEED, VAL_SIZE

__all__ = [
    "DEFAULT_VAL_SEED",
    "VAL_SIZE",
    "NORMALIZATION",
    "build_federated_data",
    "partition_indices",
    "GpuShard",
    "GpuEvalSet",
    # backwards-compatible helpers
    "get_cifar10_transforms",
    "get_mnist_transform",
]

#: ``(mean, std, pad)`` per dataset. ``pad`` is the random-crop padding, 0 when the
#: dataset is not augmented (MNIST, matching the original code).
NORMALIZATION = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), 4),
    "mnist": ((0.5,), (0.5,), 0),
}


# --------------------------------------------------------------------------
# Raw tensors
# --------------------------------------------------------------------------


def load_raw(dataset: str, datadir: str, download: bool = False):
    """``(train_x, train_y, test_x, test_y)`` as ``uint8`` NCHW / int64 tensors.

    Bypasses the torchvision transform pipeline entirely: the images are needed
    as one contiguous block so they can be uploaded to the device once.
    """
    if dataset == "cifar10":
        tr = datasets.CIFAR10(datadir, train=True, download=download)
        te = datasets.CIFAR10(datadir, train=False, download=download)
        # .data is [N, 32, 32, 3] uint8 (numpy)
        tx = torch.from_numpy(tr.data).permute(0, 3, 1, 2).contiguous()
        ex = torch.from_numpy(te.data).permute(0, 3, 1, 2).contiguous()
        ty = torch.as_tensor(tr.targets, dtype=torch.long)
        ey = torch.as_tensor(te.targets, dtype=torch.long)
    elif dataset == "mnist":
        tr = datasets.MNIST(datadir, train=True, download=download)
        te = datasets.MNIST(datadir, train=False, download=download)
        tx = tr.data.unsqueeze(1).contiguous()          # [N, 1, 28, 28] uint8
        ex = te.data.unsqueeze(1).contiguous()
        ty = tr.targets.clone().long()
        ey = te.targets.clone().long()
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return tx, ty, ex, ey


# --------------------------------------------------------------------------
# Partitioning
# --------------------------------------------------------------------------


def partition_indices(
    y_train: np.ndarray,
    y_test: np.ndarray,
    partition: str,
    n_parties: int,
    beta: float = 0.5,
    rng: Optional[np.random.Generator] = None,
    train_pool: Optional[np.ndarray] = None,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Client index maps for the train and test sets.

    ``train_pool`` restricts the partition to a subset of the training indices --
    this is how the validation images are held out *before* the data is split, so
    no client ever sees them.

    ``rng`` is an explicit generator; the previous implementation drew from the
    global ``numpy`` RNG, which tied the partition to whatever else had already
    consumed randomness.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    pool = np.arange(len(y_train)) if train_pool is None else np.asarray(train_pool)
    test_pool = np.arange(len(y_test))

    if partition == "homo":
        tr = rng.permutation(pool)
        te = rng.permutation(test_pool)
        return ({i: s for i, s in enumerate(np.array_split(tr, n_parties))},
                {i: s for i, s in enumerate(np.array_split(te, n_parties))})

    if partition != "noniid-labeldir":
        raise ValueError(f"Unknown partition {partition!r}")

    # Dirichlet label skew. Each class is split across the clients in proportions
    # drawn from Dir(beta), with a cap that keeps any one client from taking more
    # than its equal share -- the standard construction (Li et al., 2022).
    n_classes = int(max(y_train.max(), y_test.max())) + 1
    cap = len(pool) / n_parties
    min_size, min_require = 0, 10
    idx_train: List[List[int]] = []
    idx_test: List[List[int]] = []

    while min_size < min_require:
        idx_train = [[] for _ in range(n_parties)]
        idx_test = [[] for _ in range(n_parties)]
        for k in range(n_classes):
            k_train = pool[y_train[pool] == k]
            k_test = test_pool[y_test[test_pool] == k]
            rng.shuffle(k_train)
            rng.shuffle(k_test)

            p = rng.dirichlet(np.repeat(beta, n_parties))
            room = np.array([len(s) < cap for s in idx_train], dtype=float)
            p = p * room
            if p.sum() == 0:
                p = room / room.sum() if room.sum() else np.full(n_parties, 1.0 / n_parties)
            else:
                p = p / p.sum()

            cuts_tr = (np.cumsum(p) * len(k_train)).astype(int)[:-1]
            cuts_te = (np.cumsum(p) * len(k_test)).astype(int)[:-1]
            for j, (a, b) in enumerate(zip(np.split(k_train, cuts_tr),
                                           np.split(k_test, cuts_te))):
                idx_train[j].extend(a.tolist())
                idx_test[j].extend(b.tolist())
        min_size = min(len(s) for s in idx_train)

    return ({j: np.array(idx_train[j], dtype=np.int64) for j in range(n_parties)},
            {j: np.array(idx_test[j], dtype=np.int64) for j in range(n_parties)})


# --------------------------------------------------------------------------
# Device-resident shards
# --------------------------------------------------------------------------


def _norm_stats(dataset: str, device, channels: int):
    mean, std, pad = NORMALIZATION[dataset]
    m = torch.tensor(mean, device=device).view(1, channels, 1, 1)
    s = torch.tensor(std, device=device).view(1, channels, 1, 1)
    return m, s, pad


class GpuShard:
    """One client's training shard, resident on the device.

    ``next_batch()`` walks a reshuffled permutation of the shard and never
    restarts the epoch mid-round, so a client's whole shard is swept -- the same
    property the persistent-iterator fix gave the ``DataLoader`` path.

    Augmentation reproduces ``RandomCrop(32, padding=4)`` followed by
    ``RandomHorizontalFlip()``: zero-pad, take a per-sample random 32x32 window,
    flip each sample independently with probability 1/2. Randomness comes from a
    CPU generator seeded per client, so the batch sequence is identical whether
    the tensors live on CPU or GPU.
    """

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, batch_size: int,
                 dataset: str, seed: int, augment: bool = True):
        self.x = images
        self.y = labels
        self.bs = int(batch_size)
        self.augment = bool(augment) and NORMALIZATION[dataset][2] > 0
        self.mean, self.std, self.pad = _norm_stats(dataset, images.device, images.shape[1])
        self.g = torch.Generator().manual_seed(int(seed))
        self._order = None
        self._pos = 0
        if len(self) == 0:
            raise ValueError("GpuShard was given an empty shard")

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def _reshuffle(self) -> None:
        self._order = torch.randperm(len(self), generator=self.g).to(self.x.device)
        self._pos = 0

    def next_batch(self):
        if self._order is None or self._pos >= len(self):
            self._reshuffle()
        idx = self._order[self._pos:self._pos + self.bs]
        self._pos += self.bs
        xb = self.x.index_select(0, idx).float().div_(255.0)
        if self.augment:
            xb = self._crop_and_flip(xb)
        return xb.sub_(self.mean).div_(self.std), self.y.index_select(0, idx)

    def _crop_and_flip(self, xb: torch.Tensor) -> torch.Tensor:
        b, c, h, w = xb.shape
        p = self.pad
        xb = F.pad(xb, (p, p, p, p))
        oy = torch.randint(0, 2 * p + 1, (b,), generator=self.g).to(xb.device)
        ox = torch.randint(0, 2 * p + 1, (b,), generator=self.g).to(xb.device)
        rows = oy[:, None] + torch.arange(h, device=xb.device)[None, :]
        cols = ox[:, None] + torch.arange(w, device=xb.device)[None, :]
        xb = xb.gather(2, rows[:, None, :, None].expand(b, c, h, xb.shape[3]))
        xb = xb.gather(3, cols[:, None, None, :].expand(b, c, h, w))
        flip = (torch.rand(b, generator=self.g) < 0.5).to(xb.device)
        return torch.where(flip[:, None, None, None], xb.flip(-1), xb)


class GpuEvalSet:
    """A device-resident evaluation set, iterated in fixed-size chunks.

    No augmentation and no shuffling, so iterating it twice gives the same
    batches -- which is what makes the reported accuracy a deterministic function
    of the model.
    """

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, batch_size: int,
                 dataset: str):
        self.x, self.y, self.bs = images, labels, int(batch_size)
        self.mean, self.std, _ = _norm_stats(dataset, images.device, images.shape[1])

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __iter__(self):
        for i in range(0, len(self), self.bs):
            xb = self.x[i:i + self.bs].float().div_(255.0)
            yield xb.sub_(self.mean).div_(self.std), self.y[i:i + self.bs]


class LoaderShard:
    """``DataLoader``-backed client shard, for the CPU / torchvision path.

    Keeps the persistent iterator that stops a client re-reading the first
    ``n_steps`` batches of a fresh epoch every round.
    """

    def __init__(self, loader: DataLoader, device=None):
        self.loader = loader
        self.device = device
        self._it = None

    def __len__(self) -> int:
        return len(self.loader.dataset)

    def next_batch(self):
        if self._it is None:
            self._it = iter(self.loader)
        try:
            x, y = next(self._it)
        except StopIteration:
            self._it = iter(self.loader)
            x, y = next(self._it)
        if self.device is not None:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
        return x, y


# --------------------------------------------------------------------------
# The builder
# --------------------------------------------------------------------------


@dataclass
class FederatedData:
    """Everything a run needs, plus what the log should say about it."""

    train_shards: List[object]
    eval_sets: List[object]
    eval_name: str                      # "val" or "test"
    n_train: int
    shard_sizes: List[int] = field(default_factory=list)
    loader: str = "gpu"


def build_federated_data(
    dataset: str,
    datadir: str,
    *,
    n_parties: int,
    batch_size: int,
    partition: str = "homo",
    beta: float = 0.5,
    split: str = "full",
    seed: int = 0,
    partition_seed: Optional[int] = None,
    val_seed: int = DEFAULT_VAL_SEED,
    val_size: int = VAL_SIZE,
    device=None,
    loader: str = "auto",
    num_workers: int = 0,
    eval_batch_size: int = 1000,
    download: bool = False,
) -> FederatedData:
    """Build the client shards and the evaluation set for one run.

    ``split="tune"`` holds out ``val_size`` training images (the same ones the
    centralized code holds out, given the same ``val_seed``), partitions the rest
    across the clients, and returns the validation set as the evaluation target.
    ``split="full"`` partitions everything and evaluates on the test set.

    Only one evaluation set is returned, deliberately: under ``split="tune"`` the
    test set is never returned and never scored, so it cannot leak into a decision
    even by accident -- and evaluation is the dominant cost of a 2000-round run.
    (``load_raw`` still reads both splits off disk, since torchvision hands them
    over together; "not loaded" would be the wrong word. What is guaranteed is
    that no test image is ever evaluated or ranked on.)
    """
    if split not in ("full", "tune"):
        raise ValueError(f"split must be 'full' or 'tune', got {split!r}")
    device = torch.device("cpu") if device is None else torch.device(device)
    if loader == "auto":
        loader = "gpu" if device.type == "cuda" else "torch"

    if n_parties < 1:
        raise ValueError(f"n_parties must be >= 1, got {n_parties}")

    tx, ty, ex, ey = load_raw(dataset, datadir, download=download)
    y_train, y_test = ty.numpy(), ey.numpy()

    # The validation images are removed from the pool BEFORE partitioning, so no
    # client holds one. The permutation is the same one centralized/data.py uses.
    perm = np.random.default_rng(val_seed).permutation(len(y_train))
    if split == "tune":
        if val_size >= len(y_train):
            raise ValueError(
                f"val_size={val_size} leaves no training data (the set has "
                f"{len(y_train)} images). Lower val_size or use --split full.")
        train_pool, val_idx = perm[val_size:], perm[:val_size]
    else:
        train_pool, val_idx = perm, np.empty(0, dtype=np.int64)

    if len(train_pool) < n_parties * batch_size:
        # Not fatal in principle, but a shard smaller than a batch means a client
        # sees the same images several times per round, which is not the
        # experiment anyone intends.
        print(f"[warn] {len(train_pool)} training images over {n_parties} clients "
              f"at batch {batch_size}: some shards are smaller than one batch")

    rng = np.random.default_rng(seed if partition_seed is None else partition_seed)
    train_map, test_map = partition_indices(
        y_train, y_test, partition, n_parties, beta=beta, rng=rng, train_pool=train_pool)

    empty = [j for j in range(n_parties) if len(train_map[j]) == 0]
    if empty:
        raise ValueError(
            f"clients {empty} received an empty training shard from a pool of "
            f"{len(train_pool)} images. An empty shard surfaces much later as a "
            f"zero-sized batch inside the model, so it is rejected here.")

    if split == "tune":
        eval_x, eval_y, eval_name = tx[val_idx], ty[val_idx], "val"
        eval_parts = [(eval_x, eval_y)]                  # one server-side set
    else:
        eval_name = "test"
        eval_parts = [(ex[test_map[j]], ey[test_map[j]]) for j in range(n_parties)]

    if loader == "gpu":
        shards = [GpuShard(tx[train_map[j]].to(device), ty[train_map[j]].to(device),
                           batch_size, dataset, seed=int(seed) + j)
                  for j in range(n_parties)]
        evals = [GpuEvalSet(x.to(device), y.to(device), eval_batch_size, dataset)
                 for x, y in eval_parts]
    else:
        train_tf, eval_tf = _transforms(dataset)
        base_train = _torchvision_dataset(dataset, datadir, True, train_tf)
        base_eval_tr = _torchvision_dataset(dataset, datadir, True, eval_tf)
        base_test = _torchvision_dataset(dataset, datadir, False, eval_tf)
        shards = []
        for j in range(n_parties):
            g = torch.Generator()
            g.manual_seed(int(seed) + j)
            dl = DataLoader(Subset(base_train, train_map[j]), batch_size=batch_size,
                            shuffle=True, generator=g, num_workers=num_workers,
                            persistent_workers=num_workers > 0,
                            pin_memory=device.type == "cuda")
            shards.append(LoaderShard(dl, device))
        if split == "tune":
            evals = [_eval_loader(Subset(base_eval_tr, val_idx), eval_batch_size, device)]
        else:
            evals = [_eval_loader(Subset(base_test, test_map[j]), eval_batch_size, device)
                     for j in range(n_parties)]

    return FederatedData(
        train_shards=shards, eval_sets=evals, eval_name=eval_name,
        n_train=int(sum(len(train_map[j]) for j in range(n_parties))),
        shard_sizes=[int(len(train_map[j])) for j in range(n_parties)],
        loader=loader,
    )


class _DeviceLoader:
    """Wraps a ``DataLoader`` so evaluation batches arrive on the device."""

    def __init__(self, dl: DataLoader, device):
        self.dl, self.device = dl, device

    def __len__(self) -> int:
        return len(self.dl.dataset)

    def __iter__(self):
        for x, y in self.dl:
            yield x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


def _eval_loader(subset, batch_size: int, device) -> _DeviceLoader:
    return _DeviceLoader(
        DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0,
                   pin_memory=device.type == "cuda"), device)


def _torchvision_dataset(dataset: str, datadir: str, train: bool, transform):
    ctor = datasets.CIFAR10 if dataset == "cifar10" else datasets.MNIST
    return ctor(datadir, train=train, download=False, transform=transform)


def _transforms(dataset: str):
    if dataset == "cifar10":
        return get_cifar10_transforms(True), get_cifar10_transforms(False)
    tf = get_mnist_transform()
    return tf, tf


# --------------------------------------------------------------------------
# Backwards-compatible helpers (the pre-rewrite API)
# --------------------------------------------------------------------------


def get_mnist_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


def get_cifar10_transforms(train: bool = True):
    norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([transforms.ToTensor(), norm])
