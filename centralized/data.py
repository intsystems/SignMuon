"""Centralized CIFAR-10 / MNIST loaders, with a held-out validation split.

Why the validation split exists
-------------------------------
Hyperparameters must be selected on data that is not the test set. With ten
methods and ~11 learning rates each, choosing by test accuracy would mean ~110
peeks at the test set, and any resulting comparison is contaminated. The protocol
is therefore:

1. **Tune** on ``split="tune"``: a fixed 45k/5k train/val partition of the 50k
   CIFAR-10 training set. Every decision (learning rate, ``lr_aux``, the scaling
   exponent) comes from validation accuracy.
2. **Retrain** on ``split="full"``: the whole 50k training set at the chosen
   hyperparameters, and touch the test set once per (method, seed).

The partition is controlled by ``val_seed``, which is deliberately **independent
of the run seed**, so that every method and every seed tunes against the same
split -- otherwise the split itself becomes a confounder.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from common.utils import seed_worker

__all__ = ["mnist_loaders", "cifar10_loaders", "build_loaders", "VAL_SIZE",
           "DEFAULT_VAL_SEED"]

#: Held-out validation examples, taken from the training set.
VAL_SIZE = 5_000

#: Seed for the train/val partition. Fixed and independent of the run seed so that
#: every method and every seed tunes against the identical split.
DEFAULT_VAL_SEED = 12_345


def _generator(seed: Optional[int]) -> Optional[torch.Generator]:
    """Seeded generator for DataLoader shuffling (``None`` keeps global RNG)."""
    if seed is None:
        return None
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def _split_indices(n: int, val_size: int, val_seed: int):
    """Deterministic train/val index partition, independent of the global RNG."""
    perm = np.random.default_rng(val_seed).permutation(n)
    return perm[val_size:], perm[:val_size]


def _cifar10_transforms(train: bool):
    norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([transforms.ToTensor(), norm])


def _mnist_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


def build_loaders(
    dataset: str,
    datadir: str,
    batch_size: int = 128,
    download: bool = False,
    seed: Optional[int] = None,
    split: str = "full",
    val_size: int = VAL_SIZE,
    val_seed: int = DEFAULT_VAL_SEED,
    num_workers: int = 4,
    eval_batch_size: Optional[int] = None,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """Return ``(train_loader, val_loader, test_loader)``.

    ``split="tune"`` holds out ``val_size`` training examples as validation and
    trains on the rest; ``split="full"`` trains on everything and returns
    ``val_loader = None``.

    The validation subset uses the *test-time* transform (no augmentation), which
    is what makes validation accuracy a usable proxy for test accuracy.
    """
    if split not in ("full", "tune"):
        raise ValueError(f"split must be 'full' or 'tune', got {split!r}")

    if dataset == "cifar10":
        train_tf, eval_tf = _cifar10_transforms(True), _cifar10_transforms(False)
        ctor = datasets.CIFAR10
    elif dataset == "mnist":
        train_tf = eval_tf = _mnist_transform()
        ctor = datasets.MNIST
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    train_ds = ctor(datadir, train=True, download=download, transform=train_tf)
    test_ds = ctor(datadir, train=False, download=download, transform=eval_tf)

    val_loader = None
    if split == "tune":
        # A second view of the training set carrying the *eval* transform, so the
        # validation subset is not augmented.
        val_source = ctor(datadir, train=True, download=False, transform=eval_tf)
        train_idx, val_idx = _split_indices(len(train_ds), val_size, val_seed)
        # num_workers=0 for evaluation: a worker pool is respawned on every
        # iterator creation (once per epoch), which on Windows costs more than
        # decoding 5-10k images single-threaded.
        val_loader = DataLoader(
            Subset(val_source, val_idx), batch_size=eval_batch_size or batch_size,
            shuffle=False, num_workers=0, pin_memory=True)
        train_view = Subset(train_ds, train_idx)
    else:
        train_view = train_ds

    # ``worker_init_fn`` matters even though the current transforms all draw from
    # torch (which the loader's ``generator`` already seeds per worker): any
    # transform or dataset reaching for ``numpy.random`` or ``random`` inside a
    # worker would silently become irreproducible, and ``common.utils`` already
    # promises otherwise. Cheap insurance, and it makes the promise true.
    train_loader = DataLoader(
        train_view, batch_size=batch_size, shuffle=True, drop_last=False,
        generator=_generator(seed), num_workers=num_workers, pin_memory=True,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        persistent_workers=num_workers > 0)
    test_loader = DataLoader(
        test_ds, batch_size=eval_batch_size or batch_size, shuffle=False,
        num_workers=0, pin_memory=True)
    return train_loader, val_loader, test_loader


# --------------------------------------------------------------------------
# Backwards-compatible two-loader helpers
# --------------------------------------------------------------------------


def cifar10_loaders(datadir: str, batch_size: int = 128, download: bool = False,
                    seed: Optional[int] = None, num_workers: int = 4):
    """``(train, test)`` over the full 50k training set."""
    train, _, test = build_loaders("cifar10", datadir, batch_size, download, seed,
                                   split="full", num_workers=num_workers)
    return train, test


def mnist_loaders(datadir: str, batch_size: int = 128, download: bool = False,
                  seed: Optional[int] = None, num_workers: int = 4):
    """``(train, test)`` over the full training set."""
    train, _, test = build_loaders("mnist", datadir, batch_size, download, seed,
                                   split="full", num_workers=num_workers)
    return train, test
