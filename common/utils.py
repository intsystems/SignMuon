"""Shared utilities: seeding, run directories, and the metrics schema.

Everything here exists so that a run is (a) reproducible from its recorded
config alone and (b) *aggregatable across seeds* without any post-hoc guessing
about which curve point corresponds to which round.

Metrics schema
--------------
Every experiment writes ``metrics.json`` with exactly this shape::

    {
      "config":  {... , "seed": 0},          # full run configuration
      "history": {
        "steps":     [0, 100, 200, ...],     # x-axis: round (federated) or epoch
        "test_acc":  [...],                  # one value per entry of "steps"
        "test_loss": [...],
        ...                                  # any number of further series
      }
    }

Only *evaluated* points are recorded, and each is paired with its step index.
The old format stored one entry per round and forward-filled the previous value
whenever ``eval_freq > 1``, which silently turns un-measured rounds into flat
plateaus and makes a cross-seed mean meaningless. ``History`` below records the
x-axis explicitly instead, so curves from different seeds (or different
``eval_freq``) line up by construction.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from common.paths import results_root as _results_root


# --------------------------------------------------------------------------
# Seeding
# --------------------------------------------------------------------------


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (all CUDA devices).

    ``deterministic=True`` also pins cuDNN to deterministic kernels, which is
    what the paper's reproducibility statement claims. Note that this makes
    convolutions measurably slower; pass ``False`` for throughput runs where the
    exact bitwise trajectory does not matter.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:  # pragma: no cover - needs num_workers>0
    """``worker_init_fn`` that seeds NumPy/random inside DataLoader workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_device(spec: Optional[str]) -> torch.device:
    """Resolve a device string, degrading to CPU when CUDA is unavailable.

    Accepts ``None``, ``"cpu"``, ``"cuda"``, ``"cuda:3"``. A CUDA request on a
    CPU-only machine returns CPU rather than raising, so the same command line
    works on a laptop and on the cluster.
    """
    if spec is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dev = torch.device(spec)
    if dev.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return dev


# --------------------------------------------------------------------------
# Metrics history
# --------------------------------------------------------------------------


class History:
    """Step-indexed metric series.

    ``record(step, test_acc=..., test_loss=...)`` appends one point. Series
    names are free-form; every series ends up the same length as ``steps``
    because a missing key is stored as ``None`` (JSON ``null``) rather than
    being forward-filled.
    """

    def __init__(self) -> None:
        self.steps: List[int] = []
        self.series: Dict[str, List[Optional[float]]] = {}

    def record(self, step: int, **values: Optional[float]) -> None:
        self.steps.append(int(step))
        n = len(self.steps)
        for key, val in values.items():
            col = self.series.setdefault(key, [])
            col.extend([None] * (n - 1 - len(col)))     # backfill a new series
            col.append(None if val is None else float(val))
        for key, col in self.series.items():            # pad series not passed
            col.extend([None] * (n - len(col)))

    def last(self, key: str) -> Optional[float]:
        col = self.series.get(key) or []
        for val in reversed(col):
            if val is not None:
                return val
        return None

    def values(self, key: str) -> List[float]:
        """Recorded (non-``None``) values of one series, in step order."""
        return [v for v in (self.series.get(key) or []) if v is not None]

    def last_k_mean(self, key: str, k: int = 5) -> Optional[float]:
        """Mean of the final ``k`` recorded values.

        The primary reported metric: averaging the tail removes the
        epoch-to-epoch fluctuation that makes a single final-epoch number a
        coin flip between two close methods, at zero extra compute.
        """
        vals = self.values(key)
        if not vals:
            return None
        tail = vals[-max(1, k):]
        return sum(tail) / len(tail)

    def argbest(self, key: str, mode: str = "max") -> Optional[int]:
        """Step index at which ``key`` is best. Used for early stopping on *val*."""
        col = self.series.get(key)
        if not col:
            return None
        pairs = [(s, v) for s, v in zip(self.steps, col) if v is not None]
        if not pairs:
            return None
        pick = max if mode == "max" else min
        return pick(pairs, key=lambda sv: sv[1])[0]

    def at(self, key: str, step: int) -> Optional[float]:
        """Value of ``key`` at a given step, or ``None``."""
        col = self.series.get(key)
        if not col:
            return None
        for s, v in zip(self.steps, col):
            if s == step:
                return v
        return None

    def steps_to_target(self, key: str, target: float, mode: str = "ge") -> Optional[int]:
        """First step at which ``key`` reaches ``target`` (``None`` if never).

        Separates *speed* from *final quality*: two methods can share a final
        accuracy while one gets there in half the epochs.
        """
        col = self.series.get(key)
        if not col:
            return None
        for s, v in zip(self.steps, col):
            if v is None:
                continue
            if (mode == "ge" and v >= target) or (mode == "le" and v <= target):
                return s
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {"steps": list(self.steps), **{k: list(v) for k, v in self.series.items()}}


# --------------------------------------------------------------------------
# Run directories
# --------------------------------------------------------------------------


def results_root() -> Path:
    """``code/results`` -- one tree for every experiment family.

    Each family writes to its own subdirectory (``results/centralized``,
    ``results/federated``, ``results/synthetic``), so ``aggregate.py`` can sweep
    everything with a single scan and the experiment code does not need to know
    where it lives on disk.

    Set ``SIGNMUON_RESULTS`` to put that tree somewhere else -- another drive, a
    scratch filesystem, a network share. A multi-day sweep writes a ``model.pt``
    per job, and running the system disk out of space mid-run loses the night;
    pointing this at a roomy volume is cheaper than discovering that at 04:00.

    Defined in `common.paths`, which imports no torch, so that the exporters and
    plotters can resolve the same override without it.
    """
    return _results_root()


def run_dir(saves_root: Path | str, run_name: str, seed: int) -> Path:
    """``<saves_root>/<run_name>/seed<seed>``.

    Seeds live in sibling directories so that repeated runs of the same config
    accumulate instead of overwriting each other -- the previous layout wrote
    every seed to the same path and ``rmtree``'d it first, so a multi-seed sweep
    silently kept only the last seed.
    """
    return Path(saves_root) / run_name / f"seed{int(seed)}"


def save_run(
    out_dir: Path | str,
    config: Any,
    history: History | Dict[str, Any],
    model: Optional[torch.nn.Module] = None,
) -> Path:
    """Write ``metrics.json`` (and optionally ``model.pt``) into ``out_dir``.

    Never deletes anything: an existing ``metrics.json`` is overwritten in
    place, but sibling files (logs, other seeds) survive.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = asdict(config) if is_dataclass(config) and not isinstance(config, type) else dict(config)
    hist = history.to_dict() if isinstance(history, History) else dict(history)

    # Stamp the machine into the run itself. Different experiments in this
    # project run on different GPUs, so hardware is a property of the run and
    # not of the repository -- recovering it later from memory is exactly the
    # thing that goes wrong. Never fatal: a metrics file without a hardware
    # block is better than a lost run.
    try:
        from common.hardware import describe
        cfg.setdefault("hardware", describe(str(cfg.get("device")) if cfg.get("device") else None))
    except Exception:                                       # noqa: BLE001
        pass

    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "history": hist}, f, indent=2)

    if model is not None:
        torch.save({"model_state_dict": model.state_dict(), "config": cfg}, out / "model.pt")

    return out


# --------------------------------------------------------------------------
# Parameter routing (shared by the centralized and federated code paths)
# --------------------------------------------------------------------------


def split_param_names(model: torch.nn.Module, n_head_tensors: int = 2) -> tuple[list[str], list[str]]:
    """Split parameters into (matrix, auxiliary) name lists.

    Following the paper (and Muon practice), the LMO/sign rule applies only to
    matrix-valued parameters; biases, BatchNorm scales (``ndim < 2``) and the
    final classification layer go to AdamW. ``n_head_tensors`` is how many
    trailing tensors count as "the classification head" (weight + bias => 2).

    Both the centralized and the federated code call this, so the two settings
    cannot drift apart in which parameters they treat as matrices.
    """
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    head = {n for n, _ in named[-n_head_tensors:]} if n_head_tensors > 0 else set()
    matrix = [n for n, p in named if p.ndim >= 2 and n not in head]
    aux = [n for n, p in named if n not in matrix]
    return matrix, aux


def cosine_lr(base_lr: float, step: int, total_steps: int, min_lr: float = 0.0) -> float:
    """Cosine-annealed learning rate, matching Equation (12) of the paper.

    ``eta_t = eta_min + 0.5*(eta_max - eta_min)*(1 + cos(pi * t / T_max))``.
    Kept here (rather than inline per method) so that every algorithm in the
    federated driver provably shares one schedule.
    """
    import math

    if total_steps <= 0:
        return base_lr
    cos = 0.5 * (1.0 + math.cos(math.pi * step / total_steps))
    return min_lr + (base_lr - min_lr) * cos


def split_list_arg(values: Optional[Sequence[str]],
                   valid: Optional[Sequence[str]] = None,
                   name: str = "value") -> Optional[list[str]]:
    """Flatten a list-valued CLI argument, accepting commas as well as spaces.

    ``--stages grid final``, ``--stages grid,final`` and ``--stages grid, final``
    all mean the same thing to a reader, so they should mean the same thing to
    argparse. With ``nargs="+"`` alone the second and third fail on a token that
    still has a comma stuck to it, and the error names the mangled token rather
    than the mistake.

    Order is preserved and duplicates dropped, so ``grid,grid final`` is
    ``["grid", "final"]``. When ``valid`` is given, unknown names raise
    ``ValueError`` listing what was accepted -- do the validation here rather
    than through argparse's ``choices``, which fires before this can split.
    """
    if values is None:
        return None
    out: list[str] = []
    for token in values:
        for piece in str(token).split(","):
            piece = piece.strip()
            if piece and piece not in out:
                out.append(piece)
    if valid is not None:
        unknown = [p for p in out if p not in valid]
        if unknown:
            raise ValueError(
                f"unknown {name}{'s' if len(unknown) > 1 else ''}: "
                f"{', '.join(unknown)}. Valid: {', '.join(valid)}")
    return out
