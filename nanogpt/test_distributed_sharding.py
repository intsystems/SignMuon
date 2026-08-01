"""
Distributed-transport test for ``signmuon_optimizers.py``.

Verifies that the sharded ``step()`` (reduce_scatter -> owning-rank update ->
all_gather, with round-robin parameter ownership, per-owner optimizer state, and
padding when ``len(params) % world_size != 0``) produces EXACTLY the same
parameters as a single-process centralized reference that calls ``update_param``
on every parameter with the mean gradient.

It runs on CPU with the gloo backend and needs no GPUs. gloo does not implement
``reduce_scatter``; we install thin SYNCHRONOUS shims for ``reduce_scatter`` and
``all_gather`` (built on gloo's ``all_reduce`` / ``all_gather``) so the *real*
optimizer code path is exercised unchanged. On a real GPU cluster the production
code uses NCCL's native async collectives instead -- the shims are test-only.

To keep the comparison exact we (a) use float64, (b) replace Newton-Schulz with a
rank-truncated exact polar factor (stable under tiny perturbations), and (c) give
every rank the identical gradient for each parameter, so reduce_scatter's average
is exact. This isolates the systems logic (ownership / padding / state / gather),
which is the part being tested; the gradient averaging itself is torch's own.

Run:  python test_distributed_sharding.py            # world_size = 4
      python test_distributed_sharding.py 2          # world_size = 2
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("SIGNMUON_NO_COMPILE", "1")

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

T_STEPS = 8
ATOL = 1e-9


# --- rank-stable exact polar (U_r V_r^T over nonzero singular directions) -----
def _exact_polar(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Batched: the merged Q/K/V/O weight reaches the LMO as 4 stacked blocks, and
    each block must be truncated at its OWN rank -- so this cannot be a single
    batched matmul over a shared `r`."""
    A = G.to(torch.float64)
    if A.ndim > 2:
        flat = A.reshape(-1, *A.shape[-2:])
        return torch.stack([_exact_polar(b) for b in flat]).reshape(A.shape).to(G.dtype)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    if S.numel() == 0:
        return torch.zeros_like(A)
    r = int((S > 1e-9 * S[0]).sum().item())
    if r == 0:
        return torch.zeros_like(A)
    return (U[:, :r] @ Vh[:r, :]).to(G.dtype)


# --- gloo shims: synchronous reduce_scatter / all_gather returning done "work" -
class _DoneWork:
    def get_future(self):
        f = torch.futures.Future()
        f.set_result(None)
        return f

    def wait(self):
        return None


def _install_gloo_shims():
    real_all_reduce = dist.all_reduce
    real_all_gather = dist.all_gather

    def reduce_scatter(output, input_list, op=dist.ReduceOp.SUM, async_op=False, group=None):
        ws = dist.get_world_size()
        rank = dist.get_rank()
        stacked = torch.stack([t.detach().clone() for t in input_list], dim=0).contiguous()
        real_all_reduce(stacked, op=dist.ReduceOp.SUM)   # gloo supports SUM
        res = stacked[rank]
        if op == dist.ReduceOp.AVG:
            res = res / ws
        output.copy_(res)
        return _DoneWork()

    def all_gather(tensor_list, tensor, async_op=False, group=None):
        real_all_gather(tensor_list, tensor.contiguous())
        return _DoneWork()

    dist.reduce_scatter = reduce_scatter
    dist.all_gather = all_gather


# --- deterministic, rank-independent parameters and gradients -----------------
def _make_params():
    """12 params over three shapes (7 of (5,5), 2 of (4,4), 3 of (6,8) tagged attn).

    The counts are chosen so both padding regimes are exercised for
    ``world_size`` in {2, 4}:

    * 2 of (4,4): a group SHORTER than world_size (>=4), so some ranks own
      nothing at all in the group's single chunk;
    * 7 of (5,5): a group spanning SEVERAL chunks whose LAST one is partial
      (ws=2 -> 2/2/2/1, ws=4 -> 4/3). This is the case that catches a rank
      which owns something in an early chunk but nothing in the last one: if
      the padded chunk's reduce_scatter reuses that rank's earlier output
      buffer instead of a fresh scratch tensor, it silently zeroes an already
      averaged gradient. Record #40's real shapes hit exactly this (10
      attn_gate params on 8 ranks).
    * 3 of (6,8) tagged ``module="attn"``: record #40's MERGED Q/K/V/O weight,
      which both the LMO and the EF21 compressor split into 4 blocks
      (``reshape(4, 6, 2)``). Without a tagged parameter here the entire
      block-splitting path -- the batched LMO call and the per-layer compressor
      scale -- is never exercised by the sharded transport at all. The blocks are
      given deliberately DISPARATE magnitudes, as record #40's zero-inited O block
      produces, so a shared compressor scale would show up as a real difference
      rather than a rounding one.
    """
    g = torch.Generator().manual_seed(1234)
    out = []
    for s in [(5, 5)] * 7 + [(4, 4)] * 2:
        out.append(torch.empty(s, dtype=torch.float64).uniform_(-0.1, 0.1, generator=g))
    for _ in range(3):
        blocks = torch.empty(4, 6, 2, dtype=torch.float64).uniform_(-0.1, 0.1, generator=g)
        blocks *= torch.tensor([1.0, 1.0, 1.0, 0.01], dtype=torch.float64)[:, None, None]
        p = blocks.reshape(6, 8).contiguous()
        p.module = "attn"
        out.append(p)
    return out


def _grad(step: int, i: int, shape) -> torch.Tensor:
    # identical on every rank -> reduce_scatter AVG is exact
    g = torch.Generator().manual_seed(10_000 * step + i)
    return torch.empty(shape, dtype=torch.float64).normal_(0.0, 1.0, generator=g)


def _reference_final(name, cls, lr, mu, wd):
    params = _make_params()
    opt = cls(params, lr=lr, momentum=mu, weight_decay=wd)
    grp = {id(p): g for g in opt.param_groups for p in g["params"]}
    for t in range(T_STEPS):
        for i, p in enumerate(params):
            p.grad = _grad(t, i, tuple(p.shape))
            opt.update_param(p, grp[id(p)])
    # for UDSign the "model" that all_gather broadcasts is W (= the param tensor)
    return [p.detach().clone() for p in params]


def _distributed_final(name, cls, lr, mu, wd):
    params = _make_params()
    opt = cls(params, lr=lr, momentum=mu, weight_decay=wd)
    # locate each param's owning position to assign the right gradient each step
    for t in range(T_STEPS):
        for g in opt.param_groups:
            for p in g["params"]:
                # every rank sets the (identical) grad for every param; step() then
                # reduce_scatters so each owner sees the average (== the grad here)
                idx = _index_of(params, p)
                p.grad = _grad(t, idx, tuple(p.shape))
        opt.step()
    return [p.detach().clone() for p in params]


def _index_of(params, p):
    for i, q in enumerate(params):
        if q is p:
            return i
    raise KeyError


def _worker(rank, world_size):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    _install_gloo_shims()

    import signmuon_optimizers as smo
    smo.polar_express = _exact_polar  # stable, exact, fp64 (record #40's LMO slot)

    lr, mu, wd = 0.1, 0.9, 0.0
    failures = []
    for name, cls in smo.OPTIMIZERS.items():
        torch.manual_seed(0)
        ref = _reference_final(name, cls, lr, mu, wd)
        torch.manual_seed(0)
        got = _distributed_final(name, cls, lr, mu, wd)
        max_err = max(float((a - b).abs().max()) for a, b in zip(ref, got))
        ok = max_err < ATOL
        if rank == 0:
            print(f"  {'OK  ' if ok else 'FAIL'} {name:<16} world_size={world_size}  max|diff|={max_err:.2e}")
        if not ok:
            failures.append(f"{name}: max|diff|={max_err:.2e} >= {ATOL}")

    dist.barrier()
    dist.destroy_process_group()
    if failures:
        raise AssertionError(f"[rank {rank}] {len(failures)} sharding mismatch(es):\n" + "\n".join(failures))


# =============================================================================
# Portable single-process simulation (no gloo, no multiprocessing)
# =============================================================================
# gloo is not available on every platform -- notably it refuses to initialise on
# Windows ("unsupported gloo device") -- and the sharding logic is exactly the
# part most worth testing before renting GPUs. So simulate the world instead of
# spawning it.
#
# The simulation is exact for THIS test's setup, where every rank holds the same
# gradient for every parameter:
#   * reduce_scatter(out, in_list, AVG) then reduces to  out <- in_list[rank],
#   * all_gather is just "take each parameter's value from its owning rank",
# so a world can be replayed as `world_size` independent single-process runs
# (each on its own copy of the parameters, each doing only its own rank's work)
# followed by a merge that picks every parameter from its owner. That is
# precisely what the collectives would have produced.
#
# It reproduces the failure mode a naive implementation has: in the padded final
# chunk of a group, a rank that owns nothing must reduce_scatter into a FRESH
# scratch buffer. Reusing the previous chunk's buffer aliases a parameter that
# rank *does* own and zeroes its already-averaged gradient, which shows up here
# as that parameter not being updated.


class _FakeDist:
    """Stand-in for ``torch.distributed`` replaying one rank of a world."""

    def __init__(self, world_size):
        self.world_size = world_size
        self.rank = 0

    def get_rank(self):
        return self.rank

    def get_world_size(self):
        return self.world_size

    def reduce_scatter(self, output, input_list, op=None, async_op=False):
        # identical inputs on every rank => the average is input_list[rank]
        output.copy_(input_list[self.rank])
        return _DoneWork()

    def all_gather(self, tensor_list, tensor, async_op=False):
        return _DoneWork()      # the merge below stands in for the broadcast

    class ReduceOp:
        AVG = "avg"
        SUM = "sum"


def _owner_of(opt, params, world_size):
    """global parameter index -> the rank that owns it, from the group layout."""
    pos = {id(p): i for i, p in enumerate(params)}
    owner = {}
    for group in opt.param_groups:
        gp = group["params"]
        for base_i in range(0, len(gp), world_size):
            for r in range(world_size):
                if base_i + r < len(gp):
                    owner[pos[id(gp[base_i + r])]] = r
    assert len(owner) == len(params), "every parameter must have exactly one owner"
    return owner


def _simulated_final(cls, world_size, lr, mu, wd):
    """Replay a ``world_size``-rank run: every rank steps, then the owners'
    parameters are broadcast to every replica (which is what all_gather does)."""
    import signmuon_optimizers as smo

    fake = _FakeDist(world_size)
    real_dist = smo.dist
    smo.dist = fake
    try:
        replicas = [_make_params() for _ in range(world_size)]
        opts = [cls(ps, lr=lr, momentum=mu, weight_decay=wd) for ps in replicas]
        owner = _owner_of(opts[0], replicas[0], world_size)
        n = len(replicas[0])
        for t in range(T_STEPS):
            for r in range(world_size):
                for i, p in enumerate(replicas[r]):
                    p.grad = _grad(t, i, tuple(p.shape))
                fake.rank = r
                opts[r].step()
            # all_gather: each parameter's value comes from its owning rank
            merged = [replicas[owner[i]][i].detach().clone() for i in range(n)]
            for r in range(world_size):
                for i in range(n):
                    replicas[r][i].detach().copy_(merged[i])
        return [p.detach().clone() for p in replicas[0]]
    finally:
        smo.dist = real_dist


def _run_simulated(world_sizes=(1, 2, 4, 8)):
    import signmuon_optimizers as smo
    smo.polar_express = _exact_polar

    lr, mu, wd = 0.1, 0.9, 0.0
    failures = []
    for world_size in world_sizes:
        for name, cls in smo.OPTIMIZERS.items():
            torch.manual_seed(0)
            ref = _reference_final(name, cls, lr, mu, wd)
            torch.manual_seed(0)
            got = _simulated_final(cls, world_size, lr, mu, wd)
            max_err = max(float((a - b).abs().max()) for a, b in zip(ref, got))
            ok = max_err < ATOL
            print(f"  {'OK  ' if ok else 'FAIL'} {name:<16} world_size={world_size}  "
                  f"max|diff|={max_err:.2e}")
            if not ok:
                failures.append(f"{name} (world_size={world_size}): max|diff|={max_err:.2e}")
    assert not failures, (f"{len(failures)} sharding mismatch(es):\n" + "\n".join(failures))


def test_sharding_simulated():
    """pytest entry point for the portable simulation."""
    _run_simulated()


def main():
    world_size = int(sys.argv[1]) if len(sys.argv) > 1 else 4

    print(f"Portable sharding simulation ({len(_make_params())} params, "
          f"{T_STEPS} steps, world sizes 1/2/4/8)...\n")
    _run_simulated()
    print("\nSimulated sharded step() matches the centralized reference. PASS.\n")

    if os.environ.get("SIGNMUON_SKIP_GLOO") == "1":
        return
    print(f"Real-collectives test (gloo, world_size={world_size})...\n")
    try:
        dist.init_process_group(backend="gloo", rank=0, world_size=1,
                                init_method="tcp://127.0.0.1:29518")
        dist.destroy_process_group()
    except Exception as exc:                     # e.g. Windows: no gloo device
        print(f"  SKIPPED: gloo unavailable here ({type(exc).__name__}: {exc}).")
        print("  Run this on the Linux server before the real runs.")
        return
    mp.spawn(_worker, args=(world_size,), nprocs=world_size, join=True)
    print("\nSharded step() matches the centralized reference for every optimizer. PASS.")


if __name__ == "__main__":
    main()
