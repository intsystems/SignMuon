# =============================================================================
# SINGLE-A100 BUILD of the record-#40 (2025-10-04, pre-NorMuon) NanoGPT speedrun.
#
# This is a copy of train_gpt.py (which is record #40 + the SignMuon/EF21 optimizer
# knob) with the two genuinely Hopper-only pieces swapped for Ampere-safe, device-
# agnostic equivalents.  Every change is wrapped in an  # ===== [A100 DIFF #k] ...
# banner, so a diff against train_gpt.py shows exactly what differs.  There are
# only three kinds of change, none of which touches the sign/EF21 optimizer math:
#
#   [A100 DIFF #1]  single GPU: rely on record #40's OWN grad-accumulation path
#                   (grad_accum_steps = 8 // world_size), so 1 A100 sees the same
#                   global batch and the same optimizer steps as 8xH100.  We only
#                   add torchrun env defaults so plain `python` works.
#   [A100 DIFF #2]  Flash Attention 3  ->  FlexAttention  (FA3 is Hopper-only).
#                   The block mask reproduces #40's mask EXACTLY (per-document
#                   causal + left sliding window of bm_size tokens).
#   [A100 DIFF #3]  FP8 lm_head  ->  bf16  (FP8 tensor cores are Hopper-only).
#                   This is the ONE unavoidable numerical difference, and it only
#                   touches the lm_head matmul (a DistAdam param), not the methods
#                   under study.  torch.compile fullgraph is relaxed for the mask.
#
# The optimizer (signmuon_optimizers.py, pure-torch Polar Express LMO) is BYTE-FOR-
# BYTE the same object on A100 and 8xH100 -- only the attention backend and the
# lm_head precision differ, so the sign/EF21 update geometry is identical.  Loss
# curves should track the 8xH100 run closely (validate with a short 1xH100-FA3 vs
# 1xA100-Flex A/B run before trusting long A100 sweeps; see README).
#
# Run:  DISABLE_FP8=1 torchrun --standalone --nproc_per_node=1 train_gpt_a100.py
#   (or plain) SIGNMUON_OPT=EF21-MuonUSign python train_gpt_a100.py
# =============================================================================
import os
import sys

# ===== [A100 DIFF #1: single-GPU env defaults] ===============================
# Record #40 already supports 1..8 GPUs via grad_accum_steps = 8 // world_size, so
# NOTHING about the batch/step math changes on one GPU -- Muon and Adam are scale-
# invariant, so the missing 1/8 in reduce_scatter(AVG) washes out (the record
# authors' own "fewer GPUs" path).  These defaults just let `python train_gpt_a100.py`
# work without torchrun; torchrun --nproc_per_node=1 overrides them.
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("DISABLE_FP8", "1")   # [A100 DIFF #3] FP8 is Hopper-only -> bf16 lm_head
# =============================================================================

with open(__file__) as f:
    code = f.read()  # read the code of this file ASAP, for logging
import copy
import glob
import json
import math
import threading
import time
import uuid
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch

torch.empty(
    1, device="cuda", requires_grad=True
).backward()  # prevents a bug on some systems
import torch._dynamo as dynamo
import torch.distributed as dist
import torch.nn.functional as F

# torch._inductor.config.coordinate_descent_tuning = True # we have banned this flag for new records because it causes compilation to take 30min
# [A100 DIFF #2] `from kernels import get_kernel` removed: it only fetched the
# Hopper-only Flash Attention 3 kernel, which we replace with FlexAttention below.
from torch import Tensor, nn

dynamo.config.recompile_limit = 64

# -----------------------------------------------------------------------------
# Custom operators: FP8 matmul by @YouJiacheng


@torch.library.custom_op("nanogpt::mm", mutates_args=())
def mm_op(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor, Tensor]:
    @torch.compile
    def impl(x: Tensor, w: Tensor):
        assert x.is_contiguous() and w.is_contiguous()
        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        w_f8 = w.div(w_s).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            x_f8,
            w_f8.T,
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(x_s, dtype=torch.float32),
            scale_b=x.new_tensor(w_s, dtype=torch.float32),
            use_fast_accum=True,
        )
        return out, x_f8, w_f8

    return impl(x, w)

@mm_op.register_fake
def _(x: Tensor, w: Tensor, *_):
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[1]
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    return x @ w.T, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)

@torch.library.custom_op("nanogpt::mm_backward", mutates_args=())
def mm_backward_op(g: Tensor, x_f8: Tensor, w_f8: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor]:
    @torch.compile
    def impl(grad: Tensor, x_f8: Tensor, w_f8: Tensor):
        assert grad.is_contiguous()
        x_inv_s = grad.new_tensor(x_s, dtype=torch.float32)
        w_inv_s = grad.new_tensor(w_s, dtype=torch.float32)
        grad_inv_s = grad.new_tensor(grad_s, dtype=torch.float32)
        grad_f8 = grad.div(grad_s).to(torch.float8_e5m2)
        grad_x = torch._scaled_mm(
            grad_f8,
            w_f8.T.contiguous().T,
            out_dtype=torch.bfloat16,
            scale_a=grad_inv_s,
            scale_b=w_inv_s,
            use_fast_accum=False,
        )
        # faster than grad_f8_t @ x_f8, for (d_out, d_in) == (50304, 768)
        grad_w = torch._scaled_mm(
            x_f8.T.contiguous(),
            grad_f8.T.contiguous().T,
            out_dtype=torch.float32,
            scale_a=x_inv_s,
            scale_b=grad_inv_s,
            use_fast_accum=False,
        ).T
        return grad_x, grad_w

    return impl(g, x_f8, w_f8)

@mm_backward_op.register_fake
def _(g: Tensor, x_f8: Tensor, w_f8: Tensor, *_):
    return x_f8.to(torch.bfloat16), w_f8.T.contiguous().T.to(torch.float32)

def backward(ctx, grad_out: Tensor, *_):
    x_f8, w_f8 = ctx.saved_tensors
    x_s, w_s, grad_s = ctx.scales
    grad_x, grad_w = torch.ops.nanogpt.mm_backward(
        grad_out, x_f8, w_f8, x_s, w_s, grad_s
    )
    return grad_x, grad_w, None, None, None

def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    *_, x_s, w_s, grad_s = inputs
    _, x_f8, w_f8 = output
    ctx.save_for_backward(x_f8, w_f8)
    ctx.scales = x_s, w_s, grad_s
    ctx.set_materialize_grads(False)

mm_op.register_autograd(backward, setup_context=setup_context)

# -----------------------------------------------------------------------------
# Optimizers.  Record #40's Triton Newton/Polar kernels and its bespoke `Muon`
# class are REPLACED here by the shared, unit-tested `signmuon_optimizers` module,
# which hosts all eight paper optimizers (SignMuon, EF21-*, MuonUSign, ... and a
# reference Muon) on a pure-torch Polar Express LMO (record #40's coeffs, no Triton
# -> identical math on A100 and H100, and CPU-testable).  DistAdam (embeddings /
# scalars / head / gates-as-Adam) is kept verbatim from record #40 below.
from signmuon_optimizers import (  # noqa: E402
    polar_express, OPTIMIZERS, PAPER_METHODS, EF21MuonSign,
    LR_SCALING_RULES, describe_lr_scaling,
)

class DistAdam(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        params = list(params)
        sizes = {p.shape for p in params}
        # create one buffer per unique parameter-size
        param_groups = []
        for size in sizes:
            group_params = [p for p in params if p.shape == size]
            param_groups.append(dict(params=group_params))
        super().__init__(param_groups, defaults)
        # DistributedAdam implementation by @vagrawal

    @torch.compile
    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures: list[torch.Future] = []
        all_gather_futures: list[torch.Future] = []
        grad_slices = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            for base_i in range(len(params)):
                grad = params[base_i].grad
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                reduce_scatter_futures.append(dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future())
                grad_slices.append(grad_slice)

        idx = 0
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            params = group['params']
            for base in range(len(params)):
                reduce_scatter_futures[idx].wait()
                p = params[base]
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]
                lr = group['lr'] * getattr(p, "lr_mul", 1.0)
                state = self.state[p]
                g_slice = grad_slices[idx]
                # State init
                if not state:
                    state["step"] = torch.tensor(
                        0, dtype=torch.int64, device=p.device
                    )
                    state["exp_avg"] = torch.zeros(
                        p_slice.shape,
                        dtype=torch.bfloat16,
                        device=p_slice.device,
                    )
                    state["exp_avg_sq"] = torch.zeros(
                        p_slice.shape,
                        dtype=torch.bfloat16,
                        device=p_slice.device,
                    )
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]
                # weight decay
                if wd != 0:
                    eff_weight_decay = lr * wd * getattr(p, "wd_mul", 1.0)
                    p_slice.mul_(1 - eff_weight_decay)
                # update running averages
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)
                # bias corrections
                bias1 = 1 - beta1 ** t
                bias2 = 1 - beta2 ** t
                # compute step
                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr * (torch.sqrt(bias2) / bias1)
                update = exp_avg.div(denom).mul_(step_size)
                p_slice.add_(other=update, alpha=-1.0)
                idx += 1
                all_gather_futures.append(dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future())
        torch.futures.collect_all(all_gather_futures).wait()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__(in_features, out_features, bias=False)
        self.use_fp8 = use_fp8
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

    def reset_parameters(self) -> None:
        std = 0.5 * (self.in_features ** -0.5) # 0.5 is a bit better than the default 1/sqrt(3)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor):
        if self.use_fp8 and self.training:
            _x = x.flatten(0, -2)
            out: Tensor = torch.ops.nanogpt.mm(_x, self.weight, x_s=self.x_s, w_s=self.w_s, grad_s=self.grad_s)[0]
            return out.reshape(*x.shape[:-1], -1)
        else:
            return F.linear(x, self.weight.type_as(x))

# yarn implementation @classiclarryd
class Yarn(nn.Module):
    def __init__(self, head_dim, max_seq_len):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.reset()
        
    def reset(self):
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=self.head_dim//4, dtype=torch.float32, device=device)
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(self.head_dim//4)])
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=device)
        theta = torch.outer(t, angular_freq)
        self.cos = nn.Buffer(
            theta.cos().to(torch.bfloat16), persistent=False
        )
        self.sin = nn.Buffer(
            theta.sin().to(torch.bfloat16), persistent=False
        )
        self.angular_freq = angular_freq
        # start with 0.1, inspired by 0.12 from @leloykun and learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        self.attn_scale = 0.1

    def apply(self, old_window: int, new_window: int, alpha: int=1, beta: int=32):
        rotations = args.block_size * old_window * self.angular_freq / (2 * torch.pi)
        scaling_factor = old_window / new_window
        interpolation_weight = torch.clamp((rotations - alpha) / (beta - alpha), 0, 1)
        self.angular_freq *= scaling_factor + interpolation_weight * (1 - scaling_factor)
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=self.angular_freq.device)
        theta = torch.outer(t, self.angular_freq)
        self.cos.copy_(theta.cos())
        self.sin.copy_(theta.sin())
        self.attn_scale *= 0.2 * math.log(new_window / old_window) + 1

def rotary(x_BTHD: Tensor, cos: Tensor, sin: Tensor):
    assert cos.size(0) >= x_BTHD.size(-3)
    cos, sin = (
        cos[None, : x_BTHD.size(-3), None, :],
        sin[None, : x_BTHD.size(-3), None, :],
    )
    x1, x2 = x_BTHD.chunk(2, dim=-1)
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), 3)

@dataclass
class AttnArgs:
    ve: torch.Tensor
    sa_lambdas: torch.Tensor
    block_mask: object          # [A100 DIFF #2] FlexAttention BlockMask (was: seqlens + bm_size)
    cos: torch.Tensor
    sin: torch.Tensor
    attn_scale: float

# ===== [A100 DIFF #2: FlexAttention instead of Flash Attention 3] ============
# Record #40 attends with flash_attn_varlen_func (FA3), which needs Hopper tensor
# cores and will not run on an A100.  We use torch's device-agnostic FlexAttention
# with a block mask that reproduces #40's mask EXACTLY:
#   * per-document causal   -- documents are delimited by the BOS token (50256),
#     the same boundaries #40's loader packs into `seqlens`;
#   * left sliding window of `bm_size` tokens -- matches FA3's
#     window_size=(bm_size, 0) up to a 1-token boundary convention.
# RoPE/YaRN are applied to q,k BEFORE attention exactly as in #40 (global token
# positions across the packed buffer), so nothing about positions changes; only the
# attention *kernel* differs.  The mask is built in a @torch.compiler.disable helper
# so create_block_mask runs eagerly even though the model is torch.compiled -- this
# keeps the data loader and the model.forward signature identical to train_gpt.py
# (the `seqlens` arg is simply ignored on this build).
from torch.nn.attention.flex_attention import BlockMask, flex_attention, create_block_mask

_BOS_ID = 50256

@torch.compiler.disable  # build the mask eagerly, outside the compiled graph
def build_flex_block_mask(input_seq: Tensor, window_tokens: int, block_size: int = 128):
    """BlockMask for one packed sequence: causal, within-document (BOS-delimited),
    within a left window of `window_tokens` tokens == FA3 window_size=(window_tokens, 0)."""
    docs = (input_seq == _BOS_ID).cumsum(0)
    def mask_mod(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        window = (q_idx - kv_idx) <= window_tokens
        same_doc = docs[q_idx] == docs[kv_idx]
        return causal & window & same_doc
    T = input_seq.numel()
    return create_block_mask(mask_mod, B=None, H=None, Q_LEN=T, KV_LEN=T,
                             device=input_seq.device, BLOCK_SIZE=block_size, _compile=True)
# =============================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = dim
        self.hdim = num_heads * head_dim

        assert self.hdim == self.dim, "num_heads * head_dim must equal model_dim"
        std = 0.5 * (self.dim ** -0.5)
        bound = (3 ** 0.5) * std # improved init scale by @YouJiacheng
        # merged QKV weights: suggested by many, implemented by @fernbear.bsky.social, and further improved by @YouJiacheng
        # https://x.com/hi_tysam/status/1879699187107033311
        # make matrices the same shape as MLP to enable batched call in optimizer
        self.qkvo_w = nn.Parameter(torch.empty(self.hdim, self.dim*4))
        # label module to enable custom optimizer sizing
        self.qkvo_w.module='attn'
        # SIGNMUON: semantic (fan_out, fan_in) of the linear map this parameter
        # implements, for lr_scaling="semantic" (see signmuon_optimizers.py). Inert
        # under the default "unit-gain" rule, which reads the STORED shape exactly as
        # record #40 does. Each of Q/K/V/O is a [hdim, dim] map.
        self.qkvo_w.fan_out_sem, self.qkvo_w.fan_in_sem = self.hdim, self.dim
        with torch.no_grad():
            self.qkvo_w.view(4,self.hdim, self.dim)[:3].uniform_(-bound, bound) # init QKV weights
            self.qkvo_w.view(4,self.hdim, self.dim)[3].zero_() # init output weights to zero

        # sparse gated attention to enable context based no-op by @classiclarryd
        self.attn_gate = CastedLinear(12, num_heads)
        # label module to enable custom optimizer sizing
        self.attn_gate.weight.module = 'attn_gate'
        self.attn_gate.weight.fan_out_sem, self.attn_gate.weight.fan_in_sem = num_heads, 12
        self.attn_gate.weight.detach().zero_()

    def forward(self, x: Tensor, attn_args: AttnArgs):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "varlen sequences requires B == 1"
        assert T % 16 == 0
        # unpack attention args
        cos, sin = attn_args.cos, attn_args.sin
        ve, sa_lambdas = attn_args.ve, attn_args.sa_lambdas
        block_mask, attn_scale = attn_args.block_mask, attn_args.attn_scale  # [A100 DIFF #2]

        q, k, v = F.linear(x, self.qkvo_w.view(4,self.hdim, self.dim)[:3].flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k) # QK norm @Grad62304977
        q, k = rotary(q, cos, sin), rotary(k, cos, sin)
        if ve is not None:
            v = sa_lambdas[0] * v + sa_lambdas[1] * ve.view_as(v) # @ KoszarskyB & @Grad62304977
        else: # skip mid-layers token value embeddings by @YouJiacheng
            v = sa_lambdas[0] * v

        # ===== [A100 DIFF #2] FlexAttention (device-agnostic) replaces FA3 varlen =====
        # FA3 took [T, H, D] + cu_seqlens; FlexAttention takes [B, H, T, D] + a BlockMask
        # (built in GPT.forward). Same per-document causal + windowed masking, same YaRN
        # softmax scale. q,k,v are bf16 as in #40, which FlexAttention supports.
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))   # [B,T,H,D] -> [B,H,T,D]
        y = flex_attention(q, k, v, block_mask=block_mask, scale=attn_scale)
        y = y.transpose(1, 2)                              # [B,H,T,D] -> [B,T,H,D]
        # =============================================================================
        y = y * torch.sigmoid(self.attn_gate(x[..., :self.attn_gate.weight.size(-1)])).view(B, T, self.num_heads, 1)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim) # re-assemble all head outputs side by side
        y = F.linear(y, self.qkvo_w.view(4,self.hdim, self.dim)[3].type_as(y))
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        # make matrices the same shape to enable batched call in optimizer
        self.c_fc = nn.Parameter(torch.empty(dim, hdim))
        self.c_proj = nn.Parameter(torch.empty(dim, hdim))
        # label modules to enable custom optimizer sizing
        self.c_fc.module='mlp'
        self.c_proj.module='mlp'
        # SIGNMUON: semantic (fan_out, fan_in). c_fc is STORED TRANSPOSED -- [dim, hdim]
        # but applied as F.linear(x, c_fc.T), i.e. a dim -> hdim map -- so its stored
        # shape is (fan_in, fan_out) and it is the ONE parameter lr_scaling="semantic"
        # moves (by 2x). c_proj is stored as (fan_out, fan_in) already.
        self.c_fc.fan_out_sem, self.c_fc.fan_in_sem = hdim, dim
        self.c_proj.fan_out_sem, self.c_proj.fan_in_sem = dim, hdim
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std # improved init scale by @YouJiacheng
        with torch.no_grad():
            self.c_fc.uniform_(-bound, bound)
            self.c_proj.zero_() # zero init suggested by @Grad62304977

    def forward(self, x: Tensor):
        x = F.linear(x, self.c_fc.T.type_as(x))
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = F.linear(x, self.c_proj.type_as(x))
        return x

class Block(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int, layer_idx: int):
        super().__init__()
        # skip attention of blocks.7 (the 8th layer) by @YouJiacheng
        self.attn = CausalSelfAttention(dim, head_dim, num_heads) if layer_idx not in [0, 7] else None
        # skip MLP blocks for first MLP layer by @EmelyanenkoK
        self.mlp = MLP(dim) if layer_idx != 0 else None

    def forward(self, x: Tensor, x0: Tensor, lambdas: Tensor, attn_args: AttnArgs):
        x = lambdas[0] * x + lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(norm(x), attn_args)
        if self.mlp is not None:
            x = x + self.mlp(norm(x))
        return x

# -----------------------------------------------------------------------------
# The main model

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, head_dim: int, model_dim: int, max_seq_len: int):
        super().__init__()
        vocab_size = next_multiple_of_n(vocab_size, n=128)
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.smear_gate = CastedLinear(12, 1)
        self.smear_gate.weight.detach().zero_()
        # label modules to enable custom optimizer sizing
        self.smear_gate.weight.module = 'smear_gate'
        self.smear_gate.weight.fan_out_sem, self.smear_gate.weight.fan_in_sem = 1, 12
        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        # value embedding code simplification inspired by @ragulpr https://github.com/KellerJordan/modded-nanogpt/pull/78
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        self.blocks = nn.ModuleList([Block(model_dim, head_dim, num_heads, i) for i in range(num_layers)])
        self.yarn = Yarn(head_dim, max_seq_len)
        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        # suggested to me by @Grad62304977. this originates from Karpathy's experiments.
        use_fp8 = not os.environ.get("DISABLE_FP8", False)
        self.lm_head = CastedLinear(model_dim, vocab_size, use_fp8=use_fp8, x_s=(model_dim**0.5)/448, w_s=2**-9, grad_s=1/448)
        self.lm_head.weight.detach().zero_() # @Grad62304977
        # Add learnable skip connection weights for decoder layers
        assert num_layers % 2 == 0
        pad = (-num_layers * 5 - 2) % dist.get_world_size()
        self.scalars = nn.Parameter(
            torch.cat(
                [
                    -1.5
                    * torch.ones(num_layers),  # skip_weights -> σ(-1.5) ≈ 0.18
                    *[
                        torch.tensor([1.0, 0.0]) for _ in range(num_layers)
                    ],  # block lambdas
                    *[
                        torch.tensor([0.5, 0.5]) for _ in range(num_layers)
                    ],  # SA lambdas
                    torch.zeros(1), # smear_lambda
                    0.5*torch.ones(1), # backout_lambda
                    torch.ones(pad),
                ]
            )
        )
        # set learning rates
        for param in self.embed.parameters():
            param.lr_mul = 75.
        for param in self.value_embeds.parameters():
            param.lr_mul = 75.
        self.lm_head.weight.lr_mul = 1.0
        self.scalars.lr_mul = 5.0

    def forward(self, input_seq: Tensor, target_seq: Tensor, seqlens: Tensor, ws_short: int, ws_long: int):
        assert input_seq.ndim == 1

        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        # 012 ... 012 structure on token value embeddings by @YouJiacheng, improved on @leloykun's U-net structure
        ve = [None, ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]
        assert len(ve) == len(self.blocks)

        # ===== [A100 DIFF #2] build FlexAttention block masks (replaces per-layer bm_size) =
        # Two masks (long/short window) for this packed sequence, assigned per layer exactly
        # as #40 assigned bm_sizes: long window at layers 4 & 11, short elsewhere, and layers
        # 0 & 7 have no attention (mask unused). The FA3 `seqlens` arg is ignored -- the mask
        # derives document boundaries from BOS(=50256) tokens in input_seq, an identical structure.
        long_bm = build_flex_block_mask(input_seq, ws_long * args.block_size, args.block_size)
        short_bm = build_flex_block_mask(input_seq, ws_short * args.block_size, args.block_size)
        block_masks = [None, short_bm, short_bm, short_bm, long_bm, short_bm, short_bm, None, short_bm, short_bm, short_bm, long_bm]
        assert len(block_masks) == len(self.blocks)
        # =================================================================================

        x = self.embed(input_seq)

        # smear token embed forward 1 position @classiclarryd
        smear_lambda = self.scalars[5 * len(self.blocks)]
        smear_gate_out = smear_lambda * torch.sigmoid(self.smear_gate(x[1:, :self.smear_gate.weight.size(-1)]))
        x = torch.cat([x[:1], x[1:] + smear_gate_out * x[:-1]])
        x = x0 = norm(x[None])

        # U-net design by @brendanh0gan
        skip_connections = []
        skip_weights = self.scalars[:(len(self.blocks) // 2)]
        lambdas = self.scalars[1 * len(self.blocks): 3 * len(self.blocks)].view(-1, 2)
        sa_lambdas = self.scalars[3 * len(self.blocks): 5 * len(self.blocks)].view(-1, 2)
        backout_lambda = self.scalars[5 * len(self.blocks)+1]

        n = len(self.blocks) // 2

        x_backout = None
        # skip layer zero
        for i in range(1,len(self.blocks)):
            attn_args = AttnArgs(
                ve=ve[i],
                sa_lambdas=sa_lambdas[i],
                block_mask=block_masks[i],   # [A100 DIFF #2] FlexAttention BlockMask (was seqlens+bm_size)
                cos=self.yarn.cos,
                sin=self.yarn.sin,
                attn_scale=self.yarn.attn_scale
            )
            if i >= n and i<11:
                gate = torch.sigmoid(skip_weights[i - n])  # in (0, 1)
                x = x + gate * skip_connections.pop()
            # x_out = self.blocks[i](x, x0, lambdas[i], attn_args)
            # x_backout += backout_lambdas[i] * (x_out-x)
            # x = x_out
            x = self.blocks[i](x, x0, lambdas[i], attn_args)
            if i < n:
                skip_connections.append(x)
            if i==8:
                x_backout=x

        # backout contributions from first 8 layers that are only required for downstream context and not direct prediction
        x -= backout_lambda*x_backout
        x = norm(x)
        logits = self.lm_head(x)
        # @Grad62304977 added tanh softcapping following Gemma 2 paper, @KoszarskyB reduced it from 30 to 15, @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1)
        logits = 30 * torch.sigmoid(logits / 7.5)
        logits_for_loss = logits.float() if not self.training else logits
        loss = F.cross_entropy(
            logits_for_loss.view(-1, logits_for_loss.size(-1)),
            target_seq,
            reduction="sum" if self.training else "mean",
        )
        return loss

# -----------------------------------------------------------------------------
# Distributed data loader

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

BOS_ID = 50256

class BOSFinder:
    # Helper for getting sequences that start at the beginning of documents by @varunneal based on work by @classiclarryd
    def __init__(self, tokens: Tensor, world_size: int = 1, quickload: bool = False):
        # Precompute BOS positions once per shard
        self.tokens=tokens
        self.size = tokens.numel()
        self.quickload = quickload
        if quickload:
            # only scan first 4 million tokens, then kickoff async thread to scan rest
            self.bos_idx = (tokens[:4_000_000] == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
            self.thread = None
            self.ready = threading.Event()
            self.start()
        else:
            self.bos_idx = (tokens == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        self.i = 0
        self.world_size = world_size
        self.batch_iter = 0

    def _load(self):
        self.bos_idx_async = (self.tokens == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        self.ready.set()
    
    def start(self):
        self.ready.clear()
        self.thread = threading.Thread(target=self._load)
        self.thread.start()
    
    def get(self):
        if self.thread:
            self.ready.wait()
            self.thread.join()
        self.bos_idx = self.bos_idx_async

    def next_batch(self, num_tokens_local: int, max_seq_len: int):
        # if quickload was used, repoint to the full dataset after 5 batches
        if self.quickload and self.batch_iter==5:
            self.get()
        n = len(self.bos_idx)
        starts = [[] for _ in range(self.world_size)]
        ends = [[] for _ in range(self.world_size)]

        idx = self.i
        for r in range(self.world_size):
            cur_len = 0
            while cur_len <= num_tokens_local:
                if idx >= n:
                    raise StopIteration(f"Insufficient BOS ahead of position {cur}; hit tail of shard.")
                cur = self.bos_idx[idx]
                starts[r].append(cur)
                end = min(self.bos_idx[idx + 1] if idx + 1 < n else self.size,
                          cur + max_seq_len,
                          cur + num_tokens_local - cur_len + 1)
                ends[r].append(end)
                cur_len += end - cur
                idx += 1

            assert cur_len == num_tokens_local + 1
        self.i = idx
        self.batch_iter+=1
        return starts, ends

class DataPreloader:
    # Helper for asynchronously loading next shard and indexing bos tokens
    def __init__(self, file_iter, world_size: int = 1):
        self.file_iter = file_iter
        self.world_size = world_size
        self.thread = None
        self.data = None
        self.ready = threading.Event()
    
    def _load(self):
        tokens = _load_data_shard(next(self.file_iter))
        self.data = (tokens, BOSFinder(tokens, self.world_size))
        self.ready.set()
    
    def start(self):
        self.ready.clear()
        self.thread = threading.Thread(target=self._load)
        self.thread.start()
    
    def get(self):
        if self.thread:
            self.ready.wait()
            self.thread.join()
        return self.data

def distributed_data_generator(filename_pattern: str, num_tokens: int, max_seq_len: int, grad_accum_steps: int = 1, align_to_bos: bool = True):
    # align_to_bos: each sequence begins with Beginning of Sequence token, sequences truncated to max_seq_len
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    assert num_tokens % (world_size * grad_accum_steps) == 0, "Batch size must be divisible by world size"
    num_tokens = num_tokens // grad_accum_steps

    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {filename_pattern}")

    file_iter = iter(files)  # Use itertools.cycle(files) for multi-epoch training
    tokens = _load_data_shard(next(file_iter))
    if align_to_bos:
        finder = BOSFinder(tokens, world_size=world_size, quickload=True)
        preloader = DataPreloader(file_iter, world_size)
        preloader.start()
    else:
        pos = 0  # for unaligned case

    while True:
        num_tokens_local = num_tokens // world_size
        max_num_docs = next_multiple_of_n(num_tokens_local // 300, n=128)  # median doc length is ~400

        if align_to_bos:
            try:
                seq_starts, seq_ends = finder.next_batch(num_tokens_local, max_seq_len)
                start_idxs, end_idxs = torch.tensor(seq_starts[rank]), torch.tensor(seq_ends[rank])
            except StopIteration:
                # This shard is exhausted, load the next one in the next loop iteration.
                tokens, finder = preloader.get()
                preloader.start()
                continue

            buf = torch.cat([tokens[i:j] for i, j in zip(start_idxs, end_idxs)])
            _inputs = buf[:-1]
            _targets = buf[1:]
            end_idxs[-1] -= 1  # last document was too long to account for _targets offset
            cum_lengths = (end_idxs - start_idxs).cumsum(0)

        else:
            if pos + num_tokens + 1 >= len(tokens):  # should not occur for val data
                tokens, pos = _load_data_shard(next(file_iter)), 0

            pos_local = pos + rank * num_tokens_local
            buf = tokens[pos_local: pos_local + num_tokens_local + 1]
            _inputs = buf[:-1].view(num_tokens_local, )
            _targets = buf[1:].view(num_tokens_local, )

            cum_lengths = torch.nonzero(_inputs == BOS_ID)[:, 0]
            pos += num_tokens


        _cum_lengths = torch.full((max_num_docs,), num_tokens_local)
        _cum_lengths[0] = 0
        _cum_lengths[1:len(cum_lengths) + 1] = cum_lengths

        new_params = yield (
            _inputs.to(device="cuda", dtype=torch.int32, non_blocking=True),
            _targets.to(device="cuda", dtype=torch.int64, non_blocking=True),
            _cum_lengths.to(device="cuda", dtype=torch.int32, non_blocking=True)
        )

        if new_params is not None:
            # makes it possible for generator to receive new (num_tokens, max_seq_len, grad_accum_steps) via .send()
            new_num_tokens, new_max_seq_len, new_grad_accum_steps = new_params
            assert new_num_tokens % (world_size * grad_accum_steps) == 0, "Num tokens must be divisible by world size"
            num_tokens = new_num_tokens
            max_seq_len = new_max_seq_len
            grad_accum_steps = new_grad_accum_steps


# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data
    train_files: str = "data/fineweb10B/fineweb_train_*.bin" # input .bin to train on
    val_files: str = "data/fineweb10B/fineweb_val_*.bin" # input .bin to eval validation loss on
    val_tokens: int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    train_batch_size: int = 2048 * 16 * 8
    train_max_seq_len: int = 128 * 16
    val_batch_size: int = 4 * 64 * 1024 * 8
    # optimization
    num_iterations: int = 2290  # number of iterations to run
    iteration_extension = 40  # number of iterations to continue training at final cooldown and window size
    cooldown_frac: int = 0.45  # fraction of training spent cooling down the learning rate
    momentum_cd_steps = 50  # number of iterations for muon momentum cooldown
    # evaluation and logging
    run_id: str = ""  # filled in below: "<opt>_lr<lr>_<short uuid>"
    val_loss_every: int = 250  # every how many steps to evaluate val loss? 0 for only at the end
    save_checkpoint: bool = False
    # attention masking
    block_size: int = 128
    ws_schedule: tuple = (3, 7, 11)
    ws_validate: int = 13 # increase final validation ws, used for YaRN extension and short window size @classiclarryd
    ws_long_validate: int = 20 # extend long windows out even further

args = Hyperparameters()

data_path = os.environ.get("DATA_PATH", ".")
args.train_files = os.path.join(data_path, args.train_files)
args.val_files = os.path.join(data_path, args.val_files)

# -----------------------------------------------------------------------------
# Which of the paper's methods drives the hidden matrices, and with what
# hyperparameters.  Resolved BEFORE logging starts so the run id, and therefore
# the log filename, names the experiment.
#
# Record #40 drives (hidden_matrix_params + gate_params) with a single Muon
# (lr=0.06, momentum=0.95, weight_decay=0.0 -- no cautious WD yet at #40). That
# grouping is kept exactly; only the *method* changes.
#
# WHY THESE LEARNING RATES
# ------------------------
# The per-layer multipliers in signmuon_optimizers.py (rule "unit-gain") make
# eta_0 mean ONE thing for every method: the per-step RMS gain of the update. For
# the LMO family the multiplier IS record #40's aspect factor, so `Muon` at 0.06
# is the record verbatim, and eta_0 is then transferable to the other seven.
#
# The LMO five: 0.06, i.e. exactly the reference's.  This is not a guess, it is a
# deliberate design choice -- their final step is an *orthogonal* matrix (or, for
# the EF21 pair, an error-feedback estimate of one), so it has the same spectral
# and Frobenius norm as Muon's step and there is nothing to rescale.  Keeping them
# AT the reference's LR is also what makes the paper's contrasts clean: each of
#   Muon vs MuonUSign            (what does 1-bit uplink cost?)
#   EF21-SignMuon vs EF21-MuonUSign / EF21-MuonSign  (Thm 4: EF21 on the LMO
#                                 OUTPUT diverges, EF21 on the momentum does not)
# is then a matched-hyperparameter comparison, differing only in the update rule.
# In particular EF21-SignMuon belongs HERE, not with SignMuon: error feedback is
# precisely what undoes the 1-bit quantization -- `d_est` is a full-precision
# accumulator tracking PE(M), so the step regains the LMO's magnitude (its
# op-norm starts ~1.1x Muon's and decays toward 1.0 as d_est tracks D). Putting
# it at Muon's own LR is the only way its divergence can be read as the rule's
# fault rather than the step size's.
#
# The sign three: 0.03.  Their step is entrywise uniform, so at equal RMS gain it
# is spectrally more aggressive than an orthogonal step -- and the smoothness
# framework these methods are analysed in (Gluon / EF21-Muon) is a SPECTRAL-norm
# framework, so that is the norm that should be matched, not the Frobenius one.
# Three independent routes agree on the discount:
#
#   (a) spectral matching.  ||lambda*sign(.)||_op / ||lambda*PE(.)||_op
#       = 0.93(sqrt m + sqrt n)/sqrt n = 1.40 (mlp [768,3072]), 1.86 (attn
#       [768,768]).  One eta_0 must satisfy the tighter one: 0.06/1.86 = 0.032.
#
#   (b) Mishra et al.'s tuned value, mapped in.  Their Sign-Muon Algorithm 1 has
#       NO shape factor (line 9 is W <- W - eta*sign(U)), so their nanoGPT sweep
#       over {1e-1..1e-5} picking eta=1e-3 for BOTH SignMuon and signSGD is a
#       global unscaled LR on a d=384 model.  Dividing by our lambda gives
#       eta_0 ~ 0.023; correcting for their broken schedule (warmup_iters=2000 >
#       max_iters=1500, so their LR only ever ramps 0 -> 7.5e-4) and for our 8.5x
#       larger batch gives ~0.032.
#
#   (c) Lion's "3-10x smaller than AdamW" rule of thumb, decomposed.  AdamW's
#       m/sqrt(v) has per-entry magnitude ~0.3 against a sign step's 1.0, so ~3x
#       of that discount is pure norm -- which unit-gain already handles exactly.
#       The residual robustness discount is 1-3x: eta_0 = 0.02 .. 0.06.
#
# 0.032 -> 0.03, the round number, and the conservative end of (a).
#
# Sanity check against the alternative reading of "sign methods want 1e-4": that
# figure is a GLOBAL, unscaled LR from standard-batch, long-schedule training.
# Per weight entry, 0.03 here means 5.4e-4 (mlp) to 1.1e-3 (attn) -- i.e. HALF of
# what Muon itself takes at the record's 0.06.  If 1e-4 per entry were right for
# this model, Muon at 0.06 would be ~10x too large too, and it is the record.
# This codebase simply operates far more aggressively than standard GPT-2
# training: 2330 steps, 262k tokens/step, Muon at 0.06 (vs ~0.02 typical), Adam at
# 0.008 with lr_mul=75 on the embeddings.
#
# CONFIDENCE.  The LMO five are pinned by the record and are not a guess. The
# sign three are the uncertain number; the evidence brackets 0.01-0.04. If you can
# afford three more runs, probe the downside:  SIGN_PROBE_LR=0.01 bash run_all.sh
#
# All values are "round" (one significant digit); the tuning ladder that contains
# both of them is 0.01, 0.02, 0.03, 0.06, 0.1, 0.2.
_ANCHOR_LR = 0.06        # record #40's Muon learning rate -- the reference, not a guess
OPTIMIZER_CONFIG = {
    # --- LMO-terminated: step is polar(.) (or an EF21 estimate of it), op-norm 1 ---
    "Muon":           dict(lr=_ANCHOR_LR, momentum=0.95, weight_decay=0.0),  # == record #40
    "MuonUSign":      dict(lr=_ANCHOR_LR, momentum=0.95, weight_decay=0.0),
    "EF21-MuonUSign": dict(lr=_ANCHOR_LR, momentum=0.95, weight_decay=0.0),
    "EF21-MuonSign":  dict(lr=_ANCHOR_LR, momentum=0.95, weight_decay=0.0),
    "EF21-SignMuon":  dict(lr=_ANCHOR_LR, momentum=0.95, weight_decay=0.0),
    # --- sign-terminated: entrywise-uniform step, 1.4-1.9x the spectral norm ---
    "SignMuon":       dict(lr=0.03,       momentum=0.95, weight_decay=0.0),
    "MuonSign":       dict(lr=0.03,       momentum=0.95, weight_decay=0.0),
    "SignSGD":        dict(lr=0.03,       momentum=0.95, weight_decay=0.0),
}
opt_name = os.environ.get("SIGNMUON_OPT", "Muon")
assert opt_name in OPTIMIZERS, f"unknown SIGNMUON_OPT={opt_name!r}; choose from {list(OPTIMIZERS)}"
opt_cfg = dict(OPTIMIZER_CONFIG[opt_name])
# optional env overrides for hyperparameter sweeps
if "SIGNMUON_LR" in os.environ:        opt_cfg["lr"] = float(os.environ["SIGNMUON_LR"])
if "SIGNMUON_MOMENTUM" in os.environ:  opt_cfg["momentum"] = float(os.environ["SIGNMUON_MOMENTUM"])
if "SIGNMUON_WD" in os.environ:        opt_cfg["weight_decay"] = float(os.environ["SIGNMUON_WD"])
opt_cfg["lr_scaling"] = os.environ.get("SIGNMUON_LR_SCALING", "unit-gain")
assert opt_cfg["lr_scaling"] in LR_SCALING_RULES, (
    f"unknown SIGNMUON_LR_SCALING={opt_cfg['lr_scaling']!r}; "
    f"choose from {sorted(LR_SCALING_RULES)}")
# shorten the run for smoke tests / cheap LR probes (full length by default)
if "NANOGPT_ITERS" in os.environ:
    args.num_iterations = int(os.environ["NANOGPT_ITERS"])
if "NANOGPT_VAL_EVERY" in os.environ:
    args.val_loss_every = int(os.environ["NANOGPT_VAL_EVERY"])
# Seed. Upstream modded-nanogpt seeds nothing at all, so every record run starts
# from a different initialization and reports a five-seed spread instead; we keep
# the loop identical but pin the draw, so a re-run of one method reproduces its
# own curve. Seeding the RNG is all that is pinned: cuDNN/cuBLAS autotuning and
# the reduce_scatter reduction order stay as they are, because forcing
# deterministic kernels would change the wall-clock that Table `tab:nanogpt`
# reports. Every rank takes the SAME seed -- initialization is broadcast from
# rank 0 anyway, and the only other RNG consumer is `sign_pm1` on the rank that
# owns the parameter.
seed = int(os.environ.get("SIGNMUON_SEED", 0))
muon_momentum_target = opt_cfg["momentum"]  # final value of the momentum warmup/cooldown

# Self-describing run id: the log filename alone identifies the experiment.
args.run_id = os.environ.get(
    "SIGNMUON_RUN_ID",
    f"{opt_name}_lr{opt_cfg['lr']:g}_{uuid.uuid4().hex[:8]}")

# torchrun sets these env variables
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
assert 8 % world_size == 0, "world_size must be a divisor of 8"
grad_accum_steps = 8 // world_size
assert torch.cuda.is_available()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
master_process = (rank == 0) # this process will do logging, checkpointing etc.

# begin logging
logfile = None
run_id = args.run_id
if master_process:
    logfile = os.path.join(
        os.environ.get("LOG_DIR", os.path.join("..", "results", "nanogpt", "logs")),
        f"{run_id}.txt")
    # `run_id` may contain a directory component, so create the *parent of the
    # logfile*, not just "logs" (record #40's default id was "new/<uuid>", which
    # crashed here on a fresh checkout because logs/new/ never got created).
    os.makedirs(os.path.dirname(logfile) or ".", exist_ok=True)
    print(logfile)
def print0(s, console=False):
    if master_process:
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)

# begin by printing this file (the Python code) plus the optimizer module, so a logged run
# fully reproduces the optimizer definitions even though they are imported.
print0(code)
print0("="*100)
import signmuon_optimizers as _smo  # noqa: E402
print0(Path(_smo.__file__).read_text())
print0("="*100)
# log information about the hardware/software environment this is running on
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")

def nvidia_smi():
    import subprocess  # avoid top level import
    return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
print0(nvidia_smi())
print0("="*100)

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
model: nn.Module = GPT(
    vocab_size=50257,
    num_layers=12,
    num_heads=6,
    head_dim=128,
    model_dim=768,
    max_seq_len=max(args.train_batch_size, args.val_batch_size) // (grad_accum_steps * world_size)
).cuda()
for m in model.modules():
    if isinstance(m, (nn.Embedding, nn.Linear)):
        m.bfloat16()
for param in model.parameters():
    dist.broadcast(param.detach(), 0)

# collect the parameters to optimize
hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n and "gate" not in n]
embed_params = [p for n, p in model.named_parameters() if "embed" in n]
scalar_params = [p for p in model.parameters() if p.ndim < 2]
head_params = [model.lm_head.weight]
gate_params = [p for n, p in model.named_parameters() if "gate" in n]

# init the optimizer(s)
# small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
# discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
optimizer1 = DistAdam(
    scalar_params + head_params + embed_params,
    lr=0.008,
    betas=(0.65, 0.95),
    eps=1e-8,
    weight_decay=0.0,
)
# --- Hidden-matrix optimizer: one of the paper's methods (see signmuon_optimizers.py) ---
# The method and its hyperparameters were resolved above (before logging started, so the
# run id names the experiment); OPTIMIZER_CONFIG documents why each learning rate is what
# it is.  `Muon` here is record #40 verbatim.
optimizer2 = OPTIMIZERS[opt_name](hidden_matrix_params + gate_params, **opt_cfg)
optimizers = [optimizer1, optimizer2]

# --- machine-readable run header (one JSON line; parse_logs.py reads this) ----
_param_names = {id(p): n for n, p in model.named_parameters()}
print0("RUNMETA " + json.dumps(dict(
    run_id=run_id,
    script=os.path.basename(__file__),
    optimizer=opt_name,
    family=type(optimizer2).family,
    lr=opt_cfg["lr"],
    momentum=opt_cfg["momentum"],
    weight_decay=opt_cfg["weight_decay"],
    lr_scaling=opt_cfg["lr_scaling"],
    seed=seed,
    adam_lr=optimizer1.param_groups[0]["lr"],
    num_iterations=args.num_iterations,
    iteration_extension=args.iteration_extension,
    train_steps=args.num_iterations + args.iteration_extension,
    val_loss_every=args.val_loss_every,
    world_size=world_size,
    grad_accum_steps=grad_accum_steps,
    train_batch_size=args.train_batch_size,
    tokens_per_step=args.train_batch_size,
    torch=torch.version.__version__,
)), console=True)
print0(f"hidden-matrix optimizer: {opt_name}  config={opt_cfg}", console=True)
print0(describe_lr_scaling(optimizer2, _param_names))
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

# learning rate schedule: stable then decay
def get_lr(step: int):
    x = min(0.9999,step / args.num_iterations)
    assert 0 <= x < 1
    lr = 1.0
    if x >= 1 - args.cooldown_frac:
        w = (1 - x) / args.cooldown_frac
        lr = w * 1.0 + (1 - w) * 0.1
    return lr

def get_ws(step: int):
    if step == args.num_iterations+args.iteration_extension:
        return args.ws_validate//2, args.ws_validate
    x = min(step / (1 + args.num_iterations),0.9999)
    assert 0 <= x < 1
    ws_idx = int(len(args.ws_schedule) * x)
    return args.ws_schedule[ws_idx]//2, args.ws_schedule[ws_idx]

def update_optimizer_params(step, optimizer1, optimizer2):
    # Update lr
    for group in optimizer1.param_groups:
        group["lr"] = group["initial_lr"] * get_lr(step)
    for group in optimizer2.param_groups:
        group["lr"] = group["initial_lr"] * get_lr(step)

    # Warmup phase: gradually increase momentum from 0.85 to the target (0.95 for #40's Muon;
    # tracks SIGNMUON_MOMENTUM so a sweep override actually takes effect past step 300).
    if step < 300:
        frac = step / 300
        momentum = 0.85 + frac * (muon_momentum_target - 0.85)
        for group in optimizer2.param_groups:
            group["momentum"] = momentum

    # Cooldown phase: gradually decrease momentum back to 0.85
    momentum_cd_start = args.num_iterations + args.iteration_extension - args.momentum_cd_steps
    if step > momentum_cd_start:
        frac = (step - momentum_cd_start) / args.momentum_cd_steps

        # Decay momentum from the target to 0.85
        momentum = muon_momentum_target - frac * (muon_momentum_target - 0.85)
        for group in optimizer2.param_groups:
            group["momentum"] = momentum

    # One instrumented optimizer step per validation. The diagnostics (compressor
    # contraction, estimator lag, per-block gradient magnitudes -- see DIAG_SLOTS
    # in signmuon_optimizers.py) cost a couple of extra reductions per parameter,
    # which is nothing next to the LMO's matmuls but is not free in kernel
    # launches. Enabling them only on the step whose numbers get logged keeps the
    # clock comparable across all eight arms.
    if hasattr(optimizer2, "diagnostics"):
        nxt = step + 1                     # validation happens at the TOP of `nxt`
        optimizer2.diagnostics = bool(
            nxt == train_steps
            or (args.val_loss_every > 0 and nxt % args.val_loss_every == 0))

# [A100 DIFF #2] fullgraph=True -> False: build_flex_block_mask is @torch.compiler.disable
# (it runs eagerly to construct the BlockMask), which introduces a graph break, so we cannot
# require a single full graph. dynamic=False is kept (shapes are static per T), and
# flex_attention itself still compiles.
model: nn.Module = torch.compile(model, dynamic=False)

########################################
#            Warmup kernels            #
########################################

# Warmup the training kernels, then re-initialize the state so we aren't cheating
warmup_steps = 30
initial_state = dict(model=copy.deepcopy(model.state_dict()),
                     optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers]) # save the initial state
train_loader = distributed_data_generator(args.train_files, args.train_batch_size, args.train_max_seq_len, grad_accum_steps=grad_accum_steps)
ws_long = args.ws_schedule[0]
for step in range(warmup_steps):
    inputs, targets, cum_seqlens = next(train_loader)
    new_ws_long = args.ws_schedule[step % len(args.ws_schedule)]  # each window size is a new graph, need to warm up each with YaRN params
    if new_ws_long > ws_long:
        model.yarn.apply(ws_long, new_ws_long)
        ws_long = new_ws_long
    elif new_ws_long<ws_long:
        model.yarn.reset()
        ws_long = new_ws_long
    model(inputs, targets, cum_seqlens, ws_long//2, ws_long).backward()
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
model.yarn.reset()
model.load_state_dict(initial_state["model"])
for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
    opt.load_state_dict(opt_state)
del train_loader, initial_state

########################################
#        Training and validation       #
########################################

train_loader = distributed_data_generator(args.train_files, args.train_batch_size, args.train_max_seq_len, grad_accum_steps=grad_accum_steps)
training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.perf_counter()
# begin training
train_steps = args.num_iterations + args.iteration_extension

# --- per-step train loss, logged for free -------------------------------------
# Record #40 logs only the 10 validation points, which is too coarse to compare
# eight optimizers on. The training loss is already computed every step, so park
# it in a GPU buffer (a device-to-device copy: NO host sync, NO collective, so the
# hot loop and its wall-clock are unperturbed) and read the buffer out only inside
# the validation block, where the clock is stopped anyway. One all_reduce per
# validation turns the rank-local losses into the global mean.
train_loss_buf = torch.zeros(train_steps, device=device)
train_loss_flushed = 0

def flush_train_losses(upto: int) -> bool:
    """Emit the buffered per-step train losses in ``[flushed, upto)``.

    Returns True if any of them is non-finite. Every rank runs the same
    all_reduce and sees the same values, so a divergence abort stays collective.
    """
    global train_loss_flushed
    if upto <= train_loss_flushed:
        return False
    chunk = train_loss_buf[train_loss_flushed:upto].clone()
    dist.all_reduce(chunk, op=dist.ReduceOp.AVG)
    values = chunk.tolist()
    for i, v in enumerate(values):
        print0(f"step:{train_loss_flushed + i} train_loss:{v:.6f}")
    train_loss_flushed = upto
    return not all(math.isfinite(v) for v in values)

diverged = False
ws_short, ws_long = get_ws(0)
for step in range(train_steps + 1):
    last_step = (step == train_steps)
    ws_short, new_ws_long = get_ws(step)
    if new_ws_long != ws_long:
        model.yarn.apply(ws_long, new_ws_long)
        ws_long=new_ws_long

    # --------------- VALIDATION SECTION -----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        if last_step:
            ws_long = args.ws_long_validate
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        diverged |= flush_train_losses(step)
        model.eval()
        assert args.val_tokens % args.val_batch_size == 0
        val_steps = grad_accum_steps * args.val_tokens // args.val_batch_size

        def evaluate():
            """Mean validation loss of the CURRENT parameters, averaged over ranks.

            A fresh loader is deterministic -- the val split is read unaligned from
            position 0 of the first sorted shard -- so every call within one
            validation sees the identical tokens. That is what makes the X-vs-W
            comparison below a comparison of MODELS rather than of batches.
            """
            loader = distributed_data_generator(args.val_files, args.val_batch_size, -1,
                                                grad_accum_steps=grad_accum_steps,
                                                align_to_bos=False)
            total = 0
            with torch.no_grad():
                for _ in range(val_steps):
                    inputs, targets, cum_seqlens = next(loader)
                    total += model(inputs, targets, cum_seqlens, ws_short, ws_long)
            del loader
            total /= val_steps
            dist.all_reduce(total, op=dist.ReduceOp.AVG)
            return total

        # EF21-MuonSign trains a sign-compressed broadcast model W but tracks an
        # exact server model X. Report BOTH: X is primary (it is the iterate the
        # convergence corollary bounds, and it keeps this column comparable with
        # the other seven arms), W is the iterate every gradient was actually
        # evaluated at. The EF21-P downlink's contraction is only alpha ~ 1/d, so
        # the two can separate by a lot, and without both numbers a reader cannot
        # tell a failure of the METHOD from a failure of the X-tracking.
        # No-op for every other method, where X and W are the same tensor.
        eval_on_exact = hasattr(optimizer2, "swap_in_exact")
        if eval_on_exact:
            optimizer2.swap_in_exact()
        val_loss = evaluate()                       # X (primary)
        val_loss_w = None
        if eval_on_exact:
            optimizer2.swap_out_exact()
            val_loss_w = evaluate()                 # W (broadcast model)
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f}"
               + (f" val_loss_W:{val_loss_w:.4f}" if val_loss_w is not None else "")
               + f" train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms",
               console=True)
        if step > 0 and hasattr(optimizer2, "diagnostics_report"):
            report = optimizer2.diagnostics_report(_param_names)
            if report:
                print0(report)
        # Several of the paper's methods are proved to diverge (Thms 1-4) and this
        # is a paid GPU: once the loss is NaN/Inf nothing more is learned, so stop.
        # The decision is identical on every rank (every quantity is all_reduced).
        diverged |= not math.isfinite(val_loss.item())
        if val_loss_w is not None:
            diverged |= not math.isfinite(val_loss_w.item())
        if diverged:
            print0(f"DIVERGED step:{step}/{train_steps} -- non-finite loss, aborting run",
                   console=True)
            break
        model.train()
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if master_process and args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            os.makedirs(f"logs/{run_id}", exist_ok=True)
            torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
        # the last step only has the validation loop, so break to avoid training
        break

    # --------------- TRAINING SECTION -----------------
    for _ in range(grad_accum_steps):
        inputs, targets, cum_seqlens = next(train_loader)
        loss = model(inputs, targets, cum_seqlens, ws_short, ws_long)
        loss.backward()
        # device-to-device accumulate; never read on the host until the flush
        train_loss_buf[step] += loss.detach() / grad_accum_steps
    update_optimizer_params(step, optimizer1, optimizer2)
    # only step Adam every other step
    if step%2==0:
        optimizer2.step()
        optimizer2.zero_grad(set_to_none=True)
    else:
        for opt in optimizers:
            opt.step()
        # null the gradients
        model.zero_grad(set_to_none=True)
    
    # logging
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
print0("RUNEND " + json.dumps(dict(
    run_id=run_id,
    optimizer=opt_name,
    lr=opt_cfg["lr"],
    diverged=bool(diverged),
    last_step=int(step),
    train_time_ms=round(training_time_ms, 1),
    peak_memory_mib=torch.cuda.max_memory_allocated() // 1024 // 1024,
)), console=True)
dist.destroy_process_group()
