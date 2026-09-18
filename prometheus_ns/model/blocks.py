"""Transformer building blocks: RMSNorm, SwiGLU, RoPE and GQA attention.

The attention implementation is a standard causal decoder block with
grouped-query attention (``n_kv_head`` <= ``n_head``). Rotary
embeddings are precomputed once per ``max_seq`` and stored as a cache
by the caller. An optional document-boundary mask enables block
diagonal attention over packed windows so cross-document attention
bleed can be suppressed during training.
"""

from __future__ import annotations

import math

import torch
from torch import nn

try:  # Optional accelerator path; the fallback below is always valid.
    from flash_attn import flash_attn_func  # type: ignore

    FLASH_ATTN_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on GPUs with flash-attn
    flash_attn_func = None
    FLASH_ATTN_AVAILABLE = False


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization without mean subtraction."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


def precompute_rope_cache(max_seq: int, head_dim: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the ``(cos, sin)`` rotary tables of shape ``[max_seq, head_dim/2]``."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to ``x`` of shape ``[B, H, T, D]``.

    The tables are indexed as ``cos[:T]`` and broadcast over batch and
    heads. Half-split convention: the first half of each head rotates
    against the second half.
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    cos = cos[None, None, : x.shape[-2], :].to(x.dtype)
    sin = sin[None, None, : x.shape[-2], :].to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class SwiGLU(nn.Module):
    """Gated feed-forward network with three matrices: gate, up, down."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        # Residual output projection (variance-scaled init).
        self.w_down._is_residual = True  # type: ignore[attr-defined]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(torch.nn.functional.silu(self.w_gate(x)) * self.w_up(x))


class CausalSelfAttention(nn.Module):
    """Multi-head causal attention with grouped-query heads and RoPE.

    Args:
        d_model: Model width.
        n_head: Number of query heads.
        n_kv_head: Number of key/value heads (GQA when < ``n_head``).
        max_seq: Context length; the causal mask is cached at this size.
    """

    def __init__(self, d_model: int, n_head: int, n_kv_head: int, max_seq: int) -> None:
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if n_kv_head > n_head or n_head % n_kv_head != 0:
            raise ValueError("n_kv_head must divide n_head and not exceed it")
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = d_model // n_head
        self.wq = nn.Linear(d_model, n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_head * self.head_dim, bias=False)
        self.wo = nn.Linear(n_head * self.head_dim, d_model, bias=False)
        # Residual output projection: initialized with a variance-scaled
        # std (see TransformerLM._init_weights) to keep the residual
        # stream stable at depth.
        self.wo._is_residual = True  # type: ignore[attr-defined]
        mask = torch.ones(max_seq, max_seq, dtype=torch.bool).tril()
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        doc_flags: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend over ``x`` of shape ``[B, T, D]``.

        Args:
            x: Input activations.
            rope_cos, rope_sin: Rotary tables from ``precompute_rope_cache``.
            doc_flags: Optional ``[B, T]`` tensor where 1 marks the first
                token of a document in a packed window; attention is then
                restricted to same-document positions (block diagonal).
        """
        bsz, seq, _ = x.shape
        q = self.wq(x).view(bsz, seq, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, seq, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bsz, seq, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        rep = self.n_head // self.n_kv_head
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        allowed = self.causal_mask[:seq, :seq]
        if doc_flags is not None:
            doc_id = doc_flags.cumsum(dim=-1)  # [B, T]; same id = same document
            same_doc = doc_id[:, None, :] == doc_id[:, :, None]  # [B, T, T]
            # [B, 1, T, T] broadcasts over the head dimension.
            allowed = allowed[None, None, :, :] & same_doc[:, None, :, :]

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~allowed, float("-inf"))
        attn = torch.softmax(scores.float(), dim=-1).to(v.dtype)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(bsz, seq, -1)
        return self.wo(out)


class TransformerBlock(nn.Module):
    """Pre-norm residual block: attention and SwiGLU feed-forward."""

    def __init__(self, d_model: int, n_head: int, n_kv_head: int, d_ff: int, max_seq: int) -> None:
        super().__init__()
        self.norm_attn = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, n_kv_head, max_seq)
        self.norm_ffn = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        doc_flags: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x), rope_cos, rope_sin, doc_flags)
        x = x + self.ffn(self.norm_ffn(x))
        return x
