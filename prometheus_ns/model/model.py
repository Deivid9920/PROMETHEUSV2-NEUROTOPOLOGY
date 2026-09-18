"""Decoder-only transformer language model.

Initialization follows normal(0, 0.02) with one deliberate refinement:
residual output projections (attention ``wo`` and SwiGLU ``w_down``)
use a variance-scaled standard deviation of ``0.02 / sqrt(2 * n_layer)``.
SwiGLU gates grow activation variance faster than ReLU/GlU layers, and
without the residual scaling the first training steps can explode to
NaN on CPU, where no loss-scaling machinery exists.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from prometheus_ns.model.blocks import RMSNorm, TransformerBlock, precompute_rope_cache
from prometheus_ns.model.config import ModelConfig

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3


class TransformerLM(nn.Module):
    """Decoder-only autoregressive model with optional tied embeddings."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab, cfg.d_model)
        self.blocks = nn.ModuleList(
            TransformerBlock(cfg.d_model, cfg.n_head, cfg.n_kv_head, cfg.d_ff, cfg.max_seq)
            for _ in range(cfg.n_layer)
        )
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        if cfg.tie_embeddings:
            # The pointer must be shared before counting: two separate
            # matrices would add vocab * d_model parameters and break
            # the contracted parameter ranges.
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = precompute_rope_cache(cfg.max_seq, cfg.head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            if getattr(module, "_is_residual", False):
                std = 0.02 / math.sqrt(2 * self.cfg.n_layer)
            else:
                std = 0.02
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        doc_flags: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run the model over ``input_ids`` of shape ``[B, T]``.

        Args:
            input_ids: Token ids.
            doc_flags: Optional ``[B, T]`` document-boundary marks that
                restrict attention to same-document positions.
            targets: When provided, also return the next-token cross
                entropy loss (padding ids are ignored).

        Returns:
            ``logits`` of shape ``[B, T, vocab]``, or ``(logits, loss)``
            when ``targets`` is given.
        """
        x = self.tok_emb(input_ids)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin, doc_flags)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits
        shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        shift_targets = targets[:, 1:].reshape(-1)
        loss = nn.functional.cross_entropy(shift_logits.float(), shift_targets, ignore_index=PAD_ID)
        return logits, loss

    def num_parameters(self) -> int:
        """Total trainable parameters (tied weights counted once)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
