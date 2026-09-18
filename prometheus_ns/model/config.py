"""Model configuration resolved from config.yaml profiles."""

from __future__ import annotations

from dataclasses import dataclass

from prometheus_ns import cfg_get


@dataclass(frozen=True)
class ModelConfig:
    """Immutable description of one transformer profile.

    ``n_kv_head`` enables grouped-query attention when it is smaller
    than ``n_head``; nano and small use full multi-head attention.
    """

    d_model: int
    n_layer: int
    n_head: int
    n_kv_head: int
    d_ff: int
    max_seq: int
    vocab: int
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.n_head <= 0 or self.n_kv_head <= 0:
            raise ValueError("n_head and n_kv_head must be positive")
        if self.n_kv_head > self.n_head:
            raise ValueError("n_kv_head must be <= n_head (GQA contract)")
        if self.n_head % self.n_kv_head != 0:
            raise ValueError("n_head must be a multiple of n_kv_head")
        if self.d_model % self.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if self.d_ff <= 0 or self.max_seq <= 0 or self.vocab <= 0:
            raise ValueError("d_ff, max_seq and vocab must be positive")

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.d_model // self.n_head


def model_config_from_yaml(cfg: dict, profile: str | None = None) -> ModelConfig:
    """Build a :class:`ModelConfig` from the active profile in config.yaml.

    Args:
        cfg: Parsed project configuration.
        profile: Override for ``model.profile``; must be one of the
            profiles defined in config.yaml.

    Raises:
        ValueError: If the profile is unknown or incomplete.
    """
    name = profile or cfg_get(cfg, "model.profile", "nano")
    values = cfg_get(cfg, f"model.{name}")
    if not isinstance(values, dict):
        raise ValueError(f"unknown model profile: {name}")
    required = ("d_model", "n_layer", "n_head", "n_kv_head", "d_ff", "max_seq", "vocab", "tie_embeddings")
    missing = [k for k in required if k not in values]
    if missing:
        raise ValueError(f"profile '{name}' missing keys: {missing}")
    return ModelConfig(
        d_model=int(values["d_model"]),
        n_layer=int(values["n_layer"]),
        n_head=int(values["n_head"]),
        n_kv_head=int(values["n_kv_head"]),
        d_ff=int(values["d_ff"]),
        max_seq=int(values["max_seq"]),
        vocab=int(values["vocab"]),
        tie_embeddings=bool(values["tie_embeddings"]),
    )


def analytic_param_count(mc: ModelConfig) -> int:
    """Parameter count without materializing the model.

    Counts embeddings (tied matrices counted once), per-layer attention
    projections (GQA-aware), SwiGLU matrices and norms. Matches
    ``sum(p.numel() for p in model.parameters())`` for the tied case.
    """
    d = mc.d_model
    head_dim = mc.head_dim
    emb = mc.vocab * d
    q = d * d
    kv = 2 * (d * mc.n_kv_head * head_dim)
    out = d * d
    attn = q + kv + out
    swiglu = 3 * d * mc.d_ff
    norms = 2 * d
    per_layer = attn + swiglu + norms
    return emb + mc.n_layer * per_layer + d
