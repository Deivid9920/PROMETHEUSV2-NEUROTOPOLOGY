"""Prometheus-NS: a neuro-symbolic language-modeling system.

The package root hosts only shared configuration utilities. Every
pipeline stage lives in its own subpackage: data, model, train,
symbolic, autoloop and eval.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    """Return the repository root (the parent directory of this package)."""
    return Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the project configuration with ``yaml.safe_load``.

    Args:
        path: Explicit config path. Defaults to ``config.yaml`` at the
            repository root.

    Returns:
        The parsed configuration mapping. The resolved config path is
        stored under the ``_config_path`` key for downstream helpers.

    Raises:
        ValueError: If the YAML root is not a mapping.
    """
    cfg_path = Path(path) if path is not None else repo_root() / "config.yaml"
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"config root must be a mapping: {cfg_path}")
    cfg["_config_path"] = str(cfg_path)
    return cfg


def cfg_get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    """Read a nested config value using dot notation.

    Example:
        ``cfg_get(cfg, "model.profile")`` returns the active profile.
    """
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) when missing and return it as a Path."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def repo_path(cfg: dict[str, Any], *parts: str) -> Path:
    """Resolve a path inside the repository that owns ``cfg``.

    Relative paths in config.yaml are interpreted against the
    repository root so runs work from any working directory.
    """
    root = Path(cfg.get("_config_path", repo_root() / "config.yaml")).resolve().parent
    return root.joinpath(*parts)
