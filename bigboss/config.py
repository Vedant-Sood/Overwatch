"""Configuration loading.

The default template lives in configs/bigboss.toml. A local override file (BIGBOSS_CONFIG env var,
or configs/local.toml if present) is deep-merged on top of it.
"""

from __future__ import annotations

import copy
import os
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "bigboss.toml"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class Config:
    """Thin wrapper over the merged TOML dict with dotted-path access."""

    def __init__(self, raw: dict[str, Any], root: Path = REPO_ROOT):
        self.raw = raw
        self.root = root

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.raw.get(name, {}))

    @property
    def data_dir(self) -> Path:
        d = Path(os.environ.get("BIGBOSS_DATA_DIR", self.get("paths.data_dir", "data")))
        if not d.is_absolute():
            d = self.root / d
        d.mkdir(parents=True, exist_ok=True)
        return d

    def universe(self, groups: list[str] | None = None) -> list[str]:
        uni = self.section("universe")
        groups = groups or list(uni)
        seen: list[str] = []
        for g in groups:
            for s in uni.get(g, []):
                if s not in seen:
                    seen.append(s)
        return seen

    def benchmark_for(self, symbol: str) -> str:
        b = self.section("benchmarks")
        return b.get(market_of(symbol), b.get("default", "SPY"))

    def with_overrides(self, override: dict) -> "Config":
        return Config(_deep_merge(self.raw, override), self.root)


def market_of(symbol: str) -> str:
    s = symbol.upper()
    if s.endswith(".NS") or s.endswith(".BO") or s == "^NSEI":
        return "IN"
    if s.startswith("^"):
        return "INDEX"
    return "US"


def load_config(path: str | os.PathLike | None = None) -> Config:
    with open(DEFAULT_CONFIG, "rb") as f:
        raw = tomllib.load(f)
    override_path = path or os.environ.get("BIGBOSS_CONFIG")
    if override_path is None and (REPO_ROOT / "configs" / "local.toml").exists():
        override_path = REPO_ROOT / "configs" / "local.toml"
    if override_path:
        with open(override_path, "rb") as f:
            raw = _deep_merge(raw, tomllib.load(f))
    return Config(raw)


@lru_cache(maxsize=1)
def get_config() -> Config:
    return load_config()
