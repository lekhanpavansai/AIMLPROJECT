from __future__ import annotations

# pyright: reportMissingImports=false

from functools import lru_cache
from pathlib import Path
from typing import Literal

try:  # pragma: no cover - optional dependency guard for static analysis
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("pydantic-settings is required for configuration management; please install it.") from exc


class RuntimeSettings(BaseSettings):
    """Runtime configuration driven by env vars."""

    model_config = SettingsConfigDict(env_prefix="UNP_", env_file=".env", env_nested_delimiter="__")

    device: str = "cuda"
    log_level: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] = "INFO"
    log_json: bool = False
    outputs_root: Path = Path("outputs")
    enable_prometheus: bool = True
    enable_tracing: bool = False


@lru_cache
def get_settings() -> RuntimeSettings:
    return RuntimeSettings()


