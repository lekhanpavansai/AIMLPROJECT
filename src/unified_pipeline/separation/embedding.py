from __future__ import annotations

# pyright: reportMissingImports=false

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

try:  # pragma: no cover - optional dependency guard
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from unified_pipeline.core_types import AudioFrame
from unified_pipeline.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class SpeakerEmbedding:
    vector: np.ndarray
    speaker_id: str


@dataclass(slots=True)
class SpeakerEncoderConfig:
    model_path: Optional[str] = None
    device: str = "cuda"
    embedding_dim: int = 192


class SpeakerEncoder:
    """Wrapper for ERes2NetV2-Large encoder."""

    def __init__(self, config: SpeakerEncoderConfig | None = None) -> None:
        self.config = config or SpeakerEncoderConfig()
        self.embedding_dim = self.config.embedding_dim
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        if not self.config.model_path:
            log.warning("speaker_encoder.model.unconfigured", reason="missing_model_path")
            return

        if torch is None:
            log.warning("speaker_encoder.model.unavailable", reason="torch_not_installed")
            return

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            log.warning("speaker_encoder.model.missing", path=str(model_path))
            return

        log.info("speaker_encoder.model.loading", path=str(model_path))
        try:
            checkpoint = torch.load(model_path, map_location=self.config.device)
            self._model = checkpoint  # Placeholder: replace with actual model
            log.info("speaker_encoder.model.loaded", path=str(model_path))
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("speaker_encoder.model.load_failed", error=str(exc))
            self._model = None

    async def embed(self, frame: AudioFrame, speaker_id: str = "unknown") -> SpeakerEmbedding:
        self._ensure_model()

        if self._model is None:
            log.warning("speaker_encoder.fallback", reason="model_not_available")
            vector = np.random.rand(self.embedding_dim).astype(np.float32)
        else:
            # Placeholder inference logic; replace with actual forward pass
            mono = frame.samples.mean(axis=1)
            vector = np.fft.rfft(mono).real[: self.embedding_dim].astype(np.float32)
            if vector.shape[0] < self.embedding_dim:
                pad = np.zeros(self.embedding_dim - vector.shape[0], dtype=np.float32)
                vector = np.concatenate([vector, pad])
            vector = vector / (np.linalg.norm(vector) + 1e-8)

        return SpeakerEmbedding(vector=vector, speaker_id=speaker_id)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        return float(np.dot(a, b) / denom)

    def match(self, target: SpeakerEmbedding, candidates: List[SpeakerEmbedding]) -> List[float]:
        return [self.cosine_similarity(target.vector, candidate.vector) for candidate in candidates]


