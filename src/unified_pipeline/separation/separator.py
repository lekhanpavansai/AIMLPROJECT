from __future__ import annotations

# pyright: reportMissingImports=false

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

try:  # pragma: no cover - optional dependency guard
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency guard
    import librosa
except ImportError:  # pragma: no cover
    librosa = None  # type: ignore[assignment]

from unified_pipeline.core_types import AudioFrame, PipelineContext
from unified_pipeline.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class SeparationResult:
    stems: List[AudioFrame]
    target_index: Optional[int] = None
    similarities: List[float] = field(default_factory=list)


@dataclass(slots=True)
class SeparationConfig:
    model_path: Optional[str] = None
    sample_rate: int = 48_000
    max_speakers: int = 4
    detection_threshold: float = 0.5
    similarity_threshold: float = 0.6
    device: str = "cuda"


class MossFormerSeparator:
    """Wrapper for MossFormer2 separation model."""

    def __init__(self, config: SeparationConfig | None = None) -> None:
        self.config = config or SeparationConfig()
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        if not self.config.model_path:
            log.warning("separator.model.unconfigured", reason="missing_model_path")
            return

        if torch is None:
            log.warning("separator.model.unavailable", reason="torch_not_installed")
            return

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            log.warning("separator.model.missing", path=str(model_path))
            return

        log.info("separator.model.loading", path=str(model_path))
        try:
            checkpoint = torch.load(model_path, map_location=self.config.device)
            self._model = checkpoint  # Placeholder: replace with actual model init
            log.info("separator.model.loaded", path=str(model_path))
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("separator.model.load_failed", error=str(exc))
            self._model = None

    async def separate(self, frame: AudioFrame, context: Optional[PipelineContext] = None) -> SeparationResult:
        if self._model is None:
            self._ensure_model()

        if self._model is None:
            log.warning("separator.fallback", reason="model_not_available")
            dummy = AudioFrame(samples=np.copy(frame.samples), sample_rate=frame.sample_rate, channels=frame.channels)
            return SeparationResult(stems=[dummy])

        mixture = frame.samples
        orig_sr = frame.sample_rate
        target_sr = self.config.sample_rate

        if target_sr != orig_sr:
            if librosa is None:
                raise RuntimeError("librosa is required for resampling separation input but is not installed.")
            mixture = librosa.resample(mixture.T, orig_sr=orig_sr, target_sr=target_sr, axis=-1).T

        if mixture.ndim == 1:
            mixture = mixture[:, None]

        # Placeholder inference: replace with actual MossFormer2 forward pass
        stems = [mixture]

        if target_sr != orig_sr:
            stems = [
                librosa.resample(stem.T, orig_sr=target_sr, target_sr=orig_sr, axis=-1).T for stem in stems
            ]

        audio_stems = [
            AudioFrame(samples=stem.astype(np.float32), sample_rate=frame.sample_rate, channels=stem.shape[1], timestamp=frame.timestamp)
            for stem in stems
        ]

        return SeparationResult(stems=audio_stems[: self.config.max_speakers])


