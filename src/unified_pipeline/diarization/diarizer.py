from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

try:  # pragma: no cover - optional dependency guard
    from pyannote.audio import Pipeline as PyannotePipeline
except ImportError:  # pragma: no cover
    PyannotePipeline = None  # type: ignore[assignment]

from unified_pipeline.separation.separator import SeparationResult
from unified_pipeline.core_types import SpeechSegment
from unified_pipeline.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class DiarizationConfig:
    overlap_model_path: Optional[str] = None
    diarization_model_path: Optional[str] = None
    min_speech_duration: float = 0.3
    max_speakers: int = 4
    device: str = "cuda"


class DiarizationPipeline:
    """Combine pyannote diarization with target speaker scoring."""

    def __init__(self, config: DiarizationConfig | None = None) -> None:
        self.config = config or DiarizationConfig()
        self._pipeline = None

    def _ensure_pipeline(self) -> None:
        if self._pipeline is not None:
            return

        if PyannotePipeline is None:
            log.warning("diarization.pipeline.unavailable", reason="pyannote_not_installed")
            return

        if not self.config.diarization_model_path:
            log.warning("diarization.pipeline.unconfigured", reason="missing_model_path")
            return

        model_path = Path(self.config.diarization_model_path)
        if not model_path.exists():
            log.warning("diarization.pipeline.missing_model", path=str(model_path))
            return

        log.info("diarization.pipeline.loading", path=str(model_path))
        try:
            self._pipeline = PyannotePipeline.from_pretrained(model_path, device=self.config.device)
            log.info("diarization.pipeline.loaded", path=str(model_path))
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("diarization.pipeline.load_failed", error=str(exc))
            self._pipeline = None

    async def diarize(
        self,
        audio_path: str,
        separation: Optional[SeparationResult] = None,
    ) -> List[SpeechSegment]:
        self._ensure_pipeline()

        if self._pipeline is None:
            log.warning("diarization.pipeline.fallback", reason="pipeline_not_available")
            if separation and separation.stems:
                return [
                    SpeechSegment(
                        speaker_id=f"speaker_{idx}",
                        start_s=0.0,
                        end_s=float(len(stem.samples) / stem.sample_rate),
                        confidence=0.5,
                    )
                    for idx, stem in enumerate(separation.stems[: self.config.max_speakers])
                ]
            return []

        try:
            diarization_result = await self._run_pipeline(audio_path)
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("diarization.pipeline.run_failed", error=str(exc))
            return []

        segments: List[SpeechSegment] = []
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            if turn.duration < self.config.min_speech_duration:
                continue
            segments.append(
                SpeechSegment(
                    speaker_id=str(speaker),
                    start_s=float(turn.start),
                    end_s=float(turn.end),
                    confidence=0.8,
                )
            )

        if not segments and separation and separation.stems:
            log.warning("diarization.pipeline.empty", reason="no_segments")
            return [
                SpeechSegment(
                    speaker_id=f"speaker_{idx}",
                    start_s=0.0,
                    end_s=float(len(stem.samples) / stem.sample_rate),
                    confidence=0.5,
                )
                for idx, stem in enumerate(separation.stems[: self.config.max_speakers])
            ]

        return segments

    async def _run_pipeline(self, audio_path: str):
        if hasattr(self._pipeline, "to_async"):
            return await self._pipeline.to_async()(audio_path)
        return self._pipeline(audio_path)


