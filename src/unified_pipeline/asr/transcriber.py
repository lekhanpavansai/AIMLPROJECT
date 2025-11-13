from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

try:  # pragma: no cover - optional dependency guard
    import whisper
except ImportError:  # pragma: no cover
    whisper = None  # type: ignore[assignment]

from unified_pipeline.core_types import SpeechSegment, TranscriptSegment
from unified_pipeline.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class ASRConfig:
    backend: str = "whisper"
    model_path: Optional[str] = None
    language: str | None = None
    device: str = "cuda"
    enable_punctuation: bool = True


@dataclass(slots=True)
class PunctuationConfig:
    model_path: Optional[str] = None
    device: str = "cuda"
    enabled: bool = True


class ASRPipeline:
    """Dispatch transcription to configured backend."""

    def __init__(
        self,
        config: ASRConfig | None = None,
        punctuation_config: PunctuationConfig | None = None,
    ) -> None:
        self.config = config or ASRConfig()
        self.punctuation_config = punctuation_config or PunctuationConfig()
        self._model = None
        self._punct_model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        if self.config.backend == "whisper":
            self._load_whisper()
        else:
            log.warning("asr.backend.unsupported", backend=self.config.backend)

    def _load_whisper(self) -> None:
        if whisper is None:
            log.warning("asr.whisper.unavailable", reason="whisper_not_installed")
            return

        model_id = self.config.model_path or "large-v3"
        log.info("asr.whisper.loading", model=model_id)
        try:
            self._model = whisper.load_model(model_id, device=self.config.device)
            log.info("asr.whisper.loaded", model=model_id)
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("asr.whisper.load_failed", error=str(exc))
            self._model = None

    def _apply_punctuation(self, text: str) -> str:
        if not self.config.enable_punctuation or not self.punctuation_config.enabled:
            return text
        if self._punct_model is None:
            self._load_punctuation_model()
        if self._punct_model is None:
            return text
        log.warning("asr.punctuation.fallback_stub", reason="punctuation_model_not_integrated")
        return text

    def _load_punctuation_model(self) -> None:
        if not self.punctuation_config.model_path:
            log.warning("asr.punctuation.unconfigured", reason="missing_model_path")
            return
        model_path = Path(self.punctuation_config.model_path)
        if not model_path.exists():
            log.warning("asr.punctuation.missing_model", path=str(model_path))
            return
        log.warning("asr.punctuation.fallback_stub", reason="loading_not_implemented")
        self._punct_model = None

    async def transcribe(self, segments: Iterable[SpeechSegment]) -> List[TranscriptSegment]:
        self._ensure_model()
        transcripts: List[TranscriptSegment] = []

        for segment in segments:
            text, confidence = await self._transcribe_segment(segment)
            text = self._apply_punctuation(text)
            transcripts.append(
                TranscriptSegment(
                    speaker_label=segment.speaker_id,
                    start_s=segment.start_s,
                    end_s=segment.end_s,
                    text=text,
                    confidence=confidence,
                )
            )

        return transcripts

    async def _transcribe_segment(self, segment: SpeechSegment) -> tuple[str, float]:
        if self._model is None:
            log.warning("asr.segment.fallback", reason="model_not_available")
            return "[transcription pending]", segment.confidence

        if self.config.backend == "whisper":
            return await self._transcribe_whisper(segment)

        log.warning("asr.segment.unsupported_backend", backend=self.config.backend)
        return "[transcription pending]", segment.confidence

    async def _transcribe_whisper(self, segment: SpeechSegment) -> tuple[str, float]:
        audio = segment.embedding if isinstance(segment.embedding, str) else None
        if audio:
            audio_path = audio
        else:
            audio_path = None

        log.warning("asr.whisper.fallback_stub", reason="no_audio_path_provided")
        return "[whisper transcription pending]", segment.confidence


