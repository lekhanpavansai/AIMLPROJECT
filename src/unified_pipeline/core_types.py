from __future__ import annotations

# pyright: reportMissingImports=false

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:  # pragma: no cover - optional dependency guard for static analysis
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]


@dataclass(slots=True)
class AudioFrame:
    """Container for audio samples flowing between modules."""

    samples: np.ndarray
    sample_rate: int
    channels: int
    timestamp: float = 0.0


@dataclass(slots=True)
class SpeechSegment:
    """Represents a diarized speech segment."""

    speaker_id: str
    start_s: float
    end_s: float
    confidence: float
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TranscriptSegment:
    """Final transcription segment with metadata."""

    speaker_label: str
    start_s: float
    end_s: float
    text: str
    confidence: float
    language: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def with_extra(self, **kwargs: Any) -> "TranscriptSegment":
        self.extra.update(kwargs)
        return self


@dataclass(slots=True)
class PipelineContext:
    """Holds runtime state shared across modules."""

    session_id: str
    device: str
    work_dir: Path
    cache: Dict[str, Any] = field(default_factory=dict)


class SupportsBatch:
    """Mixin for modules that can process iterables of items."""

    async def process_batch(self, items: Iterable[Any], **kwargs: Any) -> List[Any]:
        return [await self.process(item, **kwargs) for item in items]

    async def process(self, item: Any, **kwargs: Any) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError


