from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from unified_pipeline.core_types import TranscriptSegment


@dataclass(slots=True)
class OutputConfig:
    pretty_json: bool = True


class TranscriptFormatter:
    """Serialize transcripts to JSON/SRT and generate auxiliary artifacts."""

    def __init__(self, config: OutputConfig | None = None) -> None:
        self.config = config or OutputConfig()

    def to_json(self, segments: Iterable[TranscriptSegment]) -> str:
        payload = [
            {
                "speaker": segment.speaker_label,
                "start": segment.start_s,
                "end": segment.end_s,
                "text": segment.text,
                "confidence": segment.confidence,
                **({"language": segment.language} if segment.language else {}),
                **segment.extra,
            }
            for segment in segments
        ]
        if self.config.pretty_json:
            return json.dumps(payload, indent=2)
        return json.dumps(payload, separators=(",", ":"))

    def save_json(self, segments: Iterable[TranscriptSegment], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(list(segments)), encoding="utf-8")


