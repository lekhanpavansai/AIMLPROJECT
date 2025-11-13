from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, TypedDict

from unified_pipeline.core_types import TranscriptSegment


class TimelineEntry(TypedDict):
    speaker: str
    start: float
    end: float
    text: str
    confidence: float
    duration: float
    is_target: bool
    chunk_index: int | None
    overlap: bool


@dataclass(slots=True)
class TimelineConfig:
    merge_gap_s: float = 0.2
    round_digits: int = 2


class TimelineBuilder:
    """Generate timeline data structures for visualization layers."""

    def __init__(self, config: TimelineConfig | None = None) -> None:
        self.config = config or TimelineConfig()

    def build(self, segments: Iterable[TranscriptSegment]) -> List[TimelineEntry]:
        ordered_segments = sorted(list(segments), key=lambda seg: (seg.start_s, seg.end_s))
        timeline: List[TimelineEntry] = []
        for segment in ordered_segments:
            start = round(segment.start_s, self.config.round_digits)
            end = round(segment.end_s, self.config.round_digits)
            entry = TimelineEntry(
                speaker=segment.speaker_label,
                start=start,
                end=end,
                text=segment.text,
                confidence=segment.confidence,
                duration=max(0.0, end - start),
                is_target=segment.speaker_label.lower() == "target",
                chunk_index=segment.extra.get("chunk_index") if segment.extra else None,
                overlap=bool(segment.extra.get("overlap")) if segment.extra else False,
            )
            timeline.append(entry)

        if not timeline:
            return timeline

        return self._merge_close_segments(timeline)

    def _merge_close_segments(self, entries: List[TimelineEntry]) -> List[TimelineEntry]:
        merged: List[TimelineEntry] = []
        for entry in entries:
            if merged and self._can_merge(merged[-1], entry):
                merged[-1] = self._merge_entries(merged[-1], entry)
            else:
                merged.append(entry)
        return merged

    def _can_merge(self, prev: TimelineEntry, current: TimelineEntry) -> bool:
        return (
            prev["speaker"] == current["speaker"]
            and current["start"] - prev["end"] <= self.config.merge_gap_s
            and prev["overlap"] == current["overlap"]
        )

    def _merge_entries(self, prev: TimelineEntry, current: TimelineEntry) -> TimelineEntry:
        start = prev["start"]
        end = current["end"]
        text = " ".join(filter(None, [prev["text"], current["text"]])).strip()
        return TimelineEntry(
            speaker=prev["speaker"],
            start=start,
            end=end,
            text=text,
            confidence=max(prev["confidence"], current["confidence"]),
            duration=max(0.0, end - start),
            is_target=prev["is_target"],
            chunk_index=current["chunk_index"],
            overlap=prev["overlap"],
        )


