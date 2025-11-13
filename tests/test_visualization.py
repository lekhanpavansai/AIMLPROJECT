from unified_pipeline.core_types import TranscriptSegment
from unified_pipeline.visualization.timeline import TimelineBuilder, TimelineConfig


def make_segment(speaker: str, start: float, end: float, text: str) -> TranscriptSegment:
    return TranscriptSegment(
        speaker_label=speaker,
        start_s=start,
        end_s=end,
        text=text,
        confidence=0.9,
    )


def test_timeline_builder_merges_close_segments() -> None:
    builder = TimelineBuilder(TimelineConfig(merge_gap_s=0.5, round_digits=2))
    segments = [
        make_segment("Target", 0.0, 1.0, "hello"),
        make_segment("Target", 1.4, 2.0, "world"),
        make_segment("Speaker_B", 2.5, 3.0, "other"),
    ]

    timeline = builder.build(segments)
    assert len(timeline) == 2
    merged_entry = timeline[0]
    assert merged_entry["speaker"] == "Target"
    assert merged_entry["duration"] == 2.0
    assert "hello" in merged_entry["text"] and "world" in merged_entry["text"]

