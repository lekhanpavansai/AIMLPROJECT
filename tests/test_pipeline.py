import numpy as np
import pytest

from unified_pipeline.pipeline import UnifiedPipeline
from unified_pipeline.separation.target_selector import TargetSelectionResult
from unified_pipeline.core_types import AudioFrame


def make_audio_frame(duration_s: float = 1.0, sample_rate: int = 16000) -> AudioFrame:
    samples = np.zeros(int(duration_s * sample_rate), dtype=np.float32)
    return AudioFrame(samples=samples, sample_rate=sample_rate, channels=1)


def test_pipeline_initializes() -> None:
    pipeline = UnifiedPipeline()
    assert pipeline is not None


def test_fallback_segments_produce_target_label() -> None:
    pipeline = UnifiedPipeline()
    selection = TargetSelectionResult(target_stem=None, target_index=None, similarities=[0.7, 0.6])
    frame = make_audio_frame()
    segments = pipeline._fallback_segments(selection, frame)
    assert segments[0].speaker_id == "Target"


@pytest.mark.asyncio
async def test_streaming_context_process_chunk() -> None:
    pipeline = UnifiedPipeline()
    target_frame = make_audio_frame()
    context = await pipeline.create_stream_context("test-session", target_frame)
    chunk = make_audio_frame()
    result = await pipeline.process_stream_chunk(context, chunk)
    assert result.chunk_index == 0
    assert result.chunk_duration > 0


