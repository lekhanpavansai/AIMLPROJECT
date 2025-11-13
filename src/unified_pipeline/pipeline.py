from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from unified_pipeline.asr.transcriber import ASRPipeline, ASRConfig, PunctuationConfig
from unified_pipeline.audio_io.ingest import AudioLoader, AudioLoaderConfig
from unified_pipeline.config.settings import get_settings
from unified_pipeline.diarization.diarizer import DiarizationPipeline, DiarizationConfig
from unified_pipeline.postprocessing.formatter import TranscriptFormatter, OutputConfig
from unified_pipeline.preprocessing.pipeline import PreprocessingPipeline, PreprocessingConfig
from unified_pipeline.separation.embedding import SpeakerEncoder, SpeakerEmbedding, SpeakerEncoderConfig
from unified_pipeline.separation.separator import MossFormerSeparator, SeparationConfig
from unified_pipeline.separation.target_selector import TargetSelector, TargetSelectionResult
from unified_pipeline.core_types import AudioFrame, PipelineContext, SpeechSegment, TranscriptSegment
from unified_pipeline.utils.audio import save_waveform
from unified_pipeline.utils.logging import get_logger
from unified_pipeline.visualization.export import TimelineExporter, VisualizationConfig

log = get_logger(__name__)


@dataclass(slots=True)
class PipelineConfig:
    audio: AudioLoaderConfig = field(default_factory=AudioLoaderConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    separation: SeparationConfig = field(default_factory=SeparationConfig)
    speaker_embedding: SpeakerEncoderConfig = field(default_factory=SpeakerEncoderConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    punctuation: PunctuationConfig = field(default_factory=PunctuationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


@dataclass(slots=True)
class StreamingContext:
    session_id: str
    sample_rate: int
    target_embedding: SpeakerEmbedding
    chunk_index: int = 0
    transcripts: List[TranscriptSegment] = field(default_factory=list)


@dataclass(slots=True)
class StreamingChunkResult:
    segments: List[TranscriptSegment]
    chunk_index: int
    chunk_start: float
    chunk_duration: float


class UnifiedPipeline:
    """High-level orchestrator connecting all modules."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.settings = get_settings()
        self.loader = AudioLoader(self.config.audio)
        self.preprocessing = PreprocessingPipeline(self.config.preprocessing)
        self.separator = MossFormerSeparator(self.config.separation)
        self.speaker_encoder = SpeakerEncoder(self.config.speaker_embedding)
        self.target_selector = TargetSelector(self.speaker_encoder, threshold=self.config.separation.similarity_threshold)
        self.diarizer = DiarizationPipeline(self.config.diarization)
        self.asr = ASRPipeline(self.config.asr, self.config.punctuation)
        self.formatter = TranscriptFormatter(self.config.output)
        self.visualization = TimelineExporter(self.config.visualization)

    async def run_offline(self, mixture_path: Path, target_path: Path, output_dir: Path) -> List[TranscriptSegment]:
        log.info("pipeline.start", mixture=str(mixture_path), target=str(target_path))
        context = PipelineContext(
            session_id=mixture_path.stem,
            device=self.settings.device,
            work_dir=output_dir,
        )

        mixture = self.loader.load_file(mixture_path)
        target = self.loader.load_file(target_path)
        target_embedding = await self.speaker_encoder.embed(target, speaker_id="Target")

        preprocessed_frames = await self.preprocessing.process(mixture)
        separation = await self.separator.separate(preprocessed_frames[0], context)
        selection = await self.target_selector.select(separation, target_embedding)
        separation.target_index = selection.target_index
        separation.similarities = selection.similarities

        # Save all separated speaker stems
        output_dir.mkdir(parents=True, exist_ok=True)
        speakers_dir = output_dir / "speakers"
        speakers_dir.mkdir(parents=True, exist_ok=True)
        
        saved_speakers = []
        for idx, stem in enumerate(separation.stems):
            speaker_path = speakers_dir / f"speaker_{idx}.wav"
            save_waveform(stem.samples, stem.sample_rate, speaker_path)
            is_target = (selection.target_index == idx) if selection.target_index is not None else False
            saved_speakers.append({
                "index": idx,
                "path": str(speaker_path),
                "is_target": is_target,
                "similarity": selection.similarities[idx] if idx < len(selection.similarities) else 0.0
            })
            log.info(
                "pipeline.speaker.saved",
                speaker_idx=idx,
                path=str(speaker_path),
                is_target=is_target,
                similarity=saved_speakers[-1]["similarity"]
            )
        
        # Also save target speaker with a dedicated name for backward compatibility
        if selection.target_stem:
            target_audio_path = output_dir / "target_speaker.wav"
            save_waveform(selection.target_stem.samples, selection.target_stem.sample_rate, target_audio_path)
            log.info("pipeline.target.saved", path=str(target_audio_path), similarity=selection.similarities)
        else:
            log.warning("pipeline.target.missing", reason="similarity_below_threshold", similarities=selection.similarities)
        
        # Save speaker metadata
        speakers_metadata_path = output_dir / "speakers_metadata.json"
        with open(speakers_metadata_path, "w", encoding="utf-8") as f:
            json.dump({"speakers": saved_speakers, "total_speakers": len(separation.stems)}, f, indent=2)
        log.info("pipeline.speakers.metadata.saved", path=str(speakers_metadata_path), count=len(separation.stems))

        diarized_segments = await self.diarizer.diarize(str(mixture_path), separation=separation)
        if not diarized_segments:
            diarized_segments = self._fallback_segments(selection, selection.target_stem or mixture)

        transcripts = await self.asr.transcribe(diarized_segments)

        output_dir.mkdir(parents=True, exist_ok=True)
        diarization_json = output_dir / "diarization.json"
        self.formatter.save_json(transcripts, diarization_json)
        timeline_entries = self.visualization.export(transcripts, output_dir)

        log.info("pipeline.complete", output=str(diarization_json))
        context.cache["transcripts"] = transcripts
        context.cache["timeline"] = timeline_entries
        return transcripts

    async def create_stream_context(self, session_id: str, target_frame: AudioFrame) -> StreamingContext:
        target_embedding = await self.speaker_encoder.embed(target_frame, speaker_id="Target")
        return StreamingContext(
            session_id=session_id,
            sample_rate=target_frame.sample_rate,
            target_embedding=target_embedding,
        )

    async def process_stream_chunk(
        self,
        context: StreamingContext,
        chunk_frame: AudioFrame,
    ) -> StreamingChunkResult:
        chunk_duration = float(len(chunk_frame.samples) / chunk_frame.sample_rate)
        chunk_start = context.chunk_index * chunk_duration
        chunk_frame.timestamp = chunk_start

        preprocessed_frames = await self.preprocessing.process(chunk_frame)
        if not preprocessed_frames:
            return StreamingChunkResult(segments=[], chunk_index=context.chunk_index, chunk_start=chunk_start, chunk_duration=chunk_duration)

        separation = await self.separator.separate(preprocessed_frames[0], context)
        selection = await self.target_selector.select(separation, context.target_embedding)
        separation.target_index = selection.target_index
        separation.similarities = selection.similarities

        target_frame = selection.target_stem or preprocessed_frames[0]
        target_frame.timestamp = chunk_start
        diarized_segments = self._fallback_segments(selection, target_frame, chunk_start=chunk_start)

        transcripts = await self.asr.transcribe(diarized_segments)
        for segment in transcripts:
            segment.with_extra(chunk_index=context.chunk_index)

        context.chunk_index += 1
        context.transcripts.extend(transcripts)
        return StreamingChunkResult(
            segments=transcripts,
            chunk_index=context.chunk_index - 1,
            chunk_start=chunk_start,
            chunk_duration=chunk_duration,
        )

    def _fallback_segments(
        self,
        selection_result: TargetSelectionResult,
        audio_frame: AudioFrame,
        chunk_start: float = 0.0,
    ) -> List[SpeechSegment]:
        duration = float(len(audio_frame.samples) / max(audio_frame.sample_rate, 1))
        similarity = max(selection_result.similarities, default=0.5)
        return [
            SpeechSegment(
                speaker_id="Target",
                start_s=chunk_start,
                end_s=chunk_start + duration,
                confidence=similarity,
            )
        ]


def run_pipeline_sync(mixture_path: Path, target_path: Path, output_dir: Path) -> List[TranscriptSegment]:
    pipeline = UnifiedPipeline()
    return asyncio.run(pipeline.run_offline(mixture_path, target_path, output_dir))


