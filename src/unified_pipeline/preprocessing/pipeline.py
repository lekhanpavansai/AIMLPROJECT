from __future__ import annotations

from dataclasses import dataclass
from typing import List

# pyright: reportMissingImports=false

import numpy as np
from scipy.signal import butter, lfilter

from unified_pipeline.core_types import AudioFrame
from unified_pipeline.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class PreprocessingConfig:
    enable_denoise: bool = True
    enable_vad: bool = True
    min_speech_duration: float = 0.2
    frame_length_ms: float = 30.0
    frame_hop_ms: float = 10.0
    pad_ms: float = 100.0
    energy_threshold: float | None = None
    highpass_hz: float = 50.0


class PreprocessingPipeline:
    """Run denoising, VAD, and endpoint detection."""

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self.config = config or PreprocessingConfig()

    async def process(self, frame: AudioFrame) -> List[AudioFrame]:
        processed = frame
        if self.config.enable_denoise:
            processed = await self._denoise(processed)

        if self.config.enable_vad:
            segments = await self._apply_vad(processed)
        else:
            segments = [processed]

        return segments

    async def _denoise(self, frame: AudioFrame) -> AudioFrame:
        cutoff_hz = max(5.0, self.config.highpass_hz)
        if cutoff_hz >= frame.sample_rate / 2:
            return frame

        nyquist = 0.5 * frame.sample_rate
        norm_cutoff = cutoff_hz / nyquist
        b, a = butter(2, norm_cutoff, btype="highpass")
        filtered = lfilter(b, a, frame.samples, axis=0)
        return AudioFrame(
            samples=filtered.astype(np.float32),
            sample_rate=frame.sample_rate,
            channels=frame.channels,
            timestamp=frame.timestamp,
        )

    async def _apply_vad(self, frame: AudioFrame) -> List[AudioFrame]:
        samples = frame.samples
        mono = samples.mean(axis=1)
        sample_rate = frame.sample_rate
        frame_length = max(1, int(self.config.frame_length_ms * sample_rate / 1000))
        hop_length = max(1, int(self.config.frame_hop_ms * sample_rate / 1000))

        if len(mono) < frame_length:
            return [frame]

        energies = self._frame_energy(mono, frame_length, hop_length)
        threshold = (
            self.config.energy_threshold
            if self.config.energy_threshold is not None
            else np.median(energies) * 0.6
        )

        active_frames = energies >= threshold
        segments: List[AudioFrame] = []
        pad_samples = int(self.config.pad_ms * sample_rate / 1000)
        min_samples = int(self.config.min_speech_duration * sample_rate)

        start_frame = None
        for idx, is_active in enumerate(active_frames):
            if is_active and start_frame is None:
                start_frame = idx
            elif not is_active and start_frame is not None:
                segment = self._create_segment(
                    frame, start_frame, idx, frame_length, hop_length, pad_samples, min_samples
                )
                if segment:
                    segments.append(segment)
                start_frame = None

        if start_frame is not None:
            segment = self._create_segment(
                frame, start_frame, len(active_frames), frame_length, hop_length, pad_samples, min_samples
            )
            if segment:
                segments.append(segment)

        if not segments:
            log.warning("vad.fallback", reason="no_active_segments")
            return [frame]

        return segments

    @staticmethod
    def _frame_energy(mono: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
        num_frames = 1 + max(0, (len(mono) - frame_length) // hop_length)
        energies = np.empty(num_frames, dtype=np.float32)
        for i in range(num_frames):
            start = i * hop_length
            end = start + frame_length
            segment = mono[start:end]
            energies[i] = float(np.mean(segment * segment) + 1e-8)
        return energies

    def _create_segment(
        self,
        frame: AudioFrame,
        start_frame: int,
        end_frame: int,
        frame_length: int,
        hop_length: int,
        pad_samples: int,
        min_samples: int,
    ) -> AudioFrame | None:
        start_sample = max(0, start_frame * hop_length - pad_samples)
        end_sample = min(len(frame.samples), end_frame * hop_length + pad_samples + frame_length)
        if end_sample - start_sample < min_samples:
            return None

        segment_samples = frame.samples[start_sample:end_sample]
        timestamp = frame.timestamp + (start_sample / frame.sample_rate)
        return AudioFrame(
            samples=segment_samples.copy(),
            sample_rate=frame.sample_rate,
            channels=frame.channels,
            timestamp=timestamp,
        )


