from __future__ import annotations

# pyright: reportMissingImports=false

from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Optional

try:  # pragma: no cover - optional dependency guard for static analysis
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency guard
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency guard
    import librosa
except ImportError:  # pragma: no cover
    librosa = None  # type: ignore[assignment]

from unified_pipeline.core_types import AudioFrame


@dataclass(slots=True)
class AudioLoaderConfig:
    target_sample_rate: int = 16_000
    stereo: bool = True
    resample: bool = True


class AudioLoader:
    """Load audio from disk into AudioFrame objects."""

    def __init__(self, config: Optional[AudioLoaderConfig] = None) -> None:
        self.config = config or AudioLoaderConfig()

    def load_file(self, path: Path) -> AudioFrame:
        if np is None:
            raise RuntimeError("numpy is required for AudioLoader; please install numpy.")
        if sf is None:
            raise RuntimeError("soundfile is required for AudioLoader; please install soundfile.")

        samples, sample_rate = sf.read(path, always_2d=True)
        array: Any = np.asarray(samples, dtype=np.float32)

        if self.config.resample and sample_rate != self.config.target_sample_rate:
            if librosa is None:
                raise RuntimeError(
                    "librosa is required for resampling but is not installed. "
                    "Install librosa or disable resample in AudioLoaderConfig."
                )
            array = librosa.resample(
                array.T,
                orig_sr=sample_rate,
                target_sr=self.config.target_sample_rate,
                axis=-1,
            ).T
            sample_rate = self.config.target_sample_rate

        if not self.config.stereo and array.shape[1] > 1:
            array = np.mean(array, axis=1, keepdims=True)
        elif self.config.stereo and array.shape[1] == 1:
            array = np.repeat(array, 2, axis=1)

        channels = array.shape[1]
        return AudioFrame(samples=array, sample_rate=sample_rate, channels=channels)


class StreamBuffer:
    """Simplified streaming buffer for future WebSocket ingestion."""

    def __init__(self, sample_rate: int, frame_size: int) -> None:
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        if np is None:
            raise RuntimeError("numpy is required for StreamBuffer; please install numpy.")
        self._buffer = np.zeros((frame_size,), dtype=np.float32)

    async def frames(self) -> AsyncIterator[AudioFrame]:
        # Placeholder for streaming frames from network/audio device.
        raise NotImplementedError("StreamBuffer.frames must be implemented for streaming mode.")


