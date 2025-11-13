from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def save_waveform(samples: np.ndarray, sample_rate: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples, sample_rate)














