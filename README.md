# Unified Neural Pipeline

Target speaker diarization and multispeaker automatic speech recognition pipeline. Combines state-of-the-art open-source models for noise suppression, separation, diarization, recognition, and punctuation to deliver clean target speaker audio and structured transcripts.

## Features
- Target speaker extraction from multi-speaker mixtures using reference clips (3-60 seconds recommended).
- Multi-speaker diarization with overlap detection and confidence estimates.
- Modular ASR backends (Paraformer, Whisper, SenseVoice) with punctuation restoration.
- Batch and real-time streaming modes (REST + WebSocket APIs).
- Timeline visualization and structured JSON output.
- Real-audio front-end: resampling, high-pass denoising, and energy-based VAD segmentation.
- Pluggable speaker separation (MossFormer2) and speaker embedding (ERes2NetV2) with configurable thresholds.
- Pyannote-based diarization hooks and ASR backends (Whisper/Paraformer/SenseVoice) with optional punctuation models.

## Quickstart
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows
pip install -e .[dev]
```

### Offline Inference (planned)
```bash
python -m unified_pipeline.cli.batch \
  --mixture data/mixture_audio.wav \
  --target data/target_sample.wav \
  --output outputs/session_001
```

### Streaming Inference (planned)
```bash
uvicorn unified_pipeline.api.app:app --reload
```

WebSocket handshake example (JSON messages):
```json
{"event": "start", "session_id": "demo", "sample_rate": 16000, "target_sample_b64": "<base64 float32 audio>"}
{"event": "audio_chunk", "chunk_b64": "<base64 float32 audio>"}
{"event": "stop"}
```

## Model Weights

Place pretrained checkpoints under `models/` and update `configs/default.yaml` (or your custom config):

- `models/mossformer2/mossformer2.pt` – separation (MossFormer2).
- `models/eres2net/eres2netv2-large.ckpt` – speaker encoder (ERes2NetV2-Large).
- `models/pyannote/*.pt` – diarization/overlap detection pipelines.
- `models/whisper/large-v3` (or Alt. ONNX) – ASR backend; adjust `asr.backend` as needed.
- `models/ct_transformer/punctuation.pt` – punctuation restoration (optional).
- Adjust `separation.sample_rate`, `separation.similarity_threshold`, and `speaker_embedding.device` as needed.

## Repository Layout
- `docs/` – architecture notes, research references.
- `configs/` – Hydra configuration bundles.
- `src/unified_pipeline/` – core Python packages.
- `data/` – placeholder for sample audio.
- `tests/` – unit and integration tests (TBD).
- `outputs/<session>/timeline.{json,html}` – timeline data and interactive visualization.

## Roadmap
1. Implement module scaffolding and configuration management.
2. Integrate pretrained models with runtime adapters and GPU acceleration.
3. Build unified pipeline orchestrator for batch + streaming.
4. Expose REST/WebSocket endpoints and visualization tooling.
5. Add automated tests, benchmarks, and deployment scripts.

## License
Apache-2.0 (to be confirmed).

