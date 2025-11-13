## Unified Neural Pipeline Architecture

### 1. Goals
- Identify and enhance a target speaker within multi-speaker, noisy audio.
- Provide diarized, punctuated transcriptions for all speakers.
- Support both offline batch jobs and low-latency streaming.
- Produce modular outputs including separated waveforms and structured JSON.

### 2. High-Level Flow
1. **Ingestion**  
   - Accept local files (`mixture_audio.wav`, `target_sample.wav`) or streaming frames via WebSocket/RTSP.  
   - Normalize sampling rate (48 kHz internal), convert to mono/stereo as required.
2. **Preprocessing**  
   - `CAM++ Endpoint Detector` for coarse speech boundary detection.  
   - `UVR-MDX-Net` denoising to remove broadband noise/reverb.  
   - `FSMN-Monophone VAD` for frame-level speech segmentation.
3. **Separation & Enhancement**  
   - `MossFormer2` multi-speaker source separation (N>2).  
   - `Apollo` restoration pass on separated stems to recover clarity.  
4. **Target Speaker Matching**  
   - `ERes2NetV2-Large` speaker embedding extractor.  
   - Cosine similarity scoring against `target_sample.wav` embedding.  
   - Select target stem, recompose residual mix for non-target speakers.
5. **Diarization**  
   - `pyannote.audio` overlap detection (`OverlapDetector`) to flag concurrent speech.  
   - `SpeakerTurnAggregator` merges VAD segments with embeddings into diarized tracks.  
6. **ASR & NLP**  
   - Hybrid ASR backend (`Paraformer` for Mandarin, `Whisper-large-v3` for multilingual, `SenseVoice` fallback).  
   - `CT-Transformer` punctuation and truecasing.  
   - Optional translation (`mBART50`) if configured.
7. **Post-Processing**  
   - Confidence calibration (ASR beam scores + embedding similarity).  
   - Timestamp smoothing (median filter).  
   - Output serialization to JSON, SRT, and timeline visualization data.

### 3. Module Breakdown
- `src/audio_io/` – loaders, resamplers, chunkers, stream handlers.
- `src/preprocessing/` – denoising, VAD, endpoint detection.
- `src/separation/` – MossFormer2 wrappers, target voice selection.
- `src/diarization/` – pyannote pipelines, speaker assignment logic.
- `src/asr/` – pluggable ASR inference engines, punctuation restoration.
- `src/postprocessing/` – confidence fusion, formatting, export.
- `src/api/` – REST (`FastAPI`) and streaming (`WebSocket`) endpoints.
- `src/visualization/` – timeline builder, static HTML + React component.
- `src/config/` – Hydra configuration schemas for pipeline and runtime.

### 4. Data Flow Contracts
- **AudioFrame**: `sample_rate`, `channels`, `np.ndarray`/`torch.Tensor`.
- **SpeechSegment**: `speaker_id`, `start_s`, `end_s`, `confidence`, `embedding`.
- **TranscriptSegment**: `speaker_label`, `start_s`, `end_s`, `text`, `confidence`, `language`.
- **PipelineContext**: run metadata, device allocation, cache handles.

### 5. Execution Modes
- **Offline Batch**: load full files, process sequentially with GPU-accelerated batching.  
  - CLI via `python -m src.cli.batch --mixture ... --target ...`.
- **Streaming**: sliding-window inference (`2s` hop, `6s` window) with stateful buffers.  
  - Back-pressure controls; results pushed incrementally over WebSocket.

### 6. Runtime Strategy
- Use `PyTorch` with CUDA / TensorRT acceleration where available.  
- Orchestrate asynchronous steps with `asyncio` + `torch.cuda.Stream`.  
- Employ `onnxruntime-gpu` for models with ONNX exports (Paraformer, CT-Transformer).
- Cache speaker embeddings, VAD states, and ASR decoder states for low latency.

### 7. Storage & Artifacts
- Separated audio saved under `outputs/<session_id>/`.  
- JSON transcripts under `outputs/<session_id>/diarization.json`.  
- Visualization data exported as `timeline.json` for frontend consumption.

### 8. Observability
- Structured logging via `structlog`.  
- Metrics with `Prometheus` exporters (latency, GPU mem, throughput).  
- Optional tracing with `OpenTelemetry`.

### 9. Next Steps
1. Define detailed module interfaces in Python packages.  
2. Prepare environment files (`pyproject.toml`, `environment.yml`).  
3. Prototype key stages (embedding matching, MossFormer2 inference) with sample audio.  
4. Build API skeleton (`FastAPI` + `uvicorn`).  
5. Implement visualization stub (React/Plotly timeline).

