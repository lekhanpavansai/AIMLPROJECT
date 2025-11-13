from __future__ import annotations

# pyright: reportMissingImports=false

from pathlib import Path
from uuid import uuid4

try:  # pragma: no cover - optional dependency guard for static analysis
    from fastapi import FastAPI, File, UploadFile, WebSocket
    from fastapi.responses import JSONResponse
except ImportError:  # pragma: no cover
    raise RuntimeError("FastAPI is required for API components; please install fastapi.")

from unified_pipeline.pipeline import UnifiedPipeline
from unified_pipeline.api.streaming import StreamingSession
from unified_pipeline.utils.logging import configure_logging, get_logger

configure_logging()
log = get_logger(__name__)
app = FastAPI(title="Unified Neural Pipeline", version="0.1.0")
pipeline = UnifiedPipeline()


@app.post("/api/v1/offline")
async def offline_inference(
    mixture_audio: UploadFile = File(...),
    target_sample: UploadFile = File(...),
) -> JSONResponse:
    session_id = uuid4().hex
    session_dir = Path("outputs") / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    mixture_path = session_dir / mixture_audio.filename
    target_path = session_dir / target_sample.filename

    mixture_path.write_bytes(await mixture_audio.read())
    target_path.write_bytes(await target_sample.read())

    transcripts = await pipeline.run_offline(mixture_path, target_path, session_dir)
    payload = [
        {
            "speaker": segment.speaker_label,
            "start": segment.start_s,
            "end": segment.end_s,
            "text": segment.text,
            "confidence": segment.confidence,
        }
        for segment in transcripts
    ]

    log.info("api.offline.complete", session=session_id)
    return JSONResponse(content={"session_id": session_id, "segments": payload})


@app.websocket("/api/v1/stream")
async def stream_audio(websocket: WebSocket) -> None:
    session = StreamingSession(websocket, pipeline)
    await session.run()

