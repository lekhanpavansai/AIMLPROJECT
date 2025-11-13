from __future__ import annotations

# pyright: reportMissingImports=false

import base64
import json
from typing import Optional

try:  # pragma: no cover - optional dependency guard for static analysis
    from fastapi import WebSocket, WebSocketDisconnect
except ImportError:  # pragma: no cover
    raise RuntimeError("FastAPI is required for streaming API; please install fastapi.")

try:  # pragma: no cover - optional dependency guard
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from unified_pipeline.pipeline import UnifiedPipeline, StreamingContext, StreamingChunkResult
from unified_pipeline.core_types import AudioFrame
from unified_pipeline.utils.logging import get_logger

log = get_logger(__name__)


class StreamingSession:
    """WebSocket session handler for streaming inference."""

    def __init__(self, websocket: WebSocket, pipeline: Optional[UnifiedPipeline] = None) -> None:
        self.websocket = websocket
        self.pipeline = pipeline or UnifiedPipeline()
        self.context: Optional[StreamingContext] = None

    async def run(self) -> None:
        await self.websocket.accept()
        try:
            while True:
                raw_message = await self.websocket.receive_text()
                data = json.loads(raw_message)
                event = data.get("event")

                if event == "start":
                    await self._handle_start(data)
                elif event == "audio_chunk":
                    await self._handle_audio_chunk(data)
                elif event == "stop":
                    await self._handle_stop()
                    break
                else:
                    await self._send_error(f"Unknown event '{event}'")
        except WebSocketDisconnect:
            log.info("streaming.disconnected")
        except Exception as exc:  # pragma: no cover - safety net
            log.exception("streaming.error", error=str(exc))
            await self._send_error(f"Internal error: {exc}")
            await self.websocket.close()

    async def _handle_start(self, data: dict) -> None:
        session_id = data.get("session_id") or "stream-session"
        sample_rate = int(data.get("sample_rate", 16000))
        target_b64 = data.get("target_sample_b64")

        if not target_b64:
            await self._send_error("Missing 'target_sample_b64' in start event.")
            return

        target_bytes = base64.b64decode(target_b64)
        dtype = data.get("dtype", "float32")
        channels = int(data.get("channels", 1))

        samples = self._decode_samples(target_bytes, dtype, channels)
        target_frame = AudioFrame(samples=samples, sample_rate=sample_rate, channels=channels)

        self.context = await self.pipeline.create_stream_context(session_id, target_frame)
        await self.websocket.send_json({"event": "started", "session_id": session_id})

    async def _handle_audio_chunk(self, data: dict) -> None:
        if not self.context:
            await self._send_error("Stream not initialized. Send 'start' event first.")
            return

        chunk_b64 = data.get("chunk_b64")
        if not chunk_b64:
            await self._send_error("Missing 'chunk_b64' in audio_chunk event.")
            return

        dtype = data.get("dtype", "float32")
        channels = int(data.get("channels", 1))
        chunk_bytes = base64.b64decode(chunk_b64)
        samples = self._decode_samples(chunk_bytes, dtype, channels)

        frame = AudioFrame(samples=samples, sample_rate=self.context.sample_rate, channels=channels)
        result = await self.pipeline.process_stream_chunk(self.context, frame)
        await self.websocket.send_json(self._serialize_chunk_result(result))

    async def _handle_stop(self) -> None:
        if not self.context:
            await self.websocket.send_json({"event": "stopped", "segments": []})
            return

        await self.websocket.send_json(
            {
                "event": "stopped",
                "segments": [self._segment_to_dict(segment) for segment in self.context.transcripts],
            }
        )
        self.context = None

    async def _send_error(self, message: str) -> None:
        await self.websocket.send_json({"event": "error", "message": message})

    @staticmethod
    def _decode_samples(raw: bytes, dtype: str, channels: int) -> np.ndarray:
        if dtype != "float32":
            raise ValueError(f"Unsupported dtype '{dtype}'. Expected 'float32'.")

        if np is None:
            raise RuntimeError("numpy is required for streaming inference; please install numpy.")

        array = np.frombuffer(raw, dtype=np.float32)
        if channels > 1:
            array = array.reshape(-1, channels)
        return array

    def _serialize_chunk_result(self, result: StreamingChunkResult) -> dict:
        return {
            "event": "chunk_result",
            "chunk_index": result.chunk_index,
            "chunk_start": result.chunk_start,
            "chunk_duration": result.chunk_duration,
            "segments": [self._segment_to_dict(segment) for segment in result.segments],
        }

    @staticmethod
    def _segment_to_dict(segment) -> dict:
        return {
            "speaker": segment.speaker_label,
            "start": segment.start_s,
            "end": segment.end_s,
            "text": segment.text,
            "confidence": segment.confidence,
            **({"language": segment.language} if segment.language else {}),
            **segment.extra,
        }


