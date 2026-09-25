"""FastAPI server exposing the real-time Swahili voice agent over WebSockets.

Audio contract (must match the client -- see config.py):
  - Client -> server: 16kHz, 16-bit mono PCM, 20ms frames, as WebSocket
    BINARY frames. WebSocket TEXT frames are reserved for JSON control
    messages, e.g. {"type": "end"}.
  - Server -> client: Azure-synthesized 16kHz, 16-bit mono PCM, as WebSocket
    BINARY frames (chunked as produced). JSON TEXT frames carry status:
    {"type": "transcript", "text": "..."}, {"type": "agent_start"},
    {"type": "agent_end"}, {"type": "barge_in"}, {"type": "error", ...}.
"""
from __future__ import annotations

import asyncio
import json
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from agent_engine import SessionState, VoiceAgentEngine
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("swahili_voice_agent.main")

app = FastAPI(title="Swahili Real-Time Voice Agent")

# Built once at startup: shared Kernel, plugins, and API clients. Per-session
# state (ChatHistory, VAD, queues) is created fresh for every connection.
engine: VoiceAgentEngine | None = None


@app.on_event("startup")
async def on_startup() -> None:
    global engine
    engine = VoiceAgentEngine(settings)
    logger.info("Swahili voice agent engine ready (locale=%s)", settings.default_locale)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/ws/swahili-voice/{client_id}")
async def swahili_voice_ws(websocket: WebSocket, client_id: str) -> None:
    assert engine is not None, "engine not initialized -- startup event did not run"

    await websocket.accept()
    session = engine.new_session(client_id)
    sender_task = asyncio.create_task(_sender_loop(websocket, session))
    logger.info("Connection opened for client_id=%s", client_id)

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            audio_bytes = message.get("bytes")
            if audio_bytes is not None:
                await engine.handle_audio_frame(session, audio_bytes)
                continue

            control_text = message.get("text")
            if control_text is not None:
                _handle_control_message(session, control_text)

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Unhandled error on connection client_id=%s", client_id)
    finally:
        await engine.shutdown_session(session)
        sender_task.cancel()
        logger.info("Connection closed for client_id=%s", client_id)


async def _sender_loop(websocket: WebSocket, session: SessionState) -> None:
    """Single writer for this connection's outbound WebSocket frames.
    Audio chunks are dropped if their epoch is stale (superseded by a
    barge-in); control/status text messages are always delivered.
    """
    try:
        while True:
            kind, epoch, payload = await session.outbound_queue.get()
            if kind == "audio":
                if epoch == session.epoch:
                    await websocket.send_bytes(payload)
            elif kind == "text":
                await websocket.send_text(payload)
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception("Sender loop failed for client_id=%s", session.client_id)


def _handle_control_message(session: SessionState, raw: str) -> None:
    try:
        message = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Ignoring malformed control message from client_id=%s", session.client_id)
        return

    msg_type = message.get("type")
    if msg_type == "start":
        logger.debug("Client %s signalled stream start", session.client_id)
    elif msg_type == "end":
        # VAD silence detection normally closes an utterance on its own;
        # this is a hook for push-to-talk style clients that want to force
        # it, left as a no-op here since it's outside the requested scope.
        logger.debug("Client %s signalled stream end", session.client_id)
    else:
        logger.debug("Unhandled control message type '%s' from %s", msg_type, session.client_id)
