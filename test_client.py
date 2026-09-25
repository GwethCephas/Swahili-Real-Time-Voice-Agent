"""Manual end-to-end test client for the Swahili voice agent WebSocket endpoint.

Streams a 16kHz/16-bit mono PCM WAV file to the server in 20ms frames
(matching the audio contract in config.py / main.py), prints any JSON
control messages received (transcript, agent_start, agent_end, barge_in,
error), and writes the agent's synthesized reply back out to a WAV file.

This is a dev/debugging tool, not part of the deployed service.

Usage:
    python test_client.py path/to/swahili_input.wav
    python test_client.py input.wav --out reply.wav --url ws://127.0.0.1:8000/ws/swahili-voice/test1

Input WAV must already be 16kHz, 16-bit, mono PCM. Convert with ffmpeg if needed:
    ffmpeg -i input.mp3 -ar 16000 -ac 1 -sample_fmt s16 input.wav
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import wave

import websockets

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
FRAME_MS = 20
FRAME_BYTES = int(SAMPLE_RATE * (FRAME_MS / 1000) * SAMPLE_WIDTH)


def load_pcm_frames(path: str) -> list[bytes]:
    with wave.open(path, "rb") as wf:
        if wf.getframerate() != SAMPLE_RATE or wf.getsampwidth() != SAMPLE_WIDTH or wf.getnchannels() != 1:
            raise ValueError(
                f"{path} must be {SAMPLE_RATE}Hz, 16-bit, mono PCM "
                f"(got {wf.getframerate()}Hz, {wf.getsampwidth() * 8}-bit, {wf.getnchannels()}ch). "
                f"Convert with: ffmpeg -i {path} -ar {SAMPLE_RATE} -ac 1 -sample_fmt s16 fixed.wav"
            )
        pcm = wf.readframes(wf.getnframes())

    frames = [pcm[i : i + FRAME_BYTES] for i in range(0, len(pcm), FRAME_BYTES)]
    if frames and len(frames[-1]) < FRAME_BYTES:
        frames[-1] = frames[-1] + b"\x00" * (FRAME_BYTES - len(frames[-1]))
    return frames


async def run(url: str, wav_path: str, out_path: str) -> None:
    frames = load_pcm_frames(wav_path)
    print(f"Loaded {len(frames)} frames (~{len(frames) * FRAME_MS / 1000:.1f}s) from {wav_path}")

    received_audio = bytearray()

    async with websockets.connect(url, max_size=None) as ws:
        print(f"Connected to {url}")

        async def sender() -> None:
            for frame in frames:
                await ws.send(frame)
                await asyncio.sleep(FRAME_MS / 1000)
            # Trailing silence so the server's VAD can close the utterance
            # (vad_end_of_speech_ms in config.py, default 700ms).
            silence = b"\x00" * FRAME_BYTES
            for _ in range(40):  # ~800ms
                await ws.send(silence)
                await asyncio.sleep(FRAME_MS / 1000)

        async def receiver() -> None:
            async for message in ws:
                if isinstance(message, (bytes, bytearray)):
                    received_audio.extend(message)
                    continue
                try:
                    payload = json.loads(message)
                except json.JSONDecodeError:
                    print("RAW:", message)
                    continue
                print(f"[{payload.get('type')}] {payload.get('text', '')}")
                if payload.get("type") == "agent_end":
                    break

        sender_task = asyncio.create_task(sender())
        try:
            await asyncio.wait_for(receiver(), timeout=60)
        except asyncio.TimeoutError:
            print("Timed out waiting for a reply (60s) -- check the server log.")
        finally:
            sender_task.cancel()

    if received_audio:
        with wave.open(out_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(bytes(received_audio))
        print(f"Wrote {len(received_audio)} bytes of reply audio to {out_path}")
    else:
        print("No audio received back from the agent.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("wav_path", help="Path to a 16kHz/16-bit mono PCM WAV file with Swahili speech")
    parser.add_argument("--url", default="ws://127.0.0.1:8000/ws/swahili-voice/test-client")
    parser.add_argument("--out", default="reply.wav", help="Where to save the agent's spoken reply")
    args = parser.parse_args()

    try:
        asyncio.run(run(args.url, args.wav_path, args.out))
    except FileNotFoundError:
        print(f"Input file not found: {args.wav_path}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    except OSError as exc:
        print(f"Could not connect to {args.url}: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
