# Swahili Real-Time Voice Agent

A real-time, bidirectional Swahili (Kiswahili) voice agent over WebSockets, built with FastAPI and Microsoft Semantic Kernel. Speak Swahili in, hear a fluent Swahili reply back out, with mid-reply interruption (barge-in) support.

## Architecture

```
Client (binary PCM frames + JSON control msgs)
   |  ws: /ws/swahili-voice/{client_id}
   v
FastAPI WebSocket handler (main.py)
   |  creates SessionState (per-connection, isolated)
   v
VAD-gated audio buffer (agent_engine.py)
   |  - fast threshold -> barge-in signal (flush TTS mid-speech)
   |  - patient threshold -> utterance-end -> fires STT
   v
Azure Speech-to-Text (batch call per VAD-segmented utterance)
   v
Semantic Kernel streaming chat (Azure OpenAI, ChatHistory per session,
SwahiliCustomerSupportPlugin registered for tool calls)
   |  token stream buffered into clauses (.?!, + min-char floor)
   v
Azure Speech Text-to-Speech (streamed per clause, sw-KE/sw-TZ neural voice)
   |  epoch-tagged output chunks
   v
Outbound queue -> single sender task -> websocket.send_bytes
```

**Stack:** Azure Speech (STT + TTS), Azure OpenAI (reasoning via Semantic Kernel), FastAPI + WebSockets, `webrtcvad` for voice activity detection.

## Project layout

| File | Purpose |
|---|---|
| `config.py` | All runtime configuration, loaded from `.env` |
| `plugins.py` | `SwahiliCustomerSupportPlugin` -- Semantic Kernel native functions (order status, account balance, opening hours) |
| `agent_engine.py` | Core pipeline: VAD, session state, STT, SK reasoning + clause buffering, TTS, barge-in |
| `main.py` | FastAPI app and the `/ws/swahili-voice/{client_id}` WebSocket route |
| `test_client.py` | Dev tool: streams a WAV file to the running server and saves the spoken reply |
| `requirements.txt` | Pinned dependencies |

## Setup

### 1. Install dependencies

Requires Python 3.10+.

```
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. Create the required Azure resources

Two **separate** Azure resources are needed -- don't confuse them, their keys are not interchangeable.

**Azure OpenAI** (reasoning):
1. Azure Portal -> Create a resource -> "Azure OpenAI"
2. Once created, open the resource -> **"Go to Azure AI Foundry portal"** (this scopes Foundry to exactly this resource, avoiding ambiguity if you have more than one)
3. In Foundry: **Deployments** -> **"+ Deploy model"** -> **"Deploy base model"** -> pick a chat model (`gpt-4o`, `gpt-4o-mini`, `gpt-4.1-mini`, etc. -- whatever's available in your region/quota)
4. Give it a deployment name (can be anything -- note it down, it is **not** the same as the model name)
5. Back on the resource's **"Keys and Endpoint"** page, copy the endpoint and a key

**Azure Speech** (STT + TTS, one resource covers both):
1. Azure Portal -> Create a resource -> "Speech service"
2. Free tier (F0) is available and sufficient for testing
3. Resource -> **"Keys and Endpoint"** -> copy "KEY 1" and the region

### 3. Configure `.env`

Create a `.env` file in the project root:

```
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
AZURE_OPENAI_API_VERSION=2024-10-21

AZURE_SPEECH_KEY=...
AZURE_SPEECH_REGION=eastus

AZURE_TTS_VOICE_KE=sw-KE-RafikiNeural
AZURE_TTS_VOICE_TZ=sw-TZ-DaudiNeural
DEFAULT_LOCALE=sw-KE
```

`.env` is git-ignored -- never commit it.

### 4. Run

```
uvicorn main:app --reload
```

`GET /healthz` should return `{"status": "ok"}` once it's up.

## Testing

`test_client.py` streams a WAV file to the running server the same way a real client would, and saves the agent's spoken reply:

```
python test_client.py path\to\swahili_speech.wav
```

Input must be 16kHz, 16-bit, mono PCM WAV:

```
ffmpeg -i your_recording.m4a -ar 16000 -ac 1 -sample_fmt s16 swahili_speech.wav
```

It prints every control message the server sends (`transcript`, `agent_start`, `agent_end`, `barge_in`, `error`) and writes the reply audio to `reply.wav` (or wherever `--out` points).

## Audio contract

- **Client -> server:** 16kHz, 16-bit signed little-endian PCM, mono, 20ms frames (640 bytes), as WebSocket **binary** frames. WebSocket **text** frames carry JSON control messages.
- **Server -> client:** Azure-synthesized 16kHz, 16-bit mono PCM, as binary frames, chunked as produced. JSON text frames carry status (`transcript`, `agent_start`, `agent_end`, `barge_in`, `error`).

## Notes and known constraints

- **STT is batch, not token-streaming.** Azure Speech-to-Text runs once per VAD-segmented utterance (roughly: speech until ~700ms of silence), not as a continuous stream. Expect turn latency on the order of utterance length + ~1-2s, not sub-200ms.
- **Swahili voice names:** Azure's real Swahili neural voices are `sw-KE-RafikiNeural` / `sw-KE-ZuriNeural` (Kenya) and `sw-TZ-DaudiNeural` / `sw-TZ-RehemaNeural` (Tanzania), configurable via `.env`.
- **Barge-in** works via an epoch counter: interrupting the agent bumps `session.epoch`, cancels the in-flight reasoning/TTS task, calls Azure's `stop_speaking_async()`, and drops any already-queued audio chunks tagged with a stale epoch.
- **Session isolation:** `ChatHistory`, VAD state, and in-flight tasks are all per-WebSocket-connection (`SessionState`). The Semantic Kernel `Kernel` and registered plugin are process-wide and shared.
