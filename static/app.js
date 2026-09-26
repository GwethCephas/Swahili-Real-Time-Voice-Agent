"use strict";

// --- Wire protocol constants (must match config.py / agent_engine.py) ---
const SAMPLE_RATE = 16000;
const FRAME_SAMPLES = 320; // 20ms @ 16kHz
const FRAME_BYTES = FRAME_SAMPLES * 2; // 16-bit PCM
const TRAILING_SILENCE_FRAMES = 40; // ~800ms, must exceed server's 700ms VAD end-of-speech window
const PLAYBACK_LEAD_IN_SEC = 0.15;

// --- DOM refs ---
const statusEl = document.getElementById("status");
const logEl = document.getElementById("log");
const errorBanner = document.getElementById("error-banner");
const startBtn = document.getElementById("start-btn");
const micBtn = document.getElementById("mic-btn");
const textForm = document.getElementById("text-form");
const textInput = document.getElementById("text-input");
const sendBtn = document.getElementById("send-btn");

// --- Connection state ---
let ws = null;
let errorTimeout = null;

// --- Mic capture state ---
let micStream = null;
let captureCtx = null;
let processorNode = null;
let micActive = false;
let resampleState = { pos: 0, tail: new Float32Array(0) };
let pcmSampleQueue = [];

// --- Playback state ---
let playCtx = null;
let nextStartTime = 0;
let activeSources = new Set();

// --- Chat bubble state ---
let currentBubble = null; // { el, turnId }
let latestTurnId = 0;

// ------------------------------------------------------------------ //
// UI helpers
// ------------------------------------------------------------------ //

function setStatus(text, cls) {
  statusEl.textContent = text;
  statusEl.className = "status " + cls;
}

function showError(text) {
  errorBanner.textContent = text;
  errorBanner.classList.remove("hidden");
  clearTimeout(errorTimeout);
  errorTimeout = setTimeout(() => errorBanner.classList.add("hidden"), 6000);
}

function addUserBubble(text) {
  const el = document.createElement("div");
  el.className = "bubble user";
  el.textContent = text;
  logEl.appendChild(el);
  logEl.scrollTop = logEl.scrollHeight;
}

function openAgentBubble() {
  const el = document.createElement("div");
  el.className = "bubble agent empty";
  logEl.appendChild(el);
  logEl.scrollTop = logEl.scrollHeight;
  currentBubble = { el, turnId: null };
}

function appendToAgentBubble(turnId, text) {
  if (typeof turnId !== "number") return;
  if (turnId < latestTurnId) return; // stale chunk from a superseded turn
  latestTurnId = Math.max(latestTurnId, turnId);

  if (!currentBubble) openAgentBubble();
  if (currentBubble.turnId === null) currentBubble.turnId = turnId;
  if (currentBubble.turnId !== turnId) return;

  currentBubble.el.classList.remove("empty");
  currentBubble.el.textContent += text;
  logEl.scrollTop = logEl.scrollHeight;
}

function closeAgentBubble(interrupted) {
  if (currentBubble) {
    currentBubble.el.classList.remove("empty");
    if (interrupted) currentBubble.el.classList.add("interrupted");
    if (!currentBubble.el.textContent) currentBubble.el.remove();
  }
  currentBubble = null;
}

function idleStatus() {
  if (micActive) setStatus("Listening…", "status-listening");
  else setStatus("Connected", "status-connected");
}

// ------------------------------------------------------------------ //
// Playback (raw 16kHz/16-bit mono PCM, no container)
// ------------------------------------------------------------------ //

function ensurePlayCtx() {
  if (!playCtx) {
    playCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
  }
  return playCtx;
}

function playPcmChunk(arrayBuffer) {
  const ctx = ensurePlayCtx();
  const int16 = new Int16Array(arrayBuffer);
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) {
    float32[i] = int16[i] / 0x8000;
  }

  const audioBuffer = ctx.createBuffer(1, float32.length, SAMPLE_RATE);
  audioBuffer.copyToChannel(float32, 0);

  const source = ctx.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(ctx.destination);

  const startAt = Math.max(ctx.currentTime, nextStartTime);
  source.start(startAt);
  nextStartTime = startAt + audioBuffer.duration;

  activeSources.add(source);
  source.onended = () => activeSources.delete(source);
}

function stopAllPlayback() {
  activeSources.forEach((s) => {
    try {
      s.stop();
    } catch (e) {
      // already ended
    }
  });
  activeSources.clear();
  if (playCtx) {
    nextStartTime = playCtx.currentTime + PLAYBACK_LEAD_IN_SEC;
  }
}

// ------------------------------------------------------------------ //
// Mic capture: resample to 16kHz, frame into exact 640-byte chunks
// ------------------------------------------------------------------ //

function resampleTo16k(float32Input, srcRate) {
  const ratio = srcRate / SAMPLE_RATE;

  const input = new Float32Array(resampleState.tail.length + float32Input.length);
  input.set(resampleState.tail, 0);
  input.set(float32Input, resampleState.tail.length);

  const outLength = Math.max(0, Math.floor((input.length - resampleState.pos) / ratio));
  const output = new Float32Array(outLength);

  let pos = resampleState.pos;
  for (let i = 0; i < outLength; i++) {
    const idx = Math.floor(pos);
    const frac = pos - idx;
    const s0 = input[idx];
    const s1 = idx + 1 < input.length ? input[idx + 1] : s0;
    output[i] = s0 + (s1 - s0) * frac;
    pos += ratio;
  }

  const consumed = Math.floor(pos);
  resampleState.tail = input.slice(consumed);
  resampleState.pos = pos - consumed;

  return output;
}

function pushSamplesAndSendFrames(samples) {
  for (let i = 0; i < samples.length; i++) {
    let s = samples[i];
    if (s > 1) s = 1;
    else if (s < -1) s = -1;
    pcmSampleQueue.push(Math.round(s * 0x7fff));
  }

  while (pcmSampleQueue.length >= FRAME_SAMPLES) {
    const frameSamples = pcmSampleQueue.splice(0, FRAME_SAMPLES);
    // Fresh ArrayBuffer per frame -- never send a typed-array view's
    // .buffer directly, since that references the whole backing buffer
    // rather than just this 640-byte window.
    const buf = new ArrayBuffer(FRAME_BYTES);
    const view = new DataView(buf);
    for (let j = 0; j < FRAME_SAMPLES; j++) {
      view.setInt16(j * 2, frameSamples[j], true);
    }
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(buf);
    }
  }
}

async function startMic() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;

  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
  } catch (err) {
    showError("Microphone access failed: " + err.message);
    return;
  }

  captureCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
  await captureCtx.resume();

  resampleState = { pos: 0, tail: new Float32Array(0) };
  pcmSampleQueue = [];

  const source = captureCtx.createMediaStreamSource(micStream);
  processorNode = captureCtx.createScriptProcessor(4096, 1, 1);

  processorNode.onaudioprocess = (event) => {
    const input = event.inputBuffer.getChannelData(0);
    const srcRate = captureCtx.sampleRate;
    const resampled = srcRate === SAMPLE_RATE ? input : resampleTo16k(input, srcRate);
    pushSamplesAndSendFrames(resampled);
  };

  // ScriptProcessorNode only fires once connected through to a destination.
  // Route through a silent gain node so the raw mic signal is never audible
  // (unity gain here would feed TTS output back into the mic and trigger
  // the server's barge-in VAD in a feedback loop).
  const silentGain = captureCtx.createGain();
  silentGain.gain.value = 0;

  source.connect(processorNode);
  processorNode.connect(silentGain);
  silentGain.connect(captureCtx.destination);

  micActive = true;
  micBtn.textContent = "⏹ Stop Mic";
  micBtn.classList.add("active");
  setStatus("Listening…", "status-listening");
}

function stopMic() {
  if (!micActive) return;
  micActive = false;

  if (processorNode) {
    processorNode.disconnect();
    processorNode.onaudioprocess = null;
    processorNode = null;
  }
  if (micStream) {
    micStream.getTracks().forEach((t) => t.stop());
    micStream = null;
  }
  if (captureCtx) {
    captureCtx.close();
    captureCtx = null;
  }
  pcmSampleQueue = [];

  // The server's VAD needs ~700ms of trailing silence to close the final
  // utterance -- {"type":"end"} is a documented no-op, so this is the only
  // way to flush it. Frame content (not send timing) drives the VAD clock,
  // so these can go out back-to-back with no pacing.
  if (ws && ws.readyState === WebSocket.OPEN) {
    for (let i = 0; i < TRAILING_SILENCE_FRAMES; i++) {
      ws.send(new ArrayBuffer(FRAME_BYTES));
    }
  }

  micBtn.textContent = "🎤 Start Mic";
  micBtn.classList.remove("active");
  idleStatus();
}

// ------------------------------------------------------------------ //
// WebSocket wiring
// ------------------------------------------------------------------ //

function handleServerMessage(event) {
  if (typeof event.data === "string") {
    let msg;
    try {
      msg = JSON.parse(event.data);
    } catch (e) {
      return;
    }

    switch (msg.type) {
      case "transcript":
        addUserBubble(msg.text);
        break;
      case "agent_start":
        openAgentBubble();
        ensurePlayCtx();
        nextStartTime = playCtx.currentTime + PLAYBACK_LEAD_IN_SEC;
        setStatus("Agent anasema…", "status-speaking");
        break;
      case "agent_reply_chunk":
        appendToAgentBubble(msg.turn_id, msg.text);
        break;
      case "agent_end":
        closeAgentBubble(false);
        idleStatus();
        break;
      case "barge_in":
        stopAllPlayback();
        closeAgentBubble(true);
        idleStatus();
        break;
      case "error":
        showError(msg.text || "Unknown error");
        break;
      default:
        break;
    }
  } else {
    playPcmChunk(event.data);
  }
}

function connect() {
  const clientId = crypto.randomUUID();
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${protocol}://${location.host}/ws/swahili-voice/${clientId}`);
  ws.binaryType = "arraybuffer";

  ws.addEventListener("open", () => {
    setStatus("Connected", "status-connected");
    micBtn.disabled = false;
    textInput.disabled = false;
    sendBtn.disabled = false;
  });

  ws.addEventListener("message", handleServerMessage);

  ws.addEventListener("close", () => {
    setStatus("Disconnected", "status-idle");
    micBtn.disabled = true;
    textInput.disabled = true;
    sendBtn.disabled = true;
    startBtn.disabled = false;
    if (micActive) stopMic();
  });

  ws.addEventListener("error", () => {
    showError("WebSocket connection error");
  });
}

// ------------------------------------------------------------------ //
// Event listeners
// ------------------------------------------------------------------ //

startBtn.addEventListener("click", async () => {
  startBtn.disabled = true;
  connect();
  // Tied to this same user gesture so autoplay policy doesn't block it.
  ensurePlayCtx();
  if (playCtx.state === "suspended") await playCtx.resume();
});

micBtn.addEventListener("click", () => {
  if (micActive) stopMic();
  else startMic();
});

textForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = textInput.value.trim();
  if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: "text_input", text }));
  textInput.value = "";
});
