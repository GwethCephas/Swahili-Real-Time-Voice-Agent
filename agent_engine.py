"""Core streaming voice pipeline: VAD -> Whisper STT -> Semantic Kernel (GPT-4o)
-> clause-buffered Azure TTS, with barge-in (interruption) support.

One `VoiceAgentEngine` is built once at process startup (it owns the shared
Kernel, registered plugins, and API clients). Each WebSocket connection gets
its own `SessionState` so `ChatHistory`, VAD state, and in-flight tasks never
leak between concurrent users.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field

import azure.cognitiveservices.speech as speechsdk
import webrtcvad
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings
from semantic_kernel.contents import ChatHistory
from semantic_kernel.contents.streaming_chat_message_content import StreamingChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.functions import KernelArguments

from config import Settings
from plugins import SwahiliCustomerSupportPlugin

logger = logging.getLogger("swahili_voice_agent.engine")

# System prompt: instructs GPT-4o to reason AND respond in natural, fluent
# Swahili -- not a literal English-to-Swahili translation -- and to reach for
# the registered plugin functions when the user asks about orders, balances,
# or opening hours. Kept short deliberately since replies get spoken aloud.
SWAHILI_SYSTEM_PROMPT = (
    "Wewe ni msaidizi wa huduma kwa wateja unayezungumza Kiswahili fasaha. "
    "Jibu maswali yote kwa Kiswahili safi, sahihi na cha asili -- si tafsiri ya moja kwa moja "
    "kutoka Kiingereza. Tumia zana (functions) zilizopo unapohitaji taarifa za agizo, salio la "
    "akaunti, au saa za ufunguzi. Kuwa mfupi na wa kirafiki, kwa sababu majibu yako yatasomwa kwa sauti."
)


@dataclass
class SessionState:
    """Per-WebSocket-connection state. Never shared across clients."""

    client_id: str
    history: ChatHistory
    vad: webrtcvad.Vad

    # Barge-in epoch: bumped every time the user interrupts the agent. Queued
    # audio chunks are tagged with the epoch active when they were produced;
    # the sender drops any chunk whose epoch no longer matches, which is what
    # kills "ghost audio" left over from a just-cancelled reply.
    epoch: int = 0

    # Turn-ownership token: bumped at the start of every _run_turn call.
    # Cancelling a task blocked inside asyncio.to_thread (STT/TTS calls)
    # doesn't take effect until the blocking call returns, which can be
    # seconds later -- this lets a superseded turn's delayed cleanup detect
    # that it's stale and avoid clobbering a newer turn's state.
    turn_id: int = 0

    agent_speaking: bool = False
    user_speaking: bool = False

    utterance_pcm: bytearray = field(default_factory=bytearray)
    silence_ms: int = 0
    speech_ms: int = 0

    # Tuples of (kind, epoch_or_none, payload). kind is "audio" (payload:
    # bytes, epoch checked before sending) or "text" (payload: JSON string,
    # always sent -- control/status messages aren't epoch-gated).
    outbound_queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=128))

    active_turn_task: asyncio.Task | None = None
    active_synthesizer: speechsdk.SpeechSynthesizer | None = None


def _enqueue_audio(session: SessionState, epoch: int, data: bytes) -> None:
    """Called from the Azure Speech SDK's own background thread via
    call_soon_threadsafe -- must not block, so drop on backpressure rather
    than awaiting queue space.
    """
    try:
        session.outbound_queue.put_nowait(("audio", epoch, data))
    except asyncio.QueueFull:
        logger.warning("Outbound audio queue full for %s; dropping chunk", session.client_id)


# asyncio only holds a *weak* reference to tasks created via create_task --
# an unreferenced task can be garbage-collected mid-run. Fire-and-forget
# background work (e.g. barge-in's stop_speaking_async call) goes through
# here so it always runs to completion.
_background_tasks: set[asyncio.Task] = set()


def _fire_and_forget(coro) -> None:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


class VoiceAgentEngine:
    """Process-wide orchestrator: one Kernel + plugin set, many sessions."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        self.kernel = Kernel()
        self.kernel.add_plugin(SwahiliCustomerSupportPlugin(), plugin_name="CustomerSupport")

        chat_service = AzureChatCompletion(
            service_id="chat",
            api_key=settings.azure_openai_api_key,
            endpoint=settings.azure_openai_endpoint,
            deployment_name=settings.azure_openai_deployment,
            api_version=settings.azure_openai_api_version,
        )
        self.kernel.add_service(chat_service)

        self.request_settings = AzureChatPromptExecutionSettings(
            service_id="chat",
            max_tokens=600,
            temperature=0.7,
            # Exclude our own prompt-templated chat wrapper function (below) from
            # the tool list -- it has no typed parameter schema (it's a raw
            # template, not a plugin method), which the model's function-calling
            # API rejects with a schema error if it's advertised as callable.
            function_choice_behavior=FunctionChoiceBehavior.Auto(filters={"excluded_plugins": ["SwahiliVoiceAgent"]}),
        )

        # A prompt-templated kernel function is what lets `kernel.invoke_stream`
        # drive token streaming and auto function-calling together.
        self.chat_function = self.kernel.add_function(
            prompt="{{$chat_history}}{{$user_input}}",
            plugin_name="SwahiliVoiceAgent",
            function_name="Chat",
        )

    def new_session(self, client_id: str) -> SessionState:
        history = ChatHistory()
        history.add_system_message(SWAHILI_SYSTEM_PROMPT)
        vad = webrtcvad.Vad(self.settings.vad_aggressiveness)
        return SessionState(client_id=client_id, history=history, vad=vad)

    async def shutdown_session(self, session: SessionState) -> None:
        """Cleanup on disconnect: cancel in-flight work, stop any synthesis."""
        if session.active_turn_task and not session.active_turn_task.done():
            session.active_turn_task.cancel()
        synth = session.active_synthesizer
        if synth is not None:
            await asyncio.to_thread(self._safe_stop_speaking, synth)

    # ------------------------------------------------------------------ #
    # Inbound audio / VAD
    # ------------------------------------------------------------------ #

    async def handle_audio_frame(self, session: SessionState, frame: bytes) -> None:
        if len(frame) != self.settings.frame_bytes:
            logger.warning(
                "Dropping malformed frame from %s: got %d bytes, expected %d",
                session.client_id,
                len(frame),
                self.settings.frame_bytes,
            )
            return

        is_speech = session.vad.is_speech(frame, self.settings.sample_rate)

        if session.agent_speaking:
            # Barge-in detector: short debounce of continuous speech while
            # the agent is talking. Utterance-end detector below is not
            # relevant here -- there is no user utterance in progress yet.
            if is_speech:
                session.speech_ms += self.settings.frame_ms
                if session.speech_ms >= self.settings.vad_barge_in_ms:
                    self._handle_barge_in(session)
                    # The frame that triggered barge-in is the start of the
                    # user's new utterance -- don't discard it.
                    session.user_speaking = True
                    session.utterance_pcm = bytearray(frame)
                    session.silence_ms = 0
                    session.speech_ms = 0
            else:
                session.speech_ms = 0
            return

        # Normal listening mode: utterance-end detector, patient silence window.
        if is_speech:
            session.user_speaking = True
            session.silence_ms = 0
            session.utterance_pcm += frame
            return

        if session.user_speaking:
            session.silence_ms += self.settings.frame_ms
            session.utterance_pcm += frame
            if session.silence_ms >= self.settings.vad_end_of_speech_ms:
                pcm = bytes(session.utterance_pcm)
                session.utterance_pcm = bytearray()
                session.user_speaking = False
                session.silence_ms = 0
                # Guard against a fast follow-up utterance arriving before the
                # previous turn's task (e.g. a slow Whisper call) has finished --
                # otherwise the old handle is lost and it can't be cancelled.
                if session.active_turn_task and not session.active_turn_task.done():
                    session.active_turn_task.cancel()
                session.active_turn_task = asyncio.create_task(self._handle_utterance(session, pcm))

    def _handle_barge_in(self, session: SessionState) -> None:
        session.epoch += 1
        session.agent_speaking = False

        while not session.outbound_queue.empty():
            try:
                session.outbound_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        session.outbound_queue.put_nowait(("text", None, json.dumps({"type": "barge_in"})))

        if session.active_turn_task and not session.active_turn_task.done():
            session.active_turn_task.cancel()

        synth = session.active_synthesizer
        if synth is not None:
            _fire_and_forget(asyncio.to_thread(self._safe_stop_speaking, synth))

        logger.info("Barge-in on client_id=%s, epoch now %d", session.client_id, session.epoch)

    @staticmethod
    def _safe_stop_speaking(synth: speechsdk.SpeechSynthesizer) -> None:
        try:
            synth.stop_speaking_async().get()
        except Exception:
            logger.exception("Error stopping Azure synthesis")

    # ------------------------------------------------------------------ #
    # STT -> SK reasoning -> TTS, run as one cancellable task per turn
    # ------------------------------------------------------------------ #

    async def _handle_utterance(self, session: SessionState, pcm: bytes) -> None:
        try:
            transcript = await asyncio.to_thread(self._azure_transcribe, pcm)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("STT failed for client_id=%s", session.client_id)
            await self._emit_text(session, "error", "Samahani, hitilafu imetokea. Tafadhali jaribu tena.")
            return

        if not transcript.strip():
            return

        await self._emit_text(session, "transcript", transcript)
        session.history.add_user_message(transcript)
        await self._run_turn(session, transcript)

    async def handle_text_input(self, session: SessionState, text: str) -> None:
        """Entry point for typed (non-voice) input, driving the same turn
        logic as a spoken utterance. Interrupts any in-flight spoken turn
        the same way a barge-in would.
        """
        text = text.strip()
        if not text:
            return

        # A stray in-progress VAD-buffered utterance shouldn't fire its own
        # turn later and interleave with this one.
        session.utterance_pcm = bytearray()
        session.user_speaking = False
        session.silence_ms = 0
        session.speech_ms = 0

        if session.agent_speaking or (session.active_turn_task and not session.active_turn_task.done()):
            self._handle_barge_in(session)

        await self._emit_text(session, "transcript", text)
        session.history.add_user_message(text)
        session.active_turn_task = asyncio.create_task(self._run_turn(session, text))

    async def _run_turn(self, session: SessionState, user_text: str) -> None:
        """Reason over `user_text` (already appended to session.history) via
        Semantic Kernel, streaming the reply out as clause-buffered TTS audio
        plus matching agent_reply_chunk text events. Shared by both the
        voice path (_handle_utterance) and the text path (handle_text_input).
        """
        session.turn_id += 1
        my_turn = session.turn_id

        try:
            session.agent_speaking = True
            await self._emit_text(session, "agent_start", "")

            args = KernelArguments(settings=self.request_settings)
            args["user_input"] = user_text
            args["chat_history"] = session.history

            buffer = ""
            full_reply = ""
            async for chunk in self.kernel.invoke_stream(
                self.chat_function,
                return_function_results=False,
                arguments=args,
            ):
                msg = chunk[0]
                if not isinstance(msg, StreamingChatMessageContent) or msg.role != AuthorRole.ASSISTANT:
                    continue
                token = str(msg)
                if not token:
                    continue
                buffer += token
                full_reply += token
                clause, buffer = self._split_clause(buffer)
                if clause:
                    await self._synthesize_and_stream(session, clause)
                    await self._emit_reply_chunk(session, my_turn, clause)

            if buffer.strip():
                await self._synthesize_and_stream(session, buffer)
                await self._emit_reply_chunk(session, my_turn, buffer)

            if full_reply.strip():
                session.history.add_assistant_message(full_reply)

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Error handling turn for client_id=%s", session.client_id)
            await self._emit_text(session, "error", "Samahani, hitilafu imetokea. Tafadhali jaribu tena.")
        finally:
            if session.turn_id == my_turn:
                session.agent_speaking = False
                session.active_synthesizer = None
                # Not awaited: on the cancellation path (barge-in), an `await`
                # here would immediately re-raise CancelledError and the
                # agent_end notice would never actually go out.
                try:
                    session.outbound_queue.put_nowait(("text", None, json.dumps({"type": "agent_end", "text": ""})))
                except asyncio.QueueFull:
                    pass

    def _split_clause(self, buffer: str) -> tuple[str | None, str]:
        """Pull one clause off the front of `buffer` once a terminator is hit
        AND the clause is long enough to avoid choppy, too-short TTS output
        (Swahili's long agglutinative words make naive comma-splitting bad).
        """
        for i, ch in enumerate(buffer):
            if ch in self.settings.clause_terminators and len(buffer[: i + 1].strip()) >= self.settings.clause_min_chars:
                return buffer[: i + 1], buffer[i + 1 :]
        return None, buffer

    async def _emit_text(self, session: SessionState, msg_type: str, text: str) -> None:
        payload = json.dumps({"type": msg_type, "text": text})
        await session.outbound_queue.put(("text", None, payload))

    async def _emit_reply_chunk(self, session: SessionState, turn_id: int, text: str) -> None:
        # Tagged with turn_id (unlike other status messages) so the client
        # can discard chunks from a turn that got superseded by a barge-in --
        # text frames aren't epoch-gated by _sender_loop like audio is.
        payload = json.dumps({"type": "agent_reply_chunk", "text": text, "turn_id": turn_id})
        await session.outbound_queue.put(("text", None, payload))

    # ------------------------------------------------------------------ #
    # Azure Speech-to-Text (batch call per VAD-segmented utterance -- same
    # resource/key as TTS below, no separate STT provider needed)
    # ------------------------------------------------------------------ #

    def _azure_transcribe(self, pcm: bytes) -> str:
        # PushAudioInputStream defaults to 16kHz/16-bit mono PCM, matching
        # our audio contract exactly -- no WAV container needed.
        push_stream = speechsdk.audio.PushAudioInputStream()
        push_stream.write(pcm)
        push_stream.close()

        speech_config = speechsdk.SpeechConfig(
            subscription=self.settings.azure_speech_key,
            region=self.settings.azure_speech_region,
        )
        speech_config.speech_recognition_language = self.settings.default_locale

        audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        result = recognizer.recognize_once_async().get()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text
        if result.reason == speechsdk.ResultReason.NoMatch:
            return ""
        if result.reason == speechsdk.ResultReason.Canceled:
            details = result.cancellation_details
            logger.error("Azure STT canceled: %s - %s", details.reason, details.error_details)
        return ""

    # ------------------------------------------------------------------ #
    # Azure Speech TTS: push-stream synthesis, epoch-tagged chunk emission
    # ------------------------------------------------------------------ #

    async def _synthesize_and_stream(self, session: SessionState, clause: str) -> None:
        epoch = session.epoch
        loop = asyncio.get_running_loop()

        class _StreamCallback(speechsdk.audio.PushAudioOutputStreamCallback):
            def write(cb_self, audio_buffer: memoryview) -> int:
                data = bytes(audio_buffer)
                loop.call_soon_threadsafe(_enqueue_audio, session, epoch, data)
                return len(data)

            def close(cb_self) -> None:
                pass

        speech_config = speechsdk.SpeechConfig(
            subscription=self.settings.azure_speech_key,
            region=self.settings.azure_speech_region,
        )
        # sw-KE / sw-TZ neural voice, resolved from DEFAULT_LOCALE (config.py).
        speech_config.speech_synthesis_voice_name = self.settings.azure_tts_voice
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm
        )

        push_stream = speechsdk.audio.PushAudioOutputStream(_StreamCallback())
        audio_config = speechsdk.audio.AudioOutputConfig(stream=push_stream)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        session.active_synthesizer = synthesizer

        try:
            # The Speech SDK's *_async methods return SDK futures, not asyncio
            # awaitables -- run the blocking .get() in a worker thread so the
            # event loop stays free to keep handling incoming audio frames
            # (including a barge-in that would call stop_speaking_async()).
            result = await asyncio.to_thread(lambda: synthesizer.speak_text_async(clause).get())
            if result.reason == speechsdk.ResultReason.Canceled:
                details = result.cancellation_details
                if details.reason == speechsdk.CancellationReason.Error:
                    logger.error("Azure TTS error for client_id=%s: %s", session.client_id, details.error_details)
        finally:
            if session.active_synthesizer is synthesizer:
                session.active_synthesizer = None
