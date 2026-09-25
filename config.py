"""Centralized runtime configuration for the Swahili voice agent.

All values are loaded from environment variables, or a local `.env` file
(see `.env.example`). Nothing here should be hardcoded elsewhere in the app.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Azure OpenAI: reasoning (GPT-4o via Semantic Kernel) ---
    # A separate Azure resource from Azure Speech below -- created under
    # "Azure OpenAI" in the portal, with a model deployed under a deployment
    # name (NOT the same as the model name -- see .env.example).
    azure_openai_api_key: str
    azure_openai_endpoint: str
    azure_openai_deployment: str = "gpt-4o"
    azure_openai_api_version: str = "2024-10-21"

    # --- Azure Cognitive Services Speech: speech-to-text AND text-to-speech ---
    # One resource/key covers both directions.
    azure_speech_key: str
    azure_speech_region: str

    # Swahili neural voices. Note: `sw-KE-AsiliaNeural` (sometimes quoted online)
    # does NOT exist on Azure. The real Swahili neural voices are:
    #   Kenya (sw-KE):    sw-KE-RafikiNeural (male), sw-KE-ZuriNeural (female)
    #   Tanzania (sw-TZ):  sw-TZ-DaudiNeural (male),  sw-TZ-RehemaNeural (female)
    azure_tts_voice_ke: str = "sw-KE-RafikiNeural"
    azure_tts_voice_tz: str = "sw-TZ-DaudiNeural"

    # BCP-47 locale driving both STT recognition language and TTS voice
    # selection. Azure Speech-to-Text and Azure voice names both use this
    # same region-qualified format directly ("sw-KE" / "sw-TZ").
    default_locale: str = "sw-KE"

    # --- Audio framing contract: MUST match what the client streams ---
    sample_rate: int = 16000
    frame_ms: int = 20
    sample_width_bytes: int = 2  # 16-bit PCM

    # --- VAD tuning ---
    # One webrtcvad instance per session drives two different state machines
    # from the same per-frame speech/silence decision (see agent_engine.py):
    #   - barge-in detection wants a SHORT debounce (interrupt fast)
    #   - end-of-utterance detection wants a LONGER silence window (don't
    #     cut the user off mid-sentence)
    vad_aggressiveness: int = 2  # webrtcvad 0-3; higher = more aggressive non-speech filtering
    vad_barge_in_ms: int = 200
    vad_end_of_speech_ms: int = 700

    # --- Clause buffering: how streamed LLM tokens are chunked for TTS ---
    clause_terminators: str = ".?!,"
    # Swahili is agglutinative (long words), so naive comma-splitting produces
    # short, choppy TTS fragments. Require a minimum length before flushing.
    clause_min_chars: int = 40

    @property
    def azure_tts_voice(self) -> str:
        """Pick the Azure neural voice matching the configured locale's region."""
        return self.azure_tts_voice_tz if self.default_locale.upper().endswith("TZ") else self.azure_tts_voice_ke

    @property
    def frame_bytes(self) -> int:
        """Bytes per audio frame given sample rate / sample width / frame duration."""
        return int(self.sample_rate * (self.frame_ms / 1000) * self.sample_width_bytes)


settings = Settings()
