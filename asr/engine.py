"""Streaming ASR inference engine.

Implements "accumulated re-encoding + prefix rollback" within fixed-length segments.

The OpenVINO encoder accepts a fixed [1,128,800] mel input (~8s of audio).
For long streams, audio is processed in ~8s segments: within each segment the
standard Qwen3-ASR strategy (accumulate audio, re-encode, prefix rollback) is
used. When a segment fills the encoder window, the transcription is committed
and a fresh segment starts.

Core loop per chunk:
  1. Accumulate new audio into audio_accum
  2. Compute mel spectrogram for accumulated audio
  3. Run encoder -> audio features [1, 104, 1024]
  4. Build prompt tokens (system + user + assistant prefix)
  5. Build inputs_embeds (text embeddings + audio features)
  6. Run decoder: reset KV-cache, prefill, greedy decode
  7. Parse output, apply prefix rollback for next chunk
  8. If segment full, commit text and reset for next segment
"""

from dataclasses import dataclass, field

import numpy as np
from transformers import AutoTokenizer

from .config import (
    HF_MODEL_DIR,
    SAMPLE_RATE,
    AUDIO_PAD_COUNT,
    IM_START,
    IM_END,
    AUDIO_START,
    AUDIO_END,
    AUDIO_PAD,
    ASR_TEXT,
    NEWLINE,
    CHUNK_SIZE_SEC,
    UNFIXED_CHUNK_NUM,
    UNFIXED_TOKEN_NUM,
    MAX_NEW_TOKENS,
    AUDIO_WINDOW_SAMPLES,
)
from .ov_encoder import OVEncoder
from .ov_decoder import OVDecoder
from .processor import MelProcessor


@dataclass
class StreamingState:
    """Mutable state for one streaming ASR session."""
    chunk_id: int = 0
    buffer: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    audio_accum: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    language: str = ""
    text: str = ""
    _raw_decoded: str = ""
    _committed_text: str = ""   # Text confirmed from prefix overflow


class ASREngine:
    """Streaming ASR engine using OpenVINO.

    Usage:
        engine = ASREngine()
        state = engine.new_session()
        # Feed PCM chunks as they arrive:
        engine.feed(pcm_chunk, state)
        # Read state.text for current transcription
        # When done:
        engine.finish(state)
    """

    def __init__(
        self,
        encoder_device: str = "NPU",
        decoder_device: str = "NPU",
        chunk_size_sec: float = CHUNK_SIZE_SEC,
        unfixed_chunk_num: int = UNFIXED_CHUNK_NUM,
        unfixed_token_num: int = UNFIXED_TOKEN_NUM,
        max_new_tokens: int = MAX_NEW_TOKENS,
        language: str | None = None,
    ):
        self.chunk_size_sec = chunk_size_sec
        self.chunk_size_samples = int(round(chunk_size_sec * SAMPLE_RATE))
        self.unfixed_chunk_num = unfixed_chunk_num
        self.unfixed_token_num = unfixed_token_num
        self.max_new_tokens = max_new_tokens
        self.language = language

        self._mel = MelProcessor()
        self._encoder = OVEncoder(device=encoder_device)
        self._decoder = OVDecoder(device=decoder_device)
        self._tokenizer = AutoTokenizer.from_pretrained(
            HF_MODEL_DIR, trust_remote_code=True, fix_mistral_regex=False
        )

        # Pre-compute prompt token sequences
        self._system_tokens = self._build_system_tokens()
        self._user_prefix_tokens, self._user_suffix_tokens = self._build_user_tokens()
        self._assistant_tokens = self._build_assistant_tokens()

    def _build_system_tokens(self) -> list[int]:
        return [IM_START] + self._tokenizer.encode("system\nYou are a helpful assistant.") + [IM_END, NEWLINE]

    def _build_user_tokens(self) -> tuple[list[int], list[int]]:
        prefix = [IM_START] + self._tokenizer.encode("user\n") + [AUDIO_START]
        suffix = [AUDIO_END, IM_END, NEWLINE]
        return prefix, suffix

    def _build_assistant_tokens(self) -> list[int]:
        tokens = [IM_START] + self._tokenizer.encode("assistant\n")
        if self.language:
            lang_text = f"language {self.language}"
            tokens += self._tokenizer.encode(lang_text) + [ASR_TEXT]
        return tokens

    def _build_prompt_tokens(self, prefix_text: str = "") -> list[int]:
        """Build full prompt token sequence with optional prefix text."""
        tokens = (
            self._system_tokens
            + self._user_prefix_tokens
            + [AUDIO_PAD] * AUDIO_PAD_COUNT
            + self._user_suffix_tokens
            + self._assistant_tokens
        )
        if prefix_text:
            tokens += self._tokenizer.encode(prefix_text)
        return tokens

    def _build_inputs_embeds(
        self, token_ids: list[int], audio_features: np.ndarray
    ) -> np.ndarray:
        """Build inputs_embeds by merging text embeddings with audio features.

        Args:
            token_ids: Full prompt token IDs (including AUDIO_PAD placeholders).
            audio_features: Encoder output, shape [1, 104, 1024].

        Returns:
            inputs_embeds, shape [seq_len, 1024].
        """
        ids = np.array(token_ids, dtype=np.int64)
        embeds = self._decoder.embed_table[ids]  # [seq_len, 1024]

        # Replace audio_pad positions with encoder features
        audio_positions = np.where(ids == AUDIO_PAD)[0]
        embeds[audio_positions] = audio_features[0]  # [104, 1024]

        return embeds

    def _compute_prefix(self, state: StreamingState) -> str:
        """Compute prefix text with rollback for streaming continuity."""
        if state.chunk_id < self.unfixed_chunk_num or not state._raw_decoded:
            return ""

        cur_ids = self._tokenizer.encode(state._raw_decoded)
        k = self.unfixed_token_num

        while True:
            end_idx = max(0, len(cur_ids) - k)
            if end_idx == 0:
                return ""
            prefix = self._tokenizer.decode(cur_ids[:end_idx])
            if "\ufffd" not in prefix:
                return prefix
            k += 1

    def _run_chunk(self, state: StreamingState):
        """Run one inference step on accumulated audio."""
        # 1. Compute mel spectrogram for all accumulated audio
        mel = self._mel(state.audio_accum)

        # 2. Run encoder
        audio_features = self._encoder(mel)

        # 3. Build prefix with rollback
        prefix = self._compute_prefix(state)

        # 4. Build prompt tokens and inputs_embeds
        token_ids = self._build_prompt_tokens(prefix)
        inputs_embeds = self._build_inputs_embeds(token_ids, audio_features)

        # 5. Run decoder (reset + prefill + decode)
        generated_ids = self._decoder.generate(inputs_embeds, self.max_new_tokens)

        # 6. Decode generated tokens
        gen_text = self._tokenizer.decode(generated_ids, skip_special_tokens=False)

        # 7. Update state
        state._raw_decoded = (prefix + gen_text) if prefix else gen_text
        lang, recent_text = self._parse_output(state._raw_decoded)
        if lang:
            state.language = lang
        state.text = state._committed_text + recent_text
        state.chunk_id += 1

    def _parse_output(self, raw: str) -> tuple[str, str]:
        """Parse raw decoder output into (language, text).

        Handles formats:
          - "language Chinese<asr_text>..." -> ("Chinese", "...")
          - "language None<asr_text>" -> ("", "")
          - plain text -> ("", text)
        """
        if not raw:
            return "", ""

        s = raw.strip()
        asr_tag = "<asr_text>"

        if asr_tag in s:
            meta, text = s.split(asr_tag, 1)
            meta_lower = meta.lower()
            if "language none" in meta_lower:
                return "", text.strip() if text.strip() else ""
            # Extract language name
            lang = ""
            for line in meta.splitlines():
                line = line.strip().lower()
                if line.startswith("language "):
                    val = line[len("language "):].strip()
                    if val:
                        lang = val.title()
                    break
            return lang, text.strip()

        return "", s

    def new_session(self) -> StreamingState:
        """Create a new streaming session state."""
        return StreamingState()

    def _commit_segment(self, state: StreamingState):
        """Commit current segment's transcription and reset for the next segment.

        Called when audio_accum fills the encoder window (~8s).  The decoded
        text is appended to _committed_text and the per-segment state (audio
        buffer, raw decoded text, chunk counter) is cleared so the next audio
        starts a fresh segment.
        """
        _, recent_text = self._parse_output(state._raw_decoded)
        if recent_text:
            state._committed_text += recent_text
        state._raw_decoded = ""
        state.audio_accum = np.zeros(0, dtype=np.float32)
        state.chunk_id = 0
        state.text = state._committed_text

    def feed(self, pcm: np.ndarray, state: StreamingState):
        """Feed PCM audio samples to the streaming engine.

        Buffers samples and runs inference whenever a full chunk is ready.
        When accumulated audio fills the encoder window (~8s), the current
        transcription is committed and a new segment begins.

        Args:
            pcm: 1D float32 PCM at 16kHz.
            state: Streaming state from new_session().
        """
        x = np.asarray(pcm, dtype=np.float32).ravel()
        if x.shape[0] > 0:
            state.buffer = np.concatenate([state.buffer, x])

        # Consume full chunks
        while state.buffer.shape[0] >= self.chunk_size_samples:
            chunk = state.buffer[:self.chunk_size_samples]
            state.buffer = state.buffer[self.chunk_size_samples:]

            # Accumulate audio
            if state.audio_accum.shape[0] == 0:
                state.audio_accum = chunk
            else:
                state.audio_accum = np.concatenate([state.audio_accum, chunk])

            self._run_chunk(state)

            # When audio fills encoder window, commit and start new segment
            if state.audio_accum.shape[0] >= AUDIO_WINDOW_SAMPLES:
                self._commit_segment(state)

    def finish(self, state: StreamingState):
        """Flush remaining buffered audio and run final inference.

        Args:
            state: Streaming state.
        """
        if state.buffer.shape[0] == 0:
            return

        tail = state.buffer
        state.buffer = np.zeros(0, dtype=np.float32)

        if state.audio_accum.shape[0] == 0:
            state.audio_accum = tail
        else:
            state.audio_accum = np.concatenate([state.audio_accum, tail])

        self._run_chunk(state)
