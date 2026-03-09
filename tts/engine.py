"""Non-streaming TTS engine -- connects talker, code predictor, and decoder."""

from __future__ import annotations

import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Generator

import numpy as np

from .config import (
    TTSModelConfig,
    CODEC_EOS_ID,
    DECODER_CHUNK_SIZE,
    DECODER_DECODE_UPSAMPLE_RATE,
    DECODER_LEFT_CONTEXT,
    REPETITION_PENALTY,
)
from .ov_code_predictor import OVCodePredictor
from .ov_decoder import OVSpeechDecoder
from .ov_talker import OVTalker
from .text_conditioner import TextConditioner


class TTSEngine:
    """TTS engine connecting all pipeline stages.

    Orchestrates:
        TextConditioner   (CPU)  -- tokenize text, build prefill embeddings
        OVTalker          (NPU)  -- 28-layer AR LLM producing layer-0 codec tokens
        OVCodePredictor   (CPU)  -- 5-layer transformer predicting residual codes 1-15
        OVSpeechDecoder   (NPU)  -- CNN speech decoder converting codes to waveform
    """

    def __init__(self, config: TTSModelConfig) -> None:
        """Initialize all pipeline components.

        Args:
            config: A ``TTSModelConfig`` instance holding all model paths
                    and device/compilation settings.
        """
        self.text_cond = TextConditioner(
            hf_model_dir=config.hf_model_dir,
            text_embedding_npy=config.text_embedding_npy,
            text_projection_npz=config.text_projection_npz,
            talker_embed_tokens_npy=config.talker_embed_tokens_npy,
        )
        self.talker = OVTalker(
            talker_xml=config.talker_xml,
            lm_head_pinv_npy=config.lm_head_pinv_npy,
            npu_config=config.npu_talker_config,
            device=config.talker_device,
        )
        self.code_predictor = OVCodePredictor(
            cp_xml=config.cp_xml,
            embeds_npz=config.cp_embeds_npz,
            lm_heads_npz=config.cp_lm_heads_npz,
            proj_in_npz=config.cp_proj_in_npz,
            device=config.cp_device,
        )
        self.decoder = OVSpeechDecoder(
            decoder_xml=config.decoder_xml,
            device=config.decoder_device,
        )

        # Talker codec embedding table [3072, 1024] -- used for layer-0 lookup
        self.codec_embed_table = np.load(
            config.talker_embed_tokens_npy
        ).astype(np.float32)

        # Code predictor embedding tables for layers 1-15, each [2048, 1024]
        cp_data = np.load(config.cp_embeds_npz)
        self.cp_embeds = {
            f"emb_{i}": cp_data[f"emb_{i}"].astype(np.float32)
            for i in range(15)
        }

        self._decoder_pool = ThreadPoolExecutor(max_workers=1)

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _suppress_tokens(self, logits: np.ndarray) -> np.ndarray:
        """Apply token suppression for talker output.

        Suppresses the special codec range [2048, 3072) except for the
        CODEC_EOS token (2150) which must remain valid for stopping.

        Args:
            logits: float32 array of shape [1, 1, 3072].

        Returns:
            Modified logits with suppressed tokens set to -inf.
        """
        saved_eos = logits[0, 0, CODEC_EOS_ID].copy()
        logits[0, 0, 2048:3072] = -np.inf
        logits[0, 0, CODEC_EOS_ID] = saved_eos
        return logits

    def _sample_token(
        self,
        logits: np.ndarray,
        temperature: float,
        top_k: int,
        top_p: float,
        generated_tokens: list[int],
    ) -> int:
        """Sample a token with suppression, repetition penalty, top-k, top-p.

        Args:
            logits:           float32 array of shape [1, 1, 3072].
            temperature:      Sampling temperature.
            top_k:            Number of top candidates.
            top_p:            Nucleus sampling threshold.
            generated_tokens: Previously generated token IDs (for repetition penalty).

        Returns:
            Sampled token index.
        """
        # 1. Suppress tokens
        logits = self._suppress_tokens(logits.copy())

        # 2. Repetition penalty
        for token_id in set(generated_tokens):
            if logits[0, 0, token_id] > 0:
                logits[0, 0, token_id] /= REPETITION_PENALTY
            else:
                logits[0, 0, token_id] *= REPETITION_PENALTY

        # 3. Squeeze to 1D
        logits_1d = logits[0, 0].astype(np.float64)

        # 4. Temperature
        logits_1d = logits_1d / temperature

        # 5. Top-k: keep only top_k highest logits, set rest to -inf
        if 0 < top_k < len(logits_1d):
            top_k_indices = np.argpartition(logits_1d, -top_k)[-top_k:]
            mask = np.full_like(logits_1d, -np.inf)
            mask[top_k_indices] = logits_1d[top_k_indices]
            logits_1d = mask

        # 6. Top-p: sort by probability, cumsum, mask beyond threshold
        if top_p < 1.0:
            sorted_indices = np.argsort(logits_1d)[::-1]
            sorted_logits = logits_1d[sorted_indices]
            # Stable softmax for cumulative probability computation
            sorted_logits_shifted = sorted_logits - np.max(sorted_logits)
            probs_sorted = np.exp(sorted_logits_shifted)
            probs_sorted = probs_sorted / probs_sorted.sum()
            cumulative_probs = np.cumsum(probs_sorted)
            # Mask tokens whose cumulative probability exceeds top_p
            sorted_mask = cumulative_probs > top_p
            # Shift right so the first token crossing the threshold is kept
            sorted_mask[1:] = sorted_mask[:-1].copy()
            sorted_mask[0] = False
            logits_1d[sorted_indices[sorted_mask]] = -np.inf

        # 7. Softmax
        logits_1d -= np.max(logits_1d)
        probs = np.exp(logits_1d)
        probs = probs / probs.sum()

        # 8. Sample
        token = np.random.choice(len(probs), p=probs)
        return int(token)

    # ------------------------------------------------------------------
    # Async decoder helper
    # ------------------------------------------------------------------

    def _async_decode(
        self, decoder_input: np.ndarray, chunk_idx: int
    ) -> tuple[int, np.ndarray]:
        """Decode a chunk in a background thread.

        Args:
            decoder_input: int64 array of shape [1, 16, 75].
            chunk_idx:     Index of this chunk for ordering.

        Returns:
            (chunk_idx, wav_new) where wav_new has left-context audio stripped.
        """
        wav = self.decoder.decode_chunk(decoder_input)
        wav_new = wav[DECODER_LEFT_CONTEXT * DECODER_DECODE_UPSAMPLE_RATE:]
        return (chunk_idx, wav_new)

    # ------------------------------------------------------------------
    # Timing summary
    # ------------------------------------------------------------------

    @staticmethod
    def _print_timing(
        t_sample: float,
        t_cp: float,
        t_embed: float,
        t_talker: float,
        t_decoder: float,
        num_steps: int,
    ) -> None:
        """Print a per-component timing breakdown to stderr.

        Args:
            t_sample:  Total time in _sample_token (seconds).
            t_cp:      Total time in code_predictor.predict (seconds).
            t_embed:   Total time building next_embed (seconds).
            t_talker:  Total time in talker.generate_step (seconds).
            t_decoder: Total time in decoder.decode_chunk (seconds).
            num_steps: Number of AR generation steps completed.
        """
        total = t_talker + t_cp + t_decoder + t_sample + t_embed
        if num_steps == 0 or total == 0:
            return
        print(
            f"--- TTS Timing Summary ---\n"
            f"Steps: {num_steps}\n"
            f"  Talker:    {t_talker*1000:.0f} ms  ({t_talker/total*100:.1f}%)"
            f"  avg {t_talker/num_steps*1000:.1f} ms/step\n"
            f"  CodePred:  {t_cp*1000:.0f} ms  ({t_cp/total*100:.1f}%)"
            f"  avg {t_cp/num_steps*1000:.1f} ms/step\n"
            f"  Decoder:   {t_decoder*1000:.0f} ms  ({t_decoder/total*100:.1f}%)"
            f"  avg {t_decoder/num_steps*1000:.1f} ms/step\n"
            f"  Sample:    {t_sample*1000:.0f} ms  ({t_sample/total*100:.1f}%)"
            f"  avg {t_sample/num_steps*1000:.1f} ms/step\n"
            f"  Embed:     {t_embed*1000:.0f} ms  ({t_embed/total*100:.1f}%)"
            f"  avg {t_embed/num_steps*1000:.1f} ms/step\n"
            f"  Total:     {total*1000:.0f} ms"
            f"  avg {total/num_steps*1000:.1f} ms/step",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    # Main generation loop
    # ------------------------------------------------------------------

    def generate(
        self,
        text: str,
        speaker_id: int = 3066,
        language_id: int = 2055,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        max_new_tokens: int = 8192,
    ) -> Generator[tuple[int, np.ndarray], None, None]:
        """Generate speech from text, yielding audio chunks as they become ready.

        Args:
            text:           Text to synthesize.
            speaker_id:     Codec speaker ID (default: serena = 3066).
            language_id:    Codec language ID (default: chinese = 2055).
            temperature:    Sampling temperature for the talker.
            top_k:          Top-k candidates for sampling.
            top_p:          Nucleus sampling threshold.
            max_new_tokens: Maximum AR generation steps.

        Yields:
            (chunk_index, wav_samples) where wav_samples is a float32 1-D array.
        """
        # ----------------------------------------------------------
        # 1. Prepare prefill embeddings
        # ----------------------------------------------------------
        prefill_embeds, tts_pad_embed = (
            self.text_cond.prepare_prefill(text, speaker_id, language_id)
        )

        # ----------------------------------------------------------
        # 2. Reset talker KV cache
        # ----------------------------------------------------------
        self.talker.reset()

        # ----------------------------------------------------------
        # 3. Prefill
        # ----------------------------------------------------------
        logits, hidden = self.talker.prefill(prefill_embeds)

        # ----------------------------------------------------------
        # 4. Initialize streaming state
        # ----------------------------------------------------------
        frames_buffer: list[np.ndarray] = []
        left_context = np.zeros((1, 16, 25), dtype=np.int64)
        chunk_idx = 0
        generated_tokens: list[int] = []

        # Async decoder state
        pending_decode: Future | None = None

        # Timing accumulators
        t_sample_total = 0.0
        t_cp_total = 0.0
        t_embed_total = 0.0
        t_talker_total = 0.0
        t_decoder_total = 0.0
        num_steps = 0

        # ----------------------------------------------------------
        # 5. Autoregressive loop
        # ----------------------------------------------------------
        for step in range(max_new_tokens):
            # (a) Sample token from last-position logits
            t0 = time.perf_counter()
            token = self._sample_token(
                logits[:, -1:, :], temperature, top_k, top_p, generated_tokens
            )
            t1 = time.perf_counter()

            # (b) Track generated tokens for repetition penalty
            generated_tokens.append(token)

            # (c) Check for end-of-speech
            if token == CODEC_EOS_ID:
                t_sample_total += t1 - t0
                break

            # (d) Extract talker hidden state at last position
            talker_hidden = hidden[:, -1:, :]  # [1, 1, 1024]

            # (e) Predict all 16 codec codes (layer-0 + 15 residual)
            codes = self.code_predictor.predict(
                talker_hidden[0, 0],
                token,
                self.codec_embed_table,
                temperature,
                top_k,
            )  # int64 [16]
            t2 = time.perf_counter()

            # (f) Buffer the frame
            frames_buffer.append(codes)

            # (g) Build next input embedding for the talker
            #     Sum of: talker embed(layer-0) + code predictor embeds(layers 1-15)
            layer0_embed = self.codec_embed_table[codes[0]]  # [1024]
            layer_embeds = []
            for i in range(15):
                layer_embeds.append(self.cp_embeds[f"emb_{i}"][codes[i + 1]])
            sum_embed = layer0_embed + sum(layer_embeds)  # [1024]

            # Non-streaming mode: all text was in prefill, use tts_pad during AR
            text_add = tts_pad_embed[0, 0]  # [1024]

            next_embed = (sum_embed + text_add).reshape(1, 1, 1024).astype(
                np.float32
            )
            t3 = time.perf_counter()

            # (h) Decode a chunk when enough frames have accumulated (async)
            if len(frames_buffer) >= DECODER_CHUNK_SIZE:
                # Wait for any previous async decode to complete and yield
                if pending_decode is not None:
                    prev_idx, prev_wav = pending_decode.result()
                    yield (prev_idx, prev_wav)

                # Prepare decoder input
                new_frames = np.stack(
                    frames_buffer[-DECODER_CHUNK_SIZE:]
                )  # [50, 16]
                combined = np.concatenate(
                    [left_context[0].T, new_frames], axis=0
                )  # [75, 16]
                decoder_input = combined.T[np.newaxis, :, :]  # [1, 16, 75]
                decoder_input_i64 = decoder_input.astype(np.int64)

                # Submit async decode
                current_chunk_idx = chunk_idx
                pending_decode = self._decoder_pool.submit(
                    self._async_decode, decoder_input_i64, current_chunk_idx
                )

                # Update state immediately (don't wait for decode)
                left_context = decoder_input[:, :, -DECODER_LEFT_CONTEXT:]
                frames_buffer = frames_buffer[DECODER_CHUNK_SIZE:]
                chunk_idx += 1
            t4 = time.perf_counter()

            # (i) Run next talker step
            logits, hidden = self.talker.generate_step(next_embed)
            t5 = time.perf_counter()

            # Accumulate timing
            t_sample_total += t1 - t0
            t_cp_total += t2 - t1
            t_embed_total += t3 - t2
            t_decoder_total += t4 - t3
            t_talker_total += t5 - t4
            num_steps += 1

        # ----------------------------------------------------------
        # 6. Yield pending async decode result
        # ----------------------------------------------------------
        if pending_decode is not None:
            prev_idx, prev_wav = pending_decode.result()
            yield (prev_idx, prev_wav)
            pending_decode = None

        # ----------------------------------------------------------
        # 7. Decode remaining frames after EOS (synchronous)
        # ----------------------------------------------------------
        if frames_buffer:
            num_remaining = len(frames_buffer)
            remaining = np.stack(frames_buffer)  # [R, 16]
            # Pad to DECODER_CHUNK_SIZE (50) frames with zeros
            padded = np.zeros(
                (DECODER_CHUNK_SIZE, 16), dtype=np.int64
            )
            padded[:num_remaining] = remaining
            combined = np.concatenate(
                [left_context[0].T, padded], axis=0
            )  # [75, 16]
            decoder_input = combined.T[np.newaxis, :, :]  # [1, 16, 75]
            wav = self.decoder.decode_chunk(
                decoder_input.astype(np.int64)
            )
            # Extract only valid new audio (skip left context, stop at end of
            # actual frames)
            start = DECODER_LEFT_CONTEXT * DECODER_DECODE_UPSAMPLE_RATE
            end = (
                (DECODER_LEFT_CONTEXT + num_remaining)
                * DECODER_DECODE_UPSAMPLE_RATE
            )
            wav_new = wav[start:end]
            yield (chunk_idx, wav_new)

        # ----------------------------------------------------------
        # 8. Print timing summary
        # ----------------------------------------------------------
        self._print_timing(
            t_sample_total, t_cp_total, t_embed_total,
            t_talker_total, t_decoder_total, num_steps,
        )
