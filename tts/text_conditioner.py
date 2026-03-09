"""CPU-side text tokenization and embedding construction (numpy only).

Loads pre-exported numpy arrays (text embedding table, text projection MLP,
and talker codec embedding table) to build the prefill embedding sequence
required by the talker LLM in non-streaming mode.

Non-streaming mode includes ALL text tokens in the prefill so the KV cache
sees the full text context, matching the original PyTorch model's default
``non_streaming_mode=True`` behaviour.

No PyTorch dependency at inference time -- all math is done in numpy.
"""

from __future__ import annotations

import numpy as np
from transformers import AutoTokenizer

from .config import (
    CODEC_BOS_ID,
    CODEC_PAD_ID,
    CODEC_THINK_BOS_ID,
    CODEC_THINK_EOS_ID,
    CODEC_THINK_ID,
    TTS_BOS_TOKEN_ID,
    TTS_EOS_TOKEN_ID,
    TTS_PAD_TOKEN_ID,
)


class TextConditioner:
    """CPU-side text tokenization and embedding construction for Qwen3-TTS.

    Builds the dual-stream (text + codec) prefill embeddings that the talker
    LLM expects, using only numpy for all linear algebra.
    """

    def __init__(
        self,
        hf_model_dir: str,
        text_embedding_npy: str,
        text_projection_npz: str,
        talker_embed_tokens_npy: str,
    ) -> None:
        """Load tokenizer and pre-exported weight arrays.

        Args:
            hf_model_dir:           Path to the HF model directory (contains tokenizer).
            text_embedding_npy:     Path to text_embedding.npy file.
            text_projection_npz:    Path to text_projection.npz file.
            talker_embed_tokens_npy: Path to talker_embed_tokens.npy file.
        """
        # HuggingFace tokenizer (used for text -> token IDs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_model_dir, trust_remote_code=True
        )

        # Text embedding table  [151936, 2048]
        self.text_embedding = np.load(text_embedding_npy)

        # Text projection MLP weights (2-layer: 2048 -> 2048 -> 1024)
        proj = np.load(text_projection_npz)
        self.fc1_weight = proj["linear_fc1.weight"]  # [2048, 2048]
        self.fc1_bias = proj["linear_fc1.bias"]       # [2048]
        self.fc2_weight = proj["linear_fc2.weight"]  # [1024, 2048]
        self.fc2_bias = proj["linear_fc2.bias"]       # [1024]

        # Talker codec embedding table  [3072, 1024]
        self.talker_embed_tokens = np.load(talker_embed_tokens_npy)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _text_proj(self, x: np.ndarray) -> np.ndarray:
        """Apply the text projection MLP: SiLU(x W1^T + b1) W2^T + b2.

        The original model uses SiLU (Swish) activation, not GELU.

        Args:
            x: Input array of shape [..., 2048].

        Returns:
            Projected array of shape [..., 1024].
        """
        h = x @ self.fc1_weight.T + self.fc1_bias
        # SiLU (Swish): h * sigmoid(h)
        h = h / (1.0 + np.exp(-h))
        return h @ self.fc2_weight.T + self.fc2_bias

    def _text_embed(self, token_ids: list[int]) -> np.ndarray:
        """Look up text embedding table and project to talker hidden dim.

        Args:
            token_ids: List of text token IDs (from the 151936-vocab tokenizer).

        Returns:
            Array of shape [N, 1024].
        """
        raw = self.text_embedding[token_ids]  # [N, 2048]
        return self._text_proj(raw)           # [N, 1024]

    def _codec_embed(self, token_ids: list[int]) -> np.ndarray:
        """Look up talker codec embedding table.

        Args:
            token_ids: List of codec token IDs (from the 3072-vocab codec).

        Returns:
            Array of shape [N, 1024].
        """
        return self.talker_embed_tokens[token_ids]  # [N, 1024]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare_prefill(
        self,
        text: str,
        speaker_id: int = 3066,
        language_id: int = 2055,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build the prefill embedding sequence for non-streaming TTS.

        Constructs the dual-stream (text + codec) embeddings that the talker
        LLM consumes as ``inputs_embeds`` during the prefill phase.

        Non-streaming mode: ALL text tokens are included in the prefill so the
        KV cache sees the full text context.  During AR generation only
        ``tts_pad_embed`` is added (no trailing text injection).

        The variable-length prefill layout (non-streaming mode, N = text tokens):

            Pos 0-2:     text_embed(im_start, assistant, newline)   -- role prefix
            Pos 3:       tts_pad  + codec_embed(THINK)
            Pos 4:       tts_pad  + codec_embed(THINK_BOS)
            Pos 5:       tts_pad  + codec_embed(language_id)
            Pos 6:       tts_pad  + codec_embed(THINK_EOS)
            Pos 7:       tts_pad  + codec_embed(speaker_id)
            Pos 8:       tts_bos  + codec_embed(CODEC_PAD)
            Pos 9..9+N-1:  text_embed(text[i]) + codec_embed(CODEC_PAD)
            Pos 9+N:     tts_eos_embed + codec_embed(CODEC_PAD)
            Pos 9+N+1:   tts_pad  + codec_embed(CODEC_BOS)

        Args:
            text:        The text to synthesize.
            speaker_id:  Codec speaker ID (default: serena = 3066).
            language_id: Codec language ID (default: chinese = 2055).

        Returns:
            Tuple of:
                prefill_embeds  [1, 9+N+2, 1024] -- prefill input for talker
                                                     (N = number of text tokens).
                tts_pad_embed   [1, 1, 1024]      -- pad embed for AR decoding.
        """
        # ----------------------------------------------------------
        # 1. Format text into the ChatML assistant template
        # ----------------------------------------------------------
        formatted = (
            f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        )

        # ----------------------------------------------------------
        # 2. Tokenize (no special tokens -- template already has them)
        # ----------------------------------------------------------
        input_ids = self.tokenizer.encode(formatted, add_special_tokens=False)

        # ----------------------------------------------------------
        # 3. Split token IDs into segments
        # ----------------------------------------------------------
        role_prefix = input_ids[:3]        # [im_start, assistant, newline]
        all_text_tokens = input_ids[3:-5]  # Full text tokens
        # input_ids[-5:] discarded: [im_end, \n, im_start, assistant, \n]

        # ----------------------------------------------------------
        # 4. Pre-compute special TTS text embeddings  (each [1, 1024])
        # ----------------------------------------------------------
        tts_pad_vec = self._text_embed([TTS_PAD_TOKEN_ID])  # [1, 1024]
        tts_bos_vec = self._text_embed([TTS_BOS_TOKEN_ID])  # [1, 1024]

        # ----------------------------------------------------------
        # 5. Build the variable-length prefill
        # ----------------------------------------------------------
        # Pos 0-2: role prefix (text embedding only, no codec side)
        role_embed = self._text_embed(role_prefix)  # [3, 1024]

        # Pos 3-7: tts_pad + codec control tokens
        text_side_3_7 = np.tile(tts_pad_vec, (5, 1))  # [5, 1024]
        codec_side_3_7 = self._codec_embed([
            CODEC_THINK_ID,      # pos 3
            CODEC_THINK_BOS_ID,  # pos 4
            language_id,         # pos 5
            CODEC_THINK_EOS_ID,  # pos 6
            speaker_id,          # pos 7
        ])  # [5, 1024]
        pos_3_7 = text_side_3_7 + codec_side_3_7  # [5, 1024]

        # Pos 8: tts_bos + codec_embed(CODEC_PAD)
        pos_8 = tts_bos_vec + self._codec_embed([CODEC_PAD_ID])  # [1, 1024]

        # Pos 9..9+N-1: all text tokens + codec_embed(CODEC_PAD)
        # Pos 9+N:      tts_eos_embed  + codec_embed(CODEC_PAD)
        all_text_embed = self._text_embed(all_text_tokens)      # [N, 1024]
        tts_eos_vec = self._text_embed([TTS_EOS_TOKEN_ID])      # [1, 1024]
        text_and_eos = np.concatenate(
            [all_text_embed, tts_eos_vec], axis=0
        )  # [N+1, 1024]

        codec_pad = self._codec_embed([CODEC_PAD_ID])           # [1, 1024]
        text_and_codec = text_and_eos + codec_pad  # broadcast: [N+1, 1024]

        # Pos 9+N+1: tts_pad + codec_embed(CODEC_BOS)
        codec_bos = self._codec_embed([CODEC_BOS_ID])           # [1, 1024]
        final_bos = tts_pad_vec + codec_bos                     # [1, 1024]

        # Concatenate all positions: role(3) + control(5) + bos(1)
        #   + text+eos(N+1) + final_bos(1)
        prefill = np.concatenate(
            [role_embed, pos_3_7, pos_8, text_and_codec, final_bos], axis=0
        )  # [9+N+2, 1024]
        prefill_embeds = prefill[np.newaxis, :, :]  # [1, 9+N+2, 1024]

        # ----------------------------------------------------------
        # 6. Return tts_pad_embed for use during AR generation
        # ----------------------------------------------------------
        tts_pad_embed = tts_pad_vec[np.newaxis, :, :]  # [1, 1, 1024]

        return prefill_embeds, tts_pad_embed
