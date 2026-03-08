"""OpenVINO Text Decoder wrapper (NPU or CPU, IR-surgery stateful model).

Uses the IR-surgery model (decoder_stateful_embeds) that accepts inputs_embeds
with KV-cache stateful behavior for O(n) autoregressive decoding.

Supports two devices:
- NPU: Uses NPUW_LLM plugin for ~21ms/token. Returns only last-position logits.
- CPU: Standard stateful inference for ~28ms/token. Returns full sequence logits.
"""

import numpy as np
import openvino as ov

from .config import DECODER_XML, EMBED_TABLE_NPY, IM_END, NPU_DECODER_CONFIG


class OVDecoder:
    """Stateful text decoder with inputs_embeds support (NPU or CPU)."""

    def __init__(self, device: str = "NPU"):
        core = ov.Core()
        if device == "NPU":
            self._compiled = core.compile_model(DECODER_XML, "NPU", NPU_DECODER_CONFIG)
        else:
            self._compiled = core.compile_model(
                DECODER_XML, "CPU", {"PERFORMANCE_HINT": "LATENCY"}
            )
        self._request = self._compiled.create_infer_request()
        self._embed_table = np.load(EMBED_TABLE_NPY)
        self._past_len = 0

    @property
    def embed_table(self) -> np.ndarray:
        """Embedding table [vocab_size, 1024]."""
        return self._embed_table

    def reset(self):
        """Reset KV-cache state for a new utterance."""
        self._request.reset_state()
        self._past_len = 0

    def prefill(self, inputs_embeds: np.ndarray) -> int:
        """Run prefill with full inputs_embeds sequence.

        Args:
            inputs_embeds: Shape [seq_len, 1024] (no batch dim).

        Returns:
            First predicted token ID.
        """
        seq_len = inputs_embeds.shape[0]
        self._request.infer({
            "inputs_embeds": inputs_embeds[np.newaxis, :, :].astype(np.float32),
            "attention_mask": np.ones((1, seq_len), dtype=np.int64),
            "position_ids": np.arange(seq_len, dtype=np.int64).reshape(1, -1),
            "beam_idx": np.array([0], dtype=np.int32),
        })
        logits = self._request.get_output_tensor(0).data
        self._past_len = seq_len
        # NPUW_LLM returns [1,1,vocab], CPU returns [1,seq_len,vocab]
        # logits[0, -1, :] works for both
        return int(np.argmax(logits[0, -1, :]))

    def decode_step(self, token_id: int) -> int:
        """Run one decode step with a single token.

        Args:
            token_id: Previous token ID to continue from.

        Returns:
            Next predicted token ID.
        """
        token_embed = self._embed_table[token_id][np.newaxis, np.newaxis, :]  # [1,1,1024]
        self._request.infer({
            "inputs_embeds": token_embed.astype(np.float32),
            "attention_mask": np.ones((1, self._past_len + 1), dtype=np.int64),
            "position_ids": np.array([[self._past_len]], dtype=np.int64),
            "beam_idx": np.array([0], dtype=np.int32),
        })
        self._past_len += 1
        logits = self._request.get_output_tensor(0).data
        return int(np.argmax(logits[0, -1, :]))

    def generate(self, inputs_embeds: np.ndarray, max_new_tokens: int = 32) -> list[int]:
        """Run full generation: prefill + greedy decode loop.

        Args:
            inputs_embeds: Shape [seq_len, 1024].
            max_new_tokens: Max tokens to generate.

        Returns:
            List of generated token IDs (excluding IM_END).
        """
        self.reset()
        token_id = self.prefill(inputs_embeds)
        generated = []

        for _ in range(max_new_tokens):
            if token_id == IM_END:
                break
            generated.append(token_id)
            token_id = self.decode_step(token_id)

        return generated
