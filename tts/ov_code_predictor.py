"""OpenVINO Code Predictor for residual codebooks 1-15.

Stateful 5-layer Qwen3 transformer with KV cache.  Uses a prefill + decode
approach: the initial 2 tokens [projected_hidden_state, projected_layer0_embed]
are processed in a single prefill step, then 14 decode steps each process 1
token [projected_group_i_embed] autoregressively.

The KV cache is managed internally by the OpenVINO stateful model.  An all-ones
attention mask is used throughout (no partial masking), which avoids the FP16 NaN
issue that occurred with the old re-prefill partial masks on GPU/NPU.

Prefill (2 tokens):
    [0]  talker_hidden   -- projected last hidden state from the talker
    [1]  layer0_embed    -- projected layer-0 codec embedding

Decode steps (1 token each):
    [2]  group1_embed    -- projected group-1 codec embedding
    ...
    [16] group15_embed   -- projected group-15 codec embedding

IR model (stateful with KV cache):
    Input:  inputs_embeds  [1, seq_len, 1024]  float32
    Input:  attention_mask [1, past_len + seq_len]  int64
    Input:  position_ids   [1, seq_len]  int64
    Input:  beam_idx       [1]  int32
    Output: hidden_states  [1, seq_len, 1024]  float32

Auxiliary numpy files:
    code_predictor_embeds.npz   -- 15 embedding tables (emb_0..emb_14)
    code_predictor_lm_heads.npz -- 15 lm_head matrices (head_0..head_14)
    code_predictor_proj_in.npz  -- projection info (is_identity + optional weight/bias)
"""

from __future__ import annotations

import numpy as np
import openvino as ov

# Code predictor hidden dimension (matches the talker hidden size).
_CP_HIDDEN_SIZE = 1024


class OVCodePredictor:
    """Stateful code predictor that predicts residual codebooks 1-15.

    For each generation step the talker produces a layer-0 codec token.
    This predictor takes the talker's last hidden state and the layer-0
    token, then autoregressively predicts the remaining 15 codebook
    entries needed by the speech decoder.

    Uses a stateful OpenVINO model with internal KV cache, following the
    same prefill + decode pattern as the ASR decoder.
    """

    def __init__(
        self,
        cp_xml: str,
        embeds_npz: str,
        lm_heads_npz: str,
        proj_in_npz: str,
        device: str = "CPU",
        n_groups: int = 15,
    ) -> None:
        """Load and compile the stateful code predictor IR and auxiliary weights.

        Args:
            cp_xml:       Path to the OpenVINO IR XML file for the code predictor.
            embeds_npz:   Path to ``code_predictor_embeds.npz`` containing
                          15 embedding tables (``emb_0`` .. ``emb_14``).
            lm_heads_npz: Path to ``code_predictor_lm_heads.npz`` containing
                          15 lm_head weight matrices (``head_0`` .. ``head_14``).
            proj_in_npz:  Path to ``code_predictor_proj_in.npz`` containing
                          projection info (``is_identity`` + optional ``weight``/``bias``).
            device:       OpenVINO device string (e.g. ``"CPU"``, ``"GPU"``).
            n_groups:     Number of residual groups to predict (1-15). Groups beyond
                          this limit are padded with 0. Fewer groups = faster but
                          lower audio quality.
        """
        self._n_groups = min(max(n_groups, 1), 15)
        core = ov.Core()
        model = core.read_model(cp_xml)
        if device == "GPU":
            # Talker hidden states can have large values (up to ~90) that overflow
            # in FP16 attention computation. Force FP32 to avoid NaN.
            config = {"INFERENCE_PRECISION_HINT": "f32"}
        else:
            config = {"PERFORMANCE_HINT": "LATENCY"}
        self._compiled = core.compile_model(model, device, config)
        self._request = self._compiled.create_infer_request()

        # 15 codec embedding tables, each [2048, 1024]
        embeds_data = np.load(embeds_npz)
        self._embeds = [
            embeds_data[f"emb_{i}"].astype(np.float32) for i in range(15)
        ]

        # 15 lm_head weight matrices, each [2048, 1024]
        heads_data = np.load(lm_heads_npz)
        self._lm_heads = [
            heads_data[f"head_{i}"].astype(np.float32) for i in range(15)
        ]

        # Projection from talker hidden dim to code predictor hidden dim
        proj_data = np.load(proj_in_npz)
        self._is_identity: bool = bool(proj_data["is_identity"])
        if not self._is_identity:
            self._proj_weight = proj_data["weight"].astype(np.float32)
            self._proj_bias = (
                proj_data["bias"].astype(np.float32)
                if "bias" in proj_data
                else None
            )
        else:
            self._proj_weight = None
            self._proj_bias = None

        # Pre-allocated inference arrays (reused across predict() calls)
        self._prefill_embeds = np.zeros((1, 2, _CP_HIDDEN_SIZE), dtype=np.float32)
        self._prefill_mask = np.ones((1, 2), dtype=np.int64)
        self._prefill_pos = np.arange(2, dtype=np.int64).reshape(1, -1)
        self._beam_idx = np.array([0], dtype=np.int32)
        self._decode_embeds = np.zeros((1, 1, _CP_HIDDEN_SIZE), dtype=np.float32)
        # Pre-allocate decode masks and positions for each possible step
        # Decode step 0 has past_len=2, step 1 has past_len=3, ..., step 13 has past_len=15
        self._decode_masks = [np.ones((1, past_len + 1), dtype=np.int64) for past_len in range(2, 17)]
        self._decode_positions = [np.array([[pos]], dtype=np.int64) for pos in range(2, 17)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _proj_in(self, x: np.ndarray) -> np.ndarray:
        """Project a vector through small_to_mtp_projection.

        Args:
            x: float32 array of shape [hidden_size].

        Returns:
            Projected vector of shape [_CP_HIDDEN_SIZE].
        """
        if self._is_identity:
            return x
        out = x @ self._proj_weight.T
        if self._proj_bias is not None:
            out = out + self._proj_bias
        return out

    @staticmethod
    def _sample_top_k(
        logits: np.ndarray,
        top_k: int = 50,
        temperature: float = 0.9,
    ) -> int:
        """Sample a token from logits using top-k with temperature.

        Args:
            logits:      1-D float array of shape [vocab_size].
            top_k:       Number of top candidates to consider.
            temperature: Sampling temperature (higher = more random).

        Returns:
            Sampled token index.
        """
        logits = logits.astype(np.float64) / temperature

        # Select top-k indices
        top_k_indices = np.argpartition(logits, -top_k)[-top_k:]
        top_k_logits = logits[top_k_indices]

        # Numerically stable softmax
        top_k_logits -= np.max(top_k_logits)
        probs = np.exp(top_k_logits)
        probs /= probs.sum()

        idx = np.random.choice(len(top_k_indices), p=probs)
        return int(top_k_indices[idx])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        talker_hidden: np.ndarray,
        layer0_token: int,
        codec_embed_table: np.ndarray,
        temperature: float = 0.9,
        top_k: int = 50,
    ) -> np.ndarray:
        """Predict all 16 codec codes (layer-0 + 15 residual groups).

        Args:
            talker_hidden:    float32 array [1024] -- talker's last hidden state.
            layer0_token:     int -- layer-0 codec token predicted by the talker.
            codec_embed_table: float32 array [vocab, 1024] -- layer-0 codec
                               embedding table (shared with the talker).
            temperature:      Sampling temperature for residual codes.
            top_k:            Top-k candidates for residual code sampling.

        Returns:
            int64 array of shape [16] containing layer-0 token followed by
            15 predicted residual codes.
        """
        # 1. Reset KV cache for a fresh sequence
        self._request.reset_state()

        # 2. Build prefill embeddings in-place: 2 tokens
        #    [0] = projected talker hidden state
        #    [1] = projected layer-0 codec embedding
        self._prefill_embeds[0, 0] = self._proj_in(talker_hidden)
        self._prefill_embeds[0, 1] = self._proj_in(codec_embed_table[layer0_token])

        # 3. Prefill: process both tokens at once (using pre-allocated arrays)
        self._request.infer({
            "inputs_embeds": self._prefill_embeds,
            "attention_mask": self._prefill_mask,
            "position_ids": self._prefill_pos,
            "beam_idx": self._beam_idx,
        })

        # 4. Extract hidden at last position, apply lm_head[0], sample
        hidden = self._request.get_output_tensor(0).data[0, -1].copy()
        logits = hidden @ self._lm_heads[0].T
        code = self._sample_top_k(logits, top_k=top_k, temperature=temperature)
        codes = [layer0_token, code]

        # 5. Decode steps for groups 1 to n_groups-1 (using pre-allocated arrays)
        for group_idx in range(1, self._n_groups):
            embed = self._proj_in(self._embeds[group_idx - 1][codes[-1]])
            self._decode_embeds[0, 0] = embed

            self._request.infer({
                "inputs_embeds": self._decode_embeds,
                "attention_mask": self._decode_masks[group_idx - 1],
                "position_ids": self._decode_positions[group_idx - 1],
                "beam_idx": self._beam_idx,
            })

            hidden = self._request.get_output_tensor(0).data[0, -1].copy()
            logits = hidden @ self._lm_heads[group_idx].T
            code = self._sample_top_k(logits, top_k=top_k, temperature=temperature)
            codes.append(code)

        # Pad remaining groups with 0 if n_groups < 15
        while len(codes) < 16:
            codes.append(0)

        return np.array(codes, dtype=np.int64)
