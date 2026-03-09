"""OpenVINO Stateful Talker LLM supporting NPU (single-output) and CPU (dual-output).

28-layer Qwen3 with internal KV cache management.  Two operational modes:

NPU mode (NPUW_LLM):
    Uses a single-output model (talker_npu/) that emits only logits.
    Hidden states are recovered via pseudo-inverse: hidden = logits @ lm_head_pinv.

CPU mode:
    Uses the dual-output model (talker_stateful_embeds/) that emits both logits
    and hidden_states directly.

Inputs (after IR surgery):
    inputs_embeds  [1, S, 1024]   float32  -- pre-computed embeddings
    attention_mask [1, S]          int64    -- 1 for valid positions
    position_ids   [1, S]          int64    -- absolute positions
    beam_idx       [1]             int32    -- always 0 (single beam)

Outputs:
    NPU: [0] logits [1, S, 3072] float32
    CPU: [0] logits [1, S, 3072] float32  +  [1] hidden_states [1, S, 1024] float32
"""

from __future__ import annotations

import numpy as np
import openvino as ov


class OVTalker:
    """Stateful talker LLM using NPUW_LLM on NPU.

    The NPUW_LLM plugin manages KV cache internally.  The caller is
    responsible for calling ``reset()`` before each new utterance and
    then using ``prefill`` followed by repeated ``generate_step`` calls.
    """

    def __init__(
        self,
        talker_xml: str,
        lm_head_pinv_npy: str,
        npu_config: dict,
        device: str = "NPU",
    ) -> None:
        """Load and compile the stateful talker IR model.

        Args:
            talker_xml:      Path to the OpenVINO IR XML file for the talker.
            lm_head_pinv_npy: Path to the ``.npy`` file containing the
                              pseudo-inverse of the lm_head weight matrix
                              (used in NPU mode to recover hidden states).
            npu_config:      NPU compilation config dict (e.g. NPUW_LLM settings).
                             Ignored when *device* is not ``"NPU"``.
            device:          OpenVINO device string (``"NPU"`` or ``"CPU"``).
        """
        core = ov.Core()
        self._npu_mode = device == "NPU"
        model = core.read_model(talker_xml)

        if self._npu_mode:
            # Single-output model for NPUW_LLM compatibility
            self._compiled = core.compile_model(model, "NPU", npu_config)
            self._lm_head_pinv = np.load(lm_head_pinv_npy).astype(np.float32)
        else:
            # Two-output model (logits + hidden_states)
            self._compiled = core.compile_model(
                model, device, {"PERFORMANCE_HINT": "LATENCY"}
            )
            self._lm_head_pinv = None

        self._request = self._compiled.create_infer_request()
        self._pos: int = 0

        # Pre-allocated arrays for generate_step (reused across calls)
        self._step_mask_buf = np.ones((1, 9000), dtype=np.int64)  # large enough for max generation
        self._step_pos = np.zeros((1, 1), dtype=np.int64)
        self._beam_idx = np.array([0], dtype=np.int32)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset KV cache state and position counter for a new utterance."""
        self._request.reset_state()
        self._pos = 0

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def prefill(
        self, inputs_embeds: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run prefill with the full prompt embeddings.

        Returns:
            (logits, hidden_states) -- on NPU, hidden_states is recovered
            via pseudo-inverse; on CPU, read directly from output tensor 1.
        """
        seq_len = inputs_embeds.shape[1]

        self._request.infer({
            "inputs_embeds": inputs_embeds.astype(np.float32),
            "attention_mask": np.ones((1, seq_len), dtype=np.int64),
            "position_ids": np.arange(seq_len, dtype=np.int64).reshape(1, -1),
            "beam_idx": np.array([0], dtype=np.int32),
        })

        self._pos = seq_len

        logits = self._request.get_output_tensor(0).data.copy()
        if self._npu_mode:
            # NPUW_LLM returns [1, 1, 3072] (only last token) during prefill
            hidden_states = logits @ self._lm_head_pinv  # [1, 1, 1024]
        else:
            hidden_states = self._request.get_output_tensor(1).data.copy()
        return logits, hidden_states

    def generate_step(
        self, inputs_embeds: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run a single autoregressive generation step.

        Returns:
            (logits, hidden_states) -- same recovery logic as prefill.
        """
        self._step_pos[0, 0] = self._pos
        mask_len = self._pos + 1
        self._request.infer({
            "inputs_embeds": inputs_embeds.astype(np.float32),
            "attention_mask": self._step_mask_buf[:, :mask_len],
            "position_ids": self._step_pos,
            "beam_idx": self._beam_idx,
        })

        self._pos += 1

        logits = self._request.get_output_tensor(0).data.copy()
        if self._npu_mode:
            hidden_states = logits @ self._lm_head_pinv  # [1, 1, 1024]
        else:
            hidden_states = self._request.get_output_tensor(1).data.copy()
        return logits, hidden_states
