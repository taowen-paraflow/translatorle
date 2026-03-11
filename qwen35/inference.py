"""
Standalone OpenVINO inference runtime for Qwen3.5.

Dependencies: torch, transformers, openvino, numpy.
No dependency on optimum or optimum-intel.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import openvino as ov
import torch
from transformers import AutoConfig, AutoTokenizer, GenerationConfig
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import ModelOutput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache state
# ---------------------------------------------------------------------------


class Qwen35CacheState:
    """Holds the KV / conv / recurrent state read back from an OpenVINO
    stateful inference request.

    Each list is indexed by the *layer-local* ordinal that appears in the
    state variable name (e.g. ``present.conv.5`` -> index 5 in
    ``conv_states``).
    """

    def __init__(
        self,
        conv_states: Optional[List[Optional[np.ndarray]]] = None,
        recurrent_states: Optional[List[Optional[np.ndarray]]] = None,
        key_cache: Optional[List[Optional[np.ndarray]]] = None,
        value_cache: Optional[List[Optional[np.ndarray]]] = None,
    ):
        self.conv_states: List[Optional[np.ndarray]] = conv_states or []
        self.recurrent_states: List[Optional[np.ndarray]] = recurrent_states or []
        self.key_cache: List[Optional[np.ndarray]] = key_cache or []
        self.value_cache: List[Optional[np.ndarray]] = value_cache or []

    # -- factory ----------------------------------------------------------

    @classmethod
    def from_query_state(cls, states) -> "Qwen35CacheState":
        """Build a ``Qwen35CacheState`` from the list returned by
        ``request.query_state()``.

        State variable names follow the pattern produced by the
        optimum-intel stateful conversion:

            ``present.conv.0``, ``present.recurrent.5``,
            ``present.key.0``, ``present.value.3``

        The ``cache_params.past.*`` / ``cache_params.present.*`` prefixes
        used by the optimum-intel non-stateful path are also accepted.
        """
        conv: Dict[int, np.ndarray] = {}
        recurrent: Dict[int, np.ndarray] = {}
        key: Dict[int, np.ndarray] = {}
        value: Dict[int, np.ndarray] = {}

        for state in states:
            name: str = state.name
            data: np.ndarray = state.state.data

            idx = int(name.rsplit(".", 1)[-1])

            if ".conv" in name:
                conv[idx] = data
            elif ".recurrent" in name:
                recurrent[idx] = data
            elif ".key" in name:
                key[idx] = data
            elif ".value" in name:
                value[idx] = data

        def _to_list(d: Dict[int, np.ndarray]) -> List[Optional[np.ndarray]]:
            if not d:
                return []
            size = max(d.keys()) + 1
            return [d.get(i) for i in range(size)]

        return cls(
            conv_states=_to_list(conv),
            recurrent_states=_to_list(recurrent),
            key_cache=_to_list(key),
            value_cache=_to_list(value),
        )


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class Qwen35Output(ModelOutput):
    """Model output carrying logits and the hybrid cache state."""

    logits: Optional[torch.FloatTensor] = None
    cache_params: Optional[Qwen35CacheState] = None


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------


class Qwen35OVModel(GenerationMixin):
    """OpenVINO inference wrapper for Qwen3.5 (Gated Delta Networks hybrid
    architecture).

    Designed to plug into ``transformers.GenerationMixin.generate()`` with
    no dependency on optimum / optimum-intel.
    """

    main_input_name = "input_ids"
    _supports_cache_class = False
    _is_stateful = True

    # -----------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------

    def __init__(
        self,
        model: ov.Model,
        compiled_model: ov.CompiledModel,
        request: ov.InferRequest,
        config,
        tokenizer,
        embed_table: np.ndarray,
        device_name: str = "CPU",
        ov_config: Optional[Dict[str, str]] = None,
    ):
        # GenerationMixin does not define __init__, but some downstream
        # helpers check for ``self.config`` / ``self.generation_config``.
        super().__init__()

        self._ov_model = model
        self._compiled_model = compiled_model
        self._request = request
        self.config = config
        self.tokenizer = tokenizer
        self._embed_table = embed_table
        self._device_name = device_name

        # NPUW_LLM mode: KV cache is managed on-device by the NPUW_LLM engine.
        # Attention mask must grow with each decode step (full context length).
        self._is_npuw_llm = bool(
            ov_config and ov_config.get("NPUW_LLM", "").upper() == "YES"
        )
        if self._is_npuw_llm:
            logger.info("NPUW_LLM mode: attention_mask grows with context")

        # Position tracking for token-by-token generation.
        self._past_length: int = 0

        # Discover the input names accepted by the compiled model so we
        # know which optional tensors to provide.
        self._input_names: set = {
            inp.get_any_name() for inp in model.inputs
        }

        # Detect whether the IR has static seq_len=1 (NPU Loop-free IR).
        # If so, force token-by-token prefill regardless of device.
        self._token_by_token = False
        for inp in model.inputs:
            if "inputs_embeds" in inp.get_any_name():
                seq_dim = inp.get_partial_shape()[1]
                if seq_dim.is_static and seq_dim.get_length() == 1:
                    self._token_by_token = True
                    logger.info("Detected static seq_len=1 IR — using token-by-token prefill")
                break

        # Discover state variable names and classify them.
        self._conv_state_names: List[str] = []
        self._recurrent_state_names: List[str] = []
        self._key_state_names: List[str] = []
        self._value_state_names: List[str] = []
        for state in request.query_state():
            name = state.name
            if ".conv" in name:
                self._conv_state_names.append(name)
            elif ".recurrent" in name:
                self._recurrent_state_names.append(name)
            elif ".key" in name:
                self._key_state_names.append(name)
            elif ".value" in name:
                self._value_state_names.append(name)
        self._conv_state_names.sort()
        self._recurrent_state_names.sort()
        self._key_state_names.sort()
        self._value_state_names.sort()

        num_state_vars = (
            len(self._conv_state_names)
            + len(self._recurrent_state_names)
            + len(self._key_state_names)
            + len(self._value_state_names)
        )
        logger.info(
            "Qwen35OVModel: %d stateful variables "
            "(conv=%d, recurrent=%d, key=%d, value=%d)",
            num_state_vars,
            len(self._conv_state_names),
            len(self._recurrent_state_names),
            len(self._key_state_names),
            len(self._value_state_names),
        )

        # Detect explicit (non-stateful) conv/recurrent Parameters.
        # These exist when KV-only stateful is used (NPU mode): conv/recurrent
        # are explicit I/O, not managed by NPUW_LLM.
        # Sort by numeric suffix (not lexicographic) so index i matches layer i.
        def _sort_by_suffix(names):
            return sorted(names, key=lambda n: int(n.rsplit(".", 1)[1]))

        self._explicit_conv_inputs: List[str] = _sort_by_suffix(
            n for n in self._input_names if "cache_params.past.conv" in n
        )
        self._explicit_recurrent_inputs: List[str] = _sort_by_suffix(
            n for n in self._input_names if "cache_params.past.recurrent" in n
        )
        # In NPUW_LLM mode, GDN states are managed on-device via passthrough
        # copy (present→past).  Python must NOT feed/read them or it will
        # overwrite the on-device state with stale data.
        self._has_explicit_gdn_states = bool(
            (self._explicit_conv_inputs or self._explicit_recurrent_inputs)
            and not self._is_npuw_llm
        )

        # Matching output names for reading back updated states
        self._output_names: set = set()
        for out in model.outputs:
            self._output_names.update(out.get_names())
        self._explicit_conv_outputs: List[str] = _sort_by_suffix(
            n for n in self._output_names if "cache_params.present.conv" in n
        )
        self._explicit_recurrent_outputs: List[str] = _sort_by_suffix(
            n for n in self._output_names if "cache_params.present.recurrent" in n
        )

        # Detect hybrid mode: IR has gdn_intermediate outputs for CPU FP32 state update
        self._gdn_intermediate_outputs: List[str] = sorted(
            n for n in self._output_names if "gdn_intermediate" in n
        )
        self._is_hybrid = bool(self._gdn_intermediate_outputs) and self._has_explicit_gdn_states
        if self._is_hybrid:
            logger.info(
                "Hybrid NPU+CPU mode: %d GDN intermediate outputs, "
                "CPU FP32 state update enabled",
                len(self._gdn_intermediate_outputs),
            )

        # Initialize GDN state tensors (zero-filled)
        self._gdn_states: Dict[str, np.ndarray] = {}
        if self._has_explicit_gdn_states:
            self._init_gdn_states()
            logger.info(
                "Explicit GDN states: %d conv + %d recurrent Parameters",
                len(self._explicit_conv_inputs),
                len(self._explicit_recurrent_inputs),
            )

        # --- Static-cache mode detection (NPU all-explicit model) ---
        # If there are no stateful variables AND we have explicit KV inputs,
        # the model uses pre-allocated static KV cache buffers.
        self._explicit_key_inputs: List[str] = _sort_by_suffix(
            n for n in self._input_names if "cache_params.past.key" in n
        )
        self._explicit_value_inputs: List[str] = _sort_by_suffix(
            n for n in self._input_names if "cache_params.past.value" in n
        )
        self._explicit_key_outputs: List[str] = _sort_by_suffix(
            n for n in self._output_names if "cache_params.present.key" in n
        )
        self._explicit_value_outputs: List[str] = _sort_by_suffix(
            n for n in self._output_names if "cache_params.present.value" in n
        )
        self._is_static_cache = bool(
            num_state_vars == 0
            and self._explicit_key_inputs
            and self._explicit_value_inputs
        )

        if self._is_static_cache:
            # Infer max_cache_len from KV input shape: [1, H, MAX_LEN, D]
            kv_shape = model.input(self._explicit_key_inputs[0]).get_partial_shape()
            self._max_cache_len = kv_shape[2].get_length()
            self._kv_buffers: Dict[str, np.ndarray] = {}
            self._init_kv_buffers()
            logger.info(
                "Static-cache mode: %d KV layers, max_cache_len=%d",
                len(self._explicit_key_inputs),
                self._max_cache_len,
            )

        # generation_config -- try to load from model config, fall back to
        # a sensible default.
        try:
            self.generation_config = GenerationConfig.from_model_config(config)
        except Exception:
            self.generation_config = GenerationConfig()

    # -----------------------------------------------------------------
    # Explicit GDN state management (KV-only stateful mode)
    # -----------------------------------------------------------------

    def _init_gdn_states(self) -> None:
        """Zero-initialize explicit GDN (conv/recurrent) state tensors."""
        for name in self._explicit_conv_inputs + self._explicit_recurrent_inputs:
            pshape = self._ov_model.input(name).get_partial_shape()
            shape = [
                dim.get_length() if dim.is_static else 1
                for dim in pshape
            ]
            self._gdn_states[name] = np.zeros(shape, dtype=np.float32)

        # FP32 shadow copies for hybrid mode
        if self._is_hybrid:
            self._fp32_recurrent_states: Dict[str, np.ndarray] = {}
            for name in self._explicit_recurrent_inputs:
                pshape = self._ov_model.input(name).get_partial_shape()
                shape = [dim.get_length() if dim.is_static else 1 for dim in pshape]
                self._fp32_recurrent_states[name] = np.zeros(shape, dtype=np.float32)

    def _feed_gdn_states(self, inp: Dict[str, Any]) -> None:
        """Add explicit GDN states to the input dict."""
        for name in self._explicit_conv_inputs:
            inp[name] = self._gdn_states[name]
        for name in self._explicit_recurrent_inputs:
            if self._is_hybrid:
                # Feed FP32 shadow state -- NPU auto-converts to FP16
                inp[name] = self._fp32_recurrent_states[name]
            else:
                inp[name] = self._gdn_states[name]

    def _read_gdn_states(self) -> None:
        """Read updated GDN states from model outputs after infer()."""
        # Conv states: always read from NPU output (no accumulation issue)
        for past_name, present_name in zip(
            self._explicit_conv_inputs, self._explicit_conv_outputs
        ):
            self._gdn_states[past_name] = (
                self._request.get_tensor(present_name).data.copy()
            )

        if self._is_hybrid:
            # Hybrid mode: update FP32 shadow state using intermediates
            self._cpu_fp32_state_update()
        else:
            # Normal mode: use NPU's state output directly
            for past_name, present_name in zip(
                self._explicit_recurrent_inputs, self._explicit_recurrent_outputs
            ):
                self._gdn_states[past_name] = (
                    self._request.get_tensor(present_name).data.copy()
                )

    def _cpu_fp32_state_update(self) -> None:
        """Update FP32 shadow recurrent states using GDN intermediates from NPU.

        For each GDN layer i:
          S = S * g_t + outer(k_t, delta_t)
          where delta_t = (v_t - (S * k_t).sum(-2)) * beta_t

        All computation in FP32. Intermediates from NPU are FP16-precision
        but converted to FP32 for accumulation.
        """
        num_layers = len(self._explicit_recurrent_inputs)
        for i in range(num_layers):
            past_name = self._explicit_recurrent_inputs[i]
            S = self._fp32_recurrent_states[past_name]  # [B, H, D_k, D_v] FP32

            # Read intermediates from NPU output
            g_t = self._request.get_tensor(
                f"gdn_intermediate.{i}.g_t"
            ).data.copy().astype(np.float32)
            k_t = self._request.get_tensor(
                f"gdn_intermediate.{i}.k_t"
            ).data.copy().astype(np.float32)
            v_t = self._request.get_tensor(
                f"gdn_intermediate.{i}.v_t"
            ).data.copy().astype(np.float32)
            beta_t = self._request.get_tensor(
                f"gdn_intermediate.{i}.beta_t"
            ).data.copy().astype(np.float32)

            # g_t: [B, H, 1, 1] -- decay factor (already exp'd)
            # k_t: [B, H, D_k]
            # v_t: [B, H, D_v]
            # beta_t: [B, H, 1]

            # Decay
            S = S * g_t  # [B, H, D_k, D_v] * [B, H, 1, 1]

            # Read: mem = sum(S * k, axis=-2) -- matmul S @ k
            # S: [B, H, D_k, D_v], k_t: [B, H, D_k]
            # mem: [B, H, D_v]
            mem = np.einsum('bhkv,bhk->bhv', S, k_t)

            # Delta
            delta = (v_t - mem) * beta_t  # [B, H, D_v]

            # Write: S += outer(k, delta)
            # k_t: [B, H, D_k], delta: [B, H, D_v]
            S = S + np.einsum('bhk,bhv->bhkv', k_t, delta)

            self._fp32_recurrent_states[past_name] = S

    # -----------------------------------------------------------------
    # Static KV cache management (all-explicit NPU model)
    # -----------------------------------------------------------------

    def _init_kv_buffers(self) -> None:
        """Zero-initialize the static KV cache buffers."""
        for name in self._explicit_key_inputs + self._explicit_value_inputs:
            pshape = self._ov_model.input(name).get_partial_shape()
            shape = [dim.get_length() for dim in pshape]
            self._kv_buffers[name] = np.zeros(shape, dtype=np.float32)

    def _feed_kv_buffers(self, inp: Dict[str, Any]) -> None:
        """Add KV cache buffers to the input dict."""
        for name in self._explicit_key_inputs + self._explicit_value_inputs:
            inp[name] = self._kv_buffers[name]

    def _read_kv_outputs(self) -> None:
        """Read new K/V from outputs and write into the buffer at _past_length."""
        pos = self._past_length  # Position to write the new entry
        if pos >= self._max_cache_len:
            # Buffer full — shift left by 1 to make room
            for name in self._explicit_key_inputs + self._explicit_value_inputs:
                self._kv_buffers[name][:, :, :-1, :] = (
                    self._kv_buffers[name][:, :, 1:, :]
                )
            pos = self._max_cache_len - 1

        for past_name, present_name in zip(
            self._explicit_key_inputs, self._explicit_key_outputs
        ):
            new_kv = self._request.get_tensor(present_name).data.copy()
            self._kv_buffers[past_name][:, :, pos:pos+1, :] = new_kv

        for past_name, present_name in zip(
            self._explicit_value_inputs, self._explicit_value_outputs
        ):
            new_kv = self._request.get_tensor(present_name).data.copy()
            self._kv_buffers[past_name][:, :, pos:pos+1, :] = new_kv

    def _build_4d_mask(self) -> np.ndarray:
        """Build 4D float attention mask for static-cache model.

        Shape: [1, 1, 1, max_cache_len + 1]
        The mask covers the concatenated KV: [past_buffer(max_len) | current(1)]

        At step t (0-indexed, _past_length tokens already processed):
        - Positions 0..t-1: 0.0 (attend to valid past tokens)
        - Positions t..max_len-1: -inf (masked, unused buffer slots)
        - Position max_len: 0.0 (attend to current token)
        """
        total_len = self._max_cache_len + 1
        mask = np.full((1, 1, 1, total_len), -np.inf, dtype=np.float32)
        # Unmask valid past positions
        t = min(self._past_length, self._max_cache_len)
        if t > 0:
            mask[0, 0, 0, :t] = 0.0
        # Unmask current token position (always last in concatenated KV)
        mask[0, 0, 0, -1] = 0.0
        return mask

    def _reset_static_cache(self) -> None:
        """Reset all static-cache state for a new generation."""
        self._past_length = 0
        if self._has_explicit_gdn_states:
            self._init_gdn_states()
        self._init_kv_buffers()

    # -----------------------------------------------------------------
    # from_pretrained
    # -----------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        device: str = "CPU",
        tokenizer_path: Optional[Union[str, Path]] = None,
        ov_config: Optional[Dict[str, str]] = None,
    ) -> "Qwen35OVModel":
        """Load an exported OpenVINO IR and return a ready-to-use model.

        Parameters
        ----------
        model_path:
            Directory containing ``openvino_model.xml`` (and ``.bin``), plus
            ``config.json``.
        device:
            OpenVINO device string, e.g. ``"CPU"``, ``"GPU"``.
        tokenizer_path:
            Path to the tokenizer.  Defaults to *model_path* (works when the
            tokenizer files live next to the IR).
        ov_config:
            Extra properties forwarded to ``core.compile_model``.
        """
        model_path = Path(model_path)
        xml_path = model_path / "openvino_model.xml"
        if not xml_path.exists():
            raise FileNotFoundError(f"Cannot find {xml_path}")

        core = ov.Core()
        ov_model = core.read_model(str(xml_path))

        compile_props = ov_config or {}
        if device == "NPU" and not compile_props:
            from .config import NPU_OV_CONFIG
            compile_props = NPU_OV_CONFIG
            logger.info("NPU device: applying default NPUW config")

        # The IR is saved with FP16 compression, so outputs are f16.
        # CPU auto-promotes to f32, but GPU and NPU do not — add a
        # PrePostProcessor pass to convert outputs to f32 explicitly.
        if device in ("NPU", "GPU"):
            from openvino.preprocess import PrePostProcessor
            ppp = PrePostProcessor(ov_model)
            for i in range(len(ov_model.outputs)):
                ppp.output(i).tensor().set_element_type(ov.Type.f32)
            ov_model = ppp.build()

        logger.info("Compiling model on %s ...", device)
        compiled = core.compile_model(ov_model, device, compile_props)
        request = compiled.create_infer_request()

        try:
            config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
        except (ValueError, KeyError):
            # Fallback for older transformers that don't know "qwen3_5"
            import json
            with open(model_path / "config.json") as f:
                cfg_dict = json.load(f)
            from transformers import PretrainedConfig
            config = PretrainedConfig(**cfg_dict)

        tok_path = str(tokenizer_path) if tokenizer_path else str(model_path)
        tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

        embed_path = model_path / "embed_tokens.npy"
        if embed_path.exists():
            embed_table = np.load(str(embed_path))
        else:
            raise FileNotFoundError(f"Cannot find embed_tokens.npy at {embed_path}")

        return cls(
            model=ov_model,
            compiled_model=compiled,
            request=request,
            config=config,
            tokenizer=tokenizer,
            embed_table=embed_table,
            device_name=device,
            ov_config=compile_props,
        )

    # -----------------------------------------------------------------
    # Properties required by GenerationMixin
    # -----------------------------------------------------------------

    @staticmethod
    def can_generate() -> bool:  # noqa: D102
        return True

    @property
    def device(self) -> torch.device:  # noqa: D102
        return torch.device("cpu")

    # -----------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        cache_params: Optional[Qwen35CacheState] = None,
        **kwargs,
    ) -> Qwen35Output:
        """Run a single forward pass through the OpenVINO model.

        Three modes:
        * **Static-cache** (NPU all-explicit): all states are explicit I/O.
          Python manages KV buffers and 4D attention mask.
        * **Token-by-token prefill** (stateful + explicit GDN): seq_len=1
          per infer() call, stateful KV + explicit GDN states.
        * **Batch prefill** (CPU/GPU stateful): sends all tokens in one call.
        """
        if input_ids is None:
            raise ValueError("input_ids must be provided")

        batch_size, seq_len = input_ids.shape

        if self._is_static_cache:
            # ----- static-cache mode (NPU all-explicit) --------------------
            return self._forward_static_cache(input_ids, cache_params)

        if seq_len > 1 and self._token_by_token:
            # ----- token-by-token prefill (stateful KV + explicit GDN) -----
            if cache_params is None:
                self._request.reset_state()
                self._past_length = 0
                if self._has_explicit_gdn_states:
                    self._init_gdn_states()

            ids_np = input_ids.numpy().astype(np.int64)
            for i in range(seq_len):
                single_embed = self._embed_table[ids_np[:, i:i+1]].astype(np.float32)
                inp: Dict[str, Any] = {"inputs_embeds": single_embed}
                if "attention_mask" in self._input_names:
                    if self._is_npuw_llm:
                        # NPUW_LLM chunk prefill: mask covers past + current
                        inp["attention_mask"] = np.ones(
                            (batch_size, self._past_length + 1), dtype=np.int64
                        )
                    else:
                        inp["attention_mask"] = np.ones((batch_size, 1), dtype=np.int64)
                if "position_ids" in self._input_names:
                    inp["position_ids"] = np.full(
                        (3, batch_size, 1), self._past_length, dtype=np.int64,
                    )
                if "beam_idx" in self._input_names:
                    inp["beam_idx"] = np.zeros(batch_size, dtype=np.int64)
                if self._has_explicit_gdn_states:
                    self._feed_gdn_states(inp)
                self._request.infer(inp)
                if self._has_explicit_gdn_states:
                    self._read_gdn_states()
                self._past_length += 1

        elif seq_len > 1:
            # ----- prefill: batch infer (CPU/GPU) --------------------------
            if cache_params is None:
                self._request.reset_state()
                self._past_length = 0

            ids_np = input_ids.numpy().astype(np.int64)
            embeds = self._embed_table[ids_np].astype(np.float32)
            inp: Dict[str, Any] = {
                "inputs_embeds": embeds,
            }

            if "attention_mask" in self._input_names:
                inp["attention_mask"] = np.ones((batch_size, seq_len), dtype=np.int64)

            if "position_ids" in self._input_names:
                positions = np.arange(self._past_length, self._past_length + seq_len, dtype=np.int64)
                inp["position_ids"] = np.tile(positions[np.newaxis, np.newaxis, :], (3, batch_size, 1))

            if "beam_idx" in self._input_names:
                inp["beam_idx"] = np.zeros(batch_size, dtype=np.int64)

            if self._has_explicit_gdn_states:
                self._feed_gdn_states(inp)
            self._request.infer(inp)
            if self._has_explicit_gdn_states:
                self._read_gdn_states()
            self._past_length = seq_len
        else:
            # ----- decode: single token ------------------------------------
            ids_np = input_ids.numpy().astype(np.int64)
            embeds = self._embed_table[ids_np].astype(np.float32)
            inp: Dict[str, Any] = {
                "inputs_embeds": embeds,
            }

            if "attention_mask" in self._input_names:
                if self._is_npuw_llm:
                    # NPUW_LLM: mask covers full context (past + current)
                    inp["attention_mask"] = np.ones(
                        (batch_size, self._past_length + 1), dtype=np.int64
                    )
                else:
                    inp["attention_mask"] = np.ones((batch_size, 1), dtype=np.int64)

            if "position_ids" in self._input_names:
                pos = self._past_length
                inp["position_ids"] = np.full((3, batch_size, 1), pos, dtype=np.int64)

            if "beam_idx" in self._input_names:
                inp["beam_idx"] = np.zeros(batch_size, dtype=np.int64)

            if self._has_explicit_gdn_states:
                self._feed_gdn_states(inp)
            self._request.infer(inp)
            if self._has_explicit_gdn_states:
                self._read_gdn_states()
            self._past_length += 1

        # --- read logits ---------------------------------------------------
        logits = torch.from_numpy(
            self._request.get_tensor("logits").data.copy()
        )

        # --- read cache state from OV variables ----------------------------
        cache_state = Qwen35CacheState.from_query_state(
            self._request.query_state()
        )

        return Qwen35Output(logits=logits, cache_params=cache_state)

    def _forward_static_cache(
        self,
        input_ids: torch.LongTensor,
        cache_params: Optional[Qwen35CacheState],
    ) -> Qwen35Output:
        """Forward pass for NPU all-explicit static-cache model.

        All states (KV cache + GDN conv/recurrent) are explicit I/O.
        KV cache is a pre-allocated buffer; Python manages position tracking,
        4D attention mask, and buffer updates.
        """
        batch_size, seq_len = input_ids.shape
        ids_np = input_ids.numpy().astype(np.int64)

        if cache_params is None and self._past_length > 0:
            # New generation — reset everything
            self._reset_static_cache()

        for i in range(seq_len):
            token_id = ids_np[:, i:i+1]
            embed = self._embed_table[token_id].astype(np.float32)

            inp: Dict[str, Any] = {
                "inputs_embeds": embed,
                "attention_mask": self._build_4d_mask(),
                "position_ids": np.full(
                    (3, batch_size, 1),
                    min(self._past_length, self._max_cache_len - 1),
                    dtype=np.int64,
                ),
            }

            # Feed all explicit states
            self._feed_gdn_states(inp)
            self._feed_kv_buffers(inp)

            self._request.infer(inp)

            # Read back updated states
            self._read_gdn_states()
            self._read_kv_outputs()
            self._past_length += 1

        logits = torch.from_numpy(
            self._request.get_tensor("logits").data.copy()
        )
        return Qwen35Output(logits=logits, cache_params=Qwen35CacheState())

    # -----------------------------------------------------------------
    # generate() plumbing
    # -----------------------------------------------------------------

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        cache_params: Optional[Qwen35CacheState] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Prepare the dict that gets unpacked into ``forward()``.

        During the decode phase (``cache_position[0] > 0``) only the last
        token of *input_ids* is kept -- all prior context is already inside
        the OV stateful variables.
        """
        if cache_position is not None and cache_position[0] > 0:
            # Decode step -- take only the freshly appended token.
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "cache_params": cache_params,
            "cache_position": cache_position,
        }

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        num_new_tokens: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        """Propagate the cache state (attention_mask is NOT extended).

        For Qwen3.5 (SSM/hybrid model), the attention_mask always matches
        the current input length.  The model handles past context via its
        stateful KV/conv/recurrent variables.  This differs from standard
        transformers where attention_mask grows with each decode step.
        """
        model_kwargs["cache_params"] = outputs.get("cache_params", None)

        if (
            model_kwargs.get("use_cache", True)
            and "cache_position" in model_kwargs
            and model_kwargs["cache_position"] is not None
        ):
            model_kwargs["cache_position"] = (
                model_kwargs["cache_position"][-1:] + num_new_tokens
            )

        # Do NOT extend attention_mask -- Qwen3.5 expects it to match
        # the current input length (1 during decode), not the full context.

        return model_kwargs

    # -----------------------------------------------------------------
    # Convenience helpers
    # -----------------------------------------------------------------

    def __call__(self, *args, **kwargs) -> Qwen35Output:
        """Make the model callable (required by GenerationMixin._sample)."""
        return self.forward(*args, **kwargs)

    def reset(self) -> None:
        """Manually reset all OV states and the position counter."""
        self._request.reset_state()
        self._past_length = 0
        if self._has_explicit_gdn_states:
            self._init_gdn_states()

    def __repr__(self) -> str:
        model_type = getattr(self.config, "model_type", "unknown")
        return (
            f"Qwen35OVModel(model_type={model_type!r}, "
            f"device={self._device_name!r}, "
            f"states=conv:{len(self._conv_state_names)}+"
            f"recurrent:{len(self._recurrent_state_names)}+"
            f"key:{len(self._key_state_names)}+"
            f"value:{len(self._value_state_names)})"
        )


# ---------------------------------------------------------------------------
# VL (Vision-Language) model class
# ---------------------------------------------------------------------------

# Special token IDs for Qwen3.5-VL (from config.json)
_IMAGE_TOKEN_ID = 248056      # <|image_pad|>
_VIDEO_TOKEN_ID = 248057      # <|video_pad|>
_VISION_START_ID = 248053     # <|vision_start|>
_VISION_END_ID = 248054       # <|vision_end|>
_IM_START = 151644            # <|im_start|>
_IM_END = 151645              # <|im_end|>
_NEWLINE = 198                # \n


class Qwen35VLModel:
    """OpenVINO VL inference wrapper for Qwen3.5-VL.

    Uses inputs_embeds (not input_ids). Combines text embeddings
    with visual features from the vision encoder.

    Prefill sends the entire prompt in a single infer() call (batch
    prefill).  The GDN recurrence is handled by an OpenVINO Loop node
    that iterates over the sequence internally.
    """

    def __init__(
        self,
        decoder_xml: str,
        embed_table_npy: str,
        vision_encoder_xml: str,
        tokenizer_path: str,
        device: str = "CPU",
        ov_config: Optional[Dict[str, str]] = None,
    ):
        """Initialize VL model components.

        Args:
            decoder_xml: Path to decoder openvino_model.xml (inputs_embeds variant).
            embed_table_npy: Path to embed_tokens.npy [vocab_size, 1024].
            vision_encoder_xml: Path to vision_encoder.xml.
            tokenizer_path: Path to directory with tokenizer.json etc.
            device: OpenVINO device string for the decoder ("CPU", "GPU").
            ov_config: Extra compile properties for the decoder.
        """
        from .ov_vision_encoder import OVVisionEncoder

        core = ov.Core()

        # --- Decoder (inputs_embeds stateful model) ---
        compile_props = ov_config or {}
        if not compile_props and device in ("CPU", "GPU"):
            compile_props = {"PERFORMANCE_HINT": "LATENCY"}
        elif not compile_props and device == "NPU":
            from .config import NPU_OV_CONFIG
            compile_props = NPU_OV_CONFIG
            logger.info("NPU device: applying default NPUW config")

        ov_model = core.read_model(decoder_xml)

        # FP16 IR outputs need explicit f32 conversion on GPU/NPU
        # (CPU auto-promotes, GPU and NPU do not).
        if device in ("NPU", "GPU"):
            from openvino.preprocess import PrePostProcessor
            ppp = PrePostProcessor(ov_model)
            for i in range(len(ov_model.outputs)):
                ppp.output(i).tensor().set_element_type(ov.Type.f32)
            ov_model = ppp.build()

        logger.info("Compiling VL decoder on %s ...", device)
        self._compiled = core.compile_model(ov_model, device, compile_props)
        self._request = self._compiled.create_infer_request()
        self._device_name = device

        # Discover input names
        self._input_names = {
            inp.any_name for inp in self._compiled.inputs
        }
        logger.info(
            "Qwen35VLModel decoder: device=%s, inputs=%s",
            device,
            sorted(self._input_names),
        )

        # --- Embedding table (stored as float16 to save disk, cast on lookup) ---
        self._embed_table = np.load(embed_table_npy)
        logger.info(
            "Qwen35VLModel embed_table: shape=%s, dtype=%s",
            self._embed_table.shape,
            self._embed_table.dtype,
        )

        # Detect explicit GDN states (KV-only stateful mode for NPU)
        self._explicit_conv_inputs: List[str] = sorted(
            n for n in self._input_names if "cache_params.past.conv" in n
        )
        self._explicit_recurrent_inputs: List[str] = sorted(
            n for n in self._input_names if "cache_params.past.recurrent" in n
        )
        self._has_explicit_gdn_states = bool(
            self._explicit_conv_inputs or self._explicit_recurrent_inputs
        )

        _output_names: set = set()
        for out in self._compiled.outputs:
            _output_names.update(out.get_names())
        self._explicit_conv_outputs: List[str] = sorted(
            n for n in _output_names if "cache_params.present.conv" in n
        )
        self._explicit_recurrent_outputs: List[str] = sorted(
            n for n in _output_names if "cache_params.present.recurrent" in n
        )

        self._gdn_states: Dict[str, np.ndarray] = {}
        if self._has_explicit_gdn_states:
            self._vl_init_gdn_states()
            logger.info(
                "VL explicit GDN states: %d conv + %d recurrent",
                len(self._explicit_conv_inputs),
                len(self._explicit_recurrent_inputs),
            )

        # --- Vision encoder ---
        self._vision_encoder = OVVisionEncoder(vision_encoder_xml, device=device)

        # --- Tokenizer ---
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, trust_remote_code=True,
            )
        except (ValueError, KeyError):
            # Fallback: tokenizer saved by transformers 5.x may use
            # "backend": "tokenizers" which older transformers can't load.
            # Try loading from the text-only model directory instead.
            tok_dir = Path(tokenizer_path)
            text_ov_path = str(tok_dir.parent / tok_dir.name.replace("-vl", "-ov"))
            self._tokenizer = AutoTokenizer.from_pretrained(
                text_ov_path, trust_remote_code=True,
            )

        # Position tracking
        self._past_len = 0
        self._rope_delta = 0  # mRoPE offset for decode after image tokens

        # Load special token IDs from config if available
        self._image_token_id = _IMAGE_TOKEN_ID
        self._vision_start_id = _VISION_START_ID
        self._vision_end_id = _VISION_END_ID

    # -----------------------------------------------------------------
    # Explicit GDN state management (KV-only stateful mode)
    # -----------------------------------------------------------------

    def _vl_init_gdn_states(self) -> None:
        """Zero-initialize explicit GDN state tensors."""
        for name in self._explicit_conv_inputs + self._explicit_recurrent_inputs:
            pshape = self._compiled.input(name).get_partial_shape()
            shape = [
                dim.get_length() if dim.is_static else 1
                for dim in pshape
            ]
            self._gdn_states[name] = np.zeros(shape, dtype=np.float32)

    def _vl_feed_gdn_states(self, feed: Dict[str, Any]) -> None:
        """Add explicit GDN states to the feed dict."""
        for name in self._explicit_conv_inputs + self._explicit_recurrent_inputs:
            feed[name] = self._gdn_states[name]

    def _vl_read_gdn_states(self) -> None:
        """Read updated GDN states from model outputs after infer()."""
        for past_name, present_name in zip(
            self._explicit_conv_inputs, self._explicit_conv_outputs
        ):
            self._gdn_states[past_name] = self._request.get_tensor(present_name).data.copy()
        for past_name, present_name in zip(
            self._explicit_recurrent_inputs, self._explicit_recurrent_outputs
        ):
            self._gdn_states[past_name] = self._request.get_tensor(present_name).data.copy()

    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        device: str = "CPU",
        ov_config: Optional[Dict[str, str]] = None,
    ) -> "Qwen35VLModel":
        """Load from a VL model directory.

        Expected layout::

            model_path/
                vision_encoder.xml/.bin
                openvino_model.xml/.bin      (decoder with inputs_embeds)
                embed_tokens.npy
                config.json
                tokenizer.json
                tokenizer_config.json

        Args:
            model_path: Root directory of the VL model.
            device: OpenVINO device string.
            ov_config: Extra compile properties for the decoder.

        Returns:
            Ready-to-use Qwen35VLModel instance.
        """
        model_path = Path(model_path)

        decoder_xml = str(model_path / "openvino_model.xml")
        embed_table_npy = str(model_path / "embed_tokens.npy")
        vision_encoder_xml = str(model_path / "vision_encoder.xml")
        tokenizer_path = str(model_path)

        for p, label in [
            (decoder_xml, "decoder"),
            (embed_table_npy, "embed_tokens"),
            (vision_encoder_xml, "vision_encoder"),
        ]:
            if not Path(p).exists():
                raise FileNotFoundError(f"Cannot find {label}: {p}")

        return cls(
            decoder_xml=decoder_xml,
            embed_table_npy=embed_table_npy,
            vision_encoder_xml=vision_encoder_xml,
            tokenizer_path=tokenizer_path,
            device=device,
            ov_config=ov_config,
        )

    # -----------------------------------------------------------------
    # Vision encoder
    # -----------------------------------------------------------------

    def encode_image(
        self,
        pixel_values: np.ndarray,
        grid_thw: np.ndarray,
    ) -> np.ndarray:
        """Run vision encoder on preprocessed image data.

        Args:
            pixel_values: Preprocessed image patches (numpy array).
            grid_thw: Grid dimensions [N, 3] (temporal, height, width).

        Returns:
            visual_features: numpy array [1, num_visual_tokens, 1024].
        """
        return self._vision_encoder(pixel_values, grid_thw)

    # -----------------------------------------------------------------
    # Prompt construction
    # -----------------------------------------------------------------

    def build_prompt_tokens(
        self,
        prompt: str,
        num_visual_tokens: int,
        system_prompt: str = "You are a helpful assistant.",
    ) -> List[int]:
        """Build the full ChatML token sequence for a VL query.

        Format::

            <|im_start|>system
            {system_prompt}<|im_end|>
            <|im_start|>user
            <|vision_start|><|image_pad|>*N<|vision_end|>
            {prompt}<|im_end|>
            <|im_start|>assistant


        Args:
            prompt: User text prompt (e.g. "Describe this image").
            num_visual_tokens: Number of visual tokens from the encoder.
            system_prompt: System instruction text.

        Returns:
            List of token IDs.
        """
        # System turn
        tokens = (
            [_IM_START]
            + self._tokenizer.encode("system\n" + system_prompt)
            + [_IM_END, _NEWLINE]
        )

        # User turn with image
        tokens += (
            [_IM_START]
            + self._tokenizer.encode("user\n")
            + [self._vision_start_id]
            + [self._image_token_id] * num_visual_tokens
            + [self._vision_end_id]
            + [_NEWLINE]
            + self._tokenizer.encode(prompt)
            + [_IM_END, _NEWLINE]
        )

        # Assistant turn (model generates from here)
        tokens += (
            [_IM_START]
            + self._tokenizer.encode("assistant\n")
        )

        return tokens

    # -----------------------------------------------------------------
    # Inputs embeds construction
    # -----------------------------------------------------------------

    def build_inputs_embeds(
        self,
        token_ids: List[int],
        visual_features: np.ndarray,
    ) -> np.ndarray:
        """Build inputs_embeds by replacing image placeholder tokens with visual features.

        Similar to ASR's _build_inputs_embeds in engine.py:
        1. Look up text embeddings from embed_table for all tokens
        2. Find positions of image_pad tokens (image_token_id)
        3. Replace those positions with visual_features

        Args:
            token_ids: List of token IDs (includes image placeholder tokens).
            visual_features: numpy array, shape [num_visual_tokens, 1024] or
                [1, num_visual_tokens, 1024].

        Returns:
            inputs_embeds: numpy array [seq_len, 1024].
        """
        # Squeeze batch dim if present
        if visual_features.ndim == 3:
            visual_features = visual_features[0]  # [num_visual_tokens, 1024]

        ids = np.array(token_ids, dtype=np.int64)
        embeds = self._embed_table[ids].astype(np.float32)  # [seq_len, 1024]

        # Replace image_pad positions with visual features
        image_positions = np.where(ids == self._image_token_id)[0]
        if len(image_positions) != visual_features.shape[0]:
            logger.warning(
                "Mismatch: %d image_pad tokens vs %d visual features. "
                "Replacing min(%d, %d) positions.",
                len(image_positions),
                visual_features.shape[0],
                len(image_positions),
                visual_features.shape[0],
            )
        n_replace = min(len(image_positions), visual_features.shape[0])
        embeds[image_positions[:n_replace]] = visual_features[:n_replace]

        return embeds

    # -----------------------------------------------------------------
    # mRoPE position IDs
    # -----------------------------------------------------------------

    def build_mrope_position_ids(
        self,
        token_ids: List[int],
        image_grid_thw: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Build 3D mRoPE position IDs for the full prompt.

        For text-only tokens, all 3 dims (temporal, height, width) share
        the same sequential position.  For visual tokens (image_pad), the
        3 dims encode spatial coordinates of the 2D grid:
          - temporal: constant (start_pos for that image)
          - height:   row index within the merged grid
          - width:    column index within the merged grid

        Args:
            token_ids: Full prompt token IDs.
            image_grid_thw: Grid dimensions [N, 3] where each row is
                (temporal, height, width) in patch units.  If None,
                treats all tokens as text (all 3 dims identical).

        Returns:
            position_ids: Shape [3, 1, seq_len] (int64).
            Also sets self._rope_delta for decode phase.
        """
        from .config import MROPE_SECTION, SPATIAL_MERGE_SIZE

        seq_len = len(token_ids)
        ids = np.array(token_ids, dtype=np.int64)

        # Find image_pad token positions
        image_positions = np.where(ids == self._image_token_id)[0]

        if len(image_positions) == 0 or image_grid_thw is None:
            # Text-only: all 3 dims identical
            positions = np.arange(seq_len, dtype=np.int64)
            position_ids = np.tile(positions[np.newaxis, np.newaxis, :], (3, 1, 1))
            self._rope_delta = 0
            return position_ids

        # Build per-token positions for each of the 3 mRoPE dims
        t_pos = np.zeros(seq_len, dtype=np.int64)
        h_pos = np.zeros(seq_len, dtype=np.int64)
        w_pos = np.zeros(seq_len, dtype=np.int64)

        # Process token by token
        text_pos = 0  # running text position counter
        img_idx = 0   # which image we're processing
        i = 0
        while i < seq_len:
            if i in set(image_positions):
                # Start of an image_pad block
                # Find contiguous block of image_pad tokens
                block_start = i
                while i < seq_len and ids[i] == self._image_token_id:
                    i += 1
                block_end = i
                num_visual_tokens = block_end - block_start

                # Get grid dims for this image
                if img_idx < len(image_grid_thw):
                    t_grid, h_grid, w_grid = image_grid_thw[img_idx]
                else:
                    # Fallback: assume square grid
                    side = int(np.sqrt(num_visual_tokens))
                    t_grid, h_grid, w_grid = 1, side, side

                # Merged grid dimensions (after spatial merge)
                h_merged = h_grid // SPATIAL_MERGE_SIZE
                w_merged = w_grid // SPATIAL_MERGE_SIZE

                start_pos = text_pos

                # Assign spatial positions to visual tokens
                for vi in range(num_visual_tokens):
                    row = vi // w_merged
                    col = vi % w_merged
                    t_pos[block_start + vi] = start_pos
                    h_pos[block_start + vi] = start_pos + row
                    w_pos[block_start + vi] = start_pos + col

                # Text position advances by max(h_merged, w_merged)
                text_pos = start_pos + max(h_merged, w_merged)
                img_idx += 1
            else:
                # Text token: all 3 dims identical
                t_pos[i] = text_pos
                h_pos[i] = text_pos
                w_pos[i] = text_pos
                text_pos += 1
                i += 1

        # rope_delta = max_position + 1 - total_seq_len
        max_position = max(t_pos.max(), h_pos.max(), w_pos.max())
        self._rope_delta = int(max_position + 1 - seq_len)

        # Stack: [3, 1, seq_len]
        position_ids = np.stack([t_pos, h_pos, w_pos], axis=0)[:, np.newaxis, :]
        return position_ids

    # -----------------------------------------------------------------
    # Decoder: prefill / decode_step / generate
    # -----------------------------------------------------------------

    def reset(self) -> None:
        """Reset KV-cache state for a new generation."""
        self._request.reset_state()
        self._past_len = 0
        self._rope_delta = 0
        if self._has_explicit_gdn_states:
            self._vl_init_gdn_states()

    def prefill(
        self,
        inputs_embeds: np.ndarray,
        position_ids: Optional[np.ndarray] = None,
    ) -> int:
        """Prefill the prompt.

        On CPU/GPU: batch prefill (single infer() call, Loop node handles GDN).
        On NPU: token-by-token prefill (no Loop node in NPU IR).

        Args:
            inputs_embeds: Shape [seq_len, hidden_size] (no batch dim).
            position_ids: Shape [3, 1, seq_len] mRoPE positions.  If None,
                uses simple sequential positions.

        Returns:
            First predicted token ID (argmax of last-token logits).
        """
        seq_len = inputs_embeds.shape[0]
        self._request.reset_state()
        if self._has_explicit_gdn_states:
            self._vl_init_gdn_states()

        if self._device_name == "NPU":
            # ----- NPU: token-by-token prefill ----------------------------
            embeds_f32 = inputs_embeds.astype(np.float32)
            for i in range(seq_len):
                token_embed = embeds_f32[np.newaxis, i:i+1, :]  # [1, 1, hidden]
                feed: Dict[str, Any] = {"inputs_embeds": token_embed}
                if "attention_mask" in self._input_names:
                    feed["attention_mask"] = np.ones((1, 1), dtype=np.int64)
                if "position_ids" in self._input_names:
                    if position_ids is not None:
                        # Use the per-token position from the full position_ids
                        feed["position_ids"] = position_ids[:, :, i:i+1]
                    else:
                        feed["position_ids"] = np.full((3, 1, 1), i, dtype=np.int64)
                if "beam_idx" in self._input_names:
                    feed["beam_idx"] = np.array([0], dtype=np.int32)
                if self._has_explicit_gdn_states:
                    self._vl_feed_gdn_states(feed)
                self._request.infer(feed)
                if self._has_explicit_gdn_states:
                    self._vl_read_gdn_states()
            self._past_len = seq_len
        else:
            # ----- CPU/GPU: batch prefill ---------------------------------
            embeds_batch = inputs_embeds[np.newaxis, :, :].astype(np.float32)
            feed: Dict[str, Any] = {"inputs_embeds": embeds_batch}

            if "attention_mask" in self._input_names:
                feed["attention_mask"] = np.ones((1, seq_len), dtype=np.int64)

            if "position_ids" in self._input_names:
                if position_ids is not None:
                    feed["position_ids"] = position_ids
                else:
                    positions = np.arange(seq_len, dtype=np.int64)
                    feed["position_ids"] = np.tile(positions[np.newaxis, np.newaxis, :], (3, 1, 1))

            if "beam_idx" in self._input_names:
                feed["beam_idx"] = np.array([0], dtype=np.int32)

            if self._has_explicit_gdn_states:
                self._vl_feed_gdn_states(feed)
            self._request.infer(feed)
            if self._has_explicit_gdn_states:
                self._vl_read_gdn_states()
            self._past_len = seq_len

        logits = self._request.get_output_tensor(0).data
        return int(np.argmax(logits[0, -1, :]))

    def decode_step(self, token_id: int) -> int:
        """Single decode step with one token.

        Args:
            token_id: Previous token ID to continue from.

        Returns:
            Next predicted token ID.
        """
        token_embed = self._embed_table[token_id].astype(np.float32)
        # [1, 1, 1024]
        token_embed = token_embed[np.newaxis, np.newaxis, :]

        feed: Dict[str, Any] = {
            "inputs_embeds": token_embed,
        }

        if "attention_mask" in self._input_names:
            feed["attention_mask"] = np.ones((1, 1), dtype=np.int64)

        if "position_ids" in self._input_names:
            # Decode position accounts for rope_delta from image tokens
            pos = self._past_len + self._rope_delta
            feed["position_ids"] = np.full((3, 1, 1), pos, dtype=np.int64)

        if "beam_idx" in self._input_names:
            feed["beam_idx"] = np.array([0], dtype=np.int32)

        if self._has_explicit_gdn_states:
            self._vl_feed_gdn_states(feed)
        self._request.infer(feed)
        if self._has_explicit_gdn_states:
            self._vl_read_gdn_states()
        self._past_len += 1

        logits = self._request.get_output_tensor(0).data
        return int(np.argmax(logits[0, -1, :]))

    def generate(
        self,
        pixel_values: np.ndarray,
        grid_thw: np.ndarray,
        prompt: str,
        max_new_tokens: int = 100,
        system_prompt: str = "You are a helpful assistant.",
    ) -> str:
        """Full VL generation: encode image + build prompt + prefill + decode.

        Args:
            pixel_values: Preprocessed image patches (from image preprocessor).
            grid_thw: Grid dimensions [N, 3].
            prompt: Text prompt (e.g., "Describe this image").
            max_new_tokens: Max tokens to generate.
            system_prompt: System instruction.

        Returns:
            Generated text string.
        """
        # 1. Run vision encoder
        visual_features = self.encode_image(pixel_values, grid_thw)
        num_visual_tokens = visual_features.shape[1]  # [1, N, 1024]

        logger.info(
            "Visual features: shape=%s, num_tokens=%d",
            visual_features.shape,
            num_visual_tokens,
        )

        # 2. Build prompt tokens
        token_ids = self.build_prompt_tokens(
            prompt, num_visual_tokens, system_prompt
        )

        logger.info("Prompt: %d tokens", len(token_ids))

        # 3. Build inputs_embeds (merge text embeddings + visual features)
        inputs_embeds = self.build_inputs_embeds(token_ids, visual_features)

        # 3b. Build mRoPE position IDs (spatial positions for image tokens)
        image_grid_thw = grid_thw  # [N, 3] from preprocessor
        position_ids = self.build_mrope_position_ids(token_ids, image_grid_thw)
        logger.info(
            "mRoPE position_ids: shape=%s, rope_delta=%d",
            position_ids.shape,
            self._rope_delta,
        )

        # 4. Reset state and prefill
        self.reset()
        token_id = self.prefill(inputs_embeds, position_ids=position_ids)

        # 5. Greedy decode loop
        generated_ids = []
        for _ in range(max_new_tokens):
            if token_id == _IM_END:
                break
            generated_ids.append(token_id)
            token_id = self.decode_step(token_id)

        # 6. Decode to text
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        return text

    def __repr__(self) -> str:
        return (
            f"Qwen35VLModel(device={self._device_name!r}, "
            f"embed_table={self._embed_table.shape}, "
            f"decoder_inputs={sorted(self._input_names)})"
        )


# ---------------------------------------------------------------------------
# Multi-subgraph model (6 subgraphs x 4 layers, FP32 between subgraphs)
# ---------------------------------------------------------------------------


class Qwen35MultiSubgraphModel(GenerationMixin):
    """Multi-subgraph NPU inference for Qwen3.5 with FP32 inter-subgraph precision.

    Loads 6 subgraph IRs (each covering 4 layers), chains them together with
    FP32 hidden_states between subgraphs. Each subgraph runs on NPU in FP16,
    but hidden_states are read back to CPU as FP32 between subgraphs, limiting
    FP16 error accumulation to 4 layers instead of 24.

    GDN recurrent states are maintained as FP32 shadow copies on CPU, updated
    using intermediates from NPU (same as Qwen35OVModel hybrid mode).
    """

    main_input_name = "input_ids"
    _supports_cache_class = False
    _is_stateful = True

    def __init__(
        self,
        subgraph_models: List[ov.Model],
        subgraph_requests: List[ov.InferRequest],
        config,
        tokenizer,
        embed_table: np.ndarray,
        max_cache_len: int = 256,
    ):
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer
        self._embed_table = embed_table
        self._num_subgraphs = len(subgraph_models)
        self._subgraph_models = subgraph_models
        self._subgraph_requests = subgraph_requests
        self._max_cache_len = max_cache_len
        self._past_length: int = 0

        # Discover input/output names per subgraph
        self._sub_input_names: List[set] = []
        self._sub_output_names: List[set] = []
        for m in subgraph_models:
            inp_names = set()
            for inp in m.inputs:
                inp_names.update(inp.get_names())
            self._sub_input_names.append(inp_names)
            out_names = set()
            for out in m.outputs:
                out_names.update(out.get_names())
            self._sub_output_names.append(out_names)

        # Per-subgraph state: conv, recurrent (FP32 shadow), KV buffers
        # Each subgraph has 3 conv + 3 recurrent + 1 key + 1 value
        self._conv_states: List[Dict[str, np.ndarray]] = []
        self._fp32_recurrent_states: List[Dict[str, np.ndarray]] = []
        self._kv_buffers: List[Dict[str, np.ndarray]] = []

        for si in range(self._num_subgraphs):
            conv_dict = {}
            rec_dict = {}
            kv_dict = {}
            model = subgraph_models[si]
            for inp in model.inputs:
                name = inp.get_any_name()
                pshape = inp.get_partial_shape()
                shape = [dim.get_length() if dim.is_static else 1 for dim in pshape]
                if "cache_params.past.conv" in name:
                    conv_dict[name] = np.zeros(shape, dtype=np.float32)
                elif "cache_params.past.recurrent" in name:
                    rec_dict[name] = np.zeros(shape, dtype=np.float32)
                elif "cache_params.past.key" in name or "cache_params.past.value" in name:
                    kv_dict[name] = np.zeros(shape, dtype=np.float32)
            self._conv_states.append(conv_dict)
            self._fp32_recurrent_states.append(rec_dict)
            self._kv_buffers.append(kv_dict)

        # generation_config
        try:
            self.generation_config = GenerationConfig.from_model_config(config)
        except Exception:
            self.generation_config = GenerationConfig()

        logger.info(
            "Qwen35MultiSubgraphModel: %d subgraphs, max_cache_len=%d",
            self._num_subgraphs, max_cache_len,
        )

    # -----------------------------------------------------------------
    # Factory
    # -----------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        device: str = "NPU",
        ov_config: Optional[Dict[str, str]] = None,
    ) -> "Qwen35MultiSubgraphModel":
        """Load multi-subgraph IRs from a directory."""
        model_path = Path(model_path)
        core = ov.Core()

        # Find all subgraph files
        subgraph_files = sorted(model_path.glob("subgraph_*.xml"))
        if not subgraph_files:
            raise FileNotFoundError(f"No subgraph_*.xml files in {model_path}")

        num_subs = len(subgraph_files)
        logger.info("Loading %d subgraph IRs from %s ...", num_subs, model_path)

        subgraph_models = []
        subgraph_requests = []

        compile_props = ov_config or {}
        if device == "NPU" and not compile_props:
            from .config import MULTISUB_OV_CONFIG
            compile_props = MULTISUB_OV_CONFIG

        for i, xml_path in enumerate(subgraph_files):
            logger.info("  Loading subgraph %d: %s", i, xml_path.name)
            ov_model = core.read_model(str(xml_path))

            # Add F32 output conversion for NPU/GPU
            if device in ("NPU", "GPU"):
                from openvino.preprocess import PrePostProcessor
                ppp = PrePostProcessor(ov_model)
                for j in range(len(ov_model.outputs)):
                    ppp.output(j).tensor().set_element_type(ov.Type.f32)
                ov_model = ppp.build()

            logger.info("  Compiling subgraph %d on %s ...", i, device)
            compiled = core.compile_model(ov_model, device, compile_props)
            request = compiled.create_infer_request()
            subgraph_models.append(ov_model)
            subgraph_requests.append(request)

        # Load config and tokenizer
        try:
            config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
        except (ValueError, KeyError):
            import json
            with open(model_path / "config.json") as f:
                cfg_dict = json.load(f)
            from transformers import PretrainedConfig
            config = PretrainedConfig(**cfg_dict)

        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

        embed_path = model_path / "embed_tokens.npy"
        if embed_path.exists():
            embed_table = np.load(str(embed_path))
        else:
            raise FileNotFoundError(f"Cannot find embed_tokens.npy at {embed_path}")

        # Infer max_cache_len from first subgraph's KV input shape
        max_cache_len = 256
        for inp in subgraph_models[0].inputs:
            name = inp.get_any_name()
            if "cache_params.past.key" in name:
                max_cache_len = inp.get_partial_shape()[2].get_length()
                break

        return cls(
            subgraph_models=subgraph_models,
            subgraph_requests=subgraph_requests,
            config=config,
            tokenizer=tokenizer,
            embed_table=embed_table,
            max_cache_len=max_cache_len,
        )

    # -----------------------------------------------------------------
    # Properties required by GenerationMixin
    # -----------------------------------------------------------------

    @staticmethod
    def can_generate() -> bool:
        return True

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    # -----------------------------------------------------------------
    # State management
    # -----------------------------------------------------------------

    def _reset_states(self) -> None:
        """Reset all states for a new generation."""
        self._past_length = 0
        for si in range(self._num_subgraphs):
            for name in self._conv_states[si]:
                self._conv_states[si][name] = np.zeros_like(self._conv_states[si][name])
            for name in self._fp32_recurrent_states[si]:
                self._fp32_recurrent_states[si][name] = np.zeros_like(
                    self._fp32_recurrent_states[si][name]
                )
            for name in self._kv_buffers[si]:
                self._kv_buffers[si][name] = np.zeros_like(self._kv_buffers[si][name])

    def _build_4d_mask(self) -> np.ndarray:
        """Build 4D float attention mask for static-cache subgraphs."""
        total_len = self._max_cache_len + 1
        mask = np.full((1, 1, 1, total_len), -np.inf, dtype=np.float32)
        t = min(self._past_length, self._max_cache_len)
        if t > 0:
            mask[0, 0, 0, :t] = 0.0
        mask[0, 0, 0, -1] = 0.0
        return mask

    def _cpu_fp32_state_update(self, sub_idx: int) -> None:
        """Update FP32 shadow recurrent states for a subgraph using GDN intermediates."""
        request = self._subgraph_requests[sub_idx]
        out_names = self._sub_output_names[sub_idx]

        # Find recurrent input names for this subgraph (sorted by index)
        rec_names = sorted(
            self._fp32_recurrent_states[sub_idx].keys(),
            key=lambda n: int(n.rsplit(".", 1)[1])
        )

        for i, past_name in enumerate(rec_names):
            S = self._fp32_recurrent_states[sub_idx][past_name]  # [B, H, D_k, D_v]

            g_t = request.get_tensor(
                f"gdn_intermediate.{i}.g_t"
            ).data.copy().astype(np.float32)
            k_t = request.get_tensor(
                f"gdn_intermediate.{i}.k_t"
            ).data.copy().astype(np.float32)
            v_t = request.get_tensor(
                f"gdn_intermediate.{i}.v_t"
            ).data.copy().astype(np.float32)
            beta_t = request.get_tensor(
                f"gdn_intermediate.{i}.beta_t"
            ).data.copy().astype(np.float32)

            # Decay
            S = S * g_t

            # Read
            mem = np.einsum('bhkv,bhk->bhv', S, k_t)

            # Delta
            delta = (v_t - mem) * beta_t

            # Write
            S = S + np.einsum('bhk,bhv->bhkv', k_t, delta)

            self._fp32_recurrent_states[sub_idx][past_name] = S

    def _read_kv_outputs(self, sub_idx: int) -> None:
        """Read new KV from subgraph outputs and write into buffers."""
        request = self._subgraph_requests[sub_idx]
        pos = self._past_length
        if pos >= self._max_cache_len:
            # Buffer full - shift left
            for name in self._kv_buffers[sub_idx]:
                self._kv_buffers[sub_idx][name][:, :, :-1, :] = (
                    self._kv_buffers[sub_idx][name][:, :, 1:, :]
                )
            pos = self._max_cache_len - 1

        kv_dict = self._kv_buffers[sub_idx]
        for past_name in sorted(kv_dict.keys()):
            # Derive present name: cache_params.past.key.0 -> cache_params.present.key.0
            present_name = past_name.replace(".past.", ".present.")
            new_kv = request.get_tensor(present_name).data.copy()
            kv_dict[past_name][:, :, pos:pos+1, :] = new_kv

    # -----------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        cache_params: Optional[Qwen35CacheState] = None,
        **kwargs,
    ) -> Qwen35Output:
        """Run a single forward pass through all 6 subgraphs."""
        if input_ids is None:
            raise ValueError("input_ids must be provided")

        batch_size, seq_len = input_ids.shape
        ids_np = input_ids.numpy().astype(np.int64)

        if cache_params is None and self._past_length > 0:
            self._reset_states()

        for token_pos in range(seq_len):
            token_id = ids_np[:, token_pos:token_pos+1]
            # Embedding lookup (FP32, CPU)
            hidden = self._embed_table[token_id].astype(np.float32)  # [1, 1, 1024]

            mask = self._build_4d_mask()
            pos_ids = np.full(
                (3, batch_size, 1),
                min(self._past_length, self._max_cache_len - 1),
                dtype=np.int64,
            )

            for si in range(self._num_subgraphs):
                # Build input dict for this subgraph
                inp: Dict[str, Any] = {
                    "hidden_states": hidden,
                    "attention_mask": mask,
                    "position_ids": pos_ids,
                }

                # Feed conv states
                for name, state in self._conv_states[si].items():
                    inp[name] = state

                # Feed FP32 recurrent states (NPU auto-truncates to FP16)
                for name, state in self._fp32_recurrent_states[si].items():
                    inp[name] = state

                # Feed KV buffers
                for name, state in self._kv_buffers[si].items():
                    inp[name] = state

                # Run subgraph on NPU
                self._subgraph_requests[si].infer(inp)

                # Read conv states (direct from NPU output)
                for past_name in self._conv_states[si]:
                    present_name = past_name.replace(".past.", ".present.")
                    self._conv_states[si][past_name] = (
                        self._subgraph_requests[si].get_tensor(present_name).data.copy()
                    )

                # CPU FP32 recurrent state update
                self._cpu_fp32_state_update(si)

                # Read KV outputs
                self._read_kv_outputs(si)

                # Read hidden_states output as FP32 for next subgraph
                hidden = self._subgraph_requests[si].get_tensor(
                    "hidden_states"
                ).data.copy().astype(np.float32)

            self._past_length += 1

        # Read logits from last subgraph
        logits = torch.from_numpy(
            self._subgraph_requests[-1].get_tensor("logits").data.copy()
        )

        return Qwen35Output(logits=logits, cache_params=Qwen35CacheState())

    # -----------------------------------------------------------------
    # generate() plumbing
    # -----------------------------------------------------------------

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        cache_params: Optional[Qwen35CacheState] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if cache_position is not None and cache_position[0] > 0:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "cache_params": cache_params,
            "cache_position": cache_position,
        }

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        num_new_tokens: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        model_kwargs["cache_params"] = outputs.get("cache_params", None)
        if (
            model_kwargs.get("use_cache", True)
            and "cache_position" in model_kwargs
            and model_kwargs["cache_position"] is not None
        ):
            model_kwargs["cache_position"] = (
                model_kwargs["cache_position"][-1:] + num_new_tokens
            )
        return model_kwargs

    def __call__(self, *args, **kwargs) -> Qwen35Output:
        return self.forward(*args, **kwargs)

    def reset(self) -> None:
        self._reset_states()

    def __repr__(self) -> str:
        model_type = getattr(self.config, "model_type", "unknown")
        return (
            f"Qwen35MultiSubgraphModel(model_type={model_type!r}, "
            f"subgraphs={self._num_subgraphs}, "
            f"max_cache_len={self._max_cache_len})"
        )
