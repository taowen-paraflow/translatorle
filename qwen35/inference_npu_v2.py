"""NPU v2 inference: chain 6 subgraph IRs with host-side rotary precomputation.

Key difference from Qwen35MultiSubgraphModel: subgraphs take precomputed cos/sin
as explicit inputs instead of position_ids. Rotary embeddings are computed on the
host in FP32 using saved inv_freq and mRoPE config.

Usage:
    from qwen35.inference_npu_v2 import Qwen35NPUv2Model
    model = Qwen35NPUv2Model.from_pretrained("models/qwen35/Qwen3.5-0.8B-npu-v2")
    outputs = model.generate(**inputs, max_new_tokens=50)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import openvino as ov
import torch
from transformers import AutoConfig, AutoTokenizer, GenerationConfig
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import ModelOutput

from .inference import Qwen35CacheState, Qwen35Output

logger = logging.getLogger(__name__)


class Qwen35NPUv2Model(GenerationMixin):
    """Multi-subgraph NPU inference with host-side rotary precomputation.

    Loads 6 subgraph IRs (each covering 4 layers), chains them together with
    FP32 hidden_states between subgraphs.  Each subgraph takes precomputed
    cos/sin as explicit inputs instead of computing rotary embeddings internally.

    GDN recurrent states are maintained as FP32 shadow copies on CPU, updated
    using intermediates from NPU (same as Qwen35MultiSubgraphModel).
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
        inv_freq: np.ndarray,
        rotary_config: dict,
        max_cache_len: int = 256,
    ):
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer
        self._embed_table = embed_table  # float16, cast to float32 on lookup
        self._num_subgraphs = len(subgraph_models)
        self._subgraph_models = subgraph_models
        self._subgraph_requests = subgraph_requests
        self._max_cache_len = max_cache_len
        self._past_length: int = 0

        # Rotary parameters for host-side precomputation
        self._inv_freq = inv_freq.astype(np.float32)  # [dim/2]
        self._mrope_section = rotary_config["mrope_section"]  # [11, 11, 10]
        self._rotary_dim = rotary_config["rotary_dim"]  # 64

        # attention_scaling is saved by the export (from model.rotary_emb.attention_scaling)
        # For default rope_type it's 1.0; for yarn/longrope it may differ
        self._attention_scaling = float(rotary_config.get("attention_scaling", 1.0))

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
            "Qwen35NPUv2Model: %d subgraphs, max_cache_len=%d, rotary_dim=%d",
            self._num_subgraphs, max_cache_len, self._rotary_dim,
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
    ) -> "Qwen35NPUv2Model":
        """Load NPU v2 multi-subgraph IRs from a directory."""
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

            logger.info("  Compiling subgraph %d on %s ...", i, device)
            compiled = core.compile_model(ov_model, device, compile_props)
            request = compiled.create_infer_request()
            subgraph_models.append(ov_model)
            subgraph_requests.append(request)

        # Load config and tokenizer
        try:
            config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
        except (ValueError, KeyError):
            with open(model_path / "config.json") as f:
                cfg_dict = json.load(f)
            from transformers import PretrainedConfig
            config = PretrainedConfig(**cfg_dict)

        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

        # Load embedding table
        embed_path = model_path / "embed_tokens.npy"
        if embed_path.exists():
            embed_table = np.load(str(embed_path))
        else:
            raise FileNotFoundError(f"Cannot find embed_tokens.npy at {embed_path}")

        # Load rotary parameters
        inv_freq_path = model_path / "rotary_inv_freq.npy"
        if inv_freq_path.exists():
            inv_freq = np.load(str(inv_freq_path))
        else:
            raise FileNotFoundError(f"Cannot find rotary_inv_freq.npy at {inv_freq_path}")

        rotary_config_path = model_path / "rotary_config.json"
        if rotary_config_path.exists():
            with open(rotary_config_path) as f:
                rotary_config = json.load(f)
        else:
            raise FileNotFoundError(f"Cannot find rotary_config.json at {rotary_config_path}")

        # Infer max_cache_len from first subgraph's KV input shape
        max_cache_len = rotary_config.get("max_cache_len", 256)
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
            inv_freq=inv_freq,
            rotary_config=rotary_config,
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
    # Host-side rotary precomputation (mRoPE)
    # -----------------------------------------------------------------

    def _compute_rotary(self, position: int) -> tuple:
        """Compute mRoPE cos/sin for a single token position.

        For text-only inference, all 3 mRoPE dimensions (temporal, height, width)
        use the same position value. The inv_freq is split by mrope_section and
        each section uses its corresponding position dimension's frequencies.

        Returns:
            (cos, sin) each of shape [1, 1, rotary_dim] as float32
        """
        pos = np.float32(position)
        sections = self._mrope_section  # [11, 11, 10]

        cos_parts = []
        sin_parts = []
        offset = 0
        for d in range(3):
            sec = sections[d]
            # For text-only, all 3 dimensions use the same position
            freqs = pos * self._inv_freq[offset:offset + sec]  # [sec]
            cos_parts.append(np.cos(freqs))
            sin_parts.append(np.sin(freqs))
            offset += sec

        cos_interleaved = np.concatenate(cos_parts)  # [32]
        sin_interleaved = np.concatenate(sin_parts)  # [32]

        # Double (repeat) to get full rotary_dim
        cos_emb = np.concatenate([cos_interleaved, cos_interleaved])  # [64]
        sin_emb = np.concatenate([sin_interleaved, sin_interleaved])  # [64]

        # Apply attention_scaling
        cos_emb = (cos_emb * self._attention_scaling).astype(np.float32)
        sin_emb = (sin_emb * self._attention_scaling).astype(np.float32)

        # Reshape to [1, 1, rotary_dim]
        cos_emb = cos_emb.reshape(1, 1, -1)
        sin_emb = sin_emb.reshape(1, 1, -1)

        return cos_emb, sin_emb

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
            new_kv = request.get_tensor(present_name).data.copy().astype(np.float32)
            kv_dict[past_name][:, :, pos:pos + 1, :] = new_kv

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
            token_id = ids_np[:, token_pos:token_pos + 1]
            # Embedding lookup (FP32, CPU)
            hidden = self._embed_table[token_id].astype(np.float32)  # [1, 1, 1024]

            # Host-side rotary precomputation
            effective_pos = min(self._past_length, self._max_cache_len - 1)
            cos, sin = self._compute_rotary(effective_pos)

            # Build 4D attention mask
            mask = self._build_4d_mask()

            for si in range(self._num_subgraphs):
                # Build input dict for this subgraph
                inp: Dict[str, Any] = {
                    "hidden_states": hidden,
                    "cos": cos,
                    "sin": sin,
                    "attention_mask": mask,
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

                # Run subgraph
                self._subgraph_requests[si].infer(inp)

                # Read conv states (direct from output, cast to FP32)
                for past_name in self._conv_states[si]:
                    present_name = past_name.replace(".past.", ".present.")
                    self._conv_states[si][past_name] = (
                        self._subgraph_requests[si].get_tensor(present_name)
                        .data.copy().astype(np.float32)
                    )

                # CPU FP32 recurrent state update
                self._cpu_fp32_state_update(si)

                # Read KV outputs
                self._read_kv_outputs(si)

                # Read hidden_states output as FP32 for next subgraph
                hidden = self._subgraph_requests[si].get_tensor(
                    "output_hidden_states"
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
            f"Qwen35NPUv2Model(model_type={model_type!r}, "
            f"subgraphs={self._num_subgraphs}, "
            f"max_cache_len={self._max_cache_len}, "
            f"rotary_dim={self._rotary_dim})"
        )
