"""Hybrid GPU+NPU inference for Qwen3.5.

Orchestrates 19 subgraph IRs exported by export_hybrid.py:
  - 6 GDN decode blocks on GPU (contain Loop nodes, need FP32)
  - 6 GDN prefill blocks on GPU (chunkwise parallel, no Loop)
  - 6 Attention blocks on NPU (decode) + GPU (prefill)
  - 1 Head block on GPU

GPU attention prefill: during prefill, attention runs on GPU (one call
per layer, full prompt, dynamic shapes) for speed. During decode,
attention switches to NPU (S=1 static shape).

Run (root venv):
  powershell.exe -Command 'cd C:\\Apps\\translatorle; uv run python -m qwen35.inference_hybrid --prompt "Hello" --device HYBRID'
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Union

import numpy as np
import openvino as ov
from openvino._offline_transformations import apply_make_stateful_transformation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main hybrid model class
# ---------------------------------------------------------------------------

class Qwen35HybridModel:
    """Hybrid GPU+NPU inference for Qwen3.5 using 19 subgraph IRs."""

    def __init__(
        self,
        model_dir: Union[str, Path],
        gdn_device: str = "GPU",
        attn_device: str = "NPU",
        head_device: str = "GPU",
        attn_past_seq: int = 256,
        attn_stateful: bool = True,
        prefill_chunk_size: int = 16,
        gdn_prefill_device: str | None = None,
        attn_gpu_prefill: bool = False,
    ):
        """Load and compile all subgraphs.

        Args:
            model_dir: Directory containing the hybrid IR files.
            gdn_device: Device for GDN decode blocks (default: GPU).
            attn_device: Device for attention blocks (default: NPU).
            head_device: Device for the head block (default: GPU).
            attn_past_seq: Static past_seq length for NPU attention blocks.
                NPU requires static shapes; KV caches are padded to this size.
            gdn_prefill_device: Device for GDN prefill blocks (default: same
                as gdn_device). When "NPU", compiles static-shape models for
                multiple chunk sizes (max S=16) to limit Neumann series
                precision loss.
            attn_gpu_prefill: When True and attn_device="NPU", also compile
                attention blocks on GPU for prefill (dynamic shapes, one call
                per layer). Decode still uses NPU. Disable with
                --no-attn-gpu-prefill for benchmarking.
        """
        model_dir = Path(model_dir)
        core = ov.Core()

        # Enable model caching — NPU compilation is ~10s/block (60s total).
        # After first run, cached blobs make subsequent starts <1s.
        cache_dir = str(model_dir / "cache")
        core.set_property({"CACHE_DIR": cache_dir})

        self._attn_past_seq = attn_past_seq
        self._gdn_device = gdn_device
        self._attn_device = attn_device
        self._gdn_prefill_device = gdn_prefill_device or gdn_device

        # Load config to get architecture params
        import json
        with open(model_dir / "config.json") as f:
            cfg = json.load(f)
        text_cfg = cfg.get("text_config", cfg)

        self._hidden_size = text_cfg["hidden_size"]
        self._num_kv_heads = text_cfg["num_key_value_heads"]
        self._head_dim = text_cfg.get("head_dim", self._hidden_size // text_cfg["num_attention_heads"])
        self._num_v_heads = text_cfg["linear_num_value_heads"]
        self._k_head_dim = text_cfg["linear_key_head_dim"]
        self._v_head_dim = text_cfg["linear_value_head_dim"]
        self._conv_dim = (
            text_cfg["linear_num_key_heads"] * text_cfg["linear_key_head_dim"] * 2
            + text_cfg["linear_num_value_heads"] * text_cfg["linear_value_head_dim"]
        )
        self._conv_kernel = text_cfg["linear_conv_kernel_dim"]

        # Detect number of blocks
        num_blocks = 0
        while (model_dir / f"gdn_block_{num_blocks}.xml").exists():
            num_blocks += 1
        self._num_blocks = num_blocks
        logger.info("Found %d GDN+Attn block pairs", num_blocks)

        # --- Compile GDN blocks (GPU, stateful — state persists in GPU memory) ---
        logger.info("Compiling %d GDN blocks on %s (stateful) ...", num_blocks, gdn_device)
        self._gdn_models: List[ov.CompiledModel] = []
        self._gdn_requests: List = []
        t0 = time.time()

        # Map state inputs to outputs for stateful transformation
        state_in_names = ["in_conv0", "in_rec0", "in_conv1", "in_rec1", "in_conv2", "in_rec2"]
        state_out_names = ["out_conv0", "out_rec0", "out_conv1", "out_rec1", "out_conv2", "out_rec2"]
        state_map = dict(zip(state_in_names, state_out_names))

        for i in range(num_blocks):
            ir = core.read_model(str(model_dir / f"gdn_block_{i}.xml"))
            # Convert explicit state I/O to OpenVINO stateful (ReadValue/Assign).
            # After this, state lives on-device; only hidden+mask are passed each call.
            apply_make_stateful_transformation(ir, state_map)
            if gdn_device in ("GPU", "NPU"):
                ir = self._add_f32_output_conversion(ir)
            if i == 0:
                logger.info("  GDN block inputs (stateful): %s",
                            [(inp.get_any_name(), str(inp.partial_shape), str(inp.element_type)) for inp in ir.inputs])
                logger.info("  GDN block sinks: %d", len(ir.get_sinks()))
            compiled = core.compile_model(ir, gdn_device)
            self._gdn_models.append(compiled)
            self._gdn_requests.append(compiled.create_infer_request())
        logger.info("  GDN compilation: %.1fs", time.time() - t0)

        # Initialize GDN stateful variables with correct shapes
        # (dynamic dims default to shape=0 after stateful transform)
        self._init_gdn_states()

        # --- Compile chunkwise GDN prefill blocks (explicit I/O, no Loop) ---
        gdn_prefill_dev = self._gdn_prefill_device
        self._gdn_prefill_requests: List = []
        self._gdn_prefill_npu_requests: dict[int, list] = {}  # seq_len -> [req per block]
        has_prefill_gdn = (model_dir / "gdn_prefill_block_0.xml").exists()
        if has_prefill_gdn and gdn_prefill_dev == "NPU":
            # NPU: compile static shapes for descending powers of 2 (max S=16).
            # NPU precision degrades above S=16 due to Neumann series error accumulation.
            gdn_chunk_sizes = []
            s = min(prefill_chunk_size, 16)
            while s >= 1:
                gdn_chunk_sizes.append(s)
                s //= 2
            logger.info("Compiling %d chunkwise GDN prefill blocks on NPU for S=%s ...",
                        num_blocks, gdn_chunk_sizes)
            t0 = time.time()
            for cs in gdn_chunk_sizes:
                requests = []
                for i in range(num_blocks):
                    ir = core.read_model(str(model_dir / f"gdn_prefill_block_{i}.xml"))
                    ir = self._reshape_gdn_prefill_static(ir, cs)
                    ir = self._add_f32_output_conversion(ir)
                    if i == 0 and cs == gdn_chunk_sizes[0]:
                        logger.info("  GDN prefill block inputs (S=%d): %s", cs,
                                    [(inp.get_any_name(), str(inp.partial_shape),
                                      str(inp.element_type)) for inp in ir.inputs])
                    npu_config = {"NPU_COMPILER_TYPE": "PREFER_PLUGIN"}
                    compiled = core.compile_model(ir, "NPU", npu_config)
                    requests.append(compiled.create_infer_request())
                self._gdn_prefill_npu_requests[cs] = requests
            logger.info("  GDN prefill NPU compilation (%d sizes): %.1fs",
                        len(gdn_chunk_sizes), time.time() - t0)
        elif has_prefill_gdn:
            # GPU/CPU: compile dynamic shape (existing behavior)
            logger.info("Compiling %d chunkwise GDN prefill blocks on %s ...",
                        num_blocks, gdn_prefill_dev)
            t0 = time.time()
            for i in range(num_blocks):
                ir = core.read_model(str(model_dir / f"gdn_prefill_block_{i}.xml"))
                if gdn_prefill_dev in ("GPU", "NPU"):
                    ir = self._add_f32_output_conversion(ir)
                if i == 0:
                    logger.info("  GDN prefill block inputs: %s",
                                [(inp.get_any_name(), str(inp.partial_shape),
                                  str(inp.element_type)) for inp in ir.inputs])
                # Force FP32 inference: the Neumann series (7 matrix squarings)
                # accumulates catastrophic precision loss in FP16.
                config = {}
                if gdn_prefill_dev == "GPU":
                    config["INFERENCE_PRECISION_HINT"] = "f32"
                compiled = core.compile_model(ir, gdn_prefill_dev, config)
                self._gdn_prefill_requests.append(compiled.create_infer_request())
            logger.info("  GDN prefill compilation: %.1fs", time.time() - t0)
        else:
            logger.info("No chunkwise GDN prefill blocks found, using Loop-based prefill")

        # --- Compile Attention blocks ---
        # Fixed-size KV cache (ScatterUpdate) enables stateful on ALL devices.
        # NPU: try stateful first; if it fails at runtime, can fall back to explicit.
        self._attn_stateful = attn_stateful
        mode_str = "stateful" if self._attn_stateful else "explicit I/O"
        logger.info("Compiling %d Attn blocks on %s (%s, fixed KV=%d) ...",
                     num_blocks, attn_device, mode_str, attn_past_seq)
        self._attn_models: List[ov.CompiledModel] = []
        self._attn_requests: List = []
        t0 = time.time()

        attn_state_map = {"in_key_cache": "out_key_cache", "in_value_cache": "out_value_cache"}

        for i in range(num_blocks):
            ir = core.read_model(str(model_dir / f"attn_block_{i}.xml"))
            if attn_device == "NPU":
                ir = self._reshape_attn_static(ir, attn_past_seq)
            if self._attn_stateful:
                apply_make_stateful_transformation(ir, attn_state_map)
            if attn_device in ("GPU", "NPU"):
                ir = self._add_f32_output_conversion(ir)
            if i == 0:
                logger.info("  Attn block inputs: %s",
                            [(inp.get_any_name(), str(inp.partial_shape), str(inp.element_type)) for inp in ir.inputs])
                if self._attn_stateful:
                    logger.info("  Attn block sinks: %d", len(ir.get_sinks()))
            npu_config = {"NPU_COMPILER_TYPE": "PREFER_PLUGIN"} if attn_device == "NPU" else {}
            compiled = core.compile_model(ir, attn_device, npu_config)
            self._attn_models.append(compiled)
            self._attn_requests.append(compiled.create_infer_request())
        logger.info("  Attn compilation: %.1fs", time.time() - t0)

        if self._attn_stateful:
            self._init_attn_states()
        else:
            self._init_kv_caches()

        # --- Compile prefill Attention blocks (NPU, descending powers of 2) ---
        self._prefill_chunk_size = prefill_chunk_size
        self._attn_prefill_requests: dict[int, list] = {}  # chunk_size -> [req per block]
        if (prefill_chunk_size > 1 and attn_device == "NPU"
                and not self._attn_stateful):
            # Compile NPU attention for descending powers of 2: 16, 8, 4, 2
            chunk_sizes = []
            s = prefill_chunk_size
            while s >= 2:
                chunk_sizes.append(s)
                s //= 2
            logger.info("Compiling prefill Attn blocks on NPU for S=%s ...", chunk_sizes)
            t0 = time.time()
            for cs in chunk_sizes:
                requests = []
                for i in range(num_blocks):
                    ir = core.read_model(str(model_dir / f"attn_block_{i}.xml"))
                    ir = self._reshape_attn_static(ir, attn_past_seq, seq_len=cs)
                    ir = self._add_f32_output_conversion(ir)
                    npu_config = {"NPU_COMPILER_TYPE": "PREFER_PLUGIN"}
                    compiled = core.compile_model(ir, "NPU", npu_config)
                    requests.append(compiled.create_infer_request())
                self._attn_prefill_requests[cs] = requests
            logger.info("  Prefill Attn compilation (%d sizes): %.1fs", len(chunk_sizes), time.time() - t0)

        # --- Compile GPU Attention blocks for HYBRID prefill ---
        # When NPU handles decode attention, also compile on GPU for prefill.
        # GPU supports dynamic shapes → one infer call for full prompt.
        self._attn_gpu_prefill_requests: List = []
        if (attn_device == "NPU" and not self._attn_stateful
                and attn_gpu_prefill):
            logger.info("Compiling %d Attn GPU prefill blocks (dynamic shape) ...", num_blocks)
            t0 = time.time()
            for i in range(num_blocks):
                ir = core.read_model(str(model_dir / f"attn_block_{i}.xml"))
                ir = self._add_f32_output_conversion(ir)
                compiled = core.compile_model(ir, "GPU")
                self._attn_gpu_prefill_requests.append(compiled.create_infer_request())
            logger.info("  Attn GPU prefill compilation: %.1fs", time.time() - t0)

        # --- Compile Head block (GPU) ---
        logger.info("Compiling Head on %s ...", head_device)
        t0 = time.time()
        head_ir = core.read_model(str(model_dir / "head.xml"))
        if head_device in ("GPU", "NPU"):
            head_ir = self._add_f32_output_conversion(head_ir)
        self._head_model = core.compile_model(head_ir, head_device)
        self._head_request = self._head_model.create_infer_request()
        logger.info("  Head compilation: %.1fs", time.time() - t0)

        # --- Embedding table ---
        # Prefer INT8 quantized embeddings (2x smaller) with FP16 fallback
        embed_int8_path = model_dir / "embed_tokens_int8.npy"
        embed_scales_path = model_dir / "embed_tokens_scales.npy"
        if embed_int8_path.exists() and embed_scales_path.exists():
            self._embed_int8 = np.load(str(embed_int8_path))        # [vocab, dim] int8
            self._embed_scales = np.load(str(embed_scales_path))    # [vocab] float16
            self._embed_table = None
            logger.info(
                "Loaded INT8 embed_tokens: shape=%s + scales shape=%s",
                self._embed_int8.shape, self._embed_scales.shape,
            )
        else:
            embed_path = model_dir / "embed_tokens.npy"
            self._embed_table = np.load(str(embed_path)).astype(np.float32)
            self._embed_int8 = None
            self._embed_scales = None
            logger.info("Loaded embed_tokens: shape=%s dtype=%s", self._embed_table.shape, self._embed_table.dtype)

        # --- Tokenizer ---
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

        self._past_length = 0

        # Profiling instrumentation (disable for clean benchmarks)
        self._profiling = False
        self._reset_profile_stats()

    def _embed_lookup(self, token_ids: np.ndarray) -> np.ndarray:
        """Look up token embeddings, dequantizing INT8 if needed.

        Args:
            token_ids: Shape [B, seq_len] int64.

        Returns:
            embeddings: Shape [B, seq_len, hidden] float32.
        """
        if self._embed_int8 is not None:
            # INT8 per-row dequantization: int8[token_ids] * scales[token_ids]
            raw = self._embed_int8[token_ids].astype(np.float32)          # [B, S, H]
            scales = self._embed_scales[token_ids].astype(np.float32)     # [B, S]
            return raw * scales[..., np.newaxis]                          # broadcast [B, S, 1]
        return self._embed_table[token_ids]

    def _reset_profile_stats(self):
        """Reset accumulated profiling counters."""
        self._profile_stats = {
            'embed': [],
            'numpy_prep': [],
            'gdn_blocks': [[] for _ in range(self._num_blocks)],
            'attn_blocks': [[] for _ in range(self._num_blocks)],
            'head': [],
            'total': [],
        }

    def _print_profile_summary(self):
        """Print per-token profiling summary table."""
        stats = self._profile_stats
        n = len(stats['total'])
        if n == 0:
            return

        def avg_ms(lst):
            return sum(lst) / len(lst) * 1000 if lst else 0.0

        embed_avg = avg_ms(stats['embed'])
        numpy_avg = avg_ms(stats['numpy_prep'])
        head_avg = avg_ms(stats['head'])
        total_avg = avg_ms(stats['total'])

        gdn_avgs = [avg_ms(stats['gdn_blocks'][i]) for i in range(self._num_blocks)]
        attn_avgs = [avg_ms(stats['attn_blocks'][i]) for i in range(self._num_blocks)]
        gdn_total_avg = sum(gdn_avgs)
        attn_total_avg = sum(attn_avgs)

        overhead = total_avg - gdn_total_avg - attn_total_avg - head_avg - embed_avg

        logger.info("--- Profile (%d forward calls) ---", n)
        logger.info("  %-22s %8.3f ms  (%5.1f%%)", "Embed lookup", embed_avg, embed_avg / total_avg * 100 if total_avg else 0)
        logger.info("  %-22s %8.3f ms  (%5.1f%%)", "Numpy prep", numpy_avg, numpy_avg / total_avg * 100 if total_avg else 0)
        for i in range(self._num_blocks):
            logger.info("  %-22s %8.3f ms  (%5.1f%%)", f"GDN block {i}", gdn_avgs[i], gdn_avgs[i] / total_avg * 100 if total_avg else 0)
            logger.info("  %-22s %8.3f ms  (%5.1f%%)", f"Attn block {i}", attn_avgs[i], attn_avgs[i] / total_avg * 100 if total_avg else 0)
        logger.info("  %-22s %8.3f ms  (%5.1f%%)", "Head", head_avg, head_avg / total_avg * 100 if total_avg else 0)
        logger.info("  %-22s %8.3f ms", "---", 0)
        logger.info("  %-22s %8.3f ms  (%5.1f%%)", "GDN total", gdn_total_avg, gdn_total_avg / total_avg * 100 if total_avg else 0)
        logger.info("  %-22s %8.3f ms  (%5.1f%%)", "Attn total", attn_total_avg, attn_total_avg / total_avg * 100 if total_avg else 0)
        logger.info("  %-22s %8.3f ms  (%5.1f%%)", "OV inference total", gdn_total_avg + attn_total_avg + head_avg, (gdn_total_avg + attn_total_avg + head_avg) / total_avg * 100 if total_avg else 0)
        logger.info("  %-22s %8.3f ms  (%5.1f%%)", "Python overhead", overhead, overhead / total_avg * 100 if total_avg else 0)
        logger.info("  %-22s %8.3f ms", "TOTAL per token", total_avg)

    def _init_gdn_states(self):
        """Initialize GDN stateful variables with correct shapes.

        After apply_make_stateful_transformation, dynamic dims default to
        shape=0. Must manually set each state tensor to the correct shape.
        Cannot use s.reset() — it restores shape=0.
        """
        conv_shape = (1, self._conv_dim, self._conv_kernel)
        rec_shape = (1, self._num_v_heads, self._k_head_dim, self._v_head_dim)
        for req in self._gdn_requests:
            for s in req.query_state():
                if "conv" in s.name:
                    s.state = ov.Tensor(np.zeros(conv_shape, dtype=np.float32))
                elif "rec" in s.name:
                    s.state = ov.Tensor(np.zeros(rec_shape, dtype=np.float32))

    def _init_gdn_prefill_states(self):
        """Initialize explicit state buffers for chunkwise GDN prefill."""
        conv_shape = (1, self._conv_dim, self._conv_kernel)
        rec_shape = (1, self._num_v_heads, self._k_head_dim, self._v_head_dim)
        self._gdn_prefill_conv_states = [
            [np.zeros(conv_shape, dtype=np.float32) for _ in range(3)]
            for _ in range(self._num_blocks)
        ]
        self._gdn_prefill_rec_states = [
            [np.zeros(rec_shape, dtype=np.float32) for _ in range(3)]
            for _ in range(self._num_blocks)
        ]

    def _init_attn_states(self):
        """Initialize attention KV cache stateful variables.

        Fixed-size KV cache: shape is always [B, H, MAX_CACHE_LEN, D].
        All positions start as zeros; the attention mask ensures only
        written positions are attended to.
        """
        kv_shape = (1, self._num_kv_heads, self._attn_past_seq, self._head_dim)
        for req in self._attn_requests:
            for s in req.query_state():
                s.state = ov.Tensor(np.zeros(kv_shape, dtype=np.float32))

    @staticmethod
    def _add_f32_output_conversion(ir):
        """Add FP16->FP32 output conversion (GPU/NPU don't auto-promote)."""
        from openvino.preprocess import PrePostProcessor
        ppp = PrePostProcessor(ir)
        for i in range(len(ir.outputs)):
            ppp.output(i).tensor().set_element_type(ov.Type.f32)
        return ppp.build()

    def _reshape_attn_static(self, ir, past_seq: int, seq_len: int = 1):
        """Reshape attention IR to fully static shapes for NPU compilation.

        Uses set_partial_shape() + validate_nodes_and_infer_types() instead of
        ir.reshape() to avoid breaking Broadcast nodes when PARO RotatedLinear
        adds reshape/permute/bmm ops that confuse OpenVINO's shape propagation.
        """
        B, S = 1, seq_len
        # Index-based: 0=hidden, 1=position_ids, 2=key_cache, 3=value_cache,
        #              4=cache_position, 5=attention_mask
        shape_map = {
            0: [B, S, self._hidden_size],                              # hidden
            1: [3, B, S],                                              # position_ids
            2: [B, self._num_kv_heads, past_seq, self._head_dim],      # key_cache
            3: [B, self._num_kv_heads, past_seq, self._head_dim],      # value_cache
            4: [S],                                                    # cache_position
            5: [B, 1, S, past_seq],                                    # attention_mask
        }
        for i, shape in shape_map.items():
            ir.inputs[i].get_node().set_partial_shape(ov.PartialShape(shape))
        ir.validate_nodes_and_infer_types()
        return ir

    def _reshape_gdn_prefill_static(self, ir, seq_len: int):
        """Reshape GDN prefill IR to static shapes for NPU compilation.

        Uses set_partial_shape() + validate_nodes_and_infer_types() instead of
        ir.reshape() to avoid breaking GroupConvolution nodes in the GDN graph.
        """
        shape_map = {
            0: [1, seq_len, self._hidden_size],                                  # in_hidden
            1: [1, seq_len],                                                     # in_mask
            2: [1, self._conv_dim, self._conv_kernel],                           # in_conv0
            3: [1, self._num_v_heads, self._k_head_dim, self._v_head_dim],       # in_rec0
            4: [1, self._conv_dim, self._conv_kernel],                           # in_conv1
            5: [1, self._num_v_heads, self._k_head_dim, self._v_head_dim],       # in_rec1
            6: [1, self._conv_dim, self._conv_kernel],                           # in_conv2
            7: [1, self._num_v_heads, self._k_head_dim, self._v_head_dim],       # in_rec2
        }
        for i, shape in shape_map.items():
            ir.inputs[i].get_node().set_partial_shape(ov.PartialShape(shape))
        ir.validate_nodes_and_infer_types()
        return ir

    # -----------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------

    def _run_gdn_block(self, block_idx: int, hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Run one GDN block (3 layers) on GPU.

        State (conv/recurrent) persists in GPU memory via OpenVINO stateful
        variables — only hidden + mask are transferred each call.
        """
        req = self._gdn_requests[block_idx]
        req.set_input_tensor(0, ov.Tensor(np.ascontiguousarray(hidden)))
        req.set_input_tensor(1, ov.Tensor(np.ascontiguousarray(attention_mask)))
        req.infer()
        return req.get_output_tensor(0).data.copy()

    def _run_gdn_prefill_block(self, block_idx: int, hidden: np.ndarray,
                                attention_mask: np.ndarray) -> np.ndarray:
        """Run one chunkwise GDN prefill block (3 layers) with explicit state I/O.

        Unlike the stateful decode block, this takes/returns states explicitly.
        States are read from and written to self._gdn_prefill_{conv,rec}_states.
        """
        req = self._gdn_prefill_requests[block_idx]
        conv_states = self._gdn_prefill_conv_states[block_idx]
        rec_states = self._gdn_prefill_rec_states[block_idx]

        req.set_input_tensor(0, ov.Tensor(np.ascontiguousarray(hidden)))
        req.set_input_tensor(1, ov.Tensor(np.ascontiguousarray(attention_mask)))
        for j in range(3):
            req.set_input_tensor(2 + j * 2, ov.Tensor(np.ascontiguousarray(conv_states[j])))
            req.set_input_tensor(3 + j * 2, ov.Tensor(np.ascontiguousarray(rec_states[j])))
        req.infer()

        hidden_out = req.get_output_tensor(0).data.copy()
        for j in range(3):
            self._gdn_prefill_conv_states[block_idx][j] = req.get_output_tensor(1 + j * 2).data.copy()
            self._gdn_prefill_rec_states[block_idx][j] = req.get_output_tensor(2 + j * 2).data.copy()
        return hidden_out

    def _run_gdn_prefill_block_npu(self, block_idx: int, hidden: np.ndarray,
                                    attention_mask: np.ndarray, seq_len: int) -> np.ndarray:
        """Run one chunkwise GDN prefill block on NPU with static shapes.

        Same state management as _run_gdn_prefill_block but uses NPU
        static-shape infer requests selected by seq_len.
        """
        req = self._gdn_prefill_npu_requests[seq_len][block_idx]
        conv_states = self._gdn_prefill_conv_states[block_idx]
        rec_states = self._gdn_prefill_rec_states[block_idx]

        req.set_input_tensor(0, ov.Tensor(np.ascontiguousarray(hidden)))
        req.set_input_tensor(1, ov.Tensor(np.ascontiguousarray(attention_mask)))
        for j in range(3):
            req.set_input_tensor(2 + j * 2, ov.Tensor(np.ascontiguousarray(conv_states[j])))
            req.set_input_tensor(3 + j * 2, ov.Tensor(np.ascontiguousarray(rec_states[j])))
        req.infer()

        hidden_out = req.get_output_tensor(0).data.copy()
        for j in range(3):
            self._gdn_prefill_conv_states[block_idx][j] = req.get_output_tensor(1 + j * 2).data.copy()
            self._gdn_prefill_rec_states[block_idx][j] = req.get_output_tensor(2 + j * 2).data.copy()
        return hidden_out

    def _run_attn_block(self, block_idx: int, hidden: np.ndarray, position_ids: np.ndarray,
                        use_prefill: bool = False, gpu_prefill: bool = False) -> np.ndarray:
        """Run one Attention block (1 layer) with fixed-size KV cache."""
        if self._attn_stateful:
            return self._run_attn_block_stateful(block_idx, hidden, position_ids)
        return self._run_attn_block_explicit(block_idx, hidden, position_ids,
                                             use_prefill=use_prefill, gpu_prefill=gpu_prefill)

    def _run_attn_block_stateful(self, block_idx: int, hidden: np.ndarray, position_ids: np.ndarray) -> np.ndarray:
        """Run attention with stateful fixed-size KV cache.

        KV cache persists on device via ReadValue/Assign (constant shape
        [B, H, MAX_CACHE_LEN, D]). Only hidden, position_ids, cache_position,
        and attention_mask are transferred each call.
        """
        batch_size = hidden.shape[0]
        seq_len = hidden.shape[1]

        cache_position = np.arange(
            self._past_length, self._past_length + seq_len, dtype=np.int64)
        attention_mask = self._build_attn_mask(batch_size, seq_len)

        # After stateful transform: inputs are [hidden, position_ids, cache_position, attention_mask]
        req = self._attn_requests[block_idx]
        req.set_input_tensor(0, ov.Tensor(np.ascontiguousarray(hidden)))
        req.set_input_tensor(1, ov.Tensor(np.ascontiguousarray(position_ids)))
        req.set_input_tensor(2, ov.Tensor(np.ascontiguousarray(cache_position)))
        req.set_input_tensor(3, ov.Tensor(np.ascontiguousarray(attention_mask)))
        req.infer()

        return req.get_output_tensor(0).data.copy().reshape(batch_size, seq_len, -1)

    def _run_attn_block_explicit(self, block_idx: int, hidden: np.ndarray, position_ids: np.ndarray,
                                  use_prefill: bool = False, gpu_prefill: bool = False) -> np.ndarray:
        """Run attention with explicit fixed-size KV cache I/O.

        KV cache is transferred each step but no padding/compaction needed
        (fixed shape [B, H, MAX_CACHE_LEN, D] in and out).
        """
        batch_size = hidden.shape[0]
        seq_len = hidden.shape[1]

        cache_position = np.arange(
            self._past_length, self._past_length + seq_len, dtype=np.int64)
        attention_mask = self._build_attn_mask(batch_size, seq_len)

        # Explicit I/O: inputs are [hidden, position_ids, key_cache, value_cache, cache_position, attention_mask]
        # Select request: GPU prefill > NPU prefill > NPU decode
        if gpu_prefill and self._attn_gpu_prefill_requests:
            req = self._attn_gpu_prefill_requests[block_idx]
        elif use_prefill and seq_len in self._attn_prefill_requests:
            req = self._attn_prefill_requests[seq_len][block_idx]
        else:
            req = self._attn_requests[block_idx]
        req.set_input_tensor(0, ov.Tensor(np.ascontiguousarray(hidden)))
        req.set_input_tensor(1, ov.Tensor(np.ascontiguousarray(position_ids)))
        req.set_input_tensor(2, ov.Tensor(np.ascontiguousarray(self._kv_caches[block_idx][0])))
        req.set_input_tensor(3, ov.Tensor(np.ascontiguousarray(self._kv_caches[block_idx][1])))
        req.set_input_tensor(4, ov.Tensor(np.ascontiguousarray(cache_position)))
        req.set_input_tensor(5, ov.Tensor(np.ascontiguousarray(attention_mask)))
        req.infer()

        hidden_out = req.get_output_tensor(0).data.copy().reshape(batch_size, seq_len, -1)
        # Update stored KV caches (output has same shape as input)
        self._kv_caches[block_idx][0] = req.get_output_tensor(1).data.copy()
        self._kv_caches[block_idx][1] = req.get_output_tensor(2).data.copy()

        return hidden_out

    def _build_attn_mask(self, batch_size: int, seq_len: int) -> np.ndarray:
        """Build 4D causal attention mask for fixed-size KV cache.

        Layout of key positions in the fixed buffer [0..MAX_CACHE_LEN-1]:
          [0 .. past_length-1]                  -> previously written tokens (attend)
          [past_length .. past_length+seq_len-1] -> current token(s) being written (attend)
          [past_length+seq_len .. MAX_CACHE_LEN-1] -> empty/future (masked)

        For seq_len > 1 (chunked prefill), applies causal masking so each
        query can only attend to positions up to its own position.

        Returns:
            attention_mask: shape [B, 1, seq_len, MAX_CACHE_LEN], dtype float32.
            0.0 = attend, -65504.0 = ignore.
        """
        MASK_VALUE = np.float32(-65504.0)  # min fp16, guarantees ~0 softmax weight
        max_len = self._attn_past_seq

        # For each query q in [0..seq_len-1]:
        #   attend to positions 0..past_length+q (inclusive)
        #   mask positions past_length+q+1..MAX_CACHE_LEN-1
        col = np.arange(max_len)
        query_ends = np.arange(self._past_length + 1, self._past_length + seq_len + 1)
        attend = col[None, :] < query_ends[:, None]  # [seq_len, max_len]
        mask = np.where(attend, np.float32(0.0), MASK_VALUE)[None, None, :, :]  # [1, 1, seq_len, max_len]
        if batch_size > 1:
            mask = np.broadcast_to(mask, (batch_size, 1, seq_len, max_len)).copy()
        return mask

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """Run one forward pass through all subgraphs.

        Args:
            token_ids: Shape [1, seq_len] int64.

        Returns:
            logits: Shape [1, seq_len, vocab_size] float32.
        """
        profiling = self._profiling
        if profiling:
            t_total_start = time.perf_counter()

        batch_size, seq_len = token_ids.shape

        # Embed tokens — float32, index gives [B, seq_len, hidden]
        if profiling:
            t0 = time.perf_counter()
        embeds = self._embed_lookup(token_ids)
        if profiling:
            self._profile_stats['embed'].append(time.perf_counter() - t0)

        # Numpy prep: attention mask + position IDs
        if profiling:
            t0 = time.perf_counter()
        # Attention mask for GDN blocks
        attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)

        # Position IDs for attention blocks (mRoPE: [3, B, seq_len])
        positions = np.arange(self._past_length, self._past_length + seq_len, dtype=np.int64)
        position_ids = np.tile(positions[np.newaxis, np.newaxis, :], (3, batch_size, 1))
        if profiling:
            self._profile_stats['numpy_prep'].append(time.perf_counter() - t0)

        hidden = embeds

        # Auto-detect prefill mode: use prefill requests when seq_len has a compiled model
        use_prefill = seq_len in self._attn_prefill_requests

        for i in range(self._num_blocks):
            if profiling:
                t0 = time.perf_counter()
            hidden = self._run_gdn_block(i, hidden, attention_mask)
            if profiling:
                self._profile_stats['gdn_blocks'][i].append(time.perf_counter() - t0)

            if profiling:
                t0 = time.perf_counter()
            hidden = self._run_attn_block(i, hidden, position_ids, use_prefill=use_prefill)
            if profiling:
                self._profile_stats['attn_blocks'][i].append(time.perf_counter() - t0)

        # Head
        if profiling:
            t0 = time.perf_counter()
        self._head_request.set_input_tensor(0, ov.Tensor(np.ascontiguousarray(hidden)))
        self._head_request.infer()
        logits = self._head_request.get_output_tensor(0).data.copy()
        if profiling:
            self._profile_stats['head'].append(time.perf_counter() - t0)

        self._past_length += seq_len

        if profiling:
            self._profile_stats['total'].append(time.perf_counter() - t_total_start)

        return logits

    def prefill(self, token_ids: np.ndarray) -> np.ndarray:
        """Optimized prefill: parallel GDN (GPU) + chunked attention (NPU).

        Layer-major ordering: for each layer, run GDN on full prompt (one
        GPU infer call), then chunk attention across NPU (multiple calls).

        If chunkwise GDN prefill blocks are available (no Loop node), uses
        parallel MatMul instead of sequential Loop — much faster for S>1.
        Falls back to Loop-based GDN blocks if prefill blocks not exported.

        Args:
            token_ids: Shape [1, prompt_len] int64, full prompt.

        Returns:
            logits: Shape [1, 1, vocab_size] float32 (last token only).
        """
        batch_size, prompt_len = token_ids.shape
        chunk_size = self._prefill_chunk_size
        use_chunkwise = bool(self._gdn_prefill_requests) or bool(self._gdn_prefill_npu_requests)

        # Initialize explicit prefill states if using chunkwise
        if use_chunkwise:
            self._init_gdn_prefill_states()

        # Compute chunk boundaries for NPU attention
        chunk_bounds = []  # [(start_pos, chunk_len), ...]
        pos = 0
        while pos < prompt_len:
            remaining = prompt_len - pos
            cs = chunk_size
            while cs > remaining:
                cs //= 2
            if cs < 1:
                cs = 1
            chunk_bounds.append((pos, cs))
            pos += cs

        # Compute separate chunk bounds for GDN NPU prefill (max S=16)
        use_gdn_npu = use_chunkwise and bool(self._gdn_prefill_npu_requests)
        if use_gdn_npu:
            gdn_max_chunk = max(self._gdn_prefill_npu_requests.keys())
            gdn_chunk_bounds = []
            gdn_pos = 0
            while gdn_pos < prompt_len:
                remaining = prompt_len - gdn_pos
                cs = gdn_max_chunk
                while cs > remaining:
                    cs //= 2
                if cs < 1:
                    cs = 1
                gdn_chunk_bounds.append((gdn_pos, cs))
                gdn_pos += cs

        # Embed full prompt
        hidden = self._embed_lookup(token_ids)  # [B, prompt_len, H]

        for layer_idx in range(self._num_blocks):
            # --- GDN block ---
            if use_gdn_npu:
                # NPU: chunk GDN prefill (static shapes, max S=16)
                gdn_outputs = []
                for start, cs in gdn_chunk_bounds:
                    chunk_hidden = hidden[:, start:start + cs, :]
                    gdn_mask = np.ones((batch_size, cs), dtype=np.int64)
                    gdn_outputs.append(
                        self._run_gdn_prefill_block_npu(
                            layer_idx, chunk_hidden, gdn_mask, cs))
                hidden = np.concatenate(gdn_outputs, axis=1)
            elif use_chunkwise:
                # GPU: process full prompt at once (dynamic shape)
                gdn_mask = np.ones((batch_size, prompt_len), dtype=np.int64)
                hidden = self._run_gdn_prefill_block(layer_idx, hidden, gdn_mask)
            else:
                gdn_mask = np.ones((batch_size, prompt_len), dtype=np.int64)
                hidden = self._run_gdn_block(layer_idx, hidden, gdn_mask)

            # --- Attention block ---
            if self._attn_stateful:
                # GPU stateful: can process full prompt at once
                self._past_length = 0
                positions = np.arange(0, prompt_len, dtype=np.int64)
                position_ids = np.tile(
                    positions[np.newaxis, np.newaxis, :], (3, batch_size, 1))
                hidden = self._run_attn_block(layer_idx, hidden, position_ids)
            elif self._attn_gpu_prefill_requests:
                # GPU explicit I/O prefill: process full prompt at once (dynamic shapes)
                self._past_length = 0
                positions = np.arange(0, prompt_len, dtype=np.int64)
                position_ids = np.tile(
                    positions[np.newaxis, np.newaxis, :], (3, batch_size, 1))
                hidden = self._run_attn_block(
                    layer_idx, hidden, position_ids, gpu_prefill=True)
            else:
                # NPU explicit I/O: process in chunks
                attn_outputs = []
                for start, cs in chunk_bounds:
                    self._past_length = start
                    chunk_hidden = hidden[:, start:start + cs, :]

                    positions = np.arange(start, start + cs, dtype=np.int64)
                    position_ids = np.tile(
                        positions[np.newaxis, np.newaxis, :], (3, batch_size, 1))

                    use_prefill = cs in self._attn_prefill_requests
                    chunk_out = self._run_attn_block(
                        layer_idx, chunk_hidden, position_ids,
                        use_prefill=use_prefill)
                    attn_outputs.append(chunk_out)

                hidden = np.concatenate(attn_outputs, axis=1)

        # Set past_length for decode phase
        self._past_length = prompt_len

        # Transfer chunkwise prefill states to stateful decode GDN blocks
        if use_chunkwise:
            for blk_idx in range(self._num_blocks):
                for s in self._gdn_requests[blk_idx].query_state():
                    for j in range(3):
                        if f"conv{j}" in s.name:
                            s.state = ov.Tensor(self._gdn_prefill_conv_states[blk_idx][j])
                        elif f"rec{j}" in s.name:
                            s.state = ov.Tensor(self._gdn_prefill_rec_states[blk_idx][j])

        # Head: last token only (save ~6ms per skipped chunk)
        last_hidden = hidden[:, -1:, :]
        self._head_request.set_input_tensor(
            0, ov.Tensor(np.ascontiguousarray(last_hidden)))
        self._head_request.infer()
        logits = self._head_request.get_output_tensor(0).data.copy()

        return logits

    # -----------------------------------------------------------------
    # Generation
    # -----------------------------------------------------------------

    def reset(self):
        """Reset all states for a new generation."""
        if self._attn_stateful:
            self._init_attn_states()  # Reset attention KV cache stateful variables
        else:
            self._init_kv_caches()    # Reset explicit KV cache buffers
        self._init_gdn_states()       # Reset GDN stateful variables (can't use s.reset())
        self._past_length = 0

    def _init_kv_caches(self):
        """Initialize explicit KV cache buffers (non-stateful mode)."""
        kv_shape = (1, self._num_kv_heads, self._attn_past_seq, self._head_dim)
        self._kv_caches = [
            [np.zeros(kv_shape, dtype=np.float32), np.zeros(kv_shape, dtype=np.float32)]
            for _ in range(self._num_blocks)
        ]

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input text.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy).

        Returns:
            Generated text (excluding prompt).
        """
        self.reset()

        # Tokenize
        token_list = self._tokenizer.encode(prompt)
        input_ids = np.array([token_list], dtype=np.int64)  # [1, seq_len]

        # Prefill — layer-major (full-batch GDN + chunked attention)
        t0 = time.time()
        prompt_len = input_ids.shape[1]
        chunk_size = self._prefill_chunk_size
        logits = self.prefill(input_ids)
        prefill_time = time.time() - t0
        logger.info(
            "Prefill: %d tokens in %.1fms (%.1f tok/s, chunk=%d, layer-major)",
            prompt_len, prefill_time * 1000, prompt_len / prefill_time, chunk_size,
        )

        # Greedy decode
        next_id = int(np.argmax(logits[0, -1, :]))
        generated = [next_id]

        eos_id = self._tokenizer.eos_token_id
        # Also stop on <|im_end|> (151645) and <|endoftext|> (151643)
        stop_ids = {eos_id, 151645, 151643} if eos_id else {151645, 151643}

        t_decode_start = time.time()
        for step in range(max_new_tokens - 1):
            if next_id in stop_ids:
                break

            token_input = np.array([[next_id]], dtype=np.int64)
            logits = self.forward(token_input)
            next_id = int(np.argmax(logits[0, -1, :]))
            generated.append(next_id)

        decode_time = time.time() - t_decode_start
        num_decoded = len(generated)

        # Print profiling summary before decode stats
        if self._profiling and self._profile_stats['total']:
            self._print_profile_summary()
            self._reset_profile_stats()

        if decode_time > 0:
            logger.info(
                "Decode: %d tokens in %.1fms (%.1f tok/s)",
                num_decoded, decode_time * 1000, num_decoded / decode_time,
            )

        # Remove stop token from output
        if generated and generated[-1] in stop_ids:
            generated = generated[:-1]

        return self._tokenizer.decode(generated, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Qwen3.5 hybrid GPU+NPU inference")
    parser.add_argument("--model-dir", default="models/qwen35/Qwen3.5-0.8B-hybrid")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument(
        "--device", default="HYBRID",
        choices=["HYBRID", "GPU_ONLY", "CPU_ONLY"],
        help="HYBRID=GDN on GPU + Attn on NPU, GPU_ONLY=all GPU, CPU_ONLY=all CPU",
    )
    parser.add_argument("--attn-past-seq", type=int, default=256, help="Static KV cache size for NPU")
    parser.add_argument("--no-attn-stateful", action="store_true", help="Use explicit KV I/O instead of stateful")
    parser.add_argument("--prefill-chunk-size", type=int, default=16, help="Prefill chunk size (1=token-by-token)")
    parser.add_argument("--gdn-prefill-device", default=None, choices=["GPU", "NPU"],
                        help="Device for GDN prefill blocks (default: same as gdn_device)")
    parser.add_argument("--no-attn-gpu-prefill", action="store_true",
                        help="Disable GPU attention for prefill (use NPU chunked instead)")
    args = parser.parse_args()

    device_map = {
        "HYBRID": ("GPU", "NPU", "GPU"),
        "GPU_ONLY": ("GPU", "GPU", "GPU"),
        "CPU_ONLY": ("CPU", "CPU", "CPU"),
    }
    gdn_dev, attn_dev, head_dev = device_map[args.device]

    model = Qwen35HybridModel(
        model_dir=args.model_dir,
        gdn_device=gdn_dev,
        attn_device=attn_dev,
        head_device=head_dev,
        attn_past_seq=args.attn_past_seq,
        attn_stateful=not args.no_attn_stateful,
        prefill_chunk_size=args.prefill_chunk_size,
        gdn_prefill_device=args.gdn_prefill_device,
        attn_gpu_prefill=not args.no_attn_gpu_prefill,
    )

    gdn_prefill_dev = args.gdn_prefill_device or gdn_dev
    print(f"\nPrompt: {args.prompt}")
    print(f"Device: {args.device} (GDN={gdn_dev}, Attn={attn_dev}, Head={head_dev}, GDN-prefill={gdn_prefill_dev})")
    print("-" * 60)

    output = model.generate(args.prompt, max_new_tokens=args.max_tokens)
    print(f"\nOutput: {output}")


if __name__ == "__main__":
    main()
