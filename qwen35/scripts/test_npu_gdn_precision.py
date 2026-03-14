"""End-to-end GDN decode precision comparison: GPU (Loop-based) vs NPU (noloop).

Runs the full Qwen3.5-0.8B hybrid inference pipeline with two configurations:
  1. Baseline (GPU): GDN decode on GPU using stateful Loop-based blocks
  2. Test (NPU): GDN decode on NPU using noloop blocks with explicit I/O

Prefill uses the same GPU chunkwise prefill blocks for both paths. After
prefill, the explicit states are copied into the NPU noloop block buffers.
Decode-phase GDN is swapped via monkey-patching _run_gdn_block.

Usage (from project root on Windows):
    $env:PYTHONIOENCODING="utf-8"; cd C:\\Apps\\translatorle
    C:\\Users\\taowen\\.local\\bin\\uv.exe run python -m qwen35.scripts.test_npu_gdn_precision
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import openvino as ov

from qwen35.inference_hybrid import Qwen35HybridModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Model directory discovery
# -----------------------------------------------------------------------

MODEL_DIRS = [
    Path("models/qwen35/Qwen3.5-0.8B-hybrid-attn-int4sym-gdn-int8sym-head-int4sym"),
    Path("models/qwen35/Qwen3.5-0.8B-hybrid"),
]


def find_model_dir() -> Path:
    for d in MODEL_DIRS:
        if d.exists() and (d / "gdn_block_0.xml").exists():
            return d
    raise FileNotFoundError(
        "No hybrid model directory found. Searched:\n"
        + "\n".join(f"  {d} (exists={d.exists()})" for d in MODEL_DIRS)
    )


# -----------------------------------------------------------------------
# NPU noloop GDN block loader
# -----------------------------------------------------------------------

class NpuGdnBlocks:
    """Manages 6 GDN noloop blocks compiled on NPU with explicit state I/O."""

    def __init__(self, model_dir: Path, core: ov.Core, text_cfg: dict):
        self.hidden_size = text_cfg["hidden_size"]
        self.num_v_heads = text_cfg["linear_num_value_heads"]
        self.k_head_dim = text_cfg["linear_key_head_dim"]
        self.v_head_dim = text_cfg["linear_value_head_dim"]
        self.conv_dim = (
            text_cfg["linear_num_key_heads"] * text_cfg["linear_key_head_dim"] * 2
            + text_cfg["linear_num_value_heads"] * text_cfg["linear_value_head_dim"]
        )
        self.conv_kernel = text_cfg["linear_conv_kernel_dim"]

        # Detect number of blocks
        num_blocks = 0
        while (model_dir / f"gdn_block_{num_blocks}.xml").exists():
            num_blocks += 1
        self.num_blocks = num_blocks

        # Find noloop IR prefix: prefer gdn_s1_block (may be quantized), fall back to gdn_noloop_block
        if (model_dir / "gdn_s1_block_0.xml").exists():
            prefix = "gdn_s1_block"
        elif (model_dir / "gdn_noloop_block_0.xml").exists():
            prefix = "gdn_noloop_block"
        else:
            raise FileNotFoundError(
                f"No noloop GDN blocks found in {model_dir}. "
                "Need gdn_s1_block_*.xml or gdn_noloop_block_*.xml"
            )

        # Static shapes for S=1 decode
        static_shapes = {
            "in_hidden": [1, 1, self.hidden_size],
            "in_mask": [1, 1],
            "in_conv0": [1, self.conv_dim, self.conv_kernel],
            "in_rec0": [1, self.num_v_heads, self.k_head_dim, self.v_head_dim],
            "in_conv1": [1, self.conv_dim, self.conv_kernel],
            "in_rec1": [1, self.num_v_heads, self.k_head_dim, self.v_head_dim],
            "in_conv2": [1, self.conv_dim, self.conv_kernel],
            "in_rec2": [1, self.num_v_heads, self.k_head_dim, self.v_head_dim],
        }

        logger.info("Loading %d GDN noloop blocks (%s) on NPU...", num_blocks, prefix)
        t0 = time.time()
        self._requests = []
        for i in range(num_blocks):
            ir = core.read_model(str(model_dir / f"{prefix}_{i}.xml"))
            ir.reshape(static_shapes)
            compiled = core.compile_model(ir, "NPU", {"NPU_COMPILER_TYPE": "PREFER_PLUGIN"})
            self._requests.append(compiled.create_infer_request())
        logger.info("  NPU GDN compilation: %.1fs", time.time() - t0)

        # Initialize explicit state buffers
        self.conv_states: list[list[np.ndarray]] = []
        self.rec_states: list[list[np.ndarray]] = []
        self.reset_states()

    def reset_states(self):
        """Reset all conv/recurrent states to zeros."""
        conv_shape = (1, self.conv_dim, self.conv_kernel)
        rec_shape = (1, self.num_v_heads, self.k_head_dim, self.v_head_dim)
        self.conv_states = [
            [np.zeros(conv_shape, dtype=np.float32) for _ in range(3)]
            for _ in range(self.num_blocks)
        ]
        self.rec_states = [
            [np.zeros(rec_shape, dtype=np.float32) for _ in range(3)]
            for _ in range(self.num_blocks)
        ]

    def run_block(self, block_idx: int, hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Run one GDN noloop block on NPU with explicit state I/O.

        Signature matches Qwen35HybridModel._run_gdn_block so it can be
        used as a monkey-patch replacement.
        """
        req = self._requests[block_idx]
        req.set_input_tensor(0, ov.Tensor(np.ascontiguousarray(hidden)))
        req.set_input_tensor(1, ov.Tensor(np.ascontiguousarray(attention_mask)))
        for j in range(3):
            req.set_input_tensor(2 + j * 2, ov.Tensor(np.ascontiguousarray(self.conv_states[block_idx][j])))
            req.set_input_tensor(3 + j * 2, ov.Tensor(np.ascontiguousarray(self.rec_states[block_idx][j])))
        req.infer()

        hidden_out = req.get_output_tensor(0).data.copy()
        for j in range(3):
            self.conv_states[block_idx][j] = req.get_output_tensor(1 + j * 2).data.copy()
            self.rec_states[block_idx][j] = req.get_output_tensor(2 + j * 2).data.copy()
        return hidden_out


# -----------------------------------------------------------------------
# Main comparison
# -----------------------------------------------------------------------

def main():
    model_dir = find_model_dir()
    logger.info("Model dir: %s", model_dir)

    # Load config
    with open(model_dir / "config.json") as f:
        cfg = json.load(f)
    text_cfg = cfg.get("text_config", cfg)

    # Set up OpenVINO core with caching for NPU
    core = ov.Core()
    core.set_property({"CACHE_DIR": str(model_dir / "cache_npu_test")})

    # Check NPU availability
    if "NPU" not in core.available_devices:
        logger.error("NPU not available. Devices: %s", core.available_devices)
        return

    # --- Load NPU noloop GDN blocks ---
    npu_gdn = NpuGdnBlocks(model_dir, core, text_cfg)

    # --- Load GPU baseline model ---
    logger.info("Loading GPU baseline model...")
    gpu_model = Qwen35HybridModel(
        model_dir=model_dir,
        gdn_device="GPU",
        attn_device="GPU",
        head_device="GPU",
        attn_stateful=True,
        attn_past_seq=256,
        prefill_chunk_size=16,
    )

    # Stop token IDs (same as generate() in inference_hybrid.py)
    eos_id = gpu_model._tokenizer.eos_token_id
    stop_ids = {eos_id, 151645, 151643} if eos_id else {151645, 151643}

    # Test prompts
    prompts = [
        "The capital of France is",
        "Write a Python function to compute fibonacci",
        "Explain quantum mechanics in simple terms",
    ]

    MAX_TOKENS = 30

    all_match = True

    for prompt in prompts:
        print(f"\n{'=' * 60}")
        print(f"Prompt: {prompt}")
        print(f"{'=' * 60}")

        # Tokenize
        token_list = gpu_model._tokenizer.encode(prompt)
        token_ids = np.array([token_list], dtype=np.int64)
        print(f"Tokens: {token_ids.shape[1]}")

        # ----- GPU baseline -----
        gpu_model.reset()
        gpu_logits = gpu_model.prefill(token_ids)
        gpu_tokens = [int(np.argmax(gpu_logits[0, -1, :]))]

        for _ in range(MAX_TOKENS - 1):
            if gpu_tokens[-1] in stop_ids:
                break
            logits = gpu_model.forward(np.array([[gpu_tokens[-1]]], dtype=np.int64))
            gpu_tokens.append(int(np.argmax(logits[0, -1, :])))

        gpu_text = gpu_model._tokenizer.decode(gpu_tokens, skip_special_tokens=True)
        print(f"\nGPU output ({len(gpu_tokens)} tokens): {gpu_text}")

        # ----- NPU GDN test -----
        # Reset NPU explicit states
        npu_gdn.reset_states()

        # Reset the model for a fresh run (resets GPU stateful GDN + attention)
        gpu_model.reset()

        # Prefill using the normal GPU path (chunkwise prefill blocks)
        npu_logits = gpu_model.prefill(token_ids)

        # After prefill, _gdn_prefill_conv_states / _gdn_prefill_rec_states
        # hold the final GDN states. Copy them into NPU explicit buffers.
        for i in range(npu_gdn.num_blocks):
            for j in range(3):
                npu_gdn.conv_states[i][j] = gpu_model._gdn_prefill_conv_states[i][j].copy()
                npu_gdn.rec_states[i][j] = gpu_model._gdn_prefill_rec_states[i][j].copy()

        # Monkey-patch _run_gdn_block to use NPU noloop blocks for decode
        original_run_gdn = gpu_model._run_gdn_block
        gpu_model._run_gdn_block = npu_gdn.run_block

        npu_tokens = [int(np.argmax(npu_logits[0, -1, :]))]

        for _ in range(MAX_TOKENS - 1):
            if npu_tokens[-1] in stop_ids:
                break
            logits = gpu_model.forward(np.array([[npu_tokens[-1]]], dtype=np.int64))
            npu_tokens.append(int(np.argmax(logits[0, -1, :])))

        # Restore original method
        gpu_model._run_gdn_block = original_run_gdn

        npu_text = gpu_model._tokenizer.decode(npu_tokens, skip_special_tokens=True)
        print(f"NPU output ({len(npu_tokens)} tokens): {npu_text}")

        # ----- Compare -----
        match = gpu_tokens == npu_tokens
        if match:
            print("[MATCH] Identical tokens")
        else:
            all_match = False
            # Find first divergence point
            for idx in range(min(len(gpu_tokens), len(npu_tokens))):
                if gpu_tokens[idx] != npu_tokens[idx]:
                    print(f"[DIVERGE] First mismatch at decode token {idx}: "
                          f"GPU={gpu_tokens[idx]} NPU={npu_tokens[idx]}")
                    break
            if len(gpu_tokens) != len(npu_tokens):
                print(f"  Length differs: GPU={len(gpu_tokens)}, NPU={len(npu_tokens)}")
            print(f"  GPU tokens: {gpu_tokens}")
            print(f"  NPU tokens: {npu_tokens}")

    # ----- Summary -----
    print(f"\n{'=' * 60}")
    if all_match:
        print("RESULT: All prompts produced identical tokens on GPU and NPU GDN.")
    else:
        print("RESULT: Some prompts diverged between GPU and NPU GDN decode.")
        print("This indicates FP16 precision loss in NPU GDN recursive states.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
