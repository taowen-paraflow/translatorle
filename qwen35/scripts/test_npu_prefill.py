"""Test GDN prefill block: GPU (FP32) vs NPU precision comparison.

Strategy:
1. GPU: compile with dynamic shapes + FP32 hint (same as inference_hybrid.py)
2. NPU: try compile with static shapes via set_partial_shape (avoid ir.reshape
   which breaks GroupConvolution validation)
3. Compare outputs

Run (root venv):
  powershell.exe -Command 'cd C:\\Apps\\translatorle; C:\\Users\\taowen\\.local\\bin\\uv.exe run python -m qwen35.scripts.test_npu_prefill'
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import openvino as ov

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path("models/qwen35/Qwen3.5-0.8B-hybrid")


def load_config(model_dir: Path) -> dict:
    with open(model_dir / "config.json") as f:
        cfg = json.load(f)
    text_cfg = cfg.get("text_config", cfg)
    return {
        "hidden_size": text_cfg["hidden_size"],
        "conv_dim": (
            text_cfg["linear_num_key_heads"] * text_cfg["linear_key_head_dim"] * 2
            + text_cfg["linear_num_value_heads"] * text_cfg["linear_value_head_dim"]
        ),
        "conv_kernel": text_cfg["linear_conv_kernel_dim"],
        "num_v_heads": text_cfg["linear_num_value_heads"],
        "k_head_dim": text_cfg["linear_key_head_dim"],
        "v_head_dim": text_cfg["linear_value_head_dim"],
    }


def make_static_via_partial_shape(ir, seq_len: int, cfg: dict):
    """Set static shapes using set_partial_shape + validate (not ir.reshape).

    ir.reshape() breaks GroupConvolution validation. Using set_partial_shape
    on each input individually avoids this issue.
    """
    B = 1
    S = seq_len
    shape_map = {
        0: [B, S, cfg["hidden_size"]],
        1: [B, S],
        2: [B, cfg["conv_dim"], cfg["conv_kernel"]],
        3: [B, cfg["num_v_heads"], cfg["k_head_dim"], cfg["v_head_dim"]],
        4: [B, cfg["conv_dim"], cfg["conv_kernel"]],
        5: [B, cfg["num_v_heads"], cfg["k_head_dim"], cfg["v_head_dim"]],
        6: [B, cfg["conv_dim"], cfg["conv_kernel"]],
        7: [B, cfg["num_v_heads"], cfg["k_head_dim"], cfg["v_head_dim"]],
    }
    for i, shape in shape_map.items():
        ir.inputs[i].get_node().set_partial_shape(ov.PartialShape(shape))
    ir.validate_nodes_and_infer_types()


def add_f32_output_conversion(ir):
    from openvino.preprocess import PrePostProcessor
    ppp = PrePostProcessor(ir)
    for i in range(len(ir.outputs)):
        ppp.output(i).tensor().set_element_type(ov.Type.f32)
    return ppp.build()


def generate_inputs(seq_len: int, cfg: dict, seed: int = 42,
                    random_states: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    B, S = 1, seq_len
    conv_shape = (B, cfg["conv_dim"], cfg["conv_kernel"])
    rec_shape = (B, cfg["num_v_heads"], cfg["k_head_dim"], cfg["v_head_dim"])

    inputs = {
        0: rng.standard_normal((B, S, cfg["hidden_size"])).astype(np.float32),
        1: np.ones((B, S), dtype=np.int64),
    }
    for j in range(3):
        if random_states:
            inputs[2 + j * 2] = rng.standard_normal(conv_shape).astype(np.float32) * 0.01
            inputs[3 + j * 2] = rng.standard_normal(rec_shape).astype(np.float32) * 0.01
        else:
            inputs[2 + j * 2] = np.zeros(conv_shape, dtype=np.float32)
            inputs[3 + j * 2] = np.zeros(rec_shape, dtype=np.float32)
    return inputs


def compare_outputs(ref_outputs: list, test_outputs: list) -> list[dict]:
    OUTPUT_NAMES = ["out_hidden", "out_conv0", "out_rec0",
                    "out_conv1", "out_rec1", "out_conv2", "out_rec2"]
    results = []
    for i, (g, n) in enumerate(zip(ref_outputs, test_outputs)):
        diff = np.abs(g - n)
        g_flat, n_flat = g.flatten(), n.flatten()
        dot = np.dot(g_flat, n_flat)
        norm_g, norm_n = np.linalg.norm(g_flat), np.linalg.norm(n_flat)
        results.append({
            "name": OUTPUT_NAMES[i] if i < len(OUTPUT_NAMES) else f"output_{i}",
            "shape": g.shape,
            "max_abs_err": float(diff.max()),
            "mean_abs_err": float(diff.mean()),
            "max_rel_err": float((diff / (np.abs(g) + 1e-7)).max()),
            "ref_range": (float(g.min()), float(g.max())),
            "cosine_sim": float(dot / (norm_g * norm_n + 1e-12)),
        })
    return results


def run_on_device(core, model_dir, seq_len, cfg, device, config, inputs,
                  label, static=False):
    """Load IR, optionally set static shapes, compile, run, return outputs."""
    ir = core.read_model(str(model_dir / "gdn_prefill_block_0.xml"))

    if static:
        logger.info("  [%s] Setting static shapes (S=%d) ...", label, seq_len)
        make_static_via_partial_shape(ir, seq_len, cfg)

    ir = add_f32_output_conversion(ir)

    logger.info("  [%s] Compiling on %s (S=%d, static=%s) ...",
                label, device, seq_len, static)
    t0 = time.time()
    try:
        compiled = core.compile_model(ir, device, config)
    except Exception as e:
        logger.error("  [%s] Compilation FAILED: %s", label, e)
        return None, 0.0, 0.0
    compile_time = time.time() - t0
    logger.info("  [%s] Compiled in %.2fs", label, compile_time)

    req = compiled.create_infer_request()
    for idx, arr in inputs.items():
        req.set_input_tensor(idx, ov.Tensor(np.ascontiguousarray(arr)))

    try:
        req.infer()  # warm-up
    except Exception as e:
        logger.error("  [%s] Inference FAILED: %s", label, e)
        return None, compile_time, 0.0

    t0 = time.time()
    req.infer()
    infer_time = time.time() - t0

    outputs = [req.get_output_tensor(i).data.copy()
               for i in range(len(compiled.outputs))]
    logger.info("  [%s] Inference: %.3fms", label, infer_time * 1000)
    return outputs, compile_time, infer_time


def print_ir_info(core, model_dir):
    ir = core.read_model(str(model_dir / "gdn_prefill_block_0.xml"))
    op_counts = {}
    for op in ir.get_ops():
        t = op.get_type_name()
        op_counts[t] = op_counts.get(t, 0) + 1

    print(f"\nGDN Prefill Block 0 IR: {len(list(ir.get_ops()))} ops")
    for name in ["MatMul", "GroupConvolution", "CumSum", "Exp", "Loop", "TensorIterator"]:
        if name in op_counts:
            print(f"  {name}: {op_counts[name]}")
    print(f"  Has Loop: {'Loop' in op_counts or 'TensorIterator' in op_counts}")

    print("\n  Inputs:")
    for i, inp in enumerate(ir.inputs):
        print(f"    [{i}] {inp.get_any_name()}: {inp.partial_shape} ({inp.element_type})")
    print("  Outputs:")
    for i, out in enumerate(ir.outputs):
        print(f"    [{i}] {out.partial_shape} ({out.element_type})")


def main():
    core = ov.Core()
    model_dir = MODEL_DIR

    if not (model_dir / "gdn_prefill_block_0.xml").exists():
        logger.error("IR not found. Run export first.")
        sys.exit(1)

    logger.info("Available devices: %s", core.available_devices)
    cfg = load_config(model_dir)
    logger.info("Config: %s", cfg)
    print_ir_info(core, model_dir)

    core.set_property({"CACHE_DIR": str(model_dir / "cache_test_npu_prefill")})

    # --- Phase 1: GPU reference (dynamic shapes, FP32) ---
    print("\n" + "=" * 80)
    print("PHASE 1: GPU FP32 reference (dynamic shapes)")
    print("=" * 80)

    test_configs = [
        (4, False), (16, False), (32, False), (64, False),
        (16, True), (64, True),
    ]

    gpu_results = {}
    for seq_len, random_states in test_configs:
        key = (seq_len, random_states)
        inputs = generate_inputs(seq_len, cfg, seed=42, random_states=random_states)
        outputs, ct, it = run_on_device(
            core, model_dir, seq_len, cfg, "GPU",
            {"INFERENCE_PRECISION_HINT": "f32"}, inputs,
            f"GPU-FP32 S={seq_len}", static=False)
        if outputs is not None:
            gpu_results[key] = {"outputs": outputs, "inputs": inputs, "ct": ct, "it": it}
        else:
            logger.error("GPU FP32 failed for S=%d", seq_len)

    if not gpu_results:
        logger.error("All GPU runs failed!")
        sys.exit(1)

    # --- Phase 2: GPU FP16 (to isolate precision effect) ---
    print("\n" + "=" * 80)
    print("PHASE 2: GPU FP16 (isolate precision effect from NPU)")
    print("=" * 80)

    gpu_fp16_results = {}
    for key, ref in gpu_results.items():
        seq_len, random_states = key
        outputs, ct, it = run_on_device(
            core, model_dir, seq_len, cfg, "GPU",
            {"INFERENCE_PRECISION_HINT": "f16"}, ref["inputs"],
            f"GPU-FP16 S={seq_len}", static=False)
        if outputs is not None:
            gpu_fp16_results[key] = {"outputs": outputs, "ct": ct, "it": it}

    # --- Phase 3: NPU (static shapes required) ---
    print("\n" + "=" * 80)
    print("PHASE 3: NPU (static shapes)")
    print("=" * 80)

    if "NPU" not in core.available_devices:
        logger.warning("NPU not available, skipping")
        npu_results = {}
    else:
        npu_results = {}
        for key, ref in gpu_results.items():
            seq_len, random_states = key
            outputs, ct, it = run_on_device(
                core, model_dir, seq_len, cfg, "NPU",
                {"NPU_COMPILER_TYPE": "PREFER_PLUGIN"}, ref["inputs"],
                f"NPU S={seq_len}", static=True)
            if outputs is not None:
                npu_results[key] = {"outputs": outputs, "ct": ct, "it": it}

    # --- Summary ---
    print("\n" + "=" * 100)
    print("SUMMARY: GDN Prefill Block 0 — Precision Comparison (hidden output only)")
    print("=" * 100)

    fmt = "  {:>4s}  {:>7s}  {:>12s}  {:>12s}  {:>10s}  {:>9s}  {:>12s}  {:>12s}  {:>10s}  {:>9s}"
    print(fmt.format(
        "S", "States",
        "FP16 MaxAbs", "FP16 MeanAbs", "FP16 Cos", "FP16(ms)",
        "NPU MaxAbs", "NPU MeanAbs", "NPU Cos", "NPU(ms)"))
    print("  " + "-" * 110)

    for key in sorted(gpu_results.keys()):
        seq_len, random_states = key
        ref = gpu_results[key]
        state_label = "random" if random_states else "zeros"

        # FP16 comparison
        fp16_str = ["—"] * 4
        if key in gpu_fp16_results:
            cmp = compare_outputs(ref["outputs"], gpu_fp16_results[key]["outputs"])[0]
            fp16_str = [
                f"{cmp['max_abs_err']:.4e}",
                f"{cmp['mean_abs_err']:.4e}",
                f"{cmp['cosine_sim']:.6f}",
                f"{gpu_fp16_results[key]['it']*1000:.2f}",
            ]

        # NPU comparison
        npu_str = ["—"] * 4
        if key in npu_results:
            cmp = compare_outputs(ref["outputs"], npu_results[key]["outputs"])[0]
            npu_str = [
                f"{cmp['max_abs_err']:.4e}",
                f"{cmp['mean_abs_err']:.4e}",
                f"{cmp['cosine_sim']:.6f}",
                f"{npu_results[key]['it']*1000:.2f}",
            ]
        elif key in gpu_results:
            npu_str = ["FAILED", "", "", ""]

        print(fmt.format(str(seq_len), state_label, *fp16_str, *npu_str))

    # Detailed per-output for largest S with NPU data
    npu_ok = {k: v for k, v in npu_results.items()}
    if npu_ok:
        last_key = max(npu_ok.keys(), key=lambda k: k[0])
        seq_len, random_states = last_key
        print(f"\n  Per-output detail for S={seq_len} (NPU vs GPU-FP32):")
        cmp_all = compare_outputs(gpu_results[last_key]["outputs"],
                                   npu_results[last_key]["outputs"])
        for c in cmp_all:
            print(f"    {c['name']:>12s}: max_abs={c['max_abs_err']:.4e} "
                  f"cos={c['cosine_sim']:.6f} range={c['ref_range']}")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
