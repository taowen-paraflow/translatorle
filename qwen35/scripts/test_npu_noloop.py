"""Test NPU compilation and inference of GDN noloop block IR.

Quick check: can the NPU compile and run the Loop-free GDN S=1 blocks?
These blocks use SingleStepRecurrentAttentionCell (unrolled MatMul/Exp/Add),
no Loop/TensorIterator nodes, so NPU compiler should handle them.

Usage (from project root on Windows):
    uv run python -m qwen35.scripts.test_npu_noloop
"""

import time
from pathlib import Path

import numpy as np
import openvino as ov

# Model directories to try (quantized first, then FP16 baseline)
MODEL_DIRS = [
    Path("models/qwen35/Qwen3.5-0.8B-hybrid-attn-int4sym-gdn-int8sym-head-int4sym"),
    Path("models/qwen35/Qwen3.5-0.8B-hybrid"),
]

# I/O shapes for B=1, S=1 (from export_hybrid.py)
# conv_dim = linear_num_key_heads * linear_key_head_dim * 2
#          + linear_num_value_heads * linear_value_head_dim
#          = 4 * 128 * 2 + 4 * 128 = 6144  (Qwen3.5-0.8B config, may differ)
# conv_kernel = 4, num_v_heads = 4, k_head_dim = 128, v_head_dim = 128
HIDDEN_SIZE = 1024
CONV_DIM = 6144
CONV_KERNEL = 4
NUM_V_HEADS = 16
K_HEAD_DIM = 128
V_HEAD_DIM = 128

NUM_BLOCKS = 6
DEVICE = "NPU"


def make_dummy_inputs():
    """Create dummy input tensors matching noloop block I/O."""
    return {
        "in_hidden": np.zeros((1, 1, HIDDEN_SIZE), dtype=np.float32),
        "in_mask": np.ones((1, 1), dtype=np.int64),
        "in_conv0": np.zeros((1, CONV_DIM, CONV_KERNEL), dtype=np.float32),
        "in_rec0": np.zeros((1, NUM_V_HEADS, K_HEAD_DIM, V_HEAD_DIM), dtype=np.float32),
        "in_conv1": np.zeros((1, CONV_DIM, CONV_KERNEL), dtype=np.float32),
        "in_rec1": np.zeros((1, NUM_V_HEADS, K_HEAD_DIM, V_HEAD_DIM), dtype=np.float32),
        "in_conv2": np.zeros((1, CONV_DIM, CONV_KERNEL), dtype=np.float32),
        "in_rec2": np.zeros((1, NUM_V_HEADS, K_HEAD_DIM, V_HEAD_DIM), dtype=np.float32),
    }


def test_block(core, model_dir, block_idx):
    """Try to compile and run a single noloop block on NPU."""
    xml_path = model_dir / f"gdn_noloop_block_{block_idx}.xml"
    if not xml_path.exists():
        print(f"  [SKIP] {xml_path} not found")
        return False

    print(f"\n  Block {block_idx}: {xml_path.name}")

    # Load IR
    model = core.read_model(str(xml_path))

    # Print model info
    ops = list(model.get_ops())
    loop_ops = [op for op in ops if op.get_type_name() in ("Loop", "TensorIterator")]
    matmul_ops = [op for op in ops if op.get_type_name() == "MatMul"]
    print(f"    Ops: {len(ops)} total, {len(matmul_ops)} MatMul, {len(loop_ops)} Loop")

    if loop_ops:
        print("    [WARN] Found Loop ops -- NPU will likely fail!")

    # Print I/O shapes
    for inp in model.inputs:
        names = inp.get_tensor().get_names()
        name = next(iter(names)) if names else f"input_{inp.get_index()}"
        print(f"    Input:  {name:12s} {inp.partial_shape}")
    for out in model.outputs:
        names = out.get_tensor().get_names()
        name = next(iter(names)) if names else f"output_{out.get_index()}"
        print(f"    Output: {name:12s} {out.partial_shape}")

    # Reshape to static B=1, S=1 shapes (NPU requires fully static shapes)
    static_shapes = {
        "in_hidden": [1, 1, HIDDEN_SIZE],
        "in_mask": [1, 1],
        "in_conv0": [1, CONV_DIM, CONV_KERNEL],
        "in_rec0": [1, NUM_V_HEADS, K_HEAD_DIM, V_HEAD_DIM],
        "in_conv1": [1, CONV_DIM, CONV_KERNEL],
        "in_rec1": [1, NUM_V_HEADS, K_HEAD_DIM, V_HEAD_DIM],
        "in_conv2": [1, CONV_DIM, CONV_KERNEL],
        "in_rec2": [1, NUM_V_HEADS, K_HEAD_DIM, V_HEAD_DIM],
    }
    model.reshape(static_shapes)
    print("    Reshaped to static B=1, S=1")

    # Compile
    npu_config = {"NPU_COMPILER_TYPE": "PREFER_PLUGIN"}
    print(f"    Compiling on {DEVICE}...", end=" ", flush=True)
    t0 = time.perf_counter()
    try:
        compiled = core.compile_model(model, DEVICE, npu_config)
        compile_time = time.perf_counter() - t0
        print(f"OK ({compile_time:.2f}s)")
    except Exception as e:
        compile_time = time.perf_counter() - t0
        print(f"FAILED ({compile_time:.2f}s)")
        print(f"    Error: {e}")
        return False

    # Inference
    dummy = make_dummy_inputs()
    infer_req = compiled.create_infer_request()

    print("    Running inference...", end=" ", flush=True)
    t0 = time.perf_counter()
    try:
        results = infer_req.infer(dummy)
        infer_time = time.perf_counter() - t0
        print(f"OK ({infer_time * 1000:.1f}ms)")
    except Exception as e:
        infer_time = time.perf_counter() - t0
        print(f"FAILED ({infer_time * 1000:.1f}ms)")
        print(f"    Error: {e}")
        return False

    # Check outputs
    for i, out in enumerate(compiled.outputs):
        names = out.get_tensor().get_names()
        name = next(iter(names)) if names else f"output_{i}"
        tensor = results[out]
        print(f"    Output {name:12s}: shape={tensor.shape}, "
              f"min={tensor.min():.6f}, max={tensor.max():.6f}, "
              f"mean={tensor.mean():.6f}")

    # Warm-up + latency measurement (5 runs)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        infer_req.infer(dummy)
        times.append(time.perf_counter() - t0)
    avg_ms = np.mean(times) * 1000
    min_ms = np.min(times) * 1000
    print(f"    Latency (5 runs): avg={avg_ms:.1f}ms, min={min_ms:.1f}ms")

    return True


def main():
    core = ov.Core()

    # Show available devices
    devices = core.available_devices
    print(f"Available devices: {devices}")
    if DEVICE not in devices:
        print(f"\n[ERROR] {DEVICE} not available. Cannot run test.")
        return

    # Find model directory
    model_dir = None
    for d in MODEL_DIRS:
        if d.exists() and (d / "gdn_noloop_block_0.xml").exists():
            model_dir = d
            break

    if model_dir is None:
        print("\n[ERROR] No model directory found with gdn_noloop_block_0.xml")
        print("Searched:")
        for d in MODEL_DIRS:
            print(f"  {d} (exists={d.exists()})")
        return

    print(f"\nUsing model dir: {model_dir}")

    # Test each block
    success = 0
    failed = 0
    for i in range(NUM_BLOCKS):
        if test_block(core, model_dir, i):
            success += 1
        else:
            failed += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"Results: {success}/{NUM_BLOCKS} blocks compiled+inferred on {DEVICE}")
    if failed:
        print(f"  {failed} blocks FAILED")
    else:
        print("  All blocks passed!")


if __name__ == "__main__":
    main()
