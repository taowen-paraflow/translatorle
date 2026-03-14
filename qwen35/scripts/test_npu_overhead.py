"""Benchmark NPU attention block infer() overhead with various configurations.

Loads a single attention block on NPU, reshapes to S=1 decode shape,
and measures per-infer() latency across multiple NPU property configurations.

Usage (from project root, Windows):
    powershell.exe -Command '$env:PYTHONIOENCODING="utf-8"; ...'
"""

import json
import time
import sys
from pathlib import Path

import numpy as np
import openvino as ov
from openvino.preprocess import PrePostProcessor


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

QUANTIZED_DIR = Path("models/qwen35/Qwen3.5-0.8B-hybrid-attn-int4sym-gdn-int8sym-head-int4sym")
FP16_DIR = Path("models/qwen35/Qwen3.5-0.8B-hybrid")

ATTN_PAST_SEQ = 256
BLOCK_IDX = 0  # which attn block to benchmark
WARMUP = 10
ITERS = 100


def load_config(model_dir: Path) -> dict:
    with open(model_dir / "config.json") as f:
        root = json.load(f)
    cfg = root.get("text_config", root)
    return {
        "hidden_size": cfg["hidden_size"],
        "num_kv_heads": cfg["num_key_value_heads"],
        "head_dim": cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"]),
    }


def add_f32_output_conversion(model):
    ppp = PrePostProcessor(model)
    for i in range(len(model.outputs)):
        ppp.output(i).tensor().set_element_type(ov.Type.f32)
    return ppp.build()


def reshape_attn_static(ir, hidden_size, num_kv_heads, head_dim, past_seq, seq_len=1):
    """Reshape attention IR to fully static shapes for NPU compilation."""
    B, S = 1, seq_len
    H, KV, D, P = hidden_size, num_kv_heads, head_dim, past_seq
    shape_map = {
        0: [B, S, H],        # hidden
        1: [3, B, S],        # position_ids
        2: [B, KV, P, D],    # key_cache
        3: [B, KV, P, D],    # value_cache
        4: [S],              # cache_position
        5: [B, 1, S, P],     # attention_mask
    }
    for i, shape in shape_map.items():
        ir.inputs[i].get_node().set_partial_shape(ov.PartialShape(shape))
    ir.validate_nodes_and_infer_types()
    return ir


def create_random_inputs(hidden_size, num_kv_heads, head_dim, past_seq):
    """Create random input tensors matching S=1 decode shapes."""
    rng = np.random.default_rng(42)
    hidden = rng.standard_normal((1, 1, hidden_size)).astype(np.float32)
    position_ids = np.array([[[10]], [[10]], [[10]]], dtype=np.int64)  # [3,1,1]
    key_cache = rng.standard_normal((1, num_kv_heads, past_seq, head_dim)).astype(np.float32)
    value_cache = rng.standard_normal((1, num_kv_heads, past_seq, head_dim)).astype(np.float32)
    cache_position = np.array([10], dtype=np.int64)
    # Causal mask: attend to positions 0..10, mask rest
    attn_mask = np.full((1, 1, 1, past_seq), -65504.0, dtype=np.float32)
    attn_mask[0, 0, 0, :11] = 0.0
    return {
        "hidden": ov.Tensor(hidden),
        "position_ids": ov.Tensor(position_ids),
        "key_cache": ov.Tensor(key_cache),
        "value_cache": ov.Tensor(value_cache),
        "cache_position": ov.Tensor(cache_position),
        "attn_mask": ov.Tensor(attn_mask),
    }


def benchmark_infer(req, inputs, warmup, iters, label):
    """Run warmup + timed iterations, report statistics."""
    # Bind inputs
    req.set_input_tensor(0, inputs["hidden"])
    req.set_input_tensor(1, inputs["position_ids"])
    req.set_input_tensor(2, inputs["key_cache"])
    req.set_input_tensor(3, inputs["value_cache"])
    req.set_input_tensor(4, inputs["cache_position"])
    req.set_input_tensor(5, inputs["attn_mask"])

    # Warmup
    for _ in range(warmup):
        req.infer()

    # Timed iterations
    latencies = []
    for _ in range(iters):
        t0 = time.perf_counter()
        req.infer()
        latencies.append((time.perf_counter() - t0) * 1000.0)  # ms

    latencies = np.array(latencies)
    mean = np.mean(latencies)
    std = np.std(latencies)
    mn = np.min(latencies)
    mx = np.max(latencies)
    p50 = np.percentile(latencies, 50)
    p99 = np.percentile(latencies, 99)

    print(f"  [{label}]")
    print(f"    mean={mean:.3f}ms  std={std:.3f}ms  min={mn:.3f}ms  max={mx:.3f}ms  p50={p50:.3f}ms  p99={p99:.3f}ms")
    return mean


def benchmark_python_overhead(req, inputs, iters):
    """Measure just the Python-side overhead of calling infer() by timing
    an empty loop vs the infer loop."""
    import ctypes

    # Time the raw infer calls
    latencies_infer = []
    for _ in range(iters):
        t0 = time.perf_counter()
        req.infer()
        latencies_infer.append((time.perf_counter() - t0) * 1000.0)

    # Time an empty loop with perf_counter overhead
    latencies_empty = []
    for _ in range(iters):
        t0 = time.perf_counter()
        latencies_empty.append((time.perf_counter() - t0) * 1000.0)

    infer_mean = np.mean(latencies_infer)
    empty_mean = np.mean(latencies_empty)
    print(f"\n  Python call overhead estimate:")
    print(f"    infer() mean: {infer_mean:.3f}ms")
    print(f"    empty loop mean: {empty_mean:.4f}ms")
    print(f"    perf_counter resolution: ~{empty_mean*1000:.1f}us")


def compile_and_benchmark(core, model_dir, cfg, config_name, npu_config, cache_subdir=None):
    """Read model, reshape, compile with given config, benchmark."""
    xml = str(model_dir / f"attn_block_{BLOCK_IDX}.xml")
    ir = core.read_model(xml)
    ir = reshape_attn_static(
        ir, cfg["hidden_size"], cfg["num_kv_heads"], cfg["head_dim"], ATTN_PAST_SEQ
    )
    ir = add_f32_output_conversion(ir)

    # Compile
    print(f"\nCompiling attn_block_{BLOCK_IDX} on NPU with config: {config_name}")
    print(f"  NPU properties: {npu_config}")
    t0 = time.time()
    compiled = core.compile_model(ir, "NPU", npu_config)
    compile_time = time.time() - t0
    print(f"  Compilation: {compile_time:.1f}s")

    req = compiled.create_infer_request()
    inputs = create_random_inputs(cfg["hidden_size"], cfg["num_kv_heads"], cfg["head_dim"], ATTN_PAST_SEQ)

    mean = benchmark_infer(req, inputs, WARMUP, ITERS, config_name)
    return mean


def main():
    # Find model dir
    model_dir = None
    for d in [QUANTIZED_DIR, FP16_DIR]:
        if (d / f"attn_block_{BLOCK_IDX}.xml").exists():
            model_dir = d
            break
    if model_dir is None:
        print("ERROR: No model directory found with attn_block files.")
        sys.exit(1)

    print(f"Model dir: {model_dir}")
    cfg = load_config(model_dir)
    print(f"Config: hidden={cfg['hidden_size']} kv_heads={cfg['num_kv_heads']} head_dim={cfg['head_dim']}")
    print(f"Shapes: hidden=[1,1,{cfg['hidden_size']}] KV=[1,{cfg['num_kv_heads']},{ATTN_PAST_SEQ},{cfg['head_dim']}]")
    print(f"Warmup={WARMUP} Iters={ITERS}")

    core = ov.Core()

    # Print available NPU properties
    print("\n--- Available NPU properties ---")
    try:
        props = core.get_property("NPU", "SUPPORTED_PROPERTIES")
        for p in sorted(props):
            try:
                val = core.get_property("NPU", p)
                print(f"  {p} = {val}")
            except Exception:
                print(f"  {p} = <unreadable>")
    except Exception as e:
        print(f"  Could not enumerate: {e}")

    # Use a separate cache dir for each config to avoid cross-contamination
    base_cache = str(model_dir / "cache_overhead_test")

    results = {}

    # -----------------------------------------------------------------------
    # Config 1: Baseline (current production config)
    # -----------------------------------------------------------------------
    core1 = ov.Core()
    core1.set_property(ov.properties.cache_dir(base_cache + "/baseline"))
    results["1_baseline"] = compile_and_benchmark(
        core1, model_dir, cfg,
        "1_baseline (PREFER_PLUGIN + CACHE_DIR)",
        {"NPU_COMPILER_TYPE": "PREFER_PLUGIN"},
    )

    # -----------------------------------------------------------------------
    # Config 2: +turbo
    # -----------------------------------------------------------------------
    core2 = ov.Core()
    core2.set_property(ov.properties.cache_dir(base_cache + "/turbo"))
    try:
        results["2_turbo"] = compile_and_benchmark(
            core2, model_dir, cfg,
            "2_baseline + turbo=True",
            {
                "NPU_COMPILER_TYPE": "PREFER_PLUGIN",
                "NPU_TURBO": True,
            },
        )
    except Exception as e:
        print(f"\n  Config 2 (turbo) FAILED: {e}")
        results["2_turbo"] = None

    # -----------------------------------------------------------------------
    # Config 3: +PERFORMANCE_HINT=LATENCY
    # -----------------------------------------------------------------------
    core3 = ov.Core()
    core3.set_property(ov.properties.cache_dir(base_cache + "/latency"))
    try:
        results["3_latency_hint"] = compile_and_benchmark(
            core3, model_dir, cfg,
            "3_baseline + PERFORMANCE_HINT=LATENCY",
            {
                "NPU_COMPILER_TYPE": "PREFER_PLUGIN",
                ov.properties.hint.performance_mode(): ov.properties.hint.PerformanceMode.LATENCY,
            },
        )
    except Exception as e:
        print(f"\n  Config 3 (latency_hint) FAILED: {e}")
        results["3_latency_hint"] = None

    # -----------------------------------------------------------------------
    # Config 4: +NUM_STREAMS=1
    # -----------------------------------------------------------------------
    core4 = ov.Core()
    core4.set_property(ov.properties.cache_dir(base_cache + "/streams1"))
    try:
        results["4_num_streams_1"] = compile_and_benchmark(
            core4, model_dir, cfg,
            "4_baseline + NUM_STREAMS=1",
            {
                "NPU_COMPILER_TYPE": "PREFER_PLUGIN",
                ov.properties.num_streams(): 1,
            },
        )
    except Exception as e:
        print(f"\n  Config 4 (num_streams=1) FAILED: {e}")
        results["4_num_streams_1"] = None

    # -----------------------------------------------------------------------
    # Config 5: +run_inferences_sequentially (requires start_async)
    # -----------------------------------------------------------------------
    core5 = ov.Core()
    core5.set_property(ov.properties.cache_dir(base_cache + "/sequential"))
    try:
        print(f"\nCompiling attn_block_{BLOCK_IDX} on NPU with config: 5_sequential (async)")
        npu_config_5 = {
            "NPU_COMPILER_TYPE": "PREFER_PLUGIN",
            "NPU_RUN_INFERENCES_SEQUENTIALLY": True,
        }
        print(f"  NPU properties: {npu_config_5}")
        xml = str(model_dir / f"attn_block_{BLOCK_IDX}.xml")
        ir = core5.read_model(xml)
        ir = reshape_attn_static(ir, cfg["hidden_size"], cfg["num_kv_heads"], cfg["head_dim"], ATTN_PAST_SEQ)
        ir = add_f32_output_conversion(ir)
        t0 = time.time()
        compiled5 = core5.compile_model(ir, "NPU", npu_config_5)
        print(f"  Compilation: {time.time()-t0:.1f}s")
        req5 = compiled5.create_infer_request()
        inputs5 = create_random_inputs(cfg["hidden_size"], cfg["num_kv_heads"], cfg["head_dim"], ATTN_PAST_SEQ)
        req5.set_input_tensor(0, inputs5["hidden"])
        req5.set_input_tensor(1, inputs5["position_ids"])
        req5.set_input_tensor(2, inputs5["key_cache"])
        req5.set_input_tensor(3, inputs5["value_cache"])
        req5.set_input_tensor(4, inputs5["cache_position"])
        req5.set_input_tensor(5, inputs5["attn_mask"])
        # Warmup with async
        for _ in range(WARMUP):
            req5.start_async()
            req5.wait()
        # Timed
        latencies5 = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            req5.start_async()
            req5.wait()
            latencies5.append((time.perf_counter() - t0) * 1000.0)
        arr5 = np.array(latencies5)
        mean5 = np.mean(arr5)
        print(f"  [5_sequential (start_async+wait)]")
        print(f"    mean={mean5:.3f}ms  std={np.std(arr5):.3f}ms  min={np.min(arr5):.3f}ms  max={np.max(arr5):.3f}ms  p50={np.percentile(arr5,50):.3f}ms  p99={np.percentile(arr5,99):.3f}ms")
        results["5_sequential"] = mean5
    except Exception as e:
        print(f"\n  Config 5 (sequential) FAILED: {e}")
        results["5_sequential"] = None

    # -----------------------------------------------------------------------
    # Config 6: turbo + latency + streams=1
    # -----------------------------------------------------------------------
    core6 = ov.Core()
    core6.set_property(ov.properties.cache_dir(base_cache + "/all_combined"))
    try:
        combined_config = {
            "NPU_COMPILER_TYPE": "PREFER_PLUGIN",
            "NPU_TURBO": True,
            ov.properties.hint.performance_mode(): ov.properties.hint.PerformanceMode.LATENCY,
            ov.properties.num_streams(): 1,
        }
        results["6_all_combined"] = compile_and_benchmark(
            core6, model_dir, cfg,
            "6_all_combined (turbo + latency + streams=1)",
            combined_config,
        )
    except Exception as e:
        print(f"\n  Config 6 (all_combined) FAILED: {e}")
        results["6_all_combined"] = None

    # -----------------------------------------------------------------------
    # Config 7: No CACHE_DIR (measure cache overhead)
    # -----------------------------------------------------------------------
    core7 = ov.Core()
    # Explicitly do NOT set cache_dir
    results["7_no_cache"] = compile_and_benchmark(
        core7, model_dir, cfg,
        "7_no_cache_dir (PREFER_PLUGIN only)",
        {"NPU_COMPILER_TYPE": "PREFER_PLUGIN"},
    )

    # -----------------------------------------------------------------------
    # Config 8: DRIVER compiler instead of PREFER_PLUGIN
    # -----------------------------------------------------------------------
    core8 = ov.Core()
    core8.set_property(ov.properties.cache_dir(base_cache + "/driver"))
    try:
        results["8_driver_compiler"] = compile_and_benchmark(
            core8, model_dir, cfg,
            "8_DRIVER compiler (no PREFER_PLUGIN)",
            {"NPU_COMPILER_TYPE": "DRIVER"},
        )
    except Exception as e:
        print(f"\n  Config 8 (driver compiler) FAILED: {e}")
        results["8_driver_compiler"] = None

    # -----------------------------------------------------------------------
    # Python overhead measurement
    # -----------------------------------------------------------------------
    print("\n--- Python overhead measurement ---")
    core_oh = ov.Core()
    core_oh.set_property(ov.properties.cache_dir(base_cache + "/overhead"))
    xml = str(model_dir / f"attn_block_{BLOCK_IDX}.xml")
    ir = core_oh.read_model(xml)
    ir = reshape_attn_static(ir, cfg["hidden_size"], cfg["num_kv_heads"], cfg["head_dim"], ATTN_PAST_SEQ)
    ir = add_f32_output_conversion(ir)
    compiled = core_oh.compile_model(ir, "NPU", {"NPU_COMPILER_TYPE": "PREFER_PLUGIN"})
    req = compiled.create_infer_request()
    inputs = create_random_inputs(cfg["hidden_size"], cfg["num_kv_heads"], cfg["head_dim"], ATTN_PAST_SEQ)
    req.set_input_tensor(0, inputs["hidden"])
    req.set_input_tensor(1, inputs["position_ids"])
    req.set_input_tensor(2, inputs["key_cache"])
    req.set_input_tensor(3, inputs["value_cache"])
    req.set_input_tensor(4, inputs["cache_position"])
    req.set_input_tensor(5, inputs["attn_mask"])
    # Warmup
    for _ in range(WARMUP):
        req.infer()
    benchmark_python_overhead(req, inputs, ITERS)

    # -----------------------------------------------------------------------
    # Batch infer test: 6 consecutive infer() calls (simulating 6 attn blocks)
    # -----------------------------------------------------------------------
    print("\n--- Batch sequential infer() test (6 blocks) ---")
    core_batch = ov.Core()
    core_batch.set_property(ov.properties.cache_dir(base_cache + "/batch"))
    reqs = []
    for i in range(6):
        xml = str(model_dir / f"attn_block_{i}.xml")
        ir = core_batch.read_model(xml)
        ir = reshape_attn_static(ir, cfg["hidden_size"], cfg["num_kv_heads"], cfg["head_dim"], ATTN_PAST_SEQ)
        ir = add_f32_output_conversion(ir)
        compiled = core_batch.compile_model(ir, "NPU", {"NPU_COMPILER_TYPE": "PREFER_PLUGIN"})
        reqs.append(compiled.create_infer_request())

    all_inputs = []
    for i in range(6):
        inp = create_random_inputs(cfg["hidden_size"], cfg["num_kv_heads"], cfg["head_dim"], ATTN_PAST_SEQ)
        reqs[i].set_input_tensor(0, inp["hidden"])
        reqs[i].set_input_tensor(1, inp["position_ids"])
        reqs[i].set_input_tensor(2, inp["key_cache"])
        reqs[i].set_input_tensor(3, inp["value_cache"])
        reqs[i].set_input_tensor(4, inp["cache_position"])
        reqs[i].set_input_tensor(5, inp["attn_mask"])
        all_inputs.append(inp)

    # Warmup
    for _ in range(WARMUP):
        for r in reqs:
            r.infer()

    # Measure: 6 sequential infer() calls
    batch_latencies = []
    per_block_latencies = [[] for _ in range(6)]
    for _ in range(ITERS):
        t0 = time.perf_counter()
        for bi, r in enumerate(reqs):
            tb = time.perf_counter()
            r.infer()
            per_block_latencies[bi].append((time.perf_counter() - tb) * 1000.0)
        batch_latencies.append((time.perf_counter() - t0) * 1000.0)

    batch_arr = np.array(batch_latencies)
    print(f"  6-block sequential: mean={np.mean(batch_arr):.3f}ms  std={np.std(batch_arr):.3f}ms  "
          f"min={np.min(batch_arr):.3f}ms  max={np.max(batch_arr):.3f}ms")
    for bi in range(6):
        arr = np.array(per_block_latencies[bi])
        print(f"    block_{bi}: mean={np.mean(arr):.3f}ms  std={np.std(arr):.3f}ms  "
              f"min={np.min(arr):.3f}ms  max={np.max(arr):.3f}ms")

    # Also measure with start_async + wait
    print("\n--- Async infer test (start_async + wait) ---")
    for _ in range(WARMUP):
        for r in reqs:
            r.start_async()
            r.wait()

    async_latencies = []
    async_per_block = [[] for _ in range(6)]
    for _ in range(ITERS):
        t0 = time.perf_counter()
        for bi, r in enumerate(reqs):
            tb = time.perf_counter()
            r.start_async()
            r.wait()
            async_per_block[bi].append((time.perf_counter() - tb) * 1000.0)
        async_latencies.append((time.perf_counter() - t0) * 1000.0)

    async_arr = np.array(async_latencies)
    print(f"  6-block async+wait: mean={np.mean(async_arr):.3f}ms  std={np.std(async_arr):.3f}ms  "
          f"min={np.min(async_arr):.3f}ms  max={np.max(async_arr):.3f}ms")
    for bi in range(6):
        arr = np.array(async_per_block[bi])
        print(f"    block_{bi}: mean={np.mean(arr):.3f}ms  std={np.std(arr):.3f}ms  "
              f"min={np.min(arr):.3f}ms  max={np.max(arr):.3f}ms")

    # -----------------------------------------------------------------------
    # Batch infer with TURBO (6 blocks, start_async+wait)
    # -----------------------------------------------------------------------
    print("\n--- Batch sequential infer() with TURBO (6 blocks) ---")
    core_turbo_batch = ov.Core()
    core_turbo_batch.set_property(ov.properties.cache_dir(base_cache + "/turbo_batch"))
    turbo_reqs = []
    for i in range(6):
        xml = str(model_dir / f"attn_block_{i}.xml")
        ir = core_turbo_batch.read_model(xml)
        ir = reshape_attn_static(ir, cfg["hidden_size"], cfg["num_kv_heads"], cfg["head_dim"], ATTN_PAST_SEQ)
        ir = add_f32_output_conversion(ir)
        compiled = core_turbo_batch.compile_model(ir, "NPU", {
            "NPU_COMPILER_TYPE": "PREFER_PLUGIN",
            "NPU_TURBO": True,
        })
        turbo_reqs.append(compiled.create_infer_request())

    for i in range(6):
        inp = create_random_inputs(cfg["hidden_size"], cfg["num_kv_heads"], cfg["head_dim"], ATTN_PAST_SEQ)
        turbo_reqs[i].set_input_tensor(0, inp["hidden"])
        turbo_reqs[i].set_input_tensor(1, inp["position_ids"])
        turbo_reqs[i].set_input_tensor(2, inp["key_cache"])
        turbo_reqs[i].set_input_tensor(3, inp["value_cache"])
        turbo_reqs[i].set_input_tensor(4, inp["cache_position"])
        turbo_reqs[i].set_input_tensor(5, inp["attn_mask"])

    # Warmup
    for _ in range(WARMUP):
        for r in turbo_reqs:
            r.start_async()
            r.wait()

    turbo_batch_latencies = []
    turbo_per_block = [[] for _ in range(6)]
    for _ in range(ITERS):
        t0 = time.perf_counter()
        for bi, r in enumerate(turbo_reqs):
            tb = time.perf_counter()
            r.start_async()
            r.wait()
            turbo_per_block[bi].append((time.perf_counter() - tb) * 1000.0)
        turbo_batch_latencies.append((time.perf_counter() - t0) * 1000.0)

    turbo_batch_arr = np.array(turbo_batch_latencies)
    print(f"  6-block turbo async+wait: mean={np.mean(turbo_batch_arr):.3f}ms  std={np.std(turbo_batch_arr):.3f}ms  "
          f"min={np.min(turbo_batch_arr):.3f}ms  max={np.max(turbo_batch_arr):.3f}ms")
    for bi in range(6):
        arr = np.array(turbo_per_block[bi])
        print(f"    block_{bi}: mean={np.mean(arr):.3f}ms  std={np.std(arr):.3f}ms  "
              f"min={np.min(arr):.3f}ms  max={np.max(arr):.3f}ms")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY (mean latency per single attn block infer, S=1 decode)")
    print("=" * 70)
    baseline = results.get("1_baseline")
    for name, val in sorted(results.items()):
        if val is None:
            print(f"  {name:35s}  FAILED")
        else:
            delta = ""
            if baseline is not None and name != "1_baseline":
                diff_pct = (val - baseline) / baseline * 100
                delta = f"  ({diff_pct:+.1f}%)"
            print(f"  {name:35s}  {val:.3f}ms{delta}")
    print("=" * 70)


if __name__ == "__main__":
    main()
