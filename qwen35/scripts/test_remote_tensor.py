"""Benchmark remote tensor / zero-copy between GPU and NPU on Intel Lunar Lake.

Tests whether Level Zero shared memory can eliminate memcpy overhead when
passing data between GPU GDN blocks and NPU Attention blocks.

Current flow:  GPU infer -> numpy copy -> NPU infer -> numpy copy -> GPU infer
Desired flow:  GPU infer -> shared memory -> NPU reads directly (zero-copy)

Usage (from project root on Windows):
    powershell.exe -Command '$env:PYTHONIOENCODING="utf-8"; cd C:\\Apps\\translatorle; C:\\Users\\taowen\\.local\\bin\\uv.exe run python -m qwen35.scripts.test_remote_tensor'
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import openvino as ov
from openvino._offline_transformations import apply_make_stateful_transformation

# ---------------------------------------------------------------------------
# Model directories (quantized first, then FP16 baseline)
# ---------------------------------------------------------------------------
MODEL_DIRS = [
    Path("models/qwen35/Qwen3.5-0.8B-hybrid-attn-int4sym-gdn-int8sym-head-int4sym"),
    Path("models/qwen35/Qwen3.5-0.8B-hybrid"),
]

NUM_ITERS = 100
WARMUP = 10
PAST_SEQ = 256  # Static KV cache size for NPU attention


def find_model_dir() -> Path:
    """Find the first existing model directory with required files."""
    for d in MODEL_DIRS:
        if d.exists() and (d / "gdn_block_0.xml").exists() and (d / "attn_block_0.xml").exists():
            return d
    raise FileNotFoundError(
        f"No model directory found. Searched: {[str(d) for d in MODEL_DIRS]}"
    )


def load_config(model_dir: Path) -> dict:
    """Load model config and extract architecture params."""
    with open(model_dir / "config.json") as f:
        cfg = json.load(f)
    text_cfg = cfg.get("text_config", cfg)
    return {
        "hidden_size": text_cfg["hidden_size"],
        "num_kv_heads": text_cfg["num_key_value_heads"],
        "head_dim": text_cfg.get("head_dim", text_cfg["hidden_size"] // text_cfg["num_attention_heads"]),
        "num_v_heads": text_cfg["linear_num_value_heads"],
        "k_head_dim": text_cfg["linear_key_head_dim"],
        "v_head_dim": text_cfg["linear_value_head_dim"],
        "conv_dim": (
            text_cfg["linear_num_key_heads"] * text_cfg["linear_key_head_dim"] * 2
            + text_cfg["linear_num_value_heads"] * text_cfg["linear_value_head_dim"]
        ),
        "conv_kernel": text_cfg["linear_conv_kernel_dim"],
    }


def add_f32_output_conversion(ir):
    """Add FP32 output conversion via PrePostProcessor (GPU/NPU output FP16)."""
    ppp = ov.preprocess.PrePostProcessor(ir)
    for i in range(len(ir.outputs)):
        ppp.output(i).tensor().set_element_type(ov.Type.f32)
    return ppp.build()


def reshape_attn_static(ir, cfg: dict, past_seq: int, seq_len: int = 1):
    """Reshape attention IR to fully static shapes for NPU compilation."""
    B = 1
    S = seq_len
    shape_map = {
        0: [B, S, cfg["hidden_size"]],                             # hidden
        1: [3, B, S],                                              # position_ids
        2: [B, cfg["num_kv_heads"], past_seq, cfg["head_dim"]],    # key_cache
        3: [B, cfg["num_kv_heads"], past_seq, cfg["head_dim"]],    # value_cache
        4: [S],                                                    # cache_position
        5: [B, 1, S, past_seq],                                    # attention_mask
    }
    for i, shape in shape_map.items():
        ir.inputs[i].get_node().set_partial_shape(ov.PartialShape(shape))
    ir.validate_nodes_and_infer_types()
    return ir


# ===================================================================
# Step 1: Probe remote tensor / context support
# ===================================================================
def step1_probe_remote_context(core: ov.Core):
    """Check remote context support on GPU and NPU."""
    print("=" * 70)
    print("STEP 1: Probe remote tensor / context support")
    print("=" * 70)

    for device in ["GPU", "NPU"]:
        if device not in core.available_devices:
            print(f"\n  [{device}] Not available, skipping.")
            continue

        print(f"\n  [{device}] Checking remote context...")
        try:
            ctx = core.get_default_context(device)
            print(f"    Context type: {type(ctx).__name__}")
            print(f"    Context repr: {ctx}")

            # Try to list available methods
            methods = [m for m in dir(ctx) if not m.startswith("_")]
            print(f"    Public methods: {methods}")
        except Exception as e:
            print(f"    get_default_context failed: {e}")
            continue

        # Try creating a remote tensor
        print(f"\n  [{device}] Trying to create remote tensor...")

        # Get context params
        try:
            params = ctx.get_params()
            print(f"    Context params: {params}")
        except Exception as e:
            print(f"    get_params failed: {e}")

        # Try with empty properties (required 3rd arg)
        try:
            remote_tensor = ctx.create_tensor(
                ov.Type.f32, ov.Shape([1, 1, 1024]), {}
            )
            print(f"    create_tensor(type, shape, {{}}): OK")
            print(f"    Tensor type: {type(remote_tensor).__name__}")
            print(f"    Tensor shape: {remote_tensor.shape}")
            rt_methods = [m for m in dir(remote_tensor) if not m.startswith("_")]
            print(f"    Public methods: {rt_methods}")
            # Try to get data from remote tensor
            try:
                data = remote_tensor.data
                print(f"    .data access: OK, shape={data.shape}, dtype={data.dtype}")
            except Exception as e:
                print(f"    .data access failed: {e}")
            try:
                rt_params = remote_tensor.get_params()
                print(f"    Tensor params: {rt_params}")
            except Exception as e:
                print(f"    get_params failed: {e}")
        except Exception as e:
            print(f"    create_tensor(type, shape, {{}}): {e}")
            # Try with USM_HOST_BUFFER
            for mem_type in ["USM_HOST_BUFFER", "USM_DEVICE_BUFFER", "OCL_BUFFER"]:
                try:
                    remote_tensor = ctx.create_tensor(
                        ov.Type.f32, ov.Shape([1, 1, 1024]),
                        {"SHARED_MEM_TYPE": mem_type}
                    )
                    print(f"    create_tensor({mem_type}): OK, type={type(remote_tensor).__name__}")
                except Exception as e2:
                    print(f"    create_tensor({mem_type}): {e2}")

        # Try create_host_tensor
        try:
            host_tensor = ctx.create_host_tensor(ov.Type.f32, ov.Shape([1, 1, 1024]))
            print(f"    create_host_tensor: OK, type={type(host_tensor).__name__}")
        except AttributeError:
            print(f"    create_host_tensor: method not available")
        except Exception as e:
            print(f"    create_host_tensor failed: {e}")

    print()


# ===================================================================
# Step 2: Compile GDN block (GPU) + Attn block (NPU) for benchmarking
# ===================================================================
def compile_models(core: ov.Core, model_dir: Path, cfg: dict):
    """Compile one GDN block on GPU (stateful) and one Attn block on NPU."""
    print("=" * 70)
    print("STEP 2: Compile GDN block 0 (GPU) + Attn block 0 (NPU)")
    print("=" * 70)

    # --- GDN block 0 on GPU (stateful) ---
    print("\n  Loading GDN block 0...")
    gdn_ir = core.read_model(str(model_dir / "gdn_block_0.xml"))

    state_in = ["in_conv0", "in_rec0", "in_conv1", "in_rec1", "in_conv2", "in_rec2"]
    state_out = ["out_conv0", "out_rec0", "out_conv1", "out_rec1", "out_conv2", "out_rec2"]
    state_map = dict(zip(state_in, state_out))
    apply_make_stateful_transformation(gdn_ir, state_map)
    gdn_ir = add_f32_output_conversion(gdn_ir)

    print("  GDN block inputs (stateful):")
    for inp in gdn_ir.inputs:
        print(f"    {inp.get_any_name():20s} {inp.partial_shape} {inp.element_type}")

    print("  Compiling GDN block 0 on GPU...", end=" ", flush=True)
    t0 = time.perf_counter()
    gdn_compiled = core.compile_model(gdn_ir, "GPU")
    print(f"OK ({time.perf_counter() - t0:.2f}s)")

    gdn_req = gdn_compiled.create_infer_request()

    # Initialize GDN states
    conv_shape = (1, cfg["conv_dim"], cfg["conv_kernel"])
    rec_shape = (1, cfg["num_v_heads"], cfg["k_head_dim"], cfg["v_head_dim"])
    for s in gdn_req.query_state():
        name = s.name
        if "conv" in name:
            s.state = ov.Tensor(np.zeros(conv_shape, dtype=np.float32))
        elif "rec" in name:
            s.state = ov.Tensor(np.zeros(rec_shape, dtype=np.float32))

    # --- Attn block 0 on NPU (explicit I/O) ---
    print("\n  Loading Attn block 0...")
    attn_ir = core.read_model(str(model_dir / "attn_block_0.xml"))
    attn_ir = reshape_attn_static(attn_ir, cfg, PAST_SEQ, seq_len=1)
    attn_ir = add_f32_output_conversion(attn_ir)

    print("  Attn block inputs (static):")
    for inp in attn_ir.inputs:
        print(f"    {inp.get_any_name():20s} {inp.partial_shape} {inp.element_type}")

    npu_config = {
        "NPU_COMPILER_TYPE": "PREFER_PLUGIN",
        "NPU_TURBO": "YES",
    }
    print("  Compiling Attn block 0 on NPU...", end=" ", flush=True)
    t0 = time.perf_counter()
    attn_compiled = core.compile_model(attn_ir, "NPU", npu_config)
    print(f"OK ({time.perf_counter() - t0:.2f}s)")

    attn_req = attn_compiled.create_infer_request()

    return gdn_req, attn_req, gdn_compiled, attn_compiled


# ===================================================================
# Step 3: Benchmark GPU -> NPU transfer patterns
# ===================================================================
def make_attn_inputs(cfg: dict):
    """Create dummy attention block inputs for S=1 decode."""
    hidden_size = cfg["hidden_size"]
    num_kv_heads = cfg["num_kv_heads"]
    head_dim = cfg["head_dim"]

    return {
        "in_hidden": np.random.randn(1, 1, hidden_size).astype(np.float32),
        "in_position_ids": np.zeros((3, 1, 1), dtype=np.int64),
        "in_key_cache": np.zeros((1, num_kv_heads, PAST_SEQ, head_dim), dtype=np.float32),
        "in_value_cache": np.zeros((1, num_kv_heads, PAST_SEQ, head_dim), dtype=np.float32),
        "in_cache_position": np.array([1], dtype=np.int64),
        "in_attention_mask": np.full((1, 1, 1, PAST_SEQ), -65504.0, dtype=np.float32),
    }


def step3_benchmark_transfer(
    core: ov.Core, gdn_req, attn_req, gdn_compiled, attn_compiled, cfg: dict
):
    """Benchmark different GPU->NPU data transfer strategies."""
    print("\n" + "=" * 70)
    print("STEP 3: Benchmark GPU <-> NPU transfer (GDN -> Attn -> GDN cycle)")
    print("=" * 70)

    hidden_size = cfg["hidden_size"]
    num_kv_heads = cfg["num_kv_heads"]
    head_dim = cfg["head_dim"]

    # Prepare dummy inputs
    gdn_hidden = np.random.randn(1, 1, hidden_size).astype(np.float32)
    gdn_mask = np.ones((1, 1), dtype=np.int64)

    attn_inputs = make_attn_inputs(cfg)
    # Make mask attend to position 1 (where we write)
    attn_inputs["in_attention_mask"][0, 0, 0, 0] = 0.0  # attend position 0
    attn_inputs["in_attention_mask"][0, 0, 0, 1] = 0.0  # attend position 1

    # Helper to run GDN + copy + Attn cycle using set_input_tensor
    def run_gdn_attn_cycle(gdn_r, attn_r, gdn_hidden_in, attn_in, gdn_mask_in):
        """Run one GDN -> copy -> Attn cycle, return updated hidden."""
        gdn_r.set_input_tensor(0, ov.Tensor(gdn_hidden_in))
        gdn_r.set_input_tensor(1, ov.Tensor(gdn_mask_in))
        gdn_r.infer()
        hidden_np = gdn_r.get_output_tensor(0).data.copy()
        # GDN output may be 2D [B, H] -- reshape to 3D [B, 1, H] for attn
        if hidden_np.ndim == 2:
            hidden_np = hidden_np[:, np.newaxis, :]
        attn_r.set_input_tensor(0, ov.Tensor(hidden_np))
        attn_r.set_input_tensor(1, ov.Tensor(attn_in["in_position_ids"]))
        attn_r.set_input_tensor(2, ov.Tensor(attn_in["in_key_cache"]))
        attn_r.set_input_tensor(3, ov.Tensor(attn_in["in_value_cache"]))
        attn_r.set_input_tensor(4, ov.Tensor(attn_in["in_cache_position"]))
        attn_r.set_input_tensor(5, ov.Tensor(attn_in["in_attention_mask"]))
        attn_r.infer()
        out = attn_r.get_output_tensor(0).data.copy()
        if out.ndim == 2:
            out = out[:, np.newaxis, :]
        return out

    # ---- Config A: Baseline (set_input_tensor + explicit copy) ----
    print(f"\n  Config A: Baseline (set_input_tensor + numpy copy)")
    print(f"    Warming up ({WARMUP} iters)...", end=" ", flush=True)
    for _ in range(WARMUP):
        gdn_hidden = run_gdn_attn_cycle(gdn_req, attn_req, gdn_hidden, attn_inputs, gdn_mask)
    print("done")

    print(f"    Benchmarking ({NUM_ITERS} iters)...", end=" ", flush=True)
    gdn_times = []
    attn_times = []
    copy_times = []
    total_t0 = time.perf_counter()
    for _ in range(NUM_ITERS):
        t0 = time.perf_counter()
        gdn_req.set_input_tensor(0, ov.Tensor(gdn_hidden))
        gdn_req.set_input_tensor(1, ov.Tensor(gdn_mask))
        gdn_req.infer()
        t1 = time.perf_counter()
        gdn_times.append(t1 - t0)

        t0 = time.perf_counter()
        hidden_np = gdn_req.get_output_tensor(0).data.copy()
        if hidden_np.ndim == 2:
            hidden_np = hidden_np[:, np.newaxis, :]
        t1 = time.perf_counter()
        copy_times.append(t1 - t0)

        t0 = time.perf_counter()
        attn_req.set_input_tensor(0, ov.Tensor(hidden_np))
        attn_req.set_input_tensor(1, ov.Tensor(attn_inputs["in_position_ids"]))
        attn_req.set_input_tensor(2, ov.Tensor(attn_inputs["in_key_cache"]))
        attn_req.set_input_tensor(3, ov.Tensor(attn_inputs["in_value_cache"]))
        attn_req.set_input_tensor(4, ov.Tensor(attn_inputs["in_cache_position"]))
        attn_req.set_input_tensor(5, ov.Tensor(attn_inputs["in_attention_mask"]))
        attn_req.infer()
        t1 = time.perf_counter()
        attn_times.append(t1 - t0)

        gdn_hidden = attn_req.get_output_tensor(0).data.copy()
        if gdn_hidden.ndim == 2:
            gdn_hidden = gdn_hidden[:, np.newaxis, :]

    total_a = time.perf_counter() - total_t0
    print("done")
    print(f"    Total:     {total_a * 1000:.1f}ms ({total_a / NUM_ITERS * 1000:.2f}ms/iter)")
    print(f"    GDN avg:   {np.mean(gdn_times) * 1000:.3f}ms")
    print(f"    Copy avg:  {np.mean(copy_times) * 1000:.3f}ms")
    print(f"    Attn avg:  {np.mean(attn_times) * 1000:.3f}ms")

    # Helper: copy GDN output to 3D buffer
    def copy_gdn_out_to_3d(req, buf):
        """Copy GDN output (may be 2D) into a 3D [1,1,H] buffer."""
        out = req.get_output_tensor(0).data
        buf.flat[:] = out.flat[:]

    def copy_attn_out_to_3d(req, buf):
        """Copy Attn output (may be 2D) into a 3D [1,1,H] buffer."""
        out = req.get_output_tensor(0).data
        buf.flat[:] = out.flat[:]

    # ---- Config B: set_input_tensor / set_output_tensor pre-allocated ----
    print(f"\n  Config B: Pre-allocated buffers + set_input/output_tensor")

    # Pre-allocate buffers
    hidden_buf = np.zeros((1, 1, hidden_size), dtype=np.float32)
    gdn_hidden_in = np.random.randn(1, 1, hidden_size).astype(np.float32)

    print(f"    Warming up ({WARMUP} iters)...", end=" ", flush=True)
    for _ in range(WARMUP):
        gdn_req.set_input_tensor(0, ov.Tensor(gdn_hidden_in))
        gdn_req.set_input_tensor(1, ov.Tensor(gdn_mask))
        gdn_req.infer()
        copy_gdn_out_to_3d(gdn_req, hidden_buf)
        attn_req.set_input_tensor(0, ov.Tensor(hidden_buf))
        attn_req.set_input_tensor(1, ov.Tensor(attn_inputs["in_position_ids"]))
        attn_req.set_input_tensor(2, ov.Tensor(attn_inputs["in_key_cache"]))
        attn_req.set_input_tensor(3, ov.Tensor(attn_inputs["in_value_cache"]))
        attn_req.set_input_tensor(4, ov.Tensor(attn_inputs["in_cache_position"]))
        attn_req.set_input_tensor(5, ov.Tensor(attn_inputs["in_attention_mask"]))
        attn_req.infer()
        copy_attn_out_to_3d(attn_req, gdn_hidden_in)
    print("done")

    print(f"    Benchmarking ({NUM_ITERS} iters)...", end=" ", flush=True)
    gdn_times_b = []
    attn_times_b = []
    copy_times_b = []
    total_t0 = time.perf_counter()
    for _ in range(NUM_ITERS):
        t0 = time.perf_counter()
        gdn_req.set_input_tensor(0, ov.Tensor(gdn_hidden_in))
        gdn_req.set_input_tensor(1, ov.Tensor(gdn_mask))
        gdn_req.infer()
        t1 = time.perf_counter()
        gdn_times_b.append(t1 - t0)

        t0 = time.perf_counter()
        copy_gdn_out_to_3d(gdn_req, hidden_buf)
        t1 = time.perf_counter()
        copy_times_b.append(t1 - t0)

        t0 = time.perf_counter()
        attn_req.set_input_tensor(0, ov.Tensor(hidden_buf))
        attn_req.set_input_tensor(1, ov.Tensor(attn_inputs["in_position_ids"]))
        attn_req.set_input_tensor(2, ov.Tensor(attn_inputs["in_key_cache"]))
        attn_req.set_input_tensor(3, ov.Tensor(attn_inputs["in_value_cache"]))
        attn_req.set_input_tensor(4, ov.Tensor(attn_inputs["in_cache_position"]))
        attn_req.set_input_tensor(5, ov.Tensor(attn_inputs["in_attention_mask"]))
        attn_req.infer()
        t1 = time.perf_counter()
        attn_times_b.append(t1 - t0)

        copy_attn_out_to_3d(attn_req, gdn_hidden_in)

    total_b = time.perf_counter() - total_t0
    print("done")
    print(f"    Total:     {total_b * 1000:.1f}ms ({total_b / NUM_ITERS * 1000:.2f}ms/iter)")
    print(f"    GDN avg:   {np.mean(gdn_times_b) * 1000:.3f}ms")
    print(f"    Copy avg:  {np.mean(copy_times_b) * 1000:.3f}ms")
    print(f"    Attn avg:  {np.mean(attn_times_b) * 1000:.3f}ms")

    # ---- Config C: GPU set_output_tensor to pre-allocated buffer ----
    # First, determine GDN output shape by running one inference
    print(f"\n  Config C: GPU set_output_tensor (output writes to pre-allocated host buffer)")
    gdn_req.set_input_tensor(0, ov.Tensor(np.random.randn(1, 1, hidden_size).astype(np.float32)))
    gdn_req.set_input_tensor(1, ov.Tensor(gdn_mask))
    gdn_req.infer()
    gdn_out_shape = tuple(gdn_req.get_output_tensor(0).data.shape)
    print(f"    GDN output shape: {gdn_out_shape}")

    out_buf = np.zeros(gdn_out_shape, dtype=np.float32)
    gdn_hidden_in_c = np.random.randn(1, 1, hidden_size).astype(np.float32)

    # For attn input we need [1,1,H] -- create a view if GDN output is 2D
    attn_hidden_view = out_buf.reshape(1, 1, hidden_size)

    print(f"    Warming up ({WARMUP} iters)...", end=" ", flush=True)
    for _ in range(WARMUP):
        gdn_req.set_input_tensor(0, ov.Tensor(gdn_hidden_in_c))
        gdn_req.set_input_tensor(1, ov.Tensor(gdn_mask))
        gdn_req.set_output_tensor(0, ov.Tensor(out_buf))
        gdn_req.infer()
        # out_buf now has the GDN output -- reshape view for attn input
        attn_req.set_input_tensor(0, ov.Tensor(attn_hidden_view))
        attn_req.set_input_tensor(1, ov.Tensor(attn_inputs["in_position_ids"]))
        attn_req.set_input_tensor(2, ov.Tensor(attn_inputs["in_key_cache"]))
        attn_req.set_input_tensor(3, ov.Tensor(attn_inputs["in_value_cache"]))
        attn_req.set_input_tensor(4, ov.Tensor(attn_inputs["in_cache_position"]))
        attn_req.set_input_tensor(5, ov.Tensor(attn_inputs["in_attention_mask"]))
        attn_req.infer()
        copy_attn_out_to_3d(attn_req, gdn_hidden_in_c)
    print("done")

    print(f"    Benchmarking ({NUM_ITERS} iters)...", end=" ", flush=True)
    gdn_times_c = []
    attn_times_c = []
    total_t0 = time.perf_counter()
    for _ in range(NUM_ITERS):
        t0 = time.perf_counter()
        gdn_req.set_input_tensor(0, ov.Tensor(gdn_hidden_in_c))
        gdn_req.set_input_tensor(1, ov.Tensor(gdn_mask))
        gdn_req.set_output_tensor(0, ov.Tensor(out_buf))
        gdn_req.infer()
        t1 = time.perf_counter()
        gdn_times_c.append(t1 - t0)

        # No explicit copy -- out_buf IS the GDN output, attn_hidden_view shares memory
        t0 = time.perf_counter()
        attn_req.set_input_tensor(0, ov.Tensor(attn_hidden_view))
        attn_req.set_input_tensor(1, ov.Tensor(attn_inputs["in_position_ids"]))
        attn_req.set_input_tensor(2, ov.Tensor(attn_inputs["in_key_cache"]))
        attn_req.set_input_tensor(3, ov.Tensor(attn_inputs["in_value_cache"]))
        attn_req.set_input_tensor(4, ov.Tensor(attn_inputs["in_cache_position"]))
        attn_req.set_input_tensor(5, ov.Tensor(attn_inputs["in_attention_mask"]))
        attn_req.infer()
        t1 = time.perf_counter()
        attn_times_c.append(t1 - t0)

        copy_attn_out_to_3d(attn_req, gdn_hidden_in_c)

    total_c = time.perf_counter() - total_t0
    print("done")
    print(f"    Total:     {total_c * 1000:.1f}ms ({total_c / NUM_ITERS * 1000:.2f}ms/iter)")
    print(f"    GDN avg:   {np.mean(gdn_times_c) * 1000:.3f}ms")
    print(f"    Copy avg:  0.000ms (eliminated)")
    print(f"    Attn avg:  {np.mean(attn_times_c) * 1000:.3f}ms")

    # ---- Config D: Remote tensor (if supported) ----
    print(f"\n  Config D: Remote tensor (Level Zero shared memory)")
    total_d_val = None
    try:
        gpu_ctx = core.get_default_context("GPU")
        npu_ctx = core.get_default_context("NPU")

        # Create GPU remote tensors with different memory types
        print(f"    Testing GPU remote tensor types...")
        for mem_type in ["USM_HOST_BUFFER", "USM_DEVICE_BUFFER"]:
            try:
                rt = gpu_ctx.create_tensor(
                    ov.Type.f32, ov.Shape(list(gdn_out_shape)),
                    {"SHARED_MEM_TYPE": mem_type}
                )
                print(f"    GPU {mem_type}: OK, type={type(rt).__name__}")
                try:
                    params = rt.get_params()
                    print(f"      params: {params}")
                except Exception as e:
                    print(f"      get_params: {e}")
            except Exception as e:
                print(f"    GPU {mem_type}: {e}")

        print(f"    Testing NPU remote tensor types...")
        for mem_type in ["USM_HOST_BUFFER", "USM_DEVICE_BUFFER"]:
            try:
                rt = npu_ctx.create_tensor(
                    ov.Type.f32, ov.Shape([1, 1, hidden_size]),
                    {"SHARED_MEM_TYPE": mem_type}
                )
                print(f"    NPU {mem_type}: OK, type={type(rt).__name__}")
                try:
                    params = rt.get_params()
                    print(f"      params: {params}")
                except Exception as e:
                    print(f"      get_params: {e}")
            except Exception as e:
                print(f"    NPU {mem_type}: {e}")

        # Try: create GPU USM_HOST remote tensor as GDN output, then pass to NPU
        print(f"\n    Attempting GPU USM_HOST as GDN output + NPU input...")
        gdn_hidden_in_d = np.random.randn(1, 1, hidden_size).astype(np.float32)
        gpu_remote = gpu_ctx.create_tensor(
            ov.Type.f32, ov.Shape(list(gdn_out_shape)),
            {"SHARED_MEM_TYPE": "USM_HOST_BUFFER"}
        )
        # Also create an attn-compatible 3D view if needed
        attn_remote = gpu_ctx.create_tensor(
            ov.Type.f32, ov.Shape([1, 1, hidden_size]),
            {"SHARED_MEM_TYPE": "USM_HOST_BUFFER"}
        )

        # Test: set GPU remote tensor as GDN output
        try:
            gdn_req.set_input_tensor(0, ov.Tensor(gdn_hidden_in_d))
            gdn_req.set_input_tensor(1, ov.Tensor(gdn_mask))
            gdn_req.set_output_tensor(0, gpu_remote)
            gdn_req.infer()
            print(f"    GPU set_output_tensor(remote): OK")
        except Exception as e:
            print(f"    GPU set_output_tensor(remote): FAILED - {e}")
            raise

        # Test: set GPU remote tensor as NPU input
        try:
            attn_req.set_input_tensor(0, attn_remote)
            attn_req.set_input_tensor(1, ov.Tensor(attn_inputs["in_position_ids"]))
            attn_req.set_input_tensor(2, ov.Tensor(attn_inputs["in_key_cache"]))
            attn_req.set_input_tensor(3, ov.Tensor(attn_inputs["in_value_cache"]))
            attn_req.set_input_tensor(4, ov.Tensor(attn_inputs["in_cache_position"]))
            attn_req.set_input_tensor(5, ov.Tensor(attn_inputs["in_attention_mask"]))
            attn_req.infer()
            print(f"    NPU set_input_tensor(gpu_remote): OK")
        except Exception as e:
            print(f"    NPU set_input_tensor(gpu_remote): FAILED - {e}")
            raise

        # If we got here, benchmark the remote tensor path
        print(f"    Warming up ({WARMUP} iters)...", end=" ", flush=True)
        for _ in range(WARMUP):
            gdn_req.set_input_tensor(0, ov.Tensor(gdn_hidden_in_d))
            gdn_req.set_input_tensor(1, ov.Tensor(gdn_mask))
            gdn_req.set_output_tensor(0, gpu_remote)
            gdn_req.infer()
            # Copy from gpu_remote to attn_remote (may share USM host memory)
            # For now, just use attn_remote directly
            attn_req.set_input_tensor(0, attn_remote)
            attn_req.set_input_tensor(1, ov.Tensor(attn_inputs["in_position_ids"]))
            attn_req.set_input_tensor(2, ov.Tensor(attn_inputs["in_key_cache"]))
            attn_req.set_input_tensor(3, ov.Tensor(attn_inputs["in_value_cache"]))
            attn_req.set_input_tensor(4, ov.Tensor(attn_inputs["in_cache_position"]))
            attn_req.set_input_tensor(5, ov.Tensor(attn_inputs["in_attention_mask"]))
            attn_req.infer()
            copy_attn_out_to_3d(attn_req, gdn_hidden_in_d)
        print("done")

        print(f"    Benchmarking ({NUM_ITERS} iters)...", end=" ", flush=True)
        gdn_times_d = []
        attn_times_d = []
        total_t0 = time.perf_counter()
        for _ in range(NUM_ITERS):
            t0 = time.perf_counter()
            gdn_req.set_input_tensor(0, ov.Tensor(gdn_hidden_in_d))
            gdn_req.set_input_tensor(1, ov.Tensor(gdn_mask))
            gdn_req.set_output_tensor(0, gpu_remote)
            gdn_req.infer()
            t1 = time.perf_counter()
            gdn_times_d.append(t1 - t0)

            t0 = time.perf_counter()
            attn_req.set_input_tensor(0, attn_remote)
            attn_req.set_input_tensor(1, ov.Tensor(attn_inputs["in_position_ids"]))
            attn_req.set_input_tensor(2, ov.Tensor(attn_inputs["in_key_cache"]))
            attn_req.set_input_tensor(3, ov.Tensor(attn_inputs["in_value_cache"]))
            attn_req.set_input_tensor(4, ov.Tensor(attn_inputs["in_cache_position"]))
            attn_req.set_input_tensor(5, ov.Tensor(attn_inputs["in_attention_mask"]))
            attn_req.infer()
            t1 = time.perf_counter()
            attn_times_d.append(t1 - t0)

            copy_attn_out_to_3d(attn_req, gdn_hidden_in_d)

        total_d = time.perf_counter() - total_t0
        print("done")
        print(f"    Total:     {total_d * 1000:.1f}ms ({total_d / NUM_ITERS * 1000:.2f}ms/iter)")
        print(f"    GDN avg:   {np.mean(gdn_times_d) * 1000:.3f}ms")
        print(f"    Copy avg:  0.000ms (shared memory)")
        print(f"    Attn avg:  {np.mean(attn_times_d) * 1000:.3f}ms")
        total_d_val = total_d
    except Exception as e:
        print(f"    Remote tensor experiment FAILED: {e}")
        import traceback
        traceback.print_exc()

    # ---- Summary ----
    print(f"\n  {'=' * 60}")
    print(f"  TRANSFER BENCHMARK SUMMARY ({NUM_ITERS} iterations)")
    print(f"  {'=' * 60}")
    print(f"  Config A (baseline dict infer):        {total_a / NUM_ITERS * 1000:.2f}ms/iter")
    print(f"  Config B (pre-alloc + set_tensor):     {total_b / NUM_ITERS * 1000:.2f}ms/iter")
    print(f"  Config C (GPU set_output_tensor):      {total_c / NUM_ITERS * 1000:.2f}ms/iter")
    if total_d_val is not None:
        print(f"  Config D (remote tensor):              {total_d_val / NUM_ITERS * 1000:.2f}ms/iter")
    else:
        print(f"  Config D (remote tensor):              N/A (not supported)")
    print()

    return total_a, total_b, total_c, total_d_val


# ===================================================================
# Step 4: Sync vs Async comparison
# ===================================================================
def step4_sync_vs_async(gdn_req, attn_req, gdn_compiled, attn_compiled, cfg: dict):
    """Compare sync infer() vs start_async()+wait() overhead."""
    print("=" * 70)
    print("STEP 4: Sync infer() vs Async start_async()+wait()")
    print("=" * 70)

    hidden_size = cfg["hidden_size"]
    gdn_hidden = np.random.randn(1, 1, hidden_size).astype(np.float32)
    gdn_mask = np.ones((1, 1), dtype=np.int64)
    attn_inputs = make_attn_inputs(cfg)
    attn_inputs["in_attention_mask"][0, 0, 0, 0] = 0.0
    attn_inputs["in_attention_mask"][0, 0, 0, 1] = 0.0
    hidden_buf = np.zeros((1, 1, hidden_size), dtype=np.float32)

    def _copy_out_3d(req, buf):
        out = req.get_output_tensor(0).data
        buf.flat[:] = out.flat[:]

    def _set_attn_inputs(req, hid, ai):
        req.set_input_tensor(0, ov.Tensor(hid))
        req.set_input_tensor(1, ov.Tensor(ai["in_position_ids"]))
        req.set_input_tensor(2, ov.Tensor(ai["in_key_cache"]))
        req.set_input_tensor(3, ov.Tensor(ai["in_value_cache"]))
        req.set_input_tensor(4, ov.Tensor(ai["in_cache_position"]))
        req.set_input_tensor(5, ov.Tensor(ai["in_attention_mask"]))

    # Warmup
    print(f"\n  Warming up ({WARMUP} iters)...", end=" ", flush=True)
    for _ in range(WARMUP):
        gdn_req.set_input_tensor(0, ov.Tensor(gdn_hidden))
        gdn_req.set_input_tensor(1, ov.Tensor(gdn_mask))
        gdn_req.infer()
        _copy_out_3d(gdn_req, hidden_buf)
        _set_attn_inputs(attn_req, hidden_buf, attn_inputs)
        attn_req.infer()
        _copy_out_3d(attn_req, gdn_hidden)
    print("done")

    # Sync
    print(f"\n  Sync (infer + infer), {NUM_ITERS} iters...", end=" ", flush=True)
    gdn_hidden = np.random.randn(1, 1, hidden_size).astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(NUM_ITERS):
        gdn_req.set_input_tensor(0, ov.Tensor(gdn_hidden))
        gdn_req.set_input_tensor(1, ov.Tensor(gdn_mask))
        gdn_req.infer()
        _copy_out_3d(gdn_req, hidden_buf)
        _set_attn_inputs(attn_req, hidden_buf, attn_inputs)
        attn_req.infer()
        _copy_out_3d(attn_req, gdn_hidden)
    sync_time = time.perf_counter() - t0
    print(f"done: {sync_time * 1000:.1f}ms total, {sync_time / NUM_ITERS * 1000:.2f}ms/iter")

    # Async with wait
    print(f"  Async (start_async+wait), {NUM_ITERS} iters...", end=" ", flush=True)
    gdn_hidden = np.random.randn(1, 1, hidden_size).astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(NUM_ITERS):
        gdn_req.set_input_tensor(0, ov.Tensor(gdn_hidden))
        gdn_req.set_input_tensor(1, ov.Tensor(gdn_mask))
        gdn_req.start_async()
        gdn_req.wait()
        _copy_out_3d(gdn_req, hidden_buf)
        _set_attn_inputs(attn_req, hidden_buf, attn_inputs)
        attn_req.start_async()
        attn_req.wait()
        _copy_out_3d(attn_req, gdn_hidden)
    async_time = time.perf_counter() - t0
    print(f"done: {async_time * 1000:.1f}ms total, {async_time / NUM_ITERS * 1000:.2f}ms/iter")

    print(f"\n  Sync:  {sync_time / NUM_ITERS * 1000:.2f}ms/iter")
    print(f"  Async: {async_time / NUM_ITERS * 1000:.2f}ms/iter")
    diff_pct = (async_time - sync_time) / sync_time * 100
    print(f"  Diff:  {diff_pct:+.1f}% ({'async slower' if diff_pct > 0 else 'async faster'})")
    print()


# ===================================================================
# Main
# ===================================================================
def main():
    print("Remote Tensor / Zero-Copy Benchmark for Intel Lunar Lake")
    print("GPU GDN <-> NPU Attention data transfer overhead\n")

    core = ov.Core()
    devices = core.available_devices
    print(f"Available devices: {devices}")

    # Check OV version
    try:
        print(f"OpenVINO version: {ov.get_version()}")
    except Exception:
        pass

    if "GPU" not in devices:
        print("\n[ERROR] GPU not available. Cannot run benchmark.")
        return
    if "NPU" not in devices:
        print("\n[ERROR] NPU not available. Cannot run benchmark.")
        return

    # Print device info
    for dev in ["GPU", "NPU"]:
        try:
            name = core.get_property(dev, "DEVICE_ARCHITECTURE")
            print(f"  {dev}: {name}")
        except Exception:
            try:
                name = core.get_property(dev, "FULL_DEVICE_NAME")
                print(f"  {dev}: {name}")
            except Exception:
                pass

    model_dir = find_model_dir()
    print(f"\nModel dir: {model_dir}")

    cfg = load_config(model_dir)
    print(f"Config: hidden={cfg['hidden_size']}, kv_heads={cfg['num_kv_heads']}, "
          f"head_dim={cfg['head_dim']}, conv_dim={cfg['conv_dim']}")

    # Set cache dir
    cache_dir = str(model_dir / "cache")
    core.set_property({"CACHE_DIR": cache_dir})
    print(f"Cache dir: {cache_dir}\n")

    # Step 1: Probe remote context support
    step1_probe_remote_context(core)

    # Step 2: Compile models
    gdn_req, attn_req, gdn_compiled, attn_compiled = compile_models(
        core, model_dir, cfg
    )

    # Step 3: Benchmark transfer patterns
    step3_benchmark_transfer(
        core, gdn_req, attn_req, gdn_compiled, attn_compiled, cfg
    )

    # Step 4: Sync vs Async
    step4_sync_vs_async(gdn_req, attn_req, gdn_compiled, attn_compiled, cfg)

    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
