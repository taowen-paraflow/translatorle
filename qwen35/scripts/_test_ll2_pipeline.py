#!/usr/bin/env python3
"""LowLatency2 pipeline: non-stateful Loop IR → Loop-free stateful IR for NPU.

Clean pipeline starting from a NON-STATEFUL IR (no ReadValue/Assign):
  1. Load non-stateful IR (18 TensorIterator/Loop nodes, explicit I/O)
  2. Reshape to static shapes (batch=1, seq_len=1)
  3. ConstantFolding (separate Manager) → trip_count becomes Constant(1)
  4. Fix timestep Parameters → Constant(0)
  5. LowLatency2 (separate Manager) → unrolls Loops, creates ReadValue/Assign
     for recurrent states (all with static variable_shape!)
  6. MakeStateful for GDN conv states
  7. Post-fix: add beam_idx, rename KV to HF standard naming
  8. Save + verify on CPU
  9. NPU compile + inference test

Key insight: Starting from non-stateful IR means ALL ReadValue nodes created
by LL2 have static variable_shape (since reshape was done first). This avoids
the "dynamic variable_shape" problem that blocked the old approach.

Usage (two steps):
    # Step 1: Export non-stateful Loop IR (needs transformers 5.x)
    uv run --project qwen35 python -m qwen35.scripts.prepare_qwen35_models --ll2

    # Step 2: Run LL2 pipeline (main project venv)
    uv run python -m qwen35.scripts._test_ll2_pipeline
"""
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import openvino as ov
import openvino.passes as ov_passes
from openvino import opset13

print("OpenVINO version:", ov.__version__)
core = ov.Core()
print("Devices:", core.available_devices)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODELS_DIR = Path("models/qwen35")
LOOP_IR_DIR = MODELS_DIR / "Qwen3.5-0.8B-loop"  # Non-stateful IR with Loops
STATEFUL_DIR = MODELS_DIR / "Qwen3.5-0.8B-ov"   # Existing stateful model (fallback for assets)
OUT_DIR = MODELS_DIR / "Qwen3.5-0.8B-ll2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# KV cache static length for NPU
KV_CACHE_LEN = 2048


def count_ops(m):
    ops = m.get_ordered_ops()
    loops = sum(1 for op in ops if op.get_type_name() in ("Loop", "TensorIterator"))
    rvs = sum(1 for op in ops if op.get_type_name() == "ReadValue")
    assigns = sum(1 for op in ops if op.get_type_name() == "Assign")
    return loops, rvs, assigns


# ============================================================
# Step 1: Load non-stateful Loop IR
# ============================================================
ir_path = LOOP_IR_DIR / "openvino_model.xml"
if not ir_path.exists():
    print(f"\nERROR: Non-stateful Loop IR not found at {ir_path}")
    print("Run first:")
    print("  uv run --project qwen35 python -m qwen35.scripts.prepare_qwen35_models --hf-model Qwen/Qwen3.5-0.8B --ll2")
    sys.exit(1)

print(f"\n[1] Load non-stateful IR from {ir_path}")
model = core.read_model(str(ir_path))
l, r, a = count_ops(model)
print(f"  Loops={l}, RV={r}, Assign={a}, Inputs={len(model.inputs)}, Outputs={len(model.outputs)}")
assert l == 18, f"Expected 18 Loops, got {l}"
assert r == 0, f"Expected 0 ReadValue (non-stateful), got {r}"

# Print all input names for reference
print("  Inputs:")
for inp in model.inputs:
    print(f"    {inp.get_any_name()}: {inp.get_partial_shape()}")
print(f"  Outputs: {len(model.outputs)} (logits + 48 cache tensors)")


# ============================================================
# Step 2: Reshape to static shapes
# ============================================================
print(f"\n[2] Reshape to static (batch=1, seq=1, kv_cache={KV_CACHE_LEN})")
new_shapes = {}
for inp in model.inputs:
    name = inp.get_any_name()
    ps = inp.get_partial_shape()
    if "inputs_embeds" in name:
        new_shapes[name] = [1, 1, ps[2].get_length()]
    elif "attention_mask" in name:
        new_shapes[name] = [1, KV_CACHE_LEN + 1]
    elif "position_ids" in name:
        new_shapes[name] = [3, 1, 1]
    elif ".conv" in name:
        new_shapes[name] = [1] + [d.get_length() for d in ps[1:]]
    elif ".recurrent" in name:
        new_shapes[name] = [1] + [d.get_length() for d in ps[1:]]
    elif ".key" in name or ".value" in name:
        new_shapes[name] = [1, ps[1].get_length(), KV_CACHE_LEN, ps[3].get_length()]
    else:
        new_shapes[name] = [1] + [
            d.get_length() if d.is_static else 1 for d in ps[1:]
        ]
model.reshape(new_shapes)
print("  OK (all static)")


# ============================================================
# Step 3: ConstantFolding (separate Manager!)
# ============================================================
print("\n[3] ConstantFolding")
mgr = ov_passes.Manager()
mgr.register_pass(ov_passes.ConstantFolding())
mgr.run_passes(model)

# Verify trip_count is Constant(1)
for op in model.get_ordered_ops():
    if op.get_type_name() in ("Loop", "TensorIterator"):
        trip_src = op.input(0).get_source_output().get_node()
        if trip_src.get_type_name() == "Constant":
            trip_val = trip_src.get_data()
            print(f"  trip_count = Constant({int(trip_val)})")
            assert int(trip_val) == 1
        else:
            print(f"  WARNING: trip_count is {trip_src.get_type_name()}, not Constant")
        break


# ============================================================
# Step 4: Fix timestep Parameters → Constant(0)
# ============================================================
print("\n[4] Fix timestep → Constant(0)")
replaced = 0
for op in model.get_ordered_ops():
    if op.get_type_name() not in ("Loop", "TensorIterator"):
        continue
    body = op.get_function()
    ts = body.get_parameters()[0]
    if (ts.get_output_element_type(0) == ov.Type.i32 and
            ts.get_output_partial_shape(0).rank.get_length() == 0):
        c0 = opset13.constant(np.int32(0))
        for consumer in list(ts.output(0).get_target_inputs()):
            consumer.replace_source_output(c0.output(0))
        replaced += 1
print(f"  Replaced {replaced} timestep params")
model.validate_nodes_and_infer_types()


# ============================================================
# Step 5: LowLatency2 (separate Manager!)
# ============================================================
print("\n[5] LowLatency2")
mgr2 = ov_passes.Manager()
mgr2.register_pass(ov_passes.LowLatency2(False))  # False = keep explicit input as initializer
mgr2.run_passes(model)

l, r, a = count_ops(model)
print(f"  After LL2: Loops={l}, RV={r}, Assign={a}")
assert l == 0, f"Expected 0 Loops, got {l}"
print("  All 18 TensorIterator nodes eliminated!")

# Check that all ReadValue have static shapes
dynamic_rv = 0
for op in model.get_ordered_ops():
    if op.get_type_name() == "ReadValue":
        ps = op.get_output_partial_shape(0)
        if any(not d.is_static for d in ps):
            dynamic_rv += 1
            print(f"  WARNING: Dynamic ReadValue: {ps}")
print(f"  Dynamic ReadValue count: {dynamic_rv} (should be 0)")


# ============================================================
# Step 6: MakeStateful for GDN conv states
# ============================================================
print("\n[6] Skip MakeStateful (keep all states as explicit I/O for Direct NPU)")
print("  Direct NPU doesn't persist ReadValue/Assign between infer() calls,")
print("  so all states must be managed via explicit inputs/outputs.")
l, r, a = count_ops(model)
print(f"  Current: Loops={l}, RV={r}, Assign={a}")


# ============================================================
# Step 7: Post-fix — beam_idx + KV renaming
# ============================================================
print("\n[7] Post-fix: beam_idx + KV rename")

# 7a. Add beam_idx
has_beam = any("beam_idx" in inp.get_any_name() for inp in model.inputs)
if not has_beam:
    beam = opset13.parameter([1], ov.Type.i32, name="beam_idx")
    beam.get_output_tensor(0).set_names({"beam_idx"})
    model.add_parameters([beam])
    print("  Added beam_idx")

# 7b. Rename KV cache to HF standard naming
renamed = 0
for port_list in (model.inputs, model.outputs):
    for port in list(port_list):
        name = port.get_any_name()
        new_name = name

        if "cache_params.past.key." in name:
            idx = name.split(".")[-1]
            new_name = f"past_key_values.{idx}.key"
        elif "cache_params.past.value." in name:
            idx = name.split(".")[-1]
            new_name = f"past_key_values.{idx}.value"
        elif "cache_params.present.key." in name:
            idx = name.split(".")[-1]
            new_name = f"present.{idx}.key"
        elif "cache_params.present.value." in name:
            idx = name.split(".")[-1]
            new_name = f"present.{idx}.value"

        if new_name != name:
            port.get_node().set_friendly_name(new_name)
            port.get_tensor().set_names({new_name})
            renamed += 1
            print(f"  {name} → {new_name}")

print(f"  Renamed {renamed} KV ports")


# ============================================================
# Step 8: Final cleanup + save
# ============================================================
print("\n[8] Final cleanup")
mgr3 = ov_passes.Manager()
mgr3.register_pass(ov_passes.ConstantFolding())
mgr3.run_passes(model)

l, r, a = count_ops(model)
print(f"  FINAL: Loops={l}, RV={r}, Assign={a}")
print(f"  Inputs={len(model.inputs)}, Outputs={len(model.outputs)}")

print("\n  Final inputs:")
for inp in model.inputs:
    print(f"    {inp.get_any_name()}: {inp.get_partial_shape()}")
print("  Final outputs:")
for out in model.outputs:
    print(f"    {out.get_any_name()}: {out.get_partial_shape()}")

xml_path = str(OUT_DIR / "openvino_model.xml")
print(f"\n  Saving to {xml_path}")
ov.save_model(model, xml_path, compress_to_fp16=True)

# Copy assets from loop IR dir (or stateful dir as fallback)
for filename in ["embed_tokens.npy", "config.json", "tokenizer.json",
                 "tokenizer_config.json", "special_tokens_map.json",
                 "generation_config.json"]:
    dst = OUT_DIR / filename
    if dst.exists():
        continue
    for src_dir in [LOOP_IR_DIR, STATEFUL_DIR]:
        src = src_dir / filename
        if src.exists():
            shutil.copy2(str(src), str(dst))
            break

print("  Saved model + assets")


# ============================================================
# Step 9: Reload verification
# ============================================================
print("\n[9] Reload verification")
try:
    model2 = core.read_model(xml_path)
    l2, r2, a2 = count_ops(model2)
    print(f"  Reloaded: Loops={l2}, RV={r2}, Assign={a2}")
    dynamic_rv = 0
    for op in model2.get_ordered_ops():
        if op.get_type_name() == "ReadValue":
            ps = op.get_output_partial_shape(0)
            if any(not d.is_static for d in ps):
                dynamic_rv += 1
    print(f"  Dynamic ReadValue after reload: {dynamic_rv} (should be 0)")
except Exception as e:
    print(f"  Reload FAILED: {e}")
    sys.exit(1)


# ============================================================
# Step 10: CPU inference verification
# ============================================================
print("\n[10] CPU inference verification")
from transformers import AutoTokenizer

tokenizer_dir = None
for d in [OUT_DIR, LOOP_IR_DIR, STATEFUL_DIR]:
    if (d / "tokenizer.json").exists():
        tokenizer_dir = d
        break

embed_path = None
for d in [OUT_DIR, LOOP_IR_DIR, STATEFUL_DIR]:
    if (d / "embed_tokens.npy").exists():
        embed_path = d / "embed_tokens.npy"
        break

if tokenizer_dir is None or embed_path is None:
    print("  Missing tokenizer or embed_tokens — skipping")
else:
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), trust_remote_code=True)
    embed_table = np.load(str(embed_path)).astype(np.float32)

    compiled = core.compile_model(model2, "CPU")
    request = compiled.create_infer_request()

    prompt = "Hello, what is your name?"
    input_ids = tokenizer(prompt, return_tensors="np")["input_ids"][0]
    print(f"  Prompt: '{prompt}' ({len(input_ids)} tokens)")

    # Initialize all explicit states with static shapes (all zeros = empty)
    gdn_states = {}  # conv + recurrent
    kv_states = {}
    for inp in model2.inputs:
        name = inp.get_any_name()
        ps = inp.get_partial_shape()
        shape = [d.get_length() for d in ps]
        if "past.recurrent" in name or "past.conv" in name:
            gdn_states[name] = np.zeros(shape, dtype=np.float32)
        elif "past_key_values" in name:
            kv_states[name] = np.zeros(shape, dtype=np.float32)

    generated = []
    past_length = 0
    all_tokens = list(input_ids)

    t0 = time.time()
    for step in range(len(input_ids) + 15):
        token_id = all_tokens[step] if step < len(all_tokens) else generated[-1]
        embeds = embed_table[token_id:token_id + 1].reshape(1, 1, -1)

        # Static attention mask: [1, KV_CACHE_LEN+1] with left-padding zeros
        mask = np.zeros((1, KV_CACHE_LEN + 1), dtype=np.int64)
        mask[0, -(past_length + 1):] = 1

        inp = {
            "inputs_embeds": embeds,
            "attention_mask": mask,
            "position_ids": np.full((3, 1, 1), past_length, dtype=np.int64),
        }
        # Add beam_idx if expected
        for model_inp in model2.inputs:
            if "beam_idx" in model_inp.get_any_name():
                inp["beam_idx"] = np.array([0], dtype=np.int32)
        inp.update(gdn_states)
        inp.update(kv_states)

        result = request.infer(inp)

        # Extract logits
        logits = None
        for out in compiled.outputs:
            if "logits" in out.get_any_name():
                logits = result[out]
                break
        if logits is None:
            logits = list(result.values())[0]

        logits_1d = logits[0, -1, :]
        has_nan = np.any(np.isnan(logits_1d))
        next_token = int(np.argmax(logits_1d))
        text = tokenizer.decode([next_token])

        if step < 3 or step == len(input_ids) - 1 or has_nan or step >= len(input_ids):
            print(f"  Step {step}: '{tokenizer.decode([token_id])}' → '{text}' "
                  f"NaN={has_nan} logits=[{float(np.min(logits_1d)):.1f}, {float(np.max(logits_1d)):.1f}]")
        if has_nan:
            print("  ERROR: NaN detected")
            break

        # Update GDN states: same shape, direct copy
        for gdn_name in list(gdn_states.keys()):
            present_name = gdn_name.replace(".past.", ".present.")
            for key in result:
                if present_name in str(key):
                    gdn_states[gdn_name] = np.array(result[key])
                    break

        # Update KV cache: slice present [1,2,2049,256] → past [1,2,2048,256]
        for kv_name in list(kv_states.keys()):
            idx_and_type = kv_name.replace("past_key_values.", "")
            present_name = f"present.{idx_and_type}"
            for key in result:
                if present_name in str(key):
                    present_kv = np.array(result[key])  # [1, 2, 2049, 256]
                    kv_states[kv_name] = present_kv[:, :, -KV_CACHE_LEN:, :]  # slide
                    break

        past_length += 1
        if step >= len(input_ids) - 1:
            generated.append(next_token)
            all_tokens.append(next_token)
        if next_token in (tokenizer.eos_token_id, 151643):
            break

    elapsed = time.time() - t0
    gen_text = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"\n  Generated: '{gen_text}'")
    if generated:
        print(f"  Speed: {len(generated) / elapsed:.1f} tok/s")
    if "qwen" in gen_text.lower() or "name" in gen_text.lower() or len(gen_text) > 10:
        print("  CPU verification: PASS")
    elif has_nan:
        print("  CPU verification: FAIL (NaN)")
    else:
        print("  CPU verification: check manually")


# ============================================================
# Step 11: NPU compile + inference
# ============================================================
if "NPU" not in core.available_devices:
    print("\nNPU not available — skipping NPU test")
    sys.exit(0)

print("\n[11] NPU compile")
# Direct NPU first (NPUW has boolean type bug, NPUW_LLM has reshape_to_static conflict)
for label, cfg in [
    ("Direct", {}),
    ("NPUW", {"NPU_USE_NPUW": "YES", "NPUW_FOLD": "NO"}),
    ("NPUW_LLM", {
        "NPU_USE_NPUW": "YES",
        "NPUW_LLM": "YES",
        "NPUW_LLM_MAX_PROMPT_LEN": "1",
        "NPUW_FOLD": "NO",
    }),
]:
    t0 = time.time()
    try:
        npu_compiled = core.compile_model(model2, "NPU", cfg)
        print(f"  {label}: OK in {time.time() - t0:.1f}s")
        # Also verify create_infer_request works
        npu_request = npu_compiled.create_infer_request()
        print(f"  {label}: infer_request OK")
        break
    except Exception as e:
        err = str(e).replace('\n', ' ')[:400]
        print(f"  {label}: FAILED ({time.time() - t0:.1f}s) - {err}")
else:
    print("  All NPU paths failed")
    sys.exit(1)

print("\n  NPU inference test:")

# Initialize all explicit states with static shapes
npu_gdn = {}  # conv + recurrent
npu_kv = {}
for inp in model2.inputs:
    name = inp.get_any_name()
    ps = inp.get_partial_shape()
    shape = [d.get_length() for d in ps]
    if "past.recurrent" in name or "past.conv" in name:
        npu_gdn[name] = np.zeros(shape, dtype=np.float32)
    elif "past_key_values" in name:
        npu_kv[name] = np.zeros(shape, dtype=np.float32)

npu_generated = []
past_len = 0
all_toks = list(input_ids)

t0 = time.time()
for step in range(len(input_ids) + 15):
    tid = all_toks[step] if step < len(all_toks) else npu_generated[-1]
    emb = embed_table[tid:tid + 1].reshape(1, 1, -1)

    # Static attention mask with left-padding
    mask = np.zeros((1, KV_CACHE_LEN + 1), dtype=np.int64)
    mask[0, -(past_len + 1):] = 1

    inp = {
        "inputs_embeds": emb,
        "attention_mask": mask,
        "position_ids": np.full((3, 1, 1), past_len, dtype=np.int64),
    }
    for mi in model2.inputs:
        if "beam_idx" in mi.get_any_name():
            inp["beam_idx"] = np.array([0], dtype=np.int32)
    inp.update(npu_gdn)
    inp.update(npu_kv)

    res = npu_request.infer(inp)

    logits = None
    for out in npu_compiled.outputs:
        if "logits" in out.get_any_name():
            logits = res[out]
            break
    if logits is None:
        logits = list(res.values())[0]

    logits_1d = logits[0, -1, :]
    has_nan = np.any(np.isnan(logits_1d))
    next_tok = int(np.argmax(logits_1d))
    txt = tokenizer.decode([next_tok])

    if step < 3 or step == len(input_ids) - 1 or has_nan or step >= len(input_ids):
        # Show top-3 tokens for comparison with CPU
        top3_idx = np.argsort(logits_1d)[-3:][::-1]
        top3_str = ", ".join(f"'{tokenizer.decode([i])}'{logits_1d[i]:.1f}" for i in top3_idx)
        print(f"  Step {step}: '{tokenizer.decode([tid])}' → '{txt}' NaN={has_nan} "
              f"logits=[{float(np.min(logits_1d)):.1f}, {float(np.max(logits_1d)):.1f}] top3=[{top3_str}]")
    if has_nan:
        break

    # Update GDN states: same shape, direct copy
    for gn in list(npu_gdn.keys()):
        pn = gn.replace(".past.", ".present.")
        for key in res:
            if pn in str(key):
                npu_gdn[gn] = np.array(res[key])
                break

    # Update KV cache with sliding window
    for kn in list(npu_kv.keys()):
        idx_type = kn.replace("past_key_values.", "")
        pn = f"present.{idx_type}"
        for key in res:
            if pn in str(key):
                present_kv = np.array(res[key])
                npu_kv[kn] = present_kv[:, :, -KV_CACHE_LEN:, :]
                break

    past_len += 1
    if step >= len(input_ids) - 1:
        npu_generated.append(next_tok)
        all_toks.append(next_tok)
    if next_tok in (tokenizer.eos_token_id, 151643):
        break

elapsed = time.time() - t0
npu_text = tokenizer.decode(npu_generated, skip_special_tokens=True)
print(f"\n  NPU Generated: '{npu_text}'")
if npu_generated:
    print(f"  NPU Speed: {len(npu_generated) / elapsed:.1f} tok/s")

print("\n" + "=" * 70)
print("Pipeline complete!")
print("=" * 70)
