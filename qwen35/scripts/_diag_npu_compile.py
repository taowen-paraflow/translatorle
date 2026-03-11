"""Diagnostic: try compiling NPU model with different configs."""
import sys
import time
import openvino as ov

core = ov.Core()
model_path = "models/qwen35/Qwen3.5-0.8B-npu/openvino_model.xml"
model = core.read_model(model_path)

# Try 1: Minimal NPUW config (no NPUW_LLM)
config = {
    "NPU_USE_NPUW": "YES",
    "NPUW_FOLD": "NO",
}

print("=== Attempt 1: NPUW without NPUW_LLM ===")
try:
    t0 = time.time()
    compiled = core.compile_model(model, "NPU", config)
    print(f"  SUCCESS! Compiled in {time.time()-t0:.1f}s")
    del compiled
except Exception as e:
    print(f"  FAILED: {e}")

# Try 2: With NPUW_LLM but no SEQ_LEN_DIM
print("\n=== Attempt 2: NPUW_LLM=YES, no SEQ_LEN_DIM ===")
config2 = {
    "NPU_USE_NPUW": "YES",
    "NPUW_LLM": "YES",
    "NPUW_LLM_MAX_PROMPT_LEN": "1",
    "NPUW_FOLD": "NO",
}
try:
    model2 = core.read_model(model_path)
    t0 = time.time()
    compiled = core.compile_model(model2, "NPU", config2)
    print(f"  SUCCESS! Compiled in {time.time()-t0:.1f}s")
    del compiled
except Exception as e:
    print(f"  FAILED: {e}")

# Try 3: Direct NPU without NPUW
print("\n=== Attempt 3: Direct NPU (no NPUW) ===")
try:
    model3 = core.read_model(model_path)
    t0 = time.time()
    compiled = core.compile_model(model3, "NPU", {})
    print(f"  SUCCESS! Compiled in {time.time()-t0:.1f}s")
    del compiled
except Exception as e:
    print(f"  FAILED: {e}")
