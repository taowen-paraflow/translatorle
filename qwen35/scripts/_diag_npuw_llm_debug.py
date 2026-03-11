"""Debug NPUW_LLM: add OPENVINO_LOG_LEVEL to see what reshape_to_static does."""
import openvino as ov
import os
import time

# Enable debug logging for NPUW
os.environ["OPENVINO_LOG_LEVEL"] = "0"  # 0=TRACE

core = ov.Core()
model = core.read_model("models/qwen35/Qwen3.5-0.8B-npu/openvino_model.xml")

config = {
    "NPU_USE_NPUW": "YES",
    "NPUW_LLM": "YES",
    "NPUW_LLM_BATCH_DIM": "0",
    "NPUW_LLM_SEQ_LEN_DIM": "2",
    "NPUW_LLM_MAX_PROMPT_LEN": "1",
    "NPUW_LLM_MIN_RESPONSE_LEN": "2048",
    "NPUW_FOLD": "NO",
    "NPUW_LLM_PREFILL_HINT": "STATIC",
    "LOG_LEVEL": "LOG_DEBUG",
}

print("Compiling with NPUW_LLM=YES + debug logging...")
t0 = time.time()
try:
    compiled = core.compile_model(model, "NPU", config)
    print(f"SUCCESS in {time.time()-t0:.1f}s")
except Exception as e:
    print(f"FAILED in {time.time()-t0:.1f}s")
    err = str(e)
    # Print just the first few lines of the error
    for line in err.split('\n')[:20]:
        print(f"  {line}")
