"""Diagnostic script to profile NPUW LLM execution on NPU.

Sets environment variables BEFORE importing openvino so the C++ plugin picks them up.

Usage:
    uv run python hymt/scripts/diag_npu.py
"""

import os
import sys

# Must set BEFORE importing openvino
os.environ["OPENVINO_NPUW_PROF"] = "1"
os.environ["OPENVINO_NPUW_LOG_LEVEL"] = "INFO"
os.environ["OPENVINO_LOG_LEVEL"] = "INFO"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import time
import openvino_genai as ov_genai
from hymt.config import MT_MODEL_DIR, MT_CACHE_DIR, NPU_CONFIG

print(f"openvino_genai version: {ov_genai.__version__}")
print(f"Model dir: {MT_MODEL_DIR}")
print(f"NPU_CONFIG: {NPU_CONFIG}")
print()

# Build config
config = {**NPU_CONFIG, "CACHE_DIR": MT_CACHE_DIR}
print(f"Full config passed to LLMPipeline: {config}")
print()

# Init pipeline
print("Compiling model on NPU...")
t0 = time.perf_counter()
pipe = ov_genai.LLMPipeline(MT_MODEL_DIR, "NPU", **config)
print(f"Compilation took: {time.perf_counter() - t0:.2f}s")
print()

# Single short inference to trigger profiling
print("Running inference: 'Hello' -> Chinese")
prompt = "将以下文本翻译为中文，注意只需要输出翻译后的结果，不要额外解释：\n\nHello"
t0 = time.perf_counter()
result = pipe.generate(prompt, max_new_tokens=32)
t1 = time.perf_counter()
print(f"Result: {result}")
print(f"Time: {t1 - t0:.2f}s")
print()

# Second inference
print("Running inference: 'The quick brown fox' -> Chinese")
prompt2 = "将以下文本翻译为中文，注意只需要输出翻译后的结果，不要额外解释：\n\nThe quick brown fox jumps over the lazy dog."
t0 = time.perf_counter()
result2 = pipe.generate(prompt2, max_new_tokens=64)
t1 = time.perf_counter()
print(f"Result: {result2}")
print(f"Time: {t1 - t0:.2f}s")
print()

# Destroy pipeline to trigger NPUW profiling report
print("Destroying pipeline (NPUW profiling report should follow)...")
print("=" * 80)
sys.stdout.flush()
sys.stderr.flush()
del pipe
sys.stdout.flush()
sys.stderr.flush()
print("=" * 80)
print("Done.")
