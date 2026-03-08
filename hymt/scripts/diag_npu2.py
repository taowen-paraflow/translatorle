"""Diagnostic script v2 - use perf_metrics API and try to inspect compiled model.

Usage:
    uv run python hymt/scripts/diag_npu2.py
"""

import os
import sys

os.environ["OPENVINO_NPUW_PROF"] = "1"
os.environ["OPENVINO_NPUW_LOG_LEVEL"] = "VERBOSE"
os.environ["OPENVINO_LOG_LEVEL"] = "DEBUG"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import time
import openvino_genai as ov_genai
from hymt.config import MT_MODEL_DIR, MT_CACHE_DIR, NPU_CONFIG

print(f"openvino_genai version: {ov_genai.__version__}")

# Check what GenerationConfig looks like
gc = ov_genai.GenerationConfig()
gc.max_new_tokens = 32
print(f"\nGenerationConfig attributes:")
for attr in sorted(dir(gc)):
    if not attr.startswith('_'):
        try:
            val = getattr(gc, attr)
            if not callable(val):
                print(f"  {attr} = {val}")
        except:
            pass

# Build config
config = {**NPU_CONFIG, "CACHE_DIR": MT_CACHE_DIR}

print(f"\nCompiling on NPU with config: {config}")
t0 = time.perf_counter()
pipe = ov_genai.LLMPipeline(MT_MODEL_DIR, "NPU", **config)
print(f"Compilation: {time.perf_counter() - t0:.2f}s")

# Check LLMPipeline for useful methods
print(f"\nLLMPipeline methods/attrs:")
for attr in sorted(dir(pipe)):
    if not attr.startswith('_'):
        print(f"  {attr}")

# Try generate with GenerationConfig to get DecodedResults with perf_metrics
prompt = "将以下文本翻译为中文，注意只需要输出翻译后的结果，不要额外解释：\n\nThe quick brown fox jumps over the lazy dog."

print(f"\nInference with GenerationConfig...")
gc2 = ov_genai.GenerationConfig()
gc2.max_new_tokens = 64

t0 = time.perf_counter()
result = pipe.generate(prompt, gc2)
t1 = time.perf_counter()
print(f"Type: {type(result)}")
print(f"Time: {t1 - t0:.2f}s")

if isinstance(result, str):
    print(f"Result (str): {result}")
else:
    print(f"Result: {result}")
    for attr in sorted(dir(result)):
        if not attr.startswith('_'):
            try:
                val = getattr(result, attr)
                if not callable(val):
                    print(f"  {attr} = {val}")
            except Exception as e:
                print(f"  {attr} -> error: {e}")

# Try with streaming to count tokens precisely
print(f"\nStreaming inference...")
token_count = [0]
token_times = []
def streamer(subword):
    token_times.append(time.perf_counter())
    token_count[0] += 1
    return False

t0 = time.perf_counter()
result2 = pipe.generate(prompt, max_new_tokens=64, streamer=streamer)
t1 = time.perf_counter()
print(f"Result: {result2}")
print(f"Tokens: {token_count[0]}")
print(f"Total: {t1 - t0:.2f}s")

if len(token_times) > 1:
    ttft = token_times[0] - t0
    print(f"TTFT: {ttft:.3f}s")

    # Per-token times
    inter_token = []
    for i in range(1, len(token_times)):
        dt = token_times[i] - token_times[i-1]
        inter_token.append(dt)

    avg_itt = sum(inter_token) / len(inter_token)
    min_itt = min(inter_token)
    max_itt = max(inter_token)
    print(f"Inter-token times: avg={avg_itt*1000:.0f}ms, min={min_itt*1000:.0f}ms, max={max_itt*1000:.0f}ms")
    print(f"Decode tok/s: {len(inter_token) / sum(inter_token):.1f}")

    print(f"\nPer-token breakdown:")
    for i, dt in enumerate(inter_token):
        print(f"  token {i+1}->{i+2}: {dt*1000:.0f}ms")

del pipe
print("\nDone.")
