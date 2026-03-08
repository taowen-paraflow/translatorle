"""Benchmark MT inference to identify NPU performance bottlenecks.

Measures time-to-first-token (TTFT / prefill) vs decode speed separately
using openvino_genai streaming callback.

Usage:
    uv run python hymt/scripts/bench_mt.py
    uv run python hymt/scripts/bench_mt.py --device CPU
"""

import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

TEST_CASES = [
    ("Hello", "Chinese"),
    ("It's on the house.", "Chinese"),
    ("The quick brown fox jumps over the lazy dog.", "Chinese"),
    ("人工智能正在改变世界的方方面面，从医疗保健到金融服务，再到教育领域。", "English"),
]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="NPU")
    args = parser.parse_args()

    import openvino_genai as ov_genai
    from hymt.config import MT_MODEL_DIR, MT_CACHE_DIR, NPU_CONFIG
    from hymt.engine import MTEngine

    print(f"openvino_genai version: {ov_genai.__version__}")
    print(f"Device: {args.device}")
    print()

    # Init
    t0 = time.perf_counter()
    engine = MTEngine(device=args.device)
    print(f"Engine init: {time.perf_counter() - t0:.2f}s")

    # Warmup
    print("Warmup...")
    engine.translate("Hi", "Chinese")
    print()

    # Benchmark each case with streaming to split TTFT vs decode
    print(f"{'Case':50s} | {'Tokens':>6s} | {'TTFT':>7s} | {'Decode':>7s} | {'Total':>7s} | {'tok/s':>6s}")
    print("-" * 100)

    for text, target_lang in TEST_CASES:
        prompt = engine._build_prompt(text, target_lang)

        token_times = []
        first_token_time = [None]
        start_time = [None]

        def streamer(subword):
            now = time.perf_counter()
            if first_token_time[0] is None:
                first_token_time[0] = now
            token_times.append(now)
            return False  # continue generation

        start_time[0] = time.perf_counter()
        result = engine._pipe.generate(prompt, max_new_tokens=256, streamer=streamer)
        end_time = time.perf_counter()

        n_tokens = len(token_times)
        ttft = first_token_time[0] - start_time[0] if first_token_time[0] else 0
        total = end_time - start_time[0]
        decode_time = total - ttft

        if n_tokens > 1:
            tok_per_sec = (n_tokens - 1) / decode_time if decode_time > 0 else 0
        else:
            tok_per_sec = 0

        label = f"{text[:35]} -> {target_lang}"
        print(f"{label:50s} | {n_tokens:6d} | {ttft:6.2f}s | {decode_time:6.2f}s | {total:6.2f}s | {tok_per_sec:5.1f}")

    # Also try to get perf metrics if available
    print()
    print("=" * 60)
    print("Checking generate() with GenerationConfig for metrics...")
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = 64
    prompt = "Translate to Chinese: Hello world"

    t0 = time.perf_counter()
    result = engine._pipe.generate(prompt, config)
    t1 = time.perf_counter()
    print(f"  Result type: {type(result)}")
    print(f"  Result: {result}")
    print(f"  Time: {t1 - t0:.2f}s")

    # Check if DecodedResults has perf_metrics
    if hasattr(result, 'perf_metrics'):
        m = result.perf_metrics
        print(f"  perf_metrics: {m}")
        for attr in dir(m):
            if not attr.startswith('_'):
                print(f"    {attr}: {getattr(m, attr)}")


if __name__ == "__main__":
    main()
