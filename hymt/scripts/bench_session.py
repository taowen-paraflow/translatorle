"""Benchmark: stateless vs session (KV cache reuse) translation.

Translates a sequence of sentences and measures per-sentence latency.
In stateless mode, context grows linearly. In session mode, KV cache is reused.

Usage:
    uv run python hymt/scripts/bench_session.py
    uv run python hymt/scripts/bench_session.py --device CPU
"""

import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# Simulate a real transcription session: 10 sentences of varying length
SENTENCES = [
    "甚至出现交易几乎停滞的情况。",
    "这些被视为华尔街最安全的交易品种。",
    "市场流动性急剧下降引发了投资者的恐慌。",
    "美国国债收益率在短短几天内大幅波动。",
    "分析师指出这种现象在过去十年中极为罕见。",
    "全球央行正在密切关注事态的发展。",
    "投资者纷纷转向黄金等避险资产寻求保护。",
    "华尔街的交易员们加班加点处理激增的订单。",
    "这场风波暴露了金融市场深层次的结构性问题。",
    "监管机构表示将采取一切必要措施稳定市场。",
]

TARGET_LANG = "English"


def bench_stateless(engine):
    """Translate with growing context (old behavior)."""
    times = []
    translated = []
    for sentence in SENTENCES:
        context = "\n".join(translated) if translated else ""
        t0 = time.perf_counter()
        result = engine.translate(sentence, TARGET_LANG, context=context)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        translated.append(result)
    return times, translated


def bench_session(engine):
    """Translate with KV cache reuse (new behavior)."""
    engine.start_session(TARGET_LANG)
    times = []
    translated = []
    for i, sentence in enumerate(SENTENCES):
        t0 = time.perf_counter()
        result = engine.translate(sentence, TARGET_LANG)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        translated.append(result)
        # Check if reset would be triggered (don't actually reset to keep comparison fair)
        if engine.needs_reset(400):
            print(f"  [token limit reached at sentence {i+1}, resetting]")
            engine.reset_session(TARGET_LANG)
    engine.finish_session()
    return times, translated


def print_results(label, times, translated):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  {'#':>3s}  {'Time':>7s}  Translation")
    print(f"  {'-'*3}  {'-'*7}  {'-'*40}")
    total = 0
    for i, (t, tr) in enumerate(zip(times, translated)):
        total += t
        print(f"  {i+1:3d}  {t:6.3f}s  {tr[:50]}")
    print(f"  {'-'*3}  {'-'*7}")
    print(f"  Avg: {total/len(times):.3f}s   Total: {total:.3f}s")
    print(f"  1st sentence: {times[0]:.3f}s")
    print(f"  Last sentence: {times[-1]:.3f}s")
    print(f"  Slowdown ratio (last/first): {times[-1]/times[0]:.2f}x")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="GPU")
    args = parser.parse_args()

    from hymt.engine import MTEngine

    print(f"Device: {args.device}")
    t0 = time.perf_counter()
    engine = MTEngine(device=args.device)
    print(f"Engine init: {time.perf_counter() - t0:.2f}s")

    # Warmup
    print("Warmup...")
    engine.translate("你好", "English")
    print()

    # Run benchmarks
    print("Benchmarking stateless mode (growing context)...")
    stateless_times, stateless_results = bench_stateless(engine)

    print("Benchmarking session mode (KV cache reuse)...")
    session_times, session_results = bench_session(engine)

    # Print results
    print_results("STATELESS (growing context)", stateless_times, stateless_results)
    print_results("SESSION (KV cache reuse)", session_times, session_results)

    # Summary comparison
    print(f"\n{'='*60}")
    print(f"  COMPARISON")
    print(f"{'='*60}")
    s_avg = sum(stateless_times) / len(stateless_times)
    k_avg = sum(session_times) / len(session_times)
    s_total = sum(stateless_times)
    k_total = sum(session_times)
    print(f"  {'':20s}  {'Stateless':>10s}  {'Session':>10s}  {'Speedup':>8s}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*8}")
    print(f"  {'Avg per sentence':20s}  {s_avg:9.3f}s  {k_avg:9.3f}s  {s_avg/k_avg:7.2f}x")
    print(f"  {'Total (10 sent)':20s}  {s_total:9.3f}s  {k_total:9.3f}s  {s_total/k_total:7.2f}x")
    print(f"  {'1st sentence':20s}  {stateless_times[0]:9.3f}s  {session_times[0]:9.3f}s  {stateless_times[0]/session_times[0]:7.2f}x")
    print(f"  {'Last sentence':20s}  {stateless_times[-1]:9.3f}s  {session_times[-1]:9.3f}s  {stateless_times[-1]/session_times[-1]:7.2f}x")
    print(f"  {'Slowdown (last/1st)':20s}  {stateless_times[-1]/stateless_times[0]:9.2f}x  {session_times[-1]/session_times[0]:9.2f}x")


if __name__ == "__main__":
    main()
