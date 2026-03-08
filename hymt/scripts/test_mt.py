"""Test MT module with sample translations.

Usage:
    uv run python hymt/scripts/test_mt.py
    uv run python hymt/scripts/test_mt.py --device CPU
    uv run python hymt/scripts/test_mt.py --text "Hello world" --target_lang Chinese
"""

import os
import sys
import time

# Project root (hymt/scripts/ -> hymt/ -> translatorle/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


TEST_CASES = [
    ("It's on the house.", "Chinese"),
    ("The quick brown fox jumps over the lazy dog.", "Chinese"),
    ("人工智能正在改变世界的方方面面，从医疗保健到金融服务，再到教育领域。", "English"),
    ("Machine translation has made remarkable progress in recent years.", "Chinese"),
]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test HY-MT translation")
    parser.add_argument("--device", default="NPU", help="Device: NPU, CPU, or GPU")
    parser.add_argument("--text", default=None, help="Single text to translate")
    parser.add_argument("--target_lang", default="Chinese", help="Target language")
    args = parser.parse_args()

    from hymt import MTEngine

    print("=" * 60)
    print(f"  HY-MT Translation Test ({args.device})")
    print("=" * 60)

    # Init engine
    t0 = time.perf_counter()
    engine = MTEngine(device=args.device)
    init_time = time.perf_counter() - t0
    print(f"  Engine init: {init_time:.1f}s")

    # Warmup
    print("  Warmup ...")
    engine.translate("Hello", "Chinese")

    if args.text:
        # Single translation
        t0 = time.perf_counter()
        result = engine.translate(args.text, args.target_lang)
        elapsed = time.perf_counter() - t0
        print(f"\n  Source:  {args.text}")
        print(f"  Target:  {result}")
        print(f"  Time:    {elapsed:.3f}s")
    else:
        # Run all test cases
        print()
        results = []
        for text, lang in TEST_CASES:
            t0 = time.perf_counter()
            result = engine.translate(text, lang)
            elapsed = time.perf_counter() - t0
            results.append((text, lang, result, elapsed))
            print(f"  [{elapsed:.3f}s] {text[:50]}")
            print(f"    -> {result}")
            print()

        # Summary
        print("=" * 60)
        print("  SUMMARY")
        print("=" * 60)
        for text, lang, result, elapsed in results:
            print(f"  {elapsed:.3f}s | {text[:30]:30s} -> {result[:40]}")


if __name__ == "__main__":
    main()
