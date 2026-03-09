#!/usr/bin/env python3
"""Quick TTS benchmark: load models, generate speech, report timing."""

import argparse
import sys
import time
from pathlib import Path

# Ensure parent package is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
_TTS_DIR = _SCRIPT_DIR.parent
_PROJECT_DIR = _TTS_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR))

from tts.config import TTS_MODELS, DECODER_SAMPLE_RATE


def benchmark(model_key: str, text: str, speaker_id: int = 3066) -> None:
    from tts.engine import TTSEngine

    config = TTS_MODELS[model_key]
    print(f"\n{'='*60}")
    print(f"  Benchmark: {model_key}")
    print(f"  Talker:  {config.talker_device}")
    print(f"  CP:      {config.cp_device}")
    print(f"  Decoder: {config.decoder_device}")
    print(f"  Text:    {text}")
    print(f"{'='*60}")

    print("Loading models...")
    t_load = time.perf_counter()
    engine = TTSEngine(config)
    t_load = time.perf_counter() - t_load
    print(f"Models loaded in {t_load:.1f}s")

    # Warmup run (first run has NPU compilation overhead)
    print("\nWarmup run...")
    t_warmup = time.perf_counter()
    wav_chunks = []
    for chunk_idx, wav in engine.generate(text, speaker_id=speaker_id):
        wav_chunks.append(wav)
    t_warmup = time.perf_counter() - t_warmup

    import numpy as np
    wav_all = np.concatenate(wav_chunks)
    duration_s = len(wav_all) / DECODER_SAMPLE_RATE
    rtf = t_warmup / duration_s if duration_s > 0 else 0
    print(f"Warmup: {t_warmup:.2f}s -> {duration_s:.2f}s audio (RTF {rtf:.2f})")

    # Actual benchmark (2 runs)
    for run in range(1, 3):
        print(f"\nRun {run}...")
        t_run = time.perf_counter()
        wav_chunks = []
        for chunk_idx, wav in engine.generate(text, speaker_id=speaker_id):
            wav_chunks.append(wav)
        t_run = time.perf_counter() - t_run
        wav_all = np.concatenate(wav_chunks)
        duration_s = len(wav_all) / DECODER_SAMPLE_RATE
        rtf = t_run / duration_s if duration_s > 0 else 0
        print(f"Run {run}: {t_run:.2f}s -> {duration_s:.2f}s audio (RTF {rtf:.2f})")

    del engine


def main():
    parser = argparse.ArgumentParser(description="TTS Benchmark")
    parser.add_argument(
        "--models", nargs="+", default=["CPU", "FP16"],
        choices=list(TTS_MODELS.keys()),
        help="Model configs to benchmark (default: CPU FP16)",
    )
    parser.add_argument(
        "--text", default="今天天气真不错，我们一起出去走走吧。",
        help="Text to synthesize",
    )
    args = parser.parse_args()

    for model_key in args.models:
        benchmark(model_key, args.text)


if __name__ == "__main__":
    main()
