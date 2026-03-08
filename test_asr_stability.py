"""Automated stability test for long-duration ASR streaming.

Repeats test WAV files to create ~60s of audio, feeds through ASR engine
in streaming chunks, and checks that output keeps growing over time.

Usage:
    uv run python test_asr_stability.py [--wav test_zh.wav] [--duration 60] [--device NPU]
"""

import argparse
import time

import librosa
import numpy as np

from asr.config import SAMPLE_RATE
from asr.engine import ASREngine


def build_long_audio(wav_path: str, target_duration: float) -> np.ndarray:
    """Load a WAV file and repeat it with silence gaps to reach target duration."""
    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
    clip_dur = len(audio) / SAMPLE_RATE
    silence = np.zeros(int(0.8 * SAMPLE_RATE), dtype=np.float32)  # 0.8s gap

    segments = []
    total = 0.0
    while total < target_duration:
        segments.append(audio)
        segments.append(silence)
        total += clip_dur + 0.8

    return np.concatenate(segments)


def run_test(wav_path: str, target_duration: float, device: str) -> bool:
    """Run the streaming ASR stability test. Returns True if passed."""
    print(f"Loading {wav_path} and repeating to ~{target_duration:.0f}s ...")
    long_audio = build_long_audio(wav_path, target_duration)
    total_sec = len(long_audio) / SAMPLE_RATE
    print(f"Total audio: {total_sec:.1f}s ({len(long_audio)} samples)")

    print(f"\nInitializing ASR engine (encoder={device}, decoder={device}) ...")
    t0 = time.perf_counter()
    engine = ASREngine(encoder_device=device, decoder_device=device)
    state = engine.new_session()
    print(f"Engine ready in {time.perf_counter() - t0:.1f}s\n")

    # Feed in 100ms chunks (simulating real-time mic input)
    chunk_samples = int(0.1 * SAMPLE_RATE)  # 1600 samples = 100ms
    pos = 0
    last_text_len = 0
    last_growth_time = 0.0
    max_stall_sec = 0.0
    checkpoints = []  # (audio_time, text_len, wall_time)

    print(f"{'Audio Time':>10} {'TextLen':>8} {'Stall':>6} {'Text (last 60 chars)':>60}")
    print("-" * 90)

    wall_start = time.perf_counter()

    while pos < len(long_audio):
        chunk = long_audio[pos : pos + chunk_samples]
        pos += len(chunk)
        engine.feed(chunk, state)

        audio_time = pos / SAMPLE_RATE
        cur_len = len(state.text)

        if cur_len > last_text_len:
            stall = audio_time - last_growth_time if last_growth_time > 0 else 0
            max_stall_sec = max(max_stall_sec, stall)
            last_growth_time = audio_time
            last_text_len = cur_len

            tail = state.text[-60:].replace("\n", " ")
            print(f"{audio_time:10.1f}s {cur_len:8d} {stall:5.1f}s  {tail}")
            checkpoints.append((audio_time, cur_len, time.perf_counter() - wall_start))

    # Finish
    engine.finish(state)
    final_len = len(state.text)
    wall_total = time.perf_counter() - wall_start

    if final_len > last_text_len:
        checkpoints.append((total_sec, final_len, wall_total))

    print("-" * 90)
    print(f"\n[FINAL] text length: {final_len} chars")
    print(f"[FINAL] text: {state.text[:200]}...")
    print(f"\nWall time: {wall_total:.1f}s for {total_sec:.1f}s audio (RTF={wall_total/total_sec:.2f}x)")
    print(f"Max stall between text growth: {max_stall_sec:.1f}s")

    # Stability check: text should keep growing across the full duration
    # Divide audio into 10s windows and check each has some text growth
    window_sec = 10.0
    n_windows = int(total_sec / window_sec)
    stale_windows = []

    for i in range(n_windows):
        t_start = i * window_sec
        t_end = (i + 1) * window_sec
        texts_in_window = [c for c in checkpoints if t_start <= c[0] < t_end]
        if not texts_in_window:
            stale_windows.append((t_start, t_end))

    print(f"\n--- Stability Report ---")
    print(f"Total 10s windows: {n_windows}")
    print(f"Stale windows (no text growth): {len(stale_windows)}")
    for s, e in stale_windows:
        print(f"  STALE: {s:.0f}s - {e:.0f}s")

    passed = len(stale_windows) == 0 and final_len > 20
    print(f"\nResult: {'PASS' if passed else 'FAIL'}")
    if not passed and stale_windows:
        print("FAIL reason: ASR stopped producing output in one or more 10s windows")
    elif not passed:
        print("FAIL reason: Final text too short")

    return passed


def main():
    parser = argparse.ArgumentParser(description="ASR streaming stability test")
    parser.add_argument("--wav", default="test_zh.wav", help="WAV file to repeat")
    parser.add_argument("--duration", type=float, default=60, help="Target audio duration in seconds")
    parser.add_argument("--device", default="NPU", help="Device for encoder/decoder (NPU, CPU, GPU)")
    args = parser.parse_args()

    passed = run_test(args.wav, args.duration, args.device)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
