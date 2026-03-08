"""Test ASR module with real WAV files.

Usage:
    powershell.exe -Command '$env:Path = "C:\\Users\\taowen\\.local\\bin;$env:Path"; $env:PYTHONIOENCODING = "utf-8"; cd C:\\Apps\\translatorle; uv run python scripts/test_asr_wav.py test_zh.wav'
    powershell.exe -Command '$env:Path = "C:\\Users\\taowen\\.local\\bin;$env:Path"; $env:PYTHONIOENCODING = "utf-8"; cd C:\\Apps\\translatorle; uv run python scripts/test_asr_wav.py test_en.wav'
"""

import os
import sys
import time
import numpy as np

# Project root (asr/scripts/ → asr/ → translatorle/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


def load_wav(path: str, sr: int = 16000) -> np.ndarray:
    """Load WAV file and resample to 16kHz mono float32."""
    import librosa
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio.astype(np.float32)


def test_file(wav_path: str, encoder_device: str, decoder_device: str,
              language: str | None = None, label: str = ""):
    """Run streaming ASR on a WAV file and return results."""
    from asr import ASREngine

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  File: {os.path.basename(wav_path)}")
    print(f"  Encoder: {encoder_device}, Decoder: {decoder_device}")
    print(f"{'='*60}")

    # Load audio
    audio = load_wav(wav_path)
    duration = len(audio) / 16000
    print(f"  Audio duration: {duration:.1f}s ({len(audio)} samples)")

    # Init engine
    t0 = time.perf_counter()
    engine = ASREngine(
        encoder_device=encoder_device,
        decoder_device=decoder_device,
        language=language,
    )
    init_time = time.perf_counter() - t0
    print(f"  Engine init: {init_time*1000:.0f}ms")

    # Run streaming: feed audio in 2s chunks
    state = engine.new_session()
    chunk_size = int(2.0 * 16000)  # 2 seconds
    chunk_times = []

    t_total = time.perf_counter()
    pos = 0
    while pos < len(audio):
        chunk = audio[pos:pos + chunk_size]
        pos += chunk_size

        t0 = time.perf_counter()
        engine.feed(chunk, state)
        elapsed = time.perf_counter() - t0
        chunk_times.append(elapsed)

        text_preview = state.text[:60] if state.text else "(empty)"
        print(f"  [Chunk {len(chunk_times)}] {elapsed*1000:.0f}ms | text: {text_preview}")

    # Flush remaining
    t0 = time.perf_counter()
    engine.finish(state)
    flush_time = time.perf_counter() - t0
    if flush_time > 0.01:
        print(f"  [Flush] {flush_time*1000:.0f}ms")
    total_time = time.perf_counter() - t_total

    print(f"\n  --- Results ---")
    print(f"  Language: {state.language}")
    print(f"  Text: {state.text}")
    print(f"  Raw decoded: {state._raw_decoded}")
    print(f"  Total inference: {total_time*1000:.0f}ms")
    print(f"  RTF: {total_time/duration:.3f}x")
    print(f"  Chunks: {len(chunk_times)}, flush: {flush_time*1000:.0f}ms")
    for i, ct in enumerate(chunk_times):
        print(f"    chunk{i+1}: {ct*1000:.0f}ms")

    return state.text, state.language, total_time, duration


def main():
    # Determine which file(s) to test
    wav_arg = sys.argv[1] if len(sys.argv) > 1 else None

    if wav_arg:
        wav_path = os.path.join(PROJECT_ROOT, wav_arg) if not os.path.isabs(wav_arg) else wav_arg
        if not os.path.exists(wav_path):
            print(f"ERROR: {wav_path} not found")
            sys.exit(1)
        # Auto-detect language from filename
        basename = os.path.basename(wav_path).lower()
        if "zh" in basename or "chinese" in basename:
            language = "Chinese"
        elif "en" in basename or "english" in basename:
            language = "English"
        else:
            language = None

        print("=" * 60)
        print(f"  Translatorle ASR Module Test")
        print("=" * 60)

        text, lang, t, dur = test_file(
            wav_path, "NPU", "NPU", language=language,
            label=f"NPU test: {os.path.basename(wav_path)}"
        )
        rtf = t / dur
        print(f"\nSummary: RTF={rtf:.3f}x, lang={lang}, text={text}")
    else:
        # Run both test files
        zh_wav = os.path.join(PROJECT_ROOT, "test_zh.wav")
        en_wav = os.path.join(PROJECT_ROOT, "test_en.wav")

        print("=" * 60)
        print(f"  Translatorle ASR Module Test (all files)")
        print("=" * 60)

        results = {}

        if os.path.exists(zh_wav):
            text, lang, t, dur = test_file(
                zh_wav, "NPU", "NPU", language="Chinese",
                label="Chinese (NPU encoder + NPU decoder)"
            )
            results["zh_npu"] = {"text": text, "lang": lang, "time": t, "duration": dur}

        if os.path.exists(en_wav):
            text, lang, t, dur = test_file(
                en_wav, "NPU", "NPU", language="English",
                label="English (NPU encoder + NPU decoder)"
            )
            results["en_npu"] = {"text": text, "lang": lang, "time": t, "duration": dur}

        # Summary
        print(f"\n{'='*60}")
        print("  SUMMARY")
        print(f"{'='*60}")
        for key, r in results.items():
            rtf = r["time"] / r["duration"]
            print(f"  {key:10s} | RTF={rtf:.3f}x | lang={r['lang']:8s} | {r['text'][:60]}")


if __name__ == "__main__":
    main()
