"""Comprehensive benchmark: ASR encoders, decoders, and MT on ALL viable devices.

Tests every model/device combination and reports a summary table.

Usage (from WSL, targeting Windows Python):
    powershell.exe -Command '
        $env:PYTHONIOENCODING = "utf-8";
        cd C:\\Apps\\translatorle;
        C:\\Users\\taowen\\.local\\bin\\uv.exe run python asr/scripts/bench_asr_models.py
    '
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ---------------------------------------------------------------------------
# Model paths (relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENCODER_0_6B = str(PROJECT_ROOT / "models" / "encoder_fp16.xml")
DECODER_0_6B = str(PROJECT_ROOT / "models" / "decoder_stateful_embeds" / "openvino_model.xml")
EMBED_0_6B = str(PROJECT_ROOT / "models" / "embed_tokens.npy")

ENCODER_1_7B = str(PROJECT_ROOT / "models" / "asr_1.7b" / "encoder_fp16.xml")
DECODER_1_7B_PRESURGERY = str(PROJECT_ROOT / "models" / "asr_1.7b" / "decoder_stateful_ov" / "openvino_model.xml")
DECODER_1_7B_POSTSURGERY = str(PROJECT_ROOT / "models" / "asr_1.7b" / "decoder_stateful_embeds" / "openvino_model.xml")
EMBED_1_7B = str(PROJECT_ROOT / "models" / "asr_1.7b" / "embed_tokens.npy")

MT_MODEL_DIR = str(PROJECT_ROOT / "models" / "hy_mt_int4sym")
MT_CACHE_DIR = str(PROJECT_ROOT / "models" / "hy_mt_cache_sym")

# ---------------------------------------------------------------------------
# Benchmark parameters
# ---------------------------------------------------------------------------
ENCODER_MEL_SHAPE = (1, 128, 800)
ENCODER_WARMUP = 1
ENCODER_RUNS = 10

DECODER_PREFILL_LEN = 10
DECODER_DECODE_TOKENS = 50

MT_PROMPT = "Translate to Chinese: Hello, how are you today?"
MT_RUNS = 3

# Device configs
NPU_DECODER_CONFIG_0_6B = {
    "NPU_USE_NPUW": "YES",
    "NPUW_LLM": "YES",
    "NPUW_LLM_BATCH_DIM": 0,
    "NPUW_LLM_SEQ_LEN_DIM": 2,
    "NPUW_LLM_MAX_PROMPT_LEN": 256,
    "NPUW_LLM_MIN_RESPONSE_LEN": 64,
}

NPU_DECODER_CONFIG_1_7B = {
    "NPU_USE_NPUW": "YES",
    "NPUW_LLM": "YES",
    "NPUW_LLM_BATCH_DIM": 0,
    "NPUW_LLM_SEQ_LEN_DIM": 2,
    "NPUW_LLM_MAX_PROMPT_LEN": 256,
    "NPUW_LLM_MIN_RESPONSE_LEN": 64,
}

MT_NPU_CONFIG = {
    "MAX_PROMPT_LEN": 512,
    "NPUW_LLM_PREFILL_CHUNK_SIZE": 512,
    "GENERATE_HINT": "BEST_PERF",
    "CACHE_DIR": MT_CACHE_DIR,
}

MT_GPU_CONFIG = {
    "PERFORMANCE_HINT": "LATENCY",
    "NUM_STREAMS": "1",
    "CACHE_DIR": MT_CACHE_DIR,
}

# ---------------------------------------------------------------------------
# Result storage
# ---------------------------------------------------------------------------
results: list[dict] = []


def record(category: str, model: str, device: str, status: str, **metrics):
    """Append a result row."""
    row = {"category": category, "model": model, "device": device, "status": status}
    row.update(metrics)
    results.append(row)


# ===================================================================
# 1. Encoder benchmarks
# ===================================================================

def bench_encoder(label: str, xml_path: str, device: str):
    """Benchmark an encoder: compile time + inference latency."""
    print(f"\n{'='*60}")
    print(f"  ENCODER: {label} on {device}")
    print(f"{'='*60}")

    if not Path(xml_path).exists():
        msg = f"Model file not found: {xml_path}"
        print(f"  SKIP: {msg}")
        record("Encoder", label, device, "SKIP", note=msg)
        return

    try:
        import openvino as ov

        core = ov.Core()
        config = {}
        if device in ("CPU", "GPU"):
            config = {"PERFORMANCE_HINT": "LATENCY"}

        # Compile
        t0 = time.perf_counter()
        compiled = core.compile_model(xml_path, device, config)
        compile_s = time.perf_counter() - t0
        print(f"  Compile time: {compile_s:.2f} s")

        input_name = compiled.inputs[0].any_name
        dummy_mel = np.random.randn(*ENCODER_MEL_SHAPE).astype(np.float32)

        # Warmup
        for _ in range(ENCODER_WARMUP):
            result = compiled({input_name: dummy_mel})

        output = list(result.values())[0]
        print(f"  Output shape: {output.shape}")

        # Timed runs
        times = []
        for i in range(ENCODER_RUNS):
            t0 = time.perf_counter()
            compiled({input_name: dummy_mel})
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

        avg_ms = (sum(times) / len(times)) * 1000
        min_ms = min(times) * 1000
        max_ms = max(times) * 1000
        print(f"  Inference ({ENCODER_RUNS} runs): avg={avg_ms:.1f}ms  min={min_ms:.1f}ms  max={max_ms:.1f}ms")

        del compiled
        record("Encoder", label, device, "OK",
               compile_s=f"{compile_s:.2f}",
               avg_ms=f"{avg_ms:.1f}",
               min_ms=f"{min_ms:.1f}",
               max_ms=f"{max_ms:.1f}",
               output_shape=str(output.shape))

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
        record("Encoder", label, device, "FAIL", note=str(e)[:120])


# ===================================================================
# 2. Decoder benchmarks (post-surgery: inputs_embeds)
# ===================================================================

def bench_decoder_embeds(label: str, xml_path: str, embed_npy: str,
                         hidden_dim: int, device: str, npu_config: dict | None):
    """Benchmark a post-surgery decoder (inputs_embeds interface)."""
    print(f"\n{'='*60}")
    print(f"  DECODER (inputs_embeds): {label} on {device}")
    print(f"{'='*60}")

    if not Path(xml_path).exists():
        msg = f"Model file not found: {xml_path}"
        print(f"  SKIP: {msg}")
        record("Decoder-embeds", label, device, "SKIP", note=msg)
        return
    if not Path(embed_npy).exists():
        msg = f"Embed table not found: {embed_npy}"
        print(f"  SKIP: {msg}")
        record("Decoder-embeds", label, device, "SKIP", note=msg)
        return

    try:
        import openvino as ov

        core = ov.Core()
        if device == "NPU":
            config = npu_config or {}
        else:
            config = {"PERFORMANCE_HINT": "LATENCY"}

        # Compile
        t0 = time.perf_counter()
        compiled = core.compile_model(xml_path, device, config)
        compile_s = time.perf_counter() - t0
        print(f"  Compile time: {compile_s:.2f} s")

        request = compiled.create_infer_request()
        embed_table = np.load(embed_npy)
        print(f"  Embed table shape: {embed_table.shape} (hidden_dim={hidden_dim})")

        # --- Prefill ---
        seq_len = DECODER_PREFILL_LEN
        dummy_embeds = np.random.randn(1, seq_len, hidden_dim).astype(np.float32) * 0.01
        attention_mask = np.ones((1, seq_len), dtype=np.int64)
        position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
        beam_idx = np.array([0], dtype=np.int32)

        request.reset_state()
        t0 = time.perf_counter()
        request.infer({
            "inputs_embeds": dummy_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "beam_idx": beam_idx,
        })
        prefill_ms = (time.perf_counter() - t0) * 1000
        logits = request.get_output_tensor(0).data
        first_token = int(np.argmax(logits[0, -1, :]))
        print(f"  Prefill ({seq_len} tokens): {prefill_ms:.1f} ms")

        # --- Decode loop ---
        past_len = seq_len
        token_id = first_token
        decode_times = []
        for step in range(DECODER_DECODE_TOKENS):
            token_embed = embed_table[token_id][np.newaxis, np.newaxis, :].astype(np.float32)
            attn_mask = np.ones((1, past_len + 1), dtype=np.int64)
            pos_ids = np.array([[past_len]], dtype=np.int64)

            t0 = time.perf_counter()
            request.infer({
                "inputs_embeds": token_embed,
                "attention_mask": attn_mask,
                "position_ids": pos_ids,
                "beam_idx": beam_idx,
            })
            decode_times.append(time.perf_counter() - t0)

            logits = request.get_output_tensor(0).data
            token_id = int(np.argmax(logits[0, -1, :]))
            past_len += 1

        avg_decode_ms = (sum(decode_times) / len(decode_times)) * 1000
        tok_s = 1000.0 / avg_decode_ms if avg_decode_ms > 0 else 0
        print(f"  Decode ({DECODER_DECODE_TOKENS} tokens): avg={avg_decode_ms:.1f} ms/tok  ({tok_s:.1f} tok/s)")

        del request, compiled
        record("Decoder-embeds", label, device, "OK",
               compile_s=f"{compile_s:.2f}",
               prefill_ms=f"{prefill_ms:.1f}",
               decode_ms_tok=f"{avg_decode_ms:.1f}",
               tok_s=f"{tok_s:.1f}")

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
        record("Decoder-embeds", label, device, "FAIL", note=str(e)[:120])


# ===================================================================
# 3. Decoder benchmark (pre-surgery: input_ids) -- for 1.7B NPU test
# ===================================================================

def bench_decoder_ids(label: str, xml_path: str, embed_npy: str,
                      device: str, npu_config: dict | None):
    """Benchmark a pre-surgery decoder (input_ids interface) on NPU."""
    print(f"\n{'='*60}")
    print(f"  DECODER (input_ids): {label} on {device}")
    print(f"{'='*60}")

    if not Path(xml_path).exists():
        msg = f"Model file not found: {xml_path}"
        print(f"  SKIP: {msg}")
        record("Decoder-ids", label, device, "SKIP", note=msg)
        return

    try:
        import openvino as ov

        core = ov.Core()
        if device == "NPU":
            config = npu_config or {}
        else:
            config = {"PERFORMANCE_HINT": "LATENCY"}

        # Compile
        t0 = time.perf_counter()
        compiled = core.compile_model(xml_path, device, config)
        compile_s = time.perf_counter() - t0
        print(f"  Compile time: {compile_s:.2f} s")

        request = compiled.create_infer_request()

        # Load embed table for decode steps
        embed_npy_path = Path(embed_npy)
        has_embed = embed_npy_path.exists()
        if has_embed:
            embed_table = np.load(str(embed_npy_path))
            print(f"  Embed table loaded: {embed_table.shape}")

        # --- Prefill with input_ids ---
        seq_len = DECODER_PREFILL_LEN
        dummy_ids = np.ones((1, seq_len), dtype=np.int64) * 100  # arbitrary token IDs
        attention_mask = np.ones((1, seq_len), dtype=np.int64)
        position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
        beam_idx = np.array([0], dtype=np.int32)

        request.reset_state()
        t0 = time.perf_counter()
        request.infer({
            "input_ids": dummy_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "beam_idx": beam_idx,
        })
        prefill_ms = (time.perf_counter() - t0) * 1000
        logits = request.get_output_tensor(0).data
        first_token = int(np.argmax(logits[0, -1, :]))
        print(f"  Prefill ({seq_len} tokens): {prefill_ms:.1f} ms")

        # --- Decode loop (still using input_ids) ---
        past_len = seq_len
        token_id = first_token
        decode_times = []
        for step in range(DECODER_DECODE_TOKENS):
            ids = np.array([[token_id]], dtype=np.int64)
            attn_mask = np.ones((1, past_len + 1), dtype=np.int64)
            pos_ids = np.array([[past_len]], dtype=np.int64)

            t0 = time.perf_counter()
            request.infer({
                "input_ids": ids,
                "attention_mask": attn_mask,
                "position_ids": pos_ids,
                "beam_idx": beam_idx,
            })
            decode_times.append(time.perf_counter() - t0)

            logits = request.get_output_tensor(0).data
            token_id = int(np.argmax(logits[0, -1, :]))
            past_len += 1

        avg_decode_ms = (sum(decode_times) / len(decode_times)) * 1000
        tok_s = 1000.0 / avg_decode_ms if avg_decode_ms > 0 else 0
        print(f"  Decode ({DECODER_DECODE_TOKENS} tokens): avg={avg_decode_ms:.1f} ms/tok  ({tok_s:.1f} tok/s)")

        del request, compiled
        record("Decoder-ids", label, device, "OK",
               compile_s=f"{compile_s:.2f}",
               prefill_ms=f"{prefill_ms:.1f}",
               decode_ms_tok=f"{avg_decode_ms:.1f}",
               tok_s=f"{tok_s:.1f}")

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
        record("Decoder-ids", label, device, "FAIL", note=str(e)[:120])


# ===================================================================
# 4. MT LLMPipeline benchmark
# ===================================================================

def bench_mt(label: str, model_dir: str, device: str, config: dict):
    """Benchmark MT model via openvino_genai.LLMPipeline."""
    print(f"\n{'='*60}")
    print(f"  MT (LLMPipeline): {label} on {device}")
    print(f"{'='*60}")

    if not Path(model_dir).exists():
        msg = f"MT model dir not found: {model_dir}"
        print(f"  SKIP: {msg}")
        record("MT", label, device, "SKIP", note=msg)
        return

    try:
        import openvino_genai as ov_genai

        # Compile
        t0 = time.perf_counter()
        pipe = ov_genai.LLMPipeline(model_dir, device, **config)
        compile_s = time.perf_counter() - t0
        print(f"  Compile time: {compile_s:.2f} s")

        # First generate (cold)
        t0 = time.perf_counter()
        out = pipe.generate(MT_PROMPT, max_new_tokens=64)
        first_s = time.perf_counter() - t0
        print(f"  First generate: {first_s:.2f} s")
        print(f"    Output: {out.strip()[:80]}")

        # Subsequent generates
        gen_times = []
        for i in range(MT_RUNS):
            t0 = time.perf_counter()
            out = pipe.generate(MT_PROMPT, max_new_tokens=64)
            elapsed = time.perf_counter() - t0
            gen_times.append(elapsed)
            print(f"    Run {i+1}: {elapsed:.3f} s  -> {out.strip()[:60]}")

        avg_s = sum(gen_times) / len(gen_times)
        print(f"  Avg generate: {avg_s:.3f} s ({MT_RUNS} runs)")

        del pipe
        record("MT", label, device, "OK",
               compile_s=f"{compile_s:.2f}",
               first_gen_s=f"{first_s:.2f}",
               avg_gen_s=f"{avg_s:.3f}")

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
        record("MT", label, device, "FAIL", note=str(e)[:120])


# ===================================================================
# Summary printer
# ===================================================================

def print_summary():
    """Print a formatted summary table of all results."""
    print()
    print("=" * 90)
    print("  BENCHMARK SUMMARY")
    print("=" * 90)

    # --- Encoder results ---
    enc_results = [r for r in results if r["category"] == "Encoder"]
    if enc_results:
        print()
        print("  ENCODERS")
        print(f"  {'Model':<16} {'Device':<6} {'Status':<6} {'Compile(s)':<11} {'Avg(ms)':<9} {'Min(ms)':<9} {'Max(ms)':<9}")
        print(f"  {'-'*16} {'-'*6} {'-'*6} {'-'*11} {'-'*9} {'-'*9} {'-'*9}")
        for r in enc_results:
            if r["status"] == "OK":
                print(f"  {r['model']:<16} {r['device']:<6} {r['status']:<6} "
                      f"{r['compile_s']:>10} {r['avg_ms']:>8} {r['min_ms']:>8} {r['max_ms']:>8}")
            else:
                note = r.get("note", "")[:50]
                print(f"  {r['model']:<16} {r['device']:<6} {r['status']:<6} {note}")

    # --- Decoder (inputs_embeds) results ---
    dec_e_results = [r for r in results if r["category"] == "Decoder-embeds"]
    if dec_e_results:
        print()
        print("  DECODERS (inputs_embeds, post-surgery)")
        print(f"  {'Model':<16} {'Device':<6} {'Status':<6} {'Compile(s)':<11} {'Prefill(ms)':<12} {'Decode(ms/t)':<13} {'tok/s':<8}")
        print(f"  {'-'*16} {'-'*6} {'-'*6} {'-'*11} {'-'*12} {'-'*13} {'-'*8}")
        for r in dec_e_results:
            if r["status"] == "OK":
                print(f"  {r['model']:<16} {r['device']:<6} {r['status']:<6} "
                      f"{r['compile_s']:>10} {r['prefill_ms']:>11} {r['decode_ms_tok']:>12} {r['tok_s']:>7}")
            else:
                note = r.get("note", "")[:50]
                print(f"  {r['model']:<16} {r['device']:<6} {r['status']:<6} {note}")

    # --- Decoder (input_ids) results ---
    dec_i_results = [r for r in results if r["category"] == "Decoder-ids"]
    if dec_i_results:
        print()
        print("  DECODERS (input_ids, pre-surgery -- proves NPU CAN run 1.7B)")
        print(f"  {'Model':<16} {'Device':<6} {'Status':<6} {'Compile(s)':<11} {'Prefill(ms)':<12} {'Decode(ms/t)':<13} {'tok/s':<8}")
        print(f"  {'-'*16} {'-'*6} {'-'*6} {'-'*11} {'-'*12} {'-'*13} {'-'*8}")
        for r in dec_i_results:
            if r["status"] == "OK":
                print(f"  {r['model']:<16} {r['device']:<6} {r['status']:<6} "
                      f"{r['compile_s']:>10} {r['prefill_ms']:>11} {r['decode_ms_tok']:>12} {r['tok_s']:>7}")
            else:
                note = r.get("note", "")[:50]
                print(f"  {r['model']:<16} {r['device']:<6} {r['status']:<6} {note}")

    # --- MT results ---
    mt_results = [r for r in results if r["category"] == "MT"]
    if mt_results:
        print()
        print("  MT (HY-MT INT4_SYM via LLMPipeline)")
        print(f"  {'Model':<16} {'Device':<6} {'Status':<6} {'Compile(s)':<11} {'1st Gen(s)':<11} {'Avg Gen(s)':<11}")
        print(f"  {'-'*16} {'-'*6} {'-'*6} {'-'*11} {'-'*11} {'-'*11}")
        for r in mt_results:
            if r["status"] == "OK":
                print(f"  {r['model']:<16} {r['device']:<6} {r['status']:<6} "
                      f"{r['compile_s']:>10} {r['first_gen_s']:>10} {r['avg_gen_s']:>10}")
            else:
                note = r.get("note", "")[:50]
                print(f"  {r['model']:<16} {r['device']:<6} {r['status']:<6} {note}")

    print()
    print("=" * 90)
    total = len(results)
    ok = sum(1 for r in results if r["status"] == "OK")
    fail = sum(1 for r in results if r["status"] == "FAIL")
    skip = sum(1 for r in results if r["status"] == "SKIP")
    print(f"  Total: {total} tests | OK: {ok} | FAIL: {fail} | SKIP: {skip}")
    print("=" * 90)


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 90)
    print("  COMPREHENSIVE ASR + MT BENCHMARK")
    print("  All models x All viable devices")
    print("=" * 90)
    print(f"  Encoder: mel input {ENCODER_MEL_SHAPE}, {ENCODER_WARMUP} warmup + {ENCODER_RUNS} runs")
    print(f"  Decoder: prefill {DECODER_PREFILL_LEN} tokens, decode {DECODER_DECODE_TOKENS} tokens")
    print(f"  MT: \"{MT_PROMPT[:50]}...\", {MT_RUNS} runs")
    print()

    # ------------------------------------------------------------------
    # 1. ENCODERS
    # ------------------------------------------------------------------
    print("\n" + "#" * 90)
    print("#  SECTION 1: ENCODERS")
    print("#" * 90)

    # 0.6B Encoder (FP16) on NPU, GPU, CPU
    for device in ("NPU", "GPU", "CPU"):
        bench_encoder("0.6B-FP16", ENCODER_0_6B, device)

    # 1.7B Encoder (FP16) on NPU, GPU, CPU
    for device in ("NPU", "GPU", "CPU"):
        bench_encoder("1.7B-FP16", ENCODER_1_7B, device)

    # ------------------------------------------------------------------
    # 2. DECODERS (post-surgery, inputs_embeds)
    # ------------------------------------------------------------------
    print("\n" + "#" * 90)
    print("#  SECTION 2: DECODERS (post-surgery, inputs_embeds)")
    print("#" * 90)

    # 0.6B Decoder (FP16, post-surgery) on NPU, GPU, CPU
    for device in ("NPU", "GPU", "CPU"):
        npu_cfg = NPU_DECODER_CONFIG_0_6B if device == "NPU" else None
        bench_decoder_embeds("0.6B-FP16", DECODER_0_6B, EMBED_0_6B,
                             hidden_dim=1024, device=device, npu_config=npu_cfg)

    # 1.7B Decoder (INT4_SYM, post-surgery) on GPU, CPU
    # NPU FAILS for post-surgery 1.7B because NPUW_LLM cannot partition the
    # inputs_embeds model after IR surgery -- the folding pass fails when
    # input_ids is removed. Only pre-surgery (input_ids) works on NPU.
    for device in ("GPU", "CPU"):
        bench_decoder_embeds("1.7B-INT4", DECODER_1_7B_POSTSURGERY, EMBED_1_7B,
                             hidden_dim=2048, device=device, npu_config=None)

    # ------------------------------------------------------------------
    # 3. DECODER (pre-surgery, input_ids) -- 1.7B on NPU
    # ------------------------------------------------------------------
    print("\n" + "#" * 90)
    print("#  SECTION 3: DECODER (pre-surgery, input_ids) -- 1.7B NPU proof")
    print("#  This proves NPU CAN run the 1.7B decoder when using input_ids")
    print("#  (before IR surgery removes it). Post-surgery fails on NPU.")
    print("#" * 90)

    bench_decoder_ids("1.7B-INT4-presurg", DECODER_1_7B_PRESURGERY, EMBED_1_7B,
                      device="NPU", npu_config=NPU_DECODER_CONFIG_1_7B)

    # ------------------------------------------------------------------
    # 4. MT (HY-MT INT4_SYM via LLMPipeline)
    # ------------------------------------------------------------------
    print("\n" + "#" * 90)
    print("#  SECTION 4: MT (HY-MT INT4_SYM via LLMPipeline)")
    print("#" * 90)

    bench_mt("HY-MT-INT4", MT_MODEL_DIR, "NPU", MT_NPU_CONFIG)
    bench_mt("HY-MT-INT4", MT_MODEL_DIR, "GPU", MT_GPU_CONFIG)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print_summary()


if __name__ == "__main__":
    main()
