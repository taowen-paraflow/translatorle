#!/usr/bin/env python3
r"""Test all precision/device combinations for the PARO INT4 single-IR model.

Finds the fastest config that produces correct output by testing various
INFERENCE_PRECISION_HINT, EXECUTION_MODE_HINT, NUM_REQUESTS, and per-op
precision enforcement settings.

Usage (from project root via Windows uv):
    powershell.exe -Command '$env:PYTHONIOENCODING="utf-8"; cd C:\Apps\translatorle; C:\Users\taowen\.local\bin\uv.exe run python -m qwen35.scripts.test_precision_combos'
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import openvino as ov
from openvino.preprocess import PrePostProcessor

_SCRIPT_DIR = Path(__file__).resolve().parent
_QWEN35_DIR = _SCRIPT_DIR.parent
_PROJECT_DIR = _QWEN35_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_DIR = str(_PROJECT_DIR / "models" / "qwen35" / "Qwen3.5-0.8B-paro-ov-int4sym")
STOP_TOKENS = {151645, 151643, 248044}
PROMPT = "Hello, what can you do?"
MAX_TOKENS = 30

# Minimum recognizable-word threshold for quality check
MIN_WORDS = 3

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_int8_embeddings(model_dir: str):
    """Load INT8 quantized embedding table and per-row scales."""
    int8_path = Path(model_dir) / "embed_tokens_int8.npy"
    scales_path = Path(model_dir) / "embed_tokens_scales.npy"

    embed_int8 = np.load(str(int8_path))        # [vocab, hidden] int8
    embed_scales = np.load(str(scales_path))     # [vocab] float16
    embed_scales_f32 = embed_scales.astype(np.float32)
    return embed_int8, embed_scales_f32


def embed_lookup(token_ids: np.ndarray, embed_int8: np.ndarray, embed_scales: np.ndarray) -> np.ndarray:
    """INT8 dequantize embedding lookup: result = int8_val * scale_per_row."""
    rows = embed_int8[token_ids]                # [seq_len, hidden] int8
    scales = embed_scales[token_ids]            # [seq_len]
    return (rows.astype(np.float32) * scales[:, np.newaxis]).astype(np.float32)


def detect_input_names(model: ov.Model) -> set:
    """Return the set of input tensor names accepted by the model."""
    return {inp.get_any_name() for inp in model.inputs}


def quality_check(text: str) -> str:
    """Return 'OK' if the text looks like recognizable English, else 'FAIL'.

    Simple heuristic: at least MIN_WORDS ASCII words of length >= 2.
    """
    import re
    words = re.findall(r"[A-Za-z]{2,}", text)
    if len(words) >= MIN_WORDS:
        return "OK"
    return "FAIL"


def run_generation(
    request: ov.InferRequest,
    input_names: set,
    tokenizer,
    embed_int8: np.ndarray,
    embed_scales: np.ndarray,
    prompt: str = PROMPT,
    max_tokens: int = MAX_TOKENS,
) -> dict:
    """Run prefill + decode loop and return results dict.

    Returns dict with keys: output_text, generated_ids, prefill_ms,
    decode_ms, decode_toks, tok_per_sec.
    """
    input_ids = tokenizer.encode(prompt)
    seq_len = len(input_ids)

    # --- Prefill ---
    request.reset_state()
    embeds = embed_lookup(np.array(input_ids, dtype=np.int64), embed_int8, embed_scales)

    feed = {
        "inputs_embeds": embeds[np.newaxis, :, :],  # [1, seq_len, hidden]
    }
    if "attention_mask" in input_names:
        feed["attention_mask"] = np.ones((1, seq_len), dtype=np.int64)
    if "position_ids" in input_names:
        positions = np.arange(seq_len, dtype=np.int64)
        feed["position_ids"] = np.tile(positions[np.newaxis, np.newaxis, :], (3, 1, 1))
    if "beam_idx" in input_names:
        feed["beam_idx"] = np.array([0], dtype=np.int32)

    t_prefill = time.perf_counter()
    request.infer(feed)
    prefill_ms = (time.perf_counter() - t_prefill) * 1000.0

    # First token
    logits = request.get_tensor("logits").data  # [1, seq_len, vocab]
    next_id = int(np.argmax(logits[0, -1, :]))

    # --- Decode loop ---
    generated = [next_id]
    past_length = seq_len
    t_decode = time.perf_counter()

    for _ in range(max_tokens - 1):
        if next_id in STOP_TOKENS:
            break

        tok_embed = embed_lookup(np.array([next_id], dtype=np.int64), embed_int8, embed_scales)
        total_len = past_length + 1

        feed = {
            "inputs_embeds": tok_embed[np.newaxis, :, :],  # [1, 1, hidden]
        }
        if "attention_mask" in input_names:
            feed["attention_mask"] = np.ones((1, total_len), dtype=np.int64)
        if "position_ids" in input_names:
            feed["position_ids"] = np.full((3, 1, 1), past_length, dtype=np.int64)
        if "beam_idx" in input_names:
            feed["beam_idx"] = np.array([0], dtype=np.int32)

        request.infer(feed)
        past_length += 1

        logits = request.get_tensor("logits").data
        next_id = int(np.argmax(logits[0, -1, :]))
        generated.append(next_id)

    decode_ms = (time.perf_counter() - t_decode) * 1000.0

    # Remove trailing stop token
    if generated and generated[-1] in STOP_TOKENS:
        generated.pop()

    output_text = tokenizer.decode(generated, skip_special_tokens=True)
    decode_toks = len(generated)
    tok_per_sec = (decode_toks / (decode_ms / 1000.0)) if decode_ms > 0 else 0

    return {
        "output_text": output_text,
        "generated_ids": generated,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "decode_toks": decode_toks,
        "tok_per_sec": tok_per_sec,
    }


# ---------------------------------------------------------------------------
# Per-op precision enforcement experiments
# ---------------------------------------------------------------------------

def count_loops(model: ov.Model) -> int:
    """Count Loop nodes in the model."""
    count = 0
    for op in model.get_ops():
        if op.get_type_name() == "Loop":
            count += 1
    return count


def disable_fp16_on_loops(model: ov.Model) -> int:
    """Mark Loop nodes with disable_fp16_compression to keep them in FP32.

    This is the official OpenVINO mechanism for mixed precision:
    the GPU plugin's ConvertPrecision pass skips FP16 conversion
    for nodes marked with this rt_info attribute.
    """
    count = 0
    for op in model.get_ops():
        if op.get_type_name() == "Loop":
            try:
                op.get_rt_info()["disable_fp16_compression"] = True
                count += 1
            except Exception:
                pass
    return count


def disable_fp16_on_all_ops(model: ov.Model) -> int:
    """Mark ALL ops with disable_fp16_compression."""
    count = 0
    for op in model.get_ops():
        try:
            op.get_rt_info()["disable_fp16_compression"] = True
            count += 1
        except Exception:
            pass
    return count


def disable_fp16_on_loops_and_neighbors(model: ov.Model) -> int:
    """Mark Loop nodes AND all their input/output producer/consumer ops.

    The GDN recurrence involves not just the Loop itself but surrounding
    ops (MatMul, Multiply, Add) that feed into and consume Loop outputs.
    Keeping them all in FP32 may improve accuracy.
    """
    count = 0
    loop_connected = set()
    for op in model.get_ops():
        if op.get_type_name() == "Loop":
            loop_connected.add(op.get_friendly_name())
            # Add input producers
            for i in range(op.get_input_size()):
                src = op.input(i).get_source_output().get_node()
                loop_connected.add(src.get_friendly_name())
            # Add output consumers
            for i in range(op.get_output_size()):
                for target_input in op.output(i).get_target_inputs():
                    loop_connected.add(target_input.get_node().get_friendly_name())
    for op in model.get_ops():
        if op.get_friendly_name() in loop_connected:
            try:
                op.get_rt_info()["disable_fp16_compression"] = True
                count += 1
            except Exception:
                pass
    return count


# ---------------------------------------------------------------------------
# Configuration definitions
# ---------------------------------------------------------------------------

def get_configs():
    """Return the list of (name, device, compile_props, model_modifier) tuples.

    model_modifier is an optional callable(model) -> model that modifies the
    ov.Model before compilation (e.g., per-op precision enforcement).
    """
    configs = [
        # --- Standard device/precision combos ---
        ("GPU-f32", "GPU", {"INFERENCE_PRECISION_HINT": "f32"}, None),
        ("GPU-f16", "GPU", {"INFERENCE_PRECISION_HINT": "f16"}, None),
        ("GPU-bf16", "GPU", {"INFERENCE_PRECISION_HINT": "bf16"}, None),
        ("GPU-default", "GPU", {}, None),
        ("GPU-accuracy", "GPU", {"EXECUTION_MODE_HINT": "ACCURACY"}, None),
        ("GPU-f16-nreq1", "GPU", {"INFERENCE_PRECISION_HINT": "f16", "NUM_REQUESTS": "1"}, None),
        ("CPU", "CPU", {}, None),

        # --- disable_fp16_compression on Loop nodes (official mixed precision API) ---
        # Keep Loop body (GDN recurrence) in FP32, rest in FP16
        (
            "GPU-f16+loop-nofp16",
            "GPU",
            {"INFERENCE_PRECISION_HINT": "f16"},
            lambda m: (disable_fp16_on_loops(m), m)[1],
        ),
        # Same but with default GPU precision (usually f16)
        (
            "GPU-default+loop-nofp16",
            "GPU",
            {},
            lambda m: (disable_fp16_on_loops(m), m)[1],
        ),
        # Loop + neighbors (surrounding MatMul/Add) kept in FP32
        (
            "GPU-f16+loopN-nofp16",
            "GPU",
            {"INFERENCE_PRECISION_HINT": "f16"},
            lambda m: (disable_fp16_on_loops_and_neighbors(m), m)[1],
        ),
        # BF16 + Loop nodes marked FP32
        (
            "GPU-bf16+loop-nofp16",
            "GPU",
            {"INFERENCE_PRECISION_HINT": "bf16"},
            lambda m: (disable_fp16_on_loops(m), m)[1],
        ),
        # All ops marked disable_fp16 + f16 hint (should behave like f32)
        (
            "GPU-f16+all-nofp16",
            "GPU",
            {"INFERENCE_PRECISION_HINT": "f16"},
            lambda m: (disable_fp16_on_all_ops(m), m)[1],
        ),
    ]
    return configs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test precision/device combos for PARO INT4 single-IR model"
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="Path to PARO model directory (default: %(default)s)",
    )
    parser.add_argument(
        "--prompt",
        default=PROMPT,
        help="Prompt to test with (default: %(default)r)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_TOKENS,
        help="Max tokens to generate per config (default: %(default)s)",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Only run configs whose name contains this substring",
    )
    args = parser.parse_args()

    model_dir = args.model_dir
    xml_path = str(Path(model_dir) / "openvino_model.xml")

    print("=" * 80)
    print("PARO INT4 Precision/Device Combo Test")
    print("=" * 80)
    print(f"Model:      {model_dir}")
    print(f"Prompt:     {args.prompt!r}")
    print(f"Max tokens: {args.max_tokens}")
    print()

    # Validate model files exist
    if not Path(xml_path).exists():
        print(f"ERROR: Model not found at {xml_path}")
        sys.exit(1)

    int8_path = Path(model_dir) / "embed_tokens_int8.npy"
    scales_path = Path(model_dir) / "embed_tokens_scales.npy"
    if not int8_path.exists() or not scales_path.exists():
        print(f"ERROR: INT8 embedding files not found in {model_dir}")
        sys.exit(1)

    # Load embeddings once (shared across all configs)
    print("Loading INT8 embeddings...")
    embed_int8, embed_scales = load_int8_embeddings(model_dir)
    print(f"  embed_int8:   shape={embed_int8.shape}, dtype={embed_int8.dtype}")
    print(f"  embed_scales: shape={embed_scales.shape}, dtype=float32")
    print()

    # Load tokenizer once
    from transformers import AutoTokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print()

    # Initialize OpenVINO core
    core = ov.Core()

    # NOTE: Do NOT set global GPU properties here — they can pollute later tests.
    print()

    # --- Count Loop nodes in model ---
    print("--- Model structure analysis ---")
    temp_model = core.read_model(xml_path)
    num_loops = count_loops(temp_model)
    input_names = detect_input_names(temp_model)
    print(f"  Loop nodes: {num_loops}")
    print(f"  Input names: {sorted(input_names)}")
    print(f"  Outputs: {len(temp_model.outputs)}")

    # Show op type distribution (top 10)
    op_types = {}
    for op in temp_model.get_ops():
        t = op.get_type_name()
        op_types[t] = op_types.get(t, 0) + 1
    top_ops = sorted(op_types.items(), key=lambda x: -x[1])[:15]
    print(f"  Top op types: {top_ops}")
    del temp_model
    print()

    # --- Run configs ---
    configs = get_configs()
    if args.filter:
        configs = [(n, d, p, m) for n, d, p, m in configs if args.filter in n]
        print(f"Filtered to {len(configs)} configs matching '{args.filter}'")
        print()

    results = []
    total = len(configs)

    for idx, (name, device, props, modifier) in enumerate(configs, 1):
        print(f"[{idx}/{total}] {name} (device={device}, props={props})")
        result = {
            "name": name,
            "compile_ms": None,
            "tok_per_sec": None,
            "output_text": None,
            "quality": "SKIP",
            "error": None,
        }

        try:
            # Read model fresh for each config
            model = core.read_model(xml_path)

            # Add FP32 output conversion for GPU/NPU
            if device in ("GPU", "NPU"):
                ppp = PrePostProcessor(model)
                for i in range(len(model.outputs)):
                    ppp.output(i).tensor().set_element_type(ov.Type.f32)
                model = ppp.build()

            # Apply model modifier if specified (per-op precision enforcement)
            if modifier is not None:
                model = modifier(model)
                print(f"  Applied model modifier")

            # Detect input names from this model instance
            model_input_names = detect_input_names(model)

            # Compile
            t_compile = time.perf_counter()
            compiled = core.compile_model(model, device, props)
            compile_ms = (time.perf_counter() - t_compile) * 1000.0
            result["compile_ms"] = compile_ms
            print(f"  Compiled in {compile_ms:.0f} ms")

            request = compiled.create_infer_request()

            # Run generation
            gen = run_generation(
                request=request,
                input_names=model_input_names,
                tokenizer=tokenizer,
                embed_int8=embed_int8,
                embed_scales=embed_scales,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
            )

            result["tok_per_sec"] = gen["tok_per_sec"]
            result["output_text"] = gen["output_text"]
            result["quality"] = quality_check(gen["output_text"])
            result["decode_toks"] = gen["decode_toks"]
            result["prefill_ms"] = gen["prefill_ms"]
            result["decode_ms"] = gen["decode_ms"]

            preview = gen["output_text"][:80].replace("\n", " ")
            print(f"  Prefill: {gen['prefill_ms']:.0f} ms")
            print(f"  Decode:  {gen['decode_toks']} toks in {gen['decode_ms']:.0f} ms "
                  f"= {gen['tok_per_sec']:.1f} tok/s")
            print(f"  Quality: {result['quality']}")
            print(f"  Output:  {preview!r}")

        except Exception as e:
            result["error"] = str(e)
            print(f"  ERROR: {e}")
            traceback.print_exc()

        results.append(result)
        print()

    # --- Summary table ---
    print("=" * 120)
    print("SUMMARY TABLE")
    print("=" * 120)
    print(f"{'Config':<28} {'Compile(ms)':>11} {'Decode(tok/s)':>14} {'Quality':>8}  Output (first 80 chars)")
    print("-" * 120)

    for r in results:
        name = r["name"]
        compile_str = f"{r['compile_ms']:.0f}" if r["compile_ms"] is not None else "---"
        speed_str = f"{r['tok_per_sec']:.1f}" if r["tok_per_sec"] is not None else "---"
        quality = r["quality"]
        if r["error"]:
            output_preview = f"ERROR: {r['error'][:60]}"
        elif r["output_text"] is not None:
            output_preview = r["output_text"][:80].replace("\n", " ")
        else:
            output_preview = "---"

        print(f"{name:<28} {compile_str:>11} {speed_str:>14} {quality:>8}  {output_preview}")

    print("-" * 120)

    # --- Find best config ---
    ok_results = [r for r in results if r["quality"] == "OK" and r["tok_per_sec"] is not None]
    if ok_results:
        best = max(ok_results, key=lambda r: r["tok_per_sec"])
        print(f"\nBEST CONFIG (quality OK): {best['name']}  "
              f"@ {best['tok_per_sec']:.1f} tok/s  "
              f"(compile {best['compile_ms']:.0f} ms)")
    else:
        print("\nNo config produced quality=OK output!")

    fail_results = [r for r in results if r["quality"] == "FAIL" and r["tok_per_sec"] is not None]
    if fail_results:
        fastest_fail = max(fail_results, key=lambda r: r["tok_per_sec"])
        print(f"FASTEST FAIL:             {fastest_fail['name']}  "
              f"@ {fastest_fail['tok_per_sec']:.1f} tok/s  "
              f"(compile {fastest_fail['compile_ms']:.0f} ms)")

    error_results = [r for r in results if r["error"] is not None]
    if error_results:
        print(f"\n{len(error_results)} config(s) failed with errors:")
        for r in error_results:
            print(f"  {r['name']}: {r['error'][:100]}")

    print()


if __name__ == "__main__":
    main()
