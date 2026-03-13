r"""Test NPU GDN prefill vs GPU GDN prefill precision at the generation level.

Loads two Qwen35HybridModel instances (one with gdn_prefill_device="GPU",
one with gdn_prefill_device="NPU") and compares their generation outputs
across a diverse set of prompts. Both use --no-attn-stateful (NPU attention)
with HYBRID device mode.

Run (root venv):
  powershell.exe -Command 'cd C:\Apps\translatorle; $env:PYTHONIOENCODING="utf-8"; C:\Users\taowen\.local\bin\uv.exe run python -m qwen35.scripts.test_npu_precision'
"""

from __future__ import annotations

import logging
import re
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_DIR))

from qwen35.inference_hybrid import Qwen35HybridModel

logger = logging.getLogger(__name__)

MODEL_DIR = "models/qwen35/Qwen3.5-0.8B-hybrid"
MAX_NEW_TOKENS = 20

# ---------------------------------------------------------------------------
# Test prompts: (name, prompt_text)
# ---------------------------------------------------------------------------

TEST_PROMPTS = [
    ("simple_factual", "The capital of France is"),
    ("math_simple", "1+1="),
    ("math_reasoning", "What is 15 * 7? Let me calculate step by step."),
    ("chinese", "\u8bf7\u7528\u4e00\u53e5\u8bdd\u89e3\u91ca\u4ec0\u4e48\u662f\u673a\u5668\u5b66\u4e60"),
    ("code", "def fibonacci(n):"),
    (
        "long_context",
        "Explain the theory of relativity in simple terms."
        " Albert Einstein proposed",
    ),
    ("translation", "Translate to English: \u4eca\u5929\u5929\u6c14\u5f88\u597d"),
    ("counting", "Count from 1 to 10:"),
]


# ---------------------------------------------------------------------------
# Log-capture handler: extracts prefill speed from model logger
# ---------------------------------------------------------------------------

class _MetricsCapture(logging.Handler):
    """Captures prefill/decode timing logged by Qwen35HybridModel.generate()."""

    _PREFILL_RE = re.compile(
        r"Prefill:\s+(\d+)\s+tokens?\s+in\s+([\d.]+)ms\s+\(([\d.]+)\s+tok/s"
    )
    _DECODE_RE = re.compile(
        r"Decode:\s+(\d+)\s+tokens?\s+in\s+([\d.]+)ms\s+\(([\d.]+)\s+tok/s"
    )

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.prefill_tokens: int = 0
        self.prefill_ms: float = 0.0
        self.prefill_tps: float = 0.0
        self.decode_tokens: int = 0
        self.decode_ms: float = 0.0
        self.decode_tps: float = 0.0

    def emit(self, record: logging.LogRecord):
        msg = record.getMessage()
        m = self._PREFILL_RE.search(msg)
        if m:
            self.prefill_tokens = int(m.group(1))
            self.prefill_ms = float(m.group(2))
            self.prefill_tps = float(m.group(3))
        m = self._DECODE_RE.search(msg)
        if m:
            self.decode_tokens = int(m.group(1))
            self.decode_ms = float(m.group(2))
            self.decode_tps = float(m.group(3))


# ---------------------------------------------------------------------------
# Generation with first-token capture
# ---------------------------------------------------------------------------

def generate_with_details(
    model: Qwen35HybridModel,
    prompt: str,
    max_new_tokens: int,
    capture: _MetricsCapture,
) -> dict:
    """Generate text and capture first-token ID, full output, and timing.

    Manually runs prefill + decode loop to capture the first token ID
    before it gets merged into the decoded string.
    """
    capture.reset()
    model.reset()

    # Tokenize
    token_list = model._tokenizer.encode(prompt)
    input_ids = np.array([token_list], dtype=np.int64)
    prompt_len = input_ids.shape[1]

    # Prefill
    t0 = time.time()
    logits = model.prefill(input_ids)
    prefill_time = time.time() - t0

    prefill_tps = prompt_len / prefill_time if prefill_time > 0 else 0.0
    logger.info(
        "Prefill: %d tokens in %.1fms (%.1f tok/s, chunk=%d, layer-major)",
        prompt_len, prefill_time * 1000, prefill_tps, model._prefill_chunk_size,
    )

    # First token
    first_logits = logits[0, -1, :]
    first_token_id = int(np.argmax(first_logits))
    first_top5_ids = np.argsort(first_logits)[-5:][::-1].tolist()
    first_top5_probs = first_logits[first_top5_ids].tolist()

    # Decode remaining tokens
    generated = [first_token_id]
    eos_id = model._tokenizer.eos_token_id
    stop_ids = {eos_id, 151645, 151643} if eos_id else {151645, 151643}

    t_decode_start = time.time()
    next_id = first_token_id
    for _ in range(max_new_tokens - 1):
        if next_id in stop_ids:
            break
        token_input = np.array([[next_id]], dtype=np.int64)
        logits = model.forward(token_input)
        next_id = int(np.argmax(logits[0, -1, :]))
        generated.append(next_id)
    decode_time = time.time() - t_decode_start

    decode_tps = len(generated) / decode_time if decode_time > 0 else 0.0
    logger.info(
        "Decode: %d tokens in %.1fms (%.1f tok/s)",
        len(generated), decode_time * 1000, decode_tps,
    )

    # Remove stop token from output
    if generated and generated[-1] in stop_ids:
        generated = generated[:-1]

    output_text = model._tokenizer.decode(generated, skip_special_tokens=True)

    return {
        "output_text": output_text,
        "first_token_id": first_token_id,
        "first_top5_ids": first_top5_ids,
        "first_top5_probs": first_top5_probs,
        "generated_ids": generated,
        "prompt_tokens": prompt_len,
        "prefill_ms": prefill_time * 1000,
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_separator(char: str = "=", width: int = 90):
    print(char * width)


def truncate(text: str, max_len: int = 70) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # Attach metrics capture to the inference_hybrid logger
    capture = _MetricsCapture()
    inference_logger = logging.getLogger("qwen35.inference_hybrid")
    inference_logger.addHandler(capture)

    print_separator()
    print("NPU GDN Prefill Precision Test")
    print(f"Model: {MODEL_DIR}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print(f"Prompts: {len(TEST_PROMPTS)}")
    print("Both models: HYBRID (GDN=GPU, Attn=NPU, Head=GPU), no-attn-stateful")
    print("Difference: gdn_prefill_device=GPU vs gdn_prefill_device=NPU")
    print_separator()

    # --- Load GPU prefill model ---
    print("\n[1/2] Loading model with gdn_prefill_device=GPU ...")
    t0 = time.time()
    gpu_model = Qwen35HybridModel(
        model_dir=MODEL_DIR,
        gdn_device="GPU",
        attn_device="NPU",
        head_device="GPU",
        attn_stateful=False,
        prefill_chunk_size=16,
        gdn_prefill_device="GPU",
    )
    print(f"  GPU prefill model loaded in {time.time() - t0:.1f}s")

    # --- Load NPU prefill model ---
    print("\n[2/2] Loading model with gdn_prefill_device=NPU ...")
    t0 = time.time()
    npu_model = Qwen35HybridModel(
        model_dir=MODEL_DIR,
        gdn_device="GPU",
        attn_device="NPU",
        head_device="GPU",
        attn_stateful=False,
        prefill_chunk_size=16,
        gdn_prefill_device="NPU",
    )
    print(f"  NPU prefill model loaded in {time.time() - t0:.1f}s")

    # --- Run all prompts ---
    results = []  # list of (name, gpu_result, npu_result)

    for idx, (name, prompt) in enumerate(TEST_PROMPTS):
        print(f"\n{'='*90}")
        print(f"[{idx + 1}/{len(TEST_PROMPTS)}] {name}")
        print(f"  Prompt: {prompt}")
        print("-" * 90)

        # GPU prefill
        print("  Running GPU prefill ...")
        gpu_result = generate_with_details(gpu_model, prompt, MAX_NEW_TOKENS, capture)

        # NPU prefill
        print("  Running NPU prefill ...")
        npu_result = generate_with_details(npu_model, prompt, MAX_NEW_TOKENS, capture)

        results.append((name, gpu_result, npu_result))

        # Display comparison
        first_match = gpu_result["first_token_id"] == npu_result["first_token_id"]
        full_match = gpu_result["generated_ids"] == npu_result["generated_ids"]
        text_match = gpu_result["output_text"] == npu_result["output_text"]

        first_status = "MATCH" if first_match else "DIFFER"
        full_status = "IDENTICAL" if full_match else "DIVERGENT"

        print(f"\n  First token: {first_status}")
        print(f"    GPU: id={gpu_result['first_token_id']}"
              f"  top5={gpu_result['first_top5_ids']}")
        print(f"    NPU: id={npu_result['first_token_id']}"
              f"  top5={npu_result['first_top5_ids']}")

        print(f"\n  Full output: {full_status}")
        print(f"    GPU ({len(gpu_result['generated_ids'])} tok): "
              f"{truncate(gpu_result['output_text'])}")
        print(f"    NPU ({len(npu_result['generated_ids'])} tok): "
              f"{truncate(npu_result['output_text'])}")

        print(f"\n  Prefill speed:")
        print(f"    GPU: {gpu_result['prefill_tps']:.1f} tok/s"
              f"  ({gpu_result['prefill_ms']:.0f}ms,"
              f" {gpu_result['prompt_tokens']} tokens)")
        print(f"    NPU: {npu_result['prefill_tps']:.1f} tok/s"
              f"  ({npu_result['prefill_ms']:.0f}ms,"
              f" {npu_result['prompt_tokens']} tokens)")

    # --- Summary table ---
    print(f"\n{'='*90}")
    print("SUMMARY TABLE")
    print("=" * 90)

    header = (
        f"{'Prompt':<16} | {'1st tok':>7} | {'Full':>9} | "
        f"{'GPU pre':>8} {'NPU pre':>8} | "
        f"{'GPU tok':>7} {'NPU tok':>7}"
    )
    print(header)
    print("-" * len(header))

    match_count = 0
    first_match_count = 0

    for name, gpu_r, npu_r in results:
        first_ok = gpu_r["first_token_id"] == npu_r["first_token_id"]
        full_ok = gpu_r["generated_ids"] == npu_r["generated_ids"]

        if first_ok:
            first_match_count += 1
        if full_ok:
            match_count += 1

        first_str = "OK" if first_ok else "DIFF"
        full_str = "IDENTICAL" if full_ok else "DIVERGENT"

        gpu_pre = f"{gpu_r['prefill_tps']:.1f}"
        npu_pre = f"{npu_r['prefill_tps']:.1f}"
        gpu_tok = str(len(gpu_r["generated_ids"]))
        npu_tok = str(len(npu_r["generated_ids"]))

        print(
            f"{name:<16} | {first_str:>7} | {full_str:>9} | "
            f"{gpu_pre:>8} {npu_pre:>8} | "
            f"{gpu_tok:>7} {npu_tok:>7}"
        )

    print("-" * len(header))

    total = len(results)
    print(f"\nFirst token match: {first_match_count}/{total}")
    print(f"Full output match: {match_count}/{total}")

    if match_count == total:
        print("\nAll outputs IDENTICAL -- NPU GDN prefill matches GPU perfectly.")
    elif first_match_count == total:
        print(
            f"\nAll first tokens match but {total - match_count}/{total} "
            f"outputs diverge during decode (expected: small precision "
            f"differences accumulate over tokens)."
        )
    else:
        divergent = [name for name, g, n in results
                     if g["first_token_id"] != n["first_token_id"]]
        print(f"\nFirst token DIFFERS on: {', '.join(divergent)}")
        print("This indicates meaningful precision loss in NPU GDN prefill.")

    # --- Prefill speed comparison ---
    gpu_prefill_speeds = [g["prefill_tps"] for _, g, _ in results]
    npu_prefill_speeds = [n["prefill_tps"] for _, _, n in results]
    avg_gpu = sum(gpu_prefill_speeds) / len(gpu_prefill_speeds)
    avg_npu = sum(npu_prefill_speeds) / len(npu_prefill_speeds)
    speedup = avg_npu / avg_gpu if avg_gpu > 0 else 0.0

    print(f"\nAvg prefill speed: GPU={avg_gpu:.1f} tok/s, NPU={avg_npu:.1f} tok/s"
          f"  ({speedup:.2f}x)")

    print_separator()


if __name__ == "__main__":
    main()
