"""Export Qwen3-0.6B to OpenVINO IR (FP16) for NPU/GPU/CPU benchmark.

Uses optimum-intel for proper stateful model export compatible with LLMPipeline.

Usage:
    uv run python -m qwen3.scripts.prepare_qwen3_models
"""

import os
import time
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "qwen3"
HF_MODEL_ID = "Qwen/Qwen3-0.6B"
EXPORT_DIR = MODELS_DIR / "Qwen3-0.6B-ov"


def main():
    os.makedirs(str(MODELS_DIR), exist_ok=True)

    if EXPORT_DIR.exists() and (EXPORT_DIR / "openvino_model.xml").exists():
        print(f"Model already exported at {EXPORT_DIR}")
        print("Delete the directory to re-export.")
        return

    print(f"Exporting {HF_MODEL_ID} to {EXPORT_DIR} ...")
    t0 = time.perf_counter()

    from optimum.intel import OVModelForCausalLM
    from transformers import AutoTokenizer

    ov_model = OVModelForCausalLM.from_pretrained(
        HF_MODEL_ID,
        export=True,
        load_in_8bit=False,
        trust_remote_code=True,
    )
    ov_model.save_pretrained(str(EXPORT_DIR))

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    tokenizer.save_pretrained(str(EXPORT_DIR))

    # Convert tokenizer to OV format for LLMPipeline
    from openvino_tokenizers import convert_tokenizer
    import openvino as ov
    ov_tok, ov_detok = convert_tokenizer(tokenizer, with_detokenizer=True)
    ov.save_model(ov_tok, str(EXPORT_DIR / "openvino_tokenizer.xml"))
    ov.save_model(ov_detok, str(EXPORT_DIR / "openvino_detokenizer.xml"))

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s")

    for f in sorted(os.listdir(str(EXPORT_DIR))):
        fpath = EXPORT_DIR / f
        if fpath.is_file():
            size_mb = fpath.stat().st_size / 1024 / 1024
            print(f"  {f}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
