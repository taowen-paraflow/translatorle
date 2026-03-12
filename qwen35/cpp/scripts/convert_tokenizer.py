"""Convert HuggingFace tokenizer to OpenVINO IR format.

Produces openvino_tokenizer.xml/.bin and openvino_detokenizer.xml/.bin in
the same directory as the source HF model, ready for the C++ OVTokenizer
wrapper to consume.

Usage:
    python -m scripts.convert_tokenizer --model-dir /path/to/hf_model
"""

import argparse
from pathlib import Path

from transformers import AutoTokenizer
from openvino_tokenizers import convert_tokenizer
import openvino as ov


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace tokenizer to OpenVINO IR"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to HuggingFace model directory containing tokenizer files",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    print(f"Loading tokenizer from {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=True
    )

    print("Converting tokenizer to OpenVINO IR ...")
    ov_tokenizer, ov_detokenizer = convert_tokenizer(
        tokenizer, with_detokenizer=True
    )

    tok_path = model_dir / "openvino_tokenizer.xml"
    detok_path = model_dir / "openvino_detokenizer.xml"

    ov.save_model(ov_tokenizer, str(tok_path))
    ov.save_model(ov_detokenizer, str(detok_path))

    print(f"Saved tokenizer IR to {model_dir}")
    print(f"  - {tok_path.name}")
    print(f"  - {detok_path.name}")


if __name__ == "__main__":
    main()
