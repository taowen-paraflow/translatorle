#!/usr/bin/env python3
"""Run Qwen3.5-VL inference on an exported OpenVINO model.

Usage (from project root, via main venv):
    uv run python -m qwen35.scripts.run_vl_inference --image path/to/image.jpg --prompt "Describe this image"

Usage (pre-processed inputs, for testing without full image preprocessor):
    uv run python -m qwen35.scripts.run_vl_inference --pixel-values pv.npy --grid-thw grid.npy --prompt "What is in this image?"

The script supports two modes:
  1. Pre-processed: provide --pixel-values and --grid-thw as .npy files
  2. Raw image: provide --image (requires qwen_vl_utils or manual preprocessing)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_QWEN35_DIR = _SCRIPT_DIR.parent
_PROJECT_DIR = _QWEN35_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR))

from qwen35.config import MODELS_DIR, VL_IMAGE_SIZE
from qwen35.inference import Qwen35VLModel


def preprocess_image_basic(image_path: str):
    """Basic image preprocessing for Qwen3.5-VL.

    This is a simplified preprocessor that:
    1. Loads and resizes the image to the fixed export resolution (VL_IMAGE_SIZE)
    2. Normalizes pixel values
    3. Duplicates the frame for temporal_patch_size=2
    4. Flattens into 2D patches [num_patches, channel_dim]

    The exported vision encoder (PatchEmbed3d) expects:
      pixel_values: [num_patches, channel_dim]  where channel_dim = C * T * pH * pW
      grid_thw:     [N, 3]  with (temporal_patches, h_patches, w_patches)

    For Qwen3.5-VL: patch_size=16, temporal_patch_size=2, in_channels=3
      -> channel_dim = 3 * 2 * 16 * 16 = 1536

    IMPORTANT: The image is always resized to VL_IMAGE_SIZE x VL_IMAGE_SIZE
    (default 384x384) to match the positional embeddings baked into the
    exported vision encoder IR.  Using a different resolution will cause a
    shape mismatch at the Add_3 node (positional embedding addition).

    For production use, the HuggingFace Qwen3VLProcessor should be preferred
    as it handles all edge cases (aspect ratio, multi-scale, etc.), but then
    the vision encoder must be re-exported with dynamic positional embeddings.

    Args:
        image_path: Path to an image file.

    Returns:
        pixel_values: numpy array [num_patches, 1536] for the vision encoder.
        grid_thw: numpy array [1, 3] with (temporal_patches, h_patches, w_patches).
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL (Pillow) is required for image preprocessing. "
                          "Install with: pip install Pillow")

    img = Image.open(image_path).convert("RGB")

    # Qwen3.5-VL vision config: patch_size=16, temporal_patch_size=2
    patch_size = 16
    temporal_patch_size = 2
    in_channels = 3

    channel_dim = in_channels * temporal_patch_size * patch_size * patch_size  # 1536

    # Force the fixed export resolution so patch count matches the IR.
    # VL_IMAGE_SIZE must be a multiple of patch_size (384 // 16 = 24).
    w = VL_IMAGE_SIZE
    h = VL_IMAGE_SIZE

    img = img.resize((w, h), Image.BICUBIC)

    # Convert to numpy [H, W, 3] -> normalize -> [3, H, W]
    pixels_np = np.array(img, dtype=np.float32) / 255.0

    # ImageNet normalization (used by Qwen VL models)
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    pixels_np = (pixels_np - mean) / std

    # [H, W, 3] -> [3, H, W]
    pixels_np = pixels_np.transpose(2, 0, 1)  # [C, H, W]

    # Duplicate for temporal_patch_size=2: [2, C, H, W]
    pixels_np = np.stack([pixels_np, pixels_np], axis=0)  # [T, C, H, W]

    # Compute patch grid dimensions
    h_patches = h // patch_size   # number of patches along height
    w_patches = w // patch_size   # number of patches along width
    temporal_patches = 1          # 2 frames / temporal_patch_size=2 = 1

    # Reshape into patches and flatten to [num_patches, channel_dim]
    # [T, C, H, W] -> [T, C, h_patches, patch_size, w_patches, patch_size]
    pixels_np = pixels_np.reshape(
        temporal_patch_size, in_channels,
        h_patches, patch_size,
        w_patches, patch_size,
    )
    # Permute to: [h_patches, w_patches, T, C, patch_size, patch_size]
    pixels_np = pixels_np.transpose(2, 4, 0, 1, 3, 5)
    # Flatten: [h_patches * w_patches, T * C * patch_size * patch_size]
    num_patches = temporal_patches * h_patches * w_patches
    pixel_values = pixels_np.reshape(num_patches, channel_dim).astype(np.float32)

    # grid_thw: raw patch counts (the vision encoder handles merge internally)
    grid_thw = np.array([[temporal_patches, h_patches, w_patches]], dtype=np.int64)

    return pixel_values, grid_thw


def main():
    default_model = str(MODELS_DIR / "Qwen3.5-0.8B-vl")

    parser = argparse.ArgumentParser(
        description="Run Qwen3.5-VL OpenVINO inference"
    )
    parser.add_argument(
        "--model-path",
        default=default_model,
        help="Path to VL model directory (default: %(default)s)",
    )
    parser.add_argument(
        "--image",
        help="Path to input image (JPG/PNG). Requires PIL.",
    )
    parser.add_argument(
        "--pixel-values",
        help="Path to pre-processed pixel_values .npy file.",
    )
    parser.add_argument(
        "--grid-thw",
        help="Path to pre-processed grid_thw .npy file.",
    )
    parser.add_argument(
        "--prompt",
        default="Describe this image in detail.",
        help="Text prompt (default: %(default)s)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum number of new tokens to generate (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        default="CPU",
        help="OpenVINO device (CPU, GPU) (default: %(default)s)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    # Validate inputs
    if not args.image and not args.pixel_values:
        parser.error("Either --image or --pixel-values must be provided")

    if args.pixel_values and not args.grid_thw:
        parser.error("--grid-thw is required when using --pixel-values")

    # Load model
    print(f"Loading VL model from {args.model_path} on {args.device}...")
    t0 = time.time()
    model = Qwen35VLModel.from_pretrained(args.model_path, device=args.device)
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")
    print(f"  {model!r}")

    # Prepare image inputs
    if args.pixel_values:
        print(f"Loading pre-processed inputs from {args.pixel_values}...")
        pixel_values = np.load(args.pixel_values)
        grid_thw = np.load(args.grid_thw)
    else:
        print(f"Preprocessing image: {args.image}")
        pixel_values, grid_thw = preprocess_image_basic(args.image)

    print(f"  pixel_values: {pixel_values.shape} {pixel_values.dtype}")
    print(f"  grid_thw: {grid_thw}")

    # Run generation
    print(f"\nPrompt: {args.prompt!r}")
    print(f"Generating (max {args.max_new_tokens} tokens)...")

    t_start = time.time()
    result = model.generate(
        pixel_values=pixel_values,
        grid_thw=grid_thw,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
    )
    elapsed = time.time() - t_start

    # Estimate token count from result
    result_tokens = model._tokenizer.encode(result)
    num_tokens = len(result_tokens)
    tok_per_sec = num_tokens / elapsed if elapsed > 0 else 0

    print(f"\nOutput: {result}")
    print(f"  ({num_tokens} tokens in {elapsed:.2f}s = {tok_per_sec:.1f} tok/s)")


if __name__ == "__main__":
    main()
