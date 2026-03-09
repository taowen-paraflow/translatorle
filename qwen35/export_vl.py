"""VL-specific export utilities for Qwen3.5 Vision-Language models.

Two functions for the VL export pipeline:
  1. export_vision_encoder()      -- ViT vision encoder -> OpenVINO IR
  2. extract_embed_tokens()       -- Save embedding table as .npy

The decoder IR is now exported with inputs_embeds directly (no surgery needed).
Both text-only and VL share the same decoder IR.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

import openvino as ov

from qwen35.config import VL_IMAGE_SIZE

logger = logging.getLogger(__name__)


# ============================================================================
# Function 1: Export vision encoder
# ============================================================================

class VisionEncoderWrapper(torch.nn.Module):
    """Wrapper around the ViT vision encoder for clean tracing.

    Provides a simple (pixel_values, grid_thw) -> hidden_states interface
    suitable for ``ov.convert_model``.
    """

    def __init__(self, visual):
        super().__init__()
        self.visual = visual

    def forward(self, pixel_values, grid_thw):
        return self.visual(pixel_values, grid_thw=grid_thw)


def export_vision_encoder(model, output_dir, device="cpu"):
    """Export the vision encoder (ViT) from a Qwen3.5 VL model to OpenVINO IR.

    Args:
        model: The full Qwen3_5ForConditionalGeneration model.
        output_dir: Directory to save vision_encoder.xml/.bin.
        device: Torch device for tracing (default: "cpu").
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Qwen3_5ForConditionalGeneration: model.model.visual (Qwen3_5VisionModel)
    visual = getattr(model, "visual", None)
    if visual is None:
        visual = getattr(model.model, "visual", None)
    if visual is None:
        raise AttributeError(
            f"Cannot find vision encoder on {type(model).__name__}. "
            f"Children: {[n for n, _ in model.named_children()]}"
        )

    # Force eager attention (required for torch tracing -- no flash_attention_2)
    if hasattr(visual, "config"):
        visual.config._attn_implementation = "eager"

    # Try different attribute names for blocks / layers
    blocks = getattr(visual, "blocks", getattr(visual, "layers", []))
    for block in blocks:
        for attn_name in ("attn", "self_attn"):
            attn = getattr(block, attn_name, None)
            if attn is not None:
                if hasattr(attn, "config"):
                    attn.config._attn_implementation = "eager"
                if hasattr(attn, "_attn_implementation"):
                    attn._attn_implementation = "eager"
                break

    wrapper = VisionEncoderWrapper(visual)
    wrapper.eval()
    wrapper.to(device)

    # Determine input shapes from vision config.
    # Qwen3.5 VisionModel.PatchEmbed3d reshapes pixel_values to:
    #   [-1, in_channels, temporal_patch_size, patch_size, patch_size]
    # So channel_dim = in_channels * temporal_patch_size * patch_size * patch_size
    vision_config = visual.config if hasattr(visual, "config") else None
    patch_size = getattr(vision_config, "patch_size", 16)
    temporal_patch_size = getattr(vision_config, "temporal_patch_size", 2)
    in_channels = getattr(vision_config, "in_channels", 3)
    channel_dim = in_channels * temporal_patch_size * patch_size * patch_size
    logger.info(
        "Vision config: patch_size=%d, temporal_patch_size=%d, in_channels=%d, channel_dim=%d",
        patch_size, temporal_patch_size, in_channels, channel_dim,
    )

    # Example: 384x384 image -> spatial grid 384/patch_size = 24 per dim
    # grid_thw = [temporal=1, h=24, w=24], num_patches = 1*24*24 = 576
    image_size = VL_IMAGE_SIZE
    grid_h = image_size // patch_size
    grid_w = image_size // patch_size
    grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.int64)
    num_patches = 1 * grid_h * grid_w
    pixel_values = torch.randn(
        num_patches, channel_dim, dtype=torch.float32, device=device
    )

    logger.info(
        "Tracing vision encoder: pixel_values=%s, grid_thw=%s",
        list(pixel_values.shape), list(grid_thw.shape),
    )

    with torch.no_grad():
        ref_output = wrapper(pixel_values, grid_thw)
    if hasattr(ref_output, "shape"):
        logger.info("Reference output shape: %s", list(ref_output.shape))
    else:
        logger.info("Reference output type: %s", type(ref_output))

    ov_model = ov.convert_model(
        wrapper,
        example_input=(pixel_values, grid_thw),
        input=[
            ov.PartialShape([-1, channel_dim]),  # pixel_values: [num_patches, 588]
            ov.PartialShape([-1, 3]),             # grid_thw: [N, 3]
        ],
    )

    # Name inputs
    ov_model.inputs[0].get_tensor().set_names({"pixel_values"})
    ov_model.inputs[1].get_tensor().set_names({"grid_thw"})

    xml_path = output_dir / "vision_encoder.xml"
    ov.save_model(ov_model, str(xml_path))

    bin_path = xml_path.with_suffix(".bin")
    xml_mb = xml_path.stat().st_size / 1024 / 1024
    bin_mb = bin_path.stat().st_size / 1024 / 1024
    logger.info("Saved: %s (%.1f MB) + .bin (%.1f MB)", xml_path.name, xml_mb, bin_mb)


# ============================================================================
# Function 2: Extract embed_tokens.npy
# ============================================================================

def extract_embed_tokens(model, output_path):
    """Extract the embedding table from a VL model and save as .npy.

    Uses ``model.get_input_embeddings()`` which works for both
    ``Qwen3_5ForConditionalGeneration`` and ``Qwen3_5ForCausalLM``.

    Args:
        model: The full model (VL or text-only).
        output_path: Path to save the .npy file.
    """
    output_path = Path(output_path)
    embed_table = model.get_input_embeddings().weight.detach().cpu().numpy()
    logger.info("Embedding table shape: %s", embed_table.shape)

    np.save(str(output_path), embed_table.astype(np.float16))
    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info("Saved: %s (%.1f MB)", output_path, size_mb)
