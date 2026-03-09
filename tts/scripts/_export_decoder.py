"""Speech Decoder model: export to OpenVINO IR (step 14)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Vendored model imports -- add scripts dir to sys.path so qwen_tts package
# can be imported directly.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from qwen_tts.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Config
from qwen_tts.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model

# ---------------------------------------------------------------------------
# Constants (must match config.py values)
# ---------------------------------------------------------------------------
_NUM_QUANTIZERS = 16      # DECODER_NUM_QUANTIZERS
_TOTAL_CHUNK = 75         # DECODER_TOTAL_CHUNK  (50 new + 25 left context)
_UPSAMPLE_RATE = 1920     # DECODER_DECODE_UPSAMPLE_RATE


# ---------------------------------------------------------------------------
# Static wrapper
# ---------------------------------------------------------------------------

class StaticSpeechDecoder(nn.Module):
    """Static-shape wrapper around Qwen3TTSTokenizerV2Decoder for OV export.

    Input:  codes [1, 16, NUM_FRAMES]  (int64)
    Output: wav   [1, 1, NUM_FRAMES * 1920]  (float32)

    This calls the decoder's forward() directly (NOT chunked_decode),
    because chunking is handled in the Python inference loop.
    """

    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        return self.decoder(codes)


# ---------------------------------------------------------------------------
# Traceable causal-mask replacements
# ---------------------------------------------------------------------------
# The default create_causal_mask in transformers uses vmap which is not
# friendly to torch.jit.trace / ov.convert_model.  We replace it with a
# simple triu-based implementation that produces identical masks.

def _traceable_create_causal_mask(
    config, input_embeds, attention_mask=None, cache_position=None,
    past_seen_tokens=0, **kwargs,
):
    dtype = input_embeds.dtype
    seq_len = input_embeds.shape[1]
    if attention_mask is not None and attention_mask.dim() == 4:
        return attention_mask
    min_val = torch.finfo(dtype).min
    causal = torch.full(
        (seq_len, seq_len), min_val, dtype=dtype, device=input_embeds.device,
    )
    causal = torch.triu(causal, diagonal=1).unsqueeze(0).unsqueeze(0)
    if attention_mask is not None and attention_mask.dim() == 2:
        pad = (1.0 - attention_mask[:, None, None, :].to(dtype)) * min_val
        causal = causal + pad
    return causal


def _traceable_create_sliding_window_causal_mask(
    config, input_embeds, attention_mask=None, cache_position=None,
    past_seen_tokens=0, **kwargs,
):
    dtype = input_embeds.dtype
    seq_len = input_embeds.shape[1]
    if attention_mask is not None and attention_mask.dim() == 4:
        return attention_mask
    sliding_window = getattr(config, "sliding_window", None) or 72
    min_val = torch.finfo(dtype).min
    causal = torch.full(
        (seq_len, seq_len), min_val, dtype=dtype, device=input_embeds.device,
    )
    causal = torch.triu(causal, diagonal=1)
    # Mask beyond sliding window
    sw_mask = torch.tril(
        torch.ones(seq_len, seq_len, dtype=dtype, device=input_embeds.device),
        diagonal=-(sliding_window + 1),
    )
    causal = causal + sw_mask * min_val
    causal = causal.unsqueeze(0).unsqueeze(0)
    if attention_mask is not None and attention_mask.dim() == 2:
        pad = (1.0 - attention_mask[:, None, None, :].to(dtype)) * min_val
        causal = causal + pad
    return causal


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_decoder(hf_model_dir: Path, models_dir: Path, skip_existing: bool = False) -> None:
    """Run speech decoder export (step 14)."""
    print("  [Step 14/15] Exporting speech decoder...")

    output_dir = models_dir / "decoder"
    output_xml = output_dir / "openvino_model.xml"

    if skip_existing and output_xml.exists():
        print(f"    Skipping: {output_xml} already exists")
        return

    speech_tokenizer_dir = hf_model_dir / "speech_tokenizer"
    num_frames = _TOTAL_CHUNK          # 75
    num_q = _NUM_QUANTIZERS            # 16
    upsample = _UPSAMPLE_RATE          # 1920
    out_samples = num_frames * upsample  # 144000

    print(f"    Source:       {speech_tokenizer_dir}")
    print(f"    Output:       {output_dir}")
    print(f"    Input shape:  codes [1, {num_q}, {num_frames}]")
    print(f"    Output shape: wav   [1, 1, {out_samples}]")

    # ------------------------------------------------------------------
    # 1. Register custom model type and load
    # ------------------------------------------------------------------
    print("    Loading speech tokenizer model...")

    from transformers import AutoConfig, AutoModel

    AutoConfig.register("qwen3_tts_tokenizer_12hz", Qwen3TTSTokenizerV2Config)
    AutoModel.register(Qwen3TTSTokenizerV2Config, Qwen3TTSTokenizerV2Model)

    model = AutoModel.from_pretrained(
        str(speech_tokenizer_dir),
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()
    print("    Model loaded OK")

    # ------------------------------------------------------------------
    # 2. Extract decoder and force eager attention
    # ------------------------------------------------------------------
    decoder = model.decoder
    for _name, module in decoder.named_modules():
        if hasattr(module, "config") and hasattr(module.config, "_attn_implementation"):
            module.config._attn_implementation = "eager"

    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"    Decoder params: {total_params:,} ({total_params * 4 / 1e6:.1f} MB fp32)")

    # ------------------------------------------------------------------
    # 3. Create static wrapper
    # ------------------------------------------------------------------
    print("    Creating StaticSpeechDecoder wrapper...")
    wrapper = StaticSpeechDecoder(decoder)
    wrapper.eval()

    # ------------------------------------------------------------------
    # 4. Verify with PyTorch forward pass
    # ------------------------------------------------------------------
    print("    Running PyTorch verification...")
    dummy_codes = torch.randint(0, 2048, (1, num_q, num_frames), dtype=torch.long)

    with torch.no_grad():
        wav = wrapper(dummy_codes)

    print(f"    Input:  {dummy_codes.shape} {dummy_codes.dtype}")
    print(f"    Output: {wav.shape} {wav.dtype}")
    print(f"    Range:  [{wav.min().item():.4f}, {wav.max().item():.4f}]")

    expected_shape = (1, 1, out_samples)
    assert wav.shape == expected_shape, (
        f"Shape mismatch! Expected {expected_shape}, got {tuple(wav.shape)}"
    )
    print("    PyTorch forward pass: OK")

    # ------------------------------------------------------------------
    # 5. Export to OpenVINO IR
    # ------------------------------------------------------------------
    import openvino as ov

    print(f"    OpenVINO version: {ov.__version__}")

    # Monkey-patch create_causal_mask to avoid vmap (not trace-friendly)
    import transformers.masking_utils
    import transformers.models.qwen3.modeling_qwen3
    from qwen_tts import modeling_qwen3_tts_tokenizer_v2

    transformers.masking_utils.create_causal_mask = _traceable_create_causal_mask
    transformers.masking_utils.create_sliding_window_causal_mask = (
        _traceable_create_sliding_window_causal_mask
    )
    transformers.models.qwen3.modeling_qwen3.create_causal_mask = (
        _traceable_create_causal_mask
    )
    modeling_qwen3_tts_tokenizer_v2.create_causal_mask = (
        _traceable_create_causal_mask
    )
    modeling_qwen3_tts_tokenizer_v2.create_sliding_window_causal_mask = (
        _traceable_create_sliding_window_causal_mask
    )
    print("    Patched create_causal_mask + create_sliding_window_causal_mask")

    print("    Converting model (this may take a minute)...")

    ov_model = ov.convert_model(
        wrapper,
        example_input=dummy_codes,
        input=ov.PartialShape([1, num_q, num_frames]),
    )

    # Name I/O tensors
    ov_model.inputs[0].set_names({"codes"})
    ov_model.outputs[0].set_names({"wav"})

    output_dir.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_model, str(output_xml), compress_to_fp16=True)

    # File sizes
    xml_size = output_xml.stat().st_size
    bin_path = output_xml.with_suffix(".bin")
    bin_size = bin_path.stat().st_size if bin_path.exists() else 0

    print(f"    XML: {output_xml} ({xml_size / 1e6:.1f} MB)")
    print(f"    BIN: {bin_path} ({bin_size / 1e6:.1f} MB)")
    print(f"    Input:  codes [1, {num_q}, {num_frames}] int64")
    print(f"    Output: wav [1, 1, {out_samples}] float32")
    print("    Speech decoder export complete.")
