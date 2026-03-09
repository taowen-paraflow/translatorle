"""Configuration constants for Qwen3.5 OpenVINO inference."""

from dataclasses import dataclass, field
from pathlib import Path

# Base path for model files
MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "qwen35"

# ---------------------------------------------------------------------------
# Qwen3.5-0.8B Architecture (from config.json -> text_config)
# ---------------------------------------------------------------------------
NUM_HIDDEN_LAYERS = 24
NUM_LINEAR_ATTENTION_LAYERS = 18   # layers with linear_attention (GDN)
NUM_FULL_ATTENTION_LAYERS = 6      # layers with full_attention (standard)
HIDDEN_SIZE = 1024
NUM_ATTENTION_HEADS = 16
NUM_KEY_VALUE_HEADS = 4
HEAD_DIM = 256
INTERMEDIATE_SIZE = 2816
VOCAB_SIZE = 248064
RMS_NORM_EPS = 1e-6
ROPE_THETA = 1_000_000
MAX_POSITION_EMBEDDINGS = 32768

# Linear attention (GDN) specific
LINEAR_NUM_KEY_HEADS = 16
LINEAR_KEY_HEAD_DIM = 128
LINEAR_VALUE_HEAD_DIM = 128
LINEAR_NUM_VALUE_HEADS = 16
LINEAR_CONV_KERNEL_DIM = 4

# Layer type pattern: every 4th layer (3,7,11,15,19,23) is full_attention
LAYER_TYPES = [
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
]

# Stateful variables: 48 total
# 18 conv + 18 recurrent (linear_attention layers)
# 6 key + 6 value (full_attention layers)
NUM_STATEFUL_VARIABLES = 48

# ---------------------------------------------------------------------------
# Model configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class Qwen35ModelConfig:
    """Paths and device config for a specific Qwen3.5 model variant."""
    model_xml: str
    hf_model_id: str              # HuggingFace model ID for downloading
    device: str = "CPU"
    ov_config: dict = field(default_factory=dict)


QWEN35_MODEL_CPU = Qwen35ModelConfig(
    model_xml=str(MODELS_DIR / "Qwen3.5-0.8B-ov" / "openvino_model.xml"),
    hf_model_id="Qwen/Qwen3.5-0.8B",
    device="CPU",
)

QWEN35_MODELS = {"CPU": QWEN35_MODEL_CPU}

DEFAULT_QWEN35_MODEL = "CPU"


# ---------------------------------------------------------------------------
# Qwen3.5-VL (Vision-Language) configuration
# ---------------------------------------------------------------------------

# Export resolution for the vision encoder.  Both export_vl.py (tracing) and
# run_vl_inference.py (preprocessing) must use the same value so that the
# number of patches matches the positional embeddings baked into the IR.
VL_IMAGE_SIZE = 384

VL_MODEL_DIR = MODELS_DIR / "Qwen3.5-0.8B-vl"
VL_VISION_ENCODER_XML = str(VL_MODEL_DIR / "vision_encoder.xml")
VL_EMBED_TOKENS_NPY = str(VL_MODEL_DIR / "embed_tokens.npy")
VL_DECODER_XML = str(VL_MODEL_DIR / "openvino_model.xml")

# Text-only model embed_tokens path
EMBED_TOKENS_NPY = str(MODELS_DIR / "Qwen3.5-0.8B-ov" / "embed_tokens.npy")

QWEN35_VL_MODEL_CPU = Qwen35ModelConfig(
    model_xml=VL_DECODER_XML,
    hf_model_id="Qwen/Qwen3.5-0.8B",
    device="CPU",
)

# mRoPE (multimodal rotary position embedding) parameters
MROPE_SECTION = [11, 11, 10]    # head dim split for temporal, height, width
SPATIAL_MERGE_SIZE = 2           # ViT spatial merge factor (2x2 -> 1 token)
VL_PATCH_SIZE = 16               # ViT patch size in pixels

# VL special token IDs (from config.json)
VL_IMAGE_TOKEN_ID = 248056      # <|image_pad|>
VL_VIDEO_TOKEN_ID = 248057      # <|video_pad|>
VL_VISION_START_ID = 248053     # <|vision_start|>
VL_VISION_END_ID = 248054       # <|vision_end|>
