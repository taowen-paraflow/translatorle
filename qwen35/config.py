"""Configuration constants for Qwen3.5 OpenVINO inference."""

from dataclasses import dataclass, field
from pathlib import Path

# Base path for model files
MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "qwen35"

# ---------------------------------------------------------------------------
# Active model size selection — change this to switch between 0.8B and 4B.
# Used only for the top-level convenience constants below; inference.py and
# export.py read architecture parameters from the model's own config.json.
# ---------------------------------------------------------------------------
MODEL_SIZE = "0.8B"   # "0.8B" or "4B"

# ---------------------------------------------------------------------------
# Architecture specs per model size (from config.json -> text_config)
# ---------------------------------------------------------------------------

def _build_layer_types(num_hidden_layers: int, full_attention_interval: int = 4):
    """Build the layer_types list: every *interval*-th layer is full_attention."""
    return [
        "full_attention" if (i + 1) % full_attention_interval == 0 else "linear_attention"
        for i in range(num_hidden_layers)
    ]


ARCH_CONFIGS = {
    "0.8B": {
        "hidden_size": 1024,
        "intermediate_size": 2816,
        "num_hidden_layers": 24,
        "vocab_size": 248064,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000,
        "max_position_embeddings": 32768,
        # Linear attention (GDN) specific
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        # Derived
        "full_attention_interval": 4,
        "num_linear_attention_layers": 18,
        "num_full_attention_layers": 6,
        "num_stateful_variables": 48,   # 18*2 + 6*2
        # IDs
        "hf_model_id": "Qwen/Qwen3.5-0.8B",
        "ov_dir_name": "Qwen3.5-0.8B-ov",
        "vl_dir_name": "Qwen3.5-0.8B-vl",
    },
    "4B": {
        "hidden_size": 2560,
        "intermediate_size": 9216,
        "num_hidden_layers": 32,
        "vocab_size": 248320,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000,
        "max_position_embeddings": 32768,
        # Linear attention (GDN) specific
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        # Derived
        "full_attention_interval": 4,
        "num_linear_attention_layers": 24,
        "num_full_attention_layers": 8,
        "num_stateful_variables": 64,   # 24*2 + 8*2
        # IDs
        "hf_model_id": "Qwen/Qwen3.5-4B",
        "ov_dir_name": "Qwen3.5-4B-ov",
        "vl_dir_name": "Qwen3.5-4B-vl",
    },
}

# ---------------------------------------------------------------------------
# Top-level convenience constants (resolve from MODEL_SIZE for backward compat)
# ---------------------------------------------------------------------------

def _get_arch(key: str):
    return ARCH_CONFIGS[MODEL_SIZE][key]

NUM_HIDDEN_LAYERS       = _get_arch("num_hidden_layers")
NUM_LINEAR_ATTENTION_LAYERS = _get_arch("num_linear_attention_layers")
NUM_FULL_ATTENTION_LAYERS   = _get_arch("num_full_attention_layers")
HIDDEN_SIZE             = _get_arch("hidden_size")
NUM_ATTENTION_HEADS     = _get_arch("num_attention_heads")
NUM_KEY_VALUE_HEADS     = _get_arch("num_key_value_heads")
HEAD_DIM                = _get_arch("head_dim")
INTERMEDIATE_SIZE       = _get_arch("intermediate_size")
VOCAB_SIZE              = _get_arch("vocab_size")
RMS_NORM_EPS            = _get_arch("rms_norm_eps")
ROPE_THETA              = _get_arch("rope_theta")
MAX_POSITION_EMBEDDINGS = _get_arch("max_position_embeddings")

LINEAR_NUM_KEY_HEADS    = _get_arch("linear_num_key_heads")
LINEAR_KEY_HEAD_DIM     = _get_arch("linear_key_head_dim")
LINEAR_VALUE_HEAD_DIM   = _get_arch("linear_value_head_dim")
LINEAR_NUM_VALUE_HEADS  = _get_arch("linear_num_value_heads")
LINEAR_CONV_KERNEL_DIM  = _get_arch("linear_conv_kernel_dim")

LAYER_TYPES = _build_layer_types(NUM_HIDDEN_LAYERS, _get_arch("full_attention_interval"))
NUM_STATEFUL_VARIABLES  = _get_arch("num_stateful_variables")

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


def _ov_dir(size: str) -> Path:
    return MODELS_DIR / ARCH_CONFIGS[size]["ov_dir_name"]


# 0.8B models
QWEN35_08B_CPU = Qwen35ModelConfig(
    model_xml=str(_ov_dir("0.8B") / "openvino_model.xml"),
    hf_model_id="Qwen/Qwen3.5-0.8B",
    device="CPU",
)
QWEN35_08B_GPU = Qwen35ModelConfig(
    model_xml=str(_ov_dir("0.8B") / "openvino_model.xml"),
    hf_model_id="Qwen/Qwen3.5-0.8B",
    device="GPU",
)

# 4B models
QWEN35_4B_CPU = Qwen35ModelConfig(
    model_xml=str(_ov_dir("4B") / "openvino_model.xml"),
    hf_model_id="Qwen/Qwen3.5-4B",
    device="CPU",
)
QWEN35_4B_GPU = Qwen35ModelConfig(
    model_xml=str(_ov_dir("4B") / "openvino_model.xml"),
    hf_model_id="Qwen/Qwen3.5-4B",
    device="GPU",
)

# Legacy aliases (resolve from MODEL_SIZE)
QWEN35_MODEL_CPU = QWEN35_08B_CPU if MODEL_SIZE == "0.8B" else QWEN35_4B_CPU
QWEN35_MODEL_GPU = QWEN35_08B_GPU if MODEL_SIZE == "0.8B" else QWEN35_4B_GPU

QWEN35_MODELS = {
    # Keyed by "<size>-<device>" for explicit selection
    "0.8B-CPU": QWEN35_08B_CPU,
    "0.8B-GPU": QWEN35_08B_GPU,
    "4B-CPU":   QWEN35_4B_CPU,
    "4B-GPU":   QWEN35_4B_GPU,
    # Legacy keys (resolve from MODEL_SIZE)
    "CPU": QWEN35_MODEL_CPU,
    "GPU": QWEN35_MODEL_GPU,
}

DEFAULT_QWEN35_MODEL = "GPU"


# ---------------------------------------------------------------------------
# Qwen3.5-VL (Vision-Language) configuration
# ---------------------------------------------------------------------------

# Export resolution for the vision encoder.  Both export_vl.py (tracing) and
# run_vl_inference.py (preprocessing) must use the same value so that the
# number of patches matches the positional embeddings baked into the IR.
VL_IMAGE_SIZE = 384

VL_MODEL_DIR = MODELS_DIR / ARCH_CONFIGS[MODEL_SIZE]["vl_dir_name"]
VL_VISION_ENCODER_XML = str(VL_MODEL_DIR / "vision_encoder.xml")
VL_EMBED_TOKENS_NPY = str(VL_MODEL_DIR / "embed_tokens.npy")
VL_DECODER_XML = str(VL_MODEL_DIR / "openvino_model.xml")

# Text-only model embed_tokens path
EMBED_TOKENS_NPY = str(_ov_dir(MODEL_SIZE) / "embed_tokens.npy")

QWEN35_VL_MODEL_CPU = Qwen35ModelConfig(
    model_xml=VL_DECODER_XML,
    hf_model_id=ARCH_CONFIGS[MODEL_SIZE]["hf_model_id"],
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
