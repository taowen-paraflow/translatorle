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

QWEN35_MODEL_GPU = Qwen35ModelConfig(
    model_xml=str(MODELS_DIR / "Qwen3.5-0.8B-ov" / "openvino_model.xml"),
    hf_model_id="Qwen/Qwen3.5-0.8B",
    device="GPU",
)

QWEN35_MODELS = {"CPU": QWEN35_MODEL_CPU, "GPU": QWEN35_MODEL_GPU}

DEFAULT_QWEN35_MODEL = "GPU"

# ---------------------------------------------------------------------------
# NPU configuration
# ---------------------------------------------------------------------------

NPU_CACHE_DIR = MODELS_DIR / "npu_cache"

# NPU model uses Loop-free IR (seq_len=1, token-by-token prefill)
NPU_MODEL_DIR = MODELS_DIR / "Qwen3.5-0.8B-npu"

# Max KV cache positions for NPU static cache.
# Limits total context length (prompt + generation) to this value.
NPU_MAX_CACHE_LEN = 256

NPU_OV_CONFIG = {
    "NPU_USE_NPUW": "YES",
    "NPUW_FOLD": "NO",
    "CACHE_DIR": str(NPU_CACHE_DIR),
}

QWEN35_MODEL_NPU = Qwen35ModelConfig(
    model_xml=str(NPU_MODEL_DIR / "openvino_model.xml"),
    hf_model_id="Qwen/Qwen3.5-0.8B",
    device="NPU",
    ov_config=NPU_OV_CONFIG,
)

QWEN35_MODELS["NPU"] = QWEN35_MODEL_NPU

# NPUW_LLM model: KV cache managed on-device by NPUW_LLM engine
NPUW_MODEL_DIR = MODELS_DIR / "Qwen3.5-0.8B-npuw"

NPUW_LLM_OV_CONFIG = {
    "NPU_USE_NPUW": "YES",
    "NPUW_LLM": "YES",
    "NPUW_LLM_MAX_PROMPT_LEN": "128",
    "NPUW_LLM_PREFILL_HINT": "DYNAMIC",
    "NPUW_LLM_PREFILL_CHUNK_SIZE": "1",
    "NPUW_FOLD": "NO",
    "CACHE_DIR": str(NPU_CACHE_DIR),
}

QWEN35_MODEL_NPUW = Qwen35ModelConfig(
    model_xml=str(NPUW_MODEL_DIR / "openvino_model.xml"),
    hf_model_id="Qwen/Qwen3.5-0.8B",
    device="NPU",
    ov_config=NPUW_LLM_OV_CONFIG,
)

QWEN35_MODELS["NPUW"] = QWEN35_MODEL_NPUW

# ---------------------------------------------------------------------------
# Hybrid NPU+CPU configuration (NPU inference + CPU FP32 GDN state update)
# ---------------------------------------------------------------------------

HYBRID_MODEL_DIR = MODELS_DIR / "Qwen3.5-0.8B-hybrid"

HYBRID_OV_CONFIG = {
    "NPU_USE_NPUW": "YES",
    "NPUW_FOLD": "NO",
    "CACHE_DIR": str(NPU_CACHE_DIR),
}

QWEN35_MODEL_HYBRID = Qwen35ModelConfig(
    model_xml=str(HYBRID_MODEL_DIR / "openvino_model.xml"),
    hf_model_id="Qwen/Qwen3.5-0.8B",
    device="NPU",
    ov_config=HYBRID_OV_CONFIG,
)

QWEN35_MODELS["HYBRID"] = QWEN35_MODEL_HYBRID

# Hybrid model embed_tokens path
HYBRID_EMBED_TOKENS_NPY = str(HYBRID_MODEL_DIR / "embed_tokens.npy")

# ---------------------------------------------------------------------------
# Multi-subgraph NPU configuration (6 subgraphs x 4 layers each)
# ---------------------------------------------------------------------------

MULTISUB_MODEL_DIR = MODELS_DIR / "Qwen3.5-0.8B-multisub"

MULTISUB_OV_CONFIG = {
    "NPU_USE_NPUW": "YES",
    "NPUW_FOLD": "NO",
    "CACHE_DIR": str(NPU_CACHE_DIR),
}

# Number of subgraphs and layers per subgraph
MULTISUB_NUM_SUBGRAPHS = 6
MULTISUB_LAYERS_PER_SUBGRAPH = 4  # [GDN, GDN, GDN, FullAttn]

# Multi-subgraph embed_tokens path
MULTISUB_EMBED_TOKENS_NPY = str(MULTISUB_MODEL_DIR / "embed_tokens.npy")

# ---------------------------------------------------------------------------
# NPU v2 configuration (6 subgraphs with host-side rotary precomputation)
# ---------------------------------------------------------------------------

NPU_V2_MODEL_DIR = MODELS_DIR / "Qwen3.5-0.8B-npu-v2"

NPU_V2_OV_CONFIG = {
    # No NPUW — each subgraph is small enough (4 layers) for direct NPU compilation.
    # NPUW's output type handling conflicts with FP16→FP32 conversion.
    "CACHE_DIR": str(NPU_CACHE_DIR),
}


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

# NPU model embed_tokens path
NPU_EMBED_TOKENS_NPY = str(NPU_MODEL_DIR / "embed_tokens.npy")

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
