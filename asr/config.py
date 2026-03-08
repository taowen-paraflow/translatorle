"""Configuration constants for Qwen3-ASR inference."""

from dataclasses import dataclass, field
from pathlib import Path

# Base path for model files (used by ASRModelConfig instances below)
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Audio
SAMPLE_RATE = 16000
MEL_T_FIXED = 800  # Fixed mel frames (~8 seconds of audio at hop=160)
AUDIO_PAD_COUNT = 104  # Encoder output token count

# Special token IDs (Qwen3-ASR ChatML)
IM_START = 151644
IM_END = 151645
AUDIO_START = 151669
AUDIO_END = 151670
AUDIO_PAD = 151676
ASR_TEXT = 151704
NEWLINE = 198

# Streaming defaults
CHUNK_SIZE_SEC = 2.0
UNFIXED_CHUNK_NUM = 2
UNFIXED_TOKEN_NUM = 5
MAX_NEW_TOKENS = 32
ENCODER_HOP_LENGTH = 160  # WhisperFeatureExtractor hop_length
AUDIO_WINDOW_SAMPLES = MEL_T_FIXED * ENCODER_HOP_LENGTH  # 128000 samples = ~8 seconds
MAX_PREFIX_TOKENS = 100  # Keep prefix within NPU prompt budget (256 - ~131 fixed tokens - ~25 margin)

# NPU decoder compilation config (NPUW_LLM)
NPU_DECODER_CONFIG = {
    "NPU_USE_NPUW": "YES",
    "NPUW_LLM": "YES",
    "NPUW_LLM_BATCH_DIM": 0,
    "NPUW_LLM_SEQ_LEN_DIM": 2,
    "NPUW_LLM_MAX_PROMPT_LEN": 256,
    "NPUW_LLM_MIN_RESPONSE_LEN": 64,
}


# ---------------------------------------------------------------------------
# Model configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class ASRModelConfig:
    """Paths and NPU config for a specific ASR model variant."""
    encoder_xml: str
    decoder_xml: str
    embed_table_npy: str
    hf_model_dir: str
    npu_decoder_config: dict = field(default_factory=lambda: dict(NPU_DECODER_CONFIG))
    encoder_device: str = "NPU"
    decoder_device: str = "NPU"


ASR_MODEL_0_6B = ASRModelConfig(
    encoder_xml=str(MODELS_DIR / "encoder_fp16.xml"),
    decoder_xml=str(MODELS_DIR / "decoder_stateful_embeds" / "openvino_model.xml"),
    embed_table_npy=str(MODELS_DIR / "embed_tokens.npy"),
    hf_model_dir=str(MODELS_DIR / "Qwen3-ASR-0.6B"),
    npu_decoder_config={
        "NPU_USE_NPUW": "YES",
        "NPUW_LLM": "YES",
        "NPUW_LLM_BATCH_DIM": 0,
        "NPUW_LLM_SEQ_LEN_DIM": 2,
        "NPUW_LLM_MAX_PROMPT_LEN": 256,
        "NPUW_LLM_MIN_RESPONSE_LEN": 64,
        "CACHE_DIR": str(MODELS_DIR / "asr_0.6b_cache"),
    },
)

ASR_MODEL_1_7B = ASRModelConfig(
    encoder_xml=str(MODELS_DIR / "asr_1.7b" / "encoder_fp16.xml"),
    decoder_xml=str(MODELS_DIR / "asr_1.7b" / "decoder_stateful_embeds" / "openvino_model.xml"),
    embed_table_npy=str(MODELS_DIR / "asr_1.7b" / "embed_tokens.npy"),
    hf_model_dir=str(MODELS_DIR / "asr_1.7b" / "Qwen3-ASR-1.7B"),
    npu_decoder_config={
        "NPU_USE_NPUW": "YES",
        "NPUW_LLM": "YES",
        "NPUW_LLM_BATCH_DIM": 0,
        "NPUW_LLM_SEQ_LEN_DIM": 2,
        "NPUW_LLM_MAX_PROMPT_LEN": 256,
        "NPUW_LLM_MIN_RESPONSE_LEN": 64,
        "NPUW_FOLD": "NO",                  # Skip folding (inputs_embeds breaks layer uniformity)
        "NPUW_LLM_PREFILL_HINT": "STATIC",  # Required with FOLD=NO
        "CACHE_DIR": str(MODELS_DIR / "asr_1.7b_cache"),
    },
    encoder_device="NPU",
    decoder_device="NPU",  # FOLD=NO bypasses folding assert, ~460s compile but correct output
)

ASR_MODELS = {"0.6B": ASR_MODEL_0_6B, "1.7B": ASR_MODEL_1_7B}

DEFAULT_ASR_MODEL = "1.7B"
