"""Configuration constants for Qwen3-ASR inference."""

from pathlib import Path

# Paths
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
HF_MODEL_DIR = str(MODELS_DIR / "Qwen3-ASR-0.6B")

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

# Model file paths
ENCODER_XML = str(MODELS_DIR / "encoder_fp16.xml")
DECODER_DIR = str(MODELS_DIR / "decoder_stateful_embeds")
DECODER_XML = str(MODELS_DIR / "decoder_stateful_embeds" / "openvino_model.xml")
EMBED_TABLE_NPY = str(MODELS_DIR / "embed_tokens.npy")

# NPU decoder compilation config (NPUW_LLM)
NPU_DECODER_CONFIG = {
    "NPU_USE_NPUW": "YES",
    "NPUW_LLM": "YES",
    "NPUW_LLM_BATCH_DIM": 0,
    "NPUW_LLM_SEQ_LEN_DIM": 2,
    "NPUW_LLM_MAX_PROMPT_LEN": 256,
    "NPUW_LLM_MIN_RESPONSE_LEN": 64,
}
