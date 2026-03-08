"""Configuration constants for HY-MT inference."""

from pathlib import Path

# Paths
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MT_MODEL_DIR = str(MODELS_DIR / "hy_mt_ov")
MT_CACHE_DIR = str(MODELS_DIR / "hy_mt_cache")

# NPU config for openvino_genai.LLMPipeline
NPU_CONFIG = {
    "MAX_PROMPT_LEN": 512,
    "NPUW_LLM_PREFILL_CHUNK_SIZE": 512,
    "GENERATE_HINT": "BEST_PERF",
}

# Generation defaults
MAX_NEW_TOKENS = 256
