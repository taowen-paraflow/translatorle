"""Configuration constants for Qwen3-TTS OpenVINO inference."""

from dataclasses import dataclass, field
from pathlib import Path

# Base path for model files (used by TTSModelConfig instances below)
MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "tts"

# ---------------------------------------------------------------------------
# Talker Architecture  (config.json -> talker_config)
# ---------------------------------------------------------------------------
TALKER_HIDDEN_SIZE = 1024
TALKER_NUM_LAYERS = 28
TALKER_NUM_HEADS = 16
TALKER_NUM_KV_HEADS = 8
TALKER_HEAD_DIM = 128
TALKER_INTERMEDIATE_SIZE = 3072
TALKER_VOCAB_SIZE = 3072          # codec token vocabulary
TALKER_TEXT_VOCAB_SIZE = 151936   # text token vocabulary
TALKER_TEXT_HIDDEN_SIZE = 2048    # thinker (text LM) hidden dim
TALKER_CODE_GROUPS = 16           # num_code_groups
TALKER_MROPE_SECTION = [24, 20, 20]
TALKER_RMS_NORM_EPS = 1e-6
TALKER_ROPE_THETA = 1_000_000
TALKER_MAX_POSITION_EMBEDDINGS = 32768
TALKER_POSITION_ID_PER_SECONDS = 13

# ---------------------------------------------------------------------------
# Code Predictor Architecture  (config.json -> talker_config -> code_predictor_config)
# ---------------------------------------------------------------------------
CP_HIDDEN_SIZE = 1024
CP_NUM_LAYERS = 5
CP_NUM_HEADS = 16
CP_NUM_KV_HEADS = 8
CP_HEAD_DIM = 128
CP_INTERMEDIATE_SIZE = 3072
CP_VOCAB_SIZE = 2048              # codebook size per group
CP_CODE_GROUPS = 16               # num_code_groups
CP_MAX_SEQ_LEN = 17               # 1 talker hidden + 16 code steps
CP_RMS_NORM_EPS = 1e-6
CP_ROPE_THETA = 1_000_000

# ---------------------------------------------------------------------------
# Speech Decoder Architecture  (speech_tokenizer/config.json -> decoder_config)
# ---------------------------------------------------------------------------
DECODER_HIDDEN_SIZE = 512
DECODER_NUM_LAYERS = 8
DECODER_NUM_HEADS = 16
DECODER_NUM_KV_HEADS = 16
DECODER_HEAD_DIM = 64
DECODER_INTERMEDIATE_SIZE = 1024
DECODER_LATENT_DIM = 1024
DECODER_CODEBOOK_DIM = 512
DECODER_CODEBOOK_SIZE = 2048
DECODER_NUM_QUANTIZERS = 16
DECODER_NUM_SEMANTIC_QUANTIZERS = 1
DECODER_UPSAMPLE_RATES = [8, 5, 4, 3]   # product = 480
DECODER_UPSAMPLING_RATIOS = [2, 2]       # waveform upsampling
DECODER_DECODE_UPSAMPLE_RATE = 1920      # total samples per code frame
DECODER_SAMPLE_RATE = 24000
DECODER_RMS_NORM_EPS = 1e-5
DECODER_ROPE_THETA = 10_000
DECODER_SLIDING_WINDOW = 72
DECODER_MAX_POSITION_EMBEDDINGS = 8000
DECODER_DECODER_DIM = 1536

# Streaming chunk sizes (in codec frames)
DECODER_CHUNK_SIZE = 50           # new frames per streaming chunk
DECODER_LEFT_CONTEXT = 25         # overlap / look-back frames
DECODER_TOTAL_CHUNK = 75          # chunk_size + left_context

# ---------------------------------------------------------------------------
# Generation Config  (generation_config.json)
# ---------------------------------------------------------------------------
TEMPERATURE = 0.9
TOP_K = 50
TOP_P = 1.0
REPETITION_PENALTY = 1.05
MAX_NEW_TOKENS = 8192

# Sub-talker (code predictor) generation
SUBTALKER_TEMPERATURE = 0.9
SUBTALKER_TOP_K = 50
SUBTALKER_TOP_P = 1.0

# ---------------------------------------------------------------------------
# Special Token IDs  (config.json top-level + tokenizer_config.json)
# ---------------------------------------------------------------------------
# ChatML structural tokens
ENDOFTEXT_TOKEN_ID = 151643       # <|endoftext|>
IM_START_TOKEN_ID = 151644        # <|im_start|>
IM_END_TOKEN_ID = 151645          # <|im_end|>
ASSISTANT_TOKEN_ID = 77091        # "assistant" role marker

# Audio boundary tokens
AUDIO_START_TOKEN_ID = 151669     # <|audio_start|>
AUDIO_END_TOKEN_ID = 151670       # <|audio_end|>
AUDIO_PAD_TOKEN_ID = 151675       # <|audio_pad|>

# TTS-specific tokens
TTS_PAD_TOKEN_ID = 151671         # <tts_pad>
TTS_BOS_TOKEN_ID = 151672         # <tts_text_bos>
TTS_EOS_TOKEN_ID = 151673         # <tts_text_eod>
TTS_BOS_SINGLE_TOKEN_ID = 151674  # <tts_text_bos_single>

# Common utility token
NEWLINE_TOKEN_ID = 198            # "\n"

# ---------------------------------------------------------------------------
# Codec Special IDs  (config.json -> talker_config)
# ---------------------------------------------------------------------------
CODEC_PAD_ID = 2148
CODEC_BOS_ID = 2149
CODEC_EOS_ID = 2150
CODEC_THINK_ID = 2154
CODEC_NOTHINK_ID = 2155
CODEC_THINK_BOS_ID = 2156
CODEC_THINK_EOS_ID = 2157

# Language IDs for the codec
CODEC_LANGUAGE_ID = {
    "chinese": 2055,
    "english": 2050,
    "german": 2053,
    "italian": 2070,
    "portuguese": 2071,
    "spanish": 2054,
    "japanese": 2058,
    "korean": 2064,
    "french": 2061,
    "russian": 2069,
    "beijing_dialect": 2074,
    "sichuan_dialect": 2062,
}

# Speaker IDs
SPEAKER_ID = {
    "serena": 3066,
    "vivian": 3065,
    "uncle_fu": 3010,
    "ryan": 3061,
    "aiden": 2861,
    "ono_anna": 2873,
    "sohee": 2864,
    "eric": 2875,
    "dylan": 2878,
}

# Speaker dialect mapping (False = standard, string = dialect name)
SPEAKER_IS_DIALECT = {
    "serena": False,
    "vivian": False,
    "uncle_fu": False,
    "ryan": False,
    "aiden": False,
    "ono_anna": False,
    "sohee": False,
    "eric": "sichuan_dialect",
    "dylan": "beijing_dialect",
}

# ---------------------------------------------------------------------------
# NPU / OpenVINO Config
# ---------------------------------------------------------------------------
NPU_TALKER_CONFIG = {
    "NPU_USE_NPUW": "YES",
    "NPUW_LLM": "YES",
    "NPUW_LLM_BATCH_DIM": 0,
    "NPUW_LLM_SEQ_LEN_DIM": 2,
    "NPUW_LLM_MAX_PROMPT_LEN": 512,
    "NPUW_LLM_MIN_RESPONSE_LEN": 128,
    "CACHE_DIR": str(MODELS_DIR / "talker_npu_cache"),
}

# INT4 on NPU: NPUW_FOLD=NO bypasses folding assertion that trips on INT4
# weight decompression subgraphs.  NPUW_LLM_PREFILL_HINT=STATIC is required
# when folding is disabled.  First compile takes ~460s; CACHE_DIR caches it.
NPU_TALKER_INT4_CONFIG = {
    "NPU_USE_NPUW": "YES",
    "NPUW_LLM": "YES",
    "NPUW_LLM_BATCH_DIM": 0,
    "NPUW_LLM_SEQ_LEN_DIM": 2,
    "NPUW_LLM_MAX_PROMPT_LEN": 512,
    "NPUW_LLM_MIN_RESPONSE_LEN": 128,
    "NPUW_FOLD": "NO",
    "NPUW_LLM_PREFILL_HINT": "STATIC",
    "CACHE_DIR": str(MODELS_DIR / "talker_npu_int4_cache"),
}


# ---------------------------------------------------------------------------
# Model configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class TTSModelConfig:
    """Paths and device config for a specific TTS model variant."""
    talker_xml: str
    lm_head_pinv_npy: str
    cp_xml: str
    cp_embeds_npz: str
    cp_lm_heads_npz: str
    cp_proj_in_npz: str
    decoder_xml: str
    hf_model_dir: str
    text_embedding_npy: str
    text_projection_npz: str
    talker_embed_tokens_npy: str
    npu_talker_config: dict = field(default_factory=lambda: dict(NPU_TALKER_CONFIG))
    talker_device: str = "NPU"
    cp_device: str = "CPU"
    decoder_device: str = "NPU"


TTS_MODEL_FP16 = TTSModelConfig(
    talker_xml=str(MODELS_DIR / "talker_npu" / "openvino_model.xml"),
    lm_head_pinv_npy=str(MODELS_DIR / "talker_lm_head_pinv.npy"),
    cp_xml=str(MODELS_DIR / "cp_stateful_embeds" / "openvino_model.xml"),
    cp_embeds_npz=str(MODELS_DIR / "cp_stateful_embeds" / "code_predictor_embeds.npz"),
    cp_lm_heads_npz=str(MODELS_DIR / "cp_stateful_embeds" / "code_predictor_lm_heads.npz"),
    cp_proj_in_npz=str(MODELS_DIR / "cp_stateful_embeds" / "code_predictor_proj_in.npz"),
    decoder_xml=str(MODELS_DIR / "decoder" / "openvino_model.xml"),
    hf_model_dir=str(MODELS_DIR),
    text_embedding_npy=str(MODELS_DIR / "talker_stateful" / "text_embedding.npy"),
    text_projection_npz=str(MODELS_DIR / "talker_stateful" / "text_projection.npz"),
    talker_embed_tokens_npy=str(MODELS_DIR / "talker_stateful" / "talker_embed_tokens.npy"),
    npu_talker_config={
        "NPU_USE_NPUW": "YES",
        "NPUW_LLM": "YES",
        "NPUW_LLM_BATCH_DIM": 0,
        "NPUW_LLM_SEQ_LEN_DIM": 2,
        "NPUW_LLM_MAX_PROMPT_LEN": 512,
        "NPUW_LLM_MIN_RESPONSE_LEN": 128,
        "CACHE_DIR": str(MODELS_DIR / "talker_npu_cache"),
    },
    talker_device="NPU",
    cp_device="CPU",
    decoder_device="NPU",
)

TTS_MODEL_INT4 = TTSModelConfig(
    talker_xml=str(MODELS_DIR / "talker_npu_int4" / "openvino_model.xml"),
    lm_head_pinv_npy=str(MODELS_DIR / "talker_lm_head_pinv.npy"),
    cp_xml=str(MODELS_DIR / "cp_stateful_embeds" / "openvino_model.xml"),
    cp_embeds_npz=str(MODELS_DIR / "cp_stateful_embeds" / "code_predictor_embeds.npz"),
    cp_lm_heads_npz=str(MODELS_DIR / "cp_stateful_embeds" / "code_predictor_lm_heads.npz"),
    cp_proj_in_npz=str(MODELS_DIR / "cp_stateful_embeds" / "code_predictor_proj_in.npz"),
    decoder_xml=str(MODELS_DIR / "decoder" / "openvino_model.xml"),
    hf_model_dir=str(MODELS_DIR),
    text_embedding_npy=str(MODELS_DIR / "talker_stateful" / "text_embedding.npy"),
    text_projection_npz=str(MODELS_DIR / "talker_stateful" / "text_projection.npz"),
    talker_embed_tokens_npy=str(MODELS_DIR / "talker_stateful" / "talker_embed_tokens.npy"),
    npu_talker_config={
        "NPU_USE_NPUW": "YES",
        "NPUW_LLM": "YES",
        "NPUW_LLM_BATCH_DIM": 0,
        "NPUW_LLM_SEQ_LEN_DIM": 2,
        "NPUW_LLM_MAX_PROMPT_LEN": 512,
        "NPUW_LLM_MIN_RESPONSE_LEN": 128,
        "NPUW_FOLD": "NO",
        "NPUW_LLM_PREFILL_HINT": "STATIC",
        "CACHE_DIR": str(MODELS_DIR / "talker_npu_int4_cache"),
    },
    talker_device="NPU",
    cp_device="CPU",
    decoder_device="NPU",
)

TTS_MODEL_CPU = TTSModelConfig(
    talker_xml=str(MODELS_DIR / "talker_stateful_embeds" / "openvino_model.xml"),
    lm_head_pinv_npy="",
    cp_xml=str(MODELS_DIR / "cp_stateful_embeds" / "openvino_model.xml"),
    cp_embeds_npz=str(MODELS_DIR / "cp_stateful_embeds" / "code_predictor_embeds.npz"),
    cp_lm_heads_npz=str(MODELS_DIR / "cp_stateful_embeds" / "code_predictor_lm_heads.npz"),
    cp_proj_in_npz=str(MODELS_DIR / "cp_stateful_embeds" / "code_predictor_proj_in.npz"),
    decoder_xml=str(MODELS_DIR / "decoder" / "openvino_model.xml"),
    hf_model_dir=str(MODELS_DIR),
    text_embedding_npy=str(MODELS_DIR / "talker_stateful" / "text_embedding.npy"),
    text_projection_npz=str(MODELS_DIR / "talker_stateful" / "text_projection.npz"),
    talker_embed_tokens_npy=str(MODELS_DIR / "talker_stateful" / "talker_embed_tokens.npy"),
    npu_talker_config={},
    talker_device="CPU",
    cp_device="CPU",
    decoder_device="CPU",
)

TTS_MODELS = {"FP16": TTS_MODEL_FP16, "INT4": TTS_MODEL_INT4, "CPU": TTS_MODEL_CPU}

DEFAULT_TTS_MODEL = "FP16"
