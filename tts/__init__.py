"""Standalone TTS module using Qwen3-TTS + OpenVINO."""

from .config import TTSModelConfig, TTS_MODELS, DEFAULT_TTS_MODEL
from .engine import TTSEngine

__all__ = ["TTSEngine", "TTSModelConfig", "TTS_MODELS", "DEFAULT_TTS_MODEL"]
