"""Machine translation engine using HY-MT1.5-1.8B on OpenVINO.

Uses openvino_genai.LLMPipeline for fast NPU/CPU inference.
"""

import time

import openvino_genai as ov_genai

from .config import MT_MODEL_DIR, MT_CACHE_DIR, NPU_CONFIG, MAX_NEW_TOKENS


class MTEngine:
    """Machine translation engine using HY-MT on OpenVINO.

    Usage:
        engine = MTEngine(device="NPU")
        result = engine.translate("Hello world", target_lang="Chinese")
        print(result)  # "你好，世界。"
    """

    def __init__(self, device: str = "NPU", max_new_tokens: int = MAX_NEW_TOKENS):
        self.device = device
        self.max_new_tokens = max_new_tokens

        if device == "NPU":
            config = {**NPU_CONFIG, "CACHE_DIR": MT_CACHE_DIR}
            self._pipe = ov_genai.LLMPipeline(MT_MODEL_DIR, device, **config)
        else:
            self._pipe = ov_genai.LLMPipeline(MT_MODEL_DIR, device)

    def translate(self, text: str, target_lang: str = "Chinese") -> str:
        """Translate text to the target language.

        Args:
            text: Source text to translate.
            target_lang: Target language name (e.g. "Chinese", "English", "Japanese").

        Returns:
            Translated text.
        """
        prompt = self._build_prompt(text, target_lang)
        result = self._pipe.generate(prompt, max_new_tokens=self.max_new_tokens)
        return result.strip()

    @staticmethod
    def _build_prompt(text: str, target_lang: str) -> str:
        """Build translation prompt following HY-MT template."""
        if target_lang.lower() in ("chinese", "中文"):
            return (
                f"将以下文本翻译为中文，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}"
            )
        else:
            return (
                f"Translate the following segment into {target_lang}, "
                f"without additional explanation.\n\n{text}"
            )
