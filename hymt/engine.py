"""Machine translation engine using HY-MT1.5-1.8B on OpenVINO.

Uses openvino_genai.LLMPipeline for fast NPU/CPU inference.
Supports stateful chat sessions for KV cache reuse across translations.
"""

import time

import openvino_genai as ov_genai

from .config import MT_MODEL_DIR, MT_CACHE_DIR, NPU_CONFIG, MAX_NEW_TOKENS


class MTEngine:
    """Machine translation engine using HY-MT on OpenVINO.

    Usage (stateless):
        engine = MTEngine(device="NPU")
        result = engine.translate("Hello world", target_lang="Chinese")
        print(result)  # "你好，世界。"

    Usage (session with KV cache reuse):
        engine = MTEngine(device="GPU")
        engine.start_session("English")
        result = engine.translate("你好世界。")     # only prefills new tokens
        result = engine.translate("今天天气很好。")  # reuses KV cache from above
        engine.finish_session()
    """

    def __init__(self, device: str = "NPU", max_new_tokens: int = MAX_NEW_TOKENS):
        self.device = device
        self.max_new_tokens = max_new_tokens

        if device == "NPU":
            config = {**NPU_CONFIG, "CACHE_DIR": MT_CACHE_DIR}
        elif device in ("CPU", "GPU"):
            config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1"}
        else:
            config = {}

        self._pipe = ov_genai.LLMPipeline(MT_MODEL_DIR, device, **config)
        self._in_session = False
        self._token_count = 0

    # ------------------------------------------------------------------
    # Session management (KV cache reuse)
    # ------------------------------------------------------------------

    def start_session(self, target_lang: str = "English") -> None:
        """Start a chat session for KV cache reuse across translations.

        Args:
            target_lang: Target language for translations in this session.
        """
        if self._in_session:
            self.finish_session()
        system_prompt = self._build_system_prompt(target_lang)
        self._pipe.start_chat(system_prompt)
        self._in_session = True
        self._token_count = len(system_prompt) // 2  # rough estimate

    def finish_session(self) -> None:
        """End the current chat session and release KV cache."""
        if self._in_session:
            self._pipe.finish_chat()
            self._in_session = False
            self._token_count = 0

    def needs_reset(self, max_tokens: int = 400) -> bool:
        """Check if the session token count is approaching the limit.

        Args:
            max_tokens: Token budget (should be less than MAX_PROMPT_LEN).

        Returns:
            True if the session should be reset to avoid exceeding the limit.
        """
        return self._in_session and self._token_count >= max_tokens

    def reset_session(self, target_lang: str = "English") -> None:
        """Reset the session: clear KV cache and start fresh.

        Args:
            target_lang: Target language for the new session.
        """
        self.finish_session()
        self.start_session(target_lang)

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------

    def translate(self, text: str, target_lang: str = "Chinese", context: str = "") -> str:
        """Translate text to the target language.

        When in a chat session, sends just the source text (the pipeline
        maintains history and reuses KV cache). When not in a session,
        falls back to stateless mode with full prompt construction.

        Args:
            text: Source text to translate.
            target_lang: Target language name (e.g. "Chinese", "English", "Japanese").
            context: Optional context for stateless mode (ignored in session mode).

        Returns:
            Translated text.
        """
        if self._in_session:
            # Chat mode: pipeline handles history/template, just send source text
            result = self._pipe.generate(text, max_new_tokens=self.max_new_tokens)
            result = result.strip()
            # Update token count estimate (input + output, ~2 chars per token for CJK)
            self._token_count += (len(text) + len(result)) // 2
            return result

        # Stateless fallback
        if context:
            prompt = self._build_context_prompt(text, target_lang, context)
        else:
            prompt = self._build_prompt(text, target_lang)
        result = self._pipe.generate(prompt, max_new_tokens=self.max_new_tokens)
        return result.strip()

    @staticmethod
    def _build_system_prompt(target_lang: str) -> str:
        """Build a system prompt for chat-mode translation sessions.

        Args:
            target_lang: Target language name.

        Returns:
            System prompt string for ``start_chat()``.
        """
        if target_lang.lower() in ("chinese", "\u4e2d\u6587"):
            return (
                "\u4f60\u662f\u4e00\u4e2a\u4e13\u4e1a\u7ffb\u8bd1\u3002"
                "\u5c06\u7528\u6237\u53d1\u9001\u7684\u6587\u672c\u7ffb\u8bd1\u4e3a\u4e2d\u6587\uff0c"
                "\u53ea\u8f93\u51fa\u7ffb\u8bd1\u7ed3\u679c\uff0c\u4e0d\u8981\u989d\u5916\u89e3\u91ca\u3002"
            )
        else:
            return (
                f"You are a professional translator. "
                f"Translate the user's text into {target_lang}. "
                f"Only output the translation, no explanation."
            )

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

    @staticmethod
    def _build_context_prompt(text: str, target_lang: str, context: str) -> str:
        """Build contextual translation prompt following HY-MT template.

        When context is provided (e.g. prior sentences or glossary), the model
        is instructed to use it as reference without translating it.

        Args:
            text: Source text to translate.
            target_lang: Target language name.
            context: Reference context for the translation.

        Returns:
            Formatted prompt string.
        """
        if not context:
            return MTEngine._build_prompt(text, target_lang)
        if target_lang.lower() in ("chinese", "中文"):
            return (
                f"{context}\n"
                f"参考上面的信息，把下面的文本翻译成中文，"
                f"注意不需要翻译上文，也不要额外解释：\n\n{text}"
            )
        else:
            return (
                f"{context}\n"
                f"Refer to the above, translate the following into {target_lang}, "
                f"without translating the above or additional explanation.\n\n{text}"
            )
