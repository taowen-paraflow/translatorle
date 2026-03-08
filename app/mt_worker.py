"""QThread-based machine-translation worker.

Runs the MT engine on a dedicated thread so that translations never
block the GUI.  Communication uses Qt signals and a command queue.
"""

import queue
import traceback

from PySide6.QtCore import QThread, Signal


class MTWorker(QThread):
    """Background worker that owns the MT engine and translates text.

    Signals:
        translation_done:    Emitted with the translated text (full translation).
        sentence_translated: Emitted with (source_sentence, translated_sentence)
                             for incremental contextual updates.
        engine_ready:        Emitted once the MT engine has finished loading.
        error:               Emitted with a message string on failure.

    Public methods (safe to call from any thread):

    * ``start_session(target_lang)`` -- start a chat session for KV cache reuse.
    * ``finish_session()`` -- end the current chat session.
    * ``translate(text, target_lang)`` -- enqueue a full translation request.
    * ``translate_sentence(sentence, target_lang)`` -- enqueue an
      incremental sentence translation (uses KV cache in session mode).
    * ``shutdown()`` -- terminate the worker thread.
    """

    # Max token budget before auto-resetting the session.
    # MAX_PROMPT_LEN is 512; leave headroom for the next translation.
    _SESSION_MAX_TOKENS = 400

    translation_done = Signal(str)
    sentence_translated = Signal(str, str)
    engine_ready = Signal()
    error = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._cmd_queue: queue.Queue[tuple | str] = queue.Queue()
        self._stop_flag = False

    # ------------------------------------------------------------------
    # Public API (called from the main / GUI thread)
    # ------------------------------------------------------------------

    def start_session(self, target_lang: str) -> None:
        """Start a chat session for KV cache reuse.

        Args:
            target_lang: Target language name (e.g. ``"English"``).
        """
        self._cmd_queue.put(("start_session", target_lang))

    def finish_session(self) -> None:
        """End the current chat session and release KV cache."""
        self._cmd_queue.put(("finish_session",))

    def translate(self, text: str, target_lang: str) -> None:
        """Enqueue a full translation request.

        Args:
            text: Source text to translate.
            target_lang: Target language name (e.g. ``"Chinese"``, ``"English"``).
        """
        self._cmd_queue.put(("translate", text, target_lang))

    def translate_sentence(
        self, sentence: str, target_lang: str
    ) -> None:
        """Enqueue an incremental sentence translation.

        In session mode, uses the KV cache for context.  In stateless
        mode, translates the sentence without additional context.

        Args:
            sentence: Single sentence to translate.
            target_lang: Target language name (e.g. ``"Chinese"``, ``"English"``).
        """
        self._cmd_queue.put(("translate_sentence", sentence, target_lang))

    def shutdown(self) -> None:
        """Request the worker to exit its run loop."""
        self._stop_flag = True
        self._cmd_queue.put("quit")

    # ------------------------------------------------------------------
    # Thread entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        try:
            from hymt import MTEngine

            engine = MTEngine(device="GPU")
        except Exception:
            self.error.emit(
                f"Failed to load MT engine:\n{traceback.format_exc()}"
            )
            return

        self.engine_ready.emit()

        while not self._stop_flag:
            try:
                cmd = self._cmd_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if cmd == "quit":
                engine.finish_session()  # clean up any active session
                break

            if not isinstance(cmd, tuple):
                continue

            action = cmd[0]

            # Session management
            if action == "start_session":
                _, target_lang = cmd
                try:
                    engine.start_session(target_lang)
                except Exception:
                    self.error.emit(
                        f"Failed to start MT session:\n{traceback.format_exc()}"
                    )

            elif action == "finish_session":
                try:
                    engine.finish_session()
                except Exception:
                    self.error.emit(
                        f"Failed to finish MT session:\n{traceback.format_exc()}"
                    )

            # Full translation: ("translate", text, target_lang)
            elif action == "translate":
                _, text, target_lang = cmd
                try:
                    result = engine.translate(text, target_lang)
                    self.translation_done.emit(result)
                except Exception:
                    self.error.emit(
                        f"Translation failed:\n{traceback.format_exc()}"
                    )

            # Incremental sentence: ("translate_sentence", sentence, target_lang)
            elif action == "translate_sentence":
                _, sentence, target_lang = cmd
                try:
                    result = engine.translate(sentence, target_lang)
                    self.sentence_translated.emit(sentence, result)
                    # Auto-reset session if approaching token limit
                    if engine.needs_reset(self._SESSION_MAX_TOKENS):
                        engine.reset_session(target_lang)
                except Exception:
                    self.error.emit(
                        f"Sentence translation failed:\n{traceback.format_exc()}"
                    )
