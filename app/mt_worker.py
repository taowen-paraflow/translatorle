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

    * ``translate(text, target_lang)`` -- enqueue a full translation request.
    * ``translate_sentence(sentence, target_lang, context)`` -- enqueue an
      incremental sentence translation with surrounding context.
    * ``shutdown()`` -- terminate the worker thread.
    """

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

    def translate(self, text: str, target_lang: str) -> None:
        """Enqueue a full translation request.

        Args:
            text: Source text to translate.
            target_lang: Target language name (e.g. ``"Chinese"``, ``"English"``).
        """
        self._cmd_queue.put(("translate", text, target_lang))

    def translate_sentence(
        self, sentence: str, target_lang: str, context: str = ""
    ) -> None:
        """Enqueue an incremental sentence translation with context.

        Args:
            sentence: Single sentence to translate.
            target_lang: Target language name (e.g. ``"Chinese"``, ``"English"``).
            context: Surrounding context to improve translation quality.
        """
        self._cmd_queue.put(("translate_sentence", sentence, target_lang, context))

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
                break

            if not isinstance(cmd, tuple):
                continue

            # Legacy 2-tuple: (text, target_lang)
            if len(cmd) == 2:
                text, target_lang = cmd
                try:
                    result = engine.translate(text, target_lang)
                    self.translation_done.emit(result)
                except Exception:
                    self.error.emit(
                        f"Translation failed:\n{traceback.format_exc()}"
                    )

            # Full translation: ("translate", text, target_lang)
            elif len(cmd) == 3 and cmd[0] == "translate":
                _, text, target_lang = cmd
                try:
                    result = engine.translate(text, target_lang)
                    self.translation_done.emit(result)
                except Exception:
                    self.error.emit(
                        f"Translation failed:\n{traceback.format_exc()}"
                    )

            # Incremental sentence: ("translate_sentence", sentence, target_lang, context)
            elif len(cmd) == 4 and cmd[0] == "translate_sentence":
                _, sentence, target_lang, context = cmd
                try:
                    result = engine.translate(
                        sentence, target_lang, context=context
                    )
                    self.sentence_translated.emit(sentence, result)
                except Exception:
                    self.error.emit(
                        f"Sentence translation failed:\n{traceback.format_exc()}"
                    )
