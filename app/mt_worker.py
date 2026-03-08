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
        translation_done: Emitted with the translated text.
        engine_ready:     Emitted once the MT engine has finished loading.
        error:            Emitted with a message string on failure.

    Public methods (safe to call from any thread):

    * ``translate(text, target_lang)`` -- enqueue a translation request.
    * ``shutdown()`` -- terminate the worker thread.
    """

    translation_done = Signal(str)
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
        """Enqueue a translation request.

        Args:
            text: Source text to translate.
            target_lang: Target language name (e.g. ``"Chinese"``, ``"English"``).
        """
        self._cmd_queue.put(("translate", text, target_lang))

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

            engine = MTEngine(device="NPU")
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

            if isinstance(cmd, tuple) and len(cmd) == 3 and cmd[0] == "translate":
                _, text, target_lang = cmd
                try:
                    result = engine.translate(text, target_lang)
                    self.translation_done.emit(result)
                except Exception:
                    self.error.emit(
                        f"Translation failed:\n{traceback.format_exc()}"
                    )
