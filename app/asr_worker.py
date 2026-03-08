"""QThread-based ASR worker.

Runs the ASR engine on a dedicated thread, communicating with the main
(GUI) thread via Qt signals and a simple command queue.
"""

import queue
import traceback

from PySide6.QtCore import QThread, Signal


class ASRWorker(QThread):
    """Background worker that owns the ASR engine and processes audio.

    Signals:
        text_updated:     Emitted whenever the running transcription changes.
        session_finished: Emitted when a session ends (final_text, detected_language).
        engine_ready:     Emitted once the ASR engine has finished loading.
        error:            Emitted with a message string when something goes wrong.

    The worker is controlled via three public methods that may be called
    from any thread (they enqueue commands internally):

    * ``start_session()`` -- begin transcribing audio from the queue.
    * ``stop_session()``  -- finish the current session.
    * ``shutdown()``      -- terminate the worker thread.
    """

    text_updated = Signal(str)
    session_finished = Signal(str, str)
    engine_ready = Signal()
    error = Signal(str)

    def __init__(self, audio_queue: queue.Queue, parent=None) -> None:
        super().__init__(parent)
        self._audio_queue = audio_queue
        self._cmd_queue: queue.Queue[str] = queue.Queue()
        self._stop_flag = False

    # ------------------------------------------------------------------
    # Public API (called from the main / GUI thread)
    # ------------------------------------------------------------------

    def start_session(self) -> None:
        """Request the worker to start a new ASR session."""
        self._cmd_queue.put("start_session")

    def stop_session(self) -> None:
        """Request the worker to finish the current ASR session."""
        self._cmd_queue.put("stop_session")

    def shutdown(self) -> None:
        """Request the worker to exit its run loop."""
        self._stop_flag = True
        self._cmd_queue.put("quit")

    # ------------------------------------------------------------------
    # Thread entry point
    # ------------------------------------------------------------------

    def run(self) -> None:  # noqa: C901 -- intentionally linear state machine
        try:
            from asr import ASREngine

            engine = ASREngine(encoder_device="NPU", decoder_device="NPU")
        except Exception:
            self.error.emit(
                f"Failed to load ASR engine:\n{traceback.format_exc()}"
            )
            return

        self.engine_ready.emit()

        while not self._stop_flag:
            # Wait for the next command
            try:
                cmd = self._cmd_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if cmd == "quit":
                break

            if cmd == "start_session":
                self._run_session(engine)

    # ------------------------------------------------------------------
    # Session loop
    # ------------------------------------------------------------------

    def _run_session(self, engine) -> None:
        """Process audio until a ``stop_session`` or ``quit`` command."""
        try:
            state = engine.new_session()
        except Exception:
            self.error.emit(
                f"Failed to create ASR session:\n{traceback.format_exc()}"
            )
            return

        prev_text = ""

        while not self._stop_flag:
            # Check for commands (non-blocking peek)
            try:
                cmd = self._cmd_queue.get_nowait()
            except queue.Empty:
                cmd = None

            if cmd == "quit":
                self._stop_flag = True
                break

            if cmd == "stop_session":
                # Drain any remaining audio still in the queue
                self._drain_audio(engine, state)
                try:
                    engine.finish(state)
                except Exception:
                    self.error.emit(
                        f"Error finishing ASR session:\n{traceback.format_exc()}"
                    )
                self.session_finished.emit(state.text, state.language)
                return

            # Read audio from the capture queue
            try:
                chunk = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                engine.feed(chunk, state)
            except Exception:
                self.error.emit(
                    f"Error during ASR feed:\n{traceback.format_exc()}"
                )
                continue

            if state.text != prev_text:
                prev_text = state.text
                self.text_updated.emit(state.text)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _drain_audio(self, engine, state) -> None:
        """Feed any remaining audio chunks still sitting in the queue."""
        while True:
            try:
                chunk = self._audio_queue.get_nowait()
            except queue.Empty:
                break
            try:
                engine.feed(chunk, state)
            except Exception:
                pass
