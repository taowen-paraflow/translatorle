import re

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.audio_capture import AudioCapture
from app.asr_worker import ASRWorker
from app.mt_worker import MTWorker

# Sentence-ending punctuation followed by optional whitespace
_SENTENCE_END_RE = re.compile(r"[。！？；.!?;]\s*")

# ASR prefix rollback can revise the last ~5 tokens (~10 chars).  Only treat a
# sentence boundary as confirmed when at least this many chars follow it.
_STABLE_MARGIN = 20


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._build_ui()

        # Workers
        self.audio = AudioCapture()
        self.asr = ASRWorker(self.audio.queue)
        self.asr.start()
        self.mt = MTWorker()
        self.mt.start()

        # Engine readiness tracking
        self._asr_ready = False
        self._mt_ready = False

        # Streaming translation state
        self._translated_pos = 0  # char position up to which text has been sent to MT
        self._translated_sentences = []  # list of translated sentence strings
        self._sent_sources: set[str] = set()  # source sentences already queued to MT

        # --- signal connections ---
        self.asr.engine_ready.connect(self._on_asr_ready)
        self.asr.text_updated.connect(self._on_text_updated)
        self.asr.session_finished.connect(self._on_session_finished)
        self.asr.error.connect(self._on_error)

        self.mt.engine_ready.connect(self._on_mt_ready)
        self.mt.sentence_translated.connect(self._on_sentence_translated)
        self.mt.error.connect(self._on_error)

        self.btn_record.toggled.connect(self._on_record_toggled)

        # Disable recording until both engines report ready
        self.btn_record.setEnabled(False)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        self.setWindowTitle("Translatorle")
        self.resize(700, 500)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # --- toolbar row ---
        toolbar = QHBoxLayout()

        self.btn_record = QPushButton("Record")
        self.btn_record.setCheckable(True)
        toolbar.addWidget(self.btn_record)

        toolbar.addWidget(QLabel("Translate to:"))

        self.combo_lang = QComboBox()
        self.combo_lang.addItems(
            ["Chinese", "English", "Japanese", "Korean", "Cantonese"]
        )
        toolbar.addWidget(self.combo_lang)

        layout.addLayout(toolbar)

        # --- transcription ---
        layout.addWidget(QLabel("Transcription"))
        self.txt_transcription = QTextEdit()
        self.txt_transcription.setReadOnly(True)
        layout.addWidget(self.txt_transcription)

        # --- translation ---
        layout.addWidget(QLabel("Translation"))
        self.txt_translation = QTextEdit()
        self.txt_translation.setReadOnly(True)
        layout.addWidget(self.txt_translation)

        # --- status bar ---
        self.lbl_status = QLabel("Status: Loading engines...")
        layout.addWidget(self.lbl_status)

    # ------------------------------------------------------------------
    # Engine-ready slots
    # ------------------------------------------------------------------
    @Slot()
    def _on_asr_ready(self):
        self._asr_ready = True
        self._check_ready()

    @Slot()
    def _on_mt_ready(self):
        self._mt_ready = True
        self._check_ready()

    def _check_ready(self):
        if self._asr_ready and self._mt_ready:
            self.btn_record.setEnabled(True)
            self.lbl_status.setText("Status: Ready")

    # ------------------------------------------------------------------
    # Auto-scroll helper
    # ------------------------------------------------------------------
    @staticmethod
    def _is_scrollbar_near_bottom(scrollbar: QScrollBar, threshold: int = 20) -> bool:
        """Return True if the scrollbar is within *threshold* px of the bottom."""
        return scrollbar.value() >= scrollbar.maximum() - threshold

    def _set_text_autoscroll(self, text_edit: QTextEdit, text: str):
        """Set plain text on *text_edit*, auto-scrolling only if already at bottom."""
        vbar = text_edit.verticalScrollBar()
        was_at_bottom = self._is_scrollbar_near_bottom(vbar)
        text_edit.setPlainText(text)
        if was_at_bottom:
            vbar.setValue(vbar.maximum())

    # ------------------------------------------------------------------
    # Recording slots
    # ------------------------------------------------------------------
    @Slot(bool)
    def _on_record_toggled(self, checked: bool):
        if checked:
            self.btn_record.setText("Stop")
            self.txt_transcription.clear()
            self.txt_translation.clear()
            # Reset streaming translation state
            self._translated_pos = 0
            self._translated_sentences = []
            self._sent_sources = set()
            # Start MT session for KV cache reuse across sentences
            target_lang = self.combo_lang.currentText()
            self.mt.start_session(target_lang)
            self.asr.start_session()
            self.audio.start()
            self.lbl_status.setText("Status: Recording...")
        else:
            self.btn_record.setText("Record")
            self.btn_record.setEnabled(False)  # wait for session_finished
            self.audio.stop()
            self.asr.stop_session()
            self.lbl_status.setText("Status: Processing...")

    # ------------------------------------------------------------------
    # Sentence boundary detection
    # ------------------------------------------------------------------
    def _check_new_sentences(self, text: str, flush: bool = False):
        """Detect confirmed sentences in new text and send them to MT.

        Scans *text* from ``self._translated_pos`` up to a **stable boundary**
        looking for sentence-ending punctuation.  The stable boundary is
        ``len(text) - _STABLE_MARGIN`` during streaming (ASR may still revise
        the tail) or ``len(text)`` when *flush* is True (session finished,
        text is final).

        Each confirmed sentence is dispatched to the MT worker. In session
        mode, the MT engine maintains context via KV cache automatically.
        """
        stable_end = len(text) if flush else len(text) - _STABLE_MARGIN
        if stable_end <= self._translated_pos:
            return

        region = text[self._translated_pos : stable_end]
        if not region:
            return

        last_boundary = 0
        for m in _SENTENCE_END_RE.finditer(region):
            sentence = region[last_boundary : m.end()].strip()
            last_boundary = m.end()
            if not sentence:
                continue

            # Content-based dedup: skip sentences already sent to MT.
            # This guards against _translated_pos becoming misaligned when
            # ASR prefix rollback or segment commits reorganise the text.
            if sentence in self._sent_sources:
                continue
            self._sent_sources.add(sentence)

            target_lang = self.combo_lang.currentText()
            self.mt.translate_sentence(sentence, target_lang)

        # Advance _translated_pos past the consumed sentences
        if last_boundary > 0:
            self._translated_pos += last_boundary

    # ------------------------------------------------------------------
    # ASR slots
    # ------------------------------------------------------------------
    @Slot(str)
    def _on_text_updated(self, text: str):
        self._set_text_autoscroll(self.txt_transcription, text)
        self._check_new_sentences(text)

    @Slot(str, str)
    def _on_session_finished(self, text: str, language: str):
        self._set_text_autoscroll(self.txt_transcription, text)

        # Auto-detect translation direction
        if "Chinese" in language or "\u4e2d\u6587" in language:
            idx = self.combo_lang.findText("English")
        else:
            idx = self.combo_lang.findText("Chinese")
        if idx >= 0:
            self.combo_lang.setCurrentIndex(idx)

        # Flush all remaining text (now stable since session is finished)
        self._check_new_sentences(text, flush=True)
        # Send any trailing text that didn't end with punctuation
        remaining = text[self._translated_pos :].strip()
        if remaining and remaining not in self._sent_sources:
            self._sent_sources.add(remaining)
            target_lang = self.combo_lang.currentText()
            self.mt.translate_sentence(remaining, target_lang)
            self._translated_pos = len(text)

        # End MT session (queued after all pending translations)
        self.mt.finish_session()

        self.btn_record.setEnabled(True)

    # ------------------------------------------------------------------
    # Translation slots
    # ------------------------------------------------------------------
    @Slot(str, str)
    def _on_sentence_translated(self, source: str, translation: str):
        """Handle incremental sentence translation result."""
        self._translated_sentences.append(translation)
        self._set_text_autoscroll(
            self.txt_translation, "\n".join(self._translated_sentences)
        )

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------
    @Slot(str)
    def _on_error(self, msg: str):
        self.lbl_status.setText(f"Status: Error: {msg}")
        self.btn_record.setChecked(False)
        self.btn_record.setEnabled(self._asr_ready and self._mt_ready)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def closeEvent(self, event):
        self.audio.stop()
        self.asr.shutdown()
        self.mt.shutdown()
        self.asr.wait(3000)
        self.mt.wait(3000)
        event.accept()
