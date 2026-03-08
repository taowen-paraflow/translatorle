from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.audio_capture import AudioCapture
from app.asr_worker import ASRWorker
from app.mt_worker import MTWorker


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

        # --- signal connections ---
        self.asr.engine_ready.connect(self._on_asr_ready)
        self.asr.text_updated.connect(self._on_text_updated)
        self.asr.session_finished.connect(self._on_session_finished)
        self.asr.error.connect(self._on_error)

        self.mt.engine_ready.connect(self._on_mt_ready)
        self.mt.translation_done.connect(self._on_translation_done)
        self.mt.error.connect(self._on_error)

        self.btn_record.toggled.connect(self._on_record_toggled)
        self.btn_translate.clicked.connect(self._on_translate_clicked)

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

        self.btn_translate = QPushButton("Translate")
        self.btn_translate.setEnabled(False)
        toolbar.addWidget(self.btn_translate)

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
    # Recording slots
    # ------------------------------------------------------------------
    @Slot(bool)
    def _on_record_toggled(self, checked: bool):
        if checked:
            self.btn_record.setText("Stop")
            self.btn_translate.setEnabled(False)
            self.txt_transcription.clear()
            self.txt_translation.clear()
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
    # ASR slots
    # ------------------------------------------------------------------
    @Slot(str)
    def _on_text_updated(self, text: str):
        self.txt_transcription.setPlainText(text)

    @Slot(str, str)
    def _on_session_finished(self, text: str, language: str):
        self.txt_transcription.setPlainText(text)

        # Auto-detect translation direction
        if "Chinese" in language or "\u4e2d\u6587" in language:
            idx = self.combo_lang.findText("English")
        else:
            idx = self.combo_lang.findText("Chinese")
        if idx >= 0:
            self.combo_lang.setCurrentIndex(idx)

        # Auto-translate if there is content
        if text.strip():
            self.mt.translate(text, self.combo_lang.currentText())
            self.lbl_status.setText("Status: Translating...")

        self.btn_record.setEnabled(True)

    # ------------------------------------------------------------------
    # Translation slots
    # ------------------------------------------------------------------
    @Slot()
    def _on_translate_clicked(self):
        text = self.txt_transcription.toPlainText()
        if text.strip():
            self.mt.translate(text, self.combo_lang.currentText())
            self.btn_translate.setEnabled(False)
            self.lbl_status.setText("Status: Translating...")

    @Slot(str)
    def _on_translation_done(self, text: str):
        self.txt_translation.setPlainText(text)
        self.btn_translate.setEnabled(True)
        self.lbl_status.setText("Status: Ready")

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------
    @Slot(str)
    def _on_error(self, msg: str):
        self.lbl_status.setText(f"Status: Error: {msg}")
        self.btn_record.setChecked(False)
        self.btn_record.setEnabled(self._asr_ready and self._mt_ready)
        self.btn_translate.setEnabled(True)

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
