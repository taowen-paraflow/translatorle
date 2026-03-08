"""Microphone audio capture using sounddevice.

Provides a simple queue-based interface for streaming PCM audio
from the default input device at 16 kHz mono float32.
"""

import queue

import numpy as np

try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except (ImportError, OSError):
    sd = None  # type: ignore[assignment]
    _SD_AVAILABLE = False


class AudioCapture:
    """Captures audio from the microphone into a queue.

    Attributes:
        queue: A ``queue.Queue`` of numpy float32 arrays, each containing
            one block (100 ms / 1600 samples) of mono 16 kHz PCM audio.

    Usage::

        cap = AudioCapture()
        cap.start()
        while recording:
            chunk = cap.queue.get(timeout=1.0)
            process(chunk)
        cap.stop()
    """

    SAMPLE_RATE = 16_000
    CHANNELS = 1
    DTYPE = "float32"
    BLOCKSIZE = 1600  # 100 ms at 16 kHz

    def __init__(self) -> None:
        self.queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: "sd.InputStream | None" = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the microphone stream and begin filling the queue.

        Raises:
            RuntimeError: If the ``sounddevice`` package is not available.
        """
        if not _SD_AVAILABLE:
            raise RuntimeError(
                "sounddevice is not available. "
                "Install it with: pip install sounddevice"
            )

        self._stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=self.CHANNELS,
            dtype=self.DTYPE,
            blocksize=self.BLOCKSIZE,
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self) -> None:
        """Stop and close the microphone stream."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            finally:
                self._stream = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: "sd.CallbackFlags",
    ) -> None:
        """sounddevice stream callback -- runs on the audio thread."""
        self.queue.put(indata[:, 0].copy())

    @staticmethod
    def is_available() -> bool:
        """Return whether sounddevice is importable and functional."""
        return _SD_AVAILABLE
