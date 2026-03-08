"""Audio preprocessing: PCM -> mel spectrogram.

Uses WhisperFeatureExtractor from transformers to match Qwen3-ASR training.
"""

import numpy as np
from transformers import WhisperFeatureExtractor

from .config import HF_MODEL_DIR, SAMPLE_RATE, MEL_T_FIXED


class MelProcessor:
    """Converts PCM audio to mel spectrogram features for the encoder."""

    def __init__(self):
        self._extractor = WhisperFeatureExtractor.from_pretrained(HF_MODEL_DIR)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """Convert PCM audio to mel spectrogram, padded/trimmed to T_FIXED frames.

        Args:
            audio: 1D float32 PCM at 16kHz.

        Returns:
            Mel features, shape [1, 128, 800].
        """
        features = self._extractor(
            audio, sampling_rate=SAMPLE_RATE, padding=True, return_tensors="np"
        )
        mel = features["input_features"].astype(np.float32)  # [1, 128, T]

        # Pad or trim to exactly MEL_T_FIXED frames
        t = mel.shape[2]
        if t < MEL_T_FIXED:
            mel = np.pad(mel, ((0, 0), (0, 0), (0, MEL_T_FIXED - t)))
        elif t > MEL_T_FIXED:
            mel = mel[:, :, :MEL_T_FIXED]

        return mel
