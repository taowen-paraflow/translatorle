"""OpenVINO Audio Encoder wrapper (NPU).

Loads encoder_fp16.xml and runs mel spectrogram -> audio features [1, 104, 1024].
"""

import numpy as np
import openvino as ov

from .config import ENCODER_XML


class OVEncoder:
    """Audio encoder running on NPU via OpenVINO."""

    def __init__(self, device: str = "NPU"):
        core = ov.Core()
        self._compiled = core.compile_model(ENCODER_XML, device)
        self._input_name = self._compiled.inputs[0].any_name

    def __call__(self, mel: np.ndarray) -> np.ndarray:
        """Run encoder on mel spectrogram.

        Args:
            mel: Mel features, shape [1, 128, T]. Will be padded/trimmed to T=800.

        Returns:
            Audio features, shape [1, 104, 1024].
        """
        result = self._compiled({self._input_name: mel})
        return list(result.values())[0]
