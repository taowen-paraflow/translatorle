"""OpenVINO Audio Encoder wrapper.

Loads encoder_fp16.xml and runs mel spectrogram -> audio features [1, 104, 1024].
Supports NPU, CPU, and GPU devices.
"""

import numpy as np
import openvino as ov

from .config import ENCODER_XML


class OVEncoder:
    """Audio encoder running on NPU/CPU/GPU via OpenVINO."""

    def __init__(self, device: str = "NPU"):
        core = ov.Core()
        config = {}
        if device in ("CPU", "GPU"):
            config = {"PERFORMANCE_HINT": "LATENCY"}
        self._compiled = core.compile_model(ENCODER_XML, device, config)
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
