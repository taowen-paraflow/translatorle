"""OpenVINO wrapper for the Qwen3-TTS speech decoder.

Converts 16-layer codec codes into a PCM waveform chunk using a statically
shaped OpenVINO IR model.  The decoder is intended to run on NPU but can
also be compiled for CPU or GPU.

Input:  codes [1, 16, 75] int64     (16 codebooks, 75 frames per chunk)
Output: wav   [144000]    float32   (75 frames * 1920 samples/frame = 144000)
"""

from __future__ import annotations

import numpy as np
import openvino as ov


class OVSpeechDecoder:
    """Static OpenVINO wrapper for the Qwen3-TTS speech decoder.

    Loads and compiles the decoder IR once, then exposes a simple
    ``decode_chunk`` method for converting codec codes to audio.
    """

    def __init__(self, decoder_xml: str, device: str = "NPU") -> None:
        """Load and compile the decoder IR model.

        Args:
            decoder_xml: Path to the OpenVINO IR XML file for the speech decoder.
            device:      OpenVINO device string (e.g. ``"NPU"``, ``"CPU"``, ``"GPU"``).
        """
        core = ov.Core()
        model = core.read_model(decoder_xml)
        self._compiled = core.compile_model(model, device)
        self._request = self._compiled.create_infer_request()

    def decode_chunk(self, codes: np.ndarray) -> np.ndarray:
        """Decode a chunk of codec codes to a PCM waveform.

        Args:
            codes: int64 array of shape [1, 16, 75].

        Returns:
            float32 waveform of shape [144000].
        """
        self._request.infer({0: codes})
        wav = self._request.get_output_tensor(0).data
        return wav.squeeze().copy()  # [1, 1, 144000] -> [144000]; copy to detach from OV buffer
