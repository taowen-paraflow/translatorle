"""OpenVINO Vision Encoder wrapper for Qwen3.5-VL.

Loads vision_encoder.xml and runs preprocessed image patches -> visual features.
Follows the same pattern as asr/ov_encoder.py.
"""

import logging

import numpy as np
import openvino as ov

logger = logging.getLogger(__name__)


class OVVisionEncoder:
    """Vision encoder (ViT) running on CPU/GPU via OpenVINO."""

    def __init__(self, encoder_xml: str, device: str = "CPU"):
        """Load and compile the vision encoder model.

        Args:
            encoder_xml: Path to vision_encoder.xml.
            device: OpenVINO device string ("CPU", "GPU").
        """
        core = ov.Core()
        config = {}
        if device in ("CPU", "GPU"):
            config = {"PERFORMANCE_HINT": "LATENCY"}

        self._compiled = core.compile_model(encoder_xml, device, config)

        # Discover input names from the compiled model
        self._input_names = {
            inp.any_name: inp for inp in self._compiled.inputs
        }
        logger.info(
            "OVVisionEncoder: device=%s, inputs=%s",
            device,
            list(self._input_names.keys()),
        )

    def __call__(
        self, pixel_values: np.ndarray, grid_thw: np.ndarray
    ) -> np.ndarray:
        """Run vision encoder on preprocessed image data.

        Args:
            pixel_values: Preprocessed image patches, shape varies by model
                (e.g. [num_patches, C, patch_H, patch_W] or flattened).
            grid_thw: Grid dimensions [N, 3] where each row is
                [temporal, height, width] in grid units.

        Returns:
            Visual features, shape [1, num_visual_tokens, hidden_size].
        """
        feed = {}
        if "pixel_values" in self._input_names:
            feed["pixel_values"] = pixel_values.astype(np.float32)
        if "grid_thw" in self._input_names:
            feed["grid_thw"] = grid_thw.astype(np.int64)

        # If the model has other inputs, try to pass them through
        for name in self._input_names:
            if name not in feed:
                logger.warning(
                    "OVVisionEncoder: unexpected input %r not provided", name
                )

        result = self._compiled(feed)

        # The vision encoder may have two outputs:
        #   - last_hidden_state (raw ViT output, e.g. [576, 768])
        #   - pooler_output (merged + projected, e.g. [144, 1024])
        # We need pooler_output which matches the LLM hidden dim (1024).
        outputs = {
            out.any_name: result[out] for out in self._compiled.outputs
        }
        if "pooler_output" in outputs:
            output = outputs["pooler_output"]
        elif len(outputs) > 1:
            # Fall back to the last output (pooler_output is typically last)
            output = list(outputs.values())[-1]
        else:
            output = list(outputs.values())[0]

        logger.info(
            "OVVisionEncoder output: name=%s, shape=%s",
            next(
                (k for k, v in outputs.items() if v is output),
                "unknown",
            ),
            output.shape,
        )

        # Ensure output is [1, num_tokens, hidden_size]
        if output.ndim == 2:
            output = output[np.newaxis, :, :]

        return output
