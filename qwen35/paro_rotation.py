"""PARO (Pairwise Rotation Quantization) rotation utilities for CPU.

Implements Givens rotation from ParoQuant (ICLR 2026) to make weights
more INT4-friendly by reducing outliers.

The rotation is block-diagonal with group_size=128 independent blocks.
For each linear layer:
  - Weight rotation:     W_rot = W @ diag(1/cs) @ R^T  (pre-computed offline)
  - Activation rotation: x_rot = x @ diag(cs) @ R^T    (computed at inference)
  - Result:              W_rot @ x_rot^T = W @ x^T      (mathematically exact)

where cs = channel_scales (stored in PARO model), R = Givens rotation matrix.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Layers that PARO does NOT rotate (from quantization_config.modules_to_not_convert)
PARO_SKIP_MODULES = {"in_proj_a", "in_proj_b", "mlp.gate", "mlp.shared_expert_gate"}


def build_group_rotation_matrices(theta, pairs, dim, group_size=128):
    """Build per-group rotation matrices from Givens parameters.

    Args:
        theta: (krot, dim//2) float array — rotation angles
        pairs: (krot, dim) int16 array — pair indices (relative to group)
        dim: input feature dimension
        group_size: rotation group size (default 128)

    Returns:
        R_blocks: (num_groups, group_size, group_size) float64 array
    """
    theta = np.asarray(theta, dtype=np.float64)
    pairs = np.asarray(pairs, dtype=np.int32)
    krot = theta.shape[0]
    num_groups = dim // group_size
    half_gs = group_size // 2

    # Build R per group independently (since groups don't interact)
    R_blocks = np.zeros((num_groups, group_size, group_size), dtype=np.float64)
    for g in range(num_groups):
        R_blocks[g] = np.eye(group_size, dtype=np.float64)

    for k in range(krot):
        for g in range(num_groups):
            R = R_blocks[g]
            for p in range(half_gs):
                i = int(pairs[k, g * group_size + 2 * p])
                j = int(pairs[k, g * group_size + 2 * p + 1])
                angle = float(theta[k, g * half_gs + p])

                cos_t = np.cos(angle)
                sin_t = np.sin(angle)

                # Left-multiply by Givens matrix: R_new = G @ R
                Ri = R[i].copy()
                Rj = R[j].copy()
                R[i] = cos_t * Ri + sin_t * Rj
                R[j] = -sin_t * Ri + cos_t * Rj

    return R_blocks


def build_activation_rotation_blocks(R_blocks, channel_scales, group_size=128):
    """Build per-group activation rotation matrices: M_g = diag(cs_g) @ R_g^T.

    Args:
        R_blocks: (num_groups, group_size, group_size) rotation matrices
        channel_scales: (dim,) channel scales from PARO model
        group_size: rotation group size

    Returns:
        M_blocks: (num_groups, group_size, group_size) float32 — activation rotation
    """
    num_groups = R_blocks.shape[0]
    cs = np.asarray(channel_scales, dtype=np.float64).ravel()
    M_blocks = np.zeros_like(R_blocks)

    for g in range(num_groups):
        cs_g = cs[g * group_size : (g + 1) * group_size]
        # M_g = diag(cs_g) @ R_g^T
        M_blocks[g] = np.diag(cs_g) @ R_blocks[g].T

    return M_blocks.astype(np.float32)


def rotate_weight(weight, R_blocks, channel_scales, group_size=128):
    """Rotate weight matrix: W_rot = W @ diag(1/cs) @ R^T (block-diagonal).

    Args:
        weight: (out_features, in_features) numpy array
        R_blocks: (num_groups, group_size, group_size) rotation matrices
        channel_scales: (in_features,) channel scales from PARO model
        group_size: rotation group size

    Returns:
        W_rot: (out_features, in_features) float32 rotated weight
    """
    out_feat, in_feat = weight.shape
    num_groups = in_feat // group_size
    cs = np.asarray(channel_scales, dtype=np.float64).ravel()
    inv_cs = 1.0 / cs

    W = np.asarray(weight, dtype=np.float64)
    W_rot = np.zeros_like(W)

    # Process per group: W_rot[:, g*gs:(g+1)*gs] = W[:, g*gs:(g+1)*gs] @ diag(inv_cs_g) @ R_g^T
    for g in range(num_groups):
        sl = slice(g * group_size, (g + 1) * group_size)
        inv_cs_g = inv_cs[sl]
        # (out, gs) @ diag(gs) @ (gs, gs) = (out, gs)
        W_scaled = W[:, sl] * inv_cs_g[np.newaxis, :]  # broadcast diag multiply
        W_rot[:, sl] = W_scaled @ R_blocks[g].T

    return W_rot.astype(np.float32)


def extract_paro_params(paro_model_dir):
    """Extract rotation parameters from PARO model safetensors.

    Args:
        paro_model_dir: path to PARO model directory

    Returns:
        dict mapping layer_key -> {theta, pairs, channel_scales}
        layer_key format: "layers.{N}.{module_path}" e.g. "layers.0.linear_attn.in_proj_qkv"
    """
    from safetensors import safe_open

    paro_dir = Path(paro_model_dir)
    params = {}

    # Find all safetensors files
    st_files = list(paro_dir.glob("*.safetensors"))
    if not st_files:
        raise FileNotFoundError(f"No safetensors files in {paro_dir}")

    for st_file in st_files:
        with safe_open(str(st_file), framework="numpy") as f:
            keys = list(f.keys())
            # Find all layers with theta params
            theta_keys = [k for k in keys if k.endswith(".theta")]

            for tk in theta_keys:
                # e.g. "model.language_model.layers.0.linear_attn.in_proj_qkv.theta"
                prefix = tk.rsplit(".theta", 1)[0]

                # Extract layer key: strip "model.language_model." prefix
                layer_key = prefix
                for strip_prefix in ("model.language_model.", "model."):
                    if layer_key.startswith(strip_prefix):
                        layer_key = layer_key[len(strip_prefix) :]
                        break

                theta = f.get_tensor(f"{prefix}.theta")
                pairs = f.get_tensor(f"{prefix}.pairs")
                channel_scales = f.get_tensor(f"{prefix}.channel_scales")

                params[layer_key] = {
                    "theta": theta.astype(np.float32),
                    "pairs": pairs.astype(np.int16),
                    "channel_scales": channel_scales.astype(np.float32),
                }

    logger.info("Extracted PARO rotation params for %d layers", len(params))
    return params


class RotatedLinear(nn.Module):
    """nn.Linear wrapper with PARO block-diagonal activation rotation.

    Stores channel_scales and R^T as separate buffers instead of fused
    M_blocks = diag(cs) @ R^T. This avoids FP16 precision loss when the
    model is saved with compress_to_fp16=True:
      - channel_scales: values up to ~21.6, FP16 precision ~0.07% (safe)
      - R_blocks_T: rotation matrix values in [-1, 1], FP16 precision ~0.1% (safe)
    The fused M_blocks mixes both ranges in a single matrix, causing error
    accumulation through 128-element dot products across 24 layers.

    During tracing, the rotation is captured as elementwise multiply +
    reshape + bmm + reshape ops in the OpenVINO IR.
    """

    def __init__(self, original_linear, R_blocks, channel_scales, group_size=128):
        """
        Args:
            original_linear: nn.Linear to wrap
            R_blocks: (num_groups, group_size, group_size) rotation matrices (numpy, float64)
            channel_scales: (in_features,) from PARO model
            group_size: rotation group size
        """
        super().__init__()
        in_feat = original_linear.in_features
        out_feat = original_linear.out_features
        num_groups = in_feat // group_size

        self.in_features = in_feat
        self.out_features = out_feat
        self.group_size = group_size
        self.num_groups = num_groups

        cs = np.asarray(channel_scales, dtype=np.float32).ravel()

        # Store channel_scales and R^T separately (FP16-safe: cs values ~20, R values in [-1,1])
        self.register_buffer(
            "channel_scales_buf", torch.from_numpy(cs).reshape(1, -1)  # (1, in_features) for broadcasting
        )
        # R_blocks_T: (num_groups, group_size, group_size), values in [-1,1]
        R_blocks_T = np.asarray(R_blocks, dtype=np.float32).transpose(0, 2, 1).copy()  # transpose last two dims
        self.register_buffer(
            "R_blocks_T", torch.from_numpy(R_blocks_T)
        )

        # Pre-rotate weight
        W = original_linear.weight.data.float().numpy()
        W_rot = rotate_weight(W, R_blocks, cs, group_size)
        self.weight = nn.Parameter(torch.from_numpy(W_rot), requires_grad=False)

        # Copy bias if present
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None

    def forward(self, x):
        shape = x.shape  # (..., dim)

        # Step 1: Channel scaling (element-wise multiply, FP16-safe)
        x_scaled = x * self.channel_scales_buf  # broadcast (1, dim) over (..., dim)

        # Step 2: Block-diagonal rotation (R^T values in [-1,1], FP16-safe)
        x_flat = x_scaled.reshape(-1, self.num_groups, self.group_size)

        # Batched MatMul: (ng, N, gs) @ (ng, gs, gs) -> (ng, N, gs)
        x_t = x_flat.permute(1, 0, 2)  # (ng, N, gs)
        x_rot = torch.bmm(x_t, self.R_blocks_T)  # (ng, N, gs)
        x_rot = x_rot.permute(1, 0, 2)  # (N, ng, gs)

        # Reshape back and apply linear
        x_out = x_rot.reshape(shape)
        return F.linear(x_out, self.weight, self.bias)


def apply_paro_rotation_to_module(module, paro_params, layer_prefix, group_size=128):
    """Replace nn.Linear submodules with RotatedLinear using PARO params.

    Args:
        module: nn.Module (e.g., a GDN layer or MLP)
        paro_params: dict from extract_paro_params()
        layer_prefix: e.g. "layers.5" — used to lookup PARO params
        group_size: rotation group size

    Returns:
        number of layers rotated
    """
    count = 0
    for name, child in list(module.named_children()):
        full_key = f"{layer_prefix}.{name}"

        # Check if this module should be skipped
        skip = False
        for skip_name in PARO_SKIP_MODULES:
            if name == skip_name or name.endswith(f".{skip_name}"):
                skip = True
                break
        if skip:
            continue

        if isinstance(child, nn.Linear) and full_key in paro_params:
            params = paro_params[full_key]
            in_feat = child.in_features
            R_blocks = build_group_rotation_matrices(
                params["theta"], params["pairs"], in_feat, group_size
            )
            rotated = RotatedLinear(child, R_blocks, params["channel_scales"], group_size)
            setattr(module, name, rotated)
            logger.info("  Rotated %s (%d, %d)", full_key, child.out_features, child.in_features)
            count += 1
        elif isinstance(child, nn.Module):
            count += apply_paro_rotation_to_module(child, paro_params, full_key, group_size)

    return count
