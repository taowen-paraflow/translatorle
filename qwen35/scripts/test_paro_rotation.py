"""Comprehensive PARO rotation test suite for Qwen3.5-0.8B.

Runs 5 progressive tests to diagnose whether PARO rotation is applied correctly:
  Test 1: Rotation matrix properties (orthogonality, no NaN/Inf)
  Test 2: Single-layer forward equivalence (RotatedLinear vs original)
  Test 3: Full model forward equivalence (all layers rotated)
  Test 4: Per-layer error accumulation (if Test 3 fails)
  Test 5: PARO parameter mapping audit (key matching)

Run: uv run --project qwen35 python -m qwen35.scripts.test_paro_rotation
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── paths ──────────────────────────────────────────────────────────────────
PARO_MODEL_DIR = Path("models/qwen35/Qwen3.5-0.8B-PARO")
HF_MODEL_ID = "Qwen/Qwen3.5-0.8B"

# ── result tracking ────────────────────────────────────────────────────────
results = []


def report(name: str, passed: bool, detail: str = ""):
    tag = "[PASS]" if passed else "[FAIL]"
    msg = f"{tag} {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    results.append((name, passed))


# ═══════════════════════════════════════════════════════════════════════════
# Test 1: Rotation Matrix Properties (actual PARO params)
# ═══════════════════════════════════════════════════════════════════════════
def test_1_rotation_matrix_properties():
    print("\n" + "=" * 70)
    print("Test 1: Rotation Matrix Properties")
    print("=" * 70)

    from qwen35.paro_rotation import (
        build_activation_rotation_blocks,
        build_group_rotation_matrices,
        extract_paro_params,
        rotate_weight,
    )

    if not PARO_MODEL_DIR.exists():
        report("Test 1", False, f"PARO model not found at {PARO_MODEL_DIR}")
        return

    paro_params = extract_paro_params(PARO_MODEL_DIR)
    print(f"  Loaded PARO params for {len(paro_params)} layers")

    max_ortho_err_global = 0.0
    any_nan_inf = False
    cs_stats = {"min": float("inf"), "max": float("-inf"), "vals": []}

    for layer_key, params in sorted(paro_params.items()):
        theta = params["theta"]
        pairs = params["pairs"]
        cs = params["channel_scales"]

        dim = pairs.shape[1]
        R_blocks = build_group_rotation_matrices(theta, pairs, dim)

        # Check orthogonality per group
        num_groups = R_blocks.shape[0]
        for g in range(num_groups):
            RRT = R_blocks[g] @ R_blocks[g].T
            err = np.max(np.abs(RRT - np.eye(R_blocks.shape[1])))
            max_ortho_err_global = max(max_ortho_err_global, err)

        # Check NaN/Inf
        if np.any(np.isnan(R_blocks)) or np.any(np.isinf(R_blocks)):
            any_nan_inf = True
            print(f"  WARNING: NaN/Inf in R_blocks for {layer_key}")

        # Build M_blocks and check
        M_blocks = build_activation_rotation_blocks(R_blocks, cs)
        if np.any(np.isnan(M_blocks)) or np.any(np.isinf(M_blocks)):
            any_nan_inf = True
            print(f"  WARNING: NaN/Inf in M_blocks for {layer_key}")

        # Check rotated weight with dummy weight
        W_dummy = np.random.randn(16, dim).astype(np.float32)
        W_rot = rotate_weight(W_dummy, R_blocks, cs)
        if np.any(np.isnan(W_rot)) or np.any(np.isinf(W_rot)):
            any_nan_inf = True
            print(f"  WARNING: NaN/Inf in rotated weight for {layer_key}")

        # Channel scales stats
        cs_flat = cs.ravel()
        cs_stats["min"] = min(cs_stats["min"], float(cs_flat.min()))
        cs_stats["max"] = max(cs_stats["max"], float(cs_flat.max()))
        cs_stats["vals"].extend(cs_flat.tolist())

    print(f"  Max orthogonality error across all layers: {max_ortho_err_global:.2e}")
    print(f"  Channel scales: min={cs_stats['min']:.6f}, max={cs_stats['max']:.6f}, "
          f"mean={np.mean(cs_stats['vals']):.6f}, std={np.std(cs_stats['vals']):.6f}")

    report("1a: Orthogonality (R @ R^T ≈ I)", max_ortho_err_global < 1e-10,
           f"max_err={max_ortho_err_global:.2e}")
    report("1b: No NaN/Inf", not any_nan_inf)
    # Channel scales can be negative (valid for Givens rotation), but should not be
    # zero (division by zero in weight rotation) or extremely large (>1000)
    cs_abs_max = max(abs(cs_stats["min"]), abs(cs_stats["max"]))
    cs_abs_min = min(abs(v) for v in cs_stats["vals"])
    report("1c: Channel scales reasonable",
           cs_abs_min > 1e-6 and cs_abs_max < 1000.0,
           f"range [{cs_stats['min']:.4f}, {cs_stats['max']:.4f}], |min|={cs_abs_min:.6f}")


# ═══════════════════════════════════════════════════════════════════════════
# Test 2: Single-Layer Forward Equivalence (PyTorch)
# ═══════════════════════════════════════════════════════════════════════════
def test_2_single_layer_equivalence():
    print("\n" + "=" * 70)
    print("Test 2: Single-Layer Forward Equivalence")
    print("=" * 70)

    from qwen35.paro_rotation import (
        RotatedLinear,
        build_group_rotation_matrices,
        extract_paro_params,
    )

    if not PARO_MODEL_DIR.exists():
        report("Test 2", False, "PARO model not found")
        return

    paro_params = extract_paro_params(PARO_MODEL_DIR)

    # Load original model
    print("  Loading original Qwen3.5-0.8B model...")
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_ID, torch_dtype=torch.float32, trust_remote_code=True
    )
    model.eval()

    # Test several representative layers
    test_layers = [
        ("layers.0.linear_attn.in_proj_qkv", model.model.layers[0].linear_attn.in_proj_qkv),
        ("layers.0.linear_attn.out_proj", model.model.layers[0].linear_attn.out_proj),
        ("layers.0.mlp.gate_proj", model.model.layers[0].mlp.gate_proj),
        ("layers.3.self_attn.q_proj", model.model.layers[3].self_attn.q_proj),
        ("layers.3.self_attn.v_proj", model.model.layers[3].self_attn.v_proj),
        ("layers.3.mlp.down_proj", model.model.layers[3].mlp.down_proj),
    ]

    all_passed = True
    for key, linear in test_layers:
        if key not in paro_params:
            print(f"  SKIP {key}: no PARO params")
            continue

        params = paro_params[key]
        in_feat = linear.in_features
        R_blocks = build_group_rotation_matrices(
            params["theta"], params["pairs"], in_feat
        )

        rotated = RotatedLinear(linear, R_blocks, params["channel_scales"])

        # Random input, batch of 4 tokens
        x = torch.randn(1, 4, in_feat)

        with torch.no_grad():
            y_orig = linear(x)
            y_rot = rotated(x)

        max_diff = (y_orig - y_rot).abs().max().item()
        rel_diff = max_diff / y_orig.abs().max().item() if y_orig.abs().max().item() > 0 else 0
        passed = max_diff < 0.01  # FP32, single layer
        if not passed:
            all_passed = False
        print(f"  {key}: max_diff={max_diff:.2e}, rel_diff={rel_diff:.2e} {'✓' if passed else '✗ FAIL'}")

    report("Test 2: Single-layer equivalence", all_passed)

    # Return model for reuse in later tests
    return model, paro_params


# ═══════════════════════════════════════════════════════════════════════════
# Test 3: Full Model Forward Equivalence (PyTorch)
# ═══════════════════════════════════════════════════════════════════════════
def test_3_full_model_equivalence(model=None, paro_params=None):
    print("\n" + "=" * 70)
    print("Test 3: Full Model Forward Equivalence")
    print("=" * 70)

    from qwen35.paro_rotation import (
        apply_paro_rotation_to_module,
        build_group_rotation_matrices,
        extract_paro_params,
    )

    if paro_params is None:
        if not PARO_MODEL_DIR.exists():
            report("Test 3", False, "PARO model not found")
            return None, None
        paro_params = extract_paro_params(PARO_MODEL_DIR)

    if model is None:
        print("  Loading original Qwen3.5-0.8B model...")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_ID, torch_dtype=torch.float32, trust_remote_code=True
        )
        model.eval()

    # Tokenize test prompt
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, trust_remote_code=True)

    test_prompts = [
        "The capital of France is",
        "1 + 1 = ",
    ]

    # Get original logits
    print("  Running original model forward pass...")
    original_logits = {}
    with torch.no_grad():
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model(**inputs)
            original_logits[prompt] = outputs.logits.clone()

    # Apply PARO rotation to ALL layers
    print("  Applying PARO rotation to all layers...")
    import copy
    rotated_model = copy.deepcopy(model)

    total_rotated = 0
    for i, layer in enumerate(rotated_model.model.layers):
        layer_prefix = f"layers.{i}"
        n = apply_paro_rotation_to_module(layer, paro_params, layer_prefix)
        total_rotated += n
    print(f"  Rotated {total_rotated} linear layers")

    # Get rotated logits
    print("  Running rotated model forward pass...")
    all_passed = True
    with torch.no_grad():
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = rotated_model(**inputs)
            rotated_logits = outputs.logits

            orig_l = original_logits[prompt]
            max_diff = (orig_l - rotated_logits).abs().max().item()
            mean_diff = (orig_l - rotated_logits).abs().mean().item()

            # Cosine similarity of logit vectors (last token)
            v1 = orig_l[0, -1].float()
            v2 = rotated_logits[0, -1].float()
            cos_sim = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()

            # Check argmax match
            orig_next = orig_l[0, -1].argmax().item()
            rot_next = rotated_logits[0, -1].argmax().item()
            orig_tok = tokenizer.decode([orig_next])
            rot_tok = tokenizer.decode([rot_next])

            passed = max_diff < 0.5 and cos_sim > 0.99
            if not passed:
                all_passed = False
            print(f'  "{prompt}"')
            print(f"    max_diff={max_diff:.4f}, mean_diff={mean_diff:.6f}, cos_sim={cos_sim:.6f}")
            print(f"    next_token: orig={orig_next}({repr(orig_tok)}) rot={rot_next}({repr(rot_tok)}) {'match' if orig_next == rot_next else 'MISMATCH'}")

    report("Test 3: Full model equivalence", all_passed,
           f"{total_rotated} layers rotated")

    # Return rotated model info for Test 4
    return model, paro_params


# ═══════════════════════════════════════════════════════════════════════════
# Test 4: Per-Layer Error Accumulation
# ═══════════════════════════════════════════════════════════════════════════
def test_4_per_layer_error(model=None, paro_params=None):
    print("\n" + "=" * 70)
    print("Test 4: Per-Layer Error Accumulation")
    print("=" * 70)

    # Only run if Test 3 failed (or always, for diagnosis)
    from qwen35.paro_rotation import (
        apply_paro_rotation_to_module,
        extract_paro_params,
    )

    if paro_params is None:
        if not PARO_MODEL_DIR.exists():
            report("Test 4", False, "PARO model not found")
            return
        paro_params = extract_paro_params(PARO_MODEL_DIR)

    if model is None:
        print("  Loading original Qwen3.5-0.8B model...")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_ID, torch_dtype=torch.float32, trust_remote_code=True
        )
        model.eval()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, trust_remote_code=True)

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Get baseline logits
    print("  Getting baseline logits (no rotation)...")
    import copy

    with torch.no_grad():
        baseline = model(**inputs).logits.clone()

    baseline_next = baseline[0, -1].argmax().item()
    print(f"  Baseline next token: {baseline_next} ({repr(tokenizer.decode([baseline_next]))})")

    # Apply rotation one layer at a time
    print("  Applying rotation layer by layer...")
    num_layers = len(model.model.layers)
    worst_layer = None
    worst_diff = 0.0

    for i in range(num_layers):
        test_model = copy.deepcopy(model)
        layer_prefix = f"layers.{i}"
        n = apply_paro_rotation_to_module(
            test_model.model.layers[i], paro_params, layer_prefix
        )
        if n == 0:
            print(f"  Layer {i:2d}: no PARO params (skipped)")
            continue

        with torch.no_grad():
            logits = test_model(**inputs).logits

        max_diff = (baseline - logits).abs().max().item()
        mean_diff = (baseline - logits).abs().mean().item()
        next_tok = logits[0, -1].argmax().item()
        match = "match" if next_tok == baseline_next else "MISMATCH"

        if max_diff > worst_diff:
            worst_diff = max_diff
            worst_layer = i

        flag = " *** " if max_diff > 0.1 else ""
        print(f"  Layer {i:2d}: {n} rotated, max_diff={max_diff:.4f}, mean_diff={mean_diff:.6f}, "
              f"next={next_tok} {match}{flag}")

        del test_model

    if worst_layer is not None:
        print(f"\n  Worst divergence: layer {worst_layer} (max_diff={worst_diff:.4f})")

    passed = worst_diff < 0.5 if worst_layer is not None else True
    report("Test 4: Per-layer error", passed,
           f"worst layer={worst_layer}, max_diff={worst_diff:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# Test 5: PARO Parameter Mapping Audit
# ═══════════════════════════════════════════════════════════════════════════
def test_5_mapping_audit():
    print("\n" + "=" * 70)
    print("Test 5: PARO Parameter Mapping Audit")
    print("=" * 70)

    from qwen35.paro_rotation import PARO_SKIP_MODULES, extract_paro_params

    if not PARO_MODEL_DIR.exists():
        report("Test 5", False, "PARO model not found")
        return

    paro_params = extract_paro_params(PARO_MODEL_DIR)

    # Load model to get all linear layer paths
    print("  Loading model to enumerate nn.Linear modules...")
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_ID, torch_dtype=torch.float32, trust_remote_code=True
    )

    # Collect all nn.Linear paths under model.model.layers
    model_linears = {}
    for i, layer in enumerate(model.model.layers):
        for name, mod in layer.named_modules():
            if isinstance(mod, torch.nn.Linear):
                key = f"layers.{i}.{name}" if name else f"layers.{i}"
                model_linears[key] = (mod.in_features, mod.out_features)

    # Collect PARO keys
    paro_keys = set(paro_params.keys())
    model_keys = set(model_linears.keys())

    # Check what PARO provides that matches model
    matched = paro_keys & model_keys
    paro_only = paro_keys - model_keys
    model_only = model_keys - paro_keys

    print(f"\n  PARO params: {len(paro_keys)} layers")
    print(f"  Model nn.Linear: {len(model_keys)} layers")
    print(f"  Matched: {len(matched)} layers")

    if paro_only:
        print(f"\n  PARO keys NOT matching any model module ({len(paro_only)}):")
        for k in sorted(paro_only):
            print(f"    {k}")

    if model_only:
        print(f"\n  Model nn.Linear NOT in PARO ({len(model_only)}):")
        for k in sorted(model_only):
            shape = model_linears[k]
            # Check if it should be skipped
            short_name = k.split(".")[-1]
            is_skip = short_name in PARO_SKIP_MODULES or any(
                k.endswith(f".{s}") for s in PARO_SKIP_MODULES
            )
            skip_note = " (in PARO_SKIP_MODULES)" if is_skip else " *** UNEXPECTED"
            print(f"    {k}: {shape}{skip_note}")

    # Verify dimensions match
    dim_mismatches = []
    for key in matched:
        in_feat, out_feat = model_linears[key]
        cs_shape = paro_params[key]["channel_scales"].shape
        expected_dim = in_feat
        actual_dim = cs_shape[-1]
        if actual_dim != expected_dim:
            dim_mismatches.append((key, expected_dim, actual_dim))

    if dim_mismatches:
        print(f"\n  Dimension mismatches ({len(dim_mismatches)}):")
        for key, expected, actual in dim_mismatches:
            print(f"    {key}: model in_features={expected}, cs dim={actual}")

    # Check how many layers apply_paro_rotation_to_module would actually rotate
    from qwen35.paro_rotation import apply_paro_rotation_to_module
    import copy

    test_model = copy.deepcopy(model)
    actual_rotated = 0
    for i, layer in enumerate(test_model.model.layers):
        layer_prefix = f"layers.{i}"
        n = apply_paro_rotation_to_module(layer, paro_params, layer_prefix)
        actual_rotated += n
    del test_model

    print(f"\n  apply_paro_rotation_to_module would rotate: {actual_rotated} layers")
    print(f"  PARO params available: {len(paro_params)} layers")
    missed = len(matched) - actual_rotated
    if missed > 0:
        print(f"  *** {missed} matched layers NOT rotated (possible key mapping bug)")

    all_ok = (len(paro_only) == 0 and len(dim_mismatches) == 0 and missed == 0)
    report("5a: No orphan PARO keys", len(paro_only) == 0,
           f"{len(paro_only)} orphan keys")
    report("5b: No dimension mismatches", len(dim_mismatches) == 0,
           f"{len(dim_mismatches)} mismatches")
    report("5c: All matched layers rotated", missed == 0,
           f"{missed} missed")

    # Print unexpected model-only keys (not in skip list)
    unexpected = []
    for k in model_only:
        short_name = k.split(".")[-1]
        is_skip = short_name in PARO_SKIP_MODULES or any(
            k.endswith(f".{s}") for s in PARO_SKIP_MODULES
        )
        if not is_skip:
            unexpected.append(k)
    report("5d: No unexpected unrotated layers", len(unexpected) == 0,
           f"{len(unexpected)} unexpected: {unexpected[:5]}" if unexpected else "")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("PARO Rotation Test Suite for Qwen3.5-0.8B")
    print(f"PARO model: {PARO_MODEL_DIR}")
    print(f"HF model: {HF_MODEL_ID}")
    start = time.time()

    # Test 1: Rotation matrix properties
    test_1_rotation_matrix_properties()

    # Test 2: Single-layer forward equivalence (also loads model)
    ret = test_2_single_layer_equivalence()
    model, paro_params = ret if ret else (None, None)

    # Test 3: Full model forward equivalence
    model, paro_params = test_3_full_model_equivalence(model, paro_params)

    # Test 4: Per-layer error accumulation (deepcopy per layer — slow but diagnostic)
    # Only run if Test 3 failed
    test3_passed = any(name.startswith("Test 3") and passed for name, passed in results)
    if not test3_passed:
        test_4_per_layer_error(model, paro_params)
    else:
        print("\n" + "=" * 70)
        print("Test 4: Per-Layer Error Accumulation — SKIPPED (Test 3 passed)")
        print("=" * 70)

    # Free model memory before Test 5 loads a fresh copy
    del model
    import gc
    gc.collect()

    # Test 5: Mapping audit
    test_5_mapping_audit()

    # Summary
    elapsed = time.time() - start
    print("\n" + "=" * 70)
    print(f"SUMMARY ({elapsed:.1f}s)")
    print("=" * 70)
    total = len(results)
    passed = sum(1 for _, p in results if p)
    failed = total - passed
    for name, p in results:
        print(f"  {'[PASS]' if p else '[FAIL]'} {name}")
    print(f"\n  {passed}/{total} passed, {failed} failed")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
