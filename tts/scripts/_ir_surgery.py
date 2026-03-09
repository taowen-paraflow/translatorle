"""OpenVINO IR surgery: remove input_ids, add inputs_embeds, optionally add hidden_states output.

NPUW_LLM uses name-based priority to select the main input.
If both input_ids and inputs_embeds exist, it picks input_ids (int64 zeros) -> garbage output.
Solution: remove input_ids from the model graph entirely.

This module provides a single public function ``do_ir_surgery()`` plus internal
helpers.  It is used by both the talker export and code predictor export modules.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import openvino as ov
from openvino import Dimension, PartialShape
from openvino import opset13 as opset


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_name(port) -> str:
    """Get tensor name from an OV port, returning '<unnamed>' on failure."""
    try:
        return port.any_name
    except RuntimeError:
        return "<unnamed>"


def _find_param(model, name: str):
    """Find a Parameter node by its output tensor name."""
    for op in model.get_ordered_ops():
        if op.type_info.name == "Parameter":
            try:
                if op.output(0).any_name == name:
                    return op
            except RuntimeError:
                pass
    return None


def _find_embedding_gather(input_ids_param):
    """Trace input_ids consumers to find the embedding Gather node.

    The graph is typically:  input_ids -> [Convert] -> Gather (embedding lookup)
    """
    for ti in input_ids_param.output(0).get_target_inputs():
        node = ti.get_node()
        if node.type_info.name == "Gather":
            return node
        # input_ids might pass through a Convert before the Gather
        if node.type_info.name == "Convert":
            for ti2 in node.output(0).get_target_inputs():
                n2 = ti2.get_node()
                if n2.type_info.name == "Gather":
                    return n2
    return None


def _find_hidden_state(model, vocab_size: int):
    """Find the hidden state tensor that feeds into the lm_head MatMul.

    Strategy:
      1. Trace backward from the first Result (logits) through input(0)
         until we hit a MatMul -- that is the lm_head projection.
      2. Fallback: scan all MatMul nodes for one whose last output dim
         equals vocab_size.

    Returns the Output port of the hidden state, or None.
    """
    results = model.get_results()
    if not results:
        return None

    # Primary: trace back from logits Result
    current = results[0].input(0).get_source_output().get_node()
    for _ in range(10):
        if current.type_info.name == "MatMul":
            # input(0) is the activation (hidden state), input(1) is the weight
            return current.input(0).get_source_output()
        if current.get_input_size() > 0:
            current = current.input(0).get_source_output().get_node()
        else:
            break

    # Fallback: search all MatMul nodes for lm_head by output shape
    for op in model.get_ordered_ops():
        if op.type_info.name == "MatMul":
            out_shape = op.output(0).partial_shape
            if len(out_shape) >= 1:
                last_dim = out_shape[-1]
                if last_dim.is_static and last_dim.get_length() == vocab_size:
                    return op.input(0).get_source_output()

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def do_ir_surgery(
    src_xml: str | Path,
    dst_dir: str | Path,
    hidden_size: int,
    add_hidden_output: bool = False,
    vocab_size: int = 3072,
    model_name: str = "model",
) -> None:
    """Remove input_ids input, add inputs_embeds, optionally add hidden_states output.

    Args:
        src_xml: Path to source openvino_model.xml
        dst_dir: Output directory (will be created)
        hidden_size: Model hidden size (e.g. 1024)
        add_hidden_output: If True, add a second output for hidden_states
        vocab_size: Vocabulary size for locating lm_head MatMul (default 3072
                    for talker, 1024 for Code Predictor)
        model_name: Name for the rebuilt OV Model object
    """
    src_xml = Path(src_xml)
    dst_dir = Path(dst_dir)

    if not src_xml.exists():
        raise FileNotFoundError(f"Source model not found: {src_xml}")

    core = ov.Core()

    # ------------------------------------------------------------------
    # Load the original stateful model (exported by optimum-intel)
    # ------------------------------------------------------------------
    print(f"[ir_surgery] Loading model: {src_xml}")
    model = core.read_model(str(src_xml))

    print("[ir_surgery] Original inputs:")
    for inp in model.inputs:
        print(f"  {_safe_name(inp):30s} shape={inp.partial_shape} type={inp.element_type}")

    print("[ir_surgery] Original outputs:")
    for out in model.outputs:
        print(f"  {_safe_name(out):30s} shape={out.partial_shape} type={out.element_type}")

    # ==================================================================
    # Step 1: Find input_ids Parameter and embedding Gather
    # ==================================================================
    print("\n[ir_surgery] Step 1: Find input_ids and embedding Gather")

    input_ids_param = _find_param(model, "input_ids")
    if input_ids_param is None:
        raise RuntimeError("input_ids Parameter not found in model!")
    print(f"  Found input_ids: {input_ids_param.get_friendly_name()}")

    embedding_gather = _find_embedding_gather(input_ids_param)
    if embedding_gather is None:
        raise RuntimeError("Embedding Gather node not found!")

    gather_output = embedding_gather.output(0)
    print(f"  Found Gather: {embedding_gather.get_friendly_name()}")
    print(f"  Gather output shape: {gather_output.partial_shape}")
    print(f"  Gather consumers: {len(gather_output.get_target_inputs())}")

    # ==================================================================
    # Step 2: Create inputs_embeds Parameter
    # ==================================================================
    print(f"\n[ir_surgery] Step 2: Create inputs_embeds Parameter (hidden={hidden_size})")

    embeds_shape = PartialShape([Dimension(-1), Dimension(-1), Dimension(hidden_size)])
    inputs_embeds_param = opset.parameter(
        embeds_shape, dtype=np.float32, name="inputs_embeds"
    )
    inputs_embeds_param.output(0).set_names({"inputs_embeds"})
    print(f"  Created: inputs_embeds shape={embeds_shape}")

    # ==================================================================
    # Step 3: Redirect Gather consumers to inputs_embeds
    # ==================================================================
    print("\n[ir_surgery] Step 3: Redirect Gather consumers to inputs_embeds")

    consumers = list(gather_output.get_target_inputs())
    for ti in consumers:
        ti.replace_source_output(inputs_embeds_param.output(0))
        node = ti.get_node()
        print(f"  Redirected: {node.type_info.name} '{node.get_friendly_name()}'")

    # ==================================================================
    # Step 4: Disconnect input_ids from ALL its consumers
    # ==================================================================
    print("\n[ir_surgery] Step 4: Disconnect input_ids from all consumers")

    input_ids_consumers = list(input_ids_param.output(0).get_target_inputs())
    print(f"  input_ids has {len(input_ids_consumers)} consumer(s)")
    for ti in input_ids_consumers:
        node = ti.get_node()
        print(f"  Disconnecting: {node.type_info.name} '{node.get_friendly_name()}'")
        dummy = opset.constant(np.array([[0]], dtype=np.int64))
        ti.replace_source_output(dummy.output(0))
    remaining = len(list(input_ids_param.output(0).get_target_inputs()))
    print(f"  input_ids now has {remaining} consumers")

    # ==================================================================
    # Step 5: Optionally find hidden state and add as output
    # ==================================================================
    hidden_result_node = None
    if add_hidden_output:
        print("\n[ir_surgery] Step 5: Add hidden_states output")

        hidden_source = _find_hidden_state(model, vocab_size)

        if hidden_source is not None:
            src_node = hidden_source.get_node()
            print(f"  lm_head input from: {src_node.type_info.name} '{src_node.get_friendly_name()}'")
            print(f"  Hidden state shape: {hidden_source.partial_shape}")

            # Create a Result node for the hidden state output
            hidden_result_node = opset.result(hidden_source, name="hidden_states")
            print("  Added hidden_states Result node")
        else:
            print("  WARNING: Could not find hidden state tensor!")
            print("  The model will work for logits but hidden_states output will be missing.")
    else:
        print("\n[ir_surgery] Step 5: Skipped (add_hidden_output=False)")

    # ==================================================================
    # Step 6: Build new model without input_ids
    # ==================================================================
    print(f"\n[ir_surgery] Step 6: Build new model without input_ids (name={model_name!r})")

    model.add_parameters([inputs_embeds_param])

    new_params = [
        p for p in model.get_parameters()
        if _safe_name(p.output(0)) != "input_ids"
    ]
    print(f"  Parameters: {[_safe_name(p.output(0)) for p in new_params]}")

    # Collect results: original logits + (optionally) hidden_states
    new_results = list(model.get_results())
    if hidden_result_node is not None:
        new_results.append(hidden_result_node)

    new_model = ov.Model(
        results=new_results,
        sinks=list(model.get_sinks()),
        parameters=new_params,
        name=model_name,
    )
    model = new_model

    # ==================================================================
    # Step 7: Validate
    # ==================================================================
    print("\n[ir_surgery] Step 7: Validate")
    model.validate_nodes_and_infer_types()
    print("  Validation passed!")

    print("\n[ir_surgery] Final inputs:")
    for inp in model.inputs:
        print(f"  {_safe_name(inp):30s} shape={inp.partial_shape} type={inp.element_type}")

    print("\n[ir_surgery] Final outputs:")
    for i, out in enumerate(model.outputs):
        print(f"  [{i}] {_safe_name(out):28s} shape={out.partial_shape} type={out.element_type}")

    # ==================================================================
    # Step 8: Save
    # ==================================================================
    print(f"\n[ir_surgery] Step 8: Save to {dst_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    xml_path = dst_dir / "openvino_model.xml"
    ov.save_model(model, str(xml_path))
    print(f"  Saved: {xml_path}")

    # Copy config/tokenizer files from source directory
    src_dir = src_xml.parent
    for fname in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "generation_config.json",
    ]:
        src_file = src_dir / fname
        if src_file.exists():
            shutil.copy2(str(src_file), str(dst_dir / fname))
    print("  Config files copied")

    print(f"\n[ir_surgery] Done. Output: {dst_dir}")
