"""Post-process GDN noloop IR to replace recurrence subgraph with FusedGDNRecurrence custom op.

Reads gdn_noloop_block_*.xml, identifies the recurrence ops (by name pattern),
replaces them with a single FusedGDNRecurrence layer, and saves as gdn_fused_block_*.xml.

The FusedGDNRecurrence custom op is dispatched by the OpenVINO GPU plugin via
CONFIG_FILE + OpenCL kernel (gdn_recurrence.cl).

Usage:
  uv run python -m qwen35.fuse_gdn_ir --model-dir models/qwen35/Qwen3.5-0.8B-hybrid

This reads gdn_noloop_block_{0-5}.xml and writes gdn_fused_block_{0-5}.xml in the same dir.
"""

import argparse
import copy
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path

logger = logging.getLogger(__name__)

# Pattern to identify recurrence ops by their name attribute.
# These ops come from SingleStepRecurrentAttentionCell tracing.
# Name format: __module.layers.{N}.linear_attn.recurrent_attention_cell/aten::op/OpName
RECURRENCE_NAME_PATTERN = re.compile(
    r'__module\.layers\.(\d+)\.linear_attn\.recurrent_attention_cell/'
)

# Ops that are part of the recurrence but might not have the recurrence name.
# These are reshape/concat ops that combine the recurrence output with state.
RECURRENCE_OUTPUT_OPS = {'Reshape', 'Concat', 'Squeeze', 'Unsqueeze'}


def build_graph(root):
    """Build adjacency lists and layer lookup from IR XML."""
    layers = {}  # id -> {type, name, element}
    out_edges = {}  # from_id -> [(to_id, from_port, to_port)]
    in_edges = {}   # to_id -> [(from_id, from_port, to_port)]

    for layer in root.findall('.//layers/layer'):
        lid = layer.get('id')
        layers[lid] = {
            'type': layer.get('type'),
            'name': layer.get('name', ''),
            'elem': layer,
        }

    for edge in root.findall('.//edges/edge'):
        fl = edge.get('from-layer')
        fp = edge.get('from-port')
        tl = edge.get('to-layer')
        tp = edge.get('to-port')
        out_edges.setdefault(fl, []).append((tl, fp, tp))
        in_edges.setdefault(tl, []).append((fl, fp, tp))

    return layers, out_edges, in_edges


def find_recurrence_ops(layers, out_edges, in_edges, layer_idx):
    """Find all op IDs that form layer_idx's recurrence subgraph.

    Returns:
        rec_op_ids: set of layer IDs to remove
        inputs: dict mapping input role to (source_layer_id, source_port)
        outputs: dict mapping output role to (layer_id, port)
    """
    rec_ops = set()
    pattern = f'__module.layers.{layer_idx}.linear_attn.recurrent_attention_cell/'

    # Find all ops with recurrence name pattern for this layer
    for lid, info in layers.items():
        if pattern in info['name']:
            rec_ops.add(lid)

    if not rec_ops:
        return None, None, None

    logger.info("  Layer %d: found %d recurrence ops", layer_idx, len(rec_ops))

    # Find the recurrence state input (Parameter node named recurrent_state_{local_idx})
    # The local_idx within the block is the position (0, 1, or 2)
    state_param_id = None
    for lid, info in layers.items():
        if info['type'] == 'Parameter' and f'recurrent_state_{layer_idx}' == info['name']:
            state_param_id = lid
            break

    # Identify recurrence inputs by tracing edges INTO the recurrence ops
    # from non-recurrence sources
    rec_inputs = {}  # {role: (source_layer_id, source_port)}
    for rop in rec_ops:
        for src_id, src_port, dst_port in in_edges.get(rop, []):
            if src_id not in rec_ops:
                # This is an external input to the recurrence
                src_info = layers.get(src_id, {})
                if src_info.get('name', '') == f'recurrent_state_{layer_idx}':
                    rec_inputs['state'] = (src_id, src_port)
                else:
                    # Determine role from the recurrence op name
                    rop_info = layers[rop]
                    rop_name = rop_info['name']
                    # The first Multiply takes state * decay -> the other input is decay
                    # Other inputs: q, k, v come from projection ops
                    if src_id not in rec_inputs.values():
                        # We'll figure out roles from shapes and op patterns
                        pass

    # Instead of complex role identification, find inputs by tracing the data flow.
    # The recurrence cell receives: q, k, v, g, beta, state
    # From SingleStepRecurrentAttentionCell.forward():
    #   decay = exp(g_t).unsqueeze(-1).unsqueeze(-1)  -> feeds first Multiply
    #   k_unsq = k_t.unsqueeze(-1)                    -> feeds second Multiply and rank-1 update
    #   kv_mem = (state * k_unsq).sum(dim=-2)          -> ReduceSum
    #   delta = (v_t - kv_mem) * beta_t.unsqueeze(-1)  -> Subtract then Multiply
    #   state_new = state + k_unsq * delta.unsqueeze(-2)
    #   q_unsq = q_t.unsqueeze(-1)
    #   out_t = (state * q_unsq).sum(dim=-2)

    # Find the Multiply that takes state as input (the decay multiply)
    decay_mul_id = None
    for rop in rec_ops:
        info = layers[rop]
        if info['type'] == 'Multiply':
            # Check if one of its inputs is the state
            for src_id, src_port, dst_port in in_edges.get(rop, []):
                if src_id == state_param_id:
                    decay_mul_id = rop
                    break
        if decay_mul_id:
            break

    if not decay_mul_id:
        logger.warning("  Could not find decay multiply for layer %d", layer_idx)
        return None, None, None

    # The other input to decay_mul is the decay value (from exp(g).unsqueeze.unsqueeze)
    decay_source = None
    for src_id, src_port, dst_port in in_edges.get(decay_mul_id, []):
        if src_id != state_param_id:
            decay_source = (src_id, src_port)
            break

    # Find all external inputs to the recurrence subgraph
    external_inputs = []
    for rop in rec_ops:
        for src_id, src_port, dst_port in in_edges.get(rop, []):
            if src_id not in rec_ops:
                external_inputs.append({
                    'src_id': src_id, 'src_port': src_port,
                    'dst_id': rop, 'dst_port': dst_port,
                    'src_type': layers.get(src_id, {}).get('type', ''),
                    'src_name': layers.get(src_id, {}).get('name', ''),
                })

    # Find all external outputs from the recurrence subgraph
    external_outputs = []
    for rop in rec_ops:
        for dst_id, from_port, to_port in out_edges.get(rop, []):
            if dst_id not in rec_ops:
                external_outputs.append({
                    'src_id': rop, 'src_port': from_port,
                    'dst_id': dst_id, 'dst_port': to_port,
                    'dst_type': layers.get(dst_id, {}).get('type', ''),
                    'dst_name': layers.get(dst_id, {}).get('name', ''),
                })

    return rec_ops, external_inputs, external_outputs


def find_recurrence_io(layers, out_edges, in_edges, rec_ops, block_layer_idx):
    """Determine the actual q, k, v, g, beta, state inputs and attn_out, state_out outputs.

    This traces the edges to identify which external inputs map to which recurrence roles.

    Returns:
        inputs: list of (layer_id, port, role) for q, k, v, g, beta, state
        outputs: list of (target_layer_id, target_port, role) for attn_output, state_output
    """
    # Collect all edges from outside into the recurrence
    ext_ins = []
    for rop in rec_ops:
        for src_id, src_port, dst_port in in_edges.get(rop, []):
            if src_id not in rec_ops:
                rop_name = layers[rop]['name']
                ext_ins.append((src_id, src_port, rop, dst_port, rop_name))

    # Collect all edges from recurrence to outside
    ext_outs = []
    for rop in rec_ops:
        for dst_id, from_port, to_port in out_edges.get(rop, []):
            if dst_id not in rec_ops:
                rop_name = layers[rop]['name']
                ext_outs.append((rop, from_port, dst_id, to_port, rop_name))

    # Identify inputs by analyzing which recurrence op they feed into:
    #
    # The recurrence has this structure (from SingleStepRecurrentAttentionCell):
    #   decay = exp(g_t).unsqueeze(-1).unsqueeze(-1)   -- decay_val
    #   state_decayed = state * decay                   -- Multiply (takes state + decay_val)
    #   k_unsq = k_t.unsqueeze(-1)                     -- Unsqueeze
    #   state_k = state_decayed * k_unsq                -- Multiply_1 (takes decayed + k_unsq)
    #   kv_mem = sum(state_k, dim=-2)                   -- ReduceSum
    #   error = v_t - kv_mem                            -- Subtract (takes v_t + kv_mem)
    #   delta = error * beta_unsq                       -- Multiply_2 (takes error + beta_unsq)
    #   delta_unsq = delta.unsqueeze(-2)                -- Unsqueeze_4
    #   rank1 = k_unsq * delta_unsq                     -- Multiply_3 (takes k_unsq + delta_unsq)
    #   new_state = state_decayed + rank1               -- Add
    #   state_q = new_state * q_unsq                    -- Multiply_4 (takes new_state + q_unsq)
    #   attn_out = sum(state_q, dim=-2)                 -- ReduceSum_1
    #
    # So external inputs are:
    #   1. state -> Multiply (input port where state connects)
    #   2. decay_val (from exp+unsqueeze chain) -> Multiply (other input)
    #   3. k_unsq -> Multiply_1 and Multiply_3
    #   4. v_t -> Subtract
    #   5. beta_unsq -> Multiply_2
    #   6. q_unsq -> Multiply_4
    #   7. reduce axes (constants) -> ReduceSum, ReduceSum_1

    # Group inputs by the recurrence op they feed
    inputs_by_role = {
        'state': None,   # [1, 16, 128, 128]
        'decay': None,   # [1, 16, 1, 1]
        'k_unsq': None,  # [1, 16, 128, 1]
        'v': None,       # [1, 16, 128]
        'beta': None,    # [1, 16, 1]
        'q_unsq': None,  # [1, 16, 128, 1]
    }

    # Find state input: it's the Parameter node named recurrent_state_{block_layer_idx}
    state_id = None
    for lid, info in layers.items():
        if info['type'] == 'Parameter' and info['name'] == f'recurrent_state_{block_layer_idx}':
            state_id = lid
            break

    if state_id:
        inputs_by_role['state'] = (state_id, '0')

    # For non-state inputs, trace by op name pattern
    for src_id, src_port, rop_id, dst_port, rop_name in ext_ins:
        if src_id == state_id:
            continue
        src_type = layers.get(src_id, {}).get('type', '')
        if src_type in ('Const', 'Convert'):
            # Skip constants (reduce axes, etc.)
            continue

        # Determine role by looking at which named recurrence op this feeds
        short_name = rop_name.split('/')[-1] if '/' in rop_name else rop_name
        parent_ops = rop_name.split('/')

        # The Multiply_4 takes q_unsq -> output query
        if 'Multiply_4' in rop_name or 'Multiply_4' in short_name:
            inputs_by_role['q_unsq'] = (src_id, src_port)
        # The Multiply_1 takes k_unsq -> state @ k
        elif 'Multiply_1' in rop_name:
            inputs_by_role['k_unsq'] = (src_id, src_port)
        # The Multiply (without number) takes decay -> state decay
        elif short_name == 'Multiply' or 'aten::mul/Multiply' in rop_name:
            if src_id != state_id:
                inputs_by_role['decay'] = (src_id, src_port)
        # The Subtract takes v
        elif 'Subtract' in rop_name:
            inputs_by_role['v'] = (src_id, src_port)
        # The Multiply_2 takes beta
        elif 'Multiply_2' in rop_name:
            inputs_by_role['beta'] = (src_id, src_port)

    return inputs_by_role, ext_outs


def replace_recurrence_in_ir(xml_path, output_path):
    """Replace recurrence subgraphs with FusedGDNRecurrence custom ops."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    layers_elem = root.find('layers')
    edges_elem = root.find('edges')

    layers, out_edges, in_edges = build_graph(root)

    # Determine which layers (0, 1, 2) are in this block
    # by looking for recurrence op name patterns
    block_layers = set()
    for lid, info in layers.items():
        m = RECURRENCE_NAME_PATTERN.search(info['name'])
        if m:
            block_layers.add(int(m.group(1)))

    if not block_layers:
        logger.warning("No recurrence ops found in %s", xml_path)
        return False

    # Sort layers: they appear as 0,1,2 or higher indices (e.g., 3,4,5 for block 1)
    block_layers = sorted(block_layers)
    logger.info("Block layers: %s", block_layers)

    # Find the max layer ID to generate new IDs
    max_id = max(int(lid) for lid in layers)

    # Process each layer's recurrence
    for idx, bl in enumerate(block_layers):
        logger.info("Processing layer %d (index %d in block)...", bl, idx)

        # Find recurrence ops
        rec_ops, ext_inputs, ext_outputs = find_recurrence_ops(
            layers, out_edges, in_edges, bl
        )
        if rec_ops is None:
            logger.warning("  Skipping layer %d: no recurrence found", bl)
            continue

        # Get detailed I/O mapping
        inputs_map, ext_outs = find_recurrence_io(
            layers, out_edges, in_edges, rec_ops, idx
        )

        logger.info("  Recurrence I/O:")
        for role, val in inputs_map.items():
            if val:
                logger.info("    Input %s: layer %s port %s", role, val[0], val[1])
        logger.info("  External outputs: %d edges", len(ext_outs))

        # Identify which external outputs are the attention output vs state output
        # by looking at what they feed into (Result node name)
        attn_out_targets = []
        state_out_targets = []
        for rop_id, from_port, dst_id, to_port, rop_name in ext_outs:
            dst_info = layers.get(dst_id, {})
            # If target is a Result with 'rec' in name, it's state output
            if dst_info.get('type') == 'Result' and 'rec' in dst_info.get('name', ''):
                state_out_targets.append((dst_id, to_port, rop_id, from_port))
            else:
                # Track the source recurrence op to identify attn vs state
                attn_out_targets.append((dst_id, to_port, rop_id, from_port))

        # Create the FusedGDNRecurrence custom layer
        max_id += 1
        fused_id = str(max_id)

        # Determine state shape from the Parameter
        state_info = inputs_map.get('state')
        state_shape = "1,16,128,128"  # Known fixed shape for Qwen3.5-0.8B
        output_shape = "1,16,1,128"   # Attention output [B, H, 1, D]

        fused_layer = ET.SubElement(layers_elem, 'layer')
        fused_layer.set('id', fused_id)
        fused_layer.set('name', f'FusedGDNRecurrence_layer{bl}')
        fused_layer.set('type', 'FusedGDNRecurrence')
        fused_layer.set('version', 'custom')

        # Input ports: 0=q, 1=k, 2=v, 3=g, 4=beta, 5=state
        input_elem = ET.SubElement(fused_layer, 'input')
        # q: [1, 16, 1, 128]
        p0 = ET.SubElement(input_elem, 'port')
        p0.set('id', '0')
        p0.set('precision', 'FP32')
        for d in [1, 16, 1, 128]:
            dim = ET.SubElement(p0, 'dim')
            dim.text = str(d)
        # k: [1, 16, 1, 128]
        p1 = ET.SubElement(input_elem, 'port')
        p1.set('id', '1')
        p1.set('precision', 'FP32')
        for d in [1, 16, 1, 128]:
            dim = ET.SubElement(p1, 'dim')
            dim.text = str(d)
        # v: [1, 16, 1, 128]
        p2 = ET.SubElement(input_elem, 'port')
        p2.set('id', '2')
        p2.set('precision', 'FP32')
        for d in [1, 16, 1, 128]:
            dim = ET.SubElement(p2, 'dim')
            dim.text = str(d)
        # g: [1, 16, 1]
        p3 = ET.SubElement(input_elem, 'port')
        p3.set('id', '3')
        p3.set('precision', 'FP32')
        for d in [1, 16, 1]:
            dim = ET.SubElement(p3, 'dim')
            dim.text = str(d)
        # beta: [1, 16, 1]
        p4 = ET.SubElement(input_elem, 'port')
        p4.set('id', '4')
        p4.set('precision', 'FP32')
        for d in [1, 16, 1]:
            dim = ET.SubElement(p4, 'dim')
            dim.text = str(d)
        # state: [1, 16, 128, 128]
        p5 = ET.SubElement(input_elem, 'port')
        p5.set('id', '5')
        p5.set('precision', 'FP32')
        for d in [1, 16, 128, 128]:
            dim = ET.SubElement(p5, 'dim')
            dim.text = str(d)

        # Output ports: 0=attn_output [1, 16, 1, 128], 1=state_out [1, 16, 128, 128]
        output_elem = ET.SubElement(fused_layer, 'output')
        po0 = ET.SubElement(output_elem, 'port')
        po0.set('id', '6')
        po0.set('precision', 'FP32')
        for d in [1, 16, 1, 128]:
            dim = ET.SubElement(po0, 'dim')
            dim.text = str(d)
        po1 = ET.SubElement(output_elem, 'port')
        po1.set('id', '7')
        po1.set('precision', 'FP32')
        for d in [1, 16, 128, 128]:
            dim = ET.SubElement(po1, 'dim')
            dim.text = str(d)

        # Remove old recurrence ops from <layers>
        for lid in rec_ops:
            elem = layers[lid]['elem']
            layers_elem.remove(elem)

        # Remove edges involving recurrence ops
        edges_to_remove = []
        for edge in edges_elem.findall('edge'):
            fl = edge.get('from-layer')
            tl = edge.get('to-layer')
            if fl in rec_ops or tl in rec_ops:
                edges_to_remove.append(edge)
        for e in edges_to_remove:
            edges_elem.remove(e)

        # Add new edges connecting the fused op
        # Input edges: connect external sources to fused op input ports
        # We need to map: q -> port 0, k -> port 1, v -> port 2, g -> port 3, beta -> port 4, state -> port 5
        role_to_port = {
            'q_unsq': '0', 'k_unsq': '1', 'v': '2',
            'decay': '3', 'beta': '4', 'state': '5',
        }
        for role, port_id in role_to_port.items():
            src = inputs_map.get(role)
            if src:
                new_edge = ET.SubElement(edges_elem, 'edge')
                new_edge.set('from-layer', src[0])
                new_edge.set('from-port', src[1])
                new_edge.set('to-layer', fused_id)
                new_edge.set('to-port', port_id)

        # Output edges: connect fused op outputs to downstream ops
        # attn_output (port 6) -> downstream consumers
        # state_output (port 7) -> state Result node
        for dst_id, to_port, rop_id, from_port in state_out_targets:
            new_edge = ET.SubElement(edges_elem, 'edge')
            new_edge.set('from-layer', fused_id)
            new_edge.set('from-port', '7')  # state output
            new_edge.set('to-layer', dst_id)
            new_edge.set('to-port', to_port)

        for dst_id, to_port, rop_id, from_port in attn_out_targets:
            new_edge = ET.SubElement(edges_elem, 'edge')
            new_edge.set('from-layer', fused_id)
            new_edge.set('from-port', '6')  # attn output
            new_edge.set('to-layer', dst_id)
            new_edge.set('to-port', to_port)

    # Write modified IR
    tree.write(output_path, xml_declaration=True, encoding='unicode')
    logger.info("Saved fused IR: %s", output_path)
    return True


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Fuse GDN recurrence ops in noloop IR")
    parser.add_argument(
        "--model-dir",
        default="models/qwen35/Qwen3.5-0.8B-hybrid",
        help="Directory containing gdn_noloop_block_*.xml files",
    )
    parser.add_argument(
        "--num-blocks", type=int, default=6,
        help="Number of GDN blocks (default 6)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Analyze only, don't write output files",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    for i in range(args.num_blocks):
        noloop_xml = model_dir / f"gdn_noloop_block_{i}.xml"
        if not noloop_xml.exists():
            logger.warning("Not found: %s, skipping", noloop_xml)
            continue

        fused_xml = model_dir / f"gdn_fused_block_{i}.xml"
        logger.info("=== Processing block %d: %s ===", i, noloop_xml)

        if args.dry_run:
            # Just analyze, don't write
            tree = ET.parse(noloop_xml)
            root = tree.getroot()
            layers, out_edges, in_edges = build_graph(root)

            # Count recurrence ops per layer
            for bl in range(3):
                count = 0
                for lid, info in layers.items():
                    pattern = f'__module.layers.{bl}.linear_attn.recurrent_attention_cell/'
                    # Need to find actual layer indices from the block
                    if 'recurrent_attention_cell' in info['name']:
                        count += 1
                logger.info("  Recurrence ops with 'recurrent_attention_cell': %d", count)
                break
        else:
            # Copy the .bin file (weights unchanged)
            noloop_bin = noloop_xml.with_suffix('.bin')
            fused_bin = fused_xml.with_suffix('.bin')
            if noloop_bin.exists():
                import shutil
                shutil.copy2(noloop_bin, fused_bin)

            replace_recurrence_in_ir(str(noloop_xml), str(fused_xml))


if __name__ == "__main__":
    main()
