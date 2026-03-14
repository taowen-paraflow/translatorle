// gdn_fuse_pass.cpp — Replace GDN recurrence subgraphs with a fused custom op.
//
// The noloop GDN blocks contain ~15 flat ops per layer for the recurrence
// (Exp, Multiply, ReduceSum, Subtract, Add, Unsqueeze, etc.).  This pass
// replaces them with a single FusedGDNRecurrenceOp node that maps to an
// OpenCL kernel via CONFIG_FILE.
//
// The pass identifies recurrence ops by their friendly name, which contains
// "recurrent_attention_cell" (from the PyTorch module hierarchy).

#include "gdn_fuse_pass.h"
#include "utils.h"  // for log()

#include <algorithm>
#include <map>
#include <queue>
#include <regex>
#include <set>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// FusedGDNRecurrenceOp implementation
// ---------------------------------------------------------------------------

FusedGDNRecurrenceOp::FusedGDNRecurrenceOp(const ov::OutputVector& inputs, int64_t num_attn_elems)
    : Op(inputs), num_attn_elems_(num_attn_elems)
{
    constructor_validate_and_infer_types();
}

void FusedGDNRecurrenceOp::validate_and_infer_types() {
    // Single output: concatenated [attn_flat, state_flat]
    // attn_flat = H*D = 2048 floats
    // state_flat = H*D*D = 262144 floats
    // Total = 264192 floats as 1D tensor
    //
    // This matches the original Concat output, so downstream Slice+Reshape
    // chains continue to work unchanged.
    int64_t total = -1;  // dynamic by default
    auto state_shape = get_input_partial_shape(5);
    auto q_shape = get_input_partial_shape(0);

    // Try to compute total output size
    if (q_shape.is_static() && state_shape.is_static()) {
        int64_t attn_size = 1;
        for (auto& d : q_shape) attn_size *= d.get_length();
        int64_t state_size = 1;
        for (auto& d : state_shape) state_size *= d.get_length();
        total = attn_size + state_size;
    }

    if (total > 0) {
        set_output_type(0, get_input_element_type(0), ov::PartialShape{total});
    } else {
        set_output_type(0, get_input_element_type(0), ov::PartialShape{ov::Dimension::dynamic()});
    }
}

std::shared_ptr<ov::Node> FusedGDNRecurrenceOp::clone_with_new_inputs(
    const ov::OutputVector& new_inputs) const
{
    return std::make_shared<FusedGDNRecurrenceOp>(new_inputs, num_attn_elems_);
}

// ---------------------------------------------------------------------------
// Recurrence subgraph identification and replacement
// ---------------------------------------------------------------------------

namespace {

// Regex to match recurrence ops by their friendly name.
// Format: __module.layers.{N}.linear_attn.recurrent_attention_cell/...
const std::regex RECURRENCE_PATTERN(
    R"(__module\.layers\.(\d+)\.linear_attn\.recurrent_attention_cell/)"
);

struct RecurrenceInfo {
    int layer_idx;                          // Global layer index (e.g., 0, 1, 2)
    std::set<std::shared_ptr<ov::Node>> ops;  // All ops in the recurrence
    // External inputs to the recurrence (from outside the rec subgraph)
    ov::Output<ov::Node> q_input;           // [B, H, 1, D]
    ov::Output<ov::Node> k_input;           // [B, H, 1, D]
    ov::Output<ov::Node> v_input;           // [B, H, 1, D]
    ov::Output<ov::Node> g_input;           // [B, H, 1]
    ov::Output<ov::Node> beta_input;        // [B, H, 1]
    ov::Output<ov::Node> state_input;       // [B, H, D, D]
    // The Concat node that produces the flattened [attn_flat, state_flat] output
    std::shared_ptr<ov::Node> concat_node;
};

/// Collect all ops belonging to each layer's recurrence subgraph.
std::map<int, std::set<std::shared_ptr<ov::Node>>> find_recurrence_ops(
    const std::shared_ptr<ov::Model>& model)
{
    std::map<int, std::set<std::shared_ptr<ov::Node>>> result;
    for (auto& op : model->get_ordered_ops()) {
        auto name = op->get_friendly_name();
        std::smatch match;
        if (std::regex_search(name, match, RECURRENCE_PATTERN)) {
            int layer_idx = std::stoi(match[1].str());
            result[layer_idx].insert(op);
        }
    }
    return result;
}

/// Find the Concat node that is the final output of the recurrence.
/// It should be the only Concat node among the recurrence ops.
std::shared_ptr<ov::Node> find_concat(
    const std::set<std::shared_ptr<ov::Node>>& rec_ops)
{
    for (auto& op : rec_ops) {
        if (op->get_type_info().name == std::string("Concat")) {
            return op;
        }
    }
    return nullptr;
}

/// Check if a source node is a small constant (reduce axes, reshape target, etc.)
bool is_constant_like(const ov::Output<ov::Node>& source) {
    auto node = source.get_node_shared_ptr();
    auto type_name = std::string(node->get_type_info().name);
    if (type_name == "Constant") return true;
    if (type_name == "Convert") {
        // Check if Convert's input is Constant (decompression pattern)
        auto input_node = node->input(0).get_source_output().get_node_shared_ptr();
        return std::string(input_node->get_type_info().name) == "Constant";
    }
    return false;
}

/// Identify the external inputs to the recurrence by role.
///
/// Uses the consumer recurrence op's friendly name to classify inputs:
///   - state: Parameter named "recurrent_state_*"
///   - decay: non-state input to "Multiply" (state * decay)
///   - k:     non-rec input to "Unsqueeze_1" (k_unsq), or source feeding Multiply_1
///   - v:     input to Subtract (v - kv_mem)
///   - beta:  non-rec input to "Multiply_2" (error * beta)
///   - q:     non-rec input to "Unsqueeze" (q_unsq), or source feeding Multiply_4
///
/// Returns false if identification fails.
bool identify_inputs(
    const std::set<std::shared_ptr<ov::Node>>& rec_ops,
    RecurrenceInfo& info)
{
    // Collect all external inputs (sources outside the recurrence)
    struct ExternalInput {
        ov::Output<ov::Node> source;
        std::shared_ptr<ov::Node> consumer;  // the rec op it feeds
        size_t consumer_port;
        std::string consumer_name;           // friendly name of consumer
    };
    std::vector<ExternalInput> ext_inputs;

    for (auto& op : rec_ops) {
        for (size_t i = 0; i < op->get_input_size(); ++i) {
            auto src_output = op->input(i).get_source_output();
            auto src_node = src_output.get_node_shared_ptr();
            if (rec_ops.count(src_node) == 0) {
                ext_inputs.push_back({
                    src_output, op, i, op->get_friendly_name()
                });
            }
        }
    }

    // Classify by consumer op name pattern.
    // The recurrence ops are named:
    //   __module.layers.{N}.linear_attn.recurrent_attention_cell/aten::op/OpName
    //
    // From IR analysis, external inputs enter recurrence via these rec ops:
    //   state → Multiply       (state * decay)
    //   g     → Exp            (exp(g))
    //   k     → Unsqueeze_1    (k.unsqueeze(-1))
    //   v     → Squeeze_1      (v.squeeze(seq_dim))
    //   beta  → Multiply_2     (error * beta), at port 1
    //   q     → Squeeze_2      (q.squeeze(seq_dim))
    //
    // We match on the trailing op name after "recurrent_attention_cell/".
    for (auto& ext : ext_inputs) {
        if (is_constant_like(ext.source)) continue;

        auto src_node = ext.source.get_node_shared_ptr();
        auto src_type = std::string(src_node->get_type_info().name);
        auto& cname = ext.consumer_name;

        // State: Parameter node whose name starts with "recurrent_state_"
        if (src_type == "Parameter" && src_node->get_friendly_name().find("recurrent_state_") == 0) {
            info.state_input = ext.source;
            continue;
        }

        // G (decay gate): external input to Exp inside recurrence
        // Pattern: .../recurrent_attention_cell/aten::exp/Exp
        if (cname.find("recurrent_attention_cell/") != std::string::npos &&
            cname.find("/Exp") != std::string::npos && ext.consumer_port == 0) {
            info.g_input = ext.source;
            continue;
        }

        // K: external input (port 0) to Unsqueeze_1 (k.unsqueeze(-1))
        if (cname.find("/Unsqueeze_1") != std::string::npos && ext.consumer_port == 0) {
            info.k_input = ext.source;
            continue;
        }

        // V: external input (port 0) to Squeeze_1 (v.squeeze)
        if (cname.find("/Squeeze_1") != std::string::npos && ext.consumer_port == 0) {
            info.v_input = ext.source;
            continue;
        }

        // Beta: external input (port 1) to Multiply_2 (error * beta)
        // Port 0 is the error (from Subtract, internal); port 1 is beta (external)
        if (cname.find("/Multiply_2") != std::string::npos && ext.consumer_port == 1) {
            info.beta_input = ext.source;
            continue;
        }

        // Q: external input (port 0) to Squeeze_2 (q.squeeze)
        if (cname.find("/Squeeze_2") != std::string::npos && ext.consumer_port == 0) {
            info.q_input = ext.source;
            continue;
        }
    }

    // Verify we found all inputs
    bool ok = info.state_input.get_node_shared_ptr() &&
              info.q_input.get_node_shared_ptr() &&
              info.k_input.get_node_shared_ptr() &&
              info.v_input.get_node_shared_ptr() &&
              info.g_input.get_node_shared_ptr() &&
              info.beta_input.get_node_shared_ptr();

    if (!ok) {
        log("WARNING: Could not identify all recurrence inputs for layer " +
            std::to_string(info.layer_idx));
        log("  state=" + std::to_string(!!info.state_input.get_node_shared_ptr()));
        log("  q=" + std::to_string(!!info.q_input.get_node_shared_ptr()));
        log("  k=" + std::to_string(!!info.k_input.get_node_shared_ptr()));
        log("  v=" + std::to_string(!!info.v_input.get_node_shared_ptr()));
        log("  g=" + std::to_string(!!info.g_input.get_node_shared_ptr()));
        log("  beta=" + std::to_string(!!info.beta_input.get_node_shared_ptr()));

        // Log all external inputs for debugging
        for (auto& ext : ext_inputs) {
            if (is_constant_like(ext.source)) continue;
            auto src = ext.source.get_node_shared_ptr();
            log("  ext: src_type=" + std::string(src->get_type_info().name) +
                " src_name=" + src->get_friendly_name() +
                " -> consumer=" + ext.consumer_name +
                " port=" + std::to_string(ext.consumer_port));
        }
    }
    return ok;
}

/// Find downstream consumers of the Concat output that are OUTSIDE the recurrence.
/// Returns the Slice/StridedSlice nodes that extract attn and state from the concat.
struct ConcatConsumers {
    // The consumer of attn_out (after Slice + Reshape)
    std::set<ov::Input<ov::Node>> attn_consumers;
    // The consumer of state_out (after Slice + Reshape)
    std::set<ov::Input<ov::Node>> state_consumers;
    // Intermediate nodes to remove (Slice + Reshape between Concat and final consumers)
    std::set<std::shared_ptr<ov::Node>> intermediate_nodes;
};

ConcatConsumers find_concat_consumers(
    const std::shared_ptr<ov::Node>& concat_node,
    const std::set<std::shared_ptr<ov::Node>>& rec_ops)
{
    ConcatConsumers result;

    // The Concat output is consumed by ops in patched_recurrent_gated_delta_rule:
    //   output_cell[:num_elems].reshape(value.shape) -> attn_out
    //   output_cell[num_elems:].reshape(state.shape) -> state_out
    //
    // In the IR, this becomes:
    //   Concat -> StridedSlice/Slice (attn) -> Reshape -> [gate multiply ...]
    //   Concat -> StridedSlice/Slice (state) -> Reshape -> [Result out_rec{i}]

    for (auto& output : concat_node->outputs()) {
        for (auto& target_input : output.get_target_inputs()) {
            auto consumer = target_input.get_node()->shared_from_this();
            if (rec_ops.count(consumer)) continue;  // Skip internal ops

            // This should be a StridedSlice or Slice
            auto consumer_type = std::string(consumer->get_type_info().name);
            if (consumer_type != "StridedSlice" && consumer_type != "Slice") {
                // Unexpected consumer — might be directly a Reshape
                // Just add to intermediate and follow
            }

            result.intermediate_nodes.insert(consumer);

            // Follow the chain: Slice -> Reshape -> final consumer
            for (auto& out2 : consumer->outputs()) {
                for (auto& target2 : out2.get_target_inputs()) {
                    auto next = target2.get_node()->shared_from_this();
                    auto next_type = std::string(next->get_type_info().name);

                    if (next_type == "Reshape") {
                        result.intermediate_nodes.insert(next);

                        // Determine if this is attn or state by output port name.
                        // State output has port name containing "out_rec".
                        // Attn output has port name containing "core_attn_out".
                        bool is_state = false;
                        for (auto& name : next->output(0).get_names()) {
                            if (name.find("out_rec") != std::string::npos) {
                                is_state = true;
                                break;
                            }
                        }

                        // Fallback: if no port names, check downstream for Result
                        // nodes with "rec" in name
                        if (!is_state && next->output(0).get_names().empty()) {
                            for (auto& out_check : next->outputs()) {
                                for (auto& t : out_check.get_target_inputs()) {
                                    auto target_node = t.get_node()->shared_from_this();
                                    if (std::string(target_node->get_type_info().name) == "Result" &&
                                        target_node->get_friendly_name().find("rec") != std::string::npos) {
                                        is_state = true;
                                    }
                                }
                            }
                        }

                        // Collect final consumers
                        for (auto& out3 : next->outputs()) {
                            for (auto& target3 : out3.get_target_inputs()) {
                                if (is_state) {
                                    result.state_consumers.insert(target3);
                                } else {
                                    result.attn_consumers.insert(target3);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return result;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

int fuse_gdn_recurrence(std::shared_ptr<ov::Model> model) {
    auto layer_ops = find_recurrence_ops(model);
    if (layer_ops.empty()) {
        log("No recurrence ops found — skipping fusion");
        return 0;
    }

    log("Found recurrence ops for " + std::to_string(layer_ops.size()) + " layers");
    int fused_count = 0;

    for (auto& [layer_idx, rec_ops] : layer_ops) {
        log("Processing layer " + std::to_string(layer_idx) +
            " (" + std::to_string(rec_ops.size()) + " recurrence ops)...");

        RecurrenceInfo info;
        info.layer_idx = layer_idx;
        info.ops = rec_ops;

        // Find the Concat node (final output of recurrence)
        info.concat_node = find_concat(rec_ops);
        if (!info.concat_node) {
            log("  WARNING: No Concat found for layer " + std::to_string(layer_idx));
            continue;
        }

        // Identify external inputs by consumer op name
        if (!identify_inputs(rec_ops, info)) {
            continue;
        }

        // Create the FusedGDNRecurrenceOp with single output matching Concat
        // The output is a 1D concatenation of [attn_flat, state_flat]
        // which is exactly what the original Concat produces.
        // Downstream Slice + Reshape nodes remain untouched.
        int64_t num_attn_elems = 16 * 128;  // H * D for this model
        auto fused_op = std::make_shared<FusedGDNRecurrenceOp>(
            ov::OutputVector{
                info.q_input,
                info.k_input,
                info.v_input,
                info.g_input,
                info.beta_input,
                info.state_input,
            },
            num_attn_elems
        );
        fused_op->set_friendly_name("FusedGDNRecurrence_layer" + std::to_string(layer_idx));

        // Replace the Concat output with fused op output.
        // All consumers of Concat (Slice nodes) automatically point to fused_op.
        info.concat_node->output(0).replace(fused_op->output(0));

        // The old recurrence ops will become disconnected from the graph
        // (no consumers). They'll be cleaned up by OV's dead code elimination
        // during compilation.

        fused_count++;
        log("  Fused layer " + std::to_string(layer_idx) +
            ": " + std::to_string(rec_ops.size()) + " ops -> 1 FusedGDNRecurrenceOp");
    }

    if (fused_count > 0) {
        model->validate_nodes_and_infer_types();
    }

    return fused_count;
}
