#ifndef GDN_FUSE_PASS_H
#define GDN_FUSE_PASS_H

#include <openvino/openvino.hpp>

/// Custom OV op that maps to the fused OpenCL recurrence kernel.
/// The GPU plugin dispatches this via CONFIG_FILE -> gdn_recurrence.cl.
///
/// Inputs:
///   0: q     [B, H, D]         queries (squeezed 3D)
///   1: k     [B, H, D]         keys (squeezed 3D)
///   2: v     [B, H, D]         values (squeezed 3D)
///   3: g     [B, H, 1]         raw decay gate (before exp)
///   4: beta  [B, H, 1]         sigmoid input gate
///   5: state [B, H, D, D]      recurrent state (128x128 per head)
///
/// Outputs:
///   0: concat_out [flat]        concatenated [attn_flat, state_flat]
///                               Same shape as original Concat output
class FusedGDNRecurrenceOp : public ov::op::Op {
public:
    OPENVINO_OP("FusedGDNRecurrence", "custom");

    FusedGDNRecurrenceOp() = default;

    FusedGDNRecurrenceOp(const ov::OutputVector& inputs, int64_t num_attn_elems);

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(
        const ov::OutputVector& new_inputs) const override;

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("num_attn_elems", num_attn_elems_);
        return true;
    }

    bool has_evaluate() const override { return false; }

    int64_t num_attn_elems() const { return num_attn_elems_; }

private:
    int64_t num_attn_elems_ = 2048;  // 16*128 = H*D (attn output flat size)
};

/// Replace recurrence subgraphs in a GDN noloop block IR with FusedGDNRecurrenceOp.
///
/// For each of the 3 layers in the block, finds the ~15 ops that form the
/// single-step recurrence (identified by name pattern "recurrent_attention_cell")
/// and replaces them with a single FusedGDNRecurrenceOp node.
///
/// Must be called BEFORE MakeStateful (operates on explicit state I/O).
///
/// @param model The noloop block model to transform (modified in-place)
/// @return Number of recurrence subgraphs replaced (0-3)
int fuse_gdn_recurrence(std::shared_ptr<ov::Model> model);

#endif // GDN_FUSE_PASS_H
