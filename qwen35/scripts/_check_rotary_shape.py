"""Quick check: what shape does rotary_emb produce?"""
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True, torch_dtype=torch.float32)
cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config

print("=== Config ===")
print(f"hidden_size={cfg.hidden_size}")
print(f"num_attention_heads={cfg.num_attention_heads}")
print(f"num_key_value_heads={cfg.num_key_value_heads}")
print(f"head_dim={getattr(cfg, 'head_dim', 'NOT SET')}")
print(f"partial_rotary_factor={getattr(cfg, 'partial_rotary_factor', 'NOT SET')}")
print(f"mrope_section={getattr(cfg, 'mrope_section', 'NOT SET')}")

# Check rotary emb
rotary = model.model.rotary_emb
print(f"\nrotary_emb class: {type(rotary).__name__}")
print(f"inv_freq shape: {rotary.inv_freq.shape}")
print(f"inv_freq dtype: {rotary.inv_freq.dtype}")

# Test with single position
dummy_h = torch.zeros(1, 1, cfg.hidden_size)
dummy_pos = torch.zeros(3, 1, 1, dtype=torch.int64)
cos, sin = rotary(dummy_h, dummy_pos)
print(f"\ncos shape: {cos.shape}")
print(f"sin shape: {sin.shape}")
print(f"cos dtype: {cos.dtype}")

# Test with position_ids = [[5]]
dummy_pos5 = torch.full((3, 1, 1), 5, dtype=torch.int64)
cos5, sin5 = rotary(dummy_h, dummy_pos5)
print(f"\ncos5 shape: {cos5.shape} (should match cos)")

# Check attention layer head_dim
for i, layer in enumerate(model.model.layers):
    if cfg.layer_types[i] == "full_attention":
        attn = layer.self_attn
        print(f"\nLayer {i} (full_attention):")
        print(f"  q_proj weight: {attn.q_proj.weight.shape}")
        print(f"  k_proj weight: {attn.k_proj.weight.shape}")
        print(f"  head_dim: {attn.head_dim}")
        print(f"  num_heads: {attn.num_heads}")
        print(f"  num_kv_heads: {attn.num_key_value_heads}")
        break

# Check GDN layer dimensions
for i, layer in enumerate(model.model.layers):
    if cfg.layer_types[i] == "linear_attention":
        gdn = layer.linear_attn
        print(f"\nLayer {i} (linear_attention / GDN):")
        print(f"  num_k_heads: {gdn.num_k_heads}")
        print(f"  num_v_heads: {gdn.num_v_heads}")
        print(f"  head_k_dim: {gdn.head_k_dim}")
        print(f"  head_v_dim: {gdn.head_v_dim}")
        print(f"  key_dim: {gdn.key_dim}")
        print(f"  value_dim: {gdn.value_dim}")
        print(f"  conv1d weight: {gdn.conv1d.weight.shape}")
        break
