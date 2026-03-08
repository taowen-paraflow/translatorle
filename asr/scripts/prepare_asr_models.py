"""Prepare Qwen3-ASR models for OpenVINO NPU deployment.

This single script performs ALL model preparation steps:
  1. Load the original Qwen3-ASR-0.6B model (HuggingFace or local)
  2. Export the audio encoder to OpenVINO IR (static shape [1, 128, 800])
  3. Extract the text decoder as standalone Qwen3ForCausalLM
  4. Export the decoder via optimum-intel as a stateful model with KV-cache
  5. Perform IR surgery: remove input_ids, add inputs_embeds Parameter
  6. Extract embed_tokens.npy for building inputs_embeds at runtime
  7. Copy tokenizer / preprocessor files for runtime use

Output layout under models/:
    models/
    +-- encoder_fp16.xml / .bin          Audio encoder (NPU, static [1,128,800])
    +-- decoder_stateful_ov/             Intermediate stateful decoder (optimum-intel export)
    +-- decoder_stateful_embeds/         Final decoder for NPU (IR-surgery, inputs_embeds)
    +-- qwen3_decoder_standalone/        Intermediate HF checkpoint (Qwen3ForCausalLM)
    +-- embed_tokens.npy                 Embedding table [151936, 1024]
    +-- Qwen3-ASR-0.6B/                  Tokenizer + preprocessor config

Usage (from WSL, targeting Windows Python):
    powershell.exe -Command '
        $env:Path = "C:\\Users\\taowen\\.local\\bin;$env:Path";
        $env:PYTHONIOENCODING = "utf-8";
        cd C:\\Apps\\translatorle;
        uv run python scripts/prepare_asr_models.py
    '

    # Or with a local model path:
    uv run python scripts/prepare_asr_models.py --model-id C:\\Models\\Qwen3-ASR-0.6B
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _timer():
    """Return a context-manager-like pair: call start(), then elapsed()."""
    t0 = time.perf_counter()
    return lambda: time.perf_counter() - t0


def _free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------

def _resolve_paths(project_root: Path):
    """Return a dict of all output paths under models/."""
    m = project_root / "models"
    return {
        "models_dir": m,
        "encoder_xml": m / "encoder_fp16.xml",
        "decoder_standalone_dir": m / "qwen3_decoder_standalone",
        "decoder_stateful_ov_dir": m / "decoder_stateful_ov",
        "decoder_stateful_embeds_dir": m / "decoder_stateful_embeds",
        "embed_tokens_npy": m / "embed_tokens.npy",
        "tokenizer_dir": m / "Qwen3-ASR-0.6B",
    }


# ============================================================================
# Step 1 -- Load the full Qwen3-ASR model
# ============================================================================

def step1_load_model(model_id: str):
    print("=" * 60)
    print("Step 1: Loading Qwen3-ASR model")
    print("=" * 60)
    elapsed = _timer()

    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import (
        Qwen3ASRConfig,
    )
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
        Qwen3ASRForConditionalGeneration,
    )
    from transformers import AutoConfig, AutoModel

    # Register so AutoModel can load the custom architecture
    AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
    AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)

    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=torch.float32,
    )
    model.eval()
    print(f"  Loaded from: {model_id}")
    print(f"  Time: {elapsed():.1f}s")
    print()
    return model


# ============================================================================
# Step 2 -- Export audio encoder to OpenVINO IR
# ============================================================================

class _StaticAudioEncoder(torch.nn.Module):
    """Static-shape wrapper around Qwen3ASRAudioEncoder for OpenVINO export.

    Replaces all dynamic operations (chunking by feature_lens, padding,
    cu_seqlens construction) with fixed constants for T_fixed=800.
    Submodules are referenced (not copied) so weights are shared.
    """

    # Constants for T_fixed = 800, n_window = 50
    N_CHUNKS = 8          # 800 / (n_window * 2)
    CHUNK_LEN = 100       # n_window * 2
    T_AFTER_CNN = 13      # per-chunk time after 3x stride-2 conv
    FREQ_AFTER_CNN = 16   # freq dim after 3x stride-2 conv
    SEQ_LEN = 104         # N_CHUNKS * T_AFTER_CNN
    D_MODEL = 896
    DOWNSAMPLE_HIDDEN = 480

    def __init__(self, audio_tower: torch.nn.Module):
        super().__init__()
        self.conv2d1 = audio_tower.conv2d1
        self.conv2d2 = audio_tower.conv2d2
        self.conv2d3 = audio_tower.conv2d3
        self.conv_out = audio_tower.conv_out
        self.layers = audio_tower.layers
        self.ln_post = audio_tower.ln_post
        self.proj1 = audio_tower.proj1
        self.act = audio_tower.act
        self.proj2 = audio_tower.proj2

        pos_embed = audio_tower.positional_embedding.positional_embedding[
            : self.T_AFTER_CNN, :
        ].clone()
        self.register_buffer("pos_embed", pos_embed)

        cu_seqlens = torch.tensor([0, self.SEQ_LEN], dtype=torch.int32)
        self.register_buffer("cu_seqlens", cu_seqlens)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = mel[0].T.reshape(self.N_CHUNKS, self.CHUNK_LEN, 128)
        x = x.transpose(1, 2).unsqueeze(1)

        x = torch.nn.functional.gelu(self.conv2d1(x))
        x = torch.nn.functional.gelu(self.conv2d2(x))
        x = torch.nn.functional.gelu(self.conv2d3(x))

        x = x.permute(0, 3, 1, 2).contiguous().view(
            self.N_CHUNKS, self.T_AFTER_CNN,
            self.DOWNSAMPLE_HIDDEN * self.FREQ_AFTER_CNN,
        )
        x = self.conv_out(x)
        x = x + self.pos_embed.unsqueeze(0)
        x = x.reshape(self.SEQ_LEN, self.D_MODEL)

        for layer in self.layers:
            x = layer(x, self.cu_seqlens)[0]

        x = self.ln_post(x)
        x = self.proj2(self.act(self.proj1(x)))
        return x.unsqueeze(0)


def step2_export_encoder(model, paths: dict):
    print("=" * 60)
    print("Step 2: Exporting audio encoder to OpenVINO IR")
    print("=" * 60)
    elapsed = _timer()

    import openvino as ov

    audio_tower = model.thinker.audio_tower

    # Force eager attention (required for torch tracing, no flash_attention_2)
    audio_tower.config._attn_implementation = "eager"
    for layer in audio_tower.layers:
        layer.self_attn.config._attn_implementation = "eager"

    # Wrap in static-shape module (bypasses dynamic feature_lens logic)
    wrapper = _StaticAudioEncoder(audio_tower)
    wrapper.eval()

    dummy_input = torch.randn(1, 128, 800, dtype=torch.float32)

    with torch.no_grad():
        ref_output = wrapper(dummy_input)
    print(f"  Reference output shape: {ref_output.shape}")
    assert ref_output.shape == (1, 104, 1024), (
        f"Expected [1, 104, 1024], got {list(ref_output.shape)}"
    )

    ov_encoder = ov.convert_model(
        wrapper,
        example_input=dummy_input,
        input=[ov.PartialShape([1, 128, 800])],
    )

    ov.save_model(
        ov_encoder,
        str(paths["encoder_xml"]),
        compress_to_fp16=True,
    )

    xml_path = paths["encoder_xml"]
    bin_path = xml_path.with_suffix(".bin")
    xml_mb = xml_path.stat().st_size / 1024 / 1024
    bin_mb = bin_path.stat().st_size / 1024 / 1024
    print(f"  Saved: {xml_path.name} ({xml_mb:.1f} MB) + .bin ({bin_mb:.1f} MB)")
    print(f"  Time: {elapsed():.1f}s")
    print()


# ============================================================================
# Step 3 -- Extract decoder weights (no audio_tower keys)
# ============================================================================

def step3_extract_decoder_weights(model):
    print("=" * 60)
    print("Step 3: Extracting decoder weights (excluding audio_tower)")
    print("=" * 60)
    elapsed = _timer()

    thinker = model.thinker
    thinker_sd = thinker.state_dict()
    print(f"  Total thinker params: {len(thinker_sd)}")

    # Remove audio_tower keys -- they belong to the encoder
    decoder_sd = {
        k: v for k, v in thinker_sd.items() if not k.startswith("audio_tower.")
    }
    print(f"  Decoder params (no audio_tower): {len(decoder_sd)}")

    prefixes = sorted({k.split(".")[0] for k in decoder_sd})
    print(f"  Key prefixes: {prefixes}")

    has_lm_head = "lm_head.weight" in decoder_sd
    print(f"  lm_head.weight present: {has_lm_head}")

    print(f"  Time: {elapsed():.1f}s")
    print()
    return decoder_sd


# ============================================================================
# Step 4 -- Create standalone Qwen3ForCausalLM, load weights, save
# ============================================================================

def step4_create_standalone_decoder(decoder_sd: dict, paths: dict):
    print("=" * 60)
    print("Step 4: Creating standalone Qwen3ForCausalLM")
    print("=" * 60)
    elapsed = _timer()

    from transformers import Qwen3Config, Qwen3ForCausalLM

    # Config values from Qwen3-ASR-0.6B/config.json -> thinker_config -> text_config
    qwen3_config = Qwen3Config(
        hidden_size=1024,
        intermediate_size=3072,
        max_position_embeddings=65536,
        num_attention_heads=16,
        num_hidden_layers=28,
        num_key_value_heads=8,
        rms_norm_eps=1e-06,
        rope_scaling={
            "interleaved": True,
            "mrope_interleaved": True,
            "mrope_section": [24, 20, 20],
            "rope_type": "default",
            "type": "default",
        },
        rope_theta=1000000.0,
        sliding_window=None,
        tie_word_embeddings=True,
        vocab_size=151936,
        head_dim=128,
        attention_dropout=0.0,
        attention_bias=False,
        hidden_act="silu",
        initializer_range=0.02,
        use_cache=True,
    )

    standalone = Qwen3ForCausalLM(qwen3_config)
    standalone.eval()

    # Load weights (strict=False: tie_word_embeddings may cause lm_head mismatch)
    missing, unexpected = standalone.load_state_dict(decoder_sd, strict=False)
    print(f"  Missing keys: {len(missing)} {missing}")
    print(f"  Unexpected keys: {len(unexpected)} {unexpected}")

    # Quick sanity check
    with torch.no_grad():
        dummy_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        output = standalone(input_ids=dummy_ids)
        assert output.logits.shape == (1, 5, 151936), (
            f"Expected [1, 5, 151936], got {list(output.logits.shape)}"
        )
    print("  Forward pass sanity check: OK")

    # Save standalone HF checkpoint
    save_dir = paths["decoder_standalone_dir"]
    os.makedirs(save_dir, exist_ok=True)
    standalone.save_pretrained(str(save_dir))
    print(f"  Saved to: {save_dir}")

    print(f"  Time: {elapsed():.1f}s")
    print()
    return str(save_dir)


# ============================================================================
# Step 5 -- Export decoder via optimum-intel (stateful + KV-cache)
# ============================================================================

def step5_export_decoder_stateful(standalone_dir: str, paths: dict, model_id: str):
    print("=" * 60)
    print("Step 5: Exporting decoder with optimum-intel (stateful=True)")
    print("=" * 60)
    elapsed = _timer()

    from optimum.intel import OVModelForCausalLM
    from transformers import AutoTokenizer

    export_dir = str(paths["decoder_stateful_ov_dir"])

    print(f"  Source: {standalone_dir}")
    print(f"  Target: {export_dir}")
    print("  This may take several minutes...")

    ov_model = OVModelForCausalLM.from_pretrained(
        standalone_dir,
        export=True,
        stateful=True,
        trust_remote_code=False,  # Standard Qwen3, no custom code needed
    )
    ov_model.save_pretrained(export_dir)
    print(f"  Exported to: {export_dir}")

    # Copy tokenizer to export dir
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.save_pretrained(standalone_dir)
    tokenizer.save_pretrained(export_dir)
    print("  Tokenizer saved to standalone and export dirs")

    # List exported files
    for f in sorted(os.listdir(export_dir)):
        fpath = os.path.join(export_dir, f)
        if os.path.isfile(fpath):
            size_mb = os.path.getsize(fpath) / 1024 / 1024
            print(f"    {f}: {size_mb:.1f} MB")

    print(f"  Time: {elapsed():.1f}s")
    print()
    return export_dir


# ============================================================================
# Step 6 -- IR surgery: remove input_ids, add inputs_embeds
# ============================================================================

def step6_ir_surgery(paths: dict):
    """Perform graph surgery on the stateful decoder IR.

    The NPUW_LLM plugin selects its main input by name priority:
      - If 'input_ids' exists  -> uses input_ids (int64 zeros -> garbage output)
      - If only 'inputs_embeds' -> uses inputs_embeds (correct path for ASR)

    Surgery steps:
      1. Find the Gather node that consumes input_ids (the embedding lookup)
      2. Create a new inputs_embeds Parameter with shape [1, ?, 1024]
      3. Redirect all Gather consumers to read from inputs_embeds instead
      4. Disconnect input_ids from all its consumers (ShapeOf, Convert, etc.)
      5. Rebuild the Model excluding input_ids entirely
      6. Save the result to decoder_stateful_embeds/
    """
    print("=" * 60)
    print("Step 6: IR surgery -- remove input_ids, add inputs_embeds")
    print("=" * 60)
    elapsed = _timer()

    import openvino as ov
    from openvino import opset13 as opset

    src_dir = paths["decoder_stateful_ov_dir"]
    dst_dir = paths["decoder_stateful_embeds_dir"]
    os.makedirs(dst_dir, exist_ok=True)

    xml_path = os.path.join(str(src_dir), "openvino_model.xml")
    core = ov.Core()
    model = core.read_model(xml_path)

    # ---- Identify the input_ids Parameter and the embedding Gather --------
    input_ids_param = None
    embed_gather = None

    for param in model.get_parameters():
        name = param.get_friendly_name()
        if name == "input_ids":
            input_ids_param = param
            break

    if input_ids_param is None:
        print("  ERROR: Could not find input_ids Parameter in the model.")
        print("  The model may have already been patched.")
        return

    # Walk consumers of input_ids to find the Gather (embedding lookup)
    for target_input in input_ids_param.output(0).get_target_inputs():
        node = target_input.get_node()
        # The Gather may be behind a Convert (int64 -> int32)
        if node.get_type_name() == "Convert":
            for t in node.output(0).get_target_inputs():
                child = t.get_node()
                if child.get_type_name() == "Gather":
                    embed_gather = child
                    break
        elif node.get_type_name() == "Gather":
            embed_gather = node
        if embed_gather is not None:
            break

    if embed_gather is None:
        print("  ERROR: Could not find embedding Gather node.")
        return

    gather_output = embed_gather.output(0)
    gather_shape = gather_output.get_partial_shape()
    print(f"  Found embedding Gather: {embed_gather.get_friendly_name()}")
    print(f"    Output shape: {gather_shape}")
    print(f"    Consumers: {len(gather_output.get_target_inputs())}")

    # ---- Determine hidden_size from the Gather output --------------------
    # Gather output shape is typically [1, ?, hidden_size]
    hidden_size = gather_shape[-1]
    if hidden_size.is_dynamic:
        hidden_size = 1024  # Qwen3-ASR-0.6B text decoder hidden_size
        print(f"  Hidden size is dynamic, defaulting to {hidden_size}")
    else:
        hidden_size = hidden_size.get_length()
        print(f"  Hidden size: {hidden_size}")

    # ---- Create new inputs_embeds Parameter ------------------------------
    inputs_embeds_param = opset.parameter(
        ov.PartialShape([1, -1, hidden_size]),
        dtype=np.float32,
        name="inputs_embeds",
    )
    inputs_embeds_param.get_output_tensor(0).set_names({"inputs_embeds"})
    print(f"  Created inputs_embeds Parameter: {inputs_embeds_param.get_partial_shape()}")

    # ---- Redirect all Gather consumers to inputs_embeds ------------------
    consumers = list(gather_output.get_target_inputs())
    print(f"  Redirecting {len(consumers)} consumer(s) of Gather -> inputs_embeds")
    for target_input in consumers:
        target_input.replace_source_output(inputs_embeds_param.output(0))

    # ---- Disconnect input_ids from ALL its consumers ---------------------
    # input_ids feeds: Convert->Gather (embedding), ShapeOf, etc.
    # We must disconnect everything so the parameter becomes isolated.
    input_ids_consumers = list(input_ids_param.output(0).get_target_inputs())
    print(f"  Disconnecting {len(input_ids_consumers)} consumer(s) of input_ids")
    for target_input in input_ids_consumers:
        node = target_input.get_node()
        node_type = node.get_type_name()

        if node_type == "ShapeOf":
            # ShapeOf(input_ids) is used for position calculations.
            # Replace with ShapeOf(inputs_embeds) -- first 2 dims are the same.
            shape_of_embeds = opset.shape_of(inputs_embeds_param, name="shapeof_embeds")
            # ShapeOf output: [1, seq_len, hidden] -> we only need [1, seq_len]
            # Gather the first 2 elements: indices [0, 1]
            gather_2d = opset.gather(
                shape_of_embeds,
                opset.constant(np.array([0, 1], dtype=np.int64)),
                opset.constant(np.int64(0)),
            )
            for shapeof_consumer in list(node.output(0).get_target_inputs()):
                shapeof_consumer.replace_source_output(gather_2d.output(0))
            print(f"    Replaced ShapeOf(input_ids) -> Gather(ShapeOf(inputs_embeds), [0,1])")

        elif node_type == "Convert":
            # Disconnect Convert node's consumers too (the embedding Gather, already redirected)
            for convert_consumer in list(node.output(0).get_target_inputs()):
                child = convert_consumer.get_node()
                if child.get_type_name() == "Gather":
                    # Already redirected above, but disconnect the Convert->Gather edge
                    pass  # Consumer already replaced
            print(f"    Disconnected Convert node: {node.get_friendly_name()}")

        else:
            print(f"    Disconnected {node_type} node: {node.get_friendly_name()}")

    # ---- Build new parameter list (excluding input_ids) ------------------
    old_params = list(model.get_parameters())
    new_params = [p for p in old_params if p.get_friendly_name() != "input_ids"]
    new_params.append(inputs_embeds_param)

    print(f"  Old parameters ({len(old_params)}):")
    for p in old_params:
        print(f"    {p.get_friendly_name()}: {p.get_partial_shape()}")
    print(f"  New parameters ({len(new_params)}):")
    for p in new_params:
        print(f"    {p.get_friendly_name()}: {p.get_partial_shape()}")

    # ---- Create new Model ------------------------------------------------
    results = list(model.get_results())
    sinks = list(model.get_sinks())

    new_model = ov.Model(
        results=results,
        sinks=sinks,
        parameters=new_params,
        name="decoder_stateful_embeds",
    )

    # Validate
    new_model.validate_nodes_and_infer_types()
    print("  Model validation: OK")

    # ---- Save ------------------------------------------------------------
    dst_xml = os.path.join(str(dst_dir), "openvino_model.xml")
    ov.save_model(new_model, dst_xml, compress_to_fp16=True)
    print(f"  Saved to: {dst_dir}")

    # Copy tokenizer + config files from the stateful_ov export
    for fname in os.listdir(str(src_dir)):
        if fname.startswith("openvino_model"):
            continue  # Skip the IR files we just wrote
        src_file = os.path.join(str(src_dir), fname)
        dst_file = os.path.join(str(dst_dir), fname)
        if os.path.isfile(src_file):
            shutil.copy2(src_file, dst_file)
    print("  Copied tokenizer and config files")

    # List output files
    for f in sorted(os.listdir(str(dst_dir))):
        fpath = os.path.join(str(dst_dir), f)
        if os.path.isfile(fpath):
            size_mb = os.path.getsize(fpath) / 1024 / 1024
            print(f"    {f}: {size_mb:.1f} MB")

    print(f"  Time: {elapsed():.1f}s")
    print()


# ============================================================================
# Step 7 -- Extract embed_tokens.npy
# ============================================================================

def step7_extract_embeddings(model, paths: dict):
    print("=" * 60)
    print("Step 7: Extracting embed_tokens.npy")
    print("=" * 60)
    elapsed = _timer()

    embed_weight = model.thinker.model.embed_tokens.weight.detach().cpu().numpy()
    print(f"  Shape: {embed_weight.shape}")  # [151936, 1024]

    npy_path = paths["embed_tokens_npy"]
    np.save(str(npy_path), embed_weight)
    size_mb = npy_path.stat().st_size / 1024 / 1024
    print(f"  Saved: {npy_path} ({size_mb:.1f} MB)")
    print(f"  Time: {elapsed():.1f}s")
    print()


# ============================================================================
# Step 8 -- Copy tokenizer / preprocessor files
# ============================================================================

def step8_copy_tokenizer(model_id: str, paths: dict):
    print("=" * 60)
    print("Step 8: Copying tokenizer and preprocessor files")
    print("=" * 60)
    elapsed = _timer()

    from transformers import AutoTokenizer

    dst_dir = paths["tokenizer_dir"]
    os.makedirs(dst_dir, exist_ok=True)

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.save_pretrained(str(dst_dir))
    print(f"  Tokenizer saved to: {dst_dir}")

    # Copy preprocessor_config.json (needed for WhisperFeatureExtractor)
    # Try to find it in the model_id path (local) or download it
    model_path = Path(model_id)
    preproc_src = model_path / "preprocessor_config.json" if model_path.is_dir() else None

    if preproc_src is not None and preproc_src.exists():
        shutil.copy2(str(preproc_src), str(dst_dir / "preprocessor_config.json"))
        print(f"  Copied preprocessor_config.json from local model")
    else:
        # Try downloading via huggingface_hub
        try:
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(
                repo_id=model_id,
                filename="preprocessor_config.json",
            )
            shutil.copy2(downloaded, str(dst_dir / "preprocessor_config.json"))
            print("  Downloaded preprocessor_config.json from HuggingFace")
        except Exception as e:
            print(f"  WARNING: Could not get preprocessor_config.json: {e}")

    # Copy config.json for reference
    config_src = model_path / "config.json" if model_path.is_dir() else None
    if config_src is not None and config_src.exists():
        shutil.copy2(str(config_src), str(dst_dir / "config.json"))
        print("  Copied config.json from local model")

    # List output files
    for f in sorted(os.listdir(str(dst_dir))):
        print(f"    {f}")

    print(f"  Time: {elapsed():.1f}s")
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare Qwen3-ASR models for OpenVINO NPU deployment",
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3-ASR-0.6B",
        help="HuggingFace model ID or local path (default: Qwen/Qwen3-ASR-0.6B)",
    )
    parser.add_argument(
        "--qwen-asr-src",
        default=None,
        help="Path to Qwen3-ASR project root containing qwen_asr/ package. "
             "If not specified, inferred from --model-id parent directory.",
    )
    args = parser.parse_args()

    # Ensure qwen_asr package is importable.
    # It lives in the Qwen3-ASR project repo, not in this project.
    qwen_asr_src = args.qwen_asr_src
    if qwen_asr_src is None:
        # Try to infer: if model-id is a local dir like .../Qwen3-ASR/Qwen3-ASR-0.6B,
        # then the parent (.../Qwen3-ASR) should contain qwen_asr/.
        model_path = Path(args.model_id)
        if model_path.is_dir():
            candidate = model_path.parent
            if (candidate / "qwen_asr").is_dir():
                qwen_asr_src = str(candidate)
    if qwen_asr_src and qwen_asr_src not in sys.path:
        sys.path.insert(0, qwen_asr_src)
        print(f"  Added to sys.path: {qwen_asr_src}")

    # Project root = translatorle/ (asr/scripts/ → asr/ → translatorle/)
    project_root = Path(__file__).resolve().parent.parent.parent
    paths = _resolve_paths(project_root)

    print()
    print("=" * 60)
    print("  Qwen3-ASR Model Preparation for OpenVINO NPU")
    print("=" * 60)
    print(f"  Model ID:     {args.model_id}")
    print(f"  Project root: {project_root}")
    print(f"  Output dir:   {paths['models_dir']}")
    print("=" * 60)
    print()

    os.makedirs(str(paths["models_dir"]), exist_ok=True)
    total_start = time.perf_counter()

    # Step 1: Load model
    model = step1_load_model(args.model_id)

    # Step 2: Export encoder
    step2_export_encoder(model, paths)

    # Step 3: Extract decoder weights
    decoder_sd = step3_extract_decoder_weights(model)

    # Step 7 (early): Extract embeddings while model is still in memory
    step7_extract_embeddings(model, paths)

    # Step 4: Create standalone decoder and save
    standalone_dir = step4_create_standalone_decoder(decoder_sd, paths)

    # Free original model memory
    del model, decoder_sd
    _free_memory()
    print("  [Freed original model memory]")
    print()

    # Step 5: Export decoder via optimum-intel
    step5_export_decoder_stateful(standalone_dir, paths, args.model_id)

    # Free standalone memory
    _free_memory()

    # Step 6: IR surgery
    step6_ir_surgery(paths)

    # Step 8: Copy tokenizer files
    step8_copy_tokenizer(args.model_id, paths)

    total_elapsed = time.perf_counter() - total_start
    print("=" * 60)
    print(f"  All done! Total time: {total_elapsed:.1f}s")
    print("=" * 60)
    print()
    print("Output files:")
    for key, path in sorted(paths.items()):
        if key == "models_dir":
            continue
        exists = path.exists() if isinstance(path, Path) else os.path.exists(path)
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {path}")
    print()


if __name__ == "__main__":
    main()
