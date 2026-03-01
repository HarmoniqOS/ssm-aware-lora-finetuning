"""
Merge LoRA adapter weights into base model and convert to GGUF.

Pure tensor math — no mamba_ssm, no model loading. Just:
  W_merged = W_base + (alpha / r) * (B @ A)

Then calls llama.cpp's convert_hf_to_gguf.py on the merged checkpoint.

Usage:
    python merge_lora_to_gguf.py --version v2 --task classification
    python merge_lora_to_gguf.py --version v2 --task all
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent.parent
TRAINED_MODELS = PROJECT_DIR / "trained_models"

BASE_MODEL_PATH = Path.home() / ".cache" / "huggingface" / "hub" / \
    "models--ibm-granite--granite-4.0-h-micro" / "snapshots" / \
    "d5f01a3ea75f088947be3aae039f4ad52837dfde"

LLAMA_CPP_DIR = PROJECT_DIR / "llama.cpp"
CONVERT_SCRIPT = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"

ADAPTER_PATHS = {
    "v2": {
        "classification": TRAINED_MODELS / "output_original" / "assessos-classification-20260225_002242" / "final",
        "mapping": TRAINED_MODELS / "output_original" / "assessos-mapping-20260225_010209" / "final",
        "rule_generation": TRAINED_MODELS / "output_original" / "assessos-rule_generation-20260225_013243" / "final",
    },
    "v3": {
        "classification": TRAINED_MODELS / "output_new" / "assessos-classification-20260225_022013" / "final",
        "mapping": TRAINED_MODELS / "output_new" / "assessos-mapping-20260225_025856" / "final",
        "rule_generation": TRAINED_MODELS / "output_new" / "assessos-rule_generation-20260225_032932" / "final",
    },
}

OUTPUT_DIR = TRAINED_MODELS / "merged_gguf"


def load_base_tensors(base_path: Path) -> dict[str, torch.Tensor]:
    """Load all base model tensors from safetensors shards."""
    index_path = base_path / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)

    tensors = {}
    loaded_files = set()
    for tensor_name, shard_file in index["weight_map"].items():
        if shard_file not in loaded_files:
            shard_path = base_path / shard_file
            with safe_open(str(shard_path), framework="pt") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)
            loaded_files.add(shard_file)

    print(f"  Loaded {len(tensors)} base tensors from {len(loaded_files)} shards")
    return tensors


def load_adapter_tensors(adapter_path: Path) -> tuple[dict, dict]:
    """Load LoRA A/B tensors and adapter config."""
    config_path = adapter_path / "adapter_config.json"
    with open(config_path) as f:
        config = json.load(f)

    adapter_file = adapter_path / "adapter_model.safetensors"
    lora_tensors = {}
    with safe_open(str(adapter_file), framework="pt") as f:
        for key in f.keys():
            lora_tensors[key] = f.get_tensor(key)

    print(f"  Loaded {len(lora_tensors)} LoRA tensors (r={config['r']}, alpha={config['lora_alpha']})")
    return lora_tensors, config


def merge_weights(base_tensors: dict, lora_tensors: dict, config: dict) -> dict:
    """Apply W_merged = W_base + (alpha/r) * B @ A for each LoRA target."""
    r = config["r"]
    alpha = config["lora_alpha"]
    scale = alpha / r

    # Group LoRA tensors by target layer
    # Adapter key format: base_model.model.model.layers.0.mamba.in_proj.lora_A.weight
    # Base key format:    model.layers.0.mamba.in_proj.weight
    lora_pairs = {}
    for key in lora_tensors:
        if ".lora_A." in key:
            base_key = key.replace("base_model.model.", "").replace(".lora_A.weight", ".weight")
            b_key = key.replace(".lora_A.", ".lora_B.")
            if b_key in lora_tensors:
                lora_pairs[base_key] = (key, b_key)

    print(f"  Found {len(lora_pairs)} LoRA target layers, scale={scale}")

    merged = dict(base_tensors)
    applied = 0
    skipped = 0

    for base_key, (a_key, b_key) in sorted(lora_pairs.items()):
        if base_key not in merged:
            print(f"    SKIP: {base_key} not in base model")
            skipped += 1
            continue

        A = lora_tensors[a_key].float()  # [r, in_features]
        B = lora_tensors[b_key].float()  # [out_features, r]
        W = merged[base_key].float()

        delta = scale * (B @ A)
        merged[base_key] = (W + delta).to(base_tensors[base_key].dtype)
        applied += 1

    print(f"  Applied {applied} LoRA merges, skipped {skipped}")
    return merged


def save_merged_checkpoint(merged_tensors: dict, base_path: Path, output_dir: Path):
    """Save merged tensors as a HuggingFace checkpoint (for convert_hf_to_gguf.py)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy all config files from base model
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                   "special_tokens_map.json", "generation_config.json",
                   "merges.txt", "vocab.json", "chat_template.jinja"]:
        src = base_path / fname
        if src.exists():
            shutil.copy2(str(src), str(output_dir / fname))

    # Save merged weights as single safetensors file
    print(f"  Saving merged weights...")
    save_file(merged_tensors, str(output_dir / "model.safetensors"))

    # Write a simple index file
    weight_map = {k: "model.safetensors" for k in merged_tensors}
    index = {
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in merged_tensors.values())},
        "weight_map": weight_map,
    }
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    size_mb = (output_dir / "model.safetensors").stat().st_size / 1024 / 1024
    print(f"  Saved merged checkpoint: {output_dir} ({size_mb:.0f} MB)")


def convert_to_gguf(merged_dir: Path, output_gguf: Path):
    """Convert merged HF checkpoint to GGUF using llama.cpp."""
    if not CONVERT_SCRIPT.exists():
        print(f"  ERROR: {CONVERT_SCRIPT} not found")
        return False

    cmd = [
        sys.executable,
        str(CONVERT_SCRIPT),
        "--outfile", str(output_gguf),
        "--outtype", "q8_0",
        str(merged_dir),
    ]

    print(f"  Converting to GGUF: {output_gguf.name}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        size_mb = output_gguf.stat().st_size / 1024 / 1024
        print(f"  OK: {output_gguf.name} ({size_mb:.0f} MB)")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  FAILED:")
        if e.stderr:
            for line in e.stderr.strip().split('\n')[-10:]:
                print(f"    {line}")
        return False


def process_task(version: str, task: str, base_tensors: dict):
    """Merge and convert one adapter."""
    adapter_path = ADAPTER_PATHS[version][task]
    if not adapter_path.exists():
        print(f"  SKIP: adapter not found at {adapter_path}")
        return False

    merged_dir = OUTPUT_DIR / f"{version}_{task}_merged"
    output_gguf = OUTPUT_DIR / f"{version}_{task}-merged.gguf"

    if output_gguf.exists():
        size_mb = output_gguf.stat().st_size / 1024 / 1024
        print(f"  EXISTS: {output_gguf.name} ({size_mb:.0f} MB)")
        return True

    # Load adapter
    lora_tensors, config = load_adapter_tensors(adapter_path)

    # Merge
    merged = merge_weights(base_tensors, lora_tensors, config)

    # Save HF checkpoint
    save_merged_checkpoint(merged, BASE_MODEL_PATH, merged_dir)

    # Convert to GGUF
    success = convert_to_gguf(merged_dir, output_gguf)

    # Clean up merged HF checkpoint (keep GGUF only)
    if success and merged_dir.exists():
        shutil.rmtree(str(merged_dir))
        print(f"  Cleaned up temp dir: {merged_dir.name}")

    return success


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA into base model → GGUF")
    parser.add_argument("--version", choices=["v2", "v3"], required=True)
    parser.add_argument("--task", choices=["classification", "mapping", "rule_generation", "all"], required=True)
    args = parser.parse_args()

    print("=" * 60)
    print("  LoRA Merge -> GGUF")
    print("=" * 60)

    if not BASE_MODEL_PATH.exists():
        print(f"ERROR: Base model not found at {BASE_MODEL_PATH}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = ["classification", "mapping", "rule_generation"] if args.task == "all" else [args.task]

    # Load base tensors once (reuse across tasks)
    print("\nLoading base model tensors...")
    base_tensors = load_base_tensors(BASE_MODEL_PATH)

    results = {}
    for task in tasks:
        print(f"\n{'-' * 60}")
        print(f"  {args.version} / {task}")
        print(f"{'-' * 60}")
        results[task] = process_task(args.version, task, base_tensors)

    print(f"\n{'=' * 60}")
    for task, ok in results.items():
        print(f"  {task}: {'OK' if ok else 'FAILED'}")
    print(f"\nOutput: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
