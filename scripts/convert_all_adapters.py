"""
Batch convert V1/V2/V3 PEFT adapters to GGUF format for llama-cpp-python.

Uses llama.cpp's convert_lora_to_gguf.py — no mamba_ssm required.

Usage:
    python convert_all_adapters.py
"""

import subprocess
import sys
from pathlib import Path

LLAMA_CPP_DIR = Path(__file__).parent.parent.parent.parent / "llama.cpp"
CONVERT_SCRIPT = LLAMA_CPP_DIR / "convert_lora_to_gguf.py"
BASE_MODEL_PATH = Path.home() / ".cache" / "huggingface" / "hub" / "models--ibm-granite--granite-4.0-h-micro" / "snapshots" / "d5f01a3ea75f088947be3aae039f4ad52837dfde"
TRAINED_MODELS = Path(__file__).parent.parent.parent.parent / "trained_models"
OUTPUT_DIR = TRAINED_MODELS / "gguf_adapters"

# All adapters to convert: (version_label, task, adapter_path)
ADAPTERS = [
    # V1 — wrong LoRA targets
    ("v1", "classification", TRAINED_MODELS / "lm_models" / "adapters" / "classification" / "assessos-classification-20260225_113640" / "final"),
    ("v1", "mapping", TRAINED_MODELS / "lm_models" / "adapters" / "mapping" / "assessos-mapping-20260225_120248" / "final"),
    ("v1", "rule_generation", TRAINED_MODELS / "lm_models" / "adapters" / "rule_generation" / "assessos-rule_generation-20260225_122323" / "final"),
    # V2 — correct LoRA targets, no SSM
    ("v2", "classification", TRAINED_MODELS / "output_original" / "assessos-classification-20260225_002242" / "final"),
    ("v2", "mapping", TRAINED_MODELS / "output_original" / "assessos-mapping-20260225_010209" / "final"),
    ("v2", "rule_generation", TRAINED_MODELS / "output_original" / "assessos-rule_generation-20260225_013243" / "final"),
    # V3 — correct LoRA + SSM core unfreezing (LoRA weights only in GGUF)
    ("v3", "classification", TRAINED_MODELS / "output_new" / "assessos-classification-20260225_022013" / "final"),
    ("v3", "mapping", TRAINED_MODELS / "output_new" / "assessos-mapping-20260225_025856" / "final"),
    ("v3", "rule_generation", TRAINED_MODELS / "output_new" / "assessos-rule_generation-20260225_032932" / "final"),
]


def convert_adapter(version: str, task: str, adapter_path: Path, output_dir: Path) -> bool:
    output_file = output_dir / f"{version}_{task}-lora.gguf"

    if not adapter_path.exists():
        print(f"  SKIP: {adapter_path} not found")
        return False

    if output_file.exists():
        print(f"  EXISTS: {output_file.name} ({output_file.stat().st_size / 1024 / 1024:.1f} MB)")
        return True

    cmd = [
        sys.executable,
        str(CONVERT_SCRIPT),
        "--base", str(BASE_MODEL_PATH),
        "--outfile", str(output_file),
        "--outtype", "f16",
        str(adapter_path),
    ]

    print(f"  Converting {version}/{task}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        size_mb = output_file.stat().st_size / 1024 / 1024
        print(f"  OK: {output_file.name} ({size_mb:.1f} MB)")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  FAILED: {version}/{task}")
        if e.stderr:
            for line in e.stderr.strip().split('\n')[-5:]:
                print(f"    {line}")
        return False


def main():
    print("=" * 60)
    print("  Batch GGUF Adapter Conversion")
    print("=" * 60)

    # Verify prerequisites
    if not CONVERT_SCRIPT.exists():
        print(f"ERROR: convert_lora_to_gguf.py not found at {CONVERT_SCRIPT}")
        sys.exit(1)
    if not BASE_MODEL_PATH.exists():
        print(f"ERROR: Base model not found at {BASE_MODEL_PATH}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {OUTPUT_DIR}\n")

    results = {}
    for version, task, adapter_path in ADAPTERS:
        label = f"{version}/{task}"
        print(f"\n[{label}]")
        results[label] = convert_adapter(version, task, adapter_path, OUTPUT_DIR)

    # Summary
    print(f"\n{'=' * 60}")
    print("  Conversion Summary")
    print(f"{'=' * 60}")
    for label, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {label:30s} {status}")

    print(f"\nGGUF adapters saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
