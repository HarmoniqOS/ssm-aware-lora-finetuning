"""
LoRA + SSM Core Fine-Tuning for Granite 4.0-H-Micro (Hybrid Mamba-2/Transformer)

Hybrid PEFT approach:
    1. LoRA on all linear projection layers (attention, Mamba projections, MLP)
    2. Direct unfreezing of Mamba-2 SSM core parameters (A_log, D, dt_bias)

Granite 4.0-H-Micro uses Mamba-2's structured state space duality design.
The scalar-times-identity constraint on A collapses the state matrix to a
per-head scalar, yielding 1D params of shape [num_heads=64]. Sparse dimension
selection (full SDT) is unnecessary at this scale — direct unfreezing adds
only 6,912 trainable params across all 36 Mamba layers.

Architecture: granitehybrid — requires mamba_ssm ops monkey-patching
before model load.

Supported tasks:
    classification  — schema classification from file headers + sample rows
    mapping         — column name mapping (file columns → expected columns)
    rule_generation — SQL rule generation from column defs + CMMC objectives

Requirements:
    pip install torch transformers peft trl datasets accelerate bitsandbytes
    pip install mamba_ssm causal-conv1d

Usage:
    python train_lora.py --task classification --data-dir /workspace/data
    python train_lora.py --task all --data-dir /workspace/data --profile a100
    python train_lora.py --task all --data-dir /workspace/data --test
"""

# === Mamba SSM monkey-patching — MUST happen before any model import ===
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined, mamba_chunk_scan_combined
import mamba_ssm
mamba_ssm.selective_state_update = selective_state_update
mamba_ssm.mamba_chunk_scan_combined = mamba_chunk_scan_combined
mamba_ssm.mamba_split_conv1d_scan_combined = mamba_split_conv1d_scan_combined

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Paths (defaults — overridden by CLI args for remote training)
ASSESSOS_DIR = Path(__file__).parent.parent.parent
SYNTH_OUTPUT_DIR = ASSESSOS_DIR / "scripts" / "synth_pipeline" / "output"
ADAPTERS_DIR = ASSESSOS_DIR / "llm_models" / "adapters"

# Task → training data file mapping
TASK_DATA = {
    "classification": SYNTH_OUTPUT_DIR / "classification_training.jsonl",
    "mapping": SYNTH_OUTPUT_DIR / "mapping_training.jsonl",
    "rule_generation": SYNTH_OUTPUT_DIR / "rule_generation_training.jsonl",
}

# Base model
BASE_MODEL_ID = "ibm-granite/granite-4.0-h-micro"

# Hardware profiles
# NOTE: load_in_4bit=False for ALL profiles — bitsandbytes quantized weights
# are incompatible with Mamba fast kernels (shape mismatch in F.linear).
# Model is small enough (~2GB bf16) that full precision fits everywhere.
PROFILES = {
    "local_16gb": {
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "max_seq_length": 1280,
        "load_in_4bit": False,
    },
    "4090": {
        "batch_size": 8,
        "gradient_accumulation_steps": 4,
        "max_seq_length": 1280,
        "load_in_4bit": False,
    },
    "a100": {
        "batch_size": 32,
        "gradient_accumulation_steps": 1,
        "max_seq_length": 1280,
        "load_in_4bit": False,
    },
}

# Base training configuration
CONFIG = {
    "model_id": BASE_MODEL_ID,

    # LoRA config
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    # Granite 4.0-H-Micro module names:
    #   Attention (4 layers): self_attn.q_proj, k_proj, v_proj, o_proj
    #   Mamba (36 layers): mamba.in_proj, mamba.out_proj
    #   MLP (all 40 layers): shared_mlp.input_linear, shared_mlp.output_linear
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj", "input_linear", "output_linear"],

    # Training config (defaults — overridden by profile)
    "num_epochs": 3,
    "batch_size": 1,
    "gradient_accumulation_steps": 16,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.06,
    "max_seq_length": 1280,

    # Quantization — MUST be False for Mamba hybrid (bitsandbytes incompatible)
    "load_in_4bit": False,
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_quant_type": "nf4",
}


def check_dependencies():
    """Check if required packages are installed."""
    required = ["torch", "transformers", "peft", "bitsandbytes", "datasets", "accelerate", "trl"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing packages. Install with:\n  pip install {' '.join(missing)}")
        return False
    return True


def load_training_data(task: str, test_mode: bool = False) -> list[dict]:
    """Load JSONL training data for a task."""
    data_path = TASK_DATA.get(task)
    if not data_path or not data_path.exists():
        logger.error(f"Training data not found: {data_path}")
        logger.error("Run the synth pipeline first (Phase 2)")
        return []

    examples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    if test_mode:
        examples = examples[:100]

    logger.info(f"Loaded {len(examples)} training examples for task '{task}'")
    return examples


def format_for_training(examples: list[dict]) -> list[dict]:
    """Convert Alpaca-format examples to native Granite chat format.

    Uses Granite's native special tokens:
        <|start_of_role|>{role}<|end_of_role|>{content}<|end_of_text|>
    """
    formatted = []
    for ex in examples:
        instruction = ex.get("instruction", "")
        inp = ex.get("input", "")
        output = ex.get("output", "")

        # Build user message
        user_content = instruction
        if inp:
            user_content += f"\n\n{inp}"

        text = (
            f"<|start_of_role|>system<|end_of_role|>You are a compliance analysis assistant.<|end_of_text|>\n"
            f"<|start_of_role|>user<|end_of_role|>{user_content}<|end_of_text|>\n"
            f"<|start_of_role|>assistant<|end_of_role|>{output}<|end_of_text|>\n"
        )
        formatted.append({"text": text})

    return formatted


SSM_CORE_PARAMS = ["mamba.A_log", "mamba.D", "mamba.dt_bias"]


def apply_ssm_core_training(model):
    """
    Unfreeze SSM core parameters across all Mamba-2 layers.

    These are 1D tensors (not 2D matrices) due to Mamba-2's structured state
    space duality — the scalar-times-identity constraint on A collapses what
    would be a full state matrix down to a per-head scalar, hence shape [64].
    Sparse dimension selection is unnecessary at this scale.

    Parameters unfrozen per layer:
        A_log    [64] — log state transition diagonal (one per head)
        D        [64] — skip connection weight (one per head)
        dt_bias  [64] — timestep discretization bias (one per head)

    Total additional trainable params: 64 × 3 × num_mamba_layers
    """
    ssm_params_unfrozen = 0
    layers_modified = 0
    details = []

    for name, param in model.named_parameters():
        if any(x in name for x in SSM_CORE_PARAMS):
            param.requires_grad = True
            ssm_params_unfrozen += param.numel()
            layers_modified += 1
            details.append(f"  {name}: shape={list(param.shape)}, numel={param.numel()}")

    print(f"\n--- SSM Core Training ---")
    print(f"Unfrozen {layers_modified} SSM core parameters ({ssm_params_unfrozen:,} values)")
    for d in details[:9]:  # Show first 3 layers (9 params)
        print(d)
    if len(details) > 9:
        print(f"  ... and {len(details) - 9} more")
    print()

    return ssm_params_unfrozen, layers_modified


def generate_layer_coverage_report(model, output_dir: Path) -> str:
    """
    Generate and save a detailed layer coverage report showing which
    parameters are trained by LoRA, SSM core unfreezing, or frozen.
    """
    import torch

    lora_params = 0
    lora_names = []
    ssm_core_params = 0
    ssm_core_names = []
    frozen_params = 0
    frozen_names = []
    other_trainable_params = 0
    other_trainable_names = []

    # Classify every parameter
    for name, param in model.named_parameters():
        if "lora_" in name:
            lora_params += param.numel()
            lora_names.append(name)
        elif any(x in name for x in SSM_CORE_PARAMS) and param.requires_grad:
            ssm_core_params += param.numel()
            ssm_core_names.append(name)
        elif param.requires_grad:
            other_trainable_params += param.numel()
            other_trainable_names.append(name)
        else:
            frozen_params += param.numel()
            frozen_names.append(name)

    total_params = lora_params + ssm_core_params + other_trainable_params + frozen_params
    trainable_params = lora_params + ssm_core_params + other_trainable_params
    pct = 100.0 * trainable_params / total_params if total_params > 0 else 0

    # Count layer types by inspecting LoRA target names
    attn_lora = set()
    mamba_lora = set()
    mlp_lora = set()
    for n in lora_names:
        # Extract layer index from names like "base_model.model.model.layers.5.self_attn.q_proj.lora_A..."
        if "self_attn" in n:
            attn_lora.add(n.split("self_attn")[0])
        elif "mamba" in n:
            mamba_lora.add(n.split("mamba")[0])
        elif "shared_mlp" in n:
            mlp_lora.add(n.split("shared_mlp")[0])

    # Count Mamba layers with SSM core params
    ssm_layers = set()
    for n in ssm_core_names:
        # Extract layer prefix
        parts = n.split("mamba.")
        if len(parts) > 1:
            ssm_layers.add(parts[0])

    num_attn_layers = len(attn_lora)
    num_mamba_layers = max(len(mamba_lora), len(ssm_layers))
    num_mlp_layers = len(mlp_lora)
    total_layers = num_attn_layers + num_mamba_layers  # attention + mamba (MLP is in every layer)

    # Build report
    lines = []
    lines.append("=" * 60)
    lines.append("Granite 4.0-H-Micro Layer Coverage Report")
    lines.append("=" * 60)
    lines.append(f"Total layers: {total_layers} ({num_attn_layers} Transformer + {num_mamba_layers} Mamba-2)")
    lines.append("")
    lines.append(f"Transformer attention layers: {num_attn_layers}")
    lines.append(f"  └─ LoRA applied: q_proj, k_proj, v_proj, o_proj ✓")
    lines.append("")
    lines.append(f"Mamba-2 layers: {num_mamba_layers}")
    lines.append(f"  └─ LoRA applied: in_proj, out_proj ✓")
    lines.append(f"  └─ SSM core unfrozen: A_log, D, dt_bias ✓")
    lines.append(f"      Shape: [num_heads=64] per param (Mamba-2 SSD scalar constraint)")
    lines.append(f"      Total SSM core params: {ssm_core_params:,}")
    lines.append("")
    lines.append(f"MLP layers (shared across all {num_mlp_layers} layers):")
    lines.append(f"  └─ LoRA applied: input_linear, output_linear ✓")
    lines.append("")
    lines.append("-" * 60)
    lines.append("Parameter Budget")
    lines.append("-" * 60)
    lines.append(f"  LoRA adapter params:      {lora_params:>12,}")
    lines.append(f"  SSM core params:          {ssm_core_params:>12,}")
    if other_trainable_params > 0:
        lines.append(f"  Other trainable:          {other_trainable_params:>12,}")
    lines.append(f"  Frozen params:            {frozen_params:>12,}")
    lines.append(f"  ─────────────────────────────────────")
    lines.append(f"  Total model params:       {total_params:>12,}")
    lines.append(f"  Total trainable:          {trainable_params:>12,} ({pct:.2f}%)")
    lines.append("")

    # List frozen layer types (unique module types, not every param)
    frozen_types = set()
    for n in frozen_names:
        if "embed_tokens" in n:
            frozen_types.add("embed_tokens (embedding layer)")
        elif "norm" in n:
            frozen_types.add("layer norms")
        elif "lm_head" in n:
            frozen_types.add("lm_head (output projection)")
        elif "conv1d" in n:
            frozen_types.add("conv1d (Mamba convolution)")
        elif "self_attn" in n and "lora_" not in n:
            frozen_types.add("attention base weights (LoRA adapts these)")
        elif "mamba" in n and "lora_" not in n and not any(x in n for x in SSM_CORE_PARAMS):
            frozen_types.add("Mamba projection base weights (LoRA adapts these)")
        elif "shared_mlp" in n and "lora_" not in n:
            frozen_types.add("MLP base weights (LoRA adapts these)")

    if frozen_types:
        lines.append("Frozen layer categories:")
        for ft in sorted(frozen_types):
            lines.append(f"  • {ft}")
    lines.append("")
    lines.append("=" * 60)
    lines.append("End Coverage Report")
    lines.append("=" * 60)

    report = "\n".join(lines)

    # Save to file
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "layer_coverage_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved to: {report_path}")

    return report


def snapshot_ssm_params(model, label: str = "before") -> dict:
    """
    Snapshot SSM core parameter values for before/after comparison.
    Captures A_log, D, dt_bias from the first 3 Mamba layers.
    Returns a dict of {param_name: tensor_clone}.
    """
    import torch
    snapshot = {}
    target_layers = set()

    for name, param in model.named_parameters():
        if any(x in name for x in SSM_CORE_PARAMS):
            # Extract layer index
            try:
                layer_idx = int(name.split("layers.")[1].split(".")[0])
            except (IndexError, ValueError):
                continue
            if layer_idx < 3:
                snapshot[name] = param.detach().cpu().clone()
                target_layers.add(layer_idx)

    print(f"\n--- SSM Parameter Snapshot ({label}) ---")
    for name in sorted(snapshot.keys()):
        vals = snapshot[name]
        print(f"  {name}: mean={vals.float().mean():.6f}, std={vals.float().std():.6f}, "
              f"min={vals.float().min():.6f}, max={vals.float().max():.6f}")

    return snapshot


def compare_ssm_snapshots(before: dict, after: dict, output_dir: Path):
    """
    Compare before/after SSM parameter snapshots and log the delta.
    This is the empirical proof that SSM core parameters actually updated.
    """
    import torch

    lines = []
    lines.append("=" * 60)
    lines.append("SSM Core Parameter Delta (Before vs After Training)")
    lines.append("=" * 60)
    lines.append("")

    any_changed = False
    for name in sorted(before.keys()):
        if name not in after:
            lines.append(f"  {name}: MISSING from after snapshot")
            continue

        b = before[name].float()
        a = after[name].float()
        delta = (a - b).abs()

        mean_delta = delta.mean().item()
        max_delta = delta.max().item()
        pct_changed = (delta > 1e-8).float().mean().item() * 100

        status = "UPDATED ✓" if max_delta > 1e-8 else "UNCHANGED ✗"
        if max_delta > 1e-8:
            any_changed = True

        lines.append(f"  {name}:")
        lines.append(f"    before: mean={b.mean():.6f}, std={b.std():.6f}")
        lines.append(f"    after:  mean={a.mean():.6f}, std={a.std():.6f}")
        lines.append(f"    delta:  mean={mean_delta:.8f}, max={max_delta:.8f}, "
                      f"changed={pct_changed:.1f}%  [{status}]")
        lines.append("")

    lines.append("-" * 60)
    if any_changed:
        lines.append("RESULT: SSM core parameters UPDATED during training ✓")
        lines.append("The model's state transition dynamics were domain-adapted.")
    else:
        lines.append("WARNING: SSM core parameters DID NOT UPDATE ✗")
        lines.append("Check that requires_grad=True and learning rate is sufficient.")
    lines.append("=" * 60)

    report = "\n".join(lines)

    # Save and print
    output_dir.mkdir(parents=True, exist_ok=True)
    delta_path = output_dir / "ssm_parameter_delta.txt"
    delta_path.write_text(report, encoding="utf-8")
    print(f"\n{report}")
    print(f"\nSaved to: {delta_path}")

    return any_changed


def setup_model_and_tokenizer(config: dict):
    """Load model, apply LoRA to linear projections, unfreeze SSM core params."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model_id = config["model_id"]
    print(f"Loading model: {model_id}")

    if config["load_in_4bit"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, config["bnb_4bit_compute_dtype"]),
            bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Phase 1: LoRA on all linear projection layers
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Phase 2: Unfreeze SSM core parameters (A_log, D, dt_bias)
    # Mamba-2 SSD design: these are 1D [num_heads=64] tensors, not 2D matrices.
    # Direct unfreezing instead of sparse dimension selection.
    apply_ssm_core_training(model)

    # Required when mixing PEFT adapters with directly unfrozen parameters
    # under gradient checkpointing — prevents leaf tensor / in-place op errors
    # during backward pass.
    model.enable_input_require_grads()

    # Combined trainable parameter count (LoRA + SSM core)
    model.print_trainable_parameters()

    return model, tokenizer


def train_task(task: str, config: dict, output_dir: Optional[Path] = None,
               test_mode: bool = False, resume_from: str = None):
    """Train a LoRA + SSM core adapter for a specific task."""
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    # Load data
    examples = load_training_data(task, test_mode)
    if not examples:
        return

    # Format for training
    formatted = format_for_training(examples)
    dataset = Dataset.from_list(formatted)

    # Split 90/10
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Output directory
    if output_dir is None:
        output_dir = ADAPTERS_DIR / task
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"assessos-{task}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if test_mode:
        run_name += "-test"
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Setup model (LoRA + SSM core unfreezing)
    model, tokenizer = setup_model_and_tokenizer(config)

    # Layer coverage report — proof of complete training coverage
    generate_layer_coverage_report(model, run_dir)

    # Snapshot SSM core params BEFORE training
    ssm_before = snapshot_ssm_params(model, label="before")

    # SFTConfig (TRL 0.27+ — combines TrainingArguments + SFT settings)
    sft_config = SFTConfig(
        output_dir=str(run_dir),
        num_train_epochs=1 if test_mode else config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50 if not test_mode else 20,
        save_strategy="steps",
        save_steps=200 if not test_mode else 20,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        report_to="none",
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        # SFT-specific settings
        max_length=config["max_seq_length"],
        dataset_text_field="text",
        packing=True,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
    )

    # Train
    if resume_from:
        logger.info(f"Resuming training from: {resume_from}")
    else:
        logger.info(f"Starting training for task '{task}'...")
    trainer.train(resume_from_checkpoint=resume_from)

    # Snapshot SSM core params AFTER training — empirical proof of parameter drift
    ssm_after = snapshot_ssm_params(model, label="after")
    compare_ssm_snapshots(ssm_before, ssm_after, run_dir)

    # Save final adapter (LoRA weights)
    final_dir = run_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info(f"Saved LoRA adapter to {final_dir}")

    # Save SSM core params separately — PEFT only saves LoRA weights,
    # the unfrozen A_log/D/dt_bias changes are lost without this.
    import torch
    ssm_state = {}
    for name, param in model.named_parameters():
        if any(x in name for x in SSM_CORE_PARAMS) and param.requires_grad:
            ssm_state[name] = param.detach().cpu().clone()
    if ssm_state:
        ssm_path = final_dir / "ssm_core_params.pt"
        torch.save(ssm_state, str(ssm_path))
        logger.info(f"Saved {len(ssm_state)} SSM core params to {ssm_path} ({sum(p.numel() for p in ssm_state.values()):,} values)")

    # Save training metadata
    eval_results = trainer.evaluate()
    meta = {
        "task": task,
        "base_model": config["model_id"],
        "training_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
        "eval_loss": eval_results.get("eval_loss"),
        "config": {k: v for k, v in config.items() if k != "target_modules"},
        "lora_r": config["lora_r"],
        "lora_alpha": config["lora_alpha"],
        "target_modules": config["target_modules"],
        "ssm_core_params": ["A_log", "D", "dt_bias"],
        "training_method": "LoRA + SSM core unfreezing (Mamba-2 SSD)",
    }
    with open(run_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Eval loss: {eval_results.get('eval_loss', 'N/A')}")
    logger.info(f"Task '{task}' training complete!")


def main():
    global TASK_DATA
    parser = argparse.ArgumentParser(description="Train LoRA adapters for AssessOS")
    parser.add_argument("--task", choices=list(TASK_DATA.keys()) + ["all"], required=True, help="Which task to train")
    parser.add_argument("--profile", choices=list(PROFILES.keys()), help="Hardware profile")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--rank", type=int, help="LoRA rank (r)")
    parser.add_argument("--no-4bit", action="store_true", help="Disable QLoRA")
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    parser.add_argument("--data-dir", type=str, help="Override training data directory")
    parser.add_argument("--base-model", type=str, help="Override base model ID")
    parser.add_argument("--test", action="store_true", help="Quick test run with small subset")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")

    print("=" * 60)
    print("AssessOS LoRA Fine-tuning (Granite 4.0-H-Micro)")
    print("=" * 60)

    if not check_dependencies():
        return

    # Build config from base + profile + args
    config = CONFIG.copy()

    # Apply hardware profile (explicit or auto-detect)
    if args.profile:
        config.update(PROFILES[args.profile])
        print(f"Using profile: {args.profile}")
    else:
        try:
            import torch
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if vram_gb >= 40:
                    config.update(PROFILES["a100"])
                    print(f"Auto-detected {vram_gb:.1f}GB VRAM, using a100 profile")
                elif vram_gb >= 20:
                    config.update(PROFILES["4090"])
                    print(f"Auto-detected {vram_gb:.1f}GB VRAM, using 4090 profile")
                else:
                    config.update(PROFILES["local_16gb"])
                    print(f"Auto-detected {vram_gb:.1f}GB VRAM, using local_16gb profile")
        except Exception:
            print("Using default local_16gb profile")
            config.update(PROFILES["local_16gb"])

    # Override with explicit args
    if args.epochs:
        config["num_epochs"] = args.epochs
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.lr:
        config["learning_rate"] = args.lr
    if args.rank:
        config["lora_r"] = args.rank
    if args.no_4bit:
        config["load_in_4bit"] = False
    if args.base_model:
        config["model_id"] = args.base_model

    # Override data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
        TASK_DATA = {
            "classification": data_dir / "classification_training.jsonl",
            "mapping": data_dir / "mapping_training.jsonl",
            "rule_generation": data_dir / "rule_generation_training.jsonl",
        }

    output_dir = Path(args.output_dir) if args.output_dir else None

    print(f"  Batch size: {config['batch_size']}")
    print(f"  Gradient accumulation: {config['gradient_accumulation_steps']}")
    print(f"  Effective batch: {config['batch_size'] * config['gradient_accumulation_steps']}")
    print(f"  Max seq length: {config['max_seq_length']}")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  4-bit quantization: {config['load_in_4bit']}")
    print(f"  Packing: True")

    tasks = list(TASK_DATA.keys()) if args.task == "all" else [args.task]

    for task in tasks:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Training LoRA adapter: {task}")
        logger.info(f"{'=' * 60}")
        train_task(task, config, output_dir, test_mode=args.test, resume_from=args.resume)

    logger.info("\nAll training complete!")


if __name__ == "__main__":
    main()
