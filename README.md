# SSM-Aware LoRA Fine-Tuning for Hybrid Mamba-Transformer Models

If you're fine-tuning IBM Granite 4.0-H-Micro (or similar hybrid Mamba-Transformer models) with LoRA, the standard `target_modules` configuration silently skips 90% of the model. This repo documents the problem and a fix, along with an SSM co-training technique that provides additional gains at zero inference cost.

**Full paper:** [`paper.md`](paper.md) | [`paper.pdf`](paper.pdf) <!-- | [arXiv](TODO) -->

## The Problem

Granite 4.0-H-Micro has 40 layers: 36 Mamba-2 SSM layers and 4 Transformer attention layers. The standard LoRA targets everyone uses (`gate_proj`, `up_proj`, `down_proj`) don't exist in this architecture. PEFT doesn't warn you. Training runs normally, loss goes down, adapters save fine — but the model barely improves because you only adapted the 4 attention layers.

The Mamba-2 layers use `in_proj`/`out_proj`, and the shared MLP uses `input_linear`/`output_linear`. You need to target those explicitly.

## Results

Evaluated on 1,000 test examples per task across three domain-specific tasks (document classification, schema mapping, structured code generation):

| Version | What Changed | Classification | Mapping | Code Gen (JSON) |
|---|---|---|---|---|
| **Base** | No training | 0.0% | 0.48% | 29.99% |
| **V1** | Wrong LoRA targets | 0.1% | 0.0% | 67.94% |
| **V2** | Correct LoRA targets | 40.8% | 96.03% | 98.58% |
| **V3** | + SSM co-training (params lost) | 52.2% | 97.65% | 98.91% |
| **V4** | + SSM co-training (params persisted) | **55.8%** | 96.73% | 98.8% |

The big jump is V1 → V2: just fixing the target modules. Everything after that is incremental.

## What's in This Repo

```
├── paper.md                                    # Full paper with methodology, results, analysis
├── paper.pdf                                   # Same content, formatted for print/arXiv
├── eval_results.json                           # Raw eval numbers (Base, V1, V2, V3)
├── eval_results_v4_lora_plus_ssm_fixed.json    # Raw eval numbers (V4)
├── configs/
│   ├── v1_wrong_targets.json                   # Default Transformer LoRA targets (broken)
│   ├── v2_correct_lora.json                    # Architecture-aware targets (fixed)
│   ├── v3_lora_plus_ssm.json                   # + SSM co-training
│   └── v4_lora_plus_ssm_fixed.json             # + SSM persistence fix
├── logs/
│   ├── v1_training.log                         # V1 training run
│   ├── v2_training.log                         # V2 training run
│   ├── v3_training.log                         # V3 SSM co-training run
│   └── v4_training.log                         # V4 SSM persistence run
├── scripts/
│   ├── train_lora_v1_baseline.py               # V1: Wrong targets (gate/up/down_proj)
│   ├── train_lora_original.py                  # V2: Correct hybrid targets, no SSM unfreezing
│   ├── train_lora.py                           # V3: LoRA + SSM co-training (save bug — SSM after save_model)
│   ├── train_lora_v4.py                        # V4: LoRA + SSM co-training (fix — SSM before save_model)
│   ├── merge_lora_to_gguf.py                   # V1-V3: LoRA-only merge → GGUF
│   ├── merge_lora_to_gguf_v4.py                # V4: LoRA merge + SSM injection → GGUF
│   ├── eval_models.py                          # Evaluation harness (llama-cpp-python, all versions)
│   ├── convert_all_adapters.py                 # Batch PEFT → GGUF conversion
│   ├── verify_ssm.py                           # V4 SSM persistence verification against base model
│   └── monkey_patch.py                         # mamba_ssm Triton kernel namespace exports
└── LICENSE                                     # Apache 2.0
```

## Quick Start

### Correct LoRA Target Modules

If you just want to fix your LoRA config for Granite hybrid models, here's what to change:

```python
# ❌ Standard Transformer targets — silently broken on hybrid models
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# ✅ Correct targets for Granite 4.0-H-Micro
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",   # attention (4 layers)
                  "in_proj", "out_proj",                      # Mamba-2 SSM (36 layers)
                  "input_linear", "output_linear"]            # shared MLP (40 layers)
```

This takes you from 851,968 trainable parameters (0.027% of model) to 28,823,552 (0.895%) — a 33.8x increase.

### SSM Co-Training (Optional)

After wrapping with PEFT, unfreeze the Mamba-2 core parameters:

```python
from peft import get_peft_model

model = get_peft_model(model, lora_config)

# Unfreeze SSM core params — adds 6,912 trainable parameters
for name, param in model.named_parameters():
    if any(x in name for x in ["mamba.A_log", "mamba.D", "mamba.dt_bias"]):
        param.requires_grad = True

model.enable_input_require_grads()
```

This reshapes the loss landscape during training and helps LoRA converge to better solutions. The effect persists even if you don't save the SSM parameters (PEFT's `save_pretrained()` silently drops them — see the paper for details on working around this).

### Mamba-2 Monkey Patch

Granite hybrid models require patching `mamba_ssm` before loading the model:

```python
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from mamba_ssm.ops.triton.ssd_combined import (
    mamba_split_conv1d_scan_combined,
    mamba_chunk_scan_combined,
)
import mamba_ssm

mamba_ssm.selective_state_update = selective_state_update
mamba_ssm.mamba_chunk_scan_combined = mamba_chunk_scan_combined
mamba_ssm.mamba_split_conv1d_scan_combined = mamba_split_conv1d_scan_combined
```

This must run before `AutoModelForCausalLM.from_pretrained()`.

## Key Findings

**1. Default LoRA targets are catastrophically wrong for hybrid models.** PEFT silently skips module names that don't exist in the model. V1 trained for 3 epochs with decreasing loss and produced adapters that did essentially nothing. There's no warning. We consider this a bug in PEFT and have filed an issue.

**2. Correct target selection is the single most important fix.** V2's architecture-aware targets account for the vast majority of improvement: 0% → 40.8% classification, 0% → 96% mapping accuracy, 30% → 98.6% valid JSON output.

**3. SSM co-training provides a free additional boost.** Unfreezing A_log, D, and dt_bias during LoRA training adds 6,912 parameters (0.024% of LoRA budget) and yields up to 28% relative improvement on classification. The co-training effect accounts for 76% of the V2→V4 classification gain — the SSM gradients help LoRA find better solutions even when the SSM changes themselves are discarded.

**4. SSM persistence benefits are task-dependent.** Persisting trained SSM values helps classification (+3.6 points over co-training alone) but has no meaningful effect on structured output tasks like schema mapping and code generation.

## Training Details

| Parameter | Value |
|---|---|
| Base model | `ibm-granite/granite-4.0-h-micro` (3.2B params) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Epochs | 3 |
| Learning rate | 2e-4 |
| Precision | bfloat16 (no quantization — bitsandbytes breaks Mamba kernels) |
| Hardware | NVIDIA RTX PRO 6000 96GB (training), RTX 4060 Ti 16GB (inference) |
| Framework | HuggingFace Transformers + PEFT 0.18.1 + SFTTrainer |
| Inference | llama-cpp-python 0.3.16 via GGUF |

Training data was generated synthetically using an LLM-based pipeline across 611 domain-specific document schemas. See the paper for full dataset details.

## Reproducing

1. Install dependencies: `pip install torch transformers peft trl datasets accelerate mamba_ssm causal-conv1d`
2. Prepare training data in JSONL format (instruction/input/output fields)
3. Run training:
   - V1: `python scripts/train_lora_v1_baseline.py --task classification --data-dir /path/to/data`
   - V2: `python scripts/train_lora_original.py --task classification --data-dir /path/to/data`
   - V3: `python scripts/train_lora.py --task classification --data-dir /path/to/data`
   - V4: `python scripts/train_lora_v4.py --task classification --data-dir /path/to/data`
4. Merge and convert:
   - V1-V3: `python scripts/merge_lora_to_gguf.py`
   - V4: `python scripts/merge_lora_to_gguf_v4.py`
5. Evaluate: `python scripts/eval_models.py`
6. Verify SSM persistence (V4): `python scripts/verify_ssm.py`

Note: `mamba_ssm` and `causal-conv1d` require building from source with CUDA. See [mamba](https://github.com/state-spaces/mamba) for installation instructions.

## Known Issues

- **PEFT silent module skipping**: PEFT does not warn when `target_modules` matches an unusually low number of modules. If you specify `gate_proj` on a model that doesn't have it, training proceeds silently with no indication that anything is wrong.
- **PEFT silent parameter loss**: `save_pretrained()` only saves LoRA adapter weights. Any non-LoRA parameters you've unfrozen and trained (like SSM core params) are silently discarded. You need to save them separately before calling `save_pretrained()`.
- **QLoRA incompatibility**: bitsandbytes 4-bit quantization breaks Mamba-2's Triton kernels due to shape mismatches in `F.linear`. Use full bfloat16 instead — the model is small enough (3.2B) to fit in memory without quantization.

## Citation

```bibtex
@article{ford2026ssm_aware_lora,
  title={SSM-Aware Fine-Tuning for Hybrid Mamba-Transformer Models: A Comparative Study on Granite 4.0-H-Micro},
  author={Ford, Cody},
  year={2026},
  url={https://github.com/HarmoniqOS/ssm-aware-lora-finetuning}
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
