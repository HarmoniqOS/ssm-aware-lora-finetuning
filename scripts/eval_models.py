"""
Model Evaluation Harness — Compare V1/V2/V3 LoRA Adapters (llama-cpp-python)

Loads the base Granite GGUF model with each adapter version and runs
inference on held-out test examples. No mamba_ssm required.

Scoring:
    Classification — exact match (normalized) on schema label
    Mapping        — JSON parse rate + field-level mapping accuracy
    Rule Generation — JSON parse rate + structural validity checks

Usage:
    python eval_models.py --samples 100
    python eval_models.py --task classification --samples 50
    python eval_models.py --samples 200 --output results.txt
"""

import argparse
import json
import random
import time
from pathlib import Path

from llama_cpp import Llama

# ─── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
ASSESSOS_DIR = SCRIPT_DIR.parent.parent
PROJECT_DIR = ASSESSOS_DIR.parent

BASE_MODEL = ASSESSOS_DIR / "llm_models" / "granite-4.0-h-micro-Q4_K_M.gguf"
GGUF_ADAPTERS_DIR = PROJECT_DIR / "trained_models" / "gguf_adapters"
SYNTH_DATA_DIR = ASSESSOS_DIR / "scripts" / "synth_pipeline" / "output"

TASK_DATA_FILES = {
    "classification": "classification_training.jsonl",
    "mapping": "mapping_training.jsonl",
    "rule_generation": "rule_generation_training.jsonl",
}

# Adapter GGUF files per version per task
# "lora:" prefix = runtime LoRA adapter on base model
# "merged:" prefix = standalone merged GGUF (no adapter needed)
VERSIONS = {
    "Base_no_training": {
        "classification": "base:",
        "mapping": "base:",
        "rule_generation": "base:",
    },
    "V1_wrong_targets": {
        "classification": "lora:v1_classification-lora.gguf",
        "mapping": "lora:v1_mapping-lora.gguf",
        "rule_generation": "lora:v1_rule_generation-lora.gguf",
    },
    "V2_correct_lora": {
        "classification": "merged:v2_classification-merged.gguf",
        "mapping": "merged:v2_mapping-merged.gguf",
        "rule_generation": "merged:v2_rule_generation-merged.gguf",
    },
    "V3_lora_plus_ssm": {
        "classification": "merged:v3_classification-merged.gguf",
        "mapping": "merged:v3_mapping-merged.gguf",
        "rule_generation": "merged:v3_rule_generation-merged.gguf",
    },
    "V4_lora_plus_ssm_fixed": {
        "classification": "merged:v4_classification-merged.gguf",
        "mapping": "merged:v4_mapping-merged.gguf",
        "rule_generation": "merged:v4_rule_generation-merged.gguf",
    },
}

MERGED_GGUF_DIR = PROJECT_DIR / "trained_models" / "merged_gguf"


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_test_examples(data_dir: Path, task: str, n_samples: int, seed: int = 42) -> list[dict]:
    """Load JSONL and return a deterministic sample from the eval split."""
    data_path = data_dir / TASK_DATA_FILES[task]
    if not data_path.exists():
        print(f"  WARNING: {data_path} not found")
        return []

    all_examples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_examples.append(json.loads(line))

    # Reproduce the 90/10 split from training (same seed=42)
    rng = random.Random(seed)
    indices = list(range(len(all_examples)))
    rng.shuffle(indices)
    split_point = int(len(indices) * 0.9)
    eval_indices = indices[split_point:]
    eval_examples = [all_examples[i] for i in eval_indices]

    # Sample from eval set
    if n_samples < len(eval_examples):
        rng2 = random.Random(seed + 1)
        eval_examples = rng2.sample(eval_examples, n_samples)

    print(f"  Loaded {len(eval_examples)} test examples for '{task}' (from {len(all_examples)} total, eval split)")
    return eval_examples


# ─── Prompt building ──────────────────────────────────────────────────────────

def build_prompt(example: dict) -> str:
    """Build Granite chat-format prompt (no assistant response — model generates that)."""
    instruction = example.get("instruction", "")
    inp = example.get("input", "")

    user_content = instruction
    if inp:
        user_content += f"\n\n{inp}"

    prompt = (
        f"<|start_of_role|>system<|end_of_role|>You are a compliance analysis assistant.<|end_of_text|>\n"
        f"<|start_of_role|>user<|end_of_role|>{user_content}<|end_of_text|>\n"
        f"<|start_of_role|>assistant<|end_of_role|>"
    )
    return prompt


# ─── Scoring functions ────────────────────────────────────────────────────────

def score_classification(prediction: str, expected: str) -> dict:
    pred_clean = prediction.strip().lower().split("\n")[0].strip()
    # Remove any trailing special tokens or whitespace
    for tok in ["<|end_of_text|>", "<|end_of_role|>"]:
        pred_clean = pred_clean.replace(tok, "").strip()
    exp_clean = expected.strip().lower()

    exact_match = pred_clean == exp_clean
    contains_match = exp_clean in pred_clean

    return {
        "exact_match": exact_match,
        "contains_match": contains_match,
        "predicted": pred_clean[:100],
        "expected": exp_clean,
    }


def score_mapping(prediction: str, expected: str) -> dict:
    result = {
        "json_valid": False,
        "mapping_accuracy": 0.0,
        "unmapped_accuracy": 0.0,
    }

    try:
        exp_obj = json.loads(expected)
    except json.JSONDecodeError:
        return result

    try:
        pred_text = prediction.strip()
        start = pred_text.find("{")
        end = pred_text.rfind("}") + 1
        if start >= 0 and end > start:
            pred_obj = json.loads(pred_text[start:end])
        else:
            return result
        result["json_valid"] = True
    except json.JSONDecodeError:
        return result

    exp_mapping = exp_obj.get("mapping", {})
    pred_mapping = pred_obj.get("mapping", {})
    if exp_mapping:
        correct = sum(1 for k, v in exp_mapping.items() if pred_mapping.get(k) == v)
        result["mapping_accuracy"] = correct / len(exp_mapping)

    exp_unmapped = set(exp_obj.get("unmapped_expected", []))
    pred_unmapped = set(pred_obj.get("unmapped_expected", []))
    if exp_unmapped or pred_unmapped:
        if exp_unmapped == pred_unmapped:
            result["unmapped_accuracy"] = 1.0
        elif exp_unmapped:
            overlap = exp_unmapped & pred_unmapped
            result["unmapped_accuracy"] = len(overlap) / len(exp_unmapped)
    else:
        result["unmapped_accuracy"] = 1.0

    return result


def score_rule_generation(prediction: str, expected: str) -> dict:
    result = {
        "json_valid": False,
        "has_violation_type": False,
        "has_severity": False,
        "has_where_clause": False,
        "has_objective_ids": False,
        "severity_correct": False,
        "violation_type_correct": False,
    }

    try:
        exp_obj = json.loads(expected)
    except json.JSONDecodeError:
        return result

    try:
        pred_text = prediction.strip()
        start = pred_text.find("{")
        end = pred_text.rfind("}") + 1
        if start >= 0 and end > start:
            pred_obj = json.loads(pred_text[start:end])
        else:
            return result
        result["json_valid"] = True
    except json.JSONDecodeError:
        return result

    result["has_violation_type"] = "violation_type" in pred_obj
    result["has_severity"] = "severity" in pred_obj
    result["has_where_clause"] = "where_clause" in pred_obj
    result["has_objective_ids"] = "objective_ids" in pred_obj and isinstance(pred_obj.get("objective_ids"), list)

    if result["has_severity"]:
        result["severity_correct"] = pred_obj.get("severity", "").upper() == exp_obj.get("severity", "").upper()
    if result["has_violation_type"]:
        result["violation_type_correct"] = pred_obj.get("violation_type", "").lower() == exp_obj.get("violation_type", "").lower()

    return result


SCORERS = {
    "classification": score_classification,
    "mapping": score_mapping,
    "rule_generation": score_rule_generation,
}

MAX_TOKENS_PER_TASK = {
    "classification": 64,
    "mapping": 512,
    "rule_generation": 512,
}


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_version(base_model_path: Path, model_spec: dict, task: str,
                     examples: list[dict]) -> dict:
    """Load model (merged GGUF or base+adapter), run inference, return scores."""

    if model_spec["mode"] == "base":
        # Raw base model — no adapter, no merge
        print(f"    Loading base model (GPU): {model_spec['path'].name}")
        llm = Llama(
            model_path=str(model_spec["path"]),
            n_ctx=2048,
            n_gpu_layers=-1,
            verbose=False,
        )
    elif model_spec["mode"] == "merged":
        # Merged models have no LoRA adapter — safe to use GPU
        print(f"    Loading merged model (GPU): {model_spec['path'].name}")
        llm = Llama(
            model_path=str(model_spec["path"]),
            n_ctx=2048,
            n_gpu_layers=-1,
            verbose=False,
        )
    else:
        # Runtime LoRA — V1 (32 tensors) works fine on GPU
        print(f"    Loading base + LoRA adapter (GPU): {model_spec['path'].name}")
        llm = Llama(
            model_path=str(base_model_path),
            lora_path=str(model_spec["path"]),
            n_ctx=2048,
            n_gpu_layers=-1,
            verbose=False,
        )

    scorer = SCORERS[task]
    max_tokens = MAX_TOKENS_PER_TASK[task]
    all_scores = []
    errors = 0

    t0 = time.time()
    for i, ex in enumerate(examples):
        prompt = build_prompt(ex)
        expected = ex.get("output", "")

        # Reset KV cache between examples
        llm.reset()

        try:
            output = llm(
                prompt,
                max_tokens=max_tokens,
                stop=["<|end_of_text|>", "<|start_of_role|>"],
                temperature=0.0,
                echo=False,
            )
            prediction = output["choices"][0]["text"]
            score = scorer(prediction, expected)
            all_scores.append(score)

            # Debug: show first 3 predictions
            if i < 3:
                print(f"      --- Example {i} ---")
                print(f"      Expected: {expected[:150]}")
                print(f"      Got:      {prediction[:150]}")
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"      Error on example {i}: {e}")

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"      {i+1}/{len(examples)} ({rate:.1f} ex/s)")

    elapsed = time.time() - t0

    # Free model
    del llm

    # Aggregate
    agg = aggregate_scores(task, all_scores)
    agg["total_examples"] = len(examples)
    agg["errors"] = errors
    agg["elapsed_seconds"] = round(elapsed, 1)
    agg["examples_per_second"] = round(len(examples) / elapsed, 2) if elapsed > 0 else 0

    return agg


def aggregate_scores(task: str, scores: list[dict]) -> dict:
    if not scores:
        return {"error": "no scores"}

    n = len(scores)

    if task == "classification":
        exact = sum(1 for s in scores if s["exact_match"]) / n
        contains = sum(1 for s in scores if s["contains_match"]) / n
        return {
            "exact_match": round(exact * 100, 2),
            "contains_match": round(contains * 100, 2),
        }
    elif task == "mapping":
        json_valid = sum(1 for s in scores if s["json_valid"]) / n
        mapping_acc = sum(s["mapping_accuracy"] for s in scores) / n
        unmapped_acc = sum(s["unmapped_accuracy"] for s in scores) / n
        return {
            "json_valid_pct": round(json_valid * 100, 2),
            "mapping_accuracy": round(mapping_acc * 100, 2),
            "unmapped_accuracy": round(unmapped_acc * 100, 2),
        }
    elif task == "rule_generation":
        json_valid = sum(1 for s in scores if s["json_valid"]) / n
        has_vt = sum(1 for s in scores if s["has_violation_type"]) / n
        has_sev = sum(1 for s in scores if s["has_severity"]) / n
        has_wc = sum(1 for s in scores if s["has_where_clause"]) / n
        has_oid = sum(1 for s in scores if s["has_objective_ids"]) / n
        sev_correct = sum(1 for s in scores if s["severity_correct"]) / n
        vt_correct = sum(1 for s in scores if s["violation_type_correct"]) / n
        return {
            "json_valid_pct": round(json_valid * 100, 2),
            "has_all_fields_pct": round(min(has_vt, has_sev, has_wc, has_oid) * 100, 2),
            "severity_correct": round(sev_correct * 100, 2),
            "violation_type_correct": round(vt_correct * 100, 2),
        }
    return {}


# ─── Report ───────────────────────────────────────────────────────────────────

def print_comparison_table(results: dict, output_path: Path = None):
    lines = []
    lines.append("")
    lines.append("=" * 80)
    version_names = " vs ".join(results.keys())
    lines.append(f"  MODEL EVALUATION RESULTS - {version_names}")
    lines.append("=" * 80)

    tasks = ["classification", "mapping", "rule_generation"]

    for task in tasks:
        lines.append(f"\n{'-' * 80}")
        lines.append(f"  TASK: {task.upper()}")
        lines.append(f"{'-' * 80}")

        for version_name, task_results in results.items():
            if task not in task_results:
                continue
            r = task_results[task]
            lines.append(f"\n  {version_name}:")

            if "error" in r:
                lines.append(f"    ERROR: {r['error']}")
                continue

            for key, val in r.items():
                if key in ("total_examples", "errors", "elapsed_seconds", "examples_per_second"):
                    continue
                lines.append(f"    {key:30s}: {val}%")

            lines.append(f"    {'elapsed':30s}: {r.get('elapsed_seconds', '?')}s ({r.get('examples_per_second', '?')} ex/s)")

    # Quick comparison
    lines.append(f"\n{'=' * 80}")
    lines.append("  QUICK COMPARISON (primary metric per task)")
    lines.append(f"{'=' * 80}")

    primary_metrics = {
        "classification": "exact_match",
        "mapping": "mapping_accuracy",
        "rule_generation": "json_valid_pct",
    }

    for task in tasks:
        metric = primary_metrics[task]
        lines.append(f"\n  {task} ({metric}):")
        for version_name, task_results in results.items():
            if task in task_results and metric in task_results[task]:
                val = task_results[task][metric]
                lines.append(f"    {version_name:30s}: {val}%")

    lines.append(f"\n{'=' * 80}")

    report = "\n".join(lines)
    print(report)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nSaved report to {output_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate V1/V2/V3 LoRA adapters via llama-cpp-python")
    parser.add_argument("--task", choices=["classification", "mapping", "rule_generation", "all"], default="all")
    parser.add_argument("--samples", type=int, default=100, help="Test examples per task")
    parser.add_argument("--output", type=str, default=None, help="Path to save results report")
    parser.add_argument("--data-dir", type=str, default=None, help="Override training data dir")
    parser.add_argument("--adapters-dir", type=str, default=None, help="Override GGUF adapters dir")
    parser.add_argument("--base-model", type=str, default=None, help="Override base model GGUF path")
    parser.add_argument("--n-gpu-layers", type=int, default=0, help="GPU layers (0 = CPU, -1 = all). LoRA adapters may not work with GPU offloading on quantized models.")
    parser.add_argument("--versions", nargs="+", default=None, help="Only eval specific versions (e.g. V1_wrong_targets V2_correct_lora)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else SYNTH_DATA_DIR
    adapters_dir = Path(args.adapters_dir) if args.adapters_dir else GGUF_ADAPTERS_DIR
    base_model = Path(args.base_model) if args.base_model else BASE_MODEL

    print("=" * 60)
    print("  AssessOS Model Evaluation (llama-cpp-python)")
    print("=" * 60)
    print(f"  Base model: {base_model}")
    print(f"  Adapters:   {adapters_dir}")
    print(f"  Data:       {data_dir}")
    print(f"  Samples:    {args.samples}")

    if not base_model.exists():
        print(f"ERROR: Base model not found: {base_model}")
        return

    tasks = ["classification", "mapping", "rule_generation"] if args.task == "all" else [args.task]

    # Filter versions and resolve paths
    versions_to_eval = {}
    for name, adapters in VERSIONS.items():
        if args.versions and name not in args.versions:
            continue
        available = {}
        for task, spec in adapters.items():
            mode, filename = spec.split(":", 1)
            if mode == "base":
                available[task] = {"mode": "base", "path": base_model}
            elif mode == "merged":
                path = MERGED_GGUF_DIR / filename
                if path.exists():
                    available[task] = {"mode": mode, "path": path}
            else:
                path = adapters_dir / filename
                if path.exists():
                    available[task] = {"mode": mode, "path": path}
        if available:
            versions_to_eval[name] = available
            print(f"  {name}: {', '.join(available.keys())}")
        else:
            print(f"  {name}: NO models found")

    if not versions_to_eval:
        print("ERROR: No models found. Check paths.")
        return

    # Run evaluations
    all_results = {}

    for version_name, task_specs in versions_to_eval.items():
        print(f"\n{'=' * 60}")
        print(f"  Evaluating: {version_name}")
        print(f"{'=' * 60}")
        all_results[version_name] = {}

        for task in tasks:
            if task not in task_specs:
                print(f"\n  [{task}] No model, skipping")
                continue

            print(f"\n  [{task}]")
            examples = load_test_examples(data_dir, task, args.samples)
            if not examples:
                continue

            scores = evaluate_version(
                base_model, task_specs[task], task, examples
            )
            all_results[version_name][task] = scores

            # Print immediately
            print(f"    Results: {json.dumps({k: v for k, v in scores.items() if k not in ('errors', 'elapsed_seconds', 'examples_per_second', 'total_examples')})}")

    # Final comparison — version-aware output to avoid overwriting previous results
    if args.output:
        output_path = Path(args.output)
    elif args.versions:
        version_tag = "_".join(v.lower() for v in args.versions)
        output_path = adapters_dir.parent / f"eval_results_{version_tag}.txt"
    else:
        output_path = adapters_dir.parent / "eval_results.txt"
    print_comparison_table(all_results, output_path)

    # Save raw JSON
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved raw results to {json_path}")


if __name__ == "__main__":
    main()
