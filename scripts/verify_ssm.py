import torch, glob, json
from safetensors import safe_open

base_snap = glob.glob("/root/.cache/huggingface/hub/models--ibm-granite--granite-4.0-h-micro/snapshots/*")[0]

for task in ["classification", "mapping", "rule_generation"]:
    matches = sorted(glob.glob(f"/workspace/output_v4/assessos-{task}-*/final/ssm_core_params.pt"))
    if not matches:
        print(f"\n{task}: NO ssm_core_params.pt found!")
        continue
    ssm_path = matches[-1]
    print(f"\n{'='*60}")
    print(f"{task}: {ssm_path}")
    ssm = torch.load(ssm_path, map_location="cpu", weights_only=True)
    print(f"  Keys: {len(ssm)}")
    for name, trained in sorted(ssm.items()):
        base_key = name.replace("base_model.model.", "")
        if "A_log" in name and "layers.0." in name:
            with open(f"{base_snap}/model.safetensors.index.json") as f:
                idx = json.load(f)
            shard = idx["weight_map"].get(base_key)
            if shard:
                with safe_open(f"{base_snap}/{shard}", framework="pt") as f:
                    base_val = f.get_tensor(base_key)
                diff = (trained.float() - base_val.float()).abs().mean().item()
                print(f"  Sample: {base_key}")
                print(f"    Base mean:    {base_val.float().mean().item():.6f}")
                print(f"    Trained mean: {trained.float().mean().item():.6f}")
                print(f"    Mean abs diff: {diff:.6f}")
                print(f"    Changed: {'YES' if diff > 1e-6 else 'NO'}")
            break
