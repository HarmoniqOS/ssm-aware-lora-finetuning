"""
Mamba-2 SSM Triton Kernel Monkey-Patch for Granite 4.0-H-Micro

WHAT:
    Exports three Triton kernel operations from mamba_ssm submodules to the
    top-level mamba_ssm namespace, where HuggingFace transformers expects them.

WHY:
    IBM's Granite 4.0-H-Micro uses the `granitehybrid` architecture — a hybrid
    Mamba-2/Transformer design with 36 Mamba-2 SSD layers and 4 global attention
    layers. When HuggingFace transformers loads this architecture, the
    GraniteHybridMambaLayer class resolves Mamba-2 ops via:

        mamba_ssm.selective_state_update
        mamba_ssm.mamba_chunk_scan_combined
        mamba_ssm.mamba_split_conv1d_scan_combined

    However, the mamba_ssm package (https://github.com/state-spaces/mamba) does
    NOT export these at the top level. They live in submodules:

        mamba_ssm.ops.triton.selective_state_update.selective_state_update
        mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined
        mamba_ssm.ops.triton.ssd_combined.mamba_split_conv1d_scan_combined

    Without this patch, model loading fails with:

        AttributeError: module 'mamba_ssm' has no attribute 'selective_state_update'

WHEN:
    This patch MUST be applied BEFORE any call to AutoModelForCausalLM.from_pretrained()
    or any other transformers model loading that triggers GraniteHybridMambaLayer
    initialization. Import order matters:

        import monkey_patch          # <-- first
        monkey_patch.apply()
        from transformers import ... # <-- then model imports

    If you import the model first, the Mamba layers fail to initialize and the
    error is unrecoverable in the same process.

OPERATIONS PATCHED:
    1. selective_state_update — Triton kernel for the selective scan state update
       step in Mamba-2's SSD (Structured State Space Duality) algorithm. Computes
       the recurrent state transition: h_t = A * h_t-1 + B * x_t, then y_t = C * h_t.
       Used during sequential (non-chunked) inference.

    2. mamba_chunk_scan_combined — Triton kernel implementing the chunked parallel
       scan for Mamba-2. Processes input sequences in fixed-size chunks using
       parallel prefix sums, achieving O(L) work with O(log L) span. This is the
       primary forward-pass kernel during training.

    3. mamba_split_conv1d_scan_combined — Fused Triton kernel that combines the
       short 1D causal convolution (width=4 in Granite) with the subsequent SSM
       scan in a single GPU kernel launch. Reduces memory traffic by avoiding
       materialization of the intermediate conv output tensor.

SCOPE:
    Tested with mamba_ssm 2.2.2 + causal-conv1d 1.4.0 on CUDA 12.x.
    The patch is idempotent — safe to call multiple times.

REFERENCES:
    - Mamba-2 paper: "Transformers are SSMs" (Dao & Gu, 2024)
    - Granite 4.0-H-Micro: ibm-granite/granite-4.0-h-micro on HuggingFace
    - mamba_ssm: https://github.com/state-spaces/mamba
"""

_applied = False


def apply():
    """
    Patch mamba_ssm top-level namespace with Triton kernel exports.

    Must be called before any transformers model loading.
    Idempotent — safe to call multiple times.
    """
    global _applied
    if _applied:
        return

    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    from mamba_ssm.ops.triton.ssd_combined import (
        mamba_chunk_scan_combined,
        mamba_split_conv1d_scan_combined,
    )
    import mamba_ssm

    mamba_ssm.selective_state_update = selective_state_update
    mamba_ssm.mamba_chunk_scan_combined = mamba_chunk_scan_combined
    mamba_ssm.mamba_split_conv1d_scan_combined = mamba_split_conv1d_scan_combined

    _applied = True


if __name__ == "__main__":
    apply()
    print("Monkey-patch applied successfully.")

    # Verify the exports are accessible
    import mamba_ssm
    for attr in [
        "selective_state_update",
        "mamba_chunk_scan_combined",
        "mamba_split_conv1d_scan_combined",
    ]:
        fn = getattr(mamba_ssm, attr, None)
        status = f"OK ({fn.__module__})" if fn else "MISSING"
        print(f"  mamba_ssm.{attr}: {status}")
