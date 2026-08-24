#!/mnt/podman_storage/ahpoddar/conda_envs/pt_nightly/bin/python
"""
04 — Liger kernels on Llama: fused Triton replacements

Applies Liger fused Triton kernels (RMSNorm, SwiGLU, RoPE) to a
LlamaDecoderLayer and profiles against vanilla eager.  Also compares
FusedLinearCrossEntropyLoss vs standard Linear+CE.

Liger does NOT touch GEMMs — those stay as cuBLAS.
"""

import argparse
import os
import sys

import torch

os.environ["PATH"] = "/usr/local/cuda-12.8/bin:" + os.environ.get("PATH", "")
os.environ["HF_HOME"] = "/mnt/podman_storage/ahpoddar/.cache/huggingface"


def print_gpu_info():
    if not torch.cuda.is_available():
        print("CUDA not available — profiling will be CPU-only")
        return
    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    print(f"GPU : {props.name}")
    print(f"SM  : {props.major}.{props.minor}  |  SMs: {props.multi_processor_count}")
    print(f"VRAM: {props.total_memory / 1024**3:.1f} GB\n")


def load_config(model_id):
    from transformers import AutoConfig

    try:
        return AutoConfig.from_pretrained(model_id, trust_remote_code=True), model_id
    except Exception:
        fallback = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        print(f"Could not load config for {model_id}, falling back to {fallback}")
        return AutoConfig.from_pretrained(fallback), fallback


def profile_layer(layer, hidden, position_embeddings, label, trace_dir):
    torch.cuda.reset_peak_memory_stats()

    schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(5):
            with torch.no_grad():
                with torch.profiler.record_function(label):
                    layer(hidden, position_embeddings=position_embeddings)
            prof.step()
        torch.cuda.synchronize()

    mem_peak = torch.cuda.max_memory_allocated()
    trace_path = os.path.join(trace_dir, f"04_{label}.json")
    prof.export_chrome_trace(trace_path)

    total_cuda_us = sum(
        e.device_time_total for e in prof.key_averages() if e.device_time_total > 0
    )
    n_kernels = len([e for e in prof.key_averages() if e.device_time_total > 0])

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    print(f"  Total CUDA time : {total_cuda_us:.0f} us")
    print(f"  Kernel launches : {n_kernels}")
    print(f"  Peak memory     : {mem_peak / 1024**2:.1f} MB")
    print(f"  Trace saved     -> {trace_path}")

    return total_cuda_us, n_kernels, mem_peak


def profile_fused_ce(hidden_size, vocab_size, batch, seq, warmup, trace_dir):
    """Compare standard Linear+CE vs Liger FusedLinearCrossEntropyLoss."""
    try:
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    except ImportError:
        print("\n  [!] LigerFusedLinearCrossEntropyLoss not available, skipping CE comparison")
        return

    device = "cuda"
    dtype = torch.bfloat16

    linear = torch.nn.Linear(hidden_size, vocab_size, bias=False, device=device, dtype=dtype)
    ce_loss = torch.nn.CrossEntropyLoss()
    fused_ce = LigerFusedLinearCrossEntropyLoss()

    hidden_input = torch.randn(batch * seq, hidden_size, device=device, dtype=dtype)
    targets = torch.randint(0, vocab_size, (batch * seq,), device=device)

    for _ in range(warmup):
        logits = linear(hidden_input)
        ce_loss(logits.float(), targets)
    torch.cuda.synchronize()

    print(f"\n{'='*60}")
    print(f"  FusedLinearCrossEntropy comparison  (vocab={vocab_size})")
    print(f"{'='*60}")

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof_std:
        with torch.profiler.record_function("standard_linear_ce"):
            logits = linear(hidden_input)
            loss = ce_loss(logits.float(), targets)
            loss.backward()
            torch.cuda.synchronize()

    prof_std.export_chrome_trace(os.path.join(trace_dir, "04_standard_ce.json"))
    t_std = sum(e.device_time_total for e in prof_std.key_averages() if e.device_time_total > 0)

    linear.zero_grad()

    weight_copy = linear.weight.detach().clone().requires_grad_(True)
    hidden_copy = hidden_input.detach().clone().requires_grad_(True)

    try:
        for _ in range(warmup):
            fused_ce(hidden_copy, weight_copy, targets)
        torch.cuda.synchronize()

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof_fused:
            with torch.profiler.record_function("fused_linear_ce"):
                loss_fused = fused_ce(hidden_copy, weight_copy, targets)
                loss_fused.backward()
                torch.cuda.synchronize()

        prof_fused.export_chrome_trace(os.path.join(trace_dir, "04_fused_ce.json"))
        t_fused = sum(e.device_time_total for e in prof_fused.key_averages() if e.device_time_total > 0)

        speedup_ce = t_std / t_fused if t_fused > 0 else float("inf")
        print(f"\n  Standard Linear+CE : {t_std:.0f} us")
        print(f"  Fused Linear CE    : {t_fused:.0f} us")
        print(f"  Speedup            : {speedup_ce:.2f}x")
    except RuntimeError as e:
        print(f"\n  FusedLinearCE failed (PyTorch nightly ABI issue): {e}")
        print(f"  Standard Linear+CE : {t_std:.0f} us")
        print(f"  The Liger RMSNorm/SwiGLU comparison above is the main demo.")


def main():
    parser = argparse.ArgumentParser(description="Profile Liger kernels on LlamaDecoderLayer")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trace-dir", type=str, default="traces")
    args = parser.parse_args()

    print_gpu_info()
    os.makedirs(args.trace_dir, exist_ok=True)

    device = "cuda"
    dtype = torch.bfloat16

    config, model_id = load_config(args.model)
    short_name = model_id.split("/")[-1]
    print(f"Model  : {model_id}")
    print(f"Shape  : batch={args.batch}, seq={args.seq}, hidden={config.hidden_size}")

    hidden = torch.randn(args.batch, args.seq, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(args.seq, device=device).unsqueeze(0).expand(args.batch, -1)

    # --- Vanilla ---
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding

    layer_vanilla = LlamaDecoderLayer(config, layer_idx=0).to(device, dtype=dtype).eval()
    rotary_emb = LlamaRotaryEmbedding(config=config).to(device, dtype=dtype)
    position_embeddings = rotary_emb(hidden, position_ids)

    with torch.no_grad():
        for _ in range(args.warmup):
            layer_vanilla(hidden, position_embeddings=position_embeddings)
        torch.cuda.synchronize()

    t_vanilla, k_vanilla, m_vanilla = profile_layer(
        layer_vanilla, hidden, position_embeddings, f"vanilla_{short_name}", args.trace_dir
    )

    # --- Liger ---
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_llama
    except ImportError:
        print("\n[!] liger-kernel not installed. Run: pip install liger-kernel")
        sys.exit(0)

    apply_liger_kernel_to_llama()

    layer_liger = LlamaDecoderLayer(config, layer_idx=0).to(device, dtype=dtype).eval()
    with torch.no_grad():
        for _ in range(args.warmup):
            layer_liger(hidden, position_embeddings=position_embeddings)
        torch.cuda.synchronize()

    t_liger, k_liger, m_liger = profile_layer(
        layer_liger, hidden, position_embeddings, f"liger_{short_name}", args.trace_dir
    )

    # --- Side-by-side ---
    speedup = t_vanilla / t_liger if t_liger > 0 else float("inf")
    mem_saved = (m_vanilla - m_liger) / 1024**2

    print(f"\n{'='*60}")
    print(f"  Side-by-side: Vanilla vs Liger decoder layer")
    print(f"{'='*60}")
    print(f"  {'Metric':<25} {'Vanilla':>15} {'Liger':>15}")
    print(f"  {'-'*55}")
    print(f"  {'CUDA time (us)':<25} {t_vanilla:>15.0f} {t_liger:>15.0f}")
    print(f"  {'Kernel launches':<25} {k_vanilla:>15} {k_liger:>15}")
    print(f"  {'Peak memory (MB)':<25} {m_vanilla/1024**2:>15.1f} {m_liger/1024**2:>15.1f}")
    print(f"  {'Speedup':<25} {'':>15} {speedup:>14.2f}x")
    print(f"  {'Memory saved (MB)':<25} {'':>15} {mem_saved:>14.1f}")

    # --- FusedLinearCrossEntropy ---
    profile_fused_ce(
        config.hidden_size, args.vocab_size, args.batch, args.seq,
        args.warmup, args.trace_dir,
    )

    print(f"\nView traces at -> https://ui.perfetto.dev/")


if __name__ == "__main__":
    main()
