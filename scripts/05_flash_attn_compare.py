#!/mnt/podman_storage/ahpoddar/conda_envs/pt_nightly/bin/python
"""
05 — FlashAttention Shootout: all attention approaches on Llama-shaped Q/K/V

Compares four attention implementations on identical inputs:
  1. SDPA math backend   — unfused Q@K, softmax, @V chain (~20 kernels)
  2. SDPA flash backend  — FlashAttention-2, single fused kernel
  3. SDPA cudnn backend  — cuDNN Hopper-optimized (WGMMA instructions)
  4. FlashAttention-3    — Hopper-exclusive (WGMMA + TMA, async softmax)

Prints a side-by-side comparison table: CUDA time, kernel count, kernel names.
"""

import argparse
import os

import torch
import torch.nn.functional as F

os.environ["PATH"] = "/usr/local/cuda-12.8/bin:" + os.environ.get("PATH", "")


SDPA_BACKENDS = {
    "math": torch.nn.attention.SDPBackend.MATH,
    "flash": torch.nn.attention.SDPBackend.FLASH_ATTENTION,
    "cudnn": torch.nn.attention.SDPBackend.CUDNN_ATTENTION,
}

try:
    from flash_attn_interface import flash_attn_func as fa3_func
    HAS_FA3 = True
except ImportError:
    HAS_FA3 = False


def print_gpu_info():
    if not torch.cuda.is_available():
        print("CUDA not available — profiling will be CPU-only")
        return
    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    print(f"GPU : {props.name}")
    print(f"SM  : {props.major}.{props.minor}  |  SMs: {props.multi_processor_count}")
    print(f"VRAM: {props.total_memory / 1024**3:.1f} GB\n")


def build_qkv(config, batch, seq, device, dtype):
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_heads
    q = torch.randn(batch, num_heads, seq, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, num_heads, seq, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, num_heads, seq, head_dim, device=device, dtype=dtype)
    return q, k, v, num_heads, head_dim


def get_cuda_kernel_names(prof):
    """Extract CUDA kernel names from profiler events."""
    names = []
    for evt in prof.key_averages():
        if evt.device_time_total > 0 and evt.key not in ("ProfilerStep*",):
            names.append(evt.key)
    return names


def profile_sdpa_backend(name, backend_enum, q, k, v, warmup, trace_dir, tag):
    try:
        with torch.nn.attention.sdpa_kernel(backend_enum):
            for _ in range(warmup):
                F.scaled_dot_product_attention(q, k, v, is_causal=True)
            torch.cuda.synchronize()
    except RuntimeError as e:
        print(f"  [{name}] Not available: {e}")
        return None

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        with torch.no_grad():
            with torch.profiler.record_function(f"sdpa_{name}"):
                with torch.nn.attention.sdpa_kernel(backend_enum):
                    F.scaled_dot_product_attention(q, k, v, is_causal=True)
                torch.cuda.synchronize()

    trace_path = os.path.join(trace_dir, f"05_attn_{name}_{tag}.json")
    prof.export_chrome_trace(trace_path)

    total_cuda_us = sum(e.device_time_total for e in prof.key_averages() if e.device_time_total > 0)
    n_kernels = len([e for e in prof.key_averages() if e.device_time_total > 0])
    kernel_names = get_cuda_kernel_names(prof)

    print(f"\n  [{name}]")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    return {
        "name": f"SDPA {name}",
        "cuda_us": total_cuda_us,
        "kernels": n_kernels,
        "top_kernel": kernel_names[0] if kernel_names else "—",
        "trace": trace_path,
    }


def profile_fa3(q, k, v, warmup, trace_dir, tag):
    if not HAS_FA3:
        print("  [FA-3] Not available (build from flash-attention/hopper/)")
        return None

    q_fa3 = q.transpose(1, 2).contiguous()
    k_fa3 = k.transpose(1, 2).contiguous()
    v_fa3 = v.transpose(1, 2).contiguous()

    for _ in range(warmup):
        fa3_func(q_fa3, k_fa3, v_fa3, causal=True)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        with torch.profiler.record_function("flash_attn_v3"):
            with torch.no_grad():
                fa3_func(q_fa3, k_fa3, v_fa3, causal=True)
            torch.cuda.synchronize()

    trace_path = os.path.join(trace_dir, f"05_attn_fa3_{tag}.json")
    prof.export_chrome_trace(trace_path)

    total_cuda_us = sum(e.device_time_total for e in prof.key_averages() if e.device_time_total > 0)
    n_kernels = len([e for e in prof.key_averages() if e.device_time_total > 0])
    kernel_names = get_cuda_kernel_names(prof)

    print(f"\n  [FA-3]")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    return {
        "name": "FlashAttn-3",
        "cuda_us": total_cuda_us,
        "kernels": n_kernels,
        "top_kernel": kernel_names[0] if kernel_names else "—",
        "trace": trace_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare all attention implementations")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trace-dir", type=str, default="traces")
    args = parser.parse_args()

    print_gpu_info()
    os.makedirs(args.trace_dir, exist_ok=True)

    device = "cuda"
    dtype = torch.bfloat16

    from transformers import AutoConfig
    try:
        config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    except Exception:
        fallback = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        print(f"Could not load {args.model}, falling back to {fallback}")
        config = AutoConfig.from_pretrained(fallback)
        args.model = fallback

    short_name = args.model.split("/")[-1]
    q, k, v, num_heads, head_dim = build_qkv(config, args.batch, args.seq, device, dtype)

    print(f"Model     : {args.model}")
    print(f"Attention : {num_heads} heads x {head_dim} dim")
    print(f"Shape     : batch={args.batch}, seq={args.seq}")
    print(f"FA-3      : {'available' if HAS_FA3 else 'not installed'}")

    results = []

    for name, backend in SDPA_BACKENDS.items():
        res = profile_sdpa_backend(name, backend, q, k, v, args.warmup, args.trace_dir, short_name)
        if res:
            results.append(res)

    fa3_res = profile_fa3(q, k, v, args.warmup, args.trace_dir, short_name)
    if fa3_res:
        results.append(fa3_res)

    if not results:
        print("\nNo backends succeeded.")
        return

    fastest = min(results, key=lambda r: r["cuda_us"])

    print(f"\n{'='*80}")
    print(f"  Attention Shootout — Side-by-side comparison")
    print(f"{'='*80}")
    print(f"  {'Backend':<15} {'CUDA (us)':>12} {'Kernels':>10} {'vs fastest':>12}  Top kernel")
    print(f"  {'-'*75}")
    for r in sorted(results, key=lambda r: r["cuda_us"]):
        ratio = r["cuda_us"] / fastest["cuda_us"]
        marker = " <-" if r["name"] == fastest["name"] else ""
        top = r["top_kernel"][:40]
        print(f"  {r['name']:<15} {r['cuda_us']:>12.0f} {r['kernels']:>10} {ratio:>11.2f}x  {top}{marker}")

    print(f"\nView traces at -> https://ui.perfetto.dev/")


if __name__ == "__main__":
    main()
