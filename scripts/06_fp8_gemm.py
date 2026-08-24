#!/mnt/podman_storage/ahpoddar/conda_envs/pt_nightly/bin/python
"""
06 — FP8 GEMM on Hopper: bf16 vs float8_e4m3fn Tensor Cores

Compares bf16 GEMM vs FP8 GEMM using torch._scaled_mm with dimensions
from a real Llama model (hidden_size -> intermediate_size).  FP8 Tensor
Cores on Hopper (SM90) deliver ~2x the FLOPS of bf16.

Shows kernel name differences between bf16 cuBLAS and FP8 paths.
"""

import argparse
import os

import torch

os.environ["PATH"] = "/usr/local/cuda-12.8/bin:" + os.environ.get("PATH", "")


def print_gpu_info():
    if not torch.cuda.is_available():
        print("CUDA not available — profiling will be CPU-only")
        return
    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    print(f"GPU : {props.name}")
    print(f"SM  : {props.major}.{props.minor}  |  SMs: {props.multi_processor_count}")
    print(f"VRAM: {props.total_memory / 1024**3:.1f} GB")
    if props.major < 9:
        print("  !!  FP8 Tensor Cores require Hopper (SM90) or newer")
    print()


def get_cuda_kernel_names(prof):
    names = []
    for evt in prof.key_averages():
        if evt.device_time_total > 0 and evt.key not in ("ProfilerStep*",):
            names.append(evt.key)
    return names


def profile_gemm(a, b, label, trace_dir, warmup, scaled_mm_args=None):
    """Profile either a torch.matmul or torch._scaled_mm call."""
    if scaled_mm_args:
        for _ in range(warmup):
            torch._scaled_mm(a, b, **scaled_mm_args)
        torch.cuda.synchronize()

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof:
            with torch.profiler.record_function(label):
                torch._scaled_mm(a, b, **scaled_mm_args)
                torch.cuda.synchronize()
    else:
        for _ in range(warmup):
            torch.matmul(a, b)
        torch.cuda.synchronize()

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof:
            with torch.profiler.record_function(label):
                torch.matmul(a, b)
                torch.cuda.synchronize()

    trace_path = os.path.join(trace_dir, f"06_{label}.json")
    prof.export_chrome_trace(trace_path)

    total_cuda_us = sum(e.device_time_total for e in prof.key_averages() if e.device_time_total > 0)
    kernel_names = get_cuda_kernel_names(prof)

    return total_cuda_us, kernel_names, prof


def main():
    parser = argparse.ArgumentParser(description="Profile bf16 vs FP8 GEMM")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trace-dir", type=str, default="traces")
    args = parser.parse_args()

    print_gpu_info()
    os.makedirs(args.trace_dir, exist_ok=True)

    device = "cuda"

    from transformers import AutoConfig
    try:
        config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    except Exception:
        fallback = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        print(f"Could not load {args.model}, falling back to {fallback}")
        config = AutoConfig.from_pretrained(fallback)
        args.model = fallback

    M = args.batch * args.seq
    K = config.hidden_size
    N = config.intermediate_size

    print(f"Model           : {args.model}")
    print(f"GEMM dimensions : [{M} x {K}] @ [{K} x {N}]")
    print(f"                  (simulates gate/up projection: hidden -> intermediate)\n")

    # --- bf16 ---
    a_bf16 = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    b_bf16 = torch.randn(K, N, device=device, dtype=torch.bfloat16)

    print(f"{'='*60}")
    print(f"  bf16 GEMM")
    print(f"{'='*60}")
    t_bf16, kn_bf16, prof_bf16 = profile_gemm(a_bf16, b_bf16, "gemm_bf16", args.trace_dir, args.warmup)
    print(prof_bf16.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # --- FP8 ---
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    a_fp8 = a_bf16.clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    b_fp8 = b_bf16.clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)

    scale_a = torch.ones(1, device=device, dtype=torch.float32)
    scale_b = torch.ones(1, device=device, dtype=torch.float32)

    print(f"\n{'='*60}")
    print(f"  FP8 (float8_e4m3fn) GEMM via torch._scaled_mm")
    print(f"{'='*60}")
    # _scaled_mm requires row-major A @ column-major B (cuBLASLt constraint)
    # A is [M, K] row-major (contiguous), B must be [K, N] column-major (= [N, K].t())
    b_fp8_col = torch.randn(N, K, device=device, dtype=torch.bfloat16).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn).t()
    t_fp8, kn_fp8, prof_fp8 = profile_gemm(
        a_fp8, b_fp8_col, "gemm_fp8", args.trace_dir, args.warmup,
        scaled_mm_args={"scale_a": scale_a, "scale_b": scale_b, "out_dtype": torch.bfloat16},
    )
    print(prof_fp8.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # --- Summary ---
    speedup = t_bf16 / t_fp8 if t_fp8 > 0 else float("inf")

    print(f"\n{'='*60}")
    print(f"  Comparison summary")
    print(f"{'='*60}")
    print(f"  {'Metric':<25} {'bf16':>20} {'FP8':>20}")
    print(f"  {'-'*65}")
    print(f"  {'CUDA time (us)':<25} {t_bf16:>20.0f} {t_fp8:>20.0f}")
    print(f"  {'Speedup':<25} {'1.00x':>20} {speedup:>19.2f}x")
    print(f"  {'Top kernel':<25} {(kn_bf16[0] if kn_bf16 else '-')[:20]:>20} {(kn_fp8[0] if kn_fp8 else '-')[:20]:>20}")

    print(f"\n  bf16 kernels: {', '.join(k[:50] for k in kn_bf16[:3])}")
    print(f"  FP8  kernels: {', '.join(k[:50] for k in kn_fp8[:3])}")
    print(f"\nView traces at -> https://ui.perfetto.dev/")
    print("Compare kernel names between 06_gemm_bf16.json and 06_gemm_fp8.json")


if __name__ == "__main__":
    main()
