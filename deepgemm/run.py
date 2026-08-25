#!/mnt/podman_storage/ahpoddar/conda_envs/pt_nightly/bin/python
"""
DeepGEMM FP8 vs bf16 GEMM profiling on NVIDIA H200.

Compares standard bf16 matrix multiplication against DeepGEMM's JIT-compiled
FP8 GEMM using Llama-3.1-8B projection dimensions (M=2048, K=4096, N=14336).

Kernel sources:
  - bf16: cuBLAS via torch.matmul → nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT
  - FP8:  DeepGEMM JIT → deep_gemm::sm90_fp8_gemm_1d2d_impl
          Source: https://github.com/deepseek-ai/DeepGEMM (main branch)
          Cloned to: /mnt/podman_storage/ahpoddar/DeepGEMM-src/
          Built with: python setup.py build_ext --inplace (C++20, DG_FORCE_BUILD=1)
          Test helpers: DeepGEMM-src/tests/generators.py (generate_normal)

Outputs:
  - deepgemm/traces/gemm_bf16.json         (Perfetto trace)
  - deepgemm/traces/gemm_fp8_deepgemm.json (Perfetto trace)

Environment: pt_nightly (Python 3.11, PyTorch 2.14.0.dev)
"""
import os
os.environ["PATH"] = "/usr/local/cuda-12.8/bin:" + os.environ.get("PATH", "")
os.environ["HF_HOME"] = "/mnt/podman_storage/ahpoddar/.cache/huggingface"

import sys
sys.path.insert(0, "/mnt/podman_storage/ahpoddar/DeepGEMM-src/tests")

import argparse
import torch
import deep_gemm
from generators import generate_normal, KernelType, MajorTypeAB


def print_gpu_info():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU : {props.name}")
    print(f"SM  : {props.major}.{props.minor}  |  SMs: {props.multi_processor_count}")
    print(f"VRAM: {props.total_memory / 1024**3:.1f} GB\n")


def profile_bf16(M, K, N, warmup, trace_dir):
    # Llama-3.1-8B MLP projection: [M=batch_tokens, K=hidden] @ [K=hidden, N=ffn_dim]
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)

    for _ in range(warmup):
        torch.matmul(a, b)
    torch.cuda.synchronize()

    # schedule: skip 1 step (init noise), warmup 1 (let caches settle), record 3 active steps
    schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
    trace_path = os.path.join(trace_dir, "gemm_bf16.json")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=schedule,
        record_shapes=True,
        with_stack=False,
        on_trace_ready=lambda p: p.export_chrome_trace(trace_path),
    ) as prof:
        for _ in range(5):
            with torch.profiler.record_function("bf16_matmul"):
                torch.matmul(a, b)
            prof.step()
        torch.cuda.synchronize()

    cuda_time = sum(
        e.device_time_total for e in prof.key_averages() if e.key == "bf16_matmul"
    )
    kernels = [
        e for e in prof.key_averages() if e.device_time_total > 0
    ]
    top_kernel = max(kernels, key=lambda e: e.device_time_total).key if kernels else "N/A"
    # divide by 3.0 because schedule records 3 active steps
    return cuda_time / 3.0, top_kernel


def profile_fp8_deepgemm(M, K, N, warmup, trace_dir):
    # generate_normal handles FP8 casting + TMA-aligned scale factor layout
    a, b, c, d, ref_d = generate_normal(
        # DeepGEMM expects (M, N, K) order, not (M, K, N)
        M, N, K,
        # KMajor = column-major tile layout, required for SM90 TMA descriptor alignment
        MajorTypeAB.KMajor, MajorTypeAB.KMajor,
        accumulate=False, out_dtype=torch.bfloat16,
        # Kernel1D1D: 1D block-scaling for both A and B (per-tile FP8 scale factors)
        kernel_type=KernelType.Kernel1D1D,
    )

    # warmup triggers JIT compilation of the FP8 kernel (first call compiles CUDA C++)
    for _ in range(warmup):
        deep_gemm.fp8_gemm_nt(a, b, d)
    torch.cuda.synchronize()

    # schedule: skip 1 step (init noise), warmup 1 (let caches settle), record 3 active steps
    schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
    trace_path = os.path.join(trace_dir, "gemm_fp8_deepgemm.json")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=schedule,
        record_shapes=True,
        with_stack=False,
        on_trace_ready=lambda p: p.export_chrome_trace(trace_path),
    ) as prof:
        for _ in range(5):
            with torch.profiler.record_function("fp8_deepgemm"):
                deep_gemm.fp8_gemm_nt(a, b, d)
            prof.step()
        torch.cuda.synchronize()

    cuda_time = sum(
        e.device_time_total for e in prof.key_averages() if e.key == "fp8_deepgemm"
    )
    kernels = [
        e for e in prof.key_averages() if e.device_time_total > 0
    ]
    top_kernel = max(kernels, key=lambda e: e.device_time_total).key if kernels else "N/A"
    # divide by 3.0 because schedule records 3 active steps
    return cuda_time / 3.0, top_kernel


def main():
    parser = argparse.ArgumentParser(description="bf16 vs DeepGEMM FP8 GEMM profiling")
    parser.add_argument("--M", type=int, default=2048)
    parser.add_argument("--K", type=int, default=4096)
    parser.add_argument("--N", type=int, default=14336)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trace-dir", type=str, default="deepgemm/traces")
    args = parser.parse_args()

    os.makedirs(args.trace_dir, exist_ok=True)

    print("=" * 70)
    print("bf16 GEMM vs DeepGEMM FP8 — Llama-3.1-8B dimensions")
    print("=" * 70)
    print_gpu_info()
    print(f"Shape: M={args.M}, K={args.K}, N={args.N}\n")

    bf16_us, bf16_kernel = profile_bf16(args.M, args.K, args.N, args.warmup, args.trace_dir)
    fp8_us, fp8_kernel = profile_fp8_deepgemm(args.M, args.K, args.N, args.warmup, args.trace_dir)

    speedup = bf16_us / fp8_us if fp8_us > 0 else float("inf")

    print(f"{'Method':<20} {'CUDA time (µs)':<18} {'Top kernel'}")
    print("-" * 70)
    print(f"{'bf16 matmul':<20} {bf16_us:<18.1f} {bf16_kernel}")
    print(f"{'DeepGEMM FP8':<20} {fp8_us:<18.1f} {fp8_kernel}")
    print("-" * 70)
    print(f"FP8 speedup: {speedup:.2f}x\n")
    print(f"Traces saved to: {args.trace_dir}/")


if __name__ == "__main__":
    main()
