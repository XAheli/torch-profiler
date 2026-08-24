#!/mnt/podman_storage/ahpoddar/conda_envs/pt_nightly/bin/python
"""
06 — FP8 GEMM on Hopper: bf16 vs PyTorch FP8 vs DeepGEMM

Compares three GEMM paths on Hopper SM90 Tensor Cores:
  1. Standard bf16 (torch.matmul) — baseline
  2. PyTorch FP8 (torch._scaled_mm) — cuBLASLt FP8
  3. DeepGEMM FP8 (deep_gemm.fp8_gemm_nt) — native JIT FP8

Dimensions come from a real Llama model (hidden_size -> intermediate_size).
"""

import argparse
import os
import sys

import torch

os.environ["PATH"] = "/usr/local/cuda-12.8/bin:" + os.environ.get("PATH", "")
os.environ["HF_HOME"] = "/mnt/podman_storage/ahpoddar/.cache/huggingface"

sys.path.insert(0, "/mnt/podman_storage/ahpoddar/DeepGEMM-src/tests")

try:
    import deep_gemm
    from generators import generate_normal, KernelType, MajorTypeAB

    HAS_DEEPGEMM = True
except ImportError:
    HAS_DEEPGEMM = False


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
    schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)

    if scaled_mm_args:
        for _ in range(warmup):
            torch._scaled_mm(a, b, **scaled_mm_args)
        torch.cuda.synchronize()

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=schedule,
            record_shapes=True,
        ) as prof:
            for _ in range(5):
                with torch.profiler.record_function(label):
                    torch._scaled_mm(a, b, **scaled_mm_args)
                prof.step()
            torch.cuda.synchronize()
    else:
        for _ in range(warmup):
            torch.matmul(a, b)
        torch.cuda.synchronize()

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=schedule,
            record_shapes=True,
        ) as prof:
            for _ in range(5):
                with torch.profiler.record_function(label):
                    torch.matmul(a, b)
                prof.step()
            torch.cuda.synchronize()

    trace_path = os.path.join(trace_dir, f"06_{label}.json")
    prof.export_chrome_trace(trace_path)

    total_cuda_us = sum(e.device_time_total for e in prof.key_averages() if e.device_time_total > 0)
    kernel_names = get_cuda_kernel_names(prof)

    return total_cuda_us, kernel_names, prof


def profile_deepgemm(a, b, d, label, trace_dir, warmup):
    """Profile deep_gemm.fp8_gemm_nt with the same schedule."""
    for _ in range(warmup):
        deep_gemm.fp8_gemm_nt(a, b, d)
    torch.cuda.synchronize()

    schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=schedule,
        record_shapes=True,
    ) as prof:
        for _ in range(5):
            with torch.profiler.record_function(label):
                deep_gemm.fp8_gemm_nt(a, b, d)
            prof.step()
        torch.cuda.synchronize()

    trace_path = os.path.join(trace_dir, f"06_{label}.json")
    prof.export_chrome_trace(trace_path)

    total_cuda_us = sum(e.device_time_total for e in prof.key_averages() if e.device_time_total > 0)
    kernel_names = get_cuda_kernel_names(prof)

    return total_cuda_us, kernel_names, prof


def main():
    parser = argparse.ArgumentParser(description="Profile bf16 vs PyTorch FP8 vs DeepGEMM FP8")
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

    # =====================================================================
    #  1. bf16 GEMM (baseline)
    # =====================================================================
    a_bf16 = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    b_bf16 = torch.randn(K, N, device=device, dtype=torch.bfloat16)

    print(f"{'='*70}")
    print(f"  [1/3]  bf16 GEMM  (torch.matmul)")
    print(f"{'='*70}")
    t_bf16, kn_bf16, prof_bf16 = profile_gemm(
        a_bf16, b_bf16, "gemm_bf16", args.trace_dir, args.warmup
    )
    print(prof_bf16.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # =====================================================================
    #  2. PyTorch FP8 via torch._scaled_mm
    # =====================================================================
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    a_fp8 = a_bf16.clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)

    scale_a = torch.ones(1, device=device, dtype=torch.float32)
    scale_b = torch.ones(1, device=device, dtype=torch.float32)

    # _scaled_mm requires row-major A @ column-major B (cuBLASLt constraint)
    b_fp8_col = (
        torch.randn(N, K, device=device, dtype=torch.bfloat16)
        .clamp(-fp8_max, fp8_max)
        .to(torch.float8_e4m3fn)
        .t()
    )

    print(f"\n{'='*70}")
    print(f"  [2/3]  PyTorch FP8  (torch._scaled_mm)")
    print(f"{'='*70}")
    t_fp8, kn_fp8, prof_fp8 = profile_gemm(
        a_fp8,
        b_fp8_col,
        "gemm_fp8_pytorch",
        args.trace_dir,
        args.warmup,
        scaled_mm_args={"scale_a": scale_a, "scale_b": scale_b, "out_dtype": torch.bfloat16},
    )
    print(prof_fp8.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # =====================================================================
    #  3. DeepGEMM FP8 via deep_gemm.fp8_gemm_nt
    # =====================================================================
    t_dg = None
    kn_dg = []
    if HAS_DEEPGEMM:
        print(f"\n{'='*70}")
        print(f"  [3/3]  DeepGEMM FP8  (deep_gemm.fp8_gemm_nt)")
        print(f"{'='*70}")

        a_dg, b_dg, c_dg, d_dg, ref_d_dg = generate_normal(
            M, N, K,
            MajorTypeAB.KMajor, MajorTypeAB.KMajor,
            accumulate=False,
            out_dtype=torch.bfloat16,
            kernel_type=KernelType.Kernel1D1D,
        )

        t_dg, kn_dg, prof_dg = profile_deepgemm(
            a_dg, b_dg, d_dg, "gemm_fp8_deepgemm", args.trace_dir, args.warmup
        )
        print(prof_dg.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    else:
        print(f"\n{'='*70}")
        print(f"  [3/3]  DeepGEMM FP8  — SKIPPED (import failed)")
        print(f"{'='*70}")
        print("  Install DeepGEMM and ensure tests/generators.py is on sys.path")

    # =====================================================================
    #  Summary
    # =====================================================================
    speedup_fp8 = t_bf16 / t_fp8 if t_fp8 > 0 else float("inf")
    speedup_dg = t_bf16 / t_dg if t_dg and t_dg > 0 else None

    col_w = 22
    print(f"\n{'='*70}")
    print(f"  Comparison summary — bf16 vs PyTorch FP8 vs DeepGEMM FP8")
    print(f"{'='*70}")

    hdr = f"  {'Metric':<25} {'bf16':>{col_w}} {'PyTorch FP8':>{col_w}}"
    if t_dg is not None:
        hdr += f" {'DeepGEMM FP8':>{col_w}}"
    print(hdr)
    print(f"  {'-'*25} {'-'*col_w} {'-'*col_w}" + (f" {'-'*col_w}" if t_dg is not None else ""))

    row_cuda = f"  {'CUDA time (us)':<25} {t_bf16:>{col_w}.0f} {t_fp8:>{col_w}.0f}"
    if t_dg is not None:
        row_cuda += f" {t_dg:>{col_w}.0f}"
    print(row_cuda)

    row_speed = f"  {'Speedup vs bf16':<25} {'1.00x':>{col_w}} {speedup_fp8:>{col_w - 1}.2f}x"
    if speedup_dg is not None:
        row_speed += f" {speedup_dg:>{col_w - 1}.2f}x"
    else:
        row_speed += f" {'N/A':>{col_w}}" if t_dg is None else ""
    print(row_speed)

    def top_k(names, k=1):
        return (names[0][:col_w] if names else "-")

    row_kern = f"  {'Top kernel':<25} {top_k(kn_bf16):>{col_w}} {top_k(kn_fp8):>{col_w}}"
    if t_dg is not None:
        row_kern += f" {top_k(kn_dg):>{col_w}}"
    print(row_kern)

    print(f"\n  bf16        kernels: {', '.join(k[:50] for k in kn_bf16[:3])}")
    print(f"  PyTorch FP8 kernels: {', '.join(k[:50] for k in kn_fp8[:3])}")
    if kn_dg:
        print(f"  DeepGEMM    kernels: {', '.join(k[:50] for k in kn_dg[:3])}")

    print(f"\nView traces at -> https://ui.perfetto.dev/")
    print("Compare 06_gemm_bf16.json, 06_gemm_fp8_pytorch.json, 06_gemm_fp8_deepgemm.json")


if __name__ == "__main__":
    main()
