#!/mnt/podman_storage/ahpoddar/conda_envs/pt_nightly/bin/python
"""
01 — Warmup: matmul + add (2-minute on-ramp)

  Small matrix (64x64): overhead-bound — kernel launch cost dominates.
  Large matrix (4096x4096): compute-bound — actual GEMM work dominates.
  With --compile, both calls fuse into a single Triton kernel (matmul + add).
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
    print(f"VRAM: {props.total_memory / 1024**3:.1f} GB\n")


def matmul_add(a, b, bias):
    return torch.matmul(a, b) + bias


def main():
    parser = argparse.ArgumentParser(description="Warmup: matmul + add profiling")
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trace-dir", type=str, default="traces")
    args = parser.parse_args()

    print_gpu_info()
    os.makedirs(args.trace_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    n = args.size
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    bias = torch.randn(n, device=device, dtype=dtype)

    fn = torch.compile(matmul_add) if args.compile else matmul_add
    tag = "compiled" if args.compile else "eager"

    for _ in range(args.warmup):
        fn(a, b, bias)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        with torch.profiler.record_function(f"matmul_add_{n}x{n}_{tag}"):
            fn(a, b, bias)
            torch.cuda.synchronize()

    trace_path = os.path.join(args.trace_dir, f"01_warmup_{n}_{tag}.json")
    prof.export_chrome_trace(trace_path)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    print(f"\nTrace saved → {trace_path}")
    print(f"View it at  → https://ui.perfetto.dev/  (drag-and-drop the JSON file)")


if __name__ == "__main__":
    main()
