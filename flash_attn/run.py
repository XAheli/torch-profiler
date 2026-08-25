#!/mnt/podman_storage/ahpoddar/conda_envs/pt_nightly/bin/python
"""
FlashAttention-3 vs FlashAttention-2 profiling on NVIDIA H200.

Compares PyTorch's built-in SDPA flash backend (FlashAttention-2) against
FlashAttention-3 built from source for Hopper, across sequence lengths
512, 2048, 4096, 8192.

FA-3 exploits three Hopper-specific features:
  - WGMMA (Warp Group MMA): new Tensor Core instruction
  - TMA (Tensor Memory Accelerator): hardware async data movement
  - Warp specialization: producer/consumer warp overlap

Kernel sources:
  - FA-2: PyTorch vendored FlashAttention-2 → flash_fwd_kernel
  - FA-3: Built from https://github.com/Dao-AILab/flash-attention (hopper/ dir)
          Cloned to: /mnt/podman_storage/ahpoddar/flash-attention-src/
          Built with: cd hopper && python setup.py install
          Import: from flash_attn_interface import flash_attn_func
          Kernel name in traces: device_kernel (CUTLASS 3.x CuTeDSL template)

Outputs:
  - flash_attn/traces/fa2_seq{S}.json (Perfetto traces, 4 files)
  - flash_attn/traces/fa3_seq{S}.json (Perfetto traces, 4 files)

Environment: pt_nightly (Python 3.11, PyTorch 2.14.0.dev)
"""
import os
os.environ["PATH"] = "/usr/local/cuda-12.8/bin:" + os.environ.get("PATH", "")
os.environ["HF_HOME"] = "/mnt/podman_storage/ahpoddar/.cache/huggingface"

import argparse
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from flash_attn_interface import flash_attn_func as fa3_func


def print_gpu_info():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU : {props.name}")
    print(f"SM  : {props.major}.{props.minor}  |  SMs: {props.multi_processor_count}")
    print(f"VRAM: {props.total_memory / 1024**3:.1f} GB\n")


def make_qkv(B, H, S, D):
    # PyTorch SDPA expects [B, H, S, D] layout (heads before sequence)
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    return q, k, v


def cuda_event_time(fn, reps=20):
    # CUDA events measure GPU-side wall time; more accurate than profiler for latency
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(reps):
        fn()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / reps


def profile_fa2(q, k, v, warmup, trace_dir, seq_len):
    for _ in range(warmup):
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.cuda.synchronize()

    # schedule: skip 1 step (init noise), warmup 1 (let caches settle), record 3 active steps
    schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
    trace_path = os.path.join(trace_dir, f"fa2_seq{seq_len}.json")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=schedule,
        record_shapes=True,
        with_stack=False,
    ) as prof:
        for _ in range(5):
            with torch.profiler.record_function("fa2_sdpa"):
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    F.scaled_dot_product_attention(q, k, v, is_causal=True)
            prof.step()
        torch.cuda.synchronize()

    prof.export_chrome_trace(trace_path)

    cuda_time = sum(
        e.device_time_total for e in prof.key_averages() if e.key == "fa2_sdpa"
    )
    return cuda_time / 3.0


def profile_fa3(q, k, v, warmup, trace_dir, seq_len):
    # FA-3 uses [B, S, H, D] layout (not [B, H, S, D] like PyTorch SDPA)
    q3 = q.transpose(1, 2).contiguous()
    k3 = k.transpose(1, 2).contiguous()
    v3 = v.transpose(1, 2).contiguous()

    for _ in range(warmup):
        fa3_func(q3, k3, v3, causal=True)
    torch.cuda.synchronize()

    # schedule: skip 1 step (init noise), warmup 1 (let caches settle), record 3 active steps
    schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
    trace_path = os.path.join(trace_dir, f"fa3_seq{seq_len}.json")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=schedule,
        record_shapes=True,
        with_stack=False,
    ) as prof:
        for _ in range(5):
            with torch.profiler.record_function("fa3_hopper"):
                fa3_func(q3, k3, v3, causal=True)
            prof.step()
        torch.cuda.synchronize()

    prof.export_chrome_trace(trace_path)

    cuda_time = sum(
        e.device_time_total for e in prof.key_averages() if e.key == "fa3_hopper"
    )
    return cuda_time / 3.0


def main():
    parser = argparse.ArgumentParser(description="FlashAttention-2 vs FlashAttention-3 profiling")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trace-dir", type=str, default="flash_attn/traces")
    args = parser.parse_args()

    os.makedirs(args.trace_dir, exist_ok=True)
    seq_lengths = [512, 2048, 4096, 8192]

    print("=" * 70)
    print("FlashAttention-2 (SDPA) vs FlashAttention-3 (Hopper)")
    print("=" * 70)
    print_gpu_info()
    print(f"B={args.batch}, H={args.heads}, D={args.head_dim}\n")

    results = []

    for S in seq_lengths:
        print(f"  Profiling seq_len={S} ...", flush=True)
        q, k, v = make_qkv(args.batch, args.heads, S, args.head_dim)

        fa2_us = profile_fa2(q, k, v, args.warmup, args.trace_dir, S)
        fa3_us = profile_fa3(q, k, v, args.warmup, args.trace_dir, S)

        # transpose again for FA-3's [B, S, H, D] layout requirement
        q3 = q.transpose(1, 2).contiguous()
        k3 = k.transpose(1, 2).contiguous()
        v3 = v.transpose(1, 2).contiguous()

        fa2_event_ms = cuda_event_time(
            lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True)
        )
        fa3_event_ms = cuda_event_time(
            lambda: fa3_func(q3, k3, v3, causal=True)
        )

        results.append((S, fa2_us, fa3_us, fa2_event_ms, fa3_event_ms))
        del q, k, v, q3, k3, v3
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("Profiler CUDA time (averaged over active steps)")
    print("=" * 70)
    print(f"  {'Seq':<8} {'FA-2 (µs)':<14} {'FA-3 (µs)':<14} {'FA-3 speedup'}")
    print("  " + "-" * 50)
    for S, fa2_us, fa3_us, _, _ in results:
        speedup = fa2_us / fa3_us if fa3_us > 0 else float("inf")
        print(f"  {S:<8} {fa2_us:<14.1f} {fa3_us:<14.1f} {speedup:.2f}x")

    print("\n" + "=" * 70)
    print("CUDA Event timing (20 iterations, more accurate)")
    print("=" * 70)
    print(f"  {'Seq':<8} {'FA-2 (ms)':<14} {'FA-3 (ms)':<14} {'FA-3 speedup'}")
    print("  " + "-" * 50)
    for S, _, _, fa2_ms, fa3_ms in results:
        speedup = fa2_ms / fa3_ms if fa3_ms > 0 else float("inf")
        print(f"  {S:<8} {fa2_ms:<14.3f} {fa3_ms:<14.3f} {speedup:.2f}x")

    print(f"\nTraces saved to: {args.trace_dir}/")


if __name__ == "__main__":
    main()
