#!/mnt/podman_storage/ahpoddar/micromamba_root/envs/py312_sonic/bin/python
"""
SonicMoE MoE forward pass profiling on NVIDIA H200.

Profiles Tri Dao's SonicMoE — a hardware-efficient MoE implementation that
overlaps IO with computation using CuTeDSL/CUTLASS 3.x on Hopper GPUs.
Configuration matches SonicMoE paper's 7B MoE (d=1536, n=256, E=128, K=8).

SonicMoE's key innovations:
  - IO-aware kernel design: overlaps GMEM loads with Tensor Core MMA
  - Ping-pong warpgroup scheduling for epilogue overlap
  - Gather fusion: token gather fused with GEMM prologue (no separate kernel)
  - 45% less activation memory than ScatterMoE

Kernel source:
  - pip install sonic-moe (from https://github.com/Dao-AILab/sonic-moe)
  - Underlying GEMM: quack-kernels (CUTLASS 3.x based)
  - Kernel names in traces:
    - kernel_cutlass_kernel_quackgemm_actGemmGatedSm90 (up projection + SwiGLU)
    - kernel_cutlass_kernel_quackgemm_default_epiGemmDefaultSm90 (down projection)

Outputs:
  - sonicmoe/traces/sonicmoe.json (Perfetto trace)

Environment: py312_sonic (Python 3.12, PyTorch 2.9.1, sonic-moe 0.1.2.post1)
"""
import os
os.environ["PATH"] = "/usr/local/cuda-12.8/bin:" + os.environ.get("PATH", "")
os.environ["HF_HOME"] = "/mnt/podman_storage/ahpoddar/.cache/huggingface"

import argparse
import torch
from sonicmoe.moe import MoE
from sonicmoe.enums import ActivationType


def print_gpu_info():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU : {props.name}")
    print(f"SM  : {props.major}.{props.minor}  |  SMs: {props.multi_processor_count}")
    print(f"VRAM: {props.total_memory / 1024**3:.1f} GB\n")


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


def main():
    parser = argparse.ArgumentParser(description="SonicMoE profiling")
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=1536)
    parser.add_argument("--intermediate", type=int, default=256)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trace-dir", type=str, default="sonicmoe/traces")
    args = parser.parse_args()

    os.makedirs(args.trace_dir, exist_ok=True)

    print("=" * 70)
    print("SonicMoE — MoE Forward Pass Profiling")
    print("=" * 70)
    print_gpu_info()
    print(f"Tokens={args.tokens}, Hidden={args.hidden}, Intermediate={args.intermediate}")
    print(f"Experts={args.experts}, TopK={args.topk}\n")

    # SonicMoE paper 7B config: 128 experts, top-8, SwiGLU gating
    moe = MoE(
        num_experts=args.experts,
        num_experts_per_tok=args.topk,
        hidden_size=args.hidden,
        # intermediate_size is per-expert FFN hidden dim (not total), kept small (256) with many experts
        intermediate_size=args.intermediate,
        activation_function=ActivationType.SWIGLU,
        add_bias=False,
        std=0.02,
    ).to("cuda", dtype=torch.bfloat16)

    # input is [tokens, hidden] — no batch dim; MoE routes individual tokens to experts
    x = torch.randn(args.tokens, args.hidden, device="cuda", dtype=torch.bfloat16)

    for _ in range(args.warmup):
        # SonicMoE returns (output, routing_metadata) tuple; [0] extracts the output tensor
        _ = moe(x)[0]
    torch.cuda.synchronize()

    # schedule: skip 1 step (init noise), warmup 1 (let caches settle), record 3 active steps
    schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
    trace_path = os.path.join(args.trace_dir, "sonicmoe.json")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=schedule,
        record_shapes=True,
        with_stack=True,
        on_trace_ready=lambda p: p.export_chrome_trace(trace_path),
    ) as prof:
        for _ in range(5):
            with torch.profiler.record_function("sonicmoe_forward"):
                out = moe(x)[0]
            prof.step()
        torch.cuda.synchronize()

    cuda_time_us = sum(
        e.device_time_total for e in prof.key_averages() if e.key == "sonicmoe_forward"
    )
    # divide by 3.0 because schedule records 3 active steps
    avg_cuda_us = cuda_time_us / 3.0

    event_ms = cuda_event_time(lambda: moe(x)[0])

    # FLOPs: up_proj (2*T*2n*d) + down_proj (2*T*n*d) per expert, K experts activated
    # factor of 2 for SwiGLU (gate + up are separate matmuls), outer *2 for multiply-add
    flops = 2 * args.tokens * 2 * args.intermediate * args.hidden * args.topk * 2
    tflops_profiler = flops / (avg_cuda_us * 1e-6) / 1e12
    tflops_event = flops / (event_ms * 1e-3) / 1e12

    print("=" * 70)
    print("Results")
    print("=" * 70)
    print(f"  Profiler CUDA time (avg active): {avg_cuda_us:.1f} µs")
    print(f"  CUDA Event time (20 iters avg):  {event_ms:.3f} ms")
    print(f"  TFLOPS (profiler):               {tflops_profiler:.2f}")
    print(f"  TFLOPS (event):                  {tflops_event:.2f}")
    print()

    print("Top CUDA kernels:")
    print(f"  {'Kernel':<60} {'CUDA time (µs)':<14} {'Calls'}")
    print("  " + "-" * 80)
    kernel_events = sorted(
        [e for e in prof.key_averages() if e.device_time_total > 0],
        key=lambda e: -e.device_time_total,
    )
    for e in kernel_events[:10]:
        print(f"  {e.key[:58]:<60} {e.device_time_total:<14.1f} {e.count}")

    print(f"\nTrace saved to: {trace_path}")


if __name__ == "__main__":
    main()
