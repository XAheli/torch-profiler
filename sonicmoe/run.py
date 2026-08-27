#!/mnt/podman_storage/ahpoddar/micromamba_root/envs/py312_sonic/bin/python
"""
SonicMoE vs ScatterMoE — MoE forward pass profiling on NVIDIA H200.

Profiles two MoE kernel implementations back-to-back on identical configs:

  ScatterMoE (baseline):
    - Triton-based MoE with gather/scatter fusion (Shawn Tan)
    - Source: https://github.com/shawntan/scattermoe
    - Kernel names in traces: triton_*  (Triton JIT-compiled)
    - NOTE: ScatterMoE is a GEMM-only kernel — it expects pre-computed
      routing weights/indices from an external gate, so we provide our own
      nn.Linear gate + softmax + topk outside the timed region.

  SonicMoE:
    - CuTeDSL/CUTLASS 3.x IO-aware MoE for Hopper GPUs (Tri Dao lab)
    - Source: https://github.com/Dao-AILab/sonic-moe (pip install sonic-moe)
    - Kernel names in traces:
        kernel_cutlass_kernel_quackgemm_actGemmGatedSm90  (up + SwiGLU)
        kernel_cutlass_kernel_quackgemm_default_epiGemmDefaultSm90  (down)
    - NOTE: SonicMoE is a complete MoE module — routing is handled
      internally, and forward() returns a (output, routing_metadata) tuple.

Configuration: 7B-class MoE (d=1536, n=256, E=128, K=8) from SonicMoE paper.

Environment: py312_sonic (Python 3.12, PyTorch 2.9.1, sonic-moe 0.1.2.post1)
"""
import os
os.environ["PATH"] = "/usr/local/cuda-12.8/bin:" + os.environ.get("PATH", "")
os.environ["HF_HOME"] = "/mnt/podman_storage/ahpoddar/.cache/huggingface"

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from scattermoe.mlp import MLP as ScatterMLP
from sonicmoe.moe import MoE
from sonicmoe.enums import ActivationType


def print_gpu_info():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU : {props.name}")
    print(f"SM  : {props.major}.{props.minor}  |  SMs: {props.multi_processor_count}")
    print(f"VRAM: {props.total_memory / 1024**3:.1f} GB\n")


def cuda_event_time(fn, reps=20):
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(reps):
        fn()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / reps


def build_scatter_moe(args, device, dtype):
    """Build ScatterMoE MLP + external gate.

    ScatterMoE is a GEMM kernel, not a full MoE layer — it needs
    pre-computed expert_weights and expert_indices from an external router.
    We create a simple nn.Linear gate to produce those.
    """
    scatter_mlp = ScatterMLP(
        input_size=args.hidden,
        hidden_size=args.intermediate,
        num_experts=args.experts,
        top_k=args.topk,
        bias=False,
        activation=F.silu,
    ).to(device, dtype=dtype)

    gate = nn.Linear(args.hidden, args.experts, bias=False).to(device, dtype=dtype)

    def forward(x):
        logits = gate(x)
        scores = F.softmax(logits, dim=-1)
        expert_weights, expert_indices = torch.topk(scores, args.topk, dim=-1)
        return scatter_mlp(x, expert_weights, expert_indices)

    return forward


def build_sonic_moe(args, device, dtype):
    """Build SonicMoE module.

    SonicMoE handles routing internally — forward() returns a
    (output, routing_metadata) tuple; we index [0] to get the output tensor.
    """
    sonic = MoE(
        num_experts=args.experts,
        num_experts_per_tok=args.topk,
        hidden_size=args.hidden,
        intermediate_size=args.intermediate,
        activation_function=ActivationType.SWIGLU,
        add_bias=False,
        std=0.02,
    ).to(device, dtype=dtype)

    def forward(x):
        return sonic(x)[0]  # [0] extracts output from (output, metadata) tuple

    return forward


def profile_one(name, fn, x, trace_path, warmup_iters):
    """Run torch.profiler for one backend and export trace."""
    for _ in range(warmup_iters):
        fn(x)
    torch.cuda.synchronize()

    # schedule: skip 1 step (init noise), warmup 1 (let caches settle), record 3 active steps
    schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        record_shapes=True,
        with_stack=False,
    ) as prof:
        for _ in range(5):  # 1 wait + 1 warmup + 3 active = 5 steps
            with torch.profiler.record_function(f"{name}_forward"):
                fn(x)
            prof.step()
        torch.cuda.synchronize()

    prof.export_chrome_trace(trace_path)

    # divide by 3.0 because schedule records 3 active steps
    cuda_time_us = sum(
        e.device_time_total
        for e in prof.key_averages()
        if e.key == f"{name}_forward"
    )
    avg_cuda_us = cuda_time_us / 3.0  # 3 active steps

    kernels = sorted(
        [e for e in prof.key_averages() if e.device_time_total > 0],
        key=lambda e: -e.device_time_total,
    )
    return avg_cuda_us, kernels


def main():
    parser = argparse.ArgumentParser(
        description="SonicMoE vs ScatterMoE profiling"
    )
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=1536)
    parser.add_argument("--intermediate", type=int, default=256)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trace-dir", type=str, default="sonicmoe/traces")
    args = parser.parse_args()

    os.makedirs(args.trace_dir, exist_ok=True)
    device = "cuda"
    dtype = torch.bfloat16

    print("=" * 72)
    print("SonicMoE vs ScatterMoE — MoE Forward Pass Profiling")
    print("=" * 72)
    print_gpu_info()
    print(f"Tokens={args.tokens}, Hidden={args.hidden}, "
          f"Intermediate={args.intermediate}")
    print(f"Experts={args.experts}, TopK={args.topk}\n")

    x = torch.randn(args.tokens, args.hidden, device=device, dtype=dtype)

    # FLOPs per forward pass:
    #   up_proj: 2 * T * K * (2n * d)  — factor 2n for SwiGLU gate+up
    #   down_proj: 2 * T * K * (n * d)
    #   total: 2 * T * K * (2*n*d + n*d) = 2 * T * K * 3*n*d
    flops = 2 * args.tokens * args.topk * (
        2 * args.intermediate * args.hidden  # up+gate (SwiGLU)
        + args.intermediate * args.hidden    # down
    )

    # ── ScatterMoE ────────────────────────────────────────────────────────
    print("-" * 72)
    print("[1/2] Profiling ScatterMoE (Triton baseline)...")
    print("-" * 72)

    scatter_fn = build_scatter_moe(args, device, dtype)
    scatter_trace = os.path.join(args.trace_dir, "scattermoe.json")

    scatter_us, scatter_kernels = profile_one(
        "scattermoe", scatter_fn, x, scatter_trace, args.warmup
    )
    scatter_event_ms = cuda_event_time(lambda: scatter_fn(x))

    print(f"  Profiler CUDA time: {scatter_us:.1f} µs")
    print(f"  Event time:         {scatter_event_ms:.3f} ms")
    print(f"  Trace: {scatter_trace}\n")

    # ── SonicMoE ──────────────────────────────────────────────────────────
    print("-" * 72)
    print("[2/2] Profiling SonicMoE (CUTLASS 3.x)...")
    print("-" * 72)

    sonic_fn = build_sonic_moe(args, device, dtype)
    sonic_trace = os.path.join(args.trace_dir, "sonicmoe.json")

    sonic_us, sonic_kernels = profile_one(
        "sonicmoe", sonic_fn, x, sonic_trace, args.warmup
    )
    sonic_event_ms = cuda_event_time(lambda: sonic_fn(x))

    print(f"  Profiler CUDA time: {sonic_us:.1f} µs")
    print(f"  Event time:         {sonic_event_ms:.3f} ms")
    print(f"  Trace: {sonic_trace}\n")

    # ── Comparison ────────────────────────────────────────────────────────
    scatter_tflops = flops / (scatter_event_ms * 1e-3) / 1e12
    sonic_tflops = flops / (sonic_event_ms * 1e-3) / 1e12
    speedup = scatter_event_ms / sonic_event_ms

    print("=" * 72)
    print("Comparison (CUDA event timing, 20-iter average)")
    print("=" * 72)
    print(f"  {'Metric':<30} {'ScatterMoE':<18} {'SonicMoE':<18}")
    print("  " + "-" * 66)
    print(f"  {'CUDA time (ms)':<30} {scatter_event_ms:<18.3f} {sonic_event_ms:<18.3f}")
    print(f"  {'TFLOPS':<30} {scatter_tflops:<18.2f} {sonic_tflops:<18.2f}")
    print(f"  {'Speedup (vs ScatterMoE)':<30} {'1.00x':<18} {f'{speedup:.2f}x':<18}")
    print()

    for label, kernels in [("ScatterMoE", scatter_kernels),
                           ("SonicMoE", sonic_kernels)]:
        print(f"Top CUDA kernels — {label}:")
        print(f"  {'Kernel':<60} {'CUDA µs':<12} {'Calls'}")
        print("  " + "-" * 78)
        for e in kernels[:8]:
            print(f"  {e.key[:58]:<60} {e.device_time_total:<12.1f} {e.count}")
        print()

    print("Traces saved to:")
    print(f"  ScatterMoE: {scatter_trace}")
    print(f"  SonicMoE:   {sonic_trace}")


if __name__ == "__main__":
    main()
