# Profiling Real Models in PyTorch: From torch.profiler to Nsight

**Red Hat PyTorch Team — IISC Bangalore Workshop**

## What this is

A hands-on workshop that profiles **real transformer model layers** on NVIDIA H200 GPUs.
Every script (except a 2-minute matmul warmup) runs actual HuggingFace Transformers code —
`LlamaDecoderLayer`, `model.generate()`, real attention backends, real fused kernels.

You will learn to read GPU traces, diagnose bottlenecks, and measure the impact of
FlashAttention-3, Liger kernels, FlashMLA, FP8, and `torch.compile` on real model workloads.

## How this differs from existing tutorials

Most profiling tutorials profile toy operations — a standalone matmul, a hand-written
attention loop, a synthetic MLP.  This workshop profiles real models:

- **Real layers, not toys.** We profile `LlamaDecoderLayer` from HuggingFace Transformers.
- **Real inference, not microbenchmarks.** We profile `model.generate()` end-to-end and
  show prefill vs decode phases.
- **SOTA kernels.** FlashAttention-3, FlashMLA, Liger, FP8 — all profiled side-by-side.
- **Memory profiling.** Most tutorials skip `profile_memory=True`.  We don't.
- **H200/Hopper focus.** cuDNN attention, FP8 Tensor Cores, WGMMA instructions.

## Prerequisites

- Python 3.10+
- NVIDIA GPU (Hopper/H200 recommended; Ampere works for most scripts)
- CUDA 12.x + PyTorch 2.4+

Model access: scripts default to `meta-llama/Llama-3.1-8B` (config only — we build
individual layers, not the full model).  If you don't have access, scripts automatically
fall back to `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.

## Setup

```bash
pip install -r requirements.txt
```

## SOTA Kernel Sources (built from source)

| Kernel | Repository | Build |
|--------|-----------|-------|
| FlashAttention-3 | https://github.com/Dao-AILab/flash-attention (hopper/ dir) | `cd hopper && python setup.py install` |
| FlashMLA | https://github.com/deepseek-ai/FlashMLA | `python setup.py install` |
| Liger Kernel | https://github.com/linkedin/Liger-Kernel | `pip install liger-kernel` |

**Important:** Do NOT run scripts from inside `FlashMLA-src/` or `DeepGEMM-src/` directories (circular imports).

## Presentation flow (30+15 min)

### Part 1: torch.profiler Foundation (10 min) — Scripts 01–03

| # | Script | Time | What it profiles |
|---|--------|------|-----------------|
| 01 | `01_warmup_matmul.py` | 2 min | Matmul+add: overhead-bound (64×64) vs compute-bound (4096×4096). Profiler basics. |
| 02 | `02_llama_layer.py` | 5 min | A single LlamaDecoderLayer — the full dispatch chain. |
| 03 | `03_prefill_vs_decode.py` | 3 min | `model.generate()` end-to-end — prefill vs decode phases, KV cache growth. |

### Part 2: SOTA Kernel Profiling (15 min) — Scripts 04–08

| # | Script | Time | What it profiles |
|---|--------|------|-----------------|
| 04 | `04_liger_on_llama.py` | 3 min | Liger fused kernels vs vanilla + FusedLinearCrossEntropy comparison. |
| 05 | `05_flash_attn_compare.py` | 5 min | **The showcase:** SDPA math vs flash vs cuDNN vs FlashAttention-3. Side-by-side. |
| 06 | `06_fp8_gemm.py` | 2 min | bf16 vs FP8 GEMM — Hopper Tensor Core differences visible in traces. |
| 07 | `07_flashmla_demo.py` | 3 min | FlashMLA's seesaw scheduling and paged KV-cache decode on H200. |
| 08 | `08_compile_modes.py` | 2 min | torch.compile `default` / `reduce-overhead` / `max-autotune` on the same layer. |

### Part 3: Nsight Deep Dive (10 min) — Script 09

| # | Script | Time | What it profiles |
|---|--------|------|-----------------|
| 09 | `09_nsight_target.py` | 10 min | NVTX-annotated target for Nsight Systems and Nsight Compute. |

## Usage

Scripts use the conda environment at `/mnt/podman_storage/ahpoddar/conda_envs/pt_nightly/bin/python`.
Since the shebang is set, you can run them directly:

```bash
# Part 1: Foundation
./scripts/01_warmup_matmul.py --size 64
./scripts/01_warmup_matmul.py --size 4096 --compile
./scripts/02_llama_layer.py --model meta-llama/Llama-3.1-8B
./scripts/03_prefill_vs_decode.py --max-new-tokens 32

# Part 2: SOTA Kernels
./scripts/04_liger_on_llama.py
./scripts/05_flash_attn_compare.py --batch 4 --seq 512
./scripts/06_fp8_gemm.py
./scripts/07_flashmla_demo.py --batch 4 --max-seqlen 2048
./scripts/08_compile_modes.py

# Part 3: Nsight
./scripts/09_nsight_target.py --workload layer
nsys profile --stats=true -o traces/nsight_layer ./scripts/09_nsight_target.py
```

## Viewing traces

All scripts export Chrome traces to `traces/`.  Open them at:
- **Perfetto UI**: https://ui.perfetto.dev/ (drag and drop the `.json` file)
- **chrome://tracing**: paste the path in Chrome's built-in trace viewer
- **Nsight Systems UI**: for `.nsys-rep` files from nsys

## Hardware tested

- NVIDIA H200 SXM (Hopper, SM90, 143 GB HBM3e)

## References

- [FlashAttention-3 Paper](https://arxiv.org/abs/2407.08608)
- [Liger Kernel Paper](https://arxiv.org/abs/2410.10989)
- [FlashMLA Repository](https://github.com/deepseek-ai/FlashMLA)
- [PyTorch Profiler Docs](https://pytorch.org/docs/stable/profiler.html)
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/)
- [Nsight Compute User Guide](https://docs.nvidia.com/nsight-compute/)
