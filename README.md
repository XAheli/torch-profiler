# Profiling in PyTorch: From torch.profiler to Nsight Compute

GPU profiling scripts for PyTorch workloads on NVIDIA Hopper (H200).

Covers `torch.profiler` with Perfetto traces and NVIDIA Nsight Compute for kernel-level analysis across real model layers, SOTA attention kernels, and quantized inference.

## Scripts

| # | Script | Profiles |
|---|--------|----------|
| 01 | `01_warmup_matmul.py` | matmul+add — overhead-bound vs compute-bound |
| 02 | `02_llama_layer.py` | Single `LlamaDecoderLayer` dispatch chain |
| 03 | `03_prefill_vs_decode.py` | `model.generate()` — prefill vs decode phases |
| 04 | `04_liger_on_llama.py` | Vanilla vs Liger fused kernels on a Llama layer |
| 05 | `05_flash_attn_compare.py` | SDPA math / flash / cuDNN / FlashAttention-3 |
| 06 | `06_fp8_gemm.py` | bf16 vs FP8 GEMM on Hopper Tensor Cores |
| 07 | `07_flashmla_demo.py` | FlashMLA paged KV-cache decode |
| 08 | `08_compile_modes.py` | `torch.compile` default / reduce-overhead / max-autotune |
| 09 | `09_nsight_target.py` | NVTX-annotated target for `nsys` and `ncu` |

## Setup

```bash
pip install -r requirements.txt
```

Optional (built from source for Hopper):

| Kernel | Source |
|--------|--------|
| FlashAttention-3 | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) `hopper/` |
| FlashMLA | [deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA) |
| Liger | `pip install liger-kernel` |

## Usage

```bash
python scripts/01_warmup_matmul.py --size 4096
python scripts/02_llama_layer.py
python scripts/05_flash_attn_compare.py --backend auto
python scripts/06_fp8_gemm.py
python scripts/07_flashmla_demo.py --head-dim-v 512

# Nsight Compute
ncu --set full -o traces/ncu_gemm --kernel-name regex:nvjet --launch-count 1 \
    python scripts/01_warmup_matmul.py --size 4096
```

## Traces

- `torch.profiler` exports Chrome traces (`.json`) to `traces/` — open at [Perfetto UI](https://ui.perfetto.dev/)
- Nsight Compute reports (`.ncu-rep`) — open in NCU GUI

## Environment

- 8x NVIDIA H200 SXM (SM 9.0, 143 GB HBM3e)
- PyTorch 2.14.0.dev (nightly) + CUDA 12.6
- Nsight Systems 2025.3.2 / Nsight Compute 2025.1.0

## References

- [FlashAttention-3 Paper](https://arxiv.org/abs/2407.08608)
- [Liger Kernel Paper](https://arxiv.org/abs/2410.10989)
- [FlashMLA Repository](https://github.com/deepseek-ai/FlashMLA)
- [PyTorch Profiler Docs](https://pytorch.org/docs/stable/profiler.html)
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/)
- [Nsight Compute User Guide](https://docs.nvidia.com/nsight-compute/)
