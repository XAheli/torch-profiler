# How Fast Is Your Kernel, Really?

GPU kernel profiling on NVIDIA H200 — three SOTA kernels analyzed with `torch.profiler` (Perfetto traces) and NVIDIA Nsight Compute (kernel-level roofline, occupancy, memory throughput).

## Kernels

```
deepgemm/          DeepGEMM FP8 vs bf16 GEMM (3.09x on H200)
├── profile.py
└── traces/

flash_attn/         FlashAttention-3 vs FA-2 (2.0–2.6x across seq lengths)
├── profile.py
└── traces/

sonicmoe/           SonicMoE IO-aware MoE forward (35 TFLOPS on H200)
├── profile.py
└── traces/
```

| Kernel | Source | What it profiles | Speedup |
|--------|--------|-----------------|---------|
| **DeepGEMM FP8** | [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | JIT-compiled FP8 GEMM vs cuBLAS bf16 | 3.09x |
| **FlashAttention-3** | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) `hopper/` | Hopper WGMMA+TMA attention vs FA-2 | 2.0–2.6x |
| **SonicMoE** | [Dao-AILab/sonic-moe](https://github.com/Dao-AILab/sonic-moe) | IO-aware MoE with gather fusion + ping-pong scheduling | 35 TFLOPS |

## Setup

Two Python environments are needed:

```bash
# Environment 1: pt_nightly (Python 3.11) — DeepGEMM + FlashAttention-3
pip install -r requirements.txt

# Environment 2: py312_sonic (Python 3.12) — SonicMoE
pip install sonic-moe
```

Source builds required for DeepGEMM and FA-3:

| Kernel | Build |
|--------|-------|
| DeepGEMM | `git clone https://github.com/deepseek-ai/DeepGEMM && cd DeepGEMM && python setup.py build_ext --inplace` |
| FlashAttention-3 | `git clone https://github.com/Dao-AILab/flash-attention && cd flash-attention/hopper && python setup.py install` |
| SonicMoE | `pip install sonic-moe` (requires Python 3.12+) |

## Usage

```bash
# DeepGEMM FP8 vs bf16
python deepgemm/profile.py --M 2048 --K 4096 --N 14336

# FlashAttention-3 vs FA-2
python flash_attn/profile.py --batch 4 --heads 32 --head-dim 128

# SonicMoE (requires py312_sonic env)
python sonicmoe/profile.py --tokens 2048 --experts 128 --topk 8
```

Nsight Compute:
```bash
ncu --set full -o deepgemm/traces/ncu_deepgemm \
    --kernel-name regex:"sm90_fp8" --launch-count 1 \
    python deepgemm/profile.py

ncu --set full -o flash_attn/traces/ncu_fa3 \
    --kernel-name regex:"device_kernel" --launch-skip 10 --launch-count 1 \
    python flash_attn/profile.py

ncu --set full -o sonicmoe/traces/ncu_sonicmoe \
    --kernel-name regex:"quackgemm" --launch-skip 3 --launch-count 1 \
    python sonicmoe/profile.py
```

## Traces

Each kernel directory has a `traces/` folder with:
- `.json` — Chrome traces for [Perfetto UI](https://ui.perfetto.dev/)
- `.ncu-rep` — Nsight Compute reports for NCU GUI

## Environment

- 8x NVIDIA H200 SXM (SM 9.0, 143 GB HBM3e)
- PyTorch 2.14.0.dev (nightly) + CUDA 12.6
- PyTorch 2.9.1 + CUDA 12.8 (SonicMoE env)
- Nsight Compute 2025.1.0

## References

- [DeepGEMM — clean and efficient FP8 GEMM kernels](https://github.com/deepseek-ai/DeepGEMM)
- [FlashAttention-3 Paper](https://arxiv.org/abs/2407.08608)
- [SonicMoE Paper](https://arxiv.org/abs/2512.14080)
- [PyTorch Profiler Docs](https://pytorch.org/docs/stable/profiler.html)
- [Nsight Compute User Guide](https://docs.nvidia.com/nsight-compute/)
