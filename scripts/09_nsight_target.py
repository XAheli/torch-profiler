#!/mnt/podman_storage/ahpoddar/conda_envs/pt_nightly/bin/python
"""
09 — Nsight Systems / Nsight Compute target script

NVTX-annotated workloads for nsys and ncu profiling.  Uses a real
LlamaDecoderLayer with labeled prefill and decode phases.

Workloads:
  - layer     : full LlamaDecoderLayer (attention + MLP + norms + residuals)
  - attention : self-attention only (QKV -> SDPA -> O projection)
  - mlp       : MLP only (gate/up -> SiLU -> down projection)
"""

import argparse
import os

import torch

os.environ["PATH"] = "/usr/local/cuda-12.8/bin:" + os.environ.get("PATH", "")
os.environ["HF_HOME"] = "/mnt/podman_storage/ahpoddar/.cache/huggingface"


def print_gpu_info():
    if not torch.cuda.is_available():
        print("CUDA not available — profiling will be CPU-only")
        return
    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    print(f"GPU : {props.name}")
    print(f"SM  : {props.major}.{props.minor}  |  SMs: {props.multi_processor_count}")
    print(f"VRAM: {props.total_memory / 1024**3:.1f} GB\n")


def load_decoder_layer(model_id, device, dtype):
    from transformers import AutoConfig
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding

    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    except Exception:
        fallback = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        print(f"Could not load config for {model_id}, falling back to {fallback}")
        model_id = fallback
        config = AutoConfig.from_pretrained(model_id)

    layer = LlamaDecoderLayer(config, layer_idx=0).to(device, dtype=dtype).eval()
    rotary_emb = LlamaRotaryEmbedding(config=config).to(device, dtype=dtype)
    return layer, rotary_emb, config, model_id


def run_layer_workload(layer, rotary_emb, config, batch, seq, device, dtype, n_decode_steps=4):
    hidden = torch.randn(batch, seq, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)
    position_embeddings = rotary_emb(hidden, position_ids)

    torch.cuda.nvtx.range_push("prefill")
    with torch.no_grad():
        layer(hidden, position_embeddings=position_embeddings)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    for step in range(n_decode_steps):
        decode_hidden = torch.randn(batch, 1, config.hidden_size, device=device, dtype=dtype)
        decode_pos = torch.full((batch, 1), seq + step, device=device, dtype=torch.long)
        decode_pos_emb = rotary_emb(decode_hidden, decode_pos)

        torch.cuda.nvtx.range_push(f"decode_step_{step}")
        with torch.no_grad():
            layer(decode_hidden, position_embeddings=decode_pos_emb)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()


def run_attention_workload(layer, rotary_emb, config, batch, seq, device, dtype):
    attn = layer.self_attn
    hidden = torch.randn(batch, seq, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)
    position_embeddings = rotary_emb(hidden, position_ids)

    torch.cuda.nvtx.range_push("self_attention_forward")
    with torch.no_grad():
        attn(hidden, position_embeddings=position_embeddings)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


def run_mlp_workload(layer, config, batch, seq, device, dtype):
    mlp = layer.mlp
    hidden = torch.randn(batch, seq, config.hidden_size, device=device, dtype=dtype)

    torch.cuda.nvtx.range_push("mlp_forward")
    with torch.no_grad():
        mlp(hidden)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


def main():
    parser = argparse.ArgumentParser(description="NVTX-annotated target for Nsight")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--workload", type=str, default="layer",
                        choices=["layer", "attention", "mlp"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trace-dir", type=str, default="traces")
    args = parser.parse_args()

    print_gpu_info()
    os.makedirs(args.trace_dir, exist_ok=True)

    device = "cuda"
    dtype = torch.bfloat16

    layer, rotary_emb, config, model_id = load_decoder_layer(args.model, device, dtype)
    short_name = model_id.split("/")[-1]
    print(f"Model    : {model_id}")
    print(f"Workload : {args.workload}")
    print(f"Shape    : batch={args.batch}, seq={args.seq}")

    hidden = torch.randn(args.batch, args.seq, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(args.seq, device=device).unsqueeze(0).expand(args.batch, -1)
    position_embeddings = rotary_emb(hidden, position_ids)
    with torch.no_grad():
        for _ in range(args.warmup):
            layer(hidden, position_embeddings=position_embeddings)
        torch.cuda.synchronize()

    print(f"\nRunning {args.workload} workload with NVTX annotations...\n")

    if args.workload == "layer":
        run_layer_workload(layer, rotary_emb, config, args.batch, args.seq, device, dtype)
    elif args.workload == "attention":
        run_attention_workload(layer, rotary_emb, config, args.batch, args.seq, device, dtype)
    elif args.workload == "mlp":
        run_mlp_workload(layer, config, args.batch, args.seq, device, dtype)

    torch.cuda.synchronize()
    print("Done.\n")

    script_path = os.path.abspath(__file__)
    trace_base = os.path.abspath(args.trace_dir)

    print("=" * 70)
    print("  Nsight Systems — full timeline with NVTX")
    print("=" * 70)
    print(f"""
  nsys profile \\
    --stats=true \\
    --output={trace_base}/09_nsight_{args.workload} \\
    --trace=cuda,nvtx \\
    --force-overwrite=true \\
    {script_path} --workload {args.workload} --model {args.model}
""")

    print("=" * 70)
    print("  Nsight Compute — single-kernel deep dive")
    print("=" * 70)
    print(f"""
  ncu \\
    --set full \\
    --output={trace_base}/09_ncu_{args.workload} \\
    --nvtx \\
    --nvtx-include "prefill/" \\
    --kernel-name regex:gemm \\
    --launch-count 5 \\
    {script_path} --workload {args.workload} --model {args.model}
""")

    print("  Tip: Use --nvtx-include to filter which NVTX regions ncu profiles.")
    print("  Tip: Use --kernel-name regex:<pattern> to profile specific kernel types.")


if __name__ == "__main__":
    main()
