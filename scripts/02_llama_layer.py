#!/mnt/podman_storage/ahpoddar/conda_envs/pt_nightly/bin/python
"""
02 — Profile a real LlamaDecoderLayer

Profiles a SINGLE decoder layer so the trace is readable without drowning
in 32+ repeated blocks.  Shows the full dispatch chain:
  RMSNorm → QKV proj → RoPE → attention → O proj → residual →
  RMSNorm → gate/up proj → SiLU → down proj → residual
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


def load_decoder_layer(model_id, device, dtype):
    from transformers import AutoConfig
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding

    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    except Exception:
        print(f"Could not load config for {model_id}, falling back to TinyLlama")
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        config = AutoConfig.from_pretrained(model_id)

    layer = LlamaDecoderLayer(config, layer_idx=0).to(device, dtype=dtype).eval()
    rotary_emb = LlamaRotaryEmbedding(config=config).to(device, dtype=dtype)
    return layer, rotary_emb, config, model_id


def main():
    parser = argparse.ArgumentParser(description="Profile a LlamaDecoderLayer")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trace-dir", type=str, default="traces")
    args = parser.parse_args()

    print_gpu_info()
    os.makedirs(args.trace_dir, exist_ok=True)

    device = "cuda"
    dtype = torch.bfloat16

    layer, rotary_emb, config, model_id = load_decoder_layer(args.model, device, dtype)
    short_name = model_id.split("/")[-1]
    print(f"Model config : {model_id}")
    print(f"Hidden size  : {config.hidden_size}")
    print(f"Num heads    : {config.num_attention_heads}  (KV heads: {config.num_key_value_heads})")
    print(f"Intermediate : {config.intermediate_size}")
    print(f"Input shape  : batch={args.batch}, seq={args.seq}\n")

    hidden = torch.randn(args.batch, args.seq, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(args.seq, device=device).unsqueeze(0).expand(args.batch, -1)
    position_embeddings = rotary_emb(hidden, position_ids)

    fn = torch.compile(layer) if args.compile else layer
    tag = "compiled" if args.compile else "eager"

    with torch.no_grad():
        for _ in range(args.warmup):
            fn(hidden, position_embeddings=position_embeddings)
        torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        with torch.no_grad():
            with torch.profiler.record_function("llama_decoder_layer"):
                fn(hidden, position_embeddings=position_embeddings)
                torch.cuda.synchronize()

    trace_path = os.path.join(args.trace_dir, f"02_llama_layer_{short_name}_{tag}.json")
    prof.export_chrome_trace(trace_path)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
    print(f"\nTrace saved → {trace_path}")
    print(f"View it at  → https://ui.perfetto.dev/")

    cuda_events = [e for e in prof.key_averages() if e.device_time_total > 0]
    print(f"\nTotal CUDA kernels launched: {len(cuda_events)}")


if __name__ == "__main__":
    main()
