#!/mnt/podman_storage/ahpoddar/conda_envs/pt_nightly/bin/python
"""
08 — torch.compile deep dive: comparing compilation modes

Profiles the SAME LlamaDecoderLayer under three torch.compile modes:
  - default         : baseline graph-mode compilation
  - reduce-overhead : wraps execution in CUDA graphs (one graph-replay kernel)
  - max-autotune    : benchmarks multiple kernel implementations, picks fastest

Includes eager baseline for comparison.
"""

import argparse
import os
import time

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
        print(f"Could not load config for {model_id}, falling back to TinyLlama")
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        config = AutoConfig.from_pretrained(model_id)

    layer = LlamaDecoderLayer(config, layer_idx=0).to(device, dtype=dtype).eval()
    rotary_emb = LlamaRotaryEmbedding(config=config).to(device, dtype=dtype)
    return layer, rotary_emb, config, model_id


def profile_mode(layer, hidden, position_embeddings, mode, warmup_iters, trace_dir, short_name):
    print(f"\n{'='*60}")
    print(f"  Compiling with mode = '{mode}'")
    print(f"{'='*60}")

    compiled = torch.compile(layer, mode=mode)

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(warmup_iters):
            compiled(hidden, position_embeddings=position_embeddings)
        torch.cuda.synchronize()
    compile_time = time.perf_counter() - t0
    print(f"  Warmup + compilation: {compile_time:.2f}s")

    schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)

    try:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=schedule,
            record_shapes=True,
            with_stack=True,
        ) as prof:
            for _ in range(5):
                with torch.no_grad():
                    with torch.profiler.record_function(f"compile_mode_{mode}"):
                        compiled(hidden, position_embeddings=position_embeddings)
                prof.step()
            torch.cuda.synchronize()
    except Exception as e:
        print(f"  Scheduled profiling failed for mode '{mode}': {e}")
        print(f"  Falling back to single-shot profiling")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
        ) as prof:
            with torch.no_grad():
                with torch.profiler.record_function(f"compile_mode_{mode}"):
                    compiled(hidden, position_embeddings=position_embeddings)
                    torch.cuda.synchronize()

    trace_path = os.path.join(trace_dir, f"08_compile_{mode}_{short_name}.json")
    prof.export_chrome_trace(trace_path)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    print(f"  Trace saved -> {trace_path}")

    total_cuda_us = sum(
        e.device_time_total for e in prof.key_averages() if e.device_time_total > 0
    )
    return {
        "mode": mode,
        "compile_time_s": compile_time,
        "cuda_time_us": total_cuda_us,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare torch.compile modes")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trace-dir", type=str, default="traces")
    args = parser.parse_args()

    print_gpu_info()
    os.makedirs(args.trace_dir, exist_ok=True)

    device = "cuda"
    dtype = torch.bfloat16

    layer, rotary_emb, config, model_id = load_decoder_layer(args.model, device, dtype)
    short_name = model_id.split("/")[-1]
    print(f"Model  : {model_id}")
    print(f"Shape  : batch={args.batch}, seq={args.seq}, hidden={config.hidden_size}")

    hidden = torch.randn(args.batch, args.seq, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(args.seq, device=device).unsqueeze(0).expand(args.batch, -1)
    position_embeddings = rotary_emb(hidden, position_ids)

    # --- Eager baseline ---
    print(f"\n{'='*60}")
    print(f"  Baseline: eager (no compile)")
    print(f"{'='*60}")

    with torch.no_grad():
        for _ in range(args.warmup):
            layer(hidden, position_embeddings=position_embeddings)
        torch.cuda.synchronize()

    schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for _ in range(5):
            with torch.no_grad():
                with torch.profiler.record_function("eager_baseline"):
                    layer(hidden, position_embeddings=position_embeddings)
            prof.step()
        torch.cuda.synchronize()

    trace_path = os.path.join(args.trace_dir, f"08_compile_eager_{short_name}.json")
    prof.export_chrome_trace(trace_path)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    print(f"  Trace saved -> {trace_path}")

    eager_cuda_us = sum(
        e.device_time_total for e in prof.key_averages() if e.device_time_total > 0
    )

    # --- Compile modes ---
    modes = ["default", "reduce-overhead", "max-autotune"]
    results = []
    for mode in modes:
        fresh_layer, _, _, _ = load_decoder_layer(args.model, device, dtype)
        res = profile_mode(
            fresh_layer, hidden, position_embeddings, mode,
            args.warmup, args.trace_dir, short_name,
        )
        results.append(res)

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  Summary: compilation time vs runtime (CUDA us)")
    print(f"{'='*60}")
    print(f"  {'Mode':<20} {'Compile (s)':>12} {'CUDA time (us)':>16} {'Speedup vs eager':>18}")
    print(f"  {'-'*66}")
    print(f"  {'eager':<20} {'-':>12} {eager_cuda_us:>16.0f} {'1.00x':>18}")
    for r in results:
        speedup = eager_cuda_us / r["cuda_time_us"] if r["cuda_time_us"] > 0 else float("inf")
        print(f"  {r['mode']:<20} {r['compile_time_s']:>12.2f} {r['cuda_time_us']:>16.0f} {speedup:>17.2f}x")
    print()


if __name__ == "__main__":
    main()
