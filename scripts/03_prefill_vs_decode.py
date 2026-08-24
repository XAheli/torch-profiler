#!/mnt/podman_storage/ahpoddar/conda_envs/pt_nightly/bin/python
"""
03 — Prefill vs Decode: the two phases of LLM inference

Profiles model.generate() end-to-end:
  1. PREFILL — process the entire prompt at once (compute-bound, large batched GEMMs).
  2. DECODE  — generate tokens one at a time (memory-bound, tiny sequential GEMMs).

Includes memory profiling (profile_memory=True) to show KV cache growth.
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


def load_model_and_tokenizer(model_id, device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device, trust_remote_code=True,
        ).eval()
    except Exception as e:
        print(f"Could not load {model_id}: {e}")
        print("Falling back to TinyLlama/TinyLlama-1.1B-Chat-v1.0\n")
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device,
        ).eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, model_id


def main():
    parser = argparse.ArgumentParser(description="Profile prefill vs decode")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--prompt", type=str, default="The key insight about GPU profiling is that")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--trace-dir", type=str, default="traces")
    args = parser.parse_args()

    print_gpu_info()
    os.makedirs(args.trace_dir, exist_ok=True)

    device = "cuda"
    dtype = torch.bfloat16

    model, tokenizer, model_id = load_model_and_tokenizer(args.model, device, dtype)
    short_name = model_id.split("/")[-1]
    print(f"Model   : {model_id}")
    print(f"Prompt  : \"{args.prompt}\"")
    print(f"Tokens  : max_new_tokens={args.max_new_tokens}\n")

    if args.compile:
        model = torch.compile(model)

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    tag = "compiled" if args.compile else "eager"

    with torch.no_grad():
        for _ in range(args.warmup):
            model.generate(**inputs, max_new_tokens=4, do_sample=False)
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with torch.no_grad():
            with torch.profiler.record_function("generate"):
                output_ids = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                )

    mem_after = torch.cuda.memory_allocated()
    mem_peak = torch.cuda.max_memory_allocated()

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    num_prompt_tokens = inputs["input_ids"].shape[1]
    num_generated = output_ids.shape[1] - num_prompt_tokens

    trace_path = os.path.join(args.trace_dir, f"03_prefill_decode_{short_name}_{tag}.json")
    prof.export_chrome_trace(trace_path)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))

    print(f"\n{'='*60}")
    print(f"Prompt tokens    : {num_prompt_tokens}")
    print(f"Generated tokens : {num_generated}")
    print(f"Memory before    : {mem_before / 1024**2:.1f} MB")
    print(f"Memory after     : {mem_after / 1024**2:.1f} MB")
    print(f"Peak memory      : {mem_peak / 1024**2:.1f} MB")
    print(f"KV cache growth  : ~{(mem_after - mem_before) / 1024**2:.1f} MB")
    print(f"{'='*60}")
    print(f"\nGenerated: {generated_text}")
    print(f"\nTrace saved → {trace_path}")
    print(f"View it at  → https://ui.perfetto.dev/")


if __name__ == "__main__":
    main()
