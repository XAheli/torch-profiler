#!/mnt/podman_storage/ahpoddar/conda_envs/pt_nightly/bin/python
"""
07 — FlashMLA: DeepSeek's flagship decode-attention kernel

Demonstrates FlashMLA's seesaw tile scheduler and paged KV-cache decode
on H200.  MLA (Multi-head Latent Attention) compresses KV into a low-rank
latent space, making decode memory-bound on the compressed dimension.

FlashMLA's key insight: a "seesaw" scheduling strategy that dynamically
balances work across SMs, avoiding the tail-effect that plagues naive
split-KV attention kernels.

Parameters inspired by DeepSeek-V2's MLA config.
"""

import argparse
import os

import torch

os.environ["PATH"] = "/usr/local/cuda-12.8/bin:" + os.environ.get("PATH", "")
os.environ["HF_HOME"] = "/mnt/podman_storage/ahpoddar/.cache/huggingface"


def print_gpu_info():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    print(f"GPU : {props.name}")
    print(f"SM  : {props.major}.{props.minor}  |  SMs: {props.multi_processor_count}")
    print(f"VRAM: {props.total_memory / 1024**3:.1f} GB\n")


def main():
    parser = argparse.ArgumentParser(description="Profile FlashMLA decode kernel")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16,
                        help="Number of query heads")
    parser.add_argument("--head-dim-k", type=int, default=576,
                        help="Compressed KV latent dim (576 for DeepSeek-V2, 512 for V3)")
    parser.add_argument("--head-dim-v", type=int, default=128,
                        help="Value output dim per query head")
    parser.add_argument("--max-seqlen", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trace-dir", type=str, default="traces")
    args = parser.parse_args()

    print_gpu_info()
    os.makedirs(args.trace_dir, exist_ok=True)

    try:
        from flash_mla import get_mla_metadata, flash_mla_with_kvcache
    except ImportError:
        print("[!] FlashMLA not installed.")
        print("    Build from source: cd FlashMLA && python setup.py install")
        print("    IMPORTANT: Do NOT run scripts from inside FlashMLA-src/")
        return

    device = "cuda"
    dtype = torch.bfloat16

    batch = args.batch
    num_heads = args.num_heads
    head_dim_k = args.head_dim_k
    head_dim_v = args.head_dim_v
    max_seqlen = args.max_seqlen
    page_size = 64

    num_pages_per_seq = (max_seqlen + page_size - 1) // page_size
    total_pages = num_pages_per_seq * batch

    print(f"FlashMLA decode configuration:")
    print(f"  Batch        : {batch}")
    print(f"  Num heads    : {num_heads}")
    print(f"  Head dim (k) : {head_dim_k}  (compressed KV latent)")
    print(f"  Head dim (v) : {head_dim_v}  (value per query head)")
    print(f"  Max seqlen   : {max_seqlen}")
    print(f"  Page size    : {page_size}")
    print(f"  Pages/seq    : {num_pages_per_seq}")
    print(f"  Total pages  : {total_pages}\n")

    cache_seqlens = torch.full((batch,), max_seqlen, device=device, dtype=torch.int32)

    # --- Profile metadata computation (tile scheduler setup) ---
    print(f"{'='*60}")
    print(f"  Step 1: get_mla_metadata — tile scheduler setup")
    print(f"{'='*60}")

    schedule_meta = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=schedule_meta,
        record_shapes=True,
    ) as prof_meta:
        for _ in range(5):
            with torch.profiler.record_function("get_mla_metadata"):
                tile_scheduler_metadata, num_splits = get_mla_metadata(
                    cache_seqlens, head_dim_k, num_heads
                )
            prof_meta.step()

    prof_meta.export_chrome_trace(os.path.join(args.trace_dir, "07_flashmla_metadata.json"))
    print(prof_meta.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print(f"  num_splits returned: {num_splits}")

    # --- Setup paged KV cache ---
    # MLA stores a single compressed latent per token: num_heads_k=1
    # k_cache shape: [num_blocks, page_block_size, num_heads_k=1, head_dim_k]
    num_heads_k = 1
    kv_cache = torch.randn(
        total_pages, page_size, num_heads_k, head_dim_k,
        device=device, dtype=dtype,
    )

    block_table = torch.arange(
        total_pages, device=device, dtype=torch.int32,
    ).reshape(batch, num_pages_per_seq)

    # q shape: [batch, seq_len_q=1 (decode), num_heads_q, head_dim_k]
    seq_len_q = 1
    q = torch.randn(batch, seq_len_q, num_heads, head_dim_k, device=device, dtype=dtype)

    # --- Warmup decode ---
    for _ in range(args.warmup):
        flash_mla_with_kvcache(
            q, kv_cache, block_table, cache_seqlens,
            head_dim_v, tile_scheduler_metadata, num_splits,
            softmax_scale=head_dim_k ** -0.5,
        )
    torch.cuda.synchronize()

    # --- Profile decode kernel ---
    print(f"\n{'='*60}")
    print(f"  Step 2: flash_mla_with_kvcache — seesaw-scheduled decode")
    print(f"{'='*60}")

    schedule_decode = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=schedule_decode,
        record_shapes=True,
        with_stack=True,
    ) as prof_decode:
        for _ in range(5):
            with torch.no_grad():
                with torch.profiler.record_function("flash_mla_decode"):
                    out, lse = flash_mla_with_kvcache(
                        q, kv_cache, block_table, cache_seqlens,
                        head_dim_v, tile_scheduler_metadata, num_splits,
                        softmax_scale=head_dim_k ** -0.5,
                    )
            prof_decode.step()
        torch.cuda.synchronize()

    trace_path = os.path.join(args.trace_dir, "07_flashmla_decode.json")
    prof_decode.export_chrome_trace(trace_path)
    print(prof_decode.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    total_cuda_us = sum(
        e.device_time_total for e in prof_decode.key_averages() if e.device_time_total > 0
    )

    tokens_per_sec = batch / (total_cuda_us * 1e-6) if total_cuda_us > 0 else 0
    kv_bytes = kv_cache.nelement() * kv_cache.element_size()
    bandwidth_gb_s = kv_bytes / (total_cuda_us * 1e-6) / 1e9 if total_cuda_us > 0 else 0

    print(f"\n{'='*60}")
    print(f"  FlashMLA decode summary")
    print(f"{'='*60}")
    print(f"  Output shape      : {out.shape}")
    print(f"  CUDA time         : {total_cuda_us:.0f} us")
    print(f"  Tokens/sec        : {tokens_per_sec:,.0f}")
    print(f"  KV cache size     : {kv_bytes / 1024**2:.1f} MB")
    print(f"  Effective BW      : {bandwidth_gb_s:.1f} GB/s")
    print(f"  Trace saved       -> {trace_path}")

    print(f"\n  Seesaw scheduling insight:")
    print(f"  Unlike split-KV (FlashDecoding), FlashMLA's tile scheduler")
    print(f"  dynamically assigns variable-length tile groups to SMs,")
    print(f"  avoiding the 'tail effect' where some SMs finish early and idle.")
    print(f"  The metadata tensor encodes this schedule — profiling shows")
    print(f"  the kernel runs as a single efficient launch.\n")

    print(f"View traces at -> https://ui.perfetto.dev/")


if __name__ == "__main__":
    main()
