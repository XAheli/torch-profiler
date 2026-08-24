#!/mnt/podman_storage/ahpoddar/conda_envs/pt_nightly/bin/python
"""
10 — Visualization dashboard for torch-profiler-workshop results

Generates publication-quality charts from profiling data gathered across
scripts 01-09.  Produces individual PNGs + one combined overview figure.

Run standalone after all profiling scripts have completed:
    python scripts/10_visualize.py
"""

import os

os.environ["PATH"] = "/usr/local/cuda-12.8/bin:" + os.environ.get("PATH", "")
os.environ["HF_HOME"] = "/mnt/podman_storage/ahpoddar/.cache/huggingface"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

TRACE_DIR = "traces"
os.makedirs(TRACE_DIR, exist_ok=True)

# ── colour palette (professional blues & greens) ────────────────────────
PAL = {
    "blue1": "#1b4f72",
    "blue2": "#2e86c1",
    "blue3": "#85c1e9",
    "green1": "#0e6655",
    "green2": "#1abc9c",
    "green3": "#a3e4d7",
    "grey":   "#aab7b8",
    "orange": "#e67e22",
    "red":    "#c0392b",
}
COLORS = [PAL["blue1"], PAL["blue2"], PAL["green1"], PAL["green2"],
          PAL["blue3"], PAL["green3"], PAL["orange"]]

DPI = 300

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("ggplot")

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "figure.facecolor": "white",
})


def _bar_labels(ax, fmt="{:.2f}", fontsize=9, pad=3):
    """Add value labels on top of each bar."""
    for bar in ax.patches:
        h = bar.get_height()
        if h == 0 or np.isnan(h):
            continue
        ax.annotate(
            fmt.format(h),
            (bar.get_x() + bar.get_width() / 2, h),
            ha="center", va="bottom", fontsize=fontsize,
            xytext=(0, pad), textcoords="offset points",
        )


# =====================================================================
#  Chart 1 — GEMM Speedup Comparison
# =====================================================================
def chart_gemm_speedup():
    labels = ["bf16\n(torch.matmul)", "PyTorch FP8\n(_scaled_mm)", "DeepGEMM FP8\n(native JIT)"]
    speedups = [1.0, 2.06, None]  # DeepGEMM TBD at runtime
    cuda_us = [1245, 605, None]

    present = [(l, s, t) for l, s, t in zip(labels, speedups, cuda_us) if s is not None]
    labs, spds, times = zip(*present)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labs, spds, color=[COLORS[i] for i in range(len(labs))],
                  edgecolor="white", width=0.55)

    for bar, spd, t in zip(bars, spds, times):
        ax.annotate(
            f"{spd:.2f}x\n({t} \u00b5s)",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center", va="bottom", fontsize=11, fontweight="bold",
            xytext=(0, 5), textcoords="offset points",
        )

    ax.axhline(1.0, color=PAL["grey"], ls="--", lw=0.8)
    ax.set_ylabel("Relative Speedup vs bf16")
    ax.set_title("FP8 GEMM Performance on NVIDIA H200")
    ax.set_ylim(0, max(spds) * 1.35)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))

    fig.tight_layout()
    path = os.path.join(TRACE_DIR, "chart1_gemm_speedup.png")
    fig.savefig(path, dpi=DPI)
    print(f"  saved {path}")
    return fig


# =====================================================================
#  Chart 2 — Attention Backend Comparison
# =====================================================================
def chart_attention():
    backends = ["SDPA math", "SDPA flash\n(FA-2)", "SDPA cuDNN", "FlashAttention-3"]
    cuda_us = [4639, 187, 177, 305]
    kernels = [35, 6, 7, 10]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    bars1 = ax1.bar(backends, cuda_us,
                    color=[COLORS[i] for i in range(4)], edgecolor="white", width=0.6)
    _bar_labels(ax1, fmt="{:.0f}", fontsize=10, pad=4)
    ax1.set_ylabel("CUDA Time (\u00b5s)")
    ax1.set_title("CUDA Time per Forward Pass")

    bars2 = ax2.bar(backends, kernels,
                    color=[COLORS[i] for i in range(4)], edgecolor="white", width=0.6)
    _bar_labels(ax2, fmt="{:.0f}", fontsize=10, pad=4)
    ax2.set_ylabel("Kernel Count")
    ax2.set_title("CUDA Kernel Count")
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    fig.suptitle("Attention Backend Comparison \u2014 TinyLlama on H200",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(TRACE_DIR, "chart2_attention.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"  saved {path}")
    return fig


# =====================================================================
#  Chart 3 — Liger Kernel Fusion Impact
# =====================================================================
def chart_liger():
    groups = ["CUDA Time (\u00b5s)", "Kernel Count", "Speedup"]
    vanilla = [18184, 45, 1.0]
    liger = [15613, 34, 1.16]

    x = np.arange(len(groups))
    w = 0.3

    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - w / 2, vanilla, w, label="Vanilla", color=PAL["blue1"], edgecolor="white")
    b2 = ax.bar(x + w / 2, liger, w, label="Liger", color=PAL["green1"], edgecolor="white")

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            fmt = "{:.0f}" if h > 2 else "{:.2f}x" if "." in str(h) and h < 2 else "{:.0f}"
            ax.annotate(
                fmt.format(h),
                (bar.get_x() + bar.get_width() / 2, h),
                ha="center", va="bottom", fontsize=10, fontweight="bold",
                xytext=(0, 4), textcoords="offset points",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_title("Liger Kernel Fusion Impact on LlamaDecoderLayer")
    ax.legend(frameon=True, framealpha=0.9)
    ax.set_ylabel("Value")

    fig.tight_layout()
    path = os.path.join(TRACE_DIR, "chart3_liger.png")
    fig.savefig(path, dpi=DPI)
    print(f"  saved {path}")
    return fig


# =====================================================================
#  Chart 4 — torch.compile Modes
# =====================================================================
def chart_compile():
    modes = ["eager", "default", "reduce-\noverhead", "max-\nautotune"]
    cuda_us = [18676, 12230, 10196, 10694]
    compile_s = [0.0, 0.71, 0.66, 0.45]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    bars1 = ax1.bar(modes, cuda_us,
                    color=[PAL["grey"]] + [COLORS[i] for i in range(1, 4)],
                    edgecolor="white", width=0.55)
    for bar, val in zip(bars1, cuda_us):
        speedup = cuda_us[0] / val
        ax1.annotate(
            f"{val:,} \u00b5s\n({speedup:.2f}x)",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            xytext=(0, 4), textcoords="offset points",
        )
    ax1.set_ylabel("CUDA Time (\u00b5s)")
    ax1.set_title("Inference CUDA Time")

    bars2 = ax2.bar(modes, compile_s,
                    color=[PAL["grey"]] + [COLORS[i] for i in range(1, 4)],
                    edgecolor="white", width=0.55)
    _bar_labels(ax2, fmt="{:.2f}s", fontsize=10, pad=4)
    ax2.set_ylabel("Compilation Time (s)")
    ax2.set_title("Compilation Overhead")

    fig.suptitle("torch.compile Mode Comparison on H200",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(TRACE_DIR, "chart4_compile.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"  saved {path}")
    return fig


# =====================================================================
#  Chart 5 — NCU Kernel Internals (heatmap)
# =====================================================================
def chart_ncu():
    kernels = ["bf16 matmul\n4096\u00b2", "FP8 GEMM", "Llama QKV\nGEMM",
               "FlashAttn-2", "FlashMLA"]
    metrics = ["SM Busy %", "Compute %", "Memory %", "Occupancy %", "Regs/Thread"]

    data = np.array([
        [95.3, 90.2, 69.6, 14.6, 168],
        [75.0, 68.7, 54.4, 14.7, 168],
        [84.3, 75.1, 58.5, 14.2, 168],
        [44.1, 32.2, 24.1, 11.6, 255],
        [18.6,  8.4, 11.1, 12.2, 240],
    ])

    # Normalise each column to [0,1] for colour mapping
    col_min = data.min(axis=0)
    col_max = data.max(axis=0)
    col_range = np.where(col_max - col_min == 0, 1, col_max - col_min)
    normed = (data - col_min) / col_range

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(normed, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_yticks(np.arange(len(kernels)))
    ax.set_yticklabels(kernels, fontsize=11)

    for i in range(len(kernels)):
        for j in range(len(metrics)):
            val = data[i, j]
            fmt = f"{val:.0f}" if val == int(val) else f"{val:.1f}"
            text_color = "white" if normed[i, j] > 0.6 else "black"
            ax.text(j, i, fmt, ha="center", va="center",
                    fontsize=11, fontweight="bold", color=text_color)

    ax.set_title("Kernel Internals \u2014 Nsight Compute Metrics on H200")
    fig.colorbar(im, ax=ax, label="Normalised intensity", shrink=0.8)

    fig.tight_layout()
    path = os.path.join(TRACE_DIR, "chart5_ncu_heatmap.png")
    fig.savefig(path, dpi=DPI)
    print(f"  saved {path}")
    return fig


# =====================================================================
#  Chart 6 — FlashMLA Performance
# =====================================================================
def chart_flashmla():
    metrics = ["CUDA Time\n(\u00b5s)", "Tokens/sec", "Effective BW\n(GB/s)", "H200 Peak BW\n(GB/s)"]
    values = [93, 43042, 101.5, 4800]
    colors = [PAL["blue1"], PAL["blue2"], PAL["green1"], PAL["grey"]]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel A: CUDA time
    axes[0].bar(["FlashMLA"], [93], color=PAL["blue1"], edgecolor="white", width=0.4)
    axes[0].annotate("93 \u00b5s", (0, 93), ha="center", va="bottom",
                     fontsize=12, fontweight="bold", xytext=(0, 5),
                     textcoords="offset points")
    axes[0].set_ylabel("CUDA Time (\u00b5s)")
    axes[0].set_title("Decode Latency")

    # Panel B: Tokens/sec
    axes[1].bar(["FlashMLA"], [43042], color=PAL["green1"], edgecolor="white", width=0.4)
    axes[1].annotate("43,042", (0, 43042), ha="center", va="bottom",
                     fontsize=12, fontweight="bold", xytext=(0, 5),
                     textcoords="offset points")
    axes[1].set_ylabel("Tokens / sec")
    axes[1].set_title("Throughput (batch=4)")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # Panel C: Bandwidth utilisation
    bw_labels = ["Effective", "H200 Peak"]
    bw_vals = [101.5, 4800]
    bw_colors = [PAL["blue2"], PAL["grey"]]
    bars = axes[2].bar(bw_labels, bw_vals, color=bw_colors, edgecolor="white", width=0.45)
    for bar, val in zip(bars, bw_vals):
        axes[2].annotate(
            f"{val:,.1f}" if val < 1000 else f"{val:,.0f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center", va="bottom", fontsize=11, fontweight="bold",
            xytext=(0, 5), textcoords="offset points",
        )
    util_pct = 101.5 / 4800 * 100
    axes[2].annotate(
        f"Utilisation: {util_pct:.1f}%",
        xy=(0.5, 0.55), xycoords="axes fraction",
        ha="center", fontsize=11, fontstyle="italic", color=PAL["red"],
    )
    axes[2].set_ylabel("Bandwidth (GB/s)")
    axes[2].set_title("Memory Bandwidth")
    axes[2].set_yscale("log")
    axes[2].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    fig.suptitle("FlashMLA Decode Performance on H200",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(TRACE_DIR, "chart6_flashmla.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"  saved {path}")
    return fig


# =====================================================================
#  Combined overview
# =====================================================================
def combined_overview(figs):
    """Re-render all charts in a single tall figure."""
    fig, axes = plt.subplots(3, 2, figsize=(22, 28))
    chart_funcs = [
        chart_gemm_speedup, chart_attention, chart_liger,
        chart_compile, chart_ncu, chart_flashmla,
    ]
    titles = [
        "1. GEMM Speedup", "2. Attention Backends", "3. Liger Fusion",
        "4. torch.compile", "5. NCU Heatmap", "6. FlashMLA",
    ]
    plt.close(fig)

    from matplotlib.image import imread

    chart_files = [
        "chart1_gemm_speedup.png", "chart2_attention.png", "chart3_liger.png",
        "chart4_compile.png", "chart5_ncu_heatmap.png", "chart6_flashmla.png",
    ]

    fig, axes = plt.subplots(3, 2, figsize=(24, 30))
    for idx, (ax, fname, title) in enumerate(zip(axes.flat, chart_files, titles)):
        img_path = os.path.join(TRACE_DIR, fname)
        if os.path.exists(img_path):
            img = imread(img_path)
            ax.imshow(img)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        ax.axis("off")

    fig.suptitle("Torch Profiler Workshop \u2014 Complete Results Dashboard (H200)",
                 fontsize=20, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    path = os.path.join(TRACE_DIR, "combined_overview.png")
    fig.savefig(path, dpi=200)
    print(f"  saved {path}")
    return fig


# =====================================================================
#  Main
# =====================================================================
def main():
    print("Generating profiler-workshop charts …\n")

    figs = []
    figs.append(chart_gemm_speedup())
    figs.append(chart_attention())
    figs.append(chart_liger())
    figs.append(chart_compile())
    figs.append(chart_ncu())
    figs.append(chart_flashmla())

    print()
    combined_overview(figs)

    for f in figs:
        plt.close(f)

    print(f"\nAll charts saved to {TRACE_DIR}/")


if __name__ == "__main__":
    main()
