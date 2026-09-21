#!/usr/bin/env python3
"""Charts from the DGX Spark runs of 2026-09-20. Numbers are copied
from the proof/ transcripts, not invented."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "fig_cpu_gpu_branching.png"
OUT_LAUNCH = HERE / "fig_launch_configs.png"

# gpu_kernel_ms, cpu_ms
n1 = {
    "label": "N = 1,048,576",
    "gpu": [0.0886, 0.1723, 0.0883],
    "cpu": [1035.5934, 1034.1339, 1036.0161],
}
n4 = {
    "label": "N = 4,194,304",
    "gpu": [0.3410, 0.6743, 0.3443],
    "cpu": [4144.4995, 4139.3271, 4145.7388],
}
names = ["Branchless\n(path A only)", "Divergent\n(i & 1)", "Warp-uniform\n(i / warpSize) & 1"]
x = np.arange(len(names))
w = 0.36

fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), layout="constrained")

# Left: GPU kernel time — this is where divergence appears.
ax = axes[0]
b1 = ax.bar(x - w / 2, n1["gpu"], w, label=n1["label"], color="#0f4c81")
b2 = ax.bar(x + w / 2, n4["gpu"], w, label=n4["label"], color="#7eb6e0")
ax.set_ylabel("GPU kernel time (ms)")
ax.set_xticks(x, names)
ax.set_title("GPU: intra-warp split costs ~2×")
ax.legend(frameon=False, fontsize=8)
ax.set_ylim(0, 0.85)
for bars in (b1, b2):
    for rect in bars:
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.015,
            f"{rect.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
# Annotate the 2x relationship on the 4M series.
ax.annotate(
    "0.674 / 0.341 = 1.98×",
    xy=(1 + w / 2, 0.674),
    xytext=(1.55, 0.78),
    fontsize=8,
    color="#0f4c81",
    arrowprops=dict(arrowstyle="->", color="#0f4c81", lw=0.8),
)

# Right: CPU time — same branch does almost nothing.
ax = axes[1]
b1 = ax.bar(x - w / 2, n1["cpu"], w, label=n1["label"], color="#57534e")
b2 = ax.bar(x + w / 2, n4["cpu"], w, label=n4["label"], color="#a8a29e")
ax.set_ylabel("CPU time (ms)")
ax.set_xticks(x, names)
ax.set_title("CPU: the same if/else is essentially free")
ax.legend(frameon=False, fontsize=8)
ax.set_ylim(0, 5000)
for bars in (b1, b2):
    for rect in bars:
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 60,
            f"{rect.get_height():.0f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

fig.suptitle(
    "EN.605.617 Module 3 — CPU vs GPU, branchless vs divergent vs warp-uniform\n"
    "DGX Spark, NVIDIA GB10, CUDA 13.0; 1024 iterations/element; block size 256",
    fontsize=10,
)
fig.savefig(OUT, dpi=160)
print(f"wrote {OUT}")

# Second figure: the five CLI launches the rubric asked to capture.
# Same N = 1,048,576; only the launch (thread count, block size) changes.
# Source: proof/run_512_256.txt, run_65536_256.txt, run_1048576_{256,128,64}.txt
thread_labels = ["512\n× 256", "65,536\n× 256", "1,048,576\n× 256"]
thread_gpu = [4.0433, 0.1069, 0.0886]
block_labels = ["256 / block\n(4,096 blocks)", "128 / block\n(8,192 blocks)", "64 / block\n(16,384 blocks)"]
block_gpu = [0.0886, 0.0883, 0.0884]

fig2, axes2 = plt.subplots(1, 2, figsize=(9.2, 4.2), layout="constrained")
ax = axes2[0]
bars = ax.bar(thread_labels, thread_gpu, color="#0f4c81")
ax.set_ylabel("GPU kernel time, branchless (ms)")
ax.set_title("Extra thread counts (block size 256)")
ax.set_ylim(0, 4.8)
for rect in bars:
    ax.text(
        rect.get_x() + rect.get_width() / 2,
        rect.get_height() + 0.08,
        f"{rect.get_height():.3f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )
ax = axes2[1]
bars = ax.bar(block_labels, block_gpu, color="#7eb6e0")
ax.set_ylabel("GPU kernel time, branchless (ms)")
ax.set_title("Extra block sizes (1,048,576 threads)")
ax.set_ylim(0, 0.14)
for rect in bars:
    ax.text(
        rect.get_x() + rect.get_width() / 2,
        rect.get_height() + 0.003,
        f"{rect.get_height():.4f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )
fig2.suptitle(
    "EN.605.617 Module 3 — five command-line launches, same N = 1,048,576\n"
    "DGX Spark, NVIDIA GB10, CUDA 13.0; captured 20 September 2026",
    fontsize=10,
)
fig2.savefig(OUT_LAUNCH, dpi=160)
print(f"wrote {OUT_LAUNCH}")
