#!/usr/bin/env python3
"""Bar charts from the DGX Spark runs of 2026-09-20. Numbers are copied
from proof/run_1048576.txt and proof/run_4194304.txt, not invented."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "fig_cpu_gpu_branching.png"

# gpu_kernel_ms, cpu_ms
n1 = {
    "label": "N = 1,048,576",
    "gpu": [0.0883, 0.1722, 0.0885],
    "cpu": [1023.0758, 1021.6631, 1023.6998],
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
