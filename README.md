# EN.605.617.81 — CUDA Threads and Blocks

Xiaohui Chen · Module 3 assignment · Fall 2026

One program, `module3/assignment.cu`, runs the same Horner-style
iteration on the CPU and on a CUDA kernel:

1. **Branchless** — every element takes the same path (minimal branching).
2. **Divergent** — `if (i & 1)` splits every 32-thread warp.
3. **Warp-uniform** — `if ((i / warpSize) & 1)` so whole warps agree.

GPU output is checked against the CPU. Both sides are timed.

## Deliverables

| Prompt item | File |
| --- | --- |
| CPU + CUDA, minimal branching, and the branching comparison | `module3/assignment.cu` |
| `assignment.exe 512 256`, `make` → `assignment.exe` | `module3/Makefile` |
| Course runner config | `assignment_config.yaml` (`folder: module3`) |
| Build / run scripts | `module3/build.sh`, `module3/run.sh` |
| Performance comparison chart | `figures/fig_cpu_gpu_branching.png` |
| Short text file on the results | `results-thoughts.txt` |
| Previous-year `main` critique (item 5; that code is not included) | `critique-previous-year.md` |

Unedited run transcripts from the DGX Spark are in `proof/`.

## Build and run

Needs an NVIDIA GPU and the CUDA toolkit (`nvcc` on `PATH`, or at
`/usr/local/cuda/bin`).

```bash
./module3/build.sh
./module3/run.sh 512 256          # required invocation
./module3/run.sh                  # defaults: 1<<20 threads, 256 / block
```

Or:

```bash
cd module3
make                              # produces assignment.exe
./assignment.exe 512 256
```

Arguments match the course starter:

| argv | meaning | default |
| --- | --- | --- |
| 1 | total number of threads (= N) | 1,048,576 |
| 2 | threads per block | 256 |

If the total is not a multiple of the block size, the program rounds the
total up and prints a warning.

## Hardware these numbers came from

DGX Spark, NVIDIA GB10, compute capability 12.1, 48 SMs, warp size 32,
CUDA Toolkit 13.0 (`nvcc` V13.0.88), driver 580.159.03. Compiled
`-arch=native`. Date: 2026-09-20.

At N = 1,048,576, 256 threads/block, 1,024 iterations/element, GPU vs
CPU results were bit-identical (`max_abs_err=0`):

| variant | CPU (ms) | GPU kernel (ms) | GPU / branchless |
| --- | ---: | ---: | ---: |
| branchless | 1035.09 | 0.0884 | 1.00× |
| divergent (`i & 1`) | 1035.53 | 0.1707 | 1.93× |
| warp-uniform | 1035.58 | 0.0884 | 1.00× |

The CPU does not care about the branch. The GPU pays almost 2× when the
`if`/`else` splits a warp, and nothing extra when the same `if` is
aligned to `warpSize`.
