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
| Performance comparison charts | `figures/fig_cpu_gpu_branching.png`, `figures/fig_launch_configs.png` |
| Short text file on the results | `results-thoughts.txt` |
| Previous-year `main` critique (item 5; that code is not included) | `critique-previous-year.md` |
| Written report (cover sheet, assumptions, captioned figure, references) | `writeup.html` |

Unedited run transcripts from the DGX Spark are in `proof/`.

## Build and run

Needs an NVIDIA GPU and the CUDA toolkit (`nvcc` on `PATH`, or at
`/usr/local/cuda/bin`).

```bash
./module3/build.sh
./module3/run.sh 512 256          # required invocation
./module3/run.sh 65536 256        # additional thread count
./module3/run.sh 1048576 256      # additional thread count
./module3/run.sh 1048576 128      # additional block size
./module3/run.sh 1048576 64       # additional block size (>= 64)
```

Those five launches are also listed in `assignment_config.yaml` so the
course Linux runner captures every run. Thread count and block size
both come from command-line arguments, not hard-coded launches.

Or:

```bash
cd module3
make                              # produces assignment.exe
./assignment.exe 512 256
```

Arguments match the course starter:

| argv | meaning | default |
| --- | --- | --- |
| 1 | GPU thread count (launch only) | 1,048,576 |
| 2 | threads per block | 256 |

The CPU does not use those arguments. N is at least 1,048,576 so
`assignment.exe 512 256` still processes a million elements (grid-stride
loop). If the total thread count is not a multiple of the block size,
the program rounds the launch up and prints a warning, same as the
starter.

## Hardware these numbers came from

DGX Spark, NVIDIA GB10, compute capability 12.1, 48 SMs, warp size 32,
CUDA Toolkit 13.0 (`nvcc` V13.0.88), driver 580.159.03. Compiled
`-arch=native`. Date: 2026-09-20.

Every listed launch printed `PASS` with `max_abs_err=0`. At
N = 1,048,576, 1,024 iterations/element:

| command | threads | block | GPU branchless (ms) | GPU divergent (ms) | div / branchless |
| --- | ---: | ---: | ---: | ---: | ---: |
| `512 256` | 512 | 256 | 4.0433 | 7.9973 | 1.98× |
| `65536 256` | 65,536 | 256 | 0.1069 | 0.2077 | 1.94× |
| `1048576 256` | 1,048,576 | 256 | 0.0886 | 0.1723 | 1.95× |
| `1048576 128` | 1,048,576 | 128 | 0.0883 | 0.1718 | 1.95× |
| `1048576 64` | 1,048,576 | 64 | 0.0884 | 0.1722 | 1.95× |

The CPU does not care about the branch (about 1.03 s on every row). The
GPU pays almost 2× when the `if`/`else` splits a warp, and nothing extra
when the same `if` is aligned to `warpSize`. Extra threads (left three
rows) occupy more SMs; extra block sizes of 128 and 64 (right two rows)
match 256 on this kernel once the grid is large.
