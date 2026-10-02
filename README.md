# EN.605.617.81 — Assignment submissions

Xiaohui Chen · Fall 2026

One repository for the graded programming assignments in this course.
Each assignment is a `moduleN/` directory. The course runner reads
`assignment_config.yaml` in this root and builds the directory named by
`folder`.

## Layout

| Path | Role |
| --- | --- |
| `assignment_config.yaml` | Runner config. `folder` is the assignment it builds and runs. Currently `module5`. |
| `module3/` | Module 3 — CUDA Threads and Blocks. Submitted. |
| `module5/` | Module 5 — CUDA memory. |
| `moduleN/` | A later assignment. Same shape: `build.sh`, `run.sh`, sources, and that assignment's notes. |

`assignment.cu` for Module 3 stays in `module3/`, which is what that
assignment prompt requires. Each assignment's `build.sh` and `run.sh`
live in its own `moduleN/` directory. The runner follows `folder` in
`assignment_config.yaml`, which is `module5`.

To add a later assignment:

1. Create `moduleN/` with `build.sh`, `run.sh`, and the sources.
2. When that assignment is the one the runner should execute, set
   `folder:` in `assignment_config.yaml` to `moduleN` and push `main`.
3. Leave earlier `moduleN/` directories in the tree.

## Module 3 — CUDA Threads and Blocks

`module3/assignment.cu` runs the same Horner-style iteration on the CPU
and on a CUDA kernel:

1. **Branchless** — every element takes the same path.
2. **Divergent** — `if (i & 1)` splits every 32-thread warp.
3. **Warp-uniform** — `if ((i / warpSize) & 1)` so whole warps agree.

GPU output is checked against the CPU. Both sides are timed.

| Prompt item | File |
| --- | --- |
| CPU + CUDA, minimal branching, and the branching comparison | `module3/assignment.cu` |
| `assignment.exe 512 256`, `make` → `assignment.exe` | `module3/Makefile` |
| Course runner config | `assignment_config.yaml` (was `folder: module3`; the file now points at `module5`) |
| Build / run scripts | `module3/build.sh`, `module3/run.sh` |
| Performance comparison charts | `module3/figures/fig_cpu_gpu_branching.png`, `module3/figures/fig_launch_configs.png` |
| Short text file on the results | `module3/results-thoughts.txt` |
| Previous-year `main` critique (item 5; that code is not included) | `module3/critique-previous-year.md` |
| Written report | `module3/writeup.html` |

Unedited run transcripts from the DGX Spark are in `module3/proof/`.

### Build and run

Needs an NVIDIA GPU and the CUDA toolkit (`nvcc` on `PATH`, or at
`/usr/local/cuda/bin`). From this repository root:

```bash
./module3/build.sh
./module3/run.sh 512 256          # required invocation
./module3/run.sh 65536 256        # additional thread count
./module3/run.sh 1048576 256      # additional thread count
./module3/run.sh 1048576 128      # additional block size
./module3/run.sh 1048576 64       # additional block size (>= 64)
```

Thread count and block size both come from command-line arguments.
Those five launches were the `run:` list while `folder` was `module3`.

Or:

```bash
cd module3
make                              # produces assignment.exe
./assignment.exe 512 256
```

| argv | meaning | default |
| --- | --- | --- |
| 1 | GPU thread count (launch only) | 1,048,576 |
| 2 | threads per block | 256 |

The CPU does not use those arguments. N is at least 1,048,576, so
`assignment.exe 512 256` still processes a million elements (grid-stride
loop). If the total thread count is not a multiple of the block size,
the program rounds the launch up and prints a warning.

### Hardware these numbers came from

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
when the same `if` is aligned to `warpSize`. The discussion is in
`module3/results-thoughts.txt` and `module3/writeup.html`.

## Module 5 — CUDA memory

`module5/assignment.cu` runs one Horner polynomial (256 weights) three
ways: weights in global memory, in `__constant__` memory, and in
`__shared__` memory. Host arrays hold `x`, `y`, and the weights.
`float xi` and the accumulator `float h` are registers (0 bytes of
local memory on this build). Thread count and block size come from
argv. Details, the timing table, and build steps are in
`module5/README.md`. Transcripts are in `module5/proof/`.

```bash
./module5/build.sh
./module5/run.sh 1048576 256
```
