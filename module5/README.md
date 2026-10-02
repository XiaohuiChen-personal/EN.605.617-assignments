# Module 5 — CUDA memory

Xiaohui Chen · EN.605.617.81 · Fall 2026

One program, `assignment.cu`, evaluates

```text
y[i] = (...((w[0] * x[i] + w[1]) * x[i] + w[2]) ...) * x[i] + w[255]
```

with `w[k] = 1/(k+1)` and `x[i] = (i % 100) * 0.01`. The Horner loop is the same in three kernels. What changes is where the 256 weights live.

| Kernel | Weights |
| --- | --- |
| `kernelGlobal` | global array `dW` |
| `kernelConstant` | `__constant__ cW`, filled with `cudaMemcpyToSymbol` |
| `kernelShared` | `__shared__ sW[256]`, loaded from `dW` once per block |

`hX`, `hY`, and `hW` are host arrays. `dX` and `dY` are the global-memory copies. `float xi` and `float h` are per-thread registers. The program prints each kernel's register count and local-memory size from `cudaFuncGetAttributes`. On this build every kernel reports 0 bytes of local memory, so those values were not spilled.

GPU output is checked against the same Horner loop on the host.

## Build and run

Needs an NVIDIA GPU and `nvcc` (`PATH`, or `/usr/local/cuda/bin`). From this directory:

```bash
./build.sh
./run.sh 512 256
./run.sh 65536 256
./run.sh 1048576 256
./run.sh 1048576 128
./run.sh 1048576 64
```

`make` alone also produces `assignment.exe`.

| argv | meaning | default |
| --- | --- | --- |
| 1 | total threads in the launch | 1,048,576 |
| 2 | threads per block | 256 |

`numBlocks = totalThreads / blockSize`, rounded up when the division is not exact (same warning as the course starter). N is at least 4,194,304. A launch smaller than that still covers every element with a grid-stride loop.

The course runner reads `assignment_config.yaml` at the repository root (`folder: module5`) and calls `build.sh`, then `run.sh` once per row above.

## Hardware and this run

DGX Spark, NVIDIA GB10, compute capability 12.1, 48 SMs, warp size 32, CUDA Toolkit 13.0 (`nvcc` V13.0.88), driver / runtime 13000 / 13000. Compiled `-arch=native -O2`. Date: 2026-10-01. Transcripts are in `proof/`.

Every launch printed `PASS` with `max_abs_err=0`. N = 4,194,304, K = 256. Times are the mean of 10 launches after one warmup, kernel only.

| command | threads | block | blocks | global (ms) | constant (ms) | shared (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `512 256` | 512 | 256 | 2 | 14.3577 | 10.2531 | 9.1360 |
| `65536 256` | 65,536 | 256 | 256 | 0.6204 | 0.3267 | 0.2092 |
| `1048576 256` | 1,048,576 | 256 | 4,096 | 0.5532 | 0.2848 | 0.1744 |
| `1048576 128` | 1,048,576 | 128 | 8,192 | 0.5496 | 0.3070 | 0.1816 |
| `1048576 64` | 1,048,576 | 64 | 16,384 | 0.5490 | 0.2834 | 0.1736 |

With the GPU occupied (1,048,576 threads), constant weights took about half the global-weight time, and shared weights took about a third. All three kernels do the same arithmetic; every thread reads `w[k]` at the same `k`. Register counts on this build: global 38, constant 20, shared 25, and 1024 bytes of shared memory in `kernelShared`.

`512 256` is only 2 blocks on a 48-SM GPU. The global-weight time there is 14.3577 ms, against 0.5532 ms for `1048576 256`, for the same N. At 1,048,576 threads, changing the block size among 64, 128, and 256 moves the global-weight time between 0.5490 ms and 0.5532 ms.
