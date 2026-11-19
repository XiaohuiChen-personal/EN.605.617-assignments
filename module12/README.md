# Module 12 — OpenCL closest word

Xiaohui Chen · EN.605.617.81 · Fall 2026

Each word is a list of numbers. The program stores a table of those lists and one query, and it prints the word closest to the query. Closest means the smallest sum of squared differences.

The table is one buffer. The search covers a slice of the table, passed to the kernel as a sub-buffer, so the slice can start after the first word. The query is a constant buffer. One work-group handles one word. The work-items split the list, loading one float, a float4, or a float8, and add the partial sums in local memory. The barrier sits outside the per-thread branch, so every work-item in the group reaches it. OpenCL C 1.2 provides `dot` through float4, so the float8 kernel adds the low half and the high half. A second kernel keeps the minimum. The host does the same search in C. PASS means the winning index matches that search. The printed absolute error is only that one distance. `main` only calls these steps.

The query is one stored row with 0.05 added to its first number, so that row should win, and the next row should be clearly worse.

## Build and run

Needs gcc and OpenCL. On the DGX Spark the headers live in `~/.local/include` and the loader is `~/.local/lib/libOpenCL.so`. `./build.sh` uses those when `/usr/include/CL/cl.h` is absent.

```bash
./build.sh
./run.sh 65536 0 256 64 4 17 0
```

`make` builds `assignment.exe`. The course runner reads `assignment_config.yaml` at the repository root (`folder: module12`) and runs the five argument lists in that file.

| argv | meaning | default |
| --- | --- | --- |
| 1 | words in the slice | 4096 |
| 2 | first word index | 0 |
| 3 | numbers per word | 256 |
| 4 | work-group size, a power of two | 64 |
| 5 | load width: 1, 4, or 8 | 4 |
| 6 | slice index the query is built from | 0 |
| 7 | 0 copies the result back, 1 maps it | 0 |

A missing trailing argument keeps its default. `dim` has to be a multiple of `vec`. The sub-buffer origin has to be a multiple of the device alignment. Here that alignment is 512 bytes, and 256 floats are 1024 bytes, so a slice may start at any word.

## This run

DGX Spark, NVIDIA GB10. The program printed platform OpenCL 3.0 CUDA 13.0.98 and OpenCL C 1.2. Host compile was `gcc -std=c11 -O2 -Wall -Wextra` with no warnings. Date: 2026-11-19. The distance time is the median of 5 device profiles after 2 warmup launches.

| arguments | median | result |
| --- | --- | --- |
| `65536 0 256 64 4 17 0` | 0.2605 ms | word 17, PASS |
| `1 10 256 64 4 0 0` | 0.0054 ms | word 10, offset 10240 bytes, PASS |
| `65536 32 256 128 1 17 0` | 0.2561 ms | word 49, PASS |
| `65536 0 256 32 8 3 1` | 0.2605 ms | word 3, mapped readback, PASS |
| `65536 64 256 256 4 100 0` | 0.2900 ms | word 164, PASS |

Every transcript prints `abs err: 0.000e+00` for the winning distance. They are in `proof/`.

With the NVIDIA program cache off, the build log is one line per kernel: the compiler may inline it. Nothing in `words.cl` calls a kernel from another kernel, and the build status is success. The distance kernel's work-group limit on this GPU is 256, which is the size of the last run.

At 65536 words, with the group sized to the load, the medians were 0.2561 ms (scalar, group 128), 0.2605 ms (float4, group 64), and 0.2605 ms (float8, group 32). The spread was 0.0044 ms. The other float4 run, group 256, was 0.2900 ms.

The one-page discussion is `discussion.pdf`.
