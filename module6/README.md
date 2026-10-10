# Module 6 — CUDA streams and events

Xiaohui Chen · EN.605.617.81 · Fall 2026

`assignment.cu` launches one kernel, `mapQuadratic`, on two arrays of length N. Both launches call that kernel. Each array is copied to the device, the kernel runs, and the result is copied back. Host memory is page-locked (`cudaMallocHost`). Copies are `cudaMemcpyAsync`. Streams are created with `cudaStreamNonBlocking`.

The kernel takes 1024 steps of `y = min(0.15 y² + 0.20 y + 0.30, 4)`, using `fmaf` and `fminf`. The host check calls the same function. The two arrays are filled with different values (`(i + salt) % 97 * 0.01`, salts 0 and 17).

| Mode | Schedule |
| --- | --- |
| `one_stream` | Both arrays queued on one stream. |
| `two_streams` | One stream per array. Neither stream waits on the other. |
| `wait_event` | Two streams. Stream 1's kernel waits on the event recorded at the end of stream 0's kernel. Stream 1's copy-in is queued before that wait. |

`k_overlap` is how many of the 5 timed reps had overlapping kernel intervals: A's start before B's end, and B's start before A's end. `wall_ms` is the mean of those 5 reps after 1 warmup, from the earlier copy-in event to the later copy-out event. `k0_ms` and `k1_ms` are the mean kernel durations.

| argv | meaning | default |
| --- | --- | --- |
| 1 | N, elements in each array | 1048576 |
| 2 | threads per block | 256 |

## Build and run

Needs an NVIDIA GPU and `nvcc` (`PATH`, or `/usr/local/cuda/bin`). From this directory:

```bash
./build.sh
./run.sh 1048576 256
```

`make` alone also produces `assignment.exe`.

The course runner reads `assignment_config.yaml` at the repository root (`folder: module6`) and calls `build.sh`, then `run.sh` once per row:

```bash
./run.sh 262144 256
./run.sh 1048576 256
./run.sh 4194304 256
./run.sh 16777216 256
```

## This run

DGX Spark, NVIDIA GB10, compute capability 12.1, 48 SMs, warp size 32. The program printed `concurrentKernels 1`, `asyncEngineCount 1`, `integrated 1`, driver / runtime 13000 / 13000. `nvcc` V13.0.88 (`proof/nvcc.txt`), compiled `-arch=native -O2`. Date: 2026-10-10. Transcripts are in `proof/`.

The check covers `i < 64`, indices at or above 4097 that are multiples of 4097, and `i == N-1`, on both arrays. It does not visit every index. Every row in the table printed `max_abs_err` 0 and `PASS`.

| N | mode | wall_ms | k0_ms | k1_ms | k_overlap |
| --- | --- | ---: | ---: | ---: | --- |
| 262144 | one_stream | 0.2513 | 0.0851 | 0.0819 | 0/5 |
| 262144 | two_streams | 0.2044 | 0.0882 | 0.1370 | 5/5 |
| 262144 | wait_event | 0.2187 | 0.0886 | 0.0816 | 0/5 |
| 1048576 | one_stream | 0.8908 | 0.3016 | 0.3047 | 0/5 |
| 1048576 | two_streams | 0.7364 | 0.3045 | 0.5219 | 5/5 |
| 1048576 | wait_event | 0.7584 | 0.3044 | 0.3060 | 0/5 |
| 4194304 | one_stream | 3.4900 | 1.1724 | 1.1723 | 0/5 |
| 4194304 | two_streams | 2.9136 | 1.1736 | 2.0452 | 5/5 |
| 4194304 | wait_event | 2.9559 | 1.1823 | 1.1828 | 0/5 |
| 16777216 | one_stream | 14.1724 | 4.7170 | 4.6936 | 0/5 |
| 16777216 | two_streams | 11.7602 | 4.7011 | 8.2489 | 5/5 |
| 16777216 | wait_event | 11.7809 | 4.7311 | 4.7118 | 0/5 |

Ratios printed by the program, `two_streams / one_stream` and `wait_event / one_stream`:

| N | two_streams / one_stream | wait_event / one_stream |
| --- | ---: | ---: |
| 262144 | 0.813 | 0.870 |
| 1048576 | 0.827 | 0.851 |
| 4194304 | 0.835 | 0.847 |
| 16777216 | 0.830 | 0.831 |

On `two_streams`, `k_overlap` was 5/5 at each N. On `one_stream` and `wait_event` it was 0/5. At N = 16777216, `k1_ms` is 8.2489 on `two_streams` and 4.6936 on `one_stream`. The walls at that N are 11.7602 ms and 14.1724 ms.

The last rep at N = 16777216, as printed:

```text
one_stream    A 1.1384..5.8298   B 8.0261..12.7182
two_streams   A 1.4199..6.1755   B 2.6310..10.9121
wait_event    A 1.1408..5.8868   B 5.8892..10.6765
```

On that `one_stream` line, B starts at 8.0261 and A ends at 5.8298 (8.0261 - 5.8298 = 2.1963 ms). On that `wait_event` line, B starts at 5.8892 and A ends at 5.8868 (5.8892 - 5.8868 = 0.0024 ms). The other three transcripts print the same kind of line.

`proof/run_1000_128.txt` is an extra launch with N = 1000 and block size 128. It printed `PASS`. It is not in `assignment_config.yaml`.

These milliseconds are this run, copied from `proof/`.
