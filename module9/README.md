# Module 9 — CUDA advanced libraries

Xiaohui Chen · EN.605.617.81 · Fall 2026

`assignment.cu` names a one-second clip as 440 Hz, 1000 Hz, or noise. The clips are generated in the program (four of each, seed from argv). cuFFT builds a Hann-windowed magnitude spectrogram. cuDNN runs a convolution, a ReLU, a max-pool, and a softmax. The three filters are fixed: bins 13–15, bins 31–33, and 0.2 on every bin. Each frame is scaled so its bins sum to 1. The 1000 Hz clips use amplitude 0.4.

The one-page note is in the course repo, `assignments/assignment9/writeup.pdf`.

| argv | meaning | default |
| --- | --- | --- |
| 1 | RNG seed | 1 |

## Build and run

Needs an NVIDIA GPU, `nvcc`, and cuDNN. The CUDA 13.0 toolkit on the DGX Spark does not ship cuDNN. `build.sh` uses `CUDNN_ROOT` if it is set, otherwise the Module 9 wheel at `~/tmp/en605617/m9/venv/lib/python3.*/site-packages/nvidia/cudnn`.

```bash
./build.sh
./run.sh 1
```

`make` needs `CUDNN_ROOT` on the command line. `./build.sh` sets it.

The course runner reads `assignment_config.yaml` at the repository root (`folder: module9`) and calls `build.sh`, then `run.sh 1`.

## This run

DGX Spark, NVIDIA GB10, compute capability 12.1, driver 580.159.03. `nvcc` V13.0.88 (`proof/nvcc.txt`). cuDNN 9.27.0, header 92700. Compiled `-arch=native -O2`. Date: 2026-10-31. Transcript: `proof/run.txt`.

The host check is a dot product of the same filters against the spectrogram, compared with the cuDNN convolution before ReLU. The run printed `max_abs_err 0.000e+00`, `correct 12/12`, and `PASS`. DIRECT, WINOGRAD, and WINOGRAD_NONFUSED returned `CUDNN_STATUS_NOT_SUPPORTED`. The program kept IMPLICIT_GEMM, which reported 0 workspace bytes.

On the noise clips the winning probability was about 0.37. The first noise row printed 0.3185, 0.3157, 0.3657.
