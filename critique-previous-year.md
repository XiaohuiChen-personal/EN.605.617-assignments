# Previous-year `main` — short critique

Xiaohui Chen · EN.605.617.81 · Module 3, item 5

The prompt asks: if the surrounding script only executes this `main`
once, what is good and bad about it? I am not fixing or submitting that
code.

**What is good.** It prints card info, it has both a GPU kernel and a
CPU counterpart, it accepts a launch configuration on the command line,
and it prints both times. Those are the right pieces.

**What is bad, especially on a single run.**

1. **The GPU clock is not measuring the kernel.** `add<<<blocks,threads>>>`
   returns when the launch is *queued*, not when it finishes. There is
   no `cudaDeviceSynchronize` (or `cudaEventSynchronize`) before
   `stop`. On one execution the “GPU” time is launch overhead — a few
   microseconds — and the speedup number is fiction. That is the
   mistake that makes a single-run script most misleading.

2. **The argument convention is not this year’s.** `argv[1]` is a
   block count and `argv[2]` is threads per block. This assignment’s
   required `assignment.exe 512 256` means *512 total threads* and
   *256 per block*. The same command on last year’s `main` would launch
   512 × 256 = 131,072 threads and still process whatever compile-time
   `N` happens to be.

3. **`N` and the launch do not have to agree.** The arrays have length
   `N`. The grid has `blocks * threads` threads. One run with the
   defaults (3 blocks × 64 threads = 192 threads) either leaves most of
   `N` untouched or, if `N` is small and on the stack (`int a[N]`),
   cannot hold the “thousands to millions” the prompt asked for.

4. **No error checks.** A failing `cudaMalloc` or a bad launch is
   silent. On a one-shot script that looks like a successful run.

5. **The CPU pass destroys the evidence.** `addHost(a, b, c)` writes
   into `c` after `c` was copied back from the device, so a single run
   cannot compare GPU output to CPU output.

6. **No warmup, one sample.** The first launch includes extra driver
   work. One number is not a measurement.

The one-execution constraint is what turns these from “rough lab code”
into a bad submission: there is no second run to notice that the GPU
time is implausibly small, no check that `c` is right, and a default
grid of 192 threads that will not keep a modern GPU busy.
