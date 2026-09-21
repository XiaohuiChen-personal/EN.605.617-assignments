// EN.605.617.81 Module 3 — CUDA Threads and Blocks
// Xiaohui Chen
//
// Argument handling follows the starter assignment.cu provided in the
// official course repository (JHU-EP-Intro2GPU/EN605.617, module3/):
//   argv[1] = total number of threads  (default 1 << 20)
//   argv[2] = threads per block        (default 256)
//   ./assignment.exe 512 256
//
// Those two numbers are the GPU *launch*, not the data size. The CPU
// does not use them (the prompt: you do not have much control over
// host threading). The array length is at least 1,048,576 so the
// required 512 256 invocation still processes "1000s to millions" of
// elements; extra threads, if requested, enlarge N. The GPU covers N
// with a grid-stride loop.
//
// Part 1: the same branchless Horner-style iteration on CPU and GPU
//         (no data-dependent if/else in that kernel or host loop).
// Part 2: the same two arithmetic paths, selected by a conditional
//         that either splits every warp or is uniform across a warp.

#include "cuda_check.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static const int kMinElements = 1 << 20;  // item 1: thousands to millions
static const int kIters = 1024;
static const int kGpuWarmup = 1;
static const int kGpuReps = 10;
static const int kCpuWarmup = 1;
static const int kCpuReps = 5;

// Shared by host and device so CPU and GPU execute the same arithmetic.
__host__ __device__ __forceinline__ float pathA(float v, int iters)
{
    for (int k = 0; k < iters; ++k)
        v = v * 1.0001f + 1.0f;
    return v;
}

__host__ __device__ __forceinline__ float pathB(float v, int iters)
{
    for (int k = 0; k < iters; ++k)
        v = v * 0.9999f - 1.0f;
    return v;
}

enum Variant {
    kBranchless = 0,
    kDivergent  = 1,
    kUniform    = 2
};

static const char *variantName(Variant v)
{
    switch (v) {
    case kBranchless: return "branchless";
    case kDivergent:  return "divergent (i & 1)";
    case kUniform:    return "warp-uniform (i / warpSize) & 1";
    }
    return "unknown";
}

// Part 1: no data-dependent branch. Grid-stride so a small launch
// (assignment.exe 512 256) still touches every element of a large N.
__global__ void kernelBranchless(const float *x, float *y, int n, int iters)
{
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        y[i] = pathA(x[i], iters);
}

// Part 2: same paths, condition splits every warp (odd/even lanes).
__global__ void kernelDivergent(const float *x, float *y, int n, int iters)
{
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        if (i & 1)
            y[i] = pathA(x[i], iters);
        else
            y[i] = pathB(x[i], iters);
    }
}

// Part 2: same if/else, but the condition is constant on a warp.
__global__ void kernelUniform(const float *x, float *y, int n, int iters)
{
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        if ((i / warpSize) & 1)
            y[i] = pathA(x[i], iters);
        else
            y[i] = pathB(x[i], iters);
    }
}

// CPU counterparts. Launch geometry is not used; we just walk the array.
static void cpuBranchless(const float *x, float *y, int n, int iters)
{
    for (int i = 0; i < n; ++i)
        y[i] = pathA(x[i], iters);
}

static void cpuDivergent(const float *x, float *y, int n, int iters)
{
    for (int i = 0; i < n; ++i) {
        if (i & 1)
            y[i] = pathA(x[i], iters);
        else
            y[i] = pathB(x[i], iters);
    }
}

static void cpuUniform(const float *x, float *y, int n, int iters, int warpSize)
{
    for (int i = 0; i < n; ++i) {
        if ((i / warpSize) & 1)
            y[i] = pathA(x[i], iters);
        else
            y[i] = pathB(x[i], iters);
    }
}

static void cpuWork(const float *x, float *y, int n, int iters, int variant, int warpSize)
{
    if (variant == kBranchless)
        cpuBranchless(x, y, n, iters);
    else if (variant == kDivergent)
        cpuDivergent(x, y, n, iters);
    else
        cpuUniform(x, y, n, iters, warpSize);
}

typedef void (*GpuKern)(const float *, float *, int, int);

static GpuKern gpuKern(Variant v)
{
    if (v == kBranchless)
        return kernelBranchless;
    if (v == kDivergent)
        return kernelDivergent;
    return kernelUniform;
}

static void printDeviceInfo()
{
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    int driver = 0, runtime = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));

    printf("=== Device ===\n");
    printf("name                      : %s\n", p.name);
    printf("compute capability        : %d.%d\n", p.major, p.minor);
    printf("SMs                       : %d\n", p.multiProcessorCount);
    printf("warpSize                  : %d\n", p.warpSize);
    printf("max threads / block       : %d\n", p.maxThreadsPerBlock);
    printf("max threads / SM          : %d\n", p.maxThreadsPerMultiProcessor);
    printf("global memory (bytes)     : %zu\n", p.totalGlobalMem);
    printf("driver / runtime          : %d / %d\n", driver, runtime);
}

struct CheckResult {
    int mismatches;
    float maxAbsErr;
    float sampleCpu;
    float sampleGpu;
};

static CheckResult compare(const float *cpu, const float *gpu, int n)
{
    CheckResult r = {0, 0.0f, 0.0f, 0.0f};
    const float tol = 1e-4f;
    for (int i = 0; i < n; ++i) {
        const float err = fabsf(cpu[i] - gpu[i]);
        if (err > r.maxAbsErr)
            r.maxAbsErr = err;
        if (err > tol)
            ++r.mismatches;
    }
    if (n > 0) {
        r.sampleCpu = cpu[0];
        r.sampleGpu = gpu[0];
    }
    return r;
}

static float timeCpu(const float *x, float *y, int n, int iters, int variant, int warpSize)
{
    cpuWork(x, y, n, iters, variant, warpSize);  // warm-up
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < kCpuReps; ++r)
        cpuWork(x, y, n, iters, variant, warpSize);
    auto t1 = std::chrono::high_resolution_clock::now();
    const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    return static_cast<float>(ns / kCpuReps / 1.0e6);  // ms
}

static float timeGpuKernel(GpuKern k, const float *dX, float *dY, int n, int iters,
                           int blocks, int blockSize)
{
    k<<<blocks, blockSize>>>(dX, dY, n, iters);  // warm-up
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int r = 0; r < kGpuReps; ++r)
        k<<<blocks, blockSize>>>(dX, dY, n, iters);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return ms / kGpuReps;
}

static float timeGpuEndToEnd(GpuKern k, const float *hX, float *hY, float *dX, float *dY,
                             int n, int iters, int blocks, int blockSize)
{
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    k<<<blocks, blockSize>>>(dX, dY, n, iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int r = 0; r < kGpuReps; ++r) {
        CUDA_CHECK(cudaMemcpy(dX, hX, bytes, cudaMemcpyHostToDevice));
        k<<<blocks, blockSize>>>(dX, dY, n, iters);
        CUDA_CHECK(cudaMemcpy(hY, dY, bytes, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return ms / kGpuReps;
}

int main(int argc, char **argv)
{
    // Starter convention: total threads and threads per block. These
    // size the GPU launch only. The CPU loop below never reads them.
    int totalThreads = (1 << 20);
    int blockSize = 256;

    if (argc >= 2)
        totalThreads = atoi(argv[1]);
    if (argc >= 3)
        blockSize = atoi(argv[2]);

    if (totalThreads <= 0) {
        fprintf(stderr, "total thread count must be positive, got %d\n", totalThreads);
        return EXIT_FAILURE;
    }
    if (blockSize <= 0) {
        fprintf(stderr, "block size must be positive, got %d\n", blockSize);
        return EXIT_FAILURE;
    }

    printDeviceInfo();

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (blockSize > prop.maxThreadsPerBlock) {
        fprintf(stderr, "block size %d exceeds device maxThreadsPerBlock %d\n",
                blockSize, prop.maxThreadsPerBlock);
        return EXIT_FAILURE;
    }
    if (blockSize % prop.warpSize != 0) {
        printf("Warning: block size %d is not a multiple of warpSize %d; "
               "the last warp of each block will have idle lanes.\n",
               blockSize, prop.warpSize);
    }

    int numBlocks = totalThreads / blockSize;
    if (totalThreads % blockSize != 0) {
        ++numBlocks;
        totalThreads = numBlocks * blockSize;
        printf("Warning: Total thread count is not evenly divisible by the block size\n");
        printf("The total number of threads will be rounded up to %d\n", totalThreads);
    }

    // Item 1: non-trivial data (1000s to millions) even when the
    // required launch is only 512 threads. Item 3: CPU ignores the
    // launch arguments and always walks this whole array.
    const int n = (totalThreads > kMinElements) ? totalThreads : kMinElements;
    const int iters = kIters;

    printf("\n=== Launch ===\n");
    printf("totalThreads (GPU launch) : %d\n", totalThreads);
    printf("blockSize                 : %d\n", blockSize);
    printf("numBlocks                 : %d\n", numBlocks);
    printf("N (elements, CPU and GPU) : %d\n", n);
    printf("iters / element           : %d\n", iters);
    printf("GPU timing                : %d warm-up + avg of %d (cudaEvent)\n",
           kGpuWarmup, kGpuReps);
    printf("CPU timing                : %d warm-up + avg of %d (chrono)\n",
           kCpuWarmup, kCpuReps);
    printf("command                   : assignment.exe %d %d\n", totalThreads, blockSize);
    if (n > totalThreads)
        printf("note                      : GPU uses a grid-stride loop so %d threads cover N\n",
               totalThreads);

    std::vector<float> hX(n), hYcpu(n), hYgpu(n);
    for (int i = 0; i < n; ++i)
        hX[i] = 0.001f * static_cast<float>(i % 1000) + 0.1f;

    float *dX = nullptr, *dY = nullptr;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&dX, bytes));
    CUDA_CHECK(cudaMalloc(&dY, bytes));
    CUDA_CHECK(cudaMemcpy(dX, hX.data(), bytes, cudaMemcpyHostToDevice));

    printf("\n=== Correctness (GPU vs CPU, tol 1e-4) ===\n");
    int failed = 0;
    const Variant variants[] = {kBranchless, kDivergent, kUniform};
    for (Variant v : variants) {
        cpuWork(hX.data(), hYcpu.data(), n, iters, v, prop.warpSize);
        gpuKern(v)<<<numBlocks, blockSize>>>(dX, dY, n, iters);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hYgpu.data(), dY, bytes, cudaMemcpyDeviceToHost));
        const CheckResult chk = compare(hYcpu.data(), hYgpu.data(), n);
        const char *status = chk.mismatches == 0 ? "PASS" : "FAIL";
        if (chk.mismatches != 0)
            ++failed;
        printf("%-36s : %s  mismatches=%d  max_abs_err=%.6g  y[0] cpu=%.6f gpu=%.6f\n",
               variantName(v), status, chk.mismatches, chk.maxAbsErr,
               chk.sampleCpu, chk.sampleGpu);
    }

    const float expect0 = pathB(hX[0], iters);
    const float expect1 = pathA(hX[1], iters);
    cpuWork(hX.data(), hYcpu.data(), n, iters, kDivergent, prop.warpSize);
    const float handErr0 = fabsf(hYcpu[0] - expect0);
    const float handErr1 = fabsf(hYcpu[1] - expect1);
    const char *handStatus = (handErr0 <= 1e-6f && handErr1 <= 1e-6f) ? "PASS" : "FAIL";
    if (handErr0 > 1e-6f || handErr1 > 1e-6f)
        ++failed;
    printf("hand check x[0]->pathB, x[1]->pathA : %s  err0=%.6g err1=%.6g\n",
           handStatus, handErr0, handErr1);

    printf("\n=== Timing (milliseconds; GPU kernel excludes H2D/D2H) ===\n");
    printf("%-36s  %12s  %14s  %12s  %10s\n",
           "variant", "cpu_ms", "gpu_kernel_ms", "gpu_e2e_ms", "cpu/gpu");
    printf("%-36s  %12s  %14s  %12s  %10s\n",
           "-------", "------", "-------------", "----------", "-------");

    FILE *csv = fopen("results.csv", "w");
    if (csv) {
        fprintf(csv, "variant,n,block_size,num_blocks,iters,cpu_ms,gpu_kernel_ms,gpu_e2e_ms,speedup\n");
    }

    for (Variant v : variants) {
        const GpuKern k = gpuKern(v);
        const float cpuMs = timeCpu(hX.data(), hYcpu.data(), n, iters, v, prop.warpSize);
        const float gpuMs = timeGpuKernel(k, dX, dY, n, iters, numBlocks, blockSize);
        const float e2eMs = timeGpuEndToEnd(k, hX.data(), hYgpu.data(), dX, dY,
                                            n, iters, numBlocks, blockSize);
        const float speedup = gpuMs > 0.0f ? cpuMs / gpuMs : 0.0f;
        printf("%-36s  %12.4f  %14.4f  %12.4f  %10.2fx\n",
               variantName(v), cpuMs, gpuMs, e2eMs, speedup);
        if (csv) {
            fprintf(csv, "\"%s\",%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f\n",
                    variantName(v), n, blockSize, numBlocks, iters,
                    cpuMs, gpuMs, e2eMs, speedup);
        }
    }
    if (csv)
        fclose(csv);

    CUDA_CHECK(cudaFree(dX));
    CUDA_CHECK(cudaFree(dY));

    if (failed) {
        fprintf(stderr, "\nCorrectness FAILED (%d check group(s)).\n", failed);
        return EXIT_FAILURE;
    }
    printf("\nAll correctness checks passed.\n");
    return EXIT_SUCCESS;
}
