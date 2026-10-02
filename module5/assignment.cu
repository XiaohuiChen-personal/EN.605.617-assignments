// EN.605.617.81 Module 5 — CUDA memory
// Xiaohui Chen
//
// y[i] = Horner(x[i], w[0..K)) with the same loop in three kernels.
// Only the weight array changes: global, __constant__, or __shared__.
// x and y are host arrays, copied to global memory. float xi and the
// Horner accumulator float h live in registers; each launch prints the
// register count and the local-memory size (0 means they did not spill).
//
// argv matches the course starter (module5/register_memory/assignment.c):
//   argv[1] = total threads      (default 1<<20)
//   argv[2] = threads per block  (default 256)
//   ./assignment.exe 512 256
// numBlocks = totalThreads / blockSize, rounded up when it does not divide.
// N is at least 4,194,304. A smaller launch covers N with a grid-stride loop.

#include "cuda_check.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static const int K = 256;
static const int kMinElements = 1 << 22;
static const int kDefaultThreads = 1 << 20;
static const int kDefaultBlock = 256;
static const int kWarmup = 1;
static const int kReps = 10;
static const float kTol = 1e-4f;

__constant__ float cW[K];

__host__ __device__ __forceinline__ float horner(float x, const float *w)
{
    float h = 0.f;
    for (int k = 0; k < K; ++k)
        h = h * x + w[k];
    return h;
}

__global__ void kernelGlobal(const float *x, float *y, int n, const float *w)
{
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        float xi = x[i];
        float h = 0.f;
#pragma unroll 16
        for (int k = 0; k < K; ++k)
            h = h * xi + w[k];
        y[i] = h;
    }
}

__global__ void kernelConstant(const float *x, float *y, int n, const float * /*w*/)
{
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        float xi = x[i];
        float h = 0.f;
#pragma unroll 16
        for (int k = 0; k < K; ++k)
            h = h * xi + cW[k];
        y[i] = h;
    }
}

__global__ void kernelShared(const float *x, float *y, int n, const float *w)
{
    __shared__ float sW[K];
    for (int k = threadIdx.x; k < K; k += blockDim.x)
        sW[k] = w[k];
    __syncthreads();

    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        float xi = x[i];
        float h = 0.f;
#pragma unroll 16
        for (int k = 0; k < K; ++k)
            h = h * xi + sW[k];
        y[i] = h;
    }
}

static void cpuEval(const float *x, float *y, int n, const float *w)
{
    for (int i = 0; i < n; ++i)
        y[i] = horner(x[i], w);
}

static int check(const float *ref, const float *got, int n, float *maxErr)
{
    int bad = 0;
    float worst = 0.f;
    for (int i = 0; i < n; ++i) {
        float e = fabsf(ref[i] - got[i]);
        if (e > worst)
            worst = e;
        if (e > kTol)
            ++bad;
    }
    *maxErr = worst;
    return bad;
}

static float timeLaunch(void (*fn)(const float *, float *, int, const float *),
                        bool passW, const float *dX, float *dY, int n, const float *dW,
                        int blocks, int blockSize)
{
    auto launch = [&]() {
        if (passW)
            fn<<<blocks, blockSize>>>(dX, dY, n, dW);
        else
            fn<<<blocks, blockSize>>>(dX, dY, n, nullptr);
    };
    for (int w = 0; w < kWarmup; ++w)
        launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int r = 0; r < kReps; ++r)
        launch();
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaGetLastError());
    return ms / kReps;
}

static void printFn(const char *name, const void *fn)
{
    cudaFuncAttributes a;
    CUDA_CHECK(cudaFuncGetAttributes(&a, fn));
    printf("  %-18s regs=%3d  local=%4zu B  shared=%4zu B\n",
           name, a.numRegs, a.localSizeBytes, a.sharedSizeBytes);
}

int main(int argc, char **argv)
{
    int totalThreads = kDefaultThreads;
    int blockSize = kDefaultBlock;
    if (argc >= 2)
        totalThreads = atoi(argv[1]);
    if (argc >= 3)
        blockSize = atoi(argv[2]);
    if (totalThreads <= 0 || blockSize <= 0) {
        fprintf(stderr, "usage: assignment.exe <totalThreads> <blockSize>\n");
        return EXIT_FAILURE;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int driver = 0, runtime = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    printf("=== Device ===\n");
    printf("name                  : %s\n", prop.name);
    printf("compute capability    : %d.%d\n", prop.major, prop.minor);
    printf("SMs                   : %d\n", prop.multiProcessorCount);
    printf("warpSize              : %d\n", prop.warpSize);
    printf("global memory (bytes) : %zu\n", prop.totalGlobalMem);
    printf("driver / runtime      : %d / %d\n", driver, runtime);

    if (blockSize > prop.maxThreadsPerBlock) {
        fprintf(stderr, "block size %d exceeds maxThreadsPerBlock %d\n",
                blockSize, prop.maxThreadsPerBlock);
        return EXIT_FAILURE;
    }
    if (blockSize < prop.warpSize)
        printf("Warning: block size %d is below warpSize %d\n", blockSize, prop.warpSize);

    int numBlocks = totalThreads / blockSize;
    if (totalThreads % blockSize != 0) {
        ++numBlocks;
        totalThreads = numBlocks * blockSize;
        printf("Warning: Total thread count is not evenly divisible by the block size\n");
        printf("The total number of threads will be rounded up to %d\n", totalThreads);
    }

    const int n = (totalThreads > kMinElements) ? totalThreads : kMinElements;

    printf("\n=== Launch ===\n");
    printf("command               : assignment.exe %d %d\n", totalThreads, blockSize);
    printf("totalThreads          : %d\n", totalThreads);
    printf("blockSize             : %d\n", blockSize);
    printf("numBlocks             : %d\n", numBlocks);
    printf("N                     : %d\n", n);
    printf("K                     : %d\n", K);
    if (n > totalThreads)
        printf("note                  : grid-stride, %d threads cover N\n", totalThreads);

    printf("\n=== Memories ===\n");
    printf("host                  : hX[%d], hY[%d], hW[%d]\n", n, n, K);
    printf("global                : dX, dY, dW\n");
    printf("constant              : __constant__ cW[%d]\n", K);
    printf("shared                : __shared__ sW[%d] in kernelShared\n", K);
    printf("registers             : float xi, float h in each kernel (local bytes below)\n");
    printFn("kernelGlobal", (const void *)kernelGlobal);
    printFn("kernelConstant", (const void *)kernelConstant);
    printFn("kernelShared", (const void *)kernelShared);

    std::vector<float> hX(n), hY(n), hGot(n), hW(K);
    for (int k = 0; k < K; ++k)
        hW[k] = 1.f / static_cast<float>(k + 1);
    for (int i = 0; i < n; ++i)
        hX[i] = static_cast<float>(i % 100) * 0.01f;

    float *dX = nullptr, *dY = nullptr, *dW = nullptr;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&dX, bytes));
    CUDA_CHECK(cudaMalloc(&dY, bytes));
    CUDA_CHECK(cudaMalloc(&dW, sizeof(float) * K));
    CUDA_CHECK(cudaMemcpy(dX, hX.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW, hW.data(), sizeof(float) * K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(cW, hW.data(), sizeof(float) * K));

    cpuEval(hX.data(), hY.data(), n, hW.data());

    struct Item {
        const char *name;
        bool passW;
        void (*fn)(const float *, float *, int, const float *);
    };
    const Item items[] = {
        {"global weights", true, kernelGlobal},
        {"constant weights", false, kernelConstant},
        {"shared weights", true, kernelShared},
    };

    printf("\n=== Correctness (vs host Horner, tol %g) ===\n", kTol);
    int failed = 0;
    for (const Item &it : items) {
        if (it.passW)
            it.fn<<<numBlocks, blockSize>>>(dX, dY, n, dW);
        else
            it.fn<<<numBlocks, blockSize>>>(dX, dY, n, nullptr);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hGot.data(), dY, bytes, cudaMemcpyDeviceToHost));
        float maxErr = 0.f;
        int bad = check(hY.data(), hGot.data(), n, &maxErr);
        if (bad)
            ++failed;
        printf("%-18s : %s  mismatches=%d  max_abs_err=%.3g  y[0]=%.6f (host %.6f)\n",
               it.name, bad ? "FAIL" : "PASS", bad, maxErr, hGot[0], hY[0]);
    }

    printf("\n=== Timing (ms, mean of %d after %d warmup; kernel only) ===\n", kReps, kWarmup);
    printf("%-18s  %10s\n", "kernel", "ms");
    float ms[3];
    for (int i = 0; i < 3; ++i) {
        ms[i] = timeLaunch(items[i].fn, items[i].passW, dX, dY, n, dW, numBlocks, blockSize);
        printf("%-18s  %10.4f\n", items[i].name, ms[i]);
    }
    if (ms[0] > 0.f) {
        printf("same Horner loop; time relative to global weights:\n");
        printf("  constant / global = %.3f\n", ms[1] / ms[0]);
        printf("  shared / global   = %.3f\n", ms[2] / ms[0]);
    }

    CUDA_CHECK(cudaFree(dX));
    CUDA_CHECK(cudaFree(dY));
    CUDA_CHECK(cudaFree(dW));

    if (failed) {
        fprintf(stderr, "Correctness FAILED (%d kernel(s)).\n", failed);
        return EXIT_FAILURE;
    }
    printf("\nAll correctness checks passed.\n");
    return EXIT_SUCCESS;
}
