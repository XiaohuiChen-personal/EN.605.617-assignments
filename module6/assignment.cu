// EN.605.617.81 Module 6 — CUDA streams and events
// Xiaohui Chen
//
// One kernel, mapQuadratic, on two arrays of length N.
// Both launches call that kernel. The three modes change only the schedule:
//   one_stream   both arrays queued on one stream
//   two_streams  one stream per array, neither waits on the other
//   wait_event   stream 1's kernel waits on stream 0's kernel-end event
//
// Host buffers are page-locked so cudaMemcpyAsync can return before the
// copy finishes. Streams are created cudaStreamNonBlocking, so work queued
// here is not ordered against the legacy default stream.
//
// argv[1] = N, elements in each array (default 1048576)
// argv[2] = threads per block (default 256)

#include "cuda_check.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

static const int kSteps = 1024;
static const int kDefaultN = 1 << 20;
static const int kDefaultBlock = 256;
static const int kMaxN = 1 << 26;
static const int kWarmup = 1;
static const int kReps = 5;
static const int kHead = 64;
static const int kSampleStride = 4097;
static const float kQuad = 0.15f;
static const float kLin = 0.20f;
static const float kBias = 0.30f;
static const float kCap = 4.f;
static const float kTol = 1e-5f;
static const int kSaltA = 0;
static const int kSaltB = 17;

enum Mode { OneStream, TwoStreams, WaitEvent };

struct Launch {
    int n;
    int block;
};

struct Buf {
    float *hostIn;
    float *hostOut;
    float *devIn;
    float *devOut;
    cudaStream_t stream;
    cudaEvent_t copyIn;
    cudaEvent_t kernelStart;
    cudaEvent_t kernelStop;
    cudaEvent_t copyOut;
};

struct Times {
    float wall;
    float kernel0;
    float kernel1;
    float a0, a1, b0, b1;
    int overlap;
};

struct ModeResult {
    float wall;
    int failed;
};

__host__ __device__ float iterate(float v, int steps)
{
    for (int s = 0; s < steps; ++s) {
        float t = v * v;
        v = fmaf(t, kQuad, fmaf(v, kLin, kBias));
        v = fminf(v, kCap);
    }
    return v;
}

__global__ void mapQuadratic(const float *x, float *y, int n, int steps)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    y[i] = iterate(x[i], steps);
}

static Launch parseLaunch(int argc, char **argv)
{
    Launch launch;
    launch.n = kDefaultN;
    launch.block = kDefaultBlock;
    if (argc >= 2)
        launch.n = atoi(argv[1]);
    if (argc >= 3)
        launch.block = atoi(argv[2]);
    return launch;
}

static int rejectLaunch(const Launch &launch, const cudaDeviceProp &prop)
{
    if (launch.n <= 0 || launch.block <= 0) {
        fprintf(stderr, "usage: assignment.exe <n> <blockSize>\n");
        return 1;
    }
    if (launch.n > kMaxN) {
        fprintf(stderr, "n %d exceeds cap %d\n", launch.n, kMaxN);
        return 1;
    }
    if (launch.block > prop.maxThreadsPerBlock) {
        fprintf(stderr, "block size %d exceeds maxThreadsPerBlock %d\n",
                launch.block, prop.maxThreadsPerBlock);
        return 1;
    }
    return 0;
}

static void printDevice(const cudaDeviceProp &prop)
{
    int driver = 0, runtime = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    printf("device                : %s\n", prop.name);
    printf("compute capability    : %d.%d\n", prop.major, prop.minor);
    printf("SMs                   : %d\n", prop.multiProcessorCount);
    printf("warpSize              : %d\n", prop.warpSize);
    printf("concurrentKernels     : %d\n", prop.concurrentKernels);
    printf("asyncEngineCount      : %d\n", prop.asyncEngineCount);
    printf("integrated            : %d\n", prop.integrated);
    printf("driver / runtime      : %d / %d\n", driver, runtime);
}

static void fillInput(float *x, int n, int salt)
{
    for (int i = 0; i < n; ++i)
        x[i] = static_cast<float>((i + salt) % 97) * 0.01f;
}

static void createEvents(Buf *b)
{
    CUDA_CHECK(cudaEventCreate(&b->copyIn));
    CUDA_CHECK(cudaEventCreate(&b->kernelStart));
    CUDA_CHECK(cudaEventCreate(&b->kernelStop));
    CUDA_CHECK(cudaEventCreate(&b->copyOut));
}

static void allocBuf(Buf *b, int n, int salt)
{
    size_t bytes = static_cast<size_t>(n) * sizeof(float);
    CUDA_CHECK(cudaMallocHost(&b->hostIn, bytes));
    CUDA_CHECK(cudaMallocHost(&b->hostOut, bytes));
    CUDA_CHECK(cudaMalloc(&b->devIn, bytes));
    CUDA_CHECK(cudaMalloc(&b->devOut, bytes));
    createEvents(b);
    fillInput(b->hostIn, n, salt);
    b->stream = nullptr;
}

static void freeBuf(Buf *b)
{
    CUDA_CHECK(cudaEventDestroy(b->copyIn));
    CUDA_CHECK(cudaEventDestroy(b->kernelStart));
    CUDA_CHECK(cudaEventDestroy(b->kernelStop));
    CUDA_CHECK(cudaEventDestroy(b->copyOut));
    CUDA_CHECK(cudaFree(b->devIn));
    CUDA_CHECK(cudaFree(b->devOut));
    CUDA_CHECK(cudaFreeHost(b->hostIn));
    CUDA_CHECK(cudaFreeHost(b->hostOut));
}

static void openStreams(cudaStream_t *s0, cudaStream_t *s1, Mode mode)
{
    // Non-blocking: a later default-stream call cannot serialize these two.
    CUDA_CHECK(cudaStreamCreateWithFlags(s0, cudaStreamNonBlocking));
    if (mode == OneStream) {
        *s1 = *s0;
        return;
    }
    CUDA_CHECK(cudaStreamCreateWithFlags(s1, cudaStreamNonBlocking));
}

static void closeStreams(cudaStream_t s0, cudaStream_t s1)
{
    CUDA_CHECK(cudaStreamDestroy(s0));
    if (s1 != s0)
        CUDA_CHECK(cudaStreamDestroy(s1));
}

static void enqueue(Buf *b, int n, int block, cudaEvent_t waitFor)
{
    size_t bytes = static_cast<size_t>(n) * sizeof(float);
    int grid = (n + block - 1) / block;
    CUDA_CHECK(cudaEventRecord(b->copyIn, b->stream));
    CUDA_CHECK(cudaMemcpyAsync(b->devIn, b->hostIn, bytes,
                               cudaMemcpyHostToDevice, b->stream));
    // Copy-in stays ahead of the wait, so it can still overlap the other stream.
    if (waitFor)
        CUDA_CHECK(cudaStreamWaitEvent(b->stream, waitFor, 0));
    CUDA_CHECK(cudaEventRecord(b->kernelStart, b->stream));
    mapQuadratic<<<grid, block, 0, b->stream>>>(b->devIn, b->devOut, n, kSteps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(b->kernelStop, b->stream));
    CUDA_CHECK(cudaMemcpyAsync(b->hostOut, b->devOut, bytes,
                               cudaMemcpyDeviceToHost, b->stream));
    CUDA_CHECK(cudaEventRecord(b->copyOut, b->stream));
}

static float elapsed(cudaEvent_t start, cudaEvent_t stop)
{
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

static Times readTimes(cudaEvent_t ref, const Buf *a, const Buf *b)
{
    CUDA_CHECK(cudaEventSynchronize(a->copyOut));
    CUDA_CHECK(cudaEventSynchronize(b->copyOut));
    CUDA_CHECK(cudaGetLastError());

    Times t;
    float aIn = elapsed(ref, a->copyIn);
    float bIn = elapsed(ref, b->copyIn);
    float aOut = elapsed(ref, a->copyOut);
    float bOut = elapsed(ref, b->copyOut);
    t.a0 = elapsed(ref, a->kernelStart);
    t.a1 = elapsed(ref, a->kernelStop);
    t.b0 = elapsed(ref, b->kernelStart);
    t.b1 = elapsed(ref, b->kernelStop);
    t.kernel0 = t.a1 - t.a0;
    t.kernel1 = t.b1 - t.b0;
    float start = aIn < bIn ? aIn : bIn;
    float stop = aOut > bOut ? aOut : bOut;
    t.wall = stop - start;
    t.overlap = (t.a0 < t.b1 && t.b0 < t.a1) ? 1 : 0;
    return t;
}

static Times oneRep(cudaEvent_t ref, Buf *a, Buf *b, Mode mode, int n, int block)
{
    // ref is completed before either stream is given work, so both
    // timelines share one origin. Recording it on the default stream
    // does not order the non-blocking streams with each other.
    CUDA_CHECK(cudaEventRecord(ref, 0));
    CUDA_CHECK(cudaEventSynchronize(ref));
    enqueue(a, n, block, nullptr);
    cudaEvent_t wait = (mode == WaitEvent) ? a->kernelStop : nullptr;
    enqueue(b, n, block, wait);
    return readTimes(ref, a, b);
}

static int missAt(const float *in, const float *out, int i, float *worst)
{
    float err = fabsf(iterate(in[i], kSteps) - out[i]);
    if (err > *worst)
        *worst = err;
    return err > kTol ? 1 : 0;
}

static int checkBuf(const Buf *b, int n, float *worst)
{
    int bad = 0;
    int head = n < kHead ? n : kHead;
    for (int i = 0; i < head; ++i)
        bad += missAt(b->hostIn, b->hostOut, i, worst);
    for (int i = kSampleStride; i < n; i += kSampleStride)
        bad += missAt(b->hostIn, b->hostOut, i, worst);
    if (n - 1 >= head && ((n - 1) % kSampleStride) != 0)
        bad += missAt(b->hostIn, b->hostOut, n - 1, worst);
    return bad;
}

static void printRow(const char *name, const Times &sum, int overlaps,
                     float worst, int bad)
{
    char overlap[16];
    snprintf(overlap, sizeof overlap, "%d/%d", overlaps, kReps);
    float inv = 1.f / static_cast<float>(kReps);
    printf("%-12s %8.4f %8.4f %8.4f %9s %12.3g  %s\n",
           name, sum.wall * inv, sum.kernel0 * inv, sum.kernel1 * inv,
           overlap, worst, bad ? "FAIL" : "PASS");
}

static void printRunHeader(const cudaDeviceProp &prop, const Launch &launch)
{
    if (launch.block < prop.warpSize) {
        printf("Warning: block size %d is below warpSize %d\n",
               launch.block, prop.warpSize);
    }
    printDevice(prop);
    printf("command               : assignment.exe %d %d\n", launch.n, launch.block);
    printf("N                     : %d\n", launch.n);
    printf("block                 : %d\n", launch.block);
    printf("steps                 : %d\n", kSteps);
    printf("reps                  : %d after %d warmup\n", kReps, kWarmup);
    printf("checked indices       : i < %d, or i >= %d with i %% %d == 0, or i == N-1\n",
           kHead, kSampleStride, kSampleStride);
}

static Times timeReps(cudaEvent_t ref, Buf *a, Buf *b, Mode mode,
                      int n, int block, int *overlaps, Times *last)
{
    for (int w = 0; w < kWarmup; ++w)
        (void)oneRep(ref, a, b, mode, n, block);

    Times sum = {};
    *overlaps = 0;
    for (int r = 0; r < kReps; ++r) {
        *last = oneRep(ref, a, b, mode, n, block);
        sum.wall += last->wall;
        sum.kernel0 += last->kernel0;
        sum.kernel1 += last->kernel1;
        *overlaps += last->overlap;
    }
    return sum;
}

static ModeResult runMode(const char *name, Buf *a, Buf *b, Mode mode, int n, int block)
{
    cudaStream_t s0, s1;
    openStreams(&s0, &s1, mode);
    a->stream = s0;
    b->stream = s1;

    cudaEvent_t ref;
    CUDA_CHECK(cudaEventCreate(&ref));
    int overlaps = 0;
    Times last = {};
    Times sum = timeReps(ref, a, b, mode, n, block, &overlaps, &last);

    float worst = 0.f;
    int bad = checkBuf(a, n, &worst) + checkBuf(b, n, &worst);
    printRow(name, sum, overlaps, worst, bad);
    printf("             last rep from ref: A %.4f..%.4f  B %.4f..%.4f ms\n",
           last.a0, last.a1, last.b0, last.b1);

    CUDA_CHECK(cudaEventDestroy(ref));
    closeStreams(s0, s1);

    ModeResult result;
    result.wall = sum.wall / static_cast<float>(kReps);
    result.failed = bad ? 1 : 0;
    return result;
}

int main(int argc, char **argv)
{
    Launch launch = parseLaunch(argc, argv);
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (rejectLaunch(launch, prop))
        return EXIT_FAILURE;

    printRunHeader(prop, launch);

    Buf a, b;
    allocBuf(&a, launch.n, kSaltA);
    allocBuf(&b, launch.n, kSaltB);

    printf("\n%-12s %8s %8s %8s %9s %12s  %s\n",
           "mode", "wall_ms", "k0_ms", "k1_ms", "k_overlap", "max_abs_err", "check");
    ModeResult one = runMode("one_stream", &a, &b, OneStream, launch.n, launch.block);
    ModeResult two = runMode("two_streams", &a, &b, TwoStreams, launch.n, launch.block);
    ModeResult waited = runMode("wait_event", &a, &b, WaitEvent, launch.n, launch.block);

    if (one.wall > 0.f) {
        printf("wall two_streams/one_stream = %.3f\n", two.wall / one.wall);
        printf("wall wait_event/one_stream  = %.3f\n", waited.wall / one.wall);
    }

    freeBuf(&a);
    freeBuf(&b);

    int failed = one.failed + two.failed + waited.failed;
    if (failed) {
        fprintf(stderr, "Correctness FAILED (%d mode(s)).\n", failed);
        return EXIT_FAILURE;
    }
    printf("All checks passed.\n");
    return EXIT_SUCCESS;
}
