// EN.605.617.81 Module 9 — CUDA advanced libraries
// Xiaohui Chen
//
// Classify a one-second clip as 440 Hz, 1000 Hz, or noise.
// cuFFT makes a magnitude spectrogram. cuDNN applies three fixed
// filters, a ReLU, a max-pool, and a softmax.
//
// Each frame is scaled so its bins sum to 1. The 1000 Hz clips are
// generated at amplitude 0.4 so a quieter tone still has to match
// the shape of the spectrum, not the loudness.
//
// argv[1] = RNG seed (default 1)

#include "cuda_check.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static const int kSr = 8000;
static const int kSamples = 8000;
static const int kFrame = 256;
static const int kHop = 128;
static const int kBins = kFrame / 2 + 1;
static const int kFrames = (kSamples - kFrame) / kHop + 1;
static const int kClasses = 3;
static const int kPerClass = 4;
static const int kClips = kPerClass * kClasses;
static const float kPi = 3.14159265f;
static const float kErrLimit = 1e-3f;

static const char *kName[kClasses] = {"440 Hz", "1000 Hz", "noise"};

__global__ void window_frames(const float *wave, float *frames)
{
    int clip = blockIdx.x;
    int frame = blockIdx.y;
    int i = threadIdx.x;
    float hann = 0.5f - 0.5f * cosf((2.f * kPi * i) / (kFrame - 1));
    int src = clip * kSamples + frame * kHop + i;
    int dst = (clip * kFrames + frame) * kFrame + i;
    frames[dst] = wave[src] * hann;
}

// cuFFT writes complex bins packed by frame. cuDNN wants NCHW with
// frequency as the row and time as the column, each frame summing to 1.
__global__ void magnitude_nchw(const cufftComplex *spec, float *image)
{
    int clip = blockIdx.x;
    int frame = threadIdx.x;
    if (frame >= kFrames)
        return;

    const cufftComplex *col = spec + (clip * kFrames + frame) * kBins;
    float sum = 0.f;
    for (int b = 0; b < kBins; ++b) {
        float re = col[b].x;
        float im = col[b].y;
        sum += sqrtf(re * re + im * im);
    }
    float inv = 1.f / (sum + 1e-12f);
    for (int b = 0; b < kBins; ++b) {
        float re = col[b].x;
        float im = col[b].y;
        float mag = sqrtf(re * re + im * im) * inv;
        image[(clip * kBins + b) * kFrames + frame] = mag;
    }
}

static void fill_clip(float *x, int kind, std::mt19937 &rng)
{
    if (kind == 2) {
        std::normal_distribution<float> norm(0.f, 1.f);
        for (int i = 0; i < kSamples; ++i)
            x[i] = norm(rng);
        return;
    }
    std::uniform_real_distribution<float> phase_dist(0.f, 2.f * kPi);
    float phase = phase_dist(rng);
    float freq = (kind == 0) ? 440.f : 1000.f;
    float amp = (kind == 0) ? 1.f : 0.4f;
    for (int i = 0; i < kSamples; ++i) {
        float t = (float)i / (float)kSr;
        x[i] = amp * sinf(2.f * kPi * freq * t + phase);
    }
}

static void build_filters(std::vector<float> &filt)
{
    filt.assign(kClasses * kBins, 0.f);
    // 440 Hz sits at bin 14.08 (31.25 Hz per bin). 1000 Hz is bin 32.
    // The extra bin on each side catches the Hann window's sidelobes.
    for (int b = 13; b <= 15; ++b)
        filt[b] = 1.f;
    for (int b = 31; b <= 33; ++b)
        filt[kBins + b] = 1.f;
    for (int b = 0; b < kBins; ++b)
        filt[2 * kBins + b] = 0.2f;
}

static int parse_seed(int argc, char **argv)
{
    if (argc < 2)
        return 1;
    char *end = nullptr;
    long seed = strtol(argv[1], &end, 10);
    if (end == argv[1] || *end != '\0' || seed < 0 || seed > 1000000) {
        fprintf(stderr, "usage: %s [seed]\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    return (int)seed;
}

int main(int argc, char **argv)
{
    int seed = parse_seed(argc, argv);

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("device %s, cc %d.%d\n", prop.name, prop.major, prop.minor);

    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));
    size_t ver = cudnnGetVersion();
    printf("cudnn %zu.%zu.%zu (header %d)\n", ver / 10000, (ver / 100) % 100,
           ver % 100, CUDNN_VERSION);
    printf("audio %d Hz, %d samples, frame %d, hop %d, %d frames, %d bins\n",
           kSr, kSamples, kFrame, kHop, kFrames, kBins);
    printf("clips %d (%d per class), seed %d, 1000 Hz amplitude 0.4\n",
           kClips, kPerClass, seed);

    std::mt19937 rng(seed);
    std::vector<float> wave(kClips * kSamples);
    std::vector<int> label(kClips);
    for (int k = 0; k < kClasses; ++k) {
        for (int r = 0; r < kPerClass; ++r) {
            int clip = k * kPerClass + r;
            label[clip] = k;
            fill_clip(wave.data() + clip * kSamples, k, rng);
        }
    }
    std::vector<float> filt;
    build_filters(filt);

    float *d_wave = nullptr;
    float *d_frames = nullptr;
    cufftComplex *d_spec = nullptr;
    float *d_image = nullptr;
    float *d_filt = nullptr;
    float *d_conv = nullptr;
    float *d_pool = nullptr;
    CUDA_CHECK(cudaMalloc(&d_wave, wave.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_frames, (size_t)kClips * kFrames * kFrame * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_spec, (size_t)kClips * kFrames * kBins * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_image, (size_t)kClips * kBins * kFrames * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_filt, filt.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv, (size_t)kClips * kClasses * kFrames * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pool, (size_t)kClips * kClasses * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_wave, wave.data(), wave.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_filt, filt.data(), filt.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    dim3 grid(kClips, kFrames);
    window_frames<<<grid, kFrame>>>(d_wave, d_frames);
    CUDA_CHECK(cudaGetLastError());

    cufftHandle plan;
    int n[1] = {kFrame};
    int batch = kClips * kFrames;
    // NULL embeds: frames are packed back to back, which is how window_frames writes them.
    CUFFT_CHECK(cufftPlanMany(&plan, 1, n, nullptr, 1, kFrame, nullptr, 1, kBins,
                              CUFFT_R2C, batch));
    CUFFT_CHECK(cufftExecR2C(plan, d_frames, d_spec));
    printf("cufft R2C batch %d\n", batch);

    magnitude_nchw<<<kClips, 64>>>(d_spec, d_image);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudnnTensorDescriptor_t x_desc, y_desc, p_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnActivationDescriptor_t act_desc;
    cudnnPoolingDescriptor_t pool_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&p_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&w_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&act_desc));
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pool_desc));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           kClips, 1, kBins, kFrames));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                           kClasses, 1, kBins, 1));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1,
                                                CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    int out_n = 0, out_c = 0, out_h = 0, out_w = 0;
    CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(conv_desc, x_desc, w_desc,
                                                      &out_n, &out_c, &out_h, &out_w));
    printf("conv output %d %d %d %d\n", out_n, out_c, out_h, out_w);
    if (out_n != kClips || out_c != kClasses || out_h != 1 || out_w != kFrames) {
        fprintf(stderr, "conv output shape is not clips x classes x 1 x frames\n");
        return 1;
    }
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           out_n, out_c, out_h, out_w));
    CUDNN_CHECK(cudnnSetActivationDescriptor(act_desc, CUDNN_ACTIVATION_RELU,
                                             CUDNN_NOT_PROPAGATE_NAN, 0.f));
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(pool_desc, CUDNN_POOLING_MAX,
                                            CUDNN_NOT_PROPAGATE_NAN, 1, kFrames, 0, 0, 1, 1));
    int pn = 0, pc = 0, ph = 0, pw = 0;
    CUDNN_CHECK(cudnnGetPooling2dForwardOutputDim(pool_desc, y_desc, &pn, &pc, &ph, &pw));
    printf("pool output %d %d %d %d\n", pn, pc, ph, pw);
    if (pn != kClips || pc != kClasses || ph != 1 || pw != 1) {
        fprintf(stderr, "pool output shape is not clips x classes x 1 x 1\n");
        return 1;
    }
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(p_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           pn, pc, ph, pw));

    cudnnConvolutionFwdAlgo_t algos[] = {
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
        CUDNN_CONVOLUTION_FWD_ALGO_FFT,
        CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
        CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
        CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED};
    const char *algo_name[] = {
        "IMPLICIT_GEMM", "IMPLICIT_PRECOMP_GEMM", "GEMM", "DIRECT",
        "FFT", "FFT_TILING", "WINOGRAD", "WINOGRAD_NONFUSED"};
    float alpha = 1.f;
    float beta = 0.f;
    int chosen = -1;
    size_t chosen_bytes = 0;
    for (int i = 0; i < 8; ++i) {
        size_t bytes = 0;
        cudnnStatus_t st = cudnnGetConvolutionForwardWorkspaceSize(
            handle, x_desc, w_desc, conv_desc, y_desc, algos[i], &bytes);
        printf("algo %s workspace %s", algo_name[i], cudnnGetErrorString(st));
        if (st != CUDNN_STATUS_SUCCESS) {
            printf("\n");
            continue;
        }
        void *work = nullptr;
        if (bytes)
            CUDA_CHECK(cudaMalloc(&work, bytes));
        st = cudnnConvolutionForward(handle, &alpha, x_desc, d_image, w_desc, d_filt,
                                     conv_desc, algos[i], work, bytes, &beta, y_desc, d_conv);
        printf(" forward %s bytes %zu\n", cudnnGetErrorString(st), bytes);
        if (work)
            CUDA_CHECK(cudaFree(work));
        if (st == CUDNN_STATUS_SUCCESS && chosen < 0) {
            chosen = i;
            chosen_bytes = bytes;
        }
    }
    if (chosen < 0) {
        fprintf(stderr, "no convolution algorithm ran\n");
        return 1;
    }
    printf("using %s\n", algo_name[chosen]);

    // A later algorithm that succeeds overwrites d_conv, so rerun the one we kept.
    void *work = nullptr;
    if (chosen_bytes)
        CUDA_CHECK(cudaMalloc(&work, chosen_bytes));
    CUDNN_CHECK(cudnnConvolutionForward(handle, &alpha, x_desc, d_image, w_desc, d_filt,
                                        conv_desc, algos[chosen], work, chosen_bytes,
                                        &beta, y_desc, d_conv));

    std::vector<float> conv((size_t)kClips * kClasses * kFrames);
    std::vector<float> image((size_t)kClips * kBins * kFrames);
    CUDA_CHECK(cudaMemcpy(conv.data(), d_conv, conv.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(image.data(), d_image, image.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float max_err = 0.f;
    for (int clip = 0; clip < kClips; ++clip) {
        for (int k = 0; k < kClasses; ++k) {
            for (int frame = 0; frame < kFrames; ++frame) {
                float cpu = 0.f;
                for (int b = 0; b < kBins; ++b) {
                    float pix = image[(clip * kBins + b) * kFrames + frame];
                    cpu += filt[k * kBins + b] * pix;
                }
                float gpu = conv[(clip * kClasses + k) * kFrames + frame];
                max_err = fmaxf(max_err, fabsf(gpu - cpu));
            }
        }
    }
    printf("conv vs cpu max_abs_err %.3e\n", max_err);

    CUDNN_CHECK(cudnnActivationForward(handle, act_desc, &alpha, y_desc, d_conv, &beta,
                                       y_desc, d_conv));
    CUDNN_CHECK(cudnnPoolingForward(handle, pool_desc, &alpha, y_desc, d_conv, &beta,
                                    p_desc, d_pool));
    CUDNN_CHECK(cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                                    &alpha, p_desc, d_pool, &beta, p_desc, d_pool));

    std::vector<float> prob((size_t)kClips * kClasses);
    CUDA_CHECK(cudaMemcpy(prob.data(), d_pool, prob.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    printf("clip  label     pred      p_440   p_1000  p_noise\n");
    int correct = 0;
    for (int clip = 0; clip < kClips; ++clip) {
        int pred = 0;
        for (int k = 1; k < kClasses; ++k) {
            if (prob[clip * kClasses + k] > prob[clip * kClasses + pred])
                pred = k;
        }
        if (pred == label[clip])
            ++correct;
        printf("%4d  %-8s  %-8s  %6.4f  %6.4f  %6.4f\n", clip, kName[label[clip]],
               kName[pred], prob[clip * kClasses + 0], prob[clip * kClasses + 1],
               prob[clip * kClasses + 2]);
    }
    printf("correct %d/%d\n", correct, kClips);
    int pass = (correct == kClips && max_err < kErrLimit);
    printf("%s\n", pass ? "PASS" : "FAIL");

    if (work)
        CUDA_CHECK(cudaFree(work));
    CUFFT_CHECK(cufftDestroy(plan));
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pool_desc));
    CUDNN_CHECK(cudnnDestroyActivationDescriptor(act_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(w_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(p_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_desc));
    CUDNN_CHECK(cudnnDestroy(handle));
    CUDA_CHECK(cudaFree(d_pool));
    CUDA_CHECK(cudaFree(d_conv));
    CUDA_CHECK(cudaFree(d_filt));
    CUDA_CHECK(cudaFree(d_image));
    CUDA_CHECK(cudaFree(d_spec));
    CUDA_CHECK(cudaFree(d_frames));
    CUDA_CHECK(cudaFree(d_wave));
    return pass ? 0 : 1;
}
