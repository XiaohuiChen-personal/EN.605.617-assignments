#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cudnn.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s -> %s\n", __FILE__,        \
                    __LINE__, #call, cudaGetErrorString(err__));                \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define CUFFT_CHECK(call)                                                       \
    do {                                                                        \
        cufftResult err__ = (call);                                             \
        if (err__ != CUFFT_SUCCESS) {                                           \
            fprintf(stderr, "cuFFT error at %s:%d: %s -> %d\n", __FILE__,       \
                    __LINE__, #call, (int)err__);                               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define CUDNN_CHECK(call)                                                       \
    do {                                                                        \
        cudnnStatus_t err__ = (call);                                           \
        if (err__ != CUDNN_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "cuDNN error at %s:%d: %s -> %s\n", __FILE__,       \
                    __LINE__, #call, cudnnGetErrorString(err__));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)
