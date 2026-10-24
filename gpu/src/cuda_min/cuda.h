// Минимальная замена <cuda.h> для заголовков OptiX (DD-005, ADR-0007): OptiX нужны только типы CUcontext,
// CUstream и CUdeviceptr. Функции CUDA Driver API загружаются из nvcuda.dll (драйвер) динамически — см. denoise.cpp.
// CUDA Toolkit не требуется.
#pragma once

#include <cstddef>

typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st* CUcontext;
typedef struct CUstream_st* CUstream;
typedef struct CUevent_st* CUevent;
#if defined(_WIN64) || defined(__LP64__)
typedef unsigned long long CUdeviceptr;
#else
typedef unsigned int CUdeviceptr;
#endif
