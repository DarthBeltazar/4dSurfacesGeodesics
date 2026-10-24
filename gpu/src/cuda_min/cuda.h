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

// Внешняя память (импорт буферов Vulkan, DD-007): раскладка — как в cuda.h CUDA 11+ (cuImportExternalMemory).
typedef struct CUextMemory_st* CUexternalMemory;
typedef enum CUexternalMemoryHandleType_enum {
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD = 1,
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32 = 2,
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT = 3
} CUexternalMemoryHandleType;
#define CUDA_EXTERNAL_MEMORY_DEDICATED 0x1
typedef struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st {
    CUexternalMemoryHandleType type;
    union {
        int fd;
        struct { void* handle; const void* name; } win32;
        const void* nvSciBufObject;
    } handle;
    unsigned long long size;
    unsigned int flags;
    unsigned int reserved[16];
} CUDA_EXTERNAL_MEMORY_HANDLE_DESC;
typedef struct CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st {
    unsigned long long offset;
    unsigned long long size;
    unsigned int flags;
    unsigned int reserved[16];
} CUDA_EXTERNAL_MEMORY_BUFFER_DESC;
