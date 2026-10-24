// Денойзер OptiX (DD-005 п. 5, ADR-0007, DD-007): CUDA Driver API из nvcuda.dll, OptiX из драйвера.
// Изображения — (H, W) float4 в памяти устройства CUDA: свои выделения или память буферов Vulkan, импортированная
// через внешнюю память (import_win32).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

class OptixDenoiserCtx {
public:
    OptixDenoiserCtx();
    ~OptixDenoiserCtx();
    OptixDenoiserCtx(const OptixDenoiserCtx&) = delete;
    OptixDenoiserCtx& operator=(const OptixDenoiserCtx&) = delete;

    // модель HDR; albedo / normal — включить направляющие слои
    void setup(unsigned width, unsigned height, bool albedo, bool normal);
    // указатели CUDA (CUdeviceptr); albedo/normal = 0, если не включены. Возвращает мс (события CUDA).
    float invoke(uint64_t color, uint64_t albedo, uint64_t normal, uint64_t output);
    float setup_ms() const;
    size_t memory_bytes() const;

    uint64_t alloc(size_t bytes);
    void upload(uint64_t dst, const void* src, size_t bytes);
    void download(void* dst, uint64_t src, size_t bytes);
    // импорт памяти Vulkan (HANDLE OPAQUE_WIN32, размер выделения) → указатель CUDA на начало буфера
    uint64_t import_win32(void* handle, size_t alloc_size, size_t buffer_size);

private:
    struct Impl;
    std::unique_ptr<Impl> d_;
};
