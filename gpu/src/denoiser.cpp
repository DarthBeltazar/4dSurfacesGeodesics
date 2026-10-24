// Денойзер OptiX: см. denoiser.hpp. CUDA Driver API — подмножество из nvcuda.dll (ADR-0007).
#include "denoiser.hpp"

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Cuda {
    HMODULE dll = nullptr;
    CUresult (*Init)(unsigned) = nullptr;
    CUresult (*DeviceGet)(CUdevice*, int) = nullptr;
    CUresult (*DevicePrimaryCtxRetain)(CUcontext*, CUdevice) = nullptr;
    CUresult (*DevicePrimaryCtxRelease)(CUdevice) = nullptr;
    CUresult (*CtxSetCurrent)(CUcontext) = nullptr;
    CUresult (*StreamCreate)(CUstream*, unsigned) = nullptr;
    CUresult (*StreamDestroy)(CUstream) = nullptr;
    CUresult (*StreamSynchronize)(CUstream) = nullptr;
    CUresult (*MemAlloc)(CUdeviceptr*, size_t) = nullptr;
    CUresult (*MemFree)(CUdeviceptr) = nullptr;
    CUresult (*MemcpyHtoD)(CUdeviceptr, const void*, size_t) = nullptr;
    CUresult (*MemcpyDtoH)(void*, CUdeviceptr, size_t) = nullptr;
    CUresult (*EventCreate)(CUevent*, unsigned) = nullptr;
    CUresult (*EventRecord)(CUevent, CUstream) = nullptr;
    CUresult (*EventSynchronize)(CUevent) = nullptr;
    CUresult (*EventElapsedTime)(float*, CUevent, CUevent) = nullptr;
    CUresult (*EventDestroy)(CUevent) = nullptr;
    CUresult (*ImportExternalMemory)(CUexternalMemory*, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC*) = nullptr;
    CUresult (*ExternalMemoryGetMappedBuffer)(CUdeviceptr*, CUexternalMemory, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC*) = nullptr;
    CUresult (*DestroyExternalMemory)(CUexternalMemory) = nullptr;

    template <class F> void get(F& f, const char* name) {
        f = reinterpret_cast<F>(GetProcAddress(dll, name));
        if (!f) throw std::runtime_error(std::string("nvcuda.dll: нет ") + name);
    }
    Cuda() {
        dll = LoadLibraryA("nvcuda.dll");
        if (!dll) throw std::runtime_error("нет nvcuda.dll (драйвер NVIDIA)");
        get(Init, "cuInit");
        get(DeviceGet, "cuDeviceGet");
        get(DevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain");
        get(DevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease_v2");
        get(CtxSetCurrent, "cuCtxSetCurrent");
        get(StreamCreate, "cuStreamCreate");
        get(StreamDestroy, "cuStreamDestroy_v2");
        get(StreamSynchronize, "cuStreamSynchronize");
        get(MemAlloc, "cuMemAlloc_v2");
        get(MemFree, "cuMemFree_v2");
        get(MemcpyHtoD, "cuMemcpyHtoD_v2");
        get(MemcpyDtoH, "cuMemcpyDtoH_v2");
        get(EventCreate, "cuEventCreate");
        get(EventRecord, "cuEventRecord");
        get(EventSynchronize, "cuEventSynchronize");
        get(EventElapsedTime, "cuEventElapsedTime_v2");
        get(EventDestroy, "cuEventDestroy_v2");
        get(ImportExternalMemory, "cuImportExternalMemory");
        get(ExternalMemoryGetMappedBuffer, "cuExternalMemoryGetMappedBuffer");
        get(DestroyExternalMemory, "cuDestroyExternalMemory");
    }
};

void ck(CUresult r, const char* what) {
    if (r != 0) throw std::runtime_error(std::string("CUDA: ") + what + " → " + std::to_string(r));
}
void ok(OptixResult r, const char* what) {
    if (r != OPTIX_SUCCESS) throw std::runtime_error(std::string("OptiX: ") + what + " → " + optixGetErrorName(r));
}

OptixImage2D image(CUdeviceptr p, unsigned W, unsigned H) {
    OptixImage2D im{};
    im.data = p;
    im.width = W;
    im.height = H;
    im.rowStrideInBytes = W * 16;
    im.pixelStrideInBytes = 16;
    im.format = OPTIX_PIXEL_FORMAT_FLOAT4;
    return im;
}

}  // namespace

struct OptixDenoiserCtx::Impl {
    Cuda cu;
    CUdevice dev = 0;
    CUcontext ctx = nullptr;
    CUstream stream = nullptr;
    OptixDeviceContext octx = nullptr;
    OptixDenoiser den = nullptr;
    unsigned W = 0, H = 0;
    bool alb = false, nrm = false;
    CUdeviceptr state = 0, scratch = 0, inten = 0;
    size_t state_n = 0, scratch_n = 0;
    float setup_ms = 0;
    CUevent e0 = nullptr, e1 = nullptr;
    std::vector<CUdeviceptr> owned;
    std::vector<CUexternalMemory> ext;
};

OptixDenoiserCtx::OptixDenoiserCtx() : d_(std::make_unique<Impl>()) {
    auto& c = d_->cu;
    ck(c.Init(0), "cuInit");
    ck(c.DeviceGet(&d_->dev, 0), "cuDeviceGet");
    ck(c.DevicePrimaryCtxRetain(&d_->ctx, d_->dev), "cuDevicePrimaryCtxRetain");
    ck(c.CtxSetCurrent(d_->ctx), "cuCtxSetCurrent");
    ck(c.StreamCreate(&d_->stream, 0), "cuStreamCreate");
    ck(c.EventCreate(&d_->e0, 0), "cuEventCreate");
    ck(c.EventCreate(&d_->e1, 0), "cuEventCreate");
    ok(optixInit(), "optixInit");
    OptixDeviceContextOptions copt{};
    ok(optixDeviceContextCreate(d_->ctx, &copt, &d_->octx), "optixDeviceContextCreate");
}

OptixDenoiserCtx::~OptixDenoiserCtx() {
    auto& c = d_->cu;
    c.StreamSynchronize(d_->stream);
    if (d_->den) optixDenoiserDestroy(d_->den);
    for (auto p : d_->owned) c.MemFree(p);
    for (auto e : d_->ext) c.DestroyExternalMemory(e);
    if (d_->octx) optixDeviceContextDestroy(d_->octx);
    c.EventDestroy(d_->e0);
    c.EventDestroy(d_->e1);
    c.StreamDestroy(d_->stream);
    c.DevicePrimaryCtxRelease(d_->dev);
}

uint64_t OptixDenoiserCtx::alloc(size_t bytes) {
    CUdeviceptr p;
    ck(d_->cu.MemAlloc(&p, std::max<size_t>(bytes, 4)), "cuMemAlloc");
    d_->owned.push_back(p);
    return p;
}

void OptixDenoiserCtx::upload(uint64_t dst, const void* src, size_t bytes) { ck(d_->cu.MemcpyHtoD(dst, src, bytes), "HtoD"); }
void OptixDenoiserCtx::download(void* dst, uint64_t src, size_t bytes) { ck(d_->cu.MemcpyDtoH(dst, src, bytes), "DtoH"); }

uint64_t OptixDenoiserCtx::import_win32(void* handle, size_t alloc_size, size_t buffer_size) {
    CUDA_EXTERNAL_MEMORY_HANDLE_DESC hd{};
    hd.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32;
    hd.handle.win32.handle = handle;
    hd.size = alloc_size;
    hd.flags = CUDA_EXTERNAL_MEMORY_DEDICATED;
    CUexternalMemory em;
    ck(d_->cu.ImportExternalMemory(&em, &hd), "cuImportExternalMemory");
    d_->ext.push_back(em);
    CUDA_EXTERNAL_MEMORY_BUFFER_DESC bd{};
    bd.offset = 0;
    bd.size = buffer_size;
    CUdeviceptr p;
    ck(d_->cu.ExternalMemoryGetMappedBuffer(&p, em, &bd), "cuExternalMemoryGetMappedBuffer");
    return p;
}

void OptixDenoiserCtx::setup(unsigned W, unsigned H, bool albedo, bool normal) {
    auto& c = d_->cu;
    d_->W = W; d_->H = H; d_->alb = albedo; d_->nrm = normal;
    OptixDenoiserOptions dopt{};
    dopt.guideAlbedo = albedo ? 1 : 0;
    dopt.guideNormal = normal ? 1 : 0;
    ok(optixDenoiserCreate(d_->octx, OPTIX_DENOISER_MODEL_KIND_HDR, &dopt, &d_->den), "optixDenoiserCreate");
    OptixDenoiserSizes sz{};
    ok(optixDenoiserComputeMemoryResources(d_->den, W, H, &sz), "ComputeMemoryResources");
    d_->state_n = sz.stateSizeInBytes;
    d_->scratch_n = std::max(sz.withoutOverlapScratchSizeInBytes, sz.computeIntensitySizeInBytes);
    d_->state = alloc(d_->state_n);
    d_->scratch = alloc(d_->scratch_n);
    d_->inten = alloc(sizeof(float));
    ck(c.EventRecord(d_->e0, d_->stream), "cuEventRecord");
    ok(optixDenoiserSetup(d_->den, d_->stream, W, H, d_->state, d_->state_n, d_->scratch, d_->scratch_n), "optixDenoiserSetup");
    ck(c.EventRecord(d_->e1, d_->stream), "cuEventRecord");
    ck(c.EventSynchronize(d_->e1), "cuEventSynchronize");
    ck(c.EventElapsedTime(&d_->setup_ms, d_->e0, d_->e1), "cuEventElapsedTime");
}

float OptixDenoiserCtx::invoke(uint64_t color, uint64_t albedo, uint64_t normal, uint64_t output) {
    auto& c = d_->cu;
    OptixDenoiserLayer layer{};
    layer.input = image(color, d_->W, d_->H);
    layer.output = image(output, d_->W, d_->H);
    OptixDenoiserGuideLayer guide{};
    if (d_->alb) guide.albedo = image(albedo, d_->W, d_->H);
    if (d_->nrm) guide.normal = image(normal, d_->W, d_->H);
    OptixDenoiserParams prm{};
    prm.hdrIntensity = d_->inten;
    prm.blendFactor = 0.0f;
    ck(c.EventRecord(d_->e0, d_->stream), "cuEventRecord");
    ok(optixDenoiserComputeIntensity(d_->den, d_->stream, &layer.input, d_->inten, d_->scratch, d_->scratch_n), "ComputeIntensity");
    ok(optixDenoiserInvoke(d_->den, d_->stream, &prm, d_->state, d_->state_n, &guide, &layer, 1, 0, 0, d_->scratch,
                           d_->scratch_n), "optixDenoiserInvoke");
    ck(c.EventRecord(d_->e1, d_->stream), "cuEventRecord");
    ck(c.EventSynchronize(d_->e1), "cuEventSynchronize");
    float ms = 0;
    ck(c.EventElapsedTime(&ms, d_->e0, d_->e1), "cuEventElapsedTime");
    return ms;
}

float OptixDenoiserCtx::setup_ms() const { return d_->setup_ms; }
size_t OptixDenoiserCtx::memory_bytes() const { return d_->state_n + d_->scratch_n; }
