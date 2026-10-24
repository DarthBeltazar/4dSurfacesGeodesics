// Денойзер OptiX (DD-005 п. 5, ADR-0007): geo4d_gpu denoise <job.txt>
//
// Формат job-файла (по строке на директиву):
//   color <npy>        (H, W, 4) float32 — линейный HDR
//   albedo <npy>       (H, W, 4) float32 — AOV альбедо (необязательно)
//   normal <npy>       (H, W, 4) float32 — AOV нормаль в базисе камеры (необязательно, только вместе с альбедо)
//   output <npy>       (H, W, 4) float32
//   repeat <n>         повторов invoke для замера (результат — от последнего)
//   report <json>      время (мс) по событиям CUDA: setup, intensity + invoke на повтор
// CUDA Driver API — из nvcuda.dll динамически (без CUDA Toolkit); OptiX — из драйвера (nvoptix.dll через optixInit).
#include "denoise.hpp"

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "npy.hpp"

namespace {

// ---- CUDA Driver API (подмножество) ---------------------------------------------------------------
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
    }
};

void ck(CUresult r, const char* what) {
    if (r != 0) throw std::runtime_error(std::string("CUDA: ") + what + " → " + std::to_string(r));
}
void ok(OptixResult r, const char* what) {
    if (r != OPTIX_SUCCESS) throw std::runtime_error(std::string("OptiX: ") + what + " → " + optixGetErrorName(r));
}

struct DevBuf {
    Cuda* cu = nullptr;
    CUdeviceptr p = 0;
    size_t n = 0;
    void alloc(Cuda& c, size_t bytes) { cu = &c; n = std::max<size_t>(bytes, 4); ck(c.MemAlloc(&p, n), "cuMemAlloc"); }
    ~DevBuf() { if (p) cu->MemFree(p); }
};

NpyArray load_rgba(const std::string& path, size_t& H, size_t& W) {
    NpyArray a = npy_read(path);
    if (a.dtype != "<f4" || a.shape.size() != 3 || a.shape[2] != 4) throw std::runtime_error(path + ": нужен (H, W, 4) float32");
    if (H == 0) { H = a.shape[0]; W = a.shape[1]; }
    if (a.shape[0] != H || a.shape[1] != W) throw std::runtime_error(path + ": размер не совпадает с color");
    return a;
}

OptixImage2D image(CUdeviceptr p, size_t W, size_t H) {
    OptixImage2D im{};
    im.data = p;
    im.width = static_cast<unsigned>(W);
    im.height = static_cast<unsigned>(H);
    im.rowStrideInBytes = static_cast<unsigned>(W * 16);
    im.pixelStrideInBytes = 16;
    im.format = OPTIX_PIXEL_FORMAT_FLOAT4;
    return im;
}

}  // namespace

int run_denoise(const std::string& job_path) {
    std::ifstream jf(job_path);
    if (!jf) throw std::runtime_error("нет job-файла: " + job_path);
    std::string color, albedo, normal, output, report;
    int repeat = 1;
    for (std::string line; std::getline(jf, line);) {
        std::istringstream ss(line);
        std::string cmd;
        ss >> cmd;
        if (cmd == "color") ss >> color;
        else if (cmd == "albedo") ss >> albedo;
        else if (cmd == "normal") ss >> normal;
        else if (cmd == "output") ss >> output;
        else if (cmd == "repeat") ss >> repeat;
        else if (cmd == "report") ss >> report;
    }
    if (color.empty() || output.empty()) throw std::runtime_error("denoise: нужны color и output");
    if (!normal.empty() && albedo.empty()) throw std::runtime_error("denoise: normal только вместе с albedo");

    Cuda cu;
    ck(cu.Init(0), "cuInit");
    CUdevice dev;
    ck(cu.DeviceGet(&dev, 0), "cuDeviceGet");
    CUcontext ctx;
    ck(cu.DevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain");
    ck(cu.CtxSetCurrent(ctx), "cuCtxSetCurrent");
    CUstream stream;
    ck(cu.StreamCreate(&stream, 0), "cuStreamCreate");
    ok(optixInit(), "optixInit");
    OptixDeviceContextOptions copt{};
    OptixDeviceContext octx;
    ok(optixDeviceContextCreate(ctx, &copt, &octx), "optixDeviceContextCreate");

    size_t H = 0, W = 0;
    NpyArray c = load_rgba(color, H, W);
    NpyArray a, n;
    if (!albedo.empty()) a = load_rgba(albedo, H, W);
    if (!normal.empty()) n = load_rgba(normal, H, W);
    size_t bytes = H * W * 16;

    OptixDenoiserOptions dopt{};
    dopt.guideAlbedo = albedo.empty() ? 0 : 1;
    dopt.guideNormal = normal.empty() ? 0 : 1;
    OptixDenoiser den;
    ok(optixDenoiserCreate(octx, OPTIX_DENOISER_MODEL_KIND_HDR, &dopt, &den), "optixDenoiserCreate");
    OptixDenoiserSizes sz{};
    ok(optixDenoiserComputeMemoryResources(den, static_cast<unsigned>(W), static_cast<unsigned>(H), &sz), "ComputeMemoryResources");

    DevBuf dc, da, dn, dout, state, scratch, inten;
    dc.alloc(cu, bytes);
    dout.alloc(cu, bytes);
    ck(cu.MemcpyHtoD(dc.p, c.data.data(), bytes), "HtoD color");
    if (dopt.guideAlbedo) { da.alloc(cu, bytes); ck(cu.MemcpyHtoD(da.p, a.data.data(), bytes), "HtoD albedo"); }
    if (dopt.guideNormal) { dn.alloc(cu, bytes); ck(cu.MemcpyHtoD(dn.p, n.data.data(), bytes), "HtoD normal"); }
    state.alloc(cu, sz.stateSizeInBytes);
    size_t scratch_bytes = std::max(sz.withoutOverlapScratchSizeInBytes, sz.computeIntensitySizeInBytes);
    scratch.alloc(cu, scratch_bytes);
    inten.alloc(cu, sizeof(float));

    CUevent e0, e1, e2;
    ck(cu.EventCreate(&e0, 0), "cuEventCreate");
    ck(cu.EventCreate(&e1, 0), "cuEventCreate");
    ck(cu.EventCreate(&e2, 0), "cuEventCreate");
    ck(cu.EventRecord(e0, stream), "cuEventRecord");
    ok(optixDenoiserSetup(den, stream, static_cast<unsigned>(W), static_cast<unsigned>(H), state.p, state.n, scratch.p,
                          scratch.n), "optixDenoiserSetup");
    ck(cu.EventRecord(e1, stream), "cuEventRecord");
    ck(cu.EventSynchronize(e1), "cuEventSynchronize");
    float setup_ms = 0;
    ck(cu.EventElapsedTime(&setup_ms, e0, e1), "cuEventElapsedTime");

    OptixDenoiserLayer layer{};
    layer.input = image(dc.p, W, H);
    layer.output = image(dout.p, W, H);
    OptixDenoiserGuideLayer guide{};
    if (dopt.guideAlbedo) guide.albedo = image(da.p, W, H);
    if (dopt.guideNormal) guide.normal = image(dn.p, W, H);
    OptixDenoiserParams prm{};
    prm.hdrIntensity = inten.p;
    prm.blendFactor = 0.0f;

    std::vector<float> times;
    for (int r = 0; r < repeat; ++r) {
        ck(cu.EventRecord(e1, stream), "cuEventRecord");
        ok(optixDenoiserComputeIntensity(den, stream, &layer.input, inten.p, scratch.p, scratch.n), "ComputeIntensity");
        ok(optixDenoiserInvoke(den, stream, &prm, state.p, state.n, &guide, &layer, 1, 0, 0, scratch.p, scratch.n),
           "optixDenoiserInvoke");
        ck(cu.EventRecord(e2, stream), "cuEventRecord");
        ck(cu.EventSynchronize(e2), "cuEventSynchronize");
        float ms = 0;
        ck(cu.EventElapsedTime(&ms, e1, e2), "cuEventElapsedTime");
        times.push_back(ms);
    }
    std::vector<uint8_t> out(bytes);
    ck(cu.MemcpyDtoH(out.data(), dout.p, bytes), "DtoH");
    npy_write(output, "<f4", {H, W, 4}, out.data(), bytes);
    if (!report.empty()) {
        std::ofstream rf(report);
        rf << "{\"width\": " << W << ", \"height\": " << H << ", \"setup_ms\": " << setup_ms
           << ", \"state_bytes\": " << sz.stateSizeInBytes << ", \"scratch_bytes\": " << scratch_bytes
           << ", \"guide_albedo\": " << dopt.guideAlbedo << ", \"guide_normal\": " << dopt.guideNormal << ", \"times_ms\": [";
        for (size_t i = 0; i < times.size(); ++i) rf << (i ? ", " : "") << times[i];
        rf << "]}\n";
    }
    cu.EventDestroy(e0);
    cu.EventDestroy(e1);
    cu.EventDestroy(e2);
    optixDenoiserDestroy(den);
    optixDeviceContextDestroy(octx);
    cu.StreamDestroy(stream);
    cu.DevicePrimaryCtxRelease(dev);
    return 0;
}
