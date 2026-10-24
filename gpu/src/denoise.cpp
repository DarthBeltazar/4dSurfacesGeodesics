// Денойзер OptiX (DD-005 п. 5, ADR-0007): geo4d_gpu denoise <job.txt>
//
// Формат job-файла (по строке на директиву):
//   color <npy>        (H, W, 4) float32 — линейный HDR
//   albedo <npy>       (H, W, 4) float32 — AOV альбедо (необязательно)
//   normal <npy>       (H, W, 4) float32 — AOV нормаль в базисе камеры (необязательно, только вместе с альбедо)
//   output <npy>       (H, W, 4) float32
//   repeat <n>         повторов invoke для замера (результат — от последнего)
//   report <json>      время (мс) по событиям CUDA: setup, intensity + invoke на повтор
#include "denoise.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "denoiser.hpp"
#include "npy.hpp"

namespace {

NpyArray load_rgba(const std::string& path, size_t& H, size_t& W) {
    NpyArray a = npy_read(path);
    if (a.dtype != "<f4" || a.shape.size() != 3 || a.shape[2] != 4) throw std::runtime_error(path + ": нужен (H, W, 4) float32");
    if (H == 0) { H = a.shape[0]; W = a.shape[1]; }
    if (a.shape[0] != H || a.shape[1] != W) throw std::runtime_error(path + ": размер не совпадает с color");
    return a;
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
    size_t H = 0, W = 0;
    NpyArray c = load_rgba(color, H, W), a, n;
    if (!albedo.empty()) a = load_rgba(albedo, H, W);
    if (!normal.empty()) n = load_rgba(normal, H, W);
    size_t bytes = H * W * 16;

    OptixDenoiserCtx den;
    den.setup(unsigned(W), unsigned(H), !albedo.empty(), !normal.empty());
    uint64_t dc = den.alloc(bytes), dout = den.alloc(bytes), da = 0, dn = 0;
    den.upload(dc, c.data.data(), bytes);
    if (!albedo.empty()) { da = den.alloc(bytes); den.upload(da, a.data.data(), bytes); }
    if (!normal.empty()) { dn = den.alloc(bytes); den.upload(dn, n.data.data(), bytes); }
    std::vector<float> times;
    for (int r = 0; r < repeat; ++r) times.push_back(den.invoke(dc, da, dn, dout));
    std::vector<uint8_t> out(bytes);
    den.download(out.data(), dout, bytes);
    npy_write(output, "<f4", {H, W, 4}, out.data(), bytes);
    if (!report.empty()) {
        std::ofstream rf(report);
        rf << "{\"width\": " << W << ", \"height\": " << H << ", \"setup_ms\": " << den.setup_ms()
           << ", \"memory_bytes\": " << den.memory_bytes() << ", \"guide_albedo\": " << !albedo.empty()
           << ", \"guide_normal\": " << !normal.empty() << ", \"times_ms\": [";
        for (size_t i = 0; i < times.size(); ++i) rf << (i ? ", " : "") << times[i];
        rf << "]}\n";
    }
    return 0;
}
