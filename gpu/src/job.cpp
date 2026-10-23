// Исполнитель одной compute-диспетчеризации по job-файлу (стенд сравнения с прототипом, DD-004 п. 10).
//
// Формат (по строке на директиву, '#' — комментарий):
//   spv <path>                     SPIR-V
//   entry <name>                   точка входа
//   push f32|u32|i32 <value>       4 байта push-констант (по порядку)
//   buf <binding> in <npy>         входной буфер
//   buf <binding> out <npy> <dtype> <dims...>   выходной (dtype: f4 | u4 | i4)
//   buf <binding> inout <in.npy> <out.npy>
//   cube <binding> <npy> rgba16f|rgba32f|bc6h <aniso> [encoder.spv]   кубмапа-массив: npy (листы, 6, N, N, 3|4) float32
//   groups <gx> <gy> <gz>
//   repeat <n>                     повторить диспетчеризацию n раз (для замеров; буферы не сбрасываются)
//   report <json>                  время, статистика конвейера, устройство
#include "job.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "npy.hpp"

namespace {

struct BufSpec {
    uint32_t binding = 0;
    std::string mode, in, out, dtype;
    std::vector<size_t> shape;
    float aniso = 16.0f;
};

std::vector<uint32_t> read_spirv(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("нет SPIR-V: " + path);
    size_t n = static_cast<size_t>(f.tellg());
    std::vector<uint32_t> code(n / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(code.data()), n);
    return code;
}

std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') o += '\\';
        if (c == '\n') { o += "\\n"; continue; }
        o += c;
    }
    return o;
}

}  // namespace

int run_job(VkContext& ctx, const std::string& job_path) {
    std::ifstream jf(job_path);
    if (!jf) throw std::runtime_error("нет job-файла: " + job_path);
    std::string spv, entry, report;
    std::vector<uint8_t> push;
    std::vector<BufSpec> bufs;
    uint32_t g[3] = {1, 1, 1}, repeat = 1;
    std::string line;
    while (std::getline(jf, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string cmd;
        ss >> cmd;
        if (cmd == "spv") ss >> spv;
        else if (cmd == "entry") ss >> entry;
        else if (cmd == "report") ss >> report;
        else if (cmd == "groups") ss >> g[0] >> g[1] >> g[2];
        else if (cmd == "repeat") ss >> repeat;
        else if (cmd == "push") {
            std::string t;
            ss >> t;
            uint8_t b[4];
            if (t == "f32") { float v; ss >> v; std::memcpy(b, &v, 4); }
            else if (t == "u32") { uint32_t v; ss >> v; std::memcpy(b, &v, 4); }
            else if (t == "i32") { int32_t v; ss >> v; std::memcpy(b, &v, 4); }
            else throw std::runtime_error("push: тип " + t);
            if (ss.fail()) throw std::runtime_error("push: не разобрано значение в строке: " + line);
            push.insert(push.end(), b, b + 4);
        } else if (cmd == "buf") {
            BufSpec b;
            ss >> b.binding >> b.mode;
            if (b.mode == "in") ss >> b.in;
            else if (b.mode == "inout") ss >> b.in >> b.out;
            else if (b.mode == "out") {
                ss >> b.out >> b.dtype;
                size_t d;
                while (ss >> d) b.shape.push_back(d);
            } else throw std::runtime_error("buf: режим " + b.mode);
            bufs.push_back(b);
        } else if (cmd == "cube") {
            BufSpec b;
            b.mode = "cube";
            ss >> b.binding >> b.in >> b.dtype >> b.aniso;
            if (b.dtype == "bc6h") ss >> b.out;            // путь к SPIR-V кодировщика
            bufs.push_back(b);
        } else throw std::runtime_error("неизвестная директива: " + cmd);
    }
    std::sort(bufs.begin(), bufs.end(), [](auto& a, auto& b) { return a.binding < b.binding; });

    std::vector<Buffer> dev(bufs.size()), host(bufs.size());
    std::vector<CubeImage> cubes(bufs.size());
    std::vector<NpyArray> inputs(bufs.size());
    std::vector<VkDescriptorType> types(bufs.size(), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    for (size_t i = 0; i < bufs.size(); ++i) {
        auto& b = bufs[i];
        if (b.binding != i) throw std::runtime_error("binding должны идти подряд с 0");
        if (b.mode == "cube") {
            NpyArray a = npy_read(b.in);
            if (a.dtype != "<f4" || a.shape.size() != 5 || a.shape[1] != 6 || a.shape[2] != a.shape[3])
                throw std::runtime_error("cube: ожидается float32 (листы, 6, N, N, C)");
            size_t C = a.shape[4], n = a.count() / C;
            std::vector<float> rgba(n * 4, 1.0f);
            const float* src = reinterpret_cast<const float*>(a.data.data());
            for (size_t t = 0; t < n; ++t)
                for (size_t k = 0; k < C && k < 4; ++k) rgba[4 * t + k] = src[C * t + k];
            VkFormat fmt = b.dtype == "rgba16f" ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R32G32B32A32_SFLOAT;
            cubes[i] = ctx.create_cube(rgba.data(), uint32_t(a.shape[0]), uint32_t(a.shape[2]), fmt, b.aniso);
            if (b.dtype == "bc6h") {
                cubes[i] = ctx.compress_bc6h(cubes[i], read_spirv(b.out), b.aniso);
                b.out.clear();
            }
            types[i] = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            continue;
        }
        size_t bytes;
        if (!b.in.empty()) {
            inputs[i] = npy_read(b.in);
            if (inputs[i].dtype == "<f8") throw std::runtime_error("float64 на вход GPU не подаётся: " + b.in);
            bytes = inputs[i].data.size();
            if (b.mode == "inout") { b.dtype = inputs[i].dtype.substr(1); b.shape = inputs[i].shape; }
        } else {
            size_t n = 1;
            for (size_t s : b.shape) n *= s;
            bytes = n * 4;
        }
        dev[i] = ctx.create_buffer(bytes, false);
        host[i] = ctx.create_buffer(bytes, true);
        if (!b.in.empty()) {
            std::memcpy(host[i].mapped, inputs[i].data.data(), bytes);
            ctx.copy_buffer(host[i], dev[i], bytes);
        } else {
            std::memset(host[i].mapped, 0, bytes);
            ctx.copy_buffer(host[i], dev[i], bytes);
        }
    }

    auto code = read_spirv(spv);
    Pipeline p = ctx.create_pipeline(code, entry, types, static_cast<uint32_t>(push.size()));
    std::vector<Binding> binds(bufs.size());
    for (size_t i = 0; i < bufs.size(); ++i) {
        if (bufs[i].mode == "cube") binds[i].cube = &cubes[i];
        else binds[i].buffer = &dev[i];
    }
    auto times = ctx.dispatch(p, binds, push.data(), static_cast<uint32_t>(push.size()), g[0], g[1], g[2], repeat);

    for (size_t i = 0; i < bufs.size(); ++i) {
        auto& b = bufs[i];
        if (b.out.empty() || b.mode == "cube") continue;
        ctx.copy_buffer(dev[i], host[i], dev[i].size);
        npy_write(b.out, "<" + (b.dtype.empty() ? std::string("f4") : b.dtype), b.shape, host[i].mapped, dev[i].size);
    }

    if (!report.empty()) {
        std::ofstream r(report);
        r << "{\n  \"device\": \"" << json_escape(ctx.device_name()) << "\",\n  \"driver\": \"" << ctx.driver_version()
          << "\",\n  \"entry\": \"" << json_escape(entry) << "\",\n  \"times_ns\": [";
        for (size_t i = 0; i < times.size(); ++i) r << (i ? ", " : "") << times[i];
        r << "],\n  \"stats\": {";
        for (size_t i = 0; i < p.stats.size(); ++i)
            r << (i ? ", " : "") << "\n    \"" << json_escape(p.stats[i].first) << "\": \"" << json_escape(p.stats[i].second) << "\"";
        r << "\n  }\n}\n";
    }
    ctx.destroy_pipeline(p);
    for (auto& b : dev) ctx.destroy_buffer(b);
    for (auto& c : cubes) ctx.destroy_cube(c);
    for (auto& b : host) ctx.destroy_buffer(b);
    return 0;
}
