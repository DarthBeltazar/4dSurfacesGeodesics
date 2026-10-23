// Граф задач: несколько диспетчеризаций в одном командном буфере (wavefront, DD-004 п. 5).
//
// Директивы (по строке, '#' — комментарий):
//   buffer <имя> zero <байт> | in <npy>
//   cube <имя> <npy> rgba16f|rgba32f|bc6h <aniso> [encoder.spv]
//   pipeline <имя> <spv> <entry> <привязки по порядку: имена буферов/кубмап...>
//   push <конвейер> <тип значение | iter>...   push-константы конвейера (iter — номер итерации цикла)
//   dispatch <конвейер> <gx> <gy> <gz>
//   indirect <конвейер> <буфер> <смещение байт>
//   loop <n> ... endloop                         (без вложенности)
//   mark <метка>                                 timestamp; в отчёт — время между соседними метками
//   repeat <n>                                   выполнить граф n раз (отчёт: все прогоны)
//   output <буфер> <npy> <dtype> <dims...>
//   report <json>
#include "graph.hpp"

#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

#include "npy.hpp"

namespace {

struct PushItem {
    bool iter = false;
    uint8_t bytes[4] = {0, 0, 0, 0};
};

struct Op {
    enum Kind { Dispatch, Indirect, Loop, EndLoop, Mark } kind;
    std::string pipe, buf, label;
    uint32_t g[3] = {1, 1, 1};
    uint32_t offset = 0, count = 0;
};

std::vector<uint32_t> load_spirv(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("нет SPIR-V: " + path);
    size_t n = static_cast<size_t>(f.tellg());
    std::vector<uint32_t> code(n / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(code.data()), n);
    return code;
}

}  // namespace

int run_graph(VkContext& ctx, const std::string& path) {
    std::ifstream jf(path);
    if (!jf) throw std::runtime_error("нет файла графа: " + path);
    std::map<std::string, Buffer> bufs;
    std::map<std::string, CubeImage> cubes;
    std::map<std::string, Pipeline> pipes;
    std::map<std::string, std::vector<std::string>> pipe_binds;
    std::map<std::string, std::vector<PushItem>> pushes;
    std::vector<Op> ops;
    struct Out { std::string buf, path, dtype; std::vector<size_t> shape; };
    std::vector<Out> outs;
    std::string report;
    uint32_t repeat = 1;
    std::string line;
    while (std::getline(jf, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string cmd;
        ss >> cmd;
        if (cmd == "buffer") {
            std::string name, mode;
            ss >> name >> mode;
            if (mode == "zero") {
                size_t bytes;
                ss >> bytes;
                bufs[name] = ctx.create_buffer(bytes, false);
                Buffer h = ctx.create_buffer(bytes, true);
                std::memset(h.mapped, 0, bytes);
                ctx.copy_buffer(h, bufs[name], bytes);
                ctx.destroy_buffer(h);
            } else if (mode == "in") {
                std::string p;
                ss >> p;
                NpyArray a = npy_read(p);
                if (a.dtype == "<f8") throw std::runtime_error("float64 на вход GPU не подаётся: " + p);
                bufs[name] = ctx.create_buffer(a.data.size(), false);
                Buffer h = ctx.create_buffer(a.data.size(), true);
                std::memcpy(h.mapped, a.data.data(), a.data.size());
                ctx.copy_buffer(h, bufs[name], a.data.size());
                ctx.destroy_buffer(h);
            } else throw std::runtime_error("buffer: режим " + mode);
        } else if (cmd == "cube") {
            std::string name, p, fmt, enc;
            float aniso;
            ss >> name >> p >> fmt >> aniso;
            if (fmt == "bc6h") ss >> enc;
            NpyArray a = npy_read(p);
            size_t C = a.shape[4], n = a.count() / C;
            std::vector<float> rgba(n * 4, 1.0f);
            const float* src = reinterpret_cast<const float*>(a.data.data());
            for (size_t t = 0; t < n; ++t)
                for (size_t k = 0; k < C && k < 4; ++k) rgba[4 * t + k] = src[C * t + k];
            VkFormat vf = fmt == "rgba16f" ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R32G32B32A32_SFLOAT;
            cubes[name] = ctx.create_cube(rgba.data(), uint32_t(a.shape[0]), uint32_t(a.shape[2]), vf, aniso);
            if (fmt == "bc6h") cubes[name] = ctx.compress_bc6h(cubes[name], load_spirv(enc), aniso);
        } else if (cmd == "pipeline") {
            std::string name, spv, entry, b;
            ss >> name >> spv >> entry;
            std::vector<std::string> names;
            while (ss >> b) names.push_back(b);
            pipe_binds[name] = names;
            pipes[name].module = VK_NULL_HANDLE;               // создаётся после push (нужен размер)
            pipes[name].stats.push_back({"spv", spv});
            pipes[name].stats.push_back({"entry", entry});
        } else if (cmd == "push") {
            std::string name, t;
            ss >> name;
            auto& v = pushes[name];
            while (ss >> t) {
                PushItem it;
                if (t == "iter") { it.iter = true; v.push_back(it); continue; }
                if (t == "f32") { float x; ss >> x; std::memcpy(it.bytes, &x, 4); }
                else if (t == "u32") { uint32_t x; ss >> x; std::memcpy(it.bytes, &x, 4); }
                else if (t == "i32") { int32_t x; ss >> x; std::memcpy(it.bytes, &x, 4); }
                else throw std::runtime_error("push: тип " + t);
                if (ss.fail()) throw std::runtime_error("push: не разобрано: " + line);
                v.push_back(it);
            }
        } else if (cmd == "dispatch") {
            Op o{Op::Dispatch};
            ss >> o.pipe >> o.g[0] >> o.g[1] >> o.g[2];
            ops.push_back(o);
        } else if (cmd == "indirect") {
            Op o{Op::Indirect};
            ss >> o.pipe >> o.buf >> o.offset;
            ops.push_back(o);
        } else if (cmd == "loop") {
            Op o{Op::Loop};
            ss >> o.count;
            ops.push_back(o);
        } else if (cmd == "endloop") {
            ops.push_back(Op{Op::EndLoop});
        } else if (cmd == "mark") {
            Op o{Op::Mark};
            ss >> o.label;
            ops.push_back(o);
        } else if (cmd == "repeat") {
            ss >> repeat;
        } else if (cmd == "output") {
            Out o;
            ss >> o.buf >> o.path >> o.dtype;
            size_t d;
            while (ss >> d) o.shape.push_back(d);
            outs.push_back(o);
        } else if (cmd == "report") {
            ss >> report;
        } else throw std::runtime_error("неизвестная директива: " + cmd);
    }

    // конвейеры
    for (auto& [name, binds] : pipe_binds) {
        std::string spv = pipes[name].stats[0].second, entry = pipes[name].stats[1].second;
        std::vector<VkDescriptorType> types;
        for (auto& b : binds) types.push_back(cubes.count(b) ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        pipes[name] = ctx.create_pipeline(load_spirv(spv), entry, types, uint32_t(pushes[name].size() * 4));
    }
    // развёртка циклов
    struct Flat { const Op* op; uint32_t iter; };
    std::vector<Flat> flat;
    for (size_t i = 0; i < ops.size(); ++i) {
        if (ops[i].kind == Op::Loop) {
            size_t j = i + 1;
            while (j < ops.size() && ops[j].kind != Op::EndLoop) ++j;
            for (uint32_t it = 0; it < ops[i].count; ++it)
                for (size_t k = i + 1; k < j; ++k) flat.push_back({&ops[k], it});
            i = j;
        } else flat.push_back({&ops[i], 0});
    }
    std::vector<GraphStep> steps;
    for (auto& f : flat) {
        GraphStep s;
        if (f.op->kind == Op::Mark) { s.mark = f.op->label; steps.push_back(s); continue; }
        s.pipe = &pipes.at(f.op->pipe);
        for (auto& b : pipe_binds.at(f.op->pipe)) {
            Binding bd;
            if (cubes.count(b)) bd.cube = &cubes.at(b);
            else bd.buffer = &bufs.at(b);
            s.bindings.push_back(bd);
        }
        for (auto& it : pushes[f.op->pipe]) {
            uint8_t b[4];
            if (it.iter) std::memcpy(b, &f.iter, 4);
            else std::memcpy(b, it.bytes, 4);
            s.push.insert(s.push.end(), b, b + 4);
        }
        if (f.op->kind == Op::Indirect) { s.indirect = &bufs.at(f.op->buf); s.offset = f.op->offset; }
        else { s.g[0] = f.op->g[0]; s.g[1] = f.op->g[1]; s.g[2] = f.op->g[2]; }
        steps.push_back(s);
    }
    std::vector<std::vector<std::pair<std::string, double>>> runs;
    for (uint32_t r = 0; r < repeat; ++r) runs.push_back(ctx.run_graph(steps));

    for (auto& o : outs) {
        Buffer& b = bufs.at(o.buf);
        Buffer h = ctx.create_buffer(b.size, true);
        ctx.copy_buffer(b, h, b.size);
        npy_write(o.path, "<" + o.dtype, o.shape, h.mapped, b.size);
        ctx.destroy_buffer(h);
    }
    if (!report.empty()) {
        std::ofstream rf(report);
        rf << "{\n  \"device\": \"" << ctx.device_name() << "\",\n  \"driver\": \"" << ctx.driver_version() << "\",\n  \"runs\": [";
        for (size_t r = 0; r < runs.size(); ++r) {
            rf << (r ? ",\n    {" : "\n    {");
            for (size_t i = 0; i < runs[r].size(); ++i)
                rf << (i ? ", " : "") << "\"" << runs[r][i].first << "\": " << runs[r][i].second;
            rf << "}";
        }
        rf << "\n  ],\n  \"stats\": {";
        bool first = true;
        for (auto& [name, p] : pipes)
            for (auto& [k, v] : p.stats) {
                rf << (first ? "\n    " : ",\n    ") << "\"" << name << ":" << k << "\": \"" << v << "\"";
                first = false;
            }
        rf << "\n  }\n}\n";
    }
    for (auto& [n, p] : pipes) ctx.destroy_pipeline(p);
    for (auto& [n, b] : bufs) ctx.destroy_buffer(b);
    for (auto& [n, c] : cubes) ctx.destroy_cube(c);
    return 0;
}
