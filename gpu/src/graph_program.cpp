// Разбор файла графа. Директивы (по строке, '#' — комментарий):
//   buffer <имя> zero <байт> | in <npy> | export <байт>   (export — память для импорта в CUDA, DD-007)
//   cube <имя> <npy> rgba16f|rgba32f|bc6h <aniso> [encoder.spv]
//   pipeline <имя> <spv> <entry> <привязки по порядку: имена буферов/кубмап...>
//   push <конвейер> <тип значение | iter>...   push-константы конвейера (iter — номер итерации цикла)
//   section <имя>                                последующие шаги — в секции (до следующей section)
//   dispatch <конвейер> <gx> <gy> <gz> | indirect <конвейер> <буфер> <смещение байт>
//   loop <n> ... endloop (без вложенности) | mark <метка>
//   остальное (repeat, output, report, app, denoise, ...) — в extra() для вызывающего
#include "graph_program.hpp"

#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "npy.hpp"

namespace {

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

GraphProgram::GraphProgram(VkContext& ctx, const std::string& path) : ctx_(ctx) {
    std::ifstream jf(path);
    if (!jf) throw std::runtime_error("нет файла графа: " + path);
    std::map<std::string, std::pair<std::string, std::string>> pipe_src;
    std::string section, line;
    while (std::getline(jf, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string cmd;
        ss >> cmd;
        if (cmd == "buffer") {
            std::string name, mode;
            ss >> name >> mode;
            if (mode == "zero" || mode == "export") {
                size_t bytes;
                ss >> bytes;
                // export без внешней памяти (macOS, старый драйвер) — обычный буфер; денойзер тогда отключается
                bool exp = mode == "export" && ctx.external_memory();
                bufs_[name] = exp ? ctx.create_exportable_buffer(bytes) : ctx.create_buffer(bytes, false);
                Buffer h = ctx.create_buffer(bytes, true);
                std::memset(h.mapped, 0, bytes);
                ctx.copy_buffer(h, bufs_[name], bytes);
                ctx.destroy_buffer(h);
            } else if (mode == "in") {
                std::string p;
                ss >> p;
                NpyArray a = npy_read(p);
                if (a.dtype == "<f8") throw std::runtime_error("float64 на вход GPU не подаётся: " + p);
                bufs_[name] = ctx.create_buffer(a.data.size(), false);
                Buffer h = ctx.create_buffer(a.data.size(), true);
                std::memcpy(h.mapped, a.data.data(), a.data.size());
                ctx.copy_buffer(h, bufs_[name], a.data.size());
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
            cubes_[name] = ctx.create_cube(rgba.data(), uint32_t(a.shape[0]), uint32_t(a.shape[2]), vf, aniso);
            if (fmt == "bc6h") cubes_[name] = ctx.compress_bc6h(cubes_[name], load_spirv(enc), aniso);
        } else if (cmd == "pipeline") {
            std::string name, spv, entry, b;
            ss >> name >> spv >> entry;
            std::vector<std::string> names;
            while (ss >> b) names.push_back(b);
            binds_[name] = names;
            pipe_src[name] = {spv, entry};
        } else if (cmd == "push") {
            std::string name, t;
            ss >> name;
            auto& v = pushes_[name];
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
        } else if (cmd == "section") {
            ss >> section;
            sections_[section];
        } else if (cmd == "dispatch") {
            Op o{Op::Dispatch};
            ss >> o.pipe >> o.g[0] >> o.g[1] >> o.g[2];
            sections_[section].push_back(o);
        } else if (cmd == "indirect") {
            Op o{Op::Indirect};
            ss >> o.pipe >> o.buf >> o.offset;
            sections_[section].push_back(o);
        } else if (cmd == "loop") {
            Op o{Op::Loop};
            ss >> o.count;
            sections_[section].push_back(o);
        } else if (cmd == "endloop") {
            sections_[section].push_back(Op{Op::EndLoop});
        } else if (cmd == "mark") {
            Op o{Op::Mark};
            ss >> o.label;
            sections_[section].push_back(o);
        } else if (cmd == "repeat") {
            ss >> repeat_;
        } else {
            std::vector<std::string> words{cmd};
            for (std::string w; ss >> w;) words.push_back(w);
            extra_.push_back(words);
        }
    }
    for (auto& [name, binds] : binds_) {
        std::vector<VkDescriptorType> types;
        for (auto& b : binds) {
            if (!cubes_.count(b) && !bufs_.count(b)) throw std::runtime_error("конвейер " + name + ": нет ресурса " + b);
            types.push_back(cubes_.count(b) ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        }
        auto [spv, entry] = pipe_src[name];
        pipes_[name] = ctx.create_pipeline(load_spirv(spv), entry, types, uint32_t(pushes_[name].size() * 4));
    }
}

GraphProgram::~GraphProgram() {
    for (auto& [n, p] : pipes_) ctx_.destroy_pipeline(p);
    for (auto& [n, b] : bufs_) ctx_.destroy_buffer(b);
    for (auto& [n, c] : cubes_) ctx_.destroy_cube(c);
}

void GraphProgram::set_push_f32(const std::string& pipe, size_t idx, float v) {
    auto& it = pushes_.at(pipe).at(idx);
    std::memcpy(it.bytes, &v, 4);
}

void GraphProgram::set_push_u32(const std::string& pipe, size_t idx, uint32_t v) {
    auto& it = pushes_.at(pipe).at(idx);
    std::memcpy(it.bytes, &v, 4);
}

std::vector<std::pair<std::string, std::string>> GraphProgram::pipeline_stats() const {
    std::vector<std::pair<std::string, std::string>> out;
    for (auto& [name, p] : pipes_)
        for (auto& [k, v] : p.stats) out.emplace_back(name + ":" + k, v);
    return out;
}

std::vector<GraphStep> GraphProgram::steps(const std::string& section) const {
    auto sit = sections_.find(section);
    if (sit == sections_.end()) throw std::runtime_error("нет секции графа: " + section);
    const auto& ops = sit->second;
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
        s.pipe = &pipes_.at(f.op->pipe);
        for (auto& b : binds_.at(f.op->pipe)) {
            Binding bd;
            if (cubes_.count(b)) bd.cube = const_cast<CubeImage*>(&cubes_.at(b));
            else bd.buffer = const_cast<Buffer*>(&bufs_.at(b));
            s.bindings.push_back(bd);
        }
        auto pit = pushes_.find(f.op->pipe);
        if (pit != pushes_.end())
            for (auto& it : pit->second) {
                uint8_t b[4];
                if (it.iter) std::memcpy(b, &f.iter, 4);
                else std::memcpy(b, it.bytes, 4);
                s.push.insert(s.push.end(), b, b + 4);
            }
        if (f.op->kind == Op::Indirect) { s.indirect = const_cast<Buffer*>(&bufs_.at(f.op->buf)); s.offset = f.op->offset; }
        else { s.g[0] = f.op->g[0]; s.g[1] = f.op->g[1]; s.g[2] = f.op->g[2]; }
        steps.push_back(s);
    }
    return steps;
}
