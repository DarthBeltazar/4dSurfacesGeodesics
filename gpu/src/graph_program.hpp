// Загружаемая программа графа (DD-004 п. 5, DD-007): буферы, кубмапы, конвейеры, push-константы и шаги по секциям.
// Используется подкомандами graph (однократный прогон) и app (цикл кадра с изменением push-констант).
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "vk_context.hpp"

class GraphProgram {
public:
    GraphProgram(VkContext& ctx, const std::string& path);
    ~GraphProgram();
    GraphProgram(const GraphProgram&) = delete;
    GraphProgram& operator=(const GraphProgram&) = delete;

    // шаги секции ("" — шаги вне секций) с текущими push-константами
    std::vector<GraphStep> steps(const std::string& section) const;
    std::vector<std::pair<std::string, double>> run(const std::string& section) { return ctx_.run_graph(steps(section)); }
    bool has_section(const std::string& s) const { return sections_.count(s) > 0; }

    // изменить 4-байтовую push-константу idx конвейера pipe
    void set_push_f32(const std::string& pipe, size_t idx, float v);
    void set_push_u32(const std::string& pipe, size_t idx, uint32_t v);

    Buffer& buffer(const std::string& name) { return bufs_.at(name); }
    bool has_buffer(const std::string& name) const { return bufs_.count(name) > 0; }

    // директивы, не относящиеся к графу (app, denoise, output, report, repeat)
    const std::vector<std::vector<std::string>>& extra() const { return extra_; }
    uint32_t repeat() const { return repeat_; }
    std::vector<std::pair<std::string, std::string>> pipeline_stats() const;

private:
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
    VkContext& ctx_;
    std::map<std::string, Buffer> bufs_;
    std::map<std::string, CubeImage> cubes_;
    std::map<std::string, Pipeline> pipes_;
    std::map<std::string, std::vector<std::string>> binds_;
    std::map<std::string, std::vector<PushItem>> pushes_;
    std::map<std::string, std::vector<Op>> sections_;
    std::vector<std::vector<std::string>> extra_;
    uint32_t repeat_ = 1;
};
