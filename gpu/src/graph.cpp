// Граф задач: несколько диспетчеризаций в одном командном буфере (wavefront, DD-004 п. 5).
// Разбор — GraphProgram (graph_program.cpp). Здесь — однократный прогон шагов вне секций:
//   repeat <n>                                   выполнить граф n раз (отчёт: все прогоны)
//   output <буфер> <npy> <dtype> <dims...>
//   report <json>
#include "graph.hpp"

#include <fstream>
#include <stdexcept>

#include "graph_program.hpp"
#include "npy.hpp"

int run_graph(VkContext& ctx, const std::string& path) {
    GraphProgram prog(ctx, path);
    struct Out { std::string buf, path, dtype; std::vector<size_t> shape; };
    std::vector<Out> outs;
    std::string report;
    for (auto& w : prog.extra()) {
        if (w[0] == "output") {
            Out o{w.at(1), w.at(2), w.at(3), {}};
            for (size_t i = 4; i < w.size(); ++i) o.shape.push_back(std::stoull(w[i]));
            outs.push_back(o);
        } else if (w[0] == "report") report = w.at(1);
        else throw std::runtime_error("неизвестная директива: " + w[0]);
    }
    std::vector<std::vector<std::pair<std::string, double>>> runs;
    for (uint32_t r = 0; r < prog.repeat(); ++r) runs.push_back(prog.run(""));

    for (auto& o : outs) {
        Buffer& b = prog.buffer(o.buf);
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
        for (auto& [k, v] : prog.pipeline_stats()) {
            if (k.find(":spv") != std::string::npos || k.find(":entry") != std::string::npos) continue;
            rf << (first ? "\n    " : ",\n    ") << "\"" << k << "\": \"" << v << "\"";
            first = false;
        }
        rf << "\n  }\n}\n";
    }
    return 0;
}
