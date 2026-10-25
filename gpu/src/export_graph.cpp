// geo4d_gpu export-graph <сцена.json> <каталог> [--res WxH] [--cube N] [--fov F] [--vsync] [--fullscreen]
// Граф окна без Python (DD-010, шаг 1): тот же app_graph.txt / app_graph_dyn.txt и .npy, что пишет
// `app.py --graph-only <каталог>` (кроме директивы rebuild — перегенерации через Python). SPIR-V не компилируется:
// пути в графе — ключи кэша gpu/build/spv, как у geo4d.gpu.compile_shader (шаг 2 — компиляция в процессе).
#include "export_graph.hpp"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

#include "codegen.hpp"
#include "npy.hpp"

using namespace geo4d;

namespace {

void write_lines(const fs::path& p, const std::vector<std::string>& lines) {
    std::ofstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("не записать " + u8(p));
    for (auto& l : lines) {
#ifdef _WIN32
        // строки с переводом внутри (не бывает в графе) и CRLF, как Path.write_text в Windows
        f << l << "\r\n";
#else
        f << l << "\n";
#endif
    }
}

}  // namespace

int run_export_graph(int argc, char** argv) {
    if (argc < 4) throw std::runtime_error("использование: geo4d_gpu export-graph <сцена.json> <каталог> [--res WxH] [--cube N] [--fov F] [--vsync] [--fullscreen]");
    AppOptions o;
    for (int i = 4; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("нет значения у " + a);
            return argv[++i];
        };
        if (a == "--res") {
            std::string v = next();
            auto x = v.find('x');
            o.width = std::stoi(v.substr(0, x));
            o.height = std::stoi(v.substr(x + 1));
        } else if (a == "--cube") o.cube = std::stoi(next());
        else if (a == "--fov") o.fov = std::stod(next());
        else if (a == "--vsync") o.vsync = true;
        else if (a == "--fullscreen") o.fullscreen = true;
        else throw std::runtime_error("неизвестный параметр " + a);
    }
    auto t0 = std::chrono::steady_clock::now();
    Dirs d = default_dirs(argv[0]);
    fs::path scene_path = fs::canonical(from_u8(argv[2]));
    fs::path td = from_u8(argv[3]);
    fs::create_directories(td);
    td = fs::canonical(td);
    Scene sc = load_scene(u8(scene_path));
    auto t1 = std::chrono::steady_clock::now();
    EnvMap env = procedural_env(o.cube);
    auto t2 = std::chrono::steady_clock::now();
    AppGraph g = app_graph(d, sc, o, td, scene_path, env);
    for (auto& [name, a] : g.arrays) npy_write(u8(td / name), a.dtype, a.shape, a.data.data(), a.data.size() * sizeof(float));
    write_lines(td / "app_graph.txt", g.lines);
    write_lines(td / "app_graph_dyn.txt", g.dyn_lines);
    auto t3 = std::chrono::steady_clock::now();
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    std::printf("граф: %s\nсцена %.1f мс, фон %.1f мс, граф и файлы %.1f мс\n", u8(td / "app_graph.txt").c_str(), ms(t0, t1),
                ms(t1, t2), ms(t2, t3));
    return 0;
}
