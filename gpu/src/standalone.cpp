// Окно без Python (DD-010, шаг 3): geo4d_gpu [сцена.json] [--res WxH] [--cube N] [--fov F] [--vsync] [--fullscreen]
//                                           [--threads N] [--work-dir DIR] [--graph-only DIR] [--extra FILE]
//                                           [--spv-dir DIR]
// Граф окна строится в памяти (codegen), шейдеры собираются в процессе (shader_compiler, кэш gpu/build/spv в дереве
// разработки), граф и массивы регистрируются в memfs, окно — тот же run_app; перегенерации редактора (структура
// объектов, разрешение, многообразие при выходе из режима редактирования) — та же функция в потоке, без Python.
// Без сцены — scenes/two_wormholes.json. --graph-only DIR — записать граф на диск и выйти (как app.py --graph-only).
// --extra FILE — строки директив, дописываемые к графу (сценарий замера и проверок: hidden, bench, dump, resize,
// editmode, edit, loadscene, …; см. app.cpp), — как app.py --bench дописывает их к своему графу.
#include "standalone.hpp"

#include <chrono>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

#include "app.hpp"
#include "codegen.hpp"
#include "memfs.hpp"
#include "npy.hpp"
#include "shader_compiler.hpp"

#ifdef _WIN32
#include <process.h>
#define GEO4D_GETPID _getpid
#else
#include <unistd.h>
#define GEO4D_GETPID getpid
#endif

using namespace geo4d;

namespace {

struct Builder {
    Dirs d;
    int threads = 0;
    std::mutex mtx;                               // окно и поток перегенерации
    std::map<int, EnvMap> envs;                   // процедурный фон по размеру грани (строится один раз)
    std::unique_ptr<ShaderCompiler> comp;

    const EnvMap& env(int n) {
        auto it = envs.find(n);
        if (it == envs.end()) it = envs.emplace(n, procedural_env(n)).first;
        return it->second;
    }

    // сцена → граф в out_dir (memfs; при to_disk — ещё и файлы); false — ошибка в err
    bool build(const std::string& scene_json, const AppOptions& o, const fs::path& out_dir, bool to_disk, std::string& err) {
        std::lock_guard<std::mutex> lock(mtx);
        auto t0 = std::chrono::steady_clock::now();
        try {
            Scene sc = load_scene(scene_json);
            const EnvMap& e = env(o.cube);
            auto t1 = std::chrono::steady_clock::now();
            AppGraph g = app_graph(d, sc, o, out_dir, fs::canonical(from_u8(scene_json)), e);
            auto t2 = std::chrono::steady_clock::now();
            if (!comp) comp = std::make_unique<ShaderCompiler>(threads);
            auto r = comp->compile(g.shaders);
            auto t3 = std::chrono::steady_clock::now();
            int compiled = 0;
            for (size_t i = 0; i < r.size(); ++i) {
                if (!r[i].ok) {
                    err = g.shaders[i].entry + ": " + r[i].log;
                    return false;
                }
                compiled += r[i].cached ? 0 : 1;
            }
            auto text = [](const std::vector<std::string>& L) {
                std::string s;
                for (auto& l : L) s += l + "\n";
                return s;
            };
            for (auto& [name, a] : g.arrays) {
                NpyArray n;
                n.dtype = a.dtype;
                n.shape = a.shape;
                n.data.resize(a.data.size() * sizeof(float));
                std::memcpy(n.data.data(), a.data.data(), n.data.size());
                if (to_disk) npy_write(u8(out_dir / name), n.dtype, n.shape, n.data.data(), n.data.size());
                memfs::put_npy(u8(out_dir / name), std::move(n));
            }
            memfs::put_text(u8(out_dir / "app_graph_dyn.txt"), text(g.dyn_lines));
            memfs::put_text(u8(out_dir) + "/app_graph.txt", text(g.lines));     // путь, который ждёт редактор
            memfs::put_text(u8(out_dir / "app_graph.txt"), text(g.lines));
            if (to_disk) {
                std::ofstream(out_dir / "app_graph.txt", std::ios::binary) << text(g.lines);
                std::ofstream(out_dir / "app_graph_dyn.txt", std::ios::binary) << text(g.dyn_lines);
            }
            auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
            std::printf("standalone_build {\"scene_env_ms\": %.1f, \"graph_ms\": %.1f, \"compile_ms\": %.1f, \"compiled\": %d, "
                        "\"shaders\": %zu}\n", ms(t0, t1), ms(t1, t2), ms(t2, t3), compiled, g.shaders.size());
            std::fflush(stdout);
            return true;
        } catch (const std::exception& ex) {
            err = ex.what();
            return false;
        }
    }
};

AppOptions parse_args(const std::vector<std::string>& args, AppOptions o, int* threads = nullptr,
                      std::string* work = nullptr, std::string* graph_only = nullptr, std::string* extra = nullptr,
                      std::string* spv = nullptr) {
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= args.size()) throw std::runtime_error("нет значения у " + a);
            return args[++i];
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
        else if (a == "--legacy-codegen") o.legacy_codegen = true;   // виды scenes.slang и MF_* (как Python)
        else if (a == "--threads" && threads) *threads = std::stoi(next());
        else if (a == "--work-dir" && work) *work = next();
        else if (a == "--graph-only" && graph_only) *graph_only = next();
        else if (a == "--extra" && extra) *extra = next();
        else if (a == "--spv-dir" && spv) *spv = next();
        else throw std::runtime_error("неизвестный параметр " + a);
    }
    return o;
}

}  // namespace

bool is_standalone_args(int argc, char** argv) {
    if (argc < 2) return true;
    std::string a = argv[1];
    return a.rfind("--", 0) == 0 || (a.size() > 5 && a.substr(a.size() - 5) == ".json");
}

int run_standalone(int argc, char** argv) {
    auto builder = std::make_shared<Builder>();
    builder->d = default_dirs(argv[0]);
    std::vector<std::string> args(argv + 1, argv + argc);
    std::string scene = u8(builder->d.scenes / "two_wormholes.json");
    if (!args.empty() && args[0].rfind("--", 0) != 0) {
        scene = args[0];
        args.erase(args.begin());
    }
    std::string work, graph_only, extra, spv;
    AppOptions o = parse_args(args, AppOptions{}, &builder->threads, &work, &graph_only, &extra, &spv);
    if (!spv.empty()) {                           // другой кэш SPIR-V (замер холодного старта)
        fs::create_directories(from_u8(spv));
        builder->d.spv = fs::canonical(from_u8(spv));
    }
    fs::path wd = !graph_only.empty() ? from_u8(graph_only)
                  : !work.empty()     ? from_u8(work)
                                      : fs::temp_directory_path() / ("geo4d-" + std::to_string(GEO4D_GETPID()));
    fs::create_directories(wd);
    wd = fs::canonical(wd);
    std::string err;
    if (!builder->build(scene, o, wd, !graph_only.empty(), err)) throw std::runtime_error(err);
    if (!graph_only.empty()) {
        std::printf("граф: %s\n", u8(wd / "app_graph.txt").c_str());
        return 0;
    }
    if (!extra.empty()) {                         // директивы сценария — к исходному графу (перезагрузки их переносят)
        std::string g = memfs::read_text(u8(wd / "app_graph.txt")) + memfs::read_text(extra);
        if (!g.empty() && g.back() != '\n') g += "\n";
        memfs::put_text(u8(wd / "app_graph.txt"), g);
    }
    // аргументы перегенерации — как у app.py (редактор меняет --res насовсем)
    std::vector<std::string> rargs = {"--res", std::to_string(o.width) + "x" + std::to_string(o.height), "--cube",
                                      std::to_string(o.cube)};
    if (o.vsync) rargs.push_back("--vsync");
    RebuildFn fn = [builder, o](const std::string& json, const std::vector<std::string>& a, const std::string& out,
                                std::string& e) {
        AppOptions oo = parse_args(a, AppOptions{});
        oo.fullscreen = false;                    // окно уже в своём режиме; fov — из файла сцены, как у Python
        return builder->build(json, oo, from_u8(out), false, e);
    };
    return run_app(u8(wd / "app_graph.txt"), fn, rargs);
}
