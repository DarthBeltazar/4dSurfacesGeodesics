// geo4d_gpu export-graph <сцена.json> <каталог> [--res WxH] [--cube N] [--fov F] [--vsync] [--fullscreen]
//                         [--compile] [--threads N] [--spv-dir DIR]
// Граф окна без Python (DD-010, шаги 1–2): тот же app_graph.txt / app_graph_dyn.txt и .npy, что пишет
// `app.py --graph-only <каталог>` (кроме директивы rebuild — перегенерации через Python). Пути SPIR-V — ключи кэша
// gpu/build/spv, как у geo4d.gpu.compile_shader; --compile — собрать недостающие в процессе (Slang API, N потоков),
// --spv-dir — другой каталог кэша (проверка: SPIR-V из API = из slangc.exe). Печатает время по этапам (JSON).
#include "export_graph.hpp"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

#include "codegen.hpp"
#include "npy.hpp"
#include "scene_graph.hpp"
#include "shader_compiler.hpp"

#include <nlohmann/json.hpp>

#include <cmath>

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
    bool do_compile = false;
    int threads = 0;
    std::string spv_dir;
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
        else if (a == "--legacy-codegen") o.legacy_codegen = true;   // виды scenes.slang и MF_* (как Python)
        else if (a == "--bake-params") o.bake_params = true;         // числа графа — литералами (DD-011 §5)
        else if (a == "--manifold-slots") o.manifold_slots = true;   // числа многообразия — слоты (DD-011 §5)
        else if (a == "--compile") do_compile = true;
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--spv-dir") spv_dir = next();
        else throw std::runtime_error("неизвестный параметр " + a);
    }
    auto t0 = std::chrono::steady_clock::now();
    Dirs d = default_dirs(argv[0]);
    if (!spv_dir.empty()) {
        fs::create_directories(from_u8(spv_dir));
        d.spv = fs::canonical(from_u8(spv_dir));
    }
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
    if (!g.prev_lines.empty()) write_lines(td / "app_graph_prev.txt", g.prev_lines);   // DD-011 §6
    if (!g.prev_obj_lines.empty()) write_lines(td / "app_graph_prev_obj.txt", g.prev_obj_lines);
    auto t3 = std::chrono::steady_clock::now();
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    std::string js = "{\"scene_ms\": " + std::to_string(ms(t0, t1)) + ", \"env_ms\": " + std::to_string(ms(t1, t2)) +
                     ", \"graph_ms\": " + std::to_string(ms(t2, t3)) + ", \"shaders\": " + std::to_string(g.shaders.size());
    int failed = 0;
    if (do_compile) {
        ShaderCompiler comp(threads);
        auto r = comp.compile(g.shaders);
        auto t4 = std::chrono::steady_clock::now();
        js += ", \"compile_ms\": " + std::to_string(ms(t3, t4)) + ", \"kernels\": [";
        for (size_t i = 0; i < r.size(); ++i) {
            if (!r[i].ok) {
                ++failed;
                std::fprintf(stderr, "%s %s: %s\n", u8(g.shaders[i].source.filename()).c_str(), g.shaders[i].entry.c_str(),
                             r[i].log.c_str());
            }
            js += std::string(i ? ", " : "") + "{\"entry\": \"" + g.shaders[i].entry + "\", \"ok\": " +
                  (r[i].ok ? "true" : "false") + ", \"cached\": " + (r[i].cached ? "true" : "false") +
                  ", \"s\": " + std::to_string(r[i].seconds) + ", \"worker\": " + std::to_string(r[i].worker) + "}";
        }
        js += "], \"session_s\": [";
        auto ss = comp.session_seconds();
        for (size_t i = 0; i < ss.size(); ++i) js += (i ? ", " : "") + std::to_string(ss[i]);
        js += "]";
    }
    std::printf("%s}\n", js.c_str());
    return failed ? 1 : 0;
}

int run_eval_graph(int argc, char** argv) {
    if (argc < 5) throw std::runtime_error("использование: geo4d_gpu eval-graph <сцена> <вход.json> <выход.json>");
    Dirs d = default_dirs(argv[0]);
    Scene sc = load_scene(argv[2], false);
    sc.bake = true;                               // заголовок для ядер вне окна (трассировка тестов): числа — литералами
    const Manifold& m = sc.mf;
    // выражение многообразия: собранное по графу или формула ноды из библиотеки с её параметрами
    ExprPtr e = m.expr;
    std::map<std::string, double> vals = m.expr_vals;
    if (!e) {
        const NodeDef* def = node_library().find(sc.mf_node);
        std::vector<std::string> names;
        for (auto& [k, v] : sc.mf_values) names.push_back(k);
        e = parse_expr(def->formula, names);
        vals = sc.mf_values;
    }
    std::ifstream fi(argv[3], std::ios::binary);
    if (!fi) throw std::runtime_error(std::string("не открыть ") + argv[3]);
    const nlohmann::json in = nlohmann::json::parse(fi);
    nlohmann::json out, f = nlohmann::json::array(), g = nlohmann::json::array(), H = nlohmann::json::array(),
                         proj = nlohmann::json::array();
    for (auto& pt : in.value("points", nlohmann::json::array())) {
        Vec4 p{pt[0].get<double>(), pt[1].get<double>(), pt[2].get<double>(), pt[3].get<double>()};
        EvalResult r = eval_expr(e, vals, p);
        f.push_back(r.f);
        g.push_back(r.g);
        H.push_back(r.H);
        proj.push_back(project(m, p));
    }
    out["f"] = f;
    out["g"] = g;
    out["H"] = H;
    out["project"] = proj;
    nlohmann::json centers = nlohmann::json::array();
    for (auto& c : m.flat.centers) centers.push_back(c);
    out["meta"] = {{"kind", m.kind}, {"scale", m.scale}, {"exit", m.exit}, {"compact", m.compact},
                   {"flat", m.flat.has}, {"sheets", m.flat.sheets}, {"centers", centers}, {"rho0", m.flat.rho0},
                   {"s_max", m.has_s_max ? m.s_max : 1000.0}, {"legacy", m.legacy()}};
    // объекты: g и ∇g в тех же точках, ограничивающая сфера
    nlohmann::json objs = nlohmann::json::array();
    for (auto& o : sc.objs) {
        ExprPtr ge = o.expr;
        std::map<std::string, double> gv = sc.slots.values();   // составная форма: числа — в слотах
        if (!ge) {
            const NodeDef* def = node_library().find(o.node);
            std::string fs = def->formula;
            for (auto& p : def->params)
                if (p.type == "int") {
                    const std::string key = "{" + p.name + "}";
                    for (size_t at; (at = fs.find(key)) != std::string::npos;)
                        fs.replace(at, key.size(), std::string(1, "xyzw"[int(o.values.at(p.name)) & 3]));
                }
            std::vector<std::string> names;
            for (auto& [k, x] : o.values) names.push_back(k);
            ge = parse_expr(fs, names);
            gv = o.values;
        }
        nlohmann::json of = nlohmann::json::array(), og = nlohmann::json::array();
        for (auto& pt : in.value("points", nlohmann::json::array())) {
            EvalResult r = eval_expr(ge, gv, Vec4{pt[0].get<double>(), pt[1].get<double>(), pt[2].get<double>(), pt[3].get<double>()});
            of.push_back(r.f);
            og.push_back(r.g);
        }
        objs.push_back({{"name", o.name}, {"kind", o.kind}, {"f", of}, {"g", og}, {"center", o.center},
                        {"radius", std::isinf(o.bound()) ? -1.0 : o.bound()}});
    }
    out["objects"] = objs;
    out["mf_scene"] = mf_header(sc);
    out["include"] = u8(write_header(d, sc, false, true));
    std::ofstream(argv[4], std::ios::binary) << out.dump();
    return 0;
}
