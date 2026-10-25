// Кодген окна (DD-010): заголовок pt_scene.slang (geo4d.gpu_pt.scene_header), дефайны (_defs, manifold_defines),
// ключи кэша SPIR-V (geo4d.gpu.compile_shader), строки буфера камеры (camera_buffer, object_rows, manifold_rows),
// push-константы (_push) и граф окна (geo4d.gpu_pt.app_graph) — строки графа и массивы в памяти.
// Форматы чисел — как в Python (%.9g, repr), чтобы заголовки, ключи и граф совпадали с Python байт в байт.
#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "environment.hpp"
#include "scene.hpp"

namespace geo4d {

namespace fs = std::filesystem;

constexpr int kObjBase = 8;                       // gpu_pt.OBJ_BASE
constexpr int kPtMaxObj = 32;                     // gpu_pt.PT_MAXOBJ
constexpr int kObjRows = 9 + 6 * kPtMaxObj;       // gpu_pt.OBJ_ROWS
constexpr int kMfBase = kObjBase + kObjRows;      // gpu_pt.MF_BASE
constexpr int kMfRows = 7;                        // gpu_pt.MF_ROWS
// DD-011 §5: слоты параметров графа — строки после строк многообразия (до kMaxSlotRows); дальше — байткод (шаг 5)
constexpr int kSlotBase = kMfBase + kMfRows;
constexpr int kMaxSlotRows = 256;
constexpr int kProgBase = kSlotBase + kMaxSlotRows;
constexpr int kSceneDynamic = 11;                 // gpu.SCENE_IDS["dynamic"]

std::string fmt9g(double v);                      // f"{v:.9g}"
std::string pyrepr(double v);                     // repr(float(v))
std::string qpath(const fs::path& p);             // gpu.qpath: в кавычках, если есть пробелы
std::string u8(const fs::path& p);                // путь в UTF-8 (родные разделители)
std::string posix(const fs::path& p);             // Path.as_posix()
fs::path from_u8(const std::string& s);           // путь из UTF-8

using Defines = std::vector<std::pair<std::string, std::string>>;   // имя → значение (как текст литерала Python)

// Каталоги (DD-010):
//  - установленная программа (zip): рядом с exe есть shaders/ — шейдеры и сцены рядом с exe, кэш SPIR-V и заголовков
//    сцен — в пользовательском кэше (каталог программы только для чтения);
//  - дерево разработки: корень — GEO4D_ROOT или exe → gpu/build → корень; кэш — gpu/build/spv и gpu/build/gen
//    (общий с geo4d.gpu.compile_shader).
struct Dirs {
    fs::path root, shaders, scenes, nodes, gen, spv;
    bool installed = false;
};
Dirs default_dirs(const char* argv0);
fs::path exe_path(const char* argv0);             // путь к exe от ОС (argv[0] — запасной)
// пользовательский кэш (SPIR-V установленной программы, VkPipelineCache): GEO4D_CACHE_DIR или %LOCALAPPDATA%\geo4d\cache
// (Linux/macOS — $XDG_CACHE_HOME/geo4d или ~/.cache/geo4d)
fs::path user_cache_dir();

// graph — многообразие и объекты из формул графа сцены (DD-011): SCENE_GRAPH, mf_scene.slang; иначе — как
// geo4d.gpu_pt (виды scenes.slang и дефайны MF_*; режим --legacy-codegen, эталон для сверки)
std::string scene_header(const Scene& sc, bool dynamic, bool graph = false);
std::string mf_header(const Scene& sc);           // mf_scene.slang: SCALE, sdf<T>, hcap
std::string slot_ref(int i);                      // слот i в Slang: gCam[kSlotBase + i / 4].<xyzw>
std::vector<float> slot_rows(const Scene& sc);    // ⌈слотов / 4⌉ × 4 (пусто при bake)
// DD-011 §5: всё, что зашито в шейдеры и граф быстрого пути (заголовки, дефайны, push многообразия, фон), — sha256;
// у той же структуры с другими числами слотов ключ тот же: правку применяет запись строк (без перекомпиляции)
std::string structure_key(const Scene& sc);
// gen/<sha256[:16]>/pt_scene.slang (+ mf_scene.slang при graph); CRLF, как Path.write_text
fs::path write_header(const Dirs& d, const Scene& sc, bool dynamic, bool graph = false);
Defines defines(const Scene& sc, int nj, bool dynamic, bool graph = false);

// шейдер графа: что компилировать и куда (shader_compiler.hpp)
struct ShaderJob {
    fs::path source;                              // gpu/shaders/<файл>.slang
    std::string entry;
    Defines defs;
    fs::path include;                             // каталог заголовка сцены (может быть пустым)
    fs::path out;                                 // путь SPIR-V в кэше
};

// geo4d.gpu.compile_shader: путь SPIR-V в кэше по ключу (sha256 исходников, include, json.dumps([source, entry, defines]))
class ShaderKeys {
public:
    explicit ShaderKeys(const Dirs& d);
    fs::path spv_path(const std::string& source, const std::string& entry, const Defines& defs,
                      const fs::path& include = {}) const;
    std::string key_json(const std::string& source, const std::string& entry, const Defines& defs) const;

private:
    Dirs d_;
    std::string shader_bytes_;                    // содержимое всех *.slang по порядку sorted(rglob)
};

struct PTParams {                                 // gpu_pt.GPUPT
    double tol_kappa = 3e-3, tol_max = 1e-2, c_hit = 0.05, k_cone = 0.25, k_lod = 0.0;
    int max_depth = 16, rr_depth = 3, nee = 1, mis = 0;
    double s_max = 1000.0;
    int n_max = 4000;
    double env_scale = 1.0;
    int seed = 1;
};

std::vector<float> camera_rows(const Camera& cam, double k_grad, double flight_r, int filter_mode, int cube_n);   // 8 × 4
std::vector<float> object_rows(const Scene& sc);                                                                 // kObjRows × 4
std::vector<float> manifold_rows(const Scene& sc, const PTParams& p = {});                                       // kMfRows × 4

// Массив для буфера графа (в export-graph — .npy)
struct Array {
    std::string dtype = "<f4";
    std::vector<size_t> shape;
    std::vector<float> data;
};

// Параметры окна, как у app.py (без режимов замера)
struct AppOptions {
    int width = 1280, height = 720;
    int cube = 512;
    double fov = 0;                               // 0 — из файла сцены
    bool vsync = false, fullscreen = false;
    bool legacy_codegen = false;                  // виды scenes.slang и MF_* (как Python) вместо формул графа
    bool bake_params = false;                     // числа графа — литералами вместо слотов (DD-011 §5: сверка, замер)
    bool manifold_slots = false;                  // числа многообразия — слоты (правка без перекомпиляции, кадр дороже)
};

struct AppGraph {
    std::vector<std::string> lines;               // app_graph.txt
    std::vector<std::string> dyn_lines;           // app_graph_dyn.txt
    std::vector<std::string> prev_lines;          // app_graph_prev.txt: предпросмотр на интерпретаторе (DD-011 §6)
    std::map<std::string, Array> arrays;          // имя файла (cam.npy, env_dyn.npy, …) → данные
    std::vector<ShaderJob> shaders;               // SPIR-V, на которые ссылаются графы (без повторов)
};

// geo4d.gpu_pt.app_graph(…, editor=…) для app.py без замера: быстрый граф и граф режима редактирования.
// work_dir — каталог графа (пути в директивах); scene_path — файл сцены (директива editor).
AppGraph app_graph(const Dirs& d, const Scene& sc, const AppOptions& o, const fs::path& work_dir,
                   const fs::path& scene_path, const EnvMap& procedural);

}  // namespace geo4d
