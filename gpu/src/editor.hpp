// Редактор сцен (DD-008): модель объектов, упаковка строк для GPU (зеркало geo4d.gpu_pt.object_rows), правило
// NEE-источников M34 (зеркало geo4d.pathtrace.nee_lights), панели ImGui, запись JSON и фоновая перегенерация графа
// в Python при смене многообразия. DD-009: чтение JSON сцены и строки многообразия (зеркала geo4d.scenefile.build и
// geo4d.gpu_pt.manifold_rows) — сцена с другим многообразием сразу открывается в режиме редактирования.
#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

constexpr int kObjBase = 8;                       // первая строка объектов в буфере cam (после 8 строк камеры)
constexpr int kMaxObj = 32;                       // PT_MAXOBJ
constexpr int kObjRows = 9 + 6 * kMaxObj;         // geo4d.gpu_pt.OBJ_ROWS
constexpr int kMfBase = kObjBase + kObjRows;      // geo4d.gpu_pt.MF_BASE: строки многообразия (scenes.slang SCENE_DYNAMIC)
constexpr int kMfRows = 7;

struct EdObject {
    std::string name = "ball";
    int geom = 0;                                 // 0 — ball4, 1 — torus, 2 — plane, 3 — составная форма графа (M36)
    float c[4] = {0, 0, 0, 0};
    float a = 0.5f, b = 0.15f;                    // ball4: r = a; torus: R = a, r = b
    int axis = 2;                                 // plane: sign·(p[axis] − offset)
    float offset = 0, sign = 1;
    int mat = 0;                                  // 0 — diffuse, 1 — metal, 2 — emit
    float albedo[3] = {0.8f, 0.8f, 0.8f};
    float emit[3] = {0, 0, 0};
    float rough = 0.3f;
    bool on_M = false;
};

struct EdFlat {                                   // плоские зоны многообразия (M29)
    bool has = false;
    std::vector<float> sheets;
    std::vector<std::array<float, 3>> balls;
    float rho0 = 0;
};

std::vector<int> nee_lights(const std::vector<EdObject>& objs, const EdFlat& flat);
std::vector<float> pack_rows(const std::vector<EdObject>& objs, const EdFlat& flat);            // kObjRows × 4
std::vector<EdObject> unpack_rows(const float* rows, const std::vector<std::string>& names);
// структура, зашитая в быстрые шейдеры: (форма, ось полупространства) по порядку (geo4d.gpu_pt.structure_key)
std::vector<std::pair<int, int>> structure(const std::vector<EdObject>& objs);

struct ManifoldParam {
    std::string key;
    std::vector<float> v;                         // 1 или 3 числа (центры горловин)
};

// Многообразие по виду и параметрам (DD-009): строки для динамических шейдеров, плоские зоны (M29), есть ли фон.
// Недопустимые параметры (geo4d.scenefile.check_params) — ok = false и сообщение в err.
struct ManifoldInfo {
    bool ok = false;
    std::string err;
    std::vector<float> rows;                      // kMfRows × 4
    EdFlat flat;
    bool background = false;
    float scale = 1;
};
// env: "procedural" | "none" | "" (по умолчанию — по виду, как в geo4d.scenefile.build)
ManifoldInfo manifold_info(const std::string& kind, const std::vector<ManifoldParam>& params, const std::string& env);
std::vector<ManifoldParam> default_params(const std::string& kind);
// ключ «что зашито в быстрые шейдеры» по многообразию: вид, фон, параметры
std::string manifold_key(const std::string& kind, const std::vector<ManifoldParam>& params, const std::string& env);

// Данные графа для редактора (директивы app_graph): editor, manifold, flat, objname, rebuild
struct EditorConfig {
    std::string scene_path;                       // файл сцены (сохранение)
    std::string kind;
    std::vector<ManifoldParam> params;
    std::string env;
    float fov = 75, speed = 1;
    std::string python, script;                   // перегенерация: python script --scene … --graph-only …
    std::vector<std::string> rebuild_args;        // --res, --cube
    std::string work_dir;                         // каталог для JSON и графов перегенерации
    EdFlat flat;
    std::vector<std::string> names;
    bool enabled = false;
    bool graph_doc = false;                       // DD-011: сцена не выразима в version 1 — сохраняется граф (.g4d)
};
EditorConfig parse_editor_config(const std::vector<std::vector<std::string>>& extra);

struct CameraState {
    float x[4], f[4], r[4], u[4];
};

std::string scene_json(const std::string& kind, const std::vector<ManifoldParam>& params, const std::string& env,
                       const std::vector<EdObject>& objs, const CameraState& cam, float fov, float speed);

// Сцена из JSON (geo4d.scenefile): многообразие, объекты, подсказки камеры (без проекции на M — её делает GPU)
struct SceneDesc {
    std::string path, kind, env;
    std::vector<ManifoldParam> params;
    std::vector<EdObject> objs;
    CameraState cam{};                            // x и подсказки f, up (в u), right (в r)
    bool has_right = false;
    float fov = 75, speed = 1;
};
SceneDesc read_scene_json(const std::string& path);      // ошибки формата — исключение с сообщением

// M3 + касательный базис (как geo4d.camera.make_camera): n — нормаль M в точке, подсказки f, u, r (r — если has_r)
void tangent_basis(const float n[4], const float fh[4], const float uh[4], const float rh[4], bool has_r, float f[4],
                   float u[4], float r[4]);

// Что окно должно сделать после панелей
// разрешение рендера (DD-009): доля размера окна в пикселях (frac > 0) или фиксированное W×H
struct ResPreset {
    const char* label;
    float frac;
    uint32_t w, h;
};
extern const ResPreset kResPresets[];
extern const int kNumResPresets;
// W×H пресета для окна ww×wh (доля — не меньше 64 по каждой оси)
std::pair<uint32_t, uint32_t> preset_size(int i, uint32_t ww, uint32_t wh);

struct UiActions {
    int res_choice = -1;                          // выбран пресет разрешения (индекс kResPresets)
    bool toggle_fullscreen = false;
    bool toggle_preview = false;                  // DD-011 §6: предпросмотр (первичный луч, «фонарь у камеры») / рендер
};

struct FrameInfo {
    double ms = 0;
    uint32_t spp = 0;
    bool has_den = false;
    bool preview_avail = false, preview = false;  // есть ядро предпросмотра; включён ли предпросмотр
    uint32_t rw = 0, rh = 0, ww = 0, wh = 0;      // разрешение рендера и окна (пиксели)
    int res_choice = -1;                          // текущий пресет (−1 — заданное в --res)
    bool fullscreen = false, interactive = false;
};

// Что пересобирается: граф под новое разрешение (сцена прежняя) или граф по документу редактора нод (DD-011); номера —
// как у прежних видов перегенерации (строка graph_switch журнала)
enum class RebuildKind { Resolution = 2, Graph = 3 };

// Перегенерация в процессе (DD-010, окно без Python): сцена (JSON) и аргументы как у app.py (--res, --cube, --vsync)
// → граф в каталоге out_dir (memfs: <out_dir>/app_graph.txt). false — ошибка, сообщение в err.
using RebuildFn = std::function<bool(const std::string& scene_json, const std::vector<std::string>& args,
                                     const std::string& out_dir, std::string& err)>;

// Окно сцены: панель «Сцена» (рендер), сохранение и перегенерации графа (в процессе или Python) по документу графа
// сцены. Саму сцену правит редактор нод (node_editor.hpp); объекты и многообразие прежних панелей DD-008/009 — его ноды.
class Editor {
public:
    explicit Editor(const EditorConfig& cfg);
    ~Editor();
    void set_config(const EditorConfig& cfg);     // после перезагрузки графа (каталог перегенераций — прежний)
    const EditorConfig& config() const { return cfg_; }
    void set_status(const std::string& s) { status_ = s; }
    const std::string& status() const { return status_; }
    // панель «Сцена»; cam — текущая камера (положение)
    UiActions draw(const FrameInfo& fi, const CameraState& cam, bool& use_den, float& exposure, float& speed);

    // документ графа сцены — источник всех перегенераций и сохранения
    void set_doc(const std::string& doc_text) { doc_override_ = doc_text; }
    bool has_doc() const { return !doc_override_.empty(); }
    // быстрый граф по документу; keep_camera — камера из документа (открыт другой файл), иначе — текущая
    void rebuild_doc(const std::string& doc_text, const CameraState& cam, bool wait, bool keep_camera = false);
    // та же сцена, новое разрешение рендера (аргумент --res перегенераций меняется насовсем)
    void rebuild_resolution(const CameraState& cam, uint32_t w, uint32_t h, bool wait);
    // сохранить сцену (документ с текущей камерой)
    void save_scene(const std::string& path, const CameraState& cam);
    // перегенерация: готово — путь к новому графу (иначе пусто); ошибка — сообщение в status()
    std::string poll_rebuild();
    RebuildKind last_rebuild_kind() const { return rebuild_kind_; }
    double last_rebuild_ms() const { return rebuild_ms_; }     // от запуска перегенерации до готового графа
    bool rebuilding() const { return worker_ != nullptr; }
    // перегенерация функцией в процессе вместо Python (директива rebuild не нужна); args — как у app.py
    void set_inproc_rebuild(RebuildFn fn, std::vector<std::string> args) {
        rebuild_fn_ = std::move(fn);
        inproc_args_ = std::move(args);
    }

private:
    EditorConfig cfg_;
    std::string status_;
    int rebuild_n_ = 0;
    std::unique_ptr<std::thread> worker_;
    std::atomic<bool> done_{false};
    int rc_ = 0;
    std::string log_path_, graph_out_;
    RebuildFn rebuild_fn_;
    std::vector<std::string> inproc_args_;        // --res меняется насовсем (rebuild_resolution)
    std::string rebuild_err_;
    std::string doc_override_;                    // документ графа из редактора нод (пусто — файл сцены)
    bool keep_doc_camera_ = false;                // перегенерация: камера — из документа
    std::chrono::steady_clock::time_point rebuild_t0_{};
    double rebuild_ms_ = 0;
    RebuildKind rebuild_kind_ = RebuildKind::Graph;
    void start_rebuild(const CameraState& cam, RebuildKind kind);
    std::string scene_text(const CameraState& cam) const;
};
