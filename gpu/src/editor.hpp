// Редактор сцен (DD-008): модель объектов, упаковка строк для GPU (зеркало geo4d.gpu_pt.object_rows), правило
// NEE-источников M34 (зеркало geo4d.pathtrace.nee_lights), панели ImGui, запись JSON и фоновая перегенерация графа
// в Python при смене многообразия. DD-009: чтение JSON сцены и строки многообразия (зеркала geo4d.scenefile.build и
// geo4d.gpu_pt.manifold_rows) — сцена с другим многообразием сразу открывается в режиме редактирования.
#pragma once

#include <array>
#include <atomic>
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
    int geom = 0;                                 // 0 — ball4, 1 — torus, 2 — plane (полупространство)
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
    bool objects_changed = false;                 // загрузить строки, сбросить накопление
    bool structural = false;                      // изменилась структура (число/формы/оси) — нужен режим редактирования
    bool toggle_edit = false;                     // войти в режим редактирования / выйти из него
    bool snap = false;                            // спроецировать центры on_M на M (ядро pt_snap) и прочитать обратно
    bool need_camera = false;                     // прочитать камеру (поставить объект перед ней) — следующий кадр
    bool manifold_now = false;                    // DD-009: применить вид/параметры многообразия сразу (режим редактирования)
    bool load_now = false;                        // DD-009: открыть сцену pending_scene() сразу (режим редактирования)
    int res_choice = -1;                          // выбран пресет разрешения (индекс kResPresets)
    bool toggle_fullscreen = false;
};

struct FrameInfo {
    double ms = 0;
    uint32_t spp = 0;
    bool has_den = false;
    uint32_t rw = 0, rh = 0, ww = 0, wh = 0;      // разрешение рендера и окна (пиксели)
    int res_choice = -1;                          // текущий пресет (−1 — заданное в --res)
    bool fullscreen = false, interactive = false;
    bool instant = false;                         // есть граф режима редактирования: многообразие меняется сразу
};

enum class RebuildKind { Manifold, Structure, Resolution };

class Editor {
public:
    explicit Editor(const EditorConfig& cfg);
    ~Editor();
    // после перезагрузки графа; keep_objs — оставить объекты редактора (имена, порядок), строки уже записаны хостом
    void set_config(const EditorConfig& cfg, const float* rows, bool keep_objs = false);
    const std::vector<EdObject>& objects() const { return objs_; }
    std::vector<EdObject>& objects() { return objs_; }
    const EdFlat& flat() const { return cfg_.flat; }
    const EditorConfig& config() const { return cfg_; }
    // ключ многообразия, зашитого в быстрые шейдеры последней перегенерации (или исходного графа)
    std::string current_manifold_key() const { return manifold_key(cfg_.kind, cfg_.params, cfg_.env); }
    std::string requested_manifold_key() const { return requested_mf_; }

    // DD-009: сцена для мгновенного применения — открытая из файла или текущая с новым многообразием из панели
    const SceneDesc& pending_scene() const { return pending_; }
    // текущая сцена с другим многообразием (как кнопка «Применить»; параметры поверх значений по умолчанию вида)
    SceneDesc with_manifold(const std::string& kind, const std::vector<ManifoldParam>& params, const CameraState& cam) const;
    // мгновенное применение удалось: многообразие, объекты (центры уже на M), fov, скорость — состояние редактора
    void commit_scene(const SceneDesc& sd, const ManifoldInfo& mi, const std::vector<EdObject>& objs);
    void set_status(const std::string& s) { status_ = s; }

    // панели; cam — текущая камера (для «перед камерой» и сохранения)
    UiActions draw(const FrameInfo& fi, const CameraState& cam, bool edit_mode, bool& use_den, float& exposure,
                   float& speed);

    // перегенерация: готово — путь к новому графу (иначе пусто); ошибка — сообщение в окне
    std::string poll_rebuild();
    // то же, что кнопка «Применить», из сценария (директива remanifold); wait — дождаться Python
    void rebuild(const std::string& kind, const std::vector<ManifoldParam>& params, const CameraState& cam, bool wait);
    const std::string& status() const { return status_; }
    // быстрые шейдеры под текущую структуру объектов (многообразие и камера прежние) — при выходе из режима редактирования
    void rebuild_structure(const CameraState& cam, bool wait);
    // то же многообразие и объекты, новое разрешение рендера (аргумент --res перегенераций меняется насовсем)
    void rebuild_resolution(const CameraState& cam, uint32_t w, uint32_t h, bool wait);
    RebuildKind last_rebuild_kind() const { return rebuild_kind_; }
    const std::vector<std::pair<int, int>>& requested_structure() const { return requested_; }
    bool rebuilding() const { return worker_ != nullptr; }

private:
    EditorConfig cfg_;
    std::vector<EdObject> objs_;
    int sel_ = -1;
    std::string kind_edit_;
    std::vector<ManifoldParam> params_edit_;
    char save_path_[512] = {};
    std::string status_;
    int rebuild_n_ = 0;
    std::unique_ptr<std::thread> worker_;
    std::atomic<bool> done_{false};
    int rc_ = 0;
    std::string log_path_, graph_out_;

    RebuildKind rebuild_kind_ = RebuildKind::Manifold;
    std::vector<std::pair<int, int>> requested_;
    std::string requested_mf_;
    SceneDesc pending_;
    std::vector<std::string> scene_files_;        // scenes/*.json для списка «Открыть»
    int scene_sel_ = -1;
    void start_rebuild(const CameraState& cam, RebuildKind kind);
    void save(const std::string& path, const CameraState& cam);
};
