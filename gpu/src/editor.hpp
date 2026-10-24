// Редактор сцен (DD-008): модель объектов, упаковка строк для GPU (зеркало geo4d.gpu_pt.object_rows), правило
// NEE-источников M34 (зеркало geo4d.pathtrace.nee_lights), панели ImGui, запись JSON и фоновая перегенерация графа
// в Python при смене многообразия.
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

struct ManifoldParam {
    std::string key;
    std::vector<float> v;                         // 1 или 3 числа (центры горловин)
};

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

// Что окно должно сделать после панелей
struct UiActions {
    bool objects_changed = false;                 // загрузить строки, сбросить накопление
    bool snap = false;                            // спроецировать центры on_M на M (ядро pt_snap) и прочитать обратно
    bool need_camera = false;                     // прочитать камеру (поставить объект перед ней) — следующий кадр
};

struct FrameInfo {
    double ms = 0;
    uint32_t spp = 0;
    bool has_den = false;
};

class Editor {
public:
    explicit Editor(const EditorConfig& cfg);
    ~Editor();
    void set_config(const EditorConfig& cfg, const float* rows);   // после перезагрузки графа
    const std::vector<EdObject>& objects() const { return objs_; }
    std::vector<EdObject>& objects() { return objs_; }
    const EdFlat& flat() const { return cfg_.flat; }

    // панели; cam — текущая камера (для «перед камерой» и сохранения)
    UiActions draw(const FrameInfo& fi, const CameraState& cam, bool& use_den, float& exposure, float& speed);

    // перегенерация: готово — путь к новому графу (иначе пусто); ошибка — сообщение в окне
    std::string poll_rebuild();
    // то же, что кнопка «Применить», из сценария (директива remanifold); wait — дождаться Python
    void rebuild(const std::string& kind, const std::vector<ManifoldParam>& params, const CameraState& cam, bool wait);
    const std::string& status() const { return status_; }
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

    void start_rebuild(const CameraState& cam);
    void save(const std::string& path, const CameraState& cam);
};
