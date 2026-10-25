// Модель сцены окна (DD-010): файл сцены (формат geo4d.scenefile, version 1), каталог многообразий (вид → scene_id,
// SCALE, выход, плоские зоны M29, s_max / ell_max — как geo4d.scenefile.build и geo4d.shapes), проекция на M (M3,
// geo4d.shapes.project_points) и касательный базис камеры (geo4d.camera.make_camera).
// Числа хранятся в double, как в Python; во float32 — только при упаковке строк для GPU (codegen.hpp).
#pragma once

#include <array>
#include <map>

#include "expr.hpp"
#include <string>
#include <utility>
#include <vector>

namespace geo4d {

using Vec4 = std::array<double, 4>;

struct Material {
    std::string kind = "diffuse";                 // diffuse | metal | emit
    std::array<double, 3> albedo{0.8, 0.8, 0.8};
    double rough = 0.3;
    std::array<double, 3> emit{0, 0, 0};
};

struct Object {
    std::string name, kind;                       // kind: ball4 | torus | plane
    Vec4 center{0, 0, 0, 0};                      // ball4, torus (после проекции, если on_M)
    double r = 0, R = 0;                          // ball4: r; torus: R, r
    int axis = 2;                                 // plane: sign·(p[axis] − offset)
    double offset = 0, sign = 1;
    bool on_M = false;
    Material mat;
    std::string node;                             // тип ноды формы в графе (DD-011): формула и строки — из библиотеки
    std::map<std::string, double> values;         // её параметры (векторы — компонентами c.x …)
    double bound() const;                         // радиус ограничивающей сферы (inf — полупространство)
};

// параметры многообразия в порядке geo4d.scenefile.MANIFOLDS (значения по умолчанию, поверх — из файла)
using Params = std::vector<std::pair<std::string, std::vector<double>>>;

struct FlatZones {                                // M29 (geo4d.pathtrace.FlatZones)
    bool has = false;
    std::vector<double> sheets;
    std::vector<std::array<double, 3>> centers;
    double rho0 = 0;
};

struct Manifold {
    std::string kind;
    Params params;
    std::string shape_name;                       // Shape.name: ключ дефайнов MF_* и SCENE_IDS
    int scene_id = 0;                             // scenes.slang
    int exit = 2;                                 // 1 — Эллис, 2 — плоский выход
    double scale = 1;                             // Shape.scale (ℓ_s)
    FlatZones flat;
    bool has_s_max = false;                       // meta["s_max"] (компактные пространства)
    double s_max = 0;
    bool has_ell_max = false;
    double ell_max = 0;
    bool compact = false;                         // COMPACT_SPACES
    // многообразие из графа без вида scenes.slang (DD-011: листы с горловинами M35, формула M37): f — формула языка
    // нод (числа подставлены), вычисляется на CPU (проекции) и выводится в mf_scene.slang
    std::string formula;
    ExprPtr expr;
    std::map<std::string, double> expr_vals;      // параметры формулы (a…d у mf.formula)
    bool legacy() const { return formula.empty(); }

    double p(const std::string& key, size_t i = 0) const;
    const std::vector<double>& pv(const std::string& key) const;
};

// вид и параметры → многообразие; недопустимые параметры (geo4d.scenefile.check_params) — исключение
Manifold make_manifold(const std::string& kind, const Params& overrides);
Params default_params(const std::string& kind);
bool is_compact_space(const std::string& kind);

struct CameraHints {
    Vec4 x{0, 0, 0, 0}, f{0, 1, 0, 0}, up{0, 0, 1, 0}, right{0, 0, 0, 0};
    bool has_right = false;
    double fov = 75;
};

struct Scene {
    std::string path;
    Manifold mf;
    std::string env_field;                        // поле "env" файла ("" — нет поля)
    std::string env;                              // procedural | none
    std::vector<Object> objs;
    CameraHints cam;                              // x — уже на M
    double speed = 1;
    std::string mf_node;                          // тип ноды многообразия в графе (DD-011)
    std::map<std::string, double> mf_values;      // её параметры (векторы — компонентами c1.x …)
};

// файл сцены (`.g4d` version 2 или scenes/*.json version 1 — переводится в граф, DD-011) → модель окна;
// project = false — центры on_M и точка камеры как в файле (редактор проецирует их на GPU)
Scene load_scene(const std::string& path, bool project = true);

// f многообразия и ∇f в точке (double; формулы geo4d.shapes)
double sdf(const Manifold& m, const Vec4& x, Vec4* grad = nullptr);
// M3: Ньютон вдоль ∇f до f = 0, как geo4d.shapes.project_points (30 итераций)
Vec4 project(const Manifold& m, Vec4 x, int iters = 30);

struct Camera {                                   // geo4d.camera.Camera
    Vec4 x, f, r, u;
    double fov = 90;
    int width = 64, height = 64;
    double focus = 5, aperture = 0;
    double tf() const;
    double theta_pix() const;
};
// geo4d.camera.make_camera: касательный базис в x по подсказкам (двойной Грам–Шмидт)
Camera make_camera(const Manifold& m, const CameraHints& h, int width, int height, double fov);

// M34 (geo4d.pathtrace.nee_lights): индексы излучающих 4D-шаров, допустимых для NEE
std::vector<int> nee_lights(const std::vector<Object>& objs, const FlatZones& fz);

}  // namespace geo4d
