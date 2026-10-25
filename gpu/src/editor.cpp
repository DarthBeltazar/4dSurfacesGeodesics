// Редактор сцен (DD-008) — см. editor.hpp. DD-010: строки объектов и многообразия, правило M34 и чтение файла сцены —
// через модель окна (scene.hpp, codegen.hpp): одна упаковка для холодного старта и правок в окне.
#include "editor.hpp"

#include "codegen.hpp"

#include <nlohmann/json.hpp>
#include "scene.hpp"

#include <imgui.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {

const char* kKinds[] = {"hyperplane", "S3", "spheritorus", "torisphere", "tiger", "ellis", "compact_wh", "two_compact_wh"};
const char* kKindNames[] = {"гиперплоскость R³", "S³", "сферитор", "ториосфера", "T³ («тигр»)", "червоточина Эллиса",
                            "компактная червоточина", "две компактные червоточины"};

const char* kCompactSpaces[] = {"S3", "spheritorus", "torisphere", "tiger"};       // geo4d.scenefile.COMPACT_SPACES

bool is_compact_space(const std::string& kind) {
    for (const char* k : kCompactSpaces) if (kind == k) return true;
    return false;
}

// ---- преобразования модели редактора (float) ↔ модель окна (double, scene.hpp) -----------------------------

geo4d::Params to_params(const std::vector<ManifoldParam>& ps) {
    geo4d::Params p;
    for (auto& x : ps) p.emplace_back(x.key, std::vector<double>(x.v.begin(), x.v.end()));
    return p;
}

geo4d::FlatZones to_flat(const EdFlat& f) {
    geo4d::FlatZones z;
    z.has = f.has;
    z.sheets.assign(f.sheets.begin(), f.sheets.end());
    for (auto& b : f.balls) z.centers.push_back({b[0], b[1], b[2]});
    z.rho0 = f.rho0;
    return z;
}

EdFlat from_flat(const geo4d::FlatZones& z) {
    EdFlat f;
    f.has = z.has;
    for (double s : z.sheets) f.sheets.push_back(float(s));
    for (auto& c : z.centers) f.balls.push_back({float(c[0]), float(c[1]), float(c[2])});
    f.rho0 = float(z.rho0);
    return f;
}

std::vector<float> to_f(const std::vector<double>& v) {
    std::vector<float> r;
    for (double x : v) r.push_back(float(x));
    return r;
}

const char* kGeoms[] = {"ball4", "torus", "plane", "graph"};
const char* kMats[] = {"diffuse", "metal", "emit"};

geo4d::Object to_object(const EdObject& e) {
    geo4d::Object o;
    o.name = e.name;
    o.kind = kGeoms[e.geom];
    for (int i = 0; i < 4; ++i) o.center[i] = e.c[i];
    if (e.geom == 0 || e.geom == 3) o.r = e.a;   // составная форма: a — радиус ограничивающей сферы
    else if (e.geom == 1) { o.R = e.a; o.r = e.b; }
    o.axis = e.axis;
    o.offset = e.offset;
    o.sign = e.sign;
    o.on_M = e.on_M;
    o.mat.kind = kMats[e.mat];
    for (int i = 0; i < 3; ++i) { o.mat.albedo[i] = e.albedo[i]; o.mat.emit[i] = e.emit[i]; }
    o.mat.rough = e.rough;
    return o;
}

EdObject from_object(const geo4d::Object& o) {
    EdObject e;
    e.name = o.name;
    std::replace(e.name.begin(), e.name.end(), ' ', '_');
    e.geom = o.kind == "ball4" ? 0 : o.kind == "torus" ? 1 : o.kind == "graph" ? 3 : 2;
    for (int i = 0; i < 4; ++i) e.c[i] = float(o.center[i]);
    if (e.geom == 0 || e.geom == 3) e.a = float(o.r);
    else if (e.geom == 1) { e.a = float(o.R); e.b = float(o.r); }
    e.axis = o.axis;
    e.offset = float(o.offset);
    e.sign = float(o.sign);
    e.on_M = o.on_M;
    e.mat = o.mat.kind == "diffuse" ? 0 : o.mat.kind == "metal" ? 1 : 2;
    for (int i = 0; i < 3; ++i) { e.albedo[i] = float(o.mat.albedo[i]); e.emit[i] = float(o.mat.emit[i]); }
    e.rough = float(o.mat.rough);
    return e;
}

}  // namespace

// параметры по умолчанию — как geo4d.scenefile.MANIFOLDS (scene.cpp); неизвестный вид — пусто
std::vector<ManifoldParam> default_params(const std::string& kind) {
    std::vector<ManifoldParam> out;
    try {
        for (auto& [k, v] : geo4d::default_params(kind)) out.push_back({k, to_f(v)});
    } catch (const std::exception&) {
    }
    return out;
}

namespace {

std::string num(float v) {
    char b[32];
    std::snprintf(b, sizeof(b), "%.9g", double(v));
    std::string s = b;
    if (s.find_first_of(".eEn") == std::string::npos) s += ".0";      // как в Python: 1.0, не 1
    return s;
}

std::string vec(const float* v, int n) {
    std::string s = "[";
    for (int i = 0; i < n; ++i) s += (i ? ", " : "") + num(v[i]);
    return s + "]";
}

std::string json_str(const std::string& s) {
    std::string o = "\"";
    for (char ch : s) {
        if (ch == '"' || ch == '\\') o += '\\';
        o += ch;
    }
    return o + "\"";
}

float max3(const float* v) { return std::max(v[0], std::max(v[1], v[2])); }

}  // namespace

// ---- многообразие: строки для динамических шейдеров (зеркало geo4d.gpu_pt.manifold_rows) ---------------------

std::string manifold_key(const std::string& kind, const std::vector<ManifoldParam>& params, const std::string& env) {
    std::string k = kind + "|" + (env.empty() ? (is_compact_space(kind) ? "none" : "procedural") : env);
    for (auto& p : default_params(kind)) {               // параметры по порядку вида, недостающие — по умолчанию
        const ManifoldParam* q = &p;
        for (auto& x : params) if (x.key == p.key) q = &x;
        k += "|" + p.key;
        for (float v : q->v) k += " " + num(v);
    }
    return k;
}

ManifoldInfo manifold_info(const std::string& kind, const std::vector<ManifoldParam>& params, const std::string& env) {
    ManifoldInfo mi;
    try {
        geo4d::Scene sc;
        sc.mf = geo4d::make_manifold(kind, to_params(params));       // check_params — исключение с сообщением
        sc.env = env.empty() ? (sc.mf.compact ? "none" : "procedural") : env;
        mi.rows = geo4d::manifold_rows(sc);
        mi.flat = from_flat(sc.mf.flat);
        mi.background = sc.mf.flat.has || sc.mf.exit == 1;
        mi.scale = float(sc.mf.scale);
        mi.ok = true;
    } catch (const std::exception& e) {
        mi.err = e.what();
    }
    return mi;
}

// ---- сцена из JSON (geo4d.scenefile.build без проекций) ----------------------------------------------

SceneDesc read_scene_json(const std::string& path) {
    geo4d::Scene s = geo4d::load_scene(path, false);              // проекции — на GPU (pt_snap, apply_scene)
    if (s.objs.size() > size_t(kMaxObj)) throw std::runtime_error("объектов больше 32");
    SceneDesc sd;
    sd.path = path;
    sd.kind = s.mf.kind;
    for (auto& [k, v] : s.mf.params) sd.params.push_back({k, to_f(v)});
    sd.env = s.env_field;
    for (auto& o : s.objs) sd.objs.push_back(from_object(o));
    for (int i = 0; i < 4; ++i) {
        sd.cam.x[i] = float(s.cam.x[i]);
        sd.cam.f[i] = float(s.cam.f[i]);
        sd.cam.u[i] = float(s.cam.up[i]);
        sd.cam.r[i] = float(s.cam.right[i]);
    }
    sd.has_right = s.cam.has_right;
    sd.fov = float(s.cam.fov);
    sd.speed = float(s.speed);
    return sd;
}

// ---- касательный базис камеры (geo4d.camera.make_camera) ---------------------------------------------

void tangent_basis(const float nf[4], const float fh[4], const float uh[4], const float rh[4], bool has_r, float f[4],
                   float u[4], float r[4]) {
    using V = std::array<double, 4>;
    auto dot = [](const V& a, const V& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]; };
    auto from = [](const float* a) { return V{a[0], a[1], a[2], a[3]}; };
    V n = from(nf);
    double nn = std::sqrt(dot(n, n));
    for (auto& a : n) a /= nn;
    // двойной Грам–Шмидт; при вырожденной подсказке — первая координатная ось с устойчивым остатком
    auto orth = [&](const V& hint, const std::vector<V>& basis) {
        std::vector<V> cands{hint, {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
        for (auto& cand : cands) {
            V c = cand;
            for (int it = 0; it < 2; ++it)
                for (auto& b : basis) {
                    double k = dot(c, b);
                    for (int i = 0; i < 4; ++i) c[i] -= k * b[i];
                }
            double l = std::sqrt(dot(c, c)), l0 = std::max(std::sqrt(dot(cand, cand)), 1e-300);
            if (l > 1e-6 * l0) {
                for (auto& a : c) a /= l;
                return c;
            }
        }
        throw std::runtime_error("не удалось построить касательный базис");
    };
    V F = orth(from(fh), {n});
    V U = orth(from(uh), {n, F});
    V R;
    if (has_r) R = orth(from(rh), {n, F, U});
    else {
        R = orth(V{0, 0, 0, 0}, {n, F, U});          // нулевая подсказка — первая ось с остатком: ±(дополнение)
        auto det3 = [](double a, double b, double c, double d, double e, double f2, double g, double h, double i) {
            return a * (e * i - f2 * h) - b * (d * i - f2 * g) + c * (d * h - e * g);
        };
        const V* M[4] = {&n, &F, &R, &U};             // det(n, f, r, u) > 0
        double det = 0;
        for (int j = 0; j < 4; ++j) {
            int c[3], k = 0;
            for (int q = 0; q < 4; ++q) if (q != j) c[k++] = q;
            double minor = det3((*M[1])[c[0]], (*M[1])[c[1]], (*M[1])[c[2]], (*M[2])[c[0]], (*M[2])[c[1]], (*M[2])[c[2]],
                                (*M[3])[c[0]], (*M[3])[c[1]], (*M[3])[c[2]]);
            det += ((j % 2) ? -1.0 : 1.0) * (*M[0])[j] * minor;
        }
        if (det < 0) for (auto& a : R) a = -a;
    }
    for (int i = 0; i < 4; ++i) { f[i] = float(F[i]); u[i] = float(U[i]); r[i] = float(R[i]); }
}

const ResPreset kResPresets[] = {{"как окно", 1.0f, 0, 0},          {"3/4 окна", 0.75f, 0, 0},
                                 {"1/2 окна", 0.5f, 0, 0},          {"3840×2160", 0, 3840, 2160},
                                 {"2560×1440", 0, 2560, 1440},      {"1920×1080", 0, 1920, 1080},
                                 {"1600×900", 0, 1600, 900},        {"1280×720", 0, 1280, 720},
                                 {"960×540", 0, 960, 540},          {"640×360", 0, 640, 360}};
const int kNumResPresets = int(sizeof(kResPresets) / sizeof(kResPresets[0]));

std::pair<uint32_t, uint32_t> preset_size(int i, uint32_t ww, uint32_t wh) {
    const ResPreset& p = kResPresets[i];
    if (p.frac <= 0) return {p.w, p.h};
    return {std::max(64u, uint32_t(std::lround(p.frac * ww))), std::max(64u, uint32_t(std::lround(p.frac * wh)))};
}

// ---- модель: M34, упаковка строк -------------------------------------------------------------------

std::vector<int> nee_lights(const std::vector<EdObject>& objs, const EdFlat& flat) {
    std::vector<geo4d::Object> o;
    for (auto& e : objs) o.push_back(to_object(e));
    return geo4d::nee_lights(o, to_flat(flat));
}

std::vector<float> pack_rows(const std::vector<EdObject>& objs, const EdFlat& flat) {
    geo4d::Scene sc;
    for (auto& e : objs) sc.objs.push_back(to_object(e));
    sc.mf.flat = to_flat(flat);
    return geo4d::object_rows(sc);
}

std::vector<EdObject> unpack_rows(const float* R, const std::vector<std::string>& names) {
    int n = int(R[0]);
    std::vector<EdObject> objs(n);
    for (int k = 0; k < n; ++k) {
        const float* r = R + size_t(9 + 6 * k) * 4;
        EdObject& o = objs[k];
        o.name = k < int(names.size()) ? names[k] : "obj" + std::to_string(k);
        o.geom = int(r[4]);
        std::copy(r, r + 4, o.c);
        if (o.geom == 2) { o.axis = int(r[5]); o.offset = r[6]; o.sign = r[7]; }
        else { o.a = r[5]; o.b = r[6]; }
        std::copy(r + 12, r + 15, o.albedo);
        o.mat = int(r[15]);
        std::copy(r + 16, r + 19, o.emit);
        o.rough = r[19];
        o.on_M = r[21] != 0;
    }
    return objs;
}

// ---- конфигурация из директив графа ------------------------------------------------------------------

EditorConfig parse_editor_config(const std::vector<std::vector<std::string>>& extra) {
    EditorConfig c;
    for (auto& w : extra) {
        if (w[0] == "editor") {               // editor <сцена.json> <рабочий каталог> <fov>
            c.enabled = true;
            c.scene_path = w.at(1);
            c.work_dir = w.at(2);
            c.fov = std::stof(w.at(3));
        } else if (w[0] == "manifold") {      // manifold <вид> <env> k=v k=v1,v2,v3 …
            c.kind = w.at(1);
            c.env = w.at(2) == "-" ? "" : w.at(2);
            for (size_t i = 3; i < w.size(); ++i) {
                auto eq = w[i].find('=');
                ManifoldParam p{w[i].substr(0, eq), {}};
                std::stringstream ss(w[i].substr(eq + 1));
                for (std::string t; std::getline(ss, t, ',');) p.v.push_back(std::stof(t));
                c.params.push_back(p);
            }
        } else if (w[0] == "flat") {          // flat <n листов> w… <n шаров> ρ₀ cx cy cz …
            size_t i = 1;
            c.flat.has = true;
            int ns = std::stoi(w.at(i++));
            for (int k = 0; k < ns; ++k) c.flat.sheets.push_back(std::stof(w.at(i++)));
            int nb = std::stoi(w.at(i++));
            c.flat.rho0 = std::stof(w.at(i++));
            for (int k = 0; k < nb; ++k) {
                std::array<float, 3> b{};
                for (auto& v : b) v = std::stof(w.at(i++));
                c.flat.balls.push_back(b);
            }
        } else if (w[0] == "objname") {       // objname <k> <имя>
            size_t k = std::stoul(w.at(1));
            if (c.names.size() <= k) c.names.resize(k + 1);
            c.names[k] = w.size() > 2 ? w[2] : "";
        } else if (w[0] == "rebuild") {       // rebuild <python> <скрипт> <аргументы…>
            c.python = w.at(1);
            c.script = w.at(2);
            c.rebuild_args.assign(w.begin() + 3, w.end());
        } else if (w[0] == "graphdoc") {      // DD-011: сцена — граф, не выразимый в version 1
            c.graph_doc = true;
        } else if (w[0] == "speed") {
            c.speed = std::stof(w.at(1));
        }
    }
    return c;
}

// ---- JSON (geo4d.scenefile) ---------------------------------------------------------------------------

std::string scene_json(const std::string& kind, const std::vector<ManifoldParam>& params, const std::string& env,
                       const std::vector<EdObject>& objs, const CameraState& cam, float fov, float speed) {
    static const char* geom[] = {"ball4", "torus", "plane"};
    static const char* mats[] = {"diffuse", "metal", "emit"};
    std::ostringstream o;
    o << "{\n \"manifold\": {\n  \"kind\": " << json_str(kind) << ",\n  \"params\": {";
    for (size_t i = 0; i < params.size(); ++i)
        o << (i ? "," : "") << "\n   " << json_str(params[i].key) << ": "
          << (params[i].v.size() == 1 ? num(params[i].v[0]) : vec(params[i].v.data(), int(params[i].v.size())));
    o << (params.empty() ? "}" : "\n  }") << "\n },\n";
    if (!env.empty()) o << " \"env\": " << json_str(env) << ",\n";
    o << " \"objects\": [";
    for (size_t k = 0; k < objs.size(); ++k) {
        const EdObject& b = objs[k];
        o << (k ? "," : "") << "\n  {\n   \"name\": " << json_str(b.name) << ",\n   \"kind\": " << json_str(geom[b.geom]) << ",\n";
        if (b.geom == 2)
            o << "   \"axis\": " << b.axis << ",\n   \"offset\": " << num(b.offset) << ",\n   \"sign\": " << num(b.sign) << ",\n";
        else {
            o << "   \"center\": " << vec(b.c, 4) << ",\n";
            if (b.geom == 0) o << "   \"r\": " << num(b.a) << ",\n";
            else o << "   \"R\": " << num(b.a) << ",\n   \"r\": " << num(b.b) << ",\n";
        }
        o << "   \"on_M\": " << (b.on_M ? "true" : "false") << ",\n   \"material\": {\n    \"kind\": " << json_str(mats[b.mat])
          << ",\n    \"albedo\": " << vec(b.albedo, 3) << ",\n    \"rough\": " << num(b.rough) << ",\n    \"emit\": "
          << vec(b.emit, 3) << "\n   }\n  }";
    }
    o << (objs.empty() ? "]" : "\n ]") << ",\n \"camera\": {\n  \"x\": " << vec(cam.x, 4) << ",\n  \"f\": " << vec(cam.f, 4)
      << ",\n  \"up\": " << vec(cam.u, 4) << ",\n  \"right\": " << vec(cam.r, 4) << ",\n  \"fov\": " << num(fov)
      << "\n },\n \"speed\": " << num(speed) << "\n}\n";
    return o.str();
}

// ---- редактор -------------------------------------------------------------------------------------------

Editor::Editor(const EditorConfig& cfg) : cfg_(cfg) {
    std::snprintf(save_path_, sizeof(save_path_), "%s", cfg.scene_path.c_str());
    kind_edit_ = cfg.kind;
    params_edit_ = cfg.params;
}

Editor::~Editor() {
    if (worker_ && worker_->joinable()) worker_->join();
}

std::vector<std::pair<int, int>> structure(const std::vector<EdObject>& objs) {
    std::vector<std::pair<int, int>> s;
    for (auto& o : objs) s.emplace_back(o.geom, o.geom == 2 ? o.axis : -1);
    return s;
}

void Editor::set_config(const EditorConfig& cfg, const float* rows, bool keep_objs) {
    // keep_objs: сцена — состояние редактора (объекты, многообразие после мгновенной смены, fov, разрешение), граф —
    // только исполнитель; иначе — сцена нового графа (перегенерация с другим многообразием)
    if (!keep_objs) {
        std::string wd = cfg_.work_dir;                // перегенерации — в каталоге первого графа (без вложенности)
        cfg_ = cfg;
        if (!wd.empty()) cfg_.work_dir = wd;
        kind_edit_ = cfg.kind;
        params_edit_ = cfg.params;
        objs_ = unpack_rows(rows, cfg.names);
    }
    if (sel_ >= int(objs_.size()) || sel_ < 0) sel_ = objs_.empty() ? -1 : 0;
}

SceneDesc Editor::with_manifold(const std::string& kind, const std::vector<ManifoldParam>& params,
                                const CameraState& cam) const {
    SceneDesc sd;
    sd.path = cfg_.scene_path;
    sd.kind = kind;
    sd.params = default_params(kind);
    for (auto& p : params)
        for (auto& q : sd.params)
            if (q.key == p.key && q.v.size() == p.v.size()) q.v = p.v;
    sd.env = kind == cfg_.kind ? cfg_.env : "";          // другой вид — фон по умолчанию для вида
    sd.objs = objs_;
    sd.cam = cam;
    sd.has_right = true;
    sd.fov = cfg_.fov;
    sd.speed = cfg_.speed;
    return sd;
}

void Editor::commit_scene(const SceneDesc& sd, const ManifoldInfo& mi, const std::vector<EdObject>& objs) {
    cfg_.kind = sd.kind;
    cfg_.params = sd.params;
    cfg_.env = sd.env;
    cfg_.flat = mi.flat;
    cfg_.fov = sd.fov;
    cfg_.speed = sd.speed;
    cfg_.scene_path = sd.path;
    kind_edit_ = sd.kind;
    params_edit_ = sd.params;
    objs_ = objs;
    sel_ = objs_.empty() ? -1 : 0;
}

// Текст сцены для сохранения и перегенерации: version 1 — из состояния редактора; документ-граф (DD-011) — исходный
// .g4d с текущей камерой (правки материалов ползунками до шага 6 в него не попадают)
std::string Editor::scene_text(const std::string& kind, const std::vector<ManifoldParam>& params, const std::string& env,
                               const CameraState& cam) const {
    if (!cfg_.graph_doc) return scene_json(kind, params, env, objs_, cam, cfg_.fov, cfg_.speed);
    std::ifstream in(cfg_.scene_path, std::ios::binary);
    if (!in) throw std::runtime_error("не открыть " + cfg_.scene_path);
    nlohmann::json doc = nlohmann::json::parse(in);
    auto v4 = [](const float* a) { return nlohmann::json::array({a[0], a[1], a[2], a[3]}); };
    doc["camera"] = {{"x", v4(cam.x)}, {"f", v4(cam.f)}, {"up", v4(cam.u)}, {"right", v4(cam.r)}, {"fov", cfg_.fov}};
    return doc.dump(1) + "\n";
}

void Editor::save(const std::string& path, const CameraState& cam) {
    std::ofstream f(path, std::ios::binary);
    if (!f) { status_ = "не удалось записать " + path; return; }
    try {
        f << scene_text(cfg_.kind, cfg_.params, cfg_.env, cam);
    } catch (const std::exception& e) {
        status_ = e.what();
        return;
    }
    status_ = "сохранено: " + path;
}

void Editor::start_rebuild(const CameraState& cam, RebuildKind rk) {
    const bool same_manifold = rk != RebuildKind::Manifold;
    if (cfg_.python.empty() && !rebuild_fn_) { status_ = "перегенерация недоступна (нет директивы rebuild)"; return; }
    ++rebuild_n_;
    rebuild_t0_ = std::chrono::steady_clock::now();
    std::string base = cfg_.work_dir + "/edit_" + std::to_string(rebuild_n_);
    std::string json = base + ".json";
    graph_out_ = base;
    log_path_ = base + ".log";
    rebuild_kind_ = rk;                         // структура объектов или разрешение: многообразие прежнее, камера — та же
    requested_ = structure(objs_);
    requested_mf_ = same_manifold ? current_manifold_key() : manifold_key(kind_edit_, params_edit_, "");
    const std::string& kind = same_manifold ? cfg_.kind : kind_edit_;
    const auto& params = same_manifold ? cfg_.params : params_edit_;
    std::string env = kind == cfg_.kind ? cfg_.env : "";              // другой вид — фон по умолчанию для вида
    {
        std::ofstream f(json, std::ios::binary);
        f << scene_text(kind, params, env, cam);
    }
    if (rebuild_fn_) {                              // в процессе (DD-010): кодген и компиляция в потоке
        done_ = false;
        status_ = rk == RebuildKind::Structure    ? "сборка быстрых шейдеров (новая структура объектов)…"
                  : rk == RebuildKind::Resolution ? "граф под новое разрешение…"
                                                  : "сборка шейдеров и графа…";
        worker_ = std::make_unique<std::thread>([this, fn = rebuild_fn_, json, base, args = inproc_args_]() {
            std::string err;
            rc_ = fn(json, args, base, err) ? 0 : 1;
            rebuild_err_ = err;
            done_ = true;
        });
        return;
    }
    auto q = [](const std::string& s) { return "\"" + s + "\""; };
    // -X utf8: лог Python (сообщение об ошибке в окне) — в UTF-8 и при перенаправлении в файл на Windows
    std::string cmd = q(cfg_.python) + " -X utf8 " + q(cfg_.script) + " --scene " + q(json) + " --graph-only " + q(base);
    for (auto& a : cfg_.rebuild_args) cmd += " " + a;
    cmd += " > " + q(log_path_) + " 2>&1";
#ifdef _WIN32
    cmd = "\"" + cmd + "\"";                       // cmd /c снимает внешние кавычки
#endif
    done_ = false;
    status_ = rk == RebuildKind::Structure    ? "перегенерация быстрых шейдеров (новая структура объектов)…"
              : rk == RebuildKind::Resolution ? "перегенерация графа под новое разрешение…"
                                              : "перегенерация: шейдеры и граф (Python)…";
    worker_ = std::make_unique<std::thread>([this, cmd]() {
        rc_ = std::system(cmd.c_str());
        done_ = true;
    });
}

void Editor::rebuild(const std::string& kind, const std::vector<ManifoldParam>& params, const CameraState& cam, bool wait) {
    kind_edit_ = kind;
    params_edit_ = default_params(kind);
    for (auto& p : params)
        for (auto& q : params_edit_)
            if (q.key == p.key) q.v = p.v;
    start_rebuild(cam, RebuildKind::Manifold);
    while (wait && worker_ && !done_) std::this_thread::sleep_for(std::chrono::milliseconds(20));
}

void Editor::rebuild_structure(const CameraState& cam, bool wait) {
    start_rebuild(cam, RebuildKind::Structure);
    while (wait && worker_ && !done_) std::this_thread::sleep_for(std::chrono::milliseconds(20));
}

void Editor::rebuild_resolution(const CameraState& cam, uint32_t w, uint32_t h, bool wait) {
    const std::string res = std::to_string(w) + "x" + std::to_string(h);
    auto& a = rebuild_fn_ ? inproc_args_ : cfg_.rebuild_args;
    auto it = std::find(a.begin(), a.end(), "--res");
    if (it != a.end() && it + 1 != a.end()) *(it + 1) = res;
    else { a.push_back("--res"); a.push_back(res); }
    start_rebuild(cam, RebuildKind::Resolution);
    while (wait && worker_ && !done_) std::this_thread::sleep_for(std::chrono::milliseconds(20));
}

std::string Editor::poll_rebuild() {
    if (!worker_ || !done_) return "";
    worker_->join();
    worker_.reset();
    rebuild_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - rebuild_t0_).count();
    if (rc_ == 0) {
        status_ = rebuild_kind_ == RebuildKind::Structure    ? "быстрый режим: шейдеры с новой структурой объектов"
                  : rebuild_kind_ == RebuildKind::Resolution ? "разрешение рендера обновлено"
                                                             : "многообразие обновлено";
        return graph_out_ + "/app_graph.txt";
    }
    if (rebuild_fn_) {
        status_ = "ошибка перегенерации (сцена не изменена):\n" + rebuild_err_.substr(0, 600);
        return "";
    }
    std::ifstream f(log_path_);
    std::string all((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto p = all.rfind("Error");
    if (p == std::string::npos) p = all.size() > 600 ? all.size() - 600 : 0;
    status_ = "ошибка перегенерации (сцена не изменена):\n" + all.substr(p, 600);
    return "";
}

UiActions Editor::draw(const FrameInfo& fi, const CameraState& cam, bool edit_mode, bool& use_den, float& exposure,
                       float& speed) {
    UiActions act;
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(360, 0), ImGuiCond_FirstUseEver);
    ImGui::Begin("Сцена (F1 — скрыть панели)");
    ImGui::Text("%.1f мс (%.0f к/с), накоплено %u spp", fi.ms, fi.ms > 0 ? 1000.0 / fi.ms : 0.0, fi.spp);
    ImGui::TextDisabled("x = (%.2f, %.2f, %.2f, %.2f)", cam.x[0], cam.x[1], cam.x[2], cam.x[3]);

    if (ImGui::CollapsingHeader("Рендер", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (fi.has_den) ImGui::Checkbox("денойзер (R)", &use_den);
        else ImGui::TextDisabled("денойзер недоступен");
        ImGui::SliderFloat("экспозиция", &exposure, 0.05f, 20.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("скорость", &speed, 0.05f, 20.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
        if (fi.interactive) {
            // разрешение рендера: новое — перегенерация графа (шейдеры из кэша); пока она идёт, кадр масштабируется
            char cur[64];
            std::snprintf(cur, sizeof(cur), "%s%s%u×%u", fi.res_choice >= 0 ? kResPresets[fi.res_choice].label : "",
                          fi.res_choice >= 0 ? " — " : "", fi.rw, fi.rh);
            ImGui::BeginDisabled(rebuilding());
            if (ImGui::BeginCombo("разрешение", cur)) {
                for (int i = 0; i < kNumResPresets; ++i) {
                    auto [w, h] = preset_size(i, fi.ww, fi.wh);
                    char lab[64];
                    if (kResPresets[i].frac > 0) std::snprintf(lab, sizeof(lab), "%s (%u×%u)", kResPresets[i].label, w, h);
                    else std::snprintf(lab, sizeof(lab), "%s", kResPresets[i].label);
                    if (ImGui::Selectable(lab, i == fi.res_choice)) act.res_choice = i;
                }
                ImGui::EndCombo();
            }
            ImGui::EndDisabled();
            bool fs = fi.fullscreen;
            if (ImGui::Checkbox("полноэкранный режим (F11)", &fs)) act.toggle_fullscreen = true;
            ImGui::TextDisabled("окно %u×%u", fi.ww, fi.wh);
        }
        ImGui::TextDisabled("WASD, Q/E — полёт; ЛКМ — взгляд мышью; C/Z — крен");
    }

    const bool busy = rebuilding();
    if (ImGui::CollapsingHeader("Многообразие", ImGuiTreeNodeFlags_DefaultOpen)) {
        int cur = 0;
        for (int i = 0; i < IM_ARRAYSIZE(kKinds); ++i) if (kind_edit_ == kKinds[i]) cur = i;
        ImGui::BeginDisabled(busy);
        if (ImGui::Combo("вид", &cur, kKindNames, IM_ARRAYSIZE(kKindNames)) && kind_edit_ != kKinds[cur]) {
            kind_edit_ = kKinds[cur];
            params_edit_ = kind_edit_ == cfg_.kind ? cfg_.params : default_params(kind_edit_);
        }
        for (auto& p : params_edit_) {
            if (p.v.size() == 3) ImGui::DragFloat3(p.key.c_str(), p.v.data(), 0.02f);
            else ImGui::DragFloat(p.key.c_str(), p.v.data(), 0.01f, 0.0f, 0.0f, "%.3f");
        }
        if (fi.instant) {          // DD-009: сразу, в режиме редактирования; быстрые шейдеры — при выходе из него
            if (ImGui::Button("Применить")) {
                pending_ = with_manifold(kind_edit_, params_edit_, cam);
                act.manifold_now = true;
            }
        } else if (ImGui::Button("Применить (перекомпиляция)")) start_rebuild(cam, RebuildKind::Manifold);
        ImGui::SameLine();
        if (ImGui::Button("Сброс")) { kind_edit_ = cfg_.kind; params_edit_ = cfg_.params; }
        ImGui::EndDisabled();
        if (busy) ImGui::TextColored(ImVec4(1, 0.8f, 0.3f, 1), "идёт перегенерация…");
    }

    if (ImGui::CollapsingHeader("Файл", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (fi.instant) {           // DD-009: сцены из каталога файла сцены (scenes/) — открываются сразу
            if (scene_files_.empty()) {
                std::error_code ec;
                auto dir = std::filesystem::path(cfg_.scene_path).parent_path();
                for (auto& de : std::filesystem::directory_iterator(dir, ec))
                    if (de.path().extension() == ".json") scene_files_.push_back(de.path().generic_string());
                std::sort(scene_files_.begin(), scene_files_.end());
            }
            auto stem = [](const std::string& p) { return std::filesystem::path(p).stem().string(); };
            std::string cur = scene_sel_ >= 0 && scene_sel_ < int(scene_files_.size()) ? stem(scene_files_[scene_sel_]) : "сцена…";
            ImGui::BeginDisabled(busy);
            if (ImGui::BeginCombo("##scenes", cur.c_str())) {
                for (int i = 0; i < int(scene_files_.size()); ++i)
                    if (ImGui::Selectable(stem(scene_files_[i]).c_str(), i == scene_sel_)) {
                        scene_sel_ = i;
                        std::snprintf(save_path_, sizeof(save_path_), "%s", scene_files_[i].c_str());
                    }
                ImGui::EndCombo();
            }
            ImGui::EndDisabled();
        }
        ImGui::InputText("##path", save_path_, sizeof(save_path_));
        ImGui::BeginDisabled(busy);
        if (ImGui::Button("Сохранить сцену")) save(save_path_, cam);
        if (fi.instant) {
            ImGui::SameLine();
            if (ImGui::Button("Открыть")) {
                try {
                    pending_ = read_scene_json(save_path_);
                    act.load_now = true;
                } catch (const std::exception& e) {
                    status_ = std::string("не удалось открыть сцену: ") + e.what();
                }
            }
        }
        ImGui::EndDisabled();
    }
    if (!status_.empty()) ImGui::TextWrapped("%s", status_.c_str());
    ImGui::End();

    const ImVec2 disp = ImGui::GetIO().DisplaySize;
    ImGui::SetNextWindowPos(ImVec2(disp.x - 370, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(360, 0), ImGuiCond_FirstUseEver);
    ImGui::Begin("Объекты");
    {
        // режим редактирования: шейдеры без зашитой структуры — добавление, удаление и смена формы сразу (кадр медленнее);
        // выход — быстрые шейдеры (при новой структуре перегенерируются в фоне, окно пока в режиме редактирования)
        bool em = edit_mode;
        ImGui::BeginDisabled(busy && edit_mode);
        if (ImGui::Checkbox("режим редактирования (Tab)", &em)) act.toggle_edit = true;
        ImGui::EndDisabled();
        if (!edit_mode) ImGui::TextDisabled("быстрый режим: форма и число объектов — в режиме редактирования");
        ImGui::BeginDisabled(busy && !edit_mode);
        auto lights = nee_lights(objs_, cfg_.flat);
        auto is_light = [&](int k) { return std::find(lights.begin(), lights.end(), k) != lights.end(); };
        if (ImGui::BeginListBox("##objs", ImVec2(-FLT_MIN, 6.5f * ImGui::GetTextLineHeightWithSpacing()))) {
            for (int k = 0; k < int(objs_.size()); ++k) {
                std::string label = objs_[k].name + (is_light(k) ? "  [NEE]" : "") + "##" + std::to_string(k);
                if (ImGui::Selectable(label.c_str(), sel_ == k)) sel_ = k;
            }
            ImGui::EndListBox();
        }
        auto add = [&](int geom) {
            if (objs_.size() >= size_t(kMaxObj)) { status_ = "не больше 32 объектов"; return; }
            EdObject o;
            o.geom = geom;
            o.name = geom == 0 ? "ball" : geom == 1 ? "torus" : "plane";
            if (geom == 2) { o.axis = 2; o.offset = cam.x[2] - 1.0f; }
            else {
                float d = geom == 0 ? 2.5f : 4.0f;
                for (int i = 0; i < 4; ++i) o.c[i] = cam.x[i] + d * cam.f[i];
                o.on_M = true;
                if (geom == 1) { o.a = 0.6f; o.b = 0.15f; }
                act.snap = true;
            }
            objs_.push_back(o);
            sel_ = int(objs_.size()) - 1;
            act.objects_changed = true;
            act.structural = true;
        };
        // без графа режима редактирования (сцены-графы DD-011) структура правится в графе (редактор нод, шаг 6)
        ImGui::BeginDisabled(!fi.instant);
        if (ImGui::Button("+ шар")) add(0);
        ImGui::SameLine();
        if (ImGui::Button("+ тор")) add(1);
        ImGui::SameLine();
        if (ImGui::Button("+ плоскость")) add(2);
        if (sel_ >= 0) {
            if (ImGui::Button("копия") && objs_.size() < size_t(kMaxObj) && objs_[sel_].geom != 3) {
                EdObject o = objs_[sel_];
                o.name += "_copy";
                if (o.geom != 2) o.c[0] += 2.2f * o.a;
                objs_.push_back(o);
                sel_ = int(objs_.size()) - 1;
                act.objects_changed = true;
                act.structural = true;
                act.snap = o.on_M;
            }
            ImGui::SameLine();
            if (ImGui::Button("удалить")) {
                objs_.erase(objs_.begin() + sel_);
                sel_ = -1;
                act.objects_changed = true;
                act.structural = true;
            }
        }
        ImGui::EndDisabled();
        if (!fi.instant) ImGui::TextDisabled("структура сцены-графа правится в графе (.g4d)");
        if (sel_ >= 0 && sel_ < int(objs_.size())) {
            EdObject& o = objs_[sel_];
            bool ch = false, moved = false;
            ImGui::SeparatorText("свойства");
            char name[64];
            std::snprintf(name, sizeof(name), "%s", o.name.c_str());
            if (ImGui::InputText("имя", name, sizeof(name))) {
                o.name = name;
                std::replace(o.name.begin(), o.name.end(), ' ', '_');   // имена — слова директив графа
            }
            const char* geoms[] = {"4D-шар", "тор в листе", "полупространство"};
            if (o.geom == 3) {
                ImGui::TextDisabled("составная форма графа (M36): форма правится в графе");
            } else if (!fi.instant) {
                ImGui::TextDisabled("форма: %s", geoms[o.geom]);
            } else if (ImGui::Combo("форма", &o.geom, geoms, 3)) {
                ch = true;
                act.structural = true;
                if (o.geom == 1 && o.b <= 0) o.b = 0.15f;
            }
            if (o.geom == 3) {
            } else if (o.geom != 2) {
                moved |= ImGui::DragFloat4("центр", o.c, 0.01f);
                if (ImGui::Checkbox("на M (проекция центра)", &o.on_M)) moved = true;
                if (ImGui::Button("перед камерой")) {
                    float d = std::max(2.0f, 4.0f * o.a);
                    for (int i = 0; i < 4; ++i) o.c[i] = cam.x[i] + d * cam.f[i];
                    o.on_M = true;
                    moved = true;
                }
                if (cfg_.flat.has) {
                    ImGui::SameLine();
                    if (ImGui::Button("на другой лист")) { o.c[3] = -o.c[3]; moved = true; }
                }
                if (o.geom == 0) ch |= ImGui::DragFloat("r", &o.a, 0.005f, 0.01f, 100.0f);
                else {
                    ch |= ImGui::DragFloat("R", &o.a, 0.005f, 0.01f, 100.0f);
                    ch |= ImGui::DragFloat("r##t", &o.b, 0.005f, 0.005f, 100.0f);
                }
            } else {
                const char* axes[] = {"x", "y", "z", "w"};
                if (fi.instant && ImGui::Combo("ось", &o.axis, axes, 4)) { ch = true; act.structural = true; }
                ch |= ImGui::DragFloat("смещение", &o.offset, 0.01f);
                bool up = o.sign > 0;
                if (ImGui::Checkbox("внутрь — в сторону «−»", &up)) { o.sign = up ? 1.0f : -1.0f; ch = true; }
            }
            const char* mats[] = {"диффузный", "металл", "излучатель"};
            ch |= ImGui::Combo("материал", &o.mat, mats, 3);
            if (o.mat != 2) ch |= ImGui::ColorEdit3("альбедо", o.albedo);
            if (o.mat == 1) ch |= ImGui::SliderFloat("шероховатость", &o.rough, 0.01f, 1.0f);
            if (o.mat == 2) {
                float s = max3(o.emit), col[3] = {1, 1, 1};
                if (s > 0) for (int i = 0; i < 3; ++i) col[i] = o.emit[i] / s;
                bool ec = ImGui::ColorEdit3("цвет", col);
                ec |= ImGui::DragFloat("яркость", &s, 0.1f, 0.0f, 1000.0f);
                if (ec) {
                    for (int i = 0; i < 3; ++i) o.emit[i] = col[i] * s;
                    ch = true;
                }
                if (is_light(sel_)) ImGui::TextColored(ImVec4(0.5f, 1, 0.5f, 1), "NEE-источник (M34)");
                else ImGui::TextDisabled(cfg_.flat.has ? "не в плоской зоне / не на листе: только BSDF (M34)"
                                                       : "плоских зон нет: свет только через BSDF");
            }
            if (ch || moved) act.objects_changed = true;
            if (moved && o.on_M && o.geom != 2 && o.geom != 3) act.snap = true;
        }
        ImGui::EndDisabled();
    }
    ImGui::End();
    return act;
}
