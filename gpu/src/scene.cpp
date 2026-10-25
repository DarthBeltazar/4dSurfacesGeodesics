// Модель сцены окна (DD-010) — см. scene.hpp.
#include "scene.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace geo4d {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kEllisEllMax = 12.0;             // geo4d.scenefile.ELLIS_ELL_MAX
constexpr double kLengthEps = 1e-150;             // ops.length во float64 (M6)

const char* kCompactSpaces[] = {"S3", "spheritorus", "torisphere", "tiger"};

// дуальное число по 4 направлениям: значение и градиент (как jax.grad на формулах geo4d.shapes)
struct D {
    double a = 0;
    Vec4 g{0, 0, 0, 0};
};
D cst(double c) { return D{c, {0, 0, 0, 0}}; }
D operator+(const D& x, const D& y) { D r{x.a + y.a}; for (int i = 0; i < 4; ++i) r.g[i] = x.g[i] + y.g[i]; return r; }
D operator-(const D& x, const D& y) { D r{x.a - y.a}; for (int i = 0; i < 4; ++i) r.g[i] = x.g[i] - y.g[i]; return r; }
D operator*(const D& x, const D& y) {
    D r{x.a * y.a};
    for (int i = 0; i < 4; ++i) r.g[i] = x.g[i] * y.a + x.a * y.g[i];
    return r;
}
D operator*(double k, const D& x) { D r{k * x.a}; for (int i = 0; i < 4; ++i) r.g[i] = k * x.g[i]; return r; }
D operator*(const D& x, double k) { D r{x.a * k}; for (int i = 0; i < 4; ++i) r.g[i] = x.g[i] * k; return r; }
D operator-(const D& x, double k) { D r = x; r.a -= k; return r; }
D operator-(double k, const D& x) { D r{k - x.a}; for (int i = 0; i < 4; ++i) r.g[i] = -x.g[i]; return r; }
D unary(const D& x, double f0, double f1) { D r{f0}; for (int i = 0; i < 4; ++i) r.g[i] = f1 * x.g[i]; return r; }
D dsqrt(const D& x) { double s = std::sqrt(x.a); return unary(x, s, 0.5 / s); }
D dexp(const D& x) { double e = std::exp(x.a); return unary(x, e, e); }
D dcosh(const D& x) { return unary(x, std::cosh(x.a), std::sinh(x.a)); }
D drecip(const D& x) { return unary(x, 1.0 / x.a, -1.0 / (x.a * x.a)); }

// M6: защищённая длина √(max(|u|², ε²))
D length(std::initializer_list<D> c) {
    auto it = c.begin();
    D q = *it * *it;
    for (++it; it != c.end(); ++it) q = q + *it * *it;
    if (q.a < kLengthEps * kLengthEps) q = cst(kLengthEps * kLengthEps);
    return dsqrt(q);
}

// M11: ψ(u) = 1 − exp(1 − 1/(1 − u)) при q = 1 − u > 1e-2, иначе 1
D compact_psi(const D& u) {
    D q = 1.0 - u;
    if (!(q.a > 1e-2)) return cst(1.0);
    return 1.0 - dexp(1.0 - drecip(q));
}
D compact_h(const D& rho, double b, double rho0, double W) { return (W * W) * compact_psi((rho - b) * (1.0 / (rho0 - b))); }

D eval(const Manifold& m, const std::array<D, 4>& p) {
    const std::string& k = m.kind;
    if (k == "S3") return length({p[0], p[1], p[2], p[3]}) - m.p("R");
    if (k == "spheritorus") return length({length({p[0], p[1], p[2]}) - m.p("R"), p[3]}) - m.p("r");
    if (k == "torisphere") return length({length({p[0], p[1]}) - m.p("R"), p[2], p[3]}) - m.p("r");
    if (k == "tiger") return length({length({p[0], p[1]}) - m.p("R1"), length({p[2], p[3]}) - m.p("R2")}) - m.p("r");
    if (k == "ellis") return length({p[0], p[1], p[2]}) - m.p("b") * dcosh(p[3] * (1.0 / m.p("b")));
    if (k == "compact_wh")
        return p[3] * p[3] - compact_h(length({p[0], p[1], p[2]}), m.p("b"), m.p("rho0"), m.p("W"));
    if (k == "two_compact_wh") {
        const auto &c1 = m.pv("c1"), &c2 = m.pv("c2");
        D r1 = length({p[0] - c1[0], p[1] - c1[1], p[2] - c1[2]});
        D r2 = length({p[0] - c2[0], p[1] - c2[1], p[2] - c2[2]});
        double b = m.p("b"), rho0 = m.p("rho0"), W = m.p("W");
        return p[3] * p[3] - (compact_h(r1, b, rho0, W) + compact_h(r2, b, rho0, W) - W * W);
    }
    return p[3];                                  // гиперплоскость w = 0
}

double num(const nlohmann::json& j, const char* key, double dflt) {
    auto it = j.find(key);
    return it == j.end() ? dflt : it->get<double>();
}

template <size_t N>
std::array<double, N> vec(const nlohmann::json& j, const char* key, std::array<double, N> dflt) {
    auto it = j.find(key);
    if (it == j.end()) return dflt;
    if (!it->is_array() || it->size() != N) throw std::runtime_error(std::string("сцена: ") + key + " — ожидался вектор из " + std::to_string(N) + " чисел");
    std::array<double, N> r;
    for (size_t i = 0; i < N; ++i) r[i] = (*it)[i].get<double>();
    return r;
}

}  // namespace

bool is_compact_space(const std::string& kind) {
    for (const char* k : kCompactSpaces) if (kind == k) return true;
    return false;
}

Params default_params(const std::string& kind) {
    if (kind == "hyperplane") return {};
    if (kind == "S3") return {{"R", {1.0}}};
    if (kind == "spheritorus" || kind == "torisphere") return {{"R", {1.0}}, {"r", {0.4}}};
    if (kind == "tiger") return {{"R1", {1.2}}, {"R2", {0.8}}, {"r", {0.3}}};
    if (kind == "ellis") return {{"b", {1.0}}};
    if (kind == "compact_wh") return {{"b", {1.0}}, {"rho0", {3.0}}, {"W", {1.5}}};
    if (kind == "two_compact_wh")
        return {{"c1", {-4.0, 0.0, 0.0}}, {"c2", {4.0, 0.0, 0.0}}, {"b", {1.0}}, {"rho0", {3.0}}, {"W", {1.5}}};
    throw std::runtime_error("нет многообразия " + kind);
}

const std::vector<double>& Manifold::pv(const std::string& key) const {
    for (auto& [k, v] : params) if (k == key) return v;
    throw std::logic_error("нет параметра " + key + " у " + kind);
}
double Manifold::p(const std::string& key, size_t i) const { return pv(key).at(i); }

Manifold make_manifold(const std::string& kind, const Params& overrides) {
    Manifold m;
    m.kind = kind;
    m.params = default_params(kind);
    for (auto& [k, v] : overrides)
        for (auto& [dk, dv] : m.params)
            if (dk == k) {
                if (v.size() != dv.size()) throw std::runtime_error(kind + ": параметр " + k + " — неверная длина");
                dv = v;
            }
    // geo4d.scenefile.check_params
    auto pos = [&](std::initializer_list<const char*> ks) { for (auto k : ks) if (!(m.p(k) > 0)) return false; return true; };
    bool ok = true;
    std::string why;
    if (kind == "S3") { ok = pos({"R"}); why = "R > 0"; }
    else if (kind == "spheritorus" || kind == "torisphere") { ok = pos({"R", "r"}) && m.p("r") < m.p("R"); why = "0 < r < R"; }
    else if (kind == "tiger") {
        ok = pos({"R1", "R2", "r"}) && m.p("r") < m.p("R1") && m.p("r") < m.p("R2");
        why = "0 < r < R1, R2";
    } else if (kind == "ellis") { ok = pos({"b"}); why = "b > 0"; }
    else if (kind == "compact_wh" || kind == "two_compact_wh") {
        ok = pos({"b", "rho0", "W"}) && m.p("b") < m.p("rho0");
        why = "0 < b < ρ₀, W > 0";
        if (ok && kind == "two_compact_wh") {
            double d2 = 0;
            for (int i = 0; i < 3; ++i) d2 += (m.p("c1", i) - m.p("c2", i)) * (m.p("c1", i) - m.p("c2", i));
            ok = std::sqrt(d2) > 2 * m.p("rho0");
            why = "|c₁ − c₂| > 2ρ₀";
        }
    }
    if (!ok) throw std::runtime_error(kind + ": недопустимые параметры (нужно " + why + ")");
    // geo4d.shapes: Shape.name и scale; geo4d.gpu.SCENE_IDS; geo4d.scenefile.build: плоские зоны и мета
    struct K { const char* kind; const char* name; int id; };
    static const K ks[] = {{"hyperplane", "hyperplane", 0}, {"S3", "S3", 1}, {"spheritorus", "spheritorus", 5},
                           {"torisphere", "torisphere", 6}, {"tiger", "tiger", 7}, {"ellis", "ellis", 8},
                           {"compact_wh", "compact_wh", 9}, {"two_compact_wh", "two_compact_wh", 10}};
    for (auto& k : ks) if (kind == k.kind) { m.shape_name = k.name; m.scene_id = k.id; }
    if (kind == "S3") m.scale = m.p("R");
    else if (kind == "spheritorus" || kind == "torisphere" || kind == "tiger") m.scale = m.p("r");
    else if (kind == "ellis" || kind == "compact_wh" || kind == "two_compact_wh") m.scale = m.p("b");
    if (kind == "hyperplane") { m.flat.has = true; m.flat.sheets = {0.0}; }
    else if (kind == "compact_wh") {
        m.flat = {true, {m.p("W"), -m.p("W")}, {{0, 0, 0}}, m.p("rho0")};
    } else if (kind == "two_compact_wh") {
        m.flat = {true, {m.p("W"), -m.p("W")},
                  {{m.p("c1", 0), m.p("c1", 1), m.p("c1", 2)}, {m.p("c2", 0), m.p("c2", 1), m.p("c2", 2)}}, m.p("rho0")};
    } else if (kind == "ellis") {
        m.exit = 1;
        m.has_ell_max = true;
        m.ell_max = kEllisEllMax;
    } else if (is_compact_space(kind)) {
        m.compact = true;
        // длина пути — от размера пространства: size = R или R1/1.2 (geo4d.scenefile.build)
        bool has_R = false;
        for (auto& [k, v] : m.params) if (k == "R") has_R = true;
        double size = has_R ? m.p("R") : m.p("R1") / 1.2;
        m.has_ell_max = true;
        m.ell_max = 1e9;
        m.has_s_max = true;
        m.s_max = 8 * kPi * size;
    }
    return m;
}

double Object::bound() const {
    if (kind == "ball4") return r;
    if (kind == "torus") return R + r;
    return std::numeric_limits<double>::infinity();
}

double sdf(const Manifold& m, const Vec4& x, Vec4* grad) {
    std::array<D, 4> p;
    for (int i = 0; i < 4; ++i) { p[i].a = x[i]; p[i].g = {0, 0, 0, 0}; p[i].g[i] = 1; }
    D f = eval(m, p);
    if (grad) *grad = f.g;
    return f.a;
}

Vec4 project(const Manifold& m, Vec4 x, int iters) {
    for (int it = 0; it < iters; ++it) {
        Vec4 g;
        double f = sdf(m, x, &g);
        double gg = g[0] * g[0] + g[1] * g[1] + g[2] * g[2] + g[3] * g[3];
        double k = f / gg;
        for (int i = 0; i < 4; ++i) x[i] = x[i] - k * g[i];
    }
    return x;
}

double Camera::tf() const { return std::tan(fov * kPi / 180.0 / 2); }
double Camera::theta_pix() const { return 2 * tf() / width; }

Camera make_camera(const Manifold& m, const CameraHints& h, int width, int height, double fov) {
    auto dot = [](const Vec4& a, const Vec4& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]; };
    auto norm = [&](const Vec4& a) { return std::sqrt(dot(a, a)); };
    Vec4 n;
    sdf(m, h.x, &n);
    double nn = norm(n);
    for (auto& a : n) a /= nn;
    // двойной Грам–Шмидт; при вырожденной подсказке — первая координатная ось с устойчивым остатком
    auto orth = [&](const Vec4& hint, const std::vector<Vec4>& basis) {
        std::vector<Vec4> cands{hint, {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
        for (auto& cand : cands) {
            Vec4 c = cand;
            for (int it = 0; it < 2; ++it)
                for (auto& b : basis) {
                    double k = dot(c, b);
                    for (int i = 0; i < 4; ++i) c[i] -= k * b[i];
                }
            double l = norm(c);
            if (l > 1e-6 * std::max(norm(cand), 1e-300)) {
                for (auto& a : c) a /= l;
                return c;
            }
        }
        throw std::runtime_error("не удалось построить касательный базис");
    };
    Camera c;
    c.x = h.x;
    c.f = orth(h.f, {n});
    c.u = orth(h.up, {n, c.f});
    if (h.has_right) c.r = orth(h.right, {n, c.f, c.u});
    else {
        // дополнение до базиса: одномерное — любая ненулевая проекция даёт ±r; знак — det(n, f, r, u) > 0
        c.r = orth(Vec4{0, 0, 0, 0}, {n, c.f, c.u});
        const Vec4* M[4] = {&n, &c.f, &c.r, &c.u};
        double det = 0;
        for (int j = 0; j < 4; ++j) {
            int cc[3], k = 0;
            for (int q = 0; q < 4; ++q) if (q != j) cc[k++] = q;
            auto e = [&](int row, int col) { return (*M[row])[cc[col]]; };
            double minor = e(1, 0) * (e(2, 1) * e(3, 2) - e(2, 2) * e(3, 1)) - e(1, 1) * (e(2, 0) * e(3, 2) - e(2, 2) * e(3, 0)) +
                           e(1, 2) * (e(2, 0) * e(3, 1) - e(2, 1) * e(3, 0));
            det += ((j % 2) ? -1.0 : 1.0) * (*M[0])[j] * minor;
        }
        if (det < 0) for (auto& a : c.r) a = -a;
    }
    c.fov = fov;
    c.width = width;
    c.height = height;
    return c;
}

std::vector<int> nee_lights(const std::vector<Object>& objs, const FlatZones& fz) {
    std::vector<int> out;
    if (!fz.has) return out;
    for (size_t k = 0; k < objs.size(); ++k) {
        const Object& o = objs[k];
        if (o.mat.kind != "emit" || o.kind != "ball4") continue;
        double dmin = std::numeric_limits<double>::infinity();
        for (double s : fz.sheets) dmin = std::min(dmin, std::abs(o.center[3] - s));
        if (dmin >= 1e-6) continue;
        bool inside = false;
        for (auto& c : fz.centers) {
            double d = std::sqrt((c[0] - o.center[0]) * (c[0] - o.center[0]) + (c[1] - o.center[1]) * (c[1] - o.center[1]) +
                                 (c[2] - o.center[2]) * (c[2] - o.center[2]));
            if (d < fz.rho0 + o.r) inside = true;
        }
        if (inside) continue;
        out.push_back(int(k));
    }
    return out;
}

Scene load_scene(const std::string& path, bool project) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("не удалось открыть " + path);
    nlohmann::json d;
    try {
        d = nlohmann::json::parse(f);
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error(path + ": " + e.what());
    }
    Scene sc;
    sc.path = path;
    const auto& md = d.at("manifold");
    Params ov;
    if (auto it = md.find("params"); it != md.end())
        for (auto& [k, v] : it->items()) {
            std::vector<double> vals;
            if (v.is_array()) for (auto& a : v) vals.push_back(a.get<double>());
            else vals.push_back(v.get<double>());
            ov.emplace_back(k, vals);
        }
    sc.mf = make_manifold(md.at("kind").get<std::string>(), ov);
    if (auto it = d.find("env"); it != d.end() && !it->is_null()) sc.env_field = it->get<std::string>();
    sc.env = !sc.env_field.empty() ? sc.env_field : (sc.mf.compact ? "none" : "procedural");
    if (auto it = d.find("objects"); it != d.end())
        for (auto& o : *it) {
            Object ob;
            ob.kind = o.at("kind").get<std::string>();
            ob.name = o.contains("name") ? o["name"].get<std::string>() : ob.kind;
            ob.on_M = o.contains("on_M") && o["on_M"].get<bool>();
            if (auto mt = o.find("material"); mt != o.end()) {
                ob.mat.kind = mt->value("kind", std::string("diffuse"));
                ob.mat.albedo = vec<3>(*mt, "albedo", ob.mat.albedo);
                ob.mat.rough = num(*mt, "rough", 0.3);
                ob.mat.emit = vec<3>(*mt, "emit", ob.mat.emit);
            }
            if (ob.kind == "plane") {
                ob.axis = o.at("axis").get<int>();
                ob.offset = o.at("offset").get<double>();
                ob.sign = num(o, "sign", 1.0);
            } else if (ob.kind == "ball4" || ob.kind == "torus") {
                Vec4 c = vec<4>(o, "center", Vec4{0, 0, 0, 0});
                if (!o.contains("center")) throw std::runtime_error("сцена: у объекта нет center");
                ob.center = ob.on_M && project ? geo4d::project(sc.mf, c) : c;
                if (ob.kind == "ball4") ob.r = o.at("r").get<double>();
                else { ob.R = o.at("R").get<double>(); ob.r = o.at("r").get<double>(); }
            } else throw std::runtime_error("нет вида объекта " + ob.kind);
            sc.objs.push_back(ob);
        }
    nlohmann::json c = d.contains("camera") ? d["camera"] : nlohmann::json::object();
    sc.cam.x = vec<4>(c, "x", Vec4{0, 0, 0, 0});
    if (project) sc.cam.x = geo4d::project(sc.mf, sc.cam.x);
    sc.cam.f = vec<4>(c, "f", Vec4{0, 1, 0, 0});
    sc.cam.up = vec<4>(c, "up", Vec4{0, 0, 1, 0});
    sc.cam.has_right = c.contains("right");
    if (sc.cam.has_right) sc.cam.right = vec<4>(c, "right", Vec4{0, 0, 0, 0});
    sc.cam.fov = num(c, "fov", 75.0);
    sc.speed = num(d, "speed", 1.0);
    return sc;
}

}  // namespace geo4d
