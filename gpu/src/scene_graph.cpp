// Граф сцены (DD-011) — см. scene_graph.hpp.
#include "scene_graph.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <set>
#include <stdexcept>

namespace geo4d {

namespace {

std::string g_lib_dir;
std::once_flag g_lib_once;
NodeLibrary g_lib;

[[noreturn]] void fail(const std::string& node, const std::string& msg) {
    throw std::runtime_error("граф сцены" + (node.empty() ? std::string() : ", нода «" + node + "»") + ": " + msg);
}

int components(const std::string& t) { return t == "vec3" || t == "color" ? 3 : t == "vec4" ? 4 : 1; }

std::pair<std::string, std::string> split_port(const std::string& s) {
    auto d = s.rfind('.');
    if (d == std::string::npos) throw std::runtime_error("связь «" + s + "»: нужно <id>.<порт>");
    return {s.substr(0, d), s.substr(d + 1)};
}

template <size_t N>
std::array<double, N> vec(const nlohmann::json& j, const char* key, std::array<double, N> dflt) {
    auto it = j.find(key);
    if (it == j.end()) return dflt;
    if (!it->is_array() || it->size() != N)
        throw std::runtime_error(std::string("сцена: ") + key + " — ожидался вектор из " + std::to_string(N) + " чисел");
    std::array<double, N> r;
    for (size_t i = 0; i < N; ++i) r[i] = (*it)[i].get<double>();
    return r;
}

// литерал числа для формулы: как repr Python; отрицательный — в скобках
std::string lit(double v) {
    char b[64];
    std::snprintf(b, sizeof(b), "%.17g", v);
    std::string s = b;
    if (s.find_first_of(".eEn") == std::string::npos) s += ".0";
    return v < 0 ? "(" + s + ")" : s;
}

}  // namespace

const NodeParamDef* NodeDef::param(const std::string& n) const {
    for (auto& p : params) if (p.name == n) return &p;
    return nullptr;
}
const NodeInputDef* NodeDef::input(const std::string& n) const {
    for (auto& i : inputs) if (i.name == n) return &i;
    return nullptr;
}

void NodeLibrary::load_dir(const std::string& dir) {
    std::vector<std::filesystem::path> files;
    for (auto& e : std::filesystem::directory_iterator(dir))
        if (e.path().extension() == ".json") files.push_back(e.path());
    std::sort(files.begin(), files.end());
    for (auto& f : files) {
        std::ifstream in(f, std::ios::binary);
        nlohmann::json j;
        try {
            j = nlohmann::json::parse(in);
        } catch (const std::exception& e) {
            throw std::runtime_error("библиотека нод " + f.string() + ": " + e.what());
        }
        std::vector<nlohmann::json> items = j.is_array() ? std::vector<nlohmann::json>(j.begin(), j.end())
                                                         : std::vector<nlohmann::json>{j};
        for (auto& n : items) {
            NodeDef d;
            d.type = n.at("type").get<std::string>();
            d.group = n.value("group", std::string());
            d.title = n.value("title", d.type);
            d.output = n.value("output", std::string());
            d.formula = n.value("formula", std::string());
            d.legacy_kind = n.value("legacy_kind", std::string());
            d.material = n.value("material", std::string());
            for (auto& p : n.value("params", nlohmann::json::array())) {
                NodeParamDef pd;
                pd.name = p.at("name").get<std::string>();
                pd.type = p.value("type", std::string("float"));
                pd.def = p.value("default", nlohmann::json());
                pd.min = p.value("min", 0.0);
                pd.max = p.value("max", 0.0);
                pd.structural = p.value("structural", false);
                pd.hidden = p.value("hidden", false);
                d.params.push_back(pd);
            }
            for (auto& i : n.value("inputs", nlohmann::json::array()))
                d.inputs.push_back({i.at("name").get<std::string>(), i.at("type").get<std::string>(), i.value("multi", false)});
            const nlohmann::json rows = n.value("rows", nlohmann::json::object());   // items() у временного — висячие ссылки
            for (auto& [k, v] : rows.items()) d.rows[k] = v.get<std::string>();
            if (find(d.type)) throw std::runtime_error("библиотека нод: тип «" + d.type + "» объявлен дважды");
            defs_.push_back(d);
        }
    }
}

const NodeDef* NodeLibrary::find(const std::string& type) const {
    for (auto& d : defs_) if (d.type == type) return &d;
    return nullptr;
}

void set_node_library_dir(const std::string& dir) { g_lib_dir = dir; }

NodeLibrary& node_library() {
    std::call_once(g_lib_once, [] {
        if (g_lib_dir.empty()) throw std::runtime_error("каталог библиотеки нод не задан");
        g_lib.load_dir(g_lib_dir);
    });
    return g_lib;
}

std::map<std::string, double> node_values(const NodeDef& def, const nlohmann::json& params) {
    std::map<std::string, double> v;
    for (auto& p : def.params) {
        if (p.type == "string") continue;
        nlohmann::json val = params.contains(p.name) ? params.at(p.name) : p.def;
        int n = components(p.type);
        if (n == 1) {
            if (!val.is_number()) fail("", "параметр «" + p.name + "» — нужно число");
            v[p.name] = val.get<double>();
        } else {
            if (!val.is_array() || int(val.size()) != n) fail("", "параметр «" + p.name + "» — нужен вектор из " + std::to_string(n));
            for (int i = 0; i < n; ++i) v[p.name + "." + "xyzw"[i]] = val[i].get<double>();
        }
    }
    return v;
}

// ---- перевод version 1 → 2 ------------------------------------------------------------------------------------

nlohmann::json migrate_v1(const nlohmann::json& v1) {
    using J = nlohmann::json;
    J nodes = J::array(), links = J::array();
    const J& md = v1.at("manifold");
    nodes.push_back({{"id", "manifold"}, {"type", "mf." + md.at("kind").get<std::string>()},
                     {"params", md.value("params", J::object())}});
    nodes.push_back({{"id", "scene"}, {"type", "out.scene"}, {"params", J::object()}});
    links.push_back({{"from", "manifold.out"}, {"to", "scene.manifold"}});
    int k = 0;
    for (auto& o : v1.value("objects", J::array())) {
        const std::string kind = o.at("kind").get<std::string>(), sk = std::to_string(k);
        J sp = J::object();
        std::string type;
        if (kind == "ball4") { type = "obj.ball4"; sp = {{"c", o.at("center")}, {"r", o.at("r")}}; }
        else if (kind == "torus") { type = "obj.torus"; sp = {{"c", o.at("center")}, {"R", o.at("R")}, {"r", o.at("r")}}; }
        else if (kind == "plane") {
            type = "obj.halfspace";
            sp = {{"axis", o.at("axis")}, {"offset", o.at("offset")}, {"sign", o.value("sign", 1.0)}};
        } else throw std::runtime_error("нет вида объекта " + kind);
        nodes.push_back({{"id", "shape_" + sk}, {"type", type}, {"params", sp}});
        std::string shape = "shape_" + sk;
        if (o.value("on_M", false)) {
            nodes.push_back({{"id", "onM_" + sk}, {"type", "xf.on_M"}, {"params", J::object()}});
            links.push_back({{"from", shape + ".out"}, {"to", "onM_" + sk + ".shape"}});
            shape = "onM_" + sk;
        }
        const J m = o.value("material", J::object());   // значения v1 — явно (умолчания нод другие)
        nodes.push_back({{"id", "mat_" + sk}, {"type", "mat." + m.value("kind", std::string("diffuse"))},
                         {"params", {{"albedo", m.value("albedo", J::array({0.8, 0.8, 0.8}))}, {"rough", m.value("rough", 0.3)},
                                     {"emit", m.value("emit", J::array({0.0, 0.0, 0.0}))}}}});
        nodes.push_back({{"id", "obj_" + sk}, {"type", "out.object"}, {"params", {{"name", o.value("name", kind)}}}});
        links.push_back({{"from", shape + ".out"}, {"to", "obj_" + sk + ".shape"}});
        links.push_back({{"from", "mat_" + sk + ".out"}, {"to", "obj_" + sk + ".material"}});
        links.push_back({{"from", "obj_" + sk + ".out"}, {"to", "scene.objects"}});
        ++k;
    }
    J doc = {{"format", "geo4d"}, {"version", 2}, {"graphs", {{"main", {{"nodes", nodes}, {"links", links}}}}},
             {"camera", v1.value("camera", J::object())}, {"settings", J::object()}};
    if (v1.contains("speed")) doc["settings"]["speed"] = v1["speed"];
    if (v1.contains("env") && !v1["env"].is_null()) doc["settings"]["env"] = v1["env"];
    return doc;
}

// ---- граф → модель окна --------------------------------------------------------------------------------------

Scene scene_from_document(const nlohmann::json& doc0, const std::string& path, bool project) {
    using J = nlohmann::json;
    const J doc = doc0.contains("manifold") && !doc0.contains("graphs") ? migrate_v1(doc0) : doc0;
    if (doc.value("version", 0) != 2) fail("", "неизвестная версия документа");
    const J& g = doc.at("graphs").at("main");
    const NodeLibrary& lib = node_library();
    struct N { const NodeDef* def; const J* params; };
    std::map<std::string, N> nodes;
    std::vector<std::string> order;
    for (auto& n : g.at("nodes")) {
        std::string id = n.at("id").get<std::string>(), type = n.at("type").get<std::string>();
        const NodeDef* d = lib.find(type);
        if (!d) fail(id, "нет типа ноды «" + type + "» в библиотеке");
        if (nodes.count(id)) fail(id, "id повторяется");
        static const J kEmpty = J::object();
        nodes[id] = {d, n.contains("params") ? &n.at("params") : &kEmpty};
        order.push_back(id);
    }
    // связи: вход → источники (по порядку); типы портов
    std::map<std::pair<std::string, std::string>, std::vector<std::string>> in;
    for (auto& l : g.value("links", J::array())) {
        auto [fn, fp] = split_port(l.at("from").get<std::string>());
        auto [tn, tp] = split_port(l.at("to").get<std::string>());
        if (!nodes.count(fn)) fail(fn, "нет такой ноды (связь)");
        if (!nodes.count(tn)) fail(tn, "нет такой ноды (связь)");
        if (fp != "out") fail(fn, "у ноды один выход — out");
        const NodeInputDef* id = nodes[tn].def->input(tp);
        if (!id) fail(tn, "нет входа «" + tp + "»");
        if (id->type != nodes[fn].def->output)
            fail(tn, "вход «" + tp + "» (" + id->type + ") нельзя соединить с выходом «" + fn + "» (" + nodes[fn].def->output + ")");
        auto& v = in[{tn, tp}];
        if (!id->multi && !v.empty()) fail(tn, "ко входу «" + tp + "» уже подключено");
        v.push_back(fn);
    }
    auto source = [&](const std::string& node, const std::string& port) -> std::string {
        auto it = in.find({node, port});
        if (it == in.end() || it->second.empty()) fail(node, "не подключён вход «" + port + "»");
        return it->second[0];
    };
    std::string out_scene;
    for (auto& id : order)
        if (nodes[id].def->type == "out.scene") {
            if (!out_scene.empty()) fail(id, "в графе две ноды «Сцена»");
            out_scene = id;
        }
    if (out_scene.empty()) fail("", "нет ноды «Сцена» (out.scene)");

    Scene sc;
    sc.path = path;
    // многообразие
    const std::string mid = source(out_scene, "manifold");
    const N& mn = nodes[mid];
    if (mn.def->type == "mf.throat" || mn.def->type == "mf.two_sheets") {
        // M35: два листа ±W и горловины (cᵢ, bᵢ, ρ₀) с непересекающимися шарами носителей
        struct T { std::string id; double cx, cy, cz, b, rho0; };
        std::vector<T> th;
        std::string cur = mid;
        while (nodes[cur].def->type == "mf.throat") {
            auto v = node_values(*nodes[cur].def, *nodes[cur].params);
            th.push_back({cur, v.at("c.x"), v.at("c.y"), v.at("c.z"), v.at("b"), v.at("rho0")});
            cur = source(cur, "base");
        }
        if (nodes[cur].def->type != "mf.two_sheets") fail(cur, "горловина вставляется только в «Два листа» (или в цепочку горловин)");
        std::reverse(th.begin(), th.end());       // порядок от базы
        const double W = node_values(*nodes[cur].def, *nodes[cur].params).at("W");
        if (!(W > 0)) fail(cur, "нужно W > 0");
        Manifold m;
        m.kind = "sheets";
        m.shape_name = "sheets";
        m.scene_id = 12;
        m.exit = 2;
        m.flat.has = true;
        m.flat.sheets = {W, -W};
        m.scale = W;
        std::string f = "w * w - " + lit(W) + " * " + lit(W);
        for (size_t i = 0; i < th.size(); ++i) {
            const T& t = th[i];
            if (!(t.b > 0 && t.b < t.rho0)) fail(t.id, "нужно 0 < b < ρ₀");
            if (i > 0 && t.rho0 != th[0].rho0)
                fail(t.id, "пока у всех горловин один ρ₀ (трассировщик: один радиус плоских зон, PT_RHO0)");
            for (size_t j = 0; j < i; ++j) {
                const T& u = th[j];
                double d = std::sqrt((t.cx - u.cx) * (t.cx - u.cx) + (t.cy - u.cy) * (t.cy - u.cy) + (t.cz - u.cz) * (t.cz - u.cz));
                if (!(d > t.rho0 + u.rho0)) fail(t.id, "шар носителя пересекается с горловиной «" + u.id + "» (нужно |cᵢ − cⱼ| > ρ₀ᵢ + ρ₀ⱼ)");
            }
            m.flat.centers.push_back({t.cx, t.cy, t.cz});
            m.flat.rho0 = t.rho0;
            m.scale = i == 0 ? t.b : std::min(m.scale, t.b);
            f += " + (" + lit(W) + " * " + lit(W) + " - compact_h(length(x - " + lit(t.cx) + ", y - " + lit(t.cy) + ", z - " +
                 lit(t.cz) + "), " + lit(t.b) + ", " + lit(t.rho0) + ", " + lit(W) + "))";
        }
        m.formula = f;
        m.expr = parse_expr(f, {});
        sc.mf = m;
        sc.mf_node = mn.def->type;
    } else if (mn.def->type == "mf.formula") {
        // M37: вид неизвестен — шаг ≤ ℓ_s, без плоских зон (NEE) и фона, длина пути ≤ s_max
        auto v = node_values(*mn.def, *mn.params);
        const std::string src = mn.params->value("formula", mn.def->param("formula")->def.get<std::string>());
        ExprPtr e;
        try {
            e = parse_expr(src, {"a", "b", "c", "d"});
        } catch (const ExprError& ex) {
            fail(mid, std::string("формула, символ ") + std::to_string(ex.pos + 1) + ": " + ex.what());
        }
        if (!e->field) fail(mid, "формула не зависит от x, y, z, w");
        // параметры a…d — числами в текст (до слотов параметров, шаг 4)
        std::string f;
        {
            std::map<std::string, std::string> pm;
            for (const char* k : {"a", "b", "c", "d"}) pm[k] = lit(v.at(k));
            f = emit_slang(e, pm, {"x", "y", "z", "w"});
        }
        Manifold m;
        m.kind = "formula";
        m.shape_name = "formula";
        m.scene_id = 12;
        m.exit = 2;
        m.compact = true;
        m.scale = v.at("scale");
        if (!(m.scale > 0)) fail(mid, "нужно scale > 0");
        m.has_s_max = true;
        m.s_max = v.at("s_max");
        m.has_ell_max = true;
        m.ell_max = 1e9;
        m.formula = src;
        std::map<std::string, double> vals;
        for (const char* k : {"a", "b", "c", "d"}) vals[k] = v.at(k);
        // вычисление на CPU — по исходной формуле с параметрами (eval_expr подставит их)
        m.expr = e;
        m.expr_vals = vals;
        sc.mf = m;
        sc.mf_node = mn.def->type;
        sc.mf_values = vals;
        (void)f;
    } else {
    if (mn.def->legacy_kind.empty()) fail(mid, "многообразие типа «" + mn.def->type + "» не собирается в окне");
    auto mv = node_values(*mn.def, *mn.params);
    Params mp;
    for (auto& p : mn.def->params) {
        std::vector<double> v;
        int n = components(p.type);
        if (n == 1) v.push_back(mv.at(p.name));
        else for (int i = 0; i < n; ++i) v.push_back(mv.at(p.name + "." + "xyzw"[i]));
        mp.emplace_back(p.name, v);
    }
    sc.mf = make_manifold(mn.def->legacy_kind, mp);
    sc.mf_node = mn.def->type;
    sc.mf_values = mv;
    }
    const J settings = doc.value("settings", J::object());
    sc.env_field = settings.contains("env") && !settings["env"].is_null() ? settings["env"].get<std::string>() : "";
    sc.env = !sc.env_field.empty() ? sc.env_field : (sc.mf.compact ? "none" : "procedural");
    sc.speed = settings.value("speed", 1.0);
    // составная форма (M36): выражение g и ограничивающая сфера {g < δ} (радиус inf — не ограничена)
    struct Bound { Vec4 c{0, 0, 0, 0}; double r = std::numeric_limits<double>::infinity(); };
    struct Shape { ExprPtr g; std::function<Bound(double)> bound; };
    const double kLn2 = 0.69314718055994531;
    std::function<Shape(const std::string&)> shape_of = [&](const std::string& id) -> Shape {
        const N& n = nodes[id];
        const std::string& t = n.def->type;
        auto v = node_values(*n.def, *n.params);
        auto num = [](double x) { return mk_num(x); };
        if (!n.def->legacy_kind.empty()) {        // примитив: формула библиотеки с числами
            std::string f = n.def->formula;
            std::vector<std::string> names;
            for (auto& p : n.def->params) {
                if (p.type == "int") {
                    const std::string key = "{" + p.name + "}";
                    for (size_t at; (at = f.find(key)) != std::string::npos;)
                        f.replace(at, key.size(), std::string(1, "xyzw"[int(v.at(p.name)) & 3]));
                }
            }
            for (auto& [k, x] : v) names.push_back(k);
            ExprPtr g = bind_params(parse_expr(f, names), v);
            Bound b;
            if (t == "obj.ball4" || t == "obj.torus") {
                b.c = {v.at("c.x"), v.at("c.y"), v.at("c.z"), v.at("c.w")};
                const double r0 = t == "obj.ball4" ? v.at("r") : v.at("R") + v.at("r");
                return {g, [b, r0](double d) { Bound q = b; q.r = r0 + d; return q; }};
            }
            return {g, [](double) { return Bound{}; }};          // полупространство не ограничено
        }
        if (t == "obj.union" || t == "obj.intersect" || t == "obj.subtract") {
            Shape a = shape_of(source(id, "a")), bb = shape_of(source(id, "b"));
            const double k = v.at("k");
            if (!(k > 0)) fail(id, "нужно k > 0");
            ExprPtr g;
            if (t == "obj.union") g = mk_call("smin", {a.g, bb.g, num(k)});
            else {
                // smax_k(a, b) − k·ln 2 = −smin_k(−a, −b) − k·ln 2; вычитание — пересечение с −b
                ExprPtr nb = t == "obj.subtract" ? bb.g : mk_op(Expr::Neg, bb.g);
                g = mk_op(Expr::Sub, mk_op(Expr::Neg, mk_call("smin", {mk_op(Expr::Neg, a.g), nb, num(k)})), num(k * kLn2));
            }
            auto fa = a.bound, fb = bb.bound;
            std::function<Bound(double)> bf;
            if (t == "obj.union")
                bf = [fa, fb, k, kLn2](double d) {
                    Bound p = fa(d + k * kLn2), q = fb(d + k * kLn2);
                    if (std::isinf(p.r) || std::isinf(q.r)) return Bound{};
                    double dd = 0;
                    for (int i = 0; i < 4; ++i) dd += (q.c[i] - p.c[i]) * (q.c[i] - p.c[i]);
                    dd = std::sqrt(dd);
                    if (dd + q.r <= p.r) return p;
                    if (dd + p.r <= q.r) return q;
                    Bound e;
                    e.r = 0.5 * (dd + p.r + q.r);
                    for (int i = 0; i < 4; ++i) e.c[i] = p.c[i] + (e.r - p.r) / dd * (q.c[i] - p.c[i]);
                    return e;
                };
            else if (t == "obj.intersect")
                bf = [fa, fb, k, kLn2](double d) {
                    Bound p = fa(d + k * kLn2), q = fb(d + k * kLn2);
                    return p.r <= q.r ? p : q;
                };
            else bf = [fa, k, kLn2](double d) { return fa(d + k * kLn2); };
            return {g, bf};
        }
        if (t == "xf.translate" || t == "xf.rotate" || t == "xf.scale") {
            Shape s = shape_of(source(id, "shape"));
            std::array<ExprPtr, 4> c;
            for (int i = 0; i < 4; ++i) c[i] = mk_coord(i);
            if (t == "xf.translate") {
                Vec4 tv{v.at("t.x"), v.at("t.y"), v.at("t.z"), v.at("t.w")};
                for (int i = 0; i < 4; ++i) c[i] = mk_op(Expr::Sub, mk_coord(i), num(tv[i]));
                auto f = s.bound;
                return {substitute_coords(s.g, c), [f, tv](double d) {
                            Bound b = f(d);
                            for (int i = 0; i < 4; ++i) b.c[i] += tv[i];
                            return b;
                        }};
            }
            Vec4 ctr{v.at("center.x"), v.at("center.y"), v.at("center.z"), v.at("center.w")};
            if (t == "xf.rotate") {
                const int i = int(v.at("i")), j = int(v.at("j"));
                if (i < 0 || i > 3 || j < 0 || j > 3 || i == j) fail(id, "оси поворота i, j — разные из 0…3");
                const double a = v.at("angle") * (3.14159265358979323846 / 180.0), ca = std::cos(a), sa = std::sin(a);
                // аргумент дочерней формы: R⁻¹(x − center) + center (R — поворот на +angle в плоскости i, j)
                auto q = [&](int k) { return mk_op(Expr::Sub, mk_coord(k), num(ctr[k])); };
                c[i] = mk_op(Expr::Add, mk_op(Expr::Add, mk_op(Expr::Mul, num(ca), q(i)), mk_op(Expr::Mul, num(sa), q(j))), num(ctr[i]));
                c[j] = mk_op(Expr::Add, mk_op(Expr::Sub, mk_op(Expr::Mul, num(ca), q(j)), mk_op(Expr::Mul, num(sa), q(i))), num(ctr[j]));
                auto f = s.bound;
                return {substitute_coords(s.g, c), [f, i, j, ca, sa, ctr](double d) {
                            Bound b = f(d);
                            if (std::isinf(b.r)) return b;
                            const double ui = b.c[i] - ctr[i], uj = b.c[j] - ctr[j];
                            b.c[i] = ctr[i] + ca * ui - sa * uj;
                            b.c[j] = ctr[j] + sa * ui + ca * uj;
                            return b;
                        }};
            }
            const double sc_ = v.at("s");
            if (!(sc_ > 0)) fail(id, "нужно s > 0");
            for (int k = 0; k < 4; ++k)
                c[k] = mk_op(Expr::Add, mk_op(Expr::Div, mk_op(Expr::Sub, mk_coord(k), num(ctr[k])), num(sc_)), num(ctr[k]));
            auto f = s.bound;
            return {mk_op(Expr::Mul, num(sc_), substitute_coords(s.g, c)), [f, sc_, ctr](double d) {
                        Bound b = f(d / sc_);
                        if (std::isinf(b.r)) return b;
                        for (int k = 0; k < 4; ++k) b.c[k] = ctr[k] + sc_ * (b.c[k] - ctr[k]);
                        b.r *= sc_;
                        return b;
                    }};
        }
        if (t == "xf.on_M") fail(id, "«Поставить на M» — только последним перед «Объектом»");
        fail(id, "форма типа «" + t + "» не поддерживается");
    };
    // объекты — в порядке связей входа scene.objects
    auto it = in.find({out_scene, "objects"});
    for (const std::string& oid : it == in.end() ? std::vector<std::string>{} : it->second) {
        const N& on = nodes[oid];
        Object ob;
        auto names = on.params->value("name", std::string());
        ob.name = names.empty() ? std::string("объект") : names;
        std::string sid = source(oid, "shape");
        while (nodes[sid].def->type == "xf.on_M") {
            ob.on_M = true;
            sid = source(sid, "shape");
        }
        const N& sn = nodes[sid];
        if (sn.def->legacy_kind.empty()) {        // составная форма (M36): выражение целиком, сфера — по графу
            Shape s = shape_of(sid);
            Bound b = s.bound(0.0);
            ob.kind = "graph";
            ob.node = sn.def->type;
            ob.expr = s.g;
            ob.r = b.r;
            ob.center = b.c;
            if (ob.on_M) {                        // сдвиг всей формы: центр сферы — на M
                if (std::isinf(b.r)) fail(sid, "«Поставить на M» — только для ограниченной формы");
                if (project) {
                    Vec4 pc = geo4d::project(sc.mf, b.c);
                    std::array<ExprPtr, 4> c;
                    for (int i = 0; i < 4; ++i) c[i] = mk_op(Expr::Sub, mk_coord(i), mk_num(pc[i] - b.c[i]));
                    ob.expr = substitute_coords(s.g, c);
                    ob.center = pc;
                }
            }
        } else {
        auto v = node_values(*sn.def, *sn.params);
        ob.kind = sn.def->legacy_kind;
        ob.node = sn.def->type;
        ob.values = v;
        if (ob.kind == "plane") {
            ob.axis = int(v.at("axis"));
            ob.offset = v.at("offset");
            ob.sign = v.at("sign");
        } else {
            Vec4 c{v.at("c.x"), v.at("c.y"), v.at("c.z"), v.at("c.w")};
            ob.center = ob.on_M && project ? geo4d::project(sc.mf, c) : c;
            if (ob.kind == "ball4") ob.r = v.at("r");
            else { ob.R = v.at("R"); ob.r = v.at("r"); }
        }
        }
        const std::string matid = source(oid, "material");
        const N& mat = nodes[matid];
        auto m = node_values(*mat.def, *mat.params);
        ob.mat.kind = mat.def->material;
        ob.mat.albedo = {m.at("albedo.x"), m.at("albedo.y"), m.at("albedo.z")};
        ob.mat.emit = {m.at("emit.x"), m.at("emit.y"), m.at("emit.z")};
        ob.mat.rough = m.at("rough");
        sc.objs.push_back(ob);
    }
    if (sc.objs.size() > 32) fail(out_scene, "объектов больше 32");
    // камера
    J c = doc.value("camera", J::object());
    sc.cam.x = vec<4>(c, "x", Vec4{0, 0, 0, 0});
    if (project) sc.cam.x = geo4d::project(sc.mf, sc.cam.x);
    sc.cam.f = vec<4>(c, "f", Vec4{0, 1, 0, 0});
    sc.cam.up = vec<4>(c, "up", Vec4{0, 0, 1, 0});
    sc.cam.has_right = c.contains("right");
    if (sc.cam.has_right) sc.cam.right = vec<4>(c, "right", Vec4{0, 0, 0, 0});
    sc.cam.fov = c.value("fov", 75.0);
    return sc;
}

int run_migrate(int argc, char** argv) {
    if (argc < 4) throw std::runtime_error("использование: geo4d_gpu migrate <сцена.json> <сцена.g4d>");
    std::ifstream f(argv[2], std::ios::binary);
    if (!f) throw std::runtime_error(std::string("не открыть ") + argv[2]);
    nlohmann::json doc = migrate_v1(nlohmann::json::parse(f));
    scene_from_document(doc, argv[2], false);     // проверка графа
    std::ofstream(argv[3], std::ios::binary) << doc.dump(1) << "\n";
    return 0;
}

}  // namespace geo4d
