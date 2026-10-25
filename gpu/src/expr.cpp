// Язык выражений нод (DD-011 §2) — см. expr.hpp.
#include "expr.hpp"

#include <nlohmann/json.hpp>

#include <cctype>
#include <cmath>
#include <fstream>
#include <functional>

namespace geo4d {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kLengthEps2 = 1e-36;            // jet.slang LENGTH_EPS2 (M6, float32)

// функции языка: имя → (число аргументов, мин–макс; 0 — переменное), какие аргументы обязаны быть однородными
struct Fn {
    int nmin, nmax;
    std::vector<int> uniform_args;                // индексы аргументов, которые должны быть однородными
    bool field_only;                              // только для полей (jsmin, compact_h — над IReal)
};
const std::map<std::string, Fn>& functions() {
    static const std::map<std::string, Fn> f = {
        {"sqrt", {1, 1, {}, false}}, {"exp", {1, 1, {}, false}}, {"log", {1, 1, {}, false}},
        {"sin", {1, 1, {}, false}},  {"cos", {1, 1, {}, false}}, {"cosh", {1, 1, {}, false}},
        {"sinh", {1, 1, {}, false}}, {"length", {2, 4, {}, false}},
        {"smin", {3, 3, {2}, true}}, {"compact_h", {4, 4, {1, 2, 3}, true}}};
    return f;
}
// негладкие — только в текстурах (DD-011 §2)
const char* kNonSmooth[] = {"abs", "min", "max", "floor", "ceil", "fract", "mod", "step", "smoothstep", "mix", "clamp",
                            "sign", "round"};

struct Parser {
    const std::string& s;
    const std::vector<std::string>& params;
    Context ctx;
    size_t i = 0;

    [[noreturn]] void fail(const std::string& m, size_t at) const { throw ExprError(m, at); }
    void ws() { while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i; }
    bool peek(char c) { ws(); return i < s.size() && s[i] == c; }
    bool eat(char c) {
        if (!peek(c)) return false;
        ++i;
        return true;
    }

    ExprPtr node(Expr::Kind k, size_t pos, std::vector<ExprPtr> args = {}) {
        auto e = std::make_shared<Expr>();
        e->kind = k;
        e->pos = pos;
        e->args = std::move(args);
        for (auto& a : e->args) e->field = e->field || a->field;
        return e;
    }

    ExprPtr parse() {
        ExprPtr e = additive();
        ws();
        if (i != s.size()) fail("лишний текст в формуле", i);
        return e;
    }
    ExprPtr additive() {
        ExprPtr a = multiplicative();
        for (;;) {
            ws();
            size_t p = i;
            if (eat('+')) a = node(Expr::Add, p, {a, multiplicative()});
            else if (eat('-')) a = node(Expr::Sub, p, {a, multiplicative()});
            else return a;
        }
    }
    ExprPtr multiplicative() {
        ExprPtr a = unary();
        for (;;) {
            ws();
            size_t p = i;
            if (eat('*')) a = node(Expr::Mul, p, {a, unary()});
            else if (eat('/')) a = node(Expr::Div, p, {a, unary()});
            else return a;
        }
    }
    ExprPtr unary() {
        ws();
        size_t p = i;
        if (eat('-')) return node(Expr::Neg, p, {unary()});
        return power();
    }
    ExprPtr power() {
        ExprPtr a = primary();
        ws();
        size_t p = i;
        if (!eat('^')) return a;
        ws();
        size_t q = i;
        while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) ++i;
        if (q == i) fail("после ^ — целое неотрицательное число (степень)", q);
        int n = std::stoi(s.substr(q, i - q));
        if (n == 0) {
            auto one = node(Expr::Num, p);
            one->value = 1;
            one->text = "1.0";
            return one;
        }
        ExprPtr r = a;                            // a^n = ((a·a)·a)… — как запись a * a * a
        for (int k = 1; k < n; ++k) r = node(Expr::Mul, p, {r, a});
        return r;
    }
    ExprPtr primary() {
        ws();
        size_t p = i;
        if (i >= s.size()) fail("формула оборвана", i);
        char c = s[i];
        if (std::isdigit(static_cast<unsigned char>(c)) || c == '.') {
            size_t q = i;
            while (i < s.size() && (std::isdigit(static_cast<unsigned char>(s[i])) || s[i] == '.')) ++i;
            if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
                ++i;
                if (i < s.size() && (s[i] == '+' || s[i] == '-')) ++i;
                while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) ++i;
            }
            auto e = node(Expr::Num, p);
            e->text = s.substr(q, i - q);
            try {
                e->value = std::stod(e->text);
            } catch (...) {
                fail("неверное число «" + e->text + "»", q);
            }
            return e;
        }
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            size_t q = i;
            while (i < s.size() && (std::isalnum(static_cast<unsigned char>(s[i])) || s[i] == '_')) ++i;
            std::string name = s.substr(q, i - q);
            // компонента векторного параметра: c1.x
            if (i + 1 < s.size() && s[i] == '.' && std::string("xyzw").find(s[i + 1]) != std::string::npos &&
                (i + 2 >= s.size() || !std::isalnum(static_cast<unsigned char>(s[i + 2])))) {
                name += s.substr(i, 2);
                i += 2;
            }
            ws();
            if (i < s.size() && s[i] == '(') return call(name, p);
            if (name.size() == 1 && std::string("xyzw").find(name[0]) != std::string::npos) {
                auto e = node(Expr::Coord, p);
                e->coord = int(std::string("xyzw").find(name[0]));
                e->text = name;
                e->field = true;
                return e;
            }
            if (name == "pi") {
                auto e = node(Expr::Num, p);
                e->value = kPi;
                e->text = "3.14159265358979323846";
                return e;
            }
            for (auto& pn : params)
                if (pn == name) {
                    auto e = node(Expr::Param, p);
                    e->text = name;
                    return e;
                }
            fail("неизвестное имя «" + name + "»", q);
        }
        if (eat('(')) {
            ExprPtr e = additive();
            if (!eat(')')) fail("нет закрывающей скобки", i);
            return e;
        }
        fail(std::string("неожиданный символ «") + c + "»", i);
    }
    ExprPtr call(const std::string& name, size_t p) {
        for (const char* ns : kNonSmooth)
            if (name == ns) {
                if (ctx == Context::Geometry)
                    fail("«" + name + "» негладкая — в геометрии запрещена (M6); используйте smin, length", p);
                fail("функции текстур ещё не реализованы (DD-011, шаг 7)", p);
            }
        auto it = functions().find(name);
        if (it == functions().end()) fail("неизвестная функция «" + name + "»", p);
        eat('(');
        std::vector<ExprPtr> args;
        if (!peek(')'))
            for (;;) {
                args.push_back(additive());
                if (eat(',')) continue;
                break;
            }
        if (!eat(')')) fail("нет закрывающей скобки вызова «" + name + "»", i);
        const Fn& fn = it->second;
        if (int(args.size()) < fn.nmin || int(args.size()) > fn.nmax)
            fail("«" + name + "»: неверное число аргументов (" + std::to_string(args.size()) + ")", p);
        for (int k : fn.uniform_args)
            if (args[k]->field) fail("«" + name + "»: аргумент " + std::to_string(k + 1) + " не должен зависеть от x, y, z, w", args[k]->pos);
        auto e = node(Expr::Call, p, args);
        e->text = name;
        if (fn.field_only && !e->field) fail("«" + name + "» — только для полей (зависящих от x, y, z, w)", p);
        return e;
    }
};

// ---- вывод в Slang ---------------------------------------------------------------------------------------

int prec(const ExprPtr& e) {
    switch (e->kind) {
        case Expr::Add: case Expr::Sub: return 1;
        case Expr::Mul: case Expr::Div: return 2;
        case Expr::Neg: return 3;
        default: return 4;
    }
}

std::string num_text(const std::string& t) {
    // вещественный литерал Slang: «2» → «2.0» (иначе int и выбор перегрузки операторов IReal)
    return t.find_first_of(".eE") == std::string::npos ? t + ".0" : t;
}

struct Emitter {
    const std::map<std::string, std::string>& params;
    const std::array<std::string, 4>& coords;

    // правый операнд того же приоритета — в скобках всегда: a * (b / c) ≠ (a * b) / c во float и в IReal
    std::string sub(const ExprPtr& e, int parent, bool right_of_noncomm) {
        std::string t = emit(e);
        int p = prec(e);
        if (p < parent || (p == parent && right_of_noncomm && p < 3)) return "(" + t + ")";
        return t;
    }
    std::string emit(const ExprPtr& e) {
        switch (e->kind) {
            case Expr::Num: return num_text(e->text);
            case Expr::Param: {
                auto it = params.find(e->text);
                if (it == params.end()) throw ExprError("нет значения параметра «" + e->text + "» для Slang", e->pos);
                return it->second;
            }
            case Expr::Coord: return coords[e->coord];
            case Expr::Neg:
                // унарный минус над полем — a.neg() (в обобщённом коде Slang `-a` неоднозначен; проба интерпретатора)
                if (e->args[0]->field) return sub(e->args[0], 4, false) + ".neg()";
                return "-" + sub(e->args[0], 3, false);
            case Expr::Add: return sub(e->args[0], 1, false) + " + " + sub(e->args[1], 1, true);
            case Expr::Sub: return sub(e->args[0], 1, false) + " - " + sub(e->args[1], 1, true);
            case Expr::Mul: return sub(e->args[0], 2, false) + " * " + sub(e->args[1], 2, true);
            case Expr::Div: return sub(e->args[0], 2, false) + " / " + sub(e->args[1], 2, true);
            case Expr::Call: {
                std::vector<std::string> a;
                for (auto& x : e->args) a.push_back(emit(x));
                auto join = [&]() {
                    std::string s;
                    for (size_t k = 0; k < a.size(); ++k) s += (k ? ", " : "") + a[k];
                    return s;
                };
                const std::string& n = e->text;
                if (!e->field) {                  // однородное — float-функции Slang
                    if (n == "length") return "length(float" + std::to_string(a.size()) + "(" + join() + "))";
                    return n + "(" + join() + ")";
                }
                if (n == "length") return "jlength" + std::to_string(a.size()) + "(" + join() + ")";
                if (n == "smin") return "jsmin(" + join() + ")";
                if (n == "compact_h") return "compact_h(" + join() + ")";
                // одноаргументные функции поля: аргумент — поле (иначе вызов однородный)
                return "j" + n + "(" + join() + ")";
            }
        }
        return "";
    }
};

// ---- вычисление на CPU: jet второго порядка в double (как Jet15 в jet.slang) ------------------------------------

struct J2 {
    double a = 0;
    std::array<double, 4> g{};
    std::array<std::array<double, 4>, 4> H{};
    static J2 cst(double c) { J2 r; r.a = c; return r; }
    J2 add(const J2& o) const {
        J2 r; r.a = a + o.a;
        for (int i = 0; i < 4; ++i) { r.g[i] = g[i] + o.g[i]; for (int j = 0; j < 4; ++j) r.H[i][j] = H[i][j] + o.H[i][j]; }
        return r;
    }
    J2 mul(const J2& o) const {
        J2 r; r.a = a * o.a;
        for (int i = 0; i < 4; ++i) {
            r.g[i] = a * o.g[i] + o.a * g[i];
            for (int j = 0; j < 4; ++j) r.H[i][j] = a * o.H[i][j] + o.a * H[i][j] + g[i] * o.g[j] + g[j] * o.g[i];
        }
        return r;
    }
    J2 neg() const { return scale(-1.0); }
    J2 scale(double k) const {
        J2 r; r.a = a * k;
        for (int i = 0; i < 4; ++i) { r.g[i] = g[i] * k; for (int j = 0; j < 4; ++j) r.H[i][j] = H[i][j] * k; }
        return r;
    }
    J2 unary(double f0, double f1, double f2) const {
        J2 r; r.a = f0;
        for (int i = 0; i < 4; ++i) { r.g[i] = f1 * g[i]; for (int j = 0; j < 4; ++j) r.H[i][j] = f1 * H[i][j] + f2 * g[i] * g[j]; }
        return r;
    }
};

J2 jrecip(const J2& u) { double a = u.a; return u.unary(1.0 / a, -1.0 / (a * a), 2.0 / (a * a * a)); }
J2 jsqrt(const J2& u) { double a = u.a, s = std::sqrt(a); return u.unary(s, 0.5 / s, -0.25 / (a * s)); }
J2 jexp(const J2& u) { double e = std::exp(u.a); return u.unary(e, e, e); }
J2 jlog(const J2& u) { double a = u.a; return u.unary(std::log(a), 1.0 / a, -1.0 / (a * a)); }

struct Evaluator {
    const std::map<std::string, double>& params;
    const std::array<double, 4>& p;

    // значение: однородное — double, поле — J2 (перегрузки jet.slang: T·float — scale, T/float — scale(1/k), …)
    struct V {
        bool field = false;
        double u = 0;
        J2 j;
    };
    static J2 as_jet(const V& v) { return v.field ? v.j : J2::cst(v.u); }

    V ev(const ExprPtr& e) {
        V r;
        r.field = e->field;
        switch (e->kind) {
            case Expr::Num: r.u = e->value; return r;
            case Expr::Param: {
                auto it = params.find(e->text);
                if (it == params.end()) throw ExprError("нет значения параметра «" + e->text + "»", e->pos);
                r.u = it->second;
                return r;
            }
            case Expr::Coord:
                r.j = J2::cst(p[e->coord]);
                r.j.g[e->coord] = 1;
                return r;
            case Expr::Neg: { V a = ev(e->args[0]); if (a.field) r.j = a.j.neg(); else r.u = -a.u; return r; }
            case Expr::Add: case Expr::Sub: case Expr::Mul: case Expr::Div: {
                V a = ev(e->args[0]), b = ev(e->args[1]);
                if (!r.field) {
                    r.u = e->kind == Expr::Add ? a.u + b.u : e->kind == Expr::Sub ? a.u - b.u : e->kind == Expr::Mul ? a.u * b.u : a.u / b.u;
                    return r;
                }
                if (e->kind == Expr::Add) r.j = a.field && b.field ? a.j.add(b.j) : a.field ? a.j.add(J2::cst(b.u)) : b.j.add(J2::cst(a.u));
                else if (e->kind == Expr::Sub)
                    r.j = a.field && b.field ? a.j.add(b.j.neg()) : a.field ? a.j.add(J2::cst(-b.u)) : b.j.neg().add(J2::cst(a.u));
                else if (e->kind == Expr::Mul) r.j = a.field && b.field ? a.j.mul(b.j) : a.field ? a.j.scale(b.u) : b.j.scale(a.u);
                else r.j = a.field && b.field ? a.j.mul(jrecip(b.j)) : a.field ? a.j.scale(1.0 / b.u) : jrecip(b.j).scale(a.u);
                return r;
            }
            case Expr::Call: return call(e);
        }
        return r;
    }

    V call(const ExprPtr& e) {
        std::vector<V> a;
        for (auto& x : e->args) a.push_back(ev(x));
        V r;
        r.field = e->field;
        const std::string& n = e->text;
        if (!e->field) {
            if (n == "length") { double q = 0; for (auto& v : a) q += v.u * v.u; r.u = std::sqrt(q); }
            else if (n == "sqrt") r.u = std::sqrt(a[0].u);
            else if (n == "exp") r.u = std::exp(a[0].u);
            else if (n == "log") r.u = std::log(a[0].u);
            else if (n == "sin") r.u = std::sin(a[0].u);
            else if (n == "cos") r.u = std::cos(a[0].u);
            else if (n == "cosh") r.u = std::cosh(a[0].u);
            else if (n == "sinh") r.u = std::sinh(a[0].u);
            return r;
        }
        if (n == "length") {                      // M6: √(max(|u|², ε²)), сумма слева направо
            J2 q = as_jet(a[0]).mul(as_jet(a[0]));
            for (size_t k = 1; k < a.size(); ++k) q = q.add(as_jet(a[k]).mul(as_jet(a[k])));
            r.j = jsqrt(q.a < kLengthEps2 ? J2::cst(kLengthEps2) : q);
            return r;
        }
        if (n == "smin") {                        // M6, как jsmin: lo − k·log(1 + exp((lo − hi)/k))
            J2 x = as_jet(a[0]), y = as_jet(a[1]);
            double k = a[2].u;
            bool lt = x.a < y.a;
            J2 lo = lt ? x : y, hi = lt ? y : x;
            J2 t = jexp(lo.add(hi.neg()).scale(1.0 / k));
            r.j = lo.add(jlog(t.add(J2::cst(1.0))).scale(k).neg());
            return r;
        }
        if (n == "compact_h") {                   // M11, как compact_h в scenes.slang
            J2 rho = as_jet(a[0]);
            double b = a[1].u, rho0 = a[2].u, W = a[3].u;
            J2 q = rho.add(J2::cst(-b)).scale(1.0 / (rho0 - b)).neg().add(J2::cst(1.0));
            bool inside = q.a > 1e-2;
            J2 qs = inside ? q : J2::cst(1.0);
            J2 psi = inside ? jexp(jrecip(qs).scale(1.0).neg().add(J2::cst(1.0))).neg().add(J2::cst(1.0)) : J2::cst(1.0);
            r.j = psi.scale(W * W);
            return r;
        }
        J2 u = as_jet(a[0]);
        double x = u.a;
        if (n == "sqrt") r.j = jsqrt(u);
        else if (n == "exp") r.j = jexp(u);
        else if (n == "log") r.j = jlog(u);
        else if (n == "sin") r.j = u.unary(std::sin(x), std::cos(x), -std::sin(x));
        else if (n == "cos") r.j = u.unary(std::cos(x), -std::sin(x), -std::cos(x));
        else if (n == "cosh") r.j = u.unary(std::cosh(x), std::sinh(x), std::cosh(x));
        else if (n == "sinh") r.j = u.unary(std::sinh(x), std::cosh(x), std::sinh(x));
        else throw ExprError("функция «" + n + "» не вычисляется на CPU", e->pos);
        return r;
    }
};

}  // namespace

ExprPtr parse_expr(const std::string& src, const std::vector<std::string>& params, Context ctx) {
    Parser p{src, params, ctx};
    return p.parse();
}

std::string emit_slang(const ExprPtr& e, const std::map<std::string, std::string>& params, const std::array<std::string, 4>& coords) {
    Emitter em{params, coords};
    return em.emit(e);
}

EvalResult eval_expr(const ExprPtr& e, const std::map<std::string, double>& params, const std::array<double, 4>& p) {
    Evaluator ev{params, p};
    auto v = ev.ev(e);
    J2 j = v.field ? v.j : J2::cst(v.u);
    EvalResult r;
    r.f = j.a;
    r.g = j.g;
    r.H = j.H;
    return r;
}

int run_eval(int argc, char** argv) {
    if (argc < 4) throw std::runtime_error("использование: geo4d_gpu eval <вход.json> <выход.json>");
    std::ifstream fi(argv[2], std::ios::binary);
    if (!fi) throw std::runtime_error(std::string("не открыть ") + argv[2]);
    nlohmann::json in = nlohmann::json::parse(fi), out = nlohmann::json::array();
    for (auto& c : in.at("cases")) {
        nlohmann::json r;
        std::vector<std::string> names;
        std::map<std::string, double> vals;
        std::map<std::string, std::string> sl;
        // value() возвращает копию — держим её, пока идём по items() (иначе висячая ссылка)
        const nlohmann::json pj = c.value("params", nlohmann::json::object()), sj = c.value("slang_params", nlohmann::json::object());
        for (auto& [k, v] : pj.items()) {
            names.push_back(k);
            vals[k] = v.get<double>();
        }
        for (auto& [k, v] : sj.items()) sl[k] = v.get<std::string>();
        Context ctx = c.value("context", std::string("geometry")) == "texture" ? Context::Texture : Context::Geometry;
        try {
            ExprPtr e = parse_expr(c.at("formula").get<std::string>(), names, ctx);
            r["ok"] = true;
            r["field"] = e->field;
            if (!sl.empty()) r["slang"] = emit_slang(e, sl);
            nlohmann::json f = nlohmann::json::array(), g = nlohmann::json::array(), H = nlohmann::json::array();
            const nlohmann::json pts = c.value("points", nlohmann::json::array());
            for (auto& pt : pts) {
                std::array<double, 4> p{pt[0].get<double>(), pt[1].get<double>(), pt[2].get<double>(), pt[3].get<double>()};
                EvalResult er = eval_expr(e, vals, p);
                f.push_back(er.f);
                g.push_back(er.g);
                H.push_back(er.H);
            }
            r["f"] = f;
            r["g"] = g;
            r["H"] = H;
        } catch (const ExprError& ex) {
            r["ok"] = false;
            r["error"] = ex.what();
            r["pos"] = ex.pos;
        }
        out.push_back(r);
    }
    std::ofstream(argv[3], std::ios::binary) << out.dump();
    return 0;
}

}  // namespace geo4d
