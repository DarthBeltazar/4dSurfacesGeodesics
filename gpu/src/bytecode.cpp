// Байткод выражений графа сцены (DD-011 §6) — см. bytecode.hpp.
#include "bytecode.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <set>

#include "codegen.hpp"
#include "scene.hpp"
#include "scene_graph.hpp"

namespace geo4d {

namespace {

// коды операций — geo4d.bytecode.OPS и switch в scenes.slang (prog_eval)
enum Op { CONST = 1, ADD = 2, SUB = 3, MUL = 4, NEG = 6, ADDK = 7, MULK = 8, RECIP = 9, SQRT = 10, EXP = 11, LOG = 12,
          COSH = 13, SIN = 14, COS = 15, SQRTG = 16, SELGT = 17, SQ = 19, SINH = 20 };

struct Ins {
    int op;
    std::vector<int> args;                        // id значений: 0…3 — x, y, z, w; 4 + i — результат инструкции i
    double k;
};

struct Builder {
    const std::map<std::string, double>& params;
    std::vector<Ins> code;
    std::map<const Expr*, int> memo;              // общие узлы дерева (a^n, переиспользованные центры) — один раз

    int emit(int op, std::vector<int> args = {}, double k = 0) {
        code.push_back({op, std::move(args), k});
        if (code.size() > size_t(kProgMaxIns)) throw std::runtime_error("программа длиннее " + std::to_string(kProgMaxIns) + " инструкций");
        return 3 + int(code.size());
    }
    double uval(const ExprPtr& e) { return eval_expr(e, params, {0, 0, 0, 0}).f; }

    int val(const ExprPtr& e) {
        auto it = memo.find(e.get());
        if (it != memo.end()) return it->second;
        int v = build(e);
        memo[e.get()] = v;
        return v;
    }

    int build(const ExprPtr& e) {
        if (!e->field) return emit(CONST, {}, uval(e));
        const auto& A = e->args;
        switch (e->kind) {
            case Expr::Coord: return e->coord;
            case Expr::Neg: return emit(NEG, {val(A[0])});
            case Expr::Add:
                if (!A[0]->field) return emit(ADDK, {val(A[1])}, uval(A[0]));
                if (!A[1]->field) return emit(ADDK, {val(A[0])}, uval(A[1]));
                return emit(ADD, {val(A[0]), val(A[1])});
            case Expr::Sub:
                if (!A[1]->field) return emit(ADDK, {val(A[0])}, -uval(A[1]));
                if (!A[0]->field) return emit(ADDK, {emit(NEG, {val(A[1])})}, uval(A[0]));
                return emit(SUB, {val(A[0]), val(A[1])});
            case Expr::Mul:
                if (!A[0]->field) return emit(MULK, {val(A[1])}, uval(A[0]));
                if (!A[1]->field) return emit(MULK, {val(A[0])}, uval(A[1]));
                if (A[0].get() == A[1].get()) return emit(SQ, {val(A[0])});
                return emit(MUL, {val(A[0]), val(A[1])});
            case Expr::Div:
                if (!A[1]->field) return emit(MULK, {val(A[0])}, 1.0 / uval(A[1]));
                if (!A[0]->field) return emit(MULK, {emit(RECIP, {val(A[1])})}, uval(A[0]));
                return emit(MUL, {val(A[0]), emit(RECIP, {val(A[1])})});
            case Expr::Call: return call(e);
            default: break;
        }
        throw ExprError("узел выражения не переводится в байткод", e->pos);
    }

    int call(const ExprPtr& e) {
        const std::string& n = e->text;
        const auto& A = e->args;
        static const std::map<std::string, int> unary = {{"sqrt", SQRT}, {"exp", EXP}, {"log", LOG}, {"sin", SIN},
                                                         {"cos", COS},   {"cosh", COSH}, {"sinh", SINH}};
        if (auto u = unary.find(n); u != unary.end()) return emit(u->second, {val(A[0])});
        if (n == "length") {                      // M6: √(max(Σ uᵢ², ε²)), как jlength*
            int s = emit(SQ, {val(A[0])});
            for (size_t i = 1; i < A.size(); ++i) s = emit(ADD, {s, emit(SQ, {val(A[i])})});
            return emit(SQRTG, {s});
        }
        if (n == "smin") {                        // M6, как jsmin: lo − k·log(1 + exp((lo − hi)/k))
            const double k = uval(A[2]);
            int a = val(A[0]), b = val(A[1]);
            int d = emit(SUB, {a, b});
            int lo = emit(SELGT, {b, a, d}, 0.0), hi = emit(SELGT, {a, b, d}, 0.0);
            int t = emit(EXP, {emit(MULK, {emit(SUB, {lo, hi})}, 1.0 / k)});
            return emit(SUB, {lo, emit(MULK, {emit(LOG, {emit(ADDK, {t}, 1.0)})}, k)});
        }
        if (n == "compact_h") {                   // M11, как compact_h в scenes.slang (geo4d.bytecode.compact_h)
            const double b = uval(A[1]), rho0 = uval(A[2]), W = uval(A[3]);
            int one = emit(CONST, {}, 1.0);
            int q = emit(ADDK, {emit(MULK, {val(A[0])}, -1.0 / (rho0 - b))}, 1.0 + b / (rho0 - b));
            int qs = emit(SELGT, {q, one, q}, 1e-2);
            int ex = emit(EXP, {emit(ADDK, {emit(NEG, {emit(RECIP, {qs})})}, 1.0)});
            int psi = emit(SELGT, {emit(ADDK, {emit(NEG, {ex})}, 1.0), one, q}, 1e-2);
            return emit(MULK, {psi}, W * W);
        }
        throw ExprError("функция «" + n + "» не переводится в байткод", e->pos);
    }
};

}  // namespace

Program compile_program(const ExprPtr& e, const std::map<std::string, double>& params) {
    Builder B{params};
    const int result = B.val(e);
    const auto& code = B.code;
    // распределение регистров по времени жизни (как geo4d.bytecode.Prog.assemble)
    std::map<int, size_t> last;
    for (size_t i = 0; i < code.size(); ++i)
        for (int a : code[i].args) last[a] = i;
    std::map<int, int> reg;
    std::vector<int> free;
    int nreg = 4;
    for (int v = 0; v < 4; ++v) {
        reg[v] = v;
        if (!last.count(v) && v != result) free.push_back(v);
    }
    Program p;
    auto push = [&](int op, int dst, int a, int b, int c, double k) {
        p.rows.push_back({float(op + 256 * dst + 65536 * a), float(b + 256 * c), float(k), 0.0f});
        p.rows64.push_back({double(op + 256 * dst + 65536 * a), double(b + 256 * c), k, 0.0});
    };
    p.rows.push_back({0, 0, 0, 0});
    p.rows64.push_back({0, 0, 0, 0});
    for (size_t i = 0; i < code.size(); ++i) {
        const Ins& in = code[i];
        int src[3] = {0, 0, 0};
        for (size_t j = 0; j < in.args.size(); ++j) src[j] = reg.at(in.args[j]);
        std::set<int> uniq(in.args.begin(), in.args.end());
        for (int a : uniq)                        // регистры аргументов освобождаются до выбора регистра результата
            if (last[a] == i && a != result) free.push_back(reg[a]);
        const int v = 4 + int(i);
        if (!free.empty()) {
            std::sort(free.begin(), free.end());
            reg[v] = free.front();
            free.erase(free.begin());
        } else reg[v] = nreg++;
        push(in.op, reg[v], src[0], src[1], src[2], in.k);
        if (!last.count(v) && v != result) free.push_back(reg[v]);
    }
    if (reg.at(result) != 0) push(MULK, 0, reg[result], 0, 0, 1.0);   // результат — в регистр 0
    if (nreg > kProgMaxReg) throw std::runtime_error("программе нужно " + std::to_string(nreg) + " регистров (предел " +
                                                     std::to_string(kProgMaxReg) + ") — только нативный путь");
    p.nins = int(p.rows.size()) - 1;
    p.nreg = nreg;
    p.rows[0] = {float(p.nins), float(nreg), 0, 0};
    p.rows64[0] = {double(p.nins), double(nreg), 0, 0};
    return p;
}

namespace {

// выражение многообразия и его параметры: собранное по графу или формула ноды из библиотеки (как eval-graph)
std::pair<ExprPtr, std::map<std::string, double>> manifold_expr(const Scene& sc) {
    if (sc.mf.expr) return {sc.mf.expr, sc.mf.expr_vals};
    const NodeDef* def = node_library().find(sc.mf_node);
    if (!def || def->formula.empty()) throw std::runtime_error("нет формулы многообразия «" + sc.mf_node + "»");
    std::vector<std::string> names;
    for (auto& [k, v] : sc.mf_values) names.push_back(k);
    return {parse_expr(def->formula, names), sc.mf_values};
}

constexpr int kDir = 1 + kProgMaxBall;           // строки начала программ объектов
constexpr int kFirst = kDir + 32;                 // программа многообразия

}  // namespace

std::vector<float> program_rows(const Scene& sc, int* max_reg) {
    std::vector<std::array<float, 4>> R(kFirst, {0, 0, 0, 0});
    const auto& balls = sc.mf.flat.centers;
    if (balls.size() > size_t(kProgMaxBall)) throw std::runtime_error("шаров кривизны больше " + std::to_string(kProgMaxBall));
    R[0][0] = float(balls.size());
    for (size_t i = 0; i < balls.size(); ++i) R[1 + i] = {float(balls[i][0]), float(balls[i][1]), float(balls[i][2]), 0};
    auto [e, vals] = manifold_expr(sc);
    Program mf = compile_program(e, vals);
    int mr = mf.nreg;
    R.insert(R.end(), mf.rows.begin(), mf.rows.end());
    const auto slots = sc.slots.values();
    for (size_t k = 0; k < sc.objs.size() && k < 32; ++k) {
        if (sc.objs[k].kind != "graph") continue;
        Program p = compile_program(sc.objs[k].expr, slots);
        mr = std::max(mr, p.nreg);
        R[kDir + k][0] = float(kProgBase + R.size());
        R.insert(R.end(), p.rows.begin(), p.rows.end());
    }
    if (max_reg) *max_reg = mr;
    if (R.size() > size_t(kProgMaxRows)) throw std::runtime_error("программы длиннее " + std::to_string(kProgMaxRows) + " строк");
    std::vector<float> out;
    for (auto& r : R) out.insert(out.end(), r.begin(), r.end());
    return out;
}

int run_bytecode(int argc, char** argv) {
    if (argc < 4) throw std::runtime_error("использование: geo4d_gpu bytecode <сцена> <выход.json>");
    Scene sc = load_scene(argv[2], false);
    auto rows = [](const Program& p) {            // {f32: строки для GPU, f64: константы в double}
        nlohmann::json a = nlohmann::json::array(), b = nlohmann::json::array();
        for (auto& r : p.rows) a.push_back({r[0], r[1], r[2], r[3]});
        for (auto& r : p.rows64) b.push_back({r[0], r[1], r[2], r[3]});
        return nlohmann::json{{"f32", a}, {"f64", b}};
    };
    nlohmann::json out;
    auto [e, vals] = manifold_expr(sc);
    out["manifold"] = rows(compile_program(e, vals));
    nlohmann::json objs = nlohmann::json::array();
    const auto slots = sc.slots.values();
    for (auto& o : sc.objs) objs.push_back(o.kind == "graph" ? rows(compile_program(o.expr, slots)) : nlohmann::json());
    out["objects"] = objs;
    const auto R = program_rows(sc);
    out["region_rows"] = R.size() / 4;
    std::ofstream(argv[3], std::ios::binary) << out.dump();
    return 0;
}

}  // namespace geo4d
