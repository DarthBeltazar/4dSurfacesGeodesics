// Кодген окна (DD-010) — см. codegen.hpp.
#include "codegen.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "expr.hpp"
#include "scene_graph.hpp"
#include "bytecode.hpp"
#include "sha256.hpp"

#ifdef _WIN32
#include <windows.h>
#endif

namespace geo4d {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kSigmaPix = 0.5;                 // gpu_modea.SIGMA_PIX (M18)
constexpr double kDetLo = 0.1, kDetHi = 10.0, kDetTau = 2.0;   // modea.Detector()
constexpr int kISeed = 1, kIUseDen = 20, kIBgra = 21, kIExposure = 22;   // gpu_pt.I_*
const char* kPtBind = "cam Acc Acc2 Env Cube Stats Pix Alb Nrm Dir Dd Status Class Flag Cnt Args Frame AlbF NrmF Den Present Color";

std::string read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("не открыть " + u8(p));
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// sorted(dir.rglob("*.slang")): пути Windows сравниваются без учёта регистра
std::vector<fs::path> slang_files(const fs::path& dir) {
    std::vector<fs::path> out;
    if (!fs::exists(dir)) return out;
    for (auto& e : fs::recursive_directory_iterator(dir))
        if (e.is_regular_file() && e.path().extension() == ".slang") out.push_back(e.path());
    auto key = [](const fs::path& p) {
        std::string s = u8(p);
#ifdef _WIN32
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return char(std::tolower(c)); });
        std::replace(s.begin(), s.end(), '\\', '/');
#endif
        return s;
    };
    std::sort(out.begin(), out.end(), [&](const fs::path& a, const fs::path& b) { return key(a) < key(b); });
    return out;
}

std::string arr(const std::string& typ, const std::string& name, std::vector<std::string> rows, size_t n) {
    if (rows.empty()) rows = {"0"};
    std::string s = "static const " + typ + " " + name + "[" + std::to_string(std::max<size_t>(n, 1)) + "] = { ";
    for (size_t i = 0; i < rows.size(); ++i) s += (i ? ", " : "") + rows[i];
    return s + " };";
}

std::string join(const std::vector<std::string>& v, const std::string& sep) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) s += (i ? sep : "") + v[i];
    return s;
}

int material_kind(const std::string& k) {
    if (k == "diffuse") return 0;
    if (k == "metal") return 1;
    if (k == "emit") return 2;
    throw std::runtime_error("нет материала " + k);
}

// _push: (тип, значение) по порядку PTParams (pathtrace.slang)
using PushItems = std::vector<std::pair<char, double>>;   // 'f' — f32, 'u' — u32
PushItems push_items(const Scene& sc, const PTParams& p, long long n, int env_n, int use_list = 0, double c_mix = 1.0,
                     int list_id = 0, long long list_off = 0, int use_den = 0, int bgra = 1, double exposure = 1.0) {
    double s_max = std::min(p.s_max, sc.mf.has_s_max ? sc.mf.s_max : p.s_max);
    double ell_max = sc.mf.has_ell_max ? sc.mf.ell_max : 0.0;
    return {{'u', double(n)}, {'u', double(p.seed)}, {'f', p.tol_kappa}, {'f', p.tol_max}, {'f', p.c_hit},
            {'f', p.k_cone}, {'u', double(p.max_depth)}, {'u', double(p.rr_depth)}, {'u', double(p.nee)},
            {'u', double(p.mis)}, {'f', s_max}, {'u', double(p.n_max)}, {'f', ell_max}, {'u', double(env_n)},
            {'f', p.env_scale * 1.0}, {'f', p.k_lod}, {'u', double(use_list)}, {'f', c_mix}, {'u', double(list_id)},
            {'u', double(list_off)}, {'u', double(use_den)}, {'u', double(bgra)}, {'f', exposure}};
}

std::string push_line(const std::string& pipe, const PushItems& items) {
    std::string s = "push " + pipe;
    for (auto& [t, v] : items) {
        if (t == 'f') s += " f32 " + fmt9g(v);
        else {
            char b[32];
            std::snprintf(b, sizeof(b), " u32 %lld", (long long)v);
            s += b;
        }
    }
    return s;
}

std::vector<float> f32(const std::vector<double>& v) {
    std::vector<float> r(v.size());
    for (size_t i = 0; i < v.size(); ++i) r[i] = static_cast<float>(v[i]);
    return r;
}

}  // namespace

// ---- форматы ---------------------------------------------------------------------------------------------

std::string fmt9g(double v) {
    char b[64];
    std::snprintf(b, sizeof(b), "%.9g", v);
    return b;
}

std::string pyrepr(double v) {
    if (std::isnan(v)) return "nan";
    if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
    if (v == 0) return std::signbit(v) ? "-0.0" : "0.0";
    char b[64];
    auto r = std::to_chars(b, b + sizeof(b), std::abs(v), std::chars_format::scientific);   // d.ddde±XX, кратчайшее
    std::string s(b, r.ptr);
    size_t e = s.find('e');
    std::string mant = s.substr(0, e);
    int exp10 = std::stoi(s.substr(e + 1));
    std::string digits;
    for (char c : mant) if (c != '.') digits += c;
    int decpt = exp10 + 1;
    std::string out = v < 0 ? "-" : "";
    if (decpt > -4 && decpt <= 16) {               // repr: фиксированная запись при −4 < decpt ≤ 16
        if (decpt <= 0) out += "0." + std::string(size_t(-decpt), '0') + digits;
        else if (size_t(decpt) >= digits.size()) out += digits + std::string(decpt - digits.size(), '0') + ".0";
        else out += digits.substr(0, decpt) + "." + digits.substr(decpt);
    } else {
        out += digits.substr(0, 1);
        if (digits.size() > 1) out += "." + digits.substr(1);
        char eb[16];
        std::snprintf(eb, sizeof(eb), "e%c%02d", exp10 < 0 ? '-' : '+', std::abs(exp10));
        out += eb;
    }
    return out;
}

std::string u8(const fs::path& p) {
    auto s = p.u8string();
    return std::string(s.begin(), s.end());
}

fs::path from_u8(const std::string& s) { return fs::path(std::u8string(s.begin(), s.end())); }

std::string posix(const fs::path& p) {
    std::string s = u8(p);
    std::replace(s.begin(), s.end(), '\\', '/');
    return s;
}

std::string qpath(const fs::path& p) {
    std::string s = u8(p);
    for (char c : s)
        if (std::isspace(static_cast<unsigned char>(c))) return "\"" + s + "\"";
    return s;
}

fs::path exe_path(const char* argv0) {
#ifdef _WIN32
    wchar_t buf[32768];
    DWORD n = GetModuleFileNameW(nullptr, buf, DWORD(std::size(buf)));
    if (n > 0 && n < std::size(buf)) return fs::path(std::wstring(buf, n));
#else
    std::error_code ec;
    fs::path p = fs::read_symlink("/proc/self/exe", ec);
    if (!ec) return p;
#endif
    return fs::canonical(fs::absolute(from_u8(argv0)));
}

Dirs default_dirs(const char* argv0) {
    Dirs d;
    const fs::path exe_dir = exe_path(argv0).parent_path();
    if (const char* r = std::getenv("GEO4D_ROOT")) d.root = fs::canonical(from_u8(r));
    else if (fs::exists(exe_dir / "shaders" / "pathtrace.slang")) {
        d.installed = true;
        d.root = exe_dir;
        d.shaders = exe_dir / "shaders";
        d.scenes = exe_dir / "scenes";
        d.nodes = exe_dir / "nodes";
        d.gen = user_cache_dir() / "gen";
        d.spv = user_cache_dir() / "spv";
        return d;
    } else d.root = exe_dir.parent_path().parent_path();
    d.shaders = d.root / "gpu" / "shaders";
    d.scenes = d.root / "scenes";
    d.nodes = d.root / "nodes";
    d.gen = d.root / "gpu" / "build" / "gen";
    d.spv = d.root / "gpu" / "build" / "spv";
    return d;
}

fs::path user_cache_dir() {
    if (const char* c = std::getenv("GEO4D_CACHE_DIR")) return from_u8(c);
#ifdef _WIN32
    if (const char* l = std::getenv("LOCALAPPDATA")) return from_u8(l) / "geo4d" / "cache";
#else
    if (const char* x = std::getenv("XDG_CACHE_HOME")) return from_u8(x) / "geo4d";
    if (const char* h = std::getenv("HOME")) return from_u8(h) / ".cache" / "geo4d";
#endif
    return fs::temp_directory_path() / "geo4d-cache";
}

// ---- заголовок и дефайны -----------------------------------------------------------------------------------

namespace {

// вещественный литерал параметра: как MF_* (repr); отрицательный — в скобках («x - (-4.0)»)
std::string param_lit(double v) {
    std::string s = pyrepr(v);
    return v < 0 ? "(" + s + ")" : s;
}

// Слоты в шейдере (DD-011 §5): строка r — SLOT_ROW(r) = gCam[kSlotBase + r] (mf_header). Числа многообразия по
// умолчанию запечены: их чтение в горячем коде стоило +31 % кадра (сцена 3, 1080p), см. DD-011, «Итог», шаг 4
std::string slot_row(int row) { return "SLOT_ROW(" + std::to_string(row) + ")"; }
std::string sref(const Scene&, int i) { return slot_row(i / 4) + "." + std::string(1, "xyzw"[i % 4]); }

// строка obj_g для объекта из формулы его ноды (DD-011 §4): параметры — поля строк объекта (rows), int-параметры
// формы ({axis}) подставляются в текст как координата

std::map<std::string, std::string> slot_params(const Scene& sc) {
    std::map<std::string, std::string> pm;
    for (size_t i = 0; i < sc.slots.v.size(); ++i) pm["$" + std::to_string(i)] = sref(sc, int(i));
    return pm;
}

std::string object_g_line(const Scene& sc, const Object& o, size_t k) {
    const std::string ks0 = std::to_string(k);
    if (o.kind == "graph") {                      // составная форма (M36): выражение целиком, числа — слоты (или bake)
        const std::string g = sc.bake ? emit_slang(bind_params(o.expr, sc.slots.values()), {}) : emit_slang(o.expr, slot_params(sc));
        return "    if (k == " + ks0 + ") { return " + g + "; }";
    }
    const NodeDef* def = node_library().find(o.node);
    if (!def || def->formula.empty()) throw std::runtime_error("нет формулы ноды «" + o.node + "»");
    std::string f = def->formula;
    std::vector<std::string> names;
    std::map<std::string, std::string> pm;
    for (auto& p : def->params) {
        if (p.type == "int") {
            const std::string key = "{" + p.name + "}";
            for (size_t at; (at = f.find(key)) != std::string::npos;)
                f.replace(at, key.size(), std::string(1, "xyzw"[int(o.values.at(p.name)) & 3]));
            continue;
        }
        auto r = def->rows.find(p.name);
        const int n = p.type == "vec3" || p.type == "color" ? 3 : p.type == "vec4" ? 4 : 1;
        for (int i = 0; i < n; ++i) {
            const std::string nm = n == 1 ? p.name : p.name + "." + "xyzw"[i];
            names.push_back(nm);
            pm[nm] = r == def->rows.end() ? param_lit(o.values.at(nm)) : (n == 1 ? r->second : r->second + "." + "xyzw"[i]);
        }
    }
    const std::string ks = std::to_string(k);
    return "    if (k == " + ks + ") { float4 c = obj_c(" + ks + "), s = obj_s(" + ks + "); return " +
           emit_slang(parse_expr(f, names), pm) + "; }";
}

}  // namespace

std::string slot_ref(int i) {
    return "gCam[" + std::to_string(kSlotBase + i / 4) + "]." + std::string(1, "xyzw"[i % 4]);
}

std::vector<float> slot_rows(const Scene& sc) {
    if (sc.bake || sc.slots.v.empty()) return {};
    if (sc.slots.v.size() > size_t(4 * kMaxSlotRows))
        throw std::runtime_error("граф сцены: чисел-параметров больше " + std::to_string(4 * kMaxSlotRows));
    std::vector<double> R = sc.slots.v;
    R.resize((R.size() + 3) / 4 * 4, 0.0);
    return f32(R);
}

std::string mf_header(const Scene& sc) {
    const NodeDef* def = node_library().find(sc.mf_node);
    // формула: у нынешних видов — из библиотеки; у листов с горловинами и mf.formula — собранная по графу (DD-011)
    const std::string formula = !sc.mf.formula.empty() ? sc.mf.formula : def ? def->formula : "";
    if (!def || formula.empty()) throw std::runtime_error("нет формулы ноды многообразия «" + sc.mf_node + "»");
    const Manifold& m = sc.mf;
    // число многообразия: слот (DD-011 §5, только с mf_slots) или литерал. Слоты многообразия стоили +5…10 % кадра в
    // движении и ~+10 % в покое (сцена 3, 1080p; бюджет +5 %) — по умолчанию числа запечены, правка = перегенерация
    const bool slots = !sc.bake && sc.mf_slots;
    auto num = [&](int slot, double v) { return slots && slot >= 0 ? sref(sc, slot) : param_lit(v); };
    std::vector<std::string> names;
    std::map<std::string, std::string> pm;
    for (auto& [n, v] : sc.mf_values) {
        names.push_back(n);
        auto it = m.slot_of.find(n);
        pm[n] = num(it == m.slot_of.end() ? -1 : it->second, v);
    }
    const std::string sdf = emit_slang(parse_expr(formula, names), pm);
    // hcap (DD-001, M37): Эллис — 0.5|q|; плоские зоны с шарами кривизны — расстояние до ближайшего шара, не меньше ¼ℓ_s
    const std::string scale = num(m.scale_slot, m.scale);
    std::string hcap = "SCALE";
    if (m.exit == 1) hcap = "0.5 * length(p.xyz)";
    else if (m.flat.has && !m.flat.centers.empty()) {
        std::string mn;
        for (size_t i = 0; i < m.flat.centers.size(); ++i) {
            const auto& c = m.flat.centers[i];
            std::string t = slots && m.ball_slot >= 0
                                ? "length(p.xyz - " + slot_row(m.ball_slot / 4 + int(i)) + ".xyz)"
                                : "length(p.xyz - float3(" + param_lit(c[0]) + ", " + param_lit(c[1]) + ", " + param_lit(c[2]) + "))";
            mn = mn.empty() ? t : "min(" + mn + ", " + t + ")";
        }
        hcap = "max(" + mn + " - " + num(m.rho0_slot, m.flat.rho0) + ", 0.25 * " + scale + ")";
    }
    std::string head = "// Сгенерировано из графа сцены (DD-011, нода " + def->type + ") — не редактировать.\n";
    if (!sc.bake)                                 // слоты — строки буфера камеры (привязка 0 у всех ядер окна)
        head += "#ifndef GCAM_DECLARED\n#define GCAM_DECLARED\n[[vk::binding(0, 0)]] RWStructuredBuffer<float4> gCam;\n#endif\n"
                "#define SLOT_ROW(r) gCam[" + std::to_string(kSlotBase) + " + (r)]\n";
    if (slots) head += "#define MF_SLOTS 1\n#define SCALE (" + scale + ")\n";
    else head += "static const float SCALE = " + scale + ";\n";
    return head + "__generic<T : IReal> T sdf(T x, T y, T z, T w) { return " + sdf + "; }\n"
                  "float hcap(float4 p) { return " + hcap + "; }\n";
}

std::string structure_key(const Scene& sc) {
    Sha256 h;
    h.update(scene_header(sc, false, true));
    h.update(mf_header(sc));
    for (auto& [k, v] : defines(sc, 0, false, true)) h.update(k + "=" + v + ";");
    const Manifold& m = sc.mf;
    // push-константы многообразия (push_items), фон и плоские зоны (секции графа, директива flat)
    h.update("s_max=" + pyrepr(m.has_s_max ? m.s_max : -1.0) + ";ell_max=" + pyrepr(m.has_ell_max ? m.ell_max : -1.0) +
             ";env=" + sc.env + ";flat=" + std::to_string(m.flat.has) + ";exit=" + std::to_string(m.exit) +
             ";nslot=" + std::to_string(sc.slots.v.size()) + ";bake=" + std::to_string(sc.bake) +
             ";mfslots=" + std::to_string(sc.mf_slots));
    return h.hexdigest();
}

std::string scene_header(const Scene& sc, bool dynamic, bool graph) {
    std::vector<std::string> L = {"// Сгенерировано geo4d.gpu_pt.scene_header из описания сцены прототипа — не редактировать.",
                                  "#define PT_MAXOBJ " + std::to_string(kPtMaxObj),
                                  std::string("#define PT_DYNAMIC ") + (dynamic ? "1" : "0")};
    if (dynamic) {
        L.insert(L.end(), {"#define PT_FLAT_DYNAMIC 1", "#define PT_HAS_FLAT (int(gCam[MF_BASE + 4].x))",
                           "#define PT_NBALL (int(gCam[MF_BASE + 4].y))", "#define PT_RHO0 (gCam[MF_BASE + 4].z)",
                           "float3 pt_ball_c(int i) { return mf_ball(i); }"});
        return join(L, "\n") + "\n";
    }
    const FlatZones& fz = sc.mf.flat;
    const bool slots = graph && !sc.bake;
    if (slots && sc.mf_slots && fz.has) {        // DD-011 §5: ρ₀ и центры шаров кривизны — слоты (строки подряд)
        // проверки плоских зон (in_flat и др.) — в циклах интегратора: центры и ρ₀ читаются потоком один раз в массив
        // static (регистры); чтение буфера на каждой проверке стоило проходу камеры +61 % (сцена 3, 1080p)
        const Manifold& m = sc.mf;
        const std::string nb = std::to_string(fz.centers.size()), nb1 = std::to_string(fz.centers.size() + 1);
        std::string load = "static float4 sPtFlat[" + nb1 + "];\nstatic bool sPtFlatOk = false;\n"
                           "void pt_flat_load() { if (sPtFlatOk) return; sPtFlatOk = true; sPtFlat[" + nb + "] = float4(" +
                           sref(sc, m.rho0_slot) + ");";
        if (m.ball_slot >= 0)
            load += " [ForceUnroll] for (int i = 0; i < " + nb + "; i++) sPtFlat[i] = SLOT_ROW(" + std::to_string(m.ball_slot / 4) + " + i);";
        load += " }";
        L.insert(L.end(), {"#define PT_FLAT_DYNAMIC 1", "#define PT_HAS_FLAT 1", "#define PT_NBALL " + nb, load,
                           "float pt_rho0() { pt_flat_load(); return sPtFlat[" + nb + "].x; }", "#define PT_RHO0 (pt_rho0())",
                           "float3 pt_ball_c(int i) { pt_flat_load(); return sPtFlat[i].xyz; }"});
    } else {
    L.push_back("#define PT_FLAT_DYNAMIC 0");
    if (fz.has) {
        std::vector<std::string> cs;
        for (auto& c : fz.centers) cs.push_back("float3(" + fmt9g(c[0]) + ", " + fmt9g(c[1]) + ", " + fmt9g(c[2]) + ")");
        L.insert(L.end(), {"#define PT_HAS_FLAT 1", "#define PT_NBALL " + std::to_string(fz.centers.size()),
                           "static const float PT_RHO0 = " + fmt9g(fz.rho0) + ";", arr("float3", "PT_BALL_C", cs, fz.centers.size())});
    } else {
        L.insert(L.end(), {"#define PT_HAS_FLAT 0", "#define PT_NBALL 0", "static const float PT_RHO0 = 0;",
                           "static const float3 PT_BALL_C[1] = { float3(0) };"});
    }
    L.push_back("float3 pt_ball_c(int i) { return PT_BALL_C[i]; }");
    }
    std::vector<std::string> isplane;
    for (auto& o : sc.objs) isplane.push_back(o.kind == "plane" ? "1" : "0");
    L.push_back("#define PT_NOBJ " + std::to_string(sc.objs.size()));
    L.push_back(arr("int", "PT_ISPLANE", isplane, sc.objs.size()));
    L.push_back("__generic<T : IReal> T obj_g(int k, T x, T y, T z, T w)\n{");
    for (size_t k = 0; k < sc.objs.size(); ++k) {
        const Object& o = sc.objs[k];
        std::string ks = std::to_string(k);
        if (graph) L.push_back(object_g_line(sc, o, k));
        else if (o.kind == "ball4")
            L.push_back("    if (k == " + ks + ") { float4 c = obj_c(" + ks + "); return jlength4(x - c.x, y - c.y, z - c.z, w - c.w) - obj_s(" + ks + ").y; }");
        else if (o.kind == "torus")
            L.push_back("    if (k == " + ks + ") { float4 c = obj_c(" + ks + "), s = obj_s(" + ks + "); "
                        "return jlength3(jlength2(x - c.x, y - c.y) - s.y, z - c.z, w - c.w) - s.z; }");
        else
            L.push_back("    if (k == " + ks + ") { float4 s = obj_s(" + ks + "); return s.w * (" + std::string(1, "xyzw"[o.axis]) + " - s.z); }");
    }
    L.push_back("    return T.cst(1e30);\n}");
    return join(L, "\n") + "\n";
}

fs::path write_header(const Dirs& d, const Scene& sc, bool dynamic, bool graph) {
    std::string text = scene_header(sc, dynamic, graph);
    std::string mf = graph ? mf_header(sc) : "";
    Sha256 h;
    h.update(text);
    h.update(mf);                                 // без графа — пусто: каталог тот же, что у Python
    fs::path dir = d.gen / h.hexdigest().substr(0, 16);
    fs::create_directories(dir);
    auto write = [&](const char* name, const std::string& t) {
#ifdef _WIN32
        std::string out;                          // Path.write_text в Windows пишет CRLF
        for (char c : t) { if (c == '\n') out += '\r'; out += c; }
#else
        const std::string& out = t;
#endif
        std::ofstream f(dir / name, std::ios::binary);
        f << out;
    };
    write("pt_scene.slang", text);
    if (graph) write("mf_scene.slang", mf);
    return dir;
}

Defines defines(const Scene& sc, int nj, bool dynamic, bool graph) {
    if (dynamic) return {{"SCENE", std::to_string(kSceneDynamic)}, {"EXIT", "0"}, {"KERNEL", "0"}, {"NJ", std::to_string(nj)}};
    if (graph) return {{"SCENE", "12"}, {"EXIT", std::to_string(sc.mf.exit)}, {"KERNEL", "0"}, {"NJ", std::to_string(nj)}};
    Defines d = {{"SCENE", std::to_string(sc.mf.scene_id)}, {"EXIT", std::to_string(sc.mf.exit)}, {"KERNEL", "0"},
                 {"NJ", std::to_string(nj)}};
    // gpu_pt.manifold_defines (_MF): параметры фигуры → MF_*
    static const std::vector<std::pair<std::string, std::vector<std::pair<std::string, std::string>>>> mf = {
        {"S3", {{"R", "MF_R"}}}, {"spheritorus", {{"R", "MF_R"}, {"r", "MF_RT"}}},
        {"torisphere", {{"R", "MF_R"}, {"r", "MF_RT"}}}, {"tiger", {{"R1", "MF_R1"}, {"R2", "MF_R2"}, {"r", "MF_RT"}}},
        {"ellis", {{"b", "MF_B"}}}, {"compact_wh", {{"b", "MF_B"}, {"rho0", "MF_RHO0"}, {"W", "MF_W"}}},
        {"two_compact_wh", {{"b", "MF_B"}, {"rho0", "MF_RHO0"}, {"W", "MF_W"}}}};
    for (auto& [name, keys] : mf)
        if (sc.mf.shape_name == name)
            for (auto& [k, def] : keys) d.emplace_back(def, pyrepr(sc.mf.p(k)));
    if (sc.mf.shape_name == "two_compact_wh")
        for (int i = 1; i <= 2; ++i)
            for (int a = 0; a < 3; ++a)
                d.emplace_back(std::string("MF_C") + char('0' + i) + "XYZ"[a], pyrepr(sc.mf.p(i == 1 ? "c1" : "c2", a)));
    return d;
}

// ---- ключи кэша SPIR-V ------------------------------------------------------------------------------------

ShaderKeys::ShaderKeys(const Dirs& d) : d_(d) {
    for (auto& p : slang_files(d.shaders)) shader_bytes_ += read_file(p);
}

std::string ShaderKeys::key_json(const std::string& source, const std::string& entry, const Defines& defs) const {
    Defines s = defs;
    std::sort(s.begin(), s.end(), [](auto& a, auto& b) { return a.first < b.first; });
    std::string j = "[\"" + source + "\", \"" + entry + "\", [";
    for (size_t i = 0; i < s.size(); ++i) {
        bool str = s[i].first.rfind("MF_", 0) == 0;          // MF_* — строки-литералы, остальные — целые
        j += (i ? ", " : "") + std::string("[\"") + s[i].first + "\", " + (str ? "\"" + s[i].second + "\"" : s[i].second) + "]";
    }
    return j + "]]";
}

fs::path ShaderKeys::spv_path(const std::string& source, const std::string& entry, const Defines& defs,
                              const fs::path& include) const {
    Sha256 h;
    h.update(shader_bytes_);
    if (!include.empty())
        for (auto& p : slang_files(include)) h.update(read_file(p));
    h.update(key_json(source, entry, defs));
    std::string stem = fs::path(source).stem().string();
    return d_.spv / (stem + "_" + entry + "_" + h.hexdigest().substr(0, 16) + ".spv");
}

// ---- строки буфера камеры -----------------------------------------------------------------------------------

std::vector<float> camera_rows(const Camera& cam, double k_grad, double flight_r, int filter_mode, int cube_n) {
    double tf = std::tan(cam.fov * (kPi / 180.0) / 2);
    std::vector<double> R;
    for (const Vec4* v : {&cam.x, &cam.f, &cam.r, &cam.u}) R.insert(R.end(), v->begin(), v->end());
    R.insert(R.end(), {tf, double(cam.width), double(cam.height), cam.focus});
    R.insert(R.end(), {cam.aperture, flight_r, k_grad, kSigmaPix});
    R.insert(R.end(), {2 * tf / cam.width, kDetLo, std::min(kDetHi, 1e30), std::min(kDetTau, 1e30)});
    R.insert(R.end(), {double(filter_mode), double(cube_n), 0, 0});
    return f32(R);
}

std::vector<float> object_rows(const Scene& sc) {
    if (sc.objs.size() > size_t(kPtMaxObj)) throw std::runtime_error("объектов больше PT_MAXOBJ = 32");
    std::vector<double> R(size_t(kObjRows) * 4, 0.0);
    auto row = [&](int i) { return &R[size_t(i) * 4]; };
    auto lights = nee_lights(sc.objs, sc.mf.flat);
    row(0)[0] = double(sc.objs.size());
    row(0)[1] = double(lights.size());
    for (size_t i = 0; i < lights.size(); ++i) R[4 + i] = lights[i];
    for (size_t k = 0; k < sc.objs.size(); ++k) {
        const Object& o = sc.objs[k];
        int b = 9 + 6 * int(k);
        auto set = [&](int i, double a, double bb, double c, double d) { row(i)[0] = a; row(i)[1] = bb; row(i)[2] = c; row(i)[3] = d; };
        if (o.kind == "graph") {                  // составная форма: центр и радиус ограничивающей сферы (M36)
            std::copy(o.center.begin(), o.center.end(), row(b));
            set(b + 1, 3, std::min(o.r, 1e30), 0, 0);
        } else if (o.kind == "ball4") {
            std::copy(o.center.begin(), o.center.end(), row(b));
            set(b + 1, 0, o.r, 0, 0);
        } else if (o.kind == "torus") {
            std::copy(o.center.begin(), o.center.end(), row(b));
            set(b + 1, 1, o.R, o.r, 0);
        } else {
            set(b + 1, 2, o.axis, o.offset, o.sign);
            set(b + 2, 1, o.axis, o.offset, o.sign);
        }
        set(b + 3, o.mat.albedo[0], o.mat.albedo[1], o.mat.albedo[2], material_kind(o.mat.kind));
        set(b + 4, o.mat.emit[0], o.mat.emit[1], o.mat.emit[2], o.mat.rough);
        set(b + 5, std::min(o.bound(), 1e30), o.on_M ? 1.0 : 0.0, 0, 0);
    }
    return f32(R);
}

std::vector<float> manifold_rows(const Scene& sc, const PTParams& p) {
    std::vector<double> R(size_t(kMfRows) * 4, 0.0);
    auto row = [&](int i) { return &R[size_t(i) * 4]; };
    const Manifold& m = sc.mf;
    row(0)[0] = m.scene_id;
    row(0)[1] = m.exit;
    row(0)[2] = m.scale;
    row(0)[3] = (m.flat.has || m.exit == 1) ? 1.0 : 0.0;
    static const std::vector<std::pair<std::string, std::vector<std::string>>> keys = {
        {"S3", {"R"}}, {"spheritorus", {"R", "r"}}, {"torisphere", {"R", "r"}}, {"tiger", {"R1", "R2", "r"}},
        {"ellis", {"b"}}, {"compact_wh", {"b", "rho0", "W"}}, {"two_compact_wh", {"b", "rho0", "W"}}};
    for (auto& [k, ks] : keys)
        if (m.kind == k)
            for (size_t i = 0; i < ks.size(); ++i) row(1)[i] = m.p(ks[i]);
    if (m.flat.has) {
        for (size_t i = 0; i < m.flat.centers.size() && i < 2; ++i)
            std::copy(m.flat.centers[i].begin(), m.flat.centers[i].end(), row(2 + int(i)));
        row(4)[0] = 1;
        row(4)[1] = double(m.flat.centers.size());
        row(4)[2] = m.flat.rho0;
    }
    row(4)[3] = sc.env == "none" ? 0.0 : 1.0;
    row(5)[0] = std::min(p.s_max, m.has_s_max ? m.s_max : p.s_max);
    row(5)[1] = m.has_ell_max ? m.ell_max : 0.0;
    return f32(R);
}

// ---- граф окна (gpu_pt.app_graph) ------------------------------------------------------------------------------

namespace {

struct GraphCtx {
    const Dirs& d;
    const ShaderKeys& keys;
    const Scene& sc;
    const AppOptions& o;
    fs::path td;
    fs::path scene_path;
    const EnvMap& procedural;
    Camera cam;
    double fov;
    std::vector<ShaderJob>* jobs;
    bool prev_ok = false;                         // сцена переводится в байткод (пределы bytecode.hpp)
};

std::vector<std::string> scene_directives(const Scene& sc) {
    std::vector<std::string> L;
    const FlatZones& fz = sc.mf.flat;
    if (fz.has) {
        std::vector<std::string> w = {"flat", std::to_string(fz.sheets.size())};
        for (double s : fz.sheets) w.push_back(fmt9g(s));
        w.push_back(std::to_string(fz.centers.size()));
        w.push_back(fmt9g(fz.rho0));
        for (auto& c : fz.centers) for (double a : c) w.push_back(fmt9g(a));
        L.push_back(join(w, " "));
    }
    for (size_t k = 0; k < sc.objs.size(); ++k) {
        std::string n = sc.objs[k].name;
        std::replace(n.begin(), n.end(), ' ', '_');
        L.push_back("objname " + std::to_string(k) + " " + n);
    }
    return L;
}

std::vector<std::string> editor_directives(const GraphCtx& c) {
    // без вида scenes.slang многообразие в редакторе DD-008/009 не правится (нет строк многообразия) — только editor
    if (!c.sc.mf.legacy()) {
        auto q = [](const std::string& s) {
            for (char ch : s) if (std::isspace(static_cast<unsigned char>(ch))) return "\"" + s + "\"";
            return s;
        };
        return {"editor " + q(posix(c.scene_path)) + " " + q(posix(c.td)) + " " + fmt9g(c.fov)};
    }
    std::vector<std::string> w = {"manifold", c.sc.mf.kind, c.sc.env_field.empty() ? "-" : c.sc.env_field};
    for (auto& [k, v] : c.sc.mf.params) {
        std::vector<std::string> s;
        for (double a : v) s.push_back(fmt9g(a));
        w.push_back(k + "=" + join(s, ","));
    }
    auto q = [](const std::string& s) {
        for (char ch : s) if (std::isspace(static_cast<unsigned char>(ch))) return "\"" + s + "\"";
        return s;
    };
    return {"editor " + q(posix(c.scene_path)) + " " + q(posix(c.td)) + " " + fmt9g(c.fov), join(w, " ")};
}

std::vector<std::string> build(const GraphCtx& c, bool dynamic, std::map<std::string, Array>& arrays) {
    const Scene& sc = c.sc;
    const int W = c.o.width, H = c.o.height;
    const long long N = (long long)W * H;
    // фон графа: у динамического — процедурный всегда (DD-009), у быстрого — как в сцене
    EnvMap none;
    const EnvMap* env = &c.procedural;
    if (!dynamic && sc.env == "none") { none = constant_env(0.0f, 8); env = &none; }
    // DD-011: многообразие и объекты быстрого графа — из формул графа сцены (режим A тоже берёт mf_scene.slang)
    // многообразие без вида scenes.slang (листы с горловинами, формула) — только из графа
    const bool graph = !dynamic && !sc.mf_node.empty() && (!c.o.legacy_codegen || !sc.dynamic_ok());
    fs::path inc = write_header(c.d, sc, dynamic, graph);
    auto df = [&](int nj) { return defines(sc, nj, dynamic, graph); };
    auto with = [](Defines d, const std::string& k, int v) { d.emplace_back(k, std::to_string(v)); return d; };
    auto spv = [&](const std::string& src, const std::string& e, const Defines& d, bool use_inc) {
        fs::path out = c.keys.spv_path(src, e, d, use_inc ? inc : fs::path{});
        bool seen = false;
        for (auto& j : *c.jobs) seen = seen || j.out == out;
        if (!seen) c.jobs->push_back({c.d.shaders / src, e, d, use_inc ? inc : fs::path{}, out});
        return qpath(out);
    };
    std::string ptc = spv("pathtrace.slang", "pt_camera", df(4), true);
    auto pts = [&](const std::string& e) { return spv("pathtrace.slang", e, with(df(0), "PT_STAGE", e == "pt_sample"), true); };
    std::string cst = spv("pathtrace.slang", "cam_step", with(df(0), "CAM_STEP", 1), true);
    std::string snp = spv("pathtrace.slang", "pt_snap", with(df(0), "CAM_STEP", 1), true);
    auto md = [&](int k, const std::string& e) {
        Defines d = df(0);
        for (auto& [n, v] : d) if (n == "KERNEL") v = std::to_string(k);
        return spv("modea.slang", e, d, graph);
    };
    const long long g = (N + 63) / 64;
    const std::string G = std::to_string(g);
    PTParams p;
    PTParams pfb = p;
    pfb.seed = p.seed + 7919;
    const std::string sfx = dynamic ? "_dyn" : "";
    // буфер камеры: камера + объекты + многообразие (pt_camera_buffer)
    Array cam;
    cam.data = camera_rows(c.cam, 2.0, 0.0, 0, env->n);
    auto orows = object_rows(sc), mrows = manifold_rows(sc);
    cam.data.insert(cam.data.end(), orows.begin(), orows.end());
    cam.data.insert(cam.data.end(), mrows.begin(), mrows.end());
    const std::vector<float> srows = graph ? slot_rows(sc) : std::vector<float>{};   // DD-011 §5
    cam.data.insert(cam.data.end(), srows.begin(), srows.end());
    cam.shape = {cam.data.size() / 4, 4};
    arrays["cam" + sfx + ".npy"] = cam;
    Array envb;
    envb.data = env->table;
    envb.shape = {envb.data.size()};
    arrays["env" + sfx + ".npy"] = envb;
    Array cube;
    cube.data = env->cube;
    cube.shape = {2, 6, size_t(env->n), size_t(env->n), 3};
    arrays["cube" + sfx + ".npy"] = cube;
    const int env_n = env->table_n;
    std::string enc = " " + spv("bc6h_encode.slang", "bc6h_encode", {}, false);
    std::vector<std::string> L = {"buffer cam in " + qpath(c.td / ("cam" + sfx + ".npy")),
                                  "buffer Env in " + qpath(c.td / ("env" + sfx + ".npy")),
                                  "cube Cube " + qpath(c.td / ("cube" + sfx + ".npy")) + " bc6h 16" + enc};
    const std::pair<const char*, long long> zero[] = {
        {"Acc", 16 * N}, {"Acc2", 16 * N}, {"Stats", 16 * N}, {"Pix", 8 * N}, {"Alb", 16 * N}, {"Nrm", 16 * N},
        {"Dir", 16 * N}, {"Dd", 64 * N}, {"Status", 4 * N}, {"Class", 4 * N}, {"Flag", 4 * N}, {"Cnt", 16},
        {"Args", 32}, {"Color", 16 * N}, {"Present", 4 * N}};
    for (auto& [n, b] : zero) L.push_back(std::string("buffer ") + n + " zero " + std::to_string(b));
    for (const char* n : {"Frame", "AlbF", "NrmF", "Den"}) L.push_back(std::string("buffer ") + n + " export " + std::to_string(16 * N));
    L.push_back("pipeline camstep " + cst + " cam_step cam");
    L.push_back("pipeline snap " + snp + " pt_snap cam");
    L.push_back("pipeline camera " + ptc + " pt_camera " + kPtBind);
    L.push_back("pipeline shade " + md(2, "modea_shade") + " modea_shade cam Dir Dd Status Color Cube");
    L.push_back("pipeline detect " + md(3, "modea_detect") + " modea_detect cam Dir Dd Status Flag");
    const bool background = sc.mf.flat.has || sc.mf.exit == 1;
    const std::pair<const char*, const char*> pipes[] = {
        {"classall", "pt_classall"}, {"clear", "pt_clear"}, {"lists", "pt_lists"}, {"args", "pt_args"}, {"ptA", "pt_sample"},
        {"ptB", "pt_sample"}, {"reset", "pt_reset"}, {"compose", "pt_compose"}, {"present", "pt_present"}};
    for (auto& [n, e] : pipes) L.push_back(std::string("pipeline ") + n + " " + pts(e) + " " + e + " " + kPtBind);
    // DD-011 §6: нативный предпросмотр (без --legacy-codegen: граф Python его не имеет)
    const bool preview = !dynamic && !c.o.legacy_codegen;
    if (preview) {
        L.push_back("buffer GB zero " + std::to_string(32 * N));
        L.push_back("pipeline preview " + pts("pt_preview") + " pt_preview " + kPtBind + " GB");
    }
    PushItems cm(7, {'f', 0.0});
    cm.push_back({'f', 0.05});
    cm.push_back({'u', 16});
    L.push_back(push_line("camstep", cm));
    L.push_back(push_line("snap", cm));
    L.push_back(push_line("camera", push_items(sc, p, N, env_n)));
    L.push_back(push_line("clear", push_items(sc, p, N, env_n)));
    L.push_back(push_line("lists", push_items(sc, p, N, env_n, 0, 1.0, 0, N)));
    L.push_back(push_line("args", push_items(sc, p, N, env_n)));
    L.push_back(push_line("ptA", push_items(sc, p, N, env_n, 2, 1.0, 0, 0)));
    L.push_back(push_line("ptB", push_items(sc, pfb, N, env_n, 2, 1.0, 1, N)));
    L.push_back(push_line("reset", push_items(sc, p, N, env_n)));
    L.push_back(push_line("compose", push_items(sc, p, N, env_n)));
    L.push_back(push_line("classall", push_items(sc, p, N, env_n)));
    L.push_back(push_line("present", push_items(sc, p, N, env_n, 0, 1.0, 0, 0, 1, 1)));
    if (preview) L.push_back(push_line("preview", push_items(sc, p, N, env_n)));
    auto body = [&](bool bg) {
        std::vector<std::string> b = bg ? std::vector<std::string>{"dispatch camera " + G + " 1 1", "mark camera",
                                                                  "dispatch shade " + G + " 1 1", "dispatch detect " + G + " 1 1"}
                                        : std::vector<std::string>{"dispatch classall " + G + " 1 1", "mark camera"};
        b.insert(b.end(), {"dispatch clear 1 1 1", "dispatch lists " + G + " 1 1", "dispatch args 1 1 1",
                           "dispatch reset " + G + " 1 1", "mark lists", "indirect ptA Args 0", "mark pt", "loop 8",
                           "indirect ptB Args 12", "endloop", "mark fallback", "dispatch compose " + G + " 1 1", "mark compose"});
        return b;
    };
    std::vector<std::pair<std::string, bool>> variants;
    if (dynamic) variants = {{"", true}, {"_nobg", false}};
    else variants = {{"", background}};
    for (auto& [s, bg] : variants) {
        L.insert(L.end(), {"section move" + s, "mark start", "dispatch camstep 1 1 1"});
        auto b = body(bg);
        L.insert(L.end(), b.begin(), b.end());
        L.insert(L.end(), {"section edit" + s, "mark start"});
        L.insert(L.end(), b.begin(), b.end());
    }
    if (preview)
        L.insert(L.end(), {"section preview_move", "mark start", "dispatch camstep 1 1 1", "dispatch preview " + G + " 1 1",
                           "mark preview", "section preview", "mark start", "dispatch preview " + G + " 1 1", "mark preview"});
    L.insert(L.end(), {"section snap", "dispatch snap 1 1 1", "section static", "mark start", "indirect ptA Args 0", "mark pt",
                       "indirect ptB Args 12", "mark fallback", "dispatch compose " + G + " 1 1", "mark compose",
                       "section post", "dispatch present " + G + " 1 1",
                       "app " + std::to_string(W) + " " + std::to_string(H) + " Present", "denoise Frame AlbF NrmF Den",
                       "campipe camstep",
                       "presentpipe present " + std::to_string(kIUseDen) + " " + std::to_string(kIBgra) + " " + std::to_string(kIExposure),
                       "seedpipes " + std::to_string(kISeed) + " ptA ptB", "speed " + pyrepr(sc.speed),
                       "objrows cam " + std::to_string(kObjBase), "mfrows cam " + std::to_string(kMfBase)});
    if (!srows.empty())                           // правка числа графа — запись строк (структура — ключ)
        L.insert(L.end(), {"slots cam " + std::to_string(kSlotBase) + " " + std::to_string(srows.size() / 4) + (sc.mf_slots ? " mf" : ""),
                           "slotkey " + structure_key(sc)});
    auto sd = scene_directives(sc);
    L.insert(L.end(), sd.begin(), sd.end());
    auto ed = editor_directives(c);
    L.insert(L.end(), ed.begin(), ed.end());
    if (!dynamic) {                               // граф режима редактирования — без vsync/fullscreen, как в Python
        if (sc.dynamic_ok()) L.push_back("editgraph " + qpath(c.td / "app_graph_dyn.txt"));   // DD-009: только виды scenes.slang
        else L.push_back("graphdoc");             // сцена не выразима в version 1: редактор сохраняет и перегенерирует граф
        if (preview && c.prev_ok) {               // DD-011 §6: варианты без / с составными объектами, 8 / 32 регистра
            std::string w = "prevgraph";
            for (const char* n : {"p8", "o8", "p32", "o32"}) w += " " + qpath(c.td / (std::string("app_graph_prev_") + n + ".txt"));
            L.push_back(w);
        }
        if (c.o.vsync) L.push_back("vsync");
        if (c.o.fullscreen) L.push_back("fullscreen");
    }
    return L;
}

// Граф предпросмотра на интерпретаторе (DD-011 §6): шейдеры от сцены не зависят (SCENE_DYNAMIC + MF_PROG, объекты —
// строки, составные формы и многообразие — программы байткода), поэтому компилируются один раз на все сцены; сцену
// задают строки буфера камеры: камера, объекты, многообразие, (слоты не нужны), область программ с kProgBase.
// Секции: move — шаг камеры и кадр предпросмотра, edit — кадр, static — ничего (кадр не меняется), post — вывод.
// Варианты: без составных объектов и с ними (objs: PROG_OBJ), на 8 и 32 регистра — окно выбирает по сцене.
std::vector<std::string> build_preview(const GraphCtx& c, std::map<std::string, Array>& arrays, bool objs, int nreg) {
    const Scene& sc = c.sc;
    const int W = c.o.width, H = c.o.height;
    const long long N = (long long)W * H;
    fs::path inc = write_header(c.d, sc, true, false);           // заголовок динамического графа — от сцены не зависит
    // дефайны: вывод от варианта не зависит; шаг камеры — только от числа регистров; кадр — от обоих (общие SPIR-V
    // у вариантов — меньше ядер при холодном старте)
    const Defines base = defines(sc, 0, true, false);
    Defines dfm = base;
    dfm.emplace_back("MF_PROG", "1");
    // массив регистров интерпретатора: 8 — вдвое быстрее, чем 16 или 32 (results/stage-5/…_preview_bench)
    dfm.emplace_back("PROG_NREG", std::to_string(nreg));
    Defines df = dfm;
    if (objs) df.emplace_back("PROG_OBJ", "1");   // составные объекты — программы (вариант дороже, pathtrace.slang)
    auto with = [](Defines d, const std::string& k, int v) { d.emplace_back(k, std::to_string(v)); return d; };
    auto spv = [&](const std::string& e, const Defines& d) {
        fs::path out = c.keys.spv_path("pathtrace.slang", e, d, inc);
        bool seen = false;
        for (auto& j : *c.jobs) seen = seen || j.out == out;
        if (!seen) c.jobs->push_back({c.d.shaders / "pathtrace.slang", e, d, inc, out});
        return qpath(out);
    };
    const EnvMap& env = c.procedural;
    Array cam;
    cam.data = camera_rows(c.cam, 2.0, 0.0, 0, env.n);
    auto orows = object_rows(sc), mrows = manifold_rows(sc);
    cam.data.insert(cam.data.end(), orows.begin(), orows.end());
    cam.data.insert(cam.data.end(), mrows.begin(), mrows.end());
    cam.data.resize(size_t(kProgBase) * 4, 0.0f);
    auto prog = program_rows(sc);
    cam.data.insert(cam.data.end(), prog.begin(), prog.end());
    cam.data.resize(size_t(kProgBase + kProgMaxRows) * 4, 0.0f);   // запас под программы любой сцены (правки в окне)
    cam.shape = {cam.data.size() / 4, 4};
    arrays["cam_prev.npy"] = cam;
    if (!arrays.count("cube_dyn.npy")) {          // фон — процедурный, как у графа режима редактирования (общие массивы)
        Array envb, cube;
        envb.data = env.table;
        envb.shape = {envb.data.size()};
        cube.data = env.cube;
        cube.shape = {2, 6, size_t(env.n), size_t(env.n), 3};
        arrays["env_dyn.npy"] = envb;
        arrays["cube_dyn.npy"] = cube;
    }
    const std::string G = std::to_string((N + 63) / 64);
    std::string enc = " " + qpath(c.keys.spv_path("bc6h_encode.slang", "bc6h_encode", {}, {}));
    {
        fs::path out = c.keys.spv_path("bc6h_encode.slang", "bc6h_encode", {}, {});
        bool seen = false;
        for (auto& j : *c.jobs) seen = seen || j.out == out;
        if (!seen) c.jobs->push_back({c.d.shaders / "bc6h_encode.slang", "bc6h_encode", {}, {}, out});
    }
    std::vector<std::string> L = {"buffer cam in " + qpath(c.td / "cam_prev.npy"), "buffer Env in " + qpath(c.td / "env_dyn.npy"),
                                  "cube Cube " + qpath(c.td / "cube_dyn.npy") + " bc6h 16" + enc};
    // буферы, которые читают ядра предпросмотра и вывода (кадр, класс, флаги, G-буфер), — полного размера; остальные
    // привязки — минимальные
    const std::pair<const char*, long long> bufs[] = {
        {"Acc", 16}, {"Acc2", 16}, {"Stats", 16}, {"Pix", 16}, {"Alb", 16}, {"Nrm", 16}, {"Dir", 16}, {"Dd", 16},
        {"Status", 16}, {"Class", 4 * N}, {"Flag", 4 * N}, {"Cnt", 16}, {"Args", 32}, {"Frame", 16 * N}, {"AlbF", 16},
        {"NrmF", 16}, {"Den", 16}, {"Present", 4 * N}, {"Color", 16}, {"GB", 32 * N}};
    for (auto& [n, b] : bufs) L.push_back(std::string("buffer ") + n + " zero " + std::to_string(b));
    L.push_back("pipeline camstep " + spv("cam_step", with(dfm, "CAM_STEP", 1)) + " cam_step cam");
    L.push_back("pipeline preview " + spv("pt_preview", with(df, "PT_STAGE", 0)) + " pt_preview " + kPtBind + " GB");
    L.push_back("pipeline present " + spv("pt_present", with(base, "PT_STAGE", 0)) + " pt_present " + kPtBind);
    PTParams p;
    const int env_n = env.table_n;
    PushItems cm(7, {'f', 0.0});
    cm.push_back({'f', 0.05});
    cm.push_back({'u', 16});
    L.push_back(push_line("camstep", cm));
    L.push_back(push_line("preview", push_items(sc, p, N, env_n)));
    L.push_back(push_line("present", push_items(sc, p, N, env_n, 0, 1.0, 0, 0, 1, 1)));
    L.insert(L.end(), {"section move", "mark start", "dispatch camstep 1 1 1", "dispatch preview " + G + " 1 1", "mark preview",
                       "section edit", "mark start", "dispatch preview " + G + " 1 1", "mark preview",
                       "section static", "mark start",
                       "section post", "dispatch present " + G + " 1 1",
                       "app " + std::to_string(W) + " " + std::to_string(H) + " Present", "campipe camstep",
                       "presentpipe present " + std::to_string(kIUseDen) + " " + std::to_string(kIBgra) + " " + std::to_string(kIExposure),
                       "speed " + pyrepr(sc.speed), "objrows cam " + std::to_string(kObjBase), "mfrows cam " + std::to_string(kMfBase),
                       "progrows cam " + std::to_string(kProgBase) + " " + std::to_string(kProgMaxRows), "preview " + std::to_string(int(objs)) + " " + std::to_string(nreg)});
    auto sd = scene_directives(sc);
    L.insert(L.end(), sd.begin(), sd.end());
    auto ed = editor_directives(c);
    L.insert(L.end(), ed.begin(), ed.end());
    L.push_back("graphdoc");                      // сцена правится документом графа (панель «Параметры»)
    if (c.o.vsync) L.push_back("vsync");
    return L;
}

}  // namespace

AppGraph app_graph(const Dirs& d, const Scene& sc0, const AppOptions& o, const fs::path& work_dir,
                   const fs::path& scene_path, const EnvMap& procedural) {
    Scene sc = sc0;
    sc.bake = sc.bake || o.bake_params;
    sc.mf_slots = sc.mf_slots || o.manifold_slots;
    ShaderKeys keys(d);
    double fov = o.fov != 0 ? o.fov : sc.cam.fov;
    AppGraph g;
    GraphCtx c{d, keys, sc, o, work_dir, scene_path, procedural, make_camera(sc.mf, sc.cam, o.width, o.height, fov), fov,
               &g.shaders};
    if (sc.dynamic_ok()) g.dyn_lines = build(c, true, g.arrays);  // как в Python: граф режима редактирования — первым
    if (!o.legacy_codegen) {                      // DD-011 §6: граф предпросмотра на интерпретаторе
        try {
            for (int nreg : {kProgNarrowReg, kProgMaxReg})
                for (bool objs : {false, true})
                    g.prev_graphs.emplace_back(std::string("app_graph_prev_") + (objs ? "o" : "p") + std::to_string(nreg) + ".txt",
                                               build_preview(c, g.arrays, objs, nreg));
            c.prev_ok = true;
        } catch (const std::exception& e) {       // программа не помещается в пределы — только нативный путь
            std::fprintf(stderr, "предпросмотр на интерпретаторе недоступен: %s\n", e.what());
        }
    }
    g.lines = build(c, false, g.arrays);
    return g;
}

}  // namespace geo4d
