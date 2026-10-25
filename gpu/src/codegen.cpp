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

#include "sha256.hpp"

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

Dirs default_dirs(const char* argv0) {
    Dirs d;
    if (const char* r = std::getenv("GEO4D_ROOT")) d.root = fs::canonical(from_u8(r));
    else d.root = fs::canonical(fs::absolute(from_u8(argv0))).parent_path().parent_path().parent_path();
    d.shaders = d.root / "gpu" / "shaders";
    d.gen = d.root / "gpu" / "build" / "gen";
    d.spv = d.root / "gpu" / "build" / "spv";
    return d;
}

// ---- заголовок и дефайны -----------------------------------------------------------------------------------

std::string scene_header(const Scene& sc, bool dynamic) {
    std::vector<std::string> L = {"// Сгенерировано geo4d.gpu_pt.scene_header из описания сцены прототипа — не редактировать.",
                                  "#define PT_MAXOBJ " + std::to_string(kPtMaxObj),
                                  std::string("#define PT_DYNAMIC ") + (dynamic ? "1" : "0")};
    if (dynamic) {
        L.insert(L.end(), {"#define PT_FLAT_DYNAMIC 1", "#define PT_HAS_FLAT (int(gCam[MF_BASE + 4].x))",
                           "#define PT_NBALL (int(gCam[MF_BASE + 4].y))", "#define PT_RHO0 (gCam[MF_BASE + 4].z)",
                           "float3 pt_ball_c(int i) { return i == 0 ? mf_c1() : mf_c2(); }"});
        return join(L, "\n") + "\n";
    }
    L.push_back("#define PT_FLAT_DYNAMIC 0");
    const FlatZones& fz = sc.mf.flat;
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
    std::vector<std::string> isplane;
    for (auto& o : sc.objs) isplane.push_back(o.kind == "plane" ? "1" : "0");
    L.push_back("#define PT_NOBJ " + std::to_string(sc.objs.size()));
    L.push_back(arr("int", "PT_ISPLANE", isplane, sc.objs.size()));
    L.push_back("__generic<T : IReal> T obj_g(int k, T x, T y, T z, T w)\n{");
    for (size_t k = 0; k < sc.objs.size(); ++k) {
        const Object& o = sc.objs[k];
        std::string ks = std::to_string(k);
        if (o.kind == "ball4")
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

fs::path write_header(const Dirs& d, const Scene& sc, bool dynamic) {
    std::string text = scene_header(sc, dynamic);
    Sha256 h;
    h.update(text);
    fs::path dir = d.gen / h.hexdigest().substr(0, 16);
    fs::create_directories(dir);
#ifdef _WIN32
    std::string out;                              // Path.write_text в Windows пишет CRLF
    for (char c : text) { if (c == '\n') out += '\r'; out += c; }
#else
    const std::string& out = text;
#endif
    std::ofstream f(dir / "pt_scene.slang", std::ios::binary);
    f << out;
    return dir;
}

Defines defines(const Scene& sc, int nj, bool dynamic) {
    if (dynamic) return {{"SCENE", std::to_string(kSceneDynamic)}, {"EXIT", "0"}, {"KERNEL", "0"}, {"NJ", std::to_string(nj)}};
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
        if (o.kind == "ball4") {
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
    fs::path inc = write_header(c.d, sc, dynamic);
    auto df = [&](int nj) { return defines(sc, nj, dynamic); };
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
        return spv("modea.slang", e, d, false);
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
    L.insert(L.end(), {"section snap", "dispatch snap 1 1 1", "section static", "mark start", "indirect ptA Args 0", "mark pt",
                       "indirect ptB Args 12", "mark fallback", "dispatch compose " + G + " 1 1", "mark compose",
                       "section post", "dispatch present " + G + " 1 1",
                       "app " + std::to_string(W) + " " + std::to_string(H) + " Present", "denoise Frame AlbF NrmF Den",
                       "campipe camstep",
                       "presentpipe present " + std::to_string(kIUseDen) + " " + std::to_string(kIBgra) + " " + std::to_string(kIExposure),
                       "seedpipes " + std::to_string(kISeed) + " ptA ptB", "speed " + pyrepr(sc.speed),
                       "objrows cam " + std::to_string(kObjBase), "mfrows cam " + std::to_string(kMfBase)});
    auto sd = scene_directives(sc);
    L.insert(L.end(), sd.begin(), sd.end());
    auto ed = editor_directives(c);
    L.insert(L.end(), ed.begin(), ed.end());
    if (!dynamic) {                               // граф режима редактирования — без vsync/fullscreen, как в Python
        L.push_back("editgraph " + qpath(c.td / "app_graph_dyn.txt"));
        if (c.o.vsync) L.push_back("vsync");
        if (c.o.fullscreen) L.push_back("fullscreen");
    }
    return L;
}

}  // namespace

AppGraph app_graph(const Dirs& d, const Scene& sc, const AppOptions& o, const fs::path& work_dir,
                   const fs::path& scene_path, const EnvMap& procedural) {
    ShaderKeys keys(d);
    double fov = o.fov != 0 ? o.fov : sc.cam.fov;
    AppGraph g;
    GraphCtx c{d, keys, sc, o, work_dir, scene_path, procedural, make_camera(sc.mf, sc.cam, o.width, o.height, fov), fov,
               &g.shaders};
    g.dyn_lines = build(c, true, g.arrays);       // как в Python: граф режима редактирования пишется первым
    g.lines = build(c, false, g.arrays);
    return g;
}

}  // namespace geo4d
