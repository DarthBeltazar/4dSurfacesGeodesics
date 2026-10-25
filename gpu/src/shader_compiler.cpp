// Компиляция шейдеров в процессе (DD-010, шаг 2) — см. shader_compiler.hpp.
#include "shader_compiler.hpp"

#ifdef GEO4D_SLANG_API
#include <slang.h>
#include <slang-com-ptr.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <thread>

#ifdef _MSC_VER
#pragma warning(disable : 4996)                   // ICompileRequest помечен [[deprecated]], но это ровно путь slangc
#endif

namespace geo4d {

#ifdef GEO4D_SLANG_API
using SessionPtr = Slang::ComPtr<slang::IGlobalSession>;
#else
struct SessionPtr {                               // без Slang API — процесс slangc (сессии нет)
    bool ok = false;
    explicit operator bool() const { return ok; }
};
#endif

struct ShaderCompiler::Impl {
    int threads = 1;
    std::vector<SessionPtr> sessions;
    std::vector<double> session_s;
    std::mutex call;                              // compile() — по одному вызову за раз
};

namespace {

double since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

// аргументы geo4d.gpu.compile_shader; out — файл -o
std::vector<std::string> slang_args(const ShaderJob& j, const fs::path& out) {
    std::vector<std::string> args = {u8(j.source), "-target", "spirv", "-profile", "spirv_1_6", "-entry", j.entry,
                                     "-fvk-use-entrypoint-name", "-O2", "-o", u8(out)};
    for (auto& [k, v] : j.defs) args.push_back("-D" + k + "=" + v);
    if (!j.include.empty()) {
        args.push_back("-I");
        args.push_back(u8(j.include));
    }
    return args;
}

fs::path tmp_spv(const ShaderJob& j) {
    return j.out.parent_path() /
           (j.out.stem().string() + ".tmp" + std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id())) + ".spv");
}

ShaderResult finish(ShaderResult r, const ShaderJob& j, const fs::path& tmp, bool ok, std::chrono::steady_clock::time_point t0) {
    std::error_code ec;
    if (!ok || !fs::exists(tmp)) {
        if (r.log.empty()) r.log = "Slang: ошибка компиляции " + j.entry;
        fs::remove(tmp, ec);
        r.seconds = since(t0);
        return r;
    }
    fs::rename(tmp, j.out, ec);
    if (ec) fs::remove(tmp, ec);
    r.ok = fs::exists(j.out);
    r.seconds = since(t0);
    return r;
}

#ifndef GEO4D_SLANG_API
// процесс slangc: GEO4D_SLANGC или slangc из PATH; диагностика — в файл рядом
ShaderResult compile_one(SessionPtr&, const ShaderJob& j) {
    ShaderResult r;
    auto t0 = std::chrono::steady_clock::now();
    fs::create_directories(j.out.parent_path());
    fs::path tmp = tmp_spv(j), log = tmp;
    log += ".log";
    const char* sc = std::getenv("GEO4D_SLANGC");
    std::string cmd = "\"" + std::string(sc ? sc : "slangc") + "\"";
    for (auto& a : slang_args(j, tmp)) cmd += " \"" + a + "\"";
    cmd += " > \"" + u8(log) + "\" 2>&1";
#ifdef _WIN32
    cmd = "\"" + cmd + "\"";                       // cmd /c снимает внешние кавычки
#endif
    int rc = std::system(cmd.c_str());
    if (std::ifstream f{log}) r.log.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    std::error_code ec;
    fs::remove(log, ec);
    return finish(r, j, tmp, rc == 0, t0);
}
#else
ShaderResult compile_one(SessionPtr& gs, const ShaderJob& j) {
    ShaderResult r;
    auto t0 = std::chrono::steady_clock::now();
    // аргументы — как у geo4d.gpu.compile_shader; -o — во временный файл рядом (то же расширение .spv), затем
    // переименование: параллельные процессы с тем же ключом не видят недописанный файл. Код из запроса
    // (getEntryPointCodeBlob) в режиме командной строки недоступен — падение, проверено.
    fs::create_directories(j.out.parent_path());
    fs::path tmp = tmp_spv(j);
    std::vector<std::string> args = slang_args(j, tmp);
    std::vector<const char*> argv;
    for (auto& a : args) argv.push_back(a.c_str());
    Slang::ComPtr<slang::ICompileRequest> req;
    if (SLANG_FAILED(gs->createCompileRequest(req.writeRef()))) {
        r.log = "Slang: не создан запрос компиляции";
        return r;
    }
    req->setCommandLineCompilerMode();            // как slangc: запрос сам пишет файлы -o
    SlangResult res = req->processCommandLineArguments(argv.data(), int(argv.size()));
    if (SLANG_SUCCEEDED(res)) res = req->compile();
    if (const char* d = req->getDiagnosticOutput()) r.log = d;
    return finish(r, j, tmp, SLANG_SUCCEEDED(res), t0);
}
#endif

}  // namespace

ShaderCompiler::ShaderCompiler(int threads) : impl_(std::make_unique<Impl>()) {
    int hw = int(std::thread::hardware_concurrency());
    impl_->threads = threads > 0 ? threads : std::clamp(hw, 1, 8);
    impl_->sessions.resize(impl_->threads);
    impl_->session_s.assign(impl_->threads, 0.0);
}

ShaderCompiler::~ShaderCompiler() = default;

std::vector<double> ShaderCompiler::session_seconds() const { return impl_->session_s; }

std::vector<ShaderResult> ShaderCompiler::compile(const std::vector<ShaderJob>& jobs) {
    std::lock_guard<std::mutex> lock(impl_->call);
    std::vector<ShaderResult> out(jobs.size());
    std::vector<size_t> todo;
    for (size_t i = 0; i < jobs.size(); ++i) {
        bool dup = false;                         // один и тот же SPIR-V в списке — компилировать один раз
        for (size_t k : todo) dup = dup || jobs[k].out == jobs[i].out;
        if (fs::exists(jobs[i].out)) out[i].ok = out[i].cached = true;
        else if (!dup) todo.push_back(i);
    }
    // сначала долгие (по замеру probe_compile: pt_sample ≫ pt_camera > cam_step > остальные) — меньше хвост
    auto rank = [&](size_t i) {
        const std::string& e = jobs[i].entry;
        return e == "pt_sample" ? 0 : e == "pt_camera" ? 1 : e == "cam_step" ? 2 : 3;
    };
    std::stable_sort(todo.begin(), todo.end(), [&](size_t a, size_t b) { return rank(a) < rank(b); });
    std::atomic<size_t> next{0};
    auto worker = [&](int w) {
        for (size_t q; (q = next.fetch_add(1)) < todo.size();) {
            auto& gs = impl_->sessions[w];
#ifdef GEO4D_SLANG_API
            if (!gs) {
                auto t0 = std::chrono::steady_clock::now();
                SlangGlobalSessionDesc desc = {};
                if (SLANG_FAILED(slang::createGlobalSession(&desc, gs.writeRef()))) {
                    out[todo[q]].log = "Slang: не создана глобальная сессия";
                    continue;
                }
                impl_->session_s[w] = since(t0);
            }
#endif
            out[todo[q]] = compile_one(gs, jobs[todo[q]]);
            out[todo[q]].worker = w;
        }
    };
    int n = std::min<int>(impl_->threads, int(todo.size()));
    std::vector<std::thread> th;
    for (int w = 1; w < n; ++w) th.emplace_back(worker, w);
    if (n > 0) worker(0);
    for (auto& t : th) t.join();
    for (size_t i = 0; i < jobs.size(); ++i)     // повторы в списке — по результату первого
        if (!out[i].ok && out[i].log.empty() && fs::exists(jobs[i].out)) out[i].ok = out[i].cached = true;
    return out;
}

}  // namespace geo4d
