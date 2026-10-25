// Компиляция шейдеров в процессе (DD-010, шаг 2): Slang API с теми же аргументами командной строки, что
// geo4d.gpu.compile_shader передаёт slangc.exe, — SPIR-V совпадает с ним. Глобальная сессия Slang не потокобезопасна
// (slang.h), поэтому у каждого рабочего потока — своя; сессии живут, пока жив компилятор (базовый модуль Slang
// грузится один раз на поток). Готовый SPIR-V в кэше не пересобирается (ключ — путь, codegen::ShaderKeys).
#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "codegen.hpp"

namespace geo4d {

struct ShaderResult {
    bool ok = false;
    bool cached = false;                          // файл уже был в кэше
    std::string log;                              // диагностика Slang
    double seconds = 0;
    int worker = -1;
};

class ShaderCompiler {
public:
    explicit ShaderCompiler(int threads = 0);    // 0 — по числу ядер, не больше 8
    ~ShaderCompiler();
    // компилирует отсутствующие в кэше, параллельно; результаты — в порядке jobs
    std::vector<ShaderResult> compile(const std::vector<ShaderJob>& jobs);
    // время создания глобальных сессий (секунды, по потокам; 0 — поток ещё не создавал)
    std::vector<double> session_seconds() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace geo4d
