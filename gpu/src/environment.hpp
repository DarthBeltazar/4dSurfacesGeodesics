// Фон окна (DD-010): процедурная кубмапа (geo4d.render.background, запекание geo4d.cubemap.bake) и таблица
// выборки фона M29 (geo4d.pathtrace.env_table / geo4d.gpu_pt.env_buffer). Порядок операций — как в numpy/JAX,
// чтобы данные совпадали с Python (проверка — tests/test_standalone_export.py).
#pragma once

#include <vector>

namespace geo4d {

struct EnvMap {
    int n = 0;                                    // размер грани уровня 0
    std::vector<float> cube;                      // (2 листа, 6 граней, n, n, 3) — как gpu.cube_for_gpu(mips[0])
    int table_n = 0;                              // размер грани уровня таблицы
    std::vector<float> table;                     // [cdf листа 0 | cdf листа 1 | pmf листа 0 | pmf листа 1]
};

EnvMap procedural_env(int n);                     // geo4d.pathtrace.procedural_env(n)
EnvMap constant_env(float value, int n = 8);      // geo4d.pathtrace.constant_env(value, n)

}  // namespace geo4d
