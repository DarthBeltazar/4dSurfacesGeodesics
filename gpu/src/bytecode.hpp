// Байткод выражений графа сцены (DD-011 §6) для интерпретатора в шейдерах (scenes.slang, prog_eval): формула →
// SSA → регистры → строки float4. Кодировка и коды операций — как у пробы prototype/geo4d/bytecode.py (её evaluate
// исполняет эти программы в тестах, bc_check):
//   строка 0 — (число инструкций, число регистров, 0, 0); инструкция — (op + 256·dst + 65536·a, b + 256·c, k, 0);
//   регистры 0…3 на входе — x, y, z, w; результат — регистр 0.
// Однородные подвыражения (не зависят от x…w) считаются на CPU в double и входят константами: правка числа — новые
// строки программы (как запись слотов), отдельной загрузки слота в байткоде нет. Деление, smin, length и compact_h
// раскладываются на унарные функции, a ± b, a·b и выбор по значению (M6).
#pragma once

#include <array>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "expr.hpp"

namespace geo4d {

struct Scene;

constexpr int kProgMaxReg = 32;                   // PROG_NREG в scenes.slang
constexpr int kProgMaxIns = 1024;                 // на программу (DD-011 §6)
constexpr int kProgMaxBall = 16;                  // центры шаров кривизны в строках программ
constexpr int kProgMaxRows = 8192;                // вся область программ

struct Program {
    std::vector<std::array<float, 4>> rows;       // строка 0 — заголовок
    std::vector<std::array<double, 4>> rows64;    // те же строки с константами k в double (сверка bc_check)
    int nins = 0, nreg = 0;
};

// params — значения параметров выражения (слоты «$i», a…d формулы и т. п.); превышение пределов — runtime_error
Program compile_program(const ExprPtr& e, const std::map<std::string, double>& params);

// Область программ буфера камеры с kProgBase (codegen.hpp), см. scenes.slang (PROG_BASE):
//   +0 — (число шаров кривизны, 0, 0, 0); +1…+16 — центры шаров (M29); +17…+48 — начальная строка программы объекта k
//   (0 — у объекта нет программы); +49 — программа многообразия, затем программы составных объектов (вид 3).
std::vector<float> program_rows(const Scene& sc);

// geo4d_gpu bytecode <сцена> <выход.json>: программы многообразия и объектов (строки) — для bc_check
int run_bytecode(int argc, char** argv);

}  // namespace geo4d
