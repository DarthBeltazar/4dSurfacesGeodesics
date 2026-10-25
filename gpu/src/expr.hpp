// Язык выражений нод (DD-011 §2): разбор, типизация (однородное / поле), вывод в Slang и вычисление на CPU.
//
// Формула — выражение над числами, параметрами ноды, `pi` и координатами x y z w (точка в R⁴ в координатах ноды):
// `+ − * /`, унарный минус, `^` (целая степень), вызовы функций. Подвыражение, не зависящее от x…w, — однородное:
// в Slang оно выводится обычной float-арифметикой (компилятор свернёт его так же, как в рукописных формулах
// scenes.slang), поле — операциями IReal (jet.slang). Числа выводятся так, как записаны. Поэтому формула нынешней
// сцены, записанная как в scenes.slang, даёт тот же текст Slang, а значит тот же SPIR-V.
//
// Контексты: геометрия (многообразие, объекты) — только гладкие функции (M6); текстура — ещё и негладкие (DD-011 §2,
// пока не реализованы: шаг 7). Ошибка — исключение ExprError с позицией в формуле.
#pragma once

#include <array>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace geo4d {

enum class Context { Geometry, Texture };

struct ExprError : std::runtime_error {
    size_t pos;                                   // символ формулы (0-based)
    ExprError(const std::string& msg, size_t p) : std::runtime_error(msg), pos(p) {}
};

struct Expr {
    enum Kind { Num, Param, Coord, Neg, Add, Sub, Mul, Div, Call } kind = Num;
    double value = 0;                             // Num
    std::string text;                             // Num — лексема; Param — имя; Call — имя функции
    int coord = 0;                                // Coord: 0…3 = x y z w
    std::vector<std::shared_ptr<Expr>> args;      // операнды
    bool field = false;                           // зависит от x…w
    size_t pos = 0;
};
using ExprPtr = std::shared_ptr<Expr>;

// разбор и проверка: имена параметров — params; неизвестное имя, функция или запрещённая в контексте — ExprError
ExprPtr parse_expr(const std::string& src, const std::vector<std::string>& params, Context ctx = Context::Geometry);

// Построение выражений из нод (DD-011, составные объекты M36): число, операция, вызов; параметры → числа;
// подстановка координат (преобразования: f(x − t), f(R⁻¹x), s·f(x/s))
ExprPtr mk_num(double v);
ExprPtr mk_op(Expr::Kind k, ExprPtr a, ExprPtr b = nullptr);          // Add Sub Mul Div; Neg — один аргумент
ExprPtr mk_call(const std::string& name, std::vector<ExprPtr> args);
ExprPtr mk_coord(int i);
ExprPtr bind_params(const ExprPtr& e, const std::map<std::string, double>& values);
ExprPtr substitute_coords(const ExprPtr& e, const std::array<ExprPtr, 4>& coords);

// Slang: params — имя параметра → выражение Slang (литерал, дефайн, слот); coords — имена x y z w в функции
std::string emit_slang(const ExprPtr& e, const std::map<std::string, std::string>& params,
                       const std::array<std::string, 4>& coords = {"x", "y", "z", "w"});

// Вычисление на CPU в double: значение, градиент и гессиан по x y z w (jet второго порядка, как Jet15)
struct EvalResult {
    double f = 0;
    std::array<double, 4> g{};
    std::array<std::array<double, 4>, 4> H{};
};
EvalResult eval_expr(const ExprPtr& e, const std::map<std::string, double>& params, const std::array<double, 4>& p);

// geo4d_gpu eval <вход.json> <выход.json>: формулы, параметры, точки → значения, градиенты, гессианы, Slang, ошибки
int run_eval(int argc, char** argv);

}  // namespace geo4d
