// Окно без Python (DD-010, шаг 3) — см. standalone.cpp.
#pragma once

// без аргументов, файл сцены .json или параметр --… первым — запуск окна (иначе — подкоманды info/run/graph/…)
bool is_standalone_args(int argc, char** argv);
int run_standalone(int argc, char** argv);
