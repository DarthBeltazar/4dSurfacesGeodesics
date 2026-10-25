// Файлы графа в памяти (DD-010, шаг 3): окно без Python строит граф (текст и .npy) в памяти и регистрирует его под
// путями каталога работы; GraphProgram и app читают сначала отсюда, затем с диска. Так графы из памяти и графы-файлы
// (app.py, тесты, отладка в CLion) проходят один и тот же путь загрузки. Потокобезопасно (перегенерация — в потоке).
#pragma once

#include <string>

#include "npy.hpp"

namespace memfs {

void put_text(const std::string& path, const std::string& text);
void put_npy(const std::string& path, NpyArray a);
// содержимое файла: из памяти или с диска (исключение, если нет нигде)
std::string read_text(const std::string& path);
NpyArray read_npy(const std::string& path);

}  // namespace memfs
