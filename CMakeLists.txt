# Корневой проект для IDE (CLion открывает корень репозитория): хост GPU-порта — в gpu/.
# Сборка из консоли: gpu/build.cmd (Windows) или gpu/build.sh (Linux/macOS); профили — CMakePresets.json.
cmake_minimum_required(VERSION 3.28)
project(geo4d_gpu LANGUAGES C CXX)
add_subdirectory(gpu)
