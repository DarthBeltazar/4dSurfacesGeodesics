#!/usr/bin/env sh
# Сборка geo4d_gpu на Linux/macOS (ADR-0008): ./gpu/build.sh [Release|Debug]
# Профили — CMakePresets.json в корне: Release → gpu/build, Debug → gpu/build-debug. Нужны CMake ≥ 3.28 и компилятор
# C++20; Vulkan SDK ($VULKAN_SDK) необязателен — заголовки Vulkan, volk, VMA и SDL3 иначе скачиваются (FetchContent).
# Без Ninja используется генератор Unix Makefiles.
set -e
cd "$(dirname "$0")/.."
preset=release
[ "$1" = "Debug" ] && preset=debug
if command -v ninja >/dev/null 2>&1; then
  cmake --preset "$preset"
else
  cmake --preset "$preset" -G "Unix Makefiles"
fi
cmake --build --preset "$preset" --parallel
