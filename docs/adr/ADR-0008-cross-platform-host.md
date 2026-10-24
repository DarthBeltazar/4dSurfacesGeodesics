# ADR-0008: Кроссплатформенный хост и необязательный денойзер

**Контекст.** DD-007 оставлял окно для Linux/macOS вне объёма. После этапа 5 пользователь попросил поддержку окон на
других ОС и удобный запуск сцен из IDE. Привязки к Windows были в четырёх местах:
- экспорт памяти `VK_KHR_external_memory_win32`;
- загрузка `nvcuda.dll`;
- SDL3 из `Lib/SDL3.lib` Vulkan SDK;
- `.exe` и `slangc.exe` в Python.

Кроме того, хост без OptiX собирался без окна: `app.cpp` был под `GEO4D_OPTIX`.

**Решение.**
- **Внешняя память по платформе.** Windows: HANDLE `OPAQUE_WIN32`. Linux: `VK_KHR_external_memory_fd` +
  `vkGetMemoryFdKHR` → `CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD`. После успешного импорта fd принадлежит CUDA, и хост
  его не закрывает (`Buffer::ext_handle = -1`). macOS: внешней памяти нет, `buffer X export` создаёт обычный буфер.
- **CUDA Driver API:** `dlopen("libcuda.so.1")` на Linux, `LoadLibrary("nvcuda.dll")` на Windows (шим ADR-0007).
- **Денойзер необязателен в рантайме.** Если CUDA/OptiX недоступны (не-NVIDIA GPU, macOS, WSL) или импорт
  памяти не удался, приложение пишет предупреждение и работает без денойзера (R ничего не делает). Окно
  (`GEO4D_APP`) и OptiX (`GEO4D_OPTIX`, на macOS по умолчанию OFF) — независимые опции CMake.
- **MoltenVK:** включаются `VK_KHR_portability_enumeration` (с флагом экземпляра) и `VK_KHR_portability_subset`,
  если они есть. Обязательные возможности устройства (int64, BC, анизотропия, массивы кубмап, scalar layout, host
  query reset) проверяются до `vkCreateDevice` с понятной ошибкой. `subgroupSizeControl`/`computeFullSubgroups`
  включаются, только если поддерживаются: шейдеры их не требуют.
- **Сборка.**
  - Корневой `CMakeLists.txt` (`add_subdirectory(gpu)`) — его открывает CLion. `CMakePresets.json`: `release` →
    `gpu/build`, где хост ищет Python; `debug` → `gpu/build-debug`.
  - Заголовки Vulkan, volk и VMA берутся из `$VULKAN_SDK` или системы, иначе скачиваются FetchContent (версии SDK
    1.4.328, VMA 3.3.0). SDL3: Vulkan SDK на Windows (его `SDL3Config.cmake` битый, цель задаётся вручную),
    системный пакет или FetchContent `release-3.2.22`.
  - `gpu/build.sh` для Linux/macOS.
  - В Python: `geo4d.gpu.EXE` без `.exe` вне Windows, переопределение `GEO4D_GPU_EXE`; `slangc` — `GEO4D_SLANGC`,
    `$VULKAN_SDK/Bin|bin`, затем PATH.
- **IDE:** общие конфигурации `.run/*.run.xml` (CLion/PyCharm) генерирует `prototype/scripts/ide_configs.py`. Путь
  интерпретатора `.venv` зависит от ОС, поэтому на Linux/macOS их нужно один раз перегенерировать.
  `app.py --graph-only DIR` пишет граф с входными данными для запуска хоста под отладчиком.

**Проверено** (`results/stage-5/2026-10-24_linux_wsl_smoke`):
- **Windows:** `pytest -m gpu` — 32 passed, без пропусков; обмен памятью с CUDA по-прежнему побитово равен пути
  через копии (`test_gpu_app`).
- **Linux:** сборка gcc 15 в WSL2 (Ubuntu 26.04), все зависимости из FetchContent, заголовки X11 из пакетов без root.
  - Приложение на lavapipe (Mesa, CPU) в окне X11 (WSLg): полёт сквозь горловину, переход на нижний лист,
    накопление 11, образ swapchain побитово равен буферу вывода.
  - Денойзер штатно отключился. В WSL `libnvoptix.so.1` — заглушка dxcore без `optixQueryFunctionTable`, но
    `libcuda.so.1` загрузилась и `cuInit` прошёл.
  - GPU-тесты `run`/`graph` на lavapipe — см. журнал 2026-10-24-07.
- **Не проверено:** OptiX и импорт fd на нативном Linux с драйвером NVIDIA; macOS целиком (нет машины). Код для
  macOS собран по документации MoltenVK. SDL-драйвер `offscreen` на lavapipe не создаёт swapchain
  (`VK_ERROR_OUT_OF_DEVICE_MEMORY`); для окна он и не нужен.

**Последствия.** Первая сборка без SDK требует сети. На Linux нужны dev-пакеты X11 или Wayland для SDL3 (или
системный `libsdl3-dev`). Шейдеры по-прежнему компилирует Python через slangc; для Linux есть сборки slang.

**Статус.** Accepted, 2026-10-24 (по запросу пользователя; меняет не-цель DD-007).
