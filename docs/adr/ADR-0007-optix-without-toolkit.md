# ADR-0007: Денойзер OptiX без CUDA Toolkit

**Контекст.** Ревью DD-005: денойзер — OptiX (DD-005 предполагал установку CUDA Toolkit, потому что `optix_host.h`
включает `<cuda.h>`). Библиотека OptiX (`nvoptix.dll`) и CUDA Driver API (`nvcuda.dll`) уже есть в драйвере 610.88.
Toolkit — ≈ 3 ГБ, установщик с правами администратора.

**Проверено.** Из `<cuda.h>` заголовкам OptiX 9.1 нужны только типы `CUcontext`, `CUstream`, `CUdeviceptr`
(подсчёт по `optix-dev` v9.1.0). Хосту денойзера нужны 17 функций Driver API (контекст, поток, память, события).

**Решение.**
- `gpu/src/cuda_min/cuda.h` — шим с тремя типами; функции Driver API загружаются `GetProcAddress` из `nvcuda.dll`
  (`gpu/src/denoise.cpp`); `optixInit` находит `nvoptix.dll` сам (`optix_stubs.h`).
- Заголовки `NVIDIA/optix-dev` (лицензия NVIDIA SDK, без права распространять исходники) скачиваются CMake
  FetchContent при сборке, в репозиторий не попадают. Опция `GEO4D_OPTIX` (по умолчанию ON).
- Стенд: `geo4d_gpu denoise <job>` — данные через `.npy`, время по событиям CUDA. Обмен памятью Vulkan ↔ CUDA
  (`VK_KHR_external_memory_win32` → `cuImportExternalMemory`) не сделан: в замерах кадра время денойзера — только
  invoke + вычисление интенсивности, без пересылок.

**Измерено** (`results/stage-4/2026-10-24_gpu_pt_quality`, `…_gpu_pt_profile`): 1080p — 6.9 мс, 4K — 27 мс;
рабочая память 349 / 1361 МБ. Качество — см. отчёт этапа 4.

**Последствия.** Сборка требует сети при первом конфигурировании (FetchContent). Для интерактивного режима (этап 5)
нужен обмен памятью без копий — следующий шаг.

**Статус.** Accepted, 2026-10-24.
