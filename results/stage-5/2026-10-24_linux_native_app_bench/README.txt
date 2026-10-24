Нативный запуск на Linux (ADR-0008): WSL2 Ubuntu 26.04, свежий git clone, ./gpu/build.sh (пресет release, Unix Makefiles,
SDL3/Vulkan-Headers/volk/VMA — FetchContent; заголовки X11 — через CMAKE_PREFIX_PATH/CFLAGS из локального префикса),
venv на Python 3.14 из requirements.txt, slangc 2025.17.2 для Linux в PATH, VULKAN_SDK не задан.
Устройство — lavapipe (llvmpipe, CPU); время кадров не показательно. Окно X11 (WSLg), скрытое.
Проверено: шейдеры скомпилированы Linux-slangc, app.py → geo4d_gpu app, образ swapchain побитово равен буферу вывода
(summary.json, swapchain_*.npy против Present_*.npy). Буферы AOV/денойзера не сохранены (денойзер в WSL недоступен).
Команда — cmd.txt (из корня клона в WSL).
