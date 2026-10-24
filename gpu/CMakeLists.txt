cmake_minimum_required(VERSION 3.28)
project(geo4d_gpu LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)          # модулей C++20 нет
include(FetchContent)

add_executable(geo4d_gpu
  src/main.cpp
  src/vk_context.cpp
  src/job.cpp
  src/graph.cpp
  src/graph_program.cpp
)
# exe всегда в корне каталога сборки (gpu/build/geo4d_gpu[.exe]) — его ищет geo4d.gpu.EXE
set_target_properties(geo4d_gpu PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# Заголовки Vulkan, volk и VMA (DD-004 п. 1): из Vulkan SDK ($VULKAN_SDK) или системы; если нет — скачиваются
# (Linux/macOS без SDK, ADR-0008). Версии — как в SDK 1.4.328.
set(VK_SDK "$ENV{VULKAN_SDK}")
find_path(GEO4D_VULKAN_INC vulkan/vulkan.h HINTS "${VK_SDK}/Include" "${VK_SDK}/include")
find_path(GEO4D_VOLK_INC volk.h HINTS "${VK_SDK}/Include/Volk" "${VK_SDK}/include/Volk" "${VK_SDK}/include")
find_path(GEO4D_VMA_INC vk_mem_alloc.h HINTS "${VK_SDK}/Include/vma" "${VK_SDK}/include/vma" PATH_SUFFIXES vma)
# SOURCE_SUBDIR без CMakeLists.txt — только исходники, без add_subdirectory (CMake ≥ 3.28)
if(NOT GEO4D_VULKAN_INC)
  FetchContent_Declare(vulkan_headers GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
                       GIT_TAG vulkan-sdk-1.4.328.0 GIT_SHALLOW TRUE SOURCE_SUBDIR none)
  FetchContent_MakeAvailable(vulkan_headers)
  set(GEO4D_VULKAN_INC "${vulkan_headers_SOURCE_DIR}/include" CACHE PATH "" FORCE)
endif()
if(NOT GEO4D_VOLK_INC)
  FetchContent_Declare(volk GIT_REPOSITORY https://github.com/zeux/volk.git
                       GIT_TAG vulkan-sdk-1.4.328.0 GIT_SHALLOW TRUE SOURCE_SUBDIR none)
  FetchContent_MakeAvailable(volk)
  set(GEO4D_VOLK_INC "${volk_SOURCE_DIR}" CACHE PATH "" FORCE)
endif()
if(NOT GEO4D_VMA_INC)
  FetchContent_Declare(vma GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
                       GIT_TAG v3.3.0 GIT_SHALLOW TRUE SOURCE_SUBDIR none)
  FetchContent_MakeAvailable(vma)
  set(GEO4D_VMA_INC "${vma_SOURCE_DIR}/include" CACHE PATH "" FORCE)
endif()
target_include_directories(geo4d_gpu PRIVATE "${GEO4D_VULKAN_INC}" "${GEO4D_VOLK_INC}" "${GEO4D_VMA_INC}" src)
target_compile_definitions(geo4d_gpu PRIVATE VK_NO_PROTOTYPES)
target_link_libraries(geo4d_gpu PRIVATE ${CMAKE_DL_LIBS})        # volk загружает libvulkan через dlopen

# Окно приложения (DD-007): SDL3 — из Vulkan SDK (Windows), системный пакет или скачивается и собирается.
option(GEO4D_APP "Интерактивное окно (SDL3)" ON)
if(GEO4D_APP)
  if(WIN32 AND EXISTS "${VK_SDK}/Lib/SDL3.lib")
    # SDL3Config.cmake из SDK 1.4.328 указывает на несуществующий Lib/cmake/lib — цель задаётся вручную
    add_library(SDL3::SDL3 SHARED IMPORTED)
    set_target_properties(SDL3::SDL3 PROPERTIES IMPORTED_IMPLIB "${VK_SDK}/Lib/SDL3.lib"
                          IMPORTED_LOCATION "${VK_SDK}/Bin/SDL3.dll" INTERFACE_INCLUDE_DIRECTORIES "${VK_SDK}/Include")
    set(SDL3_FOUND TRUE)
  else()
    find_package(SDL3 3.2 CONFIG QUIET)
  endif()
  if(NOT SDL3_FOUND)
    message(STATUS "SDL3 не найден — скачивается release-3.2.22")
    set(SDL_SHARED ON CACHE BOOL "" FORCE)
    set(SDL_STATIC OFF CACHE BOOL "" FORCE)
    set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(SDL3 GIT_REPOSITORY https://github.com/libsdl-org/SDL.git GIT_TAG release-3.2.22 GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(SDL3)
  endif()
  # GUI редактора сцен (DD-008): Dear ImGui, бэкенды SDL3 + Vulkan (через volk)
  FetchContent_Declare(imgui GIT_REPOSITORY https://github.com/ocornut/imgui.git GIT_TAG v1.91.9b GIT_SHALLOW TRUE
                       SOURCE_SUBDIR none)
  FetchContent_MakeAvailable(imgui)
  add_library(geo4d_imgui STATIC
    "${imgui_SOURCE_DIR}/imgui.cpp" "${imgui_SOURCE_DIR}/imgui_draw.cpp" "${imgui_SOURCE_DIR}/imgui_tables.cpp"
    "${imgui_SOURCE_DIR}/imgui_widgets.cpp" "${imgui_SOURCE_DIR}/imgui_demo.cpp"
    "${imgui_SOURCE_DIR}/backends/imgui_impl_sdl3.cpp" "${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp")
  target_include_directories(geo4d_imgui PUBLIC "${imgui_SOURCE_DIR}" "${imgui_SOURCE_DIR}/backends"
                             PRIVATE "${GEO4D_VULKAN_INC}" "${GEO4D_VOLK_INC}")
  target_compile_definitions(geo4d_imgui PUBLIC IMGUI_IMPL_VULKAN_USE_VOLK PRIVATE VK_NO_PROTOTYPES)
  target_link_libraries(geo4d_imgui PUBLIC SDL3::SDL3)
  target_sources(geo4d_gpu PRIVATE src/app.cpp src/editor.cpp)
  target_link_libraries(geo4d_gpu PRIVATE SDL3::SDL3 geo4d_imgui)
  target_compile_definitions(geo4d_gpu PRIVATE GEO4D_APP=1)
  if(WIN32)       # SDL3.dll рядом с exe
    add_custom_command(TARGET geo4d_gpu POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       "$<TARGET_FILE:SDL3::SDL3>" "$<TARGET_FILE_DIR:geo4d_gpu>")
  endif()
endif()

# Денойзер OptiX (DD-005, ADR-0007): заголовки optix-dev скачиваются при сборке (лицензия NVIDIA SDK — не в репозиторий),
# библиотека — в драйвере; вместо CUDA Toolkit — шим src/cuda_min/cuda.h и динамическая загрузка nvcuda.dll /
# libcuda.so.1. На macOS CUDA нет.
if(APPLE)
  set(GEO4D_OPTIX_DEFAULT OFF)
else()
  set(GEO4D_OPTIX_DEFAULT ON)
endif()
option(GEO4D_OPTIX "Денойзер OptiX" ${GEO4D_OPTIX_DEFAULT})
if(GEO4D_OPTIX)
  FetchContent_Declare(optix_dev GIT_REPOSITORY https://github.com/NVIDIA/optix-dev.git GIT_TAG v9.1.0 GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(optix_dev)
  target_sources(geo4d_gpu PRIVATE src/denoise.cpp src/denoiser.cpp)
  target_include_directories(geo4d_gpu PRIVATE "${optix_dev_SOURCE_DIR}/include" src/cuda_min)
  target_compile_definitions(geo4d_gpu PRIVATE GEO4D_OPTIX=1)
  if(WIN32)
    target_link_libraries(geo4d_gpu PRIVATE cfgmgr32)
  endif()
endif()

if(WIN32)
  target_compile_definitions(geo4d_gpu PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN VK_USE_PLATFORM_WIN32_KHR)
endif()
if(MSVC)
  target_compile_options(geo4d_gpu PRIVATE /W4 /permissive- /utf-8 /wd4100 /wd4189 /wd4324)
else()
  target_compile_options(geo4d_gpu PRIVATE -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable)
endif()
