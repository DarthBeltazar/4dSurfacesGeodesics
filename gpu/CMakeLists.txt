cmake_minimum_required(VERSION 3.24)
project(geo4d_gpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Vulkan SDK: заголовки, volk и VMA берутся из SDK (DD-004 п. 1)
if(NOT DEFINED ENV{VULKAN_SDK})
  message(FATAL_ERROR "VULKAN_SDK не задан")
endif()
set(VK_SDK "$ENV{VULKAN_SDK}")

add_executable(geo4d_gpu
  src/main.cpp
  src/vk_context.cpp
  src/job.cpp
  src/graph.cpp
  src/graph_program.cpp
)

# Денойзер OptiX (DD-005, ADR-0007): заголовки optix-dev скачиваются при сборке (лицензия NVIDIA SDK — не в репозиторий),
# библиотека — в драйвере; вместо CUDA Toolkit — шим src/cuda_min/cuda.h и динамическая загрузка nvcuda.dll.
option(GEO4D_OPTIX "Денойзер OptiX" ON)
if(GEO4D_OPTIX)
  include(FetchContent)
  FetchContent_Declare(optix_dev GIT_REPOSITORY https://github.com/NVIDIA/optix-dev.git GIT_TAG v9.1.0 GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(optix_dev)
  target_sources(geo4d_gpu PRIVATE src/denoise.cpp src/denoiser.cpp src/app.cpp)
  # окно приложения (DD-007): SDL3 из Vulkan SDK
  target_link_libraries(geo4d_gpu PRIVATE "${VK_SDK}/Lib/SDL3.lib")
  add_custom_command(TARGET geo4d_gpu POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     "${VK_SDK}/Bin/SDL3.dll" "$<TARGET_FILE_DIR:geo4d_gpu>")
  target_include_directories(geo4d_gpu PRIVATE "${optix_dev_SOURCE_DIR}/include" src/cuda_min)
  target_compile_definitions(geo4d_gpu PRIVATE GEO4D_OPTIX=1)
  target_link_libraries(geo4d_gpu PRIVATE cfgmgr32)
endif()
target_include_directories(geo4d_gpu PRIVATE
  "${VK_SDK}/Include"
  "${VK_SDK}/Include/Volk"
  "${VK_SDK}/Include/vma"
  src)
target_compile_definitions(geo4d_gpu PRIVATE VK_NO_PROTOTYPES NOMINMAX WIN32_LEAN_AND_MEAN VK_USE_PLATFORM_WIN32_KHR)
if(MSVC)
  target_compile_options(geo4d_gpu PRIVATE /W4 /permissive- /utf-8 /wd4100 /wd4189 /wd4324)
endif()
