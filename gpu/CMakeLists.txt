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
)
target_include_directories(geo4d_gpu PRIVATE
  "${VK_SDK}/Include"
  "${VK_SDK}/Include/Volk"
  "${VK_SDK}/Include/vma"
  src)
target_compile_definitions(geo4d_gpu PRIVATE VK_NO_PROTOTYPES NOMINMAX WIN32_LEAN_AND_MEAN)
if(MSVC)
  target_compile_options(geo4d_gpu PRIVATE /W4 /permissive- /utf-8 /wd4100 /wd4189 /wd4324)
endif()
