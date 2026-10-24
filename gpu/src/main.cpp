// geo4d_gpu: хост GPU-порта (DD-004).
//   geo4d_gpu info            — устройство и драйвер (JSON)
//   geo4d_gpu run <job.txt>   — одна compute-диспетчеризация по job-файлу (см. job.cpp)
//   geo4d_gpu graph <файл>    — граф диспетчеризаций одним командным буфером (см. graph.cpp)
//   geo4d_gpu denoise <job>   — денойзер OptiX (см. denoise.cpp)
#include <cstdio>
#include <exception>
#include <iostream>
#include <string>

#include "graph.hpp"
#ifdef GEO4D_OPTIX
#include "denoise.hpp"
#endif
#include "job.hpp"
#include "vk_context.hpp"

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::fprintf(stderr, "использование: geo4d_gpu info | run <job.txt> | graph <graph.txt> | denoise <job.txt>\n");
            return 2;
        }
        std::string cmd = argv[1];
#ifdef GEO4D_OPTIX
        if (cmd == "denoise" && argc >= 3) return run_denoise(argv[2]);     // без Vulkan
#endif
        VkContext ctx;
        if (cmd == "info") {
            std::cout << "{\"device\": \"" << ctx.device_name() << "\", \"driver\": \"" << ctx.driver_version()
                      << "\", \"timestamp_period_ns\": " << ctx.timestamp_period_ns() << "}\n";
            return 0;
        }
        if (cmd == "run" && argc >= 3) return run_job(ctx, argv[2]);
        if (cmd == "graph" && argc >= 3) return run_graph(ctx, argv[2]);
        std::fprintf(stderr, "неизвестная команда: %s\n", cmd.c_str());
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ошибка: %s\n", e.what());
        return 1;
    }
}
