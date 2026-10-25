// geo4d_gpu: хост GPU-порта (DD-004).
//   geo4d_gpu info            — устройство и драйвер (JSON)
//   geo4d_gpu run <job.txt>   — одна compute-диспетчеризация по job-файлу (см. job.cpp)
//   geo4d_gpu graph <файл>    — граф диспетчеризаций одним командным буфером (см. graph.cpp)
//   geo4d_gpu denoise <job>   — денойзер OptiX (см. denoise.cpp)
//   geo4d_gpu app <graph>     — интерактивное окно (см. app.cpp, DD-007)
//   geo4d_gpu export-graph <сцена.json> <каталог> [...] — граф окна без Python (см. export_graph.cpp, DD-010)
//   geo4d_gpu [сцена.json] [--res WxH] [...] — окно без Python (см. standalone.cpp, DD-010)
//   geo4d_gpu eval <вход.json> <выход.json> — формулы языка нод: значения, производные, Slang (expr.cpp, DD-011)
//   geo4d_gpu migrate <сцена.json> <сцена.g4d> — сцена version 1 → граф (scene_graph.cpp, DD-011)
#include <cstdio>
#include <exception>
#include <iostream>
#include <string>

#include "export_graph.hpp"
#include "codegen.hpp"
#include "expr.hpp"
#include "scene_graph.hpp"
#include "graph.hpp"
#ifdef GEO4D_APP
#include "app.hpp"
#include "standalone.hpp"
#endif
#ifdef GEO4D_OPTIX
#include "denoise.hpp"
#endif
#include "job.hpp"
#include "vk_context.hpp"

int main(int argc, char** argv) {
    try {
        // библиотека нод (DD-011) — рядом с exe или в дереве разработки; грузится при первом обращении
        try {
            geo4d::set_node_library_dir(geo4d::u8(geo4d::default_dirs(argv[0]).nodes));
        } catch (const std::exception&) {
        }
#ifdef GEO4D_APP
        if (is_standalone_args(argc, argv)) return run_standalone(argc, argv);
#endif
        if (argc < 2) {
            std::fprintf(stderr, "использование: geo4d_gpu info | run <job.txt> | graph <graph.txt> | denoise <job.txt> | app <graph.txt> | export-graph <сцена.json> <каталог>\n");
            return 2;
        }
        std::string cmd = argv[1];
        if (cmd == "export-graph") return run_export_graph(argc, argv);     // без Vulkan
        if (cmd == "eval") return geo4d::run_eval(argc, argv);               // язык выражений (DD-011), без Vulkan
        if (cmd == "migrate") return geo4d::run_migrate(argc, argv);         // сцена version 1 → .g4d (DD-011)
#ifdef GEO4D_OPTIX
        if (cmd == "denoise" && argc >= 3) return run_denoise(argv[2]);     // без Vulkan
#endif
#ifdef GEO4D_APP
        if (cmd == "app" && argc >= 3) return run_app(argv[2]);             // свой контекст с окном
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
