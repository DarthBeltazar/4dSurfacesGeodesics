// Интерактивное приложение (DD-007, редактор сцен — DD-008): geo4d_gpu app <graph.txt>
//
// Граф (graph_program.cpp) с секциями move / edit / static / post / snap и директивами приложения:
//   app <W> <H> <буфер RGBA8>       размер кадра и буфер вывода (uint на пиксель)
//   denoise <кадр> <альбедо> <нормаль> <выход>   экспортируемые буферы (обмен с CUDA без копий); без OptiX/CUDA
//                                    (сборка без GEO4D_OPTIX, не-NVIDIA GPU, macOS) кадр идёт без денойзера (ADR-0008)
//   campipe <конвейер>               cam_step: push 0..2 — перемещение (f, r, u), 4..6 — рыскание/тангаж/крен
//   presentpipe <конвейер> <i_use_den> <i_bgra> [<i_exposure>]   индексы push-констант pt_present
//   seedpipes <i_seed> <конвейеры...>             зерно ГСЧ: при движении — по номеру кадра, иначе — по номеру сэмпла
//                                    накопления (правка объектов = холодный старт той же сцены, побитово)
//   speed <ед./с>                    скорость полёта
//   objrows <буфер> <строка>         строки объектов (DD-008) в буфере камеры; при запуске хост распаковывает и
//                                    упаковывает их заново и печатает «rows_check <макс. расхождение>» (C++ = Python)
//   editor / manifold / flat / objname / rebuild — редактор сцен (editor.hpp)
//   hidden | vsync                   скрытое окно (замеры) / вертикальная синхронизация
//   ui                               оверлей и в режиме замера/дампа (снимок окна с панелями — проверка оверлея)
//   bench <сценарий> <json>          движение из файла (строка на кадр: a b c ψ θ φ за кадр), журнал времени, выход
//   edit <кадр> <строки.npy>         в сценарии замера: заменить строки объектов перед кадром (как правка в окне)
//   remanifold <кадр> <вид> k=v…     в сценарии: смена многообразия как кнопкой «Применить» (ждёт Python, перезагрузка)
//   editgraph <граф>                 граф режима редактирования (шейдеры без зашитой структуры объектов, DD-008)
//   editmode <кадр> on|off           в сценарии: войти в режим редактирования / выйти (ждёт перегенерацию)
//   record <файл.raw> <период>       дописывать вывод (BGRA8/RGBA8, W·H·4 байт) каждого период-го кадра — для ролика
//   dump <кадр> <каталог>            сохранить вывод (u32 RGBA8), кадр, выход денойзера и камеру этого кадра (.npy)
// Управление: W/S, A/D — вперёд/вбок; Q/E — вниз/вверх; мышь (ЛКМ — захват) и стрелки — взгляд; Z/C — крен;
//             Shift — ×4; R — денойзер вкл/выкл; F1 — панели редактора; Esc — выход.
// Оверлей ImGui — только в интерактивном режиме (не hidden/bench/dump: там вывод сверяется с образом swapchain),
// если нет директивы ui.
#include "app.hpp"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef GEO4D_OPTIX
#include "denoiser.hpp"
#endif
#include "editor.hpp"
#include "graph_program.hpp"
#include "npy.hpp"

namespace {

void check(VkResult r, const char* what) {
    if (r != VK_SUCCESS) throw std::runtime_error(std::string(what) + ": VkResult " + std::to_string(r));
}

struct Swapchain {
    VkSwapchainKHR sc = VK_NULL_HANDLE;
    std::vector<VkImage> images;
    std::vector<VkImageView> views;              // для оверлея (COLOR_ATTACHMENT)
    VkExtent2D extent{};
    VkFormat format = VK_FORMAT_UNDEFINED;
};

Swapchain make_swapchain(VkContext& ctx, VkSurfaceKHR surf, uint32_t W, uint32_t H, bool vsync, bool overlay) {
    Swapchain s;
    VkSurfaceCapabilitiesKHR caps;
    check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(ctx.physical(), surf, &caps), "SurfaceCapabilities");
    uint32_t nf = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(ctx.physical(), surf, &nf, nullptr);
    std::vector<VkSurfaceFormatKHR> fmts(nf);
    vkGetPhysicalDeviceSurfaceFormatsKHR(ctx.physical(), surf, &nf, fmts.data());
    VkSurfaceFormatKHR fmt = fmts[0];
    for (auto& f : fmts)
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_R8G8B8A8_UNORM) { fmt = f; break; }
    uint32_t npm = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(ctx.physical(), surf, &npm, nullptr);
    std::vector<VkPresentModeKHR> pms(npm);
    vkGetPhysicalDeviceSurfacePresentModesKHR(ctx.physical(), surf, &npm, pms.data());
    VkPresentModeKHR pm = VK_PRESENT_MODE_FIFO_KHR;
    if (!vsync)
        for (auto m : pms)
            if (m == VK_PRESENT_MODE_IMMEDIATE_KHR || m == VK_PRESENT_MODE_MAILBOX_KHR) { pm = m; break; }
    s.extent = caps.currentExtent.width != UINT32_MAX ? caps.currentExtent : VkExtent2D{W, H};
    if (!(caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) throw std::runtime_error("swapchain без TRANSFER_DST");
    VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    ci.surface = surf;
    ci.minImageCount = std::max(caps.minImageCount, 2u);
    if (caps.maxImageCount) ci.minImageCount = std::min(ci.minImageCount, caps.maxImageCount);
    ci.imageFormat = fmt.format;
    ci.imageColorSpace = fmt.colorSpace;
    ci.imageExtent = s.extent;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | (caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) |
                    (overlay ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT : 0);
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode = pm;
    ci.clipped = VK_TRUE;
    check(vkCreateSwapchainKHR(ctx.device(), &ci, nullptr, &s.sc), "vkCreateSwapchainKHR");
    uint32_t ni = 0;
    vkGetSwapchainImagesKHR(ctx.device(), s.sc, &ni, nullptr);
    s.images.resize(ni);
    vkGetSwapchainImagesKHR(ctx.device(), s.sc, &ni, s.images.data());
    s.format = fmt.format;
    if (overlay)
        for (auto img : s.images) {
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vi.image = img;
            vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vi.format = s.format;
            vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VkImageView v;
            check(vkCreateImageView(ctx.device(), &vi, nullptr, &v), "vkCreateImageView");
            s.views.push_back(v);
        }
    return s;
}

void destroy_swapchain(VkContext& ctx, Swapchain& s) {
    for (auto v : s.views) vkDestroyImageView(ctx.device(), v, nullptr);
    vkDestroySwapchainKHR(ctx.device(), s.sc, nullptr);
    s = Swapchain{};
}

double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

// директивы приложения из загруженного графа
struct AppDirs {
    std::string present_buf, campipe, presentpipe, bench_in, bench_out, record_path, objbuf;
    std::vector<std::string> den_bufs, seedpipes;
    std::vector<std::pair<uint32_t, std::string>> dumps, edits;
    std::vector<std::pair<uint32_t, std::vector<std::string>>> remanifold;     // кадр, вид, k=v…
    std::vector<std::pair<uint32_t, bool>> editmode;                            // кадр, вкл/выкл
    std::string editgraph;
    uint32_t record_period = 1;
    size_t i_use_den = 0, i_bgra = 0, i_seed = 1, i_exposure = SIZE_MAX;
    float speed = 1.0f;
    int obj_base = -1;
};

AppDirs parse_dirs(const GraphProgram& prog) {
    AppDirs d;
    for (auto& w : prog.extra()) {
        if (w[0] == "app") d.present_buf = w.at(3);
        else if (w[0] == "denoise") d.den_bufs = {w.at(1), w.at(2), w.at(3), w.at(4)};
        else if (w[0] == "campipe") d.campipe = w.at(1);
        else if (w[0] == "presentpipe") {
            d.presentpipe = w.at(1);
            d.i_use_den = std::stoul(w.at(2));
            d.i_bgra = std::stoul(w.at(3));
            if (w.size() > 4) d.i_exposure = std::stoul(w[4]);
        } else if (w[0] == "seedpipes") { d.i_seed = std::stoul(w.at(1)); d.seedpipes.assign(w.begin() + 2, w.end()); }
        else if (w[0] == "speed") d.speed = std::stof(w.at(1));
        else if (w[0] == "bench") { d.bench_in = w.at(1); d.bench_out = w.at(2); }
        else if (w[0] == "dump") d.dumps.emplace_back(uint32_t(std::stoul(w.at(1))), w.at(2));
        else if (w[0] == "edit") d.edits.emplace_back(uint32_t(std::stoul(w.at(1))), w.at(2));
        else if (w[0] == "record") { d.record_path = w.at(1); d.record_period = uint32_t(std::stoul(w.at(2))); }
        else if (w[0] == "objrows") { d.objbuf = w.at(1); d.obj_base = std::stoi(w.at(2)); }
        else if (w[0] == "editgraph") d.editgraph = w.at(1);
        else if (w[0] == "editmode") d.editmode.emplace_back(uint32_t(std::stoul(w.at(1))), w.at(2) == "on");
        else if (w[0] == "remanifold") d.remanifold.emplace_back(uint32_t(std::stoul(w.at(1))),
                                                                 std::vector<std::string>(w.begin() + 2, w.end()));
    }
    return d;
}

const char* kFonts[] = {"C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
                        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans.ttf",
                        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"};

}  // namespace

int run_app(const std::string& graph_path) {
    // директивы приложения — до создания контекста (нужны размер окна и флаги)
    uint32_t W = 0, H = 0;
    bool hidden = false, vsync = false, batch = false, editor_on = false, ui_forced = false;
    {
        std::ifstream f(graph_path);
        for (std::string line; std::getline(f, line);) {
            std::istringstream ss(line);
            std::string cmd;
            ss >> cmd;
            if (cmd == "app") ss >> W >> H;
            else if (cmd == "hidden") hidden = true;
            else if (cmd == "vsync") vsync = true;
            else if (cmd == "bench" || cmd == "dump" || cmd == "record") batch = true;
            else if (cmd == "editor") editor_on = true;
            else if (cmd == "ui") ui_forced = true;
        }
    }
    if (!W || !H) throw std::runtime_error("app: нет директивы app <W> <H>");
    if (!SDL_Init(SDL_INIT_VIDEO)) throw std::runtime_error(std::string("SDL_Init: ") + SDL_GetError());
    SDL_Window* win = SDL_CreateWindow("geo4d", int(W), int(H), SDL_WINDOW_VULKAN | (hidden ? SDL_WINDOW_HIDDEN : 0));
    if (!win) throw std::runtime_error(std::string("SDL_CreateWindow: ") + SDL_GetError());
    Uint32 next = 0;
    const char* const* se = SDL_Vulkan_GetInstanceExtensions(&next);
    std::vector<const char*> iexts(se, se + next);
    int rc = 0;
    {
        VkContext ctx(iexts, true);
        const bool overlay = editor_on && (ui_forced || (!hidden && !batch)) && ctx.dynamic_rendering();
        VkSurfaceKHR surf;
        if (!SDL_Vulkan_CreateSurface(win, ctx.instance(), nullptr, &surf))
            throw std::runtime_error(std::string("SDL_Vulkan_CreateSurface: ") + SDL_GetError());
        VkBool32 present_ok = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(ctx.physical(), ctx.queue_family(), surf, &present_ok);
        if (!present_ok) throw std::runtime_error("очередь compute не умеет present");
        Swapchain sc = make_swapchain(ctx, surf, W, H, vsync, overlay);
        auto recreate = [&]() {
            vkDeviceWaitIdle(ctx.device());
            destroy_swapchain(ctx, sc);
            sc = make_swapchain(ctx, surf, W, H, vsync, overlay);
        };

        // программа графа и денойзер на памяти её буферов; перезагружаются при смене многообразия (DD-008)
        std::unique_ptr<GraphProgram> prog;
        AppDirs d;
#ifdef GEO4D_OPTIX
        std::unique_ptr<OptixDenoiserCtx> den;
        uint64_t dptr[4] = {0, 0, 0, 0};
#endif
        bool has_den = false;
        Buffer rows_host = ctx.create_buffer(size_t(kObjRows) * 16, true);
        Buffer cam_host = ctx.create_buffer(64, true);
        auto read_rows = [&]() -> const float* {
            ctx.copy_buffer(prog->buffer(d.objbuf), rows_host, rows_host.size, size_t(d.obj_base) * 16, 0);
            return static_cast<const float*>(rows_host.mapped);
        };
        auto write_rows = [&](const float* rows) {
            std::memcpy(rows_host.mapped, rows, rows_host.size);
            ctx.copy_buffer(rows_host, prog->buffer(d.objbuf), rows_host.size, 0, size_t(d.obj_base) * 16);
        };
        auto load = [&](const std::string& path) {
#ifdef GEO4D_OPTIX
            den.reset();                               // импорт в CUDA — до освобождения памяти буферов
#endif
            prog.reset();
            prog = std::make_unique<GraphProgram>(ctx, path);
            AppDirs old = d;
            d = parse_dirs(*prog);
            if (!old.present_buf.empty()) {            // перезагрузка: сценарий, дампы и запись — из исходного графа
                d.bench_in = old.bench_in; d.bench_out = old.bench_out;
                d.dumps = old.dumps; d.edits = old.edits; d.remanifold = old.remanifold; d.editmode = old.editmode;
                d.record_path = old.record_path; d.record_period = old.record_period;
            }
            prog->set_push_u32(d.presentpipe, d.i_bgra, sc.format == VK_FORMAT_B8G8R8A8_UNORM ? 1u : 0u);
            has_den = false;
#ifdef GEO4D_OPTIX
            if (!d.den_bufs.empty()) {
                try {
                    den = std::make_unique<OptixDenoiserCtx>();
                    den->setup(W, H, true, true);
                    for (int k = 0; k < 4; ++k) {
                        Buffer& b = prog->buffer(d.den_bufs[k]);
                        if (b.ext_handle == -1) throw std::runtime_error("буфер " + d.den_bufs[k] + " без экспорта памяти");
                        dptr[k] = den->import_external(b.ext_handle, b.alloc_size, b.size);
#ifndef _WIN32
                        b.ext_handle = -1;                 // fd теперь принадлежит CUDA
#endif
                    }
                    has_den = true;
                } catch (const std::exception& e) {
                    std::fprintf(stderr, "денойзер недоступен (%s) — вывод без денойзера\n", e.what());
                    den.reset();
                }
            }
#endif
            // строки объектов: C++ (editor.cpp) = Python (gpu_pt.object_rows) — распаковка и упаковка заново
            if (d.obj_base >= 0) {
                const float* R = read_rows();
                auto again = pack_rows(unpack_rows(R, {}), parse_editor_config(prog->extra()).flat);
                double md = 0;
                for (size_t i = 0; i < again.size(); ++i)
                    md = std::max(md, double(std::fabs(again[i] - R[i])) / std::max(1.0, double(std::fabs(R[i]))));
                std::printf("rows_check %.3g\n", md);
                std::fflush(stdout);
            }
        };
        load(graph_path);
        bool use_den = has_den;
        float exposure = 1.0f, speed = d.speed;

        // редактор и ImGui
        std::unique_ptr<Editor> editor;
        bool show_ui = true;
        VkFormat ui_format = sc.format;                // ImGui хранит указатель до создания конвейера
        // быстрый граф (структура объектов зашита) и граф режима редактирования (DD-008)
        std::string baked_path = graph_path, dyn_path = d.editgraph;
        std::vector<std::pair<int, int>> baked_struct;
        bool edit_mode = false, pending_exit = false, force_edit = false;
        if (editor_on) {
            editor = std::make_unique<Editor>(parse_editor_config(prog->extra()));
            editor->set_config(parse_editor_config(prog->extra()), read_rows());
            baked_struct = structure(editor->objects());
        }
        // смена графа без закрытия окна; keep_cam — камера старого графа побитово, ed_rows — объекты редактора
        auto switch_graph = [&](const std::string& path, bool keep_cam, bool ed_rows) {
            ctx.copy_buffer(prog->buffer("cam"), cam_host, 64);
            std::vector<uint8_t> camb(static_cast<uint8_t*>(cam_host.mapped), static_cast<uint8_t*>(cam_host.mapped) + 64);
            auto flat = editor->flat();
            vkDeviceWaitIdle(ctx.device());
            load(path);
            if (ed_rows) write_rows(pack_rows(editor->objects(), flat).data());
            if (keep_cam) {
                std::memcpy(cam_host.mapped, camb.data(), 64);
                ctx.copy_buffer(cam_host, prog->buffer("cam"), 64);
            }
            editor->set_config(parse_editor_config(prog->extra()), read_rows(), ed_rows);
            use_den = use_den && has_den;
            force_edit = true;
        };
        auto enter_edit = [&]() {
            if (edit_mode || dyn_path.empty()) return;
            switch_graph(dyn_path, true, true);
            edit_mode = true;
        };
        auto exit_edit = [&](const CameraState& cam, bool wait) {
            if (!edit_mode || editor->rebuilding()) return;
            if (structure(editor->objects()) == baked_struct) {      // структура прежняя — быстрый граф уже есть
                switch_graph(baked_path, true, true);
                edit_mode = false;
            } else {
                pending_exit = true;
                editor->rebuild_structure(cam, wait);
            }
        };
        if (overlay) {
            IMGUI_CHECKVERSION();
            ImGui::CreateContext();
            ImGuiIO& io = ImGui::GetIO();
            io.IniFilename = nullptr;
            float scale = SDL_GetWindowDisplayScale(win);
            if (scale <= 0) scale = 1;
            // латиница, греческий (ρ), кириллица, пунктуация (—, …), индексы (ρ₀)
            static const ImWchar ranges[] = {0x0020, 0x00FF, 0x0370, 0x03FF, 0x0400, 0x052F, 0x2000, 0x206F,
                                             0x2070, 0x209F, 0x2190, 0x21FF, 0x2200, 0x22FF, 0};
            for (const char* fp : kFonts)
                if (std::filesystem::exists(fp)) {
                    io.Fonts->AddFontFromFileTTF(fp, 16.0f * scale, nullptr, ranges);
                    break;
                }
            ImGui::StyleColorsDark();
            ImGui::GetStyle().ScaleAllSizes(scale);
            ImGui::GetStyle().Alpha = 0.94f;
            ImGui_ImplSDL3_InitForVulkan(win);
            ImGui_ImplVulkan_InitInfo ii{};
            ii.ApiVersion = VK_API_VERSION_1_3;
            ii.Instance = ctx.instance();
            ii.PhysicalDevice = ctx.physical();
            ii.Device = ctx.device();
            ii.QueueFamily = ctx.queue_family();
            ii.Queue = ctx.queue();
            ii.DescriptorPoolSize = 16;
            ii.MinImageCount = 2;
            ii.ImageCount = uint32_t(std::max<size_t>(sc.images.size(), 2));
            ii.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
            ii.UseDynamicRendering = true;
            ii.PipelineRenderingCreateInfo = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
            ii.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
            ii.PipelineRenderingCreateInfo.pColorAttachmentFormats = &ui_format;
            if (!ImGui_ImplVulkan_Init(&ii)) throw std::runtime_error("ImGui_ImplVulkan_Init");
        } else if (editor_on && !hidden && !batch) {
            std::fprintf(stderr, "нет dynamicRendering — окно без редактора\n");
        }

        std::vector<std::vector<float>> script;
        if (!d.bench_in.empty()) {
            std::ifstream f(d.bench_in);
            for (std::string line; std::getline(f, line);) {
                std::istringstream ss(line);
                std::vector<float> v(6, 0.0f);
                for (auto& x : v) ss >> x;
                script.push_back(v);
            }
        }
        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VkSemaphore acquired, rendered;
        check(vkCreateSemaphore(ctx.device(), &sci, nullptr, &acquired), "vkCreateSemaphore");
        check(vkCreateSemaphore(ctx.device(), &sci, nullptr, &rendered), "vkCreateSemaphore");
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence fence;
        check(vkCreateFence(ctx.device(), &fci, nullptr, &fence), "vkCreateFence");
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = ctx.command_pool();
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        VkCommandBuffer cb;
        check(vkAllocateCommandBuffers(ctx.device(), &cai, &cb), "vkAllocateCommandBuffers");

        std::ofstream rec;
        Buffer rec_host{};
        if (!d.record_path.empty()) {
            rec.open(d.record_path, std::ios::binary);
            rec_host = ctx.create_buffer(size_t(W) * H * 4, true);
        }
        std::ofstream log;
        if (!d.bench_out.empty()) { log.open(d.bench_out); log << "[\n"; }
        bool quit = false, grab = false, minimized = false, toggle_req = false;
        uint32_t frame = 0, accum = 0;
        auto t_prev = std::chrono::steady_clock::now();
        double fps_acc = 0, last_ms = 0;
        int fps_n = 0;
        CameraState camst{};
        while (!quit) {
            auto t0 = std::chrono::steady_clock::now();
            double dt = std::min(std::chrono::duration<double>(t0 - t_prev).count(), 0.1);
            t_prev = t0;
            float mv[3] = {0, 0, 0}, rot[3] = {0, 0, 0};
            float mdx = 0, mdy = 0;
            const bool ui_kb = overlay && show_ui && ImGui::GetIO().WantCaptureKeyboard;
            const bool ui_mouse = overlay && show_ui && ImGui::GetIO().WantCaptureMouse;
            SDL_Event e;
            while (SDL_PollEvent(&e)) {
                if (overlay && show_ui && !grab) ImGui_ImplSDL3_ProcessEvent(&e);
                if (e.type == SDL_EVENT_QUIT) quit = true;
                else if (e.type == SDL_EVENT_KEY_DOWN && e.key.key == SDLK_ESCAPE) {
                    if (grab) { grab = false; SDL_SetWindowRelativeMouseMode(win, false); }
                    else quit = true;
                } else if (e.type == SDL_EVENT_KEY_DOWN && e.key.key == SDLK_F1 && overlay) show_ui = !show_ui;
                else if (e.type == SDL_EVENT_KEY_DOWN && e.key.key == SDLK_TAB && overlay && !ui_kb) toggle_req = true;
                else if (e.type == SDL_EVENT_KEY_DOWN && e.key.key == SDLK_R && has_den && !ui_kb) { use_den = !use_den; accum = 0; }
                else if (e.type == SDL_EVENT_MOUSE_BUTTON_DOWN && e.button.button == SDL_BUTTON_LEFT && (grab || !ui_mouse)) {
                    grab = !grab;
                    SDL_SetWindowRelativeMouseMode(win, grab);
                } else if (e.type == SDL_EVENT_MOUSE_MOTION && grab) { mdx += e.motion.xrel; mdy += e.motion.yrel; }
                else if (e.type == SDL_EVENT_WINDOW_MINIMIZED) minimized = true;
                else if (e.type == SDL_EVENT_WINDOW_RESTORED) { minimized = false; recreate(); }
            }
            if (minimized && script.empty()) { SDL_Delay(20); continue; }   // свёрнуто: extent 0×0 — не рисуем
            if (!script.empty()) {
                if (frame >= script.size()) break;
                auto& s = script[frame];
                mv[0] = s[0]; mv[1] = s[1]; mv[2] = s[2]; rot[0] = s[3]; rot[1] = s[4]; rot[2] = s[5];
            } else if (!ui_kb) {
                const bool* k = SDL_GetKeyboardState(nullptr);
                float v = speed * float(dt) * (k[SDL_SCANCODE_LSHIFT] ? 4.0f : 1.0f), av = 1.2f * float(dt);
                mv[0] = v * (float(k[SDL_SCANCODE_W]) - float(k[SDL_SCANCODE_S]));
                mv[1] = v * (float(k[SDL_SCANCODE_D]) - float(k[SDL_SCANCODE_A]));
                mv[2] = v * (float(k[SDL_SCANCODE_E]) - float(k[SDL_SCANCODE_Q]));
                rot[0] = av * (float(k[SDL_SCANCODE_RIGHT]) - float(k[SDL_SCANCODE_LEFT])) + 0.003f * mdx;
                rot[1] = av * (float(k[SDL_SCANCODE_UP]) - float(k[SDL_SCANCODE_DOWN])) - 0.003f * mdy;
                rot[2] = av * (float(k[SDL_SCANCODE_C]) - float(k[SDL_SCANCODE_Z]));
            } else {
                rot[0] = 0.003f * mdx;
                rot[1] = -0.003f * mdy;
            }
            bool moved = frame == 0 || mv[0] != 0 || mv[1] != 0 || mv[2] != 0 || rot[0] != 0 || rot[1] != 0 || rot[2] != 0;
            if (editor)
                for (auto& [fi, on] : d.editmode)      // сценарий: режим редактирования
                    if (fi == frame) {
                        if (on) enter_edit();
                        else exit_edit(camst, true);
                    }
            for (auto& [fi, p] : d.edits)              // сценарий замера: правка объектов перед кадром
                if (fi == frame) {
                    NpyArray a = npy_read(p);
                    if (a.data.size() != rows_host.size) throw std::runtime_error("edit: размер строк " + p);
                    const float* rows = reinterpret_cast<const float*>(a.data.data());
                    write_rows(rows);
                    if (editor) {                      // редактор видит правку (имена — по номеру)
                        std::vector<std::string> names;
                        for (auto& o : editor->objects()) names.push_back(o.name);
                        editor->objects() = unpack_rows(rows, names);
                    }
                    force_edit = true;
                }
            // панели редактора (до кадра: правка видна в этом же кадре)
            if (overlay) {
                ImGui_ImplVulkan_NewFrame();
                ImGui_ImplSDL3_NewFrame();
                ImGui::NewFrame();
                if (show_ui) {
                    FrameInfo fi{last_ms, accum, has_den};
                    UiActions act = editor->draw(fi, camst, edit_mode, use_den, exposure, speed);
                    if (act.toggle_edit) toggle_req = true;
                    if (act.structural && !edit_mode) enter_edit();      // форма/число объектов — в режиме редактирования
                    if (act.objects_changed) {
                        write_rows(pack_rows(editor->objects(), editor->flat()).data());
                        force_edit = true;
                        if (act.snap) {                // центры on_M — на M (pt_snap), обратно в редактор
                            prog->run("snap");
                            auto snapped = unpack_rows(read_rows(), {});
                            auto& objs = editor->objects();
                            for (size_t k = 0; k < objs.size() && k < snapped.size(); ++k)
                                std::copy(snapped[k].c, snapped[k].c + 4, objs[k].c);
                            write_rows(pack_rows(objs, editor->flat()).data());     // M34 — по новым центрам
                        }
                    }
                }
                ImGui::Render();
            }
            if (toggle_req && editor) {
                if (edit_mode) exit_edit(camst, false);
                else enter_edit();
            }
            toggle_req = false;
            const bool edited = force_edit;
            force_edit = false;
            // кадр со сбросом накопления: move — шаг камеры, edit — только правка объектов (камера на месте)
            const bool reset = moved || edited;
            const char* section = moved ? "move" : edited ? "edit" : "static";
            for (int i = 0; i < 3; ++i) { prog->set_push_f32(d.campipe, i, mv[i]); prog->set_push_f32(d.campipe, 4 + i, rot[i]); }
            const uint32_t seed = (moved && frame > 0) ? 1000000u + frame : 1000u + (reset ? 0u : accum);
            for (auto& p : d.seedpipes) prog->set_push_u32(p, d.i_seed, seed);
            auto tg = std::chrono::steady_clock::now();
            auto marks = prog->run(section);
            double gpu_ms = ms_since(tg);
            accum = reset ? 1 : accum + 1;
            double den_ms = 0;
#ifdef GEO4D_OPTIX
            if (use_den && den) den_ms = den->invoke(dptr[0], dptr[1], dptr[2], dptr[3]);
#endif
            prog->set_push_u32(d.presentpipe, d.i_use_den, use_den ? 1u : 0u);
            if (d.i_exposure != SIZE_MAX) prog->set_push_f32(d.presentpipe, d.i_exposure, exposure);
            auto tp = std::chrono::steady_clock::now();
            prog->run("post");
            // копия в образ swapchain, оверлей и present
            uint32_t idx;
            VkResult ar = vkAcquireNextImageKHR(ctx.device(), sc.sc, UINT64_MAX, acquired, VK_NULL_HANDLE, &idx);
            if (ar == VK_ERROR_OUT_OF_DATE_KHR) { recreate(); ++frame; continue; }      // кадр не показан
            if (ar != VK_SUCCESS && ar != VK_SUBOPTIMAL_KHR) check(ar, "vkAcquireNextImageKHR");
            bool readback = false;
            for (auto& dm : d.dumps) readback |= dm.first == frame;
            Buffer rb{};
            if (readback) rb = ctx.create_buffer(size_t(sc.extent.width) * sc.extent.height * 4, true);
            vkResetCommandBuffer(cb, 0);
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(cb, &bi);
            VkImageMemoryBarrier ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            ib.image = sc.images[idx];
            ib.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            ib.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            ib.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            ib.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &ib);
            VkBufferImageCopy reg{};
            reg.bufferRowLength = W;
            reg.bufferImageHeight = H;
            reg.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            reg.imageExtent = {std::min(W, sc.extent.width), std::min(H, sc.extent.height), 1};
            vkCmdCopyBufferToImage(cb, prog->buffer(d.present_buf).buf, sc.images[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &reg);
            // образ: TRANSFER_DST → [оверлей ImGui: COLOR_ATTACHMENT] → [чтение: TRANSFER_SRC] → PRESENT_SRC
            VkPipelineStageFlags last_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            ib.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            ib.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            if (overlay && show_ui) {                  // оверлей ImGui поверх кадра (dynamic rendering, LOAD)
                ib.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                ib.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                vkCmdPipelineBarrier(cb, last_stage, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0, nullptr, 1, &ib);
                VkRenderingAttachmentInfo att{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
                att.imageView = sc.views[idx];
                att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                att.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
                att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
                ri.renderArea = {{0, 0}, sc.extent};
                ri.layerCount = 1;
                ri.colorAttachmentCount = 1;
                ri.pColorAttachments = &att;
                vkCmdBeginRendering(cb, &ri);
                ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cb);
                vkCmdEndRendering(cb);
                ib.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                ib.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                last_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            }
            if (readback) {        // проверка пути вывода: образ swapchain обратно в буфер (формат, копия, размер)
                ib.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                ib.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                vkCmdPipelineBarrier(cb, last_stage, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &ib);
                VkBufferImageCopy back{};
                back.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                back.imageExtent = {sc.extent.width, sc.extent.height, 1};
                vkCmdCopyImageToBuffer(cb, sc.images[idx], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb.buf, 1, &back);
                ib.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                ib.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                last_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            }
            ib.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            ib.dstAccessMask = 0;
            vkCmdPipelineBarrier(cb, last_stage, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &ib);
            vkEndCommandBuffer(cb);
            VkPipelineStageFlags ws = VK_PIPELINE_STAGE_TRANSFER_BIT;
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.waitSemaphoreCount = 1;
            si.pWaitSemaphores = &acquired;
            si.pWaitDstStageMask = &ws;
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cb;
            si.signalSemaphoreCount = 1;
            si.pSignalSemaphores = &rendered;
            check(vkQueueSubmit(ctx.queue(), 1, &si, fence), "vkQueueSubmit");
            VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
            pi.waitSemaphoreCount = 1;
            pi.pWaitSemaphores = &rendered;
            pi.swapchainCount = 1;
            pi.pSwapchains = &sc.sc;
            pi.pImageIndices = &idx;
            VkResult pr = vkQueuePresentKHR(ctx.queue(), &pi);
            check(vkWaitForFences(ctx.device(), 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
            vkResetFences(ctx.device(), 1, &fence);
            if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR) recreate();
            if (readback) {
                for (auto& [fi, dir] : d.dumps)
                    if (fi == frame)
                        npy_write(dir + "/swapchain_" + std::to_string(frame) + ".npy", "<u4",
                                  {sc.extent.height, sc.extent.width}, rb.mapped, rb.size);
                ctx.destroy_buffer(rb);
            }
            double post_ms = ms_since(tp), total_ms = ms_since(t0);
            if (rec.is_open() && frame % d.record_period == d.record_period - 1) {
                ctx.copy_buffer(prog->buffer(d.present_buf), rec_host, size_t(W) * H * 4);
                rec.write(static_cast<const char*>(rec_host.mapped), std::streamsize(size_t(W) * H * 4));
            }
            for (auto& [fi, dir] : d.dumps) {
                if (fi != frame) continue;
                auto save = [&](const std::string& name, const std::string& dtype, std::vector<size_t> shape) {
                    Buffer& b = prog->buffer(name);
                    Buffer h = ctx.create_buffer(b.size, true);
                    ctx.copy_buffer(b, h, b.size);
                    npy_write(dir + "/" + name + "_" + std::to_string(frame) + ".npy", dtype, shape, h.mapped, b.size);
                    ctx.destroy_buffer(h);
                };
                save(d.present_buf, "<u4", {H, W});
                save(d.den_bufs.empty() ? d.present_buf : d.den_bufs[0], "<f4", {H, W, 4});
                if (!d.den_bufs.empty())
                    for (int k = 1; k < 4; ++k) save(d.den_bufs[k], "<f4", {H, W, 4});
                save("cam", "<f4", {prog->buffer("cam").size / 16, 4});    // камера (8 строк) + объекты (DD-008)
            }
            if (log.is_open()) {
                log << (frame ? ",\n" : "") << "{\"frame\": " << frame << ", \"moving\": " << reset << ", \"accum\": " << accum
                    << ", \"extent\": [" << sc.extent.width << ", " << sc.extent.height << "], \"format\": " << int(sc.format)
                    << ", \"total_ms\": " << total_ms << ", \"graph_ms\": " << gpu_ms << ", \"denoise_ms\": " << den_ms
                    << ", \"post_present_ms\": " << post_ms;
                for (auto& [k, v] : marks) if (k != "start") log << ", \"" << k << "\": " << v / 1e6;
                log << "}";
            }
            if (editor) {                              // камера для панелей (положение, «перед камерой», сохранение)
                ctx.copy_buffer(prog->buffer("cam"), cam_host, 64);
                std::memcpy(&camst, cam_host.mapped, sizeof(camst));
                for (auto& [fi, w] : d.remanifold)
                    if (fi == frame) {
                        std::vector<ManifoldParam> ps;
                        for (size_t i = 1; i < w.size(); ++i) {
                            auto eq = w[i].find('=');
                            ManifoldParam p{w[i].substr(0, eq), {}};
                            std::stringstream ss(w[i].substr(eq + 1));
                            for (std::string t; std::getline(ss, t, ',');) p.v.push_back(std::stof(t));
                            ps.push_back(p);
                        }
                        editor->rebuild(w.at(0), ps, camst, true);
                    }
                std::string next_graph = editor->poll_rebuild();
                if (next_graph.empty() && !editor->rebuilding() && !d.remanifold.empty() && !editor->status().empty() &&
                    editor->status().rfind("ошибка", 0) == 0) {
                    std::fprintf(stderr, "%s\n", editor->status().c_str());
                    throw std::runtime_error("remanifold: перегенерация не удалась");
                }
                if (pending_exit && next_graph.empty() && !editor->rebuilding()) pending_exit = false;   // ошибка Python
                if (!next_graph.empty()) {             // перезагрузка графа без закрытия окна
                    const std::string dir = next_graph.substr(0, next_graph.rfind('/'));
                    const std::string next_dyn = dir + "/app_graph_dyn.txt";
                    if (editor->last_rebuild_structural()) {      // быстрые шейдеры под новую структуру объектов
                        baked_path = next_graph;
                        dyn_path = next_dyn;
                        baked_struct = editor->requested_structure();
                        if (pending_exit) {
                            if (structure(editor->objects()) == baked_struct) {
                                switch_graph(baked_path, true, true);
                                edit_mode = false;
                                pending_exit = false;
                            } else {
                                editor->rebuild_structure(camst, false);   // структура менялась во время перегенерации
                            }
                        }
                    } else {                                       // новое многообразие: камера и объекты — из Python
                        baked_path = next_graph;
                        dyn_path = next_dyn;
                        switch_graph(edit_mode ? dyn_path : baked_path, false, false);
                        baked_struct = structure(editor->objects());
                        pending_exit = false;
                    }
                }
            }
            last_ms = total_ms;
            fps_acc += total_ms;
            if (++fps_n == 20) {
                char title[256];
                std::snprintf(title, sizeof(title), "geo4d — %.1f мс (%.0f к/с), накоплено %u spp, денойзер %s", fps_acc / fps_n,
                              1000.0 * fps_n / fps_acc, accum, use_den ? "вкл" : "выкл");
                SDL_SetWindowTitle(win, title);
                fps_acc = 0;
                fps_n = 0;
            }
            ++frame;
        }
        if (log.is_open()) log << "\n]\n";
        if (rec.is_open()) { rec.close(); ctx.destroy_buffer(rec_host); }
        vkDeviceWaitIdle(ctx.device());
        if (overlay) {
            ImGui_ImplVulkan_Shutdown();
            ImGui_ImplSDL3_Shutdown();
            ImGui::DestroyContext();
        }
        editor.reset();
        ctx.destroy_buffer(rows_host);
        ctx.destroy_buffer(cam_host);
        vkFreeCommandBuffers(ctx.device(), ctx.command_pool(), 1, &cb);
        vkDestroyFence(ctx.device(), fence, nullptr);
        vkDestroySemaphore(ctx.device(), acquired, nullptr);
        vkDestroySemaphore(ctx.device(), rendered, nullptr);
#ifdef GEO4D_OPTIX
        den.reset();
#endif
        prog.reset();
        destroy_swapchain(ctx, sc);
        SDL_Vulkan_DestroySurface(ctx.instance(), surf, nullptr);
    }
    SDL_DestroyWindow(win);
    SDL_Quit();
    return rc;
}
