// Интерактивное приложение (DD-007): geo4d_gpu app <graph.txt>
//
// Граф (graph_program.cpp) с секциями move / static / post и директивами приложения:
//   app <W> <H> <буфер RGBA8>       размер кадра и буфер вывода (uint на пиксель)
//   denoise <кадр> <альбедо> <нормаль> <выход>   экспортируемые буферы (обмен с CUDA без копий)
//   campipe <конвейер>               cam_step: push 0..2 — перемещение (f, r, u), 4..6 — рыскание/тангаж/крен
//   presentpipe <конвейер> <i_use_den> <i_bgra>   индексы push-констант pt_present
//   seedpipes <i_seed> <конвейеры...>             зерно ГСЧ меняется каждый кадр
//   speed <ед./с>                    скорость полёта
//   hidden | vsync                   скрытое окно (замеры) / вертикальная синхронизация
//   bench <сценарий> <json>          движение из файла (строка на кадр: a b c ψ θ φ за кадр), журнал времени, выход
//   dump <кадр> <каталог>            сохранить вывод (u32 RGBA8), кадр, выход денойзера и камеру этого кадра (.npy)
// Управление: W/S, A/D — вперёд/вбок; Q/E — вниз/вверх; мышь (ЛКМ — захват) и стрелки — взгляд; Z/C — крен;
//             Shift — ×4; R — денойзер вкл/выкл; Esc — выход.
#include "app.hpp"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "denoiser.hpp"
#include "graph_program.hpp"
#include "npy.hpp"

namespace {

void check(VkResult r, const char* what) {
    if (r != VK_SUCCESS) throw std::runtime_error(std::string(what) + ": VkResult " + std::to_string(r));
}

struct Swapchain {
    VkSwapchainKHR sc = VK_NULL_HANDLE;
    std::vector<VkImage> images;
    VkExtent2D extent{};
    VkFormat format = VK_FORMAT_UNDEFINED;
};

Swapchain make_swapchain(VkContext& ctx, VkSurfaceKHR surf, uint32_t W, uint32_t H, bool vsync) {
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
    ci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
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
    return s;
}

double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

}  // namespace

int run_app(const std::string& graph_path) {
    // директивы приложения — до создания контекста (нужны размер окна и флаги)
    uint32_t W = 0, H = 0;
    bool hidden = false, vsync = false;
    {
        std::ifstream f(graph_path);
        for (std::string line; std::getline(f, line);) {
            std::istringstream ss(line);
            std::string cmd;
            ss >> cmd;
            if (cmd == "app") ss >> W >> H;
            else if (cmd == "hidden") hidden = true;
            else if (cmd == "vsync") vsync = true;
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
        VkSurfaceKHR surf;
        if (!SDL_Vulkan_CreateSurface(win, ctx.instance(), nullptr, &surf))
            throw std::runtime_error(std::string("SDL_Vulkan_CreateSurface: ") + SDL_GetError());
        VkBool32 present_ok = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(ctx.physical(), ctx.queue_family(), surf, &present_ok);
        if (!present_ok) throw std::runtime_error("очередь compute не умеет present");
        Swapchain sc = make_swapchain(ctx, surf, W, H, vsync);

        GraphProgram prog(ctx, graph_path);
        std::string present_buf, campipe, presentpipe, bench_in, bench_out;
        std::vector<std::string> den_bufs, seedpipes;
        std::vector<std::pair<uint32_t, std::string>> dumps;
        size_t i_use_den = 0, i_bgra = 0, i_seed = 1;
        float speed = 1.0f;
        for (auto& w : prog.extra()) {
            if (w[0] == "app") present_buf = w.at(3);
            else if (w[0] == "denoise") den_bufs = {w.at(1), w.at(2), w.at(3), w.at(4)};
            else if (w[0] == "campipe") campipe = w.at(1);
            else if (w[0] == "presentpipe") { presentpipe = w.at(1); i_use_den = std::stoul(w.at(2)); i_bgra = std::stoul(w.at(3)); }
            else if (w[0] == "seedpipes") { i_seed = std::stoul(w.at(1)); seedpipes.assign(w.begin() + 2, w.end()); }
            else if (w[0] == "speed") speed = std::stof(w.at(1));
            else if (w[0] == "bench") { bench_in = w.at(1); bench_out = w.at(2); }
            else if (w[0] == "dump") dumps.emplace_back(uint32_t(std::stoul(w.at(1))), w.at(2));
        }
        prog.set_push_u32(presentpipe, i_bgra, sc.format == VK_FORMAT_B8G8R8A8_UNORM ? 1u : 0u);

        // денойзер на памяти буферов Vulkan (импорт в CUDA)
        std::unique_ptr<OptixDenoiserCtx> den;
        uint64_t dptr[4] = {0, 0, 0, 0};
        if (!den_bufs.empty()) {
            den = std::make_unique<OptixDenoiserCtx>();
            den->setup(W, H, true, true);
            for (int k = 0; k < 4; ++k) {
                Buffer& b = prog.buffer(den_bufs[k]);
                dptr[k] = den->import_win32(b.win32, b.alloc_size, b.size);
            }
        }
        bool use_den = den != nullptr;

        std::vector<std::vector<float>> script;
        if (!bench_in.empty()) {
            std::ifstream f(bench_in);
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

        std::ofstream log;
        if (!bench_out.empty()) { log.open(bench_out); log << "[\n"; }
        bool quit = false, grab = false;
        uint32_t frame = 0, accum = 0;
        auto t_prev = std::chrono::steady_clock::now();
        double fps_acc = 0;
        int fps_n = 0;
        while (!quit) {
            auto t0 = std::chrono::steady_clock::now();
            double dt = std::min(std::chrono::duration<double>(t0 - t_prev).count(), 0.1);
            t_prev = t0;
            float mv[3] = {0, 0, 0}, rot[3] = {0, 0, 0};
            float mdx = 0, mdy = 0;
            SDL_Event e;
            while (SDL_PollEvent(&e)) {
                if (e.type == SDL_EVENT_QUIT) quit = true;
                else if (e.type == SDL_EVENT_KEY_DOWN && e.key.key == SDLK_ESCAPE) quit = true;
                else if (e.type == SDL_EVENT_KEY_DOWN && e.key.key == SDLK_R && den) { use_den = !use_den; accum = 0; }
                else if (e.type == SDL_EVENT_MOUSE_BUTTON_DOWN && e.button.button == SDL_BUTTON_LEFT) {
                    grab = !grab;
                    SDL_SetWindowRelativeMouseMode(win, grab);
                } else if (e.type == SDL_EVENT_MOUSE_MOTION && grab) { mdx += e.motion.xrel; mdy += e.motion.yrel; }
            }
            if (!script.empty()) {
                if (frame >= script.size()) break;
                auto& s = script[frame];
                mv[0] = s[0]; mv[1] = s[1]; mv[2] = s[2]; rot[0] = s[3]; rot[1] = s[4]; rot[2] = s[5];
            } else {
                const bool* k = SDL_GetKeyboardState(nullptr);
                float v = speed * float(dt) * (k[SDL_SCANCODE_LSHIFT] ? 4.0f : 1.0f), av = 1.2f * float(dt);
                mv[0] = v * (float(k[SDL_SCANCODE_W]) - float(k[SDL_SCANCODE_S]));
                mv[1] = v * (float(k[SDL_SCANCODE_D]) - float(k[SDL_SCANCODE_A]));
                mv[2] = v * (float(k[SDL_SCANCODE_E]) - float(k[SDL_SCANCODE_Q]));
                rot[0] = av * (float(k[SDL_SCANCODE_RIGHT]) - float(k[SDL_SCANCODE_LEFT])) + 0.003f * mdx;
                rot[1] = av * (float(k[SDL_SCANCODE_UP]) - float(k[SDL_SCANCODE_DOWN])) - 0.003f * mdy;
                rot[2] = av * (float(k[SDL_SCANCODE_C]) - float(k[SDL_SCANCODE_Z]));
            }
            bool moving = frame == 0 || mv[0] != 0 || mv[1] != 0 || mv[2] != 0 || rot[0] != 0 || rot[1] != 0 || rot[2] != 0;
            for (int i = 0; i < 3; ++i) { prog.set_push_f32(campipe, i, mv[i]); prog.set_push_f32(campipe, 4 + i, rot[i]); }
            for (auto& p : seedpipes) prog.set_push_u32(p, i_seed, 1000u + frame);
            auto tg = std::chrono::steady_clock::now();
            auto marks = prog.run(moving ? "move" : "static");
            double gpu_ms = ms_since(tg);
            accum = moving ? 1 : accum + 1;
            double den_ms = 0;
            if (use_den) den_ms = den->invoke(dptr[0], dptr[1], dptr[2], dptr[3]);
            prog.set_push_u32(presentpipe, i_use_den, use_den ? 1u : 0u);
            auto tp = std::chrono::steady_clock::now();
            prog.run("post");
            // копия в образ swapchain и present
            uint32_t idx;
            VkResult ar = vkAcquireNextImageKHR(ctx.device(), sc.sc, UINT64_MAX, acquired, VK_NULL_HANDLE, &idx);
            if (ar != VK_SUCCESS && ar != VK_SUBOPTIMAL_KHR) check(ar, "vkAcquireNextImageKHR");
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
            vkCmdCopyBufferToImage(cb, prog.buffer(present_buf).buf, sc.images[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &reg);
            ib.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            ib.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            ib.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            ib.dstAccessMask = 0;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &ib);
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
            vkQueuePresentKHR(ctx.queue(), &pi);
            check(vkWaitForFences(ctx.device(), 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
            vkResetFences(ctx.device(), 1, &fence);
            double post_ms = ms_since(tp), total_ms = ms_since(t0);
            for (auto& [fi, dir] : dumps) {
                if (fi != frame) continue;
                auto save = [&](const std::string& name, const std::string& dtype, std::vector<size_t> shape) {
                    Buffer& b = prog.buffer(name);
                    Buffer h = ctx.create_buffer(b.size, true);
                    ctx.copy_buffer(b, h, b.size);
                    npy_write(dir + "/" + name + "_" + std::to_string(frame) + ".npy", dtype, shape, h.mapped, b.size);
                    ctx.destroy_buffer(h);
                };
                save(present_buf, "<u4", {H, W});
                save(den_bufs.empty() ? present_buf : den_bufs[0], "<f4", {H, W, 4});
                if (!den_bufs.empty())
                    for (int k = 1; k < 4; ++k) save(den_bufs[k], "<f4", {H, W, 4});
                save("cam", "<f4", {8, 4});
            }
            if (log.is_open()) {
                log << (frame ? ",\n" : "") << "{\"frame\": " << frame << ", \"moving\": " << moving << ", \"accum\": " << accum
                    << ", \"total_ms\": " << total_ms << ", \"graph_ms\": " << gpu_ms << ", \"denoise_ms\": " << den_ms
                    << ", \"post_present_ms\": " << post_ms;
                for (auto& [k, v] : marks) if (k != "start") log << ", \"" << k << "\": " << v / 1e6;
                log << "}";
            }
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
        vkDeviceWaitIdle(ctx.device());
        vkFreeCommandBuffers(ctx.device(), ctx.command_pool(), 1, &cb);
        vkDestroyFence(ctx.device(), fence, nullptr);
        vkDestroySemaphore(ctx.device(), acquired, nullptr);
        vkDestroySemaphore(ctx.device(), rendered, nullptr);
        den.reset();
        vkDestroySwapchainKHR(ctx.device(), sc.sc, nullptr);
        SDL_Vulkan_DestroySurface(ctx.instance(), surf, nullptr);
    }
    SDL_DestroyWindow(win);
    SDL_Quit();
    return rc;
}
