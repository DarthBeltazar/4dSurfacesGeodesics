#define VOLK_IMPLEMENTATION
#include <volk.h>
#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vk_mem_alloc.h>

#include "vk_context.hpp"

#ifndef _WIN32
#include <unistd.h>
#endif

#include <chrono>
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>

namespace {

void check(VkResult r, const char* what) {
    if (r != VK_SUCCESS) throw std::runtime_error(std::string(what) + ": VkResult " + std::to_string(r));
}

constexpr uint32_t kMaxTimestamps = 512;

// внешняя память для обмена с CUDA (ADR-0008): Win32 HANDLE или POSIX fd; на macOS — нет
#if defined(_WIN32)
constexpr const char* kExtMemExt = VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME;
constexpr VkExternalMemoryHandleTypeFlagBits kExtMemType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#elif defined(__linux__)
constexpr const char* kExtMemExt = VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME;
constexpr VkExternalMemoryHandleTypeFlagBits kExtMemType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#else
constexpr const char* kExtMemExt = nullptr;
constexpr VkExternalMemoryHandleTypeFlagBits kExtMemType = VkExternalMemoryHandleTypeFlagBits(0);
#endif

bool has_ext(const std::vector<VkExtensionProperties>& exts, const char* name) {
    for (auto& e : exts)
        if (std::strcmp(e.extensionName, name) == 0) return true;
    return false;
}

}  // namespace

VkContext::VkContext(const std::vector<const char*>& instance_exts, bool swapchain) {
    check(volkInitialize(), "volkInitialize");

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "geo4d_gpu";
    app.apiVersion = VK_API_VERSION_1_3;
    // MoltenVK (macOS) виден только с VK_KHR_portability_enumeration
    uint32_t nie = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &nie, nullptr);
    std::vector<VkExtensionProperties> iext(nie);
    vkEnumerateInstanceExtensionProperties(nullptr, &nie, iext.data());
    std::vector<const char*> iexts = instance_exts;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    if (has_ext(iext, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
        bool listed = false;                     // SDL на macOS может уже вернуть его в списке расширений окна
        for (auto* e : iexts) listed |= std::strcmp(e, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == 0;
        if (!listed) iexts.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        ici.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = static_cast<uint32_t>(iexts.size());
    ici.ppEnabledExtensionNames = iexts.data();
    check(vkCreateInstance(&ici, nullptr, &instance_), "vkCreateInstance");
    volkLoadInstance(instance_);

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(instance_, &n, nullptr);
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(instance_, &n, devs.data());
    for (auto d : devs) {                        // предпочтительно дискретный GPU
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(d, &p);
        if (!phys_ || p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) phys_ = d;
    }
    if (!phys_) throw std::runtime_error("нет устройства Vulkan");
    vkGetPhysicalDeviceProperties(phys_, &props_);
    ts_period_ = props_.limits.timestampPeriod;

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys_, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(phys_, &nq, qf.data());
    qfam_ = UINT32_MAX;
    // окно (DD-008): оверлей ImGui рисуется на той же очереди — нужна и GRAPHICS
    const VkQueueFlags need = VK_QUEUE_COMPUTE_BIT | (swapchain ? VK_QUEUE_GRAPHICS_BIT : 0);
    for (uint32_t i = 0; i < nq; ++i)
        if ((qf[i].queueFlags & need) == need && qf[i].timestampValidBits > 0) { qfam_ = i; break; }
    if (qfam_ == UINT32_MAX) throw std::runtime_error("нет очереди compute с timestamp");

    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(phys_, nullptr, &ne, nullptr);
    std::vector<VkExtensionProperties> exts(ne);
    vkEnumerateDeviceExtensionProperties(phys_, nullptr, &ne, exts.data());
    exec_props_ = has_ext(exts, VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
    ext_mem_ = kExtMemExt && has_ext(exts, kExtMemExt);
    const bool portability = has_ext(exts, "VK_KHR_portability_subset");

    // обязательные возможности — понятная ошибка вместо отказа vkCreateDevice (например, MoltenVK, lavapipe)
    VkPhysicalDeviceVulkan12Features s12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features s13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    s13.pNext = &s12;
    VkPhysicalDeviceFeatures2 s2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    s2.pNext = &s13;
    vkGetPhysicalDeviceFeatures2(phys_, &s2);
    std::string missing;
    if (!s2.features.shaderInt64) missing += " shaderInt64";
    if (!s2.features.samplerAnisotropy) missing += " samplerAnisotropy";
    if (!s2.features.imageCubeArray) missing += " imageCubeArray";
    if (!s2.features.textureCompressionBC) missing += " textureCompressionBC";
    if (!s12.scalarBlockLayout) missing += " scalarBlockLayout";
    if (!s12.hostQueryReset) missing += " hostQueryReset";
    if (!missing.empty())
        throw std::runtime_error(std::string("устройство ") + props_.deviceName + " не поддерживает:" + missing);

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qfam_;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pexec{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
    pexec.pipelineExecutableInfo = VK_TRUE;
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.scalarBlockLayout = VK_TRUE;
    f12.hostQueryReset = VK_TRUE;
    if (exec_props_) f12.pNext = &pexec;
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13.pNext = &f12;
    f13.computeFullSubgroups = s13.computeFullSubgroups;
    f13.subgroupSizeControl = s13.subgroupSizeControl;
    f13.dynamicRendering = swapchain ? s13.dynamicRendering : VK_FALSE;      // оверлей ImGui (DD-008)
    dyn_render_ = f13.dynamicRendering == VK_TRUE;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &f13;
    f2.features.shaderInt64 = VK_TRUE;
    f2.features.samplerAnisotropy = VK_TRUE;
    f2.features.imageCubeArray = VK_TRUE;
    f2.features.textureCompressionBC = VK_TRUE;

    std::vector<const char*> dexts;
    if (exec_props_) dexts.push_back(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
    if (ext_mem_) dexts.push_back(kExtMemExt);
    if (portability) dexts.push_back("VK_KHR_portability_subset");
    if (swapchain) dexts.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &f2;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = static_cast<uint32_t>(dexts.size());
    dci.ppEnabledExtensionNames = dexts.data();
    check(vkCreateDevice(phys_, &dci, nullptr, &device_), "vkCreateDevice");
    volkLoadDevice(device_);
    vkGetDeviceQueue(device_, qfam_, 0, &queue_);

    VmaVulkanFunctions vf{};
    vf.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vf.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
    VmaAllocatorCreateInfo aci{};
    aci.physicalDevice = phys_;
    aci.device = device_;
    aci.instance = instance_;
    aci.vulkanApiVersion = VK_API_VERSION_1_3;
    aci.pVulkanFunctions = &vf;
    check(vmaCreateAllocator(&aci, &vma_), "vmaCreateAllocator");

    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = qfam_;
    check(vkCreateCommandPool(device_, &cpi, nullptr, &pool_), "vkCreateCommandPool");

    VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2048}, {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 128}};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpi.maxSets = 128;
    dpi.poolSizeCount = 2;
    dpi.pPoolSizes = ps;
    check(vkCreateDescriptorPool(device_, &dpi, nullptr, &dpool_), "vkCreateDescriptorPool");

    VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpi.queryCount = kMaxTimestamps;
    check(vkCreateQueryPool(device_, &qpi, nullptr, &qpool_), "vkCreateQueryPool");
}

VkContext::~VkContext() {
    if (device_) {
        vkDeviceWaitIdle(device_);
        if (pcache_) {
            size_t n = 0;
            vkGetPipelineCacheData(device_, pcache_, &n, nullptr);
            std::vector<char> data(n);
            if (n && vkGetPipelineCacheData(device_, pcache_, &n, data.data()) == VK_SUCCESS) {
                std::string tmp = pcache_path_ + ".tmp" + std::to_string(reinterpret_cast<uintptr_t>(this));
                {
                    std::ofstream f(tmp, std::ios::binary);
                    f.write(data.data(), std::streamsize(n));
                }
                std::error_code ec;
                std::filesystem::rename(tmp, pcache_path_, ec);
                if (ec) std::filesystem::remove(tmp, ec);
            }
            vkDestroyPipelineCache(device_, pcache_, nullptr);
        }
        vkDestroyQueryPool(device_, qpool_, nullptr);
        vkDestroyDescriptorPool(device_, dpool_, nullptr);
        vkDestroyCommandPool(device_, pool_, nullptr);
        vmaDestroyAllocator(vma_);
        vkDestroyDevice(device_, nullptr);
    }
    if (instance_) vkDestroyInstance(instance_, nullptr);
}

void VkContext::use_pipeline_cache(const std::string& path) {
    if (pcache_) return;
    pcache_path_ = path;
    std::vector<char> data;
    if (std::ifstream f{path, std::ios::binary | std::ios::ate}) {
        data.resize(size_t(f.tellg()));
        f.seekg(0);
        f.read(data.data(), std::streamsize(data.size()));
    }
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
    VkPipelineCacheCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    ci.initialDataSize = data.size();                // чужой или повреждённый кэш драйвер отбрасывает сам
    ci.pInitialData = data.empty() ? nullptr : data.data();
    if (vkCreatePipelineCache(device_, &ci, nullptr, &pcache_) != VK_SUCCESS) {
        ci.initialDataSize = 0;
        ci.pInitialData = nullptr;
        if (vkCreatePipelineCache(device_, &ci, nullptr, &pcache_) != VK_SUCCESS) pcache_ = VK_NULL_HANDLE;
    }
}

std::string VkContext::device_name() const { return props_.deviceName; }

std::string VkContext::driver_version() const {
    uint32_t v = props_.driverVersion;   // NVIDIA: 10.8.8.6 бит
    if (props_.vendorID == 0x10DE)
        return std::to_string(v >> 22) + "." + std::to_string((v >> 14) & 0xff);
    return std::to_string(v);
}

Buffer VkContext::create_buffer(size_t bytes, bool host_visible) {
    Buffer b;
    b.size = bytes;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = bytes > 0 ? bytes : 4;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
               VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    if (host_visible) {
        ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        ai.preferredFlags = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    } else {
        ai.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
        ai.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    }
    VmaAllocationInfo info{};
    check(vmaCreateBuffer(vma_, &bi, &ai, &b.buf, &b.alloc, &info), "vmaCreateBuffer");
    b.mapped = info.pMappedData;
    return b;
}

void VkContext::destroy_buffer(Buffer& b) {
    if (b.memory) {
        if (b.ext_handle != -1) {          // не передан CUDA (после импорта fd принадлежит CUDA)
#ifdef _WIN32
            CloseHandle(reinterpret_cast<HANDLE>(b.ext_handle));
#else
            close(int(b.ext_handle));
#endif
        }
        vkDestroyBuffer(device_, b.buf, nullptr);
        vkFreeMemory(device_, b.memory, nullptr);
    } else if (b.buf) vmaDestroyBuffer(vma_, b.buf, b.alloc);
    b = Buffer{};
}

Buffer VkContext::create_exportable_buffer(size_t bytes) {
    if (!ext_mem_) throw std::runtime_error("нет расширения внешней памяти Vulkan (обмен с CUDA)");
    Buffer b;
    b.size = bytes;
    VkExternalMemoryBufferCreateInfo ebi{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    ebi.handleTypes = kExtMemType;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.pNext = &ebi;
    bi.size = bytes > 0 ? bytes : 4;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    check(vkCreateBuffer(device_, &bi, nullptr, &b.buf), "vkCreateBuffer (export)");
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device_, b.buf, &req);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys_, &mp);
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((req.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) { type = i; break; }
    if (type == UINT32_MAX) throw std::runtime_error("нет типа памяти для экспорта");
    VkMemoryDedicatedAllocateInfo ded{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    ded.buffer = b.buf;
    VkExportMemoryAllocateInfo exi{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
    exi.pNext = &ded;
    exi.handleTypes = kExtMemType;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.pNext = &exi;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = type;
    check(vkAllocateMemory(device_, &mai, nullptr, &b.memory), "vkAllocateMemory (export)");
    check(vkBindBufferMemory(device_, b.buf, b.memory, 0), "vkBindBufferMemory");
#if defined(_WIN32)
    VkMemoryGetWin32HandleInfoKHR hi{VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
    hi.memory = b.memory;
    hi.handleType = kExtMemType;
    HANDLE h = nullptr;
    check(vkGetMemoryWin32HandleKHR(device_, &hi, &h), "vkGetMemoryWin32HandleKHR");
    b.ext_handle = reinterpret_cast<intptr_t>(h);
#elif defined(__linux__)
    VkMemoryGetFdInfoKHR hi{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
    hi.memory = b.memory;
    hi.handleType = kExtMemType;
    int fd = -1;
    check(vkGetMemoryFdKHR(device_, &hi, &fd), "vkGetMemoryFdKHR");
    b.ext_handle = fd;
#endif
    b.alloc_size = req.size;
    return b;
}

VkCommandBuffer VkContext::begin_cmd() {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = pool_;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cb;
    check(vkAllocateCommandBuffers(device_, &ai, &cb), "vkAllocateCommandBuffers");
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);
    return cb;
}

void VkContext::submit_wait(VkCommandBuffer cb) {
    vkEndCommandBuffer(cb);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    check(vkCreateFence(device_, &fi, nullptr, &fence), "vkCreateFence");
    check(vkQueueSubmit(queue_, 1, &si, fence), "vkQueueSubmit");
    check(vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
    vkDestroyFence(device_, fence, nullptr);
    vkFreeCommandBuffers(device_, pool_, 1, &cb);
}

void VkContext::copy_buffer(const Buffer& src, const Buffer& dst, size_t bytes, size_t src_off, size_t dst_off) {
    if (bytes == 0) return;
    VkCommandBuffer cb = begin_cmd();
    VkBufferCopy c{src_off, dst_off, bytes};
    vkCmdCopyBuffer(cb, src.buf, dst.buf, 1, &c);
    submit_wait(cb);
}

Pipeline VkContext::create_pipeline(const std::vector<uint32_t>& spirv, const std::string& entry,
                                    const std::vector<VkDescriptorType>& types, uint32_t push_bytes) {
    Pipeline p;
    uint32_t nbindings = static_cast<uint32_t>(types.size());
    p.nbindings = nbindings;
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = spirv.size() * 4;
    smi.pCode = spirv.data();
    check(vkCreateShaderModule(device_, &smi, nullptr, &p.module), "vkCreateShaderModule");

    std::vector<VkDescriptorSetLayoutBinding> b(nbindings);
    for (uint32_t i = 0; i < nbindings; ++i) {
        b[i].binding = i;
        b[i].descriptorType = types[i];
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = nbindings;
    dli.pBindings = b.data();
    check(vkCreateDescriptorSetLayout(device_, &dli, nullptr, &p.dsl), "vkCreateDescriptorSetLayout");

    VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &p.dsl;
    pli.pushConstantRangeCount = push_bytes ? 1 : 0;
    pli.pPushConstantRanges = &pr;
    check(vkCreatePipelineLayout(device_, &pli, nullptr, &p.layout), "vkCreatePipelineLayout");

    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = p.module;
    cpi.stage.pName = entry.c_str();
    cpi.layout = p.layout;
    if (exec_props_) cpi.flags = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
    const auto t_pipe = std::chrono::steady_clock::now();
    check(vkCreateComputePipelines(device_, pcache_, 1, &cpi, nullptr, &p.pipe), "vkCreateComputePipelines");
    if (std::getenv("GEO4D_PIPE_TIME"))     // время сборки конвейера драйвером (проба быстрой перекомпиляции)
        std::fprintf(stderr, "pipe_time %s %zu %.3f\n", entry.c_str(), spirv.size() * 4,
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_pipe).count());

    if (exec_props_) {      // регистры, спиллы и пр. — как отдаёт драйвер (DD-004 п. 4)
        VkPipelineInfoKHR pinfo{VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR};
        pinfo.pipeline = p.pipe;
        uint32_t ne = 0;
        vkGetPipelineExecutablePropertiesKHR(device_, &pinfo, &ne, nullptr);
        std::vector<VkPipelineExecutablePropertiesKHR> ep(ne, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR});
        vkGetPipelineExecutablePropertiesKHR(device_, &pinfo, &ne, ep.data());
        for (uint32_t e = 0; e < ne; ++e) {
            VkPipelineExecutableInfoKHR xi{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR};
            xi.pipeline = p.pipe;
            xi.executableIndex = e;
            uint32_t ns = 0;
            vkGetPipelineExecutableStatisticsKHR(device_, &xi, &ns, nullptr);
            std::vector<VkPipelineExecutableStatisticKHR> st(ns, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
            vkGetPipelineExecutableStatisticsKHR(device_, &xi, &ns, st.data());
            for (auto& s : st) {
                std::string v;
                switch (s.format) {
                    case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR: v = s.value.b32 ? "true" : "false"; break;
                    case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR: v = std::to_string(s.value.i64); break;
                    case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR: v = std::to_string(s.value.u64); break;
                    case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR: v = std::to_string(s.value.f64); break;
                    default: v = "?";
                }
                p.stats.emplace_back(std::string(ep[e].name) + "/" + s.name, v);
            }
        }
    }
    return p;
}

void VkContext::destroy_pipeline(Pipeline& p) {
    vkDestroyPipeline(device_, p.pipe, nullptr);
    vkDestroyPipelineLayout(device_, p.layout, nullptr);
    vkDestroyDescriptorSetLayout(device_, p.dsl, nullptr);
    vkDestroyShaderModule(device_, p.module, nullptr);
    p = Pipeline{};
}

std::vector<double> VkContext::dispatch(const Pipeline& p, const std::vector<Binding>& bindings, const void* push,
                                        uint32_t push_bytes, uint32_t gx, uint32_t gy, uint32_t gz, uint32_t repeat) {
    if (2 * repeat > kMaxTimestamps) throw std::runtime_error("слишком много повторов");
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = dpool_;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &p.dsl;
    VkDescriptorSet ds;
    check(vkAllocateDescriptorSets(device_, &ai, &ds), "vkAllocateDescriptorSets");
    std::vector<VkDescriptorBufferInfo> bi(bindings.size());
    std::vector<VkDescriptorImageInfo> ii(bindings.size());
    std::vector<VkWriteDescriptorSet> w(bindings.size());
    for (size_t i = 0; i < bindings.size(); ++i) {
        w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[i].dstSet = ds;
        w[i].dstBinding = static_cast<uint32_t>(i);
        w[i].descriptorCount = 1;
        if (bindings[i].cube) {
            ii[i] = {bindings[i].cube->sampler, bindings[i].cube->view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[i].pImageInfo = &ii[i];
        } else {
            bi[i] = {bindings[i].buffer->buf, 0, VK_WHOLE_SIZE};
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[i].pBufferInfo = &bi[i];
        }
    }
    vkUpdateDescriptorSets(device_, static_cast<uint32_t>(w.size()), w.data(), 0, nullptr);

    vkResetQueryPool(device_, qpool_, 0, 2 * repeat);
    VkCommandBuffer cb = begin_cmd();
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &ds, 0, nullptr);
    if (push_bytes) vkCmdPushConstants(cb, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes, push);
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    for (uint32_t r = 0; r < repeat; ++r) {
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool_, 2 * r);
        vkCmdDispatch(cb, gx, gy, gz);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool_, 2 * r + 1);
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb,
                             0, nullptr, 0, nullptr);
    }
    submit_wait(cb);
    std::vector<uint64_t> ts(2 * repeat);
    check(vkGetQueryPoolResults(device_, qpool_, 0, 2 * repeat, ts.size() * 8, ts.data(), 8,
                                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
          "vkGetQueryPoolResults");
    vkFreeDescriptorSets(device_, dpool_, 1, &ds);
    std::vector<double> out(repeat);
    for (uint32_t r = 0; r < repeat; ++r) out[r] = double(ts[2 * r + 1] - ts[2 * r]) * ts_period_;
    return out;
}

namespace {

uint16_t to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000, mant = x & 0x7fffff;
    int32_t exp = int32_t((x >> 23) & 0xff) - 127 + 15;
    if (exp <= 0) {                                   // субнормали / ноль
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant |= 0x800000;
        uint32_t shift = static_cast<uint32_t>(14 - exp);
        uint32_t h = mant >> shift;
        if ((mant >> (shift - 1)) & 1) h += 1;        // округление
        return static_cast<uint16_t>(sign | h);
    }
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00);
    uint32_t h = sign | (uint32_t(exp) << 10) | (mant >> 13);
    if (mant & 0x1000) h += 1;                        // округление к ближайшему
    return static_cast<uint16_t>(h);
}

}  // namespace

Image2D VkContext::create_image2d(uint32_t w, uint32_t h, VkFormat format, VkImageUsageFlags usage) {
    Image2D im;
    im.w = w;
    im.h = h;
    im.format = format;
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = format;
    ici.extent = {w, h, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = usage;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    check(vmaCreateImage(vma_, &ici, &ai, &im.image, &im.alloc, nullptr), "vmaCreateImage(2D)");
    return im;
}

void VkContext::destroy_image2d(Image2D& im) {
    if (im.image) vmaDestroyImage(vma_, im.image, im.alloc);
    im = Image2D{};
}

CubeImage VkContext::create_cube(const float* texels, uint32_t cubes, uint32_t N, VkFormat format, float max_aniso) {
    CubeImage c;
    c.size = N;
    c.cubes = cubes;
    c.format = format;
    c.levels = 1;
    while ((N >> c.levels) > 0) ++c.levels;
    const uint32_t layers = 6 * cubes;
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = format;
    ici.extent = {N, N, 1};
    ici.mipLevels = c.levels;
    ici.arrayLayers = layers;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    ai.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    check(vmaCreateImage(vma_, &ici, &ai, &c.image, &c.alloc, nullptr), "vmaCreateImage");

    // загрузка уровня 0
    const size_t texel_count = size_t(layers) * N * N;
    const size_t bpp = format == VK_FORMAT_R16G16B16A16_SFLOAT ? 8 : 16;
    Buffer staging = create_buffer(texel_count * bpp, true);
    if (bpp == 8) {
        auto* dst = static_cast<uint16_t*>(staging.mapped);
        for (size_t i = 0; i < texel_count * 4; ++i) dst[i] = to_half(texels[i]);
    } else {
        std::memcpy(staging.mapped, texels, texel_count * 16);
    }
    VkCommandBuffer cb = begin_cmd();
    auto barrier = [&](uint32_t level, uint32_t nlev, VkImageLayout from, VkImageLayout to, VkAccessFlags sa,
                       VkAccessFlags da, VkPipelineStageFlags ss, VkPipelineStageFlags dst_stage) {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.oldLayout = from;
        b.newLayout = to;
        b.srcAccessMask = sa;
        b.dstAccessMask = da;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = c.image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, level, nlev, 0, layers};
        vkCmdPipelineBarrier(cb, ss, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &b);
    };
    barrier(0, c.levels, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, layers};
    copy.imageExtent = {N, N, 1};
    vkCmdCopyBufferToImage(cb, staging.buf, c.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    // мипы: blit с линейным фильтром ровно ×2 = бокс 2×2 внутри грани (как в прототипе)
    for (uint32_t l = 1; l < c.levels; ++l) {
        barrier(l - 1, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkImageBlit blit{};
        int32_t s0 = int32_t(N >> (l - 1)), s1 = int32_t(N >> l);
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, l - 1, 0, layers};
        blit.srcOffsets[1] = {s0, s0, 1};
        blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, l, 0, layers};
        blit.dstOffsets[1] = {s1, s1, 1};
        vkCmdBlitImage(cb, c.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, c.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1, &blit, VK_FILTER_LINEAR);
    }
    barrier(c.levels - 1, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);
    barrier(0, c.levels, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    submit_wait(cb);
    destroy_buffer(staging);

    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = c.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
    vi.format = format;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, c.levels, 0, layers};
    check(vkCreateImageView(device_, &vi, nullptr, &c.view), "vkCreateImageView");

    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.anisotropyEnable = max_aniso > 1.0f ? VK_TRUE : VK_FALSE;
    si.maxAnisotropy = max_aniso;
    si.maxLod = float(c.levels);
    check(vkCreateSampler(device_, &si, nullptr, &c.sampler), "vkCreateSampler");
    return c;
}

void VkContext::destroy_cube(CubeImage& c) {
    if (c.sampler) vkDestroySampler(device_, c.sampler, nullptr);
    if (c.view) vkDestroyImageView(device_, c.view, nullptr);
    if (c.image) vmaDestroyImage(vma_, c.image, c.alloc);
    c = CubeImage{};
}

std::vector<std::pair<std::string, double>> VkContext::run_graph(const std::vector<GraphStep>& steps) {
    // по одному набору дескрипторов на конвейер (привязки одинаковы во всех его шагах)
    std::map<const Pipeline*, VkDescriptorSet> sets;
    std::vector<VkDescriptorBufferInfo> bi;
    std::vector<VkDescriptorImageInfo> ii;
    bi.reserve(4096);
    ii.reserve(512);
    for (auto& s : steps) {
        if (!s.pipe || sets.count(s.pipe)) continue;
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = dpool_;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &s.pipe->dsl;
        VkDescriptorSet ds;
        check(vkAllocateDescriptorSets(device_, &ai, &ds), "vkAllocateDescriptorSets");
        std::vector<VkWriteDescriptorSet> w(s.bindings.size());
        for (size_t i = 0; i < s.bindings.size(); ++i) {
            w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w[i].dstSet = ds;
            w[i].dstBinding = static_cast<uint32_t>(i);
            w[i].descriptorCount = 1;
            if (s.bindings[i].cube) {
                ii.push_back({s.bindings[i].cube->sampler, s.bindings[i].cube->view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
                w[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                w[i].pImageInfo = &ii.back();
            } else {
                bi.push_back({s.bindings[i].buffer->buf, 0, VK_WHOLE_SIZE});
                w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                w[i].pBufferInfo = &bi.back();
            }
        }
        vkUpdateDescriptorSets(device_, static_cast<uint32_t>(w.size()), w.data(), 0, nullptr);
        sets[s.pipe] = ds;
    }
    std::vector<std::string> labels;
    for (auto& s : steps)
        if (!s.mark.empty()) labels.push_back(s.mark);
    if (labels.size() > kMaxTimestamps) throw std::runtime_error("слишком много меток");
    if (!labels.empty()) vkResetQueryPool(device_, qpool_, 0, static_cast<uint32_t>(labels.size()));

    VkCommandBuffer cb = begin_cmd();
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    uint32_t q = 0;
    const Pipeline* bound = nullptr;
    for (auto& s : steps) {
        if (!s.mark.empty()) {
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool_, q++);
            continue;
        }
        if (bound != s.pipe) {
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s.pipe->pipe);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s.pipe->layout, 0, 1, &sets[s.pipe], 0, nullptr);
            bound = s.pipe;
        }
        if (!s.push.empty())
            vkCmdPushConstants(cb, s.pipe->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(s.push.size()), s.push.data());
        if (s.indirect) vkCmdDispatchIndirect(cb, s.indirect->buf, s.offset);
        else vkCmdDispatch(cb, s.g[0], s.g[1], s.g[2]);
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1, &mb, 0,
                             nullptr, 0, nullptr);
    }
    submit_wait(cb);
    std::vector<std::pair<std::string, double>> out;
    if (!labels.empty()) {
        std::vector<uint64_t> ts(labels.size());
        check(vkGetQueryPoolResults(device_, qpool_, 0, static_cast<uint32_t>(ts.size()), ts.size() * 8, ts.data(), 8,
                                    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
              "vkGetQueryPoolResults");
        for (size_t i = 0; i < labels.size(); ++i)
            out.emplace_back(labels[i], i ? double(ts[i] - ts[i - 1]) * ts_period_ : 0.0);
    }
    for (auto& [p, ds] : sets) vkFreeDescriptorSets(device_, dpool_, 1, &ds);
    return out;
}

CubeImage VkContext::compress_bc6h(CubeImage& src, const std::vector<uint32_t>& encoder_spirv, float max_aniso) {
    if (src.format != VK_FORMAT_R32G32B32A32_SFLOAT) throw std::runtime_error("BC6H: исходная кубмапа должна быть RGBA32F");
    const uint32_t layers = 6 * src.cubes, N = src.size;
    CubeImage c;
    c.size = N;
    c.cubes = src.cubes;
    c.levels = src.levels;
    c.format = VK_FORMAT_BC6H_UFLOAT_BLOCK;
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = c.format;
    ici.extent = {N, N, 1};
    ici.mipLevels = c.levels;
    ici.arrayLayers = layers;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    ai.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    check(vmaCreateImage(vma_, &ici, &ai, &c.image, &c.alloc, nullptr), "vmaCreateImage(BC6H)");

    Pipeline enc = create_pipeline(encoder_spirv, "bc6h_encode",
                                   {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, 8);
    auto image_barrier = [&](VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to, VkAccessFlags sa,
                             VkAccessFlags da, VkPipelineStageFlags ss, VkPipelineStageFlags ds) {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.oldLayout = from;
        b.newLayout = to;
        b.srcAccessMask = sa;
        b.dstAccessMask = da;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = img;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, c.levels, 0, layers};
        vkCmdPipelineBarrier(cb, ss, ds, 0, 0, nullptr, 0, nullptr, 1, &b);
    };
    {
        VkCommandBuffer cb = begin_cmd();
        image_barrier(cb, src.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                      VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      VK_PIPELINE_STAGE_TRANSFER_BIT);
        image_barrier(cb, c.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
                      VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        submit_wait(cb);
    }
    for (uint32_t l = 0; l < c.levels; ++l) {
        const uint32_t s = std::max(N >> l, 1u), nb = (s + 3) / 4;
        Buffer tex = create_buffer(size_t(layers) * s * s * 16, false);
        Buffer blocks = create_buffer(size_t(layers) * nb * nb * 16, false);
        VkCommandBuffer cb = begin_cmd();
        VkBufferImageCopy cp{};
        cp.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, l, 0, layers};
        cp.imageExtent = {s, s, 1};
        vkCmdCopyImageToBuffer(cb, src.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, tex.buf, 1, &cp);
        submit_wait(cb);
        uint32_t push[2] = {s, layers};
        dispatch(enc, {Binding{&tex, nullptr}, Binding{&blocks, nullptr}}, push, 8, (nb + 7) / 8, (nb + 7) / 8, layers, 1);
        cb = begin_cmd();
        VkBufferImageCopy cb2{};
        cb2.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, l, 0, layers};
        cb2.imageExtent = {s, s, 1};
        vkCmdCopyBufferToImage(cb, blocks.buf, c.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cb2);
        submit_wait(cb);
        destroy_buffer(tex);
        destroy_buffer(blocks);
    }
    {
        VkCommandBuffer cb = begin_cmd();
        image_barrier(cb, c.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                      VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        submit_wait(cb);
    }
    destroy_pipeline(enc);
    destroy_cube(src);

    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = c.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
    vi.format = c.format;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, c.levels, 0, layers};
    check(vkCreateImageView(device_, &vi, nullptr, &c.view), "vkCreateImageView(BC6H)");
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.anisotropyEnable = max_aniso > 1.0f ? VK_TRUE : VK_FALSE;
    si.maxAnisotropy = max_aniso;
    si.maxLod = float(c.levels);
    check(vkCreateSampler(device_, &si, nullptr, &c.sampler), "vkCreateSampler(BC6H)");
    return c;
}
