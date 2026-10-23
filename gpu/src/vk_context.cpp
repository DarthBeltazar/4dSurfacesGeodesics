#define VOLK_IMPLEMENTATION
#include <volk.h>
#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vk_mem_alloc.h>

#include "vk_context.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

namespace {

void check(VkResult r, const char* what) {
    if (r != VK_SUCCESS) throw std::runtime_error(std::string(what) + ": VkResult " + std::to_string(r));
}

constexpr uint32_t kMaxTimestamps = 512;

}  // namespace

VkContext::VkContext() {
    check(volkInitialize(), "volkInitialize");

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "geo4d_gpu";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
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
    for (uint32_t i = 0; i < nq; ++i)
        if ((qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && qf[i].timestampValidBits > 0) { qfam_ = i; break; }
    if (qfam_ == UINT32_MAX) throw std::runtime_error("нет очереди compute с timestamp");

    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(phys_, nullptr, &ne, nullptr);
    std::vector<VkExtensionProperties> exts(ne);
    vkEnumerateDeviceExtensionProperties(phys_, nullptr, &ne, exts.data());
    for (auto& e : exts)
        if (std::strcmp(e.extensionName, VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME) == 0) exec_props_ = true;

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
    f13.computeFullSubgroups = VK_TRUE;
    f13.subgroupSizeControl = VK_TRUE;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &f13;
    f2.features.shaderInt64 = VK_TRUE;

    std::vector<const char*> dexts;
    if (exec_props_) dexts.push_back(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
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

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 256};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpi.maxSets = 32;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &ps;
    check(vkCreateDescriptorPool(device_, &dpi, nullptr, &dpool_), "vkCreateDescriptorPool");

    VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpi.queryCount = kMaxTimestamps;
    check(vkCreateQueryPool(device_, &qpi, nullptr, &qpool_), "vkCreateQueryPool");
}

VkContext::~VkContext() {
    if (device_) {
        vkDeviceWaitIdle(device_);
        vkDestroyQueryPool(device_, qpool_, nullptr);
        vkDestroyDescriptorPool(device_, dpool_, nullptr);
        vkDestroyCommandPool(device_, pool_, nullptr);
        vmaDestroyAllocator(vma_);
        vkDestroyDevice(device_, nullptr);
    }
    if (instance_) vkDestroyInstance(instance_, nullptr);
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
    if (b.buf) vmaDestroyBuffer(vma_, b.buf, b.alloc);
    b = Buffer{};
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

void VkContext::copy_buffer(const Buffer& src, const Buffer& dst, size_t bytes) {
    if (bytes == 0) return;
    VkCommandBuffer cb = begin_cmd();
    VkBufferCopy c{0, 0, bytes};
    vkCmdCopyBuffer(cb, src.buf, dst.buf, 1, &c);
    submit_wait(cb);
}

Pipeline VkContext::create_pipeline(const std::vector<uint32_t>& spirv, const std::string& entry, uint32_t nbindings,
                                    uint32_t push_bytes) {
    Pipeline p;
    p.nbindings = nbindings;
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = spirv.size() * 4;
    smi.pCode = spirv.data();
    check(vkCreateShaderModule(device_, &smi, nullptr, &p.module), "vkCreateShaderModule");

    std::vector<VkDescriptorSetLayoutBinding> b(nbindings);
    for (uint32_t i = 0; i < nbindings; ++i) {
        b[i].binding = i;
        b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
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
    check(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cpi, nullptr, &p.pipe), "vkCreateComputePipelines");

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

std::vector<double> VkContext::dispatch(const Pipeline& p, const std::vector<Buffer*>& bindings, const void* push,
                                        uint32_t push_bytes, uint32_t gx, uint32_t gy, uint32_t gz, uint32_t repeat) {
    if (2 * repeat > kMaxTimestamps) throw std::runtime_error("слишком много повторов");
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = dpool_;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &p.dsl;
    VkDescriptorSet ds;
    check(vkAllocateDescriptorSets(device_, &ai, &ds), "vkAllocateDescriptorSets");
    std::vector<VkDescriptorBufferInfo> bi(bindings.size());
    std::vector<VkWriteDescriptorSet> w(bindings.size());
    for (size_t i = 0; i < bindings.size(); ++i) {
        bi[i] = {bindings[i]->buf, 0, VK_WHOLE_SIZE};
        w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[i].dstSet = ds;
        w[i].dstBinding = static_cast<uint32_t>(i);
        w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = &bi[i];
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
