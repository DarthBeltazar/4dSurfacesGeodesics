// Минимальный Vulkan-контекст для compute: устройство, VMA, буферы, конвейеры, замеры (DD-004 п. 1, 9).
#pragma once

#include <volk.h>
#include <vk_mem_alloc.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

struct Buffer {
    VkBuffer buf = VK_NULL_HANDLE;
    VmaAllocation alloc = VK_NULL_HANDLE;
    size_t size = 0;
    void* mapped = nullptr;      // буферы хоста отображены постоянно
};

struct Pipeline {
    VkPipeline pipe = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkShaderModule module = VK_NULL_HANDLE;
    uint32_t nbindings = 0;
    std::vector<std::pair<std::string, std::string>> stats;   // VK_KHR_pipeline_executable_properties
};

class VkContext {
public:
    VkContext();
    ~VkContext();
    VkContext(const VkContext&) = delete;
    VkContext& operator=(const VkContext&) = delete;

    std::string device_name() const;
    std::string driver_version() const;
    float timestamp_period_ns() const { return ts_period_; }

    Buffer create_buffer(size_t bytes, bool host_visible);
    void destroy_buffer(Buffer& b);
    void copy_buffer(const Buffer& src, const Buffer& dst, size_t bytes);

    Pipeline create_pipeline(const std::vector<uint32_t>& spirv, const std::string& entry, uint32_t nbindings,
                             uint32_t push_bytes);
    void destroy_pipeline(Pipeline& p);

    // Одна диспетчеризация repeat раз; возвращает время каждой (нс) по timestamp queries.
    std::vector<double> dispatch(const Pipeline& p, const std::vector<Buffer*>& bindings, const void* push,
                                 uint32_t push_bytes, uint32_t gx, uint32_t gy, uint32_t gz, uint32_t repeat);

private:
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t qfam_ = 0;
    VmaAllocator vma_ = VK_NULL_HANDLE;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkDescriptorPool dpool_ = VK_NULL_HANDLE;
    VkQueryPool qpool_ = VK_NULL_HANDLE;
    float ts_period_ = 1.0f;
    bool exec_props_ = false;
    VkPhysicalDeviceProperties props_{};

    VkCommandBuffer begin_cmd();
    void submit_wait(VkCommandBuffer cb);
};
