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
    // экспортируемая память (обмен с CUDA, DD-007): выделена без VMA
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* win32 = nullptr;       // HANDLE
    size_t alloc_size = 0;
};

// Кубмапа-массив (листы × 6 граней) с мипами и сэмплером (DD-004 п. 8)
struct CubeImage {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation alloc = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    uint32_t size = 0, levels = 0, cubes = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
};

// Привязка: буфер хранения или комбинированный сэмплер кубмапы
struct Binding {
    Buffer* buffer = nullptr;
    CubeImage* cube = nullptr;
};

struct Pipeline;

// Шаг графа: диспетчеризация (прямая или косвенная) или метка времени
struct GraphStep {
    std::string mark;                 // непусто — метка
    const Pipeline* pipe = nullptr;
    std::vector<Binding> bindings;
    std::vector<uint8_t> push;
    uint32_t g[3] = {1, 1, 1};
    Buffer* indirect = nullptr;
    uint32_t offset = 0;
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
    // instance_exts — расширения экземпляра для окна (SDL); swapchain — включить VK_KHR_swapchain
    explicit VkContext(const std::vector<const char*>& instance_exts = {}, bool swapchain = false);
    ~VkContext();
    VkContext(const VkContext&) = delete;
    VkContext& operator=(const VkContext&) = delete;

    std::string device_name() const;
    std::string driver_version() const;
    float timestamp_period_ns() const { return ts_period_; }

    Buffer create_buffer(size_t bytes, bool host_visible);
    // буфер на устройстве с экспортом памяти (VK_KHR_external_memory_win32) — для импорта в CUDA
    Buffer create_exportable_buffer(size_t bytes);
    bool external_memory() const { return ext_mem_; }
    VkInstance instance() const { return instance_; }
    VkPhysicalDevice physical() const { return phys_; }
    VkDevice device() const { return device_; }
    VkQueue queue() const { return queue_; }
    uint32_t queue_family() const { return qfam_; }
    VkCommandPool command_pool() const { return pool_; }
    void destroy_buffer(Buffer& b);
    void copy_buffer(const Buffer& src, const Buffer& dst, size_t bytes);

    // texels: cubes × 6 × N × N × 4 float (RGBA, уровень 0); формат RGBA16F или RGBA32F; мипы — blit (бокс 2×2)
    CubeImage create_cube(const float* texels, uint32_t cubes, uint32_t N, VkFormat format, float max_aniso);
    void destroy_cube(CubeImage& c);
    // BC6H (UFLOAT) из float-кубмапы с мипами: каждый уровень кодируется на GPU (bc6h_encode.slang); src уничтожается
    CubeImage compress_bc6h(CubeImage& src, const std::vector<uint32_t>& encoder_spirv, float max_aniso);

    // types[i] — тип дескриптора привязки i
    Pipeline create_pipeline(const std::vector<uint32_t>& spirv, const std::string& entry,
                             const std::vector<VkDescriptorType>& types, uint32_t push_bytes);
    void destroy_pipeline(Pipeline& p);

    // Одна диспетчеризация repeat раз; возвращает время каждой (нс) по timestamp queries.
    std::vector<double> dispatch(const Pipeline& p, const std::vector<Binding>& bindings, const void* push,
                                 uint32_t push_bytes, uint32_t gx, uint32_t gy, uint32_t gz, uint32_t repeat);

    // Граф шагов одним командным буфером; возвращает (метка, нс от предыдущей метки)
    std::vector<std::pair<std::string, double>> run_graph(const std::vector<GraphStep>& steps);

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
    bool ext_mem_ = false;
    VkPhysicalDeviceProperties props_{};

    VkCommandBuffer begin_cmd();
    void submit_wait(VkCommandBuffer cb);
};
