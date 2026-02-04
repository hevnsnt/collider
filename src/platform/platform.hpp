/**
 * Collider Platform Abstraction Layer
 *
 * Provides unified interface for GPU compute across:
 * - NVIDIA CUDA (Windows/Linux - RTX 3060, 5090)
 * - Apple Metal (macOS M1/M2/M3)
 * - CPU fallback (testing/development)
 *
 * Architecture:
 *   Application Code -> Platform API -> Backend Implementation
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <optional>

namespace collider {
namespace platform {

// Platform detection
#if defined(__APPLE__)
    #include <TargetConditionals.h>
    #if TARGET_OS_MAC
        #define COLLIDER_PLATFORM_MACOS 1
        #define COLLIDER_PLATFORM_NAME "macOS"
        #if defined(__arm64__) || defined(__aarch64__)
            #define COLLIDER_APPLE_SILICON 1
        #endif
    #endif
#elif defined(_WIN32) || defined(_WIN64)
    #define COLLIDER_PLATFORM_WINDOWS 1
    #define COLLIDER_PLATFORM_NAME "Windows"
#elif defined(__linux__)
    #define COLLIDER_PLATFORM_LINUX 1
    #define COLLIDER_PLATFORM_NAME "Linux"
#else
    #define COLLIDER_PLATFORM_UNKNOWN 1
    #define COLLIDER_PLATFORM_NAME "Unknown"
#endif

// Backend detection
#if defined(COLLIDER_USE_CUDA) || (defined(__CUDACC__) || defined(CUDA_VERSION))
    #define COLLIDER_BACKEND_CUDA 1
    #define COLLIDER_BACKEND_NAME "CUDA"
#elif defined(COLLIDER_USE_METAL) || defined(COLLIDER_APPLE_SILICON)
    #define COLLIDER_BACKEND_METAL 1
    #define COLLIDER_BACKEND_NAME "Metal"
#else
    #define COLLIDER_BACKEND_CPU 1
    #define COLLIDER_BACKEND_NAME "CPU"
#endif

/**
 * GPU Device Information
 */
struct DeviceInfo {
    int device_id;
    std::string name;
    std::string vendor;

    // Memory
    size_t total_memory;        // Total VRAM in bytes
    size_t available_memory;    // Currently available

    // Compute capability
    int compute_major;
    int compute_minor;

    // Architecture hints
    bool is_blackwell;          // RTX 5090
    bool is_ampere;             // RTX 3060
    bool is_apple_silicon;      // M1/M2/M3
    bool supports_fp16;
    bool supports_int8;

    // Performance hints
    int multiprocessor_count;
    int max_threads_per_block;
    int warp_size;
    size_t shared_memory_per_block;
    size_t l2_cache_size;
};

/**
 * Memory allocation flags
 */
enum class MemoryFlags : uint32_t {
    Default = 0,
    HostVisible = 1 << 0,       // Can be accessed from CPU
    DeviceLocal = 1 << 1,       // Fast GPU memory
    Pinned = 1 << 2,            // Pinned host memory
    Coherent = 1 << 3,          // Automatic sync
    Cached = 1 << 4,            // Use L2 cache persistence
};

inline MemoryFlags operator|(MemoryFlags a, MemoryFlags b) {
    return static_cast<MemoryFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

inline bool operator&(MemoryFlags a, MemoryFlags b) {
    return (static_cast<uint32_t>(a) & static_cast<uint32_t>(b)) != 0;
}

/**
 * GPU Buffer handle
 */
struct Buffer {
    void* device_ptr = nullptr;
    void* host_ptr = nullptr;
    size_t size = 0;
    MemoryFlags flags = MemoryFlags::Default;
    int device_id = 0;
};

/**
 * Compute stream/queue handle
 */
struct Stream {
    void* native_handle = nullptr;
    int device_id = 0;
};

/**
 * Event for synchronization
 */
struct Event {
    void* native_handle = nullptr;
    int device_id = 0;
};

/**
 * Kernel/shader configuration
 */
struct KernelConfig {
    size_t grid_size[3] = {1, 1, 1};
    size_t block_size[3] = {256, 1, 1};
    size_t shared_memory = 0;
    Stream* stream = nullptr;
};

/**
 * Platform-specific error codes
 */
enum class ErrorCode {
    Success = 0,
    OutOfMemory,
    InvalidDevice,
    InvalidArgument,
    NotSupported,
    NotInitialized,
    DeviceLost,
    KernelFailed,
    SyncFailed,
    Unknown
};

/**
 * Platform result wrapper
 */
struct Result {
    ErrorCode code;
    std::string message;

    bool ok() const { return code == ErrorCode::Success; }
    operator bool() const { return ok(); }
};

/**
 * Platform Interface
 *
 * Pure virtual interface implemented by each backend.
 */
class IPlatform {
public:
    virtual ~IPlatform() = default;

    // Initialization
    virtual Result initialize() = 0;
    virtual void shutdown() = 0;
    virtual bool is_initialized() const = 0;

    // Device management
    virtual int get_device_count() const = 0;
    virtual DeviceInfo get_device_info(int device_id) const = 0;
    virtual Result set_device(int device_id) = 0;
    virtual int get_current_device() const = 0;

    // Memory management
    virtual Result allocate(Buffer& buffer, size_t size, MemoryFlags flags) = 0;
    virtual void free(Buffer& buffer) = 0;
    virtual Result copy_to_device(Buffer& dst, const void* src, size_t size) = 0;
    virtual Result copy_to_host(void* dst, const Buffer& src, size_t size) = 0;
    virtual Result copy_device_to_device(Buffer& dst, const Buffer& src, size_t size) = 0;

    // Async memory operations
    virtual Result copy_to_device_async(Buffer& dst, const void* src, size_t size, Stream& stream) = 0;
    virtual Result copy_to_host_async(void* dst, const Buffer& src, size_t size, Stream& stream) = 0;

    // Stream management
    virtual Result create_stream(Stream& stream) = 0;
    virtual void destroy_stream(Stream& stream) = 0;
    virtual Result synchronize_stream(Stream& stream) = 0;
    virtual Result synchronize_device() = 0;

    // Event management
    virtual Result create_event(Event& event) = 0;
    virtual void destroy_event(Event& event) = 0;
    virtual Result record_event(Event& event, Stream& stream) = 0;
    virtual Result wait_event(Stream& stream, Event& event) = 0;
    virtual Result synchronize_event(Event& event) = 0;

    // Platform info
    virtual std::string get_platform_name() const = 0;
    virtual std::string get_backend_name() const = 0;
};

/**
 * Get the platform singleton.
 * Returns appropriate backend based on compile-time detection.
 */
IPlatform& get_platform();

/**
 * Adaptive Configuration
 *
 * Auto-configures batch sizes, buffer sizes based on available hardware.
 */
struct AdaptiveConfig {
    // Batch sizes
    size_t candidates_per_batch;
    size_t max_passphrase_length;
    size_t passphrase_buffer_size;

    // Bloom filter
    size_t bloom_filter_size;
    bool bloom_in_texture_memory;

    // Double buffering
    int num_buffers;
    bool use_pinned_memory;

    // Kernel configuration
    int threads_per_block;
    int blocks_per_multiprocessor;

    // Memory limits
    size_t max_gpu_memory_usage;
    size_t reserved_memory;

    /**
     * Create configuration for specific device.
     */
    static AdaptiveConfig for_device(const DeviceInfo& device) {
        AdaptiveConfig config;

        // Base on available memory
        size_t available = device.total_memory;

        if (device.is_blackwell) {
            // RTX 5090 - 32GB VRAM, Blackwell architecture
            config.candidates_per_batch = 4'000'000;
            config.max_passphrase_length = 128;
            config.passphrase_buffer_size = 512 * 1024 * 1024;  // 512 MB
            config.bloom_filter_size = 6ULL * 1024 * 1024 * 1024;  // 6 GB
            config.bloom_in_texture_memory = true;
            config.num_buffers = 3;  // Triple buffering
            config.use_pinned_memory = true;
            config.threads_per_block = 256;
            config.blocks_per_multiprocessor = 4;
            config.max_gpu_memory_usage = 28ULL * 1024 * 1024 * 1024;  // 28 GB
            config.reserved_memory = 4ULL * 1024 * 1024 * 1024;  // 4 GB reserved

        } else if (device.is_ampere || available >= 10ULL * 1024 * 1024 * 1024) {
            // RTX 3060 12GB or similar
            config.candidates_per_batch = 1'000'000;
            config.max_passphrase_length = 64;
            config.passphrase_buffer_size = 128 * 1024 * 1024;  // 128 MB
            config.bloom_filter_size = 4ULL * 1024 * 1024 * 1024;  // 4 GB (fits in 12GB)
            config.bloom_in_texture_memory = true;
            config.num_buffers = 2;  // Double buffering
            config.use_pinned_memory = true;
            config.threads_per_block = 256;
            config.blocks_per_multiprocessor = 2;
            config.max_gpu_memory_usage = 10ULL * 1024 * 1024 * 1024;  // 10 GB
            config.reserved_memory = 2ULL * 1024 * 1024 * 1024;  // 2 GB reserved

        } else if (device.is_apple_silicon) {
            // M1/M2/M3 - Unified memory
            config.candidates_per_batch = 500'000;
            config.max_passphrase_length = 64;
            config.passphrase_buffer_size = 64 * 1024 * 1024;  // 64 MB
            config.bloom_filter_size = 2ULL * 1024 * 1024 * 1024;  // 2 GB
            config.bloom_in_texture_memory = false;  // Metal handles this differently
            config.num_buffers = 2;
            config.use_pinned_memory = false;  // Unified memory doesn't need pinning
            config.threads_per_block = 256;
            config.blocks_per_multiprocessor = 2;
            config.max_gpu_memory_usage = available / 2;  // Use half of unified memory
            config.reserved_memory = 1ULL * 1024 * 1024 * 1024;

        } else {
            // CPU fallback or unknown GPU
            config.candidates_per_batch = 100'000;
            config.max_passphrase_length = 64;
            config.passphrase_buffer_size = 32 * 1024 * 1024;
            config.bloom_filter_size = 512 * 1024 * 1024;  // 512 MB
            config.bloom_in_texture_memory = false;
            config.num_buffers = 2;
            config.use_pinned_memory = false;
            config.threads_per_block = 1;  // CPU: 1 "thread" per work item
            config.blocks_per_multiprocessor = 1;
            config.max_gpu_memory_usage = 2ULL * 1024 * 1024 * 1024;
            config.reserved_memory = 512 * 1024 * 1024;
        }

        return config;
    }

    /**
     * Adjust configuration for specific bloom filter size.
     */
    void adjust_for_bloom_size(size_t actual_bloom_size, size_t device_memory) {
        // Ensure bloom filter fits with headroom
        size_t needed = actual_bloom_size + passphrase_buffer_size * num_buffers +
                        candidates_per_batch * 200;  // ~200 bytes per candidate

        if (needed > device_memory - reserved_memory) {
            // Scale down
            double scale = static_cast<double>(device_memory - reserved_memory - actual_bloom_size) /
                          (passphrase_buffer_size * num_buffers + candidates_per_batch * 200);
            scale = std::max(0.1, std::min(1.0, scale));

            candidates_per_batch = static_cast<size_t>(candidates_per_batch * scale);
            passphrase_buffer_size = static_cast<size_t>(passphrase_buffer_size * scale);
        }
    }
};

}  // namespace platform
}  // namespace collider
