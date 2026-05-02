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
#include <algorithm>
#include "../core/edition.hpp"

#if defined(__APPLE__)
    #include <TargetConditionals.h>
#endif

namespace collider {
namespace platform {

// Platform detection
#if defined(__APPLE__)
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

enum class VRAMTier : uint8_t {
    Minimal,    // < 8 GB (RTX 2060)
    Standard,   // 8-16 GB (RTX 3060)
    Enhanced,   // 16-48 GB (RTX 5090)
    Maximum     // 48+ GB (RTX PRO 6000)
};

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
    bool is_turing;             // SM 7.5 (RTX 2060/2070/2080)
    bool is_ampere;             // SM 8.0-8.6 (RTX 3060/3090)
    bool is_ada;                // SM 8.9 (RTX 4090)
    bool is_hopper;             // SM 9.0 (H100)
    bool is_blackwell;          // SM 12.0 (RTX 5090, PRO 6000)
    bool is_apple_silicon;      // M1/M2/M3
    VRAMTier vram_tier;
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


    // Double buffering
    int num_buffers;
    bool use_pinned_memory;

    // Kernel configuration
    int threads_per_block;
    int blocks_per_multiprocessor;

    // Memory limits
    size_t max_gpu_memory_usage;
    size_t reserved_memory;

    // EC precomputation (set by VRAM budget)
    int ec_window_bits;
    size_t ec_table_size_bytes;

    // VRAM budget breakdown (logged at startup)
    size_t vram_total;
    size_t vram_reserved;
    size_t vram_ec_table;
    size_t vram_batch_buffers;
    size_t vram_other;

    /**
     * Create configuration for specific device.
     */
    static AdaptiveConfig for_device(const DeviceInfo& device) {
        AdaptiveConfig config;
        constexpr size_t GB = 1024ULL * 1024 * 1024;
        // Use available memory for budget calculations instead of total memory.
        // This prevents over-allocation when other processes are using VRAM.
        // Fall back to total_memory if available_memory is not populated.
        size_t total = (device.available_memory > 0 && device.available_memory <= device.total_memory)
                       ? device.available_memory : device.total_memory;

        // Apple Silicon -- unchanged
        if (device.is_apple_silicon) {
            config.candidates_per_batch = 500'000;
            config.max_passphrase_length = 64;
            config.passphrase_buffer_size = 64 * 1024 * 1024;
            config.num_buffers = 2;
            config.use_pinned_memory = false;
            config.threads_per_block = 256;
            config.blocks_per_multiprocessor = 2;
            config.max_gpu_memory_usage = total / 2;
            config.reserved_memory = 1 * GB;
            config.ec_window_bits = 5;
            config.ec_table_size_bytes = 52 * 32 * 64;
            config.vram_total = total;
            config.vram_reserved = config.reserved_memory;
            config.vram_ec_table = config.ec_table_size_bytes;
            config.vram_batch_buffers = config.max_gpu_memory_usage / 2;
            config.vram_other = config.max_gpu_memory_usage - config.vram_batch_buffers - config.vram_ec_table;
            return config;
        }

        // CPU fallback
        if (total < 1 * GB) {
            config.candidates_per_batch = 100'000;
            config.max_passphrase_length = 64;
            config.passphrase_buffer_size = 32 * 1024 * 1024;
            config.num_buffers = 2;
            config.use_pinned_memory = false;
            config.threads_per_block = 1;
            config.blocks_per_multiprocessor = 1;
            config.max_gpu_memory_usage = 2 * GB;
            config.reserved_memory = 512 * 1024 * 1024;
            config.ec_window_bits = 5;
            config.ec_table_size_bytes = 52 * 32 * 64;
            config.vram_total = total;
            config.vram_reserved = config.reserved_memory;
            config.vram_ec_table = config.ec_table_size_bytes;
            config.vram_batch_buffers = 0;
            config.vram_other = 0;
            return config;
        }

        // --- CUDA GPU: VRAM-based continuous scaling ---

        // Reserved memory
        if (total >= 16 * GB)
            config.reserved_memory = 4 * GB;
        else if (total >= 8 * GB)
            config.reserved_memory = 2 * GB;
        else
            config.reserved_memory = 1 * GB;

        size_t usable = total - config.reserved_memory;
        config.max_gpu_memory_usage = usable;

        // EC window bits scaled by usable VRAM
        if (usable >= 64 * GB) config.ec_window_bits = 12;
        else if (usable >= 48 * GB) config.ec_window_bits = 10;
        else if (usable >= 16 * GB) config.ec_window_bits = 8;
        else if (usable >= 8 * GB) config.ec_window_bits = 6;
        else config.ec_window_bits = 5;

        int pts_per_window = 1 << config.ec_window_bits;
        int num_windows = (256 + config.ec_window_bits - 1) / config.ec_window_bits;
        config.ec_table_size_bytes = (size_t)num_windows * pts_per_window * 64;

        // Remaining VRAM for batch buffers
        size_t allocated = config.ec_table_size_bytes + 1 * 1024 * 1024;
        size_t remaining = (allocated < usable) ? (usable - allocated) : 0;
        size_t batch_budget = remaining * 80 / 100;

        config.candidates_per_batch = std::min(batch_budget / 200, (size_t)16'000'000);
        config.max_passphrase_length = (usable >= 16 * GB) ? 128 : 64;
        config.passphrase_buffer_size = std::min(remaining / 4, (size_t)(1 * GB));

        // Buffering strategy
        config.num_buffers = (usable >= 16 * GB) ? 3 : 2;
        config.use_pinned_memory = true;

        // Kernel config: 256 threads for all CUDA archs
        config.threads_per_block = 256;
        config.blocks_per_multiprocessor = (device.is_blackwell || device.is_hopper) ? 6 : 4;

        // Budget breakdown
        config.vram_total = total;
        config.vram_reserved = config.reserved_memory;
        config.vram_ec_table = config.ec_table_size_bytes;
        config.vram_batch_buffers = batch_budget;
        config.vram_other = remaining - batch_budget;

        return config;
    }

};

}  // namespace platform
}  // namespace collider
