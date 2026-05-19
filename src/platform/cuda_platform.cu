/**
 * CUDA Platform Implementation
 *
 * Implements IPlatform for NVIDIA GPUs (RTX 3060, RTX 5090).
 * Optimized for both Ampere (3060) and Blackwell (5090) architectures.
 *
 * Features:
 * - Multi-GPU support
 * - Pinned memory for fast transfers
 * - Async streams and events
 * - Architecture-specific optimizations
 */

#include "platform.hpp"

#if defined(COLLIDER_BACKEND_CUDA)

#include <cuda_runtime.h>
#include <vector>

namespace collider {
namespace platform {

/**
 * CUDA error checking macro
 */
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        return {ErrorCode::Unknown, cudaGetErrorString(err)}; \
    } \
} while(0)

/**
 * CUDA Platform Implementation
 */
class CUDAPlatform : public IPlatform {
public:
    CUDAPlatform() = default;
    ~CUDAPlatform() override { shutdown(); }

    Result initialize() override {
        if (initialized_) {
            return {ErrorCode::Success, "Already initialized"};
        }

        // Get device count
        int count;
        cudaError_t err = cudaGetDeviceCount(&count);
        if (err != cudaSuccess || count == 0) {
            return {ErrorCode::InvalidDevice, "No CUDA devices found"};
        }

        device_count_ = count;

        // Initialize device info
        device_infos_.resize(count);
        for (int i = 0; i < count; i++) {
            cudaDeviceProp prop;
            err = cudaGetDeviceProperties(&prop, i);
            if (err != cudaSuccess) continue;

            DeviceInfo& info = device_infos_[i];
            info.device_id = i;
            info.name = prop.name;
            info.vendor = "NVIDIA";

            info.total_memory = prop.totalGlobalMem;

            // Get available memory
            size_t free_mem, total_mem;
            cudaSetDevice(i);
            cudaMemGetInfo(&free_mem, &total_mem);
            info.available_memory = free_mem;

            info.compute_major = prop.major;
            info.compute_minor = prop.minor;

            // Architecture detection
            // Blackwell: SM 10.0+ (RTX 5090)
            // Ada Lovelace: SM 8.9 (RTX 4090)
            // Ampere: SM 8.0-8.6 (RTX 3060, 3070, 3080, 3090)
            info.is_blackwell = (prop.major >= 10);
            info.is_ampere = (prop.major == 8 && prop.minor <= 6);
            info.is_apple_silicon = false;

            info.supports_fp16 = (prop.major >= 6);
            info.supports_int8 = (prop.major >= 6);

            info.multiprocessor_count = prop.multiProcessorCount;
            info.max_threads_per_block = prop.maxThreadsPerBlock;
            info.warp_size = prop.warpSize;
            info.shared_memory_per_block = prop.sharedMemPerBlock;
            info.l2_cache_size = prop.l2CacheSize;
        }

        // Set default device (prefer highest compute capability)
        current_device_ = 0;
        int best_sm = 0;
        for (int i = 0; i < count; i++) {
            int sm = device_infos_[i].compute_major * 10 + device_infos_[i].compute_minor;
            if (sm > best_sm) {
                best_sm = sm;
                current_device_ = i;
            }
        }

        err = cudaSetDevice(current_device_);
        if (err != cudaSuccess) {
            return {ErrorCode::InvalidDevice, cudaGetErrorString(err)};
        }

        initialized_ = true;
        return {ErrorCode::Success, ""};
    }

    void shutdown() override {
        if (!initialized_) return;

        // Free all buffers
        for (auto& buf : allocated_buffers_) {
            if (buf.device_ptr) {
                cudaSetDevice(buf.device_id);
                cudaFree(buf.device_ptr);
            }
            if (buf.host_ptr && (buf.flags & MemoryFlags::Pinned)) {
                cudaFreeHost(buf.host_ptr);
            }
        }
        allocated_buffers_.clear();

        // Destroy streams
        for (auto& stream : allocated_streams_) {
            if (stream.native_handle) {
                cudaSetDevice(stream.device_id);
                cudaStreamDestroy(static_cast<cudaStream_t>(stream.native_handle));
            }
        }
        allocated_streams_.clear();

        // Destroy events
        for (auto& event : allocated_events_) {
            if (event.native_handle) {
                cudaSetDevice(event.device_id);
                cudaEventDestroy(static_cast<cudaEvent_t>(event.native_handle));
            }
        }
        allocated_events_.clear();

        // Reset all devices
        for (int i = 0; i < device_count_; i++) {
            cudaSetDevice(i);
            cudaDeviceReset();
        }

        initialized_ = false;
    }

    bool is_initialized() const override { return initialized_; }

    // Device management
    int get_device_count() const override { return device_count_; }

    DeviceInfo get_device_info(int device_id) const override {
        if (device_id >= 0 && device_id < device_count_) {
            return device_infos_[device_id];
        }
        return {};
    }

    Result set_device(int device_id) override {
        if (device_id < 0 || device_id >= device_count_) {
            return {ErrorCode::InvalidDevice, "Invalid device ID"};
        }
        CUDA_CHECK(cudaSetDevice(device_id));
        current_device_ = device_id;
        return {ErrorCode::Success, ""};
    }

    int get_current_device() const override { return current_device_; }

    // Memory management
    Result allocate(Buffer& buffer, size_t size, MemoryFlags flags) override {
        // Ensure we're on the right device
        CUDA_CHECK(cudaSetDevice(current_device_));

        buffer.device_ptr = nullptr;
        buffer.host_ptr = nullptr;
        buffer.size = size;
        buffer.flags = flags;
        buffer.device_id = current_device_;

        // Allocate device memory (128-byte aligned for optimal coalescing)
        size_t aligned_size = ((size + 127) / 128) * 128;

        if (flags & MemoryFlags::HostVisible) {
            // Managed memory (accessible from both host and device)
            CUDA_CHECK(cudaMallocManaged(&buffer.device_ptr, aligned_size));
            buffer.host_ptr = buffer.device_ptr;
        } else {
            // Device-only memory
            CUDA_CHECK(cudaMalloc(&buffer.device_ptr, aligned_size));
        }

        // Allocate pinned host memory if requested
        if (flags & MemoryFlags::Pinned && !(flags & MemoryFlags::HostVisible)) {
            CUDA_CHECK(cudaMallocHost(&buffer.host_ptr, aligned_size));
        }

        // L2 cache persistence for Cached buffers would require stream context
        // which is not available at allocation time. The cache hint is applied
        // at kernel launch time via stream attributes instead.
        (void)flags;  // Cached flag handled at kernel launch

        allocated_buffers_.push_back(buffer);
        return {ErrorCode::Success, ""};
    }

    void free(Buffer& buffer) override {
        if (!buffer.device_ptr) return;

        cudaSetDevice(buffer.device_id);

        // Free device memory
        if (buffer.flags & MemoryFlags::HostVisible) {
            cudaFree(buffer.device_ptr);  // Managed memory
        } else {
            cudaFree(buffer.device_ptr);
            if (buffer.host_ptr && (buffer.flags & MemoryFlags::Pinned)) {
                cudaFreeHost(buffer.host_ptr);
            }
        }

        // Remove from tracking
        allocated_buffers_.erase(
            std::remove_if(allocated_buffers_.begin(), allocated_buffers_.end(),
                [&](const Buffer& b) { return b.device_ptr == buffer.device_ptr; }),
            allocated_buffers_.end()
        );

        buffer.device_ptr = nullptr;
        buffer.host_ptr = nullptr;
        buffer.size = 0;
    }

    Result copy_to_device(Buffer& dst, const void* src, size_t size) override {
        if (!dst.device_ptr || !src) {
            return {ErrorCode::InvalidArgument, "Invalid buffer or source"};
        }
        if (size > dst.size) {
            return {ErrorCode::InvalidArgument, "Size exceeds buffer capacity"};
        }

        CUDA_CHECK(cudaSetDevice(dst.device_id));
        CUDA_CHECK(cudaMemcpy(dst.device_ptr, src, size, cudaMemcpyHostToDevice));
        return {ErrorCode::Success, ""};
    }

    Result copy_to_host(void* dst, const Buffer& src, size_t size) override {
        if (!dst || !src.device_ptr) {
            return {ErrorCode::InvalidArgument, "Invalid destination or buffer"};
        }
        if (size > src.size) {
            return {ErrorCode::InvalidArgument, "Size exceeds buffer capacity"};
        }

        CUDA_CHECK(cudaSetDevice(src.device_id));
        CUDA_CHECK(cudaMemcpy(dst, src.device_ptr, size, cudaMemcpyDeviceToHost));
        return {ErrorCode::Success, ""};
    }

    Result copy_device_to_device(Buffer& dst, const Buffer& src, size_t size) override {
        if (!dst.device_ptr || !src.device_ptr) {
            return {ErrorCode::InvalidArgument, "Invalid buffers"};
        }
        if (size > dst.size || size > src.size) {
            return {ErrorCode::InvalidArgument, "Size exceeds buffer capacity"};
        }

        if (dst.device_id == src.device_id) {
            // Same device
            CUDA_CHECK(cudaSetDevice(dst.device_id));
            CUDA_CHECK(cudaMemcpy(dst.device_ptr, src.device_ptr, size, cudaMemcpyDeviceToDevice));
        } else {
            // Cross-device copy (requires peer access or staging)
            CUDA_CHECK(cudaMemcpyPeer(dst.device_ptr, dst.device_id,
                                       src.device_ptr, src.device_id, size));
        }
        return {ErrorCode::Success, ""};
    }

    // Async memory operations
    Result copy_to_device_async(Buffer& dst, const void* src, size_t size, Stream& stream) override {
        if (!dst.device_ptr || !src) {
            return {ErrorCode::InvalidArgument, "Invalid buffer or source"};
        }
        if (size > dst.size) {
            return {ErrorCode::InvalidArgument, "Size exceeds buffer capacity"};
        }

        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream.native_handle);
        CUDA_CHECK(cudaSetDevice(dst.device_id));
        CUDA_CHECK(cudaMemcpyAsync(dst.device_ptr, src, size, cudaMemcpyHostToDevice, cuda_stream));
        return {ErrorCode::Success, ""};
    }

    Result copy_to_host_async(void* dst, const Buffer& src, size_t size, Stream& stream) override {
        if (!dst || !src.device_ptr) {
            return {ErrorCode::InvalidArgument, "Invalid destination or buffer"};
        }
        if (size > src.size) {
            return {ErrorCode::InvalidArgument, "Size exceeds buffer capacity"};
        }

        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream.native_handle);
        CUDA_CHECK(cudaSetDevice(src.device_id));
        CUDA_CHECK(cudaMemcpyAsync(dst, src.device_ptr, size, cudaMemcpyDeviceToHost, cuda_stream));
        return {ErrorCode::Success, ""};
    }

    // Stream management
    Result create_stream(Stream& stream) override {
        CUDA_CHECK(cudaSetDevice(current_device_));

        cudaStream_t cuda_stream;
        // Use non-blocking stream for better concurrency
        CUDA_CHECK(cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking));

        stream.native_handle = cuda_stream;
        stream.device_id = current_device_;

        allocated_streams_.push_back(stream);
        return {ErrorCode::Success, ""};
    }

    void destroy_stream(Stream& stream) override {
        if (stream.native_handle) {
            cudaSetDevice(stream.device_id);
            cudaStreamDestroy(static_cast<cudaStream_t>(stream.native_handle));

            allocated_streams_.erase(
                std::remove_if(allocated_streams_.begin(), allocated_streams_.end(),
                    [&](const Stream& s) { return s.native_handle == stream.native_handle; }),
                allocated_streams_.end()
            );

            stream.native_handle = nullptr;
        }
    }

    Result synchronize_stream(Stream& stream) override {
        if (!stream.native_handle) {
            return {ErrorCode::InvalidArgument, "Invalid stream"};
        }

        CUDA_CHECK(cudaSetDevice(stream.device_id));
        CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(stream.native_handle)));
        return {ErrorCode::Success, ""};
    }

    Result synchronize_device() override {
        CUDA_CHECK(cudaSetDevice(current_device_));
        CUDA_CHECK(cudaDeviceSynchronize());
        return {ErrorCode::Success, ""};
    }

    // Event management
    Result create_event(Event& event) override {
        CUDA_CHECK(cudaSetDevice(current_device_));

        cudaEvent_t cuda_event;
        // Use blocking sync for accurate timing
        CUDA_CHECK(cudaEventCreateWithFlags(&cuda_event, cudaEventDisableTiming));

        event.native_handle = cuda_event;
        event.device_id = current_device_;

        allocated_events_.push_back(event);
        return {ErrorCode::Success, ""};
    }

    void destroy_event(Event& event) override {
        if (event.native_handle) {
            cudaSetDevice(event.device_id);
            cudaEventDestroy(static_cast<cudaEvent_t>(event.native_handle));

            allocated_events_.erase(
                std::remove_if(allocated_events_.begin(), allocated_events_.end(),
                    [&](const Event& e) { return e.native_handle == event.native_handle; }),
                allocated_events_.end()
            );

            event.native_handle = nullptr;
        }
    }

    Result record_event(Event& event, Stream& stream) override {
        if (!event.native_handle || !stream.native_handle) {
            return {ErrorCode::InvalidArgument, "Invalid event or stream"};
        }

        CUDA_CHECK(cudaSetDevice(event.device_id));
        CUDA_CHECK(cudaEventRecord(
            static_cast<cudaEvent_t>(event.native_handle),
            static_cast<cudaStream_t>(stream.native_handle)
        ));
        return {ErrorCode::Success, ""};
    }

    Result wait_event(Stream& stream, Event& event) override {
        if (!event.native_handle || !stream.native_handle) {
            return {ErrorCode::InvalidArgument, "Invalid event or stream"};
        }

        CUDA_CHECK(cudaSetDevice(stream.device_id));
        CUDA_CHECK(cudaStreamWaitEvent(
            static_cast<cudaStream_t>(stream.native_handle),
            static_cast<cudaEvent_t>(event.native_handle),
            0  // flags
        ));
        return {ErrorCode::Success, ""};
    }

    Result synchronize_event(Event& event) override {
        if (!event.native_handle) {
            return {ErrorCode::InvalidArgument, "Invalid event"};
        }

        CUDA_CHECK(cudaSetDevice(event.device_id));
        CUDA_CHECK(cudaEventSynchronize(static_cast<cudaEvent_t>(event.native_handle)));
        return {ErrorCode::Success, ""};
    }

    // Platform info
    std::string get_platform_name() const override {
#ifdef _WIN32
        return "Windows";
#else
        return "Linux";
#endif
    }

    std::string get_backend_name() const override { return "CUDA"; }

private:
    bool initialized_ = false;
    int device_count_ = 0;
    int current_device_ = 0;

    std::vector<DeviceInfo> device_infos_;
    std::vector<Buffer> allocated_buffers_;
    std::vector<Stream> allocated_streams_;
    std::vector<Event> allocated_events_;
};

// Factory function
IPlatform& get_platform() {
    static CUDAPlatform platform;
    static bool initialized = false;
    if (!initialized) {
        platform.initialize();
        initialized = true;
    }
    return platform;
}

}  // namespace platform
}  // namespace collider

#endif  // COLLIDER_BACKEND_CUDA
