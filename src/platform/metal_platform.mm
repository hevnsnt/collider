/**
 * Metal Platform Implementation
 *
 * Implements IPlatform for Apple Silicon (M1/M2/M3).
 * Uses Metal for GPU compute with unified memory architecture.
 *
 * Key differences from CUDA:
 * - Unified memory (no explicit host/device transfers needed)
 * - Command buffers instead of streams
 * - Compute pipelines instead of kernels
 * - MTLBuffer instead of cudaMalloc
 *
 * NOTE: Metal initialization requires care on macOS CLI apps.
 * MTLCopyAllDevices() can block if there's no run loop active.
 * We use a timeout with async dispatch to prevent hangs.
 */

#include "platform.hpp"

#if defined(COLLIDER_PLATFORM_MACOS) && defined(COLLIDER_BACKEND_METAL)

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <dispatch/dispatch.h>
#include <unordered_map>
#include <iostream>

namespace collider {
namespace platform {

/**
 * Metal Platform Implementation
 */
class MetalPlatform : public IPlatform {
public:
    MetalPlatform() = default;
    ~MetalPlatform() override { shutdown(); }

    // Initialization
    Result initialize() override {
        if (initialized_) {
            return {ErrorCode::Success, "Already initialized"};
        }

        // Flush stdout before Metal init - CLI apps may have buffered output
        std::cout << std::flush;

        @autoreleasepool {
            // Metal device enumeration can block on CLI apps without a run loop.
            // Use async dispatch with timeout to prevent indefinite hangs.
            __block NSArray<id<MTLDevice>>* devices = nil;
            __block bool device_query_complete = false;

            dispatch_semaphore_t sema = dispatch_semaphore_create(0);

            dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^{
                @autoreleasepool {
                    devices = MTLCopyAllDevices();
                    device_query_complete = true;
                    dispatch_semaphore_signal(sema);
                }
            });

            // Wait up to 5 seconds for Metal device enumeration
            // This timeout prevents the app from hanging indefinitely
            long wait_result = dispatch_semaphore_wait(sema,
                dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC));

            if (wait_result != 0) {
                std::cerr << "[!] Metal initialization timed out - falling back to CPU\n";
                std::cerr << "    This can happen in headless environments or SSH sessions.\n";
                std::cerr << std::flush;
                return {ErrorCode::NotInitialized, "Metal initialization timed out"};
            }

            if (!device_query_complete || devices == nil || devices.count == 0) {
                return {ErrorCode::InvalidDevice, "No Metal devices found"};
            }

            // Store devices
            for (id<MTLDevice> device in devices) {
                devices_.push_back(device);
            }

            // Create command queues for each device
            for (id<MTLDevice> device : devices_) {
                id<MTLCommandQueue> queue = [device newCommandQueue];
                if (!queue) {
                    return {ErrorCode::NotInitialized, "Failed to create command queue"};
                }
                command_queues_.push_back(queue);
            }

            // Set default device (prefer discrete GPU if available)
            current_device_ = 0;
            for (size_t i = 0; i < devices_.size(); i++) {
                if (!devices_[i].isLowPower) {
                    current_device_ = static_cast<int>(i);
                    break;
                }
            }

            initialized_ = true;
        }

        return {ErrorCode::Success, ""};
    }

    void shutdown() override {
        if (!initialized_) return;

        @autoreleasepool {
            // Release all buffers
            for (auto& [ptr, buffer] : buffer_map_) {
                // MTLBuffer is ARC-managed
            }
            buffer_map_.clear();

            // Release command queues and devices
            command_queues_.clear();
            devices_.clear();

            initialized_ = false;
        }
    }

    bool is_initialized() const override { return initialized_; }

    // Device management
    int get_device_count() const override {
        return static_cast<int>(devices_.size());
    }

    DeviceInfo get_device_info(int device_id) const override {
        DeviceInfo info = {};

        if (device_id < 0 || device_id >= static_cast<int>(devices_.size())) {
            return info;
        }

        @autoreleasepool {
            id<MTLDevice> device = devices_[device_id];

            info.device_id = device_id;
            info.name = std::string([device.name UTF8String]);
            info.vendor = "Apple";

            // Memory - Metal uses unified memory
            // recommendedMaxWorkingSetSize is a hint, not hard limit
            info.total_memory = device.recommendedMaxWorkingSetSize;
            info.available_memory = info.total_memory;  // Unified, always "available"

            // Apple Silicon detection
            info.is_apple_silicon = true;
            info.is_blackwell = false;
            info.is_ampere = false;

            // Compute capabilities
            info.supports_fp16 = true;  // All Apple Silicon supports FP16
            info.supports_int8 = true;

            // Metal doesn't expose these directly, use reasonable defaults
            info.compute_major = 3;  // Metal 3
            info.compute_minor = 0;

            // Performance hints for Apple Silicon
            // These are approximations based on typical M1/M2/M3 specs
            if (info.name.find("M3") != std::string::npos) {
                info.multiprocessor_count = 10;  // M3 Pro: 14-18, M3 Max: 40
                info.max_threads_per_block = 1024;
                info.warp_size = 32;  // SIMD width
                info.shared_memory_per_block = 32 * 1024;
                info.l2_cache_size = 24 * 1024 * 1024;
            } else if (info.name.find("M2") != std::string::npos) {
                info.multiprocessor_count = 10;
                info.max_threads_per_block = 1024;
                info.warp_size = 32;
                info.shared_memory_per_block = 32 * 1024;
                info.l2_cache_size = 16 * 1024 * 1024;
            } else {
                // M1 or unknown
                info.multiprocessor_count = 8;
                info.max_threads_per_block = 1024;
                info.warp_size = 32;
                info.shared_memory_per_block = 32 * 1024;
                info.l2_cache_size = 8 * 1024 * 1024;
            }
        }

        return info;
    }

    Result set_device(int device_id) override {
        if (device_id < 0 || device_id >= static_cast<int>(devices_.size())) {
            return {ErrorCode::InvalidDevice, "Invalid device ID"};
        }
        current_device_ = device_id;
        return {ErrorCode::Success, ""};
    }

    int get_current_device() const override { return current_device_; }

    // Memory management
    Result allocate(Buffer& buffer, size_t size, MemoryFlags flags) override {
        @autoreleasepool {
            id<MTLDevice> device = devices_[current_device_];

            // Metal resource options
            MTLResourceOptions options = MTLResourceStorageModeShared;  // Unified memory

            if (flags & MemoryFlags::Cached) {
                options |= MTLResourceCPUCacheModeWriteCombined;
            }

            id<MTLBuffer> mtl_buffer = [device newBufferWithLength:size options:options];
            if (!mtl_buffer) {
                return {ErrorCode::OutOfMemory, "Failed to allocate Metal buffer"};
            }

            buffer.device_ptr = mtl_buffer.contents;
            buffer.host_ptr = mtl_buffer.contents;  // Unified memory - same pointer
            buffer.size = size;
            buffer.flags = flags;
            buffer.device_id = current_device_;

            // Store buffer reference
            buffer_map_[buffer.device_ptr] = mtl_buffer;
        }

        return {ErrorCode::Success, ""};
    }

    void free(Buffer& buffer) override {
        if (buffer.device_ptr) {
            @autoreleasepool {
                buffer_map_.erase(buffer.device_ptr);
                // MTLBuffer released by ARC
            }
            buffer.device_ptr = nullptr;
            buffer.host_ptr = nullptr;
            buffer.size = 0;
        }
    }

    Result copy_to_device(Buffer& dst, const void* src, size_t size) override {
        // Unified memory - just memcpy
        if (!dst.device_ptr || !src) {
            return {ErrorCode::InvalidArgument, "Invalid buffer or source"};
        }
        if (size > dst.size) {
            return {ErrorCode::InvalidArgument, "Size exceeds buffer capacity"};
        }

        std::memcpy(dst.device_ptr, src, size);
        return {ErrorCode::Success, ""};
    }

    Result copy_to_host(void* dst, const Buffer& src, size_t size) override {
        // Unified memory - just memcpy
        if (!dst || !src.device_ptr) {
            return {ErrorCode::InvalidArgument, "Invalid destination or buffer"};
        }
        if (size > src.size) {
            return {ErrorCode::InvalidArgument, "Size exceeds buffer capacity"};
        }

        std::memcpy(dst, src.device_ptr, size);
        return {ErrorCode::Success, ""};
    }

    Result copy_device_to_device(Buffer& dst, const Buffer& src, size_t size) override {
        // Unified memory - just memcpy
        if (!dst.device_ptr || !src.device_ptr) {
            return {ErrorCode::InvalidArgument, "Invalid buffers"};
        }
        if (size > dst.size || size > src.size) {
            return {ErrorCode::InvalidArgument, "Size exceeds buffer capacity"};
        }

        std::memcpy(dst.device_ptr, src.device_ptr, size);
        return {ErrorCode::Success, ""};
    }

    // Async operations - Metal doesn't need explicit async for unified memory
    // but we use blit encoders for GPU-side copies
    Result copy_to_device_async(Buffer& dst, const void* src, size_t size, Stream& stream) override {
        // For unified memory, async copy is just memcpy
        return copy_to_device(dst, src, size);
    }

    Result copy_to_host_async(void* dst, const Buffer& src, size_t size, Stream& stream) override {
        return copy_to_host(dst, src, size);
    }

    // Stream management (command queues in Metal)
    Result create_stream(Stream& stream) override {
        @autoreleasepool {
            id<MTLDevice> device = devices_[current_device_];
            id<MTLCommandQueue> queue = [device newCommandQueue];
            if (!queue) {
                return {ErrorCode::NotInitialized, "Failed to create command queue"};
            }

            stream.native_handle = (__bridge_retained void*)queue;
            stream.device_id = current_device_;
        }
        return {ErrorCode::Success, ""};
    }

    void destroy_stream(Stream& stream) override {
        if (stream.native_handle) {
            @autoreleasepool {
                id<MTLCommandQueue> queue = (__bridge_transfer id<MTLCommandQueue>)stream.native_handle;
                // Released by ARC
                (void)queue;
            }
            stream.native_handle = nullptr;
        }
    }

    Result synchronize_stream(Stream& stream) override {
        // Metal synchronization happens at command buffer level
        // For stream sync, we commit and wait on a command buffer
        @autoreleasepool {
            id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)stream.native_handle;
            id<MTLCommandBuffer> buffer = [queue commandBuffer];
            [buffer commit];
            [buffer waitUntilCompleted];

            if (buffer.error) {
                return {ErrorCode::SyncFailed,
                    std::string([buffer.error.localizedDescription UTF8String])};
            }
        }
        return {ErrorCode::Success, ""};
    }

    Result synchronize_device() override {
        // Synchronize all command queues
        @autoreleasepool {
            for (id<MTLCommandQueue> queue : command_queues_) {
                id<MTLCommandBuffer> buffer = [queue commandBuffer];
                [buffer commit];
                [buffer waitUntilCompleted];
            }
        }
        return {ErrorCode::Success, ""};
    }

    // Event management
    Result create_event(Event& event) override {
        @autoreleasepool {
            id<MTLDevice> device = devices_[current_device_];
            id<MTLEvent> mtl_event = [device newEvent];
            if (!mtl_event) {
                return {ErrorCode::NotInitialized, "Failed to create Metal event"};
            }

            event.native_handle = (__bridge_retained void*)mtl_event;
            event.device_id = current_device_;
        }
        return {ErrorCode::Success, ""};
    }

    void destroy_event(Event& event) override {
        if (event.native_handle) {
            @autoreleasepool {
                id<MTLEvent> mtl_event = (__bridge_transfer id<MTLEvent>)event.native_handle;
                (void)mtl_event;
            }
            event.native_handle = nullptr;
        }
    }

    Result record_event(Event& event, Stream& stream) override {
        @autoreleasepool {
            id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)stream.native_handle;
            id<MTLEvent> mtl_event = (__bridge id<MTLEvent>)event.native_handle;

            id<MTLCommandBuffer> buffer = [queue commandBuffer];
            [buffer encodeSignalEvent:mtl_event value:++event_counter_];
            [buffer commit];
        }
        return {ErrorCode::Success, ""};
    }

    Result wait_event(Stream& stream, Event& event) override {
        @autoreleasepool {
            id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)stream.native_handle;
            id<MTLEvent> mtl_event = (__bridge id<MTLEvent>)event.native_handle;

            id<MTLCommandBuffer> buffer = [queue commandBuffer];
            [buffer encodeWaitForEvent:mtl_event value:event_counter_];
            [buffer commit];
        }
        return {ErrorCode::Success, ""};
    }

    Result synchronize_event(Event& event) override {
        // Metal events don't have direct CPU wait
        // We use a shared event for CPU synchronization
        @autoreleasepool {
            id<MTLDevice> device = devices_[current_device_];
            id<MTLSharedEvent> shared = [device newSharedEvent];

            id<MTLCommandQueue> queue = command_queues_[current_device_];
            id<MTLCommandBuffer> buffer = [queue commandBuffer];
            [buffer encodeSignalEvent:shared value:1];
            [buffer commit];
            [buffer waitUntilCompleted];
        }
        return {ErrorCode::Success, ""};
    }

    // Platform info
    std::string get_platform_name() const override { return "macOS"; }
    std::string get_backend_name() const override { return "Metal"; }

private:
    bool initialized_ = false;
    int current_device_ = 0;
    uint64_t event_counter_ = 0;

    std::vector<id<MTLDevice>> devices_;
    std::vector<id<MTLCommandQueue>> command_queues_;
    std::unordered_map<void*, id<MTLBuffer>> buffer_map_;
};

// Factory function
IPlatform& get_platform() {
    static MetalPlatform platform;
    static bool initialized = false;
    if (!initialized) {
        platform.initialize();
        initialized = true;
    }
    return platform;
}

}  // namespace platform
}  // namespace collider

#endif  // COLLIDER_PLATFORM_MACOS && COLLIDER_BACKEND_METAL
