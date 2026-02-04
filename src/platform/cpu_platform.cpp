/**
 * CPU Platform Implementation
 *
 * Fallback implementation for testing and development.
 * Provides same interface as GPU backends but runs on CPU.
 *
 * Features:
 * - Thread pool for parallel execution
 * - Standard memory allocation (malloc/free)
 * - Compatible with all platforms
 */

#include "platform.hpp"

#if defined(COLLIDER_BACKEND_CPU)

#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <atomic>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/sysinfo.h>
#include <unistd.h>
#endif

namespace collider {
namespace platform {

/**
 * Simple thread pool for CPU compute simulation.
 */
class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads) : stop_(false) {
        for (size_t i = 0; i < num_threads; i++) {
            workers_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                        if (stop_ && tasks_.empty()) return;
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    task();
                }
            });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    template<typename F>
    void enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            tasks_.emplace(std::forward<F>(f));
        }
        cv_.notify_one();
    }

    void wait() {
        // Simple busy-wait for all tasks to complete
        while (true) {
            std::unique_lock<std::mutex> lock(mutex_);
            if (tasks_.empty()) break;
            lock.unlock();
            std::this_thread::yield();
        }
    }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_;
};

/**
 * CPU Stream (simulated with thread pool task group)
 */
struct CPUStream {
    std::atomic<uint64_t> pending_tasks{0};
    std::mutex mutex;
    std::condition_variable cv;
};

/**
 * CPU Event (simple flag)
 */
struct CPUEvent {
    std::atomic<bool> signaled{false};
    std::mutex mutex;
    std::condition_variable cv;
};

/**
 * CPU Platform Implementation
 */
class CPUPlatform : public IPlatform {
public:
    CPUPlatform() = default;
    ~CPUPlatform() override { shutdown(); }

    Result initialize() override {
        if (initialized_) {
            return {ErrorCode::Success, "Already initialized"};
        }

        // Detect CPU info
        num_threads_ = std::thread::hardware_concurrency();
        if (num_threads_ == 0) num_threads_ = 4;  // Fallback

        // Get system memory
#ifdef _WIN32
        MEMORYSTATUSEX status;
        status.dwLength = sizeof(status);
        GlobalMemoryStatusEx(&status);
        total_memory_ = status.ullTotalPhys;
        available_memory_ = status.ullAvailPhys;
#elif defined(__APPLE__)
        // macOS
        int64_t mem;
        size_t len = sizeof(mem);
        sysctlbyname("hw.memsize", &mem, &len, nullptr, 0);
        total_memory_ = mem;
        available_memory_ = mem / 2;  // Estimate
#else
        // Linux
        struct sysinfo info;
        if (sysinfo(&info) == 0) {
            total_memory_ = info.totalram * info.mem_unit;
            available_memory_ = info.freeram * info.mem_unit;
        }
#endif

        // Create thread pool
        thread_pool_ = std::make_unique<ThreadPool>(num_threads_);

        initialized_ = true;
        return {ErrorCode::Success, ""};
    }

    void shutdown() override {
        if (!initialized_) return;

        // Free all allocated buffers
        for (auto ptr : allocations_) {
            std::free(ptr);
        }
        allocations_.clear();

        // Destroy streams and events
        for (auto stream : streams_) {
            delete static_cast<CPUStream*>(stream);
        }
        streams_.clear();

        for (auto event : events_) {
            delete static_cast<CPUEvent*>(event);
        }
        events_.clear();

        thread_pool_.reset();
        initialized_ = false;
    }

    bool is_initialized() const override { return initialized_; }

    // Device management (CPU has 1 "device")
    int get_device_count() const override { return 1; }

    DeviceInfo get_device_info(int device_id) const override {
        DeviceInfo info = {};
        if (device_id != 0) return info;

        info.device_id = 0;
        info.name = "CPU Fallback";
        info.vendor = "Generic";

        info.total_memory = total_memory_;
        info.available_memory = available_memory_;

        info.compute_major = 0;
        info.compute_minor = 0;

        info.is_blackwell = false;
        info.is_ampere = false;
        info.is_apple_silicon = false;
        info.supports_fp16 = false;
        info.supports_int8 = true;

        info.multiprocessor_count = static_cast<int>(num_threads_);
        info.max_threads_per_block = 1;
        info.warp_size = 1;
        info.shared_memory_per_block = 0;
        info.l2_cache_size = 0;

        return info;
    }

    Result set_device(int device_id) override {
        if (device_id != 0) {
            return {ErrorCode::InvalidDevice, "CPU has only device 0"};
        }
        return {ErrorCode::Success, ""};
    }

    int get_current_device() const override { return 0; }

    // Memory management
    Result allocate(Buffer& buffer, size_t size, MemoryFlags flags) override {
        // Allocate aligned memory
        void* ptr = nullptr;

#ifdef _WIN32
        ptr = _aligned_malloc(size, 128);
#else
        if (posix_memalign(&ptr, 128, size) != 0) {
            ptr = nullptr;
        }
#endif

        if (!ptr) {
            return {ErrorCode::OutOfMemory, "Failed to allocate memory"};
        }

        buffer.device_ptr = ptr;
        buffer.host_ptr = ptr;  // Same for CPU
        buffer.size = size;
        buffer.flags = flags;
        buffer.device_id = 0;

        std::lock_guard<std::mutex> lock(alloc_mutex_);
        allocations_.push_back(ptr);

        return {ErrorCode::Success, ""};
    }

    void free(Buffer& buffer) override {
        if (buffer.device_ptr) {
            std::lock_guard<std::mutex> lock(alloc_mutex_);
            allocations_.erase(
                std::remove(allocations_.begin(), allocations_.end(), buffer.device_ptr),
                allocations_.end()
            );

#ifdef _WIN32
            _aligned_free(buffer.device_ptr);
#else
            std::free(buffer.device_ptr);
#endif

            buffer.device_ptr = nullptr;
            buffer.host_ptr = nullptr;
            buffer.size = 0;
        }
    }

    Result copy_to_device(Buffer& dst, const void* src, size_t size) override {
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
        if (!dst.device_ptr || !src.device_ptr) {
            return {ErrorCode::InvalidArgument, "Invalid buffers"};
        }
        if (size > dst.size || size > src.size) {
            return {ErrorCode::InvalidArgument, "Size exceeds buffer capacity"};
        }
        std::memcpy(dst.device_ptr, src.device_ptr, size);
        return {ErrorCode::Success, ""};
    }

    // Async operations (simulated with thread pool)
    Result copy_to_device_async(Buffer& dst, const void* src, size_t size, Stream& stream) override {
        CPUStream* cpu_stream = static_cast<CPUStream*>(stream.native_handle);
        cpu_stream->pending_tasks++;

        // Capture by value for async
        void* dst_ptr = dst.device_ptr;
        thread_pool_->enqueue([dst_ptr, src, size, cpu_stream] {
            std::memcpy(dst_ptr, src, size);
            cpu_stream->pending_tasks--;
            cpu_stream->cv.notify_all();
        });

        return {ErrorCode::Success, ""};
    }

    Result copy_to_host_async(void* dst, const Buffer& src, size_t size, Stream& stream) override {
        CPUStream* cpu_stream = static_cast<CPUStream*>(stream.native_handle);
        cpu_stream->pending_tasks++;

        void* src_ptr = src.device_ptr;
        thread_pool_->enqueue([dst, src_ptr, size, cpu_stream] {
            std::memcpy(dst, src_ptr, size);
            cpu_stream->pending_tasks--;
            cpu_stream->cv.notify_all();
        });

        return {ErrorCode::Success, ""};
    }

    // Stream management
    Result create_stream(Stream& stream) override {
        CPUStream* cpu_stream = new CPUStream();

        stream.native_handle = cpu_stream;
        stream.device_id = 0;

        std::lock_guard<std::mutex> lock(stream_mutex_);
        streams_.push_back(cpu_stream);

        return {ErrorCode::Success, ""};
    }

    void destroy_stream(Stream& stream) override {
        if (stream.native_handle) {
            CPUStream* cpu_stream = static_cast<CPUStream*>(stream.native_handle);

            std::lock_guard<std::mutex> lock(stream_mutex_);
            streams_.erase(
                std::remove(streams_.begin(), streams_.end(), cpu_stream),
                streams_.end()
            );

            delete cpu_stream;
            stream.native_handle = nullptr;
        }
    }

    Result synchronize_stream(Stream& stream) override {
        CPUStream* cpu_stream = static_cast<CPUStream*>(stream.native_handle);
        if (!cpu_stream) {
            return {ErrorCode::InvalidArgument, "Invalid stream"};
        }

        std::unique_lock<std::mutex> lock(cpu_stream->mutex);
        cpu_stream->cv.wait(lock, [cpu_stream] {
            return cpu_stream->pending_tasks == 0;
        });

        return {ErrorCode::Success, ""};
    }

    Result synchronize_device() override {
        // Wait for all streams
        for (auto stream : streams_) {
            CPUStream* cpu_stream = static_cast<CPUStream*>(stream);
            std::unique_lock<std::mutex> lock(cpu_stream->mutex);
            cpu_stream->cv.wait(lock, [cpu_stream] {
                return cpu_stream->pending_tasks == 0;
            });
        }
        return {ErrorCode::Success, ""};
    }

    // Event management
    Result create_event(Event& event) override {
        CPUEvent* cpu_event = new CPUEvent();

        event.native_handle = cpu_event;
        event.device_id = 0;

        std::lock_guard<std::mutex> lock(event_mutex_);
        events_.push_back(cpu_event);

        return {ErrorCode::Success, ""};
    }

    void destroy_event(Event& event) override {
        if (event.native_handle) {
            CPUEvent* cpu_event = static_cast<CPUEvent*>(event.native_handle);

            std::lock_guard<std::mutex> lock(event_mutex_);
            events_.erase(
                std::remove(events_.begin(), events_.end(), cpu_event),
                events_.end()
            );

            delete cpu_event;
            event.native_handle = nullptr;
        }
    }

    Result record_event(Event& event, Stream& stream) override {
        CPUEvent* cpu_event = static_cast<CPUEvent*>(event.native_handle);
        CPUStream* cpu_stream = static_cast<CPUStream*>(stream.native_handle);

        if (!cpu_event || !cpu_stream) {
            return {ErrorCode::InvalidArgument, "Invalid event or stream"};
        }

        cpu_event->signaled = false;

        // Record event when stream becomes idle
        thread_pool_->enqueue([cpu_event, cpu_stream] {
            std::unique_lock<std::mutex> lock(cpu_stream->mutex);
            cpu_stream->cv.wait(lock, [cpu_stream] {
                return cpu_stream->pending_tasks == 0;
            });
            cpu_event->signaled = true;
            cpu_event->cv.notify_all();
        });

        return {ErrorCode::Success, ""};
    }

    Result wait_event(Stream& stream, Event& event) override {
        CPUEvent* cpu_event = static_cast<CPUEvent*>(event.native_handle);
        CPUStream* cpu_stream = static_cast<CPUStream*>(stream.native_handle);

        if (!cpu_event || !cpu_stream) {
            return {ErrorCode::InvalidArgument, "Invalid event or stream"};
        }

        cpu_stream->pending_tasks++;

        thread_pool_->enqueue([cpu_event, cpu_stream] {
            std::unique_lock<std::mutex> lock(cpu_event->mutex);
            cpu_event->cv.wait(lock, [cpu_event] {
                return cpu_event->signaled.load();
            });
            cpu_stream->pending_tasks--;
            cpu_stream->cv.notify_all();
        });

        return {ErrorCode::Success, ""};
    }

    Result synchronize_event(Event& event) override {
        CPUEvent* cpu_event = static_cast<CPUEvent*>(event.native_handle);
        if (!cpu_event) {
            return {ErrorCode::InvalidArgument, "Invalid event"};
        }

        std::unique_lock<std::mutex> lock(cpu_event->mutex);
        cpu_event->cv.wait(lock, [cpu_event] {
            return cpu_event->signaled.load();
        });

        return {ErrorCode::Success, ""};
    }

    // Platform info
    std::string get_platform_name() const override {
#ifdef _WIN32
        return "Windows";
#elif defined(__APPLE__)
        return "macOS";
#else
        return "Linux";
#endif
    }

    std::string get_backend_name() const override { return "CPU"; }

private:
    bool initialized_ = false;
    size_t num_threads_ = 0;
    size_t total_memory_ = 0;
    size_t available_memory_ = 0;

    std::unique_ptr<ThreadPool> thread_pool_;

    std::vector<void*> allocations_;
    std::mutex alloc_mutex_;

    std::vector<void*> streams_;
    std::mutex stream_mutex_;

    std::vector<void*> events_;
    std::mutex event_mutex_;
};

// Factory function
IPlatform& get_platform() {
    static CPUPlatform platform;
    static bool initialized = false;
    if (!initialized) {
        platform.initialize();
        initialized = true;
    }
    return platform;
}

}  // namespace platform
}  // namespace collider

#endif  // COLLIDER_BACKEND_CPU
