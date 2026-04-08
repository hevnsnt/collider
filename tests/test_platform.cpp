/**
 * Platform Abstraction Tests
 *
 * Verifies platform detection, memory allocation, and basic operations
 * across CUDA, Metal, and CPU backends.
 */

#include "platform/platform.hpp"
#include <iostream>
#include <cassert>
#include <cstring>
#include <vector>

using namespace collider::platform;

void test_initialization() {
    std::cout << "Testing initialization... ";

    IPlatform& platform = get_platform();

    assert(platform.is_initialized());
    assert(platform.get_device_count() > 0);

    std::cout << "PASS\n";
    std::cout << "  Platform: " << platform.get_platform_name() << "\n";
    std::cout << "  Backend:  " << platform.get_backend_name() << "\n";
    std::cout << "  Devices:  " << platform.get_device_count() << "\n";
}

void test_device_info() {
    std::cout << "Testing device info... ";

    IPlatform& platform = get_platform();

    for (int i = 0; i < platform.get_device_count(); i++) {
        DeviceInfo info = platform.get_device_info(i);

        assert(info.device_id == i);
        assert(!info.name.empty());
        assert(info.total_memory > 0);

        std::cout << "\n  Device " << i << ": " << info.name << "\n";
        std::cout << "    Memory: " << (info.total_memory / 1024 / 1024) << " MB\n";
        std::cout << "    Compute: " << info.compute_major << "." << info.compute_minor << "\n";

        if (info.is_blackwell) std::cout << "    Architecture: Blackwell (RTX 5090)\n";
        else if (info.is_ampere) std::cout << "    Architecture: Ampere (RTX 30xx)\n";
        else if (info.is_apple_silicon) std::cout << "    Architecture: Apple Silicon\n";
    }

    std::cout << "PASS\n";
}

void test_memory_allocation() {
    std::cout << "Testing memory allocation... ";

    IPlatform& platform = get_platform();

    // Test basic allocation
    Buffer buffer;
    Result result = platform.allocate(buffer, 1024 * 1024, MemoryFlags::Default);
    assert(result.ok());
    assert(buffer.device_ptr != nullptr);
    assert(buffer.size == 1024 * 1024);

    platform.free(buffer);
    assert(buffer.device_ptr == nullptr);

    // Test pinned memory
    Buffer pinned;
    result = platform.allocate(pinned, 64 * 1024, MemoryFlags::Pinned);
    assert(result.ok());
    platform.free(pinned);

    // Test host-visible memory
    Buffer managed;
    result = platform.allocate(managed, 64 * 1024, MemoryFlags::HostVisible);
    assert(result.ok());
    platform.free(managed);

    std::cout << "PASS\n";
}

void test_memory_transfers() {
    std::cout << "Testing memory transfers... ";

    IPlatform& platform = get_platform();

    const size_t size = 4096;
    std::vector<uint8_t> host_src(size);
    std::vector<uint8_t> host_dst(size);

    // Fill source with pattern
    for (size_t i = 0; i < size; i++) {
        host_src[i] = static_cast<uint8_t>(i & 0xFF);
    }

    // Allocate device buffer
    Buffer device_buf;
    Result result = platform.allocate(device_buf, size, MemoryFlags::Default);
    assert(result.ok());

    // Copy to device
    result = platform.copy_to_device(device_buf, host_src.data(), size);
    assert(result.ok());

    // Copy back to host
    result = platform.copy_to_host(host_dst.data(), device_buf, size);
    assert(result.ok());

    // Verify data
    assert(std::memcmp(host_src.data(), host_dst.data(), size) == 0);

    platform.free(device_buf);

    std::cout << "PASS\n";
}

void test_streams() {
    std::cout << "Testing streams... ";

    IPlatform& platform = get_platform();

    // Create stream
    Stream stream;
    Result result = platform.create_stream(stream);
    assert(result.ok());
    assert(stream.native_handle != nullptr);

    // Test async operations
    const size_t size = 1024;
    std::vector<uint8_t> host_data(size, 0xAB);

    Buffer device_buf;
    result = platform.allocate(device_buf, size, MemoryFlags::Pinned);
    assert(result.ok());

    // Async copy to device
    result = platform.copy_to_device_async(device_buf, host_data.data(), size, stream);
    assert(result.ok());

    // Synchronize stream
    result = platform.synchronize_stream(stream);
    assert(result.ok());

    // Cleanup
    platform.free(device_buf);
    platform.destroy_stream(stream);

    std::cout << "PASS\n";
}

void test_events() {
    std::cout << "Testing events... ";

    IPlatform& platform = get_platform();

    // Create stream and event
    Stream stream;
    Event event;

    Result result = platform.create_stream(stream);
    assert(result.ok());

    result = platform.create_event(event);
    assert(result.ok());

    // Record event
    result = platform.record_event(event, stream);
    assert(result.ok());

    // Wait for event
    result = platform.synchronize_event(event);
    assert(result.ok());

    // Cleanup
    platform.destroy_event(event);
    platform.destroy_stream(stream);

    std::cout << "PASS\n";
}

void test_adaptive_config() {
    std::cout << "Testing adaptive configuration... ";

    IPlatform& platform = get_platform();

    for (int i = 0; i < platform.get_device_count(); i++) {
        DeviceInfo info = platform.get_device_info(i);
        AdaptiveConfig config = AdaptiveConfig::for_device(info);

        std::cout << "\n  Device " << i << " configuration:\n";
        std::cout << "    Batch size: " << config.candidates_per_batch << "\n";
        std::cout << "    Bloom filter: " << (config.bloom_filter_size / 1024 / 1024) << " MB\n";
        std::cout << "    Buffers: " << config.num_buffers << "\n";
        std::cout << "    Threads/block: " << config.threads_per_block << "\n";

        // Verify reasonable values
        assert(config.candidates_per_batch > 0);
        assert(config.bloom_filter_size > 0);
        assert(config.num_buffers >= 2);
    }

    std::cout << "PASS\n";
}

int main() {
    std::cout << "=== Superflayer Platform Tests ===\n\n";

    try {
        test_initialization();
        test_device_info();
        test_memory_allocation();
        test_memory_transfers();
        test_streams();
        test_events();
        test_adaptive_config();

        std::cout << "\n=== All tests passed! ===\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nTest FAILED: " << e.what() << "\n";
        return 1;
    }
}
