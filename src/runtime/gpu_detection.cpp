/**
 * gpu_detection.cpp - Implementation of pre-dispatch GPU detection.
 *
 * Extracted verbatim from src/main.cpp during the v1.4.1 A.3 (6/6)
 * refactor; no behavior changes.
 */
#include "runtime/gpu_detection.hpp"

#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "platform/platform.hpp"

GPUDetectionResult detect_gpus(std::vector<int>& requested_ids) {
    GPUDetectionResult result;
    result.device_count = 0;
    result.estimated_speed = 0;
    result.backend = "CPU";

    // Flush stdout before platform init - prevents buffered output from
    // appearing after Metal/CUDA initialization messages
    std::cout << std::flush;

    try {
        auto& platform = collider::platform::get_platform();
        auto init_result = platform.initialize();

        if (init_result.code == collider::platform::ErrorCode::Success) {
            result.backend = platform.get_backend_name();
            int total_devices = platform.get_device_count();

            // Auto-detect all GPUs if none specified
            if (requested_ids.empty() && total_devices > 0) {
                for (int i = 0; i < total_devices; i++) {
                    requested_ids.push_back(i);
                }
            }

            std::vector<std::string> names;
            for (int id : requested_ids) {
                if (id < total_devices) {
                    auto info = platform.get_device_info(id);
                    names.push_back(info.name);
                    result.device_count++;

                    // Estimate speed based on GPU type
                    // NOTE: These estimates are for EC scalar multiply (puzzle search)
                    // which is ~100x slower than SHA256 due to modular arithmetic.
                    // Optimized implementations with precomputed tables can be 10-20x faster.
                    if (info.is_apple_silicon) {
                        // Apple Silicon estimates for naive EC multiply
                        if (info.name.find("M3 Max") != std::string::npos) {
                            result.estimated_speed += 8'000'000;   // ~8M/s
                        } else if (info.name.find("M3 Pro") != std::string::npos) {
                            result.estimated_speed += 6'000'000;   // ~6M/s
                        } else if (info.name.find("M3") != std::string::npos) {
                            result.estimated_speed += 4'000'000;   // ~4M/s
                        } else if (info.name.find("M2") != std::string::npos) {
                            result.estimated_speed += 3'000'000;   // ~3M/s
                        } else {
                            result.estimated_speed += 2'000'000;   // ~2M/s
                        }
                    } else if (info.is_blackwell) {
                        // RTX 5090 estimate
                        result.estimated_speed += 80'000'000;      // ~80M/s
                    } else if (info.is_ampere) {
                        // Ampere estimates (RTX 30xx series)
                        if (info.name.find("3090") != std::string::npos) {
                            result.estimated_speed += 20'000'000;  // ~20M/s
                        } else if (info.name.find("3080") != std::string::npos) {
                            result.estimated_speed += 15'000'000;  // ~15M/s
                        } else if (info.name.find("3070") != std::string::npos) {
                            result.estimated_speed += 10'000'000;  // ~10M/s
                        } else {
                            result.estimated_speed += 5'000'000;   // ~5M/s (3060)
                        }
                    } else {
                        // Ada Lovelace (RTX 40xx series)
                        if (info.name.find("4090") != std::string::npos) {
                            result.estimated_speed += 50'000'000;  // ~50M/s
                        } else if (info.name.find("4080") != std::string::npos) {
                            result.estimated_speed += 35'000'000;  // ~35M/s
                        } else {
                            result.estimated_speed += 25'000'000;  // ~25M/s
                        }
                    }
                }
            }

            // Format GPU names
            if (names.empty()) {
                result.gpu_names = "No GPUs detected";
            } else if (names.size() == 1) {
                result.gpu_names = names[0];
            } else {
                // Check if all same
                bool all_same = true;
                for (size_t i = 1; i < names.size(); i++) {
                    if (names[i] != names[0]) {
                        all_same = false;
                        break;
                    }
                }
                if (all_same) {
                    result.gpu_names = std::to_string(names.size()) + "x " + names[0];
                } else {
                    result.gpu_names = names[0];
                    for (size_t i = 1; i < names.size(); i++) {
                        result.gpu_names += ", " + names[i];
                    }
                }
            }
        } else {
            result.gpu_names = "GPU init failed: " + init_result.message;
        }
    } catch (const std::exception& e) {
        result.gpu_names = std::string("Detection error: ") + e.what();
    }

    // CPU fallback estimation
    if (result.device_count == 0 || result.estimated_speed == 0) {
        result.device_count = 1;
        result.gpu_names = "CPU (reference mode)";
        result.estimated_speed = 10'000;  // 10K/s for CPU reference
        result.backend = "CPU";
    }

    return result;
}
