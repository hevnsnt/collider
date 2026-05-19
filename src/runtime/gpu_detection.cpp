/**
 * gpu_detection.cpp - Implementation of pre-dispatch GPU detection.
 *
 * Extracted verbatim from src/main.cpp during the v1.4.1 A.3 (6/6)
 * refactor; no behavior changes.
 */
#include "runtime/gpu_detection.hpp"

#include <algorithm>
#include <exception>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "platform/platform.hpp"

// T4.2: parse the compile-time CUDA arch list ("7.5,8.6,8.9,12.0") into a
// set of (major,minor) pairs. The CMakeLists.txt build wires this through
// as COLLIDER_CUDA_ARCH_LIST; absence means the binary was built without
// CUDA (Metal / CPU path) and the SM coverage check is a no-op.
//
// External linkage (declared in runtime/gpu_detection.hpp) so the session
// log's hardware enumeration can fire a milestone("sm_mismatch", ...) using
// the same parser that drives the startup stderr warning here. Duplicating
// the parser would risk the two emitting different verdicts after any
// future COLLIDER_CUDA_ARCH_LIST format change.
std::set<std::pair<int, int>> compile_time_sm_set() {
    std::set<std::pair<int, int>> out;
#ifdef COLLIDER_CUDA_ARCH_LIST
    const std::string list = COLLIDER_CUDA_ARCH_LIST;
    std::stringstream ss(list);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        auto dot = tok.find('.');
        if (dot == std::string::npos) continue;
        try {
            int major = std::stoi(tok.substr(0, dot));
            int minor = std::stoi(tok.substr(dot + 1));
            out.emplace(major, minor);
        } catch (...) {
            // Bad token: skip silently. The CMake formatter only emits
            // well-formed "X.Y" pairs, so a malformed entry here would
            // mean a build-system bug we can't address from runtime.
        }
    }
#endif
    return out;
}

// Render the SM set as "{7.5, 8.6, 8.9, 12.0}" for the warning banner and
// the session-log milestone. Shared so both formats stay in lockstep.
std::string sm_set_to_string(const std::set<std::pair<int, int>>& s) {
    std::string out = "{";
    bool first = true;
    for (const auto& sm : s) {
        if (!first) out += ", ";
        out += std::to_string(sm.first) + "." + std::to_string(sm.second);
        first = false;
    }
    out += "}";
    return out;
}

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

            // T4.2: compile-time CUDA arch set (parsed once per call).
            // Empty set means non-CUDA build; the per-device warning loop
            // below skips silently in that case.
            const auto compiled_sm_set = compile_time_sm_set();

            std::vector<std::string> names;
            for (int id : requested_ids) {
                if (id < total_devices) {
                    auto info = platform.get_device_info(id);
                    names.push_back(info.name);
                    result.device_count++;

                    // T4.2: warn if this device's SM is missing from the
                    // compile-time arch list. CUDA's PTX JIT will still
                    // produce a runnable kernel from the embedded PTX, but
                    // the JIT step is multi-second on first launch and the
                    // generated code is slower than a native SM-target
                    // build. We do this BEFORE the heuristic so the
                    // operator sees the warning even on apple/cpu builds
                    // (where compiled_sm_set is empty and the loop body
                    // short-circuits).
                    if (!compiled_sm_set.empty() && !info.is_apple_silicon) {
                        std::pair<int, int> device_sm{info.compute_major,
                                                       info.compute_minor};
                        if (compiled_sm_set.find(device_sm) ==
                            compiled_sm_set.end()) {
                            std::cerr << "WARNING: device " << id
                                      << " (" << info.name
                                      << ") reports SM " << device_sm.first
                                      << "." << device_sm.second
                                      << "; binary was built for "
                                      << sm_set_to_string(compiled_sm_set)
                                      << "; PTX JIT will be slow or may "
                                         "fail.\n";
                        }
                    }

                    // Estimate speed based on GPU type.
                    // NOTE: These estimates are for EC scalar multiply
                    // (puzzle search) which is ~100x slower than SHA256
                    // due to modular arithmetic. Optimized implementations
                    // with precomputed tables can be 10-20x faster.
                    if (info.is_apple_silicon) {
                        // Apple Silicon estimates for naive EC multiply.
                        // More-specific names (Max / Pro) must be matched
                        // before the bare "M3" / "M4" substring to avoid
                        // mis-classifying an "Apple M4 Pro" device as
                        // plain M4.
                        if (info.name.find("M4 Max") != std::string::npos) {
                            result.estimated_speed += 9'000'000;   // ~9M/s (32-40 core GPU)
                        } else if (info.name.find("M4 Pro") != std::string::npos) {
                            result.estimated_speed += 6'500'000;   // ~6.5M/s (16-20 core GPU)
                        } else if (info.name.find("M4") != std::string::npos) {
                            result.estimated_speed += 4'500'000;   // ~4.5M/s (10 core GPU)
                        } else if (info.name.find("M3 Max") != std::string::npos) {
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
                    } else if (info.name.find("4090") != std::string::npos) {
                        // Ada Lovelace, named:
                        result.estimated_speed += 50'000'000;      // ~50M/s
                    } else if (info.name.find("4080") != std::string::npos) {
                        result.estimated_speed += 35'000'000;      // ~35M/s
                    } else if (info.compute_major == 8 &&
                               info.compute_minor == 9) {
                        // Unknown Ada (e.g. 4070 / mobile 4060): default
                        // to 25M instead of the previous catch-all that
                        // misapplied 4090-class numbers to every unknown
                        // card.
                        result.estimated_speed += 25'000'000;
                    } else {
                        // T4.7: previous fall-through dumped 25M (Ada-
                        // class) onto any unknown card, which over-
                        // counts Turing / Pascal / older Maxwell rigs
                        // by 5-10x. Replace with an SM-count-based
                        // heuristic that's correct in order of magnitude
                        // across the lineup. Per-SM throughput numbers
                        // are anchored to the published-name estimates
                        // above (Ampere 3090: 20M/82 SMs = ~244K,
                        // Ada 4090: 50M/128 = ~390K, Blackwell 5090:
                        // 80M/170 = ~470K). Turing (SM 7.x): ~30K/SM
                        // per the 2060 SUPER reference (34 SMs, ~1M/s).
                        // Hopper / datacenter (SM 9.x): treat as Ada
                        // grade. Older Maxwell / Pascal (SM <= 6.x):
                        // ~25K/SM. Unknown SM 10+ (datacenter
                        // Blackwell): use the Blackwell per-SM rate.
                        const int sms = info.multiprocessor_count;
                        const int major = info.compute_major;
                        uint64_t per_sm = 0;
                        if (major >= 10) {
                            per_sm = 470'000;       // Blackwell-class
                        } else if (major == 9 ||
                                   (major == 8 && info.compute_minor == 9)) {
                            per_sm = 390'000;       // Hopper / Ada
                        } else if (major == 8) {
                            per_sm = 244'000;       // Ampere
                        } else if (major == 7) {
                            per_sm = 30'000;        // Turing / Volta
                        } else {
                            per_sm = 25'000;        // Pascal and older
                        }
                        uint64_t est = static_cast<uint64_t>(sms) * per_sm;
                        // Floor at 1M/s so an unrecognized card still
                        // contributes a non-trivial estimate; this
                        // matches the prior behavior's "this is not
                        // CPU-class" signal without blindly inheriting
                        // the 25M Ada-class default.
                        if (est < 1'000'000) est = 1'000'000;
                        result.estimated_speed += est;
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
