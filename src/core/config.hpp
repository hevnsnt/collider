/**
 * Collider Configuration
 *
 * Handles persistent configuration for puzzle solving.
 * Config file: ~/.collider/config
 */

#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <iostream>
#include <map>

namespace collider {

struct UserConfig {
    int default_puzzle = 71;       // Default puzzle number to target
    bool save_progress = true;     // Save progress for resume

    // GPU Calibration results (device_id -> optimal batch size)
    std::map<int, uint64_t> gpu_batch_sizes;
    bool calibration_done = false;

    /**
     * Get optimal batch size for a GPU.
     * Returns 0 if not calibrated (use default).
     */
    uint64_t get_gpu_batch_size(int device_id) const {
        auto it = gpu_batch_sizes.find(device_id);
        return (it != gpu_batch_sizes.end()) ? it->second : 0;
    }

    /**
     * Set optimal batch size for a GPU.
     */
    void set_gpu_batch_size(int device_id, uint64_t batch_size) {
        gpu_batch_sizes[device_id] = batch_size;
        calibration_done = true;
    }

    /**
     * Get config file path.
     */
    static std::string get_config_path() {
        std::string home;
#ifdef _WIN32
        const char* userprofile = std::getenv("USERPROFILE");
        home = userprofile ? userprofile : ".";
#else
        const char* home_env = std::getenv("HOME");
        home = home_env ? home_env : ".";
#endif
        return home + "/.collider/config";
    }

    /**
     * Load config from file.
     */
    bool load() {
        std::string path = get_config_path();

        std::ifstream file(path);
        if (!file.is_open()) {
            return false;  // No config file yet
        }

        std::string line;
        while (std::getline(file, line)) {
            // Skip comments and empty lines
            if (line.empty() || line[0] == '#') continue;

            // Parse key=value
            auto pos = line.find('=');
            if (pos == std::string::npos) continue;

            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);

            // Trim whitespace
            while (!key.empty() && (key.back() == ' ' || key.back() == '\t')) key.pop_back();
            while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) value.erase(0, 1);

            if (key == "default_puzzle") {
                default_puzzle = std::stoi(value);
            } else if (key == "save_progress") {
                save_progress = (value == "true" || value == "1");
            } else if (key == "calibration_done") {
                calibration_done = (value == "true" || value == "1");
            } else if (key.length() > 10 && key.substr(0, 10) == "gpu_batch_") {
                // Parse gpu_batch_<device_id>=<batch_size>
                int device_id = std::stoi(key.substr(10));
                gpu_batch_sizes[device_id] = std::stoull(value);
            }
        }

        return true;
    }

    /**
     * Save config to file.
     */
    bool save() const {
        std::string path = get_config_path();

        // Create directory if needed
        std::filesystem::path dir = std::filesystem::path(path).parent_path();
        if (!std::filesystem::exists(dir)) {
            try {
                std::filesystem::create_directories(dir);
            } catch (const std::exception& e) {
                std::cerr << "[!] Failed to create config directory: " << e.what() << "\n";
                return false;
            }
        }

        std::ofstream file(path);
        if (!file.is_open()) {
            std::cerr << "[!] Failed to open config file for writing: " << path << "\n";
            return false;
        }

        file << "# Collider Configuration\n";
        file << "# Bitcoin Puzzle Solver\n";
        file << "\n";
        file << "# Default puzzle number to solve (66-160)\n";
        file << "default_puzzle=" << default_puzzle << "\n";
        file << "\n";
        file << "# Save progress for resume after interruption\n";
        file << "save_progress=" << (save_progress ? "true" : "false") << "\n";
        file << "\n";
        file << "# GPU Calibration Results (auto-detected optimal batch sizes)\n";
        file << "calibration_done=" << (calibration_done ? "true" : "false") << "\n";
        for (const auto& [device_id, batch_size] : gpu_batch_sizes) {
            file << "gpu_batch_" << device_id << "=" << batch_size << "\n";
        }

        return true;
    }
};

}  // namespace collider
