/**
 * Collider Logger
 *
 * File-based logging for crash diagnosis and overnight run monitoring.
 * Logs to ~/.collider/collider.log with timestamps and rotation.
 */

#pragma once

#include <string>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <mutex>
#include <cstdint>

namespace collider {

class Logger {
public:
    enum class Level {
        DEBUG,
        INFO,
        WARN,
        ERR,    // Named ERR to avoid Windows ERROR macro conflict
        FATAL
    };

    static Logger& instance() {
        static Logger logger;
        return logger;
    }

    bool init(const std::string& log_dir = "") {
        std::lock_guard<std::mutex> lock(mutex_);

        // Determine log directory
        std::string dir = log_dir;
        if (dir.empty()) {
            const char* home = nullptr;
#ifdef _WIN32
            home = std::getenv("USERPROFILE");
#else
            home = std::getenv("HOME");
#endif
            if (home) {
                dir = std::string(home) + "/.collider";
            } else {
                dir = ".";
            }
        }

        // Create directory if needed
        try {
            std::filesystem::create_directories(dir);
        } catch (...) {
            return false;
        }

        log_path_ = dir + "/collider.log";

        // Rotate log if too large (> 10MB)
        try {
            if (std::filesystem::exists(log_path_)) {
                auto size = std::filesystem::file_size(log_path_);
                if (size > 10 * 1024 * 1024) {
                    std::string backup = log_path_ + ".old";
                    std::filesystem::remove(backup);
                    std::filesystem::rename(log_path_, backup);
                }
            }
        } catch (...) {
            // Ignore rotation errors
        }

        // Open log file in append mode
        log_file_.open(log_path_, std::ios::app);
        if (!log_file_.is_open()) {
            return false;
        }

        initialized_ = true;

        // Log startup - write directly to avoid deadlock (we already hold the mutex)
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;

        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S")
           << "." << std::setfill('0') << std::setw(3) << ms.count()
           << " [INFO ] === Collider Logger Started ===\n";

        log_file_ << ss.str();
        log_file_.flush();

        return true;
    }

    void log(Level level, const std::string& message) {
        if (!initialized_) return;

        std::lock_guard<std::mutex> lock(mutex_);

        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;

        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S")
           << "." << std::setfill('0') << std::setw(3) << ms.count()
           << " [" << level_str(level) << "] "
           << message << "\n";

        log_file_ << ss.str();
        log_file_.flush();  // Always flush to ensure crash data is written
    }

    void log_startup(int puzzle_number, int gpu_count, const std::string& gpu_names,
                     uint64_t batch_size, const std::string& search_mode) {
        std::stringstream ss;
        ss << "STARTUP: Puzzle #" << puzzle_number
           << ", GPUs=" << gpu_count << " [" << gpu_names << "]"
           << ", BatchSize=" << (batch_size / 1'000'000) << "M"
           << ", Mode=" << search_mode;
        log(Level::INFO, ss.str());
    }

    void log_progress(uint64_t total_checked, double rate, int zone_idx, int total_zones) {
        std::stringstream ss;
        ss << "PROGRESS: Checked=" << total_checked
           << " (" << std::fixed << std::setprecision(1) << (rate / 1e6) << " M/s)"
           << ", Zone=" << (zone_idx + 1) << "/" << total_zones;
        log(Level::INFO, ss.str());
    }

    void log_zone_complete(int zone_idx, const std::string& zone_name, uint64_t keys_checked) {
        std::stringstream ss;
        ss << "ZONE_COMPLETE: Zone " << (zone_idx + 1) << " (" << zone_name << ")"
           << ", KeysChecked=" << keys_checked;
        log(Level::INFO, ss.str());
    }

    void log_shutdown(const std::string& reason, uint64_t total_checked, double elapsed_sec) {
        std::stringstream ss;
        ss << "SHUTDOWN: Reason=" << reason
           << ", TotalChecked=" << total_checked
           << ", ElapsedSec=" << std::fixed << std::setprecision(1) << elapsed_sec;
        log(Level::INFO, ss.str());
    }

    void log_error(const std::string& error_msg) {
        log(Level::ERR, "ERROR: " + error_msg);
    }

    void log_gpu_error(int device_id, const std::string& error_msg) {
        std::stringstream ss;
        ss << "GPU_ERROR: Device " << device_id << " - " << error_msg;
        log(Level::ERR, ss.str());
    }

    void log_state_save(int puzzle_number, int zone_idx, uint64_t position_lo, uint64_t position_hi) {
        std::stringstream ss;
        ss << "STATE_SAVE: Puzzle=" << puzzle_number
           << ", Zone=" << zone_idx
           << ", Position=0x" << std::hex << position_hi << std::setfill('0') << std::setw(16) << position_lo;
        log(Level::INFO, ss.str());
    }

    void log_found(uint64_t key_lo, uint64_t key_hi, const std::string& address) {
        std::stringstream ss;
        ss << "FOUND: Key=0x" << std::hex << key_hi << std::setfill('0') << std::setw(16) << key_lo
           << ", Address=" << address;
        log(Level::INFO, ss.str());
    }

    std::string get_log_path() const { return log_path_; }

    ~Logger() {
        if (initialized_) {
            log(Level::INFO, "=== Collider Logger Stopped ===");
            log_file_.close();
        }
    }

private:
    Logger() : initialized_(false) {}

    // Delete copy/move
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    Logger(Logger&&) = delete;
    Logger& operator=(Logger&&) = delete;

    static const char* level_str(Level level) {
        switch (level) {
            case Level::DEBUG: return "DEBUG";
            case Level::INFO:  return "INFO ";
            case Level::WARN:  return "WARN ";
            case Level::ERR:   return "ERROR";
            case Level::FATAL: return "FATAL";
            default: return "?????";
        }
    }

    bool initialized_;
    std::string log_path_;
    std::ofstream log_file_;
    std::mutex mutex_;
};

// Convenience macros
#define LOG_INFO(msg)  collider::Logger::instance().log(collider::Logger::Level::INFO, msg)
#define LOG_WARN(msg)  collider::Logger::instance().log(collider::Logger::Level::WARN, msg)
#define LOG_ERROR(msg) collider::Logger::instance().log(collider::Logger::Level::ERR, msg)
#define LOG_DEBUG(msg) collider::Logger::instance().log(collider::Logger::Level::DEBUG, msg)

}  // namespace collider
