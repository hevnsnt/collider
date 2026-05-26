/**
 * Collider Logger
 *
 * File-based logging for crash diagnosis and overnight run monitoring.
 * Logs to ~/.collider/collider.log with timestamps and rotation.
 */

#pragma once

#include "paths.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <ctime>      // localtime_r / localtime_s (track-f F-17)
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>

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
            // paths::collider_home() returns "./.collider" as last-resort
            // fallback when neither USERPROFILE nor HOME is set, matching the
            // prior open-coded behaviour (home defaulted to "." then suffix
            // "/.collider" was appended).
            dir = collider::paths::collider_home().string();
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

        // log_file_ is now a shared_ptr so background threads
        // that grab a snapshot at the start of log() keep the underlying
        // ofstream alive even if the destructor races and tries to tear the
        // singleton down. The shared_ptr is swapped under mutex_ on shutdown.
        auto fresh = std::make_shared<std::ofstream>();
        fresh->open(log_path_, std::ios::app);
        if (!fresh->is_open()) {
            return false;
        }
        log_file_ = std::move(fresh);

        initialized_.store(true, std::memory_order_release);

        // Log startup - write directly to avoid deadlock (we already hold the mutex)
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;

        std::tm tm_buf{};
#ifdef _WIN32
        localtime_s(&tm_buf, &time_t);
#else
        localtime_r(&time_t, &tm_buf);
#endif

        std::stringstream ss;
        ss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S")
           << "." << std::setfill('0') << std::setw(3) << ms.count()
           << " [INFO ] === Collider Logger Started ===\n";

        (*log_file_) << ss.str();
        log_file_->flush();

        return true;
    }

    void log(Level level, const std::string& message) {
        // fix: take the mutex BEFORE inspecting initialized_ /
        // log_file_, and snapshot log_file_ as a shared_ptr local. That
        // prevents the destructor from tearing down log_file_ while we are
        // mid-write. The mutex pairs with the destructor's mutex acquire
        // before it clears initialized_ and resets log_file_.
        std::shared_ptr<std::ofstream> file;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!initialized_.load(std::memory_order_acquire)) return;
            file = log_file_;
        }
        if (!file || !file->is_open()) return;

        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;

        // localtime_r is the thread-safe variant. The legacy
        // std::localtime returns a pointer into a single static buffer that
        // can be torn between concurrent callers (Logger thread + main).
        std::tm tm_buf{};
#ifdef _WIN32
        localtime_s(&tm_buf, &time_t);
#else
        localtime_r(&time_t, &tm_buf);
#endif

        std::stringstream ss;
        ss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S")
           << "." << std::setfill('0') << std::setw(3) << ms.count()
           << " [" << level_str(level) << "] "
           << message << "\n";

        // Re-take the mutex for the actual stream write so concurrent log()
        // calls do not interleave their output. The shared_ptr we hold
        // (`file`) keeps the ofstream alive even if the destructor moved
        // log_file_ out from under us between the two mutex critical
        // sections.
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_.load(std::memory_order_acquire)) return;
        (*file) << ss.str();
        file->flush();  // Always flush to ensure crash data is written
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

    // dropped zone_idx/total_zones parameters together with the
    // Center-Heavy scanning strategy that produced them.
    void log_progress(uint64_t total_checked, double rate) {
        std::stringstream ss;
        ss << "PROGRESS: Checked=" << total_checked
           << " (" << std::fixed << std::setprecision(1) << (rate / 1e6) << " M/s)";
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

    void log_state_save(int puzzle_number, uint64_t position_lo, uint64_t position_hi) {
        std::stringstream ss;
        ss << "STATE_SAVE: Puzzle=" << puzzle_number
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
        // No mutex here on purpose (see runtime/balance.cpp::~BalanceFetcher
        // and core/session_log.cpp::~SessionLogSink for the same pattern).
        // This is a Meyers-singleton destructor running during static
        // teardown after main(). On macOS, Apple's pthread library
        // invalidates mutex internal state during teardown earlier than
        // glibc/MSVC, so taking lock_guard<mutex> here throws EINVAL via
        // std::system_error -- propagates uncaught and aborts the process
        // AFTER the run finished cleanly. Customer-facing log: clean
        // session summary followed by abort 6 noise. Drop the lock.
        //
        // Safety: by the time we reach this dtor no other thread should
        // still be calling log() -- main returned, workers joined. The
        // log_file_ shared_ptr copy is a single 8-byte read; even if a
        // last-gasp logger thread is racing, worst case is a missed
        // final line, not a crash.
        if (initialized_.load(std::memory_order_acquire)) {
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()) % 1000;
            std::tm tm_buf{};
#ifdef _WIN32
            localtime_s(&tm_buf, &time_t);
#else
            localtime_r(&time_t, &tm_buf);
#endif
            std::stringstream ss;
            ss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S")
               << "." << std::setfill('0') << std::setw(3) << ms.count()
               << " [INFO ] === Collider Logger Stopped ===\n";

            auto file = log_file_;
            if (file) {
                (*file) << ss.str();
                file->flush();
            }
            initialized_.store(false, std::memory_order_release);
            log_file_.reset();
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

    // fix: initialized_ is now atomic so we can read it without
    // holding the mutex when needed (and so a single store on shutdown is
    // visible to all threads). log_file_ is shared_ptr so background threads
    // can hold a stable snapshot across mutex release windows.
    std::atomic<bool> initialized_;
    std::string log_path_;
    std::shared_ptr<std::ofstream> log_file_;
    mutable std::mutex mutex_;
};

// Convenience macros
#define LOG_INFO(msg)  collider::Logger::instance().log(collider::Logger::Level::INFO, msg)
#define LOG_WARN(msg)  collider::Logger::instance().log(collider::Logger::Level::WARN, msg)
#define LOG_ERROR(msg) collider::Logger::instance().log(collider::Logger::Level::ERR, msg)
#define LOG_DEBUG(msg) collider::Logger::instance().log(collider::Logger::Level::DEBUG, msg)

}  // namespace collider
