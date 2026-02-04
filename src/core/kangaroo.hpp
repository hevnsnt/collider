/**
 * Pollard's Kangaroo Algorithm for ECDLP
 *
 * Reduces search complexity from O(n) to O(√n) for Bitcoin puzzle solving.
 *
 * The algorithm uses two "kangaroos" (tame and wild) that jump through
 * the search space. When they land on the same point (collision), we can
 * compute the private key.
 *
 * For puzzle with N-bit key (range 2^(N-1) to 2^N):
 * - Brute force: O(2^(N-1)) operations
 * - Kangaroo: O(2^(N/2)) operations
 *
 * Features:
 * - Multi-threaded execution with configurable thread count
 * - Work file save/resume for long-running searches
 * - Progress callbacks for status updates
 * - Thread-safe distinguished point storage
 */

#pragma once

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <random>
#include <functional>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include "crypto_cpu.hpp"
#include "puzzle_config.hpp"

// Cross-platform count leading zeros
#ifdef _MSC_VER
#include <intrin.h>
inline int kangaroo_clz64(uint64_t x) {
    unsigned long index;
    if (_BitScanReverse64(&index, x)) {
        return 63 - static_cast<int>(index);
    }
    return 64;
}
#define KANGAROO_CLZ64(x) kangaroo_clz64(x)
#else
#define KANGAROO_CLZ64(x) __builtin_clzll(x)
#endif

namespace collider {

/**
 * Jump configuration for kangaroo algorithm
 * Uses power-of-2 jumps for efficient computation
 */
struct JumpTable {
    static constexpr int NUM_JUMPS = 32;  // Number of distinct jump sizes

    // Jump distances (scalars)
    cpu::uint256_t distances[NUM_JUMPS];

    // Pre-computed jump points (distance[i] * G)
    cpu::ECPoint points[NUM_JUMPS];

    // Mean jump size (for expected collision distance)
    cpu::uint256_t mean_jump;

    /**
     * Initialize jump table for a given range
     * Jump sizes are roughly: 2^(log2(sqrt(range))/2 + i) for i in 0..NUM_JUMPS-1
     */
    void init(const UInt256& range_size) {
        // Calculate sqrt(range) for expected walk length
        int range_bits = range_size.bit_length();
        int sqrt_bits = range_bits / 2;

        // Jump sizes span from ~sqrt(sqrt(range)) to ~sqrt(range)
        // This gives good mixing while keeping expected collision at sqrt(range)
        cpu::uint256_t total(0);

        for (int i = 0; i < NUM_JUMPS; i++) {
            // Jump size: 2^(sqrt_bits/4 + i*3/4)
            int shift = std::max(1, sqrt_bits / 4 + (i * sqrt_bits * 3) / (4 * NUM_JUMPS));
            shift = std::min(shift, 250);  // Cap at 250 bits

            distances[i] = cpu::uint256_t(0);
            int word = shift / 64;
            int bit = shift % 64;
            if (word < 4) {
                distances[i].d[word] = 1ULL << bit;
            }

            // Compute jump point: distances[i] * G
            cpu::ec_mul(points[i], distances[i]);

            // Accumulate for mean
            cpu::add256(total, total, distances[i]);
        }

        // Mean jump = total / NUM_JUMPS
        mean_jump = total;
        // Simple division by 32: shift right by 5
        mean_jump.d[0] = (mean_jump.d[0] >> 5) | (mean_jump.d[1] << 59);
        mean_jump.d[1] = (mean_jump.d[1] >> 5) | (mean_jump.d[2] << 59);
        mean_jump.d[2] = (mean_jump.d[2] >> 5) | (mean_jump.d[3] << 59);
        mean_jump.d[3] = mean_jump.d[3] >> 5;
    }

    /**
     * Get jump index from point X coordinate
     * Uses low bits for pseudo-random selection
     */
    int get_jump_index(const cpu::uint256_t& x) const {
        return x.d[0] & (NUM_JUMPS - 1);
    }
};

/**
 * Distinguished point for collision detection
 * A point is distinguished if its X coordinate has a certain number of trailing zeros
 */
struct DistinguishedPoint {
    cpu::uint256_t x;           // X coordinate (used as key)
    cpu::uint256_t distance;    // Total distance traveled from start
    bool is_tame;               // True if from tame kangaroo

    bool operator==(const DistinguishedPoint& other) const {
        return x == other.x;
    }

    // Serialization for work files
    std::string serialize() const {
        std::ostringstream oss;
        oss << std::hex << std::setfill('0');
        for (int i = 3; i >= 0; i--) oss << std::setw(16) << x.d[i];
        oss << ":";
        for (int i = 3; i >= 0; i--) oss << std::setw(16) << distance.d[i];
        oss << ":" << (is_tame ? "T" : "W");
        return oss.str();
    }

    static DistinguishedPoint deserialize(const std::string& s) {
        DistinguishedPoint dp;
        size_t pos1 = s.find(':');
        size_t pos2 = s.rfind(':');

        std::string x_hex = s.substr(0, pos1);
        std::string d_hex = s.substr(pos1 + 1, pos2 - pos1 - 1);
        std::string type = s.substr(pos2 + 1);

        // Parse x
        for (int i = 0; i < 4; i++) {
            dp.x.d[3 - i] = std::stoull(x_hex.substr(i * 16, 16), nullptr, 16);
        }
        // Parse distance
        for (int i = 0; i < 4; i++) {
            dp.distance.d[3 - i] = std::stoull(d_hex.substr(i * 16, 16), nullptr, 16);
        }
        dp.is_tame = (type == "T");
        return dp;
    }
};

/**
 * Hash function for distinguished points
 */
struct DPHash {
    size_t operator()(const cpu::uint256_t& x) const {
        // Use lower 64 bits for hash (good enough for our purposes)
        return std::hash<uint64_t>{}(x.d[0] ^ x.d[1] ^ x.d[2] ^ x.d[3]);
    }
};

/**
 * Single kangaroo state
 */
struct Kangaroo {
    cpu::ECPoint point;         // Current point (Jacobian)
    cpu::uint256_t distance;    // Total scalar distance from start
    bool is_tame;               // Tame or wild
    uint64_t steps;             // Number of steps taken
    int thread_id;              // Which thread owns this kangaroo

    Kangaroo() : distance(0), is_tame(false), steps(0), thread_id(0) {
        point.set_infinity();
    }
};

/**
 * Kangaroo search result
 */
struct KangarooResult {
    bool found;
    cpu::uint256_t private_key;
    uint64_t tame_steps;
    uint64_t wild_steps;
    uint64_t distinguished_points;
    uint64_t collisions_checked;
    double elapsed_seconds;
};

/**
 * Work file format for save/resume
 */
struct KangarooWorkFile {
    std::string target_pubkey_hex;
    std::string range_start_hex;
    std::string range_end_hex;
    int dp_bits;
    uint64_t total_steps;
    std::vector<DistinguishedPoint> dps;
    std::chrono::system_clock::time_point started_at;
    double elapsed_seconds;

    bool save(const std::string& filename) const {
        std::ofstream f(filename);
        if (!f) return false;

        f << "# Collider Kangaroo Work File v1.0\n";
        f << "target=" << target_pubkey_hex << "\n";
        f << "range_start=" << range_start_hex << "\n";
        f << "range_end=" << range_end_hex << "\n";
        f << "dp_bits=" << dp_bits << "\n";
        f << "total_steps=" << total_steps << "\n";
        f << "elapsed_seconds=" << std::fixed << std::setprecision(1) << elapsed_seconds << "\n";
        f << "dp_count=" << dps.size() << "\n";
        f << "# Distinguished points (x:distance:type)\n";
        for (const auto& dp : dps) {
            f << dp.serialize() << "\n";
        }
        return true;
    }

    bool load(const std::string& filename) {
        std::ifstream f(filename);
        if (!f) return false;

        std::string line;
        size_t dp_count = 0;

        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;

            size_t eq = line.find('=');
            if (eq != std::string::npos) {
                std::string key = line.substr(0, eq);
                std::string val = line.substr(eq + 1);

                if (key == "target") target_pubkey_hex = val;
                else if (key == "range_start") range_start_hex = val;
                else if (key == "range_end") range_end_hex = val;
                else if (key == "dp_bits") dp_bits = std::stoi(val);
                else if (key == "total_steps") total_steps = std::stoull(val);
                else if (key == "elapsed_seconds") elapsed_seconds = std::stod(val);
                else if (key == "dp_count") dp_count = std::stoull(val);
            } else if (line.find(':') != std::string::npos) {
                // DP entry
                dps.push_back(DistinguishedPoint::deserialize(line));
            }
        }

        return !target_pubkey_hex.empty();
    }
};

/**
 * Thread-safe Distinguished Point Table
 */
class ThreadSafeDPTable {
public:
    bool insert_and_check(const DistinguishedPoint& dp, cpu::uint256_t& result_key) {
        std::unique_lock<std::shared_mutex> lock(mutex_);

        uint64_t key = DPHash{}(dp.x);
        auto it = table_.find(key);

        if (it != table_.end()) {
            // Potential collision - verify X coordinates match
            if (it->second.x == dp.x && it->second.is_tame != dp.is_tame) {
                // Real collision between tame and wild!
                const DistinguishedPoint& other = it->second;

                if (dp.is_tame) {
                    cpu::sub256(result_key, dp.distance, other.distance);
                } else {
                    cpu::sub256(result_key, other.distance, dp.distance);
                }
                return true;
            }
        }

        table_[key] = dp;
        return false;
    }

    size_t size() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return table_.size();
    }

    std::vector<DistinguishedPoint> get_all() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        std::vector<DistinguishedPoint> result;
        result.reserve(table_.size());
        for (const auto& [key, dp] : table_) {
            result.push_back(dp);
        }
        return result;
    }

    void load(const std::vector<DistinguishedPoint>& dps) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        for (const auto& dp : dps) {
            uint64_t key = DPHash{}(dp.x);
            table_[key] = dp;
        }
    }

private:
    mutable std::shared_mutex mutex_;
    std::unordered_map<uint64_t, DistinguishedPoint> table_;
};

/**
 * Pollard's Kangaroo ECDLP Solver
 *
 * Features:
 * - Multi-threaded execution
 * - Work file save/resume
 * - Progress callbacks
 *
 * Usage:
 *   KangarooSolver solver;
 *   solver.set_range(range_start, range_end);
 *   solver.set_target_pubkey(target_x, target_y);
 *   solver.set_num_threads(8);
 *   solver.set_work_file("puzzle135.work");
 *   auto result = solver.solve();
 */
class KangarooSolver {
public:
    // Configuration
    int dp_bits = 20;  // Distinguished point: trailing zeros in X coordinate
    uint64_t max_steps = 0;  // 0 = no limit (use expected steps)
    std::atomic<bool> stop_flag{false};

    // Progress callback: (tame_steps, wild_steps, dp_count, keys_per_sec) -> continue?
    std::function<bool(uint64_t, uint64_t, uint64_t, double)> progress_callback;

    /**
     * Set number of threads for parallel execution
     */
    void set_num_threads(int n) {
        num_threads_ = std::max(1, std::min(n, 256));
    }

    /**
     * Set work file path for save/resume
     * If file exists, work will be resumed
     * Progress is auto-saved every save_interval_seconds
     */
    void set_work_file(const std::string& path, int save_interval_seconds = 60) {
        work_file_path_ = path;
        save_interval_ = save_interval_seconds;
    }

    /**
     * Set the search range
     */
    void set_range(const UInt256& start, const UInt256& end) {
        range_start_.d[0] = start.parts[0];
        range_start_.d[1] = start.parts[1];
        range_start_.d[2] = start.parts[2];
        range_start_.d[3] = start.parts[3];

        range_end_.d[0] = end.parts[0];
        range_end_.d[1] = end.parts[1];
        range_end_.d[2] = end.parts[2];
        range_end_.d[3] = end.parts[3];

        // Calculate range size
        cpu::sub256(range_size_, range_end_, range_start_);

        // Initialize jump table
        UInt256 rs;
        rs.parts[0] = range_size_.d[0];
        rs.parts[1] = range_size_.d[1];
        rs.parts[2] = range_size_.d[2];
        rs.parts[3] = range_size_.d[3];
        jumps_.init(rs);

        // Set DP bits based on range size
        // DP bits ~ sqrt(range_bits) / 2 gives good memory/speed tradeoff
        int range_bits = rs.bit_length();
        dp_bits = std::max(10, std::min(28, range_bits / 4));

        // Expected steps ~ 2 * sqrt(range)
        // For large puzzles (> ~120 bits), the expected steps overflow uint64_t
        // Set to max value in that case (effectively unlimited)
        int exp_bits = range_bits / 2 + 2;
        if (exp_bits >= 63) {
            expected_steps_ = UINT64_MAX;
        } else {
            expected_steps_ = 1ULL << exp_bits;
        }

        // Set max_steps only if not already set by user
        // For astronomical expected_steps, use unlimited (0 means no limit after this)
        if (max_steps == 0) {
            if (expected_steps_ == UINT64_MAX) {
                max_steps = UINT64_MAX;  // Unlimited for large puzzles
            } else {
                max_steps = expected_steps_ * 4;
            }
        }

        // Store hex for work file
        range_start_hex_ = uint256_to_hex(range_start_);
        range_end_hex_ = uint256_to_hex(range_end_);
    }

    /**
     * Set target public key (the point we're trying to find the discrete log of)
     */
    void set_target_pubkey(const cpu::uint256_t& x, const cpu::uint256_t& y) {
        target_x_ = x;
        target_y_ = y;
        target_pubkey_hex_ = uint256_to_hex(x) + uint256_to_hex(y);
    }

    /**
     * Set target from Hash160 (compute pubkey from address)
     * Note: For Bitcoin puzzles, we have the address, not the pubkey.
     * This won't work directly - we need to use hash-based collision instead.
     */
    void set_target_h160(const std::array<uint8_t, 20>& h160) {
        target_h160_ = h160;
        use_h160_target_ = true;
    }

    /**
     * Run the kangaroo algorithm with multi-threading
     */
    KangarooResult solve() {
        KangarooResult result{false, cpu::uint256_t(0), 0, 0, 0, 0, 0.0};

        // Try to resume from work file
        if (!work_file_path_.empty()) {
            KangarooWorkFile wf;
            if (wf.load(work_file_path_)) {
                dp_table_.load(wf.dps);
                total_steps_.store(wf.total_steps);
                elapsed_before_resume_ = wf.elapsed_seconds;
                std::cout << "[*] Resumed from work file: " << wf.dps.size()
                          << " DPs, " << wf.total_steps << " steps\n";
            }
        }

        // For hash-based targeting (Bitcoin puzzles), use modified approach
        if (use_h160_target_) {
            return solve_h160_hybrid_mt();
        }

        // Standard Kangaroo with known target pubkey
        return solve_standard_mt();
    }

private:
    cpu::uint256_t range_start_;
    cpu::uint256_t range_end_;
    cpu::uint256_t range_size_;
    JumpTable jumps_;
    uint64_t expected_steps_;

    cpu::uint256_t target_x_;
    cpu::uint256_t target_y_;
    std::array<uint8_t, 20> target_h160_;
    bool use_h160_target_ = false;

    // Multi-threading
    int num_threads_ = std::thread::hardware_concurrency();
    std::atomic<uint64_t> total_steps_{0};
    std::atomic<bool> solution_found_{false};
    cpu::uint256_t solution_key_;
    std::mutex solution_mutex_;

    // Work file
    std::string work_file_path_;
    int save_interval_ = 60;
    double elapsed_before_resume_ = 0.0;
    std::string target_pubkey_hex_;
    std::string range_start_hex_;
    std::string range_end_hex_;

    // Thread-safe DP table
    ThreadSafeDPTable dp_table_;

    /**
     * Convert uint256 to hex string
     */
    static std::string uint256_to_hex(const cpu::uint256_t& v) {
        std::ostringstream oss;
        oss << std::hex << std::setfill('0');
        for (int i = 3; i >= 0; i--) {
            oss << std::setw(16) << v.d[i];
        }
        return oss.str();
    }

    /**
     * Check if point has distinguished property (trailing zeros in X)
     */
    bool is_distinguished(const cpu::uint256_t& x) const {
        uint64_t mask = (1ULL << dp_bits) - 1;
        return (x.d[0] & mask) == 0;
    }

    /**
     * Save work to file
     */
    void save_work(uint64_t steps, double elapsed) {
        if (work_file_path_.empty()) return;

        KangarooWorkFile wf;
        wf.target_pubkey_hex = target_pubkey_hex_;
        wf.range_start_hex = range_start_hex_;
        wf.range_end_hex = range_end_hex_;
        wf.dp_bits = dp_bits;
        wf.total_steps = steps;
        wf.elapsed_seconds = elapsed + elapsed_before_resume_;
        wf.dps = dp_table_.get_all();

        wf.save(work_file_path_);
    }

    /**
     * Multi-threaded standard kangaroo algorithm
     */
    KangarooResult solve_standard_mt() {
        KangarooResult result{false, cpu::uint256_t(0), 0, 0, 0, 0, 0.0};
        auto start_time = std::chrono::steady_clock::now();
        auto last_save = start_time;

        std::vector<std::thread> threads;
        threads.reserve(num_threads_);

        // Each thread runs pairs of tame/wild kangaroos
        for (int t = 0; t < num_threads_; t++) {
            threads.emplace_back([this, t]() {
                thread_worker_standard(t);
            });
        }

        // Monitor progress and handle saves
        while (!solution_found_.load() && !stop_flag.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));

            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - start_time).count();
            uint64_t steps = total_steps_.load();

            if (progress_callback) {
                double rate = steps / std::max(0.001, elapsed);
                if (!progress_callback(steps / 2, steps / 2, dp_table_.size(), rate)) {
                    stop_flag.store(true);
                    break;
                }
            }

            // Auto-save work file
            if (!work_file_path_.empty()) {
                auto save_elapsed = std::chrono::duration<double>(now - last_save).count();
                if (save_elapsed >= save_interval_) {
                    save_work(steps, elapsed);
                    last_save = now;
                }
            }

            // Check max steps
            if (max_steps > 0 && steps >= max_steps) {
                stop_flag.store(true);
                break;
            }
        }

        // Wait for all threads
        for (auto& t : threads) {
            if (t.joinable()) t.join();
        }

        auto end_time = std::chrono::steady_clock::now();
        result.elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count()
                                + elapsed_before_resume_;
        result.tame_steps = total_steps_.load() / 2;
        result.wild_steps = total_steps_.load() / 2;
        result.distinguished_points = dp_table_.size();

        if (solution_found_.load()) {
            result.found = true;
            result.private_key = solution_key_;
        }

        // Final save
        if (!work_file_path_.empty()) {
            save_work(total_steps_.load(), result.elapsed_seconds - elapsed_before_resume_);
        }

        return result;
    }

    /**
     * Worker thread for standard kangaroo
     */
    void thread_worker_standard(int thread_id) {
        std::random_device rd;
        std::mt19937_64 rng(rd() + thread_id);

        // Initialize tame kangaroo at range midpoint + small offset for this thread
        Kangaroo tame;
        tame.is_tame = true;
        tame.thread_id = thread_id;

        // Tame starts at (range_start + range_size/2 + thread_offset) * G
        cpu::uint256_t tame_start;
        tame_start.d[0] = (range_size_.d[0] >> 1) | (range_size_.d[1] << 63);
        tame_start.d[1] = (range_size_.d[1] >> 1) | (range_size_.d[2] << 63);
        tame_start.d[2] = (range_size_.d[2] >> 1) | (range_size_.d[3] << 63);
        tame_start.d[3] = range_size_.d[3] >> 1;

        // Add thread offset to spread starting points
        cpu::uint256_t offset;
        offset.d[0] = rng() % (1ULL << 20);  // Random offset within 2^20
        offset.d[1] = offset.d[2] = offset.d[3] = 0;
        cpu::add256(tame_start, tame_start, offset);
        cpu::add256(tame_start, tame_start, range_start_);
        tame.distance = tame_start;
        cpu::ec_mul(tame.point, tame_start);

        // Initialize wild kangaroo at target point
        Kangaroo wild;
        wild.is_tame = false;
        wild.thread_id = thread_id;
        wild.point.X = target_x_;
        wild.point.Y = target_y_;
        wild.point.Z = cpu::uint256_t(1);
        wild.distance = cpu::uint256_t(0);

        // Add small random offset to wild too for thread diversity
        cpu::uint256_t wild_offset;
        wild_offset.d[0] = rng() % (1ULL << 16);
        wild_offset.d[1] = wild_offset.d[2] = wild_offset.d[3] = 0;

        // wild.point += wild_offset * G
        cpu::ECPoint offset_point;
        cpu::ec_mul(offset_point, wild_offset);
        cpu::uint256_t ox, oy;
        cpu::ec_to_affine(ox, oy, offset_point);
        cpu::ec_add(wild.point, wild.point, ox, oy);
        wild.distance = wild_offset;

        // Main loop
        while (!stop_flag.load() && !solution_found_.load()) {
            // Step tame kangaroo
            step_kangaroo(tame);
            total_steps_.fetch_add(1);

            // Check if tame is at distinguished point
            cpu::uint256_t tame_x, tame_y;
            cpu::ec_to_affine(tame_x, tame_y, tame.point);

            if (is_distinguished(tame_x)) {
                DistinguishedPoint dp;
                dp.x = tame_x;
                dp.distance = tame.distance;
                dp.is_tame = true;

                cpu::uint256_t result_key;
                if (dp_table_.insert_and_check(dp, result_key)) {
                    std::lock_guard<std::mutex> lock(solution_mutex_);
                    solution_key_ = result_key;
                    solution_found_.store(true);
                    return;
                }
            }

            // Step wild kangaroo
            step_kangaroo(wild);
            total_steps_.fetch_add(1);

            // Check if wild is at distinguished point
            cpu::uint256_t wild_x, wild_y;
            cpu::ec_to_affine(wild_x, wild_y, wild.point);

            if (is_distinguished(wild_x)) {
                DistinguishedPoint dp;
                dp.x = wild_x;
                dp.distance = wild.distance;
                dp.is_tame = false;

                cpu::uint256_t result_key;
                if (dp_table_.insert_and_check(dp, result_key)) {
                    std::lock_guard<std::mutex> lock(solution_mutex_);
                    solution_key_ = result_key;
                    solution_found_.store(true);
                    return;
                }
            }
        }
    }

    /**
     * Multi-threaded hybrid approach for Hash160 targets
     */
    KangarooResult solve_h160_hybrid_mt() {
        KangarooResult result{false, cpu::uint256_t(0), 0, 0, 0, 0, 0.0};
        auto start_time = std::chrono::steady_clock::now();
        auto last_save = start_time;

        std::vector<std::thread> threads;
        threads.reserve(num_threads_);

        for (int t = 0; t < num_threads_; t++) {
            threads.emplace_back([this, t]() {
                thread_worker_h160(t);
            });
        }

        // Monitor progress
        while (!solution_found_.load() && !stop_flag.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));

            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - start_time).count();
            uint64_t steps = total_steps_.load();

            if (progress_callback) {
                double rate = steps / std::max(0.001, elapsed);
                if (!progress_callback(steps, 0, 0, rate)) {
                    stop_flag.store(true);
                    break;
                }
            }

            if (max_steps > 0 && steps >= max_steps) {
                stop_flag.store(true);
                break;
            }
        }

        for (auto& t : threads) {
            if (t.joinable()) t.join();
        }

        auto end_time = std::chrono::steady_clock::now();
        result.elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count()
                                + elapsed_before_resume_;
        result.tame_steps = total_steps_.load();

        if (solution_found_.load()) {
            result.found = true;
            result.private_key = solution_key_;
        }

        return result;
    }

    /**
     * Worker thread for H160 hybrid search
     */
    void thread_worker_h160(int thread_id) {
        std::random_device rd;
        std::mt19937_64 rng(rd() + thread_id);

        // Start at random position in range
        cpu::uint256_t position;
        for (int j = 0; j < 4; j++) {
            position.d[j] = rng();
        }

        // Clamp to range
        int range_bits = 0;
        for (int j = 3; j >= 0; j--) {
            if (range_size_.d[j] != 0) {
                range_bits = j * 64 + 64 - KANGAROO_CLZ64(range_size_.d[j]);
                break;
            }
        }

        if (range_bits < 256) {
            int word = range_bits / 64;
            int bit = range_bits % 64;
            for (int j = word + 1; j < 4; j++) position.d[j] = 0;
            if (word < 4 && bit < 64) {
                position.d[word] &= (1ULL << bit) - 1;
            }
        }

        cpu::add256(position, position, range_start_);

        uint8_t privkey_bytes[32];

        while (!stop_flag.load() && !solution_found_.load()) {
            // Convert position to bytes
            for (int j = 0; j < 4; j++) {
                uint64_t val = position.d[3 - j];
                privkey_bytes[j * 8 + 0] = (val >> 56) & 0xff;
                privkey_bytes[j * 8 + 1] = (val >> 48) & 0xff;
                privkey_bytes[j * 8 + 2] = (val >> 40) & 0xff;
                privkey_bytes[j * 8 + 3] = (val >> 32) & 0xff;
                privkey_bytes[j * 8 + 4] = (val >> 24) & 0xff;
                privkey_bytes[j * 8 + 5] = (val >> 16) & 0xff;
                privkey_bytes[j * 8 + 6] = (val >> 8) & 0xff;
                privkey_bytes[j * 8 + 7] = val & 0xff;
            }

            // Compute hash160
            auto h160 = cpu::compute_hash160(privkey_bytes);
            total_steps_.fetch_add(1);

            // Check for match
            if (h160 == target_h160_) {
                std::lock_guard<std::mutex> lock(solution_mutex_);
                solution_key_ = position;
                solution_found_.store(true);
                return;
            }

            // Kangaroo-style jump
            int jump_idx = jumps_.get_jump_index(position);
            cpu::add256(position, position, jumps_.distances[jump_idx]);

            // Wrap around if we exceed range
            if (position >= range_end_) {
                cpu::sub256(position, position, range_size_);
            }
        }
    }

    /**
     * Take one step with a kangaroo
     */
    void step_kangaroo(Kangaroo& k) {
        // Get affine X for jump selection
        cpu::uint256_t x, y;
        cpu::ec_to_affine(x, y, k.point);

        // Select jump based on X coordinate
        int idx = jumps_.get_jump_index(x);

        // Update distance
        cpu::add256(k.distance, k.distance, jumps_.distances[idx]);

        // Update point: P = P + jump_point[idx]
        cpu::ECPoint new_point;
        cpu::uint256_t jump_x, jump_y;
        cpu::ec_to_affine(jump_x, jump_y, jumps_.points[idx]);
        cpu::ec_add(new_point, k.point, jump_x, jump_y);
        k.point = new_point;

        k.steps++;
    }
};

}  // namespace collider
