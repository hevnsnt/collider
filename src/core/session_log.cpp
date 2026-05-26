// session_log.cpp -- implementation of the per-session log + state +
// crash handler declared in core/session_log.hpp.
//
// Three independent subsystems, three internal singletons:
//
//   * SessionLogSink: owns the std::ofstream for the per-session
//     log file, the rotation policy, and the write mutex. Created in
//     init_session_log(), destroyed at process exit (Meyers singleton).
//
//   * SessionStateStore: owns the most-recent SessionState snapshot,
//     the throttle clock, and the atomic-write helper. Created lazily
//     on the first update_session_state() call.
//
//   * CrashHandler: owns the Windows SEH filter / POSIX sigaction
//     registration. Created in install_crash_handler(). The handler
//     itself is signal-async-safe: it reads pre-allocated buffers and
//     writes to a pre-opened file descriptor (Windows: a path it
//     opens-on-fault with raw CreateFileA; POSIX: ::open + ::write).
//
// The three singletons do not depend on each other except that the
// crash handler reads the *path* (not the file handle) of the
// session_state.json that SessionStateStore writes to. The path is
// pre-computed at install_crash_handler() time and stored in a static
// array; no allocation in signal context.

#include "core/session_log.hpp"

#include "cli/cli_parser.hpp"
#include "core/log.hpp"
#include "core/logger.hpp"
#include "core/paths.hpp"
#include "core/secure_write.hpp"
#include "core/version.hpp"
#include "platform/platform.hpp"
#include "runtime/gpu_detection.hpp"  // compile_time_sm_set, sm_set_to_string

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
    #include <process.h>     // _getpid
    #include <psapi.h>       // MODULEINFO for crash_dump_modules
    #include <dbghelp.h>
    // dbghelp must come AFTER windows.h. Link with -ldbghelp; the
    // CMakeLists.txt wires this through for collider_core.
    #include <io.h>          // _commit, _fileno used by the FILE*-based
                             // atomic_write_state_file fsync sequence
                             // (mirrors src/gpu/rckangaroo_wrapper.cu:1585+).
#else
    #include <unistd.h>
    #include <sys/types.h>
    #include <signal.h>
    #include <fcntl.h>
    #ifdef __GLIBC__
        #include <execinfo.h>
    #endif
#endif

namespace collider::log {

namespace {

// ---------------------------------------------------------------------------
// Time formatting helpers (shared by log lines and JSON timestamps)
// ---------------------------------------------------------------------------

// "2026-05-17T15-23-45" (note dashes between time fields; this is used as
// part of a filename and the colons that Windows path components forbid
// would otherwise force a separate sanitization pass).
std::string format_ts_for_filename(std::chrono::system_clock::time_point tp) {
    auto time_t_val = std::chrono::system_clock::to_time_t(tp);
    std::tm tm_buf{};
#ifdef _WIN32
    localtime_s(&tm_buf, &time_t_val);
#else
    localtime_r(&time_t_val, &tm_buf);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H-%M-%S", &tm_buf);
    return buf;
}

// "2026-05-17T15:23:45.123Z" (ISO-8601 with UTC marker). Used inside log
// lines and the JSON snapshot so external tooling (jq, grep, etc.) has a
// single, sortable, parseable timestamp format.
std::string format_ts_iso(std::chrono::system_clock::time_point tp) {
    auto time_t_val = std::chrono::system_clock::to_time_t(tp);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  tp.time_since_epoch()) % 1000;
    std::tm tm_buf{};
#ifdef _WIN32
    gmtime_s(&tm_buf, &time_t_val);
#else
    gmtime_r(&time_t_val, &tm_buf);
#endif
    char buf[40];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm_buf);
    std::ostringstream oss;
    oss << buf << '.' << std::setw(3) << std::setfill('0') << ms.count() << 'Z';
    return oss.str();
}

int current_pid() {
#ifdef _WIN32
    return static_cast<int>(GetCurrentProcessId());
#else
    return static_cast<int>(::getpid());
#endif
}

// ---------------------------------------------------------------------------
// JSON helpers (hand-rolled; the codebase has no JSON dep we can lean on
// and adding one for ~150 bytes of output is overkill)
// ---------------------------------------------------------------------------

// Escape a string for JSON. Handles the standard control-character set
// per RFC 8259 section 7. Bytes < 0x20 that have no short escape are
// emitted as \u00XX.
std::string json_escape(const std::string& in) {
    std::string out;
    out.reserve(in.size() + 8);
    for (unsigned char c : in) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

// Emit a "key": "value" pair, comma-separated against `first`. `first`
// is updated to false after the first emission so callers can iterate
// over a mix of present / absent fields without bookkeeping at every
// site.
struct JsonWriter {
    std::ostringstream oss;
    bool first = true;

    void open() { oss << '{'; first = true; }
    void close() { oss << '}'; }

    void key(const char* k) {
        if (!first) oss << ',';
        first = false;
        oss << '"' << k << "\":";
    }

    void str(const char* k, const std::string& v) {
        key(k);
        oss << '"' << json_escape(v) << '"';
    }
    void num(const char* k, long long v)         { key(k); oss << v; }
    void unum(const char* k, unsigned long long v){ key(k); oss << v; }
    void boolean(const char* k, bool v)          { key(k); oss << (v ? "true" : "false"); }

    template <typename T>
    void opt_num(const char* k, const std::optional<T>& v) {
        if (!v) return;
        key(k);
        oss << *v;
    }
    void opt_str(const char* k, const std::optional<std::string>& v) {
        if (!v) return;
        str(k, *v);
    }
    void opt_bool(const char* k, const std::optional<bool>& v) {
        if (!v) return;
        boolean(k, *v);
    }
    void opt_ts(const char* k,
                const std::optional<std::chrono::system_clock::time_point>& v) {
        if (!v) return;
        str(k, format_ts_iso(*v));
    }

    std::string str_out() const { return oss.str(); }
};

std::string serialize_state(const SessionState& s) {
    JsonWriter w;
    w.open();

    // Common
    w.str("mode", s.mode);
    w.num("pid", s.pid);
    w.str("log_path", s.log_path);
    w.str("boot_ts", format_ts_iso(s.boot_ts));
    w.str("last_update_ts", format_ts_iso(s.last_update_ts));

    // Brainwallet
    w.opt_num("total_checked", s.total_checked);
    w.opt_num("bloom_hits", s.bloom_hits);
    w.opt_str("wordlist_path", s.wordlist_path);
    w.opt_num("wordlist_hash", s.wordlist_hash);
    w.opt_num("current_phase_idx", s.current_phase_idx);
    w.opt_str("current_phase_name", s.current_phase_name);
    w.opt_num("last_save_count", s.last_save_count);

    // Pool
    w.opt_num("current_work_id", s.current_work_id);
    w.opt_str("work_range_start_hex", s.work_range_start_hex);
    w.opt_str("work_range_end_hex", s.work_range_end_hex);
    w.opt_num("work_dp_bits", s.work_dp_bits);
    w.opt_num("dp_count_submitted_this_work", s.dp_count_submitted_this_work);
    w.opt_num("dp_count_submitted_total", s.dp_count_submitted_total);
    w.opt_num("dp_seq_last", s.dp_seq_last);
    w.opt_ts("work_started_at", s.work_started_at);
    w.opt_ts("last_dp_submit_at", s.last_dp_submit_at);
    w.opt_bool("connected", s.connected);
    w.opt_str("pool_endpoint", s.pool_endpoint);

    // Puzzle
    w.opt_num("puzzle_number", s.puzzle_number);
    w.opt_str("puzzle_algorithm", s.puzzle_algorithm);
    w.opt_num("total_steps", s.total_steps);
    w.opt_str("position_full_hex", s.position_full_hex);

    // GPUs
    if (!s.gpus.empty()) {
        w.key("gpus");
        w.oss << '[';
        bool first_gpu = true;
        for (const auto& g : s.gpus) {
            if (!first_gpu) w.oss << ',';
            first_gpu = false;
            JsonWriter g_w;
            g_w.open();
            g_w.num("device_id", g.device_id);
            g_w.str("name", g.name);
            g_w.str("phase", g.phase);
            g_w.opt_num("util_pct", g.util_pct);
            g_w.opt_num("power_w", g.power_w);
            g_w.opt_num("temp_c", g.temp_c);
            g_w.opt_num("pcie_gen", g.pcie_gen);
            g_w.close();
            w.oss << g_w.str_out();
        }
        w.oss << ']';
    }

    w.close();
    return w.str_out();
}

// Atomic file write helper. Mirrors the contract in
// pool/jlp_pool_client.cpp::atomic_write but is intentionally
// duplicated here rather than re-exported because:
//   * pool/atomic_write uses FailHard for recovered-key safety;
//     session_state.json contains no key material (pid, log path,
//     counters, GPU phase, all non-sensitive). Availability beats
//     confidentiality here: the recovery snapshot must never be
//     silently disabled by a permission glitch, so we pass
//     FallbackLoud and let secure_open_ofstream warn on stderr while
//     still opening the file with whatever permissions it can.
//   * The pool helper sits in an anonymous namespace inside its TU
//     and exposing it would force pool/jlp_pool_client.cpp into the
//     session_log link island, which inverts the existing dep graph
//     (core/ should not pull in pool/).
// Atomic, durable write of session_state.json. Ports the canonical
// fsync+unwind pattern from src/gpu/rckangaroo_wrapper.cu:1495-1639
// (save_herd_state) so the crash handler's verbatim read of
// session_state.json sees a payload that has actually reached stable
// storage, not just the libc / OS write cache.
//
// Sequence:
//   1. secure_open_ofstream the .tmp once with FallbackLoud so the file
//      is created with the owner-only DACL contract this header has
//      always promised. The stream is closed immediately; we only used
//      it to anchor the permissions on the freshly created tmp file.
//   2. Re-open the same .tmp via fopen("wb") so we have a real FILE*
//      whose fd we can pass to fsync (POSIX) / _commit (Windows).
//      Writing through the FILE* and then fsync'ing its fd BEFORE
//      fclose is the only signal-safe way to guarantee the bytes are
//      on disk before the rename: a sync-by-name reopen would race a
//      concurrent unlink, and skipping the sync entirely (the original
//      ofs.flush()-only path) leaves a power-loss race wide open.
//   3. fwrite the payload, fflush libc->OS, fsync/_commit the fd, then
//      fclose. Any failure along the way takes the unwind path: close
//      the fp (if still open), remove(tmp), return false. The previous
//      session_state.json (if any) is unchanged, so the crash handler's
//      verbatim read still finds a valid snapshot.
//   4. Atomic rename tmp -> final. POSIX rename(2) is atomic against
//      concurrent readers; Windows std::filesystem::rename throws on a
//      pre-existing target so we remove() it first and retry. The
//      remove+rename window is small but not zero; a power loss inside
//      it leaves the new payload sitting at .tmp under its tmp name,
//      which is strictly better than the pre-fix behavior (a torn,
//      half-written session_state.json the crash handler would have
//      cat-ed into the crash log verbatim).
bool atomic_write_state_file(const std::filesystem::path& path,
                             const std::string& content) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);
    if (ec && !fs::exists(path.parent_path())) {
        return false;
    }
    fs::path tmp = path;
    tmp += ".tmp";

    // Step 1: secure_open + close to anchor owner-only DACL on the new
    // tmp file. session_state.json contains no key material (pid, log
    // path, counters, GPU phase), so availability beats confidentiality
    // here and we pass FallbackLoud rather than FailHard. The stream's
    // scope is intentionally tight: we only want the permissions to
    // stick before we reopen via FILE* for the fsync ride.
    {
        std::ofstream ofs = collider::secure_open_ofstream(
            tmp, std::ios::binary | std::ios::trunc,
            collider::SecureWriteOnFailure::FallbackLoud);
        if (!ofs.is_open()) return false;
        // Close immediately; secure_open_ofstream truncated the file
        // and set the DACL. The fopen below reopens in "wb" mode which
        // truncates again (no-op since the file is empty) and gives us
        // a real fd.
    }

    // Step 2: reopen via FILE* so fsync / _commit can target a fd.
    FILE* fp = std::fopen(tmp.string().c_str(), "wb");
    if (!fp) {
        // Unwind the empty tmp we created in step 1 so it does not
        // accumulate on disk across retries.
        fs::remove(tmp, ec);
        return false;
    }

    // Unwind helper: close fp (if still open) and unlink the partial
    // .tmp. Used by every failure path between here and the rename.
    auto unwind = [&fp, &tmp]() -> bool {
        if (fp) {
            std::fclose(fp);
            fp = nullptr;
        }
        std::error_code rm_ec;
        fs::remove(tmp, rm_ec);
        return false;
    };

    // Step 3: write payload, flush libc->OS, then sync OS->disk before
    // closing the fd.
    if (std::fwrite(content.data(), 1, content.size(), fp) != content.size()) {
        return unwind();
    }
    if (std::fflush(fp) != 0) {
        return unwind();
    }
#ifdef _WIN32
    // _commit returns 0 on success, -1 on failure. Failure means the
    // OS cache flush did not complete; abort the save so the rename
    // below does not race with the unflushed cache on a power-loss
    // restart.
    if (_commit(_fileno(fp)) != 0) {
        return unwind();
    }
#else
    if (::fsync(::fileno(fp)) != 0) {
        return unwind();
    }
#endif

    std::fclose(fp);
    fp = nullptr;

    // Step 4: atomic rename tmp -> final. On Windows, std::filesystem::
    // rename throws on a pre-existing target; the remove+rename retry
    // window is small but not zero. POSIX rename is atomic against
    // concurrent readers, so this branch is effectively a no-op there.
    fs::rename(tmp, path, ec);
    if (ec) {
        fs::remove(path, ec);
        ec.clear();
        fs::rename(tmp, path, ec);
        if (ec) {
            // Final failure: leave the .tmp on disk so the operator
            // can recover it manually if needed; do NOT swallow the
            // error silently.
            std::error_code rm_ec;
            fs::remove(tmp, rm_ec);
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// SessionLogSink -- owns the per-session log file + write mutex.
// ---------------------------------------------------------------------------

class SessionLogSink {
public:
    static SessionLogSink& instance() {
        static SessionLogSink s;
        return s;
    }

    bool initialize() {
        std::lock_guard<std::mutex> lock(mu_);
        if (initialized_) return true;

        namespace fs = std::filesystem;
        boot_ts_ = std::chrono::system_clock::now();

        fs::path log_dir = collider::paths::collider_home() / "logs";
        std::error_code ec;
        fs::create_directories(log_dir, ec);
        if (ec && !fs::exists(log_dir)) {
            std::cerr << "[session_log] WARNING: cannot create "
                      << log_dir.string() << " (" << ec.message()
                      << "); session log disabled for this run."
                      << std::endl;
            return false;
        }

        // Rotation: keep at most kSessionLogRetainCount files. We
        // delete oldest-first by sorting on the filesystem-reported
        // last_write_time. A failure here is non-fatal; we log and
        // continue so a sticky permission glitch on an old file does
        // not block the new session.
        prune_old_logs(log_dir);

        int pid = current_pid();
        std::string ts = format_ts_for_filename(boot_ts_);
        log_path_ = (log_dir /
                     ("collider-" + ts + "-" + std::to_string(pid) + ".log"))
                        .string();

        // Append mode so a same-pid reopen (rare, but the wizard does
        // it for the brainwallet setup path) does not stomp prior
        // content.
        file_ = collider::secure_open_ofstream(
            log_path_, std::ios::out | std::ios::app,
            collider::SecureWriteOnFailure::FallbackLoud);
        if (!file_.is_open()) {
            std::cerr << "[session_log] WARNING: cannot open "
                      << log_path_ << "; session log disabled for this run."
                      << std::endl;
            return false;
        }

        initialized_ = true;
        write_line_locked("INIT", "session log opened at " + log_path_);
        return true;
    }

    bool is_initialized() const {
        std::lock_guard<std::mutex> lock(mu_);
        return initialized_;
    }

    const std::string& log_path() const {
        std::lock_guard<std::mutex> lock(mu_);
        return log_path_;
    }

    std::chrono::system_clock::time_point boot_ts() const {
        std::lock_guard<std::mutex> lock(mu_);
        return boot_ts_;
    }

    void write_milestone(const char* event, const std::string& detail) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!initialized_) return;
        write_line_locked(event, detail);
    }

    // Write a multi-line block (banner / hardware enum). Each line is
    // prefixed with the standard timestamp + tag so the file remains
    // grep-friendly. The block is wrapped between BEGIN_<tag> /
    // END_<tag> sentinels so a downstream reader can extract the full
    // block deterministically.
    void write_block(const char* tag, const std::string& body) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!initialized_) return;
        write_line_locked(std::string("BEGIN_") + tag, std::string{});

        std::istringstream iss(body);
        std::string line;
        while (std::getline(iss, line)) {
            write_line_locked(tag, line);
        }
        write_line_locked(std::string("END_") + tag, std::string{});
    }

    ~SessionLogSink() {
        // No lock on purpose. Same pattern as BalanceFetcher (see
        // runtime/balance.cpp::~BalanceFetcher comment). This is a
        // Meyers-singleton destructor that runs during process teardown
        // after main() returned; on macOS, Apple's pthread library
        // invalidates the mutex's internal state during teardown earlier
        // than glibc/MSVC do, so the lock_guard ctor throws EINVAL via
        // std::system_error -- which propagates uncaught and terminates
        // the process AFTER the session has already finished cleanly,
        // making customer-facing logs end with an abort message instead
        // of a clean exit. By the time we reach this dtor, no other
        // thread should still be writing (runtime is torn down, log()
        // call sites are all dead) so the lock was theatrical anyway.
        if (initialized_ && file_.is_open()) {
            write_line_locked("SHUTDOWN", "session log closing");
            file_.flush();
            file_.close();
        }
    }

private:
    SessionLogSink() = default;
    SessionLogSink(const SessionLogSink&) = delete;
    SessionLogSink& operator=(const SessionLogSink&) = delete;

    // Caller must hold mu_.
    void write_line_locked(const std::string& tag, const std::string& detail) {
        if (!file_.is_open()) return;
        auto now = std::chrono::system_clock::now();
        file_ << format_ts_iso(now) << " [" << tag << "]";
        if (!detail.empty()) {
            file_ << ' ' << detail;
        }
        file_ << '\n';
        file_.flush();
    }

    void prune_old_logs(const std::filesystem::path& log_dir) {
        namespace fs = std::filesystem;
        struct Entry {
            fs::path path;
            fs::file_time_type mtime;
        };
        std::vector<Entry> entries;
        std::error_code ec;
        for (auto& de : fs::directory_iterator(log_dir, ec)) {
            if (!de.is_regular_file()) continue;
            const auto name = de.path().filename().string();
            // Match collider-*-*.log only; do not touch unrelated
            // files (collider.log lives in the parent dir; *.tmp from
            // an interrupted state-file write should age out via the
            // state-store's own retry logic).
            if (name.rfind("collider-", 0) != 0) continue;
            if (de.path().extension() != ".log") continue;
            Entry e;
            e.path = de.path();
            e.mtime = de.last_write_time(ec);
            entries.push_back(std::move(e));
        }
        if (entries.size() <= static_cast<size_t>(kSessionLogRetainCount)) return;
        std::sort(entries.begin(), entries.end(),
                  [](const Entry& a, const Entry& b){ return a.mtime < b.mtime; });
        size_t to_delete = entries.size() -
                           static_cast<size_t>(kSessionLogRetainCount);
        for (size_t i = 0; i < to_delete; ++i) {
            fs::remove(entries[i].path, ec);
            // Ignore ec: a failed delete just means the file survives
            // one more cycle.
        }
    }

    mutable std::mutex mu_;
    bool initialized_ = false;
    std::string log_path_;
    std::chrono::system_clock::time_point boot_ts_{};
    std::ofstream file_;
};

// ---------------------------------------------------------------------------
// SessionStateStore -- owns the in-memory SessionState snapshot, the
// debounce clock, and the path to the on-disk JSON.
// ---------------------------------------------------------------------------

class SessionStateStore {
public:
    static SessionStateStore& instance() {
        static SessionStateStore s;
        return s;
    }

    std::filesystem::path state_path() const {
        return collider::paths::collider_home() / "session_state.json";
    }

    // Enable disk writes. Mirrors SessionLogSink::initialize(): until
    // init_session_log() runs the store accepts merges into the in-memory
    // snapshot but never touches the on-disk session_state.json. This
    // matters for test binaries that exercise wire handlers (handle_auth_ok,
    // handle_work_asn, handle_solution) without ever initializing the
    // session log. Previously every such call clobbered the operator's
    // real ~/.collider/session_state.json from a background thread, and the
    // resulting concurrent fopen/fwrite/rename traffic across many tests
    // intermittently crashed the JLP test binaries with RIP=0.
    void enable_disk_writes() {
        std::lock_guard<std::mutex> lock(mu_);
        disk_writes_enabled_ = true;
    }

    void update(const SessionState& state) {
        std::unique_lock<std::mutex> lock(mu_);
        // Merge semantics: the caller fills in the fields relevant to
        // their wire-in site (pool wants pool_*, brainwallet wants
        // wordlist_*, etc.); the common bookkeeping (pid, log_path,
        // boot_ts) was seeded once by write_startup_banner and must
        // survive every subsequent update. We do this by copying only
        // the field groups the caller actually wrote, leaving the rest
        // of `latest_` intact.
        merge_into_latest(state);
        latest_.last_update_ts = std::chrono::system_clock::now();
        if (!disk_writes_enabled_) return;
        auto now = std::chrono::steady_clock::now();
        if (now - last_write_at_ <
            std::chrono::milliseconds(kSessionStateMinIntervalMs)) {
            return;
        }
        // Take a copy under the lock, drop the lock, then do the
        // disk write. We do not want fsync latency on the lock.
        SessionState copy = latest_;
        last_write_at_ = now;
        lock.unlock();
        (void)write_to_disk(copy);
    }

    void flush() {
        std::unique_lock<std::mutex> lock(mu_);
        if (latest_.mode.empty() && latest_.pid == 0) return;  // never updated
        if (!disk_writes_enabled_) return;
        // Stamp BOTH the in-memory snapshot and the outgoing copy with
        // the same wall-clock instant so a subsequent reader of latest_
        // (e.g. the next update() merge) cannot observe a copy-on-disk
        // newer than the in-memory record. The two timestamps used to
        // drift: flush() updated only the copy, leaving latest_ with a
        // stale ts until the next update() call rewrote it.
        auto now = std::chrono::system_clock::now();
        latest_.last_update_ts = now;
        SessionState copy = latest_;
        last_write_at_ = std::chrono::steady_clock::now();
        lock.unlock();
        (void)write_to_disk(copy);
    }

    bool write_to_disk(const SessionState& state) {
        const std::string json = serialize_state(state);
        return atomic_write_state_file(state_path(), json);
    }

private:
    SessionStateStore() = default;
    SessionStateStore(const SessionStateStore&) = delete;
    SessionStateStore& operator=(const SessionStateStore&) = delete;

    // Apply `incoming` on top of latest_ with merge semantics:
    //   * Always-present bookkeeping (pid, log_path, boot_ts) is taken
    //     from latest_ if the caller did not set it. The seed call in
    //     write_startup_banner() owns these; later callers only set
    //     mode + their domain-specific fields.
    //   * Mode is taken from incoming if non-empty; otherwise latest_
    //     stays as-is. This lets the brainwallet runner overwrite
    //     mode="brainwallet" without the pool path having to first
    //     set mode="pool" before every update.
    //   * std::optional fields: a caller's NULLOPT means "no opinion;
    //     keep the prior value"; an engaged optional overrides.
    //   * gpus vector: if the caller passed a non-empty vector, it
    //     fully replaces the prior list; an empty vector is treated
    //     as "no opinion" (otherwise a pool-mode update that does
    //     not touch GPUs would wipe the hardware enum's seeded list).
    void merge_into_latest(const SessionState& incoming) {
        if (!incoming.mode.empty()) latest_.mode = incoming.mode;
        if (incoming.pid != 0) latest_.pid = incoming.pid;
        if (!incoming.log_path.empty()) latest_.log_path = incoming.log_path;
        if (incoming.boot_ts != std::chrono::system_clock::time_point{}) {
            latest_.boot_ts = incoming.boot_ts;
        }

        auto take_opt = [](auto& dst, const auto& src) {
            if (src.has_value()) dst = src;
        };
        take_opt(latest_.total_checked,        incoming.total_checked);
        take_opt(latest_.bloom_hits,           incoming.bloom_hits);
        take_opt(latest_.wordlist_path,        incoming.wordlist_path);
        take_opt(latest_.wordlist_hash,        incoming.wordlist_hash);
        take_opt(latest_.current_phase_idx,    incoming.current_phase_idx);
        take_opt(latest_.current_phase_name,   incoming.current_phase_name);
        take_opt(latest_.last_save_count,      incoming.last_save_count);

        take_opt(latest_.current_work_id,             incoming.current_work_id);
        take_opt(latest_.work_range_start_hex,        incoming.work_range_start_hex);
        take_opt(latest_.work_range_end_hex,          incoming.work_range_end_hex);
        take_opt(latest_.work_dp_bits,                incoming.work_dp_bits);
        take_opt(latest_.dp_count_submitted_this_work,incoming.dp_count_submitted_this_work);
        take_opt(latest_.dp_count_submitted_total,    incoming.dp_count_submitted_total);
        take_opt(latest_.dp_seq_last,                 incoming.dp_seq_last);
        take_opt(latest_.work_started_at,             incoming.work_started_at);
        take_opt(latest_.last_dp_submit_at,           incoming.last_dp_submit_at);
        take_opt(latest_.connected,                   incoming.connected);
        take_opt(latest_.pool_endpoint,               incoming.pool_endpoint);

        take_opt(latest_.puzzle_number,    incoming.puzzle_number);
        take_opt(latest_.puzzle_algorithm, incoming.puzzle_algorithm);
        take_opt(latest_.total_steps,      incoming.total_steps);
        take_opt(latest_.position_full_hex,incoming.position_full_hex);

        if (!incoming.gpus.empty()) latest_.gpus = incoming.gpus;
    }

    std::mutex mu_;
    SessionState latest_{};
    // last_write_at_ starts at the steady_clock epoch so the FIRST
    // update_session_state() call always writes through (no warm-up
    // period during which the operational snapshot is invisible).
    std::chrono::steady_clock::time_point last_write_at_{};
    // Disk writes stay off until init_session_log() runs. See the
    // enable_disk_writes() comment for why.
    bool disk_writes_enabled_ = false;
};

// ---------------------------------------------------------------------------
// Argv redaction (used by the startup banner)
// ---------------------------------------------------------------------------

// Flag names whose VALUES must be redacted in the startup banner.
// --pool-password and --pool-password-file leak credentials; --activate
// leaks the user's license key; --pool-api-key carries a pool credential
// even though the flag is deprecated (cli_parser.cpp still accepts it
// for back-compat and would otherwise log the value verbatim). The value
// is the argv element directly after the flag (or the part after the
// first '=' for --flag=value).
const char* const kRedactedFlags[] = {
    "--pool-password",
    "--pool-password-file",
    "--pool-api-key",
    "--activate",
};

bool is_redacted_flag(const std::string& s) {
    for (const char* f : kRedactedFlags) {
        if (s == f) return true;
        // --flag=value form: match the prefix up to '='.
        std::string flag_eq = std::string(f) + "=";
        if (s.compare(0, flag_eq.size(), flag_eq) == 0) return true;
    }
    return false;
}

std::string redact_argv(int argc, char** argv) {
    std::ostringstream oss;
    for (int i = 0; i < argc; ++i) {
        if (i > 0) oss << ' ';
        std::string s = argv[i];
        // Case 1: --flag=value -> --flag=REDACTED
        auto eq = s.find('=');
        if (eq != std::string::npos) {
            std::string head = s.substr(0, eq);
            if (is_redacted_flag(head) || is_redacted_flag(s)) {
                oss << head << "=REDACTED";
                continue;
            }
        }
        // Case 2: --flag value -> --flag REDACTED (consume the next argv)
        if (is_redacted_flag(s) && i + 1 < argc) {
            oss << s << " REDACTED";
            ++i;
            continue;
        }
        oss << s;
    }
    return oss.str();
}

// ---------------------------------------------------------------------------
// Build-info detection (compile-time defines wired through CMakeLists)
// ---------------------------------------------------------------------------

std::string build_flags() {
    std::ostringstream oss;
#ifdef COLLIDER_PRO
    oss << "Pro";
#elif defined(COLLIDER_FREE)
    oss << "Free";
#else
    oss << "Unknown";
#endif
#ifdef COLLIDER_USE_CUDA
    oss << " CUDA";
#endif
#ifdef COLLIDER_USE_METAL
    oss << " Metal";
#endif
#ifdef COLLIDER_USE_CPU
    oss << " CPU";
#endif
#ifdef COLLIDER_HAS_NVML
    oss << " NVML";
#endif
#ifdef COLLIDER_HAS_OPENSSL
    oss << " OpenSSL";
#endif
    return oss.str();
}

// ---------------------------------------------------------------------------
// Crash handler internals
// ---------------------------------------------------------------------------
//
// The handler runs in async context (Windows SEH filter; POSIX signal
// handler). Async-signal-safety constraints:
//   * No std::ostream, std::string, std::filesystem, malloc.
//   * Use only ::open, ::write, ::close, ::read, ::_exit (POSIX);
//     CreateFileA, WriteFile, CloseHandle, ReadFile, TerminateProcess
//     (Windows).
//
// The strategy: pre-allocate every buffer at install_crash_handler()
// time. The handler then uses only stack locals (signal-safe) and the
// pre-allocated buffers (also safe because nothing in the handler
// frees or reallocates them).

constexpr size_t kCrashPathMax = 512;
constexpr size_t kCrashScratchMax = 16 * 1024;

struct CrashPaths {
    // Path the handler writes the crash report to. Filled on
    // install_crash_handler() with
    // ~/.collider/crash-<install-ts>.log. Using the install timestamp
    // (instead of generating one inside the handler) keeps the
    // handler async-signal-safe.
    char crash_log[kCrashPathMax]{};
    // Path the handler reads for the latest session_state.json
    // snapshot. The session state file is overwritten in place, so
    // the path is stable for the life of the process.
    char state_json[kCrashPathMax]{};
    // Path the handler reads for the active session log (where milestones
    // land). The crash report tails the last few KB so the diagnostic
    // milestones immediately preceding the fault appear in the crash
    // dump itself instead of forcing a reader to correlate by timestamp.
    char session_log[kCrashPathMax]{};
    // The boot_ts as a steady_clock time_point, captured at install
    // time. The handler reports "uptime_ms = (now - boot)" so an
    // operator can correlate the crash against the run length.
    std::chrono::steady_clock::time_point install_at{};
    std::atomic<bool> installed{false};
};

CrashPaths& crash_paths() {
    static CrashPaths cp;
    return cp;
}

// Async-signal-safe write of a NUL-terminated string to an open
// descriptor. We use the C-style helpers (write / WriteFile) directly
// because std::ofstream is not safe in this context.
void crash_write_str(
#ifdef _WIN32
    HANDLE h,
#else
    int fd,
#endif
    const char* s) {
    if (!s) return;
    size_t len = std::strlen(s);
#ifdef _WIN32
    DWORD written = 0;
    WriteFile(h, s, static_cast<DWORD>(len), &written, nullptr);
#else
    ssize_t written = 0;
    while (written < static_cast<ssize_t>(len)) {
        ssize_t n = ::write(fd, s + written, len - written);
        if (n < 0) {
            if (errno == EINTR) continue;
            return;
        }
        written += n;
    }
#endif
}

// Integer formatting without snprintf (signal-safe). Writes the
// decimal representation of `v` into `out` and returns the number of
// chars written. `out` must have room for at least 24 chars.
size_t crash_format_u64(unsigned long long v, char* out) {
    if (v == 0) { out[0] = '0'; return 1; }
    char tmp[24];
    size_t n = 0;
    while (v) {
        tmp[n++] = static_cast<char>('0' + (v % 10));
        v /= 10;
    }
    for (size_t i = 0; i < n; ++i) {
        out[i] = tmp[n - 1 - i];
    }
    return n;
}

void crash_write_u64(
#ifdef _WIN32
    HANDLE h,
#else
    int fd,
#endif
    unsigned long long v) {
    char buf[24];
    size_t n = crash_format_u64(v, buf);
    buf[n] = '\0';
    crash_write_str(
#ifdef _WIN32
        h,
#else
        fd,
#endif
        buf);
}

// Hex formatting helper. `width` is the minimum field width; 0 means
// "no padding". Async-signal-safe (no malloc, no iostreams).
void crash_write_hex(
#ifdef _WIN32
    HANDLE h,
#else
    int fd,
#endif
    unsigned long long v, int width) {
    static const char* kHex = "0123456789abcdef";
    char buf[24];
    int n = 0;
    if (v == 0) {
        buf[n++] = '0';
    } else {
        char tmp[24];
        int t = 0;
        while (v) {
            tmp[t++] = kHex[v & 0xF];
            v >>= 4;
        }
        for (int i = 0; i < t; ++i) buf[n++] = tmp[t - 1 - i];
    }
    while (n < width) {
        for (int i = n; i > 0; --i) buf[i] = buf[i - 1];
        buf[0] = '0';
        ++n;
    }
    buf[n] = '\0';
    crash_write_str(
#ifdef _WIN32
        h,
#else
        fd,
#endif
        buf);
}

// Copy the contents of `src_path` into the open output sink. Used to
// append the session_state.json verbatim to the crash report.
// Async-signal-safe: ::open / ::read / ::close (POSIX) or CreateFileA /
// ReadFile / CloseHandle (Windows). Reads up to kCrashScratchMax
// bytes; longer files are truncated to keep the handler bounded.
void crash_append_file(
#ifdef _WIN32
    HANDLE out,
#else
    int out_fd,
#endif
    const char* src_path) {
#ifdef _WIN32
    HANDLE in = CreateFileA(src_path, GENERIC_READ, FILE_SHARE_READ,
                            nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL,
                            nullptr);
    if (in == INVALID_HANDLE_VALUE) {
        crash_write_str(out, "(session_state.json not available)\n");
        return;
    }
    static char buf[kCrashScratchMax];
    DWORD read = 0;
    if (ReadFile(in, buf, static_cast<DWORD>(sizeof(buf)), &read, nullptr) && read > 0) {
        DWORD written = 0;
        WriteFile(out, buf, read, &written, nullptr);
    }
    CloseHandle(in);
#else
    int in_fd = ::open(src_path, O_RDONLY);
    if (in_fd < 0) {
        crash_write_str(out_fd, "(session_state.json not available)\n");
        return;
    }
    static char buf[kCrashScratchMax];
    ssize_t n;
    while ((n = ::read(in_fd, buf, sizeof(buf))) > 0) {
        ssize_t off = 0;
        while (off < n) {
            ssize_t w = ::write(out_fd, buf + off, n - off);
            if (w < 0) {
                if (errno == EINTR) continue;
                break;
            }
            off += w;
        }
        if (n < static_cast<ssize_t>(sizeof(buf))) break;
    }
    ::close(in_fd);
#endif
}

#ifdef _WIN32

// Dump the integer registers from a CONTEXT record. Useful for AVs
// where the disassembly of the faulting instruction references a
// specific GPR -- the operator can read the register value here
// instead of attaching a debugger. x64 only; the build is x64 only
// per CMake. Async-signal-not-strictly-safe but the SEH filter runs
// synchronously on the faulting thread, same risk envelope as the
// existing StackWalk64 path.
void crash_dump_registers(HANDLE out, const CONTEXT* ctx) {
    if (!ctx) return;
#ifdef _M_X64
    struct Reg { const char* name; DWORD64 val; };
    Reg regs[] = {
        {"rax", ctx->Rax}, {"rbx", ctx->Rbx}, {"rcx", ctx->Rcx},
        {"rdx", ctx->Rdx}, {"rsi", ctx->Rsi}, {"rdi", ctx->Rdi},
        {"rbp", ctx->Rbp}, {"rsp", ctx->Rsp},
        {"r8",  ctx->R8},  {"r9",  ctx->R9},  {"r10", ctx->R10},
        {"r11", ctx->R11}, {"r12", ctx->R12}, {"r13", ctx->R13},
        {"r14", ctx->R14}, {"r15", ctx->R15}, {"rip", ctx->Rip},
    };
    crash_write_str(out, "registers:\n");
    for (size_t i = 0; i < sizeof(regs) / sizeof(regs[0]); ++i) {
        crash_write_str(out, "  ");
        crash_write_str(out, regs[i].name);
        crash_write_str(out, " = 0x");
        crash_write_hex(out, regs[i].val, 16);
        crash_write_str(out, "\n");
    }
#else
    (void)out;
#endif
}

// Dump base address + image size of every loaded module so the
// reader can map an absolute crash address (RIP / stack frame PC)
// to an RVA = pc - base. Without this the operator has to know the
// ASLR base separately. Uses GetModuleHandleEx + GetModuleFileNameA;
// both can take loader-lock but the SEH filter is synchronous so
// the loader is in a defined state.
void crash_dump_modules(HANDLE out) {
    crash_write_str(out, "modules:\n");
    HMODULE mods[256];
    DWORD needed = 0;
    HANDLE proc = GetCurrentProcess();
    // EnumProcessModules lives in psapi (loaded on demand). Use
    // K32EnumProcessModules from kernel32 directly (Windows 7+) to
    // avoid the link-time dep -- it has the identical signature.
    using EnumModsFn = BOOL(WINAPI*)(HANDLE, HMODULE*, DWORD, LPDWORD);
    HMODULE k32 = GetModuleHandleA("kernel32.dll");
    if (!k32) return;
    auto pEnum = reinterpret_cast<EnumModsFn>(
        GetProcAddress(k32, "K32EnumProcessModules"));
    if (!pEnum) return;
    if (!pEnum(proc, mods, sizeof(mods), &needed)) return;
    DWORD count = needed / sizeof(HMODULE);
    if (count > 256) count = 256;
    for (DWORD i = 0; i < count; ++i) {
        MODULEINFO mi{};
        using GetInfoFn = BOOL(WINAPI*)(HANDLE, HMODULE, LPMODULEINFO, DWORD);
        auto pInfo = reinterpret_cast<GetInfoFn>(
            GetProcAddress(k32, "K32GetModuleInformation"));
        if (pInfo && pInfo(proc, mods[i], &mi, sizeof(mi))) {
            char name[260];
            using GetNameFn = DWORD(WINAPI*)(HANDLE, HMODULE, LPSTR, DWORD);
            auto pName = reinterpret_cast<GetNameFn>(
                GetProcAddress(k32, "K32GetModuleBaseNameA"));
            name[0] = '\0';
            if (pName) pName(proc, mods[i], name, sizeof(name));
            crash_write_str(out, "  0x");
            crash_write_hex(out,
                            reinterpret_cast<uintptr_t>(mi.lpBaseOfDll), 16);
            crash_write_str(out, " size=0x");
            crash_write_hex(out, mi.SizeOfImage, 0);
            crash_write_str(out, " ");
            crash_write_str(out, name[0] ? name : "(unknown)");
            crash_write_str(out, "\n");
        }
    }
}

// Append the trailing kTailBytes of the session log to the crash
// report. Captures the milestones immediately preceding the fault,
// which is what a forensic reader actually wants. Bounded so a
// gigabyte-sized log can not blow up the crash dump.
void crash_append_log_tail(HANDLE out, const char* src_path) {
    if (!src_path || !src_path[0]) return;
    constexpr DWORD kTailBytes = 16 * 1024;  // 16 KB of recent log
    HANDLE in = CreateFileA(src_path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                            nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (in == INVALID_HANDLE_VALUE) {
        crash_write_str(out, "(session log not available)\n");
        return;
    }
    LARGE_INTEGER size{};
    if (!GetFileSizeEx(in, &size)) {
        CloseHandle(in);
        return;
    }
    LARGE_INTEGER seek_to{};
    if (size.QuadPart > static_cast<LONGLONG>(kTailBytes)) {
        seek_to.QuadPart = size.QuadPart - kTailBytes;
        SetFilePointerEx(in, seek_to, nullptr, FILE_BEGIN);
    }
    static char buf[16 * 1024];
    DWORD read = 0;
    if (ReadFile(in, buf, sizeof(buf), &read, nullptr) && read > 0) {
        DWORD written = 0;
        WriteFile(out, buf, read, &written, nullptr);
    }
    CloseHandle(in);
}

LONG WINAPI windows_crash_filter(EXCEPTION_POINTERS* ep) {
    // Re-entry guard. SetUnhandledExceptionFilter is the last-chance
    // handler so it normally runs once, but a secondary fault inside
    // dbghelp's StackWalk64 / SymFromAddr (both of which DO allocate
    // and are not technically async-signal-safe) can re-enter the
    // filter on the same thread. The static atomic_flag wins exactly
    // once; every subsequent re-entry hands the exception back to the
    // OS by returning EXCEPTION_EXECUTE_HANDLER so the C runtime's own
    // terminate path can finish the job instead of looping the filter.
    static std::atomic_flag in_handler = ATOMIC_FLAG_INIT;
    if (in_handler.test_and_set()) {
        return EXCEPTION_EXECUTE_HANDLER;
    }

    CrashPaths& cp = crash_paths();
    if (!cp.installed.load()) return EXCEPTION_EXECUTE_HANDLER;

    HANDLE out = CreateFileA(cp.crash_log,
                             GENERIC_WRITE, FILE_SHARE_READ,
                             nullptr, CREATE_ALWAYS,
                             FILE_ATTRIBUTE_NORMAL, nullptr);
    if (out == INVALID_HANDLE_VALUE) {
        return EXCEPTION_EXECUTE_HANDLER;
    }

    crash_write_str(out, "=== theCollider crash report ===\n");
    crash_write_str(out, "exception_code = 0x");
    crash_write_hex(out, ep->ExceptionRecord->ExceptionCode, 8);
    crash_write_str(out, "\nexception_addr = 0x");
    crash_write_hex(out,
                    reinterpret_cast<uintptr_t>(
                        ep->ExceptionRecord->ExceptionAddress),
                    16);
    crash_write_str(out, "\n");

    // Exception-record parameter dump. For AV (0xC0000005) the spec
    // is: NumberParameters >= 2, [0]=type (0 read, 1 write, 8 DEP),
    // [1]=faulting VA. Emit raw values so a reader can interpret.
    crash_write_str(out, "exception_param_count = ");
    crash_write_u64(out, ep->ExceptionRecord->NumberParameters);
    crash_write_str(out, "\n");
    for (DWORD i = 0;
         i < ep->ExceptionRecord->NumberParameters && i < EXCEPTION_MAXIMUM_PARAMETERS;
         ++i) {
        crash_write_str(out, "exception_param[");
        crash_write_u64(out, i);
        crash_write_str(out, "] = 0x");
        crash_write_hex(out, ep->ExceptionRecord->ExceptionInformation[i], 16);
        crash_write_str(out, "\n");
    }

    auto uptime = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - cp.install_at).count();
    crash_write_str(out, "uptime_ms = ");
    crash_write_u64(out, static_cast<unsigned long long>(uptime));
    crash_write_str(out, "\n");

    // Register dump from the CONTEXT record. Reading a GPR here is
    // cheaper than re-running with a debugger attached.
    crash_dump_registers(out, ep->ContextRecord);

    // Module base + size table. The crash reader can compute any
    // absolute address's RVA as (addr - module_base) and feed that
    // into a PDB resolver without needing the runtime image base.
    crash_dump_modules(out);

    // Best-effort stack walk. SymInitialize / StackWalk64 / SymFromAddr
    // do allocate, so technically they are not async-signal-safe. SEH
    // filters run on the same thread that faulted but with the
    // dispatcher's own stack, so we accept the risk of a secondary
    // crash inside dbghelp to get a stack trace. The crash report up
    // to this point has already been flushed, so a dbghelp failure
    // does not lose the headers.
    crash_write_str(out, "stack:\n");
    HANDLE process = GetCurrentProcess();
    HANDLE thread = GetCurrentThread();
    SymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_UNDNAME);
    SymInitialize(process, nullptr, TRUE);

    CONTEXT* ctx = ep->ContextRecord;
    STACKFRAME64 frame{};
    DWORD machine_type;
#ifdef _M_X64
    machine_type = IMAGE_FILE_MACHINE_AMD64;
    frame.AddrPC.Offset = ctx->Rip;
    frame.AddrPC.Mode = AddrModeFlat;
    frame.AddrFrame.Offset = ctx->Rbp;
    frame.AddrFrame.Mode = AddrModeFlat;
    frame.AddrStack.Offset = ctx->Rsp;
    frame.AddrStack.Mode = AddrModeFlat;
#elif defined(_M_IX86)
    machine_type = IMAGE_FILE_MACHINE_I386;
    frame.AddrPC.Offset = ctx->Eip;
    frame.AddrPC.Mode = AddrModeFlat;
    frame.AddrFrame.Offset = ctx->Ebp;
    frame.AddrFrame.Mode = AddrModeFlat;
    frame.AddrStack.Offset = ctx->Esp;
    frame.AddrStack.Mode = AddrModeFlat;
#else
    machine_type = IMAGE_FILE_MACHINE_UNKNOWN;
#endif

    for (int i = 0; i < 64; ++i) {
        if (!StackWalk64(machine_type, process, thread, &frame, ctx,
                         nullptr, SymFunctionTableAccess64,
                         SymGetModuleBase64, nullptr)) {
            break;
        }
        if (frame.AddrPC.Offset == 0) break;

        crash_write_str(out, "  #");
        crash_write_u64(out, static_cast<unsigned long long>(i));
        crash_write_str(out, " 0x");
        crash_write_hex(out, frame.AddrPC.Offset, 16);

        // Symbol resolution (best effort).
        char buf[sizeof(SYMBOL_INFO) + 256];
        SYMBOL_INFO* sym = reinterpret_cast<SYMBOL_INFO*>(buf);
        std::memset(buf, 0, sizeof(buf));
        sym->SizeOfStruct = sizeof(SYMBOL_INFO);
        sym->MaxNameLen = 255;
        DWORD64 displacement = 0;
        if (SymFromAddr(process, frame.AddrPC.Offset, &displacement, sym)) {
            crash_write_str(out, " ");
            crash_write_str(out, sym->Name);
            crash_write_str(out, "+0x");
            crash_write_hex(out, displacement, 0);
        }
        crash_write_str(out, "\n");
    }

    crash_write_str(out, "\n--- session_state.json (verbatim) ---\n");
    crash_append_file(out, cp.state_json);
    crash_write_str(out, "\n--- end session_state.json ---\n");

    // Tail of the active session log. Captures the milestones
    // emitted in the seconds immediately before the fault. For
    // crashes inside a known hot path (GPU dispatch, network rx,
    // bloom load) this is usually the diagnostic record the
    // operator actually wants.
    crash_write_str(out, "\n--- session log tail ---\n");
    crash_append_log_tail(out, cp.session_log);
    crash_write_str(out, "\n--- end session log tail ---\n");

    crash_write_str(out, "=== end crash report ===\n");

    FlushFileBuffers(out);
    CloseHandle(out);

    // TerminateProcess (not return EXCEPTION_EXECUTE_HANDLER + fall
    // through) so we do not loop if the C runtime's own teardown
    // triggers another fault.
    TerminateProcess(GetCurrentProcess(),
                     ep->ExceptionRecord->ExceptionCode);
    return EXCEPTION_EXECUTE_HANDLER;
}

#else  // POSIX

void posix_crash_handler(int signo, siginfo_t* info, void* /*ucontext*/) {
    // Re-entry guard. If a second async-signal hits while the first
    // invocation of this handler is mid-write (or worse: dbghelp /
    // backtrace itself triggered a secondary fault), test_and_set wins
    // exactly once and every later signal bypasses the handler entirely.
    // Without this, a double-fault loops the handler on itself, racing
    // file descriptors and never giving the OS its default-disposition
    // chance to produce a core dump. The static-flag form is the
    // documented async-signal-safe idiom; std::atomic_flag's load/store
    // is required by the standard to be lock-free and signal-safe.
    static std::atomic_flag in_handler = ATOMIC_FLAG_INIT;
    if (in_handler.test_and_set()) {
        signal(signo, SIG_DFL);
        raise(signo);
        return;
    }

    CrashPaths& cp = crash_paths();
    if (!cp.installed.load()) {
        // Re-raise with the default handler so the OS can produce its
        // own diagnostic (core dump, etc.).
        signal(signo, SIG_DFL);
        raise(signo);
        return;
    }

    int fd = ::open(cp.crash_log, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) {
        signal(signo, SIG_DFL);
        raise(signo);
        return;
    }

    crash_write_str(fd, "=== theCollider crash report ===\n");
    crash_write_str(fd, "signal = ");
    crash_write_u64(fd, static_cast<unsigned long long>(signo));
    crash_write_str(fd, "\nsi_code = ");
    if (info) crash_write_u64(fd, static_cast<unsigned long long>(info->si_code));
    crash_write_str(fd, "\nsi_addr = 0x");
    if (info) crash_write_hex(fd, reinterpret_cast<uintptr_t>(info->si_addr), 16);
    crash_write_str(fd, "\n");

    auto uptime = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - cp.install_at).count();
    crash_write_str(fd, "uptime_ms = ");
    crash_write_u64(fd, static_cast<unsigned long long>(uptime));
    crash_write_str(fd, "\n");

#ifdef __GLIBC__
    crash_write_str(fd, "stack:\n");
    void* frames[64];
    int n = backtrace(frames, 64);
    // backtrace_symbols allocates; backtrace_symbols_fd writes directly
    // and is async-signal-safe.
    backtrace_symbols_fd(frames, n, fd);
#else
    crash_write_str(fd, "stack: (backtrace() not available on this libc)\n");
#endif

    crash_write_str(fd, "\n--- session_state.json (verbatim) ---\n");
    crash_append_file(fd, cp.state_json);
    crash_write_str(fd, "\n--- end session_state.json ---\n");
    crash_write_str(fd, "=== end crash report ===\n");

    ::close(fd);

    // Re-raise with the default handler so the kernel can still
    // produce its own core dump for offline debugging.
    signal(signo, SIG_DFL);
    raise(signo);
}

#endif

// Copy a std::string into a fixed buffer with NUL termination,
// truncating if needed. Used by install_crash_handler() to populate
// the pre-allocated paths.
void copy_to_buf(char* buf, size_t cap, const std::string& s) {
    if (cap == 0) return;
    size_t n = std::min(s.size(), cap - 1);
    std::memcpy(buf, s.data(), n);
    buf[n] = '\0';
}

}  // anonymous namespace

// ===========================================================================
// Test-only detail::* trampolines
// ===========================================================================
//
// The three helpers below were anonymous-namespace TU locals. The
// 2026-05-17 review asked for direct test coverage of argv redaction,
// the SessionState JSON shape, and the atomic-write durability
// contract, so we expose them through collider::log::detail via thin
// pass-through wrappers. The production call sites (write_startup_banner,
// SessionStateStore::write_to_disk) keep their existing anonymous-NS
// references, so no link-island shuffling or visibility changes to the
// rest of the TU.
namespace detail {

// Each wrapper calls the anonymous-namespace implementation by
// unqualified name; the anonymous namespace is enclosed in
// collider::log so the lookup succeeds within this TU. The wrappers
// add no logic; they exist purely to give the test suite (which lives
// in a separate TU) a callable handle.
std::string redact_argv(int argc, char** argv) {
    return ::collider::log::redact_argv(argc, argv);
}

std::string serialize_state(const SessionState& s) {
    return ::collider::log::serialize_state(s);
}

bool atomic_write_state_file(const std::filesystem::path& path,
                             const std::string& content) {
    return ::collider::log::atomic_write_state_file(path, content);
}

}  // namespace detail

// ===========================================================================
// Public API
// ===========================================================================

bool init_session_log() {
    bool ok = SessionLogSink::instance().initialize();
    if (ok) {
        // Flip the SessionStateStore out of "no-op" mode so subsequent
        // milestone()/update_session_state() calls actually persist to
        // ~/.collider/session_state.json. Test binaries that never call
        // init_session_log() stay in no-op mode and therefore never
        // touch the operator's real state file.
        SessionStateStore::instance().enable_disk_writes();
        // Register a one-shot atexit hook so a clean exit (normal
        // return from main, std::exit, std::quick_exit on POSIX) always
        // force-flushes the SessionStateStore. The Meyers singleton's
        // destructor closes the log file on its own, but the state.json
        // is owned by a separate singleton with no observable destructor
        // side effect; without this hook a process that exits between
        // two debounce windows leaves a stale state.json on disk.
        // std::atexit only runs on clean exit (not on a signal or
        // std::_Exit), which is exactly the right scope: crash paths
        // are already covered by the SEH / sigaction filter that dumps
        // the verbatim state.json into the crash log.
        static std::atomic_flag atexit_registered = ATOMIC_FLAG_INIT;
        if (!atexit_registered.test_and_set()) {
            std::atexit(&shutdown_session_log);
        }
    }
    return ok;
}

void write_startup_banner(int argc, char** argv, const Arguments& args) {
    auto& sink = SessionLogSink::instance();
    if (!sink.is_initialized()) return;

    std::ostringstream oss;
    oss << "version=" << collider::kVersion << '\n';
    oss << "build_flags=" << build_flags() << '\n';
    oss << "compiled=" << __DATE__ << ' ' << __TIME__ << '\n';
#ifdef COLLIDER_GIT_SHA
    oss << "git_sha=" << COLLIDER_GIT_SHA << '\n';
#else
    oss << "git_sha=(unset)\n";
#endif
    oss << "pid=" << current_pid() << '\n';
    oss << "boot_ts=" << format_ts_iso(sink.boot_ts()) << '\n';
    std::error_code ec;
    auto cwd = std::filesystem::current_path(ec);
    oss << "cwd=" << (ec ? std::string("(unknown)") : cwd.string()) << '\n';
    oss << "config_file=" << (args.config_file.empty()
                              ? std::string("(default)")
                              : args.config_file)
        << '\n';
    oss << "argv=" << redact_argv(argc, argv) << '\n';

    sink.write_block("STARTUP", oss.str());

    // Also seed the SessionState minimally so the very first crash
    // (before any mode runner had a chance to call update) still has
    // a real PID + log_path in the JSON.
    SessionState seed;
    seed.mode = "starting";
    seed.pid = current_pid();
    seed.log_path = sink.log_path();
    seed.boot_ts = sink.boot_ts();
    seed.last_update_ts = std::chrono::system_clock::now();
    SessionStateStore::instance().update(seed);
    SessionStateStore::instance().flush();
}

void write_hardware_enum(const std::vector<platform::DeviceInfo>& devices) {
    auto& sink = SessionLogSink::instance();
    if (!sink.is_initialized()) return;

    std::ostringstream oss;
    oss << "device_count=" << devices.size() << '\n';
    for (const auto& d : devices) {
        oss << "device " << d.device_id << ": "
            << "name=\"" << d.name << "\" "
            << "vendor=\"" << d.vendor << "\" "
            << "sm=" << d.compute_major << '.' << d.compute_minor << ' '
            << "sms=" << d.multiprocessor_count << ' '
            << "vram_total_mb=" << (d.total_memory / (1024 * 1024)) << ' '
            << "vram_free_mb=" << (d.available_memory / (1024 * 1024)) << ' '
            << "fp16=" << (d.supports_fp16 ? 1 : 0) << ' '
            << "int8=" << (d.supports_int8 ? 1 : 0)
            << '\n';
    }

#ifdef COLLIDER_HAS_NVML
    oss << "nvml=linked\n";
#else
    oss << "nvml=absent\n";
#endif
#ifdef COLLIDER_CUDA_ARCH_LIST
    oss << "compile_arch_list=" << COLLIDER_CUDA_ARCH_LIST << '\n';
#endif

    sink.write_block("HARDWARE", oss.str());

    // T4.2 follow-up (2026-05-17): the header has long promised a
    // milestone("sm_mismatch", ...) for every device whose SM is missing
    // from the compile-time arch list, but the body only emitted
    // compile_arch_list and never fired the milestone. Wire it now so the
    // session log carries a forensic record of PTX-JIT fallback events
    // alongside the operator-facing stderr warning that detect_gpus()
    // prints at startup. Both fire on purpose: stderr is for the human
    // running the binary, the milestone is for after-the-fact log
    // forensics when a perf complaint comes in days later.
    //
    // Parser is shared with runtime/gpu_detection.cpp via the helper in
    // runtime/gpu_detection.hpp; duplicating the parser here would risk
    // the two emitting different verdicts after a future
    // COLLIDER_CUDA_ARCH_LIST format change.
    const auto compiled_sm_set = ::compile_time_sm_set();
    if (!compiled_sm_set.empty()) {
        const std::string compile_list_str = ::sm_set_to_string(compiled_sm_set);
        for (const auto& d : devices) {
            // Apple Silicon reports a Metal version in compute_major /
            // compute_minor (e.g. 3.0), not a CUDA SM. Skip it the same
            // way detect_gpus() does so we do not fire a spurious
            // sm_mismatch on Mac builds.
            if (d.is_apple_silicon) continue;
            std::pair<int, int> device_sm{d.compute_major, d.compute_minor};
            if (compiled_sm_set.find(device_sm) == compiled_sm_set.end()) {
                std::ostringstream detail;
                detail << "device=" << d.device_id
                       << " name=\"" << d.name << "\""
                       << " observed_sm=" << device_sm.first << '.'
                       << device_sm.second
                       << " compile_arch_list=" << compile_list_str;
                milestone("sm_mismatch", detail.str());
            }
        }
    }
}

void milestone(const char* event, const std::string& detail) {
    if (!event) return;
    SessionLogSink::instance().write_milestone(event, detail);
    // Every milestone forces a state flush so the JSON snapshot the
    // crash dump would read is fresh enough to match the last logged
    // event.
    SessionStateStore::instance().flush();
}

void update_session_state(const SessionState& state) {
    SessionStateStore::instance().update(state);
}

void flush_session_state() {
    SessionStateStore::instance().flush();
}

void shutdown_session_log() {
    // Idempotency: once-flag wins exactly once across the std::atexit
    // hook + any explicit test-side call. The order is "flush state
    // first, then milestone the shutdown" so the final on-disk state.json
    // already reflects the SHUTDOWN milestone's wall-clock by the time
    // a crash-handler reader would parse the log.
    static std::atomic_flag once = ATOMIC_FLAG_INIT;
    if (once.test_and_set()) return;
    SessionStateStore::instance().flush();
    SessionLogSink::instance().write_milestone(
        "shutdown_session_log", "explicit teardown");
}

void install_crash_handler() {
    CrashPaths& cp = crash_paths();
    bool expected = false;
    if (!cp.installed.compare_exchange_strong(expected, true)) return;

    cp.install_at = std::chrono::steady_clock::now();

    // Pre-compute crash-log + session-state paths once so the handler
    // does not allocate. The crash filename embeds the install
    // timestamp + PID so concurrent processes (or successive crash
    // dumps from one badly-behaved run) do not collide.
    auto& sink = SessionLogSink::instance();
    auto ts = format_ts_for_filename(sink.boot_ts());
    auto crash_path = (collider::paths::collider_home() /
                       ("crash-" + ts + "-" +
                        std::to_string(current_pid()) + ".log"))
                          .string();
    auto state_path = SessionStateStore::instance().state_path().string();
    copy_to_buf(cp.crash_log, sizeof(cp.crash_log), crash_path);
    copy_to_buf(cp.state_json, sizeof(cp.state_json), state_path);
    copy_to_buf(cp.session_log, sizeof(cp.session_log), sink.log_path());

#ifdef _WIN32
    SetUnhandledExceptionFilter(windows_crash_filter);
#else
    struct sigaction sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = posix_crash_handler;
    sa.sa_flags = SA_SIGINFO | SA_RESETHAND;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGABRT, &sa, nullptr);
    sigaction(SIGFPE,  &sa, nullptr);
    sigaction(SIGBUS,  &sa, nullptr);
#endif

    milestone("crash_handler_installed",
              std::string("crash_log=") + crash_path);
}

}  // namespace collider::log
