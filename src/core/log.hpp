// log.hpp - cross-thread stdout/stderr serialization helper (T4.3).
//
// Policy: any code that emits a multi-line or multi-chunk log line from
// more than one thread MUST take collider::log::mu() for the duration of
// the line, or use the ScopedLine RAII below. The streaming insertion
// operators in C++ iostreams are not atomic across threads; without this
// guard a pool reconnect banner can interleave with a GPU OOM warning and
// produce an unreadable hybrid line that operators can't grep.
//
// Migration policy: this header intentionally does NOT auto-migrate every
// existing std::cout / std::cerr call site. The existing sites are mostly
// single-threaded (TUI render thread, scan loop main thread, signal
// handler) and adding the lock everywhere would add contention without
// preventing a bug we have. Use ScopedLine for any NEW multi-thread
// log line. When an existing site is shown to interleave in practice,
// migrate that site in isolation.
//
// Usage:
//   #include "core/log.hpp"
//   ...
//   {
//     collider::log::ScopedLine line(std::cerr);
//     line << "[gpu " << device_id << "] OOM at " << bytes << " bytes\n";
//   }   // mutex released, stream flushed
//
// The mutex is a Meyers-style function-local static, which gives the
// helper a single process-wide instance with no static-init-order
// dependency. The mutex is shared between std::cout and std::cerr
// callers; iostream policy ties them to the same underlying tty FD on
// every platform we ship, so a single lock serializes both streams.
#pragma once

#include <mutex>
#include <ostream>

namespace collider::log {

inline std::mutex& mu() {
    static std::mutex m;
    return m;
}

// RAII helper. Locks collider::log::mu() on construction, releases on
// destruction, flushes the stream as the last act before unlock so the
// composed line hits the OS write() buffer before another thread can
// interleave. The operator<< chain is the standard ostream pattern;
// every call returns the same ScopedLine& so the user can chain.
struct ScopedLine {
    std::lock_guard<std::mutex> lk;
    std::ostream& os;

    explicit ScopedLine(std::ostream& o) : lk(mu()), os(o) {}

    // Non-copyable, non-movable: the lock_guard owns the mutex and the
    // ostream reference must stay tied to this scope.
    ScopedLine(const ScopedLine&) = delete;
    ScopedLine& operator=(const ScopedLine&) = delete;
    ScopedLine(ScopedLine&&) = delete;
    ScopedLine& operator=(ScopedLine&&) = delete;

    template <typename T>
    ScopedLine& operator<<(const T& v) {
        os << v;
        return *this;
    }

    // ostream manipulators (std::endl, std::flush, std::hex, etc.) are
    // function pointers; the templated operator above would deduce the
    // wrong type, so we forward them through an explicit overload.
    ScopedLine& operator<<(std::ostream& (*manip)(std::ostream&)) {
        os << manip;
        return *this;
    }

    ~ScopedLine() { os << std::flush; }
};

}  // namespace collider::log
