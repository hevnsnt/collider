// stdio_capture.hpp -- FREE STUB (installed by scripts/sync-to-free.sh)
//
// The Pro tree provides a heavyweight StdioCapture that hijacks std::cout
// / std::cerr rdbufs AND dup2-redirects fd 1/2 into a pipe drained by a
// background thread, so post-menu boot text lands in
// ~/.collider/logs/tui-boot-<ts>.log instead of bleeding onto the FTXUI
// alt-screen. None of that applies in the free build: there is no
// FTXUI dashboard to bleed into, so the operator's terminal stays
// cooked the whole time and stdio just goes where it normally goes.
//
// The stub keeps the symbol surface that pool_solver.cpp /
// puzzle_solver*.cpp expect (ctor + the static no-op helpers) so those
// translation units compile without any #ifdef gating at call sites.
// Every method is a no-op; release_active_capture_silent / release_
// active_capture / is_underlying_stdout_tty fall through to plain
// stdio behavior (the latter just returns the underlying isatty
// result).
#pragma once

#include <cstdio>
#include <filesystem>
#include <string>

#ifdef _WIN32
    #include <io.h>
    #define COLLIDER_STDIO_ISATTY  _isatty
    #define COLLIDER_STDIO_FILENO  _fileno
#else
    #include <unistd.h>
    #define COLLIDER_STDIO_ISATTY  ::isatty
    #define COLLIDER_STDIO_FILENO  ::fileno
#endif

namespace collider::ui::tui {

class StdioCapture {
public:
    StdioCapture() = default;
    explicit StdioCapture(const std::filesystem::path& /*log_dir*/) {}
    ~StdioCapture() = default;

    StdioCapture(const StdioCapture&) = delete;
    StdioCapture& operator=(const StdioCapture&) = delete;

    static void release_active_capture() noexcept {}
    static void release_active_capture_silent() noexcept {}
    static StdioCapture* current() noexcept { return nullptr; }
    static bool is_underlying_stdout_tty() noexcept {
        return COLLIDER_STDIO_ISATTY(COLLIDER_STDIO_FILENO(stdout)) != 0;
    }

    std::string drain() { return {}; }
};

}  // namespace collider::ui::tui
