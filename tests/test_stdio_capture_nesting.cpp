// test_stdio_capture_nesting.cpp -- T1-H regression test for the
// stack-based StdioCapture::current() introduced 2026-05-23.
//
// Pre-fix: ctor wrote `current_ = this`, dtor cleared only if
// `current_ == this`. A nested second StdioCapture overwrote the
// outer's pointer; on inner dtor, `current_` went to nullptr and the
// outer's pointer was permanently lost. Any runner that called
// StdioCapture::current() after the inner scope returned saw nullptr
// even though the outer was still alive.
//
// Post-fix: ctor pushes onto a mutex-guarded stack. current() returns
// top-of-stack. ~ctor pops self (regardless of position) so the stack
// stays consistent if destruction order is unusual.

#include "ui/tui/stdio_capture.hpp"

#include <cstdio>
#include <cstdlib>

using ::collider::ui::tui::StdioCapture;

static int g_failures = 0;
static int g_passes   = 0;

static void check(const char* tag, bool ok) {
    if (ok) {
        std::printf("[ ok  ] %s\n", tag);
        ++g_passes;
    } else {
        std::fprintf(stderr, "[FAIL] %s\n", tag);
        ++g_failures;
    }
}

int main() {
    std::printf("=== test_stdio_capture_nesting (T1-H) ===\n");

    // 1. No captures installed -> current() returns nullptr.
    check("no_capture_returns_null", StdioCapture::current() == nullptr);

    // 2. Single capture: current() returns it; after scope exit,
    // current() returns nullptr again.
    {
        StdioCapture outer;
        check("single_capture_visible", StdioCapture::current() == &outer);
    }
    check("single_capture_cleared", StdioCapture::current() == nullptr);

    // 3. Nested captures: inner takes priority via top-of-stack. When
    // inner exits, outer becomes current again. THIS is the case that
    // regressed pre-fix: outer's pointer was lost when inner exited.
    //
    // It also exercises the fd-redirect drain-thread teardown across a
    // nested construct/destruct: each ctor dup2's the pipe write end
    // onto fd 1 / fd 2 and spawns a drain thread; the inner ctor dup()s
    // the (currently pipe-aliased) fd 1 / fd 2 into its own saved fds,
    // which keeps the OUTER pipe's write end open. The outer dtor must
    // therefore wake its drain thread explicitly (sentinel byte) rather
    // than wait for an EOF that the inner's aliases prevent. We emit
    // real output inside each scope so the drain threads have something
    // to capture and the teardown is not a trivial no-op.
    {
        StdioCapture outer;
        std::printf("outer-line-via-c-stdio\n");
        std::fflush(stdout);
        check("outer_visible_pre_inner", StdioCapture::current() == &outer);
        {
            StdioCapture inner;
            std::printf("inner-line-via-c-stdio\n");
            std::fflush(stdout);
            check("inner_overrides_outer",
                  StdioCapture::current() == &inner);
            check("outer_not_current_during_inner",
                  StdioCapture::current() != &outer);
        }  // inner dtor: must terminate inner's drain thread promptly.
        check("outer_restored_post_inner",
              StdioCapture::current() == &outer);
        std::printf("outer-line-after-inner\n");
        std::fflush(stdout);
    }  // outer dtor: must terminate outer's drain thread despite the now
       // -destroyed inner having previously aliased its write end.
    check("nesting_fully_cleared", StdioCapture::current() == nullptr);

    // 4. release_active_capture() called when no capture is installed
    // is a safe no-op (must not crash, must not throw).
    StdioCapture::release_active_capture();
    check("release_when_empty_is_safe", true);

    // 5. Out-of-order destruction (synthetic; not the normal pattern
    // but the dtor must not corrupt the stack if it ever happens).
    {
        auto* a = new StdioCapture();
        auto* b = new StdioCapture();
        check("b_top_after_both_ctors", StdioCapture::current() == b);
        delete a;  // pop a (the lower element)
        check("b_still_top_after_a_destroyed",
              StdioCapture::current() == b);
        delete b;
        check("empty_after_both_destroyed",
              StdioCapture::current() == nullptr);
    }

    std::printf("\n%d passes, %d failures\n", g_passes, g_failures);
    return g_failures == 0 ? 0 : 1;
}
