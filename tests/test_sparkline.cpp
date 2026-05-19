// phase 1 (builder-panels: baseline-panels).
//
// Sparkline widget tests. Exercises the pure-function render path (no FTXUI
// Screen needed) via render_string() so the assertions are independent of
// the terminal column accounting that Screen::ToString() applies. Five
// scenarios:
//
//   1. Empty data: returns a string of opts.width spaces (no crash, no
//      partial UTF-8 sequences).
//   2. All-zero data: every output column is a space (the documented
//      "no signal yet" visual).
//   3. Linear ramp [0..N]: in Unicode mode the rendered string contains
//      multiple distinct block glyphs from the U+2581..U+2588 ramp,
//      confirming the bucket math walks the full range.
//   4. ASCII fallback mode (forced via the options flag) renders only
//      ASCII characters and uses at least two distinct levels of the
//      kAsciiRamp.
//   5. Data longer than opts.width: the rendered string corresponds to
//      ONLY the last opts.width samples (newer values dominate).

#include "ui/tui/widgets/sparkline.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int fail(const char* what) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    return 1;
}

// Count how many bytes in `s` are NOT ASCII spaces. Helpful for asserting
// "the output is entirely whitespace" without depending on UTF-8 byte width.
size_t count_non_space(const std::string& s) {
    size_t n = 0;
    for (char c : s) if (c != ' ') ++n;
    return n;
}

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

}  // namespace

int main() {
    using namespace collider::ui::tui::widgets;

    // =================================================================
    // 1. Empty data.
    // =================================================================
    {
        SparklineOptions opts;
        opts.width = 16;
        opts.unicode_blocks = false;
        std::string out = render_string({}, opts);
        if (out.size() != static_cast<size_t>(opts.width)) {
            return fail("empty: width mismatch");
        }
        if (count_non_space(out) != 0) {
            return fail("empty: expected all spaces");
        }
    }

    // =================================================================
    // 2. All zeros.
    // =================================================================
    {
        SparklineOptions opts;
        opts.width = 10;
        opts.unicode_blocks = false;
        std::vector<double> data(opts.width, 0.0);
        std::string out = render_string(data, opts);
        if (out.size() != static_cast<size_t>(opts.width)) {
            return fail("all zeros: width mismatch");
        }
        if (count_non_space(out) != 0) {
            return fail("all zeros: expected all spaces");
        }
    }

    // =================================================================
    // 3. Linear ramp in Unicode mode must populate nearly the full
    //    U+2581..U+2588 ramp. Bucket math: floor(v / max * 8) clamped
    //    into [0,7]. For data = [0..10] with max=10:
    //       v=0  -> space (zero_threshold)
    //       v=1  -> idx 0
    //       v=2  -> idx 1
    //       v=3  -> idx 2
    //       v=4  -> idx 3
    //       v=5  -> idx 4
    //       v=6  -> idx 4 (collision with v=5)
    //       v=7  -> idx 5
    //       v=8  -> idx 6
    //       v=9  -> idx 7
    //       v=10 -> idx 7 (clamped)
    //    Distinct indices touched: {0,1,2,3,4,5,6,7} = 8.
    //    Assertion is tight at >= 7 so a future regression that
    //    collapses six of the eight buckets to one glyph (the original
    //    >= 3 threshold did not catch this) trips immediately. We allow
    //    7 instead of demanding 8 to absorb a single legitimate
    //    collision from rounding on alternative platforms; collapsing
    //    further is a real defect.
    // =================================================================
    {
        SparklineOptions opts;
        opts.width = 11;                  // matches data length
        opts.unicode_blocks = true;
        // Force the unicode path on Windows ctest where the default
        // console code page may not be 65001.
#if defined(_WIN32)
        _putenv_s("COLLIDER_UNICODE", "1");
        _putenv_s("COLLIDER_ASCII", "");
#else
        setenv("COLLIDER_UNICODE", "1", 1);
        unsetenv("COLLIDER_ASCII");
#endif
        // The unicode_available cache is initialized lazily on first
        // call. Skip the assertion on hosts where the cache was already
        // populated to false before we set the env var; we still verify
        // the ASCII path below.
        if (unicode_available()) {
            std::vector<double> data;
            for (int i = 0; i <= 10; ++i) data.push_back(static_cast<double>(i));
            std::string out = render_string(data, opts);
            // The leftmost column corresponds to value 0 (gap, ASCII space)
            // and the remainder ramps up. Verify the bucket math walks the
            // full ramp without collapsing.
            const std::string ramp[] = {
                "\xE2\x96\x81", "\xE2\x96\x82", "\xE2\x96\x83",
                "\xE2\x96\x84", "\xE2\x96\x85", "\xE2\x96\x86",
                "\xE2\x96\x87", "\xE2\x96\x88",
            };
            int distinct = 0;
            for (const auto& g : ramp) {
                if (contains(out, g)) ++distinct;
            }
            if (distinct < 7) {
                return fail("unicode ramp: expected >= 7 distinct glyphs");
            }
        }
    }

    // =================================================================
    // 4. ASCII fallback. Forced via options (does not depend on env).
    //    Asserts every output byte is in the printable ASCII range and
    //    that the bucket math walks nearly the full ramp.
    //
    //    Bucket math for data = [1..8] with max=8:
    //       v=1 -> idx 1, v=2 -> idx 2, ..., v=7 -> idx 7,
    //       v=8 -> idx 7 (clamped).
    //    Distinct indices touched: {1,2,3,4,5,6,7} = 7 distinct ASCII
    //    ramp chars. Assertion tightened from >= 2 to >= 6 so a
    //    bucket-collapse regression that drops the rendered output to
    //    five or fewer distinct glyphs trips immediately. We allow 6
    //    instead of demanding 7 to absorb a single legitimate
    //    rounding collision on an alternative platform; collapsing
    //    further is a real defect.
    // =================================================================
    {
        SparklineOptions opts;
        opts.width = 8;
        opts.unicode_blocks = false;
        std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        std::string out = render_string(data, opts);
        if (out.size() != static_cast<size_t>(opts.width)) {
            return fail("ascii: width mismatch");
        }
        bool all_ascii = true;
        for (unsigned char c : out) {
            if (c > 0x7F) { all_ascii = false; break; }
        }
        if (!all_ascii) return fail("ascii: non-ASCII byte present");
        int distinct = 0;
        const char ramp[] = {'.', ':', '-', '=', '+', '*', '#', '@'};
        for (char r : ramp) if (out.find(r) != std::string::npos) ++distinct;
        if (distinct < 6) return fail("ascii: expected >= 6 distinct ramp chars");
    }

    // =================================================================
    // 5. Oversize data. Only the last opts.width samples render. Build
    //    a ramp where only the trailing slice is non-zero and verify the
    //    rendered string ends in the highest-intensity glyph while the
    //    older slice (zeros) is dropped.
    // =================================================================
    {
        SparklineOptions opts;
        opts.width = 4;
        opts.unicode_blocks = false;
        // 10 samples but width is 4; only the last 4 (values 6,7,8,9)
        // should render. With max=9 and value=9 the last column should
        // be the topmost ASCII ramp char '@'.
        std::vector<double> data = {0, 0, 0, 0, 0, 0, 6, 7, 8, 9};
        std::string out = render_string(data, opts);
        if (out.size() != 4u) return fail("oversize: width mismatch");
        if (out.back() != '@') {
            return fail("oversize: last column should be top of ramp");
        }
        // First column corresponds to value 6 / max 9 = 0.66 -> bucket 5
        // -> ramp[5] == '*'. Anything in the leading-non-space range is
        // acceptable; the strict assertion is that NO column is a space
        // (because every sample is > zero_threshold).
        if (count_non_space(out) != 4u) {
            return fail("oversize: every column should be non-space");
        }
    }

    std::printf("test_sparkline: OK\n");
    return 0;
}
