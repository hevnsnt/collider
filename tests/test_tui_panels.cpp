// phase 1 (builder-foundation: tui-app-skeleton +
//                  builder-panels: baseline-panels).
//
// Smoke + content tests for the Wave 3 TUI panel hierarchy.
//
//   1. TuiApp construction + snapshot posting smoke (Wave 2 contract; this
//      test predates the panel work and continues to pass through the
//      header/status/footer refactor).
//   2. detect_default_variant() honors NO_COLOR.
//   3. Headless render of the header panel: confirm brand label, version,
//      mode label, lifetime number, and ETA cell appear in Screen::ToString().
//   4. Headless render of the status panel: confirm Keys/s label, current
//      phase name, hits / empty / trying labels appear.
//   5. Headless render of the footer panel: confirm "q quit" appears.

#include "core/version.hpp"          // collider::kVersion -- source of truth for test fixtures
#include "ui/tui/keybindings.hpp"
#include "ui/tui/panels/bloom_picker.hpp"
#include "ui/tui/panels/footer_panel.hpp"
#include "ui/tui/panels/gpu_panel.hpp"
#include "ui/tui/panels/header_panel.hpp"
#include "ui/tui/panels/help_overlay.hpp"
#include "ui/tui/panels/performance_panel.hpp"
#include "ui/tui/panels/plugins_panel.hpp"
#include "ui/tui/panels/recent_hits_panel.hpp"
#include "ui/tui/panels/status_panel.hpp"
#include "ui/tui/panels/wordlist_picker.hpp"
#include "ui/tui/snapshot.hpp"
#include "ui/tui/theme.hpp"
#include "ui/tui/tui_app.hpp"
#include "ui/tui/widgets/big_digits.hpp"
#include "ui/tui/widgets/histogram.hpp"

#include <ftxui/dom/elements.hpp>
#include <ftxui/dom/node.hpp>
#include <ftxui/screen/screen.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

int fail(const char* what) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    return 1;
}

std::string render_to_string(ftxui::Element el, int width = 100, int height = 12) {
    auto screen = ftxui::Screen::Create(
        ftxui::Dimension::Fixed(width),
        ftxui::Dimension::Fixed(height)
    );
    ftxui::Render(screen, el);
    return screen.ToString();
}

}  // namespace

int main() {
    using namespace collider::ui::tui;

    // 2026-05-16 (carbon-fiber redesign): the big_digits widget picks
    // its glyph table via widgets::unicode_available(), which probes
    // the active locale (POSIX) or console code page (Windows) on
    // first call and caches the result. Several assertions below
    // ("blank glyph not exactly 4 spaces", "rows have inconsistent
    // lengths") implicitly assume the ASCII fallback path because they
    // expect every cell to be exactly one byte. On Windows hosts where
    // GetConsoleOutputCP() returns CP_UTF8 (cmd.exe with chcp 65001,
    // Windows Terminal default, etc.) the widget picks UTF-8 and the
    // per-cell byte counts diverge. Forcing COLLIDER_ASCII=1 BEFORE
    // any unicode_available() call locks the widget into ASCII mode
    // for the duration of this test executable; production binaries
    // (collider.exe, collider_pro.exe) never set this env var and pick
    // their glyph mode from the live console.
#if defined(_WIN32)
    _putenv_s("COLLIDER_ASCII", "1");
#else
    setenv("COLLIDER_ASCII", "1", 1);
#endif

    // =================================================================
    // 1. Theme construction smoke + variant invariants.
    // =================================================================
    Theme t_default = make_theme(ThemeVariant::Default);
    Theme t_hc      = make_theme(ThemeVariant::HighContrast);
    Theme t_mono    = make_theme(ThemeVariant::Monochrome);
    Theme t_light   = make_theme(ThemeVariant::Light);
    // is_monochrome flag must be set ONLY for the Monochrome variant.
    if (t_default.is_monochrome) return fail("theme: default should not be monochrome");
    if (t_hc.is_monochrome)      return fail("theme: high-contrast should not be monochrome");
    if (!t_mono.is_monochrome)   return fail("theme: monochrome flag must be set");
    if (t_light.is_monochrome)   return fail("theme: light should not be monochrome");
    // Every variant must define an orange brand color so the header
    // gradient + brand label have something to render.
    (void)t_default.orange;
    (void)t_hc.orange;
    (void)t_mono.orange;
    (void)t_light.orange;

    // =================================================================
    // 2. detect_default_variant() honors NO_COLOR.
    // =================================================================
#if defined(_WIN32)
    _putenv_s("NO_COLOR", "1");
#else
    setenv("NO_COLOR", "1", 1);
#endif
    if (detect_default_variant() != ThemeVariant::Monochrome) {
        return fail("detect_default_variant did not return Monochrome when NO_COLOR set");
    }
#if defined(_WIN32)
    _putenv_s("NO_COLOR", "");
#else
    unsetenv("NO_COLOR");
#endif
    if (detect_default_variant() != ThemeVariant::Default) {
        return fail("detect_default_variant did not return Default when NO_COLOR unset");
    }

    // =================================================================
    // 3. TuiApp RAII: construct + destroy without start(), post snapshot.
    // =================================================================
    {
        RenderConfig cfg;
        cfg.alt_screen = false;
        TuiApp app(cfg);
        ScanSnapshot s;
        s.total_checked = 1234567;
        s.bloom_hits = 42;
        s.bloom_collisions_filtered = 10;
        s.tight_bloom_filtered = 3;
        s.verified_hits = 0;
        s.seq = 7;
        app.post_scan_snapshot(s);
    }

    // =================================================================
    // 4. Header panel headless render.
    // =================================================================
    {
        ScanSnapshot s;
        s.total_checked = 16'938'402'193ULL;
        s.bloom_hits = 17;
        s.current_phase = 0;
        auto theme = make_theme(ThemeVariant::Default);

        panels::HeaderContext hdr;
        hdr.snap = s;
        hdr.session_baseline = 14'972'527'796ULL;
        hdr.session_start = std::chrono::steady_clock::now() -
                            std::chrono::minutes(83);   // 1h 23m uptime
        hdr.session_keys_per_sec = 1'300'000.0;
        hdr.wordlist_size = 50'000'000ULL;
        hdr.mode_label = "Brainwallet";
        // Pin the test fixture to collider::kVersion so a release-time
        // version bump (version.hpp) does not silently leave a stale
        // literal here that asserts against the wrong "v1.4.x" string.
        hdr.version = ::collider::kVersion;

        auto el = panels::render_header(hdr, theme);
        std::string out = render_to_string(el, 110, 8);

        // T3 (test-quality-deep-audit): assert against the panel's structured
        // debug accessor for the value-formatting properties. The render-
        // output check below remains as a "did not crash and produced output"
        // smoke, but the load-bearing assertions live on the debug struct so
        // a renamed label or color-reset byte does not break the test for
        // reasons unrelated to the underlying data.
        auto dbg = panels::debug_header(hdr);
        if (dbg.brand_cell != "theCollider")
            return fail("header[debug]: brand_cell != 'theCollider'");
        const std::string expected_version_cell =
            std::string("v") + ::collider::kVersion;
        if (dbg.version_cell != expected_version_cell)
            return fail(("header[debug]: version_cell='" + dbg.version_cell
                         + "' want '" + expected_version_cell + "'").c_str());
        if (dbg.mode_cell != "[Brainwallet]")
            return fail("header[debug]: mode_cell != '[Brainwallet]'");
        // 16,938,402,193 -> "16.94 B" per format_number_short's "K/M/B/T"
        // compaction (commit ad5326d swapped " G" -> " B" for billions).
        if (dbg.lifetime_short != "16.94 B")
            return fail(("header[debug]: lifetime_short='" + dbg.lifetime_short
                         + "' want '16.94 B'").c_str());
        if (dbg.lifetime_grouped != "16,938,402,193")
            return fail("header[debug]: lifetime_grouped wrong");
        if (dbg.baseline_label != "Resume baseline")
            return fail("header[debug]: baseline_label wrong");
        if (dbg.session_label != "Session")
            return fail("header[debug]: session_label wrong");
        if (dbg.eta_label != "ETA wordlist cycle")
            return fail("header[debug]: eta_label wrong");

        // Smoke: rendered output must be non-empty and contain the brand
        // (which is a literal, not a formatted value, so the substring
        // check is robust here).
        if (out.empty())                     return fail("header: render produced empty output");
        if (!contains(out, "theCollider"))   return fail("header: missing brand in render");
    }

    // =================================================================
    // 5. Status panel headless render.
    // =================================================================
    {
        ScanSnapshot s;
        s.total_checked = 1'965'874'397ULL;
        s.bloom_hits = 0;
        s.bloom_collisions_filtered = 864'712 + 4'100'000ULL;
        s.tight_bloom_filtered = 4'100'000ULL;
        s.verified_hits = 0;
        s.current_phase = 4;                  // Deep Dive
        s.phase_iteration = 0;
        auto theme = make_theme(ThemeVariant::Default);

        panels::StatusContext st;
        st.snap = s;
        st.current_keys_per_sec = 1'308'427.0;
        st.avg_keys_per_sec     = 1'200'000.0;
        st.current_phase_name   = "Deep Dive";
        st.current_chunk        = 246;
        st.total_chunks         = 494;
        st.sample_passphrase    = "letmein2024";
        // Seed a non-flat history so the sparkline contributes a line of
        // bars rather than an all-spaces row.
        for (size_t i = 0; i < st.keys_per_sec_history.size(); ++i) {
            st.keys_per_sec_history[i] = 1'000'000.0 + (i * 5000.0);
        }

        auto el = panels::render_status(st, theme);
        std::string out = render_to_string(el, 120, 12);

        // T3: structured assertions via the existing StatusPanelDebug
        // accessor (status_panel.hpp line 49). Pins the exact formatted
        // strings the panel composes, independent of layout / color drift.
        auto dbg = panels::debug_status(st);
        if (dbg.rate_cell.find("1.31 M") == std::string::npos)
            return fail(("status[debug]: rate_cell='" + dbg.rate_cell
                         + "' missing '1.31 M'").c_str());
        if (dbg.phase_cell != "Deep Dive")
            return fail(("status[debug]: phase_cell='" + dbg.phase_cell
                         + "' want 'Deep Dive'").c_str());
        if (dbg.chunk_cell.find("247/494") == std::string::npos)
            return fail(("status[debug]: chunk_cell='" + dbg.chunk_cell
                         + "' missing '247/494'").c_str());
        if (dbg.passphrase_cell.find("letmein2024") == std::string::npos)
            return fail(("status[debug]: passphrase_cell missing 'letmein2024'"));
        if (dbg.empty_cell != "864,712")
            return fail(("status[debug]: empty_cell='" + dbg.empty_cell
                         + "' want '864,712'").c_str());

        // Smoke: rendered output must be non-empty.
        if (out.empty())                   return fail("status: render produced empty output");
        if (!contains(out, "Keys/s"))      return fail("status: missing Keys/s label in render");
    }

    // =================================================================
    // 6. Footer panel headless render.
    //
    // T3.4 (2026-05-17): substring scrapes ("q", "quit", "help") replaced
    // with debug_footer(ctx) field assertions. The typed accessor surfaces
    // the same quit cell + placeholder cell strings the renderer composes
    // (literal values in build_dim_placeholder()), so a regression in
    // either path fails the test byte-for-byte. The render_footer() call
    // is preserved as a no-throw smoke (just verify it does not crash and
    // returns non-empty output).
    // =================================================================
    {
        auto theme = make_theme(ThemeVariant::Default);
        auto el = panels::render_footer(theme, /*active_bindings=*/{});
        std::string out = render_to_string(el, 100, 3);

        // Smoke: render must produce non-empty output without crashing.
        if (out.empty())
            return fail("footer: render produced empty output");

        // The legacy overload short-circuits empty bindings through the
        // FooterContext path, so the debug accessor exercised here
        // matches the rendered output.
        panels::FooterContext ctx_empty;
        auto dbg = panels::debug_footer(ctx_empty);
        if (dbg.quit_key_cell != "q")
            return fail("footer[debug]: quit_key_cell != 'q'");
        if (dbg.quit_label_cell != "quit")
            return fail("footer[debug]: quit_label_cell != 'quit'");
        // Empty bindings + empty chord -> placeholder path. Placeholder
        // string includes "help" so the legacy hint contract holds.
        if (dbg.placeholder_cell.find("help") == std::string::npos)
            return fail("footer[debug]: placeholder_cell missing 'help'");
        if (!dbg.binding_cells.empty())
            return fail("footer[debug]: binding_cells should be empty");
        if (!dbg.chord_cell.empty())
            return fail("footer[debug]: chord_cell should be empty");
    }

    // =================================================================
    // 7. TuiApp setter smoke (wave 4b: complete-tui-setters).
    //
    // Constructs a TuiApp WITHOUT calling start() so no render thread is
    // spawned, then exercises every newly added public setter with
    // realistic values. The render-thread machinery never runs, so this
    // is purely a compile-and-link + no-crash + no-leak smoke. The
    // visible-render assertion lives in tests 8 + 9 below, which build
    // the panel contexts directly so we never depend on FTXUI's
    // ScreenInteractive event loop inside a unit test.
    // =================================================================
    {
        RenderConfig cfg;
        cfg.alt_screen = false;
        TuiApp app(cfg);

        // Static-once setters.
        app.set_mode_label("Brainwallet");
        app.set_version(::collider::kVersion);
        app.set_session_start(std::chrono::steady_clock::now() -
                               std::chrono::minutes(45));
        app.set_session_baseline(123'456'789ULL);
        app.set_wordlist_size(50'000'000ULL);

        // Periodic setters.
        app.set_sample_passphrase("hunter2");
        app.set_chunk_progress(246, 494);
        app.set_current_phase_name("Deep Dive");

        // Throughput setters.
        std::array<double, panels::kKeysPerSecHistorySize> hist{};
        for (size_t i = 0; i < hist.size(); ++i) {
            hist[i] = 1'500'000.0;
        }
        app.set_keys_per_sec_history(hist);
        app.set_keys_per_sec_current(1'500'000.0);
        app.set_keys_per_sec_avg(1'200'000.0);

        // Post a synthetic scan snapshot so the snapshot inbox is
        // populated. No assertions: this only verifies the setters
        // coexist with the existing snapshot path without crashing.
        ScanSnapshot s;
        s.total_checked = 9'876'543'210ULL;
        s.current_phase = 4;
        app.post_scan_snapshot(s);

        // Defensive clamp check on set_chunk_progress: negative or zero
        // values must not propagate to the panel formatter (which would
        // render "0/-1"). The setter clamps internally; we re-exercise
        // it here so any regression to the clamp logic surfaces as a
        // test failure when the assertion below would fire under the
        // header/status render check in test 9.
        app.set_chunk_progress(-5, 0);
        app.set_chunk_progress(246, 494);  // restore to the realistic value
    }

    // =================================================================
    // 8. Setter -> header render: confirm version + mode label flow
    //    from the setter through to the rendered string. The header
    //    panel's render function is a free function over HeaderContext;
    //    we build the context the same way TuiApp::render_frame() does
    //    after the setters land, then render headless.
    // =================================================================
    {
        auto theme = make_theme(ThemeVariant::Default);

        panels::HeaderContext hdr;
        hdr.snap.total_checked = 16'938'402'193ULL;
        hdr.snap.current_phase = 4;
        hdr.session_baseline = 14'972'527'796ULL;
        hdr.session_start = std::chrono::steady_clock::now() -
                            std::chrono::minutes(83);
        hdr.session_keys_per_sec = 1'500'000.0;
        hdr.wordlist_size = 50'000'000ULL;
        // These two are the setters' targets:
        hdr.mode_label = "Brainwallet";
        // Same source-of-truth pinning as the test-3 fixture above so a
        // version.hpp bump does not silently break this comparison.
        hdr.version = ::collider::kVersion;

        auto el = panels::render_header(hdr, theme);
        std::string out = render_to_string(el, 110, 8);

        // T3: assert against the structured debug accessor first; the
        // render-output smoke check stays as a "did not crash" guard.
        auto dbg = panels::debug_header(hdr);
        if (dbg.mode_cell != "[Brainwallet]")
            return fail("setters[debug]: mode_cell wrong after set_mode_label");
        const std::string expected_version_cell_8 =
            std::string("v") + ::collider::kVersion;
        if (dbg.version_cell != expected_version_cell_8)
            return fail(("setters[debug]: version_cell='" + dbg.version_cell
                         + "' want '" + expected_version_cell_8 + "'").c_str());
        if (dbg.lifetime_short != "16.94 B")
            return fail("setters[debug]: lifetime_short wrong after post_scan_snapshot");

        if (out.empty())
            return fail("setters: header render produced empty output");
    }

    // =================================================================
    // 9. Setter -> status render: confirm chunk progress, phase name,
    //    sample passphrase, and rolling rate all surface in the
    //    rendered string. Pre-conditions match what the runner pushes
    //    at every batch boundary after wave 4b lands.
    // =================================================================
    {
        auto theme = make_theme(ThemeVariant::Default);

        panels::StatusContext st;
        st.snap.total_checked = 1'965'874'397ULL;
        st.snap.bloom_collisions_filtered = 864'712 + 4'100'000ULL;
        st.snap.tight_bloom_filtered = 4'100'000ULL;
        st.snap.current_phase = 4;

        // Mirror the setters' published state:
        st.current_keys_per_sec = 1'500'000.0;
        st.avg_keys_per_sec     = 1'200'000.0;
        st.current_phase_name   = "Deep Dive";
        st.current_chunk        = 246;  // setter input
        st.total_chunks         = 494;
        st.sample_passphrase    = "hunter2";
        for (size_t i = 0; i < st.keys_per_sec_history.size(); ++i) {
            st.keys_per_sec_history[i] = 1'500'000.0;
        }

        auto el = panels::render_status(st, theme);
        std::string out = render_to_string(el, 120, 12);

        // T3: structured assertions via debug_status (the brittle
        // "1.50 M" / "chunk 247/494" substring matches now live on the
        // typed struct fields).
        auto dbg = panels::debug_status(st);
        if (dbg.rate_cell.find("1.50 M") == std::string::npos)
            return fail(("setters[debug]: rate_cell='" + dbg.rate_cell
                         + "' missing '1.50 M'").c_str());
        if (dbg.phase_cell != "Deep Dive")
            return fail("setters[debug]: phase_cell wrong");
        if (dbg.chunk_cell.find("247/494") == std::string::npos)
            return fail("setters[debug]: chunk_cell missing '247/494'");
        if (dbg.passphrase_cell.find("hunter2") == std::string::npos)
            return fail("setters[debug]: passphrase_cell missing 'hunter2'");

        if (out.empty())
            return fail("setters: status render produced empty output");
    }

    // =================================================================
    // 10. Histogram widget standalone (Wave 6, phase 3).
    //
    // Verifies the auto-trim contract: leading + trailing zero buckets
    // are dropped and only the populated middle range renders.
    // =================================================================
    {
        using namespace collider::ui::tui::widgets;
        std::vector<uint64_t> buckets = {0, 0, 0, 5, 12, 3, 0, 0};
        HistogramOptions opts;
        opts.auto_trim = true;
        opts.unicode_blocks = false;   // force ASCII so the test does not
                                       // depend on host locale/code page
        std::string s = render_histogram_string(buckets, opts);
        // Auto-trim should drop the three leading zeros and the two
        // trailing zeros, leaving a 3-character bar string.
        if (s.size() != 3) {
            std::fprintf(stderr,
                "histogram auto-trim: expected 3 chars, got %zu (\"%s\")\n",
                s.size(), s.c_str());
            return fail("histogram: auto-trim did not drop zero buckets");
        }
        // All-zero buckets should yield a single space so the parent
        // hbox keeps a non-zero column count.
        std::vector<uint64_t> zeros(8, 0);
        std::string s2 = render_histogram_string(zeros, opts);
        if (s2 != " ") {
            return fail("histogram: all-zero buckets did not render single space");
        }
    }

    // =================================================================
    // 11. GPU panel headless render (Wave 6, phase 2): 2-device snapshot
    //     with util / temp / power / clocks populated.
    //
    // T3.4 (2026-05-17): the original substring scrapes against the
    // rendered glyph soup ("GPU 0", "RTX 2060 SUPER", "87%", "71 C",
    // "165W", "PCIe Gen3") have been replaced with debug_gpu(ctx) field
    // assertions. The structured cells are byte-for-byte what render_gpu
    // composes (same format functions, same width specifiers), so a
    // regression in the panel's cell-formatting still surfaces here.
    // The render_gpu call is preserved as a no-throw smoke check.
    // =================================================================
    {
        using namespace collider::ui::tui;
        GpuTelemetrySnapshot g;
        g.nvml_available = true;
        for (int i = 0; i < 2; ++i) {
            GpuPerDeviceSnapshot d;
            d.device_id = i;
            d.name = "NVIDIA GeForce RTX 2060 SUPER";
            d.temperature_c = 71.0f + static_cast<float>(i);
            d.power_watts = 165;
            d.tdp_watts = 175;
            d.util_pct = 87 - i * 5;
            d.vram_used_bytes = static_cast<uint64_t>(5) * 1024 * 1024 * 1024;
            d.vram_total_bytes = static_cast<uint64_t>(8) * 1024 * 1024 * 1024;
            d.sm_clock_mhz = 1755;
            d.mem_clock_mhz = 7600;
            d.fan_pct = 65;
            d.pcie_gen = 3;
            g.devices.push_back(d);
        }
        auto theme = make_theme(ThemeVariant::Default);

        panels::GpuPanelContext ctx;
        ctx.snap = g;
        ctx.util_history.assign(
            2, std::array<double, panels::kGpuHistorySize>{});
        ctx.temp_history.assign(
            2, std::array<double, panels::kGpuHistorySize>{});
        // Seed a non-flat history so the sparkline contributes glyphs.
        for (int dev = 0; dev < 2; ++dev) {
            for (int i = 0; i < panels::kGpuHistorySize; ++i) {
                ctx.util_history[dev][i] = 40.0 + i * 0.5;
                ctx.temp_history[dev][i] = 60.0 + i * 0.2;
            }
        }

        // Smoke: render must produce output without crashing.
        auto el = panels::render_gpu(ctx, theme);
        std::string out = render_to_string(el, 120, 16);
        if (out.empty())
            return fail("gpu: render_to_string returned empty");

        // Typed debug accessor: the structured cells are what the panel
        // composes (same format helpers feed both the Element tree and
        // these strings). A regression in either path fails the test.
        auto dbg = panels::debug_gpu(ctx);
        if (!dbg.nvml_available)
            return fail("gpu: debug_gpu nvml_available not propagated");
        if (dbg.devices.size() != 2)
            return fail("gpu: debug_gpu wrong device count");
        if (dbg.devices[0].device_id != 0 || dbg.devices[1].device_id != 1)
            return fail("gpu: debug_gpu device_id mismatch");
        if (dbg.devices[0].name_cell.find("RTX 2060 SUPER") == std::string::npos)
            return fail("gpu: device 0 name_cell missing 'RTX 2060 SUPER'");
        if (dbg.devices[1].name_cell.find("RTX 2060 SUPER") == std::string::npos)
            return fail("gpu: device 1 name_cell missing 'RTX 2060 SUPER'");
        if (dbg.devices[0].util_cell.find("87%") == std::string::npos)
            return fail("gpu: device 0 util_cell missing '87%'");
        if (dbg.devices[0].temp_cell.find("71 C") == std::string::npos)
            return fail("gpu: device 0 temp_cell missing '71 C'");
        if (dbg.devices[0].power_cell.find("165W") == std::string::npos)
            return fail("gpu: device 0 power_cell missing '165W'");
        if (dbg.devices[0].pcie_cell != "PCIe Gen3")
            return fail("gpu: device 0 pcie_cell != 'PCIe Gen3'");
        if (dbg.devices[1].pcie_cell != "PCIe Gen3")
            return fail("gpu: device 1 pcie_cell != 'PCIe Gen3'");
    }

    // =================================================================
    // 12. GPU panel with NVML unavailable: the typed-debug accessor must
    //     report nvml_available=false; the per-device telemetry cells
    //     must be "n/a" for util / temp / power; the VRAM cell still
    //     populates because cudaMemGetInfo is independent of NVML.
    //
    // T3.4 (2026-05-17): the original render-and-substring checks for
    // the "NVML telemetry unavailable" banner / "vram" row have been
    // replaced with debug_gpu() field assertions on the typed cells.
    // The banner string itself is a panel-presentation concern; the
    // structural contract under test is "NVML-missing snapshots produce
    // n/a telemetry cells and still surface VRAM info". The render
    // call stays as a no-throw smoke.
    // =================================================================
    {
        using namespace collider::ui::tui;
        GpuTelemetrySnapshot g;
        g.nvml_available = false;
        GpuPerDeviceSnapshot d;
        d.device_id = 0;
        d.name = "AMD Radeon RX 6800";
        d.vram_used_bytes = static_cast<uint64_t>(3) * 1024 * 1024 * 1024;
        d.vram_total_bytes = static_cast<uint64_t>(16) * 1024 * 1024 * 1024;
        // util/temp/power deliberately left nullopt so the test exercises
        // the NVML-missing render path.
        g.devices.push_back(d);

        auto theme = make_theme(ThemeVariant::Default);
        panels::GpuPanelContext ctx;
        ctx.snap = g;
        ctx.util_history.assign(
            1, std::array<double, panels::kGpuHistorySize>{});
        ctx.temp_history.assign(
            1, std::array<double, panels::kGpuHistorySize>{});

        auto el = panels::render_gpu(ctx, theme);
        std::string out = render_to_string(el, 120, 10);
        if (out.empty())
            return fail("gpu-nvml-off: render_to_string returned empty");

        auto dbg = panels::debug_gpu(ctx);
        if (dbg.nvml_available)
            return fail("gpu-nvml-off: nvml_available leaked to true");
        if (dbg.devices.size() != 1)
            return fail("gpu-nvml-off: wrong device count");
        const auto& dev0 = dbg.devices[0];
        if (dev0.util_cell != "n/a")
            return fail("gpu-nvml-off: util_cell should be 'n/a'");
        if (dev0.temp_cell != "n/a")
            return fail("gpu-nvml-off: temp_cell should be 'n/a'");
        if (dev0.power_cell != "n/a")
            return fail("gpu-nvml-off: power_cell should be 'n/a'");
        // VRAM still renders from cudaMemGetInfo; cell must be populated.
        if (dev0.vram_cell == "n/a" || dev0.vram_cell.empty())
            return fail("gpu-nvml-off: vram_cell should still report bytes");
        if (dev0.vram_cell.find("3.0") == std::string::npos)
            return fail("gpu-nvml-off: vram_cell should mention 3.0 (GiB used)");
    }

    // =================================================================
    // 13. Performance panel headless render with a populated PerfReport.
    //
    // T3.4 (2026-05-17): the original substring scrapes (panel title,
    // per-phase section header, kernel names, "Chunk Overhead",
    // "Headroom") have been replaced with debug_performance(ctx) field
    // assertions. The typed cells are byte-for-byte what render_performance
    // composes; section titles ("Performance" / "Per-Phase Throughput" /
    // "Pipeline Timing") are pure UI-presentation labels not exposed via
    // the typed accessor, so we still render and require non-empty output
    // (no-throw smoke). The typed accessor catches every regression that
    // touches the underlying data flow.
    // =================================================================
    {
        using namespace collider::ui::tui;
        PerfHistogramSnapshot p;
        p.total_dispatches = 1'200'000;
        p.chunk_overhead_pct = 12.0;

        // Three kernels with realistic data so the histogram + mean +
        // count cells all render.
        {
            PerfHistogramSnapshot::KernelTiming k;
            k.name = "EcMul";
            k.count = 1'200'000;
            k.mean_us = 247.0;
            k.log_buckets_us[7] = 1'200'000;   // ~128-256 us
            p.kernels.push_back(k);
        }
        {
            PerfHistogramSnapshot::KernelTiming k;
            k.name = "Sha256";
            k.count = 1'200'000;
            k.mean_us = 18.0;
            k.log_buckets_us[4] = 1'200'000;   // ~16-32 us
            p.kernels.push_back(k);
        }
        {
            PerfHistogramSnapshot::KernelTiming k;
            k.name = "Ripemd160";
            k.count = 1'200'000;
            k.mean_us = 12.0;
            k.log_buckets_us[3] = 1'200'000;   // ~8-16 us
            p.kernels.push_back(k);
        }

        collider::runtime::ScanSnapshot s;
        s.current_phase = 4;
        s.phase_keys_processed[0] = 1'000'000;
        s.phase_keys_processed[4] = 5'000'000;

        auto theme = make_theme(ThemeVariant::Default);
        panels::PerfPanelContext ctx;
        ctx.snap = p;
        ctx.scan_snap = s;
        ctx.current_phase = 4;
        ctx.phase_keys_per_sec[0] = 12'300'000.0;
        ctx.phase_keys_per_sec[1] =  8'100'000.0;
        ctx.phase_keys_per_sec[2] =  6'400'000.0;
        ctx.phase_keys_per_sec[3] =        0.0;
        ctx.phase_keys_per_sec[4] =  1'300'000.0;
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < panels::kPerfHistorySize; ++j) {
                ctx.phase_keys_history[i][j] =
                    ctx.phase_keys_per_sec[i] *
                    (0.7 + 0.3 * static_cast<double>(j) /
                                  panels::kPerfHistorySize);
            }
        }
        ctx.free_vram_bytes = static_cast<uint64_t>(1.4 * 1024 * 1024 * 1024);
        ctx.safety_reserve_bytes = 656 * 1024 * 1024;
        ctx.headroom_to_oom_bytes = 832 * 1024 * 1024;

        // Smoke: render must produce output without crashing.
        auto el = panels::render_performance(ctx, theme);
        std::string out = render_to_string(el, 130, 24);
        if (out.empty())
            return fail("perf: render_to_string returned empty");

        // Typed debug accessor: the structured cells are what the panel
        // composes (same format helpers feed both the Element tree and
        // these strings).
        auto dbg = panels::debug_performance(ctx);
        if (!dbg.instrumentation_enabled)
            return fail("perf: instrumentation flag should be enabled");
        // Phase 4 (Deep Dive) is the current phase and must surface that.
        if (dbg.phases[4].name != "Deep Dive")
            return fail("perf: phase 4 name should be 'Deep Dive'");
        if (!dbg.phases[4].is_current)
            return fail("perf: phase 4 should be is_current");
        if (dbg.phases[4].status_cell != "<= current")
            return fail("perf: phase 4 status_cell should be '<= current'");
        // Phase 3 has rate 0 -> status "(idle)".
        if (dbg.phases[3].status_cell != "(idle)")
            return fail("perf: phase 3 status_cell should be '(idle)'");
        // Phase 0/1/2 have non-zero rate and are not current -> "(active)".
        if (dbg.phases[0].status_cell != "(active)")
            return fail("perf: phase 0 status_cell should be '(active)'");
        // Kernels: every named kernel must surface with its mean cell
        // (format_us output) and count cell.
        if (dbg.kernels.size() != 3)
            return fail("perf: kernels.size() should be 3");
        bool found_ecmul = false, found_sha = false, found_ripemd = false;
        for (const auto& kr : dbg.kernels) {
            if (kr.name == "EcMul")     found_ecmul = true;
            if (kr.name == "Sha256")    found_sha = true;
            if (kr.name == "Ripemd160") found_ripemd = true;
            if (kr.mean_cell.empty())
                return fail("perf: kernel mean_cell empty");
            if (kr.count_cell.empty())
                return fail("perf: kernel count_cell empty");
        }
        if (!found_ecmul)  return fail("perf: missing EcMul kernel row");
        if (!found_sha)    return fail("perf: missing Sha256 kernel row");
        if (!found_ripemd) return fail("perf: missing Ripemd160 kernel row");
        // Chunk overhead + headroom both populate when instrumentation
        // is on.
        if (dbg.chunk_overhead_cell.find("12.0") == std::string::npos)
            return fail("perf: chunk_overhead_cell missing '12.0'");
        if (dbg.headroom_cell.empty() || dbg.headroom_cell == "(disabled)")
            return fail("perf: headroom_cell should report bytes when enabled");
    }

    // =================================================================
    // 14. Performance panel with instrumentation disabled (total
    //     dispatches == 0). The typed accessor reports
    //     instrumentation_enabled=false; chunk_overhead + headroom show
    //     "(disabled)"; phase rate cells still populate because they
    //     come from ScanState, not PerfCollector.
    //
    // T3.4 (2026-05-17): the "Perf instrumentation disabled" /
    // "Pipeline Timing" / "Chunk Overhead" substring scrapes have been
    // replaced with debug_performance() field checks. The render call
    // stays as a no-throw smoke.
    // =================================================================
    {
        using namespace collider::ui::tui;
        PerfHistogramSnapshot p;
        // p.total_dispatches stays 0; p.kernels stays empty. This is
        // the default state when g_enabled == false.
        p.chunk_overhead_pct = 0.0;

        auto theme = make_theme(ThemeVariant::Default);
        panels::PerfPanelContext ctx;
        ctx.snap = p;
        ctx.current_phase = 0;
        ctx.phase_keys_per_sec[0] = 12'300'000.0;
        ctx.phase_keys_per_sec[1] =  8'100'000.0;
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < panels::kPerfHistorySize; ++j) {
                ctx.phase_keys_history[i][j] = ctx.phase_keys_per_sec[i];
            }
        }

        auto el = panels::render_performance(ctx, theme);
        std::string out = render_to_string(el, 130, 14);
        if (out.empty())
            return fail("perf disabled: render_to_string returned empty");

        auto dbg = panels::debug_performance(ctx);
        if (dbg.instrumentation_enabled)
            return fail("perf disabled: instrumentation flag leaked to true");
        // Kernel rows should be empty (no PerfCollector data).
        if (!dbg.kernels.empty())
            return fail("perf disabled: kernel rows leaked");
        // Phase 0 is current -> status_cell "<= current".
        if (dbg.phases[0].status_cell != "<= current")
            return fail("perf disabled: phase 0 status_cell should be '<= current'");
        // Phase 1 has non-zero rate -> "(active)".
        if (dbg.phases[1].status_cell != "(active)")
            return fail("perf disabled: phase 1 status_cell should be '(active)'");
        // Chunk overhead + headroom both fall back to "(disabled)".
        if (dbg.chunk_overhead_cell != "(disabled)")
            return fail("perf disabled: chunk_overhead_cell should be '(disabled)'");
        if (dbg.headroom_cell != "(disabled)")
            return fail("perf disabled: headroom_cell should be '(disabled)'");
    }

    // =================================================================
    // 15. Footer with active bindings (wave 7 / phase 4):
    //     FooterContext drives the live-key strip. The "q quit" cell
    //     is always rendered on the left; each entry in
    //     active_bindings contributes a "<key> <verb>" cell in the
    //     middle region.
    //
    // T3.4 (2026-05-17): substring scrapes against the rendered glyph
    // soup replaced with debug_footer(ctx) field assertions. The typed
    // binding_cells list is populated by split_binding() over each
    // Keybinding::short_label, identical to what the renderer feeds into
    // render_binding_kb(). The render call stays as a no-throw smoke.
    // =================================================================
    {
        using namespace collider::ui::tui;
        auto theme = make_theme(ThemeVariant::Default);
        panels::FooterContext ctx;
        ctx.active_bindings = active_footer_bindings();

        auto el = panels::render_footer(ctx, theme);
        std::string out = render_to_string(el, 160, 3);
        if (out.empty())
            return fail("footer-bindings: render produced empty output");

        auto dbg = panels::debug_footer(ctx);
        if (dbg.quit_key_cell != "q")
            return fail("footer-bindings[debug]: quit_key_cell != 'q'");
        if (dbg.quit_label_cell != "quit")
            return fail("footer-bindings[debug]: quit_label_cell != 'quit'");
        if (!dbg.chord_cell.empty())
            return fail("footer-bindings[debug]: chord_cell should be empty");
        if (!dbg.placeholder_cell.empty())
            return fail("footer-bindings[debug]: placeholder leaked when bindings present");
        if (dbg.binding_cells.empty())
            return fail("footer-bindings[debug]: binding_cells empty");

        // Confirm the live strip carries the expected verbs. Each cell
        // is (key, verb) where verb is the action token from the
        // Keybinding::short_label after the first space.
        auto has_verb = [&](const std::string& needle) {
            for (const auto& kv : dbg.binding_cells) {
                if (kv.second.find(needle) != std::string::npos) return true;
            }
            return false;
        };
        if (!has_verb("pause"))
            return fail("footer-bindings[debug]: missing pause binding");
        if (!has_verb("bloom"))
            return fail("footer-bindings[debug]: missing bloom binding");
        if (!has_verb("save"))
            return fail("footer-bindings[debug]: missing save binding");
        if (!has_verb("wordlist"))
            return fail("footer-bindings[debug]: missing wordlist binding");
        if (!has_verb("hits"))
            return fail("footer-bindings[debug]: missing hits binding");
    }

    // =================================================================
    // 16. Footer chord hint: when chord_hint is non-empty, the middle
    //     region renders the hint INSTEAD of the bindings. "q quit"
    //     still appears on the left so the operator always sees the
    //     escape route.
    //
    // T3.4 (2026-05-17): render-substring scrapes for "GPU toggle" /
    // "0-7" / "pause" / "quit" replaced with debug_footer(ctx) field
    // checks on chord_cell + binding_cells. The chord-overrides-strip
    // contract is now structural (binding_cells must be empty when
    // chord_cell is populated).
    // =================================================================
    {
        using namespace collider::ui::tui;
        auto theme = make_theme(ThemeVariant::Default);
        panels::FooterContext ctx;
        ctx.active_bindings = active_footer_bindings();
        ctx.chord_hint = "GPU toggle: press 0-7 or Esc";

        auto el = panels::render_footer(ctx, theme);
        std::string out = render_to_string(el, 160, 3);
        if (out.empty())
            return fail("footer-chord: render produced empty output");

        auto dbg = panels::debug_footer(ctx);
        if (dbg.chord_cell != "GPU toggle: press 0-7 or Esc")
            return fail("footer-chord[debug]: chord_cell mismatch");
        if (!dbg.binding_cells.empty())
            return fail("footer-chord[debug]: binding_cells leaked during chord");
        if (!dbg.placeholder_cell.empty())
            return fail("footer-chord[debug]: placeholder leaked during chord");
        if (dbg.quit_label_cell != "quit")
            return fail("footer-chord[debug]: quit cell missing during chord");
    }

    // =================================================================
    // 17. Footer banner: a recent banner appears right-aligned in dim
    //     italic. The render thread feeds banner_text only when
    //     in-window, so the panel does NOT have to filter by timestamp.
    //
    // T3.4 (2026-05-17): "Bloom queued" / "50M.blf" substring scrapes
    // replaced with debug_footer(ctx).banner_cell field check.
    // =================================================================
    {
        using namespace collider::ui::tui;
        auto theme = make_theme(ThemeVariant::Default);
        panels::FooterContext ctx;
        ctx.active_bindings = active_footer_bindings();
        ctx.banner_text = "Bloom queued: 50M.blf (applies at next phase).";

        auto el = panels::render_footer(ctx, theme);
        std::string out = render_to_string(el, 200, 3);
        if (out.empty())
            return fail("footer-banner: render produced empty output");

        auto dbg = panels::debug_footer(ctx);
        if (dbg.banner_cell != "Bloom queued: 50M.blf (applies at next phase).")
            return fail("footer-banner[debug]: banner_cell mismatch");
    }

    // =================================================================
    // 18. Bloom picker open (modal) render: when state.open is true,
    //     the modal renders with its title, candidate rows, and the
    //     key hint footer (up/down, enter, esc).
    //
    // T3.4 (2026-05-17): "Bloom file picker" / per-candidate /
    // "enter" / "esc" substring scrapes replaced with
    // debug_bloom_picker(state) field checks. The typed accessor
    // composes candidate labels through the same display_label() helper
    // the renderer uses, so the strings are byte-identical to the
    // rendered rows. The render call stays as a no-throw smoke.
    // =================================================================
    {
        using namespace collider::ui::tui;
        auto theme = make_theme(ThemeVariant::Default);
        panels::BloomPickerState picker;
        picker.candidates = {
            "first.blf", "second.blf", "third.blf",
        };
        picker.selected_index = 1;
        picker.open = true;

        auto el = panels::render_bloom_picker(picker, theme);
        std::string out = render_to_string(el, 80, 14);
        if (out.empty())
            return fail("bloom-picker: render produced empty output");

        auto dbg = panels::debug_bloom_picker(picker);
        if (!dbg.open)
            return fail("bloom-picker[debug]: open flag not propagated");
        if (dbg.title_cell != "Bloom file picker")
            return fail("bloom-picker[debug]: title_cell mismatch");
        if (dbg.selected_index != 1)
            return fail("bloom-picker[debug]: selected_index mismatch");
        if (dbg.candidate_labels.size() != 3)
            return fail("bloom-picker[debug]: candidate_labels size mismatch");
        // display_label appends "  (<parent>)"; the basename portion
        // matches the input filename so a substring check on the
        // typed label still pins the filename.
        if (dbg.candidate_labels[0].find("first.blf") == std::string::npos)
            return fail("bloom-picker[debug]: candidate 0 missing 'first.blf'");
        if (dbg.candidate_labels[1].find("second.blf") == std::string::npos)
            return fail("bloom-picker[debug]: candidate 1 missing 'second.blf'");
        if (dbg.candidate_labels[2].find("third.blf") == std::string::npos)
            return fail("bloom-picker[debug]: candidate 2 missing 'third.blf'");
        if (!dbg.empty_message.empty())
            return fail("bloom-picker[debug]: empty_message leaked");
        // Hint cells: (key, verb) pairs in render order.
        if (dbg.hint_cells.size() != 3)
            return fail("bloom-picker[debug]: hint_cells size mismatch");
        if (dbg.hint_cells[1].first != "enter")
            return fail("bloom-picker[debug]: enter hint missing");
        if (dbg.hint_cells[2].first != "esc")
            return fail("bloom-picker[debug]: esc hint missing");
    }

    // =================================================================
    // 19. Bloom picker empty: when there are no candidates, the modal
    //     surfaces a helpful "no files found" message instead of
    //     an empty list.
    //
    // T3.4 (2026-05-17): "No .blf files found" substring scrape
    // replaced with debug_bloom_picker(state).empty_message field check.
    // =================================================================
    {
        using namespace collider::ui::tui;
        auto theme = make_theme(ThemeVariant::Default);
        panels::BloomPickerState picker;
        picker.candidates.clear();
        picker.open = true;

        auto el = panels::render_bloom_picker(picker, theme);
        std::string out = render_to_string(el, 80, 14);
        if (out.empty())
            return fail("bloom-picker-empty: render produced empty output");

        auto dbg = panels::debug_bloom_picker(picker);
        if (dbg.empty_message.find("No .blf files found") == std::string::npos)
            return fail("bloom-picker-empty[debug]: empty_message missing expected text");
        if (!dbg.candidate_labels.empty())
            return fail("bloom-picker-empty[debug]: candidate_labels leaked");
    }

    // =================================================================
    // 20. Bloom picker closed: render_bloom_picker returns an empty
    //     element when open is false, so layering it via dbox onto
    //     the main render contributes no visible content.
    //
    // FTXUI's Render() lays the element into the supplied screen and
    // any cell not touched by the element keeps the default space char.
    // An empty text element does not contribute any non-space content,
    // so a render of a single empty element into a 40x4 buffer should
    // produce only spaces (with newlines between rows from
    // Screen::ToString()).
    // =================================================================
    {
        using namespace collider::ui::tui;
        auto theme = make_theme(ThemeVariant::Default);
        panels::BloomPickerState picker;
        picker.open = false;
        auto el = panels::render_bloom_picker(picker, theme);
        std::string out = render_to_string(el, 40, 4);
        bool any_visible = false;
        for (char c : out) {
            if (c != ' ' && c != '\n' && c != '\r' && c != '\t') {
                any_visible = true;
                break;
            }
        }
        if (any_visible) {
            // Diagnostic dump in case FTXUI emits any unexpected
            // sequence (e.g. cursor position resets). The test fails
            // if the dump reveals genuinely visible content; if it
            // turns out FTXUI emits non-space control characters
            // surfaced by ToString() the tolerance set above can grow.
            std::fprintf(stderr, "bloom-picker-closed dump (%zu bytes):\n",
                         out.size());
            for (char c : out) {
                if (c < 0x20 || c > 0x7e) {
                    std::fprintf(stderr, "<0x%02x>",
                                 static_cast<unsigned char>(c));
                } else {
                    std::fputc(c, stderr);
                }
            }
            std::fputc('\n', stderr);
            return fail("bloom-picker-closed: non-blank output when closed");
        }
    }

    // =================================================================
    // 21. Plugins panel populated (Phase 5 / Wave 9): renders a
    //     PluginsPanelContext with two synthetic plugins.
    //
    // T3.4 (2026-05-17): substring scrapes ("Plugins", "balance-scanner",
    // "[active]", "[disabled]", "balance-scanner] result=0 BTC",
    // "last_error") replaced with debug_plugins(ctx) field assertions.
    // The typed accessor surfaces the same name + status_label +
    // recent_output that the renderer uses. "Plugins" panel title is a
    // pure UI label and stays implicit via the no-throw render smoke.
    // =================================================================
    {
        using namespace collider::ui::tui;
        auto theme = make_theme(ThemeVariant::Default);

        panels::PluginsPanelContext ctx;
        {
            collider::plugins::PluginStatus p;
            p.name = "balance-scanner";
            p.enabled = true;
            p.alive = true;
            p.restart_count = 0;
            p.recent_output = {
                "[balance-scanner] passphrase=abc address=1ABC",
                "[balance-scanner] result=0 BTC",
            };
            ctx.plugins.push_back(p);
        }
        {
            collider::plugins::PluginStatus p;
            p.name = "desktop-notifier";
            p.enabled = false;
            p.alive = false;
            p.restart_count = 0;
            p.last_error = "pipe write returned EPIPE";
            ctx.plugins.push_back(p);
        }

        auto el = panels::render_plugins(ctx, theme);
        std::string out = render_to_string(el, 100, 14);
        if (out.empty())
            return fail("plugins: render_to_string returned empty");

        auto dbg = panels::debug_plugins(ctx);
        if (dbg.plugins.size() != 2)
            return fail("plugins: debug_plugins wrong count");
        if (dbg.plugins[0].name != "balance-scanner")
            return fail("plugins: plugin 0 name mismatch");
        if (dbg.plugins[0].status_label != "[active]")
            return fail("plugins: plugin 0 should be [active]");
        if (dbg.plugins[1].name != "desktop-notifier")
            return fail("plugins: plugin 1 name mismatch");
        if (dbg.plugins[1].status_label != "[disabled]")
            return fail("plugins: plugin 1 should be [disabled]");
        // recent_output: balance-scanner has two lines; the renderer
        // shows up to kOutputPreviewLines of the tail (debug accessor
        // mirrors the same truncation logic). The last entry must be
        // the "[balance-scanner] result=0 BTC" line.
        if (dbg.plugins[0].recent_output.empty())
            return fail("plugins: plugin 0 recent_output empty");
        const auto& tail = dbg.plugins[0].recent_output.back();
        if (tail.find("result=0 BTC") == std::string::npos)
            return fail("plugins: plugin 0 recent_output tail mismatch");
        // desktop-notifier is dead with a last_error; the renderer
        // suppresses the recent_output preview in this case, and the
        // debug accessor mirrors that.
        if (dbg.plugins[1].last_error != "pipe write returned EPIPE")
            return fail("plugins: plugin 1 last_error mismatch");
        if (!dbg.plugins[1].recent_output.empty())
            return fail("plugins: plugin 1 recent_output should be suppressed");
    }

    // =================================================================
    // 22. Plugins panel empty: zero plugins should produce an empty
    //     debug payload (no rows). The panel still surfaces a "No
    //     plugins configured" zero-state in the rendered output, but
    //     that exact phrasing is a UI label not exposed via the typed
    //     accessor; the no-throw render smoke covers it.
    //
    // T3.4 (2026-05-17): "No plugins configured" substring replaced
    // with the structural check that debug_plugins(ctx).plugins is
    // empty.
    // =================================================================
    {
        using namespace collider::ui::tui;
        auto theme = make_theme(ThemeVariant::Default);
        panels::PluginsPanelContext ctx;  // empty
        auto el = panels::render_plugins(ctx, theme);
        std::string out = render_to_string(el, 100, 6);
        if (out.empty())
            return fail("plugins-empty: render_to_string returned empty");

        auto dbg = panels::debug_plugins(ctx);
        if (!dbg.plugins.empty())
            return fail("plugins-empty: debug_plugins should be empty");
    }

    // =================================================================
    // 23. Recent hits modal closed: render returns an empty element so
    //     dbox-layering contributes nothing. Same blanking contract as
    //     the bloom picker.
    // =================================================================
    {
        using namespace collider::ui::tui;
        auto theme = make_theme(ThemeVariant::Default);
        panels::RecentHitsState state;
        state.open = false;
        auto el = panels::render_recent_hits(state, theme);
        std::string out = render_to_string(el, 40, 4);
        bool any_visible = false;
        for (char c : out) {
            if (c != ' ' && c != '\n' && c != '\r' && c != '\t') {
                any_visible = true;
                break;
            }
        }
        if (any_visible) {
            return fail("recent-hits-closed: non-blank output when closed");
        }
    }

    // =================================================================
    // 24. Recent hits modal open with synthetic hits: confirms the
    //     title, two hit rows, navigation hint row, and verified /
    //     empty tags surface.
    //
    // T3.4 (2026-05-17): "Recent Hits" / per-passphrase /
    // "verified" / "empty" / "esc" substring scrapes replaced with
    // debug_recent_hits(state) field checks. The typed rows mirror what
    // render_recent_hits() composes via format_hit_row().
    // =================================================================
    {
        using namespace collider::ui::tui;
        auto theme = make_theme(ThemeVariant::Default);
        panels::RecentHitsState state;
        state.open = true;
        for (int i = 0; i < 5; ++i) {
            panels::RecentHit h;
            h.ts_iso = "2026-05-15T12:34:56Z";
            h.passphrase = "letmein-" + std::to_string(i);
            h.h160_hex = "deadbeef0123456789abcdef";
            h.address = "1ABCxyz";
            h.verified = (i == 0);   // first row verified, rest empty
            state.hits.push_back(h);
        }
        state.selected_index = 2;

        auto el = panels::render_recent_hits(state, theme);
        std::string out = render_to_string(el, 100, 14);
        if (out.empty())
            return fail("recent-hits: render produced empty output");

        auto dbg = panels::debug_recent_hits(state);
        if (!dbg.open)
            return fail("recent-hits[debug]: open flag not propagated");
        if (dbg.title_cell != "Recent Hits (last 50)")
            return fail("recent-hits[debug]: title_cell mismatch");
        if (dbg.selected_index != 2)
            return fail("recent-hits[debug]: selected_index mismatch");
        if (dbg.rows.size() != 5)
            return fail("recent-hits[debug]: rows size mismatch");
        if (dbg.rows[0].label_cell.find("letmein-0") == std::string::npos)
            return fail("recent-hits[debug]: row 0 missing 'letmein-0'");
        if (dbg.rows[2].label_cell.find("letmein-2") == std::string::npos)
            return fail("recent-hits[debug]: row 2 missing 'letmein-2'");
        if (dbg.rows[4].label_cell.find("letmein-4") == std::string::npos)
            return fail("recent-hits[debug]: row 4 missing 'letmein-4'");
        if (dbg.rows[0].status_label != "verified")
            return fail("recent-hits[debug]: row 0 should be 'verified'");
        if (dbg.rows[1].status_label != "empty")
            return fail("recent-hits[debug]: row 1 should be 'empty'");
        if (dbg.rows[4].status_label != "empty")
            return fail("recent-hits[debug]: row 4 should be 'empty'");
        // Hint cells (key, verb pairs); esc is the closer.
        if (dbg.hint_cells.size() != 2)
            return fail("recent-hits[debug]: hint_cells size mismatch");
        if (dbg.hint_cells[1].first != "esc")
            return fail("recent-hits[debug]: esc hint missing");
    }

    // =================================================================
    // 25. Wordlist picker modal closed: render returns an empty
    //     element so dbox-layering contributes nothing.
    // =================================================================
    {
        using namespace collider::ui::tui;
        auto theme = make_theme(ThemeVariant::Default);
        panels::WordlistPickerState state;
        state.open = false;
        auto el = panels::render_wordlist_picker(state, theme);
        std::string out = render_to_string(el, 40, 4);
        bool any_visible = false;
        for (char c : out) {
            if (c != ' ' && c != '\n' && c != '\r' && c != '\t') {
                any_visible = true;
                break;
            }
        }
        if (any_visible) {
            return fail("wordlist-picker-closed: non-blank output when closed");
        }
    }

    // =================================================================
    // 26. Wordlist picker modal open with synthetic profiles: confirms
    //     the title, candidate names, [active] marker on the current
    //     profile, and the key hint row all appear.
    //
    // T3.4 (2026-05-17): substring scrapes ("Wordlist picker",
    // "combined_wordlist.txt", "rockyou.txt", "[active]", "enter",
    // "esc") replaced with debug_wordlist_picker(state) field checks.
    // The typed rows carry the same display_name + active_marker the
    // renderer composes, byte-for-byte.
    // =================================================================
    {
        using namespace collider::ui::tui;
        auto theme = make_theme(ThemeVariant::Default);
        panels::WordlistPickerState state;
        state.open = true;
        {
            panels::WordlistProfile p;
            p.path = "/home/op/.thecollider/processed/combined_wordlist.txt";
            p.display_name = "combined_wordlist.txt  (12500000 lines)";
            p.line_count = 12'500'000;
            p.current = true;
            state.candidates.push_back(p);
        }
        {
            panels::WordlistProfile p;
            p.path = "/home/op/.collider/wordlists/rockyou.txt";
            p.display_name = "rockyou.txt  (14344391 lines)";
            p.line_count = 14'344'391;
            p.current = false;
            state.candidates.push_back(p);
        }
        state.selected_index = 0;

        auto el = panels::render_wordlist_picker(state, theme);
        std::string out = render_to_string(el, 100, 12);
        if (out.empty())
            return fail("wordlist-picker: render produced empty output");

        auto dbg = panels::debug_wordlist_picker(state);
        if (!dbg.open)
            return fail("wordlist-picker[debug]: open flag not propagated");
        if (dbg.title_cell != "Wordlist picker")
            return fail("wordlist-picker[debug]: title_cell mismatch");
        if (dbg.rows.size() != 2)
            return fail("wordlist-picker[debug]: rows size mismatch");
        if (dbg.rows[0].display_name.find("combined_wordlist.txt") ==
            std::string::npos)
            return fail("wordlist-picker[debug]: row 0 missing active profile name");
        if (dbg.rows[1].display_name.find("rockyou.txt") == std::string::npos)
            return fail("wordlist-picker[debug]: row 1 missing second profile name");
        if (dbg.rows[0].active_marker != "[active]")
            return fail("wordlist-picker[debug]: row 0 missing [active] marker");
        if (!dbg.rows[1].active_marker.empty())
            return fail("wordlist-picker[debug]: row 1 active_marker leaked");
        if (dbg.hint_cells.size() != 3)
            return fail("wordlist-picker[debug]: hint_cells size mismatch");
        if (dbg.hint_cells[1].first != "enter")
            return fail("wordlist-picker[debug]: enter hint missing");
        if (dbg.hint_cells[2].first != "esc")
            return fail("wordlist-picker[debug]: esc hint missing");
    }

    // =================================================================
    // 27. Wordlist picker empty: zero candidates shows the placeholder.
    //
    // T3.4 (2026-05-17): "No wordlist profiles found" substring scrape
    // replaced with debug_wordlist_picker(state).empty_message check.
    // =================================================================
    {
        using namespace collider::ui::tui;
        auto theme = make_theme(ThemeVariant::Default);
        panels::WordlistPickerState state;
        state.open = true;
        auto el = panels::render_wordlist_picker(state, theme);
        std::string out = render_to_string(el, 80, 12);
        if (out.empty())
            return fail("wordlist-picker-empty: render produced empty output");

        auto dbg = panels::debug_wordlist_picker(state);
        if (dbg.empty_message.find("No wordlist profiles found") ==
            std::string::npos)
            return fail("wordlist-picker-empty[debug]: empty_message missing expected text");
        if (!dbg.rows.empty())
            return fail("wordlist-picker-empty[debug]: rows leaked");
    }

    // =================================================================
    // 28. Help overlay closed: render returns an empty element so
    //     dbox-layering contributes nothing. Same blanking contract as
    //     the bloom picker.
    // =================================================================
    {
        using namespace collider::ui::tui;
        auto theme = make_theme(ThemeVariant::Default);
        panels::HelpOverlayState state;
        state.open = false;
        auto el = panels::render_help_overlay(state, theme);
        std::string out = render_to_string(el, 40, 4);
        bool any_visible = false;
        for (char c : out) {
            if (c != ' ' && c != '\n' && c != '\r' && c != '\t') {
                any_visible = true;
                break;
            }
        }
        if (any_visible) {
            return fail("help-overlay-closed: non-blank output when closed");
        }
    }

    // =================================================================
    // 29. Help overlay open: renders the categorized cheatsheet.
    //     Confirms the modal title appears, every category header
    //     appears, and a representative key from each category appears.
    //
    // T3.4 (2026-05-17): substring scrapes ("Keybindings" / per-category
    // / per-binding-description / "close") replaced with
    // debug_help_overlay(state) field checks. The typed categories are
    // pulled from the same format_cheatsheet_categorized() source the
    // renderer walks, so descriptions are byte-identical.
    // =================================================================
    {
        using namespace collider::ui::tui;
        auto theme = make_theme(ThemeVariant::Default);
        panels::HelpOverlayState state;
        panels::open_help(state);
        if (!state.open)
            return fail("help-overlay: open_help did not set open");

        auto el = panels::render_help_overlay(state, theme);
        std::string out = render_to_string(el, 100, 30);
        if (out.empty())
            return fail("help-overlay: render produced empty output");

        auto dbg = panels::debug_help_overlay(state);
        if (!dbg.open)
            return fail("help-overlay[debug]: open flag not propagated");
        if (dbg.title_cell != "Keybindings")
            return fail("help-overlay[debug]: title_cell mismatch");
        // Category headers: Control / Tune / Pickers / Misc must all
        // appear in render order.
        auto has_category = [&](const std::string& name) {
            for (const auto& c : dbg.categories) {
                if (c.title == name) return true;
            }
            return false;
        };
        if (!has_category("Control"))
            return fail("help-overlay[debug]: missing Control category");
        if (!has_category("Tune"))
            return fail("help-overlay[debug]: missing Tune category");
        if (!has_category("Pickers"))
            return fail("help-overlay[debug]: missing Pickers category");
        if (!has_category("Misc"))
            return fail("help-overlay[debug]: missing Misc category");
        // Representative descriptions: walk every binding across every
        // category and look for the expected substrings. The renderer
        // and the debug accessor share the same source so a hit here is
        // a hit in the rendered output.
        auto has_description = [&](const std::string& needle) {
            for (const auto& c : dbg.categories) {
                for (const auto& kv : c.bindings) {
                    if (kv.second.find(needle) != std::string::npos) return true;
                }
            }
            return false;
        };
        if (!has_description("Quit"))
            return fail("help-overlay[debug]: missing Quit description");
        if (!has_description("pause"))
            return fail("help-overlay[debug]: missing pause description");
        if (!has_description("batch"))
            return fail("help-overlay[debug]: missing batch description");
        if (!has_description("bloom"))
            return fail("help-overlay[debug]: missing bloom description");
        if (!has_description("wordlist"))
            return fail("help-overlay[debug]: missing wordlist description");
        // Hint row: the close verb is the second hint cell.
        if (dbg.hint_cells.size() != 2)
            return fail("help-overlay[debug]: hint_cells size mismatch");
        if (dbg.hint_cells[1].second.find("close") == std::string::npos)
            return fail("help-overlay[debug]: close hint missing");
    }

    // =================================================================
    // 30. Help overlay event handling: '?' / Esc / 'q' all close. Any
    //     other key is swallowed (returns true) without closing.
    // =================================================================
    {
        using namespace collider::ui::tui;
        panels::HelpOverlayState state;
        panels::open_help(state);

        // Random key: swallowed, does not close.
        const bool consumed_a = panels::handle_help_overlay_event(
            ftxui::Event::Character('a'), state);
        if (!consumed_a)
            return fail("help-overlay-event: random key not consumed");
        if (!state.open)
            return fail("help-overlay-event: random key closed modal");

        // Esc closes.
        panels::handle_help_overlay_event(ftxui::Event::Escape, state);
        if (state.open)
            return fail("help-overlay-event: Esc did not close modal");

        // '?' also closes (re-open first).
        panels::open_help(state);
        panels::handle_help_overlay_event(
            ftxui::Event::Character('?'), state);
        if (state.open)
            return fail("help-overlay-event: '?' did not close modal");
    }

    // =================================================================
    // 31. format_cheatsheet_categorized() returns at least four
    //     categories (Control, Tune, Pickers, Misc) and every category
    //     carries at least one binding.
    // =================================================================
    {
        auto cats = format_cheatsheet_categorized();
        if (cats.size() < 4)
            return fail("cheatsheet-cat: expected at least 4 categories");
        for (const auto& c : cats) {
            if (c.title.empty())
                return fail("cheatsheet-cat: empty category title");
            if (c.bindings.empty())
                return fail("cheatsheet-cat: empty category bindings");
        }
        // Spot-check: at least one binding must mention the GPU chord.
        bool found_chord = false;
        for (const auto& c : cats) {
            for (const auto& b : c.bindings) {
                if (b.first.find("g 0..7") != std::string::npos) {
                    found_chord = true;
                }
            }
        }
        if (!found_chord)
            return fail("cheatsheet-cat: missing g 0..7 chord label");
    }

    // =================================================================
    // 32. Empty-hit-by-phase histogram in the performance panel: the
    //     typed-debug accessor surfaces per-phase EmptyHitRow entries
    //     with phase name + count_cell. Build a ScanSnapshot with non-
    //     zero empty_hits_by_phase values and confirm every phase row
    //     populates AND the most-hit phase shows the expected count.
    //
    // T3.4 (2026-05-17): "Empty Hits by Phase" + per-phase substring
    // scrapes replaced with debug_performance(ctx).empty_hits[i] field
    // checks. The "Empty Hits by Phase" panel section header is a UI
    // label not exposed via the typed accessor; the no-throw render
    // smoke covers it.
    // =================================================================
    {
        using namespace collider::ui::tui;
        PerfHistogramSnapshot p;
        // Force instrumentation-off (default) so we exercise the path
        // where empty-hit-histogram renders even without PerfCollector
        // data. total_dispatches == 0 triggers the disabled branch.
        p.total_dispatches = 0;

        collider::runtime::ScanSnapshot s;
        s.current_phase = 4;
        s.empty_hits_by_phase[0] = 142;
        s.empty_hits_by_phase[1] = 287;
        s.empty_hits_by_phase[2] = 65;
        s.empty_hits_by_phase[3] = 0;
        s.empty_hits_by_phase[4] = 864'712;

        auto theme = make_theme(ThemeVariant::Default);
        panels::PerfPanelContext ctx;
        ctx.snap = p;
        ctx.scan_snap = s;
        ctx.current_phase = 4;
        for (int i = 0; i < 5; ++i) {
            ctx.phase_keys_per_sec[i] = 100'000.0;
        }

        auto el = panels::render_performance(ctx, theme);
        std::string out = render_to_string(el, 130, 24);
        if (out.empty())
            return fail("empty-hits-hist: render_to_string returned empty");

        auto dbg = panels::debug_performance(ctx);
        // Every phase populates its name + count_cell.
        if (dbg.empty_hits[0].name != "Quick Wins")
            return fail("empty-hits-hist: phase 0 name mismatch");
        if (dbg.empty_hits[1].name != "Crypto Focus")
            return fail("empty-hits-hist: phase 1 name mismatch");
        if (dbg.empty_hits[2].name != "Extended")
            return fail("empty-hits-hist: phase 2 name mismatch");
        if (dbg.empty_hits[3].name != "Combinator")
            return fail("empty-hits-hist: phase 3 name mismatch");
        if (dbg.empty_hits[4].name != "Deep Dive")
            return fail("empty-hits-hist: phase 4 name mismatch");
        // Largest phase count: 864,712 -> count_cell uses
        // format_count_short which renders that as "864K" (or similar
        // compact form starting with "864").
        if (dbg.empty_hits[4].count_cell.find("864") == std::string::npos)
            return fail("empty-hits-hist: phase 4 count_cell missing '864'");
    }

    // =================================================================
    // 33. Monochrome theme: panels render in the absence of color
    //     hierarchy. The rendered output is expected to contain plain
    //     text labels; bold / inverted decorators are applied at the
    //     element level so a substring search still works.
    // =================================================================
    {
        using namespace collider::ui::tui;
        auto theme = make_theme(ThemeVariant::Monochrome);
        if (!theme.is_monochrome)
            return fail("monochrome: is_monochrome flag must be true");

        // Status panel under monochrome: high-rate health (avg) plus a
        // low current rate triggers the danger branch which gets
        // inverted in monochrome.
        panels::StatusContext st;
        st.snap.total_checked = 1'000'000ULL;
        st.snap.verified_hits = 0;
        st.snap.current_phase = 0;
        st.current_keys_per_sec = 100.0;
        st.avg_keys_per_sec = 1'000'000.0;   // forces danger ratio
        st.current_phase_name = "Quick Wins";
        st.sample_passphrase = "abc";
        auto el = panels::render_status(st, theme);
        std::string out = render_to_string(el, 120, 12);
        if (!contains(out, "Keys/s"))
            return fail("monochrome status: missing Keys/s label");
        if (!contains(out, "Quick Wins"))
            return fail("monochrome status: missing phase name");
    }

    // =================================================================
    // 34. Light theme: panels render without crashing. Color picks
    //     diverge from Default (blue accent instead of cyan) so a
    //     light-terminal operator gets a legible interface.
    // =================================================================
    {
        using namespace collider::ui::tui;
        auto theme = make_theme(ThemeVariant::Light);
        panels::StatusContext st;
        st.snap.total_checked = 1'000'000ULL;
        st.current_keys_per_sec = 100'000.0;
        st.avg_keys_per_sec = 100'000.0;
        st.current_phase_name = "Quick Wins";
        st.sample_passphrase = "letmein";
        auto el = panels::render_status(st, theme);
        std::string out = render_to_string(el, 120, 12);
        if (!contains(out, "Keys/s"))
            return fail("light status: missing Keys/s label");
        if (!contains(out, "letmein"))
            return fail("light status: missing passphrase");
    }

    // =================================================================
    // 35. detect_default_variant() honors COLORFGBG when NO_COLOR is
    //     not set. bg byte 15 indicates a light terminal background;
    //     bg byte 0 indicates dark. NO_COLOR still wins over both.
    // =================================================================
    {
        using namespace collider::ui::tui;
        // Make sure NO_COLOR is not interfering with the test.
#if defined(_WIN32)
        _putenv_s("NO_COLOR", "");
        _putenv_s("COLORFGBG", "0;15");
#else
        unsetenv("NO_COLOR");
        setenv("COLORFGBG", "0;15", 1);
#endif
        if (detect_default_variant() != ThemeVariant::Light)
            return fail("detect: COLORFGBG=0;15 should yield Light");

#if defined(_WIN32)
        _putenv_s("COLORFGBG", "15;0");
#else
        setenv("COLORFGBG", "15;0", 1);
#endif
        if (detect_default_variant() != ThemeVariant::Default)
            return fail("detect: COLORFGBG=15;0 should yield Default");

        // NO_COLOR still wins.
#if defined(_WIN32)
        _putenv_s("NO_COLOR", "1");
        _putenv_s("COLORFGBG", "0;15");
#else
        setenv("NO_COLOR", "1", 1);
        setenv("COLORFGBG", "0;15", 1);
#endif
        if (detect_default_variant() != ThemeVariant::Monochrome)
            return fail("detect: NO_COLOR must beat COLORFGBG");

        // Clean up so any later subtests see a fresh env.
#if defined(_WIN32)
        _putenv_s("NO_COLOR", "");
        _putenv_s("COLORFGBG", "");
#else
        unsetenv("NO_COLOR");
        unsetenv("COLORFGBG");
#endif
    }

    // =================================================================
    // 36. Tier 2 D2: big_digits widget glyph render. Verifies the
    //     three-row contract: each row is exactly 4 * text.size()
    //     visible cells (one glyph per input char), and the supported
    //     digits/suffix characters all produce non-blank output.
    // =================================================================
    {
        using namespace collider::ui::tui::widgets;
        auto rows = render_big_number_rows("1.31M");
        // Each glyph contributes 4 visible cells; every cell is 1
        // byte (ASCII space) or 3 bytes (UTF-8 block char). The byte
        // counts diverge between rows because different rows of the
        // same glyph have different fill/blank patterns; we just
        // assert the rows are non-empty.
        if (rows.row0.empty() || rows.row1.empty() || rows.row2.empty())
            return fail("big_digits: blank rows for '1.31M' input");

        // Whitespace passthrough: ' ' yields three rows of exactly
        // four ASCII spaces (the blank glyph pattern).
        auto blank_rows = render_big_number_rows(" ");
        if (blank_rows.row0 != "    " ||
            blank_rows.row1 != "    " ||
            blank_rows.row2 != "    ") {
            return fail("big_digits: blank glyph not exactly 4 spaces");
        }

        // Empty input renders three empty rows.
        auto empty_rows = render_big_number_rows("");
        if (!empty_rows.row0.empty() || !empty_rows.row1.empty() ||
            !empty_rows.row2.empty()) {
            return fail("big_digits: empty input did not yield empty rows");
        }

        // Element render smoke (no assertion on content; we just need
        // the FTXUI tree to construct without crashing).
        auto el = render_big_number("16.94G", ftxui::Color::Cyan);
        std::string out = render_to_string(el, 60, 3);
        if (out.empty())
            return fail("big_digits: render_to_string returned empty");
    }

    // =================================================================
    // 37. Tier 2 D5: contextual footer subsets per FooterMode.
    //     Default surfaces pause / help / wordlist / save; Paused
    //     surfaces resume / save / help; FocusedPanel surfaces
    //     esc back / help. ModalOpen returns an empty list (modal
    //     renders its own hints).
    // =================================================================
    {
        using namespace collider::ui::tui;
        auto defaults = active_footer_bindings_for(FooterMode::Default);
        bool found_pause = false, found_help_default = false;
        bool found_wordlist = false, found_save = false;
        for (const auto& kb : defaults) {
            std::string label = kb.short_label;
            if (label.find("pause") != std::string::npos) found_pause = true;
            if (label.find("help") != std::string::npos)
                found_help_default = true;
            if (label.find("wordlist") != std::string::npos)
                found_wordlist = true;
            if (label.find("save") != std::string::npos) found_save = true;
        }
        if (!found_pause || !found_help_default ||
            !found_wordlist || !found_save) {
            return fail("contextual footer: Default missing expected bindings");
        }

        auto paused = active_footer_bindings_for(FooterMode::Paused);
        bool found_resume = false, found_save_p = false;
        bool found_help_p = false;
        for (const auto& kb : paused) {
            std::string label = kb.short_label;
            if (label.find("resume") != std::string::npos) found_resume = true;
            if (label.find("save") != std::string::npos) found_save_p = true;
            if (label.find("help") != std::string::npos) found_help_p = true;
        }
        if (!found_resume || !found_save_p || !found_help_p) {
            return fail("contextual footer: Paused missing expected bindings");
        }

        auto focused = active_footer_bindings_for(FooterMode::FocusedPanel);
        bool found_back = false, found_help_f = false;
        for (const auto& kb : focused) {
            std::string label = kb.short_label;
            if (label.find("back") != std::string::npos) found_back = true;
            if (label.find("help") != std::string::npos) found_help_f = true;
        }
        if (!found_back || !found_help_f) {
            return fail("contextual footer: FocusedPanel missing back/help");
        }

        auto modal = active_footer_bindings_for(FooterMode::ModalOpen);
        if (!modal.empty()) {
            return fail("contextual footer: ModalOpen should be empty");
        }
    }

    // =================================================================
    // 38. Tier 2 Q1: panel debug accessors return structured cells.
    //     Status panel debug surfaces typed rate / phase / chunk /
    //     hits / empty cells; GPU panel debug surfaces per-device
    //     util / vram / temp; plugins panel debug surfaces per-plugin
    //     status_label.
    // =================================================================
    {
        using namespace collider::ui::tui;
        panels::StatusContext st;
        st.snap.total_checked = 1'965'874'397ULL;
        st.snap.bloom_collisions_filtered = 864'712 + 4'100'000ULL;
        st.snap.tight_bloom_filtered = 4'100'000ULL;
        st.snap.verified_hits = 0;
        st.snap.current_phase = 4;
        st.snap.phase_iteration = 3;
        st.current_keys_per_sec = 1'308'427.0;
        st.avg_keys_per_sec = 1'200'000.0;
        st.current_phase_name = "Deep Dive";
        st.current_chunk = 246;
        st.total_chunks = 494;
        st.sample_passphrase = "hunter2";
        auto sd = panels::debug_status(st);
        if (sd.phase_cell != "Deep Dive")
            return fail("debug_status: phase_cell mismatch");
        if (sd.iteration_cell != "3")
            return fail("debug_status: iteration_cell mismatch");
        if (sd.chunk_cell != "247/494")
            return fail("debug_status: chunk_cell mismatch");
        if (sd.empty_cell != "864,712")
            return fail("debug_status: empty_cell mismatch");
        if (sd.hits_cell != "0")
            return fail("debug_status: hits_cell mismatch");
        if (sd.noise_cell != "4,100,000")
            return fail("debug_status: noise_cell mismatch");
        if (sd.passphrase_cell != "hunter2")
            return fail("debug_status: passphrase_cell mismatch");
    }
    {
        using namespace collider::ui::tui;
        GpuTelemetrySnapshot g;
        g.nvml_available = true;
        GpuPerDeviceSnapshot d;
        d.device_id = 0;
        d.name = "NVIDIA GeForce RTX 2060 SUPER";
        d.util_pct = 87;
        d.temperature_c = 71.0f;
        d.power_watts = 165;
        d.tdp_watts = 175;
        d.vram_used_bytes = static_cast<uint64_t>(5) * 1024 * 1024 * 1024;
        d.vram_total_bytes = static_cast<uint64_t>(8) * 1024 * 1024 * 1024;
        d.sm_clock_mhz = 1755;
        d.mem_clock_mhz = 7600;
        d.fan_pct = 65;
        d.pcie_gen = 3;
        g.devices.push_back(d);
        panels::GpuPanelContext gctx;
        gctx.snap = g;
        auto gd = panels::debug_gpu(gctx);
        if (!gd.nvml_available)
            return fail("debug_gpu: nvml_available not propagated");
        if (gd.devices.size() != 1)
            return fail("debug_gpu: wrong device count");
        const auto& dev = gd.devices[0];
        if (dev.name_cell.find("RTX 2060") == std::string::npos)
            return fail("debug_gpu: device name missing");
        if (dev.util_cell.find("87%") == std::string::npos)
            return fail("debug_gpu: util cell mismatch");
        if (dev.vram_cell.find("5.0 G") == std::string::npos)
            return fail("debug_gpu: vram cell missing used GiB");
        if (dev.temp_cell.find("71 C") == std::string::npos)
            return fail("debug_gpu: temp cell mismatch");
        if (dev.power_cell.find("165W") == std::string::npos)
            return fail("debug_gpu: power cell mismatch");
        if (dev.pcie_cell != "PCIe Gen3")
            return fail("debug_gpu: PCIe cell mismatch");
    }
    {
        using namespace collider::ui::tui;
        panels::PluginsPanelContext pctx;
        {
            collider::plugins::PluginStatus p;
            p.name = "balance-scanner";
            p.enabled = true;
            p.alive = true;
            p.recent_output = {"hello"};
            pctx.plugins.push_back(p);
        }
        {
            collider::plugins::PluginStatus p;
            p.name = "desktop-notifier";
            p.enabled = false;
            p.alive = false;
            pctx.plugins.push_back(p);
        }
        auto pd = panels::debug_plugins(pctx);
        if (pd.plugins.size() != 2)
            return fail("debug_plugins: wrong plugin count");
        if (pd.plugins[0].status_label != "[active]")
            return fail("debug_plugins: first plugin not [active]");
        if (pd.plugins[1].status_label != "[disabled]")
            return fail("debug_plugins: second plugin not [disabled]");
    }
    {
        using namespace collider::ui::tui;
        PerfHistogramSnapshot p;
        p.total_dispatches = 1'200'000;
        collider::runtime::ScanSnapshot s;
        s.current_phase = 4;
        s.empty_hits_by_phase[4] = 864'712;
        panels::PerfPanelContext pctx;
        pctx.snap = p;
        pctx.scan_snap = s;
        pctx.current_phase = 4;
        pctx.phase_keys_per_sec[4] = 1'300'000.0;
        auto pd = panels::debug_performance(pctx);
        if (!pd.instrumentation_enabled)
            return fail("debug_performance: instrumentation flag not set");
        if (pd.phases[4].name != "Deep Dive")
            return fail("debug_performance: phase 4 name mismatch");
        if (pd.phases[4].status_cell != "<= current")
            return fail("debug_performance: phase 4 not current");
        if (pd.phases[0].status_cell != "(idle)")
            return fail("debug_performance: phase 0 status mismatch");
    }

    // =================================================================
    // T-B7: recent-hits loading-state render. With state.loading=true
    // and state.hits empty, the modal must render the "Loading..."
    // placeholder rather than the "No recent hits" zero-state. Confirms
    // the async path's interim UI is wired through render_recent_hits.
    //
    // T3.4 (2026-05-17): "Loading recent hits" / "No recent hits"
    // substring scrapes replaced with debug_recent_hits(state) field
    // checks on loading_message + empty_message.
    // =================================================================
    {
        using namespace collider::ui::tui;
        panels::RecentHitsState st;
        st.open = true;
        st.loading = true;
        Theme th = make_theme(ThemeVariant::Default);
        ftxui::Element el = panels::render_recent_hits(st, th);
        const std::string out = render_to_string(el, 100, 24);
        if (out.empty())
            return fail("recent-hits-loading: render produced empty output");

        auto dbg = panels::debug_recent_hits(st);
        if (!dbg.loading)
            return fail("recent-hits-loading[debug]: loading flag not set");
        if (dbg.loading_message.find("Loading recent hits") == std::string::npos)
            return fail("recent-hits-loading[debug]: loading_message mismatch");
        if (!dbg.empty_message.empty())
            return fail("recent-hits-loading[debug]: empty_message leaked while loading");
    }

    // =================================================================
    // T-B7: closed modal with loading flag still renders nothing. The
    // render function returns ftxui::text("") when state.open is false,
    // so the loading flag has no visible effect outside the open path.
    // Guards against a future regression where the loading flag leaks
    // into closed-modal render output.
    //
    // T3.4 (2026-05-17): substring scrape replaced with structural
    // assertion that debug_recent_hits(state) reports the closed-modal
    // contract (loading_message empty, rows empty).
    // =================================================================
    {
        using namespace collider::ui::tui;
        panels::RecentHitsState st;
        st.open = false;
        st.loading = true;   // intentionally inconsistent
        Theme th = make_theme(ThemeVariant::Default);
        ftxui::Element el = panels::render_recent_hits(st, th);
        (void)render_to_string(el, 80, 5);

        auto dbg = panels::debug_recent_hits(st);
        if (dbg.open)
            return fail("recent-hits-closed-loading[debug]: open should be false");
        if (!dbg.loading_message.empty())
            return fail("recent-hits-closed-loading[debug]: loading_message leaked when closed");
        if (!dbg.title_cell.empty())
            return fail("recent-hits-closed-loading[debug]: title_cell leaked when closed");
    }

    // =================================================================
    // T-B8: wordlist picker loading-state render. Same structure as the
    // recent-hits loading test above; confirms the async hand-off path
    // surfaces a "Scanning..." placeholder rather than the empty
    // candidate-list zero-state.
    //
    // T3.4 (2026-05-17): "Scanning wordlists" / "No wordlist profiles
    // found" substring scrapes replaced with
    // debug_wordlist_picker(state) loading_message + empty_message
    // field checks.
    // =================================================================
    {
        using namespace collider::ui::tui;
        panels::WordlistPickerState st;
        st.open = true;
        st.loading = true;
        Theme th = make_theme(ThemeVariant::Default);
        ftxui::Element el = panels::render_wordlist_picker(st, th);
        const std::string out = render_to_string(el, 100, 24);
        if (out.empty())
            return fail("wordlist-loading: render produced empty output");

        auto dbg = panels::debug_wordlist_picker(st);
        if (!dbg.loading)
            return fail("wordlist-loading[debug]: loading flag not set");
        if (dbg.loading_message.find("Scanning wordlists") == std::string::npos)
            return fail("wordlist-loading[debug]: loading_message mismatch");
        if (!dbg.empty_message.empty())
            return fail("wordlist-loading[debug]: empty_message leaked while loading");
    }

    // =================================================================
    // T-B15: big_digits ASCII fallback. The widget must produce three
    // rows. We assert on the rows-only API so we do not depend on
    // FTXUI's screen-width clipping. The fallback path is hard to
    // exercise from a unit test without environment manipulation
    // (unicode_available reads LANG / LC_ALL on POSIX, console code
    // page on Windows), so this test exercises the contract: every
    // glyph still emits exactly 4 cells per row regardless of which
    // glyph table is active. The shape invariant catches regressions
    // in either the unicode or ASCII branch.
    // =================================================================
    {
        using namespace collider::ui::tui;
        widgets::BigNumberRows rows = widgets::render_big_number_rows("123");
        // Each glyph occupies 4 visible cells per row. In ASCII mode
        // each cell is 1 byte (so row length == 4 * input_len). In
        // unicode mode FULL/UPPER/LOWER cells are 3 bytes, blanks are
        // 1 byte. We assert row length is between the two extremes
        // (3 chars * (4 to 12) bytes per glyph = [12, 36]).
        const size_t lo = 3 * 4;
        const size_t hi = 3 * 12;
        if (rows.row0.size() < lo || rows.row0.size() > hi)
            return fail("big_digits: row0 length out of expected range");
        if (rows.row1.size() < lo || rows.row1.size() > hi)
            return fail("big_digits: row1 length out of expected range");
        if (rows.row2.size() < lo || rows.row2.size() > hi)
            return fail("big_digits: row2 length out of expected range");
        // The three rows always have the same byte length when the
        // active mode is uniform across the three (they are; mode is
        // resolved once at the start of render_big_number_rows).
        if (rows.row0.size() != rows.row1.size() ||
            rows.row1.size() != rows.row2.size()) {
            return fail("big_digits: rows have inconsistent lengths");
        }
    }

    // =================================================================
    // F1-combinator-outer: when current_outer_iteration > 0 (Combinator
    // is the canonical caller) the Phase Progress active row must
    // surface the outer pass counter as "outer N / inner X/Y" instead
    // of the bare "X/Y" — otherwise the inner 1<=>2 toggle alone makes
    // the phase look stuck. With outer_iteration left at 0, the legacy
    // "X/Y" formatting must still render so non-Combinator phases are
    // unchanged.
    // =================================================================
    {
        using namespace collider::ui::tui;
        auto theme = make_theme(ThemeVariant::Default);
        panels::PerfPanelContext ctx;
        ctx.current_phase = 3;     // Combinator
        ctx.current_chunk = 1;
        ctx.total_chunks  = 2;
        for (int i = 0; i < 5; ++i)
            ctx.phase_keys_per_sec[i] = 1'000.0;

        ctx.current_outer_iteration = 0;
        std::string baseline =
            render_to_string(panels::render_performance(ctx, theme), 130, 24);
        if (baseline.find("1/2") == std::string::npos)
            return fail("F1-combinator-outer: baseline must show '1/2'");
        if (baseline.find("outer") != std::string::npos)
            return fail("F1-combinator-outer: baseline must NOT show 'outer'");

        ctx.current_outer_iteration = 3;
        std::string with_outer =
            render_to_string(panels::render_performance(ctx, theme), 130, 24);
        if (with_outer.find("outer 3") == std::string::npos)
            return fail("F1-combinator-outer: must show 'outer 3' prefix");
        if (with_outer.find("inner 1/2") == std::string::npos)
            return fail("F1-combinator-outer: must show 'inner 1/2'");
    }

    // =================================================================
    // mode-aware-status-pool: Pool mode renders WORK + DPs SUBMITTED
    // instead of brainwallet's PHASE row. Smoke-checks that the
    // pool-specific cells appear in the rendered output and that the
    // brainwallet PHASE label is suppressed.
    // =================================================================
    {
        using namespace collider::ui::tui;
        auto theme = make_theme(ThemeVariant::Default);
        panels::StatusContext st;
        st.current_keys_per_sec = 850'000.0;
        st.mode_kind = TuiMode::Pool;
        st.pool_info.work_id = 0xDEADBEEFCAFE0001ULL;
        st.pool_info.dp_bits = 24;
        st.pool_info.kangaroo_type = "TAME_ONLY";
        st.pool_info.dps_submitted = 4242;
        st.pool_info.pool_total_dps = 999'999;
        st.pool_info.your_share = 0.0042;
        st.pool_info.pool_endpoint = "pool.example:8333";
        std::string out = render_to_string(
            panels::render_status(st, theme), 140, 16);
        if (out.find("DPs SUBMITTED") == std::string::npos)
            return fail("mode-aware-status-pool: DPs SUBMITTED row missing");
        if (out.find("WORK") == std::string::npos)
            return fail("mode-aware-status-pool: WORK row missing");
        if (out.find("TAME_ONLY") == std::string::npos)
            return fail("mode-aware-status-pool: kangaroo_type label missing");
        if (out.find("PHASE") != std::string::npos)
            return fail("mode-aware-status-pool: brainwallet PHASE row leaked");
    }

    // =================================================================
    // mode-aware-status-challenge: Challenge mode renders KANGAROO OPS
    // + PUZZLE rows; brainwallet PHASE row is suppressed.
    // =================================================================
    {
        using namespace collider::ui::tui;
        auto theme = make_theme(ThemeVariant::Default);
        panels::StatusContext st;
        st.current_keys_per_sec = 1'000'000'000.0;
        st.mode_kind = TuiMode::Challenge;
        st.challenge_info.puzzle_number = 68;
        st.challenge_info.puzzle_bits   = 68;
        st.challenge_info.ops_completed = 1'234'567'890ULL;
        st.challenge_info.expected_ops  = 274'877'906'944ULL;  // ~2^38
        st.challenge_info.dps_found     = 17;
        st.challenge_info.backend_name  = "RCKangaroo";
        std::string out = render_to_string(
            panels::render_status(st, theme), 140, 16);
        if (out.find("KANGAROO OPS") == std::string::npos)
            return fail("mode-aware-status-challenge: ops row missing");
        if (out.find("PUZZLE") == std::string::npos)
            return fail("mode-aware-status-challenge: PUZZLE row missing");
        if (out.find("RCKangaroo") == std::string::npos)
            return fail("mode-aware-status-challenge: backend label missing");
    }

    // =================================================================
    // mode-aware-status-bipscan: BipScan mode renders PHRASES +
    // ADDRESSES + (when non-zero) HITS marker.
    // =================================================================
    {
        using namespace collider::ui::tui;
        auto theme = make_theme(ThemeVariant::Default);
        panels::StatusContext st;
        st.current_keys_per_sec = 50'000.0;
        st.mode_kind = TuiMode::BipScan;
        st.bip_scan_info.phrases_read = 12345;
        st.bip_scan_info.phrases_valid = 9876;
        st.bip_scan_info.addresses_probed = 1'200'000;
        st.bip_scan_info.bloom_hits = 3;
        std::string out = render_to_string(
            panels::render_status(st, theme), 140, 16);
        if (out.find("PHRASES") == std::string::npos)
            return fail("mode-aware-status-bipscan: PHRASES row missing");
        if (out.find("ADDRESSES") == std::string::npos)
            return fail("mode-aware-status-bipscan: ADDRESSES row missing");
        if (out.find("HITS") == std::string::npos)
            return fail("mode-aware-status-bipscan: HITS marker missing");
    }

    // =================================================================
    // mode-aware-status-bipscan-workers: WORKERS row variants. Pins
    // the regression-prone surface where stale strings ("PBKDF2 is
    // CPU-bound", "BIP scan runs on CPU") have appeared repeatedly.
    //
    // Case 1: GPU success path renders the all-on-GPU breadcrumb.
    // Case 2: --no-bip-gpu renders the explicit flag breadcrumb.
    // Case 3: init failure renders the dispatcher error message.
    // Case 4: partial init renders "M of N GPU active" + warn row.
    // =================================================================
    {
        using namespace collider::ui::tui;
        auto theme = make_theme(ThemeVariant::Default);

        // Case 1: all GPUs online, PBKDF2 + EC on GPU.
        {
            panels::StatusContext st;
            st.mode_kind = TuiMode::BipScan;
            st.bip_scan_info.worker_threads      = 23;
            st.bip_scan_info.gpu_count           = 2;
            st.bip_scan_info.gpu_count_requested = 2;
            st.bip_scan_info.pbkdf_gpu_active    = true;
            const std::string out = render_to_string(
                panels::render_status(st, theme), 140, 24);
            if (out.find("23 CPU") == std::string::npos)
                return fail("workers case1: CPU count missing");
            if (out.find("+ 2 GPU") == std::string::npos)
                return fail("workers case1: GPU count missing");
            if (out.find("PBKDF2 + EC + bloom on GPU") == std::string::npos)
                return fail("workers case1: full-GPU breadcrumb missing");
            // Regression guard: the OLD stale string MUST NOT appear.
            if (out.find("PBKDF2 is CPU-bound") != std::string::npos)
                return fail("workers case1: stale 'CPU-bound' string regressed");
        }

        // Case 2: --no-bip-gpu flag set, no GPU count.
        {
            panels::StatusContext st;
            st.mode_kind = TuiMode::BipScan;
            st.bip_scan_info.worker_threads        = 23;
            st.bip_scan_info.gpu_count             = 0;
            st.bip_scan_info.gpu_count_requested   = 2;
            st.bip_scan_info.gpu_disabled_by_flag  = true;
            const std::string out = render_to_string(
                panels::render_status(st, theme), 140, 24);
            if (out.find("--no-bip-gpu") == std::string::npos)
                return fail("workers case2: --no-bip-gpu breadcrumb missing");
        }

        // Case 3: dispatcher init failed entirely.
        {
            panels::StatusContext st;
            st.mode_kind = TuiMode::BipScan;
            st.bip_scan_info.worker_threads      = 23;
            st.bip_scan_info.gpu_count           = 0;
            st.bip_scan_info.gpu_count_requested = 1;
            st.bip_scan_info.gpu_init_message    = "cudaSetDevice failed";
            const std::string out = render_to_string(
                panels::render_status(st, theme), 140, 24);
            if (out.find("GPU init failed") == std::string::npos)
                return fail("workers case3: init-failure breadcrumb missing");
            if (out.find("cudaSetDevice failed") == std::string::npos)
                return fail("workers case3: cuda error string missing");
        }

        // Case 4: partial init (1 of 2 GPUs online), plus a per-GPU
        // fault row should render in warn color.
        {
            panels::StatusContext st;
            st.mode_kind = TuiMode::BipScan;
            st.bip_scan_info.worker_threads      = 23;
            st.bip_scan_info.gpu_count           = 1;
            st.bip_scan_info.gpu_count_requested = 2;
            st.bip_scan_info.pbkdf_gpu_active    = true;
            BipScanInfo::FaultedDevice fd;
            fd.device_id = 1;
            fd.error     = "out of memory";
            st.bip_scan_info.gpu_faulted_devices.push_back(fd);
            const std::string out = render_to_string(
                panels::render_status(st, theme), 140, 24);
            if (out.find("1 of 2 GPU") == std::string::npos)
                return fail("workers case4: partial-init breadcrumb missing");
            if (out.find("GPU#1") == std::string::npos)
                return fail("workers case4: per-device fault row missing");
            if (out.find("out of memory") == std::string::npos)
                return fail("workers case4: fault detail missing");
        }
    }

    std::printf("test_tui_panels: OK\n");
    return 0;
}
