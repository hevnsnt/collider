/**
 * pool_solver.cpp - Implementation of theCollider's pool-mode runtime
 * driver.
 *
 * Extracted verbatim from src/main.cpp during the v1.4.1 A.3 refactor.
 *
 * B4 / R-B9 / R-B10 audit: the previous version called
 * backend->initialize(work) and backend->solve(cb) exactly once for the
 * lifetime of the session. That left three observable failure modes:
 *
 *   1. WORK_ASN epoch change (R-B4): if the pool reassigned a new chunk
 *      mid-run, current_work_ updated but the running backend still
 *      walked the OLD range/target. DPs were tagged with the NEW work_id
 *      yet computed against the OLD pubkey, polluting the server's DP
 *      table with garbage that can never collide.
 *   2. Reconnect supervisor giveup (R-B10): the supervisor would set
 *      supervisor_gave_up_ after 16 failed reconnects, but the cb's
 *      should_continue only watched g_shutdown, so the worker kept
 *      grinding even though the pool was unreachable.
 *   3. SIGINT herd loss (R-B9): on Ctrl+C, backend->solve() exits and
 *      the GPU buffers free in the dtor without serializing the kangaroo
 *      herd state. Hours of work gone.
 *
 * The control flow now wraps solve() in an outer loop that detects
 * work_id change, supervisor giveup, and shutdown. On work_id change the
 * loop re-initializes the backend and re-enters solve. On giveup or
 * shutdown the loop exits.
 */
#include "runtime/pool_solver.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <system_error>
#include <thread>

#include "core/kangaroo_backend.hpp"
#include "core/paths.hpp"
#include "core/session_log.hpp"  // milestone() for state_saved (kangaroo herd)
#include "core/version.hpp"
#include "pool/pool_manager.hpp"
#include "runtime/runtime_control.hpp"  // banner queue for TUI status messages
#include "ui/banner.hpp"        // ProfessionalUI
#include "ui/box_render.hpp"    // single-source-of-truth boxed UI
#include "ui/pool_progress.hpp"
#include "core/settings_sidecar.hpp"          // TR-5: persist settings
#include "ui/tui/panels/settings_panel.hpp"  // TP-1: live settings poll
#include "ui/tui/tui_launcher.hpp"  // Phase C: unified TuiApp across modes
#include "ui/tui/stdio_capture.hpp" // release_active_capture for fatal cerr

namespace collider::runtime {

namespace {

// Local helper: where on disk to drop kangaroo herd checkpoints. Lives
// next to puzzle search state and (eventually) pool DP-pending state so
// recovery flows have a single root to scan. The path convention is
// suggested by the R-B9 audit: ~/.collider/state/kangaroo_herd_<id>.kang.
// paths::state_dir() falls back to ./.collider/state when HOME/USERPROFILE
// is unset; the only failure modes are filesystem permission errors, which
// we surface to stderr and otherwise tolerate (saving the herd is
// best-effort; the loss is regrettable but not fatal).
std::string kangaroo_state_dir() {
    return collider::paths::state_dir().string();
}

std::string kangaroo_herd_path_for_work(uint64_t work_id) {
    return kangaroo_state_dir() + "/kangaroo_herd_" + std::to_string(work_id) + ".kang";
}

// Best-effort wait for the JLP sender thread to drain the in-flight DP
// queue before we disconnect. PoolManager doesn't expose queue-size
// directly (B-coordinate boundary: src/pool/* is owned by builder-pool),
// so the heuristic here watches the submitted-counter for steady state:
// when the counter stops advancing for a full wall-clock interval, the
// queue is either drained or wedged, and either way further waiting
// won't help. Bounded by max_wait so a totally stuck sender doesn't
// block shutdown forever.
void drain_in_flight_dps(::collider::pool::PoolManager& pm,
                         std::chrono::milliseconds max_wait =
                             std::chrono::seconds(5))
{
    using clock = std::chrono::steady_clock;
    const auto deadline   = clock::now() + max_wait;
    const auto idle_quota = std::chrono::milliseconds(500);

    // Drain heuristic watches the enqueue counter for forward progress;
    // pre-Q3c this used the now-deleted get_submitted_count() alias which
    // also returned enqueued_count_, so behaviour is preserved exactly.
    // (Switching to get_sent_count() would be MORE correct semantically --
    // "DPs actually on the wire" -- but would change observable drain
    // timing under load and is deferred to a future audit.)
    uint64_t last_seen = pm.get_enqueued_count();
    auto     last_change = clock::now();
    while (clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        const uint64_t now_seen = pm.get_enqueued_count();
        if (now_seen != last_seen) {
            last_seen = now_seen;
            last_change = clock::now();
            continue;
        }
        if (clock::now() - last_change >= idle_quota) {
            // Submitted count quiescent for idle_quota; assume drained.
            return;
        }
    }
}

// Best-effort kangaroo herd serialization. The IKangarooBackend surface
// gains save_herd_state(path) / load_herd_state(path) as part of task
// #49 (builder-kangaroo). Until that lands we cannot actually serialize
// the herd from the runner; emit a stderr breadcrumb so an operator who
// SIGINTs a long pool run sees that the runner ASKED to save and that
// the missing API is the gap. Once #49 ships, replace the breadcrumb
// with backend->save_herd_state(checkpoint_path).
void attempt_save_herd(::collider::kangaroo::IKangarooBackend* backend,
                       uint64_t work_id)
{
    if (!backend) return;
    const std::string checkpoint = kangaroo_herd_path_for_work(work_id);
    try {
        std::filesystem::create_directories(kangaroo_state_dir());
    } catch (const std::exception& e) {
        std::cerr << "[!] Failed to ensure kangaroo state dir: " << e.what()
                  << "\n";
        return;
    }
    // persist the kangaroo herd to disk so a later resume can
    // pick up from this position. RCKangaroo backend returns false
    // (third-party state opaque); the JLP/SOTA multi-GPU backend writes
    // a real checkpoint file. The RCKangaroo "false" return is
    // expected, not an error -- the operator sees it after every
    // single quit when they're on the only currently-shipping pool
    // backend, and the "[!]" prefix made the routine session
    // teardown look like a fault (user-reported 2026-05-25 "I keep
    // seeing this error"). Demote that path to a one-line session-
    // log milestone; only emit a stderr breadcrumb on a TRUE save
    // (the SOTA multi-GPU path) or on a hard exception.
    if (backend->save_herd_state(checkpoint)) {
        std::cerr << "[*] Saved kangaroo herd to " << checkpoint << "\n";
        std::error_code size_ec;
        const auto bytes = std::filesystem::file_size(checkpoint, size_ec);
        std::ostringstream d;
        d << "kind=kangaroo_herd"
          << " work_id=" << work_id
          << " path=" << checkpoint
          << " bytes=" << (size_ec ? 0ULL : static_cast<unsigned long long>(bytes));
        ::collider::log::milestone("state_saved", d.str());
    } else {
        // Quietly record that the active backend does not support
        // herd serialization. No stderr breadcrumb -- this is the
        // documented behavior of RCKangaroo and is not actionable
        // for the operator.
        std::ostringstream d;
        d << "kind=kangaroo_herd"
          << " work_id=" << work_id
          << " path=" << checkpoint
          << " result=backend_no_serialize";
        ::collider::log::milestone("state_save_skipped", d.str());
    }
}

}  // namespace

// Pool-mining entry point. The local PoolManager's destructor disconnects
// the TLS/TCP session automatically, so error paths can `return 1;`
// without manual cleanup; the explicit disconnect at end-of-flow exists
// only so the final "Disconnected from pool" message lands deterministically
// before the session summary.
int run_pool_mode(const Arguments& args, const GPUDetectionResult& gpu_info) {
    using namespace ::collider::pool;

    // POOL MODE banner used to cout here (top/centered/bottom box plus
    // an explanatory paragraph about asymmetric protocol v3). All of
    // it ran AFTER main.cpp installed the stdio_capture and thus only
    // ever reached the boot log, not the operator. The TUI dashboard's
    // header (mode_label="Pool") + status panel's PoolInfo overlay
    // (kangaroo_type / work_id / dp_bits) already surface the same
    // info to the operator in real time, every frame. Banner deleted.

    // Validate arguments. Audit #20 / #40: any fatal cerr must release
    // the StdioCapture first so the operator actually sees the error on
    // the real terminal rather than having it buried in the post-mortem
    // tui-boot-<ts>.log.
    if (args.pool_url.empty()) {
        ::collider::ui::tui::StdioCapture::release_active_capture();
        std::cerr << "[!] Error: Pool URL required (--pool <url>)\n";
        return 1;
    }

    if (args.pool_worker.empty()) {
        ::collider::ui::tui::StdioCapture::release_active_capture();
        std::cerr << "[!] Error: Worker name required (--worker <bitcoin_address>)\n";
        std::cerr << "    Your Bitcoin address is used for reward distribution.\n";
        return 1;
    }

    // Parse pool URL
    PoolConfig pool_config;
    if (!parse_pool_url(args.pool_url, pool_config)) {
        ::collider::ui::tui::StdioCapture::release_active_capture();
        std::cerr << "[!] Error: Invalid pool URL format\n";
        std::cerr << "    Expected: jlps://host:port (TLS) or jlp://host:port\n";
        return 1;
    }

    // Phase C: launch the unified TUI shell. Pool mode renders the
    // header (mode="Pool"), throughput, GPU panel, and a chunk counter
    // that tracks (DPs submitted) / (DPs submitted + a small buffer) so
    // the progress bar shows monotonic advance. The cout/cerr lines
    // below still execute but vanish behind FTXUI's alt-screen --
    // future v1.5.x work migrates each one to a status-line setter.
    ::collider::ui::tui::LaunchConfig pool_launch_cfg;
    pool_launch_cfg.mode_label              = "Pool";
    pool_launch_cfg.version                 = std::string(::collider::kVersion);
    pool_launch_cfg.tui_mode                = ::collider::ui::tui::TuiMode::Pool;
    pool_launch_cfg.gpu_ids                 = args.gpu_ids;
    pool_launch_cfg.session_start           = std::chrono::steady_clock::now();
    pool_launch_cfg.initial_phase_name      = "Connecting";
    pool_launch_cfg.initial_current_chunk   = 0;
    pool_launch_cfg.initial_total_chunks    = 1;
    pool_launch_cfg.render_cfg.refresh_hz   = 10;
    pool_launch_cfg.render_cfg.alt_screen   = true;
    pool_launch_cfg.guard_opts.alt_screen              = true;
    pool_launch_cfg.guard_opts.hide_cursor             = true;
    pool_launch_cfg.guard_opts.install_signal_handlers = true;
    // Restore real stdout BEFORE launch_session -- otherwise FTXUI's
    // draw stream is captured into the boot log and the alt-screen
    // stays empty / black. Use the SILENT release so the captured
    // boot text persists to ~/.collider/logs/tui-boot-<ts>.log
    // instead of scrolling through the terminal right before the
    // alt-screen takes over.
    ::collider::ui::tui::StdioCapture::release_active_capture_silent();
    auto pool_session = ::collider::ui::tui::launch_session(pool_launch_cfg);
    auto* pool_tui    = pool_session.app.get();

    pool_config.worker_name = args.pool_worker;
    // PoolConfig::password is a std::string in pool_manager.hpp (the
    // PoolManager+JLPPoolClient ABI consumes it via authenticate(const
    // std::string&)). Copy the bytes out of the upstream SecureString
    // and IMMEDIATELY wipe the Arguments-side copy. After this point:
    //   - args.pool_password: wiped (zero bytes left on the heap). The
    //     field is `mutable` so this wipe is legal through a const
    //     reference; conceptually the password is transient state, not
    //     part of the Arguments value identity.
    //   - pool_config.password: std::string in PoolConfig, lives until
    //     PoolManager is torn down. That is the residual gap; closing it
    //     fully requires migrating PoolConfig::password to SecureString
    //     and adjusting JLPPoolClient::authenticate's signature, which
    //     touches pool_manager.hpp/.cpp/jlp_pool_client.hpp owned by
    //     parallel agents this cycle. Tracked for follow-up.
    pool_config.password.assign(args.pool_password.data(),
                                args.pool_password.size());
    args.pool_password.wipe();
    pool_config.api_key = args.pool_api_key;
    pool_config.debug_mode = args.debug;
    // B1 wire-v4: forward the --worker-key path to PoolManager which
    // loads + caches the WIF identity on first connect and re-attaches
    // it to every reconnect-created JLPPoolClient.
    pool_config.worker_key_file = args.pool_worker_key_file;

    // Pool Configuration cout used to print here (5 lines: Type / Host
    // / Worker / GPUs). All captured to the log, never visible to the
    // operator. The TUI dashboard's header (mode_label="Pool") plus
    // the pool_endpoint field in PoolInfo (set below after work is
    // assigned) already surface host:port. The worker name + GPU
    // count are visible in the WORKERS row of the status panel via
    // the same mechanism every other mode uses.

    // Connect to pool. Update the phase name BEFORE the blocking
    // connect call so the operator sees "Connecting" in the dashboard.
    if (pool_tui) pool_tui->set_current_phase_name("Connecting to pool");
    auto& pool_manager = get_pool_manager();
    pool_manager.set_config(pool_config);

    if (!pool_manager.connect()) {
        if (pool_tui) {
            pool_tui->set_current_phase_name("Connection failed");
        }
        ::collider::ui::tui::StdioCapture::release_active_capture();
        std::cerr << "[!] Failed to connect to pool: " << pool_config.host
                  << ":" << pool_config.port << "\n";
        return 1;
    }

    if (pool_tui) pool_tui->set_current_phase_name("Connected; requesting work");

    // Get initial work assignment.
    WorkAssignment work;
    if (!pool_manager.get_work(work)) {
        if (pool_tui) {
            pool_tui->set_current_phase_name("Work request failed");
        }
        ::collider::ui::tui::StdioCapture::release_active_capture();
        std::cerr << "[!] Failed to get work assignment from pool\n";
        return 1;  // PoolManager dtor disconnects automatically.
    }
    // Work assignment details flow through PoolInfo (set below); the
    // duplicate cout lines were captured to the log only, so deleted.

    // Phase C: surface the assigned chunk in the TUI header / status.
    // Mode-aware overlay (pool_info) carries the work_id, dp_bits, and
    // assigned tame/wild type so the status panel renders them in the
    // operator-facing WORK row instead of stuffing the data into the
    // legacy current_phase_name string. The phase name still gets a
    // short label for the header.
    if (pool_tui) {
        pool_tui->set_current_phase_name("Mining");
        ::collider::ui::tui::PoolInfo pi;
        pi.work_id = work.work_id;
        pi.dp_bits = static_cast<int>(work.dp_bits);
        switch (work.kangaroo_type) {
            case 1: pi.kangaroo_type = "TAME_ONLY"; break;
            case 2: pi.kangaroo_type = "WILD_ONLY"; break;
            default: pi.kangaroo_type = "BOTH(illegal)"; break;
        }
        pi.pool_endpoint = pool_config.host + ":" +
                           std::to_string(pool_config.port);
        pool_tui->set_pool_info(pi);
    }

    // v1.5: SOLUTION receive callback. Fires when the pool server
    // broadcasts that it has solved the chunk; the payload byte stream
    // is treated as a STOP signal, not as a key to display, persist, or
    // upload anywhere. This is the structural enforcement of theft-
    // resistance on the worker side: the recovered key bytes are still
    // (briefly) in the broadcast payload because the server publishes
    // them for transparency, but the worker code path here neither
    // prints them nor stores them. g_shutdown latches so the outer
    // solve() loop exits and the worker disconnects cleanly. The 'key'
    // and 'worker' parameters are unused on purpose.
    pool_manager.set_solution_callback([pool_tui](const uint8_t* /*key*/,
                                                  const std::string& /*worker*/) {
        // Pre-fix this fired a boxui banner via cout, which was
        // captured to the boot log invisibly. Now the same event drives
        // the TUI directly: phase name flips to "POOL SOLVED" + a
        // banner posts to the runtime banner queue so the footer
        // surfaces it on the dashboard for the operator. The worker
        // shuts down immediately after via g_shutdown.
        if (pool_tui) {
            pool_tui->set_current_phase_name(
                "POOL SOLVED -- stopping worker");
        }
        {
            auto& rc = ::collider::runtime::global_runtime_control();
            std::lock_guard<std::mutex> lk(rc.banner_mu);
            rc.banner_text =
                "POOL SOLVED. Recovered key held on server only "
                "(v1.5 asymmetric design); worker exiting.";
            rc.banner_set_at = std::chrono::steady_clock::now();
        }
        // Latch the global shutdown so the outer solve loop exits on
        // its next should_continue / on_progress poll, draining DPs and
        // disconnecting cleanly.
        g_shutdown.store(true);
    });

    // ---- Backend dispatch ----------------------------------------------
    // The three pool branches that used to live here (CUDA RCKangaroo /
    // Apple Metal Kangaroo / CPU KangarooSolver) collapsed behind the
    // IKangarooBackend interface. v1.5.x: backend choice is runtime, not
    // compile-time. args.backend_kind (set by --backend cpu|cuda|metal,
    // empty = pick default_backend()) drives the factory; unknown labels
    // are a hard error here so the operator sees a clear message instead
    // of a silent fallback to the default that may not be what they
    // wanted.
    ::collider::kangaroo::BackendKind backend_kind =
        ::collider::kangaroo::default_backend();
    if (!args.backend_kind.empty()) {
        if (!::collider::kangaroo::parse_backend_kind(
                args.backend_kind, backend_kind)) {
            std::cerr << "[!] --backend: unknown value '"
                      << args.backend_kind
                      << "'. Expected one of: cpu, cuda, metal.\n";
            return 1;
        }
    }
    std::unique_ptr<::collider::kangaroo::IKangarooBackend> backend;
    try {
        backend = ::collider::kangaroo::create_kangaroo_backend(
            backend_kind, args.gpu_ids);
    } catch (const std::exception& e) {
        std::cerr << "[!] backend init failed: " << e.what() << "\n";
        return 1;
    }

    // Header is rendered after the first initialize() call so that
    // device_summary() reflects the actual GPU count (num_gpus_ is 0
    // until rc_.init() runs inside initialize()).

    // Wire the backend callbacks. Pool-side baseline capture for the
    // session-share % is owned by PoolProgressDisplay; the lambdas here
    // are pure adapters between IKangarooBackend's surface and the pool
    // client / progress widget.
    ::collider::ui::PoolProgressDisplay progress;

    // latched when a should_continue / on_progress poll observes
    // a new work_id from PoolManager. The backend exits its current
    // solve(), the outer loop re-initializes against the new work, and
    // we re-enter solve(). Atomic so the worker thread (RCKangaroo
    // callback) and the outer thread agree.
    std::atomic<bool> restart_pending{false};
    // Outer-loop state. The work_id tracked here is the one currently
    // being mined; comparing PoolManager::get_work() against it is the
    // epoch-change probe.
    uint64_t current_work_id = work.work_id;

    auto poll_work_id_change = [&]() {
        WorkAssignment probe;
        if (!pool_manager.get_work(probe)) return;
        if (probe.work_id != current_work_id) {
            // pool issued a new chunk. Signal the backend to exit
            // its solve() so we can re-initialize. The actual swap of
            // current_work_id happens in the outer loop after solve()
            // returns; we don't update it here because we still want
            // every in-flight DP path observation to mark this as a
            // pending restart, not as an immediate stop.
            restart_pending.store(true, std::memory_order_release);
        }
    };

    ::collider::kangaroo::BackendCallbacks cb;
    cb.on_dp = [&pool_manager](const uint8_t* x_be, const uint8_t* d_be,
                                uint8_t type, uint32_t dp_bits) {
        pool_manager.submit_dp(x_be, d_be, type, dp_bits);
    };
    cb.on_progress = [&progress, &pool_manager, &poll_work_id_change,
                      &restart_pending, pool_tui,
                      &current_work_id, &work, &pool_config](
                          double ops_per_sec,
                          uint64_t local_dps) -> bool {
        if (g_shutdown.load()) return false;
        // stop honoring solve loop when the supervisor has
        // given up. Otherwise the worker grinds against a stale
        // target until SIGINT, wasting compute AND polluting the
        // server (the JLP client drops DPs while disconnected, but
        // any in-flight DPs that survived the supervisor's first
        // disconnect went to a server that no longer has the work
        // assignment cached).
        if (pool_manager.reconnect_supervisor_gave_up()) return false;
        // every progress tick (~1-10 Hz) also checks whether
        // the pool re-issued the chunk. If so, latch restart_pending
        // and return false to exit the current solve loop cleanly.
        poll_work_id_change();
        if (restart_pending.load(std::memory_order_acquire)) return false;

        const ::collider::pool::PoolStatsLocal ps = pool_manager.get_stats();
        // User report 2026-05-23: when the TUI is active, the legacy
        // "Speed: X | Local DPs: Y | Sent: ..." single-line repaint
        // from progress.tick() still wrote to stdout. FTXUI's alt-
        // screen captured the bytes but the partial repaints leaked
        // through as visible flicker. Skip the legacy line entirely
        // when pool_tui is non-null; the unified status panel already
        // surfaces every value progress.tick used to print (DPs
        // SUBMITTED row, THROUGHPUT row, PoolInfo overlay).
        if (!pool_tui) {
            progress.tick(ops_per_sec,
                          local_dps,
                          pool_manager.get_enqueued_count(),
                          ps.total_dps,
                          ps.your_dps);
        }

        // Phase C: push live pool telemetry into the TUI panels.
        if (pool_tui) {
            pool_tui->set_keys_per_sec_current(ops_per_sec);
            // chunk progress shows "DPs submitted of (DPs + 1)" so the
            // bar always shows monotonic forward motion. Pool has no
            // hard stop count; the +1 ensures total > current.
            const uint64_t submitted = pool_manager.get_enqueued_count();
            const int cur_chunk      = static_cast<int>(submitted);
            const int tot_chunks     = static_cast<int>(submitted + 1);
            pool_tui->set_chunk_progress(cur_chunk, tot_chunks);
            // Mode-aware overlay: refresh live counters so the operator
            // sees DPs sent, pool aggregate, and their share-percentage
            // tick on every progress beat.
            ::collider::ui::tui::PoolInfo pi;
            pi.work_id        = current_work_id;
            pi.dp_bits        = static_cast<int>(work.dp_bits);
            switch (work.kangaroo_type) {
                case 1: pi.kangaroo_type = "TAME_ONLY"; break;
                case 2: pi.kangaroo_type = "WILD_ONLY"; break;
                default: pi.kangaroo_type = "BOTH(illegal)"; break;
            }
            pi.dps_submitted  = submitted;
            pi.pool_total_dps = ps.total_dps;
            pi.your_share     = ps.your_share;
            pi.pool_endpoint  = pool_config.host + ":" +
                                std::to_string(pool_config.port);
            pool_tui->set_pool_info(pi);
            // Bridge the TUI quit signal ('q' / Ctrl+C inside FTXUI) into
            // the global shutdown latch the solve loop already honours.
            if (pool_tui->requested_quit() && !g_shutdown.load()) {
                g_shutdown.store(true);
                return false;
            }
            // TP-1: settings live-apply. Poll snapshot_and_clear at the
            // 1Hz on_progress tick; apply theme + refresh-rate edits
            // immediately, latch restart_requested for backend changes
            // (the supervisor's WORK_ASN-change loop will pick it up at
            // the next solve() boundary so the new backend is built
            // against the next chunk, not torn down mid-walk).
            if (auto* st = pool_tui->settings_state()) {
                auto snap = ::collider::ui::tui::panels::snapshot_and_clear(*st);
                const bool any_change =
                    snap.dirty.num_kangaroos || snap.dirty.batch_size ||
                    snap.dirty.dp_bits || snap.dirty.refresh_hz ||
                    snap.dirty.theme || snap.dirty.verbose ||
                    snap.dirty.backend_kind || snap.dirty.solver;
                if (snap.dirty.refresh_hz) {
                    pool_tui->set_refresh_hz(snap.values.refresh_hz);
                }
                // TR-5: persist any dirty edits to
                // ~/.collider/settings.json so the next launch reads
                // them back. Best-effort: a write failure is logged
                // (via the capture ring) but does not block the solve.
                if (any_change) {
                    ::collider::settings_sidecar::save(snap.values);
                }
                if (snap.restart_requested) {
                    // Backend / solver edit needs a fresh backend init.
                    // The outer solve loop re-runs initialize() between
                    // chunks; treat this as the same signal a WORK_ASN
                    // change would set.
                    restart_pending.store(true, std::memory_order_release);
                    return false;
                }
            }
        }
        return true;
    };
    // v1.5: the cb.on_solution lambda that printed "[+] SOLUTION FOUND!"
    // + the 32-byte private key to stdout and called
    // pool_manager.report_solution(key) was DELETED. In pool mode the
    // RCKangaroo Mode is always TAME_ONLY or WILD_ONLY (enforced by
    // CudaRCKangarooBackend::initialize); the kangaroo-fork contract
    // guarantees no Mode != BOTH instance can return found=true.
    // Leaving cb.on_solution unset means BackendCallbacks::on_solution
    // is the default (empty) std::function, so if a future refactor
    // accidentally re-enables a local-solve path the call site is a
    // no-op rather than a key leak. The pool's server-to-client
    // SOLUTION broadcast is handled by pool_manager.set_solution_callback
    // above, which prints a stop-only banner and triggers g_shutdown.
    cb.should_continue = [&pool_manager, &poll_work_id_change,
                          &restart_pending, pool_tui]() {
        if (g_shutdown.load()) return false;
        // stop when the reconnect supervisor has burned through
        // its budget. Worker exits cleanly and the host loop prints a
        // diagnostic before disconnecting.
        if (pool_manager.reconnect_supervisor_gave_up()) return false;
        // cheap, lock-light epoch probe on the hot path. The
        // worker (RCKangaroo / Metal / CPU) calls this every few
        // thousand kangaroo steps, which is far more often than the
        // on_progress 1Hz tick, so a WORK_ASN that arrives between
        // two progress ticks does not let the worker run another full
        // second against the dead chunk.
        poll_work_id_change();
        if (restart_pending.load(std::memory_order_acquire)) return false;
        // Phase C: same TUI-quit bridge as on_progress, here on the hot
        // path so 'q' is observed within a few thousand kangaroo steps
        // instead of having to wait for the next on_progress tick.
        if (pool_tui && pool_tui->requested_quit() && !g_shutdown.load()) {
            g_shutdown.store(true);
            return false;
        }
        return true;
    };

    // ====================================================================
    // Outer loop: handles initial initialize() and every WORK_ASN epoch
    // change. Exits on shutdown, supervisor giveup, or backend
    // initialization failure. Each iteration is one chunk's worth of
    // solving.
    // ====================================================================
    int  exit_code            = 0;
    bool first_chunk          = true;
    bool supervisor_gave_up   = false;
    while (!g_shutdown.load()) {
        // Phase name flips on every chunk transition so the operator
        // sees "Mining work_id=N (dp=B)" in the dashboard header
        // instead of a static "Mining" forever. Previously this
        // information was cout'd post-launch -> captured -> invisible.
        if (pool_tui) {
            std::ostringstream phase;
            phase << (first_chunk ? "Mining" : "Re-mining (reassigned)")
                  << " work_id=" << current_work_id
                  << " dp=" << work.dp_bits;
            pool_tui->set_current_phase_name(phase.str());
        }

        if (!backend->initialize(work)) {
            if (pool_tui) {
                pool_tui->set_current_phase_name(
                    std::string("Backend init failed: ") + backend->name());
            }
            ::collider::ui::tui::StdioCapture::release_active_capture();
            std::cerr << "[!] " << backend->name()
                      << " initialization failed: " << backend->error() << "\n";
            exit_code = 1;
            break;
        }

        // ProfessionalUI section/kv/footer used to render here on
        // first_chunk via cout (Pool / Worker / Device / Press Ctrl+C).
        // All captured to the log. The dashboard header carries the
        // same info: mode_label="Pool" + PoolInfo.pool_endpoint;
        // backend device summary lands in the GPU panel. Block deleted.

#ifdef COLLIDER_PRO
        // Bloom filter loading is a Pro feature and is supported only by the
        // CUDA backend today. Other backends silently return false from
        // try_set_bloom_filter (interface default). Only attempt on the
        // first chunk; the backend retains the loaded filter across
        // initialize() calls on the same instance. Status flows through
        // a runtime banner so the operator sees it on the dashboard,
        // not in a captured log.
        if (first_chunk && !args.bloom_file.empty()) {
            if (backend->try_set_bloom_filter(args.bloom_file)) {
                auto& rc = ::collider::runtime::global_runtime_control();
                std::lock_guard<std::mutex> lk(rc.banner_mu);
                rc.banner_text = "Bloom filter loaded";
                rc.banner_set_at = std::chrono::steady_clock::now();
            } else if (!backend->error().empty()) {
                auto& rc = ::collider::runtime::global_runtime_control();
                std::lock_guard<std::mutex> lk(rc.banner_mu);
                rc.banner_text = std::string("WARN: bloom load failed: ") +
                                 backend->error();
                rc.banner_set_at = std::chrono::steady_clock::now();
            }
            // Backends that simply don't support bloom filtering fall through
            // silently; not an error; the user opted in but the platform
            // doesn't have the feature.
        }
#endif

        // if a checkpoint exists for this work_id, load it so the
        // backend picks up where the previous run left off. Done AFTER
        // initialize() (which seeds a fresh herd) and BEFORE solve(). The
        // backend's load_herd_state validates the file's config fingerprint
        // matches the current init; on mismatch it returns false and the
        // fresh herd from initialize() proceeds untouched.
        {
            const std::string checkpoint =
                kangaroo_herd_path_for_work(current_work_id);
            std::error_code ec;
            if (std::filesystem::exists(checkpoint, ec) && !ec) {
                if (backend->load_herd_state(checkpoint)) {
                    std::cerr << "[*] Resumed kangaroo herd from "
                              << checkpoint << "\n";
                } else {
                    std::cerr << "[!] Checkpoint at " << checkpoint
                              << " did not load (config mismatch or unsupported "
                                 "backend); starting from a fresh herd.\n";
                }
            }
        }

        // Reset epoch latch before entering solve(); the inner callbacks
        // will set it again when the pool issues the next chunk.
        restart_pending.store(false, std::memory_order_release);

        // Run the backend's solve loop. Returns when on_progress /
        // should_continue returns false (shutdown / supervisor giveup /
        // epoch change) or, in theory, when the chunk completes.
        backend->solve(cb);

        // Why did we exit?
        if (g_shutdown.load()) {
            if (pool_tui) pool_tui->set_current_phase_name("Shutting down");
            break;
        }
        if (pool_manager.reconnect_supervisor_gave_up()) {
            supervisor_gave_up = true;
            if (pool_tui) {
                pool_tui->set_current_phase_name(
                    "Pool reconnect supervisor gave up");
            }
            break;
        }
        if (!restart_pending.load(std::memory_order_acquire)) {
            // Backend exited but no restart was signaled and no shutdown
            // was requested. Likely the backend hit an internal limit
            // (e.g. expected_ops exhausted) or the connection dropped
            // long enough for is_connected() to flip false. In either
            // case, fall out of the outer loop and let the operator
            // diagnose; resuming silently would mask a real fault.
            if (pool_tui) {
                pool_tui->set_current_phase_name(
                    "Backend stopped without restart signal");
            }
            break;
        }

        // pull the new work assignment from the cache before the
        // next initialize(). PoolManager kept current_work_ updated via
        // the WORK_ASN handler; get_work() returns it without an extra
        // network round-trip.
        WorkAssignment next_work;
        if (!pool_manager.get_work(next_work)) {
            if (pool_tui) {
                pool_tui->set_current_phase_name(
                    "Failed to fetch reassigned work");
            }
            ::collider::ui::tui::StdioCapture::release_active_capture();
            std::cerr << "[!] Failed to fetch reassigned work; "
                         "exiting outer loop.\n";
            exit_code = 1;
            break;
        }
        work             = next_work;
        current_work_id  = next_work.work_id;
        first_chunk      = false;
    }

    // SIGINT save kangaroo herd. The save_herd_state API on
    // IKangarooBackend ships in builder-kangaroo's task #49; until
    // then attempt_save_herd emits a stderr breadcrumb so an operator
    // SIGINTing a long pool run sees the missing API as the gap, not
    // a silent loss. The save attempt happens BEFORE disconnect so
    // any in-flight DPs the backend is mid-emission for don't race
    // a closed socket.
    if (g_shutdown.load()) {
        attempt_save_herd(backend.get(), current_work_id);
    }

    // Supervisor giveup: phase name already updated above; this
    // sets the exit code and surfaces the actionable message via
    // the TUI banner queue (the dashboard renders banner_text in
    // the footer with a TTL). Post-launch cerr goes to the log
    // only; the banner is what the operator actually sees.
    if (supervisor_gave_up) {
        auto& rc = ::collider::runtime::global_runtime_control();
        std::lock_guard<std::mutex> lk(rc.banner_mu);
        rc.banner_text = "Pool reconnect gave up after 16 attempts; "
                         "resume later with --resume";
        rc.banner_set_at = std::chrono::steady_clock::now();
        exit_code = 1;
    }

    // best-effort drain of the JLP send queue before tearing
    // down the TLS connection. Bounded by 5s so a wedged sender
    // doesn't hang shutdown.
    drain_in_flight_dps(pool_manager);

    // Disconnect from pool. Phase name flips so the operator sees
    // the disconnect happen in the dashboard (last visible state
    // before TUI tear-down).
    if (pool_tui) pool_tui->set_current_phase_name("Disconnecting");
    pool_manager.disconnect();

    // Tear the TUI down BEFORE writing the cout session summary so
    // the summary lands in the operator's terminal scrollback, not
    // the captured boot log. Drop the session bundle so its
    // destructor runs (joins render thread, restores cooked mode,
    // flushes stdio_capture). The session-scoped capture stays
    // installed because main.cpp's interactive loop tears it down
    // at the end of each iteration -- letting it persist here
    // keeps the post-TUI summary lines below from scrolling on the
    // terminal between TUI exit and menu return.

    PoolStatsLocal stats = pool_manager.get_stats();
    // Post-mode session summary: previously written to stdout and
    // surfaced between the TUI's alt-screen exit and the menu's
    // re-entry -- which made it look like an error report. Route
    // through the session log so the same data is preserved for
    // post-mortem without flashing on the terminal (user-reported
    // 2026-05-25: this block was indistinguishable from a fault).
    {
        std::ostringstream d;
        d << "dps_submitted=" << pool_manager.get_enqueued_count()
          << " your_share_pct=" << std::fixed << std::setprecision(4)
          << (stats.your_share * 100)
          << " exit_code=" << exit_code;
        ::collider::log::milestone("pool_session_complete", d.str());
    }
    pool_session = ::collider::ui::tui::LaunchedSession{};

    return exit_code;
}

}  // namespace collider::runtime
