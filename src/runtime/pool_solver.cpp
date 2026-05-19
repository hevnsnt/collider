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
#include "pool/pool_manager.hpp"
#include "ui/banner.hpp"        // ProfessionalUI
#include "ui/box_render.hpp"    // single-source-of-truth boxed UI
#include "ui/pool_progress.hpp"

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
    // a real checkpoint file.
    if (backend->save_herd_state(checkpoint)) {
        std::cerr << "[*] Saved kangaroo herd to " << checkpoint << "\n";
        // Session log: record the herd checkpoint as a state_saved
        // milestone alongside the existing stderr breadcrumb. The
        // milestone is additive (does not replace the stderr line);
        // operators tailing collider.log see it on the console while
        // post-hoc readers (crash dump diagnostics, run-history tooling)
        // pick it out of the per-session log. Size is best-effort: a
        // file_size() failure right after a successful save indicates
        // a concurrent unlink, which is informational, not fatal.
        std::error_code size_ec;
        const auto bytes = std::filesystem::file_size(checkpoint, size_ec);
        std::ostringstream d;
        d << "kind=kangaroo_herd"
          << " work_id=" << work_id
          << " path=" << checkpoint
          << " bytes=" << (size_ec ? 0ULL : static_cast<unsigned long long>(bytes));
        ::collider::log::milestone("state_saved", d.str());
    } else {
        std::cerr << "[!] Kangaroo herd save returned false (backend may not "
                     "support serialization, e.g. RCKangaroo). Path: "
                  << checkpoint << "\n";
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

    {
        namespace boxui = ::collider::ui::box;
        std::cout << "\n";
        boxui::top(std::cout);
        boxui::centered(std::cout, "POOL MODE - Distributed Kangaroo Solving");
        boxui::bottom(std::cout);
        std::cout << "\n";
    }

    // Validate arguments
    if (args.pool_url.empty()) {
        std::cerr << "[!] Error: Pool URL required (--pool <url>)\n";
        return 1;
    }

    if (args.pool_worker.empty()) {
        std::cerr << "[!] Error: Worker name required (--worker <bitcoin_address>)\n";
        std::cerr << "    Your Bitcoin address is used for reward distribution.\n";
        return 1;
    }

    // Parse pool URL
    PoolConfig pool_config;
    if (!parse_pool_url(args.pool_url, pool_config)) {
        std::cerr << "[!] Error: Invalid pool URL format\n";
        std::cerr << "    Expected: jlps://host:port (TLS) or jlp://host:port\n";
        return 1;
    }

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

    std::cout << "[*] Pool Configuration:\n";
    std::cout << "    Type:   " << pool_config.type << "\n";
    std::cout << "    Host:   " << pool_config.host << ":" << pool_config.port << "\n";
    std::cout << "    Worker: " << pool_config.worker_name << "\n";
    std::cout << "    GPUs:   " << gpu_info.device_count << " detected\n\n";

    // Connect to pool
    std::cout << "[*] Connecting to pool...\n";
    auto& pool_manager = get_pool_manager();
    pool_manager.set_config(pool_config);

    if (!pool_manager.connect()) {
        std::cerr << "[!] Failed to connect to pool\n";
        return 1;
    }

    std::cout << "[+] Connected to pool successfully!\n\n";

    // Get initial work assignment.
    std::cout << "[*] Requesting work from pool...\n";
    WorkAssignment work;
    if (!pool_manager.get_work(work)) {
        std::cerr << "[!] Failed to get work assignment from pool\n";
        return 1;  // PoolManager dtor disconnects automatically.
    }

    std::cout << "[+] Work assigned: " << work.puzzle_name << "\n";
    std::cout << "    DP Bits: " << work.dp_bits << "\n";
    std::cout << "    Work ID: " << work.work_id << "\n\n";

    // Set solution callback
    pool_manager.set_solution_callback([](const uint8_t* key, const std::string& worker) {
        namespace boxui = ::collider::ui::box;
        std::cout << "\n";
        boxui::top(std::cout);
        boxui::centered(std::cout, "SOLUTION FOUND!", boxui::ansi::BRIGHT_GREEN);
        boxui::top(std::cout);
        boxui::kv(std::cout, "Worker", worker);
        boxui::bottom(std::cout);
        // Full 64-char hex key prints below the box; the kv() budget is
        // narrower than the key length, so emitting it outside the border
        // preserves every byte without "..." truncation.
        std::cout << "  Key: ";
        for (int i = 0; i < 32; i++) {
            std::printf("%02x", key[i]);
        }
        std::cout << "\n";
    });

    // ---- Backend dispatch ----------------------------------------------
    // The three pool branches that used to live here (CUDA RCKangaroo /
    // Apple Metal Kangaroo / CPU KangarooSolver) collapsed behind the
    // IKangarooBackend interface. The factory picks the right one for
    // this build configuration; everything past this point is
    // backend-agnostic.
    auto backend = ::collider::kangaroo::create_kangaroo_backend(args.gpu_ids);

    // Initial header (rendered once; the per-chunk header would re-render
    // every reassignment, which is too noisy for a long-running worker).
    std::cout << "\n";
    ::collider::ui::ProfessionalUI::render_section("Pool Solving - " + backend->name());
    ::collider::ui::ProfessionalUI::render_kv("Pool",
        pool_config.host + ":" + std::to_string(pool_config.port));
    ::collider::ui::ProfessionalUI::render_kv("Worker", pool_config.worker_name);
    ::collider::ui::ProfessionalUI::render_kv("Device", backend->device_summary());
    std::cout << "\n";
    ::collider::ui::ProfessionalUI::render_footer("Press Ctrl+C to stop");

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
                      &restart_pending](double ops_per_sec,
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
        progress.tick(ops_per_sec,
                      local_dps,
                      // pre-Q3c this was get_submitted_count() which aliased
                      // get_enqueued_count(); keep the same value to avoid
                      // a visible jump in the operator's DP counter.
                      pool_manager.get_enqueued_count(),
                      ps.total_dps,
                      ps.your_dps);
        return true;
    };
    cb.on_solution = [&pool_manager](const uint8_t key[32]) {
        std::cout << "\n[+] SOLUTION FOUND!\n    Private Key: ";
        for (int i = 0; i < 32; ++i) {
            std::printf("%02x", key[i]);
        }
        std::cout << "\n";
        pool_manager.report_solution(key);
    };
    cb.should_continue = [&pool_manager, &poll_work_id_change,
                          &restart_pending]() {
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
        if (first_chunk) {
            std::cout << "[*] Initializing " << backend->name()
                      << " for pool solving (work_id=" << current_work_id
                      << ", dp_bits=" << work.dp_bits << ")\n";
        } else {
            std::cout << "\n[*] Pool reassigned work (new work_id="
                      << current_work_id << ", dp_bits=" << work.dp_bits
                      << "); re-initializing backend\n";
        }

        if (!backend->initialize(work)) {
            std::cerr << "[!] " << backend->name()
                      << " initialization failed: " << backend->error() << "\n";
            exit_code = 1;
            break;
        }

#ifdef COLLIDER_PRO
        // Bloom filter loading is a Pro feature and is supported only by the
        // CUDA backend today. Other backends silently return false from
        // try_set_bloom_filter (interface default). Only attempt on the
        // first chunk; the backend retains the loaded filter across
        // initialize() calls on the same instance.
        if (first_chunk && !args.bloom_file.empty()) {
            if (backend->try_set_bloom_filter(args.bloom_file)) {
                std::cout << "[*] Bloom filter loaded - opportunistic address checking enabled\n";
            } else if (!backend->error().empty()) {
                std::cerr << "[!] WARNING: Failed to load bloom filter: "
                          << args.bloom_file << " (" << backend->error() << ")\n";
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
            std::cout << "\n[*] Shutdown requested\n";
            break;
        }
        if (pool_manager.reconnect_supervisor_gave_up()) {
            supervisor_gave_up = true;
            break;
        }
        if (!restart_pending.load(std::memory_order_acquire)) {
            // Backend exited but no restart was signaled and no shutdown
            // was requested. Likely the backend hit an internal limit
            // (e.g. expected_ops exhausted) or the connection dropped
            // long enough for is_connected() to flip false. In either
            // case, fall out of the outer loop and let the operator
            // diagnose; resuming silently would mask a real fault.
            std::cout << "\n[*] Backend solve() returned without restart "
                         "or shutdown signal; exiting outer loop.\n";
            break;
        }

        // pull the new work assignment from the cache before the
        // next initialize(). PoolManager kept current_work_ updated via
        // the WORK_ASN handler; get_work() returns it without an extra
        // network round-trip.
        WorkAssignment next_work;
        if (!pool_manager.get_work(next_work)) {
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

    // announce supervisor giveup with a clear actionable message
    // before we shut down.
    if (supervisor_gave_up) {
        std::cerr << "[!] Pool supervisor gave up after 16 reconnect "
                     "attempts; exiting. Resume later with --resume "
                     "(once builder-kangaroo task #49 lands the "
                     "kangaroo-herd persistence API).\n";
        exit_code = 1;
    }

    std::cout << "\n[*] Solving stopped\n";

    // best-effort drain of the JLP send queue before tearing
    // down the TLS connection. Bounded by 5s so a wedged sender
    // doesn't hang shutdown.
    drain_in_flight_dps(pool_manager);

    // Disconnect from pool
    pool_manager.disconnect();
    std::cout << "\n[*] Disconnected from pool\n";

    // Print final stats
    PoolStatsLocal stats = pool_manager.get_stats();
    std::cout << "\n[*] Session Summary:\n";
    // pre-Q3c this was get_submitted_count() which aliased
    // get_enqueued_count(); keep the same value so the operator-visible
    // session-summary line does not change wording or number.
    std::cout << "    DPs Submitted: " << pool_manager.get_enqueued_count() << "\n";
    std::cout << "    Your Share:    " << std::fixed << std::setprecision(4)
              << (stats.your_share * 100) << "%\n";

    return exit_code;
}

}  // namespace collider::runtime
