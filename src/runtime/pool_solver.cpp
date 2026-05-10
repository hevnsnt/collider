/**
 * pool_solver.cpp - Implementation of theCollider's pool-mode runtime
 * driver.
 *
 * Extracted verbatim from src/main.cpp during the v1.4.1 A.3 refactor;
 * no behavior changes.
 */
#include "runtime/pool_solver.hpp"

#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string>

#include "core/kangaroo_backend.hpp"
#include "pool/pool_manager.hpp"
#include "ui/banner.hpp"        // ProfessionalUI
#include "ui/box_render.hpp"    // single-source-of-truth boxed UI
#include "ui/pool_progress.hpp"

namespace collider::runtime {

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
    pool_config.password = args.pool_password;
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

    // Get work assignment
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

    std::cout << "[*] Initializing " << backend->name() << " for pool solving...\n";
    if (!backend->initialize(work)) {
        std::cerr << "[!] " << backend->name()
                  << " initialization failed: " << backend->error() << "\n";
        return 1;
    }

#ifdef COLLIDER_PRO
    // Bloom filter loading is a Pro feature and is supported only by the
    // CUDA backend today. Other backends silently return false from
    // try_set_bloom_filter (interface default).
    if (!args.bloom_file.empty()) {
        if (backend->try_set_bloom_filter(args.bloom_file)) {
            std::cout << "[*] Bloom filter loaded - opportunistic address checking enabled\n";
        } else if (!backend->error().empty()) {
            std::cerr << "[!] WARNING: Failed to load bloom filter: "
                      << args.bloom_file << " (" << backend->error() << ")\n";
        }
        // Backends that simply don't support bloom filtering fall through
        // silently -- not an error; the user opted in but the platform
        // doesn't have the feature.
    }
#endif

    // Pool Solving header. Backend supplies its own name + device summary.
    std::cout << "\n";
    ::collider::ui::ProfessionalUI::render_section("Pool Solving - " + backend->name());
    ::collider::ui::ProfessionalUI::render_kv("Pool",
        pool_config.host + ":" + std::to_string(pool_config.port));
    ::collider::ui::ProfessionalUI::render_kv("Worker", pool_config.worker_name);
    ::collider::ui::ProfessionalUI::render_kv("Device", backend->device_summary());
    ::collider::ui::ProfessionalUI::render_kv("DP Bits", std::to_string(work.dp_bits));
    std::cout << "\n";
    ::collider::ui::ProfessionalUI::render_footer("Press Ctrl+C to stop");

    // Wire the backend callbacks. Pool-side baseline capture for the
    // session-share % is owned by PoolProgressDisplay; the lambdas here
    // are pure adapters between IKangarooBackend's surface and the pool
    // client / progress widget.
    ::collider::ui::PoolProgressDisplay progress;
    ::collider::kangaroo::BackendCallbacks cb;
    cb.on_dp = [&pool_manager](const uint8_t* x_be, const uint8_t* d_be,
                                uint8_t type, uint32_t dp_bits) {
        pool_manager.submit_dp(x_be, d_be, type, dp_bits);
    };
    cb.on_progress = [&progress, &pool_manager](double ops_per_sec,
                                                 uint64_t local_dps) -> bool {
        if (g_shutdown.load()) return false;
        const ::collider::pool::PoolStats ps = pool_manager.get_stats();
        progress.tick(ops_per_sec,
                      local_dps,
                      pool_manager.get_submitted_count(),
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
    cb.should_continue = []() { return !g_shutdown.load(); };

    backend->solve(cb);
    std::cout << "\n[*] Solving stopped\n";

    // Disconnect from pool
    pool_manager.disconnect();
    std::cout << "\n[*] Disconnected from pool\n";

    // Print final stats
    PoolStats stats = pool_manager.get_stats();
    std::cout << "\n[*] Session Summary:\n";
    std::cout << "    DPs Submitted: " << pool_manager.get_submitted_count() << "\n";
    std::cout << "    Your Share:    " << std::fixed << std::setprecision(4)
              << (stats.your_share * 100) << "%\n";

    return 0;
}

}  // namespace collider::runtime
