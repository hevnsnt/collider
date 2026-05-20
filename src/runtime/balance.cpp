#include "runtime/balance.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "ui/banner.hpp"      // collider::ui::ansi color codes
#include "ui/box_render.hpp"  // collider::ui::box helpers

namespace collider::runtime {

namespace {

// Process-local owner for the otherwise-detached mempool.space probes.
// Pre-1.4.2 each check_balance_async() call spawned a std::thread and
// immediately detached it, capturing snapshots by value (so no UAF of the
// caller's state) but writing to std::cout after the runtime might have
// begun shutdown. The TUI render thread also writes to std::cout, so the
// detached probe was racy on the terminal as well as the process lifetime.
//
// BalanceFetcher owns every spawned probe as a std::thread. The probes
// remain "fire-and-forget" from the caller's perspective (the public API
// is unchanged) but on process exit the static destructor requests stop
// on every in-flight probe and joins them. Successful prints are gated
// on an atomic stop flag so a cancelled probe stays silent.
//
// std::jthread would be more ergonomic here, but Apple's libc++ ships
// without it through at least Xcode 16.4, so this file uses std::thread
// + atomic stop flag and joins explicitly on shutdown.
//
// Reaping: each new spawn walks the in-flight vector and drops slots
// whose lambda has set its per-slot done flag to true. A previous
// implementation reaped "the first N entries" where N was the delta of a
// global done counter; that erased the WRONG slots when probes finished
// out of order (a fast probe completing before a slow one would cause
// the reaper to call erase() on the slow slot, whose ~thread blocks on
// join() until that slow probe's HTTP call returns, freezing the spawn
// caller for up to the curl timeout). The per-slot flag fixes this so
// the reaper only erases slots that are actually safe to join now.
class BalanceFetcher {
public:
    static BalanceFetcher& instance() {
        // Meyers singleton. The destructor runs at process exit BEFORE
        // global destructors of the standard streams (function-local
        // statics are destroyed in reverse construction order, and the
        // streams are constructed by <iostream>'s ios_base::Init which
        // runs at translation-unit init time before any local static).
        // That ordering means our final print attempts after stop are
        // safe; we just don't print anything once stop is requested.
        static BalanceFetcher inst;
        return inst;
    }

    void spawn(std::string address, std::string passphrase) {
        std::lock_guard<std::mutex> lk(mu_);
        prune_finished_locked();
        // Construct the slot in place so the per-slot done and stop flags
        // have stable addresses before the worker thread starts. Slot is
        // held by std::unique_ptr (the vector relocates the unique_ptr,
        // not the Slot itself, so the flag addresses never move).
        auto slot = std::make_unique<Slot>();
        std::atomic<bool>* done_flag = &slot->done;
        std::atomic<bool>* stop_flag = &slot->stop_requested;
        slot->thread = std::thread(
            [done_flag, stop_flag, this,
             address = std::move(address),
             passphrase = std::move(passphrase)]() {
                run_probe(*stop_flag, address, passphrase);
                // Mark this slot finished. Release ordering pairs with
                // the acquire load in prune_finished_locked so the
                // reaper sees a true here only after the probe body and
                // its destructors have completed. Setting the flag is
                // the LAST thing the lambda does so the reaper never
                // races a live probe.
                done_flag->store(true, std::memory_order_release);
            });
        slots_.push_back(std::move(slot));
    }

    ~BalanceFetcher() {
        // Request stop on every slot first, then join. With the
        // libcurl/curl-cli backend the HTTP call is bounded by its own
        // CONNECT_TIMEOUT (curl default 300s) and we cannot interrupt
        // popen() mid-read. In practice mempool.space responds in <1s
        // so the join is fast; on shutdown we accept a short wait
        // rather than leaking a detached process state.
        std::lock_guard<std::mutex> lk(mu_);
        for (auto& s : slots_) {
            s->stop_requested.store(true, std::memory_order_release);
        }
        for (auto& s : slots_) {
            if (s->thread.joinable()) s->thread.join();
        }
        slots_.clear();
    }

private:
    BalanceFetcher() = default;
    BalanceFetcher(const BalanceFetcher&) = delete;
    BalanceFetcher& operator=(const BalanceFetcher&) = delete;

    // Owning record for one in-flight probe. Both flags live in a
    // separately-allocated Slot so their addresses are stable across
    // vector reallocation and the worker lambda's stored pointers stay
    // valid.
    struct Slot {
        std::atomic<bool> done{false};
        std::atomic<bool> stop_requested{false};
        std::thread       thread;
    };

    // Drop slots whose lambda has set done=true. Caller holds mu_.
    // Walks the whole vector and only erases entries that have actually
    // finished, so out-of-order completion (fast probe behind a slow
    // probe in the vector) does NOT cause the reaper to block joining
    // a still-running slot. Before erasing we join the finished thread
    // so the OS-level handle is released cleanly (std::thread does NOT
    // auto-join on destruction).
    void prune_finished_locked() {
        for (auto it = slots_.begin(); it != slots_.end();) {
            if ((*it)->done.load(std::memory_order_acquire)) {
                if ((*it)->thread.joinable()) (*it)->thread.join();
                it = slots_.erase(it);
            } else {
                ++it;
            }
        }
    }

    // The actual probe body. Pulled out of the lambda so the spawn site
    // stays compact and so the stop-flag gating is visible.
    void run_probe(std::atomic<bool>& stop_requested,
                   const std::string& address,
                   const std::string& passphrase) {
        using namespace collider::ui::ansi;
        auto stop = [&] { return stop_requested.load(std::memory_order_acquire); };
        if (stop()) return;
        try {
            std::string cmd;
#ifdef _WIN32
            cmd = "curl -s --max-time 10 \"https://mempool.space/api/address/"
                  + address + "\" 2>nul";
#else
            cmd = "curl -s --max-time 10 \"https://mempool.space/api/address/"
                  + address + "\" 2>/dev/null";
#endif
            std::array<char, 4096> buffer;
            std::string result;

#ifdef _WIN32
            FILE* pipe = _popen(cmd.c_str(), "r");
#else
            FILE* pipe = popen(cmd.c_str(), "r");
#endif
            if (!pipe) return;

            while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
                if (stop()) break;
                result += buffer.data();
            }

#ifdef _WIN32
            _pclose(pipe);
#else
            pclose(pipe);
#endif

            // Gate the post-HTTP printing: if stop was requested while
            // we were blocked in curl, we drop the result rather than
            // racing the TUI render thread on std::cout during shutdown.
            if (stop()) return;

            // Parse JSON for chain_stats.funded_txo_sum / spent_txo_sum.
            int64_t funded = 0;
            int64_t spent = 0;

            size_t pos = result.find("\"funded_txo_sum\":");
            if (pos != std::string::npos) {
                pos += 17;
                funded = std::stoll(result.substr(pos));
            }

            pos = result.find("\"spent_txo_sum\":");
            if (pos != std::string::npos) {
                pos += 16;
                spent = std::stoll(result.substr(pos));
            }

            const int64_t balance_sats = funded - spent;
            const double balance_btc = balance_sats / 100000000.0;

            if (stop()) return;

            std::cout << "\n";
            if (balance_sats > 0) {
                namespace boxui = ::collider::ui::box;
                boxui::top(std::cout);
                boxui::centered(std::cout, "*** VERIFIED HIT - ADDRESS HAS BALANCE! ***",
                                boxui::ansi::BRIGHT_GREEN);
                boxui::top(std::cout);
                boxui::kv(std::cout, "Address",    address,    {}, boxui::ansi::BRIGHT_CYAN);
                boxui::kv(std::cout, "Passphrase", passphrase, {}, boxui::ansi::BRIGHT_WHITE);
                {
                    std::ostringstream bal;
                    bal << std::fixed << std::setprecision(8) << balance_btc << " BTC";
                    boxui::kv(std::cout, "Balance", bal.str(), {}, boxui::ansi::BRIGHT_GREEN);
                }
                {
                    std::ostringstream sat;
                    sat << balance_sats;
                    boxui::kv(std::cout, "Satoshis", sat.str(), {}, boxui::ansi::BRIGHT_WHITE);
                }
                boxui::bottom(std::cout);
            } else {
                std::cout << CYAN << "[*] " << RESET << "Balance check: " << DIM << address << RESET << " = "
                          << BRIGHT_RED << std::fixed << std::setprecision(8) << balance_btc
                          << " BTC" << RESET << DIM << " (false positive)" << RESET << "\n";
            }

        } catch (const std::exception& e) {
            if (stop()) return;
            std::cout << BRIGHT_RED << "[!] " << RESET << "Balance check failed for " << address << ": " << DIM << e.what() << RESET << "\n";
        }
    }

    std::mutex mu_;
    // unique_ptr keeps each Slot's done-flag address stable across vector
    // reallocation (which the lambda captures by pointer). slots_ itself
    // is guarded by mu_; the per-slot done flag is atomic so the worker
    // lambda can publish completion without holding the lock.
    std::vector<std::unique_ptr<Slot>> slots_;
};

}  // namespace

void check_balance_async(const std::string& address,
                         const std::string& passphrase) {
    // Public API preserved: callers in brain_wallet_runner.cpp and
    // puzzle_solver.cpp still invoke this as a fire-and-forget probe.
    // The thread itself is owned by the process-local BalanceFetcher
    // singleton so process shutdown joins instead of leaking.
    BalanceFetcher::instance().spawn(address, passphrase);
}

}  // namespace collider::runtime
