/**
 * puzzle_solver_helpers.cpp - Internal helpers extracted out of
 * run_puzzle_mode during the v1.4.2 structural decomposition.
 *
 * Each helper is a verbatim move of an inline block from
 * src/runtime/puzzle_solver.cpp. Behavior is preserved exactly; no
 * contracts change. See puzzle_solver_helpers.hpp for the invariants
 * checklist.
 */
#include "runtime/puzzle_solver_helpers.hpp"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#ifdef _WIN32
#include <io.h>      // _isatty
#include <stdio.h>   // _fileno
#else
#include <unistd.h>  // isatty
#endif

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "core/byte_codec.hpp"
#include "core/config.hpp"
#include "core/crypto_cpu.hpp"
#include "core/puzzle_config.hpp"
#include "core/types.hpp"
#include "gpu/puzzle_gpu.hpp"
#include "runtime/format.hpp"
#include "runtime/runtime_globals.hpp"
#include "ui/box_render.hpp"

// format_number lives in puzzle_solver.cpp at namespace-global scope and
// is consumed unqualified from this TU as well. Provide a forward decl so
// we do not pull the definition; the linker resolves it via the shared
// translation unit.
std::string format_number(uint64_t n);

namespace collider::runtime::detail {

std::vector<int> build_puzzle_worklist(const Arguments& args) {
    std::vector<int> puzzles_to_solve;
    if (args.puzzle_all_unsolved) {
        auto unsolved = PuzzleDatabase::get_unsolved();
        for (const auto* p : unsolved) {
            puzzles_to_solve.push_back(p->number);
        }
        if (puzzles_to_solve.empty()) {
            // Caller checks .empty() and prints + returns 0.
            return puzzles_to_solve;
        }
        std::cout << "\n[*] Auto-progression mode: " << puzzles_to_solve.size()
                  << " unsolved puzzles\n";
        std::cout << "    Starting with puzzle #" << puzzles_to_solve[0] << "\n";
    } else {
        puzzles_to_solve.push_back(args.puzzle_number);
        // --auto-next means "after this puzzle, keep going through
        // every higher-numbered unsolved puzzle." Pre-1.4.1 the flag was
        // parsed but never read, so it was a silent no-op.
        if (args.puzzle_auto_next) {
            auto unsolved = PuzzleDatabase::get_unsolved();
            for (const auto* p : unsolved) {
                if (p->number > args.puzzle_number) {
                    puzzles_to_solve.push_back(p->number);
                }
            }
            if (puzzles_to_solve.size() > 1) {
                std::cout << "\n[*] --auto-next: will continue through "
                          << (puzzles_to_solve.size() - 1)
                          << " higher unsolved puzzle(s) after #"
                          << args.puzzle_number << "\n";
            }
        }
    }
    return puzzles_to_solve;
}

void maybe_run_calibration(Arguments& args, UserConfig& config) {
#ifdef COLLIDER_USE_CUDA
    // Check if calibration is needed or requested
    bool need_calibration = args.calibrate || args.force_calibrate;
    if (!config.calibration_done && !need_calibration) {
        std::cout << "\n[*] First run detected - running GPU batch size calibration...\n";
        std::cout << "    (This optimizes performance for your specific hardware)\n";
        std::cout << "    (Use --force-calibrate to re-run calibration later)\n\n";
        need_calibration = true;
    }

    // Run calibration if needed
    if (need_calibration) {
        ::collider::gpu::MultiGPUPuzzleSolver calibration_solver;
        ::collider::gpu::MultiGPUPuzzleSolver::Config calib_config;
        calib_config.gpu_ids = args.gpu_ids;

        if (calibration_solver.init(calib_config)) {
            auto results = calibration_solver.calibrate_all(5);

            // Save results to config
            for (const auto& [device_id, batch_size] : results) {
                config.set_gpu_batch_size(device_id, batch_size);
            }

            // Use the calibrated batch size for this run
            if (!results.empty()) {
                args.batch_size = calibration_solver.get_batch_size();
                std::cout << "\n[*] Calibration complete. Using batch size: "
                          << (args.batch_size / 1'000'000) << "M\n";
            }

            config.save();
            std::cout << "[*] Calibration results saved to: "
                      << UserConfig::get_config_path() << "\n\n";
        } else {
            std::cerr << "[!] GPU calibration failed - using default batch size\n";
        }
    } else if (config.calibration_done) {
        // Load calibrated batch size from config
        // Use the first GPU's optimal batch size (or could average/min)
        for (int gpu_id : args.gpu_ids) {
            uint64_t optimal = config.get_gpu_batch_size(gpu_id);
            if (optimal > 0) {
                args.batch_size = optimal;
                std::cout << "[*] Using calibrated batch size: "
                          << (args.batch_size / 1'000'000) << "M (from saved config)\n";
                break;
            }
        }
    }
#else
    (void)args;
    (void)config;
#endif
}

bool resolve_puzzle_target(const Arguments& args,
                           int current_puzzle,
                           const PuzzleInfo* puzzle,
                           PuzzleTarget& out) {
    if (!args.puzzle_range_start.empty() && !args.puzzle_range_end.empty()) {
        // Custom range override
        out.range_start = UInt256(args.puzzle_range_start);
        out.range_end = UInt256(args.puzzle_range_end);
        out.bits = out.range_end.bit_length();
        out.target_address = args.puzzle_target;
        ::collider::ui::box::kv(std::cout, "Mode", "Custom Range");
    } else if (puzzle) {
        // Use known puzzle data
        out.range_start = puzzle->range_start();
        out.range_end = puzzle->range_end();
        out.bits = puzzle->bits;
        out.target_address = args.puzzle_target.empty() ? puzzle->target_address
                                                        : args.puzzle_target;

        std::ostringstream puzzle_label;
        puzzle_label << "#" << puzzle->number;
        if (puzzle->solved) {
            puzzle_label << " (SOLVED - for testing)";
        } else {
            puzzle_label << " (" << std::fixed << std::setprecision(1)
                         << puzzle->btc_reward << " BTC reward)";
        }
        ::collider::ui::box::kv(std::cout, "Puzzle", puzzle_label.str());
    } else {
        std::cerr << "[!] Error: Unknown puzzle number: " << current_puzzle << "\n";
        std::cerr << "    Use --puzzle-start and --puzzle-end for custom ranges.\n";
        return false;
    }

    // Limb decomposition mirrors the inline body.
    out.start_lo = out.range_start.parts[0];
    out.start_hi = out.range_start.parts[1];
    out.end_lo = out.range_end.parts[0];
    out.end_hi = out.range_end.parts[1];

    // Hash160 resolution
    if (puzzle && puzzle->target_h160_hex != "unknown" && puzzle->target_h160_hex.length() == 40) {
        out.h160_hex = puzzle->target_h160_hex;
        out.target_hash160 = ::collider::cpu::hex_to_hash160(out.h160_hex);
        out.have_target_hash = true;
        std::cout << "[*] Target Hash160: " << out.h160_hex << "\n";
    } else if (puzzle && !puzzle->target_address.empty()) {
        // Try to decode h160 from the Bitcoin address
        out.h160_hex = ::collider::Base58::address_to_h160_hex(puzzle->target_address);
        if (out.h160_hex.length() == 40) {
            out.target_hash160 = ::collider::cpu::hex_to_hash160(out.h160_hex);
            out.have_target_hash = true;
            std::cout << "[*] Target Hash160 (decoded from address): " << out.h160_hex << "\n";
        } else {
            std::cout << "[!] Warning: Could not decode hash160 from address: "
                      << puzzle->target_address << "\n";
            std::cout << "    Searching blind (will report any found addresses)\n";
        }
    } else {
        std::cout << "[!] Warning: Target hash160 not available for this puzzle\n";
        std::cout << "    Searching blind (will report any found addresses)\n";
    }

    // For small puzzles (< 40 bits), use sequential exhaustive search
    out.force_sequential = (out.bits <= 40);
    if (out.force_sequential && args.puzzle_random) {
        std::cout << "[*] Small puzzle detected - using sequential search for completeness\n";
    }

    return true;
}

void print_search_space_analysis(int bits) {
    uint64_t search_space_bits = static_cast<uint64_t>(bits - 1);  // 2^(N-1) keys
    double years_at_1b_per_sec = std::pow(2.0, static_cast<double>(search_space_bits))
                                 / (1e9 * 86400 * 365);

    std::cout << "[*] Search Space Analysis:\n";
    std::cout << "    Keys in range:    2^" << search_space_bits << "\n";
    if (search_space_bits <= 40) {
        uint64_t total_keys = 1ULL << search_space_bits;
        std::cout << "    Exact count:      " << format_number(total_keys) << "\n";
    }
    std::cout << "    At 1B keys/sec:   ";
    if (years_at_1b_per_sec < 1.0/365) {
        std::cout << std::fixed << std::setprecision(1)
                  << (years_at_1b_per_sec * 365 * 24) << " hours\n";
    } else if (years_at_1b_per_sec < 1.0) {
        std::cout << std::fixed << std::setprecision(1)
                  << (years_at_1b_per_sec * 365) << " days\n";
    } else if (years_at_1b_per_sec < 1000) {
        std::cout << std::fixed << std::setprecision(1) << years_at_1b_per_sec << " years\n";
    } else {
        std::cout << std::scientific << std::setprecision(2) << years_at_1b_per_sec << " years\n";
    }
    std::cout << "\n";
}

void select_algorithm(Arguments& args,
                      const PuzzleInfo* puzzle,
                      int bits,
                      bool is_multi_puzzle) {
    const bool kangaroo_was_requested = args.puzzle_kangaroo;
    const bool pubkey_known = !args.puzzle_pubkey.empty()
                              || (puzzle && !puzzle->public_key_hex.empty());

    if (args.puzzle_kangaroo && bits > 40 && !pubkey_known) {
        // User asked for kangaroo but we have no pubkey for this puzzle.
        // Three paths:
        //   1. Multi-puzzle worklist (--auto-next / --all-unsolved):
        //      silently downgrade. Stopping the whole batch on one
        //      pubkey-less puzzle would be hostile.
        //   2. Single-puzzle interactive (TTY on stdin): prompt the
        //      operator for a pubkey -- they may have one from an
        //      external source. ENTER falls back to brute force.
        //   3. Single-puzzle non-interactive (piped, CI, etc.):
        //      silently downgrade and log it.
#ifdef _WIN32
        const bool stdin_is_tty = _isatty(_fileno(stdin)) != 0;
#else
        const bool stdin_is_tty = isatty(fileno(stdin)) != 0;
#endif
        const bool prompt_allowed = !is_multi_puzzle && stdin_is_tty;

        std::cout << "\n[!] Puzzle #" << (puzzle ? puzzle->number : 0)
                  << " has no bundled public key.\n";
        std::cout << "    Kangaroo requires a 33-byte compressed pubkey (02.../03...).\n";
        std::cout << "    Most non-multiples-of-5 in puzzles 71-160 have never had a\n";
        std::cout << "    spending transaction, so the pubkey is mathematically unknown.\n";

        if (prompt_allowed) {
            std::cout << "    If you have the pubkey from an external source, paste it now\n";
            std::cout << "    (66 hex chars, e.g. 02abcdef...). Otherwise press ENTER to\n";
            std::cout << "    fall back to brute force: ";
            std::cout.flush();
            std::string entered;
            std::getline(std::cin, entered);
            // Trim whitespace
            while (!entered.empty() && std::isspace(static_cast<unsigned char>(entered.front())))
                entered.erase(entered.begin());
            while (!entered.empty() && std::isspace(static_cast<unsigned char>(entered.back())))
                entered.pop_back();
            const bool looks_like_pubkey =
                (entered.size() == 66 || entered.size() == 130) &&
                (entered[0] == '0' && (entered[1] == '2' || entered[1] == '3' || entered[1] == '4'));
            if (looks_like_pubkey) {
                args.puzzle_pubkey = entered;
                std::cout << "[*] Pubkey accepted, continuing with Kangaroo.\n";
            } else {
                if (!entered.empty()) {
                    std::cout << "[!] That doesn't look like a 33-byte compressed pubkey; "
                                 "falling back to brute force.\n";
                } else {
                    std::cout << "[*] Falling back to brute force.\n";
                }
                args.puzzle_kangaroo = false;
            }
        } else {
            std::cout << "    " << (is_multi_puzzle ? "Auto-progression mode" : "Non-interactive mode")
                      << ": silently demoting --kangaroo to brute force for this puzzle.\n";
            args.puzzle_kangaroo = false;
        }
    }

    if (!args.puzzle_kangaroo && bits > 40) {
        // Auto-pick the algorithm: kangaroo if pubkey known, else brute force.
        // This runs in two cases:
        //   (a) user didn't pass --kangaroo at all
        //   (b) user passed --kangaroo but we just demoted it above
        const bool pubkey_known_now = !args.puzzle_pubkey.empty()
                                      || (puzzle && !puzzle->public_key_hex.empty());
        if (pubkey_known_now) {
            args.puzzle_kangaroo = true;
            std::cout << "\n\033[36m[*] Algorithm Selection\033[0m\n";
            std::cout << "    Method: \033[1;32mKangaroo\033[0m (Pollard's Rho variant)\n";
            if (kangaroo_was_requested) {
                std::cout << "    Reason: Pubkey available, --kangaroo respected\n";
            } else {
                std::cout << "    Reason: Public key is known for this puzzle\n";
            }
            std::cout << "    Advantage: O(sqrt(n)) vs O(n) - dramatically faster\n";
        } else {
            std::cout << "\n\033[36m[*] Algorithm Selection\033[0m\n";
            std::cout << "    Method: \033[1;33mBrute Force\033[0m\n";
            std::cout << "    Reason: No public key available (Kangaroo requires pubkey)\n";
        }
    }
}

}  // namespace collider::runtime::detail
