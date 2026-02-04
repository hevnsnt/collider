/**
 * Banner Test Tool
 *
 * Simple test program to demonstrate the animated ANSI banner.
 * Usage: ./test_banner [--no-animation] [--no-color] [--puzzle] [--benchmark]
 */

#include <iostream>
#include <string>
#include "../ui/banner.hpp"

int main(int argc, char* argv[]) {
    using namespace collider::ui;

    BannerConfig config;
    config.enable_animation = true;
    config.enable_color = true;
    config.animation_frames = 2;
    config.frame_delay_ms = 100;
    config.show_stats = true;
    config.mode = OperationMode::UNKNOWN;

    // Parse simple args
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--no-animation") {
            config.enable_animation = false;
        } else if (arg == "--no-color") {
            config.enable_color = false;
        } else if (arg == "--puzzle") {
            config.mode = OperationMode::PUZZLE_SEARCH;
        } else if (arg == "--brain") {
            config.mode = OperationMode::BRAIN_WALLET;
        } else if (arg == "--benchmark") {
            config.mode = OperationMode::BENCHMARK;
        } else if (arg == "--fast") {
            config.animation_frames = 1;
            config.frame_delay_ms = 50;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: test_banner [options]\n";
            std::cout << "  --no-animation  Disable animation\n";
            std::cout << "  --no-color      Disable colors\n";
            std::cout << "  --puzzle        Show puzzle mode banner\n";
            std::cout << "  --brain         Show brain wallet mode banner\n";
            std::cout << "  --benchmark     Show benchmark mode banner\n";
            std::cout << "  --fast          Fast animation (1 cycle, 50ms)\n";
            return 0;
        }
    }

    // Create mock stats
    BannerStats stats;
    stats.gpu_count = 1;
    stats.gpu_names = "Apple M3 Max";
    stats.backend = "Metal";
    stats.estimated_speed = 200'000'000ULL;  // 200M/s
    stats.bloom_file = "funded_addresses.blf";
    stats.bloom_entries = 47'000'000;
    stats.version = "1.0.0";

    // Puzzle-specific
    if (config.mode == OperationMode::PUZZLE_SEARCH) {
        stats.puzzle_number = 66;
        stats.puzzle_bits = 66;
        stats.puzzle_reward = 6.6;
    }

    // Display!
    display_banner(stats, config);

    std::cout << "\n[*] Banner test complete.\n";

    return 0;
}
