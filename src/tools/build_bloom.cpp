/**
 * UTXO Bloom Filter Builder CLI
 *
 * Builds a Bloom filter from utxo-dump CSV output.
 *
 * Usage:
 *   build_bloom -i utxo.csv -o filter.blf [options]
 *
 * Options:
 *   -i, --input      Input CSV file from utxo-dump
 *   -o, --output     Output .blf bloom filter file
 *   -m, --min-sats   Minimum satoshis to include (default: 100000)
 *   -f, --fp-rate    Target false positive rate (default: 0.00001)
 *   -e, --expected   Expected number of elements (default: 50000000)
 *   -v, --verify     Verify output file (optional input H160 hex file)
 *   -s, --stats      Show detailed statistics
 *   -h, --help       Show this help message
 */

#include "utxo_bloom_builder.hpp"
#include "../core/hit_verifier.hpp"
#include <iostream>
#include <chrono>
#include "getopt_compat.h"

using namespace collider;

void print_usage(const char* prog) {
    std::cout << "UTXO Bloom Filter Builder\n"
              << "Usage: " << prog << " -i <input.csv> -o <output.blf> [options]\n\n"
              << "Options:\n"
              << "  -i, --input      Input CSV file from utxo-dump\n"
              << "  -o, --output     Output .blf bloom filter file\n"
              << "  -m, --min-sats   Minimum satoshis to include (default: 100000)\n"
              << "  -f, --fp-rate    Target false positive rate (default: 0.00001)\n"
              << "  -e, --expected   Expected number of elements (default: 50000000)\n"
              << "  -v, --verify     Verification set output file (.uvrf)\n"
              << "  -s, --stats      Show detailed statistics\n"
              << "  -h, --help       Show this help message\n\n"
              << "Example:\n"
              << "  " << prog << " -i utxo-dump.csv -o addresses.blf -m 100000\n"
              << "  " << prog << " -i utxo-dump.csv -o addresses.blf -v verify.uvrf\n";
}

int main(int argc, char* argv[]) {
    std::string input_file;
    std::string output_file;
    std::string verify_file;
    uint64_t min_satoshis = 100000;
    double fp_rate = 0.00001;
    uint64_t expected_elements = 50000000;
    bool show_stats = false;

    static struct option long_options[] = {
        {"input",    required_argument, nullptr, 'i'},
        {"output",   required_argument, nullptr, 'o'},
        {"min-sats", required_argument, nullptr, 'm'},
        {"fp-rate",  required_argument, nullptr, 'f'},
        {"expected", required_argument, nullptr, 'e'},
        {"verify",   required_argument, nullptr, 'v'},
        {"stats",    no_argument,       nullptr, 's'},
        {"help",     no_argument,       nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "i:o:m:f:e:v:sh", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'i': input_file = optarg; break;
            case 'o': output_file = optarg; break;
            case 'm': min_satoshis = std::stoull(optarg); break;
            case 'f': fp_rate = std::stod(optarg); break;
            case 'e': expected_elements = std::stoull(optarg); break;
            case 'v': verify_file = optarg; break;
            case 's': show_stats = true; break;
            case 'h':
            default:
                print_usage(argv[0]);
                return opt == 'h' ? 0 : 1;
        }
    }

    if (input_file.empty() || output_file.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    try {
        std::cout << "UTXO Bloom Filter Builder\n"
                  << "========================\n\n"
                  << "Configuration:\n"
                  << "  Input:            " << input_file << "\n"
                  << "  Output:           " << output_file << "\n"
                  << "  Min Satoshis:     " << min_satoshis << " ("
                  << (min_satoshis / 100000000.0) << " BTC)\n"
                  << "  Target FP Rate:   " << fp_rate * 100 << "%\n"
                  << "  Expected Elements: " << expected_elements << "\n\n";

        // Configure builder
        utxo::UTXOBloomBuilder::Config config;
        config.target_fp_rate = fp_rate;
        config.expected_elements = expected_elements;
        config.min_satoshis = min_satoshis;

        utxo::UTXOBloomBuilder builder(config);

        std::cout << "Filter Parameters:\n"
                  << "  Bits:       " << builder.num_bits() << " ("
                  << (builder.num_bits() / 8 / 1024 / 1024) << " MB)\n"
                  << "  Hash Funcs: " << builder.num_hashes() << "\n\n";

        // Optional: build verification set
        std::unique_ptr<HitVerifier> verifier;
        if (!verify_file.empty()) {
            verifier = std::make_unique<HitVerifier>();
            std::cout << "Building verification set: " << verify_file << "\n";
        }

        // Process CSV
        std::cout << "Processing CSV...\n";
        auto start = std::chrono::steady_clock::now();

        builder.process_csv(input_file);

        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);

        std::cout << "Processed in " << duration.count() << " seconds\n\n";

        // Build verification set if requested
        if (verifier) {
            std::cout << "Building verification set...\n";
            verifier->load_from_csv(input_file, min_satoshis);
            verifier->save(verify_file);
            std::cout << "  Entries: " << verifier->size() << "\n"
                      << "  Total Value: " << (verifier->total_satoshis() / 100000000.0) << " BTC\n\n";
        }

        // Save bloom filter
        std::cout << "Saving bloom filter to " << output_file << "...\n";
        builder.save(output_file);

        // Show statistics
        auto stats = builder.get_stats();
        std::cout << "\nBloom Filter Statistics:\n"
                  << "  Elements Added:     " << stats.elements_added << "\n"
                  << "  Size:               " << stats.size_mb << " MB\n"
                  << "  Estimated FP Rate:  " << (stats.estimated_fp_rate * 100) << "%\n"
                  << "  Fill Ratio:         " << (stats.fill_ratio * 100) << "%\n";

        if (show_stats) {
            std::cout << "\nDetailed Parameters:\n"
                      << "  Bits:              " << stats.num_bits << "\n"
                      << "  Hash Functions:    " << stats.num_hashes << "\n"
                      << "  Bits per Element:  "
                      << (static_cast<double>(stats.num_bits) / stats.elements_added) << "\n";
        }

        // v1.4.2 undersize warning: the bloom was sized for `expected_elements`
        // but the CSV contained substantially more usable addresses. The
        // actual FP rate is now much higher than the target, so every scan
        // produces far more false positives than the operator planned for.
        // Recommend a re-run with the correct -e value so the FP budget is
        // honoured.
        if (stats.elements_added > config.expected_elements * 11 / 10) {
            uint64_t suggested_e = stats.elements_added + stats.elements_added / 10;
            std::cerr << "\n[!] WARNING: bloom was sized for " << config.expected_elements
                      << " elements but " << stats.elements_added << " were added.\n"
                      << "    Estimated FP rate is " << (stats.estimated_fp_rate * 100)
                      << "% (target was " << (fp_rate * 100) << "%).\n"
                      << "    Re-run with: -e " << suggested_e
                      << " for the target FP rate.\n";
        } else if (stats.elements_added * 2 < config.expected_elements) {
            // Inverse problem: bloom is way oversized. Less harmful but
            // wastes GPU memory.
            std::cerr << "\n[!] NOTE: bloom is oversized (" << stats.elements_added
                      << " elements added, expected " << config.expected_elements
                      << "). Disk + VRAM footprint is " << stats.size_mb
                      << " MB; you could rebuild with -e " << stats.elements_added
                      << " to halve the size at the same FP rate.\n";
        }

        std::cout << "\nDone! Filter saved to: " << output_file << "\n";

        if (!verify_file.empty()) {
            std::cout << "Verification set saved to: " << verify_file << "\n";
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
