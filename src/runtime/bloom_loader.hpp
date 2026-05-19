/**
 * bloom_loader.hpp - Read + validate a .blf bloom-filter file into memory.
 *
 * Extracted from brain_wallet_runner.cpp in the v1.4.2 quality wave so the
 * file-IO + header validation logic can be reused by both the primary
 * brain-wallet bloom load and the optional --bloom-tight side bloom load
 * without duplicating the 100-line validation sequence inline.
 *
 * Pro-only: the function references collider::utxo::BloomFilterHeader from
 * tools/utxo_bloom_builder.hpp, which is only on the include path when the
 * Pro target is configured. The Free build never sees a brain-wallet symbol
 * (main.cpp gates the CLI flag).
 */
#pragma once

#ifdef COLLIDER_PRO

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "core/hit_verifier.hpp"  // pulls in tools/utxo_bloom_builder.hpp
                                  // for ::collider::utxo::BloomFilterHeader.

namespace collider::runtime {

/**
 * Result of load_bloom_file_into_memory(). When ok is false, err_message
 * holds the operator-facing error string and the other fields are
 * unspecified. When ok is true, header / data are populated and
 * data_size_padded matches data.size(); the caller can hand them straight to
 * bw_pipeline.load_bloom_filter().
 */
struct BloomLoadResult {
    bool ok = false;
    std::string err_message;
    ::collider::utxo::BloomFilterHeader header{};
    std::vector<uint8_t> data;
    std::size_t data_size_padded = 0;
};

/**
 * Read + validate a .blf file into memory. Performs every check the inline
 * load used to do: stream-open, header read + size check, BLF1 magic,
 * reserved-bytes warning (non-fatal; printed to stderr), degenerate header
 * refusal (num_hashes / num_bits both > 0), seekg validation, padded read
 * with short-read rejection, and bad-bit check on the stream. Errors are
 * reported via the returned BloomLoadResult::err_message; the caller
 * prints with the same "[!] ERROR:" prefix the inline code used (one
 * prefix at the call site instead of nine here).
 */
BloomLoadResult load_bloom_file_into_memory(const std::string& path);

}  // namespace collider::runtime

#endif  // COLLIDER_PRO
