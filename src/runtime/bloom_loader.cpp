/**
 * bloom_loader.cpp - Implementation of the .blf bloom-filter loader.
 *
 * See bloom_loader.hpp for the rationale of the extraction; the function
 * body is byte-identical to the prior inline definition in
 * brain_wallet_runner.cpp.
 */
#ifdef COLLIDER_PRO

#include "runtime/bloom_loader.hpp"

#include <fstream>
#include <iostream>
#include <ios>
#include <string>

namespace collider::runtime {

BloomLoadResult load_bloom_file_into_memory(const std::string& path)
{
    BloomLoadResult out;
    std::ifstream bloom_in(path, std::ios::binary);
    if (!bloom_in) {
        out.err_message = "Cannot open bloom filter: " + path;
        return out;
    }

    // Use the single source-of-truth struct from utxo_bloom_builder.hpp.
    // The pre-existing local duplicate was a hand-aligned mirror that drifted
    // from the builder once already.
    bloom_in.read(reinterpret_cast<char*>(&out.header), sizeof(out.header));
    if (!bloom_in.good() || bloom_in.gcount() != sizeof(out.header)) {
        out.err_message = "Truncated bloom filter header (read " +
                          std::to_string(bloom_in.gcount()) + " of " +
                          std::to_string(sizeof(out.header)) + " bytes)";
        return out;
    }
    if (std::string(out.header.magic, 4) != "BLF1") {
        out.err_message = "Invalid bloom filter format (expected BLF1 header)";
        return out;
    }

    // Reserved bytes are required to be zero in the BLF1 format. A future
    // builder may use them for new metadata (e.g. a bloom-version tag or a
    // separate seed for a P2TR-specific sub-bloom). If we see non-zero
    // reserved bytes today, the file likely came from a newer builder than
    // this binary knows about; warn (don't reject) so the user can choose
    // to rebuild or upgrade theCollider. v1.5 will gate this on the version
    // field. The warning prints inline because err_message is the failure
    // channel only; success-path warnings go straight to stderr.
    {
        bool reserved_nonzero = false;
        for (size_t i = 0; i < sizeof(out.header.reserved); i++) {
            if (out.header.reserved[i] != 0) { reserved_nonzero = true; break; }
        }
        if (reserved_nonzero) {
            std::cerr << "[*] WARNING: bloom header reserved bytes are non-zero. "
                      << "This .blf may have been built by a newer build_bloom "
                      << "with format extensions this binary does not understand. "
                      << "Scan will proceed using BLF1 semantics; upgrade if "
                      << "results look wrong.\n";
        }
    }

    // Refuse pathological header values that the kernel would interpret as
    // "everything matches" (num_hashes == 0 means the for-loop runs zero
    // iterations and returns true on any input).
    if (out.header.num_hashes == 0 || out.header.num_bits == 0) {
        out.err_message = "bloom filter header has num_hashes=" +
                          std::to_string(out.header.num_hashes) +
                          " num_bits=" + std::to_string(out.header.num_bits) +
                          "; refusing (would cause every passphrase to match).";
        return out;
    }

    // The builder writes a 128-byte-aligned bit buffer (see
    // utxo_bloom_builder.hpp::allocate_filter + save), but `data_size` was
    // computed as `(num_bits + 7) / 8` which can underrun the file by up to
    // 127 bytes. Reading only the unpadded size left the GPU tail
    // uninitialised; bloom probes that mod into those bit indices returned
    // undefined results. Read the full padded extent and let the kernel
    // ignore the trailing bits (the bit_position MOD num_bits constraint
    // ensures probe never reads past num_bits).
    const size_t data_size_bits = (out.header.num_bits + 7) / 8;
    out.data_size_padded = ((data_size_bits + 127) / 128) * 128;
    out.data.assign(out.data_size_padded, 0u);

    // A malformed data_offset (past EOF, negative, garbage) would silently
    // leave the stream in a fail state. seekg failure makes the subsequent
    // read return 0 bytes from the wrong region.
    bloom_in.seekg(out.header.data_offset);
    if (!bloom_in.good()) {
        out.err_message = "bloom filter header data_offset=" +
                          std::to_string(out.header.data_offset) +
                          " is invalid (seek failed)";
        return out;
    }
    bloom_in.read(reinterpret_cast<char*>(out.data.data()), out.data_size_padded);
    const std::streamsize bytes_read = bloom_in.gcount();
    // The prior check accepted bytes_read >= data_size_bits. That accepts
    // truncation up to 127 bytes inside the padded extent. If the file is
    // genuinely truncated we want to reject loudly: a partial bloom load
    // produces a silent zero-hit scan on probes that land in the truncated
    // region. Require we read EITHER the full unpadded data extent (in case
    // the builder wrote unpadded) OR the full padded extent (modern builder).
    // gcount short of data_size_bits is always wrong.
    if (bytes_read < static_cast<std::streamsize>(data_size_bits)) {
        out.err_message = "Truncated bloom filter data (read " +
                          std::to_string(bytes_read) + " of " +
                          std::to_string(data_size_bits) + " required bytes; "
                          "padded extent was " +
                          std::to_string(out.data_size_padded) + ")."
                          " Rebuild the .blf via `build_bloom` or restore "
                          "from backup.";
        return out;
    }
    if (bloom_in.bad()) {
        out.err_message = "I/O error reading bloom filter (rare but possible "
                          "on flaky storage; rebuild or restore the .blf)";
        return out;
    }
    out.ok = true;
    return out;
}

}  // namespace collider::runtime

#endif  // COLLIDER_PRO
