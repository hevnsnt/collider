/**
 * test_gpu_multi_addr -- Oracle test for the V2 multi_addr path of the
 * brain-wallet fused kernel.
 *
 * Targets the cudaErrorIllegalAddress fault that reproduces only on
 * --brainwallet-v2 --resume after ~40 minutes of scanning the user's
 * production bloom (num_elements=2.7B, k=10). The existing
 * test_phase_change_fault Scenario D exercises the multi_addr=true path
 * but with a tiny 8192-bit bloom and only 4096 synthetic words; that
 * is not enough surface to hit a content-dependent or
 * size-dependent OOB.
 *
 * This test extends the gpu_hash160 oracle approach (build a CPU-derived
 * h160 set, stuff it into a bloom, ask the GPU to find them) along three
 * axes that the existing tests do not cover together:
 *
 *   1. multi_addr = true. Forces the kernel into fused_multiaddr_extra_check
 *      for every non-matching passphrase, exercising the uncompressed
 *      P2PKH and P2SH-P2WPKH chains.
 *
 *   2. Three h160 flavours per passphrase. The CPU pre-image set
 *      contains:
 *        - compressed P2PKH h160       (RIPEMD160(SHA256(0x02|0x03 || x)))
 *        - uncompressed P2PKH h160     (RIPEMD160(SHA256(0x04 || x || y)))
 *        - P2SH-P2WPKH h160            (RIPEMD160(SHA256(0x00 0x14 || compressed_h160)))
 *      Each is inserted into the bloom under the SAME MurmurHash3-128
 *      probe pattern the kernel uses, so a hit on any of the three
 *      counts as a match. A correct kernel reports N matches (one per
 *      passphrase). A kernel that walks off the end of the bloom buffer
 *      under the V2 path will surface a cudaErrorIllegalAddress here.
 *
 *   3. Bloom-size matrix. We run the test against two bloom sizes:
 *        - 2^17 bits  (pow-of-2 fast path; bloom_mask = num_bits - 1)
 *        - 100003 bits (prime, forces the 64-bit modulo path the
 *                       user's 2.57 GB production bloom is on)
 *      Both sizes are kept TU-local so the test still runs on a 4 GB GPU.
 *      The mod path is the one that matters for the production fault;
 *      the pow-of-2 path is included so a regression that only breaks
 *      mask vs. modulo is caught.
 *
 *  4. Random privkeys with a fixed PRNG seed. 2048 distinct privkeys
 *      means 2048 * 3 = 6144 bloom inserts, well inside the 32K match
 *      cap. The 2048-input run is deterministic across machines (the
 *      seed is hard-coded) so a flake is a real regression.
 *
 * This test will not catch a fault that requires literally billions of
 * passphrases; the compute-sanitizer wrapper in
 * scripts/run_compute_sanitizer.bat is the tool for that. What this DOES
 * catch is any structural OOB in the multi_addr extra-check path: a wrong
 * stride, an off-by-one in the uncompressed-pubkey serialisation, a wrong
 * length passed to sha256_short, a divergent murmur3 seed between insert
 * and probe, or a missing pow-of-2 fast-path equivalence.
 *
 * Exit codes:
 *   0   pass (every passphrase matched once, no kernel error)
 *   1   fail (match count != N, or a kernel error was reported)
 *   2   environment error (allocation, init, etc.)
 *   77  skip (no CUDA device available)
 */

#include <cuda_runtime.h>
#include "../src/core/crypto_cpu.hpp"
#include "../src/tools/utxo_bloom_builder.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <array>
#include <string>
#include <random>

extern "C" {
    cudaError_t fused_pipeline_init(cudaStream_t stream);
    cudaError_t fused_brain_wallet_batch_fixed_stride(
        const uint8_t* d_passphrases,
        const uint32_t* d_lengths,
        uint32_t stride,
        const uint8_t* d_bloom_filter,
        uint64_t bloom_bits,
        uint64_t bloom_mask,
        int bloom_hashes,
        uint32_t bloom_seed,
        uint32_t* d_match_indices,
        uint32_t* d_match_count,
        uint8_t* d_private_keys,
        size_t count,
        cudaStream_t stream,
        uint8_t multi_addr
    );
}

// ---------------------------------------------------------------------------
// CPU references for the three address flavours the V2 kernel probes.
// Mirror the kernel's serialisation byte-for-byte so an insert here
// equals a probe on the device.
// ---------------------------------------------------------------------------

namespace {

// Encode pub_x/pub_y as 0x04 || x_be || y_be (65 bytes), then hash160 it.
std::array<uint8_t, 20> uncompressed_hash160(const collider::cpu::uint256_t& pub_x,
                                             const collider::cpu::uint256_t& pub_y)
{
    uint8_t buf[65];
    buf[0] = 0x04;
    auto write_be = [&](size_t base, const collider::cpu::uint256_t& v) {
        for (int limb = 3; limb >= 0; --limb) {
            uint64_t w = v.d[limb];
            buf[base + 0] = (uint8_t)((w >> 56) & 0xff);
            buf[base + 1] = (uint8_t)((w >> 48) & 0xff);
            buf[base + 2] = (uint8_t)((w >> 40) & 0xff);
            buf[base + 3] = (uint8_t)((w >> 32) & 0xff);
            buf[base + 4] = (uint8_t)((w >> 24) & 0xff);
            buf[base + 5] = (uint8_t)((w >> 16) & 0xff);
            buf[base + 6] = (uint8_t)((w >>  8) & 0xff);
            buf[base + 7] = (uint8_t)( w        & 0xff);
            base += 8;
        }
    };
    write_be(1, pub_x);
    write_be(33, pub_y);
    auto sha = collider::cpu::SHA256::hash(buf, 65);
    return collider::cpu::RIPEMD160::hash(sha.data(), 32);
}

// Encode 0x00 0x14 || compressed_h160 (22 bytes), then hash160 it.
std::array<uint8_t, 20> p2sh_p2wpkh_hash160(const std::array<uint8_t, 20>& compressed_h160) {
    uint8_t redeem[22];
    redeem[0] = 0x00;
    redeem[1] = 0x14;
    std::memcpy(redeem + 2, compressed_h160.data(), 20);
    auto sha = collider::cpu::SHA256::hash(redeem, 22);
    return collider::cpu::RIPEMD160::hash(sha.data(), 32);
}

struct H160Triple {
    std::array<uint8_t, 20> compressed;
    std::array<uint8_t, 20> uncompressed;
    std::array<uint8_t, 20> p2sh;
};

H160Triple derive_triple(const uint8_t privkey_bytes[32]) {
    H160Triple out;
    // Compressed (the canonical compute_hash160 path also exercised by
    // test_gpu_hash160 -- reuse it so any future tweak to compute_hash160
    // automatically applies here too).
    out.compressed = collider::cpu::compute_hash160(privkey_bytes);

    // Re-derive pub_x / pub_y for the uncompressed pubkey.
    collider::cpu::uint256_t k;
    k.d[3] = ((uint64_t)privkey_bytes[0]  << 56) | ((uint64_t)privkey_bytes[1]  << 48) |
             ((uint64_t)privkey_bytes[2]  << 40) | ((uint64_t)privkey_bytes[3]  << 32) |
             ((uint64_t)privkey_bytes[4]  << 24) | ((uint64_t)privkey_bytes[5]  << 16) |
             ((uint64_t)privkey_bytes[6]  <<  8) |  (uint64_t)privkey_bytes[7];
    k.d[2] = ((uint64_t)privkey_bytes[8]  << 56) | ((uint64_t)privkey_bytes[9]  << 48) |
             ((uint64_t)privkey_bytes[10] << 40) | ((uint64_t)privkey_bytes[11] << 32) |
             ((uint64_t)privkey_bytes[12] << 24) | ((uint64_t)privkey_bytes[13] << 16) |
             ((uint64_t)privkey_bytes[14] <<  8) |  (uint64_t)privkey_bytes[15];
    k.d[1] = ((uint64_t)privkey_bytes[16] << 56) | ((uint64_t)privkey_bytes[17] << 48) |
             ((uint64_t)privkey_bytes[18] << 40) | ((uint64_t)privkey_bytes[19] << 32) |
             ((uint64_t)privkey_bytes[20] << 24) | ((uint64_t)privkey_bytes[21] << 16) |
             ((uint64_t)privkey_bytes[22] <<  8) |  (uint64_t)privkey_bytes[23];
    k.d[0] = ((uint64_t)privkey_bytes[24] << 56) | ((uint64_t)privkey_bytes[25] << 48) |
             ((uint64_t)privkey_bytes[26] << 40) | ((uint64_t)privkey_bytes[27] << 32) |
             ((uint64_t)privkey_bytes[28] << 24) | ((uint64_t)privkey_bytes[29] << 16) |
             ((uint64_t)privkey_bytes[30] <<  8) |  (uint64_t)privkey_bytes[31];
    collider::cpu::ECPoint P;
    collider::cpu::ec_mul(P, k);
    collider::cpu::uint256_t pub_x, pub_y;
    collider::cpu::ec_to_affine(pub_x, pub_y, P);

    out.uncompressed = uncompressed_hash160(pub_x, pub_y);
    out.p2sh = p2sh_p2wpkh_hash160(out.compressed);
    return out;
}

// MurmurHash3-128 bloom insert/probe -- byte-for-byte identical to
// utxo_bloom_builder.hpp::murmurhash3_128 and the kernel's
// bloom_check_inline.
void bloom_insert(uint8_t* bloom, uint64_t bloom_bits, int num_hashes,
                  uint32_t seed, const uint8_t* h160)
{
    auto [h1, h2] = ::collider::utxo::murmurhash3_128(h160, 20, seed);
    for (int i = 0; i < num_hashes; i++) {
        uint64_t idx = (h1 + (uint64_t)i * h2) % bloom_bits;
        bloom[idx / 8] |= (uint8_t)(1u << (idx % 8));
    }
}

bool bloom_probe(const uint8_t* bloom, uint64_t bloom_bits, int num_hashes,
                 uint32_t seed, const uint8_t* h160)
{
    auto [h1, h2] = ::collider::utxo::murmurhash3_128(h160, 20, seed);
    for (int i = 0; i < num_hashes; i++) {
        uint64_t idx = (h1 + (uint64_t)i * h2) % bloom_bits;
        if (!(bloom[idx / 8] & (1u << (idx % 8)))) return false;
    }
    return true;
}

// Generate N random privkeys + their (passphrase, expected h160 triple)
// pairs. The passphrase is just the hex form of an arbitrary seed; the
// actual privkey on the GPU is SHA256(passphrase), and the CPU
// reference uses the same chain, so this is end-to-end equivalent to
// what the production runner does with rule-engine output.
struct OracleEntry {
    std::string passphrase;
    H160Triple expected;
    // Which flavour we inserted for this entry. The kernel only needs
    // ONE bloom hit per passphrase to report a match; we rotate flavours
    // so all three paths get covered. 0 = compressed, 1 = uncompressed,
    // 2 = P2SH-P2WPKH.
    int flavour;
};

std::vector<OracleEntry> build_oracle(size_t n, uint64_t prng_seed) {
    std::vector<OracleEntry> out;
    out.reserve(n);
    std::mt19937_64 rng(prng_seed);
    for (size_t i = 0; i < n; ++i) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "v2-oracle-%llu",
                      (unsigned long long)rng());
        std::string pp = buf;
        auto sha = collider::cpu::SHA256::hash(
            reinterpret_cast<const uint8_t*>(pp.data()), pp.size());
        OracleEntry e;
        e.passphrase = std::move(pp);
        e.expected = derive_triple(sha.data());
        e.flavour = static_cast<int>(i % 3);
        out.push_back(std::move(e));
    }
    return out;
}

// One bloom-size pass. Returns true on success, false on any failure.
bool run_pass(const std::vector<OracleEntry>& entries,
              uint64_t bloom_bits,
              int bloom_hashes,
              uint32_t bloom_seed,
              const char* label)
{
    const size_t N = entries.size();
    const uint32_t STRIDE = 64;  // enough for our short passphrases

    // Build bloom: each entry contributes ONE h160 flavour, so the
    // kernel must hit it via the right address-derivation path.
    const size_t bloom_bytes = (bloom_bits + 7) / 8;
    std::vector<uint8_t> bloom(bloom_bytes, 0);
    for (const auto& e : entries) {
        const uint8_t* h = nullptr;
        switch (e.flavour) {
            case 0: h = e.expected.compressed.data();   break;
            case 1: h = e.expected.uncompressed.data(); break;
            case 2: h = e.expected.p2sh.data();         break;
        }
        bloom_insert(bloom.data(), bloom_bits, bloom_hashes, bloom_seed, h);
    }

    // Self-check: every inserted h160 must probe true on host.
    for (const auto& e : entries) {
        const uint8_t* h = nullptr;
        switch (e.flavour) {
            case 0: h = e.expected.compressed.data();   break;
            case 1: h = e.expected.uncompressed.data(); break;
            case 2: h = e.expected.p2sh.data();         break;
        }
        if (!bloom_probe(bloom.data(), bloom_bits, bloom_hashes, bloom_seed, h)) {
            std::fprintf(stderr, "[%s] INTERNAL: host bloom probe failed\n", label);
            return false;
        }
    }

    // Pack passphrases.
    std::vector<uint8_t> packed(N * STRIDE, 0);
    std::vector<uint32_t> lengths(N, 0);
    for (size_t i = 0; i < N; ++i) {
        const auto& pp = entries[i].passphrase;
        if (pp.size() > STRIDE) {
            std::fprintf(stderr, "[%s] passphrase too long\n", label);
            return false;
        }
        std::memcpy(&packed[i * STRIDE], pp.data(), pp.size());
        lengths[i] = (uint32_t)pp.size();
    }

    // Allocate device.
    uint8_t*  d_passphrases = nullptr;
    uint32_t* d_lengths     = nullptr;
    uint8_t*  d_bloom       = nullptr;
    uint32_t* d_match_idx   = nullptr;
    uint32_t* d_match_cnt   = nullptr;

    cudaMalloc(&d_passphrases, packed.size());
    cudaMalloc(&d_lengths,     N * sizeof(uint32_t));
    cudaMalloc(&d_bloom,       bloom.size());
    cudaMalloc(&d_match_idx,   N * sizeof(uint32_t));
    cudaMalloc(&d_match_cnt,   sizeof(uint32_t));

    cudaMemcpy(d_passphrases, packed.data(), packed.size(),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths,     lengths.data(), N * sizeof(uint32_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_bloom,       bloom.data(), bloom.size(),
               cudaMemcpyHostToDevice);
    cudaMemset(d_match_cnt, 0, sizeof(uint32_t));

    // bloom_mask is num_bits - 1 only when num_bits is a power of two;
    // 0 otherwise (forces the modulo path on the device).
    uint64_t bloom_mask = 0;
    if ((bloom_bits & (bloom_bits - 1)) == 0) {
        bloom_mask = bloom_bits - 1;
    }

    cudaError_t err = fused_brain_wallet_batch_fixed_stride(
        d_passphrases, d_lengths, STRIDE,
        d_bloom, bloom_bits, bloom_mask, bloom_hashes,
        bloom_seed,
        d_match_idx, d_match_cnt,
        /*d_private_keys=*/nullptr,
        N,
        /*stream=*/0,
        /*multi_addr=*/uint8_t{1}
    );

    if (err != cudaSuccess) {
        std::fprintf(stderr, "[%s] kernel launch failed: %s\n",
                     label, cudaGetErrorString(err));
        cudaFree(d_passphrases); cudaFree(d_lengths); cudaFree(d_bloom);
        cudaFree(d_match_idx);   cudaFree(d_match_cnt);
        return false;
    }
    cudaDeviceSynchronize();

    cudaError_t async_err = cudaGetLastError();
    if (async_err != cudaSuccess) {
        std::fprintf(stderr, "[%s] async kernel fault: %s\n",
                     label, cudaGetErrorString(async_err));
        cudaFree(d_passphrases); cudaFree(d_lengths); cudaFree(d_bloom);
        cudaFree(d_match_idx);   cudaFree(d_match_cnt);
        return false;
    }

    uint32_t match_count = 0;
    cudaMemcpy(&match_count, d_match_cnt, sizeof(uint32_t),
               cudaMemcpyDeviceToHost);

    cudaFree(d_passphrases); cudaFree(d_lengths); cudaFree(d_bloom);
    cudaFree(d_match_idx);   cudaFree(d_match_cnt);

    std::printf("[%s] bloom_bits=%llu bloom_mask=%s matches=%u expected=%zu\n",
                label,
                (unsigned long long)bloom_bits,
                bloom_mask ? "pow-of-2" : "modulo",
                match_count,
                N);

    if (static_cast<size_t>(match_count) != N) {
        std::fprintf(stderr,
            "[%s] FAIL: V2 multi_addr kernel produced wrong match count.\n",
            label);
        return false;
    }
    return true;
}

}  // namespace

int main() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::fprintf(stderr, "No CUDA devices: %s\n",
                     err == cudaSuccess ? "device count == 0"
                                        : cudaGetErrorString(err));
        return 77;
    }
    cudaSetDevice(0);

    err = fused_pipeline_init(/*stream=*/0);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "fused_pipeline_init failed: %s\n",
                     cudaGetErrorString(err));
        return 2;
    }
    cudaDeviceSynchronize();

    // 2048 entries * 3 flavours = 6144 bloom inserts; small enough to fit
    // in the 32K match cap with headroom, large enough that an off-by-one
    // in the V2 multi_addr serialisation will surface as a non-zero miss
    // count.
    auto entries = build_oracle(/*n=*/2048, /*prng_seed=*/0xC011DE52E15ED5EEull);
    constexpr uint32_t BLOOM_SEED = 0xCAFED00Du;
    constexpr int BLOOM_HASHES = 10;  // matches user's production k=10

    // Pass 1: pow-of-2 bloom (fast-path mask = num_bits - 1).
    if (!run_pass(entries, /*bloom_bits=*/(1ull << 17),
                  BLOOM_HASHES, BLOOM_SEED, "pow2-128Kbit"))
    {
        return 1;
    }

    // Pass 2: prime-size bloom (forces modulo path, the same code path
    // the user's 2.57 GB production bloom is on).
    if (!run_pass(entries, /*bloom_bits=*/100003ull,
                  BLOOM_HASHES, BLOOM_SEED, "mod-100003"))
    {
        return 1;
    }

    std::printf("PASS: V2 multi_addr kernel produced exact match count on both bloom shapes.\n");
    return 0;
}
