/**
 * Brain Wallet v2 — unit tests.
 *
 * Plain-assert style to match the rest of tests/ (test_rule_engine,
 * test_priority_queue, etc.). No external test framework required.
 *
 * Two layers:
 *   1. Host-only: PuzzleTarget construction and mask math. Runs anywhere.
 *   2. GPU-end-to-end: install a synthetic puzzle target, run the kernel
 *      against a small batch of passphrases (one of which is constructed to
 *      hit the target by design), verify exactly one match. Auto-skipped
 *      when no CUDA device is present.
 *
 * The GPU test is the high-confidence correctness check: if the synthetic
 * target round-trips through {host construction → constant memory →
 * kernel SHA-256 → mask compare → match record}, the implementation is
 * end-to-end consistent.
 */

#include "../../src/gpu/v2/brain_wallet_v2.hpp"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <vector>
#include <string>

#ifdef COLLIDER_USE_CUDA
#include <cuda_runtime.h>
#endif

using namespace collider::gpu::v2;

// ---------------------------------------------------------------------------
// CPU SHA-256 reference (self-contained -- avoid forcing the test to depend
// on OpenSSL build-time availability).
// ---------------------------------------------------------------------------
namespace cpu_sha256_ref {

static const uint32_t K256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

static inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

static void compress(uint32_t state[8], const uint32_t W_in[16]) {
    uint32_t W[64];
    for (int i = 0; i < 16; i++) W[i] = W_in[i];
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr(W[i-15], 7) ^ rotr(W[i-15], 18) ^ (W[i-15] >> 3);
        uint32_t s1 = rotr(W[i-2], 17) ^ rotr(W[i-2], 19) ^ (W[i-2] >> 10);
        W[i] = W[i-16] + s0 + W[i-7] + s1;
    }
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = h + S1 + ch + K256[i] + W[i];
        uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + mj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

static void hash(const uint8_t* msg, size_t len, uint8_t out[32]) {
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    // Pad and process
    std::vector<uint8_t> buf(msg, msg + len);
    uint64_t bitlen = (uint64_t)len * 8;
    buf.push_back(0x80);
    while (buf.size() % 64 != 56) buf.push_back(0x00);
    for (int i = 7; i >= 0; i--) buf.push_back((uint8_t)(bitlen >> (i * 8)));
    for (size_t b = 0; b < buf.size(); b += 64) {
        uint32_t W[16];
        for (int i = 0; i < 16; i++) {
            W[i] = ((uint32_t)buf[b + i*4    ] << 24) |
                   ((uint32_t)buf[b + i*4 + 1] << 16) |
                   ((uint32_t)buf[b + i*4 + 2] <<  8) |
                   ((uint32_t)buf[b + i*4 + 3]);
        }
        compress(state, W);
    }
    for (int i = 0; i < 8; i++) {
        out[i*4    ] = (uint8_t)(state[i] >> 24);
        out[i*4 + 1] = (uint8_t)(state[i] >> 16);
        out[i*4 + 2] = (uint8_t)(state[i] >>  8);
        out[i*4 + 3] = (uint8_t)(state[i]);
    }
}

}  // namespace cpu_sha256_ref

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void hex_to_bytes(const std::string& hex_in, uint8_t* out, size_t out_size) {
    // Right-justify the hex into out. Auto-left-pad odd-length input with a
    // zero nibble so the puzzle keys (which are written in their natural form,
    // e.g. "33e7665..." for puzzle 130) round-trip correctly to a 32-byte BE
    // representation.
    std::string hex = (hex_in.size() % 2 == 0) ? hex_in : (std::string("0") + hex_in);
    std::memset(out, 0, out_size);
    size_t bytes_in = hex.size() / 2;
    if (bytes_in > out_size) {
        std::cerr << "hex_to_bytes: hex too long for buffer" << std::endl;
        std::abort();
    }
    size_t start = out_size - bytes_in;
    for (size_t i = 0; i < bytes_in; i++) {
        unsigned int b;
        std::sscanf(hex.c_str() + 2 * i, "%2x", &b);
        out[start + i] = (uint8_t)b;
    }
}

static void cpu_sha256(const uint8_t* msg, size_t len, uint8_t out[32]) {
    cpu_sha256_ref::hash(msg, len, out);
}

static int report_status(const char* label, bool ok) {
    std::cout << "  " << (ok ? "PASS" : "FAIL") << "  " << label << std::endl;
    return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Host tests
// ---------------------------------------------------------------------------

static int test_make_puzzle_target_small_n() {
    // Puzzle 1 has zero "low bits" to compare. To prevent the kernel from
    // reporting a vacuous match against EVERY input, make_puzzle_target
    // installs a sentinel: mask=0, value=1 (impossible to satisfy because
    // (any & 0) == 0, which is never == 1).
    uint8_t priv[32];
    hex_to_bytes("01", priv, 32);
    auto t = make_puzzle_target(1, priv);
    bool ok = (t.puzzle_n == 1)
           && (t.low_mask[0] == 0) && (t.low_mask[1] == 0)
           && (t.low_mask[2] == 0) && (t.low_mask[3] == 0)
           && (t.low_value[0] == 1) && (t.low_value[1] == 1)
           && (t.low_value[2] == 1) && (t.low_value[3] == 1);
    return report_status("make_puzzle_target N=1, sentinel never-match", ok);
}

static int test_make_puzzle_target_n65() {
    // Puzzle 65: priv = 0x1a838b13505b26867, low 64 bits = 0xa838b13505b26867
    uint8_t priv[32];
    hex_to_bytes("1a838b13505b26867", priv, 32);
    auto t = make_puzzle_target(65, priv);
    // low_bits = 64. full_limbs = 1 (limb[0] full). partial_bits = 0.
    bool ok = (t.puzzle_n == 65)
           && (t.low_mask[0] == ~0ULL)
           && (t.low_mask[1] == 0ULL)
           && (t.low_mask[2] == 0ULL)
           && (t.low_mask[3] == 0ULL)
           && (t.low_value[0] == 0xa838b13505b26867ULL)
           && (t.low_value[1] == 0ULL);
    return report_status("make_puzzle_target N=65 limb-0 boundary", ok);
}

static int test_make_puzzle_target_n130() {
    // Puzzle 130: priv = 0x33e7665705359f04f28b88cf897c603c9 (130 bits)
    // low_bits = 129. full_limbs = 2 (limbs 0,1 full). partial_bits = 1.
    uint8_t priv[32];
    hex_to_bytes("33e7665705359f04f28b88cf897c603c9", priv, 32);
    auto t = make_puzzle_target(130, priv);
    bool ok = (t.puzzle_n == 130)
           && (t.low_mask[0] == ~0ULL)
           && (t.low_mask[1] == ~0ULL)
           && (t.low_mask[2] == 1ULL)            // (1 << 1) - 1 = 1
           && (t.low_mask[3] == 0ULL);
    if (!ok) {
        std::cerr << "  [diag] mask=[0x" << std::hex
                  << t.low_mask[0] << ", 0x" << t.low_mask[1]
                  << ", 0x" << t.low_mask[2] << ", 0x" << t.low_mask[3] << "]"
                  << std::dec << std::endl;
    }
    return report_status("make_puzzle_target N=130 spans 3 limbs", ok);
}

static int test_make_puzzle_target_n160() {
    // Puzzle 160 (max): low_bits = 159. full_limbs = 2 (limbs 0,1).
    // partial_bits = 31 -> limb[2] = (1 << 31) - 1.
    uint8_t priv[32];
    // Synthetic priv with 160 bits: high bit at position 159, all-ones below.
    // = (1 << 159) | ((1 << 159) - 1)  in 32-byte BE form
    // Easier: 160 bits all set = 0xFFFFFFFF...F (40 hex chars = 160 bits)
    hex_to_bytes(std::string(40, 'f'), priv, 32);
    auto t = make_puzzle_target(160, priv);
    bool ok = (t.puzzle_n == 160)
           && (t.low_mask[0] == ~0ULL)
           && (t.low_mask[1] == ~0ULL)
           && (t.low_mask[2] == ((1ULL << 31) - 1ULL))
           && (t.low_mask[3] == 0ULL);
    if (!ok) {
        std::cerr << "  [diag] mask=[0x" << std::hex
                  << t.low_mask[0] << ", 0x" << t.low_mask[1]
                  << ", 0x" << t.low_mask[2] << ", 0x" << t.low_mask[3] << "]"
                  << std::dec << std::endl;
    }
    return report_status("make_puzzle_target N=160 max puzzle", ok);
}

static int test_scheme_bit_constants() {
    bool ok = (scheme_bit(DerivationScheme::SHA256_PW) == 1u)
           && (scheme_bit(DerivationScheme::SHA256_SHA256_PW) == 2u)
           && (SCHEME_MASK_STOCK == 1u)
           && (SCHEME_MASK_ALL == 0xFFu);
    return report_status("scheme_bit and SCHEME_MASK_* constants", ok);
}

static int test_addr_bit_constants() {
    bool ok = (addr_bit(AddressType::P2PKH_UNCOMPRESSED) == 1u)
           && (addr_bit(AddressType::P2TR_BIP86) == (1u << 4))
           && (ADDR_MASK_ALL == 0x1Fu);
    return report_status("addr_bit and ADDR_MASK_* constants", ok);
}

// ---------------------------------------------------------------------------
// GPU end-to-end test (skip when no CUDA device)
// ---------------------------------------------------------------------------

#ifdef COLLIDER_USE_CUDA
static bool has_cuda_device() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

/**
 * End-to-end: build a synthetic puzzle target whose private key low bits
 * equal SHA256(passphrase || N_be) low (N-1) bits, install it, run the
 * kernel, verify a match.
 *
 * Choose N=42 (small enough that hand-computation is easy, large enough
 * that chance match probability is < 2^-41 ≈ 5e-13 per trial).
 */
static int test_gpu_puzzle_kernel_end_to_end() {
    if (!has_cuda_device()) {
        std::cout << "  SKIP  GPU puzzle kernel end-to-end (no CUDA device)" << std::endl;
        return 0;
    }

    auto err = v2_init(nullptr);
    if (err != cudaSuccess) {
        std::cerr << "  v2_init returned " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Construct synthetic puzzle target from a chosen passphrase.
    // The kernel called below dispatches DerivationScheme::SHA256_PW
    // (stock), which derives priv = SHA256(passphrase). Build the
    // synthetic target against that exact derivation.
    const char* PHRASE = "collider-v2-self-test-phrase-do-not-use";
    const uint16_t N = 42;
    const uint64_t low_mask = (1ULL << (N - 1)) - 1ULL;

    // CPU reference: priv = SHA256(passphrase) (matches SCHEME_MASK_STOCK).
    size_t pw_len = std::strlen(PHRASE);
    uint8_t hash[32];
    cpu_sha256(reinterpret_cast<const uint8_t*>(PHRASE), pw_len, hash);

    // Synthesize the puzzle priv: top bit set at position (N-1), low bits
    // taken from the hash.  Read the low 64 bits of the hash big-endian.
    uint64_t hash_low_be = 0;
    for (int i = 0; i < 8; i++) hash_low_be = (hash_low_be << 8) | hash[24 + i];
    uint64_t low_value = hash_low_be & low_mask;
    uint64_t synthetic_priv = (1ULL << (N - 1)) | low_value;

    // Express synthetic_priv as 32-byte big-endian for make_puzzle_target.
    uint8_t synthetic_priv_be[32]{};
    for (int i = 0; i < 8; i++) {
        synthetic_priv_be[24 + i] = (uint8_t)(synthetic_priv >> (8 * (7 - i)));
    }
    auto target = make_puzzle_target(N, synthetic_priv_be);

    err = v2_set_puzzle_targets({target});
    if (err != cudaSuccess) {
        std::cerr << "  v2_set_puzzle_targets failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Build a batch of 4 passphrases. Index 1 is the synthetic match;
    // the others are random text that should not collide.
    std::vector<std::string> phrases = {
        "noise-phrase-aaaa",
        PHRASE,
        "noise-phrase-bbbb",
        "noise-phrase-cccc",
    };
    std::vector<uint32_t> offsets;
    std::vector<uint32_t> lengths;
    std::vector<uint8_t> packed;
    for (const auto& p : phrases) {
        offsets.push_back((uint32_t)packed.size());
        lengths.push_back((uint32_t)p.size());
        packed.insert(packed.end(), p.begin(), p.end());
    }

    uint8_t* d_passphrases = nullptr;
    uint32_t* d_offsets = nullptr;
    uint32_t* d_lengths = nullptr;
    V2MatchRecord* d_matches = nullptr;
    uint32_t* d_match_count = nullptr;
    cudaMalloc(&d_passphrases, packed.size());
    cudaMalloc(&d_offsets, offsets.size() * sizeof(uint32_t));
    cudaMalloc(&d_lengths, lengths.size() * sizeof(uint32_t));
    cudaMalloc(&d_matches, V2_MAX_MATCHES_PER_BATCH * sizeof(V2MatchRecord));
    cudaMalloc(&d_match_count, sizeof(uint32_t));
    cudaMemset(d_match_count, 0, sizeof(uint32_t));
    cudaMemcpy(d_passphrases, packed.data(), packed.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets.data(), offsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths, lengths.data(), lengths.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);

    err = v2_brain_wallet_batch(
        d_passphrases, d_offsets, d_lengths,
        phrases.size(),
        SCHEME_MASK_STOCK,   // S2 alias for now
        0u,                  // addr_mask = 0 (puzzle-only)
        nullptr, 0, 0,       // no bloom
        d_matches, d_match_count,
        nullptr);            // default stream
    if (err != cudaSuccess) {
        std::cerr << "  v2_brain_wallet_batch failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    cudaDeviceSynchronize();

    uint32_t h_match_count = 0;
    cudaMemcpy(&h_match_count, d_match_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    std::vector<V2MatchRecord> h_matches(h_match_count);
    if (h_match_count > 0) {
        cudaMemcpy(h_matches.data(), d_matches, h_match_count * sizeof(V2MatchRecord),
                   cudaMemcpyDeviceToHost);
    }

    cudaFree(d_passphrases); cudaFree(d_offsets); cudaFree(d_lengths);
    cudaFree(d_matches); cudaFree(d_match_count);

    bool ok = (h_match_count == 1)
           && (h_matches[0].pp_idx == 1u)
           && (h_matches[0].puzzle_n == N)
           && (h_matches[0].kind == (uint8_t)V2MatchRecord::Kind::PUZZLE_KEY_HIT);
    if (!ok) {
        std::cerr << "  [diag] h_match_count=" << h_match_count;
        if (h_match_count > 0) {
            std::cerr << " pp_idx=" << h_matches[0].pp_idx
                      << " puzzle_n=" << h_matches[0].puzzle_n
                      << " kind=" << (int)h_matches[0].kind;
        }
        std::cerr << std::endl;
    }
    return report_status("GPU puzzle kernel end-to-end with synthetic target", ok);
}
#endif  // COLLIDER_USE_CUDA

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    std::cout << "=== Brain Wallet v2 unit tests ===" << std::endl;
    int fails = 0;
    fails += test_make_puzzle_target_small_n();
    fails += test_make_puzzle_target_n65();
    fails += test_make_puzzle_target_n130();
    fails += test_make_puzzle_target_n160();
    fails += test_scheme_bit_constants();
    fails += test_addr_bit_constants();
#ifdef COLLIDER_USE_CUDA
    fails += test_gpu_puzzle_kernel_end_to_end();
#else
    std::cout << "  SKIP  GPU end-to-end (CUDA disabled at compile time)" << std::endl;
#endif
    std::cout << "=== " << (fails == 0 ? "OK" : "FAILED")
              << " (" << fails << " failures) ===" << std::endl;
    return fails == 0 ? 0 : 1;
}
