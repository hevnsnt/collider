/**
 * test_v2_scheme_bloom -- proves the v1.5.4 scheme->bloom DISCOVERY path.
 *
 * Before v1.5.4 the only DerivationScheme that could discover a funded
 * wallet was SHA256_PW, baked into the fused kernel
 * (fused_pipeline.cu). Every other scheme reached only
 * v2_check_priv_against_puzzles, a compare against a fixed list of
 * already-solved puzzle keys -- it could confirm a known key but never
 * find a funded address. The audit graded the brainwallet multi-scheme
 * feature as failing for exactly this reason.
 *
 * v2_derive_priv_batch closes the gap: it runs a scheme's derivation and
 * writes the priv in the little-endian-limb layout
 * secp256k1_batch_mul_simple consumes, so the existing tested EC ->
 * v2_multi_address_check -> bloom tail can then discover a funded wallet
 * under ANY scheme.
 *
 * Oracle approach (same shape as test_gpu_multi_addr): for each
 * passphrase, derive the priv on the CPU under the SAME scheme recipe,
 * compute its compressed-P2PKH hash160, and seed a bloom with it. Then
 * run the GPU path (derive -> EC -> multi_address_check) and require a
 * bloom hit for every passphrase. A wrong derivation, a wrong BE->LE
 * limb permutation, or a broken EC step all surface as a missed match.
 *
 * Two schemes are covered so the per-scheme dispatch and a non-trivial
 * (double-hash) recipe are both exercised:
 *   - SHA256_PW          : priv = SHA256(pw)
 *   - SHA256_SHA256_PW   : priv = SHA256(SHA256(pw))
 *
 * Exit codes:
 *   0  pass
 *   1  fail (wrong match count or kernel error)
 *   2  environment error (alloc / init)
 *   77 skip (no CUDA device)
 */

#include <cuda_runtime.h>

#include "gpu/v2/brain_wallet_v2.hpp"   // DerivationScheme, AddressType,
                                        // V2MatchRecord, v2_derive_priv_batch,
                                        // v2_multi_address_check
#include "gpu/v2/v2_orchestrator.hpp"   // MultiAddressSession
#include "../src/core/crypto_cpu.hpp"   // SHA256, compute_hash160
#include "gpu/v2/sha512_cpu.hpp"        // internal::sha512 / hmac_sha512
#include "gpu/v2/keccak256_cpu.hpp"     // keccak256::keccak256 (Ethereum flavor)
#include "../src/tools/utxo_bloom_builder.hpp"  // murmurhash3_128, BloomFilterHeader

#include <cstdio>
#include <fstream>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// secp256k1.cu exposes these with C linkage (see v2_orchestrator.cpp).
extern "C" cudaError_t secp256k1_init_table(cudaStream_t stream);
extern "C" cudaError_t secp256k1_batch_mul_simple(
    const void* d_private_keys, void* d_public_keys,
    size_t count, cudaStream_t stream);

namespace {

using collider::gpu::v2::DerivationScheme;
using collider::gpu::v2::AddressType;
using collider::gpu::v2::V2MatchRecord;

// Small helper: SHA256 over a byte vector into a 32-byte array.
std::array<uint8_t, 32> sha256_vec(const std::vector<uint8_t>& v) {
    auto h = collider::cpu::SHA256::hash(v.data(), v.size());
    std::array<uint8_t, 32> out{};
    std::memcpy(out.data(), h.data(), 32);
    return out;
}

// SHA256(salt || pw). Shared by the four salt-prefix schemes. Mirrors the
// device helper v2_d_sha256_fixed_salt_pw byte-for-byte.
std::array<uint8_t, 32> sha256_salt_pw(const char* salt,
                                       const std::string& pw) {
    std::vector<uint8_t> buf(salt, salt + std::strlen(salt));
    buf.insert(buf.end(), pw.begin(), pw.end());
    return sha256_vec(buf);
}

// CPU mirror of the scheme's priv derivation (32-byte big-endian priv).
// Each branch mirrors the matching device helper v2_d_* in
// src/gpu/v2/brain_wallet_v2.cu byte-for-byte (trailing bytes, salt strings,
// iteration counts, byte reversal, HMAC/SHA512 truncation, keccak flavor).
std::array<uint8_t, 32> cpu_derive(DerivationScheme scheme,
                                   const std::string& pw)
{
    const auto* p = reinterpret_cast<const uint8_t*>(pw.data());
    std::array<uint8_t, 32> out{};
    switch (scheme) {
    case DerivationScheme::SHA256_PW: {
        // priv = SHA256(pw)
        auto h = collider::cpu::SHA256::hash(p, pw.size());
        std::memcpy(out.data(), h.data(), 32);
        break;
    }
    case DerivationScheme::SHA256_SHA256_PW: {
        // priv = SHA256(SHA256(pw))
        auto h1 = collider::cpu::SHA256::hash(p, pw.size());
        auto h2 = collider::cpu::SHA256::hash(h1.data(), 32);
        std::memcpy(out.data(), h2.data(), 32);
        break;
    }
    case DerivationScheme::SHA256_PW_NEWLINE: {
        // priv = SHA256(pw || 0x0a)
        std::vector<uint8_t> buf(p, p + pw.size());
        buf.push_back(0x0a);
        out = sha256_vec(buf);
        break;
    }
    case DerivationScheme::SHA256_PW_PW: {
        // priv = SHA256(pw || pw)
        std::vector<uint8_t> buf(p, p + pw.size());
        buf.insert(buf.end(), p, p + pw.size());
        out = sha256_vec(buf);
        break;
    }
    case DerivationScheme::SHA256_SHA256_PW_PW: {
        // priv = SHA256(SHA256(pw) || pw)
        auto inner = collider::cpu::SHA256::hash(p, pw.size());
        std::vector<uint8_t> buf(inner.begin(), inner.end());
        buf.insert(buf.end(), p, p + pw.size());
        out = sha256_vec(buf);
        break;
    }
    case DerivationScheme::SHA256_ITER_16: {
        // priv = SHA256^16(pw): one hash of pw, then 15 more over the digest.
        auto a = collider::cpu::SHA256::hash(p, pw.size());
        for (int i = 0; i < 15; ++i) a = collider::cpu::SHA256::hash(a.data(), 32);
        std::memcpy(out.data(), a.data(), 32);
        break;
    }
    case DerivationScheme::HMAC_SHA512_PW: {
        // priv = HMAC-SHA512(key="", msg=pw)[:32]
        uint8_t mac[64];
        static const uint8_t empty_key[1] = {0};
        collider::gpu::v2::internal::hmac_sha512(empty_key, 0, p, pw.size(), mac);
        std::memcpy(out.data(), mac, 32);
        break;
    }
    case DerivationScheme::SHA512_PW_HALF: {
        // priv = SHA512(pw)[:32]
        uint8_t full[64];
        collider::gpu::v2::internal::sha512(p, pw.size(), full);
        std::memcpy(out.data(), full, 32);
        break;
    }
    case DerivationScheme::SHA256_PW_CRLF: {
        // priv = SHA256(pw || 0x0d 0x0a)
        std::vector<uint8_t> buf(p, p + pw.size());
        buf.push_back(0x0d);
        buf.push_back(0x0a);
        out = sha256_vec(buf);
        break;
    }
    case DerivationScheme::SHA256_PW_CR: {
        // priv = SHA256(pw || 0x0d)
        std::vector<uint8_t> buf(p, p + pw.size());
        buf.push_back(0x0d);
        out = sha256_vec(buf);
        break;
    }
    case DerivationScheme::SHA256_PW_NUL: {
        // priv = SHA256(pw || 0x00)
        std::vector<uint8_t> buf(p, p + pw.size());
        buf.push_back(0x00);
        out = sha256_vec(buf);
        break;
    }
    case DerivationScheme::SHA256_PW_LE_SCALAR: {
        // priv_bytes = reverse(SHA256(pw))
        auto h = collider::cpu::SHA256::hash(p, pw.size());
        for (int i = 0; i < 32; ++i) out[i] = h[31 - i];
        break;
    }
    case DerivationScheme::SHA256_SALT_BRAINWALLET:
        out = sha256_salt_pw("brainwallet", pw);
        break;
    case DerivationScheme::SHA256_SALT_BITCOIN:
        out = sha256_salt_pw("bitcoin", pw);
        break;
    case DerivationScheme::SHA256_SALT_WALLET:
        out = sha256_salt_pw("wallet", pw);
        break;
    case DerivationScheme::SHA256_SALT_PASSWORD:
        out = sha256_salt_pw("password", pw);
        break;
    case DerivationScheme::KECCAK_PW: {
        // priv = keccak256(pw) (Ethereum flavor, NOT FIPS-202 SHA-3).
        collider::gpu::v2::keccak256::keccak256(p, pw.size(), out.data());
        break;
    }
    case DerivationScheme::ELECTRUM_V1: {
        // x_0 = seed; x_{n+1} = SHA256(x_n || seed); priv = x_100000.
        // First round hashes (seed || seed); mirrors the device helper.
        std::vector<uint8_t> buf0(p, p + pw.size());
        buf0.insert(buf0.end(), p, p + pw.size());
        auto st = collider::cpu::SHA256::hash(buf0.data(), buf0.size());
        std::memcpy(out.data(), st.data(), 32);
        std::vector<uint8_t> buf(32 + pw.size());
        std::memcpy(buf.data() + 32, p, pw.size());
        for (int i = 1; i < 100000; ++i) {
            std::memcpy(buf.data(), out.data(), 32);
            auto h = collider::cpu::SHA256::hash(buf.data(), buf.size());
            std::memcpy(out.data(), h.data(), 32);
        }
        break;
    }
    default:
        std::fprintf(stderr, "cpu_derive: unhandled scheme %d\n",
                     (int)scheme);
        break;
    }
    return out;
}

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

bool run_scheme(DerivationScheme scheme, const char* label, size_t N = 1024)
{
    constexpr uint32_t STRIDE = 48;          // ample for the short test pws
    constexpr uint64_t BLOOM_BITS = (1ull << 17);
    constexpr int BLOOM_HASHES = 10;
    constexpr uint32_t BLOOM_SEED = 0xCAFED00Du;
    // Probe the compressed-P2PKH derivation; bloom is seeded with the same.
    const uint32_t addr_mask = collider::gpu::v2::addr_bit(
        AddressType::P2PKH_COMPRESSED);

    // Build oracle: passphrase -> CPU priv -> compressed h160.
    std::mt19937_64 rng(0x5C13E3B100ULL + (uint64_t)scheme);
    std::vector<std::string> pws(N);
    std::vector<std::array<uint8_t, 20>> h160s(N);
    for (size_t i = 0; i < N; ++i) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "sbloom-%llu",
                      (unsigned long long)rng());
        pws[i] = buf;
        auto priv = cpu_derive(scheme, pws[i]);
        h160s[i] = collider::cpu::compute_hash160(priv.data());
    }

    // Seed + self-check the bloom.
    const size_t bloom_bytes = (BLOOM_BITS + 7) / 8;
    std::vector<uint8_t> bloom(bloom_bytes, 0);
    for (size_t i = 0; i < N; ++i)
        bloom_insert(bloom.data(), BLOOM_BITS, BLOOM_HASHES, BLOOM_SEED,
                     h160s[i].data());
    for (size_t i = 0; i < N; ++i) {
        if (!bloom_probe(bloom.data(), BLOOM_BITS, BLOOM_HASHES, BLOOM_SEED,
                         h160s[i].data())) {
            std::fprintf(stderr, "[%s] INTERNAL: host bloom probe failed\n",
                         label);
            return false;
        }
    }

    // Pack passphrases with per-entry offsets/lengths (the derive kernel
    // takes offsets, not a fixed stride).
    std::vector<uint8_t> packed(N * STRIDE, 0);
    std::vector<uint32_t> offsets(N, 0);
    std::vector<uint32_t> lengths(N, 0);
    for (size_t i = 0; i < N; ++i) {
        if (pws[i].size() > STRIDE) {
            std::fprintf(stderr, "[%s] passphrase too long\n", label);
            return false;
        }
        offsets[i] = (uint32_t)(i * STRIDE);
        lengths[i] = (uint32_t)pws[i].size();
        std::memcpy(&packed[i * STRIDE], pws[i].data(), pws[i].size());
    }

    // Device allocations.
    uint8_t*  d_pass    = nullptr;
    uint32_t* d_off     = nullptr;
    uint32_t* d_len     = nullptr;
    uint8_t*  d_priv    = nullptr;   // count * 32, LE limbs
    uint8_t*  d_pub_xy  = nullptr;   // count * 64 (ECPointAffine X||Y)
    uint8_t*  d_bloom   = nullptr;
    V2MatchRecord* d_matches = nullptr;
    uint32_t* d_match_cnt = nullptr;

    auto cleanup = [&]() {
        cudaFree(d_pass); cudaFree(d_off); cudaFree(d_len);
        cudaFree(d_priv); cudaFree(d_pub_xy); cudaFree(d_bloom);
        cudaFree(d_matches); cudaFree(d_match_cnt);
    };

    if (cudaMalloc(&d_pass, packed.size())          != cudaSuccess ||
        cudaMalloc(&d_off,  N * sizeof(uint32_t))   != cudaSuccess ||
        cudaMalloc(&d_len,  N * sizeof(uint32_t))   != cudaSuccess ||
        cudaMalloc(&d_priv, N * 32)                 != cudaSuccess ||
        cudaMalloc(&d_pub_xy, N * 64)               != cudaSuccess ||
        cudaMalloc(&d_bloom, bloom.size())          != cudaSuccess ||
        cudaMalloc(&d_matches, N * sizeof(V2MatchRecord)) != cudaSuccess ||
        cudaMalloc(&d_match_cnt, sizeof(uint32_t))  != cudaSuccess) {
        std::fprintf(stderr, "[%s] cudaMalloc failed\n", label);
        cleanup();
        return false;
    }

    cudaMemcpy(d_pass, packed.data(), packed.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_off, offsets.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_len, lengths.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bloom, bloom.data(), bloom.size(), cudaMemcpyHostToDevice);
    cudaMemset(d_match_cnt, 0, sizeof(uint32_t));

    // Step 1: scheme derivation -> LE-limb privs.
    cudaError_t rc = collider::gpu::v2::v2_derive_priv_batch(
        scheme, d_pass, d_off, d_len, N, d_priv, /*stream=*/0);
    if (rc != cudaSuccess) {
        std::fprintf(stderr, "[%s] v2_derive_priv_batch: %s\n",
                     label, cudaGetErrorString(rc));
        cleanup();
        return false;
    }

    // Step 2: priv -> pub (X||Y).
    rc = secp256k1_batch_mul_simple(d_priv, d_pub_xy, N, /*stream=*/0);
    if (rc != cudaSuccess) {
        std::fprintf(stderr, "[%s] secp256k1_batch_mul_simple: %s\n",
                     label, cudaGetErrorString(rc));
        cleanup();
        return false;
    }

    // Step 3: pub -> h160 -> bloom probe.
    rc = collider::gpu::v2::v2_multi_address_check(
        d_pub_xy, N, addr_mask,
        d_bloom, BLOOM_BITS, BLOOM_HASHES, BLOOM_SEED,
        d_matches, d_match_cnt, /*stream=*/0);
    if (rc != cudaSuccess) {
        std::fprintf(stderr, "[%s] v2_multi_address_check: %s\n",
                     label, cudaGetErrorString(rc));
        cleanup();
        return false;
    }

    cudaDeviceSynchronize();
    cudaError_t async_err = cudaGetLastError();
    if (async_err != cudaSuccess) {
        std::fprintf(stderr, "[%s] async fault: %s\n",
                     label, cudaGetErrorString(async_err));
        cleanup();
        return false;
    }

    uint32_t match_count = 0;
    cudaMemcpy(&match_count, d_match_cnt, sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    cleanup();

    std::printf("[%s] matches=%u expected=%zu\n", label, match_count, N);
    if (static_cast<size_t>(match_count) != N) {
        std::fprintf(stderr,
            "[%s] FAIL: scheme->bloom path missed wallets "
            "(got %u, want %zu)\n", label, match_count, N);
        return false;
    }
    return true;
}

// Exercises the MultiAddressSession::process_passphrases entry point with a
// multi-scheme mask: one passphrase set, a bloom seeded with the compressed
// h160 under BOTH schemes, so every passphrase must match once per scheme
// (2*N total) and each record must carry the scheme_id that produced it.
bool run_session_multischeme(const char* label)
{
    constexpr size_t N = 512;
    constexpr uint64_t BLOOM_BITS = (1ull << 17);
    constexpr int BLOOM_HASHES = 10;
    constexpr uint32_t BLOOM_SEED = 0xCAFED00Du;
    const uint32_t addr_mask =
        collider::gpu::v2::addr_bit(AddressType::P2PKH_COMPRESSED);
    const uint32_t scheme_mask =
        collider::gpu::v2::scheme_bit(DerivationScheme::SHA256_PW) |
        collider::gpu::v2::scheme_bit(DerivationScheme::SHA256_SHA256_PW);

    std::mt19937_64 rng(0x9E3779B97F4A7C15ULL);
    std::vector<std::string> pws(N);
    const size_t bloom_bytes = (BLOOM_BITS + 7) / 8;
    std::vector<uint8_t> bloom(bloom_bytes, 0);
    for (size_t i = 0; i < N; ++i) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "sess-%llu",
                      (unsigned long long)rng());
        pws[i] = buf;
        for (auto scheme : {DerivationScheme::SHA256_PW,
                            DerivationScheme::SHA256_SHA256_PW}) {
            auto priv = cpu_derive(scheme, pws[i]);
            auto h = collider::cpu::compute_hash160(priv.data());
            bloom_insert(bloom.data(), BLOOM_BITS, BLOOM_HASHES, BLOOM_SEED,
                         h.data());
        }
    }

    collider::gpu::v2::MultiAddressSession session;
    int rc = session.init(bloom.data(), BLOOM_BITS, BLOOM_HASHES, BLOOM_SEED, N);
    if (rc != 0) {
        std::fprintf(stderr, "[%s] session.init failed: rc=%d (%s)\n",
                     label, rc, session.init_error_detail().c_str());
        return false;
    }

    rc = session.process_passphrases(pws, scheme_mask, addr_mask);
    if (rc != 0) {
        std::fprintf(stderr, "[%s] process_passphrases failed: rc=%d\n",
                     label, rc);
        return false;
    }

    auto recs = session.last_matches();
    size_t sha_pw = 0, sha_sha_pw = 0;
    for (const auto& r : recs) {
        if (r.scheme_id == (uint8_t)DerivationScheme::SHA256_PW) ++sha_pw;
        else if (r.scheme_id == (uint8_t)DerivationScheme::SHA256_SHA256_PW)
            ++sha_sha_pw;
    }
    std::printf("[%s] records=%zu (sha256_pw=%zu sha256_sha256_pw=%zu) "
                "expected %zu each\n",
                label, recs.size(), sha_pw, sha_sha_pw, N);

    if (recs.size() != 2 * N || sha_pw != N || sha_sha_pw != N) {
        std::fprintf(stderr,
            "[%s] FAIL: multi-scheme session match/attribution mismatch\n",
            label);
        return false;
    }
    return true;
}

// Write a loader-compatible .blf: 128-byte BloomFilterHeader + the bit
// array (num_bits/8 bytes), seeded with the given h160s. data_offset = 128
// matches what load_bloom_file_into_memory expects.
bool write_blf(const std::string& path, uint64_t num_bits, uint32_t num_hashes,
               uint32_t seed, const std::vector<std::array<uint8_t, 20>>& h160s)
{
    collider::utxo::BloomFilterHeader hdr{};  // magic "BLF1", version 1
    hdr.num_bits       = num_bits;
    hdr.num_hashes     = num_hashes;
    hdr.seed           = seed;
    hdr.num_elements   = h160s.size();
    hdr.target_fp_rate = 1e-5;
    hdr.data_offset    = sizeof(collider::utxo::BloomFilterHeader);

    const size_t nbytes = (num_bits + 7) / 8;
    std::vector<uint8_t> bits(nbytes, 0);
    for (const auto& h : h160s)
        bloom_insert(bits.data(), num_bits, num_hashes, seed, h.data());

    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    f.write(reinterpret_cast<const char*>(bits.data()),
            static_cast<std::streamsize>(nbytes));
    return f.good();
}

// End-to-end: drive run_v2_orchestrator in DISCOVERY mode (addr_mask != 0)
// against a generated .blf + wordlist, and confirm the seeded passphrase is
// reported as a hit while an un-seeded chaff line is not. This exercises the
// full live path the operator hits: dispatch -> bloom load -> session ->
// process_passphrases -> hit reporting.
bool run_discovery_e2e(const char* label)
{
    const std::string blf  = "test_disc_bloom.blf";
    const std::string wl   = "test_disc_words.txt";
    const std::string hits = "test_disc_hits.txt";
    const std::string target_pw = "discovery-test-pw-001";

    // Seed the bloom with the compressed h160 of SHA256(target_pw).
    auto priv = cpu_derive(DerivationScheme::SHA256_PW, target_pw);
    auto h = collider::cpu::compute_hash160(priv.data());
    std::vector<std::array<uint8_t, 20>> seeded{h};
    constexpr uint64_t BLOOM_BITS = (1ull << 20);
    constexpr uint32_t BLOOM_HASHES = 10;
    constexpr uint32_t BLOOM_SEED = 0x5F3759DFu;
    if (!write_blf(blf, BLOOM_BITS, BLOOM_HASHES, BLOOM_SEED, seeded)) {
        std::fprintf(stderr, "[%s] could not write .blf\n", label);
        return false;
    }

    {
        std::ofstream w(wl, std::ios::trunc);
        w << "chaff-line-not-in-bloom\n";
        w << target_pw << "\n";
        w << "another-chaff-line\n";
    }
    std::remove(hits.c_str());  // fresh sink (orchestrator appends)

    collider::gpu::v2::OrchestratorOptions opts;
    opts.wordlist_path = wl;
    opts.bloom_path    = blf;
    opts.hits_out_path = hits;
    opts.scheme_mask   = collider::gpu::v2::scheme_bit(DerivationScheme::SHA256_PW);
    opts.addr_mask     =
        collider::gpu::v2::addr_bit(AddressType::P2PKH_COMPRESSED);
    opts.show_summary  = false;

    int rc = collider::gpu::v2::run_v2_orchestrator(opts);
    if (rc != 0) {
        std::fprintf(stderr, "[%s] run_v2_orchestrator rc=%d\n", label, rc);
        return false;
    }

    std::ifstream hf(hits);
    std::string body((std::istreambuf_iterator<char>(hf)),
                     std::istreambuf_iterator<char>());
    std::remove(blf.c_str());
    std::remove(wl.c_str());
    std::remove(hits.c_str());

    const bool found_target = body.find(target_pw) != std::string::npos;
    const bool found_chaff =
        body.find("chaff-line-not-in-bloom") != std::string::npos ||
        body.find("another-chaff-line") != std::string::npos;

    std::printf("[%s] target_reported=%d chaff_reported=%d\n",
                label, (int)found_target, (int)found_chaff);
    if (!found_target) {
        std::fprintf(stderr,
            "[%s] FAIL: seeded passphrase not reported by discovery scan\n",
            label);
        return false;
    }
    if (found_chaff) {
        std::fprintf(stderr,
            "[%s] FAIL: un-seeded chaff reported (bloom false positive or "
            "wiring bug)\n", label);
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

    err = secp256k1_init_table(/*stream=*/0);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "secp256k1_init_table failed: %s\n",
                     cudaGetErrorString(err));
        return 2;
    }
    cudaDeviceSynchronize();

    // Exercise the scheme->bloom discovery path for ALL 18 DerivationScheme
    // values. Each entry derives on the CPU (cpu_derive mirror), seeds the
    // bloom with the compressed h160, then requires the GPU
    // derive -> EC -> multi_address_check path to find every wallet. The
    // light SHA-256 / keccak / SHA-512 schemes use the default N; ELECTRUM_V1
    // runs 100k SHA-256 rounds per entry on BOTH sides, so it uses a small N
    // to keep the CPU reference quick.
    struct SchemeCase {
        DerivationScheme scheme;
        const char*      label;
        size_t           n;
    };
    const SchemeCase kCases[] = {
        { DerivationScheme::SHA256_PW,              "sha256_pw",              1024 },
        { DerivationScheme::SHA256_SHA256_PW,       "sha256_sha256_pw",       1024 },
        { DerivationScheme::SHA256_PW_NEWLINE,      "sha256_pw_newline",      1024 },
        { DerivationScheme::SHA256_PW_PW,           "sha256_pw_pw",           1024 },
        { DerivationScheme::SHA256_SHA256_PW_PW,    "sha256_sha256_pw_pw",    1024 },
        { DerivationScheme::SHA256_ITER_16,         "sha256_iter_16",         1024 },
        { DerivationScheme::HMAC_SHA512_PW,         "hmac_sha512_pw",         1024 },
        { DerivationScheme::SHA512_PW_HALF,         "sha512_pw_half",         1024 },
        { DerivationScheme::SHA256_PW_CRLF,         "sha256_pw_crlf",         1024 },
        { DerivationScheme::SHA256_PW_CR,           "sha256_pw_cr",           1024 },
        { DerivationScheme::SHA256_PW_NUL,          "sha256_pw_nul",          1024 },
        { DerivationScheme::SHA256_PW_LE_SCALAR,    "sha256_pw_le_scalar",    1024 },
        { DerivationScheme::SHA256_SALT_BRAINWALLET,"sha256_salt_brainwallet",1024 },
        { DerivationScheme::SHA256_SALT_BITCOIN,    "sha256_salt_bitcoin",    1024 },
        { DerivationScheme::SHA256_SALT_WALLET,     "sha256_salt_wallet",     1024 },
        { DerivationScheme::SHA256_SALT_PASSWORD,   "sha256_salt_password",   1024 },
        { DerivationScheme::KECCAK_PW,              "keccak_pw",              1024 },
        { DerivationScheme::ELECTRUM_V1,            "electrum_v1",             128 },
    };
    for (const auto& c : kCases) {
        if (!run_scheme(c.scheme, c.label, c.n)) return 1;
    }

    if (!run_session_multischeme("session-multischeme")) return 1;
    if (!run_discovery_e2e("orchestrator-discovery-e2e")) return 1;

    std::printf("PASS: scheme->bloom discovery path found every wallet "
                "under all 18 derivation schemes, and the multi-scheme "
                "session accumulated + attributed hits correctly.\n");
    return 0;
}
