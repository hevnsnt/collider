/**
 * RCKangaroo Wrapper Implementation for theCollider
 *
 * Integrates RetiredCoder's RCKangaroo (GPLv3) as the Kangaroo solver backend.
 *
 * Original software: (c) 2024, RetiredCoder (RC)
 * https://github.com/RetiredC/RCKangaroo
 */

#include "rckangaroo_wrapper.hpp"

#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>
#include <thread>
#include <chrono>
#include <mutex>
#include <iomanip>

// RCKangaroo headers
#include "defs.h"
#include "Ec.h"
#include "GpuKang.h"
#include "utils.h"

#include "cuda_runtime.h"

// Global variables required by RCKangaroo (defined in RCKangaroo.cpp but we use our wrapper)
bool gGenMode = false;      // Tames generation mode
u32 gTotalErrors = 0;       // Error counter

// ============================================================================
// CPU-side SHA256 for bloom filter checking
// ============================================================================

static const uint32_t SHA256_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

inline uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
inline uint32_t rotl32(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }

static void cpu_sha256(const uint8_t* data, size_t len, uint8_t* hash) {
    uint32_t H[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    // Pad message
    size_t padded_len = ((len + 9 + 63) / 64) * 64;
    std::vector<uint8_t> padded(padded_len, 0);
    memcpy(padded.data(), data, len);
    padded[len] = 0x80;
    uint64_t bit_len = len * 8;
    for (int i = 0; i < 8; i++) {
        padded[padded_len - 1 - i] = (bit_len >> (i * 8)) & 0xff;
    }

    // Process blocks
    for (size_t block = 0; block < padded_len; block += 64) {
        uint32_t W[64];
        for (int i = 0; i < 16; i++) {
            W[i] = (padded[block + i*4] << 24) | (padded[block + i*4 + 1] << 16) |
                   (padded[block + i*4 + 2] << 8) | padded[block + i*4 + 3];
        }
        for (int i = 16; i < 64; i++) {
            uint32_t s0 = rotr32(W[i-15], 7) ^ rotr32(W[i-15], 18) ^ (W[i-15] >> 3);
            uint32_t s1 = rotr32(W[i-2], 17) ^ rotr32(W[i-2], 19) ^ (W[i-2] >> 10);
            W[i] = W[i-16] + s0 + W[i-7] + s1;
        }

        uint32_t a = H[0], b = H[1], c = H[2], d = H[3];
        uint32_t e = H[4], f = H[5], g = H[6], h = H[7];

        for (int i = 0; i < 64; i++) {
            uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t temp1 = h + S1 + ch + SHA256_K[i] + W[i];
            uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t temp2 = S0 + maj;

            h = g; g = f; f = e; e = d + temp1;
            d = c; c = b; b = a; a = temp1 + temp2;
        }

        H[0] += a; H[1] += b; H[2] += c; H[3] += d;
        H[4] += e; H[5] += f; H[6] += g; H[7] += h;
    }

    for (int i = 0; i < 8; i++) {
        hash[i*4] = (H[i] >> 24) & 0xff;
        hash[i*4 + 1] = (H[i] >> 16) & 0xff;
        hash[i*4 + 2] = (H[i] >> 8) & 0xff;
        hash[i*4 + 3] = H[i] & 0xff;
    }
}

// ============================================================================
// CPU-side RIPEMD160 for bloom filter checking
// ============================================================================

static const uint32_t RIPEMD160_KL[5] = {0x00000000, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E};
static const uint32_t RIPEMD160_KR[5] = {0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0x00000000};
static const int RIPEMD160_RL[80] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
    3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12,
    1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
    4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13
};
static const int RIPEMD160_RR[80] = {
    5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12,
    6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
    15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13,
    8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
    12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11
};
static const int RIPEMD160_SL[80] = {
    11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8,
    7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12,
    11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5,
    11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12,
    9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6
};
static const int RIPEMD160_SR[80] = {
    8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6,
    9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11,
    9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5,
    15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8,
    8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11
};

static void cpu_ripemd160(const uint8_t* data, size_t len, uint8_t* hash) {
    uint32_t H[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};

    // Pad message
    size_t padded_len = ((len + 9 + 63) / 64) * 64;
    std::vector<uint8_t> padded(padded_len, 0);
    memcpy(padded.data(), data, len);
    padded[len] = 0x80;
    uint64_t bit_len = len * 8;
    // Little-endian length
    for (int i = 0; i < 8; i++) {
        padded[padded_len - 8 + i] = (bit_len >> (i * 8)) & 0xff;
    }

    auto f0 = [](uint32_t x, uint32_t y, uint32_t z) { return x ^ y ^ z; };
    auto f1 = [](uint32_t x, uint32_t y, uint32_t z) { return (x & y) | (~x & z); };
    auto f2 = [](uint32_t x, uint32_t y, uint32_t z) { return (x | ~y) ^ z; };
    auto f3 = [](uint32_t x, uint32_t y, uint32_t z) { return (x & z) | (y & ~z); };
    auto f4 = [](uint32_t x, uint32_t y, uint32_t z) { return x ^ (y | ~z); };

    for (size_t block = 0; block < padded_len; block += 64) {
        uint32_t X[16];
        for (int i = 0; i < 16; i++) {
            X[i] = padded[block + i*4] | (padded[block + i*4 + 1] << 8) |
                   (padded[block + i*4 + 2] << 16) | (padded[block + i*4 + 3] << 24);
        }

        uint32_t al = H[0], bl = H[1], cl = H[2], dl = H[3], el = H[4];
        uint32_t ar = H[0], br = H[1], cr = H[2], dr = H[3], er = H[4];

        for (int j = 0; j < 80; j++) {
            uint32_t fl, fr;
            int round = j / 16;
            switch (round) {
                case 0: fl = f0(bl, cl, dl); fr = f4(br, cr, dr); break;
                case 1: fl = f1(bl, cl, dl); fr = f3(br, cr, dr); break;
                case 2: fl = f2(bl, cl, dl); fr = f2(br, cr, dr); break;
                case 3: fl = f3(bl, cl, dl); fr = f1(br, cr, dr); break;
                case 4: fl = f4(bl, cl, dl); fr = f0(br, cr, dr); break;
            }

            uint32_t tl = rotl32(al + fl + X[RIPEMD160_RL[j]] + RIPEMD160_KL[round], RIPEMD160_SL[j]) + el;
            al = el; el = dl; dl = rotl32(cl, 10); cl = bl; bl = tl;

            uint32_t tr = rotl32(ar + fr + X[RIPEMD160_RR[j]] + RIPEMD160_KR[round], RIPEMD160_SR[j]) + er;
            ar = er; er = dr; dr = rotl32(cr, 10); cr = br; br = tr;
        }

        uint32_t t = H[1] + cl + dr;
        H[1] = H[2] + dl + er;
        H[2] = H[3] + el + ar;
        H[3] = H[4] + al + br;
        H[4] = H[0] + bl + cr;
        H[0] = t;
    }

    for (int i = 0; i < 5; i++) {
        hash[i*4] = H[i] & 0xff;
        hash[i*4 + 1] = (H[i] >> 8) & 0xff;
        hash[i*4 + 2] = (H[i] >> 16) & 0xff;
        hash[i*4 + 3] = (H[i] >> 24) & 0xff;
    }
}

// Hash160 = RIPEMD160(SHA256(data))
static void cpu_hash160(const uint8_t* data, size_t len, uint8_t* hash160) {
    uint8_t sha256_hash[32];
    cpu_sha256(data, len, sha256_hash);
    cpu_ripemd160(sha256_hash, 32, hash160);
}

// ============================================================================
// Bloom filter structure for CPU checking
// ============================================================================

struct BloomFilter {
    std::vector<uint8_t> data;
    uint64_t num_bits;
    uint32_t num_hashes;
    uint32_t seed;
    bool loaded = false;

    bool load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) return false;

        // Read header (128 bytes)
        struct Header {
            char magic[4];
            uint32_t version;
            uint64_t num_bits;
            uint32_t num_hashes;
            uint32_t seed;
            uint64_t num_elements;
            double target_fp_rate;
            uint64_t data_offset;
            uint8_t reserved[80];
        } header;

        file.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (std::string(header.magic, 4) != "BLF1") {
            std::cerr << "Invalid bloom filter format" << std::endl;
            return false;
        }

        num_bits = header.num_bits;
        num_hashes = header.num_hashes;
        seed = header.seed;

        size_t data_size = (num_bits + 7) / 8;
        data.resize(data_size);

        file.seekg(header.data_offset);
        file.read(reinterpret_cast<char*>(data.data()), data_size);

        loaded = true;
        std::cout << "[Bloom] Loaded: " << (data_size / (1024*1024)) << " MB, "
                  << header.num_elements << " addresses, k=" << num_hashes << std::endl;
        return true;
    }

    // MurmurHash3 fmix64
    static uint64_t fmix64(uint64_t k) {
        k ^= k >> 33;
        k *= 0xff51afd7ed558ccdULL;
        k ^= k >> 33;
        k *= 0xc4ceb9fe1a85ec53ULL;
        k ^= k >> 33;
        return k;
    }

    bool check(const uint8_t* h160) const {
        if (!loaded) return false;

        // MurmurHash3 128-bit for 20-byte input
        const uint64_t c1 = 0x87c37b91114253d5ULL;
        const uint64_t c2 = 0x4cf5ad432745937fULL;

        uint64_t h1 = seed;
        uint64_t h2 = seed;

        // First 16 bytes
        uint64_t k1 = *reinterpret_cast<const uint64_t*>(h160);
        uint64_t k2 = *reinterpret_cast<const uint64_t*>(h160 + 8);

        k1 *= c1; k1 = (k1 << 31) | (k1 >> 33); k1 *= c2; h1 ^= k1;
        h1 = (h1 << 27) | (h1 >> 37); h1 += h2; h1 = h1 * 5 + 0x52dce729;

        k2 *= c2; k2 = (k2 << 33) | (k2 >> 31); k2 *= c1; h2 ^= k2;
        h2 = (h2 << 31) | (h2 >> 33); h2 += h1; h2 = h2 * 5 + 0x38495ab5;

        // Last 4 bytes (tail)
        uint64_t k1_tail = 0;
        k1_tail ^= uint64_t(h160[19]) << 24;
        k1_tail ^= uint64_t(h160[18]) << 16;
        k1_tail ^= uint64_t(h160[17]) << 8;
        k1_tail ^= uint64_t(h160[16]);
        k1_tail *= c1; k1_tail = (k1_tail << 31) | (k1_tail >> 33); k1_tail *= c2;
        h1 ^= k1_tail;

        // Finalization
        h1 ^= 20; h2 ^= 20;
        h1 += h2; h2 += h1;
        h1 = fmix64(h1); h2 = fmix64(h2);
        h1 += h2; h2 += h1;

        // Check all k hash positions
        for (uint32_t i = 0; i < num_hashes; i++) {
            uint64_t hash = h1 + i * h2;
            uint64_t bit_idx = hash % num_bits;
            uint64_t byte_idx = bit_idx / 8;
            uint32_t bit_offset = bit_idx % 8;

            if (!((data[byte_idx] >> bit_offset) & 1)) {
                return false;  // Definitely not in set
            }
        }
        return true;  // Probably in set
    }
};

static BloomFilter g_bloom_filter;
static std::atomic<uint64_t> g_bloom_checks{0};
static std::vector<collider::gpu::BloomHit> g_bloom_hits;
static std::mutex g_bloom_hits_mutex;
static std::function<void(const collider::gpu::BloomHit&)> g_bloom_hit_callback;
static std::function<void(const uint8_t*, const uint8_t*, uint8_t)> g_dp_callback;

// ============================================================================
// Global state (required by RCKangaroo's architecture)
// ============================================================================

static EcJMP g_EcJumps1[JMP_CNT];
static EcJMP g_EcJumps2[JMP_CNT];
static EcJMP g_EcJumps3[JMP_CNT];
static RCGpuKang* g_GpuKangs[MAX_GPU_CNT];
static int g_GpuCnt = 0;
static volatile bool g_Solved = false;
static volatile long g_ThrCnt = 0;

static EcInt g_Int_HalfRange;
static EcPoint g_Pnt_HalfRange;
static EcInt g_PrivKey;
static EcPoint g_PntToSolve;

static CriticalSection g_csAddPoints;
static u8* g_pPntList = nullptr;
static u8* g_pPntList2 = nullptr;
static volatile int g_PntIndex = 0;
static TFastBase g_db;
static u64 g_PntTotalOps = 0;
static u32 g_TotalErrors = 0;
static bool g_GenMode = false;

// ============================================================================
// AddPointsToList - Called by RCGpuKang::Execute() after kernel completes
// This is required by GpuKang.cpp (extern declaration at line 16)
// ============================================================================
void AddPointsToList(u32* data, int pnt_cnt, u64 ops_cnt) {
    g_csAddPoints.Enter();
    if (g_PntIndex + pnt_cnt >= MAX_CNT_LIST) {
        g_csAddPoints.Leave();
        std::cerr << "DPs buffer overflow, increase DP value!" << std::endl;
        return;
    }
    memcpy(g_pPntList + GPU_DP_SIZE * g_PntIndex, data, pnt_cnt * GPU_DP_SIZE);
    g_PntIndex = g_PntIndex + pnt_cnt;  // Avoid deprecated volatile compound assignment
    g_PntTotalOps += ops_cnt;
    g_csAddPoints.Leave();
}

// ============================================================================
// Namespace for theCollider integration
// ============================================================================

namespace collider {
namespace gpu {

// Thread procedure for GPU workers
#ifdef _WIN32
static u32 __stdcall kang_thr_proc(void* data) {
    RCGpuKang* Kang = (RCGpuKang*)data;
    Kang->Execute();
    InterlockedDecrement(&g_ThrCnt);
    return 0;
}
#else
static void* kang_thr_proc(void* data) {
    RCGpuKang* Kang = (RCGpuKang*)data;
    Kang->Execute();
    __sync_fetch_and_sub(&g_ThrCnt, 1);
    return nullptr;
}
#endif

// Collision detection using SOTA method
static bool Collision_SOTA(EcPoint& pnt, EcInt t, int TameType, EcInt w, int WildType, bool IsNeg) {
    Ec ec;
    if (IsNeg)
        t.Neg();
    if (TameType == TAME) {
        g_PrivKey = t;
        g_PrivKey.Sub(w);
        EcInt sv = g_PrivKey;
        g_PrivKey.Add(g_Int_HalfRange);
        EcPoint P = ec.MultiplyG(g_PrivKey);
        if (P.IsEqual(pnt))
            return true;
        g_PrivKey = sv;
        g_PrivKey.Neg();
        g_PrivKey.Add(g_Int_HalfRange);
        P = ec.MultiplyG(g_PrivKey);
        return P.IsEqual(pnt);
    } else {
        g_PrivKey = t;
        g_PrivKey.Sub(w);
        if (g_PrivKey.data[4] >> 63)
            g_PrivKey.Neg();
        g_PrivKey.ShiftRight(1);
        EcInt sv = g_PrivKey;
        g_PrivKey.Add(g_Int_HalfRange);
        EcPoint P = ec.MultiplyG(g_PrivKey);
        if (P.IsEqual(pnt))
            return true;
        g_PrivKey = sv;
        g_PrivKey.Neg();
        g_PrivKey.Add(g_Int_HalfRange);
        P = ec.MultiplyG(g_PrivKey);
        return P.IsEqual(pnt);
    }
}

#pragma pack(push, 1)
struct DBRec {
    u8 x[12];
    u8 d[22];
    u8 type;
};
#pragma pack(pop)

// Compute compressed public key from EC point (33 bytes: 02/03 + X)
static void compress_pubkey(const EcPoint& pnt, uint8_t* out33) {
    // Prefix: 02 if Y is even, 03 if Y is odd (check lowest bit of y.data[0])
    out33[0] = (pnt.y.data[0] & 1) ? 0x03 : 0x02;
    // X coordinate in big-endian
    for (int i = 0; i < 32; i++) {
        out33[1 + i] = ((uint8_t*)pnt.x.data)[31 - i];
    }
}

// Check a single DP against bloom filter
static void check_dp_bloom(const DBRec& nrec, Ec& ec) {
    if (!g_bloom_filter.loaded) return;

    // Extract distance from record
    EcInt dist;
    memset(dist.data, 0, sizeof(dist.data));
    memcpy(dist.data, nrec.d, sizeof(nrec.d));
    if (nrec.d[21] == 0xFF) memset(((u8*)dist.data) + 22, 0xFF, 18);

    // Compute the public key at this DP position
    EcPoint dp_pubkey;
    if (nrec.type == TAME) {
        // TAME: pubkey = G * (half_range + dist)
        EcInt pk = g_Int_HalfRange;
        pk.Add(dist);
        dp_pubkey = ec.MultiplyG(pk);
    } else {
        // WILD: pubkey = target_point + G * dist (or - G * dist)
        EcPoint dp = ec.MultiplyG(dist);
        dp_pubkey = ec.AddPoints(g_PntToSolve, dp);
    }

    // Compress public key and compute Hash160
    uint8_t compressed[33];
    compress_pubkey(dp_pubkey, compressed);

    uint8_t h160[20];
    cpu_hash160(compressed, 33, h160);

    g_bloom_checks++;

    // Check bloom filter
    if (g_bloom_filter.check(h160)) {
        // Potential hit! Record it
        collider::gpu::BloomHit hit;

        // Compute the private key for this position
        EcInt priv_key;
        if (nrec.type == TAME) {
            priv_key = g_Int_HalfRange;
            priv_key.Add(dist);
        } else {
            // For WILD, we need the actual private key which requires solving
            // For now, store the distance - would need full solve to get actual key
            priv_key = dist;
        }
        memcpy(hit.private_key.data(), priv_key.data, 32);
        memcpy(hit.hash160.data(), h160, 20);
        hit.ops_at_hit = g_PntTotalOps;

        // Convert H160 to hex for logging
        char h160_hex[41];
        for (int i = 0; i < 20; i++) {
            snprintf(h160_hex + i*2, 3, "%02x", h160[i]);
        }

        std::cout << "\n[BLOOM HIT] Potential match! H160=" << h160_hex << std::endl;

        // Store and callback
        {
            std::lock_guard<std::mutex> lock(g_bloom_hits_mutex);
            g_bloom_hits.push_back(hit);
        }

        if (g_bloom_hit_callback) {
            g_bloom_hit_callback(hit);
        }
    }
}

// Check new distinguished points for collisions
static void CheckNewPoints() {
    g_csAddPoints.Enter();
    if (!g_PntIndex) {
        g_csAddPoints.Leave();
        return;
    }

    int cnt = g_PntIndex;
    memcpy(g_pPntList2, g_pPntList, GPU_DP_SIZE * cnt);
    g_PntIndex = 0;
    g_csAddPoints.Leave();

    Ec ec;  // EC context for bloom checking

    for (int i = 0; i < cnt; i++) {
        DBRec nrec;
        u8* p = g_pPntList2 + i * GPU_DP_SIZE;
        memcpy(nrec.x, p, 12);
        memcpy(nrec.d, p + 16, 22);
        nrec.type = g_GenMode ? TAME : p[40];

        // Check bloom filter for this DP (opportunistic address scan)
        if (g_bloom_filter.loaded && !g_GenMode) {
            check_dp_bloom(nrec, ec);
        }

        DBRec* pref = (DBRec*)g_db.FindOrAddDataBlock((u8*)&nrec);

        // Export DP to pool via callback (if in pool mode)
        if (g_dp_callback && !g_GenMode) {
            uint8_t x_full[32] = {0};
            uint8_t d_full[32] = {0};
            // x is 12 bytes, d is 22 bytes in the DB record
            memcpy(x_full, nrec.x, 12);
            memcpy(d_full, nrec.d, 22);
            g_dp_callback(x_full, d_full, nrec.type);
        }

        if (g_GenMode)
            continue;
        if (pref) {
            // Restore first 3 bytes
            DBRec tmp_pref;
            memcpy(&tmp_pref, &nrec, 3);
            memcpy(((u8*)&tmp_pref) + 3, pref, sizeof(DBRec) - 3);
            pref = &tmp_pref;

            if (pref->type == nrec.type) {
                if (pref->type == TAME)
                    continue;
                if (*(u64*)pref->d == *(u64*)nrec.d)
                    continue;
            }

            EcInt w, t;
            int TameType, WildType;
            if (pref->type != TAME) {
                memcpy(w.data, pref->d, sizeof(pref->d));
                if (pref->d[21] == 0xFF) memset(((u8*)w.data) + 22, 0xFF, 18);
                memcpy(t.data, nrec.d, sizeof(nrec.d));
                if (nrec.d[21] == 0xFF) memset(((u8*)t.data) + 22, 0xFF, 18);
                TameType = nrec.type;
                WildType = pref->type;
            } else {
                memcpy(w.data, nrec.d, sizeof(nrec.d));
                if (nrec.d[21] == 0xFF) memset(((u8*)w.data) + 22, 0xFF, 18);
                memcpy(t.data, pref->d, sizeof(pref->d));
                if (pref->d[21] == 0xFF) memset(((u8*)t.data) + 22, 0xFF, 18);
                TameType = TAME;
                WildType = nrec.type;
            }

            bool res = Collision_SOTA(g_PntToSolve, t, TameType, w, WildType, false) ||
                       Collision_SOTA(g_PntToSolve, t, TameType, w, WildType, true);
            if (!res) {
                bool w12 = ((pref->type == WILD1) && (nrec.type == WILD2)) ||
                           ((pref->type == WILD2) && (nrec.type == WILD1));
                if (!w12) {
                    g_TotalErrors++;
                }
                continue;
            }
            g_Solved = true;
            break;
        }
    }
}

struct RCKangarooManager::Impl {
    std::vector<int> gpu_ids;
    EcPoint target_pubkey;
    EcInt start_offset;
    bool pubkey_set = false;
    bool start_set = false;
    std::string tames_file;
    int current_speed = 0;
    bool initialized = false;

    ~Impl() {
        cleanup();
    }

    void cleanup() {
        if (g_pPntList) {
            free(g_pPntList);
            g_pPntList = nullptr;
        }
        if (g_pPntList2) {
            free(g_pPntList2);
            g_pPntList2 = nullptr;
        }
        for (int i = 0; i < g_GpuCnt; i++) {
            if (g_GpuKangs[i]) {
                delete g_GpuKangs[i];
                g_GpuKangs[i] = nullptr;
            }
        }
        g_GpuCnt = 0;
        g_db.Clear();
        if (initialized) {
            DeInitEc();
            initialized = false;
        }
    }
};

RCKangarooManager::RCKangarooManager() : impl_(new Impl()) {
}

RCKangarooManager::~RCKangarooManager() {
    delete impl_;
}

int RCKangarooManager::init(const std::vector<int>& gpu_ids) {
    // Initialize EC library
    InitEc();
    impl_->initialized = true;

    // Detect GPUs
    int gcnt = 0;
    cudaGetDeviceCount(&gcnt);
    if (gcnt > MAX_GPU_CNT)
        gcnt = MAX_GPU_CNT;

    if (!gcnt) {
        std::cerr << "No CUDA devices found" << std::endl;
        return 0;
    }

    int drv, rt;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    std::cout << "CUDA driver/runtime: " << drv/1000 << "." << (drv%100)/10
              << "/" << rt/1000 << "." << (rt%100)/10 << std::endl;

    g_GpuCnt = 0;

    for (int i = 0; i < gcnt; i++) {
        // Check if this GPU should be used
        if (!gpu_ids.empty()) {
            bool found = false;
            for (int id : gpu_ids) {
                if (id == i) { found = true; break; }
            }
            if (!found) continue;
        }

        cudaError_t status = cudaSetDevice(i);
        if (status != cudaSuccess) {
            std::cerr << "cudaSetDevice for GPU " << i << " failed" << std::endl;
            continue;
        }

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "GPU " << i << ": " << prop.name
                  << ", " << (prop.totalGlobalMem / (1024*1024*1024.0)) << " GB"
                  << ", " << prop.multiProcessorCount << " SMs"
                  << ", cap " << prop.major << "." << prop.minor
                  << ", L2: " << (prop.l2CacheSize/1024) << " KB" << std::endl;

        if (prop.major < 6) {
            std::cout << "GPU " << i << " not supported (need compute 6.0+), skip" << std::endl;
            continue;
        }

        cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

        g_GpuKangs[g_GpuCnt] = new RCGpuKang();
        g_GpuKangs[g_GpuCnt]->CudaIndex = i;
        g_GpuKangs[g_GpuCnt]->persistingL2CacheMaxSize = prop.persistingL2CacheMaxSize;
        g_GpuKangs[g_GpuCnt]->mpCnt = prop.multiProcessorCount;
        g_GpuKangs[g_GpuCnt]->IsOldGpu = prop.l2CacheSize < 16 * 1024 * 1024;
        g_GpuCnt++;
    }

    std::cout << "Total GPUs initialized: " << g_GpuCnt << std::endl;

    // Allocate DP buffers
    g_pPntList = (u8*)malloc(MAX_CNT_LIST * GPU_DP_SIZE);
    g_pPntList2 = (u8*)malloc(MAX_CNT_LIST * GPU_DP_SIZE);

    impl_->gpu_ids = gpu_ids;
    return g_GpuCnt;
}

int RCKangarooManager::num_gpus() const {
    return g_GpuCnt;
}

bool RCKangarooManager::set_target_pubkey(const std::string& compressed_hex) {
    if (!impl_->target_pubkey.SetHexStr(compressed_hex.c_str())) {
        std::cerr << "Invalid public key format" << std::endl;
        return false;
    }
    impl_->pubkey_set = true;
    return true;
}

bool RCKangarooManager::set_target_pubkey(const std::array<uint64_t, 4>& x,
                                           const std::array<uint64_t, 4>& y) {
    memcpy(impl_->target_pubkey.x.data, x.data(), 32);
    memcpy(impl_->target_pubkey.y.data, y.data(), 32);
    impl_->pubkey_set = true;
    return true;
}

void RCKangarooManager::set_start_offset(const std::string& start_hex) {
    impl_->start_offset.SetHexStr(start_hex.c_str());
    impl_->start_set = true;
}

bool RCKangarooManager::load_tames(const std::string& filename) {
    impl_->tames_file = filename;
    return g_db.LoadFromFile(const_cast<char*>(filename.c_str()));
}

bool RCKangarooManager::generate_tames(const std::string& filename, double max_ops) {
    impl_->tames_file = filename;
    Ec ec;

    if (g_GpuCnt == 0) {
        std::cerr << "No GPUs initialized for tames generation" << std::endl;
        return false;
    }

    int Range = range_bits;
    int DP = dp_bits;
    g_GenMode = true;  // Enable tames generation mode

    std::cout << "\n=== TAMES GENERATION MODE ===" << std::endl;
    std::cout << "Range: " << Range << " bits, DP: " << DP << std::endl;
    std::cout << "Output file: " << filename << std::endl;

    // Calculate expected operations
    double ops = 1.15 * pow(2.0, Range / 2.0);
    double dp_val = (double)(1ull << DP);
    double max_total_ops = max_ops > 0 ? max_ops * ops : ops * 0.5;  // Default to 0.5x expected ops

    std::cout << "Expected ops: 2^" << log2(ops) << std::endl;
    std::cout << "Max ops for tames: 2^" << log2(max_total_ops) << std::endl;

    // Initialize state
    g_PntTotalOps = 0;
    g_PntIndex = 0;
    g_TotalErrors = 0;
    g_Solved = false;

    // Use a fixed seed for reproducible tames generation
    // This allows tames files to be compatible across runs
    SetRndSeed(0);

    // Prepare jump tables (same as in solve, for consistency)
    EcInt minjump, t;
    minjump.Set(1);
    minjump.ShiftLeft(Range / 2 + 3);
    for (int i = 0; i < JMP_CNT; i++) {
        g_EcJumps1[i].dist = minjump;
        t.RndMax(minjump);
        g_EcJumps1[i].dist.Add(t);
        g_EcJumps1[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE;  // Must be even
        g_EcJumps1[i].p = ec.MultiplyG(g_EcJumps1[i].dist);
    }

    minjump.Set(1);
    minjump.ShiftLeft(Range - 10);
    for (int i = 0; i < JMP_CNT; i++) {
        g_EcJumps2[i].dist = minjump;
        t.RndMax(minjump);
        g_EcJumps2[i].dist.Add(t);
        g_EcJumps2[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE;
        g_EcJumps2[i].p = ec.MultiplyG(g_EcJumps2[i].dist);
    }

    minjump.Set(1);
    minjump.ShiftLeft(Range - 10 - 2);
    for (int i = 0; i < JMP_CNT; i++) {
        g_EcJumps3[i].dist = minjump;
        t.RndMax(minjump);
        g_EcJumps3[i].dist.Add(t);
        g_EcJumps3[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE;
        g_EcJumps3[i].p = ec.MultiplyG(g_EcJumps3[i].dist);
    }

    // Restore random seed for randomized starting points
#ifdef _WIN32
    SetRndSeed(GetTickCount64());
#else
    SetRndSeed(std::chrono::steady_clock::now().time_since_epoch().count());
#endif

    // Set half range
    g_Int_HalfRange.Set(1);
    g_Int_HalfRange.ShiftLeft(Range - 1);
    g_Pnt_HalfRange = ec.MultiplyG(g_Int_HalfRange);

    // For tames generation, we use the generator point G as the "target"
    // This creates tames that can be used for any public key in this range
    EcPoint PntForTames;
    PntForTames.x.SetZero();
    PntForTames.y.SetZero();
    // Use a dummy point - tames are generated relative to halfrange
    g_PntToSolve = g_Pnt_HalfRange;  // Use half range point as reference

    // Prepare GPUs for tames generation
    for (int i = 0; i < g_GpuCnt; i++) {
        if (!g_GpuKangs[i]->Prepare(g_Pnt_HalfRange, Range, DP, g_EcJumps1, g_EcJumps2, g_EcJumps3)) {
            g_GpuKangs[i]->Failed = true;
            std::cerr << "GPU " << g_GpuKangs[i]->CudaIndex << " Prepare failed for tames generation" << std::endl;
        }
    }

    auto start_time = std::chrono::steady_clock::now();
    std::cout << "Starting tames generation on " << g_GpuCnt << " GPUs..." << std::endl;

    // Launch worker threads
#ifdef _WIN32
    HANDLE thr_handles[MAX_GPU_CNT];
    u32 ThreadID;
    g_ThrCnt = g_GpuCnt;
    for (int i = 0; i < g_GpuCnt; i++) {
        thr_handles[i] = (HANDLE)_beginthreadex(NULL, 0, kang_thr_proc,
                                                 (void*)g_GpuKangs[i], 0, &ThreadID);
    }
#else
    pthread_t thr_handles[MAX_GPU_CNT];
    g_ThrCnt = g_GpuCnt;
    for (int i = 0; i < g_GpuCnt; i++) {
        pthread_create(&thr_handles[i], NULL, kang_thr_proc, (void*)g_GpuKangs[i]);
    }
#endif

    // Main loop - collect tames until we hit the operations limit
    auto last_stats = std::chrono::steady_clock::now();
    while (!stop_flag.load()) {
        // In gen mode, CheckNewPoints just adds to database without looking for collisions
        CheckNewPoints();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        // Check if we've reached the operations limit
        if (g_PntTotalOps >= static_cast<u64>(max_total_ops)) {
            std::cout << "\nOperations limit reached, stopping..." << std::endl;
            break;
        }

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_stats).count() >= 10) {
            int speed = 0;
            for (int i = 0; i < g_GpuCnt; i++) {
                if (!g_GpuKangs[i]->Failed) {
                    speed += g_GpuKangs[i]->GetStatsSpeed();
                }
            }

            double progress = (static_cast<double>(g_PntTotalOps) / max_total_ops) * 100.0;
            std::cout << "GEN: Speed: " << speed << " MKeys/s, DPs: " << g_db.GetBlockCnt()
                      << ", Ops: 2^" << std::fixed << std::setprecision(2) << log2(static_cast<double>(g_PntTotalOps))
                      << ", Progress: " << std::setprecision(1) << progress << "%" << std::endl;
            last_stats = now;
        }
    }

    // Stop workers
    for (int i = 0; i < g_GpuCnt; i++)
        g_GpuKangs[i]->Stop();
    while (g_ThrCnt)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Close thread handles
#ifdef _WIN32
    for (int i = 0; i < g_GpuCnt; i++)
        CloseHandle(thr_handles[i]);
#else
    for (int i = 0; i < g_GpuCnt; i++)
        pthread_join(thr_handles[i], NULL);
#endif

    auto end_time = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end_time - start_time).count();

    // Save tames to file
    std::cout << "\nSaving tames to " << filename << "..." << std::endl;
    g_db.Header[0] = static_cast<u8>(Range);  // Store range in header for compatibility check

    // Need to cast away const for the C-style API
    char* fn_cstr = const_cast<char*>(filename.c_str());
    bool saved = g_db.SaveToFile(fn_cstr);

    if (saved) {
        std::cout << "=== TAMES GENERATION COMPLETE ===" << std::endl;
        std::cout << "Tames saved: " << g_db.GetBlockCnt() << std::endl;
        std::cout << "Total ops: 2^" << log2(static_cast<double>(g_PntTotalOps)) << std::endl;
        std::cout << "Time: " << std::setprecision(1) << elapsed << " seconds" << std::endl;
        std::cout << "File: " << filename << std::endl;
    } else {
        std::cerr << "ERROR: Failed to save tames to " << filename << std::endl;
    }

    g_db.Clear();
    g_GenMode = false;  // Reset generation mode
    return saved;
}

RCKangarooResult RCKangarooManager::solve() {
    RCKangarooResult result = {};
    Ec ec;

    if (!impl_->pubkey_set) {
        std::cerr << "Target public key not set" << std::endl;
        return result;
    }

    if (g_GpuCnt == 0) {
        std::cerr << "No GPUs initialized" << std::endl;
        return result;
    }

    int Range = range_bits;
    int DP = dp_bits;
    g_GenMode = false;

    std::cout << "\nSolving: Range " << Range << " bits, DP " << DP << std::endl;
    double ops = 1.15 * pow(2.0, Range / 2.0);
    double dp_val = (double)(1ull << DP);
    std::cout << "SOTA method, estimated ops: 2^" << log2(ops) << std::endl;

    // Prepare target point
    EcPoint PntToSolve = impl_->target_pubkey;
    if (impl_->start_set && !impl_->start_offset.IsZero()) {
        EcPoint PntOfs = ec.MultiplyG(impl_->start_offset);
        PntOfs.y.NegModP();
        PntToSolve = ec.AddPoints(PntToSolve, PntOfs);
    }
    g_PntToSolve = PntToSolve;

    // Initialize state
    g_PntTotalOps = 0;
    g_PntIndex = 0;
    g_TotalErrors = 0;

    // Reset bloom filter stats
    g_bloom_checks = 0;
    {
        std::lock_guard<std::mutex> lock(g_bloom_hits_mutex);
        g_bloom_hits.clear();
    }
    g_bloom_hit_callback = bloom_hit_callback;
    g_dp_callback = dp_callback;

    if (bloom_enabled && g_bloom_filter.loaded) {
        std::cout << "[Bloom] Opportunistic address checking enabled" << std::endl;
    }
    g_Solved = false;

    // Prepare jump tables
    SetRndSeed(0);
    EcInt minjump, t;
    minjump.Set(1);
    minjump.ShiftLeft(Range / 2 + 3);
    for (int i = 0; i < JMP_CNT; i++) {
        g_EcJumps1[i].dist = minjump;
        t.RndMax(minjump);
        g_EcJumps1[i].dist.Add(t);
        g_EcJumps1[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE;
        g_EcJumps1[i].p = ec.MultiplyG(g_EcJumps1[i].dist);
    }

    minjump.Set(1);
    minjump.ShiftLeft(Range - 10);
    for (int i = 0; i < JMP_CNT; i++) {
        g_EcJumps2[i].dist = minjump;
        t.RndMax(minjump);
        g_EcJumps2[i].dist.Add(t);
        g_EcJumps2[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE;
        g_EcJumps2[i].p = ec.MultiplyG(g_EcJumps2[i].dist);
    }

    minjump.Set(1);
    minjump.ShiftLeft(Range - 10 - 2);
    for (int i = 0; i < JMP_CNT; i++) {
        g_EcJumps3[i].dist = minjump;
        t.RndMax(minjump);
        g_EcJumps3[i].dist.Add(t);
        g_EcJumps3[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE;
        g_EcJumps3[i].p = ec.MultiplyG(g_EcJumps3[i].dist);
    }

#ifdef _WIN32
    SetRndSeed(GetTickCount64());
#else
    SetRndSeed(std::chrono::steady_clock::now().time_since_epoch().count());
#endif

    // Set half range
    g_Int_HalfRange.Set(1);
    g_Int_HalfRange.ShiftLeft(Range - 1);
    g_Pnt_HalfRange = ec.MultiplyG(g_Int_HalfRange);

    // Prepare GPUs
    for (int i = 0; i < g_GpuCnt; i++) {
        if (!g_GpuKangs[i]->Prepare(PntToSolve, Range, DP, g_EcJumps1, g_EcJumps2, g_EcJumps3)) {
            g_GpuKangs[i]->Failed = true;
            std::cerr << "GPU " << g_GpuKangs[i]->CudaIndex << " Prepare failed" << std::endl;
        }
    }

    auto start_time = std::chrono::steady_clock::now();
    std::cout << "GPUs started..." << std::endl;

    // Launch worker threads
#ifdef _WIN32
    HANDLE thr_handles[MAX_GPU_CNT];
    u32 ThreadID;
    g_ThrCnt = g_GpuCnt;
    for (int i = 0; i < g_GpuCnt; i++) {
        thr_handles[i] = (HANDLE)_beginthreadex(NULL, 0, kang_thr_proc,
                                                 (void*)g_GpuKangs[i], 0, &ThreadID);
    }
#else
    pthread_t thr_handles[MAX_GPU_CNT];
    g_ThrCnt = g_GpuCnt;
    for (int i = 0; i < g_GpuCnt; i++) {
        pthread_create(&thr_handles[i], NULL, kang_thr_proc, (void*)g_GpuKangs[i]);
    }
#endif

    // Main loop
    auto last_stats = std::chrono::steady_clock::now();
    while (!g_Solved && !stop_flag.load()) {
        CheckNewPoints();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_stats).count() >= 10) {
            int speed = 0;
            for (int i = 0; i < g_GpuCnt; i++) {
                // Only query speed from GPUs that haven't failed
                if (!g_GpuKangs[i]->Failed) {
                    speed += g_GpuKangs[i]->GetStatsSpeed();
                }
            }
            impl_->current_speed = speed;

            u64 est_dps_cnt = (u64)(ops / dp_val);
            // Note: Primary progress display is handled by the progress_callback
            // This is supplementary debug output
            if (!progress_callback) {
                // Format speed appropriately (GKeys/s if >= 1000 MKeys/s)
                std::string speed_str;
                if (speed >= 1000) {
                    speed_str = std::to_string(speed / 1000) + "." + std::to_string((speed % 1000) / 100) + " GKeys/s";
                } else {
                    speed_str = std::to_string(speed) + " MKeys/s";
                }
                std::cout << "Speed: " << speed_str << ", Err: " << g_TotalErrors
                          << ", DPs: " << g_db.GetBlockCnt() << "/" << est_dps_cnt
                          << std::endl;
            }

            if (progress_callback) {
                if (!progress_callback(g_PntTotalOps, g_db.GetBlockCnt(), speed)) {
                    stop_flag.store(true);
                }
            }
            last_stats = now;
        }
    }

    // Stop workers
    for (int i = 0; i < g_GpuCnt; i++)
        g_GpuKangs[i]->Stop();
    while (g_ThrCnt)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Close thread handles
#ifdef _WIN32
    for (int i = 0; i < g_GpuCnt; i++)
        CloseHandle(thr_handles[i]);
#else
    for (int i = 0; i < g_GpuCnt; i++)
        pthread_join(thr_handles[i], NULL);
#endif

    auto end_time = std::chrono::steady_clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
    result.total_ops = g_PntTotalOps;
    result.dp_count = g_db.GetBlockCnt();
    result.error_count = g_TotalErrors;
    result.k_value = (double)g_PntTotalOps / pow(2.0, Range / 2.0);

    // Copy bloom filter results
    result.bloom_checks = g_bloom_checks.load();
    {
        std::lock_guard<std::mutex> lock(g_bloom_hits_mutex);
        result.bloom_hits = g_bloom_hits;
    }

    if (bloom_enabled && result.bloom_checks > 0) {
        std::cout << "[Bloom] Total checks: " << result.bloom_checks
                  << ", Hits: " << result.bloom_hits.size() << std::endl;
    }

    if (g_Solved) {
        // Apply start offset
        if (impl_->start_set) {
            g_PrivKey.Add(impl_->start_offset);
        }

        // Verify solution
        EcPoint verify = ec.MultiplyG(g_PrivKey);
        if (verify.IsEqual(impl_->target_pubkey)) {
            result.found = true;
            memcpy(result.private_key.data(), g_PrivKey.data, 32);

            char hex[100];
            g_PrivKey.GetHexStr(hex);
            std::cout << "\n+============================================================+\n"
                      << "|                      PUZZLE SOLVED!                        |\n"
                      << "+============================================================+\n"
                      << "PRIVATE KEY: " << hex << "\n"
                      << "K value: " << result.k_value << std::endl;
        } else {
            std::cerr << "FATAL: Collision found but key verification failed!" << std::endl;
        }
    }

    g_db.Clear();
    return result;
}

double RCKangarooManager::benchmark(int num_points) {
    benchmark_mode = true;
    Ec ec;
    double total_k = 0.0;
    int solved = 0;

    for (int p = 0; p < num_points && !stop_flag.load(); p++) {
        // Generate random key
        EcInt pk;
        pk.RndBits(range_bits);
        EcPoint pnt = ec.MultiplyG(pk);

        // Set as target
        memcpy(impl_->target_pubkey.x.data, pnt.x.data, 40);
        memcpy(impl_->target_pubkey.y.data, pnt.y.data, 40);
        impl_->pubkey_set = true;
        impl_->start_set = false;

        auto result = solve();
        if (result.found) {
            if (memcmp(result.private_key.data(), pk.data, 32) == 0) {
                solved++;
                total_k += result.k_value;
                std::cout << "Benchmark " << (p+1) << "/" << num_points
                          << ": K=" << result.k_value << std::endl;
            } else {
                std::cerr << "Benchmark FAILED: found wrong key!" << std::endl;
            }
        }
    }

    benchmark_mode = false;
    return solved > 0 ? total_k / solved : 0.0;
}

int RCKangarooManager::get_speed() const {
    return impl_->current_speed;
}

bool RCKangarooManager::load_bloom_filter(const std::string& filename) {
    if (g_bloom_filter.load(filename)) {
        bloom_enabled = true;
        bloom_file = filename;
        g_bloom_hit_callback = bloom_hit_callback;
        return true;
    }
    return false;
}

uint64_t RCKangarooManager::get_bloom_checks() const {
    return g_bloom_checks.load();
}

std::string private_key_to_hex(const std::array<uint64_t, 4>& key) {
    char hex[65];
    snprintf(hex, sizeof(hex), "%016llx%016llx%016llx%016llx",
             (unsigned long long)key[3], (unsigned long long)key[2],
             (unsigned long long)key[1], (unsigned long long)key[0]);
    // Remove leading zeros
    char* p = hex;
    while (*p == '0' && *(p+1) != '\0') p++;
    return std::string(p);
}

bool hex_to_private_key(const std::string& hex, std::array<uint64_t, 4>& key) {
    if (hex.length() > 64)
        return false;

    std::string padded = std::string(64 - hex.length(), '0') + hex;
    for (int i = 0; i < 4; i++) {
        key[3-i] = strtoull(padded.substr(i*16, 16).c_str(), nullptr, 16);
    }
    return true;
}

}  // namespace gpu
}  // namespace collider
