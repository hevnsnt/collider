/**
 * RCKangaroo Wrapper Implementation for theCollider
 *
 * Integrates RetiredCoder's RCKangaroo (GPLv3) as the Kangaroo solver backend.
 *
 * Original software: (c) 2024, RetiredCoder (RC)
 * https://github.com/RetiredC/RCKangaroo
 */

#include "rckangaroo_wrapper.hpp"
#include "../core/byte_codec.hpp"
// Q-T1.1 inversion (2026-05-17): formerly #include
// "../runtime/runtime_control.hpp" purely for the kMaxGpus constant used
// in the static_assert below. The constant is now owned by the gpu layer
// (gpu_caps.hpp::kMaxDispatchableGpus); runtime_control.hpp pulls it back
// in and binds RuntimeControlState::kMaxGpus to it via a static_assert,
// so the two stay locked while this file no longer reaches up into the
// runtime layer for an integer.
#include "gpu_caps.hpp"
#include "hash_rounds.cuh"

#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>
#include <thread>
#include <chrono>
#include <mutex>
#include <iomanip>
#include <filesystem>   // T0.2: atomic tmp+rename for save_herd_state
#include <system_error> // std::error_code on the rename path

#ifdef _WIN32
    // _commit() is the Windows equivalent of POSIX fsync(): flushes the
    // OS write cache for a given fd. Only used in save_herd_state() so
    // a power loss between tmp-write and rename does not leave behind a
    // .kang.tmp whose tail is in the OS cache but not yet on disk.
    #include <io.h>
#else
    #include <unistd.h>
#endif

// RCKangaroo headers
#include "defs.h"
#include "Ec.h"
#include "GpuKang.h"
#include "utils.h"

#include "cuda_runtime.h"

#ifdef COLLIDER_PRO
// Opportunistic bloom address checking lives entirely in this Pro-only
// translation unit. The wrapper calls the collider::gpu::bloom hook below
// only under COLLIDER_PRO, so a Free build links zero bloom symbols and
// ships zero bloom source (the file is excluded from the Free repo via
// scripts/sync-to-free.sh and from the Free build via CMake).
#include "rckangaroo_bloom.hpp"
#endif

// T4.8: keep the RCKangaroo upper bound aligned with the runtime's
// per-GPU phase-array bound. If a future edit nudges either constant,
// the build fails here rather than silently leaving one of the two
// array sizes wrong.
static_assert(MAX_GPU_CNT == ::collider::gpu::kMaxDispatchableGpus,
              "MAX_GPU_CNT (third_party/RCKangaroo/defs.h) must equal "
              "gpu::kMaxDispatchableGpus (src/gpu/gpu_caps.hpp). "
              "RuntimeControlState::kMaxGpus is bound to the same constant; "
              "update third_party/RCKangaroo/defs.h alongside gpu_caps.hpp.");

// Global variables required by RCKangaroo (defined in RCKangaroo.cpp but we use our wrapper)
bool gGenMode = false;      // Tames generation mode
u32 gTotalErrors = 0;       // Error counter

// ============================================================================
// CPU-side SHA256. Shared host helper used by compute_config_hash() for the
// herd-state on-disk fingerprint (NOT a bloom-only helper; the bloom
// feature carries its own self-contained copy in rckangaroo_bloom.cu so it
// stays fully removable at the source level for Free builds).
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

// rotr now lives in hash_rounds.cuh (host+device).
inline uint32_t rotr32(uint32_t x, int n) {
    return collider::gpu::sha256::rotr(x, n);
}

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

// Bloom filter checking (CPU SHA256/RIPEMD160 hash160 derivation + bloom
// membership) lives entirely in the Pro-only rckangaroo_bloom.cu. The
// statics that backed it (s_bloom_filter, s_bloom_checks, s_bloom_hits,
// s_bloom_hits_mutex, s_bloom_hit_callback) moved there; this file only
// reaches the feature through the collider::gpu::bloom hook under
// #ifdef COLLIDER_PRO. The DP-export callback below is NOT a bloom symbol
// and stays here (pool mode uses it to submit DPs to the server).
static std::function<void(const uint8_t*, const uint8_t*, uint8_t)> s_dp_callback;

// ============================================================================
// Process-singleton state.
// RCKangaroo's third_party/RCKangaroo/GpuKang.cpp declares
// `void AddPointsToList(u32*, int, u64)` as an extern free function at
// line 16 and calls it from the kernel-completion path at line 472 with
// no `void* user_data` to thread instance state through. Our wrapper
// implementation of AddPointsToList therefore has to reach process-
// scope state to do anything useful. Encapsulating this into a per-
// instance Impl member would require either:
//   (a) forking RCKangaroo to add a context pointer to its API, or
//   (b) routing through a static "active instance" pointer that is
//       just a global by another name.
// We instead enforce the implicit singleton contract via
// g_active_instance_ + an atomic check in RCKangarooManager's
// constructor: a second instance fails loudly rather than silently
// stomping the first. The g_X globals below are renamed s_X to make
// their internal-linkage scope clearer to readers; they remain file-
// scope because that is the only way to satisfy GpuKang.cpp's link
// surface.
// audit finding: process-globals are a known hazard pattern;
// the singleton guard converts that hazard into a deterministic crash
// on misuse rather than data corruption. Full encapsulation requires
// patching third_party/RCKangaroo to thread a context pointer through
// AddPointsToList -- a separate effort once the upstream patches are
// vendored.
// ============================================================================

static EcJMP s_EcJumps1[JMP_CNT];
static EcJMP s_EcJumps2[JMP_CNT];
static EcJMP s_EcJumps3[JMP_CNT];
static RCGpuKang* s_GpuKangs[MAX_GPU_CNT];
static int s_GpuCnt = 0;
static volatile bool s_Solved = false;
static volatile long s_ThrCnt = 0;

static EcInt s_Int_HalfRange;
static EcPoint s_Pnt_HalfRange;
static EcInt s_PrivKey;
static EcPoint s_PntToSolve;

static CriticalSection s_csAddPoints;
static u8* s_pPntList = nullptr;
static u8* s_pPntList2 = nullptr;
static volatile int s_PntIndex = 0;
static TFastBase s_db;
static u64 s_PntTotalOps = 0;
static u32 s_TotalErrors = 0;
static bool s_GenMode = false;

// Active-instance tracker. NULL when no RCKangarooManager exists; non-
// null only between construct/destroy of the single permitted instance.
// Stored as void* to dodge forward-declaration namespace resolution
// (RCKangarooManager lives in collider::gpu, declared further below);
// we only need atomic compare-exchange semantics, not pointer
// dereferences.
static std::atomic<void*> g_active_rckangaroo{nullptr};

// ============================================================================
// AddPointsToList - Called by RCGpuKang::Execute() after kernel completes
// This is required by GpuKang.cpp (extern declaration at line 16)
// ============================================================================
void AddPointsToList(u32* data, int pnt_cnt, u64 ops_cnt) {
    s_csAddPoints.Enter();
    if (s_PntIndex + pnt_cnt >= MAX_CNT_LIST) {
        s_csAddPoints.Leave();
        std::cerr << "DPs buffer overflow, increase DP value!" << std::endl;
        return;
    }
    memcpy(s_pPntList + GPU_DP_SIZE * s_PntIndex, data, pnt_cnt * GPU_DP_SIZE);
    s_PntIndex = s_PntIndex + pnt_cnt;  // Avoid deprecated volatile compound assignment
    s_PntTotalOps += ops_cnt;
    s_csAddPoints.Leave();
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
    InterlockedDecrement(&s_ThrCnt);
    return 0;
}
#else
static void* kang_thr_proc(void* data) {
    RCGpuKang* Kang = (RCGpuKang*)data;
    Kang->Execute();
    __sync_fetch_and_sub(&s_ThrCnt, 1);
    return nullptr;
}
#endif

// Collision detection using SOTA method
static bool Collision_SOTA(EcPoint& pnt, EcInt t, int TameType, EcInt w, int WildType, bool IsNeg) {
    Ec ec;
    if (IsNeg)
        t.Neg();
    if (TameType == TAME) {
        s_PrivKey = t;
        s_PrivKey.Sub(w);
        EcInt sv = s_PrivKey;
        s_PrivKey.Add(s_Int_HalfRange);
        EcPoint P = ec.MultiplyG(s_PrivKey);
        if (P.IsEqual(pnt))
            return true;
        s_PrivKey = sv;
        s_PrivKey.Neg();
        s_PrivKey.Add(s_Int_HalfRange);
        P = ec.MultiplyG(s_PrivKey);
        return P.IsEqual(pnt);
    } else {
        s_PrivKey = t;
        s_PrivKey.Sub(w);
        if (s_PrivKey.data[4] >> 63)
            s_PrivKey.Neg();
        s_PrivKey.ShiftRight(1);
        EcInt sv = s_PrivKey;
        s_PrivKey.Add(s_Int_HalfRange);
        EcPoint P = ec.MultiplyG(s_PrivKey);
        if (P.IsEqual(pnt))
            return true;
        s_PrivKey = sv;
        s_PrivKey.Neg();
        s_PrivKey.Add(s_Int_HalfRange);
        P = ec.MultiplyG(s_PrivKey);
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

// Check new distinguished points for collisions
static void CheckNewPoints() {
    s_csAddPoints.Enter();
    if (!s_PntIndex) {
        s_csAddPoints.Leave();
        return;
    }

    int cnt = s_PntIndex;
    memcpy(s_pPntList2, s_pPntList, GPU_DP_SIZE * cnt);
    s_PntIndex = 0;
    s_csAddPoints.Leave();

#ifdef COLLIDER_PRO
    // EC context reused across this batch's opportunistic bloom probes.
    // Pro-only: with COLLIDER_PRO undefined there is no bloom hook call so
    // the context (and the probe) compile out entirely.
    Ec ec;
#endif

    for (int i = 0; i < cnt; i++) {
        DBRec nrec;
        u8* p = s_pPntList2 + i * GPU_DP_SIZE;
        // GPU_DP_SIZE=64 layout: 0-15 x LS 128b, 16-31 x MS 128b, 32-53 d, 56 type
        memcpy(nrec.x, p, 12);
        memcpy(nrec.d, p + 32, 22);
        nrec.type = s_GenMode ? TAME : p[56];

#ifdef COLLIDER_PRO
        // Opportunistic address scan: probe this DP's candidate address
        // against the loaded bloom filter (Pro only). All bloom logic lives
        // in rckangaroo_bloom.cu; here we only extract the distance and hand
        // it plus the run's half-range / target point to the hook.
        if (collider::gpu::bloom::is_loaded() && !s_GenMode) {
            EcInt dist;
            memset(dist.data, 0, sizeof(dist.data));
            memcpy(dist.data, nrec.d, sizeof(nrec.d));
            if (nrec.d[21] == 0xFF) memset(((u8*)dist.data) + 22, 0xFF, 18);
            collider::gpu::bloom::probe_dp(dist, nrec.type == TAME,
                                           s_Int_HalfRange, s_PntToSolve, ec,
                                           s_PntTotalOps);
        }
#endif

        DBRec* pref = (DBRec*)s_db.FindOrAddDataBlock((u8*)&nrec);

        // Export DP to pool via callback (if in pool mode)
        if (s_dp_callback && !s_GenMode) {
            // x: GPU stores 4 u64s in little-endian; reverse to big-endian so
            // the server sees leading zeros at byte 0 (secp256k1 x is < 2^256).
            uint8_t x_be[32];
            for (int j = 0; j < 32; j++) x_be[j] = p[31 - j];

            // d: GPU stores 22-byte two's complement in little-endian (nrec.d[0]
            // = LSByte, nrec.d[21] = MSByte / sign bit).  The wire protocol and
            // server both expect a 32-byte big-endian field where the 22-byte
            // value sits in the LOW (rightmost) 22 bytes and the HIGH 10 bytes
            // are zero (d < 2^176 == _D_MOD_22).  Reverse to achieve this layout.
            uint8_t d_be[32] = {0};
            for (int j = 0; j < 22; j++) d_be[10 + j] = nrec.d[21 - j];

            // Pool server only accepts types 0 (tame) and 1 (wild).
            // RCKangaroo uses type 2 (WILD2) for kangaroos starting from -PntA.
            // The server's type-1 math check already covers both wild branches:
            // branch-1 checks (PntA+d*G).x==X, branch-2 checks (PntA-d*G).x==X.
            // Since WILD2 walks from -PntA, its DP satisfies (-PntA+d*G).x==X
            // which equals (PntA-d*G).x==X, matching branch-2. Map to 1.
            const uint8_t pool_type = (nrec.type == 2) ? 1 : nrec.type;
            s_dp_callback(x_be, d_be, pool_type);
            // Pool mode: the server owns collision detection.
            // Skip the local FindOrAddDataBlock / s_Solved path so the
            // solve loop runs indefinitely rather than exiting on the
            // first internal collision (which happens within milliseconds
            // for small puzzles like 41-bit with 2M kangaroos).
            continue;
        }

        if (s_GenMode)
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

            bool res = Collision_SOTA(s_PntToSolve, t, TameType, w, WildType, false) ||
                       Collision_SOTA(s_PntToSolve, t, TameType, w, WildType, true);
            if (!res) {
                bool w12 = ((pref->type == WILD1) && (nrec.type == WILD2)) ||
                           ((pref->type == WILD2) && (nrec.type == WILD1));
                if (!w12) {
                    s_TotalErrors++;
                }
                continue;
            }
            s_Solved = true;
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

    // herd save/load buffers. save_bufs holds one
    // host-side cudaMemcpy destination per GPU, sized KangCnt * 96. They
    // get armed onto s_GpuKangs[i]->SaveKangsHost before solve() runs.
    // load_bufs holds parsed file contents and gets armed onto
    // s_GpuKangs[i]->InitKangsHost. Both are cleared after solve()
    // returns OR after save_herd_state writes the file. Sized for at
    // most MAX_GPU_CNT entries; only the first num_active_gpus are used.
    std::vector<std::vector<uint8_t>> save_bufs;
    std::vector<std::vector<uint8_t>> load_bufs;
    bool save_armed = false;
    bool load_armed = false;
    // Saved at solve() entry for use by save_herd_state's config_hash
    // field. The hash binds the on-disk state to a specific target so
    // that loading state from puzzle 75 against puzzle 80 fails fast
    // rather than silently corrupting the run.
    int saved_range_bits = 0;
    int saved_dp_bits = 0;
    std::string saved_pubkey_hex;
    std::string saved_start_hex;

    ~Impl() {
        cleanup();
    }

    void cleanup() {
        if (s_pPntList) {
            free(s_pPntList);
            s_pPntList = nullptr;
        }
        if (s_pPntList2) {
            free(s_pPntList2);
            s_pPntList2 = nullptr;
        }
        for (int i = 0; i < s_GpuCnt; i++) {
            if (s_GpuKangs[i]) {
                delete s_GpuKangs[i];
                s_GpuKangs[i] = nullptr;
            }
        }
        s_GpuCnt = 0;
        s_db.Clear();
        if (initialized) {
            DeInitEc();
            initialized = false;
        }
    }
};

RCKangarooManager::RCKangarooManager() : impl_(new Impl()) {
    // Enforce the implicit singleton constraint described at the top of
    // this file. RCKangaroo's GpuKang.cpp calls AddPointsToList as a
    // free function with no context pointer, so all RCKangaroo state
    // lives in the s_X file-scope variables above. A second
    // RCKangarooManager instance would silently corrupt the first one's
    // search; we'd rather crash deterministically than keep solving
    // with poisoned state.
    void* expected = nullptr;
    if (!g_active_rckangaroo.compare_exchange_strong(
            expected, static_cast<void*>(this), std::memory_order_acq_rel)) {
        delete impl_;
        impl_ = nullptr;
        std::cerr << "[!] RCKangarooManager: a second instance was constructed "
                     "while another is active. RCKangaroo's third-party "
                     "kernel state is process-singleton (see "
                     "third_party/RCKangaroo/GpuKang.cpp). Aborting before "
                     "concurrent state corruption." << std::endl;
        std::abort();
    }
}

RCKangarooManager::~RCKangarooManager() {
    // Release the singleton slot so a future instance can construct.
    g_active_rckangaroo.store(nullptr, std::memory_order_release);
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

    s_GpuCnt = 0;

    for (int i = 0; i < gcnt; i++) {
        // Check if this GPU should be used
        if (!gpu_ids.empty()) {
            bool found = false;
            for (int id : gpu_ids) {
                if (id == i) { found = true; break; }
            }
            if (!found) continue;
        }

        // T2.2: cudaSetDeviceFlags MUST run BEFORE any CUDA context
        // exists on device `i`. Per the CUDA Runtime API contract
        // (cudaSetDeviceFlags docs): "If the current device has been
        // set and that device has already been initialized then this
        // call will fail with the error cudaErrorSetOnActiveProcess."
        // The pre-fix code called this AFTER cudaSetDevice(i), which
        // implicitly creates the context, so the flag-set was a no-op
        // returning cudaErrorSetOnActiveProcess silently (return value
        // was never checked). Move the flag-set to BEFORE cudaSetDevice
        // and log any non-success return so the operator sees ordering
        // violations introduced by future init reordering. We swallow
        // cudaErrorSetOnActiveProcess (not a hard error -- means some
        // other init path beat us to creating the context, e.g.
        // secp256k1_init_table or a bench warmup); any other error is
        // surfaced as a warning. Schedule policy is not required for
        // correctness; this is purely a CPU-spin-vs-sleep hint to the
        // CUDA runtime during synchronous waits. We continue past the
        // failure either way.
        cudaError_t flag_status = cudaSetDeviceFlags(
            cudaDeviceScheduleBlockingSync);
        if (flag_status != cudaSuccess &&
            flag_status != cudaErrorSetOnActiveProcess) {
            std::cerr << "[!] RCKangaroo: cudaSetDeviceFlags(BlockingSync) "
                         "for GPU " << i << " returned "
                      << cudaGetErrorString(flag_status)
                      << "; continuing with default schedule policy.\n";
            // Clear the sticky last-error so the next CUDA call's
            // diagnostic isn't muddied by this non-fatal return.
            cudaGetLastError();
        } else if (flag_status == cudaErrorSetOnActiveProcess) {
            // Expected when another init path (secp256k1 setup, a
            // pre-flight benchmark, an earlier RCKangarooManager
            // instance, etc.) has already created a context on this
            // device. Clear the sticky error and move on; the device
            // keeps whatever schedule flag it was first set with.
            cudaGetLastError();
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


        s_GpuKangs[s_GpuCnt] = new RCGpuKang();
        s_GpuKangs[s_GpuCnt]->CudaIndex = i;
        s_GpuKangs[s_GpuCnt]->persistingL2CacheMaxSize = prop.persistingL2CacheMaxSize;
        s_GpuKangs[s_GpuCnt]->mpCnt = prop.multiProcessorCount;
        s_GpuKangs[s_GpuCnt]->IsOldGpu = prop.l2CacheSize < 16 * 1024 * 1024;
        s_GpuCnt++;
    }

    std::cout << "Total GPUs initialized: " << s_GpuCnt << std::endl;

    // Allocate DP buffers
    s_pPntList = (u8*)malloc(MAX_CNT_LIST * GPU_DP_SIZE);
    s_pPntList2 = (u8*)malloc(MAX_CNT_LIST * GPU_DP_SIZE);

    impl_->gpu_ids = gpu_ids;
    return s_GpuCnt;
}

int RCKangarooManager::num_gpus() const {
    return s_GpuCnt;
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
    return s_db.LoadFromFile(const_cast<char*>(filename.c_str()));
}

bool RCKangarooManager::generate_tames(const std::string& filename, double max_ops) {
    impl_->tames_file = filename;
    Ec ec;

    if (s_GpuCnt == 0) {
        std::cerr << "No GPUs initialized for tames generation" << std::endl;
        return false;
    }

    int Range = range_bits;
    int DP = dp_bits;
    s_GenMode = true;  // Enable tames generation mode

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
    s_PntTotalOps = 0;
    s_PntIndex = 0;
    s_TotalErrors = 0;
    s_Solved = false;

    // Use a fixed seed for reproducible tames generation
    // This allows tames files to be compatible across runs
    SetRndSeed(0);

    // Prepare jump tables (same as in solve, for consistency)
    EcInt minjump, t;
    minjump.Set(1);
    minjump.ShiftLeft(Range / 2 + 3);
    for (int i = 0; i < JMP_CNT; i++) {
        s_EcJumps1[i].dist = minjump;
        t.RndMax(minjump);
        s_EcJumps1[i].dist.Add(t);
        s_EcJumps1[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE;  // Must be even
        s_EcJumps1[i].p = ec.MultiplyG(s_EcJumps1[i].dist);
    }

    minjump.Set(1);
    minjump.ShiftLeft(Range - 10);
    for (int i = 0; i < JMP_CNT; i++) {
        s_EcJumps2[i].dist = minjump;
        t.RndMax(minjump);
        s_EcJumps2[i].dist.Add(t);
        s_EcJumps2[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE;
        s_EcJumps2[i].p = ec.MultiplyG(s_EcJumps2[i].dist);
    }

    minjump.Set(1);
    minjump.ShiftLeft(Range - 10 - 2);
    for (int i = 0; i < JMP_CNT; i++) {
        s_EcJumps3[i].dist = minjump;
        t.RndMax(minjump);
        s_EcJumps3[i].dist.Add(t);
        s_EcJumps3[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE;
        s_EcJumps3[i].p = ec.MultiplyG(s_EcJumps3[i].dist);
    }

    // Restore random seed for randomized starting points
#ifdef _WIN32
    SetRndSeed(GetTickCount64());
#else
    SetRndSeed(std::chrono::steady_clock::now().time_since_epoch().count());
#endif

    // Set half range
    s_Int_HalfRange.Set(1);
    s_Int_HalfRange.ShiftLeft(Range - 1);
    s_Pnt_HalfRange = ec.MultiplyG(s_Int_HalfRange);

    // For tames generation, we use the generator point G as the "target"
    // This creates tames that can be used for any public key in this range
    EcPoint PntForTames;
    PntForTames.x.SetZero();
    PntForTames.y.SetZero();
    // Use a dummy point - tames are generated relative to halfrange
    s_PntToSolve = s_Pnt_HalfRange;  // Use half range point as reference

    // Prepare GPUs for tames generation
    for (int i = 0; i < s_GpuCnt; i++) {
        if (!s_GpuKangs[i]->Prepare(s_Pnt_HalfRange, Range, DP, s_EcJumps1, s_EcJumps2, s_EcJumps3)) {
            s_GpuKangs[i]->Failed = true;
            std::cerr << "GPU " << s_GpuKangs[i]->CudaIndex << " Prepare failed for tames generation" << std::endl;
        }
    }

    auto start_time = std::chrono::steady_clock::now();
    std::cout << "Starting tames generation on " << s_GpuCnt << " GPUs..." << std::endl;

    // Launch worker threads
#ifdef _WIN32
    HANDLE thr_handles[MAX_GPU_CNT];
    u32 ThreadID;
    s_ThrCnt = s_GpuCnt;
    for (int i = 0; i < s_GpuCnt; i++) {
        thr_handles[i] = (HANDLE)_beginthreadex(NULL, 0, kang_thr_proc,
                                                 (void*)s_GpuKangs[i], 0, &ThreadID);
    }
#else
    pthread_t thr_handles[MAX_GPU_CNT];
    s_ThrCnt = s_GpuCnt;
    for (int i = 0; i < s_GpuCnt; i++) {
        pthread_create(&thr_handles[i], NULL, kang_thr_proc, (void*)s_GpuKangs[i]);
    }
#endif

    // Main loop - collect tames until we hit the operations limit
    auto last_stats = std::chrono::steady_clock::now();
    while (!stop_flag.load()) {
        // In gen mode, CheckNewPoints just adds to database without looking for collisions
        CheckNewPoints();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        // Check if we've reached the operations limit
        if (s_PntTotalOps >= static_cast<u64>(max_total_ops)) {
            std::cout << "\nOperations limit reached, stopping..." << std::endl;
            break;
        }

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_stats).count() >= 10) {
            int speed = 0;
            for (int i = 0; i < s_GpuCnt; i++) {
                if (!s_GpuKangs[i]->Failed) {
                    speed += s_GpuKangs[i]->GetStatsSpeed();
                }
            }

            double progress = (static_cast<double>(s_PntTotalOps) / max_total_ops) * 100.0;
            std::cout << "GEN: Speed: " << speed << " MKeys/s, DPs: " << s_db.GetBlockCnt()
                      << ", Ops: 2^" << std::fixed << std::setprecision(2) << log2(static_cast<double>(s_PntTotalOps))
                      << ", Progress: " << std::setprecision(1) << progress << "%" << std::endl;
            last_stats = now;
        }
    }

    // Stop workers
    for (int i = 0; i < s_GpuCnt; i++)
        s_GpuKangs[i]->Stop();
    while (s_ThrCnt)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Close thread handles
#ifdef _WIN32
    for (int i = 0; i < s_GpuCnt; i++)
        CloseHandle(thr_handles[i]);
#else
    for (int i = 0; i < s_GpuCnt; i++)
        pthread_join(thr_handles[i], NULL);
#endif

    auto end_time = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end_time - start_time).count();

    // Save tames to file
    std::cout << "\nSaving tames to " << filename << "..." << std::endl;
    s_db.Header[0] = static_cast<u8>(Range);  // Store range in header for compatibility check

    // Need to cast away const for the C-style API
    char* fn_cstr = const_cast<char*>(filename.c_str());
    bool saved = s_db.SaveToFile(fn_cstr);

    if (saved) {
        std::cout << "=== TAMES GENERATION COMPLETE ===" << std::endl;
        std::cout << "Tames saved: " << s_db.GetBlockCnt() << std::endl;
        std::cout << "Total ops: 2^" << log2(static_cast<double>(s_PntTotalOps)) << std::endl;
        std::cout << "Time: " << std::setprecision(1) << elapsed << " seconds" << std::endl;
        std::cout << "File: " << filename << std::endl;
    } else {
        std::cerr << "ERROR: Failed to save tames to " << filename << std::endl;
    }

    s_db.Clear();
    s_GenMode = false;  // Reset generation mode
    return saved;
}

RCKangarooResult RCKangarooManager::solve() {
    RCKangarooResult result = {};
    Ec ec;

    if (!impl_->pubkey_set) {
        std::cerr << "Target public key not set" << std::endl;
        return result;
    }

    if (s_GpuCnt == 0) {
        std::cerr << "No GPUs initialized" << std::endl;
        return result;
    }

    int Range = range_bits;
    int DP = dp_bits;
    s_GenMode = false;

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
    s_PntToSolve = PntToSolve;

    // Initialize state
    s_PntTotalOps = 0;
    s_PntIndex = 0;
    s_TotalErrors = 0;

#ifdef COLLIDER_PRO
    // Reset opportunistic-bloom stats and arm the hit callback for this run.
    collider::gpu::bloom::reset_stats();
    collider::gpu::bloom::set_hit_callback(bloom_hit_callback);

    if (bloom_enabled && collider::gpu::bloom::is_loaded()) {
        std::cout << "[Bloom] Opportunistic address checking enabled" << std::endl;
    }
#endif
    s_dp_callback = dp_callback;

    s_Solved = false;

    // Prepare jump tables
    SetRndSeed(0);
    EcInt minjump, t;
    minjump.Set(1);
    minjump.ShiftLeft(Range / 2 + 3);
    for (int i = 0; i < JMP_CNT; i++) {
        s_EcJumps1[i].dist = minjump;
        t.RndMax(minjump);
        s_EcJumps1[i].dist.Add(t);
        s_EcJumps1[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE;
        s_EcJumps1[i].p = ec.MultiplyG(s_EcJumps1[i].dist);
    }

    minjump.Set(1);
    minjump.ShiftLeft(Range - 10);
    for (int i = 0; i < JMP_CNT; i++) {
        s_EcJumps2[i].dist = minjump;
        t.RndMax(minjump);
        s_EcJumps2[i].dist.Add(t);
        s_EcJumps2[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE;
        s_EcJumps2[i].p = ec.MultiplyG(s_EcJumps2[i].dist);
    }

    minjump.Set(1);
    minjump.ShiftLeft(Range - 10 - 2);
    for (int i = 0; i < JMP_CNT; i++) {
        s_EcJumps3[i].dist = minjump;
        t.RndMax(minjump);
        s_EcJumps3[i].dist.Add(t);
        s_EcJumps3[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE;
        s_EcJumps3[i].p = ec.MultiplyG(s_EcJumps3[i].dist);
    }

#ifdef _WIN32
    SetRndSeed(GetTickCount64());
#else
    SetRndSeed(std::chrono::steady_clock::now().time_since_epoch().count());
#endif

    // Set half range
    s_Int_HalfRange.Set(1);
    s_Int_HalfRange.ShiftLeft(Range - 1);
    s_Pnt_HalfRange = ec.MultiplyG(s_Int_HalfRange);

    // Prepare GPUs
    for (int i = 0; i < s_GpuCnt; i++) {
        // v1.5: propagate the asymmetric kangaroo mode chosen by the
        // caller (pool runtime) to each per-GPU RCGpuKang BEFORE its
        // Prepare() runs. Prepare() snapshots Mode into Kparams.Mode
        // and configures TAME_ONLY / WILD_ONLY starting points + kernel
        // dispatch. Default (mode == KANG_MODE_BOTH == 0) preserves
        // upstream behavior for the standalone solver.
        s_GpuKangs[i]->Mode = mode;
        if (!s_GpuKangs[i]->Prepare(PntToSolve, Range, DP, s_EcJumps1, s_EcJumps2, s_EcJumps3)) {
            s_GpuKangs[i]->Failed = true;
            std::cerr << "GPU " << s_GpuKangs[i]->CudaIndex << " Prepare failed" << std::endl;
        }
    }

    // arm herd save / load hooks on each GPU.
    // Done AFTER Prepare so KangCnt is finalized, and BEFORE the worker
    // threads start (kang_thr_proc calls Execute which is the only
    // reader of InitKangsHost / writer of SaveKangsHost). The hooks are
    // file-pointer hooks; the actual file I/O happens outside solve()
    // in request_save_on_stop / save_herd_state / load_herd_state.
    if (impl_->load_armed) {
        for (int i = 0; i < s_GpuCnt; i++) {
            const int kc = s_GpuKangs[i]->KangCnt;
            if (i >= static_cast<int>(impl_->load_bufs.size())) break;
            if (impl_->load_bufs[i].size() == static_cast<size_t>(kc) * 96) {
                s_GpuKangs[i]->InitKangsHost = impl_->load_bufs[i].data();
            } else {
                std::cerr << "[!] RCKangaroo: load buffer size mismatch on GPU "
                          << i << " (expected " << (kc * 96)
                          << ", got " << impl_->load_bufs[i].size() << "). "
                             "Falling back to random init for this GPU.\n";
                s_GpuKangs[i]->InitKangsHost = nullptr;
            }
        }
    }
    if (impl_->save_armed) {
        // Allocate save buffers sized to each GPU's KangCnt. Each GPU's
        // KangCnt depends on its mpCnt and IsOldGpu, so they may differ
        // across heterogeneous configurations; size each buffer
        // independently rather than assuming uniformity.
        impl_->save_bufs.resize(s_GpuCnt);
        for (int i = 0; i < s_GpuCnt; i++) {
            const int kc = s_GpuKangs[i]->KangCnt;
            impl_->save_bufs[i].assign(static_cast<size_t>(kc) * 96, 0);
            s_GpuKangs[i]->SaveKangsHost = impl_->save_bufs[i].data();
        }
    }
    // Capture the config parameters that go into the on-disk fingerprint.
    impl_->saved_range_bits = Range;
    impl_->saved_dp_bits = DP;
    {
        char pubhex[200];
        impl_->target_pubkey.x.GetHexStr(pubhex);
        impl_->saved_pubkey_hex = pubhex;
        impl_->start_offset.GetHexStr(pubhex);
        impl_->saved_start_hex = pubhex;
    }

    auto start_time = std::chrono::steady_clock::now();
    std::cout << "GPUs started..." << std::endl;

    // Launch worker threads
#ifdef _WIN32
    HANDLE thr_handles[MAX_GPU_CNT];
    u32 ThreadID;
    s_ThrCnt = s_GpuCnt;
    for (int i = 0; i < s_GpuCnt; i++) {
        thr_handles[i] = (HANDLE)_beginthreadex(NULL, 0, kang_thr_proc,
                                                 (void*)s_GpuKangs[i], 0, &ThreadID);
    }
#else
    pthread_t thr_handles[MAX_GPU_CNT];
    s_ThrCnt = s_GpuCnt;
    for (int i = 0; i < s_GpuCnt; i++) {
        pthread_create(&thr_handles[i], NULL, kang_thr_proc, (void*)s_GpuKangs[i]);
    }
#endif

    // Main loop. Two cadences run off the same clock:
    //   * progress_callback fires every 1 s so the TUI dashboard's
    //     THROUGHPUT row and the pool runner's poll-work-id check
    //     get a real-time signal (the documented on_progress
    //     contract is 1 Hz; the older 10 s interval starved the
    //     dashboard and the operator saw 0 Keys/s for the first
    //     ~10 s of every session even with kernels at full tilt).
    //   * the legacy fallback "Speed: ... DPs: ..." stdout line
    //     (active only when no progress_callback is wired) keeps
    //     its 10 s cadence so unattended CLI runs don't spam.
    auto last_progress = std::chrono::steady_clock::now();
    auto last_stdout   = last_progress;
    while (!s_Solved && !stop_flag.load()) {
        CheckNewPoints();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        auto now = std::chrono::steady_clock::now();
        const bool progress_due =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                now - last_progress).count() >= 1000;
        const bool stdout_due =
            std::chrono::duration_cast<std::chrono::seconds>(
                now - last_stdout).count() >= 10;

        if (progress_due || stdout_due) {
            int speed = 0;
            for (int i = 0; i < s_GpuCnt; i++) {
                // Only query speed from GPUs that haven't failed
                if (!s_GpuKangs[i]->Failed) {
                    speed += s_GpuKangs[i]->GetStatsSpeed();
                }
            }
            impl_->current_speed = speed;

            if (progress_due && progress_callback) {
                if (!progress_callback(s_PntTotalOps, s_db.GetBlockCnt(), speed)) {
                    stop_flag.store(true);
                }
                last_progress = now;
            } else if (progress_due) {
                last_progress = now;
            }

            if (stdout_due && !progress_callback) {
                u64 est_dps_cnt = (u64)(ops / dp_val);
                std::string speed_str;
                if (speed >= 1000) {
                    speed_str = std::to_string(speed / 1000) + "." +
                                std::to_string((speed % 1000) / 100) + " GKeys/s";
                } else {
                    speed_str = std::to_string(speed) + " MKeys/s";
                }
                std::cout << "Speed: " << speed_str << ", Err: " << s_TotalErrors
                          << ", DPs: " << s_db.GetBlockCnt() << "/" << est_dps_cnt
                          << std::endl;
            }
            if (stdout_due) last_stdout = now;
        }
    }

    // Stop workers
    for (int i = 0; i < s_GpuCnt; i++)
        s_GpuKangs[i]->Stop();
    while (s_ThrCnt)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Close thread handles
#ifdef _WIN32
    for (int i = 0; i < s_GpuCnt; i++)
        CloseHandle(thr_handles[i]);
#else
    for (int i = 0; i < s_GpuCnt; i++)
        pthread_join(thr_handles[i], NULL);
#endif

    auto end_time = std::chrono::steady_clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
    result.total_ops = s_PntTotalOps;
    result.dp_count = s_db.GetBlockCnt();
    result.error_count = s_TotalErrors;
    result.k_value = (double)s_PntTotalOps / pow(2.0, Range / 2.0);

#ifdef COLLIDER_PRO
    // Copy opportunistic-bloom results out of the Pro-only bloom TU.
    result.bloom_checks = collider::gpu::bloom::checks();
    result.bloom_hits = collider::gpu::bloom::hits();

    if (bloom_enabled && result.bloom_checks > 0) {
        std::cout << "[Bloom] Total checks: " << result.bloom_checks
                  << ", Hits: " << result.bloom_hits.size() << std::endl;
    }
#endif

    if (s_Solved) {
        // Apply start offset
        if (impl_->start_set) {
            s_PrivKey.Add(impl_->start_offset);
        }

        // Verify solution
        EcPoint verify = ec.MultiplyG(s_PrivKey);
        if (verify.IsEqual(impl_->target_pubkey)) {
            result.found = true;
            memcpy(result.private_key.data(), s_PrivKey.data, 32);

            char hex[100];
            s_PrivKey.GetHexStr(hex);
            std::cout << "\n+============================================================+\n"
                      << "|                      PUZZLE SOLVED!                        |\n"
                      << "+============================================================+\n"
                      << "PRIVATE KEY: " << hex << "\n"
                      << "K value: " << result.k_value << std::endl;
        } else {
            std::cerr << "FATAL: Collision found but key verification failed!" << std::endl;
        }
    }

    s_db.Clear();
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

#ifdef COLLIDER_PRO
bool RCKangarooManager::load_bloom_filter(const std::string& filename) {
    if (collider::gpu::bloom::load(filename)) {
        bloom_enabled = true;
        bloom_file = filename;
        collider::gpu::bloom::set_hit_callback(bloom_hit_callback);
        return true;
    }
    return false;
}

uint64_t RCKangarooManager::get_bloom_checks() const {
    return collider::gpu::bloom::checks();
}
#endif

// ============================================================================
// herd save / load
// ============================================================================
// File format documented in third_party/RCKangaroo/.patches/save-load-state.patch.
// Reads/writes match the cudaMemcpy device-buffer layout byte-for-byte (96
// bytes per kangaroo: x[32] || y[32] || priv[32]). The "fingerprint" header
// fields bind the file to a specific config so that loading a checkpoint
// from a different puzzle / range / dp_bits fails fast.

namespace {

constexpr char kRckHerdMagic[16] = {
    'C','O','L','L','I','D','E','R','_','R','C','K','\x01','\x00','\x00','\x00'
};
constexpr uint32_t kRckHerdVersion = 1u;
constexpr size_t   kRckBytesPerKangaroo = 96;

inline void write_u32_le(uint8_t* dst, uint32_t v) {
    dst[0] = static_cast<uint8_t>(v & 0xFF);
    dst[1] = static_cast<uint8_t>((v >> 8) & 0xFF);
    dst[2] = static_cast<uint8_t>((v >> 16) & 0xFF);
    dst[3] = static_cast<uint8_t>((v >> 24) & 0xFF);
}
inline uint32_t read_u32_le(const uint8_t* src) {
    return static_cast<uint32_t>(src[0])
         | (static_cast<uint32_t>(src[1]) << 8)
         | (static_cast<uint32_t>(src[2]) << 16)
         | (static_cast<uint32_t>(src[3]) << 24);
}

// Compute 32-byte SHA256 over (pubkey_hex || '|' || start_hex). The
// '|' separator prevents ambiguity if either field is empty (otherwise
// "abc" + "" and "ab" + "c" would hash to the same string).
void compute_config_hash(const std::string& pubkey_hex,
                         const std::string& start_hex,
                         uint8_t out[32]) {
    std::string combined = pubkey_hex;
    combined.push_back('|');
    combined += start_hex;
    cpu_sha256(reinterpret_cast<const uint8_t*>(combined.data()),
               combined.size(), out);
}

}  // namespace

bool RCKangarooManager::request_save_on_stop() {
    if (s_GpuCnt == 0) {
        std::cerr << "[!] RCKangaroo: request_save_on_stop called before "
                     "init(); ignored.\n";
        return false;
    }
    impl_->save_armed = true;
    return true;
}

bool RCKangarooManager::save_herd_state(const std::string& path) {
    if (s_GpuCnt == 0) return false;
    if (!impl_->save_armed) {
        std::cerr << "[!] RCKangaroo: save_herd_state called without prior "
                     "request_save_on_stop(); nothing to write.\n";
        return false;
    }
    if (impl_->save_bufs.size() != static_cast<size_t>(s_GpuCnt)) {
        // solve() never ran to completion; save buffers are not populated.
        return false;
    }

    // Validate every per-GPU buffer is properly sized. A zero-size buffer
    // indicates the GPU was marked Failed during Prepare and the patch's
    // save hook never fired.
    for (int i = 0; i < s_GpuCnt; i++) {
        const int kc = s_GpuKangs[i]->KangCnt;
        if (impl_->save_bufs[i].size() != static_cast<size_t>(kc) * kRckBytesPerKangaroo) {
            std::cerr << "[!] RCKangaroo: save buffer for GPU " << i
                      << " has unexpected size; aborting save.\n";
            return false;
        }
    }

    // T0.2: atomic tmp+rename save. The pre-fix path wrote directly to
    // `path` with fopen/fwrite/fclose; a SIGINT, power loss, or kernel
    // panic between the first fwrite() and the final fclose() would
    // leave a truncated .kang file on disk. load_herd_state() then
    // rejects it on the header check, the operator restarts, and the
    // herd starts over from scratch -- potentially throwing away days
    // of kangaroo work. Mirror BrainWalletStateManager::save_state's
    // contract:
    //
    //   1. Write the full payload to "<path>.tmp".
    //   2. fflush() + fsync() (POSIX) / _commit() (Windows) so the OS
    //      cache is on stable storage before the rename.
    //   3. std::filesystem::rename("<path>.tmp", path). POSIX rename is
    //      atomic against concurrent readers; Windows rename overwrites
    //      iff we remove the target first (handled below) and is
    //      atomic at the filesystem-metadata level.
    //
    // On any write/flush/sync failure we close the temp fp, remove the
    // partial .tmp, and return false. The previous primary .kang (if
    // any) is unchanged -- load_herd_state will still find it.
    namespace fs = std::filesystem;
    const fs::path final_path(path);
    fs::path tmp_path = final_path;
    tmp_path += ".tmp";

    FILE* fp = std::fopen(tmp_path.string().c_str(), "wb");
    if (!fp) {
        std::cerr << "[!] RCKangaroo: failed to open " << tmp_path.string()
                  << " for save.\n";
        return false;
    }

    // Helper for the unwind path: close fp (if still open) and unlink
    // the partial tmp so it does not accumulate on disk across retries.
    auto unwind = [&fp, &tmp_path](const char* where) -> bool {
        if (fp) {
            std::fclose(fp);
            fp = nullptr;
        }
        std::error_code rm_ec;
        fs::remove(tmp_path, rm_ec);
        std::cerr << "[!] RCKangaroo: " << where << " failed during atomic "
                     "save; partial " << tmp_path.string() << " removed.\n";
        return false;
    };

    // Header.
    if (std::fwrite(kRckHerdMagic, 1, 16, fp) != 16) {
        return unwind("magic write");
    }
    uint8_t hdr_nums[20];
    write_u32_le(hdr_nums + 0,  kRckHerdVersion);
    write_u32_le(hdr_nums + 4,  static_cast<uint32_t>(s_GpuCnt));
    write_u32_le(hdr_nums + 8,  static_cast<uint32_t>(s_GpuKangs[0]->KangCnt));
    write_u32_le(hdr_nums + 12, static_cast<uint32_t>(impl_->saved_range_bits));
    write_u32_le(hdr_nums + 16, static_cast<uint32_t>(impl_->saved_dp_bits));
    if (std::fwrite(hdr_nums, 1, sizeof(hdr_nums), fp) != sizeof(hdr_nums)) {
        return unwind("header write");
    }
    uint8_t config_hash[32];
    compute_config_hash(impl_->saved_pubkey_hex, impl_->saved_start_hex,
                        config_hash);
    if (std::fwrite(config_hash, 1, 32, fp) != 32) {
        return unwind("config-hash write");
    }

    // Per-GPU body.
    for (int i = 0; i < s_GpuCnt; i++) {
        uint8_t gpu_hdr[8];
        write_u32_le(gpu_hdr + 0, static_cast<uint32_t>(s_GpuKangs[i]->CudaIndex));
        write_u32_le(gpu_hdr + 4, 0);  // reserved
        if (std::fwrite(gpu_hdr, 1, sizeof(gpu_hdr), fp) != sizeof(gpu_hdr)) {
            return unwind("gpu header write");
        }
        const auto& buf = impl_->save_bufs[i];
        if (std::fwrite(buf.data(), 1, buf.size(), fp) != buf.size()) {
            return unwind("gpu body write");
        }
    }

    // Flush libc -> OS, then sync OS -> disk before we close fp. Doing
    // the sync BEFORE fclose() means we still have a usable fd to pass
    // to fsync / _commit; reopening the file by name to sync (the way
    // BrainWalletStateManager::save_state does on POSIX) would race
    // with any concurrent unlink and skipping the sync on Windows
    // entirely (the way that header used to) leaves a power-loss race
    // open. Here we always sync before rename.
    if (std::fflush(fp) != 0) {
        return unwind("fflush");
    }
#ifdef _WIN32
    // _commit returns 0 on success, -1 on failure. Failure means the
    // OS cache flush did not complete; aborting the save is the safe
    // option since the rename below would race with the unflushed
    // cache on a power-loss restart.
    if (_commit(_fileno(fp)) != 0) {
        return unwind("_commit");
    }
#else
    if (::fsync(::fileno(fp)) != 0) {
        return unwind("fsync");
    }
#endif

    std::fclose(fp);
    fp = nullptr;

    // Atomic rename: tmp -> final. On POSIX rename(2) is atomic against
    // concurrent readers and replaces an existing target. On Windows
    // std::filesystem::rename throws if the target exists, so remove it
    // first; the gap between remove() and rename() is small but not
    // zero -- a power loss in that window leaves the old .kang gone
    // and the .kang.tmp still on disk under its tmp name. load_herd_state
    // will then return false ("file not found"), the operator restarts
    // herd-from-scratch, and the .kang.tmp can be manually renamed if
    // the operator wants to recover it. This is strictly better than
    // the pre-fix behaviour (partial .kang silently rejected at load).
    std::error_code ec;
    fs::rename(tmp_path, final_path, ec);
    if (ec) {
        // Most-common reason on Windows is "destination exists"; retry
        // after explicit remove. We accept the small remove+rename
        // window described above.
        std::error_code rm_ec;
        fs::remove(final_path, rm_ec);
        ec.clear();
        fs::rename(tmp_path, final_path, ec);
        if (ec) {
            std::cerr << "[!] RCKangaroo: rename " << tmp_path.string()
                      << " -> " << final_path.string()
                      << " failed: " << ec.message()
                      << "; previous .kang (if any) may be lost.\n";
            // Leave .tmp on disk so the operator can manually recover.
            return false;
        }
    }

    // Clear the SaveKangsHost pointers so a subsequent solve() without
    // a fresh request_save_on_stop() does not silently overwrite the
    // buffers (they are still owned by impl_ but the hooks are off).
    for (int i = 0; i < s_GpuCnt; i++) {
        s_GpuKangs[i]->SaveKangsHost = nullptr;
    }
    impl_->save_armed = false;
    return true;
}

bool RCKangarooManager::load_herd_state(const std::string& path) {
    if (s_GpuCnt == 0) {
        std::cerr << "[!] RCKangaroo: load_herd_state called before init(); "
                     "ignoring.\n";
        return false;
    }

    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return false;

    uint8_t magic[16];
    if (std::fread(magic, 1, 16, fp) != 16) {
        std::fclose(fp); return false;
    }
    if (std::memcmp(magic, kRckHerdMagic, 16) != 0) {
        std::cerr << "[!] RCKangaroo: " << path << " magic mismatch; not a "
                     "kangaroo herd checkpoint.\n";
        std::fclose(fp); return false;
    }

    uint8_t hdr_nums[20];
    if (std::fread(hdr_nums, 1, sizeof(hdr_nums), fp) != sizeof(hdr_nums)) {
        std::fclose(fp); return false;
    }
    const uint32_t file_version    = read_u32_le(hdr_nums + 0);
    const uint32_t file_gpus       = read_u32_le(hdr_nums + 4);
    const uint32_t file_kang_cnt   = read_u32_le(hdr_nums + 8);
    const uint32_t file_range_bits = read_u32_le(hdr_nums + 12);
    const uint32_t file_dp_bits    = read_u32_le(hdr_nums + 16);

    if (file_version != kRckHerdVersion) {
        std::cerr << "[!] RCKangaroo: checkpoint version " << file_version
                  << " unsupported (expected " << kRckHerdVersion << ").\n";
        std::fclose(fp); return false;
    }
    if (file_gpus != static_cast<uint32_t>(s_GpuCnt)) {
        std::cerr << "[!] RCKangaroo: checkpoint has " << file_gpus
                  << " GPUs but current run has " << s_GpuCnt
                  << "; refusing to load.\n";
        std::fclose(fp); return false;
    }
    if (file_kang_cnt != static_cast<uint32_t>(s_GpuKangs[0]->KangCnt)) {
        std::cerr << "[!] RCKangaroo: checkpoint KangCnt " << file_kang_cnt
                  << " does not match current " << s_GpuKangs[0]->KangCnt
                  << " (GPU model change?); refusing to load.\n";
        std::fclose(fp); return false;
    }
    if (file_range_bits != static_cast<uint32_t>(range_bits)) {
        std::cerr << "[!] RCKangaroo: checkpoint range_bits " << file_range_bits
                  << " does not match current " << range_bits
                  << "; refusing to load.\n";
        std::fclose(fp); return false;
    }
    if (file_dp_bits != static_cast<uint32_t>(dp_bits)) {
        std::cerr << "[!] RCKangaroo: checkpoint dp_bits " << file_dp_bits
                  << " does not match current " << dp_bits
                  << "; refusing to load.\n";
        std::fclose(fp); return false;
    }

    uint8_t file_config_hash[32];
    if (std::fread(file_config_hash, 1, 32, fp) != 32) {
        std::fclose(fp); return false;
    }
    char pubhex[200], starthex[200];
    impl_->target_pubkey.x.GetHexStr(pubhex);
    impl_->start_offset.GetHexStr(starthex);
    uint8_t expected_config_hash[32];
    compute_config_hash(pubhex, starthex, expected_config_hash);
    if (std::memcmp(file_config_hash, expected_config_hash, 32) != 0) {
        std::cerr << "[!] RCKangaroo: checkpoint config fingerprint does not "
                     "match current target pubkey + range_start; refusing to "
                     "load.\n";
        std::fclose(fp); return false;
    }

    // Per-GPU body.
    impl_->load_bufs.assign(s_GpuCnt, std::vector<uint8_t>{});
    for (int i = 0; i < s_GpuCnt; i++) {
        uint8_t gpu_hdr[8];
        if (std::fread(gpu_hdr, 1, sizeof(gpu_hdr), fp) != sizeof(gpu_hdr)) {
            std::fclose(fp); return false;
        }
        // cuda_index in gpu_hdr is informational; GPU renumbering between
        // runs is tolerated (the order in the file is the order they
        // appear in s_GpuKangs).

        const size_t body_size = static_cast<size_t>(file_kang_cnt)
                                 * kRckBytesPerKangaroo;
        impl_->load_bufs[i].resize(body_size);
        if (std::fread(impl_->load_bufs[i].data(), 1, body_size, fp) != body_size) {
            std::fclose(fp); return false;
        }
    }
    std::fclose(fp);

    impl_->load_armed = true;
    return true;
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
