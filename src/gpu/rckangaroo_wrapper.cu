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
#ifdef COLLIDER_CHECKPOINT_CAPTURE
// Mod-n reduction + big-endian Distance encoding for the captured per-kangaroo
// checkpoint chain. Header-only (crypto_cpu.hpp is all inline).
#include "../core/checkpoint_commit.hpp"
#endif
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
#include <array>        // checkpoint chain Distance (std::array<uint8_t,32>)
#include <vector>       // checkpoint chain readback buffers

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
// #ifdef COLLIDER_PRO. The DP-export callback (pool mode) now lives inside
// RckSingletonState::dp_callback (declared below) rather than as a loose
// file-scope std::function, so all per-run state has a single owner.

// ============================================================================
// RckSingletonState - encapsulated RCKangaroo run state.
//
// Why this is still a single process-wide owner (and not a per-instance
// Impl member):
//   RCKangaroo's device side is irreducibly process-global. RCGpuCore.cu
//   declares __constant__ jump-table symbols (jmp2_table) that are bound
//   per process via cudaMemcpyToSymbol inside cuSetGpuParams, and the EC
//   library's precomputed tables are owned by the global InitEc()/DeInitEc()
//   pair. A second concurrent RCKangaroo search would stomp that shared
//   device constant memory regardless of how the host-side accumulator is
//   scoped. Threading a context pointer therefore CANNOT remove the
//   singleton; the singleton is a property of the upstream device code, not
//   of this wrapper. The g_active_rckangaroo guard below converts that real
//   constraint into a deterministic abort on misuse rather than silent
//   state corruption, and the review explicitly requires it to remain.
//
// What the refactor DOES fix (the testability / reentrancy axis the review
// flagged): the ~15 loose s_* file-scope variables are collapsed into this
// one struct with a single named owner (g_rck_state), and AddPointsToList no
// longer reaches blindly into file scope -- it now receives its accumulator
// explicitly through the context pointer threaded from
// RCGpuKang::PointSinkCtx (see third_party/RCKangaroo/GpuKang.cpp). The
// function body is a pure function of (ctx, data, cnt, ops): given a state
// pointer it is independently exercisable, and a future multi-context build
// (should upstream ever make InitEc/jmp2_table per-instance) only has to
// hand a different pointer in. The members keep their former names so the
// diff against the rest of the file stays mechanical (s_X -> g_rck_state.X).
// ============================================================================
struct RckSingletonState {
    EcJMP EcJumps1[JMP_CNT];
    EcJMP EcJumps2[JMP_CNT];
    EcJMP EcJumps3[JMP_CNT];
    RCGpuKang* GpuKangs[MAX_GPU_CNT] = {};
    int GpuCnt = 0;
    volatile bool Solved = false;
    volatile long ThrCnt = 0;

    EcInt Int_HalfRange;
    EcPoint Pnt_HalfRange;
    EcInt PrivKey;
    EcPoint PntToSolve;

    CriticalSection csAddPoints;
    u8* pPntList = nullptr;
    u8* pPntList2 = nullptr;
    volatile int PntIndex = 0;
    TFastBase db;
    u64 PntTotalOps = 0;
    u32 TotalErrors = 0;
    bool GenMode = false;

    // DP-export callback (pool mode). Not a bloom symbol; lives here so all
    // run state has one owner. Cleared between runs by solve().
    //
    // v1.5.5 (task #9): the trailing two pointers carry the COMMITTABLE
    // checkpoint chain for the kangaroo that produced this DP (ordered 32-byte
    // big-endian distances mod n + per-checkpoint L1S2 bits), read back at
    // harvest in CheckNewPoints when this build captured one. nullptr on the
    // non-capture build, and nullptr for any DP whose walk is not
    // server-replayable, so the pool client falls back to DP_BATCH_V2.
    std::function<void(const uint8_t*, const uint8_t*, uint8_t,
                       const std::vector<std::array<uint8_t, 32>>*,
                       const std::vector<uint8_t>*)> dp_callback;
};

// The single permitted instance. Justified above: RCKangaroo's __constant__
// device memory and global EC tables make a second concurrent search
// impossible regardless of host-side scoping. The g_active_rckangaroo guard
// enforces that at the C++ level.
static RckSingletonState g_rck_state;

// Active-instance tracker. NULL when no RCKangarooManager exists; non-
// null only between construct/destroy of the single permitted instance.
// Stored as void* to dodge forward-declaration namespace resolution
// (RCKangarooManager lives in collider::gpu, declared further below);
// we only need atomic compare-exchange semantics, not pointer
// dereferences.
static std::atomic<void*> g_active_rckangaroo{nullptr};

#ifdef COLLIDER_CHECKPOINT_CAPTURE
// v1.5.5 checkpoint-replay (task #9). When solve() arms the device capture it
// publishes the active manager (so CheckNewPoints can call read_checkpoint_chain
// on it) and the armed window size (so CheckNewPoints can bound the DP byte-60
// kangaroo index to the captured slots). Both are written on the solve() thread
// BEFORE the worker threads launch and the main loop calls CheckNewPoints, and
// cleared after the worker threads join, so the single-reader CheckNewPoints
// (run from the same solve() thread) never races them. They are plain pointers
// because the capture readback is single-threaded within solve(); the
// g_active_rckangaroo singleton guard already forbids a second concurrent
// manager. nullptr / 0 means "capture not armed" -> CheckNewPoints emits no
// chain and the client falls back to V2. The RCKangarooManager type is fully
// defined by the included rckangaroo_wrapper.hpp above.
static const ::collider::gpu::RCKangarooManager* g_active_capture_mgr = nullptr;
static uint32_t g_capture_window_size = 0;
#endif

// ============================================================================
// AddPointsToList - Called by RCGpuKang::Execute() after the kernel completes.
// Required by GpuKang.cpp (extern declaration there). The first argument is
// the opaque context pointer threaded from RCGpuKang::PointSinkCtx; the
// wrapper sets it to &g_rck_state before launching workers, so this function
// operates purely on its arguments rather than reaching into file scope.
// ============================================================================
void AddPointsToList(void* ctx, u32* data, int pnt_cnt, u64 ops_cnt) {
    // ctx is always &g_rck_state in this build (set in solve()/generate_tames
    // before any worker thread can call back). Guard against a null context
    // defensively so a future caller that forgets to wire it fails loudly at
    // its own call site rather than corrupting unrelated memory.
    if (!ctx) {
        std::cerr << "AddPointsToList called with null context; DP dropped."
                  << std::endl;
        return;
    }
    RckSingletonState& st = *static_cast<RckSingletonState*>(ctx);
    st.csAddPoints.Enter();
    if (st.PntIndex + pnt_cnt >= MAX_CNT_LIST) {
        st.csAddPoints.Leave();
        std::cerr << "DPs buffer overflow, increase DP value!" << std::endl;
        return;
    }
    memcpy(st.pPntList + GPU_DP_SIZE * st.PntIndex, data, pnt_cnt * GPU_DP_SIZE);
    st.PntIndex = st.PntIndex + pnt_cnt;  // Avoid deprecated volatile compound assignment
    st.PntTotalOps += ops_cnt;
    st.csAddPoints.Leave();
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
    InterlockedDecrement(&g_rck_state.ThrCnt);
    return 0;
}
#else
static void* kang_thr_proc(void* data) {
    RCGpuKang* Kang = (RCGpuKang*)data;
    Kang->Execute();
    __sync_fetch_and_sub(&g_rck_state.ThrCnt, 1);
    return nullptr;
}
#endif

// Collision detection using SOTA method
static bool Collision_SOTA(EcPoint& pnt, EcInt t, int TameType, EcInt w, int WildType, bool IsNeg) {
    Ec ec;
    if (IsNeg)
        t.Neg();
    if (TameType == TAME) {
        g_rck_state.PrivKey = t;
        g_rck_state.PrivKey.Sub(w);
        EcInt sv = g_rck_state.PrivKey;
        g_rck_state.PrivKey.Add(g_rck_state.Int_HalfRange);
        EcPoint P = ec.MultiplyG(g_rck_state.PrivKey);
        if (P.IsEqual(pnt))
            return true;
        g_rck_state.PrivKey = sv;
        g_rck_state.PrivKey.Neg();
        g_rck_state.PrivKey.Add(g_rck_state.Int_HalfRange);
        P = ec.MultiplyG(g_rck_state.PrivKey);
        return P.IsEqual(pnt);
    } else {
        g_rck_state.PrivKey = t;
        g_rck_state.PrivKey.Sub(w);
        if (g_rck_state.PrivKey.data[4] >> 63)
            g_rck_state.PrivKey.Neg();
        g_rck_state.PrivKey.ShiftRight(1);
        EcInt sv = g_rck_state.PrivKey;
        g_rck_state.PrivKey.Add(g_rck_state.Int_HalfRange);
        EcPoint P = ec.MultiplyG(g_rck_state.PrivKey);
        if (P.IsEqual(pnt))
            return true;
        g_rck_state.PrivKey = sv;
        g_rck_state.PrivKey.Neg();
        g_rck_state.PrivKey.Add(g_rck_state.Int_HalfRange);
        P = ec.MultiplyG(g_rck_state.PrivKey);
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
    g_rck_state.csAddPoints.Enter();
    if (!g_rck_state.PntIndex) {
        g_rck_state.csAddPoints.Leave();
        return;
    }

    int cnt = g_rck_state.PntIndex;
    memcpy(g_rck_state.pPntList2, g_rck_state.pPntList, GPU_DP_SIZE * cnt);
    g_rck_state.PntIndex = 0;
    g_rck_state.csAddPoints.Leave();

#ifdef COLLIDER_PRO
    // EC context reused across this batch's opportunistic bloom probes.
    // Pro-only: with COLLIDER_PRO undefined there is no bloom hook call so
    // the context (and the probe) compile out entirely.
    Ec ec;
#endif

    for (int i = 0; i < cnt; i++) {
        DBRec nrec;
        u8* p = g_rck_state.pPntList2 + i * GPU_DP_SIZE;
        // GPU_DP_SIZE=64 layout: 0-15 x LS 128b, 16-31 x MS 128b, 32-53 d, 56 type
        memcpy(nrec.x, p, 12);
        memcpy(nrec.d, p + 32, 22);
        nrec.type = g_rck_state.GenMode ? TAME : p[56];

#ifdef COLLIDER_PRO
        // Opportunistic address scan: probe this DP's candidate address
        // against the loaded bloom filter (Pro only). All bloom logic lives
        // in rckangaroo_bloom.cu; here we only extract the distance and hand
        // it plus the run's half-range / target point to the hook.
        if (collider::gpu::bloom::is_loaded() && !g_rck_state.GenMode) {
            EcInt dist;
            memset(dist.data, 0, sizeof(dist.data));
            memcpy(dist.data, nrec.d, sizeof(nrec.d));
            if (nrec.d[21] == 0xFF) memset(((u8*)dist.data) + 22, 0xFF, 18);
            collider::gpu::bloom::probe_dp(dist, nrec.type == TAME,
                                           g_rck_state.Int_HalfRange, g_rck_state.PntToSolve, ec,
                                           g_rck_state.PntTotalOps);
        }
#endif

        DBRec* pref = (DBRec*)g_rck_state.db.FindOrAddDataBlock((u8*)&nrec);

        // Export DP to pool via callback (if in pool mode)
        if (g_rck_state.dp_callback && !g_rck_state.GenMode) {
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

            // v1.5.5 checkpoint-replay (task #9): if this build captured a
            // per-kangaroo checkpoint chain AND the producing kangaroo's chain
            // is server-replayable, read it back here (host side, where the
            // kangaroo's global index is known) and hand it to the callback so
            // the pool client can emit a real DP_BATCH_V3 commitment. Anything
            // not committable (out-of-window index, loop-escape, < 2
            // checkpoints) yields nullptr -> the client stays on V2. This reads
            // ONLY the capture buffers; it never touches x_be / d_be / type, so
            // the walk math and the V2 DP contract are byte-for-byte unchanged.
            const std::vector<std::array<uint8_t, 32>>* ckpt_dists_ptr = nullptr;
            const std::vector<uint8_t>* ckpt_l1s2_ptr = nullptr;
#ifdef COLLIDER_CHECKPOINT_CAPTURE
            std::vector<std::array<uint8_t, 32>> ckpt_dists;
            std::vector<uint8_t> ckpt_l1s2;
            // DP bytes 60..63 are the producing kangaroo's global index (a u32
            // written by RCGpuCore.cu BuildDP at DPs[15]). Read the full 4 bytes,
            // not just p[60]: a single-byte read is only correct while the capture
            // window is base-0 with CHK_MAX_KANG<=256, and silently aliases the
            // low byte if either ever changes.
            uint32_t kang_index;
            memcpy(&kang_index, p + 60, sizeof(uint32_t));
            if (g_active_capture_mgr &&
                kang_index < g_capture_window_size &&
                g_active_capture_mgr->read_checkpoint_chain(
                    kang_index, ckpt_dists, ckpt_l1s2)) {
                ckpt_dists_ptr = &ckpt_dists;
                ckpt_l1s2_ptr = &ckpt_l1s2;
            }
            g_rck_state.dp_callback(x_be, d_be, pool_type,
                                    ckpt_dists_ptr, ckpt_l1s2_ptr);
#else
            g_rck_state.dp_callback(x_be, d_be, pool_type,
                                    ckpt_dists_ptr, ckpt_l1s2_ptr);
#endif
            // Pool mode: the server owns collision detection.
            // Skip the local FindOrAddDataBlock / g_rck_state.Solved path so the
            // solve loop runs indefinitely rather than exiting on the
            // first internal collision (which happens within milliseconds
            // for small puzzles like 41-bit with 2M kangaroos).
            continue;
        }

        if (g_rck_state.GenMode)
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

            bool res = Collision_SOTA(g_rck_state.PntToSolve, t, TameType, w, WildType, false) ||
                       Collision_SOTA(g_rck_state.PntToSolve, t, TameType, w, WildType, true);
            if (!res) {
                bool w12 = ((pref->type == WILD1) && (nrec.type == WILD2)) ||
                           ((pref->type == WILD2) && (nrec.type == WILD1));
                if (!w12) {
                    g_rck_state.TotalErrors++;
                }
                continue;
            }
            g_rck_state.Solved = true;
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
    // get armed onto g_rck_state.GpuKangs[i]->SaveKangsHost before solve() runs.
    // load_bufs holds parsed file contents and gets armed onto
    // g_rck_state.GpuKangs[i]->InitKangsHost. Both are cleared after solve()
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
        if (g_rck_state.pPntList) {
            free(g_rck_state.pPntList);
            g_rck_state.pPntList = nullptr;
        }
        if (g_rck_state.pPntList2) {
            free(g_rck_state.pPntList2);
            g_rck_state.pPntList2 = nullptr;
        }
        for (int i = 0; i < g_rck_state.GpuCnt; i++) {
            if (g_rck_state.GpuKangs[i]) {
                delete g_rck_state.GpuKangs[i];
                g_rck_state.GpuKangs[i] = nullptr;
            }
        }
        g_rck_state.GpuCnt = 0;
        g_rck_state.db.Clear();
        if (initialized) {
            DeInitEc();
            initialized = false;
        }
    }
};

RCKangarooManager::RCKangarooManager() : impl_(new Impl()) {
    // Enforce the singleton constraint described at the RckSingletonState
    // definition above. The host-side run state is now encapsulated in the
    // single g_rck_state object and reached by AddPointsToList through an
    // explicit context pointer, but the constraint itself is irreducible:
    // RCKangaroo's __constant__ device jump tables and the global
    // InitEc()/DeInitEc() EC tables are process-wide, so a second concurrent
    // RCKangarooManager would corrupt the first one's device-side search no
    // matter how the host accumulator is scoped. We crash deterministically
    // here rather than keep solving with poisoned state.
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

    g_rck_state.GpuCnt = 0;

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


        g_rck_state.GpuKangs[g_rck_state.GpuCnt] = new RCGpuKang();
        g_rck_state.GpuKangs[g_rck_state.GpuCnt]->CudaIndex = i;
        g_rck_state.GpuKangs[g_rck_state.GpuCnt]->persistingL2CacheMaxSize = prop.persistingL2CacheMaxSize;
        g_rck_state.GpuKangs[g_rck_state.GpuCnt]->mpCnt = prop.multiProcessorCount;
        g_rck_state.GpuKangs[g_rck_state.GpuCnt]->IsOldGpu = prop.l2CacheSize < 16 * 1024 * 1024;
        // Thread the DP-sink context into this worker. Execute() forwards it
        // verbatim to AddPointsToList so the sink receives &g_rck_state
        // explicitly instead of reaching into file scope. Set at construction
        // so it is in place before any solve()/generate_tames() launches the
        // worker thread that calls Execute().
        g_rck_state.GpuKangs[g_rck_state.GpuCnt]->PointSinkCtx = &g_rck_state;
        g_rck_state.GpuCnt++;
    }

    std::cout << "Total GPUs initialized: " << g_rck_state.GpuCnt << std::endl;

    // Allocate DP buffers
    g_rck_state.pPntList = (u8*)malloc(MAX_CNT_LIST * GPU_DP_SIZE);
    g_rck_state.pPntList2 = (u8*)malloc(MAX_CNT_LIST * GPU_DP_SIZE);

    impl_->gpu_ids = gpu_ids;
    return g_rck_state.GpuCnt;
}

int RCKangarooManager::num_gpus() const {
    return g_rck_state.GpuCnt;
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
    return g_rck_state.db.LoadFromFile(const_cast<char*>(filename.c_str()));
}

bool RCKangarooManager::generate_tames(const std::string& filename, double max_ops) {
    impl_->tames_file = filename;
    Ec ec;

    if (g_rck_state.GpuCnt == 0) {
        std::cerr << "No GPUs initialized for tames generation" << std::endl;
        return false;
    }

    int Range = range_bits;
    int DP = dp_bits;
    g_rck_state.GenMode = true;  // Enable tames generation mode

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
    g_rck_state.PntTotalOps = 0;
    g_rck_state.PntIndex = 0;
    g_rck_state.TotalErrors = 0;
    g_rck_state.Solved = false;

    // Use a fixed seed for reproducible tames generation
    // This allows tames files to be compatible across runs
    SetRndSeed(0);

    // Prepare jump tables (same as in solve, for consistency)
    EcInt minjump, t;
    minjump.Set(1);
    minjump.ShiftLeft(Range / 2 + 3);
    for (int i = 0; i < JMP_CNT; i++) {
        g_rck_state.EcJumps1[i].dist = minjump;
        t.RndMax(minjump);
        g_rck_state.EcJumps1[i].dist.Add(t);
        g_rck_state.EcJumps1[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE;  // Must be even
        g_rck_state.EcJumps1[i].p = ec.MultiplyG(g_rck_state.EcJumps1[i].dist);
    }

    minjump.Set(1);
    minjump.ShiftLeft(Range - 10);
    for (int i = 0; i < JMP_CNT; i++) {
        g_rck_state.EcJumps2[i].dist = minjump;
        t.RndMax(minjump);
        g_rck_state.EcJumps2[i].dist.Add(t);
        g_rck_state.EcJumps2[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE;
        g_rck_state.EcJumps2[i].p = ec.MultiplyG(g_rck_state.EcJumps2[i].dist);
    }

    minjump.Set(1);
    minjump.ShiftLeft(Range - 10 - 2);
    for (int i = 0; i < JMP_CNT; i++) {
        g_rck_state.EcJumps3[i].dist = minjump;
        t.RndMax(minjump);
        g_rck_state.EcJumps3[i].dist.Add(t);
        g_rck_state.EcJumps3[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE;
        g_rck_state.EcJumps3[i].p = ec.MultiplyG(g_rck_state.EcJumps3[i].dist);
    }

    // Restore random seed for randomized starting points
#ifdef _WIN32
    SetRndSeed(GetTickCount64());
#else
    SetRndSeed(std::chrono::steady_clock::now().time_since_epoch().count());
#endif

    // Set half range
    g_rck_state.Int_HalfRange.Set(1);
    g_rck_state.Int_HalfRange.ShiftLeft(Range - 1);
    g_rck_state.Pnt_HalfRange = ec.MultiplyG(g_rck_state.Int_HalfRange);

    // For tames generation, we use the generator point G as the "target"
    // This creates tames that can be used for any public key in this range
    EcPoint PntForTames;
    PntForTames.x.SetZero();
    PntForTames.y.SetZero();
    // Use a dummy point - tames are generated relative to halfrange
    g_rck_state.PntToSolve = g_rck_state.Pnt_HalfRange;  // Use half range point as reference

    // Prepare GPUs for tames generation
    for (int i = 0; i < g_rck_state.GpuCnt; i++) {
        if (!g_rck_state.GpuKangs[i]->Prepare(g_rck_state.Pnt_HalfRange, Range, DP, g_rck_state.EcJumps1, g_rck_state.EcJumps2, g_rck_state.EcJumps3)) {
            g_rck_state.GpuKangs[i]->Failed = true;
            std::cerr << "GPU " << g_rck_state.GpuKangs[i]->CudaIndex << " Prepare failed for tames generation" << std::endl;
        }
    }

    auto start_time = std::chrono::steady_clock::now();
    std::cout << "Starting tames generation on " << g_rck_state.GpuCnt << " GPUs..." << std::endl;

    // Launch worker threads
#ifdef _WIN32
    HANDLE thr_handles[MAX_GPU_CNT];
    u32 ThreadID;
    g_rck_state.ThrCnt = g_rck_state.GpuCnt;
    for (int i = 0; i < g_rck_state.GpuCnt; i++) {
        thr_handles[i] = (HANDLE)_beginthreadex(NULL, 0, kang_thr_proc,
                                                 (void*)g_rck_state.GpuKangs[i], 0, &ThreadID);
    }
#else
    pthread_t thr_handles[MAX_GPU_CNT];
    g_rck_state.ThrCnt = g_rck_state.GpuCnt;
    for (int i = 0; i < g_rck_state.GpuCnt; i++) {
        pthread_create(&thr_handles[i], NULL, kang_thr_proc, (void*)g_rck_state.GpuKangs[i]);
    }
#endif

    // Main loop - collect tames until we hit the operations limit
    auto last_stats = std::chrono::steady_clock::now();
    while (!stop_flag.load()) {
        // In gen mode, CheckNewPoints just adds to database without looking for collisions
        CheckNewPoints();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        // Check if we've reached the operations limit
        if (g_rck_state.PntTotalOps >= static_cast<u64>(max_total_ops)) {
            std::cout << "\nOperations limit reached, stopping..." << std::endl;
            break;
        }

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_stats).count() >= 10) {
            int speed = 0;
            for (int i = 0; i < g_rck_state.GpuCnt; i++) {
                if (!g_rck_state.GpuKangs[i]->Failed) {
                    speed += g_rck_state.GpuKangs[i]->GetStatsSpeed();
                }
            }

            double progress = (static_cast<double>(g_rck_state.PntTotalOps) / max_total_ops) * 100.0;
            std::cout << "GEN: Speed: " << speed << " MKeys/s, DPs: " << g_rck_state.db.GetBlockCnt()
                      << ", Ops: 2^" << std::fixed << std::setprecision(2) << log2(static_cast<double>(g_rck_state.PntTotalOps))
                      << ", Progress: " << std::setprecision(1) << progress << "%" << std::endl;
            last_stats = now;
        }
    }

    // Stop workers
    for (int i = 0; i < g_rck_state.GpuCnt; i++)
        g_rck_state.GpuKangs[i]->Stop();
    while (g_rck_state.ThrCnt)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Close thread handles
#ifdef _WIN32
    for (int i = 0; i < g_rck_state.GpuCnt; i++)
        CloseHandle(thr_handles[i]);
#else
    for (int i = 0; i < g_rck_state.GpuCnt; i++)
        pthread_join(thr_handles[i], NULL);
#endif

    auto end_time = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end_time - start_time).count();

    // Save tames to file
    std::cout << "\nSaving tames to " << filename << "..." << std::endl;
    g_rck_state.db.Header[0] = static_cast<u8>(Range);  // Store range in header for compatibility check

    // Need to cast away const for the C-style API
    char* fn_cstr = const_cast<char*>(filename.c_str());
    bool saved = g_rck_state.db.SaveToFile(fn_cstr);

    if (saved) {
        std::cout << "=== TAMES GENERATION COMPLETE ===" << std::endl;
        std::cout << "Tames saved: " << g_rck_state.db.GetBlockCnt() << std::endl;
        std::cout << "Total ops: 2^" << log2(static_cast<double>(g_rck_state.PntTotalOps)) << std::endl;
        std::cout << "Time: " << std::setprecision(1) << elapsed << " seconds" << std::endl;
        std::cout << "File: " << filename << std::endl;
    } else {
        std::cerr << "ERROR: Failed to save tames to " << filename << std::endl;
    }

    g_rck_state.db.Clear();
    g_rck_state.GenMode = false;  // Reset generation mode
    return saved;
}

RCKangarooResult RCKangarooManager::solve() {
    RCKangarooResult result = {};
    Ec ec;

    if (!impl_->pubkey_set) {
        std::cerr << "Target public key not set" << std::endl;
        return result;
    }

    if (g_rck_state.GpuCnt == 0) {
        std::cerr << "No GPUs initialized" << std::endl;
        return result;
    }

    int Range = range_bits;
    int DP = dp_bits;
    g_rck_state.GenMode = false;

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
    g_rck_state.PntToSolve = PntToSolve;

    // Initialize state
    g_rck_state.PntTotalOps = 0;
    g_rck_state.PntIndex = 0;
    g_rck_state.TotalErrors = 0;

#ifdef COLLIDER_PRO
    // Reset opportunistic-bloom stats and arm the hit callback for this run.
    collider::gpu::bloom::reset_stats();
    collider::gpu::bloom::set_hit_callback(bloom_hit_callback);

    if (bloom_enabled && collider::gpu::bloom::is_loaded()) {
        std::cout << "[Bloom] Opportunistic address checking enabled" << std::endl;
    }
#endif
    g_rck_state.dp_callback = dp_callback;

    g_rck_state.Solved = false;

    // Prepare jump tables
    SetRndSeed(0);
    EcInt minjump, t;
    minjump.Set(1);
    minjump.ShiftLeft(Range / 2 + 3);
    for (int i = 0; i < JMP_CNT; i++) {
        g_rck_state.EcJumps1[i].dist = minjump;
        t.RndMax(minjump);
        g_rck_state.EcJumps1[i].dist.Add(t);
        g_rck_state.EcJumps1[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE;
        g_rck_state.EcJumps1[i].p = ec.MultiplyG(g_rck_state.EcJumps1[i].dist);
    }

    minjump.Set(1);
    minjump.ShiftLeft(Range - 10);
    for (int i = 0; i < JMP_CNT; i++) {
        g_rck_state.EcJumps2[i].dist = minjump;
        t.RndMax(minjump);
        g_rck_state.EcJumps2[i].dist.Add(t);
        g_rck_state.EcJumps2[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE;
        g_rck_state.EcJumps2[i].p = ec.MultiplyG(g_rck_state.EcJumps2[i].dist);
    }

    minjump.Set(1);
    minjump.ShiftLeft(Range - 10 - 2);
    for (int i = 0; i < JMP_CNT; i++) {
        g_rck_state.EcJumps3[i].dist = minjump;
        t.RndMax(minjump);
        g_rck_state.EcJumps3[i].dist.Add(t);
        g_rck_state.EcJumps3[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE;
        g_rck_state.EcJumps3[i].p = ec.MultiplyG(g_rck_state.EcJumps3[i].dist);
    }

#ifdef _WIN32
    SetRndSeed(GetTickCount64());
#else
    SetRndSeed(std::chrono::steady_clock::now().time_since_epoch().count());
#endif

    // Set half range
    g_rck_state.Int_HalfRange.Set(1);
    g_rck_state.Int_HalfRange.ShiftLeft(Range - 1);
    g_rck_state.Pnt_HalfRange = ec.MultiplyG(g_rck_state.Int_HalfRange);

    // Prepare GPUs
    for (int i = 0; i < g_rck_state.GpuCnt; i++) {
        // v1.5: propagate the asymmetric kangaroo mode chosen by the
        // caller (pool runtime) to each per-GPU RCGpuKang BEFORE its
        // Prepare() runs. Prepare() snapshots Mode into Kparams.Mode
        // and configures TAME_ONLY / WILD_ONLY starting points + kernel
        // dispatch. Default (mode == KANG_MODE_BOTH == 0) preserves
        // upstream behavior for the standalone solver.
        g_rck_state.GpuKangs[i]->Mode = mode;
        if (!g_rck_state.GpuKangs[i]->Prepare(PntToSolve, Range, DP, g_rck_state.EcJumps1, g_rck_state.EcJumps2, g_rck_state.EcJumps3)) {
            g_rck_state.GpuKangs[i]->Failed = true;
            std::cerr << "GPU " << g_rck_state.GpuKangs[i]->CudaIndex << " Prepare failed" << std::endl;
        }
    }

    // arm herd save / load hooks on each GPU.
    // Done AFTER Prepare so KangCnt is finalized, and BEFORE the worker
    // threads start (kang_thr_proc calls Execute which is the only
    // reader of InitKangsHost / writer of SaveKangsHost). The hooks are
    // file-pointer hooks; the actual file I/O happens outside solve()
    // in request_save_on_stop / save_herd_state / load_herd_state.
    if (impl_->load_armed) {
        for (int i = 0; i < g_rck_state.GpuCnt; i++) {
            const int kc = g_rck_state.GpuKangs[i]->KangCnt;
            if (i >= static_cast<int>(impl_->load_bufs.size())) break;
            if (impl_->load_bufs[i].size() == static_cast<size_t>(kc) * 96) {
                g_rck_state.GpuKangs[i]->InitKangsHost = impl_->load_bufs[i].data();
            } else {
                std::cerr << "[!] RCKangaroo: load buffer size mismatch on GPU "
                          << i << " (expected " << (kc * 96)
                          << ", got " << impl_->load_bufs[i].size() << "). "
                             "Falling back to random init for this GPU.\n";
                g_rck_state.GpuKangs[i]->InitKangsHost = nullptr;
            }
        }
    }
    if (impl_->save_armed) {
        // Allocate save buffers sized to each GPU's KangCnt. Each GPU's
        // KangCnt depends on its mpCnt and IsOldGpu, so they may differ
        // across heterogeneous configurations; size each buffer
        // independently rather than assuming uniformity.
        impl_->save_bufs.resize(g_rck_state.GpuCnt);
        for (int i = 0; i < g_rck_state.GpuCnt; i++) {
            const int kc = g_rck_state.GpuKangs[i]->KangCnt;
            impl_->save_bufs[i].assign(static_cast<size_t>(kc) * 96, 0);
            g_rck_state.GpuKangs[i]->SaveKangsHost = impl_->save_bufs[i].data();
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

#ifdef COLLIDER_CHECKPOINT_CAPTURE
    // v1.5.5 checkpoint-replay (task #9): arm the device capture window at base
    // 0 (first checkpoint_window_size() global kangaroo indices) and publish
    // this manager + the window size so CheckNewPoints can read back each
    // captured kangaroo's chain at DP harvest. Armed AFTER Prepare (KangCnt
    // final) and BEFORE the worker threads launch, exactly like the herd-save
    // hooks above. The capture is a validated read-only side channel (RTX 3060,
    // 2026-06-04); it does not alter the walk. Disarmed after the workers join.
    enable_checkpoint_capture(0);
    g_active_capture_mgr = this;
    g_capture_window_size = checkpoint_window_size();
#endif

    // Launch worker threads
#ifdef _WIN32
    HANDLE thr_handles[MAX_GPU_CNT];
    u32 ThreadID;
    g_rck_state.ThrCnt = g_rck_state.GpuCnt;
    for (int i = 0; i < g_rck_state.GpuCnt; i++) {
        thr_handles[i] = (HANDLE)_beginthreadex(NULL, 0, kang_thr_proc,
                                                 (void*)g_rck_state.GpuKangs[i], 0, &ThreadID);
    }
#else
    pthread_t thr_handles[MAX_GPU_CNT];
    g_rck_state.ThrCnt = g_rck_state.GpuCnt;
    for (int i = 0; i < g_rck_state.GpuCnt; i++) {
        pthread_create(&thr_handles[i], NULL, kang_thr_proc, (void*)g_rck_state.GpuKangs[i]);
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
    while (!g_rck_state.Solved && !stop_flag.load()) {
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
            for (int i = 0; i < g_rck_state.GpuCnt; i++) {
                // Only query speed from GPUs that haven't failed
                if (!g_rck_state.GpuKangs[i]->Failed) {
                    speed += g_rck_state.GpuKangs[i]->GetStatsSpeed();
                }
            }
            impl_->current_speed = speed;

            if (progress_due && progress_callback) {
                if (!progress_callback(g_rck_state.PntTotalOps, g_rck_state.db.GetBlockCnt(), speed)) {
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
                std::cout << "Speed: " << speed_str << ", Err: " << g_rck_state.TotalErrors
                          << ", DPs: " << g_rck_state.db.GetBlockCnt() << "/" << est_dps_cnt
                          << std::endl;
            }
            if (stdout_due) last_stdout = now;
        }
    }

    // Stop workers
    for (int i = 0; i < g_rck_state.GpuCnt; i++)
        g_rck_state.GpuKangs[i]->Stop();
    while (g_rck_state.ThrCnt)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Close thread handles
#ifdef _WIN32
    for (int i = 0; i < g_rck_state.GpuCnt; i++)
        CloseHandle(thr_handles[i]);
#else
    for (int i = 0; i < g_rck_state.GpuCnt; i++)
        pthread_join(thr_handles[i], NULL);
#endif

#ifdef COLLIDER_CHECKPOINT_CAPTURE
    // v1.5.5: workers have joined and the main loop's final CheckNewPoints has
    // already run, so no reader of the capture statics remains. Clear them and
    // disarm the device window before returning. (solve() path only; the
    // GenMode tame-generation loop never exports DPs to the pool.)
    g_active_capture_mgr = nullptr;
    g_capture_window_size = 0;
    disable_checkpoint_capture();
#endif

    auto end_time = std::chrono::steady_clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
    result.total_ops = g_rck_state.PntTotalOps;
    result.dp_count = g_rck_state.db.GetBlockCnt();
    result.error_count = g_rck_state.TotalErrors;
    result.k_value = (double)g_rck_state.PntTotalOps / pow(2.0, Range / 2.0);

#ifdef COLLIDER_PRO
    // Copy opportunistic-bloom results out of the Pro-only bloom TU.
    result.bloom_checks = collider::gpu::bloom::checks();
    result.bloom_hits = collider::gpu::bloom::hits();

    if (bloom_enabled && result.bloom_checks > 0) {
        std::cout << "[Bloom] Total checks: " << result.bloom_checks
                  << ", Hits: " << result.bloom_hits.size() << std::endl;
    }
#endif

    if (g_rck_state.Solved) {
        // Apply start offset
        if (impl_->start_set) {
            g_rck_state.PrivKey.Add(impl_->start_offset);
        }

        // Verify solution
        EcPoint verify = ec.MultiplyG(g_rck_state.PrivKey);
        if (verify.IsEqual(impl_->target_pubkey)) {
            result.found = true;
            memcpy(result.private_key.data(), g_rck_state.PrivKey.data, 32);

            char hex[100];
            g_rck_state.PrivKey.GetHexStr(hex);
            std::cout << "\n+============================================================+\n"
                      << "|                      PUZZLE SOLVED!                        |\n"
                      << "+============================================================+\n"
                      << "PRIVATE KEY: " << hex << "\n"
                      << "K value: " << result.k_value << std::endl;
        } else {
            std::cerr << "FATAL: Collision found but key verification failed!" << std::endl;
        }
    }

    g_rck_state.db.Clear();
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
    if (g_rck_state.GpuCnt == 0) {
        std::cerr << "[!] RCKangaroo: request_save_on_stop called before "
                     "init(); ignored.\n";
        return false;
    }
    impl_->save_armed = true;
    return true;
}

bool RCKangarooManager::save_herd_state(const std::string& path) {
    if (g_rck_state.GpuCnt == 0) return false;
    if (!impl_->save_armed) {
        std::cerr << "[!] RCKangaroo: save_herd_state called without prior "
                     "request_save_on_stop(); nothing to write.\n";
        return false;
    }
    if (impl_->save_bufs.size() != static_cast<size_t>(g_rck_state.GpuCnt)) {
        // solve() never ran to completion; save buffers are not populated.
        return false;
    }

    // Validate every per-GPU buffer is properly sized. A zero-size buffer
    // indicates the GPU was marked Failed during Prepare and the patch's
    // save hook never fired.
    for (int i = 0; i < g_rck_state.GpuCnt; i++) {
        const int kc = g_rck_state.GpuKangs[i]->KangCnt;
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
    write_u32_le(hdr_nums + 4,  static_cast<uint32_t>(g_rck_state.GpuCnt));
    write_u32_le(hdr_nums + 8,  static_cast<uint32_t>(g_rck_state.GpuKangs[0]->KangCnt));
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
    for (int i = 0; i < g_rck_state.GpuCnt; i++) {
        uint8_t gpu_hdr[8];
        write_u32_le(gpu_hdr + 0, static_cast<uint32_t>(g_rck_state.GpuKangs[i]->CudaIndex));
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
    for (int i = 0; i < g_rck_state.GpuCnt; i++) {
        g_rck_state.GpuKangs[i]->SaveKangsHost = nullptr;
    }
    impl_->save_armed = false;
    return true;
}

bool RCKangarooManager::load_herd_state(const std::string& path) {
    if (g_rck_state.GpuCnt == 0) {
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
    if (file_gpus != static_cast<uint32_t>(g_rck_state.GpuCnt)) {
        std::cerr << "[!] RCKangaroo: checkpoint has " << file_gpus
                  << " GPUs but current run has " << g_rck_state.GpuCnt
                  << "; refusing to load.\n";
        std::fclose(fp); return false;
    }
    if (file_kang_cnt != static_cast<uint32_t>(g_rck_state.GpuKangs[0]->KangCnt)) {
        std::cerr << "[!] RCKangaroo: checkpoint KangCnt " << file_kang_cnt
                  << " does not match current " << g_rck_state.GpuKangs[0]->KangCnt
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
    impl_->load_bufs.assign(g_rck_state.GpuCnt, std::vector<uint8_t>{});
    for (int i = 0; i < g_rck_state.GpuCnt; i++) {
        uint8_t gpu_hdr[8];
        if (std::fread(gpu_hdr, 1, sizeof(gpu_hdr), fp) != sizeof(gpu_hdr)) {
            std::fclose(fp); return false;
        }
        // cuda_index in gpu_hdr is informational; GPU renumbering between
        // runs is tolerated (the order in the file is the order they
        // appear in g_rck_state.GpuKangs).

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

// ============================================================================
// v1.5.5 checkpoint-replay capture (task #9). HARDWARE-VALIDATED.
//
// The device-side per-kangaroo checkpoint capture is implemented in
// third_party/RCKangaroo/RCGpuCore.cu under COLLIDER_CHECKPOINT_CAPTURE:
// KernelB records each kangaroo's walk DISTANCE (signed 192-bit) + the L1S2
// loop-state bit every CHECKPOINT_INTERVAL (65536) jumps into a per-kangaroo
// device ring, for a fixed window [base, base+CHK_MAX_KANG) of global kangaroo
// indices. BuildDP tags each DP with its producing kangaroo's global index
// (byte offset 60 of the 64-byte GPU DP record). The host arms the capture
// window, and on a DP from a captured kangaroo reads back that kangaroo's
// ordered checkpoint chain.
//
// VALIDATED on an RTX 3060 (2026-06-04) by tests/test_checkpoint_capture.cu:
// for loop-free kangaroos the captured chain (distance + L1S2 bit) reproduces
// the canonical CPU walk (src/core/checkpoint_walk.hpp ==
// collision-protocol/src/checkpoint_replay.py) byte-for-byte over a full
// 65536-jump segment. Measured jmp3 loop-escape (KernelC) incidence
// ~0.17%/segment; such segments are NOT modeled by the server's pure
// jmp1/jmp2 replay, so the capture exposes a per-kangaroo loop-escape count
// and the caller MUST exclude loop-escape walks from the committed/
// challengeable set (the server runs challenge_mode="shadow", logging not
// banning, until that exclusion is soaked).

#ifdef COLLIDER_CHECKPOINT_CAPTURE
// Device-side capture helpers (RCGpuCore.cu, COLLIDER_CHECKPOINT_CAPTURE).
extern "C" void ckpt_enable_capture(unsigned int base);
extern "C" void ckpt_disable_capture();
extern "C" unsigned int ckpt_interval();
extern "C" unsigned int ckpt_max_kang();
extern "C" unsigned int ckpt_max_cp();
extern "C" unsigned int ckpt_readback_slot(unsigned int slot,
                                           unsigned long long* dist_out,
                                           unsigned char* l1s2_out,
                                           unsigned int* loopesc_out);

namespace {

// Reduce a GPU signed 192-bit distance (3 little-endian u64) mod n into a
// 32-byte big-endian Distance (the Merkle leaf / on-wire form), exactly as the
// validated test does and as the server's reconstruct_checkpoint_point expects
// (distance mod n). A "negative" birth-relative distance (top bit of the
// 192-bit word set) maps to (value - 2^192) mod n.
::collider::checkpoint_commit::Distance gpu_dist_to_be_mod_n(
        const unsigned long long d3[3]) {
    using ::collider::cpu::uint256_t;
    const uint256_t& n = ::collider::cpu::SECP256K1_N;
    const bool negative = (d3[2] >> 63) != 0;
    uint256_t v; v.d[0] = d3[0]; v.d[1] = d3[1]; v.d[2] = d3[2]; v.d[3] = 0;
    uint256_t res;
    if (!negative) {
        res = v;
        while (res >= n) { uint256_t r; ::collider::cpu::sub256(r, res, n); res = r; }
    } else {
        uint256_t two192; two192.d[0] = 0; two192.d[1] = 0; two192.d[2] = 0; two192.d[3] = 1;
        while (two192 >= n) { uint256_t r; ::collider::cpu::sub256(r, two192, n); two192 = r; }
        uint256_t vmod = v;
        while (vmod >= n) { uint256_t r; ::collider::cpu::sub256(r, vmod, n); vmod = r; }
        uint64_t borrow = ::collider::cpu::sub256(res, vmod, two192);
        if (borrow) { uint256_t r; ::collider::cpu::add256(r, res, n); res = r; }
    }
    ::collider::checkpoint_commit::Distance out{};
    for (int limb = 0; limb < 4; ++limb) {
        const uint64_t lv = res.d[limb];
        uint8_t* p = out.data() + (3 - limb) * 8;
        for (int b = 0; b < 8; ++b)
            p[b] = static_cast<uint8_t>((lv >> (8 * (7 - b))) & 0xFF);
    }
    return out;
}

}  // namespace

// Arm/disarm the device capture window. Default base 0 captures the first
// CHK_MAX_KANG global kangaroo indices; the DP tag (byte 60) tells the host
// which captured slot a DP came from.
void RCKangarooManager::enable_checkpoint_capture(uint32_t base) {
    ckpt_enable_capture(base);
}
void RCKangarooManager::disable_checkpoint_capture() {
    ckpt_disable_capture();
}

// Read back the ordered checkpoint chain for a captured kangaroo `slot`
// (0-based within the armed window). Returns true and fills `out` iff the
// kangaroo has >= 2 checkpoints (>= one full segment), the birth L1S2 bit is 0,
// and it hit ZERO loop-escapes (so every committed segment replays cleanly
// under the server's pure jmp1/jmp2 walk). A loop-escape walk returns false:
// the caller must NOT commit it. The distances are reduced mod n and encoded
// big-endian, ready for checkpoint_commit::build_root.
bool RCKangarooManager::read_checkpoint_chain(
        uint32_t slot,
        std::vector<::collider::checkpoint_commit::Distance>& out,
        std::vector<uint8_t>& l1s2_out) const {
    out.clear();
    l1s2_out.clear();
    const unsigned max_cp = ckpt_max_cp();
    std::vector<unsigned long long> dist(static_cast<size_t>(max_cp) * 3);
    std::vector<unsigned char> l1s2(max_cp);
    unsigned loopesc = 0;
    unsigned cnt = ckpt_readback_slot(slot, dist.data(), l1s2.data(), &loopesc);
    if (cnt < 2) return false;       // need birth + >= 1 full segment
    if (loopesc != 0) return false;  // loop-escape: not server-replayable
    if (l1s2[0] != 0) return false;  // birth must enter at l1s2 == 0
    out.reserve(cnt);
    l1s2_out.reserve(cnt);
    for (unsigned cp = 0; cp < cnt; ++cp) {
        out.push_back(gpu_dist_to_be_mod_n(&dist[static_cast<size_t>(cp) * 3]));
        l1s2_out.push_back(l1s2[cp]);
    }
    return true;
}

uint32_t RCKangarooManager::checkpoint_window_size() { return ckpt_max_kang(); }
uint32_t RCKangarooManager::checkpoint_interval()    { return ckpt_interval(); }
bool RCKangarooManager::checkpoint_capture_built()   { return true; }
#else
uint32_t RCKangarooManager::checkpoint_window_size() { return 0; }
uint32_t RCKangarooManager::checkpoint_interval()    { return 0; }
bool RCKangarooManager::checkpoint_capture_built()   { return false; }
#endif  // COLLIDER_CHECKPOINT_CAPTURE

}  // namespace gpu
}  // namespace collider
