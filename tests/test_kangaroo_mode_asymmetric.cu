// theCollider v1.5 -- asymmetric kangaroo mode self-test
//
// Exercises the RCGpuKang fork (third_party/RCKangaroo/GpuKang.{h,cpp}
// and RCGpuCore.cu) by running two SEPARATE instances against the SAME
// target Q:
//
//   pass 1: KANG_MODE_TAME_ONLY  -- entire herd is tame, DPs.type==TAME
//   pass 2: KANG_MODE_WILD_ONLY  -- entire herd is wild1, DPs.type==WILD1
//
// Each pass captures every DP it emits and the host-side hashtable
// collision detection is bypassed (the test never observes a private
// key in either kangaroo process). After both passes complete, the
// test simulates the pool server's role: it scans the tame x-set and
// wild x-set for matching x-coordinates and, for each match, computes
// the candidate private key from the SOTA collision formula
//
//   k = (d_tame - d_wild + HalfRange)  mod n
//
// and verifies k * G == Q. The test passes iff at least one collision
// recovers the known private key.
//
// This pins the theft-resistance design at the algorithm level:
// neither pass alone has enough data to compute the key. Only the
// (out-of-process) aggregator that sees both halves can.
//
// Return codes (ctest convention):
//   0  pass
//   1  fail (collisions found but none recovered the correct key, OR
//      mode propagation broken so emitted DP types disagree with the
//      requested mode)
//   77 skip (no CUDA device, or RCKangaroo's GPU support refused our
//      hardware: SM<6.0)

#include "Ec.h"
#include "GpuKang.h"
#include "defs.h"
#include "utils.h"

#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <Windows.h>
#include <process.h>
#else
#include <pthread.h>
#endif

// ---------------------------------------------------------------------------
// RCKangaroo extern free-function callbacks. GpuKang.cpp calls these
// with no context pointer, so we route DPs into a process-global
// vector and use a swappable "current pass" pointer between runs.
// ---------------------------------------------------------------------------
bool gGenMode = false;
u32  gTotalErrors = 0;

namespace {

struct CapturedDP {
    uint8_t x[32];   // 4 u64 little-endian, as the kernel wrote them
    uint8_t d[22];   // 22-byte two's-complement little-endian distance
    uint8_t type;    // 0 = TAME, 1 = WILD1, 2 = WILD2
};

std::vector<CapturedDP>* g_active_sink = nullptr;
std::atomic<uint64_t>    g_total_ops_seen{0};

} // namespace

// Linker-visible (called from GpuKang.cpp post-kernel-completion).
void AddPointsToList(u32* data, int pnt_cnt, u64 ops_cnt) {
    if (!g_active_sink) return;
    g_total_ops_seen.fetch_add(ops_cnt, std::memory_order_relaxed);
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(data);
    for (int i = 0; i < pnt_cnt; ++i) {
        const uint8_t* p = bytes + i * GPU_DP_SIZE;
        CapturedDP dp;
        std::memcpy(dp.x, p, 32);
        std::memcpy(dp.d, p + 32, 22);
        dp.type = p[56];
        g_active_sink->push_back(dp);
    }
}

namespace {

// Build the jump tables exactly the way SolvePoint() does in
// third_party/RCKangaroo/RCKangaroo.cpp -- same SetRndSeed(0) sequence
// so the jumps are reproducible across both passes. The choice of
// minjump per table is documented in the upstream source.
void prepare_jumps(int Range, EcJMP* j1, EcJMP* j2, EcJMP* j3) {
    Ec ec;
    EcInt minjump, t;

    SetRndSeed(0);

    minjump.Set(1);
    minjump.ShiftLeft(Range / 2 + 3);
    for (int i = 0; i < JMP_CNT; i++) {
        j1[i].dist = minjump;
        t.RndMax(minjump);
        j1[i].dist.Add(t);
        j1[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFEULL;
        j1[i].p = ec.MultiplyG(j1[i].dist);
    }

    minjump.Set(1);
    minjump.ShiftLeft(Range - 10);
    for (int i = 0; i < JMP_CNT; i++) {
        j2[i].dist = minjump;
        t.RndMax(minjump);
        j2[i].dist.Add(t);
        j2[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFEULL;
        j2[i].p = ec.MultiplyG(j2[i].dist);
    }

    minjump.Set(1);
    minjump.ShiftLeft(Range - 10 - 2);
    for (int i = 0; i < JMP_CNT; i++) {
        j3[i].dist = minjump;
        t.RndMax(minjump);
        j3[i].dist.Add(t);
        j3[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFEULL;
        j3[i].p = ec.MultiplyG(j3[i].dist);
    }
}

#ifdef _WIN32
unsigned __stdcall kang_thread_proc(void* arg) {
    RCGpuKang* k = static_cast<RCGpuKang*>(arg);
    k->Execute();
    return 0;
}
#else
void* kang_thread_proc(void* arg) {
    RCGpuKang* k = static_cast<RCGpuKang*>(arg);
    k->Execute();
    return nullptr;
}
#endif

// Hash an x-coordinate (32 bytes) for unordered_map keying.
struct XHash {
    size_t operator()(const std::array<uint8_t, 32>& a) const noexcept {
        // FNV-1a over 32 bytes; tiny, fast, and we're only seeking
        // exact-equal matches, not low collision rate.
        size_t h = 1469598103934665603ULL;
        for (uint8_t b : a) {
            h ^= b;
            h *= 1099511628211ULL;
        }
        return h;
    }
};
using XKey = std::array<uint8_t, 32>;

// Decode the 22-byte little-endian two's-complement distance back into
// an EcInt. The kernel stores at most 22 bytes (sign-extended above
// the 21st byte when negative). EcInt's data[] is 4+1 u64s; we copy
// the 22 bytes into the low 3 limbs and sign-extend if d[21]==0xFF.
EcInt decode_distance(const uint8_t d[22]) {
    EcInt out;
    out.SetZero();
    std::memcpy(out.data, d, 22);
    if (d[21] == 0xFF) {
        // Sign-extend to fill the rest of the 5 limbs.
        uint8_t* raw = reinterpret_cast<uint8_t*>(out.data);
        std::memset(raw + 22, 0xFF, sizeof(out.data) - 22);
    }
    return out;
}

// Drive one RCGpuKang instance for `seconds`, capturing DPs into
// `sink`. Returns false on any RCKangaroo-side failure.
bool run_pass(int gpu_idx,
              int mp_count,
              bool is_old_gpu,
              EcPoint Q,
              int Range,
              int DP,
              int Mode,
              EcJMP* j1, EcJMP* j2, EcJMP* j3,
              std::vector<CapturedDP>* sink,
              int seconds) {
    std::printf("[*] running pass: mode=%d, %d seconds...\n", Mode, seconds);
    g_active_sink = sink;
    g_total_ops_seen.store(0, std::memory_order_relaxed);

    RCGpuKang kang;
    kang.CudaIndex = gpu_idx;
    kang.mpCnt = mp_count;
    kang.IsOldGpu = is_old_gpu;
    kang.persistingL2CacheMaxSize = 32 * 1024 * 1024;
    kang.Mode = Mode;

    if (!kang.Prepare(Q, Range, DP, j1, j2, j3)) {
        std::fprintf(stderr, "[!] Prepare failed for mode %d\n", Mode);
        g_active_sink = nullptr;
        return false;
    }

#ifdef _WIN32
    unsigned tid;
    HANDLE h = reinterpret_cast<HANDLE>(_beginthreadex(
        nullptr, 0, kang_thread_proc, &kang, 0, &tid));
    if (!h) {
        std::fprintf(stderr, "[!] _beginthreadex failed\n");
        g_active_sink = nullptr;
        return false;
    }
#else
    pthread_t th;
    if (pthread_create(&th, nullptr, kang_thread_proc, &kang) != 0) {
        std::fprintf(stderr, "[!] pthread_create failed\n");
        g_active_sink = nullptr;
        return false;
    }
#endif

    std::this_thread::sleep_for(std::chrono::seconds(seconds));
    kang.Stop();

#ifdef _WIN32
    WaitForSingleObject(h, INFINITE);
    CloseHandle(h);
#else
    pthread_join(th, nullptr);
#endif

    std::printf("[*] pass mode=%d: captured %zu DPs, ~%llu kernel ops\n",
                Mode, sink->size(),
                static_cast<unsigned long long>(g_total_ops_seen.load()));
    g_active_sink = nullptr;
    return !kang.Failed;
}

} // namespace

int main() {
    // Pre-flight: we need at least one CUDA device. RCKangaroo's
    // hardware compatibility check (SM>=6.0) lives inside the upstream
    // InitGpus, which our test does not invoke; instead we accept any
    // device that cudaSetDevice opens. RCGpuKang::Prepare will return
    // false at runtime if the hardware can't run the kernels.
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count <= 0) {
        std::printf("[skip] no CUDA device available\n");
        return 77;
    }
    if (cudaSetDevice(0) != cudaSuccess) {
        std::printf("[skip] cudaSetDevice(0) failed\n");
        return 77;
    }
    // RCKangaroo requires SM>=6.0 (see InitGpus() in
    // third_party/RCKangaroo/RCKangaroo.cpp). Honor that gate here too
    // so older hardware skips rather than fails.
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        std::printf("[skip] cudaGetDeviceProperties failed\n");
        return 77;
    }
    if (prop.major < 6) {
        std::printf("[skip] GPU compute capability %d.%d < 6.0\n",
                    prop.major, prop.minor);
        return 77;
    }

    InitEc();

    // -----------------------------------------------------------------
    // Build a tiny puzzle: pick a privkey well within the range bits we
    // pass to RCKangaroo, compute Q = k*G. The RCKangaroo CLI rejects
    // Range < 32 (see RCKangaroo.cpp::SolvePoint), so we use Range=32.
    // -----------------------------------------------------------------
    const int Range = 32;
    const int DP    = 14;       // minimum allowed

    Ec ec;
    EcInt k;
    k.SetZero();
    k.data[0] = 0x12345ULL;     // ~17 bits, comfortably inside [0, 2^Range)
    EcPoint Q = ec.MultiplyG(k);

    char khex[200];
    k.GetHexStr(khex);
    std::printf("[*] target privkey  k = %s\n", khex);
    char qxhex[200];
    Q.x.GetHexStr(qxhex);
    std::printf("[*] target Q.x      = %s\n", qxhex);

    // HalfRange = 2^(Range-1); used in the collision formula.
    EcInt HalfRange;
    HalfRange.Set(1);
    HalfRange.ShiftLeft(Range - 1);

    // Jump tables: same as the upstream SolvePoint construction.
    static EcJMP j1[JMP_CNT], j2[JMP_CNT], j3[JMP_CNT];
    prepare_jumps(Range, j1, j2, j3);

    // Hardware shape. RCKangaroo's upstream InitGpus classifies a GPU
    // as "old" iff its L2 cache is smaller than 16 MiB (covers
    // Pascal, Volta, Turing, Ampere consumer; Ada/Hopper/Blackwell are
    // "new"). The block-size / per-group-count split keys on this; if
    // we set it wrong the KernelABC launch hits an illegal memory
    // access because the kernel's BLOCK_SIZE compile-time constant
    // disagrees with the host's BlockSize used to size the shared
    // memory buffers.
    const int gpu_idx     = 0;
    const int mp_count    = prop.multiProcessorCount;
    const bool old_gpu    = (prop.l2CacheSize < 16 * 1024 * 1024);
    const int seconds     = 15;
    std::printf("[*] GPU 0: %s, SM %d.%d, %d MPs, L2=%d KB, classified as %s\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                prop.l2CacheSize / 1024, old_gpu ? "OLD" : "NEW");

    std::vector<CapturedDP> tame_dps;
    std::vector<CapturedDP> wild_dps;
    tame_dps.reserve(64 * 1024);
    wild_dps.reserve(64 * 1024);

    if (!run_pass(gpu_idx, mp_count, old_gpu, Q, Range, DP,
                  KANG_MODE_TAME_ONLY, j1, j2, j3, &tame_dps, seconds)) {
        std::fprintf(stderr, "[FAIL] TAME_ONLY pass failed\n");
        DeInitEc();
        return 1;
    }
    if (!run_pass(gpu_idx, mp_count, old_gpu, Q, Range, DP,
                  KANG_MODE_WILD_ONLY, j1, j2, j3, &wild_dps, seconds)) {
        std::fprintf(stderr, "[FAIL] WILD_ONLY pass failed\n");
        DeInitEc();
        return 1;
    }

    // -----------------------------------------------------------------
    // Type-tag check: every DP from TAME_ONLY must be TAME, every DP
    // from WILD_ONLY must be WILD1. If the kernel-side mode propagation
    // is broken, this is where we catch it.
    // -----------------------------------------------------------------
    size_t tame_bad = 0;
    for (const auto& d : tame_dps) if (d.type != TAME) ++tame_bad;
    size_t wild_bad = 0;
    for (const auto& d : wild_dps) if (d.type != WILD1) ++wild_bad;
    if (tame_bad || wild_bad) {
        std::fprintf(stderr,
            "[FAIL] type mismatch -- tame_bad=%zu wild_bad=%zu\n",
            tame_bad, wild_bad);
        DeInitEc();
        return 1;
    }

    if (tame_dps.empty() || wild_dps.empty()) {
        std::fprintf(stderr,
            "[FAIL] empty DP set after %d s -- tame=%zu wild=%zu. Bump "
            "seconds or lower DP value if your hardware is slow.\n",
            seconds, tame_dps.size(), wild_dps.size());
        DeInitEc();
        return 1;
    }

    // -----------------------------------------------------------------
    // Cross-set collision search. Build a map of {x -> tame DP}, then
    // probe wild DPs against it. The first match that recovers the
    // correct k via the SOTA collision formula wins.
    // -----------------------------------------------------------------
    std::unordered_map<XKey, const CapturedDP*, XHash> tame_by_x;
    tame_by_x.reserve(tame_dps.size() * 2);
    for (const auto& d : tame_dps) {
        XKey key;
        std::memcpy(key.data(), d.x, 32);
        tame_by_x.emplace(key, &d);
    }

    size_t collisions = 0;
    size_t recovered = 0;
    for (const auto& w : wild_dps) {
        XKey key;
        std::memcpy(key.data(), w.x, 32);
        auto it = tame_by_x.find(key);
        if (it == tame_by_x.end()) continue;
        ++collisions;

        const CapturedDP& tdp = *it->second;
        EcInt t_dist = decode_distance(tdp.d);
        EcInt w_dist = decode_distance(w.d);

        // SOTA formula (mirror of RCKangaroo's Collision_SOTA):
        //   k_candidate = (t_dist - w_dist) + HalfRange
        // The mirror branch flips the sign of (t_dist - w_dist) first,
        // then adds HalfRange. Either may match depending on which
        // wild branch the collision came from.
        EcInt cand = t_dist;
        cand.Sub(w_dist);
        EcInt saved = cand;
        cand.Add(HalfRange);
        EcPoint Pcand = ec.MultiplyG(cand);
        if (Pcand.IsEqual(Q)) {
            ++recovered;
            char ck[200];
            cand.GetHexStr(ck);
            std::printf("[+] cross-set collision recovered k=%s\n", ck);
            break;
        }
        cand = saved;
        cand.Neg();
        cand.Add(HalfRange);
        Pcand = ec.MultiplyG(cand);
        if (Pcand.IsEqual(Q)) {
            ++recovered;
            char ck[200];
            cand.GetHexStr(ck);
            std::printf("[+] cross-set collision recovered k=%s (mirror)\n", ck);
            break;
        }
    }

    std::printf("[*] tame DPs=%zu  wild DPs=%zu  x-collisions=%zu  recovered=%zu\n",
                tame_dps.size(), wild_dps.size(), collisions, recovered);

    if (recovered == 0) {
        std::fprintf(stderr,
            "[FAIL] no cross-set collision recovered k. Either the herds "
            "did not collide in the time budget, or one side's distance "
            "encoding diverged from upstream.\n");
        DeInitEc();
        return 1;
    }

    std::printf("[PASS] asymmetric kangaroo mode verified -- TAME_ONLY and "
                "WILD_ONLY DPs collide cross-instance and the aggregator "
                "recovers the known privkey.\n");
    DeInitEc();
    return 0;
}
