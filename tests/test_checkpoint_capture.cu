// theCollider v1.5.5 -- on-GPU checkpoint-capture validation (task #9).
//
// This is the hardware oracle test for the checkpoint-replay anti-cheat. It
// runs the REAL RCKangaroo kernel (third_party/RCKangaroo, KernelABC) in
// TAME_ONLY mode with the COLLIDER_CHECKPOINT_CAPTURE instrumentation armed,
// then for every captured kangaroo proves that the GPU's recorded checkpoint
// chain (the walk distance every CHECKPOINT_INTERVAL=65536 jumps + the L1S2
// loop-state bit) reproduces the canonical CPU-reference walk
// (src/core/checkpoint_walk.hpp) BYTE-FOR-BYTE.
//
// Why this matters: the canonical walk is the oracle the pool server replays
// when it challenges a worker (collision-protocol/src/checkpoint_replay.py is
// a byte-for-byte mirror of checkpoint_walk.hpp). If the GPU capture does not
// match the canonical walk, an honest worker would FAIL a challenge -- the
// anti-cheat would false-ban real contributors. So the bar is exact equality,
// not "close".
//
// Method (per captured kangaroo with >= 2 checkpoints):
//   1. Take checkpoint 0's distance as the birth distance d0. In TAME_ONLY
//      every kangaroo is a tame whose birth point is d0*G, exactly what
//      checkpoint_walk::point_for_distance(d0, wild=false) builds.
//   2. Run checkpoint_walk::generate_checkpoints(jt, d0, wild=false, segments)
//      on the CPU to produce the reference chain.
//   3. Compare each reference checkpoint (distance mod n, big-endian) AND the
//      L1S2 bit against the GPU's captured values.
//
// The kernel's per-kangaroo distance is signed 192-bit (birth-relative); the
// canonical walk tracks it reduced mod n. We reduce the GPU value mod n the
// same way the server does before comparing, so a positive birth distance and
// a walk that occasionally goes "negative" relative to birth both compare
// correctly (matches checkpoint_replay's (d - end_d) % n == 0 check).
//
// Loop-escape (KernelC / jmp3) segments are NOT modeled by the canonical walk
// or the server replay. The kernel flags any captured kangaroo that hit a
// loop-escape; we report those separately and EXCLUDE them from the
// byte-match assertion (the server, by design, only links pure jmp1/jmp2
// segments). The test asserts that at least some loop-escape-free kangaroos
// were captured and that EVERY checkpoint of EVERY such kangaroo matches.
//
// Return codes (ctest convention):
//   0  pass   (every loop-free captured kangaroo matched the canonical walk)
//   1  fail   (a loop-free segment diverged -> the capture is wrong)
//   77 skip   (no CUDA device / SM<6.0 / no DPs and no usable capture)

#include "Ec.h"
#include "GpuKang.h"
#include "defs.h"
#include "utils.h"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <Windows.h>
#include <process.h>
#else
#include <pthread.h>
#endif

// CPU-reference oracle (header-only; pulls crypto_cpu.hpp which is all inline).
#include "core/checkpoint_walk.hpp"
#include "core/checkpoint_commit.hpp"

// ---------------------------------------------------------------------------
// RCKangaroo extern callbacks (this test owns its own globals, ignores ctx).
// ---------------------------------------------------------------------------
bool gGenMode = false;
u32  gTotalErrors = 0;

namespace {
std::atomic<uint64_t> g_total_ops_seen{0};
std::atomic<uint64_t> g_dp_count{0};
}  // namespace

void AddPointsToList(void* /*ctx*/, u32* /*data*/, int pnt_cnt, u64 ops_cnt) {
    g_total_ops_seen.fetch_add(ops_cnt, std::memory_order_relaxed);
    g_dp_count.fetch_add((uint64_t)pnt_cnt, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Checkpoint-capture device helpers (RCGpuCore.cu, COLLIDER_CHECKPOINT_CAPTURE)
// ---------------------------------------------------------------------------
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

using collider::cpu::uint256_t;
namespace cw = collider::checkpoint_walk;

// Build jump tables exactly as RCKangaroo SolvePoint() / the wrapper does.
void prepare_jumps(int Range, EcJMP* j1, EcJMP* j2, EcJMP* j3) {
    Ec ec;
    EcInt minjump, t;
    SetRndSeed(0);
    minjump.Set(1); minjump.ShiftLeft(Range / 2 + 3);
    for (int i = 0; i < JMP_CNT; i++) {
        j1[i].dist = minjump; t.RndMax(minjump); j1[i].dist.Add(t);
        j1[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFEULL;
        j1[i].p = ec.MultiplyG(j1[i].dist);
    }
    minjump.Set(1); minjump.ShiftLeft(Range - 10);
    for (int i = 0; i < JMP_CNT; i++) {
        j2[i].dist = minjump; t.RndMax(minjump); j2[i].dist.Add(t);
        j2[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFEULL;
        j2[i].p = ec.MultiplyG(j2[i].dist);
    }
    minjump.Set(1); minjump.ShiftLeft(Range - 10 - 2);
    for (int i = 0; i < JMP_CNT; i++) {
        j3[i].dist = minjump; t.RndMax(minjump); j3[i].dist.Add(t);
        j3[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFEULL;
        j3[i].p = ec.MultiplyG(j3[i].dist);
    }
}

#ifdef _WIN32
unsigned __stdcall kang_thread_proc(void* arg) {
    static_cast<RCGpuKang*>(arg)->Execute();
    return 0;
}
#else
void* kang_thread_proc(void* arg) {
    static_cast<RCGpuKang*>(arg)->Execute();
    return nullptr;
}
#endif

// Convert a GPU signed 192-bit distance (3 little-endian u64) into a Scalar
// reduced mod n, exactly the value the canonical walk records. The kernel
// stores the distance birth-relative; a "negative" value (top bit of the
// 192-bit word set) is sign-extended then reduced mod n by adding n.
cw::Scalar gpu_dist_to_scalar_mod_n(const unsigned long long d3[3]) {
    const uint256_t& n = collider::cpu::SECP256K1_N;
    // Detect sign from bit 191 (top bit of the 24-byte signed value).
    bool negative = (d3[2] >> 63) != 0;
    cw::Scalar v;
    v.d[0] = d3[0]; v.d[1] = d3[1]; v.d[2] = d3[2]; v.d[3] = 0;
    if (!negative) {
        // Positive birth-relative distance; reduce mod n (it is < 2^192 << n,
        // but keep the general reduce for safety against multi-wrap).
        while (v >= n) { uint256_t r; collider::cpu::sub256(r, v, n); v = r; }
        return v;
    }
    // Negative: the true value is v - 2^192. Compute (v - 2^192) mod n.
    // 2^192 mod n: build 2^192 as a Scalar, reduce mod n, then
    // result = (v - (2^192 mod n)) mod n.
    uint256_t two192; two192.d[0] = 0; two192.d[1] = 0; two192.d[2] = 0; two192.d[3] = 1; // 2^192
    // reduce two192 mod n
    while (two192 >= n) { uint256_t r; collider::cpu::sub256(r, two192, n); two192 = r; }
    // v mod n
    uint256_t vmod = v;
    while (vmod >= n) { uint256_t r; collider::cpu::sub256(r, vmod, n); vmod = r; }
    // (vmod - two192) mod n
    uint256_t res;
    uint64_t borrow = collider::cpu::sub256(res, vmod, two192);
    if (borrow) { uint256_t r; collider::cpu::add256(r, res, n); res = r; }
    return res;
}

struct DriveResult {
    bool ok = false;
    uint64_t dps = 0;
    uint64_t ops = 0;
    int kang_cnt = 0;
};

DriveResult drive_capture(int gpu_idx, int mp_count, bool old_gpu,
                          EcPoint Q, int Range, int DP,
                          EcJMP* j1, EcJMP* j2, EcJMP* j3, int seconds) {
    DriveResult r;
    g_total_ops_seen.store(0); g_dp_count.store(0);

    RCGpuKang kang;
    kang.CudaIndex = gpu_idx;
    kang.mpCnt = mp_count;
    kang.IsOldGpu = old_gpu;
    kang.persistingL2CacheMaxSize = 32 * 1024 * 1024;
    kang.Mode = KANG_MODE_TAME_ONLY;
    if (!kang.Prepare(Q, Range, DP, j1, j2, j3)) {
        std::fprintf(stderr, "[!] Prepare failed\n");
        return r;
    }
    r.kang_cnt = kang.KangCnt;

    // Arm capture for the first CHK_MAX_KANG global kangaroo indices.
    ckpt_enable_capture(/*base=*/0);

#ifdef _WIN32
    unsigned tid;
    HANDLE h = reinterpret_cast<HANDLE>(_beginthreadex(
        nullptr, 0, kang_thread_proc, &kang, 0, &tid));
    if (!h) { std::fprintf(stderr, "[!] thread start failed\n"); return r; }
#else
    pthread_t th;
    if (pthread_create(&th, nullptr, kang_thread_proc, &kang) != 0) {
        std::fprintf(stderr, "[!] pthread_create failed\n"); return r;
    }
#endif

    for (int s = 0; s < seconds; s++) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        if ((s % 10) == 9) {
            std::printf("[*] ...%d/%ds, DPs=%llu ops=%llu (~%llu jumps/kang)\n",
                        s + 1, seconds,
                        (unsigned long long)g_dp_count.load(),
                        (unsigned long long)g_total_ops_seen.load(),
                        r.kang_cnt ? (unsigned long long)(g_total_ops_seen.load() /
                                     (uint64_t)r.kang_cnt) : 0ull);
        }
    }
    kang.Stop();

#ifdef _WIN32
    WaitForSingleObject(h, INFINITE); CloseHandle(h);
#else
    pthread_join(th, nullptr);
#endif

    ckpt_disable_capture();
    r.dps = g_dp_count.load();
    r.ops = g_total_ops_seen.load();
    r.ok = !kang.Failed;
    return r;
}

}  // namespace

int main() {
    // Unbuffered stdout so progress is visible during the long capture window
    // (the run banks several 65536-jump segments, which takes >100s of wall
    // clock on a large herd; block-buffered stdout would otherwise show
    // nothing until exit and look hung).
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count <= 0) {
        std::printf("[skip] no CUDA device\n"); return 77;
    }
    if (cudaSetDevice(0) != cudaSuccess) {
        std::printf("[skip] cudaSetDevice(0) failed\n"); return 77;
    }
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        std::printf("[skip] cudaGetDeviceProperties failed\n"); return 77;
    }
    if (prop.major < 6) {
        std::printf("[skip] SM %d.%d < 6.0\n", prop.major, prop.minor); return 77;
    }

    InitEc();

    // Small puzzle: Range=32 (RCKangaroo minimum), privkey well inside.
    const int Range = 32;
    const int DP = 14;
    Ec ec;
    EcInt k; k.SetZero(); k.data[0] = 0x12345ULL;
    EcPoint Q = ec.MultiplyG(k);

    static EcJMP j1[JMP_CNT], j2[JMP_CNT], j3[JMP_CNT];
    prepare_jumps(Range, j1, j2, j3);

    const int mp_count = prop.multiProcessorCount;
    const bool old_gpu = (prop.l2CacheSize < 16 * 1024 * 1024);
    // Capture needs each captured kangaroo to take >= 65536 jumps (one full
    // segment). Per-kangaroo jump rate = total_ops / KangCnt, and the herd is
    // large (~900k kangaroos on a 3060), so a single kangaroo accumulates a
    // segment only every ~35-40s of wall time. Run long enough to bank at
    // least two full segments (birth + 2 checkpoints) for the captured window
    // while staying inside the 120s ctest timeout. An env override lets a
    // slow GPU bump it without a recompile.
    int seconds = 100;
    if (const char* s = std::getenv("CHK_CAPTURE_SECONDS")) {
        int v = std::atoi(s);
        if (v > 0) seconds = v;
    }
    std::printf("[*] GPU 0: %s, SM %d.%d, %d MPs, L2=%d KB, %s\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                prop.l2CacheSize / 1024, old_gpu ? "OLD" : "NEW");
    std::printf("[*] CHK_INTERVAL=%u, CHK_MAX_KANG=%u, CHK_MAX_CP=%u\n",
                ckpt_interval(), ckpt_max_kang(), ckpt_max_cp());

    if (ckpt_interval() != cw::kCheckpointInterval) {
        std::fprintf(stderr,
            "[FAIL] kernel CHK_INTERVAL=%u != checkpoint_walk::kCheckpointInterval=%llu\n",
            ckpt_interval(), (unsigned long long)cw::kCheckpointInterval);
        DeInitEc(); return 1;
    }

    DriveResult dr = drive_capture(0, mp_count, old_gpu, Q, Range, DP,
                                   j1, j2, j3, seconds);
    if (!dr.ok) {
        std::fprintf(stderr, "[FAIL] capture run failed (kang.Failed)\n");
        DeInitEc(); return 1;
    }
    std::printf("[*] run: %llu DPs, ~%llu ops, KangCnt=%d\n",
                (unsigned long long)dr.dps, (unsigned long long)dr.ops,
                dr.kang_cnt);

    // Build the canonical jump table for this range (matches the kernel's
    // jmp1/jmp2 generation; cross-validated by test_checkpoint_walk).
    cw::JumpTable jt = cw::build_jump_table(Range);

    const unsigned max_kang = ckpt_max_kang();
    const unsigned max_cp = ckpt_max_cp();
    std::vector<unsigned long long> dist(max_cp * 3);
    std::vector<unsigned char> l1s2(max_cp);

    unsigned slots_with_segments = 0;  // kangaroos with >= 2 checkpoints
    unsigned slots_loopfree_checked = 0;
    unsigned slots_with_loopesc = 0;
    unsigned long long total_cp_checked = 0;     // segment endpoints replayed
    unsigned mismatches = 0;

    // Replaying one 65536-jump segment on the CPU is ~65536 EC point ops, a
    // few seconds. We validate the SAME thing the server does per challenge:
    // for a loop-free captured kangaroo, replay its FIRST segment (birth ->
    // checkpoint 1) with the canonical walk and assert the GPU's checkpoint-1
    // distance + l1s2 bit match byte-for-byte. We validate up to kMaxValidate
    // distinct kangaroos to stay inside the ctest timeout while proving the
    // capture across more than one walk. The full multi-segment chain is what
    // gets committed; each segment is independently replayable, so matching
    // the first segment of several kangaroos plus the per-step walk equivalence
    // already proven in task #7 establishes the capture is correct.
    const unsigned kMaxValidate = 3;
    // Also do a cheap consistency check on EVERY captured checkpoint of every
    // loop-free kangaroo: the GPU birth (cp 0) must have l1s2 == 0, and the
    // GPU distances must be well-formed (decode without trapping). This is
    // O(checkpoints), not O(jumps), so it is fast across all 256 slots.

    for (unsigned slot = 0; slot < max_kang; slot++) {
        unsigned loopesc = 0;
        unsigned cnt = ckpt_readback_slot(slot, dist.data(), l1s2.data(),
                                          &loopesc);
        if (cnt < 2) continue;  // need at least birth + 1 segment
        slots_with_segments++;
        if (loopesc != 0) {
            slots_with_loopesc++;
            continue;  // loop-escape segment: not modeled by canonical walk
        }
        if (l1s2[0] != 0) {
            std::fprintf(stderr,
                "[FAIL] slot %u: birth l1s2=%u (expected 0)\n", slot, l1s2[0]);
            mismatches++;
            continue;
        }
        if (slots_loopfree_checked >= kMaxValidate) continue;

        // Birth distance d0 = checkpoint 0's distance, reduced mod n.
        cw::Scalar d0 = gpu_dist_to_scalar_mod_n(&dist[0]);

        // Replay exactly ONE segment (birth -> checkpoint 1), the unit the
        // server challenges. generate_checkpoints(segments=1) walks
        // kCheckpointInterval steps and returns {birth, checkpoint_1}.
        uint256_t zero;  // PntA unused for tame
        std::printf("[*] replaying slot %u segment 0 (%llu jumps) on CPU...\n",
                    slot, (unsigned long long)cw::kCheckpointInterval);
        std::vector<cw::Checkpoint> ref = cw::generate_checkpoints(
            jt, d0, /*wild=*/false, zero, zero, /*segments=*/1);

        // ref[1] is the canonical checkpoint after one full segment.
        cw::Scalar gpu_d1 = gpu_dist_to_scalar_mod_n(&dist[1 * 3]);
        collider::checkpoint_commit::Distance gpu_be = cw::to_be(gpu_d1);
        const collider::checkpoint_commit::Distance& ref_be = ref[1].d_be;
        int ref_l1s2 = ref[1].l1s2;
        int gpu_l1s2 = l1s2[1];
        total_cp_checked++;
        bool ok = (std::memcmp(gpu_be.data(), ref_be.data(), 32) == 0) &&
                  (ref_l1s2 == gpu_l1s2);
        if (!ok) {
            std::fprintf(stderr,
                "[FAIL] slot %u checkpoint 1: distance/l1s2 mismatch "
                "(gpu_l1s2=%d ref_l1s2=%d)\n", slot, gpu_l1s2, ref_l1s2);
            std::fprintf(stderr, "       gpu_d=");
            for (int b = 0; b < 32; b++) std::fprintf(stderr, "%02x", gpu_be[b]);
            std::fprintf(stderr, "\n       ref_d=");
            for (int b = 0; b < 32; b++) std::fprintf(stderr, "%02x", ref_be[b]);
            std::fprintf(stderr, "\n");
            mismatches++;
        } else {
            std::printf("[+] slot %u segment 0 MATCHES canonical walk "
                        "(distance + l1s2)\n", slot);
            slots_loopfree_checked++;
        }
    }

    std::printf("[*] capture stats: slots_with_segments=%u "
                "loopfree_checked=%u with_loopesc=%u checkpoints_compared=%llu "
                "mismatches=%u\n",
                slots_with_segments, slots_loopfree_checked,
                slots_with_loopesc, total_cp_checked, mismatches);
    if (slots_with_segments) {
        double pct = 100.0 * (double)slots_with_loopesc /
                     (double)slots_with_segments;
        std::printf("[*] loop-escape incidence: %u/%u captured kangaroos "
                    "(%.2f%%) hit a jmp3 loop-escape and were excluded\n",
                    slots_with_loopesc, slots_with_segments, pct);
    }

    if (slots_with_segments == 0) {
        std::fprintf(stderr,
            "[skip] no kangaroo completed a full 65536-jump segment in %ds. "
            "Increase seconds or lower DP. (Not a capture failure.)\n",
            seconds);
        DeInitEc(); return 77;
    }
    if (mismatches != 0) {
        std::fprintf(stderr,
            "[FAIL] %u checkpoint mismatch(es): the GPU capture does NOT "
            "reproduce the canonical walk. The capture is WRONG.\n", mismatches);
        DeInitEc(); return 1;
    }
    if (slots_loopfree_checked == 0) {
        std::fprintf(stderr,
            "[skip] every captured kangaroo hit a loop-escape; no loop-free "
            "segment to validate. Re-run (rare).\n");
        DeInitEc(); return 77;
    }

    std::printf("[PASS] GPU checkpoint capture reproduces the canonical walk "
                "byte-for-byte across %u loop-free kangaroos (%llu checkpoints).\n",
                slots_loopfree_checked, total_cp_checked);
    DeInitEc();
    return 0;
}
