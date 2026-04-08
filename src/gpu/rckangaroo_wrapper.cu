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
static EcPoint g_Pnt_NegHalfRange;  // -G*HalfRange for WILD point computation
static EcPoint g_PntA;  // PntToSolve - G*HalfRange (WILD1 base point)
static EcPoint g_PntB;  // -PntA (WILD2 base point)
static EcInt g_PrivKey;
static EcPoint g_PntToSolve;
static int g_DPBits = 0;  // Current DP bits for validation
static std::atomic<uint64_t> g_InvalidDPCount{0};  // Track invalid DPs for diagnostics
static FILE* g_InvalidDPLog = nullptr;  // Log file for invalid DPs
static std::mutex g_InvalidDPLogMutex;  // Protect log file access

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
    u8 x[12];   // Truncated X for DB collision detection (unchanged)
    u8 d[22];
    u8 type;
};
#pragma pack(pop)


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


    for (int i = 0; i < cnt; i++) {
        DBRec nrec;
        u8* p = g_pPntList2 + i * GPU_DP_SIZE;
        // GPU_DP_SIZE=48 layout: x_truncated[16] + d[24] + type[4] + pad[4]
        memcpy(nrec.x, p, 12);
        memcpy(nrec.d, p + 16, 22);
        nrec.type = g_GenMode ? TAME : p[40];


        DBRec* pref = (DBRec*)g_db.FindOrAddDataBlock((u8*)&nrec);

        // Export DP to pool via callback (if in pool mode)
        if (g_dp_callback && !g_GenMode) {
            // Reconstruct full EC point from distance to get complete X coordinate
            // GPU DP buffer only stores truncated X (128 bits), so we recompute
            EcInt dist;
            memset(dist.data, 0, sizeof(dist.data));
            memcpy(dist.data, nrec.d, sizeof(nrec.d));
            if (nrec.d[21] == 0xFF) memset(((u8*)dist.data) + 22, 0xFF, 18);

            // Use a fresh Ec context per-DP to avoid any state corruption
            Ec dp_ec;
            EcPoint dp_point;
            if (nrec.type == TAME) {
                dp_point = dp_ec.MultiplyG(dist);
            } else if (nrec.type == WILD1) {
                EcPoint dp = dp_ec.MultiplyG(dist);
                dp_point = dp_ec.AddPoints(g_PntA, dp);
            } else {
                EcPoint dp = dp_ec.MultiplyG(dist);
                dp_point = dp_ec.AddPoints(g_PntB, dp);
            }

            // Extract X coordinate in big-endian (32 bytes) for pool protocol
            uint8_t x_full[32];
            for (int j = 0; j < 32; j++) {
                x_full[j] = ((uint8_t*)dp_point.x.data)[31 - j];
            }


            // Distance in big-endian (32 bytes)
            uint8_t d_full[32] = {0};
            for (int j = 0; j < 22 && j < 32; j++) {
                d_full[31 - j] = nrec.d[j];
            }

            // Map WILD2 (type=2) to WILD (type=1) for pool protocol
            uint8_t pool_type = (nrec.type == TAME) ? 0 : 1;
            g_dp_callback(x_full, d_full, pool_type);
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
    g_DPBits = DP;  // Store for validation
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
    g_DPBits = DP;  // Store for validation
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

    g_dp_callback = dp_callback;
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
    
    // Compute WILD base points for correct X coordinate reconstruction
    // PntA = PntToSolve - G*HalfRange (WILD1 uses this as base)
    // PntB = -PntA (WILD2 uses this as base)
    g_Pnt_NegHalfRange = g_Pnt_HalfRange;
    g_Pnt_NegHalfRange.y.NegModP();  // Negate Y to get -G*HalfRange
    g_PntA = ec.AddPoints(PntToSolve, g_Pnt_NegHalfRange);  // PntToSolve - G*HalfRange
    g_PntB = g_PntA;
    g_PntB.y.NegModP();  // -PntA

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


uint64_t RCKangarooManager::get_invalid_dp_count() const {
    return g_InvalidDPCount.load();
}

bool RCKangarooManager::open_invalid_dp_log(const std::string& filename) {
    std::lock_guard<std::mutex> lock(g_InvalidDPLogMutex);
    if (g_InvalidDPLog) {
        fclose(g_InvalidDPLog);
    }
    g_InvalidDPLog = fopen(filename.c_str(), "w");
    if (g_InvalidDPLog) {
        fprintf(g_InvalidDPLog, "=== Invalid DP Log ===\n");
        fprintf(g_InvalidDPLog, "DP Bits: %d\n\n", g_DPBits);
        fflush(g_InvalidDPLog);
        return true;
    }
    return false;
}

void RCKangarooManager::close_invalid_dp_log() {
    std::lock_guard<std::mutex> lock(g_InvalidDPLogMutex);
    if (g_InvalidDPLog) {
        fclose(g_InvalidDPLog);
        g_InvalidDPLog = nullptr;
    }
}

void RCKangarooManager::reset_invalid_dp_count() {
    g_InvalidDPCount.store(0);
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
