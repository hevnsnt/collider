/**
 * test_phase_change_fault -- v1.4.2 deterministic reproduction harness for
 * the "minute-47" brain-wallet CUDA fault.
 *
 * Production symptom (operator-reported run):
 *   [phase_change] phase="Crypto Focus" rules=90 reload_ms=100
 *   [hit_verified] total_verified_hits=1
 *   [gpu_faulted] device_id=1 action=removed_from_enable_mask
 *   [gpu_faulted] device_id=0 action=removed_from_enable_mask
 *
 * Both GPUs faulted 15-21 ms after the first verified bloom hit, with the
 * fault landing across a phase boundary (large rule set, ~98670 rules, to
 * the 90-rule "Crypto Focus" phase). The rest of the run (7 minutes)
 * rotated rule phases without GPU work because both devices had been
 * removed from the enable mask.
 *
 * Five hypotheses motivate the five scenarios in this harness:
 *
 *   A. Pure phase-change race. The large-to-small rule transition mutates
 *      engine state via load_rules() while a prior batch's stream may still
 *      be draining. Production code does NOT issue an explicit barrier
 *      between the last in-flight batch of the prior phase and the
 *      load_rules() call of the new phase; it relies on the per-chunk
 *      sync_and_collect_matches drains. This scenario exercises that path
 *      in isolation: no bloom hit, no file I/O, just back-to-back
 *      large-to-small-to-large phase swings.
 *
 *   B. Hit handling racing the next dispatch. brainwallet_hits.txt and
 *      brainwallet.pot are created on first hit; on Windows with Defender
 *      that first-time create can stall 1-3 seconds. If any GPU work is
 *      still in flight (e.g. an async dispatch from the prior batch that
 *      the runner didn't drain before handle_bloom_hits), Windows TDR
 *      could fire on the still-running kernel. Scenario B reproduces the
 *      sequence: dispatch async, simulate hit-handling file I/O with a
 *      controllable delay, then dispatch the next batch after the phase
 *      change.
 *
 *   C. Cumulative resource pressure. The bug appeared at minute 47 of a
 *      long run; a slow leak in the per-batch path could push allocator
 *      state past a fault threshold. Scenario C runs 1000 large-to-small
 *      phase swings back-to-back with no other work, looking for a fault
 *      that only appears after sustained rule churn.
 *
 *   D. V2 multi_addr exercise. The operator's faulting run was started
 *      with `--brainwallet-v2 --resume`. V2 mode causes the runner to
 *      pass `multi_addr=true` on every process_batch_from_gpu call,
 *      which routes the fused kernel through an extra per-passphrase
 *      branch (`fused_multiaddr_extra_check`) doing 2 extra SHA256, 2
 *      extra RIPEMD160, and 2 extra bloom probes per non-matching
 *      passphrase. Scenarios A/B/C all dispatched with multi_addr=false,
 *      which is the V1 path. If the fault depends on the multi_addr
 *      branch (e.g. register pressure under that path crossing some
 *      threshold under specific load, or a stack-frame footprint in the
 *      device function that interacts with a still-in-flight prior
 *      batch), only Scenario D will reproduce it. Scenario D replays
 *      Scenario B's phase-change + hit-IO pattern with multi_addr=true.
 *
 *   E. Extended-runtime cumulative load. The operator's fault hit at
 *      minute 47 of normal scanning. Scenario C runs up to ~60 minutes
 *      depending on cycle count, but its tuning may stop short. Scenario
 *      E is a bare 50-minute wall-clock cumulative load (no hit, no IO,
 *      just large-to-small phase swings until 50 minutes elapse) to
 *      exhaust the minute-47 threshold cleanly.
 *
 * The harness drives MultiGPUBrainWallet + GPURuleEngine directly. It does
 * NOT touch BrainWalletRunner (recently restructured into a class) so the
 * production runner's pre-existing test surface stays intact. The dispatch
 * pattern below mirrors the runner's two-phase chunk loop closely enough
 * that any fault here is also a fault in the production path; the only
 * production behaviours not reproduced are TUI rendering, plugin events,
 * and the wordlist-hot-swap path (all unrelated to the GPU pipeline).
 *
 * Exit codes:
 *   0  -- All scenarios completed without a fault.
 *   77 -- ctest "skip"; CUDA unavailable or no devices.
 *   2  -- A scenario reproduced a fault. Stderr carries the smoking-gun
 *         attribution (which device, which stage, which CUDA error,
 *         which cycle).
 *   3  -- Setup error (bloom load failure, init failure, etc.); not a
 *         brainwallet fault, but the harness can't proceed.
 */

#include <cuda_runtime.h>

#include "../src/core/crypto_cpu.hpp"
#include "../src/core/secure_write.hpp"
#include "../src/gpu/brain_wallet_gpu.hpp"
#include "../src/gpu/gpu_rules.hpp"
#include "../src/gpu/gpu_fault_observer.hpp"
#include "../src/tools/utxo_bloom_builder.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// Windows rpcndr.h (pulled in transitively via <filesystem>) defines `small`
// as a macro substituting to `char`. The harness uses the identifier `small`
// nowhere visible to the compiler today, but this #undef is defence in
// depth: a later edit that uses `small` as a variable name would otherwise
// fail with a baffling "invalid combination of type specifiers" error.
#ifdef small
#undef small
#endif

namespace {

// Capture every fault report so we can attribute the cause without
// shelling out to the runtime layer's fault registry. report_fault() is
// thread-safe (relaxed atomic load on the callback slot); the callback
// itself only writes a guarded vector.
struct FaultRecord {
    int         device_id = -1;
    std::string reason;
    int         cycle = -1;
    int         batch = -1;
    std::string phase;
};
std::mutex g_fault_mutex;
std::vector<FaultRecord> g_faults;
std::atomic<int> g_current_cycle{-1};
std::atomic<int> g_current_batch{-1};
std::string      g_current_phase = "uninitialised";
std::mutex       g_phase_mutex;

void on_gpu_fault(int device_id, std::string_view reason) noexcept {
    FaultRecord rec;
    rec.device_id = device_id;
    rec.reason    = std::string(reason);
    rec.cycle     = g_current_cycle.load(std::memory_order_relaxed);
    rec.batch     = g_current_batch.load(std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(g_phase_mutex);
        rec.phase = g_current_phase;
    }
    std::lock_guard<std::mutex> lk(g_fault_mutex);
    g_faults.push_back(std::move(rec));
}

bool faults_observed() {
    std::lock_guard<std::mutex> lk(g_fault_mutex);
    return !g_faults.empty();
}

void set_phase(const std::string& phase) {
    std::lock_guard<std::mutex> lk(g_phase_mutex);
    g_current_phase = phase;
}

void dump_faults(const char* scenario_label) {
    std::lock_guard<std::mutex> lk(g_fault_mutex);
    if (g_faults.empty()) return;
    std::cerr << "\n=== FAULT REPORT (" << scenario_label << ") ===\n";
    for (const auto& f : g_faults) {
        std::cerr << "  device_id=" << f.device_id
                  << " stage=\"" << f.reason << "\""
                  << " cycle=" << f.cycle
                  << " batch=" << f.batch
                  << " phase=\"" << f.phase << "\"\n";
    }
    std::cerr << "  total_faults=" << g_faults.size() << "\n";
}

void clear_faults() {
    std::lock_guard<std::mutex> lk(g_fault_mutex);
    g_faults.clear();
}

// Synthesize a deterministic word list of the requested size. Each word is
// 4-12 ASCII bytes ('a'-'z'); deterministic seed so reruns hit identical
// inputs. We do NOT bother with anything fancy: the harness only needs to
// exercise the rule -> crypto -> bloom-probe pipeline, not test passphrase
// distributions.
std::vector<std::string> make_words(size_t count, uint64_t seed) {
    std::vector<std::string> out;
    out.reserve(count);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> len_dist(4, 12);
    std::uniform_int_distribution<int> chr_dist('a', 'z');
    for (size_t i = 0; i < count; i++) {
        int len = len_dist(rng);
        std::string w(len, 'a');
        for (int j = 0; j < len; j++) w[j] = static_cast<char>(chr_dist(rng));
        out.push_back(std::move(w));
    }
    return out;
}

// Build a tiny bloom filter populated with a few known-h160s. The contents
// do not matter for fault reproduction; we just need a valid bloom for the
// kernel to probe against.
struct TestBloom {
    std::vector<uint8_t> bytes;
    uint64_t bits = 0;
    uint64_t mask = 0;
    uint32_t hashes = 0;
    uint32_t seed   = 0;
};

TestBloom make_bloom() {
    TestBloom b;
    b.bits   = 8192;
    b.mask   = b.bits - 1;
    b.hashes = 8;
    b.seed   = 0xCAFEF00Du;
    b.bytes.assign(b.bits / 8, 0);

    // Stuff 10 arbitrary h160s into the bloom so the kernel has something
    // to probe. We use SHA256("seed_N") as filler; whether they match
    // anything in the generated wordlist is unimportant.
    for (int i = 0; i < 10; i++) {
        std::string seed_str = "filler_" + std::to_string(i);
        auto pk = collider::cpu::SHA256::hash(
            reinterpret_cast<const uint8_t*>(seed_str.data()), seed_str.size());
        auto h160 = collider::cpu::compute_hash160(pk.data());
        auto [h1, h2] = ::collider::utxo::murmurhash3_128(h160.data(), 20, b.seed);
        for (uint32_t k = 0; k < b.hashes; k++) {
            uint64_t idx = (h1 + (uint64_t)k * h2) % b.bits;
            b.bytes[idx / 8] |= static_cast<uint8_t>(1u << (idx % 8));
        }
    }
    return b;
}

// Construct rule sets of a target size. Mix of length-extending,
// duplicating, and substitution rules so the worst-case output bucket
// classifier picks a non-trivial stride. "small" sets stay at the same
// per-rule shape; "large" sets pad with no-op (`:`) entries to grow the
// count without ballooning the per-rule byte length beyond the engine's
// max_rule_bytes cap.
std::vector<std::string> build_rule_set(size_t count, bool extending) {
    std::vector<std::string> rules;
    rules.reserve(count);
    // Base seed rules that the bucket classifier will see.
    // Stride budget: every rule below classifies to bucket <= 128, so the
    // engine's initial 64-byte stride allocation is grown ONCE to 128 on
    // first load_rules() and then stable. We deliberately avoid the
    // length-multiplying rules (p2, p3, d, f, q) so the harness exercises
    // the rule COUNT transition (98670 -> 90) without conflating it with
    // a stride bucket grow / shrink. The user's reported production bug
    // crossed phases at rules counts (98670 / 90), with no evidence the
    // bucket itself was changing.
    const std::vector<std::string> seed_extending = {
        ":", "u", "l", "c", "$1", "$2", "^A", "^X", "$!",
        "$@", "$#", "$$"
    };
    const std::vector<std::string> seed_short = {
        ":", "u", "l", "c", "r", "[", "]", "T0", "T1"
    };
    const auto& seed = extending ? seed_extending : seed_short;
    for (size_t i = 0; i < count; i++) {
        rules.push_back(seed[i % seed.size()]);
    }
    return rules;
}

// Per-GPU dispatch context. One per device.
struct PerGpuState {
    int device_id = -1;
    std::unique_ptr<collider::gpu::GPURuleEngine> engine;
};

struct HarnessConfig {
    std::vector<int> device_ids;
    // Brain-wallet pipeline buffer: holds the per-batch passphrase output
    // count from the rule engine. The fused kernel walks (count * stride)
    // bytes from the rule engine's d_output_, so this must be >= the
    // engine's max_words * max_rules upper bound. Set generously below.
    size_t batch_size_per_gpu = 4'000'000;
    size_t max_passphrase_length = 64;
    // Number of input WORDS the harness ships to the rule engine per
    // dispatch. Each input word fans out into max_rules passphrases via
    // the cross-product kernel, so the per-dispatch passphrase count is
    // (words_per_engine_input * max_rules_per_engine_chunk) and must
    // stay under batch_size_per_gpu so the brain-wallet pipeline buffer
    // can hold the output.
    size_t words_per_engine_input = 4096;
    // max_rules cap on the rule engine. Production runs up to 4096 per
    // chunk (and chunks larger phases). 4096 here matches the typical
    // crypto.rule + best64.rule load.
    size_t max_rules_per_engine  = 4096;
    size_t max_word_bytes_per_engine = 1ull * 1024 * 1024;  // 1 MB packed words
    // Initial output-bucket stride for the engine's d_output_ allocation.
    // Production picks one of {16,32,64,128,256}; 128 covers our
    // build_rule_set output (every seed rule classifies to <= 128) so the
    // engine never needs to grow d_output_ inside load_rules. That keeps
    // the harness's focus on the rule-COUNT transition rather than
    // accidentally also exercising the cudaFree + cudaMalloc grow path.
    size_t output_stride_bytes = 128;
};

bool autodetect_devices(HarnessConfig& cfg) {
    int n = 0;
    cudaError_t err = cudaGetDeviceCount(&n);
    if (err != cudaSuccess || n == 0) {
        std::fprintf(stderr, "No CUDA devices: %s\n",
                     err == cudaSuccess ? "0 detected" : cudaGetErrorString(err));
        return false;
    }
    cfg.device_ids.clear();
    for (int i = 0; i < n; i++) cfg.device_ids.push_back(i);

    // Scale batch_size_per_gpu down if the smallest visible card has
    // less headroom than the default. The harness should not OOM on a
    // mid-range rig; per-device init's own halving retry handles the
    // residual.
    size_t smallest_free = SIZE_MAX;
    for (int dev : cfg.device_ids) {
        auto info = collider::gpu::get_device_mem_info(dev);
        if (info.free_bytes > 0 && info.free_bytes < smallest_free) {
            smallest_free = info.free_bytes;
        }
    }
    if (smallest_free != SIZE_MAX) {
        // Rule-engine d_output_ dominates the VRAM footprint. Cap it at
        // 30% of the smallest visible card's free bytes so the
        // brain-wallet pipeline + bloom + scratch buffers can coexist.
        // engine_buf_bytes = max_words * max_rules * output_stride_bytes.
        size_t engine_budget = smallest_free * 30 / 100;
        // Derive max words from the budget given current stride + rules.
        size_t per_slot = cfg.max_rules_per_engine * cfg.output_stride_bytes;
        if (per_slot == 0) per_slot = 1;
        size_t max_words_budget = engine_budget / per_slot;
        if (max_words_budget < cfg.words_per_engine_input) {
            cfg.words_per_engine_input = std::max<size_t>(max_words_budget, 256ull);
        }
        // Brain-wallet pipeline buffer must hold the rule engine's
        // per-dispatch fanout (words * rules passphrases). Size with a
        // 5% margin above the worst-case expected count so a stride
        // round-up cannot push the count past the buffer.
        size_t expected_passphrases = cfg.words_per_engine_input
                                     * cfg.max_rules_per_engine;
        cfg.batch_size_per_gpu = expected_passphrases + (expected_passphrases / 20);
    }
    return true;
}

// Initialize one rule engine per GPU. Returns false on any per-GPU init
// failure; the caller is expected to bail out.
bool init_engines(std::vector<PerGpuState>& gpus,
                  const HarnessConfig& cfg)
{
    gpus.resize(cfg.device_ids.size());
    for (size_t i = 0; i < cfg.device_ids.size(); i++) {
        gpus[i].device_id = cfg.device_ids[i];
        gpus[i].engine = std::make_unique<collider::gpu::GPURuleEngine>(
            collider::gpu::GPURuleEngine::Config{
                /*device_id=*/        cfg.device_ids[i],
                /*max_words=*/        cfg.words_per_engine_input,
                /*max_rules=*/        cfg.max_rules_per_engine,
                /*max_word_bytes=*/   cfg.max_word_bytes_per_engine,
                /*max_rule_bytes=*/   256ull * 1024,
                /*gpu_only_mode=*/    true,
                /*output_stride_bytes=*/ cfg.output_stride_bytes
            });
        if (!gpus[i].engine->init()) {
            std::fprintf(stderr,
                         "GPURuleEngine init failed on device %d\n",
                         cfg.device_ids[i]);
            return false;
        }
    }
    return true;
}

// Construct the brain-wallet pipeline. The MultiGPUBrainWallet type is
// strictly non-copyable AND non-movable (it owns per-GPU CUDA handles),
// so the harness builds it once via std::unique_ptr and the caller owns
// the pointer.
std::unique_ptr<collider::gpu::MultiGPUBrainWallet> make_pipeline(
    const HarnessConfig& cfg)
{
    collider::gpu::MultiGPUBrainWallet::Config c;
    c.gpu_ids = cfg.device_ids;
    c.batch_size = cfg.batch_size_per_gpu * cfg.device_ids.size();
    c.max_passphrase_length = cfg.max_passphrase_length;
    c.store_private_keys = true;
    auto bw = std::make_unique<collider::gpu::MultiGPUBrainWallet>(c);
    if (!bw->init()) return nullptr;
    return bw;
}

// Drain any sticky cudaGetLastError on every initialized device. The
// rule-engine + pipeline init paths run a number of probing operations
// (e.g. cudaFuncGetAttributes lookups, vendor-specific bloom builder
// preflight checks) that can leave the "last error" slot at
// cudaErrorInvalidValue or similar non-sticky values. apply_rules_to_words_gpu's
// pre-launch probe (gpu_rules.cpp::pre_launch_context_probe) then
// misattributes them as a poisoned-context cascade, polluting the harness
// output and masking the actual phase-change fault we're trying to repro.
// Draining the slot once after init keeps subsequent probes clean.
void drain_post_init_sticky(const std::vector<PerGpuState>& gpus) {
    for (const auto& g : gpus) {
        cudaSetDevice(g.device_id);
        (void)cudaGetLastError();
    }
}

// Run one dispatch on every GPU: load_rules, apply_rules async, fused
// brain-wallet kernel async on the same stream, then sync_and_collect.
// Mirrors the runner's two-phase chunk loop minus the work_balancer
// slicing. Returns false if any per-GPU dispatch reported !ok.
//
// multi_addr selects the V1 (false) or V2 (true) fused-kernel path. The
// production runner passes args.brainwallet_v2_mode here; the harness
// honours the same selector so Scenarios A/B/C exercise the V1 path and
// Scenario D exercises the V2 multi_addr=true path that the operator's
// faulting run was actually using.
bool run_one_dispatch(
    std::vector<PerGpuState>& gpus,
    collider::gpu::MultiGPUBrainWallet& bw,
    const std::vector<std::string>& rules,
    const std::vector<std::string>& words,
    bool multi_addr = false)
{
    // Phase 0: load rules on every engine. Production code does this
    // sequentially in the chunk loop; replicate that ordering.
    for (auto& g : gpus) {
        cudaSetDevice(g.device_id);
        if (!g.engine->load_rules(rules)) {
            std::fprintf(stderr, "load_rules failed on device %d (rules=%zu)\n",
                         g.device_id, rules.size());
            return false;
        }
    }

    // Phase 1: async issue on every GPU. Split the word list evenly across
    // GPUs (the work balancer's weighting is irrelevant for fault repro).
    const size_t words_per_gpu = words.size() / gpus.size();
    std::vector<size_t> per_gpu_counts(gpus.size(), 0);
    std::vector<bool>   per_gpu_issued(gpus.size(), false);
    for (size_t i = 0; i < gpus.size(); i++) {
        cudaSetDevice(gpus[i].device_id);
        const size_t start = i * words_per_gpu;
        const size_t end   = (i + 1 == gpus.size()) ? words.size()
                                                    : start + words_per_gpu;
        std::vector<std::string> slice(words.begin() + start,
                                       words.begin() + end);

        size_t generated = gpus[i].engine->apply_rules_to_words_gpu(slice,
                                                                    /*sync=*/false);
        per_gpu_counts[i] = generated;
        if (generated == 0) continue;

        auto issued = bw.process_batch_from_gpu(
            reinterpret_cast<const uint8_t*>(gpus[i].engine->d_output()),
            gpus[i].engine->d_output_lengths(),
            gpus[i].engine->output_stride(),
            generated,
            static_cast<int>(i),
            gpus[i].engine->get_stream(),
            /*sync=*/false,
            multi_addr
        );
        if (!issued.ok) {
            std::fprintf(stderr,
                "[harness] process_batch_from_gpu refused on device %d at issue\n",
                gpus[i].device_id);
            return false;
        }
        per_gpu_issued[i] = true;
    }

    // Phase 2: collect from every GPU. Each call syncs its own stream.
    bool any_fault = false;
    for (size_t i = 0; i < gpus.size(); i++) {
        if (!per_gpu_issued[i]) continue;
        auto result = bw.sync_and_collect_matches(
            static_cast<int>(i),
            gpus[i].engine->get_stream()
        );
        if (!result.ok) {
            any_fault = true;
        }
    }
    if (any_fault) return false;
    return true;
}

void cleanup_all(std::vector<PerGpuState>& gpus,
                 std::unique_ptr<collider::gpu::MultiGPUBrainWallet>& bw)
{
    if (bw) bw->cleanup();
    bw.reset();
    for (auto& g : gpus) {
        if (g.engine) g.engine->cleanup();
    }
    gpus.clear();
}

// --------------------------------------------------------------------------
// Scenario A: large-to-small phase change without a simulated hit.
// --------------------------------------------------------------------------
bool scenario_a(const HarnessConfig& cfg,
                size_t large_count,
                size_t small_count,
                size_t cycles,
                size_t batches_per_side)
{
    std::printf("\n--- Scenario A | large=%zu small=%zu cycles=%zu batches=%zu ---\n",
                large_count, small_count, cycles, batches_per_side);

    std::vector<PerGpuState> gpus;
    if (!init_engines(gpus, cfg)) return false;
    auto bw = make_pipeline(cfg);
    if (!bw) {
        cleanup_all(gpus, bw);
        return false;
    }
    TestBloom bloom = make_bloom();
    if (!bw->load_bloom_filter(bloom.bytes.data(), bloom.bytes.size(),
                               bloom.bits, bloom.hashes, bloom.seed,
                               /*use_texture=*/false))
    {
        std::fprintf(stderr, "load_bloom_filter failed\n");
        cleanup_all(gpus, bw);
        return false;
    }
    drain_post_init_sticky(gpus);

    auto rules_large = build_rule_set(large_count, /*extending=*/true);
    auto rules_small = build_rule_set(small_count, /*extending=*/false);
    auto words = make_words(cfg.words_per_engine_input * cfg.device_ids.size(), /*seed=*/0xA1A1A1A1ull);

    // Mutator that returns a fresh-content rule subset of the requested
    // size every call. Production's chunked dispatch path loads a
    // DIFFERENT 4096-rule slice on each chunk of an oversize phase; the
    // harness reproduces that by rotating the start offset each batch so
    // subsequent load_rules() calls actually copy different bytes into
    // d_rules_. Same total count, different content, exercising the
    // cudaMemcpyAsync + cudaStreamSynchronize chain in load_rules() under
    // realistic churn.
    std::vector<std::string> rotation_pool;
    rotation_pool.reserve(large_count * 2);
    {
        auto seg1 = build_rule_set(large_count, /*extending=*/true);
        auto seg2 = build_rule_set(large_count, /*extending=*/false);
        rotation_pool.insert(rotation_pool.end(), seg1.begin(), seg1.end());
        rotation_pool.insert(rotation_pool.end(), seg2.begin(), seg2.end());
    }
    auto rotated_large = [&](size_t batch_idx) {
        size_t off = batch_idx % rotation_pool.size();
        std::vector<std::string> out;
        out.reserve(large_count);
        for (size_t i = 0; i < large_count; i++) {
            out.push_back(rotation_pool[(off + i) % rotation_pool.size()]);
        }
        return out;
    };

    for (size_t cyc = 0; cyc < cycles; cyc++) {
        g_current_cycle.store(static_cast<int>(cyc), std::memory_order_relaxed);

        // Large phase
        set_phase("Large");
        for (size_t b = 0; b < batches_per_side; b++) {
            g_current_batch.store(static_cast<int>(b), std::memory_order_relaxed);
            auto chunk_rules = rotated_large(cyc * batches_per_side + b);
            if (!run_one_dispatch(gpus, *bw, chunk_rules, words)) {
                std::fprintf(stderr, "[A] fault in large-phase batch %zu cycle %zu\n",
                             b, cyc);
                cleanup_all(gpus, bw);
                return false;
            }
            if (faults_observed()) {
                std::fprintf(stderr, "[A] observer reported fault during large batch %zu cycle %zu\n",
                             b, cyc);
                cleanup_all(gpus, bw);
                return false;
            }
        }

        // Phase change: large -> small
        set_phase("PhaseChange_LtoS");
        // Small phase
        set_phase("Small");
        for (size_t b = 0; b < batches_per_side; b++) {
            g_current_batch.store(static_cast<int>(b), std::memory_order_relaxed);
            if (!run_one_dispatch(gpus, *bw, rules_small, words)) {
                std::fprintf(stderr, "[A] fault in small-phase batch %zu cycle %zu\n",
                             b, cyc);
                cleanup_all(gpus, bw);
                return false;
            }
            if (faults_observed()) {
                std::fprintf(stderr, "[A] observer reported fault during small batch %zu cycle %zu\n",
                             b, cyc);
                cleanup_all(gpus, bw);
                return false;
            }
        }
        set_phase("PhaseChange_StoL");
    }
    cleanup_all(gpus, bw);
    return true;
}

// --------------------------------------------------------------------------
// Scenario B: large-to-small phase change + simulated hit handling delay.
// The simulated hit-handler opens two files via secure_open_ofstream (one
// for hits.txt, one for .pot), writes a stub line to each, sleeps for the
// configured delay (to model Defender first-time-create on Windows), then
// the next dispatch is issued.
// --------------------------------------------------------------------------
bool scenario_b(const HarnessConfig& cfg,
                size_t large_count,
                size_t small_count,
                size_t cycles,
                size_t batches_per_side,
                std::chrono::milliseconds io_delay)
{
    std::printf("\n--- Scenario B | large=%zu small=%zu cycles=%zu batches=%zu io_delay_ms=%lld ---\n",
                large_count, small_count, cycles, batches_per_side,
                static_cast<long long>(io_delay.count()));

    std::vector<PerGpuState> gpus;
    if (!init_engines(gpus, cfg)) return false;
    auto bw = make_pipeline(cfg);
    if (!bw) {
        cleanup_all(gpus, bw);
        return false;
    }
    TestBloom bloom = make_bloom();
    if (!bw->load_bloom_filter(bloom.bytes.data(), bloom.bytes.size(),
                               bloom.bits, bloom.hashes, bloom.seed,
                               /*use_texture=*/false))
    {
        std::fprintf(stderr, "load_bloom_filter failed\n");
        cleanup_all(gpus, bw);
        return false;
    }
    drain_post_init_sticky(gpus);

    auto rules_large = build_rule_set(large_count, /*extending=*/true);
    auto rules_small = build_rule_set(small_count, /*extending=*/false);
    auto words = make_words(cfg.words_per_engine_input * cfg.device_ids.size(), /*seed=*/0xB2B2B2B2ull);

    // Use a per-process temp directory for the simulated hit / pot files
    // so concurrent test runs do not collide. Files are recreated on
    // every cycle to model the "first-time create" Defender behaviour.
    // Per-process temp dir keyed on steady_clock so concurrent runs do
    // not collide. We avoid Windows-specific GetCurrentProcessId() so
    // the harness stays portable to a Linux CUDA build path.
    const auto pid_like = std::chrono::steady_clock::now().time_since_epoch().count();
    auto temp_dir = std::filesystem::temp_directory_path() /
                    ("phase_change_fault_" + std::to_string(pid_like));
    std::error_code mkdir_ec;
    std::filesystem::create_directories(temp_dir, mkdir_ec);

    auto simulate_hit_io = [&](size_t cyc, size_t b) {
        auto hits_path = temp_dir /
            ("brainwallet_hits_" + std::to_string(cyc) + "_" +
             std::to_string(b) + ".txt");
        auto pot_path  = temp_dir /
            ("brainwallet_pot_" + std::to_string(cyc) + "_" +
             std::to_string(b) + ".pot");
        {
            std::ofstream hf = ::collider::secure_open_ofstream(
                hits_path,
                std::ios::out | std::ios::trunc,
                ::collider::SecureWriteOnFailure::FailHard);
            if (hf) {
                hf << "simulated hit line cycle=" << cyc << " batch=" << b << "\n";
            }
        }
        {
            std::ofstream pf = ::collider::secure_open_ofstream(
                pot_path,
                std::ios::out | std::ios::trunc,
                ::collider::SecureWriteOnFailure::FailHard);
            if (pf) {
                pf << "simulated pot line cycle=" << cyc << " batch=" << b << "\n";
            }
        }
        std::cout << "[simulated_hit] cycle=" << cyc << " batch=" << b << "\n";
        if (io_delay.count() > 0) std::this_thread::sleep_for(io_delay);
    };

    // Same rotation trick as scenario A so load_rules() inside the large
    // phase actually copies different bytes each batch.
    std::vector<std::string> rotation_pool;
    rotation_pool.reserve(large_count * 2);
    {
        auto seg1 = build_rule_set(large_count, /*extending=*/true);
        auto seg2 = build_rule_set(large_count, /*extending=*/false);
        rotation_pool.insert(rotation_pool.end(), seg1.begin(), seg1.end());
        rotation_pool.insert(rotation_pool.end(), seg2.begin(), seg2.end());
    }
    auto rotated_large = [&](size_t batch_idx) {
        size_t off = batch_idx % rotation_pool.size();
        std::vector<std::string> out;
        out.reserve(large_count);
        for (size_t i = 0; i < large_count; i++) {
            out.push_back(rotation_pool[(off + i) % rotation_pool.size()]);
        }
        return out;
    };

    for (size_t cyc = 0; cyc < cycles; cyc++) {
        g_current_cycle.store(static_cast<int>(cyc), std::memory_order_relaxed);

        set_phase("Large");
        for (size_t b = 0; b < batches_per_side; b++) {
            g_current_batch.store(static_cast<int>(b), std::memory_order_relaxed);
            auto chunk_rules = rotated_large(cyc * batches_per_side + b);
            if (!run_one_dispatch(gpus, *bw, chunk_rules, words)) {
                std::fprintf(stderr, "[B] fault in large-phase batch %zu cycle %zu\n",
                             b, cyc);
                cleanup_all(gpus, bw);
                return false;
            }
        }
        // Simulated bloom-hit handler delay BEFORE the phase change. This
        // matches the production sequence: handle_bloom_hits runs at the
        // end of a batch, the loop returns to the top, and the
        // phase_changed() check fires.
        simulate_hit_io(cyc, /*b=*/0);
        if (faults_observed()) {
            std::fprintf(stderr, "[B] observer reported fault after hit IO cycle %zu\n", cyc);
            cleanup_all(gpus, bw);
            return false;
        }

        set_phase("Small");
        for (size_t b = 0; b < batches_per_side; b++) {
            g_current_batch.store(static_cast<int>(b), std::memory_order_relaxed);
            if (!run_one_dispatch(gpus, *bw, rules_small, words)) {
                std::fprintf(stderr, "[B] fault in small-phase batch %zu cycle %zu\n",
                             b, cyc);
                cleanup_all(gpus, bw);
                return false;
            }
        }
    }

    // Best-effort cleanup of the temp dir.
    std::error_code rm_ec;
    std::filesystem::remove_all(temp_dir, rm_ec);

    cleanup_all(gpus, bw);
    return true;
}

// --------------------------------------------------------------------------
// Scenario C: cumulative load test. Pure large-to-small phase swings, no
// hit handling, run for `cycles` cycles. The bug appeared at minute 47, so
// the goal here is to push past the same wall-clock duration; 1000 cycles
// of large+small at ~50ms-1s each gives ~10-60 minutes depending on rig.
// --------------------------------------------------------------------------
bool scenario_c(const HarnessConfig& cfg,
                size_t large_count,
                size_t small_count,
                size_t cycles)
{
    std::printf("\n--- Scenario C | large=%zu small=%zu cycles=%zu (cumulative load) ---\n",
                large_count, small_count, cycles);

    std::vector<PerGpuState> gpus;
    if (!init_engines(gpus, cfg)) return false;
    auto bw = make_pipeline(cfg);
    if (!bw) {
        cleanup_all(gpus, bw);
        return false;
    }
    TestBloom bloom = make_bloom();
    if (!bw->load_bloom_filter(bloom.bytes.data(), bloom.bytes.size(),
                               bloom.bits, bloom.hashes, bloom.seed,
                               /*use_texture=*/false))
    {
        std::fprintf(stderr, "load_bloom_filter failed\n");
        cleanup_all(gpus, bw);
        return false;
    }
    drain_post_init_sticky(gpus);

    auto rules_large = build_rule_set(large_count, /*extending=*/true);
    auto rules_small = build_rule_set(small_count, /*extending=*/false);
    auto words = make_words(cfg.words_per_engine_input * cfg.device_ids.size(), /*seed=*/0xC3C3C3C3ull);

    auto t0 = std::chrono::steady_clock::now();
    for (size_t cyc = 0; cyc < cycles; cyc++) {
        g_current_cycle.store(static_cast<int>(cyc), std::memory_order_relaxed);
        g_current_batch.store(0, std::memory_order_relaxed);
        set_phase("Large");
        if (!run_one_dispatch(gpus, *bw, rules_large, words)) {
            std::fprintf(stderr, "[C] fault in large phase at cycle %zu\n", cyc);
            cleanup_all(gpus, bw);
            return false;
        }
        set_phase("Small");
        if (!run_one_dispatch(gpus, *bw, rules_small, words)) {
            std::fprintf(stderr, "[C] fault in small phase at cycle %zu\n", cyc);
            cleanup_all(gpus, bw);
            return false;
        }
        if (faults_observed()) {
            std::fprintf(stderr, "[C] observer reported fault at cycle %zu\n", cyc);
            cleanup_all(gpus, bw);
            return false;
        }
        if ((cyc % 25) == 0) {
            auto el = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - t0).count();
            std::printf("[C] cycle %zu / %zu (%lld s elapsed)\n",
                        cyc, cycles, static_cast<long long>(el));
            std::fflush(stdout);
        }
    }
    cleanup_all(gpus, bw);
    return true;
}

// --------------------------------------------------------------------------
// Scenario D: V2 multi_addr exercise.
//
// Same shape as Scenario B (large-to-small phase change with a simulated
// hit-IO delay inserted at the phase boundary) but every
// process_batch_from_gpu call uses multi_addr=true. This is the production
// path the operator's faulting `--brainwallet-v2 --resume` run was on.
// The fused kernel branches into fused_multiaddr_extra_check for every
// passphrase whose primary hash160 misses, adding 2 SHA256 + 2 RIPEMD160
// + 2 bloom probes per miss. Register / stack footprint and bloom probe
// pressure both increase.
//
// Defaults are deliberately closer to the operator's production config
// than Scenarios A/B: 2 GPUs forced (skipped if only 1 visible),
// large_count default 4096 (the engine's per-chunk cap; production runs
// 98670 rules but chunks them down to <=4096 per dispatch, so 4096 is
// the largest *load_rules* size production ever passes), small_count 90
// matching "Crypto Focus" exactly, and 10 cycles minimum.
// --------------------------------------------------------------------------
bool scenario_d(const HarnessConfig& cfg,
                size_t large_count,
                size_t small_count,
                size_t cycles,
                size_t batches_per_side,
                std::chrono::milliseconds io_delay)
{
    if (cfg.device_ids.size() < 2) {
        std::printf("\n--- Scenario D | SKIPPED: requires 2 GPUs, found %zu ---\n",
                    cfg.device_ids.size());
        return true;
    }
    std::printf("\n--- Scenario D | V2 multi_addr | large=%zu small=%zu cycles=%zu batches=%zu io_delay_ms=%lld ---\n",
                large_count, small_count, cycles, batches_per_side,
                static_cast<long long>(io_delay.count()));

    std::vector<PerGpuState> gpus;
    if (!init_engines(gpus, cfg)) return false;
    auto bw = make_pipeline(cfg);
    if (!bw) {
        cleanup_all(gpus, bw);
        return false;
    }
    TestBloom bloom = make_bloom();
    if (!bw->load_bloom_filter(bloom.bytes.data(), bloom.bytes.size(),
                               bloom.bits, bloom.hashes, bloom.seed,
                               /*use_texture=*/false))
    {
        std::fprintf(stderr, "load_bloom_filter failed\n");
        cleanup_all(gpus, bw);
        return false;
    }
    drain_post_init_sticky(gpus);

    auto rules_large = build_rule_set(large_count, /*extending=*/true);
    auto rules_small = build_rule_set(small_count, /*extending=*/false);
    auto words = make_words(cfg.words_per_engine_input * cfg.device_ids.size(),
                            /*seed=*/0xD4D4D4D4ull);

    const auto pid_like = std::chrono::steady_clock::now().time_since_epoch().count();
    auto temp_dir = std::filesystem::temp_directory_path() /
                    ("phase_change_fault_d_" + std::to_string(pid_like));
    std::error_code mkdir_ec;
    std::filesystem::create_directories(temp_dir, mkdir_ec);

    auto simulate_hit_io = [&](size_t cyc, size_t b) {
        auto hits_path = temp_dir /
            ("brainwallet_hits_" + std::to_string(cyc) + "_" +
             std::to_string(b) + ".txt");
        auto pot_path  = temp_dir /
            ("brainwallet_pot_" + std::to_string(cyc) + "_" +
             std::to_string(b) + ".pot");
        {
            std::ofstream hf = ::collider::secure_open_ofstream(
                hits_path,
                std::ios::out | std::ios::trunc,
                ::collider::SecureWriteOnFailure::FailHard);
            if (hf) {
                hf << "simulated v2 hit line cycle=" << cyc << " batch=" << b << "\n";
            }
        }
        {
            std::ofstream pf = ::collider::secure_open_ofstream(
                pot_path,
                std::ios::out | std::ios::trunc,
                ::collider::SecureWriteOnFailure::FailHard);
            if (pf) {
                pf << "simulated v2 pot line cycle=" << cyc << " batch=" << b << "\n";
            }
        }
        std::cout << "[simulated_hit_v2] cycle=" << cyc << " batch=" << b << "\n";
        if (io_delay.count() > 0) std::this_thread::sleep_for(io_delay);
    };

    std::vector<std::string> rotation_pool;
    rotation_pool.reserve(large_count * 2);
    {
        auto seg1 = build_rule_set(large_count, /*extending=*/true);
        auto seg2 = build_rule_set(large_count, /*extending=*/false);
        rotation_pool.insert(rotation_pool.end(), seg1.begin(), seg1.end());
        rotation_pool.insert(rotation_pool.end(), seg2.begin(), seg2.end());
    }
    auto rotated_large = [&](size_t batch_idx) {
        size_t off = batch_idx % rotation_pool.size();
        std::vector<std::string> out;
        out.reserve(large_count);
        for (size_t i = 0; i < large_count; i++) {
            out.push_back(rotation_pool[(off + i) % rotation_pool.size()]);
        }
        return out;
    };

    auto t0 = std::chrono::steady_clock::now();
    for (size_t cyc = 0; cyc < cycles; cyc++) {
        g_current_cycle.store(static_cast<int>(cyc), std::memory_order_relaxed);

        set_phase("V2_Large");
        for (size_t b = 0; b < batches_per_side; b++) {
            g_current_batch.store(static_cast<int>(b), std::memory_order_relaxed);
            auto chunk_rules = rotated_large(cyc * batches_per_side + b);
            if (!run_one_dispatch(gpus, *bw, chunk_rules, words,
                                  /*multi_addr=*/true)) {
                std::fprintf(stderr,
                    "[D] fault in V2 large-phase batch %zu cycle %zu\n",
                    b, cyc);
                cleanup_all(gpus, bw);
                return false;
            }
            if (faults_observed()) {
                std::fprintf(stderr,
                    "[D] observer reported fault during V2 large batch %zu cycle %zu\n",
                    b, cyc);
                cleanup_all(gpus, bw);
                return false;
            }
        }
        // Simulated bloom-hit handler delay before the phase change. Same
        // ordering as Scenario B: hit handler runs at end of last large
        // batch, then the loop returns to top and the phase change fires.
        simulate_hit_io(cyc, /*b=*/0);
        if (faults_observed()) {
            std::fprintf(stderr,
                "[D] observer reported fault after V2 hit IO cycle %zu\n", cyc);
            cleanup_all(gpus, bw);
            return false;
        }

        set_phase("V2_Small");
        for (size_t b = 0; b < batches_per_side; b++) {
            g_current_batch.store(static_cast<int>(b), std::memory_order_relaxed);
            if (!run_one_dispatch(gpus, *bw, rules_small, words,
                                  /*multi_addr=*/true)) {
                std::fprintf(stderr,
                    "[D] fault in V2 small-phase batch %zu cycle %zu\n",
                    b, cyc);
                cleanup_all(gpus, bw);
                return false;
            }
            if (faults_observed()) {
                std::fprintf(stderr,
                    "[D] observer reported fault during V2 small batch %zu cycle %zu\n",
                    b, cyc);
                cleanup_all(gpus, bw);
                return false;
            }
        }
        auto el = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - t0).count();
        std::printf("[D] cycle %zu / %zu (%lld s elapsed)\n",
                    cyc, cycles, static_cast<long long>(el));
        std::fflush(stdout);
    }

    std::error_code rm_ec;
    std::filesystem::remove_all(temp_dir, rm_ec);

    cleanup_all(gpus, bw);
    return true;
}

// --------------------------------------------------------------------------
// Scenario E: extended-runtime cumulative load.
//
// The operator hit the fault at minute 47 of a brainwallet run. Scenario
// E runs Scenario A's bare large-to-small phase swings (no hit IO, no
// V2 path, no rule-content rotation beyond what A already does) for AT
// LEAST 50 wall-clock minutes. The loop terminates on either the
// configured cycle ceiling OR the 50-minute deadline, whichever comes
// first. Cycle ceiling defaults to 200000 (a safety upper bound that
// will not be reached in 50 minutes at any plausible cycle latency).
//
// Tagged with --only-e in the argument parser for standalone runs.
// --------------------------------------------------------------------------
bool scenario_e(const HarnessConfig& cfg,
                size_t large_count,
                size_t small_count,
                std::chrono::seconds wall_clock_budget,
                size_t cycle_ceiling)
{
    std::printf("\n--- Scenario E | extended cumulative | large=%zu small=%zu budget=%llds ceiling=%zu ---\n",
                large_count, small_count,
                static_cast<long long>(wall_clock_budget.count()),
                cycle_ceiling);

    std::vector<PerGpuState> gpus;
    if (!init_engines(gpus, cfg)) return false;
    auto bw = make_pipeline(cfg);
    if (!bw) {
        cleanup_all(gpus, bw);
        return false;
    }
    TestBloom bloom = make_bloom();
    if (!bw->load_bloom_filter(bloom.bytes.data(), bloom.bytes.size(),
                               bloom.bits, bloom.hashes, bloom.seed,
                               /*use_texture=*/false))
    {
        std::fprintf(stderr, "load_bloom_filter failed\n");
        cleanup_all(gpus, bw);
        return false;
    }
    drain_post_init_sticky(gpus);

    auto rules_large = build_rule_set(large_count, /*extending=*/true);
    auto rules_small = build_rule_set(small_count, /*extending=*/false);
    auto words = make_words(cfg.words_per_engine_input * cfg.device_ids.size(),
                            /*seed=*/0xE5E5E5E5ull);

    const auto t0 = std::chrono::steady_clock::now();
    const auto deadline = t0 + wall_clock_budget;

    size_t cyc = 0;
    for (; cyc < cycle_ceiling; cyc++) {
        if (std::chrono::steady_clock::now() >= deadline) {
            std::printf("[E] wall-clock budget reached at cycle %zu\n", cyc);
            break;
        }
        g_current_cycle.store(static_cast<int>(cyc), std::memory_order_relaxed);
        g_current_batch.store(0, std::memory_order_relaxed);

        set_phase("E_Large");
        if (!run_one_dispatch(gpus, *bw, rules_large, words,
                              /*multi_addr=*/false)) {
            std::fprintf(stderr, "[E] fault in large phase at cycle %zu\n", cyc);
            cleanup_all(gpus, bw);
            return false;
        }
        set_phase("E_Small");
        if (!run_one_dispatch(gpus, *bw, rules_small, words,
                              /*multi_addr=*/false)) {
            std::fprintf(stderr, "[E] fault in small phase at cycle %zu\n", cyc);
            cleanup_all(gpus, bw);
            return false;
        }
        if (faults_observed()) {
            std::fprintf(stderr, "[E] observer reported fault at cycle %zu\n", cyc);
            cleanup_all(gpus, bw);
            return false;
        }
        if ((cyc % 100) == 0) {
            auto el = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - t0).count();
            std::printf("[E] cycle %zu (%lld s elapsed, budget %llds)\n",
                        cyc, static_cast<long long>(el),
                        static_cast<long long>(wall_clock_budget.count()));
            std::fflush(stdout);
        }
    }

    auto el = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - t0).count();
    std::printf("[E] completed %zu cycles in %llds (budget %llds)\n",
                cyc, static_cast<long long>(el),
                static_cast<long long>(wall_clock_budget.count()));
    cleanup_all(gpus, bw);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::fprintf(stderr, "No CUDA devices: %s\n",
                     err == cudaSuccess ? "none" : cudaGetErrorString(err));
        return 77;
    }

    // Argument parser: simple flag handling for cycle / scenario selection
    // so the harness can be tuned without a rebuild. Default config is
    // tuned to land under the ctest 30-minute timeout on a 2-GPU rig.
    //
    // Scenario D (V2 multi_addr) is opt-in by default to keep the
    // ctest default-run wall clock predictable; --with-d or --only-d
    // enables it. Scenario E (50-minute extended cumulative load) is
    // opt-in only (--with-e or --only-e) because it exceeds the
    // ctest TIMEOUT and is intended for standalone manual runs.
    bool run_a = true, run_b = true, run_c = false;
    bool run_d = false, run_e = false;
    size_t a_cycles = 5;
    size_t b_cycles = 3;
    size_t c_cycles = 200;
    size_t d_cycles = 10;     // operator's requested minimum for V2 repro
    size_t batches_per_side = 5;
    size_t large_count = 4096;
    size_t small_count = 90;
    long long io_delay_ms = 500;
    long long e_budget_s = 50 * 60;   // 50 minutes default
    size_t e_ceiling = 200000;        // safety upper bound on cycles
    bool grow_mode = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto get_next = [&](size_t& out) {
            if (i + 1 < argc) out = static_cast<size_t>(std::stoull(argv[++i]));
        };
        auto get_next_ll = [&](long long& out) {
            if (i + 1 < argc) out = std::stoll(argv[++i]);
        };
        if (a == "--skip-a") run_a = false;
        else if (a == "--skip-b") run_b = false;
        else if (a == "--with-c") run_c = true;
        else if (a == "--with-d") run_d = true;
        else if (a == "--with-e") run_e = true;
        else if (a == "--only-a") { run_a = true; run_b = false; run_c = false; run_d = false; run_e = false; }
        else if (a == "--only-b") { run_a = false; run_b = true; run_c = false; run_d = false; run_e = false; }
        else if (a == "--only-c") { run_a = false; run_b = false; run_c = true; run_d = false; run_e = false; }
        else if (a == "--only-d") { run_a = false; run_b = false; run_c = false; run_d = true; run_e = false; }
        else if (a == "--only-e") { run_a = false; run_b = false; run_c = false; run_d = false; run_e = true; }
        else if (a == "--a-cycles") get_next(a_cycles);
        else if (a == "--b-cycles") get_next(b_cycles);
        else if (a == "--c-cycles") get_next(c_cycles);
        else if (a == "--d-cycles") get_next(d_cycles);
        else if (a == "--batches")  get_next(batches_per_side);
        else if (a == "--large")    get_next(large_count);
        else if (a == "--small")    get_next(small_count);
        else if (a == "--io-delay") get_next_ll(io_delay_ms);
        else if (a == "--e-budget-s") get_next_ll(e_budget_s);
        else if (a == "--e-ceiling")  get_next(e_ceiling);
        else if (a == "--grow")     grow_mode = true;
    }

    // Install fault observer BEFORE any GPU work so a fault during
    // pipeline init is captured.
    collider::gpu::set_fault_callback(on_gpu_fault);

    HarnessConfig cfg;
    if (!autodetect_devices(cfg)) return 77;

    // If --grow is set, walk the large rule count UP (32768, 65536, 98670)
    // and shrink-and-grow alternations through 1, 10, 90, 1000. Otherwise
    // run a single (large_count, small_count) pair.
    std::vector<std::pair<size_t, size_t>> pairs;
    if (grow_mode) {
        pairs = {
            {32768, 1}, {32768, 10}, {32768, 90}, {32768, 1000},
            // 65536 and 98670 exceed many engines' default max_rules. The
            // shrink side stays the same; the grow side exercises the
            // F4 bucket grow path.
        };
    } else {
        pairs.emplace_back(large_count, small_count);
    }

    int rc = 0;
    auto t_start = std::chrono::steady_clock::now();

    for (const auto& [L, S] : pairs) {
        if (rc != 0) break;
        if (run_a) {
            clear_faults();
            bool ok = scenario_a(cfg, L, S, a_cycles, batches_per_side);
            if (!ok || faults_observed()) {
                dump_faults("Scenario A");
                rc = 2;
                break;
            }
            std::printf("[A] PASS L=%zu S=%zu (%zu cycles)\n", L, S, a_cycles);
        }
        if (run_b) {
            clear_faults();
            bool ok = scenario_b(cfg, L, S, b_cycles, batches_per_side,
                                 std::chrono::milliseconds(io_delay_ms));
            if (!ok || faults_observed()) {
                dump_faults("Scenario B");
                rc = 2;
                break;
            }
            std::printf("[B] PASS L=%zu S=%zu (%zu cycles, %lld ms io_delay)\n",
                        L, S, b_cycles, io_delay_ms);
        }
        if (run_c) {
            clear_faults();
            bool ok = scenario_c(cfg, L, S, c_cycles);
            if (!ok || faults_observed()) {
                dump_faults("Scenario C");
                rc = 2;
                break;
            }
            std::printf("[C] PASS L=%zu S=%zu (%zu cycles)\n", L, S, c_cycles);
        }
        if (run_d) {
            clear_faults();
            bool ok = scenario_d(cfg, L, S, d_cycles, batches_per_side,
                                 std::chrono::milliseconds(io_delay_ms));
            if (!ok || faults_observed()) {
                dump_faults("Scenario D");
                rc = 2;
                break;
            }
            std::printf("[D] PASS L=%zu S=%zu (%zu cycles, %lld ms io_delay, multi_addr=true)\n",
                        L, S, d_cycles, io_delay_ms);
        }
        if (run_e) {
            clear_faults();
            bool ok = scenario_e(cfg, L, S,
                                 std::chrono::seconds(e_budget_s),
                                 e_ceiling);
            if (!ok || faults_observed()) {
                dump_faults("Scenario E");
                rc = 2;
                break;
            }
            std::printf("[E] PASS L=%zu S=%zu (budget %llds, ceiling %zu cycles)\n",
                        L, S, e_budget_s, e_ceiling);
        }
    }

    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - t_start).count();
    std::printf("\n=== test_phase_change_fault summary: rc=%d elapsed=%llds ===\n",
                rc, static_cast<long long>(elapsed));

    // Unregister observer before any post-main static dtors fire (the
    // observer slot is process-global and used by other CUDA code paths).
    collider::gpu::set_fault_callback(nullptr);
    return rc;
}
