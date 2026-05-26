/**
 * test_bip_gpu_dispatcher -- lifecycle + hit-routing test for the
 * BipGpuDispatcher class.
 *
 * Why this exists: the 307-line BipGpuDispatcher class (commit
 * de78e60 and follow-ups) had zero direct tests until this file
 * landed. The class is exercised in production only through
 * run_combinatorial_scan / run_bip_scan_mode, neither of which is
 * in any integration test. A regression that breaks init failure
 * paths, lifecycle teardown, or the hit-callback wiring would not
 * be caught by ctest until an operator reported it -- which is
 * exactly the pattern the user has been complaining about.
 *
 * What's covered:
 *   1. Rejection of malformed Config: nullptr on_hit, empty gpu_ids.
 *   2. Successful init + enqueue + shutdown + on_hit fires.
 *   3. on_hit fires EXACTLY ONCE for a priv whose hash160 is seeded
 *      in the bloom; ZERO times for priv keys whose hash160 is not.
 *   4. shutdown() is idempotent (callable twice without crash).
 *   5. Destructor without explicit shutdown joins worker threads
 *      cleanly (no detached thread leak).
 *   6. last_error() / faulted_devices() are clean on success.
 *
 * Skipped (ctest exit 77) on non-CUDA builds. Built only when
 * COLLIDER_USE_CUDA + COLLIDER_PRO are both on.
 */

#include <array>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if !defined(COLLIDER_USE_CUDA) || !defined(COLLIDER_PRO)
int main() {
    std::printf("[SKIP] BipGpuDispatcher test needs COLLIDER_USE_CUDA + COLLIDER_PRO\n");
    return 77;
}
#else

#include <cuda_runtime.h>

#include "../src/runtime/bip_gpu_dispatcher.hpp"
#include "../src/core/bip32.hpp"
#include "../src/core/crypto_cpu.hpp"          // compute_hash160 (matches GPU)
#include "../src/runtime/bip_address.hpp"
#include "../src/tools/utxo_bloom_builder.hpp"
#include "../src/gpu/v2/brain_wallet_v2.hpp"   // AddressType + addr_bit

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>

namespace {

int g_pass = 0;
int g_fail = 0;

void check(const char* tag, bool ok) {
    if (ok) {
        ++g_pass;
        std::printf("[ ok  ] %s\n", tag);
    } else {
        ++g_fail;
        std::fprintf(stderr, "[FAIL] %s\n", tag);
    }
}

// Raw bloom representation matching what UTXOBloomBuilder produces +
// what the GPU kernel probes. Insert uses MurmurHash3-128 with the
// same seed and the same (h1, h2)-derived bit indices as the kernel.
// Mirrors test_gpu_multi_addr.cu's bloom_insert -- the proven oracle
// path -- to take UTXOBloomBuilder out of the equation entirely.
struct TestBloom {
    std::vector<uint8_t> bits;
    uint64_t             num_bits = 0;
    int                  num_hashes = 0;
    uint32_t             seed = 0;

    void insert(const std::array<uint8_t, 20>& h160) {
        auto [h1, h2] =
            ::collider::utxo::murmurhash3_128(h160.data(), 20, seed);
        for (int i = 0; i < num_hashes; ++i) {
            uint64_t idx = (h1 + (uint64_t)i * h2) % num_bits;
            bits[idx / 8] |= static_cast<uint8_t>(1u << (idx % 8));
        }
    }
};

// Build a tiny in-memory bloom seeded with the hash160(pubkey) of one
// known priv key.
TestBloom seed_bloom_for_priv(const std::array<uint8_t, 32>& priv,
                              bool p2sh_p2wpkh) {
    TestBloom b;
    b.num_bits   = 4096;          // pow-of-2, fits trivially in any VRAM
    b.num_hashes = 11;            // matches production blooms
    b.seed       = 0x5F3759DF;
    b.bits.assign(b.num_bits / 8, 0);

    auto compressed_h160 = collider::cpu::compute_hash160(priv.data());
    std::array<uint8_t, 20> h160 = compressed_h160;
    if (p2sh_p2wpkh) {
        uint8_t redeem[22];
        redeem[0] = 0x00;
        redeem[1] = 0x14;
        std::memcpy(redeem + 2, compressed_h160.data(), 20);
        auto sha = collider::cpu::SHA256::hash(redeem, 22);
        h160 = collider::cpu::RIPEMD160::hash(sha.data(), 32);
    }
    b.insert(h160);
    return b;
}

// Test 1: nullptr on_hit must be rejected.
void test_init_rejects_null_on_hit() {
    collider::runtime::BipGpuDispatcher disp;
    collider::runtime::BipGpuDispatcher::Config cfg{};
    cfg.gpu_ids = {0};
    cfg.bloom_data = nullptr;
    cfg.bloom_bits = 0;
    cfg.bloom_hashes = 1;
    cfg.bloom_seed = 0;
    // cfg.on_hit deliberately unset.
    const int rc = disp.init(cfg);
    check("init rejects nullptr on_hit (rc != 0)", rc != 0);
    check("init rejects nullptr on_hit (last_error mentions callback)",
          disp.last_error().find("on_hit") != std::string::npos);
}

// Test 2: empty gpu_ids must be rejected (no silent default to {0}).
void test_init_rejects_empty_gpu_ids() {
    collider::runtime::BipGpuDispatcher disp;
    collider::runtime::BipGpuDispatcher::Config cfg{};
    cfg.gpu_ids = {};   // empty -- caller forgot detect_gpus()
    cfg.bloom_data = nullptr;
    cfg.bloom_bits = 0;
    cfg.bloom_hashes = 1;
    cfg.bloom_seed = 0;
    cfg.on_hit = [](const std::string&, const std::string&,
                    const std::string&, const uint8_t*,
                    const char*, const uint8_t*) {};
    const int rc = disp.init(cfg);
    check("init rejects empty gpu_ids (rc != 0)", rc != 0);
    check("init rejects empty gpu_ids (last_error mentions detect_gpus)",
          disp.last_error().find("detect_gpus") != std::string::npos);
}

// Test 3: end-to-end happy path. Seed a bloom with one known h160,
// init the dispatcher, enqueue 4 priv keys (1 matching, 3 not),
// verify on_hit fires exactly once for the matching priv.
//
// Validates the kernel byte-order fix (commit follows this test):
// pre-fix, priv was passed BE bytes but the EC mul read LE limbs;
// pub_xy was LE limbs but multi_address_kernel read it as BE bytes;
// every h160 was wrong and BIP scan silently produced 0 hits in
// production. This test pins the round-trip end-to-end so any
// future byte-order regression breaks the build.
void test_end_to_end_hit_routing() {
    // priv = 1 (the generator G). The compressed pubkey is well-known
    // (02 || 0x79BE667E...), so a kernel-vs-CPU h160 mismatch here is
    // a regression in the EC mul, not in the test setup.
    std::array<uint8_t, 32> matching_priv{};
    matching_priv[31] = 0x01;

    auto bloom = seed_bloom_for_priv(matching_priv, /*p2sh=*/false);

    collider::runtime::BipGpuDispatcher disp;
    collider::runtime::BipGpuDispatcher::Config cfg{};
    cfg.gpu_ids = {0};
    cfg.bloom_data    = bloom.bits.data();
    cfg.bloom_bits    = bloom.num_bits;
    cfg.bloom_hashes  = bloom.num_hashes;
    cfg.bloom_seed    = bloom.seed;
    cfg.batch_size    = 64;
    cfg.queue_max     = 256;

    std::atomic<int> hit_count{0};
    std::array<uint8_t, 32> hit_priv{};
    std::string hit_path;
    std::string hit_label;
    std::mutex hit_mu;
    cfg.on_hit = [&](const std::string& mnemonic,
                     const std::string& path,
                     const std::string& profile_label,
                     const uint8_t      priv[32],
                     const char*        addr_type_label,
                     const uint8_t      h160[20]) {
        (void)mnemonic; (void)addr_type_label; (void)h160;
        std::lock_guard<std::mutex> lk(hit_mu);
        std::memcpy(hit_priv.data(), priv, 32);
        hit_path  = path;
        hit_label = profile_label;
        ++hit_count;
    };

    const int rc = disp.init(cfg);
    check("end-to-end: init returns 0", rc == 0);
    check("end-to-end: faulted_devices empty",
          disp.faulted_devices().empty());
    check("end-to-end: last_error empty on success",
          disp.last_error().empty());
    check("end-to-end: 1 device stat row",
          disp.device_stats().size() == 1);

    // Sanity: verify the bloom was seeded correctly by probing it
    // CPU-side with the same matching priv. If this fails, the test
    // setup (priv -> h160 -> bloom insert chain) is broken and the
    // dispatcher test is meaningless. If this passes but the GPU
    // probe below fails, the kernel itself has a regression.
    {
        auto h = collider::cpu::compute_hash160(matching_priv.data());
        auto [h1, h2] = collider::utxo::murmurhash3_128(
            h.data(), 20, bloom.seed);
        bool all_set = true;
        for (int i = 0; i < bloom.num_hashes; ++i) {
            uint64_t idx = (h1 + (uint64_t)i * h2) % bloom.num_bits;
            if (!(bloom.bits[idx / 8] & (1u << (idx % 8)))) {
                all_set = false; break;
            }
        }
        check("end-to-end: CPU bloom probe finds matching priv "
              "(test setup sanity)", all_set);
    }

    // Enqueue the matching priv plus 3 priv keys that should NOT hit.
    auto make_item = [](std::array<uint8_t, 32> priv,
                        std::string path,
                        std::string label) {
        collider::runtime::BipGpuWorkItem item;
        item.priv = priv;
        item.mnemonic = "test mnemonic " + path;
        item.derivation_path = std::move(path);
        item.profile_label = std::move(label);
        using collider::gpu::v2::AddressType;
        using collider::gpu::v2::addr_bit;
        item.addr_mask =
            static_cast<int>(addr_bit(AddressType::P2PKH_COMPRESSED));
        return item;
    };

    check("enqueue matching",
          disp.enqueue(make_item(matching_priv,
                                 "m/test/match", "TestProfile")));

    // 3 non-matching priv keys (different first byte → totally
    // different pub → different h160 → bloom rejects).
    for (int i = 0; i < 3; ++i) {
        std::array<uint8_t, 32> priv{};
        for (int j = 0; j < 32; ++j) priv[j] = static_cast<uint8_t>(0x42 + i + j);
        check("enqueue non-match",
              disp.enqueue(make_item(priv, "m/test/nomatch",
                                     "TestProfile")));
    }

    // Wait up to 2 seconds for the worker to drain the queue. The
    // dispatcher worker pulls in batches, processes, calls on_hit.
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (std::chrono::steady_clock::now() < deadline) {
        if (disp.total_addresses_dispatched() >= 4) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    disp.shutdown();

    check("end-to-end: dispatched all 4 enqueued items",
          disp.total_addresses_dispatched() == 4);
    check("end-to-end: exactly 1 hit fired",
          hit_count.load() == 1);
    {
        std::lock_guard<std::mutex> lk(hit_mu);
        check("end-to-end: hit priv matches enqueued matching priv",
              std::memcmp(hit_priv.data(), matching_priv.data(), 32) == 0);
        check("end-to-end: hit path correct",
              hit_path == "m/test/match");
        check("end-to-end: hit profile_label correct",
              hit_label == "TestProfile");
    }
}

// Test 4: shutdown() called twice in a row must not crash.
void test_shutdown_idempotent() {
    std::array<uint8_t, 32> priv{};
    for (int i = 0; i < 32; ++i) priv[i] = static_cast<uint8_t>(i + 1);
    auto bloom = seed_bloom_for_priv(priv, /*p2sh=*/false);

    collider::runtime::BipGpuDispatcher disp;
    collider::runtime::BipGpuDispatcher::Config cfg{};
    cfg.gpu_ids = {0};
    cfg.bloom_data   = bloom.bits.data();
    cfg.bloom_bits   = bloom.num_bits;
    cfg.bloom_hashes = bloom.num_hashes;
    cfg.bloom_seed   = bloom.seed;
    cfg.batch_size   = 32;
    cfg.on_hit = [](const std::string&, const std::string&,
                    const std::string&, const uint8_t*,
                    const char*, const uint8_t*) {};
    const int rc = disp.init(cfg);
    check("shutdown-idempotent: init 0", rc == 0);

    disp.shutdown();
    disp.shutdown();  // must not crash
    check("shutdown-idempotent: second shutdown safe", true);
}

// Test 5: destructor without explicit shutdown.
void test_dtor_without_shutdown() {
    std::array<uint8_t, 32> priv{};
    for (int i = 0; i < 32; ++i) priv[i] = static_cast<uint8_t>(i + 1);
    auto bloom = seed_bloom_for_priv(priv, /*p2sh=*/false);

    {
        collider::runtime::BipGpuDispatcher disp;
        collider::runtime::BipGpuDispatcher::Config cfg{};
        cfg.gpu_ids = {0};
        cfg.bloom_data   = bloom.bits.data();
        cfg.bloom_bits   = bloom.num_bits;
        cfg.bloom_hashes = bloom.num_hashes;
        cfg.bloom_seed   = bloom.seed;
        cfg.batch_size   = 32;
        cfg.on_hit = [](const std::string&, const std::string&,
                        const std::string&, const uint8_t*,
                        const char*, const uint8_t*) {};
        const int rc = disp.init(cfg);
        check("dtor-test: init 0", rc == 0);
        // Let disp go out of scope WITHOUT calling shutdown().
        // The dtor must join all worker threads cleanly.
    }
    check("dtor-test: implicit shutdown via dtor OK", true);
}

// Test 6: P2SH-P2WPKH end-to-end. Same shape as the P2PKH end-to-end
// test but with addr_mask = P2SH_P2WPKH and a bloom seeded with the
// P2SH-P2WPKH h160 of the matching priv. Pins the OTHER kernel branch
// in the byte-order fix so a regression in either branch trips.
void test_end_to_end_p2sh_p2wpkh() {
    std::array<uint8_t, 32> matching_priv{};
    matching_priv[31] = 0x07;  // priv = 7 (well-known scalar)

    auto bloom = seed_bloom_for_priv(matching_priv, /*p2sh=*/true);

    collider::runtime::BipGpuDispatcher disp;
    collider::runtime::BipGpuDispatcher::Config cfg{};
    cfg.gpu_ids = {0};
    cfg.bloom_data    = bloom.bits.data();
    cfg.bloom_bits    = bloom.num_bits;
    cfg.bloom_hashes  = bloom.num_hashes;
    cfg.bloom_seed    = bloom.seed;
    cfg.batch_size    = 32;

    std::atomic<int> hit_count{0};
    std::string hit_addr_type;
    std::mutex hit_mu;
    cfg.on_hit = [&](const std::string&,
                     const std::string&,
                     const std::string&,
                     const uint8_t      [32],
                     const char*        addr_type_label,
                     const uint8_t      [20]) {
        std::lock_guard<std::mutex> lk(hit_mu);
        hit_addr_type = addr_type_label;
        ++hit_count;
    };

    check("p2sh-p2wpkh: init returns 0", disp.init(cfg) == 0);

    collider::runtime::BipGpuWorkItem item;
    item.priv            = matching_priv;
    item.mnemonic        = "p2sh test";
    item.derivation_path = "m/49'/0'/0'/0/0";
    item.profile_label   = "BIP-49 test";
    using collider::gpu::v2::AddressType;
    using collider::gpu::v2::addr_bit;
    item.addr_mask =
        static_cast<int>(addr_bit(AddressType::P2SH_P2WPKH));
    check("p2sh-p2wpkh: enqueue", disp.enqueue(std::move(item)));

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (std::chrono::steady_clock::now() < deadline) {
        if (disp.total_addresses_dispatched() >= 1) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    disp.shutdown();

    check("p2sh-p2wpkh: exactly 1 hit fired", hit_count.load() == 1);
    {
        std::lock_guard<std::mutex> lk(hit_mu);
        check("p2sh-p2wpkh: addr_type label correct",
              hit_addr_type == std::string("P2SH-P2WPKH"));
    }
}

// Test 7: addr_mask=0 in enqueue items. The dispatcher's batch worker
// OR's per-item addr_mask values across the batch and substitutes the
// (P2PKH_COMPRESSED | P2SH_P2WPKH) default when the OR is zero. Pins
// that fallback so a future caller passing a zero mask still gets a
// sensible probe shape rather than the kernel rejecting the batch
// outright (process_batch returns 64 on addr_mask==0).
void test_addr_mask_zero_falls_back_to_default() {
    std::array<uint8_t, 32> matching_priv{};
    matching_priv[31] = 0x01;
    auto bloom = seed_bloom_for_priv(matching_priv, /*p2sh=*/false);

    collider::runtime::BipGpuDispatcher disp;
    collider::runtime::BipGpuDispatcher::Config cfg{};
    cfg.gpu_ids = {0};
    cfg.bloom_data   = bloom.bits.data();
    cfg.bloom_bits   = bloom.num_bits;
    cfg.bloom_hashes = bloom.num_hashes;
    cfg.bloom_seed   = bloom.seed;
    cfg.batch_size   = 32;

    std::atomic<int> hit_count{0};
    cfg.on_hit = [&](const std::string&, const std::string&,
                     const std::string&, const uint8_t  [32],
                     const char*, const uint8_t [20]) {
        ++hit_count;
    };
    check("addr_mask=0: init 0", disp.init(cfg) == 0);

    collider::runtime::BipGpuWorkItem item;
    item.priv = matching_priv;
    item.addr_mask = 0;   // deliberately zero
    item.mnemonic = "zero-mask test";
    item.derivation_path = "m/0/0";
    item.profile_label = "test";
    check("addr_mask=0: enqueue", disp.enqueue(std::move(item)));

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (std::chrono::steady_clock::now() < deadline) {
        if (disp.total_addresses_dispatched() >= 1) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    disp.shutdown();

    // Bloom was seeded for P2PKH_COMPRESSED; default mask covers it.
    check("addr_mask=0: default mask catches matching priv",
          hit_count.load() == 1);
}

}  // namespace

int main() {
    // Skip on hosts with no CUDA device (CI without GPU).
    int device_count = 0;
    cudaError_t rc = cudaGetDeviceCount(&device_count);
    if (rc != cudaSuccess || device_count == 0) {
        std::printf("[SKIP] no CUDA device visible (rc=%d count=%d)\n",
                    static_cast<int>(rc), device_count);
        return 77;
    }

    test_init_rejects_null_on_hit();
    test_init_rejects_empty_gpu_ids();
    test_end_to_end_hit_routing();
    test_shutdown_idempotent();
    test_dtor_without_shutdown();
    test_end_to_end_p2sh_p2wpkh();
    test_addr_mask_zero_falls_back_to_default();

    std::printf("=== Result: %d pass, %d fail ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

#endif  // COLLIDER_USE_CUDA && COLLIDER_PRO
