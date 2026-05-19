/**
 * test_dp_table_full256.cpp -- v1.4.2 A.4 regression test.
 *
 * Pre-fix the kangaroo DP table was keyed on x_low (uint64_t). Two
 * Distinguished Points with identical low 64 bits but DIFFERENT upper
 * 192 bits collided silently - the second-comer was discarded. This
 * could lose a real DP collision (e.g., a Tame-Wild crossing) if the
 * map already held an x_low-colliding entry.
 *
 * With the full-256 DPKey, both entries coexist at distinct map slots.
 */

#include "../src/gpu/kangaroo_dp_table.hpp"

#include <iostream>
#include <unordered_map>

using collider::kangaroo::DPKey;

namespace {

struct DPEntry {
    uint64_t dist[4];
    uint32_t type;
    int counter;  // For tracking which insertion landed
};

int g_pass = 0;
int g_fail = 0;

#define EXPECT_TRUE(cond, label)                                            \
    do {                                                                    \
        if (cond) {                                                         \
            ++g_pass;                                                       \
        } else {                                                            \
            ++g_fail;                                                       \
            std::cerr << "[FAIL] " << (label) << "\n";                      \
        }                                                                   \
    } while (0)

void test_x_low_collision_distinct_entries() {
    // Two DPs that would have collided in the pre-fix table:
    //   x_low (d[0]) identical, upper limbs DIFFER.
    // Under full-256 keying they MUST land at distinct map slots.
    std::unordered_map<DPKey, DPEntry> table;

    DPKey k1{{0xCAFEBABE12345678ULL, 0x1111111111111111ULL,
              0x2222222222222222ULL, 0x3333333333333333ULL}};
    DPKey k2{{0xCAFEBABE12345678ULL, 0x9999999999999999ULL,
              0x8888888888888888ULL, 0x7777777777777777ULL}};

    EXPECT_TRUE(!(k1 == k2), "k1 != k2 (full-256)");

    table[k1] = DPEntry{{0,0,0,0}, 0, 1};
    table[k2] = DPEntry{{0,0,0,0}, 0, 2};

    EXPECT_TRUE(table.size() == 2, "both keys retained");
    EXPECT_TRUE(table.find(k1) != table.end() && table[k1].counter == 1,
                "k1 retrievable");
    EXPECT_TRUE(table.find(k2) != table.end() && table[k2].counter == 2,
                "k2 retrievable");
}

void test_exact_duplicate_replaces() {
    std::unordered_map<DPKey, DPEntry> table;
    DPKey k{{0xDEADBEEFDEADBEEFULL, 0, 0, 0}};
    table[k] = DPEntry{{0,0,0,0}, 0, 1};
    table[k] = DPEntry{{0,0,0,0}, 0, 99};  // same key
    EXPECT_TRUE(table.size() == 1, "duplicate at full-256 key is a single slot");
    EXPECT_TRUE(table[k].counter == 99, "duplicate replaces (operator[] semantics)");
}

void test_hash_distinguishes_upper_limbs() {
    // The pre-fix hash collapse was: hash(k) = k.x[0]. Two keys with same
    // d[0] hashed to the same bucket. With the new hash, this should not
    // happen - the two hashes differ.
    DPKey k1{{0xCAFEBABE12345678ULL, 0, 0, 0}};
    DPKey k2{{0xCAFEBABE12345678ULL, 1, 0, 0}};
    std::hash<DPKey> h;
    EXPECT_TRUE(h(k1) != h(k2),
                "upper-limb diff produces different hash");
}

void test_many_collisions_isolated() {
    // 1000 keys with identical d[0] but unique d[1]. All must survive.
    std::unordered_map<DPKey, DPEntry> table;
    for (int i = 0; i < 1000; ++i) {
        DPKey k{{0xFFFFFFFF00000000ULL, (uint64_t)i, 0, 0}};
        table[k] = DPEntry{{0,0,0,0}, 0, i};
    }
    EXPECT_TRUE(table.size() == 1000, "1000 colliding-low entries all retained");
    for (int i = 0; i < 1000; ++i) {
        DPKey k{{0xFFFFFFFF00000000ULL, (uint64_t)i, 0, 0}};
        if (table[k].counter != i) {
            ++g_fail;
            std::cerr << "[FAIL] retrieval at i=" << i << "\n";
            return;
        }
    }
    ++g_pass;
    std::cout << "[ok  ] 1000 keys with same d[0] all retrievable at correct counter\n";
}

}  // namespace

int main() {
    std::cout << "test_dp_table_full256 (v1.4.2 A.4 regression suite)\n";

    test_x_low_collision_distinct_entries();
    test_exact_duplicate_replaces();
    test_hash_distinguishes_upper_limbs();
    test_many_collisions_isolated();

    std::cout << "Summary: " << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail == 0 ? 0 : 1;
}
