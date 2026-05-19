/*
 * Apple Metal Pollard's Kangaroo kernel for secp256k1 (v1.4.1).
 *
 * Self-contained MSL: 256-bit mod-p arithmetic, Jacobian point arithmetic
 * (no inversion in the inner loop), threadgroup-cooperative Montgomery
 * batch inversion for distinguished-point detection. Mirrors the CUDA
 * path in src/gpu/kangaroo_kernel.cu (warp_batch_jacobian_to_affine +
 * kangaroo_step_kernel) but expressed for Metal's threadgroup model.
 *
 * Field representation: a 256-bit element is uint4 of ulong (4 packed
 * uint64), little-endian by limb.
 *   limb 0 = bits   0..63
 *   limb 1 = bits  64..127
 *   limb 2 = bits 128..191
 *   limb 3 = bits 192..255
 *
 * secp256k1 prime: p = 2^256 - 2^32 - 977.
 *   p_limb = (FFFFFFFEFFFFFC2F, FFFFFFFFFFFFFFFF, FFFFFFFFFFFFFFFF, FFFFFFFFFFFFFFFF)
 *
 * Reduction trick: 2^256 = 2^32 + 977 (mod p). After a 256x256 -> 512
 * multiply we split into (high, low) and compute low + high*(2^32+977)
 * to fold high back in.
 *
 * ===========================================================================
 *                              JACOBIAN INVARIANT
 * ===========================================================================
 *
 * Each kangaroo carries (X, Y, Z) such that the affine point is
 * (X / Z^2, Y / Z^3). Z = 0 means "point at infinity"; Z = 1 means the
 * Jacobian point is already affine. Seeds enter with Z = 1 (the host's
 * be32_to_limbs_le seed loop populates affine x, affine y; the kernel
 * initializer or host-side seeder also writes Z = 1 to the z buffer).
 *
 * Inner walk loop runs Jacobian point addition (mixed Jacobian + affine)
 * per step. NO modular inversion in the inner loop. Cost per step:
 *   ec_add_mixed: 11 mod_mul + 5 mod_sqr + ~6 mod_add/sub
 * vs the previous affine path:
 *   point_add_distinct: 1 mod_inv + 2 mod_mul + 1 mod_sqr + ...
 *
 * mod_inv is roughly 256 mod_sqr + 13 mod_mul (~269 mod_muls). So the
 * inner-loop saving per step is ~250 mod_muls. The net throughput win
 * is dominated by amortizing the mod_inv across a threadgroup at
 * DP-check time.
 *
 * ===========================================================================
 *                    BATCH INVERSION THREADGROUP CONTRACT
 * ===========================================================================
 *
 * KANGAROO_BATCH_SIZE = 32 threads per threadgroup. Host MUST dispatch
 * with threadsPerThreadgroup.x = KANGAROO_BATCH_SIZE and num_kangaroos
 * divisible by KANGAROO_BATCH_SIZE. The host check lives in
 * kangaroo_metal.mm submit_step(); a bad config trips a sticky error
 * and the kernel is never launched.
 *
 * Why 32?
 *   - Matches the CUDA WARP_SIZE so the algorithm and tuning carry over.
 *   - Apple GPU SIMD-group width is 32 on M1/M2/M3 (the threadgroup
 *     barrier resolves to a SIMD shuffle on Apple silicon at this size).
 *   - Threadgroup memory cost: 3 arrays * 32 lanes * 4 limbs * 8 bytes
 *     = 3 KiB per threadgroup. Apple GPUs allow >=32 KiB threadgroup
 *     memory per dispatch, so headroom is fine.
 *
 * EVERY thread in the threadgroup MUST hit every threadgroup_barrier;
 * Metal deadlocks on partial participation. The kangaroo_step kernel
 * therefore never early-returns before the loop and unconditionally
 * participates in every barrier inside the loop body. Threads with
 * gid >= count (only possible if num_kangaroos isn't a multiple of 32,
 * which the host rejects) would still execute correctly via the
 * sentinel-Z=1 path; their writes go to out-of-range slots in the state
 * buffers, which the host allocates with rounding-up sizing.
 *
 * Algorithm (Montgomery batch inversion, 1 inv + 3*(N-1) mul):
 *   1. Each thread writes its Jacobian Z into s_z_coords[lane].
 *   2. threadgroup_barrier.
 *   3. Lane 0 builds prefix products: s_products[i] = z[0] * z[1] * ... * z[i].
 *   4. Lane 0 computes one mod_inv on the final product.
 *   5. Lane 0 walks backwards: s_z_inv[i] = inv_all * s_products[i-1];
 *      inv_all *= s_z_coords[i]. End condition: s_z_inv[0] = inv_all.
 *   6. threadgroup_barrier.
 *   7. Each thread reads its s_z_inv[lane], computes
 *      affine_x = X * z_inv^2, affine_y = Y * z_inv^3.
 *
 * The prefix-product walk is sequential on lane 0; for N=32 that's 31
 * mod_mul on a single lane, 1 mod_inv (~269 mod_mul equivalent), then
 * 31 * 2 = 62 mod_mul on the back-pass. Total ~362 mod_mul shared
 * across the threadgroup, vs 32 mod_inv = ~32 * 269 = 8608 mod_mul
 * if every thread inverted independently. ~24x reduction at the DP
 * check point, every DP_CHECK_INTERVAL=32 steps.
 *
 * The single-thread serialization on lane 0 leaves lanes 1..31 idle
 * during the prefix walk. A SIMD-group-shuffle (Hillis-Steele) variant
 * would give log-N parallelism, but it requires SIMD-group intrinsics
 * (Metal's [[simdgroup_size]] family) that complicate the kernel and
 * historically have driver edge cases on older Apple GPUs. The CUDA
 * reference also uses lane-0 serialization (see warp_batch_jacobian_to_affine
 * at kangaroo_kernel.cu:682). Match the CUDA approach for parity; if
 * profiling on M3 Max ever shows lane-0 stalling as the bottleneck,
 * that's a follow-up.
 *
 * ===========================================================================
 *                           DP DETECTION SEMANTICS
 * ===========================================================================
 *
 * DP detection MUST run on the affine X coordinate. Jacobian X is
 * X_aff * Z^2; its low bits have no correlation with the low bits of
 * X_aff. Calling is_distinguished() on Jacobian X catches ~0% of true
 * DPs. (CUDA had this exact bug; see comment block at kangaroo_kernel.cu:1319.)
 *
 * We check at intervals of DP_CHECK_INTERVAL = 32 walk steps OR on the
 * very last step of the round. On a hit, the kernel:
 *   1. Atomically reserves a slot in dp_records[].
 *   2. Writes the 74-byte JLPDistinguishedPointV2-shaped record using
 *      the affine X (recovered via the batch inversion).
 *   3. Continues walking. (Unlike the CUDA path which returns on first
 *      DP, the Metal kernel keeps going until steps_per_round elapses
 *      so multiple DPs per round still surface.)
 *
 * State persistence at kernel exit: x/y/z buffers carry Jacobian
 * coordinates between rounds. On entry to round N+1, the prior round's
 * Jacobian state is loaded as-is and the walk continues. The first
 * round's seeds enter with Z = 1 (host writes Z = 1 to the z buffer
 * during seed_kangaroos / replace_seed). After the first DP-check
 * conversion, Z is no longer 1, but the point is still valid Jacobian.
 */

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

// Jump-table size shared with the host (see kJumpTableSize in
// src/gpu/kangaroo_metal.hpp). Must be a power of two; KANGAROO_JUMP_MASK
// is the index-selector mask. If you bump the size on either side, bump
// it on both -- the host static_assert will catch a non-power-of-two,
// but the host/device sync is enforced only by convention.
#define KANGAROO_JUMP_TABLE_SIZE 32u
#define KANGAROO_JUMP_MASK       (KANGAROO_JUMP_TABLE_SIZE - 1u)
static_assert((KANGAROO_JUMP_TABLE_SIZE & KANGAROO_JUMP_MASK) == 0u,
              "KANGAROO_JUMP_TABLE_SIZE must be a power of two");

// Threadgroup batch size for Montgomery batch inversion at DP check time.
// Matches CUDA's WARP_SIZE = 32. Host enforces num_kangaroos % 32 == 0
// and threadsPerThreadgroup = 32 in kangaroo_metal.mm submit_step().
#define KANGAROO_BATCH_SIZE 32u

// DP detection cadence: convert to affine and check every N walk steps,
// plus on the very last step of the round (so a DP that lands on step
// steps_per_round - 1 isn't lost). 32 matches CUDA's DP_CHECK_INTERVAL.
#define DP_CHECK_INTERVAL 32u

// ---------------------------------------------------------------------------
// uint128 emulation via two ulongs. Metal has no native u128; we manually
// chain 64-bit add-with-carry and 64x64 -> 128 multiply.
// ---------------------------------------------------------------------------

inline ulong addc(ulong a, ulong b, thread ulong &carry) {
    ulong sum = a + b;
    ulong c1 = (sum < a) ? 1ul : 0ul;
    ulong sum2 = sum + carry;
    ulong c2 = (sum2 < sum) ? 1ul : 0ul;
    carry = c1 + c2;
    return sum2;
}

inline ulong subb(ulong a, ulong b, thread ulong &borrow) {
    ulong diff = a - b;
    ulong b1 = (a < b) ? 1ul : 0ul;
    ulong diff2 = diff - borrow;
    ulong b2 = (diff < borrow) ? 1ul : 0ul;
    borrow = b1 + b2;
    return diff2;
}

// 64x64 -> 128. Returns low; sets `hi` to the high 64.
inline ulong mul128(ulong a, ulong b, thread ulong &hi) {
    // Split into 32-bit halves. ulong on Metal is 64-bit unsigned.
    ulong al = a & 0xFFFFFFFFul, ah = a >> 32;
    ulong bl = b & 0xFFFFFFFFul, bh = b >> 32;
    ulong p0 = al * bl;
    ulong p1 = al * bh;
    ulong p2 = ah * bl;
    ulong p3 = ah * bh;
    ulong mid = (p0 >> 32) + (p1 & 0xFFFFFFFFul) + (p2 & 0xFFFFFFFFul);
    ulong lo = (p0 & 0xFFFFFFFFul) | (mid << 32);
    hi = p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32);
    return lo;
}

// ---------------------------------------------------------------------------
// secp256k1 prime constants.
// p = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
// In our LE-by-limb layout: (FFFFFFFEFFFFFC2F, FFFFFFFFFFFFFFFF, FFFFFFFFFFFFFFFF, FFFFFFFFFFFFFFFF)
// ---------------------------------------------------------------------------

constant ulong P0 = 0xFFFFFFFEFFFFFC2Ful;
constant ulong P1 = 0xFFFFFFFFFFFFFFFFul;
constant ulong P2 = 0xFFFFFFFFFFFFFFFFul;
constant ulong P3 = 0xFFFFFFFFFFFFFFFFul;
// 2^32 + 977 = 0x1000003D1
constant ulong K0 = 0x00000001000003D1ul;

// ---------------------------------------------------------------------------
// 256-bit add / sub mod p
// ---------------------------------------------------------------------------

// r = a + b mod p, where a, b are <= p-1.
// After add we may have a 257-bit value; if the carry-out of the top limb is
// 1, OR if the result is >= p, subtract p once.
inline void mod_add(thread ulong r[4], thread const ulong a[4], thread const ulong b[4]) {
    ulong c = 0;
    ulong t0 = addc(a[0], b[0], c);
    ulong t1 = addc(a[1], b[1], c);
    ulong t2 = addc(a[2], b[2], c);
    ulong t3 = addc(a[3], b[3], c);
    // c is 0 or 1; t3..t0 plus c*2^256 = a+b
    // If carry, fold via 2^256 -> K0: t += K0
    if (c) {
        ulong cc = 0;
        t0 = addc(t0, K0, cc);
        t1 = addc(t1, 0, cc);
        t2 = addc(t2, 0, cc);
        t3 = addc(t3, 0, cc);
    }
    // Conditional final subtract if t >= p.
    bool ge = (t3 > P3) ||
              (t3 == P3 && t2 > P2) ||
              (t3 == P3 && t2 == P2 && t1 > P1) ||
              (t3 == P3 && t2 == P2 && t1 == P1 && t0 >= P0);
    if (ge) {
        ulong b2 = 0;
        t0 = subb(t0, P0, b2);
        t1 = subb(t1, P1, b2);
        t2 = subb(t2, P2, b2);
        t3 = subb(t3, P3, b2);
    }
    r[0] = t0; r[1] = t1; r[2] = t2; r[3] = t3;
}

// r = a - b mod p
inline void mod_sub(thread ulong r[4], thread const ulong a[4], thread const ulong b[4]) {
    ulong bo = 0;
    ulong t0 = subb(a[0], b[0], bo);
    ulong t1 = subb(a[1], b[1], bo);
    ulong t2 = subb(a[2], b[2], bo);
    ulong t3 = subb(a[3], b[3], bo);
    if (bo) {
        // Underflow: add p.
        ulong c = 0;
        t0 = addc(t0, P0, c);
        t1 = addc(t1, P1, c);
        t2 = addc(t2, P2, c);
        t3 = addc(t3, P3, c);
    }
    r[0] = t0; r[1] = t1; r[2] = t2; r[3] = t3;
}

// ---------------------------------------------------------------------------
// 256x256 -> 512-bit multiply, then secp256k1 reduction.
// Standard schoolbook: 16 64x64 muls. Reduction folds the high half via
// (2^32 + 977) twice.
// ---------------------------------------------------------------------------

inline void mod_mul(thread ulong r[4], thread const ulong a[4], thread const ulong b[4]) {
    ulong w[8];
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i) w[i] = 0;

    // Schoolbook multiply. Each column adds three values: the running
    // accumulator w[i+j], the new partial product `lo`, and the
    // incoming `carry` from the prior column. We do that as two
    // separate addc calls and propagate BOTH carries to the next
    // column. The earlier version reused c1 as the seed for c2, which
    // double-counted on one column and lost the c1 carry entirely.
    for (int i = 0; i < 4; ++i) {
        ulong carry = 0;
        for (int j = 0; j < 4; ++j) {
            ulong hi;
            ulong lo = mul128(a[i], b[j], hi);
            ulong c1 = 0, c2 = 0;
            ulong sum0 = addc(w[i + j], lo,    c1);   // c1 = (w + lo) carry
            ulong sum1 = addc(sum0,    carry, c2);    // c2 = (sum0 + carry) carry
            w[i + j] = sum1;
            carry = hi + c1 + c2;
        }
        w[i + 4] = carry;
    }

    // Reduction: split into hi (w[4..7]) + lo (w[0..3]); compute
    //   r_init = lo + hi*K0 + hi*2^32   (since 2^256 = 2^32+977 mod p)
    // We do this as r = lo + hi*K0  where K0 = 2^32 + 977.
    // hi*K0 is 256x65 -> 320 bits. We do hi[i] * K0 row by row.
    // Then fold any 257th-bit carry once more.

    // First fold: r[0..3] = lo, then add hi * K0 across.
    ulong lo[4]   = { w[0], w[1], w[2], w[3] };
    ulong hi[4]   = { w[4], w[5], w[6], w[7] };

    // Compute hi * K0 -> 5-limb result h[0..4]. Same fixed add chain
    // as the schoolbook multiply (propagate both carries; do not reuse
    // c1 as the seed for c2).
    ulong h[5] = {0,0,0,0,0};
    {
        ulong carry = 0;
        for (int i = 0; i < 4; ++i) {
            ulong hh;
            ulong ll = mul128(hi[i], K0, hh);
            ulong c1 = 0, c2 = 0;
            ulong s0 = addc(h[i], ll,    c1);
            ulong s1 = addc(s0,   carry, c2);
            h[i] = s1;
            carry = hh + c1 + c2;
        }
        h[4] = carry;
    }

    // r = lo + h[0..3], then propagate h[4] folded by K0 (since 2^256 -> K0).
    ulong c = 0;
    ulong t0 = addc(lo[0], h[0], c);
    ulong t1 = addc(lo[1], h[1], c);
    ulong t2 = addc(lo[2], h[2], c);
    ulong t3 = addc(lo[3], h[3], c);
    // Now c + h[4] is the overflow that needs another K0 fold.
    ulong overflow = c + h[4];
    if (overflow) {
        // overflow * K0 fits in 96 bits; add to t.
        ulong oh, ol;
        ol = mul128(overflow, K0, oh);
        ulong c2 = 0;
        t0 = addc(t0, ol, c2);
        t1 = addc(t1, oh, c2);
        t2 = addc(t2, 0, c2);
        t3 = addc(t3, 0, c2);
        // If we carried out of t3, fold one more time (extremely rare).
        if (c2) {
            ulong c3 = 0;
            t0 = addc(t0, K0, c3);
            t1 = addc(t1, 0, c3);
            t2 = addc(t2, 0, c3);
            t3 = addc(t3, 0, c3);
        }
    }
    // Final conditional subtract if t >= p.
    bool ge = (t3 > P3) ||
              (t3 == P3 && t2 > P2) ||
              (t3 == P3 && t2 == P2 && t1 > P1) ||
              (t3 == P3 && t2 == P2 && t1 == P1 && t0 >= P0);
    if (ge) {
        ulong bo = 0;
        t0 = subb(t0, P0, bo);
        t1 = subb(t1, P1, bo);
        t2 = subb(t2, P2, bo);
        t3 = subb(t3, P3, bo);
    }
    r[0] = t0; r[1] = t1; r[2] = t2; r[3] = t3;
}

inline void mod_sqr(thread ulong r[4], thread const ulong a[4]) {
    mod_mul(r, a, a);
}

// ---------------------------------------------------------------------------
// Modular inverse via Fermat's little theorem: x^(p-2) mod p.
// p - 2 = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D
// libsecp256k1-style addition chain: 256 squarings + 13 multiplies.
// ---------------------------------------------------------------------------

constant ulong PMINUS2_0 = 0xFFFFFFFEFFFFFC2Dul;
constant ulong PMINUS2_1 = 0xFFFFFFFFFFFFFFFFul;
constant ulong PMINUS2_2 = 0xFFFFFFFFFFFFFFFFul;
constant ulong PMINUS2_3 = 0xFFFFFFFFFFFFFFFFul;

// Helpers for the inv chain. mod_sqr/mod_mul take fresh-temp + assign to
// avoid aliasing concerns. Metal Shading Language does not support
// C++ lambdas, so these are file-scope inline helpers used only by
// mod_inv below.
inline void inv_sqr(thread ulong r[4], thread const ulong x[4]) {
    ulong t[4]; mod_sqr(t, x);
    r[0] = t[0]; r[1] = t[1]; r[2] = t[2]; r[3] = t[3];
}
inline void inv_mul(thread ulong r[4],
                    thread const ulong x[4],
                    thread const ulong y[4]) {
    ulong t[4]; mod_mul(t, x, y);
    r[0] = t[0]; r[1] = t[1]; r[2] = t[2]; r[3] = t[3];
}

inline void mod_inv(thread ulong r[4], thread const ulong a[4]) {
    // libsecp256k1-style addition chain for a^(p-2) mod p. Same
    // algorithm now used by all GPU mod_inv impls (kangaroo_kernel.cu,
    // secp256k1.cu, puzzle_optimized.cu). Cost: 256 squarings + 13
    // multiplications (~269 mod_muls), down from the previous binary
    // walk's 256 + 248 = 504 mod_muls. CRITICAL: x223 step
    // multiplies by x3 (a^7), NOT x2 (a^3); using x2 produces
    // a^(2^223 - 5) instead of a^(2^223 - 1) and poisons the tail.

    ulong x2[4], x3[4], x6[4], x9[4], x11[4], x22[4], x44[4];
    ulong x88[4], x176[4], x220[4], x223[4], t1[4];

    // x2 = a^3 = a^2 * a
    inv_sqr(x2, a); inv_mul(x2, x2, a);
    // x3 = a^7 = (a^3)^2 * a
    inv_sqr(x3, x2); inv_mul(x3, x3, a);

    // x6 = a^(2^6 - 1)
    inv_sqr(x6, x3); inv_sqr(x6, x6); inv_sqr(x6, x6); inv_mul(x6, x6, x3);
    // x9 = a^(2^9 - 1)
    inv_sqr(x9, x6); inv_sqr(x9, x9); inv_sqr(x9, x9); inv_mul(x9, x9, x3);
    // x11 = a^(2^11 - 1)
    inv_sqr(x11, x9); inv_sqr(x11, x11); inv_mul(x11, x11, x2);

    // x22 = a^(2^22 - 1)
    inv_sqr(x22, x11);
    for (int i = 1; i < 11; ++i) inv_sqr(x22, x22);
    inv_mul(x22, x22, x11);

    // x44 = a^(2^44 - 1)
    inv_sqr(x44, x22);
    for (int i = 1; i < 22; ++i) inv_sqr(x44, x44);
    inv_mul(x44, x44, x22);

    // x88 = a^(2^88 - 1)
    inv_sqr(x88, x44);
    for (int i = 1; i < 44; ++i) inv_sqr(x88, x88);
    inv_mul(x88, x88, x44);

    // x176 = a^(2^176 - 1)
    inv_sqr(x176, x88);
    for (int i = 1; i < 88; ++i) inv_sqr(x176, x176);
    inv_mul(x176, x176, x88);

    // x220 = a^(2^220 - 1)
    inv_sqr(x220, x176);
    for (int i = 1; i < 44; ++i) inv_sqr(x220, x220);
    inv_mul(x220, x220, x44);

    // x223 = a^(2^223 - 1). x3 (a^7), NOT x2.
    inv_sqr(x223, x220); inv_sqr(x223, x223); inv_sqr(x223, x223);
    inv_mul(x223, x223, x3);

    // Tail: ^(2^23) * x22 ; ^(2^5) * a ; ^(2^3) * x2 ; ^(2^2) * a.
    inv_sqr(t1, x223);
    for (int i = 1; i < 23; ++i) inv_sqr(t1, t1);
    inv_mul(t1, t1, x22);
    for (int i = 0; i < 5; ++i) inv_sqr(t1, t1);
    inv_mul(t1, t1, a);
    for (int i = 0; i < 3; ++i) inv_sqr(t1, t1);
    inv_mul(t1, t1, x2);
    inv_sqr(t1, t1); inv_sqr(t1, t1);
    inv_mul(t1, t1, a);

    r[0] = t1[0]; r[1] = t1[1]; r[2] = t1[2]; r[3] = t1[3];
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

inline bool limbs_equal(thread const ulong a[4], thread const ulong b[4]) {
    return (a[0] == b[0]) & (a[1] == b[1]) & (a[2] == b[2]) & (a[3] == b[3]);
}

inline bool limbs_zero(thread const ulong a[4]) {
    return (a[0] == 0) & (a[1] == 0) & (a[2] == 0) & (a[3] == 0);
}

inline void copy4(thread ulong dst[4], thread const ulong src[4]) {
    dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; dst[3] = src[3];
}

inline void set4(thread ulong dst[4], ulong v0, ulong v1, ulong v2, ulong v3) {
    dst[0] = v0; dst[1] = v1; dst[2] = v2; dst[3] = v3;
}

// ---------------------------------------------------------------------------
// Jacobian secp256k1 point operations.
// A Jacobian point (X, Y, Z) represents the affine point (X / Z^2, Y / Z^3).
// secp256k1 has curve parameter a = 0, which simplifies the doubling
// formulas. The infinity point is encoded as Z = 0.
// Affine point -> Jacobian: (x, y) -> (x, y, 1).
// Jacobian point -> Affine: requires one modular inversion (x = X*Z^-2,
// y = Y*Z^-3). Inversion is the expensive operation, so the kangaroo
// step keeps points in Jacobian and only converts at DP-check time
// using Montgomery batch inversion.
// ---------------------------------------------------------------------------

// R = 2 * P (Jacobian). Doubling formula for a = 0 secp256k1:
//   S = 4 * X * Y^2
//   M = 3 * X^2
//   X' = M^2 - 2 * S
//   Y' = M * (S - X') - 8 * Y^4
//   Z' = 2 * Y * Z
// Cost: 1 mod_mul + 4 mod_sqr + 4 mod_add/sub plus the constant scaling.
// Returns infinity (Z = 0) if input is infinity or Y = 0.
inline void jac_double(thread ulong rx[4], thread ulong ry[4], thread ulong rz[4],
                       thread const ulong px[4], thread const ulong py[4],
                       thread const ulong pz[4])
{
    if (limbs_zero(pz) || limbs_zero(py)) {
        set4(rx, 0, 0, 0, 0);
        set4(ry, 0, 0, 0, 0);
        set4(rz, 0, 0, 0, 0);
        return;
    }

    ulong y2[4], s[4], m[4], t[4];

    // Y^2
    mod_sqr(y2, py);

    // S = 4 * X * Y^2
    mod_mul(s, px, y2);
    mod_add(s, s, s);    // 2*X*Y^2
    mod_add(s, s, s);    // 4*X*Y^2

    // M = 3 * X^2 (a = 0 simplification)
    mod_sqr(m, px);
    mod_add(t, m, m);    // 2*X^2
    mod_add(m, t, m);    // 3*X^2

    // X' = M^2 - 2*S
    mod_sqr(rx, m);
    mod_sub(rx, rx, s);
    mod_sub(rx, rx, s);

    // Y' = M*(S - X') - 8*Y^4
    mod_sub(t, s, rx);
    mod_mul(t, m, t);
    ulong y4[4];
    mod_sqr(y4, y2);     // Y^4
    mod_add(y4, y4, y4); // 2*Y^4
    mod_add(y4, y4, y4); // 4*Y^4
    mod_add(y4, y4, y4); // 8*Y^4
    mod_sub(ry, t, y4);

    // Z' = 2*Y*Z
    mod_mul(rz, py, pz);
    mod_add(rz, rz, rz);
}

// R = P + Q where P is Jacobian (px, py, pz) and Q is affine (qx, qy).
// Mixed-coordinate addition formula:
//   U2 = X2 * Z1^2
//   = Y2 * Z1^3
//   H  = U2 - X1
//   R  = S2 - Y1
//   if H == 0:
//       if R == 0:  return jac_double(P)   (P == Q)
//       else:       return infinity        (P == -Q)
//   X3 = R^2 - H^3 - 2*X1*H^2
//   Y3 = R * (X1*H^2 - X3) - Y1*H^3
//   Z3 = Z1 * H
// Cost: 7 mod_mul + 4 mod_sqr + 6 mod_add/sub (no inversion).
inline void jac_add_mixed(thread ulong rx[4], thread ulong ry[4], thread ulong rz[4],
                          thread const ulong px[4], thread const ulong py[4],
                          thread const ulong pz[4],
                          thread const ulong qx[4], thread const ulong qy[4])
{
    // P = infinity -> R = Q (with Z = 1).
    if (limbs_zero(pz)) {
        copy4(rx, qx);
        copy4(ry, qy);
        set4(rz, 1, 0, 0, 0);
        return;
    }

    ulong z2[4], z3[4], u2[4], s2[4];
    ulong h[4], rr[4], h2[4], h3[4], v[4], tmp[4];

    mod_sqr(z2, pz);                 // Z1^2
    mod_mul(z3, z2, pz);             // Z1^3
    mod_mul(u2, qx, z2);             // U2 = X2 * Z1^2
    mod_mul(s2, qy, z3);             // = Y2 * Z1^3
    mod_sub(h,  u2, px);             // H = U2 - X1
    mod_sub(rr, s2, py);             // R = S2 - Y1

    if (limbs_zero(h)) {
        if (limbs_zero(rr)) {
            // P == Q -> doubling.
            jac_double(rx, ry, rz, px, py, pz);
        } else {
            // P == -Q -> infinity.
            set4(rx, 0, 0, 0, 0);
            set4(ry, 0, 0, 0, 0);
            set4(rz, 0, 0, 0, 0);
        }
        return;
    }

    // Normal case.
    mod_sqr(h2, h);                  // H^2
    mod_mul(h3, h2, h);              // H^3
    mod_mul(v,  px, h2);             // V = X1 * H^2

    // X3 = R^2 - H^3 - 2*V
    mod_sqr(tmp, rr);
    mod_sub(tmp, tmp, h3);
    mod_sub(tmp, tmp, v);
    mod_sub(rx,  tmp, v);

    // Y3 = R*(V - X3) - Y1*H^3
    mod_sub(tmp, v, rx);
    mod_mul(tmp, rr, tmp);
    ulong yh3[4];
    mod_mul(yh3, py, h3);
    mod_sub(ry, tmp, yh3);

    // Z3 = Z1 * H
    mod_mul(rz, pz, h);
}

// ---------------------------------------------------------------------------
// Affine point operations. Used by priv_to_pub (test KAT vehicle) and as
// the last-mile conversion at DP-check time.
// ---------------------------------------------------------------------------

// R = P + Q (affine, P != Q, P != -Q). Used internally by point_op_affine
// after the dispatch checks.
inline void point_add_distinct(thread ulong rx[4], thread ulong ry[4],
                               thread const ulong px[4], thread const ulong py[4],
                               thread const ulong qx[4], thread const ulong qy[4])
{
    ulong dx[4], dy[4], dx_inv[4], lambda[4], lambda_sq[4];

    mod_sub(dx, qx, px);                    // dx = qx - px
    mod_sub(dy, qy, py);                    // dy = qy - py
    mod_inv(dx_inv, dx);                    // 1 / dx
    mod_mul(lambda, dy, dx_inv);            // lambda = dy / dx
    mod_sqr(lambda_sq, lambda);

    ulong tmp[4];
    mod_sub(tmp, lambda_sq, px);
    mod_sub(rx, tmp, qx);                   // rx = lambda^2 - px - qx

    ulong dxr[4], lr[4];
    mod_sub(dxr, px, rx);
    mod_mul(lr, lambda, dxr);
    mod_sub(ry, lr, py);                    // ry = lambda*(px - rx) - py
}

// R = 2 * P (affine). Used by priv_to_pub; tangent-slope formula.
inline void point_double_affine(thread ulong rx[4], thread ulong ry[4],
                                thread const ulong px[4], thread const ulong py[4])
{
    ulong x2[4], three_x2[4], two_y[4], two_y_inv[4];
    ulong lambda[4], lambda_sq[4], two_x[4];

    mod_sqr(x2, px);                         // x^2
    mod_add(three_x2, x2, x2);
    {
        ulong tmp[4];
        mod_add(tmp, three_x2, x2);
        copy4(three_x2, tmp);                // 3*x^2
    }
    mod_add(two_y, py, py);                  // 2*y
    mod_inv(two_y_inv, two_y);               // 1 / (2*y)
    mod_mul(lambda, three_x2, two_y_inv);    // lambda = 3*x^2 / (2*y)
    mod_sqr(lambda_sq, lambda);

    mod_add(two_x, px, px);
    mod_sub(rx, lambda_sq, two_x);           // rx = lambda^2 - 2*x

    ulong dxr[4], lr[4];
    mod_sub(dxr, px, rx);
    mod_mul(lr, lambda, dxr);
    mod_sub(ry, lr, py);                     // ry = lambda*(px - rx) - py
}

// Affine R = P + Q dispatcher (used by priv_to_pub).
inline void point_op_affine(thread ulong rx[4], thread ulong ry[4],
                            thread const ulong px[4], thread const ulong py[4],
                            thread const ulong qx[4], thread const ulong qy[4])
{
    if (limbs_equal(px, qx)) {
        if (limbs_equal(py, qy)) {
            point_double_affine(rx, ry, px, py);
        } else {
            // P == -Q -> identity. priv_to_pub doesn't expect this; emit zeros.
            set4(rx, 0, 0, 0, 0);
            set4(ry, 0, 0, 0, 0);
        }
        return;
    }
    point_add_distinct(rx, ry, px, py, qx, qy);
}

// ---------------------------------------------------------------------------
// DP record + serialization.
// ---------------------------------------------------------------------------

// Byte-packed: pure uchar fields so the natural alignment is 1 and the
// total size is exactly 74 bytes (no trailing padding). If we used a
// real `ulong work_id` field, MSL would align the struct to 8 and pad
// each record to 80 bytes, breaking the 74-byte stride the host reads
// and the JLPDistinguishedPointV2 wire format we want to feed.
struct DPRecord {
    uchar  work_id_le[8];   // little-endian (host is also LE)
    uchar  x_be[32];
    uchar  d_be[32];
    uchar  type;
    uchar  dp_bits;
};

inline void limbs_to_be(thread uchar out[32], thread const ulong limbs[4]) {
    // limb[3] is the most-significant 64 bits (big-endian byte 0..7).
    for (int i = 0; i < 4; ++i) {
        ulong l = limbs[3 - i];
        for (int j = 0; j < 8; ++j) {
            out[i * 8 + j] = (uchar)(l >> (56 - 8 * j));
        }
    }
}

// Shift-safe top-N-bit zero mask: returns the 64-bit mask that selects the
// top `n` bits of a limb. `n` must be in [0, 64]. The naive expression
// `~((1ul << (64-n)) - 1ul)` is undefined when n==0 (shift by 64) and the
// kangaroo dispatcher legitimately accepts dp_bits == 64 (one full limb of
// leading zeros) and 0 (no DP gating, used in tests). Branchless form:
//   if n == 0  -> 0  (no bits required to be zero -> trivially "set")
//   if n == 64 -> ~0 (entire limb required zero)
//   else       -> ~((1<<(64-n))-1)
inline ulong top_n_bits_mask(int n) {
    if (n <= 0)  return 0ul;
    if (n >= 64) return ~0ul;
    return ~((1ul << (64 - n)) - 1ul);
}

inline bool is_distinguished(thread const ulong x[4], int dp_bits) {
    // Test that the top `dp_bits` bits of the big-endian X are zero.
    // limb[3] is the most-significant 64 bits.
    if (dp_bits <= 0) return true;
    if (dp_bits <= 64) {
        return (x[3] & top_n_bits_mask(dp_bits)) == 0;
    }
    // dp_bits > 64: limb[3] must be entirely zero, plus top (dp_bits-64)
    // bits of limb[2] must be zero. Cap at 128.
    if (x[3] != 0) return false;
    if (dp_bits >= 128) return false;  // unsupported
    return (x[2] & top_n_bits_mask(dp_bits - 64)) == 0;
}

// ---------------------------------------------------------------------------
// Threadgroup-cooperative Montgomery batch inversion.
// Each of the 32 threads in a threadgroup contributes its Jacobian Z. Lane 0
// builds prefix products, computes ONE mod_inv on the final product, then
// walks backwards to recover per-thread inverses. All 32 threads then
// compute their affine X = X * z_inv^2 in parallel.
// Threadgroup memory layout (caller passes in pointers):
//   tg_z_coords [BATCH * 4]  - original Z per lane (input to the inversion)
//   tg_products [BATCH * 4]  - prefix products (lane 0 only writes)
//   tg_z_inv    [BATCH * 4]  - per-lane Z^-1 (lane 0 writes; all read)
// ---------------------------------------------------------------------------

inline void tg_batch_jac_to_affine(
    thread       ulong          affine_x[4],
    thread       ulong          affine_y[4],
    thread const ulong          jac_x[4],
    thread const ulong          jac_y[4],
    thread const ulong          jac_z[4],
    threadgroup  ulong*         tg_z_coords,   // [BATCH * 4]
    threadgroup  ulong*         tg_products,   // [BATCH * 4]
    threadgroup  ulong*         tg_z_inv,      // [BATCH * 4]
    uint                        lane)
{
    // Step 1: every lane stores its Z (or sentinel 1 if Z=0/infinity, so
    // the prefix product doesn't collapse to zero -- we'll zero the
    // affine coords for those lanes at the end).
    bool is_inf = limbs_zero(jac_z);
    {
        ulong z_for_chain[4];
        if (is_inf) {
            // Sentinel: Z=1 keeps the chain product nonzero. Lane gets
            // is_inf-zeroed output below, so this Z is never visible
            // outside the chain.
            set4(z_for_chain, 1, 0, 0, 0);
        } else {
            copy4(z_for_chain, jac_z);
        }
        for (int i = 0; i < 4; ++i) {
            tg_z_coords[lane * 4 + i] = z_for_chain[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2-4: lane 0 does the serial prefix product, single inversion,
    // and back-pass to fill tg_z_inv[]. Other lanes wait at the next
    // barrier.
    if (lane == 0) {
        // Prefix products: products[0] = z[0]; products[i] = products[i-1] * z[i].
        ulong prod[4];
        for (int i = 0; i < 4; ++i) prod[i] = tg_z_coords[i];
        for (int i = 0; i < 4; ++i) tg_products[i] = prod[i];

        for (uint t = 1; t < KANGAROO_BATCH_SIZE; ++t) {
            ulong z_t[4];
            for (int i = 0; i < 4; ++i) z_t[i] = tg_z_coords[t * 4 + i];
            ulong next[4];
            mod_mul(next, prod, z_t);
            for (int i = 0; i < 4; ++i) {
                prod[i] = next[i];
                tg_products[t * 4 + i] = next[i];
            }
        }

        // Single modular inversion of the final product.
        ulong inv_all[4];
        mod_inv(inv_all, prod);

        // Back-pass: recover each lane's z_inv.
        //   z_inv[t]   = inv_all * products[t-1]
        //   inv_all   *= z_coords[t]
        // End condition: z_inv[0] = inv_all (after the loop).
        for (int t = (int)KANGAROO_BATCH_SIZE - 1; t > 0; --t) {
            ulong prev_prod[4], my_z_inv[4];
            for (int i = 0; i < 4; ++i) prev_prod[i] = tg_products[(t - 1) * 4 + i];
            mod_mul(my_z_inv, inv_all, prev_prod);
            for (int i = 0; i < 4; ++i) tg_z_inv[t * 4 + i] = my_z_inv[i];

            // inv_all *= z_coords[t] for the next iteration.
            ulong z_t[4], new_inv_all[4];
            for (int i = 0; i < 4; ++i) z_t[i] = tg_z_coords[t * 4 + i];
            mod_mul(new_inv_all, inv_all, z_t);
            for (int i = 0; i < 4; ++i) inv_all[i] = new_inv_all[i];
        }
        // Lane 0's inverse is inv_all after all back-propagation.
        for (int i = 0; i < 4; ++i) tg_z_inv[i] = inv_all[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 5: every lane converts its Jacobian -> affine in parallel.
    if (is_inf) {
        set4(affine_x, 0, 0, 0, 0);
        set4(affine_y, 0, 0, 0, 0);
    } else {
        ulong my_z_inv[4];
        for (int i = 0; i < 4; ++i) my_z_inv[i] = tg_z_inv[lane * 4 + i];

        ulong z_inv2[4], z_inv3[4];
        mod_sqr(z_inv2, my_z_inv);
        mod_mul(z_inv3, z_inv2, my_z_inv);

        mod_mul(affine_x, jac_x, z_inv2);
        mod_mul(affine_y, jac_y, z_inv3);
    }
    // No barrier here: the next loop iteration starts with state writes
    // to thread-private px/py/pz (no threadgroup memory) and the next
    // barrier on the next DP check naturally fences any later use of
    // tg_* memory.
}

// ---------------------------------------------------------------------------
// Kangaroo step kernel.
// State buffers (one slot per kangaroo, 4 ulong each):
//   x_buf, y_buf, z_buf : Jacobian point coordinates
//   dist_buf            : walked distance (256-bit, no mod)
//   type_buf            : 0 = tame, 1 = wild
// Threadgroup contract: KANGAROO_BATCH_SIZE = 32 threads per group.
// Host enforces num_kangaroos % 32 == 0 and threadsPerThreadgroup = 32.
// DP detection runs every DP_CHECK_INTERVAL steps OR on the last step.
// On detection, the kernel atomically reserves a slot in dp_records[],
// emits a 74-byte JLPDistinguishedPointV2-shaped record, and CONTINUES
// walking. Multiple DPs per round are fine.
// ---------------------------------------------------------------------------

kernel void kangaroo_step(
    device       ulong*           x_buf       [[buffer(0)]],
    device       ulong*           y_buf       [[buffer(1)]],
    device       ulong*           z_buf       [[buffer(2)]],   // NEW: Jacobian Z
    device       ulong*           dist_buf    [[buffer(3)]],
    device const uchar*           type_buf    [[buffer(4)]],
    device const ulong*           jump_x      [[buffer(5)]],
    device const ulong*           jump_y      [[buffer(6)]],
    device const ulong*           jump_d      [[buffer(7)]],
    constant     uint&            count       [[buffer(8)]],
    constant     uint&            steps       [[buffer(9)]],
    constant     uint&            dp_bits     [[buffer(10)]],
    constant     ulong&           work_id     [[buffer(11)]],
    device       DPRecord*        dp_records  [[buffer(12)]],
    device       atomic_uint*     dp_count    [[buffer(13)]],
    constant     uint&            dp_max      [[buffer(14)]],
    uint                          gid         [[thread_position_in_grid]],
    uint                          lane        [[thread_position_in_threadgroup]])
{
    // Threadgroup scratch for batch inversion. Three arrays of
    // [BATCH][4] ulong = 3 * 32 * 4 * 8 = 3 KiB.
    threadgroup ulong tg_z_coords [KANGAROO_BATCH_SIZE * 4];
    threadgroup ulong tg_products [KANGAROO_BATCH_SIZE * 4];
    threadgroup ulong tg_z_inv    [KANGAROO_BATCH_SIZE * 4];

    // We MUST NOT early-return before threadgroup barriers; every thread
    // in the threadgroup must hit every barrier or the device deadlocks.
    // The host enforces num_kangaroos % BATCH_SIZE == 0 so gid >= count
    // never happens during steady-state operation. But if a future config
    // accidentally violates that, we fall through with a sentinel state
    // (inactive thread; reads/writes are still in-bounds because the
    // host allocates rounded-up-to-batch buffers).
    const bool active = (gid < count);

    // Load state. If inactive, use sentinel values (Z = 1, identity-ish)
    // so the batch-inversion chain doesn't see a zero from this lane.
    ulong px[4], py[4], pz[4], dist[4];
    if (active) {
        for (int i = 0; i < 4; ++i) {
            px[i]   = x_buf[gid * 4 + i];
            py[i]   = y_buf[gid * 4 + i];
            pz[i]   = z_buf[gid * 4 + i];
            dist[i] = dist_buf[gid * 4 + i];
        }
    } else {
        // Sentinel: doesn't matter mathematically; just keep Z != 0.
        set4(px, 1, 0, 0, 0);
        set4(py, 1, 0, 0, 0);
        set4(pz, 1, 0, 0, 0);
        set4(dist, 0, 0, 0, 0);
    }
    uchar ktype = active ? type_buf[gid] : (uchar)0;

    // Inner walk loop.
    for (uint step = 0; step < steps; ++step) {
        // Jump selection: low bits of Jacobian X (low bits are sufficient
        // for pseudo-random selection; only DP detection needs the true
        // affine X). Same trick as the CUDA path.
        const uint jidx = (uint)(px[0] & KANGAROO_JUMP_MASK);
        ulong jx[4], jy[4], jd[4];
        for (int i = 0; i < 4; ++i) {
            jx[i] = jump_x[jidx * 4 + i];
            jy[i] = jump_y[jidx * 4 + i];
            jd[i] = jump_d[jidx * 4 + i];
        }

        // Mixed Jacobian + affine point addition. NO modular inversion
        // in the inner loop -- this is the main perf win vs the affine
        // baseline (which did 1 mod_inv per step).
        ulong rx[4], ry[4], rz[4];
        jac_add_mixed(rx, ry, rz, px, py, pz, jx, jy);
        copy4(px, rx); copy4(py, ry); copy4(pz, rz);

        // Distance += jump distance (no modular reduction; host stores
        // raw 256-bit value).
        ulong c = 0;
        ulong d0 = addc(dist[0], jd[0], c);
        ulong d1 = addc(dist[1], jd[1], c);
        ulong d2 = addc(dist[2], jd[2], c);
        ulong d3 = addc(dist[3], jd[3], c);
        dist[0] = d0; dist[1] = d1; dist[2] = d2; dist[3] = d3;

        // Infinity recovery: if the addition produced point-at-infinity
        // (P == -Q for the jump-table entry), reseed to jump 0 and keep
        // going. Same recovery path the CUDA kernel uses
        // (kangaroo_kernel.cu:1413), but unlike CUDA we MUST NOT
        // `continue;` here -- doing so would skip this thread out of
        // any DP-check threadgroup barriers below, deadlocking the
        // threadgroup since other lanes would still enter the barrier.
        // Instead, fix up px/py/pz inline and let this iteration's
        // DP check run on the recovered point. If the recovered point
        // happens to be a DP that's fine; the affine X is well-defined.
        // We do tag `recovered_this_step` so we skip the recovered-point
        // DP report (which would be a false positive anchored on the
        // jump-0 point rather than the walk).
        bool recovered_this_step = false;
        if (limbs_zero(pz)) {
            for (int i = 0; i < 4; ++i) {
                px[i] = jump_x[i];
                py[i] = jump_y[i];
            }
            set4(pz, 1, 0, 0, 0);   // affine reseed
            recovered_this_step = true;
            // Distance is left as-is so the collision math still works
            // (the host folds the cumulative distance on collision).
        }

        // DP-check gating: every DP_CHECK_INTERVAL steps OR on the very
        // last step of the round. This expression evaluates uniformly
        // across the threadgroup (depends only on `step` and `steps`,
        // both kernel-uniform), so all lanes enter or none enter the
        // batch-inversion block. That is the load-bearing invariant
        // that keeps threadgroup_barrier() inside tg_batch_jac_to_affine
        // from deadlocking.
        const bool do_dp_check =
            (((step + 1) % DP_CHECK_INTERVAL) == 0u) ||
            (step == steps - 1u);

        if (do_dp_check) {
            ulong affine_x[4], affine_y[4];
            tg_batch_jac_to_affine(affine_x, affine_y,
                                   px, py, pz,
                                   tg_z_coords, tg_products, tg_z_inv,
                                   lane);

            if (active && !recovered_this_step
                && is_distinguished(affine_x, (int)dp_bits)) {
                uint slot = atomic_fetch_add_explicit(dp_count, 1u,
                                                     memory_order_relaxed);
                if (slot < dp_max) {
                    DPRecord rec;
                    // Byte-store work_id as little-endian into the packed
                    // struct. Both Metal device and Apple Silicon host are
                    // LE so the host can memcpy 8 bytes straight into a
                    // uint64_t.
                    ulong w = work_id;
                    for (int b = 0; b < 8; ++b) {
                        rec.work_id_le[b] = (uchar)(w >> (8 * b));
                    }
                    limbs_to_be(rec.x_be, affine_x);
                    limbs_to_be(rec.d_be, dist);
                    rec.type = ktype;
                    rec.dp_bits = (uchar)dp_bits;
                    dp_records[slot] = rec;
                }
            }
            // After conversion the affine values are scratch; px/py/pz
            // are still valid Jacobian for the same point so we keep
            // walking from them.
        }
    }

    // Persist updated Jacobian state (only active lanes write back).
    if (active) {
        for (int i = 0; i < 4; ++i) {
            x_buf[gid * 4 + i]    = px[i];
            y_buf[gid * 4 + i]    = py[i];
            z_buf[gid * 4 + i]    = pz[i];
            dist_buf[gid * 4 + i] = dist[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Initial point setup helper. priv_to_pub via repeated double-and-add.
// Used by test_metal_secp256k1 to validate the field arithmetic. Kept on
// the AFFINE path -- this kernel is a KAT vehicle, not the hot path, and
// the affine version is the simpler reference implementation.
// ---------------------------------------------------------------------------

kernel void priv_to_pub(
    device const ulong*  priv_keys [[buffer(0)]],   // count * 4
    device       ulong*  pub_x     [[buffer(1)]],   // count * 4
    device       ulong*  pub_y     [[buffer(2)]],   // count * 4
    constant     ulong*  Gx        [[buffer(3)]],   // 4 ulongs (generator X)
    constant     ulong*  Gy        [[buffer(4)]],   // 4 ulongs (generator Y)
    constant     uint&   count     [[buffer(5)]],
    uint                 gid       [[thread_position_in_grid]])
{
    if (gid >= count) return;

    // Square-and-add: walk priv from MSB to LSB, doubling result and adding
    // base when the bit is set. This is intentionally simple (not constant-
    // time, not windowed) -- this kernel exists only as a KAT vehicle for
    // the field arithmetic.
    ulong scalar[4];
    for (int i = 0; i < 4; ++i) scalar[i] = priv_keys[gid * 4 + i];

    bool have_result = false;
    ulong rx[4], ry[4];
    ulong basex[4] = { Gx[0], Gx[1], Gx[2], Gx[3] };
    ulong basey[4] = { Gy[0], Gy[1], Gy[2], Gy[3] };

    for (int bit = 0; bit < 256; ++bit) {
        const int limb = bit >> 6;
        const int b    = bit & 63;
        if ((scalar[limb] >> b) & 1ul) {
            if (!have_result) {
                copy4(rx, basex);
                copy4(ry, basey);
                have_result = true;
            } else {
                ulong nx[4], ny[4];
                point_op_affine(nx, ny, rx, ry, basex, basey);
                copy4(rx, nx);
                copy4(ry, ny);
            }
        }
        // Double base for the next bit.
        ulong nx[4], ny[4];
        point_double_affine(nx, ny, basex, basey);
        copy4(basex, nx);
        copy4(basey, ny);
    }

    if (!have_result) {
        // Identity point; emit zeros.
        for (int i = 0; i < 4; ++i) { pub_x[gid * 4 + i] = 0; pub_y[gid * 4 + i] = 0; }
        return;
    }
    for (int i = 0; i < 4; ++i) {
        pub_x[gid * 4 + i] = rx[i];
        pub_y[gid * 4 + i] = ry[i];
    }
}
