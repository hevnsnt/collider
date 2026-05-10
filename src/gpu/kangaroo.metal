/*
 * Apple Metal Pollard's Kangaroo kernel for secp256k1 (v1.4.0).
 *
 * Self-contained MSL: 256-bit mod-p arithmetic + affine point add +
 * walk loop + DP detection. Free-licensed; mirrors the CUDA path in
 * src/gpu/kangaroo_kernel.cu but uses affine coordinates for clarity
 * (one modular inverse per step). Targets correctness over peak
 * throughput; expected K factor ~5-10 vs RCKangaroo's K=1.15.
 *
 * Throughput note: a Jacobian-coordinate rewrite plus Montgomery
 * batch inversion (one inversion amortized over N kangaroos via
 * prefix-product + single inversion + reverse pass) would yield
 * roughly 5-10x throughput. Implementing it requires Mac hardware for
 * KAT validation and per-kernel perf comparison; landed as a
 * standalone effort once that test loop is stable.
 *
 * Field representation: a 256-bit element is uint4 (4 packed uint64).
 *   limb 0 = bits   0..63
 *   limb 1 = bits  64..127
 *   limb 2 = bits 128..191
 *   limb 3 = bits 192..255
 *
 * secp256k1 prime: p = 2^256 - 2^32 - 977.
 *   p_limb = (FFFFFFFEFFFFFC2F, FFFFFFFFFFFFFFFF, FFFFFFFFFFFFFFFF, FFFFFFFFFFFFFFFF)
 *
 * Reduction trick: 2^256 ≡ 2^32 + 977 (mod p). After a 256x256 -> 512
 * multiply we split into (high, low) and compute low + high*(2^32+977)
 * to fold high back in.
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
    //   r_init = lo + hi*K0 + hi*2^32   (since 2^256 ≡ 2^32+977 mod p)
    // We do this as r = lo + hi*K0  where K0 = 2^32 + 977.
    //
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
// 256 squarings + ~128 multiplies in the worst case; we walk the exponent
// from MSB to LSB.
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
    // walk's 256 + 248 = 504 mod_muls. A ~1.9x reduction on every
    // affine point_add in the kangaroo step. CRITICAL: x223 step
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
// Affine secp256k1 point operations.
//
// Two primitives (R = 2P, R = P + Q) plus a dispatcher that selects
// between them based on whether P and Q are the same point. The plain
// add formula computes lambda = (qy - py) / (qx - px) which divides by
// zero when qx == px; the doubling formula uses the tangent slope
// lambda = (3*px^2) / (2*py) instead.
//
// Identity (P == -Q, i.e. qx == px && qy == -py) is the only remaining
// undefined input; with a 32-jump table whose distances are unique at
// configuration time, the realistic per-walk-pair probability of two
// kangaroos arriving at exact mutual inverses is birthday-bound 2^-128
// over the secp256k1 group order (~2^256). The dispatcher returns
// (0, 0) for that case; the host re-seeds any kangaroo whose state
// goes to zero on the next round.
// ---------------------------------------------------------------------------

// Constant-time equality check: a == b for 4-limb LE-by-limb operands.
inline bool limbs_equal(thread const ulong a[4], thread const ulong b[4]) {
    return (a[0] == b[0]) & (a[1] == b[1]) & (a[2] == b[2]) & (a[3] == b[3]);
}

// R = P + Q under the strict precondition that P != Q (no doubling)
// and P != -Q (no identity). Used internally by `point_op` after the
// dispatch check.
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

// R = 2 * P. Tangent-slope formula. Used internally and by priv_to_pub.
inline void point_double(thread ulong rx[4], thread ulong ry[4],
                         thread const ulong px[4], thread const ulong py[4])
{
    ulong x2[4], three_x2[4], two_y[4], two_y_inv[4];
    ulong lambda[4], lambda_sq[4], two_x[4];

    mod_sqr(x2, px);                         // x^2
    mod_add(three_x2, x2, x2);
    {
        ulong tmp[4];
        mod_add(tmp, three_x2, x2);
        three_x2[0]=tmp[0]; three_x2[1]=tmp[1];
        three_x2[2]=tmp[2]; three_x2[3]=tmp[3];   // 3*x^2
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

// Dispatcher: R = P + Q with the doubling and identity cases handled.
// The kangaroo step kernel calls this; priv_to_pub also uses it via the
// double-and-add walk. Identity (P == -Q) -> R = (0, 0); host code
// detects that as a stall and re-seeds the affected kangaroo.
inline void point_op(thread ulong rx[4], thread ulong ry[4],
                     thread const ulong px[4], thread const ulong py[4],
                     thread const ulong qx[4], thread const ulong qy[4])
{
    if (limbs_equal(px, qx)) {
        if (limbs_equal(py, qy)) {
            // P == Q -> doubling.
            point_double(rx, ry, px, py);
        } else {
            // P == -Q -> identity. Caller treats (0,0) as "re-seed me".
            rx[0] = rx[1] = rx[2] = rx[3] = 0;
            ry[0] = ry[1] = ry[2] = ry[3] = 0;
        }
        return;
    }
    point_add_distinct(rx, ry, px, py, qx, qy);
}

// ---------------------------------------------------------------------------
// Kangaroo state laid out as:
//   x[count][4], y[count][4], dist[count][4], type[count]
// All in device buffers. One thread per kangaroo.
//
// DP detection: leading-zero bits of x. We test the top limb (x[3]) and
// shift the count if exactly == 0; equivalent to checking that the first
// `dp_bits` MSBs of the big-endian X are zero.
//
// Output: each detected DP appended to dp_records[] under an atomic counter.
// Layout per record (74 bytes, matches JLPDistinguishedPointV2 wire format):
//   work_id (8) || x_be (32) || d_be (32) || type (1) || dp_bits (1)
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

inline void limbs_to_be(uchar out[32], thread const ulong limbs[4]) {
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
    // bits of limb[2] must be zero. Cap at 128; the protocol never goes
    // higher (at dp_bits=128 the chunk is ~unfindable on any GPU).
    if (x[3] != 0) return false;
    if (dp_bits >= 128) return false;  // unsupported
    return (x[2] & top_n_bits_mask(dp_bits - 64)) == 0;
}

kernel void kangaroo_step(
    device       ulong*           x_buf       [[buffer(0)]],   // count*4 ulongs
    device       ulong*           y_buf       [[buffer(1)]],
    device       ulong*           dist_buf    [[buffer(2)]],
    device const uchar*           type_buf    [[buffer(3)]],   // count uchars
    device const ulong*           jump_x      [[buffer(4)]],   // 32*4 ulongs
    device const ulong*           jump_y      [[buffer(5)]],
    device const ulong*           jump_d      [[buffer(6)]],   // 32*4 ulongs (distance)
    constant     uint&            count       [[buffer(7)]],
    constant     uint&            steps       [[buffer(8)]],
    constant     uint&            dp_bits     [[buffer(9)]],
    constant     ulong&           work_id     [[buffer(10)]],
    device       DPRecord*        dp_records  [[buffer(11)]],
    device       atomic_uint*     dp_count    [[buffer(12)]],
    constant     uint&            dp_max      [[buffer(13)]],
    uint                          gid         [[thread_position_in_grid]])
{
    if (gid >= count) return;

    // Load state.
    ulong px[4], py[4], dist[4];
    for (int i = 0; i < 4; ++i) {
        px[i]   = x_buf[gid * 4 + i];
        py[i]   = y_buf[gid * 4 + i];
        dist[i] = dist_buf[gid * 4 + i];
    }
    uchar ktype = type_buf[gid];

    for (uint step = 0; step < steps; ++step) {
        // Pick jump from low log2(KANGAROO_JUMP_TABLE_SIZE) bits of px[0].
        const uint jidx = (uint)(px[0] & KANGAROO_JUMP_MASK);
        ulong jx[4], jy[4], jd[4];
        for (int i = 0; i < 4; ++i) {
            jx[i] = jump_x[jidx * 4 + i];
            jy[i] = jump_y[jidx * 4 + i];
            jd[i] = jump_d[jidx * 4 + i];
        }

        ulong rx[4], ry[4];
        point_op(rx, ry, px, py, jx, jy);
        for (int i = 0; i < 4; ++i) {
            px[i] = rx[i];
            py[i] = ry[i];
        }
        // dist += jump distance (mod 2^256, no reduction needed; the host
        // stores it as-is and folds on use).
        ulong c = 0;
        ulong d0 = addc(dist[0], jd[0], c);
        ulong d1 = addc(dist[1], jd[1], c);
        ulong d2 = addc(dist[2], jd[2], c);
        ulong d3 = addc(dist[3], jd[3], c);
        dist[0] = d0; dist[1] = d1; dist[2] = d2; dist[3] = d3;

        if (is_distinguished(px, (int)dp_bits)) {
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
                limbs_to_be(rec.x_be, px);
                limbs_to_be(rec.d_be, dist);
                rec.type = ktype;
                rec.dp_bits = (uchar)dp_bits;
                dp_records[slot] = rec;
            }
        }
    }

    // Persist updated state.
    for (int i = 0; i < 4; ++i) {
        x_buf[gid * 4 + i]    = px[i];
        y_buf[gid * 4 + i]    = py[i];
        dist_buf[gid * 4 + i] = dist[i];
    }
}

// ---------------------------------------------------------------------------
// Initial point setup helpers. Test kernel: priv -> pub via repeated double-and-add.
// Used by test_metal_secp256k1 to validate the field arithmetic.
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
                rx[0] = basex[0]; rx[1] = basex[1]; rx[2] = basex[2]; rx[3] = basex[3];
                ry[0] = basey[0]; ry[1] = basey[1]; ry[2] = basey[2]; ry[3] = basey[3];
                have_result = true;
            } else {
                ulong nx[4], ny[4];
                point_op(nx, ny, rx, ry, basex, basey);
                rx[0]=nx[0]; rx[1]=nx[1]; rx[2]=nx[2]; rx[3]=nx[3];
                ry[0]=ny[0]; ry[1]=ny[1]; ry[2]=ny[2]; ry[3]=ny[3];
            }
        }
        // Double base for the next bit. Reuse the shared point_double
        // helper instead of re-implementing the tangent formula here.
        ulong nx[4], ny[4];
        point_double(nx, ny, basex, basey);
        basex[0]=nx[0]; basex[1]=nx[1]; basex[2]=nx[2]; basex[3]=nx[3];
        basey[0]=ny[0]; basey[1]=ny[1]; basey[2]=ny[2]; basey[3]=ny[3];
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
