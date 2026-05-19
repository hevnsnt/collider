// secp256k1_field.cuh: shared 13-multiply addition-chain mod_inv used
// by both secp256k1.cu and fused_pipeline.cu. Templated __device__
// inline; resolves mod_mul via ADL against the calling TU's own
// uint256 + field implementation so each consumer stays bit-identical
// to its existing KATs without forcing a canonical type.
// See docs/internals/secp256k1-field.md for design rationale.
#pragma once

#include <cuda_runtime.h>
#include <cassert>
#include <cstdint>
#include <type_traits>

namespace collider {
namespace gpu {

/**
 * Modular inverse via libsecp256k1-style addition chain.
 *
 * Computes r = a^(p-2) mod p in ~255 squarings + ~13 multiplications.
 *
 * Template parameter U256 is the calling TU's uint256 type (any type
 * with `uint32_t limbs[8]`). The function resolves mod_mul via
 * unqualified lookup at instantiation; the calling TU MUST have
 * `__device__ void mod_mul(U256& r, const U256& a, const U256& b)`
 * in scope (defined in `namespace collider::gpu`).
 *
 * Aliasing safety: every sqr/mul writes through a fresh local and
 * then assigns, matching the convention from the source addition
 * chain in secp256k1.cu. Belt-and-suspenders for a function called
 * millions of times per kernel launch.
 */
template <typename U256>
__device__ __forceinline__ void mod_inv_addition_chain(U256& r, const U256& a) {
    auto sqr = [](U256& r_, const U256& x) {
        U256 t; mod_mul(t, x, x); r_ = t;
    };
    auto mul = [](U256& r_, const U256& x, const U256& y) {
        U256 t; mod_mul(t, x, y); r_ = t;
    };

    U256 x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223, t1;

    // x2 = a^3
    sqr(x2, a);  mul(x2, x2, a);
    // x3 = a^7
    sqr(x3, x2); mul(x3, x3, a);

    // x6 = a^(2^6 - 1)
    sqr(x6, x3); sqr(x6, x6); sqr(x6, x6); mul(x6, x6, x3);
    // x9 = a^(2^9 - 1)
    sqr(x9, x6); sqr(x9, x9); sqr(x9, x9); mul(x9, x9, x3);
    // x11 = a^(2^11 - 1)
    sqr(x11, x9); sqr(x11, x11); mul(x11, x11, x2);

    // x22 = a^(2^22 - 1)
    sqr(x22, x11);
    for (int i = 1; i < 11; ++i) sqr(x22, x22);
    mul(x22, x22, x11);

    // x44 = a^(2^44 - 1)
    sqr(x44, x22);
    for (int i = 1; i < 22; ++i) sqr(x44, x44);
    mul(x44, x44, x22);

    // x88 = a^(2^88 - 1)
    sqr(x88, x44);
    for (int i = 1; i < 44; ++i) sqr(x88, x88);
    mul(x88, x88, x44);

    // x176 = a^(2^176 - 1)
    sqr(x176, x88);
    for (int i = 1; i < 88; ++i) sqr(x176, x176);
    mul(x176, x176, x88);

    // x220 = a^(2^220 - 1)
    sqr(x220, x176);
    for (int i = 1; i < 44; ++i) sqr(x220, x220);
    mul(x220, x220, x44);

    // x223 = a^(2^223 - 1).
    // CRITICAL: multiply by x3 (a^7), NOT x2 (a^3). The libsecp256k1
    // chain has (2^220 - 1) * 2^3 + 7 = 2^223 - 1; multiplying by
    // a^3 instead gives 2^223 - 5 (off by 4) and poisons the tail.
    // The pre-v1.4.0 port made this mistake (commit e8fc629, reverted
    // as 5c673c5); the KAT tests/test_secp256k1_inv.cu reports 63/64
    // wrong when this multiplier is wrong, which is the single fastest
    // signal that the chain has been corrupted.
    sqr(x223, x220); sqr(x223, x223); sqr(x223, x223);
    mul(x223, x223, x3);

    // Tail: x223^(2^23) * x22 ; ^(2^5) * a ; ^(2^3) * x2 ; ^(2^2) * a.
    // Produces a^(p-2) where p = 2^256 - 2^32 - 977.
    sqr(t1, x223);
    for (int i = 1; i < 23; ++i) sqr(t1, t1);
    mul(t1, t1, x22);
    for (int i = 0; i < 5; ++i) sqr(t1, t1);
    mul(t1, t1, a);
    for (int i = 0; i < 3; ++i) sqr(t1, t1);
    mul(t1, t1, x2);
    sqr(t1, t1); sqr(t1, t1);
    mul(t1, t1, a);

    r = t1;
}

/**
 * Mixed-coordinate point addition: R = P + Q where P is Jacobian and Q is affine.
 *
 * B1 (2026-05-15): hosts the SHARED implementation of ec_add_mixed
 * for both secp256k1.cu and fused_pipeline.cu. Before extraction, the fused
 * pipeline carried a copy that omitted the H==0 / r==0 special case present
 * in secp256k1.cu lines 580-619. When H = U2 - X1 = 0 (i.e. P and Q have the
 * same affine x-coordinate), the prior fused-pipeline copy fell through into
 * the generic add formula with H = 0; HH = 0; HHH = 0; V = 0; X3 = r^2 - 0
 * -2*0 = r^2; the resulting Jacobian R was NOT the point at infinity but a
 * degenerate non-zero (X, Y, Z=0) tuple where Z came out 0 because Z3 = Z1*H
 * = 0. The next iteration's is_infinity() check (Z.is_zero()) would then
 * silently RESET R to Q (the "if (P.is_infinity())" branch at the top), so
 * the entire scalar accumulator built up to this window was discarded. For
 * brain-wallet hashes where window k happens to contain a value that makes
 * windowed_partial == +/- table[w][window_val] at affine x, the kernel
 * produced WRONG public keys, silently emitting the wrong h160 and missing
 * legitimate hits.
 *
 * Fix: detect H==0 here and dispatch to the right operation:
 *   - If r == 0 too (P == Q at affine x), call ec_double_jacobian via the
 *     calling TU's name (resolved by ADL).
 *   - If r != 0 (P == -Q), set R = infinity. The next iteration's
 *     is_infinity() correctly bypasses the reset because P.is_infinity() is
 *     the branch that handles Z==0 by copying Q in.
 *
 * The doubling-required branch uses unqualified lookup against the calling
 * TU's `ec_double_jacobian` (secp256k1.cu) or `ec_double_jac`
 * (fused_pipeline.cu). The two TUs spell the function differently, so the
 * template takes the doubler as a callable parameter rather than relying on
 * ADL on the bare name.
 *
 * Formula choice: ports the secp256k1.cu variant (r = 2*(S2-Y1), I = 4*HH,
 * J = H*I, V = X1*I, Y3 = r*(V-X3) - 2*Y1*J, Z3 = 2*Z1*H). Mathematically
 * equivalent to the prior fused-pipeline variant under any
 * (X/Z^2, Y/Z^3) affine projection, so the h160 outputs after
 * jac_to_affine are bit-identical. KAT validation: tests/test_ec_mul_known_answers,
 * tests/test_gpu_hash160, and tests/test_multi_gpu_brain_wallet must
 * remain green after the migration.
 *
 * Z-init: the calling TU may or may not have a `set_one()` method on its
 * uint256 (secp256k1.cu does, fused_pipeline.cu does not). The template
 * writes the limbs directly to avoid the dependency.
 *
 * Aliasing contract (T4.6, A-tier wave 1, 2026-05-17):
 *   - R MUST NOT alias P. The body reads `P.X` at line 226 (mod_mul(V,
 *     P.X, I)) and `P.Y` at line 237 (mod_mul(J, P.Y, J)) AFTER writing
 *     `R.X` at lines 229-232 and `R.Y` at line 236. If R == P, those
 *     reads return the partially-written outputs, not the original
 *     inputs. The corresponding doublers (puzzle_optimized.cu::ec_double,
 *     secp256k1.cu::ec_double_jacobian, fused_pipeline.cu::ec_double_jac)
 *     all snapshot P.X/P.Y/P.Z to locals before writing R; ec_add_mixed
 *     does not, by deliberate choice (the body is large and the snapshot
 *     would inflate the register footprint).
 *   - R aliasing Q IS safe. Q is read in the prologue (lines 186-189:
 *     U2 from Q.x, S2 from Q.y) before any writes to R.
 *   - The CUDA `assert()` below is enabled in DEBUG builds (when
 *     `__CUDA_ARCH__` is defined and NDEBUG is not). Production CUDA
 *     Release builds typically define NDEBUG; the assertion compiles
 *     out and adds zero per-thread cost.
 *
 * C++ does not have a constexpr/static_assert way to express "these two
 * runtime pointers are distinct," so the runtime assert is the only
 * mechanism available. The comment above is the structural contract,
 * the assert is the runtime enforcement, and the per-TU doublers
 * already enforce the inverse (R == P IS safe for ec_double_*).
 */
template <typename Jacobian, typename Affine, typename DoubleFn>
__device__ __forceinline__ void ec_add_mixed_shared(
    Jacobian& R,
    const Jacobian& P,
    const Affine& Q,
    DoubleFn&& double_jacobian
) {
    // Runtime alias check: R may not alias P. In debug builds this fires
    // the moment a caller violates the contract; in release builds the
    // assert is a no-op (zero cost). The doublers (which CAN alias)
    // snapshot inputs; this function does not. See contract docblock above.
    assert(static_cast<const void*>(&R) != static_cast<const void*>(&P) &&
           "ec_add_mixed_shared: R must not alias P (aliasing Q is fine)");

    // P at infinity: R = (Q.x, Q.y, 1). The Z=1 init writes limbs directly
    // because the two calling TUs spell their uint256::set_one differently
    // (one has it, one does not).
    if (P.is_infinity()) {
        R.X = Q.x;
        R.Y = Q.y;
        R.Z.limbs[0] = 1;
        #pragma unroll
        for (int i = 1; i < 8; i++) R.Z.limbs[i] = 0;
        return;
    }

    // U256 derives from the Jacobian's X member type. P.X is a const
    // reference here, so strip cv-ref to get the storage type. Both TUs
    // expose `uint256` (8 x uint32_t limbs); the template body never
    // depends on member functions beyond what both provide (.limbs[],
    // .is_zero(), .set_zero()).
    using U256 = typename std::remove_cv<typename std::remove_reference<decltype(P.X)>::type>::type;
    U256 Z1Z1, U2, S2, H, HH, I, J, r, V;

    // Z1Z1 = Z1^2
    mod_sqr(Z1Z1, P.Z);

    // U2 = X2 * Z1Z1
    mod_mul(U2, Q.x, Z1Z1);

    // = Y2 * Z1 * Z1Z1
    mod_mul(S2, Q.y, P.Z);
    mod_mul(S2, S2, Z1Z1);

    // H = U2 - X1
    mod_sub(H, U2, P.X);

    // r = 2 * (S2 - Y1)
    mod_sub(r, S2, P.Y);
    mod_add(r, r, r);

    // CRITICAL G-B1 special case: P and Q share an affine x.
    // - r == 0 means P == Q (same y too): need a doubling, not an add.
    // - r != 0 means P == -Q (opposite y): result is the point at infinity.
    // Without this branch, the generic formula below produces a degenerate
    // (X, Y, Z=0) that the next iteration silently resets to Q, dropping
    // the prior accumulator. Match secp256k1.cu::ec_add_mixed:580-619.
    if (H.is_zero()) {
        if (r.is_zero()) {
            double_jacobian(R, P);
            return;
        } else {
            R.set_infinity();
            return;
        }
    }

    // HH = H^2
    mod_sqr(HH, H);

    // I = 4 * HH
    mod_add(I, HH, HH);
    mod_add(I, I, I);

    // J = H * I
    mod_mul(J, H, I);

    // V = X1 * I
    mod_mul(V, P.X, I);

    // X3 = r^2 - J - 2*V
    mod_sqr(R.X, r);
    mod_sub(R.X, R.X, J);
    mod_sub(R.X, R.X, V);
    mod_sub(R.X, R.X, V);

    // Y3 = r * (V - X3) - 2 * Y1 * J
    mod_sub(V, V, R.X);
    mod_mul(R.Y, r, V);
    mod_mul(J, P.Y, J);
    mod_add(J, J, J);
    mod_sub(R.Y, R.Y, J);

    // Z3 = 2 * Z1 * H
    mod_mul(R.Z, P.Z, H);
    mod_add(R.Z, R.Z, R.Z);
}

}  // namespace gpu
}  // namespace collider
