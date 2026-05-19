# secp256k1 field primitives design rationale

`secp256k1_field.cuh` hosts the libsecp256k1-style 13-multiply
addition-chain modular inverse so BOTH compilation units that need it
(`src/gpu/secp256k1.cu` and `src/gpu/fused_pipeline.cu`) share a single
implementation.

## Why this exists

Before this extraction, `fused_pipeline.cu` carried its own 256-bit
binary exponentiation of `(p-2)`, costing ~256 squarings plus ~256
multiplications per inverse, while `secp256k1.cu` already had the
addition-chain form. Switching the fused kernel to this shared version
replaces ~256 mod_muls with ~13 mod_muls per inverse, a ~1.9x reduction
on the dominant `mod_inv` cost inside `jac_to_affine`.

## Why a header (not a .cu)

Both call sites (`secp256k1.cu`, `fused_pipeline.cu`) define their own
translation-unit-local `struct uint256 { uint32_t limbs[8]; }` and their
own `__device__ void mod_mul(uint256&, const uint256&, const uint256&)`.
Pulling the inverse into a free-standing `.cu` would force a single
canonical `uint256` plus a single canonical `mod_mul`, which is the
structural surgery the next-version pipeline rewrite plans to do (see
`.claude/tasks/v1.5.0-crypto-pipeline-rewrite.md` Phase 1). Until then, a
templated `__device__ inline` function in this header gives every
consumer the same algorithm against its own field types via ADL on
`mod_mul`: the calling TU's namespace `collider::gpu` lookup resolves to
that TU's `mod_mul`, so the inverse stays bit-identical to whichever
`mod_mul` implementation the TU already validates against the KATs.

## Correctness contract

- Input `a` must be reduced mod p (`0 <= a < p`) and non-zero.
  `mod_inv(0)` is undefined for a multiplicative inverse; the caller
  (`jac_to_affine`, `ec_double_jac`, `ec_add_mixed`) never invokes it on
  the point at infinity in practice. Behaviour on `a == 0` is "result is
  0" (which is still mathematically wrong for an inverse, but matches
  the prior `fused_pipeline.cu` binary exponentiation and the
  `secp256k1.cu` addition chain; the bit-equal parity test against the
  prior pipeline relies on this).
- `mod_mul` / `mod_sqr` in the calling TU must be carry-safe (see
  `fused_pipeline.cu` `mod_reduce_512` carry handling). This file does
  not patch field arithmetic; it only sequences calls to whatever
  `mod_mul` the TU exports.
- Validated against `tests/test_secp256k1_inv.cu` (which lives in
  `secp256k1.cu`'s TU) and `tests/test_ec_mul_known_answers.cu` (which
  exercises the fused pipeline TU via `ec_mul`). Both tests must pass
  bit-equal after the migration.

## Naming conventions

The chain naming convention (`xN = a^(2^N - 1)`) and the comments are
copied from `secp256k1.cu::mod_inv` so reviewers can diff the extracted
form against its source. The CRITICAL note about multiplying `x223` by
`x3` (NOT `x2`) is preserved verbatim because the prior failed port made
the same mistake and is the single place every reviewer should look at
first.
