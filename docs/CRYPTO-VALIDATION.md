# Crypto Validation

How theCollider verifies that its on-GPU cryptography matches the canonical Bitcoin convention. Read this when adding a new GPU kernel that touches SHA256, RIPEMD160, secp256k1, or the brain-wallet pipeline **(PRO VERSION ONLY)**.

## Why this exists

The most expensive failure mode for a Bitcoin solver is **silent wrong hash**: if the GPU computes the wrong hash160 for a given private key, the bloom filter (or the puzzle target hash) never matches, and the scanner runs at full speed forever finding nothing. theCollider has caught at least three independent versions of this bug class in its history. Every one of them lived undetected for weeks because no test compared GPU output to a CPU reference.

The validation suite below is the standing defense.

## The four validation tests

Located under `tests/`. CUDA-only; built when `COLLIDER_USE_CUDA=ON` (which derives the internal `COLLIDER_BACKEND` to `CUDA`). Each test returns the CTest skip code (77) on hosts without a CUDA device, so the CI matrix on Mac and CPU-only Linux runners skips them cleanly.

### `test_hash_vectors` (CPU only, sanity floor)

CPU reference vs known NIST and Bitcoin test vectors for SHA256, RIPEMD160, and the combined hash160. Confirms the CPU side of every other test's comparison is itself correct. Builds and runs everywhere, including Mac and CPU-only Linux.

Expected: 10 of 10 pass.

### `test_secp256k1_inv` (CUDA)

Calls `secp256k1_test_inverse_correctness` with 64 random scalars (deterministic seed) plus four edge cases (1, 2, small composite, high-bits). On the GPU: computes `inv = mod_inv(a)`, then `product = mod_mul(a, inv)`, then checks `product` is congruent to 1 (mod p). Expects all 64 to pass.

Catches: any bug in `secp256k1.cu`'s `mod_inv`, which underpins `jacobian_to_affine` and the precomputed EC table generation. A broken addition chain (the canonical libsecp256k1-style sequence of squarings and multiplies that computes `a^(p-2) mod p`) fails this test for almost every scalar.

### `test_ec_table_consistency` (CUDA)

Calls `secp256k1_init_table()` to populate the per-GPU precomputed EC table, then `secp256k1_test_table_on_curve` to verify every entry satisfies the secp256k1 curve equation `y^2` congruent to `x^3 + 7` (mod p). Expects 0 off-curve entries out of 1664 (52 windows times 32 entries).

Catches: any error in `jacobian_to_affine` (which uses `mod_inv`), the table-generation kernel itself, or per-GPU table corruption. Independent verification chain from `test_secp256k1_inv`.

### `test_gpu_hash160` (CUDA, the headline test) **(PRO VERSION ONLY)**

End-to-end brain-wallet pipeline test using the production `fused_brain_wallet_batch_fixed_stride` API with no production code modification.

Strategy:

1. Generate N test passphrases on the host (currently 16, including edge cases: empty, 55-byte, 56-byte to exercise the multi-block SHA256 path, and 89-byte).
2. For each passphrase, compute the EXPECTED hash160 via `crypto_cpu.hpp::compute_hash160(SHA256(passphrase))`. This is the canonical CPU reference.
3. Build a small bloom filter (8192 bits, 8 hashes; false-positive rate effectively 0 at N=16) populated with all N expected hash160s, using the SAME bit-slicing scheme as `fused_pipeline.cu`'s `bloom_check_inline`.
4. Call the production `fused_brain_wallet_batch_fixed_stride` with the N passphrases and the oracle bloom.
5. Pass iff `match_count == N`.

Catches:

- SHA256-to-scalar byte-swap bugs: GPU computes a wrong scalar, then wrong pubkey, then wrong hash160, then bloom miss.
- Broken `mod_inv` poisoning the precomputed EC table: same downstream effect.
- `sha256_short` truncation past 55 bytes: the 56-byte and 89-byte test passphrases trigger the multi-block path.
- Missing scalar range checks: scalar 0 or scalar at or above curve order n produces a bogus pubkey at infinity (the host reference computes a real hash160; the GPU computes garbage from a (0, 0) point; the mismatch is flagged).
- Future regressions in any kernel between SHA256(passphrase) input and bloom-probe output.

The bloom-oracle approach means **any divergence from the CPU reference** flags the test, regardless of which intermediate stage broke. This is the core property: the test passes or fails based on byte-for-byte agreement with the trusted oracle, not on any internal kernel assertion.

## Where the CPU reference lives

`src/core/crypto_cpu.hpp` is the header-only inline implementation of:

- `cpu::SHA256` class (verified against NIST vectors in `test_hash_vectors`).
- `cpu::RIPEMD160` class (verified against RFC 1320 vectors).
- `cpu::compute_hash160(privkey_bytes[32])` runs the full Bitcoin chain: `privkey -> ec_mul(G) -> compressed_pubkey -> SHA256 -> RIPEMD160`.

This is the trusted oracle. If the CPU reference is wrong, all four tests can pass against wrong GPU output and still produce a non-functioning scanner. Treat changes to `crypto_cpu.hpp` with the same care as a kernel change: validate against external tools (`openssl`, `libsecp256k1`, Python `hashlib`) before committing.

## Adding new validation tests

When a new GPU kernel is added that touches the hash chain or any secp256k1 operation, add a parity test in `tests/`:

1. If the kernel exposes a host-callable API, use it directly and compare output to a CPU reference. Write the CPU reference in `crypto_cpu.hpp` first if it doesn't exist.
2. If the kernel only operates on `__device__` internals, add a minimal test wrapper kernel plus an `extern "C"` host function in the production `.cu` file (see `secp256k1_test_inverse_correctness` for the pattern). Mark them clearly as test infrastructure.
3. Register the new test in `CMakeLists.txt` under the CUDA-only block with `set_tests_properties(<name> PROPERTIES SKIP_RETURN_CODE 77)` so it skips cleanly on non-CUDA hosts.

## Running the validation suite

```bash
# Linux / Windows (CUDA available)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCOLLIDER_USE_CUDA=ON \
  -DCOLLIDER_BUILD_TESTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89"
cmake --build build --config Release
ctest --test-dir build --output-on-failure
```

For first-time CUDA validation on Windows, two wrapper scripts in `scripts/` run the suite end-to-end (the names reflect their original purpose during the Wave 0 / Wave 1 fix sweep; they still work):

```cmd
scripts\wave-0-windows-validate.bat
scripts\wave-1-windows-validate.bat
```

## When a test fails

A failure here is almost always a real GPU correctness bug. Do NOT "fix" it by adjusting the test's expected value. Investigate the kernel.

If the failure is a regression after a recent commit, bisect:

```bash
git bisect run ctest --test-dir build -R GpuHash160
```

If the failure is on a new GPU architecture (your tests pass on RTX 4090 but fail on RTX 5090, say), suspect register pressure, warp divergence behavior, or arch-specific PTX intrinsics. Check `cuobjdump -ptx <binary>` for surprises and consider `__launch_bounds__` adjustments.

## Hand-typed expected values are a known failure mode

A long-lived bug in `tests/test_hash_vectors.cpp` once shipped with the wrong expected value at line 247 (the Puzzle 2 pubkey hash160 was typed as `91b24bf...` instead of the correct `06afd46b...`). This made the hash-chain tests show "1 fail" forever, which masked any other regression that came along.

The rule: when adding new test vectors, verify expected values against an independent ground truth (Python `hashlib`, `openssl`, or `libsecp256k1`) before committing. Hand-typed expected values are a known failure mode.

## Where to go next

| Topic                                    | Doc                                  |
| ---------------------------------------- | ------------------------------------ |
| Building with tests enabled (Linux)      | [BUILD-LINUX.md](BUILD-LINUX.md)     |
| Building with tests enabled (Windows)    | [BUILD-WINDOWS.md](BUILD-WINDOWS.md) |
| Building with tests enabled (macOS)      | [BUILD-MACOS.md](BUILD-MACOS.md)     |
| Source-tree map (where the kernels live) | [ARCHITECTURE.md](ARCHITECTURE.md)   |
