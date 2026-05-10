/**
 * Unity translation unit for the Brain Wallet v2 kernels.
 *
 * Consolidates the v2 kernel sources into a single .cu so nvcc / nvlink
 * have ONE device-symbol graph to resolve instead of N. On Windows,
 * nvlink has a 1MB default stack that overflows at C00000FD on the
 * cumulative symbol graph of collider_gpu + several v2 kernel TUs --
 * see the v1.4.0 build investigation log.
 *
 * Each #include below is treated as raw source paste (NOT a header). The
 * individual files are NOT compiled as standalone TUs (CMakeLists adds
 * only this unity file to GPU_SOURCES). Their layout was already designed
 * to be #include-friendly: each opens the same `collider::gpu::v2`
 * namespace and uses anonymous namespaces for internals.
 *
 * If you add a new v2 kernel .cu file, append its #include here and
 * remove it from GPU_SOURCES in CMakeLists.txt.
 */

// Order matters: brain_wallet_v2.cu defines c_puzzle_targets[] +
// c_puzzle_target_count; the other kernels rely on those being visible.
// puzzle_check.cuh provides the static __device__ priv-checker used by
// every kernel below.
#include "brain_wallet_v2.cu"
#include "weak_prng_kernel.cu"
#include "encoding_munge_kernel.cu"
#include "legacy_kdf_kernel.cu"
#include "electrum_kernel.cu"
#include "multi_address_kernel.cu"
