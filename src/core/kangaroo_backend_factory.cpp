/**
 * Kangaroo backend factory.
 *
 * Picks the right IKangarooBackend implementation for this build
 * configuration. The compile-time choice (CUDA / Metal / CPU) is the
 * only #ifdef left in the pool driver after the v1.4 backend-interface
 * refactor; the previous run_pool_mode() body had three sibling
 * #ifdef branches that each independently re-implemented the pool
 * solve flow.
 *
 * Build-system gating (CMakeLists.txt) ensures only the relevant
 * adapter sources are compiled into the binary; the headers are
 * included unconditionally below so the factory dispatch is in one
 * place.
 */

#include "kangaroo_backend.hpp"
#include "cpu_kangaroo_backend.hpp"

#if defined(COLLIDER_USE_RCKANGAROO)
#include "../gpu/cuda_rckangaroo_backend.hpp"
#endif

#if defined(__APPLE__) && defined(COLLIDER_USE_METAL)
#include "../gpu/metal_kangaroo_backend.hpp"
#endif

namespace collider {
namespace kangaroo {

std::unique_ptr<IKangarooBackend> create_kangaroo_backend(
    const std::vector<int>& gpu_ids)
{
#if defined(COLLIDER_USE_RCKANGAROO)
    return std::make_unique<CudaRCKangarooBackend>(gpu_ids);
#elif defined(__APPLE__) && defined(COLLIDER_USE_METAL)
    (void)gpu_ids;  // Metal selects the system default device.
    return std::make_unique<MetalKangarooBackend>();
#else
    (void)gpu_ids;
    return std::make_unique<CpuKangarooBackend>();
#endif
}

}  // namespace kangaroo
}  // namespace collider
