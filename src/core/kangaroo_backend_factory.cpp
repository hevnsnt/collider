/**
 * Kangaroo backend factory.
 *
 * Two responsibilities:
 *
 *   1. Enumerate which backends this binary was built with. CPU is
 *      always built (it's the portable fallback); CUDA is built when
 *      COLLIDER_USE_RCKANGAROO is set; Metal is built on Apple
 *      platforms with COLLIDER_USE_METAL. The build-system gates
 *      compile the relevant adapter sources; the headers below are
 *      included unconditionally so the dispatch is in one place.
 *
 *   2. Construct the requested backend. Pre-1.5.x this dispatched
 *      at compile time via #if defined(COLLIDER_USE_RCKANGAROO) /
 *      __APPLE__, so each binary had exactly one backend. Operators
 *      who wanted to A/B CPU vs GPU on the same machine had to build
 *      two binaries. The runtime enum lets a single binary expose
 *      every backend the build pulled in; the CLI flag / settings
 *      panel picks which one this run uses.
 */

#include "kangaroo_backend.hpp"
#include "cpu_kangaroo_backend.hpp"

#if defined(COLLIDER_USE_RCKANGAROO)
#include "../gpu/cuda_rckangaroo_backend.hpp"
#endif

#if defined(__APPLE__) && defined(COLLIDER_USE_METAL)
#include "../gpu/metal_kangaroo_backend.hpp"
#endif

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <vector>

namespace collider {
namespace kangaroo {

std::string backend_kind_label(BackendKind kind) {
    switch (kind) {
        case BackendKind::CPU:   return "cpu";
        case BackendKind::CUDA:  return "cuda";
        case BackendKind::METAL: return "metal";
    }
    return "unknown";  // unreachable; switch is exhaustive
}

bool parse_backend_kind(const std::string& label, BackendKind& out) {
    std::string lower;
    lower.reserve(label.size());
    for (char c : label) {
        lower.push_back(static_cast<char>(
            std::tolower(static_cast<unsigned char>(c))));
    }
    if (lower == "cpu")   { out = BackendKind::CPU;   return true; }
    if (lower == "cuda")  { out = BackendKind::CUDA;  return true; }
    if (lower == "metal") { out = BackendKind::METAL; return true; }
    return false;
}

std::vector<BackendKind> available_backends() {
    // Preferred-first ordering: the most-capable backend in this build
    // is the operator's default. CUDA (NVIDIA) > Metal (Apple) > CPU.
    std::vector<BackendKind> kinds;
#if defined(COLLIDER_USE_RCKANGAROO)
    kinds.push_back(BackendKind::CUDA);
#endif
#if defined(__APPLE__) && defined(COLLIDER_USE_METAL)
    kinds.push_back(BackendKind::METAL);
#endif
    // CPU is always available (always compiled in).
    kinds.push_back(BackendKind::CPU);
    return kinds;
}

BackendKind default_backend() {
    // available_backends() guarantees at least CPU is in the list, so
    // .front() is always valid. The preferred-first ordering means
    // .front() is the most-capable backend this binary has.
    return available_backends().front();
}

std::unique_ptr<IKangarooBackend> create_kangaroo_backend(
    BackendKind kind,
    const std::vector<int>& gpu_ids)
{
    switch (kind) {
        case BackendKind::CPU:
            (void)gpu_ids;  // CPU backend ignores GPU ids.
            return std::make_unique<CpuKangarooBackend>();

        case BackendKind::CUDA:
#if defined(COLLIDER_USE_RCKANGAROO)
            return std::make_unique<CudaRCKangarooBackend>(gpu_ids);
#else
            throw std::runtime_error(
                "CUDA backend requested but this binary was built without "
                "COLLIDER_USE_RCKANGAROO. Rebuild with -DCOLLIDER_USE_CUDA=ON.");
#endif

        case BackendKind::METAL:
#if defined(__APPLE__) && defined(COLLIDER_USE_METAL)
            (void)gpu_ids;  // Metal selects the system default device.
            return std::make_unique<MetalKangarooBackend>();
#else
            throw std::runtime_error(
                "Metal backend requested but this binary was built without "
                "Apple Metal support. Metal is macOS-only.");
#endif
    }
    // switch is exhaustive over BackendKind; unreachable.
    throw std::runtime_error("unknown BackendKind");
}

}  // namespace kangaroo
}  // namespace collider
