// gpu_caps.hpp -- gpu-layer-owned dispatch caps.
//
// Q-T1.1 inversion (2026-05-17): puzzle_optimized.cu and rckangaroo_wrapper.cu
// previously included ../runtime/runtime_control.hpp solely to reference
// `RuntimeControlState::kMaxGpus` in a static_assert that bound a gpu-side
// array size to the runtime-side dispatch ceiling. That made a GPU leaf file
// depend on the runtime layer purely for a single integer constant, in
// violation of the gpu-is-a-leaf module layering.
//
// This header inverts the source of truth. The maximum number of devices
// the GPU dispatcher can address is a property of the gpu layer (it bounds
// PTX-side per-device tables, RCKangaroo's MAX_GPU_CNT, slot arrays in
// puzzle_optimized.cu, etc.). The runtime's RuntimeControlState::kMaxGpus
// is the per-GPU phase array bound; it MUST equal this constant or the
// dispatch loop and the gpu slot tables will drift.
//
// runtime_control.hpp includes this header and static_asserts the equality.
// Bumping the cap requires editing exactly one constant here; the runtime
// assertion fails at compile time if anyone forgets to re-check the slot
// tables in puzzle_optimized.cu (kMaxDevices) or third_party/RCKangaroo's
// defs.h (MAX_GPU_CNT).

#pragma once

namespace collider::gpu {

// Maximum number of GPU devices the dispatcher can address. This bounds:
//   - puzzle_optimized.cu's per-device slot table (kMaxDevices padding)
//   - rckangaroo_wrapper.cu's third_party MAX_GPU_CNT
//   - brain_wallet_gpu.cpp's BrainWalletGPUContext arrays
//   - the runtime's per-GPU phase + enable mask (RuntimeControlState::kMaxGpus)
//
// 8 is enough to address every consumer GPU rig anyone has reported
// running theCollider on; bumping past 8 needs a parallel review of the
// rule-engine batch sizing path (which currently treats num_gpus as a
// small constant for stack-allocated per-GPU work-balancer tables).
inline constexpr int kMaxDispatchableGpus = 8;

}  // namespace collider::gpu
