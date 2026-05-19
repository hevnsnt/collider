// fused_oob_record.hpp -- device-resident OOB capture for the
// COLLIDER_DEBUG_FUSED_BOUNDS diagnostic build.
//
// The production --brainwallet-v2 path faults with cudaErrorIllegalAddress
// at minute ~45 of continuous scanning. We need to know which bloom probe
// (if any) overruns its declared num_bits. Device-side printf is unusable
// because the TUI owns stdout/stderr and overwrites them; the diagnostic
// has to land in the persistent session log at ~/.collider/logs/.
//
// Design: a single __managed__ BloomOobRecord global lives in
// fused_pipeline.cu. On the first OOB the kernel CAS-claims the record
// and writes its diagnostics. After every cudaStreamSynchronize the host
// polls via the extern "C" getter; if claimed, the host emits a
// milestone("bloom_oob", ...) into the session log and resets the flag.
// Subsequent OOBs in the same batch are ignored (only the first matters
// for attribution).
//
// Guarded entirely by COLLIDER_DEBUG_FUSED_BOUNDS. Zero footprint in
// production builds.
#pragma once

#ifdef COLLIDER_DEBUG_FUSED_BOUNDS

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

struct BloomOobRecord {
    unsigned int claimed;      // 0 = empty, 1 = populated (atomic CAS gate)
    unsigned int path_tag;     // 0 = mask path, 1 = modulo path
    unsigned int probe_idx;    // which of the k hashes within this probe
    unsigned int tid;          // threadIdx.x at fault
    unsigned int bid;          // blockIdx.x at fault
    unsigned long long idx;    // the OOB index
    unsigned long long num_bits;
    unsigned long long bloom_mask;
    unsigned long long h1;
    unsigned long long h2;
};

// Returns true and fills *out with the first OOB observed since the
// last successful poll, then clears the claimed flag so the next OOB
// can be captured. Returns false (and does not touch *out) when no
// OOB is currently recorded. Host-callable. Safe to call from any
// thread; the CAS guarantees a single observer per OOB event.
bool collider_gpu_poll_bloom_oob(BloomOobRecord* out);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // COLLIDER_DEBUG_FUSED_BOUNDS
