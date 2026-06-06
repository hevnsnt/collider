// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC

// ============================================================================
// Modifications by SixCyber LLC, licensed under GPLv3 per
// the original work. See LICENSE at the repository root and
// THIRD_PARTY_LICENSES.md for the project-wide dependency inventory.
//
// Modification history for this file:
//   2026-05-21 (v1.5.0):
//     - Build-system changes to support the v1.5 asymmetric KangarooMode
//       (CUDA_SEPARABLE_COMPILATION discipline pinned for downstream
//       test-target device-link correctness; see CMakeLists.txt). No
//       device-code semantics changed in this file for v1.5.
//   v1.4.2:
//     - Added explicit <cstdint> include because NVCC 12.8 on Ubuntu
//       22.04 doesn't pull it through "defs.h" transitively, causing
//       Kparams.DP shift math to fail to resolve uint32_t / uint64_t.
//       Windows and macOS toolchains pre-pull it so they don't see the
//       break; this keeps the Linux GH Actions build green. Upstream
//       this if a future RCKangaroo sync drops the patch.
// ============================================================================

#include <cstdint>

#include "defs.h"
#include "RCGpuUtils.h"

//imp2 table points for KernelA
// theCollider v1.5.x: __align__(16) is REQUIRED. The kernel below reads
// jmp2_table via Copy_int4_x2 (== ((int4*)src)[0/1]), which is a 16-byte
// vectorized load. Without an explicit alignment hint, __constant__ arrays
// of u64 get only u64-natural (8-byte) alignment, and the link-order of
// other __constant__ symbols in the same module decides whether
// jmp2_table happens to land on a 16-byte boundary or only an 8-byte one.
//   * collider-pro.exe link graph: 8-byte-only, kernel hits
//     cudaErrorMisalignedAddress at line 466 (KernelA) on every Turing
//     GPU and on Ampere GPUs under compute-sanitizer.
//   * test_kangaroo_mode_asymmetric.exe link graph: lands 16-aligned by
//     luck, kernel runs.
// Explicit __align__(16) closes this so the alignment is a property of the
// declaration, not the link order. Verified by re-running with
// compute-sanitizer after the fix: no Invalid __global__ stanza on either
// GPU.
__device__ __constant__ __align__(16) u64 jmp2_table[8 * JMP_CNT];

// v1.5.5 walk-step validation (task #7): optional ground-truth trace of a
// single kangaroo (global index 0 = block 0, thread 0, group 0). When
// g_dbg_on is set by the host, KernelA records, for the FIRST g_dbg_cap
// steps that kangaroo takes from its birth point, the post-step affine
// (x, y) limbs and the flagged jmp_ind (index | INV_FLAG | JMP2_FLAG |
// DP_FLAG). The server's checkpoint_replay.walk_step is then cross-checked
// against this to prove the CPU model matches the real silicon before
// challenge_mode is ever enabled.
//
// Compiled ONLY when COLLIDER_WALKSTEP_TRACE is defined (a validation
// build). Release builds exclude it entirely, so the shipped kernel
// carries no debug globals or capture branch. Re-enable with
// -DCOLLIDER_WALKSTEP_TRACE to re-run scripts/validate_walkstep_trace.py.
#ifdef COLLIDER_WALKSTEP_TRACE
#define DBG_CAP STEP_CNT
__device__ u32 g_dbg_on = 0;
__device__ u32 g_dbg_n  = 0;          // steps captured so far (<= DBG_CAP)
__device__ u64 g_dbg_birth_x[4];
__device__ u64 g_dbg_birth_y[4];
__device__ u64 g_dbg_x[DBG_CAP * 4];
__device__ u64 g_dbg_y[DBG_CAP * 4];
__device__ u32 g_dbg_jmp[DBG_CAP];
#endif // COLLIDER_WALKSTEP_TRACE

#ifdef COLLIDER_CHECKPOINT_CAPTURE
// v1.5.5 per-kangaroo checkpoint capture (task #9).
//
// The checkpoint-replay anti-cheat needs, for the kangaroo that lands a DP,
// the ORDERED sequence of its walk DISTANCE every CHECKPOINT_INTERVAL jumps
// plus the loop-state (L1S2) bit at each of those points. The DP record only
// carries the final distance; the chain of intermediate checkpoint distances
// lets the pool server challenge a random segment and replay it forward.
//
// The distance per kangaroo is the signed 192-bit value KernelB accumulates
// into Kparams.Kangs[kang*12 + 8..10] (jmp1/jmp2 add/sub). We capture it from
// INSIDE the KernelB DO_ITER loop -- the exact point the canonical CPU walk
// (src/core/checkpoint_walk.hpp) reproduces -- so the captured chain is what
// the server replays. A per-kangaroo global jump counter (g_ckpt_jumps) is
// persisted in device memory across the STEP_CNT-sized kernel launches so the
// 65536-jump checkpoint cadence is honored regardless of launch boundaries.
//
// Plumbing mirrors COLLIDER_WALKSTEP_TRACE: gated __device__ buffers the host
// arms with cudaMemcpyToSymbol and reads with cudaMemcpyFromSymbol. The ring
// is per (designated) kangaroo so the validation harness can drive a tiny
// solve, read one kangaroo's ordered checkpoints, and replay them against the
// CPU oracle.
//
// Capacity: CHK_MAX_KANG kangaroos x CHK_MAX_CP checkpoints. The host
// designates a contiguous window [g_ckpt_base, g_ckpt_base+CHK_MAX_KANG) of
// global kangaroo indices to capture (default base 0). Each captured leaf is
// the 192-bit distance (3 u64) + the L1S2 bit + a loop-escape flag so the
// host can tell a true capture mismatch from an (expected) jmp3 loop-escape
// segment.
#define CHK_INTERVAL    65536u   // == checkpoint_walk::kCheckpointInterval
#define CHK_MAX_KANG    256u     // kangaroos whose chain we retain on device
#define CHK_MAX_CP      64u      // checkpoints retained per kangaroo (incl birth)

__device__ u32 g_ckpt_on   = 0;          // master enable
__device__ u32 g_ckpt_base = 0;          // first global kang index captured
// Per-kangaroo running jump count (persisted across launches).
__device__ u64 g_ckpt_jumps[CHK_MAX_KANG];
// Per-kangaroo count of checkpoints written so far (incl. birth at index 0).
__device__ u32 g_ckpt_cnt[CHK_MAX_KANG];
// Per-kangaroo loop-escape (jmp3 / KernelC) event count since birth. A
// non-zero value on any segment means the canonical jmp1/jmp2 walk cannot
// reproduce that segment; the host uses it to interpret a replay mismatch.
__device__ u32 g_ckpt_loopesc[CHK_MAX_KANG];
// Ring of checkpoint distances: [kang][cp][3 u64 little-endian signed 192b].
__device__ u64 g_ckpt_dist[CHK_MAX_KANG * CHK_MAX_CP * 3];
// Ring of L1S2 bit at each checkpoint (0/1).
__device__ u8  g_ckpt_l1s2[CHK_MAX_KANG * CHK_MAX_CP];

// Record one checkpoint sample for global kangaroo `gkang` if it falls in the
// captured window and the per-kangaroo ring is not full. `d` is the signed
// 192-bit distance (3 u64), `l1s2_bit` the loop-state bit for the NEXT step.
__device__ __forceinline__ void chk_record(u32 gkang, const u64* d, u32 l1s2_bit)
{
    if (!g_ckpt_on) return;
    u32 base = g_ckpt_base;
    if (gkang < base || gkang >= base + CHK_MAX_KANG) return;
    u32 slot = gkang - base;
    u32 cp = g_ckpt_cnt[slot];
    if (cp >= CHK_MAX_CP) return;
    u64* dst = g_ckpt_dist + (slot * CHK_MAX_CP + cp) * 3;
    dst[0] = d[0]; dst[1] = d[1]; dst[2] = d[2];
    g_ckpt_l1s2[slot * CHK_MAX_CP + cp] = (u8)(l1s2_bit & 1u);
    g_ckpt_cnt[slot] = cp + 1;
}
#endif // COLLIDER_CHECKPOINT_CAPTURE


#define BLOCK_CNT	gridDim.x
#define BLOCK_X		blockIdx.x
#define THREAD_X	threadIdx.x

//coalescing
#define LOAD_VAL_256(dst, ptr, group) { *((int4*)&(dst)[0]) = *((int4*)&(ptr)[BLOCK_SIZE * 4 * BLOCK_CNT * (group)]); *((int4*)&(dst)[2]) = *((int4*)&(ptr)[2 * BLOCK_SIZE + BLOCK_SIZE * 4 * BLOCK_CNT * (group)]); }
#define SAVE_VAL_256(ptr, src, group) { *((int4*)&(ptr)[BLOCK_SIZE * 4 * BLOCK_CNT * (group)]) = *((int4*)&(src)[0]); *((int4*)&(ptr)[2 * BLOCK_SIZE + BLOCK_SIZE * 4 * BLOCK_CNT * (group)]) = *((int4*)&(src)[2]); }


extern __shared__ u64 LDS[]; 

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef OLD_GPU

//this kernel performs main jumps
extern "C" __launch_bounds__(BLOCK_SIZE, 1)
__global__ void KernelA(const TKparams Kparams)
{
	u64* L2x = Kparams.L2 + 2 * THREAD_X + 4 * BLOCK_SIZE * BLOCK_X;
	u64* L2y = L2x + 4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE;
	u64* L2s = L2y + 4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE;
	//list of distances of performed jumps for KernelB
	int4* jlist = (int4*)(Kparams.JumpsList + (u64)BLOCK_X * STEP_CNT * PNT_GROUP_CNT * BLOCK_SIZE / 4);
	jlist += (THREAD_X / 32) * 32 * PNT_GROUP_CNT / 8;
	//list of last visited points for KernelC
	u64* x_last0 = Kparams.LastPnts + 2 * THREAD_X + 4 * BLOCK_SIZE * BLOCK_X;
	u64* y_last0 = x_last0 + 4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE;
      
	u64* jmp1_table = LDS; //32KB
	u16* lds_jlist = (u16*)&LDS[8 * JMP_CNT]; //4KB, must be aligned 16bytes

	int i = THREAD_X;
	while (i < JMP_CNT)
    {	
		*(int4*)&jmp1_table[8 * i + 0] = *(int4*)&Kparams.Jumps1[12 * i + 0];
		*(int4*)&jmp1_table[8 * i + 2] = *(int4*)&Kparams.Jumps1[12 * i + 2];
		*(int4*)&jmp1_table[8 * i + 4] = *(int4*)&Kparams.Jumps1[12 * i + 4];
		*(int4*)&jmp1_table[8 * i + 6] = *(int4*)&Kparams.Jumps1[12 * i + 6];
		i += BLOCK_SIZE;
    }

    __syncthreads(); 

	__align__(16) u64 x[4], y[4], tmp[4], tmp2[4];
	u64 dp_mask64;
	if (Kparams.DP <= 0)
		dp_mask64 = 0ULL;
	else if (Kparams.DP >= 64)
		dp_mask64 = ~0ULL;
	else {
		uint32_t shift = (uint32_t)(64ULL - (uint64_t)Kparams.DP);
		if (shift > 63) shift = 63;
		dp_mask64 = ~((1ULL << shift) - 1ULL);
	}
	u16 jmp_ind;

	//copy kangs from global to L2
	u32 kang_ind = PNT_GROUP_CNT * (THREAD_X + BLOCK_X * BLOCK_SIZE);
	for (u32 group = 0; group < PNT_GROUP_CNT; group++)
	{	
		tmp[0] = Kparams.Kangs[(kang_ind + group) * 12 + 0];
		tmp[1] = Kparams.Kangs[(kang_ind + group) * 12 + 1];
		tmp[2] = Kparams.Kangs[(kang_ind + group) * 12 + 2];
		tmp[3] = Kparams.Kangs[(kang_ind + group) * 12 + 3];
		SAVE_VAL_256(L2x, tmp, group);
		tmp[0] = Kparams.Kangs[(kang_ind + group) * 12 + 4];
		tmp[1] = Kparams.Kangs[(kang_ind + group) * 12 + 5];
		tmp[2] = Kparams.Kangs[(kang_ind + group) * 12 + 6];
		tmp[3] = Kparams.Kangs[(kang_ind + group) * 12 + 7];
		SAVE_VAL_256(L2y, tmp, group);
	}

	u32 L1S2 = Kparams.L1S2[BLOCK_X * BLOCK_SIZE + THREAD_X];

    for (int step_ind = 0; step_ind < STEP_CNT; step_ind++)
    {
        __align__(16) u64 inverse[5];
		u64* jmp_table;
		__align__(16) u64 jmp_x[4];
		__align__(16) u64 jmp_y[4];
		
		//first group
		LOAD_VAL_256(x, L2x, 0);
		jmp_ind = x[0] % JMP_CNT;
		jmp_table = ((L1S2 >> 0) & 1) ? jmp2_table : jmp1_table;
		Copy_int4_x2(jmp_x, jmp_table + 8 * jmp_ind);
		SubModP(inverse, x, jmp_x);
		SAVE_VAL_256(L2s, inverse, 0);
		//the rest
		for (int group = 1; group < PNT_GROUP_CNT; group++)
		{
			LOAD_VAL_256(x, L2x, group);
			jmp_ind = x[0] % JMP_CNT;
			jmp_table = ((L1S2 >> group) & 1) ? jmp2_table : jmp1_table;
			Copy_int4_x2(jmp_x, jmp_table + 8 * jmp_ind);
			SubModP(tmp, x, jmp_x);
			MulModP(inverse, inverse, tmp);
			SAVE_VAL_256(L2s, inverse, group);
		}

		InvModP((u32*)inverse);

        for (int group = PNT_GROUP_CNT - 1; group >= 0; group--)
        {
            __align__(16) u64 x0[4];
            __align__(16) u64 y0[4];
            __align__(16) u64 dxs[4];

			LOAD_VAL_256(x0, L2x, group);
            LOAD_VAL_256(y0, L2y, group);
			jmp_ind = x0[0] % JMP_CNT;
			jmp_table = ((L1S2 >> group) & 1) ? jmp2_table : jmp1_table;
			Copy_int4_x2(jmp_x, jmp_table + 8 * jmp_ind);
			Copy_int4_x2(jmp_y, jmp_table + 8 * jmp_ind + 4);
			u32 inv_flag = (u32)y0[0] & 1;
			if (inv_flag)
			{
				jmp_ind |= INV_FLAG;
				NegModP(jmp_y);
			}
            if (group)
            {
				LOAD_VAL_256(tmp, L2s, group - 1);
				SubModP(tmp2, x0, jmp_x);
				MulModP(dxs, tmp, inverse);
				MulModP(inverse, inverse, tmp2);
            }
			else
				Copy_u64_x4(dxs, inverse);

			SubModP(tmp2, y0, jmp_y);
			MulModP(tmp, tmp2, dxs);
			SqrModP(tmp2, tmp);

			SubModP(x, tmp2, jmp_x);
			SubModP(x, x, x0); 
			SAVE_VAL_256(L2x, x, group); 

			SubModP(y, x0, x);
			MulModP(y, y, tmp);
			SubModP(y, y, y0);
			SAVE_VAL_256(L2y, y, group);

			if (((L1S2 >> group) & 1) == 0) //normal mode, check L1S2 loop
			{
				u32 jmp_next = x[0] % JMP_CNT;
				jmp_next |= ((u32)y[0] & 1) ? 0 : INV_FLAG; //inverted
				L1S2 |= (jmp_ind == jmp_next) ? (1u << group) : 0; //loop L1S2 detected
			}
			else
			{
				L1S2 &= ~(1u << group);
				jmp_ind |= JMP2_FLAG;
			}
			
			if ((x[3] & dp_mask64) == 0)
			{
				u32 kang_ind = (THREAD_X + BLOCK_X * BLOCK_SIZE) * PNT_GROUP_CNT + group;
				u32 ind = atomicAdd(Kparams.DPTable + kang_ind, 1);
				ind = min(ind, DPTABLE_MAX_CNT - 1);
				int4* dst = (int4*)(Kparams.DPTable + Kparams.KangCnt + (kang_ind * DPTABLE_MAX_CNT + ind) * 8);
				dst[0] = ((int4*)x)[0];
				dst[1] = ((int4*)x)[1];
				jmp_ind |= DP_FLAG;
			}

			lds_jlist[8 * THREAD_X + (group % 8)] = jmp_ind;
			if ((group % 8) == 0)
				st_cs_v4_b32(&jlist[(group / 8) * 32 + (THREAD_X % 32)], *(int4*)&lds_jlist[8 * THREAD_X]); //skip L2 cache

#ifdef COLLIDER_WALKSTEP_TRACE
			// v1.5.5 walk-step trace capture (task #7): record kangaroo
			// (0,0,0) only. x0/y0 are the pre-step (birth at i==0) point;
			// x/y are the post-step point; jmp_ind carries the full
			// index | INV_FLAG | JMP2_FLAG | DP_FLAG used this step.
			if (g_dbg_on && BLOCK_X == 0 && THREAD_X == 0 && group == 0)
			{
				u32 i = g_dbg_n;
				if (i < DBG_CAP)
				{
					if (i == 0)
					{
#pragma unroll
						for (int q = 0; q < 4; q++) { g_dbg_birth_x[q] = x0[q]; g_dbg_birth_y[q] = y0[q]; }
					}
#pragma unroll
					for (int q = 0; q < 4; q++) { g_dbg_x[i * 4 + q] = x[q]; g_dbg_y[i * 4 + q] = y[q]; }
					g_dbg_jmp[i] = jmp_ind;
					g_dbg_n = i + 1;
				}
			}
#endif // COLLIDER_WALKSTEP_TRACE

			if (step_ind + MD_LEN >= STEP_CNT) //store last kangs to be able to find loop exit point
			{
				int n = step_ind + MD_LEN - STEP_CNT;
				u64* x_last = x_last0 + n * 2 * (4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE);
				u64* y_last = y_last0 + n * 2 * (4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE);
				SAVE_VAL_256(x_last, x, group);
				SAVE_VAL_256(y_last, y, group);
			}
        }
		jlist += PNT_GROUP_CNT * BLOCK_SIZE / 8;
    } 

	Kparams.L1S2[BLOCK_X * BLOCK_SIZE + THREAD_X] = L1S2;
	//copy kangs from L2 to global
	kang_ind = PNT_GROUP_CNT * (THREAD_X + BLOCK_X * BLOCK_SIZE);
	for (u32 group = 0; group < PNT_GROUP_CNT; group++)
	{
		LOAD_VAL_256(tmp, L2x, group);
		Kparams.Kangs[(kang_ind + group) * 12 + 0] = tmp[0];
		Kparams.Kangs[(kang_ind + group) * 12 + 1] = tmp[1];
		Kparams.Kangs[(kang_ind + group) * 12 + 2] = tmp[2];
		Kparams.Kangs[(kang_ind + group) * 12 + 3] = tmp[3];
		LOAD_VAL_256(tmp, L2y, group);
		Kparams.Kangs[(kang_ind + group) * 12 + 4] = tmp[0];
		Kparams.Kangs[(kang_ind + group) * 12 + 5] = tmp[1];
		Kparams.Kangs[(kang_ind + group) * 12 + 6] = tmp[2];
		Kparams.Kangs[(kang_ind + group) * 12 + 7] = tmp[3];
	}
} 

#else

#define LOAD_VAL_256_m(dst,p,i) { *((int4*)&(dst)[0]) = *((int4*)&(p)[4 * (i)]); *((int4*)&(dst)[2]) = *((int4*)&(p)[4 * (i) + 2]); }
#define SAVE_VAL_256_m(p,src,i) { *((int4*)&(p)[4 * (i)]) = *((int4*)&(src)[0]); *((int4*)&(p)[4 * (i) + 2]) = *((int4*)&(src)[2]); }


//this kernel performs main jumps for old cards
//not good but works
extern "C" __launch_bounds__(BLOCK_SIZE, 1)
__global__ void KernelA(const TKparams Kparams)
{
	__align__(16) u64 Lx[4 * PNT_GROUP_CNT];
	__align__(16) u64 Ly[4 * PNT_GROUP_CNT];
	__align__(16) u64 Ls[4 * PNT_GROUP_CNT / 2]; //we store only half so need only half mem

	//list of distances of performed jumps for KernelB
	int4* jlist = (int4*)(Kparams.JumpsList + (u64)BLOCK_X * STEP_CNT * PNT_GROUP_CNT * BLOCK_SIZE / 4);
	jlist += (THREAD_X / 32) * 32 * PNT_GROUP_CNT / 8;
	//list of last visited points for KernelC
	u64* x_last0 = Kparams.LastPnts + 2 * THREAD_X + 4 * BLOCK_SIZE * BLOCK_X;
	u64* y_last0 = x_last0 + 4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE;

	u64* jmp1_table = LDS; //32KB
	u16* lds_jlist = (u16*)&LDS[8 * JMP_CNT]; //8KB, must be aligned 16bytes

	int i = THREAD_X;
	while (i < JMP_CNT)
	{
		*(int4*)&jmp1_table[8 * i + 0] = *(int4*)&Kparams.Jumps1[12 * i + 0];
		*(int4*)&jmp1_table[8 * i + 2] = *(int4*)&Kparams.Jumps1[12 * i + 2];
		*(int4*)&jmp1_table[8 * i + 4] = *(int4*)&Kparams.Jumps1[12 * i + 4];
		*(int4*)&jmp1_table[8 * i + 6] = *(int4*)&Kparams.Jumps1[12 * i + 6];
		i += BLOCK_SIZE;
	}

	__syncthreads();

	__align__(16) u64 inverse[5];
	__align__(16) u64 x[4], y[4], tmp[4], tmp2[4];
	u64 dp_mask64;
	if (Kparams.DP <= 0)
		dp_mask64 = 0ULL;
	else if (Kparams.DP >= 64)
		dp_mask64 = ~0ULL;
	else {
		uint32_t shift = (uint32_t)(64ULL - (uint64_t)Kparams.DP);
		if (shift > 63) shift = 63;
		dp_mask64 = ~((1ULL << shift) - 1ULL);
	}
	u16 jmp_ind;

	//copy kangs from global to local
	u32 kang_ind = PNT_GROUP_CNT * (THREAD_X + BLOCK_X * BLOCK_SIZE);
	for (u32 group = 0; group < PNT_GROUP_CNT; group++)
	{
		tmp[0] = Kparams.Kangs[(kang_ind + group) * 12 + 0];
		tmp[1] = Kparams.Kangs[(kang_ind + group) * 12 + 1];
		tmp[2] = Kparams.Kangs[(kang_ind + group) * 12 + 2];
		tmp[3] = Kparams.Kangs[(kang_ind + group) * 12 + 3];
		SAVE_VAL_256_m(Lx, tmp, group);
		tmp[0] = Kparams.Kangs[(kang_ind + group) * 12 + 4];
		tmp[1] = Kparams.Kangs[(kang_ind + group) * 12 + 5];
		tmp[2] = Kparams.Kangs[(kang_ind + group) * 12 + 6];
		tmp[3] = Kparams.Kangs[(kang_ind + group) * 12 + 7];
		SAVE_VAL_256_m(Ly, tmp, group);
	}

	u64 L1S2 = ((u64*)Kparams.L1S2)[BLOCK_X * BLOCK_SIZE + THREAD_X];
	u64* jmp_table;
	__align__(16) u64 jmp_x[4];
	__align__(16) u64 jmp_y[4];

	//preparations (first calc for inv)
	for (int group = 0; group < PNT_GROUP_CNT; group++)
	{
		LOAD_VAL_256_m(x, Lx, group);
		jmp_ind = x[0] % JMP_CNT;
		jmp_table = ((L1S2 >> group) & 1) ? jmp2_table : jmp1_table;
		Copy_int4_x2(jmp_x, jmp_table + 8 * jmp_ind);
		SubModP(tmp, x, jmp_x);
		if (group == 0)
		{
			Copy_u64_x4(inverse, tmp);
			SAVE_VAL_256_m(Ls, tmp, 0);
		}
		else
		{
			MulModP(inverse, inverse, tmp);
			if ((group & 1) == 0)
				SAVE_VAL_256_m(Ls, inverse, group / 2);
		}
	}

	//main loop
	int g_beg = PNT_GROUP_CNT - 1; //start val
	int g_end = -1; //first val after range
	int g_inc = -1;
	int s_mask = 1;
	int jlast_add = 0;
	__align__(16) u64 t_cache[4], x0_cache[4], jmpx_cached[4];
	t_cache[0] = t_cache[1] = t_cache[2] = t_cache[3] = 0;
	x0_cache[0] = x0_cache[1] = x0_cache[2] = x0_cache[3] = 0;

	for (int step_ind = 0; step_ind < STEP_CNT; step_ind++)
	{
		__align__(16) u64 next_inv[4];

		InvModP((u32*)inverse);

		int group = g_beg;
		bool cached = false;
		while (group != g_end)
		{
			__align__(16) u64 dx[4], x0[4], y0[4], dx0[4];
			if (cached)
			{
				Copy_u64_x4(x0, x0_cache);
			}
			else
			{
				LOAD_VAL_256_m(x0, Lx, group);
			}
			LOAD_VAL_256_m(y0, Ly, group);

			jmp_ind = x0[0] % JMP_CNT;
			jmp_table = ((L1S2 >> group) & 1) ? jmp2_table : jmp1_table;
			if (cached)
			{
				Copy_u64_x4(jmp_x, jmpx_cached); 
			}
			else
			{
				Copy_int4_x2(jmp_x, jmp_table + 8 * jmp_ind);
			}
			Copy_int4_x2(jmp_y, jmp_table + 8 * jmp_ind + 4);
			u32 inv_flag = (u32)y0[0] & 1;
			if (inv_flag)
			{
				jmp_ind |= INV_FLAG;
				NegModP(jmp_y);
			}

			if (group == g_end - g_inc)
			{
				Copy_u64_x4(dx0, inverse);
			}
			else
			{
				if ((group & 1) == s_mask) //simple case
				{
					if (cached)
					{
						Copy_u64_x4(tmp, t_cache);
						cached = false;
					}
					else
					{
						LOAD_VAL_256_m(tmp, Ls, (group + g_inc) / 2);
					}
				}
				else //no s(-1), need to calc it
				{
					LOAD_VAL_256_m(t_cache, Ls, (group + g_inc + g_inc) / 2);
					cached = true;				
					LOAD_VAL_256_m(x0_cache, Lx, group + g_inc);
					u32 jmp_tmp = x0_cache[0] % JMP_CNT;
					__align__(16) u64 dx2[4];
					u64* jmp_table_tmp = ((L1S2 >> (group + g_inc)) & 1) ? jmp2_table : jmp1_table;
					Copy_int4_x2(jmpx_cached, jmp_table_tmp + 8 * jmp_tmp);
					SubModP(dx2, x0_cache, jmpx_cached);
					MulModP(tmp, t_cache, dx2); //t = s(-1)
				}

				SubModP(dx, x0, jmp_x);
				MulModP(dx0, tmp, inverse);
				MulModP(inverse, inverse, dx);
			}

			SubModP(tmp2, y0, jmp_y);
			MulModP(tmp, tmp2, dx0);
			SqrModP(tmp2, tmp);

			SubModP(x, tmp2, jmp_x);
			SubModP(x, x, x0);
			SAVE_VAL_256_m(Lx, x, group);

			SubModP(y, x0, x);
			MulModP(y, y, tmp);
			SubModP(y, y, y0);
			SAVE_VAL_256_m(Ly, y, group);

			if (((L1S2 >> group) & 1) == 0) //normal mode, check L1S2 loop
			{
				u32 jmp_next = x[0] % JMP_CNT;
				jmp_next |= ((u32)y[0] & 1) ? 0 : INV_FLAG; //inverted
				L1S2 |= (jmp_ind == jmp_next) ? (1ull << group) : 0; //loop L1S2 detected
			}
			else
			{
				L1S2 &= ~(1ull << group);
				jmp_ind |= JMP2_FLAG;
			}

			if ((x[3] & dp_mask64) == 0)
			{
				u32 kang_ind = (THREAD_X + BLOCK_X * BLOCK_SIZE) * PNT_GROUP_CNT + group;
				u32 ind = atomicAdd(Kparams.DPTable + kang_ind, 1);
				ind = min(ind, DPTABLE_MAX_CNT - 1);
				int4* dst = (int4*)(Kparams.DPTable + Kparams.KangCnt + (kang_ind * DPTABLE_MAX_CNT + ind) * 8);
				dst[0] = ((int4*)x)[0];
				dst[1] = ((int4*)x)[1];
				jmp_ind |= DP_FLAG;
			}

			lds_jlist[8 * THREAD_X + (group % 8)] = jmp_ind;
			if (((group + jlast_add) % 8) == 0)
				st_cs_v4_b32(&jlist[(group / 8) * 32 + (THREAD_X % 32)], *(int4*)&lds_jlist[8 * THREAD_X]); //skip L2 cache

#ifdef COLLIDER_WALKSTEP_TRACE
			// v1.5.5 walk-step trace capture (task #7), OLD_GPU KernelA
			// (the variant compiled for __CUDA_ARCH__ < 890, i.e. pre-RTX-40xx).
			// Mirrors the new-GPU capture: kangaroo (0,0,0) only.
			if (g_dbg_on && BLOCK_X == 0 && THREAD_X == 0 && group == 0)
			{
				u32 i = g_dbg_n;
				if (i < DBG_CAP)
				{
					if (i == 0)
					{
#pragma unroll
						for (int q = 0; q < 4; q++) { g_dbg_birth_x[q] = x0[q]; g_dbg_birth_y[q] = y0[q]; }
					}
#pragma unroll
					for (int q = 0; q < 4; q++) { g_dbg_x[i * 4 + q] = x[q]; g_dbg_y[i * 4 + q] = y[q]; }
					g_dbg_jmp[i] = jmp_ind;
					g_dbg_n = i + 1;
				}
			}
#endif // COLLIDER_WALKSTEP_TRACE

			if (step_ind + MD_LEN >= STEP_CNT) //store last kangs to be able to find loop exit point
			{
				int n = step_ind + MD_LEN - STEP_CNT;
				u64* x_last = x_last0 + n * 2 * (4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE);
				u64* y_last = y_last0 + n * 2 * (4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE);
				SAVE_VAL_256(x_last, x, group);
				SAVE_VAL_256(y_last, y, group);
			}
		
			//preps to calc next inv
			jmp_ind = x[0] % JMP_CNT;
			jmp_table = ((L1S2 >> group) & 1) ? jmp2_table : jmp1_table;
			Copy_int4_x2(jmp_x, jmp_table + 8 * jmp_ind);
			SubModP(dx, x, jmp_x);
			if (group == g_beg)
			{
				Copy_u64_x4(next_inv, dx);
				SAVE_VAL_256_m(Ls, dx, g_beg / 2);
			}
			else
			{
				MulModP(next_inv, next_inv, dx);
				if ((group & 1) == s_mask)
				{
					SAVE_VAL_256_m(Ls, next_inv, group / 2);
				}
			}

			group += g_inc;
		} //group
		jlist += PNT_GROUP_CNT * BLOCK_SIZE / 8;
		Copy_u64_x4(inverse, next_inv);
		if (g_inc < 0) //invert direction
		{
			g_beg = 0;
			g_end = PNT_GROUP_CNT;
			g_inc = 1;
			s_mask = 0;
			jlast_add = 1;
		}
		else
		{
			g_beg = PNT_GROUP_CNT - 1;
			g_end = -1;
			g_inc = -1;
			s_mask = 1;
			jlast_add = 0;
		}
	}

	((u64*)Kparams.L1S2)[BLOCK_X * BLOCK_SIZE + THREAD_X] = L1S2;
	//copy kangs from local to global
	kang_ind = PNT_GROUP_CNT * (THREAD_X + BLOCK_X * BLOCK_SIZE);
	for (u32 group = 0; group < PNT_GROUP_CNT; group++)
	{
		LOAD_VAL_256_m(tmp, Lx, group);
		Kparams.Kangs[(kang_ind + group) * 12 + 0] = tmp[0];
		Kparams.Kangs[(kang_ind + group) * 12 + 1] = tmp[1];
		Kparams.Kangs[(kang_ind + group) * 12 + 2] = tmp[2];
		Kparams.Kangs[(kang_ind + group) * 12 + 3] = tmp[3];
		LOAD_VAL_256_m(tmp, Ly, group);
		Kparams.Kangs[(kang_ind + group) * 12 + 4] = tmp[0];
		Kparams.Kangs[(kang_ind + group) * 12 + 5] = tmp[1];
		Kparams.Kangs[(kang_ind + group) * 12 + 6] = tmp[2];
		Kparams.Kangs[(kang_ind + group) * 12 + 7] = tmp[3];
	}
}

#endif

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ void BuildDP(const TKparams& Kparams, int kang_ind, u64* d)
{
	int ind = atomicAdd(Kparams.DPTable + kang_ind, 0x10000);
	ind >>= 16;
	if (ind >= DPTABLE_MAX_CNT)
		return;
	int4* src = (int4*)(Kparams.DPTable + Kparams.KangCnt + (kang_ind * DPTABLE_MAX_CNT + ind) * 8);
	int4 rx0 = src[0];
	int4 rx1 = src[1];
	u32 pos = atomicAdd(Kparams.DPs_out, 1);
	pos = min(pos, MAX_DP_CNT - 1);
	u32* DPs = Kparams.DPs_out + 4 + pos * GPU_DP_SIZE / 4;
	*(int4*)&DPs[0] = rx0;
	*(int4*)&DPs[4] = rx1;
	*(int4*)&DPs[8] = ((int4*)d)[0];
	*(u64*)&DPs[12] = d[2];
	// theCollider v1.5: in pool TAME_ONLY / WILD_ONLY modes the herd is
	// homogeneous, so the type byte is fixed regardless of kang_ind.
	// In BOTH (upstream) the type is derived from the 1/3 partitioning
	// of kang_ind across [0, KangCnt). The wrapper / upstream
	// CheckNewPoints both read this byte at DPs[14] (= byte offset 56
	// in the 64-byte GPU DP record).
	u32 dp_type;
	if (Kparams.Mode == KANG_MODE_TAME_ONLY)
		dp_type = TAME;
	else if (Kparams.Mode == KANG_MODE_WILD_ONLY)
		dp_type = WILD1;
	else
		dp_type = 3 * kang_ind / Kparams.KangCnt; //kang type (BOTH)
	DPs[14] = dp_type;
#ifdef COLLIDER_CHECKPOINT_CAPTURE
	// Tag the DP with the producing kangaroo's global index in the otherwise
	// unused 64-byte slot (u32 index 15 == byte offset 60). The wrapper uses
	// this to pair a DP with the per-kangaroo checkpoint chain captured for
	// that index, so the client can emit a real DP_BATCH_V3 commitment. This
	// is gated to the capture build and never read by the v2 path; it does
	// NOT touch x, d, type, or any walk state, so the walk math and the v2 DP
	// contract are byte-for-byte unchanged.
	DPs[15] = kang_ind;
#endif
}

__device__ __forceinline__ bool ProcessJumpDistance(u32 step_ind, u32 d_cur, u64* d, u32 kang_ind, u64* jmp1_d, u64* jmp2_d, const TKparams& Kparams, u64* table, u32* cur_ind, u8 iter)
{
#ifdef COLLIDER_CHECKPOINT_CAPTURE
	// Capture a checkpoint BEFORE this jump mutates `d`. At this point `d`
	// holds the distance after exactly g_ckpt_jumps[slot] jumps and `d_cur`
	// carries the JMP2_FLAG (== loop-state bit) for the jump we are ABOUT to
	// apply, i.e. the l1s2 bit ENTERING the next step. That pair
	// (d_after_m_jumps, l1s2_entering_step_m+1) is exactly what
	// checkpoint_walk::generate_checkpoints records at checkpoint m, so the
	// captured chain is byte-for-byte what the server replays. The birth
	// checkpoint (m == 0) is captured here on the very first jump.
	if (g_ckpt_on)
	{
		u32 base = g_ckpt_base;
		if (kang_ind >= base && kang_ind < base + CHK_MAX_KANG)
		{
			u32 slot = kang_ind - base;
			u64 jumps = g_ckpt_jumps[slot];
			if ((jumps % CHK_INTERVAL) == 0)
				chk_record(kang_ind, d, (d_cur & JMP2_FLAG) ? 1u : 0u);
			g_ckpt_jumps[slot] = jumps + 1;
		}
	}
#endif // COLLIDER_CHECKPOINT_CAPTURE

	u64* jmp_d = (d_cur & JMP2_FLAG) ? jmp2_d : jmp1_d;

	__align__(16) u64 jmp[3];
	((int4*)(jmp))[0] = ((int4*)(jmp_d + 4 * (d_cur & JMP_MASK)))[0];
	jmp[2] = *(jmp_d + 4 * (d_cur & JMP_MASK) + 2);

	if (d_cur & INV_FLAG)
		Sub192from192(d, jmp)
	else
		Add192to192(d, jmp);

	//check in table
	int found_ind = iter + MD_LEN - 4;
	while (1)
	{
		if (table[found_ind % MD_LEN] == d[0])
			break;
		found_ind -= 2;
		if (table[found_ind % MD_LEN] == d[0])
			break;
		found_ind -= 2;
		if (table[found_ind % MD_LEN] == d[0])
			break;
		found_ind = iter;
		if (table[found_ind] == d[0])
			break;
		found_ind = -1;
		break;
	}
	table[iter] = d[0];
	*cur_ind = (iter + 1) % MD_LEN;

	if (found_ind < 0)
	{		
		if (d_cur & DP_FLAG)
			BuildDP(Kparams, kang_ind, d);
		return false;
	}

	u32 LoopSize = (iter + MD_LEN - found_ind) % MD_LEN;
	if (!LoopSize)
		LoopSize = MD_LEN;
	atomicAdd(Kparams.dbg_buf + LoopSize, 1); //dbg

#ifdef COLLIDER_CHECKPOINT_CAPTURE
	// A loop was detected for this kangaroo: KernelC will apply a jmp3
	// loop-escape jump, which mutates the distance OFF the jmp1/jmp2 path the
	// canonical walk (and the server replay) model. Flag the captured
	// kangaroo so the host can distinguish a true capture mismatch from an
	// (expected) loop-escape segment. This does NOT change the walk; it only
	// counts the event for the gated validation build.
	if (g_ckpt_on)
	{
		u32 cbase = g_ckpt_base;
		if (kang_ind >= cbase && kang_ind < cbase + CHK_MAX_KANG)
			atomicAdd(&g_ckpt_loopesc[kang_ind - cbase], 1u);
	}
#endif // COLLIDER_CHECKPOINT_CAPTURE

	//calc index in LastPnts
	u32 ind_LastPnts = MD_LEN - 1 - ((STEP_CNT - 1 - step_ind) % LoopSize);
	u32 ind = atomicAdd(Kparams.LoopedKangs, 1);
	Kparams.LoopedKangs[2 + ind] = kang_ind | (ind_LastPnts << 28);
	return true;
}

#define DO_ITER(iter) {\
	u32 cur_dAB = jlist[THREAD_X]; \
	u16 cur_dA = cur_dAB & 0xFFFF; \
	u16 cur_dB = cur_dAB >> 16; \
	if (!LoopedA) \
		LoopedA = ProcessJumpDistance(step_ind, cur_dA, dA, kang_ind, jmp1_d, jmp2_d, Kparams, RegsA, &cur_indA, iter); \
	if (!LoopedB) \
		LoopedB = ProcessJumpDistance(step_ind, cur_dB, dB, kang_ind + 1, jmp1_d, jmp2_d, Kparams, RegsB, &cur_indB, iter); \
	jlist += BLOCK_SIZE * PNT_GROUP_CNT / 2; \
	step_ind++; \
}

//this kernel counts distances and detects loops Size>2
//Loops Level1 statistics for JMP_CNT=512: L1S2 = 1/1024 (so one loop every 1024 jumps), L1S4 = L1S2/1024, L1S6 = L1S4/256, L1S8 = L1S6/158, L1S10 = L1S8/82. L1S12 = L1S10/50. 
// For RTX4090 at 8HG/s for 24 hours and JMP_CNT=512: jumps = 691200bln, L1S2 = 682bln, L1S4 = 666mln, L1S6 = 2.6mln, L1S8 = 16.5k, L1S10 = 201. L1S12 = 4.
// I don't see any reasons to catch L1S12 because we have 786432 kangs, if we lose 4 kangs every day, we lose 1460 kangs a year which is about 0.19%.
// This degradation depends only on speed of a single kangaroo, so it's about the same for all 40xx GPUs (50xx GPUs will have +20% clock speed may be).
// Since we lose kangs gradually, for a year we lose 0.19/2 = 0.1% of speed, so you should catch L1S12 only if you are going to solve same point for decades.
// Or you can check all kangs for L1S12 on CPU once a day and restart looped kangs.
// Level2 loops are very rare and they have even size too so they will be handled by the same code. We don't know what loop level we catch so we use JmpTable3 for escaping.
extern "C" __launch_bounds__(BLOCK_SIZE, 1)
__global__ void KernelB(const TKparams Kparams)
{
	u64* jmp1_d = LDS; //16KB, 192bit jumps
	u64* jmp2_d = LDS + 4 * JMP_CNT; //16KB, 192bit jumps

	int i = THREAD_X;
	while (i < JMP_CNT)
	{
		//192bits but we need align 128 so use 256
		jmp1_d[4 * i + 0] = Kparams.Jumps1[12 * i + 8];
		jmp1_d[4 * i + 1] = Kparams.Jumps1[12 * i + 9];
		jmp1_d[4 * i + 2] = Kparams.Jumps1[12 * i + 10];
		jmp2_d[4 * i + 0] = Kparams.Jumps2[12 * i + 8];
		jmp2_d[4 * i + 1] = Kparams.Jumps2[12 * i + 9];
		jmp2_d[4 * i + 2] = Kparams.Jumps2[12 * i + 10];
		i += BLOCK_SIZE;
	}

	u32* jlist0 = (u32*)(Kparams.JumpsList + (u64)BLOCK_X * STEP_CNT * PNT_GROUP_CNT * BLOCK_SIZE / 4);

	__syncthreads();

	u64 RegsA[MD_LEN], RegsB[MD_LEN];

	//we process two kangs at once
	for (u32 gr_ind2 = 0; gr_ind2 < PNT_GROUP_CNT/2; gr_ind2++)
	{	
		#pragma unroll
		for (int i = 0; i < MD_LEN; i++)
		{
			RegsA[i] = Kparams.LoopTable[MD_LEN * BLOCK_SIZE * PNT_GROUP_CNT * BLOCK_X + 2 * MD_LEN * BLOCK_SIZE * gr_ind2 + i * BLOCK_SIZE + BLOCK_X];
			RegsB[i] = Kparams.LoopTable[MD_LEN * BLOCK_SIZE * PNT_GROUP_CNT * BLOCK_X + 2 * MD_LEN * BLOCK_SIZE * gr_ind2 + (i + MD_LEN) * BLOCK_SIZE + BLOCK_X];
		}
		u32 cur_indA = 0;
		u32 cur_indB = 0;

		u32* jlist = jlist0 + gr_ind2 * BLOCK_SIZE;

		//calc original kang_ind
		u32 tind = (THREAD_X + gr_ind2 * BLOCK_SIZE); //0..3071
		u32 warp_ind = tind / (32 * PNT_GROUP_CNT / 2); // 0..7	
		u32 thr_ind = (tind / 4) % 32; //index in warp 0..31
		u32 g8_ind = (tind % (32 * PNT_GROUP_CNT / 2)) / 128; // 0..2
		u32 gr_ind = 2 * (tind % 4); // 0, 2, 4, 6

		u32 kang_ind = (BLOCK_X * BLOCK_SIZE) * PNT_GROUP_CNT;
		kang_ind += (32 * warp_ind + thr_ind) * PNT_GROUP_CNT + 8 * g8_ind + gr_ind;

		__align__(8) u64 dA[3], dB[3];
		dA[0] = Kparams.Kangs[kang_ind * 12 + 8];
		dA[1] = Kparams.Kangs[kang_ind * 12 + 9];
		dA[2] = Kparams.Kangs[kang_ind * 12 + 10];
		dB[0] = Kparams.Kangs[(kang_ind + 1) * 12 + 8];
		dB[1] = Kparams.Kangs[(kang_ind + 1) * 12 + 9];
		dB[2] = Kparams.Kangs[(kang_ind + 1) * 12 + 10];

		bool LoopedA = false;
		bool LoopedB = false;
		u32 step_ind = 0;
		while (step_ind < STEP_CNT)
		{
			DO_ITER(0);
			DO_ITER(1);
			DO_ITER(2);
			DO_ITER(3);
			DO_ITER(4);
			DO_ITER(5);
			DO_ITER(6);
			DO_ITER(7);
			DO_ITER(8);
			DO_ITER(9);
		}

		Kparams.Kangs[kang_ind * 12 + 8] = dA[0];
		Kparams.Kangs[kang_ind * 12 + 9] = dA[1];
		Kparams.Kangs[kang_ind * 12 + 10] = dA[2];
		Kparams.Kangs[(kang_ind + 1) * 12 + 8] = dB[0];
		Kparams.Kangs[(kang_ind + 1) * 12 + 9] = dB[1];
		Kparams.Kangs[(kang_ind + 1) * 12 + 10] = dB[2];

		//store so cur_ind is 0 at next loading
		#pragma unroll
		for (int i = 0; i < MD_LEN; i++)
		{
			int ind = (i + MD_LEN - cur_indA) % MD_LEN;
			Kparams.LoopTable[MD_LEN * BLOCK_SIZE * PNT_GROUP_CNT * BLOCK_X + 2 * MD_LEN * BLOCK_SIZE * gr_ind2 + ind * BLOCK_SIZE + BLOCK_X] = RegsA[i];
			ind = (i + MD_LEN - cur_indB) % MD_LEN;
			Kparams.LoopTable[MD_LEN * BLOCK_SIZE * PNT_GROUP_CNT * BLOCK_X + 2 * MD_LEN * BLOCK_SIZE * gr_ind2 + (ind + MD_LEN) * BLOCK_SIZE + BLOCK_X] = RegsB[i];
		}
	}
}

//this kernel performes single jump3 for looped kangs
extern "C" __launch_bounds__(BLOCK_SIZE, 1)
__global__ void KernelC(const TKparams Kparams)
{
	u64* jmp3_table = LDS; //48KB

	int i = THREAD_X;
	while (i < JMP_CNT)
	{
		*(int4*)&jmp3_table[12 * i + 0] = *(int4*)&Kparams.Jumps3[12 * i + 0];
		*(int4*)&jmp3_table[12 * i + 2] = *(int4*)&Kparams.Jumps3[12 * i + 2];
		*(int4*)&jmp3_table[12 * i + 4] = *(int4*)&Kparams.Jumps3[12 * i + 4];
		*(int4*)&jmp3_table[12 * i + 6] = *(int4*)&Kparams.Jumps3[12 * i + 6];
		*(int4*)&jmp3_table[12 * i + 8] = *(int4*)&Kparams.Jumps3[12 * i + 8];
		*(int4*)&jmp3_table[12 * i + 10] = *(int4*)&Kparams.Jumps3[12 * i + 10];
		i += BLOCK_SIZE;
	}

	__syncthreads();

	while (1)
	{
		u32 ind = atomicAdd(Kparams.LoopedKangs + 1, 1);
		if (ind >= Kparams.LoopedKangs[0])
			break;
		u32 kang_ind = Kparams.LoopedKangs[2 + ind] & 0x0FFFFFFF;
		u32 last_ind = Kparams.LoopedKangs[2 + ind] >> 28;

		__align__(16) u64 x0[4], x[4];
		__align__(16) u64 y0[4], y[4];
		__align__(16) u64 jmp_x[4];
		__align__(16) u64 jmp_y[4];
		__align__(16) u64 inverse[5];
		u64 tmp[4], tmp2[4];

		u64* x_last0 = Kparams.LastPnts;
		u64* y_last0 = x_last0 + 4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE;

		u32 block_ind = kang_ind / (BLOCK_SIZE * PNT_GROUP_CNT);
		u32 thr_ind = (kang_ind - block_ind * (BLOCK_SIZE * PNT_GROUP_CNT)) / PNT_GROUP_CNT;
		u32 gr_ind = (kang_ind - block_ind * (BLOCK_SIZE * PNT_GROUP_CNT) - thr_ind * PNT_GROUP_CNT);

		y_last0 += 2 * thr_ind + 4 * BLOCK_SIZE * block_ind;
		x_last0 += 2 * thr_ind + 4 * BLOCK_SIZE * block_ind;
		u64* x_last = x_last0 + last_ind * 2 * (4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE);
		u64* y_last = y_last0 + last_ind * 2 * (4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE);
		LOAD_VAL_256(x0, x_last, gr_ind);
		LOAD_VAL_256(y0, y_last, gr_ind);

		u32 jmp_ind = x0[0] % JMP_CNT;
		Copy_int4_x2(jmp_x, jmp3_table + 12 * jmp_ind);
		Copy_int4_x2(jmp_y, jmp3_table + 12 * jmp_ind + 4);
		SubModP(inverse, x0, jmp_x);
		InvModP((u32*)inverse);

		u32 inv_flag = y0[0] & 1;
		if (inv_flag)
			NegModP(jmp_y);

		SubModP(tmp, y0, jmp_y);
		MulModP(tmp2, tmp, inverse);
		SqrModP(tmp, tmp2);

		SubModP(x, tmp, jmp_x);
		SubModP(x, x, x0);
		SubModP(y, x0, x);
		MulModP(y, y, tmp2);
		SubModP(y, y, y0);

		//save kang
		Kparams.Kangs[kang_ind * 12 + 0] = x[0];
		Kparams.Kangs[kang_ind * 12 + 1] = x[1];
		Kparams.Kangs[kang_ind * 12 + 2] = x[2];
		Kparams.Kangs[kang_ind * 12 + 3] = x[3];
		Kparams.Kangs[kang_ind * 12 + 4] = y[0];
		Kparams.Kangs[kang_ind * 12 + 5] = y[1];
		Kparams.Kangs[kang_ind * 12 + 6] = y[2];
		Kparams.Kangs[kang_ind * 12 + 7] = y[3];

		//add distance
		u64 d[3];
		d[0] = Kparams.Kangs[kang_ind * 12 + 8];
		d[1] = Kparams.Kangs[kang_ind * 12 + 9];
		d[2] = Kparams.Kangs[kang_ind * 12 + 10];
		if (inv_flag)
			Sub192from192(d, jmp3_table + 12 * jmp_ind + 8)
		else
			Add192to192(d, jmp3_table + 12 * jmp_ind + 8);
		Kparams.Kangs[kang_ind * 12 + 8] = d[0];
		Kparams.Kangs[kang_ind * 12 + 9] = d[1];
		Kparams.Kangs[kang_ind * 12 + 10] = d[2];

#ifndef OLD_GPU
		atomicAnd(&Kparams.L1S2[block_ind * BLOCK_SIZE + thr_ind], ~(1u << gr_ind));
#else
		atomicAnd(&((u64*)Kparams.L1S2)[block_ind * BLOCK_SIZE + thr_ind], ~(1ull << gr_ind));
#endif
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define GX_0	0x59F2815B16F81798ull
#define GX_1	0x029BFCDB2DCE28D9ull
#define GX_2	0x55A06295CE870B07ull
#define GX_3	0x79BE667EF9DCBBACull
#define GY_0	0x9C47D08FFB10D4B8ull
#define GY_1	0xFD17B448A6855419ull
#define GY_2	0x5DA4FBFC0E1108A8ull
#define GY_3	0x483ADA7726A3C465ull

__device__ __forceinline__ void AddPoints(u64* res_x, u64* res_y, u64* pnt1x, u64* pnt1y, u64* pnt2x, u64* pnt2y)
{
	__align__(16) u64 tmp[4], tmp2[4], lambda[4], lambda2[4];
	__align__(16) u64 inverse[5];
	SubModP(inverse, pnt2x, pnt1x);
	InvModP((u32*)inverse);
	SubModP(tmp, pnt2y, pnt1y);
	MulModP(lambda, tmp, inverse);
	MulModP(lambda2, lambda, lambda);
	SubModP(tmp, lambda2, pnt1x);
	SubModP(res_x, tmp, pnt2x);
	SubModP(tmp, pnt2x, res_x);
	MulModP(tmp2, tmp, lambda);
	SubModP(res_y, tmp2, pnt2y);
}

__device__ __forceinline__ void DoublePoint(u64* res_x, u64* res_y, u64* pntx, u64* pnty)
{
	__align__(16) u64 tmp[4], tmp2[4], lambda[4], lambda2[4];
	__align__(16) u64 inverse[5];
	AddModP(inverse, pnty, pnty);
	InvModP((u32*)inverse);
	MulModP(tmp2, pntx, pntx);
	AddModP(tmp, tmp2, tmp2);
	AddModP(tmp, tmp, tmp2);
	MulModP(lambda, tmp, inverse);
	MulModP(lambda2, lambda, lambda);
	SubModP(tmp, lambda2, pntx);
	SubModP(res_x, tmp, pntx);
	SubModP(tmp, pntx, res_x);
	MulModP(tmp2, tmp, lambda);
	SubModP(res_y, tmp2, pnty);
}

//this kernel calculates start points of kangs
extern "C" __launch_bounds__(BLOCK_SIZE, 1)
__global__ void KernelGen(const TKparams Kparams)
{
	for (u32 group = 0; group < PNT_GROUP_CNT; group++)
	{
		__align__(16) u64 x0[4], y0[4], d[3];
		__align__(16) u64 x[4], y[4];
		__align__(16) u64 tx[4], ty[4];
		__align__(16) u64 t2x[4], t2y[4];

		u32 kang_ind = PNT_GROUP_CNT * (THREAD_X + BLOCK_X * BLOCK_SIZE) + group;
		x0[0] = Kparams.Kangs[kang_ind * 12 + 0];
		x0[1] = Kparams.Kangs[kang_ind * 12 + 1];
		x0[2] = Kparams.Kangs[kang_ind * 12 + 2];
		x0[3] = Kparams.Kangs[kang_ind * 12 + 3];
		y0[0] = Kparams.Kangs[kang_ind * 12 + 4];
		y0[1] = Kparams.Kangs[kang_ind * 12 + 5];
		y0[2] = Kparams.Kangs[kang_ind * 12 + 6];
		y0[3] = Kparams.Kangs[kang_ind * 12 + 7];
		d[0] = Kparams.Kangs[kang_ind * 12 + 8];
		d[1] = Kparams.Kangs[kang_ind * 12 + 9];
		d[2] = Kparams.Kangs[kang_ind * 12 + 10];
		
		tx[0] = GX_0; tx[1] = GX_1; tx[2] = GX_2; tx[3] = GX_3;
		ty[0] = GY_0; ty[1] = GY_1; ty[2] = GY_2; ty[3] = GY_3;

		bool first = true;
		int n = 2;
		while ((n >= 0) && !d[n]) 
			n--;
		if (n < 0)
			continue; //error
		int index = __clzll(d[n]);
		for (int i = 0; i <= 64 * n + (63 - index); i++)
		{
			u8 v = (d[i / 64] >> (i % 64)) & 1;
			if (v)
			{
				if (first)
				{
					first = false;
					Copy_u64_x4(x, tx);
					Copy_u64_x4(y, ty);
				}
				else
				{
					AddPoints(t2x, t2y, x, y, tx, ty);
					Copy_u64_x4(x, t2x);
					Copy_u64_x4(y, t2y);
				}
			}
			DoublePoint(t2x, t2y, tx, ty);
			Copy_u64_x4(tx, t2x);
			Copy_u64_x4(ty, t2y);
		}

		// theCollider v1.5: decide whether this kangaroo's starting
		// point gets the wild offset (PntA or PntB, supplied through
		// x0/y0) added to its tame-only `d*G` result.
		//   BOTH (upstream): only the upper 2/3 of indices (the wild
		//     kangaroos) take the offset. Tames stay at d*G.
		//   TAME_ONLY: never add the offset. Every kangaroo stays at
		//     d*G (a tame). x0/y0 are zero per the host seeding in
		//     GpuKang.cpp:Start.
		//   WILD_ONLY: every kangaroo takes the offset (PntA for the
		//     whole herd). They are all wild1.
		// IsGenMode (tames-generation mode) suppresses the wild branch
		// in BOTH the way it did upstream; it is mutually exclusive
		// with the pool TAME_ONLY / WILD_ONLY modes (pool mode never
		// runs tame-file generation).
		bool add_wild_offset = false;
		if (!Kparams.IsGenMode)
		{
			if (Kparams.Mode == KANG_MODE_TAME_ONLY)
				add_wild_offset = false;
			else if (Kparams.Mode == KANG_MODE_WILD_ONLY)
				add_wild_offset = true;
			else // KANG_MODE_BOTH
				add_wild_offset = (kang_ind >= Kparams.KangCnt / 3);
		}
		if (add_wild_offset)
		{
			AddPoints(t2x, t2y, x, y, x0, y0);
			Copy_u64_x4(x, t2x);
			Copy_u64_x4(y, t2y);
		}

		Kparams.Kangs[kang_ind * 12 + 0] = x[0];
		Kparams.Kangs[kang_ind * 12 + 1] = x[1];
		Kparams.Kangs[kang_ind * 12 + 2] = x[2];
		Kparams.Kangs[kang_ind * 12 + 3] = x[3];
		Kparams.Kangs[kang_ind * 12 + 4] = y[0];
		Kparams.Kangs[kang_ind * 12 + 5] = y[1];
		Kparams.Kangs[kang_ind * 12 + 6] = y[2];
		Kparams.Kangs[kang_ind * 12 + 7] = y[3];
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CallGpuKernelABC(TKparams Kparams)
{
	KernelA <<< Kparams.BlockCnt, Kparams.BlockSize, Kparams.KernelA_LDS_Size >>> (Kparams);
	KernelB <<< Kparams.BlockCnt, Kparams.BlockSize, Kparams.KernelB_LDS_Size >>> (Kparams);
	KernelC <<< Kparams.BlockCnt, Kparams.BlockSize, Kparams.KernelC_LDS_Size >>> (Kparams);
}

void CallGpuKernelGen(TKparams Kparams)
{
	KernelGen << < Kparams.BlockCnt, Kparams.BlockSize, 0 >> > (Kparams);
}

#ifdef COLLIDER_CHECKPOINT_CAPTURE
// v1.5.5 per-kangaroo checkpoint capture (task #9) host helpers. extern "C"
// so the wrapper / validation harness can arm and drain capture without
// touching the __device__ symbols directly. Compiled only when
// COLLIDER_CHECKPOINT_CAPTURE is defined.

// Arm capture for the window [base, base + CHK_MAX_KANG) of global kangaroo
// indices. Resets all per-kangaroo counters and rings so a fresh run starts
// from each captured kangaroo's birth.
extern "C" void ckpt_enable_capture(unsigned int base)
{
	u32 one = 1, b = base;
	static u64 zeros_jumps[CHK_MAX_KANG];
	static u32 zeros_cnt[CHK_MAX_KANG];
	static u32 zeros_loop[CHK_MAX_KANG];
	for (u32 i = 0; i < CHK_MAX_KANG; i++) { zeros_jumps[i] = 0; zeros_cnt[i] = 0; zeros_loop[i] = 0; }
	cudaMemcpyToSymbol(g_ckpt_jumps, zeros_jumps, sizeof(zeros_jumps));
	cudaMemcpyToSymbol(g_ckpt_cnt, zeros_cnt, sizeof(zeros_cnt));
	cudaMemcpyToSymbol(g_ckpt_loopesc, zeros_loop, sizeof(zeros_loop));
	cudaMemcpyToSymbol(g_ckpt_base, &b, sizeof(u32));
	cudaMemcpyToSymbol(g_ckpt_on, &one, sizeof(u32));
}

extern "C" void ckpt_disable_capture()
{
	u32 zero = 0;
	cudaMemcpyToSymbol(g_ckpt_on, &zero, sizeof(u32));
}

// Constants for the host so it can size buffers without re-deriving them.
extern "C" unsigned int ckpt_interval()      { return CHK_INTERVAL; }
extern "C" unsigned int ckpt_max_kang()       { return CHK_MAX_KANG; }
extern "C" unsigned int ckpt_max_cp()         { return CHK_MAX_CP; }

// Read back the captured chain for one captured slot (0-based within the
// window). `dist_out` receives cnt*3 u64 (signed 192-bit little-endian
// distances), `l1s2_out` cnt bytes, `loopesc_out` the loop-escape event count
// for that kangaroo. Returns the number of checkpoints captured (incl. birth).
extern "C" unsigned int ckpt_readback_slot(unsigned int slot,
                                           unsigned long long* dist_out,
                                           unsigned char* l1s2_out,
                                           unsigned int* loopesc_out)
{
	if (slot >= CHK_MAX_KANG) return 0;
	u32 cnt = 0;
	cudaMemcpyFromSymbol(&cnt, g_ckpt_cnt, sizeof(u32),
	                     (size_t)slot * sizeof(u32));
	if (cnt > CHK_MAX_CP) cnt = CHK_MAX_CP;
	if (loopesc_out)
		cudaMemcpyFromSymbol(loopesc_out, g_ckpt_loopesc, sizeof(u32),
		                     (size_t)slot * sizeof(u32));
	if (cnt)
	{
		cudaMemcpyFromSymbol(dist_out, g_ckpt_dist,
		                     (size_t)cnt * 3 * sizeof(u64),
		                     (size_t)slot * CHK_MAX_CP * 3 * sizeof(u64));
		cudaMemcpyFromSymbol(l1s2_out, g_ckpt_l1s2,
		                     (size_t)cnt * sizeof(u8),
		                     (size_t)slot * CHK_MAX_CP * sizeof(u8));
	}
	return cnt;
}
#endif // COLLIDER_CHECKPOINT_CAPTURE

#ifdef COLLIDER_WALKSTEP_TRACE
// v1.5.5 walk-step validation (task #7) host helpers. Toggle / read the
// single-kangaroo debug trace captured by KernelA. extern "C" so the test
// harness can call them without touching the __device__ symbols directly.
// Compiled only in a -DCOLLIDER_WALKSTEP_TRACE validation build.
extern "C" void dbg_enable_capture()
{
	u32 one = 1, zero = 0;
	cudaMemcpyToSymbol(g_dbg_n, &zero, sizeof(u32));
	cudaMemcpyToSymbol(g_dbg_on, &one, sizeof(u32));
}

extern "C" void dbg_disable_capture()
{
	u32 zero = 0;
	cudaMemcpyToSymbol(g_dbg_on, &zero, sizeof(u32));
}

// Copies up to DBG_CAP captured steps into caller buffers. birth_x/birth_y
// are 4 u64 each; xs/ys are 4*DBG_CAP u64 each; jmps is DBG_CAP u32.
// Returns the number of steps actually captured.
extern "C" u32 dbg_readback(u64* birth_x, u64* birth_y,
                            u64* xs, u64* ys, u32* jmps)
{
	u32 n = 0;
	cudaMemcpyFromSymbol(&n, g_dbg_n, sizeof(u32));
	if (n > DBG_CAP) n = DBG_CAP;
	cudaMemcpyFromSymbol(birth_x, g_dbg_birth_x, 4 * sizeof(u64));
	cudaMemcpyFromSymbol(birth_y, g_dbg_birth_y, 4 * sizeof(u64));
	if (n)
	{
		cudaMemcpyFromSymbol(xs, g_dbg_x, (size_t)n * 4 * sizeof(u64));
		cudaMemcpyFromSymbol(ys, g_dbg_y, (size_t)n * 4 * sizeof(u64));
		cudaMemcpyFromSymbol(jmps, g_dbg_jmp, (size_t)n * sizeof(u32));
	}
	return n;
}
#endif // COLLIDER_WALKSTEP_TRACE

cudaError_t cuSetGpuParams(TKparams Kparams, u64* _jmp2_table)
{
	cudaError_t err = cudaFuncSetAttribute(KernelA, cudaFuncAttributeMaxDynamicSharedMemorySize, Kparams.KernelA_LDS_Size);
	if (err != cudaSuccess)
		return err;
	err = cudaFuncSetAttribute(KernelB, cudaFuncAttributeMaxDynamicSharedMemorySize, Kparams.KernelB_LDS_Size);
	if (err != cudaSuccess)
		return err;
	err = cudaFuncSetAttribute(KernelC, cudaFuncAttributeMaxDynamicSharedMemorySize, Kparams.KernelC_LDS_Size);
	if (err != cudaSuccess)
		return err;
	err = cudaMemcpyToSymbol(jmp2_table, _jmp2_table, JMP_CNT * 64);
	if (err != cudaSuccess)
		return err;
	return cudaSuccess;
}
