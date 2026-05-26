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
//     - Added a public Mode field on RCGpuKang (default KANG_MODE_BOTH)
//       so the consumer can configure asymmetric tame-only or wild-only
//       execution before Prepare() is called. Supports the v1.5 pool
//       protocol's asymmetric work assignment.
// ============================================================================


#pragma once

#include "Ec.h"

#define STATS_WND_SIZE	16

struct EcJMP
{
	EcPoint p;
	EcInt dist;
};

//96bytes size
struct TPointPriv
{
	u64 x[4];
	u64 y[4];
	u64 priv[4];
};

class RCGpuKang
{
private:
	bool StopFlag;
	EcPoint PntToSolve;
	int Range; //in bits
	int DP; //in bits
	Ec ec;

	u32* DPs_out;
	TKparams Kparams;

	EcInt HalfRange;
	EcPoint PntHalfRange;
	EcPoint NegPntHalfRange;
	TPointPriv* RndPnts;
	EcJMP* EcJumps1;
	EcJMP* EcJumps2;
	EcJMP* EcJumps3;

	EcPoint PntA;
	EcPoint PntB;

	int cur_stats_ind;
	int SpeedStats[STATS_WND_SIZE];

	void GenerateRndDistances();
	bool Start();
	void Release();
#ifdef DEBUG_MODE
	int Dbg_CheckKangs();
#endif
public:
	int persistingL2CacheMaxSize;
	int CudaIndex; //gpu index in cuda
	int mpCnt;
	int KangCnt;
	bool Failed;
	bool IsOldGpu;

	// theCollider v1.4.2 patch: herd save/load hooks (see
	// third_party/RCKangaroo/.patches/save-load-state.patch for the
	// rationale and on-disk format). When InitKangsHost is non-null at
	// Start() time, the GPU's per-kangaroo buffer is seeded directly
	// from it instead of randomly generated. When SaveKangsHost is
	// non-null, the buffer is downloaded to it AFTER the Execute loop
	// stops but BEFORE Release() frees the device memory. Both pointers
	// are owned by the caller (RCKangarooManager); the patch only reads
	// the InitKangs buffer and writes the SaveKangs buffer.
	//
	// Buffer size in bytes: KangCnt * 96 (24 u64 per kangaroo: x[4],
	// y[4], priv[4]). Matches the cudaMalloc size at GpuKang.cpp Prepare.
	const unsigned char* InitKangsHost = nullptr;
	unsigned char*       SaveKangsHost = nullptr;

	// theCollider v1.5: asymmetric kangaroo mode (default BOTH preserves
	// upstream behavior). Set BEFORE calling Prepare(). Persisted into
	// Kparams.Mode by Prepare(). See defs.h KANG_MODE_* and the v1.5
	// plan for the theft-resistance rationale.
	//
	// In TAME_ONLY: every kangaroo starts as a tame (RndPnts[i].x = 0,
	// distance ~ Range - 4), the host-side hashtable is the caller's
	// problem (the upstream RCKangaroo.cpp main still uses it; the
	// pool-mode wrapper bypasses it), the DPs that come back through
	// AddPointsToList all carry type = TAME, and the solve loop does
	// not terminate on found=true (the caller stops via Stop()).
	//
	// In WILD_ONLY: every kangaroo starts as wild1 (RndPnts[i].x =
	// PntA, distance ~ Range - 1, even), DPs carry type = WILD1, and
	// the solve loop similarly runs until external Stop().
	int Mode = KANG_MODE_BOTH;

	int CalcKangCnt();
	bool Prepare(EcPoint _PntToSolve, int _Range, int _DP, EcJMP* _EcJumps1, EcJMP* _EcJumps2, EcJMP* _EcJumps3);
	void Stop();
	void Execute();

	u32 dbg[256];

	int GetStatsSpeed();
};
