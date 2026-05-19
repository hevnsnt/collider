// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


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

	int CalcKangCnt();
	bool Prepare(EcPoint _PntToSolve, int _Range, int _DP, EcJMP* _EcJumps1, EcJMP* _EcJumps2, EcJMP* _EcJumps3);
	void Stop();
	void Execute();

	u32 dbg[256];

	int GetStatsSpeed();
};
