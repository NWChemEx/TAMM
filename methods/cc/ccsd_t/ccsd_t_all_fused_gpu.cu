/*
	Two Different Fully-Fused Kernels
	(1) Pure FP64
	(2) 3rd. Generation Tensor Cores (FP64)
*/
// (1) Pure FP64
#include "ccsd_t_common.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <vector>

// created by tc_gen_definition()
#define FUSION_SIZE_SLICE_1_H3 	4
#define FUSION_SIZE_SLICE_1_H2 	4
#define FUSION_SIZE_SLICE_1_H1 	4
#define FUSION_SIZE_SLICE_1_P6 	4
#define FUSION_SIZE_SLICE_1_P5 	4
#define FUSION_SIZE_SLICE_1_P4 	4
#define FUSION_SIZE_SLICE_1_H7 	16

#define FUSION_SIZE_SLICE_2_H3 	4
#define FUSION_SIZE_SLICE_2_H2 	4
#define FUSION_SIZE_SLICE_2_H1 	4
#define FUSION_SIZE_SLICE_2_P6 	4
#define FUSION_SIZE_SLICE_2_P5 	4
#define FUSION_SIZE_SLICE_2_P4 	4
#define FUSION_SIZE_SLICE_2_H7 	16

#define FUSION_SIZE_INT_UNIT 	FUSION_SIZE_SLICE_1_H7

#define FUSION_SIZE_TB_1_X 		FUSION_SIZE_SLICE_1_H3 * FUSION_SIZE_SLICE_1_H2
#define FUSION_SIZE_TB_1_Y 		FUSION_SIZE_SLICE_1_P6 * FUSION_SIZE_SLICE_1_H1
#define FUSION_SIZE_REG_1_X 	FUSION_SIZE_SLICE_1_P5
#define FUSION_SIZE_REG_1_Y 	FUSION_SIZE_SLICE_1_P4

#define FUSION_SIZE_TB_2_X 		FUSION_SIZE_SLICE_2_H3 * FUSION_SIZE_SLICE_2_H2
#define FUSION_SIZE_TB_2_Y 		FUSION_SIZE_SLICE_2_P4 * FUSION_SIZE_SLICE_2_H1
#define FUSION_SIZE_REG_2_X 	FUSION_SIZE_SLICE_2_P5
#define FUSION_SIZE_REG_2_Y 	FUSION_SIZE_SLICE_2_P6

#define NUM_INDEX 	    		6
#define CEIL(a, b)      		(((a) + (b) - 1) / (b))

#define NUM_IA6_LOOPS           9
#define NUM_D1_EQUATIONS        9
#define NUM_D2_EQUATIONS        9
#define NUM_S1_EQUATIONS        9
#define NUM_D1_INDEX            7
#define NUM_D2_INDEX            7
#define NUM_S1_INDEX            6
#define NUM_ENERGIES            2
#define FULL_MASK 				0xffffffff

#define MAX_NOAB								30
#define MAX_NVAB 								120

// 64 KB = 65536 bytes = 16384 (int) = 8192 (size_t)
// 9 * 9 * noab = 81 * noab 

// 
// 	|constant memory| = sizeof(int) * {(6 + 9) + ((7 + 9) * MAX_NOAB) + ((7 + 9) * MAX_NVAB)}
// 										= 4 bytes * (15 + 16 * 20 + 16 * 80) = 8 bytes * (15 + 320 + 1280) = 1615 * 4 bytes = 6460 bytes (6.30 KB)
// 
__constant__ int const_df_s1_size[6];
__constant__ int const_df_s1_exec[9];
__constant__ int const_df_d1_size[7 * MAX_NOAB];
__constant__ int const_df_d1_exec[9 * MAX_NOAB];
__constant__ int const_df_d2_size[7 * MAX_NVAB];
__constant__ int const_df_d2_exec[9 * MAX_NVAB];

__global__ 
void revised_jk_ccsd_t_fully_fused_kernel(int size_noab, int size_nvab, 
																	// 	common
																	int size_max_dim_s1_t1, int size_max_dim_s1_v2, 
																	int size_max_dim_d1_t2, int size_max_dim_d1_v2, 
																	int size_max_dim_d2_t2, int size_max_dim_d2_v2, 
																	// 
																	double* df_dev_d1_t2_all, double* df_dev_d1_v2_all,
																	double* df_dev_d2_t2_all, double* df_dev_d2_v2_all,
																	double* df_dev_s1_t1_all, double* df_dev_s1_v2_all,									
																	//  energies
																	const double* dev_evl_sorted_h1b, const double* dev_evl_sorted_h2b, const double* dev_evl_sorted_h3b,
																	const double* dev_evl_sorted_p4b, const double* dev_evl_sorted_p5b, const double* dev_evl_sorted_p6b, 
																	// 	not-fully reduced results
																	double* reduced_energy,
																	//  common
																	int num_blks_h3b, int num_blks_h2b, int num_blks_h1b, 
																	int num_blks_p6b, int num_blks_p5b, int num_blks_p4b, 
																	// 
																	int base_size_h1b, int base_size_h2b, int base_size_h3b, 
																	int base_size_p4b, int base_size_p5b, int base_size_p6b)
{
	// For Shared Memory,
	__shared__ double sm_a[16][64 + 1];
	__shared__ double sm_b[16][64 + 1];
	
	int internal_upperbound = 0;
	int internal_offset;

	// should support for non-full tiles
	int idx_h3 			= threadIdx.x % FUSION_SIZE_SLICE_1_H3;
	int idx_h2 			= threadIdx.x / FUSION_SIZE_SLICE_1_H3;
	int idx_p6 			= threadIdx.y % FUSION_SIZE_SLICE_1_P6;
	int idx_h1 			= threadIdx.y / FUSION_SIZE_SLICE_1_P6;
	   
	int blk_idx_p4b     = blockIdx.x / (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b * num_blks_p5b);
	int tmp_blkIdx      = blockIdx.x % (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b * num_blks_p5b);
	int blk_idx_p5b     = (tmp_blkIdx) / (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b);
	tmp_blkIdx          = (tmp_blkIdx) % (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b);
	int blk_idx_p6b     = (tmp_blkIdx) / (num_blks_h3b * num_blks_h2b * num_blks_h1b);
	tmp_blkIdx          = (tmp_blkIdx) % (num_blks_h3b * num_blks_h2b * num_blks_h1b);
	int blk_idx_h1b     = (tmp_blkIdx) / (num_blks_h3b * num_blks_h2b);
	tmp_blkIdx          = (tmp_blkIdx) % (num_blks_h3b * num_blks_h2b);
	int blk_idx_h2b     = (tmp_blkIdx) / (num_blks_h3b);
	int blk_idx_h3b     = blockIdx.x % (num_blks_h3b);

	int str_blk_idx_h3 	= blk_idx_h3b * FUSION_SIZE_SLICE_1_H3;
	int str_blk_idx_h2 	= blk_idx_h2b * FUSION_SIZE_SLICE_1_H2;
	int str_blk_idx_h1 	= blk_idx_h1b * FUSION_SIZE_SLICE_1_H1;
	int str_blk_idx_p6 	= blk_idx_p6b * FUSION_SIZE_SLICE_1_P6;
	int str_blk_idx_p5 	= blk_idx_p5b * FUSION_SIZE_SLICE_1_P5;
	int str_blk_idx_p4 	= blk_idx_p4b * FUSION_SIZE_SLICE_1_P4;

	// 
	int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;
	int energy_rng_h3, energy_rng_h2, energy_rng_h1, energy_rng_p6, energy_rng_p5, energy_rng_p4;
	if ((base_size_h3b - (str_blk_idx_h3)) >= FUSION_SIZE_SLICE_1_H3)
	{
		energy_rng_h3 = FUSION_SIZE_SLICE_1_H3;
	}
	else
	{
		energy_rng_h3 = base_size_h3b % FUSION_SIZE_SLICE_1_H3;
	}
	
	if ((base_size_h2b - (str_blk_idx_h2)) >= FUSION_SIZE_SLICE_1_H2)
	{
		energy_rng_h2 = FUSION_SIZE_SLICE_1_H2;
	}
	else
	{
		energy_rng_h2 = base_size_h2b % FUSION_SIZE_SLICE_1_H2;
	}

	if ((base_size_h1b - (str_blk_idx_h1)) >= FUSION_SIZE_SLICE_1_H1)
	{
		energy_rng_h1 = FUSION_SIZE_SLICE_1_H1;
	}
	else
	{
		energy_rng_h1 = base_size_h1b % FUSION_SIZE_SLICE_1_H1;
	}
	
	if ((base_size_p6b - (str_blk_idx_p6)) >= FUSION_SIZE_SLICE_1_P6)
	{
		energy_rng_p6 = FUSION_SIZE_SLICE_1_P6;
	}
	else
	{
		energy_rng_p6 = base_size_p6b % FUSION_SIZE_SLICE_1_P6;
	}

	if ((base_size_p5b - (str_blk_idx_p5)) >= FUSION_SIZE_SLICE_1_P5)
	{
		energy_rng_p5 = FUSION_SIZE_SLICE_1_P5;
	}
	else
	{
		energy_rng_p5 = base_size_p5b % FUSION_SIZE_SLICE_1_P5;
	}

	if ((base_size_p4b - (str_blk_idx_p4)) >= FUSION_SIZE_SLICE_1_P4)
	{
		energy_rng_p4 = FUSION_SIZE_SLICE_1_P4;
	}
	else
	{
		energy_rng_p4 = base_size_p4b % FUSION_SIZE_SLICE_1_P4;
	}

	// 
	double temp_av;
	double temp_bv[4];
	double reg_tile[4][4];
	double reg_singles[4][4];

	int base_size_h7b, base_size_p7b;

	for (int i = 0; i < 4; i++)
	for (int j = 0; j < 4; j++)
	{
		reg_tile[i][j]      = 0.0;
		reg_singles[i][j]   = 0.0;
	}
	

	int energy_str_blk_idx_p4 = str_blk_idx_p4;
	int energy_str_blk_idx_p5 = str_blk_idx_p5;
	double eval_h3 = dev_evl_sorted_h3b[str_blk_idx_h3 + idx_h3];
	double eval_h2 = dev_evl_sorted_h2b[str_blk_idx_h2 + idx_h2];
	double eval_p6 = dev_evl_sorted_p6b[str_blk_idx_p6 + idx_p6];
	double eval_h1 = dev_evl_sorted_h1b[str_blk_idx_h1 + idx_h1];

	double partial_inner_factor = eval_h3 + eval_h2 + eval_h1 - eval_p6;

	//
	//  energies
	// 
	double energy_1 = 0.0;
	double energy_2 = 0.0;

	#pragma unroll 1
	for (int iter_noab = 0; iter_noab < size_noab; iter_noab++)
	{
		// 
		int flag_d1_1 = const_df_d1_exec[0 + (iter_noab) * NUM_D1_EQUATIONS];
		int flag_d1_2 = const_df_d1_exec[1 + (iter_noab) * NUM_D1_EQUATIONS];
		int flag_d1_3 = const_df_d1_exec[2 + (iter_noab) * NUM_D1_EQUATIONS];

		// 
		base_size_h1b = const_df_d1_size[0 + (iter_noab) * NUM_D1_INDEX];
		base_size_h2b = const_df_d1_size[1 + (iter_noab) * NUM_D1_INDEX];
		base_size_h3b = const_df_d1_size[2 + (iter_noab) * NUM_D1_INDEX];
		base_size_h7b = const_df_d1_size[3 + (iter_noab) * NUM_D1_INDEX];
		base_size_p4b = const_df_d1_size[4 + (iter_noab) * NUM_D1_INDEX];
		base_size_p5b = const_df_d1_size[5 + (iter_noab) * NUM_D1_INDEX];
		base_size_p6b = const_df_d1_size[6 + (iter_noab) * NUM_D1_INDEX];

		// 
		num_blks_h1b = CEIL(base_size_h1b, FUSION_SIZE_SLICE_1_H1);
		num_blks_h2b = CEIL(base_size_h2b, FUSION_SIZE_SLICE_1_H2);
		num_blks_h3b = CEIL(base_size_h3b, FUSION_SIZE_SLICE_1_H3);
		num_blks_p4b = CEIL(base_size_p4b, FUSION_SIZE_SLICE_1_P4);
		num_blks_p5b = CEIL(base_size_p5b, FUSION_SIZE_SLICE_1_P5);
		num_blks_p6b = CEIL(base_size_p6b, FUSION_SIZE_SLICE_1_P6);

		// 	(2) blk_idx_h/p*b
		blk_idx_p4b     = blockIdx.x / (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b * num_blks_p5b);
		tmp_blkIdx      = blockIdx.x % (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b * num_blks_p5b);
		blk_idx_p5b     = (tmp_blkIdx) / (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b);
		tmp_blkIdx  	= (tmp_blkIdx) % (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b);
		blk_idx_p6b     = (tmp_blkIdx) / (num_blks_h3b * num_blks_h2b * num_blks_h1b);
		tmp_blkIdx 		= (tmp_blkIdx) % (num_blks_h3b * num_blks_h2b * num_blks_h1b);
		blk_idx_h1b     = (tmp_blkIdx) / (num_blks_h3b * num_blks_h2b);
		tmp_blkIdx 		= (tmp_blkIdx) % (num_blks_h3b * num_blks_h2b);
		blk_idx_h2b 	= (tmp_blkIdx) / (num_blks_h3b);
		blk_idx_h3b		= blockIdx.x % (num_blks_h3b);

		// 	(3) str_blk_idx_h/p*
		str_blk_idx_h3 	= blk_idx_h3b * FUSION_SIZE_SLICE_1_H3;
		str_blk_idx_h2 	= blk_idx_h2b * FUSION_SIZE_SLICE_1_H2;
		str_blk_idx_h1 	= blk_idx_h1b * FUSION_SIZE_SLICE_1_H1;
		str_blk_idx_p6 	= blk_idx_p6b * FUSION_SIZE_SLICE_1_P6;
		str_blk_idx_p5 	= blk_idx_p5b * FUSION_SIZE_SLICE_1_P5;
		str_blk_idx_p4 	= blk_idx_p4b * FUSION_SIZE_SLICE_1_P4;

		// 	(4) rng_h/p*
		if ((base_size_h3b - (str_blk_idx_h3)) >= FUSION_SIZE_SLICE_1_H3)
			rng_h3 = FUSION_SIZE_SLICE_1_H3;
		else
			rng_h3 = base_size_h3b % FUSION_SIZE_SLICE_1_H3;
		
		if ((base_size_h2b - (str_blk_idx_h2)) >= FUSION_SIZE_SLICE_1_H2)
			rng_h2 = FUSION_SIZE_SLICE_1_H2;
		else
			rng_h2 = base_size_h2b % FUSION_SIZE_SLICE_1_H2;

		if ((base_size_h1b - (str_blk_idx_h1)) >= FUSION_SIZE_SLICE_1_H1)
			rng_h1 = FUSION_SIZE_SLICE_1_H1;
		else
			rng_h1 = base_size_h1b % FUSION_SIZE_SLICE_1_H1;
		
		if ((base_size_p6b - (str_blk_idx_p6)) >= FUSION_SIZE_SLICE_1_P6)
			rng_p6 = FUSION_SIZE_SLICE_1_P6;
		else
			rng_p6 = base_size_p6b % FUSION_SIZE_SLICE_1_P6;

		if ((base_size_p5b - (str_blk_idx_p5)) >= FUSION_SIZE_SLICE_1_P5)
			rng_p5 = FUSION_SIZE_SLICE_1_P5;
		else
			rng_p5 = base_size_p5b % FUSION_SIZE_SLICE_1_P5;

		if ((base_size_p4b - (str_blk_idx_p4)) >= FUSION_SIZE_SLICE_1_P4)
			rng_p4 = FUSION_SIZE_SLICE_1_P4;
		else
			rng_p4 = base_size_p4b % FUSION_SIZE_SLICE_1_P4;

		//  sd1_1
		if (flag_d1_1 >= 0)
		{
			// 
			double* tmp_dev_d1_t2 = df_dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_1;
			double* tmp_dev_d1_v2 = df_dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_1;

			// 
			internal_upperbound = 0;
			#pragma unroll 1
			for (int l = 0; l < base_size_h7b; l+= FUSION_SIZE_INT_UNIT)
			{
				// Part: Generalized Contraction Index (p7b)
				internal_offset = (l + FUSION_SIZE_INT_UNIT) - base_size_h7b;
				if (internal_offset > 0) internal_upperbound = internal_offset;

				// Load Input Tensor to Shared Memory: 16:16
				// # of size_internal Indices: 1
				if (idx_p6 < rng_p4 && idx_h1 < rng_h1 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
				for (int ll = 0; ll < rng_p5; ll++)
				{
					sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = tmp_dev_d1_t2[(str_blk_idx_p4 + idx_p6 + (str_blk_idx_p5 + ll + (str_blk_idx_h1 + idx_h1) * base_size_p5b) * base_size_p4b) * base_size_h7b + (threadIdx.x + l)];
				}

				// Load Input Tensor to Shared Memory
				if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
				for (int ll = 0; ll < rng_p6; ll++)
				{
					sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_2_X] = tmp_dev_d1_v2[str_blk_idx_h3 + idx_h3 + (str_blk_idx_h2 + idx_h2 + (str_blk_idx_p6 + ll + (threadIdx.y + l) * base_size_p6b) * base_size_h2b) * base_size_h3b];
				}
				__syncthreads();

				// Cross-Product: -1
				// Part: Generalized Threads
				for (int ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++)
				{
					temp_bv[0] = sm_b[ll][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_2_H3 + 0];
					temp_bv[1] = sm_b[ll][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_2_H3 + 16];
					temp_bv[2] = sm_b[ll][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_2_H3 + 32];
					temp_bv[3] = sm_b[ll][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_2_H3 + 48];

					for (int xx = 0 ; xx < 4; xx++)
					{
						temp_av = sm_a[ll][idx_p6 + (idx_h1) * FUSION_SIZE_SLICE_1_P6 + (xx * 16)];

						reg_tile[0][xx] -= temp_av * temp_bv[0];
						reg_tile[1][xx] -= temp_av * temp_bv[1];
						reg_tile[2][xx] -= temp_av * temp_bv[2];
						reg_tile[3][xx] -= temp_av * temp_bv[3];
					}
				}
				__syncthreads();
			}
		}

		//  sd1_2
		if (flag_d1_2 >= 0)
		{
			// 
			double* tmp_dev_d1_t2 = df_dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_2;
			double* tmp_dev_d1_v2 = df_dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_2;

			internal_upperbound = 0;
			#pragma unroll 1
			for (int l = 0; l < base_size_h7b; l+= FUSION_SIZE_INT_UNIT)
			{
				// Part: Generalized Contraction Index (p7b)
				internal_offset = (l + FUSION_SIZE_INT_UNIT) - base_size_h7b;
				if (internal_offset > 0) internal_upperbound = internal_offset;

				// Load Input Tensor to Shared Memory: 16:16
				// # of size_internal Indices: 1
				if (idx_p6 < rng_p4 && idx_h1 < rng_h2 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
				for (int ll = 0; ll < rng_p5; ll++)
				{
					sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = tmp_dev_d1_t2[(str_blk_idx_p4 + idx_p6 + (str_blk_idx_p5 + ll + (str_blk_idx_h2 + idx_h1) * base_size_p5b) * base_size_p4b) * base_size_h7b + (threadIdx.x + l)];
				}

				// Load Input Tensor to Shared Memory
				if (idx_h3 < rng_h3 && idx_h2 < rng_h1 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
				for (int ll = 0; ll < rng_p6; ll++)
				{
					sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_2_X] = tmp_dev_d1_v2[str_blk_idx_h3 + idx_h3 + (str_blk_idx_h1 + idx_h2 + (str_blk_idx_p6 + ll + (threadIdx.y + l) * base_size_p6b) * base_size_h1b) * base_size_h3b]; 
				}
				__syncthreads();

				// Cross-Product: -1
				// Part: Generalized Threads
				for (int ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++)
				{
					temp_bv[0] = sm_b[ll][idx_h3 + (idx_h1) * FUSION_SIZE_SLICE_2_H3 + 0];
					temp_bv[1] = sm_b[ll][idx_h3 + (idx_h1) * FUSION_SIZE_SLICE_2_H3 + 16];
					temp_bv[2] = sm_b[ll][idx_h3 + (idx_h1) * FUSION_SIZE_SLICE_2_H3 + 32];
					temp_bv[3] = sm_b[ll][idx_h3 + (idx_h1) * FUSION_SIZE_SLICE_2_H3 + 48];

					for (int xx = 0 ; xx < 4; xx++)
					{
						temp_av = sm_a[ll][idx_p6 + (idx_h2) * FUSION_SIZE_SLICE_2_P4 + (xx * 16)];

						reg_tile[0][xx] += temp_av * temp_bv[0];
						reg_tile[1][xx] += temp_av * temp_bv[1];
						reg_tile[2][xx] += temp_av * temp_bv[2];
						reg_tile[3][xx] += temp_av * temp_bv[3];
					}
				}
				__syncthreads();
			}
		}

		//  sd1_3
		if (flag_d1_3 >= 0)
		{
			// 
			double* tmp_dev_d1_t2 = df_dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_3;
			double* tmp_dev_d1_v2 = df_dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_3;
			
			internal_upperbound = 0;
			#pragma unroll 1
			for (int l = 0; l < base_size_h7b; l+= FUSION_SIZE_INT_UNIT)
			{
				// Part: Generalized Contraction Index (p7b)
				internal_offset = (l + FUSION_SIZE_INT_UNIT) - base_size_h7b;
				if (internal_offset > 0) internal_upperbound = internal_offset;

				// Load Input Tensor to Shared Memory: 16:16
				// # of size_internal Indices: 1
				if (idx_p6 < rng_p4 && idx_h1 < rng_h3 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
				for (int ll = 0; ll < rng_p5; ll++)
				{
					sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = tmp_dev_d1_t2[(str_blk_idx_p4 + idx_p6 + (str_blk_idx_p5 + ll + (str_blk_idx_h3 + idx_h1) * base_size_p5b) * base_size_p4b) * base_size_h7b + (threadIdx.x + l)];
				}

				// Load Input Tensor to Shared Memory
				if (idx_h3 < rng_h2 && idx_h2 < rng_h1 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
				for (int ll = 0; ll < rng_p6; ll++)
				{
					sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_2_X] = tmp_dev_d1_v2[(str_blk_idx_h2 + idx_h3 + (str_blk_idx_h1 + idx_h2 + (str_blk_idx_p6 + ll + (threadIdx.y + l) * base_size_p6b) * base_size_h1b) * base_size_h2b)];
				}
				__syncthreads();

				// Cross-Product: -1
				// Part: Generalized Threads
				for (int ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++)
				{
					temp_bv[0] = sm_b[ll][idx_h2 + (idx_h1) * FUSION_SIZE_SLICE_2_H2 + 0];
					temp_bv[1] = sm_b[ll][idx_h2 + (idx_h1) * FUSION_SIZE_SLICE_2_H2 + 16];
					temp_bv[2] = sm_b[ll][idx_h2 + (idx_h1) * FUSION_SIZE_SLICE_2_H2 + 32];
					temp_bv[3] = sm_b[ll][idx_h2 + (idx_h1) * FUSION_SIZE_SLICE_2_H2 + 48];

					for (int xx = 0 ; xx < 4; xx++)
					{
						temp_av = sm_a[ll][idx_p6 + (idx_h3) * FUSION_SIZE_SLICE_2_P4 + (xx * 16)];

						reg_tile[0][xx] -= temp_av * temp_bv[0];
						reg_tile[1][xx] -= temp_av * temp_bv[1];
						reg_tile[2][xx] -= temp_av * temp_bv[2];
						reg_tile[3][xx] -= temp_av * temp_bv[3];
					}
				}
				__syncthreads();
			}
		}
	}

	//  d2-top: sd2_7, 8 and 9
	#pragma unroll 1
	for (int iter_nvab = 0; iter_nvab < size_nvab; iter_nvab++)
	{
		// 
		int flag_d2_7 = const_df_d2_exec[6 + (iter_nvab) * NUM_D2_EQUATIONS];
		int flag_d2_8 = const_df_d2_exec[7 + (iter_nvab) * NUM_D2_EQUATIONS];
		int flag_d2_9 = const_df_d2_exec[8 + (iter_nvab) * NUM_D2_EQUATIONS];

		// 
		base_size_h1b = const_df_d2_size[0 + (iter_nvab) * NUM_D2_INDEX];
		base_size_h2b = const_df_d2_size[1 + (iter_nvab) * NUM_D2_INDEX];
		base_size_h3b = const_df_d2_size[2 + (iter_nvab) * NUM_D2_INDEX];
		base_size_p4b = const_df_d2_size[3 + (iter_nvab) * NUM_D2_INDEX];
		base_size_p5b = const_df_d2_size[4 + (iter_nvab) * NUM_D2_INDEX];
		base_size_p6b = const_df_d2_size[5 + (iter_nvab) * NUM_D2_INDEX];
		base_size_p7b = const_df_d2_size[6 + (iter_nvab) * NUM_D2_INDEX];

		// 
		num_blks_h1b = CEIL(base_size_h1b, FUSION_SIZE_SLICE_1_H1);
		num_blks_h2b = CEIL(base_size_h2b, FUSION_SIZE_SLICE_1_H2);
		num_blks_h3b = CEIL(base_size_h3b, FUSION_SIZE_SLICE_1_H3);
		num_blks_p4b = CEIL(base_size_p4b, FUSION_SIZE_SLICE_1_P4);
		num_blks_p5b = CEIL(base_size_p5b, FUSION_SIZE_SLICE_1_P5);
		num_blks_p6b = CEIL(base_size_p6b, FUSION_SIZE_SLICE_1_P6);

		// 	(2) blk_idx_h/p*b
		blk_idx_p4b     = blockIdx.x / (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b * num_blks_p5b);
		tmp_blkIdx      = blockIdx.x % (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b * num_blks_p5b);
		blk_idx_p5b     = (tmp_blkIdx) / (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b);
		tmp_blkIdx  	= (tmp_blkIdx) % (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b);
		blk_idx_p6b     = (tmp_blkIdx) / (num_blks_h3b * num_blks_h2b * num_blks_h1b);
		tmp_blkIdx 		= (tmp_blkIdx) % (num_blks_h3b * num_blks_h2b * num_blks_h1b);
		blk_idx_h1b     = (tmp_blkIdx) / (num_blks_h3b * num_blks_h2b);
		tmp_blkIdx 		= (tmp_blkIdx) % (num_blks_h3b * num_blks_h2b);
		blk_idx_h2b 	= (tmp_blkIdx) / (num_blks_h3b);
		blk_idx_h3b		= blockIdx.x % (num_blks_h3b);

		// 	(3) str_blk_idx_h/p*
		str_blk_idx_h3 	= blk_idx_h3b * FUSION_SIZE_SLICE_1_H3;
		str_blk_idx_h2 	= blk_idx_h2b * FUSION_SIZE_SLICE_1_H2;
		str_blk_idx_h1 	= blk_idx_h1b * FUSION_SIZE_SLICE_1_H1;
		str_blk_idx_p6 	= blk_idx_p6b * FUSION_SIZE_SLICE_1_P6;
		str_blk_idx_p5 	= blk_idx_p5b * FUSION_SIZE_SLICE_1_P5;
		str_blk_idx_p4 	= blk_idx_p4b * FUSION_SIZE_SLICE_1_P4;

		// 	(4) rng_h/p*
		if ((base_size_h3b - (str_blk_idx_h3)) >= FUSION_SIZE_SLICE_1_H3)
			rng_h3 = FUSION_SIZE_SLICE_1_H3;
		else
			rng_h3 = base_size_h3b % FUSION_SIZE_SLICE_1_H3;
		
		if ((base_size_h2b - (str_blk_idx_h2)) >= FUSION_SIZE_SLICE_1_H2)
			rng_h2 = FUSION_SIZE_SLICE_1_H2;
		else
			rng_h2 = base_size_h2b % FUSION_SIZE_SLICE_1_H2;

		if ((base_size_h1b - (str_blk_idx_h1)) >= FUSION_SIZE_SLICE_1_H1)
			rng_h1 = FUSION_SIZE_SLICE_1_H1;
		else
			rng_h1 = base_size_h1b % FUSION_SIZE_SLICE_1_H1;
		
		if ((base_size_p6b - (str_blk_idx_p6)) >= FUSION_SIZE_SLICE_1_P6)
			rng_p6 = FUSION_SIZE_SLICE_1_P6;
		else
			rng_p6 = base_size_p6b % FUSION_SIZE_SLICE_1_P6;

		if ((base_size_p5b - (str_blk_idx_p5)) >= FUSION_SIZE_SLICE_1_P5)
			rng_p5 = FUSION_SIZE_SLICE_1_P5;
		else
			rng_p5 = base_size_p5b % FUSION_SIZE_SLICE_1_P5;

		if ((base_size_p4b - (str_blk_idx_p4)) >= FUSION_SIZE_SLICE_1_P4)
			rng_p4 = FUSION_SIZE_SLICE_1_P4;
		else
			rng_p4 = base_size_p4b % FUSION_SIZE_SLICE_1_P4;

		//	sd2_7
		if (flag_d2_7 >= 0)
		{
			// 
			double* tmp_dev_d2_t2_7 = df_dev_d2_t2_all + size_max_dim_d2_t2 * flag_d2_7;//const_list_d2_flags_offset[local_offset];
			double* tmp_dev_d2_v2_7 = df_dev_d2_v2_all + size_max_dim_d2_v2 * flag_d2_7;//const_list_d2_flags_offset[local_offset];
			
			//	sd2_7
			internal_upperbound = 0;
			#pragma unroll 1
			for (int l = 0; l < base_size_p7b; l+= FUSION_SIZE_INT_UNIT)
			{
				// Part: Generalized Contraction Index (p7b)
				internal_offset = (l + FUSION_SIZE_INT_UNIT) - base_size_p7b;
				if (internal_offset > 0) internal_upperbound = internal_offset;

				// Load Input Tensor to Shared Memory: 16:16
				// # of size_internal Indices: 1
				if (idx_p6 < rng_h1 && idx_h1 < rng_h2 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
				for (int ll = 0; ll < rng_p6; ll++)
				{
					sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = tmp_dev_d2_t2_7[(blk_idx_p6b *  FUSION_SIZE_SLICE_2_P6 + ll + (str_blk_idx_h1 + idx_p6 + (str_blk_idx_h2 + idx_h1) * base_size_h1b) * base_size_p6b) * base_size_p7b + (threadIdx.x + l)];
				}

				// Load Input Tensor to Shared Memory
				if (idx_p6 < rng_h3 && idx_h1 < rng_p4 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
				for (int ll = 0; ll < rng_p5; ll++)
				{
					sm_b[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = tmp_dev_d2_v2_7[(str_blk_idx_h3 + idx_p6 + (str_blk_idx_p5 + ll + (str_blk_idx_p4 + idx_h1) * base_size_p5b) * base_size_h3b) * base_size_p7b + (threadIdx.x + l)];
				}
				__syncthreads();

				// Cross-Product: 16
				// Part: Generalized Threads
				for (int ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++)
				{
					temp_bv[0] = sm_a[ll][idx_h1 + (idx_h2) * FUSION_SIZE_SLICE_2_H1 + 0];
					temp_bv[1] = sm_a[ll][idx_h1 + (idx_h2) * FUSION_SIZE_SLICE_2_H1 + 16];
					temp_bv[2] = sm_a[ll][idx_h1 + (idx_h2) * FUSION_SIZE_SLICE_2_H1 + 32];
					temp_bv[3] = sm_a[ll][idx_h1 + (idx_h2) * FUSION_SIZE_SLICE_2_H1 + 48];

					for (int xx = 0 ; xx < 4; xx++)
					{
						temp_av = sm_b[ll][idx_h3 + (idx_p6) * FUSION_SIZE_SLICE_2_H3 + (xx * 16)];

						reg_tile[0][xx] -= temp_av * temp_bv[0];
						reg_tile[1][xx] -= temp_av * temp_bv[1];
						reg_tile[2][xx] -= temp_av * temp_bv[2];
						reg_tile[3][xx] -= temp_av * temp_bv[3];
					}
				}
				__syncthreads();
			}
		}

		// 	sd2_8
		if (flag_d2_8 >= 0)
		{
			// 
			double* tmp_dev_d2_t2_8 = df_dev_d2_t2_all + size_max_dim_d2_t2 * flag_d2_8;//const_list_d2_flags_offset[local_offset];
			double* tmp_dev_d2_v2_8 = df_dev_d2_v2_all + size_max_dim_d2_v2 * flag_d2_8;//const_list_d2_flags_offset[local_offset];

			internal_upperbound = 0;
			#pragma unroll 1
			for (int l = 0; l < base_size_p7b; l+= FUSION_SIZE_INT_UNIT)
			{
				// Part: Generalized Contraction Index (p7b)
				// internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
				internal_offset = (l + FUSION_SIZE_INT_UNIT) - base_size_p7b;
				if (internal_offset > 0) internal_upperbound = internal_offset;
		
				// Load Input Tensor to Shared Memory: 16:16
				// # of size_internal Indices: 1
				if (idx_p6 < rng_h2 && idx_h1 < rng_h3 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
				for (int ll = 0; ll < rng_p6; ll++)
				{
					sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = tmp_dev_d2_t2_8[(str_blk_idx_p6 + ll + (str_blk_idx_h2 + idx_p6 + (str_blk_idx_h3 + idx_h1) * base_size_h2b) * base_size_p6b) * base_size_p7b + (threadIdx.x + l)];
				}
		
				// Load Input Tensor to Shared Memory
				if (idx_p6 < rng_h1 && idx_h1 < rng_p4 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
				for (int ll = 0; ll < rng_p5; ll++)
				{
					sm_b[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = tmp_dev_d2_v2_8[(str_blk_idx_h1 + idx_p6 + (str_blk_idx_p5 + ll + (str_blk_idx_p4 + idx_h1) * base_size_p5b) * base_size_h1b) * base_size_p7b + (threadIdx.x + l)];
				}
				__syncthreads();
		
				// Cross-Product: 16
				// Part: Generalized Threads
				for (int ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++)
				{
					temp_bv[0] = sm_a[ll][idx_h2 + (idx_h3) * FUSION_SIZE_SLICE_2_H2 + 0];
					temp_bv[1] = sm_a[ll][idx_h2 + (idx_h3) * FUSION_SIZE_SLICE_2_H2 + 16];
					temp_bv[2] = sm_a[ll][idx_h2 + (idx_h3) * FUSION_SIZE_SLICE_2_H2 + 32];
					temp_bv[3] = sm_a[ll][idx_h2 + (idx_h3) * FUSION_SIZE_SLICE_2_H2 + 48];
		
					for (int xx = 0 ; xx < 4; xx++)
					{
						temp_av = sm_b[ll][idx_h1 + (idx_p6) * FUSION_SIZE_SLICE_2_H1 + (xx * 16)];
		
						reg_tile[0][xx] -= temp_av * temp_bv[0];
						reg_tile[1][xx] -= temp_av * temp_bv[1];
						reg_tile[2][xx] -= temp_av * temp_bv[2];
						reg_tile[3][xx] -= temp_av * temp_bv[3];
					}
				}
				__syncthreads();
			}
		}
	
		// 	sd2_9
		if (flag_d2_9 >= 0)
		{
			// 
			double* tmp_dev_d2_t2_9 = df_dev_d2_t2_all + size_max_dim_d2_t2 * flag_d2_9;//const_list_d2_flags_offset[local_offset];
			double* tmp_dev_d2_v2_9 = df_dev_d2_v2_all + size_max_dim_d2_v2 * flag_d2_9;//const_list_d2_flags_offset[local_offset];

			internal_upperbound = 0;
			#pragma unroll 1
			for (int l = 0; l < base_size_p7b; l+= FUSION_SIZE_INT_UNIT)
			{
				// Part: Generalized Contraction Index (p7b)
				// internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
				internal_offset = (l + FUSION_SIZE_INT_UNIT) - base_size_p7b;
				if (internal_offset > 0) internal_upperbound = internal_offset;
		
				// Load Input Tensor to Shared Memory: 16:16
				// # of size_internal Indices: 1
				if (idx_p6 < rng_h1 && idx_h1 < rng_h3 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
				for (int ll = 0; ll < rng_p6; ll++)
				{
					sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = tmp_dev_d2_t2_9[(str_blk_idx_p6 + ll + (str_blk_idx_h1 + idx_p6 + (str_blk_idx_h3 + idx_h1) * base_size_h1b) * base_size_p6b) * base_size_p7b + (threadIdx.x + l)];
				}
		
				// Load Input Tensor to Shared Memory
				if (idx_p6 < rng_h2 && idx_h1 < rng_p4 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
				for (int ll = 0; ll < rng_p5; ll++)
				{
					sm_b[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = tmp_dev_d2_v2_9[(str_blk_idx_h2 + idx_p6 + (str_blk_idx_p5 + ll + (str_blk_idx_p4 + idx_h1) * base_size_p5b) * base_size_h2b) * base_size_p7b + (threadIdx.x + l)];
				}
				__syncthreads();
		
				// Cross-Product: 16
				// Part: Generalized Threads
				for (int ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++)
				{
					temp_bv[0] = sm_a[ll][idx_h1 + (idx_h3) * FUSION_SIZE_SLICE_2_H1 + 0];
					temp_bv[1] = sm_a[ll][idx_h1 + (idx_h3) * FUSION_SIZE_SLICE_2_H1 + 16];
					temp_bv[2] = sm_a[ll][idx_h1 + (idx_h3) * FUSION_SIZE_SLICE_2_H1 + 32];
					temp_bv[3] = sm_a[ll][idx_h1 + (idx_h3) * FUSION_SIZE_SLICE_2_H1 + 48];
		
					for (int xx = 0 ; xx < 4; xx++)
					{
						temp_av = sm_b[ll][idx_h2 + (idx_p6) * FUSION_SIZE_SLICE_2_H2 + (xx * 16)];
		
						reg_tile[0][xx] += temp_av * temp_bv[0];
						reg_tile[1][xx] += temp_av * temp_bv[1];
						reg_tile[2][xx] += temp_av * temp_bv[2];
						reg_tile[3][xx] += temp_av * temp_bv[3];
					}
				}
				__syncthreads();
			}
		}
	}
	
	// 
	//  Register Rranspose (top - bottom)
	//
	{
		if (threadIdx.y < 4)						// 0, 1, 2, 3
		{
			// sm_a[16][64] <-- (4 x 16) x (4 x 4) = (16 x 64)		  'y''x'
			sm_a[0 + threadIdx.y * 4][threadIdx.x] 			= reg_tile[0][0];
			sm_a[1 + threadIdx.y * 4][threadIdx.x] 			= reg_tile[1][0];
			sm_a[2 + threadIdx.y * 4][threadIdx.x] 			= reg_tile[2][0];
			sm_a[3 + threadIdx.y * 4][threadIdx.x] 			= reg_tile[3][0];
			
			sm_a[0 + threadIdx.y * 4][threadIdx.x + 16] 	= reg_tile[0][1];
			sm_a[1 + threadIdx.y * 4][threadIdx.x + 16] 	= reg_tile[1][1];
			sm_a[2 + threadIdx.y * 4][threadIdx.x + 16] 	= reg_tile[2][1];
			sm_a[3 + threadIdx.y * 4][threadIdx.x + 16] 	= reg_tile[3][1];
					
			sm_a[0 + threadIdx.y * 4][threadIdx.x + 32] 	= reg_tile[0][2];
			sm_a[1 + threadIdx.y * 4][threadIdx.x + 32] 	= reg_tile[1][2];
			sm_a[2 + threadIdx.y * 4][threadIdx.x + 32] 	= reg_tile[2][2];
			sm_a[3 + threadIdx.y * 4][threadIdx.x + 32] 	= reg_tile[3][2];
			
			sm_a[0 + threadIdx.y * 4][threadIdx.x + 48] 	= reg_tile[0][3];
			sm_a[1 + threadIdx.y * 4][threadIdx.x + 48] 	= reg_tile[1][3];
			sm_a[2 + threadIdx.y * 4][threadIdx.x + 48] 	= reg_tile[2][3];
			sm_a[3 + threadIdx.y * 4][threadIdx.x + 48] 	= reg_tile[3][3];
		}

		if (threadIdx.y >= 4 && threadIdx.y < 8)	// 4, 5, 6, 7
		{
			sm_b[0 + (threadIdx.y - 4) * 4][threadIdx.x] 		= reg_tile[0][0];
			sm_b[1 + (threadIdx.y - 4) * 4][threadIdx.x] 		= reg_tile[1][0];
			sm_b[2 + (threadIdx.y - 4) * 4][threadIdx.x] 		= reg_tile[2][0];
			sm_b[3 + (threadIdx.y - 4) * 4][threadIdx.x] 		= reg_tile[3][0];

			sm_b[0 + (threadIdx.y - 4) * 4][threadIdx.x + 16] 	= reg_tile[0][1];
			sm_b[1 + (threadIdx.y - 4) * 4][threadIdx.x + 16] 	= reg_tile[1][1];
			sm_b[2 + (threadIdx.y - 4) * 4][threadIdx.x + 16] 	= reg_tile[2][1];
			sm_b[3 + (threadIdx.y - 4) * 4][threadIdx.x + 16] 	= reg_tile[3][1];

			sm_b[0 + (threadIdx.y - 4) * 4][threadIdx.x + 32] 	= reg_tile[0][2];
			sm_b[1 + (threadIdx.y - 4) * 4][threadIdx.x + 32] 	= reg_tile[1][2];
			sm_b[2 + (threadIdx.y - 4) * 4][threadIdx.x + 32] 	= reg_tile[2][2];
			sm_b[3 + (threadIdx.y - 4) * 4][threadIdx.x + 32] 	= reg_tile[3][2];

			sm_b[0 + (threadIdx.y - 4) * 4][threadIdx.x + 48] 	= reg_tile[0][3];
			sm_b[1 + (threadIdx.y - 4) * 4][threadIdx.x + 48] 	= reg_tile[1][3];
			sm_b[2 + (threadIdx.y - 4) * 4][threadIdx.x + 48] 	= reg_tile[2][3];
			sm_b[3 + (threadIdx.y - 4) * 4][threadIdx.x + 48] 	= reg_tile[3][3];
		}
		__syncthreads();

		if (threadIdx.y < 4)						// 0, 1, 2, 3
		{
			reg_tile[0][0] = sm_a[threadIdx.y + 0][(threadIdx.x)];
			reg_tile[1][0] = sm_a[threadIdx.y + 4][(threadIdx.x)];
			reg_tile[2][0] = sm_a[threadIdx.y + 8][(threadIdx.x)];
			reg_tile[3][0] = sm_a[threadIdx.y + 12][(threadIdx.x)];

			reg_tile[0][1] = sm_a[threadIdx.y + 0][(threadIdx.x) + 16];
			reg_tile[1][1] = sm_a[threadIdx.y + 4][(threadIdx.x) + 16];
			reg_tile[2][1] = sm_a[threadIdx.y + 8][(threadIdx.x) + 16];
			reg_tile[3][1] = sm_a[threadIdx.y + 12][(threadIdx.x) + 16];
			
			reg_tile[0][2] = sm_a[threadIdx.y + 0][(threadIdx.x) + 32];
			reg_tile[1][2] = sm_a[threadIdx.y + 4][(threadIdx.x) + 32];
			reg_tile[2][2] = sm_a[threadIdx.y + 8][(threadIdx.x) + 32];
			reg_tile[3][2] = sm_a[threadIdx.y + 12][(threadIdx.x) + 32];
			
			reg_tile[0][3] = sm_a[threadIdx.y + 0][(threadIdx.x) + 48];
			reg_tile[1][3] = sm_a[threadIdx.y + 4][(threadIdx.x) + 48];
			reg_tile[2][3] = sm_a[threadIdx.y + 8][(threadIdx.x) + 48];
			reg_tile[3][3] = sm_a[threadIdx.y + 12][(threadIdx.x) + 48];
		}

		if (threadIdx.y >= 4 && threadIdx.y < 8)	// 4, 5, 6, 7
		{
			reg_tile[0][0] = sm_b[(threadIdx.y - 4) + 0][(threadIdx.x)];
			reg_tile[1][0] = sm_b[(threadIdx.y - 4) + 4][(threadIdx.x)];
			reg_tile[2][0] = sm_b[(threadIdx.y - 4) + 8][(threadIdx.x)];
			reg_tile[3][0] = sm_b[(threadIdx.y - 4) + 12][(threadIdx.x)];

			reg_tile[0][1] = sm_b[(threadIdx.y - 4) + 0][(threadIdx.x) + 16];
			reg_tile[1][1] = sm_b[(threadIdx.y - 4) + 4][(threadIdx.x) + 16];
			reg_tile[2][1] = sm_b[(threadIdx.y - 4) + 8][(threadIdx.x) + 16];
			reg_tile[3][1] = sm_b[(threadIdx.y - 4) + 12][(threadIdx.x) + 16];
			
			reg_tile[0][2] = sm_b[(threadIdx.y - 4) + 0][(threadIdx.x) + 32];
			reg_tile[1][2] = sm_b[(threadIdx.y - 4) + 4][(threadIdx.x) + 32];
			reg_tile[2][2] = sm_b[(threadIdx.y - 4) + 8][(threadIdx.x) + 32];
			reg_tile[3][2] = sm_b[(threadIdx.y - 4) + 12][(threadIdx.x) + 32];
			
			reg_tile[0][3] = sm_b[(threadIdx.y - 4) + 0][(threadIdx.x) + 48];	
			reg_tile[1][3] = sm_b[(threadIdx.y - 4) + 4][(threadIdx.x) + 48];
			reg_tile[2][3] = sm_b[(threadIdx.y - 4) + 8][(threadIdx.x) + 48];
			reg_tile[3][3] = sm_b[(threadIdx.y - 4) + 12][(threadIdx.x) + 48];
		}
		__syncthreads();

		if (threadIdx.y >= 8 && threadIdx.y < 12)	// 8, 9, 10, 11
		{
			sm_a[0 + (threadIdx.y - 8) * 4][threadIdx.x] = reg_tile[0][0];
			sm_a[1 + (threadIdx.y - 8) * 4][threadIdx.x] = reg_tile[1][0];
			sm_a[2 + (threadIdx.y - 8) * 4][threadIdx.x] = reg_tile[2][0];
			sm_a[3 + (threadIdx.y - 8) * 4][threadIdx.x] = reg_tile[3][0];

			sm_a[0 + (threadIdx.y - 8) * 4][threadIdx.x + 16] = reg_tile[0][1];
			sm_a[1 + (threadIdx.y - 8) * 4][threadIdx.x + 16] = reg_tile[1][1];
			sm_a[2 + (threadIdx.y - 8) * 4][threadIdx.x + 16] = reg_tile[2][1];
			sm_a[3 + (threadIdx.y - 8) * 4][threadIdx.x + 16] = reg_tile[3][1];

			sm_a[0 + (threadIdx.y - 8) * 4][threadIdx.x + 32] = reg_tile[0][2];
			sm_a[1 + (threadIdx.y - 8) * 4][threadIdx.x + 32] = reg_tile[1][2];
			sm_a[2 + (threadIdx.y - 8) * 4][threadIdx.x + 32] = reg_tile[2][2];
			sm_a[3 + (threadIdx.y - 8) * 4][threadIdx.x + 32] = reg_tile[3][2];

			sm_a[0 + (threadIdx.y - 8) * 4][threadIdx.x + 48] = reg_tile[0][3];
			sm_a[1 + (threadIdx.y - 8) * 4][threadIdx.x + 48] = reg_tile[1][3];
			sm_a[2 + (threadIdx.y - 8) * 4][threadIdx.x + 48] = reg_tile[2][3];
			sm_a[3 + (threadIdx.y - 8) * 4][threadIdx.x + 48] = reg_tile[3][3];
		}

		if (threadIdx.y >= 12)	// 12, 13, 14, 15
		{
			sm_b[0 + (threadIdx.y - 12) * 4][threadIdx.x] = reg_tile[0][0];
			sm_b[1 + (threadIdx.y - 12) * 4][threadIdx.x] = reg_tile[1][0];
			sm_b[2 + (threadIdx.y - 12) * 4][threadIdx.x] = reg_tile[2][0];
			sm_b[3 + (threadIdx.y - 12) * 4][threadIdx.x] = reg_tile[3][0];
			
			sm_b[0 + (threadIdx.y - 12) * 4][threadIdx.x + 16] = reg_tile[0][1];
			sm_b[1 + (threadIdx.y - 12) * 4][threadIdx.x + 16] = reg_tile[1][1];
			sm_b[2 + (threadIdx.y - 12) * 4][threadIdx.x + 16] = reg_tile[2][1];
			sm_b[3 + (threadIdx.y - 12) * 4][threadIdx.x + 16] = reg_tile[3][1];
			
			sm_b[0 + (threadIdx.y - 12) * 4][threadIdx.x + 32] = reg_tile[0][2];
			sm_b[1 + (threadIdx.y - 12) * 4][threadIdx.x + 32] = reg_tile[1][2];
			sm_b[2 + (threadIdx.y - 12) * 4][threadIdx.x + 32] = reg_tile[2][2];
			sm_b[3 + (threadIdx.y - 12) * 4][threadIdx.x + 32] = reg_tile[3][2];
			
			sm_b[0 + (threadIdx.y - 12) * 4][threadIdx.x + 48] = reg_tile[0][3];
			sm_b[1 + (threadIdx.y - 12) * 4][threadIdx.x + 48] = reg_tile[1][3];
			sm_b[2 + (threadIdx.y - 12) * 4][threadIdx.x + 48] = reg_tile[2][3];
			sm_b[3 + (threadIdx.y - 12) * 4][threadIdx.x + 48] = reg_tile[3][3];
		}
		__syncthreads();

		if (threadIdx.y >= 8 && threadIdx.y < 12)	// 8, 9, 10, 11
		{
			reg_tile[0][0] = sm_a[(threadIdx.y - 8) + 0][(threadIdx.x)];
			reg_tile[1][0] = sm_a[(threadIdx.y - 8) + 4][(threadIdx.x)];
			reg_tile[2][0] = sm_a[(threadIdx.y - 8) + 8][(threadIdx.x)];
			reg_tile[3][0] = sm_a[(threadIdx.y - 8) + 12][(threadIdx.x)];

			reg_tile[0][1] = sm_a[(threadIdx.y - 8) + 0][(threadIdx.x) + 16];
			reg_tile[1][1] = sm_a[(threadIdx.y - 8) + 4][(threadIdx.x) + 16];
			reg_tile[2][1] = sm_a[(threadIdx.y - 8) + 8][(threadIdx.x) + 16];
			reg_tile[3][1] = sm_a[(threadIdx.y - 8) + 12][(threadIdx.x) + 16];

			reg_tile[0][2] = sm_a[(threadIdx.y - 8) + 0][(threadIdx.x) + 32];
			reg_tile[1][2] = sm_a[(threadIdx.y - 8) + 4][(threadIdx.x) + 32];		
			reg_tile[2][2] = sm_a[(threadIdx.y - 8) + 8][(threadIdx.x) + 32];
			reg_tile[3][2] = sm_a[(threadIdx.y - 8) + 12][(threadIdx.x) + 32];

			reg_tile[0][3] = sm_a[(threadIdx.y - 8) + 0][(threadIdx.x) + 48];
			reg_tile[1][3] = sm_a[(threadIdx.y - 8) + 4][(threadIdx.x) + 48];
			reg_tile[2][3] = sm_a[(threadIdx.y - 8) + 8][(threadIdx.x) + 48];
			reg_tile[3][3] = sm_a[(threadIdx.y - 8) + 12][(threadIdx.x) + 48];
		}

		if (threadIdx.y >= 12)	// 12, 13, 14, 15
		{
			reg_tile[0][0] = sm_b[(threadIdx.y - 12) + 0][(threadIdx.x)];
			reg_tile[1][0] = sm_b[(threadIdx.y - 12) + 4][(threadIdx.x)];
			reg_tile[2][0] = sm_b[(threadIdx.y - 12) + 8][(threadIdx.x)];
			reg_tile[3][0] = sm_b[(threadIdx.y - 12) + 12][(threadIdx.x)];

			reg_tile[0][1] = sm_b[(threadIdx.y - 12) + 0][(threadIdx.x) + 16];
			reg_tile[1][1] = sm_b[(threadIdx.y - 12) + 4][(threadIdx.x) + 16];
			reg_tile[2][1] = sm_b[(threadIdx.y - 12) + 8][(threadIdx.x) + 16];
			reg_tile[3][1] = sm_b[(threadIdx.y - 12) + 12][(threadIdx.x) + 16];

			reg_tile[0][2] = sm_b[(threadIdx.y - 12) + 0][(threadIdx.x) + 32];
			reg_tile[1][2] = sm_b[(threadIdx.y - 12) + 4][(threadIdx.x) + 32];
			reg_tile[2][2] = sm_b[(threadIdx.y - 12) + 8][(threadIdx.x) + 32];
			reg_tile[3][2] = sm_b[(threadIdx.y - 12) + 12][(threadIdx.x) + 32];

			reg_tile[0][3] = sm_b[(threadIdx.y - 12) + 0][(threadIdx.x) + 48];
			reg_tile[1][3] = sm_b[(threadIdx.y - 12) + 4][(threadIdx.x) + 48];
			reg_tile[2][3] = sm_b[(threadIdx.y - 12) + 8][(threadIdx.x) + 48];
			reg_tile[3][3] = sm_b[(threadIdx.y - 12) + 12][(threadIdx.x) + 48];
		}
		__syncthreads();
	}
	//
	// 	End of Register Transpose
	// 

	// 
	// 	based on "noab"	
	//  d1-bottom: sd1_4, 5 , 6 , 7 , 8 and 9.
	// 
	#pragma unroll 1
	for (int iter_noab = 0; iter_noab < size_noab; iter_noab++)
	{
		// 	flags
		int flag_d1_4 = const_df_d1_exec[3 + (iter_noab) * NUM_D1_EQUATIONS];
		int flag_d1_5 = const_df_d1_exec[4 + (iter_noab) * NUM_D1_EQUATIONS];
		int flag_d1_6 = const_df_d1_exec[5 + (iter_noab) * NUM_D1_EQUATIONS];
		int flag_d1_7 = const_df_d1_exec[6 + (iter_noab) * NUM_D1_EQUATIONS];
		int flag_d1_8 = const_df_d1_exec[7 + (iter_noab) * NUM_D1_EQUATIONS];
		int flag_d1_9 = const_df_d1_exec[8 + (iter_noab) * NUM_D1_EQUATIONS];

		base_size_h1b = const_df_d1_size[0 + (iter_noab) * NUM_D1_INDEX];
		base_size_h2b = const_df_d1_size[1 + (iter_noab) * NUM_D1_INDEX];
		base_size_h3b = const_df_d1_size[2 + (iter_noab) * NUM_D1_INDEX];
		base_size_h7b = const_df_d1_size[3 + (iter_noab) * NUM_D1_INDEX];
		base_size_p4b = const_df_d1_size[4 + (iter_noab) * NUM_D1_INDEX];
		base_size_p5b = const_df_d1_size[5 + (iter_noab) * NUM_D1_INDEX];
		base_size_p6b = const_df_d1_size[6 + (iter_noab) * NUM_D1_INDEX];

		// 
		num_blks_h1b = CEIL(base_size_h1b, FUSION_SIZE_SLICE_1_H1);
		num_blks_h2b = CEIL(base_size_h2b, FUSION_SIZE_SLICE_1_H2);
		num_blks_h3b = CEIL(base_size_h3b, FUSION_SIZE_SLICE_1_H3);
		num_blks_p4b = CEIL(base_size_p4b, FUSION_SIZE_SLICE_1_P4);
		num_blks_p5b = CEIL(base_size_p5b, FUSION_SIZE_SLICE_1_P5);
		num_blks_p6b = CEIL(base_size_p6b, FUSION_SIZE_SLICE_1_P6);

		// 	(2) blk_idx_h/p*b
		blk_idx_p4b     = blockIdx.x / (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b * num_blks_p5b);
		tmp_blkIdx      = blockIdx.x % (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b * num_blks_p5b);
		blk_idx_p5b     = (tmp_blkIdx) / (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b);
		tmp_blkIdx  	= (tmp_blkIdx) % (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b);
		blk_idx_p6b     = (tmp_blkIdx) / (num_blks_h3b * num_blks_h2b * num_blks_h1b);
		tmp_blkIdx 		= (tmp_blkIdx) % (num_blks_h3b * num_blks_h2b * num_blks_h1b);
		blk_idx_h1b     = (tmp_blkIdx) / (num_blks_h3b * num_blks_h2b);
		tmp_blkIdx 		= (tmp_blkIdx) % (num_blks_h3b * num_blks_h2b);
		blk_idx_h2b 	= (tmp_blkIdx) / (num_blks_h3b);
		blk_idx_h3b		= blockIdx.x % (num_blks_h3b);

		// 	(3) str_blk_idx_h/p*
		str_blk_idx_h3 	= blk_idx_h3b * FUSION_SIZE_SLICE_1_H3;
		str_blk_idx_h2 	= blk_idx_h2b * FUSION_SIZE_SLICE_1_H2;
		str_blk_idx_h1 	= blk_idx_h1b * FUSION_SIZE_SLICE_1_H1;
		str_blk_idx_p6 	= blk_idx_p6b * FUSION_SIZE_SLICE_1_P6;
		str_blk_idx_p5 	= blk_idx_p5b * FUSION_SIZE_SLICE_1_P5;
		str_blk_idx_p4 	= blk_idx_p4b * FUSION_SIZE_SLICE_1_P4;

		// 	(4) rng_h/p*
		if ((base_size_h3b - (str_blk_idx_h3)) >= FUSION_SIZE_SLICE_1_H3)
			rng_h3 = FUSION_SIZE_SLICE_1_H3;
		else
			rng_h3 = base_size_h3b % FUSION_SIZE_SLICE_1_H3;
		
		if ((base_size_h2b - (str_blk_idx_h2)) >= FUSION_SIZE_SLICE_1_H2)
			rng_h2 = FUSION_SIZE_SLICE_1_H2;
		else
			rng_h2 = base_size_h2b % FUSION_SIZE_SLICE_1_H2;

		if ((base_size_h1b - (str_blk_idx_h1)) >= FUSION_SIZE_SLICE_1_H1)
			rng_h1 = FUSION_SIZE_SLICE_1_H1;
		else
			rng_h1 = base_size_h1b % FUSION_SIZE_SLICE_1_H1;
		
		if ((base_size_p6b - (str_blk_idx_p6)) >= FUSION_SIZE_SLICE_1_P6)
			rng_p6 = FUSION_SIZE_SLICE_1_P6;
		else
			rng_p6 = base_size_p6b % FUSION_SIZE_SLICE_1_P6;

		if ((base_size_p5b - (str_blk_idx_p5)) >= FUSION_SIZE_SLICE_1_P5)
			rng_p5 = FUSION_SIZE_SLICE_1_P5;
		else
			rng_p5 = base_size_p5b % FUSION_SIZE_SLICE_1_P5;

		if ((base_size_p4b - (str_blk_idx_p4)) >= FUSION_SIZE_SLICE_1_P4)
			rng_p4 = FUSION_SIZE_SLICE_1_P4;
		else
			rng_p4 = base_size_p4b % FUSION_SIZE_SLICE_1_P4;

		// 	sd1_4
		if (flag_d1_4 >= 0)
		{
			// 
			double* tmp_dev_d1_t2_4 = df_dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_4;
			double* tmp_dev_d1_v2_4 = df_dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_4;

			internal_upperbound = 0;
			#pragma unroll 1
			for (int l = 0; l < base_size_h7b; l+= FUSION_SIZE_INT_UNIT)
			{
				// Part: Generalized Contraction Index (p7b)
				internal_offset = (l + FUSION_SIZE_INT_UNIT) - base_size_h7b;
				if (internal_offset > 0) internal_upperbound = internal_offset;

				// Load Input Tensor to Shared Memory: 16:16
				// # of size_internal Indices: 1
				if (idx_p6 < rng_p6 && idx_h1 < rng_h1 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
				for (int ll = 0; ll < rng_p5; ll++)
				{
					sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = tmp_dev_d1_t2_4[(str_blk_idx_p5 + ll + (str_blk_idx_p6 + idx_p6 + (str_blk_idx_h1 + idx_h1) * base_size_p6b) * base_size_p5b) * base_size_h7b + (threadIdx.x + l)];					
				}

				// Load Input Tensor to Shared Memory
				if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
				for (int ll = 0; ll < rng_p4; ll++)
				{
					sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = tmp_dev_d1_v2_4[(str_blk_idx_h3 + idx_h3 + (str_blk_idx_h2 + idx_h2 + (str_blk_idx_p4 + ll + (threadIdx.y + l) * base_size_p4b) * base_size_h2b) * base_size_h3b)];
				}
				__syncthreads();

				// Cross-Product: -1
				// Part: Generalized Threads
				for (int ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++)
				{
					temp_bv[0] = sm_b[ll][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_1_H3 + 0];
					temp_bv[1] = sm_b[ll][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_1_H3 + 16];
					temp_bv[2] = sm_b[ll][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_1_H3 + 32];
					temp_bv[3] = sm_b[ll][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_1_H3 + 48];

					for (int xx = 0 ; xx < 4; xx++)
					{
						temp_av = sm_a[ll][idx_p6 + (idx_h1) * FUSION_SIZE_SLICE_1_P6 + (xx * 16)];

						reg_tile[0][xx] -= temp_av * temp_bv[0];
						reg_tile[1][xx] -= temp_av * temp_bv[1];
						reg_tile[2][xx] -= temp_av * temp_bv[2];
						reg_tile[3][xx] -= temp_av * temp_bv[3];
					}
				}
				__syncthreads();
			}
		}

		// 	sd1_5
		if (flag_d1_5 >= 0)
		{
			// 
			double* tmp_dev_d1_t2_5 = df_dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_5;
			double* tmp_dev_d1_v2_5 = df_dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_5;

			internal_upperbound = 0;
			#pragma unroll 1
			for (int l = 0; l < base_size_h7b; l+= FUSION_SIZE_INT_UNIT)
			{
				// Part: Generalized Contraction Index (p7b)
				// internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
				internal_offset = (l + FUSION_SIZE_INT_UNIT) - base_size_h7b;
				if (internal_offset > 0) internal_upperbound = internal_offset;

				// Load Input Tensor to Shared Memory: 16:16
				// # of Internal Indices: 1
				if (idx_p6 < rng_p6 && idx_h1 < rng_h2 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
				for (int ll = 0; ll < rng_p5; ll++)
				{
					sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = tmp_dev_d1_t2_5[(str_blk_idx_p5 + ll + (str_blk_idx_p6 + idx_p6 + (str_blk_idx_h2 + idx_h1) * base_size_p6b) * base_size_p5b) * base_size_h7b + (threadIdx.x + l)]; 
				}

				// Load Input Tensor to Shared Memory
				if (idx_h3 < rng_h3 && idx_h2 < rng_h1 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
				for (int ll = 0; ll < rng_p4; ll++)
				{
					sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = tmp_dev_d1_v2_5[(str_blk_idx_h3 + idx_h3 + (str_blk_idx_h1 + idx_h2 + (str_blk_idx_p4 + ll + (threadIdx.y + l) * base_size_p4b) * base_size_h1b) * base_size_h3b)];
				}
				__syncthreads();

				// Cross-Product: -1
				// Part: Generalized Threads
				for (int ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++)
				{
					temp_bv[0] = sm_b[ll][idx_h3 + (idx_h1) * FUSION_SIZE_SLICE_1_H3 + 0];
					temp_bv[1] = sm_b[ll][idx_h3 + (idx_h1) * FUSION_SIZE_SLICE_1_H3 + 16];
					temp_bv[2] = sm_b[ll][idx_h3 + (idx_h1) * FUSION_SIZE_SLICE_1_H3 + 32];
					temp_bv[3] = sm_b[ll][idx_h3 + (idx_h1) * FUSION_SIZE_SLICE_1_H3 + 48];

					for (int xx = 0 ; xx < 4; xx++)
					{
						temp_av = sm_a[ll][idx_p6 + (idx_h2) * FUSION_SIZE_SLICE_1_P6 + (xx * 16)];

						reg_tile[0][xx] += temp_av * temp_bv[0];
						reg_tile[1][xx] += temp_av * temp_bv[1];
						reg_tile[2][xx] += temp_av * temp_bv[2];
						reg_tile[3][xx] += temp_av * temp_bv[3];
					}
				}
				__syncthreads();
			}
		}

		// 	sd1_6
		if (flag_d1_6 >= 0)
		{
			// 
			double* tmp_dev_d1_t2_6 = df_dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_6;
			double* tmp_dev_d1_v2_6 = df_dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_6;

			internal_upperbound = 0;
			#pragma unroll 1
			for (int l = 0; l < base_size_h7b; l+= FUSION_SIZE_INT_UNIT)
			{
				// Part: Generalized Contraction Index (p7b)
				// internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
				internal_offset = (l + FUSION_SIZE_INT_UNIT) - base_size_h7b;
				if (internal_offset > 0) internal_upperbound = internal_offset;

				// Load Input Tensor to Shared Memory: 16:16
				// # of Internal Indices: 1 //63, 21
				if (idx_p6 < rng_p6 && idx_h1 < rng_h3 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
				for (int ll = 0; ll < rng_p5; ll++)
				{
					sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = tmp_dev_d1_t2_6[(str_blk_idx_p5 + ll + (str_blk_idx_p6 + idx_p6 + (str_blk_idx_h3 + idx_h1) * base_size_p6b) * base_size_p5b) * base_size_h7b + (threadIdx.x + l)];
				}

				// Load Input Tensor to Shared Memory
				if (idx_h3 < rng_h2 && idx_h2 < rng_h1 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
				for (int ll = 0; ll < rng_p4; ll++)
				{
					sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = tmp_dev_d1_v2_6[(str_blk_idx_h2 + idx_h3 + (str_blk_idx_h1 + idx_h2 + (str_blk_idx_p4 + ll + (threadIdx.y + l) * base_size_p4b) * base_size_h1b) * base_size_h2b)];
																								
				}
				__syncthreads();

				// Cross-Product: -1
				// Part: Generalized Threads
				for (int ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++)
				{
					temp_bv[0] = sm_b[ll][idx_h2 + (idx_h1) * FUSION_SIZE_SLICE_1_H2 + 0];
					temp_bv[1] = sm_b[ll][idx_h2 + (idx_h1) * FUSION_SIZE_SLICE_1_H2 + 16];
					temp_bv[2] = sm_b[ll][idx_h2 + (idx_h1) * FUSION_SIZE_SLICE_1_H2 + 32];
					temp_bv[3] = sm_b[ll][idx_h2 + (idx_h1) * FUSION_SIZE_SLICE_1_H2 + 48];

					for (int xx = 0 ; xx < 4; xx++)
					{
						temp_av = sm_a[ll][idx_p6 + (idx_h3) * FUSION_SIZE_SLICE_1_P6 + (xx * 16)];

						reg_tile[0][xx] -= temp_av * temp_bv[0];
						reg_tile[1][xx] -= temp_av * temp_bv[1];
						reg_tile[2][xx] -= temp_av * temp_bv[2];
						reg_tile[3][xx] -= temp_av * temp_bv[3];
					}
				}
				__syncthreads();
			}
		}

		// 	sd1_7
		if (flag_d1_7 >= 0)
		{
			// 
			double* tmp_dev_d1_t2_7 = df_dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_7;
			double* tmp_dev_d1_v2_7 = df_dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_7;

			internal_upperbound = 0;
			#pragma unroll 1
			for (int l = 0; l < base_size_h7b; l+= FUSION_SIZE_INT_UNIT)
			{
				// Part: Generalized Contraction Index (p7b)
				internal_offset = (l + FUSION_SIZE_INT_UNIT) - base_size_h7b;
				if (internal_offset > 0) internal_upperbound = internal_offset;

				// Load Input Tensor to Shared Memory: 16:16
				// # of Internal Indices: 1
				if (idx_p6 < rng_p6 && idx_h1 < rng_h1 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
				for (int ll = 0; ll < rng_p4; ll++)
				{
					sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = tmp_dev_d1_t2_7[(str_blk_idx_p4 + ll + (str_blk_idx_p6 + idx_p6 + (str_blk_idx_h1 + idx_h1) * base_size_p6b) * base_size_p4b) * base_size_h7b + (threadIdx.x + l)];
				}

				// Load Input Tensor to Shared Memory
				if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
				for (int ll = 0; ll < rng_p5; ll++)
				{
					sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = tmp_dev_d1_v2_7[(str_blk_idx_h3 + idx_h3 + (str_blk_idx_h2 + idx_h2 + (str_blk_idx_p5 + ll + (threadIdx.y + l) * base_size_p5b) * base_size_h2b) * base_size_h3b)];
				}
				__syncthreads();

				// Cross-Product: -1
				// Part: Generalized Threads
				for (int ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++)
				{
					temp_bv[0] = sm_a[ll][idx_p6 + (idx_h1) * FUSION_SIZE_SLICE_1_P6 + 0];
					temp_bv[1] = sm_a[ll][idx_p6 + (idx_h1) * FUSION_SIZE_SLICE_1_P6 + 16];
					temp_bv[2] = sm_a[ll][idx_p6 + (idx_h1) * FUSION_SIZE_SLICE_1_P6 + 32];
					temp_bv[3] = sm_a[ll][idx_p6 + (idx_h1) * FUSION_SIZE_SLICE_1_P6 + 48];

					for (int xx = 0 ; xx < 4; xx++)
					{
						temp_av = sm_b[ll][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_1_H3 + (xx * 16)];

						reg_tile[0][xx] += temp_av * temp_bv[0];
						reg_tile[1][xx] += temp_av * temp_bv[1];
						reg_tile[2][xx] += temp_av * temp_bv[2];
						reg_tile[3][xx] += temp_av * temp_bv[3];
					}
				}
				__syncthreads();
			}
		}
		
		// 	sd1_8
		if (flag_d1_8 >= 0)
		{
			// 
			double* tmp_dev_d1_t2_8 = df_dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_8;
			double* tmp_dev_d1_v2_8 = df_dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_8;

			internal_upperbound = 0;
			#pragma unroll 1
			for (int l = 0; l < base_size_h7b; l+= FUSION_SIZE_INT_UNIT)
			{
				// Part: Generalized Contraction Index (p7b)
				internal_offset = (l + FUSION_SIZE_INT_UNIT) - base_size_h7b;
				if (internal_offset > 0) internal_upperbound = internal_offset;

				// Load Input Tensor to Shared Memory: 16:16
				// # of Internal Indices: 1
				if (idx_p6 < rng_p6 && idx_h1 < rng_h2 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
				for (int ll = 0; ll < rng_p4; ll++)
				{
					sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = tmp_dev_d1_t2_8[(str_blk_idx_p4 + ll + (str_blk_idx_p6 + idx_p6 + (str_blk_idx_h2 + idx_h1) * base_size_p6b) * base_size_p4b) * base_size_h7b + (threadIdx.x + l)];
				}

				// Load Input Tensor to Shared Memory
				if (idx_h3 < rng_h3 && idx_h2 < rng_h1 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
				for (int ll = 0; ll < rng_p5; ll++)
				{
					sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = tmp_dev_d1_v2_8[(str_blk_idx_h3 + idx_h3 + (str_blk_idx_h1 + idx_h2 + (str_blk_idx_p5 + ll + (threadIdx.y + l) * base_size_p5b) * base_size_h1b) * base_size_h3b)];
				}
				__syncthreads();

				// Cross-Product: -1
				// Part: Generalized Threads
				for (int ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++)
				{
					temp_bv[0] = sm_a[ll][idx_p6 + (idx_h2) * FUSION_SIZE_SLICE_1_P6 + 0];
					temp_bv[1] = sm_a[ll][idx_p6 + (idx_h2) * FUSION_SIZE_SLICE_1_P6 + 16];
					temp_bv[2] = sm_a[ll][idx_p6 + (idx_h2) * FUSION_SIZE_SLICE_1_P6 + 32];
					temp_bv[3] = sm_a[ll][idx_p6 + (idx_h2) * FUSION_SIZE_SLICE_1_P6 + 48];

					for (int xx = 0 ; xx < 4; xx++)
					{
						temp_av = sm_b[ll][idx_h3 + (idx_h1) * FUSION_SIZE_SLICE_1_H3 + (xx * 16)];

						reg_tile[0][xx] -= temp_av * temp_bv[0];
						reg_tile[1][xx] -= temp_av * temp_bv[1];
						reg_tile[2][xx] -= temp_av * temp_bv[2];
						reg_tile[3][xx] -= temp_av * temp_bv[3];
					}
				}
				__syncthreads();
			}
		}

		// 	sd1_9
		if (flag_d1_9 >= 0)
		{	
			// 
			double* tmp_dev_d1_t2_9 = df_dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_9;
			double* tmp_dev_d1_v2_9 = df_dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_9;

			internal_upperbound = 0;
			#pragma unroll 1
			for (int l = 0; l < base_size_h7b; l+= FUSION_SIZE_INT_UNIT)
			{
				// Part: Generalized Contraction Index (p7b)
				internal_offset = (l + FUSION_SIZE_INT_UNIT) - base_size_h7b;
				if (internal_offset > 0) internal_upperbound = internal_offset;

				// Load Input Tensor to Shared Memory: 16:16
				// # of Internal Indices: 1
				if (idx_p6 < rng_p6 && idx_h1 < rng_h3 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
				for (int ll = 0; ll < rng_p4; ll++)
				{
					sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = tmp_dev_d1_t2_9[(str_blk_idx_p4 + ll + (str_blk_idx_p6 + idx_p6 + (str_blk_idx_h3 + idx_h1) * base_size_p6b) * base_size_p4b) * base_size_h7b + (threadIdx.x + l)];
				}

				// Load Input Tensor to Shared Memory
				if (idx_h3 < rng_h2 && idx_h2 < rng_h1 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
				for (int ll = 0; ll < rng_p5; ll++)
				{
					sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = tmp_dev_d1_v2_9[(str_blk_idx_h2 + idx_h3 + (str_blk_idx_h1 + idx_h2 + (str_blk_idx_p5 + ll + (threadIdx.y + l) * base_size_p5b) * base_size_h1b) * base_size_h2b)];
				}
				__syncthreads();

				// Cross-Product: -1
				// Part: Generalized Threads
				for (int ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++)
				{
					temp_bv[0] = sm_a[ll][idx_p6 + (idx_h3) * FUSION_SIZE_SLICE_1_P6 + 0];
					temp_bv[1] = sm_a[ll][idx_p6 + (idx_h3) * FUSION_SIZE_SLICE_1_P6 + 16];
					temp_bv[2] = sm_a[ll][idx_p6 + (idx_h3) * FUSION_SIZE_SLICE_1_P6 + 32];
					temp_bv[3] = sm_a[ll][idx_p6 + (idx_h3) * FUSION_SIZE_SLICE_1_P6 + 48];

					for (int xx = 0 ; xx < 4; xx++)
					{
						temp_av = sm_b[ll][idx_h2 + (idx_h1) * FUSION_SIZE_SLICE_1_H2 + (xx * 16)];

						reg_tile[0][xx] += temp_av * temp_bv[0];
						reg_tile[1][xx] += temp_av * temp_bv[1];
						reg_tile[2][xx] += temp_av * temp_bv[2];
						reg_tile[3][xx] += temp_av * temp_bv[3];
					}
				}
				__syncthreads();
			}
		}
	}

	//  d2-bottom: sd2_1, 2, 3, 4, 5 and 6.
	#pragma unroll 1
	for (int iter_nvab = 0; iter_nvab < size_nvab; iter_nvab++)
	{
		//
		int flag_d2_1 = const_df_d2_exec[0 + (iter_nvab) * NUM_D2_EQUATIONS];
		int flag_d2_2 = const_df_d2_exec[1 + (iter_nvab) * NUM_D2_EQUATIONS];
		int flag_d2_3 = const_df_d2_exec[2 + (iter_nvab) * NUM_D2_EQUATIONS];
		int flag_d2_4 = const_df_d2_exec[3 + (iter_nvab) * NUM_D2_EQUATIONS];
		int flag_d2_5 = const_df_d2_exec[4 + (iter_nvab) * NUM_D2_EQUATIONS];
		int flag_d2_6 = const_df_d2_exec[5 + (iter_nvab) * NUM_D2_EQUATIONS];

		// 
		base_size_h1b = const_df_d2_size[0 + (iter_nvab) * NUM_D2_INDEX];
		base_size_h2b = const_df_d2_size[1 + (iter_nvab) * NUM_D2_INDEX];
		base_size_h3b = const_df_d2_size[2 + (iter_nvab) * NUM_D2_INDEX];
		base_size_p4b = const_df_d2_size[3 + (iter_nvab) * NUM_D2_INDEX];
		base_size_p5b = const_df_d2_size[4 + (iter_nvab) * NUM_D2_INDEX];
		base_size_p6b = const_df_d2_size[5 + (iter_nvab) * NUM_D2_INDEX];
		base_size_p7b = const_df_d2_size[6 + (iter_nvab) * NUM_D2_INDEX];

		// 
		num_blks_h1b = CEIL(base_size_h1b, FUSION_SIZE_SLICE_1_H1);
		num_blks_h2b = CEIL(base_size_h2b, FUSION_SIZE_SLICE_1_H2);
		num_blks_h3b = CEIL(base_size_h3b, FUSION_SIZE_SLICE_1_H3);
		num_blks_p4b = CEIL(base_size_p4b, FUSION_SIZE_SLICE_1_P4);
		num_blks_p5b = CEIL(base_size_p5b, FUSION_SIZE_SLICE_1_P5);
		num_blks_p6b = CEIL(base_size_p6b, FUSION_SIZE_SLICE_1_P6);

		// 	(2) blk_idx_h/p*b
		blk_idx_p4b     = blockIdx.x / (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b * num_blks_p5b);
		tmp_blkIdx      = blockIdx.x % (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b * num_blks_p5b);
		blk_idx_p5b     = (tmp_blkIdx) / (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b);
		tmp_blkIdx  	= (tmp_blkIdx) % (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b);
		blk_idx_p6b     = (tmp_blkIdx) / (num_blks_h3b * num_blks_h2b * num_blks_h1b);
		tmp_blkIdx 		= (tmp_blkIdx) % (num_blks_h3b * num_blks_h2b * num_blks_h1b);
		blk_idx_h1b     = (tmp_blkIdx) / (num_blks_h3b * num_blks_h2b);
		tmp_blkIdx 		= (tmp_blkIdx) % (num_blks_h3b * num_blks_h2b);
		blk_idx_h2b 	= (tmp_blkIdx) / (num_blks_h3b);
		blk_idx_h3b		= blockIdx.x % (num_blks_h3b);

		// 	(3) str_blk_idx_h/p*
		str_blk_idx_h3 	= blk_idx_h3b * FUSION_SIZE_SLICE_1_H3;
		str_blk_idx_h2 	= blk_idx_h2b * FUSION_SIZE_SLICE_1_H2;
		str_blk_idx_h1 	= blk_idx_h1b * FUSION_SIZE_SLICE_1_H1;
		str_blk_idx_p6 	= blk_idx_p6b * FUSION_SIZE_SLICE_1_P6;
		str_blk_idx_p5 	= blk_idx_p5b * FUSION_SIZE_SLICE_1_P5;
		str_blk_idx_p4 	= blk_idx_p4b * FUSION_SIZE_SLICE_1_P4;

		// 	(4) rng_h/p*
		if ((base_size_h3b - (str_blk_idx_h3)) >= FUSION_SIZE_SLICE_1_H3)
			rng_h3 = FUSION_SIZE_SLICE_1_H3;
		else
			rng_h3 = base_size_h3b % FUSION_SIZE_SLICE_1_H3;
		
		if ((base_size_h2b - (str_blk_idx_h2)) >= FUSION_SIZE_SLICE_1_H2)
			rng_h2 = FUSION_SIZE_SLICE_1_H2;
		else
			rng_h2 = base_size_h2b % FUSION_SIZE_SLICE_1_H2;

		if ((base_size_h1b - (str_blk_idx_h1)) >= FUSION_SIZE_SLICE_1_H1)
			rng_h1 = FUSION_SIZE_SLICE_1_H1;
		else
			rng_h1 = base_size_h1b % FUSION_SIZE_SLICE_1_H1;
		
		if ((base_size_p6b - (str_blk_idx_p6)) >= FUSION_SIZE_SLICE_1_P6)
			rng_p6 = FUSION_SIZE_SLICE_1_P6;
		else
			rng_p6 = base_size_p6b % FUSION_SIZE_SLICE_1_P6;

		if ((base_size_p5b - (str_blk_idx_p5)) >= FUSION_SIZE_SLICE_1_P5)
			rng_p5 = FUSION_SIZE_SLICE_1_P5;
		else
			rng_p5 = base_size_p5b % FUSION_SIZE_SLICE_1_P5;

		if ((base_size_p4b - (str_blk_idx_p4)) >= FUSION_SIZE_SLICE_1_P4)
			rng_p4 = FUSION_SIZE_SLICE_1_P4;
		else
			rng_p4 = base_size_p4b % FUSION_SIZE_SLICE_1_P4;

		//  sd2_1
		if (flag_d2_1 >= 0)
		{
			// 
			double* tmp_dev_d2_t2_1 = df_dev_d2_t2_all + size_max_dim_d2_t2 * flag_d2_1;
			double* tmp_dev_d2_v2_1 = df_dev_d2_v2_all + size_max_dim_d2_v2 * flag_d2_1;

			internal_upperbound = 0;
			#pragma unroll 1
			for (int l = 0; l < base_size_p7b; l+= FUSION_SIZE_INT_UNIT)
			{
				// Part: Generalized Contraction Index (p7b)
				// internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
				internal_offset = (l + FUSION_SIZE_INT_UNIT) - base_size_p7b;
				if (internal_offset > 0) internal_upperbound = internal_offset;

				// Load Input Tensor to Shared Memory: 16:16
				// # of size_internal Indices: 1
				if (idx_p6 < rng_h1 && idx_h1 < rng_h2 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
				for (int ll = 0; ll < rng_p4; ll++)
				{
					sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = tmp_dev_d2_t2_1[(str_blk_idx_p4 + ll + (str_blk_idx_h1 + idx_p6 + (str_blk_idx_h2 + idx_h1) * base_size_h1b) * base_size_p4b) * base_size_p7b + (threadIdx.x + l)];
				}

				// Load Input Tensor to Shared Memory
				if (idx_p6 < rng_h3 && idx_h1 < rng_p6 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
				for (int ll = 0; ll < rng_p5; ll++)
				{
					sm_b[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = tmp_dev_d2_v2_1[(str_blk_idx_h3 + idx_p6 + (str_blk_idx_p6 + idx_h1 + (str_blk_idx_p5 + ll) * base_size_p6b) * base_size_h3b) * base_size_p7b + (threadIdx.x + l)];
				}
				__syncthreads();

				// Cross-Product: 16
				// Part: Generalized Threads
				for (int ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++)
				{   
					temp_bv[0] = sm_a[ll][idx_h1 + (idx_h2) * FUSION_SIZE_SLICE_1_H1 + 0];
					temp_bv[1] = sm_a[ll][idx_h1 + (idx_h2) * FUSION_SIZE_SLICE_1_H1 + 16];
					temp_bv[2] = sm_a[ll][idx_h1 + (idx_h2) * FUSION_SIZE_SLICE_1_H1 + 32];
					temp_bv[3] = sm_a[ll][idx_h1 + (idx_h2) * FUSION_SIZE_SLICE_1_H1 + 48];
					
					for (int xx = 0 ; xx < 4; xx++)
					{
						temp_av = sm_b[ll][idx_h3 + (idx_p6) * FUSION_SIZE_SLICE_1_H3 + (xx * 16)];

						reg_tile[0][xx] -= temp_av * temp_bv[0];
						reg_tile[1][xx] -= temp_av * temp_bv[1];
						reg_tile[2][xx] -= temp_av * temp_bv[2];
						reg_tile[3][xx] -= temp_av * temp_bv[3];
					}
				}
				__syncthreads();
			}
		}
	
		// 	sd2_2
		if (flag_d2_2 >= 0)
		{
			// 
			double* tmp_dev_d2_t2_2 = df_dev_d2_t2_all + size_max_dim_d2_t2 * flag_d2_2;
			double* tmp_dev_d2_v2_2 = df_dev_d2_v2_all + size_max_dim_d2_v2 * flag_d2_2;

			internal_upperbound = 0;
			#pragma unroll 1
			for (int l = 0; l < base_size_p7b; l+= FUSION_SIZE_INT_UNIT)
			{
				// Part: Generalized Contraction Index (p7b)
				internal_offset = (l + FUSION_SIZE_INT_UNIT) - base_size_p7b;
				if (internal_offset > 0) internal_upperbound = internal_offset;
		
				// Load Input Tensor to Shared Memory: 16:16
				// # of size_internal Indices: 1
				if (idx_p6 < rng_h2 && idx_h1 < rng_h3 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
				for (int ll = 0; ll < rng_p4; ll++)
				{
					sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = tmp_dev_d2_t2_2[	(str_blk_idx_p4 + ll + (str_blk_idx_h2 + idx_p6 + (str_blk_idx_h3 + idx_h1) * base_size_h2b) * base_size_p4b) * base_size_p7b + (threadIdx.x + l)];
				}
		
				// Load Input Tensor to Shared Memory
				if (idx_p6 < rng_h1 && idx_h1 < rng_p6 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
				for (int ll = 0; ll < rng_p5; ll++)
				{
					sm_b[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = tmp_dev_d2_v2_2[	(str_blk_idx_h1 + idx_p6 + (str_blk_idx_p6 + idx_h1 + (str_blk_idx_p5 + ll) * base_size_p6b) * base_size_h1b) * base_size_p7b + (threadIdx.x + l)];
				}
				__syncthreads();
		
				// Cross-Product: 16
				// Part: Generalized Threads
				for (int ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++)
				{
					temp_bv[0] = sm_a[ll][idx_h2 + (idx_h3) * FUSION_SIZE_SLICE_1_H2 + 0];
					temp_bv[1] = sm_a[ll][idx_h2 + (idx_h3) * FUSION_SIZE_SLICE_1_H2 + 16];
					temp_bv[2] = sm_a[ll][idx_h2 + (idx_h3) * FUSION_SIZE_SLICE_1_H2 + 32];
					temp_bv[3] = sm_a[ll][idx_h2 + (idx_h3) * FUSION_SIZE_SLICE_1_H2 + 48];
		
					for (int xx = 0 ; xx < 4; xx++)
					{
						temp_av = sm_b[ll][idx_h1 + (idx_p6) * FUSION_SIZE_SLICE_1_H1 + (xx * 16)];
		
						reg_tile[0][xx] -= temp_av * temp_bv[0];
						reg_tile[1][xx] -= temp_av * temp_bv[1];
						reg_tile[2][xx] -= temp_av * temp_bv[2];
						reg_tile[3][xx] -= temp_av * temp_bv[3];
					}
				}
				__syncthreads();
			}
		}

		// 	sd2_3
		if (flag_d2_3 >= 0)
		{
			// 
			double* tmp_dev_d2_t2_3 = df_dev_d2_t2_all + size_max_dim_d2_t2 * flag_d2_3;
			double* tmp_dev_d2_v2_3 = df_dev_d2_v2_all + size_max_dim_d2_v2 * flag_d2_3;

			internal_upperbound = 0;
			#pragma unroll 1
			for (int l = 0; l < base_size_p7b; l+= FUSION_SIZE_INT_UNIT)
			{
				// Part: Generalized Contraction Index (p7b)
				internal_offset = (l + FUSION_SIZE_INT_UNIT) - base_size_p7b;
				if (internal_offset > 0) internal_upperbound = internal_offset;
		
				// Load Input Tensor to Shared Memory: 16:16
				// # of size_internal Indices: 1
				if (idx_p6 < rng_h1 && idx_h1 < rng_h3 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
				for (int ll = 0; ll < rng_p4; ll++)
				{
					sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = tmp_dev_d2_t2_3[	(str_blk_idx_p4 + ll + 
						(str_blk_idx_h1 + idx_p6 + 
						(str_blk_idx_h3 + idx_h1) * base_size_h1b) * base_size_p4b) * base_size_p7b + 
						(threadIdx.x + l)];
				}
		
				// Load Input Tensor to Shared Memory
				if (idx_p6 < rng_h2 && idx_h1 < rng_p6 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
				for (int ll = 0; ll < rng_p5; ll++)
				{
					sm_b[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = tmp_dev_d2_v2_3[	(str_blk_idx_h2 + idx_p6 + 
						(str_blk_idx_p6 + idx_h1 + 
						(str_blk_idx_p5 + ll) * base_size_p6b) * base_size_h2b) * base_size_p7b + 
						(threadIdx.x + l)];
				}
				__syncthreads();
		
				// Cross-Product: 16
				// Part: Generalized Threads
				for (int ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++)
				{
					temp_bv[0] = sm_a[ll][idx_h1 + (idx_h3) * FUSION_SIZE_SLICE_1_H1 + 0];
					temp_bv[1] = sm_a[ll][idx_h1 + (idx_h3) * FUSION_SIZE_SLICE_1_H1 + 16];
					temp_bv[2] = sm_a[ll][idx_h1 + (idx_h3) * FUSION_SIZE_SLICE_1_H1 + 32];
					temp_bv[3] = sm_a[ll][idx_h1 + (idx_h3) * FUSION_SIZE_SLICE_1_H1 + 48];
		
					for (int xx = 0 ; xx < 4; xx++)
					{
						temp_av = sm_b[ll][idx_h2 + (idx_p6) * FUSION_SIZE_SLICE_1_H2 + (xx * 16)];
		
						reg_tile[0][xx] += temp_av * temp_bv[0];
						reg_tile[1][xx] += temp_av * temp_bv[1];
						reg_tile[2][xx] += temp_av * temp_bv[2];
						reg_tile[3][xx] += temp_av * temp_bv[3];
					}
				}
				__syncthreads();
			}
		}
	
		// 	sd2_4
		if (flag_d2_4 >= 0)
		{
			// 
			double* tmp_dev_d2_t2_4 = df_dev_d2_t2_all + size_max_dim_d2_t2 * flag_d2_4;
			double* tmp_dev_d2_v2_4 = df_dev_d2_v2_all + size_max_dim_d2_v2 * flag_d2_4;

			internal_upperbound = 0;
			#pragma unroll 1
			for (int l = 0; l < base_size_p7b; l+= FUSION_SIZE_INT_UNIT)
			{
				// Part: Generalized Contraction Index (p7b)
				internal_offset = (l + FUSION_SIZE_INT_UNIT) - base_size_p7b;
				if (internal_offset > 0) internal_upperbound = internal_offset;
		
				// Load Input Tensor to Shared Memory: 16:16
				// # of size_internal Indices: 1
				if (idx_p6 < rng_h1 && idx_h1 < rng_h2 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
				for (int ll = 0; ll < rng_p5; ll++)
				{
					sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = tmp_dev_d2_t2_4[	(str_blk_idx_p5 + ll + (str_blk_idx_h1 + idx_p6 + (str_blk_idx_h2 + idx_h1) * base_size_h1b) * base_size_p5b) * base_size_p7b + (threadIdx.x + l)];
				}
		
				// Load Input Tensor to Shared Memory
				if (idx_p6 < rng_h3 && idx_h1 < rng_p6 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
				for (int ll = 0; ll < rng_p4; ll++)
				{
					sm_b[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = tmp_dev_d2_v2_4[	(str_blk_idx_h3 + idx_p6 + (str_blk_idx_p6 + idx_h1 + (str_blk_idx_p4 + ll) * base_size_p6b) * base_size_h3b) * base_size_p7b + (threadIdx.x + l)];
				}
				__syncthreads();
		
				// Cross-Product: 16
				// Part: Generalized Threads
				for (int ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++)
				{
					temp_bv[0] = sm_b[ll][idx_h3 + (idx_p6) * FUSION_SIZE_SLICE_1_H1 + 0];
					temp_bv[1] = sm_b[ll][idx_h3 + (idx_p6) * FUSION_SIZE_SLICE_1_H1 + 16];
					temp_bv[2] = sm_b[ll][idx_h3 + (idx_p6) * FUSION_SIZE_SLICE_1_H1 + 32];
					temp_bv[3] = sm_b[ll][idx_h3 + (idx_p6) * FUSION_SIZE_SLICE_1_H1 + 48];
		
					for (int xx = 0 ; xx < 4; xx++)
					{
						temp_av = sm_a[ll][idx_h1 + (idx_h2) * FUSION_SIZE_SLICE_1_H1 + (xx * 16)];
		
						reg_tile[0][xx] += temp_av * temp_bv[0];
						reg_tile[1][xx] += temp_av * temp_bv[1];
						reg_tile[2][xx] += temp_av * temp_bv[2];
						reg_tile[3][xx] += temp_av * temp_bv[3];
					}
				}
				__syncthreads();
			}
		}
	
		// 	sd2_5
		if (flag_d2_5 >= 0)
		{
			// 
			double* tmp_dev_d2_t2_5 = df_dev_d2_t2_all + size_max_dim_d2_t2 * flag_d2_5;
			double* tmp_dev_d2_v2_5 = df_dev_d2_v2_all + size_max_dim_d2_v2 * flag_d2_5;

			internal_upperbound = 0;
			#pragma unroll 1
			for (int l = 0; l < base_size_p7b; l+= FUSION_SIZE_INT_UNIT)
			{
				// Part: Generalized Contraction Index (p7b)
				internal_offset = (l + FUSION_SIZE_INT_UNIT) - base_size_p7b;
				if (internal_offset > 0) internal_upperbound = internal_offset;
		
				// Load Input Tensor to Shared Memory: 16:16
				// # of size_internal Indices: 1
				if (idx_p6 < rng_h2 && idx_h1 < rng_h3 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
				for (int ll = 0; ll < rng_p5; ll++)
				{
					sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = tmp_dev_d2_t2_5[	(str_blk_idx_p5 + ll + (str_blk_idx_h2 + idx_p6 + (str_blk_idx_h3 + idx_h1) * base_size_h2b) * base_size_p5b) * base_size_p7b + (threadIdx.x + l)];
				}
		
				// Load Input Tensor to Shared Memory
				if (idx_p6 < rng_h1 && idx_h1 < rng_p6 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
				for (int ll = 0; ll < rng_p4; ll++)
				{
					sm_b[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = tmp_dev_d2_v2_5[	(str_blk_idx_h1 + idx_p6 + (str_blk_idx_p6 + idx_h1 + (str_blk_idx_p4 + ll) * base_size_p6b) * base_size_h1b) * base_size_p7b + (threadIdx.x + l)];
				}
				__syncthreads();
		
				// Cross-Product: 16
				// Part: Generalized Threads
				for (int ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++)
				{
					temp_bv[0] = sm_b[ll][idx_h1 + (idx_p6) * FUSION_SIZE_SLICE_1_H1 + 0];
					temp_bv[1] = sm_b[ll][idx_h1 + (idx_p6) * FUSION_SIZE_SLICE_1_H1 + 16];
					temp_bv[2] = sm_b[ll][idx_h1 + (idx_p6) * FUSION_SIZE_SLICE_1_H1 + 32];
					temp_bv[3] = sm_b[ll][idx_h1 + (idx_p6) * FUSION_SIZE_SLICE_1_H1 + 48];
		
					for (int xx = 0 ; xx < 4; xx++)
					{
						temp_av = sm_a[ll][idx_h2 + (idx_h3) * FUSION_SIZE_SLICE_1_H2 + (xx * 16)];
		
						reg_tile[0][xx] += temp_av * temp_bv[0];
						reg_tile[1][xx] += temp_av * temp_bv[1];
						reg_tile[2][xx] += temp_av * temp_bv[2];
						reg_tile[3][xx] += temp_av * temp_bv[3];
					}
				}
				__syncthreads();
			}
		}
	
		// 	sd2_6
		if (flag_d2_6 >= 0)
		{
			// 
			double* tmp_dev_d2_t2_6 = df_dev_d2_t2_all + size_max_dim_d2_t2 * flag_d2_6;
			double* tmp_dev_d2_v2_6 = df_dev_d2_v2_all + size_max_dim_d2_v2 * flag_d2_6;

			internal_upperbound = 0;
			#pragma unroll 1
			for (int l = 0; l < base_size_p7b; l+= FUSION_SIZE_INT_UNIT)
			{
				// Part: Generalized Contraction Index (p7b)
				internal_offset = (l + FUSION_SIZE_INT_UNIT) - base_size_p7b;
				if (internal_offset > 0) internal_upperbound = internal_offset;
		
				// Load Input Tensor to Shared Memory: 16:16
				// # of size_internal Indices: 1
				if (idx_p6 < rng_h1 && idx_h1 < rng_h3 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
				for (int ll = 0; ll < rng_p5; ll++)
				{
					sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = tmp_dev_d2_t2_6[	(blk_idx_p5b * FUSION_SIZE_SLICE_1_P6 + ll + (str_blk_idx_h1 + idx_p6 + (str_blk_idx_h3 + idx_h1) * base_size_h1b) * base_size_p5b) * base_size_p7b + (threadIdx.x + l)];
				}
		
				// Load Input Tensor to Shared Memory
				if (idx_p6 < rng_h2 && idx_h1 < rng_p6 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
				for (int ll = 0; ll < rng_p4; ll++)
				{
					sm_b[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = tmp_dev_d2_v2_6[	(str_blk_idx_h2 + idx_p6 + (str_blk_idx_p6 + idx_h1 + (str_blk_idx_p4 + ll) * base_size_p6b) * base_size_h2b) * base_size_p7b + (threadIdx.x + l)];
				}
				__syncthreads();
		
				// Cross-Product: 16
				// Part: Generalized Threads
				for (int ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++)
				{
					temp_bv[0] = sm_b[ll][idx_h2 + (idx_p6) * FUSION_SIZE_SLICE_1_H2 + 0];
					temp_bv[1] = sm_b[ll][idx_h2 + (idx_p6) * FUSION_SIZE_SLICE_1_H2 + 16];
					temp_bv[2] = sm_b[ll][idx_h2 + (idx_p6) * FUSION_SIZE_SLICE_1_H2 + 32];
					temp_bv[3] = sm_b[ll][idx_h2 + (idx_p6) * FUSION_SIZE_SLICE_1_H2 + 48];
		
					for (int xx = 0; xx < 4; xx++)	// 4 -> rng_p4: Local Transactions...
					{
						temp_av = sm_a[ll][idx_h1 + (idx_h3) * FUSION_SIZE_SLICE_1_H1 + (xx * 16)];
		
						reg_tile[0][xx] -= temp_av * temp_bv[0];
						reg_tile[1][xx] -= temp_av * temp_bv[1];
						reg_tile[2][xx] -= temp_av * temp_bv[2];
						reg_tile[3][xx] -= temp_av * temp_bv[3];
					}
				}
				__syncthreads();
			}
		}
	}

	// 
	// 	>>> s1 <<<
	// 		- if 
	// 
	//  singles (s1)
	{	
		base_size_h1b = const_df_s1_size[0];
		base_size_h2b = const_df_s1_size[1];
		base_size_h3b = const_df_s1_size[2];
		base_size_p4b = const_df_s1_size[3];
		base_size_p5b = const_df_s1_size[4];
		base_size_p6b = const_df_s1_size[5];

		num_blks_h1b = CEIL(base_size_h1b, FUSION_SIZE_SLICE_1_H1);
		num_blks_h2b = CEIL(base_size_h2b, FUSION_SIZE_SLICE_1_H2);
		num_blks_h3b = CEIL(base_size_h3b, FUSION_SIZE_SLICE_1_H3);
		num_blks_p4b = CEIL(base_size_p4b, FUSION_SIZE_SLICE_1_P4);
		num_blks_p5b = CEIL(base_size_p5b, FUSION_SIZE_SLICE_1_P5);
		num_blks_p6b = CEIL(base_size_p6b, FUSION_SIZE_SLICE_1_P6);

		// 	(2) blk_idx_h/p*b
		blk_idx_p4b     = blockIdx.x / (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b * num_blks_p5b);
		tmp_blkIdx      = blockIdx.x % (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b * num_blks_p5b);
		blk_idx_p5b     = (tmp_blkIdx) / (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b);
		tmp_blkIdx  	= (tmp_blkIdx) % (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b);
		blk_idx_p6b     = (tmp_blkIdx) / (num_blks_h3b * num_blks_h2b * num_blks_h1b);
		tmp_blkIdx 		= (tmp_blkIdx) % (num_blks_h3b * num_blks_h2b * num_blks_h1b);
		blk_idx_h1b     = (tmp_blkIdx) / (num_blks_h3b * num_blks_h2b);
		tmp_blkIdx 		= (tmp_blkIdx) % (num_blks_h3b * num_blks_h2b);
		blk_idx_h2b 	= (tmp_blkIdx) / (num_blks_h3b);
		blk_idx_h3b		= blockIdx.x % (num_blks_h3b);

		// 	(3) str_blk_idx_h/p*
		str_blk_idx_h3 	= blk_idx_h3b * FUSION_SIZE_SLICE_1_H3;
		str_blk_idx_h2 	= blk_idx_h2b * FUSION_SIZE_SLICE_1_H2;
		str_blk_idx_h1 	= blk_idx_h1b * FUSION_SIZE_SLICE_1_H1;
		str_blk_idx_p6 	= blk_idx_p6b * FUSION_SIZE_SLICE_1_P6;
		str_blk_idx_p5 	= blk_idx_p5b * FUSION_SIZE_SLICE_1_P5;
		str_blk_idx_p4 	= blk_idx_p4b * FUSION_SIZE_SLICE_1_P4;

		// 	(4) rng_h/p*
		if ((base_size_h3b - (str_blk_idx_h3)) >= FUSION_SIZE_SLICE_1_H3)
			rng_h3 = FUSION_SIZE_SLICE_1_H3;
		else
			rng_h3 = base_size_h3b % FUSION_SIZE_SLICE_1_H3;
		
		if ((base_size_h2b - (str_blk_idx_h2)) >= FUSION_SIZE_SLICE_1_H2)
			rng_h2 = FUSION_SIZE_SLICE_1_H2;
		else
			rng_h2 = base_size_h2b % FUSION_SIZE_SLICE_1_H2;

		if ((base_size_h1b - (str_blk_idx_h1)) >= FUSION_SIZE_SLICE_1_H1)
			rng_h1 = FUSION_SIZE_SLICE_1_H1;
		else
			rng_h1 = base_size_h1b % FUSION_SIZE_SLICE_1_H1;
		
		if ((base_size_p6b - (str_blk_idx_p6)) >= FUSION_SIZE_SLICE_1_P6)
			rng_p6 = FUSION_SIZE_SLICE_1_P6;
		else
			rng_p6 = base_size_p6b % FUSION_SIZE_SLICE_1_P6;

		if ((base_size_p5b - (str_blk_idx_p5)) >= FUSION_SIZE_SLICE_1_P5)
			rng_p5 = FUSION_SIZE_SLICE_1_P5;
		else
			rng_p5 = base_size_p5b % FUSION_SIZE_SLICE_1_P5;

		if ((base_size_p4b - (str_blk_idx_p4)) >= FUSION_SIZE_SLICE_1_P4)
			rng_p4 = FUSION_SIZE_SLICE_1_P4;
		else
			rng_p4 = base_size_p4b % FUSION_SIZE_SLICE_1_P4;

		
		// 	flags
		int flag_s1_1 = const_df_s1_exec[0];
		int flag_s1_2 = const_df_s1_exec[1];
		int flag_s1_3 = const_df_s1_exec[2];
		int flag_s1_4 = const_df_s1_exec[3];
		int flag_s1_5 = const_df_s1_exec[4];
		int flag_s1_6 = const_df_s1_exec[5];
		int flag_s1_7 = const_df_s1_exec[6];
		int flag_s1_8 = const_df_s1_exec[7];
		int flag_s1_9 = const_df_s1_exec[8];
		
		//                                        "x"         "x"
		//  >> s1_1:   t3[h3,h2,h1,p6,p5,p4] -= t2[p4,h1] * v2[h3,h2,p6,p5]
		//
		if (flag_s1_1 >= 0)	// these if-conditions make 100 ms..
		{
			//
			double* tmp_dev_s1_t1_1 = df_dev_s1_t1_all + size_max_dim_s1_t1 * flag_s1_1;
			double* tmp_dev_s1_v2_1 = df_dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_1;

			if (idx_h3 < rng_p4 && idx_h2 < rng_h1 && idx_p6 == 0 && idx_h1 == 0)
			sm_a[0][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_1_P4] = tmp_dev_s1_t1_1[str_blk_idx_p4 + idx_h3 + (str_blk_idx_h1 + idx_h2) * base_size_p4b];

			if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6 && idx_h1 < rng_p5)
			sm_b[idx_h1][idx_h3 + (idx_h2 + (idx_p6) * 4) * 4] = tmp_dev_s1_v2_1[blk_idx_h3b * 4 + idx_h3 + (blk_idx_h2b * 4 + idx_h2 + (blk_idx_p6b * 4 + idx_p6 + (blk_idx_p5b * 4 + idx_h1) * base_size_p6b) * base_size_h2b) * base_size_h3b];
			__syncthreads();

			//  "p4"
			temp_av = sm_a[0][0 + (idx_h1) * 4];
			
			//  "p5"
			temp_bv[0] = sm_b[0][idx_h3 + (idx_h2 + (idx_p6) * 4) * 4];
			temp_bv[1] = sm_b[1][idx_h3 + (idx_h2 + (idx_p6) * 4) * 4];
			temp_bv[2] = sm_b[2][idx_h3 + (idx_h2 + (idx_p6) * 4) * 4];
			temp_bv[3] = sm_b[3][idx_h3 + (idx_h2 + (idx_p6) * 4) * 4];

			//  "p4 x p5"
			reg_singles[0][0] += temp_av * temp_bv[0]; 
			reg_singles[0][1] += temp_av * temp_bv[1]; 
			reg_singles[0][2] += temp_av * temp_bv[2]; 
			reg_singles[0][3] += temp_av * temp_bv[3]; 

			temp_av = sm_a[0][1 + (idx_h1) * 4];
			
			reg_singles[1][0] += temp_av * temp_bv[0]; 
			reg_singles[1][1] += temp_av * temp_bv[1]; 
			reg_singles[1][2] += temp_av * temp_bv[2]; 
			reg_singles[1][3] += temp_av * temp_bv[3]; 
			
			temp_av = sm_a[0][2 + (idx_h1) * 4];

			reg_singles[2][0] += temp_av * temp_bv[0]; 
			reg_singles[2][1] += temp_av * temp_bv[1]; 
			reg_singles[2][2] += temp_av * temp_bv[2]; 
			reg_singles[2][3] += temp_av * temp_bv[3]; 

			temp_av = sm_a[0][3 + (idx_h1) * 4];

			reg_singles[3][0] += temp_av * temp_bv[0]; 
			reg_singles[3][1] += temp_av * temp_bv[1]; 
			reg_singles[3][2] += temp_av * temp_bv[2]; 
			reg_singles[3][3] += temp_av * temp_bv[3]; 
			__syncthreads();
		}

		//                                        "x1,x2"     "x1,x2,x3,y1"
		//  >> s1_2:   t3[h3,h2,h1,p6,p5,p4] -= t2[p4,h2] * v2[h3,h1,p6,p5] (h3,h2,p6), (h1)
		//
		if (flag_s1_2 >= 0)	// these if-conditions make 100 ms..
		{
			// 
			double* tmp_dev_s1_t1_2 = df_dev_s1_t1_all + size_max_dim_s1_t1 * flag_s1_2;
			double* tmp_dev_s1_v2_2 = df_dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_2;

			if (idx_h3 < rng_p4 && idx_h2 < rng_h2 && idx_p6 == 0 && idx_h1 == 0)
			sm_a[0][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_1_P4] = tmp_dev_s1_t1_2[str_blk_idx_p4 + idx_h3 + (str_blk_idx_h2 + idx_h2) * base_size_p4b];
			
			if (idx_h3 < rng_h3 && idx_h2 < rng_h1 && idx_p6 < rng_p6 && idx_h1 < rng_p5)
			sm_b[idx_h1][idx_h3 + (idx_h2 + (idx_p6) * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H3] = tmp_dev_s1_v2_2[str_blk_idx_h3 + idx_h3 + (str_blk_idx_h1 + idx_h2 + (str_blk_idx_p6 + idx_p6 + (str_blk_idx_p5 + idx_h1) * base_size_p6b) * base_size_h1b) * base_size_h3b];
			__syncthreads();

			//  "p4"
			temp_av = sm_a[0][0 + (idx_h2) * 4];
			
			//  "p5"
			temp_bv[0] = sm_b[0][idx_h3 + (idx_h1 + (idx_p6) * 4) * 4];
			temp_bv[1] = sm_b[1][idx_h3 + (idx_h1 + (idx_p6) * 4) * 4];
			temp_bv[2] = sm_b[2][idx_h3 + (idx_h1 + (idx_p6) * 4) * 4];
			temp_bv[3] = sm_b[3][idx_h3 + (idx_h1 + (idx_p6) * 4) * 4];

			//  "p4 x p5"
			reg_singles[0][0] -= temp_av * temp_bv[0];
			reg_singles[0][1] -= temp_av * temp_bv[1];
			reg_singles[0][2] -= temp_av * temp_bv[2];
			reg_singles[0][3] -= temp_av * temp_bv[3];

			temp_av = sm_a[0][1 + (idx_h2) * 4];

			reg_singles[1][0] -= temp_av * temp_bv[0];
			reg_singles[1][1] -= temp_av * temp_bv[1];
			reg_singles[1][2] -= temp_av * temp_bv[2];
			reg_singles[1][3] -= temp_av * temp_bv[3];

			temp_av = sm_a[0][2 + (idx_h2) * 4];

			reg_singles[2][0] -= temp_av * temp_bv[0];
			reg_singles[2][1] -= temp_av * temp_bv[1];
			reg_singles[2][2] -= temp_av * temp_bv[2];
			reg_singles[2][3] -= temp_av * temp_bv[3];

			temp_av = sm_a[0][3 + (idx_h2) * 4];

			reg_singles[3][0] -= temp_av * temp_bv[0];
			reg_singles[3][1] -= temp_av * temp_bv[1];
			reg_singles[3][2] -= temp_av * temp_bv[2];
			reg_singles[3][3] -= temp_av * temp_bv[3];
			__syncthreads();
		}

		//
		//  >> s1_3:   t3[h3,h2,h1,p6,p5,p4] -= t2[p4,h1] * v2[h3,h2,p6,p5] >> t3[h3,h2,h1,p6,p5,p4] += t2[p4,h3] * v2[h2,h1,p6,p5]
		//
		if (flag_s1_3 >= 0)	// these if-conditions make 100 ms..
		{
			// 
			double* tmp_dev_s1_t1_3 = df_dev_s1_t1_all + size_max_dim_s1_t1 * flag_s1_3;
			double* tmp_dev_s1_v2_3 = df_dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_3;

			if (idx_h3 < rng_p4 && idx_h2 < rng_h3 && idx_p6 == 0 && idx_h1 == 0)
			sm_a[0][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_1_P4] = tmp_dev_s1_t1_3[str_blk_idx_p4 + idx_h3 + (str_blk_idx_h3 + idx_h2) * base_size_p4b];

			if (idx_h3 < rng_h2 && idx_h2 < rng_h1 && idx_p6 < rng_p6 && idx_h1 < rng_p5)
			sm_b[idx_h1][idx_h3 + (idx_h2 + (idx_p6) * 4) * 4] = tmp_dev_s1_v2_3[blk_idx_h2b * 4 + idx_h3 + (blk_idx_h1b * 4 + idx_h2 + (blk_idx_p6b * 4 + idx_p6 + (blk_idx_p5b * 4 + idx_h1) * base_size_p6b) * base_size_h1b) * base_size_h2b];
			__syncthreads();

			//  "p4"
			temp_av = sm_a[0][0 + (idx_h3) * 4];
			
			//  "p5"
			temp_bv[0] = sm_b[0][idx_h2 + (idx_h1 + (idx_p6) * 4) * 4];
			temp_bv[1] = sm_b[1][idx_h2 + (idx_h1 + (idx_p6) * 4) * 4];
			temp_bv[2] = sm_b[2][idx_h2 + (idx_h1 + (idx_p6) * 4) * 4];
			temp_bv[3] = sm_b[3][idx_h2 + (idx_h1 + (idx_p6) * 4) * 4];

			//  "p4 x p5"
			reg_singles[0][0] += temp_av * temp_bv[0];
			reg_singles[0][1] += temp_av * temp_bv[1];
			reg_singles[0][2] += temp_av * temp_bv[2];
			reg_singles[0][3] += temp_av * temp_bv[3];

			temp_av = sm_a[0][1 + (idx_h3) * 4];

			reg_singles[1][0] += temp_av * temp_bv[0];
			reg_singles[1][1] += temp_av * temp_bv[1];
			reg_singles[1][2] += temp_av * temp_bv[2];
			reg_singles[1][3] += temp_av * temp_bv[3];

			temp_av = sm_a[0][2 + (idx_h3) * 4];

			reg_singles[2][0] += temp_av * temp_bv[0];
			reg_singles[2][1] += temp_av * temp_bv[1];
			reg_singles[2][2] += temp_av * temp_bv[2];
			reg_singles[2][3] += temp_av * temp_bv[3];

			temp_av = sm_a[0][3 + (idx_h3) * 4];

			reg_singles[3][0] += temp_av * temp_bv[0];
			reg_singles[3][1] += temp_av * temp_bv[1];
			reg_singles[3][2] += temp_av * temp_bv[2];
			reg_singles[3][3] += temp_av * temp_bv[3];
			__syncthreads();
		}
	
		//
		//  >> s1_4:   t3[h3,h2,h1,p6,p5,p4] -= t2[p5,h1] * v2[h3,h2,p6,p4] (h3,h2,p6), (h1)
		//
		if (flag_s1_4 >= 0)	// these if-conditions make 100 ms..
		{
			double* tmp_dev_s1_t1_4 = df_dev_s1_t1_all + size_max_dim_s1_t1 * flag_s1_4;
			double* tmp_dev_s1_v2_4 = df_dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_4;

			if (idx_h3 < rng_p5 && idx_h2 < rng_h1 && idx_p6 == 0 && idx_h1 == 0)
			sm_a[0][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_1_P5] = tmp_dev_s1_t1_4[str_blk_idx_p5 + idx_h3 + (str_blk_idx_h1 + idx_h2) * base_size_p5b];
			
			if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6 && idx_h1 < rng_p4)
			sm_b[idx_h1][idx_h3 + (idx_h2 + (idx_p6) * 4) * 4] = tmp_dev_s1_v2_4[str_blk_idx_h3 + idx_h3 + (str_blk_idx_h2 + idx_h2 + (str_blk_idx_p6 + idx_p6 + (str_blk_idx_p4 + idx_h1) * base_size_p6b) * base_size_h2b) * base_size_h3b];
			__syncthreads();

			//  "p5"
			temp_av = sm_a[0][0 + (idx_h1) * 4];
			
			//  "p4"
			temp_bv[0] = sm_b[0][idx_h3 + (idx_h2 + (idx_p6) * 4) * 4];
			temp_bv[1] = sm_b[1][idx_h3 + (idx_h2 + (idx_p6) * 4) * 4];
			temp_bv[2] = sm_b[2][idx_h3 + (idx_h2 + (idx_p6) * 4) * 4];
			temp_bv[3] = sm_b[3][idx_h3 + (idx_h2 + (idx_p6) * 4) * 4];

			//  "p4 x p5"
			reg_singles[0][0] -= temp_av * temp_bv[0];
			reg_singles[1][0] -= temp_av * temp_bv[1];
			reg_singles[2][0] -= temp_av * temp_bv[2];
			reg_singles[3][0] -= temp_av * temp_bv[3];

			temp_av = sm_a[0][1 + (idx_h1) * 4];

			reg_singles[0][1] -= temp_av * temp_bv[0];
			reg_singles[1][1] -= temp_av * temp_bv[1];
			reg_singles[2][1] -= temp_av * temp_bv[2];
			reg_singles[3][1] -= temp_av * temp_bv[3];

			temp_av = sm_a[0][2 + (idx_h1) * 4];

			reg_singles[0][2] -= temp_av * temp_bv[0];
			reg_singles[1][2] -= temp_av * temp_bv[1];
			reg_singles[2][2] -= temp_av * temp_bv[2];
			reg_singles[3][2] -= temp_av * temp_bv[3];

			temp_av = sm_a[0][3 + (idx_h1) * 4];

			reg_singles[0][3] -= temp_av * temp_bv[0];
			reg_singles[1][3] -= temp_av * temp_bv[1];
			reg_singles[2][3] -= temp_av * temp_bv[2];
			reg_singles[3][3] -= temp_av * temp_bv[3];
			__syncthreads();
		}

		//
		//  >> s1_5:   t3[h3,h2,h1,p6,p5,p4] -= t2[p5,h2] * v2[h3,h1,p6,p4] (h3,h2,p6), (h1)
		//
		if (flag_s1_5 >= 0)	// these if-conditions make 100 ms..
		{
			// 
			double* tmp_dev_s1_t1_5 = df_dev_s1_t1_all + size_max_dim_s1_t1 * flag_s1_5;
			double* tmp_dev_s1_v2_5 = df_dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_5;

			if (idx_h3 < rng_p5 && idx_h2 < rng_h2 && idx_p6 == 0 && idx_h1 == 0)
			sm_a[0][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_1_P5] = tmp_dev_s1_t1_5[str_blk_idx_p5 + idx_h3 + (str_blk_idx_h2 + idx_h2) * base_size_p5b];

			if (idx_h3 < rng_h3 && idx_h2 < rng_h1 && idx_p6 < rng_p6 && idx_h1 < rng_p4)
			sm_b[idx_h1][idx_h3 + (idx_h2 + (idx_p6) * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H3] = tmp_dev_s1_v2_5[str_blk_idx_h3 + idx_h3 + (str_blk_idx_h1 + idx_h2 + (str_blk_idx_p6 + idx_p6 + (str_blk_idx_p4 + idx_h1) * base_size_p6b) * base_size_h1b) * base_size_h3b];
			__syncthreads();

			//  "p5"
			temp_av = sm_a[0][0 + (idx_h2) * 4];
			
			//  "p4"
			temp_bv[0] = sm_b[0][idx_h3 + (idx_h1 + (idx_p6) * 4) * 4];
			temp_bv[1] = sm_b[1][idx_h3 + (idx_h1 + (idx_p6) * 4) * 4];
			temp_bv[2] = sm_b[2][idx_h3 + (idx_h1 + (idx_p6) * 4) * 4];
			temp_bv[3] = sm_b[3][idx_h3 + (idx_h1 + (idx_p6) * 4) * 4];

			//  "p4 x p5"
			reg_singles[0][0] += temp_av * temp_bv[0];
			reg_singles[1][0] += temp_av * temp_bv[1];
			reg_singles[2][0] += temp_av * temp_bv[2];
			reg_singles[3][0] += temp_av * temp_bv[3];
			
			temp_av = sm_a[0][1 + (idx_h2) * 4];

			reg_singles[0][1] += temp_av * temp_bv[0];
			reg_singles[1][1] += temp_av * temp_bv[1];
			reg_singles[2][1] += temp_av * temp_bv[2];
			reg_singles[3][1] += temp_av * temp_bv[3];

			temp_av = sm_a[0][2 + (idx_h2) * 4];

			reg_singles[0][2] += temp_av * temp_bv[0];
			reg_singles[1][2] += temp_av * temp_bv[1];
			reg_singles[2][2] += temp_av * temp_bv[2];
			reg_singles[3][2] += temp_av * temp_bv[3];

			temp_av = sm_a[0][3 + (idx_h2) * 4];

			reg_singles[0][3] += temp_av * temp_bv[0];
			reg_singles[1][3] += temp_av * temp_bv[1];
			reg_singles[2][3] += temp_av * temp_bv[2];
			reg_singles[3][3] += temp_av * temp_bv[3];
			__syncthreads();
		}
		
		//
		//  >> s1_6:   t3[h3,h2,h1,p6,p5,p4] -= t2[p5,h3] * v2[h2,h1,p6,p4] (h3,h2,p6), (h1)
		//
		if (flag_s1_6 >= 0)	// these if-conditions make 100 ms..
		{
			// 
			double* tmp_dev_s1_t1_6 = df_dev_s1_t1_all + size_max_dim_s1_t1 * flag_s1_6;
			double* tmp_dev_s1_v2_6 = df_dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_6;

			if (idx_h3 < rng_p5 && idx_h2 < rng_h3 && idx_p6 == 0 && idx_h1 == 0)
			sm_a[0][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_1_P5] = tmp_dev_s1_t1_6[str_blk_idx_p5 + idx_h3 + (str_blk_idx_h3 + idx_h2) * base_size_p5b];

			if (idx_h3 < rng_h2 && idx_h2 < rng_h1 && idx_p6 < rng_p6 && idx_h1 < rng_p4)
			sm_b[idx_h1][idx_h3 + (idx_h2 + (idx_p6) * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H2] = tmp_dev_s1_v2_6[str_blk_idx_h2 + idx_h3 + (str_blk_idx_h1 + idx_h2 + (str_blk_idx_p6 + idx_p6 + (str_blk_idx_p4 + idx_h1) * base_size_p6b) * base_size_h1b) * base_size_h2b];
			__syncthreads();

			//  "p5"
			temp_av = sm_a[0][0 + (idx_h3) * FUSION_SIZE_SLICE_1_P5];
			
			//  "p4"
			temp_bv[0] = sm_b[0][idx_h2 + (idx_h1 + (idx_p6) * 4) * 4];
			temp_bv[1] = sm_b[1][idx_h2 + (idx_h1 + (idx_p6) * 4) * 4];
			temp_bv[2] = sm_b[2][idx_h2 + (idx_h1 + (idx_p6) * 4) * 4];
			temp_bv[3] = sm_b[3][idx_h2 + (idx_h1 + (idx_p6) * 4) * 4];

			//  "p4 x p5"
			reg_singles[0][0] -= temp_av * temp_bv[0];
			reg_singles[1][0] -= temp_av * temp_bv[1];
			reg_singles[2][0] -= temp_av * temp_bv[2];
			reg_singles[3][0] -= temp_av * temp_bv[3];
			
			temp_av = sm_a[0][1 + (idx_h3) * FUSION_SIZE_SLICE_1_P5];

			reg_singles[0][1] -= temp_av * temp_bv[0];
			reg_singles[1][1] -= temp_av * temp_bv[1];
			reg_singles[2][1] -= temp_av * temp_bv[2];
			reg_singles[3][1] -= temp_av * temp_bv[3];
			
			temp_av = sm_a[0][2 + (idx_h3) * FUSION_SIZE_SLICE_1_P5];

			reg_singles[0][2] -= temp_av * temp_bv[0];
			reg_singles[1][2] -= temp_av * temp_bv[1];
			reg_singles[2][2] -= temp_av * temp_bv[2];
			reg_singles[3][2] -= temp_av * temp_bv[3];

			temp_av = sm_a[0][3 + (idx_h3) * FUSION_SIZE_SLICE_1_P5];

			reg_singles[0][3] -= temp_av * temp_bv[0];
			reg_singles[1][3] -= temp_av * temp_bv[1];
			reg_singles[2][3] -= temp_av * temp_bv[2];
			reg_singles[3][3] -= temp_av * temp_bv[3];
			__syncthreads();
		}
		
		//
		//  >> s1_7:   t3[h3,h2,h1,p6,p5,p4] -= t2[p6,h1] * v2[h3,h2,p5,p4] (h3,h2,p6), (h1)
		//
		if (flag_s1_7 >= 0)	// these if-conditions make 100 ms..
		{
			//
			double* tmp_dev_s1_t1_7 = df_dev_s1_t1_all + size_max_dim_s1_t1 * flag_s1_7;
			double* tmp_dev_s1_v2_7 = df_dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_7;

			if (idx_h3 < rng_p6 && idx_h2 < rng_h1 && idx_p6 == 0 && idx_h1 == 0)
			sm_a[0][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_1_P6] = tmp_dev_s1_t1_7[str_blk_idx_p6 + idx_h3 + (str_blk_idx_h1 + idx_h2) * base_size_p6b];
			
			if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p5 && idx_h1 < rng_p4)
			sm_b[idx_h1][idx_h3 + (idx_h2 + (idx_p6) * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3] = tmp_dev_s1_v2_7[str_blk_idx_h3 + idx_h3 + (str_blk_idx_h2 + idx_h2 + (str_blk_idx_p5 + idx_p6 + (str_blk_idx_p4 + idx_h1) * base_size_p5b) * base_size_h2b) * base_size_h3b];
			__syncthreads();

			//  "p4" x "p5"
			reg_singles[0][0] += sm_a[0][idx_p6 + (idx_h1) * FUSION_SIZE_SLICE_1_P6] * sm_b[0][idx_h3 + (idx_h2 + (0) * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3];
			reg_singles[0][1] += sm_a[0][idx_p6 + (idx_h1) * FUSION_SIZE_SLICE_1_P6] * sm_b[0][idx_h3 + (idx_h2 + (1) * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3];
			reg_singles[0][2] += sm_a[0][idx_p6 + (idx_h1) * FUSION_SIZE_SLICE_1_P6] * sm_b[0][idx_h3 + (idx_h2 + (2) * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3];
			reg_singles[0][3] += sm_a[0][idx_p6 + (idx_h1) * FUSION_SIZE_SLICE_1_P6] * sm_b[0][idx_h3 + (idx_h2 + (3) * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3];

			reg_singles[1][0] += sm_a[0][idx_p6 + (idx_h1) * FUSION_SIZE_SLICE_1_P6] * sm_b[1][idx_h3 + (idx_h2 + (0) * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3];
			reg_singles[1][1] += sm_a[0][idx_p6 + (idx_h1) * FUSION_SIZE_SLICE_1_P6] * sm_b[1][idx_h3 + (idx_h2 + (1) * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3];
			reg_singles[1][2] += sm_a[0][idx_p6 + (idx_h1) * FUSION_SIZE_SLICE_1_P6] * sm_b[1][idx_h3 + (idx_h2 + (2) * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3];
			reg_singles[1][3] += sm_a[0][idx_p6 + (idx_h1) * FUSION_SIZE_SLICE_1_P6] * sm_b[1][idx_h3 + (idx_h2 + (3) * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3];

			reg_singles[2][0] += sm_a[0][idx_p6 + (idx_h1) * FUSION_SIZE_SLICE_1_P6] * sm_b[2][idx_h3 + (idx_h2 + (0) * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3];
			reg_singles[2][1] += sm_a[0][idx_p6 + (idx_h1) * FUSION_SIZE_SLICE_1_P6] * sm_b[2][idx_h3 + (idx_h2 + (1) * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3];
			reg_singles[2][2] += sm_a[0][idx_p6 + (idx_h1) * FUSION_SIZE_SLICE_1_P6] * sm_b[2][idx_h3 + (idx_h2 + (2) * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3];
			reg_singles[2][3] += sm_a[0][idx_p6 + (idx_h1) * FUSION_SIZE_SLICE_1_P6] * sm_b[2][idx_h3 + (idx_h2 + (3) * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3];

			reg_singles[3][0] += sm_a[0][idx_p6 + (idx_h1) * FUSION_SIZE_SLICE_1_P6] * sm_b[3][idx_h3 + (idx_h2 + (0) * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3];
			reg_singles[3][1] += sm_a[0][idx_p6 + (idx_h1) * FUSION_SIZE_SLICE_1_P6] * sm_b[3][idx_h3 + (idx_h2 + (1) * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3];
			reg_singles[3][2] += sm_a[0][idx_p6 + (idx_h1) * FUSION_SIZE_SLICE_1_P6] * sm_b[3][idx_h3 + (idx_h2 + (2) * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3];
			reg_singles[3][3] += sm_a[0][idx_p6 + (idx_h1) * FUSION_SIZE_SLICE_1_P6] * sm_b[3][idx_h3 + (idx_h2 + (3) * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3];
			__syncthreads();
		}
		
		//
		//  >> s1_8:   t3[h3,h2,h1,p6,p5,p4] -= t2[p6,h2] * v2[h3,h1,p5,p4] (h3,h2,p6), (h1)
		//
		if (flag_s1_8 >= 0)	// these if-conditions make 100 ms..
		{
			// 
			double* tmp_dev_s1_t1_8 = df_dev_s1_t1_all + size_max_dim_s1_t1 * flag_s1_8;
			double* tmp_dev_s1_v2_8 = df_dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_8;

			if (idx_h3 < rng_p6 && idx_h2 < rng_h2 && idx_p6 == 0 && idx_h1 == 0)
			sm_a[0][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_1_P6] = tmp_dev_s1_t1_8[str_blk_idx_p6 + idx_h3 + (str_blk_idx_h2 + idx_h2) * base_size_p6b];
					
			if (idx_h3 < rng_h3 && idx_h2 < rng_h1 && idx_p6 < rng_p5 && idx_h1 < rng_p4)
			sm_b[idx_h1][idx_h3 + (idx_h2 + (idx_p6) * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H3] = tmp_dev_s1_v2_8[str_blk_idx_h3 + idx_h3 + (str_blk_idx_h1 + idx_h2 + (str_blk_idx_p5 + idx_p6 + (str_blk_idx_p4 + idx_h1) * base_size_p5b) * base_size_h1b) * base_size_h3b];
			__syncthreads();

			//  "p4" x "p5"
			reg_singles[0][0] -= sm_a[0][idx_p6 + (idx_h2) * FUSION_SIZE_SLICE_1_P6] * sm_b[0][idx_h3 + (idx_h1 + (0) * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H3];
			reg_singles[0][1] -= sm_a[0][idx_p6 + (idx_h2) * FUSION_SIZE_SLICE_1_P6] * sm_b[0][idx_h3 + (idx_h1 + (1) * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H3];
			reg_singles[0][2] -= sm_a[0][idx_p6 + (idx_h2) * FUSION_SIZE_SLICE_1_P6] * sm_b[0][idx_h3 + (idx_h1 + (2) * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H3];
			reg_singles[0][3] -= sm_a[0][idx_p6 + (idx_h2) * FUSION_SIZE_SLICE_1_P6] * sm_b[0][idx_h3 + (idx_h1 + (3) * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H3];

			reg_singles[1][0] -= sm_a[0][idx_p6 + (idx_h2) * FUSION_SIZE_SLICE_1_P6] * sm_b[1][idx_h3 + (idx_h1 + (0) * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H3];
			reg_singles[1][1] -= sm_a[0][idx_p6 + (idx_h2) * FUSION_SIZE_SLICE_1_P6] * sm_b[1][idx_h3 + (idx_h1 + (1) * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H3];
			reg_singles[1][2] -= sm_a[0][idx_p6 + (idx_h2) * FUSION_SIZE_SLICE_1_P6] * sm_b[1][idx_h3 + (idx_h1 + (2) * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H3];
			reg_singles[1][3] -= sm_a[0][idx_p6 + (idx_h2) * FUSION_SIZE_SLICE_1_P6] * sm_b[1][idx_h3 + (idx_h1 + (3) * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H3];

			reg_singles[2][0] -= sm_a[0][idx_p6 + (idx_h2) * FUSION_SIZE_SLICE_1_P6] * sm_b[2][idx_h3 + (idx_h1 + (0) * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H3];
			reg_singles[2][1] -= sm_a[0][idx_p6 + (idx_h2) * FUSION_SIZE_SLICE_1_P6] * sm_b[2][idx_h3 + (idx_h1 + (1) * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H3];
			reg_singles[2][2] -= sm_a[0][idx_p6 + (idx_h2) * FUSION_SIZE_SLICE_1_P6] * sm_b[2][idx_h3 + (idx_h1 + (2) * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H3];
			reg_singles[2][3] -= sm_a[0][idx_p6 + (idx_h2) * FUSION_SIZE_SLICE_1_P6] * sm_b[2][idx_h3 + (idx_h1 + (3) * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H3];

			reg_singles[3][0] -= sm_a[0][idx_p6 + (idx_h2) * FUSION_SIZE_SLICE_1_P6] * sm_b[3][idx_h3 + (idx_h1 + (0) * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H3];
			reg_singles[3][1] -= sm_a[0][idx_p6 + (idx_h2) * FUSION_SIZE_SLICE_1_P6] * sm_b[3][idx_h3 + (idx_h1 + (1) * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H3];
			reg_singles[3][2] -= sm_a[0][idx_p6 + (idx_h2) * FUSION_SIZE_SLICE_1_P6] * sm_b[3][idx_h3 + (idx_h1 + (2) * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H3];
			reg_singles[3][3] -= sm_a[0][idx_p6 + (idx_h2) * FUSION_SIZE_SLICE_1_P6] * sm_b[3][idx_h3 + (idx_h1 + (3) * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H3];
			__syncthreads();
		}

		//
		//  >> s1_9:   t3[h3,h2,h1,p6,p5,p4] -= t2[p6,h3] * v2[h2,h1,p5,p4] (h3,h2,p6), (h1)
		//
		if (flag_s1_9 >= 0)	// these if-conditions make 100 ms..
		{
			// 
			double* tmp_dev_s1_t1_9 = df_dev_s1_t1_all + size_max_dim_s1_t1 * flag_s1_9;
			double* tmp_dev_s1_v2_9 = df_dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_9;

			if (idx_h3 < rng_p6 && idx_h2 < rng_h3 && idx_p6 == 0 && idx_h1 == 0)
			sm_a[0][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_1_P6] = tmp_dev_s1_t1_9[str_blk_idx_p6 + idx_h3 + (str_blk_idx_h3 + idx_h2) * base_size_p6b];

			if (idx_h3 < rng_h2 && idx_h2 < rng_h1 && idx_p6 < rng_p5 && idx_h1 < rng_p4)
			sm_b[idx_h1][idx_h3 + (idx_h2 + (idx_p6) * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H2] = tmp_dev_s1_v2_9[str_blk_idx_h2 + idx_h3 + (str_blk_idx_h1 + idx_h2 + (str_blk_idx_p5 + idx_p6 + (str_blk_idx_p4 + idx_h1) * base_size_p5b) * base_size_h1b) * base_size_h2b];
			__syncthreads();

			//  "p4" x "p5"
			reg_singles[0][0] += sm_a[0][idx_p6 + (idx_h3) * FUSION_SIZE_SLICE_1_P6] * sm_b[0][idx_h2 + (idx_h1 + (0) * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H2];
			reg_singles[0][1] += sm_a[0][idx_p6 + (idx_h3) * FUSION_SIZE_SLICE_1_P6] * sm_b[0][idx_h2 + (idx_h1 + (1) * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H2];
			reg_singles[0][2] += sm_a[0][idx_p6 + (idx_h3) * FUSION_SIZE_SLICE_1_P6] * sm_b[0][idx_h2 + (idx_h1 + (2) * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H2];
			reg_singles[0][3] += sm_a[0][idx_p6 + (idx_h3) * FUSION_SIZE_SLICE_1_P6] * sm_b[0][idx_h2 + (idx_h1 + (3) * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H2];

			reg_singles[1][0] += sm_a[0][idx_p6 + (idx_h3) * FUSION_SIZE_SLICE_1_P6] * sm_b[1][idx_h2 + (idx_h1 + (0) * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H2];
			reg_singles[1][1] += sm_a[0][idx_p6 + (idx_h3) * FUSION_SIZE_SLICE_1_P6] * sm_b[1][idx_h2 + (idx_h1 + (1) * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H2];
			reg_singles[1][2] += sm_a[0][idx_p6 + (idx_h3) * FUSION_SIZE_SLICE_1_P6] * sm_b[1][idx_h2 + (idx_h1 + (2) * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H2];
			reg_singles[1][3] += sm_a[0][idx_p6 + (idx_h3) * FUSION_SIZE_SLICE_1_P6] * sm_b[1][idx_h2 + (idx_h1 + (3) * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H2];

			reg_singles[2][0] += sm_a[0][idx_p6 + (idx_h3) * FUSION_SIZE_SLICE_1_P6] * sm_b[2][idx_h2 + (idx_h1 + (0) * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H2];
			reg_singles[2][1] += sm_a[0][idx_p6 + (idx_h3) * FUSION_SIZE_SLICE_1_P6] * sm_b[2][idx_h2 + (idx_h1 + (1) * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H2];
			reg_singles[2][2] += sm_a[0][idx_p6 + (idx_h3) * FUSION_SIZE_SLICE_1_P6] * sm_b[2][idx_h2 + (idx_h1 + (2) * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H2];
			reg_singles[2][3] += sm_a[0][idx_p6 + (idx_h3) * FUSION_SIZE_SLICE_1_P6] * sm_b[2][idx_h2 + (idx_h1 + (3) * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H2];

			reg_singles[3][0] += sm_a[0][idx_p6 + (idx_h3) * FUSION_SIZE_SLICE_1_P6] * sm_b[3][idx_h2 + (idx_h1 + (0) * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H2];
			reg_singles[3][1] += sm_a[0][idx_p6 + (idx_h3) * FUSION_SIZE_SLICE_1_P6] * sm_b[3][idx_h2 + (idx_h1 + (1) * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H2];
			reg_singles[3][2] += sm_a[0][idx_p6 + (idx_h3) * FUSION_SIZE_SLICE_1_P6] * sm_b[3][idx_h2 + (idx_h1 + (2) * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H2];
			reg_singles[3][3] += sm_a[0][idx_p6 + (idx_h3) * FUSION_SIZE_SLICE_1_P6] * sm_b[3][idx_h2 + (idx_h1 + (3) * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H2];
			__syncthreads();
		}
	}


	// 
	if (idx_h3 < energy_rng_h3 && idx_h2 < energy_rng_h2 && idx_p6 < energy_rng_p6 && idx_h1 < energy_rng_h1)
	{
		for (int i = 0; i < FUSION_SIZE_SLICE_1_P5; i++)
		{
			for (int j = 0; j < FUSION_SIZE_SLICE_1_P4; j++)
			{
				if (i < energy_rng_p5 && j < energy_rng_p4)
				{
					// 
					double inner_factor = partial_inner_factor - dev_evl_sorted_p5b[i + (energy_str_blk_idx_p5)] - dev_evl_sorted_p4b[j + (energy_str_blk_idx_p4)];

					// 
					energy_1 += (reg_tile[j][i] * reg_tile[j][i]) / inner_factor;
					energy_2 += (reg_tile[j][i] * (reg_tile[j][i] + reg_singles[j][i])) / inner_factor;
				}
			}
		}
	}
	__syncthreads();
	
	// 
	//  to partially reduce the energies--- E(4) and E(5)
	//  a warp: 32 -(1)-> 16 -(2)-> 8 -(3)-> 4 -(4)-> 2 
	// 
	for (int offset = 16; offset > 0; offset /= 2)
	{
		energy_1 += __shfl_down_sync(FULL_MASK, energy_1, offset);
		energy_2 += __shfl_down_sync(FULL_MASK, energy_2, offset);
	}

	if (threadIdx.x == 0 && threadIdx.y % 2 == 0)
	{
		sm_a[0][threadIdx.y / 2] = energy_1;
		sm_b[0][threadIdx.y / 2] = energy_2;
	}
	__syncthreads();

	// 
	double final_energy_1 = 0.0;
	double final_energy_2 = 0.0;
	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		for (int i = 0; i < 8; i++)
		{
			final_energy_1 += sm_a[0][i];
			final_energy_2 += sm_b[0][i];
		}
		reduced_energy[blockIdx.x]              = final_energy_1;
		reduced_energy[blockIdx.x + gridDim.x]  = final_energy_2;
	}
}

void fully_fused_ccsd_t_gpu(cudaStream_t* stream_id, size_t num_blocks, 
	size_t base_size_h1b, size_t base_size_h2b, size_t base_size_h3b, 
	size_t base_size_p4b, size_t base_size_p5b, size_t base_size_p6b,
	// 
	double* df_dev_d1_t2_all, double* df_dev_d1_v2_all,
	double* df_dev_d2_t2_all, double* df_dev_d2_v2_all,
	double* df_dev_s1_t1_all, double* df_dev_s1_v2_all,
	// 
	size_t size_d1_t2_all, size_t size_d1_v2_all,
	size_t size_d2_t2_all, size_t size_d2_v2_all,
	size_t size_s1_t1_all, size_t size_s1_v2_all,
	// 
	int* host_d1_size, int* host_d1_exec, 	// used
	int* host_d2_size, int* host_d2_exec, 
	int* host_s1_size, int* host_s1_exec, 
	// 
	size_t size_noab, size_t size_max_dim_d1_t2, size_t size_max_dim_d1_v2,
	size_t size_nvab, size_t size_max_dim_d2_t2, size_t size_max_dim_d2_v2,
										size_t size_max_dim_s1_t1, size_t size_max_dim_s1_v2, 
	// 
	double factor, 
	// 
	double* dev_evl_sorted_h1b, double* dev_evl_sorted_h2b, double* dev_evl_sorted_h3b, 
	double* dev_evl_sorted_p4b, double* dev_evl_sorted_p5b, double* dev_evl_sorted_p6b,
	double* partial_energies,
	gpuEvent_t done_compute, gpuEvent_t done_copy) 
{
	// 	
	// 	to handle constant memories
	// 
	cudaMemcpyToSymbolAsync(const_df_s1_size, host_s1_size, sizeof(int) * (6), 0, cudaMemcpyHostToDevice, *stream_id);
	cudaMemcpyToSymbolAsync(const_df_s1_exec, host_s1_exec, sizeof(int) * (9), 0, cudaMemcpyHostToDevice, *stream_id);

	cudaMemcpyToSymbolAsync(const_df_d1_size, host_d1_size, sizeof(int) * (7 * size_noab), 0, cudaMemcpyHostToDevice, *stream_id);
	cudaMemcpyToSymbolAsync(const_df_d1_exec, host_d1_exec, sizeof(int) * (9 * size_noab), 0, cudaMemcpyHostToDevice, *stream_id);

	cudaMemcpyToSymbolAsync(const_df_d2_size, host_d2_size, sizeof(int) * (7 * size_nvab), 0, cudaMemcpyHostToDevice, *stream_id);
	cudaMemcpyToSymbolAsync(const_df_d2_exec, host_d2_exec, sizeof(int) * (9 * size_nvab), 0, cudaMemcpyHostToDevice, *stream_id);

	cudaEventRecord(done_copy);

	// 
	// 	Depends on # of Fused Kernel
	// 
	dim3 gridsize_1(num_blocks);
	dim3 blocksize_1(FUSION_SIZE_TB_1_X, FUSION_SIZE_TB_1_Y);

#ifdef DEBUG_PRINT_OLD_KERNEL_TIME
  cudaEvent_t start_kernel;
  cudaEvent_t stop_kernel;
  cudaEventCreate(&start_kernel);
  cudaEventCreate(&stop_kernel);
  cudaEventRecord(start_kernel);
#endif

	// printf ("[old] s1: %d,%d,%d/%d,%d,%d/%d,%d,%d\n", host_s1_exec[0], host_s1_exec[1], host_s1_exec[2], host_s1_exec[3], host_s1_exec[4], host_s1_exec[5], host_s1_exec[6], host_s1_exec[7], host_s1_exec[8]);
	// for (int i = 0; i < (int)size_noab; i++) {
	// 	printf ("[old] noab: %d, d1: %d,%d,%d/%d,%d,%d/%d,%d,%d\n", i, 
	// 	host_d1_exec[0 + (i) * 9], host_d1_exec[1 + (i) * 9], host_d1_exec[2 + (i) * 9],
	// 	host_d1_exec[3 + (i) * 9], host_d1_exec[4 + (i) * 9], host_d1_exec[5 + (i) * 9],
	// 	host_d1_exec[6 + (i) * 9], host_d1_exec[7 + (i) * 9], host_d1_exec[8 + (i) * 9]);
	// }

	// for (int i = 0; i < (int)size_nvab; i++) {
	// 	printf ("[old] nvab: %d, d2: %d,%d,%d/%d,%d,%d/%d,%d,%d\n", i, 
	// 	host_d2_exec[0 + (i) * 9], host_d2_exec[1 + (i) * 9], host_d2_exec[2 + (i) * 9],
	// 	host_d2_exec[3 + (i) * 9], host_d2_exec[4 + (i) * 9], host_d2_exec[5 + (i) * 9],
	// 	host_d2_exec[6 + (i) * 9], host_d2_exec[7 + (i) * 9], host_d2_exec[8 + (i) * 9]);
	// }

	// // 	
	// 	to call the fused kernel for singles, doubles and energies.
	// 
	// jk_ccsd_t_fully_fused_kernel_associative
	revised_jk_ccsd_t_fully_fused_kernel<<<gridsize_1, blocksize_1, 0, *stream_id>>>((int)size_noab, (int)size_nvab, 
											// 
											(int)size_max_dim_s1_t1, (int)size_max_dim_s1_v2,
											(int)size_max_dim_d1_t2, (int)size_max_dim_d1_v2,
											(int)size_max_dim_d2_t2, (int)size_max_dim_d2_v2,
											// 
											df_dev_d1_t2_all, df_dev_d1_v2_all, 
											df_dev_d2_t2_all, df_dev_d2_v2_all, 
											df_dev_s1_t1_all, df_dev_s1_v2_all,
											//  
											dev_evl_sorted_h1b, dev_evl_sorted_h2b, dev_evl_sorted_h3b,
											dev_evl_sorted_p4b, dev_evl_sorted_p5b, dev_evl_sorted_p6b,
											// 
											partial_energies, 
											// 
											CEIL(base_size_h3b, FUSION_SIZE_SLICE_1_H3), CEIL(base_size_h2b, FUSION_SIZE_SLICE_1_H2), CEIL(base_size_h1b, FUSION_SIZE_SLICE_1_H1), 
											CEIL(base_size_p6b, FUSION_SIZE_SLICE_1_P6), CEIL(base_size_p5b, FUSION_SIZE_SLICE_1_P5), CEIL(base_size_p4b, FUSION_SIZE_SLICE_1_P4),
											// 
											(int)base_size_h1b, (int)base_size_h2b, (int)base_size_h3b, 
											(int)base_size_p4b, (int)base_size_p5b, (int)base_size_p6b);	

#ifdef DEBUG_PRINT_OLD_KERNEL_TIME
	cudaEventRecord(stop_kernel);
	cudaEventSynchronize(stop_kernel);
	float kernel_ms = 0;
	cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel);
	printf ("[%s] kernel: %f (ms)\n", __func__, kernel_ms);
#endif
}
// end of (1) Pure FP64

// (2) 3rd. Generation Tensor Cores (FP64)
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>

#include "tensor_core_helper.cuh"

using namespace std;

#define CUCHK(call) {	\
	cudaError_t err = call; \
	if( cudaSuccess != err) {	\
		fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",	\
				__FILE__, __LINE__, cudaGetErrorString(err) );	\
		fflush(stderr); \
		exit(EXIT_FAILURE);	\
}}

//
#define SIZE_TILE_P7 16
#define SIZE_TILE_H3 4
#define SIZE_TILE_P4 4
#define SIZE_TILE_H2 4
#define SIZE_TILE_H1 4
#define SIZE_TILE_P6 4
#define SIZE_TILE_P5 4

#define SIZE_UNIT_INT SIZE_TILE_P7

#define NUM_INDEX 		6
#define CEIL(a, b) 		(((a) + (b) - 1) / (b))

#define PAD 3
#define STAGE_ALIGN 32
#define SINGLE_STAGE_SIZE (64 * (PAD + 16))
#define STAGE_OFFSET ((SINGLE_STAGE_SIZE + STAGE_ALIGN - 1) / STAGE_ALIGN) * STAGE_ALIGN

#define NUM_STAGE 	2

#define NUM_ENERGY 	2
#define FULL_MASK 	0xffffffff

#define TEST_ENABLE_RT

// 
// 	helpers
// 
#define MAX_NOAB 30
#define MAX_NVAB 120

// 9 * (1 + MAX_NOAB + MAX_NVAB) + (MAX_NOAB + MAX_NVAB) * sizeof(int) <= 64KB
__constant__ int const_s1_exec[9];
__constant__ int const_d1_exec[9 * MAX_NOAB];
__constant__ int const_d2_exec[9 * MAX_NVAB];

__constant__ int const_d1_h7b[MAX_NOAB];
__constant__ int const_d2_p7b[MAX_NVAB];

//------------------------------------------------------------------------------ device helper fuctions
__device__ inline void zero_shared(double *smem, const int start_row, const int num_rows) {
	const int t_id = threadIdx.y * blockDim.x + threadIdx.x;
	const int col_idx = t_id % 64;
	const int row_idx = t_id / 64;
	const int row_inc = blockDim.x * blockDim.y / 64;
	for (int i = row_idx + start_row; i < num_rows; i += row_inc) {
		smem[col_idx * (16 + PAD) + i] = 0.0;
	}
}


// fixed (reg_x, reg_y)
__device__ inline void rt_store_fixed(double* smem, const int idx_x_1, const int idx_x_2, const int idx_y_1, const int idx_y_2, MmaOperandC& op_c) {
	#pragma unroll 4
	for (int i = 0; i < 4; i++) {
		#pragma unroll 4
		for (int j = 0; j < 4; j++) {
			smem[idx_x_1 + (idx_x_2 + (j) * 4) * 4 + (idx_y_1 + (idx_y_2 + (i) * 4) * 4) * 65] = op_c.reg[j + (i) * 4];
		}
	}
}

// fixed (reg_x, reg_y)
__device__ inline void rt_load_fixed(double* smem, const int idx_x_1, const int idx_x_2, const int idx_y_1, const int idx_y_2, MmaOperandC& op_c) {
	#pragma unroll 4
	for (int i = 0; i < 4; i++) {
		#pragma unroll 4
		for (int j = 0; j < 4; j++) {
			op_c.reg[j + (i) * 4] = smem[idx_x_1 + (idx_x_2 + (j) * 4) * 4 + (idx_y_1 + (idx_y_2 + (i) * 4) * 4) * 65];
		}
	}
}

#include "ccsd_t_g2s_device_functions.cu"

//------------------------------------------------------------------------------
// created by tc_gen_code_Kernel()
__global__ __launch_bounds__(256, 3)
void fully_fused_kernel_ccsd_t_nvidia_tc_fp64(int size_noab, int size_nvab, 
	// common
	int size_max_dim_s1_t1, int size_max_dim_s1_v2, 
	int size_max_dim_d1_t2, int size_max_dim_d1_v2, 
	int size_max_dim_d2_t2, int size_max_dim_d2_v2, 
	// 
	double* __restrict__ dev_s1_t1_all, double* __restrict__ dev_s1_v2_all, 
	double* __restrict__ dev_d1_t2_all, double* __restrict__ dev_d1_v2_all, 
	double* __restrict__ dev_d2_t2_all, double* __restrict__ dev_d2_v2_all, 
	// 
	double* dev_energy, 
	const double* dev_evl_sorted_h3b, const double* dev_evl_sorted_h2b, const double* dev_evl_sorted_h1b, 
	const double* dev_evl_sorted_p6b, const double* dev_evl_sorted_p5b, const double* dev_evl_sorted_p4b, 
	// 
	const int size_h3, const int size_h2, const int size_h1, 
	const int size_p6, const int size_p5, const int size_p4, 
	const int numBlk_h3, const int numBlk_h2, const int numBlk_h1, 
	const int numBlk_p6, const int numBlk_p5, const int numBlk_p4)
{
	auto grid = cooperative_groups::this_grid();
	auto block = cooperative_groups::this_thread_block();
	// For Shared Memory,
	const int lda = 16 + PAD;
	extern __shared__ double sm_block[];
	double *sm_a = reinterpret_cast<double *>(sm_block) + 0 * STAGE_OFFSET;
	double *sm_b = reinterpret_cast<double *>(sm_block) + NUM_STAGE * STAGE_OFFSET;

	#pragma unroll
	for (int i = 0; i < NUM_STAGE; i++) {
		zero_shared(sm_a + STAGE_OFFSET * i, 0, 16);
		zero_shared(sm_b + STAGE_OFFSET * i, 0, 16);
	}
	block.sync();

	// Allocate shared storage for a N-stage cuda::pipeline:
	cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

	const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
	const int warp_id = thread_id / 32; // 0:7
	WarpRegisterMapping wrm(thread_id);

	const int tile_m = warp_id % 2; // 0:1
	const int tile_n = warp_id / 2; // 0:3

	MmaOperandC op_c;
	MmaOperandC op_c_s;

	int internal_upperbound = 0;
	int internal_offset;

  //  
  //  based on sd2_1
  // 
	int idx_p6 = threadIdx.x % SIZE_TILE_P6; // this is not used for sd2. 
	int idx_h2 = threadIdx.x / SIZE_TILE_P6;
	int idx_h1 = threadIdx.y % SIZE_TILE_H1;
	int idx_h3 = threadIdx.y / SIZE_TILE_H1;

	int blk_idx_p4 = blockIdx.x / (numBlk_p5 * numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);
	int tmp_blkIdx = blockIdx.x % (numBlk_p5 * numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_p5 = tmp_blkIdx / (numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_p6 = tmp_blkIdx / (numBlk_h1 * numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_h1 = tmp_blkIdx / (numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h2 * numBlk_h3);

	int blk_idx_h2 = tmp_blkIdx / numBlk_h3;
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h3);

	int blk_idx_h3 = tmp_blkIdx;

	// need to support partial tiles
	int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;
	if ((size_h3 - (blk_idx_h3 * SIZE_TILE_H3)) >= SIZE_TILE_H3) { rng_h3 = SIZE_TILE_H3; }
	else { rng_h3 = size_h3 % SIZE_TILE_H3; }
	
  if ((size_h2 - (blk_idx_h2 * SIZE_TILE_H2)) >= SIZE_TILE_H2) { rng_h2 = SIZE_TILE_H2; }
	else { rng_h2 = size_h2 % SIZE_TILE_H2; }
	
  if ((size_h1 - (blk_idx_h1 * SIZE_TILE_H1)) >= SIZE_TILE_H1) { rng_h1 = SIZE_TILE_H1; }
	else { rng_h1 = size_h1 % SIZE_TILE_H1; }

	if ((size_p6 - (blk_idx_p6 * SIZE_TILE_P6)) >= SIZE_TILE_P6) { rng_p6 = SIZE_TILE_P6; }
	else { rng_p6 = size_p6 % SIZE_TILE_P6; }

	if ((size_p5 - (blk_idx_p5 * SIZE_TILE_P5)) >= SIZE_TILE_P5) { rng_p5 = SIZE_TILE_P5; }
	else { rng_p5 = size_p5 % SIZE_TILE_P5; }

	if ((size_p4 - (blk_idx_p4 * SIZE_TILE_P4)) >= SIZE_TILE_P4) { rng_p4 = SIZE_TILE_P4; }
	else { rng_p4 = size_p4 % SIZE_TILE_P4; }

  // 
	// const size_t num_batches = (size_internal + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;

	// TB_X(p4,h3), TB_Y(h1,h2)
	// if (idx_p6 < rng_p4 && idx_h2 < rng_h3 && idx_h1 < rng_h1 && idx_h3 < rng_h2) 
	double partial_inner_factor = dev_evl_sorted_h3b[blk_idx_h3 * SIZE_TILE_H3 + idx_h2] + dev_evl_sorted_h2b[blk_idx_h2 * SIZE_TILE_H2 + idx_h3] + dev_evl_sorted_h1b[blk_idx_h1 * SIZE_TILE_H1 + idx_h1] - dev_evl_sorted_p4b[blk_idx_p4 * SIZE_TILE_P4 + idx_p6];

	//
#if 1
	// sd2_1: t3[h3,h2,h1,p6,p5,p4] −= t2[p7,p4,h1,h2] * v2[p7,h3,p6,p5] --> TB_X(p6,h2), TB_Y(h1,h3), REG_X,Y(p5,p4)
	for (int iter_nvab = 0; iter_nvab < size_nvab; iter_nvab++) {
		int size_p7 	= const_d2_p7b[iter_nvab];
		int flag_d2_1 = const_d2_exec[0 + (iter_nvab) * 9];

		const size_t num_batches = (size_p7 + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;
		const size_t size_internal_up = ((size_p7 + 3) / 4) * 4;

		if (flag_d2_1 >= 0) {
			double* tmp_dev_d2_t2 = dev_d2_t2_all + size_max_dim_d2_t2 * flag_d2_1;
			double* tmp_dev_d2_v2 = dev_d2_v2_all + size_max_dim_d2_v2 * flag_d2_1;

			internal_upperbound = 0;
			#pragma unroll 1
			for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
				#pragma unroll 1
				for (; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE); ++fetch_batch) {
					pipeline.producer_acquire();

					const int l_fetch = fetch_batch * SIZE_UNIT_INT;
					const size_t shared_idx = fetch_batch % NUM_STAGE;
					// internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
					internal_offset = (l_fetch + SIZE_UNIT_INT) - size_p7;
					block.sync();

					if (internal_offset > 0) { 
						const int start_row = size_p7 - l_fetch;
						const int max_row = ((start_row+3)/4)*4;
						internal_upperbound = internal_offset;
						zero_shared(sm_a + STAGE_OFFSET * shared_idx, start_row, max_row); // Zero out shared memory if partial tile
						zero_shared(sm_b + STAGE_OFFSET * shared_idx, start_row, max_row);
						block.sync();
					}

					if ((idx_h3 < rng_h1) && (idx_h1 < rng_h2) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
						g2s_d2_t2_1<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, tmp_dev_d2_t2, 
							blk_idx_h2, 					idx_h1, 
							blk_idx_h1, size_h1, 	idx_h3, 
							blk_idx_p4, size_p4, 
													size_p7, 	threadIdx.x + l_fetch, rng_p4, pipeline);
					}

					if ((idx_h3 < rng_h3) && (idx_h1 < rng_p6) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
						g2s_d2_v2_1<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, tmp_dev_d2_v2, 
							blk_idx_p5, 
							blk_idx_p6, size_p6, idx_h1, 
							blk_idx_h3, size_h3, idx_h3, 
													size_p7, threadIdx.x + l_fetch, rng_p5, pipeline);
					}
					pipeline.producer_commit();
				}

				pipeline.consumer_wait();
				block.sync();
				const size_t shared_idx = compute_batch % NUM_STAGE;

				const int max_iter = (size_internal_up - (compute_batch * SIZE_UNIT_INT)) / 4;
				#pragma unroll 1
				for (int ll = 0; ll < 4 && ll < max_iter; ll++) {
					MmaOperandA op_a;
					op_a.template load<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
					MmaOperandB op_b;
					op_b.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
					mma(op_c, op_a, op_b);
				}
				pipeline.consumer_release();
			}
			block.sync(); 
		}
	}

	// sd1_6:  	t3[h3,h2,h1,p6,p5,p4] -= t2[h7,p5,p6,h3] * v2[h2,h1,p4,h7]
	// sd1_6': 	t3[h3,h2,h1,p6,p5,p4] -= v2[h2,h1,p4,h7] * t2[h7,p5,p6,h3] --> TB_X(p6,h2), TB_Y(h1,h3), REG_X,Y(p5,p4)
	for (int iter_noab = 0; iter_noab < size_noab; iter_noab++) {
		// 
		int size_h7 	= const_d1_h7b[iter_noab];
		int flag_d1_6 = const_d1_exec[5 + (iter_noab) * 9];
		
		const size_t num_batches = (size_h7 + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;
		const size_t size_internal_up = ((size_h7 + 3) / 4) * 4;

		if (flag_d1_6 >= 0) {
			double* tmp_dev_d1_t2 = dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_6;
			double* tmp_dev_d1_v2 = dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_6;

			internal_upperbound = 0;
			#pragma unroll 1
			for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
				#pragma unroll 1
				for (; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE); ++fetch_batch) {
					pipeline.producer_acquire();

					const int l_fetch = fetch_batch * SIZE_UNIT_INT;
					const size_t shared_idx = fetch_batch % NUM_STAGE;
					// internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
					internal_offset = (l_fetch + SIZE_UNIT_INT) - size_h7;
					block.sync();

					if (internal_offset > 0) { 
						const int start_row = size_h7 - l_fetch;
						const int max_row = ((start_row+3)/4)*4;
						internal_upperbound = internal_offset;
						zero_shared(sm_a + STAGE_OFFSET * shared_idx, start_row, max_row); // Zero out shared memory if partial tile
						zero_shared(sm_b + STAGE_OFFSET * shared_idx, start_row, max_row);
						block.sync();
					}

					if ((idx_h3 < rng_h3) && (idx_h1 < rng_p6) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
						g2s_d1_t2_6<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, tmp_dev_d1_t2, 
							blk_idx_h3, 					idx_h3, 
							blk_idx_p6, size_p6,  idx_h1,  	
							blk_idx_p5, size_p5,
													size_h7, 	threadIdx.x + l_fetch, 
													rng_p5, 	pipeline);
					}

					if ((idx_h2 < rng_h1) && (idx_p6 < rng_h2) && threadIdx.y < SIZE_UNIT_INT - internal_upperbound) {
						g2s_d1_v2_6<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, tmp_dev_d1_v2, 
							blk_idx_p4, size_p4,	
							blk_idx_h1, size_h1, 	idx_h2, 
							blk_idx_h2, size_h2, 	idx_p6, 
																		threadIdx.y + l_fetch, 
													rng_p4, 	pipeline);
					}
					pipeline.producer_commit();
				}
				pipeline.consumer_wait();
				block.sync();
				const size_t shared_idx = compute_batch % NUM_STAGE;

				#pragma unroll
				for (int ll = 0; ll < 4; ll++) {
					MmaOperandA op_a;
					op_a.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
					MmaOperandB op_b;
					op_b.template load<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
					mma(op_c, op_a, op_b);
				}
				pipeline.consumer_release();
			}
			block.sync(); 
		}
	}
#endif

	// rt #1
	// from TB_X(p6,h2), TB_Y(h1,h3), REG_X,Y(p5,p4) // d2_1
	// to 	TB_X(p6,h3), TB_Y(h2,h1), REG_X,Y(p5,p4) // d2_2
#ifdef TEST_ENABLE_RT
	rt_store_fixed(sm_block, idx_p6, idx_h2, idx_h1, idx_h3, op_c);
	block.sync();
	rt_load_fixed(sm_block, idx_p6, idx_h1, idx_h3, idx_h2, op_c);
	block.sync();
#endif

#if 1
	// sd2_2: t3[h3,h2,h1,p6,p5,p4] -= t2[p7,p4,h2,h3] * v2[p7,h1,p6,p5]
	// t2[p7,ry,h2,h3] * v2[p7,h1,p6,rx] -> TB_X(p6,h3), TB_Y(h2,h1), REG_X,Y(p5,p4)	
	for (int iter_nvab = 0; iter_nvab < size_nvab; iter_nvab++) {	
		int size_p7 	= const_d2_p7b[iter_nvab];
		int flag_d2_2 = const_d2_exec[1 + (iter_nvab) * 9];

		const size_t num_batches = (size_p7 + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;
		const size_t size_internal_up = ((size_p7 + 3) / 4) * 4;

		if (flag_d2_2 >= 0) {
			double* tmp_dev_d2_t2 = dev_d2_t2_all + size_max_dim_d2_t2 * flag_d2_2;
			double* tmp_dev_d2_v2 = dev_d2_v2_all + size_max_dim_d2_v2 * flag_d2_2;

			// 
			internal_upperbound = 0;
			#pragma unroll 1
			for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
				#pragma unroll 1
				for (; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE); ++fetch_batch) {
					pipeline.producer_acquire();

					const int l_fetch = fetch_batch * SIZE_UNIT_INT;
					const size_t shared_idx = fetch_batch % NUM_STAGE;
					// internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
					internal_offset = (l_fetch + SIZE_UNIT_INT) - size_p7;
					block.sync();

					if (internal_offset > 0) { 
						const int start_row = size_p7 - l_fetch;
						const int max_row = ((start_row+3)/4)*4;
						internal_upperbound = internal_offset;
						zero_shared(sm_a + STAGE_OFFSET * shared_idx, start_row, max_row); // Zero out shared memory if partial tile
						zero_shared(sm_b + STAGE_OFFSET * shared_idx, start_row, max_row);
						block.sync();
					}

					if ((idx_h3 < rng_h2) && (idx_h1 < rng_h3) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
						g2s_d2_t2_2<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, tmp_dev_d2_t2, 
							blk_idx_h3, 					idx_h1, 
							blk_idx_h2, size_h2, 	idx_h3, 
							blk_idx_p4, size_p4,  
													size_p7, threadIdx.x + l_fetch, rng_p4, pipeline);
					}

					if ((idx_h3 < rng_h1) && (idx_h1 < rng_p6) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
						g2s_d2_v2_2<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, tmp_dev_d2_v2, 
							blk_idx_p5, 
							blk_idx_p6, size_p6, idx_h1, 
							blk_idx_h1, size_h1, idx_h3, 
													size_p7, threadIdx.x + l_fetch, rng_p5, pipeline);
					}
					pipeline.producer_commit();
				}

				pipeline.consumer_wait();
				block.sync();
				const size_t shared_idx = compute_batch % NUM_STAGE;
				
				const int max_iter = (size_internal_up - (compute_batch * SIZE_UNIT_INT)) / 4;
				#pragma unroll 1
				for (int ll = 0; ll < 4 && ll < max_iter; ll++) {
					MmaOperandA op_a;
					op_a.template load<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
					MmaOperandB op_b;
					op_b.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
					mma(op_c, op_a, op_b);
				}
				pipeline.consumer_release();
			}
			block.sync(); 
		}
	}

	// sd1_4: 	t3[h3,h2,h1,p6,p5,p4] -= t2[h7,p5,p6,h1] * v2[h3,h2,p4,h7]
	// sd1_4': 	t3[h3,h2,h1,p6,p5,p4] -= v2[h3,h2,p4,h7] * t2[h7,p5,p6,h1] --> TB_X(p6,h3), TB_Y(h2,h1), REG_X,Y(p5,p4)
	for (int iter_noab = 0; iter_noab < size_noab; iter_noab++) {
		// 
		int size_h7 	= const_d1_h7b[iter_noab];
		int flag_d1_4 = const_d1_exec[3 + (iter_noab) * 9];
		
		const size_t num_batches = (size_h7 + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;
		const size_t size_internal_up = ((size_h7 + 3) / 4) * 4;

		if (flag_d1_4 >= 0) {
			double* tmp_dev_d1_t2 = dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_4;
			double* tmp_dev_d1_v2 = dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_4;

			// 
			internal_upperbound = 0;
			#pragma unroll 1
			for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
				#pragma unroll 1
				for (; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE); ++fetch_batch) {
					pipeline.producer_acquire();

					const int l_fetch = fetch_batch * SIZE_UNIT_INT;
					const size_t shared_idx = fetch_batch % NUM_STAGE;
					// internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
					internal_offset = (l_fetch + SIZE_UNIT_INT) - size_h7;
					block.sync();

					if (internal_offset > 0) { 
						const int start_row = size_h7 - l_fetch;
						const int max_row = ((start_row+3)/4)*4;
						internal_upperbound = internal_offset;
						zero_shared(sm_a + STAGE_OFFSET * shared_idx, start_row, max_row); // Zero out shared memory if partial tile
						zero_shared(sm_b + STAGE_OFFSET * shared_idx, start_row, max_row);
						block.sync();
					}

					if ((idx_h3 < rng_h1) && (idx_h1 < rng_p6) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
						g2s_d1_t2_4<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, tmp_dev_d1_t2, 
							blk_idx_h1, 					idx_h3, 
							blk_idx_p6, size_p6,  idx_h1,  	
							blk_idx_p5, size_p5,
													size_h7, 	threadIdx.x + l_fetch, 
													rng_p5, 	pipeline);
					}

					if ((idx_h2 < rng_h2) && (idx_p6 < rng_h3) && threadIdx.y < SIZE_UNIT_INT - internal_upperbound) {
						g2s_d1_v2_4<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, tmp_dev_d1_v2, 
							blk_idx_p4, size_p4,	
							blk_idx_h2, size_h2, 	idx_h2, 
							blk_idx_h3, size_h3, 	idx_p6, 
																		threadIdx.y + l_fetch, 
													rng_p4, 	pipeline);
					}
					pipeline.producer_commit();
				}
				pipeline.consumer_wait();
				block.sync();
				const size_t shared_idx = compute_batch % NUM_STAGE;

				const int max_iter = (size_internal_up - (compute_batch * SIZE_UNIT_INT)) / 4;
				#pragma unroll 1
				for (int ll = 0; ll < 4 && ll < max_iter; ll++) {
					MmaOperandA op_a;
					op_a.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
					MmaOperandB op_b;
					op_b.template load<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
					mma(op_c, op_a, op_b);
				}
				pipeline.consumer_release();
			}
			block.sync(); 
		}
	}
#endif

	// rt 	TB_X(p6,h2), TB_Y(h1,h3) #2
	// from TB_X(p6,h3), TB_Y(h2,h1), REG_X,Y(p5,p4) // d2_2
	// to 	TB_X(p6,h3), TB_Y(h1,h2), REG_X,Y(p5,p4) // d2_3
#ifdef TEST_ENABLE_RT
	rt_store_fixed(sm_block, idx_p6, idx_h2, idx_h1, idx_h3, op_c); // x=[(p6,h3),(p5)], y=[(h2,h1),(p4)] -> x=[(p4,h2),(p6)], y=[(h1,h3),(p5)]
	block.sync();
	rt_load_fixed(sm_block, idx_p6, idx_h2, idx_h3, idx_h1, op_c);
	block.sync();
#endif

#if 1
	// sd2_3: t3[h3,h2,h1,p6,p5,p4] += t2[p7,p4,h1,h3] * v2[p7,h2,p6,p5]
	// t2[p7,ry,h1,h3] * v2[p7,h2,p6,rx] -> TB_X(p6,h3), TB_Y(h1,h2)
	for (int iter_nvab = 0; iter_nvab < size_nvab; iter_nvab++) {	
		int size_p7 	= const_d2_p7b[iter_nvab];
		int flag_d2_3 = const_d2_exec[2 + (iter_nvab) * 9];

		const size_t num_batches = (size_p7 + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;
		const size_t size_internal_up = ((size_p7 + 3) / 4) * 4;

		if (flag_d2_3 >= 0) {
			double* tmp_dev_d2_t2 = dev_d2_t2_all + size_max_dim_d2_t2 * flag_d2_3;
			double* tmp_dev_d2_v2 = dev_d2_v2_all + size_max_dim_d2_v2 * flag_d2_3;

			// 
			internal_upperbound = 0;
			#pragma unroll 1
			for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
				#pragma unroll 1
				for (; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE); ++fetch_batch) {
					pipeline.producer_acquire();

					const int l_fetch = fetch_batch * SIZE_UNIT_INT;
					const size_t shared_idx = fetch_batch % NUM_STAGE;
					// internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
					internal_offset = (l_fetch + SIZE_UNIT_INT) - size_p7;
					block.sync();

					if (internal_offset > 0) { 
						const int start_row = size_p7 - l_fetch;
						const int max_row = ((start_row+3)/4)*4;
						internal_upperbound = internal_offset;
						zero_shared(sm_a + STAGE_OFFSET * shared_idx, start_row, max_row); // Zero out shared memory if partial tile
						zero_shared(sm_b + STAGE_OFFSET * shared_idx, start_row, max_row);
						block.sync();
					}

					if ((idx_h3 < rng_h1) && (idx_h1 < rng_h3) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
						g2s_d2_t2_3<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, tmp_dev_d2_t2,  
							blk_idx_h3, 					idx_h1, 
							blk_idx_h1, size_h1, 	idx_h3, 
							blk_idx_p4, size_p4,  
													size_p7, 	threadIdx.x + l_fetch, rng_p4, pipeline);
					}

					if ((idx_h3 < rng_h2) && (idx_h1 < rng_p6) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
						g2s_d2_v2_3<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, tmp_dev_d2_v2, 
							blk_idx_p5, 				 	
							blk_idx_p6, size_p6, 	idx_h1, 
							blk_idx_h2, size_h2, 	idx_h3, 
													size_p7, 	threadIdx.x + l_fetch, rng_p5, pipeline);
					}
					pipeline.producer_commit();
				}

				pipeline.consumer_wait();
				block.sync();
				const size_t shared_idx = compute_batch % NUM_STAGE;
				
				const int max_iter = (size_internal_up - (compute_batch * SIZE_UNIT_INT)) / 4;
				#pragma unroll 1
				for (int ll = 0; ll < 4 && ll < max_iter; ll++) {
					MmaOperandA op_a;
					op_a.template load_plus<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
					MmaOperandB op_b;
					op_b.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
					mma(op_c, op_a, op_b);
				}
				pipeline.consumer_release();
			}
			block.sync(); 
		}
	}

	// sd1_5: 	t3[h3,h2,h1,p6,p5,p4] += t2[h7,p5,p6,h2] * v2[h3,h1,p4,h7]
	// sd1_5': 	t3[h3,h2,h1,p6,p5,p4] += v2[h3,h1,p4,h7] * t2[h7,p5,p6,h2] --> TB_X(p6,h3), TB_Y(h1,h2),  REG_X,Y(p5,p4)
	for (int iter_noab = 0; iter_noab < size_noab; iter_noab++) {
		// 
		int size_h7 	= const_d1_h7b[iter_noab];
		int flag_d1_5 = const_d1_exec[4 + (iter_noab) * 9];
		
		const size_t num_batches = (size_h7 + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;
		const size_t size_internal_up = ((size_h7 + 3) / 4) * 4;

		if (flag_d1_5 >= 0) {
			double* tmp_dev_d1_t2 = dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_5;
			double* tmp_dev_d1_v2 = dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_5;

			// 
			internal_upperbound = 0;
			#pragma unroll 1
			for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
				#pragma unroll 1
				for (; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE); ++fetch_batch) {
					pipeline.producer_acquire();

					const int l_fetch = fetch_batch * SIZE_UNIT_INT;
					const size_t shared_idx = fetch_batch % NUM_STAGE;
					// internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
					internal_offset = (l_fetch + SIZE_UNIT_INT) - size_h7;
					block.sync();

					if (internal_offset > 0) { 
						const int start_row = size_h7 - l_fetch;
						const int max_row = ((start_row+3)/4)*4;
						internal_upperbound = internal_offset;
						zero_shared(sm_a + STAGE_OFFSET * shared_idx, start_row, max_row); // Zero out shared memory if partial tile
						zero_shared(sm_b + STAGE_OFFSET * shared_idx, start_row, max_row);
						block.sync();
					}

					if ((idx_h3 < rng_h2) && (idx_h1 < rng_p6) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
						g2s_d1_t2_5<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, tmp_dev_d1_t2, 
							blk_idx_h2, 					idx_h3, 
							blk_idx_p6, size_p6,  idx_h1,  	
							blk_idx_p5, size_p5,
													size_h7, 	threadIdx.x + l_fetch, 
													rng_p5, 	pipeline);
					}

					if ((idx_h2 < rng_h1) && (idx_p6 < rng_h3) && threadIdx.y < SIZE_UNIT_INT - internal_upperbound) {
						g2s_d1_v2_5<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, tmp_dev_d1_v2, 
							blk_idx_p4, size_p4,	
							blk_idx_h1, size_h1, 	idx_h2, 
							blk_idx_h3, size_h3, 	idx_p6, 
																		threadIdx.y + l_fetch, 
													rng_p4, 	pipeline);
					}
					pipeline.producer_commit();
				}
				pipeline.consumer_wait();
				block.sync();
				const size_t shared_idx = compute_batch % NUM_STAGE;

				const int max_iter = (size_internal_up - (compute_batch * SIZE_UNIT_INT)) / 4;
				#pragma unroll 1
				for (int ll = 0; ll < 4 && ll < max_iter; ll++) {
					MmaOperandA op_a;
					op_a.template load_plus<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
					MmaOperandB op_b;
					op_b.template load<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
					mma(op_c, op_a, op_b);
				}
				pipeline.consumer_release();
			}
			block.sync(); 
		}
	}
#endif

	// rt 	TB_X(p6,h2), TB_Y(h1,h3) #3
	// from TB_X(p6,h3), TB_Y(h1,h2), REG_X,Y(p5,p4) // d2_3
	// to 	TB_X(p6,h2), TB_Y(h1,h3), REG_X,Y(p4,p5) // d2_4 
#ifdef TEST_ENABLE_RT
	rt_store_fixed(sm_block, idx_p6, idx_h2, idx_h1, idx_h3, op_c); // x=[(p6,h3),(p5)], y=[(h1,h2),(p4)] -> x=[(p4,h2),(p6)], y=[(h1,h3),(p5)]
	block.sync();
	#pragma unroll 4
	for (int i = 0; i < 4; i++) { // p5
		#pragma unroll 4
		for (int j = 0; j < 4; j++) { // p4
			op_c.reg[j + (i) * 4] = sm_block[idx_p6 + (idx_h3 + (i) * 4) * 4 + 
																			(idx_h1 + (idx_h2 + (j) * 4) * 4) * 65];
		}
	}
	block.sync();
#endif

#if 1
	// 
	// sd2_4: t3[h3,h2,h1,p6,p5,p4] += t2[p7,p5,h1,h2] * v2[p7,h3,p6,p4] -> TB_X(p6,h2), TB_Y(h1,h3), REG_X,Y(p4,p5)
	for (int iter_nvab = 0; iter_nvab < size_nvab; iter_nvab++) {	
		int size_p7 	= const_d2_p7b[iter_nvab];
		int flag_d2_4 = const_d2_exec[3 + (iter_nvab) * 9];

		const size_t num_batches = (size_p7 + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;
		const size_t size_internal_up = ((size_p7 + 3) / 4) * 4;

		if (flag_d2_4 >= 0) {
			double* tmp_dev_d2_t2 = dev_d2_t2_all + size_max_dim_d2_t2 * flag_d2_4;
			double* tmp_dev_d2_v2 = dev_d2_v2_all + size_max_dim_d2_v2 * flag_d2_4;

			internal_upperbound = 0;
			#pragma unroll 1
			for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
				#pragma unroll 1
				for (; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE); ++fetch_batch) {
					pipeline.producer_acquire();

					const int l_fetch = fetch_batch * SIZE_UNIT_INT;
					const size_t shared_idx = fetch_batch % NUM_STAGE;
					// internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
					internal_offset = (l_fetch + SIZE_UNIT_INT) - size_p7;
					block.sync();

					if (internal_offset > 0) { 
						const int start_row = size_p7 - l_fetch;
						const int max_row = ((start_row+3)/4)*4;
						internal_upperbound = internal_offset;
						zero_shared(sm_a + STAGE_OFFSET * shared_idx, start_row, max_row); // Zero out shared memory if partial tile
						zero_shared(sm_b + STAGE_OFFSET * shared_idx, start_row, max_row);
						block.sync();
					}

					if ((idx_h3 < rng_h1) && (idx_h1 < rng_h2) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
						g2s_d2_t2_4<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, tmp_dev_d2_t2,  
							blk_idx_h2, 					idx_h1, 
							blk_idx_h1, size_h1, 	idx_h3, 
							blk_idx_p5, size_p5,  // reg_y: p5
													size_p7, 	threadIdx.x + l_fetch, rng_p5, pipeline);
					}

					if ((idx_h3 < rng_h3) && (idx_h1 < rng_p6) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
						g2s_d2_v2_4<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, tmp_dev_d2_v2, 
							blk_idx_p4, 				 	// reg_x: p4
							blk_idx_p6, size_p6, 	idx_h1, 
							blk_idx_h3, size_h3, 	idx_h3, 
													size_p7, 	threadIdx.x + l_fetch, rng_p4, pipeline);
					}
					pipeline.producer_commit();
				}

				pipeline.consumer_wait();
				block.sync();
				const size_t shared_idx = compute_batch % NUM_STAGE;
				
				const int max_iter = (size_internal_up - (compute_batch * SIZE_UNIT_INT)) / 4;
				#pragma unroll 1
				for (int ll = 0; ll < 4 && ll < max_iter; ll++) {
					MmaOperandA op_a;
					op_a.template load_plus<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
					MmaOperandB op_b;
					op_b.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
					mma(op_c, op_a, op_b);
				}
				pipeline.consumer_release();
			}
			block.sync(); 
		}
	}

	// sd1_9: 	t3[h3,h2,h1,p6,p5,p4] += t2[h7,p4,p6,h3] * v2[h2,h1,p5,h7]
	// sd1_9': 	t3[h3,h2,h1,p6,p5,p4] += v2[h2,h1,p5,h7] * t2[h7,p4,p6,h3] --> TB_X(p6,h2), TB_Y(h1,h3), REG_X,Y(p4,p5)
	for (int iter_noab = 0; iter_noab < size_noab; iter_noab++) {
		// 
		int size_h7 	= const_d1_h7b[iter_noab];
		int flag_d1_9 = const_d1_exec[8 + (iter_noab) * 9];
		
		const size_t num_batches = (size_h7 + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;
		const size_t size_internal_up = ((size_h7 + 3) / 4) * 4;

		if (flag_d1_9 >= 0) {
			double* tmp_dev_d1_t2 = dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_9;
			double* tmp_dev_d1_v2 = dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_9;

			internal_upperbound = 0;
			#pragma unroll 1
			for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
				#pragma unroll 1
				for (; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE); ++fetch_batch) {
					pipeline.producer_acquire();

					const int l_fetch = fetch_batch * SIZE_UNIT_INT;
					const size_t shared_idx = fetch_batch % NUM_STAGE;
					// internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
					internal_offset = (l_fetch + SIZE_UNIT_INT) - size_h7;
					block.sync();

					if (internal_offset > 0) { 
						const int start_row = size_h7 - l_fetch;
						const int max_row = ((start_row+3)/4)*4;
						internal_upperbound = internal_offset;
						zero_shared(sm_a + STAGE_OFFSET * shared_idx, start_row, max_row); // Zero out shared memory if partial tile
						zero_shared(sm_b + STAGE_OFFSET * shared_idx, start_row, max_row);
						block.sync();
					}

					if ((idx_h3 < rng_h3) && (idx_h1 < rng_p6) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
						g2s_d1_t2_9<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, tmp_dev_d1_t2, 
							blk_idx_h3, 					idx_h3, 
							blk_idx_p6, size_p6,  idx_h1,  	
							blk_idx_p4, size_p4,
													size_h7, 	threadIdx.x + l_fetch, 
													rng_p4, 	pipeline);
					}

					if ((idx_h2 < rng_h1) && (idx_p6 < rng_h2) && threadIdx.y < SIZE_UNIT_INT - internal_upperbound) {
						g2s_d1_v2_9<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, tmp_dev_d1_v2, 
							blk_idx_p5, size_p5,	
							blk_idx_h1, size_h1, 	idx_h2, 
							blk_idx_h2, size_h2, 	idx_p6, 
																		threadIdx.y + l_fetch, 
													rng_p5, 	pipeline);
					}
					pipeline.producer_commit();
				}
				pipeline.consumer_wait();
				block.sync();
				const size_t shared_idx = compute_batch % NUM_STAGE;

				const int max_iter = (size_internal_up - (compute_batch * SIZE_UNIT_INT)) / 4;
				#pragma unroll 1
				for (int ll = 0; ll < 4 && ll < max_iter; ll++) {
					MmaOperandA op_a;
					op_a.template load_plus<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
					MmaOperandB op_b;
					op_b.template load<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
					mma(op_c, op_a, op_b);
				}
				pipeline.consumer_release();
			}
			block.sync(); 
		}
	}
#endif


	// rt 	TB_X(p6,h2), TB_Y(h1,h3) #4
	// from TB_X(p6,h2), TB_Y(h1,h3), REG_X,Y(p4,p5) // sd2_4
	// to 	TB_X(p6,h3), TB_Y(h2,h1), REG_X,Y(p4,p5) // sd2_5
#ifdef TEST_ENABLE_RT
	rt_store_fixed(sm_block, idx_p6, idx_h2, idx_h1, idx_h3, op_c); 
	block.sync();
	rt_load_fixed(sm_block, idx_p6, idx_h1, idx_h3, idx_h2, op_c);
	block.sync();
#endif

#if 1
	// sd2_5: t3[h3,h2,h1,p6,p5,p4] += t2[p7,p5,h2,h3] * v2[p7,h1,p6,p4] (pending)
	// [1] t2[p7,rx,h2,h3] * v2[p7,h1,ry,p4] -> TB_X(h3,p4), TB_Y(h1,h2)
	// [2] t2[p7,ry,h2,h3] * v2[p7,h1,rx,p4] -> TB_X(p6,h3), TB_Y(h2,h1), REG_X,Y(p4,p5)
	for (int iter_nvab = 0; iter_nvab < size_nvab; iter_nvab++) {	
		int size_p7 	= const_d2_p7b[iter_nvab];
		int flag_d2_5 = const_d2_exec[4 + (iter_nvab) * 9];

		const size_t num_batches = (size_p7 + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;
		const size_t size_internal_up = ((size_p7 + 3) / 4) * 4;

		if (flag_d2_5 >= 0) {
			double* tmp_dev_d2_t2 = dev_d2_t2_all + size_max_dim_d2_t2 * flag_d2_5;
			double* tmp_dev_d2_v2 = dev_d2_v2_all + size_max_dim_d2_v2 * flag_d2_5;

			internal_upperbound = 0;
			#pragma unroll 1
			for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
				#pragma unroll 1
				for (; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE); ++fetch_batch) {
					pipeline.producer_acquire();

					const int l_fetch = fetch_batch * SIZE_UNIT_INT;
					const size_t shared_idx = fetch_batch % NUM_STAGE;
					// internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
					internal_offset = (l_fetch + SIZE_UNIT_INT) - size_p7;
					block.sync();

					if (internal_offset > 0) { 
						const int start_row = size_p7 - l_fetch;
						const int max_row = ((start_row+3)/4)*4;
						internal_upperbound = internal_offset;
						zero_shared(sm_a + STAGE_OFFSET * shared_idx, start_row, max_row); // Zero out shared memory if partial tile
						zero_shared(sm_b + STAGE_OFFSET * shared_idx, start_row, max_row);
						block.sync();
					}

					if ((idx_h3 < rng_h2) && (idx_h1 < rng_h3) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
						g2s_d2_t2_5<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, tmp_dev_d2_t2, 
							blk_idx_h3, 					idx_h1, 
							blk_idx_h2, size_h2, 	idx_h3, 
							blk_idx_p5, size_p5,  
													size_p7, 	threadIdx.x + l_fetch, rng_p5, pipeline);
					}

					if ((idx_h3 < rng_h1) && (idx_h1 < rng_p6) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
						g2s_d2_v2_5<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, tmp_dev_d2_v2, 
							blk_idx_p4, 				 	
							blk_idx_p6, size_p6, 	idx_h1, 
							blk_idx_h1, size_h1, 	idx_h3, 
													size_p7, 	threadIdx.x + l_fetch, rng_p4, pipeline);
					}
					pipeline.producer_commit();
				}

				pipeline.consumer_wait();
				block.sync();
				const size_t shared_idx = compute_batch % NUM_STAGE;
				
				const int max_iter = (size_internal_up - (compute_batch * SIZE_UNIT_INT)) / 4;
				#pragma unroll 1
				for (int ll = 0; ll < 4 && ll < max_iter; ll++) {
					MmaOperandA op_a;
					op_a.template load_plus<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
					MmaOperandB op_b;
					op_b.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
					mma(op_c, op_a, op_b);
				}
				pipeline.consumer_release();
			}
			block.sync(); 
		}
	}

	// sd1_7: 	t3[h3,h2,h1,p6,p5,p4] += t2[h7,p4,p6,h1] * v2[h3,h2,p5,h7]
	// sd1_7': 	t3[h3,h2,h1,p6,p5,p4] += v2[h3,h2,p5,h7] * t2[h7,p4,p6,h1] --> TB_X(p6,h3), TB_Y(h2,h1), REG_X,Y(p4,p5)
	for (int iter_noab = 0; iter_noab < size_noab; iter_noab++) {
		// 
		int size_h7 	= const_d1_h7b[iter_noab];
		int flag_d1_7 = const_d1_exec[6 + (iter_noab) * 9];
		
		const size_t num_batches = (size_h7 + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;
		const size_t size_internal_up = ((size_h7 + 3) / 4) * 4;

		if (flag_d1_7 >= 0) {
			double* tmp_dev_d1_t2 = dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_7;
			double* tmp_dev_d1_v2 = dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_7;

			internal_upperbound = 0;
			#pragma unroll 1
			for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
				#pragma unroll 1
				for (; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE); ++fetch_batch) {
					pipeline.producer_acquire();

					const int l_fetch = fetch_batch * SIZE_UNIT_INT;
					const size_t shared_idx = fetch_batch % NUM_STAGE;
					// internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
					internal_offset = (l_fetch + SIZE_UNIT_INT) - size_h7;
					block.sync();

					if (internal_offset > 0) { 
						const int start_row = size_h7 - l_fetch;
						const int max_row = ((start_row+3)/4)*4;
						internal_upperbound = internal_offset;
						zero_shared(sm_a + STAGE_OFFSET * shared_idx, start_row, max_row); // Zero out shared memory if partial tile
						zero_shared(sm_b + STAGE_OFFSET * shared_idx, start_row, max_row);
						block.sync();
					}

					if ((idx_h3 < rng_h1) && (idx_h1 < rng_p6) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
						g2s_d1_t2_7<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, tmp_dev_d1_t2, 
							blk_idx_h1, 					idx_h3, 
							blk_idx_p6, size_p6,  idx_h1,  	
							blk_idx_p4, size_p4,
													size_h7, 	threadIdx.x + l_fetch, 
													rng_p4, 	pipeline);
					}

					if ((idx_h2 < rng_h2) && (idx_p6 < rng_h3) && threadIdx.y < SIZE_UNIT_INT - internal_upperbound) {
						g2s_d1_v2_7<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, tmp_dev_d1_v2, 
							blk_idx_p5, size_p5,	
							blk_idx_h2, size_h2, 	idx_h2, 
							blk_idx_h3, size_h3, 	idx_p6, 
																		threadIdx.y + l_fetch, 
													rng_p5, 	pipeline);
					}
					pipeline.producer_commit();
				}
				pipeline.consumer_wait();
				block.sync();
				const size_t shared_idx = compute_batch % NUM_STAGE;

				const int max_iter = (size_internal_up - (compute_batch * SIZE_UNIT_INT)) / 4;
				#pragma unroll 1
				for (int ll = 0; ll < 4 && ll < max_iter; ll++) {
					MmaOperandA op_a;
					op_a.template load_plus<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
					MmaOperandB op_b;
					op_b.template load<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
					mma(op_c, op_a, op_b);
				}
				pipeline.consumer_release();
			}
			block.sync(); 
		}
	}
#endif

	// rt 	TB_X(p6,h2), TB_Y(h1,h3) #5
	// from TB_X(p6,h3), TB_Y(h2,h1), REG_X,Y(p4,p5) // sd2_5
	// to 	TB_X(p6,h3), TB_Y(h1,h2), REG_X,Y(p4,p5) // sd2_6
#ifdef TEST_ENABLE_RT
	rt_store_fixed(sm_block, idx_p6, idx_h2, idx_h1, idx_h3, op_c); 
	block.sync();
	rt_load_fixed(sm_block, idx_p6, idx_h2, idx_h3, idx_h1, op_c);
	block.sync();
#endif

#if 1
	// sd2_6: t3[h3,h2,h1,p6,p5,p4] −= t2[p7,p5,h1,h3] * v2[p7,h2,p6,p4]
	// [1] t2[p7,rx,h1,h3] * v2[p7,h2,ry,p4] -> TB_X(h3,p4), TB_Y(h2,h3)
	// [2] t2[p7,ry,h1,h3] * v2[p7,h2,rx,p4] -> TB_X(p6,h3), TB_Y(h1,h2) <----
	for (int iter_nvab = 0; iter_nvab < size_nvab; iter_nvab++) {	
		int size_p7 	= const_d2_p7b[iter_nvab];
		int flag_d2_6 = const_d2_exec[5 + (iter_nvab) * 9];

		const size_t num_batches = (size_p7 + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;
		const size_t size_internal_up = ((size_p7 + 3) / 4) * 4;

		if (flag_d2_6 >= 0) {
			double* tmp_dev_d2_t2 = dev_d2_t2_all + size_max_dim_d2_t2 * flag_d2_6;
			double* tmp_dev_d2_v2 = dev_d2_v2_all + size_max_dim_d2_v2 * flag_d2_6;

			internal_upperbound = 0;
			#pragma unroll 1
			for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
				#pragma unroll 1
				for (; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE); ++fetch_batch) {
					pipeline.producer_acquire();

					const int l_fetch = fetch_batch * SIZE_UNIT_INT;
					const size_t shared_idx = fetch_batch % NUM_STAGE;
					// internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
					internal_offset = (l_fetch + SIZE_UNIT_INT) - size_p7;
					block.sync();

					if (internal_offset > 0) { 
						const int start_row = size_p7 - l_fetch;
						const int max_row = ((start_row+3)/4)*4;
						internal_upperbound = internal_offset;
						zero_shared(sm_a + STAGE_OFFSET * shared_idx, start_row, max_row); // Zero out shared memory if partial tile
						zero_shared(sm_b + STAGE_OFFSET * shared_idx, start_row, max_row);
						block.sync();
					}

					if ((idx_h3 < rng_h1) && (idx_h1 < rng_h3) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
						g2s_d2_t2_6<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, tmp_dev_d2_t2, 
							blk_idx_h3, 					idx_h1, 
							blk_idx_h1, size_h1, 	idx_h3, 
							blk_idx_p5, size_p5,  
													size_p7, 	threadIdx.x + l_fetch, rng_p5, pipeline);
					}

					if ((idx_h3 < rng_h2) && (idx_h1 < rng_p6) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
						g2s_d2_v2_6<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, tmp_dev_d2_v2, 
							blk_idx_p4, 				 	
							blk_idx_p6, size_p6, 	idx_h1,
							blk_idx_h2, size_h2, 	idx_h3, 
													size_p7, 	threadIdx.x + l_fetch, rng_p4, pipeline);
					}
					pipeline.producer_commit();
				}

				pipeline.consumer_wait();
				block.sync();
				const size_t shared_idx = compute_batch % NUM_STAGE;
				
				const int max_iter = (size_internal_up - (compute_batch * SIZE_UNIT_INT)) / 4;
				#pragma unroll 1
				for (int ll = 0; ll < 4 && ll < max_iter; ll++) {
					MmaOperandA op_a;
					op_a.template load<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
					MmaOperandB op_b;
					op_b.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
					mma(op_c, op_a, op_b);
				}
				pipeline.consumer_release();
			}
			block.sync();
		}
	}

	// sd1_8: t3[h3,h2,h1,p6,p5,p4] -= t2[h7,p4,p6,h2] * v2[h3,h1,p5,h7]
	// sd1_8: t3[h3,h2,h1,p6,p5,p4] -= v2[h3,h1,p5,h7] * t2[h7,p4,p6,h2] --> TB_X(p6,h3), TB_Y(h1,h2), REG_X,Y(p4,p5)
	for (int iter_noab = 0; iter_noab < size_noab; iter_noab++) {
		// 
		int size_h7 	= const_d1_h7b[iter_noab];
		int flag_d1_8 = const_d1_exec[7 + (iter_noab) * 9];
		
		const size_t num_batches = (size_h7 + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;
		const size_t size_internal_up = ((size_h7 + 3) / 4) * 4;

		if (flag_d1_8 >= 0) {
			double* tmp_dev_d1_t2 = dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_8;
			double* tmp_dev_d1_v2 = dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_8;

			internal_upperbound = 0;
			#pragma unroll 1
			for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
				#pragma unroll 1
				for (; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE); ++fetch_batch) {
					pipeline.producer_acquire();

					const int l_fetch = fetch_batch * SIZE_UNIT_INT;
					const size_t shared_idx = fetch_batch % NUM_STAGE;
					// internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
					internal_offset = (l_fetch + SIZE_UNIT_INT) - size_h7;
					block.sync();

					if (internal_offset > 0) { 
						const int start_row = size_h7 - l_fetch;
						const int max_row = ((start_row+3)/4)*4;
						internal_upperbound = internal_offset;
						zero_shared(sm_a + STAGE_OFFSET * shared_idx, start_row, max_row); // Zero out shared memory if partial tile
						zero_shared(sm_b + STAGE_OFFSET * shared_idx, start_row, max_row);
						block.sync();
					}

					if ((idx_h3 < rng_h2) && (idx_h1 < rng_p6) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
						g2s_d1_t2_8<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, tmp_dev_d1_t2, 
							blk_idx_h2, 					idx_h3, 
							blk_idx_p6, size_p6,  idx_h1,  	
							blk_idx_p4, size_p4,
													size_h7, 	threadIdx.x + l_fetch, 
													rng_p4, 	pipeline);
					}

					if ((idx_h2 < rng_h1) && (idx_p6 < rng_h3) && threadIdx.y < SIZE_UNIT_INT - internal_upperbound) {
						g2s_d1_v2_8<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, tmp_dev_d1_v2, 
							blk_idx_p5, size_p5,	
							blk_idx_h1, size_h1, 	idx_h2, 
							blk_idx_h3, size_h3, 	idx_p6, 
																		threadIdx.y + l_fetch, 
													rng_p5, 	pipeline);
					}
					pipeline.producer_commit();
				}
				pipeline.consumer_wait();
				block.sync();
				const size_t shared_idx = compute_batch % NUM_STAGE;

				const int max_iter = (size_internal_up - (compute_batch * SIZE_UNIT_INT)) / 4;
				#pragma unroll 1
				for (int ll = 0; ll < 4 && ll < max_iter; ll++) {
					MmaOperandA op_a;
					op_a.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
					MmaOperandB op_b;
					op_b.template load<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
					mma(op_c, op_a, op_b);
				}
				pipeline.consumer_release();
			}
			block.sync(); 
		}
	}
#endif

	// rt 	TB_X(p6,h2), TB_Y(h1,h3) #6
	// from TB_X(p6,h3), TB_Y(h1,h2), REG_X,Y(p4,p5)
	// to 	TB_X(p4,h2), TB_Y(h1,h3), REG_X,Y(p5,p6)
#ifdef TEST_ENABLE_RT
	rt_store_fixed(sm_block, idx_p6, idx_h2, idx_h1, idx_h3, op_c); // x=[(p4,h3),(p6)], y=[(h1,h2),(p5)] -> x=[(p4,h2),(p5)], y=[(h1,h3),(p6)] 
	block.sync();
	#pragma unroll 4
	for (int i = 0; i < 4; i++) { // p6
		#pragma unroll 4
		for (int j = 0; j < 4; j++) { // p5
			op_c.reg[j + (i) * 4] = sm_block[i + (idx_h3 + (idx_p6) * 4) * 4 + 
																			(idx_h1 + (idx_h2 + (j) * 4) * 4) * 65];
		}
	}
	block.sync();
#endif

//---------------------------------------------------------------------------- 
// >> REG_X,Y(p5,p6)
// sd2_7 && sd1_3
// sd2_8 && sd1_1
// sd2_9 && sd1_2
//---------------------------------------------------------------------------- 
#if 1
	// sd2_7: 	t3[h3,h2,h1,p6,p5,p4] −= t2[p7,p6,h1,h2] * v2[p7,h3,p5,p4] --> TB_X(p4,h2), TB_Y(h1,h3)
	for (int iter_nvab = 0; iter_nvab < size_nvab; iter_nvab++) {	
		int size_p7 	= const_d2_p7b[iter_nvab];
		int flag_d2_7 = const_d2_exec[6 + (iter_nvab) * 9];

		const size_t num_batches = (size_p7 + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;
		const size_t size_internal_up = ((size_p7 + 3) / 4) * 4;

		if (flag_d2_7 >= 0) {
			double* tmp_dev_d2_t2 = dev_d2_t2_all + size_max_dim_d2_t2 * flag_d2_7;
			double* tmp_dev_d2_v2 = dev_d2_v2_all + size_max_dim_d2_v2 * flag_d2_7;

			internal_upperbound = 0;
			#pragma unroll 1
			for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
				#pragma unroll 1
				for (; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE); ++fetch_batch) {
					pipeline.producer_acquire();

					const int l_fetch = fetch_batch * SIZE_UNIT_INT;
					const size_t shared_idx = fetch_batch % NUM_STAGE;
					// internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
					internal_offset = (l_fetch + SIZE_UNIT_INT) - size_p7;
					block.sync();

					if (internal_offset > 0) { 
						const int start_row = size_p7 - l_fetch;
						const int max_row = ((start_row+3)/4)*4;
						internal_upperbound = internal_offset;
						zero_shared(sm_a + STAGE_OFFSET * shared_idx, start_row, max_row); // Zero out shared memory if partial tile
						zero_shared(sm_b + STAGE_OFFSET * shared_idx, start_row, max_row);
						block.sync();
					}

					if ((idx_h3 < rng_h1) && (idx_h1 < rng_h2) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
						g2s_d2_t2_7<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, tmp_dev_d2_t2, 
							blk_idx_h2, 					idx_h1, 
							blk_idx_h1, size_h1, 	idx_h3, 
							blk_idx_p6, size_p6,  
													size_p7, 	threadIdx.x + l_fetch, rng_p6, pipeline);
					}

					if ((idx_h3 < rng_h3) && (idx_h1 < rng_p4) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
						g2s_d2_v2_7<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, tmp_dev_d2_v2,  
							blk_idx_p4, 				 	idx_h1, 
							blk_idx_p5, size_p5, 
							blk_idx_h3, size_h3, 	idx_h3, 
													size_p7, 	threadIdx.x + l_fetch, rng_p5, pipeline);
					}
					pipeline.producer_commit();
				}

				pipeline.consumer_wait();
				block.sync();
				const size_t shared_idx = compute_batch % NUM_STAGE;
				
				const int max_iter = (size_internal_up - (compute_batch * SIZE_UNIT_INT)) / 4;
				#pragma unroll 1
				for (int ll = 0; ll < 4 && ll < max_iter; ll++) {
					MmaOperandA op_a;
					op_a.template load<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
					MmaOperandB op_b;
					op_b.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
					mma(op_c, op_a, op_b);
				}
				pipeline.consumer_release();
			}
			block.sync();
		}
	}

	// sd1_3: 	t3[h3,h2,h1,p6,p5,p4] -= t2[h7,p4,p5,h3] * v2[h2,h1,p6,h7]
	// sd1_3': 	t3[h3,h2,h1,p6,p5,p4] -= v2[h2,h1,p6,h7] * t2[h7,p4,p5,h3] --> TB_X(h3,h1), TB_Y(h2,p4)
	// sd1_3'': t3[h3,h2,h1,p6,p5,p4] -= v2[h2,h1,p6,h7] * t2[h7,p4,p5,h3] --> TB_X(p4,h2), TB_X(h1,h3)
	for (int iter_noab = 0; iter_noab < size_noab; iter_noab++) {
		// 
		int size_h7 	= const_d1_h7b[iter_noab];
		int flag_d1_3 = const_d1_exec[2 + (iter_noab) * 9];
		
		const size_t num_batches = (size_h7 + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;
		const size_t size_internal_up = ((size_h7 + 3) / 4) * 4;

		if (flag_d1_3 >= 0) {
			double* tmp_dev_d1_t2 = dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_3;
			double* tmp_dev_d1_v2 = dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_3;

			internal_upperbound = 0;
			#pragma unroll 1
			for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
				#pragma unroll 1
				for (; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE); ++fetch_batch) {
					pipeline.producer_acquire();

					const int l_fetch = fetch_batch * SIZE_UNIT_INT;
					const size_t shared_idx = fetch_batch % NUM_STAGE;
					// internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
					internal_offset = (l_fetch + SIZE_UNIT_INT) - size_h7;
					block.sync();

					if (internal_offset > 0) { 
						const int start_row = size_h7 - l_fetch;
						const int max_row = ((start_row+3)/4)*4;
						internal_upperbound = internal_offset;
						zero_shared(sm_a + STAGE_OFFSET * shared_idx, start_row, max_row); // Zero out shared memory if partial tile
						zero_shared(sm_b + STAGE_OFFSET * shared_idx, start_row, max_row);
						block.sync();
					}

					if ((idx_h3 < rng_h3) && (idx_h1 < rng_p4) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
						g2s_d1_t2_3<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, tmp_dev_d1_t2, 
							blk_idx_h3, 					idx_h3, 
							blk_idx_p5, size_p5, 	
							blk_idx_p4, size_p4,  idx_h1, 
													size_h7, 	threadIdx.x + l_fetch, 
													rng_p5, 	pipeline);
					}

					if ((idx_h2 < rng_h1) && (idx_p6 < rng_h2) && threadIdx.y < SIZE_UNIT_INT - internal_upperbound) {
						g2s_d1_v2_3<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, tmp_dev_d1_v2, 
							blk_idx_p6, size_p6,	
							blk_idx_h1, size_h1, 	idx_h2, 
							blk_idx_h2, size_h2, 	idx_p6, 
																		threadIdx.y + l_fetch, 
													rng_p6, 	pipeline);
					}
					pipeline.producer_commit();
				}
				pipeline.consumer_wait();
				block.sync();
				const size_t shared_idx = compute_batch % NUM_STAGE;

				const int max_iter = (size_internal_up - (compute_batch * SIZE_UNIT_INT)) / 4;
				#pragma unroll 1
				for (int ll = 0; ll < 4 && ll < max_iter; ll++) {
					MmaOperandA op_a;
					op_a.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
					MmaOperandB op_b;
					op_b.template load<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
					mma(op_c, op_a, op_b);
				}
				pipeline.consumer_release();
			}
			block.sync(); 
		}
	}
#endif

	// rt 	TB_X(p6,h2), TB_Y(h1,h3) #7
	// from TB_X(p4,h2), TB_Y(h1,h3)
	// to 	TB_X(p4,h3), TB_Y(h2,h1)
#ifdef TEST_ENABLE_RT
	rt_store_fixed(sm_block, idx_p6, idx_h2, idx_h1, idx_h3, op_c); 
	block.sync();
	rt_load_fixed(sm_block, idx_p6, idx_h1, idx_h3, idx_h2, op_c); 
	block.sync();
#endif

#if 1
	// sd2_8: t3[h3,h2,h1,p6,p5,p4] −= t2[p7,p6,h2,h3] * v2[p7,h1,p5,p4]
	// t2[p7,ry,h2,h3] * v2[p7,h1,rx,p4] -> TB_X(p4,h3), TB_Y(h2,h1)
	for (int iter_nvab = 0; iter_nvab < size_nvab; iter_nvab++) {	
		int size_p7 	= const_d2_p7b[iter_nvab];
		int flag_d2_8 = const_d2_exec[7 + (iter_nvab) * 9];

		const size_t num_batches = (size_p7 + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;
		const size_t size_internal_up = ((size_p7 + 3) / 4) * 4;

		if (flag_d2_8 >= 0) {
			double* tmp_dev_d2_t2 = dev_d2_t2_all + size_max_dim_d2_t2 * flag_d2_8;
			double* tmp_dev_d2_v2 = dev_d2_v2_all + size_max_dim_d2_v2 * flag_d2_8;

			internal_upperbound = 0;
			#pragma unroll 1
			for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
				#pragma unroll 1
				for (; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE); ++fetch_batch) {
					pipeline.producer_acquire();

					const int l_fetch = fetch_batch * SIZE_UNIT_INT;
					const size_t shared_idx = fetch_batch % NUM_STAGE;
					// internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
					internal_offset = (l_fetch + SIZE_UNIT_INT) - size_p7;
					block.sync();

					if (internal_offset > 0) { 
						const int start_row = size_p7 - l_fetch;
						const int max_row = ((start_row+3)/4)*4;
						internal_upperbound = internal_offset;
						zero_shared(sm_a + STAGE_OFFSET * shared_idx, start_row, max_row); // Zero out shared memory if partial tile
						zero_shared(sm_b + STAGE_OFFSET * shared_idx, start_row, max_row);
						block.sync();
					}

					if ((idx_h3 < rng_h2) && (idx_h1 < rng_h3) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
						g2s_d2_t2_8<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, tmp_dev_d2_t2, 
							blk_idx_h3, 					idx_h1, 
							blk_idx_h2, size_h2, 	idx_h3, 
							blk_idx_p6, size_p6,  
													size_p7, threadIdx.x + l_fetch, rng_p6, pipeline);
					}

					if ((idx_h3 < rng_h1) && (idx_h1 < rng_p4) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
						g2s_d2_v2_8<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, tmp_dev_d2_v2, 
							blk_idx_p4, 				 	idx_h1, 
							blk_idx_p5, size_p5, 
							blk_idx_h1, size_h1, 	idx_h3, 
													size_p7, 	threadIdx.x + l_fetch, rng_p5, pipeline);
					}
					pipeline.producer_commit();
				}
				pipeline.consumer_wait();
				block.sync();

				const size_t shared_idx = compute_batch % NUM_STAGE;

				const int max_iter = (size_internal_up - (compute_batch * SIZE_UNIT_INT)) / 4;
				#pragma unroll 1
				for (int ll = 0; ll < 4 && ll < max_iter; ll++) {
					MmaOperandA op_a;
					op_a.template load<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
					MmaOperandB op_b;
					op_b.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
					mma(op_c, op_a, op_b);
				}
				pipeline.consumer_release();
			}
			block.sync(); 
		}
	}

	// sd1_1: 	t3[h3,h2,h1,p6,p5,p4] -= t2[h7,p4,p5,h1] * v2[h3,h2,p6,h7]
	// sd1_1': 	t3[h3,h2,h1,p6,p5,p4] -= v2[h3,h2,p6,h7] * t2[h7,p4,p5,h1] --> TB_X(h1,h2), TB_Y(h3,p4), REG_X,Y(p5,p6)
	// sd1_1'': t3[h3,h2,h1,p6,p5,p4] -= v2[h3,h2,p6,h7] * t2[h7,p4,p5,h1] --> TB_X(p4,h3), TB_Y(h2,h1), REG_X,Y(p5,p6)
	for (int iter_noab = 0; iter_noab < size_noab; iter_noab++) {
		// 
		int size_h7 	= const_d1_h7b[iter_noab];
		int flag_d1_1 = const_d1_exec[0 + (iter_noab) * 9];
		
		const size_t num_batches = (size_h7 + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;
		const size_t size_internal_up = ((size_h7 + 3) / 4) * 4;

		if (flag_d1_1 >= 0) {
			double* tmp_dev_d1_t2 = dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_1;
			double* tmp_dev_d1_v2 = dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_1;

			internal_upperbound = 0;
			#pragma unroll 1
			for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
				#pragma unroll 1
				for (; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE); ++fetch_batch) {
					pipeline.producer_acquire();

					const int l_fetch = fetch_batch * SIZE_UNIT_INT;
					const size_t shared_idx = fetch_batch % NUM_STAGE;
					// internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
					internal_offset = (l_fetch + SIZE_UNIT_INT) - size_h7;
					block.sync();

					if (internal_offset > 0) { 
						const int start_row = size_h7 - l_fetch;
						const int max_row = ((start_row+3)/4)*4;
						internal_upperbound = internal_offset;
						zero_shared(sm_a + STAGE_OFFSET * shared_idx, start_row, max_row); // Zero out shared memory if partial tile
						zero_shared(sm_b + STAGE_OFFSET * shared_idx, start_row, max_row);
						block.sync();
					}

					if ((idx_h3 < rng_h1) && (idx_h1 < rng_p4) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
						g2s_d1_t2_1<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, tmp_dev_d1_t2, 
							blk_idx_h1, 					idx_h3, 
							blk_idx_p5, size_p5, 	
							blk_idx_p4, size_p4,  idx_h1, 
													size_h7, 	threadIdx.x + l_fetch, 
													rng_p5, 	pipeline);
					}

					if ((idx_h2 < rng_h2) && (idx_p6 < rng_h3) && threadIdx.y < SIZE_UNIT_INT - internal_upperbound) {
						g2s_d1_v2_1<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, tmp_dev_d1_v2, 
							blk_idx_p6, size_p6,	
							blk_idx_h2, size_h2, 	idx_h2, 
							blk_idx_h3, size_h3, 	idx_p6, 
																		threadIdx.y + l_fetch, 
													rng_p6, 	pipeline);
					}
					pipeline.producer_commit();
				}
				pipeline.consumer_wait();
				block.sync();
				const size_t shared_idx = compute_batch % NUM_STAGE;

				const int max_iter = (size_internal_up - (compute_batch * SIZE_UNIT_INT)) / 4;
				#pragma unroll 1
				for (int ll = 0; ll < 4 && ll < max_iter; ll++) {
					MmaOperandA op_a;
					op_a.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
					MmaOperandB op_b;
					op_b.template load<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
					mma(op_c, op_a, op_b);
				}
				pipeline.consumer_release();
			}
			block.sync(); 
		}
	}
#endif

	// rt 	TB_X(p6,h2), TB_Y(h1,h3) #8
	// from TB_X(p4,h3), TB_Y(h2,h1)
	// to 	TB_X(p4,h3), TB_Y(h1,h2)
#ifdef TEST_ENABLE_RT
	rt_store_fixed(sm_block, idx_p6, idx_h2, idx_h1, idx_h3, op_c); 
	block.sync();
	rt_load_fixed(sm_block, idx_p6, idx_h2, idx_h3, idx_h1, op_c); 
	block.sync();
#endif

#if 1
	// sd2_9: t3[h3,h2,h1,p6,p5,p4] += t2[p7,p6,h1,h3] * v2[p7,h2,p5,p4]
	// TB_X(p4,h3), TB_Y(h1,h2), REG_X,Y(p5,p6)
	for (int iter_nvab = 0; iter_nvab < size_nvab; iter_nvab++) {	
		int size_p7 	= const_d2_p7b[iter_nvab];
		int flag_d2_9 = const_d2_exec[8 + (iter_nvab) * 9];

		const size_t num_batches = (size_p7 + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;
		const size_t size_internal_up = ((size_p7 + 3) / 4) * 4;

		if (flag_d2_9 >= 0) {
			double* tmp_dev_d2_t2 = dev_d2_t2_all + size_max_dim_d2_t2 * flag_d2_9;
			double* tmp_dev_d2_v2 = dev_d2_v2_all + size_max_dim_d2_v2 * flag_d2_9;

			internal_upperbound = 0;
			#pragma unroll 1
			for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
				#pragma unroll 1
				for (; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE); ++fetch_batch) {
					pipeline.producer_acquire();

					const int l_fetch = fetch_batch * SIZE_UNIT_INT;
					const size_t shared_idx = fetch_batch % NUM_STAGE;
					// internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
					internal_offset = (l_fetch + SIZE_UNIT_INT) - size_p7;
					block.sync();

					if (internal_offset > 0) { 
						const int start_row = size_p7 - l_fetch;
						const int max_row = ((start_row+3)/4)*4;
						internal_upperbound = internal_offset;
						zero_shared(sm_a + STAGE_OFFSET * shared_idx, start_row, max_row); // Zero out shared memory if partial tile
						zero_shared(sm_b + STAGE_OFFSET * shared_idx, start_row, max_row);
						block.sync();
					}

					if ((idx_h3 < rng_h1) && (idx_h1 < rng_h3) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
						g2s_d2_t2_9<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, tmp_dev_d2_t2, 
							blk_idx_h3, 					idx_h1, 
							blk_idx_h1, size_h1, 	idx_h3, 
							blk_idx_p6, size_p6,  
													size_p7, 	threadIdx.x + l_fetch, 
													rng_p6, pipeline);
					}

					if ((idx_h3 < rng_h2) && (idx_h1 < rng_p4) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
						g2s_d2_v2_9<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, tmp_dev_d2_v2, 
							blk_idx_p4, 				 	idx_h1, 
							blk_idx_p5, size_p5, 
							blk_idx_h2, size_h2, 	idx_h3, 
													size_p7, 	threadIdx.x + l_fetch, 
													rng_p5, pipeline);
					}
					pipeline.producer_commit();
				}
				pipeline.consumer_wait();
				block.sync();
				const size_t shared_idx = compute_batch % NUM_STAGE;

				const int max_iter = (size_internal_up - (compute_batch * SIZE_UNIT_INT)) / 4;
				#pragma unroll 1
				for (int ll = 0; ll < 4 && ll < max_iter; ll++) {
					MmaOperandA op_a;
					op_a.template load_plus<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
					MmaOperandB op_b;
					op_b.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
					mma(op_c, op_a, op_b);
				}
				pipeline.consumer_release();
			}
			block.sync(); 
		}
	}

	// sd1_2: 	t3[h3,h2,h1,p6,p5,p4] += t2[h7,p4,p5,h2] * v2[h3,h1,p6,h7] 
	// sd1_2': 	t3[h3,h2,h1,p6,p5,p4] += v2[h3,h1,p6,h7] * t2[h7,p4,p5,h2] --> TB_X(p2,h1), TB_Y(h3,p4), REG_X,Y(p5,p6)
	// sd1_2'':	t3[h3,h2,h1,p6,p5,p4] += v2[h3,h1,p6,h7] * t2[h7,p4,p5,h2] --> TB_X(p4,h3), TB_Y(h1,h2), REG_X,Y(p5,p6)
	for (int iter_noab = 0; iter_noab < size_noab; iter_noab++) {
		// 
		int size_h7 	= const_d1_h7b[iter_noab];
		int flag_d1_2 = const_d1_exec[1 + (iter_noab) * 9];
		
		const size_t num_batches = (size_h7 + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;
		const size_t size_internal_up = ((size_h7 + 3) / 4) * 4;

		if (flag_d1_2 >= 0) {
			double* tmp_dev_d1_t2 = dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_2;
			double* tmp_dev_d1_v2 = dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_2;

			internal_upperbound = 0;
			#pragma unroll 1
			for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
				#pragma unroll 1
				for (; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE); ++fetch_batch) {
					pipeline.producer_acquire();

					const int l_fetch = fetch_batch * SIZE_UNIT_INT;
					const size_t shared_idx = fetch_batch % NUM_STAGE;
					// internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
					internal_offset = (l_fetch + SIZE_UNIT_INT) - size_h7;
					block.sync();

					if (internal_offset > 0) { 
						const int start_row = size_h7 - l_fetch;
						const int max_row = ((start_row+3)/4)*4;
						internal_upperbound = internal_offset;
						zero_shared(sm_a + STAGE_OFFSET * shared_idx, start_row, max_row); // Zero out shared memory if partial tile
						zero_shared(sm_b + STAGE_OFFSET * shared_idx, start_row, max_row);
						block.sync();
					}

					if ((idx_h3 < rng_h2) && (idx_h1 < rng_p4) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
						g2s_d1_t2_2<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, tmp_dev_d1_t2, 
							blk_idx_h2, 					idx_h3, 
							blk_idx_p5, size_p5, 	
							blk_idx_p4, size_p4,  idx_h1, 
													size_h7, 	threadIdx.x + l_fetch, 
													rng_p5, 	pipeline);
					}

					if ((idx_h2 < rng_h1) && (idx_p6 < rng_h3) && threadIdx.y < SIZE_UNIT_INT - internal_upperbound) {
						g2s_d1_v2_2<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, tmp_dev_d1_v2, 
							blk_idx_p6, size_p6,	
							blk_idx_h1, size_h1, 	idx_h2, 
							blk_idx_h3, size_h3, 	idx_p6, 
																		threadIdx.y + l_fetch, 
													rng_p6, 	pipeline);
					}
					pipeline.producer_commit();
				}
				pipeline.consumer_wait();
				block.sync();
				const size_t shared_idx = compute_batch % NUM_STAGE;

				const int max_iter = (size_internal_up - (compute_batch * SIZE_UNIT_INT)) / 4;
				#pragma unroll 1
				for (int ll = 0; ll < 4 && ll < max_iter; ll++) {
					MmaOperandA op_a;
					op_a.template load_plus<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
					MmaOperandB op_b;
					op_b.template load<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
					mma(op_c, op_a, op_b);
				}
				pipeline.consumer_release();
			}
			block.sync();
		}
	}
#endif

	//---------------------------------------------------------------------------- 
	// 	
	// 	S (Singles)
	// 
	//---------------------------------------------------------------------------- 

	// 	flags
	int flag_s1_1 = const_s1_exec[0];
	int flag_s1_2 = const_s1_exec[1];
	int flag_s1_3 = const_s1_exec[2];
	int flag_s1_4 = const_s1_exec[3];
	int flag_s1_5 = const_s1_exec[4];
	int flag_s1_6 = const_s1_exec[5];
	int flag_s1_7 = const_s1_exec[6];
	int flag_s1_8 = const_s1_exec[7];
	int flag_s1_9 = const_s1_exec[8];
	
	//---------------------------------------------------------------------------- 
	// [1] TB_X(p4,h3), TB_Y(h1,h2), REG_X,Y(p5,p6) by TB_X(p6,h2) and TB_Y(h1,h3)
	// [2] TB_X(p6,h3), TB_Y(h1,h2), REG_X,Y(p5,p4) by TB_X(p6,h2) and TB_Y(h1,h3)
	//---------------------------------------------------------------------------- 
	// 																 t1[ry,h1] * v2[h3,h2,p6,rx]
	// 	s1_1: t3[h3,h2,h1,p6,p5,p4] += t1[p4,h1] * v2[h3,h2,p6,p5]
	if (flag_s1_1 >= 0) {
		double* dev_s1_t1_1 = dev_s1_t1_all + size_max_dim_s1_t1 * flag_s1_1;
		double* dev_s1_v2_1 = dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_1;

		//
		if (idx_h1 == 0 && idx_h3 == 0) {
			if (idx_p6 < rng_p4 && idx_h2 < rng_h1) {
				sm_a[idx_p6 + (idx_h2) * SIZE_TILE_P4] = dev_s1_t1_1[blk_idx_p4 * SIZE_TILE_P4 + idx_p6 + (blk_idx_h1 * SIZE_TILE_H1 + idx_h2) * size_p4];
			}
		} 

		if (idx_p6 < rng_h3 && idx_h2 < rng_h2 && idx_h1 < rng_p6 && idx_h3 < rng_p5) {
			sm_b[idx_p6 + (idx_h2 + (idx_h1 + (idx_h3) * SIZE_TILE_P6) * SIZE_TILE_H2) * SIZE_TILE_H3] = dev_s1_v2_1[blk_idx_h3 * SIZE_TILE_H3 + idx_p6 + 
													(blk_idx_h2 * SIZE_TILE_H2 + idx_h2 + 
													(blk_idx_p6 * SIZE_TILE_P6 + idx_h1 + 
													(blk_idx_p5 * SIZE_TILE_P5 + idx_h3) * size_p6) * size_h2) * size_h3];
		}
		block.sync();

		// TB_X(p4,h3), TB_Y(h1,h2), REG_X,Y(p5,p6) by TB_X(p6(p4),h2(h3)) and TB_Y(h1(h1),h3(h2))
		op_c_s.reg[0 + (0) * SIZE_TILE_P5] += sm_a[0 + (idx_h1) * SIZE_TILE_P4] * sm_b[idx_h2 + (idx_h3 + (idx_p6 + (0) * SIZE_TILE_P6) * SIZE_TILE_H2) * SIZE_TILE_H3];
		op_c_s.reg[1 + (0) * SIZE_TILE_P5] += sm_a[0 + (idx_h1) * SIZE_TILE_P4] * sm_b[idx_h2 + (idx_h3 + (idx_p6 + (1) * SIZE_TILE_P6) * SIZE_TILE_H2) * SIZE_TILE_H3];
		op_c_s.reg[2 + (0) * SIZE_TILE_P5] += sm_a[0 + (idx_h1) * SIZE_TILE_P4] * sm_b[idx_h2 + (idx_h3 + (idx_p6 + (2) * SIZE_TILE_P6) * SIZE_TILE_H2) * SIZE_TILE_H3];
		op_c_s.reg[3 + (0) * SIZE_TILE_P5] += sm_a[0 + (idx_h1) * SIZE_TILE_P4] * sm_b[idx_h2 + (idx_h3 + (idx_p6 + (3) * SIZE_TILE_P6) * SIZE_TILE_H2) * SIZE_TILE_H3];

		op_c_s.reg[0 + (1) * SIZE_TILE_P5] += sm_a[1 + (idx_h1) * SIZE_TILE_P4] * sm_b[idx_h2 + (idx_h3 + (idx_p6 + (0) * SIZE_TILE_P6) * SIZE_TILE_H2) * SIZE_TILE_H3];
		op_c_s.reg[1 + (1) * SIZE_TILE_P5] += sm_a[1 + (idx_h1) * SIZE_TILE_P4] * sm_b[idx_h2 + (idx_h3 + (idx_p6 + (1) * SIZE_TILE_P6) * SIZE_TILE_H2) * SIZE_TILE_H3];
		op_c_s.reg[2 + (1) * SIZE_TILE_P5] += sm_a[1 + (idx_h1) * SIZE_TILE_P4] * sm_b[idx_h2 + (idx_h3 + (idx_p6 + (2) * SIZE_TILE_P6) * SIZE_TILE_H2) * SIZE_TILE_H3];
		op_c_s.reg[3 + (1) * SIZE_TILE_P5] += sm_a[1 + (idx_h1) * SIZE_TILE_P4] * sm_b[idx_h2 + (idx_h3 + (idx_p6 + (3) * SIZE_TILE_P6) * SIZE_TILE_H2) * SIZE_TILE_H3];

		op_c_s.reg[0 + (2) * SIZE_TILE_P5] += sm_a[2 + (idx_h1) * SIZE_TILE_P4] * sm_b[idx_h2 + (idx_h3 + (idx_p6 + (0) * SIZE_TILE_P6) * SIZE_TILE_H2) * SIZE_TILE_H3];
		op_c_s.reg[1 + (2) * SIZE_TILE_P5] += sm_a[2 + (idx_h1) * SIZE_TILE_P4] * sm_b[idx_h2 + (idx_h3 + (idx_p6 + (1) * SIZE_TILE_P6) * SIZE_TILE_H2) * SIZE_TILE_H3];
		op_c_s.reg[2 + (2) * SIZE_TILE_P5] += sm_a[2 + (idx_h1) * SIZE_TILE_P4] * sm_b[idx_h2 + (idx_h3 + (idx_p6 + (2) * SIZE_TILE_P6) * SIZE_TILE_H2) * SIZE_TILE_H3];
		op_c_s.reg[3 + (2) * SIZE_TILE_P5] += sm_a[2 + (idx_h1) * SIZE_TILE_P4] * sm_b[idx_h2 + (idx_h3 + (idx_p6 + (3) * SIZE_TILE_P6) * SIZE_TILE_H2) * SIZE_TILE_H3];

		op_c_s.reg[0 + (3) * SIZE_TILE_P5] += sm_a[3 + (idx_h1) * SIZE_TILE_P4] * sm_b[idx_h2 + (idx_h3 + (idx_p6 + (0) * SIZE_TILE_P6) * SIZE_TILE_H2) * SIZE_TILE_H3];
		op_c_s.reg[1 + (3) * SIZE_TILE_P5] += sm_a[3 + (idx_h1) * SIZE_TILE_P4] * sm_b[idx_h2 + (idx_h3 + (idx_p6 + (1) * SIZE_TILE_P6) * SIZE_TILE_H2) * SIZE_TILE_H3];
		op_c_s.reg[2 + (3) * SIZE_TILE_P5] += sm_a[3 + (idx_h1) * SIZE_TILE_P4] * sm_b[idx_h2 + (idx_h3 + (idx_p6 + (2) * SIZE_TILE_P6) * SIZE_TILE_H2) * SIZE_TILE_H3];
		op_c_s.reg[3 + (3) * SIZE_TILE_P5] += sm_a[3 + (idx_h1) * SIZE_TILE_P4] * sm_b[idx_h2 + (idx_h3 + (idx_p6 + (3) * SIZE_TILE_P6) * SIZE_TILE_H2) * SIZE_TILE_H3];
		
		block.sync();
	}

	// s1_2: t3[h3,h2,h1,p6,p5,p4] -= t2[p4,h2] * v2[h3,h1,p6,p5]
	if (flag_s1_2 >= 0) {
		// 
		double* dev_s1_t1_2 = dev_s1_t1_all + size_max_dim_s1_t1 * flag_s1_2;
		double* dev_s1_v2_2 = dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_2;

		if (threadIdx.y == 0) { 
			if (idx_p6 < rng_p4 && idx_h2 < rng_h2) {
				sm_a[idx_p6 + (idx_h2) * SIZE_TILE_P4] = dev_s1_t1_2[blk_idx_p4 * SIZE_TILE_P4 + idx_p6 + (blk_idx_h2 * SIZE_TILE_H2 + idx_h2) * size_p4];
			}
		}

		if (idx_p6 < rng_h3 && idx_h2 < rng_h1 && idx_h1 < rng_p6 && idx_h3 < rng_p5) {
			sm_b[idx_p6 + (idx_h2 + (idx_h1 + (idx_h3) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2] = 
			dev_s1_v2_2[ blk_idx_h3 * SIZE_TILE_H3 + idx_p6 + 
									(blk_idx_h1 * SIZE_TILE_H1 + idx_h2 + 
									(blk_idx_p6 * SIZE_TILE_P6 + idx_h1 + 
									(blk_idx_p5 * SIZE_TILE_P5 + idx_h3) * size_p6) * size_h1) * size_h3];
		}
		block.sync();

		//    TB_X(p6,h3), TB_Y(h1,h2), REG_X,Y(p5,p4)
		// by TB_X(p6,h2), TB_Y(h1,h3)
		op_c_s.reg[0 + (0) * SIZE_TILE_P5] -= sm_a[0 + (idx_h3) * SIZE_TILE_P4] * sm_b[idx_h2 + (idx_h1 + (idx_p6 + (0) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[1 + (0) * SIZE_TILE_P5] -= sm_a[0 + (idx_h3) * SIZE_TILE_P4] * sm_b[idx_h2 + (idx_h1 + (idx_p6 + (1) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[2 + (0) * SIZE_TILE_P5] -= sm_a[0 + (idx_h3) * SIZE_TILE_P4] * sm_b[idx_h2 + (idx_h1 + (idx_p6 + (2) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[3 + (0) * SIZE_TILE_P5] -= sm_a[0 + (idx_h3) * SIZE_TILE_P4] * sm_b[idx_h2 + (idx_h1 + (idx_p6 + (3) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];

		op_c_s.reg[0 + (1) * SIZE_TILE_P5] -= sm_a[1 + (idx_h3) * SIZE_TILE_P4] * sm_b[idx_h2 + (idx_h1 + (idx_p6 + (0) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[1 + (1) * SIZE_TILE_P5] -= sm_a[1 + (idx_h3) * SIZE_TILE_P4] * sm_b[idx_h2 + (idx_h1 + (idx_p6 + (1) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[2 + (1) * SIZE_TILE_P5] -= sm_a[1 + (idx_h3) * SIZE_TILE_P4] * sm_b[idx_h2 + (idx_h1 + (idx_p6 + (2) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[3 + (1) * SIZE_TILE_P5] -= sm_a[1 + (idx_h3) * SIZE_TILE_P4] * sm_b[idx_h2 + (idx_h1 + (idx_p6 + (3) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];

		op_c_s.reg[0 + (2) * SIZE_TILE_P5] -= sm_a[2 + (idx_h3) * SIZE_TILE_P4] * sm_b[idx_h2 + (idx_h1 + (idx_p6 + (0) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[1 + (2) * SIZE_TILE_P5] -= sm_a[2 + (idx_h3) * SIZE_TILE_P4] * sm_b[idx_h2 + (idx_h1 + (idx_p6 + (1) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[2 + (2) * SIZE_TILE_P5] -= sm_a[2 + (idx_h3) * SIZE_TILE_P4] * sm_b[idx_h2 + (idx_h1 + (idx_p6 + (2) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[3 + (2) * SIZE_TILE_P5] -= sm_a[2 + (idx_h3) * SIZE_TILE_P4] * sm_b[idx_h2 + (idx_h1 + (idx_p6 + (3) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];

		op_c_s.reg[0 + (3) * SIZE_TILE_P5] -= sm_a[3 + (idx_h3) * SIZE_TILE_P4] * sm_b[idx_h2 + (idx_h1 + (idx_p6 + (0) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[1 + (3) * SIZE_TILE_P5] -= sm_a[3 + (idx_h3) * SIZE_TILE_P4] * sm_b[idx_h2 + (idx_h1 + (idx_p6 + (1) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[2 + (3) * SIZE_TILE_P5] -= sm_a[3 + (idx_h3) * SIZE_TILE_P4] * sm_b[idx_h2 + (idx_h1 + (idx_p6 + (2) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[3 + (3) * SIZE_TILE_P5] -= sm_a[3 + (idx_h3) * SIZE_TILE_P4] * sm_b[idx_h2 + (idx_h1 + (idx_p6 + (3) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];

		block.sync();
	}

	// s1_3: t3[h3,h2,h1,p6,p5,p4] += t2[p4,h3] * v2[h2,h1,p6,p5]
	if (flag_s1_3 >= 0) {
		// 
		double* dev_s1_t1_3 = dev_s1_t1_all + size_max_dim_s1_t1 * flag_s1_3;
		double* dev_s1_v2_3 = dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_3;

		if (threadIdx.y == 0) { 
			if (idx_p6 < rng_p4 && idx_h2 < rng_h3) {
				sm_a[idx_p6 + (idx_h2) * SIZE_TILE_P4] = dev_s1_t1_3[blk_idx_p4 * SIZE_TILE_P4 + idx_p6 + (blk_idx_h3 * SIZE_TILE_H3 + idx_h2) * size_p4];
			}
		}

		if (idx_p6 < rng_h2 && idx_h2 < rng_h1 && idx_h1 < rng_p6 && idx_h3 < rng_p5) {
			sm_b[idx_p6 + (idx_h2 + (idx_h1 + (idx_h3) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2] = 
			dev_s1_v2_3[ blk_idx_h2 * SIZE_TILE_H2 + idx_p6 + 
									(blk_idx_h1 * SIZE_TILE_H1 + idx_h2 + 
									(blk_idx_p6 * SIZE_TILE_P6 + idx_h1 + 
									(blk_idx_p5 * SIZE_TILE_P5 + idx_h3) * size_p6) * size_h1) * size_h2];
		}
		block.sync();

		// TB_X(p6,h3), TB_Y(h1,h2), REG_X,Y(p5,p4)
		// by TB_X(p6,h2), TB_Y(h1,h3)
		op_c_s.reg[0 + (0) * SIZE_TILE_P5] += sm_a[0 + (idx_h2) * SIZE_TILE_P4] * sm_b[idx_h3 + (idx_h1 + (idx_p6 + (0) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[1 + (0) * SIZE_TILE_P5] += sm_a[0 + (idx_h2) * SIZE_TILE_P4] * sm_b[idx_h3 + (idx_h1 + (idx_p6 + (1) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[2 + (0) * SIZE_TILE_P5] += sm_a[0 + (idx_h2) * SIZE_TILE_P4] * sm_b[idx_h3 + (idx_h1 + (idx_p6 + (2) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[3 + (0) * SIZE_TILE_P5] += sm_a[0 + (idx_h2) * SIZE_TILE_P4] * sm_b[idx_h3 + (idx_h1 + (idx_p6 + (3) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		
		op_c_s.reg[0 + (1) * SIZE_TILE_P5] += sm_a[1 + (idx_h2) * SIZE_TILE_P4] * sm_b[idx_h3 + (idx_h1 + (idx_p6 + (0) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[1 + (1) * SIZE_TILE_P5] += sm_a[1 + (idx_h2) * SIZE_TILE_P4] * sm_b[idx_h3 + (idx_h1 + (idx_p6 + (1) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[2 + (1) * SIZE_TILE_P5] += sm_a[1 + (idx_h2) * SIZE_TILE_P4] * sm_b[idx_h3 + (idx_h1 + (idx_p6 + (2) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[3 + (1) * SIZE_TILE_P5] += sm_a[1 + (idx_h2) * SIZE_TILE_P4] * sm_b[idx_h3 + (idx_h1 + (idx_p6 + (3) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];

		op_c_s.reg[0 + (2) * SIZE_TILE_P5] += sm_a[2 + (idx_h2) * SIZE_TILE_P4] * sm_b[idx_h3 + (idx_h1 + (idx_p6 + (0) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[1 + (2) * SIZE_TILE_P5] += sm_a[2 + (idx_h2) * SIZE_TILE_P4] * sm_b[idx_h3 + (idx_h1 + (idx_p6 + (1) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[2 + (2) * SIZE_TILE_P5] += sm_a[2 + (idx_h2) * SIZE_TILE_P4] * sm_b[idx_h3 + (idx_h1 + (idx_p6 + (2) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[3 + (2) * SIZE_TILE_P5] += sm_a[2 + (idx_h2) * SIZE_TILE_P4] * sm_b[idx_h3 + (idx_h1 + (idx_p6 + (3) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];

		op_c_s.reg[0 + (3) * SIZE_TILE_P5] += sm_a[3 + (idx_h2) * SIZE_TILE_P4] * sm_b[idx_h3 + (idx_h1 + (idx_p6 + (0) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[1 + (3) * SIZE_TILE_P5] += sm_a[3 + (idx_h2) * SIZE_TILE_P4] * sm_b[idx_h3 + (idx_h1 + (idx_p6 + (1) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[2 + (3) * SIZE_TILE_P5] += sm_a[3 + (idx_h2) * SIZE_TILE_P4] * sm_b[idx_h3 + (idx_h1 + (idx_p6 + (2) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[3 + (3) * SIZE_TILE_P5] += sm_a[3 + (idx_h2) * SIZE_TILE_P4] * sm_b[idx_h3 + (idx_h1 + (idx_p6 + (3) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];

		block.sync();
	}

	// from TB_X(p6,h3), TB_Y(h1,h2), REG_X,Y(p5,p4)
	// to 	TB_X(p4,h3), TB_Y(h1,h2), REG_X,Y(p5,p6)
#ifdef TEST_ENABLE_RT
	rt_store_fixed(sm_block, idx_p6, idx_h2, idx_h1, idx_h3, op_c_s); // x=[(p6,h3),(p5)], y=[(h1,h2),(p4)] -> x=[(p4,h3),(p5)], y=[(h1,h2),(p6)] 
	block.sync();
	#pragma unroll 4
	for (int i = 0; i < 4; i++) { // p6
		#pragma unroll 4
		for (int j = 0; j < 4; j++) { // p5
			op_c_s.reg[j + (i) * 4] = sm_block[ 		i + (idx_h2 + 		 (j) * 4) * 4 + 
																				(idx_h1 + (idx_h3 + (idx_p6) * 4) * 4) * 65];
		}
	}
	block.sync();
#endif

	// s1_4: t3[h3,h2,h1,p6,p5,p4] -= t2[p5,h1] * v2[h3,h2,p6,p4]
	if (flag_s1_4 >= 0) {
		double* dev_s1_t1_4 = dev_s1_t1_all + size_max_dim_s1_t1 * flag_s1_4;
		double* dev_s1_v2_4 = dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_4;

		if (threadIdx.y == 0) { 
			if (idx_p6 < rng_p5 && idx_h2 < rng_h1) {
				sm_a[idx_p6 + (idx_h2) * SIZE_TILE_P5] = dev_s1_t1_4[blk_idx_p5 * SIZE_TILE_P5 + idx_p6 + (blk_idx_h1 * SIZE_TILE_H1 + idx_h2) * size_p5];
			}
		}

		if (idx_p6 < rng_h3 && idx_h2 < rng_h2 && idx_h1 < rng_p6 && idx_h3 < rng_p4) {
			sm_b[idx_p6 + (idx_h2 + (idx_h1 + (idx_h3) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2] = 
			dev_s1_v2_4[ blk_idx_h3 * SIZE_TILE_H3 + idx_p6 + 
									(blk_idx_h2 * SIZE_TILE_H2 + idx_h2 + 
									(blk_idx_p6 * SIZE_TILE_P6 + idx_h1 + 
									(blk_idx_p4 * SIZE_TILE_P4 + idx_h3) * size_p6) * size_h2) * size_h3];
		}
		block.sync();

		// 		TB_X(p4,h3), TB_Y(h1,h2)
		// 		REG_X,Y(p5,p6) 
		// by TB_X(p6,h2), TB_Y(h1,h3)
		op_c_s.reg[0 + (0) * SIZE_TILE_P5] -= sm_a[0 + (idx_h1) * SIZE_TILE_P5] * sm_b[idx_h2 + (idx_h3 + (0 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[1 + (0) * SIZE_TILE_P5] -= sm_a[1 + (idx_h1) * SIZE_TILE_P5] * sm_b[idx_h2 + (idx_h3 + (0 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[2 + (0) * SIZE_TILE_P5] -= sm_a[2 + (idx_h1) * SIZE_TILE_P5] * sm_b[idx_h2 + (idx_h3 + (0 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[3 + (0) * SIZE_TILE_P5] -= sm_a[3 + (idx_h1) * SIZE_TILE_P5] * sm_b[idx_h2 + (idx_h3 + (0 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];

		op_c_s.reg[0 + (1) * SIZE_TILE_P5] -= sm_a[0 + (idx_h1) * SIZE_TILE_P5] * sm_b[idx_h2 + (idx_h3 + (1 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[1 + (1) * SIZE_TILE_P5] -= sm_a[1 + (idx_h1) * SIZE_TILE_P5] * sm_b[idx_h2 + (idx_h3 + (1 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[2 + (1) * SIZE_TILE_P5] -= sm_a[2 + (idx_h1) * SIZE_TILE_P5] * sm_b[idx_h2 + (idx_h3 + (1 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[3 + (1) * SIZE_TILE_P5] -= sm_a[3 + (idx_h1) * SIZE_TILE_P5] * sm_b[idx_h2 + (idx_h3 + (1 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];

		op_c_s.reg[0 + (2) * SIZE_TILE_P5] -= sm_a[0 + (idx_h1) * SIZE_TILE_P5] * sm_b[idx_h2 + (idx_h3 + (2 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[1 + (2) * SIZE_TILE_P5] -= sm_a[1 + (idx_h1) * SIZE_TILE_P5] * sm_b[idx_h2 + (idx_h3 + (2 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[2 + (2) * SIZE_TILE_P5] -= sm_a[2 + (idx_h1) * SIZE_TILE_P5] * sm_b[idx_h2 + (idx_h3 + (2 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[3 + (2) * SIZE_TILE_P5] -= sm_a[3 + (idx_h1) * SIZE_TILE_P5] * sm_b[idx_h2 + (idx_h3 + (2 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];

		op_c_s.reg[0 + (3) * SIZE_TILE_P5] -= sm_a[0 + (idx_h1) * SIZE_TILE_P5] * sm_b[idx_h2 + (idx_h3 + (3 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[1 + (3) * SIZE_TILE_P5] -= sm_a[1 + (idx_h1) * SIZE_TILE_P5] * sm_b[idx_h2 + (idx_h3 + (3 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[2 + (3) * SIZE_TILE_P5] -= sm_a[2 + (idx_h1) * SIZE_TILE_P5] * sm_b[idx_h2 + (idx_h3 + (3 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[3 + (3) * SIZE_TILE_P5] -= sm_a[3 + (idx_h1) * SIZE_TILE_P5] * sm_b[idx_h2 + (idx_h3 + (3 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];

		block.sync();
	}

	// s1_5: t3[h3,h2,h1,p6,p5,p4] -= t2[p5,h2] * v2[h3,h1,p6,p4]
	if (flag_s1_5 >= 0) {
		double* dev_s1_t1_5 = dev_s1_t1_all + size_max_dim_s1_t1 * flag_s1_5;
		double* dev_s1_v2_5 = dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_5;

		if (threadIdx.y == 0) { 
			if (idx_p6 < rng_p5 && idx_h2 < rng_h2) {
				sm_a[idx_p6 + (idx_h2) * SIZE_TILE_P5] = dev_s1_t1_5[blk_idx_p5 * SIZE_TILE_P5 + idx_p6 + (blk_idx_h2 * SIZE_TILE_H2 + idx_h2) * size_p5];
			}
		}

		if (idx_p6 < rng_h3 && idx_h2 < rng_h1 && idx_h1 < rng_p6 && idx_h3 < rng_p4) {
			sm_b[idx_p6 + (idx_h2 + (idx_h1 + (idx_h3) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2] = 
			dev_s1_v2_5[ blk_idx_h3 * SIZE_TILE_H3 + idx_p6 + 
									(blk_idx_h1 * SIZE_TILE_H1 + idx_h2 + 
									(blk_idx_p6 * SIZE_TILE_P6 + idx_h1 + 
									(blk_idx_p4 * SIZE_TILE_P4 + idx_h3) * size_p6) * size_h1) * size_h3];
		}
		block.sync();

		// 		TB_X(p4,h3), TB_Y(h1,h2)
		// 		REG_X,Y(p5,p6) 
		// by TB_X(p6,h2), TB_Y(h1,h3)
		op_c_s.reg[0 + (0) * SIZE_TILE_P5] += sm_a[0 + (idx_h3) * SIZE_TILE_P5] * sm_b[idx_h2 + (idx_h1 + (0 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[1 + (0) * SIZE_TILE_P5] += sm_a[1 + (idx_h3) * SIZE_TILE_P5] * sm_b[idx_h2 + (idx_h1 + (0 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[2 + (0) * SIZE_TILE_P5] += sm_a[2 + (idx_h3) * SIZE_TILE_P5] * sm_b[idx_h2 + (idx_h1 + (0 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[3 + (0) * SIZE_TILE_P5] += sm_a[3 + (idx_h3) * SIZE_TILE_P5] * sm_b[idx_h2 + (idx_h1 + (0 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		
		op_c_s.reg[0 + (1) * SIZE_TILE_P5] += sm_a[0 + (idx_h3) * SIZE_TILE_P5] * sm_b[idx_h2 + (idx_h1 + (1 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[1 + (1) * SIZE_TILE_P5] += sm_a[1 + (idx_h3) * SIZE_TILE_P5] * sm_b[idx_h2 + (idx_h1 + (1 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[2 + (1) * SIZE_TILE_P5] += sm_a[2 + (idx_h3) * SIZE_TILE_P5] * sm_b[idx_h2 + (idx_h1 + (1 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[3 + (1) * SIZE_TILE_P5] += sm_a[3 + (idx_h3) * SIZE_TILE_P5] * sm_b[idx_h2 + (idx_h1 + (1 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];

		op_c_s.reg[0 + (2) * SIZE_TILE_P5] += sm_a[0 + (idx_h3) * SIZE_TILE_P5] * sm_b[idx_h2 + (idx_h1 + (2 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[1 + (2) * SIZE_TILE_P5] += sm_a[1 + (idx_h3) * SIZE_TILE_P5] * sm_b[idx_h2 + (idx_h1 + (2 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[2 + (2) * SIZE_TILE_P5] += sm_a[2 + (idx_h3) * SIZE_TILE_P5] * sm_b[idx_h2 + (idx_h1 + (2 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[3 + (2) * SIZE_TILE_P5] += sm_a[3 + (idx_h3) * SIZE_TILE_P5] * sm_b[idx_h2 + (idx_h1 + (2 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];

		op_c_s.reg[0 + (3) * SIZE_TILE_P5] += sm_a[0 + (idx_h3) * SIZE_TILE_P5] * sm_b[idx_h2 + (idx_h1 + (3 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[1 + (3) * SIZE_TILE_P5] += sm_a[1 + (idx_h3) * SIZE_TILE_P5] * sm_b[idx_h2 + (idx_h1 + (3 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[2 + (3) * SIZE_TILE_P5] += sm_a[2 + (idx_h3) * SIZE_TILE_P5] * sm_b[idx_h2 + (idx_h1 + (3 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[3 + (3) * SIZE_TILE_P5] += sm_a[3 + (idx_h3) * SIZE_TILE_P5] * sm_b[idx_h2 + (idx_h1 + (3 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];

		block.sync();
	}

	// s1_6: t3[h3,h2,h1,p6,p5,p4] -= t2[p5,h3] * v2[h2,h1,p6,p4]
	if (flag_s1_6 >= 0) {
		double* dev_s1_t1_6 = dev_s1_t1_all + size_max_dim_s1_t1 * flag_s1_6;
		double* dev_s1_v2_6 = dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_6;

		if (threadIdx.y == 0) { 
			if (idx_p6 < rng_p5 && idx_h2 < rng_h3) {
				sm_a[idx_p6 + (idx_h2) * SIZE_TILE_P5] = dev_s1_t1_6[blk_idx_p5 * SIZE_TILE_P5 + idx_p6 + (blk_idx_h3 * SIZE_TILE_H3 + idx_h2) * size_p5];
			}
		}

		if (idx_p6 < rng_h2 && idx_h2 < rng_h1 && idx_h1 < rng_p6 && idx_h3 < rng_p4) {
			sm_b[idx_p6 + (idx_h2 + (idx_h1 + (idx_h3) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2] = 
			dev_s1_v2_6[ blk_idx_h2 * SIZE_TILE_H2 + idx_p6 + 
									(blk_idx_h1 * SIZE_TILE_H1 + idx_h2 + 
									(blk_idx_p6 * SIZE_TILE_P6 + idx_h1 + 
									(blk_idx_p4 * SIZE_TILE_P4 + idx_h3) * size_p6) * size_h1) * size_h2];
		}
		block.sync();

		// 		TB_X(p4,h3), TB_Y(h1,h2)
		// 		REG_X,Y(p5,p6) 
		// by TB_X(p6,h2), TB_Y(h1,h3)
		op_c_s.reg[0 + (0) * SIZE_TILE_P5] -= sm_a[0 + (idx_h2) * SIZE_TILE_P5] * sm_b[idx_h3 + (idx_h1 + (0 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[1 + (0) * SIZE_TILE_P5] -= sm_a[1 + (idx_h2) * SIZE_TILE_P5] * sm_b[idx_h3 + (idx_h1 + (0 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[2 + (0) * SIZE_TILE_P5] -= sm_a[2 + (idx_h2) * SIZE_TILE_P5] * sm_b[idx_h3 + (idx_h1 + (0 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[3 + (0) * SIZE_TILE_P5] -= sm_a[3 + (idx_h2) * SIZE_TILE_P5] * sm_b[idx_h3 + (idx_h1 + (0 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];

		op_c_s.reg[0 + (1) * SIZE_TILE_P5] -= sm_a[0 + (idx_h2) * SIZE_TILE_P5] * sm_b[idx_h3 + (idx_h1 + (1 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[1 + (1) * SIZE_TILE_P5] -= sm_a[1 + (idx_h2) * SIZE_TILE_P5] * sm_b[idx_h3 + (idx_h1 + (1 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[2 + (1) * SIZE_TILE_P5] -= sm_a[2 + (idx_h2) * SIZE_TILE_P5] * sm_b[idx_h3 + (idx_h1 + (1 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[3 + (1) * SIZE_TILE_P5] -= sm_a[3 + (idx_h2) * SIZE_TILE_P5] * sm_b[idx_h3 + (idx_h1 + (1 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];

		op_c_s.reg[0 + (2) * SIZE_TILE_P5] -= sm_a[0 + (idx_h2) * SIZE_TILE_P5] * sm_b[idx_h3 + (idx_h1 + (2 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[1 + (2) * SIZE_TILE_P5] -= sm_a[1 + (idx_h2) * SIZE_TILE_P5] * sm_b[idx_h3 + (idx_h1 + (2 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[2 + (2) * SIZE_TILE_P5] -= sm_a[2 + (idx_h2) * SIZE_TILE_P5] * sm_b[idx_h3 + (idx_h1 + (2 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[3 + (2) * SIZE_TILE_P5] -= sm_a[3 + (idx_h2) * SIZE_TILE_P5] * sm_b[idx_h3 + (idx_h1 + (2 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];

		op_c_s.reg[0 + (3) * SIZE_TILE_P5] -= sm_a[0 + (idx_h2) * SIZE_TILE_P5] * sm_b[idx_h3 + (idx_h1 + (3 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[1 + (3) * SIZE_TILE_P5] -= sm_a[1 + (idx_h2) * SIZE_TILE_P5] * sm_b[idx_h3 + (idx_h1 + (3 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[2 + (3) * SIZE_TILE_P5] -= sm_a[2 + (idx_h2) * SIZE_TILE_P5] * sm_b[idx_h3 + (idx_h1 + (3 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[3 + (3) * SIZE_TILE_P5] -= sm_a[3 + (idx_h2) * SIZE_TILE_P5] * sm_b[idx_h3 + (idx_h1 + (3 + (idx_p6) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
		
		// 
		block.sync();
	}

	// s1_7: t3[h3,h2,h1,p6,p5,p4] -= t2[p6,h1] * v2[h3,h2,p5,p4]
	if (flag_s1_7 >= 0) {
		double* dev_s1_t1_7 = dev_s1_t1_all + size_max_dim_s1_t1 * flag_s1_7;
		double* dev_s1_v2_7 = dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_7;

		if (threadIdx.y == 0) { 
			if (idx_p6 < rng_p6 && idx_h2 < rng_h1) {
				sm_a[idx_p6 + (idx_h2) * SIZE_TILE_P6] = dev_s1_t1_7[blk_idx_p6 * SIZE_TILE_P6 + idx_p6 + (blk_idx_h1 * SIZE_TILE_H1 + idx_h2) * size_p6];
			}
		}

		if (idx_p6 < rng_h3 && idx_h2 < rng_h2 && idx_h1 < rng_p5 && idx_h3 < rng_p4) {
			sm_b[idx_p6 + (idx_h2 + (idx_h1 + (idx_h3) * SIZE_TILE_P5) * SIZE_TILE_H2) * SIZE_TILE_H3] = 
			dev_s1_v2_7[blk_idx_h3 * SIZE_TILE_H3 + idx_p6 + (blk_idx_h2 * SIZE_TILE_H2 + idx_h2 + (blk_idx_p5 * SIZE_TILE_P5 + idx_h1 + (blk_idx_p4 * SIZE_TILE_P4 + idx_h3) * size_p5) * size_h2) * size_h3];
		}
		block.sync();

		// 		TB_X(p4,h3), TB_Y(h1,h2)
		// 		REG_X,Y(p5,p6) 
		// by TB_X(p6,h2) and TB_Y(h1,h3)
		op_c_s.reg[0 + (0) * SIZE_TILE_P5] += sm_a[0 + (idx_h1) * SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h3 + (0 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[0 + (1) * SIZE_TILE_P5] += sm_a[1 + (idx_h1) * SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h3 + (0 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[0 + (2) * SIZE_TILE_P5] += sm_a[2 + (idx_h1) * SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h3 + (0 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[0 + (3) * SIZE_TILE_P5] += sm_a[3 + (idx_h1) * SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h3 + (0 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];

		op_c_s.reg[1 + (0) * SIZE_TILE_P5] += sm_a[0 + (idx_h1) * SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h3 + (1 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[1 + (1) * SIZE_TILE_P5] += sm_a[1 + (idx_h1) * SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h3 + (1 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[1 + (2) * SIZE_TILE_P5] += sm_a[2 + (idx_h1) * SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h3 + (1 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[1 + (3) * SIZE_TILE_P5] += sm_a[3 + (idx_h1) * SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h3 + (1 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];

		op_c_s.reg[2 + (0) * SIZE_TILE_P5] += sm_a[0 + (idx_h1) * SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h3 + (2 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[2 + (1) * SIZE_TILE_P5] += sm_a[1 + (idx_h1) * SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h3 + (2 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[2 + (2) * SIZE_TILE_P5] += sm_a[2 + (idx_h1) * SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h3 + (2 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[2 + (3) * SIZE_TILE_P5] += sm_a[3 + (idx_h1) * SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h3 + (2 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];

		op_c_s.reg[3 + (0) * SIZE_TILE_P5] += sm_a[0 + (idx_h1) * SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h3 + (3 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[3 + (1) * SIZE_TILE_P5] += sm_a[1 + (idx_h1) * SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h3 + (3 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[3 + (2) * SIZE_TILE_P5] += sm_a[2 + (idx_h1) * SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h3 + (3 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[3 + (3) * SIZE_TILE_P5] += sm_a[3 + (idx_h1) * SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h3 + (3 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
		block.sync();
	}

	// s1_8: t3[h3,h2,h1,p6,p5,p4] -= t2[p6,h2] * v2[h3,h1,p5,p4]
	if (flag_s1_8 >= 0) {
		double* dev_s1_t1_8 = dev_s1_t1_all + size_max_dim_s1_t1 * flag_s1_8;
		double* dev_s1_v2_8 = dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_8;
		
		if (threadIdx.y == 0) { 
			if (idx_p6 < rng_p6 && idx_h2 < rng_h2) {
				sm_a[idx_p6 + (idx_h2) * SIZE_TILE_P6] = dev_s1_t1_8[blk_idx_p6 * SIZE_TILE_P6 + idx_p6 + (blk_idx_h2 * SIZE_TILE_H2 + idx_h2) * size_p6];
			}
		}

		if (idx_p6 < rng_h3 && idx_h2 < rng_h1 && idx_h1 < rng_p5 && idx_h3 < rng_p4) {
			sm_b[idx_p6 + (idx_h2 + (idx_h1 + (idx_h3) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H3] = 
			dev_s1_v2_8[blk_idx_h3 * SIZE_TILE_H3 + idx_p6 + 
								 (blk_idx_h1 * SIZE_TILE_H1 + idx_h2 + 
								 (blk_idx_p5 * SIZE_TILE_P5 + idx_h1 + 
								 (blk_idx_p4 * SIZE_TILE_P4 + idx_h3) * size_p5) * size_h1) * size_h3];
		}
		block.sync();

		// 
		// TB_X(p4,h3), TB_Y(h1,h2) (target)
		// TB_X(p6,h2), TB_Y(h1,h3) (index)
		// REG_X,Y(p5,p6) 
		op_c_s.reg[0 + (0) * SIZE_TILE_P5] -= sm_a[0 + (idx_h3) * SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h1 + (0 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[0 + (1) * SIZE_TILE_P5] -= sm_a[1 + (idx_h3) * SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h1 + (0 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[0 + (2) * SIZE_TILE_P5] -= sm_a[2 + (idx_h3) * SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h1 + (0 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[0 + (3) * SIZE_TILE_P5] -= sm_a[3 + (idx_h3) * SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h1 + (0 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];

		op_c_s.reg[1 + (0) * SIZE_TILE_P5] -= sm_a[0 + (idx_h3) * SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h1 + (1 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[1 + (1) * SIZE_TILE_P5] -= sm_a[1 + (idx_h3) * SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h1 + (1 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[1 + (2) * SIZE_TILE_P5] -= sm_a[2 + (idx_h3) * SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h1 + (1 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[1 + (3) * SIZE_TILE_P5] -= sm_a[3 + (idx_h3) * SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h1 + (1 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];

		op_c_s.reg[2 + (0) * SIZE_TILE_P5] -= sm_a[0 + (idx_h3) * SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h1 + (2 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[2 + (1) * SIZE_TILE_P5] -= sm_a[1 + (idx_h3) * SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h1 + (2 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[2 + (2) * SIZE_TILE_P5] -= sm_a[2 + (idx_h3) * SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h1 + (2 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[2 + (3) * SIZE_TILE_P5] -= sm_a[3 + (idx_h3) * SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h1 + (2 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];

		op_c_s.reg[3 + (0) * SIZE_TILE_P5] -= sm_a[0 + (idx_h3) * SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h1 + (3 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[3 + (1) * SIZE_TILE_P5] -= sm_a[1 + (idx_h3) * SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h1 + (3 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[3 + (2) * SIZE_TILE_P5] -= sm_a[2 + (idx_h3) * SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h1 + (3 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[3 + (3) * SIZE_TILE_P5] -= sm_a[3 + (idx_h3) * SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h1 + (3 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
	
		block.sync();
	}

	// 		TB_X(p4,h3), TB_Y(h1,h2)
	// 		REG_X,Y(p5,p6) 
	// by TB_X(p6,h2) and TB_Y(h1,h3)
	// 																 t1[ry,h3] * v2[h2,h1,rx,p4]
	// 	s1_9: t3[h3,h2,h1,p6,p5,p4] -= t1[p6,h3] * v2[h2,h1,p5,p4]
	if (flag_s1_9 >= 0) {
		double* dev_s1_t1_9 = dev_s1_t1_all + size_max_dim_s1_t1 * flag_s1_9;
		double* dev_s1_v2_9 = dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_9;

		if (threadIdx.y == 0) { 
			if (idx_p6 < rng_p6 && idx_h2 < rng_h3) {
				sm_a[idx_p6 + (idx_h2) * SIZE_TILE_P6] = dev_s1_t1_9[blk_idx_p6 * SIZE_TILE_P6 + idx_p6 + (blk_idx_h3 * SIZE_TILE_H3 + idx_h2) * size_p6];
			}
		}

		if (idx_p6 < rng_h2 && idx_h2 < rng_h1 && idx_h1 < rng_p5 && idx_h3 < rng_p4) {
			sm_b[idx_p6 + (idx_h2 + (idx_h1 + (idx_h3) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2] = 
			dev_s1_v2_9[blk_idx_h2 * SIZE_TILE_H2 + idx_p6 + (blk_idx_h1 * SIZE_TILE_H1 + idx_h2 + (blk_idx_p5 * SIZE_TILE_P5 + idx_h1 + (blk_idx_p4 * SIZE_TILE_P4 + idx_h3) * size_p5) * size_h1) * size_h2];
		}
		block.sync();

		op_c_s.reg[0 + (0) * SIZE_TILE_P5] += sm_a[0 + (idx_h2) * SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h1 + (0 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[0 + (1) * SIZE_TILE_P5] += sm_a[1 + (idx_h2) * SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h1 + (0 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[0 + (2) * SIZE_TILE_P5] += sm_a[2 + (idx_h2) * SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h1 + (0 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[0 + (3) * SIZE_TILE_P5] += sm_a[3 + (idx_h2) * SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h1 + (0 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];

		op_c_s.reg[1 + (0) * SIZE_TILE_P5] += sm_a[0 + (idx_h2) * SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h1 + (1 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[1 + (1) * SIZE_TILE_P5] += sm_a[1 + (idx_h2) * SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h1 + (1 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[1 + (2) * SIZE_TILE_P5] += sm_a[2 + (idx_h2) * SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h1 + (1 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[1 + (3) * SIZE_TILE_P5] += sm_a[3 + (idx_h2) * SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h1 + (1 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];

		op_c_s.reg[2 + (0) * SIZE_TILE_P5] += sm_a[0 + (idx_h2) * SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h1 + (2 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[2 + (1) * SIZE_TILE_P5] += sm_a[1 + (idx_h2) * SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h1 + (2 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[2 + (2) * SIZE_TILE_P5] += sm_a[2 + (idx_h2) * SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h1 + (2 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[2 + (3) * SIZE_TILE_P5] += sm_a[3 + (idx_h2) * SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h1 + (2 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];

		op_c_s.reg[3 + (0) * SIZE_TILE_P5] += sm_a[0 + (idx_h2) * SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h1 + (3 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[3 + (1) * SIZE_TILE_P5] += sm_a[1 + (idx_h2) * SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h1 + (3 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[3 + (2) * SIZE_TILE_P5] += sm_a[2 + (idx_h2) * SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h1 + (3 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
		op_c_s.reg[3 + (3) * SIZE_TILE_P5] += sm_a[3 + (idx_h2) * SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h1 + (3 + (idx_p6) * SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
		block.sync();
	}

#if 1
	block.sync();

	if (threadIdx.y == 0) {
		if (idx_h2 < rng_p6 && idx_p6 < rng_p5) { 
		sm_a[idx_h2 * 4 + idx_p6] = (dev_evl_sorted_p5b[blk_idx_p5 * SIZE_TILE_P5 + idx_p6] + dev_evl_sorted_p6b[blk_idx_p6 * SIZE_TILE_P6 + idx_h2]);
		}
	}
	block.sync();
#endif
  // 
  // 	kernel x(idx_p6,idx_h2), y(idx_h1,idx_h3)
  // 
	double energy_1 = 0.0;
	double energy_2 = 0.0;
	if (idx_p6 < rng_p4 && idx_h2 < rng_h3 && idx_h1 < rng_h1 && idx_h3 < rng_h2) {
		#pragma unroll 4
		for (int idx_reg_y = 0; idx_reg_y < 4; idx_reg_y++) {
			#pragma unroll 4
			for (int idx_reg_x = 0; idx_reg_x < 4; idx_reg_x++) {
				// 
				if (idx_reg_y < rng_p6 && idx_reg_x < rng_p5) { 
#if 1
					double inner_factor = (partial_inner_factor - sm_a[idx_reg_y * 4 + idx_reg_x]);
					double temp = op_c.reg[idx_reg_y * 4 + idx_reg_x] / inner_factor;
					energy_1 += temp *  op_c.reg[idx_reg_y * 4 + idx_reg_x];
					energy_2 += temp * (op_c.reg[idx_reg_y * 4 + idx_reg_x] + op_c_s.reg[idx_reg_y * 4 + idx_reg_x]);
#else
					double inner_factor = partial_inner_factor - dev_evl_sorted_p5b[blk_idx_p5 * SIZE_TILE_P5 + idx_reg_x] - dev_evl_sorted_p6b[blk_idx_p6 * SIZE_TILE_P6 + idx_reg_y];
					energy_1 += op_c.reg[idx_reg_y * 4 + idx_reg_x] *  op_c.reg[idx_reg_y * 4 + idx_reg_x] / inner_factor;
					energy_2 += op_c.reg[idx_reg_y * 4 + idx_reg_x] * (op_c.reg[idx_reg_y * 4 + idx_reg_x] + op_c_s.reg[idx_reg_y * 4 + idx_reg_x]) / inner_factor;
#endif
				}
			}
		}
	}
	__syncthreads();

	// 
	//  to partially reduce the energies--- E(4) and E(5)
	//  a warp: 32 -(1)-> 16 -(2)-> 8 -(3)-> 4 -(4)-> 2 
	// 
	for (int offset = 16; offset > 0; offset /= 2) {
		energy_1 += __shfl_down_sync(FULL_MASK, energy_1, offset);
		energy_2 += __shfl_down_sync(FULL_MASK, energy_2, offset);
	}

	if (threadIdx.x == 0 && threadIdx.y % 2 == 0)  {
		sm_a[threadIdx.y / 2] = energy_1;
		sm_b[threadIdx.y / 2] = energy_2;
		// atomicAdd(&dev_energy[0], energy_1);
		// atomicAdd(&dev_energy[1], energy_2);
	}
	__syncthreads();

	// 
	double final_energy_1 = 0.0;
	double final_energy_2 = 0.0;
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		for (int i = 0; i < 8; i++) {
			final_energy_1 += sm_a[i];
			final_energy_2 += sm_b[i];
		}

		// 
		// 	[TODO] atomicAdd vs. Memcpy 
		// 
	#if 0
		atomicAdd(&dev_energy[0], final_energy_1);
		atomicAdd(&dev_energy[1], final_energy_2);
	#else
		dev_energy[blockIdx.x] 							= final_energy_1;
		dev_energy[blockIdx.x + gridDim.x] 	= final_energy_2;
	#endif
	}
}


// #define DEBUG_PRINT_KERNEL_TIME
/**
 *	@brief the driver of the fully-fused kernel for CCSD(T)
**/
void ccsd_t_fully_fused_nvidia_tc_fp64(cudaStream_t* stream_id, size_t numBlks, 
	size_t size_h3, size_t size_h2, size_t size_h1, 
	size_t size_p6, size_t size_p5, size_t size_p4, 
	// 
	double* dev_s1_t1_all, double* dev_s1_v2_all, 
	double* dev_d1_t2_all, double* dev_d1_v2_all, 
	double* dev_d2_t2_all, double* dev_d2_v2_all, 
	// 
											int* host_size_d1_h7b, 	int* host_size_d2_p7b,
	int* host_exec_s1, 	int* host_exec_d1, 			int* host_exec_d2, 
	// 
	size_t size_noab, size_t size_nvab, 
	size_t size_max_dim_s1_t1, size_t size_max_dim_s1_v2, 
	size_t size_max_dim_d1_t2, size_t size_max_dim_d1_v2, 
	size_t size_max_dim_d2_t2, size_t size_max_dim_d2_v2, 
	// 
	double factor, 
	double* dev_evl_sorted_h1b, double* dev_evl_sorted_h2b, double* dev_evl_sorted_h3b, 
	double* dev_evl_sorted_p4b, double* dev_evl_sorted_p5b, double* dev_evl_sorted_p6b, 
	double* dev_energies,
	gpuEvent_t done_compute, gpuEvent_t done_copy) 
{	
	// 
	// 	constant memories
	// 
	cudaMemcpyToSymbolAsync(const_d1_h7b, host_size_d1_h7b, sizeof(int) * size_noab, 0, cudaMemcpyHostToDevice, *stream_id);
	cudaMemcpyToSymbolAsync(const_d2_p7b, host_size_d2_p7b, sizeof(int) * size_nvab, 0, cudaMemcpyHostToDevice, *stream_id);

	cudaMemcpyToSymbolAsync(const_s1_exec, host_exec_s1, sizeof(int) * (9), 0, cudaMemcpyHostToDevice, *stream_id);
	cudaMemcpyToSymbolAsync(const_d1_exec, host_exec_d1, sizeof(int) * (9 * size_noab), 0, cudaMemcpyHostToDevice, *stream_id);
	cudaMemcpyToSymbolAsync(const_d2_exec, host_exec_d2, sizeof(int) * (9 * size_nvab), 0, cudaMemcpyHostToDevice, *stream_id);

	// printf ("[new] s1: %d,%d,%d/%d,%d,%d/%d,%d,%d\n", host_exec_s1[0], host_exec_s1[1], host_exec_s1[2], host_exec_s1[3], host_exec_s1[4], host_exec_s1[5], host_exec_s1[6], host_exec_s1[7], host_exec_s1[8]);
	// for (int i = 0; i < (int)size_noab; i++) {
	// 	printf ("[new] noab: %d, d1: %d,%d,%d/%d,%d,%d/%d,%d,%d\n", i, 
	// 	host_exec_d1[0 + (i) * 9], host_exec_d1[1 + (i) * 9], host_exec_d1[2 + (i) * 9],
	// 	host_exec_d1[3 + (i) * 9], host_exec_d1[4 + (i) * 9], host_exec_d1[5 + (i) * 9],
	// 	host_exec_d1[6 + (i) * 9], host_exec_d1[7 + (i) * 9], host_exec_d1[8 + (i) * 9]);
	// }

	// for (int i = 0; i < (int)size_nvab; i++) {
	// 	printf ("[new] nvab: %d, d2: %d,%d,%d/%d,%d,%d/%d,%d,%d\n", i, 
	// 	host_exec_d2[0 + (i) * 9], host_exec_d2[1 + (i) * 9], host_exec_d2[2 + (i) * 9],
	// 	host_exec_d2[3 + (i) * 9], host_exec_d2[4 + (i) * 9], host_exec_d2[5 + (i) * 9],
	// 	host_exec_d2[6 + (i) * 9], host_exec_d2[7 + (i) * 9], host_exec_d2[8 + (i) * 9]);
	// }
	cudaEventRecord(done_copy);

	// 
	dim3 gridsize_1(numBlks);
	dim3 blocksize_1(16, 16);

	// 
	// printf ("[%s] called with # blocks: %d\n", __func__, numBlks);

#ifdef DEBUG_PRINT_KERNEL_TIME
  cudaEvent_t start_kernel;
  cudaEvent_t stop_kernel;
  cudaEventCreate(&start_kernel);
  cudaEventCreate(&stop_kernel);
  cudaEventRecord(start_kernel);
#endif

	// double host_energies_zero[2] = {0.0, 0.0};
	// cudaMemcpyAsync(dev_energies, host_energies_zero, sizeof(double) * 2, cudaMemcpyHostToDevice, *stream_id);

	// 
	// cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	// int maxbytes = 98304; // 96 KB
	// int maxbytes = 196608; // 192 KB
	// int maxbytes = 135168; // 132 KB
	// CUCHK(cudaFuncSetAttribute(fused_kernel_d2, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes));
#if 1
	fully_fused_kernel_ccsd_t_nvidia_tc_fp64<<<gridsize_1, blocksize_1, 2 * NUM_STAGE * 8 * STAGE_OFFSET, *stream_id>>>((int)size_noab, (int)size_nvab, 
		// 
		(int)size_max_dim_s1_t1, (int)size_max_dim_s1_v2, 
		(int)size_max_dim_d1_t2, (int)size_max_dim_d1_v2, 
		(int)size_max_dim_d2_t2, (int)size_max_dim_d2_v2, 
		// 
		dev_s1_t1_all, dev_s1_v2_all, 
		dev_d1_t2_all, dev_d1_v2_all, 
		dev_d2_t2_all, dev_d2_v2_all, 
		// 
		dev_energies, 
		dev_evl_sorted_h3b, dev_evl_sorted_h2b, dev_evl_sorted_h1b, 
		dev_evl_sorted_p6b, dev_evl_sorted_p5b, dev_evl_sorted_p4b, 
		// 
    (int)size_h3, (int)size_h2, (int)size_h1, 
		(int)size_p6, (int)size_p5, (int)size_p4, 
    CEIL(size_h3, SIZE_TILE_H3), CEIL(size_h2, SIZE_TILE_H2), CEIL(size_h1, SIZE_TILE_H1), 
		CEIL(size_p6, SIZE_TILE_P6), CEIL(size_p5, SIZE_TILE_P5), CEIL(size_p4, SIZE_TILE_P4));
	CUCHK(cudaGetLastError());
#endif

#ifdef DEBUG_PRINT_KERNEL_TIME
  cudaEventRecord(stop_kernel);
  cudaEventSynchronize(stop_kernel);
  float kernel_ms = 0;
  cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel);
  printf ("[%s] kernel: %f (ms)\n", __func__, kernel_ms);
#endif

	// double host_energies[2];
	// CUCHK(cudaMemcpy(host_energies, dev_energies, sizeof(double) * NUM_ENERGY, cudaMemcpyDeviceToHost));

	// *final_energy_4 = factor * host_energies[0];
	// *final_energy_5 = factor * host_energies[1];
	// printf ("[%s] (gpu) energy: %.10f, %.10f\n", __func__, *final_energy_4, *final_energy_5);
}
// end of (2) 3rd. Generation Tensor Cores (FP64)