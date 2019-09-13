#include "header.hpp"
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

#define MAX_NOAB				10
#define MAX_NVAB 				30

// #define DEBUG_TIME_FUSED_CCSD_T
// #define DEBUG_KERNEL_DETAIL
// #define DEBUG_HOST_ENERGIES

// 64 KB = 65536 bytes = 16384 (int) = 8192 (size_t)
// 9 * 9 * noab = 81 * noab 
__constant__ int const_list_s1_flags_offset[NUM_IA6_LOOPS * NUM_S1_EQUATIONS];
__constant__ int const_list_d1_flags_offset[NUM_IA6_LOOPS * NUM_D1_EQUATIONS * MAX_NOAB];
__constant__ int const_list_d2_flags_offset[NUM_IA6_LOOPS * NUM_D2_EQUATIONS * MAX_NVAB];
__constant__ int const_list_s1_problem_size[NUM_IA6_LOOPS * NUM_S1_INDEX];
__constant__ int const_list_d1_problem_size[NUM_IA6_LOOPS * NUM_D1_INDEX * MAX_NOAB];
__constant__ int const_list_d2_problem_size[NUM_IA6_LOOPS * NUM_D2_INDEX * MAX_NVAB];

#if 0
	//  s1
	size_t size_s1_t2_1 = size_p4 * size_h1;
	size_t size_s1_v2_1 = size_h3 * size_h2 * size_p6 * size_p5;
	size_t size_s1_t2_2 = size_p4 * size_h2;
	size_t size_s1_v2_2 = size_h3 * size_h1 * size_p6 * size_p5;
	size_t size_s1_t2_3 = size_p4 * size_h1;
	size_t size_s1_v2_3 = size_h3 * size_h2 * size_p6 * size_p5;
	size_t size_s1_t2_4 = size_p5 * size_h1;
	size_t size_s1_v2_4 = size_h3 * size_h2 * size_p6 * size_p4;
	size_t size_s1_t2_5 = size_p5 * size_h2;
	size_t size_s1_v2_5 = size_h3 * size_h1 * size_p6 * size_p4;
	size_t size_s1_t2_6 = size_p5 * size_h3;
	size_t size_s1_v2_6 = size_h2 * size_h1 * size_p6 * size_p4;
	size_t size_s1_t2_7 = size_p6 * size_h1;
	size_t size_s1_v2_7 = size_h3 * size_h2 * size_p5 * size_p4;
	size_t size_s1_t2_8 = size_p6 * size_h2;
	size_t size_s1_v2_8 = size_h3 * size_h1 * size_p5 * size_p4;
	size_t size_s1_t2_9 = size_p6 * size_h3;
	size_t size_s1_v2_9 = size_h2 * size_h1 * size_p5 * size_p4;

	//  d1
	size_t size_d1_t2_1 = size_h1 * size_p5 * size_p4 * size_h7;
	size_t size_d1_v2_1 = size_h7 * size_p6 * size_h2 * size_h3;
	size_t size_d1_t2_2 = size_h2 * size_p5 * size_p4 * size_h7;
	size_t size_d1_v2_2 = size_h7 * size_p6 * size_h1 * size_h3;
	size_t size_d1_t2_3 = size_h3 * size_p5 * size_p4 * size_h7;
	size_t size_d1_v2_3 = size_h7 * size_p6 * size_h1 * size_h2;
	size_t size_d1_t2_4 = size_h1 * size_p6 * size_p5 * size_h7;
	size_t size_d1_v2_4 = size_h7 * size_p4 * size_h2 * size_h3;
	size_t size_d1_t2_5 = size_h2 * size_p6 * size_p5 * size_h7;
	size_t size_d1_v2_5 = size_h7 * size_p4 * size_h1 * size_h3;
	size_t size_d1_t2_6 = size_h3 * size_p6 * size_p5 * size_h7;
	size_t size_d1_v2_6 = size_h7 * size_p4 * size_h1 * size_h2;
	size_t size_d1_t2_7 = size_h1 * size_p6 * size_p4 * size_h7;
	size_t size_d1_v2_7 = size_h7 * size_p5 * size_h2 * size_h3;
	size_t size_d1_t2_8 = size_h2 * size_p6 * size_p4 * size_h7;
	size_t size_d1_v2_8 = size_h7 * size_p5 * size_h1 * size_h3;
	size_t size_d1_t2_9 = size_h3 * size_p6 * size_p4 * size_h7;
	size_t size_d1_v2_9 = size_h7 * size_p5 * size_h1 * size_h2;

	//  d2
	size_t size_d2_t2_1 = size_h2 * size_h1 * size_p4 * size_p7;
	size_t size_d2_v2_1 = size_p5 * size_p6 * size_h3 * size_p7;
	size_t size_d2_t2_2 = size_h3 * size_h2 * size_p4 * size_p7;
	size_t size_d2_v2_2 = size_p5 * size_p6 * size_h1 * size_p7;
	size_t size_d2_t2_3 = size_h3 * size_h1 * size_p4 * size_p7;
	size_t size_d2_v2_3 = size_p5 * size_p6 * size_h2 * size_p7;
	size_t size_d2_t2_4 = size_h2 * size_h1 * size_p5 * size_p7;
	size_t size_d2_v2_4 = size_p4 * size_p6 * size_h3 * size_p7;
	size_t size_d2_t2_5 = size_h3 * size_h2 * size_p5 * size_p7;
	size_t size_d2_v2_5 = size_p4 * size_p6 * size_h1 * size_p7;
	size_t size_d2_t2_6 = size_h3 * size_h1 * size_p5 * size_p7;
	size_t size_d2_v2_6 = size_p4 * size_p6 * size_h2 * size_p7;
	size_t size_d2_t2_7 = size_h2 * size_h1 * size_p6 * size_p7;
	size_t size_d2_v2_7 = size_p4 * size_p5 * size_h3 * size_p7;
	size_t size_d2_t2_8 = size_h3 * size_h2 * size_p6 * size_p7;
	size_t size_d2_v2_8 = size_p4 * size_p5 * size_h1 * size_p7;
	size_t size_d2_t2_9 = size_h3 * size_h1 * size_p6 * size_p7;
	size_t size_d2_v2_9 = size_p4 * size_p5 * size_h2 * size_p7;
	 
	// 
	size_t stride_d1_t2_1 = 0;
	size_t stride_d1_t2_2 = stride_d1_t2_1 + size_d1_t2_1;
	size_t stride_d1_t2_3 = stride_d1_t2_2 + size_d1_t2_2;
	size_t stride_d1_t2_4 = stride_d1_t2_3 + size_d1_t2_3;
	size_t stride_d1_t2_5 = stride_d1_t2_4 + size_d1_t2_4;
	size_t stride_d1_t2_6 = stride_d1_t2_5 + size_d1_t2_5;
	size_t stride_d1_t2_7 = stride_d1_t2_6 + size_d1_t2_6;
	size_t stride_d1_t2_8 = stride_d1_t2_7 + size_d1_t2_7;
	size_t stride_d1_t2_9 = stride_d1_t2_8 + size_d1_t2_8;
	
	size_t stride_d1_v2_1 = 0;
	size_t stride_d1_v2_2 = stride_d1_v2_1 + size_d1_v2_1;
	size_t stride_d1_v2_3 = stride_d1_v2_2 + size_d1_v2_2;
	size_t stride_d1_v2_4 = stride_d1_v2_3 + size_d1_v2_3;
	size_t stride_d1_v2_5 = stride_d1_v2_4 + size_d1_v2_4;
	size_t stride_d1_v2_6 = stride_d1_v2_5 + size_d1_v2_5;
	size_t stride_d1_v2_7 = stride_d1_v2_6 + size_d1_v2_6;
	size_t stride_d1_v2_8 = stride_d1_v2_7 + size_d1_v2_7;
	size_t stride_d1_v2_9 = stride_d1_v2_8 + size_d1_v2_8;

	// 
	size_t stride_d2_t2_1 = 0;
	size_t stride_d2_t2_2 = stride_d2_t2_1 + size_d2_t2_1;
	size_t stride_d2_t2_3 = stride_d2_t2_2 + size_d2_t2_2;
	size_t stride_d2_t2_4 = stride_d2_t2_3 + size_d2_t2_3;
	size_t stride_d2_t2_5 = stride_d2_t2_4 + size_d2_t2_4;
	size_t stride_d2_t2_6 = stride_d2_t2_5 + size_d2_t2_5;
	size_t stride_d2_t2_7 = stride_d2_t2_6 + size_d2_t2_6;
	size_t stride_d2_t2_8 = stride_d2_t2_7 + size_d2_t2_7;
	size_t stride_d2_t2_9 = stride_d2_t2_8 + size_d2_t2_8;
	
	size_t stride_d2_v2_1 = 0;
	size_t stride_d2_v2_2 = stride_d2_v2_1 + size_d2_v2_1;
	size_t stride_d2_v2_3 = stride_d2_v2_2 + size_d2_v2_2;
	size_t stride_d2_v2_4 = stride_d2_v2_3 + size_d2_v2_3;
	size_t stride_d2_v2_5 = stride_d2_v2_4 + size_d2_v2_4;
	size_t stride_d2_v2_6 = stride_d2_v2_5 + size_d2_v2_5;
	size_t stride_d2_v2_7 = stride_d2_v2_6 + size_d2_v2_6;
	size_t stride_d2_v2_8 = stride_d2_v2_7 + size_d2_v2_7;
	size_t stride_d2_v2_9 = stride_d2_v2_8 + size_d2_v2_8;

	// 
	size_t stride_s1_t2_1 = 0;
	size_t stride_s1_t2_2 = stride_s1_t2_1 + size_s1_t2_1;
	size_t stride_s1_t2_3 = stride_s1_t2_2 + size_s1_t2_2;
	size_t stride_s1_t2_4 = stride_s1_t2_3 + size_s1_t2_3;
	size_t stride_s1_t2_5 = stride_s1_t2_4 + size_s1_t2_4;
	size_t stride_s1_t2_6 = stride_s1_t2_5 + size_s1_t2_5;
	size_t stride_s1_t2_7 = stride_s1_t2_6 + size_s1_t2_6;
	size_t stride_s1_t2_8 = stride_s1_t2_7 + size_s1_t2_7;
	size_t stride_s1_t2_9 = stride_s1_t2_8 + size_s1_t2_8;
	
	size_t stride_s1_v2_1 = 0;
	size_t stride_s1_v2_2 = stride_s1_v2_1 + size_s1_v2_1;
	size_t stride_s1_v2_3 = stride_s1_v2_2 + size_s1_v2_2;
	size_t stride_s1_v2_4 = stride_s1_v2_3 + size_s1_v2_3;
	size_t stride_s1_v2_5 = stride_s1_v2_4 + size_s1_v2_4;
	size_t stride_s1_v2_6 = stride_s1_v2_5 + size_s1_v2_5;
	size_t stride_s1_v2_7 = stride_s1_v2_6 + size_s1_v2_6;
	size_t stride_s1_v2_8 = stride_s1_v2_7 + size_s1_v2_7;
	size_t stride_s1_v2_9 = stride_s1_v2_8 + size_s1_v2_8;
#endif

// 
__global__ 
void jk_ccsd_t_fully_fused_kernel(	int size_noab, int size_nvab, 
									// 	common
									int size_max_dim_s1_t2, int size_max_dim_s1_v2, 
									int size_max_dim_d1_t2, int size_max_dim_d1_v2, 
									int size_max_dim_d2_t2, int size_max_dim_d2_v2, 
									//  doubles (sd1)
									double* dev_d1_t2_all,  double* dev_d1_v2_all,
									//  doubles (sd2)
									double* dev_d2_t2_all, double* dev_d2_v2_all,
									//  single 	(s1)
									double* dev_s1_t2_all, double* dev_s1_v2_all,
									//  energies
									const double* dev_evl_sorted_h1b, const double* dev_evl_sorted_h2b, const double* dev_evl_sorted_h3b,
									const double* dev_evl_sorted_p4b, const double* dev_evl_sorted_p5b, const double* dev_evl_sorted_p6b, 
									// 	not-fully reduced results
									double* reduced_energy,
									//  common
									int num_blks_h3b, int num_blks_h2b, int num_blks_h1b, 
									int num_blks_p6b, int num_blks_p5b, int num_blks_p4b, 
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
		rng_h1 = FUSION_SIZE_SLICE_1_H1;
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
		rng_p4 = base_size_p4b % FUSION_SIZE_SLICE_1_P4;
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
	//  loops
	// 
	#pragma unroll 1
	for (int iter_ia6 = 0; iter_ia6 < NUM_IA6_LOOPS; iter_ia6++)
	{
		//  doubles (d1 and d2) 
		{
			//  d1-top: sd1_1, 2 and 3 
			#pragma unroll 1
			for (int iter_noab = 0; iter_noab < size_noab; iter_noab++)
			{
				// 
				int flag_d1_1 = const_list_d1_flags_offset[0 + (iter_noab + (iter_ia6) * size_noab) * NUM_D1_EQUATIONS];
				int flag_d1_2 = const_list_d1_flags_offset[1 + (iter_noab + (iter_ia6) * size_noab) * NUM_D1_EQUATIONS];
				int flag_d1_3 = const_list_d1_flags_offset[2 + (iter_noab + (iter_ia6) * size_noab) * NUM_D1_EQUATIONS];

				// 
				// int local_d1_size_idx_h1b = const_list_d1_problem_size[0 + (iter_noab + (iter_ia6) * size_noab) * NUM_D1_INDEX];
				// int local_d1_size_idx_h2b = const_list_d1_problem_size[1 + (iter_noab + (iter_ia6) * size_noab) * NUM_D1_INDEX];
				// int local_d1_size_idx_h3b = const_list_d1_problem_size[2 + (iter_noab + (iter_ia6) * size_noab) * NUM_D1_INDEX];
				// // int local_d1_size_idx_h7b = const_list_d1_problem_size[3 + (iter_noab + (iter_ia6) * size_noab) * NUM_D1_INDEX];
				// int local_d1_size_idx_p4b = const_list_d1_problem_size[4 + (iter_noab + (iter_ia6) * size_noab) * NUM_D1_INDEX];
				// int local_d1_size_idx_p5b = const_list_d1_problem_size[5 + (iter_noab + (iter_ia6) * size_noab) * NUM_D1_INDEX];
				// int local_d1_size_idx_p6b = const_list_d1_problem_size[6 + (iter_noab + (iter_ia6) * size_noab) * NUM_D1_INDEX];
				base_size_h1b = const_list_d1_problem_size[0 + (iter_noab + (iter_ia6) * size_noab) * NUM_D1_INDEX];
				base_size_h2b = const_list_d1_problem_size[1 + (iter_noab + (iter_ia6) * size_noab) * NUM_D1_INDEX];
				base_size_h3b = const_list_d1_problem_size[2 + (iter_noab + (iter_ia6) * size_noab) * NUM_D1_INDEX];
				base_size_h7b = const_list_d1_problem_size[3 + (iter_noab + (iter_ia6) * size_noab) * NUM_D1_INDEX];
				base_size_p4b = const_list_d1_problem_size[4 + (iter_noab + (iter_ia6) * size_noab) * NUM_D1_INDEX];
				base_size_p5b = const_list_d1_problem_size[5 + (iter_noab + (iter_ia6) * size_noab) * NUM_D1_INDEX];
				base_size_p6b = const_list_d1_problem_size[6 + (iter_noab + (iter_ia6) * size_noab) * NUM_D1_INDEX];

				//	otheres according to the above problem-sizes
				//	(1) num_blks_h/p*b
				// num_blks_h1b = CEIL(local_d1_size_idx_h1b, FUSION_SIZE_SLICE_1_H1);
				// num_blks_h2b = CEIL(local_d1_size_idx_h2b, FUSION_SIZE_SLICE_1_H2);
				// num_blks_h3b = CEIL(local_d1_size_idx_h3b, FUSION_SIZE_SLICE_1_H3);
				// num_blks_p4b = CEIL(local_d1_size_idx_p4b, FUSION_SIZE_SLICE_1_P4);
				// num_blks_p5b = CEIL(local_d1_size_idx_p5b, FUSION_SIZE_SLICE_1_P5);
				// num_blks_p6b = CEIL(local_d1_size_idx_p6b, FUSION_SIZE_SLICE_1_P6);
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
					double* tmp_dev_d1_t2 = dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_1;
					double* tmp_dev_d1_v2 = dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_1;

					// 
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
					double* tmp_dev_d1_t2 = dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_2;
					double* tmp_dev_d1_v2 = dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_2;

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
					double* tmp_dev_d1_t2 = dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_3;
					double* tmp_dev_d1_v2 = dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_3;
					
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
				int flag_d2_7 = const_list_d2_flags_offset[6 + (iter_nvab + (iter_ia6) * size_nvab) * NUM_D2_EQUATIONS];
				int flag_d2_8 = const_list_d2_flags_offset[7 + (iter_nvab + (iter_ia6) * size_nvab) * NUM_D2_EQUATIONS];
				int flag_d2_9 = const_list_d2_flags_offset[8 + (iter_nvab + (iter_ia6) * size_nvab) * NUM_D2_EQUATIONS];

				// 
				// int local_d2_size_idx_h1b = const_list_d2_problem_size[0 + (iter_nvab + (iter_ia6) * size_nvab) * NUM_D2_INDEX];
				// int local_d2_size_idx_h2b = const_list_d2_problem_size[1 + (iter_nvab + (iter_ia6) * size_nvab) * NUM_D2_INDEX];
				// int local_d2_size_idx_h3b = const_list_d2_problem_size[2 + (iter_nvab + (iter_ia6) * size_nvab) * NUM_D2_INDEX];
				// int local_d2_size_idx_p4b = const_list_d2_problem_size[3 + (iter_nvab + (iter_ia6) * size_nvab) * NUM_D2_INDEX];
				// int local_d2_size_idx_p5b = const_list_d2_problem_size[4 + (iter_nvab + (iter_ia6) * size_nvab) * NUM_D2_INDEX];
				// int local_d2_size_idx_p6b = const_list_d2_problem_size[5 + (iter_nvab + (iter_ia6) * size_nvab) * NUM_D2_INDEX];
				// int local_d2_size_idx_p7b = const_list_d2_problem_size[6 + (iter_nvab + (iter_ia6) * size_nvab) * NUM_D2_INDEX];
				base_size_h1b = const_list_d2_problem_size[0 + (iter_nvab + (iter_ia6) * size_nvab) * NUM_D2_INDEX];
				base_size_h2b = const_list_d2_problem_size[1 + (iter_nvab + (iter_ia6) * size_nvab) * NUM_D2_INDEX];
				base_size_h3b = const_list_d2_problem_size[2 + (iter_nvab + (iter_ia6) * size_nvab) * NUM_D2_INDEX];
				base_size_p4b = const_list_d2_problem_size[3 + (iter_nvab + (iter_ia6) * size_nvab) * NUM_D2_INDEX];
				base_size_p5b = const_list_d2_problem_size[4 + (iter_nvab + (iter_ia6) * size_nvab) * NUM_D2_INDEX];
				base_size_p6b = const_list_d2_problem_size[5 + (iter_nvab + (iter_ia6) * size_nvab) * NUM_D2_INDEX];
				base_size_p7b = const_list_d2_problem_size[6 + (iter_nvab + (iter_ia6) * size_nvab) * NUM_D2_INDEX];

				//	otheres according to the above problem-sizes
				//	(1) num_blks_h/p*b
				// num_blks_h1b = CEIL(local_d2_size_idx_h1b, FUSION_SIZE_SLICE_1_H1);
				// num_blks_h2b = CEIL(local_d2_size_idx_h2b, FUSION_SIZE_SLICE_1_H2);
				// num_blks_h3b = CEIL(local_d2_size_idx_h3b, FUSION_SIZE_SLICE_1_H3);
				// num_blks_p4b = CEIL(local_d2_size_idx_p4b, FUSION_SIZE_SLICE_1_P4);
				// num_blks_p5b = CEIL(local_d2_size_idx_p5b, FUSION_SIZE_SLICE_1_P5);
				// num_blks_p6b = CEIL(local_d2_size_idx_p6b, FUSION_SIZE_SLICE_1_P6);
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
					double* tmp_dev_d2_t2_7 = dev_d2_t2_all + size_max_dim_d2_t2 * flag_d2_7;//const_list_d2_flags_offset[local_offset];
					double* tmp_dev_d2_v2_7 = dev_d2_v2_all + size_max_dim_d2_v2 * flag_d2_7;//const_list_d2_flags_offset[local_offset];
					
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
					double* tmp_dev_d2_t2_8 = dev_d2_t2_all + size_max_dim_d2_t2 * flag_d2_8;//const_list_d2_flags_offset[local_offset];
					double* tmp_dev_d2_v2_8 = dev_d2_v2_all + size_max_dim_d2_v2 * flag_d2_8;//const_list_d2_flags_offset[local_offset];

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
					double* tmp_dev_d2_t2_9 = dev_d2_t2_all + size_max_dim_d2_t2 * flag_d2_9;//const_list_d2_flags_offset[local_offset];
					double* tmp_dev_d2_v2_9 = dev_d2_v2_all + size_max_dim_d2_v2 * flag_d2_9;//const_list_d2_flags_offset[local_offset];

					internal_upperbound = 0;
					#pragma unroll 1
					for (int l = 0; l < base_size_p7b; l+= FUSION_SIZE_INT_UNIT)
					{
						// Part: Generalized Contraction Index (p7b)
						// internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
						internal_offset = (l + FUSION_SIZE_INT_UNIT) - base_size_p7b;
						if (internal_offset > rng_p6) internal_upperbound = internal_offset;
				
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
		}
	}
	

	// 
	//  register_transpose (top - bottom)
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
	#pragma unroll 1
	for (int iter_ia6 = 0; iter_ia6 < NUM_IA6_LOOPS; iter_ia6++)
	{
		
		//  doubles (d1 and d2) 
		{
		// #if 0
			//  d1-bottom: sd1_4, 5 , 6 , 7 , 8 and 9.
			#pragma unroll 1
			for (int iter_noab = 0; iter_noab < size_noab; iter_noab++)
			{
				// 	flags
				int flag_d1_4 = const_list_d1_flags_offset[3 + (iter_noab + (iter_ia6) * size_noab) * NUM_D1_EQUATIONS];
				int flag_d1_5 = const_list_d1_flags_offset[4 + (iter_noab + (iter_ia6) * size_noab) * NUM_D1_EQUATIONS];
				int flag_d1_6 = const_list_d1_flags_offset[5 + (iter_noab + (iter_ia6) * size_noab) * NUM_D1_EQUATIONS];
				int flag_d1_7 = const_list_d1_flags_offset[6 + (iter_noab + (iter_ia6) * size_noab) * NUM_D1_EQUATIONS];
				int flag_d1_8 = const_list_d1_flags_offset[7 + (iter_noab + (iter_ia6) * size_noab) * NUM_D1_EQUATIONS];
				int flag_d1_9 = const_list_d1_flags_offset[8 + (iter_noab + (iter_ia6) * size_noab) * NUM_D1_EQUATIONS];

				// 
				// int local_d1_size_idx_h1b = const_list_d1_problem_size[0 + (iter_noab + (iter_ia6) * size_noab) * NUM_D1_INDEX];
				// int local_d1_size_idx_h2b = const_list_d1_problem_size[1 + (iter_noab + (iter_ia6) * size_noab) * NUM_D1_INDEX];
				// int local_d1_size_idx_h3b = const_list_d1_problem_size[2 + (iter_noab + (iter_ia6) * size_noab) * NUM_D1_INDEX];
				// // int local_d1_size_idx_h7b = const_list_d1_problem_size[3 + (iter_noab + (iter_ia6) * size_noab) * NUM_D1_INDEX];
				// int local_d1_size_idx_p4b = const_list_d1_problem_size[4 + (iter_noab + (iter_ia6) * size_noab) * NUM_D1_INDEX];
				// int local_d1_size_idx_p5b = const_list_d1_problem_size[5 + (iter_noab + (iter_ia6) * size_noab) * NUM_D1_INDEX];
				// int local_d1_size_idx_p6b = const_list_d1_problem_size[6 + (iter_noab + (iter_ia6) * size_noab) * NUM_D1_INDEX];
				base_size_h1b = const_list_d1_problem_size[0 + (iter_noab + (iter_ia6) * size_noab) * NUM_D1_INDEX];
				base_size_h2b = const_list_d1_problem_size[1 + (iter_noab + (iter_ia6) * size_noab) * NUM_D1_INDEX];
				base_size_h3b = const_list_d1_problem_size[2 + (iter_noab + (iter_ia6) * size_noab) * NUM_D1_INDEX];
				base_size_h7b = const_list_d1_problem_size[3 + (iter_noab + (iter_ia6) * size_noab) * NUM_D1_INDEX];
				base_size_p4b = const_list_d1_problem_size[4 + (iter_noab + (iter_ia6) * size_noab) * NUM_D1_INDEX];
				base_size_p5b = const_list_d1_problem_size[5 + (iter_noab + (iter_ia6) * size_noab) * NUM_D1_INDEX];
				base_size_p6b = const_list_d1_problem_size[6 + (iter_noab + (iter_ia6) * size_noab) * NUM_D1_INDEX];

				//	otheres according to the above problem-sizes
				//	(1) num_blks_h/p*b
				// num_blks_h1b = CEIL(local_d1_size_idx_h1b, FUSION_SIZE_SLICE_1_H1);
				// num_blks_h2b = CEIL(local_d1_size_idx_h2b, FUSION_SIZE_SLICE_1_H2);
				// num_blks_h3b = CEIL(local_d1_size_idx_h3b, FUSION_SIZE_SLICE_1_H3);
				// num_blks_p4b = CEIL(local_d1_size_idx_p4b, FUSION_SIZE_SLICE_1_P4);
				// num_blks_p5b = CEIL(local_d1_size_idx_p5b, FUSION_SIZE_SLICE_1_P5);
				// num_blks_p6b = CEIL(local_d1_size_idx_p6b, FUSION_SIZE_SLICE_1_P6);
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
					double* tmp_dev_d1_t2_4 = dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_4;
					double* tmp_dev_d1_v2_4 = dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_4;

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
					double* tmp_dev_d1_t2_5 = dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_5;
					double* tmp_dev_d1_v2_5 = dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_5;

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
					double* tmp_dev_d1_t2_6 = dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_6;
					double* tmp_dev_d1_v2_6 = dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_6;

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
					double* tmp_dev_d1_t2_7 = dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_7;
					double* tmp_dev_d1_v2_7 = dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_7;

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
					double* tmp_dev_d1_t2_8 = dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_8;
					double* tmp_dev_d1_v2_8 = dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_8;

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
					double* tmp_dev_d1_t2_9 = dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_9;
					double* tmp_dev_d1_v2_9 = dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_9;

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
				int flag_d2_1 = const_list_d2_flags_offset[0 + (iter_nvab + (iter_ia6) * size_nvab) * NUM_D2_EQUATIONS];
				int flag_d2_2 = const_list_d2_flags_offset[1 + (iter_nvab + (iter_ia6) * size_nvab) * NUM_D2_EQUATIONS];
				int flag_d2_3 = const_list_d2_flags_offset[2 + (iter_nvab + (iter_ia6) * size_nvab) * NUM_D2_EQUATIONS];
				int flag_d2_4 = const_list_d2_flags_offset[3 + (iter_nvab + (iter_ia6) * size_nvab) * NUM_D2_EQUATIONS];
				int flag_d2_5 = const_list_d2_flags_offset[4 + (iter_nvab + (iter_ia6) * size_nvab) * NUM_D2_EQUATIONS];
				int flag_d2_6 = const_list_d2_flags_offset[5 + (iter_nvab + (iter_ia6) * size_nvab) * NUM_D2_EQUATIONS];

				// 
				// int local_d2_size_idx_h1b = const_list_d2_problem_size[0 + (iter_nvab + (iter_ia6) * size_nvab) * NUM_D2_INDEX];
				// int local_d2_size_idx_h2b = const_list_d2_problem_size[1 + (iter_nvab + (iter_ia6) * size_nvab) * NUM_D2_INDEX];
				// int local_d2_size_idx_h3b = const_list_d2_problem_size[2 + (iter_nvab + (iter_ia6) * size_nvab) * NUM_D2_INDEX];
				// int local_d2_size_idx_p4b = const_list_d2_problem_size[3 + (iter_nvab + (iter_ia6) * size_nvab) * NUM_D2_INDEX];
				// int local_d2_size_idx_p5b = const_list_d2_problem_size[4 + (iter_nvab + (iter_ia6) * size_nvab) * NUM_D2_INDEX];
				// int local_d2_size_idx_p6b = const_list_d2_problem_size[5 + (iter_nvab + (iter_ia6) * size_nvab) * NUM_D2_INDEX];
				// int local_d2_size_idx_p7b = const_list_d2_problem_size[6 + (iter_nvab + (iter_ia6) * size_nvab) * NUM_D2_INDEX];
				base_size_h1b = const_list_d2_problem_size[0 + (iter_nvab + (iter_ia6) * size_nvab) * NUM_D2_INDEX];
				base_size_h2b = const_list_d2_problem_size[1 + (iter_nvab + (iter_ia6) * size_nvab) * NUM_D2_INDEX];
				base_size_h3b = const_list_d2_problem_size[2 + (iter_nvab + (iter_ia6) * size_nvab) * NUM_D2_INDEX];
				base_size_p4b = const_list_d2_problem_size[3 + (iter_nvab + (iter_ia6) * size_nvab) * NUM_D2_INDEX];
				base_size_p5b = const_list_d2_problem_size[4 + (iter_nvab + (iter_ia6) * size_nvab) * NUM_D2_INDEX];
				base_size_p6b = const_list_d2_problem_size[5 + (iter_nvab + (iter_ia6) * size_nvab) * NUM_D2_INDEX];
				base_size_p7b = const_list_d2_problem_size[6 + (iter_nvab + (iter_ia6) * size_nvab) * NUM_D2_INDEX];

				//	otheres according to the above problem-sizes
				//	(1) num_blks_h/p*b
				// num_blks_h1b = CEIL(local_d2_size_idx_h1b, FUSION_SIZE_SLICE_1_H1);
				// num_blks_h2b = CEIL(local_d2_size_idx_h2b, FUSION_SIZE_SLICE_1_H2);
				// num_blks_h3b = CEIL(local_d2_size_idx_h3b, FUSION_SIZE_SLICE_1_H3);
				// num_blks_p4b = CEIL(local_d2_size_idx_p4b, FUSION_SIZE_SLICE_1_P4);
				// num_blks_p5b = CEIL(local_d2_size_idx_p5b, FUSION_SIZE_SLICE_1_P5);
				// num_blks_p6b = CEIL(local_d2_size_idx_p6b, FUSION_SIZE_SLICE_1_P6);
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
					double* tmp_dev_d2_t2_1 = dev_d2_t2_all + size_max_dim_d2_t2 * flag_d2_1;
					double* tmp_dev_d2_v2_1 = dev_d2_v2_all + size_max_dim_d2_v2 * flag_d2_1;

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
					double* tmp_dev_d2_t2_2 = dev_d2_t2_all + size_max_dim_d2_t2 * flag_d2_2;
					double* tmp_dev_d2_v2_2 = dev_d2_v2_all + size_max_dim_d2_v2 * flag_d2_2;

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
					double* tmp_dev_d2_t2_3 = dev_d2_t2_all + size_max_dim_d2_t2 * flag_d2_3;
					double* tmp_dev_d2_v2_3 = dev_d2_v2_all + size_max_dim_d2_v2 * flag_d2_3;

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
					double* tmp_dev_d2_t2_4 = dev_d2_t2_all + size_max_dim_d2_t2 * flag_d2_4;
					double* tmp_dev_d2_v2_4 = dev_d2_v2_all + size_max_dim_d2_v2 * flag_d2_4;

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
					double* tmp_dev_d2_t2_5 = dev_d2_t2_all + size_max_dim_d2_t2 * flag_d2_5;
					double* tmp_dev_d2_v2_5 = dev_d2_v2_all + size_max_dim_d2_v2 * flag_d2_5;

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
					double* tmp_dev_d2_t2_6 = dev_d2_t2_all + size_max_dim_d2_t2 * flag_d2_6;
					double* tmp_dev_d2_v2_6 = dev_d2_v2_all + size_max_dim_d2_v2 * flag_d2_6;

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
		
		}

		//  singles (s1)
		{	
			// 	flags
			int flag_s1_1 = const_list_s1_flags_offset[0 + iter_ia6 * NUM_S1_EQUATIONS];
			int flag_s1_2 = const_list_s1_flags_offset[1 + iter_ia6 * NUM_S1_EQUATIONS];
			int flag_s1_3 = const_list_s1_flags_offset[2 + iter_ia6 * NUM_S1_EQUATIONS];
			int flag_s1_4 = const_list_s1_flags_offset[3 + iter_ia6 * NUM_S1_EQUATIONS];
			int flag_s1_5 = const_list_s1_flags_offset[4 + iter_ia6 * NUM_S1_EQUATIONS];
			int flag_s1_6 = const_list_s1_flags_offset[5 + iter_ia6 * NUM_S1_EQUATIONS];
			int flag_s1_7 = const_list_s1_flags_offset[6 + iter_ia6 * NUM_S1_EQUATIONS];
			int flag_s1_8 = const_list_s1_flags_offset[7 + iter_ia6 * NUM_S1_EQUATIONS];
			int flag_s1_9 = const_list_s1_flags_offset[8 + iter_ia6 * NUM_S1_EQUATIONS];

			// if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0)
			// {
			// 	printf ("[Device][s1] ia6=%d, flag_s1_(1, 2, 3, 4, 5, 6, 7, 8, 9) = (%2d,%2d,%2d,%2d,%2d,%2d,%2d,%2d,%2d)\n", iter_ia6, flag_s1_1, flag_s1_2, flag_s1_3, flag_s1_4, flag_s1_5, flag_s1_6, flag_s1_7, flag_s1_8, flag_s1_9);
			// }

			// 	problem-sizes
			// int local_s1_size_idx_h1b = const_list_s1_problem_size[0 + iter_ia6 * NUM_S1_INDEX];
			// int local_s1_size_idx_h2b = const_list_s1_problem_size[1 + iter_ia6 * NUM_S1_INDEX];
			// int local_s1_size_idx_h3b = const_list_s1_problem_size[2 + iter_ia6 * NUM_S1_INDEX];
			// int local_s1_size_idx_p4b = const_list_s1_problem_size[3 + iter_ia6 * NUM_S1_INDEX];
			// int local_s1_size_idx_p5b = const_list_s1_problem_size[4 + iter_ia6 * NUM_S1_INDEX];
			// int local_s1_size_idx_p6b = const_list_s1_problem_size[5 + iter_ia6 * NUM_S1_INDEX];
			base_size_h1b = const_list_s1_problem_size[0 + iter_ia6 * NUM_S1_INDEX];
			base_size_h2b = const_list_s1_problem_size[1 + iter_ia6 * NUM_S1_INDEX];
			base_size_h3b = const_list_s1_problem_size[2 + iter_ia6 * NUM_S1_INDEX];
			base_size_p4b = const_list_s1_problem_size[3 + iter_ia6 * NUM_S1_INDEX];
			base_size_p5b = const_list_s1_problem_size[4 + iter_ia6 * NUM_S1_INDEX];
			base_size_p6b = const_list_s1_problem_size[5 + iter_ia6 * NUM_S1_INDEX];

			//	otheres according to the above problem-sizes
			//	(1) num_blks_h/p*b
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

			//                                        "x"         "x"
			//  >> s1_1:   t3[h3,h2,h1,p6,p5,p4] -= t2[p4,h1] * v2[h3,h2,p6,p5]
			//
			if (flag_s1_1 >= 0)	// these if-conditions make 100 ms..
			{
				//
				double* tmp_dev_s1_t2_1 = dev_s1_t2_all + size_max_dim_s1_t2 * flag_s1_1;
				double* tmp_dev_s1_v2_1 = dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_1;

				if (idx_h3 < rng_p4 && idx_h2 < rng_h1 && idx_p6 == 0 && idx_h1 == 0)
				sm_a[0][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_1_P4] = tmp_dev_s1_t2_1[str_blk_idx_p4 + idx_h3 + (str_blk_idx_h1 + idx_h2) * base_size_p4b];

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
				reg_singles[0][0] += temp_av * temp_bv[0];// * reg_singles[0][0];
				reg_singles[0][1] += temp_av * temp_bv[1];// * reg_singles[0][1];
				reg_singles[0][2] += temp_av * temp_bv[2];// * reg_singles[0][2];
				reg_singles[0][3] += temp_av * temp_bv[3];// * reg_singles[0][3];

				temp_av = sm_a[0][1 + (idx_h1) * 4];
				
				reg_singles[1][0] += temp_av * temp_bv[0];// * reg_singles[1][0];
				reg_singles[1][1] += temp_av * temp_bv[1];// * reg_singles[1][1];
				reg_singles[1][2] += temp_av * temp_bv[2];// * reg_singles[1][2];
				reg_singles[1][3] += temp_av * temp_bv[3];// * reg_singles[1][3];
				
				temp_av = sm_a[0][2 + (idx_h1) * 4];

				reg_singles[2][0] += temp_av * temp_bv[0];// * reg_singles[2][0];
				reg_singles[2][1] += temp_av * temp_bv[1];// * reg_singles[2][1];
				reg_singles[2][2] += temp_av * temp_bv[2];// * reg_singles[2][2];
				reg_singles[2][3] += temp_av * temp_bv[3];// * reg_singles[2][3];

				temp_av = sm_a[0][3 + (idx_h1) * 4];

				reg_singles[3][0] += temp_av * temp_bv[0];// * reg_singles[3][0];
				reg_singles[3][1] += temp_av * temp_bv[1];// * reg_singles[3][1];
				reg_singles[3][2] += temp_av * temp_bv[2];// * reg_singles[3][2];
				reg_singles[3][3] += temp_av * temp_bv[3];// * reg_singles[3][3];
				__syncthreads();
			}

			//                                        "x1,x2"     "x1,x2,x3,y1"
			//  >> s1_2:   t3[h3,h2,h1,p6,p5,p4] -= t2[p4,h2] * v2[h3,h1,p6,p5] (h3,h2,p6), (h1)
			//
			if (flag_s1_2 >= 0)	// these if-conditions make 100 ms..
			{
				// 
				double* tmp_dev_s1_t2_2 = dev_s1_t2_all + size_max_dim_s1_t2 * flag_s1_2;
				double* tmp_dev_s1_v2_2 = dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_2;

				if (idx_h3 < rng_p4 && idx_h2 < rng_h2 && idx_p6 == 0 && idx_h1 == 0)
				sm_a[0][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_1_P4] = tmp_dev_s1_t2_2[str_blk_idx_p4 + idx_h3 + (str_blk_idx_h2 + idx_h2) * base_size_p4b];
				
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
				double* tmp_dev_s1_t2_3 = dev_s1_t2_all + size_max_dim_s1_t2 * flag_s1_3;
				double* tmp_dev_s1_v2_3 = dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_3;

				if (idx_h3 < rng_p4 && idx_h2 < rng_h3 && idx_p6 == 0 && idx_h1 == 0)
				sm_a[0][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_1_P4] = tmp_dev_s1_t2_3[str_blk_idx_p4 + idx_h3 + (str_blk_idx_h3 + idx_h2) * base_size_p4b];

				if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6 && idx_h1 < rng_p5)
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
				double* tmp_dev_s1_t2_4 = dev_s1_t2_all + size_max_dim_s1_t2 * flag_s1_4;
				double* tmp_dev_s1_v2_4 = dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_4;

				if (idx_h3 < rng_p5 && idx_h2 < rng_h1 && idx_p6 == 0 && idx_h1 == 0)
				sm_a[0][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_1_P5] = tmp_dev_s1_t2_4[str_blk_idx_p5 + idx_h3 + (str_blk_idx_h1 + idx_h2) * base_size_p5b];
				
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
				reg_singles[0][0] += temp_av * temp_bv[0];
				reg_singles[1][0] += temp_av * temp_bv[1];
				reg_singles[2][0] += temp_av * temp_bv[2];
				reg_singles[3][0] += temp_av * temp_bv[3];

				temp_av = sm_a[0][1 + (idx_h1) * 4];

				reg_singles[0][1] += temp_av * temp_bv[0];
				reg_singles[1][1] += temp_av * temp_bv[1];
				reg_singles[2][1] += temp_av * temp_bv[2];
				reg_singles[3][1] += temp_av * temp_bv[3];

				temp_av = sm_a[0][2 + (idx_h1) * 4];

				reg_singles[0][2] += temp_av * temp_bv[0];
				reg_singles[1][2] += temp_av * temp_bv[1];
				reg_singles[2][2] += temp_av * temp_bv[2];
				reg_singles[3][2] += temp_av * temp_bv[3];

				temp_av = sm_a[0][3 + (idx_h1) * 4];

				reg_singles[0][3] += temp_av * temp_bv[0];
				reg_singles[1][3] += temp_av * temp_bv[1];
				reg_singles[2][3] += temp_av * temp_bv[2];
				reg_singles[3][3] += temp_av * temp_bv[3];
				__syncthreads();
			}

			//
			//  >> s1_5:   t3[h3,h2,h1,p6,p5,p4] -= t2[p5,h2] * v2[h3,h1,p6,p4] (h3,h2,p6), (h1)
			//
			if (flag_s1_5 >= 0)	// these if-conditions make 100 ms..
			{
				// 
				double* tmp_dev_s1_t2_5 = dev_s1_t2_all + size_max_dim_s1_t2 * flag_s1_5;
				double* tmp_dev_s1_v2_5 = dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_5;

				if (idx_h3 < rng_p5 && idx_h2 < rng_h2 && idx_p6 == 0 && idx_h1 == 0)
				sm_a[0][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_1_P5] = tmp_dev_s1_t2_5[str_blk_idx_p5 + idx_h3 + (str_blk_idx_h2 + idx_h2) * base_size_p5b];

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
				double* tmp_dev_s1_t2_6 = dev_s1_t2_all + size_max_dim_s1_t2 * flag_s1_6;
				double* tmp_dev_s1_v2_6 = dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_6;

				if (idx_h3 < rng_p5 && idx_h2 < rng_h3 && idx_p6 == 0 && idx_h1 == 0)
				sm_a[0][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_1_P5] = tmp_dev_s1_t2_6[str_blk_idx_p5 + idx_h3 + (str_blk_idx_h3 + idx_h2) * base_size_p5b];

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
				double* tmp_dev_s1_t2_7 = dev_s1_t2_all + size_max_dim_s1_t2 * flag_s1_7;
				double* tmp_dev_s1_v2_7 = dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_7;

				if (idx_h3 < rng_p6 && idx_h2 < rng_h1 && idx_p6 == 0 && idx_h1 == 0)
				sm_a[0][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_1_P6] = tmp_dev_s1_t2_7[str_blk_idx_p6 + idx_h3 + (str_blk_idx_h1 + idx_h2) * base_size_p6b];
				
				if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p5 && idx_h1 < rng_p4)
				sm_b[idx_h1][idx_h3 + (idx_h2 + (idx_p6) * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3] = tmp_dev_s1_v2_7[str_blk_idx_h3 + idx_h3 + (str_blk_idx_h2 + idx_h2 + (str_blk_idx_p5 + idx_p6 + (str_blk_idx_p4 + idx_h1) * base_size_p5b) * base_size_h2b) * base_size_h3b];
				__syncthreads();

				//  "p4" x "p5"
				reg_singles[0][0] -= sm_a[0][idx_p6 + (idx_h1) * FUSION_SIZE_SLICE_1_P6] * sm_b[0][idx_h3 + (idx_h2 + (0) * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3];
				reg_singles[0][1] -= sm_a[0][idx_p6 + (idx_h1) * FUSION_SIZE_SLICE_1_P6] * sm_b[0][idx_h3 + (idx_h2 + (1) * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3];
				reg_singles[0][2] -= sm_a[0][idx_p6 + (idx_h1) * FUSION_SIZE_SLICE_1_P6] * sm_b[0][idx_h3 + (idx_h2 + (2) * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3];
				reg_singles[0][3] -= sm_a[0][idx_p6 + (idx_h1) * FUSION_SIZE_SLICE_1_P6] * sm_b[0][idx_h3 + (idx_h2 + (3) * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3];

				reg_singles[1][0] -= sm_a[0][idx_p6 + (idx_h1) * FUSION_SIZE_SLICE_1_P6] * sm_b[1][idx_h3 + (idx_h2 + (0) * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3];
				reg_singles[1][1] -= sm_a[0][idx_p6 + (idx_h1) * FUSION_SIZE_SLICE_1_P6] * sm_b[1][idx_h3 + (idx_h2 + (1) * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3];
				reg_singles[1][2] -= sm_a[0][idx_p6 + (idx_h1) * FUSION_SIZE_SLICE_1_P6] * sm_b[1][idx_h3 + (idx_h2 + (2) * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3];
				reg_singles[1][3] -= sm_a[0][idx_p6 + (idx_h1) * FUSION_SIZE_SLICE_1_P6] * sm_b[1][idx_h3 + (idx_h2 + (3) * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3];

				reg_singles[2][0] -= sm_a[0][idx_p6 + (idx_h1) * FUSION_SIZE_SLICE_1_P6] * sm_b[2][idx_h3 + (idx_h2 + (0) * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3];
				reg_singles[2][1] -= sm_a[0][idx_p6 + (idx_h1) * FUSION_SIZE_SLICE_1_P6] * sm_b[2][idx_h3 + (idx_h2 + (1) * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3];
				reg_singles[2][2] -= sm_a[0][idx_p6 + (idx_h1) * FUSION_SIZE_SLICE_1_P6] * sm_b[2][idx_h3 + (idx_h2 + (2) * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3];
				reg_singles[2][3] -= sm_a[0][idx_p6 + (idx_h1) * FUSION_SIZE_SLICE_1_P6] * sm_b[2][idx_h3 + (idx_h2 + (3) * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3];

				reg_singles[3][0] -= sm_a[0][idx_p6 + (idx_h1) * FUSION_SIZE_SLICE_1_P6] * sm_b[3][idx_h3 + (idx_h2 + (0) * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3];
				reg_singles[3][1] -= sm_a[0][idx_p6 + (idx_h1) * FUSION_SIZE_SLICE_1_P6] * sm_b[3][idx_h3 + (idx_h2 + (1) * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3];
				reg_singles[3][2] -= sm_a[0][idx_p6 + (idx_h1) * FUSION_SIZE_SLICE_1_P6] * sm_b[3][idx_h3 + (idx_h2 + (2) * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3];
				reg_singles[3][3] -= sm_a[0][idx_p6 + (idx_h1) * FUSION_SIZE_SLICE_1_P6] * sm_b[3][idx_h3 + (idx_h2 + (3) * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3];
				__syncthreads();
			}
			
			//
			//  >> s1_8:   t3[h3,h2,h1,p6,p5,p4] -= t2[p6,h2] * v2[h3,h1,p5,p4] (h3,h2,p6), (h1)
			//
			if (flag_s1_8 >= 0)	// these if-conditions make 100 ms..
			{
				// 
				double* tmp_dev_s1_t2_8 = dev_s1_t2_all + size_max_dim_s1_t2 * flag_s1_8;
				double* tmp_dev_s1_v2_8 = dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_8;

				if (idx_h3 < rng_p6 && idx_h2 < rng_h2 && idx_p6 == 0 && idx_h1 == 0)
				sm_a[0][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_1_P6] = tmp_dev_s1_t2_8[str_blk_idx_p6 + idx_h3 + (str_blk_idx_h2 + idx_h2) * base_size_p6b];
						
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
				double* tmp_dev_s1_t2_9 = dev_s1_t2_all + size_max_dim_s1_t2 * flag_s1_9;
				double* tmp_dev_s1_v2_9 = dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_9;

				if (idx_h3 < rng_p6 && idx_h2 < rng_h3 && idx_p6 == 0 && idx_h1 == 0)
				sm_a[0][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_1_P6] = tmp_dev_s1_t2_9[str_blk_idx_p6 + idx_h3 + (str_blk_idx_h3 + idx_h2) * base_size_p6b];

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
	}
	
	//
	//  energies
	// 
	double energy_1 = 0.0;
	double energy_2 = 0.0;

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
					energy_1 += (reg_tile[j][i] *  reg_tile[j][i]) 						/ inner_factor;
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

// 
void total_fused_ccsd_t(size_t base_size_h1b, size_t base_size_h2b, size_t base_size_h3b, 
						size_t base_size_p4b, size_t base_size_p5b, size_t base_size_p6b,
						// 
						double* host_d1_t2_all, double* host_d1_v2_all,
						double* host_d2_t2_all, double* host_d2_v2_all,
						double* host_s1_t2_all, double* host_s1_v2_all,
						// 
						size_t size_d1_t2_all, size_t size_d1_v2_all,
						size_t size_d2_t2_all, size_t size_d2_v2_all,
						size_t size_s1_t2_all, size_t size_s1_v2_all,
						// 
						size_t* list_d1_sizes, 
						size_t* list_d2_sizes, 
						size_t* list_s1_sizes, 
						// 
						std::vector<int> vec_d1_flags,
						std::vector<int> vec_d2_flags,
						std::vector<int> vec_s1_flags,
						// 
						size_t size_noab, size_t size_max_dim_d1_t2, size_t size_max_dim_d1_v2,
						size_t size_nvab, size_t size_max_dim_d2_t2, size_t size_max_dim_d2_v2,
                                          size_t size_max_dim_s1_t2, size_t size_max_dim_s1_v2, 
						// 
						double factor, 
						double* host_evl_sorted_h1b, double* host_evl_sorted_h2b, double* host_evl_sorted_h3b, 
						double* host_evl_sorted_p4b, double* host_evl_sorted_p5b, double* host_evl_sorted_p6b,
						double* final_energy_4, double* final_energy_5)
{
	// 
	double* dev_d1_t2_all; double* dev_d1_v2_all;
	double* dev_d2_t2_all; double* dev_d2_v2_all;
	double* dev_s1_t2_all; double* dev_s1_v2_all;
	dev_d1_t2_all = (double*)getGpuMem(size_d1_t2_all * sizeof(double));
	dev_d1_v2_all = (double*)getGpuMem(size_d1_v2_all * sizeof(double));
	dev_d2_t2_all = (double*)getGpuMem(size_d2_t2_all * sizeof(double));
	dev_d2_v2_all = (double*)getGpuMem(size_d2_v2_all * sizeof(double));
	dev_s1_t2_all = (double*)getGpuMem(size_s1_t2_all * sizeof(double));
	dev_s1_v2_all = (double*)getGpuMem(size_s1_v2_all * sizeof(double));
	// cudaMalloc((void **)&dev_d1_t2_all, size_d1_t2_all * sizeof(double));
	// cudaMalloc((void **)&dev_d1_v2_all, size_d1_v2_all * sizeof(double));
	// cudaMalloc((void **)&dev_d2_t2_all, size_d2_t2_all * sizeof(double));
	// cudaMalloc((void **)&dev_d2_v2_all, size_d2_v2_all * sizeof(double));
	// cudaMalloc((void **)&dev_s1_t2_all, size_s1_t2_all * sizeof(double));
	// cudaMalloc((void **)&dev_s1_v2_all, size_s1_v2_all * sizeof(double));

	//
	double* dev_evl_sorted_h1b; double* dev_evl_sorted_h2b; double* dev_evl_sorted_h3b;
	double* dev_evl_sorted_p4b; double* dev_evl_sorted_p5b; double* dev_evl_sorted_p6b;
	// cudaMalloc((void **)&dev_evl_sorted_h1b, base_size_h1b * sizeof(double));
	// cudaMalloc((void **)&dev_evl_sorted_h2b, base_size_h2b * sizeof(double));
	// cudaMalloc((void **)&dev_evl_sorted_h3b, base_size_h3b * sizeof(double));
	// cudaMalloc((void **)&dev_evl_sorted_p4b, base_size_p4b * sizeof(double));
	// cudaMalloc((void **)&dev_evl_sorted_p5b, base_size_p5b * sizeof(double));
	// cudaMalloc((void **)&dev_evl_sorted_p6b, base_size_p6b * sizeof(double));
	dev_evl_sorted_h1b = (double*)getGpuMem(base_size_h1b * sizeof(double));
	dev_evl_sorted_h2b = (double*)getGpuMem(base_size_h2b * sizeof(double));
	dev_evl_sorted_h3b = (double*)getGpuMem(base_size_h3b * sizeof(double));
	dev_evl_sorted_p4b = (double*)getGpuMem(base_size_p4b * sizeof(double));
	dev_evl_sorted_p5b = (double*)getGpuMem(base_size_p5b * sizeof(double));
	dev_evl_sorted_p6b = (double*)getGpuMem(base_size_p6b * sizeof(double));


	// 
	int* int_list_s1_flags_offsets = (int*)malloc(sizeof(int) * (NUM_IA6_LOOPS * NUM_S1_EQUATIONS));
	int* int_list_d1_flags_offsets = (int*)malloc(sizeof(int) * (NUM_IA6_LOOPS * NUM_D1_EQUATIONS * size_noab));
	int* int_list_d2_flags_offsets = (int*)malloc(sizeof(int) * (NUM_IA6_LOOPS * NUM_D1_EQUATIONS * size_nvab));

	int offset = 0;
	for (int i = 0; i < NUM_IA6_LOOPS * NUM_S1_EQUATIONS; i++)
	{		
		if (vec_s1_flags[i] > 0)
		{
			int_list_s1_flags_offsets[i] = offset++;
		}
		else
		{
			int_list_s1_flags_offsets[i] = -1;
		}
	}

	offset = 0;
	for (int i = 0; i < NUM_IA6_LOOPS * size_noab * NUM_D1_EQUATIONS; i++)
	{
		if (vec_d1_flags[i] > 0)
		{
			int_list_d1_flags_offsets[i] = offset++;
		}
		else
		{
			int_list_d1_flags_offsets[i] = -1;
		}
	}

	offset = 0;
	for (int i = 0; i < NUM_IA6_LOOPS * size_nvab * NUM_D2_EQUATIONS; i++)
	{
		if (vec_d2_flags[i] > 0)
		{
			int_list_d2_flags_offsets[i] = offset++;
		}
		else
		{
			int_list_d2_flags_offsets[i] = -1;
		}
	}
	
	// 
	cudaMemcpyToSymbol(const_list_s1_flags_offset, int_list_s1_flags_offsets, sizeof(int) * (NUM_IA6_LOOPS * NUM_S1_EQUATIONS));
	cudaMemcpyToSymbol(const_list_d1_flags_offset, int_list_d1_flags_offsets, sizeof(int) * (NUM_IA6_LOOPS * MAX_NOAB * NUM_D1_EQUATIONS));
	cudaMemcpyToSymbol(const_list_d2_flags_offset, int_list_d2_flags_offsets, sizeof(int) * (NUM_IA6_LOOPS * MAX_NVAB * NUM_D2_EQUATIONS));

	int tmp_list_s1_sizes[NUM_IA6_LOOPS * NUM_S1_INDEX];
	for (int i = 0; i < NUM_IA6_LOOPS; i++)
	{
		tmp_list_s1_sizes[0 + (i) * NUM_S1_INDEX] = (int)list_s1_sizes[0 + (i) * NUM_S1_INDEX];
		tmp_list_s1_sizes[1 + (i) * NUM_S1_INDEX] = (int)list_s1_sizes[1 + (i) * NUM_S1_INDEX];
		tmp_list_s1_sizes[2 + (i) * NUM_S1_INDEX] = (int)list_s1_sizes[2 + (i) * NUM_S1_INDEX];
		tmp_list_s1_sizes[3 + (i) * NUM_S1_INDEX] = (int)list_s1_sizes[3 + (i) * NUM_S1_INDEX];
		tmp_list_s1_sizes[4 + (i) * NUM_S1_INDEX] = (int)list_s1_sizes[4 + (i) * NUM_S1_INDEX];
		tmp_list_s1_sizes[5 + (i) * NUM_S1_INDEX] = (int)list_s1_sizes[5 + (i) * NUM_S1_INDEX];
	}

	int tmp_list_d1_sizes[NUM_IA6_LOOPS * size_noab * NUM_D1_INDEX];
	for (int i = 0; i < NUM_IA6_LOOPS; i++)
	{
		for (int j = 0; j < size_noab; j++)
		{
			tmp_list_d1_sizes[0 + (j + (i) * size_noab) * NUM_D1_INDEX] = (int)list_d1_sizes[0 + (j + (i) * size_noab) * NUM_D1_INDEX];
			tmp_list_d1_sizes[1 + (j + (i) * size_noab) * NUM_D1_INDEX] = (int)list_d1_sizes[1 + (j + (i) * size_noab) * NUM_D1_INDEX];
			tmp_list_d1_sizes[2 + (j + (i) * size_noab) * NUM_D1_INDEX] = (int)list_d1_sizes[2 + (j + (i) * size_noab) * NUM_D1_INDEX];
			tmp_list_d1_sizes[3 + (j + (i) * size_noab) * NUM_D1_INDEX] = (int)list_d1_sizes[3 + (j + (i) * size_noab) * NUM_D1_INDEX];
			tmp_list_d1_sizes[4 + (j + (i) * size_noab) * NUM_D1_INDEX] = (int)list_d1_sizes[4 + (j + (i) * size_noab) * NUM_D1_INDEX];
			tmp_list_d1_sizes[5 + (j + (i) * size_noab) * NUM_D1_INDEX] = (int)list_d1_sizes[5 + (j + (i) * size_noab) * NUM_D1_INDEX];
			tmp_list_d1_sizes[6 + (j + (i) * size_noab) * NUM_D1_INDEX] = (int)list_d1_sizes[6 + (j + (i) * size_noab) * NUM_D1_INDEX];
		}
	}

	int tmp_list_d2_sizes[NUM_IA6_LOOPS * size_nvab * NUM_D2_INDEX];
	for (int i = 0; i < NUM_IA6_LOOPS; i++)
	{
		for (int j = 0; j < size_nvab; j++)
		{
			tmp_list_d2_sizes[0 + (j + (i) * size_nvab) * NUM_D2_INDEX] = (int)list_d2_sizes[0 + (j + (i) * size_nvab) * NUM_D2_INDEX];
			tmp_list_d2_sizes[1 + (j + (i) * size_nvab) * NUM_D2_INDEX] = (int)list_d2_sizes[1 + (j + (i) * size_nvab) * NUM_D2_INDEX];
			tmp_list_d2_sizes[2 + (j + (i) * size_nvab) * NUM_D2_INDEX] = (int)list_d2_sizes[2 + (j + (i) * size_nvab) * NUM_D2_INDEX];
			tmp_list_d2_sizes[3 + (j + (i) * size_nvab) * NUM_D2_INDEX] = (int)list_d2_sizes[3 + (j + (i) * size_nvab) * NUM_D2_INDEX];
			tmp_list_d2_sizes[4 + (j + (i) * size_nvab) * NUM_D2_INDEX] = (int)list_d2_sizes[4 + (j + (i) * size_nvab) * NUM_D2_INDEX];
			tmp_list_d2_sizes[5 + (j + (i) * size_nvab) * NUM_D2_INDEX] = (int)list_d2_sizes[5 + (j + (i) * size_nvab) * NUM_D2_INDEX];
			tmp_list_d2_sizes[6 + (j + (i) * size_nvab) * NUM_D2_INDEX] = (int)list_d2_sizes[6 + (j + (i) * size_nvab) * NUM_D2_INDEX];
		}
	}

	//
	cudaMemcpyToSymbol(const_list_s1_problem_size, tmp_list_s1_sizes, sizeof(int) * (NUM_IA6_LOOPS * NUM_S1_INDEX));
	cudaMemcpyToSymbol(const_list_d1_problem_size, tmp_list_d1_sizes, sizeof(int) * (NUM_IA6_LOOPS * NUM_D1_INDEX * size_noab));
	cudaMemcpyToSymbol(const_list_d2_problem_size, tmp_list_d2_sizes, sizeof(int) * (NUM_IA6_LOOPS * NUM_D2_INDEX * size_nvab));

	// 
	cudaMemcpy(dev_d1_t2_all, host_d1_t2_all, (size_d1_t2_all) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_d1_v2_all, host_d1_v2_all, (size_d1_v2_all) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_d2_t2_all, host_d2_t2_all, (size_d2_t2_all) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_d2_v2_all, host_d2_v2_all, (size_d2_v2_all) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_s1_t2_all, host_s1_t2_all, (size_s1_t2_all) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_s1_v2_all, host_s1_v2_all, (size_s1_v2_all) * sizeof(double), cudaMemcpyHostToDevice);

	// 
	cudaMemcpy(dev_evl_sorted_h1b, host_evl_sorted_h1b, base_size_h1b * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_evl_sorted_h2b, host_evl_sorted_h2b, base_size_h2b * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_evl_sorted_h3b, host_evl_sorted_h3b, base_size_h3b * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_evl_sorted_p4b, host_evl_sorted_p4b, base_size_p4b * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_evl_sorted_p5b, host_evl_sorted_p5b, base_size_p5b * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_evl_sorted_p6b, host_evl_sorted_p6b, base_size_p6b * sizeof(double), cudaMemcpyHostToDevice);
	
	//
	//  the kernel should be based on the based problem sizes.
	// 
	size_t num_blocks_kernel_1 = CEIL(base_size_h3b, FUSION_SIZE_SLICE_1_H3) * CEIL(base_size_h2b, FUSION_SIZE_SLICE_1_H2) * 
								 CEIL(base_size_h1b, FUSION_SIZE_SLICE_1_H1) * CEIL(base_size_p6b, FUSION_SIZE_SLICE_1_P6) * 
								 CEIL(base_size_p5b, FUSION_SIZE_SLICE_1_P5) * CEIL(base_size_p4b, FUSION_SIZE_SLICE_1_P4);
	
	// Depends on # of Fused Kernel
	dim3 gridsize_1(num_blocks_kernel_1);
	dim3 blocksize_1(FUSION_SIZE_TB_1_X, FUSION_SIZE_TB_1_Y);

	// 
	double* host_energies = (double*)malloc(num_blocks_kernel_1 * NUM_ENERGIES * sizeof(double));
	memset(host_energies, 0.0, num_blocks_kernel_1 * NUM_ENERGIES * sizeof(double));
	double* dev_energies;
	// cudaMalloc((void **)&dev_energies, num_blocks_kernel_1 * NUM_ENERGIES * sizeof(double));
	dev_energies = (double*)getGpuMem(num_blocks_kernel_1 * NUM_ENERGIES * sizeof(double));

#ifdef DEBUG_TIME_FUSED_CCSD_T
	cudaEvent_t start_ccsd_t, stop_ccsd_t, stop_kernely_only;
	cudaEventCreate(&start_ccsd_t);
	cudaEventCreate(&stop_ccsd_t);
	cudaEventCreate(&stop_kernely_only);
	cudaEventRecord(start_ccsd_t);
#endif

	// 
	jk_ccsd_t_fully_fused_kernel<<<gridsize_1, blocksize_1>>>((int)size_noab, (int)size_nvab, 
																// 
																(int)size_max_dim_s1_t2, (int)size_max_dim_s1_v2,
																(int)size_max_dim_d1_t2, (int)size_max_dim_d1_v2,
																(int)size_max_dim_d2_t2, (int)size_max_dim_d2_v2,
																// 
																dev_d1_t2_all, dev_d1_v2_all, 
																dev_d2_t2_all, dev_d2_v2_all, 
																dev_s1_t2_all, dev_s1_v2_all, 
																//  
																dev_evl_sorted_h1b, dev_evl_sorted_h2b, dev_evl_sorted_h3b,
																dev_evl_sorted_p4b, dev_evl_sorted_p5b, dev_evl_sorted_p6b,
																dev_energies, 
																// 
																CEIL(base_size_h3b, FUSION_SIZE_SLICE_1_H3), CEIL(base_size_h2b, FUSION_SIZE_SLICE_1_H2), CEIL(base_size_h1b, FUSION_SIZE_SLICE_1_H1), 
																CEIL(base_size_p6b, FUSION_SIZE_SLICE_1_P6), CEIL(base_size_p5b, FUSION_SIZE_SLICE_1_P5), CEIL(base_size_p4b, FUSION_SIZE_SLICE_1_P4),
																// 
																(int)base_size_h1b, (int)base_size_h2b, (int)base_size_h1b, 
																(int)base_size_p4b, (int)base_size_p5b, (int)base_size_p6b);

	// 
	// printf ("%s\n", cudaGetErrorString(cudaGetLastError()));

	//
#ifdef DEBUG_TIME_FUSED_CCSD_T
	cudaEventRecord(stop_kernely_only);
	cudaEventSynchronize(stop_kernely_only);
	float time_ms_ccsd_t_kernel_only = 0.0;
	cudaEventElapsedTime(&time_ms_ccsd_t_kernel_only, start_ccsd_t, stop_kernely_only);
#endif
	
	//
	cudaMemcpy(host_energies, dev_energies, num_blocks_kernel_1 * NUM_ENERGIES * sizeof(double), cudaMemcpyDeviceToHost);

	// 
	double final_energy_1 = 0.0;
	double final_energy_2 = 0.0;
	for (size_t i = 0; i < num_blocks_kernel_1; i++)
	{
		// 
		final_energy_1 += host_energies[i];
		final_energy_2 += host_energies[i + num_blocks_kernel_1];
	}

	// 
	final_energy_1 *= factor;
	final_energy_2 *= factor;
	*final_energy_4 = final_energy_1 * factor;
	*final_energy_5 = final_energy_2 * factor;

#ifdef DEBUG_TIME_FUSED_CCSD_T
	cudaEventRecord(stop_ccsd_t);
	cudaEventSynchronize(stop_ccsd_t);
	float time_ms_ccsd_t_kernel = 0.0;
	cudaEventElapsedTime(&time_ms_ccsd_t_kernel, start_ccsd_t, stop_ccsd_t);
	printf ("========================================================================================\n");
	printf ("[%s][fused] kernel-only-time: %f (ms)\n", __func__, time_ms_ccsd_t_kernel_only);
	printf ("[%s][fused] total-time: %f (ms)\n", __func__, time_ms_ccsd_t_kernel);
	printf ("[%s][fused] E(4): %.15f, E(5): %.15f\n", __func__,  final_energy_1,  final_energy_2);
	printf ("[%s][fused] E(4): %.15f, E(5): %.15f\n", __func__, *final_energy_4, *final_energy_5);
	printf ("========================================================================================\n");
#endif

	//
	// cudaFree(dev_s1_t2_all);  cudaFree(dev_s1_v2_all); 
	// cudaFree(dev_d1_t2_all);  cudaFree(dev_d1_v2_all);
	// cudaFree(dev_d2_t2_all);  cudaFree(dev_d2_v2_all);
	freeGpuMem(dev_s1_t2_all);	freeGpuMem(dev_s1_v2_all);
	freeGpuMem(dev_d1_t2_all);	freeGpuMem(dev_d1_v2_all);
	freeGpuMem(dev_d2_t2_all);	freeGpuMem(dev_d2_v2_all);

	// cudaFree(dev_evl_sorted_h1b); cudaFree(dev_evl_sorted_h2b); cudaFree(dev_evl_sorted_h3b);
	// cudaFree(dev_evl_sorted_p4b); cudaFree(dev_evl_sorted_p5b); cudaFree(dev_evl_sorted_p6b);
	freeGpuMem(dev_evl_sorted_h1b); freeGpuMem(dev_evl_sorted_h2b); freeGpuMem(dev_evl_sorted_h3b);
	freeGpuMem(dev_evl_sorted_p4b); freeGpuMem(dev_evl_sorted_p5b); freeGpuMem(dev_evl_sorted_p6b);

	// cudaFree(dev_energies);	
	freeGpuMem(dev_energies);
	free(host_energies);
}

__device__ double* ma_t3_s;
__device__ double* ma_t3_d;

void ma_dev_mem_d(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d,size_t p6d)
{
    // size_t size_t3;
    size_t size_t3 = h1d*h2d*h3d*p4d*p5d*p6d;
    ma_t3_d = (double *) getGpuMem(size_t3*sizeof(double));
    cudaMemset(ma_t3_d,0,size_t3*sizeof(double));
}

void ma_dev_mem_s(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d,size_t p6d)
{
    // size_t size_t3;
    size_t size_t3 = h1d*h2d*h3d*p4d*p5d*p6d;
    ma_t3_s = (double *) getGpuMem(size_t3*sizeof(double));
    cudaMemset(ma_t3_s,0,size_t3*sizeof(double));
}

void ma_dev_release()
{
	freeGpuMem(ma_t3_d);
	freeGpuMem(ma_t3_s);
}

/*----------------------------------------------------------------------*
 *triplesx[h3,h1,p6,p5,p4] -= t2sub[h7,p4,p5,h1] * v2sub[h3,p6,h7]
 *----------------------------------------------------------------------*/
 #define T1 16
 #define T2 16
 #define Tcomm 16
__global__ void ma_sd_t_d1_1_kernel(size_t h1d,size_t h3d,size_t h7d,size_t p4d,size_t p5d,size_t p6d,
				size_t h7ld_t2sub,size_t p4ld_t2sub,size_t p5ld_t2sub,size_t h1ld_t2sub,size_t h3ld_v2sub,size_t p6ld_v2sub,size_t h7ld_v2sub,
				size_t h3ld_triplesx,size_t h1ld_triplesx,size_t p6ld_triplesx,size_t p5ld_triplesx,size_t p4ld_triplesx,
				double *triplesx_d, double *t2sub_d, double *v2sub_d,size_t unused_idx, size_t total_x, size_t total_y) 
{
	size_t h1_0,h1_1,h1_2,h1_3,h3_0,h3_1,h3_2,h3_3,h7,p4_0,p4_1,p4_2,p4_3,p5_0,p5_1,p5_2,p5_3,p6_0,p6_1,p6_2,p6_3;
	double a1,b1;
	double a2,b2;
	double a3,b3;
	double a4,b4;
	size_t in1_idxl,in2_idxl,h7l,h7T;
	__shared__ double t2sub_shm[4*T1][Tcomm];
	__shared__ double v2sub_shm[Tcomm][4*T2];
	size_t rest_x=blockIdx.x;
	size_t rest_y=blockIdx.y;
	size_t thread_x = T2*4 * rest_x + threadIdx.x;
	size_t thread_y = T1*4 * rest_y + threadIdx.y;
	in1_idxl=threadIdx.y;
	in2_idxl=threadIdx.x ;
	double tlocal1=0;
	double tlocal2=0;
	double tlocal3=0;
	double tlocal4=0;
	double tlocal5=0;
	double tlocal6=0;
	double tlocal7=0;
	double tlocal8=0;
	double tlocal9=0;
	double tlocal10=0;
	double tlocal11=0;
	double tlocal12=0;
	double tlocal13=0;
	double tlocal14=0;
	double tlocal15=0;
	double tlocal16=0;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
	h3_0=rest_x%h3d;
	rest_x=rest_x/h3d;
	h1_0=rest_y%h1d;
	rest_y=rest_y/h1d;
	p5_0=rest_y%p5d;
	rest_y=rest_y/p5d;
	p4_0=rest_y;
	p6_0=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
	h3_1=rest_x%h3d;
	rest_x=rest_x/h3d;
	h1_1=rest_y%h1d;
	rest_y=rest_y/h1d;
	p5_1=rest_y%p5d;
	rest_y=rest_y/p5d;
	p4_1=rest_y;
	p6_1=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
	h3_2=rest_x%h3d;
	rest_x=rest_x/h3d;
	h1_2=rest_y%h1d;
	rest_y=rest_y/h1d;
	p5_2=rest_y%p5d;
	rest_y=rest_y/p5d;
	p4_2=rest_y;
	p6_2=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
	h3_3=rest_x%h3d;
	rest_x=rest_x/h3d;
	h1_3=rest_y%h1d;
	rest_y=rest_y/h1d;
	p5_3=rest_y%p5d;
	rest_y=rest_y/p5d;
	p4_3=rest_y;
	p6_3=rest_x;
	size_t t2sub_d_off, v2sub_d_off;for(h7T=0;h7T<h7d;h7T+=Tcomm){size_t h7l_hi;
		h7l_hi = MIN(Tcomm+h7T,h7d)-h7T;
		t2sub_d_off=p4_0*p4ld_t2sub+p5_0*p5ld_t2sub+h1_0*h1ld_t2sub;
		v2sub_d_off=h3_0*h3ld_v2sub+p6_0*p6ld_v2sub;
		if(thread_y+T1*0<total_y)for(h7l=threadIdx.x;h7l<h7l_hi;h7l+=blockDim.x){
		h7=h7l+h7T;
		t2sub_shm[in1_idxl+T1*0][h7l] = t2sub_d[t2sub_d_off+h7*h7ld_t2sub];
		}
		if(thread_x+T1*0<total_x)for(h7l=threadIdx.y;h7l<h7l_hi;h7l+=blockDim.y){
		h7=h7l+h7T;
		v2sub_shm[h7l][in2_idxl+T1*0] = v2sub_d[v2sub_d_off+h7*h7ld_v2sub];
		}
		t2sub_d_off=p4_1*p4ld_t2sub+p5_1*p5ld_t2sub+h1_1*h1ld_t2sub;
		v2sub_d_off=h3_1*h3ld_v2sub+p6_1*p6ld_v2sub;
		if(thread_y+T1*1<total_y)for(h7l=threadIdx.x;h7l<h7l_hi;h7l+=blockDim.x){
		h7=h7l+h7T;
		t2sub_shm[in1_idxl+T1*1][h7l] = t2sub_d[t2sub_d_off+h7*h7ld_t2sub];
		}
		if(thread_x+T1*1<total_x)for(h7l=threadIdx.y;h7l<h7l_hi;h7l+=blockDim.y){
		h7=h7l+h7T;
		v2sub_shm[h7l][in2_idxl+T1*1] = v2sub_d[v2sub_d_off+h7*h7ld_v2sub];
		}
		t2sub_d_off=p4_2*p4ld_t2sub+p5_2*p5ld_t2sub+h1_2*h1ld_t2sub;
		v2sub_d_off=h3_2*h3ld_v2sub+p6_2*p6ld_v2sub;
		if(thread_y+T1*2<total_y)for(h7l=threadIdx.x;h7l<h7l_hi;h7l+=blockDim.x){
		h7=h7l+h7T;
		t2sub_shm[in1_idxl+T1*2][h7l] = t2sub_d[t2sub_d_off+h7*h7ld_t2sub];
		}
		if(thread_x+T1*2<total_x)for(h7l=threadIdx.y;h7l<h7l_hi;h7l+=blockDim.y){
		h7=h7l+h7T;
		v2sub_shm[h7l][in2_idxl+T1*2] = v2sub_d[v2sub_d_off+h7*h7ld_v2sub];
		}
		t2sub_d_off=p4_3*p4ld_t2sub+p5_3*p5ld_t2sub+h1_3*h1ld_t2sub;
		v2sub_d_off=h3_3*h3ld_v2sub+p6_3*p6ld_v2sub;
		if(thread_y+T1*3<total_y)for(h7l=threadIdx.x;h7l<h7l_hi;h7l+=blockDim.x){
		h7=h7l+h7T;
		t2sub_shm[in1_idxl+T1*3][h7l] = t2sub_d[t2sub_d_off+h7*h7ld_t2sub];
		}
		if(thread_x+T1*3<total_x)for(h7l=threadIdx.y;h7l<h7l_hi;h7l+=blockDim.y){
		h7=h7l+h7T;
		v2sub_shm[h7l][in2_idxl+T1*3] = v2sub_d[v2sub_d_off+h7*h7ld_v2sub];
		}
		__syncthreads();
		for(h7l=0;h7l<h7l_hi;++h7l){
		a1=t2sub_shm[in1_idxl+T1*0][h7l];
		a2=t2sub_shm[in1_idxl+T1*1][h7l];
		a3=t2sub_shm[in1_idxl+T1*2][h7l];
		a4=t2sub_shm[in1_idxl+T1*3][h7l];
		b1=v2sub_shm[h7l][in2_idxl+T2*0];
		b2=v2sub_shm[h7l][in2_idxl+T2*1];
		b3=v2sub_shm[h7l][in2_idxl+T2*2];
		b4=v2sub_shm[h7l][in2_idxl+T2*3];
		tlocal1+=a1*b1;
		tlocal2+=a2*b1;
		tlocal3+=a3*b1;
		tlocal4+=a4*b1;
		tlocal5+=a1*b2;
		tlocal6+=a2*b2;
		tlocal7+=a3*b2;
		tlocal8+=a4*b2;
		tlocal9+=a1*b3;
		tlocal10+=a2*b3;
		tlocal11+=a3*b3;
		tlocal12+=a4*b3;
		tlocal13+=a1*b4;
		tlocal14+=a2*b4;
		tlocal15+=a3*b4;
		tlocal16+=a4*b4;
		}
		__syncthreads();
	}
	if(thread_x+T1*0<total_x){
		if(thread_y+T2*3<total_y) {
		triplesx_d[h3_0*h3ld_triplesx+h1_0*h1ld_triplesx+p6_0*p6ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]-=tlocal1;
		triplesx_d[h3_0*h3ld_triplesx+h1_1*h1ld_triplesx+p6_0*p6ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx]-=tlocal2;
		triplesx_d[h3_0*h3ld_triplesx+h1_2*h1ld_triplesx+p6_0*p6ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx]-=tlocal3;
		triplesx_d[h3_0*h3ld_triplesx+h1_3*h1ld_triplesx+p6_0*p6ld_triplesx+p5_3*p5ld_triplesx+p4_3*p4ld_triplesx]-=tlocal4;
		}
		else if(thread_y+T2*2<total_y) {
		triplesx_d[h3_0*h3ld_triplesx+h1_0*h1ld_triplesx+p6_0*p6ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]-=tlocal1;
		triplesx_d[h3_0*h3ld_triplesx+h1_1*h1ld_triplesx+p6_0*p6ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx]-=tlocal2;
		triplesx_d[h3_0*h3ld_triplesx+h1_2*h1ld_triplesx+p6_0*p6ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx]-=tlocal3;
		}
		else if(thread_y+T2*1<total_y) {
		triplesx_d[h3_0*h3ld_triplesx+h1_0*h1ld_triplesx+p6_0*p6ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]-=tlocal1;
		triplesx_d[h3_0*h3ld_triplesx+h1_1*h1ld_triplesx+p6_0*p6ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx]-=tlocal2;
		}
		else if(thread_y+T2*0<total_y) {
		triplesx_d[h3_0*h3ld_triplesx+h1_0*h1ld_triplesx+p6_0*p6ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]-=tlocal1;
		}
	}
	if(thread_x+T1*1<total_x){
		if(thread_y+T2*3<total_y) {
		triplesx_d[h3_1*h3ld_triplesx+h1_0*h1ld_triplesx+p6_1*p6ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]-=tlocal5;
		triplesx_d[h3_1*h3ld_triplesx+h1_1*h1ld_triplesx+p6_1*p6ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx]-=tlocal6;
		triplesx_d[h3_1*h3ld_triplesx+h1_2*h1ld_triplesx+p6_1*p6ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx]-=tlocal7;
		triplesx_d[h3_1*h3ld_triplesx+h1_3*h1ld_triplesx+p6_1*p6ld_triplesx+p5_3*p5ld_triplesx+p4_3*p4ld_triplesx]-=tlocal8;
		}
		else if(thread_y+T2*2<total_y) {
		triplesx_d[h3_1*h3ld_triplesx+h1_0*h1ld_triplesx+p6_1*p6ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]-=tlocal5;
		triplesx_d[h3_1*h3ld_triplesx+h1_1*h1ld_triplesx+p6_1*p6ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx]-=tlocal6;
		triplesx_d[h3_1*h3ld_triplesx+h1_2*h1ld_triplesx+p6_1*p6ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx]-=tlocal7;
		}
		else if(thread_y+T2*1<total_y) {
		triplesx_d[h3_1*h3ld_triplesx+h1_0*h1ld_triplesx+p6_1*p6ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]-=tlocal5;
		triplesx_d[h3_1*h3ld_triplesx+h1_1*h1ld_triplesx+p6_1*p6ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx]-=tlocal6;
		}
		else if(thread_y+T2*0<total_y) {
		triplesx_d[h3_1*h3ld_triplesx+h1_0*h1ld_triplesx+p6_1*p6ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]-=tlocal5;
		}
	}
	if(thread_x+T1*2<total_x){
		if(thread_y+T2*3<total_y) {
		triplesx_d[h3_2*h3ld_triplesx+h1_0*h1ld_triplesx+p6_2*p6ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]-=tlocal9;
		triplesx_d[h3_2*h3ld_triplesx+h1_1*h1ld_triplesx+p6_2*p6ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx]-=tlocal10;
		triplesx_d[h3_2*h3ld_triplesx+h1_2*h1ld_triplesx+p6_2*p6ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx]-=tlocal11;
		triplesx_d[h3_2*h3ld_triplesx+h1_3*h1ld_triplesx+p6_2*p6ld_triplesx+p5_3*p5ld_triplesx+p4_3*p4ld_triplesx]-=tlocal12;
		}
		else if(thread_y+T2*2<total_y) {
		triplesx_d[h3_2*h3ld_triplesx+h1_0*h1ld_triplesx+p6_2*p6ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]-=tlocal9;
		triplesx_d[h3_2*h3ld_triplesx+h1_1*h1ld_triplesx+p6_2*p6ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx]-=tlocal10;
		triplesx_d[h3_2*h3ld_triplesx+h1_2*h1ld_triplesx+p6_2*p6ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx]-=tlocal11;
		}
		else if(thread_y+T2*1<total_y) {
		triplesx_d[h3_2*h3ld_triplesx+h1_0*h1ld_triplesx+p6_2*p6ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]-=tlocal9;
		triplesx_d[h3_2*h3ld_triplesx+h1_1*h1ld_triplesx+p6_2*p6ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx]-=tlocal10;
		}
		else if(thread_y+T2*0<total_y) {
		triplesx_d[h3_2*h3ld_triplesx+h1_0*h1ld_triplesx+p6_2*p6ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]-=tlocal9;
		}
	}
	if(thread_x+T1*3<total_x){
		if(thread_y+T2*3<total_y) {
		triplesx_d[h3_3*h3ld_triplesx+h1_0*h1ld_triplesx+p6_3*p6ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]-=tlocal13;
		triplesx_d[h3_3*h3ld_triplesx+h1_1*h1ld_triplesx+p6_3*p6ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx]-=tlocal14;
		triplesx_d[h3_3*h3ld_triplesx+h1_2*h1ld_triplesx+p6_3*p6ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx]-=tlocal15;
		triplesx_d[h3_3*h3ld_triplesx+h1_3*h1ld_triplesx+p6_3*p6ld_triplesx+p5_3*p5ld_triplesx+p4_3*p4ld_triplesx]-=tlocal16;
		}
		else if(thread_y+T2*2<total_y) {
		triplesx_d[h3_3*h3ld_triplesx+h1_0*h1ld_triplesx+p6_3*p6ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]-=tlocal13;
		triplesx_d[h3_3*h3ld_triplesx+h1_1*h1ld_triplesx+p6_3*p6ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx]-=tlocal14;
		triplesx_d[h3_3*h3ld_triplesx+h1_2*h1ld_triplesx+p6_3*p6ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx]-=tlocal15;
		}
		else if(thread_y+T2*1<total_y) {
		triplesx_d[h3_3*h3ld_triplesx+h1_0*h1ld_triplesx+p6_3*p6ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]-=tlocal13;
		triplesx_d[h3_3*h3ld_triplesx+h1_1*h1ld_triplesx+p6_3*p6ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx]-=tlocal14;
		}
		else if(thread_y+T2*0<total_y) {
		triplesx_d[h3_3*h3ld_triplesx+h1_0*h1ld_triplesx+p6_3*p6ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]-=tlocal13;
		}
	}
	__syncthreads();
}

void ma_sd_t_d1_1_cuda(size_t h1d, size_t h2d, size_t h3d, size_t h7d, size_t p4d, size_t p5d, size_t p6d, double *triplesx, double *t2sub, double *v2sub) 
{
	h3d=h3d*h2d;
	size_t h7ld_t2sub,p4ld_t2sub,p5ld_t2sub,h1ld_t2sub,h3ld_v2sub,p6ld_v2sub,h7ld_v2sub,h3ld_triplesx,h1ld_triplesx,p6ld_triplesx,p5ld_triplesx,p4ld_triplesx;
	size_t size_triplesx,size_block_triplesx,size_el_block_triplesx,size_t2sub,size_v2sub;
	cudaStream_t *streams;
	size_t nstreams,i;
	double *t2sub_d,*v2sub_d;
	size_triplesx=h3d*h1d*p6d*p5d*p4d*sizeof(double);
	size_t2sub=h7d*p4d*p5d*h1d*sizeof(double);
	size_v2sub=h3d*p6d*h7d*sizeof(double);
	cudaFuncSetCacheConfig(ma_sd_t_d1_1_kernel, cudaFuncCachePreferShared);
	nstreams=1;
	size_block_triplesx=size_triplesx/nstreams;
	size_el_block_triplesx=size_block_triplesx/sizeof(double);
	t2sub_d=(double*)getGpuMem(size_t2sub);
	v2sub_d=(double*)getGpuMem(size_v2sub);
	streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
	assert(streams!= NULL);
	for(i=0;i<nstreams;++i) {
		CUDA_SAFE(cudaStreamCreate(&streams[i])) ;
	}
	CUDA_SAFE(cudaMemcpy(t2sub_d,t2sub,size_t2sub,cudaMemcpyHostToDevice));
	CUDA_SAFE(cudaMemcpy(v2sub_d,v2sub,size_v2sub,cudaMemcpyHostToDevice));
	h7ld_t2sub=1;
	p4ld_t2sub=h7d;
	p5ld_t2sub=p4d*h7d;
	h1ld_t2sub=p5d*p4d*h7d;
	h3ld_v2sub=1;
	p6ld_v2sub=h3d;
	h7ld_v2sub=p6d*h3d;
	h3ld_triplesx=1;
	h1ld_triplesx=h3d;
	p6ld_triplesx=h1d*h3d;
	p5ld_triplesx=p6d*h1d*h3d;
	p4ld_triplesx=p5d*p6d*h1d*h3d;
	size_t total_x = h3d*p6d*1;
	size_t total_y = p4d*p5d*h1d;
	dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
	for(i=0;i<nstreams;++i){
		ma_sd_t_d1_1_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(h1d,h3d,h7d,p4d,p5d,p6d,h7ld_t2sub,p4ld_t2sub,p5ld_t2sub,h1ld_t2sub,h3ld_v2sub,p6ld_v2sub,h7ld_v2sub,h3ld_triplesx,h1ld_triplesx,p6ld_triplesx,p5ld_triplesx,p4ld_triplesx,ma_t3_d,t2sub_d,v2sub_d,i,total_x,total_y);
		CHECK_ERR("Kernel execution failed");
	}
	cudaThreadSynchronize();
	for(i=0;i<nstreams;++i){
		cudaStreamDestroy(streams[i]);}
	freeGpuMem(t2sub_d);
	freeGpuMem(v2sub_d);
	free(streams);
}
#undef T1
#undef T2
#undef Tcomm
void ma_sd_t_d1_1_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* h7d, Integer* p4d, Integer* p5d, Integer* p6d, double *triplesx, double *t2sub, double *v2sub) 
{
	ma_sd_t_d1_1_cuda((int)*h1d,(int)*h2d,(int)*h3d,(int)*h7d,(int)*p4d,(int)*p5d,(int)*p6d,triplesx,t2sub,v2sub);
}
/*----------------------------------------------------------------------*
*triplesx[h3,h1,h2,p5,p4] += t2sub[h7,p4,p5,h1] * v2sub[h3,h2,h7]
*----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void ma_sd_t_d1_2_kernel(size_t h1d,size_t h2d,size_t h3d,size_t h7d,size_t p4d,size_t p5d,
	size_t h7ld_t2sub,size_t p4ld_t2sub,size_t p5ld_t2sub,size_t h1ld_t2sub,size_t h3ld_v2sub,size_t h2ld_v2sub,size_t h7ld_v2sub,
	size_t h3ld_triplesx,size_t h1ld_triplesx,size_t h2ld_triplesx,size_t p5ld_triplesx,size_t p4ld_triplesx,
	double *triplesx_d, double *t2sub_d, double *v2sub_d,size_t unused_idx, size_t total_x, size_t total_y) 
{
	size_t h1_0,h1_1,h1_2,h1_3,h2_0,h2_1,h2_2,h2_3,h3_0,h3_1,h3_2,h3_3,h7,p4_0,p4_1,p4_2,p4_3,p5_0,p5_1,p5_2,p5_3;
	double a1,b1;
	double a2,b2;
	double a3,b3;
	double a4,b4;
	size_t in1_idxl,in2_idxl,h7l,h7T;
	__shared__ double t2sub_shm[4*T1][Tcomm];
	__shared__ double v2sub_shm[Tcomm][4*T2];
	size_t rest_x=blockIdx.x;
	size_t rest_y=blockIdx.y;
	size_t thread_x = T2*4 * rest_x + threadIdx.x;
	size_t thread_y = T1*4 * rest_y + threadIdx.y;
	in1_idxl=threadIdx.y;
	in2_idxl=threadIdx.x ;
	double tlocal1=0;
	double tlocal2=0;
	double tlocal3=0;
	double tlocal4=0;
	double tlocal5=0;
	double tlocal6=0;
	double tlocal7=0;
	double tlocal8=0;
	double tlocal9=0;
	double tlocal10=0;
	double tlocal11=0;
	double tlocal12=0;
	double tlocal13=0;
	double tlocal14=0;
	double tlocal15=0;
	double tlocal16=0;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
	h3_0=rest_x%h3d;
	rest_x=rest_x/h3d;
	h1_0=rest_y%h1d;
	rest_y=rest_y/h1d;
	p5_0=rest_y%p5d;
	rest_y=rest_y/p5d;
	p4_0=rest_y;
	h2_0=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
	h3_1=rest_x%h3d;
	rest_x=rest_x/h3d;
	h1_1=rest_y%h1d;
	rest_y=rest_y/h1d;
	p5_1=rest_y%p5d;
	rest_y=rest_y/p5d;
	p4_1=rest_y;
	h2_1=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
	h3_2=rest_x%h3d;
	rest_x=rest_x/h3d;
	h1_2=rest_y%h1d;
	rest_y=rest_y/h1d;
	p5_2=rest_y%p5d;
	rest_y=rest_y/p5d;
	p4_2=rest_y;
	h2_2=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
	h3_3=rest_x%h3d;
	rest_x=rest_x/h3d;
	h1_3=rest_y%h1d;
	rest_y=rest_y/h1d;
	p5_3=rest_y%p5d;
	rest_y=rest_y/p5d;
	p4_3=rest_y;
	h2_3=rest_x;
	size_t t2sub_d_off, v2sub_d_off;for(h7T=0;h7T<h7d;h7T+=Tcomm){size_t h7l_hi;
		h7l_hi = MIN(Tcomm+h7T,h7d)-h7T;
		t2sub_d_off=p4_0*p4ld_t2sub+p5_0*p5ld_t2sub+h1_0*h1ld_t2sub;
		v2sub_d_off=h3_0*h3ld_v2sub+h2_0*h2ld_v2sub;
		if(thread_y+T1*0<total_y)for(h7l=threadIdx.x;h7l<h7l_hi;h7l+=blockDim.x){
		h7=h7l+h7T;
		t2sub_shm[in1_idxl+T1*0][h7l] = t2sub_d[t2sub_d_off+h7*h7ld_t2sub];
		}
		if(thread_x+T1*0<total_x)for(h7l=threadIdx.y;h7l<h7l_hi;h7l+=blockDim.y){
		h7=h7l+h7T;
		v2sub_shm[h7l][in2_idxl+T1*0] = v2sub_d[v2sub_d_off+h7*h7ld_v2sub];
		}
		t2sub_d_off=p4_1*p4ld_t2sub+p5_1*p5ld_t2sub+h1_1*h1ld_t2sub;
		v2sub_d_off=h3_1*h3ld_v2sub+h2_1*h2ld_v2sub;
		if(thread_y+T1*1<total_y)for(h7l=threadIdx.x;h7l<h7l_hi;h7l+=blockDim.x){
		h7=h7l+h7T;
		t2sub_shm[in1_idxl+T1*1][h7l] = t2sub_d[t2sub_d_off+h7*h7ld_t2sub];
		}
		if(thread_x+T1*1<total_x)for(h7l=threadIdx.y;h7l<h7l_hi;h7l+=blockDim.y){
		h7=h7l+h7T;
		v2sub_shm[h7l][in2_idxl+T1*1] = v2sub_d[v2sub_d_off+h7*h7ld_v2sub];
		}
		t2sub_d_off=p4_2*p4ld_t2sub+p5_2*p5ld_t2sub+h1_2*h1ld_t2sub;
		v2sub_d_off=h3_2*h3ld_v2sub+h2_2*h2ld_v2sub;
		if(thread_y+T1*2<total_y)for(h7l=threadIdx.x;h7l<h7l_hi;h7l+=blockDim.x){
		h7=h7l+h7T;
		t2sub_shm[in1_idxl+T1*2][h7l] = t2sub_d[t2sub_d_off+h7*h7ld_t2sub];
		}
		if(thread_x+T1*2<total_x)for(h7l=threadIdx.y;h7l<h7l_hi;h7l+=blockDim.y){
		h7=h7l+h7T;
		v2sub_shm[h7l][in2_idxl+T1*2] = v2sub_d[v2sub_d_off+h7*h7ld_v2sub];
		}
		t2sub_d_off=p4_3*p4ld_t2sub+p5_3*p5ld_t2sub+h1_3*h1ld_t2sub;
		v2sub_d_off=h3_3*h3ld_v2sub+h2_3*h2ld_v2sub;
		if(thread_y+T1*3<total_y)for(h7l=threadIdx.x;h7l<h7l_hi;h7l+=blockDim.x){
		h7=h7l+h7T;
		t2sub_shm[in1_idxl+T1*3][h7l] = t2sub_d[t2sub_d_off+h7*h7ld_t2sub];
		}
		if(thread_x+T1*3<total_x)for(h7l=threadIdx.y;h7l<h7l_hi;h7l+=blockDim.y){
		h7=h7l+h7T;
		v2sub_shm[h7l][in2_idxl+T1*3] = v2sub_d[v2sub_d_off+h7*h7ld_v2sub];
		}
		__syncthreads();
		for(h7l=0;h7l<h7l_hi;++h7l){
		a1=t2sub_shm[in1_idxl+T1*0][h7l];
		a2=t2sub_shm[in1_idxl+T1*1][h7l];
		a3=t2sub_shm[in1_idxl+T1*2][h7l];
		a4=t2sub_shm[in1_idxl+T1*3][h7l];
		b1=v2sub_shm[h7l][in2_idxl+T2*0];
		b2=v2sub_shm[h7l][in2_idxl+T2*1];
		b3=v2sub_shm[h7l][in2_idxl+T2*2];
		b4=v2sub_shm[h7l][in2_idxl+T2*3];
		tlocal1+=a1*b1;
		tlocal2+=a2*b1;
		tlocal3+=a3*b1;
		tlocal4+=a4*b1;
		tlocal5+=a1*b2;
		tlocal6+=a2*b2;
		tlocal7+=a3*b2;
		tlocal8+=a4*b2;
		tlocal9+=a1*b3;
		tlocal10+=a2*b3;
		tlocal11+=a3*b3;
		tlocal12+=a4*b3;
		tlocal13+=a1*b4;
		tlocal14+=a2*b4;
		tlocal15+=a3*b4;
		tlocal16+=a4*b4;
		}
		__syncthreads();
	}
	if(thread_x+T1*0<total_x){
		if(thread_y+T2*3<total_y) {
		triplesx_d[h3_0*h3ld_triplesx+h1_0*h1ld_triplesx+h2_0*h2ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]+=tlocal1;
		triplesx_d[h3_0*h3ld_triplesx+h1_1*h1ld_triplesx+h2_0*h2ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx]+=tlocal2;
		triplesx_d[h3_0*h3ld_triplesx+h1_2*h1ld_triplesx+h2_0*h2ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx]+=tlocal3;
		triplesx_d[h3_0*h3ld_triplesx+h1_3*h1ld_triplesx+h2_0*h2ld_triplesx+p5_3*p5ld_triplesx+p4_3*p4ld_triplesx]+=tlocal4;
		}
		else if(thread_y+T2*2<total_y) {
		triplesx_d[h3_0*h3ld_triplesx+h1_0*h1ld_triplesx+h2_0*h2ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]+=tlocal1;
		triplesx_d[h3_0*h3ld_triplesx+h1_1*h1ld_triplesx+h2_0*h2ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx]+=tlocal2;
		triplesx_d[h3_0*h3ld_triplesx+h1_2*h1ld_triplesx+h2_0*h2ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx]+=tlocal3;
		}
		else if(thread_y+T2*1<total_y) {
		triplesx_d[h3_0*h3ld_triplesx+h1_0*h1ld_triplesx+h2_0*h2ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]+=tlocal1;
		triplesx_d[h3_0*h3ld_triplesx+h1_1*h1ld_triplesx+h2_0*h2ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx]+=tlocal2;
		}
		else if(thread_y+T2*0<total_y) {
		triplesx_d[h3_0*h3ld_triplesx+h1_0*h1ld_triplesx+h2_0*h2ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]+=tlocal1;
		}
	}
	if(thread_x+T1*1<total_x){
		if(thread_y+T2*3<total_y) {
		triplesx_d[h3_1*h3ld_triplesx+h1_0*h1ld_triplesx+h2_1*h2ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]+=tlocal5;
		triplesx_d[h3_1*h3ld_triplesx+h1_1*h1ld_triplesx+h2_1*h2ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx]+=tlocal6;
		triplesx_d[h3_1*h3ld_triplesx+h1_2*h1ld_triplesx+h2_1*h2ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx]+=tlocal7;
		triplesx_d[h3_1*h3ld_triplesx+h1_3*h1ld_triplesx+h2_1*h2ld_triplesx+p5_3*p5ld_triplesx+p4_3*p4ld_triplesx]+=tlocal8;
		}
		else if(thread_y+T2*2<total_y) {
		triplesx_d[h3_1*h3ld_triplesx+h1_0*h1ld_triplesx+h2_1*h2ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]+=tlocal5;
		triplesx_d[h3_1*h3ld_triplesx+h1_1*h1ld_triplesx+h2_1*h2ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx]+=tlocal6;
		triplesx_d[h3_1*h3ld_triplesx+h1_2*h1ld_triplesx+h2_1*h2ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx]+=tlocal7;
		}
		else if(thread_y+T2*1<total_y) {
		triplesx_d[h3_1*h3ld_triplesx+h1_0*h1ld_triplesx+h2_1*h2ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]+=tlocal5;
		triplesx_d[h3_1*h3ld_triplesx+h1_1*h1ld_triplesx+h2_1*h2ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx]+=tlocal6;
		}
		else if(thread_y+T2*0<total_y) {
		triplesx_d[h3_1*h3ld_triplesx+h1_0*h1ld_triplesx+h2_1*h2ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]+=tlocal5;
		}
	}
	if(thread_x+T1*2<total_x){
		if(thread_y+T2*3<total_y) {
		triplesx_d[h3_2*h3ld_triplesx+h1_0*h1ld_triplesx+h2_2*h2ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]+=tlocal9;
		triplesx_d[h3_2*h3ld_triplesx+h1_1*h1ld_triplesx+h2_2*h2ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx]+=tlocal10;
		triplesx_d[h3_2*h3ld_triplesx+h1_2*h1ld_triplesx+h2_2*h2ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx]+=tlocal11;
		triplesx_d[h3_2*h3ld_triplesx+h1_3*h1ld_triplesx+h2_2*h2ld_triplesx+p5_3*p5ld_triplesx+p4_3*p4ld_triplesx]+=tlocal12;
		}
		else if(thread_y+T2*2<total_y) {
		triplesx_d[h3_2*h3ld_triplesx+h1_0*h1ld_triplesx+h2_2*h2ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]+=tlocal9;
		triplesx_d[h3_2*h3ld_triplesx+h1_1*h1ld_triplesx+h2_2*h2ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx]+=tlocal10;
		triplesx_d[h3_2*h3ld_triplesx+h1_2*h1ld_triplesx+h2_2*h2ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx]+=tlocal11;
		}
		else if(thread_y+T2*1<total_y) {
		triplesx_d[h3_2*h3ld_triplesx+h1_0*h1ld_triplesx+h2_2*h2ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]+=tlocal9;
		triplesx_d[h3_2*h3ld_triplesx+h1_1*h1ld_triplesx+h2_2*h2ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx]+=tlocal10;
		}
		else if(thread_y+T2*0<total_y) {
		triplesx_d[h3_2*h3ld_triplesx+h1_0*h1ld_triplesx+h2_2*h2ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]+=tlocal9;
		}
	}
	if(thread_x+T1*3<total_x){
		if(thread_y+T2*3<total_y) {
		triplesx_d[h3_3*h3ld_triplesx+h1_0*h1ld_triplesx+h2_3*h2ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]+=tlocal13;
		triplesx_d[h3_3*h3ld_triplesx+h1_1*h1ld_triplesx+h2_3*h2ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx]+=tlocal14;
		triplesx_d[h3_3*h3ld_triplesx+h1_2*h1ld_triplesx+h2_3*h2ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx]+=tlocal15;
		triplesx_d[h3_3*h3ld_triplesx+h1_3*h1ld_triplesx+h2_3*h2ld_triplesx+p5_3*p5ld_triplesx+p4_3*p4ld_triplesx]+=tlocal16;
		}
		else if(thread_y+T2*2<total_y) {
		triplesx_d[h3_3*h3ld_triplesx+h1_0*h1ld_triplesx+h2_3*h2ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]+=tlocal13;
		triplesx_d[h3_3*h3ld_triplesx+h1_1*h1ld_triplesx+h2_3*h2ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx]+=tlocal14;
		triplesx_d[h3_3*h3ld_triplesx+h1_2*h1ld_triplesx+h2_3*h2ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx]+=tlocal15;
		}
		else if(thread_y+T2*1<total_y) {
		triplesx_d[h3_3*h3ld_triplesx+h1_0*h1ld_triplesx+h2_3*h2ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]+=tlocal13;
		triplesx_d[h3_3*h3ld_triplesx+h1_1*h1ld_triplesx+h2_3*h2ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx]+=tlocal14;
		}
		else if(thread_y+T2*0<total_y) {
		triplesx_d[h3_3*h3ld_triplesx+h1_0*h1ld_triplesx+h2_3*h2ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]+=tlocal13;
		}
	}
	__syncthreads();
}
void ma_sd_t_d1_2_cuda(size_t h1d, size_t h2d, size_t h3d, size_t h7d, size_t p4d, size_t p5d, size_t p6d, double *triplesx, double *t2sub, double *v2sub) 
{
	h2d=h2d*p6d;
	size_t h7ld_t2sub,p4ld_t2sub,p5ld_t2sub,h1ld_t2sub,h3ld_v2sub,h2ld_v2sub,h7ld_v2sub,h3ld_triplesx,h1ld_triplesx,h2ld_triplesx,p5ld_triplesx,p4ld_triplesx;
	size_t size_triplesx,size_block_triplesx,size_el_block_triplesx,size_t2sub,size_v2sub;
	cudaStream_t *streams;
	size_t nstreams,i;
	double *t2sub_d,*v2sub_d;
	size_triplesx=h3d*h1d*h2d*p5d*p4d*sizeof(double);
	size_t2sub=h7d*p4d*p5d*h1d*sizeof(double);
	size_v2sub=h3d*h2d*h7d*sizeof(double);
	cudaFuncSetCacheConfig(ma_sd_t_d1_2_kernel, cudaFuncCachePreferShared);
	nstreams=1;
	size_block_triplesx=size_triplesx/nstreams;
	size_el_block_triplesx=size_block_triplesx/sizeof(double);
	t2sub_d=(double*)getGpuMem(size_t2sub);
	v2sub_d=(double*)getGpuMem(size_v2sub);
	streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
	assert(streams!= NULL);
	for(i=0;i<nstreams;++i) {
		// CUDA_SAFE(
			cudaStreamCreate(&streams[i]);//) ;
	}
	//CUDA_SAFE(
		cudaMemcpy(t2sub_d,t2sub,size_t2sub,cudaMemcpyHostToDevice); //);
	//CUDA_SAFE(  
		cudaMemcpy(v2sub_d,v2sub,size_v2sub,cudaMemcpyHostToDevice); //);
	h7ld_t2sub=1;
	p4ld_t2sub=h7d;
	p5ld_t2sub=p4d*h7d;
	h1ld_t2sub=p5d*p4d*h7d;
	h3ld_v2sub=1;
	h2ld_v2sub=h3d;
	h7ld_v2sub=h2d*h3d;
	h3ld_triplesx=1;
	h1ld_triplesx=h3d;
	h2ld_triplesx=h1d*h3d;
	p5ld_triplesx=h2d*h1d*h3d;
	p4ld_triplesx=p5d*h2d*h1d*h3d;
	size_t total_x = h3d*h2d*1;
	size_t total_y = p4d*p5d*h1d;
	//printf("Blocks %d %d\n", total_x, total_y); 
	//fflush(stdout);    
	dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
	for(i=0;i<nstreams;++i){
		ma_sd_t_d1_2_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(h1d,h2d,h3d,h7d,p4d,p5d,h7ld_t2sub,p4ld_t2sub,p5ld_t2sub,h1ld_t2sub,h3ld_v2sub,h2ld_v2sub,h7ld_v2sub,h3ld_triplesx,h1ld_triplesx,h2ld_triplesx,p5ld_triplesx,p4ld_triplesx,ma_t3_d,t2sub_d,v2sub_d,i,total_x,total_y);
		CHECK_ERR("Kernel execution failed");
	}
	cudaThreadSynchronize();
	for(i=0;i<nstreams;++i){
		cudaStreamDestroy(streams[i]);}
	freeGpuMem(t2sub_d);
	freeGpuMem(v2sub_d);
	free(streams);
}
#undef T1
#undef T2
#undef Tcomm
void ma_sd_t_d1_2_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* h7d, Integer* p4d, Integer* p5d, Integer* p6d, double *triplesx, double *t2sub, double *v2sub) 
{
	ma_sd_t_d1_2_cuda((int)*h1d,(int)*h2d,(int)*h3d,(int)*h7d,(int)*p4d,(int)*p5d,(int)*p6d,triplesx,t2sub,v2sub);
}
/*----------------------------------------------------------------------*
*triplesx[h1,h3,p5,p4] -= t2sub[h7,p4,p5,h1] * v2sub[h3,h7]
*----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void ma_sd_t_d1_3_kernel(size_t h1d,size_t h3d,size_t h7d,size_t p4d,size_t p5d,
	size_t h7ld_t2sub,size_t p4ld_t2sub,size_t p5ld_t2sub,size_t h1ld_t2sub,size_t h3ld_v2sub,size_t h7ld_v2sub,
	size_t h1ld_triplesx,size_t h3ld_triplesx,size_t p5ld_triplesx,size_t p4ld_triplesx,
	double *triplesx_d, double *t2sub_d, double *v2sub_d,size_t unused_idx, size_t total_x, size_t total_y) 
{
	size_t h1_0,h1_1,h1_2,h1_3,h3_0,h3_1,h3_2,h3_3,h7,p4_0,p4_1,p4_2,p4_3,p5_0,p5_1,p5_2,p5_3;
	double a1,b1;
	double a2,b2;
	double a3,b3;
	double a4,b4;
	size_t in1_idxl,in2_idxl,h7l,h7T;
	__shared__ double t2sub_shm[4*T1][Tcomm];
	__shared__ double v2sub_shm[Tcomm][4*T2];
	size_t rest_x=blockIdx.x;
	size_t rest_y=blockIdx.y;
	size_t thread_x = T2*4 * rest_x + threadIdx.x;
	size_t thread_y = T1*4 * rest_y + threadIdx.y;
	in1_idxl=threadIdx.y;
	in2_idxl=threadIdx.x ;
	double tlocal1=0;
	double tlocal2=0;
	double tlocal3=0;
	double tlocal4=0;
	double tlocal5=0;
	double tlocal6=0;
	double tlocal7=0;
	double tlocal8=0;
	double tlocal9=0;
	double tlocal10=0;
	double tlocal11=0;
	double tlocal12=0;
	double tlocal13=0;
	double tlocal14=0;
	double tlocal15=0;
	double tlocal16=0;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
	h1_0=rest_y%h1d;
	rest_y=rest_y/h1d;
	p5_0=rest_y%p5d;
	rest_y=rest_y/p5d;
	p4_0=rest_y;
	h3_0=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
	h1_1=rest_y%h1d;
	rest_y=rest_y/h1d;
	p5_1=rest_y%p5d;
	rest_y=rest_y/p5d;
	p4_1=rest_y;
	h3_1=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
	h1_2=rest_y%h1d;
	rest_y=rest_y/h1d;
	p5_2=rest_y%p5d;
	rest_y=rest_y/p5d;
	p4_2=rest_y;
	h3_2=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
	h1_3=rest_y%h1d;
	rest_y=rest_y/h1d;
	p5_3=rest_y%p5d;
	rest_y=rest_y/p5d;
	p4_3=rest_y;
	h3_3=rest_x;
	size_t t2sub_d_off, v2sub_d_off;for(h7T=0;h7T<h7d;h7T+=Tcomm){size_t h7l_hi;
		h7l_hi = MIN(Tcomm+h7T,h7d)-h7T;
		t2sub_d_off=p4_0*p4ld_t2sub+p5_0*p5ld_t2sub+h1_0*h1ld_t2sub;
		v2sub_d_off=h3_0*h3ld_v2sub;
		if(thread_y+T1*0<total_y)for(h7l=threadIdx.x;h7l<h7l_hi;h7l+=blockDim.x){
		h7=h7l+h7T;
		t2sub_shm[in1_idxl+T1*0][h7l] = t2sub_d[t2sub_d_off+h7*h7ld_t2sub];
		}
		if(thread_x+T1*0<total_x)for(h7l=threadIdx.y;h7l<h7l_hi;h7l+=blockDim.y){
		h7=h7l+h7T;
		v2sub_shm[h7l][in2_idxl+T1*0] = v2sub_d[v2sub_d_off+h7*h7ld_v2sub];
		}
		t2sub_d_off=p4_1*p4ld_t2sub+p5_1*p5ld_t2sub+h1_1*h1ld_t2sub;
		v2sub_d_off=h3_1*h3ld_v2sub;
		if(thread_y+T1*1<total_y)for(h7l=threadIdx.x;h7l<h7l_hi;h7l+=blockDim.x){
		h7=h7l+h7T;
		t2sub_shm[in1_idxl+T1*1][h7l] = t2sub_d[t2sub_d_off+h7*h7ld_t2sub];
		}
		if(thread_x+T1*1<total_x)for(h7l=threadIdx.y;h7l<h7l_hi;h7l+=blockDim.y){
		h7=h7l+h7T;
		v2sub_shm[h7l][in2_idxl+T1*1] = v2sub_d[v2sub_d_off+h7*h7ld_v2sub];
		}
		t2sub_d_off=p4_2*p4ld_t2sub+p5_2*p5ld_t2sub+h1_2*h1ld_t2sub;
		v2sub_d_off=h3_2*h3ld_v2sub;
		if(thread_y+T1*2<total_y)for(h7l=threadIdx.x;h7l<h7l_hi;h7l+=blockDim.x){
		h7=h7l+h7T;
		t2sub_shm[in1_idxl+T1*2][h7l] = t2sub_d[t2sub_d_off+h7*h7ld_t2sub];
		}
		if(thread_x+T1*2<total_x)for(h7l=threadIdx.y;h7l<h7l_hi;h7l+=blockDim.y){
		h7=h7l+h7T;
		v2sub_shm[h7l][in2_idxl+T1*2] = v2sub_d[v2sub_d_off+h7*h7ld_v2sub];
		}
		t2sub_d_off=p4_3*p4ld_t2sub+p5_3*p5ld_t2sub+h1_3*h1ld_t2sub;
		v2sub_d_off=h3_3*h3ld_v2sub;
		if(thread_y+T1*3<total_y)for(h7l=threadIdx.x;h7l<h7l_hi;h7l+=blockDim.x){
		h7=h7l+h7T;
		t2sub_shm[in1_idxl+T1*3][h7l] = t2sub_d[t2sub_d_off+h7*h7ld_t2sub];
		}
		if(thread_x+T1*3<total_x)for(h7l=threadIdx.y;h7l<h7l_hi;h7l+=blockDim.y){
		h7=h7l+h7T;
		v2sub_shm[h7l][in2_idxl+T1*3] = v2sub_d[v2sub_d_off+h7*h7ld_v2sub];
		}
		__syncthreads();
		for(h7l=0;h7l<h7l_hi;++h7l){
		a1=t2sub_shm[in1_idxl+T1*0][h7l];
		a2=t2sub_shm[in1_idxl+T1*1][h7l];
		a3=t2sub_shm[in1_idxl+T1*2][h7l];
		a4=t2sub_shm[in1_idxl+T1*3][h7l];
		b1=v2sub_shm[h7l][in2_idxl+T2*0];
		b2=v2sub_shm[h7l][in2_idxl+T2*1];
		b3=v2sub_shm[h7l][in2_idxl+T2*2];
		b4=v2sub_shm[h7l][in2_idxl+T2*3];
		tlocal1+=a1*b1;
		tlocal2+=a2*b1;
		tlocal3+=a3*b1;
		tlocal4+=a4*b1;
		tlocal5+=a1*b2;
		tlocal6+=a2*b2;
		tlocal7+=a3*b2;
		tlocal8+=a4*b2;
		tlocal9+=a1*b3;
		tlocal10+=a2*b3;
		tlocal11+=a3*b3;
		tlocal12+=a4*b3;
		tlocal13+=a1*b4;
		tlocal14+=a2*b4;
		tlocal15+=a3*b4;
		tlocal16+=a4*b4;
		}
		__syncthreads();
	}
	if(thread_x+T1*0<total_x){
		if(thread_y+T2*3<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_0*h3ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]-=tlocal1;
		triplesx_d[h1_1*h1ld_triplesx+h3_0*h3ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx]-=tlocal2;
		triplesx_d[h1_2*h1ld_triplesx+h3_0*h3ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx]-=tlocal3;
		triplesx_d[h1_3*h1ld_triplesx+h3_0*h3ld_triplesx+p5_3*p5ld_triplesx+p4_3*p4ld_triplesx]-=tlocal4;
		}
		else if(thread_y+T2*2<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_0*h3ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]-=tlocal1;
		triplesx_d[h1_1*h1ld_triplesx+h3_0*h3ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx]-=tlocal2;
		triplesx_d[h1_2*h1ld_triplesx+h3_0*h3ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx]-=tlocal3;
		}
		else if(thread_y+T2*1<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_0*h3ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]-=tlocal1;
		triplesx_d[h1_1*h1ld_triplesx+h3_0*h3ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx]-=tlocal2;
		}
		else if(thread_y+T2*0<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_0*h3ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]-=tlocal1;
		}
	}
	if(thread_x+T1*1<total_x){
		if(thread_y+T2*3<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_1*h3ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]-=tlocal5;
		triplesx_d[h1_1*h1ld_triplesx+h3_1*h3ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx]-=tlocal6;
		triplesx_d[h1_2*h1ld_triplesx+h3_1*h3ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx]-=tlocal7;
		triplesx_d[h1_3*h1ld_triplesx+h3_1*h3ld_triplesx+p5_3*p5ld_triplesx+p4_3*p4ld_triplesx]-=tlocal8;
		}
		else if(thread_y+T2*2<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_1*h3ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]-=tlocal5;
		triplesx_d[h1_1*h1ld_triplesx+h3_1*h3ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx]-=tlocal6;
		triplesx_d[h1_2*h1ld_triplesx+h3_1*h3ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx]-=tlocal7;
		}
		else if(thread_y+T2*1<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_1*h3ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]-=tlocal5;
		triplesx_d[h1_1*h1ld_triplesx+h3_1*h3ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx]-=tlocal6;
		}
		else if(thread_y+T2*0<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_1*h3ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]-=tlocal5;
		}
	}
	if(thread_x+T1*2<total_x){
		if(thread_y+T2*3<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_2*h3ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]-=tlocal9;
		triplesx_d[h1_1*h1ld_triplesx+h3_2*h3ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx]-=tlocal10;
		triplesx_d[h1_2*h1ld_triplesx+h3_2*h3ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx]-=tlocal11;
		triplesx_d[h1_3*h1ld_triplesx+h3_2*h3ld_triplesx+p5_3*p5ld_triplesx+p4_3*p4ld_triplesx]-=tlocal12;
		}
		else if(thread_y+T2*2<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_2*h3ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]-=tlocal9;
		triplesx_d[h1_1*h1ld_triplesx+h3_2*h3ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx]-=tlocal10;
		triplesx_d[h1_2*h1ld_triplesx+h3_2*h3ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx]-=tlocal11;
		}
		else if(thread_y+T2*1<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_2*h3ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]-=tlocal9;
		triplesx_d[h1_1*h1ld_triplesx+h3_2*h3ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx]-=tlocal10;
		}
		else if(thread_y+T2*0<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_2*h3ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]-=tlocal9;
		}
	}
	if(thread_x+T1*3<total_x){
		if(thread_y+T2*3<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_3*h3ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]-=tlocal13;
		triplesx_d[h1_1*h1ld_triplesx+h3_3*h3ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx]-=tlocal14;
		triplesx_d[h1_2*h1ld_triplesx+h3_3*h3ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx]-=tlocal15;
		triplesx_d[h1_3*h1ld_triplesx+h3_3*h3ld_triplesx+p5_3*p5ld_triplesx+p4_3*p4ld_triplesx]-=tlocal16;
		}
		else if(thread_y+T2*2<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_3*h3ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]-=tlocal13;
		triplesx_d[h1_1*h1ld_triplesx+h3_3*h3ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx]-=tlocal14;
		triplesx_d[h1_2*h1ld_triplesx+h3_3*h3ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx]-=tlocal15;
		}
		else if(thread_y+T2*1<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_3*h3ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]-=tlocal13;
		triplesx_d[h1_1*h1ld_triplesx+h3_3*h3ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx]-=tlocal14;
		}
		else if(thread_y+T2*0<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_3*h3ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx]-=tlocal13;
		}
	}
	__syncthreads();
}
void ma_sd_t_d1_3_cuda(size_t h1d, size_t h2d, size_t h3d, size_t h7d, size_t p4d, size_t p5d, size_t p6d, double *triplesx, double *t2sub, double *v2sub) 
{
	h3d=h3d*h2d;
	h3d=h3d*p6d;
	size_t h7ld_t2sub,p4ld_t2sub,p5ld_t2sub,h1ld_t2sub,h3ld_v2sub,h7ld_v2sub,h1ld_triplesx,h3ld_triplesx,p5ld_triplesx,p4ld_triplesx;
	size_t size_triplesx,size_block_triplesx,size_el_block_triplesx,size_t2sub,size_v2sub;
	cudaStream_t *streams;
	size_t nstreams,i;
	double *t2sub_d,*v2sub_d;
	size_triplesx=h1d*h3d*p5d*p4d*sizeof(double);
	size_t2sub=h7d*p4d*p5d*h1d*sizeof(double);
	size_v2sub=h3d*h7d*sizeof(double);
	cudaFuncSetCacheConfig(ma_sd_t_d1_3_kernel, cudaFuncCachePreferShared);
	nstreams=1;
	size_block_triplesx=size_triplesx/nstreams;
	size_el_block_triplesx=size_block_triplesx/sizeof(double);
	t2sub_d=(double*)getGpuMem(size_t2sub);
	v2sub_d=(double*)getGpuMem(size_v2sub);
	streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
	assert(streams!= NULL);
	for(i=0;i<nstreams;++i) {
		CUDA_SAFE(cudaStreamCreate(&streams[i])) ;
	}
	CUDA_SAFE(cudaMemcpy(t2sub_d,t2sub,size_t2sub,cudaMemcpyHostToDevice));
	CUDA_SAFE(cudaMemcpy(v2sub_d,v2sub,size_v2sub,cudaMemcpyHostToDevice));
	h7ld_t2sub=1;
	p4ld_t2sub=h7d;
	p5ld_t2sub=p4d*h7d;
	h1ld_t2sub=p5d*p4d*h7d;
	h3ld_v2sub=1;
	h7ld_v2sub=h3d;
	h1ld_triplesx=1;
	h3ld_triplesx=h1d;
	p5ld_triplesx=h3d*h1d;
	p4ld_triplesx=p5d*h3d*h1d;
	size_t total_x = h3d*1;
	size_t total_y = p4d*p5d*h1d;
	dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
	for(i=0;i<nstreams;++i){
		ma_sd_t_d1_3_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(h1d,h3d,h7d,p4d,p5d,h7ld_t2sub,p4ld_t2sub,p5ld_t2sub,h1ld_t2sub,h3ld_v2sub,h7ld_v2sub,h1ld_triplesx,h3ld_triplesx,p5ld_triplesx,p4ld_triplesx,ma_t3_d,t2sub_d,v2sub_d,i,total_x,total_y);
		CHECK_ERR("Kernel execution failed");
	}
	cudaThreadSynchronize();
	for(i=0;i<nstreams;++i){
		cudaStreamDestroy(streams[i]);}
	freeGpuMem(t2sub_d);
	freeGpuMem(v2sub_d);
	free(streams);
}
#undef T1
#undef T2
#undef Tcomm
void ma_sd_t_d1_3_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* h7d, Integer* p4d, Integer* p5d, Integer* p6d, double *triplesx, double *t2sub, double *v2sub) 
{
	ma_sd_t_d1_3_cuda((int)*h1d,(int)*h2d,(int)*h3d,(int)*h7d,(int)*p4d,(int)*p5d,(int)*p6d,triplesx,t2sub,v2sub);
}
/*----------------------------------------------------------------------*
*triplesx[h3,h1,p5,p4,p6] -= t2sub[h7,p4,p5,h1] * v2sub[h3,p6,h7]
*----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void ma_sd_t_d1_4_kernel(size_t h1d,size_t h3d,size_t h7d,size_t p4d,size_t p5d,size_t p6d,
	size_t h7ld_t2sub,size_t p4ld_t2sub,size_t p5ld_t2sub,size_t h1ld_t2sub,size_t h3ld_v2sub,size_t p6ld_v2sub,size_t h7ld_v2sub,
	size_t h3ld_triplesx,size_t h1ld_triplesx,size_t p5ld_triplesx,size_t p4ld_triplesx,size_t p6ld_triplesx,
	double *triplesx_d, double *t2sub_d, double *v2sub_d,size_t unused_idx, size_t total_x, size_t total_y) 
{
	size_t h1_0,h1_1,h1_2,h1_3,h3_0,h3_1,h3_2,h3_3,h7,p4_0,p4_1,p4_2,p4_3,p5_0,p5_1,p5_2,p5_3,p6_0,p6_1,p6_2,p6_3;
	double a1,b1;
	double a2,b2;
	double a3,b3;
	double a4,b4;
	size_t in1_idxl,in2_idxl,h7l,h7T;
	__shared__ double t2sub_shm[4*T1][Tcomm];
	__shared__ double v2sub_shm[Tcomm][4*T2];
	size_t rest_x=blockIdx.x;
	size_t rest_y=blockIdx.y;
	size_t thread_x = T2*4 * rest_x + threadIdx.x;
	size_t thread_y = T1*4 * rest_y + threadIdx.y;
	in1_idxl=threadIdx.y;
	in2_idxl=threadIdx.x ;
	double tlocal1=0;
	double tlocal2=0;
	double tlocal3=0;
	double tlocal4=0;
	double tlocal5=0;
	double tlocal6=0;
	double tlocal7=0;
	double tlocal8=0;
	double tlocal9=0;
	double tlocal10=0;
	double tlocal11=0;
	double tlocal12=0;
	double tlocal13=0;
	double tlocal14=0;
	double tlocal15=0;
	double tlocal16=0;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
	h3_0=rest_x%h3d;
	rest_x=rest_x/h3d;
	h1_0=rest_y%h1d;
	rest_y=rest_y/h1d;
	p5_0=rest_y%p5d;
	rest_y=rest_y/p5d;
	p4_0=rest_y;
	p6_0=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
	h3_1=rest_x%h3d;
	rest_x=rest_x/h3d;
	h1_1=rest_y%h1d;
	rest_y=rest_y/h1d;
	p5_1=rest_y%p5d;
	rest_y=rest_y/p5d;
	p4_1=rest_y;
	p6_1=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
	h3_2=rest_x%h3d;
	rest_x=rest_x/h3d;
	h1_2=rest_y%h1d;
	rest_y=rest_y/h1d;
	p5_2=rest_y%p5d;
	rest_y=rest_y/p5d;
	p4_2=rest_y;
	p6_2=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
	h3_3=rest_x%h3d;
	rest_x=rest_x/h3d;
	h1_3=rest_y%h1d;
	rest_y=rest_y/h1d;
	p5_3=rest_y%p5d;
	rest_y=rest_y/p5d;
	p4_3=rest_y;
	p6_3=rest_x;
	size_t t2sub_d_off, v2sub_d_off;for(h7T=0;h7T<h7d;h7T+=Tcomm){size_t h7l_hi;
		h7l_hi = MIN(Tcomm+h7T,h7d)-h7T;
		t2sub_d_off=p4_0*p4ld_t2sub+p5_0*p5ld_t2sub+h1_0*h1ld_t2sub;
		v2sub_d_off=h3_0*h3ld_v2sub+p6_0*p6ld_v2sub;
		if(thread_y+T1*0<total_y)for(h7l=threadIdx.x;h7l<h7l_hi;h7l+=blockDim.x){
		h7=h7l+h7T;
		t2sub_shm[in1_idxl+T1*0][h7l] = t2sub_d[t2sub_d_off+h7*h7ld_t2sub];
		}
		if(thread_x+T1*0<total_x)for(h7l=threadIdx.y;h7l<h7l_hi;h7l+=blockDim.y){
		h7=h7l+h7T;
		v2sub_shm[h7l][in2_idxl+T1*0] = v2sub_d[v2sub_d_off+h7*h7ld_v2sub];
		}
		t2sub_d_off=p4_1*p4ld_t2sub+p5_1*p5ld_t2sub+h1_1*h1ld_t2sub;
		v2sub_d_off=h3_1*h3ld_v2sub+p6_1*p6ld_v2sub;
		if(thread_y+T1*1<total_y)for(h7l=threadIdx.x;h7l<h7l_hi;h7l+=blockDim.x){
		h7=h7l+h7T;
		t2sub_shm[in1_idxl+T1*1][h7l] = t2sub_d[t2sub_d_off+h7*h7ld_t2sub];
		}
		if(thread_x+T1*1<total_x)for(h7l=threadIdx.y;h7l<h7l_hi;h7l+=blockDim.y){
		h7=h7l+h7T;
		v2sub_shm[h7l][in2_idxl+T1*1] = v2sub_d[v2sub_d_off+h7*h7ld_v2sub];
		}
		t2sub_d_off=p4_2*p4ld_t2sub+p5_2*p5ld_t2sub+h1_2*h1ld_t2sub;
		v2sub_d_off=h3_2*h3ld_v2sub+p6_2*p6ld_v2sub;
		if(thread_y+T1*2<total_y)for(h7l=threadIdx.x;h7l<h7l_hi;h7l+=blockDim.x){
		h7=h7l+h7T;
		t2sub_shm[in1_idxl+T1*2][h7l] = t2sub_d[t2sub_d_off+h7*h7ld_t2sub];
		}
		if(thread_x+T1*2<total_x)for(h7l=threadIdx.y;h7l<h7l_hi;h7l+=blockDim.y){
		h7=h7l+h7T;
		v2sub_shm[h7l][in2_idxl+T1*2] = v2sub_d[v2sub_d_off+h7*h7ld_v2sub];
		}
		t2sub_d_off=p4_3*p4ld_t2sub+p5_3*p5ld_t2sub+h1_3*h1ld_t2sub;
		v2sub_d_off=h3_3*h3ld_v2sub+p6_3*p6ld_v2sub;
		if(thread_y+T1*3<total_y)for(h7l=threadIdx.x;h7l<h7l_hi;h7l+=blockDim.x){
		h7=h7l+h7T;
		t2sub_shm[in1_idxl+T1*3][h7l] = t2sub_d[t2sub_d_off+h7*h7ld_t2sub];
		}
		if(thread_x+T1*3<total_x)for(h7l=threadIdx.y;h7l<h7l_hi;h7l+=blockDim.y){
		h7=h7l+h7T;
		v2sub_shm[h7l][in2_idxl+T1*3] = v2sub_d[v2sub_d_off+h7*h7ld_v2sub];
		}
		__syncthreads();
		for(h7l=0;h7l<h7l_hi;++h7l){
		a1=t2sub_shm[in1_idxl+T1*0][h7l];
		a2=t2sub_shm[in1_idxl+T1*1][h7l];
		a3=t2sub_shm[in1_idxl+T1*2][h7l];
		a4=t2sub_shm[in1_idxl+T1*3][h7l];
		b1=v2sub_shm[h7l][in2_idxl+T2*0];
		b2=v2sub_shm[h7l][in2_idxl+T2*1];
		b3=v2sub_shm[h7l][in2_idxl+T2*2];
		b4=v2sub_shm[h7l][in2_idxl+T2*3];
		tlocal1+=a1*b1;
		tlocal2+=a2*b1;
		tlocal3+=a3*b1;
		tlocal4+=a4*b1;
		tlocal5+=a1*b2;
		tlocal6+=a2*b2;
		tlocal7+=a3*b2;
		tlocal8+=a4*b2;
		tlocal9+=a1*b3;
		tlocal10+=a2*b3;
		tlocal11+=a3*b3;
		tlocal12+=a4*b3;
		tlocal13+=a1*b4;
		tlocal14+=a2*b4;
		tlocal15+=a3*b4;
		tlocal16+=a4*b4;
		}
		__syncthreads();
	}
	if(thread_x+T1*0<total_x){
		if(thread_y+T2*3<total_y) {
		triplesx_d[h3_0*h3ld_triplesx+h1_0*h1ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_0*p6ld_triplesx]-=tlocal1;
		triplesx_d[h3_0*h3ld_triplesx+h1_1*h1ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx+p6_0*p6ld_triplesx]-=tlocal2;
		triplesx_d[h3_0*h3ld_triplesx+h1_2*h1ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx+p6_0*p6ld_triplesx]-=tlocal3;
		triplesx_d[h3_0*h3ld_triplesx+h1_3*h1ld_triplesx+p5_3*p5ld_triplesx+p4_3*p4ld_triplesx+p6_0*p6ld_triplesx]-=tlocal4;
		}
		else if(thread_y+T2*2<total_y) {
		triplesx_d[h3_0*h3ld_triplesx+h1_0*h1ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_0*p6ld_triplesx]-=tlocal1;
		triplesx_d[h3_0*h3ld_triplesx+h1_1*h1ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx+p6_0*p6ld_triplesx]-=tlocal2;
		triplesx_d[h3_0*h3ld_triplesx+h1_2*h1ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx+p6_0*p6ld_triplesx]-=tlocal3;
		}
		else if(thread_y+T2*1<total_y) {
		triplesx_d[h3_0*h3ld_triplesx+h1_0*h1ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_0*p6ld_triplesx]-=tlocal1;
		triplesx_d[h3_0*h3ld_triplesx+h1_1*h1ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx+p6_0*p6ld_triplesx]-=tlocal2;
		}
		else if(thread_y+T2*0<total_y) {
		triplesx_d[h3_0*h3ld_triplesx+h1_0*h1ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_0*p6ld_triplesx]-=tlocal1;
		}
	}
	if(thread_x+T1*1<total_x){
		if(thread_y+T2*3<total_y) {
		triplesx_d[h3_1*h3ld_triplesx+h1_0*h1ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_1*p6ld_triplesx]-=tlocal5;
		triplesx_d[h3_1*h3ld_triplesx+h1_1*h1ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx+p6_1*p6ld_triplesx]-=tlocal6;
		triplesx_d[h3_1*h3ld_triplesx+h1_2*h1ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx+p6_1*p6ld_triplesx]-=tlocal7;
		triplesx_d[h3_1*h3ld_triplesx+h1_3*h1ld_triplesx+p5_3*p5ld_triplesx+p4_3*p4ld_triplesx+p6_1*p6ld_triplesx]-=tlocal8;
		}
		else if(thread_y+T2*2<total_y) {
		triplesx_d[h3_1*h3ld_triplesx+h1_0*h1ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_1*p6ld_triplesx]-=tlocal5;
		triplesx_d[h3_1*h3ld_triplesx+h1_1*h1ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx+p6_1*p6ld_triplesx]-=tlocal6;
		triplesx_d[h3_1*h3ld_triplesx+h1_2*h1ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx+p6_1*p6ld_triplesx]-=tlocal7;
		}
		else if(thread_y+T2*1<total_y) {
		triplesx_d[h3_1*h3ld_triplesx+h1_0*h1ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_1*p6ld_triplesx]-=tlocal5;
		triplesx_d[h3_1*h3ld_triplesx+h1_1*h1ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx+p6_1*p6ld_triplesx]-=tlocal6;
		}
		else if(thread_y+T2*0<total_y) {
		triplesx_d[h3_1*h3ld_triplesx+h1_0*h1ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_1*p6ld_triplesx]-=tlocal5;
		}
	}
	if(thread_x+T1*2<total_x){
		if(thread_y+T2*3<total_y) {
		triplesx_d[h3_2*h3ld_triplesx+h1_0*h1ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_2*p6ld_triplesx]-=tlocal9;
		triplesx_d[h3_2*h3ld_triplesx+h1_1*h1ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx+p6_2*p6ld_triplesx]-=tlocal10;
		triplesx_d[h3_2*h3ld_triplesx+h1_2*h1ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx+p6_2*p6ld_triplesx]-=tlocal11;
		triplesx_d[h3_2*h3ld_triplesx+h1_3*h1ld_triplesx+p5_3*p5ld_triplesx+p4_3*p4ld_triplesx+p6_2*p6ld_triplesx]-=tlocal12;
		}
		else if(thread_y+T2*2<total_y) {
		triplesx_d[h3_2*h3ld_triplesx+h1_0*h1ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_2*p6ld_triplesx]-=tlocal9;
		triplesx_d[h3_2*h3ld_triplesx+h1_1*h1ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx+p6_2*p6ld_triplesx]-=tlocal10;
		triplesx_d[h3_2*h3ld_triplesx+h1_2*h1ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx+p6_2*p6ld_triplesx]-=tlocal11;
		}
		else if(thread_y+T2*1<total_y) {
		triplesx_d[h3_2*h3ld_triplesx+h1_0*h1ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_2*p6ld_triplesx]-=tlocal9;
		triplesx_d[h3_2*h3ld_triplesx+h1_1*h1ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx+p6_2*p6ld_triplesx]-=tlocal10;
		}
		else if(thread_y+T2*0<total_y) {
		triplesx_d[h3_2*h3ld_triplesx+h1_0*h1ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_2*p6ld_triplesx]-=tlocal9;
		}
	}
	if(thread_x+T1*3<total_x){
		if(thread_y+T2*3<total_y) {
		triplesx_d[h3_3*h3ld_triplesx+h1_0*h1ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_3*p6ld_triplesx]-=tlocal13;
		triplesx_d[h3_3*h3ld_triplesx+h1_1*h1ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx+p6_3*p6ld_triplesx]-=tlocal14;
		triplesx_d[h3_3*h3ld_triplesx+h1_2*h1ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx+p6_3*p6ld_triplesx]-=tlocal15;
		triplesx_d[h3_3*h3ld_triplesx+h1_3*h1ld_triplesx+p5_3*p5ld_triplesx+p4_3*p4ld_triplesx+p6_3*p6ld_triplesx]-=tlocal16;
		}
		else if(thread_y+T2*2<total_y) {
		triplesx_d[h3_3*h3ld_triplesx+h1_0*h1ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_3*p6ld_triplesx]-=tlocal13;
		triplesx_d[h3_3*h3ld_triplesx+h1_1*h1ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx+p6_3*p6ld_triplesx]-=tlocal14;
		triplesx_d[h3_3*h3ld_triplesx+h1_2*h1ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx+p6_3*p6ld_triplesx]-=tlocal15;
		}
		else if(thread_y+T2*1<total_y) {
		triplesx_d[h3_3*h3ld_triplesx+h1_0*h1ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_3*p6ld_triplesx]-=tlocal13;
		triplesx_d[h3_3*h3ld_triplesx+h1_1*h1ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx+p6_3*p6ld_triplesx]-=tlocal14;
		}
		else if(thread_y+T2*0<total_y) {
		triplesx_d[h3_3*h3ld_triplesx+h1_0*h1ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_3*p6ld_triplesx]-=tlocal13;
		}
	}
	__syncthreads();
}

void ma_sd_t_d1_4_cuda(size_t h1d, size_t h2d, size_t h3d, size_t h7d, size_t p4d, size_t p5d, size_t p6d, double *triplesx, double *t2sub, double *v2sub) 
{
	h3d=h3d*h2d;
	size_t h7ld_t2sub,p4ld_t2sub,p5ld_t2sub,h1ld_t2sub,h3ld_v2sub,p6ld_v2sub,h7ld_v2sub,h3ld_triplesx,h1ld_triplesx,p5ld_triplesx,p4ld_triplesx,p6ld_triplesx;
	size_t size_triplesx,size_block_triplesx,size_el_block_triplesx,size_t2sub,size_v2sub;
	cudaStream_t *streams;
	size_t nstreams,i;
	double *t2sub_d,*v2sub_d;
	size_triplesx=h3d*h1d*p5d*p4d*p6d*sizeof(double);
	size_t2sub=h7d*p4d*p5d*h1d*sizeof(double);
	size_v2sub=h3d*p6d*h7d*sizeof(double);
	cudaFuncSetCacheConfig(ma_sd_t_d1_4_kernel, cudaFuncCachePreferShared);
	nstreams=1;
	size_block_triplesx=size_triplesx/nstreams;
	size_el_block_triplesx=size_block_triplesx/sizeof(double);
	t2sub_d=(double*)getGpuMem(size_t2sub);
	v2sub_d=(double*)getGpuMem(size_v2sub);
	streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
	assert(streams!= NULL);
	for(i=0;i<nstreams;++i) {
		CUDA_SAFE(cudaStreamCreate(&streams[i])) ;
	}
	CUDA_SAFE(cudaMemcpy(t2sub_d,t2sub,size_t2sub,cudaMemcpyHostToDevice));
	CUDA_SAFE(cudaMemcpy(v2sub_d,v2sub,size_v2sub,cudaMemcpyHostToDevice));
	h7ld_t2sub=1;
	p4ld_t2sub=h7d;
	p5ld_t2sub=p4d*h7d;
	h1ld_t2sub=p5d*p4d*h7d;
	h3ld_v2sub=1;
	p6ld_v2sub=h3d;
	h7ld_v2sub=p6d*h3d;
	h3ld_triplesx=1;
	h1ld_triplesx=h3d;
	p5ld_triplesx=h1d*h3d;
	p4ld_triplesx=p5d*h1d*h3d;
	p6ld_triplesx=p4d*p5d*h1d*h3d;
	size_t total_x = h3d*p6d*1;
	size_t total_y = p4d*p5d*h1d;
	dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
	for(i=0;i<nstreams;++i){
		ma_sd_t_d1_4_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(h1d,h3d,h7d,p4d,p5d,p6d,h7ld_t2sub,p4ld_t2sub,p5ld_t2sub,h1ld_t2sub,h3ld_v2sub,p6ld_v2sub,h7ld_v2sub,h3ld_triplesx,h1ld_triplesx,p5ld_triplesx,p4ld_triplesx,p6ld_triplesx,ma_t3_d,t2sub_d,v2sub_d,i,total_x,total_y);
		CHECK_ERR("Kernel execution failed");
	}
	cudaThreadSynchronize();
	for(i=0;i<nstreams;++i){
		cudaStreamDestroy(streams[i]);}
	freeGpuMem(t2sub_d);
	freeGpuMem(v2sub_d);
	free(streams);
}
#undef T1
#undef T2
#undef Tcomm
void ma_sd_t_d1_4_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* h7d, Integer* p4d, Integer* p5d, Integer* p6d, double *triplesx, double *t2sub, double *v2sub) 
{
	ma_sd_t_d1_4_cuda((int)*h1d,(int)*h2d,(int)*h3d,(int)*h7d,(int)*p4d,(int)*p5d,(int)*p6d,triplesx,t2sub,v2sub);
}
/*----------------------------------------------------------------------*
*triplesx[h3,h1,h2,p5,p4,p6] += t2sub[h7,p4,p5,h1] * v2sub[h3,h2,p6,h7]
*----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void ma_sd_t_d1_5_kernel(size_t h1d,size_t h2d,size_t h3d,size_t h7d,size_t p4d,size_t p5d,size_t p6d,
	size_t h7ld_t2sub,size_t p4ld_t2sub,size_t p5ld_t2sub,size_t h1ld_t2sub,size_t h3ld_v2sub,size_t h2ld_v2sub,size_t p6ld_v2sub,size_t h7ld_v2sub,
	size_t h3ld_triplesx,size_t h1ld_triplesx,size_t h2ld_triplesx,size_t p5ld_triplesx,size_t p4ld_triplesx,size_t p6ld_triplesx,
	double *triplesx_d, double *t2sub_d, double *v2sub_d,size_t unused_idx, size_t total_x, size_t total_y)
{
	size_t h1_0,h1_1,h1_2,h1_3,h2_0,h2_1,h2_2,h2_3,h3_0,h3_1,h3_2,h3_3,h7,p4_0,p4_1,p4_2,p4_3,p5_0,p5_1,p5_2,p5_3,p6_0,p6_1,p6_2,p6_3;
	double a1,b1;
	double a2,b2;
	double a3,b3;
	double a4,b4;
	size_t in1_idxl,in2_idxl,h7l,h7T;
	__shared__ double t2sub_shm[4*T1][Tcomm];
	__shared__ double v2sub_shm[Tcomm][4*T2];
	size_t rest_x=blockIdx.x;
	size_t rest_y=blockIdx.y;
	size_t thread_x = T2*4 * rest_x + threadIdx.x;
	size_t thread_y = T1*4 * rest_y + threadIdx.y;
	in1_idxl=threadIdx.y;
	in2_idxl=threadIdx.x ;
	double tlocal1=0;
	double tlocal2=0;
	double tlocal3=0;
	double tlocal4=0;
	double tlocal5=0;
	double tlocal6=0;
	double tlocal7=0;
	double tlocal8=0;
	double tlocal9=0;
	double tlocal10=0;
	double tlocal11=0;
	double tlocal12=0;
	double tlocal13=0;
	double tlocal14=0;
	double tlocal15=0;
	double tlocal16=0;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
	h3_0=rest_x%h3d;
	rest_x=rest_x/h3d;
	h1_0=rest_y%h1d;
	rest_y=rest_y/h1d;
	h2_0=rest_x%h2d;
	rest_x=rest_x/h2d;
	p5_0=rest_y%p5d;
	rest_y=rest_y/p5d;
	p4_0=rest_y;
	p6_0=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
	h3_1=rest_x%h3d;
	rest_x=rest_x/h3d;
	h1_1=rest_y%h1d;
	rest_y=rest_y/h1d;
	h2_1=rest_x%h2d;
	rest_x=rest_x/h2d;
	p5_1=rest_y%p5d;
	rest_y=rest_y/p5d;
	p4_1=rest_y;
	p6_1=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
	h3_2=rest_x%h3d;
	rest_x=rest_x/h3d;
	h1_2=rest_y%h1d;
	rest_y=rest_y/h1d;
	h2_2=rest_x%h2d;
	rest_x=rest_x/h2d;
	p5_2=rest_y%p5d;
	rest_y=rest_y/p5d;
	p4_2=rest_y;
	p6_2=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
	h3_3=rest_x%h3d;
	rest_x=rest_x/h3d;
	h1_3=rest_y%h1d;
	rest_y=rest_y/h1d;
	h2_3=rest_x%h2d;
	rest_x=rest_x/h2d;
	p5_3=rest_y%p5d;
	rest_y=rest_y/p5d;
	p4_3=rest_y;
	p6_3=rest_x;
	size_t t2sub_d_off, v2sub_d_off;for(h7T=0;h7T<h7d;h7T+=Tcomm){size_t h7l_hi;
		h7l_hi = MIN(Tcomm+h7T,h7d)-h7T;
		t2sub_d_off=p4_0*p4ld_t2sub+p5_0*p5ld_t2sub+h1_0*h1ld_t2sub;
		v2sub_d_off=h3_0*h3ld_v2sub+h2_0*h2ld_v2sub+p6_0*p6ld_v2sub;
		if(thread_y+T1*0<total_y)for(h7l=threadIdx.x;h7l<h7l_hi;h7l+=blockDim.x){
		h7=h7l+h7T;
		t2sub_shm[in1_idxl+T1*0][h7l] = t2sub_d[t2sub_d_off+h7*h7ld_t2sub];
		}
		if(thread_x+T1*0<total_x)for(h7l=threadIdx.y;h7l<h7l_hi;h7l+=blockDim.y){
		h7=h7l+h7T;
		v2sub_shm[h7l][in2_idxl+T1*0] = v2sub_d[v2sub_d_off+h7*h7ld_v2sub];
		}
		t2sub_d_off=p4_1*p4ld_t2sub+p5_1*p5ld_t2sub+h1_1*h1ld_t2sub;
		v2sub_d_off=h3_1*h3ld_v2sub+h2_1*h2ld_v2sub+p6_1*p6ld_v2sub;
		if(thread_y+T1*1<total_y)for(h7l=threadIdx.x;h7l<h7l_hi;h7l+=blockDim.x){
		h7=h7l+h7T;
		t2sub_shm[in1_idxl+T1*1][h7l] = t2sub_d[t2sub_d_off+h7*h7ld_t2sub];
		}
		if(thread_x+T1*1<total_x)for(h7l=threadIdx.y;h7l<h7l_hi;h7l+=blockDim.y){
		h7=h7l+h7T;
		v2sub_shm[h7l][in2_idxl+T1*1] = v2sub_d[v2sub_d_off+h7*h7ld_v2sub];
		}
		t2sub_d_off=p4_2*p4ld_t2sub+p5_2*p5ld_t2sub+h1_2*h1ld_t2sub;
		v2sub_d_off=h3_2*h3ld_v2sub+h2_2*h2ld_v2sub+p6_2*p6ld_v2sub;
		if(thread_y+T1*2<total_y)for(h7l=threadIdx.x;h7l<h7l_hi;h7l+=blockDim.x){
		h7=h7l+h7T;
		t2sub_shm[in1_idxl+T1*2][h7l] = t2sub_d[t2sub_d_off+h7*h7ld_t2sub];
		}
		if(thread_x+T1*2<total_x)for(h7l=threadIdx.y;h7l<h7l_hi;h7l+=blockDim.y){
		h7=h7l+h7T;
		v2sub_shm[h7l][in2_idxl+T1*2] = v2sub_d[v2sub_d_off+h7*h7ld_v2sub];
		}
		t2sub_d_off=p4_3*p4ld_t2sub+p5_3*p5ld_t2sub+h1_3*h1ld_t2sub;
		v2sub_d_off=h3_3*h3ld_v2sub+h2_3*h2ld_v2sub+p6_3*p6ld_v2sub;
		if(thread_y+T1*3<total_y)for(h7l=threadIdx.x;h7l<h7l_hi;h7l+=blockDim.x){
		h7=h7l+h7T;
		t2sub_shm[in1_idxl+T1*3][h7l] = t2sub_d[t2sub_d_off+h7*h7ld_t2sub];
		}
		if(thread_x+T1*3<total_x)for(h7l=threadIdx.y;h7l<h7l_hi;h7l+=blockDim.y){
		h7=h7l+h7T;
		v2sub_shm[h7l][in2_idxl+T1*3] = v2sub_d[v2sub_d_off+h7*h7ld_v2sub];
		}
		__syncthreads();
		for(h7l=0;h7l<h7l_hi;++h7l){
		a1=t2sub_shm[in1_idxl+T1*0][h7l];
		a2=t2sub_shm[in1_idxl+T1*1][h7l];
		a3=t2sub_shm[in1_idxl+T1*2][h7l];
		a4=t2sub_shm[in1_idxl+T1*3][h7l];
		b1=v2sub_shm[h7l][in2_idxl+T2*0];
		b2=v2sub_shm[h7l][in2_idxl+T2*1];
		b3=v2sub_shm[h7l][in2_idxl+T2*2];
		b4=v2sub_shm[h7l][in2_idxl+T2*3];
		tlocal1+=a1*b1;
		tlocal2+=a2*b1;
		tlocal3+=a3*b1;
		tlocal4+=a4*b1;
		tlocal5+=a1*b2;
		tlocal6+=a2*b2;
		tlocal7+=a3*b2;
		tlocal8+=a4*b2;
		tlocal9+=a1*b3;
		tlocal10+=a2*b3;
		tlocal11+=a3*b3;
		tlocal12+=a4*b3;
		tlocal13+=a1*b4;
		tlocal14+=a2*b4;
		tlocal15+=a3*b4;
		tlocal16+=a4*b4;
		}
		__syncthreads();
	}
	if(thread_x+T1*0<total_x){
		if(thread_y+T2*3<total_y) {
		triplesx_d[h3_0*h3ld_triplesx+h1_0*h1ld_triplesx+h2_0*h2ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_0*p6ld_triplesx]+=tlocal1;
		triplesx_d[h3_0*h3ld_triplesx+h1_1*h1ld_triplesx+h2_0*h2ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx+p6_0*p6ld_triplesx]+=tlocal2;
		triplesx_d[h3_0*h3ld_triplesx+h1_2*h1ld_triplesx+h2_0*h2ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx+p6_0*p6ld_triplesx]+=tlocal3;
		triplesx_d[h3_0*h3ld_triplesx+h1_3*h1ld_triplesx+h2_0*h2ld_triplesx+p5_3*p5ld_triplesx+p4_3*p4ld_triplesx+p6_0*p6ld_triplesx]+=tlocal4;
		}
		else if(thread_y+T2*2<total_y) {
		triplesx_d[h3_0*h3ld_triplesx+h1_0*h1ld_triplesx+h2_0*h2ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_0*p6ld_triplesx]+=tlocal1;
		triplesx_d[h3_0*h3ld_triplesx+h1_1*h1ld_triplesx+h2_0*h2ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx+p6_0*p6ld_triplesx]+=tlocal2;
		triplesx_d[h3_0*h3ld_triplesx+h1_2*h1ld_triplesx+h2_0*h2ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx+p6_0*p6ld_triplesx]+=tlocal3;
		}
		else if(thread_y+T2*1<total_y) {
		triplesx_d[h3_0*h3ld_triplesx+h1_0*h1ld_triplesx+h2_0*h2ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_0*p6ld_triplesx]+=tlocal1;
		triplesx_d[h3_0*h3ld_triplesx+h1_1*h1ld_triplesx+h2_0*h2ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx+p6_0*p6ld_triplesx]+=tlocal2;
		}
		else if(thread_y+T2*0<total_y) {
		triplesx_d[h3_0*h3ld_triplesx+h1_0*h1ld_triplesx+h2_0*h2ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_0*p6ld_triplesx]+=tlocal1;
		}
	}
	if(thread_x+T1*1<total_x){
		if(thread_y+T2*3<total_y) {
		triplesx_d[h3_1*h3ld_triplesx+h1_0*h1ld_triplesx+h2_1*h2ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_1*p6ld_triplesx]+=tlocal5;
		triplesx_d[h3_1*h3ld_triplesx+h1_1*h1ld_triplesx+h2_1*h2ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx+p6_1*p6ld_triplesx]+=tlocal6;
		triplesx_d[h3_1*h3ld_triplesx+h1_2*h1ld_triplesx+h2_1*h2ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx+p6_1*p6ld_triplesx]+=tlocal7;
		triplesx_d[h3_1*h3ld_triplesx+h1_3*h1ld_triplesx+h2_1*h2ld_triplesx+p5_3*p5ld_triplesx+p4_3*p4ld_triplesx+p6_1*p6ld_triplesx]+=tlocal8;
		}
		else if(thread_y+T2*2<total_y) {
		triplesx_d[h3_1*h3ld_triplesx+h1_0*h1ld_triplesx+h2_1*h2ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_1*p6ld_triplesx]+=tlocal5;
		triplesx_d[h3_1*h3ld_triplesx+h1_1*h1ld_triplesx+h2_1*h2ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx+p6_1*p6ld_triplesx]+=tlocal6;
		triplesx_d[h3_1*h3ld_triplesx+h1_2*h1ld_triplesx+h2_1*h2ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx+p6_1*p6ld_triplesx]+=tlocal7;
		}
		else if(thread_y+T2*1<total_y) {
		triplesx_d[h3_1*h3ld_triplesx+h1_0*h1ld_triplesx+h2_1*h2ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_1*p6ld_triplesx]+=tlocal5;
		triplesx_d[h3_1*h3ld_triplesx+h1_1*h1ld_triplesx+h2_1*h2ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx+p6_1*p6ld_triplesx]+=tlocal6;
		}
		else if(thread_y+T2*0<total_y) {
		triplesx_d[h3_1*h3ld_triplesx+h1_0*h1ld_triplesx+h2_1*h2ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_1*p6ld_triplesx]+=tlocal5;
		}
	}
	if(thread_x+T1*2<total_x){
		if(thread_y+T2*3<total_y) {
		triplesx_d[h3_2*h3ld_triplesx+h1_0*h1ld_triplesx+h2_2*h2ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_2*p6ld_triplesx]+=tlocal9;
		triplesx_d[h3_2*h3ld_triplesx+h1_1*h1ld_triplesx+h2_2*h2ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx+p6_2*p6ld_triplesx]+=tlocal10;
		triplesx_d[h3_2*h3ld_triplesx+h1_2*h1ld_triplesx+h2_2*h2ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx+p6_2*p6ld_triplesx]+=tlocal11;
		triplesx_d[h3_2*h3ld_triplesx+h1_3*h1ld_triplesx+h2_2*h2ld_triplesx+p5_3*p5ld_triplesx+p4_3*p4ld_triplesx+p6_2*p6ld_triplesx]+=tlocal12;
		}
		else if(thread_y+T2*2<total_y) {
		triplesx_d[h3_2*h3ld_triplesx+h1_0*h1ld_triplesx+h2_2*h2ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_2*p6ld_triplesx]+=tlocal9;
		triplesx_d[h3_2*h3ld_triplesx+h1_1*h1ld_triplesx+h2_2*h2ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx+p6_2*p6ld_triplesx]+=tlocal10;
		triplesx_d[h3_2*h3ld_triplesx+h1_2*h1ld_triplesx+h2_2*h2ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx+p6_2*p6ld_triplesx]+=tlocal11;
		}
		else if(thread_y+T2*1<total_y) {
		triplesx_d[h3_2*h3ld_triplesx+h1_0*h1ld_triplesx+h2_2*h2ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_2*p6ld_triplesx]+=tlocal9;
		triplesx_d[h3_2*h3ld_triplesx+h1_1*h1ld_triplesx+h2_2*h2ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx+p6_2*p6ld_triplesx]+=tlocal10;
		}
		else if(thread_y+T2*0<total_y) {
		triplesx_d[h3_2*h3ld_triplesx+h1_0*h1ld_triplesx+h2_2*h2ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_2*p6ld_triplesx]+=tlocal9;
		}
	}
	if(thread_x+T1*3<total_x){
		if(thread_y+T2*3<total_y) {
		triplesx_d[h3_3*h3ld_triplesx+h1_0*h1ld_triplesx+h2_3*h2ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_3*p6ld_triplesx]+=tlocal13;
		triplesx_d[h3_3*h3ld_triplesx+h1_1*h1ld_triplesx+h2_3*h2ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx+p6_3*p6ld_triplesx]+=tlocal14;
		triplesx_d[h3_3*h3ld_triplesx+h1_2*h1ld_triplesx+h2_3*h2ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx+p6_3*p6ld_triplesx]+=tlocal15;
		triplesx_d[h3_3*h3ld_triplesx+h1_3*h1ld_triplesx+h2_3*h2ld_triplesx+p5_3*p5ld_triplesx+p4_3*p4ld_triplesx+p6_3*p6ld_triplesx]+=tlocal16;
		}
		else if(thread_y+T2*2<total_y) {
		triplesx_d[h3_3*h3ld_triplesx+h1_0*h1ld_triplesx+h2_3*h2ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_3*p6ld_triplesx]+=tlocal13;
		triplesx_d[h3_3*h3ld_triplesx+h1_1*h1ld_triplesx+h2_3*h2ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx+p6_3*p6ld_triplesx]+=tlocal14;
		triplesx_d[h3_3*h3ld_triplesx+h1_2*h1ld_triplesx+h2_3*h2ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx+p6_3*p6ld_triplesx]+=tlocal15;
		}
		else if(thread_y+T2*1<total_y) {
		triplesx_d[h3_3*h3ld_triplesx+h1_0*h1ld_triplesx+h2_3*h2ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_3*p6ld_triplesx]+=tlocal13;
		triplesx_d[h3_3*h3ld_triplesx+h1_1*h1ld_triplesx+h2_3*h2ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx+p6_3*p6ld_triplesx]+=tlocal14;
		}
		else if(thread_y+T2*0<total_y) {
		triplesx_d[h3_3*h3ld_triplesx+h1_0*h1ld_triplesx+h2_3*h2ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_3*p6ld_triplesx]+=tlocal13;
		}
	}
	__syncthreads();
}
void ma_sd_t_d1_5_cuda(size_t h1d, size_t h2d, size_t h3d, size_t h7d, size_t p4d, size_t p5d, size_t p6d, double *triplesx, double *t2sub, double *v2sub) 
{
	size_t h7ld_t2sub,p4ld_t2sub,p5ld_t2sub,h1ld_t2sub,h3ld_v2sub,h2ld_v2sub,p6ld_v2sub,h7ld_v2sub,h3ld_triplesx,h1ld_triplesx,h2ld_triplesx,p5ld_triplesx,p4ld_triplesx,p6ld_triplesx;
	size_t size_triplesx,size_block_triplesx,size_el_block_triplesx,size_t2sub,size_v2sub;
	cudaStream_t *streams;
	size_t nstreams,i;
	double *t2sub_d,*v2sub_d;
	size_triplesx=h3d*h1d*h2d*p5d*p4d*p6d*sizeof(double);
	size_t2sub=h7d*p4d*p5d*h1d*sizeof(double);
	size_v2sub=h3d*h2d*p6d*h7d*sizeof(double);
	cudaFuncSetCacheConfig(ma_sd_t_d1_5_kernel, cudaFuncCachePreferShared);
	nstreams=1;
	size_block_triplesx=size_triplesx/nstreams;
	size_el_block_triplesx=size_block_triplesx/sizeof(double);
	t2sub_d=(double*)getGpuMem(size_t2sub);
	v2sub_d=(double*)getGpuMem(size_v2sub);
	streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
	assert(streams!= NULL);
	for(i=0;i<nstreams;++i) {
		CUDA_SAFE(cudaStreamCreate(&streams[i])) ;
	}
	CUDA_SAFE(cudaMemcpy(t2sub_d,t2sub,size_t2sub,cudaMemcpyHostToDevice));
	CUDA_SAFE(cudaMemcpy(v2sub_d,v2sub,size_v2sub,cudaMemcpyHostToDevice));
	h7ld_t2sub=1;
	p4ld_t2sub=h7d;
	p5ld_t2sub=p4d*h7d;
	h1ld_t2sub=p5d*p4d*h7d;
	h3ld_v2sub=1;
	h2ld_v2sub=h3d;
	p6ld_v2sub=h2d*h3d;
	h7ld_v2sub=p6d*h2d*h3d;
	h3ld_triplesx=1;
	h1ld_triplesx=h3d;
	h2ld_triplesx=h1d*h3d;
	p5ld_triplesx=h2d*h1d*h3d;
	p4ld_triplesx=p5d*h2d*h1d*h3d;
	p6ld_triplesx=p4d*p5d*h2d*h1d*h3d;
	size_t total_x = h3d*h2d*p6d*1;
	size_t total_y = p4d*p5d*h1d;
	dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
	for(i=0;i<nstreams;++i){
		ma_sd_t_d1_5_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(h1d,h2d,h3d,h7d,p4d,p5d,p6d,h7ld_t2sub,p4ld_t2sub,p5ld_t2sub,h1ld_t2sub,h3ld_v2sub,h2ld_v2sub,p6ld_v2sub,h7ld_v2sub,h3ld_triplesx,h1ld_triplesx,h2ld_triplesx,p5ld_triplesx,p4ld_triplesx,p6ld_triplesx,ma_t3_d,t2sub_d,v2sub_d,i,total_x,total_y);
		CHECK_ERR("Kernel execution failed");
	}
	cudaThreadSynchronize();
	for(i=0;i<nstreams;++i){
		cudaStreamDestroy(streams[i]);}
	freeGpuMem(t2sub_d);
	freeGpuMem(v2sub_d);
	free(streams);
}
#undef T1
#undef T2
#undef Tcomm
void ma_sd_t_d1_5_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* h7d, Integer* p4d, Integer* p5d, Integer* p6d, double *triplesx, double *t2sub, double *v2sub) 
{
	ma_sd_t_d1_5_cuda((int)*h1d,(int)*h2d,(int)*h3d,(int)*h7d,(int)*p4d,(int)*p5d,(int)*p6d,triplesx,t2sub,v2sub);
}
/*----------------------------------------------------------------------*
*triplesx[h1,h3,p5,p4,p6] -= t2sub[h7,p4,p5,h1] * v2sub[h3,p6,h7]
*----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void ma_sd_t_d1_6_kernel(size_t h1d,size_t h3d,size_t h7d,size_t p4d,size_t p5d,size_t p6d,
	size_t h7ld_t2sub,size_t p4ld_t2sub,size_t p5ld_t2sub,size_t h1ld_t2sub,size_t h3ld_v2sub,size_t p6ld_v2sub,size_t h7ld_v2sub,
	size_t h1ld_triplesx,size_t h3ld_triplesx,size_t p5ld_triplesx,size_t p4ld_triplesx,size_t p6ld_triplesx,
	double *triplesx_d, double *t2sub_d, double *v2sub_d,size_t unused_idx, size_t total_x, size_t total_y) 
{
	size_t h1_0,h1_1,h1_2,h1_3,h3_0,h3_1,h3_2,h3_3,h7,p4_0,p4_1,p4_2,p4_3,p5_0,p5_1,p5_2,p5_3,p6_0,p6_1,p6_2,p6_3;
	double a1,b1;
	double a2,b2;
	double a3,b3;
	double a4,b4;
	size_t in1_idxl,in2_idxl,h7l,h7T;
	__shared__ double t2sub_shm[4*T1][Tcomm];
	__shared__ double v2sub_shm[Tcomm][4*T2];
	size_t rest_x=blockIdx.x;
	size_t rest_y=blockIdx.y;
	size_t thread_x = T2*4 * rest_x + threadIdx.x;
	size_t thread_y = T1*4 * rest_y + threadIdx.y;
	in1_idxl=threadIdx.y;
	in2_idxl=threadIdx.x ;
	double tlocal1=0;
	double tlocal2=0;
	double tlocal3=0;
	double tlocal4=0;
	double tlocal5=0;
	double tlocal6=0;
	double tlocal7=0;
	double tlocal8=0;
	double tlocal9=0;
	double tlocal10=0;
	double tlocal11=0;
	double tlocal12=0;
	double tlocal13=0;
	double tlocal14=0;
	double tlocal15=0;
	double tlocal16=0;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
	h1_0=rest_y%h1d;
	rest_y=rest_y/h1d;
	h3_0=rest_x%h3d;
	rest_x=rest_x/h3d;
	p5_0=rest_y%p5d;
	rest_y=rest_y/p5d;
	p4_0=rest_y;
	p6_0=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
	h1_1=rest_y%h1d;
	rest_y=rest_y/h1d;
	h3_1=rest_x%h3d;
	rest_x=rest_x/h3d;
	p5_1=rest_y%p5d;
	rest_y=rest_y/p5d;
	p4_1=rest_y;
	p6_1=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
	h1_2=rest_y%h1d;
	rest_y=rest_y/h1d;
	h3_2=rest_x%h3d;
	rest_x=rest_x/h3d;
	p5_2=rest_y%p5d;
	rest_y=rest_y/p5d;
	p4_2=rest_y;
	p6_2=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
	h1_3=rest_y%h1d;
	rest_y=rest_y/h1d;
	h3_3=rest_x%h3d;
	rest_x=rest_x/h3d;
	p5_3=rest_y%p5d;
	rest_y=rest_y/p5d;
	p4_3=rest_y;
	p6_3=rest_x;
	size_t t2sub_d_off, v2sub_d_off;for(h7T=0;h7T<h7d;h7T+=Tcomm){size_t h7l_hi;
		h7l_hi = MIN(Tcomm+h7T,h7d)-h7T;
		t2sub_d_off=p4_0*p4ld_t2sub+p5_0*p5ld_t2sub+h1_0*h1ld_t2sub;
		v2sub_d_off=h3_0*h3ld_v2sub+p6_0*p6ld_v2sub;
		if(thread_y+T1*0<total_y)for(h7l=threadIdx.x;h7l<h7l_hi;h7l+=blockDim.x){
		h7=h7l+h7T;
		t2sub_shm[in1_idxl+T1*0][h7l] = t2sub_d[t2sub_d_off+h7*h7ld_t2sub];
		}
		if(thread_x+T1*0<total_x)for(h7l=threadIdx.y;h7l<h7l_hi;h7l+=blockDim.y){
		h7=h7l+h7T;
		v2sub_shm[h7l][in2_idxl+T1*0] = v2sub_d[v2sub_d_off+h7*h7ld_v2sub];
		}
		t2sub_d_off=p4_1*p4ld_t2sub+p5_1*p5ld_t2sub+h1_1*h1ld_t2sub;
		v2sub_d_off=h3_1*h3ld_v2sub+p6_1*p6ld_v2sub;
		if(thread_y+T1*1<total_y)for(h7l=threadIdx.x;h7l<h7l_hi;h7l+=blockDim.x){
		h7=h7l+h7T;
		t2sub_shm[in1_idxl+T1*1][h7l] = t2sub_d[t2sub_d_off+h7*h7ld_t2sub];
		}
		if(thread_x+T1*1<total_x)for(h7l=threadIdx.y;h7l<h7l_hi;h7l+=blockDim.y){
		h7=h7l+h7T;
		v2sub_shm[h7l][in2_idxl+T1*1] = v2sub_d[v2sub_d_off+h7*h7ld_v2sub];
		}
		t2sub_d_off=p4_2*p4ld_t2sub+p5_2*p5ld_t2sub+h1_2*h1ld_t2sub;
		v2sub_d_off=h3_2*h3ld_v2sub+p6_2*p6ld_v2sub;
		if(thread_y+T1*2<total_y)for(h7l=threadIdx.x;h7l<h7l_hi;h7l+=blockDim.x){
		h7=h7l+h7T;
		t2sub_shm[in1_idxl+T1*2][h7l] = t2sub_d[t2sub_d_off+h7*h7ld_t2sub];
		}
		if(thread_x+T1*2<total_x)for(h7l=threadIdx.y;h7l<h7l_hi;h7l+=blockDim.y){
		h7=h7l+h7T;
		v2sub_shm[h7l][in2_idxl+T1*2] = v2sub_d[v2sub_d_off+h7*h7ld_v2sub];
		}
		t2sub_d_off=p4_3*p4ld_t2sub+p5_3*p5ld_t2sub+h1_3*h1ld_t2sub;
		v2sub_d_off=h3_3*h3ld_v2sub+p6_3*p6ld_v2sub;
		if(thread_y+T1*3<total_y)for(h7l=threadIdx.x;h7l<h7l_hi;h7l+=blockDim.x){
		h7=h7l+h7T;
		t2sub_shm[in1_idxl+T1*3][h7l] = t2sub_d[t2sub_d_off+h7*h7ld_t2sub];
		}
		if(thread_x+T1*3<total_x)for(h7l=threadIdx.y;h7l<h7l_hi;h7l+=blockDim.y){
		h7=h7l+h7T;
		v2sub_shm[h7l][in2_idxl+T1*3] = v2sub_d[v2sub_d_off+h7*h7ld_v2sub];
		}
		__syncthreads();
		for(h7l=0;h7l<h7l_hi;++h7l){
		a1=t2sub_shm[in1_idxl+T1*0][h7l];
		a2=t2sub_shm[in1_idxl+T1*1][h7l];
		a3=t2sub_shm[in1_idxl+T1*2][h7l];
		a4=t2sub_shm[in1_idxl+T1*3][h7l];
		b1=v2sub_shm[h7l][in2_idxl+T2*0];
		b2=v2sub_shm[h7l][in2_idxl+T2*1];
		b3=v2sub_shm[h7l][in2_idxl+T2*2];
		b4=v2sub_shm[h7l][in2_idxl+T2*3];
		tlocal1+=a1*b1;
		tlocal2+=a2*b1;
		tlocal3+=a3*b1;
		tlocal4+=a4*b1;
		tlocal5+=a1*b2;
		tlocal6+=a2*b2;
		tlocal7+=a3*b2;
		tlocal8+=a4*b2;
		tlocal9+=a1*b3;
		tlocal10+=a2*b3;
		tlocal11+=a3*b3;
		tlocal12+=a4*b3;
		tlocal13+=a1*b4;
		tlocal14+=a2*b4;
		tlocal15+=a3*b4;
		tlocal16+=a4*b4;
		}
		__syncthreads();
	}
	if(thread_x+T1*0<total_x){
		if(thread_y+T2*3<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_0*h3ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_0*p6ld_triplesx]-=tlocal1;
		triplesx_d[h1_1*h1ld_triplesx+h3_0*h3ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx+p6_0*p6ld_triplesx]-=tlocal2;
		triplesx_d[h1_2*h1ld_triplesx+h3_0*h3ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx+p6_0*p6ld_triplesx]-=tlocal3;
		triplesx_d[h1_3*h1ld_triplesx+h3_0*h3ld_triplesx+p5_3*p5ld_triplesx+p4_3*p4ld_triplesx+p6_0*p6ld_triplesx]-=tlocal4;
		}
		else if(thread_y+T2*2<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_0*h3ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_0*p6ld_triplesx]-=tlocal1;
		triplesx_d[h1_1*h1ld_triplesx+h3_0*h3ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx+p6_0*p6ld_triplesx]-=tlocal2;
		triplesx_d[h1_2*h1ld_triplesx+h3_0*h3ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx+p6_0*p6ld_triplesx]-=tlocal3;
		}
		else if(thread_y+T2*1<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_0*h3ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_0*p6ld_triplesx]-=tlocal1;
		triplesx_d[h1_1*h1ld_triplesx+h3_0*h3ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx+p6_0*p6ld_triplesx]-=tlocal2;
		}
		else if(thread_y+T2*0<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_0*h3ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_0*p6ld_triplesx]-=tlocal1;
		}
	}
	if(thread_x+T1*1<total_x){
		if(thread_y+T2*3<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_1*h3ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_1*p6ld_triplesx]-=tlocal5;
		triplesx_d[h1_1*h1ld_triplesx+h3_1*h3ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx+p6_1*p6ld_triplesx]-=tlocal6;
		triplesx_d[h1_2*h1ld_triplesx+h3_1*h3ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx+p6_1*p6ld_triplesx]-=tlocal7;
		triplesx_d[h1_3*h1ld_triplesx+h3_1*h3ld_triplesx+p5_3*p5ld_triplesx+p4_3*p4ld_triplesx+p6_1*p6ld_triplesx]-=tlocal8;
		}
		else if(thread_y+T2*2<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_1*h3ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_1*p6ld_triplesx]-=tlocal5;
		triplesx_d[h1_1*h1ld_triplesx+h3_1*h3ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx+p6_1*p6ld_triplesx]-=tlocal6;
		triplesx_d[h1_2*h1ld_triplesx+h3_1*h3ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx+p6_1*p6ld_triplesx]-=tlocal7;
		}
		else if(thread_y+T2*1<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_1*h3ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_1*p6ld_triplesx]-=tlocal5;
		triplesx_d[h1_1*h1ld_triplesx+h3_1*h3ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx+p6_1*p6ld_triplesx]-=tlocal6;
		}
		else if(thread_y+T2*0<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_1*h3ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_1*p6ld_triplesx]-=tlocal5;
		}
	}
	if(thread_x+T1*2<total_x){
		if(thread_y+T2*3<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_2*h3ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_2*p6ld_triplesx]-=tlocal9;
		triplesx_d[h1_1*h1ld_triplesx+h3_2*h3ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx+p6_2*p6ld_triplesx]-=tlocal10;
		triplesx_d[h1_2*h1ld_triplesx+h3_2*h3ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx+p6_2*p6ld_triplesx]-=tlocal11;
		triplesx_d[h1_3*h1ld_triplesx+h3_2*h3ld_triplesx+p5_3*p5ld_triplesx+p4_3*p4ld_triplesx+p6_2*p6ld_triplesx]-=tlocal12;
		}
		else if(thread_y+T2*2<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_2*h3ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_2*p6ld_triplesx]-=tlocal9;
		triplesx_d[h1_1*h1ld_triplesx+h3_2*h3ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx+p6_2*p6ld_triplesx]-=tlocal10;
		triplesx_d[h1_2*h1ld_triplesx+h3_2*h3ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx+p6_2*p6ld_triplesx]-=tlocal11;
		}
		else if(thread_y+T2*1<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_2*h3ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_2*p6ld_triplesx]-=tlocal9;
		triplesx_d[h1_1*h1ld_triplesx+h3_2*h3ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx+p6_2*p6ld_triplesx]-=tlocal10;
		}
		else if(thread_y+T2*0<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_2*h3ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_2*p6ld_triplesx]-=tlocal9;
		}
	}
	if(thread_x+T1*3<total_x){
		if(thread_y+T2*3<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_3*h3ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_3*p6ld_triplesx]-=tlocal13;
		triplesx_d[h1_1*h1ld_triplesx+h3_3*h3ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx+p6_3*p6ld_triplesx]-=tlocal14;
		triplesx_d[h1_2*h1ld_triplesx+h3_3*h3ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx+p6_3*p6ld_triplesx]-=tlocal15;
		triplesx_d[h1_3*h1ld_triplesx+h3_3*h3ld_triplesx+p5_3*p5ld_triplesx+p4_3*p4ld_triplesx+p6_3*p6ld_triplesx]-=tlocal16;
		}
		else if(thread_y+T2*2<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_3*h3ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_3*p6ld_triplesx]-=tlocal13;
		triplesx_d[h1_1*h1ld_triplesx+h3_3*h3ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx+p6_3*p6ld_triplesx]-=tlocal14;
		triplesx_d[h1_2*h1ld_triplesx+h3_3*h3ld_triplesx+p5_2*p5ld_triplesx+p4_2*p4ld_triplesx+p6_3*p6ld_triplesx]-=tlocal15;
		}
		else if(thread_y+T2*1<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_3*h3ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_3*p6ld_triplesx]-=tlocal13;
		triplesx_d[h1_1*h1ld_triplesx+h3_3*h3ld_triplesx+p5_1*p5ld_triplesx+p4_1*p4ld_triplesx+p6_3*p6ld_triplesx]-=tlocal14;
		}
		else if(thread_y+T2*0<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_3*h3ld_triplesx+p5_0*p5ld_triplesx+p4_0*p4ld_triplesx+p6_3*p6ld_triplesx]-=tlocal13;
		}
	}
	__syncthreads();
}
void ma_sd_t_d1_6_cuda(size_t h1d, size_t h2d, size_t h3d, size_t h7d, size_t p4d, size_t p5d, size_t p6d, double *triplesx, double *t2sub, double *v2sub) 
{
	h3d=h3d*h2d;
	size_t h7ld_t2sub,p4ld_t2sub,p5ld_t2sub,h1ld_t2sub,h3ld_v2sub,p6ld_v2sub,h7ld_v2sub,h1ld_triplesx,h3ld_triplesx,p5ld_triplesx,p4ld_triplesx,p6ld_triplesx;
	size_t size_triplesx,size_block_triplesx,size_el_block_triplesx,size_t2sub,size_v2sub;
	cudaStream_t *streams;
	size_t nstreams,i;
	double *t2sub_d,*v2sub_d;
	size_triplesx=h1d*h3d*p5d*p4d*p6d*sizeof(double);
	size_t2sub=h7d*p4d*p5d*h1d*sizeof(double);
	size_v2sub=h3d*p6d*h7d*sizeof(double);
	cudaFuncSetCacheConfig(ma_sd_t_d1_6_kernel, cudaFuncCachePreferShared);
	nstreams=1;
	size_block_triplesx=size_triplesx/nstreams;
	size_el_block_triplesx=size_block_triplesx/sizeof(double);
	t2sub_d=(double*)getGpuMem(size_t2sub);
	v2sub_d=(double*)getGpuMem(size_v2sub);
	streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
	assert(streams!= NULL);
	for(i=0;i<nstreams;++i) {
		CUDA_SAFE(cudaStreamCreate(&streams[i])) ;
	}
	CUDA_SAFE(cudaMemcpy(t2sub_d,t2sub,size_t2sub,cudaMemcpyHostToDevice));
	CUDA_SAFE(cudaMemcpy(v2sub_d,v2sub,size_v2sub,cudaMemcpyHostToDevice));
	h7ld_t2sub=1;
	p4ld_t2sub=h7d;
	p5ld_t2sub=p4d*h7d;
	h1ld_t2sub=p5d*p4d*h7d;
	h3ld_v2sub=1;
	p6ld_v2sub=h3d;
	h7ld_v2sub=p6d*h3d;
	h1ld_triplesx=1;
	h3ld_triplesx=h1d;
	p5ld_triplesx=h3d*h1d;
	p4ld_triplesx=p5d*h3d*h1d;
	p6ld_triplesx=p4d*p5d*h3d*h1d;
	size_t total_x = h3d*p6d*1;
	size_t total_y = p4d*p5d*h1d;
	dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
	for(i=0;i<nstreams;++i){
		ma_sd_t_d1_6_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(h1d,h3d,h7d,p4d,p5d,p6d,h7ld_t2sub,p4ld_t2sub,p5ld_t2sub,h1ld_t2sub,h3ld_v2sub,p6ld_v2sub,h7ld_v2sub,h1ld_triplesx,h3ld_triplesx,p5ld_triplesx,p4ld_triplesx,p6ld_triplesx,ma_t3_d,t2sub_d,v2sub_d,i,total_x,total_y);
		CHECK_ERR("Kernel execution failed");
	}
	cudaThreadSynchronize();
	for(i=0;i<nstreams;++i){
		cudaStreamDestroy(streams[i]);}
	freeGpuMem(t2sub_d);
	freeGpuMem(v2sub_d);
	free(streams);
}
#undef T1
#undef T2
#undef Tcomm
void ma_sd_t_d1_6_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* h7d, Integer* p4d, Integer* p5d, Integer* p6d, double *triplesx, double *t2sub, double *v2sub) 
{
	ma_sd_t_d1_6_cuda((int)*h1d,(int)*h2d,(int)*h3d,(int)*h7d,(int)*p4d,(int)*p5d,(int)*p6d,triplesx,t2sub,v2sub);
}
/*----------------------------------------------------------------------*
*triplesx[h3,h1,p5,p6,p4] += t2sub[h7,p4,p5,h1] * v2sub[h3,p6,h7]
*----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void ma_sd_t_d1_7_kernel(size_t h1d,size_t h3d,size_t h7d,size_t p4d,size_t p5d,size_t p6d,
	size_t h7ld_t2sub,size_t p4ld_t2sub,size_t p5ld_t2sub,size_t h1ld_t2sub,size_t h3ld_v2sub,size_t p6ld_v2sub,size_t h7ld_v2sub,
	size_t h3ld_triplesx,size_t h1ld_triplesx,size_t p5ld_triplesx,size_t p6ld_triplesx,size_t p4ld_triplesx,
	double *triplesx_d, double *t2sub_d, double *v2sub_d,size_t unused_idx, size_t total_x, size_t total_y) 
{
	size_t h1_0,h1_1,h1_2,h1_3,h3_0,h3_1,h3_2,h3_3,h7,p4_0,p4_1,p4_2,p4_3,p5_0,p5_1,p5_2,p5_3,p6_0,p6_1,p6_2,p6_3;
	double a1,b1;
	double a2,b2;
	double a3,b3;
	double a4,b4;
	size_t in1_idxl,in2_idxl,h7l,h7T;
	__shared__ double t2sub_shm[4*T1][Tcomm];
	__shared__ double v2sub_shm[Tcomm][4*T2];
	size_t rest_x=blockIdx.x;
	size_t rest_y=blockIdx.y;
	size_t thread_x = T2*4 * rest_x + threadIdx.x;
	size_t thread_y = T1*4 * rest_y + threadIdx.y;
	in1_idxl=threadIdx.y;
	in2_idxl=threadIdx.x ;
	double tlocal1=0;
	double tlocal2=0;
	double tlocal3=0;
	double tlocal4=0;
	double tlocal5=0;
	double tlocal6=0;
	double tlocal7=0;
	double tlocal8=0;
	double tlocal9=0;
	double tlocal10=0;
	double tlocal11=0;
	double tlocal12=0;
	double tlocal13=0;
	double tlocal14=0;
	double tlocal15=0;
	double tlocal16=0;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
	h3_0=rest_x%h3d;
	rest_x=rest_x/h3d;
	h1_0=rest_y%h1d;
	rest_y=rest_y/h1d;
	p5_0=rest_y%p5d;
	rest_y=rest_y/p5d;
	p4_0=rest_y;
	p6_0=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
	h3_1=rest_x%h3d;
	rest_x=rest_x/h3d;
	h1_1=rest_y%h1d;
	rest_y=rest_y/h1d;
	p5_1=rest_y%p5d;
	rest_y=rest_y/p5d;
	p4_1=rest_y;
	p6_1=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
	h3_2=rest_x%h3d;
	rest_x=rest_x/h3d;
	h1_2=rest_y%h1d;
	rest_y=rest_y/h1d;
	p5_2=rest_y%p5d;
	rest_y=rest_y/p5d;
	p4_2=rest_y;
	p6_2=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
	h3_3=rest_x%h3d;
	rest_x=rest_x/h3d;
	h1_3=rest_y%h1d;
	rest_y=rest_y/h1d;
	p5_3=rest_y%p5d;
	rest_y=rest_y/p5d;
	p4_3=rest_y;
	p6_3=rest_x;
	size_t t2sub_d_off, v2sub_d_off;for(h7T=0;h7T<h7d;h7T+=Tcomm){size_t h7l_hi;
		h7l_hi = MIN(Tcomm+h7T,h7d)-h7T;
		t2sub_d_off=p4_0*p4ld_t2sub+p5_0*p5ld_t2sub+h1_0*h1ld_t2sub;
		v2sub_d_off=h3_0*h3ld_v2sub+p6_0*p6ld_v2sub;
		if(thread_y+T1*0<total_y)for(h7l=threadIdx.x;h7l<h7l_hi;h7l+=blockDim.x){
		h7=h7l+h7T;
		t2sub_shm[in1_idxl+T1*0][h7l] = t2sub_d[t2sub_d_off+h7*h7ld_t2sub];
		}
		if(thread_x+T1*0<total_x)for(h7l=threadIdx.y;h7l<h7l_hi;h7l+=blockDim.y){
		h7=h7l+h7T;
		v2sub_shm[h7l][in2_idxl+T1*0] = v2sub_d[v2sub_d_off+h7*h7ld_v2sub];
		}
		t2sub_d_off=p4_1*p4ld_t2sub+p5_1*p5ld_t2sub+h1_1*h1ld_t2sub;
		v2sub_d_off=h3_1*h3ld_v2sub+p6_1*p6ld_v2sub;
		if(thread_y+T1*1<total_y)for(h7l=threadIdx.x;h7l<h7l_hi;h7l+=blockDim.x){
		h7=h7l+h7T;
		t2sub_shm[in1_idxl+T1*1][h7l] = t2sub_d[t2sub_d_off+h7*h7ld_t2sub];
		}
		if(thread_x+T1*1<total_x)for(h7l=threadIdx.y;h7l<h7l_hi;h7l+=blockDim.y){
		h7=h7l+h7T;
		v2sub_shm[h7l][in2_idxl+T1*1] = v2sub_d[v2sub_d_off+h7*h7ld_v2sub];
		}
		t2sub_d_off=p4_2*p4ld_t2sub+p5_2*p5ld_t2sub+h1_2*h1ld_t2sub;
		v2sub_d_off=h3_2*h3ld_v2sub+p6_2*p6ld_v2sub;
		if(thread_y+T1*2<total_y)for(h7l=threadIdx.x;h7l<h7l_hi;h7l+=blockDim.x){
		h7=h7l+h7T;
		t2sub_shm[in1_idxl+T1*2][h7l] = t2sub_d[t2sub_d_off+h7*h7ld_t2sub];
		}
		if(thread_x+T1*2<total_x)for(h7l=threadIdx.y;h7l<h7l_hi;h7l+=blockDim.y){
		h7=h7l+h7T;
		v2sub_shm[h7l][in2_idxl+T1*2] = v2sub_d[v2sub_d_off+h7*h7ld_v2sub];
		}
		t2sub_d_off=p4_3*p4ld_t2sub+p5_3*p5ld_t2sub+h1_3*h1ld_t2sub;
		v2sub_d_off=h3_3*h3ld_v2sub+p6_3*p6ld_v2sub;
		if(thread_y+T1*3<total_y)for(h7l=threadIdx.x;h7l<h7l_hi;h7l+=blockDim.x){
		h7=h7l+h7T;
		t2sub_shm[in1_idxl+T1*3][h7l] = t2sub_d[t2sub_d_off+h7*h7ld_t2sub];
		}
		if(thread_x+T1*3<total_x)for(h7l=threadIdx.y;h7l<h7l_hi;h7l+=blockDim.y){
		h7=h7l+h7T;
		v2sub_shm[h7l][in2_idxl+T1*3] = v2sub_d[v2sub_d_off+h7*h7ld_v2sub];
		}
		__syncthreads();
		for(h7l=0;h7l<h7l_hi;++h7l){
		a1=t2sub_shm[in1_idxl+T1*0][h7l];
		a2=t2sub_shm[in1_idxl+T1*1][h7l];
		a3=t2sub_shm[in1_idxl+T1*2][h7l];
		a4=t2sub_shm[in1_idxl+T1*3][h7l];
		b1=v2sub_shm[h7l][in2_idxl+T2*0];
		b2=v2sub_shm[h7l][in2_idxl+T2*1];
		b3=v2sub_shm[h7l][in2_idxl+T2*2];
		b4=v2sub_shm[h7l][in2_idxl+T2*3];
		tlocal1+=a1*b1;
		tlocal2+=a2*b1;
		tlocal3+=a3*b1;
		tlocal4+=a4*b1;
		tlocal5+=a1*b2;
		tlocal6+=a2*b2;
		tlocal7+=a3*b2;
		tlocal8+=a4*b2;
		tlocal9+=a1*b3;
		tlocal10+=a2*b3;
		tlocal11+=a3*b3;
		tlocal12+=a4*b3;
		tlocal13+=a1*b4;
		tlocal14+=a2*b4;
		tlocal15+=a3*b4;
		tlocal16+=a4*b4;
		}
		__syncthreads();
	}
	if(thread_x+T1*0<total_x){
		if(thread_y+T2*3<total_y) {
		triplesx_d[h3_0*h3ld_triplesx+h1_0*h1ld_triplesx+p5_0*p5ld_triplesx+p6_0*p6ld_triplesx+p4_0*p4ld_triplesx]+=tlocal1;
		triplesx_d[h3_0*h3ld_triplesx+h1_1*h1ld_triplesx+p5_1*p5ld_triplesx+p6_0*p6ld_triplesx+p4_1*p4ld_triplesx]+=tlocal2;
		triplesx_d[h3_0*h3ld_triplesx+h1_2*h1ld_triplesx+p5_2*p5ld_triplesx+p6_0*p6ld_triplesx+p4_2*p4ld_triplesx]+=tlocal3;
		triplesx_d[h3_0*h3ld_triplesx+h1_3*h1ld_triplesx+p5_3*p5ld_triplesx+p6_0*p6ld_triplesx+p4_3*p4ld_triplesx]+=tlocal4;
		}
		else if(thread_y+T2*2<total_y) {
		triplesx_d[h3_0*h3ld_triplesx+h1_0*h1ld_triplesx+p5_0*p5ld_triplesx+p6_0*p6ld_triplesx+p4_0*p4ld_triplesx]+=tlocal1;
		triplesx_d[h3_0*h3ld_triplesx+h1_1*h1ld_triplesx+p5_1*p5ld_triplesx+p6_0*p6ld_triplesx+p4_1*p4ld_triplesx]+=tlocal2;
		triplesx_d[h3_0*h3ld_triplesx+h1_2*h1ld_triplesx+p5_2*p5ld_triplesx+p6_0*p6ld_triplesx+p4_2*p4ld_triplesx]+=tlocal3;
		}
		else if(thread_y+T2*1<total_y) {
		triplesx_d[h3_0*h3ld_triplesx+h1_0*h1ld_triplesx+p5_0*p5ld_triplesx+p6_0*p6ld_triplesx+p4_0*p4ld_triplesx]+=tlocal1;
		triplesx_d[h3_0*h3ld_triplesx+h1_1*h1ld_triplesx+p5_1*p5ld_triplesx+p6_0*p6ld_triplesx+p4_1*p4ld_triplesx]+=tlocal2;
		}
		else if(thread_y+T2*0<total_y) {
		triplesx_d[h3_0*h3ld_triplesx+h1_0*h1ld_triplesx+p5_0*p5ld_triplesx+p6_0*p6ld_triplesx+p4_0*p4ld_triplesx]+=tlocal1;
		}
	}
	if(thread_x+T1*1<total_x){
		if(thread_y+T2*3<total_y) {
		triplesx_d[h3_1*h3ld_triplesx+h1_0*h1ld_triplesx+p5_0*p5ld_triplesx+p6_1*p6ld_triplesx+p4_0*p4ld_triplesx]+=tlocal5;
		triplesx_d[h3_1*h3ld_triplesx+h1_1*h1ld_triplesx+p5_1*p5ld_triplesx+p6_1*p6ld_triplesx+p4_1*p4ld_triplesx]+=tlocal6;
		triplesx_d[h3_1*h3ld_triplesx+h1_2*h1ld_triplesx+p5_2*p5ld_triplesx+p6_1*p6ld_triplesx+p4_2*p4ld_triplesx]+=tlocal7;
		triplesx_d[h3_1*h3ld_triplesx+h1_3*h1ld_triplesx+p5_3*p5ld_triplesx+p6_1*p6ld_triplesx+p4_3*p4ld_triplesx]+=tlocal8;
		}
		else if(thread_y+T2*2<total_y) {
		triplesx_d[h3_1*h3ld_triplesx+h1_0*h1ld_triplesx+p5_0*p5ld_triplesx+p6_1*p6ld_triplesx+p4_0*p4ld_triplesx]+=tlocal5;
		triplesx_d[h3_1*h3ld_triplesx+h1_1*h1ld_triplesx+p5_1*p5ld_triplesx+p6_1*p6ld_triplesx+p4_1*p4ld_triplesx]+=tlocal6;
		triplesx_d[h3_1*h3ld_triplesx+h1_2*h1ld_triplesx+p5_2*p5ld_triplesx+p6_1*p6ld_triplesx+p4_2*p4ld_triplesx]+=tlocal7;
		}
		else if(thread_y+T2*1<total_y) {
		triplesx_d[h3_1*h3ld_triplesx+h1_0*h1ld_triplesx+p5_0*p5ld_triplesx+p6_1*p6ld_triplesx+p4_0*p4ld_triplesx]+=tlocal5;
		triplesx_d[h3_1*h3ld_triplesx+h1_1*h1ld_triplesx+p5_1*p5ld_triplesx+p6_1*p6ld_triplesx+p4_1*p4ld_triplesx]+=tlocal6;
		}
		else if(thread_y+T2*0<total_y) {
		triplesx_d[h3_1*h3ld_triplesx+h1_0*h1ld_triplesx+p5_0*p5ld_triplesx+p6_1*p6ld_triplesx+p4_0*p4ld_triplesx]+=tlocal5;
		}
	}
	if(thread_x+T1*2<total_x){
		if(thread_y+T2*3<total_y) {
		triplesx_d[h3_2*h3ld_triplesx+h1_0*h1ld_triplesx+p5_0*p5ld_triplesx+p6_2*p6ld_triplesx+p4_0*p4ld_triplesx]+=tlocal9;
		triplesx_d[h3_2*h3ld_triplesx+h1_1*h1ld_triplesx+p5_1*p5ld_triplesx+p6_2*p6ld_triplesx+p4_1*p4ld_triplesx]+=tlocal10;
		triplesx_d[h3_2*h3ld_triplesx+h1_2*h1ld_triplesx+p5_2*p5ld_triplesx+p6_2*p6ld_triplesx+p4_2*p4ld_triplesx]+=tlocal11;
		triplesx_d[h3_2*h3ld_triplesx+h1_3*h1ld_triplesx+p5_3*p5ld_triplesx+p6_2*p6ld_triplesx+p4_3*p4ld_triplesx]+=tlocal12;
		}
		else if(thread_y+T2*2<total_y) {
		triplesx_d[h3_2*h3ld_triplesx+h1_0*h1ld_triplesx+p5_0*p5ld_triplesx+p6_2*p6ld_triplesx+p4_0*p4ld_triplesx]+=tlocal9;
		triplesx_d[h3_2*h3ld_triplesx+h1_1*h1ld_triplesx+p5_1*p5ld_triplesx+p6_2*p6ld_triplesx+p4_1*p4ld_triplesx]+=tlocal10;
		triplesx_d[h3_2*h3ld_triplesx+h1_2*h1ld_triplesx+p5_2*p5ld_triplesx+p6_2*p6ld_triplesx+p4_2*p4ld_triplesx]+=tlocal11;
		}
		else if(thread_y+T2*1<total_y) {
		triplesx_d[h3_2*h3ld_triplesx+h1_0*h1ld_triplesx+p5_0*p5ld_triplesx+p6_2*p6ld_triplesx+p4_0*p4ld_triplesx]+=tlocal9;
		triplesx_d[h3_2*h3ld_triplesx+h1_1*h1ld_triplesx+p5_1*p5ld_triplesx+p6_2*p6ld_triplesx+p4_1*p4ld_triplesx]+=tlocal10;
		}
		else if(thread_y+T2*0<total_y) {
		triplesx_d[h3_2*h3ld_triplesx+h1_0*h1ld_triplesx+p5_0*p5ld_triplesx+p6_2*p6ld_triplesx+p4_0*p4ld_triplesx]+=tlocal9;
		}
	}
	if(thread_x+T1*3<total_x){
		if(thread_y+T2*3<total_y) {
		triplesx_d[h3_3*h3ld_triplesx+h1_0*h1ld_triplesx+p5_0*p5ld_triplesx+p6_3*p6ld_triplesx+p4_0*p4ld_triplesx]+=tlocal13;
		triplesx_d[h3_3*h3ld_triplesx+h1_1*h1ld_triplesx+p5_1*p5ld_triplesx+p6_3*p6ld_triplesx+p4_1*p4ld_triplesx]+=tlocal14;
		triplesx_d[h3_3*h3ld_triplesx+h1_2*h1ld_triplesx+p5_2*p5ld_triplesx+p6_3*p6ld_triplesx+p4_2*p4ld_triplesx]+=tlocal15;
		triplesx_d[h3_3*h3ld_triplesx+h1_3*h1ld_triplesx+p5_3*p5ld_triplesx+p6_3*p6ld_triplesx+p4_3*p4ld_triplesx]+=tlocal16;
		}
		else if(thread_y+T2*2<total_y) {
		triplesx_d[h3_3*h3ld_triplesx+h1_0*h1ld_triplesx+p5_0*p5ld_triplesx+p6_3*p6ld_triplesx+p4_0*p4ld_triplesx]+=tlocal13;
		triplesx_d[h3_3*h3ld_triplesx+h1_1*h1ld_triplesx+p5_1*p5ld_triplesx+p6_3*p6ld_triplesx+p4_1*p4ld_triplesx]+=tlocal14;
		triplesx_d[h3_3*h3ld_triplesx+h1_2*h1ld_triplesx+p5_2*p5ld_triplesx+p6_3*p6ld_triplesx+p4_2*p4ld_triplesx]+=tlocal15;
		}
		else if(thread_y+T2*1<total_y) {
		triplesx_d[h3_3*h3ld_triplesx+h1_0*h1ld_triplesx+p5_0*p5ld_triplesx+p6_3*p6ld_triplesx+p4_0*p4ld_triplesx]+=tlocal13;
		triplesx_d[h3_3*h3ld_triplesx+h1_1*h1ld_triplesx+p5_1*p5ld_triplesx+p6_3*p6ld_triplesx+p4_1*p4ld_triplesx]+=tlocal14;
		}
		else if(thread_y+T2*0<total_y) {
		triplesx_d[h3_3*h3ld_triplesx+h1_0*h1ld_triplesx+p5_0*p5ld_triplesx+p6_3*p6ld_triplesx+p4_0*p4ld_triplesx]+=tlocal13;
		}
	}
	__syncthreads();
}
void ma_sd_t_d1_7_cuda(size_t h1d, size_t h2d, size_t h3d, size_t h7d, size_t p4d, size_t p5d, size_t p6d, double *triplesx, double *t2sub, double *v2sub) 
{
	h3d=h3d*h2d;
	size_t h7ld_t2sub,p4ld_t2sub,p5ld_t2sub,h1ld_t2sub,h3ld_v2sub,p6ld_v2sub,h7ld_v2sub,h3ld_triplesx,h1ld_triplesx,p5ld_triplesx,p6ld_triplesx,p4ld_triplesx;
	size_t size_triplesx,size_block_triplesx,size_el_block_triplesx,size_t2sub,size_v2sub;
	cudaStream_t *streams;
	size_t nstreams,i;
	double *t2sub_d,*v2sub_d;
	size_triplesx=h3d*h1d*p5d*p6d*p4d*sizeof(double);
	size_t2sub=h7d*p4d*p5d*h1d*sizeof(double);
	size_v2sub=h3d*p6d*h7d*sizeof(double);
	cudaFuncSetCacheConfig(ma_sd_t_d1_7_kernel, cudaFuncCachePreferShared);
	nstreams=1;
	size_block_triplesx=size_triplesx/nstreams;
	size_el_block_triplesx=size_block_triplesx/sizeof(double);
	t2sub_d=(double*)getGpuMem(size_t2sub);
	v2sub_d=(double*)getGpuMem(size_v2sub);
	streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
	assert(streams!= NULL);
	for(i=0;i<nstreams;++i) {
		CUDA_SAFE(cudaStreamCreate(&streams[i])) ;
	}
	CUDA_SAFE(cudaMemcpy(t2sub_d,t2sub,size_t2sub,cudaMemcpyHostToDevice));
	CUDA_SAFE(cudaMemcpy(v2sub_d,v2sub,size_v2sub,cudaMemcpyHostToDevice));
	h7ld_t2sub=1;
	p4ld_t2sub=h7d;
	p5ld_t2sub=p4d*h7d;
	h1ld_t2sub=p5d*p4d*h7d;
	h3ld_v2sub=1;
	p6ld_v2sub=h3d;
	h7ld_v2sub=p6d*h3d;
	h3ld_triplesx=1;
	h1ld_triplesx=h3d;
	p5ld_triplesx=h1d*h3d;
	p6ld_triplesx=p5d*h1d*h3d;
	p4ld_triplesx=p6d*p5d*h1d*h3d;
	size_t total_x = h3d*p6d*1;
	size_t total_y = p4d*p5d*h1d;
	dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
	for(i=0;i<nstreams;++i){
		ma_sd_t_d1_7_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(h1d,h3d,h7d,p4d,p5d,p6d,h7ld_t2sub,p4ld_t2sub,p5ld_t2sub,h1ld_t2sub,h3ld_v2sub,p6ld_v2sub,h7ld_v2sub,h3ld_triplesx,h1ld_triplesx,p5ld_triplesx,p6ld_triplesx,p4ld_triplesx,ma_t3_d,t2sub_d,v2sub_d,i,total_x,total_y);
		CHECK_ERR("Kernel execution failed");
	}
	cudaThreadSynchronize();
	for(i=0;i<nstreams;++i){
		cudaStreamDestroy(streams[i]);}
	freeGpuMem(t2sub_d);
	freeGpuMem(v2sub_d);
	free(streams);
}
#undef T1
#undef T2
#undef Tcomm
void ma_sd_t_d1_7_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* h7d, Integer* p4d, Integer* p5d, Integer* p6d, double *triplesx, double *t2sub, double *v2sub) 
{
	ma_sd_t_d1_7_cuda((int)*h1d,(int)*h2d,(int)*h3d,(int)*h7d,(int)*p4d,(int)*p5d,(int)*p6d,triplesx,t2sub,v2sub);
}
/*----------------------------------------------------------------------*
*triplesx[h3,h1,h2,p5,p6,p4] -= t2sub[h7,p4,p5,h1] * v2sub[h3,h2,p6,h7]
*----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void ma_sd_t_d1_8_kernel(size_t h1d,size_t h2d,size_t h3d,size_t h7d,size_t p4d,size_t p5d,size_t p6d,
	size_t h7ld_t2sub,size_t p4ld_t2sub,size_t p5ld_t2sub,size_t h1ld_t2sub,size_t h3ld_v2sub,size_t h2ld_v2sub,size_t p6ld_v2sub,size_t h7ld_v2sub,
	size_t h3ld_triplesx,size_t h1ld_triplesx,size_t h2ld_triplesx,size_t p5ld_triplesx,size_t p6ld_triplesx,size_t p4ld_triplesx,
	double *triplesx_d, double *t2sub_d, double *v2sub_d,size_t unused_idx, size_t total_x, size_t total_y) 
{
	size_t h1_0,h1_1,h1_2,h1_3,h2_0,h2_1,h2_2,h2_3,h3_0,h3_1,h3_2,h3_3,h7,p4_0,p4_1,p4_2,p4_3,p5_0,p5_1,p5_2,p5_3,p6_0,p6_1,p6_2,p6_3;
	double a1,b1;
	double a2,b2;
	double a3,b3;
	double a4,b4;
	size_t in1_idxl,in2_idxl,h7l,h7T;
	__shared__ double t2sub_shm[4*T1][Tcomm];
	__shared__ double v2sub_shm[Tcomm][4*T2];
	size_t rest_x=blockIdx.x;
	size_t rest_y=blockIdx.y;
	size_t thread_x = T2*4 * rest_x + threadIdx.x;
	size_t thread_y = T1*4 * rest_y + threadIdx.y;
	in1_idxl=threadIdx.y;
	in2_idxl=threadIdx.x ;
	double tlocal1=0;
	double tlocal2=0;
	double tlocal3=0;
	double tlocal4=0;
	double tlocal5=0;
	double tlocal6=0;
	double tlocal7=0;
	double tlocal8=0;
	double tlocal9=0;
	double tlocal10=0;
	double tlocal11=0;
	double tlocal12=0;
	double tlocal13=0;
	double tlocal14=0;
	double tlocal15=0;
	double tlocal16=0;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
	h3_0=rest_x%h3d;
	rest_x=rest_x/h3d;
	h1_0=rest_y%h1d;
	rest_y=rest_y/h1d;
	h2_0=rest_x%h2d;
	rest_x=rest_x/h2d;
	p5_0=rest_y%p5d;
	rest_y=rest_y/p5d;
	p4_0=rest_y;
	p6_0=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
	h3_1=rest_x%h3d;
	rest_x=rest_x/h3d;
	h1_1=rest_y%h1d;
	rest_y=rest_y/h1d;
	h2_1=rest_x%h2d;
	rest_x=rest_x/h2d;
	p5_1=rest_y%p5d;
	rest_y=rest_y/p5d;
	p4_1=rest_y;
	p6_1=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
	h3_2=rest_x%h3d;
	rest_x=rest_x/h3d;
	h1_2=rest_y%h1d;
	rest_y=rest_y/h1d;
	h2_2=rest_x%h2d;
	rest_x=rest_x/h2d;
	p5_2=rest_y%p5d;
	rest_y=rest_y/p5d;
	p4_2=rest_y;
	p6_2=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
	h3_3=rest_x%h3d;
	rest_x=rest_x/h3d;
	h1_3=rest_y%h1d;
	rest_y=rest_y/h1d;
	h2_3=rest_x%h2d;
	rest_x=rest_x/h2d;
	p5_3=rest_y%p5d;
	rest_y=rest_y/p5d;
	p4_3=rest_y;
	p6_3=rest_x;
	size_t t2sub_d_off, v2sub_d_off;for(h7T=0;h7T<h7d;h7T+=Tcomm){size_t h7l_hi;
		h7l_hi = MIN(Tcomm+h7T,h7d)-h7T;
		t2sub_d_off=p4_0*p4ld_t2sub+p5_0*p5ld_t2sub+h1_0*h1ld_t2sub;
		v2sub_d_off=h3_0*h3ld_v2sub+h2_0*h2ld_v2sub+p6_0*p6ld_v2sub;
		if(thread_y+T1*0<total_y)for(h7l=threadIdx.x;h7l<h7l_hi;h7l+=blockDim.x){
		h7=h7l+h7T;
		t2sub_shm[in1_idxl+T1*0][h7l] = t2sub_d[t2sub_d_off+h7*h7ld_t2sub];
		}
		if(thread_x+T1*0<total_x)for(h7l=threadIdx.y;h7l<h7l_hi;h7l+=blockDim.y){
		h7=h7l+h7T;
		v2sub_shm[h7l][in2_idxl+T1*0] = v2sub_d[v2sub_d_off+h7*h7ld_v2sub];
		}
		t2sub_d_off=p4_1*p4ld_t2sub+p5_1*p5ld_t2sub+h1_1*h1ld_t2sub;
		v2sub_d_off=h3_1*h3ld_v2sub+h2_1*h2ld_v2sub+p6_1*p6ld_v2sub;
		if(thread_y+T1*1<total_y)for(h7l=threadIdx.x;h7l<h7l_hi;h7l+=blockDim.x){
		h7=h7l+h7T;
		t2sub_shm[in1_idxl+T1*1][h7l] = t2sub_d[t2sub_d_off+h7*h7ld_t2sub];
		}
		if(thread_x+T1*1<total_x)for(h7l=threadIdx.y;h7l<h7l_hi;h7l+=blockDim.y){
		h7=h7l+h7T;
		v2sub_shm[h7l][in2_idxl+T1*1] = v2sub_d[v2sub_d_off+h7*h7ld_v2sub];
		}
		t2sub_d_off=p4_2*p4ld_t2sub+p5_2*p5ld_t2sub+h1_2*h1ld_t2sub;
		v2sub_d_off=h3_2*h3ld_v2sub+h2_2*h2ld_v2sub+p6_2*p6ld_v2sub;
		if(thread_y+T1*2<total_y)for(h7l=threadIdx.x;h7l<h7l_hi;h7l+=blockDim.x){
		h7=h7l+h7T;
		t2sub_shm[in1_idxl+T1*2][h7l] = t2sub_d[t2sub_d_off+h7*h7ld_t2sub];
		}
		if(thread_x+T1*2<total_x)for(h7l=threadIdx.y;h7l<h7l_hi;h7l+=blockDim.y){
		h7=h7l+h7T;
		v2sub_shm[h7l][in2_idxl+T1*2] = v2sub_d[v2sub_d_off+h7*h7ld_v2sub];
		}
		t2sub_d_off=p4_3*p4ld_t2sub+p5_3*p5ld_t2sub+h1_3*h1ld_t2sub;
		v2sub_d_off=h3_3*h3ld_v2sub+h2_3*h2ld_v2sub+p6_3*p6ld_v2sub;
		if(thread_y+T1*3<total_y)for(h7l=threadIdx.x;h7l<h7l_hi;h7l+=blockDim.x){
		h7=h7l+h7T;
		t2sub_shm[in1_idxl+T1*3][h7l] = t2sub_d[t2sub_d_off+h7*h7ld_t2sub];
		}
		if(thread_x+T1*3<total_x)for(h7l=threadIdx.y;h7l<h7l_hi;h7l+=blockDim.y){
		h7=h7l+h7T;
		v2sub_shm[h7l][in2_idxl+T1*3] = v2sub_d[v2sub_d_off+h7*h7ld_v2sub];
		}
		__syncthreads();
		for(h7l=0;h7l<h7l_hi;++h7l){
		a1=t2sub_shm[in1_idxl+T1*0][h7l];
		a2=t2sub_shm[in1_idxl+T1*1][h7l];
		a3=t2sub_shm[in1_idxl+T1*2][h7l];
		a4=t2sub_shm[in1_idxl+T1*3][h7l];
		b1=v2sub_shm[h7l][in2_idxl+T2*0];
		b2=v2sub_shm[h7l][in2_idxl+T2*1];
		b3=v2sub_shm[h7l][in2_idxl+T2*2];
		b4=v2sub_shm[h7l][in2_idxl+T2*3];
		tlocal1+=a1*b1;
		tlocal2+=a2*b1;
		tlocal3+=a3*b1;
		tlocal4+=a4*b1;
		tlocal5+=a1*b2;
		tlocal6+=a2*b2;
		tlocal7+=a3*b2;
		tlocal8+=a4*b2;
		tlocal9+=a1*b3;
		tlocal10+=a2*b3;
		tlocal11+=a3*b3;
		tlocal12+=a4*b3;
		tlocal13+=a1*b4;
		tlocal14+=a2*b4;
		tlocal15+=a3*b4;
		tlocal16+=a4*b4;
		}
		__syncthreads();
	}
	if(thread_x+T1*0<total_x){
		if(thread_y+T2*3<total_y) {
		triplesx_d[h3_0*h3ld_triplesx+h1_0*h1ld_triplesx+h2_0*h2ld_triplesx+p5_0*p5ld_triplesx+p6_0*p6ld_triplesx+p4_0*p4ld_triplesx]-=tlocal1;
		triplesx_d[h3_0*h3ld_triplesx+h1_1*h1ld_triplesx+h2_0*h2ld_triplesx+p5_1*p5ld_triplesx+p6_0*p6ld_triplesx+p4_1*p4ld_triplesx]-=tlocal2;
		triplesx_d[h3_0*h3ld_triplesx+h1_2*h1ld_triplesx+h2_0*h2ld_triplesx+p5_2*p5ld_triplesx+p6_0*p6ld_triplesx+p4_2*p4ld_triplesx]-=tlocal3;
		triplesx_d[h3_0*h3ld_triplesx+h1_3*h1ld_triplesx+h2_0*h2ld_triplesx+p5_3*p5ld_triplesx+p6_0*p6ld_triplesx+p4_3*p4ld_triplesx]-=tlocal4;
		}
		else if(thread_y+T2*2<total_y) {
		triplesx_d[h3_0*h3ld_triplesx+h1_0*h1ld_triplesx+h2_0*h2ld_triplesx+p5_0*p5ld_triplesx+p6_0*p6ld_triplesx+p4_0*p4ld_triplesx]-=tlocal1;
		triplesx_d[h3_0*h3ld_triplesx+h1_1*h1ld_triplesx+h2_0*h2ld_triplesx+p5_1*p5ld_triplesx+p6_0*p6ld_triplesx+p4_1*p4ld_triplesx]-=tlocal2;
		triplesx_d[h3_0*h3ld_triplesx+h1_2*h1ld_triplesx+h2_0*h2ld_triplesx+p5_2*p5ld_triplesx+p6_0*p6ld_triplesx+p4_2*p4ld_triplesx]-=tlocal3;
		}
		else if(thread_y+T2*1<total_y) {
		triplesx_d[h3_0*h3ld_triplesx+h1_0*h1ld_triplesx+h2_0*h2ld_triplesx+p5_0*p5ld_triplesx+p6_0*p6ld_triplesx+p4_0*p4ld_triplesx]-=tlocal1;
		triplesx_d[h3_0*h3ld_triplesx+h1_1*h1ld_triplesx+h2_0*h2ld_triplesx+p5_1*p5ld_triplesx+p6_0*p6ld_triplesx+p4_1*p4ld_triplesx]-=tlocal2;
		}
		else if(thread_y+T2*0<total_y) {
		triplesx_d[h3_0*h3ld_triplesx+h1_0*h1ld_triplesx+h2_0*h2ld_triplesx+p5_0*p5ld_triplesx+p6_0*p6ld_triplesx+p4_0*p4ld_triplesx]-=tlocal1;
		}
	}
	if(thread_x+T1*1<total_x){
		if(thread_y+T2*3<total_y) {
		triplesx_d[h3_1*h3ld_triplesx+h1_0*h1ld_triplesx+h2_1*h2ld_triplesx+p5_0*p5ld_triplesx+p6_1*p6ld_triplesx+p4_0*p4ld_triplesx]-=tlocal5;
		triplesx_d[h3_1*h3ld_triplesx+h1_1*h1ld_triplesx+h2_1*h2ld_triplesx+p5_1*p5ld_triplesx+p6_1*p6ld_triplesx+p4_1*p4ld_triplesx]-=tlocal6;
		triplesx_d[h3_1*h3ld_triplesx+h1_2*h1ld_triplesx+h2_1*h2ld_triplesx+p5_2*p5ld_triplesx+p6_1*p6ld_triplesx+p4_2*p4ld_triplesx]-=tlocal7;
		triplesx_d[h3_1*h3ld_triplesx+h1_3*h1ld_triplesx+h2_1*h2ld_triplesx+p5_3*p5ld_triplesx+p6_1*p6ld_triplesx+p4_3*p4ld_triplesx]-=tlocal8;
		}
		else if(thread_y+T2*2<total_y) {
		triplesx_d[h3_1*h3ld_triplesx+h1_0*h1ld_triplesx+h2_1*h2ld_triplesx+p5_0*p5ld_triplesx+p6_1*p6ld_triplesx+p4_0*p4ld_triplesx]-=tlocal5;
		triplesx_d[h3_1*h3ld_triplesx+h1_1*h1ld_triplesx+h2_1*h2ld_triplesx+p5_1*p5ld_triplesx+p6_1*p6ld_triplesx+p4_1*p4ld_triplesx]-=tlocal6;
		triplesx_d[h3_1*h3ld_triplesx+h1_2*h1ld_triplesx+h2_1*h2ld_triplesx+p5_2*p5ld_triplesx+p6_1*p6ld_triplesx+p4_2*p4ld_triplesx]-=tlocal7;
		}
		else if(thread_y+T2*1<total_y) {
		triplesx_d[h3_1*h3ld_triplesx+h1_0*h1ld_triplesx+h2_1*h2ld_triplesx+p5_0*p5ld_triplesx+p6_1*p6ld_triplesx+p4_0*p4ld_triplesx]-=tlocal5;
		triplesx_d[h3_1*h3ld_triplesx+h1_1*h1ld_triplesx+h2_1*h2ld_triplesx+p5_1*p5ld_triplesx+p6_1*p6ld_triplesx+p4_1*p4ld_triplesx]-=tlocal6;
		}
		else if(thread_y+T2*0<total_y) {
		triplesx_d[h3_1*h3ld_triplesx+h1_0*h1ld_triplesx+h2_1*h2ld_triplesx+p5_0*p5ld_triplesx+p6_1*p6ld_triplesx+p4_0*p4ld_triplesx]-=tlocal5;
		}
	}
	if(thread_x+T1*2<total_x){
		if(thread_y+T2*3<total_y) {
		triplesx_d[h3_2*h3ld_triplesx+h1_0*h1ld_triplesx+h2_2*h2ld_triplesx+p5_0*p5ld_triplesx+p6_2*p6ld_triplesx+p4_0*p4ld_triplesx]-=tlocal9;
		triplesx_d[h3_2*h3ld_triplesx+h1_1*h1ld_triplesx+h2_2*h2ld_triplesx+p5_1*p5ld_triplesx+p6_2*p6ld_triplesx+p4_1*p4ld_triplesx]-=tlocal10;
		triplesx_d[h3_2*h3ld_triplesx+h1_2*h1ld_triplesx+h2_2*h2ld_triplesx+p5_2*p5ld_triplesx+p6_2*p6ld_triplesx+p4_2*p4ld_triplesx]-=tlocal11;
		triplesx_d[h3_2*h3ld_triplesx+h1_3*h1ld_triplesx+h2_2*h2ld_triplesx+p5_3*p5ld_triplesx+p6_2*p6ld_triplesx+p4_3*p4ld_triplesx]-=tlocal12;
		}
		else if(thread_y+T2*2<total_y) {
		triplesx_d[h3_2*h3ld_triplesx+h1_0*h1ld_triplesx+h2_2*h2ld_triplesx+p5_0*p5ld_triplesx+p6_2*p6ld_triplesx+p4_0*p4ld_triplesx]-=tlocal9;
		triplesx_d[h3_2*h3ld_triplesx+h1_1*h1ld_triplesx+h2_2*h2ld_triplesx+p5_1*p5ld_triplesx+p6_2*p6ld_triplesx+p4_1*p4ld_triplesx]-=tlocal10;
		triplesx_d[h3_2*h3ld_triplesx+h1_2*h1ld_triplesx+h2_2*h2ld_triplesx+p5_2*p5ld_triplesx+p6_2*p6ld_triplesx+p4_2*p4ld_triplesx]-=tlocal11;
		}
		else if(thread_y+T2*1<total_y) {
		triplesx_d[h3_2*h3ld_triplesx+h1_0*h1ld_triplesx+h2_2*h2ld_triplesx+p5_0*p5ld_triplesx+p6_2*p6ld_triplesx+p4_0*p4ld_triplesx]-=tlocal9;
		triplesx_d[h3_2*h3ld_triplesx+h1_1*h1ld_triplesx+h2_2*h2ld_triplesx+p5_1*p5ld_triplesx+p6_2*p6ld_triplesx+p4_1*p4ld_triplesx]-=tlocal10;
		}
		else if(thread_y+T2*0<total_y) {
		triplesx_d[h3_2*h3ld_triplesx+h1_0*h1ld_triplesx+h2_2*h2ld_triplesx+p5_0*p5ld_triplesx+p6_2*p6ld_triplesx+p4_0*p4ld_triplesx]-=tlocal9;
		}
	}
	if(thread_x+T1*3<total_x){
		if(thread_y+T2*3<total_y) {
		triplesx_d[h3_3*h3ld_triplesx+h1_0*h1ld_triplesx+h2_3*h2ld_triplesx+p5_0*p5ld_triplesx+p6_3*p6ld_triplesx+p4_0*p4ld_triplesx]-=tlocal13;
		triplesx_d[h3_3*h3ld_triplesx+h1_1*h1ld_triplesx+h2_3*h2ld_triplesx+p5_1*p5ld_triplesx+p6_3*p6ld_triplesx+p4_1*p4ld_triplesx]-=tlocal14;
		triplesx_d[h3_3*h3ld_triplesx+h1_2*h1ld_triplesx+h2_3*h2ld_triplesx+p5_2*p5ld_triplesx+p6_3*p6ld_triplesx+p4_2*p4ld_triplesx]-=tlocal15;
		triplesx_d[h3_3*h3ld_triplesx+h1_3*h1ld_triplesx+h2_3*h2ld_triplesx+p5_3*p5ld_triplesx+p6_3*p6ld_triplesx+p4_3*p4ld_triplesx]-=tlocal16;
		}
		else if(thread_y+T2*2<total_y) {
		triplesx_d[h3_3*h3ld_triplesx+h1_0*h1ld_triplesx+h2_3*h2ld_triplesx+p5_0*p5ld_triplesx+p6_3*p6ld_triplesx+p4_0*p4ld_triplesx]-=tlocal13;
		triplesx_d[h3_3*h3ld_triplesx+h1_1*h1ld_triplesx+h2_3*h2ld_triplesx+p5_1*p5ld_triplesx+p6_3*p6ld_triplesx+p4_1*p4ld_triplesx]-=tlocal14;
		triplesx_d[h3_3*h3ld_triplesx+h1_2*h1ld_triplesx+h2_3*h2ld_triplesx+p5_2*p5ld_triplesx+p6_3*p6ld_triplesx+p4_2*p4ld_triplesx]-=tlocal15;
		}
		else if(thread_y+T2*1<total_y) {
		triplesx_d[h3_3*h3ld_triplesx+h1_0*h1ld_triplesx+h2_3*h2ld_triplesx+p5_0*p5ld_triplesx+p6_3*p6ld_triplesx+p4_0*p4ld_triplesx]-=tlocal13;
		triplesx_d[h3_3*h3ld_triplesx+h1_1*h1ld_triplesx+h2_3*h2ld_triplesx+p5_1*p5ld_triplesx+p6_3*p6ld_triplesx+p4_1*p4ld_triplesx]-=tlocal14;
		}
		else if(thread_y+T2*0<total_y) {
		triplesx_d[h3_3*h3ld_triplesx+h1_0*h1ld_triplesx+h2_3*h2ld_triplesx+p5_0*p5ld_triplesx+p6_3*p6ld_triplesx+p4_0*p4ld_triplesx]-=tlocal13;
		}
	}
	__syncthreads();
}

void ma_sd_t_d1_8_cuda(size_t h1d, size_t h2d, size_t h3d, size_t h7d, size_t p4d, size_t p5d, size_t p6d, double *triplesx, double *t2sub, double *v2sub) 
{
	size_t h7ld_t2sub,p4ld_t2sub,p5ld_t2sub,h1ld_t2sub,h3ld_v2sub,h2ld_v2sub,p6ld_v2sub,h7ld_v2sub,h3ld_triplesx,h1ld_triplesx,h2ld_triplesx,p5ld_triplesx,p6ld_triplesx,p4ld_triplesx;
	size_t size_triplesx,size_block_triplesx,size_el_block_triplesx,size_t2sub,size_v2sub;
	cudaStream_t *streams;
	size_t nstreams,i;
	double *t2sub_d,*v2sub_d;
	size_triplesx=h3d*h1d*h2d*p5d*p6d*p4d*sizeof(double);
	size_t2sub=h7d*p4d*p5d*h1d*sizeof(double);
	size_v2sub=h3d*h2d*p6d*h7d*sizeof(double);
	cudaFuncSetCacheConfig(ma_sd_t_d1_8_kernel, cudaFuncCachePreferShared);
	nstreams=1;
	size_block_triplesx=size_triplesx/nstreams;
	size_el_block_triplesx=size_block_triplesx/sizeof(double);
	t2sub_d=(double*)getGpuMem(size_t2sub);
	v2sub_d=(double*)getGpuMem(size_v2sub);
	streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
	assert(streams!= NULL);
	for(i=0;i<nstreams;++i) {
		CUDA_SAFE(cudaStreamCreate(&streams[i])) ;
	}
	CUDA_SAFE(cudaMemcpy(t2sub_d,t2sub,size_t2sub,cudaMemcpyHostToDevice));
	CUDA_SAFE(cudaMemcpy(v2sub_d,v2sub,size_v2sub,cudaMemcpyHostToDevice));
	h7ld_t2sub=1;
	p4ld_t2sub=h7d;
	p5ld_t2sub=p4d*h7d;
	h1ld_t2sub=p5d*p4d*h7d;
	h3ld_v2sub=1;
	h2ld_v2sub=h3d;
	p6ld_v2sub=h2d*h3d;
	h7ld_v2sub=p6d*h2d*h3d;
	h3ld_triplesx=1;
	h1ld_triplesx=h3d;
	h2ld_triplesx=h1d*h3d;
	p5ld_triplesx=h2d*h1d*h3d;
	p6ld_triplesx=p5d*h2d*h1d*h3d;
	p4ld_triplesx=p6d*p5d*h2d*h1d*h3d;
	size_t total_x = h3d*h2d*p6d*1;
	size_t total_y = p4d*p5d*h1d;
	dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
	for(i=0;i<nstreams;++i){
		ma_sd_t_d1_8_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(h1d,h2d,h3d,h7d,p4d,p5d,p6d,h7ld_t2sub,p4ld_t2sub,p5ld_t2sub,h1ld_t2sub,h3ld_v2sub,h2ld_v2sub,p6ld_v2sub,h7ld_v2sub,h3ld_triplesx,h1ld_triplesx,h2ld_triplesx,p5ld_triplesx,p6ld_triplesx,p4ld_triplesx,ma_t3_d,t2sub_d,v2sub_d,i,total_x,total_y);
		CHECK_ERR("Kernel execution failed");
	}
	cudaThreadSynchronize();
	for(i=0;i<nstreams;++i){
		cudaStreamDestroy(streams[i]);}
	freeGpuMem(t2sub_d);
	freeGpuMem(v2sub_d);
	free(streams);
}
#undef T1
#undef T2
#undef Tcomm
void ma_sd_t_d1_8_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* h7d, Integer* p4d, Integer* p5d, Integer* p6d, double *triplesx, double *t2sub, double *v2sub) 
{
	ma_sd_t_d1_8_cuda((int)*h1d,(int)*h2d,(int)*h3d,(int)*h7d,(int)*p4d,(int)*p5d,(int)*p6d,triplesx,t2sub,v2sub);
}
/*----------------------------------------------------------------------*
*triplesx[h1,h3,p5,p6,p4] += t2sub[h7,p4,p5,h1] * v2sub[h3,p6,h7]
*----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void ma_sd_t_d1_9_kernel(size_t h1d,size_t h3d,size_t h7d,size_t p4d,size_t p5d,size_t p6d,
	size_t h7ld_t2sub,size_t p4ld_t2sub,size_t p5ld_t2sub,size_t h1ld_t2sub,size_t h3ld_v2sub,size_t p6ld_v2sub,size_t h7ld_v2sub,
	size_t h1ld_triplesx,size_t h3ld_triplesx,size_t p5ld_triplesx,size_t p6ld_triplesx,size_t p4ld_triplesx,
	double *triplesx_d, double *t2sub_d, double *v2sub_d,size_t unused_idx, size_t total_x, size_t total_y) 
{
	size_t h1_0,h1_1,h1_2,h1_3,h3_0,h3_1,h3_2,h3_3,h7,p4_0,p4_1,p4_2,p4_3,p5_0,p5_1,p5_2,p5_3,p6_0,p6_1,p6_2,p6_3;
	double a1,b1;
	double a2,b2;
	double a3,b3;
	double a4,b4;
	size_t in1_idxl,in2_idxl,h7l,h7T;
	__shared__ double t2sub_shm[4*T1][Tcomm];
	__shared__ double v2sub_shm[Tcomm][4*T2];
	size_t rest_x=blockIdx.x;
	size_t rest_y=blockIdx.y;
	size_t thread_x = T2*4 * rest_x + threadIdx.x;
	size_t thread_y = T1*4 * rest_y + threadIdx.y;
	in1_idxl=threadIdx.y;
	in2_idxl=threadIdx.x ;
	double tlocal1=0;
	double tlocal2=0;
	double tlocal3=0;
	double tlocal4=0;
	double tlocal5=0;
	double tlocal6=0;
	double tlocal7=0;
	double tlocal8=0;
	double tlocal9=0;
	double tlocal10=0;
	double tlocal11=0;
	double tlocal12=0;
	double tlocal13=0;
	double tlocal14=0;
	double tlocal15=0;
	double tlocal16=0;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
	h1_0=rest_y%h1d;
	rest_y=rest_y/h1d;
	h3_0=rest_x%h3d;
	rest_x=rest_x/h3d;
	p5_0=rest_y%p5d;
	rest_y=rest_y/p5d;
	p4_0=rest_y;
	p6_0=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
	h1_1=rest_y%h1d;
	rest_y=rest_y/h1d;
	h3_1=rest_x%h3d;
	rest_x=rest_x/h3d;
	p5_1=rest_y%p5d;
	rest_y=rest_y/p5d;
	p4_1=rest_y;
	p6_1=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
	h1_2=rest_y%h1d;
	rest_y=rest_y/h1d;
	h3_2=rest_x%h3d;
	rest_x=rest_x/h3d;
	p5_2=rest_y%p5d;
	rest_y=rest_y/p5d;
	p4_2=rest_y;
	p6_2=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
	h1_3=rest_y%h1d;
	rest_y=rest_y/h1d;
	h3_3=rest_x%h3d;
	rest_x=rest_x/h3d;
	p5_3=rest_y%p5d;
	rest_y=rest_y/p5d;
	p4_3=rest_y;
	p6_3=rest_x;
	size_t t2sub_d_off, v2sub_d_off;for(h7T=0;h7T<h7d;h7T+=Tcomm){size_t h7l_hi;
		h7l_hi = MIN(Tcomm+h7T,h7d)-h7T;
		t2sub_d_off=p4_0*p4ld_t2sub+p5_0*p5ld_t2sub+h1_0*h1ld_t2sub;
		v2sub_d_off=h3_0*h3ld_v2sub+p6_0*p6ld_v2sub;
		if(thread_y+T1*0<total_y)for(h7l=threadIdx.x;h7l<h7l_hi;h7l+=blockDim.x){
		h7=h7l+h7T;
		t2sub_shm[in1_idxl+T1*0][h7l] = t2sub_d[t2sub_d_off+h7*h7ld_t2sub];
		}
		if(thread_x+T1*0<total_x)for(h7l=threadIdx.y;h7l<h7l_hi;h7l+=blockDim.y){
		h7=h7l+h7T;
		v2sub_shm[h7l][in2_idxl+T1*0] = v2sub_d[v2sub_d_off+h7*h7ld_v2sub];
		}
		t2sub_d_off=p4_1*p4ld_t2sub+p5_1*p5ld_t2sub+h1_1*h1ld_t2sub;
		v2sub_d_off=h3_1*h3ld_v2sub+p6_1*p6ld_v2sub;
		if(thread_y+T1*1<total_y)for(h7l=threadIdx.x;h7l<h7l_hi;h7l+=blockDim.x){
		h7=h7l+h7T;
		t2sub_shm[in1_idxl+T1*1][h7l] = t2sub_d[t2sub_d_off+h7*h7ld_t2sub];
		}
		if(thread_x+T1*1<total_x)for(h7l=threadIdx.y;h7l<h7l_hi;h7l+=blockDim.y){
		h7=h7l+h7T;
		v2sub_shm[h7l][in2_idxl+T1*1] = v2sub_d[v2sub_d_off+h7*h7ld_v2sub];
		}
		t2sub_d_off=p4_2*p4ld_t2sub+p5_2*p5ld_t2sub+h1_2*h1ld_t2sub;
		v2sub_d_off=h3_2*h3ld_v2sub+p6_2*p6ld_v2sub;
		if(thread_y+T1*2<total_y)for(h7l=threadIdx.x;h7l<h7l_hi;h7l+=blockDim.x){
		h7=h7l+h7T;
		t2sub_shm[in1_idxl+T1*2][h7l] = t2sub_d[t2sub_d_off+h7*h7ld_t2sub];
		}
		if(thread_x+T1*2<total_x)for(h7l=threadIdx.y;h7l<h7l_hi;h7l+=blockDim.y){
		h7=h7l+h7T;
		v2sub_shm[h7l][in2_idxl+T1*2] = v2sub_d[v2sub_d_off+h7*h7ld_v2sub];
		}
		t2sub_d_off=p4_3*p4ld_t2sub+p5_3*p5ld_t2sub+h1_3*h1ld_t2sub;
		v2sub_d_off=h3_3*h3ld_v2sub+p6_3*p6ld_v2sub;
		if(thread_y+T1*3<total_y)for(h7l=threadIdx.x;h7l<h7l_hi;h7l+=blockDim.x){
		h7=h7l+h7T;
		t2sub_shm[in1_idxl+T1*3][h7l] = t2sub_d[t2sub_d_off+h7*h7ld_t2sub];
		}
		if(thread_x+T1*3<total_x)for(h7l=threadIdx.y;h7l<h7l_hi;h7l+=blockDim.y){
		h7=h7l+h7T;
		v2sub_shm[h7l][in2_idxl+T1*3] = v2sub_d[v2sub_d_off+h7*h7ld_v2sub];
		}
		__syncthreads();
		for(h7l=0;h7l<h7l_hi;++h7l){
		a1=t2sub_shm[in1_idxl+T1*0][h7l];
		a2=t2sub_shm[in1_idxl+T1*1][h7l];
		a3=t2sub_shm[in1_idxl+T1*2][h7l];
		a4=t2sub_shm[in1_idxl+T1*3][h7l];
		b1=v2sub_shm[h7l][in2_idxl+T2*0];
		b2=v2sub_shm[h7l][in2_idxl+T2*1];
		b3=v2sub_shm[h7l][in2_idxl+T2*2];
		b4=v2sub_shm[h7l][in2_idxl+T2*3];
		tlocal1+=a1*b1;
		tlocal2+=a2*b1;
		tlocal3+=a3*b1;
		tlocal4+=a4*b1;
		tlocal5+=a1*b2;
		tlocal6+=a2*b2;
		tlocal7+=a3*b2;
		tlocal8+=a4*b2;
		tlocal9+=a1*b3;
		tlocal10+=a2*b3;
		tlocal11+=a3*b3;
		tlocal12+=a4*b3;
		tlocal13+=a1*b4;
		tlocal14+=a2*b4;
		tlocal15+=a3*b4;
		tlocal16+=a4*b4;
		}
		__syncthreads();
	}
	if(thread_x+T1*0<total_x){
		if(thread_y+T2*3<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_0*h3ld_triplesx+p5_0*p5ld_triplesx+p6_0*p6ld_triplesx+p4_0*p4ld_triplesx]+=tlocal1;
		triplesx_d[h1_1*h1ld_triplesx+h3_0*h3ld_triplesx+p5_1*p5ld_triplesx+p6_0*p6ld_triplesx+p4_1*p4ld_triplesx]+=tlocal2;
		triplesx_d[h1_2*h1ld_triplesx+h3_0*h3ld_triplesx+p5_2*p5ld_triplesx+p6_0*p6ld_triplesx+p4_2*p4ld_triplesx]+=tlocal3;
		triplesx_d[h1_3*h1ld_triplesx+h3_0*h3ld_triplesx+p5_3*p5ld_triplesx+p6_0*p6ld_triplesx+p4_3*p4ld_triplesx]+=tlocal4;
		}
		else if(thread_y+T2*2<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_0*h3ld_triplesx+p5_0*p5ld_triplesx+p6_0*p6ld_triplesx+p4_0*p4ld_triplesx]+=tlocal1;
		triplesx_d[h1_1*h1ld_triplesx+h3_0*h3ld_triplesx+p5_1*p5ld_triplesx+p6_0*p6ld_triplesx+p4_1*p4ld_triplesx]+=tlocal2;
		triplesx_d[h1_2*h1ld_triplesx+h3_0*h3ld_triplesx+p5_2*p5ld_triplesx+p6_0*p6ld_triplesx+p4_2*p4ld_triplesx]+=tlocal3;
		}
		else if(thread_y+T2*1<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_0*h3ld_triplesx+p5_0*p5ld_triplesx+p6_0*p6ld_triplesx+p4_0*p4ld_triplesx]+=tlocal1;
		triplesx_d[h1_1*h1ld_triplesx+h3_0*h3ld_triplesx+p5_1*p5ld_triplesx+p6_0*p6ld_triplesx+p4_1*p4ld_triplesx]+=tlocal2;
		}
		else if(thread_y+T2*0<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_0*h3ld_triplesx+p5_0*p5ld_triplesx+p6_0*p6ld_triplesx+p4_0*p4ld_triplesx]+=tlocal1;
		}
	}
	if(thread_x+T1*1<total_x){
		if(thread_y+T2*3<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_1*h3ld_triplesx+p5_0*p5ld_triplesx+p6_1*p6ld_triplesx+p4_0*p4ld_triplesx]+=tlocal5;
		triplesx_d[h1_1*h1ld_triplesx+h3_1*h3ld_triplesx+p5_1*p5ld_triplesx+p6_1*p6ld_triplesx+p4_1*p4ld_triplesx]+=tlocal6;
		triplesx_d[h1_2*h1ld_triplesx+h3_1*h3ld_triplesx+p5_2*p5ld_triplesx+p6_1*p6ld_triplesx+p4_2*p4ld_triplesx]+=tlocal7;
		triplesx_d[h1_3*h1ld_triplesx+h3_1*h3ld_triplesx+p5_3*p5ld_triplesx+p6_1*p6ld_triplesx+p4_3*p4ld_triplesx]+=tlocal8;
		}
		else if(thread_y+T2*2<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_1*h3ld_triplesx+p5_0*p5ld_triplesx+p6_1*p6ld_triplesx+p4_0*p4ld_triplesx]+=tlocal5;
		triplesx_d[h1_1*h1ld_triplesx+h3_1*h3ld_triplesx+p5_1*p5ld_triplesx+p6_1*p6ld_triplesx+p4_1*p4ld_triplesx]+=tlocal6;
		triplesx_d[h1_2*h1ld_triplesx+h3_1*h3ld_triplesx+p5_2*p5ld_triplesx+p6_1*p6ld_triplesx+p4_2*p4ld_triplesx]+=tlocal7;
		}
		else if(thread_y+T2*1<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_1*h3ld_triplesx+p5_0*p5ld_triplesx+p6_1*p6ld_triplesx+p4_0*p4ld_triplesx]+=tlocal5;
		triplesx_d[h1_1*h1ld_triplesx+h3_1*h3ld_triplesx+p5_1*p5ld_triplesx+p6_1*p6ld_triplesx+p4_1*p4ld_triplesx]+=tlocal6;
		}
		else if(thread_y+T2*0<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_1*h3ld_triplesx+p5_0*p5ld_triplesx+p6_1*p6ld_triplesx+p4_0*p4ld_triplesx]+=tlocal5;
		}
	}
	if(thread_x+T1*2<total_x){
		if(thread_y+T2*3<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_2*h3ld_triplesx+p5_0*p5ld_triplesx+p6_2*p6ld_triplesx+p4_0*p4ld_triplesx]+=tlocal9;
		triplesx_d[h1_1*h1ld_triplesx+h3_2*h3ld_triplesx+p5_1*p5ld_triplesx+p6_2*p6ld_triplesx+p4_1*p4ld_triplesx]+=tlocal10;
		triplesx_d[h1_2*h1ld_triplesx+h3_2*h3ld_triplesx+p5_2*p5ld_triplesx+p6_2*p6ld_triplesx+p4_2*p4ld_triplesx]+=tlocal11;
		triplesx_d[h1_3*h1ld_triplesx+h3_2*h3ld_triplesx+p5_3*p5ld_triplesx+p6_2*p6ld_triplesx+p4_3*p4ld_triplesx]+=tlocal12;
		}
		else if(thread_y+T2*2<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_2*h3ld_triplesx+p5_0*p5ld_triplesx+p6_2*p6ld_triplesx+p4_0*p4ld_triplesx]+=tlocal9;
		triplesx_d[h1_1*h1ld_triplesx+h3_2*h3ld_triplesx+p5_1*p5ld_triplesx+p6_2*p6ld_triplesx+p4_1*p4ld_triplesx]+=tlocal10;
		triplesx_d[h1_2*h1ld_triplesx+h3_2*h3ld_triplesx+p5_2*p5ld_triplesx+p6_2*p6ld_triplesx+p4_2*p4ld_triplesx]+=tlocal11;
		}
		else if(thread_y+T2*1<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_2*h3ld_triplesx+p5_0*p5ld_triplesx+p6_2*p6ld_triplesx+p4_0*p4ld_triplesx]+=tlocal9;
		triplesx_d[h1_1*h1ld_triplesx+h3_2*h3ld_triplesx+p5_1*p5ld_triplesx+p6_2*p6ld_triplesx+p4_1*p4ld_triplesx]+=tlocal10;
		}
		else if(thread_y+T2*0<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_2*h3ld_triplesx+p5_0*p5ld_triplesx+p6_2*p6ld_triplesx+p4_0*p4ld_triplesx]+=tlocal9;
		}
	}
	if(thread_x+T1*3<total_x){
		if(thread_y+T2*3<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_3*h3ld_triplesx+p5_0*p5ld_triplesx+p6_3*p6ld_triplesx+p4_0*p4ld_triplesx]+=tlocal13;
		triplesx_d[h1_1*h1ld_triplesx+h3_3*h3ld_triplesx+p5_1*p5ld_triplesx+p6_3*p6ld_triplesx+p4_1*p4ld_triplesx]+=tlocal14;
		triplesx_d[h1_2*h1ld_triplesx+h3_3*h3ld_triplesx+p5_2*p5ld_triplesx+p6_3*p6ld_triplesx+p4_2*p4ld_triplesx]+=tlocal15;
		triplesx_d[h1_3*h1ld_triplesx+h3_3*h3ld_triplesx+p5_3*p5ld_triplesx+p6_3*p6ld_triplesx+p4_3*p4ld_triplesx]+=tlocal16;
		}
		else if(thread_y+T2*2<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_3*h3ld_triplesx+p5_0*p5ld_triplesx+p6_3*p6ld_triplesx+p4_0*p4ld_triplesx]+=tlocal13;
		triplesx_d[h1_1*h1ld_triplesx+h3_3*h3ld_triplesx+p5_1*p5ld_triplesx+p6_3*p6ld_triplesx+p4_1*p4ld_triplesx]+=tlocal14;
		triplesx_d[h1_2*h1ld_triplesx+h3_3*h3ld_triplesx+p5_2*p5ld_triplesx+p6_3*p6ld_triplesx+p4_2*p4ld_triplesx]+=tlocal15;
		}
		else if(thread_y+T2*1<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_3*h3ld_triplesx+p5_0*p5ld_triplesx+p6_3*p6ld_triplesx+p4_0*p4ld_triplesx]+=tlocal13;
		triplesx_d[h1_1*h1ld_triplesx+h3_3*h3ld_triplesx+p5_1*p5ld_triplesx+p6_3*p6ld_triplesx+p4_1*p4ld_triplesx]+=tlocal14;
		}
		else if(thread_y+T2*0<total_y) {
		triplesx_d[h1_0*h1ld_triplesx+h3_3*h3ld_triplesx+p5_0*p5ld_triplesx+p6_3*p6ld_triplesx+p4_0*p4ld_triplesx]+=tlocal13;
		}
	}
	__syncthreads();
}
void ma_sd_t_d1_9_cuda(size_t h1d, size_t h2d, size_t h3d, size_t h7d, size_t p4d, size_t p5d, size_t p6d, double *triplesx, double *t2sub, double *v2sub) 
{
	h3d=h3d*h2d;
	size_t h7ld_t2sub,p4ld_t2sub,p5ld_t2sub,h1ld_t2sub,h3ld_v2sub,p6ld_v2sub,h7ld_v2sub,h1ld_triplesx,h3ld_triplesx,p5ld_triplesx,p6ld_triplesx,p4ld_triplesx;
	size_t size_triplesx,size_block_triplesx,size_el_block_triplesx,size_t2sub,size_v2sub;
	cudaStream_t *streams;
	size_t nstreams,i;
	double *t2sub_d,*v2sub_d;
	size_triplesx=h1d*h3d*p5d*p6d*p4d*sizeof(double);
	size_t2sub=h7d*p4d*p5d*h1d*sizeof(double);
	size_v2sub=h3d*p6d*h7d*sizeof(double);
	cudaFuncSetCacheConfig(ma_sd_t_d1_9_kernel, cudaFuncCachePreferShared);
	nstreams=1;
	size_block_triplesx=size_triplesx/nstreams;
	size_el_block_triplesx=size_block_triplesx/sizeof(double);
	t2sub_d=(double*)getGpuMem(size_t2sub);
	v2sub_d=(double*)getGpuMem(size_v2sub);
	streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
	assert(streams!= NULL);
	for(i=0;i<nstreams;++i) {
		CUDA_SAFE(cudaStreamCreate(&streams[i])) ;
	}
	CUDA_SAFE(cudaMemcpy(t2sub_d,t2sub,size_t2sub,cudaMemcpyHostToDevice));
	CUDA_SAFE(cudaMemcpy(v2sub_d,v2sub,size_v2sub,cudaMemcpyHostToDevice));
	h7ld_t2sub=1;
	p4ld_t2sub=h7d;
	p5ld_t2sub=p4d*h7d;
	h1ld_t2sub=p5d*p4d*h7d;
	h3ld_v2sub=1;
	p6ld_v2sub=h3d;
	h7ld_v2sub=p6d*h3d;
	h1ld_triplesx=1;
	h3ld_triplesx=h1d;
	p5ld_triplesx=h3d*h1d;
	p6ld_triplesx=p5d*h3d*h1d;
	p4ld_triplesx=p6d*p5d*h3d*h1d;
	size_t total_x = h3d*p6d*1;
	size_t total_y = p4d*p5d*h1d;
	dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
	for(i=0;i<nstreams;++i){
		ma_sd_t_d1_9_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(h1d,h3d,h7d,p4d,p5d,p6d,h7ld_t2sub,p4ld_t2sub,p5ld_t2sub,h1ld_t2sub,h3ld_v2sub,p6ld_v2sub,h7ld_v2sub,h1ld_triplesx,h3ld_triplesx,p5ld_triplesx,p6ld_triplesx,p4ld_triplesx,ma_t3_d,t2sub_d,v2sub_d,i,total_x,total_y);
		CHECK_ERR("Kernel execution failed");
	}
	cudaThreadSynchronize();
	for(i=0;i<nstreams;++i){
		cudaStreamDestroy(streams[i]);}
	freeGpuMem(t2sub_d);
	freeGpuMem(v2sub_d);
	free(streams);
}
#undef T1
#undef T2
#undef Tcomm
void ma_sd_t_d1_9_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* h7d, Integer* p4d, Integer* p5d, Integer* p6d, double *triplesx, double *t2sub, double *v2sub) 
{
	ma_sd_t_d1_9_cuda((int)*h1d,(int)*h2d,(int)*h3d,(int)*h7d,(int)*p4d,(int)*p5d,(int)*p6d,triplesx,t2sub,v2sub);
}


/*----------------------------------------------------------------------*
*t3[h3,h2,h1,p6,p4] -= t2[p7,p4,h1,h2] * v2[p7,h3,p6]
*----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void ma_sd_t_d2_1_kernel(size_t h1d,size_t h2d,size_t h3d,size_t p4d,size_t p6d,size_t p7d,
	size_t p7ld_t2,size_t p4ld_t2,size_t h1ld_t2,size_t h2ld_t2,size_t p7ld_v2,size_t h3ld_v2,size_t p6ld_v2,
	size_t h3ld_t3,size_t h2ld_t3,size_t h1ld_t3,size_t p6ld_t3,size_t p4ld_t3,
	double *t3d, double *t2_d, double *v2_d,size_t unused_idx, size_t total_x, size_t total_y) 
{
	size_t h1_0,h1_1,h1_2,h1_3,h2_0,h2_1,h2_2,h2_3,h3_0,h3_1,h3_2,h3_3,p4_0,p4_1,p4_2,p4_3,p6_0,p6_1,p6_2,p6_3,p7;
	double a1,b1;
	double a2,b2;
	double a3,b3;
	double a4,b4;
	size_t in1_idxl,in2_idxl,p7l,p7T;
	__shared__ double t2_shm[4*T1][Tcomm];
	__shared__ double v2_shm[Tcomm][4*T2];
	size_t rest_x=blockIdx.x;
	size_t rest_y=blockIdx.y;
	size_t thread_x = T2*4 * rest_x + threadIdx.x;
	size_t thread_y = T1*4 * rest_y + threadIdx.y;
	in1_idxl=threadIdx.y;
	in2_idxl=threadIdx.x ;
	double tlocal1=0;
	double tlocal2=0;
	double tlocal3=0;
	double tlocal4=0;
	double tlocal5=0;
	double tlocal6=0;
	double tlocal7=0;
	double tlocal8=0;
	double tlocal9=0;
	double tlocal10=0;
	double tlocal11=0;
	double tlocal12=0;
	double tlocal13=0;
	double tlocal14=0;
	double tlocal15=0;
	double tlocal16=0;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
	h3_0=rest_x%h3d;
	rest_x=rest_x/h3d;
	h2_0=rest_y%h2d;
	rest_y=rest_y/h2d;
	h1_0=rest_y%h1d;
	rest_y=rest_y/h1d;
	p4_0=rest_y;
	p6_0=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
	h3_1=rest_x%h3d;
	rest_x=rest_x/h3d;
	h2_1=rest_y%h2d;
	rest_y=rest_y/h2d;
	h1_1=rest_y%h1d;
	rest_y=rest_y/h1d;
	p4_1=rest_y;
	p6_1=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
	h3_2=rest_x%h3d;
	rest_x=rest_x/h3d;
	h2_2=rest_y%h2d;
	rest_y=rest_y/h2d;
	h1_2=rest_y%h1d;
	rest_y=rest_y/h1d;
	p4_2=rest_y;
	p6_2=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
	h3_3=rest_x%h3d;
	rest_x=rest_x/h3d;
	h2_3=rest_y%h2d;
	rest_y=rest_y/h2d;
	h1_3=rest_y%h1d;
	rest_y=rest_y/h1d;
	p4_3=rest_y;
	p6_3=rest_x;
	size_t t2_d_off, v2_d_off;for(p7T=0;p7T<p7d;p7T+=Tcomm){size_t p7l_hi;
		p7l_hi = MIN(Tcomm+p7T,p7d)-p7T;
		t2_d_off=p4_0*p4ld_t2+h1_0*h1ld_t2+h2_0*h2ld_t2;
		v2_d_off=h3_0*h3ld_v2+p6_0*p6ld_v2;
		if(thread_y+T1*0<total_y)for(p7l=threadIdx.x;p7l<p7l_hi;p7l+=blockDim.x){
		p7=p7l+p7T;
		t2_shm[in1_idxl+T1*0][p7l] = t2_d[t2_d_off+p7*p7ld_t2];
		}
		if(thread_x+T1*0<total_x)for(p7l=threadIdx.y;p7l<p7l_hi;p7l+=blockDim.y){
		p7=p7l+p7T;
		v2_shm[p7l][in2_idxl+T1*0] = v2_d[v2_d_off+p7*p7ld_v2];
		}
		t2_d_off=p4_1*p4ld_t2+h1_1*h1ld_t2+h2_1*h2ld_t2;
		v2_d_off=h3_1*h3ld_v2+p6_1*p6ld_v2;
		if(thread_y+T1*1<total_y)for(p7l=threadIdx.x;p7l<p7l_hi;p7l+=blockDim.x){
		p7=p7l+p7T;
		t2_shm[in1_idxl+T1*1][p7l] = t2_d[t2_d_off+p7*p7ld_t2];
		}
		if(thread_x+T1*1<total_x)for(p7l=threadIdx.y;p7l<p7l_hi;p7l+=blockDim.y){
		p7=p7l+p7T;
		v2_shm[p7l][in2_idxl+T1*1] = v2_d[v2_d_off+p7*p7ld_v2];
		}
		t2_d_off=p4_2*p4ld_t2+h1_2*h1ld_t2+h2_2*h2ld_t2;
		v2_d_off=h3_2*h3ld_v2+p6_2*p6ld_v2;
		if(thread_y+T1*2<total_y)for(p7l=threadIdx.x;p7l<p7l_hi;p7l+=blockDim.x){
		p7=p7l+p7T;
		t2_shm[in1_idxl+T1*2][p7l] = t2_d[t2_d_off+p7*p7ld_t2];
		}
		if(thread_x+T1*2<total_x)for(p7l=threadIdx.y;p7l<p7l_hi;p7l+=blockDim.y){
		p7=p7l+p7T;
		v2_shm[p7l][in2_idxl+T1*2] = v2_d[v2_d_off+p7*p7ld_v2];
		}
		t2_d_off=p4_3*p4ld_t2+h1_3*h1ld_t2+h2_3*h2ld_t2;
		v2_d_off=h3_3*h3ld_v2+p6_3*p6ld_v2;
		if(thread_y+T1*3<total_y)for(p7l=threadIdx.x;p7l<p7l_hi;p7l+=blockDim.x){
		p7=p7l+p7T;
		t2_shm[in1_idxl+T1*3][p7l] = t2_d[t2_d_off+p7*p7ld_t2];
		}
		if(thread_x+T1*3<total_x)for(p7l=threadIdx.y;p7l<p7l_hi;p7l+=blockDim.y){
		p7=p7l+p7T;
		v2_shm[p7l][in2_idxl+T1*3] = v2_d[v2_d_off+p7*p7ld_v2];
		}
		__syncthreads();
		for(p7l=0;p7l<p7l_hi;++p7l){
		a1=t2_shm[in1_idxl+T1*0][p7l];
		a2=t2_shm[in1_idxl+T1*1][p7l];
		a3=t2_shm[in1_idxl+T1*2][p7l];
		a4=t2_shm[in1_idxl+T1*3][p7l];
		b1=v2_shm[p7l][in2_idxl+T2*0];
		b2=v2_shm[p7l][in2_idxl+T2*1];
		b3=v2_shm[p7l][in2_idxl+T2*2];
		b4=v2_shm[p7l][in2_idxl+T2*3];
		tlocal1+=a1*b1;
		tlocal2+=a2*b1;
		tlocal3+=a3*b1;
		tlocal4+=a4*b1;
		tlocal5+=a1*b2;
		tlocal6+=a2*b2;
		tlocal7+=a3*b2;
		tlocal8+=a4*b2;
		tlocal9+=a1*b3;
		tlocal10+=a2*b3;
		tlocal11+=a3*b3;
		tlocal12+=a4*b3;
		tlocal13+=a1*b4;
		tlocal14+=a2*b4;
		tlocal15+=a3*b4;
		tlocal16+=a4*b4;
		}
		__syncthreads();
	}
	if(thread_x+T1*0<total_x){
		if(thread_y+T2*3<total_y) {
		t3d[h3_0*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p6_0*p6ld_t3+p4_0*p4ld_t3]-=tlocal1;
		t3d[h3_0*h3ld_t3+h2_1*h2ld_t3+h1_1*h1ld_t3+p6_0*p6ld_t3+p4_1*p4ld_t3]-=tlocal2;
		t3d[h3_0*h3ld_t3+h2_2*h2ld_t3+h1_2*h1ld_t3+p6_0*p6ld_t3+p4_2*p4ld_t3]-=tlocal3;
		t3d[h3_0*h3ld_t3+h2_3*h2ld_t3+h1_3*h1ld_t3+p6_0*p6ld_t3+p4_3*p4ld_t3]-=tlocal4;
		}
		else if(thread_y+T2*2<total_y) {
		t3d[h3_0*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p6_0*p6ld_t3+p4_0*p4ld_t3]-=tlocal1;
		t3d[h3_0*h3ld_t3+h2_1*h2ld_t3+h1_1*h1ld_t3+p6_0*p6ld_t3+p4_1*p4ld_t3]-=tlocal2;
		t3d[h3_0*h3ld_t3+h2_2*h2ld_t3+h1_2*h1ld_t3+p6_0*p6ld_t3+p4_2*p4ld_t3]-=tlocal3;
		}
		else if(thread_y+T2*1<total_y) {
		t3d[h3_0*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p6_0*p6ld_t3+p4_0*p4ld_t3]-=tlocal1;
		t3d[h3_0*h3ld_t3+h2_1*h2ld_t3+h1_1*h1ld_t3+p6_0*p6ld_t3+p4_1*p4ld_t3]-=tlocal2;
		}
		else if(thread_y+T2*0<total_y) {
		t3d[h3_0*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p6_0*p6ld_t3+p4_0*p4ld_t3]-=tlocal1;
		}
	}
	if(thread_x+T1*1<total_x){
		if(thread_y+T2*3<total_y) {
		t3d[h3_1*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p6_1*p6ld_t3+p4_0*p4ld_t3]-=tlocal5;
		t3d[h3_1*h3ld_t3+h2_1*h2ld_t3+h1_1*h1ld_t3+p6_1*p6ld_t3+p4_1*p4ld_t3]-=tlocal6;
		t3d[h3_1*h3ld_t3+h2_2*h2ld_t3+h1_2*h1ld_t3+p6_1*p6ld_t3+p4_2*p4ld_t3]-=tlocal7;
		t3d[h3_1*h3ld_t3+h2_3*h2ld_t3+h1_3*h1ld_t3+p6_1*p6ld_t3+p4_3*p4ld_t3]-=tlocal8;
		}
		else if(thread_y+T2*2<total_y) {
		t3d[h3_1*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p6_1*p6ld_t3+p4_0*p4ld_t3]-=tlocal5;
		t3d[h3_1*h3ld_t3+h2_1*h2ld_t3+h1_1*h1ld_t3+p6_1*p6ld_t3+p4_1*p4ld_t3]-=tlocal6;
		t3d[h3_1*h3ld_t3+h2_2*h2ld_t3+h1_2*h1ld_t3+p6_1*p6ld_t3+p4_2*p4ld_t3]-=tlocal7;
		}
		else if(thread_y+T2*1<total_y) {
		t3d[h3_1*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p6_1*p6ld_t3+p4_0*p4ld_t3]-=tlocal5;
		t3d[h3_1*h3ld_t3+h2_1*h2ld_t3+h1_1*h1ld_t3+p6_1*p6ld_t3+p4_1*p4ld_t3]-=tlocal6;
		}
		else if(thread_y+T2*0<total_y) {
		t3d[h3_1*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p6_1*p6ld_t3+p4_0*p4ld_t3]-=tlocal5;
		}
	}
	if(thread_x+T1*2<total_x){
		if(thread_y+T2*3<total_y) {
		t3d[h3_2*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p6_2*p6ld_t3+p4_0*p4ld_t3]-=tlocal9;
		t3d[h3_2*h3ld_t3+h2_1*h2ld_t3+h1_1*h1ld_t3+p6_2*p6ld_t3+p4_1*p4ld_t3]-=tlocal10;
		t3d[h3_2*h3ld_t3+h2_2*h2ld_t3+h1_2*h1ld_t3+p6_2*p6ld_t3+p4_2*p4ld_t3]-=tlocal11;
		t3d[h3_2*h3ld_t3+h2_3*h2ld_t3+h1_3*h1ld_t3+p6_2*p6ld_t3+p4_3*p4ld_t3]-=tlocal12;
		}
		else if(thread_y+T2*2<total_y) {
		t3d[h3_2*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p6_2*p6ld_t3+p4_0*p4ld_t3]-=tlocal9;
		t3d[h3_2*h3ld_t3+h2_1*h2ld_t3+h1_1*h1ld_t3+p6_2*p6ld_t3+p4_1*p4ld_t3]-=tlocal10;
		t3d[h3_2*h3ld_t3+h2_2*h2ld_t3+h1_2*h1ld_t3+p6_2*p6ld_t3+p4_2*p4ld_t3]-=tlocal11;
		}
		else if(thread_y+T2*1<total_y) {
		t3d[h3_2*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p6_2*p6ld_t3+p4_0*p4ld_t3]-=tlocal9;
		t3d[h3_2*h3ld_t3+h2_1*h2ld_t3+h1_1*h1ld_t3+p6_2*p6ld_t3+p4_1*p4ld_t3]-=tlocal10;
		}
		else if(thread_y+T2*0<total_y) {
		t3d[h3_2*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p6_2*p6ld_t3+p4_0*p4ld_t3]-=tlocal9;
		}
	}
	if(thread_x+T1*3<total_x){
		if(thread_y+T2*3<total_y) {
		t3d[h3_3*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p6_3*p6ld_t3+p4_0*p4ld_t3]-=tlocal13;
		t3d[h3_3*h3ld_t3+h2_1*h2ld_t3+h1_1*h1ld_t3+p6_3*p6ld_t3+p4_1*p4ld_t3]-=tlocal14;
		t3d[h3_3*h3ld_t3+h2_2*h2ld_t3+h1_2*h1ld_t3+p6_3*p6ld_t3+p4_2*p4ld_t3]-=tlocal15;
		t3d[h3_3*h3ld_t3+h2_3*h2ld_t3+h1_3*h1ld_t3+p6_3*p6ld_t3+p4_3*p4ld_t3]-=tlocal16;
		}
		else if(thread_y+T2*2<total_y) {
		t3d[h3_3*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p6_3*p6ld_t3+p4_0*p4ld_t3]-=tlocal13;
		t3d[h3_3*h3ld_t3+h2_1*h2ld_t3+h1_1*h1ld_t3+p6_3*p6ld_t3+p4_1*p4ld_t3]-=tlocal14;
		t3d[h3_3*h3ld_t3+h2_2*h2ld_t3+h1_2*h1ld_t3+p6_3*p6ld_t3+p4_2*p4ld_t3]-=tlocal15;
		}
		else if(thread_y+T2*1<total_y) {
		t3d[h3_3*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p6_3*p6ld_t3+p4_0*p4ld_t3]-=tlocal13;
		t3d[h3_3*h3ld_t3+h2_1*h2ld_t3+h1_1*h1ld_t3+p6_3*p6ld_t3+p4_1*p4ld_t3]-=tlocal14;
		}
		else if(thread_y+T2*0<total_y) {
		t3d[h3_3*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p6_3*p6ld_t3+p4_0*p4ld_t3]-=tlocal13;
		}
	}
	__syncthreads();
}

void ma_sd_t_d2_1_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d, size_t p7d, double *t3, double *t2, double *v2) 
{
	p6d=p6d*p5d;
	size_t p7ld_t2,p4ld_t2,h1ld_t2,h2ld_t2,p7ld_v2,h3ld_v2,p6ld_v2,h3ld_t3,h2ld_t3,h1ld_t3,p6ld_t3,p4ld_t3;
	size_t size_t3,size_block_t3,size_el_block_t3,size_t2,size_v2;
	cudaStream_t *streams;
	size_t nstreams,i;
	double *t2_d,*v2_d;
	size_t3=h3d*h2d*h1d*p6d*p4d*sizeof(double);
	size_t2=p7d*p4d*h1d*h2d*sizeof(double);
	size_v2=p7d*h3d*p6d*sizeof(double);
	cudaFuncSetCacheConfig(ma_sd_t_d2_1_kernel, cudaFuncCachePreferShared);
	nstreams=1;
	size_block_t3=size_t3/nstreams;
	size_el_block_t3=size_block_t3/sizeof(double);
	//t3d=(double*)getGpuMem(size_t3);
	t2_d=(double*)getGpuMem(size_t2);
	v2_d=(double*)getGpuMem(size_v2);
	streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
	assert(streams!= NULL);
	for(i=0;i<nstreams;++i) {
		CUDA_SAFE(cudaStreamCreate(&streams[i])) ;
	}
	CUDA_SAFE(cudaMemcpy(t2_d,t2,size_t2,cudaMemcpyHostToDevice));
	CUDA_SAFE(cudaMemcpy(v2_d,v2,size_v2,cudaMemcpyHostToDevice));
	p7ld_t2=1;
	p4ld_t2=p7d;
	h1ld_t2=p4d*p7d;
	h2ld_t2=h1d*p4d*p7d;
	p7ld_v2=1;
	h3ld_v2=p7d;
	p6ld_v2=h3d*p7d;
	h3ld_t3=1;
	h2ld_t3=h3d;
	h1ld_t3=h2d*h3d;
	p6ld_t3=h1d*h2d*h3d;
	p4ld_t3=p6d*h1d*h2d*h3d;
	size_t total_x = h3d*p6d;
	size_t total_y = p4d*h1d*h2d;
	dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
	for(i=0;i<nstreams;++i){
		ma_sd_t_d2_1_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(h1d,h2d,h3d,p4d,p6d,p7d,p7ld_t2,p4ld_t2,h1ld_t2,h2ld_t2,p7ld_v2,h3ld_v2,p6ld_v2,h3ld_t3,h2ld_t3,h1ld_t3,p6ld_t3,p4ld_t3,ma_t3_d,t2_d,v2_d,i,total_x,total_y);
		CHECK_ERR("Kernel execution failed");
	}
	cudaThreadSynchronize();
	for(i=0;i<nstreams;++i){
		cudaStreamDestroy(streams[i]);}
	//freeGpuMem(t3d);
	freeGpuMem(t2_d);
	freeGpuMem(v2_d);
	free(streams);
}
#undef T1
#undef T2
#undef Tcomm
void ma_sd_t_d2_1_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* p4d, Integer* p5d, Integer* p6d, Integer* p7d, double *t3, double *t2, double *v2) 
{
	ma_sd_t_d2_1_cuda((int)*h1d,(int)*h2d,(int)*h3d,(int)*p4d,(int)*p5d,(int)*p6d,(int)*p7d,t3,t2,v2);
}
/*----------------------------------------------------------------------*
*t3[h2,h1,h3,p4] -= t2[p7,p4,h1,h2] * v2[p7,h3]
*----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void ma_sd_t_d2_2_kernel(size_t h1d,size_t h2d,size_t h3d,size_t p4d,size_t p7d,
	size_t p7ld_t2,size_t p4ld_t2,size_t h1ld_t2,size_t h2ld_t2,size_t p7ld_v2,size_t h3ld_v2,
	size_t h2ld_t3,size_t h1ld_t3,size_t h3ld_t3,size_t p4ld_t3,
	double *t3d, double *t2_d, double *v2_d,size_t unused_idx, size_t total_x, size_t total_y) 
{
	size_t h1_0,h1_1,h1_2,h1_3,h2_0,h2_1,h2_2,h2_3,h3_0,h3_1,h3_2,h3_3,p4_0,p4_1,p4_2,p4_3,p7;
	double a1,b1;
	double a2,b2;
	double a3,b3;
	double a4,b4;
	size_t in1_idxl,in2_idxl,p7l,p7T;
	__shared__ double t2_shm[4*T1][Tcomm];
	__shared__ double v2_shm[Tcomm][4*T2];
	size_t rest_x=blockIdx.x;
	size_t rest_y=blockIdx.y;
	size_t thread_x = T2*4 * rest_x + threadIdx.x;
	size_t thread_y = T1*4 * rest_y + threadIdx.y;
	in1_idxl=threadIdx.y;
	in2_idxl=threadIdx.x ;
	double tlocal1=0;
	double tlocal2=0;
	double tlocal3=0;
	double tlocal4=0;
	double tlocal5=0;
	double tlocal6=0;
	double tlocal7=0;
	double tlocal8=0;
	double tlocal9=0;
	double tlocal10=0;
	double tlocal11=0;
	double tlocal12=0;
	double tlocal13=0;
	double tlocal14=0;
	double tlocal15=0;
	double tlocal16=0;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
	h2_0=rest_y%h2d;
	rest_y=rest_y/h2d;
	h1_0=rest_y%h1d;
	rest_y=rest_y/h1d;
	p4_0=rest_y;
	h3_0=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
	h2_1=rest_y%h2d;
	rest_y=rest_y/h2d;
	h1_1=rest_y%h1d;
	rest_y=rest_y/h1d;
	p4_1=rest_y;
	h3_1=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
	h2_2=rest_y%h2d;
	rest_y=rest_y/h2d;
	h1_2=rest_y%h1d;
	rest_y=rest_y/h1d;
	p4_2=rest_y;
	h3_2=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
	h2_3=rest_y%h2d;
	rest_y=rest_y/h2d;
	h1_3=rest_y%h1d;
	rest_y=rest_y/h1d;
	p4_3=rest_y;
	h3_3=rest_x;
	size_t t2_d_off, v2_d_off;for(p7T=0;p7T<p7d;p7T+=Tcomm){size_t p7l_hi;
		p7l_hi = MIN(Tcomm+p7T,p7d)-p7T;
		t2_d_off=p4_0*p4ld_t2+h1_0*h1ld_t2+h2_0*h2ld_t2;
		v2_d_off=h3_0*h3ld_v2;
		if(thread_y+T1*0<total_y)for(p7l=threadIdx.x;p7l<p7l_hi;p7l+=blockDim.x){
		p7=p7l+p7T;
		t2_shm[in1_idxl+T1*0][p7l] = t2_d[t2_d_off+p7*p7ld_t2];
		}
		if(thread_x+T1*0<total_x)for(p7l=threadIdx.y;p7l<p7l_hi;p7l+=blockDim.y){
		p7=p7l+p7T;
		v2_shm[p7l][in2_idxl+T1*0] = v2_d[v2_d_off+p7*p7ld_v2];
		}
		t2_d_off=p4_1*p4ld_t2+h1_1*h1ld_t2+h2_1*h2ld_t2;
		v2_d_off=h3_1*h3ld_v2;
		if(thread_y+T1*1<total_y)for(p7l=threadIdx.x;p7l<p7l_hi;p7l+=blockDim.x){
		p7=p7l+p7T;
		t2_shm[in1_idxl+T1*1][p7l] = t2_d[t2_d_off+p7*p7ld_t2];
		}
		if(thread_x+T1*1<total_x)for(p7l=threadIdx.y;p7l<p7l_hi;p7l+=blockDim.y){
		p7=p7l+p7T;
		v2_shm[p7l][in2_idxl+T1*1] = v2_d[v2_d_off+p7*p7ld_v2];
		}
		t2_d_off=p4_2*p4ld_t2+h1_2*h1ld_t2+h2_2*h2ld_t2;
		v2_d_off=h3_2*h3ld_v2;
		if(thread_y+T1*2<total_y)for(p7l=threadIdx.x;p7l<p7l_hi;p7l+=blockDim.x){
		p7=p7l+p7T;
		t2_shm[in1_idxl+T1*2][p7l] = t2_d[t2_d_off+p7*p7ld_t2];
		}
		if(thread_x+T1*2<total_x)for(p7l=threadIdx.y;p7l<p7l_hi;p7l+=blockDim.y){
		p7=p7l+p7T;
		v2_shm[p7l][in2_idxl+T1*2] = v2_d[v2_d_off+p7*p7ld_v2];
		}
		t2_d_off=p4_3*p4ld_t2+h1_3*h1ld_t2+h2_3*h2ld_t2;
		v2_d_off=h3_3*h3ld_v2;
		if(thread_y+T1*3<total_y)for(p7l=threadIdx.x;p7l<p7l_hi;p7l+=blockDim.x){
		p7=p7l+p7T;
		t2_shm[in1_idxl+T1*3][p7l] = t2_d[t2_d_off+p7*p7ld_t2];
		}
		if(thread_x+T1*3<total_x)for(p7l=threadIdx.y;p7l<p7l_hi;p7l+=blockDim.y){
		p7=p7l+p7T;
		v2_shm[p7l][in2_idxl+T1*3] = v2_d[v2_d_off+p7*p7ld_v2];
		}
		__syncthreads();
		for(p7l=0;p7l<p7l_hi;++p7l){
		a1=t2_shm[in1_idxl+T1*0][p7l];
		a2=t2_shm[in1_idxl+T1*1][p7l];
		a3=t2_shm[in1_idxl+T1*2][p7l];
		a4=t2_shm[in1_idxl+T1*3][p7l];
		b1=v2_shm[p7l][in2_idxl+T2*0];
		b2=v2_shm[p7l][in2_idxl+T2*1];
		b3=v2_shm[p7l][in2_idxl+T2*2];
		b4=v2_shm[p7l][in2_idxl+T2*3];
		tlocal1+=a1*b1;
		tlocal2+=a2*b1;
		tlocal3+=a3*b1;
		tlocal4+=a4*b1;
		tlocal5+=a1*b2;
		tlocal6+=a2*b2;
		tlocal7+=a3*b2;
		tlocal8+=a4*b2;
		tlocal9+=a1*b3;
		tlocal10+=a2*b3;
		tlocal11+=a3*b3;
		tlocal12+=a4*b3;
		tlocal13+=a1*b4;
		tlocal14+=a2*b4;
		tlocal15+=a3*b4;
		tlocal16+=a4*b4;
		}
		__syncthreads();
	}
	if(thread_x+T1*0<total_x){
		if(thread_y+T2*3<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_0*h3ld_t3+p4_0*p4ld_t3]-=tlocal1;
		t3d[h2_1*h2ld_t3+h1_1*h1ld_t3+h3_0*h3ld_t3+p4_1*p4ld_t3]-=tlocal2;
		t3d[h2_2*h2ld_t3+h1_2*h1ld_t3+h3_0*h3ld_t3+p4_2*p4ld_t3]-=tlocal3;
		t3d[h2_3*h2ld_t3+h1_3*h1ld_t3+h3_0*h3ld_t3+p4_3*p4ld_t3]-=tlocal4;
		}
		else if(thread_y+T2*2<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_0*h3ld_t3+p4_0*p4ld_t3]-=tlocal1;
		t3d[h2_1*h2ld_t3+h1_1*h1ld_t3+h3_0*h3ld_t3+p4_1*p4ld_t3]-=tlocal2;
		t3d[h2_2*h2ld_t3+h1_2*h1ld_t3+h3_0*h3ld_t3+p4_2*p4ld_t3]-=tlocal3;
		}
		else if(thread_y+T2*1<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_0*h3ld_t3+p4_0*p4ld_t3]-=tlocal1;
		t3d[h2_1*h2ld_t3+h1_1*h1ld_t3+h3_0*h3ld_t3+p4_1*p4ld_t3]-=tlocal2;
		}
		else if(thread_y+T2*0<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_0*h3ld_t3+p4_0*p4ld_t3]-=tlocal1;
		}
	}
	if(thread_x+T1*1<total_x){
		if(thread_y+T2*3<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_1*h3ld_t3+p4_0*p4ld_t3]-=tlocal5;
		t3d[h2_1*h2ld_t3+h1_1*h1ld_t3+h3_1*h3ld_t3+p4_1*p4ld_t3]-=tlocal6;
		t3d[h2_2*h2ld_t3+h1_2*h1ld_t3+h3_1*h3ld_t3+p4_2*p4ld_t3]-=tlocal7;
		t3d[h2_3*h2ld_t3+h1_3*h1ld_t3+h3_1*h3ld_t3+p4_3*p4ld_t3]-=tlocal8;
		}
		else if(thread_y+T2*2<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_1*h3ld_t3+p4_0*p4ld_t3]-=tlocal5;
		t3d[h2_1*h2ld_t3+h1_1*h1ld_t3+h3_1*h3ld_t3+p4_1*p4ld_t3]-=tlocal6;
		t3d[h2_2*h2ld_t3+h1_2*h1ld_t3+h3_1*h3ld_t3+p4_2*p4ld_t3]-=tlocal7;
		}
		else if(thread_y+T2*1<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_1*h3ld_t3+p4_0*p4ld_t3]-=tlocal5;
		t3d[h2_1*h2ld_t3+h1_1*h1ld_t3+h3_1*h3ld_t3+p4_1*p4ld_t3]-=tlocal6;
		}
		else if(thread_y+T2*0<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_1*h3ld_t3+p4_0*p4ld_t3]-=tlocal5;
		}
	}
	if(thread_x+T1*2<total_x){
		if(thread_y+T2*3<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_2*h3ld_t3+p4_0*p4ld_t3]-=tlocal9;
		t3d[h2_1*h2ld_t3+h1_1*h1ld_t3+h3_2*h3ld_t3+p4_1*p4ld_t3]-=tlocal10;
		t3d[h2_2*h2ld_t3+h1_2*h1ld_t3+h3_2*h3ld_t3+p4_2*p4ld_t3]-=tlocal11;
		t3d[h2_3*h2ld_t3+h1_3*h1ld_t3+h3_2*h3ld_t3+p4_3*p4ld_t3]-=tlocal12;
		}
		else if(thread_y+T2*2<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_2*h3ld_t3+p4_0*p4ld_t3]-=tlocal9;
		t3d[h2_1*h2ld_t3+h1_1*h1ld_t3+h3_2*h3ld_t3+p4_1*p4ld_t3]-=tlocal10;
		t3d[h2_2*h2ld_t3+h1_2*h1ld_t3+h3_2*h3ld_t3+p4_2*p4ld_t3]-=tlocal11;
		}
		else if(thread_y+T2*1<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_2*h3ld_t3+p4_0*p4ld_t3]-=tlocal9;
		t3d[h2_1*h2ld_t3+h1_1*h1ld_t3+h3_2*h3ld_t3+p4_1*p4ld_t3]-=tlocal10;
		}
		else if(thread_y+T2*0<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_2*h3ld_t3+p4_0*p4ld_t3]-=tlocal9;
		}
	}
	if(thread_x+T1*3<total_x){
		if(thread_y+T2*3<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_3*h3ld_t3+p4_0*p4ld_t3]-=tlocal13;
		t3d[h2_1*h2ld_t3+h1_1*h1ld_t3+h3_3*h3ld_t3+p4_1*p4ld_t3]-=tlocal14;
		t3d[h2_2*h2ld_t3+h1_2*h1ld_t3+h3_3*h3ld_t3+p4_2*p4ld_t3]-=tlocal15;
		t3d[h2_3*h2ld_t3+h1_3*h1ld_t3+h3_3*h3ld_t3+p4_3*p4ld_t3]-=tlocal16;
		}
		else if(thread_y+T2*2<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_3*h3ld_t3+p4_0*p4ld_t3]-=tlocal13;
		t3d[h2_1*h2ld_t3+h1_1*h1ld_t3+h3_3*h3ld_t3+p4_1*p4ld_t3]-=tlocal14;
		t3d[h2_2*h2ld_t3+h1_2*h1ld_t3+h3_3*h3ld_t3+p4_2*p4ld_t3]-=tlocal15;
		}
		else if(thread_y+T2*1<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_3*h3ld_t3+p4_0*p4ld_t3]-=tlocal13;
		t3d[h2_1*h2ld_t3+h1_1*h1ld_t3+h3_3*h3ld_t3+p4_1*p4ld_t3]-=tlocal14;
		}
		else if(thread_y+T2*0<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_3*h3ld_t3+p4_0*p4ld_t3]-=tlocal13;
		}
	}
	__syncthreads();
}

void ma_sd_t_d2_2_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d, size_t p7d, double *t3, double *t2, double *v2) 
{
	h3d=h3d*p6d;
	h3d=h3d*p5d;
	size_t p7ld_t2,p4ld_t2,h1ld_t2,h2ld_t2,p7ld_v2,h3ld_v2,h2ld_t3,h1ld_t3,h3ld_t3,p4ld_t3;
	size_t size_t3,size_block_t3,size_el_block_t3,size_t2,size_v2;
	cudaStream_t *streams;
	size_t nstreams,i;
	double *t2_d,*v2_d;
	size_t3=h2d*h1d*h3d*p4d*sizeof(double);
	size_t2=p7d*p4d*h1d*h2d*sizeof(double);
	size_v2=p7d*h3d*sizeof(double);
	cudaFuncSetCacheConfig(ma_sd_t_d2_2_kernel, cudaFuncCachePreferShared);
	nstreams=1;
	size_block_t3=size_t3/nstreams;
	size_el_block_t3=size_block_t3/sizeof(double);
	//t3d=(double*)getGpuMem(size_t3);
	t2_d=(double*)getGpuMem(size_t2);
	v2_d=(double*)getGpuMem(size_v2);
	streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
	assert(streams!= NULL);
	for(i=0;i<nstreams;++i) {
		CUDA_SAFE(cudaStreamCreate(&streams[i])) ;
	}
	CUDA_SAFE(cudaMemcpy(t2_d,t2,size_t2,cudaMemcpyHostToDevice));
	CUDA_SAFE(cudaMemcpy(v2_d,v2,size_v2,cudaMemcpyHostToDevice));
	p7ld_t2=1;
	p4ld_t2=p7d;
	h1ld_t2=p4d*p7d;
	h2ld_t2=h1d*p4d*p7d;
	p7ld_v2=1;
	h3ld_v2=p7d;
	h2ld_t3=1;
	h1ld_t3=h2d;
	h3ld_t3=h1d*h2d;
	p4ld_t3=h3d*h1d*h2d;
	size_t total_x = h3d;
	size_t total_y = p4d*h1d*h2d;
	dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
	for(i=0;i<nstreams;++i){
		ma_sd_t_d2_2_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(h1d,h2d,h3d,p4d,p7d,p7ld_t2,p4ld_t2,h1ld_t2,h2ld_t2,p7ld_v2,h3ld_v2,h2ld_t3,h1ld_t3,h3ld_t3,p4ld_t3,ma_t3_d,t2_d,v2_d,i,total_x,total_y);
		CHECK_ERR("Kernel execution failed");
	}
	cudaThreadSynchronize();
	for(i=0;i<nstreams;++i){
		cudaStreamDestroy(streams[i]);}
	//freeGpuMem(t3d);
	freeGpuMem(t2_d);
	freeGpuMem(v2_d);
	free(streams);
}
#undef T1
#undef T2
#undef Tcomm
void ma_sd_t_d2_2_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* p4d, Integer* p5d, Integer* p6d, Integer* p7d, double *t3, double *t2, double *v2) 
{
	ma_sd_t_d2_2_cuda((int)*h1d,(int)*h2d,(int)*h3d,(int)*p4d,(int)*p5d,(int)*p6d,(int)*p7d,t3,t2,v2);
}
/*----------------------------------------------------------------------*
*t3[h2,h3,h1,p6,p4] += t2[p7,p4,h1,h2] * v2[p7,h3,p6]
*----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void ma_sd_t_d2_3_kernel(size_t h1d,size_t h2d,size_t h3d,size_t p4d,size_t p6d,size_t p7d,
	size_t p7ld_t2,size_t p4ld_t2,size_t h1ld_t2,size_t h2ld_t2,size_t p7ld_v2,size_t h3ld_v2,size_t p6ld_v2,
	size_t h2ld_t3,size_t h3ld_t3,size_t h1ld_t3,size_t p6ld_t3,size_t p4ld_t3,
	double *t3d, double *t2_d, double *v2_d,size_t unused_idx, size_t total_x, size_t total_y) 
{
	size_t h1_0,h1_1,h1_2,h1_3,h2_0,h2_1,h2_2,h2_3,h3_0,h3_1,h3_2,h3_3,p4_0,p4_1,p4_2,p4_3,p6_0,p6_1,p6_2,p6_3,p7;
	double a1,b1;
	double a2,b2;
	double a3,b3;
	double a4,b4;
	size_t in1_idxl,in2_idxl,p7l,p7T;
	__shared__ double t2_shm[4*T1][Tcomm];
	__shared__ double v2_shm[Tcomm][4*T2];
	size_t rest_x=blockIdx.x;
	size_t rest_y=blockIdx.y;
	size_t thread_x = T2*4 * rest_x + threadIdx.x;
	size_t thread_y = T1*4 * rest_y + threadIdx.y;
	in1_idxl=threadIdx.y;
	in2_idxl=threadIdx.x ;
	double tlocal1=0;
	double tlocal2=0;
	double tlocal3=0;
	double tlocal4=0;
	double tlocal5=0;
	double tlocal6=0;
	double tlocal7=0;
	double tlocal8=0;
	double tlocal9=0;
	double tlocal10=0;
	double tlocal11=0;
	double tlocal12=0;
	double tlocal13=0;
	double tlocal14=0;
	double tlocal15=0;
	double tlocal16=0;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
	h2_0=rest_y%h2d;
	rest_y=rest_y/h2d;
	h3_0=rest_x%h3d;
	rest_x=rest_x/h3d;
	h1_0=rest_y%h1d;
	rest_y=rest_y/h1d;
	p4_0=rest_y;
	p6_0=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
	h2_1=rest_y%h2d;
	rest_y=rest_y/h2d;
	h3_1=rest_x%h3d;
	rest_x=rest_x/h3d;
	h1_1=rest_y%h1d;
	rest_y=rest_y/h1d;
	p4_1=rest_y;
	p6_1=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
	h2_2=rest_y%h2d;
	rest_y=rest_y/h2d;
	h3_2=rest_x%h3d;
	rest_x=rest_x/h3d;
	h1_2=rest_y%h1d;
	rest_y=rest_y/h1d;
	p4_2=rest_y;
	p6_2=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
	h2_3=rest_y%h2d;
	rest_y=rest_y/h2d;
	h3_3=rest_x%h3d;
	rest_x=rest_x/h3d;
	h1_3=rest_y%h1d;
	rest_y=rest_y/h1d;
	p4_3=rest_y;
	p6_3=rest_x;
	size_t t2_d_off, v2_d_off;for(p7T=0;p7T<p7d;p7T+=Tcomm){size_t p7l_hi;
		p7l_hi = MIN(Tcomm+p7T,p7d)-p7T;
		t2_d_off=p4_0*p4ld_t2+h1_0*h1ld_t2+h2_0*h2ld_t2;
		v2_d_off=h3_0*h3ld_v2+p6_0*p6ld_v2;
		if(thread_y+T1*0<total_y)for(p7l=threadIdx.x;p7l<p7l_hi;p7l+=blockDim.x){
		p7=p7l+p7T;
		t2_shm[in1_idxl+T1*0][p7l] = t2_d[t2_d_off+p7*p7ld_t2];
		}
		if(thread_x+T1*0<total_x)for(p7l=threadIdx.y;p7l<p7l_hi;p7l+=blockDim.y){
		p7=p7l+p7T;
		v2_shm[p7l][in2_idxl+T1*0] = v2_d[v2_d_off+p7*p7ld_v2];
		}
		t2_d_off=p4_1*p4ld_t2+h1_1*h1ld_t2+h2_1*h2ld_t2;
		v2_d_off=h3_1*h3ld_v2+p6_1*p6ld_v2;
		if(thread_y+T1*1<total_y)for(p7l=threadIdx.x;p7l<p7l_hi;p7l+=blockDim.x){
		p7=p7l+p7T;
		t2_shm[in1_idxl+T1*1][p7l] = t2_d[t2_d_off+p7*p7ld_t2];
		}
		if(thread_x+T1*1<total_x)for(p7l=threadIdx.y;p7l<p7l_hi;p7l+=blockDim.y){
		p7=p7l+p7T;
		v2_shm[p7l][in2_idxl+T1*1] = v2_d[v2_d_off+p7*p7ld_v2];
		}
		t2_d_off=p4_2*p4ld_t2+h1_2*h1ld_t2+h2_2*h2ld_t2;
		v2_d_off=h3_2*h3ld_v2+p6_2*p6ld_v2;
		if(thread_y+T1*2<total_y)for(p7l=threadIdx.x;p7l<p7l_hi;p7l+=blockDim.x){
		p7=p7l+p7T;
		t2_shm[in1_idxl+T1*2][p7l] = t2_d[t2_d_off+p7*p7ld_t2];
		}
		if(thread_x+T1*2<total_x)for(p7l=threadIdx.y;p7l<p7l_hi;p7l+=blockDim.y){
		p7=p7l+p7T;
		v2_shm[p7l][in2_idxl+T1*2] = v2_d[v2_d_off+p7*p7ld_v2];
		}
		t2_d_off=p4_3*p4ld_t2+h1_3*h1ld_t2+h2_3*h2ld_t2;
		v2_d_off=h3_3*h3ld_v2+p6_3*p6ld_v2;
		if(thread_y+T1*3<total_y)for(p7l=threadIdx.x;p7l<p7l_hi;p7l+=blockDim.x){
		p7=p7l+p7T;
		t2_shm[in1_idxl+T1*3][p7l] = t2_d[t2_d_off+p7*p7ld_t2];
		}
		if(thread_x+T1*3<total_x)for(p7l=threadIdx.y;p7l<p7l_hi;p7l+=blockDim.y){
		p7=p7l+p7T;
		v2_shm[p7l][in2_idxl+T1*3] = v2_d[v2_d_off+p7*p7ld_v2];
		}
		__syncthreads();
		for(p7l=0;p7l<p7l_hi;++p7l){
		a1=t2_shm[in1_idxl+T1*0][p7l];
		a2=t2_shm[in1_idxl+T1*1][p7l];
		a3=t2_shm[in1_idxl+T1*2][p7l];
		a4=t2_shm[in1_idxl+T1*3][p7l];
		b1=v2_shm[p7l][in2_idxl+T2*0];
		b2=v2_shm[p7l][in2_idxl+T2*1];
		b3=v2_shm[p7l][in2_idxl+T2*2];
		b4=v2_shm[p7l][in2_idxl+T2*3];
		tlocal1+=a1*b1;
		tlocal2+=a2*b1;
		tlocal3+=a3*b1;
		tlocal4+=a4*b1;
		tlocal5+=a1*b2;
		tlocal6+=a2*b2;
		tlocal7+=a3*b2;
		tlocal8+=a4*b2;
		tlocal9+=a1*b3;
		tlocal10+=a2*b3;
		tlocal11+=a3*b3;
		tlocal12+=a4*b3;
		tlocal13+=a1*b4;
		tlocal14+=a2*b4;
		tlocal15+=a3*b4;
		tlocal16+=a4*b4;
		}
		__syncthreads();
	}
	if(thread_x+T1*0<total_x){
		if(thread_y+T2*3<total_y) {
		t3d[h2_0*h2ld_t3+h3_0*h3ld_t3+h1_0*h1ld_t3+p6_0*p6ld_t3+p4_0*p4ld_t3]+=tlocal1;
		t3d[h2_1*h2ld_t3+h3_0*h3ld_t3+h1_1*h1ld_t3+p6_0*p6ld_t3+p4_1*p4ld_t3]+=tlocal2;
		t3d[h2_2*h2ld_t3+h3_0*h3ld_t3+h1_2*h1ld_t3+p6_0*p6ld_t3+p4_2*p4ld_t3]+=tlocal3;
		t3d[h2_3*h2ld_t3+h3_0*h3ld_t3+h1_3*h1ld_t3+p6_0*p6ld_t3+p4_3*p4ld_t3]+=tlocal4;
		}
		else if(thread_y+T2*2<total_y) {
		t3d[h2_0*h2ld_t3+h3_0*h3ld_t3+h1_0*h1ld_t3+p6_0*p6ld_t3+p4_0*p4ld_t3]+=tlocal1;
		t3d[h2_1*h2ld_t3+h3_0*h3ld_t3+h1_1*h1ld_t3+p6_0*p6ld_t3+p4_1*p4ld_t3]+=tlocal2;
		t3d[h2_2*h2ld_t3+h3_0*h3ld_t3+h1_2*h1ld_t3+p6_0*p6ld_t3+p4_2*p4ld_t3]+=tlocal3;
		}
		else if(thread_y+T2*1<total_y) {
		t3d[h2_0*h2ld_t3+h3_0*h3ld_t3+h1_0*h1ld_t3+p6_0*p6ld_t3+p4_0*p4ld_t3]+=tlocal1;
		t3d[h2_1*h2ld_t3+h3_0*h3ld_t3+h1_1*h1ld_t3+p6_0*p6ld_t3+p4_1*p4ld_t3]+=tlocal2;
		}
		else if(thread_y+T2*0<total_y) {
		t3d[h2_0*h2ld_t3+h3_0*h3ld_t3+h1_0*h1ld_t3+p6_0*p6ld_t3+p4_0*p4ld_t3]+=tlocal1;
		}
	}
	if(thread_x+T1*1<total_x){
		if(thread_y+T2*3<total_y) {
		t3d[h2_0*h2ld_t3+h3_1*h3ld_t3+h1_0*h1ld_t3+p6_1*p6ld_t3+p4_0*p4ld_t3]+=tlocal5;
		t3d[h2_1*h2ld_t3+h3_1*h3ld_t3+h1_1*h1ld_t3+p6_1*p6ld_t3+p4_1*p4ld_t3]+=tlocal6;
		t3d[h2_2*h2ld_t3+h3_1*h3ld_t3+h1_2*h1ld_t3+p6_1*p6ld_t3+p4_2*p4ld_t3]+=tlocal7;
		t3d[h2_3*h2ld_t3+h3_1*h3ld_t3+h1_3*h1ld_t3+p6_1*p6ld_t3+p4_3*p4ld_t3]+=tlocal8;
		}
		else if(thread_y+T2*2<total_y) {
		t3d[h2_0*h2ld_t3+h3_1*h3ld_t3+h1_0*h1ld_t3+p6_1*p6ld_t3+p4_0*p4ld_t3]+=tlocal5;
		t3d[h2_1*h2ld_t3+h3_1*h3ld_t3+h1_1*h1ld_t3+p6_1*p6ld_t3+p4_1*p4ld_t3]+=tlocal6;
		t3d[h2_2*h2ld_t3+h3_1*h3ld_t3+h1_2*h1ld_t3+p6_1*p6ld_t3+p4_2*p4ld_t3]+=tlocal7;
		}
		else if(thread_y+T2*1<total_y) {
		t3d[h2_0*h2ld_t3+h3_1*h3ld_t3+h1_0*h1ld_t3+p6_1*p6ld_t3+p4_0*p4ld_t3]+=tlocal5;
		t3d[h2_1*h2ld_t3+h3_1*h3ld_t3+h1_1*h1ld_t3+p6_1*p6ld_t3+p4_1*p4ld_t3]+=tlocal6;
		}
		else if(thread_y+T2*0<total_y) {
		t3d[h2_0*h2ld_t3+h3_1*h3ld_t3+h1_0*h1ld_t3+p6_1*p6ld_t3+p4_0*p4ld_t3]+=tlocal5;
		}
	}
	if(thread_x+T1*2<total_x){
		if(thread_y+T2*3<total_y) {
		t3d[h2_0*h2ld_t3+h3_2*h3ld_t3+h1_0*h1ld_t3+p6_2*p6ld_t3+p4_0*p4ld_t3]+=tlocal9;
		t3d[h2_1*h2ld_t3+h3_2*h3ld_t3+h1_1*h1ld_t3+p6_2*p6ld_t3+p4_1*p4ld_t3]+=tlocal10;
		t3d[h2_2*h2ld_t3+h3_2*h3ld_t3+h1_2*h1ld_t3+p6_2*p6ld_t3+p4_2*p4ld_t3]+=tlocal11;
		t3d[h2_3*h2ld_t3+h3_2*h3ld_t3+h1_3*h1ld_t3+p6_2*p6ld_t3+p4_3*p4ld_t3]+=tlocal12;
		}
		else if(thread_y+T2*2<total_y) {
		t3d[h2_0*h2ld_t3+h3_2*h3ld_t3+h1_0*h1ld_t3+p6_2*p6ld_t3+p4_0*p4ld_t3]+=tlocal9;
		t3d[h2_1*h2ld_t3+h3_2*h3ld_t3+h1_1*h1ld_t3+p6_2*p6ld_t3+p4_1*p4ld_t3]+=tlocal10;
		t3d[h2_2*h2ld_t3+h3_2*h3ld_t3+h1_2*h1ld_t3+p6_2*p6ld_t3+p4_2*p4ld_t3]+=tlocal11;
		}
		else if(thread_y+T2*1<total_y) {
		t3d[h2_0*h2ld_t3+h3_2*h3ld_t3+h1_0*h1ld_t3+p6_2*p6ld_t3+p4_0*p4ld_t3]+=tlocal9;
		t3d[h2_1*h2ld_t3+h3_2*h3ld_t3+h1_1*h1ld_t3+p6_2*p6ld_t3+p4_1*p4ld_t3]+=tlocal10;
		}
		else if(thread_y+T2*0<total_y) {
		t3d[h2_0*h2ld_t3+h3_2*h3ld_t3+h1_0*h1ld_t3+p6_2*p6ld_t3+p4_0*p4ld_t3]+=tlocal9;
		}
	}
	if(thread_x+T1*3<total_x){
		if(thread_y+T2*3<total_y) {
		t3d[h2_0*h2ld_t3+h3_3*h3ld_t3+h1_0*h1ld_t3+p6_3*p6ld_t3+p4_0*p4ld_t3]+=tlocal13;
		t3d[h2_1*h2ld_t3+h3_3*h3ld_t3+h1_1*h1ld_t3+p6_3*p6ld_t3+p4_1*p4ld_t3]+=tlocal14;
		t3d[h2_2*h2ld_t3+h3_3*h3ld_t3+h1_2*h1ld_t3+p6_3*p6ld_t3+p4_2*p4ld_t3]+=tlocal15;
		t3d[h2_3*h2ld_t3+h3_3*h3ld_t3+h1_3*h1ld_t3+p6_3*p6ld_t3+p4_3*p4ld_t3]+=tlocal16;
		}
		else if(thread_y+T2*2<total_y) {
		t3d[h2_0*h2ld_t3+h3_3*h3ld_t3+h1_0*h1ld_t3+p6_3*p6ld_t3+p4_0*p4ld_t3]+=tlocal13;
		t3d[h2_1*h2ld_t3+h3_3*h3ld_t3+h1_1*h1ld_t3+p6_3*p6ld_t3+p4_1*p4ld_t3]+=tlocal14;
		t3d[h2_2*h2ld_t3+h3_3*h3ld_t3+h1_2*h1ld_t3+p6_3*p6ld_t3+p4_2*p4ld_t3]+=tlocal15;
		}
		else if(thread_y+T2*1<total_y) {
		t3d[h2_0*h2ld_t3+h3_3*h3ld_t3+h1_0*h1ld_t3+p6_3*p6ld_t3+p4_0*p4ld_t3]+=tlocal13;
		t3d[h2_1*h2ld_t3+h3_3*h3ld_t3+h1_1*h1ld_t3+p6_3*p6ld_t3+p4_1*p4ld_t3]+=tlocal14;
		}
		else if(thread_y+T2*0<total_y) {
		t3d[h2_0*h2ld_t3+h3_3*h3ld_t3+h1_0*h1ld_t3+p6_3*p6ld_t3+p4_0*p4ld_t3]+=tlocal13;
		}
	}
	__syncthreads();
}
void ma_sd_t_d2_3_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d, size_t p7d, double *t3, double *t2, double *v2) 
{
	p6d=p6d*p5d;
	size_t p7ld_t2,p4ld_t2,h1ld_t2,h2ld_t2,p7ld_v2,h3ld_v2,p6ld_v2,h2ld_t3,h3ld_t3,h1ld_t3,p6ld_t3,p4ld_t3;
	size_t size_t3,size_block_t3,size_el_block_t3,size_t2,size_v2;
	cudaStream_t *streams;
	size_t nstreams,i;
	double *t2_d,*v2_d;
	size_t3=h2d*h3d*h1d*p6d*p4d*sizeof(double);
	size_t2=p7d*p4d*h1d*h2d*sizeof(double);
	size_v2=p7d*h3d*p6d*sizeof(double);
	cudaFuncSetCacheConfig(ma_sd_t_d2_3_kernel, cudaFuncCachePreferShared);
	nstreams=1;
	size_block_t3=size_t3/nstreams;
	size_el_block_t3=size_block_t3/sizeof(double);
	//t3d=(double*)getGpuMem(size_t3);
	t2_d=(double*)getGpuMem(size_t2);
	v2_d=(double*)getGpuMem(size_v2);
	streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
	assert(streams!= NULL);
	for(i=0;i<nstreams;++i) {
		CUDA_SAFE(cudaStreamCreate(&streams[i])) ;
	}
	CUDA_SAFE(cudaMemcpy(t2_d,t2,size_t2,cudaMemcpyHostToDevice));
	CUDA_SAFE(cudaMemcpy(v2_d,v2,size_v2,cudaMemcpyHostToDevice));
	p7ld_t2=1;
	p4ld_t2=p7d;
	h1ld_t2=p4d*p7d;
	h2ld_t2=h1d*p4d*p7d;
	p7ld_v2=1;
	h3ld_v2=p7d;
	p6ld_v2=h3d*p7d;
	h2ld_t3=1;
	h3ld_t3=h2d;
	h1ld_t3=h3d*h2d;
	p6ld_t3=h1d*h3d*h2d;
	p4ld_t3=p6d*h1d*h3d*h2d;
	size_t total_x = h3d*p6d;
	size_t total_y = p4d*h1d*h2d;
	dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
	for(i=0;i<nstreams;++i){
		ma_sd_t_d2_3_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(h1d,h2d,h3d,p4d,p6d,p7d,p7ld_t2,p4ld_t2,h1ld_t2,h2ld_t2,p7ld_v2,h3ld_v2,p6ld_v2,h2ld_t3,h3ld_t3,h1ld_t3,p6ld_t3,p4ld_t3,ma_t3_d,t2_d,v2_d,i,total_x,total_y);
		CHECK_ERR("Kernel execution failed");
	}
	cudaThreadSynchronize();
	for(i=0;i<nstreams;++i){
		cudaStreamDestroy(streams[i]);}
	//  freeGpuMem(ma_t3_d);
	freeGpuMem(t2_d);
	freeGpuMem(v2_d);
	free(streams);
}
#undef T1
#undef T2
#undef Tcomm
void ma_sd_t_d2_3_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* p4d, Integer* p5d, Integer* p6d, Integer* p7d, double *t3, double *t2, double *v2) 
{
	ma_sd_t_d2_3_cuda((int)*h1d,(int)*h2d,(int)*h3d,(int)*p4d,(int)*p5d,(int)*p6d,(int)*p7d,t3,t2,v2);
}
/*----------------------------------------------------------------------*
*t3[h3,h2,h1,p6,p4,p5] += t2[p7,p4,h1,h2] * v2[p7,h3,p6,p5]
*----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void ma_sd_t_d2_4_kernel(size_t h1d,size_t h2d,size_t h3d,size_t p4d,size_t p5d,size_t p6d,size_t p7d,
	size_t p7ld_t2,size_t p4ld_t2,size_t h1ld_t2,size_t h2ld_t2,size_t p7ld_v2,size_t h3ld_v2,size_t p6ld_v2,size_t p5ld_v2,
	size_t h3ld_t3,size_t h2ld_t3,size_t h1ld_t3,size_t p6ld_t3,size_t p4ld_t3,size_t p5ld_t3,
	double *t3d, double *t2_d, double *v2_d,size_t unused_idx, size_t total_x, size_t total_y) 
{
	size_t h1_0,h1_1,h1_2,h1_3,h2_0,h2_1,h2_2,h2_3,h3_0,h3_1,h3_2,h3_3,p4_0,p4_1,p4_2,p4_3,p5_0,p5_1,p5_2,p5_3,p6_0,p6_1,p6_2,p6_3,p7;
	double a1,b1;
	double a2,b2;
	double a3,b3;
	double a4,b4;
	size_t in1_idxl,in2_idxl,p7l,p7T;
	__shared__ double t2_shm[4*T1][Tcomm];
	__shared__ double v2_shm[Tcomm][4*T2];
	size_t rest_x=blockIdx.x;
	size_t rest_y=blockIdx.y;
	size_t thread_x = T2*4 * rest_x + threadIdx.x;
	size_t thread_y = T1*4 * rest_y + threadIdx.y;
	in1_idxl=threadIdx.y;
	in2_idxl=threadIdx.x ;
	double tlocal1=0;
	double tlocal2=0;
	double tlocal3=0;
	double tlocal4=0;
	double tlocal5=0;
	double tlocal6=0;
	double tlocal7=0;
	double tlocal8=0;
	double tlocal9=0;
	double tlocal10=0;
	double tlocal11=0;
	double tlocal12=0;
	double tlocal13=0;
	double tlocal14=0;
	double tlocal15=0;
	double tlocal16=0;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
	h3_0=rest_x%h3d;
	rest_x=rest_x/h3d;
	h2_0=rest_y%h2d;
	rest_y=rest_y/h2d;
	h1_0=rest_y%h1d;
	rest_y=rest_y/h1d;
	p6_0=rest_x%p6d;
	rest_x=rest_x/p6d;
	p4_0=rest_y;
	p5_0=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
	h3_1=rest_x%h3d;
	rest_x=rest_x/h3d;
	h2_1=rest_y%h2d;
	rest_y=rest_y/h2d;
	h1_1=rest_y%h1d;
	rest_y=rest_y/h1d;
	p6_1=rest_x%p6d;
	rest_x=rest_x/p6d;
	p4_1=rest_y;
	p5_1=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
	h3_2=rest_x%h3d;
	rest_x=rest_x/h3d;
	h2_2=rest_y%h2d;
	rest_y=rest_y/h2d;
	h1_2=rest_y%h1d;
	rest_y=rest_y/h1d;
	p6_2=rest_x%p6d;
	rest_x=rest_x/p6d;
	p4_2=rest_y;
	p5_2=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
	h3_3=rest_x%h3d;
	rest_x=rest_x/h3d;
	h2_3=rest_y%h2d;
	rest_y=rest_y/h2d;
	h1_3=rest_y%h1d;
	rest_y=rest_y/h1d;
	p6_3=rest_x%p6d;
	rest_x=rest_x/p6d;
	p4_3=rest_y;
	p5_3=rest_x;
	size_t t2_d_off, v2_d_off;for(p7T=0;p7T<p7d;p7T+=Tcomm){size_t p7l_hi;
		p7l_hi = MIN(Tcomm+p7T,p7d)-p7T;
		t2_d_off=p4_0*p4ld_t2+h1_0*h1ld_t2+h2_0*h2ld_t2;
		v2_d_off=h3_0*h3ld_v2+p6_0*p6ld_v2+p5_0*p5ld_v2;
		if(thread_y+T1*0<total_y)for(p7l=threadIdx.x;p7l<p7l_hi;p7l+=blockDim.x){
		p7=p7l+p7T;
		t2_shm[in1_idxl+T1*0][p7l] = t2_d[t2_d_off+p7*p7ld_t2];
		}
		if(thread_x+T1*0<total_x)for(p7l=threadIdx.y;p7l<p7l_hi;p7l+=blockDim.y){
		p7=p7l+p7T;
		v2_shm[p7l][in2_idxl+T1*0] = v2_d[v2_d_off+p7*p7ld_v2];
		}
		t2_d_off=p4_1*p4ld_t2+h1_1*h1ld_t2+h2_1*h2ld_t2;
		v2_d_off=h3_1*h3ld_v2+p6_1*p6ld_v2+p5_1*p5ld_v2;
		if(thread_y+T1*1<total_y)for(p7l=threadIdx.x;p7l<p7l_hi;p7l+=blockDim.x){
		p7=p7l+p7T;
		t2_shm[in1_idxl+T1*1][p7l] = t2_d[t2_d_off+p7*p7ld_t2];
		}
		if(thread_x+T1*1<total_x)for(p7l=threadIdx.y;p7l<p7l_hi;p7l+=blockDim.y){
		p7=p7l+p7T;
		v2_shm[p7l][in2_idxl+T1*1] = v2_d[v2_d_off+p7*p7ld_v2];
		}
		t2_d_off=p4_2*p4ld_t2+h1_2*h1ld_t2+h2_2*h2ld_t2;
		v2_d_off=h3_2*h3ld_v2+p6_2*p6ld_v2+p5_2*p5ld_v2;
		if(thread_y+T1*2<total_y)for(p7l=threadIdx.x;p7l<p7l_hi;p7l+=blockDim.x){
		p7=p7l+p7T;
		t2_shm[in1_idxl+T1*2][p7l] = t2_d[t2_d_off+p7*p7ld_t2];
		}
		if(thread_x+T1*2<total_x)for(p7l=threadIdx.y;p7l<p7l_hi;p7l+=blockDim.y){
		p7=p7l+p7T;
		v2_shm[p7l][in2_idxl+T1*2] = v2_d[v2_d_off+p7*p7ld_v2];
		}
		t2_d_off=p4_3*p4ld_t2+h1_3*h1ld_t2+h2_3*h2ld_t2;
		v2_d_off=h3_3*h3ld_v2+p6_3*p6ld_v2+p5_3*p5ld_v2;
		if(thread_y+T1*3<total_y)for(p7l=threadIdx.x;p7l<p7l_hi;p7l+=blockDim.x){
		p7=p7l+p7T;
		t2_shm[in1_idxl+T1*3][p7l] = t2_d[t2_d_off+p7*p7ld_t2];
		}
		if(thread_x+T1*3<total_x)for(p7l=threadIdx.y;p7l<p7l_hi;p7l+=blockDim.y){
		p7=p7l+p7T;
		v2_shm[p7l][in2_idxl+T1*3] = v2_d[v2_d_off+p7*p7ld_v2];
		}
		__syncthreads();
		for(p7l=0;p7l<p7l_hi;++p7l){
		a1=t2_shm[in1_idxl+T1*0][p7l];
		a2=t2_shm[in1_idxl+T1*1][p7l];
		a3=t2_shm[in1_idxl+T1*2][p7l];
		a4=t2_shm[in1_idxl+T1*3][p7l];
		b1=v2_shm[p7l][in2_idxl+T2*0];
		b2=v2_shm[p7l][in2_idxl+T2*1];
		b3=v2_shm[p7l][in2_idxl+T2*2];
		b4=v2_shm[p7l][in2_idxl+T2*3];
		tlocal1+=a1*b1;
		tlocal2+=a2*b1;
		tlocal3+=a3*b1;
		tlocal4+=a4*b1;
		tlocal5+=a1*b2;
		tlocal6+=a2*b2;
		tlocal7+=a3*b2;
		tlocal8+=a4*b2;
		tlocal9+=a1*b3;
		tlocal10+=a2*b3;
		tlocal11+=a3*b3;
		tlocal12+=a4*b3;
		tlocal13+=a1*b4;
		tlocal14+=a2*b4;
		tlocal15+=a3*b4;
		tlocal16+=a4*b4;
		}
		__syncthreads();
	}
	if(thread_x+T1*0<total_x){
		if(thread_y+T2*3<total_y) {
		t3d[h3_0*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p6_0*p6ld_t3+p4_0*p4ld_t3+p5_0*p5ld_t3]+=tlocal1;
		t3d[h3_0*h3ld_t3+h2_1*h2ld_t3+h1_1*h1ld_t3+p6_0*p6ld_t3+p4_1*p4ld_t3+p5_0*p5ld_t3]+=tlocal2;
		t3d[h3_0*h3ld_t3+h2_2*h2ld_t3+h1_2*h1ld_t3+p6_0*p6ld_t3+p4_2*p4ld_t3+p5_0*p5ld_t3]+=tlocal3;
		t3d[h3_0*h3ld_t3+h2_3*h2ld_t3+h1_3*h1ld_t3+p6_0*p6ld_t3+p4_3*p4ld_t3+p5_0*p5ld_t3]+=tlocal4;
		}
		else if(thread_y+T2*2<total_y) {
		t3d[h3_0*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p6_0*p6ld_t3+p4_0*p4ld_t3+p5_0*p5ld_t3]+=tlocal1;
		t3d[h3_0*h3ld_t3+h2_1*h2ld_t3+h1_1*h1ld_t3+p6_0*p6ld_t3+p4_1*p4ld_t3+p5_0*p5ld_t3]+=tlocal2;
		t3d[h3_0*h3ld_t3+h2_2*h2ld_t3+h1_2*h1ld_t3+p6_0*p6ld_t3+p4_2*p4ld_t3+p5_0*p5ld_t3]+=tlocal3;
		}
		else if(thread_y+T2*1<total_y) {
		t3d[h3_0*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p6_0*p6ld_t3+p4_0*p4ld_t3+p5_0*p5ld_t3]+=tlocal1;
		t3d[h3_0*h3ld_t3+h2_1*h2ld_t3+h1_1*h1ld_t3+p6_0*p6ld_t3+p4_1*p4ld_t3+p5_0*p5ld_t3]+=tlocal2;
		}
		else if(thread_y+T2*0<total_y) {
		t3d[h3_0*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p6_0*p6ld_t3+p4_0*p4ld_t3+p5_0*p5ld_t3]+=tlocal1;
		}
	}
	if(thread_x+T1*1<total_x){
		if(thread_y+T2*3<total_y) {
		t3d[h3_1*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p6_1*p6ld_t3+p4_0*p4ld_t3+p5_1*p5ld_t3]+=tlocal5;
		t3d[h3_1*h3ld_t3+h2_1*h2ld_t3+h1_1*h1ld_t3+p6_1*p6ld_t3+p4_1*p4ld_t3+p5_1*p5ld_t3]+=tlocal6;
		t3d[h3_1*h3ld_t3+h2_2*h2ld_t3+h1_2*h1ld_t3+p6_1*p6ld_t3+p4_2*p4ld_t3+p5_1*p5ld_t3]+=tlocal7;
		t3d[h3_1*h3ld_t3+h2_3*h2ld_t3+h1_3*h1ld_t3+p6_1*p6ld_t3+p4_3*p4ld_t3+p5_1*p5ld_t3]+=tlocal8;
		}
		else if(thread_y+T2*2<total_y) {
		t3d[h3_1*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p6_1*p6ld_t3+p4_0*p4ld_t3+p5_1*p5ld_t3]+=tlocal5;
		t3d[h3_1*h3ld_t3+h2_1*h2ld_t3+h1_1*h1ld_t3+p6_1*p6ld_t3+p4_1*p4ld_t3+p5_1*p5ld_t3]+=tlocal6;
		t3d[h3_1*h3ld_t3+h2_2*h2ld_t3+h1_2*h1ld_t3+p6_1*p6ld_t3+p4_2*p4ld_t3+p5_1*p5ld_t3]+=tlocal7;
		}
		else if(thread_y+T2*1<total_y) {
		t3d[h3_1*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p6_1*p6ld_t3+p4_0*p4ld_t3+p5_1*p5ld_t3]+=tlocal5;
		t3d[h3_1*h3ld_t3+h2_1*h2ld_t3+h1_1*h1ld_t3+p6_1*p6ld_t3+p4_1*p4ld_t3+p5_1*p5ld_t3]+=tlocal6;
		}
		else if(thread_y+T2*0<total_y) {
		t3d[h3_1*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p6_1*p6ld_t3+p4_0*p4ld_t3+p5_1*p5ld_t3]+=tlocal5;
		}
	}
	if(thread_x+T1*2<total_x){
		if(thread_y+T2*3<total_y) {
		t3d[h3_2*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p6_2*p6ld_t3+p4_0*p4ld_t3+p5_2*p5ld_t3]+=tlocal9;
		t3d[h3_2*h3ld_t3+h2_1*h2ld_t3+h1_1*h1ld_t3+p6_2*p6ld_t3+p4_1*p4ld_t3+p5_2*p5ld_t3]+=tlocal10;
		t3d[h3_2*h3ld_t3+h2_2*h2ld_t3+h1_2*h1ld_t3+p6_2*p6ld_t3+p4_2*p4ld_t3+p5_2*p5ld_t3]+=tlocal11;
		t3d[h3_2*h3ld_t3+h2_3*h2ld_t3+h1_3*h1ld_t3+p6_2*p6ld_t3+p4_3*p4ld_t3+p5_2*p5ld_t3]+=tlocal12;
		}
		else if(thread_y+T2*2<total_y) {
		t3d[h3_2*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p6_2*p6ld_t3+p4_0*p4ld_t3+p5_2*p5ld_t3]+=tlocal9;
		t3d[h3_2*h3ld_t3+h2_1*h2ld_t3+h1_1*h1ld_t3+p6_2*p6ld_t3+p4_1*p4ld_t3+p5_2*p5ld_t3]+=tlocal10;
		t3d[h3_2*h3ld_t3+h2_2*h2ld_t3+h1_2*h1ld_t3+p6_2*p6ld_t3+p4_2*p4ld_t3+p5_2*p5ld_t3]+=tlocal11;
		}
		else if(thread_y+T2*1<total_y) {
		t3d[h3_2*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p6_2*p6ld_t3+p4_0*p4ld_t3+p5_2*p5ld_t3]+=tlocal9;
		t3d[h3_2*h3ld_t3+h2_1*h2ld_t3+h1_1*h1ld_t3+p6_2*p6ld_t3+p4_1*p4ld_t3+p5_2*p5ld_t3]+=tlocal10;
		}
		else if(thread_y+T2*0<total_y) {
		t3d[h3_2*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p6_2*p6ld_t3+p4_0*p4ld_t3+p5_2*p5ld_t3]+=tlocal9;
		}
	}
	if(thread_x+T1*3<total_x){
		if(thread_y+T2*3<total_y) {
		t3d[h3_3*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p6_3*p6ld_t3+p4_0*p4ld_t3+p5_3*p5ld_t3]+=tlocal13;
		t3d[h3_3*h3ld_t3+h2_1*h2ld_t3+h1_1*h1ld_t3+p6_3*p6ld_t3+p4_1*p4ld_t3+p5_3*p5ld_t3]+=tlocal14;
		t3d[h3_3*h3ld_t3+h2_2*h2ld_t3+h1_2*h1ld_t3+p6_3*p6ld_t3+p4_2*p4ld_t3+p5_3*p5ld_t3]+=tlocal15;
		t3d[h3_3*h3ld_t3+h2_3*h2ld_t3+h1_3*h1ld_t3+p6_3*p6ld_t3+p4_3*p4ld_t3+p5_3*p5ld_t3]+=tlocal16;
		}
		else if(thread_y+T2*2<total_y) {
		t3d[h3_3*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p6_3*p6ld_t3+p4_0*p4ld_t3+p5_3*p5ld_t3]+=tlocal13;
		t3d[h3_3*h3ld_t3+h2_1*h2ld_t3+h1_1*h1ld_t3+p6_3*p6ld_t3+p4_1*p4ld_t3+p5_3*p5ld_t3]+=tlocal14;
		t3d[h3_3*h3ld_t3+h2_2*h2ld_t3+h1_2*h1ld_t3+p6_3*p6ld_t3+p4_2*p4ld_t3+p5_3*p5ld_t3]+=tlocal15;
		}
		else if(thread_y+T2*1<total_y) {
		t3d[h3_3*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p6_3*p6ld_t3+p4_0*p4ld_t3+p5_3*p5ld_t3]+=tlocal13;
		t3d[h3_3*h3ld_t3+h2_1*h2ld_t3+h1_1*h1ld_t3+p6_3*p6ld_t3+p4_1*p4ld_t3+p5_3*p5ld_t3]+=tlocal14;
		}
		else if(thread_y+T2*0<total_y) {
		t3d[h3_3*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p6_3*p6ld_t3+p4_0*p4ld_t3+p5_3*p5ld_t3]+=tlocal13;
		}
	}
	__syncthreads();
}
void ma_sd_t_d2_4_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d, size_t p7d, double *t3, double *t2, double *v2) 
{
	size_t p7ld_t2,p4ld_t2,h1ld_t2,h2ld_t2,p7ld_v2,h3ld_v2,p6ld_v2,p5ld_v2,h3ld_t3,h2ld_t3,h1ld_t3,p6ld_t3,p4ld_t3,p5ld_t3;
	size_t size_t3,size_block_t3,size_el_block_t3,size_t2,size_v2;
	cudaStream_t *streams;
	size_t nstreams,i;
	double *t2_d,*v2_d;
	size_t3=h3d*h2d*h1d*p6d*p4d*p5d*sizeof(double);
	size_t2=p7d*p4d*h1d*h2d*sizeof(double);
	size_v2=p7d*h3d*p6d*p5d*sizeof(double);
	cudaFuncSetCacheConfig(ma_sd_t_d2_4_kernel, cudaFuncCachePreferShared);
	nstreams=1;
	size_block_t3=size_t3/nstreams;
	size_el_block_t3=size_block_t3/sizeof(double);
	//t3d=(double*)getGpuMem(size_t3);
	t2_d=(double*)getGpuMem(size_t2);
	v2_d=(double*)getGpuMem(size_v2);
	streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
	assert(streams!= NULL);
	for(i=0;i<nstreams;++i) {
		CUDA_SAFE(cudaStreamCreate(&streams[i])) ;
	}
	CUDA_SAFE(cudaMemcpy(t2_d,t2,size_t2,cudaMemcpyHostToDevice));
	CUDA_SAFE(cudaMemcpy(v2_d,v2,size_v2,cudaMemcpyHostToDevice));
	p7ld_t2=1;
	p4ld_t2=p7d;
	h1ld_t2=p4d*p7d;
	h2ld_t2=h1d*p4d*p7d;
	p7ld_v2=1;
	h3ld_v2=p7d;
	p6ld_v2=h3d*p7d;
	p5ld_v2=p6d*h3d*p7d;
	h3ld_t3=1;
	h2ld_t3=h3d;
	h1ld_t3=h2d*h3d;
	p6ld_t3=h1d*h2d*h3d;
	p4ld_t3=p6d*h1d*h2d*h3d;
	p5ld_t3=p4d*p6d*h1d*h2d*h3d;
	size_t total_x = h3d*p6d*p5d;
	size_t total_y = p4d*h1d*h2d;
	dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
	for(i=0;i<nstreams;++i){
		ma_sd_t_d2_4_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(h1d,h2d,h3d,p4d,p5d,p6d,p7d,p7ld_t2,p4ld_t2,h1ld_t2,h2ld_t2,p7ld_v2,h3ld_v2,p6ld_v2,p5ld_v2,h3ld_t3,h2ld_t3,h1ld_t3,p6ld_t3,p4ld_t3,p5ld_t3,ma_t3_d,t2_d,v2_d,i,total_x,total_y);
		CHECK_ERR("Kernel execution failed");
	}
	cudaThreadSynchronize();
	for(i=0;i<nstreams;++i){
		cudaStreamDestroy(streams[i]);}
	//freeGpuMem(t3d);
	freeGpuMem(t2_d);
	freeGpuMem(v2_d);
	free(streams);
}
#undef T1
#undef T2
#undef Tcomm
void ma_sd_t_d2_4_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* p4d, Integer* p5d, Integer* p6d, Integer* p7d, double *t3, double *t2, double *v2) 
{
	ma_sd_t_d2_4_cuda((int)*h1d,(int)*h2d,(int)*h3d,(int)*p4d,(int)*p5d,(int)*p6d,(int)*p7d,t3,t2,v2);
}
/*----------------------------------------------------------------------*
*t3[h2,h1,h3,p4,p5] += t2[p7,p4,h1,h2] * v2[p7,h3,p5]
*----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void ma_sd_t_d2_5_kernel(size_t h1d,size_t h2d,size_t h3d,size_t p4d,size_t p5d,size_t p7d,
	size_t p7ld_t2,size_t p4ld_t2,size_t h1ld_t2,size_t h2ld_t2,size_t p7ld_v2,size_t h3ld_v2,size_t p5ld_v2,
	size_t h2ld_t3,size_t h1ld_t3,size_t h3ld_t3,size_t p4ld_t3,size_t p5ld_t3,
	double *t3d, double *t2_d, double *v2_d,size_t unused_idx, size_t total_x, size_t total_y) 
{
	size_t h1_0,h1_1,h1_2,h1_3,h2_0,h2_1,h2_2,h2_3,h3_0,h3_1,h3_2,h3_3,p4_0,p4_1,p4_2,p4_3,p5_0,p5_1,p5_2,p5_3,p7;
	double a1,b1;
	double a2,b2;
	double a3,b3;
	double a4,b4;
	size_t in1_idxl,in2_idxl,p7l,p7T;
	__shared__ double t2_shm[4*T1][Tcomm];
	__shared__ double v2_shm[Tcomm][4*T2];
	size_t rest_x=blockIdx.x;
	size_t rest_y=blockIdx.y;
	size_t thread_x = T2*4 * rest_x + threadIdx.x;
	size_t thread_y = T1*4 * rest_y + threadIdx.y;
	in1_idxl=threadIdx.y;
	in2_idxl=threadIdx.x ;
	double tlocal1=0;
	double tlocal2=0;
	double tlocal3=0;
	double tlocal4=0;
	double tlocal5=0;
	double tlocal6=0;
	double tlocal7=0;
	double tlocal8=0;
	double tlocal9=0;
	double tlocal10=0;
	double tlocal11=0;
	double tlocal12=0;
	double tlocal13=0;
	double tlocal14=0;
	double tlocal15=0;
	double tlocal16=0;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
	h2_0=rest_y%h2d;
	rest_y=rest_y/h2d;
	h1_0=rest_y%h1d;
	rest_y=rest_y/h1d;
	h3_0=rest_x%h3d;
	rest_x=rest_x/h3d;
	p4_0=rest_y;
	p5_0=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
	h2_1=rest_y%h2d;
	rest_y=rest_y/h2d;
	h1_1=rest_y%h1d;
	rest_y=rest_y/h1d;
	h3_1=rest_x%h3d;
	rest_x=rest_x/h3d;
	p4_1=rest_y;
	p5_1=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
	h2_2=rest_y%h2d;
	rest_y=rest_y/h2d;
	h1_2=rest_y%h1d;
	rest_y=rest_y/h1d;
	h3_2=rest_x%h3d;
	rest_x=rest_x/h3d;
	p4_2=rest_y;
	p5_2=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
	h2_3=rest_y%h2d;
	rest_y=rest_y/h2d;
	h1_3=rest_y%h1d;
	rest_y=rest_y/h1d;
	h3_3=rest_x%h3d;
	rest_x=rest_x/h3d;
	p4_3=rest_y;
	p5_3=rest_x;
	size_t t2_d_off, v2_d_off;for(p7T=0;p7T<p7d;p7T+=Tcomm){size_t p7l_hi;
		p7l_hi = MIN(Tcomm+p7T,p7d)-p7T;
		t2_d_off=p4_0*p4ld_t2+h1_0*h1ld_t2+h2_0*h2ld_t2;
		v2_d_off=h3_0*h3ld_v2+p5_0*p5ld_v2;
		if(thread_y+T1*0<total_y)for(p7l=threadIdx.x;p7l<p7l_hi;p7l+=blockDim.x){
		p7=p7l+p7T;
		t2_shm[in1_idxl+T1*0][p7l] = t2_d[t2_d_off+p7*p7ld_t2];
		}
		if(thread_x+T1*0<total_x)for(p7l=threadIdx.y;p7l<p7l_hi;p7l+=blockDim.y){
		p7=p7l+p7T;
		v2_shm[p7l][in2_idxl+T1*0] = v2_d[v2_d_off+p7*p7ld_v2];
		}
		t2_d_off=p4_1*p4ld_t2+h1_1*h1ld_t2+h2_1*h2ld_t2;
		v2_d_off=h3_1*h3ld_v2+p5_1*p5ld_v2;
		if(thread_y+T1*1<total_y)for(p7l=threadIdx.x;p7l<p7l_hi;p7l+=blockDim.x){
		p7=p7l+p7T;
		t2_shm[in1_idxl+T1*1][p7l] = t2_d[t2_d_off+p7*p7ld_t2];
		}
		if(thread_x+T1*1<total_x)for(p7l=threadIdx.y;p7l<p7l_hi;p7l+=blockDim.y){
		p7=p7l+p7T;
		v2_shm[p7l][in2_idxl+T1*1] = v2_d[v2_d_off+p7*p7ld_v2];
		}
		t2_d_off=p4_2*p4ld_t2+h1_2*h1ld_t2+h2_2*h2ld_t2;
		v2_d_off=h3_2*h3ld_v2+p5_2*p5ld_v2;
		if(thread_y+T1*2<total_y)for(p7l=threadIdx.x;p7l<p7l_hi;p7l+=blockDim.x){
		p7=p7l+p7T;
		t2_shm[in1_idxl+T1*2][p7l] = t2_d[t2_d_off+p7*p7ld_t2];
		}
		if(thread_x+T1*2<total_x)for(p7l=threadIdx.y;p7l<p7l_hi;p7l+=blockDim.y){
		p7=p7l+p7T;
		v2_shm[p7l][in2_idxl+T1*2] = v2_d[v2_d_off+p7*p7ld_v2];
		}
		t2_d_off=p4_3*p4ld_t2+h1_3*h1ld_t2+h2_3*h2ld_t2;
		v2_d_off=h3_3*h3ld_v2+p5_3*p5ld_v2;
		if(thread_y+T1*3<total_y)for(p7l=threadIdx.x;p7l<p7l_hi;p7l+=blockDim.x){
		p7=p7l+p7T;
		t2_shm[in1_idxl+T1*3][p7l] = t2_d[t2_d_off+p7*p7ld_t2];
		}
		if(thread_x+T1*3<total_x)for(p7l=threadIdx.y;p7l<p7l_hi;p7l+=blockDim.y){
		p7=p7l+p7T;
		v2_shm[p7l][in2_idxl+T1*3] = v2_d[v2_d_off+p7*p7ld_v2];
		}
		__syncthreads();
		for(p7l=0;p7l<p7l_hi;++p7l){
		a1=t2_shm[in1_idxl+T1*0][p7l];
		a2=t2_shm[in1_idxl+T1*1][p7l];
		a3=t2_shm[in1_idxl+T1*2][p7l];
		a4=t2_shm[in1_idxl+T1*3][p7l];
		b1=v2_shm[p7l][in2_idxl+T2*0];
		b2=v2_shm[p7l][in2_idxl+T2*1];
		b3=v2_shm[p7l][in2_idxl+T2*2];
		b4=v2_shm[p7l][in2_idxl+T2*3];
		tlocal1+=a1*b1;
		tlocal2+=a2*b1;
		tlocal3+=a3*b1;
		tlocal4+=a4*b1;
		tlocal5+=a1*b2;
		tlocal6+=a2*b2;
		tlocal7+=a3*b2;
		tlocal8+=a4*b2;
		tlocal9+=a1*b3;
		tlocal10+=a2*b3;
		tlocal11+=a3*b3;
		tlocal12+=a4*b3;
		tlocal13+=a1*b4;
		tlocal14+=a2*b4;
		tlocal15+=a3*b4;
		tlocal16+=a4*b4;
		}
		__syncthreads();
	}
	if(thread_x+T1*0<total_x){
		if(thread_y+T2*3<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_0*h3ld_t3+p4_0*p4ld_t3+p5_0*p5ld_t3]+=tlocal1;
		t3d[h2_1*h2ld_t3+h1_1*h1ld_t3+h3_0*h3ld_t3+p4_1*p4ld_t3+p5_0*p5ld_t3]+=tlocal2;
		t3d[h2_2*h2ld_t3+h1_2*h1ld_t3+h3_0*h3ld_t3+p4_2*p4ld_t3+p5_0*p5ld_t3]+=tlocal3;
		t3d[h2_3*h2ld_t3+h1_3*h1ld_t3+h3_0*h3ld_t3+p4_3*p4ld_t3+p5_0*p5ld_t3]+=tlocal4;
		}
		else if(thread_y+T2*2<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_0*h3ld_t3+p4_0*p4ld_t3+p5_0*p5ld_t3]+=tlocal1;
		t3d[h2_1*h2ld_t3+h1_1*h1ld_t3+h3_0*h3ld_t3+p4_1*p4ld_t3+p5_0*p5ld_t3]+=tlocal2;
		t3d[h2_2*h2ld_t3+h1_2*h1ld_t3+h3_0*h3ld_t3+p4_2*p4ld_t3+p5_0*p5ld_t3]+=tlocal3;
		}
		else if(thread_y+T2*1<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_0*h3ld_t3+p4_0*p4ld_t3+p5_0*p5ld_t3]+=tlocal1;
		t3d[h2_1*h2ld_t3+h1_1*h1ld_t3+h3_0*h3ld_t3+p4_1*p4ld_t3+p5_0*p5ld_t3]+=tlocal2;
		}
		else if(thread_y+T2*0<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_0*h3ld_t3+p4_0*p4ld_t3+p5_0*p5ld_t3]+=tlocal1;
		}
	}
	if(thread_x+T1*1<total_x){
		if(thread_y+T2*3<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_1*h3ld_t3+p4_0*p4ld_t3+p5_1*p5ld_t3]+=tlocal5;
		t3d[h2_1*h2ld_t3+h1_1*h1ld_t3+h3_1*h3ld_t3+p4_1*p4ld_t3+p5_1*p5ld_t3]+=tlocal6;
		t3d[h2_2*h2ld_t3+h1_2*h1ld_t3+h3_1*h3ld_t3+p4_2*p4ld_t3+p5_1*p5ld_t3]+=tlocal7;
		t3d[h2_3*h2ld_t3+h1_3*h1ld_t3+h3_1*h3ld_t3+p4_3*p4ld_t3+p5_1*p5ld_t3]+=tlocal8;
		}
		else if(thread_y+T2*2<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_1*h3ld_t3+p4_0*p4ld_t3+p5_1*p5ld_t3]+=tlocal5;
		t3d[h2_1*h2ld_t3+h1_1*h1ld_t3+h3_1*h3ld_t3+p4_1*p4ld_t3+p5_1*p5ld_t3]+=tlocal6;
		t3d[h2_2*h2ld_t3+h1_2*h1ld_t3+h3_1*h3ld_t3+p4_2*p4ld_t3+p5_1*p5ld_t3]+=tlocal7;
		}
		else if(thread_y+T2*1<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_1*h3ld_t3+p4_0*p4ld_t3+p5_1*p5ld_t3]+=tlocal5;
		t3d[h2_1*h2ld_t3+h1_1*h1ld_t3+h3_1*h3ld_t3+p4_1*p4ld_t3+p5_1*p5ld_t3]+=tlocal6;
		}
		else if(thread_y+T2*0<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_1*h3ld_t3+p4_0*p4ld_t3+p5_1*p5ld_t3]+=tlocal5;
		}
	}
	if(thread_x+T1*2<total_x){
		if(thread_y+T2*3<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_2*h3ld_t3+p4_0*p4ld_t3+p5_2*p5ld_t3]+=tlocal9;
		t3d[h2_1*h2ld_t3+h1_1*h1ld_t3+h3_2*h3ld_t3+p4_1*p4ld_t3+p5_2*p5ld_t3]+=tlocal10;
		t3d[h2_2*h2ld_t3+h1_2*h1ld_t3+h3_2*h3ld_t3+p4_2*p4ld_t3+p5_2*p5ld_t3]+=tlocal11;
		t3d[h2_3*h2ld_t3+h1_3*h1ld_t3+h3_2*h3ld_t3+p4_3*p4ld_t3+p5_2*p5ld_t3]+=tlocal12;
		}
		else if(thread_y+T2*2<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_2*h3ld_t3+p4_0*p4ld_t3+p5_2*p5ld_t3]+=tlocal9;
		t3d[h2_1*h2ld_t3+h1_1*h1ld_t3+h3_2*h3ld_t3+p4_1*p4ld_t3+p5_2*p5ld_t3]+=tlocal10;
		t3d[h2_2*h2ld_t3+h1_2*h1ld_t3+h3_2*h3ld_t3+p4_2*p4ld_t3+p5_2*p5ld_t3]+=tlocal11;
		}
		else if(thread_y+T2*1<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_2*h3ld_t3+p4_0*p4ld_t3+p5_2*p5ld_t3]+=tlocal9;
		t3d[h2_1*h2ld_t3+h1_1*h1ld_t3+h3_2*h3ld_t3+p4_1*p4ld_t3+p5_2*p5ld_t3]+=tlocal10;
		}
		else if(thread_y+T2*0<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_2*h3ld_t3+p4_0*p4ld_t3+p5_2*p5ld_t3]+=tlocal9;
		}
	}
	if(thread_x+T1*3<total_x){
		if(thread_y+T2*3<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_3*h3ld_t3+p4_0*p4ld_t3+p5_3*p5ld_t3]+=tlocal13;
		t3d[h2_1*h2ld_t3+h1_1*h1ld_t3+h3_3*h3ld_t3+p4_1*p4ld_t3+p5_3*p5ld_t3]+=tlocal14;
		t3d[h2_2*h2ld_t3+h1_2*h1ld_t3+h3_3*h3ld_t3+p4_2*p4ld_t3+p5_3*p5ld_t3]+=tlocal15;
		t3d[h2_3*h2ld_t3+h1_3*h1ld_t3+h3_3*h3ld_t3+p4_3*p4ld_t3+p5_3*p5ld_t3]+=tlocal16;
		}
		else if(thread_y+T2*2<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_3*h3ld_t3+p4_0*p4ld_t3+p5_3*p5ld_t3]+=tlocal13;
		t3d[h2_1*h2ld_t3+h1_1*h1ld_t3+h3_3*h3ld_t3+p4_1*p4ld_t3+p5_3*p5ld_t3]+=tlocal14;
		t3d[h2_2*h2ld_t3+h1_2*h1ld_t3+h3_3*h3ld_t3+p4_2*p4ld_t3+p5_3*p5ld_t3]+=tlocal15;
		}
		else if(thread_y+T2*1<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_3*h3ld_t3+p4_0*p4ld_t3+p5_3*p5ld_t3]+=tlocal13;
		t3d[h2_1*h2ld_t3+h1_1*h1ld_t3+h3_3*h3ld_t3+p4_1*p4ld_t3+p5_3*p5ld_t3]+=tlocal14;
		}
		else if(thread_y+T2*0<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_3*h3ld_t3+p4_0*p4ld_t3+p5_3*p5ld_t3]+=tlocal13;
		}
	}
	__syncthreads();
}
void ma_sd_t_d2_5_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d, size_t p7d, double *t3, double *t2, double *v2) 
{
	h3d=h3d*p6d;
	size_t p7ld_t2,p4ld_t2,h1ld_t2,h2ld_t2,p7ld_v2,h3ld_v2,p5ld_v2,h2ld_t3,h1ld_t3,h3ld_t3,p4ld_t3,p5ld_t3;
	size_t size_t3,size_block_t3,size_el_block_t3,size_t2,size_v2;
	cudaStream_t *streams;
	size_t nstreams,i;
	double *t2_d,*v2_d;
	size_t3=h2d*h1d*h3d*p4d*p5d*sizeof(double);
	size_t2=p7d*p4d*h1d*h2d*sizeof(double);
	size_v2=p7d*h3d*p5d*sizeof(double);
	cudaFuncSetCacheConfig(ma_sd_t_d2_5_kernel, cudaFuncCachePreferShared);
	nstreams=1;
	size_block_t3=size_t3/nstreams;
	size_el_block_t3=size_block_t3/sizeof(double);
	//t3d=(double*)getGpuMem(size_t3);
	t2_d=(double*)getGpuMem(size_t2);
	v2_d=(double*)getGpuMem(size_v2);
	streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
	assert(streams!= NULL);
	for(i=0;i<nstreams;++i) {
		CUDA_SAFE(cudaStreamCreate(&streams[i])) ;
	}
	CUDA_SAFE(cudaMemcpy(t2_d,t2,size_t2,cudaMemcpyHostToDevice));
	CUDA_SAFE(cudaMemcpy(v2_d,v2,size_v2,cudaMemcpyHostToDevice));
	p7ld_t2=1;
	p4ld_t2=p7d;
	h1ld_t2=p4d*p7d;
	h2ld_t2=h1d*p4d*p7d;
	p7ld_v2=1;
	h3ld_v2=p7d;
	p5ld_v2=h3d*p7d;
	h2ld_t3=1;
	h1ld_t3=h2d;
	h3ld_t3=h1d*h2d;
	p4ld_t3=h3d*h1d*h2d;
	p5ld_t3=p4d*h3d*h1d*h2d;
	size_t total_x = h3d*p5d;
	size_t total_y = p4d*h1d*h2d;
	dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
	for(i=0;i<nstreams;++i){
		ma_sd_t_d2_5_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(h1d,h2d,h3d,p4d,p5d,p7d,p7ld_t2,p4ld_t2,h1ld_t2,h2ld_t2,p7ld_v2,h3ld_v2,p5ld_v2,h2ld_t3,h1ld_t3,h3ld_t3,p4ld_t3,p5ld_t3,ma_t3_d,t2_d,v2_d,i,total_x,total_y);
		CHECK_ERR("Kernel execution failed");
	}
	cudaThreadSynchronize();
	for(i=0;i<nstreams;++i){
		cudaStreamDestroy(streams[i]);}
	//freeGpuMem(t3d);
	freeGpuMem(t2_d);
	freeGpuMem(v2_d);
	free(streams);
}
#undef T1
#undef T2
#undef Tcomm
void ma_sd_t_d2_5_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* p4d, Integer* p5d, Integer* p6d, Integer* p7d, double *t3, double *t2, double *v2) 
{
	ma_sd_t_d2_5_cuda((int)*h1d,(int)*h2d,(int)*h3d,(int)*p4d,(int)*p5d,(int)*p6d,(int)*p7d,t3,t2,v2);
}
/*----------------------------------------------------------------------*
*t3[h2,h3,h1,p6,p4,p5] -= t2[p7,p4,h1,h2] * v2[p7,h3,p6,p5]
*----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void ma_sd_t_d2_6_kernel(size_t h1d,size_t h2d,size_t h3d,size_t p4d,size_t p5d,size_t p6d,size_t p7d,
	size_t p7ld_t2,size_t p4ld_t2,size_t h1ld_t2,size_t h2ld_t2,size_t p7ld_v2,size_t h3ld_v2,size_t p6ld_v2,size_t p5ld_v2,
	size_t h2ld_t3,size_t h3ld_t3,size_t h1ld_t3,size_t p6ld_t3,size_t p4ld_t3,size_t p5ld_t3,
	double *t3d, double *t2_d, double *v2_d,size_t unused_idx, size_t total_x, size_t total_y) 
{
	size_t h1_0,h1_1,h1_2,h1_3,h2_0,h2_1,h2_2,h2_3,h3_0,h3_1,h3_2,h3_3,p4_0,p4_1,p4_2,p4_3,p5_0,p5_1,p5_2,p5_3,p6_0,p6_1,p6_2,p6_3,p7;
	double a1,b1;
	double a2,b2;
	double a3,b3;
	double a4,b4;
	size_t in1_idxl,in2_idxl,p7l,p7T;
	__shared__ double t2_shm[4*T1][Tcomm];
	__shared__ double v2_shm[Tcomm][4*T2];
	size_t rest_x=blockIdx.x;
	size_t rest_y=blockIdx.y;
	size_t thread_x = T2*4 * rest_x + threadIdx.x;
	size_t thread_y = T1*4 * rest_y + threadIdx.y;
	in1_idxl=threadIdx.y;
	in2_idxl=threadIdx.x ;
	double tlocal1=0;
	double tlocal2=0;
	double tlocal3=0;
	double tlocal4=0;
	double tlocal5=0;
	double tlocal6=0;
	double tlocal7=0;
	double tlocal8=0;
	double tlocal9=0;
	double tlocal10=0;
	double tlocal11=0;
	double tlocal12=0;
	double tlocal13=0;
	double tlocal14=0;
	double tlocal15=0;
	double tlocal16=0;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
	h2_0=rest_y%h2d;
	rest_y=rest_y/h2d;
	h3_0=rest_x%h3d;
	rest_x=rest_x/h3d;
	h1_0=rest_y%h1d;
	rest_y=rest_y/h1d;
	p6_0=rest_x%p6d;
	rest_x=rest_x/p6d;
	p4_0=rest_y;
	p5_0=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
	h2_1=rest_y%h2d;
	rest_y=rest_y/h2d;
	h3_1=rest_x%h3d;
	rest_x=rest_x/h3d;
	h1_1=rest_y%h1d;
	rest_y=rest_y/h1d;
	p6_1=rest_x%p6d;
	rest_x=rest_x/p6d;
	p4_1=rest_y;
	p5_1=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
	h2_2=rest_y%h2d;
	rest_y=rest_y/h2d;
	h3_2=rest_x%h3d;
	rest_x=rest_x/h3d;
	h1_2=rest_y%h1d;
	rest_y=rest_y/h1d;
	p6_2=rest_x%p6d;
	rest_x=rest_x/p6d;
	p4_2=rest_y;
	p5_2=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
	h2_3=rest_y%h2d;
	rest_y=rest_y/h2d;
	h3_3=rest_x%h3d;
	rest_x=rest_x/h3d;
	h1_3=rest_y%h1d;
	rest_y=rest_y/h1d;
	p6_3=rest_x%p6d;
	rest_x=rest_x/p6d;
	p4_3=rest_y;
	p5_3=rest_x;
	size_t t2_d_off, v2_d_off;for(p7T=0;p7T<p7d;p7T+=Tcomm){size_t p7l_hi;
		p7l_hi = MIN(Tcomm+p7T,p7d)-p7T;
		t2_d_off=p4_0*p4ld_t2+h1_0*h1ld_t2+h2_0*h2ld_t2;
		v2_d_off=h3_0*h3ld_v2+p6_0*p6ld_v2+p5_0*p5ld_v2;
		if(thread_y+T1*0<total_y)for(p7l=threadIdx.x;p7l<p7l_hi;p7l+=blockDim.x){
		p7=p7l+p7T;
		t2_shm[in1_idxl+T1*0][p7l] = t2_d[t2_d_off+p7*p7ld_t2];
		}
		if(thread_x+T1*0<total_x)for(p7l=threadIdx.y;p7l<p7l_hi;p7l+=blockDim.y){
		p7=p7l+p7T;
		v2_shm[p7l][in2_idxl+T1*0] = v2_d[v2_d_off+p7*p7ld_v2];
		}
		t2_d_off=p4_1*p4ld_t2+h1_1*h1ld_t2+h2_1*h2ld_t2;
		v2_d_off=h3_1*h3ld_v2+p6_1*p6ld_v2+p5_1*p5ld_v2;
		if(thread_y+T1*1<total_y)for(p7l=threadIdx.x;p7l<p7l_hi;p7l+=blockDim.x){
		p7=p7l+p7T;
		t2_shm[in1_idxl+T1*1][p7l] = t2_d[t2_d_off+p7*p7ld_t2];
		}
		if(thread_x+T1*1<total_x)for(p7l=threadIdx.y;p7l<p7l_hi;p7l+=blockDim.y){
		p7=p7l+p7T;
		v2_shm[p7l][in2_idxl+T1*1] = v2_d[v2_d_off+p7*p7ld_v2];
		}
		t2_d_off=p4_2*p4ld_t2+h1_2*h1ld_t2+h2_2*h2ld_t2;
		v2_d_off=h3_2*h3ld_v2+p6_2*p6ld_v2+p5_2*p5ld_v2;
		if(thread_y+T1*2<total_y)for(p7l=threadIdx.x;p7l<p7l_hi;p7l+=blockDim.x){
		p7=p7l+p7T;
		t2_shm[in1_idxl+T1*2][p7l] = t2_d[t2_d_off+p7*p7ld_t2];
		}
		if(thread_x+T1*2<total_x)for(p7l=threadIdx.y;p7l<p7l_hi;p7l+=blockDim.y){
		p7=p7l+p7T;
		v2_shm[p7l][in2_idxl+T1*2] = v2_d[v2_d_off+p7*p7ld_v2];
		}
		t2_d_off=p4_3*p4ld_t2+h1_3*h1ld_t2+h2_3*h2ld_t2;
		v2_d_off=h3_3*h3ld_v2+p6_3*p6ld_v2+p5_3*p5ld_v2;
		if(thread_y+T1*3<total_y)for(p7l=threadIdx.x;p7l<p7l_hi;p7l+=blockDim.x){
		p7=p7l+p7T;
		t2_shm[in1_idxl+T1*3][p7l] = t2_d[t2_d_off+p7*p7ld_t2];
		}
		if(thread_x+T1*3<total_x)for(p7l=threadIdx.y;p7l<p7l_hi;p7l+=blockDim.y){
		p7=p7l+p7T;
		v2_shm[p7l][in2_idxl+T1*3] = v2_d[v2_d_off+p7*p7ld_v2];
		}
		__syncthreads();
		for(p7l=0;p7l<p7l_hi;++p7l){
		a1=t2_shm[in1_idxl+T1*0][p7l];
		a2=t2_shm[in1_idxl+T1*1][p7l];
		a3=t2_shm[in1_idxl+T1*2][p7l];
		a4=t2_shm[in1_idxl+T1*3][p7l];
		b1=v2_shm[p7l][in2_idxl+T2*0];
		b2=v2_shm[p7l][in2_idxl+T2*1];
		b3=v2_shm[p7l][in2_idxl+T2*2];
		b4=v2_shm[p7l][in2_idxl+T2*3];
		tlocal1+=a1*b1;
		tlocal2+=a2*b1;
		tlocal3+=a3*b1;
		tlocal4+=a4*b1;
		tlocal5+=a1*b2;
		tlocal6+=a2*b2;
		tlocal7+=a3*b2;
		tlocal8+=a4*b2;
		tlocal9+=a1*b3;
		tlocal10+=a2*b3;
		tlocal11+=a3*b3;
		tlocal12+=a4*b3;
		tlocal13+=a1*b4;
		tlocal14+=a2*b4;
		tlocal15+=a3*b4;
		tlocal16+=a4*b4;
		}
		__syncthreads();
	}
	if(thread_x+T1*0<total_x){
		if(thread_y+T2*3<total_y) {
		t3d[h2_0*h2ld_t3+h3_0*h3ld_t3+h1_0*h1ld_t3+p6_0*p6ld_t3+p4_0*p4ld_t3+p5_0*p5ld_t3]-=tlocal1;
		t3d[h2_1*h2ld_t3+h3_0*h3ld_t3+h1_1*h1ld_t3+p6_0*p6ld_t3+p4_1*p4ld_t3+p5_0*p5ld_t3]-=tlocal2;
		t3d[h2_2*h2ld_t3+h3_0*h3ld_t3+h1_2*h1ld_t3+p6_0*p6ld_t3+p4_2*p4ld_t3+p5_0*p5ld_t3]-=tlocal3;
		t3d[h2_3*h2ld_t3+h3_0*h3ld_t3+h1_3*h1ld_t3+p6_0*p6ld_t3+p4_3*p4ld_t3+p5_0*p5ld_t3]-=tlocal4;
		}
		else if(thread_y+T2*2<total_y) {
		t3d[h2_0*h2ld_t3+h3_0*h3ld_t3+h1_0*h1ld_t3+p6_0*p6ld_t3+p4_0*p4ld_t3+p5_0*p5ld_t3]-=tlocal1;
		t3d[h2_1*h2ld_t3+h3_0*h3ld_t3+h1_1*h1ld_t3+p6_0*p6ld_t3+p4_1*p4ld_t3+p5_0*p5ld_t3]-=tlocal2;
		t3d[h2_2*h2ld_t3+h3_0*h3ld_t3+h1_2*h1ld_t3+p6_0*p6ld_t3+p4_2*p4ld_t3+p5_0*p5ld_t3]-=tlocal3;
		}
		else if(thread_y+T2*1<total_y) {
		t3d[h2_0*h2ld_t3+h3_0*h3ld_t3+h1_0*h1ld_t3+p6_0*p6ld_t3+p4_0*p4ld_t3+p5_0*p5ld_t3]-=tlocal1;
		t3d[h2_1*h2ld_t3+h3_0*h3ld_t3+h1_1*h1ld_t3+p6_0*p6ld_t3+p4_1*p4ld_t3+p5_0*p5ld_t3]-=tlocal2;
		}
		else if(thread_y+T2*0<total_y) {
		t3d[h2_0*h2ld_t3+h3_0*h3ld_t3+h1_0*h1ld_t3+p6_0*p6ld_t3+p4_0*p4ld_t3+p5_0*p5ld_t3]-=tlocal1;
		}
	}
	if(thread_x+T1*1<total_x){
		if(thread_y+T2*3<total_y) {
		t3d[h2_0*h2ld_t3+h3_1*h3ld_t3+h1_0*h1ld_t3+p6_1*p6ld_t3+p4_0*p4ld_t3+p5_1*p5ld_t3]-=tlocal5;
		t3d[h2_1*h2ld_t3+h3_1*h3ld_t3+h1_1*h1ld_t3+p6_1*p6ld_t3+p4_1*p4ld_t3+p5_1*p5ld_t3]-=tlocal6;
		t3d[h2_2*h2ld_t3+h3_1*h3ld_t3+h1_2*h1ld_t3+p6_1*p6ld_t3+p4_2*p4ld_t3+p5_1*p5ld_t3]-=tlocal7;
		t3d[h2_3*h2ld_t3+h3_1*h3ld_t3+h1_3*h1ld_t3+p6_1*p6ld_t3+p4_3*p4ld_t3+p5_1*p5ld_t3]-=tlocal8;
		}
		else if(thread_y+T2*2<total_y) {
		t3d[h2_0*h2ld_t3+h3_1*h3ld_t3+h1_0*h1ld_t3+p6_1*p6ld_t3+p4_0*p4ld_t3+p5_1*p5ld_t3]-=tlocal5;
		t3d[h2_1*h2ld_t3+h3_1*h3ld_t3+h1_1*h1ld_t3+p6_1*p6ld_t3+p4_1*p4ld_t3+p5_1*p5ld_t3]-=tlocal6;
		t3d[h2_2*h2ld_t3+h3_1*h3ld_t3+h1_2*h1ld_t3+p6_1*p6ld_t3+p4_2*p4ld_t3+p5_1*p5ld_t3]-=tlocal7;
		}
		else if(thread_y+T2*1<total_y) {
		t3d[h2_0*h2ld_t3+h3_1*h3ld_t3+h1_0*h1ld_t3+p6_1*p6ld_t3+p4_0*p4ld_t3+p5_1*p5ld_t3]-=tlocal5;
		t3d[h2_1*h2ld_t3+h3_1*h3ld_t3+h1_1*h1ld_t3+p6_1*p6ld_t3+p4_1*p4ld_t3+p5_1*p5ld_t3]-=tlocal6;
		}
		else if(thread_y+T2*0<total_y) {
		t3d[h2_0*h2ld_t3+h3_1*h3ld_t3+h1_0*h1ld_t3+p6_1*p6ld_t3+p4_0*p4ld_t3+p5_1*p5ld_t3]-=tlocal5;
		}
	}
	if(thread_x+T1*2<total_x){
		if(thread_y+T2*3<total_y) {
		t3d[h2_0*h2ld_t3+h3_2*h3ld_t3+h1_0*h1ld_t3+p6_2*p6ld_t3+p4_0*p4ld_t3+p5_2*p5ld_t3]-=tlocal9;
		t3d[h2_1*h2ld_t3+h3_2*h3ld_t3+h1_1*h1ld_t3+p6_2*p6ld_t3+p4_1*p4ld_t3+p5_2*p5ld_t3]-=tlocal10;
		t3d[h2_2*h2ld_t3+h3_2*h3ld_t3+h1_2*h1ld_t3+p6_2*p6ld_t3+p4_2*p4ld_t3+p5_2*p5ld_t3]-=tlocal11;
		t3d[h2_3*h2ld_t3+h3_2*h3ld_t3+h1_3*h1ld_t3+p6_2*p6ld_t3+p4_3*p4ld_t3+p5_2*p5ld_t3]-=tlocal12;
		}
		else if(thread_y+T2*2<total_y) {
		t3d[h2_0*h2ld_t3+h3_2*h3ld_t3+h1_0*h1ld_t3+p6_2*p6ld_t3+p4_0*p4ld_t3+p5_2*p5ld_t3]-=tlocal9;
		t3d[h2_1*h2ld_t3+h3_2*h3ld_t3+h1_1*h1ld_t3+p6_2*p6ld_t3+p4_1*p4ld_t3+p5_2*p5ld_t3]-=tlocal10;
		t3d[h2_2*h2ld_t3+h3_2*h3ld_t3+h1_2*h1ld_t3+p6_2*p6ld_t3+p4_2*p4ld_t3+p5_2*p5ld_t3]-=tlocal11;
		}
		else if(thread_y+T2*1<total_y) {
		t3d[h2_0*h2ld_t3+h3_2*h3ld_t3+h1_0*h1ld_t3+p6_2*p6ld_t3+p4_0*p4ld_t3+p5_2*p5ld_t3]-=tlocal9;
		t3d[h2_1*h2ld_t3+h3_2*h3ld_t3+h1_1*h1ld_t3+p6_2*p6ld_t3+p4_1*p4ld_t3+p5_2*p5ld_t3]-=tlocal10;
		}
		else if(thread_y+T2*0<total_y) {
		t3d[h2_0*h2ld_t3+h3_2*h3ld_t3+h1_0*h1ld_t3+p6_2*p6ld_t3+p4_0*p4ld_t3+p5_2*p5ld_t3]-=tlocal9;
		}
	}
	if(thread_x+T1*3<total_x){
		if(thread_y+T2*3<total_y) {
		t3d[h2_0*h2ld_t3+h3_3*h3ld_t3+h1_0*h1ld_t3+p6_3*p6ld_t3+p4_0*p4ld_t3+p5_3*p5ld_t3]-=tlocal13;
		t3d[h2_1*h2ld_t3+h3_3*h3ld_t3+h1_1*h1ld_t3+p6_3*p6ld_t3+p4_1*p4ld_t3+p5_3*p5ld_t3]-=tlocal14;
		t3d[h2_2*h2ld_t3+h3_3*h3ld_t3+h1_2*h1ld_t3+p6_3*p6ld_t3+p4_2*p4ld_t3+p5_3*p5ld_t3]-=tlocal15;
		t3d[h2_3*h2ld_t3+h3_3*h3ld_t3+h1_3*h1ld_t3+p6_3*p6ld_t3+p4_3*p4ld_t3+p5_3*p5ld_t3]-=tlocal16;
		}
		else if(thread_y+T2*2<total_y) {
		t3d[h2_0*h2ld_t3+h3_3*h3ld_t3+h1_0*h1ld_t3+p6_3*p6ld_t3+p4_0*p4ld_t3+p5_3*p5ld_t3]-=tlocal13;
		t3d[h2_1*h2ld_t3+h3_3*h3ld_t3+h1_1*h1ld_t3+p6_3*p6ld_t3+p4_1*p4ld_t3+p5_3*p5ld_t3]-=tlocal14;
		t3d[h2_2*h2ld_t3+h3_3*h3ld_t3+h1_2*h1ld_t3+p6_3*p6ld_t3+p4_2*p4ld_t3+p5_3*p5ld_t3]-=tlocal15;
		}
		else if(thread_y+T2*1<total_y) {
		t3d[h2_0*h2ld_t3+h3_3*h3ld_t3+h1_0*h1ld_t3+p6_3*p6ld_t3+p4_0*p4ld_t3+p5_3*p5ld_t3]-=tlocal13;
		t3d[h2_1*h2ld_t3+h3_3*h3ld_t3+h1_1*h1ld_t3+p6_3*p6ld_t3+p4_1*p4ld_t3+p5_3*p5ld_t3]-=tlocal14;
		}
		else if(thread_y+T2*0<total_y) {
		t3d[h2_0*h2ld_t3+h3_3*h3ld_t3+h1_0*h1ld_t3+p6_3*p6ld_t3+p4_0*p4ld_t3+p5_3*p5ld_t3]-=tlocal13;
		}
	}
	__syncthreads();
}
void ma_sd_t_d2_6_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d, size_t p7d, double *t3, double *t2, double *v2) 
{
	size_t p7ld_t2,p4ld_t2,h1ld_t2,h2ld_t2,p7ld_v2,h3ld_v2,p6ld_v2,p5ld_v2,h2ld_t3,h3ld_t3,h1ld_t3,p6ld_t3,p4ld_t3,p5ld_t3;
	size_t size_t3,size_block_t3,size_el_block_t3,size_t2,size_v2;
	cudaStream_t *streams;
	size_t nstreams,i;
	double *t2_d,*v2_d;
	size_t3=h2d*h3d*h1d*p6d*p4d*p5d*sizeof(double);
	size_t2=p7d*p4d*h1d*h2d*sizeof(double);
	size_v2=p7d*h3d*p6d*p5d*sizeof(double);
	cudaFuncSetCacheConfig(ma_sd_t_d2_6_kernel, cudaFuncCachePreferShared);
	nstreams=1;
	size_block_t3=size_t3/nstreams;
	size_el_block_t3=size_block_t3/sizeof(double);
	//t3d=(double*)getGpuMem(size_t3);
	t2_d=(double*)getGpuMem(size_t2);
	v2_d=(double*)getGpuMem(size_v2);
	streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
	assert(streams!= NULL);
	for(i=0;i<nstreams;++i) {
		CUDA_SAFE(cudaStreamCreate(&streams[i])) ;
	}
	CUDA_SAFE(cudaMemcpy(t2_d,t2,size_t2,cudaMemcpyHostToDevice));
	CUDA_SAFE(cudaMemcpy(v2_d,v2,size_v2,cudaMemcpyHostToDevice));
	p7ld_t2=1;
	p4ld_t2=p7d;
	h1ld_t2=p4d*p7d;
	h2ld_t2=h1d*p4d*p7d;
	p7ld_v2=1;
	h3ld_v2=p7d;
	p6ld_v2=h3d*p7d;
	p5ld_v2=p6d*h3d*p7d;
	h2ld_t3=1;
	h3ld_t3=h2d;
	h1ld_t3=h3d*h2d;
	p6ld_t3=h1d*h3d*h2d;
	p4ld_t3=p6d*h1d*h3d*h2d;
	p5ld_t3=p4d*p6d*h1d*h3d*h2d;
	size_t total_x = h3d*p6d*p5d;
	size_t total_y = p4d*h1d*h2d;
	dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
	for(i=0;i<nstreams;++i){
		ma_sd_t_d2_6_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(h1d,h2d,h3d,p4d,p5d,p6d,p7d,p7ld_t2,p4ld_t2,h1ld_t2,h2ld_t2,p7ld_v2,h3ld_v2,p6ld_v2,p5ld_v2,h2ld_t3,h3ld_t3,h1ld_t3,p6ld_t3,p4ld_t3,p5ld_t3,ma_t3_d,t2_d,v2_d,i,total_x,total_y);
		CHECK_ERR();
	}
	cudaThreadSynchronize();
	for(i=0;i<nstreams;++i){
		cudaStreamDestroy(streams[i]);}
	//freeGpuMem(t3d);
	freeGpuMem(t2_d);
	freeGpuMem(v2_d);
	free(streams);
}
#undef T1
#undef T2
#undef Tcomm
void ma_sd_t_d2_6_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* p4d, Integer* p5d, Integer* p6d, Integer* p7d, double *t3, double *t2, double *v2) 
{
	ma_sd_t_d2_6_cuda((int)*h1d,(int)*h2d,(int)*h3d,(int)*p4d,(int)*p5d,(int)*p6d,(int)*p7d,t3,t2,v2);
}
/*----------------------------------------------------------------------*
*t3[h3,h2,h1,p4,p6,p5] -= t2[p7,p4,h1,h2] * v2[p7,h3,p6,p5]
*----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void ma_sd_t_d2_7_kernel(size_t h1d,size_t h2d,size_t h3d,size_t p4d,size_t p5d,size_t p6d,size_t p7d,
	size_t p7ld_t2,size_t p4ld_t2,size_t h1ld_t2,size_t h2ld_t2,size_t p7ld_v2,size_t h3ld_v2,size_t p6ld_v2,size_t p5ld_v2,
	size_t h3ld_t3,size_t h2ld_t3,size_t h1ld_t3,size_t p4ld_t3,size_t p6ld_t3,size_t p5ld_t3,
	double *t3d, double *t2_d, double *v2_d,size_t unused_idx, size_t total_x, size_t total_y) 
{
	size_t h1_0,h1_1,h1_2,h1_3,h2_0,h2_1,h2_2,h2_3,h3_0,h3_1,h3_2,h3_3,p4_0,p4_1,p4_2,p4_3,p5_0,p5_1,p5_2,p5_3,p6_0,p6_1,p6_2,p6_3,p7;
	double a1,b1;
	double a2,b2;
	double a3,b3;
	double a4,b4;
	size_t in1_idxl,in2_idxl,p7l,p7T;
	__shared__ double t2_shm[4*T1][Tcomm];
	__shared__ double v2_shm[Tcomm][4*T2];
	size_t rest_x=blockIdx.x;
	size_t rest_y=blockIdx.y;
	size_t thread_x = T2*4 * rest_x + threadIdx.x;
	size_t thread_y = T1*4 * rest_y + threadIdx.y;
	in1_idxl=threadIdx.y;
	in2_idxl=threadIdx.x ;
	double tlocal1=0;
	double tlocal2=0;
	double tlocal3=0;
	double tlocal4=0;
	double tlocal5=0;
	double tlocal6=0;
	double tlocal7=0;
	double tlocal8=0;
	double tlocal9=0;
	double tlocal10=0;
	double tlocal11=0;
	double tlocal12=0;
	double tlocal13=0;
	double tlocal14=0;
	double tlocal15=0;
	double tlocal16=0;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
	h3_0=rest_x%h3d;
	rest_x=rest_x/h3d;
	h2_0=rest_y%h2d;
	rest_y=rest_y/h2d;
	h1_0=rest_y%h1d;
	rest_y=rest_y/h1d;
	p6_0=rest_x%p6d;
	rest_x=rest_x/p6d;
	p4_0=rest_y;
	p5_0=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
	h3_1=rest_x%h3d;
	rest_x=rest_x/h3d;
	h2_1=rest_y%h2d;
	rest_y=rest_y/h2d;
	h1_1=rest_y%h1d;
	rest_y=rest_y/h1d;
	p6_1=rest_x%p6d;
	rest_x=rest_x/p6d;
	p4_1=rest_y;
	p5_1=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
	h3_2=rest_x%h3d;
	rest_x=rest_x/h3d;
	h2_2=rest_y%h2d;
	rest_y=rest_y/h2d;
	h1_2=rest_y%h1d;
	rest_y=rest_y/h1d;
	p6_2=rest_x%p6d;
	rest_x=rest_x/p6d;
	p4_2=rest_y;
	p5_2=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
	h3_3=rest_x%h3d;
	rest_x=rest_x/h3d;
	h2_3=rest_y%h2d;
	rest_y=rest_y/h2d;
	h1_3=rest_y%h1d;
	rest_y=rest_y/h1d;
	p6_3=rest_x%p6d;
	rest_x=rest_x/p6d;
	p4_3=rest_y;
	p5_3=rest_x;
	size_t t2_d_off, v2_d_off;for(p7T=0;p7T<p7d;p7T+=Tcomm){size_t p7l_hi;
		p7l_hi = MIN(Tcomm+p7T,p7d)-p7T;
		t2_d_off=p4_0*p4ld_t2+h1_0*h1ld_t2+h2_0*h2ld_t2;
		v2_d_off=h3_0*h3ld_v2+p6_0*p6ld_v2+p5_0*p5ld_v2;
		if(thread_y+T1*0<total_y)for(p7l=threadIdx.x;p7l<p7l_hi;p7l+=blockDim.x){
		p7=p7l+p7T;
		t2_shm[in1_idxl+T1*0][p7l] = t2_d[t2_d_off+p7*p7ld_t2];
		}
		if(thread_x+T1*0<total_x)for(p7l=threadIdx.y;p7l<p7l_hi;p7l+=blockDim.y){
		p7=p7l+p7T;
		v2_shm[p7l][in2_idxl+T1*0] = v2_d[v2_d_off+p7*p7ld_v2];
		}
		t2_d_off=p4_1*p4ld_t2+h1_1*h1ld_t2+h2_1*h2ld_t2;
		v2_d_off=h3_1*h3ld_v2+p6_1*p6ld_v2+p5_1*p5ld_v2;
		if(thread_y+T1*1<total_y)for(p7l=threadIdx.x;p7l<p7l_hi;p7l+=blockDim.x){
		p7=p7l+p7T;
		t2_shm[in1_idxl+T1*1][p7l] = t2_d[t2_d_off+p7*p7ld_t2];
		}
		if(thread_x+T1*1<total_x)for(p7l=threadIdx.y;p7l<p7l_hi;p7l+=blockDim.y){
		p7=p7l+p7T;
		v2_shm[p7l][in2_idxl+T1*1] = v2_d[v2_d_off+p7*p7ld_v2];
		}
		t2_d_off=p4_2*p4ld_t2+h1_2*h1ld_t2+h2_2*h2ld_t2;
		v2_d_off=h3_2*h3ld_v2+p6_2*p6ld_v2+p5_2*p5ld_v2;
		if(thread_y+T1*2<total_y)for(p7l=threadIdx.x;p7l<p7l_hi;p7l+=blockDim.x){
		p7=p7l+p7T;
		t2_shm[in1_idxl+T1*2][p7l] = t2_d[t2_d_off+p7*p7ld_t2];
		}
		if(thread_x+T1*2<total_x)for(p7l=threadIdx.y;p7l<p7l_hi;p7l+=blockDim.y){
		p7=p7l+p7T;
		v2_shm[p7l][in2_idxl+T1*2] = v2_d[v2_d_off+p7*p7ld_v2];
		}
		t2_d_off=p4_3*p4ld_t2+h1_3*h1ld_t2+h2_3*h2ld_t2;
		v2_d_off=h3_3*h3ld_v2+p6_3*p6ld_v2+p5_3*p5ld_v2;
		if(thread_y+T1*3<total_y)for(p7l=threadIdx.x;p7l<p7l_hi;p7l+=blockDim.x){
		p7=p7l+p7T;
		t2_shm[in1_idxl+T1*3][p7l] = t2_d[t2_d_off+p7*p7ld_t2];
		}
		if(thread_x+T1*3<total_x)for(p7l=threadIdx.y;p7l<p7l_hi;p7l+=blockDim.y){
		p7=p7l+p7T;
		v2_shm[p7l][in2_idxl+T1*3] = v2_d[v2_d_off+p7*p7ld_v2];
		}
		__syncthreads();
		for(p7l=0;p7l<p7l_hi;++p7l){
		a1=t2_shm[in1_idxl+T1*0][p7l];
		a2=t2_shm[in1_idxl+T1*1][p7l];
		a3=t2_shm[in1_idxl+T1*2][p7l];
		a4=t2_shm[in1_idxl+T1*3][p7l];
		b1=v2_shm[p7l][in2_idxl+T2*0];
		b2=v2_shm[p7l][in2_idxl+T2*1];
		b3=v2_shm[p7l][in2_idxl+T2*2];
		b4=v2_shm[p7l][in2_idxl+T2*3];
		tlocal1+=a1*b1;
		tlocal2+=a2*b1;
		tlocal3+=a3*b1;
		tlocal4+=a4*b1;
		tlocal5+=a1*b2;
		tlocal6+=a2*b2;
		tlocal7+=a3*b2;
		tlocal8+=a4*b2;
		tlocal9+=a1*b3;
		tlocal10+=a2*b3;
		tlocal11+=a3*b3;
		tlocal12+=a4*b3;
		tlocal13+=a1*b4;
		tlocal14+=a2*b4;
		tlocal15+=a3*b4;
		tlocal16+=a4*b4;
		}
		__syncthreads();
	}
	if(thread_x+T1*0<total_x){
		if(thread_y+T2*3<total_y) {
		t3d[h3_0*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p4_0*p4ld_t3+p6_0*p6ld_t3+p5_0*p5ld_t3]-=tlocal1;
		t3d[h3_0*h3ld_t3+h2_1*h2ld_t3+h1_1*h1ld_t3+p4_1*p4ld_t3+p6_0*p6ld_t3+p5_0*p5ld_t3]-=tlocal2;
		t3d[h3_0*h3ld_t3+h2_2*h2ld_t3+h1_2*h1ld_t3+p4_2*p4ld_t3+p6_0*p6ld_t3+p5_0*p5ld_t3]-=tlocal3;
		t3d[h3_0*h3ld_t3+h2_3*h2ld_t3+h1_3*h1ld_t3+p4_3*p4ld_t3+p6_0*p6ld_t3+p5_0*p5ld_t3]-=tlocal4;
		}
		else if(thread_y+T2*2<total_y) {
		t3d[h3_0*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p4_0*p4ld_t3+p6_0*p6ld_t3+p5_0*p5ld_t3]-=tlocal1;
		t3d[h3_0*h3ld_t3+h2_1*h2ld_t3+h1_1*h1ld_t3+p4_1*p4ld_t3+p6_0*p6ld_t3+p5_0*p5ld_t3]-=tlocal2;
		t3d[h3_0*h3ld_t3+h2_2*h2ld_t3+h1_2*h1ld_t3+p4_2*p4ld_t3+p6_0*p6ld_t3+p5_0*p5ld_t3]-=tlocal3;
		}
		else if(thread_y+T2*1<total_y) {
		t3d[h3_0*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p4_0*p4ld_t3+p6_0*p6ld_t3+p5_0*p5ld_t3]-=tlocal1;
		t3d[h3_0*h3ld_t3+h2_1*h2ld_t3+h1_1*h1ld_t3+p4_1*p4ld_t3+p6_0*p6ld_t3+p5_0*p5ld_t3]-=tlocal2;
		}
		else if(thread_y+T2*0<total_y) {
		t3d[h3_0*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p4_0*p4ld_t3+p6_0*p6ld_t3+p5_0*p5ld_t3]-=tlocal1;
		}
	}
	if(thread_x+T1*1<total_x){
		if(thread_y+T2*3<total_y) {
		t3d[h3_1*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p4_0*p4ld_t3+p6_1*p6ld_t3+p5_1*p5ld_t3]-=tlocal5;
		t3d[h3_1*h3ld_t3+h2_1*h2ld_t3+h1_1*h1ld_t3+p4_1*p4ld_t3+p6_1*p6ld_t3+p5_1*p5ld_t3]-=tlocal6;
		t3d[h3_1*h3ld_t3+h2_2*h2ld_t3+h1_2*h1ld_t3+p4_2*p4ld_t3+p6_1*p6ld_t3+p5_1*p5ld_t3]-=tlocal7;
		t3d[h3_1*h3ld_t3+h2_3*h2ld_t3+h1_3*h1ld_t3+p4_3*p4ld_t3+p6_1*p6ld_t3+p5_1*p5ld_t3]-=tlocal8;
		}
		else if(thread_y+T2*2<total_y) {
		t3d[h3_1*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p4_0*p4ld_t3+p6_1*p6ld_t3+p5_1*p5ld_t3]-=tlocal5;
		t3d[h3_1*h3ld_t3+h2_1*h2ld_t3+h1_1*h1ld_t3+p4_1*p4ld_t3+p6_1*p6ld_t3+p5_1*p5ld_t3]-=tlocal6;
		t3d[h3_1*h3ld_t3+h2_2*h2ld_t3+h1_2*h1ld_t3+p4_2*p4ld_t3+p6_1*p6ld_t3+p5_1*p5ld_t3]-=tlocal7;
		}
		else if(thread_y+T2*1<total_y) {
		t3d[h3_1*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p4_0*p4ld_t3+p6_1*p6ld_t3+p5_1*p5ld_t3]-=tlocal5;
		t3d[h3_1*h3ld_t3+h2_1*h2ld_t3+h1_1*h1ld_t3+p4_1*p4ld_t3+p6_1*p6ld_t3+p5_1*p5ld_t3]-=tlocal6;
		}
		else if(thread_y+T2*0<total_y) {
		t3d[h3_1*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p4_0*p4ld_t3+p6_1*p6ld_t3+p5_1*p5ld_t3]-=tlocal5;
		}
	}
	if(thread_x+T1*2<total_x){
		if(thread_y+T2*3<total_y) {
		t3d[h3_2*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p4_0*p4ld_t3+p6_2*p6ld_t3+p5_2*p5ld_t3]-=tlocal9;
		t3d[h3_2*h3ld_t3+h2_1*h2ld_t3+h1_1*h1ld_t3+p4_1*p4ld_t3+p6_2*p6ld_t3+p5_2*p5ld_t3]-=tlocal10;
		t3d[h3_2*h3ld_t3+h2_2*h2ld_t3+h1_2*h1ld_t3+p4_2*p4ld_t3+p6_2*p6ld_t3+p5_2*p5ld_t3]-=tlocal11;
		t3d[h3_2*h3ld_t3+h2_3*h2ld_t3+h1_3*h1ld_t3+p4_3*p4ld_t3+p6_2*p6ld_t3+p5_2*p5ld_t3]-=tlocal12;
		}
		else if(thread_y+T2*2<total_y) {
		t3d[h3_2*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p4_0*p4ld_t3+p6_2*p6ld_t3+p5_2*p5ld_t3]-=tlocal9;
		t3d[h3_2*h3ld_t3+h2_1*h2ld_t3+h1_1*h1ld_t3+p4_1*p4ld_t3+p6_2*p6ld_t3+p5_2*p5ld_t3]-=tlocal10;
		t3d[h3_2*h3ld_t3+h2_2*h2ld_t3+h1_2*h1ld_t3+p4_2*p4ld_t3+p6_2*p6ld_t3+p5_2*p5ld_t3]-=tlocal11;
		}
		else if(thread_y+T2*1<total_y) {
		t3d[h3_2*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p4_0*p4ld_t3+p6_2*p6ld_t3+p5_2*p5ld_t3]-=tlocal9;
		t3d[h3_2*h3ld_t3+h2_1*h2ld_t3+h1_1*h1ld_t3+p4_1*p4ld_t3+p6_2*p6ld_t3+p5_2*p5ld_t3]-=tlocal10;
		}
		else if(thread_y+T2*0<total_y) {
		t3d[h3_2*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p4_0*p4ld_t3+p6_2*p6ld_t3+p5_2*p5ld_t3]-=tlocal9;
		}
	}
	if(thread_x+T1*3<total_x){
		if(thread_y+T2*3<total_y) {
		t3d[h3_3*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p4_0*p4ld_t3+p6_3*p6ld_t3+p5_3*p5ld_t3]-=tlocal13;
		t3d[h3_3*h3ld_t3+h2_1*h2ld_t3+h1_1*h1ld_t3+p4_1*p4ld_t3+p6_3*p6ld_t3+p5_3*p5ld_t3]-=tlocal14;
		t3d[h3_3*h3ld_t3+h2_2*h2ld_t3+h1_2*h1ld_t3+p4_2*p4ld_t3+p6_3*p6ld_t3+p5_3*p5ld_t3]-=tlocal15;
		t3d[h3_3*h3ld_t3+h2_3*h2ld_t3+h1_3*h1ld_t3+p4_3*p4ld_t3+p6_3*p6ld_t3+p5_3*p5ld_t3]-=tlocal16;
		}
		else if(thread_y+T2*2<total_y) {
		t3d[h3_3*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p4_0*p4ld_t3+p6_3*p6ld_t3+p5_3*p5ld_t3]-=tlocal13;
		t3d[h3_3*h3ld_t3+h2_1*h2ld_t3+h1_1*h1ld_t3+p4_1*p4ld_t3+p6_3*p6ld_t3+p5_3*p5ld_t3]-=tlocal14;
		t3d[h3_3*h3ld_t3+h2_2*h2ld_t3+h1_2*h1ld_t3+p4_2*p4ld_t3+p6_3*p6ld_t3+p5_3*p5ld_t3]-=tlocal15;
		}
		else if(thread_y+T2*1<total_y) {
		t3d[h3_3*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p4_0*p4ld_t3+p6_3*p6ld_t3+p5_3*p5ld_t3]-=tlocal13;
		t3d[h3_3*h3ld_t3+h2_1*h2ld_t3+h1_1*h1ld_t3+p4_1*p4ld_t3+p6_3*p6ld_t3+p5_3*p5ld_t3]-=tlocal14;
		}
		else if(thread_y+T2*0<total_y) {
		t3d[h3_3*h3ld_t3+h2_0*h2ld_t3+h1_0*h1ld_t3+p4_0*p4ld_t3+p6_3*p6ld_t3+p5_3*p5ld_t3]-=tlocal13;
		}
	}
	__syncthreads();
}
void ma_sd_t_d2_7_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d, size_t p7d, double *t3, double *t2, double *v2) 
{
	size_t p7ld_t2,p4ld_t2,h1ld_t2,h2ld_t2,p7ld_v2,h3ld_v2,p6ld_v2,p5ld_v2,h3ld_t3,h2ld_t3,h1ld_t3,p4ld_t3,p6ld_t3,p5ld_t3;
	size_t size_t3,size_block_t3,size_el_block_t3,size_t2,size_v2;
	cudaStream_t *streams;
	size_t nstreams,i;
	double *t2_d,*v2_d;
	size_t3=h3d*h2d*h1d*p4d*p6d*p5d*sizeof(double);
	size_t2=p7d*p4d*h1d*h2d*sizeof(double);
	size_v2=p7d*h3d*p6d*p5d*sizeof(double);
	cudaFuncSetCacheConfig(ma_sd_t_d2_7_kernel, cudaFuncCachePreferShared);
	nstreams=1;
	size_block_t3=size_t3/nstreams;
	size_el_block_t3=size_block_t3/sizeof(double);
	//t3d=(double*)getGpuMem(size_t3);
	t2_d=(double*)getGpuMem(size_t2);
	v2_d=(double*)getGpuMem(size_v2);
	streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
	assert(streams!= NULL);
	for(i=0;i<nstreams;++i) {
		CUDA_SAFE(cudaStreamCreate(&streams[i])) ;
	}
	CUDA_SAFE(cudaMemcpy(t2_d,t2,size_t2,cudaMemcpyHostToDevice));
	CUDA_SAFE(cudaMemcpy(v2_d,v2,size_v2,cudaMemcpyHostToDevice));
	p7ld_t2=1;
	p4ld_t2=p7d;
	h1ld_t2=p4d*p7d;
	h2ld_t2=h1d*p4d*p7d;
	p7ld_v2=1;
	h3ld_v2=p7d;
	p6ld_v2=h3d*p7d;
	p5ld_v2=p6d*h3d*p7d;
	h3ld_t3=1;
	h2ld_t3=h3d;
	h1ld_t3=h2d*h3d;
	p4ld_t3=h1d*h2d*h3d;
	p6ld_t3=p4d*h1d*h2d*h3d;
	p5ld_t3=p6d*p4d*h1d*h2d*h3d;
	size_t total_x = h3d*p6d*p5d;
	size_t total_y = p4d*h1d*h2d;
	dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
	for(i=0;i<nstreams;++i){
		ma_sd_t_d2_7_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(h1d,h2d,h3d,p4d,p5d,p6d,p7d,p7ld_t2,p4ld_t2,h1ld_t2,h2ld_t2,p7ld_v2,h3ld_v2,p6ld_v2,p5ld_v2,h3ld_t3,h2ld_t3,h1ld_t3,p4ld_t3,p6ld_t3,p5ld_t3,ma_t3_d,t2_d,v2_d,i,total_x,total_y);
		CHECK_ERR("Kernel execution failed");
	}
	cudaThreadSynchronize();
	for(i=0;i<nstreams;++i){
		cudaStreamDestroy(streams[i]);}
	//freeGpuMem(t3d);
	freeGpuMem(t2_d);
	freeGpuMem(v2_d);
	free(streams);
}
#undef T1
#undef T2
#undef Tcomm
void ma_sd_t_d2_7_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* p4d, Integer* p5d, Integer* p6d, Integer* p7d, double *t3, double *t2, double *v2) 
{
	ma_sd_t_d2_7_cuda((int)*h1d,(int)*h2d,(int)*h3d,(int)*p4d,(int)*p5d,(int)*p6d,(int)*p7d,t3,t2,v2);
}
/*----------------------------------------------------------------------*
*t3[h2,h1,h3,p4,p6,p5] -= t2[p7,p4,h1,h2] * v2[p7,h3,p6,p5]
*----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void ma_sd_t_d2_8_kernel(size_t h1d,size_t h2d,size_t h3d,size_t p4d,size_t p5d,size_t p6d,size_t p7d,
	size_t p7ld_t2,size_t p4ld_t2,size_t h1ld_t2,size_t h2ld_t2,size_t p7ld_v2,size_t h3ld_v2,size_t p6ld_v2,size_t p5ld_v2,
	size_t h2ld_t3,size_t h1ld_t3,size_t h3ld_t3,size_t p4ld_t3,size_t p6ld_t3,size_t p5ld_t3,
	double *t3d, double *t2_d, double *v2_d,size_t unused_idx, size_t total_x, size_t total_y) 
{
	size_t h1_0,h1_1,h1_2,h1_3,h2_0,h2_1,h2_2,h2_3,h3_0,h3_1,h3_2,h3_3,p4_0,p4_1,p4_2,p4_3,p5_0,p5_1,p5_2,p5_3,p6_0,p6_1,p6_2,p6_3,p7;
	double a1,b1;
	double a2,b2;
	double a3,b3;
	double a4,b4;
	size_t in1_idxl,in2_idxl,p7l,p7T;
	__shared__ double t2_shm[4*T1][Tcomm];
	__shared__ double v2_shm[Tcomm][4*T2];
	size_t rest_x=blockIdx.x;
	size_t rest_y=blockIdx.y;
	size_t thread_x = T2*4 * rest_x + threadIdx.x;
	size_t thread_y = T1*4 * rest_y + threadIdx.y;
	in1_idxl=threadIdx.y;
	in2_idxl=threadIdx.x ;
	double tlocal1=0;
	double tlocal2=0;
	double tlocal3=0;
	double tlocal4=0;
	double tlocal5=0;
	double tlocal6=0;
	double tlocal7=0;
	double tlocal8=0;
	double tlocal9=0;
	double tlocal10=0;
	double tlocal11=0;
	double tlocal12=0;
	double tlocal13=0;
	double tlocal14=0;
	double tlocal15=0;
	double tlocal16=0;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
	h2_0=rest_y%h2d;
	rest_y=rest_y/h2d;
	h1_0=rest_y%h1d;
	rest_y=rest_y/h1d;
	h3_0=rest_x%h3d;
	rest_x=rest_x/h3d;
	p6_0=rest_x%p6d;
	rest_x=rest_x/p6d;
	p4_0=rest_y;
	p5_0=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
	h2_1=rest_y%h2d;
	rest_y=rest_y/h2d;
	h1_1=rest_y%h1d;
	rest_y=rest_y/h1d;
	h3_1=rest_x%h3d;
	rest_x=rest_x/h3d;
	p6_1=rest_x%p6d;
	rest_x=rest_x/p6d;
	p4_1=rest_y;
	p5_1=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
	h2_2=rest_y%h2d;
	rest_y=rest_y/h2d;
	h1_2=rest_y%h1d;
	rest_y=rest_y/h1d;
	h3_2=rest_x%h3d;
	rest_x=rest_x/h3d;
	p6_2=rest_x%p6d;
	rest_x=rest_x/p6d;
	p4_2=rest_y;
	p5_2=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
	h2_3=rest_y%h2d;
	rest_y=rest_y/h2d;
	h1_3=rest_y%h1d;
	rest_y=rest_y/h1d;
	h3_3=rest_x%h3d;
	rest_x=rest_x/h3d;
	p6_3=rest_x%p6d;
	rest_x=rest_x/p6d;
	p4_3=rest_y;
	p5_3=rest_x;
	size_t t2_d_off, v2_d_off;for(p7T=0;p7T<p7d;p7T+=Tcomm){size_t p7l_hi;
		p7l_hi = MIN(Tcomm+p7T,p7d)-p7T;
		t2_d_off=p4_0*p4ld_t2+h1_0*h1ld_t2+h2_0*h2ld_t2;
		v2_d_off=h3_0*h3ld_v2+p6_0*p6ld_v2+p5_0*p5ld_v2;
		if(thread_y+T1*0<total_y)for(p7l=threadIdx.x;p7l<p7l_hi;p7l+=blockDim.x){
		p7=p7l+p7T;
		t2_shm[in1_idxl+T1*0][p7l] = t2_d[t2_d_off+p7*p7ld_t2];
		}
		if(thread_x+T1*0<total_x)for(p7l=threadIdx.y;p7l<p7l_hi;p7l+=blockDim.y){
		p7=p7l+p7T;
		v2_shm[p7l][in2_idxl+T1*0] = v2_d[v2_d_off+p7*p7ld_v2];
		}
		t2_d_off=p4_1*p4ld_t2+h1_1*h1ld_t2+h2_1*h2ld_t2;
		v2_d_off=h3_1*h3ld_v2+p6_1*p6ld_v2+p5_1*p5ld_v2;
		if(thread_y+T1*1<total_y)for(p7l=threadIdx.x;p7l<p7l_hi;p7l+=blockDim.x){
		p7=p7l+p7T;
		t2_shm[in1_idxl+T1*1][p7l] = t2_d[t2_d_off+p7*p7ld_t2];
		}
		if(thread_x+T1*1<total_x)for(p7l=threadIdx.y;p7l<p7l_hi;p7l+=blockDim.y){
		p7=p7l+p7T;
		v2_shm[p7l][in2_idxl+T1*1] = v2_d[v2_d_off+p7*p7ld_v2];
		}
		t2_d_off=p4_2*p4ld_t2+h1_2*h1ld_t2+h2_2*h2ld_t2;
		v2_d_off=h3_2*h3ld_v2+p6_2*p6ld_v2+p5_2*p5ld_v2;
		if(thread_y+T1*2<total_y)for(p7l=threadIdx.x;p7l<p7l_hi;p7l+=blockDim.x){
		p7=p7l+p7T;
		t2_shm[in1_idxl+T1*2][p7l] = t2_d[t2_d_off+p7*p7ld_t2];
		}
		if(thread_x+T1*2<total_x)for(p7l=threadIdx.y;p7l<p7l_hi;p7l+=blockDim.y){
		p7=p7l+p7T;
		v2_shm[p7l][in2_idxl+T1*2] = v2_d[v2_d_off+p7*p7ld_v2];
		}
		t2_d_off=p4_3*p4ld_t2+h1_3*h1ld_t2+h2_3*h2ld_t2;
		v2_d_off=h3_3*h3ld_v2+p6_3*p6ld_v2+p5_3*p5ld_v2;
		if(thread_y+T1*3<total_y)for(p7l=threadIdx.x;p7l<p7l_hi;p7l+=blockDim.x){
		p7=p7l+p7T;
		t2_shm[in1_idxl+T1*3][p7l] = t2_d[t2_d_off+p7*p7ld_t2];
		}
		if(thread_x+T1*3<total_x)for(p7l=threadIdx.y;p7l<p7l_hi;p7l+=blockDim.y){
		p7=p7l+p7T;
		v2_shm[p7l][in2_idxl+T1*3] = v2_d[v2_d_off+p7*p7ld_v2];
		}
		__syncthreads();
		for(p7l=0;p7l<p7l_hi;++p7l){
		a1=t2_shm[in1_idxl+T1*0][p7l];
		a2=t2_shm[in1_idxl+T1*1][p7l];
		a3=t2_shm[in1_idxl+T1*2][p7l];
		a4=t2_shm[in1_idxl+T1*3][p7l];
		b1=v2_shm[p7l][in2_idxl+T2*0];
		b2=v2_shm[p7l][in2_idxl+T2*1];
		b3=v2_shm[p7l][in2_idxl+T2*2];
		b4=v2_shm[p7l][in2_idxl+T2*3];
		tlocal1+=a1*b1;
		tlocal2+=a2*b1;
		tlocal3+=a3*b1;
		tlocal4+=a4*b1;
		tlocal5+=a1*b2;
		tlocal6+=a2*b2;
		tlocal7+=a3*b2;
		tlocal8+=a4*b2;
		tlocal9+=a1*b3;
		tlocal10+=a2*b3;
		tlocal11+=a3*b3;
		tlocal12+=a4*b3;
		tlocal13+=a1*b4;
		tlocal14+=a2*b4;
		tlocal15+=a3*b4;
		tlocal16+=a4*b4;
		}
		__syncthreads();
	}
	if(thread_x+T1*0<total_x){
		if(thread_y+T2*3<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_0*h3ld_t3+p4_0*p4ld_t3+p6_0*p6ld_t3+p5_0*p5ld_t3]-=tlocal1;
		t3d[h2_1*h2ld_t3+h1_1*h1ld_t3+h3_0*h3ld_t3+p4_1*p4ld_t3+p6_0*p6ld_t3+p5_0*p5ld_t3]-=tlocal2;
		t3d[h2_2*h2ld_t3+h1_2*h1ld_t3+h3_0*h3ld_t3+p4_2*p4ld_t3+p6_0*p6ld_t3+p5_0*p5ld_t3]-=tlocal3;
		t3d[h2_3*h2ld_t3+h1_3*h1ld_t3+h3_0*h3ld_t3+p4_3*p4ld_t3+p6_0*p6ld_t3+p5_0*p5ld_t3]-=tlocal4;
		}
		else if(thread_y+T2*2<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_0*h3ld_t3+p4_0*p4ld_t3+p6_0*p6ld_t3+p5_0*p5ld_t3]-=tlocal1;
		t3d[h2_1*h2ld_t3+h1_1*h1ld_t3+h3_0*h3ld_t3+p4_1*p4ld_t3+p6_0*p6ld_t3+p5_0*p5ld_t3]-=tlocal2;
		t3d[h2_2*h2ld_t3+h1_2*h1ld_t3+h3_0*h3ld_t3+p4_2*p4ld_t3+p6_0*p6ld_t3+p5_0*p5ld_t3]-=tlocal3;
		}
		else if(thread_y+T2*1<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_0*h3ld_t3+p4_0*p4ld_t3+p6_0*p6ld_t3+p5_0*p5ld_t3]-=tlocal1;
		t3d[h2_1*h2ld_t3+h1_1*h1ld_t3+h3_0*h3ld_t3+p4_1*p4ld_t3+p6_0*p6ld_t3+p5_0*p5ld_t3]-=tlocal2;
		}
		else if(thread_y+T2*0<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_0*h3ld_t3+p4_0*p4ld_t3+p6_0*p6ld_t3+p5_0*p5ld_t3]-=tlocal1;
		}
	}
	if(thread_x+T1*1<total_x){
		if(thread_y+T2*3<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_1*h3ld_t3+p4_0*p4ld_t3+p6_1*p6ld_t3+p5_1*p5ld_t3]-=tlocal5;
		t3d[h2_1*h2ld_t3+h1_1*h1ld_t3+h3_1*h3ld_t3+p4_1*p4ld_t3+p6_1*p6ld_t3+p5_1*p5ld_t3]-=tlocal6;
		t3d[h2_2*h2ld_t3+h1_2*h1ld_t3+h3_1*h3ld_t3+p4_2*p4ld_t3+p6_1*p6ld_t3+p5_1*p5ld_t3]-=tlocal7;
		t3d[h2_3*h2ld_t3+h1_3*h1ld_t3+h3_1*h3ld_t3+p4_3*p4ld_t3+p6_1*p6ld_t3+p5_1*p5ld_t3]-=tlocal8;
		}
		else if(thread_y+T2*2<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_1*h3ld_t3+p4_0*p4ld_t3+p6_1*p6ld_t3+p5_1*p5ld_t3]-=tlocal5;
		t3d[h2_1*h2ld_t3+h1_1*h1ld_t3+h3_1*h3ld_t3+p4_1*p4ld_t3+p6_1*p6ld_t3+p5_1*p5ld_t3]-=tlocal6;
		t3d[h2_2*h2ld_t3+h1_2*h1ld_t3+h3_1*h3ld_t3+p4_2*p4ld_t3+p6_1*p6ld_t3+p5_1*p5ld_t3]-=tlocal7;
		}
		else if(thread_y+T2*1<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_1*h3ld_t3+p4_0*p4ld_t3+p6_1*p6ld_t3+p5_1*p5ld_t3]-=tlocal5;
		t3d[h2_1*h2ld_t3+h1_1*h1ld_t3+h3_1*h3ld_t3+p4_1*p4ld_t3+p6_1*p6ld_t3+p5_1*p5ld_t3]-=tlocal6;
		}
		else if(thread_y+T2*0<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_1*h3ld_t3+p4_0*p4ld_t3+p6_1*p6ld_t3+p5_1*p5ld_t3]-=tlocal5;
		}
	}
	if(thread_x+T1*2<total_x){
		if(thread_y+T2*3<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_2*h3ld_t3+p4_0*p4ld_t3+p6_2*p6ld_t3+p5_2*p5ld_t3]-=tlocal9;
		t3d[h2_1*h2ld_t3+h1_1*h1ld_t3+h3_2*h3ld_t3+p4_1*p4ld_t3+p6_2*p6ld_t3+p5_2*p5ld_t3]-=tlocal10;
		t3d[h2_2*h2ld_t3+h1_2*h1ld_t3+h3_2*h3ld_t3+p4_2*p4ld_t3+p6_2*p6ld_t3+p5_2*p5ld_t3]-=tlocal11;
		t3d[h2_3*h2ld_t3+h1_3*h1ld_t3+h3_2*h3ld_t3+p4_3*p4ld_t3+p6_2*p6ld_t3+p5_2*p5ld_t3]-=tlocal12;
		}
		else if(thread_y+T2*2<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_2*h3ld_t3+p4_0*p4ld_t3+p6_2*p6ld_t3+p5_2*p5ld_t3]-=tlocal9;
		t3d[h2_1*h2ld_t3+h1_1*h1ld_t3+h3_2*h3ld_t3+p4_1*p4ld_t3+p6_2*p6ld_t3+p5_2*p5ld_t3]-=tlocal10;
		t3d[h2_2*h2ld_t3+h1_2*h1ld_t3+h3_2*h3ld_t3+p4_2*p4ld_t3+p6_2*p6ld_t3+p5_2*p5ld_t3]-=tlocal11;
		}
		else if(thread_y+T2*1<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_2*h3ld_t3+p4_0*p4ld_t3+p6_2*p6ld_t3+p5_2*p5ld_t3]-=tlocal9;
		t3d[h2_1*h2ld_t3+h1_1*h1ld_t3+h3_2*h3ld_t3+p4_1*p4ld_t3+p6_2*p6ld_t3+p5_2*p5ld_t3]-=tlocal10;
		}
		else if(thread_y+T2*0<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_2*h3ld_t3+p4_0*p4ld_t3+p6_2*p6ld_t3+p5_2*p5ld_t3]-=tlocal9;
		}
	}
	if(thread_x+T1*3<total_x){
		if(thread_y+T2*3<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_3*h3ld_t3+p4_0*p4ld_t3+p6_3*p6ld_t3+p5_3*p5ld_t3]-=tlocal13;
		t3d[h2_1*h2ld_t3+h1_1*h1ld_t3+h3_3*h3ld_t3+p4_1*p4ld_t3+p6_3*p6ld_t3+p5_3*p5ld_t3]-=tlocal14;
		t3d[h2_2*h2ld_t3+h1_2*h1ld_t3+h3_3*h3ld_t3+p4_2*p4ld_t3+p6_3*p6ld_t3+p5_3*p5ld_t3]-=tlocal15;
		t3d[h2_3*h2ld_t3+h1_3*h1ld_t3+h3_3*h3ld_t3+p4_3*p4ld_t3+p6_3*p6ld_t3+p5_3*p5ld_t3]-=tlocal16;
		}
		else if(thread_y+T2*2<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_3*h3ld_t3+p4_0*p4ld_t3+p6_3*p6ld_t3+p5_3*p5ld_t3]-=tlocal13;
		t3d[h2_1*h2ld_t3+h1_1*h1ld_t3+h3_3*h3ld_t3+p4_1*p4ld_t3+p6_3*p6ld_t3+p5_3*p5ld_t3]-=tlocal14;
		t3d[h2_2*h2ld_t3+h1_2*h1ld_t3+h3_3*h3ld_t3+p4_2*p4ld_t3+p6_3*p6ld_t3+p5_3*p5ld_t3]-=tlocal15;
		}
		else if(thread_y+T2*1<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_3*h3ld_t3+p4_0*p4ld_t3+p6_3*p6ld_t3+p5_3*p5ld_t3]-=tlocal13;
		t3d[h2_1*h2ld_t3+h1_1*h1ld_t3+h3_3*h3ld_t3+p4_1*p4ld_t3+p6_3*p6ld_t3+p5_3*p5ld_t3]-=tlocal14;
		}
		else if(thread_y+T2*0<total_y) {
		t3d[h2_0*h2ld_t3+h1_0*h1ld_t3+h3_3*h3ld_t3+p4_0*p4ld_t3+p6_3*p6ld_t3+p5_3*p5ld_t3]-=tlocal13;
		}
	}
	__syncthreads();
}
void ma_sd_t_d2_8_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d, size_t p7d, double *t3, double *t2, double *v2) 
{
	size_t p7ld_t2,p4ld_t2,h1ld_t2,h2ld_t2,p7ld_v2,h3ld_v2,p6ld_v2,p5ld_v2,h2ld_t3,h1ld_t3,h3ld_t3,p4ld_t3,p6ld_t3,p5ld_t3;
	size_t size_t3,size_block_t3,size_el_block_t3,size_t2,size_v2;
	cudaStream_t *streams;
	size_t nstreams,i;
	double *t2_d,*v2_d;
	size_t3=h2d*h1d*h3d*p4d*p6d*p5d*sizeof(double);
	size_t2=p7d*p4d*h1d*h2d*sizeof(double);
	size_v2=p7d*h3d*p6d*p5d*sizeof(double);
	cudaFuncSetCacheConfig(ma_sd_t_d2_8_kernel, cudaFuncCachePreferShared);
	nstreams=1;
	size_block_t3=size_t3/nstreams;
	size_el_block_t3=size_block_t3/sizeof(double);
	//t3d=(double*)getGpuMem(size_t3);
	t2_d=(double*)getGpuMem(size_t2);
	v2_d=(double*)getGpuMem(size_v2);
	streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
	assert(streams!= NULL);
	for(i=0;i<nstreams;++i) {
		CUDA_SAFE(cudaStreamCreate(&streams[i])) ;
	}
	CUDA_SAFE(cudaMemcpy(t2_d,t2,size_t2,cudaMemcpyHostToDevice));
	CUDA_SAFE(cudaMemcpy(v2_d,v2,size_v2,cudaMemcpyHostToDevice));
	p7ld_t2=1;
	p4ld_t2=p7d;
	h1ld_t2=p4d*p7d;
	h2ld_t2=h1d*p4d*p7d;
	p7ld_v2=1;
	h3ld_v2=p7d;
	p6ld_v2=h3d*p7d;
	p5ld_v2=p6d*h3d*p7d;
	h2ld_t3=1;
	h1ld_t3=h2d;
	h3ld_t3=h1d*h2d;
	p4ld_t3=h3d*h1d*h2d;
	p6ld_t3=p4d*h3d*h1d*h2d;
	p5ld_t3=p6d*p4d*h3d*h1d*h2d;
	size_t total_x = h3d*p6d*p5d;
	size_t total_y = p4d*h1d*h2d;
	dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
	for(i=0;i<nstreams;++i){
		ma_sd_t_d2_8_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(h1d,h2d,h3d,p4d,p5d,p6d,p7d,p7ld_t2,p4ld_t2,h1ld_t2,h2ld_t2,p7ld_v2,h3ld_v2,p6ld_v2,p5ld_v2,h2ld_t3,h1ld_t3,h3ld_t3,p4ld_t3,p6ld_t3,p5ld_t3,ma_t3_d,t2_d,v2_d,i,total_x,total_y);
		CHECK_ERR("Kernel execution failed");
	}
	cudaThreadSynchronize();
	for(i=0;i<nstreams;++i){
		cudaStreamDestroy(streams[i]);}
	//freeGpuMem(t3d);
	freeGpuMem(t2_d);
	freeGpuMem(v2_d);
	free(streams);
}
#undef T1
#undef T2
#undef Tcomm
void ma_sd_t_d2_8_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* p4d, Integer* p5d, Integer* p6d, Integer* p7d, double *t3, double *t2, double *v2) 
{
	ma_sd_t_d2_8_cuda((int)*h1d,(int)*h2d,(int)*h3d,(int)*p4d,(int)*p5d,(int)*p6d,(int)*p7d,t3,t2,v2);
}
/*----------------------------------------------------------------------*
*t3[h2,h3,h1,p4,p6,p5] += t2[p7,p4,h1,h2] * v2[p7,h3,p6,p5]
*----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void ma_sd_t_d2_9_kernel(size_t h1d,size_t h2d,size_t h3d,size_t p4d,size_t p5d,size_t p6d,size_t p7d,
	size_t p7ld_t2,size_t p4ld_t2,size_t h1ld_t2,size_t h2ld_t2,size_t p7ld_v2,size_t h3ld_v2,size_t p6ld_v2,size_t p5ld_v2,
	size_t h2ld_t3,size_t h3ld_t3,size_t h1ld_t3,size_t p4ld_t3,size_t p6ld_t3,size_t p5ld_t3,
	double *t3d, double *t2_d, double *v2_d,size_t unused_idx, size_t total_x, size_t total_y) 
{
	size_t h1_0,h1_1,h1_2,h1_3,h2_0,h2_1,h2_2,h2_3,h3_0,h3_1,h3_2,h3_3,p4_0,p4_1,p4_2,p4_3,p5_0,p5_1,p5_2,p5_3,p6_0,p6_1,p6_2,p6_3,p7;
	double a1,b1;
	double a2,b2;
	double a3,b3;
	double a4,b4;
	size_t in1_idxl,in2_idxl,p7l,p7T;
	__shared__ double t2_shm[4*T1][Tcomm];
	__shared__ double v2_shm[Tcomm][4*T2];
	size_t rest_x=blockIdx.x;
	size_t rest_y=blockIdx.y;
	size_t thread_x = T2*4 * rest_x + threadIdx.x;
	size_t thread_y = T1*4 * rest_y + threadIdx.y;
	in1_idxl=threadIdx.y;
	in2_idxl=threadIdx.x ;
	double tlocal1=0;
	double tlocal2=0;
	double tlocal3=0;
	double tlocal4=0;
	double tlocal5=0;
	double tlocal6=0;
	double tlocal7=0;
	double tlocal8=0;
	double tlocal9=0;
	double tlocal10=0;
	double tlocal11=0;
	double tlocal12=0;
	double tlocal13=0;
	double tlocal14=0;
	double tlocal15=0;
	double tlocal16=0;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
	h2_0=rest_y%h2d;
	rest_y=rest_y/h2d;
	h3_0=rest_x%h3d;
	rest_x=rest_x/h3d;
	h1_0=rest_y%h1d;
	rest_y=rest_y/h1d;
	p6_0=rest_x%p6d;
	rest_x=rest_x/p6d;
	p4_0=rest_y;
	p5_0=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
	h2_1=rest_y%h2d;
	rest_y=rest_y/h2d;
	h3_1=rest_x%h3d;
	rest_x=rest_x/h3d;
	h1_1=rest_y%h1d;
	rest_y=rest_y/h1d;
	p6_1=rest_x%p6d;
	rest_x=rest_x/p6d;
	p4_1=rest_y;
	p5_1=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
	h2_2=rest_y%h2d;
	rest_y=rest_y/h2d;
	h3_2=rest_x%h3d;
	rest_x=rest_x/h3d;
	h1_2=rest_y%h1d;
	rest_y=rest_y/h1d;
	p6_2=rest_x%p6d;
	rest_x=rest_x/p6d;
	p4_2=rest_y;
	p5_2=rest_x;
	rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
	rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
	h2_3=rest_y%h2d;
	rest_y=rest_y/h2d;
	h3_3=rest_x%h3d;
	rest_x=rest_x/h3d;
	h1_3=rest_y%h1d;
	rest_y=rest_y/h1d;
	p6_3=rest_x%p6d;
	rest_x=rest_x/p6d;
	p4_3=rest_y;
	p5_3=rest_x;
	size_t t2_d_off, v2_d_off;for(p7T=0;p7T<p7d;p7T+=Tcomm){
		size_t p7l_hi;
		p7l_hi = MIN(Tcomm+p7T,p7d)-p7T;
		t2_d_off=p4_0*p4ld_t2+h1_0*h1ld_t2+h2_0*h2ld_t2;
		v2_d_off=h3_0*h3ld_v2+p6_0*p6ld_v2+p5_0*p5ld_v2;
		if(thread_y+T1*0<total_y)for(p7l=threadIdx.x;p7l<p7l_hi;p7l+=blockDim.x){
		p7=p7l+p7T;
		t2_shm[in1_idxl+T1*0][p7l] = t2_d[t2_d_off+p7*p7ld_t2];
		}
		if(thread_x+T1*0<total_x)for(p7l=threadIdx.y;p7l<p7l_hi;p7l+=blockDim.y){
		p7=p7l+p7T;
		v2_shm[p7l][in2_idxl+T1*0] = v2_d[v2_d_off+p7*p7ld_v2];
		}
		t2_d_off=p4_1*p4ld_t2+h1_1*h1ld_t2+h2_1*h2ld_t2;
		v2_d_off=h3_1*h3ld_v2+p6_1*p6ld_v2+p5_1*p5ld_v2;
		if(thread_y+T1*1<total_y)for(p7l=threadIdx.x;p7l<p7l_hi;p7l+=blockDim.x){
		p7=p7l+p7T;
		t2_shm[in1_idxl+T1*1][p7l] = t2_d[t2_d_off+p7*p7ld_t2];
		}
		if(thread_x+T1*1<total_x)for(p7l=threadIdx.y;p7l<p7l_hi;p7l+=blockDim.y){
		p7=p7l+p7T;
		v2_shm[p7l][in2_idxl+T1*1] = v2_d[v2_d_off+p7*p7ld_v2];
		}
		t2_d_off=p4_2*p4ld_t2+h1_2*h1ld_t2+h2_2*h2ld_t2;
		v2_d_off=h3_2*h3ld_v2+p6_2*p6ld_v2+p5_2*p5ld_v2;
		if(thread_y+T1*2<total_y)for(p7l=threadIdx.x;p7l<p7l_hi;p7l+=blockDim.x){
		p7=p7l+p7T;
		t2_shm[in1_idxl+T1*2][p7l] = t2_d[t2_d_off+p7*p7ld_t2];
		}
		if(thread_x+T1*2<total_x)for(p7l=threadIdx.y;p7l<p7l_hi;p7l+=blockDim.y){
		p7=p7l+p7T;
		v2_shm[p7l][in2_idxl+T1*2] = v2_d[v2_d_off+p7*p7ld_v2];
		}
		t2_d_off=p4_3*p4ld_t2+h1_3*h1ld_t2+h2_3*h2ld_t2;
		v2_d_off=h3_3*h3ld_v2+p6_3*p6ld_v2+p5_3*p5ld_v2;
		if(thread_y+T1*3<total_y)for(p7l=threadIdx.x;p7l<p7l_hi;p7l+=blockDim.x){
		p7=p7l+p7T;
		t2_shm[in1_idxl+T1*3][p7l] = t2_d[t2_d_off+p7*p7ld_t2];
		}
		if(thread_x+T1*3<total_x)for(p7l=threadIdx.y;p7l<p7l_hi;p7l+=blockDim.y){
		p7=p7l+p7T;
		v2_shm[p7l][in2_idxl+T1*3] = v2_d[v2_d_off+p7*p7ld_v2];
		}
		__syncthreads();
		for(p7l=0;p7l<p7l_hi;++p7l){
		a1=t2_shm[in1_idxl+T1*0][p7l];
		a2=t2_shm[in1_idxl+T1*1][p7l];
		a3=t2_shm[in1_idxl+T1*2][p7l];
		a4=t2_shm[in1_idxl+T1*3][p7l];
		b1=v2_shm[p7l][in2_idxl+T2*0];
		b2=v2_shm[p7l][in2_idxl+T2*1];
		b3=v2_shm[p7l][in2_idxl+T2*2];
		b4=v2_shm[p7l][in2_idxl+T2*3];
		tlocal1+=a1*b1;
		tlocal2+=a2*b1;
		tlocal3+=a3*b1;
		tlocal4+=a4*b1;
		tlocal5+=a1*b2;
		tlocal6+=a2*b2;
		tlocal7+=a3*b2;
		tlocal8+=a4*b2;
		tlocal9+=a1*b3;
		tlocal10+=a2*b3;
		tlocal11+=a3*b3;
		tlocal12+=a4*b3;
		tlocal13+=a1*b4;
		tlocal14+=a2*b4;
		tlocal15+=a3*b4;
		tlocal16+=a4*b4;
		}
		__syncthreads();
	}
	if(thread_x+T1*0<total_x){
		if(thread_y+T2*3<total_y) {
		t3d[h2_0*h2ld_t3+h3_0*h3ld_t3+h1_0*h1ld_t3+p4_0*p4ld_t3+p6_0*p6ld_t3+p5_0*p5ld_t3]+=tlocal1;
		t3d[h2_1*h2ld_t3+h3_0*h3ld_t3+h1_1*h1ld_t3+p4_1*p4ld_t3+p6_0*p6ld_t3+p5_0*p5ld_t3]+=tlocal2;
		t3d[h2_2*h2ld_t3+h3_0*h3ld_t3+h1_2*h1ld_t3+p4_2*p4ld_t3+p6_0*p6ld_t3+p5_0*p5ld_t3]+=tlocal3;
		t3d[h2_3*h2ld_t3+h3_0*h3ld_t3+h1_3*h1ld_t3+p4_3*p4ld_t3+p6_0*p6ld_t3+p5_0*p5ld_t3]+=tlocal4;
		}
		else if(thread_y+T2*2<total_y) {
		t3d[h2_0*h2ld_t3+h3_0*h3ld_t3+h1_0*h1ld_t3+p4_0*p4ld_t3+p6_0*p6ld_t3+p5_0*p5ld_t3]+=tlocal1;
		t3d[h2_1*h2ld_t3+h3_0*h3ld_t3+h1_1*h1ld_t3+p4_1*p4ld_t3+p6_0*p6ld_t3+p5_0*p5ld_t3]+=tlocal2;
		t3d[h2_2*h2ld_t3+h3_0*h3ld_t3+h1_2*h1ld_t3+p4_2*p4ld_t3+p6_0*p6ld_t3+p5_0*p5ld_t3]+=tlocal3;
		}
		else if(thread_y+T2*1<total_y) {
		t3d[h2_0*h2ld_t3+h3_0*h3ld_t3+h1_0*h1ld_t3+p4_0*p4ld_t3+p6_0*p6ld_t3+p5_0*p5ld_t3]+=tlocal1;
		t3d[h2_1*h2ld_t3+h3_0*h3ld_t3+h1_1*h1ld_t3+p4_1*p4ld_t3+p6_0*p6ld_t3+p5_0*p5ld_t3]+=tlocal2;
		}
		else if(thread_y+T2*0<total_y) {
		t3d[h2_0*h2ld_t3+h3_0*h3ld_t3+h1_0*h1ld_t3+p4_0*p4ld_t3+p6_0*p6ld_t3+p5_0*p5ld_t3]+=tlocal1;
		}
	}
	if(thread_x+T1*1<total_x){
		if(thread_y+T2*3<total_y) {
		t3d[h2_0*h2ld_t3+h3_1*h3ld_t3+h1_0*h1ld_t3+p4_0*p4ld_t3+p6_1*p6ld_t3+p5_1*p5ld_t3]+=tlocal5;
		t3d[h2_1*h2ld_t3+h3_1*h3ld_t3+h1_1*h1ld_t3+p4_1*p4ld_t3+p6_1*p6ld_t3+p5_1*p5ld_t3]+=tlocal6;
		t3d[h2_2*h2ld_t3+h3_1*h3ld_t3+h1_2*h1ld_t3+p4_2*p4ld_t3+p6_1*p6ld_t3+p5_1*p5ld_t3]+=tlocal7;
		t3d[h2_3*h2ld_t3+h3_1*h3ld_t3+h1_3*h1ld_t3+p4_3*p4ld_t3+p6_1*p6ld_t3+p5_1*p5ld_t3]+=tlocal8;
		}
		else if(thread_y+T2*2<total_y) {
		t3d[h2_0*h2ld_t3+h3_1*h3ld_t3+h1_0*h1ld_t3+p4_0*p4ld_t3+p6_1*p6ld_t3+p5_1*p5ld_t3]+=tlocal5;
		t3d[h2_1*h2ld_t3+h3_1*h3ld_t3+h1_1*h1ld_t3+p4_1*p4ld_t3+p6_1*p6ld_t3+p5_1*p5ld_t3]+=tlocal6;
		t3d[h2_2*h2ld_t3+h3_1*h3ld_t3+h1_2*h1ld_t3+p4_2*p4ld_t3+p6_1*p6ld_t3+p5_1*p5ld_t3]+=tlocal7;
		}
		else if(thread_y+T2*1<total_y) {
		t3d[h2_0*h2ld_t3+h3_1*h3ld_t3+h1_0*h1ld_t3+p4_0*p4ld_t3+p6_1*p6ld_t3+p5_1*p5ld_t3]+=tlocal5;
		t3d[h2_1*h2ld_t3+h3_1*h3ld_t3+h1_1*h1ld_t3+p4_1*p4ld_t3+p6_1*p6ld_t3+p5_1*p5ld_t3]+=tlocal6;
		}
		else if(thread_y+T2*0<total_y) {
		t3d[h2_0*h2ld_t3+h3_1*h3ld_t3+h1_0*h1ld_t3+p4_0*p4ld_t3+p6_1*p6ld_t3+p5_1*p5ld_t3]+=tlocal5;
		}
	}
	if(thread_x+T1*2<total_x){
		if(thread_y+T2*3<total_y) {
		t3d[h2_0*h2ld_t3+h3_2*h3ld_t3+h1_0*h1ld_t3+p4_0*p4ld_t3+p6_2*p6ld_t3+p5_2*p5ld_t3]+=tlocal9;
		t3d[h2_1*h2ld_t3+h3_2*h3ld_t3+h1_1*h1ld_t3+p4_1*p4ld_t3+p6_2*p6ld_t3+p5_2*p5ld_t3]+=tlocal10;
		t3d[h2_2*h2ld_t3+h3_2*h3ld_t3+h1_2*h1ld_t3+p4_2*p4ld_t3+p6_2*p6ld_t3+p5_2*p5ld_t3]+=tlocal11;
		t3d[h2_3*h2ld_t3+h3_2*h3ld_t3+h1_3*h1ld_t3+p4_3*p4ld_t3+p6_2*p6ld_t3+p5_2*p5ld_t3]+=tlocal12;
		}
		else if(thread_y+T2*2<total_y) {
		t3d[h2_0*h2ld_t3+h3_2*h3ld_t3+h1_0*h1ld_t3+p4_0*p4ld_t3+p6_2*p6ld_t3+p5_2*p5ld_t3]+=tlocal9;
		t3d[h2_1*h2ld_t3+h3_2*h3ld_t3+h1_1*h1ld_t3+p4_1*p4ld_t3+p6_2*p6ld_t3+p5_2*p5ld_t3]+=tlocal10;
		t3d[h2_2*h2ld_t3+h3_2*h3ld_t3+h1_2*h1ld_t3+p4_2*p4ld_t3+p6_2*p6ld_t3+p5_2*p5ld_t3]+=tlocal11;
		}
		else if(thread_y+T2*1<total_y) {
		t3d[h2_0*h2ld_t3+h3_2*h3ld_t3+h1_0*h1ld_t3+p4_0*p4ld_t3+p6_2*p6ld_t3+p5_2*p5ld_t3]+=tlocal9;
		t3d[h2_1*h2ld_t3+h3_2*h3ld_t3+h1_1*h1ld_t3+p4_1*p4ld_t3+p6_2*p6ld_t3+p5_2*p5ld_t3]+=tlocal10;
		}
		else if(thread_y+T2*0<total_y) {
		t3d[h2_0*h2ld_t3+h3_2*h3ld_t3+h1_0*h1ld_t3+p4_0*p4ld_t3+p6_2*p6ld_t3+p5_2*p5ld_t3]+=tlocal9;
		}
	}
	if(thread_x+T1*3<total_x){
		if(thread_y+T2*3<total_y) {
		t3d[h2_0*h2ld_t3+h3_3*h3ld_t3+h1_0*h1ld_t3+p4_0*p4ld_t3+p6_3*p6ld_t3+p5_3*p5ld_t3]+=tlocal13;
		t3d[h2_1*h2ld_t3+h3_3*h3ld_t3+h1_1*h1ld_t3+p4_1*p4ld_t3+p6_3*p6ld_t3+p5_3*p5ld_t3]+=tlocal14;
		t3d[h2_2*h2ld_t3+h3_3*h3ld_t3+h1_2*h1ld_t3+p4_2*p4ld_t3+p6_3*p6ld_t3+p5_3*p5ld_t3]+=tlocal15;
		t3d[h2_3*h2ld_t3+h3_3*h3ld_t3+h1_3*h1ld_t3+p4_3*p4ld_t3+p6_3*p6ld_t3+p5_3*p5ld_t3]+=tlocal16;
		}
		else if(thread_y+T2*2<total_y) {
		t3d[h2_0*h2ld_t3+h3_3*h3ld_t3+h1_0*h1ld_t3+p4_0*p4ld_t3+p6_3*p6ld_t3+p5_3*p5ld_t3]+=tlocal13;
		t3d[h2_1*h2ld_t3+h3_3*h3ld_t3+h1_1*h1ld_t3+p4_1*p4ld_t3+p6_3*p6ld_t3+p5_3*p5ld_t3]+=tlocal14;
		t3d[h2_2*h2ld_t3+h3_3*h3ld_t3+h1_2*h1ld_t3+p4_2*p4ld_t3+p6_3*p6ld_t3+p5_3*p5ld_t3]+=tlocal15;
		}
		else if(thread_y+T2*1<total_y) {
		t3d[h2_0*h2ld_t3+h3_3*h3ld_t3+h1_0*h1ld_t3+p4_0*p4ld_t3+p6_3*p6ld_t3+p5_3*p5ld_t3]+=tlocal13;
		t3d[h2_1*h2ld_t3+h3_3*h3ld_t3+h1_1*h1ld_t3+p4_1*p4ld_t3+p6_3*p6ld_t3+p5_3*p5ld_t3]+=tlocal14;
		}
		else if(thread_y+T2*0<total_y) {
		t3d[h2_0*h2ld_t3+h3_3*h3ld_t3+h1_0*h1ld_t3+p4_0*p4ld_t3+p6_3*p6ld_t3+p5_3*p5ld_t3]+=tlocal13;
		}
	}
	__syncthreads();
}
void ma_sd_t_d2_9_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d, size_t p7d, double *t3, double *t2, double *v2) 
{
	size_t p7ld_t2,p4ld_t2,h1ld_t2,h2ld_t2,p7ld_v2,h3ld_v2,p6ld_v2,p5ld_v2,h2ld_t3,h3ld_t3,h1ld_t3,p4ld_t3,p6ld_t3,p5ld_t3;
	size_t size_t3,size_block_t3,size_el_block_t3,size_t2,size_v2;
	cudaStream_t *streams;
	size_t nstreams,i;
	double *t2_d,*v2_d;
	size_t3=h2d*h3d*h1d*p4d*p6d*p5d*sizeof(double);
	size_t2=p7d*p4d*h1d*h2d*sizeof(double);
	size_v2=p7d*h3d*p6d*p5d*sizeof(double);
	cudaFuncSetCacheConfig(ma_sd_t_d2_9_kernel, cudaFuncCachePreferShared);
	nstreams=1;
	size_block_t3=size_t3/nstreams;
	size_el_block_t3=size_block_t3/sizeof(double);
	//t3d=(double*)getGpuMem(size_t3);
	t2_d=(double*)getGpuMem(size_t2);
	v2_d=(double*)getGpuMem(size_v2);
	streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
	assert(streams!= NULL);
	for(i=0;i<nstreams;++i) {
		CUDA_SAFE(cudaStreamCreate(&streams[i])) ;
	}
	CUDA_SAFE(cudaMemcpy(t2_d,t2,size_t2,cudaMemcpyHostToDevice));
	CUDA_SAFE(cudaMemcpy(v2_d,v2,size_v2,cudaMemcpyHostToDevice));
	p7ld_t2=1;
	p4ld_t2=p7d;
	h1ld_t2=p4d*p7d;
	h2ld_t2=h1d*p4d*p7d;
	p7ld_v2=1;
	h3ld_v2=p7d;
	p6ld_v2=h3d*p7d;
	p5ld_v2=p6d*h3d*p7d;
	h2ld_t3=1;
	h3ld_t3=h2d;
	h1ld_t3=h3d*h2d;
	p4ld_t3=h1d*h3d*h2d;
	p6ld_t3=p4d*h1d*h3d*h2d;
	p5ld_t3=p6d*p4d*h1d*h3d*h2d;
	size_t total_x = h3d*p6d*p5d;
	size_t total_y = p4d*h1d*h2d;
	dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
	for(i=0;i<nstreams;++i){
		ma_sd_t_d2_9_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(h1d,h2d,h3d,p4d,p5d,p6d,p7d,p7ld_t2,p4ld_t2,h1ld_t2,h2ld_t2,p7ld_v2,h3ld_v2,p6ld_v2,p5ld_v2,h2ld_t3,h3ld_t3,h1ld_t3,p4ld_t3,p6ld_t3,p5ld_t3,ma_t3_d,t2_d,v2_d,i,total_x,total_y);
		CHECK_ERR("Kernel execution failed");
	}
	cudaThreadSynchronize();
	for(i=0;i<nstreams;++i){
		cudaStreamDestroy(streams[i]);}
	//freeGpuMem(t3d);
	freeGpuMem(t2_d);
	freeGpuMem(v2_d);
	free(streams);
}

void ma_sd_t_d2_9_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* p4d, Integer* p5d, Integer* p6d, Integer* p7d, double *t3, double *t2, double *v2) 
{
	ma_sd_t_d2_9_cuda((int)*h1d,(int)*h2d,(int)*h3d,(int)*p4d,(int)*p5d,(int)*p6d,(int)*p7d,t3,t2,v2);
}


__global__ void ma_compute_energy_kernel(size_t h1d,size_t h2d,size_t h3d,size_t p4d,size_t p5d,size_t p6d,
	double* eval1,double* eval2,double* eval3,double* eval4,double* eval5,double* eval6, 
	double* energy, double factor, size_t total_size, double* t3d, double* t3_sd)
{
	size_t h1,h2,p6,p4,p5, h3,i=0;
	double e1,e2,e4,e5,e6;
	//  __shared__ double t2_shm[MAX_h3];
	__shared__ double energy_s[T1];
	__shared__ double energy2_s[T1];
	double inner_fac;
	size_t limit;
	size_t rest_x=blockIdx.x;
	size_t thread_x = T2*T1 * rest_x + threadIdx.x;
	if(threadIdx.x==0)
	{
		energy[blockIdx.x]=0;
		energy[blockIdx.x+gridDim.x]=0;
		energy_s[threadIdx.x] = 0.0;
		energy2_s[threadIdx.x] = 0.0;
	}

	// printf("rest_x, thread_x = %d %d %d\n",rest_x,thread_x,T2*T1);

	for(size_t j =0; j<T2*T1;j++) 
	{
		thread_x = T2*T1*blockIdx.x + j;  
		rest_x = thread_x;
		__syncthreads();
		h2=rest_x%h2d;
		rest_x=rest_x/h2d;
		h1=rest_x%h1d;
		rest_x=rest_x/h1d;
		p6=rest_x%p6d;
		rest_x=rest_x/p6d;
		p5=rest_x%p5d;
		rest_x=rest_x/p5d;
		p4=rest_x%p4d;
		e1 = eval1[h1];
		e2 = eval2[h2];
		e4 = eval4[p4];
		e5 = eval5[p5];
		e6 = eval6[p6];

		// printf("e123456= %lf %lf %lf %lf %lf \n",e1,e2,e4,e5,e6);
		/*
		for(p4=0;p4<p4d;p4++) 
		for(p5 = 0;p5<p5d;p5++)
			for(p6=0;p6<p6d;p6++) 
				for(h1= 0;h1<h1d;h1++) 
					for(h2=0;h2<h2d;h2++) 
						for(h3=0;h3<h3d;h3++) {
							inner_fac = -eval4[p4]-eval5[p5]-eval6[p6]+eval1[h1]
								+eval2[h2]+eval3[h3];
							energy_s[0]+=factor*t3d[i]*t3d[i]/inner_fac;
							energy2_s[0]+=factor*t3d[i]*(t3_sd[i]+t3d[i])/inner_fac;
							i++;
						}
		*/
		if(thread_x<total_size)
		for(size_t i=0;i<h3d;i++)
		{
			inner_fac = -e4-e5-e6+e1+e2+eval3[i]; //t2_shm[i];
				//ckbn avoid e1 in case we need just (T)
			energy_s[threadIdx.x] += factor* t3d[thread_x*h3d+i]*t3d[thread_x*h3d+i]/inner_fac;
			energy2_s[threadIdx.x] += factor* t3d[thread_x*h3d+i]*(t3_sd[thread_x*h3d+i]+t3d[thread_x*h3d+i])/inner_fac;

			// printf("inner_fac,e1s,e2s = %lf %lf %lf\n", inner_fac,energy_s[threadIdx.x],energy2_s[threadIdx.x]);
    	}
    	__syncthreads();
	}

  	if(threadIdx.x==0)
	{
		/*	  limit = blockDim.x;
		if (blockIdx.x == (gridDim.x-1)) limit = total_size%blockDim.x;
		for(size_t i=0;i<limit;i++)
		{
			energy[blockIdx.x]+=energy_s[i];
			energy[blockIdx.x+gridDim.x]+=energy2_s[i];
		}
		*/
		energy[blockIdx.x] = energy_s[0];
		energy[blockIdx.x+gridDim.x] = energy2_s[0];
	}
	__syncthreads();
}

void ma_compute_energy(double factor, double* energy, double* eval1, double* eval2,double* eval3,double* eval4,double* eval5,double* eval6,size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d,size_t p6d, double* host1, double* host2)
{
    double* energy_d, *energy_h;
    double* eval_d1,*eval_d2,*eval_d3,*eval_d4,*eval_d5,*eval_d6;
    size_t size_energy = 2*sizeof(double);
    size_t total_block = DIV_UB((h1d*h2d*p4d*p5d*p6d), (T2*T1));

    size_t total_elements = h1d*h2d*p4d*p5d*p6d;
    
    energy_d = (double*)getGpuMem(size_energy*total_block*2);
    size_t i=0,in; 
    double* t3 = (double*)malloc(sizeof(double)*h3d*total_elements);
    double* ts3 = (double*)malloc(sizeof(double)*h3d*total_elements);

    energy_h = (double*)getHostMem(size_energy*2*total_block);
    eval_d1 = (double*)getGpuMem(h1d*sizeof(double));
    eval_d2 = (double*)getGpuMem(h2d*sizeof(double));
    eval_d3 = (double*)getGpuMem(h3d*sizeof(double));
    eval_d4 = (double*)getGpuMem(p4d*sizeof(double));
    eval_d5 = (double*)getGpuMem(p5d*sizeof(double));
    eval_d6 = (double*)getGpuMem(p6d*sizeof(double));

    CUDA_SAFE(cudaMemcpy(eval_d1, eval1, h1d*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_SAFE(cudaMemcpy(eval_d2, eval2, h2d*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_SAFE(cudaMemcpy(eval_d3, eval3, h3d*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_SAFE(cudaMemcpy(eval_d4, eval4, p4d*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_SAFE(cudaMemcpy(eval_d5, eval5, p5d*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_SAFE(cudaMemcpy(eval_d6, eval6, p6d*sizeof(double), cudaMemcpyHostToDevice));

    dim3 dimBlock(1); //T2*T1);
    dim3 dimGrid(total_block);
    ma_compute_energy_kernel<<<dimGrid,dimBlock,0>>>(h1d,h2d,h3d,p4d,p5d,p6d, eval_d1,eval_d2,eval_d3,eval_d4,eval_d5,eval_d6,energy_d, factor, h1d*h2d*p4d*p5d*p6d, ma_t3_d, ma_t3_s);
	cudaThreadSynchronize();
    //CHECK_ERR("Kernel execution failed");
    CUDA_SAFE(cudaMemcpy(((char *) energy_h) , ((char *) energy_d) , 
    size_energy*total_block*2, cudaMemcpyDeviceToHost));

    for(size_t i=1;i<dimGrid.x;i++)
	{
		energy_h[0]+=energy_h[i];
        energy_h[dimGrid.x]+=energy_h[i+dimGrid.x];
	}

    energy[0] = energy_h[0];
    energy[1] = energy_h[dimGrid.x];
    freeGpuMem(energy_d);
    freeGpuMem(eval_d1);
    freeGpuMem(eval_d2);
    freeGpuMem(eval_d3);
    freeGpuMem(eval_d4);
    freeGpuMem(eval_d5);
    freeGpuMem(eval_d6);
    freeHostMem(energy_h);
}

void ma_compute_en_(double * factor, double * energy, double * eval1,double* eval2,double* eval3,double* eval4,double* eval5,double* eval6, Integer * h1d, Integer * h2d, Integer * h3d, Integer * p4d, Integer * p5d, Integer * p6d, double* host1, double* host2)
{
    ma_compute_energy((double) *factor, energy, eval1,eval2, eval3, eval4, eval5, eval6,(int) *h1d, (int) *h2d, (int) *h3d, (int) *p4d, (int) *p5d, (int) *p6d, host1, host2);
}


/*----------------------------------------------------------------------*
*t3[h3,h2,h1,p6,p5,p4] -= t2[p4,h1] * v2[h3,h2,p6,p5]
*----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void ma_sd_t_s1_1_kernel(size_t h1d,size_t h2d,size_t h3d,size_t p4d,size_t p6d,
	size_t p4ld_t2,size_t h1ld_t2,size_t h3ld_v2,size_t h2ld_v2,size_t p6ld_v2,
	size_t h3ld_t3,size_t h2ld_t3,size_t h1ld_t3,size_t p6ld_t3,size_t p4ld_t3, 
	double *t2_d, double *v2_d,size_t p4, size_t total_x, double* t3d) 
{
	size_t h1,h2,h3,p6;
	__shared__ double t2_shm[T1*4*Tcomm];

	for(size_t i=threadIdx.x;i<h1d*p4d;i+=blockDim.x)
	if(i<h1d*p4d) 
	t2_shm[i] = t2_d[i];
	size_t rest_x=blockIdx.x;
	size_t thread_x = T2*T1 * rest_x + threadIdx.x;
	rest_x = thread_x;
		__syncthreads();
	/* the following computation may need to happen inside the loop */
	for(size_t i=0;i<total_x;i+=gridDim.x*blockDim.x)
	{
		rest_x += i;
		h3=rest_x%h3d;
		rest_x=rest_x/h3d;
		h2=rest_x%h2d;
		rest_x=rest_x/h2d;
		p6=rest_x%p6d;

		if((thread_x+i)<total_x)
		for(h1=0;h1<h1d;h1++)
		for(p4=0;p4<p4d;p4++)
		{
			t3d[h3*h3ld_t3+h2*h2ld_t3+h1*h1ld_t3+p6*p6ld_t3+p4*p4ld_t3]+=t2_shm[h1*p4d+p4]*v2_d[h3*h3ld_v2+h2*h2ld_v2+p6*p6ld_v2];
		}
	}
		__syncthreads();
}

void ma_sd_t_s1_1_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d, double *t3, double *t2, double *v2)
{
	double st, et;
	//ckbn    st = timer(); 
	size_t          p7ld_t2, p4ld_t2, h1ld_t2, h2ld_v2, p7ld_v2, h3ld_v2,
					p6ld_v2, p5ld_v2, h3ld_t3, h2ld_t3, h1ld_t3, p6ld_t3,
					p5ld_t3, p4ld_t3;
	size_t          size_t3, size_block_t3, size_el_block_t3, size_t2,
					size_v2;
	cudaStream_t   *streams;
	size_t          nstreams, i;
	double         *t2_d, *v2_d, *t3_p;
	//size_t3 = h3d * h2d * h1d * p6d * p5d * p4d * sizeof(double);
	size_t2 = p4d * h1d * sizeof(double);
	size_v2 = h3d * h2d * p6d * p5d * sizeof(double);
	nstreams = 1;
	size_block_t3 = size_t3 / nstreams;
	size_el_block_t3 = size_block_t3 / sizeof(double);
	//CUDA_SAFE(cudaMalloc((void**) &ma_t3_d, size_t3));
	//CUDA_SAFE(cudaMalloc((void**) &t2_d, size_t2));
	//CUDA_SAFE(cudaMalloc((void**) &v2_d, size_v2));
	//	ma_t3_d = (double *) getGpuMem(size_t3);
	t2_d = (double *) getGpuMem(size_t2);
	v2_d = (double *) getGpuMem(size_v2);
	//t3_p = (double *) getHostMem(size_t3);
	streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));
	assert(streams != NULL);
	for (i = 0; i < nstreams; ++i) {
		CUDA_SAFE(cudaStreamCreate(&streams[i]));
	}
	CUDA_SAFE(cudaMemcpy(t2_d, t2, size_t2, cudaMemcpyHostToDevice));
	CUDA_SAFE(cudaMemcpy(v2_d, v2, size_v2, cudaMemcpyHostToDevice));

	p4ld_t2 = 1;
	h1ld_t2 = p4d;

	h3ld_v2 = 1;
	h2ld_v2 = h3d;
	p6ld_v2 = h3d *  h2d;
	//	p5ld_v2 = p6d * h3d * p7d;
	h3ld_t3 = 1;
	h2ld_t3 = h3d;
	h1ld_t3 = h2d * h3d;
	p6ld_t3 = h1d * h2d * h3d;
	//	p5ld_t3 = p6d * h1d * h2d * h3d;
	p4ld_t3 = p5d * p6d * h1d * h2d * h3d;
	size_t total_x = h3d*h2d*p6d*p5d;
	dim3 dimBlock(T2*T1);dim3 dimGrid(DIV_UB(total_x,T2*T1), 1);
	for(i=0;i<nstreams;++i){
		ma_sd_t_s1_1_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(h1d,h2d,h3d,p4d,p5d*p6d,p4ld_t2,h1ld_t2,h3ld_v2,h2ld_v2,p6ld_v2,h3ld_t3,h2ld_t3,h1ld_t3,p6ld_t3,p4ld_t3,t2_d,v2_d,i,total_x, ma_t3_s);
			CHECK_ERR("Kernel execution failed");
	}
	/*
	st = timer();
	for (i = 0; i < nstreams; ++i) {
		CUDA_SAFE(cudaMemcpyAsync(((char *) t3_p) + i * size_block_t3, ((char *) ma_t3_s) + i * size_block_t3, size_block_t3, cudaMemcpyDeviceToHost, streams[i]));
	}

	stream = 0;
	while (stream < nstreams) {
		while (cudaStreamQuery(streams[stream]) != cudaSuccess);
		double         *src = &t3_p[stream * size_el_block_t3];
		double         *dst = &t3[stream * size_el_block_t3];
		for (i = 0; i < size_el_block_t3; ++i) {
			dst[i] = src[i];
		}
		stream++;
	}
	*/
	cudaThreadSynchronize();

	//	CUDA_SAFE(cudaMemcpy(((char *) t3) , ((char *) ma_t3_s) , size_t3, cudaMemcpyDeviceToHost));
	for (i = 0; i < nstreams; ++i) {
		cudaStreamDestroy(streams[i]);
	}
	//	freeGpuMem(ma_t3_d);
	freeGpuMem(t2_d);
	freeGpuMem(v2_d);
	//  cudaFree(t2_d);
	//  cudaFree(v2_d);
	//freeHostMem(t3_p);
	free(streams);
}
#undef T1
#undef T2
#undef Tcomm
void ma_sd_t_s1_1_cuda_(Integer * h1d, Integer * h2d, Integer * h3d, Integer * p4d, Integer * p5d, Integer * p6d, double *t3, double *t2, double *v2)
{
	ma_sd_t_s1_1_cuda((int) *h1d, (int) *h2d, (int) *h3d, (int) *p4d, (int) *p5d, (int) *p6d,  t3, t2, v2);
}

/*----------------------------------------------------------------------*
*t3[h3,h1,h2,p6,p5,p4] -= t2[p4,h1] * v2[h3,h2,p6,p5]
*----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void ma_sd_t_s1_2_kernel(size_t h1d,size_t h2d,size_t h3d,size_t p4d,size_t p6d,
	size_t p4ld_t2,size_t h1ld_t2,size_t h3ld_v2,size_t h2ld_v2,size_t p6ld_v2,
	size_t h3ld_t3,size_t h2ld_t3,size_t h1ld_t3,size_t p6ld_t3,size_t p4ld_t3,
	double *t2_d, double *v2_d,size_t p4, size_t total_x, double* t3d) 
{
	size_t h1,h2,h3,p6;
	__shared__ double t2_shm[T1*4*Tcomm];

	for(size_t i=threadIdx.x;i<h1d*p4d;i+=blockDim.x)
	if(i<h1d*p4d)
	t2_shm[i] = t2_d[i];
	size_t rest_x=blockIdx.x;
	size_t thread_x = T2*T1 * rest_x + threadIdx.x;
	rest_x = thread_x;
		__syncthreads();
	/* the following computation may need to happen inside the loop */
	for(size_t i=0;i<total_x;i+=gridDim.x*blockDim.x)
	{
		rest_x += i;
		h3=rest_x%h3d;
		rest_x=rest_x/h3d;
		h2=rest_x%h2d;
		rest_x=rest_x/h2d;
		p6=rest_x%p6d;

		if((thread_x+i)<total_x)
		for(h1=0;h1<h1d;h1++)
		for(p4=0;p4<p4d;p4++)
		{
			t3d[h3*h3ld_t3+h2*h2ld_t3+h1*h1ld_t3+p6*p6ld_t3+p4*p4ld_t3]-=t2_shm[h1*p4d+p4]*v2_d[h3*h3ld_v2+h2*h2ld_v2+p6*p6ld_v2];
		}
	}
		__syncthreads();
}

void ma_sd_t_s1_2_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d, double *t3, double *t2, double *v2)
{
	double st, et;
	//ckbn    st = timer(); 
	size_t          p7ld_t2, p4ld_t2, h1ld_t2, h2ld_v2, p7ld_v2, h3ld_v2,
					p6ld_v2, p5ld_v2, h3ld_t3, h2ld_t3, h1ld_t3, p6ld_t3,
					p5ld_t3, p4ld_t3;
	size_t          size_t3, size_block_t3, size_el_block_t3, size_t2,
					size_v2;
	cudaStream_t   *streams;
	size_t          nstreams, i;
	double         *t2_d, *v2_d, *t3_p;
	//size_t3 = h3d * h2d * h1d * p6d * p5d * p4d * sizeof(double);
	size_t2 = p4d * h1d * sizeof(double);
	size_v2 = h3d * h2d * p6d * p5d * sizeof(double);
	nstreams = 1;
	size_block_t3 = size_t3 / nstreams;
	size_el_block_t3 = size_block_t3 / sizeof(double);
	/*    if(first==1)
	{
		ma_t3_d = (double *) getGpuMem(size_t3);
		cudaMemset(ma_t3_d,0,size_t3*sizeof(double));
		first = 0;
	}*/
	//CUDA_SAFE(cudaMalloc((void**) &t2_d, size_t2));
	//CUDA_SAFE(cudaMalloc((void**) &v2_d, size_v2));
	t2_d = (double *) getGpuMem(size_t2);
	v2_d = (double *) getGpuMem(size_v2);
	//t3_p = (double *) getHostMem(size_t3);
	streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));
	/*	assert(streams != NULL);
	for (i = 0; i < nstreams; ++i) {
		CUDA_SAFE(cudaStreamCreate(&streams[i]));
	}*/
	CUDA_SAFE(cudaMemcpy(t2_d, t2, size_t2, cudaMemcpyHostToDevice));
	CUDA_SAFE(cudaMemcpy(v2_d, v2, size_v2, cudaMemcpyHostToDevice));

	p4ld_t2 = 1;
	h1ld_t2 = p4d ;

	h3ld_v2 = 1;
	h2ld_v2 = h3d;
	p6ld_v2 = h3d *  h2d;
	//	p5ld_v2 = p6d * h3d * p7d;
	h3ld_t3 = 1;
	h1ld_t3 = h3d;
	h2ld_t3 = h1d * h3d;
	p6ld_t3 = h1d * h2d * h3d;
	//	p5ld_t3 = p6d * h1d * h2d * h3d;
	p4ld_t3 = p5d * p6d * h1d * h2d * h3d;
	size_t total_x = h3d*h2d*p6d*p5d;
	dim3 dimBlock(T2*T1);dim3 dimGrid(DIV_UB(total_x,T2*T1), 1);
	//  for(i=0;i<nstreams;++i){

	ma_sd_t_s1_2_kernel<<<dimGrid,dimBlock,0>>>(h1d,h2d,h3d,p4d,p5d*p6d,p4ld_t2,h1ld_t2,h3ld_v2,h2ld_v2,p6ld_v2,h3ld_t3,h2ld_t3,h1ld_t3,p6ld_t3,p4ld_t3,t2_d,v2_d,i,total_x, ma_t3_s);
		CHECK_ERR("Kernel execution failed");
	//	}
	/*
	for (i = 0; i < nstreams; ++i) {
		CUDA_SAFE(cudaMemcpyAsync(((char *) t3_p) + i * size_block_t3, ((char *) ma_t3_s) + i * size_block_t3, size_block_t3, cudaMemcpyDeviceToHost, streams[i]));
	}

	stream = 0;
	while (stream < nstreams) {
		while (cudaStreamQuery(streams[stream]) != cudaSuccess);
		double         *src = &t3_p[stream * size_el_block_t3];
		double         *dst = &t3[stream * size_el_block_t3];
		for (i = 0; i < size_el_block_t3; ++i) {
			dst[i] = src[i];
		}
		stream++;
	}*/
	cudaThreadSynchronize();
	//	CUDA_SAFE(cudaMemcpy(((char *) t3) , ((char *) ma_t3_s) , size_t3, cudaMemcpyDeviceToHost));
	/*
	for (i = 0; i < nstreams; ++i) {
		cudaStreamDestroy(streams[i]);
	}*/
	freeGpuMem(t2_d);
	freeGpuMem(v2_d);
	//freeHostMem(t3_p);
	free(streams);
}

void ma_sd_t_s1_2_cuda_(Integer * h1d, Integer * h2d, Integer * h3d, Integer * p4d, Integer * p5d, Integer * p6d, double *t3, double *t2, double *v2)
{
	ma_sd_t_s1_2_cuda((int) *h1d, (int) *h2d, (int) *h3d, (int) *p4d, (int) *p5d, (int) *p6d,  t3, t2, v2);
}
	
void  ma_sd_t_s1_3_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d,  double *t3, double *t2, double *v2)
{
	double st, et;
	//ckbn    st = timer(); 
	size_t          p7ld_t2, p4ld_t2, h1ld_t2, h2ld_v2, p7ld_v2, h3ld_v2,
					p6ld_v2, p5ld_v2, h3ld_t3, h2ld_t3, h1ld_t3, p6ld_t3,
					p5ld_t3, p4ld_t3;
	size_t          size_t3, size_block_t3, size_el_block_t3, size_t2,
					size_v2;
	cudaStream_t   *streams;
	size_t          nstreams, i;
	double         *t2_d, *v2_d, *t3_p;
	//size_t3 = h3d * h2d * h1d * p6d * p5d * p4d * sizeof(double);
	size_t2 = p4d * h1d * sizeof(double);
	size_v2 = h3d * h2d * p6d * p5d * sizeof(double);
	nstreams = 1;
	size_block_t3 = size_t3 / nstreams;
	size_el_block_t3 = size_block_t3 / sizeof(double);
	/*  if(first==1)
		{
			ma_t3_d = (double *) getGpuMem(size_t3);
			cudaMemset(ma_t3_d,0,size_t3*sizeof(double));
			first = 0;
		}
	*/
	t2_d = (double *) getGpuMem(size_t2);
	v2_d = (double *) getGpuMem(size_v2);
	//t3_p = (double *) getHostMem(size_t3);
	streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));
	assert(streams != NULL);
	for (i = 0; i < nstreams; ++i) {
		CUDA_SAFE(cudaStreamCreate(&streams[i]));
	}
	CUDA_SAFE(cudaMemcpy(t2_d, t2, size_t2, cudaMemcpyHostToDevice));
	CUDA_SAFE(cudaMemcpy(v2_d, v2, size_v2, cudaMemcpyHostToDevice));

	p4ld_t2 = 1;
	h1ld_t2 = p4d ;

	h3ld_v2 = 1;
	h2ld_v2 = h3d;
	p6ld_v2 = h3d *  h2d;
	//	p5ld_v2 = p6d * h3d * p7d;
	h1ld_t3 = 1;
	h3ld_t3 = h1d;
	h2ld_t3 = h1d * h3d;
	p6ld_t3 = h1d * h2d * h3d;
	//	p5ld_t3 = p6d * h1d * h2d * h3d;
	p4ld_t3 = p5d * p6d * h1d * h2d * h3d;
	size_t total_x = h3d*h2d*p6d*p5d;
	dim3 dimBlock(T2*T1);dim3 dimGrid(DIV_UB(total_x,T2*T1), 1);
	for(i=0;i<nstreams;++i){
		ma_sd_t_s1_1_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(h1d,h2d,h3d,p4d,p5d*p6d,p4ld_t2,h1ld_t2,h3ld_v2,h2ld_v2,p6ld_v2,h3ld_t3,h2ld_t3,h1ld_t3,p6ld_t3,p4ld_t3,t2_d,v2_d,i,total_x, ma_t3_s);
			CHECK_ERR("Kernel execution failed");
		}
	/*
	for (i = 0; i < nstreams; ++i) {
		CUDA_SAFE(cudaMemcpyAsync(((char *) t3_p) + i * size_block_t3, ((char *) ma_t3_s) + i * size_block_t3, size_block_t3, cudaMemcpyDeviceToHost, streams[i]));
	}

	stream = 0;
	while (stream < nstreams) {
		while (cudaStreamQuery(streams[stream]) != cudaSuccess);
		double         *src = &t3_p[stream * size_el_block_t3];
		double         *dst = &t3[stream * size_el_block_t3];
		for (i = 0; i < size_el_block_t3; ++i) {
			dst[i] = src[i];
		}
		stream++;
	}
	*/	
	cudaThreadSynchronize();
	//CUDA_SAFE(cudaMemcpy(((char *) t3) , ((char *) ma_t3_s) , size_t3, cudaMemcpyDeviceToHost));

	for (i = 0; i < nstreams; ++i) {
		cudaStreamDestroy(streams[i]);
	}
	//	freeGpuMem(ma_t3_d);
	freeGpuMem(t2_d);
	freeGpuMem(v2_d);
	//freeHostMem(t3_p);
	free(streams);
}
#undef T1
#undef T2
#undef Tcomm
void ma_sd_t_s1_3_cuda_(Integer * h1d, Integer * h2d, Integer * h3d, Integer * p4d, Integer * p5d, Integer * p6d,  double *t3, double *t2, double *v2)
{
	ma_sd_t_s1_3_cuda((int) *h1d, (int) *h2d, (int) *h3d, (int) *p4d, (int) *p5d, (int) *p6d, t3, t2, v2);
}
/*----------------------------------------------------------------------*
*t3[h3,h2,h1,p6,p4,p5] -= t2[p4,h1] * v2[h3,h2,p6,p5]
*----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void ma_sd_t_s1_4_kernel(size_t h1d,size_t h2d,size_t h3d,size_t p4d,size_t p5d,size_t p6d,
	size_t p4ld_t2,size_t h1ld_t2,size_t h3ld_v2,size_t h2ld_v2,size_t p6ld_v2,size_t p5ld_v2,
	size_t h3ld_t3,size_t h2ld_t3,size_t h1ld_t3,size_t p6ld_t3,size_t p5ld_t3,size_t p4ld_t3,
	double *t3d, double *t2_d, double *v2_d,size_t p4, size_t total_x) 
{
	size_t h1,h2,h3,p6,p5;
	__shared__ double t2_shm[T1*4*Tcomm];

	for(size_t i=threadIdx.x;i<h1d*p4d;i+=blockDim.x)
	if(i<h1d*p4d)
	t2_shm[i] = t2_d[i];
	size_t rest_x=blockIdx.x;
	size_t thread_x = T2*T1 * rest_x + threadIdx.x;
	rest_x = thread_x;
		__syncthreads();
	/* the following computation may need to happen inside the loop */
	for(size_t i=0;i<total_x;i+=gridDim.x*blockDim.x)
	{
		rest_x += i;
		h3=rest_x%h3d;
		rest_x=rest_x/h3d;
		h2=rest_x%h2d;
		rest_x=rest_x/h2d;
		p6=rest_x%p6d;
		rest_x=rest_x/p6d;
		p5=rest_x%p5d;

		if((thread_x+i)<total_x)
		for(h1=0;h1<h1d;h1++)
		for(p4=0;p4<p4d;p4++)
		{
			t3d[h3*h3ld_t3+h2*h2ld_t3+h1*h1ld_t3+p6*p6ld_t3+p5*p5ld_t3+p4*p4ld_t3]-=t2_shm[h1*p4d+p4]*v2_d[h3*h3ld_v2+h2*h2ld_v2+p6*p6ld_v2+p5*p5ld_v2];
		}
	}
		__syncthreads();
}

void  ma_sd_t_s1_4_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d,  double *t3, double *t2, double *v2)
{
	double st, et;
	//ckbn    st = timer(); 
	size_t          p7ld_t2, p4ld_t2, h1ld_t2, h2ld_v2, p7ld_v2, h3ld_v2,
					p6ld_v2, p5ld_v2, h3ld_t3, h2ld_t3, h1ld_t3, p6ld_t3,
					p5ld_t3, p4ld_t3;
	size_t          size_t3, size_block_t3, size_el_block_t3, size_t2,
					size_v2;
	cudaStream_t   *streams;
	size_t          nstreams, i;
	double         *t2_d, *v2_d, *t3_p;
	//size_t3 = h3d * h2d * h1d * p6d * p5d * p4d * sizeof(double);
	size_t2 = p4d * h1d * sizeof(double);
	size_v2 = h3d * h2d * p6d * p5d * sizeof(double);
	nstreams = 1;
	size_block_t3 = size_t3 / nstreams;
	size_el_block_t3 = size_block_t3 / sizeof(double);
	/*  if(first==1)
		{
			ma_t3_d = (double *) getGpuMem(size_t3);
			cudaMemset(ma_t3_d,0,size_t3*sizeof(double));
			first = 0;
		}
	*/
	//	ma_t3_d = (double *) getGpuMem(size_t3);
	t2_d = (double *) getGpuMem(size_t2);
	v2_d = (double *) getGpuMem(size_v2);
	//t3_p = (double *) getHostMem(size_t3);
	streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));
	/*	assert(streams != NULL);
	for (i = 0; i < nstreams; ++i) {
		CUDA_SAFE(cudaStreamCreate(&streams[i]));
	}*/
	CUDA_SAFE(cudaMemcpy(t2_d, t2, size_t2, cudaMemcpyHostToDevice));
	CUDA_SAFE(cudaMemcpy(v2_d, v2, size_v2, cudaMemcpyHostToDevice));

	p4ld_t2 = 1;
	h1ld_t2 = p4d;

	h3ld_v2 = 1;
	h2ld_v2 = h3d;
	p6ld_v2 = h3d * h2d;
	p5ld_v2 = p6d * h3d * h2d;
	h3ld_t3 = 1;
	h2ld_t3 = h3d;
	h1ld_t3 = h2d * h3d;
	p6ld_t3 = h1d * h2d * h3d;
	p4ld_t3 = p6d * h1d * h2d * h3d;
	p5ld_t3 = p4d * p6d * h1d * h2d * h3d;
	size_t total_x = h3d*h2d*p6d*p5d;
	dim3 dimBlock(T2*T1);dim3 dimGrid(DIV_UB(total_x,T2*T1), 1);
	i=0;
	// for(i=0;i<nstreams;++i){
	ma_sd_t_s1_4_kernel<<<dimGrid,dimBlock,0>>>(h1d,h2d,h3d,p4d,p5d,p6d,p4ld_t2,h1ld_t2,h3ld_v2,h2ld_v2,p6ld_v2,p5ld_v2,h3ld_t3,h2ld_t3,h1ld_t3,p6ld_t3,p5ld_t3,p4ld_t3,ma_t3_s,t2_d,v2_d,i,total_x);
	//ma_sd_t_s1_4_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(h1d,h2d,h3d,p4d,p5d,p6d,p4ld_t2,h1ld_t2,h3ld_v2,h2ld_v2,p6ld_v2,p5ld_v2,h3ld_t3,h2ld_t3,h1ld_t3,p6ld_t3,p5ld_t3,p4ld_t3,ma_t3_d,t2_d,v2_d,i,total_x);
		CHECK_ERR("Kernel execution failed");
	//	}


	cudaThreadSynchronize();
	/*	CUDA_SAFE(cudaMemcpy(((char *) t3_p) , ((char *) ma_t3_d) , size_block_t3, cudaMemcpyDeviceToHost));
	printf("Time for Async DeviceToHost %f\n", et-st);
	stream = 0;
	//	while (stream < nstreams) {
	//		while (cudaStreamQuery(streams[stream]) != cudaSuccess);
			double         *src = t3_p; //[stream * size_el_block_t3];
			double         *dst = t3;  //[stream * size_el_block_t3];
			for (i = 0; i < size_el_block_t3; ++i) {
				dst[i] -= src[i];
			}
	//		stream++;
	//	}
	*/
	//	cudaThreadSynchronize();
	/*
		for (i = 0; i < nstreams; ++i) {
			cudaStreamDestroy(streams[i]);
		}*/
	//	freeGpuMem(ma_t3_d);
	freeGpuMem(t2_d);
	freeGpuMem(v2_d);
	//freeHostMem(t3_p);
	free(streams);
}
#undef T1
#undef T2
#undef Tcomm
void ma_sd_t_s1_4_cuda_(Integer * h1d, Integer * h2d, Integer * h3d, Integer * p4d, Integer * p5d, Integer * p6d, double *t3, double *t2, double *v2)
{
	ma_sd_t_s1_4_cuda((int) *h1d, (int) *h2d, (int) *h3d, (int) *p4d, (int) *p5d, (int) *p6d,  t3, t2, v2);
}

/*----------------------------------------------------------------------*
*t3[h3,h1,h2,p6,p4,p5] -= t2[p4,h1] * v2[h3,h2,p6,p5]
*----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void ma_sd_t_s1_5_kernel(size_t h1d,size_t h2d,size_t h3d,size_t p4d,size_t p5d,size_t p6d,
	size_t p4ld_t2,size_t h1ld_t2,size_t h3ld_v2,size_t h2ld_v2,size_t p6ld_v2,size_t p5ld_v2,
	size_t h3ld_t3,size_t h2ld_t3,size_t h1ld_t3,size_t p6ld_t3,size_t p5ld_t3,size_t p4ld_t3,
	double *t3d, double *t2_d, double *v2_d,size_t p4, size_t total_x) 
{
	size_t h1,h2,h3,p6,p5;
	__shared__ double t2_shm[T1*4*Tcomm];

	for(size_t i=threadIdx.x;i<h1d*p4d;i+=blockDim.x)
	if(i<h1d*p4d)
	t2_shm[i] = t2_d[i];
	size_t rest_x=blockIdx.x;
	size_t thread_x = T2*T1 * rest_x + threadIdx.x;
	rest_x = thread_x;
		__syncthreads();
	/* the following computation may need to happen inside the loop */
	for(size_t i=0;i<total_x;i+=gridDim.x*blockDim.x)
	{
		rest_x += i;
		h3=rest_x%h3d;
		rest_x=rest_x/h3d;
		h2=rest_x%h2d;
		rest_x=rest_x/h2d;
		p6=rest_x%p6d;
		rest_x=rest_x/p6d;
		p5=rest_x%p5d;

		if((thread_x+i)<total_x)
		for(h1=0;h1<h1d;h1++)
		for(p4=0;p4<p4d;p4++)
		{
			t3d[h3*h3ld_t3+h2*h2ld_t3+h1*h1ld_t3+p6*p6ld_t3+p5*p5ld_t3+p4*p4ld_t3]+=t2_shm[h1*p4d+p4]*v2_d[h3*h3ld_v2+h2*h2ld_v2+p6*p6ld_v2+p5*p5ld_v2];
		}
	}
	__syncthreads();
}

void ma_sd_t_s1_5_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d,  double *t3, double *t2, double *v2)
{
	double st, et;
	//ckbn    st = timer(); 
	size_t          p7ld_t2, p4ld_t2, h1ld_t2, h2ld_v2, p7ld_v2, h3ld_v2,
					p6ld_v2, p5ld_v2, h3ld_t3, h2ld_t3, h1ld_t3, p6ld_t3,
					p5ld_t3, p4ld_t3;
	size_t          size_t3, size_block_t3, size_el_block_t3, size_t2,
					size_v2;
	cudaStream_t   *streams;
	size_t          nstreams, i;
	double         *t2_d, *v2_d, *t3_p;
	//size_t3 = h3d * h2d * h1d * p6d * p5d * p4d * sizeof(double);
	size_t2 = p4d * h1d * sizeof(double);
	size_v2 = h3d * h2d * p6d * p5d * sizeof(double);
	nstreams = 1;
	size_block_t3 = size_t3 / nstreams;
	size_el_block_t3 = size_block_t3 / sizeof(double);
	/*  if(first==1)
		{
			ma_t3_d = (double *) getGpuMem(size_t3);
			cudaMemset(ma_t3_d,0,size_t3*sizeof(double));
			first = 0;
		}
	*/
	//	ma_t3_d = (double *) getGpuMem(size_t3);
	t2_d = (double *) getGpuMem(size_t2);
	v2_d = (double *) getGpuMem(size_v2);
	//t3_p = (double *) getHostMem(size_t3);
	streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));
	assert(streams != NULL);
	for (i = 0; i < nstreams; ++i) {
		CUDA_SAFE(cudaStreamCreate(&streams[i]));
	}
	CUDA_SAFE(cudaMemcpy(t2_d, t2, size_t2, cudaMemcpyHostToDevice));
	CUDA_SAFE(cudaMemcpy(v2_d, v2, size_v2, cudaMemcpyHostToDevice));

	p4ld_t2 = 1;
	h1ld_t2 = p4d ;

	h3ld_v2 = 1;
	h2ld_v2 = h3d;
	p6ld_v2 = h3d * h2d;
	p5ld_v2 = p6d * h3d * h2d;
	h3ld_t3 = 1;
	h1ld_t3 = h3d;
	h2ld_t3 = h1d * h3d;
	p6ld_t3 = h1d * h2d * h3d;
	p4ld_t3 = p6d * h1d * h2d * h3d;
	p5ld_t3 = p4d * p6d * h1d * h2d * h3d;
	size_t total_x = h3d*h2d*p6d*p5d;
	dim3 dimBlock(T2*T1);dim3 dimGrid(DIV_UB(total_x,T2*T1), 1);
	for(i=0;i<nstreams;++i)
	{
		ma_sd_t_s1_5_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(h1d,h2d,h3d,p4d,p5d,p6d,p4ld_t2,h1ld_t2,h3ld_v2,h2ld_v2,p6ld_v2,p5ld_v2,h3ld_t3,h2ld_t3,h1ld_t3,p6ld_t3,p5ld_t3,p4ld_t3,ma_t3_s,t2_d,v2_d,i,total_x);
		CHECK_ERR("Kernel execution failed");
	}
	/*
		for (i = 0; i < nstreams; ++i) {
			CUDA_SAFE(cudaMemcpyAsync(((char *) t3_p) + i * size_block_t3, ((char *) ma_t3_s) + i * size_block_t3, size_block_t3, cudaMemcpyDeviceToHost, streams[i]));
		}

		stream = 0;
		while (stream < nstreams) {
			while (cudaStreamQuery(streams[stream]) != cudaSuccess);
			double         *src = &t3_p[stream * size_el_block_t3];
			double         *dst = &t3[stream * size_el_block_t3];
			for (i = 0; i < size_el_block_t3; ++i) {
				dst[i] = src[i];
			}
			stream++;
		}
	*/
	cudaThreadSynchronize();

	//CUDA_SAFE(cudaMemcpy(((char *) t3) , ((char *) ma_t3_s) , size_t3, cudaMemcpyDeviceToHost));
	for (i = 0; i < nstreams; ++i) {
		cudaStreamDestroy(streams[i]);
	}
	//	freeGpuMem(ma_t3_d);
	freeGpuMem(t2_d);
	freeGpuMem(v2_d);
	//freeHostMem(t3_p);
	free(streams);
}
#undef T1
#undef T2
#undef Tcomm
void ma_sd_t_s1_5_cuda_(Integer * h1d, Integer * h2d, Integer * h3d, Integer * p4d, Integer * p5d, Integer * p6d,  double *t3, double *t2, double *v2)
{
	ma_sd_t_s1_5_cuda((int) *h1d, (int) *h2d, (int) *h3d, (int) *p4d, (int) *p5d, (int) *p6d,  t3, t2, v2);
}

/*----------------------------------------------------------------------*
*t3[h1,h3,h2,p6,p4,p5] -= t2[p4,h1] * v2[h3,h2,p6,p5]
*----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void ma_sd_t_s1_6_kernel(size_t h1d,size_t h2d,size_t h3d,size_t p4d,size_t p5d,size_t p6d,
	size_t p4ld_t2,size_t h1ld_t2,size_t h3ld_v2,size_t h2ld_v2,size_t p6ld_v2,size_t p5ld_v2,
	size_t h3ld_t3,size_t h2ld_t3,size_t h1ld_t3,size_t p6ld_t3,size_t p5ld_t3,size_t p4ld_t3,
	double *t3d, double *t2_d, double *v2_d,size_t p4, size_t total_x) 
{
	size_t h1,h2,h3,p6,p5;
	__shared__ double t2_shm[T1*4*Tcomm];

	for(size_t i=threadIdx.x;i<h1d*p4d;i+=blockDim.x)
	if(i<h1d*p4d)
	t2_shm[i] = t2_d[i];
	size_t rest_x=blockIdx.x;
	size_t thread_x = T2*T1 * rest_x + threadIdx.x;
	rest_x = thread_x;
		__syncthreads();
	/* the following computation may need to happen inside the loop */
	for(size_t i=0;i<total_x;i+=gridDim.x*blockDim.x)
	{
		rest_x += i;
		h3=rest_x%h3d;
		rest_x=rest_x/h3d;
		h2=rest_x%h2d;
		rest_x=rest_x/h2d;
		p6=rest_x%p6d;
		rest_x=rest_x/p6d;
		p5=rest_x%p5d;

		if((thread_x+i)<total_x)
		for(h1=0;h1<h1d;h1++)
		for(p4=0;p4<p4d;p4++)
		{
			t3d[h3*h3ld_t3+h2*h2ld_t3+h1*h1ld_t3+p6*p6ld_t3+p5*p5ld_t3+p4*p4ld_t3]-=t2_shm[h1*p4d+p4]*v2_d[h3*h3ld_v2+h2*h2ld_v2+p6*p6ld_v2+p5*p5ld_v2];
		}
	}
	__syncthreads();
}

void ma_sd_t_s1_6_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d,  double *t3, double *t2, double *v2)
{
	double st, et;
	//ckbn    st = timer(); 
	size_t          p7ld_t2, p4ld_t2, h1ld_t2, h2ld_v2, p7ld_v2, h3ld_v2,
					p6ld_v2, p5ld_v2, h3ld_t3, h2ld_t3, h1ld_t3, p6ld_t3,
					p5ld_t3, p4ld_t3;
	size_t          size_t3, size_block_t3, size_el_block_t3, size_t2,
					size_v2;
	cudaStream_t   *streams;
	size_t          nstreams, i;
	double          *t2_d, *v2_d, *t3_p;
	//size_t3 = h3d * h2d * h1d * p6d * p5d * p4d * sizeof(double);
	size_t2 = p4d * h1d * sizeof(double);
	size_v2 = h3d * h2d * p6d * p5d * sizeof(double);
	nstreams = 1;
	size_block_t3 = size_t3 / nstreams;
	size_el_block_t3 = size_block_t3 / sizeof(double);
	/*  if(first==1)
		{
			ma_t3_d = (double *) getGpuMem(size_t3);
			cudaMemset(ma_t3_d,0,size_t3*sizeof(double));
			first = 0;
		}
	*/
	//	ma_t3_d = (double *) getGpuMem(size_t3);
	t2_d = (double *) getGpuMem(size_t2);
	v2_d = (double *) getGpuMem(size_v2);
	//t3_p = (double *) getHostMem(size_t3);
	streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));
	assert(streams != NULL);
	for (i = 0; i < nstreams; ++i) {
		CUDA_SAFE(cudaStreamCreate(&streams[i]));
	}
	CUDA_SAFE(cudaMemcpy(t2_d, t2, size_t2, cudaMemcpyHostToDevice));
	CUDA_SAFE(cudaMemcpy(v2_d, v2, size_v2, cudaMemcpyHostToDevice));

	p4ld_t2 = 1;
	h1ld_t2 = p4d;

	h3ld_v2 = 1;
	h2ld_v2 = h3d;
	p6ld_v2 = h3d * h2d;
	p5ld_v2 = p6d * h3d * h2d;
	h1ld_t3 = 1;
	h3ld_t3 = h1d;
	h2ld_t3 = h1d * h3d;
	p6ld_t3 = h1d * h2d * h3d;
	p4ld_t3 = p6d * h1d * h2d * h3d;
	p5ld_t3 = p4d * p6d * h1d * h2d * h3d;
	size_t total_x = h3d*h2d*p6d*p5d;
	dim3 dimBlock(T2*T1);dim3 dimGrid(DIV_UB(total_x,T2*T1), 1);
	for(i=0;i<nstreams;++i){
		ma_sd_t_s1_6_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(h1d,h2d,h3d,p4d,p5d,p6d,p4ld_t2,h1ld_t2,h3ld_v2,h2ld_v2,p6ld_v2,p5ld_v2,h3ld_t3,h2ld_t3,h1ld_t3,p6ld_t3,p5ld_t3,p4ld_t3,ma_t3_s,t2_d,v2_d,i,total_x);
		CHECK_ERR("Kernel execution failed");
	}
	/*	for (i = 0; i < nstreams; ++i) {
		CUDA_SAFE(cudaMemcpyAsync(((char *) t3_p) + i * size_block_t3, ((char *) ma_t3_s) + i * size_block_t3, size_block_t3, cudaMemcpyDeviceToHost, streams[i]));
	}

	stream = 0;
	while (stream < nstreams) {
		while (cudaStreamQuery(streams[stream]) != cudaSuccess);
		double         *src = &t3_p[stream * size_el_block_t3];
		double         *dst = &t3[stream * size_el_block_t3];
		for (i = 0; i < size_el_block_t3; ++i) {
			dst[i] = src[i];
		}
		stream++;
	}*/
	cudaThreadSynchronize();
	//CUDA_SAFE(cudaMemcpy(((char *) t3) , ((char *) ma_t3_s) , size_t3, cudaMemcpyDeviceToHost));

	for (i = 0; i < nstreams; ++i) {
		cudaStreamDestroy(streams[i]);
	}
	//	freeGpuMem(ma_t3_d);
	freeGpuMem(t2_d);
	freeGpuMem(v2_d);
	//freeHostMem(t3_p);
	free(streams);
}
#undef T1
#undef T2
#undef Tcomm
void ma_sd_t_s1_6_cuda_(Integer * h1d, Integer * h2d, Integer * h3d, Integer * p4d, Integer * p5d, Integer * p6d, double *t3, double *t2, double *v2)
{
	ma_sd_t_s1_6_cuda((int) *h1d, (int) *h2d, (int) *h3d, (int) *p4d, (int) *p5d, (int) *p6d, t3, t2, v2);
}


/*----------------------------------------------------------------------*
*t3[h3,h2,h1,p4,p6,p5] -= t2[p4,h1] * v2[h3,h2,p6,p5]
*----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void ma_sd_t_s1_7_kernel(size_t h1d,size_t h2d,size_t h3d,size_t p4d,size_t p6d,
	size_t p4ld_t2,size_t h1ld_t2,size_t h3ld_v2,size_t h2ld_v2,size_t p6ld_v2,
	size_t h3ld_t3,size_t h2ld_t3,size_t h1ld_t3,size_t p6ld_t3,size_t p4ld_t3,
	double *t3d, double *t2_d, double *v2_d,size_t p4, size_t total_x) 
{
	size_t h1,h2,h3,p6;
	__shared__ double t2_shm[T1*4*Tcomm];

	for(size_t i=threadIdx.x;i<h1d*p4d;i+=blockDim.x)
	if(i<h1d*p4d)
	t2_shm[i] = t2_d[i];
	size_t rest_x=blockIdx.x;
	size_t thread_x = T2*T1 * rest_x + threadIdx.x;
	rest_x = thread_x;
		__syncthreads();
	/* the following computation may need to happen inside the loop */
	for(size_t i=0;i<total_x;i+=gridDim.x*blockDim.x)
	{
		rest_x += i;
		h3=rest_x%h3d;
		rest_x=rest_x/h3d;
		h2=rest_x%h2d;
		rest_x=rest_x/h2d;
		p6=rest_x%p6d;

		if((thread_x+i)<total_x)
		for(h1=0;h1<h1d;h1++)
		for(p4=0;p4<p4d;p4++)
		{
			t3d[h3*h3ld_t3+h2*h2ld_t3+h1*h1ld_t3+p6*p6ld_t3+p4*p4ld_t3]+=t2_shm[h1*p4d+p4]*v2_d[h3*h3ld_v2+h2*h2ld_v2+p6*p6ld_v2];
		}
	}
		__syncthreads();
}

void ma_sd_t_s1_7_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d,  double *t3, double *t2, double *v2)
{
	double st, et;
	//ckbn    st = timer(); 
	size_t          p7ld_t2, p4ld_t2, h1ld_t2, h2ld_v2, p7ld_v2, h3ld_v2,
					p6ld_v2, p5ld_v2, h3ld_t3, h2ld_t3, h1ld_t3, p6ld_t3,
					p5ld_t3, p4ld_t3;
	size_t          size_t3, size_block_t3, size_el_block_t3, size_t2,
					size_v2;
	cudaStream_t   *streams;
	size_t          nstreams, i;
	double         *t2_d, *v2_d, *t3_p;
	//size_t3 = h3d * h2d * h1d * p6d * p5d * p4d * sizeof(double);
	size_t2 = p4d * h1d * sizeof(double);
	size_v2 = h3d * h2d * p6d * p5d * sizeof(double);
	nstreams = 1;
	size_block_t3 = size_t3 / nstreams;
	size_el_block_t3 = size_block_t3 / sizeof(double);
	/*  if(first==1)
		{
			ma_t3_d = (double *) getGpuMem(size_t3);
			cudaMemset(ma_t3_d,0,size_t3*sizeof(double));
			first = 0;
		}
	*/
	//	ma_t3_d = (double *) getGpuMem(size_t3);
	t2_d = (double *) getGpuMem(size_t2);
	v2_d = (double *) getGpuMem(size_v2);
	//t3_p = (double *) getHostMem(size_t3);
	streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));
	assert(streams != NULL);
	for (i = 0; i < nstreams; ++i) {
		CUDA_SAFE(cudaStreamCreate(&streams[i]));
	}
	CUDA_SAFE(cudaMemcpy(t2_d, t2, size_t2, cudaMemcpyHostToDevice));
	CUDA_SAFE(cudaMemcpy(v2_d, v2, size_v2, cudaMemcpyHostToDevice));

	p4ld_t2 = 1;
	h1ld_t2 = p4d;

	h3ld_v2 = 1;
	h2ld_v2 = h3d;
	p6ld_v2 = h3d * h2d;
	//	p5ld_v2 = p6d * h3d * p7d;
	h3ld_t3 = 1;
	h2ld_t3 = h3d;
	h1ld_t3 = h2d * h3d;
	p4ld_t3 = h1d * h2d * h3d;
	//	p5ld_t3 = p6d * h1d * h2d * h3d;
	p6ld_t3 = p4d * h1d * h2d * h3d;
	size_t total_x = h3d*h2d*p6d*p5d;
	dim3 dimBlock(T2*T1);dim3 dimGrid(DIV_UB(total_x,T2*T1), 1);
	for(i=0;i<nstreams;++i){
		ma_sd_t_s1_7_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(h1d,h2d,h3d,p4d,p5d*p6d,p4ld_t2,h1ld_t2,h3ld_v2,h2ld_v2,p6ld_v2,h3ld_t3,h2ld_t3,h1ld_t3,p6ld_t3,p4ld_t3,ma_t3_s,t2_d,v2_d,i,total_x);
			CHECK_ERR("Kernel execution failed");
		}
	/*
	for (i = 0; i < nstreams; ++i) {
		CUDA_SAFE(cudaMemcpyAsync(((char *) t3_p) + i * size_block_t3, ((char *) ma_t3_s) + i * size_block_t3, size_block_t3, cudaMemcpyDeviceToHost, streams[i]));
	}

	stream = 0;
	while (stream < nstreams) {
		while (cudaStreamQuery(streams[stream]) != cudaSuccess);
		double         *src = &t3_p[stream * size_el_block_t3];
		double         *dst = &t3[stream * size_el_block_t3];
		for (i = 0; i < size_el_block_t3; ++i) {
			dst[i] = src[i];
		}
		stream++;
	}*/
	cudaThreadSynchronize();
	//CUDA_SAFE(cudaMemcpy(((char *) t3) , ((char *) ma_t3_s) , size_t3, cudaMemcpyDeviceToHost));

	for (i = 0; i < nstreams; ++i) {
		cudaStreamDestroy(streams[i]);
	}
	//	freeGpuMem(ma_t3_d);
	freeGpuMem(t2_d);
	freeGpuMem(v2_d);
	//freeHostMem(t3_p);
	free(streams);
}
#undef T1
#undef T2
#undef Tcomm
void ma_sd_t_s1_7_cuda_(Integer * h1d, Integer * h2d, Integer * h3d, Integer * p4d, Integer * p5d, Integer * p6d, double *t3, double *t2, double *v2)
{
	ma_sd_t_s1_7_cuda((int) *h1d, (int) *h2d, (int) *h3d, (int) *p4d, (int) *p5d, (int) *p6d, t3, t2, v2);
}
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void ma_sd_t_s1_8_kernel(size_t h1d,size_t h2d,size_t h3d,size_t p4d,size_t p6d,
	size_t p4ld_t2,size_t h1ld_t2,size_t h3ld_v2,size_t h2ld_v2,size_t p6ld_v2,
	size_t h3ld_t3,size_t h2ld_t3,size_t h1ld_t3,size_t p6ld_t3,size_t p4ld_t3,
	double *t3d, double *t2_d, double *v2_d,size_t p4, size_t total_x) 
{
	size_t h1,h2,h3,p6;
	__shared__ double t2_shm[T1*4*Tcomm];

	for(size_t i=threadIdx.x;i<h1d*p4d;i+=blockDim.x)
	if(i<h1d*p4d)
	t2_shm[i] = t2_d[i];
	size_t rest_x=blockIdx.x;
	size_t thread_x = T2*T1 * rest_x + threadIdx.x;
	rest_x = thread_x;
		__syncthreads();
	/* the following computation may need to happen inside the loop */
	for(size_t i=0;i<total_x;i+=gridDim.x*blockDim.x)
	{
		rest_x += i;
		h3=rest_x%h3d;
		rest_x=rest_x/h3d;
		h2=rest_x%h2d;
		rest_x=rest_x/h2d;
		p6=rest_x%p6d;

		if((thread_x+i)<total_x)
		for(h1=0;h1<h1d;h1++)
		for(p4=0;p4<p4d;p4++)
		{
			t3d[h3*h3ld_t3+h2*h2ld_t3+h1*h1ld_t3+p6*p6ld_t3+p4*p4ld_t3]-=t2_shm[h1*p4d+p4]*v2_d[h3*h3ld_v2+h2*h2ld_v2+p6*p6ld_v2];
		}
	}
	__syncthreads();
}
/*----------------------------------------------------------------------*
*t3[h3,h1,h2,p4,p6,p5] -= t2[p4,h1] * v2[h3,h2,p6,p5]
*----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
void ma_sd_t_s1_8_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d,  double *t3, double *t2, double *v2)
{
	double st, et;
	//ckbn    st = timer(); 
	size_t          p7ld_t2, p4ld_t2, h1ld_t2, h2ld_v2, p7ld_v2, h3ld_v2,
					p6ld_v2, p5ld_v2, h3ld_t3, h2ld_t3, h1ld_t3, p6ld_t3,
					p5ld_t3, p4ld_t3;
	size_t          size_t3, size_block_t3, size_el_block_t3, size_t2,
					size_v2;
	cudaStream_t   *streams;
	size_t          nstreams, i;
	double          *t2_d, *v2_d, *t3_p;
	//size_t3 = h3d * h2d * h1d * p6d * p5d * p4d * sizeof(double);
	size_t2 = p4d * h1d * sizeof(double);
	size_v2 = h3d * h2d * p6d * p5d * sizeof(double);
	nstreams = 1;
	size_block_t3 = size_t3 / nstreams;
	size_el_block_t3 = size_block_t3 / sizeof(double);
	/*  if(first==1)
		{
			ma_t3_d = (double *) getGpuMem(size_t3);
			cudaMemset(ma_t3_d,0,size_t3*sizeof(double));
			first = 0;
		}
	*/
	//	ma_t3_d = (double *) getGpuMem(size_t3);
	t2_d = (double *) getGpuMem(size_t2);
	v2_d = (double *) getGpuMem(size_v2);
	//t3_p = (double *) getHostMem(size_t3);
	streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));
	assert(streams != NULL);
	for (i = 0; i < nstreams; ++i) {
		CUDA_SAFE(cudaStreamCreate(&streams[i]));
	}
	CUDA_SAFE(cudaMemcpy(t2_d, t2, size_t2, cudaMemcpyHostToDevice));
	CUDA_SAFE(cudaMemcpy(v2_d, v2, size_v2, cudaMemcpyHostToDevice));

	p4ld_t2 = 1;
	h1ld_t2 = p4d;

	h3ld_v2 = 1;
	h2ld_v2 = h3d;
	p6ld_v2 = h3d * h2d;
	//	p5ld_v2 = p6d * h3d * p7d;
	h3ld_t3 = 1;
	h1ld_t3 = h3d;
	h2ld_t3 = h1d * h3d;
	p4ld_t3 = h1d * h2d * h3d;
	//	p5ld_t3 = p6d * h1d * h2d * h3d;
	p6ld_t3 = p4d * h1d * h2d * h3d;
	size_t total_x = h3d*h2d*p6d*p5d;
	dim3 dimBlock(T2*T1);dim3 dimGrid(DIV_UB(total_x,T2*T1), 1);
	for(i=0;i<nstreams;++i){
		ma_sd_t_s1_8_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(h1d,h2d,h3d,p4d,p5d*p6d,p4ld_t2,h1ld_t2,h3ld_v2,h2ld_v2,p6ld_v2,h3ld_t3,h2ld_t3,h1ld_t3,p6ld_t3,p4ld_t3,ma_t3_s,t2_d,v2_d,i,total_x);
		CHECK_ERR("Kernel execution failed");
	}
	/*
	for (i = 0; i < nstreams; ++i) {
		CUDA_SAFE(cudaMemcpyAsync(((char *) t3_p) + i * size_block_t3, ((char *) ma_t3_s) + i * size_block_t3, size_block_t3, cudaMemcpyDeviceToHost, streams[i]));
	}
	stream = 0;
	while (stream < nstreams) {
		while (cudaStreamQuery(streams[stream]) != cudaSuccess);
		double         *src = &t3_p[stream * size_el_block_t3];
		double         *dst = &t3[stream * size_el_block_t3];
		for (i = 0; i < size_el_block_t3; ++i) {
			dst[i] = src[i];
		}
		stream++;
	}*/
	cudaThreadSynchronize();
	//	CUDA_SAFE(cudaMemcpy(((char *) t3) , ((char *) ma_t3_s) , size_t3, cudaMemcpyDeviceToHost));

	for (i = 0; i < nstreams; ++i) {
		cudaStreamDestroy(streams[i]);
	}
	//	freeGpuMem(ma_t3_d);
	freeGpuMem(t2_d);
	freeGpuMem(v2_d);
	//freeHostMem(t3_p);
	free(streams);
}
		void 
ma_sd_t_s1_8_cuda_(Integer * h1d, Integer * h2d, Integer * h3d, Integer * p4d, Integer * p5d, Integer * p6d, double *t3, double *t2, double *v2)
{
	ma_sd_t_s1_8_cuda((int) *h1d, (int) *h2d, (int) *h3d, (int) *p4d, (int) *p5d, (int) *p6d, t3, t2, v2);
}
/*----------------------------------------------------------------------*
*t3[h1,h3,h2,p4,p6,p5] -= t2[p4,h1] * v2[h3,h2,p6,p5]
*----------------------------------------------------------------------*/
void ma_sd_t_s1_9_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d,  double *t3, double *t2, double *v2)
{
	double st, et;
	//ckbn    st = timer(); 
	size_t          p7ld_t2, p4ld_t2, h1ld_t2, h2ld_v2, p7ld_v2, h3ld_v2,
					p6ld_v2, p5ld_v2, h3ld_t3, h2ld_t3, h1ld_t3, p6ld_t3,
					p5ld_t3, p4ld_t3;
	size_t          size_t3, size_block_t3, size_el_block_t3, size_t2,
					size_v2;
	cudaStream_t   *streams;
	size_t          nstreams, i;
	double          *t2_d, *v2_d, *t3_p;
	//size_t3 = h3d * h2d * h1d * p6d * p5d * p4d * sizeof(double);
	size_t2 = p4d * h1d * sizeof(double);
	size_v2 = h3d * h2d * p6d * p5d * sizeof(double);
	nstreams = 1;
	size_block_t3 = size_t3 / nstreams;
	size_el_block_t3 = size_block_t3 / sizeof(double);
	/*  if(first==1)
		{
			ma_t3_d = (double *) getGpuMem(size_t3);
			cudaMemset(ma_t3_d,0,size_t3*sizeof(double));
			first = 0;
		}
	*/
	//	ma_t3_d = (double *) getGpuMem(size_t3);
	t2_d = (double *) getGpuMem(size_t2);
	v2_d = (double *) getGpuMem(size_v2);
	//t3_p = (double *) getHostMem(size_t3);
	streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));
	assert(streams != NULL);
	for (i = 0; i < nstreams; ++i) {
		CUDA_SAFE(cudaStreamCreate(&streams[i]));
	}
	CUDA_SAFE(cudaMemcpy(t2_d, t2, size_t2, cudaMemcpyHostToDevice));
	CUDA_SAFE(cudaMemcpy(v2_d, v2, size_v2, cudaMemcpyHostToDevice));

	p4ld_t2 = 1;
	h1ld_t2 = p4d;

	h3ld_v2 = 1;
	h2ld_v2 = h3d;
	p6ld_v2 = h3d * h2d;
	//	p5ld_v2 = p6d * h3d * p7d;
	h1ld_t3 = 1;
	h3ld_t3 = h1d;
	h2ld_t3 = h1d * h3d;
	p4ld_t3 = h1d * h2d * h3d;
	//	p5ld_t3 = p6d * h1d * h2d * h3d;
	p6ld_t3 = p4d * h1d * h2d * h3d;
	size_t total_x = h3d*h2d*p6d*p5d;
	dim3 dimBlock(T2*T1);dim3 dimGrid(DIV_UB(total_x,T2*T1), 1);
	for(i=0;i<nstreams;++i){
		ma_sd_t_s1_7_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(h1d,h2d,h3d,p4d,p5d*p6d,p4ld_t2,h1ld_t2,h3ld_v2,h2ld_v2,p6ld_v2,h3ld_t3,h2ld_t3,h1ld_t3,p6ld_t3,p4ld_t3,ma_t3_s,t2_d,v2_d,i,total_x);
			CHECK_ERR("Kernel execution failed");
	}
	/*
	for (i = 0; i < nstreams; ++i) {
		CUDA_SAFE(cudaMemcpyAsync(((char *) t3_p) + i * size_block_t3, ((char *) ma_t3_s) + i * size_block_t3, size_block_t3, cudaMemcpyDeviceToHost, streams[i]));
	}
	stream = 0;
	while (stream < nstreams) {
		while (cudaStreamQuery(streams[stream]) != cudaSuccess);
		double         *src = &t3_p[stream * size_el_block_t3];
		double         *dst = &t3[stream * size_el_block_t3];
		for (i = 0; i < size_el_block_t3; ++i) {
			dst[i] = src[i];
		}
		stream++;
	}*/
	cudaThreadSynchronize();
	//CUDA_SAFE(cudaMemcpy(((char *) t3) , ((char *) ma_t3_s) , size_t3, cudaMemcpyDeviceToHost));

	//  printf("out is %lf\n", t3_p[0]);
	for (i = 0; i < nstreams; ++i) {
		cudaStreamDestroy(streams[i]);
	}
	//freeGpuMem(ma_t3_d);
	freeGpuMem(t2_d);
	freeGpuMem(v2_d);
	//freeHostMem(t3_p);
	free(streams);
}

void ma_sd_t_s1_9_cuda_(Integer * h1d, Integer * h2d, Integer * h3d, Integer * p4d, Integer * p5d, Integer * p6d,  double *t3, double *t2, double *v2)
{
	ma_sd_t_s1_9_cuda((int) *h1d, (int) *h2d, (int) *h3d, (int) *p4d, (int) *p5d, (int) *p6d,  t3, t2, v2);
}


// 
void total_fused_ccsd_t_Ma(size_t base_size_h1b, size_t base_size_h2b, size_t base_size_h3b, 
	size_t base_size_p4b, size_t base_size_p5b, size_t base_size_p6b,
	// 
	double* host_d1_t2_all, double* host_d1_v2_all,
	double* host_d2_t2_all, double* host_d2_v2_all,
	double* host_s1_t2_all, double* host_s1_v2_all,
	// 
	size_t size_d1_t2_all, size_t size_d1_v2_all,
	size_t size_d2_t2_all, size_t size_d2_v2_all,
	size_t size_s1_t2_all, size_t size_s1_v2_all,
	// 
	size_t* list_d1_sizes, 
	size_t* list_d2_sizes, 
	size_t* list_s1_sizes, 
	// 
	std::vector<int> vec_d1_flags,
	std::vector<int> vec_d2_flags,
	std::vector<int> vec_s1_flags,
	// 
	size_t size_noab, size_t size_max_dim_d1_t2, size_t size_max_dim_d1_v2,
	size_t size_nvab, size_t size_max_dim_d2_t2, size_t size_max_dim_d2_v2,
					  size_t size_max_dim_s1_t2, size_t size_max_dim_s1_v2, 
	// 
	double factor, 
	double* host_evl_sorted_h1b, double* host_evl_sorted_h2b, double* host_evl_sorted_h3b, 
	double* host_evl_sorted_p4b, double* host_evl_sorted_p5b, double* host_evl_sorted_p6b,
	double* final_energy_4, double* final_energy_5)
{
	// 
	// initmemmodule();

	// 
	ma_dev_mem_d(base_size_h1b, base_size_h2b, base_size_h3b, base_size_p4b, base_size_p5b, base_size_p6b);
	ma_dev_mem_s(base_size_h1b, base_size_h2b, base_size_h3b, base_size_p4b, base_size_p5b, base_size_p6b);

	// flags
	int* int_list_s1_flags_offsets = (int*)malloc(sizeof(int) * (NUM_IA6_LOOPS * NUM_S1_EQUATIONS));
	int* int_list_d1_flags_offsets = (int*)malloc(sizeof(int) * (NUM_IA6_LOOPS * NUM_D1_EQUATIONS * size_noab));
	int* int_list_d2_flags_offsets = (int*)malloc(sizeof(int) * (NUM_IA6_LOOPS * NUM_D1_EQUATIONS * size_nvab));

	int offset = 0;
	for (int i = 0; i < NUM_IA6_LOOPS * NUM_S1_EQUATIONS; i++)
	{		
		if (vec_s1_flags[i] > 0)
		{
			int_list_s1_flags_offsets[i] = offset++;
		}
		else
		{
			int_list_s1_flags_offsets[i] = -1;
		}
	}

	offset = 0;
	for (int i = 0; i < NUM_IA6_LOOPS * size_noab * NUM_D1_EQUATIONS; i++)
	{
		if (vec_d1_flags[i] > 0)
		{
			int_list_d1_flags_offsets[i] = offset++;
		}
		else
		{
			int_list_d1_flags_offsets[i] = -1;
		}
	}

	offset = 0;
	for (int i = 0; i < NUM_IA6_LOOPS * size_nvab * NUM_D2_EQUATIONS; i++)
	{
		if (vec_d2_flags[i] > 0)
		{
			int_list_d2_flags_offsets[i] = offset++;
		}
		else
		{
			int_list_d2_flags_offsets[i] = -1;
		}
	}

	// 
	for (int i = 0; i < NUM_IA6_LOOPS; i++)
	{
		// doubles
		{
			// sd1
			for (int j = 0; j < size_noab; j++)
			{
				size_t size_idx_h1b = list_d1_sizes[0 + (j + (i) * size_noab) * NUM_D1_INDEX];
				size_t size_idx_h2b = list_d1_sizes[1 + (j + (i) * size_noab) * NUM_D1_INDEX];
				size_t size_idx_h3b = list_d1_sizes[2 + (j + (i) * size_noab) * NUM_D1_INDEX];
				size_t size_idx_h7b = list_d1_sizes[3 + (j + (i) * size_noab) * NUM_D1_INDEX];
				size_t size_idx_p4b = list_d1_sizes[4 + (j + (i) * size_noab) * NUM_D1_INDEX];
				size_t size_idx_p5b = list_d1_sizes[5 + (j + (i) * size_noab) * NUM_D1_INDEX];
				size_t size_idx_p6b = list_d1_sizes[6 + (j + (i) * size_noab) * NUM_D1_INDEX];

				int flag_d1_1 = int_list_d1_flags_offsets[0 + (j + (i) * size_noab) * NUM_D1_EQUATIONS];
				int flag_d1_2 = int_list_d1_flags_offsets[1 + (j + (i) * size_noab) * NUM_D1_EQUATIONS];
				int flag_d1_3 = int_list_d1_flags_offsets[2 + (j + (i) * size_noab) * NUM_D1_EQUATIONS];
				int flag_d1_4 = int_list_d1_flags_offsets[3 + (j + (i) * size_noab) * NUM_D1_EQUATIONS];
				int flag_d1_5 = int_list_d1_flags_offsets[4 + (j + (i) * size_noab) * NUM_D1_EQUATIONS];
				int flag_d1_6 = int_list_d1_flags_offsets[5 + (j + (i) * size_noab) * NUM_D1_EQUATIONS];
				int flag_d1_7 = int_list_d1_flags_offsets[6 + (j + (i) * size_noab) * NUM_D1_EQUATIONS];
				int flag_d1_8 = int_list_d1_flags_offsets[7 + (j + (i) * size_noab) * NUM_D1_EQUATIONS];
				int flag_d1_9 = int_list_d1_flags_offsets[8 + (j + (i) * size_noab) * NUM_D1_EQUATIONS];

				printf ("[%s][ia6=%d][noab=%d] flags: %2d,%2d,%2d,%2d,%2d,%2d,%2d,%2d,%2d\n", __func__, i, j, flag_d1_1,
				flag_d1_2, flag_d1_3, flag_d1_4, flag_d1_5, flag_d1_6, flag_d1_7, flag_d1_8, flag_d1_9);

				// sd1_1
				if (flag_d1_1 >= 0)
				{
					printf ("[%s] sd1_1\n", __func__);
					double* tmp_d1_t2 = host_d1_t2_all + ((int)size_max_dim_d1_t2) * flag_d1_1;
					double* tmp_d1_v2 = host_d1_v2_all + ((int)size_max_dim_d1_v2) * flag_d1_1;
					ma_sd_t_d1_1_cuda(size_idx_h1b, size_idx_h2b, size_idx_h3b, size_idx_p4b, size_idx_p5b, size_idx_p6b, size_idx_h7b,
					ma_t3_d, tmp_d1_t2, tmp_d1_v2);
				}

				// sd1_2
				if (flag_d1_2 >= 0)
				{
					printf ("[%s] sd1_2\n", __func__);
					double* tmp_d1_t2 = host_d1_t2_all + ((int)size_max_dim_d1_t2) * flag_d1_2;
					double* tmp_d1_v2 = host_d1_v2_all + ((int)size_max_dim_d1_v2) * flag_d1_2;
					// ma_sd_t_d1_2_cuda(size_idx_h1b, size_idx_h2b, size_idx_h3b, size_idx_p4b, size_idx_p5b, size_idx_p6b, size_idx_h7b,
					// ma_t3_d, tmp_d1_t2, tmp_d1_v2);
				}

				// sd1_3
				if (flag_d1_3 >= 0)
				{
					printf ("[%s] sd1_3\n", __func__);
					double* tmp_d1_t2 = host_d1_t2_all + size_max_dim_d1_t2 * flag_d1_3;
					double* tmp_d1_v2 = host_d1_v2_all + size_max_dim_d1_v2 * flag_d1_3;
				}

				// sd1_4
				if (flag_d1_4 >= 0)
				{
					printf ("[%s] sd1_4\n", __func__);
					double* tmp_d1_t2 = host_d1_t2_all + size_max_dim_d1_t2 * flag_d1_4;
					double* tmp_d1_v2 = host_d1_v2_all + size_max_dim_d1_v2 * flag_d1_4;
				}

				// sd1_5
				if (flag_d1_5 >= 0)
				{
					printf ("[%s] sd1_5\n", __func__);
					double* tmp_d1_t2 = host_d1_t2_all + size_max_dim_d1_t2 * flag_d1_5;
					double* tmp_d1_v2 = host_d1_v2_all + size_max_dim_d1_v2 * flag_d1_5;
				}

				// sd1_6
				if (flag_d1_6 >= 0)
				{
					printf ("[%s] sd1_6\n", __func__);
					double* tmp_d1_t2 = host_d1_t2_all + size_max_dim_d1_t2 * flag_d1_6;
					double* tmp_d1_v2 = host_d1_v2_all + size_max_dim_d1_v2 * flag_d1_6;
				}

				// sd1_7
				if (flag_d1_7 >= 0)
				{
					printf ("[%s] sd1_7\n", __func__);
					double* tmp_d1_t2 = host_d1_t2_all + size_max_dim_d1_t2 * flag_d1_7;
					double* tmp_d1_v2 = host_d1_v2_all + size_max_dim_d1_v2 * flag_d1_7;
				}

				// sd1_8
				if (flag_d1_8 >= 0)
				{
					printf ("[%s] sd1_8\n", __func__);
					double* tmp_d1_t2 = host_d1_t2_all + size_max_dim_d1_t2 * flag_d1_8;
					double* tmp_d1_v2 = host_d1_v2_all + size_max_dim_d1_v2 * flag_d1_8;
				}

				// sd1_9
				if (flag_d1_9 >= 0)
				{
					printf ("[%s] sd1_9\n", __func__);
					double* tmp_d1_t2 = host_d1_t2_all + size_max_dim_d1_t2 * flag_d1_9;
					double* tmp_d1_v2 = host_d1_v2_all + size_max_dim_d1_v2 * flag_d1_9;
				}
			}

			// sd2
			for (int j = 0; j < size_nvab; j++)
			{
				size_t size_idx_h1b = list_d2_sizes[0 + (j + (i) * size_nvab) * NUM_D2_INDEX];
				size_t size_idx_h2b = list_d2_sizes[1 + (j + (i) * size_nvab) * NUM_D2_INDEX];
				size_t size_idx_h3b = list_d2_sizes[2 + (j + (i) * size_nvab) * NUM_D2_INDEX];
				size_t size_idx_p4b = list_d2_sizes[3 + (j + (i) * size_nvab) * NUM_D2_INDEX];
				size_t size_idx_p5b = list_d2_sizes[4 + (j + (i) * size_nvab) * NUM_D2_INDEX];
				size_t size_idx_p6b = list_d2_sizes[5 + (j + (i) * size_nvab) * NUM_D2_INDEX];
				size_t size_idx_p7b = list_d2_sizes[6 + (j + (i) * size_nvab) * NUM_D2_INDEX];

				int flag_d2_1 = int_list_d2_flags_offsets[0 + (j + (i) * size_nvab) * NUM_D2_EQUATIONS];
				int flag_d2_2 = int_list_d2_flags_offsets[1 + (j + (i) * size_nvab) * NUM_D2_EQUATIONS];
				int flag_d2_3 = int_list_d2_flags_offsets[2 + (j + (i) * size_nvab) * NUM_D2_EQUATIONS];
				int flag_d2_4 = int_list_d2_flags_offsets[3 + (j + (i) * size_nvab) * NUM_D2_EQUATIONS];
				int flag_d2_5 = int_list_d2_flags_offsets[4 + (j + (i) * size_nvab) * NUM_D2_EQUATIONS];
				int flag_d2_6 = int_list_d2_flags_offsets[5 + (j + (i) * size_nvab) * NUM_D2_EQUATIONS];
				int flag_d2_7 = int_list_d2_flags_offsets[6 + (j + (i) * size_nvab) * NUM_D2_EQUATIONS];
				int flag_d2_8 = int_list_d2_flags_offsets[7 + (j + (i) * size_nvab) * NUM_D2_EQUATIONS];
				int flag_d2_9 = int_list_d2_flags_offsets[8 + (j + (i) * size_nvab) * NUM_D2_EQUATIONS];

				printf ("[%s][ia6=%d][nvab=%d] flags: %2d,%2d,%2d,%2d,%2d,%2d,%2d,%2d,%2d\n", __func__, i, j, flag_d2_1,
				flag_d2_2, flag_d2_3, flag_d2_4, flag_d2_5, flag_d2_6, flag_d2_7, flag_d2_8, flag_d2_9);

				// sd2_1
				if (flag_d2_1 >= 0)
				{
					printf ("[%s] sd2_1\n", __func__);
					double* tmp_d2_t2 = host_d2_t2_all + size_max_dim_d2_t2 * flag_d2_1;
					double* tmp_d2_v2 = host_d2_v2_all + size_max_dim_d2_v2 * flag_d2_1;
					ma_sd_t_d2_1_cuda(size_idx_h1b, size_idx_h2b, size_idx_h3b, 
						size_idx_p4b, size_idx_p5b, size_idx_p6b, size_idx_p7b,
						ma_t3_d, tmp_d2_t2, tmp_d2_v2);
				}

				// sd2_2
				if (flag_d2_2 >= 0)
				{
					printf ("[%s] sd2_2\n", __func__);
				}

				// sd2_3
				if (flag_d2_3 >= 0)
				{
					printf ("[%s] sd2_3\n", __func__);
				}

				// sd2_4
				if (flag_d2_4 >= 0)
				{
					printf ("[%s] sd2_4\n", __func__);
				}

				// sd2_5
				if (flag_d2_5 >= 0)
				{
					printf ("[%s] sd2_5\n", __func__);
				}

				// sd2_6
				if (flag_d2_6 >= 0)
				{
					printf ("[%s] sd2_6\n", __func__);
				}

				// sd2_7
				if (flag_d2_7 >= 0)
				{
					printf ("[%s] sd2_7\n", __func__);
				}

				// sd2_8
				if (flag_d2_8 >= 0)
				{
					printf ("[%s] sd2_8\n", __func__);
				}

				// sd2_9
				if (flag_d2_9 >= 0)
				{
					printf ("[%s] sd2_9\n", __func__);
				}
			}
		}

		// singles
		{
			size_t size_idx_h1b = list_s1_sizes[0 + (i) * NUM_S1_INDEX];
			size_t size_idx_h2b = list_s1_sizes[1 + (i) * NUM_S1_INDEX];
			size_t size_idx_h3b = list_s1_sizes[2 + (i) * NUM_S1_INDEX];
			size_t size_idx_p4b = list_s1_sizes[3 + (i) * NUM_S1_INDEX];
			size_t size_idx_p5b = list_s1_sizes[4 + (i) * NUM_S1_INDEX];
			size_t size_idx_p6b = list_s1_sizes[5 + (i) * NUM_S1_INDEX];

			int flag_s1_1 = int_list_s1_flags_offsets[0 + (i) * NUM_S1_EQUATIONS];
			int flag_s1_2 = int_list_s1_flags_offsets[1 + (i) * NUM_S1_EQUATIONS];
			int flag_s1_3 = int_list_s1_flags_offsets[2 + (i) * NUM_S1_EQUATIONS];
			int flag_s1_4 = int_list_s1_flags_offsets[3 + (i) * NUM_S1_EQUATIONS];
			int flag_s1_5 = int_list_s1_flags_offsets[4 + (i) * NUM_S1_EQUATIONS];
			int flag_s1_6 = int_list_s1_flags_offsets[5 + (i) * NUM_S1_EQUATIONS];
			int flag_s1_7 = int_list_s1_flags_offsets[6 + (i) * NUM_S1_EQUATIONS];
			int flag_s1_8 = int_list_s1_flags_offsets[7 + (i) * NUM_S1_EQUATIONS];
			int flag_s1_9 = int_list_s1_flags_offsets[8 + (i) * NUM_S1_EQUATIONS];

			printf ("[%s][s1][ia6=%d] flags: %2d,%2d,%2d,%2d,%2d,%2d,%2d,%2d,%2d\n", __func__, i, flag_s1_1,
			flag_s1_2, flag_s1_3, flag_s1_4, flag_s1_5, flag_s1_6, flag_s1_7, flag_s1_8, flag_s1_9);


			// s1_1
			if (vec_s1_flags[0 + (i) * NUM_S1_EQUATIONS] == 1)
			{
				printf ("[%s] s1_1\n", __func__);
			}

			// s1_2
			if (vec_s1_flags[1 + (i) * NUM_S1_EQUATIONS] == 1)
			{
				printf ("[%s] s1_2\n", __func__);
			}

			// s1_3
			if (vec_s1_flags[2 + (i) * NUM_S1_EQUATIONS] == 1)
			{
				printf ("[%s] s1_3\n", __func__);
			}

			// s1_4
			if (vec_s1_flags[3 + (i) * NUM_S1_EQUATIONS] == 1)
			{
				printf ("[%s] s1_4\n", __func__);
			}

			// s1_5
			if (vec_s1_flags[4 + (i) * NUM_S1_EQUATIONS] == 1)
			{
				printf ("[%s] s1_5\n", __func__);
			}

			// s1_6
			if (vec_s1_flags[5 + (i) * NUM_S1_EQUATIONS] == 1)
			{
				printf ("[%s] s1_6\n", __func__);
			}

			// s1_7
			if (vec_s1_flags[6 + (i) * NUM_S1_EQUATIONS] == 1)
			{
				printf ("[%s] s1_7\n", __func__);
			}

			// s1_8
			if (vec_s1_flags[7 + (i) * NUM_S1_EQUATIONS] == 1)
			{
				printf ("[%s] s1_8\n", __func__);
			}

			// s1_9
			if (vec_s1_flags[8 + (i) * NUM_S1_EQUATIONS] == 1)
			{
				printf ("[%s] s1_9\n", __func__);
			}
		}
	}


	// 
	double* final_energies = (double*)malloc(sizeof(double) * 2);
	final_energies[0] = 0.0;
	final_energies[1] = 0.0;
	double* temp;
	ma_compute_energy(factor, final_energies,
					host_evl_sorted_h1b, host_evl_sorted_h2b, host_evl_sorted_h3b, 
					host_evl_sorted_p4b, host_evl_sorted_p5b, host_evl_sorted_p6b,
					base_size_h1b, base_size_h2b, base_size_h3b,
					base_size_p4b, base_size_p5b, base_size_p6b,
					temp, temp);

	printf ("[%s][WenJing] %.15f, %.15f\n", __func__, final_energies[0], final_energies[1]);

	ma_dev_release();
	// finalizememmodule();

	free(int_list_s1_flags_offsets);
	free(int_list_d1_flags_offsets);
	free(int_list_d2_flags_offsets);
}