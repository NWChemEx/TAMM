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
// 
// __constant__ int const_s1_ia6[NUM_S1_EQUATIONS];			// 9 
// __constant__ int const_d1_ia6[NUM_D1_EQUATIONS * MAX_NOAB];	// 9 * noab
// __constant__ int const_d2_ia6[NUM_D2_EQUATIONS * MAX_NVAB];	// 9 * nvab
// // 
// __constant__ int const_s1_sizes_ext[NUM_IA6_LOOPS * NUM_S1_INDEX];	// 9 * 6
// __constant__ int const_d1_sizes_ext[NUM_IA6_LOOPS * NUM_D1_INDEX];	// 9 * 6
// __constant__ int const_d2_sizes_ext[NUM_IA6_LOOPS * NUM_D2_INDEX];	// 9 * 6
// // 
// __constant__ int const_s1_flags_offset[NUM_S1_EQUATIONS];
// __constant__ int const_d1_flags_offset[NUM_D1_EQUATIONS * MAX_NOAB];
// __constant__ int const_d2_flags_offset[NUM_D2_EQUATIONS * MAX_NVAB];
// 
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

				// 
				// if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0)
				// {
				// 	printf ("[sd1][part1][ia6 = %2d] h1,h2,h3,p4,p5,p6 = %2d,%2d,%2d,%2d,%2d,%2d\n", iter_ia6, base_size_h1b, base_size_h2b, base_size_h3b, base_size_p4b, base_size_p5b, base_size_p6b);
				// }

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

				// // 
				// if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0)
				// {
				// 	printf ("[sd2][part1][ia6 = %2d] h1,h2,h3,p4,p5,p6 = %2d,%2d,%2d,%2d,%2d,%2d\n", iter_ia6, base_size_h1b, base_size_h2b, base_size_h3b, base_size_p4b, base_size_p5b, base_size_p6b);
				// }

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

				// 
				// if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0)
				// {
				// 	printf ("[sd1][part2][ia6 = %2d] h1,h2,h3,p4,p5,p6 = %2d,%2d,%2d,%2d,%2d,%2d\n", iter_ia6, base_size_h1b, base_size_h2b, base_size_h3b, base_size_p4b, base_size_p5b, base_size_p6b);
				// }

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

				// // 
				// if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0)
				// {
				// 	printf ("[sd2][part2][ia6 = %2d] h1,h2,h3,p4,p5,p6 = %2d,%2d,%2d,%2d,%2d,%2d\n", iter_ia6, base_size_h1b, base_size_h2b, base_size_h3b, base_size_p4b, base_size_p5b, base_size_p6b);
				// }

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

					// if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0)
					// {
					// 	printf ("d2_t2_6[0] = %.15f\n", tmp_dev_d2_t2_6[0]);
					// 	printf ("d2_t2_6[1] = %.15f\n", tmp_dev_d2_t2_6[1]);
					// }

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

			// 
			// if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0)
			// {
			// 	printf ("[s1][ia6 = %2d] h1,h2,h3,p4,p5,p6 = %2d,%2d,%2d,%2d,%2d,%2d\n", iter_ia6, base_size_h1b, base_size_h2b, base_size_h3b, base_size_p4b, base_size_p5b, base_size_p6b);
			// }

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
						std::vector<int> vec_s1_flags, std::vector<int> vec_s1_ai6, 
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
	double* dev_d1_t2_all = (double*)getGpuMem(size_d1_t2_all * sizeof(double));
	double* dev_d1_v2_all = (double*)getGpuMem(size_d1_v2_all * sizeof(double));
	double* dev_d2_t2_all = (double*)getGpuMem(size_d2_t2_all * sizeof(double));
	double* dev_d2_v2_all = (double*)getGpuMem(size_d2_v2_all * sizeof(double));
	double* dev_s1_t2_all = (double*)getGpuMem(size_s1_t2_all * sizeof(double));
	double* dev_s1_v2_all = (double*)getGpuMem(size_s1_v2_all * sizeof(double));

	double* dev_evl_sorted_h1b = (double*)getGpuMem(base_size_h1b * sizeof(double));
	double* dev_evl_sorted_h2b = (double*)getGpuMem(base_size_h2b * sizeof(double));
	double* dev_evl_sorted_h3b = (double*)getGpuMem(base_size_h3b * sizeof(double));
	double* dev_evl_sorted_p4b = (double*)getGpuMem(base_size_p4b * sizeof(double));
	double* dev_evl_sorted_p5b = (double*)getGpuMem(base_size_p5b * sizeof(double));
	double* dev_evl_sorted_p6b = (double*)getGpuMem(base_size_p6b * sizeof(double));

	// 
	cudaMemcpyToSymbol(const_list_s1_flags_offset, &vec_s1_flags[0], sizeof(int) * (NUM_IA6_LOOPS * NUM_S1_EQUATIONS));
	cudaMemcpyToSymbol(const_list_d1_flags_offset, &vec_d1_flags[0], sizeof(int) * (NUM_IA6_LOOPS * MAX_NOAB * NUM_D1_EQUATIONS));
	cudaMemcpyToSymbol(const_list_d2_flags_offset, &vec_d2_flags[0], sizeof(int) * (NUM_IA6_LOOPS * MAX_NVAB * NUM_D2_EQUATIONS));

	// 
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
	// memset(host_energies, 0.0, num_blocks_kernel_1 * NUM_ENERGIES * sizeof(double));
	double* dev_energies = (double*)getGpuMem(num_blocks_kernel_1 * NUM_ENERGIES * sizeof(double));

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
	// final_energy_1 *= factor;
	// final_energy_2 *= factor;
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
	freeGpuMem(dev_s1_t2_all);	freeGpuMem(dev_s1_v2_all);
	freeGpuMem(dev_d1_t2_all);	freeGpuMem(dev_d1_v2_all);
	freeGpuMem(dev_d2_t2_all);	freeGpuMem(dev_d2_v2_all);

	freeGpuMem(dev_evl_sorted_h1b); freeGpuMem(dev_evl_sorted_h2b); freeGpuMem(dev_evl_sorted_h3b);
	freeGpuMem(dev_evl_sorted_p4b); freeGpuMem(dev_evl_sorted_p5b); freeGpuMem(dev_evl_sorted_p6b);

	freeGpuMem(dev_energies);
	free(host_energies);
}