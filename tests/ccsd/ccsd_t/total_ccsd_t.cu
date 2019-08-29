#include "header.hpp"

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

#define NUM_ENERGIES            2
#define FULL_MASK 0xffffffff

//
__constant__ int list_stride_t2[9]; // unused
__constant__ int list_stride_v2[9];

//
//  
//
__global__ void jk_ccsd_t_doubles_fully_fused_kernel(
    //  doubles (sd1)
    double* dev_d1_t2_1, double* dev_d1_v2_1, double* dev_d1_t2_2, double* dev_d1_v2_2, double* dev_d1_t2_3, double* dev_d1_v2_3, 
    double* dev_d1_t2_4, double* dev_d1_v2_4, double* dev_d1_t2_5, double* dev_d1_v2_5, double* dev_d1_t2_6, double* dev_d1_v2_6, 
    double* dev_d1_t2_7, double* dev_d1_v2_7, double* dev_d1_t2_8, double* dev_d1_v2_8, double* dev_d1_t2_9, double* dev_d1_v2_9, 
    int opt_d1_1, int opt_d1_2, int opt_d1_3, 
    int opt_d1_4, int opt_d1_5, int opt_d1_6, 
    int opt_d1_7, int opt_d1_8, int opt_d1_9, 
    //  doubles (sd2)
    double* dev_d2_t2_1, double* dev_d2_v2_1, double* dev_d2_t2_2, double* dev_d2_v2_2, double* dev_d2_t2_3, double* dev_d2_v2_3, 
    double* dev_d2_t2_4, double* dev_d2_v2_4, double* dev_d2_t2_5, double* dev_d2_v2_5, double* dev_d2_t2_6, double* dev_d2_v2_6, 
    double* dev_d2_t2_7, double* dev_d2_v2_7, double* dev_d2_t2_8, double* dev_d2_v2_8, double* dev_d2_t2_9, double* dev_d2_v2_9, 
    int opt_d2_1, int opt_d2_2, int opt_d2_3, 
    int opt_d2_4, int opt_d2_5, int opt_d2_6, 
    int opt_d2_7, int opt_d2_8, int opt_d2_9, 
    //  single
    double* dev_s1_t2_1, double* dev_s1_v2_1, double* dev_s1_t2_2, double* dev_s1_v2_2, double* dev_s1_t2_3, double* dev_s1_v2_3, 
    double* dev_s1_t2_4, double* dev_s1_v2_4, double* dev_s1_t2_5, double* dev_s1_v2_5, double* dev_s1_t2_6, double* dev_s1_v2_6, 
    double* dev_s1_t2_7, double* dev_s1_v2_7, double* dev_s1_t2_8, double* dev_s1_v2_8, double* dev_s1_t2_9, double* dev_s1_v2_9, 
    int opt_s1_1, int opt_s1_2, int opt_s1_3, 
    int opt_s1_4, int opt_s1_5, int opt_s1_6, 
    int opt_s1_7, int opt_s1_8, int opt_s1_9, 
    //  energies
    const double* dev_eval_h1, const double* dev_eval_h2, const double* dev_eval_h3, const double* dev_eval_p4, const double* dev_eval_p5, const double* dev_eval_p6, 
    double* reduced_energy, double factor,
    //  common
    int num_blks_h3, int num_blks_h2, int num_blks_h1, int num_blks_p6, int num_blks_p5, int num_blks_p4, 
    int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_h7, int size_p7, 
    int size_internal,
    int stride_reg_x, int stride_reg_y)
{
    // For Shared Memory,
	__shared__ double sm_a[16][64 + 1];
    __shared__ double sm_b[16][64 + 1];
    
	int internal_upperbound   = 0;
	int internal_offset;

	// should support for non-full tiles
	int idx_h3 = threadIdx.x % FUSION_SIZE_SLICE_1_H3;
	int idx_h2 = threadIdx.x / FUSION_SIZE_SLICE_1_H3;
	int idx_p6 = threadIdx.y % FUSION_SIZE_SLICE_1_P6;
	int idx_h1 = threadIdx.y / FUSION_SIZE_SLICE_1_P6;
       
    int blk_idx_p4  = blockIdx.x / (num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6 * num_blks_p5);
    int tmp_blkIdx  = blockIdx.x % (num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6 * num_blks_p5);
    int blk_idx_p5  = (tmp_blkIdx) / (num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6);
    tmp_blkIdx      = (tmp_blkIdx) % (num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6);
    int blk_idx_p6  = (tmp_blkIdx) / (num_blks_h3 * num_blks_h2 * num_blks_h1);
    tmp_blkIdx      = (tmp_blkIdx) % (num_blks_h3 * num_blks_h2 * num_blks_h1);
    int blk_idx_h1  = (tmp_blkIdx) / (num_blks_h3 * num_blks_h2);
    tmp_blkIdx      = (tmp_blkIdx) % (num_blks_h3 * num_blks_h2);
    int blk_idx_h2  = (tmp_blkIdx) / (num_blks_h3);
    int blk_idx_h3  = blockIdx.x % (num_blks_h3);

    // 
    int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;
    if ((size_h3 - (blk_idx_h3 * FUSION_SIZE_SLICE_1_H3)) >= FUSION_SIZE_SLICE_1_H3)
    {
        rng_h3 = FUSION_SIZE_SLICE_1_H3;
    }
    else
    {
        rng_h3 = size_h3 % FUSION_SIZE_SLICE_1_H3;
    }
    
    if ((size_h2 - (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2)) >= FUSION_SIZE_SLICE_1_H2)
    {
        rng_h2 = FUSION_SIZE_SLICE_1_H2;
    }
    else
    {
        rng_h2 = size_h2 % FUSION_SIZE_SLICE_1_H2;
    }

    if ((size_h1 - (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1)) >= FUSION_SIZE_SLICE_1_H1)
    {
        rng_h1 = FUSION_SIZE_SLICE_1_H1;
    }
    else
    {
        rng_h1 = size_h1 % FUSION_SIZE_SLICE_1_H1;
    }
    
    if ((size_p6 - (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6)) >= FUSION_SIZE_SLICE_1_P6)
    {
        rng_p6 = FUSION_SIZE_SLICE_1_P6;
    }
    else
    {
        rng_p6 = size_p6 % FUSION_SIZE_SLICE_1_P6;
    }

    if ((size_p5 - (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5)) >= FUSION_SIZE_SLICE_1_P5)
    {
        rng_p5 = FUSION_SIZE_SLICE_1_P5;
    }
    else
    {
        rng_p5 = size_p5 % FUSION_SIZE_SLICE_1_P5;
    }

    if ((size_p4 - (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4)) >= FUSION_SIZE_SLICE_1_P4)
    {
        rng_p4 = FUSION_SIZE_SLICE_1_P4;
    }
    else
    {
        rng_p4 = size_p4 % FUSION_SIZE_SLICE_1_P4;
    }

	double temp_av;
	double temp_bv[4];
    double reg_tile[4][4];
    double reg_singles[4][4];

	for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
    {
        reg_tile[i][j]      = 0.0;
        reg_singles[i][j]   = 0.0;
    }


	//
	//	sd1_1,2,3
	//
	// tensor contraction
	#pragma unroll 1
	for (int l = 0; l < size_internal && opt_d1_1 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of size_internal Indices: 1
		if (idx_p6 < rng_p4 && idx_h1 < rng_h1 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p5; ll++)
		{
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = dev_d1_t2_1[(blk_idx_p4 * FUSION_SIZE_SLICE_2_P4 + idx_p6 + (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5 + ll + (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1 + idx_h1) * size_p5) * size_p4) * size_h7 + (threadIdx.x + l)];
		}

		// Load Input Tensor to Shared Memory
		if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p6; ll++)
		{
            sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_2_X] = dev_d1_v2_1[blk_idx_h3 * FUSION_SIZE_SLICE_2_H3 + idx_h3 + (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2 + idx_h2 + (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6 + ll) * size_h2) * size_h3 + (threadIdx.y + l) * list_stride_v2[6]];
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

	// tensor contraction
	internal_upperbound = 0;
	#pragma unroll 1
	for (int l = 0; l < size_internal && opt_d1_2 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of size_internal Indices: 1
		if (idx_p6 < rng_p4 && idx_h1 < rng_h2 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p5; ll++)
		{
			sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = dev_d1_t2_2[(blk_idx_p4 * FUSION_SIZE_SLICE_2_P4 + idx_p6 + (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5 + ll + (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2 + idx_h1) * size_p5) * size_p4) * size_h7 + (threadIdx.x + l)];
		}

		// Load Input Tensor to Shared Memory
		if (idx_h3 < rng_h3 && idx_h2 < rng_h1 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p6; ll++)
		{
			sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_2_X] = dev_d1_v2_2[(blk_idx_h3 * FUSION_SIZE_SLICE_2_H3 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6 + ll) * size_h1) * size_h3) + (threadIdx.y + l) * list_stride_v2[7]];
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

	// tensor contraction
	internal_upperbound = 0;
	#pragma unroll 1
	for (int l = 0; l < size_internal && opt_d1_3 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of size_internal Indices: 1
		if (idx_p6 < rng_p4 && idx_h1 < rng_h3 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p5; ll++)
		{
			sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = dev_d1_t2_3[(blk_idx_p4 * FUSION_SIZE_SLICE_2_P4 + idx_p6 + (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5 + ll + (blk_idx_h3 * FUSION_SIZE_SLICE_2_H3 + idx_h1) * size_p5) * size_p4) * size_h7 + (threadIdx.x + l)];
		}

		// Load Input Tensor to Shared Memory
		if (idx_h3 < rng_h2 && idx_h2 < rng_h1 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p6; ll++)
		{
			sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_2_X] = dev_d1_v2_3[(blk_idx_h2 * FUSION_SIZE_SLICE_2_H2 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1 + idx_h2 + (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6 + ll) * size_h1) * size_h2) + (threadIdx.y + l) * list_stride_v2[8]];
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

    //
    //	sd2_7, 8 and 9
    //
    // tensor contraction
    internal_upperbound = 0;
    #pragma unroll 1
    for (int l = 0; l < size_internal && opt_d2_7 == 1; l+= FUSION_SIZE_INT_UNIT)
    {
        // Part: Generalized Contraction Index (p7b)
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
        if (internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of size_internal Indices: 1
        if (idx_p6 < rng_h1 && idx_h1 < rng_h2 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p6; ll++)
        {
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = dev_d2_t2_7[(blk_idx_p6 *  FUSION_SIZE_SLICE_2_P6 + ll + (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1 + idx_p6 + (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2 + idx_h1) * size_h1) * size_p6) * size_p7 + (threadIdx.x + l)];
        }

        // Load Input Tensor to Shared Memory
        if (idx_p6 < rng_h3 && idx_h1 < rng_p4 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p5; ll++)
        {
            sm_b[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = dev_d2_v2_7[(blk_idx_h3 * FUSION_SIZE_SLICE_2_H3 + idx_p6 + (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5 + ll + (blk_idx_p4 * FUSION_SIZE_SLICE_2_P4 + idx_h1) * size_p5) * size_h3) * size_p7 + (threadIdx.x + l)];
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

    // tensor contraction
    internal_upperbound = 0;
    #pragma unroll 1
    for (int l = 0; l < size_internal && opt_d2_8 == 1; l+= FUSION_SIZE_INT_UNIT)
    {
        // Part: Generalized Contraction Index (p7b)
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
        if (internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of size_internal Indices: 1
        if (idx_p6 < rng_h2 && idx_h1 < rng_h3 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p6; ll++)
        {
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = dev_d2_t2_8[(blk_idx_p6 * FUSION_SIZE_SLICE_2_P6 + ll + (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2 + idx_p6 + (blk_idx_h3 * FUSION_SIZE_SLICE_2_H3 + idx_h1) * size_h2) * size_p6) * size_p7 + (threadIdx.x + l)];
        }

        // Load Input Tensor to Shared Memory
        if (idx_p6 < rng_h1 && idx_h1 < rng_p4 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p5; ll++)
        {
            sm_b[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = dev_d2_v2_8[(blk_idx_h1 * FUSION_SIZE_SLICE_2_H1 + idx_p6 + (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5 + ll + (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + idx_h1) * size_p5) * size_h1) * size_p7 + (threadIdx.x + l)];
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

    // tensor contraction
    internal_upperbound = 0;
    #pragma unroll 1
    for (int l = 0; l < size_internal && opt_d2_9 == 1; l+= FUSION_SIZE_INT_UNIT)
    {
        // Part: Generalized Contraction Index (p7b)
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
        if (internal_offset > rng_p6) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of size_internal Indices: 1
        if (idx_p6 < rng_h1 && idx_h1 < rng_h3 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p6; ll++)
        {
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = dev_d2_t2_9[(blk_idx_p6 * FUSION_SIZE_SLICE_2_P6 + ll + (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1 + idx_p6 + (blk_idx_h3 * FUSION_SIZE_SLICE_2_H3 + idx_h1) * size_h1) * size_p6) * size_p7 + (threadIdx.x + l)];
        }

        // Load Input Tensor to Shared Memory
        if (idx_p6 < rng_h2 && idx_h1 < rng_p4 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p5; ll++)
        {
            sm_b[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = dev_d2_v2_9[(blk_idx_h2 * FUSION_SIZE_SLICE_2_H2 + idx_p6 + (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5 + ll + (blk_idx_p4 * FUSION_SIZE_SLICE_2_P4 + idx_h1) * size_p5) * size_h2) * size_p7 + (threadIdx.x + l)];
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



	//
	//
	//
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

	//
	//	sd1_4,5,6,7,8,9
	//
	// tensor contraction
	internal_upperbound = 0;
	#pragma unroll 1
	for (int l = 0; l < size_internal && opt_d1_4 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of size_internal Indices: 1
		if (idx_p6 < rng_p6 && idx_h1 < rng_h1 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p5; ll++)
		{
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = dev_d1_t2_4[(blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h1) * size_p6) * size_p5) * size_h7 + (threadIdx.x + l)];
           
		}

		// Load Input Tensor to Shared Memory
		if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p4; ll++)
		{
            sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = dev_d1_v2_4[(blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h2 + (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll) * size_h2) * size_h3) + (threadIdx.y + l) * list_stride_v2[0]];
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

	// tensor contraction
	internal_upperbound = 0;
	#pragma unroll 1
	for (int l = 0; l < size_internal && opt_d1_5 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		if (idx_p6 < rng_p6 && idx_h1 < rng_h2 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p5; ll++)
		{
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = dev_d1_t2_5[(blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h1) * size_p6) * size_p5) * size_h7 + (threadIdx.x + l)]; 
		}

		// Load Input Tensor to Shared Memory
		if (idx_h3 < rng_h3 && idx_h2 < rng_h1 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p4; ll++)
		{
            sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = dev_d1_v2_5[(blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll) * size_h1) * size_h3) + (threadIdx.y + l) * list_stride_v2[1]];
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

	// tensor contraction
	internal_upperbound = 0;
	#pragma unroll 1
	for (int l = 0; l < size_internal && opt_d1_6 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1 //63, 21
		if (idx_p6 < rng_p6 && idx_h1 < rng_h3 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p5; ll++)
		{
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = dev_d1_t2_6[(blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h1) * size_p6) * size_p5) * size_h7 + (threadIdx.x + l)];
		}

		// Load Input Tensor to Shared Memory
		if (idx_h3 < rng_h2 && idx_h2 < rng_h1 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p4; ll++)
		{
            sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = dev_d1_v2_6[(blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll) * size_h1) * size_h2) + (threadIdx.y + l) * list_stride_v2[2]];
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

	// tensor contraction
	internal_upperbound = 0;
	#pragma unroll 1
	for (int l = 0; l < size_internal && opt_d1_7 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		if (idx_p6 < rng_p6 && idx_h1 < rng_h1 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p4; ll++)
		{
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = dev_d1_t2_7[(blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h1) * size_p6) * size_p4) * size_h7 + (threadIdx.x + l)];
		}

		// Load Input Tensor to Shared Memory
		if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p5; ll++)
		{
            sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = dev_d1_v2_7[(blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h2 + (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll) * size_h2) * size_h3) + (threadIdx.y + l) * list_stride_v2[3]];
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

	// tensor contraction
	internal_upperbound = 0;
	#pragma unroll 1
	for (int l = 0; l < size_internal && opt_d1_8 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		if (idx_p6 < rng_p6 && idx_h1 < rng_h2 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p4; ll++)
		{
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = dev_d1_t2_8[(blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h1) * size_p6) * size_p4) * size_h7 + (threadIdx.x + l)];
		}

		// Load Input Tensor to Shared Memory
		if (idx_h3 < rng_h3 && idx_h2 < rng_h1 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p5; ll++)
		{
            sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = dev_d1_v2_8[(blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll) * size_h1) * size_h3) + (threadIdx.y + l) * list_stride_v2[4]];
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

	// tensor contraction
	internal_upperbound = 0;
	#pragma unroll 1
	for (int l = 0; l < size_internal && opt_d1_9 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		if (idx_p6 < rng_p6 && idx_h1 < rng_h3 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p4; ll++)
		{
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = dev_d1_t2_9[(blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h1) * size_p6) * size_p4) * size_h7 + (threadIdx.x + l)];
		}

		// Load Input Tensor to Shared Memory
		if (idx_h3 < rng_h2 && idx_h2 < rng_h1 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p5; ll++)
		{
			sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = dev_d1_v2_9[(blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll) * size_h1) * size_h2) + (threadIdx.y + l) * list_stride_v2[5]];
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




    //
    //	sd2_1, 2, 3, 4, 5 and 6.
    //
    // tensor contraction
    internal_upperbound = 0;
    
    // tensor contraction
    #pragma unroll 1
    for (int l = 0; l < size_internal && opt_d2_1 == 1; l+= FUSION_SIZE_INT_UNIT)
    {
        // Part: Generalized Contraction Index (p7b)
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
        if (internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of size_internal Indices: 1
        if (idx_p6 < rng_h1 && idx_h1 < rng_h2 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p4; ll++)
        {
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = dev_d2_t2_1[(blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_p6 + (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h1) * size_h1) * size_p4) * size_p7 + (threadIdx.x + l)];
        }

        // Load Input Tensor to Shared Memory
        if (idx_p6 < rng_h3 && idx_h1 < rng_p6 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p5; ll++)
        {
            sm_b[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = dev_d2_v2_1[(blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_p6 + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_h1 + (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll) * size_p6) * size_h3) * size_p7 + (threadIdx.x + l)];
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

    // tensor contraction
    internal_upperbound = 0;
    #pragma unroll 1
    for (int l = 0; l < size_internal && opt_d2_2 == 1; l+= FUSION_SIZE_INT_UNIT)
    {
        // Part: Generalized Contraction Index (p7b)
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
        if (internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of size_internal Indices: 1
        if (idx_p6 < rng_h2 && idx_h1 < rng_h3 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p4; ll++)
        {
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = dev_d2_t2_2[(blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll + (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_p6 + (blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h1) * size_h2) * size_p4) * size_p7 + (threadIdx.x + l)];
        }

        // Load Input Tensor to Shared Memory
        if (idx_p6 < rng_h1 && idx_h1 < rng_p6 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p5; ll++)
        {
            sm_b[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = dev_d2_v2_2[(blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_p6 + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_h1 + (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll) * size_p6) * size_h1) * size_p7 + (threadIdx.x + l)];
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

    // tensor contraction
    internal_upperbound = 0;
    #pragma unroll 1
    for (int l = 0; l < size_internal && opt_d2_3 == 1; l+= FUSION_SIZE_INT_UNIT)
    {
        // Part: Generalized Contraction Index (p7b)
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
        if (internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of size_internal Indices: 1
        if (idx_p6 < rng_h1 && idx_h1 < rng_h3 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p4; ll++)
        {
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = dev_d2_t2_3[(blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_p6 + (blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h1) * size_h1) * size_p4) * size_p7 + (threadIdx.x + l)];
        }

        // Load Input Tensor to Shared Memory
        if (idx_p6 < rng_h2 && idx_h1 < rng_p6 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p5; ll++)
        {
            sm_b[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = dev_d2_v2_3[(blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_p6 + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_h1 + (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll) * size_p6) * size_h2) * size_p7 + (threadIdx.x + l)];
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

    // tensor contraction
    internal_upperbound = 0;
    #pragma unroll 1
    for (int l = 0; l < size_internal && opt_d2_4 == 1; l+= FUSION_SIZE_INT_UNIT)
    {
        // Part: Generalized Contraction Index (p7b)
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
        if (internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of size_internal Indices: 1
        if (idx_p6 < rng_h1 && idx_h1 < rng_h2 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p5; ll++)
        {
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = dev_d2_t2_4[(blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_p6 + (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h1) * size_h1) * size_p5) * size_p7 + (threadIdx.x + l)];
        }

        // Load Input Tensor to Shared Memory
        if (idx_p6 < rng_h3 && idx_h1 < rng_p6 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p4; ll++)
        {
            sm_b[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = dev_d2_v2_4[(blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_p6 + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_h1 + (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll) * size_p6) * size_h3) * size_p7 + (threadIdx.x + l)];
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

    // tensor contraction
    internal_upperbound = 0;
    #pragma unroll 1
    for (int l = 0; l < size_internal && opt_d2_5 == 1; l+= FUSION_SIZE_INT_UNIT)
    {
        // Part: Generalized Contraction Index (p7b)
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
        if (internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of size_internal Indices: 1
        if (idx_p6 < rng_h2 && idx_h1 < rng_h3 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p5; ll++)
        {
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = dev_d2_t2_5[(blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll + (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_p6 + (blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h1) * size_h2) * size_p5) * size_p7 + (threadIdx.x + l)];
        }

        // Load Input Tensor to Shared Memory
        if (idx_p6 < rng_h1 && idx_h1 < rng_p6 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p4; ll++)
        {
            sm_b[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = dev_d2_v2_5[(blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_p6 + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_h1 + (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll) * size_p6) * size_h1) * size_p7 + (threadIdx.x + l)];
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

    // tensor contraction
    internal_upperbound = 0;
    #pragma unroll 1
    for (int l = 0; l < size_internal && opt_d2_6 == 1; l+= FUSION_SIZE_INT_UNIT)
    {
        // Part: Generalized Contraction Index (p7b)
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
        if (internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of size_internal Indices: 1
        if (idx_p6 < rng_h1 && idx_h1 < rng_h3 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p5; ll++)
        {
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = dev_d2_t2_6[(blk_idx_p5 * FUSION_SIZE_SLICE_1_P6 + ll + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_p6 + (blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h1) * size_h1) * size_p5) * size_p7 + (threadIdx.x + l)];
        }

        // Load Input Tensor to Shared Memory
        if (idx_p6 < rng_h2 && idx_h1 < rng_p6 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p4; ll++)
        {
            sm_b[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = dev_d2_v2_6[(blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_p6 + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_h1 + (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll) * size_p6) * size_h2) * size_p7 + (threadIdx.x + l)];
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



    //                                        "x"         "x"
    //  >> s1_1:   t3[h3,h2,h1,p6,p5,p4] -= t2[p4,h1] * v2[h3,h2,p6,p5]
    //
    if (opt_s1_1 == 1)
    {
        if (idx_h3 < rng_p4 && idx_h2 < rng_h1 && idx_p6 == 0 && idx_h1 == 0)
        sm_a[0][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_1_P4] = dev_s1_t2_1[blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + idx_h3 + 
                                                                         (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2) * size_p4];

        if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6 && idx_h1 < rng_p5)
        sm_b[idx_h1][idx_h3 + (idx_h2 + (idx_p6) * 4) * 4] = dev_s1_v2_1[blk_idx_h3 * 4 + idx_h3 + 
                                                                        (blk_idx_h2 * 4 + idx_h2 + 
                                                                        (blk_idx_p6 * 4 + idx_p6 + 
                                                                        (blk_idx_p5 * 4 + idx_h1) * size_p6) * size_h2) * size_h3];
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
    }

    //                                        "x1,x2"     "x1,x2,x3,y1"
    //  >> s1_2:   t3[h3,h2,h1,p6,p5,p4] -= t2[p4,h2] * v2[h3,h1,p6,p5] (h3,h2,p6), (h1)
    //
    if (opt_s1_2 == 1)
    {
        if (idx_h3 < rng_p4 && idx_h2 < rng_h2 && idx_p6 == 0 && idx_h1 == 0)
        sm_a[0][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_1_P4] = dev_s1_t2_2[blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + idx_h3 + 
                                                                         (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h2) * size_p4];

        if (idx_h3 < rng_h3 && idx_h2 < rng_h1 && idx_p6 < rng_p6 && idx_h1 < rng_p5)
        sm_b[idx_h1][idx_h3 + (idx_h2 + (idx_p6) * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H3] 
        = dev_s1_v2_2[blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + 
                (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + 
                (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + 
                (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + idx_h1) * size_p6) * size_h1) * size_h3];
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
    }

    //
    //  >> s1_3:   t3[h3,h2,h1,p6,p5,p4] -= t2[p4,h1] * v2[h3,h2,p6,p5] ??
    //
    if (opt_s1_3 == 1)
    {
        if (idx_h3 < rng_p4 && idx_h2 < rng_h1 && idx_p6 == 0 && idx_h1 == 0)
        sm_a[0][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_1_P4] = dev_s1_t2_3[blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + idx_h3 + 
                                                                         (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2) * size_p4];

        if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6 && idx_h1 < rng_p5)
        sm_b[idx_h1][idx_h3 + (idx_h2 + (idx_p6) * 4) * 4] = dev_s1_v2_3[blk_idx_h3 * 4 + idx_h3 + 
                                                                                       (blk_idx_h2 * 4 + idx_h2 + 
                                                                                       (blk_idx_p6 * 4 + idx_p6 + 
                                                                                       (blk_idx_p5 * 4 + idx_h1) * size_p6) * size_h2) * size_h3];
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
    }

    //
    //  >> s1_4:   t3[h3,h2,h1,p6,p5,p4] -= t2[p5,h1] * v2[h3,h2,p6,p4] (h3,h2,p6), (h1)
    //
    if (opt_s1_4 == 1)
    {
        if (idx_h3 < rng_p5 && idx_h2 < rng_h1 && idx_p6 == 0 && idx_h1 == 0)
        sm_a[0][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_1_P5] = dev_s1_t2_4[blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + idx_h3 + 
                                                                         (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2) * size_p5];

        if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6 && idx_h1 < rng_p4)
        sm_b[idx_h1][idx_h3 + (idx_h2 + (idx_p6) * 4) * 4] = dev_s1_v2_4[blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + 
                                                                        (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h2 + 
                                                                        (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + 
                                                                        (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + idx_h1) * size_p6) * size_h2) * size_h3];
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
    }

    //
    //  >> s1_5:   t3[h3,h2,h1,p6,p5,p4] -= t2[p5,h2] * v2[h3,h1,p6,p4] (h3,h2,p6), (h1)
    //
    if (opt_s1_5 == 1)
    {
        if (idx_h3 < rng_p5 && idx_h2 < rng_h2 && idx_p6 == 0 && idx_h1 == 0)
        sm_a[0][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_1_P5] 
        = dev_s1_t2_5[blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + idx_h3 + 
                     (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h2) * size_p5];

        if (idx_h3 < rng_h3 && idx_h2 < rng_h1 && idx_p6 < rng_p6 && idx_h1 < rng_p4)
        sm_b[idx_h1][idx_h3 + (idx_h2 + (idx_p6) * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H3] 
        = dev_s1_v2_5[blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + 
                    (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + 
                    (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + 
                    (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + idx_h1) * size_p6) * size_h1) * size_h3];
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
    }


    //
    //  >> s1_6:   t3[h3,h2,h1,p6,p5,p4] -= t2[p5,h3] * v2[h2,h1,p6,p4] (h3,h2,p6), (h1)
    //
    if (opt_s1_6 == 1)
    {
        if (idx_h3 < rng_p5 && idx_h2 < rng_h3 && idx_p6 == 0 && idx_h1 == 0)
        sm_a[0][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_1_P5] = dev_s1_t2_6[blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + idx_h3 + (blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h2) * size_p5];

        if (idx_h3 < rng_h2 && idx_h2 < rng_h1 && idx_p6 < rng_p6 && idx_h1 < rng_p4)
        sm_b[idx_h1][idx_h3 + (idx_h2 + (idx_p6) * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H2] = dev_s1_v2_6[blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + idx_h1) * size_p6) * size_h1) * size_h2];
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
    }

    //
    //  >> Register-Transpose << 
    //  : Shared Memory Sizes is too small to be an intermediate space to transpose register files.
    //

    //
    //  >> s1_7:   t3[h3,h2,h1,p6,p5,p4] -= t2[p6,h1] * v2[h3,h2,p5,p4] (h3,h2,p6), (h1)
    //
    if (opt_s1_7 == 1)
    {
        if (idx_h3 < rng_p6 && idx_h2 < rng_h1 && idx_p6 == 0 && idx_h1 == 0)
        sm_a[0][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_1_P6] 
        = dev_s1_t2_7[blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_h3 + 
                (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2) * size_p6];

        if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p5 && idx_h1 < rng_p4)
        sm_b[idx_h1][idx_h3 + (idx_h2 + (idx_p6) * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3] 
        = dev_s1_v2_7[blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + 
                (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h2 + 
                (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + idx_p6 + 
                (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + idx_h1) * size_p5) * size_h2) * size_h3];
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
    }

    //
    //  >> s1_8:   t3[h3,h2,h1,p6,p5,p4] -= t2[p6,h2] * v2[h3,h1,p5,p4] (h3,h2,p6), (h1)
    //
    if (opt_s1_8 == 1)
    {
        if (idx_h3 < rng_p6 && idx_h2 < rng_h2 && idx_p6 == 0 && idx_h1 == 0)
        sm_a[0][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_1_P6] 
        = dev_s1_t2_8[blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_h3 + 
                (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h2) * size_p6];
                
        if (idx_h3 < rng_h3 && idx_h2 < rng_h1 && idx_p6 < rng_p5 && idx_h1 < rng_p4)
        sm_b[idx_h1][idx_h3 + (idx_h2 + (idx_p6) * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H3] 
        = dev_s1_v2_8[blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + 
                (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + 
                (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + idx_p6 + 
                (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + idx_h1) * size_p5) * size_h1) * size_h3];
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
    }
    
    //
    //  >> s1_9:   t3[h3,h2,h1,p6,p5,p4] -= t2[p6,h3] * v2[h2,h1,p5,p4] (h3,h2,p6), (h1)
    //
    if (opt_s1_9 == 1)
    {
        if (idx_h3 < rng_p6 && idx_h2 < rng_h3 && idx_p6 == 0 && idx_h1 == 0)
        sm_a[0][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_1_P6] 
        = dev_s1_t2_9[blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_h3 + 
                (blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h2) * size_p6];

        if (idx_h3 < rng_h2 && idx_h2 < rng_h1 && idx_p6 < rng_p5 && idx_h1 < rng_p4)
        sm_b[idx_h1][idx_h3 + (idx_h2 + (idx_p6) * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H2] 
        = dev_s1_v2_9[blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h3 + 
                (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + 
                (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + idx_p6 + 
                (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + idx_h1) * size_p5) * size_h1) * size_h2];
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
    }


    double energy_1 = 0.0;
    double energy_2 = 0.0;

    double eval_h3 = dev_eval_h3[blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3];
    double eval_h2 = dev_eval_h2[blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h2];
    double eval_p6 = dev_eval_p6[blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6];
    double eval_h1 = dev_eval_h1[blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h1];

    // 
    if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6 && idx_h1 < rng_h1)
    {
        for (int i = 0; i < FUSION_SIZE_SLICE_1_P5; i++)
        {
            for (int j = 0; j < FUSION_SIZE_SLICE_1_P4; j++)
            {
                if (i < rng_p5 && j < rng_p4)
                {
                    double inner_factor = eval_h3 + eval_h2 + eval_h1 - eval_p6 - dev_eval_p5[i + (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5)] - dev_eval_p4[j + (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4)];

                    // 
                    // energy_1    += (factor * dev_t3_doubles[] *  dev_t3_doubles[]) / inner_factor;
                    // energy_2    += (factor * dev_t3_doubles[] * (dev_t3_doubles[] + dev_t3_singles[])) / inner_factor;
                    energy_1 += (factor * reg_tile[j][i] *  reg_tile[j][i]) / inner_factor;
                    energy_2 += (factor * reg_tile[j][i] * (reg_tile[j][i] + reg_singles[j][i])) / inner_factor;

                    // if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0 && i == 1 && j == 0)
                    // {
                    //     printf ("[debug] h3,h2,h1,p6,p5,p4 = %d,%d,%d,%d,%d,%d: inner_factor: %.15f\n", idx_h3, idx_h2, idx_h1, idx_p6, i, j, inner_factor);
                    //     printf ("[debug] energy_1: %.15f\n", factor * reg_tile[j][i] * reg_tile[j][i] / inner_factor);
                    //     printf ("[debug] energy_2: %.15f\n", (factor * reg_tile[j][i] * (reg_tile[j][i] + reg_singles[j][i])) / inner_factor);
                    // }
                }
            }
        }
    }
    __syncthreads();

    // 
    for (int offset = 16; offset > 0; offset /= 2)
    {
        energy_1 += __shfl_down_sync(FULL_MASK, energy_1, offset);
        energy_2 += __shfl_down_sync(FULL_MASK, energy_2, offset);
    }

    // __shared__ double sm_a[16][64 + 1];
    // __shared__ double sm_b[16][64];
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
extern "C" void fused_ccsd_t(Integer* sizes, 
        // 
        double* host_d1_t2_1, double* host_d1_v2_1, double* host_d1_t2_2, double* host_d1_v2_2,	double* host_d1_t2_3, double* host_d1_v2_3, 
        double* host_d1_t2_4, double* host_d1_v2_4, double* host_d1_t2_5, double* host_d1_v2_5,	double* host_d1_t2_6, double* host_d1_v2_6, 
        double* host_d1_t2_7, double* host_d1_v2_7, double* host_d1_t2_8, double* host_d1_v2_8, double* host_d1_t2_9, double* host_d1_v2_9,
        int d1_kernel_1, int d1_kernel_2, int d1_kernel_3, 
        int d1_kernel_4, int d1_kernel_5, int d1_kernel_6, 
        int d1_kernel_7, int d1_kernel_8, int d1_kernel_9,
        // 
        double* host_d2_t2_1, double* host_d2_v2_1, double* host_d2_t2_2, double* host_d2_v2_2,	double* host_d2_t2_3, double* host_d2_v2_3, 
        double* host_d2_t2_4, double* host_d2_v2_4, double* host_d2_t2_5, double* host_d2_v2_5,	double* host_d2_t2_6, double* host_d2_v2_6, 
        double* host_d2_t2_7, double* host_d2_v2_7, double* host_d2_t2_8, double* host_d2_v2_8, double* host_d2_t2_9, double* host_d2_v2_9,
        int d2_kernel_1, int d2_kernel_2, int d2_kernel_3, 
        int d2_kernel_4, int d2_kernel_5, int d2_kernel_6, 
        int d2_kernel_7, int d2_kernel_8, int d2_kernel_9,
        // 
        double* host_s1_t2_1, double* host_s1_v2_1, double* host_s1_t2_2, double* host_s1_v2_2,	double* host_s1_t2_3, double* host_s1_v2_3, 
        double* host_s1_t2_4, double* host_s1_v2_4, double* host_s1_t2_5, double* host_s1_v2_5,	double* host_s1_t2_6, double* host_s1_v2_6, 
        double* host_s1_t2_7, double* host_s1_v2_7, double* host_s1_t2_8, double* host_s1_v2_8, double* host_s1_t2_9, double* host_s1_v2_9,
        int s1_kernel_1, int s1_kernel_2, int s1_kernel_3, 
        int s1_kernel_4, int s1_kernel_5, int s1_kernel_6, 
        int s1_kernel_7, int s1_kernel_8, int s1_kernel_9,
        // 
        double factor, 
        double* host_eval_h1, double* host_eval_h2, double* host_eval_h3, double* host_eval_p4, double* host_eval_p5, double* host_eval_p6,
        double* energy)
{
#if 0
    printf ("[%s] d1: %d, %d, %d, %d, %d, %d, %d, %d, %d\n", __func__, d1_kernel_1, d1_kernel_2, d1_kernel_3, d1_kernel_4, d1_kernel_5, d1_kernel_6, d1_kernel_7, d1_kernel_8, d1_kernel_9);
    printf ("[%s] d2: %d, %d, %d, %d, %d, %d, %d, %d, %d\n", __func__, d2_kernel_1, d2_kernel_2, d2_kernel_3, d2_kernel_4, d2_kernel_5, d2_kernel_6, d2_kernel_7, d2_kernel_8, d2_kernel_9);
    printf ("[%s] s1: %d, %d, %d, %d, %d, %d, %d, %d, %d\n", __func__, s1_kernel_1, s1_kernel_2, s1_kernel_3, s1_kernel_4, s1_kernel_5, s1_kernel_6, s1_kernel_7, s1_kernel_8, s1_kernel_9);
#endif
	//TODO: Fix for all kernels, use sizes[0-63] for kernels 1-9
	int size_h1 = sizes[0];
	int size_h2 = sizes[1];
	int size_h3 = sizes[2];
	int size_h7 = sizes[3];
	int size_p4 = sizes[4];
	int size_p5 = sizes[5];
    int size_p6 = sizes[6];
    int size_p7 = size_h7;

	// # of Blocks for Each Kernel
	int	 num_blocks_kernel_1,		num_blocks_kernel_2;
    
	// Device Memory for Inputs
	double *dev_d1_t2_1, *dev_d1_t2_2, *dev_d1_t2_3, *dev_d1_t2_4, *dev_d1_t2_5, *dev_d1_t2_6, *dev_d1_t2_7, *dev_d1_t2_8, *dev_d1_t2_9;
    double *dev_d1_v2_1, *dev_d1_v2_2, *dev_d1_v2_3, *dev_d1_v2_4, *dev_d1_v2_5, *dev_d1_v2_6, *dev_d1_v2_7, *dev_d1_v2_8, *dev_d1_v2_9;
    
    double *dev_d2_t2_1, *dev_d2_t2_2, *dev_d2_t2_3, *dev_d2_t2_4, *dev_d2_t2_5, *dev_d2_t2_6, *dev_d2_t2_7, *dev_d2_t2_8, *dev_d2_t2_9;
    double *dev_d2_v2_1, *dev_d2_v2_2, *dev_d2_v2_3, *dev_d2_v2_4, *dev_d2_v2_5, *dev_d2_v2_6, *dev_d2_v2_7, *dev_d2_v2_8, *dev_d2_v2_9;
    
    double *dev_s1_t2_1, *dev_s1_t2_2, *dev_s1_t2_3, *dev_s1_t2_4, *dev_s1_t2_5, *dev_s1_t2_6, *dev_s1_t2_7, *dev_s1_t2_8, *dev_s1_t2_9;
    double *dev_s1_v2_1, *dev_s1_v2_2, *dev_s1_v2_3, *dev_s1_v2_4, *dev_s1_v2_5, *dev_s1_v2_6, *dev_s1_v2_7, *dev_s1_v2_8, *dev_s1_v2_9;

    double *dev_eval_h1, *dev_eval_h2, *dev_eval_h3, *dev_eval_p4, *dev_eval_p5, *dev_eval_p6;
    double *dev_energies;
    double *host_energies;
        
    //  s1
    size_t size_s1_t2_1 = sizeof(double) * size_p4 * size_h1;
    size_t size_s1_v2_1 = sizeof(double) * size_h3 * size_h2 * size_p6 * size_p5;
    size_t size_s1_t2_2 = sizeof(double) * size_p4 * size_h2;
    size_t size_s1_v2_2 = sizeof(double) * size_h3 * size_h1 * size_p6 * size_p5;
    size_t size_s1_t2_3 = sizeof(double) * size_p4 * size_h1;
    size_t size_s1_v2_3 = sizeof(double) * size_h3 * size_h2 * size_p6 * size_p5;
    size_t size_s1_t2_4 = sizeof(double) * size_p5 * size_h1;
    size_t size_s1_v2_4 = sizeof(double) * size_h3 * size_h2 * size_p6 * size_p4;
    size_t size_s1_t2_5 = sizeof(double) * size_p5 * size_h2;
    size_t size_s1_v2_5 = sizeof(double) * size_h3 * size_h1 * size_p6 * size_p4;
    size_t size_s1_t2_6 = sizeof(double) * size_p5 * size_h3;
    size_t size_s1_v2_6 = sizeof(double) * size_h2 * size_h1 * size_p6 * size_p4;
    size_t size_s1_t2_7 = sizeof(double) * size_p6 * size_h1;
    size_t size_s1_v2_7 = sizeof(double) * size_h3 * size_h2 * size_p5 * size_p4;
    size_t size_s1_t2_8 = sizeof(double) * size_p6 * size_h2;
    size_t size_s1_v2_8 = sizeof(double) * size_h3 * size_h1 * size_p5 * size_p4;
    size_t size_s1_t2_9 = sizeof(double) * size_p6 * size_h3;
    size_t size_s1_v2_9 = sizeof(double) * size_h2 * size_h1 * size_p5 * size_p4;

    //  d1
    size_t size_d1_t2_1 = sizeof(double) * size_h1 * size_p5 * size_p4 * size_h7;
	size_t size_d1_v2_1 = sizeof(double) * size_h7 * size_p6 * size_h2 * size_h3;
	size_t size_d1_t2_2 = sizeof(double) * size_h2 * size_p5 * size_p4 * size_h7;
	size_t size_d1_v2_2 = sizeof(double) * size_h7 * size_p6 * size_h1 * size_h3;
	size_t size_d1_t2_3 = sizeof(double) * size_h3 * size_p5 * size_p4 * size_h7;
    size_t size_d1_v2_3 = sizeof(double) * size_h7 * size_p6 * size_h1 * size_h2;
    size_t size_d1_t2_4 = sizeof(double) * size_h1 * size_p6 * size_p5 * size_h7;
    size_t size_d1_v2_4 = sizeof(double) * size_h7 * size_p4 * size_h2 * size_h3;
	size_t size_d1_t2_5 = sizeof(double) * size_h2 * size_p6 * size_p5 * size_h7;
	size_t size_d1_v2_5 = sizeof(double) * size_h7 * size_p4 * size_h1 * size_h3;
	size_t size_d1_t2_6 = sizeof(double) * size_h3 * size_p6 * size_p5 * size_h7;
	size_t size_d1_v2_6 = sizeof(double) * size_h7 * size_p4 * size_h1 * size_h2;
	size_t size_d1_t2_7 = sizeof(double) * size_h1 * size_p6 * size_p4 * size_h7;
	size_t size_d1_v2_7 = sizeof(double) * size_h7 * size_p5 * size_h2 * size_h3;
	size_t size_d1_t2_8 = sizeof(double) * size_h2 * size_p6 * size_p4 * size_h7;
	size_t size_d1_v2_8 = sizeof(double) * size_h7 * size_p5 * size_h1 * size_h3;
	size_t size_d1_t2_9 = sizeof(double) * size_h3 * size_p6 * size_p4 * size_h7;
    size_t size_d1_v2_9 = sizeof(double) * size_h7 * size_p5 * size_h1 * size_h2;
    
    // sd1
    if (d1_kernel_1 == 1)
    {
        dev_d1_t2_1 = (double*)getGpuMem(size_d1_t2_1);
        dev_d1_v2_1 = (double*)getGpuMem(size_d1_v2_1);
    }

    if (d1_kernel_2 == 1)
    {
        dev_d1_t2_2 = (double*)getGpuMem(size_d1_t2_2);
        dev_d1_v2_2 = (double*)getGpuMem(size_d1_v2_2);
    }
    
    if (d1_kernel_3 == 1)
    {
        dev_d1_t2_3 = (double*)getGpuMem(size_d1_t2_3);
        dev_d1_v2_3 = (double*)getGpuMem(size_d1_v2_3);
    }

    if (d1_kernel_4 == 1)
    {
        dev_d1_t2_4 = (double*)getGpuMem(size_d1_t2_4);
        dev_d1_v2_4 = (double*)getGpuMem(size_d1_v2_4);
    }
        
    if (d1_kernel_5 == 1)
    {
        dev_d1_t2_5 = (double*)getGpuMem(size_d1_t2_5);
        dev_d1_v2_5 = (double*)getGpuMem(size_d1_v2_5);
    }
        
    if (d1_kernel_6 == 1)
    {
        dev_d1_t2_6 = (double*)getGpuMem(size_d1_t2_6);
        dev_d1_v2_6 = (double*)getGpuMem(size_d1_v2_6);
    }
        
    if (d1_kernel_7 == 1)
    {
        dev_d1_t2_7 = (double*)getGpuMem(size_d1_t2_7);
        dev_d1_v2_7 = (double*)getGpuMem(size_d1_v2_7);
    }
        
    if (d1_kernel_8 == 1)
    {
        dev_d1_t2_8 = (double*)getGpuMem(size_d1_t2_8);
        dev_d1_v2_8 = (double*)getGpuMem(size_d1_v2_8);
    }
        
    if (d1_kernel_9 == 1)
    {
        dev_d1_t2_9 = (double*)getGpuMem(size_d1_t2_9);
        dev_d1_v2_9 = (double*)getGpuMem(size_d1_v2_9);
    }
    
    // sd2
    if (d2_kernel_1 == 1)
    {
        dev_d2_t2_1 = (double*)getGpuMem(sizeof(double) * size_h2 * size_h1 * size_p4 * size_p7);
        dev_d2_v2_1 = (double*)getGpuMem(sizeof(double) * size_p5 * size_p6 * size_h3 * size_p7);
    }

    if (d2_kernel_2 == 1)
    {
        dev_d2_t2_2 = (double*)getGpuMem(sizeof(double) * size_h3 * size_h2 * size_p4 * size_p7);
        dev_d2_v2_2 = (double*)getGpuMem(sizeof(double) * size_p5 * size_p6 * size_h1 * size_p7);
    }

    if (d2_kernel_3 == 1)
    {
        dev_d2_t2_3 = (double*)getGpuMem(sizeof(double) * size_h3 * size_h1 * size_p4 * size_p7);
        dev_d2_v2_3 = (double*)getGpuMem(sizeof(double) * size_p5 * size_p6 * size_h2 * size_p7);
    }

    if (d2_kernel_4 == 1)
    {
        dev_d2_t2_4 = (double*)getGpuMem(sizeof(double) * size_h2 * size_h1 * size_p5 * size_p7);
        dev_d2_v2_4 = (double*)getGpuMem(sizeof(double) * size_p4 * size_p6 * size_h3 * size_p7);
    }

    if (d2_kernel_5 == 1)
    {
        dev_d2_t2_5 = (double*)getGpuMem(sizeof(double) * size_h3 * size_h2 * size_p5 * size_p7);
        dev_d2_v2_5 = (double*)getGpuMem(sizeof(double) * size_p4 * size_p6 * size_h1 * size_p7);
    }

    if (d2_kernel_6 == 1)
    {
        dev_d2_t2_6 = (double*)getGpuMem(sizeof(double) * size_h3 * size_h1 * size_p5 * size_p7);
        dev_d2_v2_6 = (double*)getGpuMem(sizeof(double) * size_p4 * size_p6 * size_h2 * size_p7);
    }

    if (d2_kernel_7 == 1)
    {
        dev_d2_t2_7 = (double*)getGpuMem(sizeof(double) * size_h2 * size_h1 * size_p6 * size_p7);
        dev_d2_v2_7 = (double*)getGpuMem(sizeof(double) * size_p4 * size_p5 * size_h3 * size_p7);
    }

    if (d2_kernel_8 == 1)
    {
        dev_d2_t2_8 = (double*)getGpuMem(sizeof(double) * size_h3 * size_h2 * size_p6 * size_p7);
        dev_d2_v2_8 = (double*)getGpuMem(sizeof(double) * size_p4 * size_p5 * size_h1 * size_p7);
    }

    if (d2_kernel_9 == 1)
    {
        dev_d2_t2_9 = (double*)getGpuMem(sizeof(double) * size_h3 * size_h1 * size_p6 * size_p7);
        dev_d2_v2_9 = (double*)getGpuMem(sizeof(double) * size_p4 * size_p5 * size_h2 * size_p7);
    }

    // s1
    if (s1_kernel_1 == 1)
    {
        dev_s1_t2_1 = (double*)getGpuMem(size_s1_t2_1);
        dev_s1_v2_1 = (double*)getGpuMem(size_s1_v2_1);
    }
        
    if (s1_kernel_2 == 1)
    {
        dev_s1_t2_2 = (double*)getGpuMem(size_s1_t2_2);
        dev_s1_v2_2 = (double*)getGpuMem(size_s1_v2_2);
    }
        
    if (s1_kernel_3 == 1)
    {
        dev_s1_t2_3 = (double*)getGpuMem(size_s1_t2_3);
        dev_s1_v2_3 = (double*)getGpuMem(size_s1_v2_3);
    }

    if (s1_kernel_4 == 1)
    {
        dev_s1_t2_4 = (double*)getGpuMem(size_s1_t2_4);
        dev_s1_v2_4 = (double*)getGpuMem(size_s1_v2_4);
    }
        
    if (s1_kernel_5 == 1)
    {
        dev_s1_t2_5 = (double*)getGpuMem(size_s1_t2_5);
        dev_s1_v2_5 = (double*)getGpuMem(size_s1_v2_5);
    }
    
    if (s1_kernel_6 == 1)
    {
        dev_s1_t2_6 = (double*)getGpuMem(size_s1_t2_6);
        dev_s1_v2_6 = (double*)getGpuMem(size_s1_v2_6);
    }
        
    if (s1_kernel_7 == 1)
    {
        dev_s1_t2_7 = (double*)getGpuMem(size_s1_t2_7);
        dev_s1_v2_7 = (double*)getGpuMem(size_s1_v2_7);
    }
        
    if (s1_kernel_8 == 1)
    {
        dev_s1_t2_8 = (double*)getGpuMem(size_s1_t2_8);
        dev_s1_v2_8 = (double*)getGpuMem(size_s1_v2_8);
    }
        
    if (s1_kernel_9 == 1)
    {
        dev_s1_t2_9 = (double*)getGpuMem(size_s1_t2_9);
        dev_s1_v2_9 = (double*)getGpuMem(size_s1_v2_9);
    }

    //
    dev_eval_h1 = (double*)getGpuMem(size_h1 * sizeof(double));
    dev_eval_h2 = (double*)getGpuMem(size_h2 * sizeof(double));
    dev_eval_h3 = (double*)getGpuMem(size_h3 * sizeof(double));
    dev_eval_p4 = (double*)getGpuMem(size_p4 * sizeof(double));
    dev_eval_p5 = (double*)getGpuMem(size_p5 * sizeof(double));
    dev_eval_p6 = (double*)getGpuMem(size_p6 * sizeof(double));
    // printf ("[%s] allocating gpu memories....\n", __func__);

    //  sd1
    if (d1_kernel_4 == 1)
    {
        cudaMemcpy(dev_d1_t2_4, host_d1_t2_4, sizeof(double) * size_h1 * size_p6 * size_p5 * size_h7, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_d1_v2_4, host_d1_v2_4, sizeof(double) * size_h7 * size_p4 * size_h2 * size_h3, cudaMemcpyHostToDevice);
    }

    if (d1_kernel_5 == 1)
    {
        cudaMemcpy(dev_d1_t2_5, host_d1_t2_5, sizeof(double) * size_h2 * size_p6 * size_p5 * size_h7, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_d1_v2_5, host_d1_v2_5, sizeof(double) * size_h7 * size_p4 * size_h1 * size_h3, cudaMemcpyHostToDevice);
    }

    if (d1_kernel_6 == 1)
    {
        cudaMemcpy(dev_d1_t2_6, host_d1_t2_6, sizeof(double) * size_h3 * size_p6 * size_p5 * size_h7, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_d1_v2_6, host_d1_v2_6, sizeof(double) * size_h7 * size_p4 * size_h1 * size_h2, cudaMemcpyHostToDevice);
    }

    if (d1_kernel_7 == 1)
    {
        cudaMemcpy(dev_d1_t2_7, host_d1_t2_7, sizeof(double) * size_h1 * size_p6 * size_p4 * size_h7, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_d1_v2_7, host_d1_v2_7, sizeof(double) * size_h7 * size_p5 * size_h2 * size_h3, cudaMemcpyHostToDevice);
    }

    if (d1_kernel_8 == 1)
    {
        cudaMemcpy(dev_d1_t2_8, host_d1_t2_8, sizeof(double) * size_h2 * size_p6 * size_p4 * size_h7, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_d1_v2_8, host_d1_v2_8, sizeof(double) * size_h7 * size_p5 * size_h1 * size_h3, cudaMemcpyHostToDevice);
    }

    if (d1_kernel_9 == 1)
    {
        cudaMemcpy(dev_d1_t2_9, host_d1_t2_9, sizeof(double) * size_h3 * size_p6 * size_p4 * size_h7, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_d1_v2_9, host_d1_v2_9, sizeof(double) * size_h7 * size_p5 * size_h1 * size_h2, cudaMemcpyHostToDevice);
    }

    if (d1_kernel_1 == 1)
    {
        cudaMemcpy(dev_d1_t2_1, host_d1_t2_1, sizeof(double) * size_h1 * size_p5 * size_p4 * size_h7, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_d1_v2_1, host_d1_v2_1, sizeof(double) * size_h7 * size_p6 * size_h2 * size_h3, cudaMemcpyHostToDevice);
    }

    if (d1_kernel_2 == 1)
    {
        cudaMemcpy(dev_d1_t2_2, host_d1_t2_2, sizeof(double) * size_h2 * size_p5 * size_p4 * size_h7, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_d1_v2_2, host_d1_v2_2, sizeof(double) * size_h7 * size_p6 * size_h1 * size_h3, cudaMemcpyHostToDevice);
    }
    
    if (d1_kernel_3 == 1)
    {
        cudaMemcpy(dev_d1_t2_3, host_d1_t2_3, sizeof(double) * size_h3 * size_p5 * size_p4 * size_h7, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_d1_v2_3, host_d1_v2_3, sizeof(double) * size_h7 * size_p6 * size_h1 * size_h2, cudaMemcpyHostToDevice);
    }


    //  sd2
    if (d2_kernel_1 == 1)
    {
        cudaMemcpy(dev_d2_t2_1, host_d2_t2_1, sizeof(double) * size_h2 * size_h1 * size_p4 * size_p7, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_d2_v2_1, host_d2_v2_1, sizeof(double) * size_p5 * size_p6 * size_h3 * size_p7, cudaMemcpyHostToDevice);
    }

    if (d2_kernel_2 == 1)
    {
        cudaMemcpy(dev_d2_t2_2, host_d2_t2_2, sizeof(double) * size_h3 * size_h2 * size_p4 * size_p7, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_d2_v2_2, host_d2_v2_2, sizeof(double) * size_p5 * size_p6 * size_h1 * size_p7, cudaMemcpyHostToDevice);
    }
    
    if (d2_kernel_3 == 1)
    {
        cudaMemcpy(dev_d2_t2_3, host_d2_t2_3, sizeof(double) * size_h3 * size_h1 * size_p4 * size_p7, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_d2_v2_3, host_d2_v2_3, sizeof(double) * size_p5 * size_p6 * size_h2 * size_p7, cudaMemcpyHostToDevice);
    }

    if (d2_kernel_4 == 1)
    {    
        cudaMemcpy(dev_d2_t2_4, host_d2_t2_4, sizeof(double) * size_h2 * size_h1 * size_p5 * size_p7, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_d2_v2_4, host_d2_v2_4, sizeof(double) * size_p4 * size_p6 * size_h3 * size_p7, cudaMemcpyHostToDevice);
    }

    if (d2_kernel_5 == 1)
    {    
        cudaMemcpy(dev_d2_t2_5, host_d2_t2_5, sizeof(double) * size_h3 * size_h2 * size_p5 * size_p7, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_d2_v2_5, host_d2_v2_5, sizeof(double) * size_p4 * size_p6 * size_h1 * size_p7, cudaMemcpyHostToDevice);
    }

    if (d2_kernel_6 == 1)
    {   
        cudaMemcpy(dev_d2_t2_6, host_d2_t2_6, sizeof(double) * size_h3 * size_h1 * size_p5 * size_p7, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_d2_v2_6, host_d2_v2_6, sizeof(double) * size_p4 * size_p6 * size_h2 * size_p7, cudaMemcpyHostToDevice);
    }

    if (d2_kernel_7 == 1)
    {   
        cudaMemcpy(dev_d2_t2_7, host_d2_t2_7, sizeof(double) * size_h2 * size_h1 * size_p6 * size_p7, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_d2_v2_7, host_d2_v2_7, sizeof(double) * size_p4 * size_p5 * size_h3 * size_p7, cudaMemcpyHostToDevice);
    }    

    if (d2_kernel_8 == 1)
    {
        cudaMemcpy(dev_d2_t2_8, host_d2_t2_8, sizeof(double) * size_h3 * size_h2 * size_p6 * size_p7, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_d2_v2_8, host_d2_v2_8, sizeof(double) * size_p4 * size_p5 * size_h1 * size_p7, cudaMemcpyHostToDevice);
    }

    if (d2_kernel_9 == 1)
    {
        cudaMemcpy(dev_d2_t2_9, host_d2_t2_9, sizeof(double) * size_h3 * size_h1 * size_p6 * size_p7, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_d2_v2_9, host_d2_v2_9, sizeof(double) * size_p4 * size_p5 * size_h2 * size_p7, cudaMemcpyHostToDevice);
    }

    // s1
    if (s1_kernel_1 == 1)
    {
        cudaMemcpy(dev_s1_t2_1, host_s1_t2_1, size_s1_t2_1, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_s1_v2_1, host_s1_v2_1, size_s1_v2_1, cudaMemcpyHostToDevice);
    }
        
    if (s1_kernel_2 == 1)
    {
        cudaMemcpy(dev_s1_t2_2, host_s1_t2_2, size_s1_t2_2, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_s1_v2_2, host_s1_v2_2, size_s1_v2_2, cudaMemcpyHostToDevice);
    }

    if (s1_kernel_3 == 1)
    {    
        cudaMemcpy(dev_s1_t2_3, host_s1_t2_3, size_s1_t2_3, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_s1_v2_3, host_s1_v2_3, size_s1_v2_3, cudaMemcpyHostToDevice);
    }

    if (s1_kernel_4 == 1)
    {
        cudaMemcpy(dev_s1_t2_4, host_s1_t2_4, size_s1_t2_4, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_s1_v2_4, host_s1_v2_4, size_s1_v2_4, cudaMemcpyHostToDevice);
    }

    if (s1_kernel_5 == 1)
    {
        cudaMemcpy(dev_s1_t2_5, host_s1_t2_5, size_s1_t2_5, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_s1_v2_5, host_s1_v2_5, size_s1_v2_5, cudaMemcpyHostToDevice);
    }

    if (s1_kernel_6 == 1)
    {
        cudaMemcpy(dev_s1_t2_6, host_s1_t2_6, size_s1_t2_6, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_s1_v2_6, host_s1_v2_6, size_s1_v2_6, cudaMemcpyHostToDevice);
    }

    if (s1_kernel_7 == 1)
    {
        cudaMemcpy(dev_s1_t2_7, host_s1_t2_7, size_s1_t2_7, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_s1_v2_7, host_s1_v2_7, size_s1_v2_7, cudaMemcpyHostToDevice);
    }

    if (s1_kernel_8 == 1)
    {
        cudaMemcpy(dev_s1_t2_8, host_s1_t2_8, size_s1_t2_8, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_s1_v2_8, host_s1_v2_8, size_s1_v2_8, cudaMemcpyHostToDevice);
    }

    if (s1_kernel_9 == 1)
    {    
        cudaMemcpy(dev_s1_t2_9, host_s1_t2_9, size_s1_t2_9, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_s1_v2_9, host_s1_v2_9, size_s1_v2_9, cudaMemcpyHostToDevice);
    }

    //  energies
    cudaMemcpy(dev_eval_h1, host_eval_h1, size_h1 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_eval_h2, host_eval_h2, size_h2 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_eval_h3, host_eval_h3, size_h3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_eval_p4, host_eval_p4, size_p4 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_eval_p5, host_eval_p5, size_p5 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_eval_p6, host_eval_p6, size_p6 * sizeof(double), cudaMemcpyHostToDevice);
    // printf ("[%s] copying gpu memories....\n", __func__);
    
    //
    num_blocks_kernel_1 = CEIL(size_h3, FUSION_SIZE_SLICE_1_H3) * CEIL(size_h2, FUSION_SIZE_SLICE_1_H2) * CEIL(size_h1, FUSION_SIZE_SLICE_1_H1) * CEIL(size_p6, FUSION_SIZE_SLICE_1_P6) * CEIL(size_p5, FUSION_SIZE_SLICE_1_P5) * CEIL(size_p4, FUSION_SIZE_SLICE_1_P4);
    // num_blocks_kernel_2 = CEIL(size_h3, FUSION_SIZE_SLICE_2_H3) * CEIL(size_h2, FUSION_SIZE_SLICE_2_H2) * CEIL(size_h1, FUSION_SIZE_SLICE_2_H1) * CEIL(size_p6, FUSION_SIZE_SLICE_2_P6) * CEIL(size_p5, FUSION_SIZE_SLICE_2_P5) * CEIL(size_p4, FUSION_SIZE_SLICE_2_P4);

	// Depends on # of Fused Kernel
	dim3 gridsize_1(num_blocks_kernel_1);
	dim3 blocksize_1(FUSION_SIZE_TB_1_X, FUSION_SIZE_TB_1_Y);

	// dim3 gridsize_2(num_blocks_kernel_2);
	// dim3 blocksize_2(FUSION_SIZE_TB_2_X, FUSION_SIZE_TB_2_Y);

	int	str_sd2_t3_h3 = 1;
	int str_sd2_t3_h2 = str_sd2_t3_h3 * size_h3;
	int str_sd2_t3_h1 = str_sd2_t3_h2 * size_h2;
	int str_sd2_t3_p6 = str_sd2_t3_h1 * size_h1;
	int str_sd2_t3_p5 = str_sd2_t3_p6 * size_p6;
	int str_sd2_t3_p4 = str_sd2_t3_p5 * size_p5;

	int str_reg_x_1 = str_sd2_t3_p5;
	int str_reg_y_1 = str_sd2_t3_p4;
	// int str_reg_x_2 = str_sd2_t3_p5;
	// int str_reg_y_2 = str_sd2_t3_p6;

    int* list_stride_sd1_v2_1 = (int*)malloc(sizeof(int) * 9);
    list_stride_sd1_v2_1[0] = size_p4 * size_h2 * size_h3;
	list_stride_sd1_v2_1[1] = size_p4 * size_h1 * size_h3;
	list_stride_sd1_v2_1[2] = size_p4 * size_h1 * size_h2; 
	list_stride_sd1_v2_1[3] = size_p5 * size_h2 * size_h3;
	list_stride_sd1_v2_1[4] = size_p5 * size_h1 * size_h3;
    list_stride_sd1_v2_1[5] = size_p5 * size_h1 * size_h2;

    list_stride_sd1_v2_1[6] = size_p6 * size_h2 * size_h3;
    list_stride_sd1_v2_1[7] = size_p6 * size_h1 * size_h3;
    list_stride_sd1_v2_1[8] = size_p6 * size_h1 * size_h2;

    cudaMemcpyToSymbol(list_stride_v2, list_stride_sd1_v2_1, sizeof(int) * 9);

    // 
    host_energies   = (double*)getHostMem(num_blocks_kernel_1 * NUM_ENERGIES * sizeof(double));
    dev_energies    = (double*)getGpuMem (num_blocks_kernel_1 * NUM_ENERGIES * sizeof(double));

#ifdef DEBUG_TIME_FUSED_CCSD_T
    cudaEvent_t start_ccsd_t, stop_ccsd_t;
    cudaEventCreate(&start_ccsd_t);
    cudaEventCreate(&stop_ccsd_t);
    cudaEventRecord(start_ccsd_t);
#endif

    //
    jk_ccsd_t_doubles_fully_fused_kernel<<<gridsize_1, blocksize_1>>>(
        // 
        dev_d1_t2_1, dev_d1_v2_1, dev_d1_t2_2, dev_d1_v2_2, dev_d1_t2_3, dev_d1_v2_3, 
        dev_d1_t2_4, dev_d1_v2_4, dev_d1_t2_5, dev_d1_v2_5, dev_d1_t2_6, dev_d1_v2_6, 
        dev_d1_t2_7, dev_d1_v2_7, dev_d1_t2_8, dev_d1_v2_8, dev_d1_t2_9, dev_d1_v2_9, 
        d1_kernel_1, d1_kernel_2, d1_kernel_3, d1_kernel_4, d1_kernel_5, d1_kernel_6, d1_kernel_7, d1_kernel_8, d1_kernel_9, 
        // 
        dev_d2_t2_1, dev_d2_v2_1, dev_d2_t2_2, dev_d2_v2_2, dev_d2_t2_3, dev_d2_v2_3, 
        dev_d2_t2_4, dev_d2_v2_4, dev_d2_t2_5, dev_d2_v2_5, dev_d2_t2_6, dev_d2_v2_6, 
        dev_d2_t2_7, dev_d2_v2_7, dev_d2_t2_8, dev_d2_v2_8, dev_d2_t2_9, dev_d2_v2_9, 
        d2_kernel_1, d2_kernel_2, d2_kernel_3, d2_kernel_4, d2_kernel_5, d2_kernel_6, d2_kernel_7, d2_kernel_8, d2_kernel_9, 
        // 
        dev_s1_t2_1, dev_s1_v2_1, dev_s1_t2_2, dev_s1_v2_2, dev_s1_t2_3, dev_s1_v2_3,
        dev_s1_t2_4, dev_s1_v2_4, dev_s1_t2_5, dev_s1_v2_5, dev_s1_t2_6, dev_s1_v2_6,
        dev_s1_t2_7, dev_s1_v2_7, dev_s1_t2_8, dev_s1_v2_8, dev_s1_t2_9, dev_s1_v2_9,
        s1_kernel_1, s1_kernel_2, s1_kernel_3, s1_kernel_4, s1_kernel_5, s1_kernel_6, s1_kernel_7, s1_kernel_8, s1_kernel_9, 
        //
        dev_eval_h1, dev_eval_h2, dev_eval_h3, dev_eval_p4, dev_eval_p5, dev_eval_p6,
        dev_energies, factor, 
        //  
        CEIL(size_h3, FUSION_SIZE_SLICE_1_H3), CEIL(size_h2, FUSION_SIZE_SLICE_1_H2), CEIL(size_h1, FUSION_SIZE_SLICE_1_H1), 
        CEIL(size_p6, FUSION_SIZE_SLICE_1_P6), CEIL(size_p5, FUSION_SIZE_SLICE_1_P5), CEIL(size_p4, FUSION_SIZE_SLICE_1_P4), 
        // 
        size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_h7, size_p7,
        size_h7,
        str_reg_x_1, str_reg_y_1);

    //
    cudaMemcpy(host_energies, dev_energies, num_blocks_kernel_1 * NUM_ENERGIES * sizeof(double), cudaMemcpyDeviceToHost);

    // 
    double final_energy_1 = 0.0;
    double final_energy_2 = 0.0;
    for (size_t i = 0; i < num_blocks_kernel_1; i++)
    {
        final_energy_1 += host_energies[i];
        final_energy_2 += host_energies[i + num_blocks_kernel_1];
    }

#ifdef DEBUG_TIME_FUSED_CCSD_T
    cudaEventRecord(stop_ccsd_t);
    cudaEventSynchronize(stop_ccsd_t);
    float time_ms_ccsd_t_kernel = 0.0;
    cudaEventElapsedTime(&time_ms_ccsd_t_kernel, start_ccsd_t, stop_ccsd_t);
    printf ("[%s][fused] total-time: %f (ms)\n", __func__, time_ms_ccsd_t_kernel);
    printf ("[%s][fused] E(4): %.15f, E(5): %.15f\n", __func__, final_energy_1, final_energy_2);    
#endif

    if (d1_kernel_1 == 1)
    {
        freeGpuMem(dev_d1_t2_1); freeGpuMem(dev_d1_v2_1); 
    }

    if (d1_kernel_2 == 1)
    {
        freeGpuMem(dev_d1_t2_2); freeGpuMem(dev_d1_v2_2); 
    }

    if (d1_kernel_3 == 1)
    {
        freeGpuMem(dev_d1_t2_3); freeGpuMem(dev_d1_v2_3); 
    }

    if (d1_kernel_4 == 1)
    {
        freeGpuMem(dev_d1_t2_4); freeGpuMem(dev_d1_v2_4); 
    }

    if (d1_kernel_5 == 1)
    {
        freeGpuMem(dev_d1_t2_5); freeGpuMem(dev_d1_v2_5); 
    }

    if (d1_kernel_6 == 1)
    {
        freeGpuMem(dev_d1_t2_6); freeGpuMem(dev_d1_v2_6);
    }

    if (d1_kernel_7 == 1)
    {
        freeGpuMem(dev_d1_t2_7); freeGpuMem(dev_d1_v2_7); 
    }

    if (d1_kernel_8 == 1)
    {
        freeGpuMem(dev_d1_t2_8); freeGpuMem(dev_d1_v2_8); 
    }

    if (d1_kernel_9 == 1)
    {
        freeGpuMem(dev_d1_t2_9); freeGpuMem(dev_d1_v2_9);
    }

    if (d2_kernel_1 == 1)
    {  
        freeGpuMem(dev_d2_t2_1); freeGpuMem(dev_d2_v2_1); 
    }

    if (d2_kernel_2 == 1)
    {
        freeGpuMem(dev_d2_t2_2); freeGpuMem(dev_d2_v2_2); 
    }

    if (d2_kernel_3 == 1)
    {
        freeGpuMem(dev_d2_t2_3); freeGpuMem(dev_d2_v2_3); 
    }

    if (d2_kernel_4 == 1)
    {
        freeGpuMem(dev_d2_t2_4); freeGpuMem(dev_d2_v2_4); 
    }

    if (d2_kernel_5 == 1)
    {
        freeGpuMem(dev_d2_t2_5); freeGpuMem(dev_d2_v2_5); 
    }

    if (d2_kernel_6 == 1)
    {   
        freeGpuMem(dev_d2_t2_6); freeGpuMem(dev_d2_v2_6);
    }

    if (d2_kernel_7 == 1)
    {   
        freeGpuMem(dev_d2_t2_7); freeGpuMem(dev_d2_v2_7); 
    }

    if (d2_kernel_8 == 1)
    {   
        freeGpuMem(dev_d2_t2_8); freeGpuMem(dev_d2_v2_8); 
    }

    if (d2_kernel_9 == 1)
    {   
        freeGpuMem(dev_d2_t2_9); freeGpuMem(dev_d2_v2_9);
    }

    // s1
    if (s1_kernel_1 == 1)
    {    
        freeGpuMem(dev_s1_t2_1); freeGpuMem(dev_s1_v2_1); 
    }

    if (s1_kernel_2 == 1)
    {        
        freeGpuMem(dev_s1_t2_2); freeGpuMem(dev_s1_v2_2); 
    }   

    if (s1_kernel_3 == 1)
    {   
        freeGpuMem(dev_s1_t2_3); freeGpuMem(dev_s1_v2_3); 
    }

    if (s1_kernel_4 == 1)
    {   
        freeGpuMem(dev_s1_t2_4); freeGpuMem(dev_s1_v2_4); 
    }

    if (s1_kernel_5 == 1)
    {   
        freeGpuMem(dev_s1_t2_5); freeGpuMem(dev_s1_v2_5); 
    }

    if (s1_kernel_6 == 1)
    {   
        freeGpuMem(dev_s1_t2_6); freeGpuMem(dev_s1_v2_6);
    }   

    if (s1_kernel_7 == 1)
    {   
        freeGpuMem(dev_s1_t2_7); freeGpuMem(dev_s1_v2_7); 
    }   

    if (s1_kernel_8 == 1)
    {   
        freeGpuMem(dev_s1_t2_8); freeGpuMem(dev_s1_v2_8); 
    }   

    if (s1_kernel_9 == 1)
    {   
        freeGpuMem(dev_s1_t2_9); freeGpuMem(dev_s1_v2_9);
    }
}
