#include "header.h"
typedef long Integer;

/*----------------------------------------------------------------------*
 *  [d1][1] triplesx[h3,h1,p6,p5,p4] -= t2sub[h7,p4,p5,h1] * v2sub[h3,p6,h7]
 *----------------------------------------------------------------------*/
// created by tc_gen_definition_new()
#define JK_CCSD_T_D1_1_SIZE_SLICE_1_G 16
#define JK_CCSD_T_D1_1_SIZE_SLICE_1_A 16
#define JK_CCSD_T_D1_1_SIZE_SLICE_1_B 4
#define JK_CCSD_T_D1_1_SIZE_SLICE_1_D 1
#define JK_CCSD_T_D1_1_SIZE_SLICE_1_F 8
#define JK_CCSD_T_D1_1_SIZE_SLICE_1_E 8
#define JK_CCSD_T_D1_1_SIZE_SLICE_1_C 1

#define JK_CCSD_T_D1_1_SIZE_INT_UNIT_1 JK_CCSD_T_D1_1_SIZE_SLICE_1_G

#define JK_CCSD_T_D1_1_SIZE_TB_1_X 	JK_CCSD_T_D1_1_SIZE_SLICE_1_A * JK_CCSD_T_D1_1_SIZE_SLICE_1_D
#define JK_CCSD_T_D1_1_SIZE_TB_1_Y 	JK_CCSD_T_D1_1_SIZE_SLICE_1_F * JK_CCSD_T_D1_1_SIZE_SLICE_1_C
#define JK_CCSD_T_D1_1_SIZE_REG_1_X 	JK_CCSD_T_D1_1_SIZE_SLICE_1_B
#define JK_CCSD_T_D1_1_SIZE_REG_1_Y 	JK_CCSD_T_D1_1_SIZE_SLICE_1_E

#define NUM_INDEX 		6
#define CEIL(a, b) 		(((a) + (b) - 1) / (b))

// created by tc_gen_code_Kernel()
__global__ void jk_ccsd_t_d1_1_kernel__4_1(double* dev_t3, double* dev_t2, double* dev_v2, int size_a, int size_b, int size_c, int size_d, int size_e, int size_f, int size_g, int numBlk_a, int numBlk_b, int numBlk_c, int numBlk_d, int numBlk_e, int numBlk_f, int stride_int_t2, int stride_int_v2, int stride_reg_x, int stride_reg_y, int size_internal)
{
	// For Shared Memory,
	__shared__ double sm_a[16][64];
	__shared__ double sm_b[16][64];


	int internal_upperbound   = 0;
	int internal_offset;

	// when opt_pre_computed == -1, all indices will be calculated manually
	// # of indices mapped on TB_X: 2
	// # of indices mapped on TB_Y: 2
	int idx_a = threadIdx.x % JK_CCSD_T_D1_1_SIZE_SLICE_1_A;
	int idx_d = threadIdx.x / JK_CCSD_T_D1_1_SIZE_SLICE_1_A;
	int idx_f = threadIdx.y % JK_CCSD_T_D1_1_SIZE_SLICE_1_F;
	int idx_c = threadIdx.y / JK_CCSD_T_D1_1_SIZE_SLICE_1_F;

	int tmp_blkIdx;
	int blk_idx_f = blockIdx.x / (numBlk_e * numBlk_d * numBlk_c * numBlk_b * numBlk_a);
	tmp_blkIdx = blockIdx.x % (numBlk_e * numBlk_d * numBlk_c * numBlk_b * numBlk_a);

	int blk_idx_e = tmp_blkIdx / (numBlk_d * numBlk_c * numBlk_b * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_d * numBlk_c * numBlk_b * numBlk_a);

	int blk_idx_d = tmp_blkIdx / (numBlk_c * numBlk_b * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_c * numBlk_b * numBlk_a);

	int blk_idx_c = tmp_blkIdx / (numBlk_b * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_b * numBlk_a);

	int blk_idx_b = tmp_blkIdx / numBlk_a;
	tmp_blkIdx = tmp_blkIdx % (numBlk_a);

	int  blk_idx_a = tmp_blkIdx;

	int t3_base_thread = blk_idx_a * JK_CCSD_T_D1_1_SIZE_SLICE_1_A + idx_a + (blk_idx_b * JK_CCSD_T_D1_1_SIZE_SLICE_1_B + (blk_idx_c * JK_CCSD_T_D1_1_SIZE_SLICE_1_C + idx_c + (blk_idx_d * JK_CCSD_T_D1_1_SIZE_SLICE_1_D + idx_d + (blk_idx_e * JK_CCSD_T_D1_1_SIZE_SLICE_1_E + (blk_idx_f * JK_CCSD_T_D1_1_SIZE_SLICE_1_F + idx_f) * size_e) * size_d) * size_c) * size_b) * size_a;

	// need to support partial tiles
	int rng_a, rng_b, rng_c, rng_d, rng_e, rng_f;
	if ((size_a - (blk_idx_a * JK_CCSD_T_D1_1_SIZE_SLICE_1_A)) >= JK_CCSD_T_D1_1_SIZE_SLICE_1_A)
	{
		rng_a = JK_CCSD_T_D1_1_SIZE_SLICE_1_A;
	}
	else
	{
		rng_a = size_a % JK_CCSD_T_D1_1_SIZE_SLICE_1_A;
	}
	if ((size_b - (blk_idx_b * JK_CCSD_T_D1_1_SIZE_SLICE_1_B)) >= JK_CCSD_T_D1_1_SIZE_SLICE_1_B)
	{
		rng_b = JK_CCSD_T_D1_1_SIZE_SLICE_1_B;
	}
	else
	{
		rng_b = size_b % JK_CCSD_T_D1_1_SIZE_SLICE_1_B;
	}
	if ((size_c - (blk_idx_c * JK_CCSD_T_D1_1_SIZE_SLICE_1_C)) >= JK_CCSD_T_D1_1_SIZE_SLICE_1_C)
	{
		rng_c = JK_CCSD_T_D1_1_SIZE_SLICE_1_C;
	}
	else
	{
		rng_c = size_c % JK_CCSD_T_D1_1_SIZE_SLICE_1_C;
	}
	if ((size_d - (blk_idx_d * JK_CCSD_T_D1_1_SIZE_SLICE_1_D)) >= JK_CCSD_T_D1_1_SIZE_SLICE_1_D)
	{
		rng_d = JK_CCSD_T_D1_1_SIZE_SLICE_1_D;
	}
	else
	{
		rng_d = size_d % JK_CCSD_T_D1_1_SIZE_SLICE_1_D;
	}
	if ((size_e - (blk_idx_e * JK_CCSD_T_D1_1_SIZE_SLICE_1_E)) >= JK_CCSD_T_D1_1_SIZE_SLICE_1_E)
	{
		rng_e = JK_CCSD_T_D1_1_SIZE_SLICE_1_E;
	}
	else
	{
		rng_e = size_e % JK_CCSD_T_D1_1_SIZE_SLICE_1_E;
	}
	if ((size_f - (blk_idx_f * JK_CCSD_T_D1_1_SIZE_SLICE_1_F)) >= JK_CCSD_T_D1_1_SIZE_SLICE_1_F)
	{
		rng_f = JK_CCSD_T_D1_1_SIZE_SLICE_1_F;
	}
	else
	{
		rng_f = size_f % JK_CCSD_T_D1_1_SIZE_SLICE_1_F;
	}

	double temp_av;
	double temp_bv[8];
	double reg_tile[8][4];

	for (int i = 0; i < 8; i++)
	for (int j = 0; j < 4; j++)
	reg_tile[i][j] = 0.0;

	// tensor contraction: [[16, 'STR_SD2_T2_H7', 'y', 't2', ['g', 'f', 'e', 'c']], [16, 'STR_SD2_V2_H7', 'x', 'v2', ['a', 'b', 'd', 'g']], '-=']
	#pragma unroll 1
	for (int l = 0; l < size_internal; l += JK_CCSD_T_D1_1_SIZE_INT_UNIT_1)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + JK_CCSD_T_D1_1_SIZE_INT_UNIT_1) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		//---------------------------------------------------------------------------------------------------
		// This is for the new version
		// This Part is for Loading Input-Left
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (idx_f < rng_f && 0 < rng_c && threadIdx.x < JK_CCSD_T_D1_1_SIZE_INT_UNIT_1 - internal_upperbound)
		for (int ll = 0; ll < rng_e; ll++)
		{
			// ['g', 'f', 'e', 'c']
			// Exception: Temp. version!: threadIdx.x + l
			// Exception: Temp. version!: idx_f < rng_f
			sm_a[threadIdx.x][threadIdx.y + ll * 8] = dev_t2[(blk_idx_f * JK_CCSD_T_D1_1_SIZE_SLICE_1_F + idx_f + (blk_idx_e * JK_CCSD_T_D1_1_SIZE_SLICE_1_E + ll + (blk_idx_c * JK_CCSD_T_D1_1_SIZE_SLICE_1_C + 0) * size_e) * size_f) * size_g + (threadIdx.x + l)];
		}
		
		// This Part is for Loading Input-Right
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (idx_a < rng_a && 0 < rng_d && threadIdx.y < JK_CCSD_T_D1_1_SIZE_INT_UNIT_1 - internal_upperbound)
		for (int ll = 0; ll < rng_b; ll++)
		{
			// ['a', 'b', 'd', 'g']
			// Exception: Temp. version!: threadIdx.y + l + 0
			// Exception: Temp. version!: idx_a < rng_a
			sm_b[threadIdx.y + 0][threadIdx.x + ll * 16] = dev_v2[blk_idx_a * JK_CCSD_T_D1_1_SIZE_SLICE_1_A + idx_a + (blk_idx_b * JK_CCSD_T_D1_1_SIZE_SLICE_1_B + ll + (blk_idx_d * JK_CCSD_T_D1_1_SIZE_SLICE_1_D + 0) * size_b) * size_a + (threadIdx.y + l + 0) * stride_int_v2];
			// Exception: Temp. version!: threadIdx.y + l + 8
			// Exception: Temp. version!: idx_a < rng_a
			if (threadIdx.y + l + 8 < size_internal && idx_a < rng_a) 
			sm_b[threadIdx.y + 8][threadIdx.x + ll * 16] = dev_v2[blk_idx_a * JK_CCSD_T_D1_1_SIZE_SLICE_1_A + idx_a + (blk_idx_b * JK_CCSD_T_D1_1_SIZE_SLICE_1_B + ll + (blk_idx_d * JK_CCSD_T_D1_1_SIZE_SLICE_1_D + 0) * size_b) * size_a + (threadIdx.y + l + 8) * stride_int_v2];
		}
		__syncthreads();
		//---------------------------------------------------------------------------------------------------
		

		// Part: Generalized Threads
		for (int ll = 0; ll < JK_CCSD_T_D1_1_SIZE_INT_UNIT_1 - internal_upperbound; ll++)
		{
			temp_bv[0] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_1_SIZE_SLICE_1_F + 0];
			temp_bv[1] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_1_SIZE_SLICE_1_F + 8];
			temp_bv[2] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_1_SIZE_SLICE_1_F + 16];
			temp_bv[3] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_1_SIZE_SLICE_1_F + 24];
			temp_bv[4] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_1_SIZE_SLICE_1_F + 32];
			temp_bv[5] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_1_SIZE_SLICE_1_F + 40];
			temp_bv[6] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_1_SIZE_SLICE_1_F + 48];
			temp_bv[7] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_1_SIZE_SLICE_1_F + 56];

			for (int xx = 0; xx < 4; xx++) // (1)
			{
				temp_av = sm_b[ll][idx_a + (idx_d) * JK_CCSD_T_D1_1_SIZE_SLICE_1_A + (xx * 16)];

				reg_tile[0][xx] -= temp_av * temp_bv[0];
				reg_tile[1][xx] -= temp_av * temp_bv[1];
				reg_tile[2][xx] -= temp_av * temp_bv[2];
				reg_tile[3][xx] -= temp_av * temp_bv[3];
				reg_tile[4][xx] -= temp_av * temp_bv[4];
				reg_tile[5][xx] -= temp_av * temp_bv[5];
				reg_tile[6][xx] -= temp_av * temp_bv[6];
				reg_tile[7][xx] -= temp_av * temp_bv[7];
			}
		}
		__syncthreads();
	}


	// Store Results (Registers) to Global Memory
	// Part: Generalized Threads
	// Part: Generalized Register-Tiling
	if (idx_a < rng_a && idx_d < rng_d && idx_f < rng_f && idx_c < rng_c)
	for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			if(i < rng_e && j < rng_b)
			{
			    dev_t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] += reg_tile[i][j];
			}
		}
	}
}

// written by tc_interface.tc_gen_code_interface_Header()
extern "C"
void jk_ccsd_t_d1_1_fusion(int size_a, int size_b, int size_c, int size_d, int size_e, int size_f, int size_g, double* t3, double* host_t2, double* host_v2, int cond_kernel_1, int opt_register_transpose)
{
	int num_thread_blocks_kernel_1;

	double* dev_t3;
	double* dev_t2;
	double* dev_v2;

	num_thread_blocks_kernel_1 = CEIL(size_a, JK_CCSD_T_D1_1_SIZE_SLICE_1_A) * CEIL(size_b, JK_CCSD_T_D1_1_SIZE_SLICE_1_B) * CEIL(size_c, JK_CCSD_T_D1_1_SIZE_SLICE_1_C) * CEIL(size_d, JK_CCSD_T_D1_1_SIZE_SLICE_1_D) * CEIL(size_e, JK_CCSD_T_D1_1_SIZE_SLICE_1_E) * CEIL(size_f, JK_CCSD_T_D1_1_SIZE_SLICE_1_F);
    
    // cudaMalloc()
	//cudaMalloc((void**) &dev_t3, sizeof(double) * size_a * size_b * size_c * size_d * size_e * size_f);
	cudaMalloc((void**) &dev_t2, sizeof(double) * size_c * size_e * size_f * size_g);
	cudaMalloc((void**) &dev_v2, sizeof(double) * size_g * size_d * size_b * size_a);

	// cudaMemcpy()
	//cudaMemcpy(dev_t3, t3, sizeof(double) * size_a * size_b * size_c * size_d * size_e * size_f, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_c * size_e * size_f * size_g, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_g * size_d * size_b * size_a, cudaMemcpyHostToDevice);

	// Related to Kernels
	// There are 1 Basic Kernels
    long long int tmp_operations = 2 * (long long int)(size_a * size_b * size_c * size_d * size_e * size_f) * size_g;
    /*
	printf ("========================================= fusedKernels =============================================\n");
	printf ("		Grid Size  : %6d (1D)\n", num_thread_blocks_kernel_1);
	printf ("		Block-size : %2d, %2d (2D)\n", JK_CCSD_T_D1_1_SIZE_TB_1_X, JK_CCSD_T_D1_1_SIZE_TB_1_Y);
	printf ("		Reg.-size  : %2d, %2d (2D)\n", JK_CCSD_T_D1_1_SIZE_REG_1_X, JK_CCSD_T_D1_1_SIZE_REG_1_Y);
	printf ("		A thread deals with (%d x %d) elements (basically)\n", JK_CCSD_T_D1_1_SIZE_TB_1_X * JK_CCSD_T_D1_1_SIZE_REG_1_X, JK_CCSD_T_D1_1_SIZE_TB_1_Y * JK_CCSD_T_D1_1_SIZE_REG_1_Y);
	printf ("		# of Operations: %lld\n", tmp_operations);
    printf ("====================================================================================================\n");
    */
	dim3 gridsize_1(num_thread_blocks_kernel_1);
	dim3 blocksize_1(JK_CCSD_T_D1_1_SIZE_TB_1_X, JK_CCSD_T_D1_1_SIZE_TB_1_Y);

	int stride_output_a = 1;
	int stride_output_b = stride_output_a * size_a;
	int stride_output_c = stride_output_b * size_b;
	int stride_output_d = stride_output_c * size_c;
	int stride_output_e = stride_output_d * size_d;
	int stride_output_f = stride_output_e * size_e;

	int stride_reg_x_1 = stride_output_b;
	int stride_reg_y_1 = stride_output_e;

	int size_internal = size_g;

	int stride_int_t2 = 1;
	int stride_int_v2 = size_a * size_b * size_d;

    dev_t3 = t3_d;

	// New Caller
	jk_ccsd_t_d1_1_kernel__4_1<<<gridsize_1, blocksize_1>>>(dev_t3, dev_t2, dev_v2, size_a, size_b, size_c, size_d, size_e, size_f, size_g, CEIL(size_a, JK_CCSD_T_D1_1_SIZE_SLICE_1_A), CEIL(size_b, JK_CCSD_T_D1_1_SIZE_SLICE_1_B), CEIL(size_c, JK_CCSD_T_D1_1_SIZE_SLICE_1_C), CEIL(size_d, JK_CCSD_T_D1_1_SIZE_SLICE_1_D), CEIL(size_e, JK_CCSD_T_D1_1_SIZE_SLICE_1_E), CEIL(size_f, JK_CCSD_T_D1_1_SIZE_SLICE_1_F), stride_int_t2, stride_int_v2, stride_reg_x_1, stride_reg_y_1, size_internal);

	// Copy the Result from Device to Host
	//cudaMemcpy(t3, dev_t3, sizeof(double) * (size_a * size_b * size_c * size_d * size_e * size_f), cudaMemcpyDeviceToHost);

	// cudaFree()
	//cudaFree(dev_t3);	
    cudaFree(dev_t2);	cudaFree(dev_v2);

	// Shoule be Fixed
	// HostFree

}

// This is written by tc_interface.tc_gen_code_interface()
// This Interface Should be Called to Run the Kernels
extern "C"
void jk_ccsd_t_d1_1_fusion_(int size_a, int size_b, int size_c, int size_d, int size_e, int size_f, int size_g, double* t3, double* t2, double* v2, int cond_kernel_1, int opt_register_transpose)
{
	// Call An Application
	jk_ccsd_t_d1_1_fusion(size_a, size_b, size_c, size_d, size_e, size_f, size_g, t3, t2, v2, cond_kernel_1, opt_register_transpose);
}

/*----------------------------------------------------------------------*
 *  [d1][2] triplesx[h3,h1,h2,p5,p4] += t2sub[h7,p4,p5,h1] * v2sub[h3,h2,h7]
 *----------------------------------------------------------------------*/
// created by tc_gen_definition_new()
#define JK_CCSD_T_D1_2_SIZE_SLICE_1_G   16
#define JK_CCSD_T_D1_2_SIZE_SLICE_1_A   16
#define JK_CCSD_T_D1_2_SIZE_SLICE_1_B   4
#define JK_CCSD_T_D1_2_SIZE_SLICE_1_D   1
#define JK_CCSD_T_D1_2_SIZE_SLICE_1_F   8
#define JK_CCSD_T_D1_2_SIZE_SLICE_1_E   8
#define JK_CCSD_T_D1_2_SIZE_SLICE_1_C   1

#define JK_CCSD_T_D1_2_SIZE_INT_UNIT_1  JK_CCSD_T_D1_2_SIZE_SLICE_1_G

#define JK_CCSD_T_D1_2_SIZE_TB_1_X 	    JK_CCSD_T_D1_2_SIZE_SLICE_1_A * JK_CCSD_T_D1_2_SIZE_SLICE_1_D
#define JK_CCSD_T_D1_2_SIZE_TB_1_Y 	    JK_CCSD_T_D1_2_SIZE_SLICE_1_F * JK_CCSD_T_D1_2_SIZE_SLICE_1_C
#define JK_CCSD_T_D1_2_SIZE_REG_1_X 	JK_CCSD_T_D1_2_SIZE_SLICE_1_B
#define JK_CCSD_T_D1_2_SIZE_REG_1_Y 	JK_CCSD_T_D1_2_SIZE_SLICE_1_E

// created by tc_gen_code_Kernel()
__global__ void jk_ccsd_t_d1_2_kernel__4_1(double* dev_t3, double* dev_t2, double* dev_v2, int size_a, int size_c, int size_b, int size_d, int size_e, int size_f, int size_g, int numBlk_a, int numBlk_c, int numBlk_b, int numBlk_d, int numBlk_e, int numBlk_f, int stride_int_t2, int stride_int_v2, int stride_reg_x, int stride_reg_y, int size_internal)
{
	// For Shared Memory,
	__shared__ double sm_a[16][64];
	__shared__ double sm_b[16][64];


	int internal_upperbound   = 0;
	int internal_offset;

	// when opt_pre_computed == -1, all indices will be calculated manually
	// # of indices mapped on TB_X: 2
	// # of indices mapped on TB_Y: 2
	int idx_a = threadIdx.x % JK_CCSD_T_D1_2_SIZE_SLICE_1_A;
	int idx_d = threadIdx.x / JK_CCSD_T_D1_2_SIZE_SLICE_1_A;
	int idx_f = threadIdx.y % JK_CCSD_T_D1_2_SIZE_SLICE_1_F;
	int idx_c = threadIdx.y / JK_CCSD_T_D1_2_SIZE_SLICE_1_F;

	int tmp_blkIdx;
	int blk_idx_f = blockIdx.x / (numBlk_e * numBlk_d * numBlk_b * numBlk_c * numBlk_a);
	tmp_blkIdx = blockIdx.x % (numBlk_e * numBlk_d * numBlk_b * numBlk_c * numBlk_a);

	int blk_idx_e = tmp_blkIdx / (numBlk_d * numBlk_b * numBlk_c * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_d * numBlk_b * numBlk_c * numBlk_a);

	int blk_idx_d = tmp_blkIdx / (numBlk_b * numBlk_c * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_b * numBlk_c * numBlk_a);

	int blk_idx_b = tmp_blkIdx / (numBlk_c * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_c * numBlk_a);

	int blk_idx_c = tmp_blkIdx / numBlk_a;
	tmp_blkIdx = tmp_blkIdx % (numBlk_a);

	int  blk_idx_a = tmp_blkIdx;

	int t3_base_thread = blk_idx_a * JK_CCSD_T_D1_2_SIZE_SLICE_1_A + idx_a + (blk_idx_c * JK_CCSD_T_D1_2_SIZE_SLICE_1_C + idx_c + (blk_idx_b * JK_CCSD_T_D1_2_SIZE_SLICE_1_B + (blk_idx_d * JK_CCSD_T_D1_2_SIZE_SLICE_1_D + idx_d + (blk_idx_e * JK_CCSD_T_D1_2_SIZE_SLICE_1_E + (blk_idx_f * JK_CCSD_T_D1_2_SIZE_SLICE_1_F + idx_f) * size_e) * size_d) * size_b) * size_c) * size_a;

	// need to support partial tiles
	int rng_a, rng_c, rng_b, rng_d, rng_e, rng_f;
	if ((size_a - (blk_idx_a * JK_CCSD_T_D1_2_SIZE_SLICE_1_A)) >= JK_CCSD_T_D1_2_SIZE_SLICE_1_A)
	{
		rng_a = JK_CCSD_T_D1_2_SIZE_SLICE_1_A;
	}
	else
	{
		rng_a = size_a % JK_CCSD_T_D1_2_SIZE_SLICE_1_A;
	}
	if ((size_c - (blk_idx_c * JK_CCSD_T_D1_2_SIZE_SLICE_1_C)) >= JK_CCSD_T_D1_2_SIZE_SLICE_1_C)
	{
		rng_c = JK_CCSD_T_D1_2_SIZE_SLICE_1_C;
	}
	else
	{
		rng_c = size_c % JK_CCSD_T_D1_2_SIZE_SLICE_1_C;
	}
	if ((size_b - (blk_idx_b * JK_CCSD_T_D1_2_SIZE_SLICE_1_B)) >= JK_CCSD_T_D1_2_SIZE_SLICE_1_B)
	{
		rng_b = JK_CCSD_T_D1_2_SIZE_SLICE_1_B;
	}
	else
	{
		rng_b = size_b % JK_CCSD_T_D1_2_SIZE_SLICE_1_B;
	}
	if ((size_d - (blk_idx_d * JK_CCSD_T_D1_2_SIZE_SLICE_1_D)) >= JK_CCSD_T_D1_2_SIZE_SLICE_1_D)
	{
		rng_d = JK_CCSD_T_D1_2_SIZE_SLICE_1_D;
	}
	else
	{
		rng_d = size_d % JK_CCSD_T_D1_2_SIZE_SLICE_1_D;
	}
	if ((size_e - (blk_idx_e * JK_CCSD_T_D1_2_SIZE_SLICE_1_E)) >= JK_CCSD_T_D1_2_SIZE_SLICE_1_E)
	{
		rng_e = JK_CCSD_T_D1_2_SIZE_SLICE_1_E;
	}
	else
	{
		rng_e = size_e % JK_CCSD_T_D1_2_SIZE_SLICE_1_E;
	}
	if ((size_f - (blk_idx_f * JK_CCSD_T_D1_2_SIZE_SLICE_1_F)) >= JK_CCSD_T_D1_2_SIZE_SLICE_1_F)
	{
		rng_f = JK_CCSD_T_D1_2_SIZE_SLICE_1_F;
	}
	else
	{
		rng_f = size_f % JK_CCSD_T_D1_2_SIZE_SLICE_1_F;
	}

	double temp_av;
	double temp_bv[8];
	double reg_tile[8][4];

	for (int i = 0; i < 8; i++)
	for (int j = 0; j < 4; j++)
	reg_tile[i][j] = 0.0;

	// tensor contraction: [[16, 'STR_SD2_T2_H7', 'y', 't2', ['g', 'f', 'e', 'c']], [16, 'STR_SD2_V2_H7', 'x', 'v2', ['a', 'b', 'd', 'g']], '-=']
	#pragma unroll 1
	for (int l = 0; l < size_internal; l += JK_CCSD_T_D1_2_SIZE_INT_UNIT_1)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + JK_CCSD_T_D1_2_SIZE_INT_UNIT_1) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		//---------------------------------------------------------------------------------------------------
		// This is for the new version
		// This Part is for Loading Input-Left
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (idx_f < rng_f && 0 < rng_c && threadIdx.x < JK_CCSD_T_D1_2_SIZE_INT_UNIT_1 - internal_upperbound)
		for (int ll = 0; ll < rng_e; ll++)
		{
			// ['g', 'f', 'e', 'c']
			// Exception: Temp. version!: threadIdx.x + l
			// Exception: Temp. version!: idx_f < rng_f
			sm_a[threadIdx.x][threadIdx.y + ll * 8] = dev_t2[(blk_idx_f * JK_CCSD_T_D1_2_SIZE_SLICE_1_F + idx_f + (blk_idx_e * JK_CCSD_T_D1_2_SIZE_SLICE_1_E + ll + (blk_idx_c * JK_CCSD_T_D1_2_SIZE_SLICE_1_C + 0) * size_e) * size_f) * size_g + (threadIdx.x + l)];
		}
		
		// This Part is for Loading Input-Right
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (idx_a < rng_a && 0 < rng_d && threadIdx.y < JK_CCSD_T_D1_2_SIZE_INT_UNIT_1 - internal_upperbound)
		for (int ll = 0; ll < rng_b; ll++)
		{
			// ['a', 'b', 'd', 'g']
			// Exception: Temp. version!: threadIdx.y + l + 0
			// Exception: Temp. version!: idx_a < rng_a
			sm_b[threadIdx.y + 0][threadIdx.x + ll * 16] = dev_v2[blk_idx_a * JK_CCSD_T_D1_2_SIZE_SLICE_1_A + idx_a + (blk_idx_b * JK_CCSD_T_D1_2_SIZE_SLICE_1_B + ll + (blk_idx_d * JK_CCSD_T_D1_2_SIZE_SLICE_1_D + 0) * size_b) * size_a + (threadIdx.y + l + 0) * stride_int_v2];
			// Exception: Temp. version!: threadIdx.y + l + 8
			// Exception: Temp. version!: idx_a < rng_a
			if (threadIdx.y + l + 8 < size_internal && idx_a < rng_a) 
			sm_b[threadIdx.y + 8][threadIdx.x + ll * 16] = dev_v2[blk_idx_a * JK_CCSD_T_D1_2_SIZE_SLICE_1_A + idx_a + (blk_idx_b * JK_CCSD_T_D1_2_SIZE_SLICE_1_B + ll + (blk_idx_d * JK_CCSD_T_D1_2_SIZE_SLICE_1_D + 0) * size_b) * size_a + (threadIdx.y + l + 8) * stride_int_v2];
		}
		__syncthreads();
		//---------------------------------------------------------------------------------------------------
		

		// Part: Generalized Threads
		for (int ll = 0; ll < JK_CCSD_T_D1_2_SIZE_INT_UNIT_1 - internal_upperbound; ll++)
		{
			temp_bv[0] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_2_SIZE_SLICE_1_F + 0];
			temp_bv[1] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_2_SIZE_SLICE_1_F + 8];
			temp_bv[2] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_2_SIZE_SLICE_1_F + 16];
			temp_bv[3] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_2_SIZE_SLICE_1_F + 24];
			temp_bv[4] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_2_SIZE_SLICE_1_F + 32];
			temp_bv[5] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_2_SIZE_SLICE_1_F + 40];
			temp_bv[6] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_2_SIZE_SLICE_1_F + 48];
			temp_bv[7] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_2_SIZE_SLICE_1_F + 56];

			for (int xx = 0; xx < 4; xx++) // (1)
			{
				temp_av = sm_b[ll][idx_a + (idx_d) * JK_CCSD_T_D1_2_SIZE_SLICE_1_A + (xx * 16)];

				reg_tile[0][xx] += temp_av * temp_bv[0];
				reg_tile[1][xx] += temp_av * temp_bv[1];
				reg_tile[2][xx] += temp_av * temp_bv[2];
				reg_tile[3][xx] += temp_av * temp_bv[3];
				reg_tile[4][xx] += temp_av * temp_bv[4];
				reg_tile[5][xx] += temp_av * temp_bv[5];
				reg_tile[6][xx] += temp_av * temp_bv[6];
				reg_tile[7][xx] += temp_av * temp_bv[7];
			}
		}
		__syncthreads();
	}


	// Store Results (Registers) to Global Memory
	// Part: Generalized Threads
	// Part: Generalized Register-Tiling
	if (idx_a < rng_a && idx_d < rng_d && idx_f < rng_f && idx_c < rng_c)
	for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			if(i < rng_e && j < rng_b)
			{
			dev_t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] += reg_tile[i][j];
			}
		}
	}
}

// written by tc_interface.tc_gen_code_interface_Header()
extern "C"
void jk_ccsd_t_d1_2_fusion(int size_a, int size_c, int size_b, int size_d, int size_e, int size_f, int size_g, double* t3, double* host_t2, double* host_v2, int cond_kernel_1, int opt_register_transpose)
{
	int num_thread_blocks_kernel_1;

	double* dev_t3;
	double* dev_t2;
	double* dev_v2;

	num_thread_blocks_kernel_1 = CEIL(size_a, JK_CCSD_T_D1_2_SIZE_SLICE_1_A) * CEIL(size_c, JK_CCSD_T_D1_2_SIZE_SLICE_1_C) * CEIL(size_b, JK_CCSD_T_D1_2_SIZE_SLICE_1_B) * CEIL(size_d, JK_CCSD_T_D1_2_SIZE_SLICE_1_D) * CEIL(size_e, JK_CCSD_T_D1_2_SIZE_SLICE_1_E) * CEIL(size_f, JK_CCSD_T_D1_2_SIZE_SLICE_1_F);
    
    // cudaMalloc()
	//cudaMalloc((void**) &dev_t3, sizeof(double) * size_a * size_c * size_b * size_d * size_e * size_f);
	cudaMalloc((void**) &dev_t2, sizeof(double) * size_c * size_e * size_f * size_g);
	cudaMalloc((void**) &dev_v2, sizeof(double) * size_g * size_d * size_b * size_a);

	// cudaMemcpy()
	//cudaMemcpy(dev_t3, t3, sizeof(double) * size_a * size_c * size_b * size_d * size_e * size_f, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_c * size_e * size_f * size_g, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_g * size_d * size_b * size_a, cudaMemcpyHostToDevice);

	// Related to Kernels
	// There are 1 Basic Kernels
    long long int tmp_operations = 2 * (long long int)(size_a * size_c * size_b * size_d * size_e * size_f) * size_g;
    /*
	printf ("========================================= fusedKernels =============================================\n");
	printf ("		Grid Size  : %6d (1D)\n", num_thread_blocks_kernel_1);
	printf ("		Block-size : %2d, %2d (2D)\n", JK_CCSD_T_D1_2_SIZE_TB_1_X, JK_CCSD_T_D1_2_SIZE_TB_1_Y);
	printf ("		Reg.-size  : %2d, %2d (2D)\n", JK_CCSD_T_D1_2_SIZE_REG_1_X, JK_CCSD_T_D1_2_SIZE_REG_1_Y);
	printf ("		A thread deals with (%d x %d) elements (basically)\n", JK_CCSD_T_D1_2_SIZE_TB_1_X * JK_CCSD_T_D1_2_SIZE_REG_1_X, JK_CCSD_T_D1_2_SIZE_TB_1_Y * JK_CCSD_T_D1_2_SIZE_REG_1_Y);
	printf ("		# of Operations: %lld\n", tmp_operations);
    printf ("====================================================================================================\n");
    */
	dim3 gridsize_1(num_thread_blocks_kernel_1);
	dim3 blocksize_1(JK_CCSD_T_D1_2_SIZE_TB_1_X, JK_CCSD_T_D1_2_SIZE_TB_1_Y);

	int stride_output_a = 1;
	int stride_output_c = stride_output_a * size_a;
	int stride_output_b = stride_output_c * size_c;
	int stride_output_d = stride_output_b * size_b;
	int stride_output_e = stride_output_d * size_d;
	int stride_output_f = stride_output_e * size_e;

	int stride_reg_x_1 = stride_output_b;
	int stride_reg_y_1 = stride_output_e;

	int size_internal = size_g;

	int stride_int_t2 = 1;
    int stride_int_v2 = size_a * size_b * size_d;
    
	// New Caller
	jk_ccsd_t_d1_2_kernel__4_1<<<gridsize_1, blocksize_1>>>(t3_d, dev_t2, dev_v2, size_a, size_c, size_b, size_d, size_e, size_f, size_g, CEIL(size_a, JK_CCSD_T_D1_2_SIZE_SLICE_1_A), CEIL(size_c, JK_CCSD_T_D1_2_SIZE_SLICE_1_C), CEIL(size_b, JK_CCSD_T_D1_2_SIZE_SLICE_1_B), CEIL(size_d, JK_CCSD_T_D1_2_SIZE_SLICE_1_D), CEIL(size_e, JK_CCSD_T_D1_2_SIZE_SLICE_1_E), CEIL(size_f, JK_CCSD_T_D1_2_SIZE_SLICE_1_F), stride_int_t2, stride_int_v2, stride_reg_x_1, stride_reg_y_1, size_internal);

	// Copy the Result from Device to Host
	//cudaMemcpy(t3, dev_t3, sizeof(double) * (size_a * size_c * size_b * size_d * size_e * size_f), cudaMemcpyDeviceToHost);

	// cudaFree()
    //cudaFree(dev_t3);	
    cudaFree(dev_t2);	cudaFree(dev_v2);
}

// This is written by tc_interface.tc_gen_code_interface()
// This Interface Should be Called to Run the Kernels
extern "C"
void jk_ccsd_t_d1_2_fusion_(int size_a, int size_c, int size_b, int size_d, int size_e, int size_f, int size_g, double* t3, double* t2, double* v2, int cond_kernel_1, int opt_register_transpose)
{
	// Call An Application
	jk_ccsd_t_d1_2_fusion(size_a, size_c, size_b, size_d, size_e, size_f, size_g, t3, t2, v2, cond_kernel_1, opt_register_transpose);
}

/*----------------------------------------------------------------------*
 *  [d1][3] triplesx[h1,h3,p5,p4] -= t2sub[h7,p4,p5,h1] * v2sub[h3,h7]
 *----------------------------------------------------------------------*/
// created by tc_gen_definition_new()
#define JK_CCSD_T_D1_3_SIZE_SLICE_1_G   16
#define JK_CCSD_T_D1_3_SIZE_SLICE_1_C   16
#define JK_CCSD_T_D1_3_SIZE_SLICE_1_F   4
#define JK_CCSD_T_D1_3_SIZE_SLICE_1_E   1
#define JK_CCSD_T_D1_3_SIZE_SLICE_1_A   16
#define JK_CCSD_T_D1_3_SIZE_SLICE_1_B   4
#define JK_CCSD_T_D1_3_SIZE_SLICE_1_D   1

#define JK_CCSD_T_D1_3_SIZE_INT_UNIT_1  JK_CCSD_T_D1_3_SIZE_SLICE_1_G

#define JK_CCSD_T_D1_3_SIZE_TB_1_X 	    JK_CCSD_T_D1_3_SIZE_SLICE_1_C * JK_CCSD_T_D1_3_SIZE_SLICE_1_E
#define JK_CCSD_T_D1_3_SIZE_TB_1_Y 	    JK_CCSD_T_D1_3_SIZE_SLICE_1_A * JK_CCSD_T_D1_3_SIZE_SLICE_1_D
#define JK_CCSD_T_D1_3_SIZE_REG_1_X 	JK_CCSD_T_D1_3_SIZE_SLICE_1_F
#define JK_CCSD_T_D1_3_SIZE_REG_1_Y 	JK_CCSD_T_D1_3_SIZE_SLICE_1_B

// created by tc_gen_code_Kernel()
__global__ void jk_ccsd_t_d1_3_kernel__4_1(double* dev_t3, double* dev_t2, double* dev_v2, int size_c, int size_a, int size_b, int size_d, int size_e, int size_f, int size_g, int numBlk_c, int numBlk_a, int numBlk_b, int numBlk_d, int numBlk_e, int numBlk_f, int stride_int_t2, int stride_int_v2, int stride_reg_x, int stride_reg_y, int size_internal)
{
	// For Shared Memory,
	__shared__ double sm_a[16][64];
	__shared__ double sm_b[16][64];


	int internal_upperbound   = 0;
	int internal_offset;

	// when opt_pre_computed == -1, all indices will be calculated manually
	// # of indices mapped on TB_X: 2
	// # of indices mapped on TB_Y: 2
	int idx_c = threadIdx.x % JK_CCSD_T_D1_3_SIZE_SLICE_1_C;
	int idx_e = threadIdx.x / JK_CCSD_T_D1_3_SIZE_SLICE_1_C;
	int idx_a = threadIdx.y % JK_CCSD_T_D1_3_SIZE_SLICE_1_A;
	int idx_d = threadIdx.y / JK_CCSD_T_D1_3_SIZE_SLICE_1_A;

	int tmp_blkIdx;
	int blk_idx_f = blockIdx.x / (numBlk_e * numBlk_d * numBlk_b * numBlk_a * numBlk_c);
	tmp_blkIdx = blockIdx.x % (numBlk_e * numBlk_d * numBlk_b * numBlk_a * numBlk_c);

	int blk_idx_e = tmp_blkIdx / (numBlk_d * numBlk_b * numBlk_a * numBlk_c);
	tmp_blkIdx = tmp_blkIdx % (numBlk_d * numBlk_b * numBlk_a * numBlk_c);

	int blk_idx_d = tmp_blkIdx / (numBlk_b * numBlk_a * numBlk_c);
	tmp_blkIdx = tmp_blkIdx % (numBlk_b * numBlk_a * numBlk_c);

	int blk_idx_b = tmp_blkIdx / (numBlk_a * numBlk_c);
	tmp_blkIdx = tmp_blkIdx % (numBlk_a * numBlk_c);

	int blk_idx_a = tmp_blkIdx / numBlk_c;
	tmp_blkIdx = tmp_blkIdx % (numBlk_c);

	int  blk_idx_c = tmp_blkIdx;

	int t3_base_thread = blk_idx_c * JK_CCSD_T_D1_3_SIZE_SLICE_1_C + idx_c + (blk_idx_a * JK_CCSD_T_D1_3_SIZE_SLICE_1_A + idx_a + (blk_idx_b * JK_CCSD_T_D1_3_SIZE_SLICE_1_B + (blk_idx_d * JK_CCSD_T_D1_3_SIZE_SLICE_1_D + idx_d + (blk_idx_e * JK_CCSD_T_D1_3_SIZE_SLICE_1_E + idx_e + (blk_idx_f * JK_CCSD_T_D1_3_SIZE_SLICE_1_F) * size_e) * size_d) * size_b) * size_a) * size_c;

	// need to support partial tiles
	int rng_c, rng_a, rng_b, rng_d, rng_e, rng_f;
	if ((size_c - (blk_idx_c * JK_CCSD_T_D1_3_SIZE_SLICE_1_C)) >= JK_CCSD_T_D1_3_SIZE_SLICE_1_C)
	{
		rng_c = JK_CCSD_T_D1_3_SIZE_SLICE_1_C;
	}
	else
	{
		rng_c = size_c % JK_CCSD_T_D1_3_SIZE_SLICE_1_C;
	}
	if ((size_a - (blk_idx_a * JK_CCSD_T_D1_3_SIZE_SLICE_1_A)) >= JK_CCSD_T_D1_3_SIZE_SLICE_1_A)
	{
		rng_a = JK_CCSD_T_D1_3_SIZE_SLICE_1_A;
	}
	else
	{
		rng_a = size_a % JK_CCSD_T_D1_3_SIZE_SLICE_1_A;
	}
	if ((size_b - (blk_idx_b * JK_CCSD_T_D1_3_SIZE_SLICE_1_B)) >= JK_CCSD_T_D1_3_SIZE_SLICE_1_B)
	{
		rng_b = JK_CCSD_T_D1_3_SIZE_SLICE_1_B;
	}
	else
	{
		rng_b = size_b % JK_CCSD_T_D1_3_SIZE_SLICE_1_B;
	}
	if ((size_d - (blk_idx_d * JK_CCSD_T_D1_3_SIZE_SLICE_1_D)) >= JK_CCSD_T_D1_3_SIZE_SLICE_1_D)
	{
		rng_d = JK_CCSD_T_D1_3_SIZE_SLICE_1_D;
	}
	else
	{
		rng_d = size_d % JK_CCSD_T_D1_3_SIZE_SLICE_1_D;
	}
	if ((size_e - (blk_idx_e * JK_CCSD_T_D1_3_SIZE_SLICE_1_E)) >= JK_CCSD_T_D1_3_SIZE_SLICE_1_E)
	{
		rng_e = JK_CCSD_T_D1_3_SIZE_SLICE_1_E;
	}
	else
	{
		rng_e = size_e % JK_CCSD_T_D1_3_SIZE_SLICE_1_E;
	}
	if ((size_f - (blk_idx_f * JK_CCSD_T_D1_3_SIZE_SLICE_1_F)) >= JK_CCSD_T_D1_3_SIZE_SLICE_1_F)
	{
		rng_f = JK_CCSD_T_D1_3_SIZE_SLICE_1_F;
	}
	else
	{
		rng_f = size_f % JK_CCSD_T_D1_3_SIZE_SLICE_1_F;
	}

	double temp_av;
	double temp_bv[4];
	double reg_tile[4][4];

	for (int i = 0; i < 4; i++)
	for (int j = 0; j < 4; j++)
	reg_tile[i][j] = 0.0;

	// tensor contraction: [[16, 'STR_SD2_T2_H7', 'x', 't2', ['g', 'f', 'e', 'c']], [16, 'STR_SD2_V2_H7', 'y', 'v2', ['a', 'b', 'd', 'g']], '+=']
	#pragma unroll 1
	for (int l = 0; l < size_internal; l += JK_CCSD_T_D1_3_SIZE_INT_UNIT_1)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + JK_CCSD_T_D1_3_SIZE_INT_UNIT_1) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		//---------------------------------------------------------------------------------------------------
		// This is for the new version
		// This Part is for Loading Input-Left
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (0 < rng_e && idx_a < rng_c && threadIdx.x < JK_CCSD_T_D1_3_SIZE_INT_UNIT_1 - internal_upperbound)
		for (int ll = 0; ll < rng_f; ll++)
		{
			// ['g', 'f', 'e', 'c']
			// Exception: Temp. version!: threadIdx.x + l
			// Exception: Temp. version!: 0 < rng_e
			sm_a[threadIdx.x][threadIdx.y + ll * 16] = dev_t2[(blk_idx_f * JK_CCSD_T_D1_3_SIZE_SLICE_1_F + ll + (blk_idx_e * JK_CCSD_T_D1_3_SIZE_SLICE_1_E + 0 + (blk_idx_c * JK_CCSD_T_D1_3_SIZE_SLICE_1_C + idx_a) * size_e) * size_f) * size_g + (threadIdx.x + l)];
		}
		
		// This Part is for Loading Input-Right
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (idx_c < rng_a && 0 < rng_d && threadIdx.y < JK_CCSD_T_D1_3_SIZE_INT_UNIT_1 - internal_upperbound)
		for (int ll = 0; ll < rng_b; ll++)
		{
			// ['a', 'b', 'd', 'g']
			// Exception: Temp. version!: threadIdx.y + l
			// Exception: Temp. version!: idx_c < rng_a
			sm_b[threadIdx.y][threadIdx.x + ll * 16] = dev_v2[blk_idx_a * JK_CCSD_T_D1_3_SIZE_SLICE_1_A + idx_c + (blk_idx_b * JK_CCSD_T_D1_3_SIZE_SLICE_1_B + ll + (blk_idx_d * JK_CCSD_T_D1_3_SIZE_SLICE_1_D + 0) * size_b) * size_a + (threadIdx.y + l) * stride_int_v2];
		}
		__syncthreads();
		//---------------------------------------------------------------------------------------------------
		

		// Part: Generalized Threads
		for (int ll = 0; ll < JK_CCSD_T_D1_3_SIZE_INT_UNIT_1 - internal_upperbound; ll++)
		{
			temp_bv[0] = sm_b[ll][idx_a + (idx_d) * JK_CCSD_T_D1_3_SIZE_SLICE_1_A + 0];
			temp_bv[1] = sm_b[ll][idx_a + (idx_d) * JK_CCSD_T_D1_3_SIZE_SLICE_1_A + 16];
			temp_bv[2] = sm_b[ll][idx_a + (idx_d) * JK_CCSD_T_D1_3_SIZE_SLICE_1_A + 32];
			temp_bv[3] = sm_b[ll][idx_a + (idx_d) * JK_CCSD_T_D1_3_SIZE_SLICE_1_A + 48];

			for (int xx = 0; xx < 4; xx++) // (1)
			{
				temp_av = sm_a[ll][idx_e + (idx_c) * JK_CCSD_T_D1_3_SIZE_SLICE_1_E + (xx * 16)];

				reg_tile[0][xx] -= temp_av * temp_bv[0];
				reg_tile[1][xx] -= temp_av * temp_bv[1];
				reg_tile[2][xx] -= temp_av * temp_bv[2];
				reg_tile[3][xx] -= temp_av * temp_bv[3];
			}
		}
		__syncthreads();
	}


	// Store Results (Registers) to Global Memory
	// Part: Generalized Threads
	// Part: Generalized Register-Tiling
	if (idx_c < rng_c && idx_e < rng_e && idx_a < rng_a && idx_d < rng_d)
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			if(i < rng_b && j < rng_f)
			{
			    dev_t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] += reg_tile[i][j];
			}
		}
	}
}

// written by tc_interface.tc_gen_code_interface_Header()
extern "C"
void jk_ccsd_t_d1_3_fusion(int size_c, int size_a, int size_b, int size_d, int size_e, int size_f, int size_g, double* t3, double* host_t2, double* host_v2, int cond_kernel_1, int opt_register_transpose)
{
	int num_thread_blocks_kernel_1;

	double* dev_t3;
	double* dev_t2;
	double* dev_v2;

	num_thread_blocks_kernel_1 = CEIL(size_c, JK_CCSD_T_D1_3_SIZE_SLICE_1_C) * CEIL(size_a, JK_CCSD_T_D1_3_SIZE_SLICE_1_A) * CEIL(size_b, JK_CCSD_T_D1_3_SIZE_SLICE_1_B) * CEIL(size_d, JK_CCSD_T_D1_3_SIZE_SLICE_1_D) * CEIL(size_e, JK_CCSD_T_D1_3_SIZE_SLICE_1_E) * CEIL(size_f, JK_CCSD_T_D1_3_SIZE_SLICE_1_F);

    // cudaMalloc()
	//cudaMalloc((void**) &dev_t3, sizeof(double) * size_c * size_a * size_b * size_d * size_e * size_f);
	cudaMalloc((void**) &dev_t2, sizeof(double) * size_c * size_e * size_f * size_g);
	cudaMalloc((void**) &dev_v2, sizeof(double) * size_g * size_d * size_b * size_a);

	// cudaMemcpy()
	//cudaMemcpy(dev_t3, t3, sizeof(double) * size_c * size_a * size_b * size_d * size_e * size_f, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_c * size_e * size_f * size_g, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_g * size_d * size_b * size_a, cudaMemcpyHostToDevice);

	// Related to Kernels
	// There are 1 Basic Kernels
    long long int tmp_operations = 2 * (long long int)(size_c * size_a * size_b * size_d * size_e * size_f) * size_g;
    /*
	printf ("========================================= fusedKernels =============================================\n");
	printf ("		Grid Size  : %6d (1D)\n", num_thread_blocks_kernel_1);
	printf ("		Block-size : %2d, %2d (2D)\n", JK_CCSD_T_D1_3_SIZE_TB_1_X, JK_CCSD_T_D1_3_SIZE_TB_1_Y);
	printf ("		Reg.-size  : %2d, %2d (2D)\n", JK_CCSD_T_D1_3_SIZE_REG_1_X, JK_CCSD_T_D1_3_SIZE_REG_1_Y);
	printf ("		A thread deals with (%d x %d) elements (basically)\n", JK_CCSD_T_D1_3_SIZE_TB_1_X * JK_CCSD_T_D1_3_SIZE_REG_1_X, JK_CCSD_T_D1_3_SIZE_TB_1_Y * JK_CCSD_T_D1_3_SIZE_REG_1_Y);
	printf ("		# of Operations: %lld\n", tmp_operations);
    printf ("====================================================================================================\n");
    */
	dim3 gridsize_1(num_thread_blocks_kernel_1);
	dim3 blocksize_1(JK_CCSD_T_D1_3_SIZE_TB_1_X, JK_CCSD_T_D1_3_SIZE_TB_1_Y);

	int stride_output_c = 1;
	int stride_output_a = stride_output_c * size_c;
	int stride_output_b = stride_output_a * size_a;
	int stride_output_d = stride_output_b * size_b;
	int stride_output_e = stride_output_d * size_d;
	int stride_output_f = stride_output_e * size_e;

	int stride_reg_x_1 = stride_output_f;
	int stride_reg_y_1 = stride_output_b;

	int size_internal = size_g;

	int stride_int_t2 = 1;
	int stride_int_v2 = size_a * size_b * size_d;

    dev_t3 = t3_d;

	// New Caller
	jk_ccsd_t_d1_3_kernel__4_1<<<gridsize_1, blocksize_1>>>(dev_t3, dev_t2, dev_v2, size_c, size_a, size_b, size_d, size_e, size_f, size_g, CEIL(size_c, JK_CCSD_T_D1_3_SIZE_SLICE_1_C), CEIL(size_a, JK_CCSD_T_D1_3_SIZE_SLICE_1_A), CEIL(size_b, JK_CCSD_T_D1_3_SIZE_SLICE_1_B), CEIL(size_d, JK_CCSD_T_D1_3_SIZE_SLICE_1_D), CEIL(size_e, JK_CCSD_T_D1_3_SIZE_SLICE_1_E), CEIL(size_f, JK_CCSD_T_D1_3_SIZE_SLICE_1_F), stride_int_t2, stride_int_v2, stride_reg_x_1, stride_reg_y_1, size_internal);

	// Copy the Result from Device to Host
	//cudaMemcpy(t3, dev_t3, sizeof(double) * (size_c * size_a * size_b * size_d * size_e * size_f), cudaMemcpyDeviceToHost);

	// cudaFree()
    //cudaFree(dev_t3);	
    cudaFree(dev_t2);	cudaFree(dev_v2);
}

// This is written by tc_interface.tc_gen_code_interface()
// This Interface Should be Called to Run the Kernels
extern "C"
void jk_ccsd_t_d1_3_fusion_(int size_c, int size_a, int size_b, int size_d, int size_e, int size_f, int size_g, double* t3, double* t2, double* v2, int cond_kernel_1, int opt_register_transpose)
{
	// Pre-Processing for Split
	// Based on Tile-Sizes and Problem-Size
	// Currently, one index can be split into two indices

	// Call An Application
	jk_ccsd_t_d1_3_fusion(size_c, size_a, size_b, size_d, size_e, size_f, size_g, t3, t2, v2, cond_kernel_1, opt_register_transpose);
}

/*----------------------------------------------------------------------*
 *  [d1][4] triplesx[h3,h1,p5,p4,p6] -= t2sub[h7,p4,p5,h1] * v2sub[h3,p6,h7]
 *----------------------------------------------------------------------*/
// created by tc_gen_definition_new()
#define JK_CCSD_T_D1_4_SIZE_SLICE_1_G   16
#define JK_CCSD_T_D1_4_SIZE_SLICE_1_A   16
#define JK_CCSD_T_D1_4_SIZE_SLICE_1_B   4
#define JK_CCSD_T_D1_4_SIZE_SLICE_1_D   1
#define JK_CCSD_T_D1_4_SIZE_SLICE_1_F   8
#define JK_CCSD_T_D1_4_SIZE_SLICE_1_E   8
#define JK_CCSD_T_D1_4_SIZE_SLICE_1_C   1

#define JK_CCSD_T_D1_4_SIZE_INT_UNIT_1  JK_CCSD_T_D1_4_SIZE_SLICE_1_G

#define JK_CCSD_T_D1_4_SIZE_TB_1_X 	    JK_CCSD_T_D1_4_SIZE_SLICE_1_A * JK_CCSD_T_D1_4_SIZE_SLICE_1_D
#define JK_CCSD_T_D1_4_SIZE_TB_1_Y 	    JK_CCSD_T_D1_4_SIZE_SLICE_1_F * JK_CCSD_T_D1_4_SIZE_SLICE_1_C
#define JK_CCSD_T_D1_4_SIZE_REG_1_X 	JK_CCSD_T_D1_4_SIZE_SLICE_1_B
#define JK_CCSD_T_D1_4_SIZE_REG_1_Y 	JK_CCSD_T_D1_4_SIZE_SLICE_1_E

// created by tc_gen_code_Kernel()
__global__ void jk_ccsd_t_d1_4_kernel__4_1(double* dev_t3, double* dev_t2, double* dev_v2, int size_a, int size_b, int size_c, int size_e, int size_f, int size_d, int size_g, int numBlk_a, int numBlk_b, int numBlk_c, int numBlk_e, int numBlk_f, int numBlk_d, int stride_int_t2, int stride_int_v2, int stride_reg_x, int stride_reg_y, int size_internal)
{
	// For Shared Memory,
	__shared__ double sm_a[16][64];
	__shared__ double sm_b[16][64];

	int internal_upperbound   = 0;
	int internal_offset;

	// when opt_pre_computed == -1, all indices will be calculated manually
	// # of indices mapped on TB_X: 2
	// # of indices mapped on TB_Y: 2
	int idx_a = threadIdx.x % JK_CCSD_T_D1_4_SIZE_SLICE_1_A;
	int idx_d = threadIdx.x / JK_CCSD_T_D1_4_SIZE_SLICE_1_A;
	int idx_f = threadIdx.y % JK_CCSD_T_D1_4_SIZE_SLICE_1_F;
	int idx_c = threadIdx.y / JK_CCSD_T_D1_4_SIZE_SLICE_1_F;

	int tmp_blkIdx;
	int blk_idx_d = blockIdx.x / (numBlk_f * numBlk_e * numBlk_c * numBlk_b * numBlk_a);
	tmp_blkIdx = blockIdx.x % (numBlk_f * numBlk_e * numBlk_c * numBlk_b * numBlk_a);

	int blk_idx_f = tmp_blkIdx / (numBlk_e * numBlk_c * numBlk_b * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_e * numBlk_c * numBlk_b * numBlk_a);

	int blk_idx_e = tmp_blkIdx / (numBlk_c * numBlk_b * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_c * numBlk_b * numBlk_a);

	int blk_idx_c = tmp_blkIdx / (numBlk_b * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_b * numBlk_a);

	int blk_idx_b = tmp_blkIdx / numBlk_a;
	tmp_blkIdx = tmp_blkIdx % (numBlk_a);

	int  blk_idx_a = tmp_blkIdx;

	int t3_base_thread = blk_idx_a * JK_CCSD_T_D1_4_SIZE_SLICE_1_A + idx_a + (blk_idx_b * JK_CCSD_T_D1_4_SIZE_SLICE_1_B + (blk_idx_c * JK_CCSD_T_D1_4_SIZE_SLICE_1_C + idx_c + (blk_idx_e * JK_CCSD_T_D1_4_SIZE_SLICE_1_E + (blk_idx_f * JK_CCSD_T_D1_4_SIZE_SLICE_1_F + idx_f + (blk_idx_d * JK_CCSD_T_D1_4_SIZE_SLICE_1_D + idx_d) * size_f) * size_e) * size_c) * size_b) * size_a;

	// need to support partial tiles
	int rng_a, rng_b, rng_c, rng_e, rng_f, rng_d;
	if ((size_a - (blk_idx_a * JK_CCSD_T_D1_4_SIZE_SLICE_1_A)) >= JK_CCSD_T_D1_4_SIZE_SLICE_1_A)
	{
		rng_a = JK_CCSD_T_D1_4_SIZE_SLICE_1_A;
	}
	else
	{
		rng_a = size_a % JK_CCSD_T_D1_4_SIZE_SLICE_1_A;
	}
	if ((size_b - (blk_idx_b * JK_CCSD_T_D1_4_SIZE_SLICE_1_B)) >= JK_CCSD_T_D1_4_SIZE_SLICE_1_B)
	{
		rng_b = JK_CCSD_T_D1_4_SIZE_SLICE_1_B;
	}
	else
	{
		rng_b = size_b % JK_CCSD_T_D1_4_SIZE_SLICE_1_B;
	}
	if ((size_c - (blk_idx_c * JK_CCSD_T_D1_4_SIZE_SLICE_1_C)) >= JK_CCSD_T_D1_4_SIZE_SLICE_1_C)
	{
		rng_c = JK_CCSD_T_D1_4_SIZE_SLICE_1_C;
	}
	else
	{
		rng_c = size_c % JK_CCSD_T_D1_4_SIZE_SLICE_1_C;
	}
	if ((size_e - (blk_idx_e * JK_CCSD_T_D1_4_SIZE_SLICE_1_E)) >= JK_CCSD_T_D1_4_SIZE_SLICE_1_E)
	{
		rng_e = JK_CCSD_T_D1_4_SIZE_SLICE_1_E;
	}
	else
	{
		rng_e = size_e % JK_CCSD_T_D1_4_SIZE_SLICE_1_E;
	}
	if ((size_f - (blk_idx_f * JK_CCSD_T_D1_4_SIZE_SLICE_1_F)) >= JK_CCSD_T_D1_4_SIZE_SLICE_1_F)
	{
		rng_f = JK_CCSD_T_D1_4_SIZE_SLICE_1_F;
	}
	else
	{
		rng_f = size_f % JK_CCSD_T_D1_4_SIZE_SLICE_1_F;
	}
	if ((size_d - (blk_idx_d * JK_CCSD_T_D1_4_SIZE_SLICE_1_D)) >= JK_CCSD_T_D1_4_SIZE_SLICE_1_D)
	{
		rng_d = JK_CCSD_T_D1_4_SIZE_SLICE_1_D;
	}
	else
	{
		rng_d = size_d % JK_CCSD_T_D1_4_SIZE_SLICE_1_D;
	}

	double temp_av;
	double temp_bv[8];
	double reg_tile[8][4];

	for (int i = 0; i < 8; i++)
	for (int j = 0; j < 4; j++)
	reg_tile[i][j] = 0.0;

	// tensor contraction: [[16, 'STR_SD2_T2_H7', 'y', 't2', ['g', 'f', 'e', 'c']], [16, 'STR_SD2_V2_H7', 'x', 'v2', ['a', 'b', 'd', 'g']], '+=']
	#pragma unroll 1
	for (int l = 0; l < size_internal; l += JK_CCSD_T_D1_4_SIZE_INT_UNIT_1)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + JK_CCSD_T_D1_4_SIZE_INT_UNIT_1) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		//---------------------------------------------------------------------------------------------------
		// This is for the new version
		// This Part is for Loading Input-Left
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (idx_f < rng_f && 0 < rng_c && threadIdx.x < JK_CCSD_T_D1_4_SIZE_INT_UNIT_1 - internal_upperbound)
		for (int ll = 0; ll < rng_e; ll++)
		{
			// ['g', 'f', 'e', 'c']
			// Exception: Temp. version!: threadIdx.x + l
			// Exception: Temp. version!: idx_f < rng_f
			sm_a[threadIdx.x][threadIdx.y + ll * 8] = dev_t2[(blk_idx_f * JK_CCSD_T_D1_4_SIZE_SLICE_1_F + idx_f + (blk_idx_e * JK_CCSD_T_D1_4_SIZE_SLICE_1_E + ll + (blk_idx_c * JK_CCSD_T_D1_4_SIZE_SLICE_1_C + 0) * size_e) * size_f) * size_g + (threadIdx.x + l)];
		}
		
		// This Part is for Loading Input-Right
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (idx_a < rng_a && 0 < rng_d && threadIdx.y < JK_CCSD_T_D1_4_SIZE_INT_UNIT_1 - internal_upperbound)
		for (int ll = 0; ll < rng_b; ll++)
		{
			// ['a', 'b', 'd', 'g']
			// Exception: Temp. version!: threadIdx.y + l + 0
			// Exception: Temp. version!: idx_a < rng_a
			sm_b[threadIdx.y + 0][threadIdx.x + ll * 16] = dev_v2[blk_idx_a * JK_CCSD_T_D1_4_SIZE_SLICE_1_A + idx_a + (blk_idx_b * JK_CCSD_T_D1_4_SIZE_SLICE_1_B + ll + (blk_idx_d * JK_CCSD_T_D1_4_SIZE_SLICE_1_D + 0) * size_b) * size_a + (threadIdx.y + l + 0) * stride_int_v2];
			// Exception: Temp. version!: threadIdx.y + l + 8
			// Exception: Temp. version!: idx_a < rng_a
			if (threadIdx.y + l + 8 < size_internal && idx_a < rng_a) 
			sm_b[threadIdx.y + 8][threadIdx.x + ll * 16] = dev_v2[blk_idx_a * JK_CCSD_T_D1_4_SIZE_SLICE_1_A + idx_a + (blk_idx_b * JK_CCSD_T_D1_4_SIZE_SLICE_1_B + ll + (blk_idx_d * JK_CCSD_T_D1_4_SIZE_SLICE_1_D + 0) * size_b) * size_a + (threadIdx.y + l + 8) * stride_int_v2];
		}
		__syncthreads();
		//---------------------------------------------------------------------------------------------------
		

		// Part: Generalized Threads
		for (int ll = 0; ll < JK_CCSD_T_D1_4_SIZE_INT_UNIT_1 - internal_upperbound; ll++)
		{
			temp_bv[0] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_4_SIZE_SLICE_1_F + 0];
			temp_bv[1] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_4_SIZE_SLICE_1_F + 8];
			temp_bv[2] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_4_SIZE_SLICE_1_F + 16];
			temp_bv[3] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_4_SIZE_SLICE_1_F + 24];
			temp_bv[4] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_4_SIZE_SLICE_1_F + 32];
			temp_bv[5] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_4_SIZE_SLICE_1_F + 40];
			temp_bv[6] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_4_SIZE_SLICE_1_F + 48];
			temp_bv[7] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_4_SIZE_SLICE_1_F + 56];

			for (int xx = 0; xx < 4; xx++) // (1)
			{
				temp_av = sm_b[ll][idx_a + (idx_d) * JK_CCSD_T_D1_4_SIZE_SLICE_1_A + (xx * 16)];

				reg_tile[0][xx] -= temp_av * temp_bv[0];
				reg_tile[1][xx] -= temp_av * temp_bv[1];
				reg_tile[2][xx] -= temp_av * temp_bv[2];
				reg_tile[3][xx] -= temp_av * temp_bv[3];
				reg_tile[4][xx] -= temp_av * temp_bv[4];
				reg_tile[5][xx] -= temp_av * temp_bv[5];
				reg_tile[6][xx] -= temp_av * temp_bv[6];
				reg_tile[7][xx] -= temp_av * temp_bv[7];
			}
		}
		__syncthreads();
	}


	// Store Results (Registers) to Global Memory
	// Part: Generalized Threads
	// Part: Generalized Register-Tiling
	if (idx_a < rng_a && idx_d < rng_d && idx_f < rng_f && idx_c < rng_c)
	for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			if(i < rng_e && j < rng_b)
			{
			    dev_t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] += reg_tile[i][j];
			}
		}
	}
}

// written by tc_interface.tc_gen_code_interface_Header()
extern "C"
void jk_ccsd_t_d1_4_fusion(int size_a, int size_b, int size_c, int size_e, int size_f, int size_d, int size_g, double* t3, double* host_t2, double* host_v2, int cond_kernel_1, int opt_register_transpose)
{
	int num_thread_blocks_kernel_1;

	double* dev_t3;
	double* dev_t2;
	double* dev_v2;

	num_thread_blocks_kernel_1 = CEIL(size_a, JK_CCSD_T_D1_4_SIZE_SLICE_1_A) * CEIL(size_b, JK_CCSD_T_D1_4_SIZE_SLICE_1_B) * CEIL(size_c, JK_CCSD_T_D1_4_SIZE_SLICE_1_C) * CEIL(size_e, JK_CCSD_T_D1_4_SIZE_SLICE_1_E) * CEIL(size_f, JK_CCSD_T_D1_4_SIZE_SLICE_1_F) * CEIL(size_d, JK_CCSD_T_D1_4_SIZE_SLICE_1_D);
    
    // cudaMalloc()
	//cudaMalloc((void**) &dev_t3, sizeof(double) * size_a * size_b * size_c * size_e * size_f * size_d);
	cudaMalloc((void**) &dev_t2, sizeof(double) * size_c * size_e * size_f * size_g);
	cudaMalloc((void**) &dev_v2, sizeof(double) * size_g * size_d * size_b * size_a);

	// cudaMemcpy()
	// cudaMemcpy(dev_t3, t3, sizeof(double) * size_a * size_b * size_c * size_e * size_f * size_d, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_c * size_e * size_f * size_g, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_g * size_d * size_b * size_a, cudaMemcpyHostToDevice);

	// Related to Kernels
	// There are 1 Basic Kernels
    long long int tmp_operations = 2 * (long long int)(size_a * size_b * size_c * size_e * size_f * size_d) * size_g;
    /*
	printf ("========================================= fusedKernels =============================================\n");
	printf ("		Grid Size  : %6d (1D)\n", num_thread_blocks_kernel_1);
	printf ("		Block-size : %2d, %2d (2D)\n", JK_CCSD_T_D1_4_SIZE_TB_1_X, JK_CCSD_T_D1_4_SIZE_TB_1_Y);
	printf ("		Reg.-size  : %2d, %2d (2D)\n", JK_CCSD_T_D1_4_SIZE_REG_1_X, JK_CCSD_T_D1_4_SIZE_REG_1_Y);
	printf ("		A thread deals with (%d x %d) elements (basically)\n", JK_CCSD_T_D1_4_SIZE_TB_1_X * JK_CCSD_T_D1_4_SIZE_REG_1_X, JK_CCSD_T_D1_4_SIZE_TB_1_Y * JK_CCSD_T_D1_4_SIZE_REG_1_Y);
	printf ("		# of Operations: %lld\n", tmp_operations);
    printf ("====================================================================================================\n");
    */
	dim3 gridsize_1(num_thread_blocks_kernel_1);
	dim3 blocksize_1(JK_CCSD_T_D1_4_SIZE_TB_1_X, JK_CCSD_T_D1_4_SIZE_TB_1_Y);

	int stride_output_a = 1;
	int stride_output_b = stride_output_a * size_a;
	int stride_output_c = stride_output_b * size_b;
	int stride_output_e = stride_output_c * size_c;
	int stride_output_f = stride_output_e * size_e;
	int stride_output_d = stride_output_f * size_f;

	int stride_reg_x_1 = stride_output_b;
	int stride_reg_y_1 = stride_output_e;

	int size_internal = size_g;

	int stride_int_t2 = 1;
	int stride_int_v2 = size_a * size_b * size_d;

    dev_t3 = t3_d;
	// New Caller
	jk_ccsd_t_d1_4_kernel__4_1<<<gridsize_1, blocksize_1>>>(dev_t3, dev_t2, dev_v2, size_a, size_b, size_c, size_e, size_f, size_d, size_g, CEIL(size_a, JK_CCSD_T_D1_4_SIZE_SLICE_1_A), CEIL(size_b, JK_CCSD_T_D1_4_SIZE_SLICE_1_B), CEIL(size_c, JK_CCSD_T_D1_4_SIZE_SLICE_1_C), CEIL(size_e, JK_CCSD_T_D1_4_SIZE_SLICE_1_E), CEIL(size_f, JK_CCSD_T_D1_4_SIZE_SLICE_1_F), CEIL(size_d, JK_CCSD_T_D1_4_SIZE_SLICE_1_D), stride_int_t2, stride_int_v2, stride_reg_x_1, stride_reg_y_1, size_internal);

	// Copy the Result from Device to Host
	//cudaMemcpy(t3, dev_t3, sizeof(double) * (size_a * size_b * size_c * size_e * size_f * size_d), cudaMemcpyDeviceToHost);

	// cudaFree()
    //cudaFree(dev_t3);	
    cudaFree(dev_t2);	cudaFree(dev_v2);
}

// This is written by tc_interface.tc_gen_code_interface()
// This Interface Should be Called to Run the Kernels
extern "C"
void jk_ccsd_t_d1_4_fusion_(int size_a, int size_b, int size_c, int size_e, int size_f, int size_d, int size_g, double* t3, double* t2, double* v2, int cond_kernel_1, int opt_register_transpose)
{
	// Call An Application
	jk_ccsd_t_d1_4_fusion(size_a, size_b, size_c, size_e, size_f, size_d, size_g, t3, t2, v2, cond_kernel_1, opt_register_transpose);
}



/*----------------------------------------------------------------------*
 *  [d1][5] triplesx[h3,h1,h2,p5,p4,p6] += t2sub[h7,p4,p5,h1] * v2sub[h3,h2,p6,h7]
 *----------------------------------------------------------------------*/
// created by tc_gen_definition_new()
#define JK_CCSD_T_D1_5_SIZE_SLICE_1_G 16
#define JK_CCSD_T_D1_5_SIZE_SLICE_1_A 16
#define JK_CCSD_T_D1_5_SIZE_SLICE_1_B 4
#define JK_CCSD_T_D1_5_SIZE_SLICE_1_D 1
#define JK_CCSD_T_D1_5_SIZE_SLICE_1_F 8
#define JK_CCSD_T_D1_5_SIZE_SLICE_1_E 8
#define JK_CCSD_T_D1_5_SIZE_SLICE_1_C 1

#define JK_CCSD_T_D1_5_SIZE_INT_UNIT_1 JK_CCSD_T_D1_5_SIZE_SLICE_1_G

#define JK_CCSD_T_D1_5_SIZE_TB_1_X 	JK_CCSD_T_D1_5_SIZE_SLICE_1_A * JK_CCSD_T_D1_5_SIZE_SLICE_1_D
#define JK_CCSD_T_D1_5_SIZE_TB_1_Y 	JK_CCSD_T_D1_5_SIZE_SLICE_1_F * JK_CCSD_T_D1_5_SIZE_SLICE_1_C
#define JK_CCSD_T_D1_5_SIZE_REG_1_X 	JK_CCSD_T_D1_5_SIZE_SLICE_1_B
#define JK_CCSD_T_D1_5_SIZE_REG_1_Y 	JK_CCSD_T_D1_5_SIZE_SLICE_1_E

// created by tc_gen_code_Kernel()
__global__ void jk_ccsd_t_d1_4_kernel__5_1(double* dev_t3, double* dev_t2, double* dev_v2, int size_a, int size_c, int size_b, int size_e, int size_f, int size_d, int size_g, int numBlk_a, int numBlk_c, int numBlk_b, int numBlk_e, int numBlk_f, int numBlk_d, int stride_int_t2, int stride_int_v2, int stride_reg_x, int stride_reg_y, int size_internal)
{
	// For Shared Memory,
	__shared__ double sm_a[16][64];
	__shared__ double sm_b[16][64];

	int internal_upperbound   = 0;
	int internal_offset;

	// when opt_pre_computed == -1, all indices will be calculated manually
	// # of indices mapped on TB_X: 2
	// # of indices mapped on TB_Y: 2
	int idx_a = threadIdx.x % JK_CCSD_T_D1_5_SIZE_SLICE_1_A;
	int idx_d = threadIdx.x / JK_CCSD_T_D1_5_SIZE_SLICE_1_A;
	int idx_f = threadIdx.y % JK_CCSD_T_D1_5_SIZE_SLICE_1_F;
	int idx_c = threadIdx.y / JK_CCSD_T_D1_5_SIZE_SLICE_1_F;

	int tmp_blkIdx;
	int blk_idx_d = blockIdx.x / (numBlk_f * numBlk_e * numBlk_b * numBlk_c * numBlk_a);
	tmp_blkIdx = blockIdx.x % (numBlk_f * numBlk_e * numBlk_b * numBlk_c * numBlk_a);

	int blk_idx_f = tmp_blkIdx / (numBlk_e * numBlk_b * numBlk_c * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_e * numBlk_b * numBlk_c * numBlk_a);

	int blk_idx_e = tmp_blkIdx / (numBlk_b * numBlk_c * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_b * numBlk_c * numBlk_a);

	int blk_idx_b = tmp_blkIdx / (numBlk_c * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_c * numBlk_a);

	int blk_idx_c = tmp_blkIdx / numBlk_a;
	tmp_blkIdx = tmp_blkIdx % (numBlk_a);

	int  blk_idx_a = tmp_blkIdx;

	int t3_base_thread = blk_idx_a * JK_CCSD_T_D1_5_SIZE_SLICE_1_A + idx_a + (blk_idx_c * JK_CCSD_T_D1_5_SIZE_SLICE_1_C + idx_c + (blk_idx_b * JK_CCSD_T_D1_5_SIZE_SLICE_1_B + (blk_idx_e * JK_CCSD_T_D1_5_SIZE_SLICE_1_E + (blk_idx_f * JK_CCSD_T_D1_5_SIZE_SLICE_1_F + idx_f + (blk_idx_d * JK_CCSD_T_D1_5_SIZE_SLICE_1_D + idx_d) * size_f) * size_e) * size_b) * size_c) * size_a;

	// need to support partial tiles
	int rng_a, rng_c, rng_b, rng_e, rng_f, rng_d;
	if ((size_a - (blk_idx_a * JK_CCSD_T_D1_5_SIZE_SLICE_1_A)) >= JK_CCSD_T_D1_5_SIZE_SLICE_1_A)
	{
		rng_a = JK_CCSD_T_D1_5_SIZE_SLICE_1_A;
	}
	else
	{
		rng_a = size_a % JK_CCSD_T_D1_5_SIZE_SLICE_1_A;
	}
	if ((size_c - (blk_idx_c * JK_CCSD_T_D1_5_SIZE_SLICE_1_C)) >= JK_CCSD_T_D1_5_SIZE_SLICE_1_C)
	{
		rng_c = JK_CCSD_T_D1_5_SIZE_SLICE_1_C;
	}
	else
	{
		rng_c = size_c % JK_CCSD_T_D1_5_SIZE_SLICE_1_C;
	}
	if ((size_b - (blk_idx_b * JK_CCSD_T_D1_5_SIZE_SLICE_1_B)) >= JK_CCSD_T_D1_5_SIZE_SLICE_1_B)
	{
		rng_b = JK_CCSD_T_D1_5_SIZE_SLICE_1_B;
	}
	else
	{
		rng_b = size_b % JK_CCSD_T_D1_5_SIZE_SLICE_1_B;
	}
	if ((size_e - (blk_idx_e * JK_CCSD_T_D1_5_SIZE_SLICE_1_E)) >= JK_CCSD_T_D1_5_SIZE_SLICE_1_E)
	{
		rng_e = JK_CCSD_T_D1_5_SIZE_SLICE_1_E;
	}
	else
	{
		rng_e = size_e % JK_CCSD_T_D1_5_SIZE_SLICE_1_E;
	}
	if ((size_f - (blk_idx_f * JK_CCSD_T_D1_5_SIZE_SLICE_1_F)) >= JK_CCSD_T_D1_5_SIZE_SLICE_1_F)
	{
		rng_f = JK_CCSD_T_D1_5_SIZE_SLICE_1_F;
	}
	else
	{
		rng_f = size_f % JK_CCSD_T_D1_5_SIZE_SLICE_1_F;
	}
	if ((size_d - (blk_idx_d * JK_CCSD_T_D1_5_SIZE_SLICE_1_D)) >= JK_CCSD_T_D1_5_SIZE_SLICE_1_D)
	{
		rng_d = JK_CCSD_T_D1_5_SIZE_SLICE_1_D;
	}
	else
	{
		rng_d = size_d % JK_CCSD_T_D1_5_SIZE_SLICE_1_D;
	}

	double temp_av;
	double temp_bv[8];
	double reg_tile[8][4];

	for (int i = 0; i < 8; i++)
	for (int j = 0; j < 4; j++)
	reg_tile[i][j] = 0.0;

	// tensor contraction: [[16, 'STR_SD2_T2_H7', 'y', 't2', ['g', 'f', 'e', 'c']], [16, 'STR_SD2_V2_H7', 'x', 'v2', ['a', 'b', 'd', 'g']], '+=']
	#pragma unroll 1
	for (int l = 0; l < size_internal; l += JK_CCSD_T_D1_5_SIZE_INT_UNIT_1)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + JK_CCSD_T_D1_5_SIZE_INT_UNIT_1) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		//---------------------------------------------------------------------------------------------------
		// This is for the new version
		// This Part is for Loading Input-Left
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (idx_f < rng_f && 0 < rng_c && threadIdx.x < JK_CCSD_T_D1_5_SIZE_INT_UNIT_1 - internal_upperbound)
		for (int ll = 0; ll < rng_e; ll++)
		{
			// ['g', 'f', 'e', 'c']
			// Exception: Temp. version!: threadIdx.x + l
			// Exception: Temp. version!: idx_f < rng_f
			sm_a[threadIdx.x][threadIdx.y + ll * 8] = dev_t2[(blk_idx_f * JK_CCSD_T_D1_5_SIZE_SLICE_1_F + idx_f + (blk_idx_e * JK_CCSD_T_D1_5_SIZE_SLICE_1_E + ll + (blk_idx_c * JK_CCSD_T_D1_5_SIZE_SLICE_1_C + 0) * size_e) * size_f) * size_g + (threadIdx.x + l)];
		}
		
		// This Part is for Loading Input-Right
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (idx_a < rng_a && 0 < rng_d && threadIdx.y < JK_CCSD_T_D1_5_SIZE_INT_UNIT_1 - internal_upperbound)
		for (int ll = 0; ll < rng_b; ll++)
		{
			// ['a', 'b', 'd', 'g']
			// Exception: Temp. version!: threadIdx.y + l + 0
			// Exception: Temp. version!: idx_a < rng_a
			sm_b[threadIdx.y + 0][threadIdx.x + ll * 16] = dev_v2[blk_idx_a * JK_CCSD_T_D1_5_SIZE_SLICE_1_A + idx_a + (blk_idx_b * JK_CCSD_T_D1_5_SIZE_SLICE_1_B + ll + (blk_idx_d * JK_CCSD_T_D1_5_SIZE_SLICE_1_D + 0) * size_b) * size_a + (threadIdx.y + l + 0) * stride_int_v2];
			// Exception: Temp. version!: threadIdx.y + l + 8
			// Exception: Temp. version!: idx_a < rng_a
			if (threadIdx.y + l + 8 < size_internal && idx_a < rng_a) 
            sm_b[threadIdx.y + 8][threadIdx.x + ll * 16] = dev_v2[blk_idx_a * JK_CCSD_T_D1_5_SIZE_SLICE_1_A + idx_a + 
                                                                 (blk_idx_b * JK_CCSD_T_D1_5_SIZE_SLICE_1_B + ll + 
                                                                 (blk_idx_d * JK_CCSD_T_D1_5_SIZE_SLICE_1_D + 0) * size_b) * size_a + (threadIdx.y + l + 8) * stride_int_v2];
		}
		__syncthreads();
		//---------------------------------------------------------------------------------------------------
		

		// Part: Generalized Threads
		for (int ll = 0; ll < JK_CCSD_T_D1_5_SIZE_INT_UNIT_1 - internal_upperbound; ll++)
		{
			temp_bv[0] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_5_SIZE_SLICE_1_F + 0];
			temp_bv[1] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_5_SIZE_SLICE_1_F + 8];
			temp_bv[2] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_5_SIZE_SLICE_1_F + 16];
			temp_bv[3] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_5_SIZE_SLICE_1_F + 24];
			temp_bv[4] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_5_SIZE_SLICE_1_F + 32];
			temp_bv[5] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_5_SIZE_SLICE_1_F + 40];
			temp_bv[6] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_5_SIZE_SLICE_1_F + 48];
			temp_bv[7] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_5_SIZE_SLICE_1_F + 56];

			for (int xx = 0; xx < 4; xx++) // (1)
			{
				temp_av = sm_b[ll][idx_a + (idx_d) * JK_CCSD_T_D1_5_SIZE_SLICE_1_A + (xx * 16)];

				reg_tile[0][xx] += temp_av * temp_bv[0];
				reg_tile[1][xx] += temp_av * temp_bv[1];
				reg_tile[2][xx] += temp_av * temp_bv[2];
				reg_tile[3][xx] += temp_av * temp_bv[3];
				reg_tile[4][xx] += temp_av * temp_bv[4];
				reg_tile[5][xx] += temp_av * temp_bv[5];
				reg_tile[6][xx] += temp_av * temp_bv[6];
				reg_tile[7][xx] += temp_av * temp_bv[7];
			}
		}
		__syncthreads();
	}


	// Store Results (Registers) to Global Memory
	// Part: Generalized Threads
	// Part: Generalized Register-Tiling
	if (idx_a < rng_a && idx_d < rng_d && idx_f < rng_f && idx_c < rng_c)
	for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			if(i < rng_e && j < rng_b)
			{
			    dev_t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] += reg_tile[i][j];
			}
		}
	}
}

// written by tc_interface.tc_gen_code_interface_Header()
extern "C"
void jk_ccsd_t_d1_5_fusion(int size_a, int size_c, int size_b, int size_e, int size_f, int size_d, int size_g, double* t3, double* host_t2, double* host_v2, int cond_kernel_1, int opt_register_transpose)
{
	int num_thread_blocks_kernel_1;

	// double* dev_t3;
	double* dev_t2;
	double* dev_v2;

	num_thread_blocks_kernel_1 = CEIL(size_a, JK_CCSD_T_D1_5_SIZE_SLICE_1_A) * CEIL(size_c, JK_CCSD_T_D1_5_SIZE_SLICE_1_C) * CEIL(size_b, JK_CCSD_T_D1_5_SIZE_SLICE_1_B) * CEIL(size_e, JK_CCSD_T_D1_5_SIZE_SLICE_1_E) * CEIL(size_f, JK_CCSD_T_D1_5_SIZE_SLICE_1_F) * CEIL(size_d, JK_CCSD_T_D1_5_SIZE_SLICE_1_D);
    
    // cudaMalloc()
	// cudaMalloc((void**) &dev_t3, sizeof(double) * size_a * size_c * size_b * size_e * size_f * size_d);
	cudaMalloc((void**) &dev_t2, sizeof(double) * size_c * size_e * size_f * size_g);
	cudaMalloc((void**) &dev_v2, sizeof(double) * size_g * size_d * size_b * size_a);

	// cudaMemcpy()
	// cudaMemcpy(dev_t3, t3, sizeof(double) * size_a * size_c * size_b * size_e * size_f * size_d, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_c * size_e * size_f * size_g, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_g * size_d * size_b * size_a, cudaMemcpyHostToDevice);

	// Related to Kernels
	// There are 1 Basic Kernels
    long long int tmp_operations = 2 * (long long int)(size_a * size_c * size_b * size_e * size_f * size_d) * size_g;
    /*
	printf ("========================================= fusedKernels =============================================\n");
	printf ("		Grid Size  : %6d (1D)\n", num_thread_blocks_kernel_1);
	printf ("		Block-size : %2d, %2d (2D)\n", JK_CCSD_T_D1_5_SIZE_TB_1_X, JK_CCSD_T_D1_5_SIZE_TB_1_Y);
	printf ("		Reg.-size  : %2d, %2d (2D)\n", JK_CCSD_T_D1_5_SIZE_REG_1_X, JK_CCSD_T_D1_5_SIZE_REG_1_Y);
	printf ("		A thread deals with (%d x %d) elements (basically)\n", JK_CCSD_T_D1_5_SIZE_TB_1_X * JK_CCSD_T_D1_5_SIZE_REG_1_X, JK_CCSD_T_D1_5_SIZE_TB_1_Y * JK_CCSD_T_D1_5_SIZE_REG_1_Y);
	printf ("		# of Operations: %lld\n", tmp_operations);
    printf ("====================================================================================================\n");
    */
	dim3 gridsize_1(num_thread_blocks_kernel_1);
	dim3 blocksize_1(JK_CCSD_T_D1_5_SIZE_TB_1_X, JK_CCSD_T_D1_5_SIZE_TB_1_Y);

	int stride_output_a = 1;
	int stride_output_c = stride_output_a * size_a;
	int stride_output_b = stride_output_c * size_c;
	int stride_output_e = stride_output_b * size_b;
	int stride_output_f = stride_output_e * size_e;
	int stride_output_d = stride_output_f * size_f;

	int stride_reg_x_1 = stride_output_b;
	int stride_reg_y_1 = stride_output_e;

	int size_internal = size_g;

	int stride_int_t2 = 1;
	int stride_int_v2 = size_a * size_b * size_d;

	// New Caller
	jk_ccsd_t_d1_4_kernel__5_1<<<gridsize_1, blocksize_1>>>(t3_d, dev_t2, dev_v2, size_a, size_c, size_b, size_e, size_f, size_d, size_g, CEIL(size_a, JK_CCSD_T_D1_5_SIZE_SLICE_1_A), CEIL(size_c, JK_CCSD_T_D1_5_SIZE_SLICE_1_C), CEIL(size_b, JK_CCSD_T_D1_5_SIZE_SLICE_1_B), CEIL(size_e, JK_CCSD_T_D1_5_SIZE_SLICE_1_E), CEIL(size_f, JK_CCSD_T_D1_5_SIZE_SLICE_1_F), CEIL(size_d, JK_CCSD_T_D1_5_SIZE_SLICE_1_D), stride_int_t2, stride_int_v2, stride_reg_x_1, stride_reg_y_1, size_internal);

	// Copy the Result from Device to Host
	// cudaMemcpy(t3, dev_t3, sizeof(double) * (size_a * size_c * size_b * size_e * size_f * size_d), cudaMemcpyDeviceToHost);

	// cudaFree()
    // cudaFree(dev_t3);	
    cudaFree(dev_t2);	cudaFree(dev_v2);

	// Shoule be Fixed
	// HostFree

}

// This is written by tc_interface.tc_gen_code_interface()
// This Interface Should be Called to Run the Kernels
extern "C"
void jk_ccsd_t_d1_5_fusion_(int size_a, int size_c, int size_b, int size_e, int size_f, int size_d, int size_g, double* t3, double* t2, double* v2, int cond_kernel_1, int opt_register_transpose)
{
	// Pre-Processing for Split
	// Based on Tile-Sizes and Problem-Size
	// Currently, one index can be split into two indices

	// Call An Application
	jk_ccsd_t_d1_5_fusion(size_a, size_c, size_b, size_e, size_f, size_d, size_g, t3, t2, v2, cond_kernel_1, opt_register_transpose);
}




/*----------------------------------------------------------------------*
 *  [d1][6] triplesx[h1,h3,p5,p4,p6] -= t2sub[h7,p4,p5,h1] * v2sub[h3,p6,h7]
 *----------------------------------------------------------------------*/
// created by tc_gen_definition_new()
#define JK_CCSD_T_D1_6_SIZE_SLICE_1_G 16
#define JK_CCSD_T_D1_6_SIZE_SLICE_1_C 16
#define JK_CCSD_T_D1_6_SIZE_SLICE_1_F 4
#define JK_CCSD_T_D1_6_SIZE_SLICE_1_E 1
#define JK_CCSD_T_D1_6_SIZE_SLICE_1_A 16
#define JK_CCSD_T_D1_6_SIZE_SLICE_1_B 4
#define JK_CCSD_T_D1_6_SIZE_SLICE_1_D 1

#define JK_CCSD_T_D1_6_SIZE_INT_UNIT_1 JK_CCSD_T_D1_6_SIZE_SLICE_1_G

#define JK_CCSD_T_D1_6_SIZE_TB_1_X 	JK_CCSD_T_D1_6_SIZE_SLICE_1_C * JK_CCSD_T_D1_6_SIZE_SLICE_1_E
#define JK_CCSD_T_D1_6_SIZE_TB_1_Y 	JK_CCSD_T_D1_6_SIZE_SLICE_1_A * JK_CCSD_T_D1_6_SIZE_SLICE_1_D
#define JK_CCSD_T_D1_6_SIZE_REG_1_X 	JK_CCSD_T_D1_6_SIZE_SLICE_1_F
#define JK_CCSD_T_D1_6_SIZE_REG_1_Y 	JK_CCSD_T_D1_6_SIZE_SLICE_1_B

#define NUM_INDEX 		6
#define CEIL(a, b) 		(((a) + (b) - 1) / (b))

// created by tc_gen_code_Kernel()
__global__ void jk_ccsd_t_d1_6_kernel__4_1(double* dev_t3, double* dev_t2, double* dev_v2, int size_c, int size_a, int size_b, int size_e, int size_f, int size_d, int size_g, int numBlk_c, int numBlk_a, int numBlk_b, int numBlk_e, int numBlk_f, int numBlk_d, int stride_int_t2, int stride_int_v2, int stride_reg_x, int stride_reg_y, int size_internal)
{
	// For Shared Memory,
	__shared__ double sm_a[16][64];
	__shared__ double sm_b[16][64];

	int internal_upperbound   = 0;
	int internal_offset;

	// when opt_pre_computed == -1, all indices will be calculated manually
	// # of indices mapped on TB_X: 2
	// # of indices mapped on TB_Y: 2
	int idx_c = threadIdx.x % JK_CCSD_T_D1_6_SIZE_SLICE_1_C;
	int idx_e = threadIdx.x / JK_CCSD_T_D1_6_SIZE_SLICE_1_C;
	int idx_a = threadIdx.y % JK_CCSD_T_D1_6_SIZE_SLICE_1_A;
	int idx_d = threadIdx.y / JK_CCSD_T_D1_6_SIZE_SLICE_1_A;

	int tmp_blkIdx;
	int blk_idx_d = blockIdx.x / (numBlk_f * numBlk_e * numBlk_b * numBlk_a * numBlk_c);
	tmp_blkIdx = blockIdx.x % (numBlk_f * numBlk_e * numBlk_b * numBlk_a * numBlk_c);

	int blk_idx_f = tmp_blkIdx / (numBlk_e * numBlk_b * numBlk_a * numBlk_c);
	tmp_blkIdx = tmp_blkIdx % (numBlk_e * numBlk_b * numBlk_a * numBlk_c);

	int blk_idx_e = tmp_blkIdx / (numBlk_b * numBlk_a * numBlk_c);
	tmp_blkIdx = tmp_blkIdx % (numBlk_b * numBlk_a * numBlk_c);

	int blk_idx_b = tmp_blkIdx / (numBlk_a * numBlk_c);
	tmp_blkIdx = tmp_blkIdx % (numBlk_a * numBlk_c);

	int blk_idx_a = tmp_blkIdx / numBlk_c;
	tmp_blkIdx = tmp_blkIdx % (numBlk_c);

	int  blk_idx_c = tmp_blkIdx;

	int t3_base_thread = blk_idx_c * JK_CCSD_T_D1_6_SIZE_SLICE_1_C + idx_c + (blk_idx_a * JK_CCSD_T_D1_6_SIZE_SLICE_1_A + idx_a + (blk_idx_b * JK_CCSD_T_D1_6_SIZE_SLICE_1_B + (blk_idx_e * JK_CCSD_T_D1_6_SIZE_SLICE_1_E + idx_e + (blk_idx_f * JK_CCSD_T_D1_6_SIZE_SLICE_1_F + (blk_idx_d * JK_CCSD_T_D1_6_SIZE_SLICE_1_D + idx_d) * size_f) * size_e) * size_b) * size_a) * size_c;

	// need to support partial tiles
	int rng_c, rng_a, rng_b, rng_e, rng_f, rng_d;
	if ((size_c - (blk_idx_c * JK_CCSD_T_D1_6_SIZE_SLICE_1_C)) >= JK_CCSD_T_D1_6_SIZE_SLICE_1_C)
	{
		rng_c = JK_CCSD_T_D1_6_SIZE_SLICE_1_C;
	}
	else
	{
		rng_c = size_c % JK_CCSD_T_D1_6_SIZE_SLICE_1_C;
	}
	if ((size_a - (blk_idx_a * JK_CCSD_T_D1_6_SIZE_SLICE_1_A)) >= JK_CCSD_T_D1_6_SIZE_SLICE_1_A)
	{
		rng_a = JK_CCSD_T_D1_6_SIZE_SLICE_1_A;
	}
	else
	{
		rng_a = size_a % JK_CCSD_T_D1_6_SIZE_SLICE_1_A;
	}
	if ((size_b - (blk_idx_b * JK_CCSD_T_D1_6_SIZE_SLICE_1_B)) >= JK_CCSD_T_D1_6_SIZE_SLICE_1_B)
	{
		rng_b = JK_CCSD_T_D1_6_SIZE_SLICE_1_B;
	}
	else
	{
		rng_b = size_b % JK_CCSD_T_D1_6_SIZE_SLICE_1_B;
	}
	if ((size_e - (blk_idx_e * JK_CCSD_T_D1_6_SIZE_SLICE_1_E)) >= JK_CCSD_T_D1_6_SIZE_SLICE_1_E)
	{
		rng_e = JK_CCSD_T_D1_6_SIZE_SLICE_1_E;
	}
	else
	{
		rng_e = size_e % JK_CCSD_T_D1_6_SIZE_SLICE_1_E;
	}
	if ((size_f - (blk_idx_f * JK_CCSD_T_D1_6_SIZE_SLICE_1_F)) >= JK_CCSD_T_D1_6_SIZE_SLICE_1_F)
	{
		rng_f = JK_CCSD_T_D1_6_SIZE_SLICE_1_F;
	}
	else
	{
		rng_f = size_f % JK_CCSD_T_D1_6_SIZE_SLICE_1_F;
	}
	if ((size_d - (blk_idx_d * JK_CCSD_T_D1_6_SIZE_SLICE_1_D)) >= JK_CCSD_T_D1_6_SIZE_SLICE_1_D)
	{
		rng_d = JK_CCSD_T_D1_6_SIZE_SLICE_1_D;
	}
	else
	{
		rng_d = size_d % JK_CCSD_T_D1_6_SIZE_SLICE_1_D;
	}

	double temp_av;
	double temp_bv[4];
	double reg_tile[4][4];

	for (int i = 0; i < 4; i++)
	for (int j = 0; j < 4; j++)
	reg_tile[i][j] = 0.0;

	// tensor contraction: [[16, 'STR_SD2_T2_H7', 'x', 't2', ['g', 'f', 'e', 'c']], [16, 'STR_SD2_V2_H7', 'y', 'v2', ['a', 'b', 'd', 'g']], '+=']
	#pragma unroll 1
	for (int l = 0; l < size_internal; l += JK_CCSD_T_D1_6_SIZE_INT_UNIT_1)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + JK_CCSD_T_D1_6_SIZE_INT_UNIT_1) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		//---------------------------------------------------------------------------------------------------
		// This is for the new version
		// This Part is for Loading Input-Left
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (0 < rng_e && idx_a < rng_c && threadIdx.x < JK_CCSD_T_D1_6_SIZE_INT_UNIT_1 - internal_upperbound)
		for (int ll = 0; ll < rng_f; ll++)
		{
			// ['g', 'f', 'e', 'c']
			// Exception: Temp. version!: threadIdx.x + l
			// Exception: Temp. version!: 0 < rng_e
			sm_a[threadIdx.x][threadIdx.y + ll * 16] = dev_t2[(blk_idx_f * JK_CCSD_T_D1_6_SIZE_SLICE_1_F + ll + (blk_idx_e * JK_CCSD_T_D1_6_SIZE_SLICE_1_E + 0 + (blk_idx_c * JK_CCSD_T_D1_6_SIZE_SLICE_1_C + idx_a) * size_e) * size_f) * size_g + (threadIdx.x + l)];
		}
		
		// This Part is for Loading Input-Right
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (idx_c < rng_a && 0 < rng_d && threadIdx.y < JK_CCSD_T_D1_6_SIZE_INT_UNIT_1 - internal_upperbound)
		for (int ll = 0; ll < rng_b; ll++)
		{
			// ['a', 'b', 'd', 'g']
			// Exception: Temp. version!: threadIdx.y + l
			// Exception: Temp. version!: idx_c < rng_a
			sm_b[threadIdx.y][threadIdx.x + ll * 16] = dev_v2[blk_idx_a * JK_CCSD_T_D1_6_SIZE_SLICE_1_A + idx_c + (blk_idx_b * JK_CCSD_T_D1_6_SIZE_SLICE_1_B + ll + (blk_idx_d * JK_CCSD_T_D1_6_SIZE_SLICE_1_D + 0) * size_b) * size_a + (threadIdx.y + l) * stride_int_v2];
		}
		__syncthreads();
		//---------------------------------------------------------------------------------------------------
		

		// Part: Generalized Threads
		for (int ll = 0; ll < JK_CCSD_T_D1_6_SIZE_INT_UNIT_1 - internal_upperbound; ll++)
		{
			temp_bv[0] = sm_b[ll][idx_a + (idx_d) * JK_CCSD_T_D1_6_SIZE_SLICE_1_A + 0];
			temp_bv[1] = sm_b[ll][idx_a + (idx_d) * JK_CCSD_T_D1_6_SIZE_SLICE_1_A + 16];
			temp_bv[2] = sm_b[ll][idx_a + (idx_d) * JK_CCSD_T_D1_6_SIZE_SLICE_1_A + 32];
			temp_bv[3] = sm_b[ll][idx_a + (idx_d) * JK_CCSD_T_D1_6_SIZE_SLICE_1_A + 48];

			for (int xx = 0; xx < 4; xx++) // (1)
			{
				temp_av = sm_a[ll][idx_e + (idx_c) * JK_CCSD_T_D1_6_SIZE_SLICE_1_E + (xx * 16)];

				reg_tile[0][xx] -= temp_av * temp_bv[0];
				reg_tile[1][xx] -= temp_av * temp_bv[1];
				reg_tile[2][xx] -= temp_av * temp_bv[2];
				reg_tile[3][xx] -= temp_av * temp_bv[3];
			}
		}
		__syncthreads();
	}


	// Store Results (Registers) to Global Memory
	// Part: Generalized Threads
	// Part: Generalized Register-Tiling
	if (idx_c < rng_c && idx_e < rng_e && idx_a < rng_a && idx_d < rng_d)
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			if(i < rng_b && j < rng_f)
			{
			    dev_t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] += reg_tile[i][j];
			}
		}
	}
}

// written by tc_interface.tc_gen_code_interface_Header()
extern "C"
void jk_ccsd_t_d1_6_fusion(int size_c, int size_a, int size_b, int size_e, int size_f, int size_d, int size_g, double* t3, double* host_t2, double* host_v2, int cond_kernel_1, int opt_register_transpose)
{
	int num_thread_blocks_kernel_1;

	double* dev_t3;
	double* dev_t2;
	double* dev_v2;

	num_thread_blocks_kernel_1 = CEIL(size_c, JK_CCSD_T_D1_6_SIZE_SLICE_1_C) * CEIL(size_a, JK_CCSD_T_D1_6_SIZE_SLICE_1_A) * CEIL(size_b, JK_CCSD_T_D1_6_SIZE_SLICE_1_B) * CEIL(size_e, JK_CCSD_T_D1_6_SIZE_SLICE_1_E) * CEIL(size_f, JK_CCSD_T_D1_6_SIZE_SLICE_1_F) * CEIL(size_d, JK_CCSD_T_D1_6_SIZE_SLICE_1_D);

    // cudaMalloc()
	// cudaMalloc((void**) &dev_t3, sizeof(double) * size_c * size_a * size_b * size_e * size_f * size_d);
	cudaMalloc((void**) &dev_t2, sizeof(double) * size_c * size_e * size_f * size_g);
	cudaMalloc((void**) &dev_v2, sizeof(double) * size_g * size_d * size_b * size_a);

	// cudaMemcpy()
	// cudaMemcpy(dev_t3, t3, sizeof(double) * size_c * size_a * size_b * size_e * size_f * size_d, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_c * size_e * size_f * size_g, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_g * size_d * size_b * size_a, cudaMemcpyHostToDevice);

	// Related to Kernels
	// There are 1 Basic Kernels
    long long int tmp_operations = 2 * (long long int)(size_c * size_a * size_b * size_e * size_f * size_d) * size_g;
    /*
	printf ("========================================= fusedKernels =============================================\n");
	printf ("		Grid Size  : %6d (1D)\n", num_thread_blocks_kernel_1);
	printf ("		Block-size : %2d, %2d (2D)\n", JK_CCSD_T_D1_6_SIZE_TB_1_X, JK_CCSD_T_D1_6_SIZE_TB_1_Y);
	printf ("		Reg.-size  : %2d, %2d (2D)\n", JK_CCSD_T_D1_6_SIZE_REG_1_X, JK_CCSD_T_D1_6_SIZE_REG_1_Y);
	printf ("		A thread deals with (%d x %d) elements (basically)\n", JK_CCSD_T_D1_6_SIZE_TB_1_X * JK_CCSD_T_D1_6_SIZE_REG_1_X, JK_CCSD_T_D1_6_SIZE_TB_1_Y * JK_CCSD_T_D1_6_SIZE_REG_1_Y);
	printf ("		# of Operations: %lld\n", tmp_operations);
    printf ("====================================================================================================\n");
    */
	dim3 gridsize_1(num_thread_blocks_kernel_1);
	dim3 blocksize_1(JK_CCSD_T_D1_6_SIZE_TB_1_X, JK_CCSD_T_D1_6_SIZE_TB_1_Y);

	int stride_output_c = 1;
	int stride_output_a = stride_output_c * size_c;
	int stride_output_b = stride_output_a * size_a;
	int stride_output_e = stride_output_b * size_b;
	int stride_output_f = stride_output_e * size_e;
	int stride_output_d = stride_output_f * size_f;

	int stride_reg_x_1 = stride_output_f;
	int stride_reg_y_1 = stride_output_b;

	int size_internal = size_g;

	int stride_int_t2 = 1;
	int stride_int_v2 = size_a * size_b * size_d;

    dev_t3 = t3_d;

	// New Caller
	jk_ccsd_t_d1_6_kernel__4_1<<<gridsize_1, blocksize_1>>>(dev_t3, dev_t2, dev_v2, size_c, size_a, size_b, size_e, size_f, size_d, size_g, CEIL(size_c, JK_CCSD_T_D1_6_SIZE_SLICE_1_C), CEIL(size_a, JK_CCSD_T_D1_6_SIZE_SLICE_1_A), CEIL(size_b, JK_CCSD_T_D1_6_SIZE_SLICE_1_B), CEIL(size_e, JK_CCSD_T_D1_6_SIZE_SLICE_1_E), CEIL(size_f, JK_CCSD_T_D1_6_SIZE_SLICE_1_F), CEIL(size_d, JK_CCSD_T_D1_6_SIZE_SLICE_1_D), stride_int_t2, stride_int_v2, stride_reg_x_1, stride_reg_y_1, size_internal);

	// Copy the Result from Device to Host
	// cudaMemcpy(t3, dev_t3, sizeof(double) * (size_c * size_a * size_b * size_e * size_f * size_d), cudaMemcpyDeviceToHost);

	// cudaFree()
    // cudaFree(dev_t3);	
    cudaFree(dev_t2);	cudaFree(dev_v2);
}

// This is written by tc_interface.tc_gen_code_interface()
// This Interface Should be Called to Run the Kernels
extern "C"
void jk_ccsd_t_d1_6_fusion_(int size_c, int size_a, int size_b, int size_e, int size_f, int size_d, int size_g, double* t3, double* t2, double* v2, int cond_kernel_1, int opt_register_transpose)
{
	// Call An Application
	jk_ccsd_t_d1_6_fusion(size_c, size_a, size_b, size_e, size_f, size_d, size_g, t3, t2, v2, cond_kernel_1, opt_register_transpose);
}


/*----------------------------------------------------------------------*
 *  [d1][7] triplesx[h3,h1,p5,p6,p4] += t2sub[h7,p4,p5,h1] * v2sub[h3,p6,h7]
 *----------------------------------------------------------------------*/
// created by tc_gen_definition_new()
#define JK_CCSD_T_D1_7_SIZE_SLICE_1_G 16
#define JK_CCSD_T_D1_7_SIZE_SLICE_1_A 16
#define JK_CCSD_T_D1_7_SIZE_SLICE_1_B 4
#define JK_CCSD_T_D1_7_SIZE_SLICE_1_D 1
#define JK_CCSD_T_D1_7_SIZE_SLICE_1_F 8
#define JK_CCSD_T_D1_7_SIZE_SLICE_1_E 8
#define JK_CCSD_T_D1_7_SIZE_SLICE_1_C 1

#define JK_CCSD_T_D1_7_SIZE_INT_UNIT_1 JK_CCSD_T_D1_7_SIZE_SLICE_1_G

#define JK_CCSD_T_D1_7_SIZE_TB_1_X 	JK_CCSD_T_D1_7_SIZE_SLICE_1_A * JK_CCSD_T_D1_7_SIZE_SLICE_1_D
#define JK_CCSD_T_D1_7_SIZE_TB_1_Y 	JK_CCSD_T_D1_7_SIZE_SLICE_1_F * JK_CCSD_T_D1_7_SIZE_SLICE_1_C
#define JK_CCSD_T_D1_7_SIZE_REG_1_X 	JK_CCSD_T_D1_7_SIZE_SLICE_1_B
#define JK_CCSD_T_D1_7_SIZE_REG_1_Y 	JK_CCSD_T_D1_7_SIZE_SLICE_1_E

#define NUM_INDEX 		6
#define CEIL(a, b) 		(((a) + (b) - 1) / (b))

// created by tc_gen_code_Kernel()
__global__ void jk_ccsd_t_d1_7_kernel__4_1(double* dev_t3, double* dev_t2, double* dev_v2, int size_a, int size_b, int size_c, int size_e, int size_d, int size_f, int size_g, int numBlk_a, int numBlk_b, int numBlk_c, int numBlk_e, int numBlk_d, int numBlk_f, int stride_int_t2, int stride_int_v2, int stride_reg_x, int stride_reg_y, int size_internal)
{
	// For Shared Memory,
	__shared__ double sm_a[16][64];
	__shared__ double sm_b[16][64];


	int internal_upperbound   = 0;
	int internal_offset;

	// when opt_pre_computed == -1, all indices will be calculated manually
	// # of indices mapped on TB_X: 2
	// # of indices mapped on TB_Y: 2
	int idx_a = threadIdx.x % JK_CCSD_T_D1_7_SIZE_SLICE_1_A;
	int idx_d = threadIdx.x / JK_CCSD_T_D1_7_SIZE_SLICE_1_A;
	int idx_f = threadIdx.y % JK_CCSD_T_D1_7_SIZE_SLICE_1_F;
	int idx_c = threadIdx.y / JK_CCSD_T_D1_7_SIZE_SLICE_1_F;

	int tmp_blkIdx;
	int blk_idx_f = blockIdx.x / (numBlk_d * numBlk_e * numBlk_c * numBlk_b * numBlk_a);
	tmp_blkIdx = blockIdx.x % (numBlk_d * numBlk_e * numBlk_c * numBlk_b * numBlk_a);

	int blk_idx_d = tmp_blkIdx / (numBlk_e * numBlk_c * numBlk_b * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_e * numBlk_c * numBlk_b * numBlk_a);

	int blk_idx_e = tmp_blkIdx / (numBlk_c * numBlk_b * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_c * numBlk_b * numBlk_a);

	int blk_idx_c = tmp_blkIdx / (numBlk_b * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_b * numBlk_a);

	int blk_idx_b = tmp_blkIdx / numBlk_a;
	tmp_blkIdx = tmp_blkIdx % (numBlk_a);

	int  blk_idx_a = tmp_blkIdx;

	int t3_base_thread = blk_idx_a * JK_CCSD_T_D1_7_SIZE_SLICE_1_A + idx_a + (blk_idx_b * JK_CCSD_T_D1_7_SIZE_SLICE_1_B + (blk_idx_c * JK_CCSD_T_D1_7_SIZE_SLICE_1_C + idx_c + (blk_idx_e * JK_CCSD_T_D1_7_SIZE_SLICE_1_E + (blk_idx_d * JK_CCSD_T_D1_7_SIZE_SLICE_1_D + idx_d + (blk_idx_f * JK_CCSD_T_D1_7_SIZE_SLICE_1_F + idx_f) * size_d) * size_e) * size_c) * size_b) * size_a;

	// need to support partial tiles
	int rng_a, rng_b, rng_c, rng_e, rng_d, rng_f;
	if ((size_a - (blk_idx_a * JK_CCSD_T_D1_7_SIZE_SLICE_1_A)) >= JK_CCSD_T_D1_7_SIZE_SLICE_1_A)
	{
		rng_a = JK_CCSD_T_D1_7_SIZE_SLICE_1_A;
	}
	else
	{
		rng_a = size_a % JK_CCSD_T_D1_7_SIZE_SLICE_1_A;
	}
	if ((size_b - (blk_idx_b * JK_CCSD_T_D1_7_SIZE_SLICE_1_B)) >= JK_CCSD_T_D1_7_SIZE_SLICE_1_B)
	{
		rng_b = JK_CCSD_T_D1_7_SIZE_SLICE_1_B;
	}
	else
	{
		rng_b = size_b % JK_CCSD_T_D1_7_SIZE_SLICE_1_B;
	}
	if ((size_c - (blk_idx_c * JK_CCSD_T_D1_7_SIZE_SLICE_1_C)) >= JK_CCSD_T_D1_7_SIZE_SLICE_1_C)
	{
		rng_c = JK_CCSD_T_D1_7_SIZE_SLICE_1_C;
	}
	else
	{
		rng_c = size_c % JK_CCSD_T_D1_7_SIZE_SLICE_1_C;
	}
	if ((size_e - (blk_idx_e * JK_CCSD_T_D1_7_SIZE_SLICE_1_E)) >= JK_CCSD_T_D1_7_SIZE_SLICE_1_E)
	{
		rng_e = JK_CCSD_T_D1_7_SIZE_SLICE_1_E;
	}
	else
	{
		rng_e = size_e % JK_CCSD_T_D1_7_SIZE_SLICE_1_E;
	}
	if ((size_d - (blk_idx_d * JK_CCSD_T_D1_7_SIZE_SLICE_1_D)) >= JK_CCSD_T_D1_7_SIZE_SLICE_1_D)
	{
		rng_d = JK_CCSD_T_D1_7_SIZE_SLICE_1_D;
	}
	else
	{
		rng_d = size_d % JK_CCSD_T_D1_7_SIZE_SLICE_1_D;
	}
	if ((size_f - (blk_idx_f * JK_CCSD_T_D1_7_SIZE_SLICE_1_F)) >= JK_CCSD_T_D1_7_SIZE_SLICE_1_F)
	{
		rng_f = JK_CCSD_T_D1_7_SIZE_SLICE_1_F;
	}
	else
	{
		rng_f = size_f % JK_CCSD_T_D1_7_SIZE_SLICE_1_F;
	}

	double temp_av;
	double temp_bv[8];
	double reg_tile[8][4];

	for (int i = 0; i < 8; i++)
	for (int j = 0; j < 4; j++)
	reg_tile[i][j] = 0.0;

	// tensor contraction: [[16, 'STR_SD2_T2_H7', 'y', 't2', ['g', 'f', 'e', 'c']], [16, 'STR_SD2_V2_H7', 'x', 'v2', ['a', 'b', 'd', 'g']], '+=']
	#pragma unroll 1
	for (int l = 0; l < size_internal; l += JK_CCSD_T_D1_7_SIZE_INT_UNIT_1)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + JK_CCSD_T_D1_7_SIZE_INT_UNIT_1) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		//---------------------------------------------------------------------------------------------------
		// This is for the new version
		// This Part is for Loading Input-Left
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (idx_f < rng_f && 0 < rng_c && threadIdx.x < JK_CCSD_T_D1_7_SIZE_INT_UNIT_1 - internal_upperbound)
		for (int ll = 0; ll < rng_e; ll++)
		{
			// ['g', 'f', 'e', 'c']
			// Exception: Temp. version!: threadIdx.x + l
			// Exception: Temp. version!: idx_f < rng_f
			sm_a[threadIdx.x][threadIdx.y + ll * 8] = dev_t2[(blk_idx_f * JK_CCSD_T_D1_7_SIZE_SLICE_1_F + idx_f + (blk_idx_e * JK_CCSD_T_D1_7_SIZE_SLICE_1_E + ll + (blk_idx_c * JK_CCSD_T_D1_7_SIZE_SLICE_1_C + 0) * size_e) * size_f) * size_g + (threadIdx.x + l)];
		}
		
		// This Part is for Loading Input-Right
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (idx_a < rng_a && 0 < rng_d && threadIdx.y < JK_CCSD_T_D1_7_SIZE_INT_UNIT_1 - internal_upperbound)
		for (int ll = 0; ll < rng_b; ll++)
		{
			// ['a', 'b', 'd', 'g']
			// Exception: Temp. version!: threadIdx.y + l + 0
			// Exception: Temp. version!: idx_a < rng_a
			sm_b[threadIdx.y + 0][threadIdx.x + ll * 16] = dev_v2[blk_idx_a * JK_CCSD_T_D1_7_SIZE_SLICE_1_A + idx_a + (blk_idx_b * JK_CCSD_T_D1_7_SIZE_SLICE_1_B + ll + (blk_idx_d * JK_CCSD_T_D1_7_SIZE_SLICE_1_D + 0) * size_b) * size_a + (threadIdx.y + l + 0) * stride_int_v2];
			// Exception: Temp. version!: threadIdx.y + l + 8
			// Exception: Temp. version!: idx_a < rng_a
			if (threadIdx.y + l + 8 < size_internal && idx_a < rng_a) 
			sm_b[threadIdx.y + 8][threadIdx.x + ll * 16] = dev_v2[blk_idx_a * JK_CCSD_T_D1_7_SIZE_SLICE_1_A + idx_a + (blk_idx_b * JK_CCSD_T_D1_7_SIZE_SLICE_1_B + ll + (blk_idx_d * JK_CCSD_T_D1_7_SIZE_SLICE_1_D + 0) * size_b) * size_a + (threadIdx.y + l + 8) * stride_int_v2];
		}
		__syncthreads();
		//---------------------------------------------------------------------------------------------------
		

		// Part: Generalized Threads
		for (int ll = 0; ll < JK_CCSD_T_D1_7_SIZE_INT_UNIT_1 - internal_upperbound; ll++)
		{
			temp_bv[0] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_7_SIZE_SLICE_1_F + 0];
			temp_bv[1] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_7_SIZE_SLICE_1_F + 8];
			temp_bv[2] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_7_SIZE_SLICE_1_F + 16];
			temp_bv[3] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_7_SIZE_SLICE_1_F + 24];
			temp_bv[4] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_7_SIZE_SLICE_1_F + 32];
			temp_bv[5] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_7_SIZE_SLICE_1_F + 40];
			temp_bv[6] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_7_SIZE_SLICE_1_F + 48];
			temp_bv[7] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_7_SIZE_SLICE_1_F + 56];

			for (int xx = 0; xx < 4; xx++) // (1)
			{
				temp_av = sm_b[ll][idx_a + (idx_d) * JK_CCSD_T_D1_7_SIZE_SLICE_1_A + (xx * 16)];

				reg_tile[0][xx] += temp_av * temp_bv[0];
				reg_tile[1][xx] += temp_av * temp_bv[1];
				reg_tile[2][xx] += temp_av * temp_bv[2];
				reg_tile[3][xx] += temp_av * temp_bv[3];
				reg_tile[4][xx] += temp_av * temp_bv[4];
				reg_tile[5][xx] += temp_av * temp_bv[5];
				reg_tile[6][xx] += temp_av * temp_bv[6];
				reg_tile[7][xx] += temp_av * temp_bv[7];
			}
		}
		__syncthreads();
	}


	// Store Results (Registers) to Global Memory
	// Part: Generalized Threads
	// Part: Generalized Register-Tiling
	if (idx_a < rng_a && idx_d < rng_d && idx_f < rng_f && idx_c < rng_c)
	for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			if(i < rng_e && j < rng_b)
			{
		    	dev_t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] += reg_tile[i][j];
			}
		}
	}
}

// written by tc_interface.tc_gen_code_interface_Header()
extern "C"
void jk_ccsd_t_d1_7_fusion(int size_a, int size_b, int size_c, int size_e, int size_d, int size_f, int size_g, double* t3, double* host_t2, double* host_v2, int cond_kernel_1, int opt_register_transpose)
{
	int num_thread_blocks_kernel_1;

	double* dev_t3;
	double* dev_t2;
	double* dev_v2;

	num_thread_blocks_kernel_1 = CEIL(size_a, JK_CCSD_T_D1_7_SIZE_SLICE_1_A) * CEIL(size_b, JK_CCSD_T_D1_7_SIZE_SLICE_1_B) * CEIL(size_c, JK_CCSD_T_D1_7_SIZE_SLICE_1_C) * CEIL(size_e, JK_CCSD_T_D1_7_SIZE_SLICE_1_E) * CEIL(size_d, JK_CCSD_T_D1_7_SIZE_SLICE_1_D) * CEIL(size_f, JK_CCSD_T_D1_7_SIZE_SLICE_1_F);
    
    // cudaMalloc()
	// cudaMalloc((void**) &dev_t3, sizeof(double) * size_a * size_b * size_c * size_e * size_d * size_f);
	cudaMalloc((void**) &dev_t2, sizeof(double) * size_c * size_e * size_f * size_g);
	cudaMalloc((void**) &dev_v2, sizeof(double) * size_g * size_d * size_b * size_a);

	// cudaMemcpy()
	// cudaMemcpy(dev_t3, t3, sizeof(double) * size_a * size_b * size_c * size_e * size_d * size_f, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_c * size_e * size_f * size_g, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_g * size_d * size_b * size_a, cudaMemcpyHostToDevice);

	// Related to Kernels
	// There are 1 Basic Kernels
    long long int tmp_operations = 2 * (long long int)(size_a * size_b * size_c * size_e * size_d * size_f) * size_g;
    /*
	printf ("========================================= fusedKernels =============================================\n");
	printf ("		Grid Size  : %6d (1D)\n", num_thread_blocks_kernel_1);
	printf ("		Block-size : %2d, %2d (2D)\n", JK_CCSD_T_D1_7_SIZE_TB_1_X, JK_CCSD_T_D1_7_SIZE_TB_1_Y);
	printf ("		Reg.-size  : %2d, %2d (2D)\n", JK_CCSD_T_D1_7_SIZE_REG_1_X, JK_CCSD_T_D1_7_SIZE_REG_1_Y);
	printf ("		A thread deals with (%d x %d) elements (basically)\n", JK_CCSD_T_D1_7_SIZE_TB_1_X * JK_CCSD_T_D1_7_SIZE_REG_1_X, JK_CCSD_T_D1_7_SIZE_TB_1_Y * JK_CCSD_T_D1_7_SIZE_REG_1_Y);
	printf ("		# of Operations: %lld\n", tmp_operations);
    printf ("====================================================================================================\n");
    */
	dim3 gridsize_1(num_thread_blocks_kernel_1);
	dim3 blocksize_1(JK_CCSD_T_D1_7_SIZE_TB_1_X, JK_CCSD_T_D1_7_SIZE_TB_1_Y);

	int stride_output_a = 1;
	int stride_output_b = stride_output_a * size_a;
	int stride_output_c = stride_output_b * size_b;
	int stride_output_e = stride_output_c * size_c;
	int stride_output_d = stride_output_e * size_e;
	int stride_output_f = stride_output_d * size_d;

	int stride_reg_x_1 = stride_output_b;
	int stride_reg_y_1 = stride_output_e;

	int size_internal = size_g;

	int stride_int_t2 = 1;
	int stride_int_v2 = size_a * size_b * size_d;

    dev_t3 = t3_d;

	// New Caller
	jk_ccsd_t_d1_7_kernel__4_1<<<gridsize_1, blocksize_1>>>(dev_t3, dev_t2, dev_v2, size_a, size_b, size_c, size_e, size_d, size_f, size_g, CEIL(size_a, JK_CCSD_T_D1_7_SIZE_SLICE_1_A), CEIL(size_b, JK_CCSD_T_D1_7_SIZE_SLICE_1_B), CEIL(size_c, JK_CCSD_T_D1_7_SIZE_SLICE_1_C), CEIL(size_e, JK_CCSD_T_D1_7_SIZE_SLICE_1_E), CEIL(size_d, JK_CCSD_T_D1_7_SIZE_SLICE_1_D), CEIL(size_f, JK_CCSD_T_D1_7_SIZE_SLICE_1_F), stride_int_t2, stride_int_v2, stride_reg_x_1, stride_reg_y_1, size_internal);

	// Copy the Result from Device to Host
	// cudaMemcpy(t3, dev_t3, sizeof(double) * (size_a * size_b * size_c * size_e * size_d * size_f), cudaMemcpyDeviceToHost);

	// cudaFree()
    // cudaFree(dev_t3);	
    cudaFree(dev_t2);	cudaFree(dev_v2);

	// Shoule be Fixed
	// HostFree

}

// This is written by tc_interface.tc_gen_code_interface()
// This Interface Should be Called to Run the Kernels
extern "C"
void jk_ccsd_t_d1_7_fusion_(int size_a, int size_b, int size_c, int size_e, int size_d, int size_f, int size_g, double* t3, double* t2, double* v2, int cond_kernel_1, int opt_register_transpose)
{
	// Call An Application
	jk_ccsd_t_d1_7_fusion(size_a, size_b, size_c, size_e, size_d, size_f, size_g, t3, t2, v2, cond_kernel_1, opt_register_transpose);
}



/*----------------------------------------------------------------------*
 *  [d1][8] triplesx[h3,h1,h2,p5,p6,p4] -= t2sub[h7,p4,p5,h1] * v2sub[h3,h2,p6,h7]
 *----------------------------------------------------------------------*/
// created by tc_gen_definition_new()
#define JK_CCSD_T_D1_8_SIZE_SLICE_1_G   16
#define JK_CCSD_T_D1_8_SIZE_SLICE_1_A   16
#define JK_CCSD_T_D1_8_SIZE_SLICE_1_B   4
#define JK_CCSD_T_D1_8_SIZE_SLICE_1_D   1
#define JK_CCSD_T_D1_8_SIZE_SLICE_1_F   8
#define JK_CCSD_T_D1_8_SIZE_SLICE_1_E   8
#define JK_CCSD_T_D1_8_SIZE_SLICE_1_C   1

#define JK_CCSD_T_D1_8_SIZE_INT_UNIT_1  JK_CCSD_T_D1_8_SIZE_SLICE_1_G

#define JK_CCSD_T_D1_8_SIZE_TB_1_X 	    JK_CCSD_T_D1_8_SIZE_SLICE_1_A * JK_CCSD_T_D1_8_SIZE_SLICE_1_D
#define JK_CCSD_T_D1_8_SIZE_TB_1_Y 	    JK_CCSD_T_D1_8_SIZE_SLICE_1_F * JK_CCSD_T_D1_8_SIZE_SLICE_1_C
#define JK_CCSD_T_D1_8_SIZE_REG_1_X 	JK_CCSD_T_D1_8_SIZE_SLICE_1_B
#define JK_CCSD_T_D1_8_SIZE_REG_1_Y 	JK_CCSD_T_D1_8_SIZE_SLICE_1_E

// created by tc_gen_code_Kernel()
__global__ void jk_ccsd_t_d1_8_kernel__4_1(double* dev_t3, double* dev_t2, double* dev_v2, int size_a, int size_c, int size_b, int size_e, int size_d, int size_f, int size_g, int numBlk_a, int numBlk_c, int numBlk_b, int numBlk_e, int numBlk_d, int numBlk_f, int stride_int_t2, int stride_int_v2, int stride_reg_x, int stride_reg_y, int size_internal)
{
	// For Shared Memory,
	__shared__ double sm_a[16][64];
	__shared__ double sm_b[16][64];

	int internal_upperbound   = 0;
	int internal_offset;

	// when opt_pre_computed == -1, all indices will be calculated manually
	// # of indices mapped on TB_X: 2
	// # of indices mapped on TB_Y: 2
	int idx_a = threadIdx.x % JK_CCSD_T_D1_8_SIZE_SLICE_1_A;
	int idx_d = threadIdx.x / JK_CCSD_T_D1_8_SIZE_SLICE_1_A;
	int idx_f = threadIdx.y % JK_CCSD_T_D1_8_SIZE_SLICE_1_F;
	int idx_c = threadIdx.y / JK_CCSD_T_D1_8_SIZE_SLICE_1_F;

	int tmp_blkIdx;
	int blk_idx_f = blockIdx.x / (numBlk_d * numBlk_e * numBlk_b * numBlk_c * numBlk_a);
	tmp_blkIdx = blockIdx.x % (numBlk_d * numBlk_e * numBlk_b * numBlk_c * numBlk_a);

	int blk_idx_d = tmp_blkIdx / (numBlk_e * numBlk_b * numBlk_c * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_e * numBlk_b * numBlk_c * numBlk_a);

	int blk_idx_e = tmp_blkIdx / (numBlk_b * numBlk_c * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_b * numBlk_c * numBlk_a);

	int blk_idx_b = tmp_blkIdx / (numBlk_c * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_c * numBlk_a);

	int blk_idx_c = tmp_blkIdx / numBlk_a;
	tmp_blkIdx = tmp_blkIdx % (numBlk_a);

	int  blk_idx_a = tmp_blkIdx;

	int t3_base_thread = blk_idx_a * JK_CCSD_T_D1_8_SIZE_SLICE_1_A + idx_a + (blk_idx_c * JK_CCSD_T_D1_8_SIZE_SLICE_1_C + idx_c + (blk_idx_b * JK_CCSD_T_D1_8_SIZE_SLICE_1_B + (blk_idx_e * JK_CCSD_T_D1_8_SIZE_SLICE_1_E + (blk_idx_d * JK_CCSD_T_D1_8_SIZE_SLICE_1_D + idx_d + (blk_idx_f * JK_CCSD_T_D1_8_SIZE_SLICE_1_F + idx_f) * size_d) * size_e) * size_b) * size_c) * size_a;

	// need to support partial tiles
	int rng_a, rng_c, rng_b, rng_e, rng_d, rng_f;
	if ((size_a - (blk_idx_a * JK_CCSD_T_D1_8_SIZE_SLICE_1_A)) >= JK_CCSD_T_D1_8_SIZE_SLICE_1_A)
	{
		rng_a = JK_CCSD_T_D1_8_SIZE_SLICE_1_A;
	}
	else
	{
		rng_a = size_a % JK_CCSD_T_D1_8_SIZE_SLICE_1_A;
	}
	if ((size_c - (blk_idx_c * JK_CCSD_T_D1_8_SIZE_SLICE_1_C)) >= JK_CCSD_T_D1_8_SIZE_SLICE_1_C)
	{
		rng_c = JK_CCSD_T_D1_8_SIZE_SLICE_1_C;
	}
	else
	{
		rng_c = size_c % JK_CCSD_T_D1_8_SIZE_SLICE_1_C;
	}
	if ((size_b - (blk_idx_b * JK_CCSD_T_D1_8_SIZE_SLICE_1_B)) >= JK_CCSD_T_D1_8_SIZE_SLICE_1_B)
	{
		rng_b = JK_CCSD_T_D1_8_SIZE_SLICE_1_B;
	}
	else
	{
		rng_b = size_b % JK_CCSD_T_D1_8_SIZE_SLICE_1_B;
	}
	if ((size_e - (blk_idx_e * JK_CCSD_T_D1_8_SIZE_SLICE_1_E)) >= JK_CCSD_T_D1_8_SIZE_SLICE_1_E)
	{
		rng_e = JK_CCSD_T_D1_8_SIZE_SLICE_1_E;
	}
	else
	{
		rng_e = size_e % JK_CCSD_T_D1_8_SIZE_SLICE_1_E;
	}
	if ((size_d - (blk_idx_d * JK_CCSD_T_D1_8_SIZE_SLICE_1_D)) >= JK_CCSD_T_D1_8_SIZE_SLICE_1_D)
	{
		rng_d = JK_CCSD_T_D1_8_SIZE_SLICE_1_D;
	}
	else
	{
		rng_d = size_d % JK_CCSD_T_D1_8_SIZE_SLICE_1_D;
	}
	if ((size_f - (blk_idx_f * JK_CCSD_T_D1_8_SIZE_SLICE_1_F)) >= JK_CCSD_T_D1_8_SIZE_SLICE_1_F)
	{
		rng_f = JK_CCSD_T_D1_8_SIZE_SLICE_1_F;
	}
	else
	{
		rng_f = size_f % JK_CCSD_T_D1_8_SIZE_SLICE_1_F;
	}

	double temp_av;
	double temp_bv[8];
	double reg_tile[8][4];

	for (int i = 0; i < 8; i++)
	for (int j = 0; j < 4; j++)
	reg_tile[i][j] = 0.0;

	// tensor contraction: [[16, 'STR_SD2_T2_H7', 'y', 't2', ['g', 'f', 'e', 'c']], [16, 'STR_SD2_V2_H7', 'x', 'v2', ['a', 'b', 'd', 'g']], '+=']
	#pragma unroll 1
	for (int l = 0; l < size_internal; l += JK_CCSD_T_D1_8_SIZE_INT_UNIT_1)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + JK_CCSD_T_D1_8_SIZE_INT_UNIT_1) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		//---------------------------------------------------------------------------------------------------
		// This is for the new version
		// This Part is for Loading Input-Left
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (idx_f < rng_f && 0 < rng_c && threadIdx.x < JK_CCSD_T_D1_8_SIZE_INT_UNIT_1 - internal_upperbound)
		for (int ll = 0; ll < rng_e; ll++)
		{
			// ['g', 'f', 'e', 'c']
			// Exception: Temp. version!: threadIdx.x + l
			// Exception: Temp. version!: idx_f < rng_f
			sm_a[threadIdx.x][threadIdx.y + ll * 8] = dev_t2[(blk_idx_f * JK_CCSD_T_D1_8_SIZE_SLICE_1_F + idx_f + (blk_idx_e * JK_CCSD_T_D1_8_SIZE_SLICE_1_E + ll + (blk_idx_c * JK_CCSD_T_D1_8_SIZE_SLICE_1_C + 0) * size_e) * size_f) * size_g + (threadIdx.x + l)];
		}
		
		// This Part is for Loading Input-Right
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (idx_a < rng_a && 0 < rng_d && threadIdx.y < JK_CCSD_T_D1_8_SIZE_INT_UNIT_1 - internal_upperbound)
		for (int ll = 0; ll < rng_b; ll++)
		{
			// ['a', 'b', 'd', 'g']
			// Exception: Temp. version!: threadIdx.y + l + 0
			// Exception: Temp. version!: idx_a < rng_a
			sm_b[threadIdx.y + 0][threadIdx.x + ll * 16] = dev_v2[blk_idx_a * JK_CCSD_T_D1_8_SIZE_SLICE_1_A + idx_a + (blk_idx_b * JK_CCSD_T_D1_8_SIZE_SLICE_1_B + ll + (blk_idx_d * JK_CCSD_T_D1_8_SIZE_SLICE_1_D + 0) * size_b) * size_a + (threadIdx.y + l + 0) * stride_int_v2];
			// Exception: Temp. version!: threadIdx.y + l + 8
			// Exception: Temp. version!: idx_a < rng_a
			if (threadIdx.y + l + 8 < size_internal && idx_a < rng_a) 
			sm_b[threadIdx.y + 8][threadIdx.x + ll * 16] = dev_v2[blk_idx_a * JK_CCSD_T_D1_8_SIZE_SLICE_1_A + idx_a + (blk_idx_b * JK_CCSD_T_D1_8_SIZE_SLICE_1_B + ll + (blk_idx_d * JK_CCSD_T_D1_8_SIZE_SLICE_1_D + 0) * size_b) * size_a + (threadIdx.y + l + 8) * stride_int_v2];
		}
		__syncthreads();
		//---------------------------------------------------------------------------------------------------
		

		// Part: Generalized Threads
		for (int ll = 0; ll < JK_CCSD_T_D1_8_SIZE_INT_UNIT_1 - internal_upperbound; ll++)
		{
			temp_bv[0] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_8_SIZE_SLICE_1_F + 0];
			temp_bv[1] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_8_SIZE_SLICE_1_F + 8];
			temp_bv[2] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_8_SIZE_SLICE_1_F + 16];
			temp_bv[3] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_8_SIZE_SLICE_1_F + 24];
			temp_bv[4] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_8_SIZE_SLICE_1_F + 32];
			temp_bv[5] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_8_SIZE_SLICE_1_F + 40];
			temp_bv[6] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_8_SIZE_SLICE_1_F + 48];
			temp_bv[7] = sm_a[ll][idx_f + (idx_c) * JK_CCSD_T_D1_8_SIZE_SLICE_1_F + 56];

			for (int xx = 0; xx < 4; xx++) // (1)
			{
				temp_av = sm_b[ll][idx_a + (idx_d) * JK_CCSD_T_D1_8_SIZE_SLICE_1_A + (xx * 16)];

				reg_tile[0][xx] -= temp_av * temp_bv[0];
				reg_tile[1][xx] -= temp_av * temp_bv[1];
				reg_tile[2][xx] -= temp_av * temp_bv[2];
				reg_tile[3][xx] -= temp_av * temp_bv[3];
				reg_tile[4][xx] -= temp_av * temp_bv[4];
				reg_tile[5][xx] -= temp_av * temp_bv[5];
				reg_tile[6][xx] -= temp_av * temp_bv[6];
				reg_tile[7][xx] -= temp_av * temp_bv[7];
			}
		}
		__syncthreads();
	}


	// Store Results (Registers) to Global Memory
	// Part: Generalized Threads
	// Part: Generalized Register-Tiling
	if (idx_a < rng_a && idx_d < rng_d && idx_f < rng_f && idx_c < rng_c)
	for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			if(i < rng_e && j < rng_b)
			{
			    dev_t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] += reg_tile[i][j];
			}
		}
	}
}

// written by tc_interface.tc_gen_code_interface_Header()
extern "C"
void jk_ccsd_t_d1_8_fusion(int size_a, int size_c, int size_b, int size_e, int size_d, int size_f, int size_g, double* t3, double* host_t2, double* host_v2, int cond_kernel_1, int opt_register_transpose)
{
	int num_thread_blocks_kernel_1;

	double* dev_t3;
	double* dev_t2;
	double* dev_v2;

	num_thread_blocks_kernel_1 = CEIL(size_a, JK_CCSD_T_D1_8_SIZE_SLICE_1_A) * CEIL(size_c, JK_CCSD_T_D1_8_SIZE_SLICE_1_C) * CEIL(size_b, JK_CCSD_T_D1_8_SIZE_SLICE_1_B) * CEIL(size_e, JK_CCSD_T_D1_8_SIZE_SLICE_1_E) * CEIL(size_d, JK_CCSD_T_D1_8_SIZE_SLICE_1_D) * CEIL(size_f, JK_CCSD_T_D1_8_SIZE_SLICE_1_F);
    
    // cudaMalloc()
	// cudaMalloc((void**) &dev_t3, sizeof(double) * size_a * size_c * size_b * size_e * size_d * size_f);
	cudaMalloc((void**) &dev_t2, sizeof(double) * size_c * size_e * size_f * size_g);
	cudaMalloc((void**) &dev_v2, sizeof(double) * size_g * size_d * size_b * size_a);

	// cudaMemcpy()
	// cudaMemcpy(dev_t3, t3, sizeof(double) * size_a * size_c * size_b * size_e * size_d * size_f, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_c * size_e * size_f * size_g, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_g * size_d * size_b * size_a, cudaMemcpyHostToDevice);

	// Related to Kernels
	// There are 1 Basic Kernels
    long long int tmp_operations = 2 * (long long int)(size_a * size_c * size_b * size_e * size_d * size_f) * size_g;
    /*
	printf ("========================================= fusedKernels =============================================\n");
	printf ("		Grid Size  : %6d (1D)\n", num_thread_blocks_kernel_1);
	printf ("		Block-size : %2d, %2d (2D)\n", JK_CCSD_T_D1_8_SIZE_TB_1_X, JK_CCSD_T_D1_8_SIZE_TB_1_Y);
	printf ("		Reg.-size  : %2d, %2d (2D)\n", JK_CCSD_T_D1_8_SIZE_REG_1_X, JK_CCSD_T_D1_8_SIZE_REG_1_Y);
	printf ("		A thread deals with (%d x %d) elements (basically)\n", JK_CCSD_T_D1_8_SIZE_TB_1_X * JK_CCSD_T_D1_8_SIZE_REG_1_X, JK_CCSD_T_D1_8_SIZE_TB_1_Y * JK_CCSD_T_D1_8_SIZE_REG_1_Y);
	printf ("		# of Operations: %lld\n", tmp_operations);
	printf ("====================================================================================================\n");
    */
    dim3 gridsize_1(num_thread_blocks_kernel_1);
	dim3 blocksize_1(JK_CCSD_T_D1_8_SIZE_TB_1_X, JK_CCSD_T_D1_8_SIZE_TB_1_Y);

	int stride_output_a = 1;
	int stride_output_c = stride_output_a * size_a;
	int stride_output_b = stride_output_c * size_c;
	int stride_output_e = stride_output_b * size_b;
	int stride_output_d = stride_output_e * size_e;
	int stride_output_f = stride_output_d * size_d;

	int stride_reg_x_1 = stride_output_b;
	int stride_reg_y_1 = stride_output_e;

	int size_internal = size_g;

	int stride_int_t2 = 1;
	int stride_int_v2 = size_a * size_b * size_d;

    dev_t3 = t3_d;

	// New Caller
	jk_ccsd_t_d1_8_kernel__4_1<<<gridsize_1, blocksize_1>>>(dev_t3, dev_t2, dev_v2, size_a, size_c, size_b, size_e, size_d, size_f, size_g, CEIL(size_a, JK_CCSD_T_D1_8_SIZE_SLICE_1_A), CEIL(size_c, JK_CCSD_T_D1_8_SIZE_SLICE_1_C), CEIL(size_b, JK_CCSD_T_D1_8_SIZE_SLICE_1_B), CEIL(size_e, JK_CCSD_T_D1_8_SIZE_SLICE_1_E), CEIL(size_d, JK_CCSD_T_D1_8_SIZE_SLICE_1_D), CEIL(size_f, JK_CCSD_T_D1_8_SIZE_SLICE_1_F), stride_int_t2, stride_int_v2, stride_reg_x_1, stride_reg_y_1, size_internal);

	// Copy the Result from Device to Host
	// cudaMemcpy(t3, dev_t3, sizeof(double) * (size_a * size_c * size_b * size_e * size_d * size_f), cudaMemcpyDeviceToHost);

	// cudaFree()
    // cudaFree(dev_t3);	
    cudaFree(dev_t2);	cudaFree(dev_v2);
}

// This is written by tc_interface.tc_gen_code_interface()
// This Interface Should be Called to Run the Kernels
extern "C"
void jk_ccsd_t_d1_8_fusion_(int size_a, int size_c, int size_b, int size_e, int size_d, int size_f, int size_g, double* t3, double* t2, double* v2, int cond_kernel_1, int opt_register_transpose)
{
	// Call An Application
	jk_ccsd_t_d1_8_fusion(size_a, size_c, size_b, size_e, size_d, size_f, size_g, t3, t2, v2, cond_kernel_1, opt_register_transpose);
}


/*----------------------------------------------------------------------*
 *  [d1][9] triplesx[h1,h3,p5,p6,p4] += t2sub[h7,p4,p5,h1] * v2sub[h3,p6,h7]
 *----------------------------------------------------------------------*/
// created by tc_gen_definition_new()
#define JK_CCSD_T_D1_9_SIZE_SLICE_1_G 16
#define JK_CCSD_T_D1_9_SIZE_SLICE_1_C 16
#define JK_CCSD_T_D1_9_SIZE_SLICE_1_F 4
#define JK_CCSD_T_D1_9_SIZE_SLICE_1_E 1
#define JK_CCSD_T_D1_9_SIZE_SLICE_1_A 16
#define JK_CCSD_T_D1_9_SIZE_SLICE_1_B 4
#define JK_CCSD_T_D1_9_SIZE_SLICE_1_D 1

#define JK_CCSD_T_D1_9_SIZE_INT_UNIT_1 JK_CCSD_T_D1_9_SIZE_SLICE_1_G

#define JK_CCSD_T_D1_9_SIZE_TB_1_X 	JK_CCSD_T_D1_9_SIZE_SLICE_1_C * JK_CCSD_T_D1_9_SIZE_SLICE_1_E
#define JK_CCSD_T_D1_9_SIZE_TB_1_Y 	JK_CCSD_T_D1_9_SIZE_SLICE_1_A * JK_CCSD_T_D1_9_SIZE_SLICE_1_D
#define JK_CCSD_T_D1_9_SIZE_REG_1_X 	JK_CCSD_T_D1_9_SIZE_SLICE_1_F
#define JK_CCSD_T_D1_9_SIZE_REG_1_Y 	JK_CCSD_T_D1_9_SIZE_SLICE_1_B

// created by tc_gen_code_Kernel()
__global__ void jk_ccsd_t_d1_9_kernel__4_1(double* dev_t3, double* dev_t2, double* dev_v2, int size_c, int size_a, int size_b, int size_e, int size_d, int size_f, int size_g, int numBlk_c, int numBlk_a, int numBlk_b, int numBlk_e, int numBlk_d, int numBlk_f, int stride_int_t2, int stride_int_v2, int stride_reg_x, int stride_reg_y, int size_internal)
{
	// For Shared Memory,
	__shared__ double sm_a[16][64];
	__shared__ double sm_b[16][64];


	int internal_upperbound   = 0;
	int internal_offset;

	// when opt_pre_computed == -1, all indices will be calculated manually
	// # of indices mapped on TB_X: 2
	// # of indices mapped on TB_Y: 2
	int idx_c = threadIdx.x % JK_CCSD_T_D1_9_SIZE_SLICE_1_C;
	int idx_e = threadIdx.x / JK_CCSD_T_D1_9_SIZE_SLICE_1_C;
	int idx_a = threadIdx.y % JK_CCSD_T_D1_9_SIZE_SLICE_1_A;
	int idx_d = threadIdx.y / JK_CCSD_T_D1_9_SIZE_SLICE_1_A;

	int tmp_blkIdx;
	int blk_idx_f = blockIdx.x / (numBlk_d * numBlk_e * numBlk_b * numBlk_a * numBlk_c);
	tmp_blkIdx = blockIdx.x % (numBlk_d * numBlk_e * numBlk_b * numBlk_a * numBlk_c);

	int blk_idx_d = tmp_blkIdx / (numBlk_e * numBlk_b * numBlk_a * numBlk_c);
	tmp_blkIdx = tmp_blkIdx % (numBlk_e * numBlk_b * numBlk_a * numBlk_c);

	int blk_idx_e = tmp_blkIdx / (numBlk_b * numBlk_a * numBlk_c);
	tmp_blkIdx = tmp_blkIdx % (numBlk_b * numBlk_a * numBlk_c);

	int blk_idx_b = tmp_blkIdx / (numBlk_a * numBlk_c);
	tmp_blkIdx = tmp_blkIdx % (numBlk_a * numBlk_c);

	int blk_idx_a = tmp_blkIdx / numBlk_c;
	tmp_blkIdx = tmp_blkIdx % (numBlk_c);

	int  blk_idx_c = tmp_blkIdx;

	int t3_base_thread = blk_idx_c * JK_CCSD_T_D1_9_SIZE_SLICE_1_C + idx_c + (blk_idx_a * JK_CCSD_T_D1_9_SIZE_SLICE_1_A + idx_a + (blk_idx_b * JK_CCSD_T_D1_9_SIZE_SLICE_1_B + (blk_idx_e * JK_CCSD_T_D1_9_SIZE_SLICE_1_E + idx_e + (blk_idx_d * JK_CCSD_T_D1_9_SIZE_SLICE_1_D + idx_d + (blk_idx_f * JK_CCSD_T_D1_9_SIZE_SLICE_1_F) * size_d) * size_e) * size_b) * size_a) * size_c;

	// need to support partial tiles
	int rng_c, rng_a, rng_b, rng_e, rng_d, rng_f;
	if ((size_c - (blk_idx_c * JK_CCSD_T_D1_9_SIZE_SLICE_1_C)) >= JK_CCSD_T_D1_9_SIZE_SLICE_1_C)
	{
		rng_c = JK_CCSD_T_D1_9_SIZE_SLICE_1_C;
	}
	else
	{
		rng_c = size_c % JK_CCSD_T_D1_9_SIZE_SLICE_1_C;
	}
	if ((size_a - (blk_idx_a * JK_CCSD_T_D1_9_SIZE_SLICE_1_A)) >= JK_CCSD_T_D1_9_SIZE_SLICE_1_A)
	{
		rng_a = JK_CCSD_T_D1_9_SIZE_SLICE_1_A;
	}
	else
	{
		rng_a = size_a % JK_CCSD_T_D1_9_SIZE_SLICE_1_A;
	}
	if ((size_b - (blk_idx_b * JK_CCSD_T_D1_9_SIZE_SLICE_1_B)) >= JK_CCSD_T_D1_9_SIZE_SLICE_1_B)
	{
		rng_b = JK_CCSD_T_D1_9_SIZE_SLICE_1_B;
	}
	else
	{
		rng_b = size_b % JK_CCSD_T_D1_9_SIZE_SLICE_1_B;
	}
	if ((size_e - (blk_idx_e * JK_CCSD_T_D1_9_SIZE_SLICE_1_E)) >= JK_CCSD_T_D1_9_SIZE_SLICE_1_E)
	{
		rng_e = JK_CCSD_T_D1_9_SIZE_SLICE_1_E;
	}
	else
	{
		rng_e = size_e % JK_CCSD_T_D1_9_SIZE_SLICE_1_E;
	}
	if ((size_d - (blk_idx_d * JK_CCSD_T_D1_9_SIZE_SLICE_1_D)) >= JK_CCSD_T_D1_9_SIZE_SLICE_1_D)
	{
		rng_d = JK_CCSD_T_D1_9_SIZE_SLICE_1_D;
	}
	else
	{
		rng_d = size_d % JK_CCSD_T_D1_9_SIZE_SLICE_1_D;
	}
	if ((size_f - (blk_idx_f * JK_CCSD_T_D1_9_SIZE_SLICE_1_F)) >= JK_CCSD_T_D1_9_SIZE_SLICE_1_F)
	{
		rng_f = JK_CCSD_T_D1_9_SIZE_SLICE_1_F;
	}
	else
	{
		rng_f = size_f % JK_CCSD_T_D1_9_SIZE_SLICE_1_F;
	}

	double temp_av;
	double temp_bv[4];
	double reg_tile[4][4];

	for (int i = 0; i < 4; i++)
	for (int j = 0; j < 4; j++)
	reg_tile[i][j] = 0.0;

	// tensor contraction: [[16, 'STR_SD2_T2_H7', 'x', 't2', ['g', 'f', 'e', 'c']], [16, 'STR_SD2_V2_H7', 'y', 'v2', ['a', 'b', 'd', 'g']], '+=']
	#pragma unroll 1
	for (int l = 0; l < size_internal; l += JK_CCSD_T_D1_9_SIZE_INT_UNIT_1)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + JK_CCSD_T_D1_9_SIZE_INT_UNIT_1) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		//---------------------------------------------------------------------------------------------------
		// This is for the new version
		// This Part is for Loading Input-Left
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (0 < rng_e && idx_a < rng_c && threadIdx.x < JK_CCSD_T_D1_9_SIZE_INT_UNIT_1 - internal_upperbound)
		for (int ll = 0; ll < rng_f; ll++)
		{
			// ['g', 'f', 'e', 'c']
			// Exception: Temp. version!: threadIdx.x + l
			// Exception: Temp. version!: 0 < rng_e
			sm_a[threadIdx.x][threadIdx.y + ll * 16] = dev_t2[(blk_idx_f * JK_CCSD_T_D1_9_SIZE_SLICE_1_F + ll + (blk_idx_e * JK_CCSD_T_D1_9_SIZE_SLICE_1_E + 0 + (blk_idx_c * JK_CCSD_T_D1_9_SIZE_SLICE_1_C + idx_a) * size_e) * size_f) * size_g + (threadIdx.x + l)];
		}
		
		// This Part is for Loading Input-Right
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (idx_c < rng_a && 0 < rng_d && threadIdx.y < JK_CCSD_T_D1_9_SIZE_INT_UNIT_1 - internal_upperbound)
		for (int ll = 0; ll < rng_b; ll++)
		{
			// ['a', 'b', 'd', 'g']
			// Exception: Temp. version!: threadIdx.y + l
			// Exception: Temp. version!: idx_c < rng_a
			sm_b[threadIdx.y][threadIdx.x + ll * 16] = dev_v2[blk_idx_a * JK_CCSD_T_D1_9_SIZE_SLICE_1_A + idx_c + (blk_idx_b * JK_CCSD_T_D1_9_SIZE_SLICE_1_B + ll + (blk_idx_d * JK_CCSD_T_D1_9_SIZE_SLICE_1_D + 0) * size_b) * size_a + (threadIdx.y + l) * stride_int_v2];
		}
		__syncthreads();
		//---------------------------------------------------------------------------------------------------
		

		// Part: Generalized Threads
		for (int ll = 0; ll < JK_CCSD_T_D1_9_SIZE_INT_UNIT_1 - internal_upperbound; ll++)
		{
			temp_bv[0] = sm_b[ll][idx_a + (idx_d) * JK_CCSD_T_D1_9_SIZE_SLICE_1_A + 0];
			temp_bv[1] = sm_b[ll][idx_a + (idx_d) * JK_CCSD_T_D1_9_SIZE_SLICE_1_A + 16];
			temp_bv[2] = sm_b[ll][idx_a + (idx_d) * JK_CCSD_T_D1_9_SIZE_SLICE_1_A + 32];
			temp_bv[3] = sm_b[ll][idx_a + (idx_d) * JK_CCSD_T_D1_9_SIZE_SLICE_1_A + 48];

			for (int xx = 0; xx < 4; xx++) // (1)
			{
				temp_av = sm_a[ll][idx_e + (idx_c) * JK_CCSD_T_D1_9_SIZE_SLICE_1_E + (xx * 16)];

				reg_tile[0][xx] += temp_av * temp_bv[0];
				reg_tile[1][xx] += temp_av * temp_bv[1];
				reg_tile[2][xx] += temp_av * temp_bv[2];
				reg_tile[3][xx] += temp_av * temp_bv[3];
			}
		}
		__syncthreads();
	}


	// Store Results (Registers) to Global Memory
	// Part: Generalized Threads
	// Part: Generalized Register-Tiling
	if (idx_c < rng_c && idx_e < rng_e && idx_a < rng_a && idx_d < rng_d)
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			if(i < rng_b && j < rng_f)
			{
			    dev_t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] += reg_tile[i][j];
			}
		}
	}
}

// written by tc_interface.tc_gen_code_interface_Header()
extern "C"
void jk_ccsd_t_d1_9_fusion(int size_c, int size_a, int size_b, int size_e, int size_d, int size_f, int size_g, double* t3, double* host_t2, double* host_v2, int cond_kernel_1, int opt_register_transpose)
{
	int num_thread_blocks_kernel_1;

	double* dev_t3;
	double* dev_t2;
	double* dev_v2;

	num_thread_blocks_kernel_1 = CEIL(size_c, JK_CCSD_T_D1_9_SIZE_SLICE_1_C) * CEIL(size_a, JK_CCSD_T_D1_9_SIZE_SLICE_1_A) * CEIL(size_b, JK_CCSD_T_D1_9_SIZE_SLICE_1_B) * CEIL(size_e, JK_CCSD_T_D1_9_SIZE_SLICE_1_E) * CEIL(size_d, JK_CCSD_T_D1_9_SIZE_SLICE_1_D) * CEIL(size_f, JK_CCSD_T_D1_9_SIZE_SLICE_1_F);
    
    // cudaMalloc()
	// cudaMalloc((void**) &dev_t3, sizeof(double) * size_c * size_a * size_b * size_e * size_d * size_f);
	cudaMalloc((void**) &dev_t2, sizeof(double) * size_c * size_e * size_f * size_g);
	cudaMalloc((void**) &dev_v2, sizeof(double) * size_g * size_d * size_b * size_a);

	// cudaMemcpy()
	// cudaMemcpy(dev_t3, t3, sizeof(double) * size_c * size_a * size_b * size_e * size_d * size_f, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_c * size_e * size_f * size_g, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_g * size_d * size_b * size_a, cudaMemcpyHostToDevice);

	// Related to Kernels
	// There are 1 Basic Kernels
    long long int tmp_operations = 2 * (long long int)(size_c * size_a * size_b * size_e * size_d * size_f) * size_g;
    /*
	printf ("========================================= fusedKernels =============================================\n");
	printf ("		Grid Size  : %6d (1D)\n", num_thread_blocks_kernel_1);
	printf ("		Block-size : %2d, %2d (2D)\n", JK_CCSD_T_D1_9_SIZE_TB_1_X, JK_CCSD_T_D1_9_SIZE_TB_1_Y);
	printf ("		Reg.-size  : %2d, %2d (2D)\n", JK_CCSD_T_D1_9_SIZE_REG_1_X, JK_CCSD_T_D1_9_SIZE_REG_1_Y);
	printf ("		A thread deals with (%d x %d) elements (basically)\n", JK_CCSD_T_D1_9_SIZE_TB_1_X * JK_CCSD_T_D1_9_SIZE_REG_1_X, JK_CCSD_T_D1_9_SIZE_TB_1_Y * JK_CCSD_T_D1_9_SIZE_REG_1_Y);
	printf ("		# of Operations: %lld\n", tmp_operations);
    printf ("====================================================================================================\n");
    */
	dim3 gridsize_1(num_thread_blocks_kernel_1);
	dim3 blocksize_1(JK_CCSD_T_D1_9_SIZE_TB_1_X, JK_CCSD_T_D1_9_SIZE_TB_1_Y);

	int stride_output_c = 1;
	int stride_output_a = stride_output_c * size_c;
	int stride_output_b = stride_output_a * size_a;
	int stride_output_e = stride_output_b * size_b;
	int stride_output_d = stride_output_e * size_e;
	int stride_output_f = stride_output_d * size_d;

	int stride_reg_x_1 = stride_output_f;
	int stride_reg_y_1 = stride_output_b;

	int size_internal = size_g;

	int stride_int_t2 = 1;
	int stride_int_v2 = size_a * size_b * size_d;

    dev_t3 = t3_d;

	// New Caller
	jk_ccsd_t_d1_9_kernel__4_1<<<gridsize_1, blocksize_1>>>(dev_t3, dev_t2, dev_v2, size_c, size_a, size_b, size_e, size_d, size_f, size_g, CEIL(size_c, JK_CCSD_T_D1_9_SIZE_SLICE_1_C), CEIL(size_a, JK_CCSD_T_D1_9_SIZE_SLICE_1_A), CEIL(size_b, JK_CCSD_T_D1_9_SIZE_SLICE_1_B), CEIL(size_e, JK_CCSD_T_D1_9_SIZE_SLICE_1_E), CEIL(size_d, JK_CCSD_T_D1_9_SIZE_SLICE_1_D), CEIL(size_f, JK_CCSD_T_D1_9_SIZE_SLICE_1_F), stride_int_t2, stride_int_v2, stride_reg_x_1, stride_reg_y_1, size_internal);

	// Copy the Result from Device to Host
	// cudaMemcpy(t3, dev_t3, sizeof(double) * (size_c * size_a * size_b * size_e * size_d * size_f), cudaMemcpyDeviceToHost);

	// cudaFree()
    // cudaFree(dev_t3);	
    cudaFree(dev_t2);	cudaFree(dev_v2);
}

// This is written by tc_interface.tc_gen_code_interface()
// This Interface Should be Called to Run the Kernels
extern "C"
void jk_ccsd_t_d1_9_fusion_(int size_c, int size_a, int size_b, int size_e, int size_d, int size_f, int size_g, double* t3, double* t2, double* v2, int cond_kernel_1, int opt_register_transpose)
{
	// Pre-Processing for Split
	// Based on Tile-Sizes and Problem-Size
	// Currently, one index can be split into two indices

	// Call An Application
	jk_ccsd_t_d1_9_fusion(size_c, size_a, size_b, size_e, size_d, size_f, size_g, t3, t2, v2, cond_kernel_1, opt_register_transpose);
}

/*
*/
// created by tc_gen_definition()
#define FUSION_SIZE_SLICE_1_H3 4
#define FUSION_SIZE_SLICE_1_H2 4
#define FUSION_SIZE_SLICE_1_H1 4
#define FUSION_SIZE_SLICE_1_P6 4
#define FUSION_SIZE_SLICE_1_P5 4
#define FUSION_SIZE_SLICE_1_P4 4
#define FUSION_SIZE_SLICE_1_H7 16

#define FUSION_SIZE_SLICE_2_H3 4
#define FUSION_SIZE_SLICE_2_H2 4
#define FUSION_SIZE_SLICE_2_H1 4
#define FUSION_SIZE_SLICE_2_P6 4
#define FUSION_SIZE_SLICE_2_P5 4
#define FUSION_SIZE_SLICE_2_P4 4
#define FUSION_SIZE_SLICE_2_H7 16

#define FUSION_SIZE_INT_UNIT 	FUSION_SIZE_SLICE_1_H7

#define FUSION_SIZE_TB_1_X 	FUSION_SIZE_SLICE_1_H3 * FUSION_SIZE_SLICE_1_H2
#define FUSION_SIZE_TB_1_Y 	FUSION_SIZE_SLICE_1_P6 * FUSION_SIZE_SLICE_1_H1
#define FUSION_SIZE_REG_1_X 	FUSION_SIZE_SLICE_1_P5
#define FUSION_SIZE_REG_1_Y 	FUSION_SIZE_SLICE_1_P4

#define FUSION_SIZE_TB_2_X 	FUSION_SIZE_SLICE_2_H3 * FUSION_SIZE_SLICE_2_H2
#define FUSION_SIZE_TB_2_Y 	FUSION_SIZE_SLICE_2_P4 * FUSION_SIZE_SLICE_2_H1
#define FUSION_SIZE_REG_2_X 	FUSION_SIZE_SLICE_2_P5
#define FUSION_SIZE_REG_2_Y 	FUSION_SIZE_SLICE_2_P6

#define NUM_INDEX 	    6

//
__constant__ int list_stride_t2[9];
__constant__ int list_stride_v2[9];

//
//  Fully-Fused Kernels (exteranl_internal)
//
__global__ void kernel_ccsdT_sd1_fully_fused_partial_partial(double* t3, 
							double* d_t2_4, double* d_t2_5, double* d_t2_6, double* d_t2_7, double* d_t2_8, double* d_t2_9, 
							double* d_v2_4, double* d_v2_5, double* d_v2_6, double* d_v2_7, double* d_v2_8, double* d_v2_9, 
							double* d_t2_1, double* d_t2_2, double* d_t2_3, 
							double* d_v2_1, double* d_v2_2, double* d_v2_3, 
                            int size_h3,    int size_h2,    int size_h1,    int size_p6,    int size_p5,    int size_p4,    int size_h7, 
                            int numBlk_h3,  int numBlk_h2,  int numBlk_h1,  int numBlk_p6,  int numBlk_p5,  int numBlk_p4,
                            int kernel_1, int kernel_2, int kernel_3, 
                            int kernel_4, int kernel_5, int kernel_6, 
                            int kernel_7, int kernel_8, int kernel_9,
							int stride_reg_x, int stride_reg_y,
							int size_internal)
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

	int tmp_blkIdx;        
    int blk_idx_p4  = blockIdx.x / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);
    tmp_blkIdx      = blockIdx.x % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);

    int blk_idx_p5  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);

    int blk_idx_p6  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1);

    int blk_idx_h1 = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2);
    tmp_blkIdx     = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2);

    int blk_idx_h2 = (tmp_blkIdx) / (numBlk_h3);
    int blk_idx_h3 = blockIdx.x % (numBlk_h3);

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

    int t3_base_thread = blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + 
                        (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h2 + 
                        (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h1 + 
                        (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + 
                        (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + 
                        (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;

	double temp_av;
	double temp_bv[4];
	double reg_tile[4][4];

	for (int i = 0; i < 4; i++)
	for (int j = 0; j < 4; j++)
	reg_tile[i][j] = 0.0;

	//
	//	sd1_1,2,3
	//
	// tensor contraction
	#pragma unroll 1
	for (int l = 0; l < size_internal && kernel_1 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		if (idx_p6 < rng_p4 && idx_h1 < rng_h1 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p5; ll++)
		{
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = d_t2_1[(blk_idx_p4 * FUSION_SIZE_SLICE_2_P4 + idx_p6 + (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5 + ll + (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1 + idx_h1) * size_p5) * size_p4) * size_h7 + (threadIdx.x + l)];
		}

		// Load Input Tensor to Shared Memory
		if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p6; ll++)
		{
            sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_2_X] = d_v2_1[blk_idx_h3 * FUSION_SIZE_SLICE_2_H3 + idx_h3 + (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2 + idx_h2 + (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6 + ll) * size_h2) * size_h3 + (threadIdx.y + l) * list_stride_v2[6]];
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
	for (int l = 0; l < size_internal && kernel_2 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		if (idx_p6 < rng_p4 && idx_h1 < rng_h2 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p5; ll++)
		{
			sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = d_t2_2[(blk_idx_p4 * FUSION_SIZE_SLICE_2_P4 + idx_p6 + (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5 + ll + (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2 + idx_h1) * size_p5) * size_p4) * size_h7 + (threadIdx.x + l)];
		}

		// Load Input Tensor to Shared Memory
		if (idx_h3 < rng_h3 && idx_h2 < rng_h1 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p6; ll++)
		{
			sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_2_X] = d_v2_2[(blk_idx_h3 * FUSION_SIZE_SLICE_2_H3 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6 + ll) * size_h1) * size_h3) + (threadIdx.y + l) * list_stride_v2[7]];
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
	for (int l = 0; l < size_internal && kernel_3 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		if (idx_p6 < rng_p4 && idx_h1 < rng_h3 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p5; ll++)
		{
			sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = d_t2_3[(blk_idx_p4 * FUSION_SIZE_SLICE_2_P4 + idx_p6 + (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5 + ll + (blk_idx_h3 * FUSION_SIZE_SLICE_2_H3 + idx_h1) * size_p5) * size_p4) * size_h7 + (threadIdx.x + l)];
		}

		// Load Input Tensor to Shared Memory
		if (idx_h3 < rng_h2 && idx_h2 < rng_h1 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p6; ll++)
		{
			sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_2_X] = d_v2_3[(blk_idx_h2 * FUSION_SIZE_SLICE_2_H2 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1 + idx_h2 + (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6 + ll) * size_h1) * size_h2) + (threadIdx.y + l) * list_stride_v2[8]];
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
	for (int l = 0; l < size_internal && kernel_4 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		if (idx_p6 < rng_p6 && idx_h1 < rng_h1 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p5; ll++)
		{
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_4[(blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h1) * size_p6) * size_p5) * size_h7 + (threadIdx.x + l)];
           
		}

		// Load Input Tensor to Shared Memory
		if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p4; ll++)
		{
            sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = d_v2_4[(blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h2 + (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll) * size_h2) * size_h3) + (threadIdx.y + l) * list_stride_v2[0]];
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
	for (int l = 0; l < size_internal && kernel_5 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		if (idx_p6 < rng_p6 && idx_h1 < rng_h2 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p5; ll++)
		{
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_5[(blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h1) * size_p6) * size_p5) * size_h7 + (threadIdx.x + l)]; 
		}

		// Load Input Tensor to Shared Memory
		if (idx_h3 < rng_h3 && idx_h2 < rng_h1 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p4; ll++)
		{
            sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = d_v2_5[(blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll) * size_h1) * size_h3) + (threadIdx.y + l) * list_stride_v2[1]];
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
	for (int l = 0; l < size_internal && kernel_6 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1 //63, 21
		if (idx_p6 < rng_p6 && idx_h1 < rng_h3 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p5; ll++)
		{
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_6[(blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h1) * size_p6) * size_p5) * size_h7 + (threadIdx.x + l)];
		}

		// Load Input Tensor to Shared Memory
		if (idx_h3 < rng_h2 && idx_h2 < rng_h1 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p4; ll++)
		{
            sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = d_v2_6[(blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll) * size_h1) * size_h2) + (threadIdx.y + l) * list_stride_v2[2]];
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
	for (int l = 0; l < size_internal && kernel_7 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		if (idx_p6 < rng_p6 && idx_h1 < rng_h1 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p4; ll++)
		{
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_7[(blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h1) * size_p6) * size_p4) * size_h7 + (threadIdx.x + l)];
		}

		// Load Input Tensor to Shared Memory
		if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p5; ll++)
		{
            sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = d_v2_7[(blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h2 + (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll) * size_h2) * size_h3) + (threadIdx.y + l) * list_stride_v2[3]];
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
	for (int l = 0; l < size_internal && kernel_8 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		if (idx_p6 < rng_p6 && idx_h1 < rng_h2 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p4; ll++)
		{
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_8[(blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h1) * size_p6) * size_p4) * size_h7 + (threadIdx.x + l)];
		}

		// Load Input Tensor to Shared Memory
		if (idx_h3 < rng_h3 && idx_h2 < rng_h1 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p5; ll++)
		{
            sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = d_v2_8[(blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll) * size_h1) * size_h3) + (threadIdx.y + l) * list_stride_v2[4]];
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
	for (int l = 0; l < size_internal && kernel_9 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		if (idx_p6 < rng_p6 && idx_h1 < rng_h3 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p4; ll++)
		{
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_9[(blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h1) * size_p6) * size_p4) * size_h7 + (threadIdx.x + l)];
		}

		// Load Input Tensor to Shared Memory
		if (idx_h3 < rng_h2 && idx_h2 < rng_h1 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p5; ll++)
		{
			sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = d_v2_9[(blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll) * size_h1) * size_h2) + (threadIdx.y + l) * list_stride_v2[5]];
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


	// Store Results (Registers) to Global Memory
	// Part: Generalized Threads
	// Part: Generalized Register-Tiling
	if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6 && idx_h1 < rng_h1)
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			if(i < rng_p4 && j < rng_p5)
			t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] = reg_tile[i][j];
		}
	}
}

// created by tc_gen_code_Kernel()
__global__ void kernel_ccsdT_sd1_fully_fused_partial_full(double* t3, 
							double* d_t2_4, double* d_t2_5, double* d_t2_6, double* d_t2_7, double* d_t2_8, double* d_t2_9, 
							double* d_v2_4, double* d_v2_5, double* d_v2_6, double* d_v2_7, double* d_v2_8, double* d_v2_9, 
                            double* d_t2_1, double* d_t2_2, double* d_t2_3, 
							double* d_v2_1, double* d_v2_2, double* d_v2_3, 
							int size_h3,    int size_h2,    int size_h1,    int size_p6,    int size_p5,    int size_p4,    int size_h7, 
                            int numBlk_h3,  int numBlk_h2,  int numBlk_h1,  int numBlk_p6,  int numBlk_p5,  int numBlk_p4,
                            int kernel_1, int kernel_2, int kernel_3, 
                            int kernel_4, int kernel_5, int kernel_6, 
                            int kernel_7, int kernel_8, int kernel_9,
							int stride_reg_x, int stride_reg_y,
							int size_internal)
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

	int tmp_blkIdx;        
    int blk_idx_p4  = blockIdx.x / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);
    tmp_blkIdx      = blockIdx.x % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);

    int blk_idx_p5  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);

    int blk_idx_p6  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1);

    int blk_idx_h1 = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2);
    tmp_blkIdx     = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2);

    int blk_idx_h2 = (tmp_blkIdx) / (numBlk_h3);
    int blk_idx_h3 = blockIdx.x % (numBlk_h3);

    int t3_base_thread = blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + 
                        (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h2 + 
                        (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h1 + 
                        (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + 
                        (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + 
                        (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;


	double temp_av;
	double temp_bv[4];
	double reg_tile[4][4];

	for (int i = 0; i < 4; i++)
	for (int j = 0; j < 4; j++)
	reg_tile[i][j] = 0.0;

	//
	//	sd1_1,2,3
	//
	// tensor contraction
	#pragma unroll 1
	for (int l = 0; l < size_internal && kernel_1 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		if (threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < 4; ll++)
		{
			sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = d_t2_1[(blk_idx_p4 * FUSION_SIZE_SLICE_2_P4 + idx_p6 + (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5 + ll + (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1 + idx_h1) * size_p5) * size_p4) * size_h7 + (threadIdx.x + l)];
		}

		// Load Input Tensor to Shared Memory
		if (threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < 4; ll++)
		{
			sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_2_X] = d_v2_1[blk_idx_h3 * FUSION_SIZE_SLICE_2_H3 + idx_h3 + (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2 + idx_h2 + (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6 + ll) * size_h2) * size_h3 + (threadIdx.y + l) * list_stride_v2[6]];
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
	for (int l = 0; l < size_internal && kernel_2 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		if (threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < 4; ll++)
		{
			sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = d_t2_2[(blk_idx_p4 * FUSION_SIZE_SLICE_2_P4 + idx_p6 + (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5 + ll + (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2 + idx_h1) * size_p5) * size_p4) * size_h7 + (threadIdx.x + l)];
		}

		// Load Input Tensor to Shared Memory
		if (threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < 4; ll++)
		{
			sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_2_X] = d_v2_2[(blk_idx_h3 * FUSION_SIZE_SLICE_2_H3 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6 + ll) * size_h1) * size_h3) + (threadIdx.y + l) * list_stride_v2[7]];
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
	for (int l = 0; l < size_internal && kernel_3 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		if (threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < 4; ll++)
		{
			sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = d_t2_3[(blk_idx_p4 * FUSION_SIZE_SLICE_2_P4 + idx_p6 + (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5 + ll + (blk_idx_h3 * FUSION_SIZE_SLICE_2_H3 + idx_h1) * size_p5) * size_p4) * size_h7 + (threadIdx.x + l)];
		}

		// Load Input Tensor to Shared Memory
		if (threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < 4; ll++)
		{
			sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_2_X] = d_v2_3[(blk_idx_h2 * FUSION_SIZE_SLICE_2_H2 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1 + idx_h2 + (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6 + ll) * size_h1) * size_h2) + (threadIdx.y + l) * list_stride_v2[8]];
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
	for (int l = 0; l < size_internal && kernel_4 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		if (threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < 4; ll++)
		{
			sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_4[(blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h1) * size_p6) * size_p5) * size_h7 + (threadIdx.x + l)];
		}

		// Load Input Tensor to Shared Memory
		if (threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < 4; ll++)
		{
			sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = d_v2_4[(blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h2 + (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll) * size_h2) * size_h3) + (threadIdx.y + l) * list_stride_v2[0]];
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
	for (int l = 0; l < size_internal && kernel_5 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		if (threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < 4; ll++)
		{
			sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_5[(blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h1) * size_p6) * size_p5) * size_h7 + (threadIdx.x + l)]; 
		}

		// Load Input Tensor to Shared Memory
		if (threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < 4; ll++)
		{
			sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = d_v2_5[(blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll) * size_h1) * size_h3) + (threadIdx.y + l) * list_stride_v2[1]];
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
	for (int l = 0; l < size_internal && kernel_6 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1 //63, 21
		if (threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < 4; ll++)
		{
			sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_6[(blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h1) * size_p6) * size_p5) * size_h7 + (threadIdx.x + l)];
		}

		// Load Input Tensor to Shared Memory
		if (threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < 4; ll++)
		{
			sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = d_v2_6[(blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll) * size_h1) * size_h2) + (threadIdx.y + l) * list_stride_v2[2]];
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
	for (int l = 0; l < size_internal && kernel_7 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		if (threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < 4; ll++)
		{
			sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_7[(blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h1) * size_p6) * size_p4) * size_h7 + (threadIdx.x + l)];
		}

		// Load Input Tensor to Shared Memory
		if (threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < 4; ll++)
		{
			sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = d_v2_7[(blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h2 + (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll) * size_h2) * size_h3) + (threadIdx.y + l) * list_stride_v2[3]];
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
	for (int l = 0; l < size_internal && kernel_8 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		if (threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < 4; ll++)
		{
			sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_8[(blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h1) * size_p6) * size_p4) * size_h7 + (threadIdx.x + l)];
		}

		// Load Input Tensor to Shared Memory
		if (threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < 4; ll++)
		{
			sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = d_v2_8[(blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll) * size_h1) * size_h3) + (threadIdx.y + l) * list_stride_v2[4]];
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
	for (int l = 0; l < size_internal && kernel_9 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		if (threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < 4; ll++)
		{
			sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_9[(blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h1) * size_p6) * size_p4) * size_h7 + (threadIdx.x + l)];
		}

		// Load Input Tensor to Shared Memory
		if (threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < 4; ll++)
		{
			sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = d_v2_9[(blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll) * size_h1) * size_h2) + (threadIdx.y + l) * list_stride_v2[5]];
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


	// Store Results (Registers) to Global Memory
	// Part: Generalized Threads
	// Part: Generalized Register-Tiling
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] = reg_tile[i][j];
		}
	}
}

// created by tc_gen_code_Kernel()
__global__ void kernel_ccsdT_sd1_fully_fused_full_full(double* t3, 
							double* d_t2_4, double* d_t2_5, double* d_t2_6, double* d_t2_7, double* d_t2_8, double* d_t2_9, 
							double* d_v2_4, double* d_v2_5, double* d_v2_6, double* d_v2_7, double* d_v2_8, double* d_v2_9, 
                            double* d_t2_1, double* d_t2_2, double* d_t2_3, 
							double* d_v2_1, double* d_v2_2, double* d_v2_3, 
							int size_h3,    int size_h2,    int size_h1,    int size_p6,    int size_p5,    int size_p4,    int size_h7, 
                            int numBlk_h3,  int numBlk_h2,  int numBlk_h1,  int numBlk_p6,  int numBlk_p5,  int numBlk_p4,
                            int kernel_1, int kernel_2, int kernel_3, 
                            int kernel_4, int kernel_5, int kernel_6, 
                            int kernel_7, int kernel_8, int kernel_9,
							int stride_reg_x, int stride_reg_y,
							int size_internal)
{
	// For Shared Memory,
	__shared__ double sm_a[16][64 + 1];
	__shared__ double sm_b[16][64 + 1];

    // should support for non-full tiles
	int idx_h3 = threadIdx.x % FUSION_SIZE_SLICE_1_H3;
	int idx_h2 = threadIdx.x / FUSION_SIZE_SLICE_1_H3;
	int idx_p6 = threadIdx.y % FUSION_SIZE_SLICE_1_P6;
	int idx_h1 = threadIdx.y / FUSION_SIZE_SLICE_1_P6;

	int tmp_blkIdx;        
    int blk_idx_p4  = blockIdx.x / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);
    tmp_blkIdx      = blockIdx.x % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);

    int blk_idx_p5  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);

    int blk_idx_p6  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1);

    int blk_idx_h1 = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2);
    tmp_blkIdx     = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2);

    int blk_idx_h2 = (tmp_blkIdx) / (numBlk_h3);
    int blk_idx_h3 = blockIdx.x % (numBlk_h3);

    int t3_base_thread = blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + 
                        (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h2 + 
                        (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h1 + 
                        (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + 
                        (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + 
                        (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;


	double temp_av;
	double temp_bv[4];
	double reg_tile[4][4];

	for (int i = 0; i < 4; i++)
	for (int j = 0; j < 4; j++)
	reg_tile[i][j] = 0.0;

	//
	//	sd1_1,2,3
	//
	// tensor contraction
	#pragma unroll 1
	for (int l = 0; l < size_internal && kernel_1 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		for (int ll = 0; ll < 4; ll++)
		{
			sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = d_t2_1[(blk_idx_p4 * FUSION_SIZE_SLICE_2_P4 + idx_p6 + (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5 + ll + (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1 + idx_h1) * size_p5) * size_p4) * size_h7 + (threadIdx.x + l)];
			sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_2_X] = d_v2_1[blk_idx_h3 * FUSION_SIZE_SLICE_2_H3 + idx_h3 + (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2 + idx_h2 + (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6 + ll) * size_h2) * size_h3 + (threadIdx.y + l) * list_stride_v2[6]];
		}
		__syncthreads();

		// Cross-Product: -1
		// Part: Generalized Threads
		for (int ll = 0; ll < FUSION_SIZE_INT_UNIT; ll++)
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
	#pragma unroll 1
	for (int l = 0; l < size_internal && kernel_2 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
	
		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		for (int ll = 0; ll < 4; ll++)
		{
			sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = d_t2_2[(blk_idx_p4 * FUSION_SIZE_SLICE_2_P4 + idx_p6 + (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5 + ll + (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2 + idx_h1) * size_p5) * size_p4) * size_h7 + (threadIdx.x + l)];
			sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_2_X] = d_v2_2[(blk_idx_h3 * FUSION_SIZE_SLICE_2_H3 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6 + ll) * size_h1) * size_h3) + (threadIdx.y + l) * list_stride_v2[7]];
		}
		__syncthreads();

		// Cross-Product: -1
		// Part: Generalized Threads
		for (int ll = 0; ll < FUSION_SIZE_INT_UNIT; ll++)
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
	#pragma unroll 1
	for (int l = 0; l < size_internal && kernel_3 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		for (int ll = 0; ll < 4; ll++)
		{
			sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = d_t2_3[(blk_idx_p4 * FUSION_SIZE_SLICE_2_P4 + idx_p6 + (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5 + ll + (blk_idx_h3 * FUSION_SIZE_SLICE_2_H3 + idx_h1) * size_p5) * size_p4) * size_h7 + (threadIdx.x + l)];
			sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_2_X] = d_v2_3[(blk_idx_h2 * FUSION_SIZE_SLICE_2_H2 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1 + idx_h2 + (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6 + ll) * size_h1) * size_h2) + (threadIdx.y + l) * list_stride_v2[8]];
		}
		__syncthreads();

		// Cross-Product: -1
		// Part: Generalized Threads
		for (int ll = 0; ll < FUSION_SIZE_INT_UNIT; ll++)
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
	#pragma unroll 1
	for (int l = 0; l < size_internal && kernel_4 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		for (int ll = 0; ll < 4; ll++)
		{
			sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_4[(blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h1) * size_p6) * size_p5) * size_h7 + (threadIdx.x + l)];
			sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = d_v2_4[(blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h2 + (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll) * size_h2) * size_h3) + (threadIdx.y + l) * list_stride_v2[0]];
		}
		__syncthreads();

		// Cross-Product: -1
		// Part: Generalized Threads
		for (int ll = 0; ll < FUSION_SIZE_INT_UNIT; ll++)
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
	#pragma unroll 1
	for (int l = 0; l < size_internal && kernel_5 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		for (int ll = 0; ll < 4; ll++)
		{
			sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_5[(blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h1) * size_p6) * size_p5) * size_h7 + (threadIdx.x + l)]; 
			sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = d_v2_5[(blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll) * size_h1) * size_h3) + (threadIdx.y + l) * list_stride_v2[1]];
		}
		__syncthreads();

		// Cross-Product: -1
		// Part: Generalized Threads
		for (int ll = 0; ll < FUSION_SIZE_INT_UNIT; ll++)
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
	#pragma unroll 1
	for (int l = 0; l < size_internal && kernel_6 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1 //63, 21
		for (int ll = 0; ll < 4; ll++)
		{
			sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_6[(blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h1) * size_p6) * size_p5) * size_h7 + (threadIdx.x + l)];
			sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = d_v2_6[(blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll) * size_h1) * size_h2) + (threadIdx.y + l) * list_stride_v2[2]];
		}
		__syncthreads();

		// Cross-Product: -1
		// Part: Generalized Threads
		for (int ll = 0; ll < FUSION_SIZE_INT_UNIT; ll++)
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
	#pragma unroll 1
	for (int l = 0; l < size_internal && kernel_7 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		for (int ll = 0; ll < 4; ll++)
		{
			sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_7[(blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h1) * size_p6) * size_p4) * size_h7 + (threadIdx.x + l)];
			sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = d_v2_7[(blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h2 + (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll) * size_h2) * size_h3) + (threadIdx.y + l) * list_stride_v2[3]];
		}
		__syncthreads();

		// Cross-Product: -1
		// Part: Generalized Threads
		for (int ll = 0; ll < FUSION_SIZE_INT_UNIT; ll++)
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
	#pragma unroll 1
	for (int l = 0; l < size_internal && kernel_8 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		for (int ll = 0; ll < 4; ll++)
		{
			sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_8[(blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h1) * size_p6) * size_p4) * size_h7 + (threadIdx.x + l)];
			sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = d_v2_8[(blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll) * size_h1) * size_h3) + (threadIdx.y + l) * list_stride_v2[4]];
		}
		__syncthreads();

		// Cross-Product: -1
		// Part: Generalized Threads
		for (int ll = 0; ll < FUSION_SIZE_INT_UNIT; ll++)
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
	#pragma unroll 1
	for (int l = 0; l < size_internal && kernel_9 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		for (int ll = 0; ll < 4; ll++)
		{
			sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_9[(blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h1) * size_p6) * size_p4) * size_h7 + (threadIdx.x + l)];
			sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = d_v2_9[(blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll) * size_h1) * size_h2) + (threadIdx.y + l) * list_stride_v2[5]];
		}
		__syncthreads();

		// Cross-Product: -1
		// Part: Generalized Threads
		for (int ll = 0; ll < FUSION_SIZE_INT_UNIT; ll++)
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


	// Store Results (Registers) to Global Memory
	// Part: Generalized Threads
	// Part: Generalized Register-Tiling
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] = reg_tile[i][j];
		}
	}
}

//
//  Partially-Fused Kernels (external_internal)
//      (1) sd1_456789
//      (2) sd1_123
//
__global__ void kernel_ccsdT_sd1_456789_partial_partial(double* t3, 
        double* d_t2_4, double* d_t2_5, double* d_t2_6, double* d_t2_7, double* d_t2_8, double* d_t2_9, 
        double* d_v2_4, double* d_v2_5, double* d_v2_6, double* d_v2_7, double* d_v2_8, double* d_v2_9, 
        int size_h3,    int size_h2,    int size_h1,    int size_p6,    int size_p5,    int size_p4,    int size_h7, 
        int numBlk_h3,  int numBlk_h2,  int numBlk_h1,  int numBlk_p6,  int numBlk_p5,  int numBlk_p4,
        int kernel_1, int kernel_2, int kernel_3, 
        int kernel_4, int kernel_5, int kernel_6, 
        int kernel_7, int kernel_8, int kernel_9,
        int stride_reg_x, int stride_reg_y,
        int size_internal)
{
	// For Shared Memory,
	__shared__ double sm_a[16][64 + 1];
	__shared__ double sm_b[16][64 + 1];

	//int l_idx_t3                = threadIdx.x + threadIdx.y * FUSION_SIZE_TB_1_X;
	int internal_upperbound     = 0;
	int internal_offset;

	// should support for non-full tiles
	int idx_h3 = threadIdx.x % FUSION_SIZE_SLICE_1_H3;
	int idx_h2 = threadIdx.x / FUSION_SIZE_SLICE_1_H3;
	int idx_p6 = threadIdx.y % FUSION_SIZE_SLICE_1_P6;
    int idx_h1 = threadIdx.y / FUSION_SIZE_SLICE_1_P6;

    int tmp_blkIdx;        
    int blk_idx_p4  = blockIdx.x / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);
    tmp_blkIdx      = blockIdx.x % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);

    int blk_idx_p5  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);

    int blk_idx_p6  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1);

    int blk_idx_h1 = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2);
    tmp_blkIdx     = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2);

    int blk_idx_h2 = (tmp_blkIdx) / (numBlk_h3);
    int blk_idx_h3 = blockIdx.x % (numBlk_h3);

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

    int t3_base_thread = blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + 
                        (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h2 + 
                        (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h1 + 
                        (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + 
                        (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + 
                        (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;

	double temp_av;
	double temp_bv[4];
	double reg_tile[4][4];

	for (int i = 0; i < 4; i++)
	for (int j = 0; j < 4; j++)
	reg_tile[i][j] = 0.0;

	// tensor contraction
	#pragma unroll 1
	for (int l = 0; l < size_internal && kernel_4 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		if (idx_p6 < rng_p6 && idx_h1 < rng_h1 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p5; ll++)
		{
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_4[(blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h1) * size_p6) * size_p5) * size_h7 + (threadIdx.x + l)];
           
		}

		// Load Input Tensor to Shared Memory
		if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p4; ll++)
		{
            sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = d_v2_4[(blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h2 + (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll) * size_h2) * size_h3) + (threadIdx.y + l) * list_stride_v2[0]];
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
	for (int l = 0; l < size_internal && kernel_5 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		if (idx_p6 < rng_p6 && idx_h1 < rng_h2 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p5; ll++)
		{
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_5[(blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h1) * size_p6) * size_p5) * size_h7 + (threadIdx.x + l)]; 
		}

		// Load Input Tensor to Shared Memory
		if (idx_h3 < rng_h3 && idx_h2 < rng_h1 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p4; ll++)
		{
            sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = d_v2_5[(blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll) * size_h1) * size_h3) + (threadIdx.y + l) * list_stride_v2[1]];
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
	for (int l = 0; l < size_internal && kernel_6 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1 //63, 21
		if (idx_p6 < rng_p6 && idx_h1 < rng_h3 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p5; ll++)
		{
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_6[(blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h1) * size_p6) * size_p5) * size_h7 + (threadIdx.x + l)];
		}

		// Load Input Tensor to Shared Memory
		if (idx_h3 < rng_h2 && idx_h2 < rng_h1 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p4; ll++)
		{
            sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = d_v2_6[(blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll) * size_h1) * size_h2) + (threadIdx.y + l) * list_stride_v2[2]];
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
	for (int l = 0; l < size_internal && kernel_7 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		if (idx_p6 < rng_p6 && idx_h1 < rng_h1 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p4; ll++)
		{
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_7[(blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h1) * size_p6) * size_p4) * size_h7 + (threadIdx.x + l)];
		}

		// Load Input Tensor to Shared Memory
		if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p5; ll++)
		{
            sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = d_v2_7[(blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h2 + (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll) * size_h2) * size_h3) + (threadIdx.y + l) * list_stride_v2[3]];
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
	for (int l = 0; l < size_internal && kernel_8 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		if (idx_p6 < rng_p6 && idx_h1 < rng_h2 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p4; ll++)
		{
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_8[(blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h1) * size_p6) * size_p4) * size_h7 + (threadIdx.x + l)];
		}

		// Load Input Tensor to Shared Memory
		if (idx_h3 < rng_h3 && idx_h2 < rng_h1 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p5; ll++)
		{
            sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = d_v2_8[(blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll) * size_h1) * size_h3) + (threadIdx.y + l) * list_stride_v2[4]];
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
	for (int l = 0; l < size_internal && kernel_9 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		if (idx_p6 < rng_p6 && idx_h1 < rng_h3 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p4; ll++)
		{
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_9[(blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h1) * size_p6) * size_p4) * size_h7 + (threadIdx.x + l)];
		}

		// Load Input Tensor to Shared Memory
		if (idx_h3 < rng_h2 && idx_h2 < rng_h1 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p5; ll++)
		{
			sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = d_v2_9[(blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll) * size_h1) * size_h2) + (threadIdx.y + l) * list_stride_v2[5]];
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


	// Store Results (Registers) to Global Memory
	// Part: Generalized Threads
	// Part: Generalized Register-Tiling
	if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6 && idx_h1 < rng_h1)
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			if(i < rng_p4 && j < rng_p5)
			t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] = reg_tile[i][j];
		}
	}
}

// created by tc_gen_code_Kernel()
__global__ void kernel_ccsdT_sd1_456789_partial_full(double* t3, 
        double* d_t2_4, double* d_t2_5, double* d_t2_6, double* d_t2_7, double* d_t2_8, double* d_t2_9, 
        double* d_v2_4, double* d_v2_5, double* d_v2_6, double* d_v2_7, double* d_v2_8, double* d_v2_9, 
        int size_h3,    int size_h2,    int size_h1,    int size_p6,    int size_p5,    int size_p4,    int size_h7, 
        int numBlk_h3,  int numBlk_h2,  int numBlk_h1,  int numBlk_p6,  int numBlk_p5,  int numBlk_p4,
        int kernel_1, int kernel_2, int kernel_3, 
        int kernel_4, int kernel_5, int kernel_6, 
        int kernel_7, int kernel_8, int kernel_9,
        int stride_reg_x, int stride_reg_y,
        int size_internal)
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

    int tmp_blkIdx;        
    int blk_idx_p4  = blockIdx.x / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);
    tmp_blkIdx      = blockIdx.x % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);

    int blk_idx_p5  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);

    int blk_idx_p6  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1);

    int blk_idx_h1 = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2);
    tmp_blkIdx     = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2);

    int blk_idx_h2 = (tmp_blkIdx) / (numBlk_h3);
    int blk_idx_h3 = blockIdx.x % (numBlk_h3);

    int t3_base_thread = blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + 
                        (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h2 + 
                        (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h1 + 
                        (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + 
                        (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + 
                        (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;

	double temp_av;
	double temp_bv[4];
	double reg_tile[4][4];

	for (int i = 0; i < 4; i++)
	for (int j = 0; j < 4; j++)
	reg_tile[i][j] = 0.0;

	// tensor contraction
	#pragma unroll 1
	for (int l = 0; l < size_internal && kernel_4 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		if (threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < 4; ll++)
		{
			sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_4[(blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h1) * size_p6) * size_p5) * size_h7 + (threadIdx.x + l)];
		}

		// Load Input Tensor to Shared Memory
		if (threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < 4; ll++)
		{
			sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = d_v2_4[(blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h2 + (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll) * size_h2) * size_h3) + (threadIdx.y + l) * list_stride_v2[0]];
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
	for (int l = 0; l < size_internal && kernel_5 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		if (threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < 4; ll++)
		{
			sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_5[(blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h1) * size_p6) * size_p5) * size_h7 + (threadIdx.x + l)]; 
		}

		// Load Input Tensor to Shared Memory
		if (threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < 4; ll++)
		{
			sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = d_v2_5[(blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll) * size_h1) * size_h3) + (threadIdx.y + l) * list_stride_v2[1]];
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
	for (int l = 0; l < size_internal && kernel_6 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1 //63, 21
		if (threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < 4; ll++)
		{
			sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_6[(blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h1) * size_p6) * size_p5) * size_h7 + (threadIdx.x + l)];
		}

		// Load Input Tensor to Shared Memory
		if (threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < 4; ll++)
		{
			sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = d_v2_6[(blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll) * size_h1) * size_h2) + (threadIdx.y + l) * list_stride_v2[2]];
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
	for (int l = 0; l < size_internal && kernel_7 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		if (threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < 4; ll++)
		{
			sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_7[(blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h1) * size_p6) * size_p4) * size_h7 + (threadIdx.x + l)];
		}

		// Load Input Tensor to Shared Memory
		if (threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < 4; ll++)
		{
			sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = d_v2_7[(blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h2 + (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll) * size_h2) * size_h3) + (threadIdx.y + l) * list_stride_v2[3]];
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
	for (int l = 0; l < size_internal && kernel_8 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		if (threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < 4; ll++)
		{
			sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_8[(blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h1) * size_p6) * size_p4) * size_h7 + (threadIdx.x + l)];
		}

		// Load Input Tensor to Shared Memory
		if (threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < 4; ll++)
		{
			sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = d_v2_8[(blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll) * size_h1) * size_h3) + (threadIdx.y + l) * list_stride_v2[4]];
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
	for (int l = 0; l < size_internal && kernel_9 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		if (threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < 4; ll++)
		{
			sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_9[(blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h1) * size_p6) * size_p4) * size_h7 + (threadIdx.x + l)];
		}

		// Load Input Tensor to Shared Memory
		if (threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < 4; ll++)
		{
			sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = d_v2_9[(blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll) * size_h1) * size_h2) + (threadIdx.y + l) * list_stride_v2[5]];
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


	// Store Results (Registers) to Global Memory
	// Part: Generalized Threads
	// Part: Generalized Register-Tiling
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] = reg_tile[i][j];
		}
	}
}

// created by tc_gen_code_Kernel()
__global__ void kernel_ccsdT_sd1_456789_full_full(double* t3, 
        double* d_t2_4, double* d_t2_5, double* d_t2_6, double* d_t2_7, double* d_t2_8, double* d_t2_9, 
        double* d_v2_4, double* d_v2_5, double* d_v2_6, double* d_v2_7, double* d_v2_8, double* d_v2_9, 
        int size_h3,    int size_h2,    int size_h1,    int size_p6,    int size_p5,    int size_p4,    int size_h7, 
        int numBlk_h3,  int numBlk_h2,  int numBlk_h1,  int numBlk_p6,  int numBlk_p5,  int numBlk_p4,
        int kernel_1, int kernel_2, int kernel_3, 
        int kernel_4, int kernel_5, int kernel_6, 
        int kernel_7, int kernel_8, int kernel_9,
        int stride_reg_x, int stride_reg_y,
        int size_internal)
{
	// For Shared Memory,
	__shared__ double sm_a[16][64 + 1];
	__shared__ double sm_b[16][64 + 1];

    // should support for non-full tiles
	int idx_h3 = threadIdx.x % FUSION_SIZE_SLICE_1_H3;
	int idx_h2 = threadIdx.x / FUSION_SIZE_SLICE_1_H3;
	int idx_p6 = threadIdx.y % FUSION_SIZE_SLICE_1_P6;
    int idx_h1 = threadIdx.y / FUSION_SIZE_SLICE_1_P6;

    int tmp_blkIdx;        
    int blk_idx_p4  = blockIdx.x / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);
    tmp_blkIdx      = blockIdx.x % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);

    int blk_idx_p5  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);

    int blk_idx_p6  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1);

    int blk_idx_h1 = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2);
    tmp_blkIdx     = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2);

    int blk_idx_h2 = (tmp_blkIdx) / (numBlk_h3);
    int blk_idx_h3 = blockIdx.x % (numBlk_h3);

    int t3_base_thread = blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + 
                        (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h2 + 
                        (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h1 + 
                        (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + 
                        (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + 
                        (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;

	double temp_av;
	double temp_bv[4];
	double reg_tile[4][4];

	for (int i = 0; i < 4; i++)
	for (int j = 0; j < 4; j++)
	reg_tile[i][j] = 0.0;

	// tensor contraction
	#pragma unroll 1
	for (int l = 0; l < size_internal && kernel_4 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		for (int ll = 0; ll < 4; ll++)
		{
            //  threadIdx.y --> idx_p6 (%) and idx_h1 (/) over ll (4)
            //  t2_4: h7,p5,p6,h1
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_4[(blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h1) * size_p6) * size_p5) * size_h7 + (threadIdx.x + l)];
           
            //  threadIdx.x --> idx_h3 (%) and idx_h2 (/) over ll (4)
            //  v2_4: h3,h2,p4,h7   // h3 (p6), h2 (h1), p4 (ll)
            sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = d_v2_4[(blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h2 + (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll) * size_h2) * size_h3) + (threadIdx.y + l) * list_stride_v2[0]];
        }
		__syncthreads();

		// Cross-Product: -1
		// Part: Generalized Threads
		for (int ll = 0; ll < FUSION_SIZE_INT_UNIT; ll++)
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
	#pragma unroll 1
	for (int l = 0; l < size_internal && kernel_5 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Load Input Tensor to Shared Memory: 16:16
        // # of Internal Indices: 1
		for (int ll = 0; ll < 4; ll++)
		{
            //  threadIdx.y --> idx_p6 (%) and idx_h1 (/) over ll (4)
            //  t2_5: h7,p5,p6,h2 >>> p6,h2,p5
            //sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_5[d_t2_5_addr[threadIdx.y + ll * FUSION_SIZE_TB_1_Y + blockIdx.x * (FUSION_SIZE_SLICE_1_P5 * FUSION_SIZE_SLICE_1_P6 * FUSION_SIZE_SLICE_1_H2)] + (threadIdx.x + l)];
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_5[(blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h1) * size_p6) * size_p5) * size_h7 + (threadIdx.x + l)]; 


            //  threadIdx.x --> idx_h3 (%) and idx_h2 (/) over ll (4)
            //  v2_5: h7,h3,h1,p4   // h3 (h3), h1 (h2), p4 (ll)
			//sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = d_v2_5[d_v2_5_addr[threadIdx.x + ll * FUSION_SIZE_TB_1_X + blockIdx.x * (FUSION_SIZE_SLICE_1_H3 * FUSION_SIZE_SLICE_1_H1 * FUSION_SIZE_SLICE_1_P4)] + (threadIdx.y + l) * list_stride_v2[1]];
            sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = d_v2_5[(blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll) * size_h1) * size_h3) + (threadIdx.y + l) * list_stride_v2[1]];
        }
		__syncthreads();

		// Cross-Product: -1
		// Part: Generalized Threads
		for (int ll = 0; ll < FUSION_SIZE_INT_UNIT; ll++)
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
	#pragma unroll 1
	for (int l = 0; l < size_internal && kernel_6 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1 //63, 21
		for (int ll = 0; ll < 4; ll++)
		{
            //  threadIdx.y --> idx_p6 (%) and idx_h1 (/) over ll (4)
            //  t2_6: h7,p5,p6,h3 >>> p6,h3,p5  // p6 (p6), h3 (h1), p5 (ll)
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_6[(blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h1) * size_p6) * size_p5) * size_h7 + (threadIdx.x + l)];
            
            //  threadIdx.x --> idx_h3 (%) and idx_h2 (/) over ll (4)
            //  v2_6: h2,h1,p4,h7 >>> h2,h1,p4  // h2 (h3), h1 (h2), p4 (ll)
            sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = d_v2_6[(blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll) * size_h1) * size_h2) + (threadIdx.y + l) * list_stride_v2[2]];
		}
		__syncthreads();

		// Cross-Product: -1
		// Part: Generalized Threads
		for (int ll = 0; ll < FUSION_SIZE_INT_UNIT; ll++)
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
	#pragma unroll 1
	for (int l = 0; l < size_internal && kernel_7 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		for (int ll = 0; ll < 4; ll++)
		{
            //  threadIdx.y --> idx_p6 (%) and idx_h1 (/) over ll (4)
            //  t2_7: h7,p4,p6,h1 >>> p6,h1,p4  // p6 (p6), h1 (h1), p4 (ll)
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_7[(blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h1) * size_p6) * size_p4) * size_h7 + (threadIdx.x + l)];
            
            //  threadIdx.x --> idx_h3 (%) and idx_h2 (/) over ll (4)
            //  v2_7: h3,h2,p5,h7 >>> h3,h2,p5  // h3 (h3), h2 (h2), p5 (ll)
            sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = d_v2_7[(blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h2 + (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll) * size_h2) * size_h3) + (threadIdx.y + l) * list_stride_v2[3]];
		}
		__syncthreads();

		// Cross-Product: -1
		// Part: Generalized Threads
		for (int ll = 0; ll < FUSION_SIZE_INT_UNIT; ll++)
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
	#pragma unroll 1
	for (int l = 0; l < size_internal && kernel_8 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		for (int ll = 0; ll < 4; ll++)
		{
            //  threadIdx.y --> idx_p6 (%) and idx_h1 (/) over ll (4)
            //  t2_8: h7,p4,p6,h2 >>> p6,h2,p4  // p6 (p6), h2 (h1), p4 (ll)
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_8[(blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h1) * size_p6) * size_p4) * size_h7 + (threadIdx.x + l)];

            //  threadIdx.x --> idx_h3 (%) and idx_h2 (/) over ll (4)
            //  v2_8: h3,h1,p5,h7 >>> h3,h1,p5  // h3 (h3), h1 (h2), p5 (ll)
            sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = d_v2_8[(blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll) * size_h1) * size_h3) + (threadIdx.y + l) * list_stride_v2[4]];
		}
		__syncthreads();

		// Cross-Product: -1
		// Part: Generalized Threads
		for (int ll = 0; ll < FUSION_SIZE_INT_UNIT; ll++)
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
	#pragma unroll 1
	for (int l = 0; l < size_internal && kernel_9 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		for (int ll = 0; ll < 4; ll++)
		{
            //  threadIdx.y --> idx_p6 (%) and idx_h1 (/) over ll (4)
            //  t2_9: h7,p4,p6,h3 >>> p6,h3,p4  // p6 (p6), h3 (h1), p4 (ll)
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_9[(blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h1) * size_p6) * size_p4) * size_h7 + (threadIdx.x + l)];
            
            //  threadIdx.x --> idx_h3 (%) and idx_h2 (/) over ll (4)
            //  v2_9: h2,h1,p5,h7 >>> h2,h1,p5  // h2 (h3), h1 (h2), p5 (ll)
			sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = d_v2_9[(blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll) * size_h1) * size_h2) + (threadIdx.y + l) * list_stride_v2[5]];
		}
		__syncthreads();

		// Cross-Product: -1
		// Part: Generalized Threads
		for (int ll = 0; ll < FUSION_SIZE_INT_UNIT; ll++)
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


	// Store Results (Registers) to Global Memory
	// Part: Generalized Threads
	// Part: Generalized Register-Tiling
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] = reg_tile[i][j];
		}
	}
}

// created by tc_gen_code_Kernel()
__global__ void kernel_ccsdT_sd1_123_partial_partial(double* t3, 
        double* d_t2_1, double* d_t2_2, double* d_t2_3, 
        double* d_v2_1, double* d_v2_2, double* d_v2_3, 
        int size_h3,    int size_h2,    int size_h1,    int size_p6,    int size_p5,    int size_p4,    int size_h7, 
        int numBlk_h3,  int numBlk_h2,  int numBlk_h1,  int numBlk_p6,  int numBlk_p5,  int numBlk_p4,
        int kernel_1, int kernel_2, int kernel_3, 
        int kernel_4, int kernel_5, int kernel_6, 
        int kernel_7, int kernel_8, int kernel_9,
        int stride_reg_x, int stride_reg_y,
        int size_internal)
{
	// For Shared Memory,
	__shared__ double sm_a[16][64 + 1];
	__shared__ double sm_b[16][64 + 1];

	int internal_upperbound   = 0;
	int internal_offset;

	// should support for non-full tiles
	int idx_h3 = threadIdx.x % FUSION_SIZE_SLICE_2_H3;
	int idx_h2 = threadIdx.x / FUSION_SIZE_SLICE_2_H3;
	int idx_p4 = threadIdx.y % FUSION_SIZE_SLICE_2_P4;
    int idx_h1 = threadIdx.y / FUSION_SIZE_SLICE_2_P4;
    
    int tmp_blkIdx;        
    int blk_idx_p4  = blockIdx.x / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);
    tmp_blkIdx      = blockIdx.x % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);

    int blk_idx_p5  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);

    int blk_idx_p6  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1);

    int blk_idx_h1 = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2);
    tmp_blkIdx     = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2);

    int blk_idx_h2 = (tmp_blkIdx) / (numBlk_h3);
    int blk_idx_h3 = (tmp_blkIdx) % (numBlk_h3);

    int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;

    if ((size_h3 - (blk_idx_h3 * FUSION_SIZE_SLICE_2_H3)) >= FUSION_SIZE_SLICE_2_H3)
    {
        rng_h3 = FUSION_SIZE_SLICE_2_H3;
    }
    else
    {
        rng_h3 = size_h3 % FUSION_SIZE_SLICE_2_H3;
    }
    
    if ((size_h2 - (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2)) >= FUSION_SIZE_SLICE_2_H2)
    {
        rng_h2 = FUSION_SIZE_SLICE_2_H2;
    }
    else
    {
        rng_h2 = size_h2 % FUSION_SIZE_SLICE_2_H2;
    }

    if ((size_h1 - (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1)) >= FUSION_SIZE_SLICE_2_H1)
    {
        rng_h1 = FUSION_SIZE_SLICE_2_H1;
    }
    else
    {
        rng_h1 = size_h1 % FUSION_SIZE_SLICE_2_H1;
    }
    
    if ((size_p6 - (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6)) >= FUSION_SIZE_SLICE_2_P6)
    {
        rng_p6 = FUSION_SIZE_SLICE_2_P6;
    }
    else
    {
        rng_p6 = size_p6 % FUSION_SIZE_SLICE_2_P6;
    }

    if ((size_p5 - (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5)) >= FUSION_SIZE_SLICE_2_P5)
    {
        rng_p5 = FUSION_SIZE_SLICE_2_P5;
    }
    else
    {
        rng_p5 = size_p5 % FUSION_SIZE_SLICE_2_P5;
    }

    if ((size_p4 - (blk_idx_p4 * FUSION_SIZE_SLICE_2_P4)) >= FUSION_SIZE_SLICE_2_P4)
    {
        rng_p4 = FUSION_SIZE_SLICE_2_P4;
    }
    else
    {
        rng_p4 = size_p4 % FUSION_SIZE_SLICE_2_P4;
    }
    
    int t3_base_thread = blk_idx_h3 * FUSION_SIZE_SLICE_2_H3 + idx_h3 + 
                        (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2 + idx_h2 + 
                        (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1 + idx_h1 + 
                        (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6 +  
                        (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5 + 
                        (blk_idx_p4 * FUSION_SIZE_SLICE_2_P4 + idx_p4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;


	double temp_av;
	double temp_bv[4];
	double reg_tile[4][4];

	for (int i = 0; i < 4; i++)
	for (int j = 0; j < 4; j++)
	reg_tile[i][j] = 0.0;

	// tensor contraction
	#pragma unroll 1
	for (int l = 0; l < size_internal && kernel_1 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		if (idx_p4 < rng_p4 && idx_h1 < rng_h1 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p5; ll++)
		{
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = d_t2_1[(blk_idx_p4 * FUSION_SIZE_SLICE_2_P4 + idx_p4 + (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5 + ll + (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1 + idx_h1) * size_p5) * size_p4) * size_h7 + (threadIdx.x + l)];
		}

		// Load Input Tensor to Shared Memory
		if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p6; ll++)
		{
            sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_2_X] = d_v2_1[blk_idx_h3 * FUSION_SIZE_SLICE_2_H3 + idx_h3 + (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2 + idx_h2 + (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6 + ll) * size_h2) * size_h3 + (threadIdx.y + l) * list_stride_v2[6]];
		}
		__syncthreads();

		// Cross-Product: -1
		// Part: Generalized Threads
		for (int ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++)
		{
			//temp_bv[0] = sm_b[ll][d_v2_1_offset[l_idx_t3] + 0];
			temp_bv[0] = sm_b[ll][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_2_H3 + 0];
			temp_bv[1] = sm_b[ll][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_2_H3 + 16];
			temp_bv[2] = sm_b[ll][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_2_H3 + 32];
			temp_bv[3] = sm_b[ll][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_2_H3 + 48];

			for (int xx = 0 ; xx < 4; xx++)
			{
				//temp_av = sm_a[ll][d_t2_1_offset[l_idx_t3] + (xx * 16)];
				temp_av = sm_a[ll][idx_p4 + (idx_h1) * FUSION_SIZE_SLICE_2_P4 + (xx * 16)];

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
	for (int l = 0; l < size_internal && kernel_2 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		if (idx_p4 < rng_p4 && idx_h1 < rng_h2 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p5; ll++)
		{
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = d_t2_2[(blk_idx_p4 * FUSION_SIZE_SLICE_2_P4 + idx_p4 + (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5 + ll + (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2 + idx_h1) * size_p5) * size_p4) * size_h7 + (threadIdx.x + l)];
		}

		// Load Input Tensor to Shared Memory
		if (idx_h3 < rng_h3 && idx_h2 < rng_h1 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p6; ll++)
		{
			sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_2_X] = d_v2_2[(blk_idx_h3 * FUSION_SIZE_SLICE_2_H3 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6 + ll) * size_h1) * size_h3) + (threadIdx.y + l) * list_stride_v2[7]];
		}
		__syncthreads();

		// Cross-Product: -1
		// Part: Generalized Threads
		for (int ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++)
		{
			//temp_bv[0] = sm_b[ll][d_v2_2_offset[l_idx_t3] + 0];
			temp_bv[0] = sm_b[ll][idx_h3 + (idx_h1) * FUSION_SIZE_SLICE_2_H3 + 0];
			temp_bv[1] = sm_b[ll][idx_h3 + (idx_h1) * FUSION_SIZE_SLICE_2_H3 + 16];
			temp_bv[2] = sm_b[ll][idx_h3 + (idx_h1) * FUSION_SIZE_SLICE_2_H3 + 32];
			temp_bv[3] = sm_b[ll][idx_h3 + (idx_h1) * FUSION_SIZE_SLICE_2_H3 + 48];

			for (int xx = 0 ; xx < 4; xx++)
			{
				//temp_av = sm_a[ll][d_t2_2_offset[l_idx_t3] + (xx * 16)];
				temp_av = sm_a[ll][idx_p4 + (idx_h2) * FUSION_SIZE_SLICE_2_P4 + (xx * 16)];

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
	for (int l = 0; l < size_internal && kernel_3 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		if (idx_p4 < rng_p4 && idx_h1 < rng_h3 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p5; ll++)
		{
			sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = d_t2_3[(blk_idx_p4 * FUSION_SIZE_SLICE_2_P4 + idx_p4 + (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5 + ll + (blk_idx_h3 * FUSION_SIZE_SLICE_2_H3 + idx_h1) * size_p5) * size_p4) * size_h7 + (threadIdx.x + l)];
        }

		// Load Input Tensor to Shared Memory
		if (idx_h3 < rng_h2 && idx_h2 < rng_h1 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < rng_p6; ll++)
		{
			sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_2_X] = d_v2_3[(blk_idx_h2 * FUSION_SIZE_SLICE_2_H2 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1 + idx_h2 + (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6 + ll) * size_h1) * size_h2) + (threadIdx.y + l) * list_stride_v2[8]];
		}
		__syncthreads();

		// Cross-Product: -1
		// Part: Generalized Threads
		for (int ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++)
		{
			//temp_bv[0] = sm_b[ll][d_v2_3_offset[l_idx_t3] + 0];
			temp_bv[0] = sm_b[ll][idx_h2 + (idx_h1) * FUSION_SIZE_SLICE_2_H2 + 0];
			temp_bv[1] = sm_b[ll][idx_h2 + (idx_h1) * FUSION_SIZE_SLICE_2_H2 + 16];
			temp_bv[2] = sm_b[ll][idx_h2 + (idx_h1) * FUSION_SIZE_SLICE_2_H2 + 32];
			temp_bv[3] = sm_b[ll][idx_h2 + (idx_h1) * FUSION_SIZE_SLICE_2_H2 + 48];

			for (int xx = 0 ; xx < 4; xx++)
			{
				//temp_av = sm_a[ll][d_t2_3_offset[l_idx_t3] + (xx * 16)];
				temp_av = sm_a[ll][idx_p4 + (idx_h3) * FUSION_SIZE_SLICE_2_P4 + (xx * 16)];

				reg_tile[0][xx] -= temp_av * temp_bv[0];
				reg_tile[1][xx] -= temp_av * temp_bv[1];
				reg_tile[2][xx] -= temp_av * temp_bv[2];
				reg_tile[3][xx] -= temp_av * temp_bv[3];
			}
		}
		__syncthreads();
	}


	// Store Results (Registers) to Global Memory
	// Part: Generalized Threads
	// Part: Generalized Register-Tiling
	if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p4 < rng_p4 && idx_h1 < rng_h1)
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			if(i < rng_p6 && j < rng_p5)
			t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] += reg_tile[i][j];
		}
	}
}

// created by tc_gen_code_Kernel()
__global__ void kernel_ccsdT_sd1_123_partial_full(double* t3, double* d_t2_1, double* d_t2_2, double* d_t2_3, double* d_v2_1, double* d_v2_2, double* d_v2_3, 
        int size_h3,    int size_h2,    int size_h1,    int size_p6,    int size_p5,    int size_p4,    int size_h7, 
        int numBlk_h3,  int numBlk_h2,  int numBlk_h1,  int numBlk_p6,  int numBlk_p5,  int numBlk_p4,
        int kernel_1, int kernel_2, int kernel_3, int kernel_4, int kernel_5, int kernel_6, int kernel_7, int kernel_8, int kernel_9,
        int stride_reg_x, int stride_reg_y, int size_internal)
{
	// For Shared Memory,
	__shared__ double sm_a[16][64 + 1];
	__shared__ double sm_b[16][64 + 1];

	int internal_upperbound   = 0;
    int internal_offset;

    // should support for non-full tiles
	int idx_h3 = threadIdx.x % FUSION_SIZE_SLICE_2_H3;
	int idx_h2 = threadIdx.x / FUSION_SIZE_SLICE_2_H3;
	int idx_p4 = threadIdx.y % FUSION_SIZE_SLICE_2_P4;
    int idx_h1 = threadIdx.y / FUSION_SIZE_SLICE_2_P4;
    
    int tmp_blkIdx;        
    int blk_idx_p4  = blockIdx.x / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);
    tmp_blkIdx      = blockIdx.x % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);

    int blk_idx_p5  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);

    int blk_idx_p6  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1);

    int blk_idx_h1 = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2);
    tmp_blkIdx     = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2);

    int blk_idx_h2 = (tmp_blkIdx) / (numBlk_h3);
    int blk_idx_h3 = blockIdx.x % (numBlk_h3);

    int t3_base_thread = blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + 
    (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h2 + 
    (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h1 + 
    (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 +  
    (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + 
    (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + idx_p4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;


	double temp_av;
	double temp_bv[4];
	double reg_tile[4][4];

	for (int i = 0; i < 4; i++)
	for (int j = 0; j < 4; j++)
	reg_tile[i][j] = 0.0;

	// tensor contraction
	#pragma unroll 1
	for (int l = 0; l < size_internal && kernel_1 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		if (threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < 4; ll++)
		{
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = d_t2_1[(blk_idx_p4 * FUSION_SIZE_SLICE_2_P4 + idx_p4 + (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5 + ll + (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1 + idx_h1) * size_p5) * size_p4) * size_h7 + (threadIdx.x + l)];
		}

		// Load Input Tensor to Shared Memory
		if (threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < 4; ll++)
		{
			sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_2_X] = d_v2_1[blk_idx_h3 * FUSION_SIZE_SLICE_2_H3 + idx_h3 + (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2 + idx_h2 + (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6 + ll) * size_h2) * size_h3 + (threadIdx.y + l) * list_stride_v2[6]];
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
				temp_av = sm_a[ll][idx_p4 + (idx_h1) * FUSION_SIZE_SLICE_2_P4 + (xx * 16)];

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
	for (int l = 0; l < size_internal && kernel_2 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		if (threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < 4; ll++)
		{
			sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = d_t2_2[(blk_idx_p4 * FUSION_SIZE_SLICE_2_P4 + idx_p4 + (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5 + ll + (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2 + idx_h1) * size_p5) * size_p4) * size_h7 + (threadIdx.x + l)];
		}

		// Load Input Tensor to Shared Memory
		if (threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < 4; ll++)
		{
			sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_2_X] = d_v2_2[(blk_idx_h3 * FUSION_SIZE_SLICE_2_H3 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6 + ll) * size_h1) * size_h3) + (threadIdx.y + l) * list_stride_v2[7]];
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
				//temp_av = sm_a[ll][d_t2_2_offset[l_idx_t3] + (xx * 16)];
				temp_av = sm_a[ll][idx_p4 + (idx_h2) * FUSION_SIZE_SLICE_2_P4 + (xx * 16)];

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
	for (int l = 0; l < size_internal && kernel_3 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		if (threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < 4; ll++)
		{
			sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = d_t2_3[(blk_idx_p4 * FUSION_SIZE_SLICE_2_P4 + idx_p4 + (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5 + ll + (blk_idx_h3 * FUSION_SIZE_SLICE_2_H3 + idx_h1) * size_p5) * size_p4) * size_h7 + (threadIdx.x + l)];
		}

		// Load Input Tensor to Shared Memory
		if (threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
		for (int ll = 0; ll < 4; ll++)
		{
			sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_2_X] = d_v2_3[(blk_idx_h2 * FUSION_SIZE_SLICE_2_H2 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1 + idx_h2 + (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6 + ll) * size_h1) * size_h2) + (threadIdx.y + l) * list_stride_v2[8]];
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
				//temp_av = sm_a[ll][d_t2_3_offset[l_idx_t3] + (xx * 16)];
				temp_av = sm_a[ll][idx_p4 + (idx_h3) * FUSION_SIZE_SLICE_2_P4 + (xx * 16)];

				reg_tile[0][xx] -= temp_av * temp_bv[0];
				reg_tile[1][xx] -= temp_av * temp_bv[1];
				reg_tile[2][xx] -= temp_av * temp_bv[2];
				reg_tile[3][xx] -= temp_av * temp_bv[3];
			}
		}
		__syncthreads();
	}

	// Store Results (Registers) to Global Memory
	// Part: Generalized Threads
	// Part: Generalized Register-Tiling
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] += reg_tile[i][j];
		}
	}
}

// created by tc_gen_code_Kernel()
__global__ void kernel_ccsdT_sd1_123_full_full(double* t3, double* d_t2_1, double* d_t2_2, double* d_t2_3, double* d_v2_1, double* d_v2_2, double* d_v2_3, 
        int size_h3,    int size_h2,    int size_h1,    int size_p6,    int size_p5,    int size_p4,    int size_h7, 
        int numBlk_h3,  int numBlk_h2,  int numBlk_h1,  int numBlk_p6,  int numBlk_p5,  int numBlk_p4,
        int kernel_1, int kernel_2, int kernel_3, int kernel_4, int kernel_5, int kernel_6, int kernel_7, int kernel_8, int kernel_9,
        int stride_reg_x, int stride_reg_y, int size_internal)
{
	// For Shared Memory,
	__shared__ double sm_a[16][64 + 1];
	__shared__ double sm_b[16][64 + 1];

    // should support for non-full tiles
	int idx_h3 = threadIdx.x % FUSION_SIZE_SLICE_2_H3;
	int idx_h2 = threadIdx.x / FUSION_SIZE_SLICE_2_H3;
	int idx_p4 = threadIdx.y % FUSION_SIZE_SLICE_2_P4;
    int idx_h1 = threadIdx.y / FUSION_SIZE_SLICE_2_P4;
    
    int tmp_blkIdx;        
    int blk_idx_p4  = blockIdx.x / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);
    tmp_blkIdx      = blockIdx.x % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);

    int blk_idx_p5  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);

    int blk_idx_p6  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1);

    int blk_idx_h1 = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2);
    tmp_blkIdx     = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2);

    int blk_idx_h2 = (tmp_blkIdx) / (numBlk_h3);
    int blk_idx_h3 = blockIdx.x % (numBlk_h3);

    int t3_base_thread = blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + 
                        (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h2 + 
                        (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h1 + 
                        (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 +  
                        (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + 
                        (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + idx_p4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;




	double temp_av;
	double temp_bv[4];
	double reg_tile[4][4];

	for (int i = 0; i < 4; i++)
	for (int j = 0; j < 4; j++)
	reg_tile[i][j] = 0.0;

	// tensor contraction
	#pragma unroll 1
	for (int l = 0; l < size_internal && kernel_1 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		for (int ll = 0; ll < 4; ll++)
		{
			sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = d_t2_1[(blk_idx_p4 * FUSION_SIZE_SLICE_2_P4 + idx_p4 + (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5 + ll + (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1 + idx_h1) * size_p5) * size_p4) * size_h7 + (threadIdx.x + l)];
		}

		// Load Input Tensor to Shared Memory
		for (int ll = 0; ll < 4; ll++)
		{
			sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_2_X] = d_v2_1[blk_idx_h3 * FUSION_SIZE_SLICE_2_H3 + idx_h3 + (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2 + idx_h2 + (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6 + ll) * size_h2) * size_h3 + (threadIdx.y + l) * list_stride_v2[6]];
		}
		__syncthreads();

		// Cross-Product: -1
		// Part: Generalized Threads
		for (int ll = 0; ll < FUSION_SIZE_INT_UNIT; ll++)
		{
			temp_bv[0] = sm_b[ll][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_2_H3 + 0];
			temp_bv[1] = sm_b[ll][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_2_H3 + 16];
			temp_bv[2] = sm_b[ll][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_2_H3 + 32];
			temp_bv[3] = sm_b[ll][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_2_H3 + 48];

			for (int xx = 0 ; xx < 4; xx++)
			{
				//temp_av = sm_a[ll][d_t2_1_offset[l_idx_t3] + (xx * 16)];
				temp_av = sm_a[ll][idx_p4 + (idx_h1) * FUSION_SIZE_SLICE_2_P4 + (xx * 16)];

				reg_tile[0][xx] -= temp_av * temp_bv[0];
				reg_tile[1][xx] -= temp_av * temp_bv[1];
				reg_tile[2][xx] -= temp_av * temp_bv[2];
				reg_tile[3][xx] -= temp_av * temp_bv[3];
			}
		}
		__syncthreads();
	}

	// tensor contraction
	#pragma unroll 1
	for (int l = 0; l < size_internal && kernel_2 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		for (int ll = 0; ll < 4; ll++)
		{
			sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = d_t2_2[(blk_idx_p4 * FUSION_SIZE_SLICE_2_P4 + idx_p4 + (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5 + ll + (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2 + idx_h1) * size_p5) * size_p4) * size_h7 + (threadIdx.x + l)];
		}

		// Load Input Tensor to Shared Memory
		for (int ll = 0; ll < 4; ll++)
		{
			sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_2_X] = d_v2_2[(blk_idx_h3 * FUSION_SIZE_SLICE_2_H3 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6 + ll) * size_h1) * size_h3) + (threadIdx.y + l) * list_stride_v2[7]];
		}
		__syncthreads();

		// Cross-Product: -1
		// Part: Generalized Threads
		for (int ll = 0; ll < FUSION_SIZE_INT_UNIT; ll++)
		{
			temp_bv[0] = sm_b[ll][idx_h3 + (idx_h1) * FUSION_SIZE_SLICE_2_H3 + 0];
			temp_bv[1] = sm_b[ll][idx_h3 + (idx_h1) * FUSION_SIZE_SLICE_2_H3 + 16];
			temp_bv[2] = sm_b[ll][idx_h3 + (idx_h1) * FUSION_SIZE_SLICE_2_H3 + 32];
			temp_bv[3] = sm_b[ll][idx_h3 + (idx_h1) * FUSION_SIZE_SLICE_2_H3 + 48];

			for (int xx = 0 ; xx < 4; xx++)
			{
				//temp_av = sm_a[ll][d_t2_2_offset[l_idx_t3] + (xx * 16)];
				temp_av = sm_a[ll][idx_p4 + (idx_h2) * FUSION_SIZE_SLICE_2_P4 + (xx * 16)];

				reg_tile[0][xx] += temp_av * temp_bv[0];
				reg_tile[1][xx] += temp_av * temp_bv[1];
				reg_tile[2][xx] += temp_av * temp_bv[2];
				reg_tile[3][xx] += temp_av * temp_bv[3];
			}
		}
		__syncthreads();
	}

	// tensor contraction
	#pragma unroll 1
	for (int l = 0; l < size_internal && kernel_3 == 1; l+= FUSION_SIZE_INT_UNIT)
	{
		// Load Input Tensor to Shared Memory: 16:16
		// # of Internal Indices: 1
		for (int ll = 0; ll < 4; ll++)
		{
			sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = d_t2_3[(blk_idx_p4 * FUSION_SIZE_SLICE_2_P4 + idx_p4 + (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5 + ll + (blk_idx_h3 * FUSION_SIZE_SLICE_2_H3 + idx_h1) * size_p5) * size_p4) * size_h7 + (threadIdx.x + l)];
		}

		// Load Input Tensor to Shared Memory
		for (int ll = 0; ll < 4; ll++)
		{
			sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_2_X] = d_v2_3[(blk_idx_h2 * FUSION_SIZE_SLICE_2_H2 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1 + idx_h2 + (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6 + ll) * size_h1) * size_h2) + (threadIdx.y + l) * list_stride_v2[8]];
		}
		__syncthreads();

		// Cross-Product: -1
		// Part: Generalized Threads
		for (int ll = 0; ll < FUSION_SIZE_INT_UNIT; ll++)
		{
			temp_bv[0] = sm_b[ll][idx_h2 + (idx_h1) * FUSION_SIZE_SLICE_2_H2 + 0];
			temp_bv[1] = sm_b[ll][idx_h2 + (idx_h1) * FUSION_SIZE_SLICE_2_H2 + 16];
			temp_bv[2] = sm_b[ll][idx_h2 + (idx_h1) * FUSION_SIZE_SLICE_2_H2 + 32];
			temp_bv[3] = sm_b[ll][idx_h2 + (idx_h1) * FUSION_SIZE_SLICE_2_H2 + 48];

			for (int xx = 0 ; xx < 4; xx++)
			{
				//temp_av = sm_a[ll][d_t2_3_offset[l_idx_t3] + (xx * 16)];
				temp_av = sm_a[ll][idx_p4 + (idx_h3) * FUSION_SIZE_SLICE_2_P4 + (xx * 16)];

				reg_tile[0][xx] -= temp_av * temp_bv[0];
				reg_tile[1][xx] -= temp_av * temp_bv[1];
				reg_tile[2][xx] -= temp_av * temp_bv[2];
				reg_tile[3][xx] -= temp_av * temp_bv[3];
			}
		}
		__syncthreads();
	}


	// Store Results (Registers) to Global Memory
	// Part: Generalized Threads
	// Part: Generalized Register-Tiling
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] += reg_tile[i][j];
		}
	}
}

//
extern "C"
void sd_t_d1_all_cuda(Integer* sizes, 
	//int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_h7,
		double* t3,
		double* t2_1, double* v2_1, double* t2_2, double* v2_2,	double* t2_3, double* v2_3, double* t2_4, double* v2_4, double* t2_5, double* v2_5,	double* t2_6, double* v2_6, double* t2_7, double* v2_7, double* t2_8, double* v2_8, double* t2_9, double* v2_9,
        int kernel_1, int kernel_2, int kernel_3, int kernel_4, int kernel_5, int kernel_6, int kernel_7, int kernel_8, int kernel_9, int opt_rt)
{
	//TODO: Fix for all kernels, use sizes[0-63] for kernels 1-9
	int size_h1 = sizes[0];
	int size_h2 = sizes[1];
	int size_h3 = sizes[2];
	int size_h7 = sizes[3];
	int size_p4 = sizes[4];
	int size_p5 = sizes[5];
	int size_p6 = sizes[6];

    // printf (">>> sd_t_d1_all_cuda(...)\n");
	// # of Blocks for Each Kernel
	int	 num_blocks_kernel_1,		num_blocks_kernel_2;
	int  size_internal = size_h7;

	// Device Memory for Inputs and Output
    double *dev_t3;

	double *dev_t2_1, *dev_t2_2, *dev_t2_3, *dev_t2_4, *dev_t2_5, *dev_t2_6, *dev_t2_7, *dev_t2_8, *dev_t2_9;
	double *dev_v2_1, *dev_v2_2, *dev_v2_3, *dev_v2_4, *dev_v2_5, *dev_v2_6, *dev_v2_7, *dev_v2_8, *dev_v2_9;

    dev_t3 = t3_d;
    
    // cudaMalloc((void**) &dev_t3,   sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4);
	cudaMalloc((void**) &dev_t2_4, sizeof(double) * size_h1 * size_p6 * size_p5 * size_h7);
	cudaMalloc((void**) &dev_v2_4, sizeof(double) * size_h7 * size_p4 * size_h2 * size_h3);
	cudaMalloc((void**) &dev_t2_5, sizeof(double) * size_h2 * size_p6 * size_p5 * size_h7);
	cudaMalloc((void**) &dev_v2_5, sizeof(double) * size_h7 * size_p4 * size_h1 * size_h3);
	cudaMalloc((void**) &dev_t2_6, sizeof(double) * size_h3 * size_p6 * size_p5 * size_h7);
	cudaMalloc((void**) &dev_v2_6, sizeof(double) * size_h7 * size_p4 * size_h1 * size_h2);
	cudaMalloc((void**) &dev_t2_7, sizeof(double) * size_h1 * size_p6 * size_p4 * size_h7);
	cudaMalloc((void**) &dev_v2_7, sizeof(double) * size_h7 * size_p5 * size_h2 * size_h3);
	cudaMalloc((void**) &dev_t2_8, sizeof(double) * size_h2 * size_p6 * size_p4 * size_h7);
	cudaMalloc((void**) &dev_v2_8, sizeof(double) * size_h7 * size_p5 * size_h1 * size_h3);
	cudaMalloc((void**) &dev_t2_9, sizeof(double) * size_h3 * size_p6 * size_p4 * size_h7);
    cudaMalloc((void**) &dev_v2_9, sizeof(double) * size_h7 * size_p5 * size_h1 * size_h2);
    
	cudaMalloc((void**) &dev_t2_1, sizeof(double) * size_h1 * size_p5 * size_p4 * size_h7);
	cudaMalloc((void**) &dev_v2_1, sizeof(double) * size_h7 * size_p6 * size_h2 * size_h3);
	cudaMalloc((void**) &dev_t2_2, sizeof(double) * size_h2 * size_p5 * size_p4 * size_h7);
	cudaMalloc((void**) &dev_v2_2, sizeof(double) * size_h7 * size_p6 * size_h1 * size_h3);
	cudaMalloc((void**) &dev_t2_3, sizeof(double) * size_h3 * size_p5 * size_p4 * size_h7);
    cudaMalloc((void**) &dev_v2_3, sizeof(double) * size_h7 * size_p6 * size_h1 * size_h2);
    
    // cudaMemcpy(dev_t3, 	 t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4, 	cudaMemcpyHostToDevice);

	cudaMemcpy(dev_t2_4, t2_4, sizeof(double) * size_h1 * size_p6 * size_p5 * size_h7, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_v2_4, v2_4, sizeof(double) * size_h7 * size_p4 * size_h2 * size_h3, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_t2_5, t2_5, sizeof(double) * size_h2 * size_p6 * size_p5 * size_h7, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_v2_5, v2_5, sizeof(double) * size_h7 * size_p4 * size_h1 * size_h3, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_t2_6, t2_6, sizeof(double) * size_h3 * size_p6 * size_p5 * size_h7, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_v2_6, v2_6, sizeof(double) * size_h7 * size_p4 * size_h1 * size_h2, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_t2_7, t2_7, sizeof(double) * size_h1 * size_p6 * size_p4 * size_h7, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_v2_7, v2_7, sizeof(double) * size_h7 * size_p5 * size_h2 * size_h3, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_t2_8, t2_8, sizeof(double) * size_h2 * size_p6 * size_p4 * size_h7, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_v2_8, v2_8, sizeof(double) * size_h7 * size_p5 * size_h1 * size_h3, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_t2_9, t2_9, sizeof(double) * size_h3 * size_p6 * size_p4 * size_h7, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_v2_9, v2_9, sizeof(double) * size_h7 * size_p5 * size_h1 * size_h2, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_t2_1, t2_1, sizeof(double) * size_h1 * size_p5 * size_p4 * size_h7, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_v2_1, v2_1, sizeof(double) * size_h7 * size_p6 * size_h2 * size_h3, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_t2_2, t2_2, sizeof(double) * size_h2 * size_p5 * size_p4 * size_h7, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_v2_2, v2_2, sizeof(double) * size_h7 * size_p6 * size_h1 * size_h3, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_t2_3, t2_3, sizeof(double) * size_h3 * size_p5 * size_p4 * size_h7, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_v2_3, v2_3, sizeof(double) * size_h7 * size_p6 * size_h1 * size_h2, cudaMemcpyHostToDevice);

    num_blocks_kernel_1 = CEIL(size_h3, FUSION_SIZE_SLICE_1_H3) * CEIL(size_h2, FUSION_SIZE_SLICE_1_H2) * CEIL(size_h1, FUSION_SIZE_SLICE_1_H1) * CEIL(size_p6, FUSION_SIZE_SLICE_1_P6) * CEIL(size_p5, FUSION_SIZE_SLICE_1_P5) * CEIL(size_p4, FUSION_SIZE_SLICE_1_P4);
    num_blocks_kernel_2 = CEIL(size_h3, FUSION_SIZE_SLICE_2_H3) * CEIL(size_h2, FUSION_SIZE_SLICE_2_H2) * CEIL(size_h1, FUSION_SIZE_SLICE_2_H1) * CEIL(size_p6, FUSION_SIZE_SLICE_2_P6) * CEIL(size_p5, FUSION_SIZE_SLICE_2_P5) * CEIL(size_p4, FUSION_SIZE_SLICE_2_P4);

	// Depends on # of Fused Kernel
	dim3 gridsize_1(num_blocks_kernel_1);
	dim3 blocksize_1(FUSION_SIZE_TB_1_X, FUSION_SIZE_TB_1_Y);

	dim3 gridsize_2(num_blocks_kernel_2);
	dim3 blocksize_2(FUSION_SIZE_TB_2_X, FUSION_SIZE_TB_2_Y);

	int	str_sd2_t3_h3 = 1;
	int str_sd2_t3_h2 = str_sd2_t3_h3 * size_h3;
	int str_sd2_t3_h1 = str_sd2_t3_h2 * size_h2;
	int str_sd2_t3_p6 = str_sd2_t3_h1 * size_h1;
	int str_sd2_t3_p5 = str_sd2_t3_p6 * size_p6;
	int str_sd2_t3_p4 = str_sd2_t3_p5 * size_p5;

	int str_reg_x_1 = str_sd2_t3_p5;	// STR_SD2_T3_P5
	int str_reg_y_1 = str_sd2_t3_p4;	// STR_SD2_T3_P4
	int str_reg_x_2 = str_sd2_t3_p5;	// STR_SD2_T3_P5
	int str_reg_y_2 = str_sd2_t3_p6;	// SDT_SD2_T3_P6

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

    // sd1: [1] 4,5,6,7,8,9     [2] 1,2,3
    if (kernel_1 || kernel_2 || kernel_3)
    {
        if (kernel_4 || kernel_5 || kernel_6 || kernel_7 || kernel_8 || kernel_9)
        {
            //printf (">>> kernel_1,2,3 (ON) && kernel_4,5,6,7,8,9 (ON) <<< %d,%d,%d,%d,%d,%d,%d,%d,%d\n", kernel_1, kernel_2, kernel_3, kernel_4, kernel_5, kernel_6, kernel_7, kernel_8, kernel_9);
            if (size_h3 % FUSION_SIZE_SLICE_1_H3 == 0 && size_h2 % FUSION_SIZE_SLICE_1_H2 == 0 && size_h1 % FUSION_SIZE_SLICE_1_H1 == 0 && 
                size_p6 % FUSION_SIZE_SLICE_1_P6 == 0 && size_p5 % FUSION_SIZE_SLICE_1_P5 == 0 && size_p4 % FUSION_SIZE_SLICE_1_P4 == 0)
            {
                if (size_h7 % FUSION_SIZE_INT_UNIT == 0)
                {
                    kernel_ccsdT_sd1_fully_fused_full_full<<<gridsize_1, blocksize_1>>>(dev_t3, 
                    dev_t2_4, dev_t2_5, dev_t2_6, dev_t2_7, dev_t2_8, dev_t2_9, 
                    dev_v2_4, dev_v2_5, dev_v2_6, dev_v2_7, dev_v2_8, dev_v2_9, 
                    dev_t2_1, dev_t2_2, dev_t2_3, 
                    dev_v2_1, dev_v2_2, dev_v2_3, 
                    size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_h7,
                    CEIL(size_h3, FUSION_SIZE_SLICE_2_H3),CEIL(size_h2, FUSION_SIZE_SLICE_2_H2),CEIL(size_h1, FUSION_SIZE_SLICE_2_H1),
                    CEIL(size_p6, FUSION_SIZE_SLICE_2_P6),CEIL(size_p5, FUSION_SIZE_SLICE_2_P5),CEIL(size_p4, FUSION_SIZE_SLICE_2_P4),
                    kernel_1, kernel_2, kernel_3,
                    kernel_4, kernel_5, kernel_6,
                    kernel_7, kernel_8, kernel_9,
                    str_reg_x_1, str_reg_y_1,
                    size_internal);
                }
                else
                {
                    kernel_ccsdT_sd1_fully_fused_partial_full<<<gridsize_1, blocksize_1>>>(dev_t3, 
                    dev_t2_4, dev_t2_5, dev_t2_6, dev_t2_7, dev_t2_8, dev_t2_9, 
                    dev_v2_4, dev_v2_5, dev_v2_6, dev_v2_7, dev_v2_8, dev_v2_9, 
                    dev_t2_1, dev_t2_2, dev_t2_3, 
                    dev_v2_1, dev_v2_2, dev_v2_3, 
                    size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_h7,
                    CEIL(size_h3, FUSION_SIZE_SLICE_2_H3),CEIL(size_h2, FUSION_SIZE_SLICE_2_H2),CEIL(size_h1, FUSION_SIZE_SLICE_2_H1),
                    CEIL(size_p6, FUSION_SIZE_SLICE_2_P6),CEIL(size_p5, FUSION_SIZE_SLICE_2_P5),CEIL(size_p4, FUSION_SIZE_SLICE_2_P4),
                    kernel_1, kernel_2, kernel_3,
                    kernel_4, kernel_5, kernel_6,
                    kernel_7, kernel_8, kernel_9,
                    str_reg_x_1, str_reg_y_1,
                    size_internal);
                }
            }
            else
            {
                kernel_ccsdT_sd1_fully_fused_partial_partial<<<gridsize_1, blocksize_1>>>(dev_t3, 
                dev_t2_4, dev_t2_5, dev_t2_6, dev_t2_7, dev_t2_8, dev_t2_9, 
                dev_v2_4, dev_v2_5, dev_v2_6, dev_v2_7, dev_v2_8, dev_v2_9, 
                dev_t2_1, dev_t2_2, dev_t2_3, 
                dev_v2_1, dev_v2_2, dev_v2_3, 
                size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_h7,
                CEIL(size_h3, FUSION_SIZE_SLICE_2_H3),CEIL(size_h2, FUSION_SIZE_SLICE_2_H2),CEIL(size_h1, FUSION_SIZE_SLICE_2_H1),
                CEIL(size_p6, FUSION_SIZE_SLICE_2_P6),CEIL(size_p5, FUSION_SIZE_SLICE_2_P5),CEIL(size_p4, FUSION_SIZE_SLICE_2_P4),
                kernel_1, kernel_2, kernel_3,
                kernel_4, kernel_5, kernel_6,
                kernel_7, kernel_8, kernel_9,
                str_reg_x_1, str_reg_y_1,
                size_internal);
            }
        }
        else
        {
            if (size_h3 % FUSION_SIZE_SLICE_1_H3 == 0 && size_h2 % FUSION_SIZE_SLICE_1_H2 == 0 && size_h1 % FUSION_SIZE_SLICE_1_H1 == 0 && 
                size_p6 % FUSION_SIZE_SLICE_1_P6 == 0 && size_p5 % FUSION_SIZE_SLICE_1_P5 == 0 && size_p4 % FUSION_SIZE_SLICE_1_P4 == 0)
            {
                if (size_h7 % FUSION_SIZE_INT_UNIT == 0)
                {
                    kernel_ccsdT_sd1_123_full_full<<<gridsize_2, blocksize_2>>>(dev_t3, 
                    dev_t2_1, dev_t2_2, dev_t2_3, 
                    dev_v2_1, dev_v2_2, dev_v2_3, 
                    size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_h7,
                    CEIL(size_h3, FUSION_SIZE_SLICE_2_H3),CEIL(size_h2, FUSION_SIZE_SLICE_2_H2),CEIL(size_h1, FUSION_SIZE_SLICE_2_H1),
                    CEIL(size_p6, FUSION_SIZE_SLICE_2_P6),CEIL(size_p5, FUSION_SIZE_SLICE_2_P5),CEIL(size_p4, FUSION_SIZE_SLICE_2_P4),
                    kernel_1, kernel_2, kernel_3,
                    kernel_4, kernel_5, kernel_6,
                    kernel_7, kernel_8, kernel_9,
                    str_reg_x_2, str_reg_y_2,
                    size_internal);
                }
                else
                {
                    kernel_ccsdT_sd1_123_partial_full<<<gridsize_2, blocksize_2>>>(dev_t3, 
                    dev_t2_1, dev_t2_2, dev_t2_3, 
                    dev_v2_1, dev_v2_2, dev_v2_3, 
                    size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_h7,
                    CEIL(size_h3, FUSION_SIZE_SLICE_2_H3),CEIL(size_h2, FUSION_SIZE_SLICE_2_H2),CEIL(size_h1, FUSION_SIZE_SLICE_2_H1),
                    CEIL(size_p6, FUSION_SIZE_SLICE_2_P6),CEIL(size_p5, FUSION_SIZE_SLICE_2_P5),CEIL(size_p4, FUSION_SIZE_SLICE_2_P4),
                    kernel_1, kernel_2, kernel_3,
                    kernel_4, kernel_5, kernel_6,
                    kernel_7, kernel_8, kernel_9,
                    str_reg_x_2, str_reg_y_2,
                    size_internal);
                }
            }
            else
            {
                kernel_ccsdT_sd1_123_partial_partial<<<gridsize_2, blocksize_2>>>(dev_t3, 
                dev_t2_1, dev_t2_2, dev_t2_3, 
                dev_v2_1, dev_v2_2, dev_v2_3, 
                size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_h7,
                CEIL(size_h3, FUSION_SIZE_SLICE_2_H3),CEIL(size_h2, FUSION_SIZE_SLICE_2_H2),CEIL(size_h1, FUSION_SIZE_SLICE_2_H1),
                CEIL(size_p6, FUSION_SIZE_SLICE_2_P6),CEIL(size_p5, FUSION_SIZE_SLICE_2_P5),CEIL(size_p4, FUSION_SIZE_SLICE_2_P4),
                kernel_1, kernel_2, kernel_3,
                kernel_4, kernel_5, kernel_6,
                kernel_7, kernel_8, kernel_9,
                str_reg_x_2, str_reg_y_2,
                size_internal);
            }
        }
    }
    else
    {
        if (kernel_4 || kernel_5 || kernel_6 || kernel_7 || kernel_8 || kernel_9)
        {
            //printf (">>> kernel_1,2,3 (OFF) && kernel_4,5,6,7,8,9 (ON) <<< %d,%d,%d,%d,%d,%d,%d,%d,%d\n", kernel_1, kernel_2, kernel_3, kernel_4, kernel_5, kernel_6, kernel_7, kernel_8, kernel_9);
            if (size_h3 % FUSION_SIZE_SLICE_1_H3 == 0 && size_h2 % FUSION_SIZE_SLICE_1_H2 == 0 && size_h1 % FUSION_SIZE_SLICE_1_H1 == 0 && 
                size_p6 % FUSION_SIZE_SLICE_1_P6 == 0 && size_p5 % FUSION_SIZE_SLICE_1_P5 == 0 && size_p4 % FUSION_SIZE_SLICE_1_P4 == 0)
            {
                if (size_h7 % FUSION_SIZE_INT_UNIT == 0)
                {
                    kernel_ccsdT_sd1_456789_full_full<<<gridsize_1, blocksize_1>>>(dev_t3, 
                    dev_t2_4, dev_t2_5, dev_t2_6, dev_t2_7, dev_t2_8, dev_t2_9, 
                    dev_v2_4, dev_v2_5, dev_v2_6, dev_v2_7, dev_v2_8, dev_v2_9, 
                    size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_h7,
                    CEIL(size_h3, FUSION_SIZE_SLICE_1_H3),CEIL(size_h2, FUSION_SIZE_SLICE_1_H2),CEIL(size_h1, FUSION_SIZE_SLICE_1_H1),
                    CEIL(size_p6, FUSION_SIZE_SLICE_1_P6),CEIL(size_p5, FUSION_SIZE_SLICE_1_P5),CEIL(size_p4, FUSION_SIZE_SLICE_1_P4),
                    kernel_1, kernel_2, kernel_3,
                    kernel_4, kernel_5, kernel_6,
                    kernel_7, kernel_8, kernel_9,
                    str_reg_x_1, str_reg_y_1,
                    size_internal);
                }
                else
                {
                    kernel_ccsdT_sd1_456789_partial_full<<<gridsize_1, blocksize_1>>>(dev_t3, 
                    dev_t2_4, dev_t2_5, dev_t2_6, dev_t2_7, dev_t2_8, dev_t2_9, 
                    dev_v2_4, dev_v2_5, dev_v2_6, dev_v2_7, dev_v2_8, dev_v2_9, 
                    size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_h7,
                    CEIL(size_h3, FUSION_SIZE_SLICE_1_H3),CEIL(size_h2, FUSION_SIZE_SLICE_1_H2),CEIL(size_h1, FUSION_SIZE_SLICE_1_H1),
                    CEIL(size_p6, FUSION_SIZE_SLICE_1_P6),CEIL(size_p5, FUSION_SIZE_SLICE_1_P5),CEIL(size_p4, FUSION_SIZE_SLICE_1_P4),
                    kernel_1, kernel_2, kernel_3,
                    kernel_4, kernel_5, kernel_6,
                    kernel_7, kernel_8, kernel_9,
                    str_reg_x_1, str_reg_y_1,
                    size_internal);
                }
            }
            else
            {
                kernel_ccsdT_sd1_456789_partial_partial<<<gridsize_1, blocksize_1>>>(dev_t3, 
                dev_t2_4, dev_t2_5, dev_t2_6, dev_t2_7, dev_t2_8, dev_t2_9, 
                dev_v2_4, dev_v2_5, dev_v2_6, dev_v2_7, dev_v2_8, dev_v2_9, 
                size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_h7,
                CEIL(size_h3, FUSION_SIZE_SLICE_1_H3),CEIL(size_h2, FUSION_SIZE_SLICE_1_H2),CEIL(size_h1, FUSION_SIZE_SLICE_1_H1),
                CEIL(size_p6, FUSION_SIZE_SLICE_1_P6),CEIL(size_p5, FUSION_SIZE_SLICE_1_P5),CEIL(size_p4, FUSION_SIZE_SLICE_1_P4),
                kernel_1, kernel_2, kernel_3,
                kernel_4, kernel_5, kernel_6,
                kernel_7, kernel_8, kernel_9,
                str_reg_x_1, str_reg_y_1,
                size_internal);
            }
        }
        else
        {
            printf (">>> kernel_1,2,3 (OFF) && kernel_4,5,6,7,8,9 (OFF) <<< %d,%d,%d,%d,%d,%d,%d,%d,%d\n", kernel_1, kernel_2, kernel_3, kernel_4, kernel_5, kernel_6, kernel_7, kernel_8, kernel_9);
        }
    }

    // Copy the Result from Device to Host
	// cudaMemcpy(t3, dev_t3, sizeof(double) * (size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4), cudaMemcpyDeviceToHost);

	// cudaFree()
    // cudaFree(dev_t3);

	// (Temporary)
	cudaFree(dev_t2_1);	cudaFree(dev_t2_2);	cudaFree(dev_t2_3);	
    cudaFree(dev_t2_4);	cudaFree(dev_t2_5);	cudaFree(dev_t2_6);	
    cudaFree(dev_t2_7);	cudaFree(dev_t2_8);	cudaFree(dev_t2_9);	
	cudaFree(dev_v2_1);	cudaFree(dev_v2_2);	cudaFree(dev_v2_3);	
    cudaFree(dev_v2_4);	cudaFree(dev_v2_5);	cudaFree(dev_v2_6);	
    cudaFree(dev_v2_7);	cudaFree(dev_v2_8);	cudaFree(dev_v2_9);
}

extern "C"
void sd_t_d1_all_cuda_master(Integer *sizes,
	    //int size_h3, int size_h2, int size_h1, 
		//int size_p6, int size_p5, int size_p4, int size_h7,
		double* t3,
		double* t2_1, double* v2_1, double* t2_2, double* v2_2,	double* t2_3, double* v2_3, 
		double* t2_4, double* v2_4, double* t2_5, double* v2_5,	double* t2_6, double* v2_6,
		double* t2_7, double* v2_7, double* t2_8, double* v2_8, double* t2_9, double* v2_9,
		int kernel_1, int kernel_2, int kernel_3, 
		int kernel_4, int kernel_5, int kernel_6, 
		int kernel_7, int kernel_8, int kernel_9,
		int opt_rt)
{
    #if 1
    if (kernel_1 && kernel_2 && kernel_3 && kernel_4 && kernel_5 && kernel_6 && kernel_7 && kernel_8 && kernel_8 && kernel_9
        && (sizes[0] == sizes[0 + 7] == sizes[0 + 14] == sizes[0 + 21] == sizes[0 + 28] == sizes[0 + 35] == sizes[0 + 42] == sizes[0 + 49] == sizes[0 + 56])
        && (sizes[1] == sizes[1 + 7] == sizes[1 + 14] == sizes[1 + 21] == sizes[1 + 28] == sizes[1 + 35] == sizes[1 + 42] == sizes[1 + 49] == sizes[1 + 56])
        && (sizes[2] == sizes[2 + 7] == sizes[2 + 14] == sizes[2 + 21] == sizes[2 + 28] == sizes[2 + 35] == sizes[2 + 42] == sizes[2 + 49] == sizes[2 + 56])
        && (sizes[3] == sizes[3 + 7] == sizes[3 + 14] == sizes[3 + 21] == sizes[3 + 28] == sizes[3 + 35] == sizes[3 + 42] == sizes[3 + 49] == sizes[3 + 56])
        && (sizes[5] == sizes[5 + 7] == sizes[5 + 14] == sizes[5 + 21] == sizes[5 + 28] == sizes[5 + 35] == sizes[5 + 42] == sizes[5 + 49] == sizes[4 + 56])
        && (sizes[4] == sizes[4 + 7] == sizes[4 + 14] == sizes[4 + 21] == sizes[4 + 28] == sizes[4 + 35] == sizes[4 + 42] == sizes[4 + 49] == sizes[5 + 56])
        && (sizes[6] == sizes[6 + 7] == sizes[6 + 14] == sizes[6 + 21] == sizes[6 + 28] == sizes[6 + 35] == sizes[6 + 42] == sizes[6 + 49] == sizes[6 + 56]))
    {
        // printf (">>[d1][fusion]>> %d, %d, %d, %d, %d, %d, %d, %d, %d\n", kernel_1, kernel_2, kernel_3, kernel_4, kernel_5, kernel_6, kernel_7, kernel_8, kernel_9);
        // printf (">>[d1][fusion]>> %d, %d, %d, %d, %d, %d, %d\n", size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_h7);
        sd_t_d1_all_cuda(sizes,
        //size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_h7,
        t3, 
        t2_1, v2_1, t2_2, v2_2, t2_3, v2_3,
        t2_4, v2_4, t2_5, v2_5, t2_6, v2_6,
        t2_7, v2_7, t2_8, v2_8, t2_9, v2_9,
        kernel_1, kernel_2, kernel_3,
        kernel_4, kernel_5, kernel_6,
        kernel_7, kernel_8, kernel_9, opt_rt);
    }
	else
	#endif
    {
        // printf (">>[d1][non-fusion]>> %d, %d, %d, %d, %d, %d, %d, %d, %d\n", kernel_1, kernel_2, kernel_3, kernel_4, kernel_5, kernel_6, kernel_7, kernel_8, kernel_9);
        // printf (">>[d1][non-fusion]>> %d, %d, %d, %d, %d, %d, %d\n", size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_h7);
        
        if (kernel_1){
			int size_h1 = sizes[0];
			int size_h2 = sizes[1];
			int size_h3 = sizes[2];
			int size_h7 = sizes[3];
			int size_p4 = sizes[4];
			int size_p5 = sizes[5];
			int size_p6 = sizes[6];
			jk_ccsd_t_d1_1_fusion_(size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_h7, t3, t2_1, v2_1, 1, 0);
		}
        
        if (kernel_2){
			int size_h1 = sizes[7];
			int size_h2 = sizes[8];
			int size_h3 = sizes[9];
			int size_h7 = sizes[10];
			int size_p4 = sizes[11];
			int size_p5 = sizes[12];
			int size_p6 = sizes[13];
			jk_ccsd_t_d1_2_fusion_(size_h3, size_h1, size_h2, size_p6, size_p5, size_p4, size_h7, t3, t2_2, v2_2, 1, 0);
		}
        
        if (kernel_3){
			int size_h1 = sizes[14];
			int size_h2 = sizes[15];
			int size_h3 = sizes[16];
			int size_h7 = sizes[17];
			int size_p4 = sizes[18];
			int size_p5 = sizes[19];
			int size_p6 = sizes[20];
			jk_ccsd_t_d1_3_fusion_(size_h1, size_h3, size_h2, size_p6, size_p5, size_p4, size_h7, t3, t2_3, v2_3, 1, 0);    
		}
        
        if (kernel_4){
			int size_h1 = sizes[21];
			int size_h2 = sizes[22];
			int size_h3 = sizes[23];
			int size_h7 = sizes[24];
			int size_p4 = sizes[25];
			int size_p5 = sizes[26];
			int size_p6 = sizes[27]; 
			jk_ccsd_t_d1_4_fusion_(size_h3, size_h2, size_h1, size_p5, size_p4, size_p6, size_h7, t3, t2_4, v2_4, 1, 0);
		}
        
        if (kernel_5){
			int size_h1 = sizes[28];
			int size_h2 = sizes[29];
			int size_h3 = sizes[30];
			int size_h7 = sizes[31];
			int size_p4 = sizes[32];
			int size_p5 = sizes[33];
			int size_p6 = sizes[34];  
			jk_ccsd_t_d1_5_fusion_(size_h3, size_h1, size_h2, size_p5, size_p4, size_p6, size_h7, t3, t2_5, v2_5, 1, 0);
		}

        if (kernel_6){
			int size_h1 = sizes[35];
			int size_h2 = sizes[36];
			int size_h3 = sizes[37];
			int size_h7 = sizes[38];
			int size_p4 = sizes[39];
			int size_p5 = sizes[40];
			int size_p6 = sizes[41];
			jk_ccsd_t_d1_6_fusion_(size_h1, size_h3, size_h2, size_p5, size_p4, size_p6, size_h7, t3, t2_6, v2_6, 1, 0);
		}
        
        if (kernel_7){
			int size_h1 = sizes[42];
			int size_h2 = sizes[43];
			int size_h3 = sizes[44];
			int size_h7 = sizes[45];
			int size_p4 = sizes[46];
			int size_p5 = sizes[47];
			int size_p6 = sizes[48];   
			jk_ccsd_t_d1_7_fusion_(size_h3, size_h2, size_h1, size_p5, size_p6, size_p4, size_h7, t3, t2_7, v2_7, 1, 0);
		}
        
        if (kernel_8){
			int size_h1 = sizes[49];
			int size_h2 = sizes[50];
			int size_h3 = sizes[51];
			int size_h7 = sizes[52];
			int size_p4 = sizes[53];
			int size_p5 = sizes[54];
			int size_p6 = sizes[55];  
			jk_ccsd_t_d1_8_fusion_(size_h3, size_h1, size_h2, size_p5, size_p6, size_p4, size_h7, t3, t2_8, v2_8, 1, 0);
		}
        
        if (kernel_9){
			int size_h1 = sizes[56];
			int size_h2 = sizes[57];
			int size_h3 = sizes[58];
			int size_h7 = sizes[59];
			int size_p4 = sizes[60];
			int size_p5 = sizes[61];
			int size_p6 = sizes[62]; 
			jk_ccsd_t_d1_9_fusion_(size_h1, size_h3, size_h2, size_p5, size_p6, size_p4, size_h7, t3, t2_9, v2_9, 1, 0);        
		}
    }
}

//
extern "C"
void sd_t_d1_all_cuda__(Integer *sizes,
	//Integer* p_size_h3, Integer* p_size_h2, Integer* p_size_h1, Integer* p_size_p6, Integer* p_size_p5, Integer* p_size_p4, Integer* p_size_h7,
			double* t3, double* t2_all, Integer* p_size_t2_all, double* v2_all, Integer* p_size_v2_all,
            Integer* p_kernel_1, Integer* p_kernel_2, Integer* p_kernel_3, 
            Integer* p_kernel_4, Integer* p_kernel_5, Integer* p_kernel_6, 
            Integer* p_kernel_7, Integer* p_kernel_8, Integer* p_kernel_9, Integer* p_opt_register_transpose)
{
    // int size_h3 = *p_size_h3;
    // int size_h2 = *p_size_h2;
    // int size_h1 = *p_size_h1;
    // int size_p6 = *p_size_p6;
    // int size_p5 = *p_size_p5;
    // int size_p4 = *p_size_p4;
    // int size_h7 = *p_size_h7;

    int kernel_1 = *p_kernel_1;
    int kernel_2 = *p_kernel_2;
    int kernel_3 = *p_kernel_3;
    int kernel_4 = *p_kernel_4;
    int kernel_5 = *p_kernel_5;
    int kernel_6 = *p_kernel_6;
    int kernel_7 = *p_kernel_7;
    int kernel_8 = *p_kernel_8;
    int kernel_9 = *p_kernel_9;

    int opt_register_transpose = *p_opt_register_transpose;

    int size_t2_all = *p_size_t2_all;
    int size_v2_all = *p_size_v2_all;

    unsigned int size_t2_each = size_t2_all / 9;
    unsigned int size_v2_each = size_v2_all / 9;

    double* t2_1 = t2_all;
    double* t2_2 = t2_all + (size_t2_each * 1);
    double* t2_3 = t2_all + (size_t2_each * 2);
    double* t2_4 = t2_all + (size_t2_each * 3);
    double* t2_5 = t2_all + (size_t2_each * 4);
    double* t2_6 = t2_all + (size_t2_each * 5);
    double* t2_7 = t2_all + (size_t2_each * 6);
    double* t2_8 = t2_all + (size_t2_each * 7);
    double* t2_9 = t2_all + (size_t2_each * 8);

    double* v2_1 = v2_all;
    double* v2_2 = v2_all + (size_v2_each * 1);
    double* v2_3 = v2_all + (size_v2_each * 2);
    double* v2_4 = v2_all + (size_v2_each * 3);
    double* v2_5 = v2_all + (size_v2_each * 4);
    double* v2_6 = v2_all + (size_v2_each * 5);
    double* v2_7 = v2_all + (size_v2_each * 6);
    double* v2_8 = v2_all + (size_v2_each * 7);
    double* v2_9 = v2_all + (size_v2_each * 8);

    //printf (">>> kernels: %d, %d, %d, %d, %d, %d, %d, %d, %d\n", kernel_1, kernel_2, kernel_3, kernel_4, kernel_5, kernel_6, kernel_7, kernel_8, kernel_9);

    //
    //  
    //
	sd_t_d1_all_cuda_master(//size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_h7,
					sizes,
                    t3, 
                    t2_1, v2_1, t2_2, v2_2, t2_3, v2_3,
                    t2_4, v2_4, t2_5, v2_5, t2_6, v2_6,
                    t2_7, v2_7, t2_8, v2_8, t2_9, v2_9,
                    kernel_1, kernel_2, kernel_3,
                    kernel_4, kernel_5, kernel_6,
                    kernel_7, kernel_8, kernel_9,
					opt_register_transpose);
}

