

#ifdef _OPENMP
#include <omp.h>
#endif

#include <CL/sycl.hpp>
#include <iostream>

void sd_t_d1_1_cpu(size_t size_idx_h1, size_t size_idx_h2, size_t size_idx_h3, size_t size_idx_h7, size_t size_idx_p4, size_t size_idx_p5, size_t size_idx_p6, double *triplesx, double *t2sub, double *v2sub) {
  #pragma omp parallel for collapse(6) 
  for (size_t t3_h3 = 0; t3_h3 < size_idx_h3; t3_h3++)
  for (size_t t3_h2 = 0; t3_h2 < size_idx_h2; t3_h2++)
  for (size_t t3_h1 = 0; t3_h1 < size_idx_h1; t3_h1++)
  for (size_t t3_p6 = 0; t3_p6 < size_idx_p6; t3_p6++)
  for (size_t t3_p5 = 0; t3_p5 < size_idx_p5; t3_p5++)
  for (size_t t3_p4 = 0; t3_p4 < size_idx_p4; t3_p4++)
  {
    size_t t3_idx   = t3_h3 + (t3_h2 + (t3_h1 + (t3_p6 + (t3_p5 + (t3_p4) * size_idx_p5) * size_idx_p6) * size_idx_h1) * size_idx_h2) * size_idx_h3;
    for (size_t t3_h7 = 0; t3_h7 < size_idx_h7; t3_h7++)
    {   
      // sd1_1:  t3[h3,h2,h1,p6,p5,p4] -= t2[h7,p4,p5,h1] * v2[h3,h2,p6,h7]
      {
        triplesx[t3_idx] -= t2sub[t3_h7 + (t3_p4 + (t3_p5 + (t3_h1) * size_idx_p5) * size_idx_p4) * size_idx_h7] * 
                            v2sub[t3_h3 + (t3_h2 + (t3_p6 + (t3_h7) * size_idx_p6) * size_idx_h2) * size_idx_h3];                            
      }
    }
  }
}


void sd_t_d1_2_cpu(size_t size_idx_h1, size_t size_idx_h2, size_t size_idx_h3, size_t size_idx_h7, size_t size_idx_p4, size_t size_idx_p5, size_t size_idx_p6, double *triplesx, double *t2sub, double *v2sub) 
{
    #pragma omp parallel for collapse(6) 
    for (size_t t3_h3 = 0; t3_h3 < size_idx_h3; t3_h3++)
    for (size_t t3_h2 = 0; t3_h2 < size_idx_h2; t3_h2++)
    for (size_t t3_h1 = 0; t3_h1 < size_idx_h1; t3_h1++)
    for (size_t t3_p6 = 0; t3_p6 < size_idx_p6; t3_p6++)
    for (size_t t3_p5 = 0; t3_p5 < size_idx_p5; t3_p5++)
    for (size_t t3_p4 = 0; t3_p4 < size_idx_p4; t3_p4++)
    {
        size_t t3_idx   = t3_h3 + (t3_h2 + (t3_h1 + (t3_p6 + (t3_p5 + (t3_p4) * size_idx_p5) * size_idx_p6) * size_idx_h1) * size_idx_h2) * size_idx_h3;
        
        for (size_t t3_h7 = 0; t3_h7 < size_idx_h7; t3_h7++)
        {   
            // sd1_2:  t3[h3,h2,h1,p6,p5,p4] += t2[h7,p4,p5,h2] * v2[h3,h1,p6,h7]
            {
                triplesx[t3_idx] += 	t2sub[t3_h7 + (t3_p4 + (t3_p5 + (t3_h2) * size_idx_p5) * size_idx_p4) * size_idx_h7] * 
                                            v2sub[t3_h3 + (t3_h1 + (t3_p6 + (t3_h7) * size_idx_p6) * size_idx_h1) * size_idx_h3];
            }
        }
    }
}

void sd_t_d1_3_cpu(size_t size_idx_h1, size_t size_idx_h2, size_t size_idx_h3, size_t size_idx_h7, size_t size_idx_p4, size_t size_idx_p5, size_t size_idx_p6, double *triplesx, double *t2sub, double *v2sub) 
{
    #pragma omp parallel for collapse(6) 
    for (size_t t3_h3 = 0; t3_h3 < size_idx_h3; t3_h3++)
    for (size_t t3_h2 = 0; t3_h2 < size_idx_h2; t3_h2++)
    for (size_t t3_h1 = 0; t3_h1 < size_idx_h1; t3_h1++)
    for (size_t t3_p6 = 0; t3_p6 < size_idx_p6; t3_p6++)
    for (size_t t3_p5 = 0; t3_p5 < size_idx_p5; t3_p5++)
    for (size_t t3_p4 = 0; t3_p4 < size_idx_p4; t3_p4++)
    {
        size_t t3_idx   = t3_h3 + (t3_h2 + (t3_h1 + (t3_p6 + (t3_p5 + (t3_p4) * size_idx_p5) * size_idx_p6) * size_idx_h1) * size_idx_h2) * size_idx_h3;
        
        for (size_t t3_h7 = 0; t3_h7 < size_idx_h7; t3_h7++)
        {       
            // sd1_3:  t3[h3,h2,h1,p6,p5,p4] -= t2[h7,p4,p5,h3] * v2[h2,h1,p6,h7]
            {
                triplesx[t3_idx] -= 	t2sub[t3_h7 + (t3_p4 + (t3_p5 + (t3_h3) * size_idx_p5) * size_idx_p4) * size_idx_h7] * 
                                            v2sub[t3_h2 + (t3_h1 + (t3_p6 + (t3_h7) * size_idx_p6) * size_idx_h1) * size_idx_h2];
            }
        }
    }
}

void sd_t_d1_4_cpu(size_t size_idx_h1, size_t size_idx_h2, size_t size_idx_h3, size_t size_idx_h7, size_t size_idx_p4, size_t size_idx_p5, size_t size_idx_p6, double *triplesx, double *t2sub, double *v2sub) 
{
    #pragma omp parallel for collapse(6) 
    for (size_t t3_h3 = 0; t3_h3 < size_idx_h3; t3_h3++)
    for (size_t t3_h2 = 0; t3_h2 < size_idx_h2; t3_h2++)
    for (size_t t3_h1 = 0; t3_h1 < size_idx_h1; t3_h1++)
    for (size_t t3_p6 = 0; t3_p6 < size_idx_p6; t3_p6++)
    for (size_t t3_p5 = 0; t3_p5 < size_idx_p5; t3_p5++)
    for (size_t t3_p4 = 0; t3_p4 < size_idx_p4; t3_p4++)
    {
        size_t t3_idx   = t3_h3 + (t3_h2 + (t3_h1 + (t3_p6 + (t3_p5 + (t3_p4) * size_idx_p5) * size_idx_p6) * size_idx_h1) * size_idx_h2) * size_idx_h3;
        
        for (size_t t3_h7 = 0; t3_h7 < size_idx_h7; t3_h7++)
        {   
            // sd1_4:  t3[h3,h2,h1,p6,p5,p4] -= t2[h7,p5,p6,h1] * v2[h3,h2,p4,h7]
            {
                triplesx[t3_idx] -= 	t2sub[t3_h7 + (t3_p5 + (t3_p6 + (t3_h1) * size_idx_p6) * size_idx_p5) * size_idx_h7] * 
                                            v2sub[t3_h3 + (t3_h2 + (t3_p4 + (t3_h7) * size_idx_p4) * size_idx_h2) * size_idx_h3];
            }
        }
    }
}

void sd_t_d1_5_cpu(size_t size_idx_h1, size_t size_idx_h2, size_t size_idx_h3, size_t size_idx_h7, size_t size_idx_p4, size_t size_idx_p5, size_t size_idx_p6, double *triplesx, double *t2sub, double *v2sub) 
{
    #pragma omp parallel for collapse(6)
    for (size_t t3_h3 = 0; t3_h3 < size_idx_h3; t3_h3++)
    for (size_t t3_h2 = 0; t3_h2 < size_idx_h2; t3_h2++)
    for (size_t t3_h1 = 0; t3_h1 < size_idx_h1; t3_h1++)
    for (size_t t3_p6 = 0; t3_p6 < size_idx_p6; t3_p6++)
    for (size_t t3_p5 = 0; t3_p5 < size_idx_p5; t3_p5++)
    for (size_t t3_p4 = 0; t3_p4 < size_idx_p4; t3_p4++)
    {
        size_t t3_idx   = t3_h3 + (t3_h2 + (t3_h1 + (t3_p6 + (t3_p5 + (t3_p4) * size_idx_p5) * size_idx_p6) * size_idx_h1) * size_idx_h2) * size_idx_h3;
        
        for (size_t t3_h7 = 0; t3_h7 < size_idx_h7; t3_h7++)
        {       
            // sd1_5:  t3[h3,h2,h1,p6,p5,p4] += t2[h7,p5,p6,h2] * v2[h3,h1,p4,h7]
            {
                triplesx[t3_idx] += 	t2sub[t3_h7 + (t3_p5 + (t3_p6 + (t3_h2) * size_idx_p6) * size_idx_p5) * size_idx_h7] * 
                                            v2sub[t3_h3 + (t3_h1 + (t3_p4 + (t3_h7) * size_idx_p4) * size_idx_h1) * size_idx_h3];
            }
        }
    }
}

void sd_t_d1_6_cpu(size_t size_idx_h1, size_t size_idx_h2, size_t size_idx_h3, size_t size_idx_h7, size_t size_idx_p4, size_t size_idx_p5, size_t size_idx_p6, double *triplesx, double *t2sub, double *v2sub) 
{
    #pragma omp parallel for collapse(6) 
    for (size_t t3_h3 = 0; t3_h3 < size_idx_h3; t3_h3++)
    for (size_t t3_h2 = 0; t3_h2 < size_idx_h2; t3_h2++)
    for (size_t t3_h1 = 0; t3_h1 < size_idx_h1; t3_h1++)
    for (size_t t3_p6 = 0; t3_p6 < size_idx_p6; t3_p6++)
    for (size_t t3_p5 = 0; t3_p5 < size_idx_p5; t3_p5++)
    for (size_t t3_p4 = 0; t3_p4 < size_idx_p4; t3_p4++)
    {
        size_t t3_idx   = t3_h3 + (t3_h2 + (t3_h1 + (t3_p6 + (t3_p5 + (t3_p4) * size_idx_p5) * size_idx_p6) * size_idx_h1) * size_idx_h2) * size_idx_h3;
        
        for (size_t t3_h7 = 0; t3_h7 < size_idx_h7; t3_h7++)
        {       
            // sd1_6:  t3[h3,h2,h1,p6,p5,p4] -= t2[h7,p5,p6,h3] * v2[h2,h1,p4,h7]
            {
                triplesx[t3_idx] -= 	t2sub[t3_h7 + (t3_p5 + (t3_p6 + (t3_h3) * size_idx_p6) * size_idx_p5) * size_idx_h7] * 
                                            v2sub[t3_h2 + (t3_h1 + (t3_p4 + (t3_h7) * size_idx_p4) * size_idx_h1) * size_idx_h2];
            }
        }
    }
}

void sd_t_d1_7_cpu(size_t size_idx_h1, size_t size_idx_h2, size_t size_idx_h3, size_t size_idx_h7, size_t size_idx_p4, size_t size_idx_p5, size_t size_idx_p6, double *triplesx, double *t2sub, double *v2sub) 
{
    #pragma omp parallel for collapse(6)
    for (size_t t3_h3 = 0; t3_h3 < size_idx_h3; t3_h3++)
    for (size_t t3_h2 = 0; t3_h2 < size_idx_h2; t3_h2++)
    for (size_t t3_h1 = 0; t3_h1 < size_idx_h1; t3_h1++)
    for (size_t t3_p6 = 0; t3_p6 < size_idx_p6; t3_p6++)
    for (size_t t3_p5 = 0; t3_p5 < size_idx_p5; t3_p5++)
    for (size_t t3_p4 = 0; t3_p4 < size_idx_p4; t3_p4++)
    {
        size_t t3_idx   = t3_h3 + (t3_h2 + (t3_h1 + (t3_p6 + (t3_p5 + (t3_p4) * size_idx_p5) * size_idx_p6) * size_idx_h1) * size_idx_h2) * size_idx_h3;
        
        for (size_t t3_h7 = 0; t3_h7 < size_idx_h7; t3_h7++)
        {   
            // sd1_7:  t3[h3,h2,h1,p6,p5,p4] += t2[h7,p4,p6,h1] * v2[h3,h2,p5,h7] 
            {
                triplesx[t3_idx] += 	t2sub[t3_h7 + (t3_p4 + (t3_p6 + (t3_h1) * size_idx_p6) * size_idx_p4) * size_idx_h7] * 
                                            v2sub[t3_h3 + (t3_h2 + (t3_p5 + (t3_h7) * size_idx_p5) * size_idx_h2) * size_idx_h3];
            }
        }
    }
}

void sd_t_d1_8_cpu(size_t size_idx_h1, size_t size_idx_h2, size_t size_idx_h3, size_t size_idx_h7, size_t size_idx_p4, size_t size_idx_p5, size_t size_idx_p6, double *triplesx, double *t2sub, double *v2sub) 
{
    #pragma omp parallel for collapse(6)
    for (size_t t3_h3 = 0; t3_h3 < size_idx_h3; t3_h3++)
    for (size_t t3_h2 = 0; t3_h2 < size_idx_h2; t3_h2++)
    for (size_t t3_h1 = 0; t3_h1 < size_idx_h1; t3_h1++)
    for (size_t t3_p6 = 0; t3_p6 < size_idx_p6; t3_p6++)
    for (size_t t3_p5 = 0; t3_p5 < size_idx_p5; t3_p5++)
    for (size_t t3_p4 = 0; t3_p4 < size_idx_p4; t3_p4++)
    {
        size_t t3_idx   = t3_h3 + (t3_h2 + (t3_h1 + (t3_p6 + (t3_p5 + (t3_p4) * size_idx_p5) * size_idx_p6) * size_idx_h1) * size_idx_h2) * size_idx_h3;
        
        for (size_t t3_h7 = 0; t3_h7 < size_idx_h7; t3_h7++)
        {           
            // sd1_8:  t3[h3,h2,h1,p6,p5,p4] -= t2[h7,p4,p6,h2] * v2[h3,h1,p5,h7]
            {
                triplesx[t3_idx] -= 	t2sub[t3_h7 + (t3_p4 + (t3_p6 + (t3_h2) * size_idx_p6) * size_idx_p4) * size_idx_h7] * 
                                            v2sub[t3_h3 + (t3_h1 + (t3_p5 + (t3_h7) * size_idx_p5) * size_idx_h1) * size_idx_h3];
            }
        }
    }
}

void sd_t_d1_9_cpu(size_t size_idx_h1, size_t size_idx_h2, size_t size_idx_h3, size_t size_idx_h7, size_t size_idx_p4, size_t size_idx_p5, size_t size_idx_p6, double *triplesx, double *t2sub, double *v2sub) 
{
        #pragma omp parallel for collapse(6) 
    for (size_t t3_h3 = 0; t3_h3 < size_idx_h3; t3_h3++)
    for (size_t t3_h2 = 0; t3_h2 < size_idx_h2; t3_h2++)
    for (size_t t3_h1 = 0; t3_h1 < size_idx_h1; t3_h1++)
    for (size_t t3_p6 = 0; t3_p6 < size_idx_p6; t3_p6++)
    for (size_t t3_p5 = 0; t3_p5 < size_idx_p5; t3_p5++)
    for (size_t t3_p4 = 0; t3_p4 < size_idx_p4; t3_p4++)
    {
        size_t t3_idx   = t3_h3 + (t3_h2 + (t3_h1 + (t3_p6 + (t3_p5 + (t3_p4) * size_idx_p5) * size_idx_p6) * size_idx_h1) * size_idx_h2) * size_idx_h3;
        
        for (size_t t3_h7 = 0; t3_h7 < size_idx_h7; t3_h7++)
        {   
            // sd1_9:  t3[h3,h2,h1,p6,p5,p4] += t2[h7,p4,p6,h3] * v2[h2,h1,p5,h7]
            {
                triplesx[t3_idx] += 	t2sub[t3_h7 + (t3_p4 + (t3_p6 + (t3_h3) * size_idx_p6) * size_idx_p4) * size_idx_h7] * 
                                            v2sub[t3_h2 + (t3_h1 + (t3_p5 + (t3_h7) * size_idx_p5) * size_idx_h1) * size_idx_h2];
            }   
        }
    }
}

/*
 *  DPCPP
 *
 */
//#include "dpc_common.hpp"
//using namespace sycl;

//------------------------------------------------------------------------------
// 	New mapping for Intel GPUs: h3 					-> tb_x, 
// 															h2, p5 			-> reg_x, 
// 															p6, h1, p4 	-> reg_y
#define F_SIZE_T_H1 	4
#define F_SIZE_T_H2 	4
#define F_SIZE_T_H3 	4
#define F_SIZE_T_H7 	4
#define F_SIZE_T_P4 	4
#define F_SIZE_T_P5 	4
#define F_SIZE_T_P6 	4
#define F_SIZE_T_P7 	F_SIZE_T_H7

#define F_SIZE_T_K		F_SIZE_T_H7
// block (2, 1)
#define F_SIZE_TB_X 	F_SIZE_T_H3 
#define F_SIZE_TB_Y 	1
// register-tile (4, 8)
#define F_SIZE_REG_X 	F_SIZE_T_H2 * F_SIZE_T_P5
#define F_SIZE_REG_Y 	F_SIZE_T_P6 * F_SIZE_T_H1 * F_SIZE_T_P4
//------------------------------------------------------------------------------
#define CEIL(a, b)      			(((a) + (b) - 1) / (b))
//------------------------------------------------------------------------------
#define NUM_EQUATIONS 9
#define NUM_D2_INDEX 	7
//------------------------------------------------------------------------------
using localAcc = sycl::accessor<sycl::cl_double, 	2, sycl::access::mode::read_write, 	sycl::access::target::local>;

// 
// 	the new (sample) fully-fused kernel for dpcpp
// 
void dpcpp_ccsd_t_unfused_sd2_1_kernel(sycl::nd_item<2>& item_ct, 
		double* dev_d2_t3, double* dev_d2_t2, double* dev_d2_v2,
		//  common
		int num_blks_h3b, int num_blks_h2b, int num_blks_h1b, 
		int num_blks_p6b, int num_blks_p5b, int num_blks_p4b, 
		// 
		int base_size_h1b, int base_size_h2b, int base_size_h3b, 
		int base_size_p4b, int base_size_p5b, int base_size_p6b, int base_size_p7b, 
		localAcc smem_a, localAcc smem_b, const sycl::stream* out) {
	// 
	int internal_upperbound = 0;
	int internal_offset;

	// 
	// 	h3 					-> tb_x, 
	// 	h2, p5 			-> reg_x 
	// 	p6, h1, p4 	-> reg_y 
	// 
	int idx_h3 = item_ct.get_local_id(1);

	// h3,h2,p5 -> blk_x
	// p6,h1,p4 -> blk_y
	int blk_idx_p5b = item_ct.get_group(0) / (num_blks_h3b * num_blks_h2b);
	int tmp_blk_idx = item_ct.get_group(0) % (num_blks_h3b * num_blks_h2b);
	int blk_idx_h2b = tmp_blk_idx / (num_blks_h3b);
	int blk_idx_h3b = tmp_blk_idx % (num_blks_h3b);

	int blk_idx_p4b = item_ct.get_group(1) / (num_blks_p6b * num_blks_h1b);
			tmp_blk_idx = item_ct.get_group(1) % (num_blks_p6b * num_blks_h1b);
	int blk_idx_h1b = tmp_blk_idx / (num_blks_p6b);
	int blk_idx_p6b = tmp_blk_idx % (num_blks_p6b);

	int str_blk_idx_h3 = blk_idx_h3b * F_SIZE_T_H3;
	int str_blk_idx_h2 = blk_idx_h2b * F_SIZE_T_H2;
	int str_blk_idx_h1 = blk_idx_h1b * F_SIZE_T_H1;
	int str_blk_idx_p6 = blk_idx_p6b * F_SIZE_T_P6;
	int str_blk_idx_p5 = blk_idx_p5b * F_SIZE_T_P5;
	int str_blk_idx_p4 = blk_idx_p4b * F_SIZE_T_P4;

	// 	(4) rng_h/p*
	int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;
	if ((base_size_h3b - (str_blk_idx_h3)) >= F_SIZE_T_H3) { rng_h3 = F_SIZE_T_H3; }
	else { rng_h3 = base_size_h3b % F_SIZE_T_H3; }
	
	if ((base_size_h2b - (str_blk_idx_h2)) >= F_SIZE_T_H2) { rng_h2 = F_SIZE_T_H2; }
	else { rng_h2 = base_size_h2b % F_SIZE_T_H2; }

	if ((base_size_h1b - (str_blk_idx_h1)) >= F_SIZE_T_H1) { rng_h1 = F_SIZE_T_H2; }
	else { rng_h1 = base_size_h1b % F_SIZE_T_H1; }
	
	if ((base_size_p6b - (str_blk_idx_p6)) >= F_SIZE_T_P6) { rng_p6 = F_SIZE_T_P6; }
	else { rng_p6 = base_size_p6b % F_SIZE_T_P6; }

	if ((base_size_p5b - (str_blk_idx_p5)) >= F_SIZE_T_P5) { rng_p5 = F_SIZE_T_P5; }
	else { rng_p5 = base_size_p5b % F_SIZE_T_P5; }

	if ((base_size_p4b - (str_blk_idx_p4)) >= F_SIZE_T_P4) { rng_p4 = F_SIZE_T_P4; }
	else {rng_p4 = base_size_p4b % F_SIZE_T_P4; }

	// 
	double tmp_av;
	double tmp_bv;
	double reg_doubles[F_SIZE_T_P4 * F_SIZE_T_H1 * F_SIZE_T_P6][F_SIZE_T_H2 * F_SIZE_T_P5];

	// 
	for (int j = 0; j < F_SIZE_T_H2 * F_SIZE_T_P5; j++) 
	for (int i = 0; i < F_SIZE_T_P4 * F_SIZE_T_H1 * F_SIZE_T_P6; i++) 
	{ reg_doubles[i][j] = 0.0f; }

#if 1
	// sd_t_d2_1 :
	// t3[h3,h2,h1,p6,p5,p4] -= t2[p7,p4,h1,h2]*v2[p7,h3,p6,p5]
	// t2[p7,p4,h1,h2] -> sm_a[p7][p4,h1,h2]
	// v2[p7,h3,p6,p5] -> sm_b[p7][h3,p6,p5]
	internal_upperbound = 0;
	for (int l = 0; l < base_size_p7b; l+= F_SIZE_T_K)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + F_SIZE_T_K) - base_size_p7b;
		if (internal_offset > 0) internal_upperbound = internal_offset;
	
		// 
		// 	h3 					-> tb_x, 
		// 	h2, p5 			-> reg_x 
		// 	p6, h1, p4 	-> reg_y 
		// 
		// 	sd2_1: t3[h3,h2,h1,p6,p5,p4] -= t2[p7,p4,h1,h2]*v2[p7,h3,p6,p5]
		// 
		// 	block (2, 1) ---> smem[2][8] // 8 times
		// 
		if (item_ct.get_local_id(0) < F_SIZE_T_K - internal_upperbound) // internal
		{
			for (int ll = 0; ll < 64; ll++)
			{
				// t2[p7,p4,h1,h2] = t2[8,4,4,4] 
				int tmp_idx_h2 = ll / (F_SIZE_T_P4 * F_SIZE_T_H1); 
				int tmp_idx_hp = ll % (F_SIZE_T_P4 * F_SIZE_T_H1); 
				int tmp_idx_h1 = tmp_idx_hp / F_SIZE_T_P4; 
				int tmp_idx_p4 = tmp_idx_hp % F_SIZE_T_P4;

				// t2[p7,p4,h1,h2] -> sm_a[p7][p4,h1,h2]
				if (tmp_idx_p4 < rng_p4 && tmp_idx_h1 < rng_h1 && tmp_idx_h2 < rng_h2) // external
				smem_a[item_ct.get_local_id(0)][item_ct.get_local_id(1) + ll] = dev_d2_t2[item_ct.get_local_id(0) + l + (str_blk_idx_p4 + tmp_idx_p4 + (str_blk_idx_h1 + tmp_idx_h1 + (str_blk_idx_h2 + tmp_idx_h2) * base_size_h1b) * base_size_p4b) * base_size_p7b];
			
				// v2[p7,h3,p6,p5] -> sm_b[p7][h3,p6,p5]
				// if (tmp_idx_h3 < rng_h3 && tmp_idx_p6 < rng_p6 && tmp_idx_p5 < rng_p5)
				if (tmp_idx_p4 < rng_h3 && tmp_idx_h1 < rng_p6 && tmp_idx_h2 < rng_p5)
				smem_b[item_ct.get_local_id(0)][item_ct.get_local_id(1) + ll] = dev_d2_v2[item_ct.get_local_id(0) + l + (str_blk_idx_h3 + tmp_idx_p4 + (str_blk_idx_p6 + tmp_idx_h1 + (str_blk_idx_p5 + tmp_idx_h2) * base_size_p6b) * base_size_h3b) * base_size_p7b];
			}
		}
		item_ct.barrier();

		// Outer-Product
		if (item_ct.get_local_id(0) < rng_h3)
		for (int ll = 0; ll < F_SIZE_T_K - internal_upperbound; ll++)
		{
			// t2[p7,p4,h1,h2] -> sm_a[p7][p4,h1,h2]
			// v2[p7,h3,p6,p5] -> sm_b[p7][h3,p6,p5]
			// (1) reg[p4,h1,p6][h2,p5] vs. (2) reg[h2,p5][p4,h1,p6]
				
			// T tmp_av[F_SIZE_T_H2 * F_SIZE_T_P5];
			// T tmp_bv[F_SIZE_T_P4 * F_SIZE_T_H1 * F_SIZE_T_P6];
			// 	--> (1) two for
			// 	--> (2) one for with bv
			// 	--> (3) one for with av
			//  --> (4) av and bv
			for (int aa = 0; aa < F_SIZE_T_H2 * F_SIZE_T_P5; aa++)
			{
				int tmp_p5 = aa / F_SIZE_T_H2;	// p5
				int tmp_h2 = aa % F_SIZE_T_H2;	// h2

				if (tmp_p5 < rng_p5 && tmp_h2 < rng_h2)
				for (int bb = 0; bb < F_SIZE_T_P4 * F_SIZE_T_H1 * F_SIZE_T_P6; bb++)
				{
					int tmp_p6 = bb / (F_SIZE_T_P4 * F_SIZE_T_H1);	// p6
					int tmp_bb = bb % (F_SIZE_T_P4 * F_SIZE_T_H1);
					int tmp_h1 = tmp_bb / F_SIZE_T_P4;							// h1
					int tmp_p4 = tmp_bb % F_SIZE_T_P4;							// p4

					// sm_a[p7][p4,h1,h2]
					// sm_b[p7][h3,p6,p5]
					// if (threadIdx.x < rng_h3 && tmp_h2 < rng_h2 && tmp_h1 < rng_h1 && tmp_p6 < rng_p6 && tmp_p5 < rng_p5 && tmp_p4 < rng_p4)
					if (tmp_h1 < rng_h1 && tmp_p6 < rng_p6 && tmp_p4 < rng_p4)
					reg_doubles[bb][aa] -= 	smem_a[ll][tmp_p4 + (tmp_h1 + (tmp_h2) * F_SIZE_T_H1) * F_SIZE_T_P4] * 
																	smem_b[ll][item_ct.get_local_id(0) + (tmp_p6 + (tmp_p5) * F_SIZE_T_P6) * F_SIZE_T_H3];
				}
			}
		}
		item_ct.barrier();
	}
#endif

	// 
	// 
	// 
	if (item_ct.get_local_id(0) < rng_h3)
	for (int j = 0; j < F_SIZE_T_H2 * F_SIZE_T_P5; j++) {
		int idx_p5 = j / F_SIZE_T_H2;
		int idx_h2 = j % F_SIZE_T_H2;
		// 
		if (idx_p5 < rng_p5 && idx_h2 < rng_h2)
		for (int i = 0; i < F_SIZE_T_P4 * F_SIZE_T_H1 * F_SIZE_T_P6; i++) {
			int idx_p6 = i / (F_SIZE_T_P4 * F_SIZE_T_H1);
			int tmp_ii = i % (F_SIZE_T_P4 * F_SIZE_T_H1);
			int idx_h1 = tmp_ii / F_SIZE_T_P4;
			int idx_p4 = tmp_ii % F_SIZE_T_P4;
		
			// 
			size_t t3_base_addr =  str_blk_idx_h3 + item_ct.get_local_id(0) + 
														(str_blk_idx_h2 + idx_h2 +  
														(str_blk_idx_h1 + idx_h1 + 
														(str_blk_idx_p6 + idx_p6 + 
														(str_blk_idx_p5 + idx_p5 + 
														(str_blk_idx_p4 + idx_p4) * base_size_p5b) * base_size_p6b) * base_size_h1b) * base_size_h2b) * base_size_h3b;
			// 
			if (idx_h1 < rng_h1 && idx_p6 < rng_p6 && idx_p4 < rng_p4)
			{ dev_d2_t3[t3_base_addr] += reg_doubles[i][j]; }
		}
	}
}

// 
// 	a driver for dpcpp
// 
void dpcpp_ready_ccsd_t_unfused_sd2_1_kernel_driver(cl::sycl::queue* syclQueue, 
		int base_size_h1b, int base_size_h2b, int base_size_h3b, 
		int base_size_p4b, int base_size_p5b, int base_size_p6b, int base_size_p7b, 
		double* host_d2_t3, double* host_d2_t2, double* host_d2_v2) {
  std::cout << "[" << __func__ << "]" << std::endl;
	// 
	sycl::queue syclQ = *syclQueue;
	unsigned int num_grid_x = CEIL(base_size_h3b, F_SIZE_T_H3) * CEIL(base_size_h2b, F_SIZE_T_H2) * CEIL(base_size_p5b, F_SIZE_T_P5);
	unsigned int num_grid_y = CEIL(base_size_p6b, F_SIZE_T_P6) * CEIL(base_size_h1b, F_SIZE_T_H1) * CEIL(base_size_p4b, F_SIZE_T_P4);
  
  //
  size_t size_d2_t3 = base_size_h3b * base_size_h2b * base_size_h1b * base_size_p6b * base_size_p5b * base_size_p4b;
  size_t size_d2_t2 = base_size_p7b * base_size_p4b * base_size_h1b * base_size_h2b;
  size_t size_d2_v2 = base_size_p7b * base_size_h3b * base_size_p6b * base_size_p5b;

	// 
	double* dev_d2_t3 = (double*)sycl::malloc_device(sizeof(double) * size_d2_t3, syclQ);
	double* dev_d2_t2 = (double*)sycl::malloc_device(sizeof(double) * size_d2_t2, syclQ);
	double* dev_d2_v2 = (double*)sycl::malloc_device(sizeof(double) * size_d2_v2, syclQ);

	// 
	syclQ.memcpy(dev_d2_t3, host_d2_t3, sizeof(double) * (size_d2_t3));
	syclQ.memcpy(dev_d2_t2, host_d2_t2, sizeof(double) * (size_d2_t2));
	syclQ.memcpy(dev_d2_v2, host_d2_v2, sizeof(double) * (size_d2_v2));

	// 
	sycl::range<2> size_grid(num_grid_x, num_grid_y);
	sycl::range<2> size_block(F_SIZE_TB_X, F_SIZE_TB_Y); // (dim0,dim1) = (2,1)

#if 1
	// 
	// 		[main]
	// 
	syclQueue->submit([&](sycl::handler &cgh)
	{
		// 
		sycl::stream out(1024, 256, cgh);
		// related to shared memories
		sycl::range<2> smem_a(4, 64);
		sycl::range<2> smem_b(4, 64);
		localAcc smem_a_acc(smem_a, cgh);
		localAcc smem_b_acc(smem_b, cgh);
		// 
		auto global_range = size_grid * size_block; // (GRID_Y * TB_Y, GRID_X * TB_X)
		// 
		cgh.parallel_for(sycl::nd_range<2>(global_range, size_block), [=](sycl::nd_item<2> item_ct)
		{
			// 
			dpcpp_ccsd_t_unfused_sd2_1_kernel(item_ct, 
				// 
				dev_d2_t3, dev_d2_t2, dev_d2_v2, 
				//
				CEIL(base_size_h3b, F_SIZE_T_H3), CEIL(base_size_h2b, F_SIZE_T_H2), CEIL(base_size_h1b, F_SIZE_T_H1), 
				CEIL(base_size_p4b, F_SIZE_T_P4), CEIL(base_size_p5b, F_SIZE_T_P6), CEIL(base_size_p6b, F_SIZE_T_P6), 
				(sycl::cl_int)base_size_h1b, (sycl::cl_int)base_size_h2b, (sycl::cl_int)base_size_h3b, 
				(sycl::cl_int)base_size_p4b, (sycl::cl_int)base_size_p5b, (sycl::cl_int)base_size_p6b, (sycl::cl_int)base_size_p7b, 
				// 
				smem_a_acc, smem_b_acc, &out);
		});
	});

	// 
	syclQ.memcpy(host_d2_t3, dev_d2_t3, sizeof(double) * (size_d2_t3));
	syclQ.wait_and_throw();
#endif
	// double time_end = omp_get_wtime();

	// 
	sycl::free(dev_d2_t3, syclQ);
	sycl::free(dev_d2_t2, syclQ); sycl::free(dev_d2_v2, syclQ);
}



void sd_t_d2_1_cpu(size_t size_idx_h1, size_t size_idx_h2, size_t size_idx_h3, size_t size_idx_p4, size_t size_idx_p5, size_t size_idx_p6, size_t size_idx_p7, double *triplesx, double *t2sub, double *v2sub) {
  std::cout << "[" << __func__ << "]" << std::endl;
  #pragma omp parallel for collapse(6)
  for (size_t t3_h3 = 0; t3_h3 < size_idx_h3; t3_h3++)
  for (size_t t3_h2 = 0; t3_h2 < size_idx_h2; t3_h2++)
  for (size_t t3_h1 = 0; t3_h1 < size_idx_h1; t3_h1++)
  for (size_t t3_p6 = 0; t3_p6 < size_idx_p6; t3_p6++)
  for (size_t t3_p5 = 0; t3_p5 < size_idx_p5; t3_p5++)
  for (size_t t3_p4 = 0; t3_p4 < size_idx_p4; t3_p4++)
  {
    size_t t3_idx   = t3_h3 + (t3_h2 + (t3_h1 + (t3_p6 + (t3_p5 + (t3_p4) * size_idx_p5) * size_idx_p6) * size_idx_h1) * size_idx_h2) * size_idx_h3;
    for (size_t t3_p7 = 0; t3_p7 < size_idx_p7; t3_p7++)
    {
      // sd2_1:  t3[h3,h2,h1,p6,p5,p4] −= t2[p7,p4,h1,h2] * v2[p7,h3,p6,p5]	
      {
        triplesx[t3_idx] -= t2sub[t3_p7 + (t3_p4 + (t3_h1 + (t3_h2) * size_idx_h1) * size_idx_p4) * size_idx_p7] * 
                            v2sub[t3_p7 + (t3_h3 + (t3_p6 + (t3_p5) * size_idx_p6) * size_idx_h3) * size_idx_p7];
      }
    }
  }
}

void sd_t_d2_2_cpu(size_t size_idx_h1, size_t size_idx_h2, size_t size_idx_h3, size_t size_idx_p4, size_t size_idx_p5, size_t size_idx_p6, size_t size_idx_p7, double *triplesx, double *t2sub, double *v2sub) 
{
    #pragma omp parallel for collapse(6) reduction(+: ops)
    for (size_t t3_h3 = 0; t3_h3 < size_idx_h3; t3_h3++)
    for (size_t t3_h2 = 0; t3_h2 < size_idx_h2; t3_h2++)
    for (size_t t3_h1 = 0; t3_h1 < size_idx_h1; t3_h1++)
    for (size_t t3_p6 = 0; t3_p6 < size_idx_p6; t3_p6++)
    for (size_t t3_p5 = 0; t3_p5 < size_idx_p5; t3_p5++)
    for (size_t t3_p4 = 0; t3_p4 < size_idx_p4; t3_p4++)
    {
        size_t t3_idx   = t3_h3 + (t3_h2 + (t3_h1 + (t3_p6 + (t3_p5 + (t3_p4) * size_idx_p5) * size_idx_p6) * size_idx_h1) * size_idx_h2) * size_idx_h3;

        for (size_t t3_p7 = 0; t3_p7 < size_idx_p7; t3_p7++)
        {
            // sd2_2:  t3[h3,h2,h1,p6,p5,p4] −= t2[p7,p4,h2,h3] * v2[p7,h1,p6,p5] 
            {
                triplesx[t3_idx] -= 	t2sub[t3_p7 + (t3_p4 + (t3_h2 + (t3_h3) * size_idx_h2) * size_idx_p4) * size_idx_p7] * 
                                            v2sub[t3_p7 + (t3_h1 + (t3_p6 + (t3_p5) * size_idx_p6) * size_idx_h1) * size_idx_p7];
            }
        }
    }
}

void sd_t_d2_3_cpu(size_t size_idx_h1, size_t size_idx_h2, size_t size_idx_h3, size_t size_idx_p4, size_t size_idx_p5, size_t size_idx_p6, size_t size_idx_p7, double *triplesx, double *t2sub, double *v2sub) 
{
    #pragma omp parallel for collapse(6)
    for (size_t t3_h3 = 0; t3_h3 < size_idx_h3; t3_h3++)
    for (size_t t3_h2 = 0; t3_h2 < size_idx_h2; t3_h2++)
    for (size_t t3_h1 = 0; t3_h1 < size_idx_h1; t3_h1++)
    for (size_t t3_p6 = 0; t3_p6 < size_idx_p6; t3_p6++)
    for (size_t t3_p5 = 0; t3_p5 < size_idx_p5; t3_p5++)
    for (size_t t3_p4 = 0; t3_p4 < size_idx_p4; t3_p4++)
    {
        size_t t3_idx   = t3_h3 + (t3_h2 + (t3_h1 + (t3_p6 + (t3_p5 + (t3_p4) * size_idx_p5) * size_idx_p6) * size_idx_h1) * size_idx_h2) * size_idx_h3;

        for (size_t t3_p7 = 0; t3_p7 < size_idx_p7; t3_p7++)
        {
            // sd2_3:  t3[h3,h2,h1,p6,p5,p4] += t2[p7,p4,h1,h3] * v2[p7,h2,p6,p5] 
            {   
                triplesx[t3_idx] += 	t2sub[t3_p7 + (t3_p4 + (t3_h1 + (t3_h3) * size_idx_h1) * size_idx_p4) * size_idx_p7] * 
                                            v2sub[t3_p7 + (t3_h2 + (t3_p6 + (t3_p5) * size_idx_p6) * size_idx_h2) * size_idx_p7];
            }
        }
    }
}

void sd_t_d2_4_cpu(size_t size_idx_h1, size_t size_idx_h2, size_t size_idx_h3, size_t size_idx_p4, size_t size_idx_p5, size_t size_idx_p6, size_t size_idx_p7, double *triplesx, double *t2sub, double *v2sub) 
{
    #pragma omp parallel for collapse(6)
    for (size_t t3_h3 = 0; t3_h3 < size_idx_h3; t3_h3++)
    for (size_t t3_h2 = 0; t3_h2 < size_idx_h2; t3_h2++)
    for (size_t t3_h1 = 0; t3_h1 < size_idx_h1; t3_h1++)
    for (size_t t3_p6 = 0; t3_p6 < size_idx_p6; t3_p6++)
    for (size_t t3_p5 = 0; t3_p5 < size_idx_p5; t3_p5++)
    for (size_t t3_p4 = 0; t3_p4 < size_idx_p4; t3_p4++)
    {
        size_t t3_idx   = t3_h3 + (t3_h2 + (t3_h1 + (t3_p6 + (t3_p5 + (t3_p4) * size_idx_p5) * size_idx_p6) * size_idx_h1) * size_idx_h2) * size_idx_h3;

        for (size_t t3_p7 = 0; t3_p7 < size_idx_p7; t3_p7++)
        {
            // sd2_4:  t3[h3,h2,h1,p6,p5,p4] += t2[p7,p5,h1,h2] * v2[p7,h3,p6,p4]                         
            {
                triplesx[t3_idx] += 	t2sub[t3_p7 + (t3_p5 + (t3_h1 + (t3_h2) * size_idx_h1) * size_idx_p5) * size_idx_p7] * 
                                            v2sub[t3_p7 + (t3_h3 + (t3_p6 + (t3_p4) * size_idx_p6) * size_idx_h3) * size_idx_p7];
            }
        }
    }
}

void sd_t_d2_5_cpu(size_t size_idx_h1, size_t size_idx_h2, size_t size_idx_h3, size_t size_idx_p4, size_t size_idx_p5, size_t size_idx_p6, size_t size_idx_p7, double *triplesx, double *t2sub, double *v2sub) 
{
    #pragma omp parallel for collapse(6) 
    for (size_t t3_h3 = 0; t3_h3 < size_idx_h3; t3_h3++)
    for (size_t t3_h2 = 0; t3_h2 < size_idx_h2; t3_h2++)
    for (size_t t3_h1 = 0; t3_h1 < size_idx_h1; t3_h1++)
    for (size_t t3_p6 = 0; t3_p6 < size_idx_p6; t3_p6++)
    for (size_t t3_p5 = 0; t3_p5 < size_idx_p5; t3_p5++)
    for (size_t t3_p4 = 0; t3_p4 < size_idx_p4; t3_p4++)
    {
        size_t t3_idx   = t3_h3 + (t3_h2 + (t3_h1 + (t3_p6 + (t3_p5 + (t3_p4) * size_idx_p5) * size_idx_p6) * size_idx_h1) * size_idx_h2) * size_idx_h3;

        for (size_t t3_p7 = 0; t3_p7 < size_idx_p7; t3_p7++)
        {
            // sd2_5:  t3[h3,h2,h1,p6,p5,p4] += t2[p7,p5,h2,h3] * v2[p7,h1,p6,p4] 
            
            {
                triplesx[t3_idx] += 	t2sub[t3_p7 + (t3_p5 + (t3_h2 + (t3_h3) * size_idx_h2) * size_idx_p5) * size_idx_p7] * 
                                            v2sub[t3_p7 + (t3_h1 + (t3_p6 + (t3_p4) * size_idx_p6) * size_idx_h1) * size_idx_p7];
            }
        }
    }
}

void sd_t_d2_6_cpu(size_t size_idx_h1, size_t size_idx_h2, size_t size_idx_h3, size_t size_idx_p4, size_t size_idx_p5, size_t size_idx_p6, size_t size_idx_p7, double *triplesx, double *t2sub, double *v2sub) 
{
    #pragma omp parallel for collapse(6)
    for (size_t t3_h3 = 0; t3_h3 < size_idx_h3; t3_h3++)
    for (size_t t3_h2 = 0; t3_h2 < size_idx_h2; t3_h2++)
    for (size_t t3_h1 = 0; t3_h1 < size_idx_h1; t3_h1++)
    for (size_t t3_p6 = 0; t3_p6 < size_idx_p6; t3_p6++)
    for (size_t t3_p5 = 0; t3_p5 < size_idx_p5; t3_p5++)
    for (size_t t3_p4 = 0; t3_p4 < size_idx_p4; t3_p4++)
    {
        size_t t3_idx   = t3_h3 + (t3_h2 + (t3_h1 + (t3_p6 + (t3_p5 + (t3_p4) * size_idx_p5) * size_idx_p6) * size_idx_h1) * size_idx_h2) * size_idx_h3;

        for (size_t t3_p7 = 0; t3_p7 < size_idx_p7; t3_p7++)
        {
            // sd2_6:  t3[h3,h2,h1,p6,p5,p4] −= t2[p7,p5,h1,h3] * v2[p7,h2,p6,p4] 
            
            {
                triplesx[t3_idx] -= 	t2sub[t3_p7 + (t3_p5 + (t3_h1 + (t3_h3) * size_idx_h1) * size_idx_p5) * size_idx_p7] * 
                                            v2sub[t3_p7 + (t3_h2 + (t3_p6 + (t3_p4) * size_idx_p6) * size_idx_h2) * size_idx_p7];
            }
        }
    }
}

void sd_t_d2_7_cpu(size_t size_idx_h1, size_t size_idx_h2, size_t size_idx_h3, size_t size_idx_p4, size_t size_idx_p5, size_t size_idx_p6, size_t size_idx_p7, double *triplesx, double *t2sub, double *v2sub) 
{
    #pragma omp parallel for collapse(6) 
    for (size_t t3_h3 = 0; t3_h3 < size_idx_h3; t3_h3++)
    for (size_t t3_h2 = 0; t3_h2 < size_idx_h2; t3_h2++)
    for (size_t t3_h1 = 0; t3_h1 < size_idx_h1; t3_h1++)
    for (size_t t3_p6 = 0; t3_p6 < size_idx_p6; t3_p6++)
    for (size_t t3_p5 = 0; t3_p5 < size_idx_p5; t3_p5++)
    for (size_t t3_p4 = 0; t3_p4 < size_idx_p4; t3_p4++)
    {
        size_t t3_idx   = t3_h3 + (t3_h2 + (t3_h1 + (t3_p6 + (t3_p5 + (t3_p4) * size_idx_p5) * size_idx_p6) * size_idx_h1) * size_idx_h2) * size_idx_h3;

        for (size_t t3_p7 = 0; t3_p7 < size_idx_p7; t3_p7++)
        {    
            // sd2_7:  t3[h3,h2,h1,p6,p5,p4] −= t2[p7,p6,h1,h2] * v2[p7,h3,p5,p4] 
            
            {   
                triplesx[t3_idx] -= 	t2sub[t3_p7 + (t3_p6 + (t3_h1 + (t3_h2) * size_idx_h1) * size_idx_p6) * size_idx_p7] * 
                                            v2sub[t3_p7 + (t3_h3 + (t3_p5 + (t3_p4) * size_idx_p5) * size_idx_h3) * size_idx_p7];
            }
        }
    }
}

void sd_t_d2_8_cpu(size_t size_idx_h1, size_t size_idx_h2, size_t size_idx_h3, size_t size_idx_p4, size_t size_idx_p5, size_t size_idx_p6, size_t size_idx_p7, double *triplesx, double *t2sub, double *v2sub) 
{
    #pragma omp parallel for collapse(6) 
    for (size_t t3_h3 = 0; t3_h3 < size_idx_h3; t3_h3++)
    for (size_t t3_h2 = 0; t3_h2 < size_idx_h2; t3_h2++)
    for (size_t t3_h1 = 0; t3_h1 < size_idx_h1; t3_h1++)
    for (size_t t3_p6 = 0; t3_p6 < size_idx_p6; t3_p6++)
    for (size_t t3_p5 = 0; t3_p5 < size_idx_p5; t3_p5++)
    for (size_t t3_p4 = 0; t3_p4 < size_idx_p4; t3_p4++)
    {
        size_t t3_idx   = t3_h3 + (t3_h2 + (t3_h1 + (t3_p6 + (t3_p5 + (t3_p4) * size_idx_p5) * size_idx_p6) * size_idx_h1) * size_idx_h2) * size_idx_h3;

        for (size_t t3_p7 = 0; t3_p7 < size_idx_p7; t3_p7++)
        {    
            // sd2_8:  t3[h3,h2,h1,p6,p5,p4] −= t2[p7,p6,h2,h3] * v2[p7,h1,p5,p4] 
            {
                triplesx[t3_idx] -= 	t2sub[t3_p7 + (t3_p6 + (t3_h2 + (t3_h3) * size_idx_h2) * size_idx_p6) * size_idx_p7] * 
                                            v2sub[t3_p7 + (t3_h1 + (t3_p5 + (t3_p4) * size_idx_p5) * size_idx_h1) * size_idx_p7];
            }
        }
    }
}

void sd_t_d2_9_cpu(size_t size_idx_h1, size_t size_idx_h2, size_t size_idx_h3, size_t size_idx_p4, size_t size_idx_p5, size_t size_idx_p6, size_t size_idx_p7, double *triplesx, double *t2sub, double *v2sub) 
{
    #pragma omp parallel for collapse(6) 
    for (size_t t3_h3 = 0; t3_h3 < size_idx_h3; t3_h3++)
    for (size_t t3_h2 = 0; t3_h2 < size_idx_h2; t3_h2++)
    for (size_t t3_h1 = 0; t3_h1 < size_idx_h1; t3_h1++)
    for (size_t t3_p6 = 0; t3_p6 < size_idx_p6; t3_p6++)
    for (size_t t3_p5 = 0; t3_p5 < size_idx_p5; t3_p5++)
    for (size_t t3_p4 = 0; t3_p4 < size_idx_p4; t3_p4++)
    {
        size_t t3_idx   = t3_h3 + (t3_h2 + (t3_h1 + (t3_p6 + (t3_p5 + (t3_p4) * size_idx_p5) * size_idx_p6) * size_idx_h1) * size_idx_h2) * size_idx_h3;

        for (size_t t3_p7 = 0; t3_p7 < size_idx_p7; t3_p7++)
        {    
            // sd2_9:  t3[h3,h2,h1,p6,p5,p4] += t2[p7,p6,h1,h3] * v2[p7,h2,p5,p4]
            {
                triplesx[t3_idx] += 	t2sub[t3_p7 + (t3_p6 + (t3_h1 + (t3_h3) * size_idx_h1) * size_idx_p6) * size_idx_p7] * 
                                            v2sub[t3_p7 + (t3_h2 + (t3_p5 + (t3_p4) * size_idx_p5) * size_idx_h2) * size_idx_p7];
            }
        
        }
    }
}

void sd_t_s1_1_cpu(size_t size_idx_h1, size_t size_idx_h2, size_t size_idx_h3, size_t size_idx_p4, size_t size_idx_p5, size_t size_idx_p6, double *triplesx, double *t1sub, double *v2sub)
{
    #pragma omp parallel for collapse(6) 
    for (size_t t3_h3 = 0; t3_h3 < size_idx_h3; t3_h3++)
    for (size_t t3_h2 = 0; t3_h2 < size_idx_h2; t3_h2++)
    for (size_t t3_h1 = 0; t3_h1 < size_idx_h1; t3_h1++)
    for (size_t t3_p6 = 0; t3_p6 < size_idx_p6; t3_p6++)
    for (size_t t3_p5 = 0; t3_p5 < size_idx_p5; t3_p5++)
    for (size_t t3_p4 = 0; t3_p4 < size_idx_p4; t3_p4++)
    {
        size_t t3_idx = t3_h3 + (t3_h2 + (t3_h1 + (t3_p6 + (t3_p5 + (t3_p4) * size_idx_p5) * size_idx_p6) * size_idx_h1) * size_idx_h2) * size_idx_h3;
    
        //  s1_1: t3[h3,h2,h1,p6,p5,p4] += t1[p4,h1] * v2[h3,h2,p6,p5]
        {
            triplesx[t3_idx] +=  t1sub[t3_p4 + (t3_h1) * size_idx_p4] * 
                                        v2sub[t3_h3 + (t3_h2 + (t3_p6 + (t3_p5) * size_idx_p6) * size_idx_h2) * size_idx_h3];

        }
    }
}

void sd_t_s1_2_cpu(size_t size_idx_h1, size_t size_idx_h2, size_t size_idx_h3, size_t size_idx_p4, size_t size_idx_p5, size_t size_idx_p6, double *triplesx, double *t1sub, double *v2sub)
{
    #pragma omp parallel for collapse(6) 
    for (size_t t3_h3 = 0; t3_h3 < size_idx_h3; t3_h3++)
    for (size_t t3_h2 = 0; t3_h2 < size_idx_h2; t3_h2++)
    for (size_t t3_h1 = 0; t3_h1 < size_idx_h1; t3_h1++)
    for (size_t t3_p6 = 0; t3_p6 < size_idx_p6; t3_p6++)
    for (size_t t3_p5 = 0; t3_p5 < size_idx_p5; t3_p5++)
    for (size_t t3_p4 = 0; t3_p4 < size_idx_p4; t3_p4++)
    {
        size_t t3_idx = t3_h3 + (t3_h2 + (t3_h1 + (t3_p6 + (t3_p5 + (t3_p4) * size_idx_p5) * size_idx_p6) * size_idx_h1) * size_idx_h2) * size_idx_h3;
    
        // s1_2:   t3[h3,h2,h1,p6,p5,p4] -= t1[p4,h2] * v2[h3,h1,p6,p5]
        {
            triplesx[t3_idx] -= 	t1sub[t3_p4 + (t3_h2) * size_idx_p4] * 
                                        v2sub[t3_h3 + (t3_h1 + (t3_p6 + (t3_p5) * size_idx_p6) * size_idx_h1) * size_idx_h3];
        }
    }
}

void sd_t_s1_3_cpu(size_t size_idx_h1, size_t size_idx_h2, size_t size_idx_h3, size_t size_idx_p4, size_t size_idx_p5, size_t size_idx_p6, double *triplesx, double *t1sub, double *v2sub)
{
    #pragma omp parallel for collapse(6) 
    for (size_t t3_h3 = 0; t3_h3 < size_idx_h3; t3_h3++)
    for (size_t t3_h2 = 0; t3_h2 < size_idx_h2; t3_h2++)
    for (size_t t3_h1 = 0; t3_h1 < size_idx_h1; t3_h1++)
    for (size_t t3_p6 = 0; t3_p6 < size_idx_p6; t3_p6++)
    for (size_t t3_p5 = 0; t3_p5 < size_idx_p5; t3_p5++)
    for (size_t t3_p4 = 0; t3_p4 < size_idx_p4; t3_p4++)
    {
        size_t t3_idx = t3_h3 + (t3_h2 + (t3_h1 + (t3_p6 + (t3_p5 + (t3_p4) * size_idx_p5) * size_idx_p6) * size_idx_h1) * size_idx_h2) * size_idx_h3;
    
        // s1_3:   t3[h1,h3,h2,p6,p5,p4] -= t1[p4,h1] * v2[h3,h2,p6,p5] >> t3[h3,h2,h1,p6,p5,p4] += t1[p4,h3] * v2[h2,h1,p6,p5] *****
        {   
            triplesx[t3_idx] += 	t1sub[t3_p4 + (t3_h3) * size_idx_p4] * 
                                        v2sub[t3_h2 + (t3_h1 + (t3_p6 + (t3_p5) * size_idx_p6) * size_idx_h1) * size_idx_h2];
        }
    }
}

void sd_t_s1_4_cpu(size_t size_idx_h1, size_t size_idx_h2, size_t size_idx_h3, size_t size_idx_p4, size_t size_idx_p5, size_t size_idx_p6, double *triplesx, double *t1sub, double *v2sub)
{
    #pragma omp parallel for collapse(6) 
    for (size_t t3_h3 = 0; t3_h3 < size_idx_h3; t3_h3++)
    for (size_t t3_h2 = 0; t3_h2 < size_idx_h2; t3_h2++)
    for (size_t t3_h1 = 0; t3_h1 < size_idx_h1; t3_h1++)
    for (size_t t3_p6 = 0; t3_p6 < size_idx_p6; t3_p6++)
    for (size_t t3_p5 = 0; t3_p5 < size_idx_p5; t3_p5++)
    for (size_t t3_p4 = 0; t3_p4 < size_idx_p4; t3_p4++)
    {
        size_t t3_idx = t3_h3 + (t3_h2 + (t3_h1 + (t3_p6 + (t3_p5 + (t3_p4) * size_idx_p5) * size_idx_p6) * size_idx_h1) * size_idx_h2) * size_idx_h3;
    
        // s1_4:   t3[h3,h2,h1,p6,p5,p4] -= t1[p5,h1] * v2[h3,h2,p6,p4] 
        {
            triplesx[t3_idx] -= 	t1sub[t3_p5 + (t3_h1) * size_idx_p5] * 
                                        v2sub[t3_h3 + (t3_h2 + (t3_p6 + (t3_p4) * size_idx_p6) * size_idx_h2) * size_idx_h3];
        }
    }
}

void sd_t_s1_5_cpu(size_t size_idx_h1, size_t size_idx_h2, size_t size_idx_h3, size_t size_idx_p4, size_t size_idx_p5, size_t size_idx_p6, double *triplesx, double *t1sub, double *v2sub)
{
    #pragma omp parallel for collapse(6)
    for (size_t t3_h3 = 0; t3_h3 < size_idx_h3; t3_h3++)
    for (size_t t3_h2 = 0; t3_h2 < size_idx_h2; t3_h2++)
    for (size_t t3_h1 = 0; t3_h1 < size_idx_h1; t3_h1++)
    for (size_t t3_p6 = 0; t3_p6 < size_idx_p6; t3_p6++)
    for (size_t t3_p5 = 0; t3_p5 < size_idx_p5; t3_p5++)
    for (size_t t3_p4 = 0; t3_p4 < size_idx_p4; t3_p4++)
    {
        size_t t3_idx = t3_h3 + (t3_h2 + (t3_h1 + (t3_p6 + (t3_p5 + (t3_p4) * size_idx_p5) * size_idx_p6) * size_idx_h1) * size_idx_h2) * size_idx_h3;
    
        // s1_5:   t3[h3,h2,h1,p6,p5,p4] += t1[p5,h2] * v2[h3,h1,p6,p4]
        
        {
            triplesx[t3_idx] += 	t1sub[t3_p5 + (t3_h2) * size_idx_p5] * 
                                        v2sub[t3_h3 + (t3_h1 + (t3_p6 + (t3_p4) * size_idx_p6) * size_idx_h1) * size_idx_h3];
        }
    }
}

void sd_t_s1_6_cpu(size_t size_idx_h1, size_t size_idx_h2, size_t size_idx_h3, size_t size_idx_p4, size_t size_idx_p5, size_t size_idx_p6, double *triplesx, double *t1sub, double *v2sub)
{
    #pragma omp parallel for collapse(6) 
    for (size_t t3_h3 = 0; t3_h3 < size_idx_h3; t3_h3++)
    for (size_t t3_h2 = 0; t3_h2 < size_idx_h2; t3_h2++)
    for (size_t t3_h1 = 0; t3_h1 < size_idx_h1; t3_h1++)
    for (size_t t3_p6 = 0; t3_p6 < size_idx_p6; t3_p6++)
    for (size_t t3_p5 = 0; t3_p5 < size_idx_p5; t3_p5++)
    for (size_t t3_p4 = 0; t3_p4 < size_idx_p4; t3_p4++)
    {
        size_t t3_idx = t3_h3 + (t3_h2 + (t3_h1 + (t3_p6 + (t3_p5 + (t3_p4) * size_idx_p5) * size_idx_p6) * size_idx_h1) * size_idx_h2) * size_idx_h3;
    
        // s1_6:   t3[h3,h2,h1,p6,p5,p4] -= t1[p5,h3] * v2[h2,h1,p6,p4]
        
        {
            triplesx[t3_idx] -= 	t1sub[t3_p5 + (t3_h3) * size_idx_p5] * 
                                        v2sub[t3_h2 + (t3_h1 + (t3_p6 + (t3_p4) * size_idx_p6) * size_idx_h1) * size_idx_h2];
        }
    }
}

void sd_t_s1_7_cpu(size_t size_idx_h1, size_t size_idx_h2, size_t size_idx_h3, size_t size_idx_p4, size_t size_idx_p5, size_t size_idx_p6, double *triplesx, double *t1sub, double *v2sub)
{
    #pragma omp parallel for collapse(6) 
    for (size_t t3_h3 = 0; t3_h3 < size_idx_h3; t3_h3++)
    for (size_t t3_h2 = 0; t3_h2 < size_idx_h2; t3_h2++)
    for (size_t t3_h1 = 0; t3_h1 < size_idx_h1; t3_h1++)
    for (size_t t3_p6 = 0; t3_p6 < size_idx_p6; t3_p6++)
    for (size_t t3_p5 = 0; t3_p5 < size_idx_p5; t3_p5++)
    for (size_t t3_p4 = 0; t3_p4 < size_idx_p4; t3_p4++)
    {
        size_t t3_idx = t3_h3 + (t3_h2 + (t3_h1 + (t3_p6 + (t3_p5 + (t3_p4) * size_idx_p5) * size_idx_p6) * size_idx_h1) * size_idx_h2) * size_idx_h3;
    
        // s1_7:   t3[h3,h2,h1,p6,p5,p4] -= t1[p6,h1] * v2[h3,h2,p5,p4] *****
        
        {   
            triplesx[t3_idx] += 	t1sub[t3_p6 + (t3_h1) * size_idx_p6] * 
                                        v2sub[t3_h3 + (t3_h2 + (t3_p5 + (t3_p4) * size_idx_p5) * size_idx_h2) * size_idx_h3];
        }
    }
}

void sd_t_s1_8_cpu(size_t size_idx_h1, size_t size_idx_h2, size_t size_idx_h3, size_t size_idx_p4, size_t size_idx_p5, size_t size_idx_p6, double *triplesx, double *t1sub, double *v2sub)
{
    #pragma omp parallel for collapse(6) 
    for (size_t t3_h3 = 0; t3_h3 < size_idx_h3; t3_h3++)
    for (size_t t3_h2 = 0; t3_h2 < size_idx_h2; t3_h2++)
    for (size_t t3_h1 = 0; t3_h1 < size_idx_h1; t3_h1++)
    for (size_t t3_p6 = 0; t3_p6 < size_idx_p6; t3_p6++)
    for (size_t t3_p5 = 0; t3_p5 < size_idx_p5; t3_p5++)
    for (size_t t3_p4 = 0; t3_p4 < size_idx_p4; t3_p4++)
    {
        size_t t3_idx = t3_h3 + (t3_h2 + (t3_h1 + (t3_p6 + (t3_p5 + (t3_p4) * size_idx_p5) * size_idx_p6) * size_idx_h1) * size_idx_h2) * size_idx_h3;
    
        // s1_8:   t3[h3,h2,h1,p6,p5,p4] -= t1[p6,h2] * v2[h3,h1,p5,p4]
        {
            triplesx[t3_idx] -= 	t1sub[t3_p6 + (t3_h2) * size_idx_p6] * 
                                        v2sub[t3_h3 + (t3_h1 + (t3_p5 + (t3_p4) * size_idx_p5) * size_idx_h1) * size_idx_h3];
        }
    }
}

void sd_t_s1_9_cpu(size_t size_idx_h1, size_t size_idx_h2, size_t size_idx_h3, size_t size_idx_p4, size_t size_idx_p5, size_t size_idx_p6, double *triplesx, double *t1sub, double *v2sub)
{
    #pragma omp parallel for collapse(6) 
    for (size_t t3_h3 = 0; t3_h3 < size_idx_h3; t3_h3++)
    for (size_t t3_h2 = 0; t3_h2 < size_idx_h2; t3_h2++)
    for (size_t t3_h1 = 0; t3_h1 < size_idx_h1; t3_h1++)
    for (size_t t3_p6 = 0; t3_p6 < size_idx_p6; t3_p6++)
    for (size_t t3_p5 = 0; t3_p5 < size_idx_p5; t3_p5++)
    for (size_t t3_p4 = 0; t3_p4 < size_idx_p4; t3_p4++)
    {
        size_t t3_idx = t3_h3 + (t3_h2 + (t3_h1 + (t3_p6 + (t3_p5 + (t3_p4) * size_idx_p5) * size_idx_p6) * size_idx_h1) * size_idx_h2) * size_idx_h3;
    
        // s1_9:   t3[h3,h2,h1,p6,p5,p4] -= t1[p6,h3] * v2[h2,h1,p5,p4]
        // *t3[h1,h3,h2,p4,p6,p5] -= t1[p4,h1] * v2[h3,h2,p6,p5] << 9
        {
            triplesx[t3_idx] += 	t1sub[t3_p6 + (t3_h3) * size_idx_p6] * 
                                        v2sub[t3_h2 + (t3_h1 + (t3_p5 + (t3_p4) * size_idx_p5) * size_idx_h1) * size_idx_h2];
        }
    }
}
