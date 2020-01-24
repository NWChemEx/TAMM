#include "ccsd_t_common.hpp"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

//
void total_fused_ccsd_t_cpu(size_t base_size_h1b, size_t base_size_h2b, size_t base_size_h3b, 
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
                            std::vector<size_t> vec_d1_flags,
                            std::vector<size_t> vec_d2_flags,
                            std::vector<size_t> vec_s1_flags, 
                            // 
                            size_t size_noab, size_t size_max_dim_d1_t2, size_t size_max_dim_d1_v2,
                            size_t size_nvab, size_t size_max_dim_d2_t2, size_t size_max_dim_d2_v2,
                                              size_t size_max_dim_s1_t2, size_t size_max_dim_s1_v2, 
                            // 
                            double factor, 
                            double* host_evl_sorted_h1, double* host_evl_sorted_h2, double* host_evl_sorted_h3, 
                            double* host_evl_sorted_p4, double* host_evl_sorted_p5, double* host_evl_sorted_p6,
                            double* final_energy_4, double* final_energy_5)
{
    // 
    size_t size_tensor_t3 = base_size_h3b * base_size_h2b * base_size_h1b * base_size_p6b * base_size_p5b * base_size_p4b;

    // 
    double* host_t3_d = (double*)malloc(sizeof(double) * size_tensor_t3);
    double* host_t3_s = (double*)malloc(sizeof(double) * size_tensor_t3);

    for (size_t i = 0; i < size_tensor_t3; i++)
    {
        host_t3_d[i] = 0.000;
        host_t3_s[i] = 0.000;
    }

    // 
    for (size_t idx_ia6 = 0; idx_ia6 < 9; idx_ia6++)
    {
        // d1
        for (size_t idx_noab = 0; idx_noab < size_noab; idx_noab++)
        {
            int flag_d1_1 = (int)vec_d1_flags[0 + (idx_noab + (idx_ia6) * size_noab) * 9];
            int flag_d1_2 = (int)vec_d1_flags[1 + (idx_noab + (idx_ia6) * size_noab) * 9];
            int flag_d1_3 = (int)vec_d1_flags[2 + (idx_noab + (idx_ia6) * size_noab) * 9];
            int flag_d1_4 = (int)vec_d1_flags[3 + (idx_noab + (idx_ia6) * size_noab) * 9];
            int flag_d1_5 = (int)vec_d1_flags[4 + (idx_noab + (idx_ia6) * size_noab) * 9];
            int flag_d1_6 = (int)vec_d1_flags[5 + (idx_noab + (idx_ia6) * size_noab) * 9];
            int flag_d1_7 = (int)vec_d1_flags[6 + (idx_noab + (idx_ia6) * size_noab) * 9];
            int flag_d1_8 = (int)vec_d1_flags[7 + (idx_noab + (idx_ia6) * size_noab) * 9];
            int flag_d1_9 = (int)vec_d1_flags[8 + (idx_noab + (idx_ia6) * size_noab) * 9];

            int d1_base_size_h1b = (int)list_d1_sizes[0 + (idx_noab + (idx_ia6) * size_noab) * 7];
            int d1_base_size_h2b = (int)list_d1_sizes[1 + (idx_noab + (idx_ia6) * size_noab) * 7];
            int d1_base_size_h3b = (int)list_d1_sizes[2 + (idx_noab + (idx_ia6) * size_noab) * 7];
            int d1_base_size_h7b = (int)list_d1_sizes[3 + (idx_noab + (idx_ia6) * size_noab) * 7];
            int d1_base_size_p4b = (int)list_d1_sizes[4 + (idx_noab + (idx_ia6) * size_noab) * 7];
            int d1_base_size_p5b = (int)list_d1_sizes[5 + (idx_noab + (idx_ia6) * size_noab) * 7];
            int d1_base_size_p6b = (int)list_d1_sizes[6 + (idx_noab + (idx_ia6) * size_noab) * 7];

            double* host_d1_t2_1 = host_d1_t2_all + size_max_dim_d1_t2 * flag_d1_1;
            double* host_d1_v2_1 = host_d1_v2_all + size_max_dim_d1_v2 * flag_d1_1;
            double* host_d1_t2_2 = host_d1_t2_all + size_max_dim_d1_t2 * flag_d1_2;
            double* host_d1_v2_2 = host_d1_v2_all + size_max_dim_d1_v2 * flag_d1_2;
            double* host_d1_t2_3 = host_d1_t2_all + size_max_dim_d1_t2 * flag_d1_3;
            double* host_d1_v2_3 = host_d1_v2_all + size_max_dim_d1_v2 * flag_d1_3;
            double* host_d1_t2_4 = host_d1_t2_all + size_max_dim_d1_t2 * flag_d1_4;
            double* host_d1_v2_4 = host_d1_v2_all + size_max_dim_d1_v2 * flag_d1_4;
            double* host_d1_t2_5 = host_d1_t2_all + size_max_dim_d1_t2 * flag_d1_5;
            double* host_d1_v2_5 = host_d1_v2_all + size_max_dim_d1_v2 * flag_d1_5;
            double* host_d1_t2_6 = host_d1_t2_all + size_max_dim_d1_t2 * flag_d1_6;
            double* host_d1_v2_6 = host_d1_v2_all + size_max_dim_d1_v2 * flag_d1_6;
            double* host_d1_t2_7 = host_d1_t2_all + size_max_dim_d1_t2 * flag_d1_7;
            double* host_d1_v2_7 = host_d1_v2_all + size_max_dim_d1_v2 * flag_d1_7;
            double* host_d1_t2_8 = host_d1_t2_all + size_max_dim_d1_t2 * flag_d1_8;
            double* host_d1_v2_8 = host_d1_v2_all + size_max_dim_d1_v2 * flag_d1_8;
            double* host_d1_t2_9 = host_d1_t2_all + size_max_dim_d1_t2 * flag_d1_9;
            double* host_d1_v2_9 = host_d1_v2_all + size_max_dim_d1_v2 * flag_d1_9;

            #pragma omp parallel for collapse(6) 
            for (int t3_h3 = 0; t3_h3 < d1_base_size_h3b; t3_h3++)
            for (int t3_h2 = 0; t3_h2 < d1_base_size_h2b; t3_h2++)
            for (int t3_h1 = 0; t3_h1 < d1_base_size_h1b; t3_h1++)
            for (int t3_p6 = 0; t3_p6 < d1_base_size_p6b; t3_p6++)
            for (int t3_p5 = 0; t3_p5 < d1_base_size_p5b; t3_p5++)
            for (int t3_p4 = 0; t3_p4 < d1_base_size_p4b; t3_p4++)
            {
                int t3_idx = t3_h3 + (t3_h2 + (t3_h1 + (t3_p6 + (t3_p5 + (t3_p4) * d1_base_size_p5b) * d1_base_size_p6b) * d1_base_size_h1b) * d1_base_size_h2b) * d1_base_size_h3b;
                
                for (int t3_h7 = 0; t3_h7 < d1_base_size_h7b; t3_h7++)
                {   
                    // sd1_1:  t3[h3,h2,h1,p6,p5,p4] -= t2[h7,p4,p5,h1] * v2[h3,h2,p6,h7]
                    if (flag_d1_1 >= 0)
                    {
                        host_t3_d[t3_idx] -= host_d1_t2_1[t3_h7 + (t3_p4 + (t3_p5 + (t3_h1) * d1_base_size_p5b) * d1_base_size_p4b) * d1_base_size_h7b] * 
                                             host_d1_v2_1[t3_h3 + (t3_h2 + (t3_p6 + (t3_h7) * d1_base_size_p6b) * d1_base_size_h2b) * d1_base_size_h3b];                            
                    }
                
                    // sd1_2:  t3[h3,h2,h1,p6,p5,p4] += t2[h7,p4,p5,h2] * v2[h3,h1,p6,h7]
                    if (flag_d1_2 >= 0)
                    {
                        host_t3_d[t3_idx] += host_d1_t2_2[t3_h7 + (t3_p4 + (t3_p5 + (t3_h2) * d1_base_size_p5b) * d1_base_size_p4b) * d1_base_size_h7b] * 
                                             host_d1_v2_2[t3_h3 + (t3_h1 + (t3_p6 + (t3_h7) * d1_base_size_p6b) * d1_base_size_h1b) * d1_base_size_h3b];
                    }

                    // sd1_3:  t3[h3,h2,h1,p6,p5,p4] -= t2[h7,p4,p5,h3] * v2[h2,h1,p6,h7]
                    if (flag_d1_3 >= 0)
                    {
                        host_t3_d[t3_idx] -= host_d1_t2_3[t3_h7 + (t3_p4 + (t3_p5 + (t3_h3) * d1_base_size_p5b) * d1_base_size_p4b) * d1_base_size_h7b] * 
                                             host_d1_v2_3[t3_h2 + (t3_h1 + (t3_p6 + (t3_h7) * d1_base_size_p6b) * d1_base_size_h1b) * d1_base_size_h2b];
                    }

                    // sd1_4:  t3[h3,h2,h1,p6,p5,p4] -= t2[h7,p5,p6,h1] * v2[h3,h2,p4,h7]
                    if (flag_d1_4 >= 0)
                    {
                        host_t3_d[t3_idx] -= host_d1_t2_4[t3_h7 + (t3_p5 + (t3_p6 + (t3_h1) * d1_base_size_p6b) * d1_base_size_p5b) * d1_base_size_h7b] * 
                                             host_d1_v2_4[t3_h3 + (t3_h2 + (t3_p4 + (t3_h7) * d1_base_size_p4b) * d1_base_size_h2b) * d1_base_size_h3b];
                    }

                    // sd1_5:  t3[h3,h2,h1,p6,p5,p4] += t2[h7,p5,p6,h2] * v2[h3,h1,p4,h7]
                    if (flag_d1_5 >= 0)
                    {
                        host_t3_d[t3_idx] += host_d1_t2_5[t3_h7 + (t3_p5 + (t3_p6 + (t3_h2) * d1_base_size_p6b) * d1_base_size_p5b) * d1_base_size_h7b] * 
                                             host_d1_v2_5[t3_h3 + (t3_h1 + (t3_p4 + (t3_h7) * d1_base_size_p4b) * d1_base_size_h1b) * d1_base_size_h3b];
                    }

                    // sd1_6:  t3[h3,h2,h1,p6,p5,p4] -= t2[h7,p5,p6,h3] * v2[h2,h1,p4,h7]
                    if (flag_d1_6 >= 0)
                    {
                        host_t3_d[t3_idx] -= host_d1_t2_6[t3_h7 + (t3_p5 + (t3_p6 + (t3_h3) * d1_base_size_p6b) * d1_base_size_p5b) * d1_base_size_h7b] * 
                                             host_d1_v2_6[t3_h2 + (t3_h1 + (t3_p4 + (t3_h7) * d1_base_size_p4b) * d1_base_size_h1b) * d1_base_size_h2b];
                    }

                    // sd1_7:  t3[h3,h2,h1,p6,p5,p4] += t2[h7,p4,p6,h1] * v2[h3,h2,p5,h7] 
                    if (flag_d1_7 >= 0)
                    {
                        host_t3_d[t3_idx] += host_d1_t2_7[t3_h7 + (t3_p4 + (t3_p6 + (t3_h1) * d1_base_size_p6b) * d1_base_size_p4b) * d1_base_size_h7b] * 
                                             host_d1_v2_7[t3_h3 + (t3_h2 + (t3_p5 + (t3_h7) * d1_base_size_p5b) * d1_base_size_h2b) * d1_base_size_h3b];
                    }

                    // sd1_8:  t3[h3,h2,h1,p6,p5,p4] -= t2[h7,p4,p6,h2] * v2[h3,h1,p5,h7]
                    if (flag_d1_8 >= 0)
                    {
                        host_t3_d[t3_idx] -= host_d1_t2_8[t3_h7 + (t3_p4 + (t3_p6 + (t3_h2) * d1_base_size_p6b) * d1_base_size_p4b) * d1_base_size_h7b] * 
                                             host_d1_v2_8[t3_h3 + (t3_h1 + (t3_p5 + (t3_h7) * d1_base_size_p5b) * d1_base_size_h1b) * d1_base_size_h3b];
                    }

                    // sd1_9:  t3[h3,h2,h1,p6,p5,p4] += t2[h7,p4,p6,h3] * v2[h2,h1,p5,h7]
                    if (flag_d1_9 >= 0)
                    {
                        host_t3_d[t3_idx] += host_d1_t2_9[t3_h7 + (t3_p4 + (t3_p6 + (t3_h3) * d1_base_size_p6b) * d1_base_size_p4b) * d1_base_size_h7b] * 
                                             host_d1_v2_9[t3_h2 + (t3_h1 + (t3_p5 + (t3_h7) * d1_base_size_p5b) * d1_base_size_h1b) * d1_base_size_h2b];
                    }   
                }
            }
        }

        // d2
        for (size_t idx_nvab = 0; idx_nvab < size_nvab; idx_nvab++)
        {
            int flag_d2_1 = (int)vec_d2_flags[0 + (idx_nvab + (idx_ia6) * size_nvab) * 9];
            int flag_d2_2 = (int)vec_d2_flags[1 + (idx_nvab + (idx_ia6) * size_nvab) * 9];
            int flag_d2_3 = (int)vec_d2_flags[2 + (idx_nvab + (idx_ia6) * size_nvab) * 9];
            int flag_d2_4 = (int)vec_d2_flags[3 + (idx_nvab + (idx_ia6) * size_nvab) * 9];
            int flag_d2_5 = (int)vec_d2_flags[4 + (idx_nvab + (idx_ia6) * size_nvab) * 9];
            int flag_d2_6 = (int)vec_d2_flags[5 + (idx_nvab + (idx_ia6) * size_nvab) * 9];
            int flag_d2_7 = (int)vec_d2_flags[6 + (idx_nvab + (idx_ia6) * size_nvab) * 9];
            int flag_d2_8 = (int)vec_d2_flags[7 + (idx_nvab + (idx_ia6) * size_nvab) * 9];
            int flag_d2_9 = (int)vec_d2_flags[8 + (idx_nvab + (idx_ia6) * size_nvab) * 9];

            int d2_base_size_h1b = (int)list_d2_sizes[0 + (idx_nvab + (idx_ia6) * size_nvab) * 7];
            int d2_base_size_h2b = (int)list_d2_sizes[1 + (idx_nvab + (idx_ia6) * size_nvab) * 7];
            int d2_base_size_h3b = (int)list_d2_sizes[2 + (idx_nvab + (idx_ia6) * size_nvab) * 7];
            int d2_base_size_p4b = (int)list_d2_sizes[3 + (idx_nvab + (idx_ia6) * size_nvab) * 7];
            int d2_base_size_p5b = (int)list_d2_sizes[4 + (idx_nvab + (idx_ia6) * size_nvab) * 7];
            int d2_base_size_p6b = (int)list_d2_sizes[5 + (idx_nvab + (idx_ia6) * size_nvab) * 7];
            int d2_base_size_p7b = (int)list_d2_sizes[6 + (idx_nvab + (idx_ia6) * size_nvab) * 7];

            double* host_d2_t2_1 = host_d2_t2_all + size_max_dim_d2_t2 * flag_d2_1;
            double* host_d2_v2_1 = host_d2_v2_all + size_max_dim_d2_v2 * flag_d2_1;
            double* host_d2_t2_2 = host_d2_t2_all + size_max_dim_d2_t2 * flag_d2_2;
            double* host_d2_v2_2 = host_d2_v2_all + size_max_dim_d2_v2 * flag_d2_2;
            double* host_d2_t2_3 = host_d2_t2_all + size_max_dim_d2_t2 * flag_d2_3;
            double* host_d2_v2_3 = host_d2_v2_all + size_max_dim_d2_v2 * flag_d2_3;
            double* host_d2_t2_4 = host_d2_t2_all + size_max_dim_d2_t2 * flag_d2_4;
            double* host_d2_v2_4 = host_d2_v2_all + size_max_dim_d2_v2 * flag_d2_4;
            double* host_d2_t2_5 = host_d2_t2_all + size_max_dim_d2_t2 * flag_d2_5;
            double* host_d2_v2_5 = host_d2_v2_all + size_max_dim_d2_v2 * flag_d2_5;
            double* host_d2_t2_6 = host_d2_t2_all + size_max_dim_d2_t2 * flag_d2_6;
            double* host_d2_v2_6 = host_d2_v2_all + size_max_dim_d2_v2 * flag_d2_6;
            double* host_d2_t2_7 = host_d2_t2_all + size_max_dim_d2_t2 * flag_d2_7;
            double* host_d2_v2_7 = host_d2_v2_all + size_max_dim_d2_v2 * flag_d2_7;
            double* host_d2_t2_8 = host_d2_t2_all + size_max_dim_d2_t2 * flag_d2_8;
            double* host_d2_v2_8 = host_d2_v2_all + size_max_dim_d2_v2 * flag_d2_8;
            double* host_d2_t2_9 = host_d2_t2_all + size_max_dim_d2_t2 * flag_d2_9;
            double* host_d2_v2_9 = host_d2_v2_all + size_max_dim_d2_v2 * flag_d2_9;

            #pragma omp parallel for collapse(6)
            for (int t3_h3 = 0; t3_h3 < d2_base_size_h3b; t3_h3++)
            for (int t3_h2 = 0; t3_h2 < d2_base_size_h2b; t3_h2++)
            for (int t3_h1 = 0; t3_h1 < d2_base_size_h1b; t3_h1++)
            for (int t3_p6 = 0; t3_p6 < d2_base_size_p6b; t3_p6++)
            for (int t3_p5 = 0; t3_p5 < d2_base_size_p5b; t3_p5++)
            for (int t3_p4 = 0; t3_p4 < d2_base_size_p4b; t3_p4++)
            {
                int t3_idx = t3_h3 + (t3_h2 + (t3_h1 + (t3_p6 + (t3_p5 + (t3_p4) * d2_base_size_p5b) * d2_base_size_p6b) * d2_base_size_h1b) * d2_base_size_h2b) * d2_base_size_h3b;

                for (int t3_p7 = 0; t3_p7 < d2_base_size_p7b; t3_p7++)
                {
                    // sd2_1:  t3[h3,h2,h1,p6,p5,p4] −= t2[p7,p4,h1,h2] * v2[p7,h3,p6,p5]	
                    if (flag_d2_1 >= 0)
                    {
                        host_t3_d[t3_idx] -= host_d2_t2_1[t3_p7 + (t3_p4 + (t3_h1 + (t3_h2) * d2_base_size_h1b) * d2_base_size_p4b) * d2_base_size_p7b] * 
                                             host_d2_v2_1[t3_p7 + (t3_h3 + (t3_p6 + (t3_p5) * d2_base_size_p6b) * d2_base_size_h3b) * d2_base_size_p7b];
                    }

                    // sd2_2:  t3[h3,h2,h1,p6,p5,p4] −= t2[p7,p4,h2,h3] * v2[p7,h1,p6,p5] 
                    if (flag_d2_2 >= 0)
                    {
                        host_t3_d[t3_idx] -= host_d2_t2_2[t3_p7 + (t3_p4 + (t3_h2 + (t3_h3) * d2_base_size_h2b) * d2_base_size_p4b) * d2_base_size_p7b] * 
                                             host_d2_v2_2[t3_p7 + (t3_h1 + (t3_p6 + (t3_p5) * d2_base_size_p6b) * d2_base_size_h1b) * d2_base_size_p7b];
                    }
                
                    // sd2_3:  t3[h3,h2,h1,p6,p5,p4] += t2[p7,p4,h1,h3] * v2[p7,h2,p6,p5] 
                    if (flag_d2_3 >= 0)
                    {   
                        host_t3_d[t3_idx] += host_d2_t2_3[t3_p7 + (t3_p4 + (t3_h1 + (t3_h3) * d2_base_size_h1b) * d2_base_size_p4b) * d2_base_size_p7b] * 
                                             host_d2_v2_3[t3_p7 + (t3_h2 + (t3_p6 + (t3_p5) * d2_base_size_p6b) * d2_base_size_h2b) * d2_base_size_p7b];
                    }
                
                    // sd2_4:  t3[h3,h2,h1,p6,p5,p4] += t2[p7,p5,h1,h2] * v2[p7,h3,p6,p4]                         
                    if (flag_d2_4 >= 0)
                    {
                        host_t3_d[t3_idx] += host_d2_t2_4[t3_p7 + (t3_p5 + (t3_h1 + (t3_h2) * d2_base_size_h1b) * d2_base_size_p5b) * d2_base_size_p7b] * 
                                             host_d2_v2_4[t3_p7 + (t3_h3 + (t3_p6 + (t3_p4) * d2_base_size_p6b) * d2_base_size_h3b) * d2_base_size_p7b];
                    }
                
                    // sd2_5:  t3[h3,h2,h1,p6,p5,p4] += t2[p7,p5,h2,h3] * v2[p7,h1,p6,p4]     
                    if (flag_d2_5 >= 0)
                    {
                        host_t3_d[t3_idx] += host_d2_t2_5[t3_p7 + (t3_p5 + (t3_h2 + (t3_h3) * d2_base_size_h2b) * d2_base_size_p5b) * d2_base_size_p7b] * 
                                             host_d2_v2_5[t3_p7 + (t3_h1 + (t3_p6 + (t3_p4) * d2_base_size_p6b) * d2_base_size_h1b) * d2_base_size_p7b];
                    }
                
                    // sd2_6:  t3[h3,h2,h1,p6,p5,p4] −= t2[p7,p5,h1,h3] * v2[p7,h2,p6,p4] 
                    if (flag_d2_6 >= 0)
                    {
                        host_t3_d[t3_idx] -= host_d2_t2_6[t3_p7 + (t3_p5 + (t3_h1 + (t3_h3) * d2_base_size_h1b) * d2_base_size_p5b) * d2_base_size_p7b] * 
                                             host_d2_v2_6[t3_p7 + (t3_h2 + (t3_p6 + (t3_p4) * d2_base_size_p6b) * d2_base_size_h2b) * d2_base_size_p7b];
                    }
                
                    // sd2_7:  t3[h3,h2,h1,p6,p5,p4] −= t2[p7,p6,h1,h2] * v2[p7,h3,p5,p4]
                    if (flag_d2_7 >= 0) 
                    {   
                        host_t3_d[t3_idx] -= host_d2_t2_7[t3_p7 + (t3_p6 + (t3_h1 + (t3_h2) * d2_base_size_h1b) * d2_base_size_p6b) * d2_base_size_p7b] * 
                                             host_d2_v2_7[t3_p7 + (t3_h3 + (t3_p5 + (t3_p4) * d2_base_size_p5b) * d2_base_size_h3b) * d2_base_size_p7b];
                    }
                
                    // sd2_8:  t3[h3,h2,h1,p6,p5,p4] −= t2[p7,p6,h2,h3] * v2[p7,h1,p5,p4]
                    if (flag_d2_8 >= 0)
                    {
                        host_t3_d[t3_idx] -= host_d2_t2_8[t3_p7 + (t3_p6 + (t3_h2 + (t3_h3) * d2_base_size_h2b) * d2_base_size_p6b) * d2_base_size_p7b] * 
                                             host_d2_v2_8[t3_p7 + (t3_h1 + (t3_p5 + (t3_p4) * d2_base_size_p5b) * d2_base_size_h1b) * d2_base_size_p7b];
                    }
                
                    // sd2_9:  t3[h3,h2,h1,p6,p5,p4] += t2[p7,p6,h1,h3] * v2[p7,h2,p5,p4]
                    if (flag_d2_9 >= 0)
                    {
                        host_t3_d[t3_idx] += host_d2_t2_9[t3_p7 + (t3_p6 + (t3_h1 + (t3_h3) * d2_base_size_h1b) * d2_base_size_p6b) * d2_base_size_p7b] * 
                                             host_d2_v2_9[t3_p7 + (t3_h2 + (t3_p5 + (t3_p4) * d2_base_size_p5b) * d2_base_size_h2b) * d2_base_size_p7b];
                    }
                }
            }
        }

        // s1
        {
            // 	flags
			int flag_s1_1 = (int)vec_s1_flags[0 + idx_ia6 * 9];
			int flag_s1_2 = (int)vec_s1_flags[1 + idx_ia6 * 9];
			int flag_s1_3 = (int)vec_s1_flags[2 + idx_ia6 * 9];
			int flag_s1_4 = (int)vec_s1_flags[3 + idx_ia6 * 9];
			int flag_s1_5 = (int)vec_s1_flags[4 + idx_ia6 * 9];
			int flag_s1_6 = (int)vec_s1_flags[5 + idx_ia6 * 9];
			int flag_s1_7 = (int)vec_s1_flags[6 + idx_ia6 * 9];
			int flag_s1_8 = (int)vec_s1_flags[7 + idx_ia6 * 9];
			int flag_s1_9 = (int)vec_s1_flags[8 + idx_ia6 * 9];

            int s1_base_size_h1b = (int)list_s1_sizes[0 + idx_ia6 * 6];
			int s1_base_size_h2b = (int)list_s1_sizes[1 + idx_ia6 * 6];
			int s1_base_size_h3b = (int)list_s1_sizes[2 + idx_ia6 * 6];
			int s1_base_size_p4b = (int)list_s1_sizes[3 + idx_ia6 * 6];
			int s1_base_size_p5b = (int)list_s1_sizes[4 + idx_ia6 * 6];
			int s1_base_size_p6b = (int)list_s1_sizes[5 + idx_ia6 * 6];

            double* host_s1_t2;
            double* host_s1_v2;

            // 
            #pragma omp parallel for collapse(6) 
            for (int t3_h3 = 0; t3_h3 < s1_base_size_h3b; t3_h3++)
            for (int t3_h2 = 0; t3_h2 < s1_base_size_h2b; t3_h2++)
            for (int t3_h1 = 0; t3_h1 < s1_base_size_h1b; t3_h1++)
            for (int t3_p6 = 0; t3_p6 < s1_base_size_p6b; t3_p6++)
            for (int t3_p5 = 0; t3_p5 < s1_base_size_p5b; t3_p5++)
            for (int t3_p4 = 0; t3_p4 < s1_base_size_p4b; t3_p4++)
            {
                int t3_idx = t3_h3 + (t3_h2 + (t3_h1 + (t3_p6 + (t3_p5 + (t3_p4) * s1_base_size_p5b) * s1_base_size_p6b) * s1_base_size_h1b) * s1_base_size_h2b) * s1_base_size_h3b;
            
                //  s1_1: t3[h3,h2,h1,p6,p5,p4] += t1[p4,h1] * v2[h3,h2,p6,p5]
                host_s1_t2 = host_s1_t2_all + size_max_dim_s1_t2 * flag_s1_1;
				host_s1_v2 = host_s1_v2_all + size_max_dim_s1_v2 * flag_s1_1;

                if (flag_s1_1 >= 0)
                {
                    host_t3_s[t3_idx] += host_s1_t2[t3_p4 + (t3_h1) * s1_base_size_p4b] * 
                                         host_s1_v2[t3_h3 + (t3_h2 + (t3_p6 + (t3_p5) * s1_base_size_p6b) * s1_base_size_h2b) * s1_base_size_h3b];

                }

                // s1_2: t3[h3,h2,h1,p6,p5,p4] -= t1[p4,h2] * v2[h3,h1,p6,p5]
                host_s1_t2 = host_s1_t2_all + size_max_dim_s1_t2 * flag_s1_2;
				host_s1_v2 = host_s1_v2_all + size_max_dim_s1_v2 * flag_s1_2;

                if (flag_s1_2 >= 0)
                {
                    host_t3_s[t3_idx] -= host_s1_t2[t3_p4 + (t3_h2) * s1_base_size_p4b] * 
                                         host_s1_v2[t3_h3 + (t3_h1 + (t3_p6 + (t3_p5) * s1_base_size_p6b) * s1_base_size_h1b) * s1_base_size_h3b];
                }

                // s1_3: t3[h3,h2,h1,p6,p5,p4] += t1[p4,h3] * v2[h2,h1,p6,p5]
                host_s1_t2 = host_s1_t2_all + size_max_dim_s1_t2 * flag_s1_3;
				host_s1_v2 = host_s1_v2_all + size_max_dim_s1_v2 * flag_s1_3;

                if (flag_s1_3 >= 0)
                {   
                    host_t3_s[t3_idx] += host_s1_t2[t3_p4 + (t3_h3) * s1_base_size_p4b] * 
                                         host_s1_v2[t3_h2 + (t3_h1 + (t3_p6 + (t3_p5) * s1_base_size_p6b) * s1_base_size_h1b) * s1_base_size_h2b];
                }

                // s1_4:   t3[h3,h2,h1,p6,p5,p4] -= t1[p5,h1] * v2[h3,h2,p6,p4] 
                host_s1_t2 = host_s1_t2_all + size_max_dim_s1_t2 * flag_s1_4;
				host_s1_v2 = host_s1_v2_all + size_max_dim_s1_v2 * flag_s1_4;

                if (flag_s1_4 >= 0)
                {
                    host_t3_s[t3_idx] -= host_s1_t2[t3_p5 + (t3_h1) * s1_base_size_p5b] * 
                                         host_s1_v2[t3_h3 + (t3_h2 + (t3_p6 + (t3_p4) * s1_base_size_p6b) * s1_base_size_h2b) * s1_base_size_h3b];
                }

                // s1_5:   t3[h3,h2,h1,p6,p5,p4] += t1[p5,h2] * v2[h3,h1,p6,p4]
                host_s1_t2 = host_s1_t2_all + size_max_dim_s1_t2 * flag_s1_5;
				host_s1_v2 = host_s1_v2_all + size_max_dim_s1_v2 * flag_s1_5;

                if (flag_s1_5 >= 0)
                {
                    host_t3_s[t3_idx] += host_s1_t2[t3_p5 + (t3_h2) * s1_base_size_p5b] * 
                                         host_s1_v2[t3_h3 + (t3_h1 + (t3_p6 + (t3_p4) * s1_base_size_p6b) * s1_base_size_h1b) * s1_base_size_h3b];
                }

                // s1_6:   t3[h3,h2,h1,p6,p5,p4] -= t1[p5,h3] * v2[h2,h1,p6,p4]
                host_s1_t2 = host_s1_t2_all + size_max_dim_s1_t2 * flag_s1_6;
				host_s1_v2 = host_s1_v2_all + size_max_dim_s1_v2 * flag_s1_6;

                if (flag_s1_6 >= 0)
                {
                    host_t3_s[t3_idx] -= host_s1_t2[t3_p5 + (t3_h3) * s1_base_size_p5b] * 
                                         host_s1_v2[t3_h2 + (t3_h1 + (t3_p6 + (t3_p4) * s1_base_size_p6b) * s1_base_size_h1b) * s1_base_size_h2b];
                }

                // s1_7:   t3[h3,h2,h1,p6,p5,p4] -= t1[p6,h1] * v2[h3,h2,p5,p4]
                host_s1_t2 = host_s1_t2_all + size_max_dim_s1_t2 * flag_s1_7;
				host_s1_v2 = host_s1_v2_all + size_max_dim_s1_v2 * flag_s1_7;

                if (flag_s1_7 >= 0)
                {   
                    host_t3_s[t3_idx] += host_s1_t2[t3_p6 + (t3_h1) * s1_base_size_p6b] * 
                                         host_s1_v2[t3_h3 + (t3_h2 + (t3_p5 + (t3_p4) * s1_base_size_p5b) * s1_base_size_h2b) * s1_base_size_h3b];
                }

                // s1_8:   t3[h3,h2,h1,p6,p5,p4] -= t1[p6,h2] * v2[h3,h1,p5,p4]
                host_s1_t2 = host_s1_t2_all + size_max_dim_s1_t2 * flag_s1_8;
				host_s1_v2 = host_s1_v2_all + size_max_dim_s1_v2 * flag_s1_8;

                if (flag_s1_8 >= 0)
                {
                    host_t3_s[t3_idx] -= host_s1_t2[t3_p6 + (t3_h2) * s1_base_size_p6b] * 
                                         host_s1_v2[t3_h3 + (t3_h1 + (t3_p5 + (t3_p4) * s1_base_size_p5b) * s1_base_size_h1b) * s1_base_size_h3b];
                }

                // s1_9:   t3[h3,h2,h1,p6,p5,p4] -= t1[p6,h3] * v2[h2,h1,p5,p4]
                host_s1_t2 = host_s1_t2_all + size_max_dim_s1_t2 * flag_s1_9;
				host_s1_v2 = host_s1_v2_all + size_max_dim_s1_v2 * flag_s1_9;

                if (flag_s1_9 >= 0)
                {
                    host_t3_s[t3_idx] += host_s1_t2[t3_p6 + (t3_h3) * s1_base_size_p6b] * 
                                         host_s1_v2[t3_h2 + (t3_h1 + (t3_p5 + (t3_p4) * s1_base_size_p5b) * s1_base_size_h1b) * s1_base_size_h2b];
                }
            }
        }
    }

    // 
    //  to calculate energies--- E(4) and E(5)
    // 
    double host_energy_4        = 0.0;
    double host_energy_5        = 0.0;

    int size_idx_h1 = (int)base_size_h1b;
    int size_idx_h2 = (int)base_size_h2b;
    int size_idx_h3 = (int)base_size_h3b;
    int size_idx_p4 = (int)base_size_p4b;
    int size_idx_p5 = (int)base_size_p5b;
    int size_idx_p6 = (int)base_size_p6b;

    //
    for (int idx_p4 = 0; idx_p4 < size_idx_p4; idx_p4++)
    for (int idx_p5 = 0; idx_p5 < size_idx_p5; idx_p5++)
    for (int idx_p6 = 0; idx_p6 < size_idx_p6; idx_p6++)
    for (int idx_h1 = 0; idx_h1 < size_idx_h1; idx_h1++)
    for (int idx_h2 = 0; idx_h2 < size_idx_h2; idx_h2++)
    for (int idx_h3 = 0; idx_h3 < size_idx_h3; idx_h3++)
    {
        // 
        int idx_t3 = idx_h3 + (idx_h2 + (idx_h1 + (idx_p6 + (idx_p5 + (idx_p4) * size_idx_p5) * size_idx_p6) * size_idx_h1) * size_idx_h2) * size_idx_h3;

        // 
        double inner_factor = (host_evl_sorted_h3[idx_h3] + host_evl_sorted_h2[idx_h2] + host_evl_sorted_h1[idx_h1] - 
                               host_evl_sorted_p6[idx_p6] - host_evl_sorted_p5[idx_p5] - host_evl_sorted_p4[idx_p4]);
        // 
        host_energy_4 += factor * host_t3_d[idx_t3] * (host_t3_d[idx_t3])                       / inner_factor;
        host_energy_5 += factor * host_t3_d[idx_t3] * (host_t3_d[idx_t3] + host_t3_s[idx_t3])   / inner_factor;
    }

    // 
    *final_energy_4 = host_energy_4;
    *final_energy_5 = host_energy_5;

    //printf ("E(4): %.14f, E(5): %.14f\n", host_energy_4, host_energy_5);
    // printf ("========================================================================================\n");
}