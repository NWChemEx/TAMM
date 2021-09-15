#ifdef _OPENMP
#include <omp.h>
#endif

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

void sd_t_d2_1_cpu(size_t size_idx_h1, size_t size_idx_h2, size_t size_idx_h3, size_t size_idx_p4, size_t size_idx_p5, size_t size_idx_p6, size_t size_idx_p7, double *triplesx, double *t2sub, double *v2sub) {
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
