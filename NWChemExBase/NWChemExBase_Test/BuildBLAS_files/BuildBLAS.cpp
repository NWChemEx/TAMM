#include "BuildBLAS/BuildBLAS.hpp"
#include <build/stage/usr/local/include/FCMangleBLAS.h>

extern "C" {

float FCBLAS_GLOBAL(sdsdot, SDSDOT)(int,float,float*,int,float*,float);

}

void BuildBLAS::run_test()
{
    int N=10;
    float alpha=3.4;
    float X[10]={1.1,2.2,3.3,4.4,5.5,6.6,7.7,8.8,9.9,10.10};
    float Y[10]={1.1,2.2,3.3,4.4,5.5,6.6,7.7,8.8,9.9,10.10};
    int incX=1;
    int incY=1;
    float result = FCBLAS_GLOBAL(sdsdot, SDSDOT)(N, alpha, X, incX, Y, incY);
}
