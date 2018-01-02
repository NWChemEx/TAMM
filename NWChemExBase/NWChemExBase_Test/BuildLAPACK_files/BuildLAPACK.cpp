#include "BuildLAPACK/BuildLAPACK.hpp"
#include <build/stage/usr/local/include/FCMangleLAPACK.h>

extern "C" {

float FCLAPACK_GLOBAL(dgeqrf, DGEQRF)(int,int,double*,int,double*);

}

void BuildLAPACK::run_test()
{
    int m=3;
    int n=3;
    int lda=3;
    int info;
    double a[9]={12.0, -51.0, 4.0,
                 6.0, 167.0, -68.0,
                 -4.0, 24, -41.0};
    double tau[3]={0.0,0.0,0.0};
    info = FCLAPACK_GLOBAL(dgeqrf, DGEQRF)( m, n, a, lda, tau );
}
