#include "BuildLAPACKE/BuildLAPACKE.hpp"
#include LAPACKE_HEADER

void BuildLAPACKE::run_test()
{
    lapack_int m=3;
    lapack_int n=3;
    lapack_int lda=3;
    lapack_int info;
    double a[9]={12.0, -51.0, 4.0,
                 6.0, 167.0, -68.0,
                 -4.0, 24, -41.0};
    double tau[3]={0.0,0.0,0.0};
    info = LAPACKE_dgeqrf( LAPACK_COL_MAJOR, m, n, a, lda, tau );
}
