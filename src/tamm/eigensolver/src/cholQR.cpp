#include "utilities.hpp"

MatrixXd cholQR(const MatrixXd &S, const MatrixXd &X0)
{
   int n     = X0.rows();
   int ncols = X0.cols();
   MatrixXd G(ncols,ncols), X(n,ncols);
   G = X0.transpose()*(S*X0);

   char   upper = 'U', right = 'R', notrans = 'N', nunit = 'N';
   double done = 1.0;
   int    ierr = 0;
   // dpotrf_(&upper, &ncols, G.data(), &ncols, &ierr);   
   ierr = LAPACKE_dpotrf(LAPACK_COL_MAJOR,upper, ncols, G.data(), ncols);
   X = X0;
   // dtrsm_(&right, &upper, &notrans, &nunit, &n, &ncols, &done, G.data(), &ncols, X.data(), &n);  
   cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
               n, ncols, done, G.data(), ncols, X.data(), n);  

//   LLT<MatrixXd> cholfac(G);
//   MatrixXd R = cholfac.matrixU();

   return X;
}

