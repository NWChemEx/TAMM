#include "utilities.hpp"
#include "sisubit.hpp"

using Eigen::MatrixXd;
//using Eigen::VectorXd;
//using Eigen::LDLT;
//using Eigen::LLT;

#ifdef NO_MPI
MatrixXd sisubit(const MatrixXd &HS, const MatrixXd &S, MatrixXd &X0, int *ipiv, int maxiter)
#else
MatrixXd sisubit(MPI_Comm comm, const MatrixXd &HS, const MatrixXd &S, MatrixXd &X0, int *ipiv, int maxiter)
#endif
{
   const double *hsdata = HS.data();
   char lower = 'L';
   int ncols = X0.cols();
   int n = X0.rows();
   int ierr = 0;

   MatrixXd X(n,ncols); 

   double t1 = omp_get_wtime();
   for (int iter=0; iter<maxiter; iter++) {
      // std::cout << "iter = " << iter << std::endl;
      X = S*X0;
      // dsytrs_(&lower, &n, &ncols, hsdata, &n, ipiv, X.data(), &n, &ierr);  
      ierr = LAPACKE_dsytrs(LAPACK_COL_MAJOR, lower, n, ncols, hsdata, n, 
                            ipiv, X.data(), n);  

      X0 = cholQR(S,X);
   }
   double t2 = omp_get_wtime();
   logOFS << "solve time = " << t2-t1 << std::endl;
   return X0;  
}
//
//  LDLT<MatrixXd> ldlt(HS);
//
//  ldlt.compute(HS);
//  L = ldlt.matrixL();
//  D = ldlt.vectorD();
//  std::cout << "LDL factorization completed" << std::endl;
//
//  int maxiter = params.maxiter;
//
// for (int iter=0; iter<maxiter; iter++) {
//   std::cout << "iter = " << iter << std::endl;
//   X  = S*X0;
//   X0 = ldlt.solve(X);
//   X  = cholQR(S,X0);
//   X0 = X;
// }

