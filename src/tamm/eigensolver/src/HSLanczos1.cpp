#include "utilities.hpp"
// S-orthogonal Lanczos
VectorXd HSLanczos1(const MatrixXd &H, const MatrixXd &S, const VectorXd &v0, MatrixXd &T, MatrixXd &V)
{
   char lower = 'L';
   int  nrhs = 1;

   int n = H.rows();
   int niter = V.cols();
   VectorXd f(n), v(n), SiHv(n);
   MatrixXd R(n,n); // a copy of S to hold the Cholesky factor
 
   R = S;
  
   int ierr = 0;
   dpotrf_(&lower, &n, R.data(), &n, &ierr);   
   if (ierr) {
      std::cout << " dpotrf failure, ierr = " << ierr << std::endl;
      exit(1);
   }

   // normalize the starting vector
   std::cout << " S = " << S.block(0,0,4,4) << std::endl;
   std::cout << " H = " << H.block(0,0,4,4) << std::endl;
   double beta = sqrt(v0.dot(S*v0));
   v = v0/beta; 

   std::cout << "beta = " << beta << std::endl;

   for (int iter = 0; iter < niter; iter++) {
      // MATVEC with inv(R')
      V.block(0,iter,n,1) = v;
      SiHv = H*v;
      dpotrs(&lower, &n, &nrhs, R.data(), &n, SiHv.data(), &n, &ierr);  
      T.block(0,iter,iter+1,1) = V.block(0,0,n,iter+1).transpose() *(S*SiHv);
      f = SiHv - V.block(0,0,n,iter+1)*T.block(0,iter,iter+1,1);
      beta = sqrt(f.dot(S*f));
      if (iter < niter-1) T(iter+1,iter) = beta;
      v =  f/beta;
   }
   return f;
}
