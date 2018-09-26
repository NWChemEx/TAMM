#include "utilities.hpp"

VectorXd HSLanczos(const MatrixXd &H, const MatrixXd &S, const VectorXd &v0, MatrixXd &T, MatrixXd &V)
{
   char lower = 'L', left = 'L', trans = 'T', notrans = 'N', nunit = 'N';
   int  nrhs = 1;
   double done = 1.0;

   int n = H.rows();
   int niter = V.cols();
   VectorXd f(n), v(n), SiHv(n);
   MatrixXd R(n,n); // a copy of S to hold the Cholesky factor
 
   R = S;
  
   int ierr = 0;
   // S = L*L'
 
   double t1, t2, t3, t4; 
   t1 = omp_get_wtime();
   dpotrf_(&lower, &n, R.data(), &n, &ierr);   
   t2 = omp_get_wtime();
   logOFS << "Cholesky factorization time = " <<  t2-t1 << std::endl;
   if (ierr) {
      logOFS << " dpotrf failure, ierr = " << ierr << std::endl;
      exit(1);
   }

   // normalize the starting vector
   double beta = v0.norm();
   v = v0/beta; 

   t1 = omp_get_wtime();
   for (int iter = 0; iter < niter; iter++) {
      // MATVEC with inv(L)*H*inv(L')
      V.block(0,iter,n,1) = v;
      t3 = omp_get_wtime();
      dtrsm_(&left, &lower, &trans, &nunit, &n, &nrhs, &done, R.data(), &n, v.data(), &n);  
      t4 = omp_get_wtime();
      SiHv = H*v;
      dtrsm_(&left, &lower, &notrans, &nunit, &n, &nrhs, &done, R.data(), &n, SiHv.data(), &n);  
      // orthogonalization 
      T.block(0,iter,iter+1,1) = V.block(0,0,n,iter+1).transpose() *SiHv;
      f = SiHv - V.block(0,0,n,iter+1)*T.block(0,iter,iter+1,1);
      beta = f.norm();
      if (iter < niter-1) T(iter+1,iter) = beta;
      v =  f/beta;
   }
   t2 = omp_get_wtime();
   logOFS << "Lanczos time = " <<  t2-t1 << std::endl;
   logOFS << "Apply Op time = " <<  (t4-t3)*niter << std::endl;
   return f;
}
