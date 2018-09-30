#include "utilities.hpp"

using Eigen::Map;

Matrix SquareRoot(Matrix &H) {

   // just call LAPACK eigensolver to diagonalize H
   int  n = H.rows();

   MatrixXd HC(n,n), H1(n,n), H2(n,n);
   Matrix S(n,n);
   VectorXd ev(n);

   // convert H to column major required by LAPACK
   HC = H;

   char  lower = 'U', needv = 'V';
   int   ldwork = 1 + 6*n + 2*n*n;
   int   liwork = 3 + 5*n;
   int   ierr=0;
   double *dwork;
   int    *iwork;
   dwork = new double[ldwork];
   iwork = new int[liwork];

   dsyevd_(&needv,&lower,&n,HC.data(),&n,ev.data(),
           dwork,&ldwork,iwork,&liwork,&ierr);

   delete [] dwork;
   delete [] iwork;
 
   for (int j = 0; j < n; j++) {
      if (ev(j) >= 0.0) {
         H1.col(j) = HC.col(j)*sqrt(ev(j));
      }
      else {
         cout << "Matrix not positive semidefinte!" << endl;
         cout << j << "th eigenvalue = " << ev(j) << endl;
         exit(1);
      }
   }

   H2 = H1*HC.transpose(); 

   S = H2; // convert back to Row major

   return S;
}
