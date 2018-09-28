#include "partev.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

int partev(const MatrixXd &T, const VectorXd &f, VectorXd &shifts, int nev, int nevloc)
{
   // partition the low end of spectrum that contains the 
   // leading nev eigenvalues into slices with approximately
   // nevloc eigenvalues in each slice. shifts contain end
   // points of the slices
   // 
   // T,f returned from the Lanczos iteration
   // n is the size of the matrix
   // nev is the number of eigenvalues to be computed (at the low end of the spectrum)
   // nevloc is the number of eigenvalues per slice

   // diagonalize T 
   int nt = T.rows(); // dimension of the tridiagonal
   int n  = f.rows(); // dimension of the original problem

   MatrixXd S(nt,nt);
   S = T;
   VectorXd d(nt);
   char vec = 'V', lower = 'L';
   int  lwork = nt*nt, ierr = 0;
   double *work;
   work = new double [lwork];
   dsyev_(&vec,&lower,&nt,S.data(),&nt,d.data(),work,&lwork,&ierr);
   if (ierr) {
      std::cout << "dsyev failed: ierr = " << ierr << std::endl;
      exit(1);
   }
   delete[] work;

   VectorXd evbnds(2);
   double beta = f.norm();
   evbnds = getevbnd(d, S, beta);

   // define the boundaries of intervals on which eigenvalues are 
   // to be assigned. The number of eigenvalues in each interval is 
   // proportional to tau*nev values;

   VectorXd ritzbnd(nt), ritzsize(nt-1);
   VectorXi ritzcnt(nt-1);

   ritzbnd(0)    = evbnds(0);
   ritzbnd(nt-1) = evbnds(1);
   for (int j = 1; j < nt-1; j++) 
      ritzbnd(j) = (d(j-1)+d(j))/2;

   for (int j = 0; j < nt-1; j++)
      ritzsize(j) = ritzbnd(j+1)-ritzbnd(j);

   for (int j = 0; j < nt-1; j++)
      ritzcnt(j)  = ceil(S(0,j)*S(0,j)*n);

   VectorXd bpoints(nev);

   int nevsum = 0, nslices, npts = 0;
   double del;
   for (int j = 0; j < nt-1; j++) {
      nslices = ceil((double) ritzcnt(j) / (double) nevloc);
      if (nslices>0) {
         del = ritzsize(j)/nslices;
         for (double t = ritzbnd(j); t <= ritzbnd(j+1)-del; t=t+del) {
            bpoints(npts) = t;
            npts++;
         }
      }
      nevsum += ritzcnt(j);
      if (nevsum > nev) break;
   }

   for (int j = 0; j < npts-1; j++)
     shifts(j) = (bpoints(j)+bpoints(j+1))/2.0;

   return npts-1;
}
