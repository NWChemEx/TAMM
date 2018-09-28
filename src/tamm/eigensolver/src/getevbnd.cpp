#include "utilities.hpp"

VectorXd getevbnd(const VectorXd &d, const MatrixXd &S, const double &beta)
{
// Compute lower/upper bounds of the spectrum of (H,S)
// returns evbnds(0) lower bound of the spectrum
//         evbnds(1) upper bound of the spectrum

   VectorXd evbnds(2);
   evbnds.setZero();

   double inf = 1.0e50; // hardwired for now, need to move the def to .hpp
   evbnds(0) = inf;
   evbnds(1) = -inf;

   int n = S.rows();

   double t, x;
   for (int j=0;j<n;j++) { 
      t = fabs(beta*S(n-1,j));
      x = d(j) - t;
      if (x < evbnds(0)) {
         evbnds(0) = x;
      }
      x = d(j) + t;
      if (x > evbnds(1)) {
         evbnds(1) = x;
      }
   }

   return evbnds;
}
