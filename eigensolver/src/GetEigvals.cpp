#include "utilities.hpp"

VectorXd GetEigvals(const MatrixXd &H, const MatrixXd &X)
{
   int ncols = X.cols();
   VectorXd d(ncols);

   for (int i = 0; i < ncols; i++)
      d(i) = (X.col(i)).dot(H*X.col(i)); 

   return d;
}
