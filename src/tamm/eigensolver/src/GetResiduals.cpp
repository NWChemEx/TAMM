#include "utilities.hpp"

VectorXd GetResiduals(const MatrixXd &H, const MatrixXd &S, const VectorXd &d, const MatrixXd &X)
{
   int ncols = X.cols();
   VectorXd res(ncols);

   for (int i = 0; i < ncols; i++)
      res(i) = (H*X.col(i) - d(i)*S*X.col(i)).norm();

   return res;
}
