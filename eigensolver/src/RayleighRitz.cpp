#include "utilities.hpp"

using Eigen::MatrixXd;
using Eigen::SelfAdjointEigenSolver;

MatrixXd RayleighRitz(const MatrixXd &H, const MatrixXd &X0)
{
   // X0 should be S orthonormal already
   int ncols = X0.cols();
   MatrixXd T(ncols,ncols);
   T = X0.transpose()*(H*X0);
   // may have to replace this with LAPACK
   SelfAdjointEigenSolver<MatrixXd> es(T);
   MatrixXd Q = es.eigenvectors();
   return X0*Q;
}
