#include <iostream>
#include <Eigen/Dense>
#include <omp.h>
#ifdef USE_MPI
#include <mpi.h>
#endif

using Eigen::MatrixXd;
using Eigen::VectorXd;

MatrixXd cholQR(const MatrixXd &S, const MatrixXd &X0);
#ifdef NO_MPI
MatrixXd sisubit(const MatrixXd &HS, const MatrixXd &S, MatrixXd &X0, int *ipiv, int maxiter);
#else
MatrixXd sisubit(MPI_Comm comm, const MatrixXd &HS, const MatrixXd &S, MatrixXd &X0, int *ipiv, int maxiter);
#endif
MatrixXd RayleighRitz(const MatrixXd &H, const MatrixXd &X0);
VectorXd GetEigvals(const MatrixXd &H, const MatrixXd &X);
VectorXd GetResiduals(const MatrixXd &H, const MatrixXd &S, const VectorXd &d, const MatrixXd &X);
