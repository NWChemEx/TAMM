#ifndef HSEIG_H
#define HSEIG_H

#include "utilities.hpp"
#include "slicing.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;

void itereig(MPI_Comm comm, const MatrixXd &H, const MatrixXd &S, int n, int maxiter, SpectralProbe *SP);

#endif
