#include <iostream>
#include <Eigen/Dense>
#include <omp.h>
#include "slice.hpp"
#ifdef USE_MPI
#include <mpi.h>
#endif

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

#ifdef USE_MPI
void enforcecounts(MPI_Comm comm, slice *slices, int nslices, int n, int nev, VectorXi *inds);
#else
void enforcecounts(slice *slices, int nslices, int n, int nev, VectorXi *inds);
#endif
VectorXi sortinds(VectorXd xs);
