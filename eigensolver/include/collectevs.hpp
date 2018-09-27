#ifndef COLLECTEVS_H
#define COLLECTEVS_H

#include "utilities.hpp"
#include "slicing.hpp"

int collectevs(MPI_Comm comm, int nshifts, SpectralProbe *SPs, VectorXd &evals, MatrixXd &evecs, VectorXd &resnrms);

#endif

