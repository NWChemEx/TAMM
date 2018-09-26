#ifndef EVALIDATE_H
#define EVALIDATE_H

#include "utilities.hpp"
#include "slicing.hpp"

void evalidate(MPI_Comm comm, SpectralProbe *SPs, int nslices, int n, int nev);
#endif
