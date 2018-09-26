
#ifndef SLICING_MPI_H
#define SLICING_MPI_H

class SpectralProbe {
   public:
      int      ind;
      double   shift;
      int      nev_below_shift;
      VectorXd evals;
      MatrixXd evecs;
      VectorXd resnrms;
      VectorXi valind; // valid indices
      VectorXi accept; // the status of each element of evals (1 accped, 0 not)
      int      prev;
      int      next;
      int      nvalid;
      int      mpirank;
};

void slicing(MPI_Comm comm, const MatrixXd &H, const MatrixXd &S, int n, int nev, int nevloc, int maxiter, int nslices, SpectralProbe *SPs);
void hsdiag(MPI_Comm comm, int iterscf, Matrix &H, Matrix &S, int nev, int nshifts, Matrix &evals, Matrix &evecs);
void Init_SPs(MPI_Comm comm, SpectralProbe *SPs, VectorXd &shifts);

#endif
