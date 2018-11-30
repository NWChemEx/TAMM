#include "itereig.hpp"
#include "sisubit.hpp"

void itereig(MPI_Comm comm, const MatrixXd &H, const MatrixXd &S, int n, int maxiter, SpectralProbe *SP)
{
  double shift = SP->shift;
  int    nev_below_shift;
  VectorXd evals;
  VectorXd resnrms;
  MatrixXd evecs = SP->evecs;

  // LDLT factorization using LAPACK, the Eigen version is too slow
  MatrixXd HS = H - shift*S;
  int lwork = 100*n;
  int ierr;
  double *work;
  int *ipiv;
  work = new double[lwork];
  ipiv = new int [n];
  char lower = 'L';

  // A = new double[n*n];
  // Map<MatrixXd>(A,n,n) = HS;
  // std::cout << "computing LDLT factorization" << std::endl;
  double t1 = omp_get_wtime();
  //dsytrf_(&lower, &n, HS.data(), &n, ipiv, work, &lwork, &ierr);  
  ierr = LAPACKE_dsytrf(LAPACK_COL_MAJOR, lower, n, HS.data(), n, ipiv);  

  double t2 = omp_get_wtime();
  logOFS << "factorization complete, time = " <<  t2-t1 << std::endl;
  int i = 0;
  nev_below_shift = 0;
  double *A = HS.data();
  while (i < n) {
    if ((i < n-1) && (ipiv[i] < 0)) {
      nev_below_shift++;
      i++;
    }
    else if (A[i+i*n] < 0) {
      nev_below_shift++;
    }
  i++;
  }
  
  logOFS << "nev_below_shift: " << nev_below_shift << std::endl;
  SP->nev_below_shift = nev_below_shift;
  
  delete work;

  MatrixXd X = sisubit(comm, HS, S, evecs, ipiv, maxiter);

  evecs = RayleighRitz(H, X);
  evals = GetEigvals(H, evecs);
  resnrms = GetResiduals(H, S, evals, evecs);
  SP->evals = evals;
  SP->evecs = evecs;
  SP->resnrms = resnrms;

  delete ipiv;
}

