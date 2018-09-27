#include "utilities.hpp"
#include "slicing.hpp"
#include "itereig.hpp"
#include "sisubit.hpp"

void slicing(MPI_Comm comm, const MatrixXd &H, const MatrixXd &S, int n, int nev, int nevloc, int maxiter, int nshifts, SpectralProbe *SPs)
{
   MatrixXd tempmat;
   int nthreads, nev_below_shift;

   int rank;
   MPI_Comm_rank(comm, &rank);
   int nproc;
   MPI_Comm_size(comm, &nproc);

   int dim;
   double lb, ub;

   logOFS << endl;
   for (int i = 0; i < nshifts; i++) {  
      // work on the shift that belongs to me
      if (i%nproc == rank) {
         logOFS << "rank " << rank << " building random basis" << endl;
         #pragma omp parallel
         {
            nthreads = omp_get_num_threads();
         }
         dim = max(nthreads,nevloc);

         // generate random initial guess of the eigenvector
         // only if no valid eigenvalue approximation is 
         // available from the previous SCF (outer) cycles
         if (SPs[i].nselect == -1) { 
            SPs[i].evecs = MatrixXd::Random(n,dim);
         }
         SPs[i].ind = i;
         SPs[i].evals.resize(dim);
         SPs[i].resnrms.resize(dim);

         logOFS << "slice: " << i << ", rank: " << rank << ", shift: " << SPs[i].shift << ", subspace dim: " << dim << std::endl;
         
         itereig(comm, H, S, n, maxiter, &SPs[i]);

         if (SPs[i].nev_below_shift >= nev) {
            break;
         }
      }
   }
}

void Init_SPs(MPI_Comm comm, SpectralProbe *SPs, VectorXd &shifts)
{
   int rank, nproc;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &nproc);

   int nshifts = shifts.size();

   for (int i = 0; i < nshifts; i++) {
      SPs[i].nev_below_shift=-1;
      SPs[i].ind = 0;
      SPs[i].shift = shifts(i);
      SPs[i].nselect = -1;
      if (i==0) {
         SPs[i].prev = -1;
      }
      else {
         SPs[i].prev = i-1;
      }
      if (i==nshifts-1) {
         SPs[i].next = -1;
      }
      else {
         SPs[i].next = i+1;
      }
      SPs[i].mpirank = i%nproc;
   } 
}
