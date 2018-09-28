#include "collectevs.hpp"

int collectevs(MPI_Comm comm, int nshifts, SpectralProbe *SPs, 
               VectorXd &evals, MatrixXd &evecs, VectorXd &resnrms)
{

   int rank;
   MPI_Comm_rank(comm, &rank);
   int nproc;
   MPI_Comm_size(comm, &nproc);

   int n = (SPs[rank].evecs).rows();
   // just to make sure every rank has the same n
   MPI_Bcast(&n, 1, MPI_INT, 0, comm);  

   int nev = evals.size();
   MatrixXd evecsloc(n,nev);
   evecsloc.setZero();

   int l = 0;
   for (int k = 0; k < nshifts; k++) {
      for (int j = 0; j < SPs[k].nselect; j++) {
         // cout << "n: " << l << ", probe: " << k << ", ind: " << SPs[k].selind(j) << " of " << SPs[k].evals.size() << endl;
         int jsel = SPs[k].selind(j);
         evals(l) = SPs[k].evals(jsel);
         resnrms(l) = SPs[k].resnrms(jsel);
         if (k%nproc == rank) {
            evecsloc.col(l) = (SPs[k].evecs).col(jsel);
         }
         l++;
      }
   }

   // MPI_Allreduce(evecsloc.data(),evecs.data(),n*l,MPI_DOUBLE,MPI_SUM,comm);

   return l;
}
