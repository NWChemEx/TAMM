#include "sync_slices.hpp"

void sync_slices(MPI_Comm comm, int *nslices, SpectralProbe *SPs)
{
   VectorXi bufglb(*nslices), bufloc(*nslices);
   VectorXi nevcount(*nslices), nevoffset(*nslices);

   int rank, nprocs;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &nprocs);

   int n = (SPs[0].evecs).rows();

   // sync up inertial counts
   bufloc.setZero();
   bufglb.setZero();
   for (int j = 0; j < *nslices; j++)
      bufloc(j) = SPs[j].nev_below_shift;
   MPI_Allreduce(bufloc.data(),bufglb.data(),*nslices,MPI_INT,MPI_MAX,comm);
   for (int j = 0; j < *nslices; j++)
      SPs[j].nev_below_shift = bufglb[j];

   int js;
   for (js = 0; js < *nslices; js++) 
      if (SPs[js].nev_below_shift < 0) break;
   *nslices = js; 

   // sync up nev in each slice
   bufloc.setZero();
   nevcount.setZero();
   for (int j = 0; j < *nslices; j++)
      if (j%nprocs == rank)
         bufloc(j) = SPs[j].evals.size();
   MPI_Allreduce(bufloc.data(),nevcount.data(),*nslices,MPI_INT,MPI_MAX,comm);

   nevoffset.setZero();
   for (int j = 1; j < *nslices; j++)
      nevoffset(j) = nevoffset(j-1) + nevcount(j-1);

   int nevsum = nevcount.sum();
   if (rank == 0) logOFS << "nevsum = " << nevsum << endl;

   VectorXd allevals(nevsum), evalsloc(nevsum);
   VectorXd allresnm(nevsum), resnmloc(nevsum);
   
   allevals.setZero();
   evalsloc.setZero();
   allresnm.setZero();
   resnmloc.setZero();

   for (int j = 0; j < *nslices; j++) {
      if (j%nprocs == rank) 
         for (int k = 0; k < nevcount(j); k++) {
            evalsloc(nevoffset(j)+k) = SPs[j].evals(k);
            resnmloc(nevoffset(j)+k) = SPs[j].resnrms(k);
         }
         
   }
   MPI_Allreduce(evalsloc.data(),allevals.data(),nevsum,MPI_DOUBLE,MPI_SUM,comm);
   MPI_Allreduce(resnmloc.data(),allresnm.data(),nevsum,MPI_DOUBLE,MPI_SUM,comm);
   // if (rank == 0) cout << allevals << endl; 
   for (int i = 0; i < *nslices; i++) {
      SPs[i].evals   = allevals.segment(nevoffset(i),nevcount(i));
      SPs[i].resnrms = allresnm.segment(nevoffset(i),nevcount(i));
   }
}
