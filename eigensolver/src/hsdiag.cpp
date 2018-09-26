#include "utilities.hpp"
#include "slicing.hpp"
#include "evalidate.hpp"
#include "collectevs.hpp"
#include "sync_slices.hpp"
#include "getshifts.hpp"

using Eigen::Map;

void hsdiag(MPI_Comm comm, int iterscf, Matrix &H, Matrix &S, int nev, int nshifts, Matrix &eps, Matrix &evecs) {
   int rank;
   std::ofstream resultsfile;
   MPI_Comm_rank(comm, &rank);
   double t1 = MPI_Wtime();

   VectorXd resnrms;

   if (rank == 0) cout << "running spectrum slicing" << endl;
   if (rank == 0) resultsfile.open("results.txt");

   int n = H.rows();

   VectorXd shifts;
   int subdim, maxcnt = 0;

   shifts = getshifts(comm, H, S, nev, nshifts, &maxcnt);
   subdim = 2*maxcnt;

   if (rank == 0) {
      std::cout << "nshifts = " << nshifts << std::endl;
      std::cout << "nev = " << nev << std::endl;
      std::cout << "subdim = " << subdim << std::endl;
   }

   // set up probes 
   SpectralProbe *SPs;
   SPs = new SpectralProbe[nshifts];
   Init_SPs(comm, SPs, shifts);

   VectorXi *inds = NULL;

   int maxiter = 5;
   logOFS << "rank " << rank << " starting slicing" << endl;
   slicing(comm, H, S, n, nev, subdim, maxiter, nshifts, SPs);
   logOFS << "rank " << rank << " done slicing" << endl;
   MPI_Barrier(comm);

   sync_slices(comm, &nshifts, SPs);

   inds = new VectorXi[nshifts];

   // returns the indices of the selcted eigenvalues in inds
   // and the total number of selected eigenvalues in nev
   evalidate(comm, SPs, nshifts, n, nev);

   // collect eigenvalues and residuals into a single array for printing
   // this may change depending on how what type of information is
   // needed by SCF
   VectorXd evals;
   evals.resize(nev);
   resnrms.resize(nev);
   //evecs.resize(n,nev);
    
   collectevs(nshifts, SPs, evals, resnrms); 
   if (rank == 0) print_results(resultsfile,0,evals,resnrms);

   MPI_Barrier(comm);

   if (rank == 0) resultsfile.close();

   // delete slices;
   // delete inds;

   // convert evals to a Matrix
   Map<Matrix> evalmat(evals.data(),evals.size(),1); 
   eps = evalmat;

   double t2 = MPI_Wtime();
   if (rank == 0) cout << "ELASPSED TIME: " << t2-t1 << endl;
/*
   MPI_Finalize();
   exit(0);
*/
}
