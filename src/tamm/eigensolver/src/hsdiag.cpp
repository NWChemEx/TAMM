#include "utilities.hpp"
#include "slicing.hpp"
#include "evalidate.hpp"
#include "collectevs.hpp"
#include "sync_slices.hpp"
#include "getshifts.hpp"

using Eigen::Map;

void hsdiag(MPI_Comm comm, int iterscf, Matrix &H, Matrix &S, int nev, Matrix &eps, Matrix &evecs) {
   int rank;
   std::ofstream resultsfile;
   MPI_Comm_rank(comm, &rank);
   double t1 = MPI_Wtime();

   const int limit = 500;

   // change to column major order
   MatrixXd HC, SC;


   int n = H.rows();
   int nshifts = std::ceil(nev/10.0);

   VectorXd shifts;
   int subdim, maxcnt = 0;

   HC = H;
   SC = S;

   if (n <= limit) {
      // just call LAPACK eigensolver
      VectorXd ev1(n);

      char  lower = 'U', needv = 'V';
      int   gtype = 1, lgvdwork = n*n, info = 0, ierr;
      double *gvdwork;
      gvdwork = new double[lgvdwork];

      dsygv_(&gtype,&needv,&lower,&n,HC.data(),&n,SC.data(),&n,ev1.data(),
             gvdwork,&lgvdwork,&ierr);
      delete [] gvdwork;
      // convert evals to a Matrix
      Map<Matrix> evalmat1(ev1.data(),n,1); 
      eps = evalmat1;
      evecs = HC;
   }
   else {

      if (rank == 0) cout << "running spectrum slicing" << endl;
      if (rank == 0) resultsfile.open("results.txt");

      shifts = getshifts(comm, HC, SC, nev, nshifts, &maxcnt);
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

      int maxiter = 5; // hard code for now
      logOFS << "rank " << rank << " starting slicing" << endl;
      slicing(comm, H, S, n, nev, subdim, maxiter, nshifts, SPs);
      logOFS << "rank " << rank << " done slicing" << endl;
      MPI_Barrier(comm);

      sync_slices(comm, &nshifts, SPs);

      // returns the indices of the selcted eigenvalues in inds
      // and the total number of selected eigenvalues in nev
      evalidate(comm, SPs, nshifts, n, nev);

      // collect eigenvalues and residuals into a single array for printing
      // this may change depending on how what type of information is
      // needed by SCF
      VectorXd evals;
      evals.resize(nev);
      MatrixXd evecmat;
      evecmat.resize(n,nev);
      VectorXd resnrms;
      resnrms.resize(nev);
  
      // nevf maybe less than nev
      int nevf = collectevs(comm, nshifts, SPs, evals, evecmat, resnrms); 
 
      if (rank == 0) {
         VectorXd evalsf = evals.head(nevf);
         VectorXd resnrmf = resnrms.head(nevf);
         print_results(resultsfile,0,evalsf,resnrmf);
      }

      MPI_Barrier(comm);

      if (rank == 0) resultsfile.close();

      // convert evals to a Matrix
      Map<Matrix> evalmat(evals.data(),nevf,1); 
      eps = evalmat;
      evecs = evecmat.block(0,0,n,nevf);
   } // endif (n <= 500)
   double t2 = MPI_Wtime();
   // if (rank == 0) cout << "ELASPSED TIME: " << t2-t1 << endl;
}
