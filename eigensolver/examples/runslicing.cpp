#include "utilities.hpp"
#include "slicing.hpp"
#include "evalidate.hpp"
#include "collectevs.hpp"
#include "sync_slices.hpp"
#include "getshifts.hpp"

void kmeans(VectorXd &, VectorXd &);

std::ofstream logOFS;

int main(int argc, char *argv[]) {

   int n, nev, nshifts, maxcnt, maxiter, maxscf, nevf;
   int args[6];
   char logfname[200];
   VectorXd shifts;
   MatrixXd H, S;
   std::ofstream resultsfile;
   sislice_param params;

   MPI_Init(NULL,NULL);

   MPI_Comm comm = MPI_COMM_WORLD;

   int rank;
   MPI_Comm_rank(comm, &rank);
   int nproc;
   MPI_Comm_size(comm, &nproc);
   if (rank == 0) cout << "number of MPI tasks = " << nproc << endl;

   double t1 = MPI_Wtime();
   if (rank == 0) {
      int ierr;
      FILE *fp, *fpmat;
      char fname[100];
      if (argc < 2) {
         fprintf(stderr, "Missing input arguments! \n");
         fprintf(stderr, "runslicing <input filename> \n");
         exit(2);
      }

      sscanf(*(argv+1),"%s", fname);
      printf("input fname =  %s\n", fname);

      ierr = parseinput(fname, &params);
      if (ierr!=0) {
         fprintf(stderr,"error in the input file\n");
         exit(1);
      }
   }
   MPI_Bcast_params(comm, &params); 

   n       = params.n;
   nev     = params.nev;
   nshifts = params.nshifts;
   maxiter = params.maxiter;
   maxscf  = params.maxscf;

   if (rank == 0) {
      cout << "nev = " << params.nev << std::endl;
      cout << "maxiter = " << maxiter << endl;
      cout << "maxscf  = " << maxscf << endl;
   }


   sprintf(logfname,"%s.log.%d", params.logprefix, rank);
   logOFS.open(logfname);
  
   H.resize(n,n);
   S.resize(n,n);

   H.setZero();
   int irow, jcol, i;
   for (jcol = 0; jcol < n; jcol++) {
      for (i = params.hcolptr[jcol]-1; i < params.hcolptr[jcol+1]-1; i++)  {
         irow = params.hrowind[i]-1;
         H(irow,jcol) = params.hnzvals[i];
         if (irow != jcol) H(jcol,irow) = params.hnzvals[i];
      }
   }

   S.setZero();
   for (jcol = 0; jcol < n; jcol++) {
      for (i = params.scolptr[jcol]-1; i < params.scolptr[jcol+1]-1; i++)  {
         irow = params.srowind[i]-1;
         S(irow,jcol) = params.snzvals[i];
         if (irow != jcol) S(jcol,irow) = params.snzvals[i];
      }
   }
   // delete sparse input matrix

#ifdef full_diag
   MatrixXd H1(n,n), S1(n,n);
   H1 = H;
   S1 = S;
   VectorXd ev1(n);

   char  lower = 'U', needv = 'V';
   int   gtype = 1, lgvdwork = n*n, info = 0;
   double *gvdwork;
   gvdwork = new double[lgvdwork];

   double t1 = omp_get_wtime();
   dsygv_(&gtype,&needv,&lower,&n,H1.data(),&n,S1.data(),&n,ev1.data(),
          gvdwork,&lgvdwork,&ierr);
   double t2 = omp_get_wtime();
   std::cout << "full diagonalization time = " <<  t2-t1 << std::endl;

   delete [] gvdwork;
#endif

   VectorXd evals, resnrms;
   MatrixXd evecs;

   if (rank == 0) resultsfile.open("results.txt");

   if (rank == 0) cout << " generate initial shifts..." << endl;
   shifts = getshifts(comm, H, S, nev, nshifts, &maxcnt);
   if (rank == 0) cout << "max estimated number of eigenvalue per slice = " 
                       << maxcnt << std::endl;

   // set up spectral probes
   SpectralProbe *SPs;
   SPs = new SpectralProbe[nshifts];
   Init_SPs(comm, SPs, shifts);

   VectorXi *inds = NULL;

   if (rank == 0) cout << " start spectrum slicing " << endl;
   for (int iter = 0; iter < maxscf; iter++) {
      if (rank == 0) cout << " SCF iteration: " << iter << std::endl;
      slicing(comm, H, S, n, nev, 2*maxcnt, maxiter, nshifts, SPs);
      MPI_Barrier(comm);

      sync_slices(comm, &nshifts, SPs);

      // inds keeps track of local indices of validated eigenvalues 
      // in each slice 
      if (iter == 0) {
         inds = new VectorXi[nshifts];
      }

      // returns the indices of the selcted eigenvalues in inds
      // and the total number of selected eigenvalues in nev
      evalidate(comm, SPs, nshifts, n, nev);

      // collect eigenvalues and residuals into a single array for printing
      // this may change depending on how what type of information is
      // needed by SCF
      evals.resize(nev);
      resnrms.resize(nev);
      evecs.resize(n,nev);
    
      nevf = collectevs(comm, nshifts, SPs, evals, evecs, resnrms); 

      if (rank == 0) {
         VectorXd evals1 = evals.head(nevf);
         VectorXd resnrm1 = resnrms.head(nevf);
         print_results(resultsfile,iter,evals1,resnrm1);
      }
#ifdef KMEANS
      if (iter < maxscf-1) {
         kmeans(shifts,evals);

         if (rank == 0)  {
            cout << " new shifts = " << endl;
            cout << shifts << endl;
         }

         // bind new shifts to SPs
         for (int i = 0; i < nshifts; i++) SPs[i].shift = shifts(i);
      }
#endif


      MPI_Barrier(comm);
   } // end scf iter

   if (rank == 0) cout << " done spectrum slicing " << endl;
   if (rank == 0) resultsfile.close();

   // delete inds;

   double t2 = MPI_Wtime();

   if (rank == 0) cout << "ELASPSED TIME: " << t2-t1 << endl;
   logOFS.close();

   return MPI_Finalize(); 
}
