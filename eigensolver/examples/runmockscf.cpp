#include "utilities.hpp"
#include "slicing.hpp"

std::ofstream logOFS;

int main(int argc, char *argv[]) {

   int n, nev, nshifts, maxscf;
   int args[6];
   VectorXd shifts;
   Matrix H, S;
   char smatfname[200], hmatfname[200], logfname[200];
   std::ofstream resultsfile;
   scf_param params;

   MPI_Init(NULL,NULL);

   MPI_Comm comm = MPI_COMM_WORLD;

   int rank;
   MPI_Comm_rank(comm, &rank);
   int nproc;
   MPI_Comm_size(comm, &nproc);

   double t1 = MPI_Wtime();
   if (rank == 0) {
      int ierr;
      FILE *fp;
      char fname[100];
      if (argc < 2) {
         fprintf(stderr, "Missing input arguments! \n");
         fprintf(stderr, "runmockscf <input filename> \n");
         exit(2);
      }

      sscanf(*(argv+1),"%s", fname);
      printf("input fname =  %s\n", fname);

      ierr = parse_scf_input(fname, &params);
      if (ierr!=0) {
         fprintf(stderr,"error in the input file\n");
         exit(1);
      }
   }

   MPI_Bcast_scf_params(comm, &params); 
   nev     = params.nev;
   nshifts = params.nshifts;
   maxscf  = params.maxscf;
  
   sprintf(logfname,"%s.log.%d", params.logprefix, rank);
   logOFS.open(logfname);

   if (rank == 0) cout << "reading the S matrices" << endl;

   sprintf(smatfname, "%s.chb\0", params.smatfname);
   S = ReadAndBcastMatrix(comm, smatfname);

   // eigenvalues and eigenvectors
   Matrix eps;
   Matrix evecs;

   for (int iterscf = 0; iterscf < maxscf; iterscf++) {
      // update H 
      sprintf(hmatfname,"%s_%d.chb\0",params.hmatfname,iterscf);
      H = ReadAndBcastMatrix(comm, hmatfname);
      hsdiag(comm, iterscf, H, S, nev, nshifts, eps, evecs);
   }

   logOFS.close();
   return MPI_Finalize(); 
}
