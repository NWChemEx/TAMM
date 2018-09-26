#include "utilities.hpp"
#include "slicing.hpp"

std::ofstream logOFS;

int main(int argc, char *argv[]) {

   int n, nev;
   int args[6];
   char logfname[200];
   VectorXd shifts;
//   MatrixXd H, S;
   Matrix H, S;
   std::ofstream resultsfile;
   sislice_param params;

   MPI_Init(NULL,NULL);

   MPI_Comm comm = MPI_COMM_WORLD;

   int rank;
   MPI_Comm_rank(comm, &rank);
   int nproc;
   MPI_Comm_size(comm, &nproc);

   double t1 = MPI_Wtime();
   if (rank == 0) {
      int ierr;
      FILE *fp, *fpmat;
      char fname[100];
      if (argc < 2) {
         fprintf(stderr, "Missing input arguments! \n");
         fprintf(stderr, "runscf1 <input filename> \n");
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
   
   sprintf(logfname,"%s.log.%d", params.logprefix, rank);
   logOFS.open(logfname);
  
   H.resize(n,n);
   S.resize(n,n);

   if (rank == 0) printf("building matrices\n");

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
//   MatrixXd evecs;
   Matrix eps;
   Matrix evecs;

   int iterscf = 0;
   int nshifts = std::ceil(nev/10.0);

   hsdiag(comm, iterscf, H, S, nev, nshifts, eps, evecs);

   logOFS.close();
   return MPI_Finalize(); 
}
