#include "utilities.hpp"

int MPI_Bcast_params(MPI_Comm comm, sislice_param *params)
{
  int ierr = 0;
  int buffer[10];
 
  int rank;
  MPI_Comm_rank(comm, &rank);

  // serialize
  if (rank == 0) {
     buffer[0] = params->n; 
     buffer[1] = params->nev; 
     buffer[2] = params->nshifts;
     buffer[3] = params->maxiter;
     buffer[4] = params->maxscf;
     buffer[5] = params->hnnz;
     buffer[6] = params->snnz;
     buffer[7] = params->maxlan;
  }

  ierr = MPI_Bcast(buffer, 10, MPI_INT, 0, comm);

  // deserialize 
  if (rank > 0) {
     params->n       = buffer[0]; 
     params->nev     = buffer[1];
     params->nshifts = buffer[2];
     params->maxiter = buffer[3]; 
     params->maxscf  = buffer[4];
     params->hnnz    = buffer[5];
     params->snnz    = buffer[6];
     params->maxlan  = buffer[7];

     params->hcolptr = new int[params->n+1];
     params->hrowind = new int[params->hnnz];
     params->hnzvals = new double[params->hnnz];

     params->scolptr = new int[params->n+1];
     params->srowind = new int[params->snnz];
     params->snzvals = new double[params->snnz];
  }

  ierr = MPI_Bcast(params->hcolptr, (params->n)+1, MPI_INT, 0, comm);
  ierr = MPI_Bcast(params->hrowind, params->hnnz, MPI_INT, 0, comm);
  ierr = MPI_Bcast(params->hnzvals, params->hnnz, MPI_DOUBLE, 0, comm);

  ierr = MPI_Bcast(params->scolptr, (params->n)+1, MPI_INT, 0, comm);
  ierr = MPI_Bcast(params->srowind, params->snnz, MPI_INT, 0, comm);
  ierr = MPI_Bcast(params->snzvals, params->snnz, MPI_DOUBLE, 0, comm);

  ierr = MPI_Bcast(params->logprefix, 200, MPI_CHAR, 0, comm);

  return ierr;
}

int MPI_Bcast_scf_params(MPI_Comm comm, scf_param *params)
{
  int ierr = 0;
  int buffer[10];
 
  int rank;
  MPI_Comm_rank(comm, &rank);

  // serialize
  if (rank == 0) {
     buffer[0] = params->nev; 
     buffer[1] = params->nshifts; 
     buffer[2] = params->maxiter;
     buffer[3] = params->maxscf;
     buffer[4] = params->maxlan;
  }

  ierr = MPI_Bcast(buffer, 5, MPI_INT, 0, comm);

  // deserialize 
  if (rank > 0) {
     params->nev     = buffer[0];
     params->nshifts = buffer[1];
     params->maxiter = buffer[2]; 
     params->maxscf  = buffer[3];
     params->maxlan  = buffer[4];
  }
  return ierr;
}

Matrix ReadAndBcastMatrix(MPI_Comm comm, char *filename)
{
   int rank, n, nnz, irow, jcol, ierr, info=0;
   MPI_Comm_rank(comm, &rank);
   FILE *fpmat = NULL;
   int *colptr, *rowind;
   double *nzvals;

   if (rank == 0) {
      fpmat = fopen(filename,"r");
      if (!fpmat) {
         fprintf(stderr,"ReadAndBcast: cannot open %s\n", filename);
         info = -1;
      }
      else {  
         fread(&n, sizeof(int), 1, fpmat);
         fread(&nnz, sizeof(int), 1, fpmat);
         printf("n = %d,  nnz = %d\n", n, nnz);

         colptr = (int*)malloc((n+1)*sizeof(int));
         rowind = (int*)malloc(nnz*sizeof(int));
         nzvals = (double*)malloc(nnz*sizeof(double));

         fread((int*)(colptr), sizeof(int), n+1, fpmat);
         fread((int*)(rowind), sizeof(int), nnz, fpmat);
         fread((double*)(nzvals), sizeof(double), nnz, fpmat);
         fclose(fpmat);
      }
   }

   ierr = MPI_Bcast(&info, 1, MPI_INT, 0, comm);
   if (!info) {
      ierr = MPI_Bcast(&n, 1, MPI_INT, 0, comm);
      MatrixXd A(n,n);
      if (rank == 0) {
         A.setZero();
         for (int jcol = 0; jcol < n; jcol++) {
            for (int i = colptr[jcol]-1; i < colptr[jcol+1]-1; i++) {
               irow = rowind[i]-1;
               A(irow,jcol) = nzvals[i];
               if (irow != jcol) A(jcol,irow) = nzvals[i];
            }
         }
      }
      MPI_Bcast(A.data(), n*n, MPI_INT, 0, comm);
      return A;
   }
   else {
      MPI_Finalize();
      exit(1);
   }
}


void print_results(std::ofstream &resultsfile, int iter, VectorXd &evals, 
                   VectorXd & resnrms)
{
   resultsfile << "SCF iteration: " << iter << std::endl;
   resultsfile << scientific;
         
   resultsfile << std::setw(7) << "n" << std::setw(14) << "eval" << std::setw(14) << "resnrm" << std::endl;

   VectorXi evalinds = sortinds(evals);
   for (int j = 0; j < evals.size(); j++) {
      resultsfile << std::setw(7) << j << std::setw(14) << evals(evalinds[j]) << std::setw(14) << resnrms(evalinds[j]) << std::endl;
   }
   resultsfile << std::endl;
}

VectorXi sortinds(VectorXd xs) {
   int m = xs.size();

   VectorXi inds(m);

   int numless, equal = 0;

   for (int i = 0; i < m; i++) {
      numless = 0;
      for (int j = 0; j < m; j++) {
         if (xs[j] < xs[i]) {
            numless++;
         }
         else if (xs[j] == xs[i] && i != j) {
            if (i < j) {
               numless++; 
            }
         }
      }
      inds[numless] = i;
   }
   return inds;
}
