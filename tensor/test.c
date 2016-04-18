#include <mpi.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

#define CHECK_MPI_RETURN(fn) assert ((fn) == MPI_SUCCESS)

int main(int argc, char *argv[]) {
  int crank, csize;
  MPI_Comm comm;
  int buflen = sizeof(double), sz = sizeof(double);
  double *buf;
  MPI_Win win;
  int proc = 1;
  size_t off = 0;
  MPI_Request req;

  MPI_Init(&argc, &argv);

  

  CHECK_MPI_RETURN(MPI_Comm_dup(MPI_COMM_WORLD, &comm));
  CHECK_MPI_RETURN(MPI_Comm_rank(comm, &crank));
  CHECK_MPI_RETURN(MPI_Comm_size(comm, &csize));

  CHECK_MPI_RETURN(MPI_Win_allocate(buflen, 1, MPI_INFO_NULL, 
                                    comm, &buf, &win));
  CHECK_MPI_RETURN(MPI_Win_lock_all(MPI_MODE_NOCHECK, win));

  /*     CHECK_MPI_RETURN(MPI_Alloc_mem(sizeof(double)*xblocks[px]*yblocks[py], MPI_INFO_NULL, &buf)); */

  MPI_Barrier(MPI_COMM_WORLD);

  {
    proc = crank^1;
    CHECK_MPI_RETURN(MPI_Rget(buf, sz, MPI_CHAR, proc, off, sz, MPI_CHAR, win, &req));
    CHECK_MPI_RETURN(MPI_Wait(&req, MPI_STATUS_IGNORE));
  }
  MPI_Barrier(MPI_COMM_WORLD);
  
  CHECK_MPI_RETURN(MPI_Win_unlock_all(win));
  CHECK_MPI_RETURN(MPI_Win_free(&win));

  MPI_Finalize();
  return 0;
}
