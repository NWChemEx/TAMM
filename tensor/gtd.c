
#include "gt.h"
#include "gtu.h"
#include "gti.h"

#if GT_MPI_PORT

int GT_wait(GTD_handle handle) {
  MPI_Wait(&handle, MPI_STATUS_IGNORE);
  return GT_SUCCESS;
}

int GT_test(GTD_handle handle, int *flag) {
  MPI_Test(&handle, flag, MPI_STATUS_IGNORE);
  return GT_SUCCESS;
}

#endif
