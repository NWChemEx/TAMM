#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "mpi.h"
#include "ga.h"
#include "macdecls.h"

#define DEFAULT_SIZE 1048576
#define MAX_LOOP 20

int main(int argc, char * argv[])
{
  int me, nproc, j, g_src;
  int64_t size, i;
  int64_t lo, hi, ld;
  int64_t *ptr;
  double tbeg, time;
  char op[2];

  MPI_Init(&argc, &argv);
  GA_Initialize();

  nproc = GA_Nnodes();
  me = GA_Nodeid();
  op[0] = '+';
  op[1] = '\0';

  size = ((long)nproc)*DEFAULT_SIZE;
  /* Find processor grid dimensions and processor grid coordinates */
  if (me==0) {
    printf("\nAllocation test running on %d processors\n",nproc);
    printf("\nArray dimension is %ld\n",size);
  }

  /* Create GA and set all elements to zero */
  tbeg = GA_Wtime();
  g_src = NGA_Create64(C_LONG, 1, &size, "source", NULL);
  time = GA_Wtime()-tbeg;

  /* Fill the global array with unique values */
  NGA_Distribution64(g_src,me,&lo,&hi);
  ld = DEFAULT_SIZE;
  NGA_Access64(g_src,&lo,&hi,&ptr,&ld);
  for (i=lo; i<=hi; i++) {
    ptr[i-lo] = i;
  }
  NGA_Release64(g_src,&lo,&hi);
  GA_Sync();
  GA_Dgop(&time,1,op);
  if (me == 0) {
    printf("\nTotal elapsed time in allocation: %f\n",time/((double)nproc));
  }

  GA_Destroy(g_src);

  GA_Terminate();
  MPI_Finalize();

  return 0;
}
