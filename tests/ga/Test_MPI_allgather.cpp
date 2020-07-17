#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

#define NLOOP 1000

int main(int argc, char **argv)
{
  double tbeg, dst;
  double t_allgather_self = 0.0;
  double t_allgather_buf = 0.0;
  int i, me, nproc;
  double *send, *recv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&me);
  MPI_Comm_size(MPI_COMM_WORLD,&nproc);

  send = (double*)malloc(nproc*sizeof(double));
  recv = (double*)malloc(nproc*sizeof(double));

  if (me == 0) {
    printf("Testing Allgather on %d ranks\n",nproc);
  }
  for (i=0; i<NLOOP; i++) {
    tbeg = MPI_Wtime();
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
        send,sizeof(double),MPI_BYTE,MPI_COMM_WORLD);
    t_allgather_self += MPI_Wtime()-tbeg;
    tbeg = MPI_Wtime();
    MPI_Allgather(send,sizeof(double),MPI_BYTE,recv,sizeof(double),
        MPI_BYTE,MPI_COMM_WORLD);
    t_allgather_buf += MPI_Wtime()-tbeg;
  }
  MPI_Allreduce(&t_allgather_self,&dst,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  t_allgather_self = dst/((double)(NLOOP*nproc));
  MPI_Allreduce(&t_allgather_buf,&dst,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  t_allgather_buf = dst/((double)(NLOOP*nproc));
  if (me == 0) {
    printf("\nTotal time in Allgather in place:     %16.6f",t_allgather_self);
    printf("\nTotal time in Allgather using buffer: %16.6f",t_allgather_buf);
  }
  MPI_Finalize();
}