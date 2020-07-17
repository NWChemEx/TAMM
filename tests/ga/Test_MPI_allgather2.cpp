#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>


#define NLOOP 10

struct reg_entry {
  void *next;
  void *buf;
  size_t len;
  void *mapped;
  int rank;
  char name[31];
};

int main(int argc, char **argv)
{
  int nproc, me;
  int i, j;
  MPI_Comm my_comm;
  int worker = 0;
  int progress = 0;
  double tbeg, tbeg0, dst;
  double t_total = 0.0;
  double t_barrier_setup = 0.0;
  double t_allgather_setup = 0.0;
  double t_split_setup = 0.0;
  double t_barrier2_setup = 0.0;
  double t_allgather = 0.0;
  double t_translate = 0.0;
  double t_barrier = 0.0;
  int ierr;
  // Initialize MPI library
  ierr = MPI_Init(&argc, &argv);
  tbeg = MPI_Wtime();
  tbeg0 = MPI_Wtime();
  ierr = MPI_Barrier(MPI_COMM_WORLD);
  t_barrier_setup += MPI_Wtime()-tbeg;

  // Check host IDs to find processors that are on the same node
  ierr = MPI_Comm_rank(MPI_COMM_WORLD,&me);
  ierr = MPI_Comm_size(MPI_COMM_WORLD,&nproc);
  long *hostid = (long*)malloc(sizeof(long)*nproc);
  hostid[me] = gethostid();
  int *orig_rank = (int*)malloc(sizeof(int)*nproc);
  tbeg = MPI_Wtime();
  ierr = MPI_Allgather(MPI_IN_PLACE, 1, MPI_LONG,
      hostid, 1, MPI_LONG, MPI_COMM_WORLD);
  t_allgather_setup += MPI_Wtime()-tbeg;
  // sort host IDs into consecutive order using bubble sort (not
  // efficient)
  for (i=0; i<nproc; i++) orig_rank[i] = i;
  for (i=0; i<nproc; i++) {
    long ltmp;
    int itmp;
    for (j=i+1; j< nproc; j++) {
      if (hostid[j] < hostid[i]) {
        ltmp = hostid[i];
        hostid[i] = hostid[j];
        hostid[j] = ltmp;
        itmp = orig_rank[i];
        orig_rank[i] = orig_rank[j];
        orig_rank[j] = itmp;
      }
    }
  }
  // find number of nodes
  int count = 1;
  for (i=1; i<nproc; i++) {
    if (hostid[i] != hostid[i-1]) {
      count++;
    }
  }
  // Find starting index of each node
  int *start_index = (int*)malloc(sizeof(int)*count);
  start_index[0] = 0;
  count = 1;
  for (i=1; i<nproc; i++) {
    if (hostid[i] != hostid[i-1]) {
      start_index[count] = i;
      count++;
    }
  }
  // Sort all ranks within a node
  for (i=0; i<count; i++) {
    int imin = start_index[i];
    int imax;
    if (i<count-1) {
      imax = start_index[i+1];
    } else {
      imax = nproc;
    }
    for (j=imin; j<imax; j++) {
      int k;
      for (k=j+1; j<imax; j++) {
        if (orig_rank[i] > orig_rank[j]) {
          int itmp;
          itmp = orig_rank[i];
          orig_rank[i] = orig_rank[j];
          orig_rank[j] = itmp;
        }
      }
    }
  }
  // split procs into two groups. One is the new "world" group, the second
  // is a group representing the progress ranks
  int color = 0;
  if (me < nproc-1) {
    if (hostid[me] != hostid[me+1]) {
      color = 1;
    }
  } else {
    color = 1;
  }

  worker = 1;
  progress = 0;
  if (color == 1) {
    worker = 0;
    progress = 1;
  }
  tbeg = MPI_Wtime();
  ierr = MPI_Comm_split(MPI_COMM_WORLD,color,me,&my_comm);
  t_split_setup += MPI_Wtime()-tbeg;

  tbeg = MPI_Wtime();
  ierr = MPI_Barrier(MPI_COMM_WORLD);
  t_barrier2_setup += MPI_Wtime()-tbeg;
  MPI_Allreduce(&t_barrier_setup,&dst,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  t_barrier_setup = dst/((double)nproc);
  MPI_Allreduce(&t_allgather_setup,&dst,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  t_allgather_setup = dst/((double)nproc);
  MPI_Allreduce(&t_split_setup,&dst,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  t_split_setup = dst/((double)nproc);
  MPI_Allreduce(&t_barrier2_setup,&dst,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  t_barrier2_setup = dst/((double)nproc);
  if (me == 0) {
    printf("Times from MPI calls in setup:\n");
    printf("    Time in first barrier:            %16.6f\n",t_barrier_setup);
    printf("    Time in all gather:               %16.6f\n",t_allgather_setup);
    printf("    Time in split:                    %16.6f\n",t_split_setup);
    printf("    Time in second barrier:           %16.6f\n",t_barrier2_setup);
  }

  if (worker) {
    double t_allgather_self = 0.0;
    double t_allgather_buf = 0.0;
    int i, me, nproc;
    double *send, *recv;

    MPI_Comm_rank(my_comm,&me);
    MPI_Comm_size(my_comm,&nproc);

    send = (double*)malloc(nproc*sizeof(double));
    recv = (double*)malloc(nproc*sizeof(double));

    if (me == 0) {
      printf("Testing Allgather on working processors %d ranks\n",nproc);
    }
    for (i=0; i<NLOOP; i++) {
      tbeg = MPI_Wtime();
      MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
          send,sizeof(double),MPI_BYTE,my_comm);
      t_allgather_self += MPI_Wtime()-tbeg;
      tbeg = MPI_Wtime();
      MPI_Allgather(send,sizeof(double),MPI_BYTE,recv,sizeof(double),
          MPI_BYTE,my_comm);
      t_allgather_buf += MPI_Wtime()-tbeg;
    }
    MPI_Allreduce(&t_allgather_self,&dst,1,MPI_DOUBLE,MPI_SUM,my_comm);
    t_allgather_self = dst/((double)(NLOOP*nproc));
    MPI_Allreduce(&t_allgather_buf,&dst,1,MPI_DOUBLE,MPI_SUM,my_comm);
    t_allgather_buf = dst/((double)(NLOOP*nproc));
    if (me == 0) {
      printf("\nTotal time in Allgather in place:     %16.6f\n",t_allgather_self);
      printf("Total time in Allgather using buffer: %16.6f\n",t_allgather_buf);
    }
    free(send);
    free(recv);
  }
  ierr = MPI_Barrier(MPI_COMM_WORLD);
  {
    double t_allgather_self = 0.0;
    double t_allgather_buf = 0.0;
    int i, me, nproc;
    double *send, *recv;

    MPI_Comm_rank(MPI_COMM_WORLD,&me);
    MPI_Comm_size(MPI_COMM_WORLD,&nproc);

    send = (double*)malloc(nproc*sizeof(double));
    recv = (double*)malloc(nproc*sizeof(double));

    if (me == 0) {
      printf("Testing Allgather on all processors %d ranks\n",nproc);
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
      printf("\nTotal time in Allgather in place:     %16.6f\n",t_allgather_self);
      printf("Total time in Allgather using buffer: %16.6f\n",t_allgather_buf);
    }
    free(send);
    free(recv);
  }
  t_total = MPI_Wtime()-tbeg0;
  MPI_Allreduce(&t_total,&dst,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  t_total = dst/((double)(nproc));
  if (me == 0) {
    printf("Total time in test program:               %16.6f\n",t_total);
  }
  ierr = MPI_Finalize();
}
