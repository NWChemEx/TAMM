#include "ga/ga-mpi.h"
#include "ga/ga.h"

#include <cstdlib>
#include <numeric>
#include <sched.h>
#include <stdio.h>
#include <unistd.h>
#include <vector>

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  char _hostname[MPI_MAX_PROCESSOR_NAME];
  int  resultlength;
  MPI_Get_processor_name(_hostname, &resultlength);

  std::vector<int> allranks(size);
  std::iota(std::begin(allranks), std::end(allranks), 0);

  printf("node %s, pid %d/%d , core %d\n", _hostname, rank, size, sched_getcpu());
  fflush(stdout);
  sleep(2);

  MPI_Barrier(MPI_COMM_WORLD);
  GA_Initialize();

  printf("node %s, pid %d/%d , core %d, GA_Nodeid %d\n", _hostname, rank, size, sched_getcpu(),
         GA_Nodeid());
  fflush(stdout);
  sleep(2);

  GA_Terminate();
  MPI_Finalize();

  return 0;
}
