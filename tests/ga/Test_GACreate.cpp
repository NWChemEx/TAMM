
#include "ga/ga-mpi.h"
#include "ga/ga.h"
#include "mpi.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

int main(int argc, char* argv[]) {
  if(argc < 2) {
    std::cout << "Please provide a dimension size!\n";
    return 0;
  }

  int64_t dim1      = atoi(argv[1]);
  int64_t dim2      = 0;
  int64_t tile_size = 0;

  if(argc == 3) dim2 = atoi(argv[2]);
  if(argc == 4) tile_size = atoi(argv[3]);

  MPI_Init(&argc, &argv);
  GA_Initialize();

  int mpi_rank;
  MPI_Comm_rank(GA_MPI_Comm(), &mpi_rank);

  int64_t dim = dim1 * dim1 * dim1 * dim1;
  if(dim2 > 0) dim = dim1 * dim1 * dim2 * dim2;

  if(mpi_rank == 0) std::cout << "GA size = " << dim * 8.0 / (1024.0 * 1024.0 * 1024.0) << "GiB\n";

  const auto  timer_start = std::chrono::high_resolution_clock::now();
  int         g_a         = 0;
  std::string array_name  = "A";
  if(tile_size == 0)
    g_a = NGA_Create64(C_DBL, 1, &dim, const_cast<char*>(array_name.c_str()), nullptr);
  else g_a = NGA_Create64(C_DBL, 1, &dim, const_cast<char*>(array_name.c_str()), &tile_size);
  const auto timer_end = std::chrono::high_resolution_clock::now();
  auto       ga_ct =
    std::chrono::duration_cast<std::chrono::duration<double>>((timer_end - timer_start)).count();

  std::string error_msg = "GA create failed: A";
  if(!g_a) GA_Error(const_cast<char*>(error_msg.c_str()), dim1 * dim2);
  if(mpi_rank == 0) printf("GA create successful\n");

  if(mpi_rank == 0) std::cout << "GA create time = " << ga_ct << "s" << std::endl;
  NGA_Destroy(g_a);

  GA_Terminate();
  MPI_Finalize();

  return 0;
}
