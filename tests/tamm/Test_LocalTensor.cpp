#include <chrono>
#include <tamm/tamm.hpp>
#include <tamm/tamm_git.hpp>

using namespace tamm;

template <typename T>
void test_local_tensor(Scheduler& sch, size_t N, Tile tilesize) {

  TiledIndexSpace tis1{IndexSpace{range(N)}, tilesize};

  auto [i, j, k, l, m] = tis1.labels<5>("all");

  Tensor<T> A{i, j, k};
  Tensor<T> B{k, l};
  Tensor<T> C{i, j, l};

  sch.allocate(A,B,C)
  (A() = 1.0) 
  (B() = 2.0)
  (C() = 3.0)
  .execute();

  LocalTensor A_local{A};
  LocalTensor B_local{B};
  LocalTensor C_local{C};

  std::cout << "A_local" << std::endl;
  for (size_t i_idx = 0; i_idx < N; i_idx++) {
    for (size_t j_idx = 0; j_idx < N; j_idx++) {
      for (size_t k_idx = 0; k_idx < N; k_idx++) {
        std::cout << A_local(i_idx, j_idx, k_idx) << "\t";
        A_local(i_idx, j_idx, k_idx) = 42.0;
      }
    }
    std::cout << std::endl;
  }
  A_local.write_back_to_dist();

  print_tensor(A);
}


int main(int argc, char* argv[]) {
  tamm::initialize(argc, argv);

  if(argc < 3) { tamm_terminate("Please provide an index space size and tile size"); }

  size_t is_size   = atoi(argv[1]);
  Tile   tile_size = atoi(argv[2]);

  if(is_size < tile_size) tile_size = is_size;

  ProcGroup        pg = ProcGroup::create_world_coll();
  ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};

  ExecutionHW ex_hw = ec.exhw();

  Scheduler sch{ec};

  if(ec.print()) {
    std::cout << tamm_git_info() << std::endl;
    auto current_time   = std::chrono::system_clock::now();
    auto current_time_t = std::chrono::system_clock::to_time_t(current_time);
    auto cur_local_time = localtime(&current_time_t);
    std::cout << std::endl << "date: " << std::put_time(cur_local_time, "%c") << std::endl;

    std::cout << "nnodes: " << ec.nnodes() << ", ";
    std::cout << "nproc: " << ec.nnodes() * ec.ppn() << std::endl;
    std::cout << "dim, tile sizes = " << is_size << ", " << tile_size << std::endl;
    ec.print_mem_info();
    std::cout << std::endl << std::endl;
  }

  test_local_tensor<double>(sch, is_size, tile_size);

  tamm::finalize();

  return 0;
}