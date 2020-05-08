#include "macdecls.h"
#include "mpi.h"
#include <chrono>
#include <tamm/tamm.hpp>

using namespace tamm;

using DependencyMap = std::map<IndexVector, TiledIndexSpace>;

DependencyMap construct_full_dependency_2(const TiledIndexSpace &in_tis1,
                                          const TiledIndexSpace &in_tis2,
                                          const TiledIndexSpace &out_tis) {
  DependencyMap result;
  for (const auto &tile_1 : in_tis1) {
    for (const auto &tile_2 : in_tis2) {
      result.insert({{tile_1, tile_2}, out_tis});
    }
  }
  return result;
}

template <typename T>
void dlpno_T1_T2_allocate(Scheduler &sch, size_t N, Tile tilesize) {

  TiledIndexSpace LMO{IndexSpace{range(N)}, tilesize};
  DependencyMap depMO = construct_full_dependency_2(LMO, LMO, LMO);
  TiledIndexSpace PNO{LMO, {LMO, LMO}, depMO};

  auto [i, j] = LMO.labels<2>("all");
  auto [a, e] = PNO.labels<2>("all");

  Tensor<T> t1{i, e(i, i)};
  Tensor<T> t2{i, j, a(i, j), e(i, j)};

  sch.allocate(t1, t2).execute();
}

int main(int argc, char *argv[]) {

  tamm::initialize(argc, argv);

  int mpi_rank;
  MPI_Comm_rank(GA_MPI_Comm(), &mpi_rank);
#ifdef USE_TALSH
  TALSH talsh_instance;
  talsh_instance.initialize(mpi_rank);
#endif

  ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
  MemoryManagerGA *mgr = MemoryManagerGA::create_coll(pg);
  Distribution_NW distribution;
  RuntimeEngine re;
  ExecutionContext ec{pg, &distribution, mgr, &re};

  Scheduler sch{ec};

  dlpno_T1_T2_allocate<double>(sch, 10, 2);

#ifdef USE_TALSH
  talsh_instance.shutdown();
#endif

  tamm::finalize();

  return 0;
}
