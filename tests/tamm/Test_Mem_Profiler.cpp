#include "ga/macdecls.h"
#include "mpi.h"
#include <chrono>
#include <tamm/tamm.hpp>

using namespace tamm;

using DependencyMap = std::map<IndexVector, TiledIndexSpace>;

void test_tensor_allocate(Scheduler& sch) {
  TiledIndexSpace tis{IndexSpace{range(10)}, 10};
  TiledIndexSpace tis2{IndexSpace{range(20)}, 20};
  TiledIndexSpace tis3{IndexSpace{range(100)}, 20};

  Tensor<double> A{tis, tis, tis, tis};
  Tensor<double> B{tis, tis, tis, tis};
  Tensor<double> C{tis, tis, tis, tis2};
  Tensor<double> D{tis, tis, tis, tis2};
  Tensor<double> E{tis, tis, tis, tis3};

  sch.allocate(A, B, C).deallocate(B, C).execute();

  sch.allocate(D, E).execute();

  print_memory_usage<double>(sch.ec().pg().rank().value());
}

int main(int argc, char* argv[]) {
  tamm::initialize(argc, argv);

  ProcGroup        pg = ProcGroup::create_world_coll();
  ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};

  Scheduler sch{ec};

  test_tensor_allocate(sch);

  tamm::finalize();

  return 0;
}
