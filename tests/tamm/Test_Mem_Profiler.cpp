#include "ga/macdecls.h"
#include "mpi.h"
#include <chrono>
#include <tamm/tamm.hpp>

using namespace tamm;

using DependencyMap = std::map<IndexVector, TiledIndexSpace>;

template <typename T>
std::string mem_to_string(double mem_size) {
  return std::to_string((mem_size * sizeof(T)) / std::pow(2, 30)) + " GBs";
}

template<typename T>
void report_mem_usage(ExecutionContext& ec) {
  auto& memprof = tamm::MemProfiler::instance();

  if(ec.pg().rank() == 0) {
    std::cout << "alloc_counter : " << memprof.alloc_counter << "\n";
    std::cout << "dealloc_counter : " << memprof.dealloc_counter << "\n";
    std::cout << "mem_allocated : " << mem_to_string<T>(memprof.mem_allocated) << "\n";
    std::cout << "mem_deallocated : " << mem_to_string<T>(memprof.mem_deallocated) << "\n";
    std::cout << "max_in_single_allocate : " << mem_to_string<T>(memprof.max_in_single_allocate) << "\n";
    std::cout << "max_total_allocated : " << mem_to_string<T>(memprof.max_total_allocated)
              << "\n";
  }
}

void test_tensor_allocate(Scheduler& sch) {
  TiledIndexSpace tis{IndexSpace{range(100)}, 40};
  TiledIndexSpace tis2{IndexSpace{range(200)}, 40};
  TiledIndexSpace tis3{IndexSpace{range(400)}, 40};

  Tensor<double> A{tis, tis, tis, tis};
  Tensor<double> B{tis, tis, tis, tis};
  Tensor<double> C{tis, tis, tis, tis2};
  Tensor<double> D{tis, tis, tis, tis2};
  Tensor<double> E{tis, tis, tis, tis3};

  sch.allocate(A, B, C).deallocate(B, C).execute();

  sch.allocate(D, E).execute();

  report_mem_usage<double>(sch.ec());
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
