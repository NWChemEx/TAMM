#include "ga/ga.h"
#include <chrono>
#include <tamm/tamm.hpp>

using namespace tamm;

void test_sparse_tensor() {
  TiledIndexSpace AOs{IndexSpace{range(7)}};
  TiledIndexSpace MOs{IndexSpace{range(4)}};

  /// Dependency map (i.e., sparse maps)
  std::map<IndexVector, TiledIndexSpace> dep_nu_mu_q{
    {{{0}, TiledIndexSpace{AOs, IndexVector{0, 3, 4}}},
     {{1}, TiledIndexSpace{AOs, IndexVector{0, 3, 6}}},
     {{2}, TiledIndexSpace{AOs, IndexVector{1, 3, 5}}}}};

  TiledIndexSpace tSubAO_AO_Q{AOs, {MOs}, dep_nu_mu_q};

  auto [i, j] = MOs.labels<2>();
  auto [a, b] = tSubAO_AO_Q.labels<2>();

  Tensor<double> T{j, i, a(i)};

  /// Example for generating COO coordinates
  auto coo_coordinates = T.base_ptr()->construct_COO_coordinates();
  std::cout << "Printing COO coordinates"
            << "\n";
  std::cout << "j\ti\ta(i)"
            << "\n";
  for(const auto& cord: coo_coordinates) {
    for(const auto& idx: cord) { std::cout << idx << "\t"; }
    std::cout << "\n";
  }
  std::cout << "-----------------"
            << "\n";

  ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());

  /// @todo Fix compile errors before uncommenting these lines!

  // MemoryManagerSparseLocal*    mgr = MemoryManagerSparseLocal::create_coll(pg);
  // Distribution_Sparse distribution;
  // ExecutionContext    ec{pg, &distribution, mgr, &re};
  // Scheduler           sch{ec};

  // sch.allocate(T).execute();

  // sch
  // (T(j, i, a(i)) = 1.0)
  // (T(j, i, a(i)) = 2.0 * T2(j, i, a(i)))
  // .execute();
}

int main(int argc, char* argv[]) {
  tamm::initialize(argc, argv);

  test_sparse_tensor();

  tamm::finalize();

  return 0;
}