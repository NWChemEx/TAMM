#include "ga/ga.h"
#include "tamm/tamm.hpp"
#include <chrono>

using namespace tamm;

// bool eigen_tensors_are_equal(Tensor1D& e1, Tensor1D& e2, double threshold = 1.0e-12) {
//   bool ret  = true;
//   auto dims = e1.dimensions();
//   for(auto i = 0; i < dims[0]; i++) {
//     if(std::abs(e1(i) - e2(i)) > std::abs(threshold * e1(i))) {
//       ret = false;
//       break;
//     }
//   }
//   return ret;
// }

// bool eigen_tensors_are_equal(Tensor2D& e1, Tensor2D& e2, double threshold = 1.0e-12) {
//   bool ret  = true;
//   auto dims = e1.dimensions();
//   for(auto i = 0; i < dims[0]; i++) {
//     for(auto j = 0; j < dims[1]; j++) {
//       if(std::abs(e1(i, j) - e2(i, j)) > std::abs(threshold * e1(i, j))) {
//         ret = false;
//         break;
//       }
//     }
//   }
//   return ret;
// }

// bool eigen_tensors_are_equal(Tensor3D& e1, Tensor3D& e2, double threshold = 1.0e-12) {
//   bool ret  = true;
//   auto dims = e1.dimensions();
//   for(auto i = 0; i < dims[0]; i++) {
//     for(auto j = 0; j < dims[1]; j++) {
//       for(auto k = 0; k < dims[2]; k++) {
//         if(std::abs(e1(i, j, k) - e2(i, j, k)) > std::abs(threshold * e1(i, j, k))) {
//           ret = false;
//           break;
//         }
//       }
//     }
//   }
//   return ret;
// }

// bool eigen_tensors_are_equal(Tensor4D& e1, Tensor4D& e2, double threshold = 1.0e-12) {
//   bool ret  = true;
//   auto dims = e1.dimensions();
//   for(auto i = 0; i < dims[0]; i++) {
//     for(auto j = 0; j < dims[1]; j++) {
//       for(auto k = 0; k < dims[2]; k++) {
//         for(auto l = 0; l < dims[3]; l++) {
//           if(std::abs(e1(i, j, k, l) - e2(i, j, k, l)) > std::abs(threshold * e1(i, j, k, l))) {
//             ret = false;
//             break;
//           }
//         }
//       }
//     }
//   }
//   return ret;
// }

template<typename T>
bool check_value(LabeledTensor<T> lt, T val) {
  LabelLoopNest loop_nest{lt.labels()};

  for(const auto& itval: loop_nest) {
    const IndexVector blockid = internal::translate_blockid(itval, lt);
    size_t            size    = lt.tensor().block_size(blockid);
    std::vector<T>    buf(size);
    lt.tensor().get(blockid, buf);
    for(TAMM_SIZE i = 0; i < size; i++) {
      if(std::abs(buf[i] - val) >= 1.0e-10) { return false; }
    }
  }
  return true;
}

template<typename T>
bool check_value(Tensor<T>& t, T val) {
  return check_value(t(), val);
}

/**
 * @brief Unit test suit for testing `unit tiled view tensor`
 *
 *  - Constructing unit tiled view of another tensor
 *    - Memory region should be the same
 *    - Different distribution
 *    - automatic allocation if other tensor is allocated
 *  - Check the various allocation/deallocation ordering
 *    - Opt Tensor allocated
 *      - w/ allocate on unit tiled tensor
 *      - w/o allocate on unit tiled tensors
 *    - Opt Tensor not allocated
 *      - w/ allocate on unit tiled tensor
 *      - w/o allocate on unit tiled tensor (should throw error when tensor used)
 *    - Deallocate calls?
 *      - Do nothing when deallocate called on unit tiled tensor
 *      - What happens if opt tensor is deallocated?
 *  - Check if all the tensor values are the same with set on opt tensor
 *    - Tamm tensor to eigen tensor for both tensors and compare should be exactly matching
 *  - Check equality on tensor ops on full tensors
 *    - SetOp on opt and unit tiled tensor
 *    - AddOp on opt and unit tiled tensor
 *    - MultOp on opt and unit tiled tensor
 *  - Check correctness on sliced ops
 */

void test_unit_tiled_view_tensor(ExecutionContext& ec, size_t size, size_t tile_size) {
  IndexSpace IS{range(size), {{"occ", {range(0, size / 2)}}, {"virt", {range(size / 2, size)}}}};
  TiledIndexSpace TIS{IS, static_cast<Tile>(tile_size)};
  auto [X, Y, Z] = TIS.labels<3>("all");

  Tensor<double> T_full{X, Y, Z};
  Tensor<double> T_copy{X, Y, Z};

  Scheduler sch{ec};

  T_full.allocate(&ec);
  T_copy.allocate(&ec);

  tamm::random_ip(T_full);

  sch(T_copy(X, Y, Z) = T_full(X, Y, Z)).execute();

  // print_tensor_all(T_full);

  Tensor<double> T_unit_1{T_full, 1};

  // Both memory region should be the same
  EXPECTS(T_full.memory_region() == T_unit_1.memory_region());
  // Different distributions for unit tiled view and normal tensor
  EXPECTS(T_full.distribution() != T_unit_1.distribution());
  // Unit tiled view tensor should already be allocated
  EXPECTS(T_unit_1.is_allocated());

  // Test two allocates doesn't break things
  T_unit_1.allocate(&ec);

  // Set op tests
  sch(T_full("X", "Y", "Z") = 42.0).execute();

  EXPECTS(check_value(T_full, 42.0));
  EXPECTS(check_value(T_unit_1, 42.0));

  sch(T_full("X", "Y", "Z") = T_copy("X", "Y", "Z")).execute();

  sch(T_unit_1("x", "Y", "Z") = 42.0).execute();

  EXPECTS(check_value(T_unit_1, 42.0));
  EXPECTS(check_value(T_full, 42.0));

  sch(T_full("X", "Y", "Z") = T_copy("X", "Y", "Z")).execute();

  // Add op tests

  Tensor<double> A_full{X, Y, Z};

  A_full.allocate(&ec);
  tamm::random_ip(A_full);

  Tensor<double> A_unit_1{A_full, 1};

  sch(T_full("X", "Y", "Z") += A_full("X", "Y", "Z")).execute();

  // Tensor3D T_full_eig = tamm_to_eigen_tensor<double,3>(T_full);

  sch(T_full("X", "Y", "Z") = T_copy("X", "Y", "Z")).execute();

  sch(T_unit_1("x", "Y", "Z") += A_unit_1("x", "Y", "Z")).execute();

  // Tensor3D T_unit_1_eig = tamm_to_eigen_tensor<double,3>(T_unit_1);

  // EXPECTS(eigen_tensors_are_equal(T_full_eig, T_unit_1_eig));

  sch(T_full("X", "Y", "Z") = T_copy("X", "Y", "Z")).execute();

  // Mult Op tests

  Tensor<double> B_full{X, Y};

  B_full.allocate(&ec);
  tamm::random_ip(B_full);

  Tensor<double> B_unit_1{B_full, 1};

  sch(T_full("X", "Y", "Z") = A_full("X", "Y", "V") * B_full("V", "Z")).execute();

  // T_full_eig = tamm_to_eigen_tensor<double, 3>(T_full);

  sch(T_full("X", "Y", "Z") = T_copy("X", "Y", "Z")).execute();

  sch(T_unit_1("x", "Y", "Z") = A_unit_1("x", "Y", "V") * B_full("V", "Z")).execute();

  // T_unit_1_eig = tamm_to_eigen_tensor<double,3>(T_unit_1);

  // EXPECTS(eigen_tensors_are_equal(T_full_eig, T_unit_1_eig));

  // Sliced update

  TiledIndexSpace unit_tis_1{T_unit_1.tiled_index_spaces()[0], range(2, 3)};
  auto            x_2 = unit_tis_1.label();

  sch(T_unit_1(x_2, Y, Z) = 1.0).execute();

  // print_tensor_all(T_unit_1);

  // print_tensor_all(T_full);

  Tensor<double> T_unit_2{T_full, 2};

  TiledIndexSpace unit_tis_2{T_unit_2.tiled_index_spaces()[1], range(2, 3)};
  auto            y_2 = unit_tis_2.label();

  sch(T_unit_2(x_2, y_2, Z) = 2.0).execute();

  // print_tensor_all(T_unit_2);

  // print_tensor_all(T_full);

  std::cout << "\n"
            << std::string(25, '*') << "\nRunning MO space tests!\n"
            << std::string(25, '*') << "\n";

  const IndexSpace MO_IS{range(0, 50), {{"occ", {range(0, 15)}}, {"virt", {range(15, 50)}}}};

  const std::vector<Tile> mo_tiles = {10, 5, 10, 10, 10, 5};
  const TiledIndexSpace   MO{MO_IS, mo_tiles};

  auto [h1, h2, h3] = MO.labels<3>("occ");
  Tensor<double> t1{h1, h2};
  Tensor<double> t2{h1, h2};
  Tensor<double> tmp{h1};

  sch.allocate(t1, t2, tmp).execute();

  Tensor<double>  t1_ut{t1, 1};
  TiledIndexSpace t1_utis{t1_ut.tiled_index_spaces()[0], range(2, 3)};
  auto            t1_ut_l1 = t1_utis.label();

  sch(tmp(h3) = t1_ut(t1_ut_l1, h2) * t2(h2, h3)).execute();

  const std::vector<Tile> ao_tiles = {1, 3};
  TiledIndexSpace         AO{IndexSpace{range(4)}, ao_tiles};

  Tensor<double> T{AO, AO};

  sch.allocate(T).execute();

  random_ip(T);

  // print_tensor(T);

  Tensor<double> T_ut{T, 2};
  Tensor<double> tmp2{};
  sch.allocate(tmp2).execute();

  for(int i = 0; i < 4; i++) {
    for(int j = 0; j < 4; j++) {
      TiledIndexSpace tis1{T_ut.tiled_index_spaces()[0], range(i, i + 1)};
      TiledIndexSpace tis2{T_ut.tiled_index_spaces()[1], range(j, j + 1)};
      auto            l1 = tis1.label();
      auto            l2 = tis2.label();

      sch(tmp2() = T_ut(l1, l2)).execute();
      auto val = get_scalar(tmp2);
      if(ec.pg().rank() == 0) std::cout << i << " " << j << " " << val << std::endl;
    }
  }

  std::cout << "Finished tests!"
            << "\n";
}

int main(int argc, char* argv[]) {
  tamm::initialize(argc, argv);

  ProcGroup        pg  = ProcGroup::create_world_coll();
  MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
  Distribution_NW  distribution;
  RuntimeEngine    re;
  ExecutionContext ec{pg, &distribution, mgr, &re};

  test_unit_tiled_view_tensor(ec, 20, 5);

  tamm::finalize();

  return 0;
}
