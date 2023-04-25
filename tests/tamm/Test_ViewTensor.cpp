#include "ga/ga.h"
#include <chrono>
#include <tamm/tamm.hpp>

using namespace tamm;

void test_view_tensor(size_t size, size_t tile_size) {
  IndexSpace IS{range(size),
                {{"occ", {range(0, size / 2)}},
                 {"virt", {range(size / 2, size)}},
                 {"alpha", {range(0, size / 4), range(size / 2, 3 * size / 4)}},
                 {"beta", {range(size / 4, size / 2), range(3 * size / 4, size)}}}};

  TiledIndexSpace TIS{IS, static_cast<Tile>(tile_size)};

  auto [i, j, k]       = TIS.labels<3>("all");
  auto [i_o, j_o, k_o] = TIS.labels<3>("occ");
  auto [i_v, j_v, k_v] = TIS.labels<3>("virt");
  auto [i_a, j_a, k_a] = TIS.labels<3>("alpha");
  auto [i_b, j_b, k_b] = TIS.labels<3>("beta");

  internal::LabelTranslator occ_translator{{i_o, j_o, k_o}, {i, j, k}};
  internal::LabelTranslator virt_translator{{i_v, j_v, k_v}, {i, j, k}};
  internal::LabelTranslator alpha_translator{{i_a, j_a, k_a}, {i, j, k}};
  internal::LabelTranslator beta_translator{{i_b, j_b, k_b}, {i, j, k}};

  internal::LabelTranslator mixed_translator{{i_o, i_b, i_v}, {i, j, k}};

  auto occ_2_all = [&](const IndexVector& idx) -> IndexVector {
    auto [translated_id, is_valid] = occ_translator.apply(idx);
    return translated_id;
  };

  auto virt_2_all = [&](const IndexVector& idx) -> IndexVector {
    auto [translated_id, is_valid] = virt_translator.apply(idx);
    return translated_id;
  };

  auto alpha_2_all = [&](const IndexVector& idx) -> IndexVector {
    auto [translated_id, is_valid] = alpha_translator.apply(idx);
    return translated_id;
  };

  auto beta_2_all = [&](const IndexVector& idx) -> IndexVector {
    auto [translated_id, is_valid] = beta_translator.apply(idx);
    return translated_id;
  };

  auto mixed_2_all = [&](const IndexVector& idx) -> IndexVector {
    auto [translated_id, is_valid] = mixed_translator.apply(idx);
    return translated_id;
  };

  ProcGroup        pg  = ProcGroup::create_world_coll();
  MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
  Distribution_NW  distribution;
  RuntimeEngine    re;
  ExecutionContext ec{pg, &distribution, mgr, &re};
  Scheduler        sch{ec};

  Tensor<double> T_full{i, j, k};

  T_full.allocate(&ec);

  // ViewTensor should be created after the reference tensor is allocated as it
  // is using the distribution information from that tensor
  Tensor<double> T_occ{T_full, {i_o, j_o, k_o}, occ_2_all};
  Tensor<double> T_virt{T_full, {i_v, j_v, k_v}, virt_2_all};
  Tensor<double> T_alpha{T_full, {i_a, j_a, k_a}, alpha_2_all};
  Tensor<double> T_beta{T_full, {i_b, j_b, k_b}, beta_2_all};
  Tensor<double> T_mixed{T_full, {i_o, j_b, k_v}, mixed_2_all};
  Tensor<double> F_occ{i_o, j_o, k_o};

  F_occ.allocate(&ec);

  sch(T_full() = 1.0)(F_occ() = 2.0).execute();
  //   std::cerr << __FUNCTION__ << " " << __LINE__ << "\n";
  //   std::cout << "Print tensor T_full" << "\n";
  //   print_tensor(T_full);

  sch(T_occ() = 2.0).execute();

  //   std::cerr << __FUNCTION__ << " " << __LINE__ << "\n";
  //   std::cout << "Print tensor T_full" << "\n";
  //   print_tensor(T_full);

  sch(T_occ(i_o, j_o, k_o) += F_occ(i_o, j_o, k_o)).execute();

  //   std::cerr << __FUNCTION__ << " " << __LINE__ << "\n";
  //   std::cout << "Print tensor T_full" << "\n";
  //   print_tensor(T_full);

  sch(T_virt() = 3.0).execute();

  //   std::cerr << __FUNCTION__ << " " << __LINE__ << "\n";
  //   std::cout << "Print tensor T_full" << "\n";
  //   print_tensor(T_full);

  sch(T_alpha() = 4.0).execute();

  //   std::cerr << __FUNCTION__ << " " << __LINE__ << "\n";
  //   std::cout << "Print tensor T_full" << "\n";
  //   print_tensor(T_full);

  sch(T_beta() = 5.0).execute();

  //   std::cerr << __FUNCTION__ << " " << __LINE__ << "\n";
  //   std::cout << "Print tensor T_full" << "\n";
  //   print_tensor(T_full);

  sch(T_mixed() = 5.0).execute();

  std::cerr << __FUNCTION__ << " " << __LINE__ << "\n";
  std::cout << "Print tensor T_full"
            << "\n";
  print_tensor(T_full);
}

int main(int argc, char* argv[]) {
  tamm::initialize(argc, argv);

  test_view_tensor(10, 2);

  tamm::finalize();

  return 0;
}