#include <vector>
#include <cassert>
#include <algorithm>
#include <iostream>
#include <string>

#undef NDEBUG

#include "tammx/tammx.h"


using namespace std;
using namespace tammx;

Irrep operator "" _ir(unsigned long long int val) {
  return Irrep{strongint_cast<Irrep::value_type>(val)};
}

Spin operator "" _sp(unsigned long long int val) {
  return Spin{strongint_cast<Spin::value_type>(val)};
}

BlockDim operator "" _bd(unsigned long long int val) {
  return BlockDim{strongint_cast<BlockDim::value_type>(val)};
}

std::vector<Spin> spins = {1_sp, -1 * 1_sp, 1_sp, -1 * 1_sp};
std::vector<Irrep> spatials = {0_ir, 0_ir, 0_ir, 0_ir};
std::vector<size_t> sizes = {4, 4, 5, 5};
BlockDim noa {1};
BlockDim noab {2};
BlockDim nva {1};
BlockDim nvab {2};
bool spin_restricted = false;
Irrep irrep_f {0};
Irrep irrep_v {0};
Irrep irrep_t {0};
Irrep irrep_x {0};
Irrep irrep_y {0};

void test() {
  using Type = Tensor::Type;
  using Distribution = Tensor::Distribution;

  TensorVec<SymmGroup> indices{SymmGroup{DimType::o, DimType::o}, SymmGroup{DimType::n}};
  Tensor ta{indices, Type::double_precision, Distribution::tce_nwma, 2, irrep_t, false};

  assert(TCE::noab() == 2);
  assert(std::get<0>(tensor_index_range(DimType::o)) == 0);
  assert(std::get<1>(tensor_index_range(DimType::o)) == 2);
  assert(std::get<0>(tensor_index_range(DimType::n)) == 0);
  assert(std::get<1>(tensor_index_range(DimType::n)) == 4);
  assert(ta.rank() == 3);
  assert(ta.element_type() == Type::double_precision);
  assert(ta.element_size() == sizeof(double));
  assert(ta.distribution() == Tensor::Distribution::tce_nwma);
  auto flindices = ta.flindices();
  auto expected = TensorVec<DimType>{DimType::o, DimType::o, DimType::n};
  assert(std::equal(flindices.begin(), flindices.end(), expected.begin()));
  
  ta.allocate();
  assert(ta.constructed() && ta.allocated() && !ta.attached());

  cout<<"num_blocks = "<<ta.num_blocks()<<endl;
  assert(ta.num_blocks() == TensorIndex({2_bd, 2_bd, 4_bd}));

  assert(ta.block_dims({1_bd, 0_bd, 3_bd}) == TensorIndex({4_bd, 4_bd, 5_bd}));
  
  assert(ta.block_size({0_bd, 0_bd, 0_bd}) == 64);
  assert(ta.block_size({0_bd, 1_bd, 1_bd}) == 64);
  assert(ta.block_size({0_bd, 0_bd, 2_bd}) == 80);
  assert(ta.block_size({0_bd, 1_bd, 3_bd}) == 80);
  assert(ta.block_size({1_bd, 0_bd, 0_bd}) == 64);
  assert(ta.block_size({1_bd, 1_bd, 1_bd}) == 64);
  assert(ta.block_size({1_bd, 0_bd, 2_bd}) == 80);
  assert(ta.block_size({1_bd, 1_bd, 3_bd}) == 80);

  assert(ta.nonzero({0_bd, 0_bd, 0_bd}));
  assert(ta.nonzero({0_bd, 0_bd, 2_bd}));
  assert(ta.nonzero({0_bd, 1_bd, 1_bd}));
  assert(ta.nonzero({1_bd, 0_bd, 1_bd}));
  assert(ta.nonzero({0_bd, 1_bd, 3_bd}));
  assert(ta.nonzero({1_bd, 0_bd, 3_bd}));

  assert(!ta.nonzero({0_bd, 0_bd, 1_bd}));
  assert(!ta.nonzero({0_bd, 0_bd, 3_bd}));
  assert(!ta.nonzero({0_bd, 1_bd, 0_bd}));
  assert(!ta.nonzero({0_bd, 1_bd, 2_bd}));
  assert(!ta.nonzero({1_bd, 0_bd, 0_bd}));
  assert(!ta.nonzero({1_bd, 0_bd, 2_bd}));
  assert(!ta.nonzero({1_bd, 1_bd, 0_bd}));
  assert(!ta.nonzero({1_bd, 1_bd, 1_bd}));
  assert(!ta.nonzero({1_bd, 1_bd, 2_bd}));
  assert(!ta.nonzero({1_bd, 1_bd, 3_bd}));

  assert(ta.find_unique_block({0_bd, 0_bd, 0_bd}) == TensorIndex({0_bd, 0_bd, 0_bd}));
  assert(ta.find_unique_block({0_bd, 1_bd, 0_bd}) == TensorIndex({0_bd, 1_bd, 0_bd}));
  assert(ta.find_unique_block({1_bd, 0_bd, 2_bd}) == TensorIndex({0_bd, 1_bd, 2_bd}));
  assert(ta.find_unique_block({0_bd, 1_bd, 3_bd}) == TensorIndex({0_bd, 1_bd, 3_bd}));

  // tensor_map(ta(), [] (Block& block) {
  //     std::fill_n(reinterpret_cast<double*>(block.buf()), block.size(), 2.0);
  //   });

  // auto block = ta.get({0_bd, 0_bd, 2_bd});
  // for(unsigned i=0; i<block.size(); i++) {
  //   assert(reinterpret_cast<double*>(block.buf())[i] == 2.0);
  // }
  // for(unsigned i=0; i<block.size(); i++) {
  //   reinterpret_cast<double*>(block.buf())[i] += 4.0;
  // }
  // ta.add(block);
  // auto block2 = ta.get({0_bd, 0_bd, 2_bd});
  // for(unsigned i=0; i<block.size(); i++) {
  //   assert(reinterpret_cast<double*>(block.buf())[i] == 6.0);
  // }

  TensorVec<SymmGroup> indicesb{SymmGroup{DimType::o}, SymmGroup{DimType::o}, SymmGroup{DimType::n}};
  //TensorVec<SymmGroup> indicesb{SymmGroup{DimType::o, DimType::o}, SymmGroup{DimType::n}};
  Tensor tb{indicesb, Type::double_precision, Distribution::tce_nwma, 2, irrep_t, false};
  tb.allocate();

  assert_zero(ta());
    
  tensor_map(ta(), [] (Block& block) {
      std::fill_n(reinterpret_cast<double*>(block.buf()), block.size(), 1.0);
    });
  // ta() += tb();
  // tb() += ta();
  // tensor_map(tb(), [] (Block& block) {
  //     std::fill_n(reinterpret_cast<double*>(block.buf()), block.size(), 1.0);
  //   });
  // ta() += -1.0 * tb();
  // assert_zero(ta());

  Tensor ta2{indices, Type::double_precision, Distribution::tce_nwma, 2, irrep_t, false};
  ta2.allocate();
  tensor_map(ta2(), [] (Block& block) {
      std::fill_n(reinterpret_cast<double*>(block.buf()), block.size(), 1.0);
    });

  std::cerr<<"----------------------------------"<<std::endl;
  ta() += -1.0 * ta2();
  std::cerr<<"----------------------------------"<<std::endl;
  assert_zero(ta());
  
  ta2.destruct();
  ta.destruct();
  assert(!ta.constructed() && !ta.allocated() && !ta.attached());
  tb.destruct();
}

int main() {
  TCE::init(spins, spatials, sizes,
            noa,
            noab,
            nva,
            nvab,
            spin_restricted,
            irrep_f,
            irrep_v,
            irrep_t,
            irrep_x,
            irrep_y);

  test();
  
  TCE::finalize();
  return 0;
}

