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
  return Irrep{strongnum_cast<Irrep::value_type>(val)};
}

Spin operator "" _sp(unsigned long long int val) {
  return Spin{strongnum_cast<Spin::value_type>(val)};
}

BlockDim operator "" _bd(unsigned long long int val) {
  return BlockDim{strongnum_cast<BlockDim::value_type>(val)};
}

std::vector<Spin> spins = {1_sp, 2_sp, 1_sp, 2_sp};
std::vector<Irrep> spatials = {0_ir, 0_ir, 0_ir, 0_ir};
std::vector<size_t> sizes = {2, 4, 2, 1};
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

struct OLabel : public IndexLabel {
  OLabel(int n)
      : IndexLabel{n, DimType::o} {}
};

struct VLabel : public IndexLabel {
  VLabel(int n)
      : IndexLabel{n, DimType::v} {}
};

void test() {
  using Type = double;
  using DistributionType = Distribution_NW;

  using tammx::O;
  using tammx::V;
  using tammx::OO;
  using tammx::OV;

  Irrep irrep{0};

  auto iinfo1 = OO|V;
  auto iinfo3 = O|O;

  Tensor<Type> ta {iinfo1, irrep, false};
  auto distribution = DistributionType();
  auto mgr = MemoryManagerSequential();
  ta.alloc(ProcGroup{}, &distribution, &mgr);

  {
    Scheduler sch(ProcGroup{}, &distribution, &mgr, Irrep{0}, false);
    
    auto &tb = sch.tensor<Type>(iinfo1);
    
    sch.io(ta)
        .execute();
  }
  ta.dealloc();
  
#if 0
  TensorVec<SymmGroup> indices1{SymmGroup{DimType::o, DimType::o}, SymmGroup{DimType::v}};
  TensorVec<SymmGroup> indices2{SymmGroup{DimType::o}, SymmGroup{DimType::o}, SymmGroup{DimType::v}};
  TensorVec<SymmGroup> indices3{SymmGroup{DimType::o}, SymmGroup{DimType::o}};

  Tensor ta{indices1, Type::double_precision, Distribution::tce_nwma, 3, irrep_t, false};
  Tensor ta2{indices3, Type::double_precision, Distribution::tce_nwma, 2, irrep_t, false};
  Tensor tb{indices2, Type::double_precision, Distribution::tce_nwma, 3, irrep_t, false};

  // TensorVec<SymmGroup> indices1{SymmGroup{DimType::o, DimType::o}};
  // TensorVec<SymmGroup> indices2{SymmGroup{DimType::o}, SymmGroup{DimType::o}};

  // Tensor ta{indices1, Type::double_precision, Distribution::tce_nwma, 2, irrep_t, false};
  // Tensor tb{indices2, Type::double_precision, Distribution::tce_nwma, 2 irrep_t, false};

  ta.allocate();
  tb.allocate();
  ta2.allocate();

  tensor_map(tb(), [] (Block& block) {
      //std::fill_n(reinterpret_cast<double*>(block.buf()), block.size(), 1.0);
      int n=0;
      //std::generate_n(reinterpret_cast<double*>(block.buf()), block.size(), std::rand);
      std::generate_n(reinterpret_cast<double*>(block.buf()), block.size(), [&]() { return n++;});
    });

  OLabel h0{0}, h1{1}, h2{2}, h3{3};
  VLabel p0{0}, p1{1}, p2{2}, p3{3};
  
  //tensor_init(tb, 1.0);
  //assert_equal(tb, 1.0);
  ta() += 1.0 * tb();
  //assert_equal(ta, 0);

  ta({h1,h2,p1}) += 1.0 * tb({h1,h2,p1});
  
  ta({h0,h1,p2}) += 1.0 * tb({h0,h3,p2}) * ta2({h1,h3});
  
  // assert_equal(tb, 1.0);

  std::cerr<<"------------------"<<std::endl;
  tensor_print(tb, std::cerr);
  std::cerr<<"------------------"<<std::endl;
  
  std::cerr<<"------------------"<<std::endl;
  tensor_print(ta, std::cerr);
  std::cerr<<"------------------"<<std::endl;

  ta.destruct();
  tb.destruct();
  ta2.destruct();
  
#elif 0
  
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

  Tensor ta2{indices, Type::double_precision, Distribution::tce_nwma, 2, irrep_t, false};
  ta2.allocate();
  
  assert_zero(ta);
  assert_zero(tb);
  assert_zero(ta2);
  
  //tensor_init(ta, 1.0);
  // std::cerr<<"------------------"<<std::endl;
  // tensor_print(ta, std::cerr);
  // std::cerr<<"------------------"<<std::endl;
  // tb() += ta();
  // tb() += -1 * ta();
  // assert_zero(tb);
  // tb() += ta();
  // tb({0,1,2}) += ta({1,0,2});
  // assert_zero(tb());

  // tensor_map(ta2(), [] (Block& block) {
  //     std::fill_n(reinterpret_cast<double*>(block.buf()), block.size(), 1.0);
  //   });

  tensor_map(tb(), [] (Block& block) {
      //std::fill_n(reinterpret_cast<double*>(block.buf()), block.size(), 1.0);
      int n=0;
      //std::generate_n(reinterpret_cast<double*>(block.buf()), block.size(), std::rand);
      std::generate_n(reinterpret_cast<double*>(block.buf()), block.size(), [&]() { return n++;});
    });

  //tensor_init(tb, 1.0);
  //assert_equal(tb, 1.0);
  ta() += 1.0 * tb();
  //assert_equal(ta, 0);

  // assert_equal(tb, 1.0);

  std::cerr<<"------------------"<<std::endl;
  tensor_print(tb, std::cerr);
  std::cerr<<"------------------"<<std::endl;

  // //assert_equal(ta2, 1.0);
  ta() += tb();

  
  std::cerr<<"------------------"<<std::endl;
  tensor_print(ta, std::cerr);
  std::cerr<<"------------------"<<std::endl;
  
  // assert_equal(ta, 0.0);

  
  // ta() += tb();
  // tb() += ta();
  // tensor_map(tb(), [] (Block& block) {
  //     std::fill_n(reinterpret_cast<double*>(block.buf()), block.size(), 1.0);
  //   });
  // ta() += -1.0 * tb();
  // assert_zero(ta());

  // ta2.allocate();
  // tensor_map(ta2(), [] (Block& block) {
  //     std::fill_n(reinterpret_cast<double*>(block.buf()), block.size(), 1.0);
  //   });

  // std::cerr<<"----------------------------------"<<std::endl;
  // ta() += -1.0 * ta2();
  // std::cerr<<"----------------------------------"<<std::endl;
  // assert_zero(ta());
  
  ta2.destruct();
  ta.destruct();
  assert(!ta.constructed() && !ta.allocated() && !ta.attached());
  tb.destruct();
#endif
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

