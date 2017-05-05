#include <vector>
#include <cassert>
#include <algorithm>
#include <iostream>

#include "tammx/tammx.h"


using namespace std;
using namespace tammx;

std::vector<Spin> spins = {1, -1, 1, -1};
std::vector<Spatial> spatials = {0, 0, 0, 0};
std::vector<size_t> sizes = {4, 4, 5, 5};
BlockDim noa = 1;
BlockDim noab = 2;
BlockDim nva = 1;
BlockDim nvab = 2;
bool spin_restricted = false;
Irrep irrep_f = 0;
Irrep irrep_v = 0;
Irrep irrep_t = 0;
Irrep irrep_x = 0;
Irrep irrep_y = 0;

void test() {
  using Type = Tensor::Type;
  using Distribution = Tensor::Distribution;

  TensorVec<SymmGroup> indices{SymmGroup{DimType::o}, SymmGroup{DimType::n}};
  Tensor ta{indices, Type::double_precision, Distribution::tce_nwma, 2, irrep_t, false};

  assert(TCE::noab() == 2);
  assert(std::get<0>(tensor_index_range(DimType::o)) == 0);
  assert(std::get<1>(tensor_index_range(DimType::o)) == 2);
  assert(std::get<0>(tensor_index_range(DimType::n)) == 0);
  assert(std::get<1>(tensor_index_range(DimType::n)) == 4);
  assert(ta.rank() == 2);
  assert(ta.element_type() == Type::double_precision);
  assert(ta.element_size() == sizeof(double));
  assert(ta.distribution() == Tensor::Distribution::tce_nwma);
  auto flindices = ta.flindices();
  auto expected = TensorVec<DimType>{DimType::o, DimType::n};
  assert(std::equal(flindices.begin(), flindices.end(), expected.begin()));
  
  ta.allocate();
  assert(ta.constructed() && ta.allocated() && !ta.attached());

  cout<<"num_blocks = "<<ta.num_blocks()<<endl;
  assert(ta.num_blocks() == TensorIndex({2, 4}));

  cout<<"block_dims(1,3) = "<<ta.block_dims({1,3})<<endl;
  assert(ta.block_dims({1, 3}) == TensorIndex({4, 5}));
  
  assert(ta.block_size({0, 0}) == 16);
  assert(ta.block_size({0, 1}) == 16);
  assert(ta.block_size({0, 2}) == 20);
  assert(ta.block_size({0, 3}) == 20);
  assert(ta.block_size({1, 0}) == 16);
  assert(ta.block_size({1, 1}) == 16);
  assert(ta.block_size({1, 2}) == 20);
  assert(ta.block_size({1, 3}) == 20);

  //assert();
  
  ta.destruct();
  assert(!ta.constructed() && !ta.allocated() && !ta.attached());
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

