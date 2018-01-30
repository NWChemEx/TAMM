#include "boundvec.h"
#include "strong_num.h"
#include "types.h"
#include "index_space.h"
#include "proc_group.h"
#include "mso.h"
#include "ao.h"
#include "tensor_base.h"
#include "tensor.h"
#include "labeled_tensor.h"
#include "ops.h"

// //#include "memory_manager_ga.h"
// #include "proc_group.h"
// #include "tensor_base.h"
// #include "types.h"
// #include "distribution.h"
// #include "generator.h"
// #include "memory_manager_local.h"
// #include "errors.h"
// #include "memory_manager.h"
// #include "perm_symmetry.h"
// #include "strong_num.h"
// #include "block.h"
// #include "labeled_block.h"
// #include "labeled_tensor.h"

// #include "tensor.h"
// #include "index_sort.h"
// #include "execution_context.h"
// #include "operators.h"

//#include "ops.h"

using namespace tammy;

int main()
{
  // IndexRange i_r, j_r, k_r, l_r;
  // TensorVec<IndexRange> ranges{};
  // TensorVec<IndexPosition> ipmasks{};
  // IndexLabel i{i_r,0}, j{j_r,1}, k{k_r,2}, l{l_r,2};
  // Tensor<double> T1 = Tensor<double>::create<TensorImpl<double>>(ranges, ipmasks);
  // Tensor<double> T2 = Tensor<double>::create<TensorImpl<double>>(ranges, ipmasks);
  // Tensor<double> T3 = Tensor<double>::create<TensorImpl<double>>(ranges, ipmasks);

  MSO mso;
  IndexLabel i, j, k, l;
  std::tie(i,j,k,l) = mso.N(0,1,2,3);

  auto T1 = Tensor<double>::create<TensorImpl<double>>(i, j);
  auto T2 = Tensor<double>::create<TensorImpl<double>>(i(k), j);
  auto T3 = Tensor<double>::create<TensorImpl<double>>(i, j(l));
  auto T4 = Tensor<double>::create<TensorImpl<double>>(i(k), j(l));

  T1(i,j) = 0;
  T1(i,j) += .52;
  T1(i,j) = T2(j,i);
  T1(i,j) += T2(j,i);
  T1(i,j) = 3 * T2(j,i);
  T1(i,j) += 3 * T2(j,i);
  T1(i,j) = T2(j,i) * T3(k,j);
  T1(i,j) += T2(j,i) * T3(k,j);
  T1(i,j) = 3 * T2(j,i) * T3(j,l);
  T1(i,j) += 3 * T2(j,i) * T3(j,l);
  

  return 0;
}

