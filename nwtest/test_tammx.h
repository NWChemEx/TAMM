//------------------------------------------------------------------------------
// Copyright (C) 2016, Pacific Northwest National Laboratory
// This software is subject to copyright protection under the laws of the
// United States and other countries
//
// All rights in this computer software are reserved by the
// Pacific Northwest National Laboratory (PNNL)
// Operated by Battelle for the U.S. Department of Energy
//
//------------------------------------------------------------------------------

#include "tammx/tammx.h"
using namespace tammx;

namespace {
  tammx::ExecutionContext* g_ec;
  }
  
  class TestEnvironment : public testing::Environment {
   public:
    explicit TestEnvironment(tammx::ExecutionContext* ec) {
      g_ec = ec;
    }
  };


  tammx::TensorVec <tammx::SymmGroup>
  tammx_tensor_dim_to_symm_groups(tammx::TensorDim dims, int nup);
  
  void tammx_init(int noa, int nob, int nva, int nvb, bool intorb, bool restricted,
    const std::vector<int> &ispins,
    const std::vector<int> &isyms,
    const std::vector<int> &isizes);
  
  void tammx_finalize();
  
  TensorVec <tammx::SymmGroup>
  tammx_label_to_indices(const tammx::TensorLabel &labels);

  bool
  test_initval_no_n(tammx::ExecutionContext &ec,
                    const tammx::TensorLabel &upper_labels,
                    const tammx::TensorLabel &lower_labels);

  bool
  test_symm_assign(tammx::ExecutionContext &ec,
                   const tammx::TensorVec <tammx::SymmGroup> &cindices,
                   const tammx::TensorVec <tammx::SymmGroup> &aindices,
                   int nupper_indices,
                   const tammx::TensorLabel &clabels,
                   double alpha,
                   const std::vector<double> &factors,
                   const std::vector<tammx::TensorLabel> &alabels);

  template<typename LabeledTensorType>
  void
  tammx_symmetrize(tammx::ExecutionContext &ec, LabeledTensorType ltensor) {
    auto &tensor = *ltensor.tensor_;
    auto &label = ltensor.label_;
    const auto &indices = tensor.indices();
    Expects(tensor.flindices().size() == label.size());
  
    auto label_copy = label;
    size_t off = 0;
    for (auto &sg: indices) {
      if (sg.size() > 1) {
        //@todo handle other cases
        assert(sg.size() == 2);
        std::swap(label_copy[off], label_copy[off + 1]);
        ec.scheduler()
          .io(tensor)
            (tensor(label) += tensor(label_copy))
            (tensor(label) += -0.5 * tensor(label))
          .execute();
        std::swap(label_copy[off], label_copy[off + 1]);
      }
      off += sg.size();
    }
  }

  template<typename LabeledTensorType>
  void
  tammx_tensor_fill(tammx::ExecutionContext &ec,
                    LabeledTensorType ltensor) {
    using T = typename LabeledTensorType::element_type;
    auto init_lambda = [](tammx::Block <T> &block) {
      double n = std::rand() % 5;
      auto dbuf = block.buf();
      for (size_t i = 0; i < block.size(); i++) {
        dbuf[i] = T{n + i};
        //std::cout << "init_lambda. dbuf[" << i << "]=" << dbuf[i] << std::endl;
      }
    };
  
    tensor_map(ltensor, init_lambda);
    tammx_symmetrize(ec, ltensor);
  }
  


template<typename T>
void
tammx_assign(tammx::ExecutionContext &ec,
             tammx::Tensor <T> &tc,
             const tammx::TensorLabel &clabel,
             double alpha,
             tammx::Tensor <T> &ta,
             const tammx::TensorLabel &alabel) {
  // auto al = tamm_label_to_tammx_label(alabel);
  // auto cl = tamm_label_to_tammx_label(clabel);
  auto &al = alabel;
  auto &cl = clabel;

  // std::cout<<"----AL="<<al<<std::endl;
  // std::cout<<"----CL="<<cl<<std::endl;
  ec.scheduler()
    .io((tc), (ta))
      ((tc)(cl) += alpha * (ta)(al))
    .execute();

  // delete ta;
  // delete tc;
}

template<typename T>
void
tammx_mult(tammx::ExecutionContext &ec,
           tammx::Tensor <T> &tc,
           const tammx::TensorLabel &clabel,
           double alpha,
           tammx::Tensor <T> &ta,
           const tammx::TensorLabel &alabel,
           tammx::Tensor <T> &tb,
           const tammx::TensorLabel &blabel) {
  // tammx::Tensor<double> *ta = tamm_tensor_to_tammx_tensor(ec.pg(), tta);
  // tammx::Tensor<double> *tb = tamm_tensor_to_tammx_tensor(ec.pg(), ttb);
  // tammx::Tensor<double> *tc = tamm_tensor_to_tammx_tensor(ec.pg(), ttc);

  // auto al = tamm_label_to_tammx_label(alabel);
  // auto bl = tamm_label_to_tammx_label(blabel);
  // auto cl = tamm_label_to_tammx_label(clabel);

  auto &al = alabel;
  auto &bl = blabel;
  auto &cl = clabel;

  // {
  //   std::cout << "tammx_mult. A = ";
  //   tammx_tensor_dump(ta, std::cout);
  //   std::cout << "tammx_mult. B = ";
  //   tammx_tensor_dump(tb, std::cout);
  //   std::cout << "tammx_mult. C = ";
  //   tammx_tensor_dump(tc, std::cout);
  // }

  // std::cout<<"----AL="<<al<<std::endl;
  // std::cout<<"----BL="<<bl<<std::endl;
  // std::cout<<"----CL="<<cl<<std::endl;
  ec.scheduler()
    .io((tc), (ta), (tb))
      ((tc)() = 0)
      ((tc)(cl) += alpha * (ta)(al) * (tb)(bl))
    .execute();

  // delete ta;
  // delete tb;
  // delete tc;
}

            
template<typename LabeledTensorType>
bool
tammx_tensors_are_equal(tammx::ExecutionContext &ec,
                        const LabeledTensorType &ta,
                        const LabeledTensorType &tb,
                        double threshold = 1.0e-12) {
  auto asz = ta.memory_manager()->local_size_in_elements().value();
  auto bsz = tb.memory_manager()->local_size_in_elements().value();

  if (asz != bsz) {
    return false;
  }

  using T = typename LabeledTensorType::element_type;
  const double *abuf = reinterpret_cast<const T *>(ta.memory_manager()->access(tammx::Offset{0}));
  const double *bbuf = reinterpret_cast<const T *>(tb.memory_manager()->access(tammx::Offset{0}));
  bool ret = true;
  for (int i = 0; i < asz; i++) {
    // std::cout << abuf[i] << ": " << bbuf[i];
    if (std::abs(abuf[i] - bbuf[i]) > std::abs(threshold * abuf[i])) {
      // std::cout << "--\n";
      ret = false;
      break;
    }
    //std::cout << "\n";
  }
  return ret;
}


