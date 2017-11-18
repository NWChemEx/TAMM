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

#ifndef TEST_TAMMX_H
#define TEST_TAMMX_H

#include "tammx/tammx.h"
using namespace tammx;

  extern tammx::ExecutionContext* g_ec;


  class TestEnvironment : public testing::Environment {
   public:
    explicit TestEnvironment(tammx::ExecutionContext* ec) {
      g_ec = ec;
    }
  };


  tammx::TensorVec <tammx::TensorSymmGroup>
  tammx_tensor_dim_to_symm_groups(tammx::DimTypeVec dims, TAMMX_INT32 nup);

  void tammx_init(TAMMX_INT32 noa, TAMMX_INT32 nob, TAMMX_INT32 nva, TAMMX_INT32 nvb, bool intorb, bool restricted,
    const std::vector<TAMMX_INT32> &ispins,
    const std::vector<TAMMX_INT32> &isyms,
    const std::vector<TAMMX_INT32> &isizes);

  void tammx_finalize();

  TensorVec <tammx::TensorSymmGroup>
  tammx_label_to_indices(const tammx::IndexLabelVec &labels);

  TensorVec <tammx::TensorSymmGroup>
  tammx_label_to_indices(const tammx::IndexLabelVec &upper_labels,
                         const tammx::IndexLabelVec &lower_labels,
                         bool all_n = false);

  bool
  test_initval_no_n(tammx::ExecutionContext &ec,
                    const tammx::IndexLabelVec &upper_labels,
                    const tammx::IndexLabelVec &lower_labels);

  bool
  test_symm_assign(tammx::ExecutionContext &ec,
                   const tammx::TensorVec <tammx::TensorSymmGroup> &cindices,
                   const tammx::TensorVec <tammx::TensorSymmGroup> &aindices,
                   TAMMX_INT32 nupper_indices,
                   const tammx::IndexLabelVec &clabels,
                   double alpha,
                   const std::vector<double> &factors,
                   const std::vector<tammx::IndexLabelVec> &alabels);

  template<typename LabeledTensorType>
  void
  tammx_symmetrize(tammx::ExecutionContext &ec, LabeledTensorType ltensor) {
    auto &tensor = *ltensor.tensor_;
    auto &label = ltensor.label_;
    const auto &indices = tensor.tindices();
    EXPECTS(tensor.flindices().size() == label.size());

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
    auto& tensor = *ltensor.tensor_;
    auto init_lambda = [&](auto& blockid) {
      double n = std::rand() % 5;
      auto block = tensor.alloc(blockid);
      auto dbuf = block.buf();      
      for (size_t i = 0; i < block.size(); i++) {
        dbuf[i] = T{n + i};
        //std::cout << "init_lambda. dbuf[" << i << "]=" << dbuf[i] << std::endl;
      }
      tensor.put(blockid, block);
    };

    tammx::block_parfor(ec.pg(), ltensor, init_lambda);
    tammx_symmetrize(ec, ltensor);
  }


enum class AllocationType {
  no_n      = 0b000,
  lhs_n     = 0b100,
  rhs1_n    = 0b010,
  rhs2_n    = 0b001,
  all_n     = 0b111,
  all_rhs_n = 0b011,
  lr1_n     = 0b110,
  lr2_n     = 0b101
};

inline bool
is_lhs_n(AllocationType at) {
  return static_cast<unsigned>(at) &
      static_cast<unsigned>(AllocationType::lhs_n);
}

inline bool
is_rhs1_n(AllocationType at) {
  return static_cast<unsigned>(at) &
      static_cast<unsigned>(AllocationType::rhs1_n);
}

inline bool
is_rhs2_n(AllocationType at) {
  return static_cast<unsigned>(at) &
      static_cast<unsigned>(AllocationType::rhs2_n);
}

template<typename T>
void
tammx_assign(tammx::ExecutionContext &ec,
             tammx::Tensor <T> &tc,
             const tammx::IndexLabelVec &clabel,
             double alpha,
             tammx::Tensor <T> &ta,
             const tammx::IndexLabelVec &alabel) {
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
           const tammx::IndexLabelVec &clabel,
           double alpha,
           tammx::Tensor <T> &ta,
           const tammx::IndexLabelVec &alabel,
           tammx::Tensor <T> &tb,
           const tammx::IndexLabelVec &blabel) {
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
  // auto asz = ta.memory_manager()->local_size_in_elements().value();
  // auto bsz = tb.memory_manager()->local_size_in_elements().value();
  auto asz = ta.memory_region().local_nelements().value();
  auto bsz = tb.memory_region().local_nelements().value();

  if (asz != bsz) {
    return false;
  }

  using T = typename LabeledTensorType::element_type;
  const double *abuf = reinterpret_cast<const T *>(ta.memory_region().access(tammx::Offset{0}));
  const double *bbuf = reinterpret_cast<const T *>(tb.memory_region().access(tammx::Offset{0}));
  bool ret = true;
  for (TAMMX_INT32 i = 0; i < asz; i++) {
    if (std::abs(abuf[i] - bbuf[i]) > std::abs(threshold * abuf[i])) {
      if(!(std::abs(abuf[i]) <= 5e-13 && std::abs(bbuf[i])<= 5e-13) ) {
      //   std::cout << abuf[i] << ": " << bbuf[i] << std::endl;
        ret = false;
        break;
      }
    }
  }
  return ret;
}


#endif // TEST_TAMMX_H
