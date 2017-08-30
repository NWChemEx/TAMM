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
#include <iostream>
#include "gtest/gtest.h"

#include "tammx/tammx.h"

#include <mpi.h>
#include <ga.h>
#include <macdecls.h>
#include "nwtest/test_eigen.h"

//namespace {
//    tammx::ExecutionContext* g_ec;
//}
//
//class TestEnvironment : public testing::Environment {
//public:
//    explicit TestEnvironment(tammx::ExecutionContext* ec) {
//        g_ec = ec;
//    }
//};

extern "C" {
void init_fortran_vars_(Integer *noa1, Integer *nob1, Integer *nva1,
                        Integer *nvb1, logical *intorb1, logical *restricted1,
                        Integer *spins, Integer *syms, Integer *ranges);
void finalize_fortran_vars_();
//void f_calls_setvars_cxx_();
}


std::pair<Integer, Integer *>
tammx_tensor_to_fortran_info(tammx::Tensor<double> &ttensor) {
  auto adst_nw = static_cast<const tammx::Distribution_NW *>(ttensor.distribution());
  auto ahash = adst_nw->hash();
  auto length = 2 * ahash[0] + 1;
  Integer *offseta = new Integer[length];
  for (size_t i = 0; i < length; i++) {
    offseta[i] = ahash[i];
  }

  auto amgr_ga = static_cast<tammx::MemoryManagerGA *>(ttensor.memory_manager());
  Integer da = amgr_ga->ga();
  return {da, offseta};
}

// void
// fortran_assign(tammx::Tensor<double> &xtc,
//                tammx::Tensor<double> &xta,
//                add_fn fn) {
//   Integer da, *offseta_map;
//   Integer dc, *offsetc_map;
//   std::tie(da, offseta_map) = tammx_tensor_to_fortran_info(xta);
//   std::tie(dc, offsetc_map) = tammx_tensor_to_fortran_info(xtc);
//   Integer irrepa = xta.irrep().value();
//   Integer irrepc = xtc.irrep().value();

//   auto offseta = offseta_map - tamm::Variables::int_mb();
//   auto offsetc = offsetc_map - tamm::Variables::int_mb();

//   fn(&da, &offseta, &irrepa, &dc, &offsetc, &irrepc);
//   delete[] offseta_map;
//   delete[] offsetc_map;
// }

// void
// fortran_mult(tammx::Tensor<double> &xtc,
//              tammx::Tensor<double> &xta,
//              tammx::Tensor<double> &xtb,
//              mult_fn fn) {
//   Integer da, *offseta_map;
//   Integer db, *offsetb_map;
//   Integer dc, *offsetc_map;
//   std::tie(da, offseta_map) = tammx_tensor_to_fortran_info(xta);
//   std::tie(db, offsetb_map) = tammx_tensor_to_fortran_info(xtb);
//   std::tie(dc, offsetc_map) = tammx_tensor_to_fortran_info(xtc);
//   Integer irrepa = xta.irrep().value();
//   Integer irrepb = xtb.irrep().value();
//   Integer irrepc = xtc.irrep().value();

//   auto offseta = offseta_map - tamm::Variables::int_mb();
//   auto offsetb = offsetb_map - tamm::Variables::int_mb();
//   auto offsetc = offsetc_map - tamm::Variables::int_mb();

//   fn(&da, &offseta, &irrepa, &db, &offsetb, &irrepb, &dc, &offsetc, &irrepc);

//   delete[] offseta_map;
//   delete[] offsetb_map;
//   delete[] offsetc_map;
// }




#define EIGEN_ASSIGN_TEST_0D 1
#define EIGEN_ASSIGN_TEST_1D 1
#define EIGEN_ASSIGN_TEST_2D 1
#define EIGEN_ASSIGN_TEST_3D 1
#define EIGEN_ASSIGN_TEST_4D 1



void fortran_init(int noa, int nob, int nva, int nvb, bool intorb, bool restricted,
                  const std::vector<int> &spins,
                  const std::vector<int> &syms,
                  const std::vector<int> &ranges) {
  Integer inoa = noa;
  Integer inob = nob;
  Integer inva = nva;
  Integer invb = nvb;

  logical lintorb = intorb ? 1 : 0;
  logical lrestricted = restricted ? 1 : 0;

  assert(spins.size() == noa + nob + nva + nvb);
  assert(syms.size() == noa + nob + nva + nvb);
  assert(ranges.size() == noa + nob + nva + nvb);

  Integer ispins[noa + nob + nvb + nvb];
  Integer isyms[noa + nob + nvb + nvb];
  Integer iranges[noa + nob + nvb + nvb];

  std::copy_n(&spins[0], noa + nob + nva + nvb, &ispins[0]);
  std::copy_n(&syms[0], noa + nob + nva + nvb, &isyms[0]);
  std::copy_n(&ranges[0], noa + nob + nva + nvb, &iranges[0]);

  init_fortran_vars_(&inoa, &inob, &inva, &invb, &lintorb, &lrestricted,
                     &ispins[0], &isyms[0], &iranges[0]);
}

void fortran_finalize() {
  finalize_fortran_vars_();
}


void tammx_init(int noa, int nob, int nva, int nvb, bool intorb, bool restricted,
                const std::vector<int> &ispins,
                const std::vector<int> &isyms,
                const std::vector<int> &isizes) {
  using Irrep = tammx::Irrep;
  using Spin = tammx::Spin;
  using BlockDim = tammx::BlockDim;

  Irrep irrep_f{0}, irrep_v{0}, irrep_t{0}, irrep_x{0}, irrep_y{0};

  std::vector <Spin> spins;
  std::vector <Irrep> irreps;
  std::vector <size_t> sizes;

  for (auto s : ispins) {
    spins.push_back(Spin{s});
  }
  for (auto r : isyms) {
    irreps.push_back(Irrep{r});
  }
  for (auto s : isizes) {
    sizes.push_back(size_t{s});
  }

  tammx::TCE::init(spins, irreps, sizes,
                   BlockDim{noa},
                   BlockDim{noa + nob},
                   BlockDim{nva},
                   BlockDim{nva + nvb},
                   restricted,
                   irrep_f,
                   irrep_v,
                   irrep_t,
                   irrep_x,
                   irrep_y);
}

void tammx_finalize() {
  tammx::TCE::finalize();
}

////////////////////////////////////////////////

using namespace tammx;

//namespace new_impl {
TensorVec <tammx::SymmGroup>
tammx_label_to_indices(const tammx::TensorLabel &labels) {
  tammx::TensorVec <tammx::SymmGroup> ret;
  tammx::TensorDim tdims;

  for (auto l : labels) {
    tdims.push_back(l.dt);
  }
  size_t n = labels.size();
  tammx::SymmGroup sg;
  for (auto dt: tdims) {
    if (sg.size() == 0) {
      sg.push_back(dt);
    } else if (sg.back() == dt) {
      sg.push_back(dt);
    } else {
      ret.push_back(sg);
      sg = SymmGroup();
      sg.push_back(dt);
    }
  }
  if (sg.size() > 0) {
    ret.push_back(sg);
  }
  return ret;
}

template<typename T>
void
tammx_tensor_dump(const tammx::Tensor <T> &tensor, std::ostream &os) {
  const auto &buf = static_cast<const T *>(tensor.memory_manager()->access(tammx::Offset{0}));
  size_t sz = tensor.memory_manager()->local_size_in_elements().value();
  os << "tensor size=" << sz << std::endl;
  os << "tensor size (from distribution)=" << tensor.distribution()->buf_size(Proc{0}) << std::endl;
  for (size_t i = 0; i < sz; i++) {
    os << buf[i] << " ";
  }
  os << std::endl;
}

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

bool
test_initval_no_n(tammx::ExecutionContext &ec,
                  const tammx::TensorLabel &upper_labels,
                  const tammx::TensorLabel &lower_labels) {
  const auto &upper_indices = tammx_label_to_indices(upper_labels);
  const auto &lower_indices = tammx_label_to_indices(lower_labels);

  tammx::TensorRank nupper{upper_labels.size()};
  tammx::TensorVec <tammx::SymmGroup> indices{upper_indices};
  indices.insert_back(lower_indices.begin(), lower_indices.end());
  tammx::Tensor<double> xta{indices, nupper, tammx::Irrep{0}, false};
  tammx::Tensor<double> xtc{indices, nupper, tammx::Irrep{0}, false};

  double init_val = 9.1;

  g_ec->allocate(xta, xtc);
  g_ec->scheduler()
    .io(xta, xtc)
      (xta() = init_val)
      (xtc() = xta())
    .execute();

  tammx::TensorIndex id{indices.size(), tammx::BlockDim{0}};
  auto sz = xta.memory_manager()->local_size_in_elements().value();

  bool ret = true;
  const double threshold = 1e-14;
  const auto abuf = reinterpret_cast<double *>(xta.memory_manager()->access(tammx::Offset{0}));
  const auto cbuf = reinterpret_cast<double *>(xtc.memory_manager()->access(tammx::Offset{0}));
  for (int i = 0; i < sz; i++) {
    if (std::abs(abuf[i] - init_val) > threshold) {
      ret = false;
      break;
    }
  }
  if (ret == true) {
    for (int i = 0; i < sz; i++) {
      if (std::abs(cbuf[i] - init_val) > threshold) {
        return false;
      }
    }
  }
  g_ec->deallocate(xta, xtc);
  return ret;
}

bool
test_symm_assign(tammx::ExecutionContext &ec,
                 const tammx::TensorVec <tammx::SymmGroup> &cindices,
                 const tammx::TensorVec <tammx::SymmGroup> &aindices,
                 int nupper_indices,
                 const tammx::TensorLabel &clabels,
                 double alpha,
                 const std::vector<double> &factors,
                 const std::vector<tammx::TensorLabel> &alabels) {
  assert(factors.size() > 0);
  assert(factors.size() == alabels.size());
  bool restricted = ec.is_spin_restricted();
  //auto restricted = tamm::Variables::restricted();
  //auto clabels = tamm_label_to_tammx_label(tclabels);
  // std::vector<tammx::TensorLabel> alabels;
  // for(const auto& tal: talabels) {
  //   alabels.push_back(tamm_label_to_tammx_label(tal));
  // }
  tammx::TensorRank nup{nupper_indices};
  tammx::Tensor<double> tc{cindices, nup, tammx::Irrep{0}, restricted};
  tammx::Tensor<double> ta{aindices, nup, tammx::Irrep{0}, restricted};
  tammx::Tensor<double> tc2{aindices, nup, tammx::Irrep{0}, restricted};

  bool status = true;

  ec.allocate(tc, tc2, ta);

  ec.scheduler()
    .io(ta, tc, tc2)
      (ta() = 0)
      (tc() = 0)
      (tc2() = 0)
    .execute();

  auto init_lambda = [](tammx::Block<double> &block) {
    double n = std::rand() % 100;
    auto dbuf = block.buf();
    for (size_t i = 0; i < block.size(); i++) {
      dbuf[i] = n + i;
      // std::cout<<"init_lambda. dbuf["<<i<<"]="<<dbuf[i]<<std::endl;
    }
    //std::generate_n(reinterpret_cast<double *>(block.buf()), block.size(), [&]() { return n++; });
  };


  tensor_map(ta(), init_lambda);
  tammx_symmetrize(ec, ta());
  // std::cout<<"TA=\n";
  // tammx_tensor_dump(ta, std::cout);

  //std::cout<<"<<<<<<<<<<<<<<<<<<<<<<<<<"<<std::endl;
  ec.scheduler()
    .io(tc, ta)
      (tc(clabels) += alpha * ta(alabels[0]))
    .execute();
  //std::cout<<">>>>>>>>>>>>>>>>>>>>>>>>>>>>"<<std::endl;

  for (size_t i = 0; i < factors.size(); i++) {
    //std::cout<<"++++++++++++++++++++++++++"<<std::endl;
    ec.scheduler()
      .io(tc2, ta)
        (tc2(clabels) += alpha * factors[i] * ta(alabels[i]))
      .execute();
    //std::cout<<"---------------------------"<<std::endl;
  }
  // std::cout<<"TA=\n";
  // tammx_tensor_dump(ta, std::cout);
  // std::cout<<"TC=\n";
  // tammx_tensor_dump(tc, std::cout);
  // std::cout<<"TC2=\n";
  // tammx_tensor_dump(tc2, std::cout);
  // std::cout<<"\n";
  ec.scheduler()
    .io(tc, tc2)
      (tc2(clabels) += -1.0 * tc(clabels))
    .execute();
  // std::cout<<"TC - TC2=\n";
  // tammx_tensor_dump(tc2, std::cout);
  // std::cout<<"\n";

  double threshold = 1e-12;
  auto lambda = [&](auto &val) {
    if (std::abs(val) > threshold) {
      //std::cout<<"----ERROR----\n";
    }
    status &= (std::abs(val) < threshold);
  };
  ec.scheduler()
    .io(tc2)
    .sop(tc2(), lambda)
    .execute();
  ec.deallocate(tc, tc2, ta);
  return status;
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


//////////////////////////////////////////////////////
//
//           Eigen stuff
//
/////////////////////////////////////////////////////

// Eigen matrix algebra library
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

//using Tensor2D = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Tensor0D = Eigen::Tensor<double, 0, Eigen::RowMajor>;
using Tensor1D = Eigen::Tensor<double, 1, Eigen::RowMajor>;
using Tensor2D = Eigen::Tensor<double, 2, Eigen::RowMajor>;
using Tensor3D = Eigen::Tensor<double, 3, Eigen::RowMajor>;
using Tensor4D = Eigen::Tensor<double, 4, Eigen::RowMajor>;

template<int ndim>
using Perm = Eigen::array<std::ptrdiff_t, ndim>;

class EigenTensorBase {
public:
  EigenTensorBase() {};

  virtual ~EigenTensorBase() {};
};

template<int ndim>
class EigenTensor : public EigenTensorBase, public Eigen::Tensor<double, ndim, Eigen::RowMajor> {
public:
  EigenTensor(const std::array<long, ndim> &dims) : Eigen::Tensor<double, ndim, Eigen::RowMajor>(dims) {}
};

template<typename T>
bool
eigen_tensors_are_equal(EigenTensor<1> &e1,
                        EigenTensor<1> &e2,
                        double threshold = 1.0e-12) {

  bool ret = true;
  auto dims = e1.dimensions();
  for (auto i = 0; i < dims[0]; i++) {
    if (std::abs(e1(i) - e2(i)) > std::abs(threshold * e1(i))) {
      ret = false;
      break;
    }
  }
  return ret;
}

template<typename T>
bool
eigen_tensors_are_equal(EigenTensor<2> &e1,
                        EigenTensor<2> &e2,
                        double threshold = 1.0e-12) {

  bool ret = true;
  auto dims = e1.dimensions();
  for (auto i = 0; i < dims[0]; i++) {
    for (auto j = 0; j < dims[1]; j++) {
      if (std::abs(e1(i, j) - e2(i, j)) > std::abs(threshold * e1(i, j))) {
        ret = false;
        break;
      }
    }
  }
  return ret;
}

template<typename T>
bool
eigen_tensors_are_equal(EigenTensor<3> &e1,
                        EigenTensor<3> &e2,
                        double threshold = 1.0e-12) {

  bool ret = true;
  auto dims = e1.dimensions();
  for (auto i = 0; i < dims[0]; i++) {
    for (auto j = 0; j < dims[1]; j++) {
      for (auto k = 0; k < dims[2]; k++) {
        if (std::abs(e1(i, j, k) - e2(i, j, k)) > std::abs(threshold * e1(i, j, k))) {
          ret = false;
          break;
        }
      }
    }
  }
  return ret;
}

template<typename T>
bool
eigen_tensors_are_equal(EigenTensor<4> &e1,
                        EigenTensor<4> &e2,
                        double threshold = 1.0e-12) {

  bool ret = true;
  auto dims = e1.dimensions();
  for (auto i = 0; i < dims[0]; i++) {
    for (auto j = 0; j < dims[1]; j++) {
      for (auto k = 0; k < dims[2]; k++) {
        for (auto l = 0; l < dims[3]; l++) {

          if (std::abs(e1(i, j, k, l) - e2(i, j, k, l)) > std::abs(threshold * e1(i, j, k, l))) {
            ret = false;
            break;
          }
        }
      }
    }
  }
  return ret;
}


template<int ndim>
inline Perm<ndim>
eigen_perm_compute(const TensorLabel &from, const TensorLabel &to) {
  Perm<ndim> layout;

  assert(from.size() == to.size());
  assert(from.size() == ndim);
  int pa_index = 0;
  for (auto p : to) {
    auto itr = std::find(from.begin(), from.end(), p);
    Expects(itr != from.end());
    layout[pa_index] = itr - from.begin();
    pa_index++;
  }
  return layout;
}

template<int ndim>
void
eigen_assign_dispatch(EigenTensorBase *tc,
                      const TensorLabel &clabel,
                      double alpha,
                      EigenTensorBase *ta,
                      const TensorLabel &alabel) {
  assert(alabel.size() == ndim);
  assert(clabel.size() == ndim);
  auto eperm = eigen_perm_compute<ndim>(alabel, clabel);
  auto ec = static_cast<EigenTensor<ndim> *>(tc);
  auto ea = static_cast<EigenTensor<ndim> *>(ta);
  auto tmp = (*ea).shuffle(eperm);
  tmp = tmp * alpha;
  (*ec) += tmp;
}

void
eigen_assign(EigenTensorBase *tc,
             const TensorLabel &clabel,
             double alpha,
             EigenTensorBase *ta,
             const TensorLabel &alabel) {
  Expects(clabel.size() == alabel.size());
  if (clabel.size() == 0) {
    assert(0); //@todo implement
  } else if (clabel.size() == 1) {
    eigen_assign_dispatch<1>(tc, clabel, alpha, ta, alabel);
  } else if (clabel.size() == 2) {
    eigen_assign_dispatch<2>(tc, clabel, alpha, ta, alabel);
  } else if (clabel.size() == 3) {
    eigen_assign_dispatch<3>(tc, clabel, alpha, ta, alabel);
  } else if (clabel.size() == 4) {
    eigen_assign_dispatch<4>(tc, clabel, alpha, ta, alabel);
  } else {
    assert(0); //@todo implement
  }
}


//void
//eigen_mult(tamm::Tensor* tc,
//          const std::vector<tamm::IndexName>& clabel,
//          double alpha,
//          tamm::Tensor* ta,
//          const std::vector<tamm::IndexName>& alabel,
//          tamm::Tensor* tb,
//          const std::vector<tamm::IndexName>& blabel) {
//    tamm::Multiplication mult(tc, clabel, ta, alabel, tb, blabel, alpha);
//    mult.execute();
//}

/////////////////////////////////////////////////////////
//
//             eigen vs tammx
//
//
////////////////////////////////////////////////////////

size_t
compute_tammx_dim_size(tammx::DimType dt) {
  BlockDim blo, bhi;
  std::tie(blo, bhi) = tensor_index_range(dt);
  return TCE::offset(bhi) - TCE::offset(blo);
}

// template<typename T, int ndim>
// void
// patch_copy(T *sbuf, Eigen::Tensor<T, ndim, Eigen::RowMajor> &etensor,
//            const std::array<int, ndim> &block_dims,
//            const std::array<int, ndim> &rel_offset) {
//   assert(0);
// }

template<typename T>
void
patch_copy(T *sbuf, Eigen::Tensor<T, 1, Eigen::RowMajor> &etensor,
           const std::array<int, 1> &block_dims,
           const std::array<int, 1> &rel_offset) {
  int c = 0;
  for (auto i = rel_offset[0]; i < rel_offset[0] + block_dims[0]; i++, c++) {
    etensor(i) = sbuf[c];
  }

}

template<typename T>
void
patch_copy(T *sbuf, Eigen::Tensor<T, 2, Eigen::RowMajor> &etensor,
           const std::array<int, 2> &block_dims,
           const std::array<int, 2> &rel_offset) {
  int c = 0;
  for (auto i = rel_offset[0]; i < rel_offset[0] + block_dims[0]; i++) {
    for (auto j = rel_offset[1]; j < rel_offset[1] + block_dims[1];
         j++, c++) {
      etensor(i, j) = sbuf[c];
    }
  }
}


template<typename T>
void
patch_copy(T *sbuf, Eigen::Tensor<T, 3, Eigen::RowMajor> &etensor,
           const std::array<int, 3> &block_dims,
           const std::array<int, 3> &rel_offset) {
  int c = 0;
  for (auto i = rel_offset[0]; i < rel_offset[0] + block_dims[0]; i++) {
    for (auto j = rel_offset[1]; j < rel_offset[1] + block_dims[1];
         j++) {
      for (auto k = rel_offset[2]; k < rel_offset[2] + block_dims[2];
           k++, c++) {
        etensor(i, j, k) = sbuf[c];
      }
    }
  }
}

template<typename T>
void
patch_copy(T *sbuf, Eigen::Tensor<T, 4, Eigen::RowMajor> &etensor,
           const std::array<int, 4> &block_dims,
           const std::array<int, 4> &rel_offset) {
  int c = 0;
  for (auto i = rel_offset[0]; i < rel_offset[0] + block_dims[0]; i++) {
    for (auto j = rel_offset[1]; j < rel_offset[1] + block_dims[1];
         j++) {
      for (auto k = rel_offset[2]; k < rel_offset[2] + block_dims[2];
           k++) {
        for (auto l = rel_offset[3]; l < rel_offset[3] + block_dims[3];
             l++, c++) {
          etensor(i, j, k, l) = sbuf[c];
        }
      }
    }
  }
}

template<typename T, int ndim>
EigenTensorBase *
tammx_tensor_to_eigen_tensor_dispatch(tammx::Tensor <T> &tensor) {
  Expects(tensor.rank() == ndim);

  std::array<int, ndim> lo_offset, hi_offset;
  std::array<long, ndim> dims;
  const auto &flindices = tensor.flindices();
  for (int i = 0; i < ndim; i++) {
    BlockDim blo, bhi;
    std::tie(blo, bhi) = tensor_index_range(flindices[i]);
    lo_offset[i] = TCE::offset(blo);
    hi_offset[i] = TCE::offset(bhi);
    dims[i] = hi_offset[i] - lo_offset[i];
  }
  EigenTensor<ndim> *etensor = new EigenTensor<ndim>(dims);
  etensor->setZero();

  tammx::block_for(tensor(), [&](const TensorIndex &blockid) {
    auto block = tensor.get(blockid);
    const TensorIndex &boffset = block.block_offset();
    const TensorIndex &block_dims = block.block_dims();
    std::array<int, ndim> rel_offset;
    for (int i = 0; i < ndim; i++) {
      Expects(boffset[i] < hi_offset[i]);
      rel_offset[i] = boffset[i].value() - lo_offset[i];
    }
    std::array<int, ndim> block_size;
    for (int i = 0; i < ndim; i++) {
      block_size[i] = block_dims[i].value();
    }

    patch_copy<T>(block.buf(), *etensor, block_size, rel_offset);
  });

  return etensor;
}

template<typename T>
EigenTensorBase *
tammx_tensor_to_eigen_tensor(tammx::Tensor <T> &tensor) {
//   if(tensor.rank() == 0) {
//     return tammx_tensor_to_eigen_tensor_dispatch<T,0>(tensor);
//   } else
  if (tensor.rank() == 1) {
    return tammx_tensor_to_eigen_tensor_dispatch<T, 1>(tensor);
  } else if (tensor.rank() == 2) {
    return tammx_tensor_to_eigen_tensor_dispatch<T, 2>(tensor);
  } else if (tensor.rank() == 3) {
    return tammx_tensor_to_eigen_tensor_dispatch<T, 3>(tensor);
  } else if (tensor.rank() == 4) {
    return tammx_tensor_to_eigen_tensor_dispatch<T, 4>(tensor);
  }
  assert(0); //@todo implement
  return nullptr;
}

EigenTensorBase *
eigen_assign(tammx::Tensor<double> &ttc,
             const tammx::TensorLabel &tclabel,
             double alpha,
             tammx::Tensor<double> &tta,
             const tammx::TensorLabel &talabel) {
  EigenTensorBase *etc, *eta;
  etc = tammx_tensor_to_eigen_tensor(ttc);
  eta = tammx_tensor_to_eigen_tensor(tta);
  eigen_assign(etc, tclabel, alpha, eta, talabel);
  delete eta;
  return etc;
}

bool
test_eigen_assign_no_n(tammx::ExecutionContext &ec,
                       double alpha,
                       const tammx::TensorLabel &cupper_labels,
                       const tammx::TensorLabel &clower_labels,
                       const tammx::TensorLabel &aupper_labels,
                       const tammx::TensorLabel &alower_labels) {
  const auto &cupper_indices = tammx_label_to_indices(cupper_labels);
  const auto &clower_indices = tammx_label_to_indices(clower_labels);
  const auto &aupper_indices = tammx_label_to_indices(aupper_labels);
  const auto &alower_indices = tammx_label_to_indices(alower_labels);

  auto cindices = cupper_indices;
  cindices.insert_back(clower_indices.begin(), clower_indices.end());
  auto aindices = aupper_indices;
  aindices.insert_back(alower_indices.begin(), alower_indices.end());
  auto irrep = ec.irrep();
  auto restricted = ec.is_spin_restricted();
  auto cnup = cupper_labels.size();
  auto anup = aupper_labels.size();

  tammx::Tensor<double> tc1{cindices, cnup, irrep, restricted};
  tammx::Tensor<double> tc2{cindices, cnup, irrep, restricted};
  tammx::Tensor<double> ta{aindices, anup, irrep, restricted};

  ec.allocate(ta, tc1, tc2);

  ec.scheduler()
    .io(ta, tc1, tc2)
      (ta() = 0)
      (tc1() = 0)
      (tc2() = 0)
    .execute();

  tammx_tensor_fill(ec, ta());

  auto clabels = cupper_labels;
  clabels.insert_back(clower_labels.begin(), clower_labels.end());
  auto alabels = aupper_labels;
  alabels.insert_back(alower_labels.begin(), alower_labels.end());

  EigenTensorBase *etc1 = eigen_assign(tc1, clabels, alpha, ta, alabels);
  tammx_assign(ec, tc2, clabels, alpha, ta, alabels);

  EigenTensorBase *etc2 = tammx_tensor_to_eigen_tensor(tc2);

  bool status = false;
  if (tc1.rank() == 1) {
    auto *et1 = dynamic_cast<EigenTensor<1> *>(etc1);
    auto *et2 = dynamic_cast<EigenTensor<1> *>(etc2);
    status = eigen_tensors_are_equal<double>(*et1, *et2);
  } else if (tc1.rank() == 2) {
    auto *et1 = dynamic_cast<EigenTensor<2> *>(etc1);
    auto *et2 = dynamic_cast<EigenTensor<2> *>(etc2);
    status = eigen_tensors_are_equal<double>(*et1, *et2);
  } else if (tc1.rank() == 3) {
    auto *et1 = dynamic_cast<EigenTensor<3> *>(etc1);
    auto *et2 = dynamic_cast<EigenTensor<3> *>(etc2);
    status = eigen_tensors_are_equal<double>(*et1, *et2);
  } else if (tc1.rank() == 4) {
    auto *et1 = dynamic_cast<EigenTensor<4> *>(etc1);
    auto *et2 = dynamic_cast<EigenTensor<4> *>(etc2);
    status = eigen_tensors_are_equal<double>(*et1, *et2);
  }

  ec.deallocate(tc1, tc2, ta);
  delete etc1;
  delete etc2;

  return status;
}


template<int ndim>
void
eigen_mult_dispatch(EigenTensorBase *tc,
                      const TensorLabel &clabel,
                      double alpha,
                      EigenTensorBase *ta,
                      const TensorLabel &alabel,
                      EigenTensorBase *tb,
                      const TensorLabel &blabel) {
  assert(alabel.size() == ndim);
  assert(blabel.size() == ndim);
  assert(clabel.size() == ndim);
  auto eperm_a = eigen_perm_compute<ndim>(alabel, clabel);
  auto eperm_b = eigen_perm_compute<ndim>(blabel, clabel);

  auto ec = static_cast<EigenTensor<ndim> *>(tc);
  auto ea = static_cast<EigenTensor<ndim> *>(ta);
  auto eb = static_cast<EigenTensor<ndim> *>(tb);

  auto tmp_a = (*ea).shuffle(eperm_a);
  auto tmp_b = (*eb).shuffle(eperm_b);
  //tmp_a = tmp_a * alpha;
  //(*ec) += tmp_a * tmp_b;
}

void
eigen_mult(EigenTensorBase *tc,
             const TensorLabel &clabel,
             double alpha,
             EigenTensorBase *ta,
             const TensorLabel &alabel,
             EigenTensorBase *tb,
             const TensorLabel &blabel) {
  Expects(clabel.size() == alabel.size());
  if (clabel.size() == 0) {
    assert(0); //@todo implement
  } else if (clabel.size() == 1) {
    eigen_mult_dispatch<1>(tc, clabel, alpha, ta, alabel, tb, blabel);
  } else if (clabel.size() == 2) {
    eigen_mult_dispatch<2>(tc, clabel, alpha, ta, alabel, tb, blabel);
  } else if (clabel.size() == 3) {
    eigen_mult_dispatch<3>(tc, clabel, alpha, ta, alabel, tb, blabel);
  } else if (clabel.size() == 4) {
    eigen_mult_dispatch<4>(tc, clabel, alpha, ta, alabel, tb, blabel);
  } else {
    assert(0); //@todo implement
  }
}

EigenTensorBase *
eigen_mult(tammx::Tensor<double> &ttc,
             const tammx::TensorLabel &tclabel,
             double alpha,
             tammx::Tensor<double> &tta,
             const tammx::TensorLabel &talabel,
             tammx::Tensor<double> &ttb,
             const tammx::TensorLabel &tblabel) {
  EigenTensorBase *etc, *eta, *etb;
  etc = tammx_tensor_to_eigen_tensor(ttc);
  eta = tammx_tensor_to_eigen_tensor(tta);
  etb = tammx_tensor_to_eigen_tensor(ttb);
  eigen_mult(etc, tclabel, alpha, eta, talabel, etb, tblabel);
  delete eta;
  delete etb;
  return etc;
}

bool
test_eigen_mult_no_n(tammx::ExecutionContext &ec,
                       double alpha,
                       const tammx::TensorLabel &cupper_labels,
                       const tammx::TensorLabel &clower_labels,
                       const tammx::TensorLabel &aupper_labels,
                       const tammx::TensorLabel &alower_labels,
                       const tammx::TensorLabel &bupper_labels,
                       const tammx::TensorLabel &blower_labels) {
  const auto &cupper_indices = tammx_label_to_indices(cupper_labels);
  const auto &clower_indices = tammx_label_to_indices(clower_labels);
  const auto &aupper_indices = tammx_label_to_indices(aupper_labels);
  const auto &alower_indices = tammx_label_to_indices(alower_labels);
  const auto &bupper_indices = tammx_label_to_indices(bupper_labels);
  const auto &blower_indices = tammx_label_to_indices(blower_labels);

  auto cindices = cupper_indices;
  cindices.insert_back(clower_indices.begin(), clower_indices.end());
  auto aindices = aupper_indices;
  aindices.insert_back(alower_indices.begin(), alower_indices.end());
  auto bindices = bupper_indices;
  bindices.insert_back(blower_indices.begin(), blower_indices.end());
  auto irrep = ec.irrep();
  auto restricted = ec.is_spin_restricted();
  auto cnup = cupper_labels.size();
  auto anup = aupper_labels.size();
  auto bnup = bupper_labels.size();

  tammx::Tensor<double> tc1{cindices, cnup, irrep, restricted};
  tammx::Tensor<double> tc2{cindices, cnup, irrep, restricted};
  tammx::Tensor<double> ta{aindices, anup, irrep, restricted};
  tammx::Tensor<double> tb{bindices, bnup, irrep, restricted};

  ec.allocate(ta, tb, tc1, tc2);

  ec.scheduler()
    .io(ta, tb, tc1, tc2)
      (ta() = 0)
      (tb() = 0)
      (tc1() = 0)
      (tc2() = 0)
    .execute();

  tammx_tensor_fill(ec, ta());
  tammx_tensor_fill(ec, tb());

  auto clabels = cupper_labels;
  clabels.insert_back(clower_labels.begin(), clower_labels.end());
  auto alabels = aupper_labels;
  alabels.insert_back(alower_labels.begin(), alower_labels.end());
  auto blabels = bupper_labels;
  blabels.insert_back(blower_labels.begin(), blower_labels.end());

  EigenTensorBase *etc1 = eigen_mult(tc1, clabels, alpha, ta, alabels, tb, blabels);
  tammx_mult(ec, tc2, clabels, alpha, ta, alabels, tb, blabels);
  EigenTensorBase *etc2 = tammx_tensor_to_eigen_tensor(tc2);

  bool status = false;
  if (tc1.rank() == 1) {
    auto *et1 = dynamic_cast<EigenTensor<1> *>(etc1);
    auto *et2 = dynamic_cast<EigenTensor<1> *>(etc2);
    status = eigen_tensors_are_equal<double>(*et1, *et2);
  } else if (tc1.rank() == 2) {
    auto *et1 = dynamic_cast<EigenTensor<2> *>(etc1);
    auto *et2 = dynamic_cast<EigenTensor<2> *>(etc2);
    status = eigen_tensors_are_equal<double>(*et1, *et2);
  } else if (tc1.rank() == 3) {
    auto *et1 = dynamic_cast<EigenTensor<3> *>(etc1);
    auto *et2 = dynamic_cast<EigenTensor<3> *>(etc2);
    status = eigen_tensors_are_equal<double>(*et1, *et2);
  } else if (tc1.rank() == 4) {
    auto *et1 = dynamic_cast<EigenTensor<4> *>(etc1);
    auto *et2 = dynamic_cast<EigenTensor<4> *>(etc2);
    status = eigen_tensors_are_equal<double>(*et1, *et2);
  }

  ec.deallocate(tc1, tc2, ta, tb);
  delete etc1;
  delete etc2;

  return status;
}

/////////////////////////////////////////////////////////
//
//             tamm vx tammx
//
//
////////////////////////////////////////////////////////




// bool
// test_mult_no_n(tammx::ExecutionContext &ec,
//                double alpha,
//                const tammx::TensorLabel &cupper_labels,
//                const tammx::TensorLabel &clower_labels,
//                const tammx::TensorLabel &aupper_labels,
//                const tammx::TensorLabel &alower_labels,
//                const tammx::TensorLabel &bupper_labels,
//                const tammx::TensorLabel &blower_labels) {
//   const auto &cupper_indices = tammx_label_to_indices(cupper_labels);
//   const auto &clower_indices = tammx_label_to_indices(clower_labels);
//   const auto &aupper_indices = tammx_label_to_indices(aupper_labels);
//   const auto &alower_indices = tammx_label_to_indices(alower_labels);
//   const auto &bupper_indices = tammx_label_to_indices(bupper_labels);
//   const auto &blower_indices = tammx_label_to_indices(blower_labels);

//   auto cindices = cupper_indices;
//   cindices.insert_back(clower_indices.begin(), clower_indices.end());
//   auto aindices = aupper_indices;
//   aindices.insert_back(alower_indices.begin(), alower_indices.end());
//   auto bindices = bupper_indices;
//   bindices.insert_back(blower_indices.begin(), blower_indices.end());
//   auto irrep = ec.irrep();
//   auto restricted = ec.is_spin_restricted();
//   auto cnup = cupper_labels.size();
//   auto anup = aupper_labels.size();
//   auto bnup = bupper_labels.size();

//   tammx::Tensor<double> tc1{cindices, cnup, irrep, restricted};
//   tammx::Tensor<double> tc2{cindices, cnup, irrep, restricted};
//   tammx::Tensor<double> ta{aindices, anup, irrep, restricted};
//   tammx::Tensor<double> tb{bindices, bnup, irrep, restricted};

//   ec.allocate(ta, tb, tc1, tc2);

//   ec.scheduler()
//     .io(ta, tb, tc1, tc2)
//       (ta() = 0)
//       (tb() = 0)
//       (tc1() = 0)
//       (tc2() = 0)
//     .execute();


//   tammx_tensor_fill(ec, ta());
//   tammx_tensor_fill(ec, tb());

//   auto clabels = cupper_labels;
//   clabels.insert_back(clower_labels.begin(), clower_labels.end());
//   auto alabels = aupper_labels;
//   alabels.insert_back(alower_labels.begin(), alower_labels.end());
//   auto blabels = bupper_labels;
//   blabels.insert_back(blower_labels.begin(), blower_labels.end());

//   tamm_mult(tc1, clabels, alpha, ta, alabels, tb, blabels);
//   tammx_mult(ec, tc2, clabels, alpha, ta, alabels, tb, blabels);

//   bool status = tammx_tensors_are_equal(ec, tc1, tc2);

//   ec.deallocate(tc1, tc2, ta, tb);
//   return status;
// }

// bool
// test_mult_no_n(tammx::ExecutionContext &ec,
//                const tammx::TensorLabel &cupper_labels,
//                const tammx::TensorLabel &clower_labels,
//                double alpha,
//                const tammx::TensorLabel &aupper_labels,
//                const tammx::TensorLabel &alower_labels,
//                const tammx::TensorLabel &bupper_labels,
//                const tammx::TensorLabel &blower_labels) {
//   return test_mult_no_n(ec, alpha, cupper_labels, clower_labels,
//                         aupper_labels, alower_labels, bupper_labels, blower_labels);
// }

bool
test_mult_no_n(tammx::ExecutionContext &ec,
               const tammx::TensorLabel &cupper_labels,
               const tammx::TensorLabel &clower_labels,
               double alpha,
               const tammx::TensorLabel &aupper_labels,
               const tammx::TensorLabel &alower_labels,
               const tammx::TensorLabel &bupper_labels,
               const tammx::TensorLabel &blower_labels){
               //mult_fn fn) {
  const auto &cupper_indices = tammx_label_to_indices(cupper_labels);
  const auto &clower_indices = tammx_label_to_indices(clower_labels);
  const auto &aupper_indices = tammx_label_to_indices(aupper_labels);
  const auto &alower_indices = tammx_label_to_indices(alower_labels);
  const auto &bupper_indices = tammx_label_to_indices(bupper_labels);
  const auto &blower_indices = tammx_label_to_indices(blower_labels);

  auto cindices = cupper_indices;
  cindices.insert_back(clower_indices.begin(), clower_indices.end());
  auto aindices = aupper_indices;
  aindices.insert_back(alower_indices.begin(), alower_indices.end());
  auto bindices = bupper_indices;
  bindices.insert_back(blower_indices.begin(), blower_indices.end());
  auto irrep = ec.irrep();
  auto restricted = ec.is_spin_restricted();
  auto cnup = cupper_labels.size();
  auto anup = aupper_labels.size();
  auto bnup = bupper_labels.size();

  tammx::Tensor<double> tc1{cindices, cnup, irrep, restricted};
  tammx::Tensor<double> tc2{cindices, cnup, irrep, restricted};
  tammx::Tensor<double> tcf{cindices, cnup, irrep, restricted};
  tammx::Tensor<double> ta{aindices, anup, irrep, restricted};
  tammx::Tensor<double> tb{bindices, bnup, irrep, restricted};

  ec.allocate(ta, tb, tc1, tc2, tcf);

  ec.scheduler()
    .io(ta, tb, tc1, tc2, tcf)
      (ta() = 0)
      (tb() = 0)
      (tc1() = 0)
      (tc2() = 0)
      (tcf() = 0)
    .execute();

  tammx_tensor_fill(ec, ta());
  tammx_tensor_fill(ec, tb());

  auto clabels = cupper_labels;
  clabels.insert_back(clower_labels.begin(), clower_labels.end());
  auto alabels = aupper_labels;
  alabels.insert_back(alower_labels.begin(), alower_labels.end());
  auto blabels = bupper_labels;
  blabels.insert_back(blower_labels.begin(), blower_labels.end());

  //tamm_mult(tc1, clabels, alpha, ta, alabels, tb, blabels);
  std::cout << "<<<<<<<<<\n";
  tammx_mult(ec, tc2, clabels, alpha, ta, alabels, tb, blabels);
  std::cout << ">>>>>>>>>>\n";
  //fortran_mult(tcf, ta, tb, fn);
  std::cout << "------------\n";

  std::cout << "AFTER mult:\n";
  std::cout << "tammx tc=";
  tammx_tensor_dump(tc2, std::cout);
  std::cout << "\n fortran tc=";
  tammx_tensor_dump(tcf, std::cout);
  std::cout << "\n";

  bool status = tammx_tensors_are_equal(ec, tc1, tcf);

  ec.deallocate(tc1, tc2, tcf, ta, tb);
  return status;
}

tammx::TensorVec <tammx::SymmGroup>
tammx_tensor_dim_to_symm_groups(tammx::TensorDim dims, int nup) {
  tammx::TensorVec <tammx::SymmGroup> ret;

  int nlo = dims.size() - nup;
  if (nup == 0) {
    //no-op
  } else if (nup == 1) {
    tammx::SymmGroup sg{dims[0]};
    ret.push_back(sg);
  } else if (nup == 2) {
    if (dims[0] == dims[1]) {
      tammx::SymmGroup sg{dims[0], dims[1]};
      ret.push_back(sg);
    } else {
      tammx::SymmGroup sg1{dims[0]}, sg2{dims[1]};
      ret.push_back(sg1);
      ret.push_back(sg2);
    }
  } else {
    assert(0);
  }

  if (nlo == 0) {
    //no-op
  } else if (nlo == 1) {
    tammx::SymmGroup sg{dims[nup]};
    ret.push_back(sg);
  } else if (nlo == 2) {
    if (dims[nup + 0] == dims[nup + 1]) {
      tammx::SymmGroup sg{dims[nup + 0], dims[nup + 1]};
      ret.push_back(sg);
    } else {
      tammx::SymmGroup sg1{dims[nup + 0]}, sg2{dims[nup + 1]};
      ret.push_back(sg1);
      ret.push_back(sg2);
    }
  } else {
    assert(0);
  }
  return ret;
}



//-----------------------------------------------------------------------
//
//                            Initval 0-d
//
//-----------------------------------------------------------------------


#if INITVAL_TEST_0D

TEST (InitvalTest, ZeroDim
) {
ASSERT_TRUE(test_initval_no_n(*g_ec, {}, {}));
}
#endif

#if INITVAL_TEST_1D

TEST (InitvalTest, OneDim
) {
ASSERT_TRUE(test_initval_no_n(*g_ec, {}, {h1}));
ASSERT_TRUE(test_initval_no_n(*g_ec, {}, {p1}));
ASSERT_TRUE(test_initval_no_n(*g_ec, {h1}, {}));
ASSERT_TRUE(test_initval_no_n(*g_ec, {p1}, {}));
}

#endif

#if INITVAL_TEST_2D

TEST (InitvalTest, TwoDim
) {
ASSERT_TRUE(test_initval_no_n(*g_ec, {h1}, {h2}));
ASSERT_TRUE(test_initval_no_n(*g_ec, {h1}, {p2}));
ASSERT_TRUE(test_initval_no_n(*g_ec, {p1}, {h2}));
ASSERT_TRUE(test_initval_no_n(*g_ec, {p1}, {p2}));
}

#endif

#if INITVAL_TEST_3D

TEST (InitvalTest, ThreeDim
) {
ASSERT_TRUE(test_initval_no_n(*g_ec, {h1}, {h2, h3}));
ASSERT_TRUE(test_initval_no_n(*g_ec, {h1}, {h2, p3}));
ASSERT_TRUE(test_initval_no_n(*g_ec, {h1}, {p2, h3}));
ASSERT_TRUE(test_initval_no_n(*g_ec, {h1}, {p2, p3}));

ASSERT_TRUE(test_initval_no_n(*g_ec, {p1}, {h2, h3}));
ASSERT_TRUE(test_initval_no_n(*g_ec, {p1}, {h2, p3}));
ASSERT_TRUE(test_initval_no_n(*g_ec, {p1}, {p2, h3}));
ASSERT_TRUE(test_initval_no_n(*g_ec, {p1}, {p2, p3}));

ASSERT_TRUE(test_initval_no_n(*g_ec, {h1, h2}, {h3}));
ASSERT_TRUE(test_initval_no_n(*g_ec, {h1, h2}, {p3}));
ASSERT_TRUE(test_initval_no_n(*g_ec, {h1, p2}, {h3}));
ASSERT_TRUE(test_initval_no_n(*g_ec, {h1, p2}, {p3}));

ASSERT_TRUE(test_initval_no_n(*g_ec, {p1, h2}, {h3}));
ASSERT_TRUE(test_initval_no_n(*g_ec, {p1, h2}, {p3}));
ASSERT_TRUE(test_initval_no_n(*g_ec, {p1, p2}, {h3}));
ASSERT_TRUE(test_initval_no_n(*g_ec, {p1, p2}, {p3}));
}

#endif

#if INITVAL_TEST_4D

TEST (InitvalTest, FourDim
) {
ASSERT_TRUE(test_initval_no_n(*g_ec, {h1, h2}, {h3, h4}));
ASSERT_TRUE(test_initval_no_n(*g_ec, {h1, h2}, {h3, p4}));
ASSERT_TRUE(test_initval_no_n(*g_ec, {h1, h2}, {p3, h4}));
ASSERT_TRUE(test_initval_no_n(*g_ec, {h1, h2}, {p3, p4}));

ASSERT_TRUE(test_initval_no_n(*g_ec, {h1, p2}, {h3, h4}));
ASSERT_TRUE(test_initval_no_n(*g_ec, {h1, p2}, {h3, p4}));
ASSERT_TRUE(test_initval_no_n(*g_ec, {h1, p2}, {p3, h4}));
ASSERT_TRUE(test_initval_no_n(*g_ec, {h1, p2}, {p3, p4}));

ASSERT_TRUE(test_initval_no_n(*g_ec, {p1, h2}, {h3, h4}));
ASSERT_TRUE(test_initval_no_n(*g_ec, {p1, h2}, {h3, p4}));
ASSERT_TRUE(test_initval_no_n(*g_ec, {p1, h2}, {p3, h4}));
ASSERT_TRUE(test_initval_no_n(*g_ec, {p1, h2}, {p3, p4}));

ASSERT_TRUE(test_initval_no_n(*g_ec, {p1, p2}, {h3, h4}));
ASSERT_TRUE(test_initval_no_n(*g_ec, {p1, p2}, {h3, p4}));
ASSERT_TRUE(test_initval_no_n(*g_ec, {p1, p2}, {p3, h4}));
ASSERT_TRUE(test_initval_no_n(*g_ec, {p1, p2}, {p3, p4}));
}

#endif

//-----------------------------------------------------------------------
//
//                            Add 0-d
//
//-----------------------------------------------------------------------

#if ASSIGN_TEST_0D

//@todo tamm might not work with zero dimensions. So directly testing tammx.
TEST (AssignTest, ZeroDim
) {


tammx::TensorRank nupper{0};
tammx::Irrep irrep{0};
tammx::TensorVec <tammx::SymmGroup> indices{};
bool restricted = false;
tammx::Tensor<double> xta{indices, nupper, irrep, restricted};
tammx::Tensor<double> xtc{indices, nupper, irrep, restricted};

double init_val_a = 9.1, init_val_c = 8.2, alpha = 3.5;

g_ec->
allocate(xta, xtc
);
g_ec->

scheduler()

.
io(xta, xtc
)
(

xta() = init_val_a

)
(

xtc() = init_val_c

)
(

xtc()

+=

alpha *xta()

)
.

execute();

auto sz = xta.memory_manager()->local_size_in_elements().value();
bool status = true;
const double threshold = 1e-14;
const auto cbuf = reinterpret_cast<double *>(xtc.memory_manager()->access(tammx::Offset{0}));
for (
int i = 0;
i<sz;
i++) {
if (
std::abs(cbuf[i]
- (
init_val_a *alpha
+ init_val_c)) > threshold) {
status = false;
break;
}
}
g_ec->
deallocate(xta, xtc
);
ASSERT_TRUE(status);
}
#endif

//-----------------------------------------------------------------------
//
//                            Add 1-d
//
//-----------------------------------------------------------------------

#if EIGEN_ASSIGN_TEST_1D

TEST (EigenAssignTest, OneDim_o1e_o1e
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h1}, {}, {h1}, {}));
}

TEST (EigenAssignTest, OneDim_eo1_eo1
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {}, {h1}, {}, {h1}));
}

TEST (EigenAssignTest, OneDim_v1e_v1e
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p1}, {}, {p1}, {}));
}

TEST (EigenAssignTest, OneDim_ev1_ev1
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {}, {p1}, {}, {p1}));
}

#endif



//-----------------------------------------------------------------------
//
//                            Add 2-d
//
//-----------------------------------------------------------------------


#if EIGEN_ASSIGN_TEST_2D

TEST (EigenAssignTest, TwoDim_O1O2_O1O2
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h4}, {h1}, {h4}, {h1}));
}

TEST (EigenAssignTest, TwoDim_O1O2_O2O1
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 1.23, {h4}, {h1}, {h1}, {h4}));
}

TEST (EigenAssignTest, TwoDim_OV_OV
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h4}, {p1}, {h4}, {p1}));
}

TEST (EigenAssignTest, TwoDim_OV_VO
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 1.23, {h4}, {p1}, {p1}, {h4}));
}

TEST (EigenAssignTest, TwoDim_VO_VO
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p1}, {h1}, {p1}, {h1}));
}

TEST (EigenAssignTest, TwoDim_VO_OV
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 1.23, {p1}, {h1}, {h1}, {p1}));
}

TEST (EigenAssignTest, TwoDim_V1V2_V1V2
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p4}, {p1}, {p4}, {p1}));
}

TEST (EigenAssignTest, TwoDim_V1V2_V2V1
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 1.23, {p4}, {p1}, {p1}, {p4}));
}

#endif


#if EIGEN_ASSIGN_TEST_3D

TEST (EigenAssignTest, ThreeDim_o1_o2o3__o1_o2o3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h1}, {h2, h3}, {h1}, {h2, h3}));
}

TEST (EigenAssignTest, ThreeDim_o1_o2o3__o1_o3o2
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h1}, {h2, h3}, {h1}, {h3, h2}));
}

TEST (EigenAssignTest, ThreeDim_o1_o2v3__o1_o2v3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h1}, {h2, p3}, {h1}, {h2, p3}));
}

TEST (EigenAssignTest, ThreeDim_o1_o2v3__o1_v3o2
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h1}, {h2, p3}, {h1}, {p3, h2}));
}

TEST (EigenAssignTest, ThreeDim_o1_v2o3__o1_v2o3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h1}, {p2, h3}, {h1}, {p2, h3}));
}

TEST (EigenAssignTest, ThreeDim_o1_v2o3__o1_o3v2
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h1}, {p2, h3}, {h1}, {h3, p2}));
}

TEST (EigenAssignTest, ThreeDim_o1_v2v3__o1_v2v3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h1}, {p2, p3}, {h1}, {p2, p3}));
}

TEST (EigenAssignTest, ThreeDim_o1_v2v3__o1_v3v2
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h1}, {p2, p3}, {h1}, {p3, p2}));
}

///////////

TEST (EigenAssignTest, ThreeDim_v1_o2o3__v1_o2o3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p1}, {h2, h3}, {p1}, {h2, h3}));
}

TEST (EigenAssignTest, ThreeDim_v1_o2o3__v1_o3o2
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p1}, {h2, h3}, {p1}, {h3, h2}));
}

TEST (EigenAssignTest, ThreeDim_v1_o2v3__v1_o2v3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p1}, {h2, p3}, {p1}, {h2, p3}));
}

TEST (EigenAssignTest, ThreeDim_v1_o2v3__v1_v3o2
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p1}, {h2, p3}, {p1}, {p3, h2}));
}

TEST (EigenAssignTest, ThreeDim_v1_v2o3__v1_v2o3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p1}, {p2, h3}, {p1}, {p2, h3}));
}

TEST (EigenAssignTest, ThreeDim_v1_v2o3__v1_o3v2
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p1}, {p2, h3}, {p1}, {h3, p2}));
}

TEST (EigenAssignTest, ThreeDim_v1_v2v3__v1_v2v3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p1}, {p2, p3}, {p1}, {p2, p3}));
}

TEST (EigenAssignTest, ThreeDim_v1_v2v3__v1_v3v2
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p1}, {p2, p3}, {p1}, {p3, p2}));
}

//////////////////

TEST (EigenAssignTest, ThreeDim_o1o2_o3__o1o2_o3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h1, h2}, {h3}, {h1, h2}, {h3}));
}

TEST (EigenAssignTest, ThreeDim_o1o2_o3__o2o1_o3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h1, h2}, {h3}, {h2, h1}, {h3}));
}

TEST (EigenAssignTest, ThreeDim_o1o2_v3__o1o2_v3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h1, h2}, {p3}, {h1, h2}, {p3}));
}

TEST (EigenAssignTest, ThreeDim_o1o2_v3__o2o1_v3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h1, h2}, {p3}, {h2, h1}, {p3}));
}

/////////

TEST (EigenAssignTest, ThreeDim_o1v2_o3__o1v2_o3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h1, p2}, {h3}, {h1, p2}, {h3}));
}

TEST (EigenAssignTest, ThreeDim_o1v2_o3__v2o1_o3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h1, p2}, {h3}, {p2, h1}, {h3}));
}

TEST (EigenAssignTest, ThreeDim_o1v2_v3__o1v2_v3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h1, p2}, {p3}, {h1, p2}, {p3}));
}

TEST (EigenAssignTest, ThreeDim_o1v2_v3__v2o1_v3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h1, p2}, {p3}, {p2, h1}, {p3}));
}

//////////////////

TEST (EigenAssignTest, ThreeDim_v1o2_o3__v1o2_o3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p1, h2}, {h3}, {p1, h2}, {h3}));
}

TEST (EigenAssignTest, ThreeDim_v1o2_o3__o2v1_o3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p1, h2}, {h3}, {h2, p1}, {h3}));
}

TEST (EigenAssignTest, ThreeDim_v1o2_v3__v1o2_v3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p1, h2}, {p3}, {p1, h2}, {p3}));
}

TEST (EigenAssignTest, ThreeDim_v1o2_v3__o2v1_v3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p1, h2}, {p3}, {h2, p1}, {p3}));
}

/////////

TEST (EigenAssignTest, ThreeDim_v1v2_o3__v1v2_o3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p1, p2}, {h3}, {p1, p2}, {h3}));
}

TEST (EigenAssignTest, ThreeDim_v1v2_o3__v2v1_o3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p1, p2}, {h3}, {p2, p1}, {h3}));
}

TEST (EigenAssignTest, ThreeDim_v1v2_v3__v1v2_v3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p1, p2}, {p3}, {p1, p2}, {p3}));
}

TEST (EigenAssignTest, ThreeDim_v1v2_v3__v2v1_v3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p1, p2}, {p3}, {p2, p1}, {p3}));
}

//////////

#endif

//-----------------------------------------------------------------------
//
//                            Add 4-d
//
//-----------------------------------------------------------------------


#if EIGEN_ASSIGN_TEST_4D

TEST (EigenAssignTest, FourDim_o1o2o3o4_o1o2o3o4
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h1, h2}, {h3, h4}, {h1, h2}, {h3, h4}));
}

TEST (EigenAssignTest, FourDim_o1o2o3o4_o1o2o4o3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h1, h2}, {h3, h4}, {h1, h2}, {h4, h3}));
}

TEST (EigenAssignTest, FourDim_o1o2o3o4_o2o1o3o4
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h1, h2}, {h3, h4}, {h2, h1}, {h3, h4}));
}

TEST (EigenAssignTest, FourDim_o1o2o3o4_o2o1o4o3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h2, h1}, {h3, h4}, {h2, h1}, {h4, h3}));
}

///////

TEST (EigenAssignTest, FourDim_o1o2o3v4_o1o2o3v4
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h2, h1}, {h3, p4}, {h1, h2}, {h3, p4}));
}

TEST (EigenAssignTest, FourDim_o1o2o3v4_o1o2v4o3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h1, h2}, {h3, p4}, {h1, h2}, {p4, h3}));
}

TEST (EigenAssignTest, FourDim_o1o2o3v4_o2o1o3v4
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h1, h2}, {h3, p4}, {h2, h1}, {h3, p4}));
}

TEST (EigenAssignTest, FourDim_o1o2o3v4_o2o1v4o3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h1, h2}, {h3, p4}, {h2, h1}, {p4, h3}));
}

////////

TEST (EigenAssignTest, FourDim_o1o2v3o4_o1o2v3o4
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h2, h1}, {p3, h4}, {h1, h2}, {p3, h4}));
}

TEST (EigenAssignTest, FourDim_o1o2v3o4_o1o2o4v3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h1, h2}, {p3, h4}, {h1, h2}, {h4, p3}));
}

TEST (EigenAssignTest, FourDim_o1o2v3o4_o2o1v3o4
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h1, h2}, {p3, h4}, {h2, h1}, {p3, h4}));
}

TEST (EigenAssignTest, FourDim_o1o2v3o4_o2o1o4v3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h1, h2}, {p3, h4}, {h2, h1}, {h4, p3}));
}


////////

TEST (EigenAssignTest, FourDim_o1o2v3v4_o1o2v3v4
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h2, h1}, {p3, p4}, {h1, h2}, {p3, p4}));
}

TEST (EigenAssignTest, FourDim_o1o2v3v4_o1o2v4v3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h1, h2}, {p3, p4}, {h1, h2}, {p4, p3}));
}

TEST (EigenAssignTest, FourDim_o1o2v3v4_o2o1v3v4
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h1, h2}, {p3, p4}, {h2, h1}, {p3, p4}));
}

TEST (EigenAssignTest, FourDim_o1o2v3v4_o2o1v4v3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h1, h2}, {p3, p4}, {h2, h1}, {p4, p3}));
}

///////////////////////

TEST (EigenAssignTest, FourDim_o1v2o3o4_o1v2o3o4
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h1, p2}, {h3, h4}, {h1, p2}, {h3, h4}));
}

TEST (EigenAssignTest, FourDim_o1v2o3o4_o1v2o4o3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h1, p2}, {h3, h4}, {h1, p2}, {h4, h3}));
}

TEST (EigenAssignTest, FourDim_o1v2o3o4_v2o1o3o4
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h1, p2}, {h3, h4}, {p2, h1}, {h3, h4}));
}

TEST (EigenAssignTest, FourDim_o1v2o3o4_v2o1o4o3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p2, h1}, {h3, h4}, {p2, h1}, {h4, h3}));
}

///////

TEST (EigenAssignTest, FourDim_o1v2o3v4_o1v2o3v4
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p2, h1}, {h3, p4}, {h1, p2}, {h3, p4}));
}

TEST (EigenAssignTest, FourDim_o1v2o3v4_o1v2v4o3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h1, p2}, {h3, p4}, {h1, p2}, {p4, h3}));
}

TEST (EigenAssignTest, FourDim_o1v2o3v4_v2o1o3v4
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h1, p2}, {h3, p4}, {p2, h1}, {h3, p4}));
}

TEST (EigenAssignTest, FourDim_o1v2o3v4_v2o1v4o3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h1, p2}, {h3, p4}, {p2, h1}, {p4, h3}));
}

////////

TEST (EigenAssignTest, FourDim_o1v2v3o4_o1v2v3o4
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p2, h1}, {p3, h4}, {h1, p2}, {p3, h4}));
}

TEST (EigenAssignTest, FourDim_o1v2v3o4_o1v2o4v3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h1, p2}, {p3, h4}, {h1, p2}, {h4, p3}));
}

TEST (EigenAssignTest, FourDim_o1v2v3o4_v2o1v3o4
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h1, p2}, {p3, h4}, {p2, h1}, {p3, h4}));
}

TEST (EigenAssignTest, FourDim_o1v2v3o4_v2o1o4v3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h1, p2}, {p3, h4}, {p2, h1}, {h4, p3}));
}


////////

TEST (EigenAssignTest, FourDim_o1v2v3v4_o1v2v3v4
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p2, h1}, {p3, p4}, {h1, p2}, {p3, p4}));
}

TEST (EigenAssignTest, FourDim_o1v2v3v4_o1v2v4v3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h1, p2}, {p3, p4}, {h1, p2}, {p4, p3}));
}

TEST (EigenAssignTest, FourDim_o1v2v3v4_v2o1v3v4
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h1, p2}, {p3, p4}, {p2, h1}, {p3, p4}));
}

TEST (EigenAssignTest, FourDim_o1v2v3v4_v2o1v4v3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h1, p2}, {p3, p4}, {p2, h1}, {p4, p3}));
}

//////////////////////////////////////

TEST (EigenAssignTest, FourDim_v1o2o3o4_v1o2o3o4
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p1, h2}, {h3, h4}, {p1, h2}, {h3, h4}));
}

TEST (EigenAssignTest, FourDim_v1o2o3o4_v1o2o4o3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p1, h2}, {h3, h4}, {p1, h2}, {h4, h3}));
}

TEST (EigenAssignTest, FourDim_v1o2o3o4_o2v1o3o4
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p1, h2}, {h3, h4}, {h2, p1}, {h3, h4}));
}

TEST (EigenAssignTest, FourDim_v1o2o3o4_o2v1o4o3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h2, p1}, {h3, h4}, {h2, p1}, {h4, h3}));
}

///////

TEST (EigenAssignTest, FourDim_v1o2o3v4_v1o2o3v4
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h2, p1}, {h3, p4}, {p1, h2}, {h3, p4}));
}

TEST (EigenAssignTest, FourDim_v1o2o3v4_v1o2v4o3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p1, h2}, {h3, p4}, {p1, h2}, {p4, h3}));
}

TEST (EigenAssignTest, FourDim_v1o2o3v4_o2v1o3v4
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p1, h2}, {h3, p4}, {h2, p1}, {h3, p4}));
}

TEST (EigenAssignTest, FourDim_v1o2o3v4_o2v1v4o3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p1, h2}, {h3, p4}, {h2, p1}, {p4, h3}));
}

////////

TEST (EigenAssignTest, FourDim_v1o2v3o4_v1o2v3o4
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h2, p1}, {p3, h4}, {p1, h2}, {p3, h4}));
}

TEST (EigenAssignTest, FourDim_v1o2v3o4_v1o2o4v3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p1, h2}, {p3, h4}, {p1, h2}, {h4, p3}));
}

TEST (EigenAssignTest, FourDim_v1o2v3o4_o2v1v3o4
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p1, h2}, {p3, h4}, {h2, p1}, {p3, h4}));
}

TEST (EigenAssignTest, FourDim_v1o2v3o4_o2v1o4v3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p1, h2}, {p3, h4}, {h2, p1}, {h4, p3}));
}


////////

TEST (EigenAssignTest, FourDim_v1o2v3v4_v1o2v3v4
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {h2, p1}, {p3, p4}, {p1, h2}, {p3, p4}));
}

TEST (EigenAssignTest, FourDim_v1o2v3v4_v1o2v4v3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p1, h2}, {p3, p4}, {p1, h2}, {p4, p3}));
}

TEST (EigenAssignTest, FourDim_v1o2v3v4_o2v1v3v4
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p1, h2}, {p3, p4}, {h2, p1}, {p3, p4}));
}

TEST (EigenAssignTest, FourDim_v1o2v3v4_o2v1v4v3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p1, h2}, {p3, p4}, {h2, p1}, {p4, p3}));
}

//////////////////////////////////////

TEST (EigenAssignTest, FourDim_v1v2o3o4_v1v2o3o4
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p1, p2}, {h3, h4}, {p1, p2}, {h3, h4}));
}

TEST (EigenAssignTest, FourDim_v1v2o3o4_v1v2o4o3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p1, p2}, {h3, h4}, {p1, p2}, {h4, h3}));
}

TEST (EigenAssignTest, FourDim_v1v2o3o4_v2v1o3o4
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p1, p2}, {h3, h4}, {p2, p1}, {h3, h4}));
}

TEST (EigenAssignTest, FourDim_v1v2o3o4_v2v1o4o3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p2, p1}, {h3, h4}, {p2, p1}, {h4, h3}));
}

///////

TEST (EigenAssignTest, FourDim_v1v2o3v4_v1v2o3v4
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p2, p1}, {h3, p4}, {p1, p2}, {h3, p4}));
}

TEST (EigenAssignTest, FourDim_v1v2o3v4_v1v2v4o3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p1, p2}, {h3, p4}, {p1, p2}, {p4, h3}));
}

TEST (EigenAssignTest, FourDim_v1v2o3v4_v2v1o3v4
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p1, p2}, {h3, p4}, {p2, p1}, {h3, p4}));
}

TEST (EigenAssignTest, FourDim_v1v2o3v4_v2v1v4o3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p1, p2}, {h3, p4}, {p2, p1}, {p4, h3}));
}

////////

TEST (EigenAssignTest, FourDim_v1v2v3o4_v1v2v3o4
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p2, p1}, {p3, h4}, {p1, p2}, {p3, h4}));
}

TEST (EigenAssignTest, FourDim_v1v2v3o4_v1v2o4v3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p1, p2}, {p3, h4}, {p1, p2}, {h4, p3}));
}

TEST (EigenAssignTest, FourDim_v1v2v3o4_v2v1v3o4
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p1, p2}, {p3, h4}, {p2, p1}, {p3, h4}));
}

TEST (EigenAssignTest, FourDim_v1v2v3o4_v2v1o4v3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p1, p2}, {p3, h4}, {p2, p1}, {h4, p3}));
}


////////

TEST (EigenAssignTest, FourDim_v1v2v3v4_v1v2v3v4
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p2, p1}, {p3, p4}, {p1, p2}, {p3, p4}));
}

TEST (EigenAssignTest, FourDim_v1v2v3v4_v1v2v4v3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p1, p2}, {p3, p4}, {p1, p2}, {p4, p3}));
}

TEST (EigenAssignTest, FourDim_v1v2v3v4_v2v1v3v4
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p1, p2}, {p3, p4}, {p2, p1}, {p3, p4}));
}

TEST (EigenAssignTest, FourDim_v1v2v3v4_v2v1v4v3
) {
ASSERT_TRUE(test_eigen_assign_no_n(*g_ec, 0.24, {p1, p2}, {p3, p4}, {p2, p1}, {p4, p3}));
}

#endif



static const auto O = tammx::SymmGroup{tammx::DimType::o};
static const auto V = tammx::SymmGroup{tammx::DimType::v};

static const auto OO = tammx::SymmGroup{tammx::DimType::o, tammx::DimType::o};
static const auto VV = tammx::SymmGroup{tammx::DimType::v, tammx::DimType::v};

using Indices = tammx::TensorVec<tammx::SymmGroup>;

#if SYMM_ASSIGN_TEST_3D

TEST (SymmAssignTest, ThreeDim_o1o2_o3_o1o2_o3
) {
ASSERT_TRUE(test_symm_assign(*g_ec,
                             {OO, O},
                             {O, O, O},
                             2,
                             {h1, h2, h3},
                             2.5,
                             {0.5, -0.5},
                             {{h1, h2, h3},
                              {h2, h1, h3}}
));
}

TEST (SymmAssignTest, ThreeDim_o1o2_v3_o1o2_v3
) {
ASSERT_TRUE(test_symm_assign(*g_ec,
                             {OO, V},
                             {O, O, V},
                             2,
                             {h1, h2, p3},
                             2.5,
                             {0.5, -0.5},
                             {{h1, h2, p3},
                              {h2, h1, p3}}
));
}

TEST (SymmAssignTest, ThreeDim_v1v2_o3_v1v2_o3
) {
ASSERT_TRUE(test_symm_assign(*g_ec,
                             {VV, O},
                             {V, V, O},
                             2,
                             {p1, p2, h3},
                             2.5,
                             {0.5, -0.5},
                             {{p1, p2, h3},
                              {p2, p1, h3}}
));
}

TEST (SymmAssignTest, ThreeDim_v1v2_v3_v1v2_v3
) {
ASSERT_TRUE(test_symm_assign(*g_ec,
                             {VV, V},
                             {V, V, V},
                             2,
                             {p1, p2, p3},
                             2.5,
                             {0.5, -0.5},
                             {{p1, p2, p3},
                              {p2, p1, p3}}
));
}

/////////////////////////////////////////////////

TEST (SymmAssignTest, ThreeDim_o1o2_o3_o2o1_o3
) {
ASSERT_TRUE(test_symm_assign(*g_ec,
                             {OO, O},
                             {O, O, O},
                             2,
                             {h1, h2, h3},
                             2.5,
                             {0.5, -0.5},
                             {{h2, h1, h3},
                              {h1, h2, h3}}
));
}

TEST (SymmAssignTest, ThreeDim_o1o2_v3_o2o1_v3
) {
ASSERT_TRUE(test_symm_assign(*g_ec,
                             {OO, V},
                             {O, O, V},
                             2,
                             {h1, h2, p3},
                             2.5,
                             {0.5, -0.5},
                             {{h2, h1, p3},
                              {h1, h2, p3}}
));
}

TEST (SymmAssignTest, ThreeDim_v1v2_o3_v2v1_o3
) {
ASSERT_TRUE(test_symm_assign(*g_ec,
                             {VV, O},
                             {V, V, O},
                             2,
                             {p1, p2, h3},
                             2.5,
                             {0.5, -0.5},
                             {{p2, p1, h3},
                              {p1, p2, h3}}
));
}

TEST (SymmAssignTest, ThreeDim_v1v2_v3_v2v1_v3
) {
ASSERT_TRUE(test_symm_assign(*g_ec,
                             {VV, V},
                             {V, V, V},
                             2,
                             {p1, p2, p3},
                             2.5,
                             {0.5, -0.5},
                             {{p2, p1, p3},
                              {p1, p2, p3}}
));
}

#endif

#if SYMM_ASSIGN_TEST_4D

TEST (SymmAssignTest, FourDim_o1o2_o3v4_o1o2_o3v4
) {
ASSERT_TRUE(test_symm_assign(*g_ec,
                             {OO, O, V},
                             {O, O, O, V},
                             2,
                             {h1, h2, h3, p4},
                             2.5,
                             {0.5, -0.5},
                             {{h1, h2, h3, p4},
                              {h2, h1, h3, p4}}
));
}

TEST (SymmAssignTest, FourDim_o1o2_v3o4_o1o2_v3o4
) {
ASSERT_TRUE(test_symm_assign(*g_ec,
                             {OO, V, O},
                             {O, O, V, O},
                             2,
                             {h1, h2, p3, h4},
                             2.5,
                             {0.5, -0.5},
                             {{h1, h2, p3, h4},
                              {h2, h1, p3, h4}}
));
}

TEST (SymmAssignTest, FourDim_v1v2_o3v4_v1v2_o3v4
) {
ASSERT_TRUE(test_symm_assign(*g_ec,
                             {VV, O, V},
                             {V, V, O, V},
                             2,
                             {p1, p2, h3, p4},
                             2.5,
                             {0.5, -0.5},
                             {{p1, p2, h3, p4},
                              {p2, p1, h3, p4}}
));
}

TEST (SymmAssignTest, FourDim_v1v2_v3o4_v1v2_v3o4
) {
ASSERT_TRUE(test_symm_assign(*g_ec,
                             {VV, V, O},
                             {V, V, V, O},
                             2,
                             {p1, p2, p3, h4},
                             2.5,
                             {0.5, -0.5},
                             {{p1, p2, p3, h4},
                              {p2, p1, p3, h4}}
));
}

/////////////////////////////////////////////////

TEST (SymmAssignTest, FourDim_o1o2_o3v4_o2o1_o3v4
) {
ASSERT_TRUE(test_symm_assign(*g_ec,
                             {OO, O, V},
                             {O, O, O, V},
                             2,
                             {h1, h2, h3, p4},
                             2.5,
                             {0.5, -0.5},
                             {{h2, h1, h3, p4},
                              {h1, h2, h3, p4}}
));
}

TEST (SymmAssignTest, FourDim_o1o2_v3o4_o2o1_v3o4
) {
ASSERT_TRUE(test_symm_assign(*g_ec,
                             {OO, V, O},
                             {O, O, V, O},
                             2,
                             {h1, h2, p3, h4},
                             2.5,
                             {0.5, -0.5},
                             {{h2, h1, p3, h4},
                              {h1, h2, p3, h4}}
));
}

TEST (SymmAssignTest, FourDim_v1v2_o3v4_v2v1_o3v4
) {
ASSERT_TRUE(test_symm_assign(*g_ec,
                             {VV, O, V},
                             {V, V, O, V},
                             2,
                             {p1, p2, h3, p4},
                             2.5,
                             {0.5, -0.5},
                             {{p2, p1, h3, p4},
                              {p1, p2, h3, p4}}
));
}

TEST (SymmAssignTest, FourDim_v1v2_v3o4_v2v1_v3o4
) {
ASSERT_TRUE(test_symm_assign(*g_ec,
                             {VV, V, O},
                             {V, V, V, O},
                             2,
                             {p1, p2, p3, h4},
                             2.5,
                             {0.5, -0.5},
                             {{p2, p1, p3, h4},
                              {p1, p2, p3, h4}}
));
}

// //////////////////////////////////////////////

TEST (SymmAssignTest, FourDim_o1o2_o3o4_o1o2_o3o4
) {
ASSERT_TRUE(test_symm_assign(*g_ec,
                             {OO, OO},
                             {O, O, O, O},
                             2,
                             {h1, h2, h3, h4},
                             2.5,
                             {0.25, -0.25, 0.25, -0.25},
                             {{h1, h2, h3, h4},
                              {h2, h1, h3, h4},
                              {h2, h1, h4, h3},
                              {h1, h2, h4, h3}}
));
}

TEST (SymmAssignTest, FourDim_o1o2_v3v4_o1o2_v3v4
) {
ASSERT_TRUE(test_symm_assign(*g_ec,
                             {OO, VV},
                             {O, O, V, V},
                             2,
                             {h1, h2, p3, p4},
                             2.5,
                             {0.25, -0.25, 0.25, -0.25},
                             {{h1, h2, p3, p4},
                              {h2, h1, p3, p4},
                              {h2, h1, p4, p3},
                              {h1, h2, p4, p3}}
));
}

TEST (SymmAssignTest, FourDim_v1v2_o3o4_v1v2_o3o4
) {
ASSERT_TRUE(test_symm_assign(*g_ec,
                             {VV, OO},
                             {V, V, O, O},
                             2,
                             {p1, p2, h3, h4},
                             2.5,
                             {0.25, -0.25, 0.25, -0.25},
                             {{p1, p2, h3, h4},
                              {p2, p1, h3, h4},
                              {p2, p1, h4, h3},
                              {p1, p2, h4, h3}}
));
}

TEST (SymmAssignTest, FourDim_v1v2_v3v4_v1v2_v3v4
) {
ASSERT_TRUE(test_symm_assign(*g_ec,
                             {VV, VV},
                             {V, V, V, V},
                             2,
                             {p1, p2, p3, p4},
                             2.5,
                             {0.25, -0.25, 0.25, -0.25},
                             {{p1, p2, p3, p4},
                              {p2, p1, p3, p4},
                              {p2, p1, p4, p3},
                              {p1, p2, p4, p3}}
));
}

///////////////////////////////////////////

TEST (SymmAssignTest, FourDim_o1o2_o3o4_o2o1_o3o4
) {
ASSERT_TRUE(test_symm_assign(*g_ec,
                             {OO, OO},
                             {O, O, O, O},
                             2,
                             {h1, h2, h3, h4},
                             2.5,
                             {0.25, -0.25, 0.25, -0.25},
                             {{h2, h1, h3, h4},
                              {h2, h1, h4, h3},
                              {h1, h2, h4, h3},
                              {h1, h2, h3, h4}}
));
}

TEST (SymmAssignTest, FourDim_o1o2_v3v4_o2o1_v3v4
) {
ASSERT_TRUE(test_symm_assign(*g_ec,
                             {OO, VV},
                             {O, O, V, V},
                             2,
                             {h1, h2, p3, p4},
                             2.5,
                             {0.25, -0.25, 0.25, -0.25},
                             {{h2, h1, p3, p4},
                              {h2, h1, p4, p3},
                              {h1, h2, p4, p3},
                              {h1, h2, p3, p4}}
));
}

TEST (SymmAssignTest, FourDim_v1v2_o3o4_v2v1_o3o4
) {
ASSERT_TRUE(test_symm_assign(*g_ec,
                             {VV, OO},
                             {V, V, O, O},
                             2,
                             {p1, p2, h3, h4},
                             2.5,
                             {0.25, -0.25, 0.25, -0.25},
                             {{p2, p1, h3, h4},
                              {p2, p1, h4, h3},
                              {p1, p2, h4, h3},
                              {p1, p2, h3, h4}}
));
}

TEST (SymmAssignTest, FourDim_v1v2_v3v4_v2v1_v3v4
) {
ASSERT_TRUE(test_symm_assign(*g_ec,
                             {VV, VV},
                             {V, V, V, V},
                             2,
                             {p1, p2, p3, p4},
                             2.5,
                             {0.25, -0.25, 0.25, -0.25},
                             {{p2, p1, p3, p4},
                              {p2, p1, p4, p3},
                              {p1, p2, p4, p3},
                              {p1, p2, p3, p4}}
));
}

//////////////////////////////////////////

TEST (SymmAssignTest, FourDim_o1o2_o3o4_o1o2_o4o3
) {
ASSERT_TRUE(test_symm_assign(*g_ec,
                             {OO, OO},
                             {O, O, O, O},
                             2,
                             {h1, h2, h3, h4},
                             2.5,
                             {0.25, -0.25, 0.25, -0.25},
                             {{h1, h2, h4, h3},
                              {h2, h1, h4, h3},
                              {h2, h1, h3, h4},
                              {h1, h2, h3, h4}}
));
}

TEST (SymmAssignTest, FourDim_o1o2_v3v4_o1o2_v4v3
) {
ASSERT_TRUE(test_symm_assign(*g_ec,
                             {OO, VV},
                             {O, O, V, V},
                             2,
                             {h1, h2, p3, p4},
                             2.5,
                             {0.25, -0.25, 0.25, -0.25},
                             {{h1, h2, p4, p3},
                              {h2, h1, p4, p3},
                              {h2, h1, p3, p4},
                              {h1, h2, p3, p4}}
));
}

TEST (SymmAssignTest, FourDim_v1v2_o3o4_v1v2_o4o3
) {
ASSERT_TRUE(test_symm_assign(*g_ec,
                             {VV, OO},
                             {V, V, O, O},
                             2,
                             {p1, p2, h3, h4},
                             2.5,
                             {0.25, -0.25, 0.25, -0.25},
                             {{p1, p2, h4, h3},
                              {p2, p1, h4, h3},
                              {p2, p1, h3, h4},
                              {p1, p2, h3, h4}}
));
}

TEST (SymmAssignTest, FourDim_v1v2_v3v4_v1v2_v4v3
) {
ASSERT_TRUE(test_symm_assign(*g_ec,
                             {VV, VV},
                             {V, V, V, V},
                             2,
                             {p1, p2, p3, p4},
                             2.5,
                             {0.25, -0.25, 0.25, -0.25},
                             {{p1, p2, p4, p3},
                              {p2, p1, p4, p3},
                              {p2, p1, p3, p4},
                              {p1, p2, p3, p4}}
));
}

////////////////////////////////////////////////

TEST (SymmAssignTest, FourDim_o1o2_o3o4_o2o1_o4o3
) {
ASSERT_TRUE(test_symm_assign(*g_ec,
                             {OO, OO},
                             {O, O, O, O},
                             2,
                             {h1, h2, h3, h4},
                             2.5,
                             {0.25, -0.25, 0.25, -0.25},
                             {{h2, h1, h4, h3},
                              {h2, h1, h3, h4},
                              {h1, h2, h3, h4},
                              {h1, h2, h4, h3}}
));
}

TEST (SymmAssignTest, FourDim_o1o2_v3v4_o2o1_v4v3
) {
ASSERT_TRUE(test_symm_assign(*g_ec,
                             {OO, VV},
                             {O, O, V, V},
                             2,
                             {h1, h2, p3, p4},
                             2.5,
                             {0.25, -0.25, 0.25, -0.25},
                             {{h2, h1, p4, p3},
                              {h2, h1, p3, p4},
                              {h1, h2, p3, p4},
                              {h1, h2, p4, p3}}
));
}

TEST (SymmAssignTest, FourDim_v1v2_o3o4_v2v1_o4o3
) {
ASSERT_TRUE(test_symm_assign(*g_ec,
                             {VV, OO},
                             {V, V, O, O},
                             2,
                             {p1, p2, h3, h4},
                             2.5,
                             {0.25, -0.25, 0.25, -0.25},
                             {{p2, p1, h4, h3},
                              {p2, p1, h3, h4},
                              {p1, p2, h3, h4},
                              {p1, p2, h4, h3}}
));
}

TEST (SymmAssignTest, FourDim_v1v2_v3v4_v2v1_v4v3
) {
ASSERT_TRUE(test_symm_assign(*g_ec,
                             {VV, VV},
                             {V, V, V, V},
                             2,
                             {p1, p2, p3, p4},
                             2.5,
                             {0.25, -0.25, 0.25, -0.25},
                             {{p2, p1, p4, p3},
                              {p2, p1, p3, p4},
                              {p1, p2, p3, p4},
                              {p1, p2, p4, p3}}
));
}

#endif


#if MULT_TEST_0D_0D

TEST (MultTest, Dim_0_0_0
) {
tammx::Tensor<double> xtc{{}, 0, tammx::Irrep{0}, false};
tammx::Tensor<double> xta{{}, 0, tammx::Irrep{0}, false};
tammx::Tensor<double> xtb{{}, 0, tammx::Irrep{0}, false};

double alpha1 = 0.91, alpha2 = 0.56;
auto &ec = *g_ec;

ec.
allocate(xta, xtb, xtc
);
ec.

scheduler()

.
io(xta, xtb, xtc
)
(

xta() = alpha1

)
(

xtb() = alpha2

)
(

xtc() = xta() * xtb()

)
.

execute();

double threshold = 1.0e-12;
bool status = true;
auto lambda = [&](auto &val) {
  status &= (std::abs(val - alpha1 * alpha2) < threshold);
};
ec.

scheduler()

.
io(xtc)
.

sop(xtc(), lambda

)
.

execute();

ec.
deallocate(xta, xtb, xtc
);
ASSERT_TRUE(status);
}
#endif


#if MULT_TEST_0D_1D

TEST (MultTest, Dim_o_0_o_up
) {
tammx::Tensor<double> xtc{{O}, 1, tammx::Irrep{0}, false};
tammx::Tensor<double> xta{{}, 0, tammx::Irrep{0}, false};
tammx::Tensor<double> xtb{{O}, 1, tammx::Irrep{0}, false};

double alpha1 = 0.91, alpha2 = 0.56;
auto &ec = *g_ec;

ec.
allocate(xta, xtb, xtc
);
ec.

scheduler()

.
io(xta, xtb, xtc
)
(

xta() = alpha1

)
(

xtb() = alpha2

)
(

xtc() = xta() * xtb()

)
.

execute();

double threshold = 1.0e-12;
bool status = true;
auto lambda = [&](auto &val) {
  status &= (std::abs(val - alpha1 * alpha2) < threshold);
};
ec.

scheduler()

.
io(xtc)
.

sop(xtc(), lambda

)
.

execute();

ec.
deallocate(xta, xtb, xtc
);
ASSERT_TRUE(status);
}

TEST (MultTest, Dim_o_0_o_lo
) {
tammx::Tensor<double> xtc{{O}, 0, tammx::Irrep{0}, false};
tammx::Tensor<double> xta{{}, 0, tammx::Irrep{0}, false};
tammx::Tensor<double> xtb{{O}, 0, tammx::Irrep{0}, false};

double alpha1 = 0.91, alpha2 = 0.56;
auto &ec = *g_ec;

ec.
allocate(xta, xtb, xtc
);
ec.

scheduler()

.
io(xta, xtb, xtc
)
(

xta() = alpha1

)
(

xtb() = alpha2

)
(

xtc() = xta() * xtb()

)
.

execute();

double threshold = 1.0e-12;
bool status = true;
auto lambda = [&](auto &val) {
  status &= (std::abs(val - alpha1 * alpha2) < threshold);
};
ec.

scheduler()

.
io(xtc)
.

sop(xtc(), lambda

)
.

execute();

ec.
deallocate(xta, xtb, xtc
);
ASSERT_TRUE(status);
}

TEST (MultTest, Dim_v_v_0_hi
) {
tammx::Tensor<double> xtc{{V}, 1, tammx::Irrep{0}, false};
tammx::Tensor<double> xta{{V}, 1, tammx::Irrep{0}, false};
tammx::Tensor<double> xtb{{}, 0, tammx::Irrep{0}, false};

double alpha1 = 0.91, alpha2 = 0.56;
auto &ec = *g_ec;

ec.
allocate(xta, xtb, xtc
);
ec.

scheduler()

.
io(xta, xtb, xtc
)
(

xta() = alpha1

)
(

xtb() = alpha2

)
(

xtc() = xta() * xtb()

)
.

execute();

double threshold = 1.0e-12;
bool status = true;
auto lambda = [&](auto &val) {
  status &= (std::abs(val - alpha1 * alpha2) < threshold);
};
ec.

scheduler()

.
io(xtc)
.

sop(xtc(), lambda

)
.

execute();

ec.
deallocate(xta, xtb, xtc
);
ASSERT_TRUE(status);
}

TEST (MultTest, Dim_v_v_0_lo
) {
tammx::Tensor<double> xtc{{V}, 0, tammx::Irrep{0}, false};
tammx::Tensor<double> xta{{V}, 0, tammx::Irrep{0}, false};
tammx::Tensor<double> xtb{{}, 0, tammx::Irrep{0}, false};

double alpha1 = 0.91, alpha2 = 0.56;
auto &ec = *g_ec;

ec.
allocate(xta, xtb, xtc
);
ec.

scheduler()

.
io(xta, xtb, xtc
)
(

xta() = alpha1

)
(

xtb() = alpha2

)
(

xtc() = xta() * xtb()

)
.

execute();

double threshold = 1.0e-12;
bool status = true;
auto lambda = [&](auto &val) {
  status &= (std::abs(val - alpha1 * alpha2) < threshold);
};
ec.

scheduler()

.
io(xtc)
.

sop(xtc(), lambda

)
.

execute();

ec.
deallocate(xta, xtb, xtc
);
ASSERT_TRUE(status);
}

#endif


int main(int argc, char *argv[]) {
  bool intorb = false;
  bool restricted = false;

#if 0
  int noa = 1;
  int nob = 1;
  int nva = 1;
  int nvb = 1;
  std::vector<int> spins = {1, 2, 1, 2};
  std::vector<int> syms = {0, 0, 0, 0};
  std::vector<int> ranges = {1, 1, 1, 1};
#else
  int noa = 2;
  int nob = 2;
  int nva = 2;
  int nvb = 2;
  std::vector<int> spins = {1, 1, 2, 2, 1, 1, 2, 2};
  std::vector<int> syms = {0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int> ranges = {4, 4, 4, 4, 4, 4, 4, 4};
#endif

  MPI_Init(&argc, &argv);
  GA_Initialize();
  MA_init(MT_DBL, 8000000, 20000000);

  fortran_init(noa, nob, nva, nvb, intorb, restricted, spins, syms, ranges);
  //tamm_init(noa, nob, nva, nvb, intorb, restricted, spins, syms, ranges);
  tammx_init(noa, nob, nva, nvb, intorb, restricted, spins, syms, ranges);

  tammx::ProcGroup pg{tammx::ProcGroup{MPI_COMM_WORLD}.clone()};
  auto default_distribution = tammx::Distribution_NW();
  tammx::MemoryManagerGA default_memory_manager{pg};
  auto default_irrep = tammx::Irrep{0};
  auto default_spin_restricted = false;

  ::testing::InitGoogleTest(&argc, argv);

  int ret = 0;

  tammx::ExecutionContext ec{pg, &default_distribution, &default_memory_manager,
                             default_irrep, default_spin_restricted};

  testing::AddGlobalTestEnvironment(new TestEnvironment(&ec));

  // temporarily commented
  ret = RUN_ALL_TESTS();
  // test_assign_2d(ec);
  // test_assign_4d(ec);
  // test_assign(ec);
  // test_mult_vo_oo(ec);
  // test_mult_vvoo_ov(ec);
#if 0
  // CCSD methods
test_assign_ccsd_e(ec);
test_assign_ccsd_t1(ec);
test_assign_ccsd_t2(ec);
test_assign_cc2_t1(ec);
test_assign_cc2_t2(ec);

test_assign_cisd_c1(ec);
test_assign_cisd_c2(ec);
test_assign_ccsd_lambda1(ec);
test_assign_ccsd_lambda2(ec);
test_assign_eaccsd_x1(ec);
test_assign_eaccsd_x2(ec);

test_assign_icsd_t1(ec);
test_assign_icsd_t2(ec);

test_assign_ipccsd_x1(ec);
test_assign_ipccsd_x2(ec);
#endif

  pg.destroy();
  tammx_finalize();
 // tamm_finalize();
  fortran_finalize();

  GA_Terminate();
  MPI_Finalize();
  return ret;
}
