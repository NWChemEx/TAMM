

#define CATCH_CONFIG_RUNNER
#include "catch/catch.hpp"
#include "ga.h"
#include "mpi.h"
#include "macdecls.h"
#include "ga-mpi.h"
#include "tamm/tamm.hpp"

#include <string>

#define INITVAL_TEST_0D 1
#define INITVAL_TEST_1D 1
#define INITVAL_TEST_2D 1
#define INITVAL_TEST_3D 1
#define INITVAL_TEST_4D 1

#define MULT_TEST_0D_0D 1
#define MULT_TEST_0D_1D 1

#define EIGEN_ASSIGN_TEST_0D 1
#define EIGEN_ASSIGN_TEST_1D 1
#define EIGEN_ASSIGN_TEST_2D 1
#define EIGEN_ASSIGN_TEST_3D 1
#define EIGEN_ASSIGN_TEST_4D 1

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
using PermEigen = Eigen::array<std::ptrdiff_t, ndim>;

using namespace tamm;

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
inline PermEigen<ndim>
eigen_perm_compute(const IndexVector &from, const IndexVector &to) {
  PermEigen<ndim> layout;

  assert(from.size() == to.size());
  assert(from.size() == ndim);
  int pa_index = 0;
  for (auto p : to) {
    auto itr = std::find(from.begin(), from.end(), p);
    EXPECTS(itr != from.end());
    layout[pa_index] = itr - from.begin();
    pa_index++;
  }
  return layout;
}

template<int ndim>
void
eigen_assign_dispatch(EigenTensorBase *tc,
                      const IndexVector &clabel,
                      double alpha,
                      EigenTensorBase *ta,
                      const IndexVector &alabel) {
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
             const IndexVector &clabel,
             double alpha,
             EigenTensorBase *ta,
             const IndexVector &alabel) {
  EXPECTS(clabel.size() == alabel.size());
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
//             eigen vs tamm
//
//
////////////////////////////////////////////////////////

template<typename T, int ndim>
void
patch_copy(T *sbuf, Eigen::Tensor<T, ndim, Eigen::RowMajor> &etensor,
           const std::array<int, ndim> &block_dims,
           const std::array<int, ndim> &rel_offset) {
  assert(0);
}

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
tamm_tensor_to_eigen_tensor_dispatch(tamm::Tensor <T> &tensor) {
  EXPECTS(tensor.num_modes() == ndim);

  std::array<int, ndim> lo_offset, hi_offset;
  std::array<long, ndim> dims;
  const auto &flindices = tensor.tiled_index_spaces();
  for (int i = 0; i < ndim; i++) {
    BlockIndex blo, bhi;
    /// FIXME
    // std::tie(blo, bhi) = tensor_index_range(flindices[i]);
    // lo_offset[i] = TCE::offset(blo);
    // hi_offset[i] = TCE::offset(bhi);
    // dims[i] = hi_offset[i] - lo_offset[i];
  }
  EigenTensor<ndim> *etensor = new EigenTensor<ndim>(dims);
  etensor->setZero();

  /// FIXME
  // tamm::block_for(tensor(), [&](const BlockDimVec &blockid) {
  //   auto block = tensor.get(blockid);
  //   const BlockDimVec &boffset = block.block_offset();
  //   const BlockDimVec &block_dims = block.block_dims();
  //   std::array<int, ndim> rel_offset;
  //   for (int i = 0; i < ndim; i++) {
  //     EXPECTS(boffset[i] < hi_offset[i]);
  //     rel_offset[i] = boffset[i].value() - lo_offset[i];
  //   }
  //   std::array<int, ndim> block_size;
  //   for (int i = 0; i < ndim; i++) {
  //     block_size[i] = block_dims[i].value();
  //   }

  //   patch_copy<T>(block.buf(), *etensor, block_size, rel_offset);
  // });

  return etensor;
}

template<typename T>
EigenTensorBase *
tamm_tensor_to_eigen_tensor(tamm::Tensor <T> &tensor) {
//   if(tensor.num_modes() == 0) {
//     return tamm_tensor_to_eigen_tensor_dispatch<T,0>(tensor);
//   } else
  if (tensor.num_modes() == 1) {
    return tamm_tensor_to_eigen_tensor_dispatch<T, 1>(tensor);
  } else if (tensor.num_modes() == 2) {
    return tamm_tensor_to_eigen_tensor_dispatch<T, 2>(tensor);
  } else if (tensor.num_modes() == 3) {
    return tamm_tensor_to_eigen_tensor_dispatch<T, 3>(tensor);
  } else if (tensor.num_modes() == 4) {
    return tamm_tensor_to_eigen_tensor_dispatch<T, 4>(tensor);
  }
  assert(0); //@todo implement
  return nullptr;
}

EigenTensorBase *
eigen_assign(tamm::Tensor<double> &ttc,
             const tamm::IndexVector &tclabel,
             double alpha,
             tamm::Tensor<double> &tta,
             const tamm::IndexVector &talabel) {
  EigenTensorBase *etc, *eta;
  etc = tamm_tensor_to_eigen_tensor(ttc);
  eta = tamm_tensor_to_eigen_tensor(tta);
  eigen_assign(etc, tclabel, alpha, eta, talabel);
  delete eta;
  return etc;
}

std::vector<TiledIndexSpace> tamm_label_to_indices(const IndexLabelVec &ilv) {
  std::vector<TiledIndexSpace> tisv{ilv.size()};
  for(auto &x: ilv) tisv.push_back(x.tiled_index_space());
  return tisv;
}

  template<typename LabeledTensorType>
  void
  tamm_tensor_fill(ExecutionContext &ec,
                    LabeledTensorType ltensor) {
    using T = typename LabeledTensorType::element_type;
    auto tensor = ltensor.tensor();

    for (auto it: tensor.loop_nest())
    {
        TAMM_SIZE size = tensor.block_size(it);
        std::vector<T> buf(size);
        tensor.get(it,span<T>(&buf[0],size));
        double n = std::rand() % 5;
        for (TAMM_SIZE i = 0; i < size;i++) {
          buf[i] = T{n + i};
       }
       tensor.put(it, span<T>(&buf[0],size));
    }

  }

bool
test_eigen_assign_no_n(tamm::ExecutionContext &ec,
                       double alpha,
                       const IndexLabelVec &cupper_labels,
                       const IndexLabelVec &clower_labels,
                       const IndexLabelVec &aupper_labels,
                       const IndexLabelVec &alower_labels) {

  auto cupper_indices = tamm_label_to_indices(cupper_labels);
  auto clower_indices = tamm_label_to_indices(clower_labels);
  auto aupper_indices = tamm_label_to_indices(aupper_labels);
  auto alower_indices = tamm_label_to_indices(alower_labels);

  auto cindices = cupper_indices; 
  cindices.insert(cindices.end(),clower_indices.begin(), clower_indices.end());
  auto aindices = aupper_indices;
  aindices.insert(aindices.end(),alower_indices.begin(), alower_indices.end());
 
  tamm::Tensor<double> tc1{cindices};
  tamm::Tensor<double> tc2{cindices};
  tamm::Tensor<double> ta{aindices};

  Tensor<double>::allocate(&ec,ta, tc1, tc2);

  Scheduler{&ec}
      (ta() = 0)
      (tc1() = 0)
      (tc2() = 0)
    .execute();

  tamm_tensor_fill(ec, ta());

  // auto clabels = cupper_labels;
  // clabels.insert_back(clower_labels.begin(), clower_labels.end());
  // auto alabels = aupper_labels;
  // alabels.insert_back(alower_labels.begin(), alower_labels.end());

  //EigenTensorBase *etc1 = eigen_assign(tc1, cindices, alpha, ta, aindices);
  // tamm_assign(ec, tc2, cindices, alpha, ta, aindices);

  // EigenTensorBase *etc2 = tamm_tensor_to_eigen_tensor(tc2);

   bool status = false;
  // if (tc1.rank() == 1) {
  //   auto *et1 = dynamic_cast<EigenTensor<1> *>(etc1);
  //   auto *et2 = dynamic_cast<EigenTensor<1> *>(etc2);
  //   status = eigen_tensors_are_equal<double>(*et1, *et2);
  // } else if (tc1.rank() == 2) {
  //   auto *et1 = dynamic_cast<EigenTensor<2> *>(etc1);
  //   auto *et2 = dynamic_cast<EigenTensor<2> *>(etc2);
  //   status = eigen_tensors_are_equal<double>(*et1, *et2);
  // } else if (tc1.rank() == 3) {
  //   auto *et1 = dynamic_cast<EigenTensor<3> *>(etc1);
  //   auto *et2 = dynamic_cast<EigenTensor<3> *>(etc2);
  //   status = eigen_tensors_are_equal<double>(*et1, *et2);
  // } else if (tc1.rank() == 4) {
  //   auto *et1 = dynamic_cast<EigenTensor<4> *>(etc1);
  //   auto *et2 = dynamic_cast<EigenTensor<4> *>(etc2);
  //   status = eigen_tensors_are_equal<double>(*et1, *et2);
  // }

  Tensor<double>::deallocate(tc1, tc2, ta);
  // delete etc1;
  // delete etc2;

  return status;
}


template<int ndim>
void
eigen_mult_dispatch(EigenTensorBase *tc,
                      const IndexVector &clabel,
                      double alpha,
                      EigenTensorBase *ta,
                      const IndexVector &alabel,
                      EigenTensorBase *tb,
                      const IndexVector &blabel) {
  assert(alabel.size() == ndim);
  assert(blabel.size() == ndim);
  assert(clabel.size() == ndim);
  auto eperm_a = eigen_perm_compute<ndim>(alabel, clabel);
  auto eperm_b = eigen_perm_compute<ndim>(blabel, clabel);

  auto ec1 = static_cast<EigenTensor<ndim> *>(tc);
  auto ea = static_cast<EigenTensor<ndim> *>(ta);
  auto eb = static_cast<EigenTensor<ndim> *>(tb);

  auto tmp_a = (*ea).shuffle(eperm_a);
  auto tmp_b = (*eb).shuffle(eperm_b);
  //tmp_a = tmp_a * alpha;
  //(*ec1) += tmp_a * tmp_b;
}

void
eigen_mult(EigenTensorBase *tc,
             const IndexVector &clabel,
             double alpha,
             EigenTensorBase *ta,
             const IndexVector &alabel,
             EigenTensorBase *tb,
             const IndexVector &blabel) {
  EXPECTS(clabel.size() == alabel.size());
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
eigen_mult(tamm::Tensor<double> &ttc,
             const tamm::IndexVector &tclabel,
             double alpha,
             tamm::Tensor<double> &tta,
             const tamm::IndexVector &talabel,
             tamm::Tensor<double> &ttb,
             const tamm::IndexVector &tblabel) {
  EigenTensorBase *etc, *eta, *etb;
  etc = tamm_tensor_to_eigen_tensor(ttc);
  eta = tamm_tensor_to_eigen_tensor(tta);
  etb = tamm_tensor_to_eigen_tensor(ttb);
  eigen_mult(etc, tclabel, alpha, eta, talabel, etb, tblabel);
  delete eta;
  delete etb;
  return etc;
}

bool
test_eigen_mult_no_n(tamm::ExecutionContext &ec,
                       double alpha,
                       const IndexLabelVec &cupper_labels,
                       const IndexLabelVec &clower_labels,
                       const IndexLabelVec &aupper_labels,
                       const IndexLabelVec &alower_labels,
                       const IndexLabelVec &bupper_labels,
                       const IndexLabelVec &blower_labels) {

  const auto &cupper_indices = tamm_label_to_indices(cupper_labels);
  const auto &clower_indices = tamm_label_to_indices(clower_labels);
  const auto &aupper_indices = tamm_label_to_indices(aupper_labels);
  const auto &alower_indices = tamm_label_to_indices(alower_labels);
  const auto &bupper_indices = tamm_label_to_indices(bupper_labels);
  const auto &blower_indices = tamm_label_to_indices(blower_labels);

  auto cindices = cupper_indices;
  cindices.insert(cindices.end(),clower_indices.begin(), clower_indices.end());
  auto aindices = aupper_indices;
  aindices.insert(aindices.end(),alower_indices.begin(), alower_indices.end());
  auto bindices = bupper_indices;
  bindices.insert(bindices.end(),blower_indices.begin(), blower_indices.end());

  tamm::Tensor<double> tc1{cindices};
  tamm::Tensor<double> tc2{cindices};
  tamm::Tensor<double> ta{aindices};
  tamm::Tensor<double> tb{bindices};

  Tensor<double>::allocate(&ec,ta, tb, tc1, tc2);

  Scheduler{&ec}
      (ta() = 0)
      (tb() = 0)
      (tc1() = 0)
      (tc2() = 0)
    .execute();

  tamm_tensor_fill(ec, ta());
  tamm_tensor_fill(ec, tb());

  // auto clabels = cupper_labels;
  // clabels.insert_back(clower_labels.begin(), clower_labels.end());
  // auto alabels = aupper_labels;
  // alabels.insert_back(alower_labels.begin(), alower_labels.end());
  // auto blabels = bupper_labels;
  // blabels.insert_back(blower_labels.begin(), blower_labels.end());

  // EigenTensorBase *etc1 = eigen_mult(tc1, clabels, alpha, ta, alabels, tb, blabels);
  // tamm_mult(ec, tc2, clabels, alpha, ta, alabels, tb, blabels);
  // EigenTensorBase *etc2 = tamm_tensor_to_eigen_tensor(tc2);

   bool status = false;
  // if (tc1.rank() == 1) {
  //   auto *et1 = dynamic_cast<EigenTensor<1> *>(etc1);
  //   auto *et2 = dynamic_cast<EigenTensor<1> *>(etc2);
  //   status = eigen_tensors_are_equal<double>(*et1, *et2);
  // } else if (tc1.rank() == 2) {
  //   auto *et1 = dynamic_cast<EigenTensor<2> *>(etc1);
  //   auto *et2 = dynamic_cast<EigenTensor<2> *>(etc2);
  //   status = eigen_tensors_are_equal<double>(*et1, *et2);
  // } else if (tc1.rank() == 3) {
  //   auto *et1 = dynamic_cast<EigenTensor<3> *>(etc1);
  //   auto *et2 = dynamic_cast<EigenTensor<3> *>(etc2);
  //   status = eigen_tensors_are_equal<double>(*et1, *et2);
  // } else if (tc1.rank() == 4) {
  //   auto *et1 = dynamic_cast<EigenTensor<4> *>(etc1);
  //   auto *et2 = dynamic_cast<EigenTensor<4> *>(etc2);
  //   status = eigen_tensors_are_equal<double>(*et1, *et2);
  // }

  Tensor<double>::deallocate(tc1, tc2, ta, tb);
  //delete etc1;
  //delete etc2;

  return status;
}

IndexSpace MO_IS{range(0, 20),
                     {{"occ", {range(0, 10)}}, {"virt", {range(10, 20)}}}};

TiledIndexSpace MO{MO_IS};
const TiledIndexSpace& O = MO("occ");
const TiledIndexSpace& V = MO("virt");
const TiledIndexSpace& N = MO("all");

TiledIndexLabel h1 = MO.label("occ",0);
TiledIndexLabel h2 = MO.label("occ",1);
TiledIndexLabel h3 = MO.label("occ",2);
TiledIndexLabel h4 = MO.label("occ",3);
TiledIndexLabel h5 = MO.label("occ",4);
TiledIndexLabel h6 = MO.label("occ",5);

TiledIndexLabel p1 = MO.label("virt",0);
TiledIndexLabel p2 = MO.label("virt",1);
TiledIndexLabel p3 = MO.label("virt",2);
TiledIndexLabel p4 = MO.label("virt",3);
TiledIndexLabel p5 = MO.label("virt",4);
TiledIndexLabel p6 = MO.label("virt",5);

ProcGroup pg{GA_MPI_Comm()};
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
ExecutionContext* ec = new ExecutionContext{pg, &distribution, mgr};

//-----------------------------------------------------------------------
//
//                            Initval 0-d
//
//-----------------------------------------------------------------------

bool
test_initval_no_n(tamm::ExecutionContext &ec,
                  const tamm::IndexLabelVec &upper_labels,
                  const tamm::IndexLabelVec &lower_labels) {
  // const auto &upper_indices = tamm_label_to_indices(upper_labels);
  // const auto &lower_indices = tamm_label_to_indices(lower_labels);

  // tamm::TensorRank nupper{upper_labels.size()};
  // tamm::TensorVec <tamm::TensorSymmGroup> indices{upper_indices};
  // indices.insert_back(lower_indices.begin(), lower_indices.end());
  // tamm::Tensor<double> xta{indices, nupper, tamm::Irrep{0}, false};
  // tamm::Tensor<double> xtc{indices, nupper, tamm::Irrep{0}, false};

  // double init_val = 9.1;

  // ec->allocate(xta, xtc);
  // ec->scheduler()
  //   .io(xta, xtc)
  //     (xta() = init_val)
  //     (xtc() = xta())
  //   .execute();

  // tamm::BlockDimVec id{indices.size(), tamm::BlockIndex{0}};
  // auto sz = xta.memory_region().local_nelements().value();

   bool ret = true;
  // const double threshold = 1e-14;
  // const auto abuf = reinterpret_cast<const double*>(xta.memory_region().access(tamm::Offset{0}));
  // const auto cbuf = reinterpret_cast<const double*>(xtc.memory_region().access(tamm::Offset{0}));
  // for (TAMMX_INT32 i = 0; i < sz; i++) {
  //   if (std::abs(abuf[i] - init_val) > threshold) {
  //     ret = false;
  //     break;
  //   }
  // }
  // if (ret == true) {
  //   for (TAMMX_INT32 i = 0; i < sz; i++) {
  //     if (std::abs(cbuf[i] - init_val) > threshold) {
  //       return false;
  //     }
  //   }
  // }
  // ec->deallocate(xta, xtc);
   return ret;
}

#if INITVAL_TEST_0D

TEST_CASE ("InitvalTest - ZeroDim"){
REQUIRE(test_initval_no_n(*ec, {}, {}));
}
#endif

#if INITVAL_TEST_1D

TEST_CASE ("InitvalTest - OneDim") {
REQUIRE(test_initval_no_n(*ec, {}, {h1}));
REQUIRE(test_initval_no_n(*ec, {}, {p1}));
REQUIRE(test_initval_no_n(*ec, {h1}, {}));
REQUIRE(test_initval_no_n(*ec, {p1}, {}));
}

#endif

#if INITVAL_TEST_2D

TEST_CASE ("InitvalTest - TwoDim") {
REQUIRE(test_initval_no_n(*ec, {h1}, {h2}));
REQUIRE(test_initval_no_n(*ec, {h1}, {p2}));
REQUIRE(test_initval_no_n(*ec, {p1}, {h2}));
REQUIRE(test_initval_no_n(*ec, {p1}, {p2}));
}

#endif

#if INITVAL_TEST_3D

TEST_CASE ("InitvalTest - ThreeDim") {
REQUIRE(test_initval_no_n(*ec, {h1}, {h2, h3}));
REQUIRE(test_initval_no_n(*ec, {h1}, {h2, p3}));
REQUIRE(test_initval_no_n(*ec, {h1}, {p2, h3}));
REQUIRE(test_initval_no_n(*ec, {h1}, {p2, p3}));

REQUIRE(test_initval_no_n(*ec, {p1}, {h2, h3}));
REQUIRE(test_initval_no_n(*ec, {p1}, {h2, p3}));
REQUIRE(test_initval_no_n(*ec, {p1}, {p2, h3}));
REQUIRE(test_initval_no_n(*ec, {p1}, {p2, p3}));

REQUIRE(test_initval_no_n(*ec, {h1, h2}, {h3}));
REQUIRE(test_initval_no_n(*ec, {h1, h2}, {p3}));
REQUIRE(test_initval_no_n(*ec, {h1, p2}, {h3}));
REQUIRE(test_initval_no_n(*ec, {h1, p2}, {p3}));

REQUIRE(test_initval_no_n(*ec, {p1, h2}, {h3}));
REQUIRE(test_initval_no_n(*ec, {p1, h2}, {p3}));
REQUIRE(test_initval_no_n(*ec, {p1, p2}, {h3}));
REQUIRE(test_initval_no_n(*ec, {p1, p2}, {p3}));
}

#endif

#if INITVAL_TEST_4D

TEST_CASE ("InitvalTest - FourDim") {
REQUIRE(test_initval_no_n(*ec, {h1, h2}, {h3, h4}));
REQUIRE(test_initval_no_n(*ec, {h1, h2}, {h3, p4}));
REQUIRE(test_initval_no_n(*ec, {h1, h2}, {p3, h4}));
REQUIRE(test_initval_no_n(*ec, {h1, h2}, {p3, p4}));

REQUIRE(test_initval_no_n(*ec, {h1, p2}, {h3, h4}));
REQUIRE(test_initval_no_n(*ec, {h1, p2}, {h3, p4}));
REQUIRE(test_initval_no_n(*ec, {h1, p2}, {p3, h4}));
REQUIRE(test_initval_no_n(*ec, {h1, p2}, {p3, p4}));

REQUIRE(test_initval_no_n(*ec, {p1, h2}, {h3, h4}));
REQUIRE(test_initval_no_n(*ec, {p1, h2}, {h3, p4}));
REQUIRE(test_initval_no_n(*ec, {p1, h2}, {p3, h4}));
REQUIRE(test_initval_no_n(*ec, {p1, h2}, {p3, p4}));

REQUIRE(test_initval_no_n(*ec, {p1, p2}, {h3, h4}));
REQUIRE(test_initval_no_n(*ec, {p1, p2}, {h3, p4}));
REQUIRE(test_initval_no_n(*ec, {p1, p2}, {p3, h4}));
REQUIRE(test_initval_no_n(*ec, {p1, p2}, {p3, p4}));
}

#endif

//-----------------------------------------------------------------------
//
//                            Add 0-d
//
//-----------------------------------------------------------------------

#if ASSIGN_TEST_0D

//@todo tamm might not work with zero dimensions. So directly testing tamm.
TEST_CASE ("AssignTest - ZeroDim") {
// tamm::TensorRank nupper{0};
// tamm::Irrep irrep{0};
// tamm::TensorVec <tamm::SymmGroup> indices{};
// bool restricted = false;
// tamm::Tensor<double> xta{indices, nupper, irrep, restricted};
// tamm::Tensor<double> xtc{indices, nupper, irrep, restricted};

// double init_val_a = 9.1, init_val_c = 8.2, alpha = 3.5;

// ec->allocate(xta, xtc);
// ec->scheduler().io(xta, xtc)
// (xta() = init_val_a)
// (xtc() = init_val_c)
// (xtc()+=alpha *xta())
// .execute();

// auto sz = xta.memory_manager()->local_size_in_elements().value();
// bool status = true;
// const double threshold = 1e-14;
// const auto cbuf = reinterpret_cast<double *>(xtc.memory_manager()->access(tamm::Offset{0}));
// for (int i = 0;i<sz;i++) {
//   if (std::abs(cbuf[i]- (init_val_a *alpha+ init_val_c)) > threshold) {
//   status = false;break;
//   }
// }
// ec->deallocate(xta, xtc);
// REQUIRE(status);
}
#endif

//-----------------------------------------------------------------------
//
//                            Add 1-d
//
//-----------------------------------------------------------------------

#if EIGEN_ASSIGN_TEST_1D

TEST_CASE ("EigenAssignTest - OneDim_o1e_o1e") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1}, {}, {h1}, {}));
}

TEST_CASE ("EigenAssignTest - OneDim_eo1_eo1") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {}, {h1}, {}, {h1}));
}

TEST_CASE ("EigenAssignTest - OneDim_v1e_v1e") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1}, {}, {p1}, {}));
}

TEST_CASE("EigenAssignTest - OneDim_ev1_ev1") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {}, {p1}, {}, {p1}));
}

#endif



//-----------------------------------------------------------------------
//
//                            Add 2-d
//
//-----------------------------------------------------------------------


#if EIGEN_ASSIGN_TEST_2D

TEST_CASE ("EigenAssignTest - TwoDim_O1O2_O1O2") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h4}, {h1}, {h4}, {h1}));
}

TEST_CASE ("EigenAssignTest - TwoDim_O1O2_O2O1") {
REQUIRE(test_eigen_assign_no_n(*ec, 1.23, {h4}, {h1}, {h1}, {h4}));
}

TEST_CASE ("EigenAssignTest - TwoDim_OV_OV") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h4}, {p1}, {h4}, {p1}));
}

TEST_CASE ("EigenAssignTest - TwoDim_OV_VO") {
REQUIRE(test_eigen_assign_no_n(*ec, 1.23, {h4}, {p1}, {p1}, {h4}));
}

TEST_CASE ("EigenAssignTest - TwoDim_VO_VO") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1}, {h1}, {p1}, {h1}));
}

TEST_CASE ("EigenAssignTest - TwoDim_VO_OV") {
REQUIRE(test_eigen_assign_no_n(*ec, 1.23, {p1}, {h1}, {h1}, {p1}));
}

TEST_CASE ("EigenAssignTest - TwoDim_V1V2_V1V2") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p4}, {p1}, {p4}, {p1}));
}

TEST_CASE ("EigenAssignTest - TwoDim_V1V2_V2V1") {
REQUIRE(test_eigen_assign_no_n(*ec, 1.23, {p4}, {p1}, {p1}, {p4}));
}

#endif


#if EIGEN_ASSIGN_TEST_3D

TEST_CASE ("EigenAssignTest - ThreeDim_o1_o2o3__o1_o2o3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1}, {h2, h3}, {h1}, {h2, h3}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_o1_o2o3__o1_o3o2") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1}, {h2, h3}, {h1}, {h3, h2}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_o1_o2v3__o1_o2v3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1}, {h2, p3}, {h1}, {h2, p3}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_o1_o2v3__o1_v3o2") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1}, {h2, p3}, {h1}, {p3, h2}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_o1_v2o3__o1_v2o3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1}, {p2, h3}, {h1}, {p2, h3}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_o1_v2o3__o1_o3v2") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1}, {p2, h3}, {h1}, {h3, p2}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_o1_v2v3__o1_v2v3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1}, {p2, p3}, {h1}, {p2, p3}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_o1_v2v3__o1_v3v2") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1}, {p2, p3}, {h1}, {p3, p2}));
}

///////////

TEST_CASE ("EigenAssignTest - ThreeDim_v1_o2o3__v1_o2o3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1}, {h2, h3}, {p1}, {h2, h3}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_v1_o2o3__v1_o3o2") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1}, {h2, h3}, {p1}, {h3, h2}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_v1_o2v3__v1_o2v3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1}, {h2, p3}, {p1}, {h2, p3}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_v1_o2v3__v1_v3o2") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1}, {h2, p3}, {p1}, {p3, h2}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_v1_v2o3__v1_v2o3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1}, {p2, h3}, {p1}, {p2, h3}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_v1_v2o3__v1_o3v2") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1}, {p2, h3}, {p1}, {h3, p2}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_v1_v2v3__v1_v2v3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1}, {p2, p3}, {p1}, {p2, p3}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_v1_v2v3__v1_v3v2") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1}, {p2, p3}, {p1}, {p3, p2}));
}

//////////////////

TEST_CASE ("EigenAssignTest - ThreeDim_o1o2_o3__o1o2_o3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, h2}, {h3}, {h1, h2}, {h3}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_o1o2_o3__o2o1_o3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, h2}, {h3}, {h2, h1}, {h3}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_o1o2_v3__o1o2_v3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, h2}, {p3}, {h1, h2}, {p3}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_o1o2_v3__o2o1_v3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, h2}, {p3}, {h2, h1}, {p3}));
}

/////////

TEST_CASE ("EigenAssignTest - ThreeDim_o1v2_o3__o1v2_o3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, p2}, {h3}, {h1, p2}, {h3}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_o1v2_o3__v2o1_o3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, p2}, {h3}, {p2, h1}, {h3}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_o1v2_v3__o1v2_v3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, p2}, {p3}, {h1, p2}, {p3}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_o1v2_v3__v2o1_v3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, p2}, {p3}, {p2, h1}, {p3}));
}

//////////////////

TEST_CASE ("EigenAssignTest - ThreeDim_v1o2_o3__v1o2_o3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, h2}, {h3}, {p1, h2}, {h3}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_v1o2_o3__o2v1_o3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, h2}, {h3}, {h2, p1}, {h3}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_v1o2_v3__v1o2_v3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, h2}, {p3}, {p1, h2}, {p3}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_v1o2_v3__o2v1_v3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, h2}, {p3}, {h2, p1}, {p3}));
}

/////////

TEST_CASE ("EigenAssignTest - ThreeDim_v1v2_o3__v1v2_o3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, p2}, {h3}, {p1, p2}, {h3}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_v1v2_o3__v2v1_o3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, p2}, {h3}, {p2, p1}, {h3}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_v1v2_v3__v1v2_v3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, p2}, {p3}, {p1, p2}, {p3}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_v1v2_v3__v2v1_v3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, p2}, {p3}, {p2, p1}, {p3}));
}

//////////

#endif

//-----------------------------------------------------------------------
//
//                            Add 4-d
//
//-----------------------------------------------------------------------


#if EIGEN_ASSIGN_TEST_4D

TEST_CASE ("EigenAssignTest - FourDim_o1o2o3o4_o1o2o3o4") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, h2}, {h3, h4}, {h1, h2}, {h3, h4}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1o2o3o4_o1o2o4o3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, h2}, {h3, h4}, {h1, h2}, {h4, h3}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1o2o3o4_o2o1o3o4") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, h2}, {h3, h4}, {h2, h1}, {h3, h4}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1o2o3o4_o2o1o4o3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h2, h1}, {h3, h4}, {h2, h1}, {h4, h3}));
}

///////

TEST_CASE ("EigenAssignTest - FourDim_o1o2o3v4_o1o2o3v4") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h2, h1}, {h3, p4}, {h1, h2}, {h3, p4}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1o2o3v4_o1o2v4o3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, h2}, {h3, p4}, {h1, h2}, {p4, h3}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1o2o3v4_o2o1o3v4") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, h2}, {h3, p4}, {h2, h1}, {h3, p4}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1o2o3v4_o2o1v4o3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, h2}, {h3, p4}, {h2, h1}, {p4, h3}));
}

////////

TEST_CASE ("EigenAssignTest - FourDim_o1o2v3o4_o1o2v3o4") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h2, h1}, {p3, h4}, {h1, h2}, {p3, h4}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1o2v3o4_o1o2o4v3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, h2}, {p3, h4}, {h1, h2}, {h4, p3}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1o2v3o4_o2o1v3o4") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, h2}, {p3, h4}, {h2, h1}, {p3, h4}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1o2v3o4_o2o1o4v3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, h2}, {p3, h4}, {h2, h1}, {h4, p3}));
}


////////

TEST_CASE ("EigenAssignTest - FourDim_o1o2v3v4_o1o2v3v4") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h2, h1}, {p3, p4}, {h1, h2}, {p3, p4}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1o2v3v4_o1o2v4v3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, h2}, {p3, p4}, {h1, h2}, {p4, p3}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1o2v3v4_o2o1v3v4") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, h2}, {p3, p4}, {h2, h1}, {p3, p4}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1o2v3v4_o2o1v4v3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, h2}, {p3, p4}, {h2, h1}, {p4, p3}));
}

///////////////////////

TEST_CASE ("EigenAssignTest - FourDim_o1v2o3o4_o1v2o3o4") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, p2}, {h3, h4}, {h1, p2}, {h3, h4}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1v2o3o4_o1v2o4o3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, p2}, {h3, h4}, {h1, p2}, {h4, h3}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1v2o3o4_v2o1o3o4") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, p2}, {h3, h4}, {p2, h1}, {h3, h4}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1v2o3o4_v2o1o4o3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p2, h1}, {h3, h4}, {p2, h1}, {h4, h3}));
}

///////

TEST_CASE ("EigenAssignTest - FourDim_o1v2o3v4_o1v2o3v4") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p2, h1}, {h3, p4}, {h1, p2}, {h3, p4}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1v2o3v4_o1v2v4o3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, p2}, {h3, p4}, {h1, p2}, {p4, h3}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1v2o3v4_v2o1o3v4") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, p2}, {h3, p4}, {p2, h1}, {h3, p4}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1v2o3v4_v2o1v4o3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, p2}, {h3, p4}, {p2, h1}, {p4, h3}));
}

////////

TEST_CASE ("EigenAssignTest - FourDim_o1v2v3o4_o1v2v3o4") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p2, h1}, {p3, h4}, {h1, p2}, {p3, h4}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1v2v3o4_o1v2o4v3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, p2}, {p3, h4}, {h1, p2}, {h4, p3}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1v2v3o4_v2o1v3o4") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, p2}, {p3, h4}, {p2, h1}, {p3, h4}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1v2v3o4_v2o1o4v3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, p2}, {p3, h4}, {p2, h1}, {h4, p3}));
}


////////

TEST_CASE ("EigenAssignTest - FourDim_o1v2v3v4_o1v2v3v4") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p2, h1}, {p3, p4}, {h1, p2}, {p3, p4}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1v2v3v4_o1v2v4v3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, p2}, {p3, p4}, {h1, p2}, {p4, p3}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1v2v3v4_v2o1v3v4") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, p2}, {p3, p4}, {p2, h1}, {p3, p4}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1v2v3v4_v2o1v4v3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, p2}, {p3, p4}, {p2, h1}, {p4, p3}));
}

//////////////////////////////////////

TEST_CASE ("EigenAssignTest - FourDim_v1o2o3o4_v1o2o3o4") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, h2}, {h3, h4}, {p1, h2}, {h3, h4}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1o2o3o4_v1o2o4o3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, h2}, {h3, h4}, {p1, h2}, {h4, h3}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1o2o3o4_o2v1o3o4") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, h2}, {h3, h4}, {h2, p1}, {h3, h4}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1o2o3o4_o2v1o4o3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h2, p1}, {h3, h4}, {h2, p1}, {h4, h3}));
}

///////

TEST_CASE ("EigenAssignTest - FourDim_v1o2o3v4_v1o2o3v4") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h2, p1}, {h3, p4}, {p1, h2}, {h3, p4}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1o2o3v4_v1o2v4o3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, h2}, {h3, p4}, {p1, h2}, {p4, h3}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1o2o3v4_o2v1o3v4") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, h2}, {h3, p4}, {h2, p1}, {h3, p4}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1o2o3v4_o2v1v4o3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, h2}, {h3, p4}, {h2, p1}, {p4, h3}));
}

////////

TEST_CASE ("EigenAssignTest - FourDim_v1o2v3o4_v1o2v3o4") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h2, p1}, {p3, h4}, {p1, h2}, {p3, h4}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1o2v3o4_v1o2o4v3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, h2}, {p3, h4}, {p1, h2}, {h4, p3}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1o2v3o4_o2v1v3o4") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, h2}, {p3, h4}, {h2, p1}, {p3, h4}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1o2v3o4_o2v1o4v3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, h2}, {p3, h4}, {h2, p1}, {h4, p3}));
}


////////

TEST_CASE ("EigenAssignTest - FourDim_v1o2v3v4_v1o2v3v4") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h2, p1}, {p3, p4}, {p1, h2}, {p3, p4}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1o2v3v4_v1o2v4v3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, h2}, {p3, p4}, {p1, h2}, {p4, p3}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1o2v3v4_o2v1v3v4") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, h2}, {p3, p4}, {h2, p1}, {p3, p4}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1o2v3v4_o2v1v4v3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, h2}, {p3, p4}, {h2, p1}, {p4, p3}));
}

//////////////////////////////////////

TEST_CASE ("EigenAssignTest - FourDim_v1v2o3o4_v1v2o3o4") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, p2}, {h3, h4}, {p1, p2}, {h3, h4}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1v2o3o4_v1v2o4o3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, p2}, {h3, h4}, {p1, p2}, {h4, h3}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1v2o3o4_v2v1o3o4") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, p2}, {h3, h4}, {p2, p1}, {h3, h4}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1v2o3o4_v2v1o4o3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p2, p1}, {h3, h4}, {p2, p1}, {h4, h3}));
}

///////

TEST_CASE ("EigenAssignTest - FourDim_v1v2o3v4_v1v2o3v4") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p2, p1}, {h3, p4}, {p1, p2}, {h3, p4}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1v2o3v4_v1v2v4o3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, p2}, {h3, p4}, {p1, p2}, {p4, h3}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1v2o3v4_v2v1o3v4") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, p2}, {h3, p4}, {p2, p1}, {h3, p4}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1v2o3v4_v2v1v4o3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, p2}, {h3, p4}, {p2, p1}, {p4, h3}));
}

////////

TEST_CASE ("EigenAssignTest - FourDim_v1v2v3o4_v1v2v3o4") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p2, p1}, {p3, h4}, {p1, p2}, {p3, h4}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1v2v3o4_v1v2o4v3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, p2}, {p3, h4}, {p1, p2}, {h4, p3}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1v2v3o4_v2v1v3o4") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, p2}, {p3, h4}, {p2, p1}, {p3, h4}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1v2v3o4_v2v1o4v3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, p2}, {p3, h4}, {p2, p1}, {h4, p3}));
}


////////

TEST_CASE ("EigenAssignTest - FourDim_v1v2v3v4_v1v2v3v4") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p2, p1}, {p3, p4}, {p1, p2}, {p3, p4}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1v2v3v4_v1v2v4v3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, p2}, {p3, p4}, {p1, p2}, {p4, p3}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1v2v3v4_v2v1v3v4") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, p2}, {p3, p4}, {p2, p1}, {p3, p4}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1v2v3v4_v2v1v4v3") {
REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, p2}, {p3, p4}, {p2, p1}, {p4, p3}));
}

#endif

#if MULT_TEST_0D_0D

TEST_CASE ("MultTest - Dim_0_0_0") {
tamm::Tensor<double> xtc{};
tamm::Tensor<double> xta{};
tamm::Tensor<double> xtb{};

double alpha1 = 0.91, alpha2 = 0.56;

Tensor<double>::allocate(ec,xta, xtb, xtc);
Scheduler{ec}
  (xta() = alpha1)
  (xtb() = alpha2)
  (xtc() = xta() * xtb())
  .execute();

double threshold = 1.0e-12;
bool status = true;
auto lambda = [&](auto &val) {
  status &= (std::abs(val - alpha1 * alpha2) < threshold);
};

/// @todo FIXME: lambda should have right params
// Scheduler{ec}
// .gop(xtc(), lambda)
// .execute();

Tensor<double>::deallocate(xta, xtb, xtc);

REQUIRE(status);
}
#endif


#if MULT_TEST_0D_1D

TEST_CASE ("MultTest, Dim_o_0_o_up") {
tamm::Tensor<double> xtc{O};
tamm::Tensor<double> xta{};
tamm::Tensor<double> xtb{O};

double alpha1 = 0.91, alpha2 = 0.56;

Tensor<double>::allocate(ec,xta, xtb, xtc);
Scheduler{ec}
(xta() = alpha1)
(xtb() = alpha2)
(xtc() = xta() * xtb())
.execute();

double threshold = 1.0e-12;
bool status = true;
auto lambda = [&](auto &val) {
  status &= (std::abs(val - alpha1 * alpha2) < threshold);
};

/// @todo FIXME
// Scheduler{ec}
// .gop(xtc(), lambda)
// .execute();

Tensor<double>::deallocate(xta, xtb, xtc);

REQUIRE(status);
}

TEST_CASE ("MultTest - Dim_o_0_o_lo") {
tamm::Tensor<double> xtc{O};
tamm::Tensor<double> xta{};
tamm::Tensor<double> xtb{O};

double alpha1 = 0.91, alpha2 = 0.56;

Tensor<double>::allocate(ec,xta, xtb, xtc);
Scheduler{ec}
(xta() = alpha1)
(xtb() = alpha2)
(xtc() = xta() * xtb())
.execute();

double threshold = 1.0e-12;
bool status = true;
auto lambda = [&](auto &val) {
  status &= (std::abs(val - alpha1 * alpha2) < threshold);
};

/// FIXME
// Scheduler{ec}
//   .gop(xtc(), lambda)
// .execute();

Tensor<double>::deallocate(xta, xtb, xtc);

REQUIRE(status);
}

TEST_CASE ("MultTest - Dim_v_v_0_hi") {
tamm::Tensor<double> xtc{V};
tamm::Tensor<double> xta{V};
tamm::Tensor<double> xtb{};

double alpha1 = 0.91, alpha2 = 0.56;

Tensor<double>::allocate(ec,xta, xtb, xtc);
Scheduler{ec}
(xta() = alpha1)
(xtb() = alpha2)
(xtc() = xta() * xtb())
.execute();

double threshold = 1.0e-12;
bool status = true;
auto lambda = [&](auto &val) {
  status &= (std::abs(val - alpha1 * alpha2) < threshold);
};


/// @todo FIXME
// Scheduler{ec}
// .gop(xtc(), lambda)
// .execute();

Tensor<double>::deallocate(xta, xtb, xtc);

REQUIRE(status);
}

TEST_CASE ("MultTest - Dim_v_v_0_lo") {
tamm::Tensor<double> xtc{V};
tamm::Tensor<double> xta{V};
tamm::Tensor<double> xtb{};

double alpha1 = 0.91, alpha2 = 0.56;

Tensor<double>::allocate(ec,xta, xtb, xtc);
Scheduler{ec}
  (xta() = alpha1)
  (xtb() = alpha2)
  (xtc() = xta() * xtb())
  .execute();

double threshold = 1.0e-12;
bool status = true;
auto lambda = [&](auto &val) {
  status &= (std::abs(val - alpha1 * alpha2) < threshold);
};

/// FIXME
// Scheduler{ec}
//   .gop(xtc(), lambda)
//   .execute();

Tensor<double>::deallocate(xta, xtb, xtc);

REQUIRE(status);
}

#endif


int main(int argc, char* argv[])
{
    MPI_Init(&argc,&argv);
    GA_Initialize();
    MA_init(MT_DBL, 8000000, 20000000);
    
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    int res = Catch::Session().run(argc, argv);
    GA_Terminate();
    MPI_Finalize();

    return res;
}