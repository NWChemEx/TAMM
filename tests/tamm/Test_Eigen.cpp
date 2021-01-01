

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest/doctest.h"
#include "ga.h"
#include "mpi.h"
#include "macdecls.h"
#include "ga-mpi.h"
#include "tamm/eigen_utils.hpp"
#include "tamm/tamm.hpp"

#include <string>

#define ASSIGN_TEST_0D 0
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
eigen_perm_compute(const IndexLabelVec &from, const IndexLabelVec &to) {
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
                      const IndexLabelVec &clabel,
                      double alpha,
                      EigenTensorBase *ta,
                      const IndexLabelVec &alabel) {
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
             const IndexLabelVec &clabel,
             double alpha,
             EigenTensorBase *ta,
             const IndexLabelVec &alabel) {
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
tamm_tensor_to_eigen_tensor_dispatch(Tensor <T> &tensor) {
  EXPECTS(tensor.num_modes() == ndim);

  std::array<long, ndim> dims;
  const auto &tindices = tensor.tiled_index_spaces();
  for (int i = 0; i < ndim; i++) {
    dims[i] = tindices[i].index_space().num_indices();
  }
  EigenTensor<ndim> *etensor = new EigenTensor<ndim>(dims);
  etensor->setZero();

  /// FIXME
  // block_for(tensor(), [&](const BlockDimVec &blockid) {
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

    
      for (auto it: tensor.loop_nest())
    {
        const TAMM_SIZE size = tensor.block_size(it);
        
        std::vector<T> buf(size);
        tensor.get(it,buf);

        std::array<int, ndim> block_size;
        std::array<int, ndim> rel_offset;
        auto &tiss = tensor.tiled_index_spaces();
        auto block_dims = tensor.block_dims(it);
        for (auto i = 0; i < ndim; i++) {
          block_size[i] = block_dims[i];
          rel_offset[i] = tiss[i].tile_offset(it[i]);
        }
 
        patch_copy<T>(buf.data(), *etensor, block_size, rel_offset);
    }

  return etensor;
}

template<typename T>
EigenTensorBase *
tamm_tensor_to_eigen_tensor(Tensor <T> &tensor) {
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

template<typename T>
void
tamm_assign(ExecutionContext &ec,
             Tensor <T> &tc,
             const IndexLabelVec &clabel,
             double alpha,
             Tensor <T> &ta,
             const IndexLabelVec &alabel) {
  auto &al = alabel;
  auto &cl = clabel;
  
  Scheduler{ec}
      ((tc)(cl) += alpha * (ta)(al))
    .execute();
}

EigenTensorBase *
eigen_assign(Tensor<double> &ttc,
             const IndexLabelVec &tclabel,
             double alpha,
             Tensor<double> &tta,
             const IndexLabelVec &talabel) {
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
        const TAMM_SIZE size = tensor.block_size(it);
        std::vector<T> buf(size);
        tensor.get(it, buf);
        double n = std::rand() % 5;
        for (TAMM_SIZE i = 0; i < size;i++) {
          buf[i] = T{n + i};
       }
       tensor.put(it, buf);
    }

  }

bool
test_eigen_assign_no_n(ExecutionContext &ec,
                       double alpha,
                       const IndexLabelVec &cupper_labels,
                       const IndexLabelVec &clower_labels,
                       const IndexLabelVec &aupper_labels,
                       const IndexLabelVec &alower_labels) {

  // auto cupper_indices = tamm_label_to_indices(cupper_labels);
  // auto clower_indices = tamm_label_to_indices(clower_labels);
  // auto aupper_indices = tamm_label_to_indices(aupper_labels);
  // auto alower_indices = tamm_label_to_indices(alower_labels);

  // auto cindices = cupper_indices; 
  // cindices.insert(cindices.end(),clower_indices.begin(), clower_indices.end());
  // auto aindices = aupper_indices;
  // aindices.insert(aindices.end(),alower_indices.begin(), alower_indices.end());

  auto clabels = cupper_labels;
  clabels.insert(clabels.end(),clower_labels.begin(), clower_labels.end());
  auto alabels = aupper_labels;
  alabels.insert(alabels.end(),alower_labels.begin(), alower_labels.end());
 
  Tensor<double> tc1{clabels};
  Tensor<double> tc2{clabels};
  Tensor<double> ta{alabels};

  Tensor<double>::allocate(&ec,ta, tc1, tc2);

  Scheduler{ec}
      (ta() = 0)
      (tc1() = 0)
      (tc2() = 0)
    .execute();

  tamm_tensor_fill(ec, ta());



  EigenTensorBase *etc1 = eigen_assign(tc1, clabels, alpha, ta, alabels);
  tamm_assign(ec, tc2, clabels, alpha, ta, alabels);

  EigenTensorBase *etc2 = tamm_tensor_to_eigen_tensor(tc2);

   bool status = false;
  if (tc1.num_modes() == 1) {
    auto *et1 = dynamic_cast<EigenTensor<1> *>(etc1);
    auto *et2 = dynamic_cast<EigenTensor<1> *>(etc2);
    status = eigen_tensors_are_equal<double>(*et1, *et2);
  } else if (tc1.num_modes() == 2) {
    auto *et1 = dynamic_cast<EigenTensor<2> *>(etc1);
    auto *et2 = dynamic_cast<EigenTensor<2> *>(etc2);
    status = eigen_tensors_are_equal<double>(*et1, *et2);
  } else if (tc1.num_modes() == 3) {
    auto *et1 = dynamic_cast<EigenTensor<3> *>(etc1);
    auto *et2 = dynamic_cast<EigenTensor<3> *>(etc2);
    status = eigen_tensors_are_equal<double>(*et1, *et2);
  } else if (tc1.num_modes() == 4) {
    auto *et1 = dynamic_cast<EigenTensor<4> *>(etc1);
    auto *et2 = dynamic_cast<EigenTensor<4> *>(etc2);
    status = eigen_tensors_are_equal<double>(*et1, *et2);
  }

  Tensor<double>::deallocate(tc1, tc2, ta);
  delete etc1;
  delete etc2;

  return status;
}



template<typename T>
void
tamm_mult(ExecutionContext &ec,
           Tensor <T> &tc,
           const IndexLabelVec &clabel,
           double alpha,
           Tensor <T> &ta,
           const IndexLabelVec &alabel,
           Tensor <T> &tb,
           const IndexLabelVec &blabel) {

  auto &al = alabel;
  auto &bl = blabel;
  auto &cl = clabel;

  Scheduler{ec}
      ((tc)() = 0.0)
      ((tc)(cl) += alpha * (ta)(al) * (tb)(bl))
    .execute();

}


template<int ndim>
void
eigen_mult_dispatch(EigenTensorBase *tc,
                      const IndexLabelVec &clabel,
                      double alpha,
                      EigenTensorBase *ta,
                      const IndexLabelVec &alabel,
                      EigenTensorBase *tb,
                      const IndexLabelVec &blabel) {
/// fixme test_eigen_mult_no_n is not used
  // assert(alabel.size() == ndim);
  // assert(blabel.size() == ndim);
  // assert(clabel.size() == ndim);
  // auto eperm_a = eigen_perm_compute<ndim>(alabel, clabel);
  // auto eperm_b = eigen_perm_compute<ndim>(blabel, clabel);

  //auto ec1 = static_cast<EigenTensor<ndim> *>(tc);
  // auto ea = static_cast<EigenTensor<ndim> *>(ta);
  // auto eb = static_cast<EigenTensor<ndim> *>(tb);

  // auto tmp_a = (*ea).shuffle(eperm_a);
  // auto tmp_b = (*eb).shuffle(eperm_b);
  //tmp_a = tmp_a * alpha;
  //(*ec1) += tmp_a * tmp_b;
}

void
eigen_mult(EigenTensorBase *tc,
             const IndexLabelVec &clabel,
             double alpha,
             EigenTensorBase *ta,
             const IndexLabelVec &alabel,
             EigenTensorBase *tb,
             const IndexLabelVec &blabel) {
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
eigen_mult(Tensor<double> &ttc,
             const IndexLabelVec &tclabel,
             double alpha,
             Tensor<double> &tta,
             const IndexLabelVec &talabel,
             Tensor<double> &ttb,
             const IndexLabelVec &tblabel) {
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
test_eigen_mult_no_n(ExecutionContext &ec,
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

  Tensor<double> tc1{cindices};
  Tensor<double> tc2{cindices};
  Tensor<double> ta{aindices};
  Tensor<double> tb{bindices};

  Tensor<double>::allocate(&ec,ta, tb, tc1, tc2);

  Scheduler{ec}
      (ta() = 0)
      (tb() = 0)
      (tc1() = 0)
      (tc2() = 0)
    .execute();

  tamm_tensor_fill(ec, ta());
  tamm_tensor_fill(ec, tb());

  auto clabels = cupper_labels;
  clabels.insert(clabels.end(),clower_labels.begin(), clower_labels.end());
  auto alabels = aupper_labels;
  alabels.insert(alabels.end(),alower_labels.begin(), alower_labels.end());
  auto blabels = bupper_labels;
  blabels.insert(blabels.end(),blower_labels.begin(), blower_labels.end());

  EigenTensorBase *etc1 = eigen_mult(tc1, clabels, alpha, ta, alabels, tb, blabels);
  tamm_mult(ec, tc2, clabels, alpha, ta, alabels, tb, blabels);
  EigenTensorBase *etc2 = tamm_tensor_to_eigen_tensor(tc2);

   bool status = false;
  if (tc1.num_modes() == 1) {
    auto *et1 = dynamic_cast<EigenTensor<1> *>(etc1);
    auto *et2 = dynamic_cast<EigenTensor<1> *>(etc2);
    status = eigen_tensors_are_equal<double>(*et1, *et2);
  } else if (tc1.num_modes() == 2) {
    auto *et1 = dynamic_cast<EigenTensor<2> *>(etc1);
    auto *et2 = dynamic_cast<EigenTensor<2> *>(etc2);
    status = eigen_tensors_are_equal<double>(*et1, *et2);
  } else if (tc1.num_modes() == 3) {
    auto *et1 = dynamic_cast<EigenTensor<3> *>(etc1);
    auto *et2 = dynamic_cast<EigenTensor<3> *>(etc2);
    status = eigen_tensors_are_equal<double>(*et1, *et2);
  } else if (tc1.num_modes() == 4) {
    auto *et1 = dynamic_cast<EigenTensor<4> *>(etc1);
    auto *et2 = dynamic_cast<EigenTensor<4> *>(etc2);
    status = eigen_tensors_are_equal<double>(*et1, *et2);
  }

  Tensor<double>::deallocate(tc1, tc2, ta, tb);
  delete etc1;
  delete etc2;

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


template<typename T>
bool check_value(LabeledTensor<T> lt, T val) {
    LabelLoopNest loop_nest{lt.labels()};

    for(const auto& itval : loop_nest) {
        const IndexVector blockid = internal::translate_blockid(itval, lt);
        size_t size               = lt.tensor().block_size(blockid);
        std::vector<T> buf(size);
        lt.tensor().get(blockid, buf);
        for(TAMM_SIZE i = 0; i < size; i++) {
            if(std::fabs(buf[i] - val) >= 1.0e-10) {
              return false;
            }
        }
    }
    return true;
}

template<typename T>
bool check_value(Tensor<T>& t, T val) {
    return check_value(t(), val);
}

//-----------------------------------------------------------------------
//
//                            Initval 0-d
//
//-----------------------------------------------------------------------

bool
test_initval_no_n(ExecutionContext &ec,
                  const IndexLabelVec &upper_labels,
                  const IndexLabelVec &lower_labels) {
  const auto &upper_indices = tamm_label_to_indices(upper_labels);
  const auto &lower_indices = tamm_label_to_indices(lower_labels);

  auto indices = upper_labels;
  indices.insert(indices.end(),lower_labels.begin(), lower_labels.end());
  Tensor<double> xta{indices};
  Tensor<double> xtc{indices};

  double init_val = 9.1;

  Tensor<double>::allocate(&ec,xta, xtc);
  Scheduler{ec}
      (xta() = init_val)
      (xtc() = xta())
    .execute();

    bool ret = true;

    ret &= check_value(xta,init_val);
    ret &= check_value(xtc,init_val);

  /// FIXME: Remove old code
  // BlockDimVec id{indices.size(), BlockIndex{0}};
  // auto sz = xta.memory_region().local_nelements().value();
  // const double threshold = 1e-14;
  // const auto abuf = reinterpret_cast<const double*>(xta.memory_region().access(Offset{0}));
  // const auto cbuf = reinterpret_cast<const double*>(xtc.memory_region().access(Offset{0}));
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
  
   Tensor<double>::deallocate(xta, xtc);
   return ret;
}

#if INITVAL_TEST_0D

TEST_CASE ("InitvalTest - ZeroDim"){

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_initval_no_n(*ec, {}, {}));
}
#endif

#if INITVAL_TEST_1D

TEST_CASE ("InitvalTest - OneDim") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_initval_no_n(*ec, {}, {h1}));
REQUIRE(test_initval_no_n(*ec, {}, {p1}));
REQUIRE(test_initval_no_n(*ec, {h1}, {}));
REQUIRE(test_initval_no_n(*ec, {p1}, {}));
}

#endif

#if INITVAL_TEST_2D

TEST_CASE ("InitvalTest - TwoDim") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_initval_no_n(*ec, {h1}, {h2}));
REQUIRE(test_initval_no_n(*ec, {h1}, {p2}));
REQUIRE(test_initval_no_n(*ec, {p1}, {h2}));
REQUIRE(test_initval_no_n(*ec, {p1}, {p2}));
}

#endif

#if INITVAL_TEST_3D

TEST_CASE ("InitvalTest - ThreeDim") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

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

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

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

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};


  Tensor<double> xta{};
  Tensor<double> xtc{};

  double init_val_a = 9.1, init_val_c = 8.2, alpha = 3.5;

  Tensor<double>::allocate(ec, xta, xtc);
  Scheduler{*ec}
  (xta() = init_val_a)
  (xtc() = init_val_c)
  (xtc() += alpha *xta())
  .execute();

  bool status = check_value(xtc,init_val_a *alpha+ init_val_c);
  Tensor<double>::deallocate(xta, xtc);

  // auto sz = xta.memory_manager()->local_size_in_elements().value();
  // bool status = true;
  // const double threshold = 1e-14;
  // const auto cbuf = reinterpret_cast<double *>(xtc.memory_manager()->access(Offset{0}));
  // for (int i = 0;i<sz;i++) {
  //   if (std::abs(cbuf[i]- (init_val_a *alpha+ init_val_c)) > threshold) {
  //   status = false;break;
  //   }
  // }
  REQUIRE(status);

}
#endif

//-----------------------------------------------------------------------
//
//                            Add 1-d
//
//-----------------------------------------------------------------------

#if EIGEN_ASSIGN_TEST_1D

TEST_CASE ("EigenAssignTest - OneDim_o1e_o1e") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1}, {}, {h1}, {}));
}

TEST_CASE ("EigenAssignTest - OneDim_eo1_eo1") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {}, {h1}, {}, {h1}));
}

TEST_CASE ("EigenAssignTest - OneDim_v1e_v1e") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1}, {}, {p1}, {}));
}

TEST_CASE("EigenAssignTest - OneDim_ev1_ev1") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

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

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h4}, {h1}, {h4}, {h1}));
}

TEST_CASE ("EigenAssignTest - TwoDim_O1O2_O2O1") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 1.23, {h4}, {h1}, {h1}, {h4}));
}

TEST_CASE ("EigenAssignTest - TwoDim_OV_OV") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h4}, {p1}, {h4}, {p1}));
}

TEST_CASE ("EigenAssignTest - TwoDim_OV_VO") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 1.23, {h4}, {p1}, {p1}, {h4}));
}

TEST_CASE ("EigenAssignTest - TwoDim_VO_VO") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1}, {h1}, {p1}, {h1}));
}

TEST_CASE ("EigenAssignTest - TwoDim_VO_OV") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 1.23, {p1}, {h1}, {h1}, {p1}));
}

TEST_CASE ("EigenAssignTest - TwoDim_V1V2_V1V2") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p4}, {p1}, {p4}, {p1}));
}

TEST_CASE ("EigenAssignTest - TwoDim_V1V2_V2V1") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 1.23, {p4}, {p1}, {p1}, {p4}));
}

#endif


#if EIGEN_ASSIGN_TEST_3D

TEST_CASE ("EigenAssignTest - ThreeDim_o1_o2o3__o1_o2o3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1}, {h2, h3}, {h1}, {h2, h3}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_o1_o2o3__o1_o3o2") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1}, {h2, h3}, {h1}, {h3, h2}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_o1_o2v3__o1_o2v3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1}, {h2, p3}, {h1}, {h2, p3}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_o1_o2v3__o1_v3o2") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1}, {h2, p3}, {h1}, {p3, h2}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_o1_v2o3__o1_v2o3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1}, {p2, h3}, {h1}, {p2, h3}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_o1_v2o3__o1_o3v2") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1}, {p2, h3}, {h1}, {h3, p2}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_o1_v2v3__o1_v2v3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1}, {p2, p3}, {h1}, {p2, p3}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_o1_v2v3__o1_v3v2") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1}, {p2, p3}, {h1}, {p3, p2}));
}

///////////

TEST_CASE ("EigenAssignTest - ThreeDim_v1_o2o3__v1_o2o3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1}, {h2, h3}, {p1}, {h2, h3}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_v1_o2o3__v1_o3o2") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1}, {h2, h3}, {p1}, {h3, h2}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_v1_o2v3__v1_o2v3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1}, {h2, p3}, {p1}, {h2, p3}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_v1_o2v3__v1_v3o2") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1}, {h2, p3}, {p1}, {p3, h2}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_v1_v2o3__v1_v2o3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1}, {p2, h3}, {p1}, {p2, h3}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_v1_v2o3__v1_o3v2") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1}, {p2, h3}, {p1}, {h3, p2}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_v1_v2v3__v1_v2v3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1}, {p2, p3}, {p1}, {p2, p3}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_v1_v2v3__v1_v3v2") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1}, {p2, p3}, {p1}, {p3, p2}));
}

//////////////////

TEST_CASE ("EigenAssignTest - ThreeDim_o1o2_o3__o1o2_o3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, h2}, {h3}, {h1, h2}, {h3}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_o1o2_o3__o2o1_o3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, h2}, {h3}, {h2, h1}, {h3}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_o1o2_v3__o1o2_v3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, h2}, {p3}, {h1, h2}, {p3}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_o1o2_v3__o2o1_v3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, h2}, {p3}, {h2, h1}, {p3}));
}

/////////

TEST_CASE ("EigenAssignTest - ThreeDim_o1v2_o3__o1v2_o3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, p2}, {h3}, {h1, p2}, {h3}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_o1v2_o3__v2o1_o3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, p2}, {h3}, {p2, h1}, {h3}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_o1v2_v3__o1v2_v3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, p2}, {p3}, {h1, p2}, {p3}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_o1v2_v3__v2o1_v3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, p2}, {p3}, {p2, h1}, {p3}));
}

//////////////////

TEST_CASE ("EigenAssignTest - ThreeDim_v1o2_o3__v1o2_o3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, h2}, {h3}, {p1, h2}, {h3}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_v1o2_o3__o2v1_o3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, h2}, {h3}, {h2, p1}, {h3}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_v1o2_v3__v1o2_v3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, h2}, {p3}, {p1, h2}, {p3}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_v1o2_v3__o2v1_v3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, h2}, {p3}, {h2, p1}, {p3}));
}

/////////

TEST_CASE ("EigenAssignTest - ThreeDim_v1v2_o3__v1v2_o3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, p2}, {h3}, {p1, p2}, {h3}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_v1v2_o3__v2v1_o3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, p2}, {h3}, {p2, p1}, {h3}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_v1v2_v3__v1v2_v3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, p2}, {p3}, {p1, p2}, {p3}));
}

TEST_CASE ("EigenAssignTest - ThreeDim_v1v2_v3__v2v1_v3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

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

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, h2}, {h3, h4}, {h1, h2}, {h3, h4}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1o2o3o4_o1o2o4o3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, h2}, {h3, h4}, {h1, h2}, {h4, h3}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1o2o3o4_o2o1o3o4") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, h2}, {h3, h4}, {h2, h1}, {h3, h4}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1o2o3o4_o2o1o4o3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h2, h1}, {h3, h4}, {h2, h1}, {h4, h3}));
}

///////

TEST_CASE ("EigenAssignTest - FourDim_o1o2o3v4_o1o2o3v4") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h2, h1}, {h3, p4}, {h1, h2}, {h3, p4}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1o2o3v4_o1o2v4o3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, h2}, {h3, p4}, {h1, h2}, {p4, h3}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1o2o3v4_o2o1o3v4") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, h2}, {h3, p4}, {h2, h1}, {h3, p4}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1o2o3v4_o2o1v4o3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, h2}, {h3, p4}, {h2, h1}, {p4, h3}));
}

////////

TEST_CASE ("EigenAssignTest - FourDim_o1o2v3o4_o1o2v3o4") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h2, h1}, {p3, h4}, {h1, h2}, {p3, h4}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1o2v3o4_o1o2o4v3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, h2}, {p3, h4}, {h1, h2}, {h4, p3}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1o2v3o4_o2o1v3o4") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, h2}, {p3, h4}, {h2, h1}, {p3, h4}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1o2v3o4_o2o1o4v3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, h2}, {p3, h4}, {h2, h1}, {h4, p3}));
}


////////

TEST_CASE ("EigenAssignTest - FourDim_o1o2v3v4_o1o2v3v4") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h2, h1}, {p3, p4}, {h1, h2}, {p3, p4}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1o2v3v4_o1o2v4v3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, h2}, {p3, p4}, {h1, h2}, {p4, p3}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1o2v3v4_o2o1v3v4") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, h2}, {p3, p4}, {h2, h1}, {p3, p4}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1o2v3v4_o2o1v4v3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, h2}, {p3, p4}, {h2, h1}, {p4, p3}));
}

///////////////////////

TEST_CASE ("EigenAssignTest - FourDim_o1v2o3o4_o1v2o3o4") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, p2}, {h3, h4}, {h1, p2}, {h3, h4}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1v2o3o4_o1v2o4o3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec   = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, p2}, {h3, h4}, {h1, p2}, {h4, h3}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1v2o3o4_v2o1o3o4") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, p2}, {h3, h4}, {p2, h1}, {h3, h4}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1v2o3o4_v2o1o4o3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p2, h1}, {h3, h4}, {p2, h1}, {h4, h3}));
}

///////

TEST_CASE ("EigenAssignTest - FourDim_o1v2o3v4_o1v2o3v4") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p2, h1}, {h3, p4}, {h1, p2}, {h3, p4}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1v2o3v4_o1v2v4o3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, p2}, {h3, p4}, {h1, p2}, {p4, h3}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1v2o3v4_v2o1o3v4") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, p2}, {h3, p4}, {p2, h1}, {h3, p4}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1v2o3v4_v2o1v4o3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, p2}, {h3, p4}, {p2, h1}, {p4, h3}));
}

////////

TEST_CASE ("EigenAssignTest - FourDim_o1v2v3o4_o1v2v3o4") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p2, h1}, {p3, h4}, {h1, p2}, {p3, h4}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1v2v3o4_o1v2o4v3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, p2}, {p3, h4}, {h1, p2}, {h4, p3}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1v2v3o4_v2o1v3o4") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, p2}, {p3, h4}, {p2, h1}, {p3, h4}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1v2v3o4_v2o1o4v3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, p2}, {p3, h4}, {p2, h1}, {h4, p3}));
}


////////

TEST_CASE ("EigenAssignTest - FourDim_o1v2v3v4_o1v2v3v4") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p2, h1}, {p3, p4}, {h1, p2}, {p3, p4}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1v2v3v4_o1v2v4v3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, p2}, {p3, p4}, {h1, p2}, {p4, p3}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1v2v3v4_v2o1v3v4") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, p2}, {p3, p4}, {p2, h1}, {p3, p4}));
}

TEST_CASE ("EigenAssignTest - FourDim_o1v2v3v4_v2o1v4v3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h1, p2}, {p3, p4}, {p2, h1}, {p4, p3}));
}

//////////////////////////////////////

TEST_CASE ("EigenAssignTest - FourDim_v1o2o3o4_v1o2o3o4") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, h2}, {h3, h4}, {p1, h2}, {h3, h4}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1o2o3o4_v1o2o4o3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, h2}, {h3, h4}, {p1, h2}, {h4, h3}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1o2o3o4_o2v1o3o4") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, h2}, {h3, h4}, {h2, p1}, {h3, h4}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1o2o3o4_o2v1o4o3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h2, p1}, {h3, h4}, {h2, p1}, {h4, h3}));
}

///////

TEST_CASE ("EigenAssignTest - FourDim_v1o2o3v4_v1o2o3v4") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h2, p1}, {h3, p4}, {p1, h2}, {h3, p4}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1o2o3v4_v1o2v4o3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, h2}, {h3, p4}, {p1, h2}, {p4, h3}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1o2o3v4_o2v1o3v4") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, h2}, {h3, p4}, {h2, p1}, {h3, p4}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1o2o3v4_o2v1v4o3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, h2}, {h3, p4}, {h2, p1}, {p4, h3}));
}

////////

TEST_CASE ("EigenAssignTest - FourDim_v1o2v3o4_v1o2v3o4") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h2, p1}, {p3, h4}, {p1, h2}, {p3, h4}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1o2v3o4_v1o2o4v3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, h2}, {p3, h4}, {p1, h2}, {h4, p3}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1o2v3o4_o2v1v3o4") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, h2}, {p3, h4}, {h2, p1}, {p3, h4}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1o2v3o4_o2v1o4v3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, h2}, {p3, h4}, {h2, p1}, {h4, p3}));
}


////////

TEST_CASE ("EigenAssignTest - FourDim_v1o2v3v4_v1o2v3v4") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {h2, p1}, {p3, p4}, {p1, h2}, {p3, p4}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1o2v3v4_v1o2v4v3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, h2}, {p3, p4}, {p1, h2}, {p4, p3}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1o2v3v4_o2v1v3v4") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, h2}, {p3, p4}, {h2, p1}, {p3, p4}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1o2v3v4_o2v1v4v3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, h2}, {p3, p4}, {h2, p1}, {p4, p3}));
}

//////////////////////////////////////

TEST_CASE ("EigenAssignTest - FourDim_v1v2o3o4_v1v2o3o4") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, p2}, {h3, h4}, {p1, p2}, {h3, h4}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1v2o3o4_v1v2o4o3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, p2}, {h3, h4}, {p1, p2}, {h4, h3}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1v2o3o4_v2v1o3o4") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, p2}, {h3, h4}, {p2, p1}, {h3, h4}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1v2o3o4_v2v1o4o3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p2, p1}, {h3, h4}, {p2, p1}, {h4, h3}));
}

///////

TEST_CASE ("EigenAssignTest - FourDim_v1v2o3v4_v1v2o3v4") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p2, p1}, {h3, p4}, {p1, p2}, {h3, p4}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1v2o3v4_v1v2v4o3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, p2}, {h3, p4}, {p1, p2}, {p4, h3}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1v2o3v4_v2v1o3v4") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, p2}, {h3, p4}, {p2, p1}, {h3, p4}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1v2o3v4_v2v1v4o3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, p2}, {h3, p4}, {p2, p1}, {p4, h3}));
}

////////

TEST_CASE ("EigenAssignTest - FourDim_v1v2v3o4_v1v2v3o4") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p2, p1}, {p3, h4}, {p1, p2}, {p3, h4}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1v2v3o4_v1v2o4v3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, p2}, {p3, h4}, {p1, p2}, {h4, p3}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1v2v3o4_v2v1v3o4") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, p2}, {p3, h4}, {p2, p1}, {p3, h4}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1v2v3o4_v2v1o4v3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, p2}, {p3, h4}, {p2, p1}, {h4, p3}));
}


////////

TEST_CASE ("EigenAssignTest - FourDim_v1v2v3v4_v1v2v3v4") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p2, p1}, {p3, p4}, {p1, p2}, {p3, p4}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1v2v3v4_v1v2v4v3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, p2}, {p3, p4}, {p1, p2}, {p4, p3}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1v2v3v4_v2v1v3v4") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, p2}, {p3, p4}, {p2, p1}, {p3, p4}));
}

TEST_CASE ("EigenAssignTest - FourDim_v1v2v3v4_v2v1v4v3") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

REQUIRE(test_eigen_assign_no_n(*ec, 0.24, {p1, p2}, {p3, p4}, {p2, p1}, {p4, p3}));
}

#endif

#if MULT_TEST_0D_0D

TEST_CASE ("MultTest - Dim_0_0_0") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

Tensor<double> xtc{};
Tensor<double> xta{};
Tensor<double> xtb{};

double alpha1 = 0.91, alpha2 = 0.56;

Tensor<double>::allocate(ec,xta, xtb, xtc);
Scheduler{*ec}
  (xta() = alpha1)
  (xtb() = alpha2)
  //fixme shud be =
  (xtc() += xta() * xtb())
  .execute();

double threshold = 1.0e-12;
bool status = true;

using T = double;
auto lambda = [&](Tensor<T>& t, const IndexVector& iv, std::vector<T>& buf) {
  for (auto& val : buf) 
    status &= (std::abs(val - alpha1 * alpha2) < threshold);
};

Scheduler{*ec}
.gop(xtc(), lambda)
.execute();

Tensor<double>::deallocate(xta, xtb, xtc);

REQUIRE(status);
}
#endif


#if MULT_TEST_0D_1D

TEST_CASE ("MultTest, Dim_o_0_o_up") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

Tensor<double> xtc{O};
Tensor<double> xta{};
Tensor<double> xtb{O};

double alpha1 = 0.91, alpha2 = 0.56;

Tensor<double>::allocate(ec,xta, xtb, xtc);
Scheduler{*ec}
(xta() = alpha1)
(xtb() = alpha2)
(xtc() = 0)
//fixme =
(xtc() += xta() * xtb())
.execute();

double threshold = 1.0e-12;
bool status = true;

using T = double;
auto lambda = [&](Tensor<T>& t, const IndexVector& iv, std::vector<T>& buf) {
  for (auto& val : buf) 
    status &= (std::abs(val - alpha1 * alpha2) < threshold);
};

Scheduler{*ec}
.gop(xtc(), lambda)
.execute();

Tensor<double>::deallocate(xta, xtb, xtc);

REQUIRE(status);
}

TEST_CASE ("MultTest - Dim_o_0_o_lo") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

Tensor<double> xtc{O};
Tensor<double> xta{};
Tensor<double> xtb{O};

double alpha1 = 0.91, alpha2 = 0.56;

Tensor<double>::allocate(ec,xta, xtb, xtc);
Scheduler{*ec}
(xta() = alpha1)
(xtb() = alpha2)
//fixme =
(xtc() += xta() * xtb())
.execute();

double threshold = 1.0e-12;
bool status = true;

using T = double;
auto lambda = [&](Tensor<T>& t, const IndexVector& iv, std::vector<T>& buf) {
  for (auto& val : buf) 
    status &= (std::abs(val - alpha1 * alpha2) < threshold);
};

Scheduler{*ec}
  .gop(xtc(), lambda)
.execute();

Tensor<double>::deallocate(xta, xtb, xtc);

REQUIRE(status);
}

TEST_CASE ("MultTest - Dim_v_v_0_hi") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

Tensor<double> xtc{V};
Tensor<double> xta{V};
Tensor<double> xtb{};

double alpha1 = 0.91, alpha2 = 0.56;

Tensor<double>::allocate(ec,xta, xtb, xtc);
Scheduler{*ec}
(xta() = alpha1)
(xtb() = alpha2)
//fixme =
(xtc() += xta() * xtb())
.execute();

double threshold = 1.0e-12;
bool status = true;

using T = double;
auto lambda = [&](Tensor<T>& t, const IndexVector& iv, std::vector<T>& buf) {
  for (auto& val : buf) 
    status &= (std::abs(val - alpha1 * alpha2) < threshold);
};


Scheduler{*ec}
.gop(xtc(), lambda)
.execute();

Tensor<double>::deallocate(xta, xtb, xtc);

REQUIRE(status);
}

TEST_CASE ("MultTest - Dim_v_v_0_lo") {

ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
Distribution_NW distribution;
RuntimeEngine re;
ExecutionContext* ec  = new ExecutionContext{pg, &distribution, mgr, &re};

Tensor<double> xtc{V};
Tensor<double> xta{V};
Tensor<double> xtb{};

double alpha1 = 0.91, alpha2 = 0.56;

Tensor<double>::allocate(ec,xta, xtb, xtc);
Scheduler{*ec}
  (xta() = alpha1)
  (xtb() = alpha2)
  //fixme =
  (xtc() += xta() * xtb())
  .execute();

double threshold = 1.0e-12;
bool status = true;

using T = double;
auto lambda = [&](Tensor<T>& t, const IndexVector& iv, std::vector<T>& buf) {
  for (auto& val : buf) 
    status &= (std::abs(val - alpha1 * alpha2) < threshold);
};

Scheduler{*ec}
  .gop(xtc(), lambda)
  .execute();

Tensor<double>::deallocate(xta, xtb, xtc);

REQUIRE(status);
}

#endif


int main(int argc, char* argv[]) {

    tamm::initialize(argc, argv);

    doctest::Context context(argc, argv);

    int res = context.run();

    tamm::finalize();

    return res;
}
