#pragma once

#include "tamm/tamm.hpp"

using EigenTensorType=double;
using Matrix   = Eigen::Matrix<EigenTensorType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Tensor1D = Eigen::Tensor<EigenTensorType, 1, Eigen::RowMajor>;
using Tensor2D = Eigen::Tensor<EigenTensorType, 2, Eigen::RowMajor>;
using Tensor3D = Eigen::Tensor<EigenTensorType, 3, Eigen::RowMajor>;
using Tensor4D = Eigen::Tensor<EigenTensorType, 4, Eigen::RowMajor>;

namespace tamm {

template<typename T, int ndim>
void patch_copy(std::vector<T>& sbuf,
                Eigen::Tensor<T, ndim, Eigen::RowMajor>& etensor,
                const std::array<int, ndim>& block_dims,
                const std::array<int, ndim>& block_offset, bool t2e = true) {
    assert(0);
}

template<typename T>
void patch_copy(std::vector<T>& sbuf,
                Eigen::Matrix<T, 1, Eigen::Dynamic>& etensor,
                const std::vector<size_t>& block_dims,
                const std::vector<size_t>& block_offset, bool t2e = true) {
    size_t c = 0;
    for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0];
        i++, c++) {
        if(t2e)
            etensor(i) = sbuf[c];
        else
            sbuf[c] = etensor(i);
    }
}

template<typename T>
void patch_copy(std::vector<T>& sbuf,
                Eigen::Tensor<T, 1, Eigen::RowMajor>& etensor,
                const std::vector<size_t>& block_dims,
                const std::vector<size_t>& block_offset, bool t2e = true) {
    size_t c = 0;
    for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0];
        i++, c++) {
        if(t2e)
            etensor(i) = sbuf[c];
        else
            sbuf[c] = etensor(i);
    }
}

template<typename T>
void patch_copy(std::vector<T>& sbuf,
                Eigen::Tensor<T, 2, Eigen::RowMajor>& etensor,
                const std::vector<size_t>& block_dims,
                const std::vector<size_t>& block_offset, bool t2e = true) {
    size_t c = 0;
    for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
        for(size_t j = block_offset[1]; j < block_offset[1] + block_dims[1];
            j++, c++) {
            if(t2e)
                etensor(i, j) = sbuf[c];
            else
                sbuf[c] = etensor(i, j);
        }
    }
}

template<typename T>
void patch_copy(std::vector<T>& sbuf,
                Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
                              Eigen::RowMajor>& etensor,
                const std::vector<size_t>& block_dims,
                const std::vector<size_t>& block_offset, bool t2e = true) {
    size_t c = 0;
    for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
        for(size_t j = block_offset[1]; j < block_offset[1] + block_dims[1];
            j++, c++) {
            if(t2e)
                etensor(i, j) = sbuf[c];
            else
                sbuf[c] = etensor(i, j);
        }
    }
}

template<typename T>
void patch_copy(std::vector<T>& sbuf,
                Eigen::Tensor<T, 3, Eigen::RowMajor>& etensor,
                const std::vector<size_t>& block_dims,
                const std::vector<size_t>& block_offset, bool t2e = true) {
    size_t c = 0;
    for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
        for(size_t j = block_offset[1]; j < block_offset[1] + block_dims[1];
            j++) {
            for(size_t k = block_offset[2]; k < block_offset[2] + block_dims[2];
                k++, c++) {
                if(t2e)
                    etensor(i, j, k) = sbuf[c];
                else
                    sbuf[c] = etensor(i, j, k);
            }
        }
    }
}

template<typename T>
void patch_copy(std::vector<T>& sbuf,
                Eigen::Tensor<T, 4, Eigen::RowMajor>& etensor,
                const std::vector<size_t>& block_dims,
                const std::vector<size_t>& block_offset, bool t2e = true) {
    size_t c = 0;
    for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
        for(size_t j = block_offset[1]; j < block_offset[1] + block_dims[1];
            j++) {
            for(size_t k = block_offset[2]; k < block_offset[2] + block_dims[2];
                k++) {
                for(size_t l = block_offset[3];
                    l < block_offset[3] + block_dims[3]; l++, c++) {
                    if(t2e)
                        etensor(i, j, k, l) = sbuf[c];
                    else
                        sbuf[c] = etensor(i, j, k, l);
                }
            }
        }
    }
}

inline Matrix eigen_tensor_to_matrix(Tensor2D &eig_tensor) {
  Matrix eig_mat(eig_tensor.dimensions()[0], eig_tensor.dimensions()[1]);
  eig_mat.setZero();
  for (int i = 0; i < eig_mat.rows(); i++)
    for (int j = 0; j < eig_mat.cols(); j++)
      eig_mat(i, j) = eig_tensor(i, j);
  return eig_mat;
}

template<typename T, int ndim>
Eigen::Tensor<T, ndim, Eigen::RowMajor> tamm_to_eigen_tensor(
    const tamm::Tensor<T>& tensor) {
    std::array<Eigen::Index, ndim> dims;
    const auto& tindices = tensor.tiled_index_spaces();
    for(int i = 0; i < ndim; i++) {
        dims[i] = tindices[i].index_space().num_indices();
    }
    Eigen::Tensor<T, ndim, Eigen::RowMajor> etensor;
    etensor = etensor.reshape(dims);
    etensor.setZero();

    for(const auto& blockid : tensor.loop_nest()) {
        const tamm::TAMM_SIZE size = tensor.block_size(blockid);
        std::vector<T> buf(size);
        tensor.get(blockid, buf);
        auto block_dims   = tensor.block_dims(blockid);
        auto block_offset = tensor.block_offsets(blockid);
        patch_copy<T>(buf, etensor, block_dims, block_offset, true);
    }

    return etensor;
}

template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> tamm_to_eigen_matrix(
  const tamm::Tensor<T>& tensor){
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> etensor;
    auto dim1 = tensor.tiled_index_spaces()[0].max_num_indices();
    auto dim2 = tensor.tiled_index_spaces()[1].max_num_indices();
    etensor.setZero(dim1,dim2);
    for(const auto& blockid : tensor.loop_nest()) {
        const tamm::TAMM_SIZE size = tensor.block_size(blockid);
        std::vector<T> buf(size);
        tensor.get(blockid, buf);
        auto block_dims   = tensor.block_dims(blockid);
        auto block_offset = tensor.block_offsets(blockid);
        patch_copy<T>(buf, etensor, block_dims, block_offset, true);
    }
    return etensor;
}

// 1D case
template<typename T>
void tamm_to_eigen_tensor(
  const tamm::Tensor<T>& tensor,
  Eigen::Matrix<T, 1, Eigen::Dynamic>& etensor) {
    for(const auto& blockid : tensor.loop_nest()) {
        const tamm::TAMM_SIZE size = tensor.block_size(blockid);
        std::vector<T> buf(size);
        tensor.get(blockid, buf);
        auto block_dims   = tensor.block_dims(blockid);
        auto block_offset = tensor.block_offsets(blockid);
        patch_copy<T>(buf, etensor, block_dims, block_offset, true);
    }
}

template<typename T>
void tamm_to_eigen_tensor(
  const tamm::Tensor<T>& tensor,
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& etensor) {
    for(const auto& blockid : tensor.loop_nest()) {
        const tamm::TAMM_SIZE size = tensor.block_size(blockid);
        std::vector<T> buf(size);
        tensor.get(blockid, buf);
        auto block_dims   = tensor.block_dims(blockid);
        auto block_offset = tensor.block_offsets(blockid);
        patch_copy<T>(buf, etensor, block_dims, block_offset, true);
    }
}

template<typename T, int ndim>
void tamm_to_eigen_tensor(const tamm::Tensor<T>& tensor,
                          Eigen::Tensor<T, ndim, Eigen::RowMajor>& etensor) {
    for(const auto& blockid : tensor.loop_nest()) {
        const tamm::TAMM_SIZE size = tensor.block_size(blockid);
        std::vector<T> buf(size);
        tensor.get(blockid, buf);
        auto block_dims   = tensor.block_dims(blockid);
        auto block_offset = tensor.block_offsets(blockid);
        patch_copy<T>(buf, etensor, block_dims, block_offset, true);
    }
}

template<typename T, int ndim>
void eigen_to_tamm_tensor(tamm::Tensor<T>& tensor,
                          Eigen::Tensor<T, ndim, Eigen::RowMajor>& etensor) {
    for(const auto& blockid : tensor.loop_nest()) {
        const tamm::TAMM_SIZE size = tensor.block_size(blockid);
        std::vector<T> buf(size);
        // tensor.get(blockid, buf);
        auto block_dims   = tensor.block_dims(blockid);
        auto block_offset = tensor.block_offsets(blockid);
        patch_copy<T>(buf, etensor, block_dims, block_offset, false);
        tensor.put(blockid, buf);
    }
}

// 1D case
template<typename T>
void eigen_to_tamm_tensor(
  tamm::Tensor<T>& tensor,
  Eigen::Matrix<T, 1, Eigen::Dynamic>& etensor) {
    for(const auto& blockid : tensor.loop_nest()) {
        const tamm::TAMM_SIZE size = tensor.block_size(blockid);
        std::vector<T> buf(size);
        // tensor.get(blockid, buf);
        auto block_dims   = tensor.block_dims(blockid);
        auto block_offset = tensor.block_offsets(blockid);
        patch_copy<T>(buf, etensor, block_dims, block_offset, false);
        tensor.put(blockid, buf);
    }
}

template<typename T>
void eigen_to_tamm_tensor(
  tamm::Tensor<T>& tensor,
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& etensor) {
    for(const auto& blockid : tensor.loop_nest()) {
        const tamm::TAMM_SIZE size = tensor.block_size(blockid);
        std::vector<T> buf(size);
        // tensor.get(blockid, buf);
        auto block_dims   = tensor.block_dims(blockid);
        auto block_offset = tensor.block_offsets(blockid);
        patch_copy<T>(buf, etensor, block_dims, block_offset, false);
        tensor.put(blockid, buf);
    }
}

template<typename T>
void eigen_to_tamm_tensor_acc(
  tamm::Tensor<T>& tensor,
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& etensor) {
    for(const auto& blockid : tensor.loop_nest()) {
        const tamm::TAMM_SIZE size = tensor.block_size(blockid);
        std::vector<T> buf(size);
        // tensor.get(blockid, buf);
        auto block_dims   = tensor.block_dims(blockid);
        auto block_offset = tensor.block_offsets(blockid);
        patch_copy<T>(buf, etensor, block_dims, block_offset, false);
        tensor.add(blockid, buf);
    }
}

template<typename T>
void eigen_to_tamm_tensor_acc(
  tamm::Tensor<T>& tensor,
  Eigen::Tensor<T, 2, Eigen::RowMajor>& etensor) {
    for(const auto& blockid : tensor.loop_nest()) {
        const tamm::TAMM_SIZE size = tensor.block_size(blockid);
        std::vector<T> buf(size);
        // tensor.get(blockid, buf);
        auto block_dims   = tensor.block_dims(blockid);
        auto block_offset = tensor.block_offsets(blockid);
        patch_copy<T>(buf, etensor, block_dims, block_offset, false);
        tensor.add(blockid, buf);
    }
}

template<typename T>
void eigen_to_tamm_tensor_acc(
  tamm::Tensor<T>& tensor,
  Eigen::Tensor<T, 3, Eigen::RowMajor>& etensor) {
    for(const auto& blockid : tensor.loop_nest()) {
        const tamm::TAMM_SIZE size = tensor.block_size(blockid);
        std::vector<T> buf(size);
        // tensor.get(blockid, buf);
        auto block_dims   = tensor.block_dims(blockid);
        auto block_offset = tensor.block_offsets(blockid);
        patch_copy<T>(buf, etensor, block_dims, block_offset, false);
        tensor.add(blockid, buf);
    }
}

template<typename T, typename fxn_t>
void call_eigen_matrix_fxn(const tamm::Tensor<T>& tensor, fxn_t fxn) {
    tamm::Tensor<T> copy_t(tensor);
    auto eigen_t = tamm_to_eigen_tensor<T, 2>(copy_t);

    using eigen_matrix =
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using eigen_map = Eigen::Map<eigen_matrix>;

    const auto nrow = eigen_t.dimensions()[0];
    const auto ncol = eigen_t.dimensions()[1];

    eigen_map mapped_t(eigen_t.data(), nrow, ncol);

    fxn(mapped_t);
}

template<int nmodes, typename T, typename matrix_type>
void eigen_matrix_to_tamm(matrix_type&& mat, tamm::Tensor<T>& t){
    using tensor1_type = Eigen::Tensor<T, 1, Eigen::RowMajor>;
    using tensor2_type = Eigen::Tensor<T, 2, Eigen::RowMajor>;

    const auto nrows = mat.rows();

    if constexpr(nmodes == 1){ //Actually a vector
        tensor1_type t1(std::array<long int, 1>{nrows});
        for(long int i = 0; i < nrows; ++i) t1(i) = mat(i);
        eigen_to_tamm_tensor(t, t1);
    }
    else if constexpr(nmodes == 2){
        const auto ncols = mat.cols();
        tensor2_type t1(std::array<long int, 2>{nrows, ncols}) ;

        for(long int i=0; i < nrows; ++i)
            for(long int j =0; j< ncols; ++j)
                t1(i, j) = mat(i, j);

        eigen_to_tamm_tensor(t, t1);
    }
}

template<typename T>
tamm::Tensor<T> retile_rank2_tensor(tamm::Tensor<T>& tensor, const tamm::TiledIndexSpace &t1, const tamm::TiledIndexSpace &t2) {
   EXPECTS(tensor.num_modes() == 2);
   //auto* ec_tmp = tensor.execution_context(); //TODO: figure out why this seg faults


   tamm::ProcGroup pg = ProcGroup::create_world_coll();
   auto mgr = tamm::MemoryManagerGA::create_coll(pg);
   tamm::Distribution_NW distribution;
   tamm::ExecutionContext ec{pg,&distribution,mgr};
   auto is1 = tensor.tiled_index_spaces()[0].index_space();
   auto is2 = tensor.tiled_index_spaces()[1].index_space();
   auto dim1 = is1.num_indices();
   auto dim2 = is2.num_indices();
   Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> eigent(dim1,dim2);

   tamm::Tensor<T> result{t1,t2};
   tamm::Tensor<T>::allocate(&ec, result);
   tamm_to_eigen_tensor(tensor,eigent);
   eigen_to_tamm_tensor(result,eigent);
   ec.pg().barrier();
   
   return result;
} 

}

inline bool eigen_tensors_are_equal(Tensor1D& e1, Tensor1D& e2, double threshold = 1.0e-12) {
  bool ret  = true;
  auto dims = e1.dimensions();
  for(auto i = 0; i < dims[0]; i++) {
    if(std::abs(e1(i) - e2(i)) > std::abs(threshold * e1(i))) {
      ret = false;
      break;
    }
  }
  return ret;
}

inline bool eigen_tensors_are_equal(Tensor2D& e1, Tensor2D& e2, double threshold = 1.0e-12) {
  bool ret  = true;
  auto dims = e1.dimensions();
  for(auto i = 0; i < dims[0]; i++) {
    for(auto j = 0; j < dims[1]; j++) {
      if(std::abs(e1(i, j) - e2(i, j)) > std::abs(threshold * e1(i, j))) {
        ret = false;
        break;
      }
    }
  }
  return ret;
}

inline bool eigen_tensors_are_equal(Tensor3D& e1, Tensor3D& e2, double threshold = 1.0e-12) {
  bool ret  = true;
  auto dims = e1.dimensions();
  for(auto i = 0; i < dims[0]; i++) {
    for(auto j = 0; j < dims[1]; j++) {
      for(auto k = 0; k < dims[2]; k++) {
        if(std::abs(e1(i, j, k) - e2(i, j, k)) > std::abs(threshold * e1(i, j, k))) {
          ret = false;
          break;
        }
      }
    }
  }
  return ret;
}

inline bool eigen_tensors_are_equal(Tensor4D& e1, Tensor4D& e2, double threshold = 1.0e-12) {
  bool ret  = true;
  auto dims = e1.dimensions();
  for(auto i = 0; i < dims[0]; i++) {
    for(auto j = 0; j < dims[1]; j++) {
      for(auto k = 0; k < dims[2]; k++) {
        for(auto l = 0; l < dims[3]; l++) {
          if(std::abs(e1(i, j, k, l) - e2(i, j, k, l)) > std::abs(threshold * e1(i, j, k, l))) {
            ret = false;
            break;
          }
        }
      }
    }
  }
  return ret;
}


