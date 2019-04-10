#ifndef TAMM_EIGEN_UTILS_HPP_
#define TAMM_EIGEN_UTILS_HPP_

// Eigen matrix algebra library
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "tamm/tamm.hpp"

template<typename T, int ndim>
void patch_copy(std::vector<T>& sbuf,
                Eigen::Tensor<T, ndim, Eigen::RowMajor>& etensor,
                const std::array<int, ndim>& block_dims,
                const std::array<int, ndim>& block_offset, bool t2e = true) {
    assert(0);
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
                Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
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

template<typename T, int ndim>
Eigen::Tensor<T, ndim, Eigen::RowMajor> tamm_to_eigen_tensor(
    const tamm::Tensor<T>& tensor) {
    std::array<long, ndim> dims;
    const auto& tindices = tensor.tiled_index_spaces();
    for(int i = 0; i < ndim; i++) {
        dims[i] = tindices[i].index_space().num_indices();
    }
    Eigen::Tensor<T, ndim, Eigen::RowMajor> etensor(dims);
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
  Eigen::Tensor<double, 2, Eigen::RowMajor>& etensor) {
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
  Eigen::Tensor<double, 3, Eigen::RowMajor>& etensor) {
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

#endif // TAMM_EIGEN_UTILS_HPP_
