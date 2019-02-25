#ifndef TAMM_TAMM_UTILS_HPP_
#define TAMM_TAMM_UTILS_HPP_

#include <vector>

namespace tamm {
/**
 * @brief Overload of << operator for printing Tensor blocks
 *
 * @tparam T template type for Tensor element type
 * @param [in] os output stream
 * @param [in] vec vector to be printed
 * @returns the reference to input output stream with vector elements printed
 * out
 */
template<typename T>
std::ostream& operator<<(std::ostream& os, std::vector<T>& vec) {
    os << "[";
    for(auto& x : vec) os << x << ",";
    os << "]\n";
    return os;
}

/**
 * @brief Prints a Tensor object
 *
 * @tparam T template type for Tensor element type
 * @param [in] tensor input Tensor object
 */
template<typename T>
void print_tensor(const Tensor<T>& tensor) {
    auto lt = tensor();
    for(auto it : tensor.loop_nest()) {
        auto blockid   = internal::translate_blockid(it, lt);
        TAMM_SIZE size = tensor.block_size(blockid);
        std::vector<T> buf(size);
        tensor.get(blockid, buf);
        std::cout << "block" << blockid;

        for(TAMM_SIZE i = 0; i < size; i++) {
            if(buf[i] > 0.0000000000001 || buf[i] < -0.0000000000001)
                std::cout << buf[i] << " ";
        }
        std::cout << std::endl;
    }
}

template<typename T>
void print_tensor_all(Tensor<T> &t){
    for (auto it: t.loop_nest())
    {
        TAMM_SIZE size = t.block_size(it);
        std::vector<T> buf(size);
        t.get(it, buf);
        std::cout << "block" << it;
        for (TAMM_SIZE i = 0; i < size;i++)
         std::cout << buf[i] << std::endl;
    }
}

/**
 * @brief Get the scalar value from the Tensor
 *
 * @tparam T template type for Tensor element type
 * @param [in] tensor input Tensor object
 * @returns a scalar value in type T
 *
 * @warning This function only works with scalar (zero dimensional) Tensor
 * objects
 */
template<typename T>
T get_scalar(Tensor<T>& tensor) {
    T scalar;
    EXPECTS(tensor.num_modes() == 0);
    tensor.get({}, {&scalar, 1});
    return scalar;
}


/**
 * @brief Update input LabeledTensor object with a lambda function
 *
 * @tparam T template type for Tensor element type
 * @tparam Func template type for the lambda function
 * @param [in] labeled_tensor tensor slice to be updated
 * @param [in] lambda function for updating the tensor
 */
template<typename T, typename Func>
void update_tensor(LabeledTensor<T> labeled_tensor, Func lambda) {
    LabelLoopNest loop_nest{labeled_tensor.labels()};

    for(const auto& itval : loop_nest) {
        const IndexVector blockid =
          internal::translate_blockid(itval, labeled_tensor);
        size_t size = labeled_tensor.tensor().block_size(blockid);
        std::vector<T> buf(size);

        labeled_tensor.tensor().get(blockid, buf);
        lambda(blockid, buf);
        labeled_tensor.tensor().put(blockid, buf);
    }
}

/**
 * @brief Update input LabeledTensor object with a lambda function
 *
 * @tparam T template type for Tensor element type
 * @tparam Func template type for the lambda function
 * @param [in] labeled_tensor tensor slice to be updated
 * @param [in] lambda function for updating the tensor
 */
template<typename T, typename Func>
void update_tensor_general(LabeledTensor<T> labeled_tensor, Func lambda) {
    LabelLoopNest loop_nest{labeled_tensor.labels()};

    for(const auto& itval : loop_nest) {
        const IndexVector blockid =
          internal::translate_blockid(itval, labeled_tensor);
        size_t size = labeled_tensor.tensor().block_size(blockid);
        std::vector<T> buf(size);

        labeled_tensor.tensor().get(blockid, buf);
        lambda(labeled_tensor.tensor(), blockid, buf);
        labeled_tensor.tensor().put(blockid, buf);
    }
}

/**
 * @brief Construct an ExecutionContext object
 *
 * @returns an Execution context
 * @todo there is possible memory leak as distribution will not be unallocated
 * when Execution context is destructed
 */
inline ExecutionContext make_execution_context() {
    ProcGroup pg{GA_MPI_Comm()};
    auto* pMM             = MemoryManagerLocal::create_coll(pg);
    Distribution_NW* dist = new Distribution_NW();
    return ExecutionContext(pg, dist, pMM);
}

/**
 * @brief method for getting the sum of the values on the diagonal
 * 
 * @returns sum of the diagonal values
 * @warning only defined for NxN tensors 
 */
template<typename TensorType>
TensorType trace(ExecutionContext &ec, LabeledTensor<TensorType> ltensor){

    TensorType lsumd=0;
    TensorType gsumd=0;

    Tensor<TensorType> tensor = ltensor.tensor();
    // Defined only for NxN tensors
    EXPECTS(tensor.num_modes() == 2);

    auto gettrace = [&](const IndexVector& bid) {
        const IndexVector blockid =
          internal::translate_blockid(bid, ltensor);
        if(blockid[0] == blockid[1]) {
                const TAMM_SIZE size = tensor.block_size(blockid);
                std::vector<TensorType> buf(size);
                tensor.get(blockid, buf);
                auto block_dims   = tensor.block_dims(blockid);
                auto block_offset = tensor.block_offsets(blockid);
                auto dim          = block_dims[0];
                auto offset       = block_offset[0];
                size_t i          = 0;
                for(auto p = offset; p < offset + dim; p++, i++) {
                    lsumd += buf[i * dim + i];
                }
            }
    };
    block_for(ec, ltensor, gettrace);
    MPI_Allreduce(&lsumd, &gsumd, 1, MPI_DOUBLE, MPI_SUM, ec.pg().comm());
    return gsumd;
}


/**
 * @brief method for getting the diagonal values in a Tensor
 * 
 * @returns the diagonal values
 * @warning only defined for NxN tensors 
 */
template<typename TensorType>
std::vector<TensorType> diagonal(ExecutionContext &ec, LabeledTensor<TensorType> ltensor){

    Tensor<TensorType> tensor = ltensor.tensor();
    // Defined only for NxN tensors
    EXPECTS(tensor.num_modes() == 2);

    LabelLoopNest loop_nest{ltensor.labels()};
    std::vector<TensorType> dest;

    for(const IndexVector& bid : loop_nest) {
        const IndexVector blockid =
          internal::translate_blockid(bid, ltensor);

        if(blockid[0] == blockid[1]) {
                const TAMM_SIZE size = tensor.block_size(blockid);
                std::vector<TensorType> buf(size);
                tensor.get(blockid, buf);
                auto block_dims   = tensor.block_dims(blockid);
                auto block_offset = tensor.block_offsets(blockid);
                auto dim          = block_dims[0];
                auto offset       = block_offset[0];
                size_t i          = 0;
                for(auto p = offset; p < offset + dim; p++, i++) {
                    dest.push_back(buf[i * dim + i]);
                }
            }
    }

    return dest;
}

/**
 * @brief applies a function elementwise to a tensor
 *
 * @tparam TensorType the type of the elements in the tensor
 * @param ec Execution context used in the blockfor
 * @param ltensor tensor to operate on
 * @param func function to be applied to each element
 */
template<typename TensorType>
void apply_ewise(ExecutionContext &ec, LabeledTensor<TensorType> ltensor,
                 std::function<TensorType(TensorType)> func){

    Tensor<TensorType> tensor = ltensor.tensor();

    auto lambda = [&](const IndexVector& bid) {
        const IndexVector blockid =
                internal::translate_blockid(bid, ltensor);
        const tamm::TAMM_SIZE dsize = tensor.block_size(blockid);
        std::vector<TensorType> dbuf(dsize);
        tensor.get(blockid, dbuf);
        for(size_t c = 0; c < dsize; c++)
            dbuf[c] = func(dbuf[c]);
        tensor.put(blockid,dbuf);
    };
    block_for(ec, ltensor, lambda);

}

// Several convenience functions using apply_ewise
template<typename TensorType>
void square(ExecutionContext &ec, LabeledTensor<TensorType> ltensor){
    std::function<TensorType(TensorType)> func = [&](TensorType a){return a*a;};
    apply_ewise(ec, ltensor, func);
}


template<typename TensorType>
void log10(ExecutionContext &ec, LabeledTensor<TensorType> ltensor){
    std::function<TensorType(TensorType)> func = [&](TensorType a){return std::log10(a);};
    apply_ewise(ec, ltensor, func);
}

template<typename TensorType>
void log(ExecutionContext &ec, LabeledTensor<TensorType> ltensor){
    std::function<TensorType(TensorType)> func = [&](TensorType a){return std::log(a);};
    apply_ewise(ec, ltensor, func);
}

template<typename TensorType>
void inverse(ExecutionContext &ec, LabeledTensor<TensorType> ltensor){
    std::function<TensorType(TensorType)> func = [&](TensorType a){return 1/a;};
    apply_ewise(ec, ltensor, func);
}

template<typename TensorType>
void pow(ExecutionContext &ec, LabeledTensor<TensorType> ltensor, TensorType alpha){
    std::function<TensorType(TensorType)> func = [&](TensorType a){return std::pow(a, alpha);};
    apply_ewise(ec, ltensor, func);
}

template<typename TensorType>
void scale(ExecutionContext &ec, LabeledTensor<TensorType> ltensor, TensorType alpha){
    std::function<TensorType(TensorType)> func = [&](TensorType a){return alpha*a;};
    apply_ewise(ec, ltensor, func);
}

template<typename TensorType>
void sqrt(ExecutionContext &ec, LabeledTensor<TensorType> ltensor){
    std::function<TensorType(TensorType)> func = [&](TensorType a){return std::sqrt(a);};
    apply_ewise(ec, ltensor, func);
}

template<typename TensorType>
TensorType norm(ExecutionContext &ec, LabeledTensor<TensorType> ltensor){

    TensorType lsumsq=0;
    TensorType gsumsq=0;
    Tensor<TensorType> tensor = ltensor.tensor();

    auto getnorm = [&](const IndexVector& bid) {
        const IndexVector blockid =
          internal::translate_blockid(bid, ltensor);
        const tamm::TAMM_SIZE dsize = tensor.block_size(blockid);
        std::vector<TensorType> dbuf(dsize);
        tensor.get(blockid, dbuf);
        for(auto val: dbuf) 
            lsumsq += val * val;
            
    };
    block_for(ec, ltensor, getnorm);
    MPI_Allreduce(&lsumsq, &gsumsq, 1, MPI_DOUBLE, MPI_SUM, ec.pg().comm());
    return std::sqrt(gsumsq);
}


//returns max_element, blockids, coordinates of max element in the block
template<typename TensorType>
std::tuple<TensorType, IndexVector, std::vector<size_t>> max_element(ExecutionContext &ec, LabeledTensor<TensorType> ltensor){
    TensorType max = 0.0;
    
    Tensor<TensorType> tensor = ltensor.tensor();
    auto nmodes = tensor.num_modes();
     //Works for only upto 6D tensors
    EXPECTS(tensor.num_modes() <= 6);

    IndexVector maxblockid(nmodes);
    std::vector<size_t> bfuv(nmodes);
    std::vector<TensorType> lmax(2,0);
    std::vector<TensorType> gmax(2,0);

    auto getmax = [&](const IndexVector& bid) {
        const IndexVector blockid =
          internal::translate_blockid(bid, ltensor);
        const tamm::TAMM_SIZE dsize = tensor.block_size(blockid);
        std::vector<TensorType> dbuf(dsize);
        tensor.get(blockid, dbuf);
        auto block_dims   = tensor.block_dims(blockid);
        auto block_offset = tensor.block_offsets(blockid);
       
        size_t c = 0;

        if(nmodes == 1) {
            for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0];
                i++, c++) {
                if(lmax[0] < dbuf[c]) {
                    lmax[0]    = dbuf[c];
                    lmax[1]    = GA_Nodeid();
                    bfuv[0]    = i - block_offset[0];
                    maxblockid = {blockid[0]};
                }
            }
        } else if(nmodes == 2) {
                auto dimi = block_offset[0] + block_dims[0];
                auto dimj = block_offset[1] + block_dims[1];
            for(size_t i = block_offset[0]; i < dimi;
                i++) {
                for(size_t j = block_offset[1];
                    j < dimj; j++, c++) {
                    if(lmax[0] < dbuf[c]) {
                        lmax[0]    = dbuf[c];
                        lmax[1]    = GA_Nodeid();
                        bfuv[0]    = i-block_offset[0];
                        bfuv[1]    = j-block_offset[1];
                        maxblockid = {blockid[0], blockid[1]};
                    }
                }
            }
        } else if(nmodes == 3) {
            auto dimi = block_offset[0] + block_dims[0];
            auto dimj = block_offset[1] + block_dims[1];
            auto dimk = block_offset[2] + block_dims[2];

            for(size_t i = block_offset[0]; i < dimi; i++) {
                for(size_t j = block_offset[1]; j < dimj; j++) {
                    for(size_t k = block_offset[2]; k < dimk; k++, c++) {
                        if(lmax[0] < dbuf[c]) {
                            lmax[0]    = dbuf[c];
                            lmax[1]    = GA_Nodeid();
                            bfuv[0]    = i - block_offset[0];
                            bfuv[1]    = j - block_offset[1];
                            bfuv[2]    = k - block_offset[2];
                            maxblockid = {blockid[0], blockid[1], blockid[2]};
                        }
                    }
                }
            }
        } else if(nmodes == 4) {
            for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0];
                i++) {
                for(size_t j = block_offset[1];
                    j < block_offset[1] + block_dims[1]; j++) {
                    for(size_t k = block_offset[2];
                        k < block_offset[2] + block_dims[2]; k++) {
                        for(size_t l = block_offset[3];
                            l < block_offset[3] + block_dims[3]; l++, c++) {
                            if(lmax[0] < dbuf[c]) {
                                lmax[0]    = dbuf[c];
                                lmax[1]    = GA_Nodeid();
                                bfuv[0]    = i - block_offset[0];
                                bfuv[1]    = j - block_offset[1];
                                bfuv[2]    = k - block_offset[2];
                                bfuv[3]    = l - block_offset[3];
                                maxblockid = {blockid[0], blockid[1],
                                              blockid[2], blockid[3]};
                            }
                        }
                    }
                }
            }
        } else if(nmodes == 5) {
            for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0];
                i++) {
                for(size_t j = block_offset[1];
                    j < block_offset[1] + block_dims[1]; j++) {
                    for(size_t k = block_offset[2];
                        k < block_offset[2] + block_dims[2]; k++) {
                        for(size_t l = block_offset[3];
                            l < block_offset[3] + block_dims[3]; l++) {
                            for(size_t m = block_offset[4];
                                m < block_offset[4] + block_dims[4]; m++, c++) {
                                if(lmax[0] < dbuf[c]) {
                                    lmax[0]    = dbuf[c];
                                    lmax[1]    = GA_Nodeid();
                                    bfuv[0]    = i - block_offset[0];
                                    bfuv[1]    = j - block_offset[1];
                                    bfuv[2]    = k - block_offset[2];
                                    bfuv[3]    = l - block_offset[3];
                                    bfuv[4]    = m - block_offset[4];
                                    maxblockid = {blockid[0], blockid[1],
                                                  blockid[2], blockid[3],
                                                  blockid[4]};
                                }
                            }
                        }
                    }
                }
            }
        }

        else if(nmodes == 6) {
            for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0];
                i++) {
                for(size_t j = block_offset[1];
                    j < block_offset[1] + block_dims[1]; j++) {
                    for(size_t k = block_offset[2];
                        k < block_offset[2] + block_dims[2]; k++) {
                        for(size_t l = block_offset[3];
                            l < block_offset[3] + block_dims[3]; l++) {
                            for(size_t m = block_offset[4];
                                m < block_offset[4] + block_dims[4]; m++) {
                                for(size_t n = block_offset[5];
                                    n < block_offset[5] + block_dims[5];
                                    n++, c++) {
                                    if(lmax[0] < dbuf[c]) {
                                        lmax[0]    = dbuf[c];
                                        lmax[1]    = GA_Nodeid();
                                        bfuv[0]    = i - block_offset[0];
                                        bfuv[1]    = j - block_offset[1];
                                        bfuv[2]    = k - block_offset[2];
                                        bfuv[3]    = l - block_offset[3];
                                        bfuv[4]    = m - block_offset[4];
                                        bfuv[5]    = n - block_offset[5];
                                        maxblockid = {blockid[0], blockid[1],
                                                      blockid[2], blockid[3],
                                                      blockid[4], blockid[5]};
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }              
    };
    block_for(ec, ltensor, getmax);

    MPI_Allreduce(lmax.data(), gmax.data(), 1, MPI_2DOUBLE_PRECISION, MPI_MAXLOC, ec.pg().comm());
    MPI_Bcast(maxblockid.data(),2,MPI_UNSIGNED,gmax[1],ec.pg().comm());
    MPI_Bcast(bfuv.data(),2,MPI_UNSIGNED_LONG,gmax[1],ec.pg().comm());

    return std::make_tuple(gmax[0], maxblockid, bfuv);
}


//returns min_element, blockids, coordinates of min element in the block
template<typename TensorType>
std::tuple<TensorType, IndexVector, std::vector<size_t>> min_element(ExecutionContext &ec, LabeledTensor<TensorType> ltensor){
    TensorType min = 0.0;

    Tensor<TensorType> tensor = ltensor.tensor();
    auto nmodes = tensor.num_modes();
     //Works for only upto 6D tensors
    EXPECTS(tensor.num_modes() <= 6);

    IndexVector minblockid(nmodes);
    std::vector<size_t> bfuv(2);
    std::vector<TensorType> lmin(2,0);
    std::vector<TensorType> gmin(2,0);

    auto getmin = [&](const IndexVector& bid) {
        const IndexVector blockid =
          internal::translate_blockid(bid, ltensor);
        const tamm::TAMM_SIZE dsize = tensor.block_size(blockid);
        std::vector<TensorType> dbuf(dsize);
        tensor.get(blockid, dbuf);
        auto block_dims   = tensor.block_dims(blockid);
        auto block_offset = tensor.block_offsets(blockid);
        size_t c = 0;

        if(nmodes == 1) {
            for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0];
                i++, c++) {
                if(lmin[0] > dbuf[c]) {
                    lmin[0]    = dbuf[c];
                    lmin[1]    = GA_Nodeid();
                    bfuv[0]    = i - block_offset[0];
                    minblockid = {blockid[0]};
                }
            }
        } else if(nmodes == 2) {
            for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0];
                i++) {
                for(size_t j = block_offset[1];
                    j < block_offset[1] + block_dims[1]; j++, c++) {
                    if(lmin[0] > dbuf[c]) {
                        lmin[0]    = dbuf[c];
                        lmin[1]    = GA_Nodeid();
                        bfuv[0]    = i - block_offset[0];
                        bfuv[1]    = j - block_offset[1];
                        minblockid = {blockid[0], blockid[1]};
                    }
                }
            }
        } else if(nmodes == 3) {
            for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0];
                i++) {
                for(size_t j = block_offset[1];
                    j < block_offset[1] + block_dims[1]; j++) {
                    for(size_t k = block_offset[2];
                        k < block_offset[2] + block_dims[2]; k++, c++) {
                        if(lmin[0] > dbuf[c]) {
                            lmin[0]    = dbuf[c];
                            lmin[1]    = GA_Nodeid();
                            bfuv[0]    = i - block_offset[0];
                            bfuv[1]    = j - block_offset[1];
                            bfuv[2]    = k - block_offset[2];
                            minblockid = {blockid[0], blockid[1], blockid[2]};
                        }
                    }
                }
            }
        } else if(nmodes == 4) {
            for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0];
                i++) {
                for(size_t j = block_offset[1];
                    j < block_offset[1] + block_dims[1]; j++) {
                    for(size_t k = block_offset[2];
                        k < block_offset[2] + block_dims[2]; k++) {
                        for(size_t l = block_offset[3];
                            l < block_offset[3] + block_dims[3]; l++, c++) {
                            if(lmin[0] > dbuf[c]) {
                                lmin[0]    = dbuf[c];
                                lmin[1]    = GA_Nodeid();
                                bfuv[0]    = i - block_offset[0];
                                bfuv[1]    = j - block_offset[1];
                                bfuv[2]    = k - block_offset[2];
                                bfuv[3]    = l - block_offset[3];
                                minblockid = {blockid[0], blockid[1],
                                              blockid[2], blockid[3]};
                            }
                        }
                    }
                }
            }
        } else if(nmodes == 5) {
            for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0];
                i++) {
                for(size_t j = block_offset[1];
                    j < block_offset[1] + block_dims[1]; j++) {
                    for(size_t k = block_offset[2];
                        k < block_offset[2] + block_dims[2]; k++) {
                        for(size_t l = block_offset[3];
                            l < block_offset[3] + block_dims[3]; l++) {
                            for(size_t m = block_offset[4];
                                m < block_offset[4] + block_dims[4]; m++, c++) {
                                if(lmin[0] > dbuf[c]) {
                                    lmin[0]    = dbuf[c];
                                    lmin[1]    = GA_Nodeid();
                                    bfuv[0]    = i - block_offset[0];
                                    bfuv[1]    = j - block_offset[1];
                                    bfuv[2]    = k - block_offset[2];
                                    bfuv[3]    = l - block_offset[3];
                                    bfuv[4]    = m - block_offset[4];
                                    minblockid = {blockid[0], blockid[1],
                                                  blockid[2], blockid[3],
                                                  blockid[4]};
                                }
                            }
                        }
                    }
                }
            }
        }

        else if(nmodes == 6) {
            for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0];
                i++) {
                for(size_t j = block_offset[1];
                    j < block_offset[1] + block_dims[1]; j++) {
                    for(size_t k = block_offset[2];
                        k < block_offset[2] + block_dims[2]; k++) {
                        for(size_t l = block_offset[3];
                            l < block_offset[3] + block_dims[3]; l++) {
                            for(size_t m = block_offset[4];
                                m < block_offset[4] + block_dims[4]; m++) {
                                for(size_t n = block_offset[5];
                                    n < block_offset[5] + block_dims[5];
                                    n++, c++) {
                                    if(lmin[0] > dbuf[c]) {
                                        lmin[0]    = dbuf[c];
                                        lmin[1]    = GA_Nodeid();
                                        bfuv[0]    = i - block_offset[0];
                                        bfuv[1]    = j - block_offset[1];
                                        bfuv[2]    = k - block_offset[2];
                                        bfuv[3]    = l - block_offset[3];
                                        bfuv[4]    = m - block_offset[4];
                                        bfuv[5]    = n - block_offset[5];
                                        minblockid = {blockid[0], blockid[1],
                                                      blockid[2], blockid[3],
                                                      blockid[4], blockid[5]};
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

    };
    block_for(ec, ltensor, getmin);

    MPI_Allreduce(lmin.data(), gmin.data(), 1, MPI_2DOUBLE_PRECISION, MPI_MINLOC, ec.pg().comm());
    MPI_Bcast(minblockid.data(),2,MPI_UNSIGNED,gmin[1],ec.pg().comm());
    MPI_Bcast(bfuv.data(),2,MPI_UNSIGNED_LONG,gmin[1],ec.pg().comm());

    return std::make_tuple(gmin[0], minblockid, bfuv);
}

//// @todo: Implement 
TiledIndexLabel compose_lbl(const TiledIndexLabel& lhs, const TiledIndexLabel& rhs) {
    auto lhs_tis = lhs.tiled_index_space();
    auto rhs_tis = rhs.tiled_index_space();

    auto res_tis = lhs_tis.compose_tis(rhs_tis);

    return res_tis.label("all");
}

/// @todo: Implement 
TiledIndexSpace compose_tis(const TiledIndexSpace& lhs, const TiledIndexSpace& rhs) {
    
    return lhs.compose_tis(rhs);
}

/// @todo: Implement 
TiledIndexLabel invert_lbl(const TiledIndexLabel& lhs) {
    auto lhs_tis = lhs.tiled_index_space();

    return lhs_tis.label("all");
}

/// @todo: Implement 
TiledIndexSpace invert_tis(const TiledIndexSpace& lhs) {

    return lhs.invert_tis();
}

/// @todo: Implement 
template<typename TensorType> 
TensorType invert_tensor(TensorType tens) {
    TensorType res;

    return res;
}

} // namespace tamm

#endif // TAMM_TAMM_UTILS_HPP_
