#pragma once

#include <vector>
#include <chrono>
#include <random>
#include <fstream>
#include <type_traits>
#include <upcxx/upcxx.hpp>
#include <iomanip>
#include <hdf5.h>
// Eigen matrix algebra library
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#undef I

// #define IO_ISIRREG 1
#define TU_SG true
#define TU_SG_IO true

namespace tamm {

// From integer type to integer type
template <typename from>
constexpr typename std::enable_if<std::is_integral<from>::value && std::is_integral<int64_t>::value, int64_t>::type
cd_ncast(const from& value)
{
    return static_cast<int64_t>(value & (static_cast<typename std::make_unsigned<from>::type>(-1)));
}

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
    os << "]";// << std::endl;
    return os;
}

template <typename Arg, typename... Args>
void print_varlist(Arg&& arg, Args&&... args)
{
    std::cout << std::forward<Arg>(arg);
    ((std::cout << ',' << std::forward<Args>(args)), ...);
    std::cout << std::endl;
}

template<typename T>
double compute_tensor_size(const Tensor<T>& tensor) {
    auto lt = tensor();
    double size = 0;
    for(auto it : tensor.loop_nest()) {
        auto blockid   = internal::translate_blockid(it, lt);
        if(!tensor.is_non_zero(blockid)) continue;
        size += tensor.block_size(blockid);
    }
    return size;
}

/**
 * @brief Prints a Tensor object
 *
 * @tparam T template type for Tensor element type
 * @param [in] tensor input Tensor object
 */
template<typename T>
void print_tensor(const Tensor<T>& tensor, std::string filename="") {
    std::stringstream tstring;
    auto lt = tensor();

    auto nz_check =
      [=](const T val) {
        if constexpr(tamm::internal::is_complex_v<T>) {
          if(val.real() > 0.0000000000001 || val.real() < -0.0000000000001) return true;
        }
        else if(val > 0.0000000000001 || val < -0.0000000000001)
          return true;
        return false;
    };

    int ndims = tensor.num_modes();
    std::vector<int64_t> dims;
    for(auto tis: tensor.tiled_index_spaces()) dims.push_back(tis.index_space().num_indices());

    tstring << "tensor dims = " << dims << std::endl;
    tstring << "actual tensor size = " << tensor.size() << std::endl;
    for(auto it : tensor.loop_nest()) {
        auto blockid   = internal::translate_blockid(it, lt);
        if(!tensor.is_non_zero(blockid)) continue;
        TAMM_SIZE size = tensor.block_size(blockid);
        std::vector<T> buf(size);
        tensor.get(blockid, buf);
        auto bdims = tensor.block_dims(blockid);
        auto boffsets = tensor.block_offsets(blockid);
        tstring << "blockid: " << blockid << ", ";
        tstring << "block_offsets: " << boffsets << ", ";
        tstring << "bdims: " << bdims << ", size: " << size << std::endl;
        
        for(TAMM_SIZE i = 0; i < size; i++) {
            if(i%6==0) tstring << "\n";
            if (nz_check(buf[i]))
                tstring << std::fixed << std::setw( 10 ) << std::setprecision(10) << buf[i] << " ";
        }
        tstring << std::endl;
    }

    if(!filename.empty()){
        std::ofstream tos(filename+".txt", std::ios::out);
        if(!tos) std::cerr << "Error opening file " << filename << std::endl;
        tos << tstring.str() << std::endl;
        tos.close();
    }
    else std::cout << tstring.str();
}

template<typename T>
void print_tensor_all(const Tensor<T>& tensor, std::string filename="", std::string name="") {
    std::stringstream tstring;
    auto lt = tensor();

    size_t neles = 0;
    for(auto it : tensor.loop_nest()) {
        auto blockid   = internal::translate_blockid(it, lt);
        if(!tensor.is_non_zero(blockid)) continue;
        TAMM_SIZE size = tensor.block_size(blockid);
        neles += size;
    }

    FILE *fp = fopen("tensor.bin", "a+b");
    assert(fp);
    fwrite(name.c_str(), sizeof(char), name.length() + 1, fp);
    fwrite(&neles, sizeof(neles), 1, fp);

    int ndims = tensor.num_modes();
    std::vector<int64_t> dims;
    for(auto tis: tensor.tiled_index_spaces()) dims.push_back(tis.index_space().num_indices());

    tstring << "tensor dims = " << dims << std::endl;
    tstring << "actual tensor size = " << tensor.size() << std::endl;
    for(auto it : tensor.loop_nest()) {
        auto blockid   = internal::translate_blockid(it, lt);
        if(!tensor.is_non_zero(blockid)) continue;
        TAMM_SIZE size = tensor.block_size(blockid);
        std::vector<T> buf(size);
        tensor.get(blockid, buf);
        auto bdims = tensor.block_dims(blockid);
        auto boffsets = tensor.block_offsets(blockid);
        tstring << "blockid: " << blockid << ", ";
        tstring << "block_offsets: " << boffsets << ", ";
        tstring << "bdims: " << bdims << ", size: " << size << std::endl;
        
        for(TAMM_SIZE i = 0; i < size; i++) {
            fwrite(&buf[i], sizeof(T), 1, fp);
            if(i%6==0) tstring << "\n";
            tstring << std::fixed << std::setw( 10 ) << std::setprecision(10) << buf[i] << " ";
        }
        tstring << std::endl;
    }
    fclose(fp);
    
    if(!filename.empty()){
        std::ofstream tos(filename+".txt", std::ios::out);
        if(!tos) std::cerr << "Error opening file " << filename << std::endl;
        tos << tstring.str() << std::endl;
        tos.close();
    }
    else std::cout << tstring.str();
}

template <typename T>
void print_tensor_reshaped(LabeledTensor<T> l_tensor,
                           const IndexLabelVec &new_labels) {
  EXPECTS(l_tensor.tensor().is_allocated());
  auto ec = l_tensor.tensor().execution_context();

  Tensor<T> new_tensor{new_labels};

  Scheduler sch{*ec};

  sch.allocate(new_tensor)(new_tensor(new_labels) = l_tensor).execute();

  print_tensor_all(new_tensor);
  sch.deallocate(new_tensor).execute();
}

template <typename T>
void print_labeled_tensor(LabeledTensor<T> l_tensor) {
  EXPECTS(l_tensor.tensor().is_allocated());
  auto ec = l_tensor.tensor().execution_context();

  Tensor<T> new_tensor{l_tensor.labels()};

  Scheduler sch{*ec};

  sch.allocate(new_tensor)(new_tensor(l_tensor.labels()) = l_tensor).execute();

  print_tensor_all(new_tensor);
  sch.deallocate(new_tensor).execute();
}

template<typename T>
std::string tensor_to_string(const Tensor<T>& tensor) {
    std::stringstream tstring;
    auto lt = tensor();

    int ndims = tensor.num_modes();
    std::vector<int64_t> dims;
    for(auto tis: tensor.tiled_index_spaces()) dims.push_back(tis.index_space().num_indices());

    tstring << "tensor dims = " << dims << std::endl;
    tstring << "actual tensor size = " << tensor.size() << std::endl;
    for(auto it : tensor.loop_nest()) {
        auto blockid   = internal::translate_blockid(it, lt);
        if(!tensor.is_non_zero(blockid)) continue;
        TAMM_SIZE size = tensor.block_size(blockid);
        std::vector<T> buf(size);
        tensor.get(blockid, buf);
        auto bdims = tensor.block_dims(blockid);
        auto boffsets = tensor.block_offsets(blockid);
        tstring << "blockid: " << blockid << ", ";
        tstring << "block_offsets: " << boffsets << ", ";
        tstring << "bdims: " << bdims << ", size: " << size << std::endl;
        
        for(TAMM_SIZE i = 0; i < size; i++) {
            if(i%6==0) tstring << "\n";
            tstring << std::fixed << std::setw(15) << std::setprecision(15)
                    << buf[i] << " ";
        }
        tstring << std::endl;
    }
    
    return tstring.str();
}

template<typename T>
void print_vector(std::vector<T> vec, std::string filename="") {
    std::stringstream tstring;
    for (size_t i=0;i<vec.size();i++) 
        tstring << i+1 << "\t" << vec[i] << std::endl;

    if(!filename.empty()) {
        std::ofstream tos(filename, std::ios::out);
        if(!tos) std::cerr << "Error opening file " << filename << std::endl;
        tos << tstring.str() << std::endl;
        tos.close();
    }
    else std::cout << tstring.str();
}

template<typename T>
void print_max_above_threshold(const Tensor<T>& tensor, double printtol, std::string filename="") {
    auto lt = tensor();
    std::stringstream tstring;

    for(auto it : tensor.loop_nest()) {
        auto blockid   = internal::translate_blockid(it, lt);
        if(!tensor.is_non_zero(blockid)) continue;
        TAMM_SIZE size = tensor.block_size(blockid);
        std::vector<T> buf(size);
        tensor.get(blockid, buf);
        auto bdims = tensor.block_dims(blockid);
        
        for(TAMM_SIZE i = 0; i < size; i++) {
            if constexpr(tamm::internal::is_complex_v<T>) {
                 if(std::fabs(buf[i].real()) > printtol)
                    tstring << buf[i] << std::endl;
            } else {
               if(std::fabs(buf[i]) > printtol)
                    tstring << buf[i] << std::endl;
            }
        }
        // tstring << std::endl;
    }
    
    if(!filename.empty()) {
        std::ofstream tos(filename, std::ios::out);
        if(!tos) std::cerr << "Error opening file " << filename << std::endl;
        tos << tstring.str() << std::endl;
        tos.close();
    }
    else std::cout << tstring.str();
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

template<typename TensorType>
ExecutionContext& get_ec(LabeledTensor<TensorType> ltensor) {
    return *(ltensor.tensor().execution_context());
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
void update_tensor(Tensor<T> tensor, Func lambda) {
  update_tensor(tensor(),lambda);
}

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
 * @brief Update a value at a given coordinate in a tensor
 * TODO: update local buf directly to avoid get/put
 */
template<typename T>
void update_tensor_val(LabeledTensor<T> ltensor, std::vector<size_t> coord, T val) {

    Tensor<T> tensor = ltensor.tensor();
    const size_t ndims = tensor.num_modes();
    EXPECTS(ndims == coord.size());

    if ((tensor.execution_context())->pg().rank() == 0) {
        LabelLoopNest loop_nest{ltensor.labels()};

        for(auto it : tensor.loop_nest()) {
            auto blockid   = internal::translate_blockid(it, ltensor);
            if(!tensor.is_non_zero(blockid)) continue;
            auto block_dims = tensor.block_dims(blockid);
            auto block_offset = tensor.block_offsets(blockid);
            // std::cout << "blockid: " << blockid << ", ";
            // std::cout << "block_offsets: " << block_offset << ", ";
            // std::cout << "bdims: " << block_dims << ", size: " << tensor.block_size(blockid) << std::endl;
            
            bool vc = true;
            for (size_t x=0;x<ndims;x++) {
                if(!( 
                    (coord[x] >= block_offset[x]) && (coord[x] < (block_offset[x] + block_dims[x]))
                   ) )
                 vc = vc && false;
            }
            if(vc) {
                TAMM_SIZE size = tensor.block_size(blockid);

                std::vector<T> buf(size);
                TAMM_SIZE val_pos = 0;
                for (size_t x=0;x<ndims;x++) {
                    TAMM_SIZE rd = coord[x] - block_offset[x];
                    for (size_t y=x+1;y<ndims;y++) rd *= block_dims[y];
                    val_pos += rd;
                }
 
                tensor.get(blockid, buf);
                buf[val_pos] = val;
                tensor.put(blockid, buf);
                break;
            }
        }
    }
    (tensor.execution_context())->pg().barrier();
}

template<typename T>
void update_tensor_val(Tensor<T>& tensor, std::vector<size_t> coord, T val) {
    update_tensor_val(tensor(),coord,val);
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
void update_tensor_general(Tensor<T> tensor, Func lambda) {
  update_tensor_general(tensor(),lambda);
}

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
/*inline ExecutionContext make_execution_context() {
    ProcGroup* pg = new ProcGroup {ProcGroup::create_coll(upcxx::world())};
    auto* pMM             = MemoryManagerGA::create_coll(pg);
    Distribution_NW* dist = new Distribution_NW();
    RuntimeEngine* re = new RuntimeEngine{};
    ExecutionContext *ec = new ExecutionContext(*pg, dist, pMM, re);
    return *ec;
} */

/**
 * @brief method for getting the sum of the values on the diagonal
 *
 * @returns sum of the diagonal values
 * @warning only defined for NxN tensors
 */
template<typename TensorType>
TensorType trace(Tensor<TensorType> tensor) {
    return trace(tensor());
}

template<typename TensorType>
TensorType trace(LabeledTensor<TensorType> ltensor) {
    ExecutionContext& ec = get_ec(ltensor);
    TensorType lsumd = 0;
    TensorType gsumd = 0;

    Tensor<TensorType> tensor = ltensor.tensor();
    // Defined only for NxN tensors
    EXPECTS(tensor.num_modes() == 2);

    auto gettrace = [&](const IndexVector& bid) {
        const IndexVector blockid = internal::translate_blockid(bid, ltensor);
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
    gsumd = ec.pg().allreduce(&lsumd, upcxx::op_fast_add);
    return gsumd;
}

/**
 * @brief method for getting the sum of the values on the diagonal
 *
 * @returns sum of the diagonal values
 * @warning only defined for NxN tensors
 */
template<typename TensorType>
TensorType trace_sqr(Tensor<TensorType> tensor) {
    return trace_sqr(tensor());
}

template<typename TensorType>
TensorType trace_sqr(LabeledTensor<TensorType> ltensor) {
    ExecutionContext& ec = get_ec(ltensor);
    TensorType lsumd = 0;
    TensorType gsumd = 0;

    Tensor<TensorType> tensor = ltensor.tensor();
    // Defined only for NxN tensors
    EXPECTS(tensor.num_modes() == 2);

    auto gettrace = [&](const IndexVector& bid) {
        const IndexVector blockid = internal::translate_blockid(bid, ltensor);
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
                // sqr of diagonal
                lsumd += buf[i * dim + i] * buf[i * dim + i];
            }
        }
    };
    block_for(ec, ltensor, gettrace);
    gsumd = ec.pg().allreduce(&lsumd, upcxx::op_fast_add);
    return gsumd;
}

/**
 * @brief method for getting the diagonal values in a Tensor
 *
 * @returns the diagonal values
 * @warning only defined for NxN tensors
 */
template<typename TensorType>
std::vector<TensorType> diagonal(Tensor<TensorType> tensor) {
    return diagonal(tensor());
}

template<typename TensorType>
std::vector<TensorType> diagonal(LabeledTensor<TensorType> ltensor) {
    ExecutionContext& ec = get_ec(ltensor);

    std::vector<TensorType> dvec;
    
    if(ec.pg().rank() == 0) {
        Tensor<TensorType> tensor = ltensor.tensor();
        // Defined only for NxN tensors
        EXPECTS(tensor.num_modes() == 2);

        LabelLoopNest loop_nest{ltensor.labels()};

        for(const IndexVector& bid : loop_nest) {
            const IndexVector blockid = internal::translate_blockid(bid, ltensor);

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
                    dvec.push_back(buf[i * dim + i]);
                }
            }
        }
    }

    int dsize = (int) dvec.size();
    ec.pg().broadcast(&dsize, 0);
    if(ec.pg().rank() != 0) dvec.resize(dsize); 
    ec.pg().broadcast(dvec.data(), dsize, 0);

    return dvec;
}

/**
 * @brief method for updating the diagonal values in a Tensor
 *
 * @warning only defined for NxN tensors
 */
template<typename TensorType>
void update_diagonal(Tensor<TensorType> tensor, const std::vector<TensorType>& dvec) {
    update_diagonal(tensor(),dvec);
}

template<typename TensorType>
void update_diagonal(LabeledTensor<TensorType> ltensor, const std::vector<TensorType>& dvec) {
    ExecutionContext& ec = get_ec(ltensor);

    if(ec.pg().rank() == 0) {
        Tensor<TensorType> tensor = ltensor.tensor();
        // Defined only for NxN tensors
        EXPECTS(tensor.num_modes() == 2);

        LabelLoopNest loop_nest{ltensor.labels()};

        for(const IndexVector& bid : loop_nest) {
            const IndexVector blockid = internal::translate_blockid(bid, ltensor);

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
                    buf[i * dim + i] += dvec[p];
                }
                tensor.put(blockid,buf);
            }
        }
    }

}

/**
 * @brief uses a function to fill in elements of a tensor
 *
 * @tparam TensorType the type of the elements in the tensor
 * @param ltensor tensor to operate on
 * @param func function to fill in the tensor with
 */
template<typename TensorType>
void fill_tensor(Tensor<TensorType> tensor,
                 std::function<void(const IndexVector&, span<TensorType>)> func) {
        fill_tensor(tensor(),func);
}

template<typename TensorType>
void fill_tensor(LabeledTensor<TensorType> ltensor,
                 std::function<void(const IndexVector&, span<TensorType>)> func) {
    ExecutionContext& ec = get_ec(ltensor);
    Tensor<TensorType> tensor = ltensor.tensor();

    auto lambda = [&](const IndexVector& bid) {
        const IndexVector blockid   = internal::translate_blockid(bid, ltensor);
        const tamm::TAMM_SIZE dsize = tensor.block_size(blockid);
        std::vector<TensorType> dbuf(dsize);
        // tensor.get(blockid, dbuf);
        func(blockid,dbuf);
        tensor.put(blockid, dbuf);
    };
    block_for(ec, ltensor, lambda);
}


template<typename TensorType>
void fill_sparse_tensor(Tensor<TensorType> tensor,
                 std::function<void(const IndexVector&, span<TensorType>)> func) {
        fill_sparse_tensor(tensor(),func);
}

template<typename TensorType>
void fill_sparse_tensor(LabeledTensor<TensorType> ltensor,
                        std::function<void(const IndexVector&, span<TensorType>)> func) {
    ExecutionContext& ec = get_ec(ltensor);
    Tensor<TensorType> tensor = ltensor.tensor();

    auto lambda = [&](const IndexVector& bid) {
        const tamm::TAMM_SIZE dsize = tensor.block_size(bid);
        const IndexVector blockid   = internal::translate_sparse_blockid(bid, ltensor);
        std::vector<TensorType> dbuf(dsize);
        // tensor.get(blockid, dbuf);
        func(blockid,dbuf);
        
        tensor.put(bid, dbuf);
    };
    block_for(ec, ltensor, lambda);
}

template<typename TensorType>
std::tuple<int, int, int> get_agg_info(ExecutionContext& gec, const int nranks, Tensor<TensorType> tensor, const int nagg_hint) {
    long double nelements = 1;
    //Heuristic: Use 1 agg for every 14gb
    const long double ne_mb = 131072 * 14.0;
    const int ndims = tensor.num_modes();
    for (auto i=0;i<ndims;i++) nelements *= tensor.tiled_index_spaces()[i].index_space().num_indices();
    // nelements = tensor.size();
    int nagg = (nelements / (ne_mb * 1024) ) + 1;
    // const int nnodes = GA_Cluster_nnodes();
    const int ppn = gec.ppn();
    const int avail_nodes = std::min(nranks/ppn + 1,gec.num_nodes());
    if(nagg > avail_nodes) nagg = avail_nodes;
    if(nagg_hint > 0) nagg = nagg_hint;

    int subranks = nagg * ppn;
    if(subranks > nranks) subranks = nranks;

    return std::make_tuple(nagg,ppn,subranks);
}

template<typename TensorType>
std::tuple<int, int, int> get_subgroup_info(ExecutionContext& gec, Tensor<TensorType> tensor, int nagg_hint=0) {

    int nranks = gec.pg().size().value();

    auto [nagg,ppn,subranks] = get_agg_info(gec, nranks,tensor,nagg_hint);
    
    return std::make_tuple(nagg,ppn,subranks);
}

/**
 * @brief convert tamm tensor to N-D GA
 *
 * @tparam TensorType the type of the elements in the tensor
 * @param ec ExecutionContext
 * @param tensor tamm tensor handle
 * @return GA handle
 */
template<typename TensorType>
int tamm_to_ga(ExecutionContext& ec, Tensor<TensorType>& tensor) {
  abort(); // Not supported in UPC++
#if 0
  int ndims = tensor.num_modes();
  std::vector<int64_t> dims;
  std::vector<int64_t> chnks(ndims,-1);

  for(auto tis: tensor.tiled_index_spaces()) dims.push_back(tis.index_space().num_indices());

  int ga_pg_default = GA_Pgroup_get_default();
  GA_Pgroup_set_default(ec.pg().ga_pg());

  auto ga_eltype = to_ga_eltype(tensor_element_type<TensorType>());
  int ga_tens = NGA_Create64(ga_eltype,ndims,&dims[0],const_cast<char*>("iotemp"),&chnks[0]);
  //GA_Zero(ga_tens);
  GA_Pgroup_set_default(ga_pg_default);

    //convert tamm tensor to GA
    auto tamm_ga_lambda = [&](const IndexVector& bid){
        const IndexVector blockid =
        internal::translate_blockid(bid, tensor());

        auto block_dims   = tensor.block_dims(blockid);
        auto block_offset = tensor.block_offsets(blockid);

        const tamm::TAMM_SIZE dsize = tensor.block_size(blockid);
        
        std::vector<int64_t> lo(ndims),hi(ndims),ld(ndims-1);

        for(size_t i=0;i<ndims;i++) lo[i]   = cd_ncast<size_t>(block_offset[i]);
        for(size_t i=0;i<ndims;i++) hi[i]   = cd_ncast<size_t>(block_offset[i] + block_dims[i]-1);
        for(size_t i=1;i<ndims;i++) ld[i-1] = cd_ncast<size_t>(block_dims[i]);

        std::vector<TensorType> sbuf(dsize);
        tensor.get(blockid, sbuf);
        NGA_Put64(ga_tens,&lo[0],&hi[0],&sbuf[0],&ld[0]);
    };

    block_for(ec, tensor(), tamm_ga_lambda);

    return ga_tens;
#endif
}

class upcxx_dist_lock {
    private:
        int rank;
        int nranks;
        upcxx::global_ptr<int64_t> gptr;
        upcxx::atomic_domain<int64_t> ad_i64;
        upcxx::team_id tid;

    public:
        upcxx_dist_lock(upcxx::team& team) : ad_i64({upcxx::atomic_op::compare_exchange}, team) {
            rank = team.rank_me();
            nranks = team.rank_n();

            gptr = upcxx::new_array<int64_t>(1);
            if (!gptr) {
                std::cerr << "Error doing allocation?" << std::endl;
                abort();
            }
            gptr.local()[0] = 0; // unlocked

            {
                //upcxx::persona_scope master_scope(master_mtx,
                //        upcxx::master_persona());
                gptr = upcxx::broadcast(gptr, 0, team).wait();
            }

            tid = team.id();

            //upcxx::persona_scope master_scope(master_mtx,
            //        upcxx::master_persona());
            upcxx::barrier(team);
        }

        void destroy() {
            // upcxx::persona_scope master_scope(master_mtx,
            //         upcxx::master_persona());
            upcxx::barrier(tid.here());
            if (rank == 0) {
                upcxx::delete_array(gptr);
            }
            ad_i64.destroy();
            upcxx::barrier(tid.here());
        }

        void lock() {
            int64_t old = ad_i64.compare_exchange(gptr, 0, 1, std::memory_order_relaxed).wait();
            while (old == 1) {
                upcxx::progress();
                old = ad_i64.compare_exchange(gptr, 0, 1, std::memory_order_relaxed).wait();
            }
        }

        void unlock() {
            int64_t old = ad_i64.compare_exchange(gptr, 1, 0, std::memory_order_relaxed).wait();
            assert(old == 1);
        }
};

//For dense->dlpno
template<typename TensorType>
int tamm_to_ga2(ExecutionContext& ec, Tensor<TensorType>& tensor) {

  int ndims = tensor.num_modes();
  std::vector<int64_t> dims;
  std::vector<int64_t> chnks(ndims,-1);

  for(auto tis: tensor.tiled_index_spaces()) dims.push_back(tis.index_space().num_indices());

  auto ga_eltype = to_ga_eltype(tensor_element_type<TensorType>());
  int ga_tens = NGA_Create64(ga_eltype,ndims,&dims[0],const_cast<char*>("iotemp"),&chnks[0]);
  //GA_Zero(ga_tens);

    //convert tamm tensor to GA
    auto tamm_ga_lambda = [&](const IndexVector& bid){
        const IndexVector blockid =
        internal::translate_blockid(bid, tensor());

        auto block_dims   = tensor.block_dims(blockid);
        auto block_offset = tensor.block_offsets(blockid);

        const tamm::TAMM_SIZE dsize = tensor.block_size(blockid);
        
        std::vector<int64_t> lo(ndims),hi(ndims),ld(ndims-1);

        for(size_t i=0;i<ndims-1;i++) lo[i]   = cd_ncast<size_t>(block_offset[i]);
        for(size_t i=0;i<ndims-1;i++) hi[i]   = cd_ncast<size_t>(block_offset[i] + block_dims[i]-1);
        for(size_t i=1;i<ndims-1;i++) ld[i-1] = cd_ncast<size_t>(block_dims[i]);

        std::vector<TensorType> sbuf(dsize);
        tensor.get(blockid, sbuf);
        NGA_Put64(ga_tens,&lo[0],&hi[0],&sbuf[0],&ld[0]);
    };

    block_for(ec, tensor(), tamm_ga_lambda);

    return ga_tens;
}

template<typename T>
hid_t get_hdf5_dt() {
    using std::is_same_v;

    if constexpr(is_same_v<int, T>)
        return H5T_NATIVE_INT;
    if constexpr(is_same_v<int64_t, T>)
        return H5T_NATIVE_LLONG;        
    else if constexpr(is_same_v<float, T>)
        return H5T_NATIVE_FLOAT;
    else if constexpr(is_same_v<double, T>)
        return H5T_NATIVE_DOUBLE;
    else if constexpr (is_same_v<std::complex<float>, T>) {
      typedef struct {
        float re; /*real part*/
        float im; /*imaginary part*/
      } complex_t;

      hid_t complex_id = H5Tcreate(H5T_COMPOUND, sizeof(complex_t));
      H5Tinsert(complex_id, "real", HOFFSET(complex_t, re), H5T_NATIVE_FLOAT);
      H5Tinsert(complex_id, "imaginary", HOFFSET(complex_t, im),
                H5T_NATIVE_FLOAT);
      return complex_id;
    }
    else if constexpr (is_same_v<std::complex<double>, T>) {
      typedef struct {
        double re; /*real part*/
        double im; /*imaginary part*/
      } complex_t;

      hid_t complex_id = H5Tcreate(H5T_COMPOUND, sizeof(complex_t));
      H5Tinsert(complex_id, "real", HOFFSET(complex_t, re), H5T_NATIVE_DOUBLE);
      H5Tinsert(complex_id, "imaginary", HOFFSET(complex_t, im),
                H5T_NATIVE_DOUBLE);
      return complex_id;
    }
}

/**
 * @brief write tensor to disk using HDF5
 *
 * @tparam TensorType the type of the elements in the tensor
 * @param tensor to write to disk
 * @param filename to write to disk
 */
template<typename TensorType>
void write_to_disk(Tensor<TensorType> tensor, const std::string& filename, 
                    bool tammio=true, bool profile=false, int nagg_hint=0) {

    ExecutionContext& gec = get_ec(tensor());
    auto io_t1 = std::chrono::high_resolution_clock::now();
    int rank = gec.pg().rank().value();

    #ifdef TU_SG_IO
    auto [nagg,ppn,subranks] = get_subgroup_info(gec,tensor);
    upcxx::team* io_comm = new upcxx::team(gec.pg().team()->split(
                gec.pg().rank() < subranks ? 0 : upcxx::team::color_none, 0));
    #else
    auto [nagg,ppn,subranks] = get_agg_info(gec, gec.pg().size().value(),tensor,nagg_hint);
    #endif 

//     // Added by Max Grossman 2022-03-27
//     throw std::runtime_error("write_to_disk unsupported");

#if 0

    int ga_tens;
    if(!tammio) ga_tens = tamm_to_ga(gec,tensor);
    size_t ndims = tensor.num_modes();
    const std::string nppn = std::to_string(nagg) + "n," + std::to_string(ppn) + "ppn";
    if(rank == 0 && profile) std::cout << "write to disk using: " << nppn << std::endl;

    int64_t tensor_dims[7] = {1,1,1,1,1,1,1};
    int ndim{1},itype{};
    ga_tens = tensor.ga_handle();
    NGA_Inquire64(ga_tens,&itype,&ndim,tensor_dims);

    // if ndim=2, this is an nD GA and assumed to be dense.
    int64_t tensor_size = std::accumulate(tensor_dims, tensor_dims+ndim, (int64_t)1, std::multiplies<int64_t>());

    hid_t hdf5_dt = get_hdf5_dt<TensorType>();

    #ifdef TU_SG_IO
    if (rank < subranks) {
        ProcGroup pg = ProcGroup::create_coll(io_comm);
        ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
    #else
        ExecutionContext& ec = gec;
    #endif
        auto ltensor = tensor();
        LabelLoopNest loop_nest{ltensor.labels()};  

        int ierr;
        // MPI_File fh;
        MPI_Info info;
        // MPI_Status status;
        hsize_t file_offset;
        MPI_Info_create(&info);
        MPI_Info_set(info,"cb_nodes",std::to_string(nagg).c_str());    
        // MPI_File_open(ec.pg().comm(), filename.c_str(), MPI_MODE_CREATE|MPI_MODE_WRONLY,
        //             info, &fh);

        /* set the file access template for parallel IO access */
        auto acc_template = H5Pcreate(H5P_FILE_ACCESS);
        // ierr = H5Pset_sieve_buf_size(acc_template, 262144); 
        // ierr = H5Pset_alignment(acc_template, 524288, 262144);
        // ierr = MPI_Info_set(info, "access_style", "write_once");
        // ierr = MPI_Info_set(info, "collective_buffering", "true");
        // ierr = MPI_Info_set(info, "cb_block_size", "1048576");
        // ierr = MPI_Info_set(info, "cb_buffer_size", "4194304");

        /* tell the HDF5 library that we want to use MPI-IO to do the writing */
        ierr = H5Pset_fapl_mpio(acc_template, ec.pg().comm(), info);
        auto file_identifier = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, 
                                         acc_template);

        /* release the file access template */
        ierr = H5Pclose(acc_template);
        ierr = MPI_Info_free(&info);

        int tensor_rank = 1;
        hsize_t dimens_1d = tensor_size;
        auto   dataspace = H5Screate_simple(tensor_rank, &dimens_1d, NULL);
        /* create a dataset collectively */
        auto dataset = H5Dcreate(file_identifier, "tensor", hdf5_dt, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        /* create a file dataspace independently */
        auto file_dataspace = H5Dget_space (dataset);

        /* Create and write additional metadata */
        // std::vector<int> attr_dims{11,29,42};
        // hsize_t attr_size = attr_dims.size();
        // auto attr_dataspace = H5Screate_simple(1, &attr_size, NULL);
        // auto attr_dataset = H5Dcreate(file_identifier, "attr", H5T_NATIVE_INT, attr_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        // H5Dwrite(attr_dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, attr_dims.data());
        // H5Dclose(attr_dataset);
        // H5Sclose(attr_dataspace);

        hid_t xfer_plist;
        /* set up the collective transfer properties list */
        xfer_plist = H5Pcreate (H5P_DATASET_XFER);
        auto ret=H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_INDEPENDENT);

        if(/*is_irreg &&*/ tammio){
            auto lambda = [&](const IndexVector& bid) {
                const IndexVector blockid   = internal::translate_blockid(bid, ltensor);

                file_offset = 0;
                for(const IndexVector& pbid : loop_nest) {
                    bool is_zero = !tensor.is_non_zero(pbid);
                    if(pbid==blockid) {
                        if(is_zero) return; 
                        break;
                    }
                    if(is_zero) continue;
                    file_offset += tensor.block_size(pbid);
                }

                // const tamm::TAMM_SIZE 
                hsize_t dsize = tensor.block_size(blockid);
                std::vector<TensorType> dbuf(dsize);
                tensor.get(blockid, dbuf);

                // std::cout << "WRITE: rank, file_offset, size = " << rank << "," << file_offset << ", " << dsize << std::endl;

                hsize_t stride = 1;
                herr_t ret=H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, &file_offset, &stride,
                                               &dsize, NULL); //stride=NULL?

                // /* create a memory dataspace independently */
                auto mem_dataspace = H5Screate_simple (tensor_rank, &dsize, NULL); 

                // /* write data independently */
                ret = H5Dwrite(dataset, hdf5_dt, mem_dataspace, file_dataspace,
                                xfer_plist, dbuf.data());	

                H5Sclose(mem_dataspace);

            };
        
            block_for(ec, ltensor, lambda);
        }
        else{
                //N-D GA
                auto ga_write_lambda = [&](const IndexVector& bid){
                    const IndexVector blockid   = internal::translate_blockid(bid, ltensor);

                    file_offset = 0;
                    for(const IndexVector& pbid : loop_nest) {
                        bool is_zero = !tensor.is_non_zero(pbid);
                        if(pbid==blockid) {
                            if(is_zero) return; 
                            break;
                        }
                        if(is_zero) continue;
                        file_offset += tensor.block_size(pbid);
                    }

                    // file_offset = file_offset*sizeof(TensorType);

                    auto block_dims   = tensor.block_dims(blockid);
                    auto block_offset = tensor.block_offsets(blockid);

                    hsize_t dsize = tensor.block_size(blockid);

                    std::vector<int64_t> lo(ndims),hi(ndims),ld(ndims-1);

                    for(size_t i=0;i<ndims;i++)  lo[i]   = cd_ncast<size_t>(block_offset[i]);
                    for(size_t i=0;i<ndims;i++)  hi[i]   = cd_ncast<size_t>(block_offset[i] + block_dims[i]-1);
                    for(size_t i=1;i<ndims;i++) ld[i-1] = cd_ncast<size_t>(block_dims[i]);

                    std::vector<TensorType> sbuf(dsize);
                    NGA_Get64(ga_tens,&lo[0],&hi[0],&sbuf[0],&ld[0]);
                    // MPI_File_write_at(fh,file_offset,reinterpret_cast<void*>(&sbuf[0]),
                    //     static_cast<int>(dsize),mpi_type<TensorType>(),&status);

                    hsize_t stride = 1;
                    herr_t ret=H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, &file_offset, &stride,
                                                &dsize, NULL); //stride=NULL?

                    // /* create a memory dataspace independently */
                    auto mem_dataspace = H5Screate_simple (tensor_rank, &dsize, NULL); 

                    // /* write data independently */
                    ret = H5Dwrite(dataset, hdf5_dt, mem_dataspace, file_dataspace,
                                    xfer_plist, sbuf.data());	

                    H5Sclose(mem_dataspace);                    

                };

                block_for(ec, ltensor, ga_write_lambda);
        }
        
        H5Sclose(file_dataspace);
        // H5Sclose(mem_dataspace);
        H5Pclose(xfer_plist);

        H5Dclose(dataset);
        H5Sclose(dataspace);
        H5Fclose(file_identifier);

    #ifdef TU_SG_IO
        ec.flush_and_sync();
        //MemoryManagerGA::destroy_coll(mgr);
        MPI_Comm_free(&io_comm);
        pg.destroy_coll();
    }
    #endif

    gec.pg().barrier();
    if(!tammio) NGA_Destroy(ga_tens);
    auto io_t2 = std::chrono::high_resolution_clock::now();

    double io_time = 
        std::chrono::duration_cast<std::chrono::duration<double>>((io_t2 - io_t1)).count();
    if(rank == 0 && profile) std::cout << "Time for writing " << filename << " to disk (" << nppn << "): " << io_time << " secs" << std::endl;
#endif

}


/**
 * @brief Write batch of tensors to disk using HDF5.
 *        Uses process groups for concurrent writes.
 * @tparam TensorType the type of the elements in the tensor
 * @param tensor to write to disk
 * @param filename to write to disk
 */
template<typename TensorType>
void write_to_disk_group(ExecutionContext& gec, std::vector<Tensor<TensorType>> tensors,
                    std::vector<std::string> filenames,
                    bool profile=false, int nagg_hint=0) {

    EXPECTS(tensors.size() == filenames.size());

    auto io_t1 = std::chrono::high_resolution_clock::now();

    hid_t hdf5_dt = get_hdf5_dt<TensorType>();

    const int  world_rank = gec.pg().rank().value();
    const auto world_size = gec.pg().size().value();
    auto       world_comm = gec.pg().team();

    int nranks        = world_size;
    int color         = -1;
    int prev_subranks = 0;

    std::vector<int> rankspertensor;
    for(size_t i = 0; i < tensors.size(); i++) {
      auto [nagg, ppn] = get_agg_info(nranks, tensors[i], nagg_hint);
      auto subranks    = nagg * ppn;
      if(subranks > nranks) subranks = nranks;
      rankspertensor.push_back(subranks);
      if(world_rank >= prev_subranks && world_rank < (subranks + prev_subranks)) color = i;
      nranks -= subranks;
      if(nranks <= 0) break;
      prev_subranks += subranks;
    }
    if(color == -1) color = MPI_UNDEFINED;

    if(world_rank == 0 && profile) {
      std::cout << "Number of tensors to be written, process groups, sizes: " << tensors.size()
                << "," << rankspertensor.size() << ", " << rankspertensor << std::endl;
    }

    // Added by Max Grossman 2022-03-27
    throw std::runtime_error("write_to_disk_group unsupported");
#if 0
    MPI_Comm io_comm;
    MPI_Comm_split(world_comm, color, world_rank, &io_comm);

    AtomicCounter* ac = new AtomicCounterGA(gec.pg(), 1);
    ac->allocate(0);
    int64_t taskcount = 0;
    int64_t next = -1; 
    // int total_pi_pg = 0;

    if(io_comm != MPI_COMM_NULL) {
      ProcGroup        pg = ProcGroup::create_coll(io_comm);
      ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};

      int root_ppi = -1;
      MPI_Comm_rank(ec.pg().comm(), &root_ppi);

      // int pg_id = rank/subranks;
      if(root_ppi == 0) next = ac->fetch_add(0, 1);
      ec.pg().broadcast(&next, 0);

      for(size_t i = 0; i < tensors.size(); i++) {
        if(next == taskcount) {
          Tensor<TensorType> tensor   = tensors[i];
          auto               filename = filenames[i];

          auto io_t1 = std::chrono::high_resolution_clock::now();

          int    ga_tens;
          size_t ndims = tensor.num_modes();
          // const std::string nppn = std::to_string(nagg) + "n," + std::to_string(ppn) + "ppn";
          // if(root_ppi == 0 && profile)
          //   std::cout << "write " << filename << " to disk using: " << ec.pg().size().value() <<
          //   " ranks" << std::endl;

          int64_t tensor_dims[7] = {1, 1, 1, 1, 1, 1, 1};
          int     ndim{1}, itype{};
          ga_tens = tensor.ga_handle();
          NGA_Inquire64(ga_tens, &itype, &ndim, tensor_dims);

          // if ndim=2, this is an nD GA and assumed to be dense.
          int64_t tensor_size = std::accumulate(tensor_dims, tensor_dims + ndim, (int64_t) 1,
                                                std::multiplies<int64_t>());

          auto          ltensor = tensor();
          LabelLoopNest loop_nest{ltensor.labels()};

          int ierr;
          // MPI_File fh;
          MPI_Info info;
          // MPI_Status status;
          hsize_t file_offset;
          MPI_Info_create(&info);
          // MPI_Info_set(info,"cb_nodes",std::to_string(nagg).c_str());
          // MPI_File_open(ec.pg().comm(), filename.c_str(), MPI_MODE_CREATE|MPI_MODE_WRONLY,
          //             info, &fh);

          /* set the file access template for parallel IO access */
          auto acc_template = H5Pcreate(H5P_FILE_ACCESS);
          // ierr = H5Pset_sieve_buf_size(acc_template, 262144);
          // ierr = H5Pset_alignment(acc_template, 524288, 262144);
          // ierr = MPI_Info_set(info, "access_style", "write_once");
          // ierr = MPI_Info_set(info, "collective_buffering", "true");
          // ierr = MPI_Info_set(info, "cb_block_size", "1048576");
          // ierr = MPI_Info_set(info, "cb_buffer_size", "4194304");

          /* tell the HDF5 library that we want to use MPI-IO to do the writing */
          ierr = H5Pset_fapl_mpio(acc_template, ec.pg().comm(), info);
          auto file_identifier =
            H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, acc_template);

          /* release the file access template */
          ierr = H5Pclose(acc_template);
          ierr = MPI_Info_free(&info);

          int     tensor_rank = 1;
          hsize_t dimens_1d   = tensor_size;
          auto    dataspace   = H5Screate_simple(tensor_rank, &dimens_1d, NULL);
          /* create a dataset collectively */
          auto dataset = H5Dcreate(file_identifier, "tensor", hdf5_dt, dataspace, H5P_DEFAULT,
                                   H5P_DEFAULT, H5P_DEFAULT);
          /* create a file dataspace independently */
          auto file_dataspace = H5Dget_space(dataset);

          /* Create and write additional metadata */
          // std::vector<int> attr_dims{11,29,42};
          // hsize_t attr_size = attr_dims.size();
          // auto attr_dataspace = H5Screate_simple(1, &attr_size, NULL);
          // auto attr_dataset = H5Dcreate(file_identifier, "attr", H5T_NATIVE_INT, attr_dataspace,
          // H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); H5Dwrite(attr_dataset, H5T_NATIVE_INT, H5S_ALL,
          // H5S_ALL, H5P_DEFAULT, attr_dims.data()); H5Dclose(attr_dataset);
          // H5Sclose(attr_dataspace);

          hid_t xfer_plist;
          /* set up the collective transfer properties list */
          xfer_plist = H5Pcreate(H5P_DATASET_XFER);
          auto ret   = H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_INDEPENDENT);

          auto lambda = [&](const IndexVector& bid) {
            const IndexVector blockid = internal::translate_blockid(bid, ltensor);

            file_offset = 0;
            for(const IndexVector& pbid: loop_nest) {
              bool is_zero = !tensor.is_non_zero(pbid);
              if(pbid == blockid) {
                if(is_zero) return;
                break;
              }
              if(is_zero) continue;
              file_offset += tensor.block_size(pbid);
            }

            hsize_t                 dsize = tensor.block_size(blockid);
            std::vector<TensorType> dbuf(dsize);
            tensor.get(blockid, dbuf);

            hsize_t stride = 1;
            herr_t  ret = H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, &file_offset, &stride,
                                             &dsize, NULL); // stride=NULL?

            // /* create a memory dataspace independently */
            auto mem_dataspace = H5Screate_simple(tensor_rank, &dsize, NULL);

            // /* write data independently */
            ret =
              H5Dwrite(dataset, hdf5_dt, mem_dataspace, file_dataspace, xfer_plist, dbuf.data());

            H5Sclose(mem_dataspace);
          };

          block_for(ec, ltensor, lambda);

          H5Sclose(file_dataspace);
          // H5Sclose(mem_dataspace);
          H5Pclose(xfer_plist);

          H5Dclose(dataset);
          H5Sclose(dataspace);
          H5Fclose(file_identifier);

          auto io_t2 = std::chrono::high_resolution_clock::now();

          double io_time =
            std::chrono::duration_cast<std::chrono::duration<double>>((io_t2 - io_t1)).count();
          if(root_ppi == 0 && profile)
            std::cout << "Time for writing " << filename << " to disk (" << ec.pg().size().value()
                      << "): " << io_time << " secs" << std::endl;

          if(root_ppi == 0) next = ac->fetch_add(0, 1);
          ec.pg().broadcast(&next, 0);

        } // next==taskcount

        if(root_ppi == 0) taskcount++;
        ec.pg().broadcast(&taskcount, 0);

      } // loop over tensors

      ec.flush_and_sync();
      MPI_Comm_free(&io_comm);
      // MemoryManagerGA::destroy_coll(mgr);
      pg.destroy_coll();
    } // io_comm != MPI_COMM_NULL

    ac->deallocate();
    delete ac;
    gec.pg().barrier();

    auto io_t2 = std::chrono::high_resolution_clock::now();

    double io_time = 
        std::chrono::duration_cast<std::chrono::duration<double>>((io_t2 - io_t1)).count();
    if(world_rank == 0 && profile)
      std::cout << "Total Time for writing tensors"
                << " to disk: " << io_time << " secs" << std::endl;
#endif
}

/**
 * @brief convert N-D GA to a tamm tensor
 *
 * @tparam TensorType the type of the elements in the tensor
 * @param ec ExecutionContext
 * @param tensor tamm tensor handle
 * @param ga_tens GA handle
 */
template<typename TensorType>
void ga_to_tamm(ExecutionContext& ec, Tensor<TensorType>& tensor, int ga_tens) {
  
    size_t ndims = tensor.num_modes();

    //convert ga to tamm tensor
    auto ga_tamm_lambda = [&](const IndexVector& bid){
        const IndexVector blockid =
        internal::translate_blockid(bid, tensor());

        auto block_dims   = tensor.block_dims(blockid);
        auto block_offset = tensor.block_offsets(blockid);

        const tamm::TAMM_SIZE dsize = tensor.block_size(blockid);

        std::vector<int64_t> lo(ndims),hi(ndims),ld(ndims-1);

        for(size_t i=0;i<ndims;i++)  lo[i]   = cd_ncast<size_t>(block_offset[i]);
        for(size_t i=0;i<ndims;i++)  hi[i]   = cd_ncast<size_t>(block_offset[i] + block_dims[i]-1);
        for(size_t i=1;i<ndims;i++)  ld[i-1] = cd_ncast<size_t>(block_dims[i]);

        std::vector<TensorType> sbuf(dsize);
        NGA_Get64(ga_tens,&lo[0],&hi[0],&sbuf[0],&ld[0]);

        tensor.put(blockid, sbuf);
    };

    block_for(ec, tensor(), ga_tamm_lambda);

    // NGA_Destroy(ga_tens);
}


template<typename TensorType>
Tensor<TensorType> redistribute_tensor(Tensor<TensorType> stensor, TiledIndexSpaceVec tis, std::vector<size_t> spins={}) {
    ExecutionContext& ec = get_ec(stensor());
    // int rank = ec.pg().rank().value();
    int wmn_ga = tamm_to_ga(ec,stensor);
    // sch.deallocate(wmn).execute();
    Tensor<TensorType> dtensor{tis};
    if(spins.size()>0) dtensor = Tensor<TensorType>{tis,spins};
    Tensor<TensorType>::allocate(&ec,dtensor);
    ga_to_tamm(ec,dtensor,wmn_ga);
    NGA_Destroy(wmn_ga);

    return dtensor;
}


/**
 * @brief retile a tamm tensor 
 *
 * @param stensor source tensor
 * @param dtensor tensor after retiling.
 *  Assumed to be created & allocated using the new tiled index space.
 */
template<typename TensorType>
void retile_tamm_tensor(Tensor<TensorType> stensor, Tensor<TensorType>& dtensor, std::string tname="") {
    auto io_t1 = std::chrono::high_resolution_clock::now();

    ExecutionContext& ec = get_ec(stensor());
    int rank = ec.pg().rank().value();

    int ga_tens = tamm_to_ga(ec,stensor);
    ga_to_tamm(ec, dtensor, ga_tens);
    NGA_Destroy(ga_tens);

    auto io_t2 = std::chrono::high_resolution_clock::now();

    double rt_time = 
        std::chrono::duration_cast<std::chrono::duration<double>>((io_t2 - io_t1)).count();
    if(rank == 0 && !tname.empty()) std::cout << "Time to re-tile " << tname << " tensor: " << rt_time << " secs" << std::endl;
}

/**
 * @brief read tensor from disk using HDF5
 *
 * @tparam TensorType the type of the elements in the tensor
 * @param tensor to read into 
 * @param filename to read from disk
 */
template<typename TensorType>
void read_from_disk(Tensor<TensorType> tensor, const std::string& filename, 
                    bool tammio=true, Tensor<TensorType> wtensor={}, 
                    bool profile=false, int nagg_hint=0) {

    ExecutionContext& gec = get_ec(tensor());
    auto io_t1 = std::chrono::high_resolution_clock::now();
    int rank = gec.pg().rank().value();

    // Added by Max Grossman 2022-03-27
    throw std::runtime_error("read_from_disk unsupported");

#if 0
    #ifdef TU_SG_IO
    MPI_Comm io_comm;
    auto [nagg,ppn] = get_subgroup_info(gec,tensor,io_comm,nagg_hint);
    #else
    auto [nagg,ppn] = get_agg_info(gec.pg().size().value(),tensor,nagg_hint);
    #endif

    const std::string nppn = std::to_string(nagg) + "n," + std::to_string(ppn) + "ppn";
    if(rank == 0 && profile) std::cout << "read from disk using: " << nppn << std::endl;

    int ga_tens;
    
    if(!tammio) {
        auto tis_dims = tensor.tiled_index_spaces();

        int ndims = tensor.num_modes();
        std::vector<int64_t> dims;
        std::vector<int64_t> chnks(ndims,-1);
        for(auto tis: tis_dims) dims.push_back(tis.index_space().num_indices());
                
        ga_tens = NGA_Create64(to_ga_eltype(tensor_element_type<TensorType>()),
                    ndims,&dims[0],const_cast<char*>("iotemp"),&chnks[0]);
    }

    hid_t hdf5_dt = get_hdf5_dt<TensorType>();

    auto tensor_back = tensor;

    #ifdef TU_SG_IO
    if(io_comm != MPI_COMM_NULL) {
        ProcGroup pg = ProcGroup::create_coll(io_comm);
        ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
    #else
        ExecutionContext& ec = gec;
    #endif

        if(wtensor.num_modes()>0) 
            tensor = wtensor;

        auto ltensor = tensor();
        LabelLoopNest loop_nest{ltensor.labels()};     

        int ierr;
        // MPI_File fh;
        MPI_Info info;
        // MPI_Status status;
        hsize_t file_offset;
        MPI_Info_create(&info);
        // MPI_Info_set(info,"romio_cb_read", "enable");
        // MPI_Info_set(info,"striping_unit","4194304"); 
        MPI_Info_set(info,"cb_nodes",std::to_string(nagg).c_str());    

        // MPI_File_open(ec.pg().comm(), filename.c_str(), MPI_MODE_RDONLY,
        //                 info, &fh);               

        /* set the file access template for parallel IO access */
        auto acc_template = H5Pcreate(H5P_FILE_ACCESS);

        /* tell the HDF5 library that we want to use MPI-IO to do the reading */
        ierr = H5Pset_fapl_mpio(acc_template, ec.pg().comm(), info);
        auto file_identifier = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, acc_template);

        /* release the file access template */
        ierr = H5Pclose(acc_template);
        ierr = MPI_Info_free(&info);

        int tensor_rank = 1;
        // hsize_t dimens_1d = tensor_size;
        /* create a dataset collectively */
        auto dataset = H5Dopen(file_identifier, "tensor", H5P_DEFAULT);
        /* create a file dataspace independently */
        auto file_dataspace = H5Dget_space (dataset);

        /* Read additional metadata */
        // std::vector<int> attr_dims(3);
        // auto attr_dataset = H5Dopen(file_identifier, "attr",  H5P_DEFAULT);
        // H5Dread(attr_dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, attr_dims.data());
        // H5Dclose(attr_dataset);

        hid_t xfer_plist;
        /* set up the collective transfer properties list */
        xfer_plist = H5Pcreate (H5P_DATASET_XFER);
        auto ret=H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_INDEPENDENT);

        if(/*is_irreg &&*/ tammio) {
            auto lambda = [&](const IndexVector& bid) {
                const IndexVector blockid   = internal::translate_blockid(bid, ltensor);

                file_offset = 0;
                for(const IndexVector& pbid : loop_nest) {
                    bool is_zero = !tensor.is_non_zero(pbid);
                    if(pbid==blockid) {
                        if(is_zero) return; 
                        break;
                    }
                    if(is_zero) continue;
                    file_offset += tensor.block_size(pbid);
                }

                // file_offset = file_offset*sizeof(TensorType);

                hsize_t dsize = tensor.block_size(blockid);
                std::vector<TensorType> dbuf(dsize);

                // std::cout << "READ: rank, file_offset, size = " << rank << "," << file_offset << ", " << dsize << std::endl;

                hsize_t stride = 1;
                herr_t ret=H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, &file_offset, &stride,
                                               &dsize, NULL); //stride=NULL?

                // /* create a memory dataspace independently */
                auto mem_dataspace = H5Screate_simple (tensor_rank, &dsize, NULL); 

                // MPI_File_read_at(fh,file_offset,reinterpret_cast<void*>(&dbuf[0]),
                //                 static_cast<int>(dsize),mpi_type<TensorType>(),&status);

                // /* read data independently */
                ret = H5Dread(dataset, hdf5_dt, mem_dataspace, file_dataspace,
                                xfer_plist, dbuf.data());	

                tensor.put(blockid,dbuf);

                H5Sclose(mem_dataspace);

            };

                block_for(ec, ltensor, lambda);
        }
         else {
                auto ga_read_lambda = [&](const IndexVector& bid){
                    const IndexVector blockid =
                    internal::translate_blockid(bid, tensor());

                    file_offset = 0;
                    for(const IndexVector& pbid : loop_nest) {
                        bool is_zero = !tensor.is_non_zero(pbid);
                        if(pbid==blockid) {
                            if(is_zero) return; 
                            break;
                        }
                        if(is_zero) continue;
                        file_offset += tensor.block_size(pbid);
                    }

                    // file_offset = file_offset*sizeof(TensorType);

                    auto block_dims   = tensor.block_dims(blockid);
                    auto block_offset = tensor.block_offsets(blockid);

                    hsize_t dsize = tensor.block_size(blockid);

                    size_t ndims = block_dims.size();
                    std::vector<int64_t> lo(ndims),hi(ndims),ld(ndims-1);

                    for(size_t i=0;i<ndims;i++)  lo[i]   = cd_ncast<size_t>(block_offset[i]);
                    for(size_t i=0;i<ndims;i++)  hi[i]   = cd_ncast<size_t>(block_offset[i] + block_dims[i]-1);
                    for(size_t i=1;i<ndims;i++)  ld[i-1] = cd_ncast<size_t>(block_dims[i]);

                    std::vector<TensorType> sbuf(dsize);

                    // MPI_File_read_at(fh,file_offset,reinterpret_cast<void*>(&sbuf[0]),
                    //             static_cast<int>(dsize),mpi_type<TensorType>(),&status);
                    hsize_t stride = 1;
                    herr_t ret=H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, &file_offset, &stride,
                                                &dsize, NULL); //stride=NULL?

                    // /* create a memory dataspace independently */
                    auto mem_dataspace = H5Screate_simple (tensor_rank, &dsize, NULL); 

                    // MPI_File_read_at(fh,file_offset,reinterpret_cast<void*>(&dbuf[0]),
                    //                 static_cast<int>(dsize),mpi_type<TensorType>(),&status);

                    // /* read data independently */
                    ret = H5Dread(dataset, hdf5_dt, mem_dataspace, file_dataspace,
                                    xfer_plist, sbuf.data());	                    
                    
                    NGA_Put64(ga_tens,&lo[0],&hi[0],&sbuf[0],&ld[0]);
                   
                };
                
                block_for(ec, tensor(), ga_read_lambda);
        }

        H5Sclose(file_dataspace);
        // H5Sclose(mem_dataspace);
        H5Pclose(xfer_plist);

        H5Dclose(dataset);
        H5Fclose(file_identifier);

    #ifdef TU_SG_IO
        ec.flush_and_sync();
        //MemoryManagerGA::destroy_coll(mgr);
        MPI_Comm_free(&io_comm);
        pg.destroy_coll();
        // MPI_File_close(&fh);
    }
    #endif

    tensor = tensor_back;

    gec.pg().barrier();

    if(!tammio){
        ga_to_tamm(gec,tensor,ga_tens);
        NGA_Destroy(ga_tens);
    }

    auto io_t2 = std::chrono::high_resolution_clock::now();

    double io_time = 
        std::chrono::duration_cast<std::chrono::duration<double>>((io_t2 - io_t1)).count();
    if(rank == 0 && profile) std::cout << "Time for reading " << filename << " from disk (" << nppn << "): " << io_time << " secs" << std::endl;
#endif
}


/**
 * @brief Read batch of tensors from disk using HDF5.
 *        Uses process groups for concurrent reads.
 * @tparam TensorType the type of the elements in the tensor
 * @param tensor to read into 
 * @param filename to read from disk
 */
template<typename TensorType>
void read_from_disk_group(ExecutionContext& gec, std::vector<Tensor<TensorType>> tensors,
                    std::vector<std::string> filenames, std::vector<Tensor<TensorType>> wtensors={}, 
                    bool profile=false, int nagg_hint=0) {

    EXPECTS(tensors.size() == filenames.size());

    // Added by Max Grossman 2022-03-27
    throw std::runtime_error("read_from_disk_group unsupported");
#if 0
    auto io_t1 = std::chrono::high_resolution_clock::now();

    hid_t hdf5_dt = get_hdf5_dt<TensorType>();

    const int  world_rank = gec.pg().rank().value();
    const auto world_size = gec.pg().size().value();
    auto       world_comm = gec.pg().comm();

    int nranks        = world_size;
    int color         = -1;
    int prev_subranks = 0;

    std::vector<int> rankspertensor;
    for(size_t i = 0; i < tensors.size(); i++) {
      auto [nagg, ppn] = get_agg_info(nranks, tensors[i], nagg_hint);
      auto subranks    = nagg * ppn;
      if(subranks > nranks) subranks = nranks;
      rankspertensor.push_back(subranks);
      if(world_rank >= prev_subranks && world_rank < (subranks + prev_subranks)) color = i;
      nranks -= subranks;
      if(nranks <= 0) break;
      prev_subranks += subranks;
    }
    if(color == -1) color = MPI_UNDEFINED;

    if(world_rank == 0 && profile) {
      std::cout << "Number of tensors to be read, process groups, sizes: " << tensors.size() << ","
                << rankspertensor.size() << ", " << rankspertensor << std::endl;
    }

    MPI_Comm io_comm;
    MPI_Comm_split(world_comm, color, world_rank, &io_comm);

    AtomicCounter* ac = new AtomicCounterGA(gec.pg(), 1);
    ac->allocate(0);
    int64_t taskcount = 0;
    int64_t next = -1; 
    // int total_pi_pg = 0;

    if(io_comm != MPI_COMM_NULL) {
      ProcGroup        pg = ProcGroup::create_coll(io_comm);
      ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};

      int root_ppi = -1;
      MPI_Comm_rank(ec.pg().comm(), &root_ppi);

      // int pg_id = rank/subranks;
      if(root_ppi == 0) next = ac->fetch_add(0, 1);
      ec.pg().broadcast(&next, 0);

      bool is_wt = wtensors.empty();
      for(size_t i = 0; i < tensors.size(); i++) {
        if(next == taskcount) {
          auto io_t1 = std::chrono::high_resolution_clock::now();

          Tensor<TensorType> tensor   = tensors[i];
          auto               filename = filenames[i];

          // auto tensor_back = tensor;

          if(!is_wt) {
            if(wtensors[i].num_modes() > 0) tensor = wtensors[i];
          }

          auto          ltensor = tensor();
          LabelLoopNest loop_nest{ltensor.labels()};

          int ierr;
          // MPI_File fh;
          MPI_Info info;
          // MPI_Status status;
          hsize_t file_offset;
          MPI_Info_create(&info);
          // MPI_Info_set(info,"romio_cb_read", "enable");
          // MPI_Info_set(info,"striping_unit","4194304");
          // MPI_Info_set(info,"cb_nodes",std::to_string(nagg).c_str());

          // MPI_File_open(ec.pg().comm(), filename.c_str(), MPI_MODE_RDONLY,
          //                 info, &fh);

          /* set the file access template for parallel IO access */
          auto acc_template = H5Pcreate(H5P_FILE_ACCESS);

          /* tell the HDF5 library that we want to use MPI-IO to do the reading */
          ierr                 = H5Pset_fapl_mpio(acc_template, ec.pg().comm(), info);
          auto file_identifier = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, acc_template);

          /* release the file access template */
          ierr = H5Pclose(acc_template);
          ierr = MPI_Info_free(&info);

          int tensor_rank = 1;
          // hsize_t dimens_1d = tensor_size;
          /* create a dataset collectively */
          auto dataset = H5Dopen(file_identifier, "tensor", H5P_DEFAULT);
          /* create a file dataspace independently */
          auto file_dataspace = H5Dget_space(dataset);

          /* Read additional metadata */
          // std::vector<int> attr_dims(3);
          // auto attr_dataset = H5Dopen(file_identifier, "attr",  H5P_DEFAULT);
          // H5Dread(attr_dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, attr_dims.data());
          // H5Dclose(attr_dataset);

          hid_t xfer_plist;
          /* set up the collective transfer properties list */
          xfer_plist = H5Pcreate(H5P_DATASET_XFER);
          auto ret   = H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_INDEPENDENT);

          auto lambda = [&](const IndexVector& bid) {
            const IndexVector blockid   = internal::translate_blockid(bid, ltensor);

            file_offset = 0;
            for(const IndexVector& pbid : loop_nest) {
              bool is_zero = !tensor.is_non_zero(pbid);
              if(pbid == blockid) {
                if(is_zero) return;
                break;
              }
              if(is_zero) continue;
              file_offset += tensor.block_size(pbid);
            }

            // file_offset = file_offset*sizeof(TensorType);

            hsize_t dsize = tensor.block_size(blockid);
            std::vector<TensorType> dbuf(dsize);

            // std::cout << "READ: rank, file_offset, size = " << rank << "," << file_offset << ", " << dsize << std::endl;

            hsize_t stride = 1;
            herr_t ret=H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, &file_offset, &stride,
                                            &dsize, NULL); //stride=NULL?

            // /* create a memory dataspace independently */
            auto mem_dataspace = H5Screate_simple (tensor_rank, &dsize, NULL); 

            // MPI_File_read_at(fh,file_offset,reinterpret_cast<void*>(&dbuf[0]),
            //                 static_cast<int>(dsize),mpi_type<TensorType>(),&status);

            // /* read data independently */
            ret = H5Dread(dataset, hdf5_dt, mem_dataspace, file_dataspace,
                            xfer_plist, dbuf.data());	

            tensor.put(blockid,dbuf);

            H5Sclose(mem_dataspace);
          };

          block_for(ec, ltensor, lambda);

          H5Sclose(file_dataspace);
          // H5Sclose(mem_dataspace);
          H5Pclose(xfer_plist);

          H5Dclose(dataset);
          H5Fclose(file_identifier);

          // tensor = tensor_back;

          auto io_t2 = std::chrono::high_resolution_clock::now();

          double io_time =
            std::chrono::duration_cast<std::chrono::duration<double>>((io_t2 - io_t1)).count();
          if(root_ppi == 0 && profile)
            std::cout << "Time for reading " << filename << " from disk (" << ec.pg().size().value()
                      << "): " << io_time << " secs" << std::endl;

          if(root_ppi == 0) next = ac->fetch_add(0, 1);
          ec.pg().broadcast(&next, 0);

        } //next==taskcount

        if(root_ppi == 0) taskcount++;
        ec.pg().broadcast(&taskcount, 0);

      } // loop over tensors

      ec.flush_and_sync();
      MPI_Comm_free(&io_comm);
      // MemoryManagerGA::destroy_coll(mgr);
      pg.destroy_coll();
    } // iocomm!=MPI_COMM_NULL

    ac->deallocate();
    delete ac;
    gec.pg().barrier();

    auto io_t2 = std::chrono::high_resolution_clock::now();

    double io_time = 
        std::chrono::duration_cast<std::chrono::duration<double>>((io_t2 - io_t1)).count();
    if(world_rank == 0 && profile)
      std::cout << "Total Time for reading tensors"
                << " from disk: " << io_time << " secs" << std::endl;
#endif
}

template<typename T>
void dlpno_to_dense(Tensor<T> src, Tensor<T> dst){
    //T1_dlpno(a_ii, ii) -> T1_dense(a, i)
    //V*(O*O) -> V*O -- if (ii / O == ii % O) i = ii / O;
    //T2_dlpno(a_ij, b_ij, ij) -> T2_dense(a, b, i, j)
    //V*V*(O*O) -> V*V*O*O -- i = ij / O, j = ij % O
    int ndims = dst.num_modes();
    // ExecutionContext& ec = get_ec(src());
    // int ga_src = tamm_to_ga(ec,src);
    // if(is_2D) ga_to_tamm(ec, dst, ga_src);
    // else ga_to_tamm2(ec, dst, ga_src);
    // NGA_Destroy(ga_src);
    std::vector<int> dims_dst;
    for(auto tis: dst.tiled_index_spaces()) dims_dst.push_back(tis.index_space().num_indices());  
    int V = dims_dst[0];

    if(ndims == 2){
        int O = dims_dst[1];          
        using Tensor2D = Eigen::Tensor<T, 2, Eigen::RowMajor>;        
        Tensor2D dense_eig(V,O);
        Tensor2D dlpno_eig(V,O*O);
        tamm_to_eigen_tensor(src,dlpno_eig);
        for (int a = 0;a<V;a++)
          for (int i = 0; i < O; i++) {
            int ii = i * O + i;
            dense_eig(a, i) = dlpno_eig(a, ii);
          }
        eigen_to_tamm_tensor(dst,dense_eig);
    }    
    else if(ndims == 4){
        int O = dims_dst[2];  
        using Tensor3D = Eigen::Tensor<T, 3, Eigen::RowMajor>;
        using Tensor4D = Eigen::Tensor<T, 4, Eigen::RowMajor>;        
        Tensor4D dense_eig(V,V,O,O);
        Tensor3D dlpno_eig(V,V,O*O);
        tamm_to_eigen_tensor(src,dlpno_eig);
        for (int a = 0;a<V;a++) // a
        for (int b = 0;b<V;b++) // b
        for (int ij = 0;ij<O*O;ij++) // ij
            dense_eig(a,b,ij/O,ij%O) = dlpno_eig(a,b,ij);
        eigen_to_tamm_tensor(dst,dense_eig);
    }        
    else NOT_ALLOWED();

}

template<typename T>
void dense_to_dlpno(Tensor<T> src, Tensor<T> dst){
    //T1_dense(a, i) -> T1_dlpno(a_ii, ii) 
    //V*O -> V*(O*O) -- ii = i * O + i 
    //T2_dense(a, b, i, j) -> T2_dlpno(a_ij, b_ij, ij)
    //V*V*O*O -->  V*V*(O*O) -- ij = i * O + j
    int ndims = src.num_modes();
    // ExecutionContext& ec = get_ec(src());
    // int ga_src;
    // if(is_2D) ga_src = tamm_to_ga(ec,src);
    // else ga_src = tamm_to_ga2(ec,src);
    // ga_to_tamm(ec, dst, ga_src);
    // NGA_Destroy(ga_src);
  std::vector<int> dims;
  for(auto tis: src.tiled_index_spaces()) 
    dims.push_back(tis.index_space().num_indices());    
  if(ndims == 2){
      using Tensor2D = Eigen::Tensor<T, 2, Eigen::RowMajor>;        
      Tensor2D dense_eig(dims[0],dims[1]);
      Tensor2D dlpno_eig(dims[0],dims[1]*dims[1]);
      dlpno_eig.setZero();

      tamm_to_eigen_tensor(src,dense_eig);
      for (int a = 0;a<dims[0];a++)
      for (int i = 0;i<dims[1];i++) {
        int ii = i*dims[1] + i;
        dlpno_eig(a,ii) = dense_eig(a,i);
      }
      eigen_to_tamm_tensor(dst,dlpno_eig);
  }
  else if(ndims == 4){
      using Tensor3D = Eigen::Tensor<T, 3, Eigen::RowMajor>;
      using Tensor4D = Eigen::Tensor<T, 4, Eigen::RowMajor>;
      Tensor4D dense_eig(dims[0],dims[1],dims[2],dims[3]);
      Tensor3D dlpno_eig(dims[0],dims[1],dims[2]*dims[3]);
      tamm_to_eigen_tensor(src,dense_eig);
      for (int a = 0;a<dims[0];a++)
      for (int b = 0;b<dims[1];b++)
      for (int i = 0;i<dims[2];i++)
      for (int j = 0;j<dims[3];j++) {
          int ij = i * dims[3] + j;
          dlpno_eig(a,b,ij) = dense_eig(a,b,i,j);
      }       
      eigen_to_tamm_tensor(dst,dlpno_eig);        
  }
  else NOT_ALLOWED();
}

template<typename TensorType>
TensorType linf_norm(Tensor<TensorType> tensor) {
    return linf_norm(tensor());
}

template<typename TensorType>
TensorType linf_norm(LabeledTensor<TensorType> ltensor) {
    ExecutionContext& gec = get_ec(ltensor);

    TensorType linfnorm         = 0;
    TensorType glinfnorm        = 0;
    Tensor<TensorType> tensor = ltensor.tensor();

    #ifdef TU_SG
    int rank = gec.pg().rank().value();
    auto [nagg,ppn,subranks] = get_subgroup_info(gec,tensor);
    upcxx::team* sub_comm = new upcxx::team(gec.pg().team()->split(
                    rank < subranks ? 0 : upcxx::team::color_none, 0));

    if (rank < subranks) {
        ProcGroup pg = ProcGroup::create_coll(*sub_comm);
        ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
    #else
        ExecutionContext& ec = gec;
    #endif
        auto getnorm = [&](const IndexVector& bid) {
            const IndexVector blockid   = internal::translate_blockid(bid, ltensor);
            const tamm::TAMM_SIZE dsize = tensor.block_size(blockid);
            std::vector<TensorType> dbuf(dsize);
            tensor.get(blockid, dbuf);
            for(TensorType val : dbuf) {
                TensorType aval = std::fabs(val);
                if(linfnorm < aval) linfnorm = aval;
            }
        };
        block_for(ec, ltensor, getnorm);

    #ifdef TU_SG
        ec.flush_and_sync();
        //MemoryManagerGA::destroy_coll(mgr);
        pg.destroy_coll();
        {
            //upcxx::persona_scope master_scope(master_mtx,
            //        upcxx::master_persona());
            sub_comm->destroy();
        }
    }
    #endif

    gec.pg().barrier();

    glinfnorm = gec.pg().allreduce(&linfnorm, upcxx::op_fast_max);
    return glinfnorm;
}

template <typename T>
void write_to_disk_hdf5(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic , Eigen::RowMajor> eigen_tensor,
                        std::string filename, bool write1D = false) {
  std::string outputfile = filename + ".data";
  hid_t file_id =
      H5Fcreate(outputfile.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  T *buf = eigen_tensor.data();
  hid_t dataspace_id;

  std::vector<hsize_t> dims(2);
  dims[0] = eigen_tensor.rows();
  dims[1] = eigen_tensor.cols();
  int rank = 2;
  dataspace_id = H5Screate_simple(rank, dims.data(), NULL);

  hid_t dataset_id = H5Dcreate(file_id, "data", get_hdf5_dt<T>(), dataspace_id,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  /* herr_t status = */ H5Dwrite(dataset_id, get_hdf5_dt<T>(), H5S_ALL, H5S_ALL,
                                 H5P_DEFAULT, buf);

  /* Create and write attribute information - reduced dims */
  std::vector<int> reduced_dims{static_cast<int>(dims[0]),static_cast<int>(dims[1])};
  hsize_t attr_size = reduced_dims.size();
  auto attr_dataspace = H5Screate_simple(1, &attr_size, NULL);
  auto attr_dataset = H5Dcreate(file_id, "rdims", H5T_NATIVE_INT, attr_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(attr_dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, reduced_dims.data());
  H5Dclose(attr_dataset);
  H5Sclose(attr_dataspace);

  H5Dclose(dataset_id);
  H5Sclose(dataspace_id);
  H5Fclose(file_id);
}

template <typename T, int N>
void write_to_disk_hdf5(Eigen::Tensor<T, N, Eigen::RowMajor> eigen_tensor,
                        std::string filename, bool write1D = false) {
  std::string outputfile = filename + ".data";
  hid_t file_id =
      H5Fcreate(outputfile.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  
  T *buf = eigen_tensor.data();

  hid_t dataspace_id;
  auto dims = eigen_tensor.dimensions();

  if (write1D) {
    hsize_t total_size = 1;
    for (int i=0; i<N; i++) {
      total_size *= dims[i];
    }
    dataspace_id = H5Screate_simple(1, &total_size, NULL);
  } else {
    std::vector<hsize_t> hdims;
    for (int i=0; i<N; i++) {
      hdims.push_back(dims[i]);
    }
    int rank = eigen_tensor.NumDimensions;
    dataspace_id = H5Screate_simple(rank, hdims.data(), NULL);
  }

  hid_t dataset_id = H5Dcreate(file_id, "data", get_hdf5_dt<T>(), dataspace_id,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  /* herr_t status = */ H5Dwrite(dataset_id, get_hdf5_dt<T>(), H5S_ALL, H5S_ALL,
                                 H5P_DEFAULT, buf);

  /* Create and write attribute information - reduced dims */
  std::vector<int> reduced_dims{static_cast<int>(dims[0]),static_cast<int>(dims[1])};
  hsize_t attr_size = reduced_dims.size();
  auto attr_dataspace = H5Screate_simple(1, &attr_size, NULL);
  auto attr_dataset = H5Dcreate(file_id, "rdims", H5T_NATIVE_INT, attr_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(attr_dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, reduced_dims.data());
  H5Dclose(attr_dataset);
  H5Sclose(attr_dataspace);

  H5Dclose(dataset_id);
  H5Sclose(dataspace_id);
  H5Fclose(file_id);
}

template <typename T, int N>
void write_to_disk_hdf5(Tensor<T> tensor, std::string filename,
                        bool write1D = false) {
  std::string outputfile = filename + ".data";
  hid_t file_id =
      H5Fcreate(outputfile.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  // Eigen::Tensor<T, N, Eigen::RowMajor> eigen_tensor = tamm_to_eigen_tensor<T,N>(tensor);
  std::array<Eigen::Index, N> dims;
  const auto &tindices = tensor.tiled_index_spaces();
  for (int i = 0; i < N; i++) {
    dims[i] = tindices[i].max_num_indices();
  }
  Eigen::Tensor<T, N, Eigen::RowMajor> eigen_tensor;
  eigen_tensor = eigen_tensor.reshape(dims);
  eigen_tensor.setZero();

  tamm_to_eigen_tensor(tensor, eigen_tensor);
  T *buf = eigen_tensor.data();

  hid_t dataspace_id;

  if (write1D) {
    hsize_t total_size = 1;
    for (int i=0; i<N; i++) {
      total_size *= dims[i];
    }
    dataspace_id = H5Screate_simple(1, &total_size, NULL);
  } else {
    std::vector<hsize_t> hdims;
    for (int i=0; i<N; i++) {
      hdims.push_back(dims[i]);
    }
    int rank = eigen_tensor.NumDimensions;
    dataspace_id = H5Screate_simple(rank, hdims.data(), NULL);
  }

  hid_t dataset_id = H5Dcreate(file_id, "data", get_hdf5_dt<T>(), dataspace_id,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  /* herr_t status = */ H5Dwrite(dataset_id, get_hdf5_dt<T>(), H5S_ALL, H5S_ALL,
                                 H5P_DEFAULT, buf);

  /* Create and write attribute information - reduced dims */
  std::vector<int> reduced_dims{static_cast<int>(dims[0]),static_cast<int>(dims[1])};
  hsize_t attr_size = reduced_dims.size();
  auto attr_dataspace = H5Screate_simple(1, &attr_size, NULL);
  auto attr_dataset = H5Dcreate(file_id, "rdims", H5T_NATIVE_INT, attr_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(attr_dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, reduced_dims.data());
  H5Dclose(attr_dataset);
  H5Sclose(attr_dataspace);
  
  H5Dclose(dataset_id);
  H5Sclose(dataspace_id);
  H5Fclose(file_id);
}

/**
 * @brief applies a function elementwise to a tensor
 *
 * @tparam TensorType the type of the elements in the tensor
 * @param ltensor tensor to operate on
 * @param func function to be applied to each element
 */
template<typename TensorType>
void apply_ewise_ip(LabeledTensor<TensorType> ltensor,
                 std::function<TensorType(TensorType)> func) {
    ExecutionContext& gec = get_ec(ltensor);
    Tensor<TensorType> tensor = ltensor.tensor();

    #ifdef TU_SG
    auto [nagg,ppn,subranks] = get_subgroup_info(gec,tensor);
    upcxx::team *sub_comm = new upcxx::team(gec.pg().team()->split(
            gec.pg().rank() < subranks ? 0 : upcxx::team::color_none, 0));
    if (gec.pg().rank() < subranks) {
        ProcGroup pg = ProcGroup::create_coll(*sub_comm);
        ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
    #else 
        ExecutionContext& ec = gec;
    #endif

        auto lambda = [&](const IndexVector& bid) {
            const IndexVector blockid   = internal::translate_blockid(bid, ltensor);
            const tamm::TAMM_SIZE dsize = tensor.block_size(blockid);
            std::vector<TensorType> dbuf(dsize);
            tensor.get(blockid, dbuf);
            for(size_t c = 0; c < dsize; c++) dbuf[c] = func(dbuf[c]);
            tensor.put(blockid, dbuf);
        };
        block_for(ec, ltensor, lambda);
    
    #ifdef TU_SG
        ec.flush_and_sync();
        //MemoryManagerGA::destroy_coll(mgr);
        pg.destroy_coll();
    }
    sub_comm->destroy();
    #endif
    gec.pg().barrier();
}

// Several convenience functions using apply_ewise_ip. 
// These routines update the tensor in-place
template<typename TensorType>
void conj_ip(LabeledTensor<TensorType> ltensor) {
    std::function<TensorType(TensorType)> func = [&](TensorType a) {
        return std::conj(a);
    };
    apply_ewise_ip(ltensor, func);
}

template<typename TensorType>
void conj_ip(Tensor<TensorType> tensor) { 
    conj_ip(tensor()); 
}

template<typename TensorType>
void scale_ip(LabeledTensor<TensorType> ltensor,
           TensorType alpha) {
    std::function<TensorType(TensorType)> func = [&](TensorType a) {
        return alpha * a;
    };
    apply_ewise_ip(ltensor, func);
}

template<typename TensorType>
void scale_ip(Tensor<TensorType> tensor, TensorType alpha) {
   scale_ip(tensor(),alpha);
}

/**
 * @brief applies a function elementwise to a tensor, returns a new tensor
 *
 * @tparam TensorType the type of the elements in the tensor
 * @param oltensor original tensor
 * @param func function to be applied to each element
 * @return resulting tensor after applying func to original tensor
 */
template<typename TensorType>
Tensor<TensorType> apply_ewise(LabeledTensor<TensorType> oltensor,
                   std::function<TensorType(TensorType)> func,
                   bool is_lt=true) {
    ExecutionContext& ec = get_ec(oltensor);
    Tensor<TensorType> otensor = oltensor.tensor();
    Tensor<TensorType> tensor{oltensor.labels()};
    LabeledTensor<TensorType> ltensor = tensor();
    Tensor<TensorType>::allocate(&ec,tensor);
    //if(is_lt) Scheduler{ec}(ltensor = oltensor).execute();    

    auto lambda = [&](const IndexVector& bid) {
        const IndexVector blockid   = internal::translate_blockid(bid, oltensor);
        const tamm::TAMM_SIZE dsize = tensor.block_size(bid);
        std::vector<TensorType> dbuf(dsize);
        otensor.get(blockid, dbuf);
        for(size_t c = 0; c < dsize; c++) dbuf[c] = func(dbuf[c]);
        tensor.put(bid, dbuf);
    };
    block_for(ec, ltensor, lambda);
    return tensor;
}

// Several convenience functions using apply_ewise
// These routines return a new tensor
template<typename TensorType>
Tensor<TensorType> conj(LabeledTensor<TensorType> ltensor, bool is_lt = true) {
    std::function<TensorType(TensorType)> func = [&](TensorType a) {
        return std::conj(a);
    };
    return apply_ewise(ltensor, func, is_lt);
}

template<typename TensorType>
Tensor<TensorType> conj(Tensor<TensorType> tensor) { 
    return conj(tensor(), false); 
}

template<typename TensorType>
Tensor<TensorType> square(LabeledTensor<TensorType> ltensor, bool is_lt = true) {
    std::function<TensorType(TensorType)> func = [&](TensorType a) {
        return a * a;
    };
    return apply_ewise(ltensor, func, is_lt);
}

template<typename TensorType>
Tensor<TensorType> square(Tensor<TensorType> tensor) {
    return square(tensor(), false);
}

template<typename TensorType>
Tensor<TensorType> log10(LabeledTensor<TensorType> ltensor, bool is_lt = true) {
    std::function<TensorType(TensorType)> func = [&](TensorType a) {
        return std::log10(a);
    };
    return apply_ewise(ltensor, func, is_lt);
}

template<typename TensorType>
Tensor<TensorType> log10(Tensor<TensorType> tensor) {
   return log10(tensor(), false);
}

template<typename TensorType>
Tensor<TensorType> log(LabeledTensor<TensorType> ltensor, bool is_lt = true) {
    std::function<TensorType(TensorType)> func = [&](TensorType a) {
        return std::log(a);
    };
    return apply_ewise(ltensor, func, is_lt);
}

template<typename TensorType>
Tensor<TensorType> log(Tensor<TensorType> tensor) {
   return log(tensor(), false);
}

template<typename TensorType>
Tensor<TensorType> einverse(LabeledTensor<TensorType> ltensor, bool is_lt = true) {
    std::function<TensorType(TensorType)> func = [&](TensorType a) {
        return 1 / a;
    };
    return apply_ewise(ltensor, func, is_lt);
}

template<typename TensorType>
Tensor<TensorType> einverse(Tensor<TensorType> tensor) {
   return einverse(tensor(), false);
}

template<typename TensorType>
Tensor<TensorType> pow(LabeledTensor<TensorType> ltensor,
         TensorType alpha, bool is_lt = true) {
    std::function<TensorType(TensorType)> func = [&](TensorType a) {
        return std::pow(a, alpha);
    };
    return apply_ewise(ltensor, func, is_lt);
}

template<typename TensorType>
Tensor<TensorType> pow(Tensor<TensorType> tensor, TensorType alpha) {
   return pow(tensor(),alpha, false);
}

template<typename TensorType>
Tensor<TensorType> scale(LabeledTensor<TensorType> ltensor,
           TensorType alpha, bool is_lt = true) {
    std::function<TensorType(TensorType)> func = [&](TensorType a) {
        return alpha * a;
    };
    return apply_ewise(ltensor, func, is_lt);
}

template<typename TensorType>
Tensor<TensorType> scale(Tensor<TensorType> tensor, TensorType alpha) {
   return scale(tensor(),alpha, false);
}

template<typename TensorType>
Tensor<TensorType> sqrt(LabeledTensor<TensorType> ltensor, bool is_lt = true) {
    std::function<TensorType(TensorType)> func = [&](TensorType a) {
        return std::sqrt(a);
    };
    return apply_ewise(ltensor, func, is_lt);
}

template<typename TensorType>
Tensor<TensorType> sqrt(Tensor<TensorType> tensor) {
   return sqrt(tensor(), false);
}

template<typename TensorType>
void random_ip(LabeledTensor<TensorType> ltensor, bool is_lt = true) {
    //std::random_device random_device;
    std::default_random_engine generator;
    std::uniform_real_distribution<TensorType> tensor_rand_dist(0.0,1.0);

    std::function<TensorType(TensorType)> func = [&](TensorType a) {
        return tensor_rand_dist(generator);
    };
    apply_ewise_ip(ltensor, func);
}

template<typename TensorType>
void random_ip(Tensor<TensorType> tensor) {
  random_ip(tensor(), false);
}

template<typename TensorType>
TensorType sum(LabeledTensor<TensorType> ltensor) {
    throw std::runtime_error("sum unsupported");
#if 0
    ExecutionContext& gec = get_ec(ltensor);
    TensorType lsumsq         = 0;
    TensorType gsumsq         = 0;
    Tensor<TensorType> tensor = ltensor.tensor();

    #ifdef TU_SG
    MPI_Comm sub_comm;
    int rank = gec.pg().rank().value();
    get_subgroup_info(gec,tensor,sub_comm);

    if(sub_comm != MPI_COMM_NULL) {
        ProcGroup pg = ProcGroup::create_coll(sub_comm);
        ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
    #else
        ExecutionContext& ec = gec;
    #endif
        auto getnorm = [&](const IndexVector& bid) {
            const IndexVector blockid   = internal::translate_blockid(bid, ltensor);
            const tamm::TAMM_SIZE dsize = tensor.block_size(blockid);
            std::vector<TensorType> dbuf(dsize);
            tensor.get(blockid, dbuf);
            if constexpr(std::is_same_v<TensorType, std::complex<double>>
                    || std::is_same_v<TensorType, std::complex<float>>)
            for(auto val : dbuf) lsumsq += val;
        };
        block_for(ec, ltensor, getnorm);

    #ifdef TU_SG
        ec.flush_and_sync();
        //MemoryManagerGA::destroy_coll(mgr);
        MPI_Comm_free(&sub_comm);
        pg.destroy_coll();
    }
    #endif

    gec.pg().barrier();   
 
    gsumsq = gec.pg().allreduce(&lsumsq, upcxx::op_fast_add);
    return gsumsq;
#endif
}

template<typename TensorType>
TensorType sum(Tensor<TensorType> tensor) {
    return sum(tensor());
}

template<typename TensorType>
TensorType norm_unused(LabeledTensor<TensorType> ltensor) {
    ExecutionContext& ec = get_ec(ltensor);
    Scheduler sch{ec};
    Tensor<TensorType> nval{};
    sch.allocate(nval);

    if constexpr(internal::is_complex_v<TensorType>){
        auto ltconj = tamm::conj(ltensor);
        sch(nval() = ltconj() * ltensor).deallocate(ltconj).execute();
    }
    else 
    sch(nval() = ltensor * ltensor).execute();
    
    auto rval = get_scalar(nval);
    sch.deallocate(nval).execute();

    return std::sqrt(rval);
}

template<typename TensorType>
TensorType norm(Tensor<TensorType> tensor) {
    ExecutionContext& gec = get_ec(tensor());
    return norm(gec, tensor());
}

template<typename TensorType>
TensorType norm(LabeledTensor<TensorType> ltensor) {
    ExecutionContext& gec = get_ec(ltensor);
    return norm(gec, ltensor);
}

template<typename TensorType>
TensorType norm(ExecutionContext& gec, Tensor<TensorType> tensor) {
    return norm(gec, tensor());
}

template<typename TensorType>
TensorType norm(ExecutionContext& gec, LabeledTensor<TensorType> ltensor) {
    // ExecutionContext& gec = get_ec(ltensor);

    TensorType lsumsq         = 0;
    TensorType gsumsq         = 0;
    Tensor<TensorType> tensor = ltensor.tensor();

    #ifdef TU_SG
    int rank = gec.pg().rank().value();
    auto [nagg,ppn,subranks] = get_subgroup_info(gec,tensor);
    upcxx::team* sub_comm = new upcxx::team(gec.pg().team()->split(
                rank < subranks ? 0 : upcxx::team::color_none, 0));

    if (rank < subranks) {
        ProcGroup pg = ProcGroup::create_coll(*sub_comm);
        ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
    #else 
        ExecutionContext& ec = gec;
    #endif

        auto getnorm = [&](const IndexVector& bid) {
            const IndexVector blockid   = internal::translate_blockid(bid, ltensor);
            const tamm::TAMM_SIZE dsize = tensor.block_size(blockid);
            std::vector<TensorType> dbuf(dsize);
            tensor.get(blockid, dbuf);
            if constexpr(std::is_same_v<TensorType, std::complex<double>>
                    || std::is_same_v<TensorType, std::complex<float>>)
                for(TensorType val : dbuf) lsumsq += val * std::conj(val);         
            else
                for(TensorType val : dbuf) lsumsq += val * val;
        };
        block_for(ec, ltensor, getnorm);

    #ifdef TU_SG
        ec.flush_and_sync();
        pg.destroy_coll();
        {
            //upcxx::persona_scope master_scope(master_mtx,
            //        upcxx::master_persona());
            sub_comm->destroy();
        }
    }
    #endif
    gec.pg().barrier();

    gsumsq = gec.pg().allreduce(&lsumsq, upcxx::op_fast_add);
    return std::sqrt(gsumsq);
}

template<typename TensorType>
void gf_peak_coord(int nmodes, std::vector<TensorType> dbuf,
std::vector<size_t> block_dims, std::vector<size_t> block_offset, 
double threshold, double x_norm_sq, int rank, std::ostringstream& spfe){
     size_t c          = 0;

        if(nmodes == 1) {
            for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0];
                i++, c++) {
                auto val = dbuf[c];
                double lv = 0;
                if constexpr(std::is_same_v<TensorType, std::complex<double>>
                    || std::is_same_v<TensorType, std::complex<float>>)
                    lv = std::real(val * std::conj(val));  
                else lv = val * val;  

                size_t x1    = i;
                double weight = lv/x_norm_sq;
		        if(weight>threshold)
                    spfe << "   coord. = (" << x1 << "), wt. = " << weight << std::endl;
                }
        } else if(nmodes == 2) {
            for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0];
                i++) {
                for(size_t j = block_offset[1];
                    j < block_offset[1] + block_dims[1]; j++, c++) {

                auto val = dbuf[c];
                double lv = 0;
                if constexpr(std::is_same_v<TensorType, std::complex<double>>
                    || std::is_same_v<TensorType, std::complex<float>>)
                    lv = std::real(val * std::conj(val));  
                else lv = val * val;  

                size_t x1    = i;
                size_t x2    = j;
                double weight = lv/x_norm_sq;
		        if(weight>threshold){
                    spfe << "   coord. = (" << x1 << "," << x2 << "), wt. = " << weight << std::endl;
                    }
		        }
            }
        }
         else if(nmodes == 3) {
            for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0];
                i++) {
                for(size_t j = block_offset[1];
                    j < block_offset[1] + block_dims[1]; j++) {
                    for(size_t k = block_offset[2];
                        k < block_offset[2] + block_dims[2]; k++, c++) {

                auto val = dbuf[c];
                double lv = 0;
                if constexpr(std::is_same_v<TensorType, std::complex<double>>
                    || std::is_same_v<TensorType, std::complex<float>>)
                    lv = std::real(val * std::conj(val));  
                else lv = val * val;  

                //if(lv>threshold) {
                size_t x1    = i; 
                size_t x2    = j; 
                size_t x3    = k; 
                double weight = lv/x_norm_sq;
		        if(weight>threshold)
                    spfe << "   coord. = (" << x1 << "," << x2 << "," 
                    << x3 << "), wt. = " << weight << std::endl;
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

                auto val = dbuf[c];
                double lv = 0;
                if constexpr(std::is_same_v<TensorType, std::complex<double>>
                    || std::is_same_v<TensorType, std::complex<float>>)
                    lv = std::real(val * std::conj(val));  
                else lv = val * val;  

                size_t x1    = i;
                size_t x2    = j;
                size_t x3    = k;
                size_t x4    = l;
                double weight = lv/x_norm_sq;
		        if(weight>threshold)
                    spfe << "   coord. = (" << x1 << "," << x2 << "," 
                    << x3 << "," << x4 << "), wt. = " << weight << std::endl;

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

                auto val = dbuf[c];
                double lv = 0;
                if constexpr(std::is_same_v<TensorType, std::complex<double>>
                    || std::is_same_v<TensorType, std::complex<float>>)
                    lv = std::real(val * std::conj(val));  
                else lv = val * val;  

                // if(lv>threshold) {
                size_t x1    = i;
                size_t x2    = j;
                size_t x3    = k;
                size_t x4    = l;
                size_t x5    = m;
                double weight = lv/x_norm_sq;
                if(weight>threshold) 
                    spfe << "   coord. = (" << x1 << "," << x2 << "," 
                    << x3 << "," << x4 << "," << x5 << "), wt. = " << weight << std::endl;
                        }
                    }
                }
            }
        }
    }

}

template<typename TensorType>
void gf_peak(Tensor<TensorType> tensor, double threshold, double x_norm_sq, std::ostringstream& spfe) {
    return gf_peak(tensor(),threshold,x_norm_sq,spfe);
}

template<typename TensorType>
void gf_peak(LabeledTensor<TensorType> ltensor, double threshold, double x_norm_sq, std::ostringstream& spfe) {
    throw std::runtime_error("gf_peak unsupported");
#if 0
    ExecutionContext& gec = get_ec(ltensor);

    Tensor<TensorType> tensor = ltensor.tensor();
    int rank = gec.pg().rank().value();

    #ifdef TU_SG
    MPI_Comm sub_comm;
    get_subgroup_info(gec,tensor,sub_comm);

    if(sub_comm != MPI_COMM_NULL) {
        ProcGroup pg = ProcGroup::create_coll(sub_comm);
        ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
    #else 
        ExecutionContext& ec = gec;
    #endif
        auto nmodes               = tensor.num_modes();
        auto getnorm = [&](const IndexVector& bid) {
            const IndexVector blockid   = internal::translate_blockid(bid, ltensor);
            const tamm::TAMM_SIZE dsize = tensor.block_size(blockid);
            std::vector<TensorType> dbuf(dsize);
            tensor.get(blockid, dbuf);
            auto block_dims   = tensor.block_dims(blockid);
            auto block_offset = tensor.block_offsets(blockid);
           
            gf_peak_coord(nmodes,dbuf,block_dims,block_offset,threshold,x_norm_sq,rank,spfe);
        };
        block_for(ec, ltensor, getnorm);
    
    #ifdef TU_SG
        ec.flush_and_sync();
        //MemoryManagerGA::destroy_coll(mgr);
        MPI_Comm_free(&sub_comm);
        pg.destroy_coll();
    }
    #endif

    gec.pg().barrier();
#endif

}

// returns max_element, blockids, coordinates of max element in the block
template<typename TensorType>
std::tuple<TensorType, IndexVector, std::vector<size_t>>
        max_element(Tensor<TensorType> tensor) {
    return max_element(tensor());
}

template<typename TensorType>
std::tuple<TensorType, IndexVector, std::vector<size_t>> 
        max_element(LabeledTensor<TensorType> ltensor) {
    throw std::runtime_error("max_element unsupported");
#if 0
    ExecutionContext& ec = get_ec(ltensor);
    TensorType max = 0.0;

    Tensor<TensorType> tensor = ltensor.tensor();
    auto nmodes               = tensor.num_modes();
    // Works for only upto 6D tensors
    EXPECTS(tensor.num_modes() <= 6);

    IndexVector maxblockid(nmodes);
    std::vector<size_t> bfuv(nmodes);
    std::vector<TensorType> lmax(2, 0);
    std::vector<TensorType> gmax(2, 0);

    auto getmax = [&](const IndexVector& bid) {
        const IndexVector blockid   = internal::translate_blockid(bid, ltensor);
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
            for(size_t i = block_offset[0]; i < dimi; i++) {
                for(size_t j = block_offset[1]; j < dimj; j++, c++) {
                    if(lmax[0] < dbuf[c]) {
                        lmax[0]    = dbuf[c];
                        lmax[1]    = GA_Nodeid();
                        bfuv[0]    = i - block_offset[0];
                        bfuv[1]    = j - block_offset[1];
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

    ec.pg().allreduce(lmax.data(), gmax.data(), 2, upcxx::op_fast_maxloc);
    ec.pg().broadcast(maxblockid.data(), nmodes, gmax[1]);
    ec.pg().broadcast(bfuv.data(), nmodes, gmax[1]);

    return std::make_tuple(gmax[0], maxblockid, bfuv);
#endif
}

// returns min_element, blockids, coordinates of min element in the block
template<typename TensorType>
std::tuple<TensorType, IndexVector, std::vector<size_t>> 
        min_element(Tensor<TensorType> tensor) {
    return min_element(tensor());
}

template<typename TensorType>
std::tuple<TensorType, IndexVector, std::vector<size_t>>
        min_element(LabeledTensor<TensorType> ltensor) {
    throw std::runtime_error("min_element unsupported");
#if 0
    ExecutionContext& ec = get_ec(ltensor);
    TensorType min = 0.0;

    Tensor<TensorType> tensor = ltensor.tensor();
    auto nmodes               = tensor.num_modes();
    // Works for only upto 6D tensors
    EXPECTS(tensor.num_modes() <= 6);

    IndexVector minblockid(nmodes);
    std::vector<size_t> bfuv(nmodes);
    std::vector<TensorType> lmin(2, 0);
    std::vector<TensorType> gmin(2, 0);

    auto getmin = [&](const IndexVector& bid) {
        const IndexVector blockid   = internal::translate_blockid(bid, ltensor);
        const tamm::TAMM_SIZE dsize = tensor.block_size(blockid);
        std::vector<TensorType> dbuf(dsize);
        tensor.get(blockid, dbuf);
        auto block_dims   = tensor.block_dims(blockid);
        auto block_offset = tensor.block_offsets(blockid);
        size_t c          = 0;

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

    ec.pg().allreduce(lmin.data(), gmin.data(), 2, upcxx::op_fast_minloc);
    ec.pg().broadcast(minblockid.data(), nmodes, gmin[1]);
    ec.pg().broadcast(bfuv.data(), nmodes, gmin[1]);

    return std::make_tuple(gmin[0], minblockid, bfuv);
#endif
}

// regular 2D tamm tensor to block-cyclic tamm tensor
template<typename TensorType>
void to_block_cyclic_tensor(Tensor<TensorType> tensor, Tensor<TensorType> bc_tensor)
{
    int ndims = tensor.num_modes();
    EXPECTS(ndims == 2);
    EXPECTS(bc_tensor.is_block_cyclic());

    //bc_tensor might be on a smaller process group
    ExecutionContext& ec = get_ec(bc_tensor());


    auto tamm_bc_lambda = [&](const IndexVector& bid){
        const IndexVector blockid =
        internal::translate_blockid(bid, tensor());

        auto block_dims   = tensor.block_dims(blockid);
        auto block_offset = tensor.block_offsets(blockid);

        const tamm::TAMM_SIZE dsize = tensor.block_size(blockid);

        int64_t lo[2] = {cd_ncast<size_t>(block_offset[0]), 
            cd_ncast<size_t>(block_offset[1])};
        int64_t hi[2] = {cd_ncast<size_t>(block_offset[0] + block_dims[0]-1), 
            cd_ncast<size_t>(block_offset[1] + block_dims[1]-1)};
        int64_t ld = cd_ncast<size_t>(block_dims[1]);
        
        std::vector<TensorType> sbuf(dsize);
        tensor.get(blockid, sbuf);
        bc_tensor.put_raw(lo, hi, &sbuf[0], &ld);
    };

    block_for(ec, tensor(), tamm_bc_lambda);    

}

template<typename TensorType>
void from_block_cyclic_tensor(Tensor<TensorType> bc_tensor, Tensor<TensorType> tensor)
{
  const auto ndims = bc_tensor.num_modes();
  EXPECTS(ndims == 2);
  EXPECTS(bc_tensor.is_block_cyclic());
  EXPECTS(bc_tensor.kind() == TensorBase::TensorKind::dense);
  EXPECTS(bc_tensor.distribution().kind() == DistributionKind::dense);

  // bc_tensor might be on a smaller process group
  ExecutionContext& ec = get_ec(bc_tensor());

  auto tamm_bc_lambda = [&](const IndexVector& bid) {
    const IndexVector blockid = internal::translate_blockid(bid, tensor());

    auto block_dims   = tensor.block_dims(blockid);
    auto block_offset = tensor.block_offsets(blockid);

    const tamm::TAMM_SIZE dsize = tensor.block_size(blockid);

    int64_t lo[2] = {cd_ncast<size_t>(block_offset[0]), 
        cd_ncast<size_t>(block_offset[1])};
    int64_t hi[2] = {cd_ncast<size_t>(block_offset[0] + block_dims[0]-1), 
        cd_ncast<size_t>(block_offset[1] + block_dims[1]-1)};
    int64_t ld = cd_ncast<size_t>(block_dims[1]);

    std::vector<TensorType> sbuf(dsize);
    bc_tensor.get_raw(lo, hi, &sbuf[0], &ld);
    tensor.put(blockid, sbuf);
  };

  block_for(ec, tensor(), tamm_bc_lambda);
}

template<typename TensorType>
Tensor<TensorType> from_block_cyclic_tensor(Tensor<TensorType> bc_tensor) {
  //TiledIndexSpaceVec tis_vec={}, bool is_dense=true) 
  
  const int ndims = bc_tensor.num_modes();
  EXPECTS(ndims == 2);
  EXPECTS(bc_tensor.is_block_cyclic());
  EXPECTS(bc_tensor.kind() == TensorBase::TensorKind::dense);
  EXPECTS(bc_tensor.distribution().kind() == DistributionKind::dense);
  
//   if(!tis_vec.empty()) EXPECTS(tis_vec.size() == ndims);
//   else tis_vec = bc_tensor.tiled_index_spaces();

  ExecutionContext& ec  = get_ec(bc_tensor());
  // this routine works only when the tiled index spaces are exactly the same 
  Tensor<TensorType> tensor{bc_tensor.tiled_index_spaces()};
  /*if(is_dense)*/ tensor.set_dense();
  Tensor<TensorType>::allocate(&ec,tensor);
  from_block_cyclic_tensor(bc_tensor, tensor);

  return tensor;
}

template<typename TensorType>
std::tuple<TensorType*,int64_t> access_local_block_cyclic_buffer(Tensor<TensorType> tensor) 
{
   EXPECTS(tensor.num_modes() == 2);
   int gah = tensor.ga_handle();
   ExecutionContext& ec = get_ec(tensor());
   TensorType* lbufptr;
   int64_t lbufsize;
   NGA_Access_block_segment64(gah, ec.pg().rank().value(), reinterpret_cast<void*>(&lbufptr), &lbufsize);
   return std::make_tuple(lbufptr,lbufsize);
}


// permute a given tensor (TODO: does not work correctly for >=3D dense tensors)
template<typename TensorType>
Tensor<TensorType> permute_tensor(Tensor<TensorType> tensor, std::vector<int> permute) {
    auto tis = tensor.tiled_index_spaces();
    int ndims = tis.size();
    EXPECTS(!permute.empty() && ndims == permute.size());
    EXPECTS(std::all_of(permute.begin(), permute.end(), [&](int i){ return i>=0 && i<ndims; }));

    std::vector<TiledIndexLabel> til(ndims), ptil(ndims);
    for (int i = 0; i < ndims; i++) til[i]  = tis[i].label("all");
    for (int i = 0; i < ndims; i++) ptil[i] = til[permute[i]];

    Tensor<TensorType> ptensor{ptil};
    if(tensor.kind() == TensorBase::TensorKind::dense) ptensor.set_dense();
    ExecutionContext& ec = get_ec(tensor());
    Scheduler{ec}.allocate(ptensor)
    (ptensor(ptil) = tensor(til)).execute();

    return ptensor; // caller responsible for dellocating this tensor
}

// Extract block of a dense tensor given by [lo, hi)
template<typename TensorType>
Tensor<TensorType> tensor_block(Tensor<TensorType> tensor, std::vector<int64_t> lo, std::vector<int64_t> hi, std::vector<int> permute={}) {
    // Max Grossman (Jan 9 2022): unsupported for now, need to implement ga_copy_patch
    abort();
   
#if 0
    const int ndims = tensor.num_modes();
    EXPECTS(tensor.kind() == TensorBase::TensorKind::dense);
    EXPECTS(tensor.distribution().kind() == DistributionKind::dense);

    auto tis = tensor.tiled_index_spaces();

    for(int i = 0; i < ndims; i++) {
      EXPECTS(hi[i] <= tis[i].index_space().num_indices() && lo[i] >= 0);
      // tamm_terminate("invalid ranges specified to tensor_block");
    }

    LabeledTensor<TensorType> ltensor = tensor();
    ExecutionContext& ec = get_ec(ltensor);
    std::vector<bool> is_irreg_tis(ndims,false);
    for(int i = 0; i < ndims; i++) is_irreg_tis[i] = !tis[i].input_tile_sizes().empty();

    std::vector<std::vector<Tile>> tiles(ndims);
    for(int i = 0; i < ndims; i++) {
      tiles[i] = is_irreg_tis[i] ? tis[i].input_tile_sizes()
                                 : std::vector<Tile>{tis[i].input_tile_size()};
    }

    std::vector<Tile> max_ts(ndims);
    for(int i = 0; i < ndims; i++)
      max_ts[i] = is_irreg_tis[i] ? *max_element(tiles[i].begin(), tiles[i].end()) : tiles[i][0];

    TiledIndexSpaceVec btis(ndims);
    for(int i = 0; i < ndims; i++) btis[i] = TiledIndexSpace{range(hi[i] - lo[i]), max_ts[i]};

    Tensor<TensorType> btensor{btis};
    btensor.set_dense();
    Tensor<TensorType>::allocate(&ec,btensor);
    int btensor_gah = btensor.ga_handle();
    int tensor_gah  = tensor.ga_handle();

    char ctrans = 'N';

    int64_t lo_src[ndims], hi_src[ndims], lo_dst[ndims], hi_dst[ndims];
    for(int i = 0; i < ndims; i++) {
      lo_src[i] = lo[i];
      hi_src[i] = hi[i] - 1;
    }
    for(int i = 0; i < ndims; i++) {
      lo_dst[i] = 0;
      hi_dst[i] = hi_src[i] - lo_src[i];
    }

    int ga_pg_default = GA_Pgroup_get_default();
    GA_Pgroup_set_default(ec.pg().ga_pg());
    
    NGA_Copy_patch64(ctrans, tensor_gah, lo_src, hi_src, btensor_gah, lo_dst, hi_dst);

    GA_Pgroup_set_default(ga_pg_default);

    Tensor<TensorType> pbtensor = btensor;
    if(!permute.empty()) {
      pbtensor = permute_tensor<TensorType>(btensor, permute);
      Tensor<TensorType>::deallocate(btensor);
    }

    return pbtensor; // caller responsible for dellocating this tensor
#endif
}


inline TiledIndexLabel compose_lbl(const TiledIndexLabel& lhs,
                                   const TiledIndexLabel& rhs) {
    auto lhs_tis = lhs.tiled_index_space();
    auto rhs_tis = rhs.tiled_index_space();

    auto res_tis = lhs_tis.compose_tis(rhs_tis);

    return res_tis.label("all");
}

inline TiledIndexSpace compose_tis(const TiledIndexSpace& lhs,
                                   const TiledIndexSpace& rhs) {
    return lhs.compose_tis(rhs);
}

inline TiledIndexLabel invert_lbl(const TiledIndexLabel& lhs) {
    auto lhs_tis = lhs.tiled_index_space().invert_tis();

    return lhs_tis.label("all");
}

inline TiledIndexSpace invert_tis(const TiledIndexSpace& lhs) {
    return lhs.invert_tis();
}

inline TiledIndexLabel intersect_lbl(const TiledIndexLabel& lhs,
                                     const TiledIndexLabel& rhs) {
    auto lhs_tis = lhs.tiled_index_space();
    auto rhs_tis = rhs.tiled_index_space();

    auto res_tis = lhs_tis.intersect_tis(rhs_tis);

    return res_tis.label("all");
}

inline TiledIndexSpace intersect_tis(const TiledIndexSpace& lhs,
                                     const TiledIndexSpace& rhs) {
    return lhs.intersect_tis(rhs);
}

inline TiledIndexLabel union_lbl(const TiledIndexLabel& lhs,
                                 const TiledIndexLabel& rhs) {
    auto lhs_tis = lhs.tiled_index_space();
    auto rhs_tis = rhs.tiled_index_space();

    auto res_tis = lhs_tis.union_tis(rhs_tis);

    return res_tis.label("all");
}

inline TiledIndexSpace union_tis(const TiledIndexSpace& lhs,
                                 const TiledIndexSpace& rhs) {
    return lhs.union_tis(rhs);
}

inline TiledIndexLabel project_lbl(const TiledIndexLabel& lhs,
                                   const TiledIndexLabel& rhs) {
    auto lhs_tis = lhs.tiled_index_space();
    auto rhs_tis = rhs.tiled_index_space();

    auto res_tis = lhs_tis.project_tis(rhs_tis);

    return res_tis.label("all");
}

inline TiledIndexSpace project_tis(const TiledIndexSpace& lhs,
                                   const TiledIndexSpace& rhs) {
    return lhs.project_tis(rhs);
}

/// @todo: Implement
template<typename TensorType>
inline TensorType invert_tensor(TensorType tens) {
    TensorType res;

    return res;
}

/**
 * @brief uses a function to fill in elements of a tensor
 *
 * @tparam TensorType the type of the elements in the tensor
 * @param ec Execution context used in the blockfor
 * @param ltensor tensor to operate on
 * @param func function to fill in the tensor with
 */
template<typename TensorType>
inline size_t hash_tensor(Tensor<TensorType> tensor) {
    auto ltensor = tensor();
    auto ec = tensor.execution_context();
    size_t hash = tensor.num_modes();
    auto lambda = [&](const IndexVector& bid) {
        const IndexVector blockid   = internal::translate_blockid(bid, ltensor);
        const tamm::TAMM_SIZE dsize = tensor.block_size(blockid);

        internal::hash_combine(hash, tensor.block_size(blockid));
        std::vector<TensorType> dbuf(dsize);
        tensor.get(blockid, dbuf);
        for(auto& val : dbuf) {
            internal::hash_combine(hash, val);        
        }
    };
    block_for(*ec, ltensor, lambda);

    return hash;
}

// Convert regular tamm tensor to dense tamm tensor
template<typename TensorType>
Tensor<TensorType> to_dense_tensor(ExecutionContext& ec_dense, Tensor<TensorType> tensor) {
    
    const int ndims = tensor.num_modes();
    EXPECTS(tensor.distribution().kind() == DistributionKind::nw);
    EXPECTS(tensor.kind() == TensorBase::TensorKind::normal || 
            tensor.kind() == TensorBase::TensorKind::spin);    

    auto tis = tensor.tiled_index_spaces();

    LabeledTensor<TensorType> ltensor = tensor();
    ExecutionContext& ec = get_ec(ltensor);

    Tensor<TensorType> btensor{tis};
    btensor.set_dense();
    Tensor<TensorType>::allocate(&ec_dense,btensor);
    int ga_stensor  = tamm_to_ga(ec_dense,tensor);
    ga_to_tamm(ec_dense,btensor,ga_stensor);
    NGA_Destroy(ga_stensor);

    return btensor; // caller responsible for dellocating this tensor
}

// Extract a single value specified by index_id from a dense tensor.
template<typename TensorType>
TensorType get_tensor_element(Tensor<TensorType> tensor, std::vector<int64_t> index_id) {

    const int ndims = tensor.num_modes();
    EXPECTS(tensor.kind() == TensorBase::TensorKind::dense);
    // EXPECTS(tensor.distribution().kind() == DistributionKind::dense);

    std::vector<int64_t> ld(ndims - 1, 1);

    int64_t lo[ndims];
    int64_t hi[ndims];

    for(int i = 0; i < ndims; i++) {
        lo[i] = index_id[i];
        hi[i] = index_id[i];
    }

    TensorType val{};
    NGA_Get64(tensor.ga_handle(), &lo[0], &hi[0], &val, &ld[0]);
    return val;
};


/**
 * @brief Prints a dense Tensor object
 *
 * @tparam T template type for Tensor element type
 * @param [in] tensor input Tensor object
 */
template<typename T>
void print_dense_tensor(const Tensor<T>& tensor, std::function<bool(std::vector<size_t>)> func, std::string filename="", bool append=false) {

    auto lt = tensor();
    int ndims = tensor.num_modes();
    ExecutionContext& ec = get_ec(lt);

    auto nz_check =
      [=](const T val) {
        if constexpr(tamm::internal::is_complex_v<T>) {
          if(val.real() > 0.0000000000001 || val.real() < -0.0000000000001) return true;
        }
        else if(val > 0.0000000000001 || val < -0.0000000000001)
          return true;
        return false;
    };

    if(ec.pg().rank() == 0) {
      std::stringstream tstring;
      EXPECTS(ndims > 0 && ndims <= 4);
      std::vector<int64_t> dims;
      for(auto tis: tensor.tiled_index_spaces()) dims.push_back(tis.index_space().num_indices());

      // tstring << "tensor dims = " << dims << std::endl;
      // tstring << "actual tensor size = " << tensor.size() << std::endl;
      for(auto it: tensor.loop_nest()) {
        auto blockid = internal::translate_blockid(it, lt);
        if(!tensor.is_non_zero(blockid)) continue;
        TAMM_SIZE      size = tensor.block_size(blockid);
        std::vector<T> buf(size);
        tensor.get(blockid, buf);
        auto block_dims   = tensor.block_dims(blockid);
        auto block_offset = tensor.block_offsets(blockid);

        tstring << std::fixed << std::setprecision(10);

        size_t c = 0;
        if(ndims == 1) {
          for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++, c++) {
            if(func({i}) && nz_check(buf[c])) tstring << i + 1 << "   " << buf[c] << std::endl;
          }
        }
        else if(ndims == 2) {
          for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
            for(size_t j = block_offset[1]; j < block_offset[1] + block_dims[1]; j++, c++) {
              if(func({i, j}) && nz_check(buf[c])) tstring << i + 1 << "   " << j + 1 << "   " << buf[c] << std::endl;
            }
          }
        }
        else if(ndims == 3) {
          for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
            for(size_t j = block_offset[1]; j < block_offset[1] + block_dims[1]; j++) {
              for(size_t k = block_offset[2]; k < block_offset[2] + block_dims[2]; k++, c++) {
                if(func({i, j, k}) && nz_check(buf[c]))
                  tstring << i + 1 << "   " << j + 1 << "   " << k + 1 << "   " << buf[c]
                          << std::endl;
              }
            }
          }
        }
        else if(ndims == 4) {
          for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
            for(size_t j = block_offset[1]; j < block_offset[1] + block_dims[1]; j++) {
              for(size_t k = block_offset[2]; k < block_offset[2] + block_dims[2]; k++) {
                for(size_t l = block_offset[3]; l < block_offset[3] + block_dims[3]; l++, c++) {
                  if(func({i, j, k, l}) && nz_check(buf[c]))
                    tstring << i + 1 << "   " << j + 1 << "   " << k + 1 << "   " << l + 1 << "   "
                            << buf[c] << std::endl;
                }
              }
            }
          }
        }
      }

      if(!filename.empty()) {
        std::ofstream tos;
        if (append) tos.open(filename + ".txt", std::ios::app);
        else tos.open(filename + ".txt", std::ios::out);
        if(!tos) std::cerr << "Error opening file " << filename << std::endl;
        tos << tstring.str() << std::endl;
        tos.close();
      }
      else std::cout << tstring.str();
    }
    ec.pg().barrier();
}

template<typename T>
void print_dense_tensor(const Tensor<T>& tensor, std::string filename="") {
    std::function<bool(std::vector<size_t>)> func = [&](std::vector<size_t> cond) {
        return true;
    };
    print_dense_tensor(tensor,func,filename);
}


} // namespace tamm
