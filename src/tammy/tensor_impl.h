#ifndef TENSOR_IMPL_H_
#define TENSOR_IMPL_H_

#include "index_space.h"
#include "labeled_tensor.h"
#include "execution_context.h"

namespace tamm {

template<typename T>
class Tensor {
    public:
    Tensor() = default;
    Tensor(const std::initializer_list<TiledIndexSpace>& tis) :
      block_indices_{tis} {}

    Tensor(const std::initializer_list<TiledIndexLabel>& lbls) {
        for(const auto& lbl : lbls) {
            block_indices_.push_back(lbl.tiled_index_space());
        }
    }

    LabeledTensor<T> operator()() const {
      return {};
    }

    template<class... Ts>
    LabeledTensor<T> operator()(Ts... inputs) const {
      //return LabeledTensor<T>{*this, IndexLabelVec{inputs...}};
      return {};
    }

  static void allocate(ExecutionContext& ec, Tensor<T>& tensor) {}
  static void deallocate(Tensor<T>& tensor) {}


     void allocate() {}
     void deallocate() {}

    template<typename... Args>
    static void allocate(const ExecutionContext& exec, Args... rest) {

    }

    template<typename... Args>
    static void deallocate(Args... rest) {}

    T* access(Index idx) {}

    void get(IndexVector idx_vec, T* buff, std::size_t buff_size) const {}

    private:
    std::vector<TiledIndexSpace> block_indices_;
};

} // namespace tamm

#endif // TENSOR_IMPL_H_
