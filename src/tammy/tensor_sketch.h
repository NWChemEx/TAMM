#ifndef TENSOR_SKETCH_H_
#define TENSOR_SKETCH_H_

#include "index_space_sketch.h"
#include "labeled_tensor_sketch.h"
#include "execution_context.h"

namespace tammy {

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

    LabeledTensor<T> operator()() const {}

    template<class... Ts>
    LabeledTensor<T> operator()(Ts... inputs) const {
        return LabeledTensor<T>{*this, IndexLabelVec{inputs...}};
    }

    static void allocate(Tensor<T>& tensor) {}
    static void deallocate(Tensor<T>& tensor) {}

    template<typename... Args>
    static void allocate(const ExecutionContext& exec, Args... rest) {}

    template<typename... Args>
    static void deallocate(const ExecutionContext& exec, Args... rest) {}

    T* access(Index idx) {}

    private:
    std::vector<TiledIndexSpace> block_indices_;
};

} // namespace tammy

#endif // TENSOR_SKETCH_H_