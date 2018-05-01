#ifndef TAMM_TENSOR_IMPL_H_
#define TAMM_TENSOR_IMPL_H_

#include "tamm/execution_context.h"
#include "tamm/index_space.h"
#include "tamm/labeled_tensor.h"

namespace tamm {

class TensorBase {
public:
    // Ctors
    TensorBase() = default;

    TensorBase(const std::vector<TiledIndexSpace>& block_indices) :
      block_indices_{block_indices} {}

    TensorBase(const std::vector<TiledIndexLabel>& lbls) {
        for(const auto& lbl : lbls) {
            block_indices_.push_back(lbl.tiled_index_space());
        }
    }

    template<class... Ts>
    TensorBase(const TiledIndexSpace& tis, Ts... rest) : TensorBase{rest...} {
        block_indices_.insert(block_indices_.begin(), tis);
    }

    template<typename Func>
    TensorBase(const TiledIndexSpace& tis, const Func& func) {
        block_indices_.insert(block_indices_.begin(), tis);
    }

    // Dtor
    virtual ~TensorBase(){};

    virtual TensorRank rank() const = 0;
    virtual void allocate()         = 0;
    virtual void deallocate()       = 0;
    virtual void detach()           = 0;

protected:
    std::vector<TiledIndexSpace> block_indices_;
    Spin spin_total_;
    bool has_spatial_symmetry_;
    bool has_spin_symmetry_;

    // std::vector<IndexPosition> ipmask_;
    // PermGroup perm_groups_;
    // Irrep irrep_;
    // std::vector<SpinMask> spin_mask_;
}; // TensorBase

class TensorImpl : public TensorBase {
public:
    using TensorBase::TensorBase;
    // Ctors
    TensorImpl() = default;

    TensorImpl(const std::vector<TiledIndexSpace>& tis) : TensorBase{tis} {}

    TensorImpl(const std::vector<TiledIndexLabel>& lbls) : TensorBase{lbls} {}

    template<class... Ts>
    TensorImpl(const TiledIndexSpace& tis, Ts... rest) :
      TensorBase{tis, rest...} {}

    // Copy/Move Ctors and Assignment Operators
    TensorImpl(TensorImpl&&)      = default;
    TensorImpl(const TensorImpl&) = default;
    TensorImpl& operator=(TensorImpl&&) = default;
    TensorImpl& operator=(const TensorImpl&) = default;

    // Dtor
    ~TensorImpl() = default;

    // Overriden methods
    TensorRank rank() const override { return rank_; }
    void allocate() override {}
    void deallocate() override {}
    void detach() override {}

protected:
    TensorRank rank_;
}; // TensorImpl

template<typename T>
class Tensor {
public:
    Tensor() = default;

    Tensor(const std::initializer_list<TiledIndexSpace>& tis) :
      impl_{std::make_shared<TensorImpl>(tis)} {}

    Tensor(const std::initializer_list<TiledIndexLabel>& lbls) :
      impl_{std::make_shared<TensorImpl>(lbls)} {}

    template<class... Ts>
    Tensor(const TiledIndexSpace& tis, Ts... rest) :
      impl_{std::make_shared<TensorImpl>(tis, rest...)} {}

    LabeledTensor<T> operator()() const { return {}; }

    template<class... Ts>
    LabeledTensor<T> operator()(Ts... inputs) const {
        // return LabeledTensor<T>{*this, IndexLabelVec{inputs...}};
        return {};
    }

    void allocate() { impl_->allocate(); }
    void deallocate() { impl_->deallocate(); }
    void detach() { impl_->detach(); }

    void get(IndexVector idx_vec, T* buff, std::size_t buff_size) const {}
    void put(IndexVector idx_vec, T* buff, std::size_t buff_size) const {}

    // Static methods for allocate/deallocate
    template<typename... Args>
    static void allocate(const ExecutionContext& exec, Args... rest) {}

    template<typename... Args>
    static void deallocate(Args... rest) {}

private:
    std::shared_ptr<TensorImpl> impl_;
};

} // namespace tamm

#endif // TENSOR_IMPL_H_
