#ifndef TAMMY_TENSOR_H_
#define TAMMY_TENSOR_H_

#include <memory>

#include "tammy/boundvec.h"
#include "tammy/errors.h"
#include "tammy/memory_manager.h"
#include "tammy/proc_group.h"
#include "tammy/tensor_base.h"
#include "tammy/types.h"

namespace tammy {

template<typename T>
class LabeledTensor;

template<typename T>
class Tensor;

template<typename T>
class TensorImpl : public TensorBase, public TensorImplBase {
    public:
    using ElementType = T;
    using TensorBase::TensorBase;

    TensorImpl() = default;

    TensorImpl(const TensorVec<IndexRange>& dim_ranges,
               const TensorVec<IndexPosition>& ipmask,
               const PermGroup& perm_groups, Irrep irrep = Irrep{0},
               Spin spin_total = Spin{0}, TensorRank rank = 0) :
      TensorBase(dim_ranges, ipmask, perm_groups, irrep, spin_total),
      rank_{rank} {}

    TensorImpl(const TensorVec<const IndexSpace*>& index_spaces,
               const TensorVec<IndexPosition>& ipmask,
               const PermGroup& perm_groups, Irrep irrep = Irrep{0},
               Spin spin_total = Spin{0}, TensorRank rank = 0) :
      TensorBase(ranges(index_spaces), ipmask, perm_groups, irrep, spin_total),
      rank_{rank} {}

    TensorImpl(const TensorVec<IndexLabel>& index_labels,
               const TensorVec<IndexPosition>& ipmask,
               const PermGroup& perm_groups, Irrep irrep = Irrep{0},
               Spin spin_total = Spin{0}, TensorRank rank = 0) :
      TensorBase(ranges(index_labels), ipmask, perm_groups, irrep, spin_total),
      rank_{rank} {}

    TensorImpl(const TensorVec<IndexRange>& dim_ranges) :
      TensorBase(dim_ranges),
      rank_{static_cast<TensorRank>(dim_ranges.size())} {}

    TensorImpl(const TensorVec<const IndexSpace*>& index_spaces) :
      TensorBase{ranges(index_spaces)},
      rank_{static_cast<TensorRank>(index_spaces.size())} {}

    TensorImpl(const TensorVec<IndexLabel>& index_labels) :
      TensorBase{ranges(index_labels)},
      rank_{static_cast<TensorRank>(index_labels.size())} {}

    // @to-do: implement.
    TensorImpl(const DependentIndexLabel& dil1);

    TensorImpl(const std::tuple<TensorVec<IndexLabel>, TensorVec<IndexPosition>,
                                PermGroup>& tpl) :
      TensorImpl{ranges(std::get<0>(tpl)), std::get<1>(tpl), std::get<2>(tpl)} {
    }

    TensorImpl(const std::tuple<TensorVec<const IndexSpace*>,
                                TensorVec<IndexPosition>, PermGroup>& tpl) :
      TensorImpl{ranges(std::get<0>(tpl)), std::get<1>(tpl), std::get<2>(tpl)} {
    }

    TensorImpl(const std::tuple<TensorVec<IndexRange>, TensorVec<IndexPosition>,
                                PermGroup>& tpl) :
      TensorImpl{std::get<0>(tpl), std::get<1>(tpl), std::get<2>(tpl)} {}

    template<typename... Args>
    TensorImpl(const IndexSpace& is, Args... rest) :
      TensorImpl{pack_tuple_is(is, rest...)} {}

    template<typename... Args>
    TensorImpl(const IndexLabel& il, Args... rest) :
      TensorImpl{pack_tuple<IndexLabel>(il, rest...)} {}

    template<typename... Args>
    TensorImpl(const IndexRange& il, Args... rest) :
      TensorImpl{pack_tuple<IndexRange>(il, rest...)} {}

    TensorRank rank() const override { return rank_; }

    TensorVec<IndexRange> dim_ranges() const { return dim_ranges_; }

    bool is_unique(const BlockDimVec& bdv) const override { return false; }
    BlockDimVec get_unique(const BlockDimVec& bdv) const override {
        return BlockDimVec{};
    }
    bool is_nonzero(const BlockDimVec& bdv) const override { return false; }
    Size block_size(const BlockDimVec& bdv) const override { return Size{0}; }

    void set_proc_group(ProcGroup proc_group) override { pg_ = proc_group; }
    // Tensor& set_distribution(Distribution* distribution) override {}
    // Tensor& set_memory_manager(MemoryManager* memory_manager) override {}

    ProcGroup proc_group() const override { return pg_; }

    // Distribution* distribution() override {}
    // MemoryManager* memory_manager() override {}
    void allocate() override {}
    void deallocate() override {}
    // void attach(MemoryRegion* memory_region) override {}
    void detach() override {}

    // std::pair<Codelet*,Codelet*> get(const BlockDimVec& bdv, Block<T>& block)
    // const override {}  std::pair<Codelet*,Codelet*> put(const BlockDimVec& bdv,
    // Block<T>& block) override {}  std::pair<Codelet*,Codelet*> add(const
    // BlockDimVec& bdv, Block<T>& block) override {}

    template<typename... Args>
    static Tensor<T> create(Args... args);

    template<unsigned N, typename... Args>
    static auto create_list(Args... args);

    protected:
    TensorRank rank_;
    ProcGroup pg_;
    // std::unique_ptr<MemoryRegion> mpb_;
    // std::shared_ptr<Distribution> distribution_;
    template<typename ItemType, typename... Args>
    TensorVec<ItemType> pack_vec(ItemType item, Args... rest) {
        TensorVec<ItemType> ret = {item};
        pack<ItemType>(ret, rest...);

        return ret;
    }

    template<typename... Args>
    std::tuple<TensorVec<const IndexSpace*>, TensorVec<IndexPosition>,
               PermGroup>
    pack_tuple_is(const IndexSpace& item, Args... rest) {
        TensorVec<const IndexSpace*> ret = {&item};
        pack_is(ret, rest...);

        TensorVec<IndexPosition> ipmask{
          TensorVec<IndexPosition>{ret.size(), IndexPosition::neither}};
        PermGroup pg{ret.size()};

        unpack(ipmask, pg);

        return std::make_tuple(ret, ipmask, pg);
    }

    template<typename... Args>
    void pack_is(TensorVec<IndexSpace*>& isl, Args...) {}

    template<typename... Args>
    void pack_is(TensorVec<const IndexSpace*>& item_vec, const IndexSpace& item,
                 Args... rest) {
        item_vec.push_back(&item);
        pack(item_vec, rest...);
    }

    template<typename ItemType, typename... Args>
    std::tuple<TensorVec<ItemType>, TensorVec<IndexPosition>, PermGroup>
    pack_tuple(ItemType item, Args... rest) {
        TensorVec<ItemType> ret = {item};
        pack<ItemType>(ret, rest...);

        TensorVec<IndexPosition> ipmask{
          TensorVec<IndexPosition>{ret.size(), IndexPosition::neither}};
        PermGroup pg{ret.size()};

        unpack(ipmask, pg);

        return std::make_tuple(ret, ipmask, pg);
    }

    void unpack(const TensorVec<IndexPosition>& ipmask, const PermGroup& pg) {}

    template<typename... Args>
    void unpack(const TensorVec<IndexPosition>& ipmask, const PermGroup& pg,
                TensorVec<IndexPosition> mask, Args... rest) {
        ipmask = mask;
        unpack(ipmask, pg, rest...);
    }

    template<typename... Args>
    void unpack(const TensorVec<IndexPosition>& ipmask, const PermGroup& pg,
                PermGroup p, Args... rest) {
        pg = p;
        unpack(ipmask, pg, rest...);
    }

    template<typename ItemType, typename... Args>
    void pack(TensorVec<ItemType>& isl, Args...) {}

    template<typename ItemType, typename... Args>
    void pack(TensorVec<ItemType>& item_vec, ItemType item, Args... rest) {
        item_vec.push_back(item);
        pack(item_vec, rest...);
    }

    TensorVec<IndexRange> ranges(const TensorVec<IndexLabel>& labels) {
        TensorVec<IndexRange> ret = {};
        for(auto& label : labels) { ret.push_back(label.ir()); }
        return ret;
    }

    TensorVec<IndexRange> ranges(
      const TensorVec<const IndexSpace*>& index_spaces) {
        TensorVec<IndexRange> ret = {};
        for(auto& is : index_spaces) { ret.push_back(is->NR()); }
        return ret;
    }
};

template<typename T>
class Tensor {
    public:
    using ElementType = T;

    Tensor()              = default;
    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;
    ~Tensor()                        = default;

    Tensor(Tensor&& tensor) : impl_{std::move(tensor.impl_)} {}

    template<typename TensorImplT, typename... Args>
    static Tensor create(Args&&... args) {
        static_assert(
          std::is_same<ElementType, typename TensorImplT::ElementType>::value,
          "Mismatched element type between Tensor and Tensor implementation "
          "classes");

        return Tensor{
          std::make_shared<TensorImplT>(std::forward<Args>(args)...)};
    }

    template<unsigned N, typename TensorImplT, typename... Args>
    static auto create_list(Args&&... args) {
        static_assert(
          std::is_same<ElementType, typename TensorImplT::ElementType>::value,
          "Mismatched element type between Tensor and Tensor implementation "
          "classes");
        return CreateListImpl<N, Args...>::create(args...);
    }

    TensorRank rank() const {
        EXPECTS(impl_);
        return impl_->rank();
    }

    TensorVec<IndexRange> dim_ranges() const {
        EXPECTS(impl_);
        return impl_->dim_ranges();
    }

    bool is_unique(const BlockDimVec& bdv) const {
        EXPECTS(impl_);
        return impl_->is_unique(bdv);
    }

    BlockDimVec get_unique(const BlockDimVec& bdv) const {
        EXPECTS(impl_);
        return impl_->get_unique(bdv);
    }

    bool is_nonzero(const BlockDimVec& bdv) const {
        EXPECTS(impl_);
        return impl_->is_nonzero(bdv);
    }

    Size block_size(const BlockDimVec& bdv) const {
        EXPECTS(impl_);
        return impl_->block_size(bdv);
    }

    // @to-do: implement
    LabeledTensor<T> operator()();

    LabeledTensor<T> operator()(const IndexLabelVec& ilv) const {
        EXPECTS(impl_);
        return LabeledTensor<T>{*this, ilv};
    }

    template<typename... Args>
    LabeledTensor<T> operator()(IndexLabel ilbl, Args... rest) {
        EXPECTS(impl_);
        IndexLabelVec label{ilbl};
        pack(label, rest...);
        return (*this)(label);
    }

    void set_proc_group(ProcGroup proc_group) {
        EXPECTS(impl_);
        return impl_->set_proc_group(proc_group);
    }

    // Tensor& set_distribution(Distribution* distribution) {
    // EXPECTS(impl_);
    //   impl_->set_distribution(distribution);
    //   return *this;
    // }

    // Tensor& set_memory_manager(MemoryManager* memory_manager) {
    // EXPECTS(impl_);
    //   impl_->set_memory_manager(memory_manager);
    //   return *this;
    // }

    ProcGroup proc_group() const {
        EXPECTS(impl_);
        return impl_->proc_group();
    }

    PermGroup perm_group() const {
        EXPECTS(impl_);
        return impl_->perm_group();
    }

    // Distribution* distribution() {
    // EXPECTS(impl_);
    //   return impl_->distribution();
    // }

    // MemoryManager* memory_manager() {
    // EXPECTS(impl_);
    //   return impl_->memory_manager();
    // }

    void allocate() {
        EXPECTS(impl_);
        impl_->allocate();
    }

    bool is_allocated() const {
        EXPECTS(impl_);
        impl_->is_allocated();
    }

    void deallocate() {
        EXPECTS(impl_);
        impl_->deallocate();
    }

    // void attach(MemoryRegion* memory_region) {
    // EXPECTS(impl_);
    //   impl_->attach(memory_region);
    // }

    bool is_attached() const {
        EXPECTS(impl_);
        impl_->is_allocated();
    }

    void detach() {
        EXPECTS(impl_);
        impl_->detach();
    }

    bool is_constructed() const {
        EXPECTS(impl_);
        impl_->is_constructed();
    }

    AllocationStatus allocation_status() const {
        EXPECTS(impl_);
        impl_->allocation_status();
    }

    // std::pair<Codelet*,Codelet*> get(const BlockDimVec& bdv, Block<T>& block)
    // const { EXPECTS(impl_);
    //   return impl_->get(bdv, block);
    // }

    // std::pair<Codelet*,Codelet*> put(const BlockDimVec& bdv, Block<T>& block)
    // { EXPECTS(impl_);
    //   return impl_->put(bdv, block);
    // }

    // std::pair<Codelet*,Codelet*> add(const BlockDimVec& bdv, Block<T>& block)
    // { EXPECTS(impl_);
    //   return impl_->add(bdv, block);
    // }

    protected:
    template<unsigned N, typename... Args>
    struct CreateListImpl {
        static auto create(Args... args) {
            return std::tuple_cat(
              CreateListImpl<N - 1, Args...>::create(args...),
              CreateListImpl<1, Args...>::create(args...));
        }
    };

    template<typename... Args>
    struct CreateListImpl<1, Args...> {
        static auto create(Args... args) {
            return std::make_tuple(
              Tensor<T>::template create<TensorImpl<T>>(args...));
        }
    };

    void pack(IndexLabelVec& label) {}

    template<typename... Args>
    void pack(IndexLabelVec& label, IndexLabel ilbl, Args... rest) {
        label.push_back(ilbl);
        pack(label, rest...);
    }

    Tensor(std::shared_ptr<TensorImpl<T>> impl) : impl_{impl} {}

    std::shared_ptr<TensorImpl<T>> impl_;
};

template<typename T>
template<typename... Args>
Tensor<T> TensorImpl<T>::create(Args... args) {
    return Tensor<T>::template create<TensorImpl<T>>(args...);
}

template<typename T>
template<unsigned N, typename... Args>
auto TensorImpl<T>::create_list(Args... args) {
    return Tensor<T>::template create_list<N, TensorImpl<T>>(args...);
}

} // namespace tammy

#endif // TAMMY_TENSOR_H_
