// Copyright 2016 Pacific Northwest National Laboratory

#ifndef TAMMY_MSO_H_
#define TAMMY_MSO_H_

#include <numeric>
#include <vector>

#include "errors.h"
#include "index_space.h"
#include "strong_num_indexed_vector.h"
#include "types.h"

namespace tammy {

class MSO : public IndexSpace {
    public:
    MSO() = default; //@todo delete after debugging
    MSO(BlockCount noa, BlockCount nob, BlockCount nva, BlockCount nvb,
        const std::vector<Irrep>& spatials, const std::vector<Spin>& spins,
        const std::vector<Size>& sizes) :
      noa_{noa},
      nob_{nob},
      nva_{nva},
      nvb_{nvb},
      spatials_{spatials},
      spins_{spins},
      sizes_(sizes) {
        // EXPECTS(noa_ >=0 && nva_ >=0);
        // EXPECTS(nob_ >=0 && nvb_ >= 0);
        BlockCount nov_ab = noa_ + nob_ + nva_ + nvb_;
        EXPECTS(spatials_.size() == nov_ab);
        EXPECTS(spins_.size() == nov_ab);
        EXPECTS(sizes_.size() == nov_ab);

        offsets_.push_back(Offset{0});
        std::partial_sum(sizes_.begin(), sizes_.end(),
                         std::back_inserter(offsets_));
        for(auto s : spins_) { EXPECTS(s == Spin{1} || s == Spin{2}); }
        for(BlockIndex i{0}; i < nov_ab; i++) { block_indices_.push_back(i); }
    }

    bool has_spatial() const override { return true; }
    Irrep spatial(BlockIndex bid) const override { return spatials_[bid]; }
    bool has_spin() const override { return true; }
    Spin spin(BlockIndex bid) const override { return spins_[bid]; }
    Size size(BlockIndex bid) const override { return sizes_[bid]; }
    Offset offset(BlockIndex bid) const override { return offsets_[bid]; }
    int num_indep_indices() const override { return 0; }
    Iterator begin(
      RangeValue rv,
      const TensorVec<IndexSpace::Iterator>& bdv = {}) const override {
        EXPECTS(bdv.size() == 0);
        auto ret = block_indices_.begin();
        BlockCount off;
        switch(rv) {
            case range_n: off = BlockCount{0}; break;
            case range_o: off = BlockCount{0}; break;
            case range_oa: off = BlockCount{0}; break;
            case range_ob: off = noa_; break;
            case range_v: off = noa_ + nob_; break;
            case range_va: off = noa_ + nob_; break;
            case range_vb: off = noa_ + nob_ + nva_; break;
            case range_e: off = noa_ + nob_ + nva_ + nvb_; break;
            default: assert(0);
        }
        using size_type = decltype(block_indices_)::size_type;
        return ret + off.template value<size_type>();
    }

    Iterator end(RangeValue rv, const TensorVec<IndexSpace::Iterator>& bdv = {})
      const override {
        EXPECTS(bdv.size() == 0);
        auto ret = block_indices_.begin();
        BlockCount off;
        switch(rv) {
            case range_n: off = noa_ + nob_ + nva_ + nvb_; break;
            case range_o: off = noa_ + nob_; break;
            case range_oa: off = noa_; break;
            case range_ob: off = noa_ + nob_; break;
            case range_v: off = noa_ + nob_ + nva_ + nvb_; break;
            case range_va: off = noa_ + nob_ + nva_; break;
            case range_vb: off = noa_ + nob_ + nva_ + nvb_; break;
            case range_e: off = noa_ + nob_ + nva_ + nvb_; break;
            default: assert(0);
        }
        using size_type = decltype(block_indices_)::size_type;
        return ret + off.template value<size_type>();
    }

    bool is_superset_of(RangeValue rv1, RangeValue rv2) const override {
        return (rv1 & rv2) == rv2;
    }

    IndexRange NR() const override { return {*this, range_n}; }
    IndexRange ER() const override { return {*this, range_e}; }

    IndexRange OR() const { return {*this, range_o}; }

    IndexRange VR() const { return {*this, range_v}; }
    IndexRange OaR() const { return {*this, range_oa}; }
    IndexRange VaR() const { return {*this, range_va}; }
    IndexRange ObR() const { return {*this, range_ob}; }
    IndexRange VbR() const { return {*this, range_vb}; }

    template<typename... LabelArgs>
    auto O(LabelArgs... labels) const {
        return OR().labels(labels...);
    }
    template<typename... LabelArgs>
    auto V(LabelArgs... labels) const {
        return VR().labels(labels...);
    }
    template<typename... LabelArgs>
    auto Oa(LabelArgs... labels) const {
        return OaR().labels(labels...);
    }
    template<typename... LabelArgs>
    auto Va(LabelArgs... labels) const {
        return VaR().labels(labels...);
    }
    template<typename... LabelArgs>
    auto Ob(LabelArgs... labels) const {
        return ObR().labels(labels...);
    }
    template<typename... LabelArgs>
    auto Vb(LabelArgs... labels) const {
        return VbR().labels(labels...);
    }

    protected:
    BlockCount noa_;
    BlockCount nob_;
    BlockCount nva_;
    BlockCount nvb_;

    StrongNumIndexedVector<Irrep, BlockIndex> spatials_;
    StrongNumIndexedVector<Spin, BlockIndex> spins_;
    StrongNumIndexedVector<Size, BlockIndex> sizes_;
    StrongNumIndexedVector<Offset, BlockIndex> offsets_;
    std::vector<BlockIndex> block_indices_;

    static const RangeValue range_e  = 0x00;
    static const RangeValue range_oa = 0x01;
    static const RangeValue range_ob = 0x02;
    static const RangeValue range_o  = 0x03;
    static const RangeValue range_va = 0x10;
    static const RangeValue range_vb = 0x20;
    static const RangeValue range_v  = 0x30;
    static const RangeValue range_n  = 0x33;
}; // MSO

} // namespace tammy

#endif // TAMMY_MSO_H_
