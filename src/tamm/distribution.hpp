#ifndef TAMM_DISTRIBUTION_H_
#define TAMM_DISTRIBUTION_H_

#include "ga.h"
#include <map>
#include <memory>
#include <tuple>
#include <type_traits>

#include "tamm/tensor_base.hpp"
#include "tamm/types.hpp"

namespace tamm {

class TensorBase;

class Distribution {
public:
    virtual ~Distribution() {}
    virtual std::pair<Proc, Offset> locate(const IndexVector& blockid) = 0;
    virtual Size buf_size(Proc proc) const                             = 0;
    // virtual std::string name() const = 0;
    virtual Distribution* clone(const TensorBase*, Proc) const = 0;

    Distribution(const TensorBase* tensor_structure, Proc nproc) :
      tensor_structure_{tensor_structure},
      nproc_{nproc} {}

protected:
    const TensorBase* tensor_structure_;
    Proc nproc_;
};

class Distribution_NW : public Distribution {
public:
    static const std::string class_name; // = "Distribution_NW";

    using HashValue = Integer;
    using Key       = HashValue;

    std::string name() const {
        return "Distribution_NW"; // Distribution_NW::class_name;
    }

    Distribution* clone(const TensorBase* tensor_structure, Proc nproc) const {
        /** \warning
         *  totalview LD on following statement
         *  back traced to tamm::Tensor<double>::alloc shared_ptr_base.h
         *  backtraced to ccsd_driver<double> execution_context.h
         *  back traced to main
         */
        return new Distribution_NW(tensor_structure, nproc);
    }

    std::pair<Proc, Offset> locate(const IndexVector& blockid) {
        auto key = compute_key(blockid);
        auto itr = std::lower_bound(
          hash_.begin(), hash_.end(), KeyOffsetPair{key, 0},
          [](const KeyOffsetPair& lhs, const KeyOffsetPair& rhs) {
              return lhs.key_ < rhs.key_;
          });
        EXPECTS(itr != hash_.end());
        EXPECTS(key == itr->key_);
        auto ioffset = itr->offset_;
        auto pptr    = std::upper_bound(std::begin(proc_offsets_),
                                     std::end(proc_offsets_), Offset{ioffset});
        EXPECTS(pptr != std::begin(proc_offsets_));
        auto proc = Proc{pptr - std::begin(proc_offsets_)};
        proc -= 1;
        auto offset = Offset{ioffset - proc_offsets_[proc.value()].value()};
        return {proc, offset};
    }

    Size buf_size(Proc proc) const {
        EXPECTS(proc >= 0);
        EXPECTS(proc < nproc_);
        EXPECTS(proc_offsets_.size() > static_cast<uint64_t>(proc.value()) + 1);
        return proc_offsets_[proc.value() + 1] - proc_offsets_[proc.value()];
    }

public:
    Distribution_NW(const TensorBase* tensor_structure = nullptr,
                    Proc nproc                         = Proc{1}) :
      Distribution{tensor_structure, nproc} {
        EXPECTS(nproc > 0);
        if(tensor_structure == nullptr) { return; }

        compute_key_offsets();
        for(const auto& blockid : tensor_structure_->loop_nest()) {
            if(tensor_structure_->is_non_zero(blockid))
                hash_.push_back({compute_key(blockid),
                                 tensor_structure_->block_size(blockid)});
        }
        EXPECTS(hash_.size() > 0);

        std::sort(hash_.begin(), hash_.end(),
                  [](const KeyOffsetPair& lhs, const KeyOffsetPair& rhs) {
                      return lhs.key_ < rhs.key_;
                  });

        Offset offset = 0;
        for(size_t i = 0; i < hash_.size(); i++) {
            auto sz          = hash_[i].offset_;
            hash_[i].offset_ = offset;
            offset += sz;
        }
        EXPECTS(offset > 0);
        total_size_ = offset;

        Offset per_proc_size = std::max(offset / nproc.value(), Offset{1});
        auto itr             = hash_.begin();
        auto itr_last        = hash_.end();
        for(int i = 0; i < nproc.value(); i++) {
            if(itr != itr_last) {
                proc_offsets_.push_back(Offset{itr->offset_});
            } else {
                proc_offsets_.push_back(Offset{total_size_});
            }
            itr =
              std::lower_bound(itr, itr_last, (Offset{i} + 1) * per_proc_size,
                               [](const KeyOffsetPair& hv, const Offset& v) {
                                   return hv.offset_ < v;
                               });
        }
        EXPECTS(proc_offsets_.size() == static_cast<uint64_t>(nproc.value()));
        proc_offsets_.push_back(total_size_);
    }

private:
    struct KeyOffsetPair {
        HashValue key_;
        Offset offset_;
    };
    void compute_key_offsets() {
        const auto& tis_list = tensor_structure_->tindices();
        int rank             = tis_list.size();
        key_offsets_.resize(rank);
        if(rank > 0) { key_offsets_[rank - 1] = 1; }
        for(int i = rank - 2; i >= 0; i--) {
            key_offsets_[i] =
              key_offsets_[i + 1] * tis_list[i + 1].max_num_tiles();
        }
    }

    Key compute_key(const IndexVector& blockid) const {
        Key key{0};
        auto rank = tensor_structure_->tindices().size();
        for(size_t i = 0; i < rank; i++) {
            key += blockid[i] * key_offsets_[i].value();
        }
        return key;
    }

    Offset total_size_;
    std::vector<KeyOffsetPair> hash_;
    std::vector<Offset> proc_offsets_;
    std::vector<Offset> key_offsets_;

    friend class DistributionFactory;
}; // class Distribution_NW

} // namespace tamm

#endif // TAMM_DISTRIBUTION_H_
