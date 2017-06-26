#ifndef TAMMX_DISTRIBUTION_H_
#define TAMMX_DISTRIBUTION_H_

#include <memory>
#include <tuple>
#include <map>
#include <type_traits>

#include "tammx/types.h"
#include "tammx/tce.h"
#include "tammx/tensor_base.h"

namespace tammx {

class TensorBase;

class Distribution {
 public:
  virtual ~Distribution() {}
  virtual std::pair<Proc,Offset> locate(const TensorIndex& blockid) = 0;
  virtual Size buf_size(Proc proc) = 0;
  virtual std::string name() const = 0;
  virtual Distribution* clone(const TensorBase*, Proc) const = 0;

 protected:
  Distribution(const TensorBase* tensor_structure, Proc nproc)
      : tensor_structure_{tensor_structure},
        nproc_{nproc} {}

  const TensorBase* tensor_structure_;
  Proc nproc_;
};

// @fixme Can this code be cleaned up?
class DistributionFactory {
 public:
  DistributionFactory() = delete;
  DistributionFactory(const DistributionFactory&) = delete;

  template<typename DistributionType>
  static std::shared_ptr<Distribution> make_distribution(const TensorBase* tensor_structure,
                                                         Proc nproc) {
    static_assert(std::is_base_of<Distribution, DistributionType>(),
                  "Distribution type being created is not a subclass of Distribution");
    auto name = DistributionType::class_name; // DistributionType().name();
    auto itr = std::get<0>(
        distributions_.emplace(std::make_tuple(name, *tensor_structure, nproc),
                               std::make_shared<DistributionType>(tensor_structure, nproc)));
    return itr->second;
  }

  static std::shared_ptr<Distribution> make_distribution(const Distribution& distribution,
                                                         const TensorBase* tensor_structure,
                                                         Proc nproc) {
    auto name = distribution.name();
    auto key = std::make_tuple(name, *tensor_structure, nproc);
    if(distributions_.find(key) != distributions_.end()) {
      return distributions_.find(key)->second;
    }
    auto dist = std::shared_ptr<Distribution>{distribution.clone(tensor_structure, nproc)};
    distributions_[key] = dist;
    // return std::get<0>(distributions_.insert(key, std::shared_ptr<Distribution>(dist)))->second;
    return dist;
  }

 private:
  using Key = std::tuple<std::string,TensorBase,Proc>;
  struct KeyLessThan {
   public:
    bool operator() (const Key& lhs, const Key& rhs) const {
      auto lname = std::get<0>(lhs);
      auto lts = std::get<1>(lhs);
      auto lproc = std::get<2>(lhs);
      auto rname = std::get<0>(rhs);
      auto rts = std::get<1>(rhs);
      auto rproc = std::get<2>(rhs);
      return (lname < rname) ||
          (lname == rname && lts < rts) ||
          (lname == rname && lts == rts && lproc < rproc);
    }
  };
  static std::map<Key, std::shared_ptr<Distribution>,KeyLessThan> distributions_;
};  // class DistributionFactory

class Distribution_NW : public Distribution {
 public:
  static const std::string class_name;// = "Distribution_NW";

  std::string name() const {
    return "Distribution_NW"; //Distribution_NW::class_name;
  }

  Distribution* clone(const TensorBase* tensor_structure, Proc nproc) const {
    return new Distribution_NW(tensor_structure, nproc);
  }

  std::pair<Proc,Offset> locate(const TensorIndex& blockid) {
    auto key = compute_key(blockid);
    auto length = hash_[0];
    auto ptr = std::lower_bound(&hash_[1], &hash_[length + 1], key);
    Expects (ptr != &hash_[length + 1]);
    // std::cout<<"locate. key="<<key<<std::endl;
    Expects (key == *ptr);
    Expects (ptr != &hash_[length + 1] && key == *ptr);
    auto ioffset = *(ptr + length);
    auto pptr = std::upper_bound(std::begin(proc_offsets_), std::end(proc_offsets_), Offset{ioffset});
    Expects(pptr != std::begin(proc_offsets_));
    auto proc = Proc{pptr - std::begin(proc_offsets_)};
    proc -= 1;
    auto offset = Offset{ioffset - proc_offsets_[proc.value()].value()};
    return {proc, offset};
  }

  Size buf_size(Proc proc) {
    Expects(proc < nproc_);
    Expects(proc_offsets_.size() > proc.value()+1);
    return proc_offsets_[proc.value()+1] - proc_offsets_[proc.value()];
  }

 public:
  Distribution_NW(const TensorBase* tensor_structure=nullptr, Proc nproc = Proc{1})
      : Distribution{tensor_structure, nproc} {
    if(tensor_structure == nullptr) {
      return;
    }

    auto indices = tensor_structure_->indices();
    auto pdt =  loop_iterator(indices);
    auto last = pdt.get_end();
    int length = 0;
    for(auto itr = pdt; itr != last; ++itr) {
      if (tensor_structure_->nonzero(*itr)) {
        length += 1;
      }
    }
    Expects(length > 0);

    hash_.resize(2*length + 1);
    hash_[0] = length;
    //start over
    pdt =  loop_iterator(indices);
    last = pdt.get_end();
    TCE::Int offset = 0;
    int addr = 1;
    for(auto itr = pdt; itr != last; ++itr) {
      auto blockid = *itr;
      if(tensor_structure_->nonzero(blockid)) {
        hash_[addr] = compute_key(blockid);
        Expects(addr==1 || hash_[addr] > hash_[addr-1]);
        hash_[length + addr] = offset;
        offset += tensor_structure_->block_size(blockid);
        addr += 1;
      }
    }
    Expects(offset > 0);
    total_size_ = offset;

    auto per_proc_size = offset / nproc.value();
    auto itr = &hash_[length+1];
    auto itr_last = &hash_[2*length+1];
    for(int i=0; i<nproc.value(); i++) {
      proc_offsets_.push_back(Offset{*itr});
      itr = std::lower_bound(itr, itr_last, i*per_proc_size);
    }
    Expects(proc_offsets_.size() == nproc.value());
    proc_offsets_.push_back(total_size_);
  }

 private:
  TCE::Int compute_key(const TensorIndex& blockid) const {
    TCE::Int key;
    TensorVec<TCE::Int> offsets, bases;

    const auto &flindices = tensor_structure_->flindices();
    for(auto ind: flindices) {
      offsets.push_back(TCE::dim_hi(ind).value() - TCE::dim_lo(ind).value());
    }
    for(auto ind: flindices) {
      bases.push_back(TCE::dim_lo(ind).value());
    }
    int rank = flindices.size();
    TCE::Int offset = 1;
    key = 0;
    // std::cout<<"compute_key. blockid="<<blockid<<std::endl;
    // std::cout<<"compute_key. bases="<<bases<<std::endl;
    for(int i=rank-1; i>=0; i--) {
      Expects(blockid[i] >= TCE::dim_lo(flindices[i]));
      Expects(blockid[i] < TCE::dim_hi(flindices[i]));
      key += ((blockid[i].value() - bases[i]) * offset);
      offset *= offsets[i];
    }
    return key;
  }

  std::vector<TCE::Int> hash_;
  std::vector<Offset> proc_offsets_;
  Offset total_size_;

  friend class DistributionFactory;
}; // class Distribution_NW

}  // namespace tammx

#endif // TAMMX_DISTRIBUTION_H_
