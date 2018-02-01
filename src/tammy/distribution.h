#ifndef TAMMY_DISTRIBUTION_H_
#define TAMMY_DISTRIBUTION_H_

#include <memory>
#include <tuple>
#include <map>
#include <type_traits>
//#include "ga.h"

#include "tammy/types.h"
//#include "tammy/tce.h"
#include "tammy/tensor_base.h"

namespace tammy {

//class TensorBase;

class Distribution {
 public:
  virtual ~Distribution() {}
  virtual std::pair<Proc,Offset> locate(const BlockDimVec& blockid) = 0;
  virtual Size buf_size(Proc proc) const = 0;
  virtual std::string name() const = 0;
  virtual Distribution* clone(const TensorBase*, Proc) const = 0;

 protected:
  Distribution(const TensorBase* tensor_structure, Proc nproc)
      : tensor_structure_{tensor_structure},
        nproc_{nproc} {}

  const TensorBase* tensor_structure_;
  Proc nproc_;
};

#if 0
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
#endif

class Distribution_NW : public Distribution {
 public:
  static const std::string class_name;// = "Distribution_NW";

  std::string name() const {
    return "Distribution_NW"; //Distribution_NW::class_name;
  }

  Distribution* clone(const TensorBase* tensor_structure, Proc nproc) const {
/** \warning
*  totalview LD on following statement
*  back traced to tammy::Tensor<double>::alloc shared_ptr_base.h
*  backtraced to ccsd_driver<double> execution_context.h
*  back traced to main
*/
    return new Distribution_NW(tensor_structure, nproc);
  }

  std::pair<Proc,Offset> locate(const BlockDimVec& blockid) {
    auto key = compute_key(blockid);
    auto length = hash_[0];
    auto ptr = std::lower_bound(&hash_[1], &hash_[length + 1], key);
    EXPECTS (ptr != &hash_[length + 1]);
    EXPECTS (key == *ptr);
    EXPECTS (ptr != &hash_[length + 1] && key == *ptr);
    auto ioffset = *(ptr + length);
    auto pptr = std::upper_bound(std::begin(proc_offsets_), std::end(proc_offsets_), Offset{ioffset});
    EXPECTS(pptr != std::begin(proc_offsets_));
    auto proc = Proc{pptr - std::begin(proc_offsets_)};
    proc -= 1;
    Offset offset = Offset{ioffset - proc_offsets_[proc.template value<size_t>()];
    return {proc, offset};
  }

  Size buf_size(Proc proc) const {
    EXPECTS(proc >= Proc{0});
    EXPECTS(proc < nproc_);
    EXPECTS(proc_offsets_.size() > static_cast<uint64_t>(proc.value()+1));
    return proc_offsets_[proc.template value<size_t>()+1] - proc_offsets_[proc.template value<size_t>()];
  }

 public:
  Distribution_NW(const TensorBase* tensor_structure=nullptr, Proc nproc = Proc{1})
      : Distribution{tensor_structure, nproc} {
    if(tensor_structure == nullptr) {
      return;
    }

#if 1
    int64_t length = 0;
    for(auto& uitr = *tensor_structure_->unique_generator();
        uitr.has_more(); uitr.next()) {
      BlockDimVec blockid = uitr.get();
      if (tensor_structure_->nonzero(blockid)) {
        length += 1;
      }
    }
    EXPECTS(length > 0);

    hash_.resize(2*length + 1);
    hash_[0] = length;

    //start over
    int64_t offset = 0;
    int addr = 1;
    for(auto& uitr = *tensor_structure_->unique_generator();
        uitr.has_more(); uitr.next()) {
      BlockDimVec blockid = uitr.get();
      if(tensor_structure_->nonzero(blockid)) {
        hash_[addr] = compute_key(blockid);
        EXPECTS(addr==1 || hash_[addr] > hash_[addr-1]);
        hash_[length + addr] = offset;
        // if(GA_Nodeid() == 1) {
        //   //std::cerr<<"-----DISTRIBUTIO_NW. addr="<<addr<<" offset="<<offset<<" block_size="<<tensor_structure_->block_size(blockid)<<"\n";
        // }
        offset += tensor_structure_->block_size(blockid);
        addr += 1;
      }
    }
#else   
    auto indices = tensor_structure_->tindices();
    auto pdt =  loop_iterator(indices);
    auto last = pdt.get_end();
    int length = 0;
    for(auto itr = pdt; itr != last; ++itr) {
      if (tensor_structure_->nonzero(*itr)) {
        length += 1;
      }
    }
    EXPECTS(length > 0);

    hash_.resize(2*length + 1);
    hash_[0] = length;
    //start over
    pdt =  loop_iterator(indices);
    last = pdt.get_end();
    int64_t offset = 0;
    int addr = 1;

    for(auto itr = pdt; itr != last; ++itr) {
      auto blockid = *itr;
      if(tensor_structure_->nonzero(blockid)) {
        hash_[addr] = compute_key(blockid);
        EXPECTS(addr==1 || hash_[addr] > hash_[addr-1]);
        hash_[length + addr] = offset;
        // if(GA_Nodeid() == 1) {
        //   //std::cerr<<"-----DISTRIBUTIO_NW. addr="<<addr<<" offset="<<offset<<" block_size="<<tensor_structure_->block_size(blockid)<<"\n";
        // }
        offset += tensor_structure_->block_size(blockid);
        addr += 1;
      }
    }
#endif
    
    EXPECTS(offset > 0);
    total_size_ = offset;

    //int64_t max_2nd_arg=1;
    int64_t per_proc_size = std::max(int64_t{offset / nproc.template value<int64_t>()}, int64_t{1});//max_2nd_arg);
    auto itr = hash_.begin() + length + 1;
    auto itr_last = hash_.end();

    // if(GA_Nodeid() == 1) {
    //   //std::cerr<<"------DISTRIB_NW. total size="<<total_size_<<" nproc="<<nproc<<" per_proc_size="<<per_proc_size<<"\n";
    // }
    for(Proc i{0}; i < nproc; i++) {
      if(itr != itr_last) {
        proc_offsets_.push_back(Offset{*itr});
      } else {
        proc_offsets_.push_back(Offset{total_size_});        
      }
      
      itr = std::lower_bound(itr, itr_last, (i+1)*per_proc_size);
      // if(GA_Nodeid() == 1) {
      //   //std::cerr<<"------DISTRIB_NW. *new_itr="<<*itr<<"\n";
      // }
    }

    EXPECTS(proc_offsets_.size() == static_cast<uint64_t>(nproc.value()));
    proc_offsets_.push_back(total_size_);

    // if(GA_Nodeid() == 1){
    //   // std::cerr<<"------PROC OFFSETS:";
    //   // for(auto off: proc_offsets_) {
    //   //   std::cerr<<off<<" ";
    //   // }
    //   // std::cerr<<"\n";
    // }
  }

  const std::vector<int64_t>& hash() const {
    return hash_;
  }

private:
  int64_t compute_key(const BlockDimVec& blockid) const {
    int64_t key;
    TensorVec<int64_t> offsets, bases;

    const auto& drs = tensor_structure_->dim_ranges();
    for(const auto &dr: drs) {
      offsets.push_back(dr.bhi().value() - dr.blo().value());
    }
    for(const auto& dr: drs) {
      bases.push_back(dr.blo().value());
    }
    int rank = tensor_structure_->rank();
    int64_t offset = 1;
    key = 0;
    for(int i=rank-1; i>=0; i--) {
      EXPECTS(blockid[i] >= drs[i].blo());
      EXPECTS(blockid[i] < drs[i].bhi());
      key += ((blockid[i].value() - bases[i]) * offset);
      offset *= offsets[i];
    }
    return key;
  }

  std::vector<int64_t> hash_;
  std::vector<Offset> proc_offsets_;
  Offset total_size_;

  friend class DistributionFactory;
}; // class Distribution_NW

}  // namespace tammy

#endif // TAMMY_DISTRIBUTION_H_
