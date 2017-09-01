// Copyright 2016 Pacific Northwest National Laboratory

#ifndef TAMMX_TCE_H__
#define TAMMX_TCE_H__

#include <cassert>
#include <iosfwd>
#include <numeric>
#include <iterator>
#include <sstream>
#include "tammx/boundvec.h"
#include "tammx/types.h"

namespace tammx {

class TCE {
 public:
  static void init(const std::vector<Spin>& spins,
                   const std::vector<Irrep>& spatials,
                   const std::vector<size_t>& sizes,
                   BlockDim noa,
                   BlockDim noab,
                   BlockDim nva,
                   BlockDim nvab,
                   bool spin_restricted,
                   Irrep irrep_f,
                   Irrep irrep_v,
                   Irrep irrep_t,
                   Irrep irrep_x,
                   Irrep irrep_y) {
    spins_ = spins;
    spatials_ = spatials;
    sizes_ = sizes;
    noa_ = noa;
    noab_ = noab;
    nva = nva_;
    nvab_ = nvab;
    spin_restricted_ = spin_restricted;
    irrep_f_ = irrep_f;
    irrep_v_ = irrep_v;
    irrep_t_ = irrep_t;
    irrep_x_ = irrep_x;
    irrep_y_ = irrep_y;

    offsets_.push_back(0);
    std::partial_sum(sizes_.begin(), sizes_.end(), std::back_inserter(offsets_));

    Expects(noab_ >=0 && nvab_ >=0);
    Expects(noa_ <= noab_ && nva_ <= nvab_);
    auto sz = noab_.value() + nvab_.value();
    Expects(spins_.size() == sz);
    Expects(spatials_.size() == sz);
    Expects(sizes_.size() == sz);
    Expects(offsets_.size() == sz+1);

    for(auto s: spins_) {
      Expects(s==Spin{1} || s==Spin{2});
    }
  }

  static void finalize() {
    // no-op
  }

  static size_t total_dim_size() {
    return offsets_.back();
  }

  static Spin spin(BlockDim block) {
    return spins_[block.value()-1];
  }

  static Irrep spatial(BlockDim block) {
    return spatials_[block.value()-1];
  }

  static size_t size(BlockDim block) {
    return sizes_[block.value()-1];
  }

  static size_t offset(BlockDim block) {
    return offsets_[block.value()-1];
  }

  static bool restricted() {
    return spin_restricted_;
  }

  static BlockDim noab() {
    return noab_;
  }

  static BlockDim nvab() {
    return nvab_;
  }

  static BlockDim noa() {
    return noa_;
  }

  static BlockDim nob() {
    return noab() - noa();
  }

  static BlockDim nva() {
    return nva_;
  }

  static BlockDim nvb() {
    return nvab() - nva();
  }

  using Int = Fint;

  static Int compute_tce_key(const TensorDim& flindices,
                             const TensorIndex& is) {
    //auto flindices = flatten(indices);
    TensorVec<Int> offsets(flindices.size()), bases(flindices.size());
    std::transform(flindices.begin(), flindices.end(), offsets.begin(),
                   [] (DimType dt) -> Int {
                     if (dt == DimType::o) {
                       return noab().value();
                     } else if (dt == DimType::v) {
                       return nvab().value();
                     } else if (dt == DimType::n) {
                       return noab().value() + nvab().value();
                     } else {
                       assert(0); //implement
                     }
                   });

    std::transform(flindices.begin(), flindices.end(), bases.begin(),
                   [] (DimType dt) -> Int {
                     if (dt == DimType::o) {
                       return 1;
                     } else if (dt == DimType::v) {
                       return noab().value() + 1;
                     } else if (dt == DimType::n) {
                       return 1;
                     } else {
                       assert(0); //implement
                     }
                   });

    int rank = flindices.size();
    Int key = 0, offset = 1;
    for(int i=rank-1; i>=0; i--) {
      key += ((is[i] - bases[i]) * offset).value();
      offset *= offsets[i];
    }
    return key;
  }

  static BlockDim dim_lo(DimType dt) {
    BlockDim ret;
    switch(dt) {
      case DimType::o:
      case DimType::oa:
      case DimType::n:
        ret = 1;
        break;
      case DimType::ob:
        ret = noa()+1;
        break;
      case DimType::v:
      case DimType::va:
        ret = noab() + 1;
        break;
      case DimType::vb:
        ret = noab() + nvb() + 1;
        break;
      default:
        assert(0);
    }
    return ret;
  }

  static BlockDim dim_hi(DimType dt) {
    BlockDim ret;
    switch(dt) {
      case DimType::oa:
        ret = noa() + 1;
        break;
      case DimType::o:
      case DimType::ob:
        ret = noab() + 1;
        break;
      case DimType::va:
        ret = noab() + nva() + 1;
        break;
      case DimType::vb:
      case DimType::v:
      case DimType::n:
        ret = noab() + nvab() + 1;
        break;
      default:
        assert(0);
    }
    return ret;
  }

 private:
  static std::vector<Spin> spins_;
  static std::vector<Irrep> spatials_;
  static std::vector<size_t> sizes_;
  static std::vector<size_t> offsets_;
  static bool spin_restricted_;
  static Irrep irrep_f_, irrep_v_, irrep_t_;
  static Irrep irrep_x_, irrep_y_;
  static BlockDim noa_, noab_;
  static BlockDim nva_, nvab_;
};

inline std::pair<BlockDim, BlockDim>
tensor_index_range(DimType dt) {
  return {TCE::dim_lo(dt), TCE::dim_hi(dt)};
  // switch(dt) {
  //   case DimType::o:
  //     return {BlockDim{0}, TCE::noab()};
  //     break;
  //   case DimType::v:
  //     return {TCE::noab(), TCE::noab()+TCE::nvab()};
  //     break;
  //   case DimType::n:
  //     return {BlockDim{0}, TCE::noab() + TCE::nvab()};
  //     break;
  //   default:
  //     assert(0);
  // }
}

class RangeType {
 public:
  RangeType() = default;
  RangeType(const RangeType&) = default;
  ~RangeType() = default;
  RangeType(RangeType&&) = default;
  RangeType& operator = (const RangeType&) = default;
  RangeType& operator = (RangeType&&) = default;
  
  RangeType(DimType dt,
            BlockDim blo = BlockDim{0},
            BlockDim bhi = BlockDim{0})
      : dt_{dt},
        blo_{blo},
        bhi_{bhi_} {
    if(dt != DimType::c) {
      std::tie(blo_,bhi_) = tensor_index_range(dt_);
    }
    Expects(blo_ >= BlockDim{1});
    if(bhi_ < BlockDim{1}) {
      bhi_ = blo_ + 1;
    }
  }

  DimType dt() const {
    return dt_;
  }

  std::pair<BlockDim,BlockDim> range() const {
    return {blo_, bhi_};
  }

  BlockDim blo() const {
    return blo_;
  }

  BlockDim bhi() const {
    return bhi_;
  }

 private:
  DimType dt_;
  BlockDim blo_, bhi_;
};

inline bool
operator == (const RangeType& lhs, const RangeType& rhs) {
  return (lhs.dt() == rhs.dt()) &&
      (lhs.blo() == rhs.blo()) &&
      (lhs.bhi() == rhs.bhi());
}

inline bool
operator != (const RangeType& lhs, const RangeType& rhs) {
  return !(lhs == rhs);
}

inline bool
operator <= (const RangeType& lhs, const RangeType& rhs) {
  return (lhs.dt() <= rhs.dt()) &&
      (lhs.blo() <= rhs.blo()) &&
      (lhs.bhi() <= rhs.bhi());
}

inline bool
operator < (const RangeType& lhs, const RangeType& rhs) {
  return (lhs <= rhs) && (lhs != rhs);
}

inline std::pair<BlockDim,BlockDim>
tensor_index_range(const RangeType& rt) {
  return rt.range();
}  

inline bool
is_range_subset(const RangeType& superset,
                const RangeType& subset) {
  if(superset.dt() != DimType::c
     && subset.dt() != DimType::c) {
    return is_dim_subset(superset.dt(), subset.dt());
  }
  return subset.blo() >= superset.blo()
      && subset.bhi() <= superset.bhi();
}

inline std::string
to_string(const RangeType& rt) {
  switch(rt.dt()) {
    case DimType::o:
      return "o";
      break;
    case DimType::v:
      return "v";
      break;
    case DimType::oa:
      return "oa";
      break;
    case DimType::ob:
      return "ob";
      break;
    case DimType::va:
      return "va";
      break;
    case DimType::vb:
      return "vb";
      break;
    case DimType::n:
      return "n";
      break;
    case DimType::c:
      return static_cast<std::ostringstream*>(&(std::ostringstream()<<"c"<<rt.blo()<<".."<<rt.bhi()<<"]"))->str();
      break;
    default:
      assert(0);
  }
}

class TensorSymmGroup {
 public:
  TensorSymmGroup() = default;
  TensorSymmGroup(const RangeType& rt, size_t grp_size)
      : rt_{rt},
        grp_size_{grp_size} {}
  
  DimType dt() const {
    return rt_.dt();
  }

  const RangeType& rt() const {
    return rt_;
  }
  
  size_t size() const {
    return grp_size_;
  }
 private:
  RangeType rt_;
  size_t grp_size_;
};

inline bool
operator == (const TensorSymmGroup& lhs, const TensorSymmGroup& rhs) {
  return lhs.size() == rhs.size()
      && lhs.rt() == rhs.rt();
}

inline bool
operator <= (const TensorSymmGroup& lhs, const TensorSymmGroup& rhs) {
  return lhs.size() <= rhs.size()
      && lhs.rt() <= rhs.rt();
}

inline bool
operator != (const TensorSymmGroup& lhs, const TensorSymmGroup& rhs) {
  return !(lhs == rhs);
}

inline bool
operator < (const TensorSymmGroup& lhs, const TensorSymmGroup& rhs) {
  return (lhs <= rhs) & (lhs != rhs);
}



using TensorRange = TensorVec<RangeType>;


}; //namespace tammx


#endif  // TAMMX_UTIL_H__
