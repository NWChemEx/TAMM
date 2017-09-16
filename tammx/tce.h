// Copyright 2016 Pacific Northwest National Laboratory

#ifndef TAMMX_TCE_H_
#define TAMMX_TCE_H_

#include <iosfwd>
#include <numeric>
#include <iterator>
#include <sstream>
#include "tammx/boundvec.h"
#include "tammx/types.h"
#include "tammx/errors.h"
#include "tammx/fortran.h"

namespace tammx {

class TCE {
 public:
  static void init(const std::vector<Spin>& spins,
                   const std::vector<Irrep>& spatials,
                   const std::vector<size_t>& sizes,
                   BlockIndex noa,
                   BlockIndex noab,
                   BlockIndex nva,
                   BlockIndex nvab,
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
    nva_ = nva;
    nvab_ = nvab;
    spin_restricted_ = spin_restricted;
    irrep_f_ = irrep_f;
    irrep_v_ = irrep_v;
    irrep_t_ = irrep_t;
    irrep_x_ = irrep_x;
    irrep_y_ = irrep_y;

    offsets_.push_back(0);
    std::partial_sum(sizes_.begin(), sizes_.end(), std::back_inserter(offsets_));

    EXPECTS(noab_ >=0 && nvab_ >=0);
    EXPECTS(noa_ <= noab_ && nva_ <= nvab_);
    auto sz = noab_.value() + nvab_.value();
    EXPECTS(spins_.size() == sz);
    EXPECTS(spatials_.size() == sz);
    EXPECTS(sizes_.size() == sz);
    EXPECTS(offsets_.size() == sz+1);

    for(auto s: spins_) {
      EXPECTS(s==Spin{1} || s==Spin{2});
    }
  }

  static void finalize() {
    // no-op
  }

  static size_t total_dim_size() {
    return offsets_.back();
  }

  static Spin spin(BlockIndex block) {
    return spins_[block.value()-1];
  }

  static Irrep spatial(BlockIndex block) {
    return spatials_[block.value()-1];
  }

  static size_t size(BlockIndex block) {
    return sizes_[block.value()-1];
  }

  static size_t offset(BlockIndex block) {
    return offsets_[block.value()-1];
  }

  static bool restricted() {
    return spin_restricted_;
  }

  static BlockIndex noab() {
    return noab_;
  }

  static BlockIndex nvab() {
    return nvab_;
  }

  static BlockIndex noa() {
    return noa_;
  }

  static BlockIndex nob() {
    return noab() - noa();
  }

  static BlockIndex nva() {
    return nva_;
  }

  static BlockIndex nvb() {
    return nvab() - nva();
  }

  using Int = FortranInt;

  static Int compute_tce_key(const DimTypeVec& flindices,
                             const BlockDimVec& is) {
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
                       NOT_IMPLEMENTED();
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
                       NOT_IMPLEMENTED();
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

  static BlockIndex dim_lo(DimType dt) {
    BlockIndex ret;
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
        UNREACHABLE();
    }
    return ret;
  }

  static BlockIndex dim_hi(DimType dt) {
    BlockIndex ret;
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
        UNREACHABLE();
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
  static BlockIndex noa_, noab_;
  static BlockIndex nva_, nvab_;
};

inline std::pair<BlockIndex, BlockIndex>
tensor_index_range(DimType dt) {
  return {TCE::dim_lo(dt), TCE::dim_hi(dt)};
}

class RangeType {
 public:
  RangeType(const RangeType&) = default;
  ~RangeType() = default;
  RangeType(RangeType&&) = default;
  RangeType& operator = (const RangeType&) = default;
  RangeType& operator = (RangeType&&) = default;

  RangeType(DimType dt = DimType::inv)
      : dt_{dt} {
    EXPECTS(dt != DimType::c);
  }

  RangeType(DimType dt,
            BlockIndex blo,
            BlockIndex bhi = BlockIndex{0})
      : dt_{dt},
        blo_{blo},
        bhi_{bhi} {
          EXPECTS(dt == DimType::c);
          if(bhi_ == BlockIndex{0}) {
            bhi_ = blo_ + 1;
          }
  }

  DimType dt() const {
    return dt_;
  }

  std::pair<BlockIndex,BlockIndex> range() const {
    return {blo(), bhi()};
  }

  BlockIndex blo() const {
    EXPECTS(dt_ != DimType::inv);
    if(dt_ == DimType::c) {
      return blo_;
    }
    return TCE::dim_lo(dt_);
  }

  BlockIndex bhi() const {
    EXPECTS(dt_ != DimType::inv);
    if(dt_ == DimType::c) {
      return bhi_;
    }
    return TCE::dim_hi(dt_);
  }

 private:
  DimType dt_;
  BlockIndex blo_, bhi_;
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
  return (lhs.dt() < rhs.dt()) ||
      (lhs.dt() == rhs.dt() && lhs.blo() < rhs.blo()) ||
      (lhs.dt() == rhs.dt() && lhs.blo() == rhs.blo() && lhs.bhi() < rhs.bhi());
}

inline std::pair<BlockIndex,BlockIndex>
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
    case DimType::c: {
      std::stringstream dtc;
      dtc << "c" << rt.blo() << ".." << rt.bhi() << "]";
      return dtc.str();
      break;
    }
    default:
      UNREACHABLE();
  }
}

class TensorSymmGroup {
 public:
  TensorSymmGroup()
      : rt_{DimType::inv},
        grp_size_{0} {}

  TensorSymmGroup(const RangeType& rt, size_t grp_size=1)
      : rt_{rt},
        grp_size_{grp_size} {}

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
  return (lhs.rt() < rhs.rt()) ||
      (lhs.rt() == rhs.rt() && lhs.size() < rhs.size());
}

class IndexLabel {
 public:
  //DimType dt;
  //RangeType rt;
  int label;

  IndexLabel() = default;
  IndexLabel(int lbl, const RangeType& rtype)
      : label{lbl},
        rt_{rtype} {
          EXPECTS(rt_.dt() != DimType::inv);
          EXPECTS(rt_.dt() != DimType::c);
          if(rt_.dt() != DimType::c) {
            EXPECTS(rt_.blo() == TCE::dim_lo(rt_.dt()));
            EXPECTS(rt_.bhi() == TCE::dim_hi(rt_.dt()));
          }
        }

  const RangeType& rt() const {
    return rt_;
  }
 private:
  RangeType rt_;
};



inline bool
operator == (const IndexLabel lhs, const IndexLabel rhs) {
  return lhs.label == rhs.label
      && lhs.rt() == rhs.rt();
}

inline bool
operator != (const IndexLabel lhs, const IndexLabel rhs) {
  return !(lhs == rhs);
}

inline bool
operator < (const IndexLabel lhs, const IndexLabel rhs) {
  return (lhs.label < rhs.label)
      || (lhs.label == rhs.label && lhs.rt() < rhs.rt());
}
inline std::ostream&
operator << (std::ostream& os, IndexLabel il) {
  std::string str;
  switch(il.rt().dt()) {
    case DimType::o:
      str = "h";
      break;
    case DimType::v:
      str = "p";
      break;
    case DimType::n:
      str = "n";
      break;
    default:
      NOT_IMPLEMENTED();
  }
  os<<str << il.label;
  return os;
}

using RangeTypeVec = TensorVec<RangeType>;
using IndexLabelVec = TensorVec<IndexLabel>;

} //namespace tammx


#endif  // TAMMX_TCE_H_
