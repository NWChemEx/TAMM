// Copyright 2016 Pacific Northwest National Laboratory

#ifndef TAMMX_TYPES_H__
#define TAMMX_TYPES_H__

#include <cassert>
#include <iosfwd>
#include "tammx/strong_int.h"
#include "tammx/boundvec.h"

namespace tammx {

using Fint = int64_t;

// using BlockDim = int64_t;
struct BlockDimSpace;
using BlockDim = StrongInt<BlockDimSpace, int64_t>;
using TensorRank = int;
struct IrrepSpace;
using Irrep = StrongInt<IrrepSpace, int>;
struct SpinSpace;
using Spin = StrongInt<SpinSpace, int>;
using Sign = int;
//struct PermSpace;
//using Perm = StrongInt<PermSace, int>;

enum class ElementType { invalid, single_precision, double_precision };

template<typename T>
const ElementType element_type = ElementType::invalid;

template<>
const ElementType element_type<double> = ElementType::double_precision;

template<>
const ElementType element_type<float> = ElementType::single_precision;

static inline constexpr size_t
element_size(ElementType eltype) {
  size_t ret = 0;
  switch(eltype) {
    case ElementType::single_precision:
      ret = sizeof(float);
      break;
    case ElementType::double_precision:
      ret = sizeof(double);
      break;
  }
  return ret;
}


enum class DimType { o  = 0b0011,
                     v  = 0b1100,
                     oa = 0b0001,
                     ob = 0b0010,
                     va = 0b0100,
                     vb = 0b1000,
                     n  = 0b1111
};

inline bool
is_dim_subset(DimType superset, DimType subset) {
  auto sup = static_cast<unsigned>(superset);
  auto sub = static_cast<unsigned>(subset);
  return (sup & sub) == sub;
}

struct IndexLabel {
  int label;
  DimType dt;

  IndexLabel() = default;
  IndexLabel(int lbl, DimType dtype)
      : label{lbl},
        dt{dtype} {}
};

inline std::string
to_string(DimType dt) {
  switch(dt) {
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
    default:
      assert(0);
  }
}

inline bool
operator == (const IndexLabel lhs, const IndexLabel rhs) {
  return lhs.label == rhs.label
      && lhs.dt == rhs.dt;
}

inline bool
operator != (const IndexLabel lhs, const IndexLabel rhs) {
  return !(lhs == rhs);
}

inline bool
operator < (const IndexLabel lhs, const IndexLabel rhs) {
  return (lhs.label < rhs.label)
      || (lhs.label == rhs.label && lhs.dt < rhs.dt);
}

inline std::ostream&
operator << (std::ostream& os, DimType dt) {
  switch(dt) {
    case DimType::o:
      os<<"DimType::o";
      break;
    case DimType::v:
      os<<"DimType::v";
      break;
    case DimType::n:
      os<<"DimType::n";
      break;
    default:
      assert(0);
  }
  return os;
}

inline std::ostream&
operator << (std::ostream& os, IndexLabel il) {
  os<<"il["<<il.label<<","<<il.dt<<"]";
  return os;
}

const TensorRank maxrank{8};

template<typename T>
using TensorVec = BoundVec<T, maxrank>;

using TensorIndex = TensorVec<BlockDim>;
using TensorLabel = TensorVec<IndexLabel>;
using SymmGroup = TensorVec<DimType>;
using TensorDim = TensorVec<DimType>;
using TensorPerm = TensorVec<int>;

}; //namespace tammx


#endif  // TAMMX_TYPES_H__

