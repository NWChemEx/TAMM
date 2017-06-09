// Copyright 2016 Pacific Northwest National Laboratory

#ifndef TAMMX_TYPES_H__
#define TAMMX_TYPES_H__

#include <cassert>
#include <iosfwd>
#include <complex>
#include "tammx/strong_num.h"
#include "tammx/boundvec.h"

namespace tammx {

using Fint = int64_t;

// using BlockDim = int64_t;
struct BlockDimSpace;
using BlockDim = StrongNum<BlockDimSpace, int64_t>;
using TensorRank = int;
struct IrrepSpace;
using Irrep = StrongNum<IrrepSpace, int>;
struct SpinSpace;
using Spin = StrongNum<SpinSpace, int>;
using Sign = int;
struct ProcSpace;
using Proc = StrongNum<ProcSpace, int>;
struct OffsetSpace;
using Offset = StrongNum<OffsetSpace, int64_t>;
using Size = Offset;


//struct PermSpace;
//using Perm = StrongNum<PermSace, int>;

enum class ElementType { invalid, single_precision, double_precision, single_complex, double_complex };

template<typename T>
const ElementType tensor_element_type = ElementType::invalid;

template<>
const ElementType tensor_element_type<double> = ElementType::double_precision;

template<>
const ElementType tensor_element_type<float> = ElementType::single_precision;

template<>
const ElementType tensor_element_type<std::complex<float>> = ElementType::single_complex;

template<>
const ElementType tensor_element_type<std::complex<double>> = ElementType::double_complex;

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
    case ElementType::single_complex:
      ret = sizeof(std::complex<float>);
      break;
    case ElementType::double_complex:
      ret = sizeof(std::complex<double>);
      break;
    default:
      assert(0);
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

