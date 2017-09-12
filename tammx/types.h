// Copyright 2016 Pacific Northwest National Laboratory

#ifndef TAMMX_TYPES_H_
#define TAMMX_TYPES_H_

#include <cassert>
#include <iosfwd>
#include <complex>
#include "tammx/strong_num.h"
#include "tammx/boundvec.h"
#include "tammx/errors.h"

namespace tammx {

using Fint = int64_t;

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


enum class AllocationStatus {
  invalid,
  created,
  attached
};

enum class ElementType {
  invalid,
  single_precision,
  double_precision,
  single_complex,
  double_complex
};

template<typename T>
constexpr ElementType
tensor_element_type() {
  return ElementType::invalid;
}

template<>
constexpr ElementType
tensor_element_type<double>() {
  return ElementType::double_precision;
}

template<>
constexpr ElementType
tensor_element_type<float>() {
  return ElementType::single_precision;
}

template<>
constexpr ElementType
tensor_element_type<std::complex<float>>() {
  return ElementType::single_complex;
}

template<>
constexpr ElementType
tensor_element_type<std::complex<double>>() {
  return ElementType::double_complex;
}

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


enum class DimType {
  o  = 0b0011,
  v  = 0b1100,
  oa = 0b0001,
  ob = 0b0010,
  va = 0b0100,
  vb = 0b1000,
  n  = 0b1111,
  c  = 0b0000,
  inv = 0b11111,
};

inline bool
is_dim_subset(DimType superset, DimType subset) {
  auto sup = static_cast<unsigned>(superset);
  auto sub = static_cast<unsigned>(subset);
  return (sup & sub) == sub;
}

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

inline std::ostream&
operator << (std::ostream& os, DimType dt) {
  os << to_string(dt);
  return os;
}

const TensorRank maxrank{8};

template<typename T>
using TensorVec = BoundVec<T, maxrank>;

using TensorIndex = TensorVec<BlockDim>;
using TensorDim = TensorVec<DimType>;
using TensorPerm = TensorVec<int>;

} //namespace tammx


#endif  // TAMMX_TYPES_H_
