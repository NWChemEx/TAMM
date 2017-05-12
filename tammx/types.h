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
using IndexLabel = int;
using Sign = int;
//struct PermSpace;
//using Perm = StrongInt<PermSace, int>;

enum class DimType { o, v, n };

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

