#ifndef TAMMX_TENSOR_DIMS_H_
#define TAMMX_TENSOR_DIMS_H_

#include "tammx/types.h"

namespace tammx {
using IndexInfo = std::pair<TensorVec<TensorSymmGroup>,int>;

namespace tensor_dims {

inline TensorVec<TensorSymmGroup>
operator - (const TensorVec<TensorSymmGroup>& tv1,
            const TensorVec<TensorSymmGroup>& tv2) {
  TensorVec<TensorSymmGroup> ret{tv1};
  ret.insert_back(tv2.begin(), tv2.end());
  return ret;
}

inline IndexInfo
operator | (const TensorVec<TensorSymmGroup>& tv1,
            const TensorVec<TensorSymmGroup>& tv2) {
  TensorVec<TensorSymmGroup> ret;
  if(tv1.size() > 0) {
    ret.insert_back(tv1.begin(), tv1.end());
  }
  if(tv2.size() > 0) {
    ret.insert_back(tv2.begin(), tv2.end());
  }
  int sz=0;
  for(auto &sg: tv1) {
    sz += sg.size();
  }
  return {ret, sz};
}

const auto E  = TensorVec<TensorSymmGroup>{};
const auto O  = TensorVec<TensorSymmGroup>{TensorSymmGroup{DimType::o}};
const auto V  = TensorVec<TensorSymmGroup>{TensorSymmGroup{DimType::v}};
const auto N  = TensorVec<TensorSymmGroup>{TensorSymmGroup{DimType::n}};
const auto OO = TensorVec<TensorSymmGroup>{TensorSymmGroup{DimType::o, 2}};
const auto OV = O-V;
const auto VO = V-O;
const auto VV = TensorVec<TensorSymmGroup>{TensorSymmGroup{DimType::v, 2}};
const auto NN = TensorVec<TensorSymmGroup>{TensorSymmGroup{DimType::n, 2}};

} // namespace tensor_dims
} // namespace tammx

#endif // TAMMX_TENSOR_DIMS_H_
