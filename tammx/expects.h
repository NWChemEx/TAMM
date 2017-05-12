// Copyright 2016 Pacific Northwest National Laboratory

#ifndef TAMMX_EXPECTS_H__
#define TAMMX_EXPECTS_H__

#include <cassert>

namespace tammx {

// inline void Expects(bool cond) {
//   assert(cond);
// }

#define Expects(cond) assert(cond)

}  // namespace tammx

#endif  // TAMMX_EXPECTS_H__


