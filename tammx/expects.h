// Copyright 2016 Pacific Northwest National Laboratory

#ifndef TAMMX_EXPECTS_H_
#define TAMMX_EXPECTS_H_

#include <cassert>

namespace tammx {

// inline void Expects(bool cond) {
//   assert(cond);
// }

#define Expects(cond) assert(cond)

}  // namespace tammx

#endif  // TAMMX_EXPECTS_H_


