// Copyright 2016 Pacific Northwest National Laboratory

#ifndef TAMMX_WORK_H__
#define TAMMX_WORK_H__

#include <algorithm>

namespace tammx {

// @todo Parallelize
template<typename Itr, typename Fn>
void parallel_work(Itr first, Itr last, Fn fn) {
  for(; first != last; ++first) {
    fn(*first);
  }
}

template<typename Itr, typename Fn>
void seq_work(Itr first, Itr last, Fn fn) {
  std::for_each(first, last, fn);
}

}; // namespace tammx

#endif  // TAMMX_WORK_H__
