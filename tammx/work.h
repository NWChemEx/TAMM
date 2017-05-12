// Copyright 2016 Pacific Northwest National Laboratory

#ifndef TAMMX_WORK_H__
#define TAMMX_WORK_H__

namespace tammx {

// @todo Parallelize
template<typename Itr, typename Fn>
void  parallel_work(Itr first, Itr last, Fn fn) {
  for(; first != last; ++first) {
    fn(*first);
  }
}

}; // namespace tammx

#endif  // TAMMX_WORK_H__
