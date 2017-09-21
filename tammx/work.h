// Copyright 2016 Pacific Northwest National Laboratory

#ifndef TAMMX_WORK_H_
#define TAMMX_WORK_H_

#include <algorithm>
#include "tammx/labeled_tensor.h"
#include "tammx/tensor.h"
#include "tammx/atomic_counter.h"

namespace tammx {

enum ExecutionPolicy {
  sequential_replicated,
  sequential_single,
  parallel
};

template<typename Itr, typename Fn>
void
parallel_work_ga(ProcGroup pg, Itr first, Itr last, Fn fn) {
  AtomicCounter *ac = new AtomicCounterGA(pg, 1);
  ac->allocate(0);
#if 1
  int64_t next = ac->fetch_add(0, 1);
  for(int64_t count=0; first != last; ++first, ++count) {
    if(next == count) {
      fn(*first);
      next = ac->fetch_add(0, 1);
    }
  }
#endif
  //GA_Sync();
  ac->deallocate();
  delete ac;
}

// template<typename Itr, typename Fn>
// void
// parallel_work(Itr first, Itr last, Fn fn) {
//   // for(; first != last; ++first) {
//   //   fn(*first);
//   // }

//   // parallel_work_omp(first, last, fn);
//   parallel_work_ga(pg_, first, last, fn);
// }

template<typename Itr, typename Fn>
void
parallel_work(const ProcGroup& pg, Itr first, Itr last, Fn fn) {
  // for(; first != last; ++first) {
  //   fn(*first);
  // }

  // parallel_work_omp(first, last, fn);
  parallel_work_ga(pg, first, last, fn);
}

template<typename Itr, typename Fn>
void
seq_work(const ProcGroup& pg, Itr first, Itr last, Fn fn) {
  //std::cerr<<pg.rank()<<" " <<__FILE__<<" "<<__LINE__<<" "<<__FUNCTION__<<"\n";
  //std::for_each(first, last, fn);
  for(; first != last; ++first) {
    //std::cerr<<pg.rank()<<" " <<__FILE__<<" "<<__LINE__<<" "<<__FUNCTION__<<"\n";    
    fn(*first);
    //std::cerr<<pg.rank()<<" " <<__FILE__<<" "<<__LINE__<<" "<<__FUNCTION__<<"\n";    
  }  
  //std::cerr<<pg.rank()<<" " <<__FILE__<<" "<<__LINE__<<" "<<__FUNCTION__<<"\n";
}

template<typename Itr, typename Fn>
void
do_work(const ProcGroup& pg, Itr first, Itr last, Fn fn, ExecutionPolicy exec_policy) {
  if(exec_policy == ExecutionPolicy::sequential_replicated) {
    seq_work(pg, first, last, fn);
  } else if(exec_policy == ExecutionPolicy::sequential_single) {
    if(pg.rank() == 0) {
      seq_work(pg, first, last, fn);
    }
  } else {
    parallel_work(pg, first, last, fn);
  }
}

template<typename T, typename Lambda>
void
block_for (const ProcGroup& ec_pg, LabeledTensor<T> ltc, Lambda func,
           ExecutionPolicy exec_policy = ExecutionPolicy::sequential_replicated) {
  Tensor<T>& tc = *ltc.tensor_;
  auto citr = loop_iterator(slice_indices(tc.tindices(), ltc.label_));
  auto lambda = [&] (const BlockDimVec& cblockid) {
    size_t dimc = tc.block_size(cblockid);
    if(tc.nonzero(cblockid) && dimc>0) {
      func(cblockid);
    }
  };
  //@assume lhs_pg is a subset of pg_, or vice-versa
  if(ec_pg.size() > ltc.tensor_->pg().size()) {
    do_work(ltc.tensor_->pg(), citr, citr.get_end(), lambda, exec_policy);
  } else {
    do_work(ec_pg, citr, citr.get_end(), lambda, exec_policy);
  }
}

template<typename T, typename Lambda>
void
block_parfor (const ProcGroup& ec_pg, LabeledTensor<T> ltc, Lambda func) {
  block_for(ec_pg, ltc, func, ExecutionPolicy::parallel);
}



#if 0
/////////////////////////////

template<typename Itr, typename Fn>
void
parallel_work_omp(Itr first, Itr last, Fn fn) {
  std::vector<decltype(*first)> tasks;
  for(; first != last; ++first) {
    tasks.push_back(*first);
  }
#pragma omp parallel for
  for(int i=0; i<tasks.size(); i++) {
    fn(tasks[i]);
  }
}

template<typename Itr, typename Fn>
void
parallel_work_ga(ProcGroup pg, Itr first, Itr last, Fn fn) {
  AtomicCounter *ac = new AtomicCounterGA(pg, 1);
  ac->allocate(0);
  int next = ac->fetch_add(0, 1);
  for(int count=0; first != last; ++first, ++count) {
    if(next == count) {
      fn(*first);
    }
  }
  ac->deallocate();
  delete ac;
}

template<typename Itr, typename Fn>
void
parallel_work(Itr first, Itr last, Fn fn) {
  for(; first != last; ++first) {
    fn(*first);
  }

  // parallel_work_omp(first, last, fn);
  //parallel_work_ga(ProcGroup{MPI_COMM_WORLD}, first, last, fn);
}

template<typename Itr, typename Fn>
void
seq_work(Itr first, Itr last, Fn fn) {
  std::for_each(first, last, fn);
}

template<typename Itr, typename Fn>
void
do_work(Itr first, Itr last, Fn fn, ExecutionPolicy exec_policy) {
  if(exec_policy == ExecutionPolicy::sequential_replicated) {
    seq_work(first, last, fn);
  } else {
    parallel_work(first, last, fn);
  }
}


template<typename T, typename Lambda>
void
block_for (LabeledTensor<T> ltc, Lambda func,
           ExecutionPolicy exec_policy = ExecutionPolicy::sequential_replicated) {
  Tensor<T>& tc = *ltc.tensor_;
  auto citr = loop_iterator(slice_indices(tc.tindices(), ltc.label_));
  auto lambda = [&] (const BlockDimVec& cblockid) {
    size_t dimc = tc.block_size(cblockid);
    if(tc.nonzero(cblockid) && dimc>0) {
      func(cblockid);
    }
  };
  do_work(citr, citr.get_end(), lambda, exec_policy);
}

template<typename T, typename Lambda>
void
block_parfor (LabeledTensor<T> ltc, Lambda func) {
  block_for(ltc, func, ExecutionPolicy::parallel);
}

template<typename T, typename Lambda>
void
tensor_map_add (LabeledTensor<T> ltc, Lambda func) {
  auto lambda = [&] (const BlockDimVec& cblockid) {
    auto cblock = ltc.tensor_->alloc(cblockid);
    cblock() = 0;
    func(cblock);
    ltc.tensor_->add(cblock.blockid(), cblock);    
  };
  block_for(ltc, lambda, ExecutionPolicy::parallel);
  // Tensor<T>& tc = *ltc.tensor_;
  // auto citr = loop_iterator(slice_indices(tc.tindices(), ltc.label_));
  // auto lambda = [&] (const BlockDimVec& cblockid) {
  //   size_t dimc = tc.block_size(cblockid);
  //   if(tc.nonzero(cblockid) && tc.spin_unique(cblockid) && dimc>0) {
  //     auto cblock = tc.alloc(cblockid);
  //     cblock() = 0;
  //     func(cblock);
  //     tc.add(cblock.blockid(), cblock);
  //   }
  // };
  // parallel_work(citr, citr.get_end(), lambda);  
}

template<typename T, typename Lambda>
void
tensor_map_put (LabeledTensor<T> ltc, Lambda func) {
  auto lambda = [&] (const BlockDimVec& cblockid) {
    auto cblock = ltc.tensor_->alloc(cblockid);
    func(cblock);
    ltc.tensor_->put(cblock.blockid(), cblock);    
  };
  block_for(ltc, lambda, ExecutionPolicy::parallel);

  // Tensor<T>& tc = *ltc.tensor_;
  // auto citr = loop_iterator(slice_indices(tc.tindices(), ltc.label_));
  // auto lambda = [&] (const BlockDimVec& cblockid) {
  //   size_t dimc = tc.block_size(cblockid);
  //   if(tc.nonzero(cblockid) && tc.spin_unique(cblockid) && dimc>0) {
  //     auto cblock = tc.alloc(cblockid);
  //     func(cblock);
  //     tc.put(cblock.blockid(), cblock);
  //   }
  // };
  // parallel_work(citr, citr.get_end(), lambda);  
}

template<typename T, typename Lambda>
void
tensor_map (LabeledTensor<T> ltc, Lambda func) {
  tensor_map_add(ltc, func);
}

class ExecutionContext;

#endif


// template<typename T, typename Lambda>
// void
// tensor_map (LabeledTensor<T> ltc, LabeledTensor<T> lta, Lambda func) {
//   Tensor<T>& tc = *ltc.tensor_;
//   Tensor<T>& ta = *lta.tensor_;
//   auto citr = loop_iterator(tc.indices());
//   auto lambda = [&] (const BlockDimVec& cblockid) {
//     size_t dimc = tc.block_size(cblockid);
//     if(tc.nonzero(cblockid) && dimc>0) {
//       auto cblock = tc.alloc(cblockid);
//       auto ablock = ta.alloc(cblockid);
//       func(cblock, ablock);
//       tc.add(cblock.blockid(), cblock);
//     }
//   };
//   parallel_work(citr, citr.get_end(), lambda);
// }

} // namespace tammx

#endif  // TAMMX_WORK_H_
