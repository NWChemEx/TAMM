// Copyright 2018 Pacific Northwest National Laboratory

#ifndef TAMM_WORK_HPP_
#define TAMM_WORK_HPP_

#include "tamm/atomic_counter.h"

namespace tamm {

/**
 * Execution policy
 */
enum class ExecutionPolicy {
  sequential_replicated, //<Sequential execution on all ranks
  // sequential_single,     //<Sequential execution on one rank
  parallel               //<Parallel distributed execution
};

/**
 * @brief Parallel execution using GA atomic counters
 * @tparam Itr Type of iterator
 * @tparam Fn Function type to be applied on each iterated element
 * @param first Begin task iterator
 * @param last End task iterator
 Loser* @param fn Function to be applied on each iterator element
 */
template<typename Itr, typename Fn>
void
parallel_work_ga(ProcGroup& ec_pg, Itr first, Itr last, Fn fn) {
  AtomicCounter *ac = new AtomicCounterGA(ec_pg, 1);
  ac->allocate(0);
  int64_t next = ac->fetch_add(0, 1);
  for(int64_t count=0; first != last; ++first, ++count) {
    if(next == count) {
      fn(*first);
      next = ac->fetch_add(0, 1);
    }
  }
  ec_pg.barrier();
  ac->deallocate();
  delete ac;
}

/**
 * @brief Parallel execution
 *
 * @copydetails parallel_work_ga()
 */
template<typename Itr, typename Fn>
void
parallel_work(ProcGroup& ec_pg, Itr first, Itr last, Fn fn) {
  parallel_work_ga(ec_pg, first, last, fn);
  // Select other types of parallel work in some way
}

/**
 * Sequential replicated execution of tasks
 * @copydetails parallel_work()
 */
template<typename Itr, typename Fn>
void
seq_work(ProcGroup& ec_pg, Itr first, Itr last, Fn fn) {
  //std::cerr<<pg.rank()<<" " <<__FILE__<<" "<<__LINE__<<" "<<__FUNCTION__<<"\n";
  //std::for_each(first, last, fn);
  for(; first != last; ++first) {
    //std::cerr<<pg.rank()<<" " <<__FILE__<<" "<<__LINE__<<" "<<__FUNCTION__<<"\n";    
    fn(*first);
    //std::cerr<<pg.rank()<<" " <<__FILE__<<" "<<__LINE__<<" "<<__FUNCTION__<<"\n";    
  }  
  //std::cerr<<pg.rank()<<" " <<__FILE__<<" "<<__LINE__<<" "<<__FUNCTION__<<"\n";
}

/**
 * @brief Execute iterator using the given execution policy
 * @copydetails parallel_work()
 * @param exec_policy Execution policy
 */
template<typename Itr, typename Fn>
void
do_work(ProcGroup& ec_pg, Itr first, Itr last, Fn fn, const ExecutionPolicy exec_policy = ExecutionPolicy::parallel) {
  if(exec_policy == ExecutionPolicy::sequential_replicated) {
    seq_work(ec_pg, first, last, fn);
  } else {
    parallel_work(ec_pg, first, last, fn);
  }
}

template<typename Iterable, typename Fn>
void
do_work(ProcGroup& ec_pg, Iterable& iterable, Fn fn, const ExecutionPolicy exec_policy = ExecutionPolicy::parallel) {
  do_work(ec_pg, iterable.begin(), iterable.end(), fn, exec_policy);
}

/**
 * Iterate over all blocks in a labeled tensor and apply a given function
 * @tparam T Type of element in each tensor
 * @tparam Lambda Function to be applied on each block
 * @param ec_pg Process group in which this call is invoked
 * @param ltc Labeled tensor whose blocks are to be iterated
 * @param func Function to be applied on each block
 * @param exec_policy Execution policy to be used
 */
template<typename T, typename Lambda>
void
block_for(ProcGroup ec_pg, LabeledTensor<T> ltc, Lambda func,
           ExecutionPolicy exec_policy = ExecutionPolicy::parallel) {
  LabelLoopNest loop_nest{ltc.labels()};

  if(exec_policy == ExecutionPolicy::sequential_replicated) {
    seq_work(ec_pg, loop_nest.begin(), loop_nest.end(), func);
  } else {
    parallel_work(ec_pg, loop_nest.begin(), loop_nest.end(), func);
  }
}

} // namespace tamm

#endif  // TAMM_WORK_HPP_