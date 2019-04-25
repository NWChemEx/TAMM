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
    parallel //<Parallel distributed execution
};

/**
 * @brief Parallel execution using GA atomic counters
 * @tparam Itr Type of iterator
 * @tparam Fn Function type to be applied on each iterated element
 * @param first Begin task iterator
 * @param last End task iterator
 * @param fn Function to be applied on each iterator element
 * 
 * @todo fix scheduler hacks for parallel execution 
 */
template<typename Itr, typename Fn>
void parallel_work_ga(ExecutionContext& ec, Itr first, Itr last, Fn fn) {
    
    if(ec.ac().ac_) {
        AtomicCounter* ac = ec.ac().ac_;
        size_t idx = ec.ac().idx_;
        int64_t next = ac->fetch_add(idx, 1);
        for(int64_t count = 0; first != last; ++first, ++count) {
            if(next == count) {
                fn(*first);
                next = ac->fetch_add(idx, 1);
            }
        }
    } else {
        AtomicCounter* ac = new AtomicCounterGA(ec.pg(), 1);
        ac->allocate(0);
        int64_t next = ac->fetch_add(0, 1);
        for(int64_t count = 0; first != last; ++first, ++count) {
            if(next == count) {
                fn(*first);
                next = ac->fetch_add(0, 1);
            }
        }
        ac->deallocate();
        delete ac;
        ec.pg().barrier();
    }
        
    // ec.pg().barrier();

}

/**
 * @brief Parallel execution
 *
 * @copydetails parallel_work_ga()
 */
template<typename Itr, typename Fn>
void parallel_work(ExecutionContext& ec, Itr first, Itr last, Fn fn) {
    parallel_work_ga(ec, first, last, fn);
    // Select other types of parallel work in some way
}

/**
 * Sequential replicated execution of tasks
 * @copydetails parallel_work()
 */
template<typename Itr, typename Fn>
void seq_work(ExecutionContext& ec, Itr first, Itr last, Fn fn) {
    for(; first != last; ++first) { fn(*first); }
}

/**
 * @brief Execute iterator using the given execution policy
 * @copydetails parallel_work()
 * @param exec_policy Execution policy
 */
template<typename Itr, typename Fn>
void do_work(ExecutionContext& ec, Itr first, Itr last, Fn fn,
             const ExecutionPolicy exec_policy = ExecutionPolicy::parallel) {
    if(exec_policy == ExecutionPolicy::sequential_replicated) {
        seq_work(ec, first, last, fn);
    } else {
        parallel_work(ec, first, last, fn);
    }
}

template<typename Iterable, typename Fn>
void do_work(ExecutionContext& ec, Iterable& iterable, Fn fn,
             const ExecutionPolicy exec_policy = ExecutionPolicy::parallel) {
    do_work(ec, iterable.begin(), iterable.end(), fn, exec_policy);
}

/**
 * Iterate over all blocks in a labeled tensor and apply a given function
 * @tparam T Type of element in each tensor
 * @tparam Lambda Function to be applied on each block
 * @param ec Process group in which this call is invoked
 * @param ltc Labeled tensor whose blocks are to be iterated
 * @param func Function to be applied on each block
 * @param exec_policy Execution policy to be used
 */
template<typename T, typename Lambda>
void block_for(ExecutionContext& ec, LabeledTensor<T> ltc, Lambda func,
               ExecutionPolicy exec_policy = ExecutionPolicy::parallel) {
    LabelLoopNest loop_nest{ltc.labels()};

    if(exec_policy == ExecutionPolicy::sequential_replicated) {
        seq_work(ec, loop_nest.begin(), loop_nest.end(), func);
    } else {
        parallel_work(ec, loop_nest.begin(), loop_nest.end(), func);
    }
}

} // namespace tamm

#endif // TAMM_WORK_HPP_
