#pragma once

#include "ga/ga-mpi.h"
#include "tamm/proc_group.hpp"
#include <atomic>
#include <upcxx/upcxx.hpp>

namespace tamm {

/**
 * Base class for atomic counters used to parallelize iterators
 */
class AtomicCounter {
 public:
  AtomicCounter() {}

  /**
   * @brief allocate one or more atomic counters. The number is decided in the derived classes.
   * @param init_val Initial value for the atomic counter
   */
  virtual void allocate(int64_t init_val) = 0;

  /**
   * Deallocate the atomic counter
   */
  virtual void deallocate() = 0;

  /**
   * Atomically fetch and add an atomic counter
   * @param index The @p index-th counter  is to be incremented
   * @param sz The counter is incrementd by @p sz
   * @return The value of the counter before @p sz is added to it
   */
  virtual int64_t fetch_add(int64_t index, int64_t sz) = 0;

  /**
   * Destructor.
   * @pre The counter has already been deallocated.
   */
  virtual ~AtomicCounter() {}
};

// class AtomicCounterCPP {
//  public:
//   AtomicCounterCPP(const ProcGroup& pg) {}

//   void allocate(int init_val) {
//     ctr_.store(init_val);
//   }

//   void deallocate() {
//     //barrier;
//   }
  
//   int fetch_add(int sz) {
//     return ctr_.fetch_add(sz);
//   }
  
//   ~AtomicCounterCPP() {}
//  private:
//   std::atomic<int> ctr_;
// };

/**
 * @brief Atomic counter using GA.
 *
 * This is analogous to the GA-based atomic counter used in NWChem TCE.
 */
class AtomicCounterGA : public AtomicCounter {
 public:
  /**
   * @brief Construct atomic counter.
   *
   * Note that this does not allocate the counter.
   * @param pg Process group in which the atomic counter GA is created
   * @param num_counters Number of counters (i.e., size of the global array)
   */
  AtomicCounterGA(const ProcGroup& pg, int64_t num_counters)
      : allocated_{false},
        num_counters_{num_counters},
        counters_per_rank_((int64_t)((num_counters + pg.size().value() - 1) / pg.size().value())),
        pg_{pg} {
      //upcxx::persona_scope master_scope(master_mtx,
      //        upcxx::master_persona());
      ad_i64 = new upcxx::atomic_domain<int64_t>({upcxx::atomic_op::fetch_add}, *pg.team());
  }

  /**
   * @brief Allocate the global array of counters and initialize all of them.
   * @param init_val Value to which all counters are initialized
   * @todo Should this have an _coll suffix to denote it is a collective
   */
  void allocate(int64_t init_val) {
    EXPECTS(allocated_ == false);
    int64_t size = num_counters_;
    int64_t nranks = pg_.size().value();

    gptrs_.resize(nranks);

    upcxx::global_ptr<int64_t> local_gptr = upcxx::new_array<int64_t>(
            counters_per_rank_);
    assert(local_gptr);

    upcxx::dist_object<upcxx::global_ptr<int64_t>> *dobj = NULL;
    {
        //upcxx::persona_scope master_scope(master_mtx,
        //        upcxx::master_persona());
        dobj = new upcxx::dist_object<upcxx::global_ptr<int64_t>>(local_gptr, *pg_.team());
    }

    pg_.barrier();

    for (int r = 0; r < nranks; r++) {
        gptrs_[r] = dobj->fetch(r).wait();
    }

    for (int i = 0; i < counters_per_rank_; i++) {
        local_gptr.local()[i] = init_val;
    }

    allocated_ = true;
    pg_.barrier();
  }

  /**
   * @brief Deallocate the global array of counters.
   *
   */
  void deallocate() {
    EXPECTS(allocated_ == true);
    pg_.barrier();
    upcxx::delete_array(gptrs_[pg_.rank().value()]);
    allocated_ = false;
  }

  /**
   * @copydoc AtomicCounter::fetch_and_add()
   */
  int64_t fetch_add(int64_t index, int64_t amount) {
    EXPECTS(allocated_ == true);

    int64_t target_rank = index / counters_per_rank_;
    int64_t offset_on_rank = index % counters_per_rank_;
    return ad_i64->fetch_add(gptrs_[target_rank] + offset_on_rank, amount,
            std::memory_order_relaxed).wait();
  }

  /**
   * @copydoc AtomicCounter::~AtomicCounter()
   */
  ~AtomicCounterGA() {
    EXPECTS_NOTHROW(allocated_ == false);
    //upcxx::persona_scope master_scope(master_mtx,
    //        upcxx::master_persona());
    ad_i64->destroy();
  }

 private:
  std::vector<upcxx::global_ptr<int64_t>> gptrs_;
  bool allocated_;
  int64_t num_counters_;
  int64_t counters_per_rank_;
  ProcGroup pg_;
  upcxx::atomic_domain<int64_t> *ad_i64;
};

} // namespace tamm

