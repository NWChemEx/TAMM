#pragma once

#include "ga/ga-mpi.h"
#include "tamm/proc_group.hpp"
#include <atomic>

#ifdef USE_UPCXX
#include <upcxx/upcxx.hpp>
#endif

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
#ifdef USE_UPCXX
        counters_per_rank_((int64_t)((num_counters + pg.size().value() - 1) / pg.size().value())),
#endif
        pg_{pg} {
#ifdef USE_UPCXX
      ad_i64 = new upcxx::atomic_domain<int64_t>({upcxx::atomic_op::fetch_add}, *pg.team());
#endif
  }

  /**
   * @brief Allocate the global array of counters and initialize all of them.
   * @param init_val Value to which all counters are initialized
   * @todo Should this have an _coll suffix to denote it is a collective
   */
  void allocate(int64_t init_val) {
    EXPECTS(allocated_ == false);
    int64_t size = num_counters_;
#ifdef USE_UPCXX
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
#else
    ga_pg_ = pg_.ga_pg();
    char name[] = "atomic-counter";
    ga_ = NGA_Create_config64(MT_C_LONGLONG, 1, &size, name, nullptr, ga_pg_);
    //EXPECTS(ga_ != 0);
    if(GA_Pgroup_nodeid(ga_pg_) == 0) {
      int64_t lo[1] = {0};
      int64_t hi[1] = {num_counters_ - 1};
      int64_t ld = -1;
      long long buf[num_counters_];
      for(int i=0; i<num_counters_; i++) {
        buf[i] = init_val;
      }
      NGA_Put64(ga_, lo, hi, buf, &ld);
    }
    GA_Pgroup_sync(ga_pg_);
    allocated_ = true;
#endif
  }

  /**
   * @brief Deallocate the global array of counters.
   *
   */
  void deallocate() {
    EXPECTS(allocated_ == true);
#ifdef USE_UPCXX
    pg_.barrier();
    upcxx::delete_array(gptrs_[pg_.rank().value()]);
#else
    //std::cerr<<GA_Nodeid()<<" " <<__FILE__<<" "<<__LINE__<<" "<<__FUNCTION__<<"\n";
    GA_Pgroup_sync(ga_pg_);
    //std::cerr<<GA_Nodeid()<<" " <<__FILE__<<" "<<__LINE__<<" "<<__FUNCTION__<<"\n";
    GA_Destroy(ga_);
    //std::cerr<<GA_Nodeid()<<" " <<__FILE__<<" "<<__LINE__<<" "<<__FUNCTION__<<"\n";
    //GA_Pgroup_destroy(ga_pg_);
    //std::cerr<<GA_Nodeid()<<" " <<__FILE__<<" "<<__LINE__<<" "<<__FUNCTION__<<"\n";
#endif
    allocated_ = false;
  }

  /**
   * @copydoc AtomicCounter::fetch_and_add()
   */
  int64_t fetch_add(int64_t index, int64_t amount) {
    EXPECTS(allocated_ == true);
#ifdef USE_UPCXX
    int64_t target_rank = index / counters_per_rank_;
    int64_t offset_on_rank = index % counters_per_rank_;
    return ad_i64->fetch_add(gptrs_[target_rank] + offset_on_rank, amount,
            std::memory_order_relaxed).wait();
#else
    //std::cerr<<GA_Nodeid()<<" " <<__FILE__<<" "<<__LINE__<<" "<<__FUNCTION__<<"\n";
    auto ret = NGA_Read_inc64(ga_, &index, amount);
    //std::cerr<<GA_Nodeid()<<" " <<__FILE__<<" "<<__LINE__<<" "<<__FUNCTION__<<"\n";
    return ret;
#endif
  }

  /**
   * @copydoc AtomicCounter::~AtomicCounter()
   */
  ~AtomicCounterGA() {
    EXPECTS_NOTHROW(allocated_ == false);
#ifdef USE_UPCXX
    ad_i64->destroy();
#endif
  }

 private:
#ifdef USE_UPCXX
  std::vector<upcxx::global_ptr<int64_t>> gptrs_;
#else
  int ga_;
#endif
  bool allocated_;
  int64_t num_counters_;
#ifdef USE_UPCXX
  int64_t counters_per_rank_;
#endif
  ProcGroup pg_;
  int ga_pg_;
#ifdef USE_UPCXX
  upcxx::atomic_domain<int64_t> *ad_i64;
#endif
};

} // namespace tamm

