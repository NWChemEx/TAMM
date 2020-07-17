#ifndef ATOMIC_COUNTER_HPP_
#define ATOMIC_COUNTER_HPP_

#include "ga-mpi.h"
#include "tamm/proc_group.hpp"
#include <atomic>

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
        pg_{pg} { }

  /**
   * @brief Allocate the global array of counters and initialize all of them.
   * @param init_val Value to which all counters are initialized
   * @todo Should this have an _coll suffix to denote it is a collective
   */
  void allocate(int64_t init_val) {
    EXPECTS(allocated_ == false);
    int64_t size = num_counters_;
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
  }

  /**
   * @brief Deallocate the global array of counters.
   *
   */
  void deallocate() {
    EXPECTS(allocated_ == true);
    //std::cerr<<GA_Nodeid()<<" " <<__FILE__<<" "<<__LINE__<<" "<<__FUNCTION__<<"\n";
    GA_Pgroup_sync(ga_pg_);
    //std::cerr<<GA_Nodeid()<<" " <<__FILE__<<" "<<__LINE__<<" "<<__FUNCTION__<<"\n";
    GA_Destroy(ga_);
    //std::cerr<<GA_Nodeid()<<" " <<__FILE__<<" "<<__LINE__<<" "<<__FUNCTION__<<"\n";
    //GA_Pgroup_destroy(ga_pg_);
    //std::cerr<<GA_Nodeid()<<" " <<__FILE__<<" "<<__LINE__<<" "<<__FUNCTION__<<"\n";
    allocated_ = false;
  }

  /**
   * @copydoc AtomicCounter::fetch_and_add()
   */
  int64_t fetch_add(int64_t index, int64_t amount) {
    EXPECTS(allocated_ == true);
    //std::cerr<<GA_Nodeid()<<" " <<__FILE__<<" "<<__LINE__<<" "<<__FUNCTION__<<"\n";
    auto ret = NGA_Read_inc64(ga_, &index, amount);
    //std::cerr<<GA_Nodeid()<<" " <<__FILE__<<" "<<__LINE__<<" "<<__FUNCTION__<<"\n";
    return ret;
  }

  /**
   * @copydoc AtomicCounter::~AtomicCounter()
   */
  ~AtomicCounterGA() {
    EXPECTS_NOTHROW(allocated_ == false);
  }

 private:
  int ga_;
  bool allocated_;
  int64_t num_counters_;
  ProcGroup pg_;
  int ga_pg_;

  /**
   * @brief Create a GA process group from a wrapped MPI communicator
   * @param pg Wrapped MPI communicator
   * @return GA process group on the MPI communicator
   * @note Collective on the current default GA process group
   */
  static int create_ga_process_group(const ProcGroup& pg) {
    MPI_Group group, group_default;
    MPI_Comm comm = pg.comm();
    int nranks = pg.size().value();
    int ranks[nranks], ranks_default[nranks];
    MPI_Comm_group(comm, &group);
  
    MPI_Comm_group(GA_MPI_Comm_pgroup_default(), &group_default);

    for (int i = 0; i < nranks; i++) {
      ranks[i] = i;
    }
    MPI_Group_translate_ranks(group, nranks, ranks, group_default, ranks_default);
    return GA_Pgroup_create(ranks_default, nranks);
  }


};


} // namespace tamm

#endif // ATOMIC_COUNTER_HPP_

