#ifndef ATOMIC_COUNTER_H_
#define ATOMIC_COUNTER_H_

#include "ga-mpi.h"
#include <atomic>
#include "tammx/proc_group.h"

namespace tammx {

class AtomicCounter {
 public:
  AtomicCounter() {}

  virtual void allocate(int64_t init_val) = 0;

  virtual void deallocate() = 0;
  
  virtual int64_t fetch_add(int64_t index, int64_t sz) = 0;
  
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

class AtomicCounterGA : public AtomicCounter {
 public:
  AtomicCounterGA(const ProcGroup& pg, int64_t num_counters)
      : pg_{pg},
        allocated_{false},
        num_counters_{num_counters} { }
  
  void allocate(int64_t init_val) {
    EXPECTS(allocated_ == false);
    int64_t size = num_counters_;
    ga_pg_ = create_ga_process_group(pg_);
    char name[] = "atomic-counter";
    ga_ = NGA_Create_config64(MT_C_LONGLONG, 1, &size, name, nullptr, ga_pg_);
    EXPECTS(ga_ != 0);
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

  void deallocate() {
    EXPECTS(allocated_ == true);
    //std::cerr<<GA_Nodeid()<<" " <<__FILE__<<" "<<__LINE__<<" "<<__FUNCTION__<<"\n";
    GA_Pgroup_sync(ga_pg_);
    //std::cerr<<GA_Nodeid()<<" " <<__FILE__<<" "<<__LINE__<<" "<<__FUNCTION__<<"\n";
    GA_Destroy(ga_);
    //std::cerr<<GA_Nodeid()<<" " <<__FILE__<<" "<<__LINE__<<" "<<__FUNCTION__<<"\n";
    GA_Pgroup_destroy(ga_pg_);
    //std::cerr<<GA_Nodeid()<<" " <<__FILE__<<" "<<__LINE__<<" "<<__FUNCTION__<<"\n";
    allocated_ = false;
  }
  
  int64_t fetch_add(int64_t index, int64_t amount) {
    EXPECTS(allocated_ == true);
    //std::cerr<<GA_Nodeid()<<" " <<__FILE__<<" "<<__LINE__<<" "<<__FUNCTION__<<"\n";
    auto ret = NGA_Read_inc64(ga_, &index, amount);
    //std::cerr<<GA_Nodeid()<<" " <<__FILE__<<" "<<__LINE__<<" "<<__FUNCTION__<<"\n";
    return ret;
  }
  
  ~AtomicCounterGA() {
    EXPECTS(allocated_ == false);
  }

 private:
  int ga_;
  bool allocated_;
  int64_t num_counters_;
  ProcGroup pg_;
  int ga_pg_;

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


} // namespace tammx

#endif // ATOMIC_COUNTER_GA_H_

