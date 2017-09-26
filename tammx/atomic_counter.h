#ifndef ATOMIC_COUNTER_H_
#define ATOMIC_COUNTER_H_

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
    //std::cerr<<GA_Nodeid()<<" " <<__FILE__<<" "<<__LINE__<<" "<<__FUNCTION__<<"\n";
    int64_t size = num_counters_;
    //std::cerr<<GA_Nodeid()<<" " <<__FILE__<<" "<<__LINE__<<" "<<__FUNCTION__<<"\n";
    ga_pg_ = create_ga_process_group(pg_);
    //std::cerr<<GA_Nodeid()<<" " <<__FILE__<<" "<<__LINE__<<" "<<__FUNCTION__<<"\n";
    int ga_pg_default = GA_Pgroup_get_default();
    //std::cerr<<GA_Nodeid()<<" " <<__FILE__<<" "<<__LINE__<<" "<<__FUNCTION__<<"\n";
    GA_Pgroup_set_default(ga_pg_);
    //std::cerr<<GA_Nodeid()<<" " <<__FILE__<<" "<<__LINE__<<" "<<__FUNCTION__<<"\n";
    char name[] = "atomic-counter";
    //std::cerr<<GA_Nodeid()<<" " <<__FILE__<<" "<<__LINE__<<" "<<__FUNCTION__<<"\n";
    ga_ = NGA_Create64(MT_C_LONGLONG, 1, &size, name, nullptr);
    //std::cerr<<GA_Nodeid()<<" " <<__FILE__<<" "<<__LINE__<<" "<<__FUNCTION__<<"\n";
    EXPECTS(ga_ != 0);
    if(GA_Pgroup_nodeid(ga_pg_) == 0) {
      int64_t lo[1] = {0};
      int64_t hi[1] = {num_counters_ - 1};
      int64_t ld = -1;
      long long buf[num_counters_];
      for(int i=0; i<num_counters_; i++) {
        buf[i] = init_val;
      }
      //std::cerr<<GA_Nodeid()<<" " <<__FILE__<<" "<<__LINE__<<" "<<__FUNCTION__<<"\n";
      NGA_Put64(ga_, lo, hi, buf, &ld);
      //std::cerr<<GA_Nodeid()<<" " <<__FILE__<<" "<<__LINE__<<" "<<__FUNCTION__<<"\n";
    }
    //std::cerr<<GA_Nodeid()<<" " <<__FILE__<<" "<<__LINE__<<" "<<__FUNCTION__<<"\n";
    GA_Pgroup_sync(ga_pg_);
    GA_Pgroup_set_default(ga_pg_default);
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
    MPI_Group group, group_world;
    MPI_Comm comm = pg.comm();
    int nranks = pg.size().value();
    int ranks[nranks], ranks_world[nranks];
    MPI_Comm_group(comm, &group);
  
    MPI_Comm_group(MPI_COMM_WORLD, &group_world);
  
    for (int i = 0; i < nranks; i++) {
      ranks[i] = i;
    }
    MPI_Group_translate_ranks(group, nranks, ranks, group_world, ranks_world);
  
    GA_Pgroup_set_default(GA_Pgroup_get_world());
    return GA_Pgroup_create(ranks, nranks);
  }


};


} // namespace tammx

#endif // ATOMIC_COUNTER_GA_H_

