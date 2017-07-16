#ifndef TAMMX_MEMORY_MANAGER_GA_H_
#define TAMMX_MEMORY_MANAGER_GA_H_

#include "tammx/memory_manager.h"
#include "ga.h"
#include "mpi.h"

#include <vector>
#include <string>

namespace tammx {

class MemoryManagerGA : public MemoryManager {
 public:
  MemoryManagerGA(ProcGroup pg)
      : MemoryManager{pg},
        created_{false} {}

  ~MemoryManagerGA() {}

  ProcGroup proc_group() const {
    return pg_;
  }

  void alloc(ElementType eltype, Size nelements) {
    Expects(nelements >= 0);
    Expects(eltype != ElementType::invalid);
    int ga_pg_default = GA_Pgroup_get_default();
    int nranks = pg_.size().value();
    long long nels = nelements.value();
    
    {
      MPI_Group group, group_world;
      MPI_Comm comm = pg_.comm();
      int ranks[nranks], ranks_world[nranks];
      MPI_Comm_group(comm, &group);

      MPI_Comm_group(MPI_COMM_WORLD, &group_world);

      for (int i = 0; i < nranks; i++) {
        ranks[i] = i;
      }
      MPI_Group_translate_ranks(group, nranks, ranks, group_world, ranks_world);

      GA_Pgroup_set_default(GA_Pgroup_get_world());
      ga_pg_ = GA_Pgroup_create(ranks, nranks);
      GA_Pgroup_set_default(ga_pg_default);
    }

    map_ = std::make_unique<int64_t[]>(nranks);

    ga_eltype_ = to_ga_type(eltype_);
    
    GA_Pgroup_set_default(ga_pg_);
    int64_t nelements_min, nelements_max;
    MPI_Allreduce(&nels, &nelements_min, 1, MPI_LONG_LONG, MPI_MIN, pg_.comm());
    MPI_Allreduce(&nels, &nelements_max, 1, MPI_LONG_LONG, MPI_MAX, pg_.comm());
    if (nelements_min == nelements.value() && nelements_max == nelements.value()) {
      int64_t dim = nranks * nels, chunk = -1;
      ga_ = NGA_Create64(ga_eltype_, 1, &dim, const_cast<char*>("array_name"), &chunk);
      std::fill_n(map_.get(), nranks, nels);
    } else {
      int64_t dim, block = nranks;
      MPI_Allreduce(&nels, &dim, 1, MPI_LONG_LONG, MPI_SUM, pg_.comm());
      MPI_Exscan(&nels, &map_[0], 1, MPI_LONG_LONG, MPI_SUM, pg_.comm());
      std::string array_name{"array_name"};
      ga_ = NGA_Create_irreg64(ga_eltype_, 1, &dim, const_cast<char*>(array_name.c_str()), &block, map_.get());
    }
    GA_Pgroup_set_default(ga_pg_default);

    int64_t lo, hi, ld;
    NGA_Distribution64(ga_, pg_.rank().value(), &lo, &hi);
    Expects(lo == map_[pg_.rank().value()]);
    Expects(hi == map_[pg_.rank().value()] + nelements.value());
    nelements_ = hi - lo;
  }

  void dealloc() {
    Expects(created_);
    NGA_Destroy(ga_);
    NGA_Pgroup_destroy(ga_pg_);
    created_ = true;
  }

  MemoryManager* clone(ProcGroup pg) const override {
    return new MemoryManagerGA(pg);
  }
    
  void* access(Offset off) {
    Proc proc{pg_.rank()};
    int64_t nels{1};
    int iproc{proc.value()};
    int64_t ioffset{map_[proc.value()] + off.value()};
    int64_t lo = ioffset, hi = ioffset + nels*elsize_, ld = -1;
    void* buf;
    NGA_Access64(ga_, &lo, &hi, reinterpret_cast<void*>(&buf), &ld);
    return buf;
  }

  void get(Proc proc, Offset off, Size nelements, void* buf) {
    int iproc{proc.value()};
    int64_t ioffset{map_[proc.value()] + off.value()};
    int64_t lo = ioffset, hi = ioffset + nelements.value()*elsize_, ld = -1;
    NGA_Get64(ga_, &lo, &hi, buf, &ld);
  }
  
  void put(Proc proc, Offset off, Size nelements, const void* buf) {
    int iproc{proc.value()};
    int64_t ioffset{map_[proc.value()] + off.value()};
    int64_t lo = ioffset, hi = ioffset + nelements.value()*elsize_, ld = -1;
    NGA_Put64(ga_, &lo, &hi, const_cast<void*>(buf), &ld);
  }
  
  void add(Proc proc, Offset off, Size nelements, const void* buf) {
    int iproc{proc.value()};
    int64_t ioffset{map_[proc.value()] + off.value()};
    int64_t lo = ioffset, hi = ioffset + nelements.value()*elsize_, ld = -1;
    void *alpha;
    
    NGA_Acc64(ga_, &lo, &hi, const_cast<void*>(buf), &ld, alpha);
  }

 protected:

  static int to_ga_type(ElementType eltype) {
    int ret;
    switch(eltype) {
      case ElementType::single_precision:
        ret = C_FLOAT;
        break;
      case ElementType::double_precision:
        ret = C_DBL;
        break;
      case ElementType::single_complex:
        ret = C_SCPL;
        break;
      case ElementType::double_complex:
        ret = C_DCPL;
        break;
      case ElementType::invalid:
      default:
        assert(0);
    }
    return ret;
  }
  
  int ga_;
  int ga_pg_;
  int ga_eltype_;
  bool created_;
  size_t elsize_;
  std::unique_ptr<int64_t[]> map_;
}; // class MemoryManagerGA

}  // namespace tammx

#endif // TAMMX_MEMORY_MANAGER_GA_H_
