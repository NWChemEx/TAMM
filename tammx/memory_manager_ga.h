#ifndef TAMMX_MEMORY_MANAGER_GA_H_
#define TAMMX_MEMORY_MANAGER_GA_H_

#include "tammx/memory_manager.h"
#include "ga.h"
#include "mpi.h"

#include <vector>
#include <string>

//@todo Check visibility: public, private, protected

//@todo implement attach()

//@todo make MemoryManager also has alloc/dealloc semantics

namespace tammx {

class MemoryManagerGA;

class MemoryPoolGA : public MemoryPoolImpl<MemoryManagerGA> {
 public:
  MemoryPoolGA(MemoryManagerGA& mgr)
      : MemoryPoolImpl<MemoryManagerGA>(mgr) {}

  int ga() const {
    return ga_;
  }

 private:
  int ga_;
  ElementType eltype_;
  std::vector<TAMMX_SIZE> map_;

  friend class MemoryManagerGA;
}; // class MemoryPoolGA


class MemoryManagerGA : public MemoryManager {
 public:
  static MemoryManagerGA* create_coll(ProcGroup pg) {
    return new MemoryManagerGA{pg};
  }

  static void destroy_coll(MemoryManagerGA* mmga) {
    delete mmga;
  }

  MemoryRegion* alloc_coll(ElementType eltype, Size local_nelements) override {
    MemoryPoolGA* pmp = new MemoryPoolGA(*this);

    int ga_pg_default = GA_Pgroup_get_default();
    GA_Pgroup_set_default(ga_pg_);
    int nranks = pg_.size().value();
    int ga_eltype = to_ga_eltype(eltype);

    pmp->map_.resize(nranks+1);
    pmp->eltype_ = eltype;
    pmp->local_nelements_ = local_nelements;
    long long nels = local_nelements.value();

    GA_Pgroup_set_default(ga_pg_);
    long long nelements_min, nelements_max;
    MPI_Allreduce(&nels, &nelements_min, 1, MPI_LONG_LONG, MPI_MIN, pg_.comm());
    MPI_Allreduce(&nels, &nelements_max, 1, MPI_LONG_LONG, MPI_MAX, pg_.comm());
    if (nelements_min == nels && nelements_max == nels) {
      long long dim = nranks * nels, chunk = -1;
      pmp->ga_ = NGA_Create64(ga_eltype, 1, &dim, const_cast<char*>("array_name"), &chunk);
      pmp->map_[0] = 0;
      std::fill_n(pmp->map_.begin()+1, nranks-1, nels);
      std::partial_sum(pmp->map_.begin(), pmp->map_.begin()+nranks, pmp->map_.begin());
    } else {
      long long dim, block = nranks;
      MPI_Allreduce(&nels, &dim, 1, MPI_LONG_LONG, MPI_SUM, pg_.comm());
      MPI_Allgather(&nels, 1, MPI_LONG_LONG, &pmp->map_[1], 1, MPI_LONG_LONG, pg_.comm());
      pmp->map_[0] = 0; // @note this is not set by MPI_Exscan
      std::partial_sum(pmp->map_.begin(), pmp->map_.begin()+nranks, pmp->map_.begin());
      std::string array_name{"array_name"};
      for(block = nranks; block>0 && pmp->map_[block-1] == dim; --block) {
        //no-op
      }
      int64_t *map_start = &pmp->map_[0];
      for(int i=0; i<nranks && *(map_start+1)==0; ++i, ++map_start, --block) {
        //no-op
      }
      pmp->ga_ = NGA_Create_irreg64(ga_eltype, 1, &dim, const_cast<char*>(array_name.c_str()), &block, map_start);
    }
    GA_Pgroup_set_default(ga_pg_default);

    long long lo, hi, ld;
    NGA_Distribution64(pmp->ga_, pg_.rank().value(), &lo, &hi);
    EXPECTS(nels<=0 || lo == pmp->map_[pg_.rank().value()]);
    EXPECTS(nels<=0 || hi == pmp->map_[pg_.rank().value()] + nels - 1);
    pmp->set_status(AllocationStatus::created);
    return pmp;
  }

  MemoryRegion* attach_coll(MemoryRegion& mpb) override {
    MemoryPoolGA& mp = static_cast<MemoryPoolGA&>(mpb);
    MemoryPoolGA* pmp = new MemoryPoolGA(*this);

    pmp->map_ = mp.map_;
    pmp->eltype_ = mp.eltype_;
    pmp->local_nelements_ = mp.local_nelements_;
    pmp->ga_ = mp.ga_;
    pmp->set_status(AllocationStatus::attached);
    return pmp;
  }

  protected:
  explicit MemoryManagerGA(ProcGroup pg)
      : MemoryManager{pg} {
    EXPECTS(pg.is_valid());
    ga_pg_ = create_ga_process_group_coll(pg);
  }

  ~MemoryManagerGA() {
    GA_Pgroup_destroy(ga_pg_);
  }

 public:
  void dealloc_coll(MemoryRegion& mpb) override {
    MemoryPoolGA& mp = static_cast<MemoryPoolGA&>(mpb);
    NGA_Destroy(mp.ga_);
    mp.ga_ = -1;
  }

  void detach_coll(MemoryRegion& mpb) override {
    MemoryPoolGA& mp = static_cast<MemoryPoolGA&>(mpb);
    mp.ga_ = -1;
  }

  const void* access(const MemoryRegion& mpb, Offset off) const override {
    const MemoryPoolGA& mp = static_cast<const MemoryPoolGA&>(mpb);
    Proc proc{pg_.rank()};
    TAMMX_SIZE nels{1};
    TAMMX_SIZE ioffset{mp.map_[proc.value()] + off.value()};
    long long lo = ioffset, hi = ioffset + nels-1, ld = -1;
    void* buf;
    NGA_Access64(mp.ga_, &lo, &hi, reinterpret_cast<void*>(&buf), &ld);
    return buf;
  }

  void get(MemoryRegion& mpb, Proc proc, Offset off, Size nelements, void* to_buf) override {
    const MemoryPoolGA& mp = static_cast<const MemoryPoolGA&>(mpb);
    TAMMX_SIZE ioffset{mp.map_[proc.value()] + off.value()};
    long long lo = ioffset, hi = ioffset + nelements.value()-1, ld = -1;
    NGA_Get64(mp.ga_, &lo, &hi, to_buf, &ld);
  }

  void put(MemoryRegion& mpb, Proc proc, Offset off, Size nelements, const void* from_buf) override {
    const MemoryPoolGA& mp = static_cast<const MemoryPoolGA&>(mpb);

    TAMMX_SIZE ioffset{mp.map_[proc.value()] + off.value()};
    long long lo = ioffset, hi = ioffset + nelements.value()-1, ld = -1;
    NGA_Put64(mp.ga_, &lo, &hi, const_cast<void*>(from_buf), &ld);
  }

  void add(MemoryRegion& mpb, Proc proc, Offset off, Size nelements, const void* from_buf) override {
    const MemoryPoolGA& mp = static_cast<const MemoryPoolGA&>(mpb);
    TAMMX_SIZE ioffset{mp.map_[proc.value()] + off.value()};
    long long lo = ioffset, hi = ioffset + nelements.value()-1, ld = -1;
    void *alpha;
    switch(mp.eltype_) {
      case ElementType::single_precision:
        alpha = reinterpret_cast<void*>(&sp_alpha);
        break;
      case ElementType::double_precision:
        alpha = reinterpret_cast<void*>(&dp_alpha);
        break;
      case ElementType::single_complex:
        alpha = reinterpret_cast<void*>(&scp_alpha);
        break;
      case ElementType::double_complex:
        alpha = reinterpret_cast<void*>(&dcp_alpha);
        break;
      case ElementType::invalid:
      default:
        UNREACHABLE();
    }
    NGA_Acc64(mp.ga_, &lo, &hi, const_cast<void*>(from_buf), &ld, alpha);
  }

  void print_coll(const MemoryRegion& mpb, std::ostream& os) override {
    const MemoryPoolGA& mp = static_cast<const MemoryPoolGA&>(mpb);
    GA_Print(mp.ga_);
  }

 private:
  static int create_ga_process_group_coll(const ProcGroup& pg) {
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

    int ga_pg_default = GA_Pgroup_get_default();
    GA_Pgroup_set_default(GA_Pgroup_get_world());
    int ga_pg = GA_Pgroup_create(ranks, nranks);
    GA_Pgroup_set_default(ga_pg_default);
    return ga_pg;
  }

  static int to_ga_eltype(ElementType eltype) {
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
        UNREACHABLE();
    }
    return ret;
  }

  static ElementType from_ga_eltype(int eltype) {
    ElementType ret;
    switch(eltype) {
      case C_FLOAT:
        ret = ElementType::single_precision;
        break;
      case C_DBL:
        ret = ElementType::double_precision;
        break;
      case C_SCPL:
        ret = ElementType::single_complex;
        break;
      case C_DCPL:
        ret = ElementType::double_complex;
        break;
      default:
        UNREACHABLE();
    }
    return ret;
  }

  int ga_pg_;

  //constants for NGA_Acc call
  float sp_alpha = 1.0;
  double dp_alpha = 1.0;
  SingleComplex scp_alpha = {1, 0};
  DoubleComplex dcp_alpha = {1, 0};
}; // class MemoryManagerGA



// #if 0

// class MemoryManagerGA : public MemoryManager {
//  public:
//   explicit MemoryManagerGA(ProcGroup pg)
//       : MemoryManager{pg},
// /** \warning
// *  totalview LD on following statement
// *  back traced to tammx::Tensor<double>::alloc shared_ptr_base.h
// *  backtraced to ccsd_driver<double> execution_context.h
// *  back traced to main
// */
//         allocation_status_{AllocationStatus::invalid} {}

//   MemoryManagerGA(ProcGroup pg, int ga)
//       : MemoryManager(pg),
//         ga_{ga} {
//     int ndim;
//     TAMMX_SIZE dims;
//     NGA_Inquire64(ga_, &ga_eltype_, &ndim, &dims);
//     eltype_ = from_ga_eltype(ga_eltype_);
//     elsize_ = element_size(eltype_);
//     ga_pg_ = GA_Get_pgroup(ga);

//     EXPECTS(pg.is_valid());
//     auto nranks = pg.size().value();
//     auto me = pg.rank().value();
//     map_ = std::make_unique<TAMMX_SIZE[]>(nranks+1);
//     map_[0] = 0;
//     for(int i = 0; i<nranks; i++) {
//       TAMMX_SIZE lo, hi;
//       NGA_Distribution64(ga_, i, &lo, &hi);
//       map_[i+1] = map_[i] + (hi - lo + 1);
//     }
//     nelements_ = map_[me+1] - map_[me];
//     allocation_status_ = AllocationStatus::attached;
//   }

//   ~MemoryManagerGA() {
//     EXPECTS(allocation_status_ == AllocationStatus::invalid ||
//             allocation_status_ == AllocationStatus::attached);
//   }

//   ProcGroup proc_group() const {
//     return pg_;
//   }

//   void print() const override {
//     GA_Print(ga());
//   }

//   void alloc(ElementType eltype, Size nelements) override {
//     EXPECTS(allocation_status_ == AllocationStatus::invalid);
//     EXPECTS(nelements >= 0);
//     EXPECTS(eltype != ElementType::invalid);
//     eltype_ = eltype;
//     int ga_pg_default = GA_Pgroup_get_default();
//     EXPECTS(pg_.is_valid());
//     int nranks = pg_.size().value();
//     long long nels = nelements.value();

//     {
//       MPI_Group group, group_world;
//       MPI_Comm comm = pg_.comm();
//       int ranks[nranks], ranks_world[nranks];
//       MPI_Comm_group(comm, &group);

//       MPI_Comm_group(MPI_COMM_WORLD, &group_world);

//       for (int i = 0; i < nranks; i++) {
//         ranks[i] = i;
//       }
//       MPI_Group_translate_ranks(group, nranks, ranks, group_world, ranks_world);

//       GA_Pgroup_set_default(GA_Pgroup_get_world());
//       ga_pg_ = GA_Pgroup_create(ranks, nranks);
// /** \warning
// *  totalview LD on following statement
// *  back traced to tammx::Tensor<double>::alloc tensor.h
// */
//       GA_Pgroup_set_default(ga_pg_default);
//     }

//     map_ = std::make_unique<TAMMX_SIZE[]>(nranks+1);

//     ga_eltype_ = to_ga_eltype(eltype_);

//     GA_Pgroup_set_default(ga_pg_);
//     TAMMX_SIZE nelements_min, nelements_max;
//     MPI_Allreduce(&nels, &nelements_min, 1, MPI_LONG_LONG, MPI_MIN, pg_.comm());
//     MPI_Allreduce(&nels, &nelements_max, 1, MPI_LONG_LONG, MPI_MAX, pg_.comm());
//     if (nelements_min == nelements.value() && nelements_max == nelements.value()) {
//       TAMMX_SIZE dim = nranks * nels, chunk = -1;
//       ga_ = NGA_Create64(ga_eltype_, 1, &dim, const_cast<char*>("array_name"), &chunk);
//       map_[0] = 0;
//       std::fill_n(map_.get()+1, nranks-1, nels);
//       std::partial_sum(map_.get(), map_.get()+nranks, map_.get());
//     } else {
//       TAMMX_SIZE dim, block = nranks;
//       MPI_Allreduce(&nels, &dim, 1, MPI_LONG_LONG, MPI_SUM, pg_.comm());
//       MPI_Allgather(&nels, 1, MPI_LONG_LONG, &map_[1], 1, MPI_LONG_LONG, pg_.comm());
//       //MPI_Exscan(&nels, &map_[0], 1, MPI_LONG_LONG, MPI_SUM, pg_.comm());
//       map_[0] = 0; // @note this is not set by MPI_Exscan
//       std::partial_sum(map_.get(), map_.get()+nranks, map_.get());
//       std::string array_name{"array_name"};
//       for(block = nranks; block>0 && map_[block-1] == dim; --block) {
//         //no-op
//       }
//       int64_t *map_start = map_.get();
//       for(int i=0; i<nranks && *(map_start+1)==0; ++i, ++map_start, --block) {
//         //no-op
//       }
//       ga_ = NGA_Create_irreg64(ga_eltype_, 1, &dim, const_cast<char*>(array_name.c_str()), &block, map_start);
//     }
//     GA_Pgroup_set_default(ga_pg_default);

//     TAMMX_SIZE lo, hi, ld;
//     NGA_Distribution64(ga_, pg_.rank().value(), &lo, &hi);
//     EXPECTS(nels<=0 || lo == map_[pg_.rank().value()]);
//     EXPECTS(nels<=0 || hi == map_[pg_.rank().value()] + nelements.value() - 1);
//     nelements_ = hi - lo + 1;

//     allocation_status_ = AllocationStatus::created;
//   }

//   void dealloc() override {
//     EXPECTS(allocation_status_ == AllocationStatus::created);
//     NGA_Destroy(ga_);
//     NGA_Pgroup_destroy(ga_pg_);
//     allocation_status_ = AllocationStatus::invalid;
//   }

//   MemoryManager* clone(ProcGroup pg) const override {
//     return new MemoryManagerGA(pg);
//   }

//   void* access(Offset off) override {
//     EXPECTS(allocation_status_ == AllocationStatus::created ||
//             allocation_status_ == AllocationStatus::attached);
//     Proc proc{pg_.rank()};
//     TAMMX_SIZE nels{1};
//     TAMMX_SIZE ioffset{map_[proc.value()] + off.value()};
//     TAMMX_SIZE lo = ioffset, hi = ioffset + nels-1, ld = -1;
//     void* buf;
//     NGA_Access64(ga_, &lo, &hi, reinterpret_cast<void*>(&buf), &ld);
//     return buf;
//   }

//   const void* access(Offset off) const override {
//     EXPECTS(allocation_status_ == AllocationStatus::created ||
//             allocation_status_ == AllocationStatus::attached);
//     Proc proc{pg_.rank()};
//     TAMMX_SIZE nels{1};
//     TAMMX_SIZE ioffset{map_[proc.value()] + off.value()};
//     TAMMX_SIZE lo = ioffset, hi = ioffset + nels-1, ld = -1;
//     void* buf;
//     NGA_Access64(ga_, &lo, &hi, reinterpret_cast<void*>(&buf), &ld);
//     return buf;
//   }

//   void get(Proc proc, Offset off, Size nelements, void* buf) override {
//     EXPECTS(allocation_status_ == AllocationStatus::created ||
//             allocation_status_ == AllocationStatus::attached);
//     TAMMX_SIZE ioffset{map_[proc.value()] + off.value()};
//     TAMMX_SIZE lo = ioffset, hi = ioffset + nelements.value()-1, ld = -1;
//     NGA_Get64(ga_, &lo, &hi, buf, &ld);
//   }

//   void put(Proc proc, Offset off, Size nelements, const void* buf) override {
//     EXPECTS(allocation_status_ == AllocationStatus::created ||
//             allocation_status_ == AllocationStatus::attached);
//     TAMMX_SIZE ioffset{map_[proc.value()] + off.value()};
//     TAMMX_SIZE lo = ioffset, hi = ioffset + nelements.value()-1, ld = -1;
//     NGA_Put64(ga_, &lo, &hi, const_cast<void*>(buf), &ld);
//   }

//   void add(Proc proc, Offset off, Size nelements, const void* buf) override {
//     EXPECTS(allocation_status_ == AllocationStatus::created ||
//             allocation_status_ == AllocationStatus::attached);
//     TAMMX_SIZE ioffset{map_[proc.value()] + off.value()};
//     TAMMX_SIZE lo = ioffset, hi = ioffset + nelements.value()-1, ld = -1;
//     void *alpha;
//     switch(eltype_) {
//       case ElementType::single_precision:
//         alpha = reinterpret_cast<void*>(&sp_alpha);
//         break;
//       case ElementType::double_precision:
//         alpha = reinterpret_cast<void*>(&dp_alpha);
//         break;
//       case ElementType::single_complex:
//         alpha = reinterpret_cast<void*>(&scp_alpha);
//         break;
//       case ElementType::double_complex:
//         alpha = reinterpret_cast<void*>(&dcp_alpha);
//         break;
//       case ElementType::invalid:
//       default:
//         assert(0);
//     }
//     NGA_Acc64(ga_, &lo, &hi, const_cast<void*>(buf), &ld, alpha);
//   }

//   int ga() const {
//     return ga_;
//   }

//   TAMMX_SIZE *map() {
//     return map_.get();
//   }

//  protected:

//   static int to_ga_eltype(ElementType eltype) {
//     int ret;
//     switch(eltype) {
//       case ElementType::single_precision:
//         ret = C_FLOAT;
//         break;
//       case ElementType::double_precision:
//         ret = C_DBL;
//         break;
//       case ElementType::single_complex:
//         ret = C_SCPL;
//         break;
//       case ElementType::double_complex:
//         ret = C_DCPL;
//         break;
//       case ElementType::invalid:
//       default:
//         assert(0);
//     }
//     return ret;
//   }

//   static ElementType from_ga_eltype(int eltype) {
//     ElementType ret;
//     switch(eltype) {
//       case C_FLOAT:
//         ret = ElementType::single_precision;
//         break;
//       case C_DBL:
//         ret = ElementType::double_precision;
//         break;
//       case C_SCPL:
//         ret = ElementType::single_complex;
//         break;
//       case C_DCPL:
//         ret = ElementType::double_complex;
//         break;
//       default:
//         assert(0);
//     }
//     return ret;
//   }

//   int ga_;
//   int ga_pg_;
//   int ga_eltype_;
//   AllocationStatus allocation_status_;
//   size_t elsize_;
//   std::unique_ptr<TAMMX_SIZE[]> map_;
//     size_t map_size_;

//   //constants for NGA_Acc call
//   float sp_alpha = 1.0;
//   double dp_alpha = 1.0;
//   SingleComplex scp_alpha = {1, 0};
//   DoubleComplex dcp_alpha = {1, 0};
// }; // class MemoryManagerGA

// #endif

}  // namespace tammx

#endif // TAMMX_MEMORY_MANAGER_GA_H_
