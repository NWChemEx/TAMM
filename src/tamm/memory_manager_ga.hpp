#ifndef TAMM_MEMORY_MANAGER_GA_H_
#define TAMM_MEMORY_MANAGER_GA_H_

#include "tamm/memory_manager.hpp"
#include "armci.h"
#include "ga.h"
#include "mpi.h"

#include <vector>
#include <string>
#include <numeric>

//@todo Check visibility: public, private, protected

//@todo implement attach()

//@todo make MemoryManager also has alloc/dealloc semantics

namespace tamm {

class MemoryManagerGA;

/**
 * @ingroup memory_management
 * @brief Memory region that allocates memory using Global Arrays
 */
class MemoryRegionGA : public MemoryRegionImpl<MemoryManagerGA> {
 public:
  MemoryRegionGA(MemoryManagerGA& mgr)
      : MemoryRegionImpl<MemoryManagerGA>(mgr) {}

  /**
   * Access the underying global arrays
   * @return Handle to underlying global array
   */
  int ga() const {
    return ga_;
  }

 private:
  int ga_;
  ElementType eltype_;
  std::vector<TAMM_SIZE> map_;

  friend class MemoryManagerGA;
}; // class MemoryRegionGA


/**
 * @ingroup memory_management
 * @brief Memory manager that wraps Glocal Arrays API
 */
class MemoryManagerGA : public MemoryManager {
 public:
  /**
   * @brief Collectively create a MemoryManagerGA.
   *
   * Creation of a GA memory manager involves creation of a GA process group.
   * To make this explicit, the call is static with an _coll suffix.
   * @param pg Process group in which to create GA memory manager
   * @return Constructed GA memory manager
   */
  static MemoryManagerGA* create_coll(ProcGroup pg) {
    return new MemoryManagerGA{pg};
  }

  /**
   * @brief Collectively destroy a GA meomry manager
   * @param mmga
   */
  static void destroy_coll(MemoryManagerGA* mmga) {
    delete mmga;
  }

  /**
   * @copydoc MemoryManager::attach_coll
   */
  MemoryRegion* alloc_coll(ElementType eltype, Size local_nelements) override {
    MemoryRegionGA* pmr = new MemoryRegionGA(*this);

    int ga_pg_default = GA_Pgroup_get_default();
    GA_Pgroup_set_default(ga_pg_);
    int nranks = pg_.size().value();
    int ga_eltype = to_ga_eltype(eltype);

    pmr->map_.resize(nranks+1);
    pmr->eltype_ = eltype;
    pmr->local_nelements_ = local_nelements;
    int64_t nels = local_nelements.value();

    GA_Pgroup_set_default(ga_pg_);
    int64_t nelements_min, nelements_max;
    MPI_Allreduce(&nels, &nelements_min, 1, MPI_LONG_LONG, MPI_MIN, pg_.comm());
    MPI_Allreduce(&nels, &nelements_max, 1, MPI_LONG_LONG, MPI_MAX, pg_.comm());
    if (nelements_min == nels && nelements_max == nels) {
      int64_t dim = nranks * nels, chunk = -1;
      pmr->ga_ = NGA_Create64(ga_eltype, 1, &dim, const_cast<char*>("array_name"), &chunk);
      pmr->map_[0] = 0;
      std::fill_n(pmr->map_.begin()+1, nranks-1, nels);
      std::partial_sum(pmr->map_.begin(), pmr->map_.begin()+nranks, pmr->map_.begin());
    } else {
      int64_t dim, block = nranks;
      MPI_Allreduce(&nels, &dim, 1, MPI_LONG_LONG, MPI_SUM, pg_.comm());
      MPI_Allgather(&nels, 1, MPI_LONG_LONG, &pmr->map_[1], 1, MPI_LONG_LONG, pg_.comm());
      pmr->map_[0] = 0; // @note this is not set by MPI_Exscan
      std::partial_sum(pmr->map_.begin(), pmr->map_.begin()+nranks, pmr->map_.begin());
      std::string array_name{"array_name"};
      for(block = nranks; block>0 && pmr->map_[block-1] == dim; --block) {
        //no-op
      }

      //uint64_t *map_start = &pmr->map_[0];
      int64_t map_start = pmr->map_[0];
      
      
      for(int i=0; i<nranks && *(&map_start+1)==0; ++i, ++map_start, --block) {
        //no-op
      }
      pmr->ga_ = NGA_Create_irreg64(ga_eltype, 1, &dim, const_cast<char*>(array_name.c_str()), &block, &map_start);
    }
    GA_Pgroup_set_default(ga_pg_default);

    int64_t lo, hi;//, ld;
    NGA_Distribution64(pmr->ga_, pg_.rank().value(), &lo, &hi);
    EXPECTS(nels<=0 || lo == pmr->map_[pg_.rank().value()]);
    EXPECTS(nels<=0 || hi == pmr->map_[pg_.rank().value()] + nels - 1);
    pmr->set_status(AllocationStatus::created);
    return pmr;
  }

  /**
   * @copydoc MemoryManager::attach_coll
   */
  MemoryRegion* attach_coll(MemoryRegion& mrb) override {
    MemoryRegionGA& mr_rhs = static_cast<MemoryRegionGA&>(mrb);
    MemoryRegionGA* pmr = new MemoryRegionGA(*this);

    pmr->map_ = mr_rhs.map_;
    pmr->eltype_ = mr_rhs.eltype_;
    pmr->local_nelements_ = mr_rhs.local_nelements_;
    pmr->ga_ = mr_rhs.ga_;
    pmr->set_status(AllocationStatus::attached);
    return pmr;
  }

  /**
   * @brief Fence all operations on this memory region
   * @param mr Memory region to be fenced
   *
   * @todo Use a possibly more efficient fence
   */
  void fence(MemoryRegion& mr) {
    ARMCI_AllFence();
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
  /**
   * @copydoc MemoryManager::dealloc_coll
   */
  void dealloc_coll(MemoryRegion& mrb) override {
    MemoryRegionGA& mr = static_cast<MemoryRegionGA&>(mrb);
    NGA_Destroy(mr.ga_);
    mr.ga_ = -1;
  }

  /**
   * @copydoc MemoryManager::detach_coll
   */
  void detach_coll(MemoryRegion& mrb) override {
    MemoryRegionGA& mr = static_cast<MemoryRegionGA&>(mrb);
    mr.ga_ = -1;
  }

  /**
   * @copydoc MemoryManager::access
   */
  const void* access(const MemoryRegion& mrb, Offset off) const override {
    const MemoryRegionGA& mr = static_cast<const MemoryRegionGA&>(mrb);
    Proc proc{pg_.rank()};
    TAMM_SIZE nels{1};
    TAMM_SIZE ioffset{mr.map_[proc.value()] + off.value()};
    int64_t lo = ioffset, hi = ioffset + nels-1, ld = -1;
    void* buf;
    NGA_Access64(mr.ga_, &lo, &hi, reinterpret_cast<void*>(&buf), &ld);
    return buf;
  }

  /**
   * @copydoc MemoryManager::get
   */
  void get(MemoryRegion& mrb, Proc proc, Offset off, Size nelements, void* to_buf) override {
    const MemoryRegionGA& mr = static_cast<const MemoryRegionGA&>(mrb);
    TAMM_SIZE ioffset{mr.map_[proc.value()] + off.value()};
    int64_t lo = ioffset, hi = ioffset + nelements.value()-1, ld = -1;
    NGA_Get64(mr.ga_, &lo, &hi, to_buf, &ld);
  }

  /**
   * @copydoc MemoryManager::put
   */
  void put(MemoryRegion& mrb, Proc proc, Offset off, Size nelements, const void* from_buf) override {
    const MemoryRegionGA& mr = static_cast<const MemoryRegionGA&>(mrb);

    TAMM_SIZE ioffset{mr.map_[proc.value()] + off.value()};
    int64_t lo = ioffset, hi = ioffset + nelements.value()-1, ld = -1;
    NGA_Put64(mr.ga_, &lo, &hi, const_cast<void*>(from_buf), &ld);
  }

  /**
   * @copydoc MemoryManager::add
   */
  void add(MemoryRegion& mrb, Proc proc, Offset off, Size nelements, const void* from_buf) override {
    const MemoryRegionGA& mr = static_cast<const MemoryRegionGA&>(mrb);
    TAMM_SIZE ioffset{mr.map_[proc.value()] + off.value()};
    int64_t lo = ioffset, hi = ioffset + nelements.value()-1, ld = -1;
    void *alpha;
    switch(mr.eltype_) {
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
    NGA_Acc64(mr.ga_, &lo, &hi, const_cast<void*>(from_buf), &ld, alpha);
  }

  /**
   * @copydoc MemoryManager::print_coll
   */
  void print_coll(const MemoryRegion& mrb, std::ostream& os) override {
    const MemoryRegionGA& mr = static_cast<const MemoryRegionGA&>(mrb);
    GA_Print(mr.ga_);
  }

 private:
  /**
   * Create a GA process group corresponding to the given proc group
   * @param pg TAMM process group
   * @return GA processes group on this TAMM process group
   */
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

  /**
   * Convert a TAMM element type to a GA element type
   * @param eltype TAMM element type
   * @return Corresponding GA element type
   */
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

  /**
   * Convert a GA element type to a TAMM element type
   * @param eltype GA element type
   * @return Corresponding TAMM element type
   */
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

}  // namespace tamm

#endif // TAMM_MEMORY_MANAGER_GA_H_
