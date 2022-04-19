#pragma once

#include "tamm/memory_manager.hpp"
#include "armci.h"
#include "ga/ga.h"
#include "ga/ga-mpi.h"

#ifdef UPCXX_DISTARRAY
#include <upcxx-extras/dist_array.hpp>
#include <type_traits>
#endif
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
      : MemoryRegionImpl<MemoryManagerGA>(mgr) {
  }

#ifndef USE_UPCXX
    /**
     * Access the underying global arrays
     * @return Handle to underlying global array
     */
  int ga() const {
    return ga_;
  }
#endif

 private:
#ifdef USE_UPCXX
#ifdef UPCXX_DISTARRAY
  // Dist array data structures
  upcxx::extras::dist_array<float>* daf_;
  upcxx::extras::dist_array<double>* dad_;
  upcxx::extras::dist_array<SingleComplex>* dasc_;
  upcxx::extras::dist_array<DoubleComplex>* dadc_;
  std::unordered_map<upcxx::intrank_t,std::chrono::duration<double,std::micro>> lookup;
#else
  // UPC++ data structures
  upcxx::future<> fut_ = upcxx::make_future();
  upcxx::global_ptr<uint8_t>* gptrs_;
  ElementType eltype_;
  size_t eltype_size_;
#endif
#else
  // GA data structures
  int ga_;
  ElementType eltype_;
  std::vector<int64_t> map_;
#endif

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
  static MemoryManagerGA* create_coll(ProcGroup* pg) {
    return new MemoryManagerGA{pg};
  }

  /**
   * @brief Collectively destroy a GA meomry manager
   * @param mmga
   */
  static void destroy_coll(MemoryManagerGA* mmga) {
    delete mmga;
  }

  static int64_t get_element_size(ElementType t) {
    switch (t) {
      case ElementType::single_precision:
        return sizeof(float);
      case ElementType::double_precision:
        return sizeof(double);
      case ElementType::single_complex:
        return sizeof(SingleComplex);
      case ElementType::double_complex:
        return sizeof(DoubleComplex);
      case ElementType::invalid:
      default:
        fprintf(stderr, "Rank %d reached unreachable code in get_element_size\n",
                upcxx::rank_me());
        UNREACHABLE();
    }
  }

#ifdef USE_UPCXX
#ifdef UPCXX_DISTARRAY
  void alloc_coll_upcxx_dist_array(ElementType eltype,
      Size local_nelements, MemoryRegionGA* pmr, int nranks, size_t elemnt_size,
      int64_t nels) {
    upcxx::team *team = pg_->team();

    size_t loc_size;
    pmr->daf_ = nullptr;
    pmr->dad_ = nullptr;
    pmr->dasc_ = nullptr;
    pmr->dadc_ = nullptr;

    switch (eltype) {
        case ElementType::single_precision:
            pmr->daf_ = new upcxx::extras::dist_array<float>(nranks*nels,nels,*teamm,
                    upcxx::extras::dist_array<float>::params(nranks));
            loc_size = pmr->daf_->process_size(teamm->rank_me());
            memset(pmr->daf_->data(), 0x00, nels * elt_size);
            break;
        case ElementType::double_precision:
            pmr->dad_ = new upcxx::extras::dist_array<double>(nranks*nels,nels,*teamm,
                    upcxx::extras::dist_array<double>::params(nranks));
            loc_size = pmr->dad_->process_size(teamm->rank_me());
            memset(pmr->dad_->data(), 0x00, nels * elt_size);
            break;
        case ElementType::single_complex:
            pmr->dasc_ = new upcxx::extras::dist_array<SingleComplex>(nranks*nels,nels,*teamm,
                    upcxx::extras::dist_array<SingleComplex>::params(nranks));
            loc_size = pmr->dasc_->process_size(teamm->rank_me());
            memset(pmr->dasc_->data(), 0x00, nels * elt_size);
            break;
        case ElementType::double_complex:
            pmr->dadc_ = new upcxx::extras::dist_array<DoubleComplex>(nranks*nels,nels,*teamm,
                    upcxx::extras::dist_array<DoubleComplex>::params(nranks));
            loc_size = pmr->dadc_->process_size(teamm->rank_me());
            memset(pmr->dadc_->data(), 0x00, nels * elt_size);
            break;
        case ElementType::invalid:
        default:
            fprintf(stderr, "Rank %d reached unreachable code in alloc_call\n",
                upcxx::rank_me());
            UNREACHABLE();
     }

    int64_t minelts = upcxx::reduce_all(loc_size, upcxx::op_fast_min, *team).wait();
    int64_t maxelts = upcxx::reduce_all(loc_size, upcxx::op_fast_max, *team).wait();
    if (minelts != maxelts) {
        abort(); // only allocating same # of elements on each rank
    }

  }
#else // UPCXX_DISTARRAY
  void alloc_coll_upcxx(ElementType eltype,
      Size local_nelements, MemoryRegionGA* pmr, int nranks, int64_t element_size,
      int64_t nels) {
    upcxx::team *team = pg_->team();
    pmr->gptrs_ = new upcxx::global_ptr<uint8_t>[nranks];
    pmr->eltype_ = eltype;
    pmr->eltype_size_ = get_element_size(eltype);
    pmr->local_nelements_ = local_nelements;

    upcxx::dist_object<upcxx::global_ptr<uint8_t>> *dobj = NULL;
    {
        int64_t nelements_min = upcxx::reduce_all(nels, upcxx::op_fast_min, *team).wait();
        int64_t nelements_max = upcxx::reduce_all(nels, upcxx::op_fast_max, *team).wait();
        if (nelements_min != nelements_max) {
            // Max Grossman (September 2 2020): For now only support alocating same # elements on each rank
            fprintf(stderr, "ERROR: unsupported operation, allocating "
                    "different number of elements on different ranks. "
                    "nelements_min=%ld, nelements_max=%ld\n", nelements_min,
                    nelements_max);
            abort();
        }

        upcxx::global_ptr<uint8_t> local_gptr = upcxx::new_array<uint8_t>(
                nels * element_size);
        memset(local_gptr.local(), 0x00, nels * element_size);

        dobj = new upcxx::dist_object<upcxx::global_ptr<uint8_t>>(
                            local_gptr, *team);
    }

    pg_->barrier(); // ensure distributed object creation

    std::vector<upcxx::future<upcxx::global_ptr<uint8_t>>> futs(nranks);
    for (int r = 0; r < nranks; r++) {
        futs[r] = dobj->fetch(r);
    }
    for (int r = 0; r < nranks; r++) {
        pmr->gptrs_[r] = futs[r].wait();
    }
  }
#endif
#endif

  /**
   * @copydoc MemoryManager::alloc_coll
   */
  MemoryRegion* alloc_coll(ElementType eltype, Size local_nelements) override {
    MemoryRegionGA* pmr;
    {
      TimerGuard tg_total{&memTime3};
      pmr = new MemoryRegionGA(*this);

      int nranks = pg_->size().value();

      int64_t element_size = get_element_size(eltype);
      int64_t nels = local_nelements.value();
#ifdef USE_UPCXX
#ifdef UPCXX_DISTARRAY
      alloc_coll_upcxx_dist_array(eltype, local_nelements, pmr, nranks,
              elemnt_size, nels);
#else // UPCXX_DISTARRAY
      alloc_coll_upcxx(eltype, local_nelements, pmr, nranks,
              element_size, nels);
#endif // UPCXX_DISTARRAY
#else // USE_UPCXX
      int ga_pg_default = GA_Pgroup_get_default();
      GA_Pgroup_set_default(ga_pg_);
    int nranks = pg_.size().value();
    int ga_eltype = to_ga_eltype(eltype);

    pmr->map_.resize(nranks + 1);
    pmr->eltype_ = eltype;
    pmr->local_nelements_ = local_nelements;
    int64_t nels = local_nelements.value();

    int64_t nelements_min, nelements_max;

    GA_Pgroup_set_default(ga_pg_);
    {
       TimerGuard tg_total{&memTime5};
       nelements_min = pg_.allreduce(&nels, ReduceOp::min);
       nelements_max = pg_.allreduce(&nels, ReduceOp::max);
    }
    std::string array_name{"array_name"+std::to_string(++ga_counter_)};

    if (nelements_min == nels && nelements_max == nels) {
      int64_t dim = nranks * nels, chunk = -1;
      pmr->ga_ = NGA_Create64(ga_eltype, 1, &dim, const_cast<char*>(array_name.c_str()), &chunk);
      pmr->map_[0] = 0;
      std::fill_n(pmr->map_.begin()+1, nranks-1, nels);
      std::partial_sum(pmr->map_.begin(), pmr->map_.begin()+nranks, pmr->map_.begin());
    } else {
      int64_t dim, block = nranks;
      {
      TimerGuard tg_total{&memTime5};
      dim = pg_.allreduce(&nels, ReduceOp::sum);
      }
      {
      TimerGuard tg_total{&memTime6};
      pg_.allgather(&nels, &pmr->map_[1]);
      }
      pmr->map_[0] = 0; // @note this is not set by MPI_Exscan
     {
       TimerGuard tg_total{&memTime7};      
       std::partial_sum(pmr->map_.begin(), pmr->map_.begin()+nranks, pmr->map_.begin());
     }
      
      for(block = nranks; block>0 && static_cast<int64_t>(pmr->map_[block-1]) == dim; --block) {
        //no-op
      }

      int64_t* map_start = &pmr->map_[0];
      for(int i=0; i<nranks && *(map_start+1)==0; ++i, ++map_start, --block) {
        //no-op
      }
      
           {
       TimerGuard tg_total{&memTime9};
      pmr->ga_ = NGA_Create_irreg64(ga_eltype, 1, &dim, const_cast<char*>(array_name.c_str()), &block, map_start);
           }
           memTime4+=memTime9;
           memTime9=0;
      
    }

    GA_Pgroup_set_default(ga_pg_default);

    int64_t lo, hi;//, ld;
     {
       TimerGuard tg_total{&memTime8};    
       NGA_Distribution64(pmr->ga_, pg_.rank().value(), &lo, &hi);
     }
    EXPECTS(nels<=0 || lo == static_cast<int64_t>(pmr->map_[pg_.rank().value()]));
    EXPECTS(nels<=0 || hi == static_cast<int64_t>(pmr->map_[pg_.rank().value()]) + nels - 1);
#endif // USE_UPCXX

    pmr->set_status(AllocationStatus::created);
    // ensure everyone has finished with dobj
    pg_->barrier();
          }
    return pmr;
  }

  MemoryRegion* alloc_coll_balanced(ElementType eltype,
                                    Size max_nelements,
                                    ProcList proc_list = {}) override {
#ifdef USE_UPCXX
    EXPECTS(proc_list.size() == 0);
    return alloc_coll(eltype, max_nelements);
#else
    MemoryRegionGA* pmr = nullptr;
     {
       TimerGuard tg_total{&memTime3}; 
    pmr = new MemoryRegionGA{*this};
    int nranks = pg_.size().value();
    int ga_eltype = to_ga_eltype(eltype);

    pmr->map_.resize(nranks + 1);
    pmr->eltype_ = eltype;
    pmr->local_nelements_ = max_nelements;
    int64_t nels = max_nelements.value();

    std::string array_name{"array_name" + std::to_string(++ga_counter_)};

    int64_t dim = nranks * nels, chunk = nels;
    pmr->ga_ = NGA_Create_handle();
    NGA_Set_data64(pmr->ga_, 1, &dim, ga_eltype);
    GA_Set_chunk64(pmr->ga_, &chunk);
    GA_Set_pgroup(pmr->ga_, pg().ga_pg());

    if (proc_list.size() > 0 ) {
      int nproc = proc_list.size();
      int proclist_c[nproc]; 
      std::copy(proc_list.begin(), proc_list.end(), proclist_c);
      GA_Set_restricted(pmr->ga_, proclist_c, nproc);
    }

        {
    TimerGuard tg_total{&memTime9};
    NGA_Allocate(pmr->ga_);
        }
    memTime4+=memTime9;
    memTime9=0;

    pmr->map_[0] = 0;
     {
       TimerGuard tg_total{&memTime7};     
    std::fill_n(pmr->map_.begin() + 1, nranks - 1, nels);
    std::partial_sum(pmr->map_.begin(), pmr->map_.begin() + nranks,
                     pmr->map_.begin());
     }
    pmr->set_status(AllocationStatus::created);
     }
    return pmr;
#endif
  }

  /**
   * @copydoc MemoryManager::attach_coll
   */
  MemoryRegion* attach_coll(MemoryRegion& mrb) override {
    MemoryRegionGA& mr_rhs = static_cast<MemoryRegionGA&>(mrb);
    MemoryRegionGA* pmr = new MemoryRegionGA(*this);

#ifdef USE_UPCXX
#ifdef UPCXX_DISTARRAY
    pmr->lookup = mr_rhs.lookup;
    pmr->daf_ = mr_rhs.daf_;
    pmr->dad_ = mr_rhs.dad_;
    pmr->dasc_ = mr_rhs.dasc_;
    pmr->dadc_ = mr_rhs.dadc_;
#else
    pmr->gptrs_ = mr_rhs.gptrs_;
    pmr->eltype_ = mr_rhs.eltype_;
    pmr->eltype_size_ = mr_rhs.eltype_size_;
#endif
#else
    pmr->map_ = mr_rhs.map_;
    pmr->eltype_ = mr_rhs.eltype_;
    pmr->ga_ = mr_rhs.ga_;
#endif
    pmr->local_nelements_ = mr_rhs.local_nelements_;
    pmr->set_status(AllocationStatus::attached);
    pg_->barrier();
    return pmr;
  }

  /**
   * @brief Fence all operations on this memory region
   * @param mr Memory region to be fenced
   *
   * @todo Use a possibly more efficient fence
   */
  void fence(MemoryRegion& mrb) override {
#ifdef USE_UPCXX
    abort(); // assert this isn't being called
#else
    ARMCI_AllFence();
#endif
  }

  protected:
  explicit MemoryManagerGA(ProcGroup* pg
#ifdef UPCXX_DISTARRAY
          , upcxx::intrank_t hint = 0
#endif
          )
      : MemoryManager{pg, MemoryManagerKind::ga} {
    EXPECTS(pg->is_valid());
#ifdef USE_UPCXX
    team_ = pg->team();
#ifdef UPCXX_DISTARRAY
    slots = (hint ? hint : pg->size().value());
#endif
#else // USE_UPCXX
    pg_ = pg;
    ga_pg_ = pg.ga_pg();
#endif // USE_UPCXX
  }

  ~MemoryManagerGA() = default;

 public:
  /**
   * @copydoc MemoryManager::dealloc_coll
   */
  void dealloc_coll(MemoryRegion& mrb) override {
#ifdef USE_UPCXX
    upcxx::barrier(*team_);
#endif
    MemoryRegionGA& mr = static_cast<MemoryRegionGA&>(mrb);
#ifdef USE_UPCXX
#ifdef UPCXX_DISTARRAY
    if (mr.daf_) mr.daf_->destroy();
    else if (mr.dad_) mr.dad_->destroy();
    else if (mr.dasc_) mr.dasc_->destroy();
    else if (mr.dadc_) mr.dadc_->destroy();
    else {
        fprintf(stderr, "Rank %d reached unreachable code in dealloc_call\n",
                upcxx::rank_me());
        UNREACHABLE();

    }
    size_t orig_size = upcxx::reduce_one(mr.lookup.size(),upcxx::op_fast_add,0,*team_).wait();
    size_t again_size = upcxx::reduce_one(again.size(),upcxx::op_fast_add,0,*team_).wait();
    size_t near_size = upcxx::reduce_one(near.size(),upcxx::op_fast_add,0,*team_).wait();
    size_t far_size = upcxx::reduce_one(far.size(),upcxx::op_fast_add,0,*team_).wait();

    auto zero = std::chrono::duration<double,std::micro>::zero();
    for (auto& it : mr.lookup) zero += it.second;
    auto orig_avg  = zero.count();
    if (mr.lookup.size()) orig_avg /= mr.lookup.size();
    auto again_avg = std::accumulate(again.begin(),again.end(),zero).count();
    if (again.size()) again_avg /= again.size();
    auto near_avg = std::accumulate(near.begin(),near.end(),zero).count();
    if (near.size()) near_avg /= near.size();
    auto far_avg = std::accumulate(far.begin(),far.end(),zero).count();
    if (far.size()) far_avg /= far.size();

    auto orig_sum = upcxx::reduce_one(orig_avg,upcxx::op_fast_add,0,*team_).wait();
    auto again_sum = upcxx::reduce_one(again_avg,upcxx::op_fast_add,0,*team_).wait();
    auto near_sum = upcxx::reduce_one(near_avg,upcxx::op_fast_add,0,*team_).wait();
    auto far_sum = upcxx::reduce_one(far_avg,upcxx::op_fast_add,0,*team_).wait();

    if (team_->rank_me() == 0) {
        size_t nranks = team_->rank_n();
        size_t total = near_size + far_size + orig_size + again_size;
        std::cout << std::endl;
        std::cout << "MEMORY_MANAGER_GA: pure blocking" << std::endl;
        std::cout << nranks << " ranks with " << slots << " slots per cache" << std::endl;
        std::cout << orig_size << " of " << total << " lookups were for new ranks" << std::endl;
        std::cout << "new lookups were " << (100.*orig_size)/(slots*nranks) << "% of cache size" << std::endl;
        std::cout << 100.*far_size/total << "% of lookups were off-node" << std::endl;
        std::cout << "Average time to obtain pointers" << std::endl;
        if (orig_size)
            std::cout << "\tnew rank: " << orig_sum/nranks << " us" << std::endl;
        else
            std::cout << "\tno lookups of new ranks" << std::endl;
        if (again_size)
            std::cout << "\trepeat rank: " << again_sum/nranks << " us" << std::endl;
        else
            std::cout << "\tno lookups of previous ranks" << std::endl;
        if (near_size)
            std::cout << "\tlocal rank: " << near_sum/nranks << " us" << std::endl;
        else
            std::cout << "\tno lookups of local ranks" << std::endl;
        if (far_size)
            std::cout << "\tremote rank: " << far_sum/nranks << " us" << std::endl;
        else
            std::cout << "\tno lookups of remote ranks" << std::endl;
        std::cout << std::endl;
    }
    again.clear(); near.clear(); far.clear(); mr.lookup.clear();
    pg_->barrier();
#else
    upcxx::delete_array(mr.gptrs_[pg_->rank().value()]);
    mr.gptrs_ = NULL;
#endif
    upcxx::barrier(*team_);
#else // USE_UPCXX
    NGA_Destroy(mr.ga_);
    mr.ga_ = -1;
#endif // USE_UPCXX
  }

  /**
   * @copydoc MemoryManager::detach_coll
   */
  void detach_coll(MemoryRegion& mrb) override {
#ifdef USE_UPCXX
    upcxx::barrier(*team_);
#endif
    MemoryRegionGA& mr = static_cast<MemoryRegionGA&>(mrb);
#ifdef USE_UPCXX
#ifdef UPCXX_DISTARRAY
    if (mr.daf_) mr.daf_ = nullptr;
    else if (mr.dad_) mr.dad_ = nullptr;
    else if (mr.dasc_) mr.dasc_ = nullptr;
    else if (mr.dadc_) mr.dadc_ = nullptr;
    else {
        fprintf(stderr, "Rank %d reached unreachable code in detach_call\n",
                upcxx::rank_me());
        UNREACHABLE();
    }
#else
    mr.gptrs_ = NULL;
#endif
    upcxx::barrier(*team_);
#else // USE_UPCXX
    mr.ga_ = -1;
#endif // USE_UPCXX
  }

  /**
   * @copydoc MemoryManager::access
   */
  const void* access(const MemoryRegion& mrb, Offset off) const override {
    const MemoryRegionGA& mr = static_cast<const MemoryRegionGA&>(mrb);
#ifdef USE_UPCXX
#ifdef UPCXX_DISTARRAY
    if (mr.daf_)
        return static_cast<void*>(mr.daf_->data() + off.value());
    else if (mr.dad_)
        return static_cast<void*>(mr.dad_->data() + off.value());
    else if (mr.dasc_)
        return static_cast<void*>(mr.dasc_->data() + off.value());
    else if (mr.dadc_)
        return static_cast<void*>(mr.dadc_->data() + off.value());
    else {
        fprintf(stderr, "Rank %d reached unreachable code in access\n",
                upcxx::rank_me());
        UNREACHABLE();
    }
#else
    size_t element_size = mr.eltype_size_;
    size_t local_byte_offset = off.value() * element_size;
    upcxx::global_ptr<uint8_t> local_arr = mr.gptrs_[pg_->rank().value()];
    return static_cast<void*>(local_arr.local() + local_byte_offset);
#endif
#else
    Proc proc{pg_.rank()};
    TAMM_SIZE nels{1};
    TAMM_SIZE ioffset{mr.map_[proc.value()] + off.value()};
    int64_t lo = ioffset, hi = ioffset + nels-1, ld = -1;
    void* buf;
    NGA_Access64(mr.ga_, &lo, &hi, reinterpret_cast<void*>(&buf), &ld);
    return buf;
#endif
  }

  /**
   * @copydoc MemoryManager::get
   */
  void get(MemoryRegion& mrb, Proc proc, Offset off, Size nelements,
          void* to_buf) override {
    const MemoryRegionGA& mr = static_cast<const MemoryRegionGA&>(mrb);
#ifdef USE_UPCXX
#ifdef UPCXX_DISTARRAY
    upcxx::future<> f;
    std::chrono::high_resolution_clock::time_point start, end;
    if (mr.daf_) {
        size_t idx = mr.daf_->process_size() * proc.value() + off.value(); // assumes equal allocation across ranks
        start = std::chrono::high_resolution_clock::now();
        upcxx::global_ptr<float> gptr = mr.daf_->ptr(idx).wait();
        end = std::chrono::high_resolution_clock::now();
        f = upcxx::rget(gptr, (float*)to_buf, nelements.value());
    } else if (mr.dad_) {
        size_t idx = mr.dad_->process_size() * proc.value() + off.value(); // assumes equal allocation across ranks
        start = std::chrono::high_resolution_clock::now();
        upcxx::global_ptr<double> gptr = mr.dad_->ptr(idx).wait();
        end = std::chrono::high_resolution_clock::now();
        f = upcxx::rget(gptr, (double*)to_buf, nelements.value());
    } else if (mr.dasc_) {
        size_t idx = mr.dasc_->process_size() * proc.value() + off.value(); // assumes equal allocation across ranks
        start = std::chrono::high_resolution_clock::now();
        upcxx::global_ptr<SingleComplex> gptr = mr.dasc_->ptr(idx).wait();
        end = std::chrono::high_resolution_clock::now();
        f = upcxx::rget(gptr, (SingleComplex*)to_buf, nelements.value());
    } else if (mr.dadc_) {
        size_t idx = mr.dadc_->process_size() * proc.value() + off.value(); // assumes equal allocation across ranks
        start = std::chrono::high_resolution_clock::now();
        upcxx::global_ptr<DoubleComplex> gptr = mr.dadc_->ptr(idx).wait();
        end = std::chrono::high_resolution_clock::now();
        f = upcxx::rget(gptr, (DoubleComplex*)to_buf, nelements.value());
    } else {
        fprintf(stderr, "Rank %d reached unreachable code in get\n",
                upcxx::rank_me());
        UNREACHABLE();
    }
    auto elapsed = std::chrono::duration_cast<std::chrono::duration<double,std::micro>>(end-start);
    if (mr.lookup.find(proc.value()) == mr.lookup.end()) 
        mr.lookup.insert({proc.value(),elapsed});
    else
        again.push_back(elapsed);
    if (upcxx::local_team_contains((*team_)[proc.value()]))
        near.push_back(elapsed);
    else
        far.push_back(elapsed);
#else
    upcxx::future<> f = upcxx::rget(
            mr.gptrs_[proc.value()] + off.value() * mr.eltype_size_,
            (uint8_t *)to_buf, nelements.value() * mr.eltype_size_);
#endif
    f.wait();
#else
    TAMM_SIZE ioffset{mr.map_[proc.value()] + off.value()};
    int64_t lo = ioffset, hi = ioffset + nelements.value()-1, ld = -1;
    NGA_Get64(mr.ga_, &lo, &hi, to_buf, &ld);
#endif
  }

    /**
   * @copydoc MemoryManager::nb_get
   */
  void nb_get(MemoryRegion& mrb, Proc proc, Offset off, Size nelements,
          void* to_buf, DataCommunicationHandlePtr data_comm_handle) override {
    MemoryRegionGA& mr = static_cast<MemoryRegionGA&>(mrb);
#ifdef USE_UPCXX
#ifdef UPCXX_DISTARRAY
    upcxx::future<> f;
    std::chrono::high_resolution_clock::time_point start, end;
    if (mr.daf_) {
        size_t idx = mr.daf_->process_size() * proc.value() + off.value(); // assumes equal allocation across ranks
        start = std::chrono::high_resolution_clock::now();
        upcxx::global_ptr<float> gptr = mr.daf_->ptr(idx).wait();
        end = std::chrono::high_resolution_clock::now();
        f = upcxx::rget(gptr, (float*)to_buf, nelements.value());
    } else if (mr.dad_) {
        size_t idx = mr.dad_->process_size() * proc.value() + off.value(); // assumes equal allocation across ranks
        start = std::chrono::high_resolution_clock::now();
        upcxx::global_ptr<double> gptr = mr.dad_->ptr(idx).wait();
        end = std::chrono::high_resolution_clock::now();
        f = upcxx::rget(gptr, (double*)to_buf, nelements.value());
    } else if (mr.dasc_) {
        size_t idx = mr.dasc_->process_size() * proc.value() + off.value(); // assumes equal allocation across ranks
        start = std::chrono::high_resolution_clock::now();
        upcxx::global_ptr<SingleComplex> gptr = mr.dasc_->ptr(idx).wait();
        end = std::chrono::high_resolution_clock::now();
        f = upcxx::rget(gptr, (SingleComplex*)to_buf, nelements.value());
    } else if (mr.dadc_) {
        size_t idx = mr.dadc_->process_size() * proc.value() + off.value(); // assumes equal allocation across ranks
        start = std::chrono::high_resolution_clock::now();
        upcxx::global_ptr<DoubleComplex> gptr = mr.dadc_->ptr(idx).wait();
        end = std::chrono::high_resolution_clock::now();
        f = upcxx::rget(gptr, (DoubleComplex*)to_buf, nelements.value());
    } else {
        fprintf(stderr, "Rank %d reached unreachable code in nb_get\n",
                upcxx::rank_me());
        UNREACHABLE();
    }
    auto elapsed = std::chrono::duration_cast<std::chrono::duration<double,std::micro>>(end-start);
    if (mr.lookup.find(proc.value()) == mr.lookup.end()) 
        mr.lookup.insert({proc.value(),elapsed});
    else
        again.push_back(elapsed);
    if (upcxx::local_team_contains((*team_)[proc.value()]))
        near.push_back(elapsed);
    else
        far.push_back(elapsed);
#else
    upcxx::future<> f = upcxx::rget(
            mr.gptrs_[proc.value()] + off.value() * mr.eltype_size_,
            (uint8_t *)to_buf, nelements.value() * mr.eltype_size_);
#endif
    mr.fut_ = upcxx::when_all(mr.fut_, f);
#else
    TAMM_SIZE ioffset{mr.map_[proc.value()] + off.value()};
    int64_t lo = ioffset, hi = ioffset + nelements.value()-1, ld = -1;
#endif

    data_comm_handle->resetCompletionStatus();
#ifdef USE_UPCXX
    data_comm_handle->data_handle_ = f;
#else
     NGA_NbGet64(mr.ga_, &lo, &hi, to_buf, &ld, data_comm_handle->getDataHandlePtr());
#endif
  }

  /**
   * @copydoc MemoryManager::put
   */
  void put(MemoryRegion& mrb, Proc proc, Offset off, Size nelements,
          const void* from_buf) override {
    const MemoryRegionGA& mr = static_cast<const MemoryRegionGA&>(mrb);

#ifdef USE_UPCXX
#ifdef UPCXX_DISTARRAY
    std::chrono::high_resolution_clock::time_point start, end;
    if (mr.daf_) {
        size_t idx = mr.daf_->process_size() * proc.value() + off.value(); // assumes equal allocation across ranks
        start = std::chrono::high_resolution_clock::now();
        upcxx::global_ptr<float> gptr = mr.daf_->ptr(idx).wait();
        end = std::chrono::high_resolution_clock::now();
        upcxx::rput((float*)from_buf, gptr, nelements.value()).wait();
    } else if (mr.dad_) {
        size_t idx = mr.dad_->process_size() * proc.value() + off.value(); // assumes equal allocation across ranks
        start = std::chrono::high_resolution_clock::now();
        upcxx::global_ptr<double> gptr = mr.dad_->ptr(idx).wait();
        end = std::chrono::high_resolution_clock::now();
        upcxx::rput((double*)from_buf, gptr, nelements.value()).wait();
    } else if (mr.dasc_) {
        size_t idx = mr.dasc_->process_size() * proc.value() + off.value(); // assumes equal allocation across ranks
        start = std::chrono::high_resolution_clock::now();
        upcxx::global_ptr<SingleComplex> gptr = mr.dasc_->ptr(idx).wait();
        end = std::chrono::high_resolution_clock::now();
        upcxx::rput((SingleComplex*)from_buf, gptr, nelements.value()).wait();
    } else if (mr.dadc_) {
        size_t idx = mr.dadc_->process_size() * proc.value() + off.value(); // assumes equal allocation across ranks
        start = std::chrono::high_resolution_clock::now();
        upcxx::global_ptr<DoubleComplex> gptr = mr.dadc_->ptr(idx).wait();
        end = std::chrono::high_resolution_clock::now();
        upcxx::rput((DoubleComplex*)from_buf, gptr, nelements.value()).wait();
    } else {
        fprintf(stderr, "Rank %d reached unreachable code in put\n",
                upcxx::rank_me());
        UNREACHABLE();
    }

    auto elapsed = std::chrono::duration_cast<std::chrono::duration<double,std::micro>>(end-start);
    if (mr.lookup.find(proc.value()) == mr.lookup.end()) 
        mr.lookup.insert({proc.value(),elapsed});
    else
        again.push_back(elapsed);
    if (upcxx::local_team_contains((*team_)[proc.value()]))
        near.push_back(elapsed);
    else
        far.push_back(elapsed);

#else // UPCXX_DISTARRAY
    upcxx::rput((uint8_t*)from_buf,
            mr.gptrs_[proc.value()] + off.value() * mr.eltype_size_,
            nelements.value() * mr.eltype_size_).wait();
#endif // UPCXX_DISTARRAY
#else
    TAMM_SIZE ioffset{mr.map_[proc.value()] + off.value()};
    int64_t lo = ioffset, hi = ioffset + nelements.value()-1, ld = -1;
    NGA_Put64(mr.ga_, &lo, &hi, const_cast<void*>(from_buf), &ld);
#endif
  }

  void nb_put(MemoryRegion& mrb, Proc proc, Offset off, Size nelements,
          const void* from_buf,
          DataCommunicationHandlePtr data_comm_handle) override {
    MemoryRegionGA& mr = static_cast<MemoryRegionGA&>(mrb);

#ifdef USE_UPCXX
#ifdef UPCXX_DISTARRAY
    std::chrono::high_resolution_clock::time_point start, end;
    upcxx::future<> f;
    if (mr.daf_) {
        size_t idx = mr.daf_->process_size() * proc.value() + off.value(); // assumes equal allocation across ranks
        start = std::chrono::high_resolution_clock::now();
        upcxx::global_ptr<float> gptr = mr.daf_->ptr(idx).wait();
        end = std::chrono::high_resolution_clock::now();
        f = upcxx::rput((float*)from_buf, gptr, nelements.value());
    } else if (mr.dad_) {
        size_t idx = mr.dad_->process_size() * proc.value() + off.value(); // assumes equal allocation across ranks
        start = std::chrono::high_resolution_clock::now();
        upcxx::global_ptr<double> gptr = mr.dad_->ptr(idx).wait();
        end = std::chrono::high_resolution_clock::now();
        f = upcxx::rput((double*)from_buf, gptr, nelements.value());
    } else if (mr.dasc_) {
        size_t idx = mr.dasc_->process_size() * proc.value() + off.value(); // assumes equal allocation across ranks
        start = std::chrono::high_resolution_clock::now();
        upcxx::global_ptr<SingleComplex> gptr = mr.dasc_->ptr(idx).wait();
        end = std::chrono::high_resolution_clock::now();
        f = upcxx::rput((SingleComplex*)from_buf, gptr, nelements.value());
    } else if (mr.dadc_) {
        size_t idx = mr.dadc_->process_size() * proc.value() + off.value(); // assumes equal allocation across ranks
        start = std::chrono::high_resolution_clock::now();
        upcxx::global_ptr<DoubleComplex> gptr = mr.dadc_->ptr(idx).wait();
        end = std::chrono::high_resolution_clock::now();
        f = upcxx::rput((DoubleComplex*)from_buf, gptr, nelements.value());
    } else {
        fprintf(stderr, "Rank %d reached unreachable code in nb_put\n",
                upcxx::rank_me());
        UNREACHABLE();
    }
    auto elapsed = std::chrono::duration_cast<std::chrono::duration<double,std::micro>>(end-start);
    if (mr.lookup.find(proc.value()) == mr.lookup.end()) 
        mr.lookup.insert({proc.value(),elapsed});
    else
        again.push_back(elapsed);
    if (upcxx::local_team_contains((*team_)[proc.value()]))
        near.push_back(elapsed);
    else
        far.push_back(elapsed);
#else
    upcxx::future<> f = upcxx::rput((uint8_t*)from_buf,
            mr.gptrs_[proc.value()] + off.value() * mr.eltype_size_,
            nelements.value() * mr.eltype_size_);
#endif
    mr.fut_ = upcxx::when_all(mr.fut_, f);
#else
    TAMM_SIZE ioffset{mr.map_[proc.value()] + off.value()};
    int64_t lo = ioffset, hi = ioffset + nelements.value()-1, ld = -1;
#endif

    data_comm_handle->resetCompletionStatus();
#ifdef USE_UPCXX
    data_comm_handle->data_handle_ = f;
#else
    NGA_NbPut64(mr.ga_, &lo, &hi, const_cast<void*>(from_buf), &ld, data_comm_handle->getDataHandlePtr());
#endif
  }

#ifdef USE_UPCXX
  void add_helper(MemoryRegion& mrb, Proc proc, Offset off,
          Size nelements, const void* from_buf) {
    const MemoryRegionGA& mr = static_cast<const MemoryRegionGA&>(mrb);

#ifdef UPCXX_DISTARRAY
    std::chrono::high_resolution_clock::time_point start, end;
    if (mr.daf_) {
        size_t idx = mr.daf_->process_size() * proc.value() + off.value(); // assumes equal allocation across ranks
        start = std::chrono::high_resolution_clock::now();
        upcxx::global_ptr<float> gptr = mr.daf_->ptr(idx).wait();
        end = std::chrono::high_resolution_clock::now();
        upcxx::view<float> buf_view = upcxx::make_view((float*)from_buf, (float*)from_buf + nelements.value());
        upcxx::rpc_ff(*team_, proc.value(),
            [] (const upcxx::global_ptr<float>& dst_buf, const upcxx::view<float>& src_buf,
                upcxx::dist_object<int64_t>& executed_ops) {
                float *dst = dst_buf.local();
                float const * const src = src_buf.begin();
                int lim = src_buf.size();
                int nthreads = (lim >= 1000000 ? 2 : 1);
// #pragma omp parallel for schedule(static) firstprivate(dst, src, lim) num_threads(nthreads)
                for (int i = 0; i < lim; i++)
                    dst[i] += src[i];
                *executed_ops += 1;
            }, gptr, buf_view, *(pg_->get_recvd_ops_object()));
    } else if (mr.dad_) {
        size_t idx = mr.dad_->process_size() * proc.value() + off.value(); // assumes equal allocation across ranks
        start = std::chrono::high_resolution_clock::now();
        upcxx::global_ptr<double> gptr = mr.dad_->ptr(idx).wait();
        end = std::chrono::high_resolution_clock::now();
        upcxx::view<double> buf_view = upcxx::make_view((double*)from_buf, (double*)from_buf + nelements.value());
        upcxx::rpc_ff(*team_, proc.value(),
            [] (const upcxx::global_ptr<double>& dst_buf, const upcxx::view<double>& src_buf,
                upcxx::dist_object<int64_t>& executed_ops) {
                double *dst = dst_buf.local();
                double const * const src = src_buf.begin();
                int lim = src_buf.size();
                int nthreads = (lim >= 1000000 ? 2 : 1);
// #pragma omp parallel for schedule(static) firstprivate(dst, src, lim) num_threads(nthreads)
                for (int i = 0; i < lim; i++)
                    dst[i] += src[i];
                *executed_ops += 1;
            }, gptr, buf_view, *(pg_->get_recvd_ops_object()));
    } else if (mr.dasc_) {
        size_t idx = mr.dasc_->process_size() * proc.value() + off.value(); // assumes equal allocation across ranks
        start = std::chrono::high_resolution_clock::now();
        upcxx::global_ptr<SingleComplex> gptr = mr.dasc_->ptr(idx).wait();
        end = std::chrono::high_resolution_clock::now();
        upcxx::view<SingleComplex> buf_view = 
            upcxx::make_view((SingleComplex*)from_buf, (SingleComplex*)from_buf + nelements.value());
        upcxx::rpc_ff(*team_, proc.value(),
            [] (const upcxx::global_ptr<SingleComplex>& dst_buf, const upcxx::view<SingleComplex>& src_buf,
                upcxx::dist_object<int64_t>& executed_ops) {
                SingleComplex *dst = dst_buf.local();
                SingleComplex const * const src = src_buf.begin();
                int lim = src_buf.size();
                int nthreads = (lim >= 1000000 ? 2 : 1);
// #pragma omp parallel for schedule(static) firstprivate(dst, src, lim) num_threads(nthreads)
                for (int i = 0; i < lim; i++) {
                    dst[i].real += src[i].real;
                    dst[i].imag += src[i].imag;
                }
                *executed_ops += 1;
            }, gptr, buf_view, *(pg_->get_recvd_ops_object()));
    } else if (mr.dadc_) {
        size_t idx = mr.dadc_->process_size() * proc.value() + off.value(); // assumes equal allocation across ranks
        start = std::chrono::high_resolution_clock::now();
        upcxx::global_ptr<DoubleComplex> gptr = mr.dadc_->ptr(idx).wait();
        end = std::chrono::high_resolution_clock::now();
        upcxx::view<DoubleComplex> buf_view = 
            upcxx::make_view((DoubleComplex*)from_buf, (DoubleComplex*)from_buf + nelements.value());
        upcxx::rpc_ff(*team_, proc.value(),
            [] (const upcxx::global_ptr<DoubleComplex>& dst_buf, const upcxx::view<DoubleComplex>& src_buf,
                upcxx::dist_object<int64_t>& executed_ops) {
                DoubleComplex *dst = dst_buf.local();
                DoubleComplex const * const src = src_buf.begin();
                int lim = src_buf.size();
                int nthreads = (lim >= 1000000 ? 2 : 1);
// #pragma omp parallel for schedule(static) firstprivate(dst, src, lim) num_threads(nthreads)
                for (int i = 0; i < lim; i++) {
                    dst[i].real += src[i].real;
                    dst[i].imag += src[i].imag;
                }
                *executed_ops += 1;
            }, gptr, buf_view, *(pg_->get_recvd_ops_object()));
    } else {
        fprintf(stderr, "Rank %d reached unreachable code in nb_put\n",
                upcxx::rank_me());
        UNREACHABLE();
    }
    auto elapsed = std::chrono::duration_cast<std::chrono::duration<double,std::micro>>(end-start);
    if (mr.lookup.find(proc.value()) == mr.lookup.end()) 
        mr.lookup.insert({proc.value(),elapsed});
    else
        again.push_back(elapsed);
    if (upcxx::local_team_contains((*team_)[proc.value()]))
        near.push_back(elapsed);
    else
        far.push_back(elapsed);

#else // UPCXX_DISTARRAY
    switch (mr.eltype_) {
        case ElementType::single_precision: {
            upcxx::global_ptr<float> typed_dst = upcxx::reinterpret_pointer_cast<float>(mr.gptrs_[proc.value()]);
            upcxx::rpc_ff(*team_, proc.value(),
                    [] (const upcxx::global_ptr<float>& dst_buf, const upcxx::view<float>& src_buf, upcxx::dist_object<int64_t>& executed_ops) {
                        float *dst = dst_buf.local();
                        float const * const src = src_buf.begin();
                        int lim = src_buf.size();
                        int nthreads = (lim >= 1000000 ? 2 : 1);
#pragma omp parallel for schedule(static) firstprivate(dst, src, lim) num_threads(nthreads)
                        for (int i = 0; i < lim; i++) {
                            dst[i] += src[i];
                        }
                        *executed_ops += 1;
                    }, typed_dst + off.value(), upcxx::make_view((float*)from_buf, (float*)from_buf + nelements.value()), *(pg_->get_recvd_ops_object()));

            break;
        }
        case ElementType::double_precision: {
            upcxx::global_ptr<double> typed_dst = upcxx::reinterpret_pointer_cast<double>(mr.gptrs_[proc.value()]);
            upcxx::rpc_ff(*team_, proc.value(),
                    [] (const upcxx::global_ptr<double>& dst_buf, const upcxx::view<double>& src_buf, upcxx::dist_object<int64_t>& executed_ops) {
                        double * const dst = dst_buf.local();
                        double const * const src = src_buf.begin();
                        int lim = src_buf.size();
                        int nthreads = (lim >= 1000000 ? 2 : 1);
#pragma omp parallel for schedule(static) firstprivate(dst, src, lim) num_threads(nthreads)
                        for (int i = 0; i < lim; i++) {
                            dst[i] += src[i];
                        }
                        *executed_ops += 1;
                    }, typed_dst + off.value(), upcxx::make_view((double*)from_buf, (double*)from_buf + nelements.value()), *(pg_->get_recvd_ops_object()));
            break;
        }
        case ElementType::single_complex: {
            upcxx::global_ptr<SingleComplex> typed_dst = upcxx::reinterpret_pointer_cast<SingleComplex>(mr.gptrs_[proc.value()]);
            upcxx::rpc_ff(*team_, proc.value(),
                    [] (const upcxx::global_ptr<SingleComplex>& dst_buf, const upcxx::view<SingleComplex>& src_buf, upcxx::dist_object<int64_t>& executed_ops) {
                        SingleComplex *dst = dst_buf.local();
                        SingleComplex const * const src = src_buf.begin();
                        int lim = src_buf.size();
                        int nthreads = (lim >= 1000000 ? 2 : 1);
#pragma omp parallel for schedule(static) firstprivate(dst, src, lim) num_threads(nthreads)
                        for (int i = 0; i < lim; i++) {
                            dst[i].real += src[i].real;
                            dst[i].imag += src[i].imag;
                        }
                        *executed_ops += 1;
                    }, typed_dst + off.value(), upcxx::make_view((SingleComplex*)from_buf, (SingleComplex*)from_buf + nelements.value()), *(pg_->get_recvd_ops_object()));

            break;
        }
        case ElementType::double_complex: {
            upcxx::global_ptr<DoubleComplex> typed_dst = upcxx::reinterpret_pointer_cast<DoubleComplex>(mr.gptrs_[proc.value()]);
            upcxx::rpc_ff(*team_, proc.value(),
                    [] (const upcxx::global_ptr<DoubleComplex>& dst_buf, const upcxx::view<DoubleComplex>& src_buf, upcxx::dist_object<int64_t>& executed_ops) {
                        DoubleComplex *dst = dst_buf.local();
                        DoubleComplex const * const src = src_buf.begin();
                        int lim = src_buf.size();
                        int nthreads = (lim >= 1000000 ? 2 : 1);
#pragma omp parallel for schedule(static) firstprivate(dst, src, lim) num_threads(nthreads)
                        for (int i = 0; i < lim; i++) {
                            dst[i].real += src[i].real;
                            dst[i].imag += src[i].imag;
                        }
                        *executed_ops += 1;
                    }, typed_dst + off.value(), upcxx::make_view((DoubleComplex*)from_buf, (DoubleComplex*)from_buf + nelements.value()), *(pg_->get_recvd_ops_object()));
            break;
        }
        case ElementType::invalid:
        default:
          UNREACHABLE();
    }
#endif
  }
#endif // USE_UPCXX

  /**
   * @copydoc MemoryManager::add
   */
  void add(MemoryRegion& mrb, Proc proc, Offset off, Size nelements,
          const void* from_buf) override {
    const MemoryRegionGA& mr = static_cast<const MemoryRegionGA&>(mrb);
#ifdef USE_UPCXX
    add_helper(mrb, proc, off, nelements, from_buf);
    pg_->add_op(proc.value());
#else
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
#endif
  }

  /**
   * @copydoc MemoryManager::nb_add
   */
  void nb_add(MemoryRegion& mrb, Proc proc, Offset off, Size nelements,
          const void* from_buf,
          DataCommunicationHandlePtr data_comm_handle) override {
#ifdef USE_UPCXX
    abort(); // verify this API isn't being used.
#else
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
#endif
  }
  
  /**
   * @copydoc MemoryManager::print_coll
   */
  void print_coll(const MemoryRegion& mrb, std::ostream& os) override {
    const MemoryRegionGA& mr = static_cast<const MemoryRegionGA&>(mrb);
#ifndef USE_UPCXX
    GA_Print(mr.ga_);
#endif
  }

 private:
#ifdef USE_UPCXX
  upcxx::team* team_;
#ifdef UPCXX_DISTARRAY
  size_t slots;
  std::vector<std::chrono::duration<double,std::micro>> near, far, again;
#endif
#else
  ProcGroup pg_;       /**< Underlying ProcGroup */
  int ga_pg_;          /**< GA pgroup underlying pg_ */
  int ga_counter_ = 0; /**< GA counter to name GAs in create call */

  // constants for NGA_Acc call
  float sp_alpha = 1.0;
  double dp_alpha = 1.0;
  SingleComplex scp_alpha = {1, 0};
  DoubleComplex dcp_alpha = {1, 0};
#endif
  friend class ExecutionContext;
};  // class MemoryManagerGA

}  // namespace tamm
