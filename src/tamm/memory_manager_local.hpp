#ifndef TAMM_MEMORY_MANAGER_LOCAL_H_
#define TAMM_MEMORY_MANAGER_LOCAL_H_

#include <iosfwd>

#include "tamm/types.hpp"
#include "tamm/proc_group.hpp"
#include "tamm/memory_manager.hpp"

///////////////////////////////////////////////////////////////////////////
//
//          Local memory manager and memory pool
//
///////////////////////////////////////////////////////////////////////////

namespace tamm {

class MemoryManagerLocal;

/**
 * @ingroup memory_management
 * @brief Local memory region.
 *
 * Local memory region allocates all data on a process group of size 1.
 * In particular, it allocates all its memory on the local rank. Thes can
 * be keep local copies of distributed memory regions, replicate tensors, etc.
 */
class MemoryRegionLocal : public MemoryRegionImpl<MemoryManagerLocal> {
 public:
  MemoryRegionLocal(MemoryManagerLocal& mgr)
      : MemoryRegionImpl<MemoryManagerLocal>(mgr) {}

 private:
  size_t elsize_;
  ElementType eltype_;
  uint8_t* buf_;

  friend class MemoryManagerLocal;
}; // class MemoryRegionLocal


/**
 * @ingroup memory_management
 * @brief Memory manager for local memory regions.
 */
class MemoryManagerLocal : public MemoryManager {
 public:
  /**
   * @brief Collective create a MemoryManagerLocal object.
   *
   * Note that this call is collective on a process group of rank 1, consisting of the invoking rank only.
   *
   * @param pg Process which on this the memory manager is to be created
   * @return Created memory manager
   *
   * @pre pg is a TAMM process group wrapping just MPI_COMM_SELF
   */
  static MemoryManagerLocal* create_coll(ProcGroup pg) {
    return new MemoryManagerLocal{pg};
  }

  /**
   * Collectively destroy this memory manager object
   * @param mms Memory manager object to be destroyed
   */
  static void destroy_coll(MemoryManagerLocal* mms) {
    delete mms;
  }

  /**
   * @copydoc MemoryManager::alloc_coll
   */
  MemoryRegion* alloc_coll(ElementType eltype, Size nelements) override {
    MemoryRegionLocal* ret = new MemoryRegionLocal(*this);
    ret->eltype_ = eltype;
    ret->elsize_ = element_size(eltype);
    ret->local_nelements_ = nelements;
    ret->buf_ = new uint8_t[nelements.value() * ret->elsize_];
    ret->set_status(AllocationStatus::created);
    return ret;
  }

  /**
   * @copydoc MemoryManager::attach_coll
   */
  MemoryRegion* attach_coll(MemoryRegion& mpb) override {
    MemoryRegionLocal& mp = static_cast<MemoryRegionLocal&>(mpb);
    MemoryRegionLocal* ret = new MemoryRegionLocal(*this);
    ret->eltype_ = mp.eltype_;
    ret->elsize_ = mp.elsize_;
    ret->local_nelements_ = mp.local_nelements_;
    ret->buf_ = mp.buf_;
    ret->set_status(AllocationStatus::attached);
    return ret;
  }
  

  /**
   * @copydoc MemoryManager::fence
   */
  void fence(MemoryRegion& mr) override {
    //no-op
  }

  ~MemoryManagerLocal() {}

  protected:
  explicit MemoryManagerLocal(ProcGroup pg)
      : MemoryManager{pg} {
    //sequential. So process group size should be 1
    EXPECTS(pg.is_valid());
    EXPECTS(pg_.size() == 1);
  }



public:
   /**
   * @copydoc MemoryManager::dealloc_coll
   */
  void dealloc_coll(MemoryRegion& mpb) override {
    MemoryRegionLocal& mp = static_cast<MemoryRegionLocal&>(mpb);
    delete [] mp.buf_;
    mp.buf_ = nullptr;
  }

  /**
   * @copydoc MemoryManager::detach_coll
   */
  void detach_coll(MemoryRegion& mpb) override {
    MemoryRegionLocal& mp = static_cast<MemoryRegionLocal&>(mpb);
    delete [] mp.buf_;
    mp.buf_ = nullptr;
  }

  /**
   * @copydoc MemoryManager::access
   */
  const void* access(const MemoryRegion& mpb, Offset off) const override {
    const MemoryRegionLocal& mp = static_cast<const MemoryRegionLocal&>(mpb);
    return &mp.buf_[mp.elsize_ * off.value()];
  }

  /**
   * @copydoc MemoryManager::get
   */
  void get(MemoryRegion& mpb, Proc proc, Offset off, Size nelements, void* to_buf) override {
    MemoryRegionLocal& mp = static_cast<MemoryRegionLocal&>(mpb);
    EXPECTS(proc.value() == 0);
    EXPECTS(mp.buf_ != nullptr);
    std::copy_n(mp.buf_ + mp.elsize_ * off.value(),
                mp.elsize_*nelements.value(),
                reinterpret_cast<uint8_t*>(to_buf));
  }

  /**
   * @copydoc MemoryManager::nb_get
   */
  void nb_get(MemoryRegion& mpb, Proc proc, Offset off, Size nelements, void* to_buf, DataCommunicationHandlePtr data_comm_handle) override {
    MemoryRegionLocal& mp = static_cast<MemoryRegionLocal&>(mpb);
    EXPECTS(proc.value() == 0);
    EXPECTS(mp.buf_ != nullptr);
    std::copy_n(mp.buf_ + mp.elsize_ * off.value(),
                mp.elsize_*nelements.value(),
                reinterpret_cast<uint8_t*>(to_buf));
  }

  /**
   * @copydoc MemoryManager::put
   */
  void put(MemoryRegion& mpb, Proc proc, Offset off, Size nelements, const void* from_buf) override {
    MemoryRegionLocal& mp = static_cast<MemoryRegionLocal&>(mpb);
    EXPECTS(proc.value() == 0);
    EXPECTS(mp.buf_ != nullptr);
    std::copy_n(reinterpret_cast<const uint8_t*>(from_buf),
                mp.elsize_*nelements.value(),
                mp.buf_ + mp.elsize_*off.value());
  }

  /**
   * @copydoc MemoryManager::nb_put
   */
  void nb_put(MemoryRegion& mpb, Proc proc, Offset off, Size nelements, const void* from_buf, DataCommunicationHandlePtr data_comm_handle) override {
    MemoryRegionLocal& mp = static_cast<MemoryRegionLocal&>(mpb);
    EXPECTS(proc.value() == 0);
    EXPECTS(mp.buf_ != nullptr);
    std::copy_n(reinterpret_cast<const uint8_t*>(from_buf),
                mp.elsize_*nelements.value(),
                mp.buf_ + mp.elsize_*off.value());
  }

  /**
   * @copydoc MemoryManager::add
   */
  void add(MemoryRegion& mpb, Proc proc, Offset off, Size nelements, const void* from_buf) override {
    MemoryRegionLocal& mp = static_cast<MemoryRegionLocal&>(mpb);
    EXPECTS(proc.value() == 0);
    EXPECTS(mp.buf_ != nullptr);
    int hi = nelements.value();
    uint8_t *to_buf = mp.buf_ + mp.elsize_*off.value();
    switch(mp.eltype_) {
      case ElementType::single_precision:
        for(int i=0; i<hi; i++) {
          reinterpret_cast<float*>(to_buf)[i] += reinterpret_cast<const float*>(from_buf)[i];
        }
        break;
      case ElementType::double_precision:
        for(int i=0; i<hi; i++) {
          reinterpret_cast<double*>(to_buf)[i] += reinterpret_cast<const double*>(from_buf)[i];
        }
        break;
      case ElementType::single_complex:
        for(int i=0; i<hi; i++) {
          reinterpret_cast<std::complex<float>*>(to_buf)[i] += reinterpret_cast<const std::complex<float>*>(from_buf)[i];
        }
        break;
      case ElementType::double_complex:
        for(int i=0; i<hi; i++) {
          reinterpret_cast<std::complex<double>*>(to_buf)[i] += reinterpret_cast<const std::complex<double>*>(from_buf)[i];
        }
        break;
      default:
        NOT_IMPLEMENTED();
    }
  }

  /**
   * @copydoc MemoryManager::nb_add
   */
  void nb_add(MemoryRegion& mpb, Proc proc, Offset off, Size nelements, const void* from_buf, DataCommunicationHandlePtr data_comm_handle) override {
    MemoryRegionLocal& mp = static_cast<MemoryRegionLocal&>(mpb);
    EXPECTS(proc.value() == 0);
    EXPECTS(mp.buf_ != nullptr);
    int hi = nelements.value();
    uint8_t *to_buf = mp.buf_ + mp.elsize_*off.value();
    switch(mp.eltype_) {
      case ElementType::single_precision:
        for(int i=0; i<hi; i++) {
          reinterpret_cast<float*>(to_buf)[i] += reinterpret_cast<const float*>(from_buf)[i];
        }
        break;
      case ElementType::double_precision:
        for(int i=0; i<hi; i++) {
          reinterpret_cast<double*>(to_buf)[i] += reinterpret_cast<const double*>(from_buf)[i];
        }
        break;
      case ElementType::single_complex:
        for(int i=0; i<hi; i++) {
          reinterpret_cast<std::complex<float>*>(to_buf)[i] += reinterpret_cast<const std::complex<float>*>(from_buf)[i];
        }
        break;
      case ElementType::double_complex:
        for(int i=0; i<hi; i++) {
          reinterpret_cast<std::complex<double>*>(to_buf)[i] += reinterpret_cast<const std::complex<double>*>(from_buf)[i];
        }
        break;
      default:
        NOT_IMPLEMENTED();
    }
  }

  /**
   * @copydoc MemoryManager::print_coll
   */
  void print_coll(const MemoryRegion& mpb, std::ostream& os) override {
    const MemoryRegionLocal& mp = static_cast<const MemoryRegionLocal&>(mpb);
    EXPECTS(mp.buf_ != nullptr);
    os<<"MemoryManagerLocal. contents\n";
    for(size_t i=0; i<mp.local_nelements().value(); i++) {
      switch(mp.eltype_) {
        case ElementType::double_precision:
          os<<i<<"     "<<(reinterpret_cast<const double*>(mp.buf_))[i]<<"\n";
          break;
        default:
          NOT_IMPLEMENTED();
      }
    }
    os<<"\n\n";
  }
}; // class MemoryManagerLocal

}  // namespace tamm

#endif // TAMM_MEMORY_MANAGER_LOCAL_H_
