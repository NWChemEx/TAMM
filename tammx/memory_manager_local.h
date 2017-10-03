#ifndef MEMORY_MANAGER_LOCAL_H_
#define MEMORY_MANAGER_LOCAL_H_

#include <iosfwd>

#include "tammx/types.h"
#include "tammx/proc_group.h"
#include "tammx/memory_manager.h"

///////////////////////////////////////////////////////////////////////////
//
//          Local memory manager and memory pool
//
///////////////////////////////////////////////////////////////////////////

namespace tammx {

class MemoryManagerLocal;

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


class MemoryManagerLocal : public MemoryManager {
 public:
  static MemoryManagerLocal* create_coll(ProcGroup pg) {
    return new MemoryManagerLocal{pg};
  }

  static void destroy_coll(MemoryManagerLocal* mms) {
    delete mms;
  }

  MemoryRegion* alloc_coll(ElementType eltype, Size nelements) override {
    MemoryRegionLocal* ret = new MemoryRegionLocal(*this);
    ret->eltype_ = eltype;
    ret->elsize_ = element_size(eltype);
    ret->local_nelements_ = nelements;
    ret->buf_ = new uint8_t[nelements.value() * ret->elsize_];
    ret->set_status(AllocationStatus::created);
    return ret;
  }

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

  void fence(MemoryRegion& mr) {
    //no-op
  }

  protected:
  explicit MemoryManagerLocal(ProcGroup pg)
      : MemoryManager{pg} {
    //sequential. So process group size should be 1
    EXPECTS(pg.is_valid());
    EXPECTS(pg_.size() == 1);
  }

  ~MemoryManagerLocal() {}

 public:
  void dealloc_coll(MemoryRegion& mpb) override {
    MemoryRegionLocal& mp = static_cast<MemoryRegionLocal&>(mpb);
    delete [] mp.buf_;
    mp.buf_ = nullptr;
  }

  void detach_coll(MemoryRegion& mpb) override {
    MemoryRegionLocal& mp = static_cast<MemoryRegionLocal&>(mpb);
    delete [] mp.buf_;
    mp.buf_ = nullptr;
  }

  const void* access(const MemoryRegion& mpb, Offset off) const override {
    const MemoryRegionLocal& mp = static_cast<const MemoryRegionLocal&>(mpb);
    return &mp.buf_[mp.elsize_ * off.value()];
  }

  void get(MemoryRegion& mpb, Proc proc, Offset off, Size nelements, void* to_buf) override {
    MemoryRegionLocal& mp = static_cast<MemoryRegionLocal&>(mpb);
    EXPECTS(proc.value() == 0);
    EXPECTS(mp.buf_ != nullptr);
    std::copy_n(mp.buf_ + mp.elsize_ * off.value(),
                mp.elsize_*nelements.value(),
                reinterpret_cast<uint8_t*>(to_buf));
  }

  void put(MemoryRegion& mpb, Proc proc, Offset off, Size nelements, const void* from_buf) override {
    MemoryRegionLocal& mp = static_cast<MemoryRegionLocal&>(mpb);
    EXPECTS(proc.value() == 0);
    EXPECTS(mp.buf_ != nullptr);
    std::copy_n(reinterpret_cast<const uint8_t*>(from_buf),
                mp.elsize_*nelements.value(),
                mp.buf_ + mp.elsize_*off.value());
  }

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

}  // namespace tammx

#endif // MEMORY_MANAGER_LOCAL_H_
