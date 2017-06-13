#ifndef TAMMX_MEMORY_MANAGER_H_
#define TAMMX_MEMORY_MANAGER_H_

#include "tammx/types.h"
#include "tammx/proc-group.h"

namespace tammx {

class MemoryManager {
 public:
  MemoryManager(ProcGroup pg)
      : pg_{pg} {}

  ~MemoryManager() {}

  ProcGroup proc_group() const {
    return pg_;
  }


  virtual void alloc(ProcGroup pg, ElementType elsize, Size nbytes) = 0;
  virtual void dealloc() = 0;

  // template<typename T1>
  virtual MemoryManager* clone(ProcGroup) const = 0;
  virtual void* access(Offset off) = 0;
  virtual void get(Proc proc, Offset off, Size nelements, void* buf) = 0;
  virtual void put(Proc proc, Offset off, Size nelements, const void* buf) = 0;
  virtual void add(Proc proc, Offset off, Size nelements, const void* buf) = 0;

 protected:
  ProcGroup pg_;
}; // class MemoryManager

class MemoryManagerSequential : public MemoryManager {
 public:
  MemoryManagerSequential(ProcGroup pg = ProcGroup{})
      : MemoryManager(pg),
        buf_{nullptr},
        nelements_{0},
        elsize_{0},
        eltype_{ElementType::invalid} {
          //sequential. So process group size should be 1
    Expects(MemoryManager::pg_.size() == 1);
  }

  ~MemoryManagerSequential() {
    Expects(buf_ == nullptr);
  }

  MemoryManager* clone(ProcGroup pg) const {
    return new MemoryManagerSequential(pg);
  }

  void alloc(ProcGroup pg, ElementType eltype, Size nelements) {
    pg_ = pg;
    eltype_ = eltype;
    elsize_ = element_size(eltype);
    nelements_ = nelements;
    buf_ = new uint8_t[nelements_.value() * elsize_];
  }

  void dealloc() {
    delete [] buf_;
  }

  void* access(Offset off) {
    Expects(off < nelements_);
    return &buf_[elsize_ * off.value()];
  }

  void get(Proc proc, Offset off, Size nelements, void* to_buf) {
    Expects(buf_ != nullptr);
    Expects(off + nelements < nelements_);
    Expects(proc.value() == 0);
    std::copy_n(buf_ + elsize_*off.value(), elsize_*nelements.value(),
                reinterpret_cast<uint8_t*>(to_buf));
  }

  void put(Proc proc, Offset off, Size nelements, const void* from_buf) {
    Expects(buf_ != nullptr);
    Expects(off + nelements < nelements_);
    Expects(proc.value() == 0);
    std::copy_n(reinterpret_cast<const uint8_t*>(from_buf),
                elsize_*nelements.value(),
                buf_ + elsize_*off.value());
  }

  void add(Proc proc, Offset off, Size nelements, const void* from_buf) {
    Expects(buf_ != nullptr);
    Expects(off + nelements < nelements_);
    Expects(proc.value() == 0);
    int hi = nelements.value();
    uint8_t *to_buf = buf_ + elsize_*off.value();
    switch(eltype_) {
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
        assert(0);
    }
  }

 private:
  ElementType eltype_;
  size_t elsize_;
  Size nelements_;
  uint8_t* buf_;
}; // class MemoryManagerSequential

}  // namespace tammx

#endif // TAMMX_MEMORY_MANAGER_H_

