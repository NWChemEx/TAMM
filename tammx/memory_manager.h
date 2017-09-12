#ifndef TAMMX_MEMORY_MANAGER_H_
#define TAMMX_MEMORY_MANAGER_H_

#include "tammx/types.h"
#include "tammx/proc_group.h"

namespace tammx {

class MemoryManager {
 public:
  explicit MemoryManager(ProcGroup pg)
      : pg_{pg},
        eltype_{ElementType::invalid},
        nelements_{0} {}

  virtual ~MemoryManager() {}

  ProcGroup proc_group() const {
    return pg_;
  }

  virtual void alloc(ElementType eltype, Size nelements) = 0;
  virtual void dealloc() = 0;

  Size local_size_in_elements() const {
    return nelements_;
  }

  // template<typename T1>
  virtual MemoryManager* clone(ProcGroup) const = 0;

  virtual void* access(Offset off) = 0;
  virtual const void* access(Offset off) const = 0;
  virtual void get(Proc proc, Offset off, Size nelements, void* buf) = 0;
  virtual void put(Proc proc, Offset off, Size nelements, const void* buf) = 0;
  virtual void add(Proc proc, Offset off, Size nelements, const void* buf) = 0;
  virtual void print() const = 0;
  
 protected:
  ProcGroup pg_;
  ElementType eltype_;
  Size nelements_;
}; // class MemoryManager

class MemoryManagerSequential : public MemoryManager {
 public:
  explicit MemoryManagerSequential(ProcGroup pg)
      : MemoryManager(pg),
        buf_{nullptr},
        elsize_{0},
        allocation_status_{AllocationStatus::invalid} {
          //sequential. So process group size should be 1
          EXPECTS(pg.is_valid());
          EXPECTS(MemoryManager::pg_.size() == 1);
  }

  MemoryManagerSequential(ProcGroup pg, uint8_t *buf, ElementType eltype, Size nelements)
      : MemoryManager(pg),
        buf_{buf},
        elsize_{element_size(eltype_)} {
          eltype_ = eltype;
          nelements_ = nelements;
          EXPECTS(pg.is_valid());
          EXPECTS(MemoryManager::pg_.size() == 1);
          allocation_status_ = AllocationStatus::attached;
  }

  ~MemoryManagerSequential() {
    EXPECTS(allocation_status_ == AllocationStatus::invalid ||
            allocation_status_ == AllocationStatus::attached);
  }

  MemoryManager* clone(ProcGroup pg) const {
    EXPECTS(pg.is_valid());
    return new MemoryManagerSequential(pg);
  }

  void alloc(ElementType eltype, Size nelements) {
    EXPECTS(allocation_status_ == AllocationStatus::invalid);
    eltype_ = eltype;
    elsize_ = element_size(eltype);
    nelements_ = nelements;
    buf_ = new uint8_t[nelements_.value() * elsize_];
    allocation_status_ = AllocationStatus::created;
  }

  void dealloc() {
    EXPECTS(allocation_status_ == AllocationStatus::created);
    delete [] buf_;
    buf_ = nullptr;
    allocation_status_ = AllocationStatus::invalid;
  }

  void* access(Offset off) {
    EXPECTS(allocation_status_ == AllocationStatus::created ||
            allocation_status_ == AllocationStatus::attached);
    EXPECTS(off < nelements_);
    return &buf_[elsize_ * off.value()];
  }

  const void* access(Offset off) const {
    EXPECTS(allocation_status_ == AllocationStatus::created ||
            allocation_status_ == AllocationStatus::attached);
    EXPECTS(off < nelements_);
    return &buf_[elsize_ * off.value()];
  }

  void get(Proc proc, Offset off, Size nelements, void* to_buf) {
    EXPECTS(allocation_status_ == AllocationStatus::created ||
            allocation_status_ == AllocationStatus::attached);
    EXPECTS(buf_ != nullptr);
    EXPECTS(nelements >= 0);
    EXPECTS(off + nelements <= nelements_);
    EXPECTS(proc.value() == 0);
    std::copy_n(buf_ + elsize_*off.value(), elsize_*nelements.value(),
                reinterpret_cast<uint8_t*>(to_buf));
  }

  void put(Proc proc, Offset off, Size nelements, const void* from_buf) {
    EXPECTS(allocation_status_ == AllocationStatus::created ||
            allocation_status_ == AllocationStatus::attached);
    EXPECTS(buf_ != nullptr);
    EXPECTS(nelements >= 0);
    EXPECTS(off + nelements <= nelements_);
    EXPECTS(proc.value() == 0);
    std::copy_n(reinterpret_cast<const uint8_t*>(from_buf),
                elsize_*nelements.value(),
                buf_ + elsize_*off.value());
  }

  void add(Proc proc, Offset off, Size nelements, const void* from_buf) {
    EXPECTS(allocation_status_ == AllocationStatus::created ||
            allocation_status_ == AllocationStatus::attached);
    EXPECTS(buf_ != nullptr);
    EXPECTS(nelements >= 0);
    EXPECTS(off + nelements <= nelements_);
    EXPECTS(proc.value() == 0);
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

  void print() const {
    std::cout<<"MemoryManagerSequential. contents\n";
    for(size_t i=0; i<nelements_.value(); i++) {
      switch(eltype_) {
        case ElementType::double_precision:
          std::cout<<i<<"     "<<(reinterpret_cast<const double*>(buf_))[i]<<"\n";
          break;
        default:
          assert(0); //not implemented yet
      }
    }
    std::cout<<"\n\n";
  }
  
 private:
  size_t elsize_;
  uint8_t* buf_;
  AllocationStatus allocation_status_;
}; // class MemoryManagerSequential

}  // namespace tammx


#include "tammx/memory_manager_ga.h"

#endif // TAMMX_MEMORY_MANAGER_H_

