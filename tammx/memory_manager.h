#ifndef TAMMX_MEMORY_MANAGER_H_
#define TAMMX_MEMORY_MANAGER_H_

#include <iosfwd>

#include "tammx/types.h"
#include "tammx/proc_group.h"

//@todo Check that offset+size for remote gets and puts is within the
//remote size

namespace tammx {

enum class MemoryManagerType { local, distributed };

class MemoryPoolBase;

class MemoryManager {
 public:
  virtual MemoryPoolBase* alloc_coll(ElementType eltype, Size nelements) = 0;
  virtual MemoryPoolBase* attach_coll(MemoryPoolBase& mpb) = 0;

  ProcGroup pg() const {
    return pg_;    
  }
  
 protected:
  MemoryManager(ProcGroup pg)
      : pg_{pg}  {}
  
  virtual ~MemoryManager() {}

 public:
  virtual void dealloc_coll(MemoryPoolBase& mp) = 0;
  virtual void detach_coll(MemoryPoolBase& mp) = 0;

  void* access(MemoryPoolBase& mp, Offset off) {
    return const_cast<void*>(static_cast<const MemoryManager&>(*this).access(mp, off));
  }

  virtual const void* access(const MemoryPoolBase& mp, Offset off) const = 0;
  virtual void get(MemoryPoolBase& mp, Proc proc, Offset off, Size nelements, void* buf) = 0;
  virtual void put(MemoryPoolBase& mp, Proc proc, Offset off, Size nelements, const void* buf) = 0;
  virtual void add(MemoryPoolBase& mp, Proc proc, Offset off, Size nelements, const void* buf) = 0;
  virtual void print_coll(const MemoryPoolBase& mp, std::ostream& os) = 0;

  ProcGroup pg_;
  
  friend class MemoryPoolBase;
}; // class MemoryManager

class MemoryPoolBase {
 public:
  MemoryPoolBase(Size nelements = Size{0})
      : allocation_status_{AllocationStatus::invalid},
        local_nelements_{nelements} {}

  AllocationStatus allocation_status() const {
    return allocation_status_;
  }
  
  virtual ~MemoryPoolBase() {
    EXPECTS(allocation_status_ == AllocationStatus::invalid);
  }

  Size local_nelements() const {
    return local_nelements_;
  }

  virtual ProcGroup pg() const = 0;
  virtual MemoryManager& mgr() const = 0;

  void dealloc_coll() {
    EXPECTS(allocation_status_ == AllocationStatus::created);
    dealloc_coll_impl();
    allocation_status_ = AllocationStatus::invalid;
  }

  void detach_coll() {
    EXPECTS(allocation_status_ == AllocationStatus::attached);
    detach_coll_impl();
    allocation_status_ = AllocationStatus::invalid;
  }

  const void* access(Offset off) const {
    EXPECTS(allocation_status_ == AllocationStatus::created ||
            allocation_status_ == AllocationStatus::attached);
    EXPECTS(off < local_nelements_);
    return access_impl(off);
  }

  void get(Proc proc, Offset off, Size nelements, void* buf) {
    EXPECTS(allocation_status_ == AllocationStatus::created ||
            allocation_status_ == AllocationStatus::attached);
    EXPECTS(nelements >= 0);
    return get_impl(proc, off, nelements, buf);
  }

  void put(Proc proc, Offset off, Size nelements, const void* buf) {
    EXPECTS(allocation_status_ == AllocationStatus::created ||
            allocation_status_ == AllocationStatus::attached);
    EXPECTS(nelements >= 0);
    return put_impl(proc, off, nelements, buf);
  }

  void add(Proc proc, Offset off, Size nelements, const void* buf) {
    EXPECTS(allocation_status_ == AllocationStatus::created ||
            allocation_status_ == AllocationStatus::attached);
    EXPECTS(nelements >= 0);
    return add_impl(proc, off, nelements, buf);
  }

  void print_coll(std::ostream& os = std::cout) {
    EXPECTS(allocation_status_ == AllocationStatus::created ||
            allocation_status_ == AllocationStatus::attached);
    print_coll_impl(os);
  }

  virtual void dealloc_coll_impl() = 0;
  virtual void detach_coll_impl() = 0;
  virtual const void* access_impl(Offset off) const = 0;
  virtual void get_impl(Proc proc, Offset off, Size nelements, void* buf) = 0;
  virtual void put_impl(Proc proc, Offset off, Size nelements, const void* buf) = 0;
  virtual void add_impl(Proc proc, Offset off, Size nelements, const void* buf) = 0;
  virtual void print_coll_impl(std::ostream& os) = 0;
  
 protected:
  void set_status(AllocationStatus allocation_status) {
    allocation_status_ = allocation_status;
  }

  Size local_nelements_;
  AllocationStatus allocation_status_;
}; // class MemoryPoolBase

template<typename MgrType>
class MemoryPoolImpl : public MemoryPoolBase {
 public:
  MemoryPoolImpl(MgrType& mgr)
      : mgr_{mgr} {}

  virtual ~MemoryPoolImpl() {}

  ProcGroup pg() const {
    return mgr_.pg();
  }

  MemoryManager& mgr() const override {
    return mgr_;
  }

  void dealloc_coll_impl() override {
    mgr_.dealloc_coll(*this);
  }

  void detach_coll_impl() override {
    mgr_.detach_coll(*this);
  }

  const void* access_impl(Offset off) const override {
    return mgr_.access(*this, off);
  }

  void get_impl(Proc proc, Offset off, Size nelements, void* buf) override {
    mgr_.get(*this, proc, off, nelements, buf);
  }

  void put_impl(Proc proc, Offset off, Size nelements, const void* buf) override {
    mgr_.put(*this, proc, off, nelements, buf);
  }

  void add_impl(Proc proc, Offset off, Size nelements, const void* buf) override {
    mgr_.add(*this, proc, off, nelements, buf);
  }
  
  void print_coll_impl(std::ostream& os) override {
    mgr_.print_coll(*this, os);
  }

 private:
  MgrType& mgr_;
};  // class MemoryPoolImpl

///////////////////////////////////////////////////////////////////////////
//
//          Sequential memory manager and memory pool
//
///////////////////////////////////////////////////////////////////////////

class MemoryManagerSequential : public MemoryManager {
 public:
  static MemoryManagerSequential* create_coll(ProcGroup pg) {
    return new MemoryManagerSequential{pg};
  }

  static void destroy_coll(MemoryManagerSequential* mms) {
    delete mms;
  }

  class MemoryPool : public MemoryPoolImpl<MemoryManagerSequential> {
   public:
    MemoryPool(MemoryManagerSequential& mgr)
        : MemoryPoolImpl<MemoryManagerSequential>(mgr) {}
    
   private:
    size_t elsize_;
    ElementType eltype_;
    uint8_t* buf_;
    
    friend class MemoryManagerSequential;
  }; // class MemoryPool

  MemoryPoolBase* alloc_coll(ElementType eltype, Size nelements) override {
    MemoryPool* ret = new MemoryPool(*this);
    ret->eltype_ = eltype;
    ret->elsize_ = element_size(eltype);
    ret->local_nelements_ = nelements;
    ret->buf_ = new uint8_t[nelements.value() * ret->elsize_];
    ret->set_status(AllocationStatus::created);
    return ret;
  }
  
  MemoryPoolBase* attach_coll(MemoryPoolBase& mpb) override {
    MemoryPool& mp = static_cast<MemoryPool&>(mpb);
    MemoryPool* ret = new MemoryPool(*this);
    ret->eltype_ = mp.eltype_;
    ret->elsize_ = mp.elsize_;
    ret->local_nelements_ = mp.local_nelements_;
    ret->buf_ = mp.buf_;
    ret->set_status(AllocationStatus::attached);
    return ret;
  }

  protected:
  explicit MemoryManagerSequential(ProcGroup pg)
      : MemoryManager{pg} {
    //sequential. So process group size should be 1
    EXPECTS(pg.is_valid());
    EXPECTS(pg_.size() == 1);
  }

  ~MemoryManagerSequential() {}

 public:
  void dealloc_coll(MemoryPoolBase& mpb) override {
    MemoryPool& mp = static_cast<MemoryPool&>(mpb);
    delete [] mp.buf_;
    mp.buf_ = nullptr;
  }

  void detach_coll(MemoryPoolBase& mpb) override {
    MemoryPool& mp = static_cast<MemoryPool&>(mpb);
    delete [] mp.buf_;
    mp.buf_ = nullptr;
  }

  const void* access(const MemoryPoolBase& mpb, Offset off) const override {
    const MemoryPool& mp = static_cast<const MemoryPool&>(mpb);
    return &mp.buf_[mp.elsize_ * off.value()];
  }
  
  void get(MemoryPoolBase& mpb, Proc proc, Offset off, Size nelements, void* to_buf) override {
    MemoryPool& mp = static_cast<MemoryPool&>(mpb);
    EXPECTS(proc.value() == 0);
    EXPECTS(mp.buf_ != nullptr);
    std::copy_n(mp.buf_ + mp.elsize_ * off.value(),
                mp.elsize_*nelements.value(),
                reinterpret_cast<uint8_t*>(to_buf));
  }
  
  void put(MemoryPoolBase& mpb, Proc proc, Offset off, Size nelements, const void* from_buf) override {
    MemoryPool& mp = static_cast<MemoryPool&>(mpb);
    EXPECTS(proc.value() == 0);
    EXPECTS(mp.buf_ != nullptr);
    std::copy_n(reinterpret_cast<const uint8_t*>(from_buf),
                mp.elsize_*nelements.value(),
                mp.buf_ + mp.elsize_*off.value());
  }

  void add(MemoryPoolBase& mpb, Proc proc, Offset off, Size nelements, const void* from_buf) override {
    MemoryPool& mp = static_cast<MemoryPool&>(mpb);
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
  
  void print_coll(const MemoryPoolBase& mpb, std::ostream& os) override {
    const MemoryPool& mp = static_cast<const MemoryPool&>(mpb);
    EXPECTS(mp.buf_ != nullptr);
    os<<"MemoryManagerSequential. contents\n";
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
}; // class MemoryManagerSequential


// class MemoryPoolSequential : public MemoryPoolImpl<MemoryManagerSequential> {
//  public:
//   MemoryPoolSequential(MemoryManagerSequential& mgr)
//       : MemoryPoolImpl<MemoryManagerSequential>(mgr) {}

//  private:
//   size_t elsize_;
//   ElementType eltype_;
//   uint8_t* buf_;
  
//   friend class MemoryManagerSequential;
// };

// #if 0
// MemoryManagerSequential::MemoryManagerSequential(ProcGroup pg)
//     : MemoryManager{pg} {
//   //sequential. So process group size should be 1
//   EXPECTS(pg.is_valid());
//   EXPECTS(pg_.size() == 1);
// }

// MemoryPool*
// MemoryManagerSequential::alloc(ElementType eltype, Size nelements) {
//   MemoryPoolSequential* ret = new MemoryPoolSequential(*this);
//   ret->eltype_ = eltype;
//   ret->elsize_ = element_size(eltype);
//   ret->local_nelements_ = nelements;
//   ret->buf_ = new uint8_t[nelements.value() * ret->elsize_];
//   return ret;
// }

// void
// MemoryManagerSequential::dealloc(MemoryPool& mp) {
//   MemoryPoolSequential& mps = *static_cast<MemoryPoolSequential*>(&mp);
//   delete [] mps.buf_;
//   mps.buf_ = nullptr;
// }

// void*
// MemoryManagerSequential::access(MemoryPool& mp, Offset off) {
//   MemoryPoolSequential& mps = *static_cast<MemoryPoolSequential*>(&mp);
//   return &mps.buf_[mps.elsize_ * off.value()];
// }

// const void*
// MemoryManagerSequential::access(MemoryPool& mp, Offset off) const {
//   MemoryPoolSequential& mps = *static_cast<MemoryPoolSequential*>(&mp);
//   return &mps.buf_[mps.elsize_ * off.value()];
// }

// void
// MemoryManagerSequential::get(MemoryPool& mp, Proc proc, Offset off, Size nelements, void* to_buf) {
//   MemoryPoolSequential& mps = *static_cast<MemoryPoolSequential*>(&mp);
//   EXPECTS(proc.value() == 0);
//   EXPECTS(mps.buf_ != nullptr);
//   std::copy_n(mps.buf_ + mps.elsize_ * off.value(),
//               mps.elsize_*nelements.value(),
//               reinterpret_cast<uint8_t*>(to_buf));
// }

// void
// MemoryManagerSequential::put(MemoryPool& mp, Proc proc, Offset off, Size nelements, const void* from_buf) {
//   MemoryPoolSequential& mps = *static_cast<MemoryPoolSequential*>(&mp);
//   EXPECTS(proc.value() == 0);
//   EXPECTS(mps.buf_ != nullptr);
//   std::copy_n(reinterpret_cast<const uint8_t*>(from_buf),
//               mps.elsize_*nelements.value(),
//               mps.buf_ + mps.elsize_*off.value());
// }

// void
// MemoryManagerSequential::add(MemoryPool& mp, Proc proc, Offset off, Size nelements, const void* from_buf) {
//   MemoryPoolSequential& mps = *static_cast<MemoryPoolSequential*>(&mp);
//   EXPECTS(proc.value() == 0);
//   EXPECTS(mps.buf_ != nullptr);
//   int hi = nelements.value();
//   uint8_t *to_buf = mps.buf_ + mps.elsize_*off.value();
//   switch(mps.eltype_) {
//     case ElementType::single_precision:
//       for(int i=0; i<hi; i++) {
//         reinterpret_cast<float*>(to_buf)[i] += reinterpret_cast<const float*>(from_buf)[i];
//       }
//       break;
//     case ElementType::double_precision:
//       for(int i=0; i<hi; i++) {
//         reinterpret_cast<double*>(to_buf)[i] += reinterpret_cast<const double*>(from_buf)[i];
//       }
//       break;
//     case ElementType::single_complex:
//       for(int i=0; i<hi; i++) {
//         reinterpret_cast<std::complex<float>*>(to_buf)[i] += reinterpret_cast<const std::complex<float>*>(from_buf)[i];
//       }
//       break;
//     case ElementType::double_complex:
//       for(int i=0; i<hi; i++) {
//         reinterpret_cast<std::complex<double>*>(to_buf)[i] += reinterpret_cast<const std::complex<double>*>(from_buf)[i];
//       }
//       break;
//     default:
//       assert(0);
//   }
// }

// void
// MemoryManagerSequential::print(const MemoryPool& mp, std::ostream& os) {
//   const MemoryPoolSequential& mps = *static_cast<const MemoryPoolSequential*>(&mp);
//   EXPECTS(mps.buf_ != nullptr);
//   os<<"MemoryManagerSequential. contents\n";
//   for(size_t i=0; i<mps.local_nelements().value(); i++) {
//     switch(mps.eltype_) {
//       case ElementType::double_precision:
//         os<<i<<"     "<<(reinterpret_cast<const double*>(mps.buf_))[i]<<"\n";
//         break;
//       default:
//         NOT_IMPLEMENTED();
//     }
//   }
//   os<<"\n\n";
// }
// #endif

/////////////////////////////////////////////////////////////////////////////////

// #if 0
// class MemoryManager {
//  public:
//   explicit MemoryManager(ProcGroup pg)
//       : pg_{pg},
//         eltype_{ElementType::invalid},
//         nelements_{0} {}

//   virtual ~MemoryManager() {}

//   ProcGroup proc_group() const {
//     return pg_;
//   }

//   virtual void alloc(ElementType eltype, Size nelements) = 0;
//   virtual void dealloc() = 0;

//   Size local_size_in_elements() const {
//     return nelements_;
//   }

//   // template<typename T1>
//   virtual MemoryManager* clone(ProcGroup) const = 0;

//   virtual void* access(Offset off) = 0;
//   virtual const void* access(Offset off) const = 0;
//   virtual void get(Proc proc, Offset off, Size nelements, void* buf) = 0;
//   virtual void put(Proc proc, Offset off, Size nelements, const void* buf) = 0;
//   virtual void add(Proc proc, Offset off, Size nelements, const void* buf) = 0;
//   virtual void print() const = 0;

//  protected:
//   ProcGroup pg_;
//   ElementType eltype_;
//   Size nelements_;
// }; // class MemoryManager

// class MemoryManagerSequential : public MemoryManager {
//  public:
//   explicit MemoryManagerSequential(ProcGroup pg)
//       : MemoryManager(pg),
//         buf_{nullptr},
//         elsize_{0},
//         allocation_status_{AllocationStatus::invalid} {
//           //sequential. So process group size should be 1
//           EXPECTS(pg.is_valid());
//           EXPECTS(MemoryManager::pg_.size() == 1);
//   }

//   MemoryManagerSequential(ProcGroup pg, uint8_t *buf, ElementType eltype, Size nelements)
//       : MemoryManager(pg),
//         buf_{buf},
//         elsize_{element_size(eltype_)} {
//           eltype_ = eltype;
//           nelements_ = nelements;
//           EXPECTS(pg.is_valid());
//           EXPECTS(MemoryManager::pg_.size() == 1);
//           allocation_status_ = AllocationStatus::attached;
//   }

//   ~MemoryManagerSequential() {
//     EXPECTS(allocation_status_ == AllocationStatus::invalid ||
//             allocation_status_ == AllocationStatus::attached);
//   }

//   MemoryManager* clone(ProcGroup pg) const {
//     EXPECTS(pg.is_valid());
//     return new MemoryManagerSequential(pg);
//   }

//   void alloc(ElementType eltype, Size nelements) {
//     EXPECTS(allocation_status_ == AllocationStatus::invalid);
//     eltype_ = eltype;
//     elsize_ = element_size(eltype);
//     nelements_ = nelements;
//     buf_ = new uint8_t[nelements_.value() * elsize_];
//     allocation_status_ = AllocationStatus::created;
//   }

//   void dealloc() {
//     EXPECTS(allocation_status_ == AllocationStatus::created);
//     delete [] buf_;
//     buf_ = nullptr;
//     allocation_status_ = AllocationStatus::invalid;
//   }

//   void* access(Offset off) {
//     EXPECTS(allocation_status_ == AllocationStatus::created ||
//             allocation_status_ == AllocationStatus::attached);
//     EXPECTS(off < nelements_);
//     return &buf_[elsize_ * off.value()];
//   }

//   const void* access(Offset off) const {
//     EXPECTS(allocation_status_ == AllocationStatus::created ||
//             allocation_status_ == AllocationStatus::attached);
//     EXPECTS(off < nelements_);
//     return &buf_[elsize_ * off.value()];
//   }

//   void get(Proc proc, Offset off, Size nelements, void* to_buf) {
//     EXPECTS(allocation_status_ == AllocationStatus::created ||
//             allocation_status_ == AllocationStatus::attached);
//     EXPECTS(buf_ != nullptr);
//     EXPECTS(nelements >= 0);
//     EXPECTS(off + nelements <= nelements_);
//     EXPECTS(proc.value() == 0);
//     std::copy_n(buf_ + elsize_*off.value(), elsize_*nelements.value(),
//                 reinterpret_cast<uint8_t*>(to_buf));
//   }

//   void put(Proc proc, Offset off, Size nelements, const void* from_buf) {
//     EXPECTS(allocation_status_ == AllocationStatus::created ||
//             allocation_status_ == AllocationStatus::attached);
//     EXPECTS(buf_ != nullptr);
//     EXPECTS(nelements >= 0);
//     EXPECTS(off + nelements <= nelements_);
//     EXPECTS(proc.value() == 0);
//     std::copy_n(reinterpret_cast<const uint8_t*>(from_buf),
//                 elsize_*nelements.value(),
//                 buf_ + elsize_*off.value());
//   }

//   void add(Proc proc, Offset off, Size nelements, const void* from_buf) {
//     EXPECTS(allocation_status_ == AllocationStatus::created ||
//             allocation_status_ == AllocationStatus::attached);
//     EXPECTS(buf_ != nullptr);
//     EXPECTS(nelements >= 0);
//     EXPECTS(off + nelements <= nelements_);
//     EXPECTS(proc.value() == 0);
//     int hi = nelements.value();
//     uint8_t *to_buf = buf_ + elsize_*off.value();
//     switch(eltype_) {
//       case ElementType::single_precision:
//         for(int i=0; i<hi; i++) {
//           reinterpret_cast<float*>(to_buf)[i] += reinterpret_cast<const float*>(from_buf)[i];
//         }
//         break;
//       case ElementType::double_precision:
//         for(int i=0; i<hi; i++) {
//           reinterpret_cast<double*>(to_buf)[i] += reinterpret_cast<const double*>(from_buf)[i];
//         }
//         break;
//       case ElementType::single_complex:
//         for(int i=0; i<hi; i++) {
//           reinterpret_cast<std::complex<float>*>(to_buf)[i] += reinterpret_cast<const std::complex<float>*>(from_buf)[i];
//         }
//         break;
//       case ElementType::double_complex:
//         for(int i=0; i<hi; i++) {
//           reinterpret_cast<std::complex<double>*>(to_buf)[i] += reinterpret_cast<const std::complex<double>*>(from_buf)[i];
//         }
//         break;
//       default:
//         assert(0);
//     }
//   }

//   void print() const {
//     std::cout<<"MemoryManagerSequential. contents\n";
//     for(size_t i=0; i<nelements_.value(); i++) {
//       switch(eltype_) {
//         case ElementType::double_precision:
//           std::cout<<i<<"     "<<(reinterpret_cast<const double*>(buf_))[i]<<"\n";
//           break;
//         default:
//           assert(0); //not implemented yet
//       }
//     }
//     std::cout<<"\n\n";
//   }

//  private:
//   size_t elsize_;
//   uint8_t* buf_;
//   AllocationStatus allocation_status_;
// }; // class MemoryManagerSequential

// #endif

}  // namespace tammx


#include "tammx/memory_manager_ga.h"

#endif // TAMMX_MEMORY_MANAGER_H_
