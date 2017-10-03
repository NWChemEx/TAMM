#ifndef TAMMX_MEMORY_MANAGER_H_
#define TAMMX_MEMORY_MANAGER_H_

#include <iosfwd>

#include "tammx/types.h"
#include "tammx/proc_group.h"

//@todo Check that offset+size for remote gets and puts is within the
//remote size

//@todo Remove uses of void*


namespace tammx {

enum class MemoryManagerType { local, distributed };

class MemoryRegion;

class MemoryManager {
 public:
  virtual MemoryRegion* alloc_coll(ElementType eltype, Size nelements) = 0;
  virtual MemoryRegion* attach_coll(MemoryRegion& mr) = 0;
  ProcGroup pg() const {
    return pg_;
  }

 protected:
  explicit MemoryManager(ProcGroup pg)
      : pg_{pg} {}

  virtual ~MemoryManager() {}

 public:
  virtual void dealloc_coll(MemoryRegion& mr) = 0;
  virtual void detach_coll(MemoryRegion& mr) = 0;

  void* access(MemoryRegion& mr, Offset off) {
    return const_cast<void*>(static_cast<const MemoryManager&>(*this).access(mr, off));
  }

  virtual void fence(MemoryRegion& mr) = 0;
  virtual const void* access(const MemoryRegion& mr, Offset off) const = 0;
  virtual void get(MemoryRegion& mr, Proc proc, Offset off, Size nelements, void* buf) = 0;
  virtual void put(MemoryRegion& mr, Proc proc, Offset off, Size nelements, const void* buf) = 0;
  virtual void add(MemoryRegion& mr, Proc proc, Offset off, Size nelements, const void* buf) = 0;
  virtual void print_coll(const MemoryRegion& mr, std::ostream& os) = 0;

 protected:
  ProcGroup pg_;

  friend class MemoryRegion;
}; // class MemoryManager

class MemoryRegion {
 public:
  MemoryRegion(Size nelements = Size{0})
      : allocation_status_{AllocationStatus::invalid},
        local_nelements_{nelements} {}

  AllocationStatus allocation_status() const {
    return allocation_status_;
  }

  bool created() const {
    return allocation_status_ == AllocationStatus::created;
  }

  bool attached() const {
    return allocation_status_ == AllocationStatus::attached;
  }

  virtual ~MemoryRegion() {
    EXPECTS(allocation_status_ == AllocationStatus::invalid);
  }

  Size local_nelements() const {
    return local_nelements_;
  }

  virtual ProcGroup pg() const = 0;
  virtual MemoryManager& mgr() const = 0;

  void dealloc_coll() {
    EXPECTS(created());
    dealloc_coll_impl();
    allocation_status_ = AllocationStatus::invalid;
  }

  void detach_coll() {
    EXPECTS(attached());
    detach_coll_impl();
    allocation_status_ = AllocationStatus::invalid;
  }

  void fence() {
    EXPECTS(attached() || created());
    fence_impl();
  }

  const void* access(Offset off) const {
    EXPECTS(created() || attached());
    EXPECTS(off < local_nelements_);
    return access_impl(off);
  }

  void get(Proc proc, Offset off, Size nelements, void* buf) {
    EXPECTS(created() || attached());
    EXPECTS(nelements >= 0);
    return get_impl(proc, off, nelements, buf);
  }

  void put(Proc proc, Offset off, Size nelements, const void* buf) {
    EXPECTS(created() || attached());
    EXPECTS(nelements >= 0);
    return put_impl(proc, off, nelements, buf);
  }

  void add(Proc proc, Offset off, Size nelements, const void* buf) {
    EXPECTS(created() || attached());
    EXPECTS(nelements >= 0);
    return add_impl(proc, off, nelements, buf);
  }

  void print_coll(std::ostream& os = std::cout) {
    EXPECTS(created() || attached());
    print_coll_impl(os);
  }

  virtual void dealloc_coll_impl() = 0;
  virtual void detach_coll_impl() = 0;
  virtual void fence_impl() = 0;
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
 private:
  // size_t elsize_;
  // uint8_t* buf_;
  AllocationStatus allocation_status_;
}; // class MemoryRegion

template<typename MgrType>
class MemoryRegionImpl : public MemoryRegion {
 public:
  MemoryRegionImpl(MgrType& mgr)
      : mgr_{mgr} {}

  virtual ~MemoryRegionImpl() {}

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

  void fence_impl() override {
    mgr_.fence(*this);
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
};  // class MemoryRegionImpl

}  // namespace tammx


#include "tammx/memory_manager_local.h"
#include "tammx/memory_manager_ga.h"

#endif // TAMMX_MEMORY_MANAGER_H_
