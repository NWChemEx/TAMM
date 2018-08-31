#ifndef TAMM_MEMORY_MANAGER_H_
#define TAMM_MEMORY_MANAGER_H_

#include <iosfwd>

#include "tamm/types.hpp"
#include "tamm/proc_group.hpp"

/**
 * @defgroup memory_management
 *
 * @todo Check that offset+size for remote gets and puts is within the remote size
 *
 * @todo Remove uses of void*
 */


namespace tamm {

enum class MemoryManagerType { local, distributed };

class MemoryRegion;

/**
 * @ingroup memory_management
 * @brief Abstract base class for memory manager.
 *
 * - Memory manager implements the operations to manage each type of memory.
 * - Memory manager is associated with a process group.
 * - Each memory allocation is tracked in a memory region.
 * - Memory region allocates a buffer at each rank in a given process group.
 * - Multiple memory regions can be allocated using the same memory manager and share the same process group.
 * - Each memory manager implementation (sub-class) provides the implementation of all
 * operations on memory regions managed by it.
 * - The user accesses data in the memory regions using the memory region API's routines. The memory region
 * delegates to the implementation in the memory manager.
 *
 * @note Memory regions use for their operations the memory manager that allocated them.
 * Thus, a memory manager should outlive all memory regions created by it.
 */
class MemoryManager {
 public:
  /**
   * @brief Collective allocation of a memory region.
   *
   * Collective on the process group.
   * @param eltype Element type (should be the same on all ranks making this call)
   * @param nelements Number of elements to allocate on this rank
   * @return Allocated memory region
   */
  virtual MemoryRegion* alloc_coll(ElementType eltype, Size nelements) = 0;

  /**
   * @brief Attach a memory region to the process group.
   *
   * This is collective on the memory region's process group.
   * @param mr Memory region being attached to.
   * @return Allocated memory region
   */
  virtual MemoryRegion* attach_coll(MemoryRegion& mr) = 0;

  /**
   * Access the underlying process group
   * @return Underlying process group
   */
  ProcGroup pg() const {
    return pg_;
  }

 protected:
  explicit MemoryManager(ProcGroup pg)
      : pg_{pg} {}

  virtual ~MemoryManager() {}

 public:
  /**
   * Collectively deallocate a memory region.
   * @param mr Memory region being deallocated
   */
  virtual void dealloc_coll(MemoryRegion& mr) = 0;

  /**
   * Collectively detach from a memory region.
   * @param mr Memory region being detached.
   */
  virtual void detach_coll(MemoryRegion& mr) = 0;

  /**
   * Access a pointer at an offset from local buffer associated with a memory region.
   * @param mr Memory region being accessed
   * @param off Offset (in number of elements) from base of this rank's buffer associated with this memory region.
   * @return Pointer to element at offset @p off in local buffer
   */
  void* access(MemoryRegion& mr, Offset off) {
    return const_cast<void*>(static_cast<const MemoryManager&>(*this).access(mr, off));
  }

  /**
   * Remote complete memory operations on a memory region
   * @param mr Memory region on which the fence operation is performed
   */
  virtual void fence(MemoryRegion& mr) = 0;

  /**
   * Access a pointer at an offset from local buffer associated with a memory region.
   * @param mr Memory region being accessed
   * @param off Offset (in number of elements) from base of this rank's buffer associated with this memory region.
   * @return Pointer to element at offset @p off in local buffer
   */
  virtual const void* access(const MemoryRegion& mr, Offset off) const = 0;

  /**
   * Get data from a buffer associated with a memory region into a local memory buffer
   * @param mr Memory region
   * @param proc Rank whose buffer is to be accessed
   * @param off Offset at which data is to be accessed
   * @param nelements Number of elements to get
   * @param buf Local buffer into which
   *
   * @post
   * @code
   * buf[0..nelements] = mr[proc].buf[off..off+nelements]
   * @endcode
   * @pre buf != nullptr
   * @pre buf[0..nelements] is valid (i.e., buffer is of sufficient size)
   */
  virtual void get(MemoryRegion& mr, Proc proc, Offset off, Size nelements, void* buf) = 0;

  /**
   * Put data to a buffer associated with a memory region from a local memory buffer
   * @param mr Memory region
   * @param proc Rank whose buffer is to be accessed
   * @param off Offset at which data is to be accessed
   * @param nelements Number of elements to get
   * @param buf Local buffer into which
   *
   * @post
   * @code
   * mr[proc].buf[off..off+nelements] = buf[0..nelements]
   * @endcode
   * @pre buf != nullptr
   * @pre buf[0..nelements] is valid (i.e., buffer is of sufficient size)
   */
  virtual void put(MemoryRegion& mr, Proc proc, Offset off, Size nelements, const void* buf) = 0;

  /**
   * Add data to a buffer associated with a memory region from a local memory buffer
   * @param mr Memory region
   * @param proc Rank whose buffer is to be accessed
   * @param off Offset at which data is to be accessed
   * @param nelements Number of elements to get
   * @param buf Local buffer into which
   *
   * @post
   * @code
   * mr[proc].buf[off..off+nelements] += buf[0..nelements]
   * @endcode
   * @pre buf != nullptr
   * @pre buf[0..nelements] is valid (i.e., buffer is of sufficient size)
   */
  virtual void add(MemoryRegion& mr, Proc proc, Offset off, Size nelements, const void* buf) = 0;

  /**
   * @brief Collectively print contents of the memory region
   *
   * This is mainly used for debugging.
   * @param mr Memory region to be printed
   * @param os output stream to be printed out
   */
  virtual void print_coll(const MemoryRegion& mr, std::ostream& os) = 0;

 protected:
  ProcGroup pg_;

  friend class MemoryRegion;
}; // class MemoryManager

/**
 * @ingroup memory_management
 * @brief Base class for a memory region.
 *
 * Memory regions allocate and manage one contiguous buffer per rank in the corresponding memory manager's process group.
 * Memory region contains meta-data associated with an individual (collective allocation by a memory manager.
 * Memory regions are allocated using static calls in the memory manager. Operations on the memory region
 * delegate to the memory manager. This base class provides some correctness checking by tracking the state
 * of the memory region as operations are performed on it.
 */
class MemoryRegion {
 public:
  /**
   * @brief Construct a memory region.
   *
   * This call does not allocate memory, just constructs the object.
   *
   * @param nelements Number of elements to be allocated on this rank for this memory manager.
   */
  MemoryRegion(Size nelements = Size{0})
      : allocation_status_{AllocationStatus::invalid},
        local_nelements_{nelements} {}

  /**
   * Access the current allocation status
   * @return This memory region's allocation status
   */
  AllocationStatus allocation_status() const {
    return allocation_status_;
  }

  /**
   * Has it been created (as in allocated using MemoryManager::allocate_coll)?
   * @return True if the memory region has been created.
   */
  bool created() const {
    return allocation_status_ == AllocationStatus::created;
  }

  /**
   * Has it been orphaned (as in allocated using MemoryManager::allocate_coll and then outlived its owner)?
   * @return True if the memory region has been created.
   */
  bool orphaned() const {
    return allocation_status_ == AllocationStatus::orphaned;
  }

  /**
   * Has the memory region been attached  (using MemoryManager::attach_coll())?
   * @return True if the memory region has been attached
   */
  bool attached() const {
    return allocation_status_ == AllocationStatus::attached;
  }

  virtual ~MemoryRegion() {}

  /**
   * Number of elements associated with this rank in this memory region
   * @return Number of local elements
   */
  Size local_nelements() const {
    return local_nelements_;
  }

  /**
   * Underlying process group
   * @return underlying process group
   */
  virtual ProcGroup pg() const = 0;

  /**
   * Access the memory manager used to create this memory region
   * @return Memory manager that created this memory region.
   */
  virtual MemoryManager& mgr() const = 0;

  /**
   * Deallocate this memory region
   */
  void dealloc_coll() {
    EXPECTS(created() || orphaned());
    dealloc_coll_impl();
    allocation_status_ = AllocationStatus::deallocated;
  }

  /**
   * Detach this memory region
   */
  void detach_coll() {
    EXPECTS(attached());
    detach_coll_impl();
    allocation_status_ = AllocationStatus::invalid;
  }

  /**
   * Fence (remote complete) all operations on this memory region
   */
  void fence() {
    EXPECTS(attached() || created());
    fence_impl();
  }

  /**
   * Access local (i.e., buffer in this rank) data in this memory region
   * @param off Offset at which to access (in number of elements)
   * @return Pointer to location at offset @p off in local buffer
   */
  const void* access(Offset off) const {
    EXPECTS(created() || attached());
    EXPECTS(off < local_nelements_);
    return access_impl(off);
  }

  /**
   * Get data from a buffer associated with a memory region into a local memory buffer
   * @param proc Rank whose buffer is to be accessed
   * @param off Offset at which data is to be accessed
   * @param nelements Number of elements to get
   * @param buf Local buffer into which
   *
   * @post
   * @code
   * buf[0..nelements] = mr[proc].buf[off..off+nelements]
   * @endcode
   * @pre buf != nullptr
   * @pre buf[0..nelements] is valid (i.e., buffer is of sufficient size)
   */
  void get(Proc proc, Offset off, Size nelements, void* buf) {
    EXPECTS(created() || attached());
    EXPECTS(nelements >= 0);
    return get_impl(proc, off, nelements, buf);
  }

  /**
   * Put data to a buffer associated with a memory region from a local memory buffer
   * @param proc Rank whose buffer is to be accessed
   * @param off Offset at which data is to be accessed
   * @param nelements Number of elements to get
   * @param buf Local buffer into which
   *
   * @post
   * @code
   * mr[proc].buf[off..off+nelements] = buf[0..nelements]
   * @endcode
   * @pre buf != nullptr
   * @pre buf[0..nelements] is valid (i.e., buffer is of sufficient size)
   */
  void put(Proc proc, Offset off, Size nelements, const void* buf) {
    EXPECTS(created() || attached());
    EXPECTS(nelements >= 0);
    return put_impl(proc, off, nelements, buf);
  }

  /**
   * Add data to a buffer associated with a memory region from a local memory buffer
   * @param proc Rank whose buffer is to be accessed
   * @param off Offset at which data is to be accessed
   * @param nelements Number of elements to get
   * @param buf Local buffer into which
   *
   * @post
   * @code
   * mr[proc].buf[off..off+nelements] += buf[0..nelements]
   * @endcode
   * @pre buf != nullptr
   * @pre buf[0..nelements] is valid (i.e., buffer is of sufficient size)
   */
  void add(Proc proc, Offset off, Size nelements, const void* buf) {
    EXPECTS(created() || attached());
    EXPECTS(nelements >= 0);
    return add_impl(proc, off, nelements, buf);
  }

  /**
   * Collectively print the contents of this memory region
   * @param os output stream to print to
   */
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

  // size_t elsize_;
  // uint8_t* buf_;
  AllocationStatus allocation_status_;
  Size local_nelements_;

  friend class TensorImpl;
}; // class MemoryRegion

/**
 * @ingroup memory_management
 * @brief Implementation of the memory region operations that delegates to the memory manager.
 *
 * @tparam MgrType Concrete memory manager type
 */
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

}  // namespace tamm


#include "tamm/memory_manager_local.hpp"
#include "tamm/memory_manager_ga.hpp"

#endif // TAMM_MEMORY_MANAGER_H_
