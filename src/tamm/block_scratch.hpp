#pragma once

#include <cstddef>
#include <vector>

namespace tamm::internal {

/**
 * @brief RAII + pooled scratch for the per-block "own-or-view" buffer pattern.
 *
 * Many op-apply loops (set/add/mult) need a working pointer to a block that is
 * EITHER a view into the tensor's local buffer (no ownership, no copy) OR a
 * freshly allocated copy fetched from a remote rank.  The legacy code expressed
 * this with a raw `new T[n]` + a `bool alloced` flag + a matching `delete[]`,
 * which leaks if anything between the `new` and the `delete[]` throws.
 *
 * BlockScratch owns a grow-only `std::vector<T>` that is reused across loop
 * iterations (pooling: no per-block malloc/free churn once it reaches the max
 * block size) and hands back a raw `T*` for the kernel.  Declare ONE instance
 * outside the block loop; call view()/owned() per block.
 *
 * Not thread-safe: use one instance per worker thread.
 */
template<typename T>
class BlockScratch {
public:
  BlockScratch()                               = default;
  BlockScratch(const BlockScratch&)            = delete;
  BlockScratch& operator=(const BlockScratch&) = delete;
  BlockScratch(BlockScratch&&)                 = default;
  BlockScratch& operator=(BlockScratch&&)      = default;
  ~BlockScratch()                              = default;

  /// Use an external (tensor-local) buffer directly; no ownership, no copy.
  [[nodiscard]] T* view(T* external) noexcept { return external; }

  /// Provide an owned, reusable buffer of at least @p n elements.
  /// The storage is grow-only and reused on subsequent calls (pooling).
  [[nodiscard]] T* owned(std::size_t n) {
    if(storage_.size() < n) { storage_.resize(n); }
    return storage_.data();
  }

private:
  std::vector<T> storage_;
};

} // namespace tamm::internal
