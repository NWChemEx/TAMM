#pragma once

#include "tamm/tensor.hpp"
#include "tamm/types.hpp"
#include <iterator>
#include <span>
#include <vector>

namespace tamm {

class RuntimeEngine;

/**
 * @brief Non-owning view + optional owner over a contiguous block of T.
 *
 * C++20 rewrite:
 *  - Ownership managed by std::vector<T> storage_ instead of raw new[]/delete[].
 *  - Exposed as std::span<T> buf_span_ (non-owning view into storage_).
 *  - The 'allocated' bool flag is eliminated entirely.
 *  - Copy / move / dtor are Rule-of-Zero (compiler-generated, always correct).
 *
 * @tparam T Element type of the block
 */
template<typename T>
class BlockBuffer {
public:
  // -------------------------------------------------------------------
  // Constructors
  // -------------------------------------------------------------------

  BlockBuffer() = default;

  /// Wrap an externally-owned span (no allocation, not an owner).
  BlockBuffer(std::span<T> buf_span, IndexedTensor<T> indexedTensor,
              RuntimeEngine* re)
    : buf_span_{buf_span}, indexedTensor_{indexedTensor}, re_{re} {}

  /// Allocate a fresh buffer, fetch the block from the tensor.
  BlockBuffer(Tensor<T> tensor, IndexVector blockid)
    : storage_(tensor.block_size(blockid)),
      indexedTensor_{tensor, blockid} {
    buf_span_ = std::span<T>{storage_.data(), storage_.size()};
    tensor.get(blockid, buf_span_);
  }

  // Rule-of-Zero: copy / move / dtor are all compiler-generated.
  // - Copy: deep-copies storage_, buf_span_ reconstructed from it below
  //         via the copy ctor body (storage_ copy + span retarget).
  // - Move: storage_ and buf_span_ are both moved correctly.
  // - Dtor: storage_ RAII-cleans up automatically.
  //
  // We need a custom copy constructor only to retarget buf_span_ after
  // the vector copy (span still points at the source's allocation).
  BlockBuffer(const BlockBuffer& other)
    : storage_{other.storage_},
      indexedTensor_{other.indexedTensor_},
      re_{other.re_} {
    // If other was owning (storage_ has data), point our span at our copy.
    // If other was non-owning (storage_ empty, span points externally),
    // copy the span view as-is.
    if (!storage_.empty())
      buf_span_ = std::span<T>{storage_.data(), storage_.size()};
    else
      buf_span_ = other.buf_span_;
  }

  BlockBuffer& operator=(const BlockBuffer& other) {
    if (this == &other) return *this;
    storage_       = other.storage_;
    indexedTensor_ = other.indexedTensor_;
    re_            = other.re_;
    if (!storage_.empty())
      buf_span_ = std::span<T>{storage_.data(), storage_.size()};
    else
      buf_span_ = other.buf_span_;
    return *this;
  }

  // Move ctor: after std::vector move, storage_ is valid in *this and
  // empty in other; retarget span.
  BlockBuffer(BlockBuffer&& other) noexcept
    : storage_{std::move(other.storage_)},
      buf_span_{other.buf_span_},
      indexedTensor_{std::move(other.indexedTensor_)},
      re_{other.re_} {
    if (!storage_.empty())
      buf_span_ = std::span<T>{storage_.data(), storage_.size()};
    other.buf_span_ = {};
    other.re_       = nullptr;
  }

  BlockBuffer& operator=(BlockBuffer&& other) noexcept {
    if (this == &other) return *this;
    storage_       = std::move(other.storage_);
    buf_span_      = other.buf_span_;
    indexedTensor_ = std::move(other.indexedTensor_);
    re_            = other.re_;
    if (!storage_.empty())
      buf_span_ = std::span<T>{storage_.data(), storage_.size()};
    other.buf_span_ = {};
    other.re_       = nullptr;
    return *this;
  }

  ~BlockBuffer() = default;

  // -------------------------------------------------------------------
  // Iterators / data access
  // -------------------------------------------------------------------
  auto begin()        { return buf_span_.begin(); }
  auto begin()  const { return buf_span_.begin(); }
  auto end()          { return buf_span_.end();   }
  auto end()    const { return buf_span_.end();   }

  [[nodiscard]] std::span<T>       get_span()       { return buf_span_; }
  [[nodiscard]] std::span<const T> get_span() const { return buf_span_; }
  [[nodiscard]] T*       data()       { return buf_span_.data(); }
  [[nodiscard]] const T* data() const { return buf_span_.data(); }

  // -------------------------------------------------------------------
  // Release helpers (write-back and free)
  // -------------------------------------------------------------------
  void release_put() {
    indexedTensor_.put(buf_span_);
    storage_.clear();
    buf_span_ = {};
  }
  void release_put(Tensor<T> tensor, IndexVector blockid) {
    tensor.put(blockid, buf_span_);
    storage_.clear();
    buf_span_ = {};
  }
  void release_add() {
    indexedTensor_.add(buf_span_);
    storage_.clear();
    buf_span_ = {};
  }
  void release_add(Tensor<T> tensor, IndexVector blockid) {
    tensor.add(blockid, buf_span_);
    storage_.clear();
    buf_span_ = {};
  }
  void release() {
    storage_.clear();
    buf_span_ = {};
  }

  [[nodiscard]] std::vector<size_t> block_dims() {
    return indexedTensor_.first.block_dims(indexedTensor_.second);
  }

  template<typename V>
  BlockBuffer& operator=(const V val) {
    std::fill(begin(), end(), val);
    return *this;
  }

private:
  std::vector<T>   storage_;       ///< owns memory when non-empty
  std::span<T>     buf_span_;      ///< non-owning view (into storage_ or external)
  IndexedTensor<T> indexedTensor_;
  RuntimeEngine*   re_{nullptr};
};

template<typename T>
bool operator==(const BlockBuffer<T>& lhs, const BlockBuffer<T>& rhs) {
  return lhs.get_span().size() == rhs.get_span().size() &&
         std::equal(lhs.get_span().begin(), lhs.get_span().end(),
                    rhs.get_span().begin());
}

template<typename T, typename Stream>
inline auto& operator<<(Stream& os, BlockBuffer<T> bf) {
  for (auto it = bf.begin(); it != bf.end(); ++it) { os << *it << " "; }
  return os;
}

} // namespace tamm
