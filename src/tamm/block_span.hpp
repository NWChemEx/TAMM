#pragma once

// C++20: std::mdspan for zero-overhead multi-dimensional block views.
// Falls back to the kokkos/mdspan reference implementation when the compiler
// stdlib does not yet ship <mdspan> (GCC < 13, Clang < 17).
#if __has_include(<mdspan>)
#  include <mdspan>
#else
#  include "experimental/mdspan"
#endif

#include <span>
#include "tamm/errors.hpp"
#include "tamm/types.hpp"

namespace tamm {

/// Dynamic-rank mdspan alias used throughout TAMM block operations.
template<typename T>
using TammMdspan = std::mdspan<T, std::dextents<size_t, std::dynamic_extent>>;

/**
 * @brief Non-owning view over a contiguous block of T with runtime-determined
 *        multi-dimensional extents.
 *
 * Backed by std::mdspan so that element access (view_(i,j,...)), strides,
 * and extent queries are all zero-overhead.
 *
 * C++20 / perf changes vs. prior version:
 *  - block_dims() now returns std::span<const size_t> into a cached
 *    TensorVec<size_t> extents_ member — zero heap allocation per call.
 *  - block_dims_vec() provided for callers that truly need a std::vector.
 *  - Internal storage replaced by std::mdspan (TammMdspan<T>).
 *  - Added make_block_span factory helpers.
 *  - BufKind tag retained for CPU/invalid distinction.
 *  - [[nodiscard]] applied to all pure query accessors.
 *
 * @tparam T Element type of the block
 */
template<typename T>
class BlockSpan {
public:
  enum class BufKind : uint8_t { cpu, invalid };

  // ------------------------------------------------------------------
  // Constructors
  // ------------------------------------------------------------------

  /// Default constructor: produces an invalid/null span.
  BlockSpan() noexcept
    : buf_kind_{BufKind::invalid},
      view_{nullptr, std::dextents<size_t, std::dynamic_extent>{}} {}

  /**
   * @brief Primary constructor: pointer + runtime dimension list.
   *
   * Caches the extents into extents_ (TensorVec, stack-allocated) so
   * that block_dims() never needs to heap-allocate.
   *
   * @param[in] buf  Pointer to the data buffer (must not be null)
   * @param[in] dims Extents of each dimension as a contiguous span
   *
   * @pre buf != nullptr
   */
  BlockSpan(T* buf, std::span<const size_t> dims)
    : buf_kind_{BufKind::cpu} {
    EXPECTS(buf != nullptr);
    extents_.resize(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) extents_[i] = dims[i];
    std::vector<size_t> ext(dims.begin(), dims.end());
    view_ = TammMdspan<T>{buf,
      std::dextents<size_t, std::dynamic_extent>(ext)};
  }

  /// Convenience constructor: accept an std::vector<size_t> directly.
  BlockSpan(T* buf, const std::vector<size_t>& block_dims)
    : BlockSpan(buf, std::span<const size_t>{block_dims}) {}

  BlockSpan(const BlockSpan&)            = default;
  BlockSpan(BlockSpan&&)                 = default;
  ~BlockSpan()                           = default;
  BlockSpan& operator=(const BlockSpan&) = default;
  BlockSpan& operator=(BlockSpan&&)      = default;

  // ------------------------------------------------------------------
  // Accessors
  // ------------------------------------------------------------------

  /// Raw data pointer (mutable).
  [[nodiscard]] T* buf() noexcept { return view_.data_handle(); }

  /// Raw data pointer (const).
  [[nodiscard]] const T* buf() const noexcept { return view_.data_handle(); }

  /// Flat element count (product of all extents).
  [[nodiscard]] size_t num_elements() const noexcept { return view_.size(); }

  /// Number of dimensions.
  [[nodiscard]] size_t rank() const noexcept { return view_.rank(); }

  /// Extent along dimension d.
  [[nodiscard]] size_t extent(size_t d) const { return view_.extent(d); }

  /**
   * @brief Zero-allocation accessor: returns a span view into the cached
   *        extents array.  Preferred over block_dims_vec() in hot paths.
   *
   * @return std::span<const size_t> of length rank()
   */
  [[nodiscard]] std::span<const size_t> block_dims() const noexcept {
    return std::span<const size_t>{extents_.data(), extents_.size()};
  }

  /**
   * @brief Backward-compatible: return all extents as a heap-allocated
   *        std::vector.  Use only when a vector is strictly required.
   *
   * @return std::vector<size_t> of extents, one per dimension
   */
  [[nodiscard]] std::vector<size_t> block_dims_vec() const {
    return std::vector<size_t>(extents_.begin(), extents_.end());
  }

  /// Direct access to the underlying mdspan for kernel code.
  [[nodiscard]] TammMdspan<T>&       mdspan()       noexcept { return view_; }
  [[nodiscard]] const TammMdspan<T>& mdspan() const noexcept { return view_; }

  /// True when the span holds a valid (non-null) buffer.
  [[nodiscard]] bool is_valid() const noexcept {
    return buf_kind_ != BufKind::invalid && view_.data_handle() != nullptr;
  }

private:
  BufKind             buf_kind_;
  TammMdspan<T>       view_;      ///< dims + pointer packed in one object
  TensorVec<size_t>   extents_;   ///< cached extents — stack-allocated, zero-copy block_dims()
};

// ---------------------------------------------------------------------------
// Factory helpers
// ---------------------------------------------------------------------------

/// Build a BlockSpan from a raw pointer + initializer-list of dimensions.
template<typename T>
[[nodiscard]] inline BlockSpan<T>
make_block_span(T* buf, std::initializer_list<size_t> dims) {
  return BlockSpan<T>{buf, std::vector<size_t>{dims}};
}

/// Build a BlockSpan from a raw pointer + contiguous range of dimensions.
template<typename T, std::ranges::contiguous_range R>
  requires std::is_convertible_v<std::ranges::range_value_t<R>, size_t>
[[nodiscard]] inline BlockSpan<T>
make_block_span(T* buf, const R& dims) {
  std::vector<size_t> dv(std::ranges::begin(dims), std::ranges::end(dims));
  return BlockSpan<T>{buf, std::span<const size_t>{dv}};
}

} // namespace tamm
