#pragma once

// Non-owning view over a contiguous tensor block with runtime extents.
//
// NOTE on std::mdspan: TAMM blocks have a *runtime* rank.  std::mdspan (and
// the kokkos reference implementation) fix the rank at compile time — there is
// no standard "dynamic-rank" mdspan (std::dextents<T, std::dynamic_extent> is
// ill-formed: it attempts to build an extents object of rank 2^64-1).  All
// TAMM block kernels treat a block as a flat, contiguous buffer plus an
// extents list (see block_assign_plan.hpp, blockops_*.hpp), so we model
// BlockSpan directly as pointer + extents and expose a std::span<T> for flat
// element access.  Kernel code that knows the rank at compile time can build a
// fixed-rank std::mdspan directly from buf() + block_dims() where beneficial.

#include <cstddef>
#include <cstdint>
#include <ranges>
#include <span>
#include <vector>

#include "tamm/errors.hpp"
#include "tamm/types.hpp"

namespace tamm {

/**
 * @brief Non-owning view over a contiguous block of T with runtime-determined
 *        multi-dimensional extents.
 *
 * The block is stored contiguously in row-major (layout_right) order.  Element
 * access is available flat (operator[](i)) or as a std::span (flat_span()).
 *
 * C++20 features used:
 *  - std::span for the flat view (flat_span()) and the extents span accessor.
 *  - std::ranges::contiguous_range constraint on the factory helper.
 *  - [[nodiscard]] on all pure query accessors.
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
  BlockSpan() noexcept: buf_kind_{BufKind::invalid}, buf_{nullptr}, num_elements_{0} {}

  /**
   * @brief Primary constructor: pointer + runtime dimension list.
   *
   * Extents are cached once (single allocation) so block_dims() can return a
   * const reference without re-allocating on every call.
   *
   * @param[in] buf  Pointer to the data buffer (must not be null)
   * @param[in] dims Extents of each dimension as a contiguous span
   *
   * @pre buf != nullptr
   */
  BlockSpan(T* buf, std::span<const size_t> dims):
    buf_kind_{BufKind::cpu}, buf_{buf}, extents_(dims.begin(), dims.end()) {
    EXPECTS(buf != nullptr);
    num_elements_ = 1;
    for(size_t d: extents_) num_elements_ *= d;
  }

  /// Convenience constructor: accept an std::vector<size_t> directly.
  BlockSpan(T* buf, const std::vector<size_t>& block_dims):
    BlockSpan(buf, std::span<const size_t>{block_dims}) {}

  BlockSpan(const BlockSpan&)            = default;
  BlockSpan(BlockSpan&&)                 = default;
  ~BlockSpan()                           = default;
  BlockSpan& operator=(const BlockSpan&) = default;
  BlockSpan& operator=(BlockSpan&&)      = default;

  // ------------------------------------------------------------------
  // Accessors
  // ------------------------------------------------------------------

  /// Raw data pointer (mutable).
  [[nodiscard]] T* buf() noexcept { return buf_; }
  /// Raw data pointer (const).
  [[nodiscard]] const T* buf() const noexcept { return buf_; }

  /// Alias for buf() — matches std::span / std::vector naming.
  [[nodiscard]] T*       data() noexcept { return buf_; }
  [[nodiscard]] const T* data() const noexcept { return buf_; }

  /// Flat element count (product of all extents).
  [[nodiscard]] size_t num_elements() const noexcept { return num_elements_; }

  /// Number of dimensions.
  [[nodiscard]] size_t rank() const noexcept { return extents_.size(); }

  /// Extent along dimension d.
  [[nodiscard]] size_t extent(size_t d) const {
    EXPECTS(d < extents_.size());
    return extents_[d];
  }

  /**
   * @brief Flat (linear) element access into the contiguous block.
   *
   * The block is stored contiguously (row-major), so flat indexing is
   * well-defined for any rank.  This is the access pattern used by the
   * element-wise block-multiply kernels (scalar/vector Hadamard paths),
   * e.g. lhs[0], lhs_vec[i].
   *
   * @param i Flat element index in [0, num_elements()).
   */
  [[nodiscard]] T&       operator[](size_t i) noexcept { return buf_[i]; }
  [[nodiscard]] const T& operator[](size_t i) const noexcept { return buf_[i]; }

  /// Flat contiguous view over the whole block.
  [[nodiscard]] std::span<T>       flat_span() noexcept { return {buf_, num_elements_}; }
  [[nodiscard]] std::span<const T> flat_span() const noexcept { return {buf_, num_elements_}; }

  /**
   * @brief Extents of the block, one per dimension.
   *
   * Returns a const reference to the cached extents vector so it can be passed
   * directly to the block kernels (index_permute_assign, ipgen_assign, ...)
   * which take `const std::vector<size_t>&`.  No allocation per call.
   *
   * @return const std::vector<size_t>& of length rank()
   */
  [[nodiscard]] const std::vector<size_t>& block_dims() const noexcept { return extents_; }

  /// Zero-allocation span view over the extents (for range algorithms).
  [[nodiscard]] std::span<const size_t> block_dims_span() const noexcept {
    return std::span<const size_t>{extents_.data(), extents_.size()};
  }

  /// Backward-compatible copy of the extents.
  [[nodiscard]] std::vector<size_t> block_dims_vec() const { return extents_; }

  /// True when the span holds a valid (non-null) buffer.
  [[nodiscard]] bool is_valid() const noexcept {
    return buf_kind_ != BufKind::invalid && buf_ != nullptr;
  }

private:
  BufKind             buf_kind_;
  T*                  buf_;
  size_t              num_elements_{0};
  std::vector<size_t> extents_; ///< cached extents (allocated once at construction)
};

// ---------------------------------------------------------------------------
// Factory helpers
// ---------------------------------------------------------------------------

/// Build a BlockSpan from a raw pointer + initializer-list of dimensions.
template<typename T>
[[nodiscard]] inline BlockSpan<T> make_block_span(T* buf, std::initializer_list<size_t> dims) {
  return BlockSpan<T>{buf, std::vector<size_t>{dims}};
}

/// Build a BlockSpan from a raw pointer + contiguous range of dimensions.
template<typename T, std::ranges::contiguous_range R>
requires std::is_convertible_v<std::ranges::range_value_t<R>, size_t>
[[nodiscard]] inline BlockSpan<T> make_block_span(T* buf, const R& dims) {
  std::vector<size_t> dv(std::ranges::begin(dims), std::ranges::end(dims));
  return BlockSpan<T>{buf, std::span<const size_t>{dv}};
}

} // namespace tamm
