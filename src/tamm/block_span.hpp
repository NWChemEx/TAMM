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
 * C++20 changes vs. original:
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

  /**
   * @brief Default constructor: produces an invalid/null span.
   */
  BlockSpan() noexcept
    : buf_kind_{BufKind::invalid},
      view_{nullptr, std::dextents<size_t, std::dynamic_extent>{}} {}

  /**
   * @brief Primary constructor: pointer + runtime dimension list.
   *
   * @param[in] buf  Pointer to the data buffer (must not be null)
   * @param[in] dims Extents of each dimension as a contiguous span
   *
   * @pre buf != nullptr
   */
  BlockSpan(T* buf, std::span<const size_t> dims)
    : buf_kind_{BufKind::cpu},
      view_{[&]{
        EXPECTS(buf != nullptr);
        std::vector<size_t> ext(dims.begin(), dims.end());
        return TammMdspan<T>{buf,
          std::dextents<size_t, std::dynamic_extent>(ext)};
      }()} {}

  /**
   * @brief Convenience constructor: accept an std::vector<size_t> directly.
   *
   * @param[in] buf        Pointer to the data buffer
   * @param[in] block_dims Dimension extents as a vector
   */
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

  /**
   * @brief Raw data pointer (mutable).
   * @return Pointer to the first element of the block
   */
  [[nodiscard]] T* buf() noexcept { return view_.data_handle(); }

  /**
   * @brief Raw data pointer (const).
   * @return Const pointer to the first element of the block
   */
  [[nodiscard]] const T* buf() const noexcept { return view_.data_handle(); }

  /**
   * @brief Flat element count (product of all extents).
   * @return Total number of elements
   */
  [[nodiscard]] size_t num_elements() const noexcept { return view_.size(); }

  /**
   * @brief Number of dimensions.
   * @return Rank of the block
   */
  [[nodiscard]] size_t rank() const noexcept { return view_.rank(); }

  /**
   * @brief Extent along dimension d.
   * @param[in] d Dimension index
   * @return Number of elements along dimension d
   */
  [[nodiscard]] size_t extent(size_t d) const { return view_.extent(d); }

  /**
   * @brief Backward-compatible accessor: return all extents as a vector.
   * @return Vector of extents, one per dimension
   */
  [[nodiscard]] std::vector<size_t> block_dims() const {
    std::vector<size_t> dims(view_.rank());
    for (size_t d = 0; d < view_.rank(); ++d)
      dims[d] = view_.extent(d);
    return dims;
  }

  /**
   * @brief Direct access to the underlying mdspan for kernel code.
   * @return Reference to the internal TammMdspan
   */
  [[nodiscard]] TammMdspan<T>&       mdspan()       noexcept { return view_; }
  [[nodiscard]] const TammMdspan<T>& mdspan() const noexcept { return view_; }

  /**
   * @brief True when the span holds a valid (non-null) buffer.
   * @return True if valid, false if default-constructed or invalidated
   */
  [[nodiscard]] bool is_valid() const noexcept {
    return buf_kind_ != BufKind::invalid && view_.data_handle() != nullptr;
  }

private:
  BufKind          buf_kind_;
  TammMdspan<T>    view_;      ///< dims + pointer packed in one object
};

// ---------------------------------------------------------------------------
// Factory helpers
// ---------------------------------------------------------------------------

/**
 * @brief Build a BlockSpan from a raw pointer + initializer-list of dimensions.
 *
 * @tparam T      Element type
 * @param[in] buf  Pointer to data buffer
 * @param[in] dims Brace-enclosed list of extents, e.g. {rows, cols}
 * @return A BlockSpan wrapping buf with the given extents
 */
template<typename T>
[[nodiscard]] inline BlockSpan<T>
make_block_span(T* buf, std::initializer_list<size_t> dims) {
  return BlockSpan<T>{buf, std::vector<size_t>{dims}};
}

/**
 * @brief Build a BlockSpan from a raw pointer + contiguous range of dimensions.
 *
 * @tparam T      Element type
 * @tparam R      Contiguous range whose value type is convertible to size_t
 * @param[in] buf  Pointer to data buffer
 * @param[in] dims Range of extents
 * @return A BlockSpan wrapping buf with the given extents
 */
template<typename T, std::ranges::contiguous_range R>
  requires std::is_convertible_v<std::ranges::range_value_t<R>, size_t>
[[nodiscard]] inline BlockSpan<T>
make_block_span(T* buf, const R& dims) {
  std::vector<size_t> dv(std::ranges::begin(dims), std::ranges::end(dims));
  return BlockSpan<T>{buf, std::span<const size_t>{dv}};
}

} // namespace tamm
