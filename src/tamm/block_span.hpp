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

// ---------------------------------------------------------------------------
// BlockSpan<T>
//
// A non-owning view over a contiguous block of T with runtime-determined
// multi-dimensional extents.  Backed by std::mdspan so that element access
// (view_(i,j,...)), strides, and extent queries are all zero-overhead.
// ---------------------------------------------------------------------------
template<typename T>
class BlockSpan {
public:
  enum class BufKind : uint8_t { cpu, invalid };

  // ------------------------------------------------------------------
  // Constructors
  // ------------------------------------------------------------------

  /// Default: invalid/null span.
  BlockSpan() noexcept
    : buf_kind_{BufKind::invalid},
      view_{nullptr, std::dextents<size_t, std::dynamic_extent>{}} {}

  /// Primary constructor: pointer + runtime dimension list.
  BlockSpan(T* buf, std::span<const size_t> dims)
    : buf_kind_{BufKind::cpu},
      view_{[&]{
        EXPECTS(buf != nullptr);
        // Build a dextents mapping from the span.
        std::vector<size_t> ext(dims.begin(), dims.end());
        return TammMdspan<T>{buf,
          std::dextents<size_t, std::dynamic_extent>(ext)};
      }()} {}

  /// Convenience: accept an std::vector<size_t> directly (common call-site).
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

  /// Backward-compatible: return extents as a vector.
  [[nodiscard]] std::vector<size_t> block_dims() const {
    std::vector<size_t> dims(view_.rank());
    for (size_t d = 0; d < view_.rank(); ++d)
      dims[d] = view_.extent(d);
    return dims;
  }

  /// Direct access to the underlying mdspan for kernel code.
  [[nodiscard]] TammMdspan<T>&       mdspan()       noexcept { return view_; }
  [[nodiscard]] const TammMdspan<T>& mdspan() const noexcept { return view_; }

  /// True when the span holds a valid (non-null) buffer.
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
