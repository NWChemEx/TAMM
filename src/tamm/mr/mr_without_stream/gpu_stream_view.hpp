#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include "../gpu_streams.hpp"

namespace tamm::rmm {

/**
 * @brief Strongly-typed non-owning wrapper for CUDA streams with default constructor.
 *
 * This wrapper is simply a "view": it does not own the lifetime of the stream it wraps.
 */
class gpu_stream_view {
public:
  constexpr gpu_stream_view()                                  = default;
  constexpr gpu_stream_view(gpu_stream_view const&)            = default;
  constexpr gpu_stream_view(gpu_stream_view&&)                 = default;
  constexpr gpu_stream_view& operator=(gpu_stream_view const&) = default;
  constexpr gpu_stream_view& operator=(gpu_stream_view&&)      = default;
  ~gpu_stream_view()                                           = default;

  // Disable construction from literal 0
  constexpr gpu_stream_view(int)            = delete; //< Prevent cast from 0
  constexpr gpu_stream_view(std::nullptr_t) = delete; //< Prevent cast from nullptr

  /**
   * @brief Implicit conversion from gpuStream_t.
   */
  constexpr gpu_stream_view(gpuStream_t stream) noexcept: stream_{stream} {}

  /**
   * @brief Get the wrapped stream.
   *
   * @return gpuStream_t The wrapped stream.
   */
  [[nodiscard]] constexpr gpuStream_t value() const noexcept { return stream_; }

  /**
   * @brief Implicit conversion to gpuStream_t.
   */
  constexpr operator gpuStream_t() const noexcept { return value(); }

private:
  gpuStream_t stream_{};
};

/**
 * @brief Equality comparison operator for streams
 *
 * @param lhs The first stream view to compare
 * @param rhs The second stream view to compare
 * @return true if equal, false if unequal
 */
inline bool operator==(gpu_stream_view lhs, gpu_stream_view rhs) {
  return lhs.value() == rhs.value();
}

/**
 * @brief Inequality comparison operator for streams
 *
 * @param lhs The first stream view to compare
 * @param rhs The second stream view to compare
 * @return true if unequal, false if equal
 */
inline bool operator!=(gpu_stream_view lhs, gpu_stream_view rhs) { return not(lhs == rhs); }

/**
 * @brief Output stream operator for printing / logging streams
 *
 * @param os The output ostream
 * @param sv The gpu_stream_view to output
 * @return std::ostream& The output ostream
 */
inline std::ostream& operator<<(std::ostream& os, gpu_stream_view stream) {
  os << stream.value();
  return os;
}

} // namespace tamm::rmm
