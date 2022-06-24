#pragma once

#ifdef USE_CUDA
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#elif defined(USE_HIP)
#include <hip/hip_runtime.h>
#include <rocblas.h>
#elif defined(USE_DPCPP)
//TODO: ABB
#endif

#if defined(USE_HIP)
using gpuStream_t       = hipStream_t;
using gpuEvent_t        = hipEvent_t;
#elif defined(USE_CUDA)
using gpuStream_t       = cudaStream_t;
using gpuEvent_t        = cudaEvent_t;
#elif defined(USE_DPCPP)
using gpuStream_t       = sycl::queue;
using gpuEvent_t        = sycl::event;
#endif

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace rmm {

/**
 * @brief Strongly-typed non-owning wrapper for CUDA streams with default constructor.
 *
 * This wrapper is simply a "view": it does not own the lifetime of the stream it wraps.
 */
class cuda_stream_view {
public:
  constexpr cuda_stream_view()                                   = default;
  constexpr cuda_stream_view(cuda_stream_view const&)            = default;
  constexpr cuda_stream_view(cuda_stream_view&&)                 = default;
  constexpr cuda_stream_view& operator=(cuda_stream_view const&) = default;
  constexpr cuda_stream_view& operator=(cuda_stream_view&&)      = default;
  ~cuda_stream_view()                                            = default;

  // Disable construction from literal 0
  constexpr cuda_stream_view(int)            = delete; //< Prevent cast from 0
  constexpr cuda_stream_view(std::nullptr_t) = delete; //< Prevent cast from nullptr

  /**
   * @brief Implicit conversion from gpuStream_t.
   */
  constexpr cuda_stream_view(gpuStream_t stream) noexcept: stream_{stream} {}

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
inline bool operator==(cuda_stream_view lhs, cuda_stream_view rhs) {
  return lhs.value() == rhs.value();
}

/**
 * @brief Inequality comparison operator for streams
 *
 * @param lhs The first stream view to compare
 * @param rhs The second stream view to compare
 * @return true if unequal, false if equal
 */
inline bool operator!=(cuda_stream_view lhs, cuda_stream_view rhs) { return not(lhs == rhs); }

/**
 * @brief Output stream operator for printing / logging streams
 *
 * @param os The output ostream
 * @param sv The cuda_stream_view to output
 * @return std::ostream& The output ostream
 */
inline std::ostream& operator<<(std::ostream& os, cuda_stream_view stream) {
  os << stream.value();
  return os;
}

} // namespace rmm
