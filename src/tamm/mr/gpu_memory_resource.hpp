#pragma once

#include "device_memory_resource.hpp"

#include <cstddef>

namespace rmm::mr {
/**
 * @brief `device_memory_resource` derived class that uses cudaMalloc/Free for
 * allocation/deallocation.
 */
class gpu_memory_resource final : public device_memory_resource {
 public:
  gpu_memory_resource()                            = default;
  ~gpu_memory_resource() override                  = default;
  gpu_memory_resource(gpu_memory_resource const&) = default;
  gpu_memory_resource(gpu_memory_resource&&)      = default;
  gpu_memory_resource& operator=(gpu_memory_resource const&) = default;
  gpu_memory_resource& operator=(gpu_memory_resource&&) = default;

 private:
  /**
   * @brief Allocates memory of size at least `bytes` using cudaMalloc.
   *
   * The returned pointer has at least 256B alignment.
   *
   * @note Stream argument is ignored
   *
   * @throws `rmm::bad_alloc` if the requested allocation could not be fulfilled
   *
   * @param bytes The size, in bytes, of the allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* do_allocate(std::size_t bytes, cuda_stream_view stream) override
  {
    void* ptr{nullptr};
#if defined(USE_CUDA)
    cudaMalloc(&ptr, bytes);
#elif defined(USE_HIP)
    hipMalloc(&ptr, bytes);    
#elif defined(USE_DPCPP)
    ptr = sycl::malloc_device(bytes, stream);
#endif
    
    std::cout << "gpu_memory_resource::do_allocate() : " << bytes << ", " << ptr << std::endl;
    return ptr;
  }

  /**
   * @brief Deallocate memory pointed to by \p p.
   *
   * @note Stream argument is ignored.
   *
   * @throws Nothing.
   *
   * @param p Pointer to be deallocated
   */
  void do_deallocate(void* ptr, std::size_t bytes, cuda_stream_view stream) override
  {
#if defined(USE_CUDA)    
    cudaFree(ptr);
#elif defined(USE_HIP)
    hipFree(ptr);    
#elif defined(USE_DPCPP)
    sycl::free(ptr, stream);
#endif    
    std::cout << "gpu_memory_resource::do_deallocate() : " << bytes << ", " << ptr << std::endl;
  }

  /**
   * @brief Compare this resource to another.
   *
   * Two gpu_memory_resources always compare equal, because they can each
   * deallocate memory allocated by the other.
   *
   * @throws Nothing.
   *
   * @param other The other resource to compare to
   * @return true If the two resources are equivalent
   * @return false If the two resources are not equal
   */
  [[nodiscard]] bool do_is_equal(device_memory_resource const& other) const noexcept override
  {
    return dynamic_cast<gpu_memory_resource const*>(&other) != nullptr;
  }

  /**
   * @brief Get free and available memory for memory resource
   *
   * @throws `rmm::cuda_error` if unable to retrieve memory info.
   *
   * @return std::pair contaiing free_size and total_size of memory
   */
  [[nodiscard]] std::pair<std::size_t, std::size_t> do_get_mem_info(cuda_stream_view) const override
  {
    std::size_t free_size{};
    std::size_t total_size{};
#if defined(USE_CUDA)
    cudaMemGetInfo(&free_size, &total_size);
#elif defined(USE_HIP)
    hipMemGetInfo(&free_size, &total_size);
#elif defined(USE_DPCPP)
    syclMemGetInfo(&free_size, &total_size);
#endif
    
    return std::make_pair(free_size, total_size);
  }
};
}  // namespace rmm::mr
