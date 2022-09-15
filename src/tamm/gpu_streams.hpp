#pragma once

#include "tamm/errors.hpp"
#include <map>

#ifdef USE_CUDA
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#elif defined(USE_HIP)
#include <hip/hip_runtime.h>
#include <rocblas.h>
#elif defined(USE_DPCPP)
#include "sycl_device.hpp"
#endif

namespace tamm {

#if defined(USE_HIP)
using gpuStream_t                            = hipStream_t;
using gpuEvent_t                             = hipEvent_t;
using gpuBlasHandle_t                        = rocblas_handle;
constexpr unsigned short int max_gpu_streams = 1;
#elif defined(USE_CUDA)
using gpuStream_t                            = cudaStream_t;
using gpuEvent_t                             = cudaEvent_t;
using gpuBlasHandle_t                        = cublasHandle_t;
constexpr unsigned short int max_gpu_streams = 1;
#elif defined(USE_DPCPP)
using gpuStream_t                            = sycl::queue;
using gpuEvent_t                             = sycl::event;
constexpr unsigned short int max_gpu_streams = 1;

auto sycl_asynchandler = [](sycl::exception_list exceptions) {
  for(std::exception_ptr const& e: exceptions) {
    try {
      std::rethrow_exception(e);
    } catch(sycl::exception const& ex) {
      std::cout << "Caught asynchronous SYCL exception:" << std::endl
                << ex.what() << ", SYCL code: " << ex.code() << std::endl;
    }
  }
};
#endif

static inline void getDeviceCount(int* id) {
#if defined(USE_CUDA)
  cudaGetDeviceCount(id);
#elif defined(USE_HIP)
  hipGetDeviceCount(id);
#elif defined(USE_DPCPP)
  syclGetDeviceCount(id);
#endif
}

class GPUStreamPool {
protected:
  bool _initialized{false};
  // Active GPU set by a given MPI-rank from execution context ctor
  int _active_device;
  // total number of GPUs on node
  int _ngpus;
  // Map of GPU-IDs and array of streams
  std::map<int, gpuStream_t> _devID2Streams;
  // counter for getting a round-robin stream used by (T) code
  unsigned int _count{0};
#if defined(USE_CUDA) || defined(USE_HIP)
  // Map of GPU-IDs and array of blashandles
  std::map<int, gpuBlasHandle_t> _devID2Handles;
#endif

private:
  GPUStreamPool() {
    // total number of GPUs on-node
    getDeviceCount(&_ngpus);
    _initialized = false;
    _count       = 0;
  }

  ~GPUStreamPool() {
    _count       = 0;
    _initialized = false;

#if !defined(USE_DPCPP)
    if(!_devID2Streams.empty()) {
      for(auto& stream: _devID2Streams) {
#if defined(USE_CUDA)
        cudaStreamDestroy(stream.second);
#elif defined(USE_HIP)
        hipStreamDestroy(stream.second);
#endif
      }
    }

    if(!_devID2Handles.empty()) {
      for(auto& handle: _devID2Handles) {
#if defined(USE_CUDA)
        cublasDestroy(handle.second);
#elif defined(USE_HIP)
        rocblas_destroy_handle(handle.second);
#endif
      }
    }

    _devID2Streams.clear();
    _devID2Handles.clear();
#endif
  }

public:
  /// sets an active device for getting streams and blas handles
  void set_device(unsigned int device) {
    EXPECTS_STR(device < _ngpus, "Error: Invalid active-device set in GPUStreamPool!");

    if(!_initialized) {
      _count         = 0;
      _active_device = device;

#ifdef USE_CUDA
      cudaSetDevice(_active_device);
#elif defined(USE_HIP)
      hipSetDevice(_active_device);
#elif defined(USE_DPCPP)
      syclSetDevice(_active_device);
#endif
      _initialized = true;
    }
  }

  /// Returns a GPU stream in a round-robin fashion
  gpuStream_t& getStream() {
    if(!_initialized) {
      EXPECTS_STR(false, "Error: active GPU-device not set! call set_device()!");
    }

    unsigned short int counter = _count++ % max_gpu_streams;

#if defined(USE_CUDA)
    auto         result   = _devID2Streams.insert({_active_device + counter, gpuStream_t()});
    gpuStream_t& stream   = (*result.first).second;
    bool&        inserted = result.second;
    if(inserted) { cudaStreamCreate(&stream); }
#elif defined(USE_HIP)
    auto         result   = _devID2Streams.insert({_active_device + counter, gpuStream_t()});
    gpuStream_t& stream   = (*result.first).second;
    bool&        inserted = result.second;
    if(inserted) { hipStreamCreate(&stream); }
#elif defined(USE_DPCPP)
    auto result =
      _devID2Streams.insert({_active_device + counter,
                             gpuStream_t(*sycl_get_device(_active_device), sycl_asynchandler,
                                         sycl::property_list{sycl::property::queue::in_order{}})});
    gpuStream_t& stream = (*result.first).second;
#endif
    return stream;
  }

#if !defined(USE_DPCPP)
  gpuBlasHandle_t& getBlasHandle() {
    if(!_initialized) {
      EXPECTS_STR(false, "Error: active GPU-device not set! call set_device()!");
    }
    auto             result   = _devID2Handles.insert({_active_device, gpuBlasHandle_t()});
    gpuBlasHandle_t& handle   = (*result.first).second;
    bool&            inserted = result.second;
#if defined(USE_CUDA)
    if(inserted) { cublasCreate(&handle); }
#elif defined(USE_HIP)
    if(inserted) { rocblas_create_handle(&handle); }
#endif
    return handle;
  }
#endif

  /// Returns the instance of device manager singleton.
  inline static GPUStreamPool& getInstance() {
    static GPUStreamPool d_m;
    return d_m;
  }

  GPUStreamPool(const GPUStreamPool&)            = delete;
  GPUStreamPool& operator=(const GPUStreamPool&) = delete;
  GPUStreamPool(GPUStreamPool&&)                 = delete;
  GPUStreamPool& operator=(GPUStreamPool&&)      = delete;
};

} // namespace tamm
