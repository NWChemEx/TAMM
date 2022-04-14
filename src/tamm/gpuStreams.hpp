#pragma once

#include <map>

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#elif defined(USE_HIP)
#include <hip/hip_runtime.h>
#elif defined(USE_DPCPP)
#include "sycl_device.hpp"
#endif

// Use (void) to silence unused warnings.
#define assertm(exp, msg) assert(((void) msg, exp))

#ifdef USE_HIP
using gpuStream_t                            = hipStream_t;
using gpuEvent_t                             = hipEvent_t;
constexpr unsigned short int max_gpu_streams = 1;
#elif defined(USE_CUDA)
using gpuStream_t                            = cudaStream_t;
using gpuEvent_t                             = cudaEvent_t;
constexpr unsigned short int max_gpu_streams = 2;
#elif defined(USE_DPCPP)
using gpuStream_t                            = sycl::queue;
using gpuEvent_t                             = std::vector<sycl::event>;
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
#endif // USE_DPCPP

namespace tamm {
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

    if(!_devID2Streams.empty()) {
      for(auto& map: _devID2Streams) {
#if defined(USE_CUDA)
        cudaStreamDestroy(map.second);
#elif defined(USE_HIP)
        hipStreamDestroy(map.second);
#elif defined(USE_DPCPP)
#endif
      }
    }

    _devID2Streams.clear();
  }

public:
  void set_device(unsigned int device) {
    assertm(device < _ngpus, "Error: Invalid active-device set in GPUStreamPool!");

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

  gpuStream_t& getStream() {
    unsigned short int counter = _count++ % max_gpu_streams;

#ifdef USE_CUDA
    auto         result   = _devID2Streams.insert({_active_device + counter, gpuStream_t()});
    gpuStream_t& stream   = (*result.first).second;
    bool&        inserted = result.second;
    if(inserted) { cudaStreamCreate(&stream); }
    return stream;
#elif defined(USE_HIP)
    auto         result   = _devID2Streams.insert({_active_device + counter, gpuStream_t()});
    gpuStream_t& stream   = (*result.first).second;
    bool&        inserted = result.second;
    if(inserted) { hipStreamCreate(&stream); }
    return stream;
#elif defined(USE_DPCPP)
    auto result = _devID2Streams.insert(
      {_active_device + counter, gpuStream_t(*sycl_get_context(_active_device),
                                             *sycl_get_device(_active_device), sycl_asynchandler)});
    gpuStream_t& stream   = (*result.first).second;
    bool&        inserted = result.second;
    return stream;
#endif
  }

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
