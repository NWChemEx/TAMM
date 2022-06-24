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

// headers related to memory pool
#include "tamm/mr/device_memory_resource.hpp"
#include "tamm/mr/gpu_memory_resource.hpp"
#include "tamm/mr/per_device_resource.hpp"
#include "tamm/mr/pool_memory_resource.hpp"

#if defined(USE_HIP)
using gpuStream_t                            = hipStream_t;
using gpuEvent_t                             = hipEvent_t;
using gpuBlasHandle_t                        = rocblas_handle;
constexpr unsigned short max_gpu_streams = 1;
#elif defined(USE_CUDA)
using gpuStream_t                            = cudaStream_t;
using gpuEvent_t                             = cudaEvent_t;
using gpuBlasHandle_t                        = cublasHandle_t;
constexpr unsigned short max_gpu_streams = 1;
#elif defined(USE_DPCPP)
using gpuStream_t                            = sycl::queue;
using gpuEvent_t                             = sycl::event;
constexpr unsigned short max_gpu_streams = 1;

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

static inline void gpuGetDeviceCount(int* id) {
#if defined(USE_CUDA)
  cudaGetDeviceCount(id);
#elif defined(USE_HIP)
  hipGetDeviceCount(id);
#elif defined(USE_DPCPP)
  syclGetDeviceCount(id);
#endif
}

static inline void gpuSetDevice(int active_device) {
#ifdef USE_CUDA
      cudaSetDevice(active_device);
#elif defined(USE_HIP)
      hipSetDevice(active_device);
#elif defined(USE_DPCPP)
      syclSetDevice(active_device);
#endif
}

class GPUPool {
protected:
  // Map of GPU-IDs and vector of streams  
  std::map <int, std::vector<gpuStream_t*>> _devID2Streams;

  // thread-safe, GPU-specific counter for getting a round-robin stream/queue from pool
  // For now this is required for (T) code and multiple streams are present
  unsigned int *_count{nullptr};
  
  #if defined(USE_CUDA) || defined(USE_HIP)
  // Map of GPU-IDs and array of blashandles
  std::map<int, gpuBlasHandle_t*> _devID2Handles;
  #endif

  bool _initialized{false};

  // Active GPU set by a given MPI-rank from execution context ctor
  // Note: This can be obtained without the need for the variable by using the
  //       vendor APIs, but this will avoid overhead from calling vendor APIs
  int _active_device;  

  // Handle memory-resource wrapper to Memory pool
  using pool_mr = rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>;
  std::vector<std::unique_ptr<pool_mr>> _devID2MR;
  
private:
    GPUPool() {
      std::cout << "calling the constructor() \n";
      
      int numGPUs_{};
      gpuGetDeviceCount(&numGPUs_);
      _count = new unsigned int[numGPUs_];
      _devID2MR.reserve(numGPUs_);
      
      for (int devID=0; devID<numGPUs_; devID++) { // # of GPUs per node
	_active_device = devID;
	gpuSetDevice(devID);
	_count[devID] = 0;

	// 1. populate gpu-streams per GPU
	std::vector<gpuStream_t*> streamVec(max_gpu_streams);
	for (unsigned short j=0; j<max_gpu_streams; j++) { // # of streams per GPU
	  gpuStream_t* stream = nullptr;
          #if defined(USE_CUDA)
  	  stream = new cudaStream_t;
	  cudaStreamCreate(stream);
          #elif defined(USE_HIP)
	  stream = new hipStream_t;
	  hipStreamCreate(stream);
          #elif defined(USE_DPCPP)
	  stream = new sycl::queue(*sycl_get_context(devID), *sycl_get_device(devID),
				   sycl_asynchandler);
          #endif
	  streamVec[j] = stream;
	} // j
	_devID2Streams[devID] = std::move(streamVec);

	// 2. populate gpu-blas handles
#if defined(USE_CUDA) || defined(USE_HIP)
	gpuBlasHandle_t* handle = new gpuBlasHandle_t;
#if defined(USE_CUDA)
	cublasCreate(handle);
#elif defined(USE_HIP)
	rocblas_create_handle(handle);
#endif
	_devID2Handles.insert( std::pair<int, gpuBlasHandle_t*>(devID, handle) );
#endif// USE_CUDA, USE_HIP

	// 3. populate the per-device memory pools
	_devID2MR[devID] = std::make_unique<pool_mr>(rmm::mr::get_per_device_resource(devID), *((_devID2Streams[devID])[0]));	
      } // devID

    }
  
  ~GPUPool() {
//     _count       = 0;
//     _initialized = false;

// #if !defined(USE_DPCPP) && !defined(USE_TALSH)
//     if(!_devID2Streams.empty()) {
//       for(auto& stream: _devID2Streams) {
// #if defined(USE_CUDA)
// 	cudaStreamDestroy(stream.second);
// #elif defined(USE_HIP)
// 	hipStreamDestroy(stream.second);
// #endif
//       }
//     }

//     if(!_devID2Handles.empty()) {
//       for(auto& handle: _devID2Handles) {
// #if defined(USE_CUDA)
// 	cublasDestroy(handle.second);
// #elif defined(USE_HIP)
// 	rocblas_destroy_handle(handle.second);
// #endif
//       }
//     }

//     _devID2Streams.clear();
//     _devID2Handles.clear();
// #endif
  }

public:

  /// sets an active device for getting streams, blas, memory-pool handles
  void set_device(int device) {
    if(!_initialized) {
      _active_device = device;
      gpuSetDevice(_active_device);
      _initialized = true;
    }
  }

  /// Returns a RMM pooll handle
  pool_mr* get_memory_pool() {
    if(!_initialized) {
      EXPECTS_STR(false, "Error: active GPU-device not set! call set_device()!");
    }
    
    return _devID2MR[_active_device].get();
  }
  
  /// Returns a GPU stream in a round-robin fashion
  gpuStream_t& getStream() {
    if(!_initialized) {
      EXPECTS_STR(false, "Error: active GPU-device not set! call set_device()!");
    }
    
    unsigned int streamCounter = (_count[_active_device])++ % max_gpu_streams;
    return *((_devID2Streams[_active_device])[streamCounter]);    
  }

#if !defined(USE_DPCPP) && !defined(USE_TALSH)
  /// Returns a GPU BLAS handle that is valid only for the CUDA and HIP builds
  gpuBlasHandle_t& getBlasHandle() {
    if(!_initialized) {
      EXPECTS_STR(false, "Error: active GPU-device not set! call set_device()!");
    }

    return *(_devID2Handles[_active_device]);
  }
#endif
  
  /// Returns the instance of device manager singleton.
  inline static GPUPool& getInstance() {
    static GPUPool d_m{};
    return d_m;
  }

  GPUPool(const GPUPool&)            = delete;
  GPUPool& operator=(const GPUPool&) = delete;
  GPUPool(GPUPool&&)                 = delete;
  GPUPool& operator=(GPUPool&&)      = delete;
};

} // namespace tamm
