#pragma once

#include "tamm/errors.hpp"
#include <vector>
#include <sstream>

#if defined(USE_CUDA)
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#elif defined(USE_HIP)
#include <hip/hip_runtime.h>
#include <rocblas.h>
#elif defined(USE_DPCPP)
#include "sycl_device.hpp"
#include <oneapi/mkl/blas.hpp>
#endif

namespace tamm {

#if defined(USE_HIP)
using gpuStream_t     = hipStream_t;
using gpuEvent_t      = hipEvent_t;
using gpuBlasHandle_t = rocblas_handle;
using gpuMemcpyKind   = hipMemcpyKind;
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice

#define HIP_CHECK(FUNC)                                                                           \
  do {                                                                                            \
    hipError_t err_ = (FUNC);                                                                     \
    if(err_ != hipSuccess) {                                                                      \
      std::ostringstream msg;                                                                     \
      msg << "HIP Error: " << hipGetErrorString(err_) << ", at " << __FILE__ << " : " << __LINE__ \
          << std::endl;                                                                           \
      throw std::runtime_error(msg.str());                                                        \
    }                                                                                             \
  } while(0)

#define ROCBLAS_CHECK(FUNC)                                                                      \
  do {                                                                                           \
    rocblas_status err_ = (FUNC);                                                                \
    if(err_ != rocblas_status_success) {                                                         \
      std::ostringstream msg;                                                                    \
      msg << "ROCBLAS Error: " << rocblas_status_to_string(err_) << ", at " << __FILE__ << " : " \
          << __LINE__ << std::endl;                                                              \
      throw std::runtime_error(msg.str());                                                       \
    }                                                                                            \
  } while(0)
#endif // USE_HIP

#if defined(USE_CUDA)
using gpuStream_t     = cudaStream_t;
using gpuEvent_t      = cudaEvent_t;
using gpuBlasHandle_t = cublasHandle_t;
using gpuMemcpyKind   = cudaMemcpyKind;
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice

#define CUDA_CHECK(FUNC)                                                                \
  do {                                                                                  \
    cudaError_t err_ = (FUNC);                                                          \
    if(err_ != cudaSuccess) {                                                           \
      std::ostringstream msg;                                                           \
      msg << "CUDA Error: " << cudaGetErrorString(err_) << ", at " << __FILE__ << " : " \
          << __LINE__ << std::endl;                                                     \
      throw std::runtime_error(msg.str());                                              \
    }                                                                                   \
  } while(0)

#define CUBLAS_CHECK(FUNC)                                                                   \
  do {                                                                                       \
    cublasStatus_t err_ = (FUNC);                                                            \
    if(err_ != CUBLAS_STATUS_SUCCESS) {                                                      \
      std::ostringstream msg;                                                                \
      msg << "CUBLAS Error: " << cublasGetStatusString(err_) << ", at " << __FILE__ << " : " \
          << __LINE__ << std::endl;                                                          \
      throw std::runtime_error(msg.str());                                                   \
    }                                                                                        \
  } while(0)
#endif // USE_CUDA

#if defined(USE_DPCPP)
using gpuStream_t   = sycl::queue;
using gpuEvent_t    = sycl::event;
using gpuMemcpyKind = int;
#define gpuMemcpyHostToDevice 0
#define gpuMemcpyDeviceToHost 1
#define gpuMemcpyDeviceToDevice 2

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

#define ONEMKLBLAS_CHECK(FUNC)                                                         \
  do {                                                                                 \
    try {                                                                              \
      (FUNC)                                                                           \
    } catch(oneapi::mkl::exception const& ex) {                                        \
      std::ostringstream msg;                                                          \
      msg << "oneMKL Error: " << ex.what() << ", at " << __FILE__ << " : " << __LINE__ \
          << std::endl;                                                                \
      throw std::runtime_error(msg.str());                                             \
    }                                                                                  \
  } while(0)
#endif // USE_DPCPP

static inline void getDeviceCount(int* id) {
#if defined(USE_CUDA)
  CUDA_CHECK(cudaGetDeviceCount(id));
#elif defined(USE_HIP)
  HIP_CHECK(hipGetDeviceCount(id));
#elif defined(USE_DPCPP)
  syclGetDeviceCount(id);
#endif
}

static inline void gpuMemGetInfo(size_t* free, size_t* total) {
#if defined(USE_CUDA)
  cudaMemGetInfo(free, total);
#elif defined(USE_HIP)
  hipMemGetInfo(free, total);
#elif defined(USE_DPCPP)
  syclMemGetInfo(free, total);
#endif
}

static inline void gpuSetDevice(int active_device) {
#ifdef USE_CUDA
  CUDA_CHECK(cudaSetDevice(active_device));
#elif defined(USE_HIP)
  HIP_CHECK(hipSetDevice(active_device));
#elif defined(USE_DPCPP)
  syclSetDevice(active_device);
#endif
}

static inline void gpuGetDevice(int* active_device) {
#ifdef USE_CUDA
  CUDA_CHECK(cudaGetDevice(active_device));
#elif defined(USE_HIP)
  HIP_CHECK(hipGetDevice(active_device));
#elif defined(USE_DPCPP)
  syclGetDevice(active_device);
#endif
}

template<typename T>
static void gpuMemcpyAsync(T* dst, const T* src, size_t count, gpuMemcpyKind kind,
                           gpuStream_t& stream) {
#if defined(USE_DPCPP)
  if(kind == gpuMemcpyDeviceToDevice) { stream.copy(src, dst, count); }
  else { stream.memcpy(dst, src, count * sizeof(T)); }
#elif defined(USE_CUDA)
  CUDA_CHECK(cudaMemcpyAsync(dst, src, count * sizeof(T), kind, stream));
#elif defined(USE_HIP)
  HIP_CHECK(hipMemcpyAsync(dst, src, count * sizeof(T), kind, stream));
#endif
}

static inline void gpuMemsetAsync(void*& ptr, size_t sizeInBytes, gpuStream_t stream) {
#if defined(USE_DPCPP)
  stream.memset(ptr, 0, sizeInBytes);
#elif defined(USE_HIP)
  hipMemsetAsync(ptr, 0, sizeInBytes, stream);
#elif defined(USE_CUDA)
  cudaMemsetAsync(ptr, 0, sizeInBytes, stream);
#endif
}

static inline void gpuStreamWaitEvent(gpuStream_t stream, gpuEvent_t event) {
#if defined(USE_DPCPP)
  auto retEvent = stream.ext_oneapi_submit_barrier({event});
  // retEvent.wait();
  event.wait();
#elif defined(USE_HIP)
  hipStreamWaitEvent(stream, event, 0);
#elif defined(USE_CUDA)
  cudaStreamWaitEvent(stream, event, 0);
#endif
}

static inline void gpuStreamSynchronize(gpuStream_t stream) {
#if defined(USE_DPCPP)
  stream.wait();
#elif defined(USE_HIP)
  hipStreamSynchronize(stream);
#elif defined(USE_CUDA)
  cudaStreamSynchronize(stream);
#endif
}

static inline void gpuEventRecord(gpuEvent_t event, gpuStream_t stream) {
#if defined(USE_DPCPP)
  // auto retEvent = stream.ext_oneapi_submit_barrier(event);
  // retEvent.wait();
  event.wait();
#elif defined(USE_HIP)
  hipEventRecord(event, stream);
#elif defined(USE_CUDA)
  cudaEventRecord(event, stream);
#endif
}

static inline void gpuEventCreateWithFlags(gpuEvent_t* event) {
#if defined(USE_HIP)
  hipEventCreateWithFlags(event, hipEventDisableTiming);
#elif defined(USE_CUDA)
  cudaEventCreateWithFlags(event, cudaEventDisableTiming);
#endif
}

static inline void gpuEventDestroy(gpuEvent_t event) {
#if defined(USE_HIP)
  hipEventDestroy(event);
#elif defined(USE_CUDA)
  cudaEventDestroy(event);
#endif
}

static inline void gpuEventSynchronize(gpuEvent_t event) {
#if defined(USE_DPCPP)
  event.wait();
#elif defined(USE_HIP)
  hipEventSynchronize(event);
#elif defined(USE_CUDA)
  cudaEventSynchronize(event);
#endif
}

class GPUStreamPool {
protected:
  int default_deviceID{0};
  int nstreams{4};

  // GPU stream
  std::vector<gpuStream_t*> _devStream;
#if defined(USE_CUDA) || defined(USE_HIP)
  // blashandle
  std::vector<gpuBlasHandle_t*> _devHandle;
#endif

private:
  GPUStreamPool():
    _devStream(nstreams)
#ifndef USE_DPCPP
    ,
    _devHandle(nstreams)
#endif
  {
    // Assert here if multi-GPUs are detected
    int ngpus{0};
    getDeviceCount(&ngpus);
    EXPECTS_STR((ngpus == 1), "Error: More than 1 GPU-device found per rank!");

    gpuSetDevice(default_deviceID);

    for(int j = 0; j < nstreams; j++) {
#if defined(USE_CUDA)
      _devStream[j] = new cudaStream_t;
      CUDA_CHECK(cudaStreamCreateWithFlags(_devStream[j], cudaStreamNonBlocking));

      _devHandle[j] = new gpuBlasHandle_t;
      CUBLAS_CHECK(cublasCreate(_devHandle[j]));
      CUBLAS_CHECK(cublasSetStream(*_devHandle[j], *_devStream[j]));
#elif defined(USE_HIP)
      _devStream[j] = new hipStream_t;
      HIP_CHECK(hipStreamCreateWithFlags(_devStream[j], hipStreamNonBlocking));

      _devHandle[j] = new gpuBlasHandle_t;
      ROCBLAS_CHECK(rocblas_create_handle(_devHandle[j]));
      ROCBLAS_CHECK(rocblas_set_stream(*_devHandle[j], *_devStream[j]));
#elif defined(USE_DPCPP)
      _devStream[j] = new sycl::queue(*sycl_get_context(default_deviceID),
                                      *sycl_get_device(default_deviceID), sycl_asynchandler,
                                      sycl::property_list{sycl::property::queue::in_order{}});
#endif
    }
  }

  ~GPUStreamPool() {
    for(int j = 0; j < nstreams; j++) {
#if defined(USE_CUDA)
      CUDA_CHECK(cudaStreamDestroy(*_devStream[j]));
      CUBLAS_CHECK(cublasDestroy(*_devHandle[j]));
      _devHandle[j] = nullptr;
#elif defined(USE_HIP)
      HIP_CHECK(hipStreamDestroy(*_devStream[j]));
      ROCBLAS_CHECK(rocblas_destroy_handle(*_devHandle[j]));
      _devHandle[j] = nullptr;
#elif defined(USE_DPCPP)
      delete _devStream[j];
#endif
    }
  }

public:
  /// Returns a GPU stream
  gpuStream_t& getStream() { return *_devStream[0]; }

#if !defined(USE_DPCPP)
  /// Returns a GPU BLAS handle that is valid only for the CUDA and HIP builds
  gpuBlasHandle_t& getBlasHandle() { return *_devHandle[0]; }
#endif

  /// Returns the instance of device manager singleton.
  inline static GPUStreamPool& getInstance() {
    static GPUStreamPool d_m{};
    return d_m;
  }

  GPUStreamPool(const GPUStreamPool&)            = delete;
  GPUStreamPool& operator=(const GPUStreamPool&) = delete;
  GPUStreamPool(GPUStreamPool&&)                 = delete;
  GPUStreamPool& operator=(GPUStreamPool&&)      = delete;
};

// This API needs to be defined after the class GPUStreamPool since the classs
// is only declared and defined before this method
} // namespace tamm
