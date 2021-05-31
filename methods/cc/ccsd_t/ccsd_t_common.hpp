#ifndef CCSD_T_COMMON_HPP_
#define CCSD_T_COMMON_HPP_

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#elif defined(USE_HIP)
#include <hip/hip_runtime.h>
#elif defined(USE_DPCPP)
#include <CL/sycl.hpp>
#endif

#ifdef USE_HIP
using gpuStream_t = hipStream_t;
using gpuEvent_t = hipEvent_t;
#elif defined(USE_CUDA)
using gpuStream_t = cudaStream_t;
using gpuEvent_t = cudaEvent_t;
#elif defined(USE_DPCPP)
using gpuStream_t = sycl::queue;
#endif

#include <stdio.h>
#include <sys/types.h>
#include <sys/time.h>
#include <assert.h>
#include <time.h>
#include <string>

#ifdef USE_TALSH
  #define USE_TALSH_T
#endif
#undef USE_TALSH

#ifdef USE_CUDA
#define CHECK_ERR(x) { \
    cudaError_t err = cudaGetLastError();\
    if (cudaSuccess != err) { \
        printf("%s\n",cudaGetErrorString(err)); \
        exit(1); \
    } }

#define CUDA_SAFE(x) if ( cudaSuccess != (x) ) {\
    printf("CUDA CALL FAILED AT LINE %d OF FILE %s error %s\n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); exit(100);}
#endif // USE_CUDA

#ifdef USE_HIP
#define CHECK_ERR(x) { \
    hipError_t err = hipGetLastError();\
    if (hipSuccess != err) { \
        printf("%s\n",hipGetErrorString(err)); \
        exit(1); \
    } }

#define HIP_SAFE(x) if ( hipSuccess != (x) ) {\
    printf("HIP CALL FAILED AT LINE %d OF FILE %s error %s\n", __LINE__, __FILE__, hipGetErrorString(hipGetLastError()) ); exit(100);}
#endif // USE_HIP

typedef long Integer;
//static int notset;

#define DIV_UB(x,y) ((x)/(y)+((x)%(y)?1:0))
#define TG_MIN(x,y) ((x)<(y)?(x):(y))

void initMemModule();
std::string check_memory_req(const int nDevices, const int cc_t_ts, const int nbf);

#if defined(USE_DPCPP)
void *getGpuMem(sycl::queue& syclQueue, size_t bytes);
void *getHostMem(sycl::queue& syclQueue, size_t bytes);
void freeHostMem(sycl::queue& syclQueue, void *p);
void freeGpuMem(sycl::queue& syclQueue, void *p);
#else
void *getGpuMem(size_t bytes);
void *getHostMem(size_t bytes);
void freeHostMem(void *p);
void freeGpuMem(void *p);
#endif
void finalizeMemModule();

#endif /*CCSD_T_COMMON_HPP_*/
