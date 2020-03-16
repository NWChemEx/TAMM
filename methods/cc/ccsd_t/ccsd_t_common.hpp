#ifndef CCSD_T_COMMON_HPP_
#define CCSD_T_COMMON_HPP_

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <sys/types.h>
#include <sys/time.h>
#include <assert.h>
#include <time.h>

#ifdef USE_TALSH
  #define USE_TALSH_T
#endif
#undef USE_TALSH

#define CHECK_ERR(x) { \
    cudaError_t err = cudaGetLastError();\
    if (cudaSuccess != err) { \
        printf("%s\n",cudaGetErrorString(err)); \
        exit(1); \
    } } 

#define CUDA_SAFE(x) if ( cudaSuccess != (x) ) {\
    printf("CUDA CALL FAILED AT LINE %d OF FILE %s error %s\n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); exit(100);}


typedef long Integer;
//static int notset;

#define DIV_UB(x,y) ((x)/(y)+((x)%(y)?1:0))
#define TG_MIN(x,y) ((x)<(y)?(x):(y))

void initMemModule();
void *getGpuMem(size_t bytes);
void *getHostMem(size_t bytes);
void freeHostMem(void *p);
void freeGpuMem(void *p);
void finalizeMemModule();

#endif /*CCSD_T_COMMON_HPP_*/
