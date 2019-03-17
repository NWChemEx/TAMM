#ifndef __header_hpp__
#define __header_hpp__

#include <cuda.h>
#include <cuda_runtime.h>

//static int notset;
// extern "C" {

#include <stdio.h>
#include <cuda.h>
#include <sys/types.h>
#include <sys/time.h>
#include <assert.h>
#include <time.h>
////#include "util.h"

#define CHECK_ERR(x) { \
    cudaError_t err = cudaGetLastError();\
    if (cudaSuccess != err) { \
        printf("%s\n",cudaGetErrorString(err)); \
        exit(1); \
    } } 

#define CUDA_SAFE(x) if ( cudaSuccess != (x) ) {\
    printf("CUDA CALL FAILED AT LINE %d OF FILE %s error %s\n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); exit(1);}


typedef long Integer;

#define DIV_UB(x,y) ((x)/(y)+((x)%(y)?1:0))
#define MIN(x,y) ((x)<(y)?(x):(y))

void initMemModule();
void *getGpuMem(size_t bytes);
void *getHostMem(size_t bytes);
void freeHostMem(void *p);
void freeGpuMem(void *p);
void finalizeMemModule();

// }


#endif /*__header_hpp__*/
/* $Id$ */
