#include "ccsd_t_common.hpp"
#include <map>
#include <set>
#include <cstdlib>
using namespace std;

// #define NO_OPT

// extern "C" {

// static int is_init=0;

static map<size_t,set<void*> > free_list_gpu, free_list_host;
static map<void *,size_t> live_ptrs_gpu, live_ptrs_host;

static void clearGpuFreeList(
#ifdef USE_DPCPP
			     sycl::queue& syclQueue
#endif
			     )
{
  for(map<size_t,set<void*> >::iterator it=free_list_gpu.begin(); it!=free_list_gpu.end(); ++it)
  {
    for(set<void*>::iterator it2=it->second.begin(); it2!=it->second.end(); ++it2)
    {
#if defined(USE_CUDA)
      cudaFree(*it2);
#elif defined(USE_HIP)
      hipFree(*it2);
#elif defined(USE_DPCPP)
      sycl::free(*it2, syclQueue);
#endif
    }
  }
  free_list_gpu.clear();
}

static void clearHostFreeList(
#if defined(USE_DPCPP)
			      sycl::queue& syclQueue
#endif
)
{
  for(map<size_t,set<void*> >::iterator it=free_list_host.begin(); it!=free_list_host.end(); ++it)
  {
    for(set<void*>::iterator it2=it->second.begin(); it2!=it->second.end(); ++it2)
    {
#if defined(USE_CUDA)
      cudaFreeHost(*it2);
#elif defined(USE_HIP)
      hipHostFree(*it2);
#elif defined(USE_DPCPP)
      sycl::free(*it2, syclQueue);
#else
      free(*it2);
#endif // USE_CUDA
    }
  }
  free_list_host.clear();
}

static size_t num_resurrections=0;// num_morecore=0;

static void *moreDeviceMem(
#ifdef USE_DPCPP
			   sycl::queue& syclQueue,
#endif
			   size_t bytes)
{
  void *ptr=nullptr;
#if defined(USE_CUDA)
  CUDA_SAFE(cudaMalloc(&ptr, bytes));
#elif defined(USE_HIP)
  HIP_SAFE(hipMalloc(&ptr, bytes));
#elif defined(USE_DPCPP)
  ptr = sycl::malloc_device(bytes, syclQueue);
#endif

  // num_morecore += 1;
  if(ptr==nullptr) {     /*try one more time*/
#if defined(USE_CUDA)
    clearHostFreeList();
    clearGpuFreeList();
    CUDA_SAFE(cudaMalloc(&ptr, bytes));
#elif defined(USE_HIP)
    clearHostFreeList();
    clearGpuFreeList();
    HIP_SAFE(hipMalloc(&ptr, bytes));
#elif defined(USE_DPCPP)
    clearHostFreeList(syclQueue);
    clearGpuFreeList(syclQueue);
    ptr = sycl::malloc_device(bytes, syclQueue);
#endif
  }
  assert(ptr!=nullptr); /*We hopefully have a pointer*/
  return ptr;
}

static void *moreHostMem(
#ifdef USE_DPCPP
			 sycl::queue& syclQueue,
#endif
			 size_t bytes)
{
  void *ptr=nullptr;
  #if defined(USE_CUDA)
    CUDA_SAFE(cudaMallocHost(&ptr, bytes));
  #elif defined(USE_HIP)
    HIP_SAFE(hipHostMalloc(&ptr, bytes));
  #elif defined(USE_DPCPP)
    ptr = sycl::malloc_host(bytes, syclQueue);
  #else
    ptr = (void *)malloc(bytes);
  #endif

  if(ptr==nullptr) {     /*try one more time*/
    #if defined(USE_CUDA)
        clearHostFreeList();
        clearGpuFreeList();
        CUDA_SAFE(cudaMallocHost(&ptr, bytes));
    #elif defined(USE_HIP)
        clearHostFreeList();
        clearGpuFreeList();
        HIP_SAFE(hipHostMalloc(&ptr, bytes));
    #elif defined(USE_DPCPP)
        clearHostFreeList(syclQueue);
        clearGpuFreeList(syclQueue);
        ptr = sycl::malloc_host(bytes, syclQueue);
    #else
      ptr = (void *)malloc(bytes);
    #endif
  }

  assert(ptr!=nullptr); /*We hopefully have a pointer*/
  return ptr;
}

static inline void *resurrect_from_free_list(map<size_t,set<void *> > &free_map,
                                             size_t bytes,
                                             map<void*,size_t>& liveset)
{
  void *ptr=nullptr;
  num_resurrections +=1 ;
  assert(free_map.find(bytes) != free_map.end());
  /* assert(free_map.find(bytes)->second.size() > 0); */
  set<void *> &st = free_map.find(bytes)->second;
  ptr = *st.begin();
  st.erase(ptr);
  if(st.size()==0)
    free_map.erase(bytes);
  liveset[ptr] = bytes;
  return ptr;
}

void initmemmodule()
{
  //is_init=1;
}


void *getGpuMem(
#ifdef USE_DPCPP
		sycl::queue& syclQueue,
#endif
      size_t bytes)
{
  //assert(is_init);
  void *ptr=nullptr;
#ifdef NO_OPT
#if defined(USE_CUDA)
  CUDA_SAFE(cudaMalloc((void **) &ptr, bytes));
#elif defined(USE_HIP)
  HIP_SAFE(hipMalloc((void **) &ptr, bytes));
#elif defined(USE_DPCPP)
  ptr = sycl::malloc_device(bytes, syclQueue);
#endif
#else
  if(free_list_gpu.find(bytes)!=free_list_gpu.end())
  {
    set<void*> &lst = free_list_gpu.find(bytes)->second;
    if(lst.size()!=0)
    {
      ptr = resurrect_from_free_list(free_list_gpu, bytes, live_ptrs_gpu);
      return ptr;
    }
  }
  else
  {
    for(map<size_t,set<void *> >::iterator it=free_list_gpu.begin(); it != free_list_gpu.end(); ++it)
    {
      if(it->first >= bytes && it->second.size()>0)
      {
        ptr = resurrect_from_free_list(free_list_gpu, it->first, live_ptrs_gpu);
        return ptr;
      }
    }
  }

  ptr = moreDeviceMem(
#ifdef USE_DPCPP
		      syclQueue,
#endif
			bytes);
  live_ptrs_gpu[ptr] = bytes;
#endif // NO_OPT
  return ptr;
}

void *getHostMem(
#ifdef USE_DPCPP
		 sycl::queue& syclQueue,
#endif
		 size_t bytes)
{
  //assert(is_init);
  void *ptr=nullptr;
#ifdef NO_OPT
#if defined(USE_CUDA)
  CUDA_SAFE(cudaMallocHost((void **) &ptr, bytes));
#elif defined(USE_HIP)
  HIP_SAFE(hipHostMalloc((void **) &ptr, bytes));
#elif defined(USE_DPCPP)
  ptr = sycl::malloc_host(bytes, syclQueue);
#else //cpu
  ptr = (void *)malloc(bytes);
#endif
#else // NO_OPT
  if(free_list_host.find(bytes)!=free_list_host.end())
  {
    set<void*> &lst = free_list_host.find(bytes)->second;
    if(lst.size()!=0)
    {
      ptr = resurrect_from_free_list(free_list_host, bytes, live_ptrs_host);
      /* ptr = *lst.begin(); */
      /* lst.erase(lst.begin()); */
      /* live_ptrs_host[ptr] = bytes; */
      return ptr;
    }
  }
  else
  {
    for(map<size_t,set<void *> >::iterator it=free_list_host.begin(); it != free_list_host.end(); ++it)
    {
      if(it->first >= bytes && it->second.size()>0)
      {
	      ptr = resurrect_from_free_list(free_list_host, it->first, live_ptrs_host);
        /* 	set<void*> &lst = it->second; */
        /* 	ptr = *lst.begin(); */
        /* 	lst.erase(lst.begin()); */
        /* 	live_ptrs_gpu[ptr] = bytes; */
	      return ptr;
      }
    }
  }
  /* cutilSafeCall(cudaMallocHost((void **) &ptr, bytes)); */

  ptr = moreHostMem(
#ifdef USE_DPCPP
		    syclQueue,
#endif
		    bytes);
  live_ptrs_host[ptr] = bytes;
#endif // NO_OPT
  return ptr;
}

void freeHostMem(
#ifdef USE_DPCPP
		 sycl::queue& syclQueue,
#endif
		 void *p)
{
  size_t bytes;
  //assert(is_init);
#ifdef NO_OPT
#if defined(USE_CUDA)
  cudaFreeHost(p);
#elif defined(USE_HIP)
  hipHostFree(p);
#elif defined(USE_DPCPP)
  sycl::free(p, syclQueue);
#else
  free(p);
#endif

#else
  assert(live_ptrs_host.find(p) != live_ptrs_host.end());
  bytes = live_ptrs_host[p];
  live_ptrs_host.erase(p);
  free_list_host[bytes].insert(p);
#endif //NO_OPT
}

void freeGpuMem(
#ifdef USE_DPCPP
		sycl::queue& syclQueue,
#endif
		void *p)
{
  size_t bytes;
  //assert(is_init);
#ifdef NO_OPT
#if defined(USE_CUDA)
  cudaFree(p);
#elif defined(USE_HIP)
  hipFree(p);
#elif defined(USE_DPCPP)
  sycl::free(p, syclQueue);
#endif //NO_OPT

#else
  assert(live_ptrs_gpu.find(p) != live_ptrs_gpu.end());
  bytes = live_ptrs_gpu[p];
  live_ptrs_gpu.erase(p);
  free_list_gpu[bytes].insert(p);
#endif
}

void finalizememmodule(
#if defined(USE_DPCPP)
		       sycl::queue& syclQueue
#endif
		       )
{

  //assert(is_init);
  //is_init = 0;

  /*there should be no live pointers*/
  assert(live_ptrs_gpu.size()==0);
  assert(live_ptrs_host.size()==0);

  /*release all freed pointers*/

  clearGpuFreeList(
#ifdef USE_DPCPP
		   syclQueue
#endif
		   );

  clearHostFreeList(
#ifdef USE_DPCPP
		    syclQueue
#endif
		    );

}
