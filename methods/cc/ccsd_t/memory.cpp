#include "ccsd_t_common.hpp"
#include <map>
#include <set>
using namespace std;

// #define NO_OPT

// extern "C" {

// static int is_init=0;

static map<size_t,set<void*> > free_list_gpu, free_list_host;
static map<void *,size_t> live_ptrs_gpu, live_ptrs_host;

#ifdef USE_DPCPP
static cl::sycl::device get_current_device() noexcept
{
  auto sycl_device = get_current_queue().get_device();
  return sycl_device;
}

static cl::sycl::queue get_current_queue() noexcept
{
  cl::sycl::gpu_selector device_selector;
  auto sycl_queue = cl::sycl::queue(device_selector);
  return sycl_queue;
}

static cl::sycl::context get_current_context() noexcept
{
  auto sycl_context = get_current_queue().get_context();
  return sycl_context;
}
#endif // USE_DPCPP (only)

static void clearGpuFreeList()
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
      cl::sycl::free(*it2, get_current_context());
#endif
    }
  }
  free_list_gpu.clear();
}

static void clearHostFreeList()
{
  for(map<size_t,set<void*> >::iterator it=free_list_host.begin(); it!=free_list_host.end(); ++it)
  {
    for(set<void*>::iterator it2=it->second.begin(); it2!=it->second.end(); ++it2)
    {
#if defined(USE_CUDA)
      cudaFreeHost(*it2);
#elif defined(USE_HIP)
      hipFreeHost(*it2);
#elif defined(USE_DPCPP)
      cl::sycl::free(*it2, get_current_context());
#endif // USE_CUDA
    }
  }
  free_list_host.clear();
}

static size_t num_resurrections=0;// num_morecore=0;

#if defined(USE_CUDA) || defined(USE_HIP)
typedef (*mallocfn_t)(void **ptr, size_t bytes);
static void *morecore(mallocfn_t fn, size_t bytes)
{
  void *ptr;
#if defined(USE_CUDA)
  CUDA_SAFE(fn((void **)&ptr, bytes));
#elif defined(USE_HIP)
  HIP_SAFE(fn((void **)&ptr, bytes));
#endif

  // num_morecore += 1;
  if(ptr==nullptr) {
    /*try one more time*/
    clearHostFreeList();
    clearGpuFreeList();
#if defined(USE_CUDA)
    CUDA_SAFE(fn((void **)&ptr, bytes));
#elif defined(USE_HIP)
    HIP_SAFE(fn((void **)&ptr, bytes));
#endif
  }
  assert(ptr!=nullptr); /*We hopefully have a pointer*/
  return ptr;
}
#endif

static inline void *resurrect_from_free_list(map<size_t,set<void *> > &free_map,
                                             size_t bytes,
                                             map<void*,size_t>& liveset)
{
  void *ptr;
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


void *getGpuMem(size_t bytes)
{
  //assert(is_init);
  void *ptr=nullptr;
#ifdef NO_OPT
#if defined(USE_CUDA)
  CUDA_SAFE(cudaMalloc((void **) &ptr, bytes));
#elif defined(USE_HIP)
  HIP_SAFE(hipMalloc((void **) &ptr, bytes));
#elif defiend(USE_DPCPP)
  ptr = (void *)cl::sycl::malloc_device(bytes, get_current_device(), get_current_context());
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

#if defined(USE_CUDA)
  ptr = morecore(cudaMalloc, bytes);
#elif defined(USE_HIP)
  ptr = morecore(hipMalloc, bytes);
#elif defined(USE_DPCPP)
  ptr = cl::sycl::malloc_device(bytes, get_current_device(), get_current_context());
  if(ptr==nullptr) { /*try one more time*/
    clearHostFreeList();
    clearGpuFreeList();
    ptr = cl::sycl::malloc_device(bytes, get_current_device(), get_current_context());
  }
  assert(ptr!=nullptr); /*We hopefully have a pointer*/
#endif

  live_ptrs_gpu[ptr] = bytes;
#endif // NO_OPT
  return ptr;
}

void *getHostMem(size_t bytes)
{
  //assert(is_init);
  void *ptr;
#ifdef NO_OPT
  #if defined(USE_CUDA)
  CUDA_SAFE(cudaMallocHost((void **) &ptr, bytes));
#elif defined(USE_HIP)
  HIP_SAFE(hipHostMalloc((void **) &ptr, bytes));
#elif defiend(USE_DPCPP)
  ptr = (void *)cl::sycl::malloc_host(bytes, get_current_context());
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

#if defined(USE_CUDA)
  ptr = morecore(cudaMallocHost, bytes);
#elif defined(USE_HIP)
  ptr = morecore(hipHostMalloc, bytes);
#elif defined(USE_DPCPP)
  ptr = cl::sycl::malloc_host(bytes, get_current_context());
  if(ptr==nullptr) { /*try one more time*/
    clearHostFreeList();
    clearGpuFreeList();
    ptr = cl::sycl::malloc_host(bytes, get_current_context());
  }
  assert(ptr!=nullptr); /*We hopefully have a pointer*/
#endif

  live_ptrs_host[ptr] = bytes;
#endif // NO_OPT
  return ptr;
}

void freeHostMem(void *p)
{
  size_t bytes;
  //assert(is_init);
#ifdef NO_OPT
#if defined(USE_CUDA)
  cudaFreeHost(p);
#elif defined(USE_HIP)
  hipHostFree(p);
#elif defined(USE_DPCPP)
  cl::sycl::free(p, get_current_context());
#endif

#else
  assert(live_ptrs_host.find(p) != live_ptrs_host.end());
  bytes = live_ptrs_host[p];
  live_ptrs_host.erase(p);
  free_list_host[bytes].insert(p);
#endif //NO_OPT
}

void freeGpuMem(void *p)
{
  size_t bytes;
  //assert(is_init);
#ifdef NO_OPT
#if defined(USE_CUDA)
  cudaFree(p);
#elif defined(USE_HIP)
  hipFree(p);
#elif defined(USE_DPCPP)
  cl::sycl::free(p, get_current_context());
#endif //NO_OPT

#else
  assert(live_ptrs_gpu.find(p) != live_ptrs_gpu.end());
  bytes = live_ptrs_gpu[p];
  live_ptrs_gpu.erase(p);
  free_list_gpu[bytes].insert(p);
#endif
}

void finalizememmodule()
{
  //assert(is_init);
  //is_init = 0;

  /*there should be no live pointers*/
  assert(live_ptrs_gpu.size()==0);
  assert(live_ptrs_host.size()==0);

  /*release all freed pointers*/
  clearGpuFreeList();
  clearHostFreeList();
  //printf("num. resurrections=%d \t num. morecore=%d\n", num_resurrections, num_morecore);
  // }
}
