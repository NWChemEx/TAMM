#include "ccsd_t_common.hpp"
#include <map>
#include <set>
using namespace std;

// #define NO_OPT

// extern "C" {

// static int is_init=0;

static map<size_t,set<void*> > free_list_gpu, free_list_host;
static map<void*,size_t> live_ptrs_gpu, live_ptrs_host;

static void clearGpuFreeList()
{
  for(map<size_t,set<void*> >::iterator it=free_list_gpu.begin(); it!=free_list_gpu.end(); ++it)
  {
    for(set<void*>::iterator it2=it->second.begin(); it2!=it->second.end(); ++it2)
    {
      delete *it2;
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
      cl::sycl::free(*it2, syclContext());
    }
  }
  free_list_host.clear();
}

static size_t num_resurrections=0;// num_morecore=0;

typedef cudaError (*mallocfn_t)(void **ptr, size_t bytes);
static void *morecore(mallocfn_t fn, size_t bytes)
{
  cl::sycl::buffer<T,1>* buf;
  buf = new cl::sycl::buffer<T,1>(N);

  // num_morecore += 1;
  if(buf==nullptr) {
    /*try one more time*/
    clearHostFreeList();
    clearGpuFreeList();
    buf = new cl::sycl::buffer<T,1>(N);
    //fn((void **)&ptr, bytes);
  }
  assert(buf != nullptr); /*We hopefully have a pointer*/
  return (void *)buf;
}

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

template <typename T>
cl::sycl::buffer<T,1>* getGpuMem(size_t numArray)
{
  //assert(is_init);

  cl::sycl::buffer<T, 1> *buf;

#ifdef NO_OPT
  buf = new cl::sycl::buffer<T, 1>(numArray);
#else
  size_t bytes = numArray * sizeof(T);
  if(free_list_gpu.find(bytes)!=free_list_gpu.end())
  {
    set<void*> &lst = free_list_gpu.find(bytes)->second;
    if(lst.size()!=0)
    {
      buf = (cl::sycl::buffer<T,1>*)resurrect_from_free_list(free_list_gpu, bytes, live_bufs_gpu);
      return buf;
    }
  }
  else
  {
    for(map<size_t,set<void *> >::iterator it=free_list_gpu.begin(); it != free_list_gpu.end(); ++it)
    {
      if(it->first >= bytes && it->second.size()>0)
      {
        ptr = (cl::sycl::buffer<T,1>*)resurrect_from_free_list(free_list_gpu, it->first, live_ptrs_gpu);
        return ptr;
      }
    }
  }
  ptr = morecore(malloc_device, bytes);
  live_ptrs_gpu[(void *)ptr] = bytes;
#endif
  return ptr;
}

void *getHostMem(size_t bytes)
{
  //assert(is_init);
  void *ptr;
#ifdef NO_OPT
  ptr = cl::sycl::malloc_host(bytes, syclDevice(), syclContext());
#else
  if(free_list_host.find(bytes)!=free_list_host.end())
  {
    set<void*> &lst = free_list_host.find(bytes)->second;
    if(lst.size()!=0)
    {
      ptr = resurrect_from_free_list(free_list_host, bytes, live_ptrs_host);
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
            return ptr;
          }
      }
  }
  ptr = morecore(malloc_host, bytes);
  live_ptrs_host[ptr] = bytes;
#endif
  return ptr;
}

void freeHostMem(void *p)
{
  size_t bytes;
  //assert(is_init);
#ifdef NO_OPT
  cl::sycl::free(p, syclContext());
#else
  assert(live_ptrs_host.find(p) != live_ptrs_host.end());
  bytes = live_ptrs_host[p];
  live_ptrs_host.erase(p);
  free_list_host[bytes].insert(p);
#endif
}

void freeGpuMem(void *p)
{
  size_t bytes;
  //assert(is_init);
#ifdef NO_OPT
  cl::sycl::free(p, syclContext());
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
