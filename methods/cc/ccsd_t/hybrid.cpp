/*------------------------------------------hybrid execution------------*/
/* $Id$ */

#include "ccsd_t_common.hpp"
#include "ga/ga-mpi.h"
#include "ga/ga.h"
#include "ga/typesf2c.h"
#include "mpi.h"
#include <assert.h>
#include <cmath>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>

#ifdef USE_UPCXX
#include <upcxx/upcxx.hpp>
#endif

//
int util_my_smp_index(){
#ifdef USE_UPCXX
  int ppn = upcxx::local_team().rank_n();
  return upcxx::rank_me() % ppn;
#else
  auto ppn = GA_Cluster_nprocs(0);
  return GA_Nodeid() % ppn;
#endif
}

//
//
//
#define NUM_RANKS_PER_GPU 1

std::string check_memory_req(const int nDevices, const int cc_t_ts, const int nbf) {
  int dev_count_check = 0;
  if(nDevices <= 0) return "";
  double      global_gpu_mem = 0;
  std::string errmsg         = "";

#if defined(USE_CUDA)
  CUDA_SAFE(cudaGetDeviceCount(&dev_count_check));
  cudaDeviceProp gpu_properties;
  CUDA_SAFE(cudaGetDeviceProperties(&gpu_properties, 0));
  global_gpu_mem = gpu_properties.totalGlobalMem;
#elif defined(USE_HIP)
  HIP_SAFE(hipGetDeviceCount(&dev_count_check));
  hipDeviceProp_t gpu_properties;
  HIP_SAFE(hipGetDeviceProperties(&gpu_properties, 0));
  global_gpu_mem = gpu_properties.totalGlobalMem;
#elif defined(USE_DPCPP)
  syclGetDeviceCount(&dev_count_check);
  global_gpu_mem = sycl_get_device(0)->get_info<sycl::info::device::global_mem_size>();
#endif

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  if(dev_count_check < nDevices) {
    errmsg = "ERROR: Please check whether you have " + std::to_string(nDevices) +
             " sycl devices per node and set the ngpu option accordingly";
  }
#endif

  const double gpu_mem_req =
    (9.0 * (std::pow(cc_t_ts, 2) + std::pow(cc_t_ts, 4) + 2 * 2 * nbf * std::pow(cc_t_ts, 3)) * 8);
  int gpu_mem_check = 0;
  if(gpu_mem_req >= global_gpu_mem) gpu_mem_check = 1;
  if(gpu_mem_check) {
    const double gib = 1024 * 1024 * 1024.0;
    errmsg = "ERROR: GPU memory not sufficient for (T) calculation, available memory per gpu: " +
             std::to_string(global_gpu_mem / gib) +
             " GiB, required: " + std::to_string(gpu_mem_req / gib) +
             " GiB. Please set a smaller tilesize and retry";
  }

  return errmsg;
}

int check_device(long iDevice) {
  /* Check whether this process is associated with a GPU */
  // printf ("[%s] util_my_smp_index(): %d\n", __func__, util_my_smp_index());
  if((util_my_smp_index()) < iDevice * NUM_RANKS_PER_GPU) return 1;
  return 0;
}
