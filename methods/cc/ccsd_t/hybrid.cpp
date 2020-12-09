/*------------------------------------------hybrid execution------------*/
/* $Id$ */
#include <assert.h>
///#define NUM_DEVICES 1
static long long device_id=-1;
#include <stdio.h>
#include <stdlib.h>
#include "ccsd_t_common.hpp"
#include "mpi.h"
#include "ga.h"
#include "ga-mpi.h"
#include "typesf2c.h"

//
int util_my_smp_index(){
  auto ppn = GA_Cluster_nprocs(0);
  return GA_Nodeid()%ppn;
}

//
//
//
#define NUM_RANKS_PER_GPU   1


int check_device(long iDevice) {
  /* Check whether this process is associated with a GPU */
  // printf ("[%s] util_my_smp_index(): %d\n", __func__, util_my_smp_index());
  if((util_my_smp_index()) < iDevice * NUM_RANKS_PER_GPU) return 1;
  return 0;
}

int device_init(
#if defined(USE_DPCPP)
		const std::vector<cl::sycl::queue*> iDevice_syclQueue,
		cl::sycl::queue **syclQue,
#endif
                long iDevice,int *gpu_device_number) {

  /* Set device_id */
  int dev_count_check = 0;
#if defined(USE_CUDA)
  cudaGetDeviceCount(&dev_count_check);
#elif defined(USE_HIP)
  hipGetDeviceCount(&dev_count_check);
#elif defined(USE_DPCPP)
  cl::sycl::gpu_selector device_selector;
  cl::sycl::platform platform(device_selector);
  auto const& gpu_devices = platform.get_devices();

  for (const auto &dev : gpu_devices) {
      if (dev.is_gpu()) {
#ifdef TAMM_INTEL_ATS
          auto SubDevicesDomainNuma = dev.create_sub_devices<cl::sycl::info::partition_property::partition_by_affinity_domain>(cl::sycl::info::partition_affinity_domain::numa);
          for (const auto &tile : SubDevicesDomainNuma) {
              dev_count_check++;
          }
#else
      dev_count_check = gpu_devices.size();
#endif
      }
  }
#endif

  //
  device_id = util_my_smp_index();
  int actual_device_id = device_id % dev_count_check;

  // printf ("[%s] device_id: %lld (%d), dev_count_check: %d, iDevice: %ld\n", __func__, device_id, actual_device_id, dev_count_check, iDevice);

  if(dev_count_check < iDevice) {
    printf("Warning: Please check whether you have %ld devices per node\n",iDevice);
    fflush(stdout);
    *gpu_device_number = 30;
  }
  else {
#if defined(USE_CUDA)
    // cudaSetDevice(device_id);
    cudaSetDevice(actual_device_id);
#elif defined(USE_HIP)
    // hipSetDevice(device_id);
    hipSetDevice(actual_device_id);
#elif defined(USE_DPCPP)
    *syclQue = iDevice_syclQueue[actual_device_id];
#endif
  }
  return 1;
}
