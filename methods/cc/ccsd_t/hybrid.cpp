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


int check_device(long icuda) {
  /* Check whether this process is associated with a GPU */
  // printf ("[%s] util_my_smp_index(): %d\n", __func__, util_my_smp_index());
  if((util_my_smp_index()) < icuda * NUM_RANKS_PER_GPU) return 1;
  return 0;
}

int device_init(long icuda,int *cuda_device_number) {
  /* Set device_id */
  int dev_count_check = 0;
  cudaGetDeviceCount(&dev_count_check);

  // 
  device_id = util_my_smp_index();


  int actual_device_id = device_id % dev_count_check;
  
  // printf ("[%s] device_id: %lld (%d), dev_count_check: %d, icuda: %ld\n", __func__, device_id, actual_device_id, dev_count_check, icuda);

  if(dev_count_check < icuda){
    printf("Warning: Please check whether you have %ld cuda devices per node\n",icuda);
    fflush(stdout);
    *cuda_device_number = 30;
  }
  else {
    // cudaSetDevice(device_id);
    cudaSetDevice(actual_device_id);
  }
  return 1;
}

