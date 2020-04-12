
#ifndef CCSD_T_FUSED_HPP_
#define CCSD_T_FUSED_HPP_

#include "ccsd_t_all_fused.hpp"
#include "ccsd_t_common.hpp"

int check_device(long);
int device_init(long icuda,int *cuda_device_number );
void dev_release();
void finalizememmodule();

// 
// 
// 
template<typename T>
std::tuple<double,double> ccsd_t_fused_driver_new(SystemData& sys_data, ExecutionContext& ec,
                   std::vector<int>& k_spin,
                   const TiledIndexSpace& MO,
                   Tensor<T>& d_t1, Tensor<T>& d_t2,
                   Tensor<T>& d_v2,
                   std::vector<T>& k_evl_sorted,
                   double hf_ccsd_energy, int icuda,
                   bool is_restricted,
                   LRUCache<Index,std::vector<T>>& cache_s1t, LRUCache<Index,std::vector<T>>& cache_s1v,
                   LRUCache<Index,std::vector<T>>& cache_d1t, LRUCache<Index,std::vector<T>>& cache_d1v,
                   LRUCache<Index,std::vector<T>>& cache_d2t, LRUCache<Index,std::vector<T>>& cache_d2v,
                   bool seq_h3b=false) 
{
  // 
  auto rank     = ec.pg().rank().value();
  bool nodezero = rank==0;

  Index noab=MO("occ").num_tiles();
  Index nvab=MO("virt").num_tiles();

  auto mo_tiles = MO.input_tile_sizes();
  std::vector<size_t> k_range;
  std::vector<size_t> k_offset;
  size_t sum = 0;
  for (auto x: mo_tiles){
    k_range.push_back(x);
    k_offset.push_back(sum);
    sum+=x;
  } 

  if(nodezero){
    cout << "noab,nvab = " << noab << ", " << nvab << endl;
    // cout << "k_spin = " << k_spin << endl;
    // cout << "k_range = " << k_range << endl;
    cout << "MO Tiles = " << mo_tiles << endl;
  }

  //Check if node has number of devices specified in input file
  int dev_count_check;
  cudaGetDeviceCount(&dev_count_check);
  if(dev_count_check < icuda){
    if(nodezero) cout << "ERROR: Please check whether you have " << icuda <<
      " cuda devices per node. Terminating program..." << endl << endl;
    return std::make_tuple(-999,-999);
  }
  
  int cuda_device_number=0;
  //Check whether this process is associated with a GPU
  auto has_GPU = check_device(icuda);

  // printf ("[%s] rank: %d, has_GPU: %d, icuda: %d\n", __func__, rank, has_GPU, icuda);

  if(icuda==0) has_GPU=0;
  // cout << "rank,has_gpu" << rank << "," << has_GPU << endl;
  if(has_GPU == 1){
    device_init(icuda, &cuda_device_number);
    // if(cuda_device_number==30) // QUIT
  }
  if(nodezero) std::cout << "Using " << icuda << " gpu devices per node" << endl << endl;
  //std::cout << std::flush;

  //TODO replicate d_t1 L84-89 ccsd_t_gpu.F

  double energy1 = 0.0;
  double energy2 = 0.0;
  std::vector<double> energy_l(2);
  std::vector<double> tmp_energy_l(2);

  AtomicCounter* ac = new AtomicCounterGA(ec.pg(), 1);
  ac->allocate(0);
  int64_t taskcount = 0;
  int64_t next = ac->fetch_add(0, 1);

  size_t max_pdim = 0;
  size_t max_hdim = 0;
  for (size_t t_p4b=noab; t_p4b < noab + nvab; t_p4b++) 
    max_pdim = std::max(max_pdim,k_range[t_p4b]);
  for (size_t t_h1b = 0; t_h1b < noab; t_h1b++) 
    max_hdim = std::max(max_hdim,k_range[t_h1b]);

  // 
  size_t size_T_s1_t1 = 9 * (max_pdim) * (max_hdim);
  size_t size_T_s1_v2 = 9 * (max_pdim * max_pdim) * (max_hdim * max_hdim);
  size_t size_T_d1_t2 = 9 * noab * (max_pdim * max_pdim) * (max_hdim * max_hdim);
  size_t size_T_d1_v2 = 9 * noab * (max_pdim) * (max_hdim * max_hdim * max_hdim);
  size_t size_T_d2_t2 = 9 * nvab * (max_pdim * max_pdim) * (max_hdim * max_hdim);
  size_t size_T_d2_v2 = 9 * nvab * (max_pdim * max_pdim * max_pdim) * (max_hdim);

  // printf ("[%s][start] the nested loops by rank: %d------------------------\n", __func__, rank);
  int num_task = 0;
  if(!seq_h3b) 
  {
    for (size_t t_p4b = noab; t_p4b < noab + nvab; t_p4b++) {
    for (size_t t_p5b = t_p4b; t_p5b < noab + nvab; t_p5b++) {
    for (size_t t_p6b = t_p5b; t_p6b < noab + nvab; t_p6b++) {
    for (size_t t_h1b = 0; t_h1b < noab; t_h1b++) { //    
    for (size_t t_h2b = t_h1b; t_h2b < noab; t_h2b++) {
    for (size_t t_h3b = t_h2b; t_h3b < noab; t_h3b++) {
      if ((k_spin[t_p4b] + k_spin[t_p5b] + k_spin[t_p6b]) ==
          (k_spin[t_h1b] + k_spin[t_h2b] + k_spin[t_h3b])) {
        if ((k_spin[t_p4b] + k_spin[t_p5b] + k_spin[t_p6b] +
            k_spin[t_h1b] + k_spin[t_h2b] + k_spin[t_h3b]) <= 8) {   
          if (next == taskcount) {            
            // if (has_GPU==1) {
            //   initmemmodule();
            // }
            
            double factor = 2.0;
            if ((t_p4b == t_p5b) && (t_p5b == t_p6b)) {
              factor /= 6.0;
            } else if ((t_p4b == t_p5b) || (t_p5b == t_p6b)) {
              factor /= 2.0;
            }

            if ((t_h1b == t_h2b) && (t_h2b == t_h3b)) {
              factor /= 6.0;
            } else if ((t_h1b == t_h2b) || (t_h2b == t_h3b)) {
              factor /= 2.0;
            }

            num_task++;
            // printf ("[%s] rank: %d >> calls the gpu code\n", __func__, rank);
            ccsd_t_fully_fused_none_df_none_task(noab, nvab, rank, 
                                                k_spin,
                                                k_range,
                                                k_offset,
                                                d_t1, d_t2, d_v2, 
                                                k_evl_sorted,
                                                // 
                                                t_h1b, t_h2b, t_h3b,
                                                t_p4b, t_p5b, t_p6b,
                                                factor,
                                                //  
                                                size_T_s1_t1, size_T_s1_v2, 
                                                size_T_d1_t2, size_T_d1_v2, 
                                                size_T_d2_t2, size_T_d2_v2, 
                                                // 
                                                tmp_energy_l, 
                                                cache_s1t, cache_s1v,
                                                cache_d1t, cache_d1v,
                                                cache_d2t, cache_d2v);
            
            next = ac->fetch_add(0, 1); 
          }            
          taskcount++;
        }
      }
    }}}}}}
  } // parallel h3b loop
  else 
  { //seq h3b loop
    for (size_t t_p4b = noab; t_p4b < noab + nvab; t_p4b++) {
    for (size_t t_p5b = t_p4b; t_p5b < noab + nvab; t_p5b++) {
    for (size_t t_p6b = t_p5b; t_p6b < noab + nvab; t_p6b++) {
    for (size_t t_h1b = 0; t_h1b < noab; t_h1b++) {
    for (size_t t_h2b = t_h1b; t_h2b < noab; t_h2b++) {
      if (next == taskcount) {
        // if (has_GPU==1) {
        //   initmemmodule();
        // }
        for (size_t t_h3b = t_h2b; t_h3b < noab; t_h3b++) {
          if ((k_spin[t_p4b] + k_spin[t_p5b] + k_spin[t_p6b]) ==
              (k_spin[t_h1b] + k_spin[t_h2b] + k_spin[t_h3b])) {
            if ((k_spin[t_p4b] + k_spin[t_p5b] + k_spin[t_p6b] +
                 k_spin[t_h1b] + k_spin[t_h2b] + k_spin[t_h3b]) <= 8) 
            {
              double factor = 2.0;

              // 
              if ((t_p4b == t_p5b) && (t_p5b == t_p6b)) {
                factor /= 6.0;
              } else if ((t_p4b == t_p5b) || (t_p5b == t_p6b)) {
                factor /= 2.0;
              }

              if ((t_h1b == t_h2b) && (t_h2b == t_h3b)) {
                factor /= 6.0;
              } else if ((t_h1b == t_h2b) || (t_h2b == t_h3b)) {
                factor /= 2.0;
              }

              // 
              num_task++;
              // printf ("[%s] rank: %d >> calls the gpu code\n", __func__, rank);
              ccsd_t_fully_fused_none_df_none_task(noab, nvab, rank, 
                                                  k_spin,
                                                  k_range,
                                                  k_offset,
                                                  d_t1, d_t2, d_v2, 
                                                  k_evl_sorted,
                                                  // 
                                                  t_h1b, t_h2b, t_h3b,
                                                  t_p4b, t_p5b, t_p6b,
                                                  factor,
                                                  //  
                                                  size_T_s1_t1, size_T_s1_v2, 
                                                  size_T_d1_t2, size_T_d1_v2, 
                                                  size_T_d2_t2, size_T_d2_v2, 
                                                  // 
                                                  tmp_energy_l, 
                                                  cache_s1t, cache_s1v,
                                                  cache_d1t, cache_d1v,
                                                  cache_d2t, cache_d2v);
            }
          }
        }//h3b
        // finalizememmodule();
        next = ac->fetch_add(0, 1); 
      }
      taskcount++;
    }}}}}
  } //end seq h3b 
  // printf ("[%s][end] the nested loops -----------------------------\n", __func__);

  energy1 = tmp_energy_l[0];
  energy2 = tmp_energy_l[1];

  // 
  finalizememmodule();

  // printf ("[%s] rank: %d, # tasks: %d\n", __func__, rank, num_task);
  
  // 
  next = ac->fetch_add(0, 1); 
  ec.pg().barrier ();
  ac->deallocate();
  delete ac;

  return std::make_tuple(energy1,energy2);
}


// 
//  the existing driver
// 
template<typename T>
std::tuple<double,double> ccsd_t_fused_driver(SystemData& sys_data, ExecutionContext& ec,
                   std::vector<int>& k_spin,
                   const TiledIndexSpace& MO,
                   Tensor<T>& d_t1, Tensor<T>& d_t2,
                   Tensor<T>& d_v2,
                   std::vector<T>& k_evl_sorted,
                   double hf_ccsd_energy, int icuda,
                   bool is_restricted,
                   LRUCache<Index,std::vector<T>>& cache_s1t, LRUCache<Index,std::vector<T>>& cache_s1v,
                   LRUCache<Index,std::vector<T>>& cache_d1t, LRUCache<Index,std::vector<T>>& cache_d1v,
                   LRUCache<Index,std::vector<T>>& cache_d2t, LRUCache<Index,std::vector<T>>& cache_d2v,
                   bool seq_h3b=false) {

  //  
  cudaEvent_t start_before, stop_before;
  cudaEvent_t start_ccsd_t, stop_ccsd_t;
  cudaEvent_t start_after, stop_after;
  float time_ms_ccsd_t = 0.0;
  float time_ms_before = 0.0;
  float time_ms_after = 0.0;
  cudaEventCreate(&start_ccsd_t); cudaEventCreate(&stop_ccsd_t);
  cudaEventCreate(&start_before); cudaEventCreate(&stop_before);
  cudaEventCreate(&start_after); cudaEventCreate(&stop_after);

  // 
  cudaEventRecord(start_before);

  auto rank = ec.pg().rank().value();
  bool nodezero = rank==0;

  // if(icuda==0) {
  //   if(nodezero)std::cout << "\nERROR: Please specify number of cuda devices to use in the input file!\n\n"; //TODO
  //   return std::make_tuple(-999,-999);
  // }

  Index noab=MO("occ").num_tiles();
  Index nvab=MO("virt").num_tiles();

  auto mo_tiles = MO.input_tile_sizes();
  std::vector<size_t> k_range;
  std::vector<size_t> k_offset;
  size_t sum = 0;
  for (auto x: mo_tiles){
    k_range.push_back(x);
    k_offset.push_back(sum);
    sum+=x;
  } 

  if(nodezero){
    cout << "noab,nvab = " << noab << ", " << nvab << endl;
    // cout << "k_spin = " << k_spin << endl;
    // cout << "k_range = " << k_range << endl;
    cout << "MO Tiles = " << mo_tiles << endl;
  }

  //Check if node has number of devices specified in input file
  int dev_count_check;
  cudaGetDeviceCount(&dev_count_check);
  if(dev_count_check < icuda){
    if(nodezero) cout << "ERROR: Please check whether you have " << icuda <<
      " cuda devices per node. Terminating program..." << endl << endl;
    return std::make_tuple(-999,-999);
  }
  
  int cuda_device_number=0;
  //Check whether this process is associated with a GPU
  auto has_GPU = check_device(icuda);
  if(icuda==0) has_GPU=0;
  // cout << "rank,has_gpu" << rank << "," << has_GPU << endl;
  if(has_GPU == 1){
    device_init(icuda, &cuda_device_number);
    // if(cuda_device_number==30) // QUIT
  }
  if(nodezero) std::cout << "Using " << icuda << " gpu devices per node" << endl << endl;
  //std::cout << std::flush;

  //TODO replicate d_t1 L84-89 ccsd_t_gpu.F

  double energy1 = 0.0;
  double energy2 = 0.0;
  std::vector<double> energy_l(2);

  AtomicCounter* ac = new AtomicCounterGA(ec.pg(), 1);
  ac->allocate(0);
  int64_t taskcount = 0;
  int64_t next = ac->fetch_add(0, 1);

  size_t max_pdim = 0;
  size_t max_hdim = 0;
  for (size_t t_p4b=noab; t_p4b < noab + nvab; t_p4b++) 
    max_pdim = std::max(max_pdim,k_range[t_p4b]);
  for (size_t t_h1b = 0; t_h1b < noab; t_h1b++) 
    max_hdim = std::max(max_hdim,k_range[t_h1b]);

  size_t abuf_size1 = 9 * noab * (max_pdim*max_pdim) * (max_hdim*max_hdim);
  size_t bbuf_size1 = 9 * noab * max_pdim * (max_hdim*max_hdim*max_hdim);
  size_t abuf_size2 = 9 * nvab * (max_pdim*max_pdim) * (max_hdim * max_hdim);
  size_t bbuf_size2 = 9 * nvab * (max_pdim*max_pdim*max_pdim) * max_hdim;

  std::vector<T> k_abuf1; 
  std::vector<T> k_bbuf1; 
  std::vector<T> k_abuf2; 
  std::vector<T> k_bbuf2; 
  std::vector<T> k_abufs1;
  std::vector<T> k_bbufs1;

  k_abufs1.resize(9*max_pdim*max_hdim);
  k_bbufs1.resize(9 * (max_pdim*max_pdim) * (max_hdim*max_hdim));

  k_abuf1.resize(abuf_size1);
  k_bbuf1.resize(bbuf_size1);
  k_abuf2.resize(abuf_size2);
  k_bbuf2.resize(bbuf_size2);

#if 1
  size_t size_T_s1_t1 = 9 * (max_pdim) * (max_hdim);
  size_t size_T_s1_v2 = 9 * (max_pdim * max_pdim) * (max_hdim * max_hdim);
  size_t size_T_d1_t2 = 9 * noab * (max_pdim * max_pdim) * (max_hdim * max_hdim);
  size_t size_T_d1_v2 = 9 * noab * (max_pdim) * (max_hdim * max_hdim * max_hdim);
  size_t size_T_d2_t2 = 9 * nvab * (max_pdim * max_pdim) * (max_hdim * max_hdim);
  size_t size_T_d2_v2 = 9 * nvab * (max_pdim * max_pdim * max_pdim) * (max_hdim);
#endif

  // temporally # of streams = 60.
  int max_streams = 60;
  int stream_id = 0;
  int gpu_devid = ec.gpu_devid();
  cudaStream_t streams[max_streams];

  for (int i = 0; i < max_streams; i++)
  cudaStreamCreate(&streams[i]);

  // 
  cudaEventRecord(stop_before);
  cudaEventSynchronize(stop_before);
  cudaEventRecord(start_ccsd_t);

  // 
  if(!seq_h3b) 
  {
    for (size_t t_p4b = noab; t_p4b < noab + nvab; t_p4b++) {
    for (size_t t_p5b = t_p4b; t_p5b < noab + nvab; t_p5b++) {
    for (size_t t_p6b = t_p5b; t_p6b < noab + nvab; t_p6b++) {
    for (size_t t_h1b = 0; t_h1b < noab; t_h1b++) { //    
    for (size_t t_h2b = t_h1b; t_h2b < noab; t_h2b++) {
    for (size_t t_h3b = t_h2b; t_h3b < noab; t_h3b++) {
      if ((k_spin[t_p4b] + k_spin[t_p5b] + k_spin[t_p6b]) ==
          (k_spin[t_h1b] + k_spin[t_h2b] + k_spin[t_h3b])) {
        if ((k_spin[t_p4b] + k_spin[t_p5b] + k_spin[t_p6b] +
            k_spin[t_h1b] + k_spin[t_h2b] + k_spin[t_h3b]) <= 8) {   
          if (next == taskcount) {            
            if (has_GPU==1) {
              initmemmodule();
            }
            
            double factor = 2.0;
            if ((t_p4b == t_p5b) && (t_p5b == t_p6b)) {
              factor /= 6.0;
            } else if ((t_p4b == t_p5b) || (t_p5b == t_p6b)) {
              factor /= 2.0;
            }

            if ((t_h1b == t_h2b) && (t_h2b == t_h3b)) {
              factor /= 6.0;
            } else if ((t_h1b == t_h2b) || (t_h2b == t_h3b)) {
              factor /= 2.0;
            }

            // printf ("[%s] (if) calls ccsd_t_all_fused(..)\n", __func__);
            ccsd_t_all_fused(ec,MO,noab,nvab,
              k_spin,k_offset,/*k_doubles,*/d_t1,d_t2,d_v2,
              k_evl_sorted,k_range,t_h1b,t_h2b,t_h3b,
              t_p4b,t_p5b,t_p6b, k_abufs1, k_bbufs1, 
              k_abuf1,k_bbuf1,k_abuf2,k_bbuf2,
              size_T_s1_t1, size_T_s1_v2, 
              size_T_d1_t2, size_T_d1_v2, 
              size_T_d2_t2, size_T_d2_v2, 
              // 
              &streams[stream_id++ % max_streams], gpu_devid, 
              // 
              factor,
              energy_l,has_GPU,is_restricted,
              cache_s1t,cache_s1v,cache_d1t,
              cache_d1v,cache_d2t,cache_d2v); 
            // printf ("[%s] E(4): %.10f (%.10f), E(5): %.10f (%.10f) by %d ---- h3,h2,h1,p6,p5,p4:%2d,%2d,%2d,%2d,%2d,%2d\n", __func__, energy_l[0], 
            // energy1, energy_l[1], energy2, omp_get_thread_num(), t_h3b, t_h2b, t_h1b, t_p6b, t_p5b, t_p4b);
                
            energy1 += energy_l[0];
            energy2 += energy_l[1];

            // dev_release();
            finalizememmodule();
            next = ac->fetch_add(0, 1); 
          }            
          taskcount++;
        }
      }
    }
    }
    }
    }
    }
    }
  } // parallel h3b loop
  else 
  { //seq h3b loop
    for (size_t t_p4b = noab; t_p4b < noab + nvab; t_p4b++) {
    for (size_t t_p5b = t_p4b; t_p5b < noab + nvab; t_p5b++) {
    for (size_t t_p6b = t_p5b; t_p6b < noab + nvab; t_p6b++) {
    for (size_t t_h1b = 0; t_h1b < noab; t_h1b++) {
    for (size_t t_h2b = t_h1b; t_h2b < noab; t_h2b++) {
      if (next == taskcount) {
        if (has_GPU==1) {
          initmemmodule();
        }
        for (size_t t_h3b = t_h2b; t_h3b < noab; t_h3b++) {
          if ((k_spin[t_p4b] + k_spin[t_p5b] + k_spin[t_p6b]) ==
              (k_spin[t_h1b] + k_spin[t_h2b] + k_spin[t_h3b])) {
            if ((k_spin[t_p4b] + k_spin[t_p5b] + k_spin[t_p6b] +
                 k_spin[t_h1b] + k_spin[t_h2b] + k_spin[t_h3b]) <= 8) 
            {
              double factor = 2.0;

              // 
              if ((t_p4b == t_p5b) && (t_p5b == t_p6b)) {
                factor /= 6.0;
              } else if ((t_p4b == t_p5b) || (t_p5b == t_p6b)) {
                factor /= 2.0;
              }

              if ((t_h1b == t_h2b) && (t_h2b == t_h3b)) {
                factor /= 6.0;
              } else if ((t_h1b == t_h2b) || (t_h2b == t_h3b)) {
                factor /= 2.0;
              }

              // printf ("[%s] (else) calls ccsd_t_all_fused(..)\n", __func__);
              ccsd_t_all_fused(ec,MO,noab,nvab,
                k_spin,k_offset,/*k_doubles,*/d_t1,d_t2,d_v2,
                k_evl_sorted,k_range,t_h1b,t_h2b,t_h3b,
                t_p4b,t_p5b,t_p6b, k_abufs1, k_bbufs1, 
                k_abuf1,k_bbuf1,k_abuf2,k_bbuf2,
                size_T_s1_t1, size_T_s1_v2, 
                size_T_d1_t2, size_T_d1_v2, 
                size_T_d2_t2, size_T_d2_v2, 
                // 
                &streams[stream_id++ % max_streams], gpu_devid, 
                // 
                factor,
                energy_l,has_GPU,is_restricted,
                cache_s1t,cache_s1v,cache_d1t,
                cache_d1v,cache_d2t,cache_d2v); 

              // printf ("[%s] E(4): %f, E(5: %f\n", __func__, energy_l[0], energy_l[1]);

                  
              energy1 += energy_l[0];
              energy2 += energy_l[1];
            }
          }
        }//h3b
        finalizememmodule();
        next = ac->fetch_add(0, 1); 
      }
      taskcount++;
    }
    }
    }
    }
    }
  } //end seq h3b 
  // printf ("[%s][end] the nested loops -----------------------------\n", __func__);

  // 
  cudaEventRecord(stop_ccsd_t);
  cudaEventSynchronize(stop_ccsd_t);
  cudaEventRecord(start_after);

  // 
  // finalizememmodule();
  k_abuf1.clear();
  k_abuf2.clear();
  k_bbuf1.clear();
  k_bbuf2.clear();
  k_abufs1.clear();
  k_bbufs1.clear();

  for (int i = 0; i < 60; i++)
  cudaStreamDestroy(streams[i]);


  next = ac->fetch_add(0, 1); 
  ec.pg().barrier();
  ac->deallocate();
  delete ac;

  cudaEventRecord(stop_after);
  cudaEventSynchronize(stop_after);

  cudaEventElapsedTime(&time_ms_ccsd_t, start_ccsd_t, stop_ccsd_t);
  cudaEventElapsedTime(&time_ms_before, start_before, stop_before);
  cudaEventElapsedTime(&time_ms_after, start_after, stop_after);
  printf ("[%s] before: %f, ccsd_t: %f, after: %f >> total: %f\n", __func__, time_ms_before, time_ms_ccsd_t, time_ms_after, time_ms_before + time_ms_ccsd_t + time_ms_after);

  return std::make_tuple(energy1,energy2);
}
#endif //CCSD_T_FUSED_HPP_
