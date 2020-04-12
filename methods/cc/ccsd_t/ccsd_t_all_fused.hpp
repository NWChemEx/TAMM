#ifndef CCSD_T_ALL_FUSED_HPP_
#define CCSD_T_ALL_FUSED_HPP_

#include "ccsd_t_common.hpp"
#include "ccsd_t_all_fused_singles.hpp"
#include "ccsd_t_all_fused_doubles1.hpp"
#include "ccsd_t_all_fused_doubles2.hpp"

void initmemmodule();
void dev_mem_s(size_t,size_t,size_t,size_t,size_t,size_t);
void dev_mem_d(size_t,size_t,size_t,size_t,size_t,size_t);

#define TEST_NEW_KERNEL
#define TEST_NEW_THREAD

#define CEIL(a, b)  (((a) + (b) - 1) / (b))


void fully_fused_ccsd_t_gpu(cudaStream_t* stream_id, size_t num_blocks, 
	size_t base_size_h1b, size_t base_size_h2b, size_t base_size_h3b, 
	size_t base_size_p4b, size_t base_size_p5b, size_t base_size_p6b,
	// 
	double* df_dev_d1_t2_all, double* df_dev_d1_v2_all,
	double* df_dev_d2_t2_all, double* df_dev_d2_v2_all,
	double* df_dev_s1_t1_all, double* df_dev_s1_v2_all,
	// 
	size_t size_d1_t2_all, size_t size_d1_v2_all,
	size_t size_d2_t2_all, size_t size_d2_v2_all,
	size_t size_s1_t1_all, size_t size_s1_v2_all,
	// 
	int* host_d1_size, int* host_d1_exec, 	// used
	int* host_d2_size, int* host_d2_exec, 
	int* host_s1_size, int* host_s1_exec, 
	// 
	size_t size_noab, size_t size_max_dim_d1_t2, size_t size_max_dim_d1_v2,
	size_t size_nvab, size_t size_max_dim_d2_t2, size_t size_max_dim_d2_v2,
										size_t size_max_dim_s1_t1, size_t size_max_dim_s1_v2, 
	// 
	double factor, 
	// 
	double* dev_evl_sorted_h1b, double* dev_evl_sorted_h2b, double* dev_evl_sorted_h3b, 
	double* dev_evl_sorted_p4b, double* dev_evl_sorted_p5b, double* dev_evl_sorted_p6b,
	double* partial_energies);

void total_fused_ccsd_t_gpu(cudaStream_t* stream_id, int gpu_id, 
            size_t base_size_h1b, size_t base_size_h2b, size_t base_size_h3b, 
						size_t base_size_p4b, size_t base_size_p5b, size_t base_size_p6b,
						// 
						double* host_d1_t2_all, double* host_d1_v2_all,
						double* host_d2_t2_all, double* host_d2_v2_all,
						double* host_s1_t2_all, double* host_s1_v2_all,
						// 
						size_t size_d1_t2_all, size_t size_d1_v2_all,
						size_t size_d2_t2_all, size_t size_d2_v2_all,
						size_t size_s1_t2_all, size_t size_s1_v2_all,
						// 
						int* list_d1_sizes, 
						int* list_d2_sizes, 
						int* list_s1_sizes, 
						// 
						std::vector<int> vec_d1_flags,
						std::vector<int> vec_d2_flags,
						std::vector<int> vec_s1_flags, 
						// 
						size_t size_noab, size_t size_max_dim_d1_t2, size_t size_max_dim_d1_v2,
						size_t size_nvab, size_t size_max_dim_d2_t2, size_t size_max_dim_d2_v2,
                                          size_t size_max_dim_s1_t2, size_t size_max_dim_s1_v2, 
						// 
						double factor, 
						double* host_evl_sortedh1, double* host_evl_sortedh2, double* host_evl_sortedh3, 
						double* host_evl_sortedp4, double* host_evl_sortedp5, double* host_evl_sortedp6,
						double* final_energy_4, double* final_energy_5);

void total_fused_ccsd_t_cpu(size_t base_size_h1b, size_t base_size_h2b, size_t base_size_h3b, 
                            size_t base_size_p4b, size_t base_size_p5b, size_t base_size_p6b,
                            // 
                            double* host_d1_t2_all, double* host_d1_v2_all,
                            double* host_d2_t2_all, double* host_d2_v2_all,
                            double* host_s1_t2_all, double* host_s1_v2_all,
                            // 
                            size_t size_d1_t2_all, size_t size_d1_v2_all,
                            size_t size_d2_t2_all, size_t size_d2_v2_all,
                            size_t size_s1_t2_all, size_t size_s1_v2_all,
                            // 
                            int* list_d1_sizes, 
                            int* list_d2_sizes, 
                            int* list_s1_sizes, 
                            // 
                            std::vector<int> vec_d1_flags,
                            std::vector<int> vec_d2_flags,
                            std::vector<int> vec_s1_flags, 
                            // 
                            size_t size_noab, size_t size_max_dim_d1_t2, size_t size_max_dim_d1_v2,
                            size_t size_nvab, size_t size_max_dim_d2_t2, size_t size_max_dim_d2_v2,
                                              size_t size_max_dim_s1_t2, size_t size_max_dim_s1_v2, 
                            // 
                            double factor, 
                            double* host_evl_sorted_h1, double* host_evl_sorted_h2, double* host_evl_sorted_h3, 
                            double* host_evl_sorted_p4, double* host_evl_sorted_p5, double* host_evl_sorted_p6,
                            double* final_energy_4, double* final_energy_5);

#define OPT_ALL_TIMING

template<typename T>
void ccsd_t_fully_fused_none_df_none_task(const Index noab, const Index nvab, auto rank, 
                                          std::vector<int>& k_spin,
                                          std::vector<size_t>& k_range,
                                          std::vector<size_t>& k_offset,
                                          Tensor<T>& d_t1, Tensor<T>& d_t2, Tensor<T>& d_v2, 
                                          std::vector<T>& k_evl_sorted,
                                          // 
                                          size_t t_h1b, size_t t_h2b, size_t t_h3b,
                                          size_t t_p4b, size_t t_p5b, size_t t_p6b,
                                          double factor,
                                          //  
                                          size_t size_T_s1_t1, size_t size_T_s1_v2, 
                                          size_t size_T_d1_t2, size_t size_T_d1_v2, 
                                          size_t size_T_d2_t2, size_t size_T_d2_v2, 
                                          // 
                                          std::vector<double>& energy_l, 
                                          LRUCache<Index,std::vector<T>>& cache_s1t, LRUCache<Index,std::vector<T>>& cache_s1v,
                                          LRUCache<Index,std::vector<T>>& cache_d1t, LRUCache<Index,std::vector<T>>& cache_d1v,
                                          LRUCache<Index,std::vector<T>>& cache_d2t, LRUCache<Index,std::vector<T>>& cache_d2v) 
{
#ifdef OPT_ALL_TIMING
  cudaEvent_t start_init,               stop_init;
  cudaEvent_t start_fused_kernel,       stop_fused_kernel;
  cudaEvent_t start_pre_processing,     stop_pre_processing;
  cudaEvent_t start_post_processing,    stop_post_processing;
  cudaEvent_t start_collecting_data,    stop_collecting_data;

  cudaEventCreate(&start_init);             cudaEventCreate(&stop_init);
  cudaEventCreate(&start_fused_kernel);     cudaEventCreate(&stop_fused_kernel);
  cudaEventCreate(&start_pre_processing);   cudaEventCreate(&stop_pre_processing);
  cudaEventCreate(&start_post_processing);  cudaEventCreate(&stop_post_processing); 
  cudaEventCreate(&start_collecting_data);  cudaEventCreate(&stop_collecting_data);

  //
  float time_ms_init              = 0.0;
  float time_ms_fused_kernel      = 0.0;
  float time_ms_pre_processing    = 0.0;
  float time_ms_post_processing   = 0.0;
  float time_ms_collecting_data   = 0.0;

  // 
  cudaEventRecord(start_init);
#endif

  cudaStream_t stream;

  // Index p4b,p5b,p6b,h1b,h2b,h3b;
  const size_t max_dim_s1_t1 = size_T_s1_t1 / 9;
  const size_t max_dim_s1_v2 = size_T_s1_v2 / 9;
  const size_t max_dim_d1_t2 = size_T_d1_t2 / (9 * noab);
  const size_t max_dim_d1_v2 = size_T_d1_v2 / (9 * noab);
  const size_t max_dim_d2_t2 = size_T_d2_t2 / (9 * nvab);
  const size_t max_dim_d2_v2 = size_T_d2_v2 / (9 * nvab);

  // 
  //  >> for pinned host memory (should support redundant calls)
  // 
  // 
  //  pinned host memory for s1 (t1, v2), d1 (t2, v2), and d2 (t2, v2)
  // 
  T* df_host_pinned_s1_t1;
  T* df_host_pinned_s1_v2;
  T* df_host_pinned_d1_t2;
  T* df_host_pinned_d1_v2;
  T* df_host_pinned_d2_t2;
  T* df_host_pinned_d2_v2;

  // 
  int*  df_simple_s1_size;   int*  df_simple_d1_size;   int* df_simple_d2_size;
  int*  df_simple_s1_exec;   int*  df_simple_d1_exec;   int* df_simple_d2_exec;
  int   df_num_s1_enabled;   int   df_num_d1_enabled;   int  df_num_d2_enabled;

  // 
  //  Host-Level
  //  to allocate host (pinned) memory for double-buffering (does not depend on a task)
  // 
  df_host_pinned_s1_t1 = (T*)getHostMem(sizeof(double) * size_T_s1_t1);
  df_host_pinned_s1_v2 = (T*)getHostMem(sizeof(double) * size_T_s1_v2);
  df_host_pinned_d1_t2 = (T*)getHostMem(sizeof(double) * size_T_d1_t2);
  df_host_pinned_d1_v2 = (T*)getHostMem(sizeof(double) * size_T_d1_v2);
  df_host_pinned_d2_t2 = (T*)getHostMem(sizeof(double) * size_T_d2_t2);
  df_host_pinned_d2_v2 = (T*)getHostMem(sizeof(double) * size_T_d2_v2);

  // 
  df_simple_s1_size = (int*)getHostMem(sizeof(int) * (6));
  df_simple_s1_exec = (int*)getHostMem(sizeof(int) * (9));

  df_simple_d1_size = (int*)getHostMem(sizeof(int) * (7 * noab));
  df_simple_d1_exec = (int*)getHostMem(sizeof(int) * (9 * noab));

  df_simple_d2_size = (int*)getHostMem(sizeof(int) * (7 * nvab));
  df_simple_d2_exec = (int*)getHostMem(sizeof(int) * (9 * nvab));

  // 
  cudaStreamCreate(&stream);

  // 
  //  variables based on a current task
  // 
  // 
  //  Device-Level
  // 
  double* df_dev_s1_t1_all = (double*)getGpuMem(sizeof(double) * size_T_s1_t1);
	double* df_dev_s1_v2_all = (double*)getGpuMem(sizeof(double) * size_T_s1_v2);
  double* df_dev_d1_t2_all = (double*)getGpuMem(sizeof(double) * size_T_d1_t2);
	double* df_dev_d1_v2_all = (double*)getGpuMem(sizeof(double) * size_T_d1_v2);
  double* df_dev_d2_t2_all = (double*)getGpuMem(sizeof(double) * size_T_d2_t2);
	double* df_dev_d2_v2_all = (double*)getGpuMem(sizeof(double) * size_T_d2_v2);


  // 
  size_t base_size_h1b = k_range[t_h1b];
  size_t base_size_h2b = k_range[t_h2b];
  size_t base_size_h3b = k_range[t_h3b];
  size_t base_size_p4b = k_range[t_p4b];
  size_t base_size_p5b = k_range[t_p5b];
  size_t base_size_p6b = k_range[t_p6b];

  // 
  //  Host-Level
  // 
  double* host_evl_sorted_h1b = &k_evl_sorted[k_offset[t_h1b]];
  double* host_evl_sorted_h2b = &k_evl_sorted[k_offset[t_h2b]];
  double* host_evl_sorted_h3b = &k_evl_sorted[k_offset[t_h3b]];
  double* host_evl_sorted_p4b = &k_evl_sorted[k_offset[t_p4b]];
  double* host_evl_sorted_p5b = &k_evl_sorted[k_offset[t_p5b]];
  double* host_evl_sorted_p6b = &k_evl_sorted[k_offset[t_p6b]];

  // 
  //  Device-Level
  // 
  double* dev_evl_sorted_h1b = (double*)getGpuMem(sizeof(double) * base_size_h1b);
  double* dev_evl_sorted_h2b = (double*)getGpuMem(sizeof(double) * base_size_h2b);
  double* dev_evl_sorted_h3b = (double*)getGpuMem(sizeof(double) * base_size_h3b);
  double* dev_evl_sorted_p4b = (double*)getGpuMem(sizeof(double) * base_size_p4b);
  double* dev_evl_sorted_p5b = (double*)getGpuMem(sizeof(double) * base_size_p5b);
  double* dev_evl_sorted_p6b = (double*)getGpuMem(sizeof(double) * base_size_p6b);

  // 
#ifdef OPT_ALL_TIMING
  cudaEventRecord(stop_init);
  cudaEventSynchronize(stop_init);
  cudaEventRecord(start_collecting_data);
#endif
  
  // 
  std::fill(df_simple_s1_exec, df_simple_s1_exec + (9), -1);
  std::fill(df_simple_d1_exec, df_simple_d1_exec + (9 * noab), -1);
  std::fill(df_simple_d2_exec, df_simple_d2_exec + (9 * nvab), -1);

  // 
  ccsd_t_data_s1_new(noab,nvab,k_spin,
                d_t1,d_t2,d_v2,
                k_evl_sorted,k_range,
                t_h1b,t_h2b,t_h3b,
                t_p4b,t_p5b,t_p6b,
                // 
                size_T_s1_t1,         size_T_s1_v2, 
                df_simple_s1_size,    df_simple_s1_exec, 
                df_host_pinned_s1_t1, df_host_pinned_s1_v2, 
                &df_num_s1_enabled, 
                // 
                cache_s1t,cache_s1v);

  // 
  ccsd_t_data_d1_new(noab,nvab,k_spin,
                d_t1,d_t2,d_v2,
                k_evl_sorted,k_range,
                t_h1b,t_h2b,t_h3b,t_p4b,t_p5b,t_p6b,
                // 
                size_T_d1_t2,         size_T_d1_v2, 
                df_host_pinned_d1_t2, df_host_pinned_d1_v2, 
                df_simple_d1_size,    df_simple_d1_exec, 
                &df_num_d1_enabled, 
                // 
                cache_d1t,cache_d1v);

  // 
  ccsd_t_data_d2_new(noab,nvab,k_spin,
                d_t1,d_t2,d_v2,
                k_evl_sorted,k_range,
                t_h1b,t_h2b,t_h3b,t_p4b,t_p5b,t_p6b,
                // 
                size_T_d2_t2,           size_T_d2_v2, 
                df_host_pinned_d2_t2,   df_host_pinned_d2_v2, 
                df_simple_d2_size,      df_simple_d2_exec, 
                &df_num_d2_enabled, 
                // 
                cache_d2t, cache_d2v);

#ifdef OPT_ALL_TIMING
  cudaEventRecord(stop_collecting_data);
  cudaEventSynchronize(stop_collecting_data);
  cudaEventRecord(start_pre_processing);
#endif
  
  // 
  // 
  //  
  // this is not pinned memory.
  cudaMemcpyAsync(dev_evl_sorted_h1b, host_evl_sorted_h1b, sizeof(double) * base_size_h1b, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(dev_evl_sorted_h2b, host_evl_sorted_h2b, sizeof(double) * base_size_h2b, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(dev_evl_sorted_h3b, host_evl_sorted_h3b, sizeof(double) * base_size_h3b, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(dev_evl_sorted_p4b, host_evl_sorted_p4b, sizeof(double) * base_size_p4b, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(dev_evl_sorted_p5b, host_evl_sorted_p5b, sizeof(double) * base_size_p5b, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(dev_evl_sorted_p6b, host_evl_sorted_p6b, sizeof(double) * base_size_p6b, cudaMemcpyHostToDevice, stream);

  // 
  //  new tensors
  // 
  cudaMemcpyAsync(df_dev_s1_t1_all, df_host_pinned_s1_t1, sizeof(double) * (max_dim_s1_t1 * df_num_s1_enabled), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(df_dev_s1_v2_all, df_host_pinned_s1_v2, sizeof(double) * (max_dim_s1_v2 * df_num_s1_enabled), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(df_dev_d1_t2_all, df_host_pinned_d1_t2, sizeof(double) * (max_dim_d1_t2 * df_num_d1_enabled), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(df_dev_d1_v2_all, df_host_pinned_d1_v2, sizeof(double) * (max_dim_d1_v2 * df_num_d1_enabled), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(df_dev_d2_t2_all, df_host_pinned_d2_t2, sizeof(double) * (max_dim_d2_t2 * df_num_d2_enabled), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(df_dev_d2_v2_all, df_host_pinned_d2_v2, sizeof(double) * (max_dim_d2_v2 * df_num_d2_enabled), cudaMemcpyHostToDevice, stream);

  // 
#ifdef OPT_ALL_TIMING
  cudaEventRecord(stop_pre_processing);
  cudaEventSynchronize(stop_pre_processing);
  cudaEventRecord(start_fused_kernel);
#endif

  // 
  // 
  // 
  size_t num_blocks = CEIL(base_size_h3b, 4) * CEIL(base_size_h2b, 4) * CEIL(base_size_h1b, 4) * 
                      CEIL(base_size_p6b, 4) * CEIL(base_size_p5b, 4) * CEIL(base_size_p4b, 4);
  
  double* host_energies = (double*)getHostMem(sizeof(double) * num_blocks * 2);
  double* dev_energies 	= (double*)getGpuMem (sizeof(double) * num_blocks * 2);

  // 
  // 
  // 
  // printf ("kernel-launch for base task based on %u\n", df_base_id);
  fully_fused_ccsd_t_gpu(&stream, num_blocks, 
                        k_range[t_h1b],k_range[t_h2b],
                        k_range[t_h3b],k_range[t_p4b],
                        k_range[t_p5b],k_range[t_p6b],
                        // 
                        df_dev_d1_t2_all, df_dev_d1_v2_all, 
                        df_dev_d2_t2_all, df_dev_d2_v2_all, 
                        df_dev_s1_t1_all, df_dev_s1_v2_all, 
                        // 
                        size_T_d1_t2, size_T_d1_v2,
                        size_T_d2_t2, size_T_d2_v2,
                        size_T_s1_t1, size_T_s1_v2,
                        // 
                        //  for constant memory
                        // 
                        df_simple_d1_size, df_simple_d1_exec, 
                        df_simple_d2_size, df_simple_d2_exec, 
                        df_simple_s1_size, df_simple_s1_exec, 
                        // 
                        noab, max_dim_d1_t2, max_dim_d1_v2, 
                        nvab, max_dim_d2_t2, max_dim_d2_v2, 
                              max_dim_s1_t1, max_dim_s1_v2, 
                        // 
                        factor, 
                        // 
                        dev_evl_sorted_h1b, dev_evl_sorted_h2b, dev_evl_sorted_h3b, 
                        dev_evl_sorted_p4b, dev_evl_sorted_p5b, dev_evl_sorted_p6b, 
                        // 
                        dev_energies);

  // 
#ifdef OPT_ALL_TIMING
  cudaEventRecord(stop_fused_kernel);
  cudaEventSynchronize(stop_fused_kernel);
  cudaEventRecord(start_post_processing);
#endif

  // 
  // 
  // 
  // printf ("post-processing for base task based on %u\n", df_base_id);
  cudaMemcpyAsync(host_energies, dev_energies, num_blocks * 2 * sizeof(double), cudaMemcpyDeviceToHost, stream);
  cudaDeviceSynchronize();

  // 
  double final_energy_1 = 0.0;
  double final_energy_2 = 0.0;
  for (size_t i = 0; i < num_blocks; i++)
  {
    final_energy_1 += host_energies[i];
    final_energy_2 += host_energies[i + num_blocks];
  }

  // 
  energy_l[0] += final_energy_1 * factor;
  energy_l[1] += final_energy_2 * factor;
  
  // printf ("[%s] %+.15f, %+.15f\n", __func__, final_energy_1 * factor, final_energy_2 * factor);

  // 
  //  free device and host mem. for a task.
  // 
  freeGpuMem(dev_energies);
  freeHostMem(host_energies);

  freeGpuMem(dev_evl_sorted_h1b); freeGpuMem(dev_evl_sorted_h2b); freeGpuMem(dev_evl_sorted_h3b);
  freeGpuMem(dev_evl_sorted_p4b); freeGpuMem(dev_evl_sorted_p5b); freeGpuMem(dev_evl_sorted_p6b);

  // 
  //  free shared deivce mem
  // 
  freeGpuMem(df_dev_s1_t1_all); freeGpuMem(df_dev_s1_v2_all);
  freeGpuMem(df_dev_d1_t2_all); freeGpuMem(df_dev_d1_v2_all);
  freeGpuMem(df_dev_d2_t2_all); freeGpuMem(df_dev_d2_v2_all);

  // 
  //  free shared host mem.
  // 
  freeHostMem(df_host_pinned_s1_t1);
  freeHostMem(df_host_pinned_s1_v2);
  freeHostMem(df_host_pinned_d1_t2);
  freeHostMem(df_host_pinned_d1_v2);
  freeHostMem(df_host_pinned_d2_t2);
  freeHostMem(df_host_pinned_d2_v2);

  // 
  freeHostMem(df_simple_s1_exec);
  freeHostMem(df_simple_s1_size);
  freeHostMem(df_simple_d1_exec);
  freeHostMem(df_simple_d1_size);
  freeHostMem(df_simple_d2_exec);
  freeHostMem(df_simple_d2_size);

  // 
#ifdef OPT_ALL_TIMING
  cudaEventRecord(stop_post_processing);
  cudaEventSynchronize(stop_post_processing);

  cudaEventElapsedTime(&time_ms_init,             start_init,             stop_init);
  cudaEventElapsedTime(&time_ms_pre_processing,   start_pre_processing,   stop_pre_processing);
  cudaEventElapsedTime(&time_ms_fused_kernel,     start_fused_kernel,     stop_fused_kernel);
  cudaEventElapsedTime(&time_ms_collecting_data,  start_collecting_data,  stop_collecting_data);
  cudaEventElapsedTime(&time_ms_post_processing,  start_post_processing,  stop_post_processing);

  if (rank == 0)
  {
    int tmp_dev_id = 0;
    cudaGetDevice(&tmp_dev_id);
    printf ("[%s] performed by rank: %d with dev-id: %d ----------------------\n", __func__, rank, tmp_dev_id);
    printf ("[%s][df-based] time-init             : %f (ms)\n", __func__, time_ms_init);
    printf ("[%s][df-based] time-pre-processing   : %f (ms)\n", __func__, time_ms_pre_processing);
    printf ("[%s][df-based] time-fused-kernel     : %f (ms)\n", __func__, time_ms_fused_kernel);
    printf ("[%s][df-based] time-collecting-data  : %f (ms)\n", __func__, time_ms_collecting_data);
    printf ("[%s][df-based] time-post-processing  : %f (ms)\n", __func__, time_ms_post_processing);
    printf ("[%s] ------------------------------------------------------------\n", __func__);
  }
#endif
}


// 
// 
// 
template<typename T>
void ccsd_t_fully_fused_df(std::vector<std::tuple<size_t, size_t, size_t, size_t, size_t, size_t, T>>& list_tasks, 
                            const Index noab, const Index nvab,
                            std::vector<int>& k_spin,
                            std::vector<size_t>& k_range,
                            std::vector<size_t>& k_offset,
                            Tensor<T>& d_t1, Tensor<T>& d_t2, Tensor<T>& d_v2, 
                            std::vector<T>& k_evl_sorted,
                            //  
                            size_t size_T_s1_t1, size_t size_T_s1_v2, 
                            size_t size_T_d1_t2, size_t size_T_d1_v2, 
                            size_t size_T_d2_t2, size_t size_T_d2_v2, 
                            // 
                            std::vector<double>& energy_l, 
                            LRUCache<Index,std::vector<T>>& cache_s1t, LRUCache<Index,std::vector<T>>& cache_s1v,
                            LRUCache<Index,std::vector<T>>& cache_d1t, LRUCache<Index,std::vector<T>>& cache_d1v,
                            LRUCache<Index,std::vector<T>>& cache_d2t, LRUCache<Index,std::vector<T>>& cache_d2v) 
{
  // 
  cudaStream_t streams[2];

  // Index p4b,p5b,p6b,h1b,h2b,h3b;
  const size_t max_dim_s1_t1 = size_T_s1_t1 / 9;
  const size_t max_dim_s1_v2 = size_T_s1_v2 / 9;
  const size_t max_dim_d1_t2 = size_T_d1_t2 / (9 * noab);
  const size_t max_dim_d1_v2 = size_T_d1_v2 / (9 * noab);
  const size_t max_dim_d2_t2 = size_T_d2_t2 / (9 * nvab);
  const size_t max_dim_d2_v2 = size_T_d2_v2 / (9 * nvab);
  
	// 
	//  >> for pinned host memory (should support redundant calls)
	// 
  // 
  //  pinned host memory for s1 (t1, v2), d1 (t2, v2), and d2 (t2, v2)
  // 
  T* df_host_pinned_s1_t1[2];
  T* df_host_pinned_s1_v2[2];
  T* df_host_pinned_d1_t2[2];
  T* df_host_pinned_d1_v2[2];
  T* df_host_pinned_d2_t2[2];
  T* df_host_pinned_d2_v2[2];  

  // 
  int*  df_simple_s1_size[2];   int*  df_simple_d1_size[2];   int* df_simple_d2_size[2];
  int*  df_simple_s1_exec[2];   int*  df_simple_d1_exec[2];   int* df_simple_d2_exec[2];
  int   df_num_s1_enabled[2];   int   df_num_d1_enabled[2];   int  df_num_d2_enabled[2];

  // 
  //  Host-Level
  //  to allocate host (pinned) memory for double-buffering (does not depend on a task)
  // 
  for (int i = 0; i < 2; i++)
  {
    // 
    df_host_pinned_s1_t1[i] = (T*)getHostMem(sizeof(double) * size_T_s1_t1);
    df_host_pinned_s1_v2[i] = (T*)getHostMem(sizeof(double) * size_T_s1_v2);
    df_host_pinned_d1_t2[i] = (T*)getHostMem(sizeof(double) * size_T_d1_t2);
    df_host_pinned_d1_v2[i] = (T*)getHostMem(sizeof(double) * size_T_d1_v2);
    df_host_pinned_d2_t2[i] = (T*)getHostMem(sizeof(double) * size_T_d2_t2);
    df_host_pinned_d2_v2[i] = (T*)getHostMem(sizeof(double) * size_T_d2_v2);

    // 
    df_simple_s1_size[i] = (int*)getHostMem(sizeof(int) * (6));
    df_simple_s1_exec[i] = (int*)getHostMem(sizeof(int) * (9));

    df_simple_d1_size[i] = (int*)getHostMem(sizeof(int) * (7 * noab));
    df_simple_d1_exec[i] = (int*)getHostMem(sizeof(int) * (9 * noab));

    df_simple_d2_size[i] = (int*)getHostMem(sizeof(int) * (7 * nvab));
    df_simple_d2_exec[i] = (int*)getHostMem(sizeof(int) * (9 * nvab));

    // 
    cudaStreamCreate(&streams[i]);
  }

  // 
  //  Device-Level
  // 
  double* df_dev_s1_t1_all = (double*)getGpuMem(sizeof(double) * size_T_s1_t1);
	double* df_dev_s1_v2_all = (double*)getGpuMem(sizeof(double) * size_T_s1_v2);
  double* df_dev_d1_t2_all = (double*)getGpuMem(sizeof(double) * size_T_d1_t2);
	double* df_dev_d1_v2_all = (double*)getGpuMem(sizeof(double) * size_T_d1_v2);
  double* df_dev_d2_t2_all = (double*)getGpuMem(sizeof(double) * size_T_d2_t2);
	double* df_dev_d2_v2_all = (double*)getGpuMem(sizeof(double) * size_T_d2_v2);

#ifdef DB_DETAILS
  cudaEventRecord(stop_init);
  cudaEventSynchronize(stop_init);
#endif
  
  // 
  // 
  //  >>> for-statement <<<
  // 
  for (unsigned int current_id = 0; current_id < list_tasks.size(); current_id++)
  {
    // 
  #ifdef DB_DETAILS
    cudaEventRecord(start_pre_processing);
  #endif
    
    // 
    unsigned int df_base_id = (current_id)      % 2;  // odd and even
    unsigned int df_next_id = (current_id + 1)  % 2;  // odd and even

    // 
    //  variables based on a current task
    // 
    size_t t_h1b = std::get<0>(list_tasks[(int)current_id]);
    size_t t_h2b = std::get<1>(list_tasks[(int)current_id]);
    size_t t_h3b = std::get<2>(list_tasks[(int)current_id]);
    size_t t_p4b = std::get<3>(list_tasks[(int)current_id]);
    size_t t_p5b = std::get<4>(list_tasks[(int)current_id]);
    size_t t_p6b = std::get<5>(list_tasks[(int)current_id]);

    double factor = std::get<6>(list_tasks[(int)current_id]);

    // 
    size_t base_size_h1b = k_range[t_h1b];
    size_t base_size_h2b = k_range[t_h2b];
    size_t base_size_h3b = k_range[t_h3b];
    size_t base_size_p4b = k_range[t_p4b];
    size_t base_size_p5b = k_range[t_p5b];
    size_t base_size_p6b = k_range[t_p6b];

    // 
    //  Host-Level
    // 
    double* host_evl_sorted_h1b = &k_evl_sorted[k_offset[t_h1b]];
    double* host_evl_sorted_h2b = &k_evl_sorted[k_offset[t_h2b]];
    double* host_evl_sorted_h3b = &k_evl_sorted[k_offset[t_h3b]];
    double* host_evl_sorted_p4b = &k_evl_sorted[k_offset[t_p4b]];
    double* host_evl_sorted_p5b = &k_evl_sorted[k_offset[t_p5b]];
    double* host_evl_sorted_p6b = &k_evl_sorted[k_offset[t_p6b]];

    // 
    //  Device-Level
    // 
    double* dev_evl_sorted_h1b = (double*)getGpuMem(sizeof(double) * base_size_h1b);
    double* dev_evl_sorted_h2b = (double*)getGpuMem(sizeof(double) * base_size_h2b);
    double* dev_evl_sorted_h3b = (double*)getGpuMem(sizeof(double) * base_size_h3b);
    double* dev_evl_sorted_p4b = (double*)getGpuMem(sizeof(double) * base_size_p4b);
    double* dev_evl_sorted_p5b = (double*)getGpuMem(sizeof(double) * base_size_p5b);
    double* dev_evl_sorted_p6b = (double*)getGpuMem(sizeof(double) * base_size_p6b);

    // 
    // 
    // 
    if (current_id == 0)
    {
      // 
      std::fill(df_simple_s1_exec[df_base_id], df_simple_s1_exec[df_base_id] + (9), -1);
      std::fill(df_simple_d1_exec[df_base_id], df_simple_d1_exec[df_base_id] + (9 * noab), -1);
      std::fill(df_simple_d2_exec[df_base_id], df_simple_d2_exec[df_base_id] + (9 * nvab), -1);

      // 
      ccsd_t_data_s1(noab,nvab,k_spin,
                    d_t1,d_t2,d_v2,
                    k_evl_sorted,k_range,
                    t_h1b,t_h2b,t_h3b,
                    t_p4b,t_p5b,t_p6b,
                    // 
                    size_T_s1_t1,                     size_T_s1_v2, 
                    df_simple_s1_size[df_base_id],    df_simple_s1_exec[df_base_id], 
                    df_host_pinned_s1_t1[df_base_id], df_host_pinned_s1_v2[df_base_id], 
                    &df_num_s1_enabled[df_base_id], 
                    // 
                    cache_s1t,cache_s1v);

      // 
      ccsd_t_data_d1(noab,nvab,k_spin,
                    d_t1,d_t2,d_v2,
                    k_evl_sorted,k_range,
                    t_h1b,t_h2b,t_h3b,t_p4b,t_p5b,t_p6b,
                    // 
                    size_T_d1_t2, size_T_d1_v2, 
                    df_host_pinned_d1_t2[df_base_id], df_host_pinned_d1_v2[df_base_id], 
                    df_simple_d1_size[df_base_id],    df_simple_d1_exec[df_base_id], 
                    &df_num_d1_enabled[df_base_id], 
                    // 
                    cache_d1t,cache_d1v);

      // 
      ccsd_t_data_d2(noab,nvab,k_spin,
                    d_t1,d_t2,d_v2,
                    k_evl_sorted,k_range,
                    t_h1b,t_h2b,t_h3b,t_p4b,t_p5b,t_p6b,
                    // 
                    size_T_d2_t2,                       size_T_d2_v2, 
                    df_host_pinned_d2_t2[df_base_id],   df_host_pinned_d2_v2[df_base_id], 
                    df_simple_d2_size[df_base_id],      df_simple_d2_exec[df_base_id], 
                    &df_num_d2_enabled[df_base_id], 
                    // 
                    cache_d2t, cache_d2v);
        }
    
    // 
    // 
    //  
    // this is not pinned memory.
    cudaMemcpyAsync(dev_evl_sorted_h1b, host_evl_sorted_h1b, sizeof(double) * base_size_h1b, cudaMemcpyHostToDevice, streams[df_base_id]);
    cudaMemcpyAsync(dev_evl_sorted_h2b, host_evl_sorted_h2b, sizeof(double) * base_size_h2b, cudaMemcpyHostToDevice, streams[df_base_id]);
    cudaMemcpyAsync(dev_evl_sorted_h3b, host_evl_sorted_h3b, sizeof(double) * base_size_h3b, cudaMemcpyHostToDevice, streams[df_base_id]);
    cudaMemcpyAsync(dev_evl_sorted_p4b, host_evl_sorted_p4b, sizeof(double) * base_size_p4b, cudaMemcpyHostToDevice, streams[df_base_id]);
    cudaMemcpyAsync(dev_evl_sorted_p5b, host_evl_sorted_p5b, sizeof(double) * base_size_p5b, cudaMemcpyHostToDevice, streams[df_base_id]);
    cudaMemcpyAsync(dev_evl_sorted_p6b, host_evl_sorted_p6b, sizeof(double) * base_size_p6b, cudaMemcpyHostToDevice, streams[df_base_id]);

    // 
    //  new tensors
    // 
    cudaMemcpyAsync(df_dev_s1_t1_all, df_host_pinned_s1_t1[df_base_id], sizeof(double) * (max_dim_s1_t1 * df_num_s1_enabled[df_base_id]), cudaMemcpyHostToDevice, streams[df_base_id]);
    cudaMemcpyAsync(df_dev_s1_v2_all, df_host_pinned_s1_v2[df_base_id], sizeof(double) * (max_dim_s1_v2 * df_num_s1_enabled[df_base_id]), cudaMemcpyHostToDevice, streams[df_base_id]);
    cudaMemcpyAsync(df_dev_d1_t2_all, df_host_pinned_d1_t2[df_base_id], sizeof(double) * (max_dim_d1_t2 * df_num_d1_enabled[df_base_id]), cudaMemcpyHostToDevice, streams[df_base_id]);
    cudaMemcpyAsync(df_dev_d1_v2_all, df_host_pinned_d1_v2[df_base_id], sizeof(double) * (max_dim_d1_v2 * df_num_d1_enabled[df_base_id]), cudaMemcpyHostToDevice, streams[df_base_id]);
    cudaMemcpyAsync(df_dev_d2_t2_all, df_host_pinned_d2_t2[df_base_id], sizeof(double) * (max_dim_d2_t2 * df_num_d2_enabled[df_base_id]), cudaMemcpyHostToDevice, streams[df_base_id]);
    cudaMemcpyAsync(df_dev_d2_v2_all, df_host_pinned_d2_v2[df_base_id], sizeof(double) * (max_dim_d2_v2 * df_num_d2_enabled[df_base_id]), cudaMemcpyHostToDevice, streams[df_base_id]);

    // 
  #ifdef DB_DETAILS
    cudaEventRecord(stop_pre_processing);
    cudaEventSynchronize(stop_pre_processing);
    cudaEventRecord(start_fused_kernel);
  #endif
    
    // 
    // 
    // 
    size_t num_blocks = CEIL(base_size_h3b, 4) * CEIL(base_size_h2b, 4) * CEIL(base_size_h1b, 4) * 
                        CEIL(base_size_p6b, 4) * CEIL(base_size_p5b, 4) * CEIL(base_size_p4b, 4);
    
    double* host_energies = (double*)getHostMem(sizeof(double) * num_blocks * 2);
    double* dev_energies 	= (double*)getGpuMem (sizeof(double) * num_blocks * 2);

    // 
    // 
    // 
    // printf ("kernel-launch for base task based on %u\n", df_base_id);
    fully_fused_ccsd_t_gpu(&streams[df_base_id], num_blocks, 
                          k_range[t_h1b],k_range[t_h2b],
                          k_range[t_h3b],k_range[t_p4b],
                          k_range[t_p5b],k_range[t_p6b],
                          // 
                          df_dev_d1_t2_all, df_dev_d1_v2_all, 
                          df_dev_d2_t2_all, df_dev_d2_v2_all, 
                          df_dev_s1_t1_all, df_dev_s1_v2_all, 
                          // 
                          size_T_d1_t2, size_T_d1_v2,
                          size_T_d2_t2, size_T_d2_v2,
                          size_T_s1_t1, size_T_s1_v2,
                          // 
                          //  for constant memory
                          // 
                          df_simple_d1_size[df_base_id], df_simple_d1_exec[df_base_id], 
                          df_simple_d2_size[df_base_id], df_simple_d2_exec[df_base_id], 
                          df_simple_s1_size[df_base_id], df_simple_s1_exec[df_base_id], 
                          // 
                          noab, max_dim_d1_t2, max_dim_d1_v2, 
                          nvab, max_dim_d2_t2, max_dim_d2_v2, 
                                max_dim_s1_t1, max_dim_s1_v2, 
                          // 
                          factor, 
                          // 
                          dev_evl_sorted_h1b, dev_evl_sorted_h2b, dev_evl_sorted_h3b, 
                          dev_evl_sorted_p4b, dev_evl_sorted_p5b, dev_evl_sorted_p6b, 
                          // 
                          dev_energies);

    // 
    // 
  #ifdef DB_DETAILS
    cudaEventRecord(stop_fused_kernel);
    cudaEventSynchronize(stop_fused_kernel);
    cudaEventRecord(start_post_processing);
  #endif
    
    if (current_id + 1 < list_tasks.size())
    {
      // printf ("pre-processing for next task based on %u\n", df_next_id);
      std::fill(df_simple_s1_exec[df_next_id], df_simple_s1_exec[df_next_id] + (9), -1);
      std::fill(df_simple_d1_exec[df_next_id], df_simple_d1_exec[df_next_id] + (9 * noab), -1);
      std::fill(df_simple_d2_exec[df_next_id], df_simple_d2_exec[df_next_id] + (9 * nvab), -1);

      size_t t_h1b = std::get<0>(list_tasks[(int)current_id + 1]);
      size_t t_h2b = std::get<1>(list_tasks[(int)current_id + 1]);
      size_t t_h3b = std::get<2>(list_tasks[(int)current_id + 1]);
      size_t t_p4b = std::get<3>(list_tasks[(int)current_id + 1]);
      size_t t_p5b = std::get<4>(list_tasks[(int)current_id + 1]);
      size_t t_p6b = std::get<5>(list_tasks[(int)current_id + 1]);

      // 
      ccsd_t_data_s1(noab,nvab,k_spin,
                    d_t1,d_t2,d_v2,
                    k_evl_sorted,k_range,
                    t_h1b,t_h2b,t_h3b,t_p4b,t_p5b,t_p6b,
                    // 
                    size_T_s1_t1,                     size_T_s1_v2, 
                    df_simple_s1_size[df_next_id],    df_simple_s1_exec[df_next_id], 
                    df_host_pinned_s1_t1[df_next_id], df_host_pinned_s1_v2[df_next_id], 
                    &df_num_s1_enabled[df_next_id], 
                    // 
                    cache_s1t,cache_s1v);

      // 
      ccsd_t_data_d1(noab,nvab,k_spin,
                    d_t1,d_t2,d_v2,
                    k_evl_sorted,k_range,
                    t_h1b,t_h2b,t_h3b,t_p4b,t_p5b,t_p6b,
                    // 
                    size_T_d1_t2, size_T_d1_v2, 
                    df_host_pinned_d1_t2[df_next_id], df_host_pinned_d1_v2[df_next_id], 
                    df_simple_d1_size[df_next_id],    df_simple_d1_exec[df_next_id], 
                    &df_num_d1_enabled[df_next_id], 
                    // 
                    cache_d1t,cache_d1v);

      // 
      ccsd_t_data_d2(noab,nvab,k_spin,
                    d_t1,d_t2,d_v2,
                    k_evl_sorted,k_range,
                    t_h1b,t_h2b,t_h3b,t_p4b,t_p5b,t_p6b,
                    // 
                    size_T_d2_t2, size_T_d2_v2, 
                    df_host_pinned_d2_t2[df_next_id], df_host_pinned_d2_v2[df_next_id], 
                    df_simple_d2_size[df_next_id],    df_simple_d2_exec[df_next_id], 
                    &df_num_d2_enabled[df_next_id], 
                    // 
                    cache_d2t, cache_d2v);
    }

    // 
    // 
    // 
    // printf ("post-processing for base task based on %u\n", df_base_id);
    cudaMemcpyAsync(host_energies, dev_energies, num_blocks * 2 * sizeof(double), cudaMemcpyDeviceToHost, streams[df_base_id]);
    cudaDeviceSynchronize();

    // 
    double final_energy_1 = 0.0;
    double final_energy_2 = 0.0;
    for (size_t i = 0; i < num_blocks; i++)
    {
      final_energy_1 += host_energies[i];
      final_energy_2 += host_energies[i + num_blocks];
    }

    // 
    energy_l[0] += final_energy_1 * factor;
    energy_l[1] += final_energy_2 * factor;
    
    printf ("[%s] %+.15f, %+.15f\n", __func__, energy_l[0], energy_l[1]);

    // 
    //  free device and host mem. for a task.
    // 
    freeGpuMem(dev_energies);
    freeHostMem(host_energies);

    freeGpuMem(dev_evl_sorted_h1b); freeGpuMem(dev_evl_sorted_h2b); freeGpuMem(dev_evl_sorted_h3b);
  	freeGpuMem(dev_evl_sorted_p4b); freeGpuMem(dev_evl_sorted_p5b); freeGpuMem(dev_evl_sorted_p6b);

  #ifdef DB_DETAILS
    cudaEventRecord(stop_post_processing);
    cudaEventSynchronize(stop_post_processing);
    cudaEventElapsedTime(&time_ms_init,             start_init,             stop_init);
    cudaEventElapsedTime(&time_ms_pre_processing,   start_pre_processing,   stop_pre_processing);
    cudaEventElapsedTime(&time_ms_fused_kernel,     start_fused_kernel,     stop_fused_kernel);
    cudaEventElapsedTime(&time_ms_post_processing,  start_post_processing,  stop_post_processing);
    
    // 
    printf ("[%s][df-based][task-id:%4d] time-init            : %f (ms)\n", __func__, current_id, time_ms_init);
    printf ("[%s][df-based][task-id:%4d] time-pre-processing  : %f (ms)\n", __func__, current_id, time_ms_pre_processing);
    printf ("[%s][df-based][task-id:%4d] time-fused-kernel    : %f (ms)\n", __func__, current_id, time_ms_fused_kernel);
    printf ("[%s][df-based][task-id:%4d] time-post-processing : %f (ms)\n", __func__, current_id, time_ms_post_processing);
  #endif
  }

  // 
  //  free shared deivce mem
  // 
  freeGpuMem(df_dev_s1_t1_all); freeGpuMem(df_dev_s1_v2_all);
  freeGpuMem(df_dev_d1_t2_all); freeGpuMem(df_dev_d1_v2_all);
  freeGpuMem(df_dev_d2_t2_all); freeGpuMem(df_dev_d2_v2_all);

  // 
  //  free shared host mem.
  // 
  for (int i = 0; i < 2; i++)
  {
    // 
    freeHostMem(df_host_pinned_s1_t1[i]);
    freeHostMem(df_host_pinned_s1_v2[i]);
    freeHostMem(df_host_pinned_d1_t2[i]);
    freeHostMem(df_host_pinned_d1_v2[i]);
    freeHostMem(df_host_pinned_d2_t2[i]);
    freeHostMem(df_host_pinned_d2_v2[i]);

    // 
    freeHostMem(df_simple_s1_exec[i]);
    freeHostMem(df_simple_s1_size[i]);
    freeHostMem(df_simple_d1_exec[i]);
    freeHostMem(df_simple_d1_size[i]);
    freeHostMem(df_simple_d2_exec[i]);
    freeHostMem(df_simple_d2_size[i]);
  }
} //end ccsd_t_fully_fused_df


template<typename T>
void ccsd_t_all_fused(ExecutionContext& ec,
                   const TiledIndexSpace& MO,
                   const Index noab, const Index nvab,
                   std::vector<int>& k_spin,
                   std::vector<size_t>& k_offset,
                   //std::vector<T>& a_c, //not used
                   Tensor<T>& d_t1, 
                   Tensor<T>& d_t2, //d_a
                   Tensor<T>& d_v2, //d_b
                   std::vector<T>& k_evl_sorted,
                   std::vector<size_t>& k_range,
                   size_t t_h1b, size_t t_h2b, size_t t_h3b,
                   size_t t_p4b, size_t t_p5b, size_t t_p6b,
                   std::vector<T>& k_abufs1, std::vector<T>& k_bbufs1,
                   std::vector<T>& k_abuf1, std::vector<T>& k_bbuf1,
                   std::vector<T>& k_abuf2, std::vector<T>& k_bbuf2, 
  //  
  size_t size_T_s1_t1, size_t size_T_s1_v2, 
  size_t size_T_d1_t2, size_t size_T_d1_v2, 
  size_t size_T_d2_t2, size_t size_T_d2_v2, 
  // 
  cudaStream_t* stream_id, int gpu_id, 
  // 
                   double& factor, std::vector<double>& energy_l, 
                   int has_gpu, bool is_restricted,
                   LRUCache<Index,std::vector<T>>& cache_s1t, LRUCache<Index,std::vector<T>>& cache_s1v,
                   LRUCache<Index,std::vector<T>>& cache_d1t, LRUCache<Index,std::vector<T>>& cache_d1v,
                   LRUCache<Index,std::vector<T>>& cache_d2t, LRUCache<Index,std::vector<T>>& cache_d2v) {

  // initmemmodule();
  //singles buffers
  size_t abufs1_size = k_abufs1.size();
  size_t bbufs1_size = k_bbufs1.size();

  //doubles1 buffers
  size_t abuf_size1 = k_abuf1.size();
  size_t bbuf_size1 = k_bbuf1.size();

  //doubles2 buffers
  size_t abuf_size2 = k_abuf2.size();
  size_t bbuf_size2 = k_bbuf2.size();

  // printf ("[%s] stream_id: %d, gpu_id: %d\n", __func__, *stream_id, gpu_id);

  // Index p4b,p5b,p6b,h1b,h2b,h3b;
  const size_t max_dima = abuf_size1 /  (9*noab);
  const size_t max_dimb = bbuf_size1 /  (9*noab);
  const size_t max_dima2 = abuf_size2 / (9*nvab);
  const size_t max_dimb2 = bbuf_size2 / (9*nvab);

  const size_t s1_max_dima = abufs1_size / 9;
  const size_t s1_max_dimb = bbufs1_size / 9;

	// 
	// for pinned host memory (should support redundant calls)
	// 
	size_t size_s1_t1 = abufs1_size;
	size_t size_s1_v2 = bbufs1_size;
	size_t size_d1_t2 = abuf_size1;
	size_t size_d1_v2 = bbuf_size1;
	size_t size_d2_t2 = abuf_size2;
	size_t size_d2_v2 = bbuf_size2;

  // 
  // cudaSetDevice(gpu_id);

	// 
	T* pinned_s1_t1 = (T*)getHostMem(sizeof(double) * size_s1_t1);
	T* pinned_s1_v2 = (T*)getHostMem(sizeof(double) * size_s1_v2);
	T* pinned_d1_t2 = (T*)getHostMem(sizeof(double) * size_d1_t2);
	T* pinned_d1_v2 = (T*)getHostMem(sizeof(double) * size_d1_v2);
	T* pinned_d2_t2 = (T*)getHostMem(sizeof(double) * size_d2_t2);
	T* pinned_d2_v2 = (T*)getHostMem(sizeof(double) * size_d2_v2);

  //singles
  std::vector<int> sd_t_s1_exec(9*9,-1);
  std::vector<int> s1_sizes_ext(9*6);

#ifdef TEST_NEW_KERNEL
  std::vector<int> s1_flags(9, -1);
  std::vector<int> s1_sizes(6);
#endif

  //size_t s1b;
  ccsd_t_data_s1(ec,MO,noab,nvab,k_spin,k_offset,d_t1,d_t2,d_v2,
        k_evl_sorted,k_range,t_h1b,t_h2b,t_h3b,t_p4b,t_p5b,t_p6b,k_abufs1,k_bbufs1,
        s1_flags, s1_sizes, 
				pinned_s1_t1, pinned_s1_v2, 
        sd_t_s1_exec,s1_sizes_ext,is_restricted,cache_s1t,cache_s1v);

#if 0
  for (int idx_ia6 = 0; idx_ia6 < 9; idx_ia6++)
  {
    printf ("[s1][ia6=%d] %2d,%2d,%2d,/,%2d,%2d,%2d,/,%2d,%2d,%2d\n", idx_ia6, 
      sd_t_s1_exec.at(0 + (idx_ia6) * 9), 
      sd_t_s1_exec.at(1 + (idx_ia6) * 9), 
      sd_t_s1_exec.at(2 + (idx_ia6) * 9),
      sd_t_s1_exec.at(3 + (idx_ia6) * 9),
      sd_t_s1_exec.at(4 + (idx_ia6) * 9),
      sd_t_s1_exec.at(5 + (idx_ia6) * 9),
      sd_t_s1_exec.at(6 + (idx_ia6) * 9),
      sd_t_s1_exec.at(7 + (idx_ia6) * 9),
      sd_t_s1_exec.at(8 + (idx_ia6) * 9));
  }
#endif

  //doubles 1
  std::vector<int> sd_t_d1_exec(9*9*noab,-1);
  std::vector<int> d1_sizes_ext(9*7*noab);

#ifdef TEST_NEW_KERNEL
  std::vector<int> d1_flags(9, -1);
  std::vector<int> d1_sizes(7 * noab);
#endif

  // size_t d1b = 0;
  ccsd_t_data_d1(ec,MO,noab,nvab,k_spin,k_offset,d_t1,d_t2,d_v2,
        k_evl_sorted,k_range,t_h1b,t_h2b,t_h3b,t_p4b,t_p5b,t_p6b,k_abuf1,k_bbuf1,
        d1_flags, d1_sizes, 
				pinned_d1_t2, pinned_d1_v2, 
        sd_t_d1_exec,d1_sizes_ext,is_restricted,cache_d1t,cache_d1v);

#if 0
  // int flag_d1_8 = dev_list_d1_flags_offset[7 + (iter_noab + (iter_ia6) * size_noab) * NUM_D1_EQUATIONS];
  for (int idx_ia6 = 0; idx_ia6 < 9; idx_ia6++)
  {
    for (int idx_noab = 0; idx_noab < noab; idx_noab++)
    {
      printf ("[d1][ia6=%d][noab=%d] %2d,%2d,%2d,/,%2d,%2d,%2d,/,%2d,%2d,%2d\n", idx_ia6, idx_noab, 
      sd_t_d1_exec.at(0 + (idx_noab + (idx_ia6) * noab) * 9), 
      sd_t_d1_exec.at(1 + (idx_noab + (idx_ia6) * noab) * 9), 
      sd_t_d1_exec.at(2 + (idx_noab + (idx_ia6) * noab) * 9),
      sd_t_d1_exec.at(3 + (idx_noab + (idx_ia6) * noab) * 9),
      sd_t_d1_exec.at(4 + (idx_noab + (idx_ia6) * noab) * 9),
      sd_t_d1_exec.at(5 + (idx_noab + (idx_ia6) * noab) * 9),
      sd_t_d1_exec.at(6 + (idx_noab + (idx_ia6) * noab) * 9),
      sd_t_d1_exec.at(7 + (idx_noab + (idx_ia6) * noab) * 9),
      sd_t_d1_exec.at(8 + (idx_noab + (idx_ia6) * noab) * 9));
    }
  }
#endif

  //doubles 2
  std::vector<int> sd_t_d2_exec(9*9*nvab,-1);
  std::vector<int> d2_sizes_ext(9*7*nvab);

#ifdef TEST_NEW_KERNEL
  std::vector<int> d2_flags(9, -1);     // if at most nine equations in d2 are executed for noab x ia6, each equation should have noab and ia6, separately
  std::vector<int> d2_sizes(7 * nvab);  // a set of problem sizes per noab
#endif

  // size_t d2b=0;
  ccsd_t_data_d2(ec,MO,noab,nvab,k_spin,k_offset,d_t1,d_t2,d_v2,
      k_evl_sorted,k_range,t_h1b,t_h2b,t_h3b,t_p4b,t_p5b,t_p6b,k_abuf2,k_bbuf2,
      d2_flags, d2_sizes, 
			pinned_d2_t2, pinned_d2_v2, 
      sd_t_d2_exec,d2_sizes_ext,is_restricted,cache_d2t,cache_d2v);

#if 0
  // int flag_d1_8 = dev_list_d1_flags_offset[7 + (iter_noab + (iter_ia6) * size_noab) * NUM_D1_EQUATIONS];
  for (int idx_ia6 = 0; idx_ia6 < 9; idx_ia6++)
  {
    for (int idx_nvab = noab; idx_nvab < noab + nvab; idx_nvab++)
    {
      printf ("[d2][ia6=%d][nvab=%d] %2d,%2d,%2d,/,%2d,%2d,%2d,/,%2d,%2d,%2d\n", idx_ia6, idx_nvab, 
      sd_t_d2_exec.at(0 + (idx_nvab - noab + (idx_ia6) * nvab) * 9), 
      sd_t_d2_exec.at(1 + (idx_nvab - noab + (idx_ia6) * nvab) * 9), 
      sd_t_d2_exec.at(2 + (idx_nvab - noab + (idx_ia6) * nvab) * 9),
      sd_t_d2_exec.at(3 + (idx_nvab - noab + (idx_ia6) * nvab) * 9),
      sd_t_d2_exec.at(4 + (idx_nvab - noab + (idx_ia6) * nvab) * 9),
      sd_t_d2_exec.at(5 + (idx_nvab - noab + (idx_ia6) * nvab) * 9),
      sd_t_d2_exec.at(6 + (idx_nvab - noab + (idx_ia6) * nvab) * 9),
      sd_t_d2_exec.at(7 + (idx_nvab - noab + (idx_ia6) * nvab) * 9),
      sd_t_d2_exec.at(8 + (idx_nvab - noab + (idx_ia6) * nvab) * 9));
    }
  }
#endif
    
#if 0
  printf ("by %d, s1_t1[20]: %.10f, s1_v2[10]: %.10f\n", omp_get_thread_num(), pinned_s1_t1[20], pinned_s1_v2[10]);
  printf ("by %d, d1_t1[20]: %.10f, d1_v2[10]: %.10f\n", omp_get_thread_num(), pinned_d1_t2[20], pinned_d1_v2[10]);
  printf ("by %d, d2_t1[20]: %.10f, d2_v2[10]: %.10f\n", omp_get_thread_num(), pinned_d2_t2[20], pinned_d2_v2[10]);
#endif

  if(has_gpu)
    total_fused_ccsd_t_gpu(stream_id, gpu_id, 
                        k_range[t_h1b],k_range[t_h2b],
                        k_range[t_h3b],k_range[t_p4b],
                        k_range[t_p5b],k_range[t_p6b],
                        // k_abuf1.data(), k_bbuf1.data(),
                        // k_abuf2.data(), k_bbuf2.data(),
                        // k_abufs1.data(), k_bbufs1.data(),
												pinned_d1_t2, pinned_d1_v2, 
												pinned_d2_t2, pinned_d2_v2, 
												pinned_s1_t1, pinned_s1_v2, 
                        abuf_size1,bbuf_size1,
                        abuf_size2,bbuf_size2,
                        abufs1_size, bbufs1_size,
                        // sd_t_d1_args.data(), 
                        // sd_t_d2_args.data(), 
                        d1_sizes_ext.data(), 
                        d2_sizes_ext.data(), 
                        s1_sizes_ext.data(), 
                        // sd_t_s1_args.data(), 
                        sd_t_d1_exec,
                        sd_t_d2_exec,
                        sd_t_s1_exec, 
                        noab, max_dima,max_dimb,
                        nvab, max_dima2,max_dimb2,
                              s1_max_dima, s1_max_dimb, 
                        factor, 
                        &k_evl_sorted[k_offset[t_h1b]],
                        &k_evl_sorted[k_offset[t_h2b]],
                        &k_evl_sorted[k_offset[t_h3b]],
                        &k_evl_sorted[k_offset[t_p4b]],
                        &k_evl_sorted[k_offset[t_p5b]],
                        &k_evl_sorted[k_offset[t_p6b]],
                        &energy_l[0], &energy_l[1]);

  else
    total_fused_ccsd_t_cpu(k_range[t_h1b],k_range[t_h2b],
                        k_range[t_h3b],k_range[t_p4b],
                        k_range[t_p5b],k_range[t_p6b],
                        // k_abuf1.data(), k_bbuf1.data(),
                        // k_abuf2.data(), k_bbuf2.data(),
                        // k_abufs1.data(), k_bbufs1.data(),
												pinned_d1_t2, pinned_d1_v2, 
												pinned_d2_t2, pinned_d2_v2, 
												pinned_s1_t1, pinned_s1_v2, 
                        abuf_size1,bbuf_size1,
                        abuf_size2,bbuf_size2,
                        abufs1_size, bbufs1_size,
                        // sd_t_d1_args.data(), 
                        // sd_t_d2_args.data(), 
                        d1_sizes_ext.data(), 
                        d2_sizes_ext.data(), 
                        s1_sizes_ext.data(), 
                        // sd_t_s1_args.data(), 
                        sd_t_d1_exec,
                        sd_t_d2_exec,
                        sd_t_s1_exec, 
                        noab, max_dima,max_dimb,
                        nvab, max_dima2,max_dimb2,
                              s1_max_dima, s1_max_dimb, 
                        factor, 
                        &k_evl_sorted[k_offset[t_h1b]],
                        &k_evl_sorted[k_offset[t_h2b]],
                        &k_evl_sorted[k_offset[t_h3b]],
                        &k_evl_sorted[k_offset[t_p4b]],
                        &k_evl_sorted[k_offset[t_p5b]],
                        &k_evl_sorted[k_offset[t_p6b]],
                        &energy_l[0], &energy_l[1]);

  // 
  freeHostMem(pinned_s1_t1);
	freeHostMem(pinned_s1_v2);
	freeHostMem(pinned_d1_t2);
	freeHostMem(pinned_d1_v2);
	freeHostMem(pinned_d2_t2);
	freeHostMem(pinned_d2_v2);

} //end ccsd_t_all_fused
#endif //CCSD_T_ALL_FUSED_HPP_