
#ifndef CCSD_T_ALL_FUSED_HPP_
#define CCSD_T_ALL_FUSED_HPP_

#include "ccsd_t_all_fused_singles.hpp"
#include "ccsd_t_all_fused_doubles1.hpp"
#include "ccsd_t_all_fused_doubles2.hpp"

void initmemmodule();
void dev_mem_s(size_t,size_t,size_t,size_t,size_t,size_t);
void dev_mem_d(size_t,size_t,size_t,size_t,size_t,size_t);

void total_fused_ccsd_t_gpu(size_t base_size_h1b, size_t base_size_h2b, size_t base_size_h3b, 
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
						size_t* list_d1_sizes, 
						size_t* list_d2_sizes, 
						size_t* list_s1_sizes, 
						// 
						std::vector<size_t> vec_d1_flags,
						std::vector<size_t> vec_d2_flags,
						std::vector<size_t> vec_s1_flags, 
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
                            size_t* list_d1_sizes, 
                            size_t* list_d2_sizes, 
                            size_t* list_s1_sizes, 
                            // 
                            std::vector<size_t> vec_d1_flags,
                            std::vector<size_t> vec_d2_flags,
                            std::vector<size_t> vec_s1_flags, 
                            // 
                            size_t size_noab, size_t size_max_dim_d1_t2, size_t size_max_dim_d1_v2,
                            size_t size_nvab, size_t size_max_dim_d2_t2, size_t size_max_dim_d2_v2,
                                              size_t size_max_dim_s1_t2, size_t size_max_dim_s1_v2, 
                            // 
                            double factor, 
                            double* host_evl_sorted_h1, double* host_evl_sorted_h2, double* host_evl_sorted_h3, 
                            double* host_evl_sorted_p4, double* host_evl_sorted_p5, double* host_evl_sorted_p6,
                            double* final_energy_4, double* final_energy_5);

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

  // Index p4b,p5b,p6b,h1b,h2b,h3b;
  const size_t max_dima = abuf_size1 /  (9*noab);
  const size_t max_dimb = bbuf_size1 /  (9*noab);
  const size_t max_dima2 = abuf_size2 / (9*nvab);
  const size_t max_dimb2 = bbuf_size2 / (9*nvab);

  const size_t s1_max_dima = abufs1_size / 9;
  const size_t s1_max_dimb = bbufs1_size / 9;

  //singles
  std::vector<size_t> sd_t_s1_exec(9*9,-1);
  std::vector<size_t> s1_sizes_ext(9*6);

  //size_t s1b;
  ccsd_t_data_s1(ec,MO,noab,nvab,k_spin,k_offset,d_t1,d_t2,d_v2,
        k_evl_sorted,k_range,t_h1b,t_h2b,t_h3b,t_p4b,t_p5b,t_p6b,k_abufs1,k_bbufs1,
        sd_t_s1_exec,s1_sizes_ext,is_restricted,cache_s1t,cache_s1v);

  //doubles 1
  std::vector<size_t> sd_t_d1_exec(9*9*noab,-1);
  std::vector<size_t> d1_sizes_ext(9*7*noab);
  // size_t d1b = 0;

  ccsd_t_data_d1(ec,MO,noab,nvab,k_spin,k_offset,d_t1,d_t2,d_v2,
        k_evl_sorted,k_range,t_h1b,t_h2b,t_h3b,t_p4b,t_p5b,t_p6b,k_abuf1,k_bbuf1,
        sd_t_d1_exec,d1_sizes_ext,is_restricted,cache_d1t,cache_d1v);

  //doubles 2
  std::vector<size_t> sd_t_d2_exec(9*9*nvab,-1);
  std::vector<size_t> d2_sizes_ext(9*7*nvab);
  // size_t d2b=0;

  ccsd_t_data_d2(ec,MO,noab,nvab,k_spin,k_offset,d_t1,d_t2,d_v2,
      k_evl_sorted,k_range,t_h1b,t_h2b,t_h3b,t_p4b,t_p5b,t_p6b,k_abuf2,k_bbuf2,
      sd_t_d2_exec,d2_sizes_ext,is_restricted,cache_d2t,cache_d2v);
    
  if(has_gpu)
    total_fused_ccsd_t_gpu(k_range[t_h1b],k_range[t_h2b],
                        k_range[t_h3b],k_range[t_p4b],
                        k_range[t_p5b],k_range[t_p6b],
                        k_abuf1.data(), k_bbuf1.data(),
                        k_abuf2.data(), k_bbuf2.data(),
                        k_abufs1.data(), k_bbufs1.data(),
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
                        k_abuf1.data(), k_bbuf1.data(),
                        k_abuf2.data(), k_bbuf2.data(),
                        k_abufs1.data(), k_bbufs1.data(),
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

} //end ccsd_t_all_fused
#endif //CCSD_T_ALL_FUSED_HPP_