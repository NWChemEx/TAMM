
#ifndef CCSD_T_FUSED_HPP_
#define CCSD_T_FUSED_HPP_

#include "ccsd_t_all_fused.hpp"
#include "ccsd_t_common.hpp"

int check_device(long);
int device_init(long icuda,int *cuda_device_number );
void dev_release();
void finalizememmodule();

template<typename T>
std::tuple<double,double> ccsd_t_fused_driver(SystemData& sys_data, ExecutionContext& ec,
                   std::vector<int>& k_spin,
                   const TiledIndexSpace& MO,
                   Tensor<T>& d_t1, Tensor<T>& d_t2,
                   Tensor<T>& d_v2,
                   std::vector<T>& k_evl_sorted,
                   double hf_ccsd_energy, int icuda,
                   bool is_restricted) {

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

    LRUCache<Index,std::vector<T>> cache_s1t{0};
    LRUCache<Index,std::vector<T>> cache_s1v{0};
    LRUCache<Index,std::vector<T>> cache_d1t{0};
    LRUCache<Index,std::vector<T>> cache_d1v{0};
    LRUCache<Index,std::vector<T>> cache_d2t{0};
    LRUCache<Index,std::vector<T>> cache_d2v{0};

  for (size_t t_p4b = noab; t_p4b < noab + nvab; t_p4b++) {
    for (size_t t_p5b = t_p4b; t_p5b < noab + nvab; t_p5b++) {
      for (size_t t_p6b = t_p5b; t_p6b < noab + nvab; t_p6b++) {
        for (size_t t_h1b = 0; t_h1b < noab; t_h1b++) {
          for (size_t t_h2b = t_h1b; t_h2b < noab; t_h2b++) {
            for (size_t t_h3b = t_h2b; t_h3b < noab; t_h3b++) {

            if ((k_spin[t_p4b] + k_spin[t_p5b] + k_spin[t_p6b]) ==
                (k_spin[t_h1b] + k_spin[t_h2b] + k_spin[t_h3b])) {
              if (//(!restricted) ||
                  (k_spin[t_p4b] + k_spin[t_p5b] + k_spin[t_p6b] +
                   k_spin[t_h1b] + k_spin[t_h2b] + k_spin[t_h3b]) <= 8) {
                // if (std::bit_xor<int>(k_sym[t_p4b],
                //         std::bit_xor<int>(k_sym[t_p5b],
                //             std::bit_xor<int>(k_sym[t_p6b],
                //                 std::bit_xor<int>(k_sym[t_h1b],
                //                     std::bit_xor<int>(k_sym[t_h2b],
                //                                       k_sym[t_h3b])))))
                //     ) {
              if (next == taskcount) {
                      // size_t size = k_range[t_p4b] * k_range[t_p5b] *
                      //               k_range[t_p6b] * k_range[t_h1b] *
                      //               k_range[t_h2b] * k_range[t_h3b];

                      //TODO: cpu buffers not needed for gpu code path                                    
                      //std::vector<double> k_singles(2);/*size*/
                      //std::vector<double> k_doubles(2);
                      if (has_GPU==1) {
                        initmemmodule();
                      }

                      // if ((has_GPU==1)) {
                      //   printf ("[%s] is it called?\n", __func__);
                      //   dev_mem_s(k_range[t_h1b],k_range[t_h2b],
                      //             k_range[t_h3b],k_range[t_p4b],
                      //             k_range[t_p5b],k_range[t_p6b]);
           
                      //   dev_mem_d(k_range[t_h1b],k_range[t_h2b],
                      //             k_range[t_h3b],k_range[t_p4b],
                      //             k_range[t_p5b],k_range[t_p6b]);
                      // }

                      //TODO:chk args, d_t1 should be local

                      double factor = 0.0;

                      // if (restricted) 
                        factor = 2.0;
                      //  else factor = 1.0;

                      // cout << "restricted = " << factor << endl;

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

                      ccsd_t_all_fused(ec,MO,noab,nvab,
                        k_spin,k_offset,/*k_doubles,*/d_t1,d_t2,d_v2,
                        k_evl_sorted,k_range,t_h1b,t_h2b,t_h3b,
                        t_p4b,t_p5b,t_p6b, k_abufs1, k_bbufs1, 
                        k_abuf1,k_bbuf1,k_abuf2,k_bbuf2,factor,
                        energy_l,has_GPU,is_restricted,
                        cache_s1t,cache_s1v,cache_d1t,cache_d1v,cache_d2t,cache_d2v); 
                          
                      energy1 += energy_l[0];
                      energy2 += energy_l[1];

                      // dev_release();
                      finalizememmodule();

                      next = ac->fetch_add(0, 1); 
                    }
                      
                      taskcount++;
                  // } if sym
                    }
                  }

                }
              }
            }
          }
        }
      }

      k_abuf1.shrink_to_fit();
      k_abuf2.shrink_to_fit();
      k_bbuf1.shrink_to_fit();
      k_bbuf2.shrink_to_fit();
      k_abufs1.shrink_to_fit();
      k_bbufs1.shrink_to_fit();

    next = ac->fetch_add(0, 1); 
    ec.pg().barrier();
    ac->deallocate();
    delete ac;

    #if 0
    std::vector<Index> cvec_s1t;
    std::vector<Index> cvec_s1v;
    std::vector<Index> cvec_d1t;
    std::vector<Index> cvec_d1v;
    std::vector<Index> cvec_d2t;
    std::vector<Index> cvec_d2v;

    cache_s1t.gather_stats(cvec_s1t);
    cache_s1v.gather_stats(cvec_s1v);
    cache_d1t.gather_stats(cvec_d1t);
    cache_d1v.gather_stats(cvec_d1v);
    cache_d2t.gather_stats(cvec_d2t);
    cache_d2v.gather_stats(cvec_d2v);

    std::vector<Index> g_cvec_s1t(cvec_s1t.size());
    std::vector<Index> g_cvec_s1v(cvec_s1v.size());
    std::vector<Index> g_cvec_d1t(cvec_d1t.size());
    std::vector<Index> g_cvec_d1v(cvec_d1v.size());
    std::vector<Index> g_cvec_d2t(cvec_d2t.size());
    std::vector<Index> g_cvec_d2v(cvec_d2v.size());
    MPI_Reduce(&cvec_s1t[0], &g_cvec_s1t[0], cvec_s1t.size(), MPI_UINT32_T, MPI_SUM, 0, ec.pg().comm());
    MPI_Reduce(&cvec_s1v[0], &g_cvec_s1v[0], cvec_s1v.size(), MPI_UINT32_T, MPI_SUM, 0, ec.pg().comm());
    MPI_Reduce(&cvec_d1t[0], &g_cvec_d1t[0], cvec_d1t.size(), MPI_UINT32_T, MPI_SUM, 0, ec.pg().comm());           
    MPI_Reduce(&cvec_d1v[0], &g_cvec_d1v[0], cvec_d1v.size(), MPI_UINT32_T, MPI_SUM, 0, ec.pg().comm());           
    MPI_Reduce(&cvec_d2t[0], &g_cvec_d2t[0], cvec_d2t.size(), MPI_UINT32_T, MPI_SUM, 0, ec.pg().comm());           
    MPI_Reduce(&cvec_d2v[0], &g_cvec_d2v[0], cvec_d2v.size(), MPI_UINT32_T, MPI_SUM, 0, ec.pg().comm());           

    std::string out_fp = sys_data.input_molecule+"."+sys_data.options_map.ccsd_options.basis;
    std::string files_dir = out_fp+"_files/ccsd_t";
    if(!fs::exists(files_dir)) fs::create_directories(files_dir);    
    std::string fp = files_dir+"/";

    auto print_stats = [&](std::ostream& os, std::vector<uint32_t>& vec){
      for (uint32_t i = 0; i < vec.size(); i++) {
        os << i << " : " << vec[i] << std::endl;
      }
    };

    if(rank == 0) {
      std::ofstream fp_s1t(fp+"s1t_rank"+std::to_string(rank));
      std::ofstream fp_s1v(fp+"s1v_rank"+std::to_string(rank));
      std::ofstream fp_d1t(fp+"d1t_rank"+std::to_string(rank));
      std::ofstream fp_d1v(fp+"d1v_rank"+std::to_string(rank));
      std::ofstream fp_d2t(fp+"d2t_rank"+std::to_string(rank));
      std::ofstream fp_d2v(fp+"d2v_rank"+std::to_string(rank));

      print_stats(fp_s1t,g_cvec_s1t);
      print_stats(fp_s1v,g_cvec_s1v);
      print_stats(fp_d1t,g_cvec_d1t);
      print_stats(fp_d1v,g_cvec_d1v);
      print_stats(fp_d2t,g_cvec_d2t);
      print_stats(fp_d2v,g_cvec_d2v);
    }
    #endif

  // if(rank == 0) cout << "Total kernel (doubles) calls = " << global_kcalls << ", #fused calls = " << global_kcalls_fused << ", #partial fused calls = " << global_kcalls_pfused << endl;

  return std::make_tuple(energy1,energy2);
 
}

#endif //CCSD_T_FUSED_HPP_