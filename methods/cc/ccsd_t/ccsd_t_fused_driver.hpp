
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

  // 
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

  // 
  energy1 = tmp_energy_l[0];
  energy2 = tmp_energy_l[1];

  // 
  finalizememmodule();

  // 
  next = ac->fetch_add(0, 1); 
  ec.pg().barrier ();
  ac->deallocate();
  delete ac;

  return std::make_tuple(energy1,energy2);
}

template<typename T>
void ccsd_t_fused_driver_calculator_ops(SystemData& sys_data, ExecutionContext& ec,
                                        std::vector<int>& k_spin,
                                        const TiledIndexSpace& MO,
                                        std::vector<T>& k_evl_sorted,
                                        double hf_ccsd_energy, int icuda,
                                        bool is_restricted,
                                        size_t* total_num_ops, 
                                        // 
                                        bool seq_h3b=false) 
{
  //
  auto rank = ec.pg().rank().value();
  bool nodezero = rank==0;

  // 
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

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

  // 
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

  // 
  //  "list of tasks": size (t_h1b, t_h2b, t_h3b, t_p4b, t_p5b, t_p6b), factor, 
  // 
  std::vector<std::tuple<size_t, size_t, size_t, size_t, size_t, size_t, T>> list_tasks;
  if(!seq_h3b) 
  {
    for (size_t t_p4b = noab; t_p4b < noab + nvab; t_p4b++) {
    for (size_t t_p5b = t_p4b; t_p5b < noab + nvab; t_p5b++) {
    for (size_t t_p6b = t_p5b; t_p6b < noab + nvab; t_p6b++) {
    for (size_t t_h1b = 0; t_h1b < noab; t_h1b++) { //    
    for (size_t t_h2b = t_h1b; t_h2b < noab; t_h2b++) {
    for (size_t t_h3b = t_h2b; t_h3b < noab; t_h3b++) {
      // 
      if ((k_spin[t_p4b] + k_spin[t_p5b] + k_spin[t_p6b]) ==
          (k_spin[t_h1b] + k_spin[t_h2b] + k_spin[t_h3b])) {
        if ((k_spin[t_p4b] + k_spin[t_p5b] + k_spin[t_p6b] +
            k_spin[t_h1b] + k_spin[t_h2b] + k_spin[t_h3b]) <= 8) {
          //  
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

          //
          list_tasks.push_back(std::make_tuple(t_h1b, t_h2b, t_h3b, t_p4b, t_p5b, t_p6b, factor));
        }
      }
    }}}}}}  // nested for loops
  } // parallel h3b loop
  else 
  { //seq h3b loop
    for (size_t t_p4b = noab; t_p4b < noab + nvab; t_p4b++) {
    for (size_t t_p5b = t_p4b; t_p5b < noab + nvab; t_p5b++) {
    for (size_t t_p6b = t_p5b; t_p6b < noab + nvab; t_p6b++) {
    for (size_t t_h1b = 0; t_h1b < noab; t_h1b++) {
    for (size_t t_h2b = t_h1b; t_h2b < noab; t_h2b++) {
      // 
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
            list_tasks.push_back(std::make_tuple(t_h1b, t_h2b, t_h3b, t_p4b, t_p5b, t_p6b, factor));
          }
        }
      }//h3b
    }}}}} // nested for loops
  } //end seq h3b 

  // 
  // 
  // 
  // printf ("[%s] rank: %d >> # of tasks: %lu\n", __func__, rank, list_tasks.size());

  // 
  //    
  // 
  *total_num_ops = ccsd_t_fully_fused_performance(list_tasks, 
                                rank, 1, 
                                noab, nvab,
                                k_spin,k_range,k_offset,
                                k_evl_sorted);
}
#endif //CCSD_T_FUSED_HPP_
