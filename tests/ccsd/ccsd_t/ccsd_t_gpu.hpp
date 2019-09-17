
#ifndef CCSD_T_GPU_HPP_
#define CCSD_T_GPU_HPP_

#include "ccsd_t_singles_gpu.hpp"
#include "ccsd_t_doubles_gpu.hpp"

#include "header.hpp"

int check_device(long);
int device_init(long icuda,int *cuda_device_number );
void dev_release();
void finalizememmodule();
void compute_energy(double factor, double* energy, double* eval1, double* eval2,double* eval3,double* eval4,double* eval5,double* eval6,
size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d,size_t p6d, double* host1, double* host2);

template <typename Arg, typename... Args>
void dprint1(Arg&& arg, Args&&... args)
{
    cout << std::forward<Arg>(arg);
    ((cout << ',' << std::forward<Args>(args)), ...);
    cout << "\n";
}

template<typename T>
std::tuple<double,double> ccsd_t_driver(ExecutionContext& ec,
                   std::vector<int>& k_spin, TAMM_SIZE nocc, TAMM_SIZE nvirt,
                   const TiledIndexSpace& MO,
                   Tensor<T>& d_t1, Tensor<T>& d_t2,
                   Tensor<T>& d_v2,
                   std::vector<T>& k_evl_sorted,
                   double hf_ccsd_energy, int icuda) {

    auto rank = GA_Nodeid();
    bool nodezero = rank==0;

    if(icuda==0) {
      if(nodezero)std::cout << "\nERROR: Please specify number of cuda devices to use in the input file!\n\n"; //TODO
      return std::make_tuple(-999,-999);
    }

    Index noab=MO("occ").num_tiles();
    Index nvab=MO("virt").num_tiles();

    auto tnocc = 2*nocc;
    auto tnvirt = 2*nvirt; 
    // auto total_orbitals = tnocc+tnvirt;
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
       " cuda devices per node. Terminating program...\n\n";
      return std::make_tuple(-999,-999);
    }
    
    int cuda_device_number=0;
    //Check whether this process is associated with a GPU
    auto has_GPU = check_device(icuda);
    // cout << "rank,has_gpu" << rank << "," << has_GPU << endl;
    if(has_GPU == 1){
      device_init(icuda, &cuda_device_number);
      // if(cuda_device_number==30) // QUIT
    }
    if(nodezero) std::cout << "Using " << icuda << " gpu devices per node\n\n";

    //TODO replicate d_t1 L84-89 ccsd_t_gpu.F

    double energy1 = 0.0;
    double energy2 = 0.0;
    std::vector<double> energy_l(2);

    AtomicCounter* ac = new AtomicCounterGA(ec.pg(), 1);
    ac->allocate(0);
    int64_t taskcount = 0;
    int64_t next = ac->fetch_add(0, 1);

  if(has_GPU == 1){

  for (size_t t_p4b = noab; t_p4b < noab + nvab; t_p4b++) {
    for (size_t t_p5b = t_p4b; t_p5b < noab + nvab; t_p5b++) {
      for (size_t t_p6b = t_p5b; t_p6b < noab + nvab; t_p6b++) {
        for (size_t t_h1b = 0; t_h1b < noab; t_h1b++) {
          for (size_t t_h2b = t_h1b; t_h2b < noab; t_h2b++) {
            for (size_t t_h3b = t_h2b; t_h3b < noab; t_h3b++) {

              // dprint(k_spin[t_p4b] + k_spin[t_p5b] + k_spin[t_p6b],
              // k_spin[t_h1b] + k_spin[t_h2b] + k_spin[t_h3b]);

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
                      std::vector<double> k_singles(2);/*size*/
                      std::vector<double> k_doubles(2);
                      has_GPU = check_device(icuda);
                      if (has_GPU==1) {
                        initmemmodule();
                      }

                      if ((has_GPU==1)) {
                        dev_mem_s(k_range[t_h1b],k_range[t_h2b],
                                  k_range[t_h3b],k_range[t_p4b],
                                  k_range[t_p5b],k_range[t_p6b]);
           
                        dev_mem_d(k_range[t_h1b],k_range[t_h2b],
                                  k_range[t_h3b],k_range[t_p4b],
                                  k_range[t_p5b],k_range[t_p6b]);
                      }

                      //TODO:chk args, d_t1 should be local

                      // cout << "p4,5,6,h1,2,3 = ";
                      // dprint(t_p4b,t_p5b,t_p6b,t_h1b,t_h2b,t_h3b);

                      ccsd_t_singles_gpu(ec,MO,noab,nvab,k_spin,
                          k_singles, d_t1, d_v2, k_evl_sorted,
                          k_range,t_h1b, t_h2b, t_h3b, t_p4b, 
                          t_p5b, t_p6b, has_GPU);
                      ccsd_t_doubles_gpu(ec,MO,noab,nvab,
                          k_spin,k_doubles,d_t2,d_v2,
                          k_evl_sorted,k_range,t_h1b,t_h2b,t_h3b,
                          t_p4b,t_p5b,t_p6b, has_GPU); 

                      //  cout << "singles = " << k_singles << endl;
                      // cout << "doubles = " << k_doubles << endl;

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
                      
                      #if 0
                      auto indx = 0;
                      for (auto t_p4=0;t_p4 < k_range[t_p4b];t_p4++)
                      for (auto t_p5=0;t_p5 < k_range[t_p5b];t_p5++)
                      for (auto t_p6=0;t_p6 < k_range[t_p6b];t_p6++)
                      for (auto t_h1=0;t_h1 < k_range[t_h1b];t_h1++)
                      for (auto t_h2=0;t_h2 < k_range[t_h2b];t_h2++)
                      for (auto t_h3=0;t_h3 < k_range[t_h3b];t_h3++)
                      {
                          energy1 += (factor * k_doubles[indx] * k_doubles[indx])
                          /(-1*k_evl_sorted[k_offset[t_p4b]+t_p4]
                              -k_evl_sorted[k_offset[t_p5b]+t_p5] 
                              -k_evl_sorted[k_offset[t_p6b]+t_p6]
                              +k_evl_sorted[k_offset[t_h1b]+t_h1]
                              +k_evl_sorted[k_offset[t_h2b]+t_h2]
                              +k_evl_sorted[k_offset[t_h3b]+t_h3] );

                          energy2 += (factor * k_doubles[indx] *
                                     (k_singles[indx] * k_doubles[indx]) )
                          /(-1*k_evl_sorted[k_offset[t_p4b]+t_p4]
                              -k_evl_sorted[k_offset[t_p5b]+t_p5] 
                              -k_evl_sorted[k_offset[t_p6b]+t_p6]
                              +k_evl_sorted[k_offset[t_h1b]+t_h1]
                              +k_evl_sorted[k_offset[t_h2b]+t_h2]
                              +k_evl_sorted[k_offset[t_h3b]+t_h3] );                              

                        indx++;
                      }
                      #else
                      auto factor_l = factor;

                      // cout << "doubles size = " << size << endl;
                      // for(auto x:k_doubles)
                      // cout << x << endl;

                      // cout << "factor-l=" << factor_l << endl;
                      // cout << "k_evl_sorted_full=" << k_evl_sorted << endl;
                      // cout << "h123,p456= ";
                      // dprint1(t_h1b,t_h2b,t_h3b,t_p4b,t_p5b,t_p6b);

                      // cout << "factor-l=" << factor_l << endl;
                      // cout << "energy-l=" << energy_l << endl;

                      //  cout << "k-range of h123,p456= ";
                      //  dprint1(k_range[t_h1b],k_range[t_h2b],
                      //             k_range[t_h3b],k_range[t_p4b],
                      //             k_range[t_p5b],k_range[t_p6b]);

                      // cout << "k_evl_sorted= ";
                      // dprint1(    k_evl_sorted[k_offset[t_h1b]],
                      //             k_evl_sorted[k_offset[t_h2b]],
                      //             k_evl_sorted[k_offset[t_h3b]],
                      //             k_evl_sorted[k_offset[t_p4b]],
                      //             k_evl_sorted[k_offset[t_p5b]],
                      //             k_evl_sorted[k_offset[t_p6b]]);

                      // cout << "k_offset= ";
                      // dprint1(    k_offset[t_h1b],
                      //             k_offset[t_h2b],
                      //             k_offset[t_h3b],
                      //             k_offset[t_p4b],
                      //             k_offset[t_p5b],
                      //             k_offset[t_p6b]);                                  

                      // cout << "omg: " << k_range[t_h1b] << ", " << k_range[t_h2b] << ", " << k_range[t_h3b] << ", " << k_range[t_p4b] << ", " << k_range[t_p5b] << ", " << k_range[t_p6b] << endl;
                      // cout << "factor_l: " << factor_l << endl;
                      

                      //TODO
                      compute_energy(factor_l, &energy_l[0],
                                  &k_evl_sorted[k_offset[t_h1b]],
                                  &k_evl_sorted[k_offset[t_h2b]],
                                  &k_evl_sorted[k_offset[t_h3b]],
                                  &k_evl_sorted[k_offset[t_p4b]],
                                  &k_evl_sorted[k_offset[t_p5b]],
                                  &k_evl_sorted[k_offset[t_p6b]],
                                  k_range[t_h1b],k_range[t_h2b],
                                  k_range[t_h3b],k_range[t_p4b],
                                  k_range[t_p5b],k_range[t_p6b],
                                  &k_doubles[0], &k_singles[0]);
                      // cout << "AFTER energy-l=" << energy_l << endl;                                  
                      energy1 += energy_l[0];
                      energy2 += energy_l[1];
                      #endif

                      // cout << "e1,e2=" << energy1 << "," << energy2 << endl;
                      // cout << "-----------------------------------------\n";
                      dev_release();
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

  } //has_gpu
    next = ac->fetch_add(0, 1); //TODO: is this needed ? 
    ec.pg().barrier();
    ac->deallocate();
    delete ac;

  return std::make_tuple(energy1,energy2);
 
}

#endif //CCSD_T_GPU_HPP_
