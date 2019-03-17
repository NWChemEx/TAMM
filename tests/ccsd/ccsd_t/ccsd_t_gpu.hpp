
#ifndef CCSD_T_GPU_HPP_
#define CCSD_T_GPU_HPP_

#include "../ccsd_util.hpp"
#include "ccsd_t_singles_gpu.hpp"
#include "ccsd_t_doubles_gpu.hpp"

#include "header.hpp"

int check_device(long);
int device_init(long icuda,int *cuda_device_number );
void dev_release();
void finalizememmodule();

template<typename T>
std::tuple<double,double> ccsd_t_driver(ExecutionContext& ec,
                   Matrix& k_spin, TAMM_SIZE nocc, TAMM_SIZE nvirt,
                   const TiledIndexSpace& MO,
                   Tensor<T>& d_t1, Tensor<T>& d_t2,
                   Tensor<T>& d_v2,
                   std::vector<T>& k_evl_sorted,
                   double hf_ccsd_energy, int icuda=1) {


    double energy1 = 0.0;
    double energy2 = 0.0;

    auto noab = 2*nocc;
    auto nvab = 2*nvirt; 
    auto total_orbitals = noab+nvab;
    std::vector<int> k_range(total_orbitals);
    std::fill(k_range.begin(),k_range.begin()+noab,nocc);
    std::fill(k_range.begin()+noab,k_range.end(),nvirt);
    
    std::vector<int> k_offset(6,0); //TODO: k_range

    int cuda_device_number=0;
    cudaGetDeviceCount(&cuda_device_number);
    auto rank = GA_Nodeid();
    bool nodezero = rank==0;
    auto has_GPU = check_device(icuda);
    if(has_GPU == 1)
    device_init(icuda, &cuda_device_number);
    if(cuda_device_number==30) {
      std::cerr << "quit appln\n";
      return std::make_tuple(-999,-999);
    }
    if(nodezero) std::cout << "Using " << icuda << " gpu devices per node\n";

    //TODO replicate d_t1 L84-89 ccsd_t_gpu.F

    AtomicCounter* ac = new AtomicCounterGA(ec.pg(), 1);
    ac->allocate(0);
    int64_t taskcount = 0;
    int64_t next = ac->fetch_add(0, 1);
    
  for (size_t t_p4b = noab; t_p4b < noab + nvab; t_p4b++) {
    for (size_t t_p5b = t_p4b; t_p5b < noab + nvab; t_p5b++) {
      for (size_t t_p6b = t_p5b; t_p6b < noab + nvab; t_p6b++) {
        for (size_t t_h1b = 0; t_h1b < noab; t_h1b++) {
          for (size_t t_h2b = t_h1b; t_h2b < noab; t_h2b++) {
            for (size_t t_h3b = t_h2b; t_h3b < noab; t_h3b++) {

            if ((k_spin(t_p4b) + k_spin(t_p5b) + k_spin(t_p6b)) ==
                (k_spin(t_h1b) + k_spin(t_h2b) + k_spin(t_h2b))) {
              if (//(!restricted) ||
                  (k_spin(t_p4b) + k_spin(t_p5b) + k_spin(t_p6b) +
                   k_spin(t_h1b) + k_spin(t_h2b) + k_spin(t_h3b)) <= 8) {
                // if (std::bit_xor<int>(k_sym[t_p4b],
                //         std::bit_xor<int>(k_sym[t_p5b],
                //             std::bit_xor<int>(k_sym[t_p6b],
                //                 std::bit_xor<int>(k_sym[t_h1b],
                //                     std::bit_xor<int>(k_sym[t_h2b],
                //                                       k_sym[t_h3b])))))
                //     ) {
              if (next == taskcount) {
                      size_t size = k_range[t_p4b] * k_range[t_p5b] *
                                    k_range[t_p6b] * k_range[t_h1b] *
                                    k_range[t_h2b] * k_range[t_h3b];
                      std::vector<double> k_singles(size, 0.0);
                      std::vector<double> k_doubles(size, 0.0);
                      has_GPU = check_device(icuda);
                      if (has_GPU) {
                        initmemmodule();
                        dev_mem_d(k_range[t_h1b], k_range[t_h2b],
                                  k_range[t_h3b], k_range[t_p4b],
                                  k_range[t_p5b], k_range[t_p6b]);
                      }

                      //TODO:chk args, d_t1 should be local
                      
                      ccsd_t_singles_gpu(ec,MO,
                          k_singles, d_t1, d_v2, k_evl_sorted,
                          k_range,t_h1b, t_h2b, t_h3b, t_p4b, t_p5b, t_p6b, has_GPU);
                      // ccsd_t_doubles_gpu(ec,MO,&k_doubles[0],d_t2,d_v2, k_evl_sorted,
                      //     &k_range,t_h1b,t_h2b,t_h3b,t_p4b,t_p5b,t_p6b, has_GPU); 

                      double factor = 0.0;

                      // if (restricted) 
                      //   factor = 2.0;
                      //  else 
                        factor = 1.0;
                      

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
                      size_t i = 0;

                      
                      auto factor_l = factor;
                      double energy_l[2];
                      //TODO
                      // compute_en(factor_l, energy_l,
                      //             k_evl_sorted[k_offset[t_h1b]],
                      //             k_evl_sorted[k_offset[t_h2b]],
                      //             k_evl_sorted[k_offset[t_h3b]],
                      //             k_evl_sorted[k_offset[t_p4b]],
                      //             k_evl_sorted[k_offset[t_p5b]],
                      //             k_evl_sorted[k_offset[t_p6b]],
                      //             k_range[t_h1b],k_range[t_h2b],
                      //             k_range[t_h3b],k_range[t_p4b],
                      //             k_range[t_p5b],k_range[t_p6b],
                      //             k_doubles, k_singles);
                      energy1 = energy1 + energy_l[0];
                      energy2 = energy2 + energy_l[1];
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

next = ac->fetch_add(0, 1); //TODO: is this needed ? 
ec.pg().barrier();
ac->deallocate();
delete ac;


return std::make_tuple(energy1,energy2);

  
}

#endif //CCSD_T_GPU_HPP_