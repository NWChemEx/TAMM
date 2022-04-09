
#pragma once

#if defined(USE_CUDA) || defined(USE_HIP)
#if defined(USE_HIP)
#include "sd_t_total_nwc.hip.hpp"
#endif
#include "ccsd_t_singles_unfused.hpp"
#include "ccsd_t_doubles_unfused.hpp"
#else
#include "ccsd_t_singles_unfused_cpu.hpp"
#include "ccsd_t_doubles_unfused_cpu.hpp"
#endif
#include "ccsd_t_common.hpp"


int check_device(long);
#if defined(USE_CUDA) || defined(USE_HIP)
void dev_release();
void finalizememmodule();
void compute_energy(double factor, double* energy, double* eval1, double* eval2,double* eval3,double* eval4,double* eval5,double* eval6,
size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d,size_t p6d, double* host1, double* host2);
#endif

template<typename T>
std::tuple<double,double,double,double> ccsd_t_unfused_driver(ExecutionContext& ec,
                   std::vector<int>& k_spin,
                   const TiledIndexSpace& MO,
                   Tensor<T>& d_t1, Tensor<T>& d_t2,
                   Tensor<T>& d_v2,
                   std::vector<T>& k_evl_sorted,
                   double hf_ccsd_energy, int iDevice,
                   bool is_restricted, bool use_nwc_gpu_kernels) {

  auto rank     = ec.pg().rank().value();
  bool nodezero = rank==0;

  Index noab=MO("occ").num_tiles();
  Index nvab=MO("virt").num_tiles();

  Index noa=MO("occ_alpha").num_tiles();
  Index nva=MO("virt_alpha").num_tiles();

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
    cout << "noa,nva = " << noa << ", " << nva << endl;
    cout << "noab,nvab = " << noab << ", " << nvab << endl;
    // cout << "k_spin = " << k_spin << endl;
    // cout << "k_range = " << k_range << endl;
    cout << "MO Tiles = " << mo_tiles << endl;
  }

  //Check if node has number of devices specified in input file
  int dev_count_check = 0;
  bool use_dpcpp = false;

#if defined(USE_CUDA)
  CUDA_SAFE(cudaGetDeviceCount(&dev_count_check));
  if(dev_count_check < iDevice){
    if(nodezero) cout << "ERROR: Please check whether you have " << iDevice <<
      " cuda devices per node. Terminating program..." << endl << endl;
    return std::make_tuple(-999,-999,0,0);
  }
#elif defined(USE_HIP)
  HIP_SAFE(hipGetDeviceCount(&dev_count_check));
  if(dev_count_check < iDevice){
    if(nodezero) cout << "ERROR: Please check whether you have " << iDevice <<
      " hip devices per node. Terminating program..." << endl << endl;
    return std::make_tuple(-999,-999,0,0);
  }
#elif defined(USE_DPCPP)
  {
    use_dpcpp = true;
    sycl::platform platform(sycl::gpu_selector{});
    auto const& gpu_devices = platform.get_devices(sycl::info::device_type::gpu);
    for (auto &gpu_device : gpu_devices) {
      if (gpu_device.get_info<sycl::info::device::partition_max_sub_devices>() > 0) {
        auto SubDevicesDomainNuma = gpu_device.create_sub_devices<sycl::info::partition_property::partition_by_affinity_domain>(
                                                                                                                                sycl::info::partition_affinity_domain::numa);
        dev_count_check += SubDevicesDomainNuma.size();
      }
      else {
        dev_count_check++;
      }
    }
    if(dev_count_check < iDevice) {
      if(nodezero) cout << "ERROR: Please check whether you have " << iDevice <<
                     " SYCL devices per node. Terminating program..." << endl << endl;
      return std::make_tuple(-999,-999,0,0);
    }
    else if (dev_count_check <= 0) {
      if(nodezero) cout << "ERROR: NO SYCL devices found on node, " <<
                     "Terminating program..." << endl << endl;
      return std::make_tuple(-999,-999,0,0);
    }
  }
#else
  iDevice = 0;
#endif

  int gpu_device_number=0;
  //Check whether this process is associated with a GPU
  auto has_GPU = check_device(iDevice);

  // printf ("[%s] rank: %d, has_GPU: %d, iDevice: %d\n", __func__, rank, has_GPU, iDevice);

  if(iDevice==0) has_GPU=0;
  // cout << "rank,has_gpu" << rank << "," << has_GPU << endl;
  if(has_GPU == 1){
    // if(gpu_device_number==30) // QUIT
  }
  if(nodezero) std::cout << "Using " << iDevice << " gpu devices per node" << endl << endl;
  //std::cout << std::flush;

  //TODO replicate d_t1 L84-89 ccsd_t_gpu.F

    T energy1 = 0.0;
    T energy2 = 0.0;
    std::vector<T> energy_l(2);

    AtomicCounter* ac = new AtomicCounterGA(ec.pg(), 1);
    ac->allocate(0);
    int64_t taskcount = 0;
    int64_t next = ac->fetch_add(0, 1);

    auto cc_t1 = std::chrono::high_resolution_clock::now();

  for (size_t t_p4b = noab; t_p4b < noab + nvab; t_p4b++) {
    for (size_t t_p5b = t_p4b; t_p5b < noab + nvab; t_p5b++) {
      for (size_t t_p6b = t_p5b; t_p6b < noab + nvab; t_p6b++) {
        for (size_t t_h1b = 0; t_h1b < noab; t_h1b++) {
          for (size_t t_h2b = t_h1b; t_h2b < noab; t_h2b++) {
            for (size_t t_h3b = t_h2b; t_h3b < noab; t_h3b++) {

            if ((k_spin[t_p4b] + k_spin[t_p5b] + k_spin[t_p6b]) ==
                (k_spin[t_h1b] + k_spin[t_h2b] + k_spin[t_h3b])) {
              if ((!is_restricted) ||
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
                      size_t size = k_range[t_p4b] * k_range[t_p5b] *
                                    k_range[t_p6b] * k_range[t_h1b] *
                                    k_range[t_h2b] * k_range[t_h3b];

                      std::vector<double> k_singles;
                      std::vector<double> k_doubles;
                      if(!has_GPU) {
                        k_singles.resize(size,0);
                        k_doubles.resize(size,0);
                      }

                      else {
                        #if defined(USE_CUDA) || defined(USE_HIP)
                        dev_mem_s(k_range[t_h1b],k_range[t_h2b],
                                  k_range[t_h3b],k_range[t_p4b],
                                  k_range[t_p5b],k_range[t_p6b]);

                        dev_mem_d(k_range[t_h1b],k_range[t_h2b],
                                  k_range[t_h3b],k_range[t_p4b],
                                  k_range[t_p5b],k_range[t_p6b]);

                        #elif defined(USE_DPCPP)
                          k_singles.resize(size,0);
                          k_doubles.resize(size,0);
                        #endif
                      }

                      //TODO:chk args, d_t1 should be local

                      ccsd_t_singles_unfused(ec,MO,noab,nvab,k_spin,
                          k_singles, d_t1, d_v2, k_evl_sorted,
                          k_range,t_h1b, t_h2b, t_h3b, t_p4b,
                          t_p5b, t_p6b, has_GPU, is_restricted, use_nwc_gpu_kernels);
                      ccsd_t_doubles_unfused(ec,MO,noab,nvab,
                          k_spin,k_doubles,d_t2,d_v2,
                          k_evl_sorted,k_range,t_h1b,t_h2b,t_h3b,
                          t_p4b,t_p5b,t_p6b, has_GPU, is_restricted, use_nwc_gpu_kernels);

                      double factor = 0.0;

                      if (is_restricted)
                        factor = 2.0;
                      else factor = 1.0;

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

                      if(!has_GPU || use_dpcpp){
                        size_t indx = 0;
                        for (size_t t_p4=0;t_p4 < k_range[t_p4b];t_p4++)
                        for (size_t t_p5=0;t_p5 < k_range[t_p5b];t_p5++)
                        for (size_t t_p6=0;t_p6 < k_range[t_p6b];t_p6++)
                        for (size_t t_h1=0;t_h1 < k_range[t_h1b];t_h1++)
                        for (size_t t_h2=0;t_h2 < k_range[t_h2b];t_h2++)
                        for (size_t t_h3=0;t_h3 < k_range[t_h3b];t_h3++)
                        {
                          double denom =
                             (-1*k_evl_sorted[k_offset[t_p4b]+t_p4]
                                -k_evl_sorted[k_offset[t_p5b]+t_p5]
                                -k_evl_sorted[k_offset[t_p6b]+t_p6]
                                +k_evl_sorted[k_offset[t_h1b]+t_h1]
                                +k_evl_sorted[k_offset[t_h2b]+t_h2]
                                +k_evl_sorted[k_offset[t_h3b]+t_h3] );

                            energy1 += (factor * k_doubles[indx] * k_doubles[indx]) / denom;

                            energy2 += (factor * k_doubles[indx] *
                                          (k_singles[indx] + k_doubles[indx])) / denom;

                          indx++;
                        }

#if defined(USE_CUDA) || defined(USE_HIP)
			finalizememmodule();
#endif
                      }
                      else {
                      auto factor_l = factor;
                      #if defined(USE_CUDA) || defined(USE_HIP)
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

                      // cout << "e1,e2=" << energy1 << "," << energy2 << endl;
                      dev_release();
#if defined(USE_CUDA) || defined(USE_HIP)
                      finalizememmodule();
#endif
                      #endif
                    }

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

      auto cc_t2 = std::chrono::high_resolution_clock::now();
      auto ccsd_t_time =
          std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();

      ec.pg().barrier();
      cc_t2 = std::chrono::high_resolution_clock::now();
      auto total_t_time =
          std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();

      next = ac->fetch_add(0, 1);
      ec.pg().barrier();
      ac->deallocate();
      delete ac;

      return std::make_tuple(energy1, energy2, ccsd_t_time, total_t_time);

}
