#pragma once

#include <iostream>
// #include <cassert>
#include <list>
#include <iostream>
// #include "talsh.h"
#include "talsh/talshxx.hpp"

#ifndef USE_HIP
#include "cudamemset.hpp"
#endif

namespace ti_internal {
  template<typename> struct is_complex : std::false_type {};
  template<typename T> struct is_complex<std::complex<T>> : std::true_type {};
  template<typename T>
  inline constexpr bool is_complex_v = is_complex<T>::value;
} // namespace ti_internal

/**
 *
 * - For now, talshTensorConstruct() only works for constructor on host. To allocate on the
 *   GPU, one would allocate on the CPU and use talshTensorPlace() to move it to the GPU.
 */

class TALSH {
  using tensor_handle = talsh_tens_t;

 public:
  int ngpu_;
  size_t small_buffer_size;
  TALSH() {
    small_buffer_size=TALSH_NO_HOST_BUFFER;
  }

  TALSH(int ngpu) {
    small_buffer_size = TALSH_NO_HOST_BUFFER;
    ngpu_ = ngpu;
  }
  ~TALSH() {
    // talshShutdown();
  }

  void initialize(int dev_id, int rank=-1) {
    int errc;
    int host_arg_max;
    // small_buffer_size=TALSH_NO_HOST_BUFFER;
    // Query the total number of NVIDIA GPU on node:
    errc=talshDeviceCount(DEV_NVIDIA_GPU,&ngpu_);
    //EXPECTS(!errc);
    if(rank==0) std::cout << "Number of NVIDIA GPUs found per node: " <<  ngpu_ << std::endl;
    // int dev_id = rank % ngpu_;
    // if(ngpu_==1) dev_id=0;
    //Initialize TAL-SH (with a negligible Host buffer since we will use external memory):
    errc=talshInit(&small_buffer_size,&host_arg_max,1,&dev_id,0,nullptr,0,nullptr);
    // int gpu_list[ngpu_];
    // for(int i=0; i<ngpu_; ++i) gpu_list[i]=i;
    // errc=talshInit(&small_buffer_size,&host_arg_max,ngpu_,gpu_list,0,nullptr,0,nullptr);
    if(rank ==0 && errc != TALSH_SUCCESS) std::cout << "TAL-SH initialize error " << errc << std::endl;
  }

  void shutdown() {
    talshShutdown();
  }

 void wait(talsh_task_t* task_p) {
    int done = NOPE;
    int sts, errc = TALSH_SUCCESS;
    while(done != YEP && errc == TALSH_SUCCESS) {
      done=talshTaskComplete(task_p, &sts, &errc);
    }
    //EXPECTS(errc == TALSH_SUCCESS);
  }

 void wait_and_destruct(talsh_task_t* task_p) {
    int done = NOPE;
    int sts, errc = TALSH_SUCCESS;
    while(done != YEP && errc == TALSH_SUCCESS) {
      done=talshTaskComplete(task_p, &sts, &errc);
    }
    //EXPECTS(errc == TALSH_SUCCESS);
    errc = talshTaskDestruct(task_p);
    //EXPECTS(errc == TALSH_SUCCESS);
  }
  
  template<typename T>
 tensor_handle host_block(int rank,
                          const int dims[],
                          T *buf = nullptr) {
    using std::is_same_v; 

    tensor_handle tens;
    int errc;
    errc = talshTensorClean(&tens);
    //EXPECTS(!errc);
    if constexpr(ti_internal::is_complex_v<T>){
      //TODO: check complex double/float
      // if constexpr(is_same_v<T,std::complex<double>){
      //   errc = talshTensorConstruct(&tens, C8,
      //                               rank, dims,
      //                               talshFlatDevId(DEV_HOST,0),
      //                               buf);
      // }
      // else
      {
        errc = talshTensorConstruct(&tens, C8,
                            rank, dims,
                            talshFlatDevId(DEV_HOST,0),
                            buf);
      }
                                          
    }
    else if constexpr(is_same_v<T,double>){
      errc = talshTensorConstruct(&tens, R8,
                                  rank, dims,
                                  talshFlatDevId(DEV_HOST,0),
                                  buf);
    }                
    else if constexpr(is_same_v<T,float>){
      errc = talshTensorConstruct(&tens, R4,
                                  rank, dims,
                                  talshFlatDevId(DEV_HOST,0),
                                  buf);
    }                                      
    //EXPECTS(!errc);
    return tens;
  }

 void free_block(tensor_handle tens) {
    int errc=talshTensorDestruct(&tens);
    //EXPECTS(!errc);
  }

  template<typename T>
 tensor_handle gpu_block(int rank,
                         const int dims[],
                         int dev_num,
                         void *buf = nullptr) {

    using std::is_same_v; 

    tensor_handle tens;
    int errc;
    errc = talshTensorClean(&tens);
    //EXPECTS(!errc);

    if constexpr(ti_internal::is_complex_v<T>) {
      //TODO: check complex double/float
      // if constexpr(is_same_v<T,std::complex<double>) {}      
        errc = talshTensorConstruct(&tens, C8,
                                rank, dims,
                                // talshFlatDevId(DEV_HOST,0),
                                talshFlatDevId(DEV_NVIDIA_GPU,dev_num),
                                buf);
    }
    else if constexpr(is_same_v<T,double>) {
        errc = talshTensorConstruct(&tens, R8,
                                rank, dims,
                                // talshFlatDevId(DEV_HOST,0),
                                talshFlatDevId(DEV_NVIDIA_GPU,dev_num),
                                buf);
    }
    else if constexpr(is_same_v<T,float>) {
        errc = talshTensorConstruct(&tens, R4,
                                rank, dims,
                                // talshFlatDevId(DEV_HOST,0),
                                talshFlatDevId(DEV_NVIDIA_GPU,dev_num),
                                buf);
    }      
    //EXPECTS(!errc);
    // set_block(tens, init_val); //initialize
    // errc = talshTensorPlace(&tens, 
    //                         dev_num, 
    //                         DEV_NVIDIA_GPU, 
    //                         nullptr, 
    //                         COPY_M); 

    // assert(!errc);
    return tens;
  }

  /**
   * ltens[llabels] += scale * rtens[rlabels]
   */
  template <typename T>  
 void add_block(std::string aop_string,
                const int dev_id, 
                tensor_handle& ltens,
                tensor_handle& rtens,
                T scale) {

    // std::cout << "Add string: " << aop_string << std::endl; 

    talsh_task_t talsh_task;
    talshTaskClean(&talsh_task);
    if constexpr(ti_internal::is_complex_v<T>){
      talshTensorAdd(aop_string.c_str(),
                    &ltens,
                    &rtens,
                    std::real(scale),
                    std::imag(scale),
                    dev_id, // DEV_DEFAULT,              //in: device id (flat or kind-specific)
                    DEV_NVIDIA_GPU, // DEV_DEFAULT,            //in: device kind (if present, <dev_id> is kind-specific)
                    COPY_MT,               //in: copy control (COPY_XX), defaults to COPY_MT
                    &talsh_task);
    }
    else {    
      talshTensorAdd(aop_string.c_str(),
                   &ltens,
                   &rtens,
                   scale, 0.0,
                   dev_id, // DEV_DEFAULT,              //in: device id (flat or kind-specific)
                   DEV_NVIDIA_GPU, // DEV_DEFAULT,            //in: device kind (if present, <dev_id> is kind-specific)
                   COPY_MT,               //in: copy control (COPY_XX), defaults to COPY_MT
                   &talsh_task);
    }
#if 0 
    double total_time;
    int ierr;
    int errc;
    errc=talshTaskWait(&talsh_task, &ierr);
    errc=talshTaskTime(&talsh_task,&total_time);
    printf(" Tensor ADD total time = %f\n",total_time);
#endif
    wait_and_destruct(&talsh_task);
  }

 /** Following mult_block method takes additional move_arg argument
  *  which is an int already defined in talsh such as 
  *  COPY_MTT, COPY_TTT etc. 
  **/ 
 template <typename T>
 void mult_block(talsh_task_t &talsh_task, int dev_id, tensor_handle& ltens,
                 tensor_handle& r1tens,
                 tensor_handle& r2tens,
                 std::string cop_string,
                 T scale,
                 int move_arg, bool is_assign) {

    talsh_tens_shape_t lshape, r1shape, r2shape;

    tensShape_clean(&lshape);
    talshTensorShape(&ltens, &lshape);

    tensShape_clean(&r1shape);
    talshTensorShape(&r1tens, &r1shape);

    tensShape_clean(&r2shape);
    talshTensorShape(&r2tens, &r2shape);

    //@todo check that the shapes of tensors match

    tensShape_destruct(&lshape);
    tensShape_destruct(&r1shape);
    tensShape_destruct(&r2shape);

    auto accum = YEP;
    if(is_assign) accum = NOPE;
    if constexpr(ti_internal::is_complex_v<T>){
      talshTensorContract(cop_string.c_str(),
                        &ltens,
                        &r1tens,
                        &r2tens,
                        std::real(scale),
                        std::imag(scale),
                        dev_id,
                        DEV_NVIDIA_GPU, 
                        // DEV_DEFAULT,
                        move_arg,accum,
                        &talsh_task);
    }
    else {
      talshTensorContract(cop_string.c_str(),
                        &ltens,
                        &r1tens,
                        &r2tens,
                        scale, 
                        0.0, 
                        dev_id,
                        DEV_NVIDIA_GPU, 
                        // DEV_DEFAULT,
                        move_arg,accum,
                        &talsh_task);
    }
  #if 0
    double total_time;
    int ierr;
    int errc;
    errc=talshTaskWait(&talsh_task, &ierr);
    errc=talshTaskTime(&talsh_task,&total_time);
    printf(" Tensor CONTRACTION total time = %f\n",total_time);
  #endif
    // wait_and_destruct(&talsh_task);
  }

  //  void tensor_destruct(tensor_handle tens) {
  //     /*int ierr = */ talshTensorDestruct(&tens);
  //   }

}; //class TALSH
