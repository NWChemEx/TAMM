#ifndef TAMM_TALSH_HPP_
#define TAMM_TALSH_HPP_

#include <iostream>
// #include <cassert>
#include <list>
#include <iostream>
// #include "talsh.h"
#include "talshxx.hpp"
#include "cudamemset.hpp"

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

/**
 * @brief memChunk to create a chunk of memory on Host
 * This memory can be used as Host Argument Buffer (HAB)
 */
struct memChunk
{
   public:
   char* start;
   char* end;
   memChunk(char* _start, char* _end) : start(_start), end(_end){ }
};

/**
 * @brief GPUmempool to create a chunk of memory on GPU
 * This memory can be used as Device Argument Buffer (DAB)
 */
class GPUmempool
{
   std::list<memChunk> chunkList;
   char* pool;
   int poolsz;
   void display()
   {
      int cnt = 0;
      printf("chunkList\n");
      for (auto chunkI = chunkList.begin(); chunkI != chunkList.end(); chunkI++, cnt++)
      {
         printf("chunk #: %d. start: %p.  end: %p\n", cnt, chunkI->start, chunkI->end);
      }
      printf("\n");
   }
   public:
   GPUmempool(int64_t sz)
   {
      pool=NULL;
      printf("Init mempool\n");
      cudaMalloc((void**)&pool, sz);
      if (pool) 
      {  
         poolsz=sz;
         // Add start and end markers
         chunkList.push_back(memChunk(pool-1,pool));
         chunkList.push_back(memChunk(pool+poolsz,pool+poolsz+1));
         display();
      } else {
        printf("Insufficient GPU memory %lu\n", sz);
      }
   }
   ~GPUmempool()
   {
      cudaFree(pool);
   }
   void* malloc(int64_t sz)
   {
      auto chunkI = chunkList.begin();
      char* last_end = chunkI->end;
      chunkI++;
      while (chunkI != chunkList.end() && chunkI->start - last_end < sz) 
                                                last_end = (chunkI++)->end;
      if (chunkI == chunkList.end()) return NULL;
      chunkList.insert(chunkI, memChunk(last_end, last_end+sz));
      printf("Alloc %lu\n", sz); display();
      return last_end;
   }
   void free(void* p)
   {
      //Don't free the start or end markers
      if (p == pool-1) return;
      if (p == pool+poolsz) return;
      auto chunkI = chunkList.begin();
      while (chunkI->start != p && chunkI != chunkList.end()) chunkI++;
      //EXPECTS(chunkI != chunkList.end());
      chunkList.erase(chunkI);
      printf("Free %p\n", p); display();
   }

};

class TALSH {
  using tensor_handle = talsh_tens_t;
  private:
 std::string talsh_tensor_string(std::string name, tensor_handle tens) {
    talsh_tens_shape_t shape;

   //LDB. Is this constant cleaning of the shapes necessary?
    tensShape_clean(&shape);
    talshTensorShape(&tens, &shape);
    std::string ret = name + "(a,";

    for(int i=1; i<shape.num_dim; i++) {
      ret += ","; 
      ret += (char)('a'+i);
    }
    tensShape_destruct(&shape);
    return ret + ")";
  }

 std::string talsh_tensor_string(std::string name, tensor_handle tens, const int labels[]) {
    talsh_tens_shape_t shape;

    tensShape_clean(&shape);
    talshTensorShape(&tens, &shape);
    std::string ret = name + "(";
    if(shape.num_dim > 0) {
      ret += (char)('a' + labels[0]);
      for(int i=1; i<shape.num_dim; i++) {
        ret += ",";
        ret += (char)('a'+ labels[i]);
      }
    }
    tensShape_destruct(&shape);
    return ret + ")";
  }

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

 tensor_handle host_block_zero(int rank,
                               const int dims[],
                               void *buf = nullptr) {
    tensor_handle tens;
    int errc;
    errc = talshTensorClean(&tens);
    //EXPECTS(!errc);
    errc = talshTensorConstruct(&tens, R8,
                                rank, dims,
                                talshFlatDevId(DEV_HOST,0),
                                buf,
                                -1,
                                NULL,
                                0.0);
    set_block(tens, 0.0);
    //EXPECTS(!errc);
    return tens;
  }

 void free_block(tensor_handle tens) {
    int errc=talshTensorDestruct(&tens);
    //EXPECTS(!errc);
  }

 tensor_handle gpu_block(int rank,
                         const int dims[],
                         int dev_num,
                         void *buf = nullptr) {
    tensor_handle tens;
    int errc;
    errc = talshTensorClean(&tens);
    //EXPECTS(!errc);
    errc = talshTensorConstruct(&tens, R8,
                                rank, dims,
                                // talshFlatDevId(DEV_HOST,0),
                                talshFlatDevId(DEV_NVIDIA_GPU,dev_num),
                                buf);
    //EXPECTS(!errc);
    // errc = talshTensorPlace(&tens, 
    //                         dev_num, 
    //                         DEV_NVIDIA_GPU, 
    //                         nullptr, 
    //                         COPY_M); 

    // assert(!errc);
    return tens;
  }

 tensor_handle gpu_block_and_set(int rank,
                                 const int dims[],
                                 double const set_val,
                                 int dev_num,
                                 void *buf = nullptr) {
    tensor_handle tens;
    int errc;
    errc = talshTensorClean(&tens);
    //EXPECTS(!errc);
    errc = talshTensorConstruct(&tens, R8,
                                rank, dims,
                                talshFlatDevId(DEV_HOST,0),
                                // talshFlatDevId(DEV_NVIDIA_GPU,dev_num),
                                buf);
    set_block(tens, set_val);
    //EXPECTS(!errc);
    errc = talshTensorPlace(&tens, 
                            dev_num, 
                            DEV_NVIDIA_GPU, 
                            nullptr, 
                            COPY_M); 

    //EXPECTS(!errc);
    return tens;
  }

 tensor_handle cpu_block_to_gpu_and_set(int rank,
                                        const int dims[],
                                        double const set_val,
                                        int dev_num,
                                        void *buf) {
    tensor_handle tens;
    int errc;
    errc = talshTensorClean(&tens);
    //EXPECTS(!errc);
    errc = talshTensorConstruct(&tens, R8,
                                rank, dims,
                                talshFlatDevId(DEV_HOST,0),
                                // talshFlatDevId(DEV_NVIDIA_GPU,dev_num),
                                buf);
    set_block(tens, set_val);
    //EXPECTS(!errc);
    errc = talshTensorPlace(&tens, 
                            dev_num, 
                            DEV_NVIDIA_GPU, 
                            nullptr, 
                            COPY_M); 

    //EXPECTS(!errc);
    return tens;
  }

  
 tensor_handle gpu_block_copy(tensor_handle tens) { 
   int errc;
   errc = talshTensorPlace(&tens,  
   0,  
   DEV_HOST,  
   nullptr, 
   COPY_M); 
    
    //EXPECTS(!errc);
    return tens;
  }

 tensor_handle gpu_block_copy(tensor_handle tens,
         double *host_ptr) { 
   int errc;
   errc = talshTensorPlace(&tens,  
   0,  
   DEV_HOST,  
   reinterpret_cast<void*>(host_ptr), 
   COPY_M); 
    
    //EXPECTS(!errc);
    return tens;
  }

  void tensor_print(tensor_handle tens) {
    size_t tens_vol = talshTensorVolume(&tens);
    int errc;
    talsh_tens_shape_t tshape;
    errc=tensShape_clean(&tshape);
    //EXPECTS(!errc);
    errc=talshTensorShape(&tens,&tshape);
    //EXPECTS(!errc);
    unsigned int nd;
    unsigned int * tdims; 
    if(errc == TALSH_SUCCESS){
      nd=(unsigned int)(tshape.num_dim);
      tdims=(unsigned int *)(tshape.dims);
    }
    void *tens_body;

    errc = talshTensorGetBodyAccess(
    &tens,  //in: pointer to a tensor block
    &tens_body,  //out: pointer to the tensor body image
    R8,  //in: requested data kind
    0,                  //in: requested device id, either kind-specific or flat
    DEV_HOST //in: requested device kind (if present, <dev_id> is kind-specific)
    );
    // const double * bpr8 = (const double *)tens_body;
    unsigned int mlndx[MAX_TENSOR_RANK];
    for(size_t l=0;l<tens_vol;++l){
      tens_elem_mlndx_f(l,nd,tdims,mlndx);
      //printf("\n%E",bpr8[l]); for(int i=0;i<nd;++i) printf(" %u",mlndx[i]);
    }
    // printf("\nTensor Volume: %zu \n", tens_vol);fflush(0);
  }

  /**
   * @todo If @param val is 0, we can just use cudaMemSet
   *
   * @todo Just use talshTensorAdd for both CPU and GPU?
   *
   * tens[...] = val
   */
 void set_block(tensor_handle tens, double val) {
    talsh_tens_shape_t shape;

    tensShape_clean(&shape);
    talshTensorShape(&tens, &shape);
    int size = 1;
    for(int i=0; i < shape.num_dim; i++) {
      size *= shape.dims[i];
    }
    tensShape_destruct(&shape);
    for(int i=0; i < tens.ndev; i++)
    {
      int dk;
      talshKindDevId(tens.dev_rsc[i].dev_id, &dk);
      
      if(dk == DEV_HOST) {
        void* buf=NULL;
        talshTensorGetBodyAccess(&tens,
                                 &buf,
                                 R8,
                                 i,
                                 DEV_HOST);
        if (!buf) printf("ERROR. No buffer of that description\n");
        double *dbuf = reinterpret_cast<double*>(buf);
        for(int i=0; i<size; i++) {
          dbuf[i] = val;
        }
      /** @todo think about consistency of copies*/
      } else if (dk == DEV_NVIDIA_GPU) {
        void* buf=NULL;
        talshTensorGetBodyAccess(&tens,
                                 &buf,
                                 R8,
                                 i,
                                 DEV_NVIDIA_GPU);
        if (!buf) printf("ERROR. No buffer of that description\n");
        double *dbuf = reinterpret_cast<double*>(buf);
        cudaMemsetAny(dbuf, val, size);
#if 0
        tensor_handle host_tens = host_block(shape.num_dim, shape.dims);
        talsh_task_t taslsh_task;
        talshTaskClean(&talsh_task);

                     &tens,
                     &host_tens,
                     1.0, 0.0,
                     DEV_DEFAULT,              //in: device id (flat or kind-specific)
                     DEV_DEFAULT,            //in: device kind (if present, <dev_id> is kind-specific)
                     COPY_TT,               //in: copy control (COPY_XX), defaults to COPY_MT
                     &talsh_task);
      wait_and_destruct(&talsh_task);
#endif
      }
    }
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

  /**
   * ltens[llabels] += scale * r1tens[r1labels] * r2tens[r2labels]
   */
 void mult_block(tensor_handle ltens,
                 const int llabels[],
                 tensor_handle r1tens,
                 const int r1labels[],
                 tensor_handle r2tens,
                 const int r2labels[],
                 double scale) {
    talsh_tens_shape_t lshape, r1shape, r2shape;

    tensShape_clean(&lshape);
    talshTensorShape(&ltens, &lshape);

    tensShape_clean(&r1shape);
    talshTensorShape(&r1tens, &r1shape);

    tensShape_clean(&r2shape);
    talshTensorShape(&r2tens, &r2shape);

    //@todo check that the shapes of tensors match

    std::string cop_string = talsh_tensor_string("D", ltens, llabels) + "+=" +
        talsh_tensor_string("L", r1tens, r1labels) + "*" +
        talsh_tensor_string("R", r2tens, r2labels);

    // std::cout << "Contract string: " << cop_string << std::endl; 

    tensShape_destruct(&lshape);
    tensShape_destruct(&r1shape);
    tensShape_destruct(&r2shape);

    talsh_task_t talsh_task;
    talshTaskClean(&talsh_task);
    talshTensorContract(cop_string.c_str(),
                        &ltens,
                        &r1tens,
                        &r2tens,
                        scale,
                        0.0,
                        0,
                        //DEV_NVIDIA_GPU, 
                        DEV_DEFAULT,
                        COPY_TTT, NOPE,
                        &talsh_task);
#if 0
    double total_time;
    int ierr;
    int errc;
    errc=talshTaskWait(&talsh_task, &ierr);
    errc=talshTaskTime(&talsh_task,&total_time);
    printf(" Tensor CONTRACTION total time = %f\n",total_time);
#endif
    wait_and_destruct(&talsh_task);
  }

/** Following mult_block method takes additional move_arg argument
 *  which is an int already defined in talsh such as 
 *  COPY_MTT, COPY_TTT etc. 
 */ 
  template <typename T>
 void mult_block(talsh_task_t &talsh_task, int dev_id, tensor_handle& ltens,
                 tensor_handle& r1tens,
                 tensor_handle& r2tens,
                 std::string cop_string,
                 T scale,
                 int move_arg, bool is_assign) {
    // int dev_id = rank % ngpu_;
    // if (ngpu_ == 1) dev_id=0;
    talsh_tens_shape_t lshape, r1shape, r2shape;

    tensShape_clean(&lshape);
    talshTensorShape(&ltens, &lshape);

    tensShape_clean(&r1shape);
    talshTensorShape(&r1tens, &r1shape);

    tensShape_clean(&r2shape);
    talshTensorShape(&r2tens, &r2shape);

    //@todo check that the shapes of tensors match
    // std::cout << "Contract string: " << cop_string << std::endl; 

    tensShape_destruct(&lshape);
    tensShape_destruct(&r1shape);
    tensShape_destruct(&r2shape);

    // talsh_task_t talsh_task;
    // talshTaskClean(&talsh_task);
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

 void tensor_destruct(tensor_handle tens) {
    /*int ierr = */ talshTensorDestruct(&tens);
  }
};
// int TALSH::ngpu_;
#endif // TAMM_TALSH_HPP_
