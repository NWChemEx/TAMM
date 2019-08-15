#ifndef TAMM_TALSH_H_
#define TAMM_TALSH_H_

#include <iostream>
#include <cassert>
#include <list>
#include <iostream>
#include "talsh.h"
// #include "talshxx.hpp"
#include "cudamemset.hpp"

// #define NO_GPU 1

namespace ti_internal {
  template<typename> struct is_complex : std::false_type {};
  template<typename T> struct is_complex<std::complex<T>> : std::true_type {};
  template<typename T>
  inline constexpr bool is_complex_v = is_complex<T>::value;
} // namespace internal

/**
 * TAL-SH usage notes:
 *
 * - DEV_HOST has only one device number 0
 *
 * - In any call passing (dev_kind, dev_num) we can pass (DEV_DEFAULT, talshFlatDevice(..)).
 *   Basically, for a flat device id, the device kind is DEV_DEFAULT
 *
 * - For now, talshTensorConstruct() only works for constructor on host. To allocate on the
 *   GPU, one would allocate on the CPU and use talshTensorPlace() to move it to the GPU.
 * - Another temporary restriction. talshTensorConstruct cannot be called 
 *      without a talsh_task_t.
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
   GPUmempool(size_t sz)
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
   void* malloc(size_t sz)
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
      assert(chunkI != chunkList.end());
      chunkList.erase(chunkI);
      printf("Free %p\n", p); display();
   }

};

class TALSH {
  using tensor_handle = talsh_tens_t;
  private:
  static std::string talsh_tensor_string(std::string name, tensor_handle tens) {
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

  static std::string talsh_tensor_string(std::string name, tensor_handle tens, const int labels[]) {
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

  static void wait_and_destruct(talsh_task_t* task_p) {
    int done = NOPE;
    int sts, errc = TALSH_SUCCESS;
    while(done != YEP && errc == TALSH_SUCCESS) {
      done=talshTaskComplete(task_p, &sts, &errc);
    }
    assert(errc == TALSH_SUCCESS);
    errc = talshTaskDestruct(task_p);
    assert(errc == TALSH_SUCCESS);
  }

 public:
  
    static size_t small_buffer_size;
    static int gpu_list[MAX_GPUS_PER_NODE];

    static int ngpu;
    static int host_arg_max;
  TALSH() {

    int errc;
    // size_t small_buffer_size=TALSH_NO_HOST_BUFFER;
    // int gpu_list[MAX_GPUS_PER_NODE];

//Query the total number of NVIDIA GPU on node:
    // int ngpu;
    // errc=talshDeviceCount(DEV_NVIDIA_GPU,&ngpu);
    // assert(!errc);
    // printf(" Number of NVIDIA GPU found on node = %d\n",ngpu);

//Initialize TAL-SH (with a negligible Host buffer since we will use external memory):
    // int host_arg_max;
    // for(int i=0; i<ngpu; ++i) gpu_list[i]=i; //list of NVIDIA GPU devices to use in this process
    // errc=talshInit(&small_buffer_size,&host_arg_max,ngpu,gpu_list,0,NULL,0,NULL);
    // printf(" TAL-SH has been initialized: Status %d\n",errc);
    // assert(!errc);
  }
  ~TALSH() {
    // talshShutdown();
  }

  static void TALSH_initialize() {
    int errc;
    // small_buffer_size=TALSH_NO_HOST_BUFFER;
    // Query the total number of NVIDIA GPU on node:
    errc=talshDeviceCount(DEV_NVIDIA_GPU,&ngpu);
    assert(!errc);
    printf(" Number of NVIDIA GPU found on node = %d\n",ngpu);

    //Initialize TAL-SH (with a negligible Host buffer since we will use external memory):
    for(int i=0; i<ngpu; ++i) gpu_list[i]=i; //list of NVIDIA GPU devices to use in this process
    errc=talshInit(&small_buffer_size,&host_arg_max,ngpu,gpu_list,0,NULL,0,NULL);
    printf(" TAL-SH has been initialized: Status %d\n",errc);
    assert(!errc);
  }

  static void TALSH_shutdown() {
    talshShutdown();
  }

  static tensor_handle host_block(int rank,
                                  const int dims[],
                                  void *buf = nullptr) {
    tensor_handle tens;
    int errc;
    errc = talshTensorClean(&tens);
    assert(!errc);
    errc = talshTensorConstruct(&tens, R8,
                                rank, dims,
                                talshFlatDevId(DEV_HOST,0),
                                buf);
    assert(!errc);
    return tens;
  }

  static tensor_handle host_block_zero(int rank,
                                  const int dims[],
                                  void *buf = nullptr) {
    tensor_handle tens;
    int errc;
    errc = talshTensorClean(&tens);
    assert(!errc);
    errc = talshTensorConstruct(&tens, R8,
                                rank, dims,
                                talshFlatDevId(DEV_HOST,0),
                                buf,
                                -1,
                                NULL,
                                0.0);
    set_block(tens, 0.0);
    assert(!errc);
    return tens;
  }

  static void free_block(tensor_handle tens) {
    int errc=talshTensorDestruct(&tens);
    assert(!errc);
  }

  static tensor_handle gpu_block(int rank,
                                 const int dims[],
                                 int dev_num,
                                 void *buf = nullptr) {
    tensor_handle tens;
    int errc;
    errc = talshTensorClean(&tens);
    assert(!errc);
    errc = talshTensorConstruct(&tens, R8,
                                rank, dims,
                                talshFlatDevId(DEV_HOST,0),
                                // talshFlatDevId(DEV_NVIDIA_GPU,dev_num),
                                buf);
    assert(!errc);
    double* tens_gpu;
    errc = talshTensorPlace(&tens, 
                            dev_num, 
                            DEV_NVIDIA_GPU, 
                            nullptr, 
                            COPY_M); 

    assert(!errc);
    return tens;
  }

  static tensor_handle gpu_block_and_set(int rank,
                                 const int dims[],
                                 double const set_val,
                                 int dev_num,
                                 void *buf = nullptr) {
    tensor_handle tens;
    int errc;
    errc = talshTensorClean(&tens);
    assert(!errc);
    errc = talshTensorConstruct(&tens, R8,
                                rank, dims,
                                talshFlatDevId(DEV_HOST,0),
                                // talshFlatDevId(DEV_NVIDIA_GPU,dev_num),
                                buf);
    set_block(tens, set_val);
    assert(!errc);
    double* tens_gpu;
    errc = talshTensorPlace(&tens, 
                            dev_num, 
                            DEV_NVIDIA_GPU, 
                            nullptr, 
                            COPY_M); 

    assert(!errc);
    return tens;
  }

  static tensor_handle cpu_block_to_gpu_and_set(int rank,
                                 const int dims[],
                                 double const set_val,
                                 int dev_num,
                                 void *buf) {
    tensor_handle tens;
    int errc;
    errc = talshTensorClean(&tens);
    assert(!errc);
    errc = talshTensorConstruct(&tens, R8,
                                rank, dims,
                                talshFlatDevId(DEV_HOST,0),
                                // talshFlatDevId(DEV_NVIDIA_GPU,dev_num),
                                buf);
    set_block(tens, set_val);
    assert(!errc);
    double* tens_gpu;
    errc = talshTensorPlace(&tens, 
                            dev_num, 
                            DEV_NVIDIA_GPU, 
                            nullptr, 
                            COPY_M); 

    assert(!errc);
    return tens;
  }

  
  static tensor_handle gpu_block_copy(tensor_handle tens) { 
   int errc;
   errc = talshTensorPlace(&tens,  
   0,  
   DEV_HOST,  
   nullptr, 
   COPY_M); 
    
    assert(!errc);
    return tens;
  }

  static tensor_handle gpu_block_copy(tensor_handle tens,
         double *host_ptr) { 
   int errc;
   errc = talshTensorPlace(&tens,  
   0,  
   DEV_HOST,  
   reinterpret_cast<void*>(host_ptr), 
   COPY_M); 
    
    assert(!errc);
    return tens;
  }

  void tensor_print(tensor_handle tens) {
    size_t tens_vol = talshTensorVolume(&tens);
    int errc;
    talsh_tens_shape_t tshape;
    errc=tensShape_clean(&tshape);
    assert(!errc);
    errc=talshTensorShape(&tens,&tshape);
    assert(!errc);
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
    const double * bpr8 = (const double *)tens_body;
    unsigned int mlndx[MAX_TENSOR_RANK];
    for(int l=0;l<tens_vol;++l){
      tens_elem_mlndx_f(l,nd,tdims,mlndx);
      printf("\n%E",bpr8[l]); for(int i=0;i<nd;++i) printf(" %u",mlndx[i]);
    }
    printf("\nTensor Volume: %zu \n", tens_vol);fflush(0);
  }

  /**
   * @todo If @param val is 0, we can just use cudaMemSet
   *
   * @todo Just use talshTensorAdd for both CPU and GPU?
   *
   * tens[...] = val
   */
  static void set_block(tensor_handle tens, double val) {
    talsh_task_t task;
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
  static void add_block(tensor_handle ltens,
                        const int llabels[],
                        tensor_handle rtens,
                        const int rlabels[],
                        double scale) {
    talsh_tens_shape_t lshape, rshape;

    tensShape_clean(&lshape);
    talshTensorShape(&ltens, &lshape);

    tensShape_clean(&rshape);
    talshTensorShape(&rtens, &rshape);

    assert(lshape.num_dim == rshape.num_dim);
    //@todo check that llabels and rlabels are permutations of each other
    //@todo check that the dimensions of ltens and rtens match

    tensShape_destruct(&lshape);
    tensShape_destruct(&rshape);

    std::string cop_string = talsh_tensor_string("C", ltens, llabels) + "+=" +
        talsh_tensor_string("A", rtens, rlabels);
    // std::cout << "Add string: " << cop_string << std::endl; 

    talsh_task_t talsh_task;
    talshTaskClean(&talsh_task);
    talshTensorAdd(cop_string.c_str(),
                   &ltens,
                   &rtens,
                   scale, 0.0,
                   0, // DEV_DEFAULT,              //in: device id (flat or kind-specific)
                   DEV_NVIDIA_GPU, // DEV_DEFAULT,            //in: device kind (if present, <dev_id> is kind-specific)
                   COPY_TT,               //in: copy control (COPY_XX), defaults to COPY_MT
                   &talsh_task);
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
  static void mult_block(tensor_handle ltens,
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
  static void mult_block(int dev_id, tensor_handle ltens,
                         tensor_handle r1tens,
                         tensor_handle r2tens,
                         std::string cop_string,
                         T scale,
                         int move_arg) {
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

    talsh_task_t talsh_task;
    talshTaskClean(&talsh_task);
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
                        move_arg,NOPE,
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
                        move_arg,NOPE,
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
    wait_and_destruct(&talsh_task);
  }

/** Following mult_block method takes additional move_arg argument
 *  which is an int already defined in talsh such as
 *  COPY_MTT, COPY_TTT etc.
 */
template <typename T>
  static void mult_block(tensor_handle ltens,
                         tensor_handle r1tens,
                         tensor_handle r2tens,
                         std::string cop_string,
                         std::complex<T> scale,
                         int move_arg) {
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

    talsh_task_t talsh_task;
    talshTaskClean(&talsh_task);
    talshTensorContract(cop_string.c_str(),
                        &ltens,
                        &r1tens,
                        &r2tens,
                        std::real(scale),
                        std::imag(scale),
                        0,
                        DEV_NVIDIA_GPU, 
                        // DEV_HOST,
                        // DEV_DEFAULT,
                        move_arg,NOPE,
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

  static void tensor_destruct(tensor_handle tens) {
    int ierr = talshTensorDestruct(&tens);
  }
};
int TALSH::ngpu;
size_t TALSH::small_buffer_size=TALSH_NO_HOST_BUFFER;
int TALSH::gpu_list[MAX_GPUS_PER_NODE];
int TALSH::host_arg_max;

#endif // TAMM_TALSH_H_
