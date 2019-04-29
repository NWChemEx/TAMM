#if 1
template <typename Type>
__global__ void memset(Type* out, Type val, size_t n)
{
   for(int tid = threadIdx.x + blockIdx.x * blockDim.x; tid < n; tid += blockDim.x*gridDim.x)
   {
      out[tid] = val;
   }

}

template <typename Type>
cudaError_t cudaMemsetAny(Type* out, Type val, size_t n, cudaStream_t st=0)
{

   memset<<<(n+255)/256,256,0,st>>>(out,val,n);
   return cudaGetLastError();

}
// Why doesn't this work!?
//template cudaError_t cudaMemsetAny<double>(double*, double, size_t, cudaStream_t);
//template cudaError_t cudaMemsetAny<float>(float*, float, size_t, cudaStream_t);
cudaError_t cudaMemsetAny(float* out, float val, size_t n, cudaStream_t st=0)
{

   memset<<<(n+255)/256,256,0,st>>>(out,val,n);
   return cudaGetLastError();

}
cudaError_t cudaMemsetAny(double* out, double val, size_t n, cudaStream_t st=0)
{

   memset<<<(n+255)/256,256,0,st>>>(out,val,n);
   return cudaGetLastError();

}
#else
template <typename Type>
int cudaMemsetAny(Type* out, Type val, size_t n, int st=0)
{
   printf("CUDA is disabled. Why are you here?\n");
   return 1;

}
template int cudaMemsetAny<double>(double*, double, size_t, int);
template int cudaMemsetAny<float>(float*, float, size_t, int);

#endif

