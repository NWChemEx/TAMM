#ifndef NO_GPU
cudaError_t cudaMemsetAny(double*, double, size_t, cudaStream_t=0);
cudaError_t cudaMemsetAny(float*, float, size_t, cudaStream_t=0);
#else
int cudaMemsetAny(double*, double, size_t, int=0);
int cudaMemsetAny(float*, float, size_t, int=0);

#endif

