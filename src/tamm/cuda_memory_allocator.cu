inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n",
            cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

double *host_pinned_memory(size_t bytes) {
  double *h_buf_pinned;
  checkCuda( cudaMallocHost((void**)&h_buf_pinned, bytes) ); // host pinned
  return h_buf_pinned;
}

void free_host_pinned_memory(double *h_buf_pinned) {
  cudaFreeHost(h_buf_pinned);
}
