#include "tamm/utils.hpp"

#include "gpu_lapack_internal.hpp"
#include "tamm/errors.hpp"
#include "tamm/rmm_memory_pool.hpp"
#include "tamm_blas.hpp"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <span>
#include <sstream>
#include <type_traits>

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)

#if defined(USE_HIP)
namespace {
// lapack::Job's underlying char values ('N'/'A'/'S'/'O') don't line up with rocblas_svect's
// enumerators, so map explicitly (unlike CUDA, where static_cast<signed char>(job) is enough).
rocblas_svect job_to_rocblas_svect(lapack::Job job) {
  switch(job) {
    case lapack::Job::AllVec: return rocblas_svect_all;
    case lapack::Job::SomeVec: return rocblas_svect_singular;
    case lapack::Job::OverwriteVec: return rocblas_svect_overwrite;
    case lapack::Job::NoVec:
    default: return rocblas_svect_none;
  }
}
} // namespace
#elif defined(USE_DPCPP)
namespace {
oneapi::mkl::jobsvd job_to_oneapi_jobsvd(lapack::Job job) {
  switch(job) {
    case lapack::Job::AllVec: return oneapi::mkl::jobsvd::vectors;
    case lapack::Job::SomeVec: return oneapi::mkl::jobsvd::somevec;
    case lapack::Job::NoVec:
    default: return oneapi::mkl::jobsvd::novec;
  }
}
} // namespace
#endif

// GPU counterpart of lapack::gesvd (see tamm_blas.hpp): cusolverDn<t>gesvd (CUDA),
// rocsolver_<t>gesvd (HIP), or oneapi::mkl::lapack::gesvd (DPCPP). A/U/VT/S are host pointers;
// this function stages A to the device, runs the vendor gesvd, and copies S/U/VT back to host,
// mirroring lapack::gesvd's host-in/host-out contract so tamm::svd's call sites barely change
// between the CPU and GPU paths. On CUDA/HIP this requires m >= n (cusolverDn/rocsolver
// <t>gesvd's native constraint; tamm::svd falls back to LAPACK/CPU instead of calling this
// when that doesn't hold).
template<typename T>
void tamm::kernels::gpu::gesvd(lapack::Job jobu, lapack::Job jobvt, int64_t m, int64_t n, T* A,
                               int64_t lda, blas::real_type<T>* S, T* U, int64_t ldu, T* VT,
                               int64_t ldvt) {
  using real_t = blas::real_type<T>;

  const int64_t k = std::min(m, n);

  auto& gpustream = tamm::GPUStreamPool::getInstance().getStream();
  auto& devpool   = tamm::RMMMemoryManager::getInstance().getDeviceMemoryPool();

  const size_t a_size  = static_cast<size_t>(lda) * static_cast<size_t>(n);
  const size_t u_size  = (U != nullptr) ? static_cast<size_t>(ldu) *
                                           static_cast<size_t>(jobu == lapack::Job::AllVec ? m : k)
                                        : 0;
  const size_t vt_size = (VT != nullptr) ? static_cast<size_t>(ldvt) * static_cast<size_t>(n) : 0;

#if defined(USE_CUDA)
  EXPECTS(m >= n);

  const int im = static_cast<int>(m), in = static_cast<int>(n);
  const int ilda = static_cast<int>(lda), ildu = static_cast<int>(ldu),
            ildvt           = static_cast<int>(ldvt);
  const signed char jobu_c  = static_cast<signed char>(jobu);
  const signed char jobvt_c = static_cast<signed char>(jobvt);

  cudaStream_t stream = gpustream.first;

  cusolverDnHandle_t handle;
  CUSOLVER_CHECK(cusolverDnCreate(&handle));
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));

  const size_t rwork_size = static_cast<size_t>(std::max<int64_t>(1, k - 1));

  // The spans own the pool blocks; .data() is what the cuSolver API takes. Each
  // deallocate() below derives its byte count from the span, so it cannot disagree with
  // the allocation.
  std::span<T>      sp_A     = devpool.allocate_span<T>(a_size);
  std::span<real_t> sp_S     = devpool.allocate_span<real_t>(static_cast<size_t>(k));
  std::span<T>      sp_U     = devpool.allocate_span<T>(u_size);
  std::span<T>      sp_VT    = devpool.allocate_span<T>(vt_size);
  std::span<real_t> sp_rwork = devpool.allocate_span<real_t>(rwork_size);
  std::span<int>    sp_info  = devpool.allocate_span<int>(1);

  T*      d_A     = sp_A.data();
  real_t* d_S     = sp_S.data();
  T*      d_U     = u_size ? sp_U.data() : nullptr;
  T*      d_VT    = vt_size ? sp_VT.data() : nullptr;
  real_t* d_rwork = sp_rwork.data();
  int*    d_info  = sp_info.data();

  gpuMemcpyAsync<T>(d_A, A, a_size, gpuMemcpyHostToDevice, gpustream);

  int lwork = 0;
  if constexpr(std::is_same_v<T, double>) {
    CUSOLVER_CHECK(cusolverDnDgesvd_bufferSize(handle, im, in, &lwork));
  }
  else if constexpr(std::is_same_v<T, std::complex<double>>) {
    CUSOLVER_CHECK(cusolverDnZgesvd_bufferSize(handle, im, in, &lwork));
  }

  std::span<T> sp_work = devpool.allocate_span<T>(static_cast<size_t>(std::max(lwork, 0)));
  T*           d_work  = sp_work.data();

  if constexpr(std::is_same_v<T, double>) {
    CUSOLVER_CHECK(cusolverDnDgesvd(handle, jobu_c, jobvt_c, im, in, d_A, ilda, d_S, d_U, ildu,
                                    d_VT, ildvt, d_work, lwork, d_rwork, d_info));
  }
  else if constexpr(std::is_same_v<T, std::complex<double>>) {
    CUSOLVER_CHECK(cusolverDnZgesvd(handle, jobu_c, jobvt_c, im, in, (cuDoubleComplex*) d_A, ilda,
                                    d_S, (cuDoubleComplex*) d_U, ildu, (cuDoubleComplex*) d_VT,
                                    ildvt, (cuDoubleComplex*) d_work, lwork, d_rwork, d_info));
  }

  gpuStreamSynchronize(gpustream);

  int info = 0;
  CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
  if(info != 0) {
    std::ostringstream msg;
    msg << "cusolverDn gesvd failed with info=" << info << " (info<0: illegal argument -info; "
        << "info>0: did not converge)";
    throw std::runtime_error(msg.str());
  }

  gpuMemcpyAsync<real_t>(S, d_S, static_cast<size_t>(k), gpuMemcpyDeviceToHost, gpustream);
  if(d_U != nullptr) gpuMemcpyAsync<T>(U, d_U, u_size, gpuMemcpyDeviceToHost, gpustream);
  if(d_VT != nullptr) gpuMemcpyAsync<T>(VT, d_VT, vt_size, gpuMemcpyDeviceToHost, gpustream);
  gpuStreamSynchronize(gpustream);

  devpool.deallocate(sp_A);
  devpool.deallocate(sp_S);
  devpool.deallocate(sp_U);
  devpool.deallocate(sp_VT);
  devpool.deallocate(sp_rwork);
  devpool.deallocate(sp_work);
  devpool.deallocate(sp_info);

  CUSOLVER_CHECK(cusolverDnDestroy(handle));

#elif defined(USE_HIP)
  EXPECTS(m >= n);

  const rocblas_int im = static_cast<rocblas_int>(m), in = static_cast<rocblas_int>(n);
  const rocblas_int ilda = static_cast<rocblas_int>(lda), ildu = static_cast<rocblas_int>(ldu),
                    ildvt = static_cast<rocblas_int>(ldvt);
  const rocblas_svect ju  = job_to_rocblas_svect(jobu);
  const rocblas_svect jvt = job_to_rocblas_svect(jobvt);

  // rocSOLVER shares rocBLAS's handle (already bound to this stream by GPUStreamPool) rather
  // than needing a separate solver handle, and it manages its own device workspace internally
  // (no explicit buffer-size query/allocation like cusolverDn's lwork).
  rocblas_handle handle = gpustream.second;

  const size_t e_size = static_cast<size_t>(std::max<int64_t>(1, k - 1));

  // The spans own the pool blocks; .data() is what the rocSOLVER API takes. Each
  // deallocate() below derives its byte count from the span, so it cannot disagree with
  // the allocation.
  std::span<T>           sp_A    = devpool.allocate_span<T>(a_size);
  std::span<real_t>      sp_S    = devpool.allocate_span<real_t>(static_cast<size_t>(k));
  std::span<T>           sp_U    = devpool.allocate_span<T>(u_size);
  std::span<T>           sp_V    = devpool.allocate_span<T>(vt_size);
  std::span<real_t>      sp_E    = devpool.allocate_span<real_t>(e_size);
  std::span<rocblas_int> sp_info = devpool.allocate_span<rocblas_int>(1);

  T*           d_A    = sp_A.data();
  real_t*      d_S    = sp_S.data();
  T*           d_U    = u_size ? sp_U.data() : nullptr;
  T*           d_V    = vt_size ? sp_V.data() : nullptr;
  real_t*      d_E    = sp_E.data();
  rocblas_int* d_info = sp_info.data();

  gpuMemcpyAsync<T>(d_A, A, a_size, gpuMemcpyHostToDevice, gpustream);

  // rocSOLVER's "V" output is V**H (LAPACK's VT convention: ldv x n), matching our VT/vt_size.
  if constexpr(std::is_same_v<T, double>) {
    ROCBLAS_CHECK(rocsolver_dgesvd(handle, ju, jvt, im, in, d_A, ilda, d_S, d_U, ildu, d_V, ildvt,
                                   d_E, rocblas_outofplace, d_info));
  }
  else if constexpr(std::is_same_v<T, std::complex<double>>) {
    ROCBLAS_CHECK(rocsolver_zgesvd(handle, ju, jvt, im, in, (rocblas_double_complex*) d_A, ilda,
                                   d_S, (rocblas_double_complex*) d_U, ildu,
                                   (rocblas_double_complex*) d_V, ildvt, d_E, rocblas_outofplace,
                                   d_info));
  }

  gpuStreamSynchronize(gpustream);

  rocblas_int info = 0;
  HIP_CHECK(hipMemcpy(&info, d_info, sizeof(rocblas_int), hipMemcpyDeviceToHost));
  if(info != 0) {
    std::ostringstream msg;
    msg << "rocsolver gesvd failed with info=" << info << " (nonzero: did not converge)";
    throw std::runtime_error(msg.str());
  }

  gpuMemcpyAsync<real_t>(S, d_S, static_cast<size_t>(k), gpuMemcpyDeviceToHost, gpustream);
  if(d_U != nullptr) gpuMemcpyAsync<T>(U, d_U, u_size, gpuMemcpyDeviceToHost, gpustream);
  if(d_V != nullptr) gpuMemcpyAsync<T>(VT, d_V, vt_size, gpuMemcpyDeviceToHost, gpustream);
  gpuStreamSynchronize(gpustream);

  devpool.deallocate(sp_A);
  devpool.deallocate(sp_S);
  devpool.deallocate(sp_U);
  devpool.deallocate(sp_V);
  devpool.deallocate(sp_E);
  devpool.deallocate(sp_info);

#elif defined(USE_DPCPP)
  // oneMKL's gesvd has no documented m>=n restriction (unlike cusolverDn/rocsolver), so
  // tamm::svd's uniform M>=N gate for the GPU path is conservative here, not a hard
  // requirement of this backend.
  sycl::queue& q = gpustream.first;

  const oneapi::mkl::jobsvd ju  = job_to_oneapi_jobsvd(jobu);
  const oneapi::mkl::jobsvd jvt = job_to_oneapi_jobsvd(jobvt);

  // The spans own the pool blocks; .data() is what the oneMKL API takes. Each deallocate()
  // below derives its byte count from the span, so it cannot disagree with the allocation.
  std::span<T>      sp_A  = devpool.allocate_span<T>(a_size);
  std::span<real_t> sp_S  = devpool.allocate_span<real_t>(static_cast<size_t>(k));
  std::span<T>      sp_U  = devpool.allocate_span<T>(u_size);
  std::span<T>      sp_VT = devpool.allocate_span<T>(vt_size);

  T*      d_A  = sp_A.data();
  real_t* d_S  = sp_S.data();
  T*      d_U  = u_size ? sp_U.data() : nullptr;
  T*      d_VT = vt_size ? sp_VT.data() : nullptr;

  gpuMemcpyAsync<T>(d_A, A, a_size, gpuMemcpyHostToDevice, gpustream);

  // Declared outside the try so the deallocate below is in scope on the success path.
  std::span<T> sp_scratch;
  try {
    std::int64_t const scratchpad_size =
      oneapi::mkl::lapack::gesvd_scratchpad_size<T>(q, ju, jvt, m, n, lda, ldu, ldvt);
    sp_scratch =
      devpool.allocate_span<T>(static_cast<size_t>(std::max<std::int64_t>(scratchpad_size, 0)));

    auto ev = oneapi::mkl::lapack::gesvd(q, ju, jvt, m, n, d_A, lda, d_S, d_U, ldu, d_VT, ldvt,
                                         sp_scratch.data(), scratchpad_size);
    ev.wait();
  } catch(oneapi::mkl::exception const& ex) {
    std::ostringstream msg;
    msg << "oneMKL LAPACK gesvd Error: " << ex.what() << ", at " << __FILE__ << " : " << __LINE__;
    throw std::runtime_error(msg.str());
  }

  gpuMemcpyAsync<real_t>(S, d_S, static_cast<size_t>(k), gpuMemcpyDeviceToHost, gpustream);
  if(d_U != nullptr) gpuMemcpyAsync<T>(U, d_U, u_size, gpuMemcpyDeviceToHost, gpustream);
  if(d_VT != nullptr) gpuMemcpyAsync<T>(VT, d_VT, vt_size, gpuMemcpyDeviceToHost, gpustream);
  gpuStreamSynchronize(gpustream);

  devpool.deallocate(sp_A);
  devpool.deallocate(sp_S);
  devpool.deallocate(sp_U);
  devpool.deallocate(sp_VT);
  devpool.deallocate(sp_scratch);
#endif
}

template void tamm::kernels::gpu::gesvd(lapack::Job jobu, lapack::Job jobvt, int64_t m, int64_t n,
                                        double* A, int64_t lda, double* S, double* U, int64_t ldu,
                                        double* VT, int64_t ldvt);

template void tamm::kernels::gpu::gesvd(lapack::Job jobu, lapack::Job jobvt, int64_t m, int64_t n,
                                        std::complex<double>* A, int64_t lda, double* S,
                                        std::complex<double>* U, int64_t ldu,
                                        std::complex<double>* VT, int64_t ldvt);

// GPU counterpart of lapack::geqrf (see tamm_blas.hpp): cusolverDn<t>geqrf (CUDA),
// rocsolver_<t>geqrf (HIP), or oneapi::mkl::lapack::geqrf (DPCPP). A/tau are host pointers; this
// function stages A to the device, runs the vendor geqrf, and copies A/tau back to host,
// mirroring lapack::geqrf's host-in/host-out contract. No shape restriction: any m, n.
template<typename T>
void tamm::kernels::gpu::geqrf(int64_t m, int64_t n, T* A, int64_t lda, T* tau) {
  const int64_t k = std::min(m, n);

  auto& gpustream = tamm::GPUStreamPool::getInstance().getStream();
  auto& devpool   = tamm::RMMMemoryManager::getInstance().getDeviceMemoryPool();

  const size_t a_size = static_cast<size_t>(lda) * static_cast<size_t>(n);

  // The spans own the pool blocks; .data() is what the vendor API takes. Each deallocate()
  // below derives its byte count from the span, so it cannot disagree with the allocation.
  std::span<T> sp_A   = devpool.allocate_span<T>(a_size);
  std::span<T> sp_tau = devpool.allocate_span<T>(static_cast<size_t>(k));

  T* d_A   = sp_A.data();
  T* d_tau = sp_tau.data();

  gpuMemcpyAsync<T>(d_A, A, a_size, gpuMemcpyHostToDevice, gpustream);

#if defined(USE_CUDA)
  const int im = static_cast<int>(m), in = static_cast<int>(n), ilda = static_cast<int>(lda);

  cudaStream_t stream = gpustream.first;

  cusolverDnHandle_t handle;
  CUSOLVER_CHECK(cusolverDnCreate(&handle));
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));

  std::span<int> sp_info = devpool.allocate_span<int>(1);
  int*           d_info  = sp_info.data();

  int lwork = 0;
  if constexpr(std::is_same_v<T, double>) {
    CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(handle, im, in, d_A, ilda, &lwork));
  }
  else if constexpr(std::is_same_v<T, std::complex<double>>) {
    CUSOLVER_CHECK(
      cusolverDnZgeqrf_bufferSize(handle, im, in, (cuDoubleComplex*) d_A, ilda, &lwork));
  }

  std::span<T> sp_work = devpool.allocate_span<T>(static_cast<size_t>(std::max(lwork, 0)));
  T*           d_work  = sp_work.data();

  if constexpr(std::is_same_v<T, double>) {
    CUSOLVER_CHECK(cusolverDnDgeqrf(handle, im, in, d_A, ilda, d_tau, d_work, lwork, d_info));
  }
  else if constexpr(std::is_same_v<T, std::complex<double>>) {
    CUSOLVER_CHECK(cusolverDnZgeqrf(handle, im, in, (cuDoubleComplex*) d_A, ilda,
                                    (cuDoubleComplex*) d_tau, (cuDoubleComplex*) d_work, lwork,
                                    d_info));
  }

  gpuStreamSynchronize(gpustream);

  int info = 0;
  CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
  if(info != 0) {
    std::ostringstream msg;
    msg << "cusolverDn geqrf failed with info=" << info << " (illegal argument -info)";
    throw std::runtime_error(msg.str());
  }

  devpool.deallocate(sp_work);
  devpool.deallocate(sp_info);

  CUSOLVER_CHECK(cusolverDnDestroy(handle));

#elif defined(USE_HIP)
  const rocblas_int im = static_cast<rocblas_int>(m), in = static_cast<rocblas_int>(n),
                    ilda = static_cast<rocblas_int>(lda);

  // rocSOLVER shares rocBLAS's handle (already bound to this stream by GPUStreamPool) and
  // manages its own device workspace internally.
  rocblas_handle handle = gpustream.second;

  if constexpr(std::is_same_v<T, double>) {
    ROCBLAS_CHECK(rocsolver_dgeqrf(handle, im, in, d_A, ilda, d_tau));
  }
  else if constexpr(std::is_same_v<T, std::complex<double>>) {
    ROCBLAS_CHECK(rocsolver_zgeqrf(handle, im, in, (rocblas_double_complex*) d_A, ilda,
                                   (rocblas_double_complex*) d_tau));
  }

  gpuStreamSynchronize(gpustream);

#elif defined(USE_DPCPP)
  sycl::queue& q = gpustream.first;

  // Declared outside the try so the deallocate below is in scope on the success path.
  std::span<T> sp_scratch;
  try {
    std::int64_t const scratchpad_size =
      oneapi::mkl::lapack::geqrf_scratchpad_size<T>(q, m, n, lda);
    sp_scratch =
      devpool.allocate_span<T>(static_cast<size_t>(std::max<std::int64_t>(scratchpad_size, 0)));

    auto ev =
      oneapi::mkl::lapack::geqrf(q, m, n, d_A, lda, d_tau, sp_scratch.data(), scratchpad_size);
    ev.wait();
  } catch(oneapi::mkl::exception const& ex) {
    std::ostringstream msg;
    msg << "oneMKL LAPACK geqrf Error: " << ex.what() << ", at " << __FILE__ << " : " << __LINE__;
    throw std::runtime_error(msg.str());
  }

  devpool.deallocate(sp_scratch);
#endif

  gpuMemcpyAsync<T>(A, d_A, a_size, gpuMemcpyDeviceToHost, gpustream);
  gpuMemcpyAsync<T>(tau, d_tau, static_cast<size_t>(k), gpuMemcpyDeviceToHost, gpustream);
  gpuStreamSynchronize(gpustream);

  devpool.deallocate(sp_A);
  devpool.deallocate(sp_tau);
}

template void tamm::kernels::gpu::geqrf(int64_t m, int64_t n, double* A, int64_t lda, double* tau);
template void tamm::kernels::gpu::geqrf(int64_t m, int64_t n, std::complex<double>* A, int64_t lda,
                                        std::complex<double>* tau);

// GPU counterpart of lapack::ungqr (see tamm_blas.hpp): cusolverDn<t>orgqr/ungqr (CUDA),
// rocsolver_<t>orgqr/ungqr (HIP), or oneapi::mkl::lapack::orgqr/ungqr (DPCPP). A/tau are host
// pointers; this function stages A/tau to the device, runs the vendor orgqr/ungqr, and copies A
// back to host, mirroring lapack::ungqr's host-in/host-out contract. No shape restriction
// beyond ungqr's own k <= n <= m (see tamm_blas.hpp).
template<typename T>
void tamm::kernels::gpu::ungqr(int64_t m, int64_t n, int64_t k, T* A, int64_t lda, const T* tau) {
  auto& gpustream = tamm::GPUStreamPool::getInstance().getStream();
  auto& devpool   = tamm::RMMMemoryManager::getInstance().getDeviceMemoryPool();

  const size_t a_size = static_cast<size_t>(lda) * static_cast<size_t>(n);

  // The spans own the pool blocks; .data() is what the vendor API takes. Each deallocate()
  // below derives its byte count from the span, so it cannot disagree with the allocation.
  std::span<T> sp_A   = devpool.allocate_span<T>(a_size);
  std::span<T> sp_tau = devpool.allocate_span<T>(static_cast<size_t>(k));

  T* d_A   = sp_A.data();
  T* d_tau = sp_tau.data();

  gpuMemcpyAsync<T>(d_A, A, a_size, gpuMemcpyHostToDevice, gpustream);
  gpuMemcpyAsync<T>(d_tau, tau, static_cast<size_t>(k), gpuMemcpyHostToDevice, gpustream);

#if defined(USE_CUDA)
  const int im = static_cast<int>(m), in = static_cast<int>(n), ik = static_cast<int>(k),
            ilda = static_cast<int>(lda);

  cudaStream_t stream = gpustream.first;

  cusolverDnHandle_t handle;
  CUSOLVER_CHECK(cusolverDnCreate(&handle));
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));

  std::span<int> sp_info = devpool.allocate_span<int>(1);
  int*           d_info  = sp_info.data();

  int lwork = 0;
  if constexpr(std::is_same_v<T, double>) {
    CUSOLVER_CHECK(cusolverDnDorgqr_bufferSize(handle, im, in, ik, d_A, ilda, d_tau, &lwork));
  }
  else if constexpr(std::is_same_v<T, std::complex<double>>) {
    CUSOLVER_CHECK(cusolverDnZungqr_bufferSize(handle, im, in, ik, (cuDoubleComplex*) d_A, ilda,
                                               (cuDoubleComplex*) d_tau, &lwork));
  }

  std::span<T> sp_work = devpool.allocate_span<T>(static_cast<size_t>(std::max(lwork, 0)));
  T*           d_work  = sp_work.data();

  if constexpr(std::is_same_v<T, double>) {
    CUSOLVER_CHECK(cusolverDnDorgqr(handle, im, in, ik, d_A, ilda, d_tau, d_work, lwork, d_info));
  }
  else if constexpr(std::is_same_v<T, std::complex<double>>) {
    CUSOLVER_CHECK(cusolverDnZungqr(handle, im, in, ik, (cuDoubleComplex*) d_A, ilda,
                                    (cuDoubleComplex*) d_tau, (cuDoubleComplex*) d_work, lwork,
                                    d_info));
  }

  gpuStreamSynchronize(gpustream);

  int info = 0;
  CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
  if(info != 0) {
    std::ostringstream msg;
    msg << "cusolverDn orgqr/ungqr failed with info=" << info << " (illegal argument -info)";
    throw std::runtime_error(msg.str());
  }

  devpool.deallocate(sp_work);
  devpool.deallocate(sp_info);

  CUSOLVER_CHECK(cusolverDnDestroy(handle));

#elif defined(USE_HIP)
  const rocblas_int im = static_cast<rocblas_int>(m), in = static_cast<rocblas_int>(n),
                    ik = static_cast<rocblas_int>(k), ilda = static_cast<rocblas_int>(lda);

  // rocSOLVER shares rocBLAS's handle (already bound to this stream by GPUStreamPool) and
  // manages its own device workspace internally.
  rocblas_handle handle = gpustream.second;

  if constexpr(std::is_same_v<T, double>) {
    ROCBLAS_CHECK(rocsolver_dorgqr(handle, im, in, ik, d_A, ilda, d_tau));
  }
  else if constexpr(std::is_same_v<T, std::complex<double>>) {
    ROCBLAS_CHECK(rocsolver_zungqr(handle, im, in, ik, (rocblas_double_complex*) d_A, ilda,
                                   (rocblas_double_complex*) d_tau));
  }

  gpuStreamSynchronize(gpustream);

#elif defined(USE_DPCPP)
  sycl::queue& q = gpustream.first;

  // Declared outside the try so the deallocate below is in scope on the success path.
  std::span<T> sp_scratch;
  try {
    std::int64_t scratchpad_size = 0;
    if constexpr(std::is_same_v<T, double>) {
      scratchpad_size = oneapi::mkl::lapack::orgqr_scratchpad_size<T>(q, m, n, k, lda);
    }
    else if constexpr(std::is_same_v<T, std::complex<double>>) {
      scratchpad_size = oneapi::mkl::lapack::ungqr_scratchpad_size<T>(q, m, n, k, lda);
    }
    sp_scratch =
      devpool.allocate_span<T>(static_cast<size_t>(std::max<std::int64_t>(scratchpad_size, 0)));

    if constexpr(std::is_same_v<T, double>) {
      auto ev =
        oneapi::mkl::lapack::orgqr(q, m, n, k, d_A, lda, d_tau, sp_scratch.data(), scratchpad_size);
      ev.wait();
    }
    else if constexpr(std::is_same_v<T, std::complex<double>>) {
      auto ev =
        oneapi::mkl::lapack::ungqr(q, m, n, k, d_A, lda, d_tau, sp_scratch.data(), scratchpad_size);
      ev.wait();
    }
  } catch(oneapi::mkl::exception const& ex) {
    std::ostringstream msg;
    msg << "oneMKL LAPACK orgqr/ungqr Error: " << ex.what() << ", at " << __FILE__ << " : "
        << __LINE__;
    throw std::runtime_error(msg.str());
  }

  devpool.deallocate(sp_scratch);
#endif

  gpuMemcpyAsync<T>(A, d_A, a_size, gpuMemcpyDeviceToHost, gpustream);
  gpuStreamSynchronize(gpustream);

  devpool.deallocate(sp_A);
  devpool.deallocate(sp_tau);
}

template void tamm::kernels::gpu::ungqr(int64_t m, int64_t n, int64_t k, double* A, int64_t lda,
                                        const double* tau);
template void tamm::kernels::gpu::ungqr(int64_t m, int64_t n, int64_t k, std::complex<double>* A,
                                        int64_t lda, const std::complex<double>* tau);

namespace tamm::kernels::gpu {
namespace {
// Eigendecomposition of the column-major n x n device matrix d_A (lower triangle used) in place on
// gpustream: d_A receives the eigenvectors (columns) and d_W the eigenvalues in ascending order.
// Synchronizes the stream; terminates on a backend error.
template<typename T>
void syevd_device(int64_t n, T* d_A, int64_t lda, T* d_W, tamm::gpuStream_t& gpustream) {
  static_assert(std::is_same_v<T, double>, "tamm::kernels::gpu::syevd: only double is supported");

  auto& devpool = tamm::RMMMemoryManager::getInstance().getDeviceMemoryPool();

#if defined(USE_CUDA)
  cudaStream_t stream = gpustream.first;

  cusolverDnHandle_t handle;
  cusolverDnParams_t params;
  CUSOLVER_CHECK(cusolverDnCreate(&handle));
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  CUSOLVER_CHECK(cusolverDnCreateParams(&params));

  std::span<int> sp_info = devpool.allocate_span<int>(1);

  // The 64-bit API: the legacy cusolverDnDsyevd sizes its workspace with an int, which
  // overflows for n in the tens of thousands.
  size_t dev_bytes = 0, host_bytes = 0;
  CUSOLVER_CHECK(cusolverDnXsyevd_bufferSize(handle, params, CUSOLVER_EIG_MODE_VECTOR,
                                             CUBLAS_FILL_MODE_LOWER, n, CUDA_R_64F, d_A, lda,
                                             CUDA_R_64F, d_W, CUDA_R_64F, &dev_bytes, &host_bytes));

  // Sized in bytes; allocated in units of T so the pool block keeps T's alignment.
  std::span<T>      sp_work = devpool.allocate_span<T>((dev_bytes + sizeof(T) - 1) / sizeof(T));
  std::vector<char> host_work(host_bytes);

  CUSOLVER_CHECK(cusolverDnXsyevd(handle, params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                                  n, CUDA_R_64F, d_A, lda, CUDA_R_64F, d_W, CUDA_R_64F,
                                  sp_work.data(), dev_bytes, host_work.data(), host_bytes,
                                  sp_info.data()));

  gpuStreamSynchronize(gpustream);

  int info = 0;
  CUDA_CHECK(cudaMemcpy(&info, sp_info.data(), sizeof(int), cudaMemcpyDeviceToHost));
  if(info != 0)
    tamm_terminate("[TAMM ERROR] cusolverDn syevd failed with info = " + std::to_string(info));

  devpool.deallocate(sp_work);
  devpool.deallocate(sp_info);

  CUSOLVER_CHECK(cusolverDnDestroyParams(params));
  CUSOLVER_CHECK(cusolverDnDestroy(handle));

#elif defined(USE_HIP)
  // rocSOLVER shares rocBLAS's handle (already bound to this stream by GPUStreamPool) and
  // manages its own device workspace internally.
  rocblas_handle    handle = gpustream.second;
  const rocblas_int in = static_cast<rocblas_int>(n), ilda = static_cast<rocblas_int>(lda);

  std::span<T>           sp_E    = devpool.allocate_span<T>(static_cast<size_t>(n));
  std::span<rocblas_int> sp_info = devpool.allocate_span<rocblas_int>(1);

  ROCBLAS_CHECK(rocsolver_dsyevd(handle, rocblas_evect_original, rocblas_fill_lower, in, d_A, ilda,
                                 d_W, sp_E.data(), sp_info.data()));

  gpuStreamSynchronize(gpustream);

  rocblas_int info = 0;
  HIP_CHECK(hipMemcpy(&info, sp_info.data(), sizeof(rocblas_int), hipMemcpyDeviceToHost));
  if(info != 0)
    tamm_terminate("[TAMM ERROR] rocsolver syevd failed with info = " + std::to_string(info));

  devpool.deallocate(sp_E);
  devpool.deallocate(sp_info);

#elif defined(USE_DPCPP)
  sycl::queue& q = gpustream.first;

  // Declared outside the try so the deallocate below is in scope on the success path.
  std::span<T> sp_scratch;
  try {
    std::int64_t const scratchpad_size = oneapi::mkl::lapack::syevd_scratchpad_size<T>(
      q, oneapi::mkl::job::vec, oneapi::mkl::uplo::lower, n, lda);
    sp_scratch =
      devpool.allocate_span<T>(static_cast<size_t>(std::max<std::int64_t>(scratchpad_size, 0)));

    auto ev = oneapi::mkl::lapack::syevd(q, oneapi::mkl::job::vec, oneapi::mkl::uplo::lower, n, d_A,
                                         lda, d_W, sp_scratch.data(), scratchpad_size);
    ev.wait();
  } catch(oneapi::mkl::exception const& ex) {
    tamm_terminate(std::string("[TAMM ERROR] oneMKL LAPACK syevd failed: ") + ex.what());
  }

  devpool.deallocate(sp_scratch);
#endif
}
} // namespace
} // namespace tamm::kernels::gpu

template<typename T>
void tamm::kernels::gpu::syevd(int64_t n, T* A, int64_t lda, T* W) {
  auto& gpustream = tamm::GPUStreamPool::getInstance().getStream();
  auto& devpool   = tamm::RMMMemoryManager::getInstance().getDeviceMemoryPool();

  const size_t a_size = static_cast<size_t>(lda) * static_cast<size_t>(n);

  // The spans own the pool blocks; each deallocate() derives its byte count from the span, so
  // it cannot disagree with the allocation.
  std::span<T> sp_A = devpool.allocate_span<T>(a_size);
  std::span<T> sp_W = devpool.allocate_span<T>(static_cast<size_t>(n));

  gpuMemcpyAsync<T>(sp_A.data(), A, a_size, gpuMemcpyHostToDevice, gpustream);
  syevd_device(n, sp_A.data(), lda, sp_W.data(), gpustream);

  gpuMemcpyAsync<T>(A, sp_A.data(), a_size, gpuMemcpyDeviceToHost, gpustream);
  gpuMemcpyAsync<T>(W, sp_W.data(), static_cast<size_t>(n), gpuMemcpyDeviceToHost, gpustream);
  gpuStreamSynchronize(gpustream);

  devpool.deallocate(sp_A);
  devpool.deallocate(sp_W);
}

template void tamm::kernels::gpu::syevd(int64_t n, double* A, int64_t lda, double* W);

template<typename T>
void tamm::kernels::gpu::generalized_eigensolve(int64_t N, int64_t M, T* F, const T* X, T* C,
                                                T* W) {
  static_assert(std::is_same_v<T, double>,
                "tamm::kernels::gpu::generalized_eigensolve: only double is supported");

  auto& gpustream = tamm::GPUStreamPool::getInstance().getStream();
  auto& devpool   = tamm::RMMMemoryManager::getInstance().getDeviceMemoryPool();

  const size_t nn = static_cast<size_t>(N) * static_cast<size_t>(N);
  const size_t nm = static_cast<size_t>(N) * static_cast<size_t>(M);
  const size_t mm = static_cast<size_t>(M) * static_cast<size_t>(M);
  const int    in = static_cast<int>(N), im = static_cast<int>(M);

  std::span<T> sp_F = devpool.allocate_span<T>(nn); // F, then F' = X^T F X in its first M*M
  std::span<T> sp_X = devpool.allocate_span<T>(nm);
  std::span<T> sp_C = devpool.allocate_span<T>(nm); // F X, then C = X C'
  std::span<T> sp_W = devpool.allocate_span<T>(static_cast<size_t>(M));

  gpuMemcpyAsync<T>(sp_F.data(), F, nn, gpuMemcpyHostToDevice, gpustream);
  gpuMemcpyAsync<T>(sp_X.data(), X, nm, gpuMemcpyHostToDevice, gpustream);

  // The same column-major gemms as the host path in tamm_linalg.cpp: the row-major N x N F,
  // N x M X and N x M C are seen as F, X^T (M x N) and C^T (M x N).
  // C (as N x M column-major scratch) = F * X
  gpu::gemm<T>(blas::Op::NoTrans, blas::Op::Trans, in, im, in, 1., sp_F.data(), in, sp_X.data(), im,
               0., sp_C.data(), in, gpustream);
  // F (as M x M) = X^T * F * X
  gpu::gemm<T>(blas::Op::NoTrans, blas::Op::NoTrans, im, im, in, 1., sp_X.data(), im, sp_C.data(),
               in, 0., sp_F.data(), im, gpustream);
  syevd_device(M, sp_F.data(), M, sp_W.data(), gpustream);
  // C (row-major N x M) = X * C'
  gpu::gemm<T>(blas::Op::Trans, blas::Op::NoTrans, im, in, im, 1., sp_F.data(), im, sp_X.data(), im,
               0., sp_C.data(), im, gpustream);

  // F gets what the host path leaves there: C' in its first M*M entries, the rest unchanged.
  gpuMemcpyAsync<T>(F, sp_F.data(), mm, gpuMemcpyDeviceToHost, gpustream);
  gpuMemcpyAsync<T>(C, sp_C.data(), nm, gpuMemcpyDeviceToHost, gpustream);
  gpuMemcpyAsync<T>(W, sp_W.data(), static_cast<size_t>(M), gpuMemcpyDeviceToHost, gpustream);
  gpuStreamSynchronize(gpustream);

  devpool.deallocate(sp_F);
  devpool.deallocate(sp_X);
  devpool.deallocate(sp_C);
  devpool.deallocate(sp_W);
}

template void tamm::kernels::gpu::generalized_eigensolve(int64_t N, int64_t M, double* F,
                                                         const double* X, double* C, double* W);

#endif // USE_CUDA || USE_HIP || USE_DPCPP
