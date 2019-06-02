#pragma once

#include <cstdint>
#include <type_traits>

// #include LAPACKE_HEADER
// #ifdef lapack_int
//   #undef lapack_int
// #endif

namespace linalg {

// #ifdef LINALG_ILP64
  using nwx_blas_int   = TAMM_LAPACK_INT;
  using nwx_lapack_int = TAMM_LAPACK_INT;
// #elif defined(MKL_INT)
//   using nwx_blas_int   = MKL_INT;
//   using nwx_lapack_int = MKL_INT;
// #else
//   using nwx_blas_int   = int32_t;
//   using nwx_lapack_int = int32_t;
// #endif

  namespace detail {

    template <typename T>
    struct real_type {
      using value_type = T;
    };

    template <typename T>
    struct real_type< std::complex<T> > {
      using value_type = T;
    };


    template <typename T>
    struct is_linalg_supported : 
      std::false_type { };
    template <>
    struct is_linalg_supported<double> : 
      std::true_type { };
    template <>
    struct is_linalg_supported<float> : 
      std::true_type { };
    //template <>
    //struct is_linalg_supported<std::complex<double>> : 
    //  std::true_type { };
    //template <>
    //struct is_linalg_supported<std::complex<float>> : 
    //  std::true_type { };

    template <typename T>
    inline constexpr bool is_linalg_supported_v = 
      is_linalg_supported<T>::value;
    

    template <typename T, typename U = void>
    using enable_if_linalg_supported_t = typename
      std::enable_if< is_linalg_supported_v<T>, U>::type;

  }
};

//#define NWX_USE_MKL
#ifndef INTEL_MKL_VERSION

#define LINALG_NAME(NAME) NAME##_

extern "C" {

//   // BLAS Prototypes
      

//   // TODO: handle implicit return for e.g. ESSL
//   double dnrm2_( const linalg::nwx_blas_int*, const double*, 
//                  const linalg::nwx_blas_int* );

//   // TODO: handle implicit return for e.g. ESSL
//   double ddot_( const linalg::nwx_blas_int*, 
//                 const double*, const linalg::nwx_blas_int*,
//                 const double*, const linalg::nwx_blas_int* );
                 
  void dcopy_( const linalg::nwx_blas_int*,
               const double*, const linalg::nwx_blas_int*,
               double*, const linalg::nwx_blas_int* );
          

//   void daxpy_( const linalg::nwx_blas_int*, const double*, 
//                const double*, const linalg::nwx_blas_int*, double*, 
//                const linalg::nwx_blas_int* );

  void dscal_( const linalg::nwx_blas_int*, const double*, double*, 
               const linalg::nwx_blas_int* );

//   void dswap_( const linalg::nwx_blas_int*, double*, 
//                const linalg::nwx_blas_int*, double*, 
//                const linalg::nwx_blas_int* );

  void dgemm_( char*, char*, const linalg::nwx_blas_int*, 
               const linalg::nwx_blas_int*, const linalg::nwx_blas_int*,  
               const double*, const double*, 
               const linalg::nwx_blas_int*, const double*, 
               const linalg::nwx_blas_int*, const double*, double *, 
               const linalg::nwx_blas_int* );

//   void dsyr2k_( char*, char*, const linalg::nwx_blas_int*,
//                 const linalg::nwx_blas_int*, const double*, 
//                 const double*, const linalg::nwx_blas_int*,
//                 const double*, const linalg::nwx_blas_int*,
//                 const double*, double*, const linalg::nwx_blas_int* );

//   void dtrsm_( char*, char*, char*, char*,
//                const linalg::nwx_blas_int*, const linalg::nwx_blas_int*, 
//                const double*, const double*, 
//                const linalg::nwx_blas_int*, double*, 
//                const linalg::nwx_blas_int* );



//   // LAPACK Prototypes
//   void dpotrf_( char*,const linalg::nwx_lapack_int*, double*, 
//                 const linalg::nwx_lapack_int*, linalg::nwx_lapack_int* );

//   void dgetrf_( const linalg::nwx_lapack_int*, 
//                 const linalg::nwx_lapack_int*, double*, 
//                 const linalg::nwx_lapack_int*, linalg::nwx_lapack_int *,
//                 linalg::nwx_lapack_int* );

//   void dgetrs_( char*, const linalg::nwx_lapack_int*, 
//                 const linalg::nwx_lapack_int*, const double*, 
//                 const linalg::nwx_lapack_int*, 
//                 const linalg::nwx_lapack_int*, double*, 
//                 const linalg::nwx_lapack_int*, linalg::nwx_lapack_int* );

//   void dsytrf_( char*, const linalg::nwx_lapack_int*, double*, 
//                 const linalg::nwx_lapack_int *,
//                 linalg::nwx_lapack_int*, double*, 
//                 const linalg::nwx_lapack_int*, linalg::nwx_lapack_int *);

//   void dsytrs_( char*, const linalg::nwx_lapack_int*, 
//                 const linalg::nwx_lapack_int*, const double*, 
//                 const linalg::nwx_lapack_int*, 
//                 const linalg::nwx_lapack_int*, double*, 
//                 const linalg::nwx_lapack_int*, linalg::nwx_lapack_int* );

//  #if __essl
  void dsyevd_( char*, char*, 
                const linalg::nwx_lapack_int*, double*,
                const linalg::nwx_lapack_int*, double*, double*, 
                const linalg::nwx_lapack_int*, linalg::nwx_lapack_int*,
                const linalg::nwx_lapack_int*, linalg::nwx_lapack_int* );
//  #endif

//   void dsygvd_( const linalg::nwx_lapack_int*, char*, 
//                 char*, const linalg::nwx_lapack_int*, double*,
//                 const linalg::nwx_lapack_int*, double*,
//                 const linalg::nwx_lapack_int*, double*, double*,
//                 const linalg::nwx_lapack_int*, linalg::nwx_lapack_int*,
//                 const linalg::nwx_lapack_int*, linalg::nwx_lapack_int* );

//   void ssygvd_( const linalg::nwx_lapack_int*, char*, 
//                 char*, const linalg::nwx_lapack_int*, float*,
//                 const linalg::nwx_lapack_int*, float*,
//                 const linalg::nwx_lapack_int*, float*, float*,
//                 const linalg::nwx_lapack_int*, linalg::nwx_lapack_int*,
//                 const linalg::nwx_lapack_int*, linalg::nwx_lapack_int* );
};
#else

#define LINALG_NAME(NAME) NAME
#endif



namespace linalg {

/*
namespace error {

  class linalg_error : public std::exception {

  };

};
*/

namespace blas {

  // NRM2
  template <typename F>
  detail::enable_if_linalg_supported_t<F,F>
  nrm2( const int64_t N, const F* X, const int64_t INCX ) {

    nwx_blas_int _N    = N;
    nwx_blas_int _INCX = INCX;

    if constexpr (std::is_same_v<F,double>)
      return LINALG_NAME(dnrm2)(&_N, X, &_INCX);

  }


  // DOT
  template <typename F>
  detail::enable_if_linalg_supported_t<F,F>
  dot( const int64_t N, const F* X, const int64_t INCX,
         const F* Y, const int64_t INCY ) {

    nwx_blas_int _N    = N;
    nwx_blas_int _INCX = INCX;
    nwx_blas_int _INCY = INCY;
  
    if constexpr (std::is_same_v<F,double>)
      return LINALG_NAME(ddot)(&_N, X, &_INCX, Y, &_INCY);
    
  }

  template <typename F>
  detail::enable_if_linalg_supported_t<F,F>
  dot( const int64_t N, const F* X, const F* Y) {

    return dot(N,X,1,Y,1);

  }


  // COPY
  template <typename F>
  detail::enable_if_linalg_supported_t<F>
  copy( const int64_t N, const F* X, const int64_t INCX,
             F* Y, const int64_t INCY ) {

    nwx_blas_int _N    = N;
    nwx_blas_int _INCX = INCX;
    nwx_blas_int _INCY = INCY;
  
    if constexpr (std::is_same_v<F,double>)
      LINALG_NAME(dcopy)(&_N, X, &_INCX, Y, &_INCY);
    
  }

  template <typename F>
  detail::enable_if_linalg_supported_t<F>
  copy( const int64_t N, const F* X, F* Y) {

    copy(N,X,1,Y,1);

  }

  // AXPY
  template <typename F>
  detail::enable_if_linalg_supported_t<F>
  axpy( const int64_t N, const F ALPHA, const F* X, 
             const int64_t INCX, F* Y, const int64_t INCY ) {

    nwx_blas_int _N    = N;
    nwx_blas_int _INCX = INCX;
    nwx_blas_int _INCY = INCY;

    if constexpr (std::is_same_v<F,double>)
      LINALG_NAME(daxpy)( &_N, &ALPHA, X, &_INCX, Y, &_INCY );

  }


  // SCAL
  template <typename F>
  detail::enable_if_linalg_supported_t<F>
  scal( const int64_t N, const F ALPHA, F* X, 
             const int64_t INCX ) {

    nwx_blas_int _N    = N;
    nwx_blas_int _INCX = INCX;

    if constexpr (std::is_same_v<F,double>)
      LINALG_NAME(dscal)(&_N, &ALPHA, X, &_INCX);
  }


  // SWAP
  template <typename F>
  detail::enable_if_linalg_supported_t<F>
  swap( const int64_t N, F* X, const int64_t INCX, F* Y, 
             const int64_t INCY ) {

    nwx_blas_int _N    = N;
    nwx_blas_int _INCX = INCX;
    nwx_blas_int _INCY = INCY;

    if constexpr (std::is_same_v<F,double>)
      LINALG_NAME(dswap)( &_N, X, &_INCX, Y, &_INCY );

  }


  // GEMM
  template <typename F>
  detail::enable_if_linalg_supported_t<F>
  gemm( 
    char TRANSA, char TRANSB,
    const int64_t M, const int64_t N, const int64_t K,
    const F ALPHA, const F* A, const int64_t LDA,
    const F* B, const int64_t LDB,
    const F BETA, F* C, const int64_t LDC) {

    nwx_blas_int _M   = M;
    nwx_blas_int _N   = N;
    nwx_blas_int _K   = K;
    nwx_blas_int _LDA = LDA;
    nwx_blas_int _LDB = LDB;
    nwx_blas_int _LDC = LDC;

    if constexpr (std::is_same_v<F,double>)
      LINALG_NAME(dgemm)(&TRANSA, &TRANSB, &_M, &_N, &_K, 
                         &ALPHA, A, &_LDA, B, &_LDB, &BETA, 
                         C, &_LDC );

  }

  // SYR2K
  template <typename F>
  detail::enable_if_linalg_supported_t<F>
  syr2k( char UPLO, char TRANS, const int64_t N,
              const int64_t K, const F ALPHA, const F* A,
              const int64_t LDA, const F* B, const int64_t LDB,
              const F BETA, F* C, const int64_t LDC ) {

    nwx_blas_int _N   = N;
    nwx_blas_int _K   = K;
    nwx_blas_int _LDA = LDA;
    nwx_blas_int _LDB = LDB;
    nwx_blas_int _LDC = LDC;
  
    if constexpr (std::is_same_v<F,double>)
      LINALG_NAME(dsyr2k)( &UPLO, &TRANS, &_N, &_K, &ALPHA, A, &_LDA, B,
                           &_LDB, &BETA, C, &_LDC );

  }


  // TRSM
  template <typename F>
  detail::enable_if_linalg_supported_t<F>
  trsm( char SIDE, char UPLO, char TRANSA,
             char DIAG, const int64_t M, const int64_t N, 
             const F ALPHA, const F* A, const int64_t LDA,
             F* B, const int64_t LDB ) {

    nwx_blas_int _M   = M;
    nwx_blas_int _N   = N;
    nwx_blas_int _LDA = LDA;
    nwx_blas_int _LDB = LDB;

    if constexpr (std::is_same_v<F,double>)
      LINALG_NAME(dtrsm)(&SIDE, &UPLO, &TRANSA, &DIAG, &_M, &_N, &ALPHA, 
                         A, &_LDA, B, &_LDB);
  }

} // namespace blas

namespace lapack {

  // Cholesky
  template <typename F>
  detail::enable_if_linalg_supported_t<F,int64_t>
  cholesky( char UPLO, const int64_t N, F *A, 
                    const int64_t LDA) {

    nwx_lapack_int _N   = N;
    nwx_lapack_int _LDA = LDA;
  
    nwx_lapack_int INFO;
    if constexpr (std::is_same_v<F,double>)
      LINALG_NAME(dpotrf)( &UPLO, &_N, A, &_LDA, &INFO );

    return int64_t(INFO);

  }

  // LU
  template <typename F>
  detail::enable_if_linalg_supported_t<F,int64_t>
  lu( const int64_t M, const int64_t N, F* A, 
              const int64_t LDA, int64_t *iPIV ) {

    nwx_lapack_int _N   = N;
    nwx_lapack_int _M   = M;
    nwx_lapack_int _LDA = LDA;
  
    std::vector< nwx_lapack_int > _iPIV(N);

    nwx_lapack_int INFO;
    if constexpr (std::is_same_v<F,double>)
      LINALG_NAME(dgetrf)(&_M, &_N, A, &_LDA, _iPIV.data(), &INFO );

    std::copy( _iPIV.begin(), _iPIV.end(), iPIV );

    return int64_t(INFO);

  }

  // LU Solve (GETRS)
  template <typename F>
  detail::enable_if_linalg_supported_t<F,int64_t>
  lu_solve( char TRANS, const int64_t N, 
                    const int64_t NRHS, const F* A,
                    const int64_t LDA, const int64_t* iPIV, 
                    F* B, const int64_t LDB ) {

    nwx_lapack_int _N     = N;
    nwx_lapack_int _NRHS  = NRHS;
    nwx_lapack_int _LDA   = LDA;
    nwx_lapack_int _LDB   = LDB;

    std::vector< nwx_lapack_int > _iPIV(N);
    std::copy_n( iPIV, N, _iPIV.data() );

    nwx_lapack_int INFO;
    if constexpr (std::is_same_v<F,double>)
      LINALG_NAME(dgetrs)( &TRANS, &_N, &_NRHS, A, &_LDA, _iPIV.data(), 
                           B, &_LDB, &INFO );

    return int64_t(INFO);

  }


  // LDLT (SYTRF)
  template <typename F>
  detail::enable_if_linalg_supported_t<F,int64_t>
  ldlt( char UPLO, const int64_t N, F* A, 
                const int64_t LDA, int64_t * IPIV ) {

    std::vector< F > WORK(3);

    nwx_lapack_int _N     = N;
    nwx_lapack_int _LDA   = LDA;
  
    std::vector< nwx_lapack_int > _iPIV(N);

    nwx_lapack_int INFO;
    nwx_lapack_int LWORK = -1;

    auto call_sytrf = [&]() {
      if constexpr (std::is_same_v<F,double>)
        LINALG_NAME(dsytrf)( &UPLO, &_N, A, &_LDA, _iPIV.data(), WORK.data(), 
                             &LWORK, &INFO );
    };

    call_sytrf();

    if( INFO == 0 ) {
      LWORK = nwx_lapack_int(WORK[0]);
      WORK.resize( LWORK );
      call_sytrf();
    }

    std::copy( _iPIV.begin(), _iPIV.end(), IPIV );

    return int64_t(INFO);

  }


  // LDLT Solve (SYTRS)
  template <typename F>
  detail::enable_if_linalg_supported_t<F,int64_t>
  ldlt_solve( char UPLO, const int64_t N, 
                      const int64_t NRHS, const F* A,
                      const int64_t LDA, const int64_t* iPIV, 
                      F* B, const int64_t LDB ) {

    nwx_lapack_int _N     = N;
    nwx_lapack_int _NRHS  = NRHS;
    nwx_lapack_int _LDA   = LDA;
    nwx_lapack_int _LDB   = LDB;

    std::vector< nwx_lapack_int > _iPIV(N);
    std::copy_n( iPIV, N, _iPIV.data() );

    nwx_lapack_int INFO;
    if constexpr (std::is_same_v<F,double>)
      LINALG_NAME(dsytrs)( &UPLO, &_N, &_NRHS, A, &_LDA, _iPIV.data(), 
                           B, &_LDB, &INFO );

    return int64_t(INFO);

  }


  // SYEVD
  template <typename F>
  detail::enable_if_linalg_supported_t<F,int64_t>
  syevd( char JOBZ, char UPLO, 
                 const int64_t N, F* A, const int64_t LDA, 
                 typename detail::real_type<F>::value_type *W ) {

    nwx_lapack_int _N     = N;
    nwx_lapack_int _LDA   = LDA;

    nwx_lapack_int LWORK  = -1;
    nwx_lapack_int LIWORK = -1;
    std::vector<F> WORK(3);
    std::vector<nwx_lapack_int>    IWORK(3);

    nwx_lapack_int INFO;
    
    auto call_syevd = [&]() {
      if constexpr (std::is_same_v<F,double>)
        LINALG_NAME(dsyevd)(&JOBZ, &UPLO, &_N, A, &_LDA, W, WORK.data(), 
                            &LWORK, IWORK.data(), &LIWORK, &INFO );
    };

    call_syevd();

    if( INFO == 0 ) {
      LWORK = nwx_lapack_int( WORK[0] );
      LIWORK = IWORK[0];

      WORK.resize( LWORK );
      IWORK.resize( LIWORK );

      call_syevd();
    }

    return int64_t(INFO);

  }

  template <typename F>
  detail::enable_if_linalg_supported_t<F,int64_t>
  sygvd( const int64_t ITYPE, char JOBZ, 
                 char UPLO, const int64_t N, F* A, 
                 const int64_t LDA, F* B, const int64_t LDB,
                 typename detail::real_type<F>::value_type *W ) {

    nwx_lapack_int _ITYPE = ITYPE;
    nwx_lapack_int _N     = N;
    nwx_lapack_int _LDA   = LDA;
    nwx_lapack_int _LDB   = LDB;

    nwx_lapack_int LWORK  = -1;
    nwx_lapack_int LIWORK = -1;
    std::vector<F> WORK(3);
    std::vector<nwx_lapack_int>    IWORK(3);

    nwx_lapack_int INFO;
    
    auto call_sygvd = [&]() {
      if constexpr (std::is_same_v<F,double>)
        LINALG_NAME(dsygvd)(&_ITYPE, &JOBZ, &UPLO, &_N, A, &_LDA, 
                            B, &_LDB, W, WORK.data(), &LWORK, 
                            IWORK.data(), &LIWORK, &INFO );
      else if constexpr (std::is_same_v<F,float>)
        LINALG_NAME(ssygvd)(&_ITYPE, &JOBZ, &UPLO, &_N, A, &_LDA, 
                            B, &_LDB, W, WORK.data(), &LWORK, 
                            IWORK.data(), &LIWORK, &INFO );
    };

    call_sygvd();

    if( INFO == 0 ) {
      LWORK = nwx_lapack_int( WORK[0] );
      LIWORK = IWORK[0];

      WORK.resize( LWORK );
      IWORK.resize( LIWORK );

      call_sygvd();
    }

    return int64_t(INFO);

  }
} // namespace lapack


namespace util {

  template <typename _F1, typename _F2>
  void submat_get(int64_t M, int64_t N, int64_t MSub, 
    int64_t NSub, _F1 *ABig, int64_t LDAB, _F2 *ASmall, 
    int64_t LDAS, 
    std::vector<std::pair<int64_t,int64_t>> &SubMatCut) {

    using mat_t1 = Eigen::Matrix<_F1,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>;
    using mat_t2 = Eigen::Matrix<_F2,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>;
    
    Eigen::Map< mat_t1 > ABigMap(ABig,LDAB,N);
    Eigen::Map< mat_t2 > ASmallMap(ASmall,LDAS,NSub);

    int64_t i(0);
    for( auto& iCut : SubMatCut ) {
      int64_t deltaI = iCut.second - iCut.first;
      int64_t j(0);
    for( auto& jCut : SubMatCut ) {
      int64_t deltaJ = jCut.second - jCut.first;
    
      ABigMap.block(iCut.first,jCut.first,deltaI,deltaJ).noalias() =
        ASmallMap.block(i,j,deltaI,deltaJ);
    
      j += deltaJ;
    }
      i += deltaI;
    }
  };

  template <typename _F1, typename _F2>
  void submat_set(int64_t M, int64_t N, int64_t MSub, 
    int64_t NSub, _F1 *ABig, int64_t LDAB, _F2 *ASmall, 
    int64_t LDAS, 
    std::vector<std::pair<int64_t,int64_t>> &SubMatCut) {

    using mat_t1 = Eigen::Matrix<_F1,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>;
    using mat_t2 = Eigen::Matrix<_F2,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>;
    
    Eigen::Map< mat_t1 > ABigMap(ABig,LDAB,N);
    Eigen::Map< mat_t2 > ASmallMap(ASmall,LDAS,NSub);

    int64_t i(0);
    for( auto& iCut : SubMatCut ) {
      int64_t deltaI = iCut.second - iCut.first;
      int64_t j(0);
    for( auto& jCut : SubMatCut ) {
      int64_t deltaJ = jCut.second - jCut.first;
    
      ASmallMap.block(i,j,deltaI,deltaJ).noalias() =
        ABigMap.block(iCut.first,jCut.first,deltaI,deltaJ);
    
      j += deltaJ;
    }
      i += deltaI;
    }
  };


  template <typename _F1, typename _F2>
  void inc_by_submat(int64_t M, int64_t N, int64_t MSub, 
    int64_t NSub, _F1 *ABig, int64_t LDAB, _F2 *ASmall, 
    int64_t LDAS, 
    std::vector<std::pair<int64_t,int64_t>> &SubMatCut) {

    using mat_t1 = Eigen::Matrix<_F1,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>;
    using mat_t2 = Eigen::Matrix<_F2,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>;
    
    Eigen::Map< mat_t1 > ABigMap(ABig,LDAB,N);
    Eigen::Map< mat_t2 > ASmallMap(ASmall,LDAS,NSub);

    int64_t i(0);
    for( auto& iCut : SubMatCut ) {
      int64_t deltaI = iCut.second - iCut.first;
      int64_t j(0);
    for( auto& jCut : SubMatCut ) {
      int64_t deltaJ = jCut.second - jCut.first;
    
      ABigMap.block(iCut.first,jCut.first,deltaI,deltaJ).noalias() +=
        ASmallMap.block(i,j,deltaI,deltaJ);
    
      j += deltaJ;
    }
      i += deltaI;
    }
  };

  template <typename _F1, typename _F2>
  void submat_inc(int64_t M, int64_t N, int64_t MSub, 
    int64_t NSub, _F1 *ABig, int64_t LDAB, _F2 *ASmall, 
    int64_t LDAS, 
    std::vector<std::pair<int64_t,int64_t>> &SubMatCut) {

    using mat_t1 = Eigen::Matrix<_F1,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>;
    using mat_t2 = Eigen::Matrix<_F2,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>;
    
    Eigen::Map< mat_t1 > ABigMap(ABig,LDAB,N);
    Eigen::Map< mat_t2 > ASmallMap(ASmall,LDAS,NSub);

    int64_t i(0);
    for( auto& iCut : SubMatCut ) {
      int64_t deltaI = iCut.second - iCut.first;
      int64_t j(0);
    for( auto& jCut : SubMatCut ) {
      int64_t deltaJ = jCut.second - jCut.first;
    
      ASmallMap.block(i,j,deltaI,deltaJ).noalias() +=
        ABigMap.block(iCut.first,jCut.first,deltaI,deltaJ);
    
      j += deltaJ;
    }
      i += deltaI;
    }
  };
} // namespace util

} // namespace lapack
