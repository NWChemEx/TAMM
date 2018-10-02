#ifndef GETSHIFTS_H
#define GETSHIFTS_H

#include "utilities.hpp"

VectorXd HSLanczos(const MatrixXd &H, const MatrixXd &S, const VectorXd &v0, MatrixXd &T, MatrixXd &V);
VectorXd getevbnd(const VectorXd &d, const MatrixXd &S, const double &beta);

  struct DensityOfStates {

    std::vector< double > domain;
    std::vector< double > eval;

    std::vector< double > prefactor;
    std::vector< double > center;
    std::vector< double > exp;

  };

  inline std::vector<double> linspace(double begin, double end, int64_t N) {

      double delta = (end - begin) / (N - 1);

      std::vector< double > x( N ); x[0] = begin;

      for( int64_t i = 1; i < N; i++ ) x[i] = x[i-1] + delta;

      return x;

    };


  inline std::vector< double >
  DeferredDOSEval(
    const DensityOfStates       &DOS,
    const std::vector< double > &x
  ) {

    std::vector< double > y( x.size(), 0 );

    for( size_t iX = 0; iX < x.size();       ++iX )
    for( size_t iG = 0; iG < DOS.exp.size(); ++iG ) {

      y[iX] += DOS.prefactor[iG] *
               std::exp( -DOS.exp[iG] *
                          (x[iX] - DOS.center[iG]) *
                          (x[iX] - DOS.center[iG])
                       );

    }
    return y;
  }


  inline std::vector< double >
  DeferredDOSIntegralEval(
    const DensityOfStates       &DOS,
    const std::vector< double > &x
  ) {

    std::vector< double > y( x.size(), 0 );

    for( size_t iX = 0; iX < x.size();       ++iX )
    for( size_t iG = 0; iG < DOS.exp.size(); ++iG ) {

      double sqrt_fact = std::sqrt(DOS.exp[iG]);
      double shift_cen = x[iX] - DOS.center[iG];

      //y[iX] += DOS.prefactor[iG] / sqrt_fact * (
      //           std::erf( sqrt_fact * shift_cen ) -
      //           std::erf( sqrt_fact * DOS.center[iG] )
      //         );
      y[iX] += DOS.prefactor[iG] / sqrt_fact * (
                 std::erf( sqrt_fact * shift_cen )  + 1
               );

    }

    for( size_t iX = 0; iX < x.size(); ++iX )
       y.at(iX) = y.at(iX)*sqrt(M_PI) / 2.;

    return y;
  }

  template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
  }


  template <typename T, class F>
  T Bisection( const F& f, T a, T b, double tol = 1e-8,
               int64_t maxIter = 128 ) {


    T f_a = f( a );

    T c;
    int64_t it = 0;
    for( int64_t iter = 0; iter < maxIter; ++iter ) {

      ++it;
      c = (a + b) / T(2.);

      T f_c = f(c);

      if( std::abs(f_c)    < tol or
          ( (b - a) / T(2.) ) < tol )
        break;

      if( sgn( f_c ) == sgn( f_a ) ) {

        a   = c;
        f_a = f_c;

      } else b = c;

    }

    // std::cout << it << std::endl;

    return c;

  }

/*
inline double GaussianRandom( double mean, double sigma ) {

//  int mpi_rank; MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank );

  std::default_random_engine generator;
  std::normal_distribution<double> distribution(mean,sigma);

//  generator.seed( 50 * mpi_rank );
  // FIX ME: use a fixed see for now
  generator.seed( 0 );

  return distribution(generator);

};
*/

inline void GaussianRandom( double mean, double sigma,
                            VectorXd &vec ) {

  std::default_random_engine generator;
  std::normal_distribution<double> distribution(mean,sigma);

  generator.seed( 0 );

  for(int i=0; i<vec.size(); i++)
    vec(i) = distribution(generator);

};

VectorXd getshifts(MPI_Comm comm, MatrixXd &H, MatrixXd &S, int nev, int nshifts, int *maxcnt);


#endif
