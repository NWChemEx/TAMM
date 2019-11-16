
#ifndef TAMM_TESTS_HF_COMMON_HPP_
#define TAMM_TESTS_HF_COMMON_HPP_

#include <cctype>

// Eigen matrix algebra library
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
// #include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/MatrixFunctions>
#undef I

#include "common/input_parser.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif
#include "tamm/eigen_utils.hpp"
#include "tamm/tamm.hpp"
#include "macdecls.h"
#include "ga-mpi.h"

// #define EIGEN_DIAG 
// #ifndef SCALAPACK
  #include "common/linalg.hpp"
#ifdef SCALAPACK
  // CXXBLACS BLACS/ScaLAPACK wrapper
  // #include LAPACKE_HEADER
  // #define CXXBLACS_HAS_LAPACK
  #define CB_INT TAMM_LAPACK_INT
  #define CXXBLACS_LAPACK_Complex16 TAMM_LAPACK_COMPLEX16
  #define CXXBLACS_LAPACK_Complex8 TAMM_LAPACK_COMPLEX8
  #include <cxxblacs.hpp>
//std::unique_ptr<CXXBLACS::BlacsGrid> blacs_grid;
#endif
#undef I 

using namespace tamm;
using std::cerr;
using std::cout;
using std::endl;
using std::string;
using libint2::Atom;

using TensorType = double;
const auto max_engine_precision = std::numeric_limits<double>::epsilon() / 1e10;


using Matrix   = Eigen::Matrix<TensorType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Tensor1D = Eigen::Tensor<TensorType, 1, Eigen::RowMajor>;
using Tensor2D = Eigen::Tensor<TensorType, 2, Eigen::RowMajor>;
using Tensor3D = Eigen::Tensor<TensorType, 3, Eigen::RowMajor>;
using Tensor4D = Eigen::Tensor<TensorType, 4, Eigen::RowMajor>;

using shellpair_list_t = std::unordered_map<size_t, std::vector<size_t>>;
shellpair_list_t obs_shellpair_list;  // shellpair list for OBS
shellpair_list_t dfbs_shellpair_list;  // shellpair list for DFBS
shellpair_list_t minbs_shellpair_list;  // shellpair list for minBS
using shellpair_data_t = std::vector<std::vector<std::shared_ptr<libint2::ShellPair>>>;  // in same order as shellpair_list_t
shellpair_data_t obs_shellpair_data;  // shellpair data for OBS
shellpair_data_t dfbs_shellpair_data;  // shellpair data for DFBS
shellpair_data_t minbs_shellpair_data;  // shellpair data for minBS

int idiis  = 0;
SCFOptions scf_options;

Matrix C_occ;

//AO
tamm::TiledIndexSpace tAO, tAOt;
std::vector<tamm::Tile> AO_tiles;
std::vector<tamm::Tile> AO_opttiles;
std::vector<size_t> shell_tile_map;
tamm::TiledIndexLabel mu, nu, ku;
tamm::TiledIndexLabel mup, nup, kup;

//DF
size_t ndf;
bool is_3c_init = false;
libint2::BasisSet dfbs;
tamm::IndexSpace dfAO; 
std::vector<Tile> dfAO_tiles;
std::vector<Tile> dfAO_opttiles;
std::vector<size_t> df_shell_tile_map;
tamm::TiledIndexSpace tdfAO, tdfAOt;
tamm::TiledIndexLabel d_mu,d_nu,d_ku;
tamm::TiledIndexLabel d_mup, d_nup, d_kup;
tamm::TiledIndexSpace tdfCocc;
tamm::TiledIndexLabel dCocc_til;

tamm::Tensor<TensorType> xyK_tamm; //n,n,ndf
tamm::Tensor<TensorType> C_occ_tamm; //n,nocc


//DENSITY FITTING
struct DFFockEngine {
  const libint2::BasisSet& obs;
  const libint2::BasisSet& dfbs;
  DFFockEngine(const libint2::BasisSet& _obs, const libint2::BasisSet& _dfbs)
      : obs(_obs), dfbs(_dfbs) {}

};

Matrix compute_soad(const std::vector<libint2::Atom> &atoms);

// computes norm of shell-blocks of A
Matrix compute_shellblock_norm(const libint2::BasisSet& obs, const Matrix& A);

std::tuple<shellpair_list_t,shellpair_data_t>
compute_shellpairs(const libint2::BasisSet& bs1,
                   const libint2::BasisSet& bs2 = libint2::BasisSet(),
                   double threshold = 1e-12);

template <libint2::Operator Kernel = libint2::Operator::coulomb>
Matrix compute_schwarz_ints(
    const libint2::BasisSet& bs1, const libint2::BasisSet& bs2 = libint2::BasisSet(),
    bool use_2norm = false,  // use infty norm by default
    typename libint2::operator_traits<Kernel>::oper_params_type params =
        libint2::operator_traits<Kernel>::default_params());      

// returns {X,X^{-1},S_condition_number_after_conditioning}, where
// X is the generalized square-root-inverse such that X.transpose() * S * X = I
// columns of Xinv is the basis conditioned such that
// the condition number of its metric (Xinv.transpose . Xinv) <
// S_condition_number_threshold
std::tuple<Matrix, Matrix, double> conditioning_orthogonalizer(
   const ExecutionContext& ec,
    const Matrix& S, double S_condition_number_threshold);

template <class T> T &unconst_cast(const T &v) { return const_cast<T &>(v); }

size_t nbasis(const std::vector<libint2::Shell>& shells) {
    size_t n = 0;
    for(const auto& shell : shells) n += shell.size();
    return n;
}

size_t max_nprim(const std::vector<libint2::Shell>& shells) {
    size_t n = 0;
    for(auto shell : shells) n = std::max(shell.nprim(), n);
    return n;
}

int max_l(const std::vector<libint2::Shell>& shells) {
    int l = 0;
    for(auto shell : shells)
        for(auto c : shell.contr) l = std::max(c.l, l);
    return l;
}

std::vector<size_t> map_shell_to_basis_function(
  const std::vector<libint2::Shell>& shells) {
    std::vector<size_t> result;
    result.reserve(shells.size());

    size_t n = 0;
    for(auto shell : shells) {
        result.push_back(n);
        n += shell.size();
    }

    return result;
}

std::vector<size_t> map_basis_function_to_shell(
  const std::vector<libint2::Shell>& shells) {
    std::vector<size_t> result(nbasis(shells));

    auto shell2bf = map_shell_to_basis_function(shells);
    for(size_t s1 = 0; s1 != shells.size(); ++s1) {
        auto bf1_first = shell2bf[s1]; // first basis function in this shell
        auto n1        = shells[s1].size();
        for(size_t f1 = 0; f1 != n1; ++f1) {
            const auto bf1 = f1 + bf1_first;
            result[bf1]    = s1;
        }
    }
    return result;
}

std::string getfilename(std::string filename){
  size_t lastindex = filename.find_last_of(".");
  auto fname = filename.substr(0,lastindex);
  return fname.substr(fname.find_last_of("/")+1,fname.length());
}

void writeC(Matrix& C, std::string filename, std::string scf_files_prefix){
  std::string outputfile = scf_files_prefix + ".movecs";
  const auto N = C.rows();
  std::vector<TensorType> Cbuf(N*N);
  TensorType *Hbuf = Cbuf.data();
  Eigen::Map<Matrix>(Hbuf,N,N) = C;  
  std::ofstream out(outputfile, std::ios::out | std::ios::binary);
  if(!out) {
    cerr << "ERROR: Cannot open file " << outputfile << endl;
    return;
  }

  out.write((char *)(Hbuf), sizeof(TensorType) *N*N);
  out.close();
}

template<typename T>
std::vector<size_t> sort_indexes(std::vector<T>& v){
    std::vector<size_t> idx(v.size());
    iota(idx.begin(),idx.end(),0);
    sort(idx.begin(),idx.end(),[&v](size_t x, size_t y) {return v[x] < v[y];});

    return idx;
}

template<typename ...Args>
auto print_2e(Args&&... args){
((std::cout << args << ", "), ...);
}

// returns {X,X^{-1},rank,A_condition_number,result_A_condition_number}, where
// X is the generalized square-root-inverse such that X.transpose() * A * X = I
//
// if symmetric is true, produce "symmetric" sqrtinv: X = U . A_evals_sqrtinv .
// U.transpose()),
// else produce "canonical" sqrtinv: X = U . A_evals_sqrtinv
// where U are eigenvectors of A
// rows and cols of symmetric X are equivalent; for canonical X the rows are
// original basis (AO),
// cols are transformed basis ("orthogonal" AO)
//
// A is conditioned to max_condition_number
std::tuple<Matrix, Matrix, size_t, double, double> gensqrtinv(
    const ExecutionContext& ec, const Matrix& S, bool symmetric = false,
    double max_condition_number = 1e8) {
#ifdef SCALAPACK
  Eigen::SelfAdjointEigenSolver<Matrix> eig_solver(S);
  auto U = eig_solver.eigenvectors();
  auto s = eig_solver.eigenvalues();
  auto s_max = s.maxCoeff();
  auto condition_number = std::min(
      s_max / std::max(s.minCoeff(), std::numeric_limits<double>::min()),
      1.0 / std::numeric_limits<double>::epsilon());
  auto threshold = s_max / max_condition_number;
  long n = s.rows();
  long n_cond = 0;
  for (long i = n - 1; i >= 0; --i) {
    if (s(i) >= threshold) {
      ++n_cond;
    } else
      i = 0;  // skip rest since eigenvalues are in ascending order
  }

  auto sigma = s.bottomRows(n_cond);
  auto result_condition_number = sigma.maxCoeff() / sigma.minCoeff();
  auto sigma_sqrt = sigma.array().sqrt().matrix().asDiagonal();
  auto sigma_invsqrt = sigma.array().sqrt().inverse().matrix().asDiagonal();

  // make canonical X/Xinv
  auto U_cond = U.block(0, n - n_cond, n, n_cond);
  Matrix X = U_cond * sigma_invsqrt;
  Matrix Xinv = U_cond * sigma_sqrt;
  // convert to symmetric, if needed
  if (symmetric) {
    X = X * U_cond.transpose();
    Xinv = Xinv * U_cond.transpose();
  }
#else

  auto world = ec.pg().comm();
  int world_rank, world_size;
  MPI_Comm_rank( world, &world_rank );
  MPI_Comm_size( world, &world_size );

  int64_t n_cond;
  double condition_number, result_condition_number;
  Matrix X, Xinv;

  const int64_t N = S.rows();

  if( world_rank == 0 ) {

  // Eigendecompose S -> VsV**T
  Eigen::MatrixXd V = S;
  std::vector<double> s(N);
  linalg::lapack::syevd( 'V', 'L', N, V.data(), N, s.data() );

  condition_number = std::min(
    s.back() / std::max( s.front(), std::numeric_limits<double>::min() ),
    1.       / std::numeric_limits<double>::epsilon()
  );

  const auto threshold = s.back() / max_condition_number;
  auto first_above_thresh = std::find_if( s.begin(), s.end(), [&](const auto& x){ return x >= threshold; } );
  result_condition_number = s.back() / *first_above_thresh;

  const int64_t n_illcond = std::distance( s.begin(), first_above_thresh );
  n_cond    = N - n_illcond;

  assert( n_cond == N ); //TODO: fix

  auto* V_cond = V.data() + n_illcond * N;
  X.resize( N, n_cond ); Xinv.resize( N, n_cond );

  // Form canonical X/Xinv
  for( auto i = 0; i < n_cond; ++i ) {

    const auto srt = std::sqrt( *(first_above_thresh + i) );

    // X is row major....
    auto* X_col    = X.data()    + i;
    auto* Xinv_col = Xinv.data() + i;

    linalg::blas::copy( N, V_cond + i*N, 1, X_col,    N );
    linalg::blas::copy( N, V_cond + i*N, 1, Xinv_col, N );
    linalg::blas::scal( N, 1./srt, X_col,    N );
    linalg::blas::scal( N, srt,    Xinv_col, N );

  }  

  if( symmetric ) {

    assert( not symmetric );

/*
    // X is row major, thus we need to form X**T = V_cond * X**T
    Matrix TMP = X;
    X.resize( N, N );
    linalg::blas::gemm( 'N', 'N', N, N, n_cond, 1., V_cond, N, TMP.data(), n_cond, 0., X.data(), N );
*/

  }
  } // compute on root 


  if( world_size > 1 ) {

    // TODO: Should buffer this
    MPI_Bcast( &n_cond,                  1, MPI_INT64_T, 0, world );
    MPI_Bcast( &condition_number,        1, MPI_DOUBLE,  0, world );
    MPI_Bcast( &result_condition_number, 1, MPI_DOUBLE,  0, world );

    if( world_rank != 0 ) {
      X.resize( N, n_cond ); Xinv.resize(N, n_cond);
    }

    MPI_Bcast( X.data(),    X.size(),    MPI_DOUBLE, 0, world );
    MPI_Bcast( Xinv.data(), Xinv.size(), MPI_DOUBLE, 0, world );
  }

#endif
  return std::make_tuple(X, Xinv, size_t(n_cond), condition_number,
                         result_condition_number);
}

std::tuple<Matrix, Matrix, double> conditioning_orthogonalizer(
  const ExecutionContext& ec,  const Matrix& S, double S_condition_number_threshold) {
  size_t obs_rank;
  double S_condition_number;
  double XtX_condition_number;
  Matrix X, Xinv;

  assert(S.rows() == S.cols());

  std::tie(X, Xinv, obs_rank, S_condition_number, XtX_condition_number) =
      gensqrtinv(ec, S, false, S_condition_number_threshold);
  auto obs_nbf_omitted = (long)S.rows() - (long)obs_rank;
//   std::cout << "overlap condition number = " << S_condition_number;
  if (obs_nbf_omitted > 0){
    if(GA_Nodeid()==0) std::cout << " (dropped " << obs_nbf_omitted << " "
              << (obs_nbf_omitted > 1 ? "fns" : "fn") << " to reduce to "
              << XtX_condition_number << ")";
  }
  if(GA_Nodeid()==0) std::cout << endl;

  if (obs_nbf_omitted > 0) {
    Matrix should_be_I = X.transpose() * S * X;
    Matrix I = Matrix::Identity(should_be_I.rows(), should_be_I.cols());
    if(GA_Nodeid()==0) std::cout << "||X^t * S * X - I||_2 = " << (should_be_I - I).norm()
              << " (should be 0)" << endl;
  }

  return std::make_tuple(X, Xinv, XtX_condition_number);
}

std::tuple<shellpair_list_t,shellpair_data_t>
compute_shellpairs(const libint2::BasisSet& bs1,
                   const libint2::BasisSet& _bs2,
                   const double threshold) {

  using libint2::Engine;
  using libint2::BasisSet;
  using libint2::Operator;
  using libint2::BraKet;

  const BasisSet& bs2 = (_bs2.empty() ? bs1 : _bs2);
  const auto nsh1 = bs1.size();
  const auto nsh2 = bs2.size();
  const auto bs1_equiv_bs2 = (&bs1 == &bs2);

  // construct the 2-electron repulsion integrals engine

  Engine engine(Operator::overlap,
                       std::max(bs1.max_nprim(), bs2.max_nprim()),
                       std::max(bs1.max_l(), bs2.max_l()), 0);


  if(GA_Nodeid()==0)
    std::cout << "computing non-negligible shell-pair list ... ";
    
  libint2::Timers<1> timer;
  timer.set_now_overhead(25);
  timer.start(0);

  shellpair_list_t splist;

  // std::mutex mx;

    const auto& buf = engine.results();

    // loop over permutationally-unique set of shells
    for (size_t s1 = 0l, s12 = 0l; s1 != nsh1; ++s1) {
      // mx.lock();
      if (splist.find(s1) == splist.end())
        splist.insert(std::make_pair(s1, std::vector<size_t>()));
      // mx.unlock();

      auto n1 = bs1[s1].size();  // number of basis functions in this shell

      auto s2_max = bs1_equiv_bs2 ? s1 : nsh2 - 1;
      for (decltype(s2_max) s2 = 0; s2 <= s2_max; ++s2, ++s12) {
        // if (s12 % nthreads != thread_id) continue;

        auto on_same_center = (bs1[s1].O == bs2[s2].O);
        bool significant = on_same_center;
        if (not on_same_center) {
          auto n2 = bs2[s2].size();
          engine.compute(bs1[s1], bs2[s2]);
          Eigen::Map<const Matrix> buf_mat(buf[0], n1, n2);
          auto norm = buf_mat.norm();
          significant = (norm >= threshold);
        }

        if (significant) {
          // mx.lock();
          splist[s1].emplace_back(s2);
          // mx.unlock();
        }
      }
    }
  


  // resort shell list in increasing order, i.e. splist[s][s1] < splist[s][s2] if s1 < s2
  // N.B. only parallelized over 1 shell index
    for (size_t s1 = 0l; s1 != nsh1; ++s1) {
      // if (s1 % nthreads == thread_id) {
        auto& list = splist[s1];
        std::sort(list.begin(), list.end());
      }
    // }


  // compute shellpair data assuming that we are computing to default_epsilon
  // N.B. only parallelized over 1 shell index
  const auto ln_max_engine_precision = std::log(max_engine_precision);
  shellpair_data_t spdata(splist.size());
  
    for (size_t s1 = 0l; s1 != nsh1; ++s1) {
      // if (s1 % nthreads == thread_id) {
        for(const auto& s2 : splist[s1]) {
          spdata[s1].emplace_back(std::make_shared<libint2::ShellPair>(bs1[s1],bs2[s2],ln_max_engine_precision));
        }
      // }
    }
  
  timer.stop(0);
  if(GA_Nodeid()==0)     
    std::cout << "done (" << timer.read(0) << " s)" << endl;

  return std::make_tuple(splist,spdata);
}

// computes Superposition-Of-Atomic-Densities guess for the molecular density
// matrix
// in minimal basis; occupies subshells by smearing electrons evenly over the
// orbitals
Matrix compute_soad(const std::vector<Atom>& atoms) {
  // compute number of atomic orbitals
  size_t nao = 0;
  for (const auto& atom : atoms) {
    const auto Z = atom.atomic_number;
    nao += libint2::sto3g_num_ao(Z);
  }

  // compute the minimal basis density
  Matrix D = Matrix::Zero(nao, nao);
  size_t ao_offset = 0;  // first AO of this atom
  for (const auto& atom : atoms) {
    const auto Z = atom.atomic_number;
    const auto& occvec = libint2::sto3g_ao_occupation_vector(Z);
    for(const auto& occ: occvec) {
      D(ao_offset, ao_offset) = occ;
      ++ao_offset;
    }
  }

  return D * 0.5;  // we use densities normalized to # of electrons/2
}

template <libint2::Operator Kernel>
Matrix compute_schwarz_ints(
    const libint2::BasisSet& bs1, const libint2::BasisSet& _bs2, bool use_2norm,
    typename libint2::operator_traits<Kernel>::oper_params_type params) {

  using libint2::BasisSet;
  using libint2::Engine;
  using libint2::BraKet;

  const BasisSet& bs2 = (_bs2.empty() ? bs1 : _bs2);
  const auto nsh1 = bs1.size();
  const auto nsh2 = bs2.size();
  const auto bs1_equiv_bs2 = (&bs1 == &bs2);

  Matrix K = Matrix::Zero(nsh1, nsh2);

  // construct the 2-electron repulsion integrals engine
  // !!! very important: cannot screen primitives in Schwarz computation !!!
  auto epsilon = 0.0;
  Engine engine = Engine(Kernel, std::max(bs1.max_nprim(), bs2.max_nprim()),
                      std::max(bs1.max_l(), bs2.max_l()), 0, epsilon, params);

    if(GA_Nodeid()==0) {
      std::cout << "computing Schwarz bound prerequisites (kernel=" 
            << (int)Kernel << ") ... ";
    }

    libint2::Timers<1> timer;
    timer.set_now_overhead(25);
    timer.start(0);
  
    const auto& buf = engine.results();

    // loop over permutationally-unique set of shells
    for (size_t s1 = 0l, s12 = 0l; s1 != nsh1; ++s1) {
      auto n1 = bs1[s1].size();  // number of basis functions in this shell

      auto s2_max = bs1_equiv_bs2 ? s1 : nsh2 - 1;
      for (decltype(s1) s2 = 0; s2 <= s2_max; ++s2, ++s12) {
        // if (s12 % nthreads != thread_id) continue;

        auto n2 = bs2[s2].size();
        auto n12 = n1 * n2;

        engine.compute2<Kernel, BraKet::xx_xx, 0>(bs1[s1], bs2[s2],
                                                              bs1[s1], bs2[s2]);
        assert(buf[0] != nullptr &&
               "to compute Schwarz ints turn off primitive screening");

        // to apply Schwarz inequality to individual integrals must use the diagonal elements
        // to apply it to sets of functions (e.g. shells) use the whole shell-set of ints here
        Eigen::Map<const Matrix> buf_mat(buf[0], n12, n12);
        auto norm2 = use_2norm ? buf_mat.norm()
                               : buf_mat.lpNorm<Eigen::Infinity>();
        K(s1, s2) = std::sqrt(norm2);
        if (bs1_equiv_bs2) K(s2, s1) = K(s1, s2);
      }
    }

  timer.stop(0);
  if(GA_Nodeid()==0) 
    std::cout << "done (" << timer.read(0) << " s)" << endl;
 
  return K;
}

Matrix compute_shellblock_norm(const libint2::BasisSet& obs, const Matrix& A) {
  const auto nsh = obs.size();
  Matrix Ash = Matrix::Zero(nsh, nsh);

  auto shell2bf = obs.shell2bf();
  for (size_t s1 = 0; s1 != nsh; ++s1) {
    const auto& s1_first = shell2bf[s1];
    const auto& s1_size = obs[s1].size();
    for (size_t s2 = 0; s2 != nsh; ++s2) {
      const auto& s2_first = shell2bf[s2];
      const auto& s2_size = obs[s2].size();

      Ash(s1, s2) = A.block(s1_first, s2_first, s1_size, s2_size)
                        .lpNorm<Eigen::Infinity>();
    }
  }

  return Ash;
}

#endif // TAMM_TESTS_HF_COMMON_HPP_
