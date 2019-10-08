
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
#ifndef SCALAPACK
  #include "linalg.hpp"
#else 
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

  // typedef btas::RangeNd<CblasRowMajor, std::array<long, 3>> Range3d;
  // typedef btas::Tensor<double, Range3d> Tensor3d;
  Tensor3D xyK;

  // a DF-based builder, using coefficients of occupied MOs
  Matrix compute_2body_fock_dfC(const Matrix& Cocc);
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

template <libint2::Operator obtype, typename OperatorParams = 
typename libint2::operator_traits<obtype>::oper_params_type>
std::array<Matrix, libint2::operator_traits<obtype>::nopers> compute_1body_ints(
    const libint2::BasisSet& obs, OperatorParams oparams = OperatorParams());

// an efficient Fock builder; *integral-driven* hence computes
// permutationally-unique ints once
// Matrix compute_2body_fock_orig(const std::vector<libint2::Shell> &shells,
// const Matrix &D);

// an Fock builder that can accept densities expressed a separate basis
Matrix compute_2body_fock_general(
    const libint2::BasisSet& obs, const Matrix& D, const libint2::BasisSet& D_bs,
    bool D_is_sheldiagonal = false,  // set D_is_shelldiagonal if doing SOAD
    double precision = std::numeric_limits<
        double>::epsilon()  // discard contributions smaller than this
    );               

Matrix compute_2body_fock(
    const libint2::BasisSet& obs, const Matrix& D,
    double precision = std::numeric_limits<
        double>::epsilon(),  // discard contributions smaller than this
    const Matrix& Schwarz = Matrix()  // K_ij = sqrt(||(ij|ij)||_\infty); if
                                       // empty, do not Schwarz screen
    );               

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

void writeC(Matrix& C, std::string filename, OptionsMap options){
  if(options.scf_options.restart) return;
  std::string outputfile = getfilename(filename) +
        "." + options.scf_options.basis + ".movecs";
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


template <libint2::Operator obtype, typename OperatorParams>
std::array<Matrix, libint2::operator_traits<obtype>::nopers> compute_1body_ints(
    const libint2::BasisSet& obs, OperatorParams oparams) {
  const auto n = obs.nbf();
  const auto nshells = obs.size();
  typedef std::array<Matrix, libint2::operator_traits<obtype>::nopers>
      result_type;
  const unsigned int nopers = libint2::operator_traits<obtype>::nopers;
  result_type result;
  for (auto& r : result) r = Matrix::Zero(n, n);

  // construct the 1-body integrals engine
  libint2::Engine engine = libint2::Engine(obtype, obs.max_nprim(), obs.max_l(), 0);
  // pass operator params to the engine, e.g.
  // nuclear attraction ints engine needs to know where the charges sit ...
  // the nuclei are charges in this case; in QM/MM there will also be classical
  // charges
  engine.set_params(oparams);

  auto shell2bf = obs.shell2bf();

    const auto& buf = engine.results();

    // loop over unique shell pairs, {s1,s2} such that s1 >= s2
    // this is due to the permutational symmetry of the real integrals over
    // Hermitian operators: (1|2) = (2|1)
    for (auto s1 = 0l/*, s12 = 0l*/; s1 != nshells; ++s1) {
      auto bf1 = shell2bf[s1];  // first basis function in this shell
      auto n1 = obs[s1].size();

      // auto s1_offset = s1 * (s1+1) / 2;
      for (auto s2: obs_shellpair_list[s1]) {
        // auto s12 = s1_offset + s2;
        ////if (s12 % nthreads != thread_id) continue;

        auto bf2 = shell2bf[s2];
        auto n2 = obs[s2].size();

        // auto n12 = n1 * n2;

        // compute shell pair; return is the pointer to the buffer
        engine.compute(obs[s1], obs[s2]);

        for (unsigned int op = 0; op != nopers; ++op) {
          // "map" buffer to a const Eigen Matrix, and copy it to the
          // corresponding blocks of the result
          Eigen::Map<const Matrix> buf_mat(buf[op], n1, n2);
          result[op].block(bf1, bf2, n1, n2) = buf_mat;
          if (s1 != s2)  // if s1 >= s2, copy {s1,s2} to the corresponding
                         // {s2,s1} block, note the transpose!
            result[op].block(bf2, bf1, n2, n1) = buf_mat.transpose();
        }
      }
    }

  return result;
}


Matrix compute_2body_fock(const libint2::BasisSet& obs, const Matrix& D,
                          double precision, const Matrix& Schwarz) {

  using libint2::Operator;                            
  const auto n = obs.nbf();
  const auto nshells = obs.size();
  Matrix G = Matrix::Zero(n, n);

  const auto do_schwarz_screen = Schwarz.cols() != 0 && Schwarz.rows() != 0;
  Matrix D_shblk_norm =
      compute_shellblock_norm(obs, D);  // matrix of infty-norms of shell blocks

  auto fock_precision = precision;
  // engine precision controls primitive truncation, assume worst-case scenario
  // (all primitive combinations add up constructively)
  auto max_nprim = obs.max_nprim();
  auto max_nprim4 = max_nprim * max_nprim * max_nprim * max_nprim;
  auto engine_precision = std::min(fock_precision / D_shblk_norm.maxCoeff(),
                                   std::numeric_limits<double>::epsilon()) /
                          max_nprim4;
  assert(engine_precision > max_engine_precision &&
      "using precomputed shell pair data limits the max engine precision"
  " ... make max_engine_precision smaller and recompile");

  // construct the 2-electron repulsion integrals engine pool
  using libint2::Engine;
  Engine engine(Operator::coulomb, obs.max_nprim(), obs.max_l(), 0);
  engine.set_precision(engine_precision);  // shellset-dependent precision
                                               // control will likely break
                                               // positive definiteness
                                               // stick with this simple recipe
  // std::cout << "compute_2body_fock:precision = " << precision << endl;
  // std::cout << "Engine::precision = " << engine.precision() << endl;
  // for (size_t i = 1; i != nthreads; ++i) {
  //   engines[i] = engines[0];
  // }
  std::atomic<size_t> num_ints_computed{0};


  auto shell2bf = obs.shell2bf();

    const auto& buf = engine.results();

    // loop over permutationally-unique set of shells
    for (size_t s1 = 0l/*, s1234 = 0l*/; s1 != nshells; ++s1) {
      auto bf1_first = shell2bf[s1];  // first basis function in this shell
      auto n1 = obs[s1].size();       // number of basis functions in this shell

      auto sp12_iter = obs_shellpair_data.at(s1).begin();

      for (const auto& s2 : obs_shellpair_list[s1]) {
        auto bf2_first = shell2bf[s2];
        auto n2 = obs[s2].size();

        const auto* sp12 = sp12_iter->get();
        ++sp12_iter;

        const auto Dnorm12 = do_schwarz_screen ? D_shblk_norm(s1, s2) : 0.;

        for (decltype(s1) s3 = 0; s3 <= s1; ++s3) {
          auto bf3_first = shell2bf[s3];
          auto n3 = obs[s3].size();

          const auto Dnorm123 =
              do_schwarz_screen
                  ? std::max(D_shblk_norm(s1, s3),
                             std::max(D_shblk_norm(s2, s3), Dnorm12))
                  : 0.;

          auto sp34_iter = obs_shellpair_data.at(s3).begin();

          const auto s4_max = (s1 == s3) ? s2 : s3;
          for (const auto& s4 : obs_shellpair_list[s3]) {
            if (s4 > s4_max)
              break;  // for each s3, s4 are stored in monotonically increasing
                      // order

            // must update the iter even if going to skip s4
            const auto* sp34 = sp34_iter->get();
            ++sp34_iter;

            // if ((s1234++) % nthreads != thread_id) continue;

            const auto Dnorm1234 =
                do_schwarz_screen
                    ? std::max(
                          D_shblk_norm(s1, s4),
                          std::max(D_shblk_norm(s2, s4),
                                   std::max(D_shblk_norm(s3, s4), Dnorm123)))
                    : 0.;

            if (do_schwarz_screen &&
                Dnorm1234 * Schwarz(s1, s2) * Schwarz(s3, s4) <
                    fock_precision)
              continue;

            auto bf4_first = shell2bf[s4];
            auto n4 = obs[s4].size();

            num_ints_computed += n1 * n2 * n3 * n4;

            // compute the permutational degeneracy (i.e. # of equivalents) of
            // the given shell set
            auto s12_deg = (s1 == s2) ? 1 : 2;
            auto s34_deg = (s3 == s4) ? 1 : 2;
            auto s12_34_deg = (s1 == s3) ? (s2 == s4 ? 1 : 2) : 2;
            auto s1234_deg = s12_deg * s34_deg * s12_34_deg;

            engine.compute2<Operator::coulomb, libint2::BraKet::xx_xx, 0>(
                obs[s1], obs[s2], obs[s3], obs[s4], sp12, sp34);
            const auto* buf_1234 = buf[0];
            if (buf_1234 == nullptr)
              continue; // if all integrals screened out, skip to next quartet

            // 1) each shell set of integrals contributes up to 6 shell sets of
            // the Fock matrix:
            //    F(a,b) += (ab|cd) * D(c,d)
            //    F(c,d) += (ab|cd) * D(a,b)
            //    F(b,d) -= 1/4 * (ab|cd) * D(a,c)
            //    F(b,c) -= 1/4 * (ab|cd) * D(a,d)
            //    F(a,c) -= 1/4 * (ab|cd) * D(b,d)
            //    F(a,d) -= 1/4 * (ab|cd) * D(b,c)
            // 2) each permutationally-unique integral (shell set) must be
            // scaled by its degeneracy,
            //    i.e. the number of the integrals/sets equivalent to it
            // 3) the end result must be symmetrized
            for (decltype(n1) f1 = 0, f1234 = 0; f1 != n1; ++f1) {
              const auto bf1 = f1 + bf1_first;
              for (decltype(n2) f2 = 0; f2 != n2; ++f2) {
                const auto bf2 = f2 + bf2_first;
                for (decltype(n3) f3 = 0; f3 != n3; ++f3) {
                  const auto bf3 = f3 + bf3_first;
                  for (decltype(n4) f4 = 0; f4 != n4; ++f4, ++f1234) {
                    const auto bf4 = f4 + bf4_first;

                    const auto value = buf_1234[f1234];

                    const auto value_scal_by_deg = value * s1234_deg;

                    G(bf1, bf2) += D(bf3, bf4) * value_scal_by_deg;
                    G(bf3, bf4) += D(bf1, bf2) * value_scal_by_deg;
                    G(bf1, bf3) -= 0.25 * D(bf2, bf4) * value_scal_by_deg;
                    G(bf2, bf4) -= 0.25 * D(bf1, bf3) * value_scal_by_deg;
                    G(bf1, bf4) -= 0.25 * D(bf2, bf3) * value_scal_by_deg;
                    G(bf2, bf3) -= 0.25 * D(bf1, bf4) * value_scal_by_deg;
                  }
                }
              }
            }
          }
        }
      }
    }

  // };  // end of lambda

  // std::cout << "# of integrals = " << num_ints_computed << endl;
  // symmetrize the result and return
   Matrix GG = 0.5 * (G + G.transpose());

  return GG;
}

Matrix compute_2body_fock_general(const libint2::BasisSet& obs, const Matrix& D,
                                  const libint2::BasisSet& D_bs, bool D_is_shelldiagonal,
                                  double precision) {
  const auto n = obs.nbf();
  const auto nshells = obs.size();
  const auto n_D = D_bs.nbf();
  EXPECTS(D.cols() == D.rows() && D.cols() == n_D);

  Matrix G = Matrix::Zero(n, n);

  // construct the 2-electron repulsion integrals engine
  using libint2::Operator;
  using libint2::BraKet;
  using libint2::Engine;

  Engine engine(libint2::Operator::coulomb,
                      std::max(obs.max_nprim(), D_bs.max_nprim()),
                      std::max(obs.max_l(), D_bs.max_l()), 0);
  engine.set_precision(precision);  // shellset-dependent precision control
                                        // will likely break positive
                                        // definiteness
                                        // stick with this simple recipe

  auto shell2bf = obs.shell2bf();
  auto shell2bf_D = D_bs.shell2bf();

    const auto& buf = engine.results();

    // loop over permutationally-unique set of shells
    for (size_t s1 = 0l, s1234 = 0l; s1 != nshells; ++s1) {
      auto bf1_first = shell2bf[s1];  // first basis function in this shell
      auto n1 = obs[s1].size();       // number of basis functions in this shell

      for (decltype(s1) s2 = 0; s2 <= s1; ++s2) {
        auto bf2_first = shell2bf[s2];
        auto n2 = obs[s2].size();

        for (decltype(s1) s3 = 0; s3 < D_bs.size(); ++s3) {
          auto bf3_first = shell2bf_D[s3];
          auto n3 = D_bs[s3].size();

          auto s4_begin = D_is_shelldiagonal ? s3 : 0;
          auto s4_fence = D_is_shelldiagonal ? s3 + 1 : D_bs.size();

          for (decltype(s4_fence) s4 = s4_begin; s4 != s4_fence; ++s4, ++s1234) {
            // if (s1234 % nthreads != thread_id) continue;

            auto bf4_first = shell2bf_D[s4];
            auto n4 = D_bs[s4].size();

            // compute the permutational degeneracy (i.e. # of equivalents) of
            // the given shell set
            auto s12_deg = (s1 == s2) ? 1.0 : 2.0;

            if (s3 >= s4) {
              auto s34_deg = (s3 == s4) ? 1.0 : 2.0;
              auto s1234_deg = s12_deg * s34_deg;
              // auto s1234_deg = s12_deg;
              engine.compute2<Operator::coulomb, BraKet::xx_xx, 0>(
                  obs[s1], obs[s2], D_bs[s3], D_bs[s4]);
              const auto* buf_1234 = buf[0];
              if (buf_1234 != nullptr) {
                for (decltype(n1) f1 = 0, f1234 = 0; f1 != n1; ++f1) {
                  const auto bf1 = f1 + bf1_first;
                  for (decltype(n2) f2 = 0; f2 != n2; ++f2) {
                    const auto bf2 = f2 + bf2_first;
                    for (decltype(n3) f3 = 0; f3 != n3; ++f3) {
                      const auto bf3 = f3 + bf3_first;
                      for (decltype(n4) f4 = 0; f4 != n4; ++f4, ++f1234) {
                        const auto bf4 = f4 + bf4_first;

                        const auto value = buf_1234[f1234];
                        const auto value_scal_by_deg = value * s1234_deg;
                        G(bf1, bf2) += 2.0 * D(bf3, bf4) * value_scal_by_deg;
                      }
                    }
                  }
                }
              }
            }

            engine.compute2<Operator::coulomb, BraKet::xx_xx, 0>(
                obs[s1], D_bs[s3], obs[s2], D_bs[s4]);
            const auto* buf_1324 = buf[0];
            if (buf_1324 == nullptr)
              continue; // if all integrals screened out, skip to next quartet

            for (decltype(n1) f1 = 0, f1324 = 0; f1 != n1; ++f1) {
              const auto bf1 = f1 + bf1_first;
              for (decltype(n3) f3 = 0; f3 != n3; ++f3) {
                const auto bf3 = f3 + bf3_first;
                for (decltype(n2) f2 = 0; f2 != n2; ++f2) {
                  const auto bf2 = f2 + bf2_first;
                  for (decltype(n4) f4 = 0; f4 != n4; ++f4, ++f1324) {
                    const auto bf4 = f4 + bf4_first;

                    const auto value = buf_1324[f1324];
                    const auto value_scal_by_deg = value * s12_deg;
                    G(bf1, bf2) -= D(bf3, bf4) * value_scal_by_deg;
                  }
                }
              }
            }
          }
        }
      }
    }

  // symmetrize the result and return
  return 0.5 * (G + G.transpose());
}

Matrix compute_2body_2index_ints(const libint2::BasisSet& bs) {
  const auto n = bs.nbf();
  const auto nshells = bs.size();
  Matrix result = Matrix::Zero(n, n);

  // build engines for each thread
  using libint2::Engine;
  using libint2::BraKet;
  
  auto engine =
      Engine(libint2::Operator::coulomb, bs.max_nprim(), bs.max_l(), 0);
  engine.set(BraKet::xs_xs);
  
  auto shell2bf = bs.shell2bf();
  auto unitshell = libint2::Shell::unit();

    const auto& buf = engine.results();

    // loop over unique shell pairs, {s1,s2} such that s1 >= s2
    // this is due to the permutational symmetry of the real integrals over
    // Hermitian operators: (1|2) = (2|1)
    for (size_t s1 = 0l, s12 = 0l; s1 != nshells; ++s1) {
      auto bf1 = shell2bf[s1];  // first basis function in this shell
      auto n1 = bs[s1].size();

      for (decltype(s1) s2 = 0; s2 <= s1; ++s2, ++s12) {
        // if (s12 % nthreads != thread_id) continue;

        auto bf2 = shell2bf[s2];
        auto n2 = bs[s2].size();

        // compute shell pair; return is the pointer to the buffer
        engine.compute(bs[s1], bs[s2]);
        if (buf[0] == nullptr)
          continue; // if all integrals screened out, skip to next shell set

        // "map" buffer to a const Eigen Matrix, and copy it to the
        // corresponding blocks of the result
        Eigen::Map<const Matrix> buf_mat(buf[0], n1, n2);
        result.block(bf1, bf2, n1, n2) = buf_mat;
        if (s1 != s2)  // if s1 >= s2, copy {s1,s2} to the corresponding {s2,s1}
                       // block, note the transpose!
          result.block(bf2, bf1, n2, n1) = buf_mat.transpose();
      }
    }
 

  return result;
}

Matrix DFFockEngine::compute_2body_fock_dfC(const Matrix& Cocc) {

  using libint2::Operator;
  using libint2::BraKet;
  using libint2::Engine;

  const auto n = obs.nbf();
  const auto ndf = dfbs.nbf();

      using idx_pair = std::pair<long int, long int>;
    idx_pair idx_20({2,0}),idx_00({0, 0}),idx_11({1,1}),
            idx_22({2,2}),idx_10({1,0}); //,idx_21({2,1});
    std::array<idx_pair,1> aidx_00({idx_00}),aidx_10({idx_10}),aidx_20({idx_20});
    std::array<idx_pair,2> idx_1122({idx_11,idx_22}), //idx_0022({idx_00,idx_22}),
                           idx_0011({idx_00,idx_11}); //idx_1021({idx_10,idx_21})

  // using first time? compute 3-center ints and transform to inv sqrt
  // representation
  // if (xyK.size() == 0) {

    const auto nshells = obs.size();
    const auto nshells_df = dfbs.size();
    const auto& unitshell = libint2::Shell::unit();

    // construct the 2-electron 3-center repulsion integrals engine
    // since the code assumes (xx|xs) braket, and Engine/libint only produces
    // (xs|xx), use 4-center engine
    
    auto engine = libint2::Engine(libint2::Operator::coulomb,
                                 std::max(obs.max_nprim(), dfbs.max_nprim()),
                                 std::max(obs.max_l(), dfbs.max_l()), 0);
    engine.set(libint2::BraKet::xs_xx);

    auto shell2bf = obs.shell2bf();
    auto shell2bf_df = dfbs.shell2bf();

    Tensor3D Zxy(ndf, n, n);
    Zxy.setZero();

      const auto& results = engine.results();

      // loop over permutationally-unique set of shells
      for (size_t s1 = 0l, s123 = 0l; s1 != nshells_df; ++s1) {
        auto bf1_first = shell2bf_df[s1];  // first basis function in this shell
        auto n1 = dfbs[s1].size();  // number of basis functions in this shell

        for (decltype(s1) s2 = 0; s2 != nshells; ++s2) {
          auto bf2_first = shell2bf[s2];
          auto n2 = obs[s2].size();
          // const auto n12 = n1 * n2;

          for (decltype(s1) s3 = 0; s3 != nshells; ++s3, ++s123) {
            // if (s123 % nthreads != thread_id) continue;

            auto bf3_first = shell2bf[s3];
            auto n3 = obs[s3].size();
            // const auto n123 = n12 * n3;

            engine.compute2<Operator::coulomb, BraKet::xs_xx, 0>(
                dfbs[s1], unitshell, obs[s2], obs[s3]);
            const auto* buf = results[0];
            if (buf == nullptr)
              continue;

          
            // auto lower_bound = {bf1_first, bf2_first, bf3_first};
            // auto upper_bound = {bf1_first + n1, bf2_first + n2, bf3_first + n3};
            // auto view = btas::make_view(
            //     Zxy.range().slice(lower_bound, upper_bound), Zxy.storage());
            // std::copy(buf, buf + n123, view.begin());

            for (decltype(n1) f1 = 0, f123 = 0; f1 != n1; ++f1) {
                const auto bf1 = f1 + bf1_first;
              for (decltype(n2) f2 = 0; f2 != n2; ++f2) {
                const auto bf2 = f2 + bf2_first;
                for (decltype(n3) f3 = 0; f3 != n3; ++f3,++f123) {
                  const auto bf3 = f3 + bf3_first;
                    
                    Zxy(bf1, bf2,bf3) = buf[f123]; 
                  }
                }
              }
            

          }  // s3
        }    // s2
      }      // s1

    Matrix V = compute_2body_2index_ints(dfbs);
    Eigen::LLT<Matrix> V_LLt(V);
    Matrix I = Matrix::Identity(ndf, ndf);
    auto L = V_LLt.matrixL();
    Matrix V_L = L;
    Matrix Linv_t = L.solve(I).transpose();
    // check
    //  std::cout << "||V - L L^t|| = " << (V - V_L * V_L.transpose()).norm() <<
    //  endl;
    //  std::cout << "||I - L L^-1|| = " << (I - V_L *
    //  Linv_t.transpose()).norm() << endl;
    //  std::cout << "||V^-1 - L^-1^t L^-1|| = " << (V.inverse() - Linv_t *
    //  Linv_t.transpose()).norm() << endl;

    Tensor2D K(ndf, ndf);
    K.setZero();
    // std::copy(Linv_t.data(), Linv_t.data() + ndf * ndf, K.begin());
    for (auto i = 0; i<ndf;i++)
    for(auto j=0;j<ndf;j++)
    K(i,j) = Linv_t(i,j);


    //contract(1.0, Zxy, {1, 2, 3}, K, {1, 4}, 0.0, xyK, {2, 3, 4});
    //xyK = Tensor3D(n, n, ndf);
    xyK = Zxy.contract(K,aidx_00); 
    Zxy.resize(0, 0, 0);  // release memory

  //}  // if (xyK.size() == 0)

  const auto nocc = Cocc.cols();
  Tensor2D Co(n, nocc);
  Co.setZero();
  // std::copy(Cocc.data(), Cocc.data() + n * nocc, Co.begin());
  for (auto i = 0; i<n;i++)
    for(auto j=0;j<nocc;j++)
    Co(i,j) = Cocc(i,j);

  // Tensor3D xiK(n, nocc, ndf);
  // contract(1.0, xyK, {1, 2, 3}, Co, {2, 4}, 0.0, xiK, {1, 4, 3});
  Tensor3D xiK_p = xyK.contract(Co, aidx_10); 
  //n,ndf,nocc
  std::array<long int, 3> idx_shuffle({0,2,1});
  Tensor3D xiK = xiK_p.shuffle(idx_shuffle);

  // compute Coulomb
  Tensor1D Jtmp(ndf);
  Jtmp.setZero(); 
  //contract(1.0, xiK, {1, 2, 3}, Co, {1, 2}, 0.0, Jtmp, {3});
  Jtmp = xiK.contract(Co,idx_0011); 

  //contract(1.0, xiK, {1, 2, 3}, xiK, {4, 2, 3}, 0.0, G, {1, 4});
  Tensor2D K_ret = xiK.contract(xiK,idx_1122); 
  xiK.resize(0, 0, 0);

  //contract(2.0, xyK, {1, 2, 3}, Jtmp, {3}, -1.0, G, {1, 2});
  Tensor2D J_ret = xyK.contract(Jtmp,aidx_20);

  Tensor2D G = 2.0*J_ret - K_ret;

  // copy result to an Eigen::Matrix
  Matrix result(n, n);
  result.setZero();
  // std::copy(G.cbegin(), G.cend(), result.data());
  for (auto i = 0; i<n;i++)
    for(auto j=0;j<n;j++)
    result(i,j) = G(i,j);

  return result;
}

#endif // TAMM_TESTS_HF_COMMON_HPP_
