
#ifndef TAMM_METHODS_HF_COMMON_HPP_
#define TAMM_METHODS_HF_COMMON_HPP_

#include <cctype>

#include "tamm/tamm.hpp"
#include "macdecls.h"
#include "ga-mpi.h"

#include "tamm/eigen_utils.hpp"

#include "common/misc.hpp"
#include "common/molden.hpp"
#include "common/linalg.hpp"
#include "common/json_data.hpp"

using namespace tamm;
using std::cerr;
using std::cout;
using std::endl;
using std::string;
using libint2::Atom;

using TensorType = double;
const auto max_engine_precision = std::numeric_limits<double>::epsilon() / 1e10;

using shellpair_list_t = std::unordered_map<size_t, std::vector<size_t>>;
shellpair_list_t obs_shellpair_list;  // shellpair list for OBS
shellpair_list_t dfbs_shellpair_list;  // shellpair list for DFBS
shellpair_list_t minbs_shellpair_list;  // shellpair list for minBS
shellpair_list_t obs_shellpair_list_atom;  // shellpair list for OBS for specfied atom
shellpair_list_t minbs_shellpair_list_atom;  // shellpair list for minBS for specfied atom
using shellpair_data_t = std::vector<std::vector<std::shared_ptr<libint2::ShellPair>>>;  // in same order as shellpair_list_t
shellpair_data_t obs_shellpair_data;  // shellpair data for OBS
shellpair_data_t dfbs_shellpair_data;  // shellpair data for DFBS
shellpair_data_t minbs_shellpair_data;  // shellpair data for minBS
shellpair_data_t obs_shellpair_data_atom;  // shellpair data for OBS for specfied atom
shellpair_data_t minbs_shellpair_data_atom;  // shellpair data for minBS for specfied atom

int idiis  = 0;
int iediis  = 0;
bool switch_diis=false;

//AO
int final_AO_tilesize;
tamm::TiledIndexSpace tAO, tAOt; //tAO_ld
std::vector<tamm::Tile> AO_tiles;
std::vector<tamm::Tile> AO_opttiles;
std::vector<size_t> shell_tile_map;
// tamm::TiledIndexLabel mu_ld, nu_ld;
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

struct EigenTensors {
  // Matrix H,S;
  Matrix F,C,C_occ,X; //only rank 0 allocates F{a,b}, C_occ, C{a,b}, X{a,b}
  Matrix F_beta,C_beta,X_beta;
  Matrix G,D;
  Matrix G_beta,D_beta;
};

struct TAMMTensors {
    std::vector<Tensor<TensorType>> ehf_tamm_hist;
    
    std::vector<Tensor<TensorType>> diis_hist;
    std::vector<Tensor<TensorType>> fock_hist;
    std::vector<Tensor<TensorType>> D_hist;

    std::vector<Tensor<TensorType>> diis_beta_hist;
    std::vector<Tensor<TensorType>> fock_beta_hist;
    std::vector<Tensor<TensorType>> D_beta_hist;
    
    Tensor<TensorType> ehf_tamm;
    Tensor<TensorType> ehf_tmp;
    Tensor<TensorType> ehf_beta_tmp;
    
    Tensor<TensorType> H1;
    Tensor<TensorType> S1;
    Tensor<TensorType> T1;
    Tensor<TensorType> V1;
    Tensor<TensorType> F1;
    Tensor<TensorType> F1_beta;
    Tensor<TensorType> F1tmp1;
    Tensor<TensorType> F1tmp1_beta;
    Tensor<TensorType> F1tmp; //not allocated {tAOt, tAOt}

    Tensor<TensorType> D_tamm;
    Tensor<TensorType> D_beta_tamm;
    Tensor<TensorType> D_diff;
    Tensor<TensorType> D_last_tamm;
    Tensor<TensorType> D_last_beta_tamm;
    
    Tensor<TensorType> FD_tamm;
    Tensor<TensorType> FDS_tamm;
    Tensor<TensorType> FD_beta_tamm;
    Tensor<TensorType> FDS_beta_tamm;    

    //DF
    Tensor<TensorType> xyK_tamm; //n,n,ndf
    Tensor<TensorType> C_occ_tamm; //n,nocc
    Tensor<TensorType> Zxy_tamm; //ndf,n,n  
};

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
                   double threshold = 1e-16);

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
   const ExecutionContext& ec, SystemData& sys_data, const Matrix& S);

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

BasisSetMap construct_basisset_maps(std::vector<libint2::Atom>& atoms, libint2::BasisSet& shells) {

    BasisSetMap bsm;

    auto a2s_map = shells.atom2shell(atoms);
    size_t natoms = atoms.size();
    size_t nshells = shells.size();
    auto nbf = nbasis(shells);

    std::vector<long> shell2atom_map = shells.shell2atom(atoms);
    auto bf2shell = map_basis_function_to_shell(shells);
    auto shell2bf = map_shell_to_basis_function(shells);

    std::vector<AtomInfo> atominfo(natoms);
    std::vector<size_t> bf2atom(nbf);
    std::vector<size_t> nbf_atom(natoms);
    std::vector<size_t> nshells_atom(natoms);
    std::vector<size_t> first_bf_atom(natoms);
    std::vector<size_t> first_bf_shell(nshells);
    std::vector<size_t> first_shell_atom(natoms);

    for(size_t s1 = 0; s1 != nshells; ++s1) first_bf_shell[s1] = shells[s1].size();

    for (size_t ai = 0; ai < natoms; ai++) {
      auto nshells_ai = a2s_map[ai].size();
      auto first = a2s_map[ai][0];
      auto last = a2s_map[ai][nshells_ai - 1];
      std::vector<libint2::Shell> atom_shells(nshells_ai);
      int as_index = 0;
      size_t atom_nbf = 0;
      first_shell_atom[ai] = first;
      for (auto si = first; si <= last; si++) {
        atom_shells[as_index] = shells[si];
        as_index++;
        atom_nbf += shells[si].size();
      }
      atominfo[ai].atomic_number = atoms[ai].atomic_number;
      atominfo[ai].shells = atom_shells;
      atominfo[ai].nbf = atom_nbf;
      atominfo[ai].nbf_lo = 0;
      atominfo[ai].nbf_hi = atom_nbf;
      if (ai > 0) {
        atominfo[ai].nbf_lo = atominfo[ai - 1].nbf_hi;
        atominfo[ai].nbf_hi = atominfo[ai].nbf_lo + atom_nbf;
      }
      
      nbf_atom[ai] = atom_nbf;
      nshells_atom[ai] = nshells_ai;
      first_bf_atom[ai] = atominfo[ai].nbf_lo;
      for(auto nlo = atominfo[ai].nbf_lo; nlo<atominfo[ai].nbf_hi; nlo++) bf2atom[nlo] = ai;
    }

    bsm.nbf = nbf;
    bsm.natoms = natoms;
    bsm.nshells = nshells;
    bsm.atominfo = atominfo;
    bsm.bf2shell = bf2shell;
    bsm.shell2bf = shell2bf;
    bsm.bf2atom = bf2atom;
    bsm.nbf_atom = nbf_atom;
    bsm.atom2shell = a2s_map;
    bsm.nshells_atom = nshells_atom;
    bsm.first_bf_atom = first_bf_atom;
    bsm.first_bf_shell = first_bf_shell;
    bsm.shell2atom = shell2atom_map;
    bsm.first_shell_atom = first_shell_atom;

    return bsm;

}

template<typename T>
Matrix read_scf_mat(std::string matfile) {

  std::string mname = fs::path(matfile).extension();
  mname.erase(0, 1); //remove "."
  
  auto mfile_id = H5Fopen(matfile.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

  // Read attributes - reduced dims
  std::vector<int64_t> rdims(2);
  auto attr_dataset = H5Dopen(mfile_id, "rdims",  H5P_DEFAULT);
  H5Dread(attr_dataset, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, rdims.data());

  Matrix mat = Matrix::Zero(rdims[0],rdims[1]);
  auto mdataset_id = H5Dopen(mfile_id, mname.c_str(),  H5P_DEFAULT);

   /* Read the datasets. */
  H5Dread(mdataset_id, get_hdf5_dt<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, mat.data());

  H5Dclose(attr_dataset);
  H5Dclose(mdataset_id);
  H5Fclose(mfile_id);

  return mat;
}

template<typename T>
void write_scf_mat(Matrix& C, std::string matfile){
  std::string mname = fs::path(matfile).extension();
  mname.erase(0, 1); //remove "."

  const auto N = C.rows();
  const auto Northo = C.cols();
  TensorType *buf = C.data();

  /* Create a file. */
  hid_t file_id = H5Fcreate(matfile.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  hsize_t tsize = N*Northo;
  hid_t dataspace_id = H5Screate_simple(1, &tsize, NULL);

  /* Create dataset. */
  hid_t dataset_id = H5Dcreate(file_id, mname.c_str(), get_hdf5_dt<T>(), dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  /* Write the dataset. */
  /* herr_t status = */ H5Dwrite(dataset_id, get_hdf5_dt<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);   

  /* Create and write attribute information - dims */
  std::vector<int64_t> rdims{N,Northo};
  hsize_t attr_size = rdims.size();
  auto attr_dataspace = H5Screate_simple(1, &attr_size, NULL);
  auto attr_dataset = H5Dcreate(file_id, "rdims", H5T_NATIVE_INT64, attr_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(attr_dataset, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, rdims.data());
  H5Dclose(attr_dataset);
  H5Sclose(attr_dataspace);

  H5Dclose(dataset_id);
  H5Sclose(dataspace_id);
  H5Fclose(file_id);
}

template<typename T>
std::vector<size_t> sort_indexes(std::vector<T>& v){
    std::vector<size_t> idx(v.size());
    iota(idx.begin(),idx.end(),0);
    sort(idx.begin(),idx.end(),[&v](size_t x, size_t y) {return v[x] < v[y];});

    return idx;
}

template<typename T, int ndim>
void t2e_hf_helper(const ExecutionContext& ec, tamm::Tensor<T>& ttensor,Matrix& etensor,
                   const std::string& ustr = "") {

    const string pstr = "(" + ustr + ")";                     

    // auto hf_t1 = std::chrono::high_resolution_clock::now();

    const auto rank = ec.pg().rank();
    const auto N = etensor.rows(); //TODO

    if(rank == 0)
      tamm_to_eigen_tensor(ttensor, etensor);
    ec.pg().barrier();
    std::vector<T> Hbufv(N*N);
    T *Hbuf = &Hbufv[0];//Hbufv.data();
    Eigen::Map<Matrix>(Hbuf,N,N) = etensor;  
    // GA_Brdcst(Hbuf,N*N*sizeof(T),0);
    MPI_Bcast(Hbuf,N*N,mpi_type<T>(),0,ec.pg().comm());
    etensor = Eigen::Map<Matrix>(Hbuf,N,N);
    Hbufv.clear(); Hbufv.shrink_to_fit();

    // auto hf_t2 = std::chrono::high_resolution_clock::now();
    // auto hf_time =
    //   std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
   
    // //ec.pg().barrier(); //TODO
    // if(rank == 0) std::cout << std::endl << "Time for tamm to eigen " << pstr << " : " << hf_time << " secs" << endl;
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
std::tuple<Matrix, Matrix, size_t, double, double, int64_t> gensqrtinv(
    ExecutionContext& ec, tamm::Tensor<double> S, bool symmetric = false,
    double threshold=1e-5) {

  using T = double;

#ifdef SCALAPACK
  Eigen::SelfAdjointEigenSolver<Matrix> eig_solver(S);
  auto U = eig_solver.eigenvectors();
  auto s = eig_solver.eigenvalues();
  auto s_max = s.maxCoeff();
  auto condition_number = std::min(
      s_max / std::max(s.minCoeff(), std::numeric_limits<double>::min()),
      1.0 / std::numeric_limits<double>::epsilon());
  // auto threshold = s_max / max_condition_number;
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
  int world_rank = ec.pg().rank().value();
  int world_size = ec.pg().size().value();

  int64_t n_cond;
  double condition_number, result_condition_number;
  Matrix X, Xinv;

  const int64_t N = S.tiled_index_spaces()[0].index_space().num_indices(); //S.rows();
  int64_t n_illcond = 0;

  if( world_rank == 0 ) {
    // Eigendecompose S -> VsV**T
    Matrix V(N,N);
    tamm_to_eigen_tensor(S,V);
    T* Vbuf = V.data();

    std::vector<double> s(N);
    linalg::lapack::syevd( 'V', 'L', N, Vbuf, N, s.data() );

    // condition_number = std::min(
    //   s.back() / std::max( s.front(), std::numeric_limits<double>::min() ),
    //   1.       / std::numeric_limits<double>::epsilon()
    // );

    // const auto threshold = s.back() / max_condition_number;
    auto first_above_thresh = std::find_if( s.begin(), s.end(), [&](const auto& x){ return x >= threshold; } );
    result_condition_number = s.back() / *first_above_thresh;

    n_illcond = std::distance( s.begin(), first_above_thresh );
    n_cond    = N - n_illcond;

    if(n_illcond > 0) {
      std::cout << std::endl << "WARNING: Found " << n_illcond << " linear dependencies" << std::endl;
      cout << "First eigen value above tol_lindep = " << *first_above_thresh << endl;
      std::cout << "The overlap matrix has " << n_illcond << " vectors deemed linearly dependent with eigenvalues:" << std::endl;
      
      for( int64_t i = 0; i < n_illcond; i++ ) cout << std::defaultfloat << i+1 << ": " << s[i] << endl;
    }

    // auto* V_cond = Vbuf + n_illcond * N;
    Matrix V_cond = V.block(n_illcond, 0, N-n_illcond, N);
    V.resize(0,0);
    X.resize( N, n_cond ); // Xinv.resize( N, n_cond );
    // X.setZero(N,N); Xinv.setZero( N, N );
    // Matrix V_cond(n_cond,N);
    // V_cond = Eigen::Map<Matrix>(Vbuf + n_illcond * N,n_cond,N);
    X = V_cond.transpose();
    // Xinv = X;
    V_cond.resize(0,0);

    // Form canonical X/Xinv
    for( auto i = 0; i < n_cond; ++i ) {

      const double srt = std::sqrt( *(first_above_thresh + i) );

      // X is row major...
      auto* X_col    = X.data()    + i;
      // auto* Xinv_col = Xinv.data() + i;

      linalg::blas::scal( N, 1./srt, X_col,    n_cond );
      // linalg::blas::scal( N, srt,    Xinv_col, n_cond );

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
    MPI_Bcast( &n_illcond,               1, MPI_INT64_T, 0, world );    
    MPI_Bcast( &condition_number,        1, MPI_DOUBLE,  0, world );
    MPI_Bcast( &result_condition_number, 1, MPI_DOUBLE,  0, world );

    // if( world_rank != 0 ) {
    //   X.resize( N, n_cond ); //Xinv.resize(N, n_cond);
    //   // X.setZero(N,N); Xinv.setZero(N,N);
    // }

    // MPI_Bcast( X.data(),    X.size(),    MPI_DOUBLE, 0, world );
    // MPI_Bcast( Xinv.data(), Xinv.size(), MPI_DOUBLE, 0, world );
  }

  // tAO_ld = TiledIndexSpace{IndexSpace(range(0, n_cond)), final_AO_tilesize};
  // Tensor<T> X{tAO, tAO_ld};

#endif
  return std::make_tuple(X, Xinv, size_t(n_cond), condition_number,
                         result_condition_number, n_illcond);
}

std::tuple<Matrix, Matrix, double> conditioning_orthogonalizer(
  ExecutionContext& ec, SystemData& sys_data, tamm::Tensor<double> S) {
  size_t obs_rank;
  double S_condition_number;
  double XtX_condition_number;
  Matrix X, Xinv;
  int64_t n_illcond;
  double S_condition_number_threshold = sys_data.options_map.scf_options.tol_lindep;

  // assert(S.rows() == S.cols());

  std::tie(X, Xinv, obs_rank, S_condition_number, XtX_condition_number, n_illcond) =
      gensqrtinv(ec, S, false, S_condition_number_threshold);
  auto obs_nbf_omitted = (long)(S.tiled_index_spaces()[0].index_space().num_indices()) - (long)obs_rank;
  // std::cout << "overlap condition number = " << S_condition_number;
  // if (obs_nbf_omitted > 0){
  //   if(GA_Nodeid()==0) std::cout << " (dropped " << obs_nbf_omitted << " "
  //             << (obs_nbf_omitted > 1 ? "fns" : "fn") << " to reduce to "
  //             << XtX_condition_number << ")";
  // }
  // if(GA_Nodeid()==0) std::cout << endl;

  // FIXME:UNCOMMENT
  // if (obs_nbf_omitted > 0) {
  //   Matrix should_be_I = X.transpose() * S * X;
  //   Matrix I = Matrix::Identity(should_be_I.rows(), should_be_I.cols());
  //   if(ec.pg().rank()==0) std::cout << std::endl << "||X^t * S * X - I||_2 = " << (should_be_I - I).norm()
  //             << " (should be 0)" << endl;
  // }

  sys_data.n_lindep = n_illcond;

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


  // if(GA_Nodeid()==0)
  //   std::cout << "computing non-negligible shell-pair list ... ";
    
  // libint2::Timers<1> timer;
  // timer.set_now_overhead(25);
  // timer.start(0);

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
  
  // timer.stop(0);
  // if(GA_Nodeid()==0)     
  //   std::cout << "done (" << timer.read(0) << " s)" << endl;

  return std::make_tuple(splist,spdata);
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
  double epsilon = 0.0;
  Engine engine = Engine(Kernel, std::max(bs1.max_nprim(), bs2.max_nprim()),
                      std::max(bs1.max_l(), bs2.max_l()), 0, epsilon, params);

    // if(GA_Nodeid()==0) {
    //   std::cout << "computing Schwarz bound prerequisites (kernel=" 
    //         << (int)Kernel << ") ... ";
    // }

    // libint2::Timers<1> timer;
    // timer.set_now_overhead(25);
    // timer.start(0);
  
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

  // timer.stop(0);
  // if(GA_Nodeid()==0) 
  //   std::cout << "done (" << timer.read(0) << " s)" << endl;
 
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

#endif // TAMM_METHODS_HF_COMMON_HPP_
