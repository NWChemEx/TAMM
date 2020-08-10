
#ifndef TAMM_METHODS_HF_COMMON_HPP_
#define TAMM_METHODS_HF_COMMON_HPP_

#include <cctype>

// Eigen matrix algebra library
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
// #include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/MatrixFunctions>
#undef I

#include "common/molden.hpp"

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

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <filesystem>
namespace fs = std::filesystem;

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

struct EigenTensors {
  Matrix H,S;
  Matrix C,G,D,F,X;
  Matrix C_occ;
  Matrix C_beta,G_beta,D_beta,F_beta,X_beta;
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

struct SystemData {
  OptionsMap options_map;  
  int  n_occ_alpha;
  int  n_vir_alpha;
  int  n_occ_beta;
  int  n_vir_beta;
  int  n_lindep;
  int  nbf;
  int  nbf_orig;
  int  nelectrons;
  int  nelectrons_alpha;
  int  nelectrons_beta;  
  int  n_frozen_core;
  int  n_frozen_virtual;
  int  nmo;
  int  nocc;
  int  nvir;
  int  focc;
  bool ediis;

  enum SCFType { uhf, rhf, rohf };
  SCFType scf_type; //1-rhf, 2-uhf, 3-rohf
  std::string scf_type_string; 
  std::string input_molecule;
  std::string output_file_prefix;

  //output data
  double scf_energy;
  int scf_iterations;
  int num_chol_vectors;
  int ccsd_iterations;
  double ccsd_corr_energy;
  double ccsd_total_energy;

  void print() {
    std::cout << std::endl << "----------------------------" << std::endl;
    std::cout << "scf_type = " << scf_type_string << std::endl;

    std::cout << "nbf = " << nbf << std::endl;
    std::cout << "nbf_orig = " << nbf_orig << std::endl;
    std::cout << "n_lindep = " << n_lindep << std::endl;
    
    std::cout << "focc = " << focc << std::endl;        
    std::cout << "nmo = " << nmo << std::endl;
    std::cout << "nocc = " << nocc << std::endl;
    std::cout << "nvir = " << nvir << std::endl;
    
    std::cout << "n_occ_alpha = " << n_occ_alpha << std::endl;
    std::cout << "n_vir_alpha = " << n_vir_alpha << std::endl;
    std::cout << "n_occ_beta = " << n_occ_beta << std::endl;
    std::cout << "n_vir_beta = " << n_vir_beta << std::endl;
    
    std::cout << "nelectrons = " << nelectrons << std::endl;
    std::cout << "nelectrons_alpha = " << nelectrons_alpha << std::endl;
    std::cout << "nelectrons_beta = " << nelectrons_beta << std::endl;  
    std::cout << "n_frozen_core = " << n_frozen_core << std::endl;
    std::cout << "n_frozen_virtual = " << n_frozen_virtual << std::endl;
    std::cout << "num_chol_vectors = " << num_chol_vectors << std::endl;
    std::cout << "----------------------------" << std::endl;
  }

  void update() {
      n_frozen_core = 0;
      n_frozen_virtual = 0;
      EXPECTS(nbf == n_occ_alpha + n_vir_alpha); //lin-deps
      EXPECTS(nbf_orig == n_occ_alpha + n_vir_alpha + n_lindep);      
      nocc = n_occ_alpha + n_occ_beta;
      nvir = n_vir_alpha + n_vir_beta;
      EXPECTS(nelectrons == n_occ_alpha + n_occ_beta);
      EXPECTS(nelectrons == nelectrons_alpha+nelectrons_beta);
      nmo = n_occ_alpha + n_vir_alpha + n_occ_beta + n_vir_beta; //lin-deps
  }

  SystemData(OptionsMap options_map_, const std::string scf_type_string)
    : options_map(options_map_), scf_type_string(scf_type_string) {
      scf_type = SCFType::rhf;
      if(scf_type_string == "uhf")       { focc = 1; scf_type = SCFType::uhf; }
      else if(scf_type_string == "rhf")  { focc = 2; scf_type = SCFType::rhf; }
      else if(scf_type_string == "rohf") { focc = -1; scf_type = SCFType::rohf; }
    }

};

std::string getfilename(std::string filename){
  size_t lastindex = filename.find_last_of(".");
  auto fname = filename.substr(0,lastindex);
  return fname.substr(fname.find_last_of("/")+1,fname.length());
}

void write_results(SystemData sys_data, const std::string module){
  auto options = sys_data.options_map;
  auto scf = options.scf_options;
  auto cd = options.cd_options;
  auto ccsd = options.ccsd_options;
  std::string l_module = module;
  to_lower(l_module);
  std::string json_file = sys_data.input_molecule+"."+l_module+".json";
  bool json_exists = std::filesystem::exists(json_file);

  json results =  json::object();

  if(json_exists){
    // std::ifstream jread(json_file);
    // jread >> results;
    std::filesystem::remove(json_file);
  }
  
  auto str_bool = [=] (const bool val) {
    if (val) return "true";
    return "false";
  };

  results["input"]["molecule"]["name"] = sys_data.input_molecule;
  results["input"]["molecule"]["basis"] = scf.basis;
  results["input"]["molecule"]["basis_sphcart"] = scf.sphcart;
  results["input"]["molecule"]["geometry_units"] = scf.geom_units;
  //SCF options
  results["input"]["SCF"]["tol_int"] = scf.tol_int;
  results["input"]["SCF"]["tol_lindep"] = scf.tol_lindep;
  results["input"]["SCF"]["conve"] = scf.conve;
  results["input"]["SCF"]["convd"] = scf.convd;
  results["input"]["SCF"]["diis_hist"] = scf.diis_hist;
  results["input"]["SCF"]["AO_tilesize"] = scf.AO_tilesize;
  results["input"]["SCF"]["force_tilesize"] = str_bool(scf.force_tilesize);
  results["input"]["SCF"]["scf_type"] = scf.scf_type;
  results["input"]["SCF"]["multiplicity"] = scf.multiplicity;

  //SCF output
  results["output"]["SCF"]["energy"] = sys_data.scf_energy;
  results["output"]["SCF"]["n_iterations"] = sys_data.scf_iterations;

  if(module == "CD" || module == "CCSD"){
    //CD options
    results["input"]["CD"]["diagtol"] = cd.diagtol;
    results["input"]["CD"]["max_cvecs_factor"] = cd.max_cvecs_factor;
    //CD output
    results["output"]["CD"]["n_cholesky_vectors"] = sys_data.num_chol_vectors;
  }

  if(module == "CCSD") {
    //CCSD options
    results["input"]["CCSD"]["threshold"] = ccsd.threshold;
    results["input"]["CCSD"]["tilesize"] = ccsd.tilesize;
    results["input"]["CCSD"]["itilesize"] = ccsd.itilesize;
    results["input"]["CCSD"]["ncuda"] = ccsd.icuda;
    results["input"]["CCSD"]["ndiis"] = ccsd.ndiis;
    results["input"]["CCSD"]["readt"] = str_bool(ccsd.readt);
    results["input"]["CCSD"]["writet"] = str_bool(ccsd.writet);
    results["input"]["CCSD"]["ccsd_maxiter"] = ccsd.ccsd_maxiter;
    results["input"]["CCSD"]["balance_tiles"] = str_bool(ccsd.balance_tiles);
  
    //CCSD output
    results["output"]["CCSD"]["n_iterations"] =   sys_data.ccsd_iterations;
    results["output"]["CCSD"]["energy"]["correlation"] =  sys_data.ccsd_corr_energy;
    results["output"]["CCSD"]["energy"]["total"] =  sys_data.ccsd_total_energy;
  }
  
  // std::cout << std::endl << std::endl << results.dump() << std::endl;
  std::ofstream res_file(json_file);
  res_file << std::setw(4) << results << std::endl;
}

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

template<typename T>
void readMD(std::vector<T>& mbuf, std::vector<T>& dbuf, std::string movecsfile, std::string densityfile) {

  auto mfile_id = H5Fopen(movecsfile.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  auto dfile_id = H5Fopen(densityfile.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

  auto mdataset_id = H5Dopen(mfile_id, "movecs",  H5P_DEFAULT);
  auto ddataset_id = H5Dopen(dfile_id, "density", H5P_DEFAULT);

   /* Read the datasets. */
  H5Dread(mdataset_id, get_hdf5_dt<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, mbuf.data());
  H5Dread(ddataset_id, get_hdf5_dt<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, dbuf.data());                    

  H5Dclose(mdataset_id);
  H5Dclose(ddataset_id);
  H5Fclose(mfile_id);
  H5Fclose(dfile_id);
}

void writeC(Matrix& C, std::string scf_files_prefix){
  std::string outputfile = scf_files_prefix + ".movecs";
  const auto N = C.rows();
  const auto Northo = C.cols();
  std::vector<TensorType> Cbuf(N*Northo);
  TensorType *buf = Cbuf.data();
  Eigen::Map<Matrix>(buf,N,Northo) = C;  

  // out.write((char *)(buf), sizeof(TensorType) *N*Northo);

  /* Create a file. */
  hid_t file_id = H5Fcreate(outputfile.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  hsize_t tsize = N*Northo;
  hid_t dataspace_id = H5Screate_simple(1, &tsize, NULL);

  /* Create dataset. */
  hid_t dataset_id = H5Dcreate(file_id, "movecs", get_hdf5_dt<TensorType>(), dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  /* Write the dataset. */
  /* herr_t status = */ H5Dwrite(dataset_id, get_hdf5_dt<TensorType>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);   

  H5Dclose(dataset_id);
  H5Sclose(dataspace_id);
  H5Fclose(file_id);   
}

void writeD(Matrix& D, std::string scf_files_prefix){
  std::string outputfile = scf_files_prefix + ".density";
  const auto N = D.rows();
  std::vector<TensorType> Dbuf(N*N);
  TensorType *buf = Dbuf.data();
  Eigen::Map<Matrix>(buf,N,N) = D;  

  // out.write((char *)(buf), sizeof(TensorType) *N*N);

  hid_t file_id = H5Fcreate(outputfile.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  hsize_t tsize = N*N;
  hid_t dataspace_id = H5Screate_simple(1, &tsize, NULL);

  hid_t dataset_id = H5Dcreate(file_id, "density", get_hdf5_dt<TensorType>(), dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  /* herr_t status = */ H5Dwrite(dataset_id, get_hdf5_dt<TensorType>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);   

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
    const ExecutionContext& ec, const Matrix& S, bool symmetric = false,
    double threshold=1e-5) {
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

  const int64_t N = S.rows();
  int64_t n_illcond = 0;

  if( world_rank == 0 ) {

  // Eigendecompose S -> VsV**T
  Eigen::MatrixXd V = S;
  std::vector<double> s(N);
  linalg::lapack::syevd( 'V', 'L', N, V.data(), N, s.data() );

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

  auto* V_cond = V.data() + n_illcond * N;
  X.resize( N, n_cond ); Xinv.resize( N, n_cond );
  // X.setZero(N,N); Xinv.setZero( N, N );

  // Form canonical X/Xinv
  for( auto i = 0; i < n_cond; ++i ) {

    const double srt = std::sqrt( *(first_above_thresh + i) );

    // X is row major...
    auto* X_col    = X.data()    + i;
    auto* Xinv_col = Xinv.data() + i;

    linalg::blas::copy( N, V_cond + i*N, 1, X_col,    n_cond );
    linalg::blas::copy( N, V_cond + i*N, 1, Xinv_col, n_cond );
    linalg::blas::scal( N, 1./srt, X_col,    n_cond );
    linalg::blas::scal( N, srt,    Xinv_col, n_cond );

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

    if( world_rank != 0 ) {
      X.resize( N, n_cond ); Xinv.resize(N, n_cond);
      // X.setZero(N,N); Xinv.setZero(N,N);
    }

    MPI_Bcast( X.data(),    X.size(),    MPI_DOUBLE, 0, world );
    MPI_Bcast( Xinv.data(), Xinv.size(), MPI_DOUBLE, 0, world );
  }

#endif
  return std::make_tuple(X, Xinv, size_t(n_cond), condition_number,
                         result_condition_number, n_illcond);
}

std::tuple<Matrix, Matrix, double> conditioning_orthogonalizer(
  const ExecutionContext& ec, SystemData& sys_data, const Matrix& S) {
  size_t obs_rank;
  double S_condition_number;
  double XtX_condition_number;
  Matrix X, Xinv;
  int64_t n_illcond;
  double S_condition_number_threshold = sys_data.options_map.scf_options.tol_lindep;

  assert(S.rows() == S.cols());

  std::tie(X, Xinv, obs_rank, S_condition_number, XtX_condition_number, n_illcond) =
      gensqrtinv(ec, S, false, S_condition_number_threshold);
  auto obs_nbf_omitted = (long)S.rows() - (long)obs_rank;
  // std::cout << "overlap condition number = " << S_condition_number;
  // if (obs_nbf_omitted > 0){
  //   if(GA_Nodeid()==0) std::cout << " (dropped " << obs_nbf_omitted << " "
  //             << (obs_nbf_omitted > 1 ? "fns" : "fn") << " to reduce to "
  //             << XtX_condition_number << ")";
  // }
  // if(GA_Nodeid()==0) std::cout << endl;

  if (obs_nbf_omitted > 0) {
    Matrix should_be_I = X.transpose() * S * X;
    Matrix I = Matrix::Identity(should_be_I.rows(), should_be_I.cols());
    if(ec.pg().rank()==0) std::cout << std::endl << "||X^t * S * X - I||_2 = " << (should_be_I - I).norm()
              << " (should be 0)" << endl;
  }

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
