
#ifndef TAMM_TESTS_HF_TAMM_HPP_
#define TAMM_TESTS_HF_TAMM_HPP_

// standard C++ headers
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <tuple>
#include <vector>

// #define EIGEN_USE_BLAS
// #define EIGEN_USE_LAPACKE
// #define EIGEN_USE_MKL_ALL

// Eigen matrix algebra library
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/MatrixFunctions>

// Libint Gaussian integrals library
#include <libint2.hpp>
#include <libint2/basis.h>
#include <libint2/chemistry/sto3g_atomic_density.h>

#ifdef _OPENMP
#include <omp.h>
#endif
#include "tamm/eigen_utils.hpp"
#include "tamm/tamm.hpp"

using namespace tamm;
using std::cerr;
using std::cout;
using std::endl;
using std::string;

using Matrix =
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Tensor2D   = Eigen::Tensor<double, 2, Eigen::RowMajor>;
using Tensor3D   = Eigen::Tensor<double, 3, Eigen::RowMajor>;
using Tensor4D   = Eigen::Tensor<double, 4, Eigen::RowMajor>;
using TensorType = double;

using shellpair_list_t = std::unordered_map<size_t, std::vector<size_t>>;
shellpair_list_t obs_shellpair_list;  // shellpair list for OBS
using shellpair_data_t = std::vector<std::vector<std::shared_ptr<libint2::ShellPair>>>;  // in same order as shellpair_list_t
shellpair_data_t obs_shellpair_data;  // shellpair data for OBS

void diis(ExecutionContext& ec, TiledIndexSpace& tAO, tamm::Tensor<TensorType> F, Tensor<TensorType> E, int iter, int max_hist,
          int idiis, std::vector<Tensor<TensorType>>& diis_hist,
          std::vector<tamm::Tensor<TensorType>>& fock_hist);

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

const auto max_engine_precision = std::numeric_limits<double>::epsilon() / 1e10;

using libint2::Atom;

string read_option(std::istream& is, string optionstr, string optiontype){
  while (std::getline(is, optionstr)){
    if (optionstr.empty()) continue;
    else {
        std::istringstream oss(optionstr);
        std::vector<std::string> option_string{
          std::istream_iterator<std::string>{oss},
          std::istream_iterator<std::string>{}};
        assert(option_string.size() == 2);
        assert(option_string[0] == optiontype);
        optionstr = option_string[1];
        break;
    }
  }
  return optionstr;
}

inline std::tuple<std::vector<Atom>, std::string, bool, int, double, double, double, int>
   read_input_xyz(std::istream& is) {
    const double angstrom_to_bohr =
      1.889725989; // 1 / bohr_to_angstrom; //1.889726125
    // first line = # of atoms
    size_t natom;
    is >> natom;
    // read off the rest of first line and discard
    std::string rest_of_line;
    std::getline(is, rest_of_line);

    // second line = comment
    std::string comment;
    std::getline(is, comment);

    // third line - geometry units
    std::string gm_units;
    std::getline(is, gm_units);
    std::istringstream iss(gm_units);
    std::vector<std::string> geom_units{std::istream_iterator<std::string>{iss},
                                        std::istream_iterator<std::string>{}};

    bool nw_units_bohr = true;
    assert(geom_units.size() == 3);
    if(geom_units[2] == "angstrom") nw_units_bohr = false;

    // rest of lines are atoms
    std::vector<Atom> atoms(natom);
    for(size_t i = 0; i < natom; i++) {
        // read line
        std::string line;
        std::getline(is, line);
        std::istringstream iss(line);
        // then parse ... this handles "extended" XYZ formats
        std::string element_symbol;
        double x, y, z;
        iss >> element_symbol >> x >> y >> z;

        // .xyz files report element labels, hence convert to atomic numbers
        int Z = -1;
        for(const auto& e : libint2::chemistry::get_element_info()) {
            if(libint2::strcaseequal(e.symbol, element_symbol)) {
                Z = e.Z;
                break;
            }
        }
        if(Z == -1) {
            std::ostringstream oss;
            oss << "read_dotxyz: element symbol \"" << element_symbol
                << "\" is not recognized" << std::endl;
            throw std::runtime_error(oss.str().c_str());
        }

        atoms[i].atomic_number = Z;

        if(nw_units_bohr) {
            atoms[i].x = x;
            atoms[i].y = y;
            atoms[i].z = z;
        }

        else { // assume angstrom
            // .xyz files report Cartesian coordinates in angstroms; convert to
            // bohr
            atoms[i].x = x * angstrom_to_bohr;
            atoms[i].y = y * angstrom_to_bohr;
            atoms[i].z = z * angstrom_to_bohr;
        }
    }

    std::string basis_set = "sto-3g";
    basis_set = read_option(is, basis_set, "basis");

    string read_debug = "false";
    read_debug = read_option(is, read_debug, "debug");
    bool debug = false;
    if(read_debug == "true") debug = true;
    string read_maxiter = "100";
    int maxiter = stoi(read_option(is, read_maxiter, "maxiter"));
    string read_tol_int = "1e-8";
    double tol_int = stod(read_option(is, read_tol_int, "tol_int"));
    string read_conve = "1e-6";
    double conve = stod(read_option(is, read_conve, "conve"));
    string read_convd = "1e-5";
    double convd = stod(read_option(is, read_convd, "convd"));
    string read_diis_hist = "10";
    int diis_hist = stod(read_option(is, read_diis_hist, "diis_hist"));

    return std::make_tuple(atoms, basis_set, debug, maxiter, tol_int, conve, convd, diis_hist);
}

void compare_eigen_tamm_tensors(Tensor<TensorType>& tamm_tensor,
                                Matrix& eigen_tensor, bool dprint = false) {
    bool he           = true;
    auto tamm_2_eigen = tamm_to_eigen_tensor<TensorType, 2>(tamm_tensor);

    const int rows = tamm_2_eigen.dimension(0);
    const int cols = tamm_2_eigen.dimension(1);
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            if(std::fabs(tamm_2_eigen(i, j) - eigen_tensor(i, j)) > 1.0e-10)
                he = false;
        }
    }

    if(dprint) {
        cout << "eigen tensor\n-----------\n";
        cout << eigen_tensor << endl;
        cout << "tamm tensor\n-----------\n";
        cout << tamm_2_eigen << endl;
    }
    if(he)
        cout << "tamm - eigen tensors are equal\n";
    else
        cout << "tamm - eigen tensors are NOT equal\n";

    // tamm_2_eigen.resize(0,0);
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
    const Matrix& S, bool symmetric = false,
    double max_condition_number = 1e8) {
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
  return std::make_tuple(X, Xinv, size_t(n_cond), condition_number,
                         result_condition_number);
}

std::tuple<Matrix, Matrix, double> conditioning_orthogonalizer(
    const Matrix& S, double S_condition_number_threshold) {
  size_t obs_rank;
  double S_condition_number;
  double XtX_condition_number;
  Matrix X, Xinv;

  assert(S.rows() == S.cols());

  std::tie(X, Xinv, obs_rank, S_condition_number, XtX_condition_number) =
      gensqrtinv(S, false, S_condition_number_threshold);
  auto obs_nbf_omitted = (long)S.rows() - (long)obs_rank;
//   std::cout << "overlap condition number = " << S_condition_number;
  if (obs_nbf_omitted > 0){
    if(GA_Nodeid()==0) std::cout << " (dropped " << obs_nbf_omitted << " "
              << (obs_nbf_omitted > 1 ? "fns" : "fn") << " to reduce to "
              << XtX_condition_number << ")";
  }
  if(GA_Nodeid()==0) std::cout << std::endl;

  if (obs_nbf_omitted > 0) {
    Matrix should_be_I = X.transpose() * S * X;
    Matrix I = Matrix::Identity(should_be_I.rows(), should_be_I.cols());
    if(GA_Nodeid()==0) std::cout << "||X^t * S * X - I||_2 = " << (should_be_I - I).norm()
              << " (should be 0)" << std::endl;
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
    for (auto s1 = 0l, s12 = 0l; s1 != nsh1; ++s1) {
      // mx.lock();
      if (splist.find(s1) == splist.end())
        splist.insert(std::make_pair(s1, std::vector<size_t>()));
      // mx.unlock();

      auto n1 = bs1[s1].size();  // number of basis functions in this shell

      auto s2_max = bs1_equiv_bs2 ? s1 : nsh2 - 1;
      for (auto s2 = 0; s2 <= s2_max; ++s2, ++s12) {
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
    for (auto s1 = 0l; s1 != nsh1; ++s1) {
      // if (s1 % nthreads == thread_id) {
        auto& list = splist[s1];
        std::sort(list.begin(), list.end());
      }
    // }


  // compute shellpair data assuming that we are computing to default_epsilon
  // N.B. only parallelized over 1 shell index
  const auto ln_max_engine_precision = std::log(max_engine_precision);
  shellpair_data_t spdata(splist.size());
  
    for (auto s1 = 0l; s1 != nsh1; ++s1) {
      // if (s1 % nthreads == thread_id) {
        for(const auto& s2 : splist[s1]) {
          spdata[s1].emplace_back(std::make_shared<libint2::ShellPair>(bs1[s1],bs2[s2],ln_max_engine_precision));
        }
      // }
    }
  
  timer.stop(0);
  if(GA_Nodeid()==0)     
    std::cout << "done (" << timer.read(0) << " s)" << std::endl;

  return std::make_tuple(splist,spdata);
}

std::tuple<int, int, double, libint2::BasisSet> hartree_fock(
  const string filename, Matrix& C, Matrix& F) {
    // Perform the simple HF calculation (Ed) and 2,4-index transform to get the
    // inputs for CCSD
    using libint2::Atom;
    using libint2::Engine;
    using libint2::Operator;
    using libint2::Shell;
    using libint2::BasisSet;

    /*** =========================== ***/
    /*** initialize molecule         ***/
    /*** =========================== ***/

    // read geometry from a file; by default read from h2o.xyz, else take
    // filename (.xyz) from the command line
    auto is = std::ifstream(filename);
    std::vector<Atom> atoms;
    std::string basis;

    int maxiter = 50;
    double conve = 1e-6;
    double convd = 1e-5;
    double tol_int = 1e-8;
    int max_hist = 10; 
    auto debug = false;

    std::tie(atoms, basis, debug, maxiter, tol_int, conve, convd, max_hist) = read_input_xyz(is);

    tol_int = std::min(1e-8, 0.01 * conve);

    if(GA_Nodeid()==0){
      cout << "\nNumber of GA ranks: " << GA_Nnodes();
      cout << "\nreading geometry from file: " << filename;
      cout << "\n----------------------------------";
      cout << "\ndiis hist = " << max_hist;
      cout << "\nBasis set = " << basis;
      cout << "\nmax iterations = " << maxiter;
      cout << "\nIntegral tolerance = " << tol_int;
      cout << "\nEnergy convergence = " << conve;
      cout << "\nDensity convergence = " << convd;
      cout << "\n----------------------------------";
    }

    auto hf_t1 = std::chrono::high_resolution_clock::now();

    //  std::cout << "Geometries in bohr units \n";
    //  for (auto i = 0; i < atoms.size(); ++i)
    //    std::cout << atoms[i].atomic_number << "  " << atoms[i].x<< "  " <<
    //    atoms[i].y<< "  " << atoms[i].z << std::endl;
    // count the number of electrons
    auto nelectron = 0;
    for(size_t i = 0; i < atoms.size(); ++i)
        nelectron += atoms[i].atomic_number;
    const auto ndocc = nelectron / 2;

    // compute the nuclear repulsion energy
    auto enuc = 0.0;
    for(size_t i = 0; i < atoms.size(); i++)
        for(size_t j = i + 1; j < atoms.size(); j++) {
            auto xij = atoms[i].x - atoms[j].x;
            auto yij = atoms[i].y - atoms[j].y;
            auto zij = atoms[i].z - atoms[j].z;
            auto r2  = xij * xij + yij * yij + zij * zij;
            auto r   = sqrt(r2);
            enuc += atoms[i].atomic_number * atoms[j].atomic_number / r;
        }
    if(GA_Nodeid()==0) cout << "\nNuclear repulsion energy = " << enuc << endl;

    // initializes the Libint integrals library ... now ready to compute
    libint2::initialize(debug);

    /*** =========================== ***/
    /*** create basis set            ***/
    /*** =========================== ***/

    // LIBINT_INSTALL_DIR/share/libint/2.4.0-beta.1/basis
    libint2::BasisSet shells(std::string(basis), atoms);
    // auto shells = make_sto3g_basis(atoms);

    // compute OBS non-negligible shell-pair list
    {
      std::tie(obs_shellpair_list, obs_shellpair_data) = compute_shellpairs(shells);
      size_t nsp = 0;
      for (auto& sp : obs_shellpair_list) {
        nsp += sp.second.size();
      }
      if(GA_Nodeid()==0) std::cout << "# of {all,non-negligible} shell-pairs = {"
                << shells.size() * (shells.size() + 1) / 2 << "," << nsp << "}"
                << std::endl;
    }

    const size_t N = nbasis(shells);
    size_t nao = N;

    if(GA_Nodeid()==0) cout << "\nNumber of basis functions: " << N << endl;

    /*** =========================== ***/
    /*** compute 1-e integrals       ***/
    /*** =========================== ***/

    Engine engine(Operator::overlap, max_nprim(shells), max_l(shells), 0);
    auto& buf = unconst_cast(engine.results());

    Tensor<TensorType> tensor1e;
    auto compute_1body_ints = [&](const IndexVector& blockid) {

        auto s1 = blockid[0];
        // auto bf1 = shell2bf[s1]; //shell2bf[s1]; // first basis function in
        // this shell
        auto n1 = shells[s1].size();

        // for (size_t s2 = 0; s2 <= s1; ++s2) {
        auto s2 = blockid[1];
        if (s2>s1) return;

        auto s2spl = obs_shellpair_list[s1];
        if(std::find(s2spl.begin(),s2spl.end(),s2) == s2spl.end()) return;

        // auto bf2 = shell2bf[s2];
        auto n2 = shells[s2].size();

        std::vector<TensorType> tbuf(n1*n2);

        // compute shell pair; return is the pointer to the buffer
        engine.compute(shells[s1], shells[s2]);
        // "map" buffer to a const Eigen Matrix, and copy it to the
        // corresponding blocks of the result
        Eigen::Map<const Matrix> buf_mat(buf[0], n1, n2);
        // result.block(bf1, bf2, n1, n2) = buf_mat;
        // for(size_t i = 0; i < n1; i++)
        //     for(size_t j = 0; j < n2; j++) tbuf[i * n2 + j] = buf_mat(i, j);
        //std::memcpy(tbuf.data(),buf, sizeof(TensorType)*n1*n2);
        Eigen::Map<Matrix>(tbuf.data(),n1,n2) = buf_mat;

        tensor1e.put(blockid, tbuf);

        if(s1!=s2){
            std::vector<TensorType> ttbuf(n1*n2);
            Eigen::Map<Matrix>(ttbuf.data(),n2,n1) = buf_mat.transpose();
            tensor1e.put({s2,s1}, ttbuf);
        }
    };
    
    ProcGroup pg{GA_MPI_Comm()};
    auto mgr = MemoryManagerGA::create_coll(pg);
    Distribution_NW distribution;
    ExecutionContext* ec = new ExecutionContext{pg, &distribution, mgr};
    auto rank = ec->pg().rank();
    

    IndexSpace AO{range(0, N)};
    std::vector<unsigned int> AO_tiles;
    for(auto s : shells) AO_tiles.push_back(s.size());
    if(rank==0) cout << "Number of AO tiles = " << AO_tiles.size() << endl;

    tamm::Tile tile_size = 6; 
    if(N>=30) tile_size = 30;
    TiledIndexSpace tAO{AO, tile_size};
    TiledIndexSpace tAOt{AO, AO_tiles};
    auto [mu, nu, ku] = tAO.labels<3>("all");
    auto [mup, nup, kup] = tAOt.labels<3>("all");

    // Tensor<TensorType> SL1{{tAO, tAO}, compute_1body_ints};
    // Tensor<TensorType> T1{{tAO, tAO}, compute_1body_ints};
    // Tensor<TensorType> V1{{tAO, tAO}, compute_1body_ints};

    Tensor<TensorType> H1{tAO, tAO};

    Tensor<TensorType> H1P{tAOt, tAOt};
    Tensor<TensorType> S1{tAOt, tAOt};
    Tensor<TensorType> T1{tAOt, tAOt};
    Tensor<TensorType> V1{tAOt, tAOt};
    Tensor<TensorType>::allocate(ec, H1, H1P, S1, T1, V1);

    Matrix H, S; 
    H.setZero(N, N);
    S.setZero(N, N);

    auto hf_t2 = std::chrono::high_resolution_clock::now();

    double hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
    if(rank == 0) std::cout << "\nTime taken for initial setup: " << hf_time << " secs\n";

    // auto T = compute_1body_ints(shells, Operator::kinetic);
    // auto S = compute_1body_ints(shells, Operator::overlap);
    // Matrix V = compute_1body_ints(shells, Operator::nuclear, atoms);
    // Matrix H = T;

    hf_t1 = std::chrono::high_resolution_clock::now();

    // Scheduler{ec}(S1(mu, nu) = SL1(mu, nu)).execute();
    tensor1e = S1;
    block_for(*ec, S1(), compute_1body_ints);

    engine.set(Operator::kinetic);
    buf = engine.results();
    // Core Hamiltonian = T + V
    tensor1e = T1;
    block_for(*ec, T1(), compute_1body_ints);
    Scheduler{*ec}(H1P(mup, nup) = T1(mup, nup)).execute();

    engine.set(Operator::nuclear);
    std::vector<std::pair<double, std::array<double, 3>>> q;
    for(const auto& atom : atoms) {
        q.push_back({static_cast<double>(atom.atomic_number),
                        {{atom.x, atom.y, atom.z}}});
    }
    engine.set_params(q);
    buf = engine.results();
    //H =  H + V;
    tensor1e = V1;
    block_for(*ec, V1(), compute_1body_ints);
    Scheduler{*ec}(H1P(mup, nup) += V1(mup, nup)).execute();

    hf_t2 = std::chrono::high_resolution_clock::now();
    hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
    //GA_Sync();
    if(rank == 0) std::cout << "\nTime taken for H=T+V, S: " << hf_time << " secs\n";

    hf_t1 = std::chrono::high_resolution_clock::now();

    //TODO: These copies are expensive since we are reshaping blocks of H1P to get H1
    //copy H1P to H using rank 0 and broadcast to get better performance
    if(rank == 0)
      tamm_to_eigen_tensor(H1P, H);
    GA_Sync();
    std::vector<TensorType> Hbufv(N*N);
    TensorType *Hbuf = Hbufv.data();
    Eigen::Map<Matrix>(Hbuf,N,N) = H;  
    GA_Brdcst(Hbuf,N*N*sizeof(TensorType),0);
    H = Eigen::Map<Matrix>(Hbuf,N,N);
    Hbufv.clear();

    hf_t2 = std::chrono::high_resolution_clock::now();
    hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
   
    if(rank == 0) std::cout << "\nTime taken for tamm to eigen - H1P-H: " << hf_time << " secs\n";

    hf_t1 = std::chrono::high_resolution_clock::now();
    // tamm_to_eigen_tensor(S1, S);
    if(rank == 0)
      tamm_to_eigen_tensor(S1, S);
    GA_Sync();
    std::vector<TensorType> Sbufv(N*N);
    TensorType *Sbuf = Sbufv.data();
    Eigen::Map<Matrix>(Sbuf,N,N) = S;  
    GA_Brdcst(Sbuf,N*N*sizeof(TensorType),0);
    S = Eigen::Map<Matrix>(Sbuf,N,N);
    Sbufv.clear();
    hf_t2 = std::chrono::high_resolution_clock::now();
    hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
    GA_Sync();
    if(rank == 0) std::cout << "\nTime taken for tamm to eigen - S1-S: " << hf_time << " secs\n";

    hf_t1 = std::chrono::high_resolution_clock::now();
    eigen_to_tamm_tensor(H1, H);
    hf_t2 = std::chrono::high_resolution_clock::now();
    hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
    if(rank == 0) std::cout << "\nTime taken for eigen to tamm - H-H1: " << hf_time << " secs\n";

    Tensor<TensorType>::deallocate(H1P, S1, T1, V1);

    /*** =========================== ***/
    /*** build initial-guess density ***/
    /*** =========================== ***/

    hf_t1 = std::chrono::high_resolution_clock::now();

    // compute orthogonalizer X such that X.transpose() . S . X = I
    Matrix X, Xinv;
    double XtX_condition_number;  // condition number of "re-conditioned"
                                  // overlap obtained as Xinv.transpose() . Xinv
    // one should think of columns of Xinv as the conditioned basis
    // Re: name ... cond # (Xinv.transpose() . Xinv) = cond # (X.transpose() .
    // X)
    // by default assume can manage to compute with condition number of S <=
    // 1/eps
    // this is probably too optimistic, but in well-behaved cases even 10^11 is
    // OK
    double S_condition_number_threshold =
        1.0 / std::numeric_limits<double>::epsilon();
    std::tie(X, Xinv, XtX_condition_number) =
        conditioning_orthogonalizer(S, S_condition_number_threshold);

    // TODO Redeclare TAMM S1 with new dims?
    hf_t2 = std::chrono::high_resolution_clock::now();
    hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

    if(rank == 0) std::cout << "Time for computing orthogonalizer: " << hf_time << " secs\n\n";

    // pre-compute data for Schwarz bounds
    auto SchwarzK = compute_schwarz_ints<>(shells);

    hf_t1 = std::chrono::high_resolution_clock::now();

  const auto use_hcore_guess = false;  // use core Hamiltonian eigenstates to guess density?
  // set to true to match the result of versions 0, 1, and 2 of the code
  // HOWEVER !!! even for medium-size molecules hcore will usually fail !!!
  // thus set to false to use Superposition-Of-Atomic-Densities (SOAD) guess
  Matrix D;
  if (use_hcore_guess) { // hcore guess

    // solve H C = e S C
    Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> gen_eig_solver(H, S);
    auto eps = gen_eig_solver.eigenvalues();
    C = gen_eig_solver.eigenvectors();

    // compute density, D = C(occ) . C(occ)T
    auto C_occ = C.leftCols(ndocc);
    D = C_occ * C_occ.transpose();

     F = H;

    // F += compute_2body_fock_general(
    //     shells, D, shells, true /* SOAD_D_is_shelldiagonal */,
    //     std::numeric_limits<double>::epsilon()  // this is cheap, no reason
    //                                             // to be cheaper
    //     );

    F +=    compute_2body_fock(shells, D, 1e-8, SchwarzK);

    Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> gen_eig_solver1(F, S);
    eps = gen_eig_solver1.eigenvalues();
    C = gen_eig_solver1.eigenvectors();
    // compute density, D = C(occ) . C(occ)T
    C_occ = C.leftCols(ndocc);
    D = C_occ * C_occ.transpose();

  } else {  // SOAD as the guess density
    
    auto D_minbs = compute_soad(atoms);  // compute guess in minimal basis
    BasisSet minbs("STO-3G", atoms);

    if(rank == 0) std::cout << 
    "\nProjecting minimal basis SOAD onto basis set specified (" << basis << ")\n";
    
    auto Ft = H;

    Ft += compute_2body_fock_general(
        shells, D_minbs, minbs, true /* SOAD_D_is_shelldiagonal */,
        std::numeric_limits<double>::epsilon()  // this is cheap, no reason
                                                // to be cheaper
        );

#if 0
  double precision = std::numeric_limits<double>::epsilon();
  bool D_is_shelldiagonal = true;
  const libint2::BasisSet& obs = shells;
  const libint2::BasisSet& D_bs = minbs;
  // const Matrix& D = D_minbs; 

  Matrix G = Matrix::Zero(N,N);
  Tensor<TensorType> F1tmp{tAOt, tAOt}; //not allocated
  Tensor<TensorType> F1tmp1{tAO, tAO};
  Tensor<TensorType>::allocate(ec, F1tmp1);

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

    auto compute_2body_fock_general_lambda = [&](IndexVector blockid) {

      //const IndexVector blockid = 
        //  internal::translate_blockid(blockid1, F1tmp());

        using libint2::Engine;
 
        auto s1 = blockid[0];
        auto bf1_first = shell2bf[s1];  // first basis function in this shell
        auto n1 = obs[s1].size();       // number of basis functions in this shell

        auto s2 = blockid[1];
        if(s2>s1) return;

        auto bf2_first = shell2bf[s2];
        auto n2 = obs[s2].size();

        for (auto s3 = 0; s3 < D_bs.size(); ++s3) {
          auto bf3_first = shell2bf_D[s3];
          auto n3 = D_bs[s3].size();

          auto s4_begin = D_is_shelldiagonal ? s3 : 0;
          auto s4_fence = D_is_shelldiagonal ? s3 + 1 : D_bs.size();

          for (auto s4 = s4_begin; s4 != s4_fence; ++s4) {

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
                for (auto f1 = 0, f1234 = 0; f1 != n1; ++f1) {
                  const auto bf1 = f1 + bf1_first;
                  for (auto f2 = 0; f2 != n2; ++f2) {
                    const auto bf2 = f2 + bf2_first;
                    for (auto f3 = 0; f3 != n3; ++f3) {
                      const auto bf3 = f3 + bf3_first;
                      for (auto f4 = 0; f4 != n4; ++f4, ++f1234) {
                        const auto bf4 = f4 + bf4_first;

                        const auto value = buf_1234[f1234];
                        const auto value_scal_by_deg = value * s1234_deg;
                        G(bf1, bf2) += 2.0 * D_minbs(bf3, bf4) * value_scal_by_deg;
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

            for (auto f1 = 0, f1324 = 0; f1 != n1; ++f1) {
              const auto bf1 = f1 + bf1_first;
              for (auto f3 = 0; f3 != n3; ++f3) {
                const auto bf3 = f3 + bf3_first;
                for (auto f2 = 0; f2 != n2; ++f2) {
                  const auto bf2 = f2 + bf2_first;
                  for (auto f4 = 0; f4 != n4; ++f4, ++f1324) {
                    const auto bf4 = f4 + bf4_first;

                    const auto value = buf_1324[f1324];
                    const auto value_scal_by_deg = value * s12_deg;
                    G(bf1, bf2) -= D_minbs(bf3, bf4) * value_scal_by_deg;
                  }
                }
              }
            }
          }
        }

      // symmetrize the result and return
      Matrix Gt = 0.5 * (G + G.transpose());
      G = Gt;
    };

    Scheduler{*ec}(F1tmp1() = 0).execute();

    block_for(*ec, F1tmp(), compute_2body_fock_general_lambda);
    eigen_to_tamm_tensor_acc(F1tmp1,G);
    GA_Sync();
    tamm_to_eigen_tensor(F1tmp1,G);
    Ft += G;
    G.resize(0,0);
    Tensor<TensorType>::deallocate(F1tmp1);
#endif

    D_minbs.resize(0,0);
    // Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> gen_eig_solver(Ft, S);
    // auto eps = gen_eig_solver.eigenvalues();
    // C = gen_eig_solver.eigenvectors();

    // solve F C = e S C by (conditioned) transformation to F' C' = e C',
    // where
    // F' = X.transpose() . F . X; the original C is obtained as C = X . C'
    Eigen::SelfAdjointEigenSolver<Matrix> eig_solver(X.transpose() * Ft * X);
    C = X * eig_solver.eigenvectors();

    // compute density, D = C(occ) . C(occ)T
    auto C_occ = C.leftCols(ndocc);
    D = C_occ * C_occ.transpose();

  }

    H.resize(0,0);
    hf_t2 = std::chrono::high_resolution_clock::now();
    hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

    GA_Sync();
    if(rank == 0) std::cout << "\nTime taken to compute initial guess: " << hf_time << " secs\n";


    hf_t1 = std::chrono::high_resolution_clock::now();
    /*** =========================== ***/
    /*** main iterative loop         ***/
    /*** =========================== ***/

    // bool simple_convergence = false;
    // double alpha = 0.5;
    auto iter          = 0;
    auto rmsd          = 1.0;
    auto ediff         = 0.0;
    auto ehf           = 0.0;
    auto is_conv       = true;
    //  Matrix C;
    //  Matrix F;
    
    // double alpha = 0.75;
    // Matrix F_old;

    int idiis                     = 0;
    std::vector<tamm::Tensor<TensorType>> diis_hist;
    std::vector<tamm::Tensor<TensorType>> fock_hist;

    Tensor<TensorType> ehf_tmp{tAO, tAO};
    Tensor<TensorType> ehf_tamm{}, rmsd_tamm{};

    Tensor<TensorType> F1{tAO, tAO};
    Tensor<TensorType> F1tmp1{tAO, tAO};
    Tensor<TensorType> F1tmp{tAOt, tAOt}; //not allocated
    Tensor<TensorType>::allocate(ec, F1, F1tmp1, ehf_tmp, ehf_tamm, rmsd_tamm);

    Tensor<TensorType> Sm12_tamm{tAO, tAO}; 
    Tensor<TensorType> Sp12_tamm{tAO, tAO};
    Tensor<TensorType> D_tamm{tAO, tAO};
    Tensor<TensorType> D_last_tamm{tAO, tAO};
    Tensor<TensorType>::allocate(ec, Sm12_tamm, Sp12_tamm, D_tamm, D_last_tamm);

    // FSm12,Sp12D,SpFS
    Tensor<TensorType> FSm12_tamm{tAO, tAO}; 
    Tensor<TensorType> Sp12D_tamm{tAO, tAO};
    Tensor<TensorType> SpFS_tamm{tAO, tAO};
    // Tensor<TensorType> err_mat_tamm{tAO, tAO};

    Tensor<TensorType>::allocate(ec, FSm12_tamm, Sp12D_tamm, SpFS_tamm);//,err_mat_tamm);

    hf_t2 = std::chrono::high_resolution_clock::now();
    hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
    //GA_Sync();
    if(rank == 0 && debug) std::cout << "\nTime taken to setup tensors for iterative loop: " << hf_time << " secs\n";

    eigen_to_tamm_tensor(D_tamm,D);
    F.setZero(N,N);
    // Matrix err_mat = Matrix::Zero(N,N);

    hf_t1 = std::chrono::high_resolution_clock::now();
    // S^1/2
    Matrix Sp12 = S.sqrt();
    S.resize(0,0);
    // S^-1/2
    Matrix Sm12 = Sp12.inverse();
    hf_t2 = std::chrono::high_resolution_clock::now();
    hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
    if(rank == 0) std::cout << "\nTime taken to compute S^1/2, S^-1/2: " << hf_time << " secs\n";

    eigen_to_tamm_tensor(Sm12_tamm,Sm12);
    eigen_to_tamm_tensor(Sp12_tamm,Sp12);   
    Sp12.resize(0,0);
    Sm12.resize(0,0);

    if(rank == 0) {
        std::cout << "\n\n";
        std::cout << " Hartree-Fock iterations" << std::endl;
        std::cout << std::string(70, '-') << std::endl;
        std::cout <<
            " Iter     Energy            E-Diff            RMSD            Time" 
                << std::endl;
        std::cout << std::string(70, '-') << std::endl;
    }

    std::cout << std::fixed << std::setprecision(2);

    double precision = tol_int;
    const libint2::BasisSet& obs = shells;
    // assert(N==obs.nbf());
    Matrix G = Matrix::Zero(N,N);
    const auto do_schwarz_screen = SchwarzK.cols() != 0 && SchwarzK.rows() != 0;
    auto fock_precision = precision;
    // engine precision controls primitive truncation, assume worst-case scenario
    // (all primitive combinations add up constructively)
    auto max_nprim = obs.max_nprim();
    auto max_nprim4 = max_nprim * max_nprim * max_nprim * max_nprim;
    auto shell2bf = obs.shell2bf();
    // const auto nshells = obs.size();

    do {
        // Scheduler sch{ec};
        const auto loop_start = std::chrono::high_resolution_clock::now();
        ++iter;

        // Save a copy of the energy and the density
        auto ehf_last = ehf;
        auto D_last   = D;

        Scheduler{*ec}
           (F1tmp1() = 0)
           (D_last_tamm(mu,nu) = D_tamm(mu,nu)).execute();

        // build a new Fock matrix
        // F           = H;

        hf_t1 = std::chrono::high_resolution_clock::now();
        
        // const auto precision_F = std::min(
        // std::min(1e-3 / XtX_condition_number, 1e-7),
        // std::max(rmsd / 1e4, std::numeric_limits<double>::epsilon()));

        // Matrix Ftmp = compute_2body_fock(shells, D, tol_int, SchwarzK);
        // eigen_to_tamm_tensor(F1tmp, Ftmp);

        G.setZero(N,N);
        Matrix D_shblk_norm =  compute_shellblock_norm(obs, D);  // matrix of infty-norms of shell blocks
      

      auto engine_precision = std::min(fock_precision / D_shblk_norm.maxCoeff(),
                                      std::numeric_limits<double>::epsilon()) /
                              max_nprim4;
      if(rank == 0)
        assert(engine_precision > max_engine_precision &&
          "using precomputed shell pair data limits the max engine precision"
      " ... make max_engine_precision smaller and recompile");

      // construct the 2-electron repulsion integrals engine pool
      using libint2::Engine;
      Engine engine(Operator::coulomb, obs.max_nprim(), obs.max_l(), 0);

      engine.set_precision(engine_precision);
      // std::atomic<size_t> num_ints_computed{0};
      const auto& buf = engine.results();
      

#if 1
    auto comp_2bf_lambda = [&](IndexVector blockid) {

      // const IndexVector blockid = 
      //     internal::translate_blockid(blockid1, F1tmp());
 
        auto s1 = blockid[0];
        auto bf1_first = shell2bf[s1]; 
        auto n1 = obs[s1].size();

        auto sp12_iter = obs_shellpair_data.at(s1).begin();

        //for (const auto& s2 : obs_shellpair_list[s1]) {
        auto s2 = blockid[1];
        
        auto s2spl = obs_shellpair_list[s1];
        auto s2_itr = std::find(s2spl.begin(),s2spl.end(),s2);
        if(s2_itr == s2spl.end()) return;
        auto s2_pos = std::distance(s2spl.begin(),s2_itr);

        auto bf2_first = shell2bf[s2];
        auto n2 = obs[s2].size();

        std::advance(sp12_iter,s2_pos);
        const auto* sp12 = sp12_iter->get();
        //++sp12_iter;

        const auto Dnorm12 = do_schwarz_screen ? D_shblk_norm(s1, s2) : 0.;

        for (auto s3 = 0; s3 <= s1; ++s3) {
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
                Dnorm1234 * SchwarzK(s1, s2) * SchwarzK(s3, s4) <
                    fock_precision)
              continue;

            auto bf4_first = shell2bf[s4];
            auto n4 = obs[s4].size();

            // num_ints_computed += n1 * n2 * n3 * n4;

            // compute the permutational degeneracy (i.e. # of equivalents) of
            // the given shell set
            auto s12_deg = (s1 == s2) ? 1 : 2;
            auto s34_deg = (s3 == s4) ? 1 : 2;
            auto s12_34_deg = (s1 == s3) ? (s2 == s4 ? 1 : 2) : 2;
            auto s1234_deg = s12_deg * s34_deg * s12_34_deg;

            engine.compute2<Operator::coulomb, libint2::BraKet::xx_xx, 0>(
              obs[s1], obs[s2], obs[s3], obs[s4], sp12, sp34); 
            // engine.compute(obs[s1], obs[s2], obs[s3], obs[s4]);
                
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
            for (auto f1 = 0, f1234 = 0; f1 != n1; ++f1) {
              const auto bf1 = f1 + bf1_first;
              for (auto f2 = 0; f2 != n2; ++f2) {
                const auto bf2 = f2 + bf2_first;
                for (auto f3 = 0; f3 != n3; ++f3) {
                  const auto bf3 = f3 + bf3_first;
                  for (auto f4 = 0; f4 != n4; ++f4, ++f1234) {
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

      Matrix Gt = 0.5*(G + G.transpose());
      G = Gt;
    };

    block_for(*ec, F1tmp(), comp_2bf_lambda);
    // GA_Sync();
    eigen_to_tamm_tensor_acc(F1tmp1,G);
#endif

        hf_t2 = std::chrono::high_resolution_clock::now();
        hf_time =
        std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

        if(rank == 0 && debug) std::cout << "2BF:" << hf_time << "s, ";

        hf_t1 = std::chrono::high_resolution_clock::now();
        // F += Ftmp;
        Scheduler{*ec}(F1(mu, nu) = H1(mu, nu))
                     (F1(mu, nu) += F1tmp1(mu, nu)).execute();

        hf_t2 = std::chrono::high_resolution_clock::now();
        hf_time =
        std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

        if(rank == 0 && debug) std::cout << "F=H+2BF:" << hf_time << "s, ";

        // tamm_to_eigen_tensor(F1,F);
        // Matrix FSm12 = F * Sm12;
        // Matrix Sp12D = Sp12 * D_last;
        // Matrix SpFS  = Sp12D * FSm12;

        // // Assemble: S^(-1/2)*F*D*S^(1/2) - S^(1/2)*D*F*S^(-1/2)
        // err_mat = SpFS.transpose() - SpFS;    

        // eigen_to_tamm_tensor(D_last_tamm,D_last);

        Tensor<TensorType> err_mat_tamm{tAO, tAO};
        Tensor<TensorType>::allocate(ec, err_mat_tamm);

        hf_t1 = std::chrono::high_resolution_clock::now();
        
        Scheduler{*ec}(FSm12_tamm() = 0)(FSm12_tamm(mu,nu) += F1(mu,ku) * Sm12_tamm(ku,nu))
        (Sp12D_tamm() = 0)(Sp12D_tamm(mu,nu) += Sp12_tamm(mu,ku) * D_last_tamm(ku,nu))
        (SpFS_tamm() = 0)(SpFS_tamm(mu,nu)  += Sp12D_tamm(mu,ku) * FSm12_tamm(ku,nu))
    
        (err_mat_tamm(mu,nu) = SpFS_tamm(nu,mu))
        (err_mat_tamm(mu,nu) += -1.0 * SpFS_tamm(mu,nu)).execute();

        hf_t2 = std::chrono::high_resolution_clock::now();
        hf_time =
        std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

        if(rank == 0 && debug) std::cout << "err_mat:" << hf_time << "s, ";        

        GA_Sync();
        //tamm_to_eigen_tensor(err_mat_tamm,err_mat);
        //tamm_to_eigen_tensor(F1,F);

        hf_t1 = std::chrono::high_resolution_clock::now();

        if(iter > 1) {
            ++idiis;
            diis(*ec, tAO, F1, err_mat_tamm, iter, max_hist, idiis,
                diis_hist, fock_hist);
        }
    
        hf_t2 = std::chrono::high_resolution_clock::now();
        hf_time =
        std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

        if(rank == 0 && debug) std::cout << "diis:" << hf_time << "s, ";    
        tamm_to_eigen_tensor(F1,F);

        hf_t1 = std::chrono::high_resolution_clock::now();
        // solve F C = e S C
        // Eigen::GeneralizedSelfAdjointEigenSolver<Matrix>
        // gen_eig_solver(F, S);
        // // auto
        // // eps = gen_eig_solver.eigenvalues();
        // C   = gen_eig_solver.eigenvectors();
        // // auto C1 = gen_eig_solver.eigenvectors();

        // solve F C = e S C by (conditioned) transformation to F' C' = e C',
        // where
        // F' = X.transpose() . F . X; the original C is obtained as C = X . C'
        Eigen::SelfAdjointEigenSolver<Matrix> eig_solver(X.transpose() * F *
                                                        X);
        //eps = eig_solver.eigenvalues();
        C = X * eig_solver.eigenvectors();

        // compute density, D = C(occ) . C(occ)T
        auto C_occ = C.leftCols(ndocc);
        D          = C_occ * C_occ.transpose();

        hf_t2 = std::chrono::high_resolution_clock::now();
        hf_time =
        std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

        if(rank == 0 && debug) std::cout << "eigen_solve:" << hf_time << "s, ";    

        hf_t1 = std::chrono::high_resolution_clock::now();

        eigen_to_tamm_tensor(D_tamm,D);
        // eigen_to_tamm_tensor(F1,F);

        hf_t2 = std::chrono::high_resolution_clock::now();
        hf_time =
        std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

        if(rank == 0 && debug) std::cout << "E2T-D,F:" << hf_time << "s, ";    

        hf_t1 = std::chrono::high_resolution_clock::now();
        // compute HF energy 
        // e = D * (H+F);
        Scheduler{*ec}(ehf_tamm()=0)
           (ehf_tmp(mu,nu) = H1(mu,nu))
           (ehf_tmp(mu,nu) += F1(mu,nu))
           (ehf_tamm() += D_tamm() * ehf_tmp()).execute();

        ehf = get_scalar(ehf_tamm);

        // compute difference with last iteration
        ediff = ehf - ehf_last;
        rmsd  = (D - D_last).norm();
        //    (rmsd_tamm() = rmsd).execute();

        hf_t2 = std::chrono::high_resolution_clock::now();
        hf_time =
        std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

        if(rank == 0 && debug) std::cout << "HF-Energy:" << hf_time << "s\n";    

        const auto loop_stop = std::chrono::high_resolution_clock::now();
        const auto loop_time =
        std::chrono::duration_cast<std::chrono::duration<double>>((loop_stop - loop_start)).count();

        if(rank == 0) {
            std::cout << std::setw(5) << iter << "  " << std::setw(14);
            std::cout << std::fixed << std::setprecision(10) << ehf + enuc;
            std::cout << ' ' << std::setw(16)  << ediff;
            std::cout << ' ' << std::setw(15)  << rmsd << ' ';
            std::cout << std::fixed << std::setprecision(2);
            std::cout << ' ' << std::setw(12)  << loop_time << ' ' << "\n";
        }

        if(iter > maxiter) {                
            is_conv = false;
            break;
        }

    } while (((fabs(ediff) > conve) || (fabs(rmsd) > convd)));

    // GA_Sync(); 
    if(rank == 0) {
        std::cout.precision(13);
        if (is_conv)
            cout << "\n** Hartree-Fock energy = " << ehf + enuc << endl;
        else {
            cout << endl << std::string(50, '*') << endl;
            cout << std::string(10, ' ') << 
                    "ERROR: HF Does not converge!!!\n";
            cout << std::string(50, '*') << endl;
        }        
    }

    for (auto x: diis_hist) Tensor<TensorType>::deallocate(x);
    for (auto x: fock_hist) Tensor<TensorType>::deallocate(x);

    Tensor<TensorType>::deallocate(H1, F1, D_tamm, ehf_tmp, 
                    ehf_tamm, rmsd_tamm);
    Tensor<TensorType>::deallocate(F1tmp1,Sm12_tamm, Sp12_tamm,
    D_last_tamm,FSm12_tamm, Sp12D_tamm, SpFS_tamm);//,err_mat_tamm);

    ec->flush_and_sync();
    MemoryManagerGA::destroy_coll(mgr);
    delete ec;

    return std::make_tuple(ndocc, nao, ehf + enuc, shells);
}

void diis(ExecutionContext& ec, TiledIndexSpace& tAO, tamm::Tensor<TensorType> F, Tensor<TensorType> err_mat, int iter, int max_hist, int ndiis,
          std::vector<Tensor<TensorType>>& diis_hist, std::vector<tamm::Tensor<TensorType>>& fock_hist) {
    using Vector =
      Eigen::Matrix<TensorType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  tamm::Scheduler sch{ec};

  if(ndiis > max_hist) {
    std::vector<TensorType> max_err(diis_hist.size());
    for (auto i=0; i<diis_hist.size(); i++) {
      max_err[i] = tamm::norm(ec, diis_hist[i]());
      // Matrix dhist = Matrix::Zero(F.rows(),F.cols());
      // tamm_to_eigen_tensor(diis_hist[i],dhist);
      // cout << "tamm norm, eigen norm = " << max_err[i] << " , " << dhist.norm() << endl;
    }

    auto maxe = std::distance(max_err.begin(), 
              std::max_element(max_err.begin(),max_err.end()));

    diis_hist.erase(diis_hist.begin()+maxe);
    fock_hist.erase(fock_hist.begin()+maxe);      
  }

    Tensor<TensorType> Fcopy{tAO, tAO};
    Tensor<TensorType>::allocate(&ec,Fcopy);
    sch(Fcopy() = F()).execute();

    diis_hist.push_back(err_mat);
    fock_hist.push_back(Fcopy);

    // ----- Construct error metric -----
    const int idim = std::min(ndiis, max_hist);
    Matrix A       = Matrix::Zero(idim + 1, idim + 1);
    Vector b       = Vector::Zero(idim + 1, 1);

    Tensor<TensorType> dhi_trans{tAO, tAO};
    Tensor<TensorType> dhi_trace{};
    auto [mu, nu, ku] = tAO.labels<3>("all");
    Tensor<TensorType>::allocate(&ec, dhi_trans,dhi_trace);

    for(int i = 0; i < idim; i++) {
        for(int j = i; j < idim; j++) {
            //A(i, j) = (diis_hist[i].transpose() * diis_hist[j]).trace();
            sch(dhi_trace() = diis_hist[i](nu,mu) * diis_hist[j](mu,nu)).execute();
            A(i,j) = get_scalar(dhi_trace); //dhi_trace.trace();
        }
    }

    Tensor<TensorType>::deallocate(dhi_trans,dhi_trace);

    for(int i = 0; i < idim; i++) {
        for(int j = i; j < idim; j++) { A(j, i) = A(i, j); }
    }
    for(int i = 0; i < idim; i++) {
        A(i, idim) = -1.0;
        A(idim, i) = -1.0;
    }

    b(idim, 0) = -1;

    Vector x = A.lu().solve(b);

    // F.setZero();
    // for(int j = 0; j < idim; j++) { F += x(j, 0) * fock_hist[j]; }
    
    sch(F() = 0).execute();
    for(int j = 0; j < idim; j++) { 
      sch(F() += x(j, 0) * fock_hist[j]()); 
    }
    sch.execute();

    // cout << "-----------iter:" << iter << "--------------\n";
    // cout << err_mat << endl;
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

  if(GA_Nodeid()==0)
    std::cout << "computing Schwarz bound prerequisites (kernel=" 
          << (int)Kernel << ") ... ";

    libint2::Timers<1> timer;
    timer.set_now_overhead(25);
    timer.start(0);
  
    const auto& buf = engine.results();

    // loop over permutationally-unique set of shells
    for (auto s1 = 0l, s12 = 0l; s1 != nsh1; ++s1) {
      auto n1 = bs1[s1].size();  // number of basis functions in this shell

      auto s2_max = bs1_equiv_bs2 ? s1 : nsh2 - 1;
      for (auto s2 = 0; s2 <= s2_max; ++s2, ++s12) {
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
    std::cout << "done (" << timer.read(0) << " s)" << std::endl;
 
  return K;
}

Matrix compute_shellblock_norm(const libint2::BasisSet& obs, const Matrix& A) {
  const auto nsh = obs.size();
  Matrix Ash(nsh, nsh);

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
  // std::cout << "compute_2body_fock:precision = " << precision << std::endl;
  // std::cout << "Engine::precision = " << engine.precision() << std::endl;
  // for (size_t i = 1; i != nthreads; ++i) {
  //   engines[i] = engines[0];
  // }
  std::atomic<size_t> num_ints_computed{0};


  auto shell2bf = obs.shell2bf();

    const auto& buf = engine.results();

    // loop over permutationally-unique set of shells
    for (auto s1 = 0l, s1234 = 0l; s1 != nshells; ++s1) {
      auto bf1_first = shell2bf[s1];  // first basis function in this shell
      auto n1 = obs[s1].size();       // number of basis functions in this shell

      auto sp12_iter = obs_shellpair_data.at(s1).begin();

      for (const auto& s2 : obs_shellpair_list[s1]) {
        auto bf2_first = shell2bf[s2];
        auto n2 = obs[s2].size();

        const auto* sp12 = sp12_iter->get();
        ++sp12_iter;

        const auto Dnorm12 = do_schwarz_screen ? D_shblk_norm(s1, s2) : 0.;

        for (auto s3 = 0; s3 <= s1; ++s3) {
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
            for (auto f1 = 0, f1234 = 0; f1 != n1; ++f1) {
              const auto bf1 = f1 + bf1_first;
              for (auto f2 = 0; f2 != n2; ++f2) {
                const auto bf2 = f2 + bf2_first;
                for (auto f3 = 0; f3 != n3; ++f3) {
                  const auto bf3 = f3 + bf3_first;
                  for (auto f4 = 0; f4 != n4; ++f4, ++f1234) {
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

  // std::cout << "# of integrals = " << num_ints_computed << std::endl;
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
  assert(D.cols() == D.rows() && D.cols() == n_D);

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
    for (auto s1 = 0l, s1234 = 0l; s1 != nshells; ++s1) {
      auto bf1_first = shell2bf[s1];  // first basis function in this shell
      auto n1 = obs[s1].size();       // number of basis functions in this shell

      for (auto s2 = 0; s2 <= s1; ++s2) {
        auto bf2_first = shell2bf[s2];
        auto n2 = obs[s2].size();

        for (auto s3 = 0; s3 < D_bs.size(); ++s3) {
          auto bf3_first = shell2bf_D[s3];
          auto n3 = D_bs[s3].size();

          auto s4_begin = D_is_shelldiagonal ? s3 : 0;
          auto s4_fence = D_is_shelldiagonal ? s3 + 1 : D_bs.size();

          for (auto s4 = s4_begin; s4 != s4_fence; ++s4, ++s1234) {
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
                for (auto f1 = 0, f1234 = 0; f1 != n1; ++f1) {
                  const auto bf1 = f1 + bf1_first;
                  for (auto f2 = 0; f2 != n2; ++f2) {
                    const auto bf2 = f2 + bf2_first;
                    for (auto f3 = 0; f3 != n3; ++f3) {
                      const auto bf3 = f3 + bf3_first;
                      for (auto f4 = 0; f4 != n4; ++f4, ++f1234) {
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

            for (auto f1 = 0, f1324 = 0; f1 != n1; ++f1) {
              const auto bf1 = f1 + bf1_first;
              for (auto f3 = 0; f3 != n3; ++f3) {
                const auto bf3 = f3 + bf3_first;
                for (auto f2 = 0; f2 != n2; ++f2) {
                  const auto bf2 = f2 + bf2_first;
                  for (auto f4 = 0; f4 != n4; ++f4, ++f1324) {
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

#endif // TAMM_TESTS_HF_TAMM_HPP_
