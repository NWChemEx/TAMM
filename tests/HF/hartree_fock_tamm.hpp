
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

// Eigen matrix algebra library
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/MatrixFunctions>

// Libint Gaussian integrals library
#include <libint2.hpp>
#include <libint2/basis.h>

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

// import dense, dynamically sized Matrix type from Eigen;
// this is a matrix with row-major storage
// (http://en.wikipedia.org/wiki/Row-major_order) to meet the layout of the
// integrals returned by the Libint integral library

Matrix compute_soad(const std::vector<libint2::Atom>& atoms);
void diis(Matrix& F, Matrix& S, Matrix& D_last, int iter, int max_hist,
          int idiis, std::vector<Matrix>& diis_hist,
          std::vector<Matrix>& fock_hist);

// an efficient Fock builder; *integral-driven* hence computes
// permutationally-unique ints once
Matrix compute_2body_fock(const std::vector<libint2::Shell>& shells,
                          const Matrix& D);

template<typename T>
std::ostream& operator<<(std::ostream& os, std::vector<T>& vec) {
    os << "[";
    for(auto& x : vec) os << x << ",";
    os << "]\n";
    return os;
}

template<typename T>
void print_tensor(Tensor<T>& t) {
    auto lt = t();
    for(auto it : t.loop_nest()) {
        auto blockid   = internal::translate_blockid(it, lt);
        TAMM_SIZE size = t.block_size(blockid);
        std::vector<T> buf(size);
        t.get(blockid, buf);
        std::cout << "block = " << blockid;
        // std::cout << "size= " << size << endl;
        for(TAMM_SIZE i = 0; i < size; i++) std::cout << buf[i] << " ";
        std::cout << endl;
    }
}

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
    for(auto s1 = 0; s1 != shells.size(); ++s1) {
        auto bf1_first = shell2bf[s1]; // first basis function in this shell
        auto n1        = shells[s1].size();
        for(auto f1 = 0; f1 != n1; ++f1) {
            const auto bf1 = f1 + bf1_first;
            result[bf1]    = s1;
        }
    }
    return result;
}

using libint2::Atom;

inline std::tuple<std::vector<Atom>, std::string> read_input_xyz(
  std::istream& is) {
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
    while(std::getline(is, basis_set)) {
        if(basis_set.empty())
            continue;
        else {
            std::istringstream bss(basis_set);
            std::vector<std::string> basis_string{
              std::istream_iterator<std::string>{bss},
              std::istream_iterator<std::string>{}};
            assert(basis_string.size() == 2);
            assert(basis_string[0] == "basis");
            basis_set = basis_string[1];
            // cout << basis_set << endl;
            break;
        }
    }

    return std::make_tuple(atoms, basis_set);
}

Matrix compute_1body_ints(
  const std::vector<libint2::Shell>& shells, libint2::Operator obtype,
  const std::vector<libint2::Atom>& atoms = std::vector<libint2::Atom>()) {
    using libint2::Engine;
    using libint2::Operator;
    using libint2::Shell;

    const auto n = nbasis(shells);
    Matrix result(n, n);

    // construct the overlap integrals engine
    Engine engine(obtype, max_nprim(shells), max_l(shells), 0);
    // nuclear attraction ints engine needs to know where the charges sit ...
    // the nuclei are charges in this case; in QM/MM there will also be
    // classical charges
    if(obtype == Operator::nuclear) {
        std::vector<std::pair<double, std::array<double, 3>>> q;
        for(const auto& atom : atoms) {
            q.push_back({static_cast<double>(atom.atomic_number),
                         {{atom.x, atom.y, atom.z}}});
        }
        engine.set_params(q);
    }

    auto shell2bf = map_shell_to_basis_function(shells);

    // buf[0] points to the target shell set after every call  to
    // engine.compute()
    const auto& buf = engine.results();

    // loop over unique shell pairs, {s1,s2} such that s1 >= s2
    // this is due to the permutational symmetry of the real integrals over
    // Hermitian operators: (1|2) = (2|1)
    for(size_t s1 = 0; s1 != shells.size(); ++s1) {
        auto bf1 = shell2bf[s1]; // first basis function in this shell
        auto n1  = shells[s1].size();

        for(size_t s2 = 0; s2 <= s1; ++s2) {
            auto bf2 = shell2bf[s2];
            auto n2  = shells[s2].size();

            // compute shell pair; return is the pointer to the buffer
            engine.compute(shells[s1], shells[s2]);

            // "map" buffer to a const Eigen Matrix, and copy it to the
            // corresponding blocks of the result
            Eigen::Map<const Matrix> buf_mat(buf[0], n1, n2);
            result.block(bf1, bf2, n1, n2) = buf_mat;
            if(s1 != s2) // if s1 >= s2, copy {s1,s2} to the corresponding
                         // {s2,s1} block, note the transpose!
                result.block(bf2, bf1, n2, n1) = buf_mat.transpose();
        }
    }

    return result;
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

std::tuple<int, int, double, libint2::BasisSet> hartree_fock(
  const string filename, Matrix& C, Matrix& F) {
    // Perform the simple HF calculation (Ed) and 2,4-index transform to get the
    // inputs for CCSD
    using libint2::Atom;
    using libint2::Engine;
    using libint2::Operator;
    using libint2::Shell;

    /*** =========================== ***/
    /*** initialize molecule         ***/
    /*** =========================== ***/

    // read geometry from a file; by default read from h2o.xyz, else take
    // filename (.xyz) from the command line
    auto is = std::ifstream(filename);
    std::vector<Atom> atoms;
    std::string basis;
    std::tie(atoms, basis) = read_input_xyz(is);

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
    cout << "\tNuclear repulsion energy = " << enuc << endl;

    // initializes the Libint integrals library ... now ready to compute
    libint2::initialize();

    /*** =========================== ***/
    /*** create basis set            ***/
    /*** =========================== ***/

    // LIBINT_INSTALL_DIR/share/libint/2.4.0-beta.1/basis
    libint2::BasisSet shells(std::string(basis), atoms);
    // auto shells = make_sto3g_basis(atoms);
    size_t nao = 0;

    for(size_t s = 0; s < shells.size(); ++s) nao += shells[s].size();

    const size_t N = nbasis(shells);
    assert(N == nao);

    /*** =========================== ***/
    /*** compute 1-e integrals       ***/
    /*** =========================== ***/

    // compute overlap integrals
    // auto S = compute_1body_ints(shells, Operator::overlap);

    // // compute kinetic-energy integrals
    // auto T = compute_1body_ints(shells, Operator::kinetic);

    // // compute nuclear-attraction integrals
    // Matrix V = compute_1body_ints(shells, Operator::nuclear, atoms);

    // // Core Hamiltonian = T + V
    // Matrix H = T + V;

    // // T and V no longer needed, free up the memory
    // T.resize(0, 0);
    // V.resize(0, 0);

    auto one_body_overlap_integral_lambda = [&](const IndexVector& blockid,
                                                span<TensorType> tbuf) {
        // construct the overlap integrals engine
        Engine engine(Operator::overlap, max_nprim(shells), max_l(shells), 0);

        // buf[0] points to the target shell set after every call  to
        // engine.compute()
        std::vector<std::pair<double, std::array<double, 3>>> q;
        for(const auto& atom : atoms) {
            q.push_back({static_cast<double>(atom.atomic_number),
                         {{atom.x, atom.y, atom.z}}});
        }
        engine.set_params(q);

        auto shell2bf   = map_shell_to_basis_function(shells);
        const auto& buf = engine.results();

        auto s1 = blockid[0];
        // auto bf1 = shell2bf[s1]; //shell2bf[s1]; // first basis function in
        // this shell
        auto n1 = shells[s1].size();

        // for (size_t s2 = 0; s2 <= s1; ++s2) {
        auto s2 = blockid[1];
        // if(s1<s2) return; //TODO
        // auto bf2 = shell2bf[s2];
        auto n2 = shells[s2].size();

        // compute shell pair; return is the pointer to the buffer
        engine.compute(shells[s1], shells[s2]);
        // "map" buffer to a const Eigen Matrix, and copy it to the
        // corresponding blocks of the result
        Eigen::Map<const Matrix> buf_mat(buf[0], n1, n2);
        // result.block(bf1, bf2, n1, n2) = buf_mat;
        for(size_t i = 0; i < n1; i++)
            for(size_t j = 0; j < n2; j++) tbuf[i * n2 + j] = buf_mat(i, j);
    };

    auto one_body_nuclear_integral_lambda = [&](const IndexVector& blockid,
                                                span<TensorType> tbuf) {
        // construct the overlap integrals engine
        Engine engine(Operator::nuclear, max_nprim(shells), max_l(shells), 0);

        // buf[0] points to the target shell set after every call  to
        // engine.compute()
        std::vector<std::pair<double, std::array<double, 3>>> q;
        for(const auto& atom : atoms) {
            q.push_back({static_cast<double>(atom.atomic_number),
                         {{atom.x, atom.y, atom.z}}});
        }
        engine.set_params(q);

        auto shell2bf   = map_shell_to_basis_function(shells);
        const auto& buf = engine.results();

        auto s1 = blockid[0];
        // auto bf1 = shell2bf[s1]; //shell2bf[s1]; // first basis function in
        // this shell
        auto n1 = shells[s1].size();

        // for (size_t s2 = 0; s2 <= s1; ++s2) {
        auto s2 = blockid[1];
        // if(s1<s2) return; //TODO
        // auto bf2 = shell2bf[s2];
        auto n2 = shells[s2].size();

        // compute shell pair; return is the pointer to the buffer
        engine.compute(shells[s1], shells[s2]);
        // "map" buffer to a const Eigen Matrix, and copy it to the
        // corresponding blocks of the result
        Eigen::Map<const Matrix> buf_mat(buf[0], n1, n2);
        // result.block(bf1, bf2, n1, n2) = buf_mat;
        for(size_t i = 0; i < n1; i++)
            for(size_t j = 0; j < n2; j++) tbuf[i * n2 + j] = buf_mat(i, j);
    };

    auto one_body_kinetic_integral_lambda = [&](const IndexVector& blockid,
                                                span<TensorType> tbuf) {
        // construct the overlap integrals engine
        Engine engine(Operator::kinetic, max_nprim(shells), max_l(shells), 0);
        auto shell2bf = map_shell_to_basis_function(shells);

        // buf[0] points to the target shell set after every call  to
        // engine.compute()
        const auto& buf = engine.results();

        // loop over unique shell pairs, {s1,s2} such that s1 >= s2
        // this is due to the permutational symmetry of the real integrals over
        // Hermitian operators: (1|2) = (2|1) for (size_t s1 = 0; s1 !=
        // shells.size(); ++s1) {

        auto s1 = blockid[0];
        // auto bf1 = shell2bf[s1]; //shell2bf[s1]; // first basis function in
        // this shell
        auto n1 = shells[s1].size();

        // for (size_t s2 = 0; s2 <= s1; ++s2) {
        auto s2 = blockid[1];
        // if(s1<s2) return; //TODO
        // auto bf2 = shell2bf[s2];
        auto n2 = shells[s2].size();

        // compute shell pair; return is the pointer to the buffer
        engine.compute(shells[s1], shells[s2]);
        // "map" buffer to a const Eigen Matrix, and copy it to the
        // corresponding blocks of the result
        Eigen::Map<const Matrix> buf_mat(buf[0], n1, n2);
        // result.block(bf1, bf2, n1, n2) = buf_mat;
        for(size_t i = 0; i < n1; i++)
            for(size_t j = 0; j < n2; j++) tbuf[i * n2 + j] = buf_mat(i, j);

        // cout << "s1,s2 == " << s1 << "," << s2 << endl;
        // cout << "n1,n2 == " << n1 << "," << n2 << endl;
        // cout << "bufs, tbufs == " << buf.size() << "," << tbuf.size() <<
        // endl; cout << "bufmat\n----------\n"; cout << buf_mat << endl;

        // assert(n1*n2 == tbuf.size());

        // cout << "buf\n----------\n";
        //                 for (size_t i = 0; i < n1; i++)
        //                 for (size_t j = 0; j < n2; j++)
        //                 cout << tmp[i*n2+j] << endl;

        // TODO
        // if (s1 != s2) // if s1 >= s2, copy {s1,s2} to the corresponding
        // {s2,s1} block, note the transpose!
        //   {  // result.block(bf2, bf1, n2, n1) = buf_mat.transpose();
        //     for (size_t i = 0; i < n1; i++) {
        //       for (size_t j = 0; j < n2; j++) {
        //         tbuf[i * n2 + j] = tmp[j*n2+i];
        //       }
        //     }
        //   }

        // }
        // }
    };

    ProcGroup pg{GA_MPI_Comm()};
    auto mgr = MemoryManagerGA::create_coll(pg);
    Distribution_NW distribution;
    ExecutionContext* ec = new ExecutionContext{pg, &distribution, mgr};
    Scheduler sch{ec};

    IndexSpace AO{range(0, N)};
    std::vector<unsigned int> AO_tiles;
    for(auto s : shells) AO_tiles.push_back(s.size());
    TiledIndexSpace tAO{AO, AO_tiles};
    auto [mu, nu] = tAO.labels<2>("all");

    Tensor<TensorType> S1{{tAO, tAO}, one_body_overlap_integral_lambda};
    Tensor<TensorType> T1{{tAO, tAO}, one_body_kinetic_integral_lambda};
    Tensor<TensorType> V1{{tAO, tAO}, one_body_nuclear_integral_lambda};

    Tensor<TensorType> H1{tAO, tAO};
    // Tensor<TensorType> T2{tAO,tAO};
    Tensor<TensorType>::allocate(ec, H1);

    sch(H1(mu, nu) = 0)(H1(mu, nu) += T1(mu, nu))(H1(mu, nu) += V1(mu, nu))
      .execute();

    // cout << "----------------tamm H-----------------\n";
    // cout << eigen_tensor << endl;
    // cout << "----------------orig H-----------------\n";
    // cout << H << endl;

    Matrix H, S, T, V;
    H.setZero(N, N);
    V.setZero(N, N);
    S.setZero(N, N);
    T.setZero(N, N);

    tamm_to_eigen_tensor(H1, H);
    tamm_to_eigen_tensor(T1, T);
    tamm_to_eigen_tensor(S1, S);
    tamm_to_eigen_tensor(V1, V);

    H = T + V;
    cout << "For H: ";
    compare_eigen_tamm_tensors(H1, H);

    /*** =========================== ***/
    /*** build initial-guess density ***/
    /*** =========================== ***/

    const auto use_hcore_guess =
      true; // use core Hamiltonian eigenstates to guess density?
    // set to true to match the result of versions 0, 1, and 2 of the code
    // HOWEVER !!! even for medium-size molecules hcore will usually fail !!!
    // thus set to false to use Superposition-Of-Atomic-Densities (SOAD) guess
    Matrix D;
    if(use_hcore_guess) { // hcore guess
        // solve H C = e S C
        Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> gen_eig_solver(H, S);
        auto eps = gen_eig_solver.eigenvalues();
        auto C   = gen_eig_solver.eigenvectors();
        //    cout << "\n\tInitial C Matrix:\n";
        //    cout << C << endl;

        // compute density, D = C(occ) . C(occ)T
        auto C_occ = C.leftCols(ndocc);
        D          = C_occ * C_occ.transpose();
    } else { // SOAD as the guess density, assumes STO-nG basis
        D = compute_soad(atoms);
    }

    //  cout << "\n\tInitial Density Matrix:\n";
    //  cout << D << endl;

    /*** =========================== ***/
    /*** main iterative loop         ***/
    /*** =========================== ***/

    const auto maxiter = 100;
    const auto conv    = 1e-7;
    auto iter          = 0;
    auto rmsd          = 0.0;
    auto ediff         = 0.0;
    auto ehf           = 0.0;
    //  Matrix C;
    //  Matrix F;
    Matrix eps;
    double alpha = 0.75;
    Matrix F_old;

    const bool simple_convergence = false;
    int idiis                     = 0;
    int max_hist                  = 10;
    std::vector<Matrix> diis_hist;
    std::vector<Matrix> fock_hist;

    // Tensor<TensorType> ehf1{}, ediff1{}, rmsd1{};
    // Tensor<TensorType> eps1{tAO, tAO};

    Tensor<TensorType> F1{tAO, tAO};
    Tensor<TensorType> F1_old{tAO, tAO};
    Tensor<TensorType>::allocate(ec, F1, F1_old);

    Tensor<TensorType> F1tmp{tAO, tAO};
    Tensor<TensorType>::allocate(ec, F1tmp);

    do {
        const auto tstart = std::chrono::high_resolution_clock::now();
        ++iter;

        // Save a copy of the energy and the density
        auto ehf_last = ehf;
        auto D_last   = D;

        // build a new Fock matrix
        F           = H;
        Matrix Ftmp = compute_2body_fock(shells, D);

// TODO
#if 1
        //-------------------------COMPUTE 2 BODY FOCK USING
        // TAMM------------------
        auto comp_2bf_lambda =
          [&](IndexVector it) {
              Tensor<TensorType> tensor = F1tmp;
              const TAMM_SIZE size      = tensor.block_size(it);

              std::vector<TensorType> tbuf(size);

              auto block_dims   = tensor.block_dims(it);
              auto block_offset = tensor.block_offsets(it);

              using libint2::Shell;
              using libint2::Engine;
              using libint2::Operator;

              Matrix G = Matrix::Zero(N, N);
              auto ns  = shells.size();

              // construct the 2-electron repulsion integrals engine
              Engine engine(Operator::coulomb, max_nprim(shells), max_l(shells),
                            0);

              auto shell2bf = map_shell_to_basis_function(shells);
              auto bf2shell = map_basis_function_to_shell(shells);

              const auto& buf = engine.results();

              // loop over permutationally-unique set of shells

              for(auto i = block_offset[0];i < (block_offset[0] + block_dims[0]);i++){
                auto s1 = bf2shell[i];
                auto bf1_first = shell2bf[s1]; 
                auto n1        = shells[s1].size();

              for(auto j = block_offset[1];j < (block_offset[1] + block_dims[1]);j++){
                auto s2 = bf2shell[j];
                auto bf2_first = shell2bf[s2];
                auto n2        = shells[s2].size();
                
              for(size_t s3 = 0; s3 != shells.size(); ++s3) {
                auto bf3_first = shell2bf[s3];
                auto n3        = shells[s3].size();

              for(size_t s4 = 0; s4 != shells.size(); ++s4) {
                auto bf4_first = shell2bf[s4];
                auto n4        = shells[s4].size();

                auto f1 = i - bf1_first;
                auto f2 = j - bf2_first;
                auto s4_max = (s1 == s2) ? s3 : s2;
                auto s4p_max = (s1 == s2) ? s3 : s1;
                auto s1_max = (s4 == s2) ? s3 : s4;
                auto s2_max = (s4 == s1) ? s3 : s4;
                auto s2p_max = (s1 == s3) ? s4 : s1;
                auto s1p_max = (s3 == s2) ? s4 : s2;
                auto s4x_max = (s1 == s3) ? s2 : s3;
                auto s4px_max = (s2 == s3) ? s1 : s3;
                auto con1 = (s3<=s1 && s4 <= s4_max && s2<=s1);
                auto con2 = (s3<=s2 && s4 <= s4p_max && s1<=s2);
                auto con3 = (s3<=s2 && s1 <= s1_max && s4<=s2);
                auto con4 = (s3<=s1 && s2 <= s2_max && s4<=s1);
                auto conx = (s2 <= s2p_max && s3 >= s1 && s4 <= s3);
                auto cony = (s1 <=s1p_max && s3 >= s2 && s4<=s3);
                auto con0 = (s3<=s1 && s4 <=s4x_max && s2 <= s1);
                auto conz = (s3<=s2 && s4 <=s4px_max && s1 <= s2);
              
              if (con0){
                auto _s1 = s1;
                auto _s2 = s2;
                auto _s3 = s3;
                auto _s4 = s4;
                auto _n1 = n1;
                auto _n2 = n2;
                auto _n3 = n3;
                auto _n4 = n4;

                auto s12_deg = (_s1 == _s2) ? 1.0 : 2.0;
                auto s34_deg = (_s3 == _s4) ? 1.0 : 2.0;
                auto s12_34_deg = (_s1 == _s3) ? (_s2 == _s4 ? 1.0 : 2.0) : 2.0;
                auto s1234_deg = s12_deg * s34_deg * s12_34_deg;
                 
                engine.compute(shells[_s1], shells[_s2], shells[_s3], shells[_s4]);
                const auto* buf_1234 = buf[0];
                if(buf_1234 == nullptr) continue; 

                for(size_t f3 = 0; f3 != n3; ++f3) {
                  const auto bf3 = f3 + bf3_first;
                  for(size_t f4 = 0; f4 != n4; ++f4) {
                    const auto bf4 = f4 + bf4_first;
                    auto _f1 = f1;
                    auto _f2 = f2;
                    auto _f3 = f3;
	                auto _f4 = f4;
                    auto f1234 = _n4*(_n3*(_n2*_f1+_f2)+_f3)+_f4;
                    const auto value = buf_1234[f1234];
                    const auto value_scal_by_deg = value * s1234_deg;
                    G(i,j) += D(bf3, bf4) * value_scal_by_deg;
                  }
                }
              }
              
              if (conz){
                auto _s1 = s2;
                auto _s2 = s1;
                auto _s3 = s3;
                auto _s4 = s4;
                auto _n1 = n2;
                auto _n2 = n1;
                auto _n3 = n3;
                auto _n4 = n4;

                auto s12_deg = (_s1 == _s2) ? 1.0 : 2.0;
                auto s34_deg = (_s3 == _s4) ? 1.0 : 2.0;
                auto s12_34_deg = (_s1 == _s3) ? (_s2 == _s4 ? 1.0 : 2.0) : 2.0;
                auto s1234_deg = s12_deg * s34_deg * s12_34_deg;
                 
                engine.compute(shells[_s1], shells[_s2], shells[_s3], shells[_s4]);
                const auto* buf_1234 = buf[0];
                if(buf_1234 == nullptr) continue; 

                for(size_t f3 = 0; f3 != n3; ++f3) {
                  const auto bf3 = f3 + bf3_first;
                  for(size_t f4 = 0; f4 != n4; ++f4) {
                    const auto bf4 = f4 + bf4_first;
                    auto _f1 = f2;
                    auto _f2 = f1;
                    auto _f3 = f3;
	                auto _f4 = f4;
                    auto f1234 = _n4*(_n3*(_n2*_f1+_f2)+_f3)+_f4;
                    const auto value = buf_1234[f1234];
                    const auto value_scal_by_deg = value * s1234_deg;
                    G(i,j) += D(bf3, bf4) * value_scal_by_deg;
                  }
                }
              }

              if (conx){
	            auto _s1 = s3;
	            auto _s2 = s4;
                auto _s3 = s1;
                auto _s4 = s2;
                auto _n1 = n3;
                auto _n2 = n4;
                auto _n3 = n1;
                auto _n4 = n2;

                auto s12_deg = (_s3 == _s4) ? 1.0 : 2.0;
                auto s34_deg = (_s1 == _s2) ? 1.0 : 2.0;
                auto s12_34_deg = (_s3 == _s1) ? (_s4 == _s2 ? 1.0 : 2.0) : 2.0;
                auto s1234_deg = s12_deg * s34_deg * s12_34_deg;

                engine.compute(shells[_s1], shells[_s2], shells[_s3], shells[_s4]);
                const auto* buf_1234 = buf[0];
                if(buf_1234 == nullptr) continue; 

                for(size_t f3 = 0; f3 != n3; ++f3) {
                  const auto bf3 = f3 + bf3_first;
                  for(size_t f4 = 0; f4 != n4; ++f4) {
                    const auto bf4 = f4 + bf4_first;
                    auto _f1 = f3;
                    auto _f2 = f4;
                    auto _f3 = f1;
	                auto _f4 = f2;
                    auto f1234 = _n4*(_n3*(_n2*_f1+_f2)+_f3)+_f4;
                    const auto value = buf_1234[f1234];
                    const auto value_scal_by_deg = value * s1234_deg;
                    G(i, j) += D(bf3, bf4) * value_scal_by_deg;
                  }
                }
              }

              if (cony){
	            auto _s1 = s3;
	            auto _s2 = s4;
                auto _s3 = s2;
                auto _s4 = s1;
                auto _n1 = n3;
                auto _n2 = n4;
                auto _n3 = n2;
                auto _n4 = n1;

                auto s12_deg = (_s3 == _s4) ? 1.0 : 2.0;
                auto s34_deg = (_s1 == _s2) ? 1.0 : 2.0;
                auto s12_34_deg = (_s3 == _s1) ? (_s4 == _s2 ? 1.0 : 2.0) : 2.0;
                auto s1234_deg = s12_deg * s34_deg * s12_34_deg;

                engine.compute(shells[_s1], shells[_s2], shells[_s3], shells[_s4]);
                const auto* buf_1234 = buf[0];
                if(buf_1234 == nullptr) continue; 

                for(size_t f3 = 0; f3 != n3; ++f3) {
                  const auto bf3 = f3 + bf3_first;
                  for(size_t f4 = 0; f4 != n4; ++f4) {
                    const auto bf4 = f4 + bf4_first;
                    auto _f1 = f3;
                    auto _f2 = f4;
	                auto _f3 = f2;
                    auto _f4 = f1;
                    auto f1234 = _n4*(_n3*(_n2*_f1+_f2)+_f3)+_f4;
                    const auto value = buf_1234[f1234];
                    const auto value_scal_by_deg = value * s1234_deg;
                    G(i, j) += D(bf3, bf4) * value_scal_by_deg;
                  }
                }
              }

	          if (con1) {	
                auto _s1 = s1;
                auto _s2 = s3;
                auto _s3 = s2;
                auto _s4 = s4;
                auto _n1 = n1;
                auto _n2 = n3;
                auto _n3 = n2;
                auto _n4 = n4;

                auto s12_deg = (_s1 == _s2) ? 1.0 : 2.0;
                auto s34_deg = (_s3 == _s4) ? 1.0 : 2.0;
                auto s12_34_deg = (_s1 == _s3) ? (_s2 == _s4 ? 1.0 : 2.0) : 2.0;
                auto s1234_deg = s12_deg * s34_deg * s12_34_deg;

                engine.compute(shells[_s1], shells[_s2], shells[_s3], shells[_s4]);
                const auto* buf_1234 = buf[0];
                if(buf_1234 == nullptr) continue; 

                for(size_t f3 = 0; f3 != n3; ++f3) {
                  const auto bf3 = f3 + bf3_first;
                  for(size_t f4 = 0; f4 != n4; ++f4) {
                    const auto bf4 = f4 + bf4_first;
                    auto _f1 = f1;
                    auto _f2 = f3;
                    auto _f3 = f2;
	                auto _f4 = f4;
                    auto f1234 = _n4*(_n3*(_n2*_f1+_f2)+_f3)+_f4;
                    const auto value = buf_1234[f1234];
                    const auto value_scal_by_deg = value * s1234_deg;
                    G(i,j) -= 0.25 * D(bf3, bf4) * value_scal_by_deg;
                  }
                }
              }

	          if (con2) {	
                auto _s1 = s2;
                auto _s2 = s3;
                auto _s3 = s1;
                auto _s4 = s4;
                auto _n1 = n2;
                auto _n2 = n3;
                auto _n3 = n1;
                auto _n4 = n4;

                auto s12_deg = (_s1 == _s2) ? 1.0 : 2.0;
                auto s34_deg = (_s3 == _s4) ? 1.0 : 2.0;
                auto s12_34_deg = (_s1 == _s3) ? (_s2 == _s4 ? 1.0 : 2.0) : 2.0;
                auto s1234_deg = s12_deg * s34_deg * s12_34_deg;

                engine.compute(shells[_s1], shells[_s2], shells[_s3], shells[_s4]);
                const auto* buf_1234 = buf[0];
                if(buf_1234 == nullptr) continue; 

                for(size_t f3 = 0; f3 != n3; ++f3) {
                  const auto bf3 = f3 + bf3_first;
                  for(size_t f4 = 0; f4 != n4; ++f4) {
                    const auto bf4 = f4 + bf4_first;
                    auto _f1 = f2;
                    auto _f2 = f3;
                    auto _f3 = f1;
	                auto _f4 = f4;
                    auto f1234 = _n4*(_n3*(_n2*_f1+_f2)+_f3)+_f4;
                    const auto value = buf_1234[f1234];
                    const auto value_scal_by_deg = value * s1234_deg;
                    G(i,j) -= 0.25 * D(bf3, bf4) * value_scal_by_deg;
                  }
                }
              }

	          if (con3){
                auto _s1 = s2;
                auto _s2 = s3;
                auto _s3 = s4;
                auto _s4 = s1;
                auto _n1 = n2;
                auto _n2 = n3;
                auto _n3 = n4;
                auto _n4 = n1;

                auto s12_deg = (_s1 == _s2) ? 1.0 : 2.0;
                auto s34_deg = (_s3 == _s4) ? 1.0 : 2.0;
                auto s12_34_deg = (_s1 == _s3) ? (_s2 == _s4 ? 1.0 : 2.0) : 2.0;
                auto s1234_deg = s12_deg * s34_deg * s12_34_deg;

                engine.compute(shells[_s1], shells[_s2], shells[_s3], shells[_s4]);
                const auto* buf_1234 = buf[0];
                if(buf_1234 == nullptr) continue; 

                for(size_t f3 = 0; f3 != n3; ++f3) {
                  const auto bf3 = f3 + bf3_first;
                  for(size_t f4 = 0; f4 != n4; ++f4) {
                    const auto bf4 = f4 + bf4_first;
                    auto _f1 = f2;
                    auto _f2 = f3;
                    auto _f3 = f4;
	                auto _f4 = f1;
                    auto f1234 = _n4*(_n3*(_n2*_f1+_f2)+_f3)+_f4;
                    const auto value = buf_1234[f1234];
                    const auto value_scal_by_deg = value * s1234_deg;
                    G(i,j) -= 0.25 * D(bf3, bf4) * value_scal_by_deg;
                  }
                }
              }

	          if (con4){
                auto _s1 = s1;
                auto _s2 = s3;
                auto _s3 = s4;
                auto _s4 = s2;
                auto _n1 = n1;
                auto _n2 = n3;
                auto _n3 = n4;
                auto _n4 = n2;

                auto s12_deg = (_s1 == _s2) ? 1.0 : 2.0;
                auto s34_deg = (_s3 == _s4) ? 1.0 : 2.0;
                auto s12_34_deg = (_s1 == _s3) ? (_s2 == _s4 ? 1.0 : 2.0) : 2.0;
                auto s1234_deg = s12_deg * s34_deg * s12_34_deg;

                engine.compute(shells[_s1], shells[_s2], shells[_s3], shells[_s4]);
                const auto* buf_1234 = buf[0];
                if(buf_1234 == nullptr) continue; 

                for(size_t f3 = 0; f3 != n3; ++f3) {
                  const auto bf3 = f3 + bf3_first;
                  for(size_t f4 = 0; f4 != n4; ++f4) {
                    const auto bf4 = f4 + bf4_first;
                    auto _f1 = f1;
                    auto _f2 = f3;
                    auto _f3 = f4;
	                auto _f4 = f2;
                    auto f1234 = _n4*(_n3*(_n2*_f1+_f2)+_f3)+_f4;
                    const auto value = buf_1234[f1234];
                    const auto value_scal_by_deg = value * s1234_deg;
                    G(i,j) -= 0.25 * D(bf3, bf4) * value_scal_by_deg;
                  }
                }
              }
              }
              }
              }
              }

              for(auto i = block_offset[0];i < (block_offset[0] + block_dims[0]);i++){
                auto s1        = bf2shell[i];
                auto bf1_first = shell2bf[s1];
                auto n1        = shells[s1].size();
              for(auto j = block_offset[1];j < (block_offset[1] + block_dims[1]);j++){
                auto s2        = bf2shell[j];
                auto bf2_first = shell2bf[s2];
                auto n2        = shells[s2].size();
              for(size_t s3 = 0; s3 != shells.size(); ++s3) {
                auto bf3_first = shell2bf[s3]; 
                auto n3        = shells[s3].size(); 
              for(size_t s4 = 0; s4 <= shells.size(); ++s4) {
                auto bf4_first = shell2bf[s4];
                auto n4        = shells[s4].size();

                auto s1p_max = (s3 == s4) ? s2 : s4;
                auto con5 = (s1<=s1p_max && s2<=s3 && s4 <= s3);
                auto s2p_max = (s3 == s4) ? s1 : s4;
                auto con6 = (s2<=s2p_max && s1<=s3 && s4 <= s3);
                auto s4_max = (s3 == s2) ? s1 : s2;
                auto con7 = (s4<=s4_max && s1<=s3 && s2 <= s3);
                auto s4p_max = (s3 == s1) ? s2 : s1;
                auto con8 = (s4<=s4p_max && s2<=s3 && s1 <= s3);
 
              if (con5){
                auto _s1 = s3;
                auto _s2 = s2;
                auto _s3 = s4;
                auto _s4 = s1;
                auto _n1 = n3;
                auto _n2 = n2;
                auto _n3 = n4;
                auto _n4 = n1;
 
                auto s12_deg = (_s1 == _s2) ? 1.0 : 2.0;
                auto s34_deg = (_s3 == _s4) ? 1.0 : 2.0;
                auto s12_34_deg = (_s1 == _s3) ? (_s2 == _s4 ? 1.0 : 2.0) : 2.0;
                auto s1234_deg = s12_deg * s34_deg * s12_34_deg;

                engine.compute(shells[_s1], shells[_s2], shells[_s3], shells[_s4]);
                const auto* buf_1234 = buf[0];
                if(buf_1234 == nullptr) continue; 

                for(size_t f3 = 0; f3 != n3; ++f3) {
                  const auto bf3 = f3 + bf3_first;
                  for(size_t f4 = 0; f4 != n4; ++f4) {
                    const auto bf4 = f4 + bf4_first;
                    auto f1 = i - bf1_first;
                    auto f2 = j - bf2_first;
                    auto _f1 = f3;
                    auto _f2 = f2;
                    auto _f3 = f4;
                    auto _f4 = f1;
                    auto f1234 = _n4*(_n3*(_n2*_f1+_f2)+_f3)+_f4;
                    const auto value = buf_1234[f1234];
                    const auto value_scal_by_deg = value * s1234_deg;
                    G(i, j) -= 0.25 * D(bf3, bf4) * value_scal_by_deg;
                  }
                }
              }

              if (con6){
                auto _s1 = s3;
                auto _s2 = s1;
                auto _s3 = s4;
                auto _s4 = s2;
                auto _n1 = n3;
                auto _n2 = n1;
                auto _n3 = n4;
                auto _n4 = n2;
 
                auto s12_deg = (_s1 == _s2) ? 1.0 : 2.0;
                auto s34_deg = (_s3 == _s4) ? 1.0 : 2.0;
                auto s12_34_deg = (_s1 == _s3) ? (_s2 == _s4 ? 1.0 : 2.0) : 2.0;
                auto s1234_deg = s12_deg * s34_deg * s12_34_deg;

                engine.compute(shells[_s1], shells[_s2], shells[_s3], shells[_s4]);
                const auto* buf_1234 = buf[0];
                if(buf_1234 == nullptr) continue; 

                for(size_t f3 = 0; f3 != n3; ++f3) {
                  const auto bf3 = f3 + bf3_first;
                  for(size_t f4 = 0; f4 != n4; ++f4) {
                    const auto bf4 = f4 + bf4_first;
                    auto f1 = i - bf1_first;
                    auto f2 = j - bf2_first;
                    auto _f1 = f3;
                    auto _f2 = f1;
                    auto _f3 = f4;
                    auto _f4 = f2;
                    auto f1234 = _n4*(_n3*(_n2*_f1+_f2)+_f3)+_f4;
                    const auto value = buf_1234[f1234];
                    const auto value_scal_by_deg = value * s1234_deg;
                    G(i, j) -= 0.25 * D(bf3, bf4) * value_scal_by_deg;
                  }
                }
              }

              if (con7){
                auto _s1 = s3;
                auto _s2 = s1;
                auto _s3 = s2;
                auto _s4 = s4;
                auto _n1 = n3;
                auto _n2 = n1;
                auto _n3 = n2;
                auto _n4 = n4;
 
                auto s12_deg = (_s1 == _s2) ? 1.0 : 2.0;
                auto s34_deg = (_s3 == _s4) ? 1.0 : 2.0;
                auto s12_34_deg = (_s1 == _s3) ? (_s2 == _s4 ? 1.0 : 2.0) : 2.0;
                auto s1234_deg = s12_deg * s34_deg * s12_34_deg;

                engine.compute(shells[_s1], shells[_s2], shells[_s3], shells[_s4]);
                const auto* buf_1234 = buf[0];
                if(buf_1234 == nullptr) continue; 

                for(size_t f3 = 0; f3 != n3; ++f3) {
                  const auto bf3 = f3 + bf3_first;
                  for(size_t f4 = 0; f4 != n4; ++f4) {
                    const auto bf4 = f4 + bf4_first;
                    auto f1 = i - bf1_first;
                    auto f2 = j - bf2_first;
                    auto _f1 = f3;
                    auto _f2 = f1;
                    auto _f3 = f2;
                    auto _f4 = f4;
                    auto f1234 = _n4*(_n3*(_n2*_f1+_f2)+_f3)+_f4;
                    const auto value = buf_1234[f1234];
                    const auto value_scal_by_deg = value * s1234_deg;
                    G(i, j) -= 0.25 * D(bf3, bf4) * value_scal_by_deg;
                  }
                }
              }

              if (con8){
                auto _s1 = s3;
                auto _s2 = s2;
                auto _s3 = s1;
                auto _s4 = s4;
                auto _n1 = n3;
                auto _n2 = n2;
                auto _n3 = n1;
                auto _n4 = n4;
 
                auto s12_deg = (_s1 == _s2) ? 1.0 : 2.0;
                auto s34_deg = (_s3 == _s4) ? 1.0 : 2.0;
                auto s12_34_deg = (_s1 == _s3) ? (_s2 == _s4 ? 1.0 : 2.0) : 2.0;
                auto s1234_deg = s12_deg * s34_deg * s12_34_deg;

                engine.compute(shells[_s1], shells[_s2], shells[_s3], shells[_s4]);
                const auto* buf_1234 = buf[0];
                if(buf_1234 == nullptr) continue; 

                for(size_t f3 = 0; f3 != n3; ++f3) {
                  const auto bf3 = f3 + bf3_first;
                  for(size_t f4 = 0; f4 != n4; ++f4) {
                    const auto bf4 = f4 + bf4_first;
                    auto f1 = i - bf1_first;
                    auto f2 = j - bf2_first;
                    auto _f1 = f3;
                    auto _f2 = f2;
                    auto _f3 = f1;
                    auto _f4 = f4;
                    auto f1234 = _n4*(_n3*(_n2*_f1+_f2)+_f3)+_f4;
                    const auto value = buf_1234[f1234];
                    const auto value_scal_by_deg = value * s1234_deg;
                    G(i, j) -= 0.25 * D(bf3, bf4) * value_scal_by_deg;
                  }
                }
              }
              }
              }
              }
              }

              TAMM_SIZE c = 0;
              for(auto i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
                for(auto j = block_offset[1]; j < block_offset[1] + block_dims[1]; j++, c++) {
                  tbuf[c] = 0.5*G(i, j);
                }
              }

              F1tmp.put(it, tbuf);
            };
                  block_for(ec->pg(), F1tmp(), comp_2bf_lambda);
                //---------------------------END COMPUTE 2-BODY FOCK USING
                // TAMM------------------
#endif

                  // sch(F1tmp() = 0).execute();
                  // eigen_to_tamm_tensor(F1tmp, Ftmp);

                  sch(F1(mu, nu) = 0)(F1(mu, nu) += H1(mu, nu)).execute();

                  F += Ftmp;
                  sch(F1(mu, nu) += F1tmp(mu, nu)).execute();

                  // print_tensor(F1tmp);
                  // cout << "G after 2bf: \n";
                  // if(iter==1){
                  //  //compare_eigen_tamm_tensors(F1,F);
                  //  cout << Ftmp << endl;
                  //  print_tensor(F1tmp);
                  // }

                  auto F1_eigen = tamm_to_eigen_tensor<TensorType, 2>(F1);
                  for(size_t i = 0; i < N; i++)
                      for(size_t j = 0; j < N; j++) F(i, j) = F1_eigen(i, j);
                  F1_eigen.resize(0, 0);

                  //  if (iter>1 && simple_convergence) {
                  //    F = alpha * F + (1.0-alpha)*F_old;
                  //  }

                  //  Eigen::EigenSolver<Matrix> sm12_diag(Sm12);
                  //  Eigen::EigenSolver<Matrix> sp12_diag(Sp12);

                  // S^-1/2
                  Matrix Sm12 = S.pow(-0.5);
                  Matrix Sp12 = S.pow(0.5);

                  Matrix FSm12 = F * Sm12;
                  Matrix Sp12D = Sp12 * D_last;
                  Matrix SpFS  = Sp12D * FSm12;

                  // Assemble: S^(-1/2)*F*D*S^(1/2) - S^(1/2)*D*F*S^(-1/2)
                  Matrix err_mat = SpFS.transpose() - SpFS;
                  //  Matrix err_mat = (Sm12 * F * D_last * Sp12) - (Sp12 *
                  //  D_last * F * Sm12);

                  //  if(iter <= 3 || simple_convergence) { cout << err_mat <<
                  //  endl; }

                  if(iter >= 1 && !simple_convergence) {
                      if(iter > 2) {
                          ++idiis;
                          diis(F, err_mat, D_last, iter, max_hist, idiis,
                               diis_hist, fock_hist);
                      }
                  }

                  // solve F C = e S C
                  Eigen::GeneralizedSelfAdjointEigenSolver<Matrix>
                    gen_eig_solver(F, S);
                  // auto
                  eps = gen_eig_solver.eigenvalues();
                  C   = gen_eig_solver.eigenvectors();
                  // auto C1 = gen_eig_solver.eigenvectors();

                  // compute density, D = C(occ) . C(occ)T
                  auto C_occ = C.leftCols(ndocc);
                  D          = C_occ * C_occ.transpose();

                  // compute HF energy
                  ehf = 0.0;
                  for(size_t i = 0; i < nao; i++)
                      for(size_t j = 0; j < nao; j++)
                          ehf += D(i, j) * (H(i, j) + F(i, j));

                  // compute difference with last iteration
                  ediff = ehf - ehf_last;
                  rmsd  = (D - D_last).norm();

                  cout << "----------------------------------------------------"
                          "-------------"
                          "--------\n";
                  cout << "iter, ehf, ediff, rmsd = " << iter << "," << ehf
                       << ", " << ediff << "," << rmsd << "\n";
                  const auto tstop = std::chrono::high_resolution_clock::now();
                  const std::chrono::duration<double> time_elapsed =
                    tstop - tstart;

                  // if (iter == 1)
                  //   std::cout <<
                  //   "\n\n Iter        E(elec)              E(tot) Delta(E)
                  //   RMS(D) Time(s)\n";
                  // printf(" %02d %20.12f %20.12f %20.12f %20.12f %10.5lf\n",
                  // iter, ehf, ehf
                  // + enuc,
                  //        ediff, rmsd, time_elapsed.count());

                  if(iter > maxiter) {
                      std::cerr << "HF Does not converge!!!\n";
                      exit(0);
                  }

                  //  if(simple_convergence) F_old = F;
              }
              while(((fabs(ediff) > conv) || (fabs(rmsd) > conv)))
                  ;

              std::cout.precision(15);
              printf("\n** Hartree-Fock energy = %20.12f\n", ehf + enuc);

              // cout << "\n** Eigen Values:\n";
              // cout << eps << endl;

              Tensor<TensorType>::deallocate(H1, F1, F1_old);
              Tensor<TensorType>::deallocate(F1tmp);
              return std::make_tuple(ndocc, nao, ehf + enuc, shells);
          }

        void
        diis(Matrix & F, Matrix & err_mat, Matrix & D_last, int iter,
             int max_hist, int ndiis, std::vector<Matrix>& diis_hist,
             std::vector<Matrix>& fock_hist) {
            using Vector = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                         Eigen::RowMajor>;

            const int N = F.rows();
            // const int epos = ((ndiis-1) % max_hist) + 1;
            int epos = ndiis - 1;
            if(ndiis > max_hist) {
                diis_hist.erase(diis_hist.begin());
                fock_hist.erase(fock_hist.begin());
            }
            diis_hist.push_back(err_mat);
            fock_hist.push_back(F);

            // --------------------- Construct error metric
            // -------------------------------
            const int idim = std::min(ndiis, max_hist);
            Matrix A       = Matrix::Zero(idim + 1, idim + 1);
            Vector b       = Vector::Zero(idim + 1, 1);

            for(int i = 0; i < idim; i++) {
                for(int j = i; j < idim; j++) {
                    A(i, j) = (diis_hist[i].transpose() * diis_hist[j]).trace();
                }
            }

            for(int i = 0; i < idim; i++) {
                for(int j = i; j < idim; j++) { A(j, i) = A(i, j); }
            }
            for(int i = 0; i < idim; i++) {
                A(i, idim) = -1.0;
                A(idim, i) = -1.0;
            }

            b(idim, 0) = -1;

            Vector x = A.lu().solve(b);

            F.setZero();
            for(int j = 0; j < idim; j++) { F += x(j, 0) * fock_hist[j]; }

            // cout << "-----------iter:" << iter << "--------------\n";
            // cout << err_mat << endl;
        }

        // computes Superposition-Of-Atomic-Densities guess for the molecular
        // density matrix in minimal basis; occupies subshells by smearing
        // electrons evenly over the orbitals
        Matrix compute_soad(const std::vector<libint2::Atom>& atoms) {
            // compute number of atomic orbitals
            size_t nao = 0;
            for(const auto& atom : atoms) {
                const auto Z = atom.atomic_number;
                if(Z == 1 || Z == 2) // H, He
                    nao += 1;
                else if(Z <= 10) // Li - Ne
                    nao += 5;
                else
                    throw "SOAD with Z > 10 is not yet supported";
            }

            // compute the minimal basis density
            Matrix D         = Matrix::Zero(nao, nao);
            size_t ao_offset = 0; // first AO of this atom
            for(const auto& atom : atoms) {
                const auto Z = atom.atomic_number;
                if(Z == 1 || Z == 2) {           // H, He
                    D(ao_offset, ao_offset) = Z; // all electrons go to the 1s
                    ao_offset += 1;
                } else if(Z <= 10) {
                    D(ao_offset, ao_offset) = 2; // 2 electrons go to the 1s
                    D(ao_offset + 1, ao_offset + 1) =
                      (Z == 3) ?
                        1 :
                        2; // Li? only 1 electron in 2s, else 2 electrons
                    // smear the remaining electrons in 2p orbitals
                    const double num_electrons_per_2p =
                      (Z > 4) ? (double)(Z - 4) / 3 : 0;
                    for(auto xyz = 0; xyz != 3; ++xyz)
                        D(ao_offset + 2 + xyz, ao_offset + 2 + xyz) =
                          num_electrons_per_2p;
                    ao_offset += 5;
                }
            }

            return D * 0.5; // we use densities normalized to # of electrons/2
        }

        Matrix compute_2body_fock(const std::vector<libint2::Shell>& shells,
                                  const Matrix& D) {
            using libint2::Engine;
            using libint2::Operator;
            using libint2::Shell;

            std::chrono::duration<double> time_elapsed =
              std::chrono::duration<double>::zero();

            const auto n = nbasis(shells);
            Matrix G     = Matrix::Zero(n, n);

            // construct the 2-electron repulsion integrals engine
            Engine engine(Operator::coulomb, max_nprim(shells), max_l(shells),
                          0);

            auto shell2bf = map_shell_to_basis_function(shells);

            const auto& buf = engine.results();

            // The problem with the simple Fock builder is that permutational
            // symmetries of the Fock, density, and two-electron integrals are
            // not taken into account to reduce the cost. To make the simple
            // Fock builder efficient we must rearrange our computation. The
            // most expensive step in Fock matrix construction is the evaluation
            // of 2-e integrals; hence we must minimize the number of computed
            // integrals by taking advantage of their permutational symmetry.
            // Due to the multiplicative and Hermitian nature of the Coulomb
            // kernel (and realness of the Gaussians) the permutational symmetry
            // of the 2-e ints is given by the following relations:
            //
            // (12|34) = (21|34) = (12|43) = (21|43) = (34|12) = (43|12) =
            // (34|21) = (43|21)
            //
            // (here we use chemists' notation for the integrals, i.e in (ab|cd)
            // a and b correspond to electron 1, and c and d -- to electron 2).
            //
            // It is easy to verify that the following set of nested loops
            // produces a permutationally-unique set of integrals: foreach a = 0
            // .. n-1
            //   foreach b = 0 .. a
            //     foreach c = 0 .. a
            //       foreach d = 0 .. (a == c ? b : c)
            //         compute (ab|cd)
            //
            // The only complication is that we must compute integrals over
            // shells. But it's not that complicated ...
            //
            // The real trick is figuring out to which matrix elements of the
            // Fock matrix each permutationally-unique (ab|cd) contributes. STOP
            // READING and try to figure it out yourself. (to check your answer
            // see below)

            // loop over permutationally-unique set of shells
            for(size_t s1 = 0; s1 != shells.size(); ++s1) {
                auto bf1_first =
                  shell2bf[s1]; // first basis function in this shell
                auto n1 =
                  shells[s1].size(); // number of basis functions in this shell

                for(size_t s2 = 0; s2 <= s1; ++s2) {
                    auto bf2_first = shell2bf[s2];
                    auto n2        = shells[s2].size();

                    for(size_t s3 = 0; s3 <= s1; ++s3) {
                        auto bf3_first = shell2bf[s3];
                        auto n3        = shells[s3].size();

                        const auto s4_max = (s1 == s3) ? s2 : s3;
                        for(size_t s4 = 0; s4 <= s4_max; ++s4) {
                            auto bf4_first = shell2bf[s4];
                            auto n4        = shells[s4].size();

                            // compute the permutational degeneracy (i.e. # of
                            // equivalents) of the given shell set
                            auto s12_deg = (s1 == s2) ? 1.0 : 2.0;
                            auto s34_deg = (s3 == s4) ? 1.0 : 2.0;
                            auto s12_34_deg =
                              (s1 == s3) ? (s2 == s4 ? 1.0 : 2.0) : 2.0;
                            auto s1234_deg = s12_deg * s34_deg * s12_34_deg;

                            const auto tstart =
                              std::chrono::high_resolution_clock::now();

                            engine.compute(shells[s1], shells[s2], shells[s3],
                                           shells[s4]);
                            const auto* buf_1234 = buf[0];
                            if(buf_1234 == nullptr)
                                continue; // if all integrals screened out, skip
                                          // to next quartet

                            const auto tstop =
                              std::chrono::high_resolution_clock::now();
                            time_elapsed += tstop - tstart;

                            // ANSWER
                            // 1) each shell set of integrals contributes up to
                            // 6 shell sets of the Fock matrix:
                            //    F(a,b) += (ab|cd) * D(c,d)
                            //    F(c,d) += (ab|cd) * D(a,b)
                            //    F(b,d) -= 1/4 * (ab|cd) * D(a,c)
                            //    F(b,c) -= 1/4 * (ab|cd) * D(a,d)
                            //    F(a,c) -= 1/4 * (ab|cd) * D(b,d)
                            //    F(a,d) -= 1/4 * (ab|cd) * D(b,c)
                            // 2) each permutationally-unique integral (shell
                            // set) must be scaled by its degeneracy,
                            //    i.e. the number of the integrals/sets
                            //    equivalent to it
                            // 3) the end result must be symmetrized
                            for(size_t f1 = 0, f1234 = 0; f1 != n1; ++f1) {
                                const auto bf1 = f1 + bf1_first;
                                for(size_t f2 = 0; f2 != n2; ++f2) {
                                    const auto bf2 = f2 + bf2_first;
                                    for(size_t f3 = 0; f3 != n3; ++f3) {
                                        const auto bf3 = f3 + bf3_first;
                                        for(size_t f4 = 0; f4 != n4;
                                            ++f4, ++f1234) {
                                            const auto bf4 = f4 + bf4_first;

                                            const auto value = buf_1234[f1234];

                                            const auto value_scal_by_deg =
                                              value * s1234_deg;

                                            G(bf1, bf2) +=
                                              D(bf3, bf4) * value_scal_by_deg;
                                            G(bf3, bf4) +=
                                              D(bf1, bf2) * value_scal_by_deg;
                                            G(bf1, bf3) -= 0.25 * D(bf2, bf4) *
                                                           value_scal_by_deg;
                                            G(bf2, bf4) -= 0.25 * D(bf1, bf3) *
                                                           value_scal_by_deg;
                                            G(bf1, bf4) -= 0.25 * D(bf2, bf3) *
                                                           value_scal_by_deg;
                                            G(bf2, bf3) -= 0.25 * D(bf1, bf4) *
                                                           value_scal_by_deg;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // symmetrize the result and return
            Matrix Gt = G.transpose();
            return 0.5 * (G + Gt);
        }

#endif // TAMM_TESTS_HF_TAMM_HPP_
