
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

void diis(Matrix& F, Matrix& S, int iter, int max_hist,
          int idiis, std::vector<Matrix>& diis_hist,
          std::vector<Matrix>& fock_hist);

// an efficient Fock builder; *integral-driven* hence computes
// permutationally-unique ints once
Matrix compute_2body_fock(const std::vector<libint2::Shell>& shells,
                          const Matrix& D);

template <class T> T &unconst_cast(const T &v) { return const_cast<T &>(v); }

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

std::tuple<int, int, double, libint2::BasisSet> hartree_fock(
  const string filename, Matrix& C, Matrix& F) {
    // Perform the simple HF calculation (Ed) and 2,4-index transform to get the
    // inputs for CCSD
    using libint2::Atom;
    using libint2::Engine;
    using libint2::Operator;
    using libint2::Shell;

    auto hf_t1 = std::chrono::high_resolution_clock::now();

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
    if(GA_Nodeid()==0) cout << "\nNuclear repulsion energy = " << enuc << endl;

    // initializes the Libint integrals library ... now ready to compute
    libint2::initialize(false);

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
    auto debug = false;

    IndexSpace AO{range(0, N)};
    std::vector<unsigned int> AO_tiles;
    for(auto s : shells) AO_tiles.push_back(s.size());
    // if(rank==0){
    //     // cout << "AO tiles = " << AO_tiles << endl;
    //     cout << "Number of AO tiles = " << AO_tiles.size() << endl;
    // }
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
    block_for(ec->pg(), S1(), compute_1body_ints);

    engine.set(Operator::kinetic);
    buf = engine.results();
    // Core Hamiltonian = T + V
    tensor1e = T1;
    block_for(ec->pg(), T1(), compute_1body_ints);
    Scheduler{ec}(H1P(mup, nup) = T1(mup, nup)).execute();

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
    block_for(ec->pg(), V1(), compute_1body_ints);
    Scheduler{ec}(H1P(mup, nup) += V1(mup, nup)).execute();

    hf_t2 = std::chrono::high_resolution_clock::now();
    hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

    if(rank == 0) std::cout << "\nTime taken for H=T+V, S: " << hf_time << " secs\n";

    hf_t1 = std::chrono::high_resolution_clock::now();

    tamm_to_eigen_tensor(H1P, H);
    tamm_to_eigen_tensor(S1, S);
    eigen_to_tamm_tensor(H1, H);

    hf_t2 = std::chrono::high_resolution_clock::now();
    hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

    if(rank == 0) std::cout << "\nTime taken for tamm to eigen - H,S: " << hf_time << " secs\n";
    
    /*** =========================== ***/
    /*** build initial-guess density ***/
    /*** =========================== ***/

    hf_t1 = std::chrono::high_resolution_clock::now();
    // hcore guess
    // solve H C = e S C
    Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> gen_eig_solver(H, S);
    Matrix eps = gen_eig_solver.eigenvalues();
    C  = gen_eig_solver.eigenvectors();

    // compute density, D = C(occ) . C(occ)T
    auto C_occ = C.leftCols(ndocc);
    Matrix D          = C_occ * C_occ.transpose();
    
    //  cout << "\n\tInitial Density Matrix:\n";
    //  cout << D << endl;

    hf_t2 = std::chrono::high_resolution_clock::now();
    hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

    if(rank == 0) std::cout << "\nTime taken for eigen_solve(H,S): " << hf_time << " secs\n";


    hf_t1 = std::chrono::high_resolution_clock::now();
    /*** =========================== ***/
    /*** main iterative loop         ***/
    /*** =========================== ***/

    const auto maxiter = 100;
    const auto conv    = 1e-7;
    auto iter          = 0;
    auto rmsd          = 0.0;
    auto ediff         = 0.0;
    auto ehf           = 0.0;
    auto is_conv       = true;
    //  Matrix C;
    //  Matrix F;
    
    double alpha = 0.75;
    // Matrix F_old;

    int idiis                     = 0;
    int max_hist                  = 10;
    std::vector<Matrix> diis_hist;
    std::vector<Matrix> fock_hist;

    Tensor<TensorType> ehf_tmp{tAO, tAO};
    Tensor<TensorType> ehf_tamm{}, rmsd_tamm{};

    Tensor<TensorType> F1{tAO, tAO};
    Tensor<TensorType> F1tmp{tAO, tAO};
    Tensor<TensorType>::allocate(ec, F1, F1tmp, ehf_tmp, ehf_tamm, rmsd_tamm);

    Tensor<TensorType> Sm12_tamm{tAO, tAO}; 
    Tensor<TensorType> Sp12_tamm{tAO, tAO};
    Tensor<TensorType> D_tamm{tAO, tAO};
    Tensor<TensorType> D_last_tamm{tAO, tAO};
    Tensor<TensorType>::allocate(ec, Sm12_tamm, Sp12_tamm, D_tamm, D_last_tamm);

    // FSm12,Sp12D,SpFS
    Tensor<TensorType> FSm12_tamm{tAO, tAO}; 
    Tensor<TensorType> Sp12D_tamm{tAO, tAO};
    Tensor<TensorType> SpFS_tamm{tAO, tAO};
    Tensor<TensorType> err_mat_tamm{tAO, tAO};
    Tensor<TensorType>::allocate(ec, FSm12_tamm, Sp12D_tamm, SpFS_tamm,err_mat_tamm);

    eigen_to_tamm_tensor(D_tamm,D);
    F.setZero(N,N);
    Matrix err_mat = Matrix::Zero(N,N);

    hf_t2 = std::chrono::high_resolution_clock::now();
    hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

   // GA_Sync();
    if(rank == 0) std::cout << "\nTime taken to setup main loop: " << hf_time << " secs\n";

    if(rank == 0) {
        std::cout << "\n\n";
        std::cout << " Hartree-Fock iterations" << std::endl;
        std::cout << std::string(70, '-') << std::endl;
        std::cout <<
            " Iter     Energy            E-Diff           RMSD           Time" 
                << std::endl;
        std::cout << std::string(70, '-') << std::endl;
    }

    std::cout << std::fixed << std::setprecision(2);

    // S^-1/2
    Matrix Sm12 = S.pow(-0.5);
    Matrix Sp12 = S.pow(0.5);
    eigen_to_tamm_tensor(Sm12_tamm,Sm12);
    eigen_to_tamm_tensor(Sp12_tamm,Sp12);    

    do {

        // Scheduler sch{ec};
        const auto loop_start = std::chrono::high_resolution_clock::now();
        ++iter;

        // Save a copy of the energy and the density
        auto ehf_last = ehf;
        auto D_last   = D;

        Scheduler{ec} //(F1tmp() = 0)
           (D_last_tamm(mu,nu) = D_tamm(mu,nu)).execute();

        // build a new Fock matrix
        // F           = H;

        hf_t1 = std::chrono::high_resolution_clock::now();
        Matrix Ftmp = compute_2body_fock(shells, D);

        eigen_to_tamm_tensor(F1tmp, Ftmp);

        hf_t2 = std::chrono::high_resolution_clock::now();
        hf_time =
        std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

        if(rank == 0 && debug) std::cout << "2BF:" << hf_time << "s, ";

        hf_t1 = std::chrono::high_resolution_clock::now();
        // F += Ftmp;
        Scheduler{ec}(F1(mu, nu) = H1(mu, nu))
                     (F1(mu, nu) += F1tmp(mu, nu)).execute();


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

        hf_t1 = std::chrono::high_resolution_clock::now();

        Scheduler{ec}(FSm12_tamm() = 0)(FSm12_tamm(mu,nu) += F1(mu,ku) * Sm12_tamm(ku,nu))
        (Sp12D_tamm() = 0)(Sp12D_tamm(mu,nu) += Sp12_tamm(mu,ku) * D_last_tamm(ku,nu))
        (SpFS_tamm() = 0)(SpFS_tamm(mu,nu)  += Sp12D_tamm(mu,ku) * FSm12_tamm(ku,nu))
    
        (err_mat_tamm(mu,nu) = SpFS_tamm(nu,mu))
        (err_mat_tamm(mu,nu) += -1.0 * SpFS_tamm(mu,nu)).execute();

        hf_t2 = std::chrono::high_resolution_clock::now();
        hf_time =
        std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

        if(rank == 0 && debug) std::cout << "err_mat:" << hf_time << "s, ";        
        tamm_to_eigen_tensor(err_mat_tamm,err_mat);
        tamm_to_eigen_tensor(F1,F);

        hf_t1 = std::chrono::high_resolution_clock::now();

        if(iter > 2) {
            ++idiis;
            diis(F, err_mat, iter, max_hist, idiis,
                diis_hist, fock_hist);
        }
    
        hf_t2 = std::chrono::high_resolution_clock::now();
        hf_time =
        std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

        if(rank == 0 && debug) std::cout << "diis:" << hf_time << "s, ";    

        hf_t1 = std::chrono::high_resolution_clock::now();
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

        hf_t2 = std::chrono::high_resolution_clock::now();
        hf_time =
        std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

        if(rank == 0 && debug) std::cout << "eigen_solve:" << hf_time << "s, ";    

        hf_t1 = std::chrono::high_resolution_clock::now();

        eigen_to_tamm_tensor(D_tamm,D);
        eigen_to_tamm_tensor(F1,F);

        hf_t2 = std::chrono::high_resolution_clock::now();
        hf_time =
        std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

        if(rank == 0 && debug) std::cout << "E2T-D,F:" << hf_time << "s, ";    

        hf_t1 = std::chrono::high_resolution_clock::now();
        // compute HF energy 
        // e = D * (H+F);
        Scheduler{ec}(ehf_tamm()=0)
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
            std::cout << std::fixed << std::setprecision(10) << ehf;
            std::cout << ' ' << std::setw(16)  << ediff;
            std::cout << ' ' << std::setw(15)  << rmsd << ' ';
            std::cout << std::fixed << std::setprecision(2);
            std::cout << ' ' << std::setw(12)  << loop_time << ' ' << "\n";
        }

        if(iter > maxiter) {                
            is_conv = false;
            break;
        }

    }
    while(((fabs(ediff) > conv) || (fabs(rmsd) > conv)))
        ;

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

    Tensor<TensorType>::deallocate(H1, H1P, S1, T1, V1, F1, D_tamm, ehf_tmp, 
                    ehf_tamm, rmsd_tamm);
    Tensor<TensorType>::deallocate(F1tmp,Sm12_tamm, Sp12_tamm,
    D_last_tamm,FSm12_tamm, Sp12D_tamm, SpFS_tamm,err_mat_tamm);

    MemoryManagerGA::destroy_coll(mgr);
    delete ec;

    return std::make_tuple(ndocc, nao, ehf + enuc, shells);
}

void diis(Matrix& F, Matrix& err_mat, int iter, int max_hist, int ndiis,
          std::vector<Matrix>& diis_hist, std::vector<Matrix>& fock_hist) {
    using Vector =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    const int N = F.rows();
    // const int epos = ((ndiis-1) % max_hist) + 1;
    int epos = ndiis - 1;
    if(ndiis > max_hist) {
        diis_hist.erase(diis_hist.begin());
        fock_hist.erase(fock_hist.begin());
    }
    diis_hist.push_back(err_mat);
    fock_hist.push_back(F);

    // ----- Construct error metric -----
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

Matrix compute_2body_fock(const std::vector<libint2::Shell>& shells,
                          const Matrix& D) {
    using libint2::Shell;
    using libint2::Engine;
    using libint2::Operator;

    std::chrono::duration<double> time_elapsed =
      std::chrono::duration<double>::zero();

    const auto n = nbasis(shells);
    Matrix G     = Matrix::Zero(n, n);

    // construct the 2-electron repulsion integrals engine
    Engine engine(Operator::coulomb, max_nprim(shells), max_l(shells), 0);

    auto shell2bf = map_shell_to_basis_function(shells);

    const auto& buf = engine.results();

    // loop over permutationally-unique set of shells
    for(size_t s1 = 0; s1 != shells.size(); ++s1) {
        auto bf1_first = shell2bf[s1]; // first basis function in this shell
        auto n1 = shells[s1].size(); // number of basis functions in this shell

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
                    auto s12_deg    = (s1 == s2) ? 1.0 : 2.0;
                    auto s34_deg    = (s3 == s4) ? 1.0 : 2.0;
                    auto s12_34_deg = (s1 == s3) ? (s2 == s4 ? 1.0 : 2.0) : 2.0;
                    auto s1234_deg  = s12_deg * s34_deg * s12_34_deg;

                    const auto tstart =
                      std::chrono::high_resolution_clock::now();

                    engine.compute2<Operator::coulomb, libint2::BraKet::xx_xx, 0>(shells[s1], shells[s2], shells[s3],
                                   shells[s4]);
                    const auto* buf_1234 = buf[0];
                    if(buf_1234 == nullptr)
                        continue; // if all integrals screened out, skip to
                                  // next quartet

                    const auto tstop =
                      std::chrono::high_resolution_clock::now();
                    time_elapsed += tstop - tstart;

                    // ANSWER
                    // 1) each shell set of integrals contributes up to 6
                    // shell sets of the Fock matrix:
                    //    F(a,b) += (ab|cd) * D(c,d)
                    //    F(c,d) += (ab|cd) * D(a,b)
                    //    F(b,d) -= 1/4 * (ab|cd) * D(a,c)
                    //    F(b,c) -= 1/4 * (ab|cd) * D(a,d)
                    //    F(a,c) -= 1/4 * (ab|cd) * D(b,d)
                    //    F(a,d) -= 1/4 * (ab|cd) * D(b,c)
                    // 2) each permutationally-unique integral (shell set)
                    // must be scaled by its degeneracy,
                    //    i.e. the number of the integrals/sets equivalent
                    //    to it
                    // 3) the end result must be symmetrized
                    for(size_t f1 = 0, f1234 = 0; f1 != n1; ++f1) {
                        const auto bf1 = f1 + bf1_first;
                        for(size_t f2 = 0; f2 != n2; ++f2) {
                            const auto bf2 = f2 + bf2_first;
                            for(size_t f3 = 0; f3 != n3; ++f3) {
                                const auto bf3 = f3 + bf3_first;
                                for(size_t f4 = 0; f4 != n4; ++f4, ++f1234) {
                                    const auto bf4 = f4 + bf4_first;

                                    const auto value = buf_1234[f1234];

                                    const auto value_scal_by_deg =
                                      value * s1234_deg;

                                    G(bf1, bf2) +=
                                      D(bf3, bf4) * value_scal_by_deg;
                                    G(bf3, bf4) +=
                                      D(bf1, bf2) * value_scal_by_deg;
                                    G(bf1, bf3) -=
                                      0.25 * D(bf2, bf4) * value_scal_by_deg;
                                    G(bf2, bf4) -=
                                      0.25 * D(bf1, bf3) * value_scal_by_deg;
                                    G(bf1, bf4) -=
                                      0.25 * D(bf2, bf3) * value_scal_by_deg;
                                    G(bf2, bf3) -=
                                      0.25 * D(bf1, bf4) * value_scal_by_deg;
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


    // auto compute_1body_ints_lambda = [&](const IndexVector& blockid,
    //                                             span<TensorType> tbuf) {

    //     auto s1 = blockid[0];
    //     // auto bf1 = shell2bf[s1]; //shell2bf[s1]; // first basis function in
    //     // this shell
    //     auto n1 = shells[s1].size();

    //     // for (size_t s2 = 0; s2 <= s1; ++s2) {
    //     auto s2 = blockid[1];

    //     //if (s2>s1) return;
    //     // if(s1<s2) return; //TODO
    //     // auto bf2 = shell2bf[s2];
    //     auto n2 = shells[s2].size();

    //     // compute shell pair; return is the pointer to the buffer
    //     engine.compute(shells[s1], shells[s2]);
    //     // "map" buffer to a const Eigen Matrix, and copy it to the
    //     // corresponding blocks of the result
    //     Eigen::Map<const Matrix> buf_mat(buf[0], n1, n2);
    //     // result.block(bf1, bf2, n1, n2) = buf_mat;
    //     // for(size_t i = 0; i < n1; i++)
    //     //     for(size_t j = 0; j < n2; j++) tbuf[i * n2 + j] = buf_mat(i, j);
    //     //std::memcpy(tbuf.data(),buf, sizeof(TensorType)*n1*n2);
    //     Eigen::Map<Matrix>(tbuf.data(),n1,n2) = buf_mat;

    //     // if(s1!=s2){
    //     //     std::vector<T> ttbuf(n1*n2);
    //     //     Eigen::Map<Matrix>(ttbuf.data(),n2,n1) = buf_mat.transpose();
    //     //     put({s2,s1}, ttbuf);
    //     // }
    // };

    // TODO
#if 0
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
            
            //   std::cout << "blockdims = " << block_dims[0] << ":" << block_dims[1] << std::endl;
              Matrix G = Matrix::Zero(N,N);
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
              
                decltype(s1) _s1 = -1;
                decltype(s1) _s2 = -1;
                decltype(s1) _s3 = -1;
                decltype(s1) _s4 = -1;
                decltype(s1) _n1 = -1;
                decltype(s1) _n2 = -1;
                decltype(s1) _n3 = -1;
                decltype(s1) _n4 = -1;

                decltype(f1) _f1 = -1;
                decltype(f1) _f2 = -1;
                decltype(f1) _f3 = -1;
                decltype(f1) _f4 = -1;

              auto lambda_2e = [&](std::vector<int> bf_order, double pf=1.0){
              
                auto s12_deg = (_s1 == _s2) ? 1.0 : 2.0;
                auto s34_deg = (_s3 == _s4) ? 1.0 : 2.0;
                auto s12_34_deg = (_s1 == _s3) ? (_s2 == _s4 ? 1.0 : 2.0) : 2.0;
                auto s1234_deg = s12_deg * s34_deg * s12_34_deg;
                 
                engine.compute(shells[_s1], shells[_s2], shells[_s3], shells[_s4]);
                const auto* buf_1234 = buf[0];
                if(buf_1234 == nullptr) return; 

                for(size_t f3 = 0; f3 != n3; ++f3) {
                  const auto bf3 = f3 + bf3_first;
                  for(size_t f4 = 0; f4 != n4; ++f4) {
                    const auto bf4 = f4 + bf4_first;
                     std::vector<decltype(f1)> fxs{f1,f2,f3,f4};
                     _f1 = fxs.at(bf_order[0]);
                     _f2 = fxs.at(bf_order[1]);
                     _f3 = fxs.at(bf_order[2]);
	                   _f4 = fxs.at(bf_order[3]);
                    auto f1234 = _n4*(_n3*(_n2*_f1+_f2)+_f3)+_f4;
                    const auto value = buf_1234[f1234];
                    const auto value_scal_by_deg = value * s1234_deg;
                    G(i,j) += pf * D(bf3, bf4) * value_scal_by_deg;
                  }
                }

                };

              if (con0){
                 _s1 = s1;
                 _s2 = s2;
                 _s3 = s3;
                 _s4 = s4;
                 _n1 = n1;
                 _n2 = n2;
                 _n3 = n3;
                 _n4 = n4;
                lambda_2e({0,1,2,3});
              }
              
              if (conz){
                 _s1 = s2;
                 _s2 = s1;
                 _s3 = s3;
                 _s4 = s4;
                 _n1 = n2;
                 _n2 = n1;
                 _n3 = n3;
                 _n4 = n4;
                lambda_2e({1,0,2,3});
              }

              if (conx){
	              _s1 = s3;
	              _s2 = s4;
                _s3 = s1;
                _s4 = s2;
                _n1 = n3;
                _n2 = n4;
                _n3 = n1;
                _n4 = n2;
                lambda_2e({2,3,0,1});
                
              }

              if (cony){
	              _s1 = s3;
	              _s2 = s4;
                _s3 = s2;
                _s4 = s1;
                _n1 = n3;
                _n2 = n4;
                _n3 = n2;
                _n4 = n1;

                lambda_2e({2,3,1,0});
              }

	          if (con1) {	
                _s1 = s1;
                _s2 = s3;
                _s3 = s2;
                _s4 = s4;
                _n1 = n1;
                _n2 = n3;
                _n3 = n2;
                _n4 = n4;

                lambda_2e({0,2,1,3}, -0.25);
              }

	          if (con2) {	
                _s1 = s2;
                _s2 = s3;
                _s3 = s1;
                _s4 = s4;
                _n1 = n2;
                _n2 = n3;
                _n3 = n1;
                _n4 = n4;

                lambda_2e({1,2,0,3}, -0.25);
              }

	          if (con3){
                _s1 = s2;
                _s2 = s3;
                _s3 = s4;
                _s4 = s1;
                _n1 = n2;
                _n2 = n3;
                _n3 = n4;
                _n4 = n1;

                lambda_2e({1,2,3,0}, -0.25);
              }

	          if (con4){
                _s1 = s1;
                _s2 = s3;
                _s3 = s4;
                _s4 = s2;
                _n1 = n1;
                _n2 = n3;
                _n3 = n4;
                _n4 = n2;

                lambda_2e({0,2,3,1}, -0.25);
              }

                 s1p_max = (s3 == s4) ? s2 : s4;
                auto con5 = (s1<=s1p_max && s2<=s3 && s4 <= s3);
                 s2p_max = (s3 == s4) ? s1 : s4;
                auto con6 = (s2<=s2p_max && s1<=s3 && s4 <= s3);
                 s4_max = (s3 == s2) ? s1 : s2;
                auto con7 = (s4<=s4_max && s1<=s3 && s2 <= s3);
                 s4p_max = (s3 == s1) ? s2 : s1;
                auto con8 = (s4<=s4p_max && s2<=s3 && s1 <= s3);
 
              if (con5){
                _s1 = s3;
                _s2 = s2;
                _s3 = s4;
                _s4 = s1;
                _n1 = n3;
                _n2 = n2;
                _n3 = n4;
                _n4 = n1;
 
                lambda_2e({2,1,3,0}, -0.25);
              }

              if (con6){
                _s1 = s3;
                _s2 = s1;
                _s3 = s4;
                _s4 = s2;
                _n1 = n3;
                _n2 = n1;
                _n3 = n4;
                _n4 = n2;
 
                lambda_2e({2,0,3,1}, -0.25);
              }

              if (con7){
                _s1 = s3;
                _s2 = s1;
                _s3 = s2;
                _s4 = s4;
                _n1 = n3;
                _n2 = n1;
                _n3 = n2;
                _n4 = n4;
 
                lambda_2e({2,0,1,3}, -0.25);
              }

              if (con8){
                _s1 = s3;
                _s2 = s2;
                _s3 = s1;
                _s4 = s4;
                _n1 = n3;
                _n2 = n2;
                _n3 = n1;
                _n4 = n4;
 
                lambda_2e({2,1,0,3}, -0.25);
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
                  
                //---------------------------END COMPUTE 2-BODY FOCK USING
                // TAMM------------------
#endif
        // block_for(ec->pg(), F1tmp(), comp_2bf_lambda);


#endif // TAMM_TESTS_HF_TAMM_HPP_
