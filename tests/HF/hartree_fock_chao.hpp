
#ifndef TAMM_TESTS_HF_HPP_
#define TAMM_TESTS_HF_HPP_

// standard C++ headers
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <tuple>
#include <functional>
#include <cassert>


#include "hf_common.hpp"
// std::ofstream logOFS;
#include <mpi.h>

void diis(Matrix& F, Matrix& err_mat, Matrix& D_last, int iter, int max_hist, 
         int ndiis, std::vector<Matrix> &diis_hist,std::vector<Matrix> &fock_hist) 

std::tuple<int,int, double, libint2::BasisSet> hartree_fock(const string filename, Matrix &C, Matrix &F) {

  // Perform the simple HF calculation (Ed) and 2,4-index transform to get the inputs for CCSD
  using libint2::Atom;
  using libint2::Shell;
  using libint2::Engine;
  using libint2::BasisSet;
  using libint2::Operator;

  /*** =========================== ***/
  /*** initialize molecule         ***/
  /*** =========================== ***/
//Chao
  MPI_Comm comm = GA_MPI_Comm();
  int rank, nprocs;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nprocs);
  cout << " rank = " << rank << " nprocs = " << nprocs << endl; 

  // read geometry from a .nwx file 
  auto is = std::ifstream(filename);
  std::vector<Atom> atoms;
  OptionsMap options_map;
  std::tie(atoms, options_map) = read_input_nwx(is);

  auto rank = ec->pg().rank();
  SCFOptions scf_options = options_map.scf_options;

  if(rank == 0) {
    cout << "\nNumber of GA ranks: " << GA_Nnodes() << endl;
    scf_options.print();
  }

  std::string basis = scf_options.basis;
  int maxiter = scf_options.maxiter;
  double conve = scf_options.conve;
  double convd = scf_options.convd;
  double tol_int = scf_options.tol_int;
  int max_hist = scf_options.diis_hist; 
  auto debug = scf_options.debug;
  auto restart = scf_options.restart;
  Tile tilesize = static_cast<Tile>(scf_options.AO_tilesize);

  tol_int = std::min(1e-8, 0.01 * conve);

//  std::cout << "Geometries in bohr units \n";
//  for (auto i = 0; i < atoms.size(); ++i)
//    std::cout << atoms[i].atomic_number << "  " << atoms[i].x<< "  " << atoms[i].y<< "  " << atoms[i].z << std::endl;
  // count the number of electrons
  auto nelectron = 0;
  for (size_t i = 0; i < atoms.size(); ++i)
    nelectron += atoms[i].atomic_number;
  const auto ndocc = nelectron / 2;

  // compute the nuclear repulsion energy
  auto enuc = 0.0;
  for (size_t i = 0; i < atoms.size(); i++)
    for (size_t j = i + 1; j < atoms.size(); j++) {
      auto xij = atoms[i].x - atoms[j].x;
      auto yij = atoms[i].y - atoms[j].y;
      auto zij = atoms[i].z - atoms[j].z;
      auto r2 = xij * xij + yij * yij + zij * zij;
      auto r = sqrt(r2);
      enuc += atoms[i].atomic_number * atoms[j].atomic_number / r;
    }
  if (rank == 0) cout << "\nNuclear repulsion energy = " << enuc << endl;

  // initializes the Libint integrals library ... now ready to compute
  libint2::initialize(false);

  /*** =========================== ***/
  /*** create basis set            ***/
  /*** =========================== ***/

  // LIBINT_INSTALL_DIR/share/libint/2.4.0-beta.1/basis
  libint2::BasisSet shells(std::string(basis), atoms);
  //auto shells = make_sto3g_basis(atoms);

      // compute OBS non-negligible shell-pair list
    {
      std::tie(obs_shellpair_list, obs_shellpair_data) = compute_shellpairs(shells);
      size_t nsp = 0;
      for (auto& sp : obs_shellpair_list) {
        nsp += sp.second.size();
      }
      std::cout << "# of {all,non-negligible} shell-pairs = {"
                << shells.size() * (shells.size() + 1) / 2 << "," << nsp << "}"
                << std::endl;
    }

  size_t nao = 0;
  for (size_t s = 0; s < shells.size(); ++s)
    nao += shells[s].size();

  const size_t N = nbasis(shells);
  assert(N == nao);

  cout << "\nNumber of basis functions: " << N << endl;

  /*** =========================== ***/
  /*** compute 1-e integrals       ***/
  /*** =========================== ***/

  auto hf_t1 = std::chrono::high_resolution_clock::now();

 
  // compute one-body integrals
  auto S = compute_1body_ints<libint2::Operator::overlap>(shells)[0];
  auto T = compute_1body_ints<libint2::Operator::kinetic>(shells)[0];
  auto V = compute_1body_ints<libint2::Operator::nuclear>(shells, libint2::make_point_charges(atoms))[0];

  // Core Hamiltonian = T + V
  Matrix H = T + V;
//  cout << "\n\tCore Hamiltonian:\n";
//  cout << H << endl;

  auto hf_t2 = std::chrono::high_resolution_clock::now();

  double hf_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
  std::cout << "\nTime taken for H = T+V, S: " << hf_time << " secs\n";

  // T and V no longer needed, free up the memory
  T.resize(0, 0);
  V.resize(0, 0);

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
        
        
  // pre-compute data for Schwarz bounds
  auto SchwarzK = compute_schwarz_ints<>(shells);

  const auto use_hcore_guess = false;  // use core Hamiltonian eigenstates to guess density?
  // set to true to match the result of versions 0, 1, and 2 of the code
  // HOWEVER !!! even for medium-size molecules hcore will usually fail !!!
  // thus set to false to use Superposition-Of-Atomic-Densities (SOAD) guess
  Matrix D;
  if (use_hcore_guess) { // hcore guess

    // solve H C = e S C
    Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> gen_eig_solver(H, S);
    auto eps = gen_eig_solver.eigenvalues();
    auto C = gen_eig_solver.eigenvectors();
//    cout << "\n\tInitial C Matrix:\n";
//    cout << C << endl;

    // compute density, D = C(occ) . C(occ)T
    auto C_occ = C.leftCols(ndocc);
    D = C_occ * C_occ.transpose();

     F = H;

    // F += compute_2body_fock_general(
    //     shells, D, shells, true /* SOAD_D_is_shelldiagonal */,
    //     std::numeric_limits<double>::epsilon()  // this is cheap, no reason
    //                                             // to be cheaper
    //     );

    F+=    compute_2body_fock(shells, D, 1e-8, SchwarzK);


    Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> gen_eig_solver1(F, S);
    eps = gen_eig_solver1.eigenvalues();
    C = gen_eig_solver1.eigenvectors();
    // compute density, D = C(occ) . C(occ)T
    C_occ = C.leftCols(ndocc);
    D = C_occ * C_occ.transpose();

  } else {  // SOAD as the guess density

    auto D_minbs = compute_soad(atoms);  // compute guess in minimal basis
    BasisSet minbs("STO-3G", atoms);

    std::cout <<
    "\nProjecting minimal basis SOAD onto basis set specified (" << basis << ")\n";

    auto F = H;

    F += compute_2body_fock_general(
        shells, D_minbs, minbs, true /* SOAD_D_is_shelldiagonal */,
        std::numeric_limits<double>::epsilon()  // this is cheap, no reason
                                                // to be cheaper
        );

    // Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> gen_eig_solver(F, S);
    // auto eps = gen_eig_solver.eigenvalues();
    // auto C = gen_eig_solver.eigenvectors();

    // solve F C = e S C by (conditioned) transformation to F' C' = e C',
    // where
    // F' = X.transpose() . F . X; the original C is obtained as C = X . C'
    Eigen::SelfAdjointEigenSolver<Matrix> eig_solver(X.transpose() * F * X);
    auto C = X * eig_solver.eigenvectors();

    // compute density, D = C(occ) . C(occ)T
    auto C_occ = C.leftCols(ndocc);
    D = C_occ * C_occ.transpose();

  }

  hf_t2 = std::chrono::high_resolution_clock::now();
  hf_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

  std::cout << "\nTime taken to compute initial guess: " << hf_time << " secs\n";

//  cout << "\n\tInitial Density Matrix:\n";
//  cout << D << endl;


  /*** =========================== ***/
  /*** main iterative loop         ***/
  /*** =========================== ***/

  //bool simple_convergence = false;
  //double alpha = 0.5;
  auto iter = 0;
  auto rmsd = 1.0;
  auto ediff = 0.0;
  auto ehf = 0.0;
//  Matrix C;
//  Matrix F;
  Matrix eps;
  Matrix F_old;
// Chao
  std::ofstream resultsfile, hmatfile, smatfile;
  if (rank == 0) 
     cout << " matrix dimension = " << S.rows() << " ndocc = " << ndocc << endl; 

  int idiis = 0;
  std::vector<Matrix> diis_hist;
  std::vector<Matrix> fock_hist;

  // S^-1/2
  Matrix Sp12 = S.sqrt();
  Matrix Sm12 = Sp12.inverse();

  std::cout << "\n\n";
  std::cout << " Hartree-Fock iterations" << std::endl;
  std::cout << std::string(70, '-') << std::endl;
  std::cout <<
      " Iter     Energy            E-Diff            RMSD            Time" 
          << std::endl;
  std::cout << std::string(70, '-') << std::endl;
  std::cout << std::fixed << std::setprecision(2);

  do {
        // Scheduler sch{ec};
    const auto loop_start = std::chrono::high_resolution_clock::now();
    ++iter;

    // Save a copy of the energy and the density
    auto ehf_last = ehf;
    auto D_last = D;

    // build a new Fock matrix
    //auto F = H;
    hf_t1 = std::chrono::high_resolution_clock::now();

    //const auto precision_F = std::min(
    //    std::min(1e-3 / XtX_condition_number, 1e-7),
    //    std::max(rmsd / 1e4, std::numeric_limits<double>::epsilon()));

    //Matrix Ftmp = compute_2body_fock(shells, D, precision_F, SchwarzK);
    Matrix Ftmp = compute_2body_fock(shells, D, tol_int, SchwarzK);

    hf_t2 = std::chrono::high_resolution_clock::now();
    hf_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

    if(debug) std::cout << "2BF:" << hf_time << "s, ";

    hf_t1 = std::chrono::high_resolution_clock::now();

    F = H;
    F += Ftmp;

    hf_t2 = std::chrono::high_resolution_clock::now();
    hf_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

    if(debug) std::cout << "F=H+2BF:" << hf_time << "s, ";


    //  if (iter>1 && simple_convergence) {
    //    F = alpha * F + (1.0-alpha)*F_old;
    //  }

    hf_t1 = std::chrono::high_resolution_clock::now();

    //  Eigen::EigenSolver<Matrix> sm12_diag(Sm12);
    //  Eigen::EigenSolver<Matrix> sp12_diag(Sp12);

     Matrix FSm12 = F * Sm12;
     Matrix Sp12D = Sp12 * D_last;
     Matrix SpFS  = Sp12D * FSm12;

     // Assemble: S^(-1/2)*F*D*S^(1/2) - S^(1/2)*D*F*S^(-1/2)
     Matrix err_mat = SpFS.transpose() - SpFS;
    //  Matrix err_mat = (Sm12 * F * D_last * Sp12) - (Sp12 * D_last * F * Sm12);

    //  if(iter <= 3 || simple_convergence) { cout << err_mat << endl; }

    hf_t2 = std::chrono::high_resolution_clock::now();
    hf_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
    if(debug) std::cout << "err_mat:" << hf_time << "s, ";    

    hf_t1 = std::chrono::high_resolution_clock::now();

    if(iter > 1) {
      ++idiis;
      diis(F, err_mat, D_last, iter, max_hist, idiis, diis_hist,
          fock_hist);
    }

    hf_t2 = std::chrono::high_resolution_clock::now();
    hf_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

    if(debug) std::cout << "diis:" << hf_time << "s, ";    


    hf_t1 = std::chrono::high_resolution_clock::now();

    // solve F C = e S C
    // Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> gen_eig_solver(F, S);
    // //auto
    // eps = gen_eig_solver.eigenvalues();
    // C = gen_eig_solver.eigenvectors();
    
    // solve F C = e S C by (conditioned) transformation to F' C' = e C',
    // where
    // F' = X.transpose() . F . X; the original C is obtained as C = X . C'
//Chao
//    Eigen::SelfAdjointEigenSolver<Matrix> eig_solver(X.transpose() * F *
//                                                      X);
//    //eps = eig_solver.eigenvalues();
//    C = X * eig_solver.eigenvectors();
    if (rank == 0) {
      std::string hmatfname = "hmat" + std::to_string(iter) + ".txt";
      hmatfile.open(hmatfname);
      cout << "write H matrix:" << endl;
      hmatfile << F;
      hmatfile.close();
    }
    hsdiag(comm, iter, F, S, ndocc, eps, C);

    // compute density, D = C(occ) . C(occ)T
    auto C_occ = C.leftCols(ndocc);
    D = C_occ * C_occ.transpose();

    hf_t2 = std::chrono::high_resolution_clock::now();
    hf_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
    if(debug) std::cout << "eigen_solve:" << hf_time << "s, "; 

    hf_t1 = std::chrono::high_resolution_clock::now();
    // compute HF energy
    ehf = 0.0;
    for (size_t i = 0; i < nao; i++)
      for (size_t j = 0; j < nao; j++)
        ehf += D(i, j) * (H(i, j) + F(i, j));

    // compute difference with last iteration
    ediff = ehf - ehf_last;
    rmsd = (D - D_last).norm();

    hf_t2 = std::chrono::high_resolution_clock::now();
    hf_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

    if(debug) std::cout << "HF-Energy:" << hf_time << "s\n";    

    const auto loop_stop = std::chrono::high_resolution_clock::now();
    const auto loop_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((loop_stop - loop_start)).count();

    // cout << "iter, ehf, ediff, rmsd = " << iter << "," << ehf <<", " << ediff <<  "," <<rmsd << "\n";
    std::cout << std::setw(5) << iter << "  " << std::setw(14);
    std::cout << std::fixed << std::setprecision(10) << ehf + enuc;
    std::cout << ' ' << std::setw(16)  << ediff;
    std::cout << ' ' << std::setw(15)  << rmsd << ' ';
    std::cout << std::fixed << std::setprecision(2);
    std::cout << ' ' << std::setw(12)  << loop_time << ' ' << "\n";    

   if(iter > maxiter) {
     std::cerr << "HF Does not converge!!!\n";
     exit(0);
   }

  //  if(simple_convergence) F_old = F;

  } while (((fabs(ediff) > conve) || (fabs(rmsd) > convd)));

  std::cout.precision(15);
  printf("\n** Hartree-Fock energy = %20.12f\n", ehf + enuc);

  // cout << "\n** Eigen Values:\n";
  // cout << eps << endl;

  return std::make_tuple(ndocc,nao,ehf+enuc,shells);
}

void diis(Matrix& F, Matrix& err_mat, Matrix& D_last, int iter, int max_hist, 
         int ndiis, std::vector<Matrix> &diis_hist,std::vector<Matrix> &fock_hist) {
      using Vector =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
 
  // const int epos = ((ndiis-1) % max_hist) + 1;
  if(ndiis >  max_hist) { 

  std::vector<double> max_err(diis_hist.size());
  // std::cout << "max_err vector: \n";
  for (auto i=0; i<diis_hist.size(); i++){
    max_err[i] = diis_hist[i].norm();
    // cout << max_err[i] << ",";
  }

  auto maxe = std::distance(max_err.begin(), 
            std::max_element(max_err.begin(),max_err.end()));

  // std::cout << "\nmax index: " << maxe << endl;

    diis_hist.erase(diis_hist.begin()+maxe);
    fock_hist.erase(fock_hist.begin()+maxe);
    
  }
  diis_hist.push_back(err_mat);
  fock_hist.push_back(F);

  // --------------------- Construct error metric -------------------------------
  const int idim = std::min(ndiis, max_hist);
  Matrix A = Matrix::Zero(idim + 1, idim + 1);
  Vector b = Vector::Zero(idim + 1, 1);

  for(int i = 0; i < idim; i++) {
      for(int j = i; j < idim; j++) {
          A(i,j) = (diis_hist[i].transpose() * diis_hist[j]).trace();
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
    for(int j = 0; j < idim; j++) {
        F += x(j, 0) * fock_hist[j];
    }

    // cout << "-----------iter:" << iter << "--------------\n";
    // cout << err_mat << endl;
 
}



#endif //TAMM_TESTS_HF_HPP_
