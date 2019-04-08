
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

#include "hf_tamm_common.hpp"


std::tuple<int, int, double, libint2::BasisSet, std::vector<size_t>, Tensor<double>, Tensor<double>, TiledIndexSpace, TiledIndexSpace> 
    hartree_fock(ExecutionContext &exc, const string filename,std::vector<libint2::Atom> atoms, OptionsMap options_map) {

    using libint2::Atom;
    using libint2::Engine;
    using libint2::Operator;
    using libint2::Shell;
    using libint2::BasisSet;

    ExecutionContext *ec = &exc;

    /*** =========================== ***/
    /*** initialize molecule         ***/
    /*** =========================== ***/

    auto rank = ec->pg().rank();
    scf_options = options_map.scf_options;

    if(rank == 0) {
      cout << "\nNumber of GA ranks: " << GA_Nnodes() << endl;
      scf_options.print();
    }

    std::string basis = scf_options.basis;
    int maxiter = scf_options.maxiter;
    double conve = scf_options.conve;
    double convd = scf_options.convd;
    double tol_int = scf_options.tol_int;
    // int max_hist = scf_options.diis_hist; 
    auto debug = scf_options.debug;
    auto restart = scf_options.restart;

    // tol_int = std::min(tol_int, 0.01 * conve);

    auto hf_t1 = std::chrono::high_resolution_clock::now();

    auto [ndocc, enuc]  = compute_NRE(*ec, atoms);

    // const auto ndocc1 = ndocc; //TODO: uhf

    // initializes the Libint integrals library ... now ready to compute
    libint2::initialize(false);

    /*** =========================== ***/
    /*** create basis set            ***/
    /*** =========================== ***/

    // LIBINT_INSTALL_DIR/share/libint/2.4.0-beta.1/basis
    libint2::BasisSet shells(std::string(basis), atoms);
    // auto shells = make_sto3g_basis(atoms);

    // compute OBS non-negligible shell-pair list
    compute_shellpair_list(exc, shells);

    const size_t N = nbasis(shells);
    size_t nao = N;

    if(rank==0) cout << "\nNumber of basis functions: " << N << endl;

    //DENSITY FITTING
    const auto dfbasisname = scf_options.dfbasis;
    bool do_density_fitting = false;
    if(!dfbasisname.empty()) do_density_fitting = true;

    BasisSet dfbs;
    if (do_density_fitting) {
      dfbs = BasisSet(dfbasisname, atoms);
      if (rank==0) cout << "density-fitting basis set rank = " << dfbs.nbf() << endl;
      // compute DFBS non-negligible shell-pair list
      #if 0
      {
        //TODO: Doesn't work to screen - revisit
        std::tie(dfbs_shellpair_list, dfbs_shellpair_data) = compute_shellpairs(dfbs);
        size_t nsp = 0;
        for (auto& sp : dfbs_shellpair_list) {
          nsp += sp.second.size();
        }
        if(rank==0) std::cout << "# of {all,non-negligible} DFBS shell-pairs = {"
                  << dfbs.size() * (dfbs.size() + 1) / 2 << "," << nsp << "}"
                  << std::endl;
      }
      #endif
      
    }
    std::unique_ptr<DFFockEngine> dffockengine(
        do_density_fitting ? new DFFockEngine(shells, dfbs) : nullptr);


    tamm::Tile tile_size = scf_options.AO_tilesize; //TODO
    IndexSpace AO{range(0, N)};
    auto [shell_tile_map, AO_tiles, AO_opttiles] = compute_AO_tiles(exc, shells);
    tAO = {AO, AO_opttiles};
    tAOt = {AO, AO_tiles};
    auto [mu, nu, ku] = tAO.labels<3>("all");
    auto [mup, nup, kup] = tAOt.labels<3>("all");

    auto hf_t2 = std::chrono::high_resolution_clock::now();

    double hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
    if(rank == 0) std::cout << "\nTime for initial setup: " << hf_time << " secs\n";


    /*** =========================== ***/
    /*** compute 1-e integrals       ***/
    /*** =========================== ***/

    auto [H, S, H1, S1] = compute_hamiltonian<TensorType>(*ec,atoms,shells,shell_tile_map,AO_tiles);

    //auto H_down = H; //TODO

    /*** =========================== ***/
    /*** build initial-guess density ***/
    /*** =========================== ***/

    Matrix X = compute_orthogonalizer(exc, S);

    // pre-compute data for Schwarz bounds
    auto SchwarzK = compute_schwarz_ints<>(shells);


    const auto use_hcore_guess = false;  // use core Hamiltonian eigenstates to guess density?
    // set to true to match the result of versions 0, 1, and 2 of the code
    // HOWEVER !!! even for medium-size molecules hcore will usually fail !!!
    // thus set to false to use Superposition-Of-Atomic-Densities (SOAD) guess

    Matrix D;
    Matrix C;
    Matrix C_occ;
    Matrix F;

  //     Matrix C_down; //TODO: all are array of 2 vectors
  // Matrix D_down;
  // Matrix F_down;
  // Matrix C_occ_down;

    hf_t1 = std::chrono::high_resolution_clock::now();

    if (use_hcore_guess) 
      compute_hcore_guess(ndocc, shells, SchwarzK, H, S, F, C, C_occ, D);
    else if (restart)
        scf_restart(exc, N, filename, ndocc, C, D);
    else   // SOAD as the guess density
      compute_initial_guess<TensorType>(*ec, ndocc, atoms, shells, basis, X, H, C, C_occ, D);
    

    //     C_down = C;
    // D_down = D;
    // C_occ_down = C_occ;

    H.resize(0,0);
    hf_t2 = std::chrono::high_resolution_clock::now();
    hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

    ec->pg().barrier();
    if(rank == 0) std::cout << "Total Time to compute initial guess: " << hf_time << " secs\n";


    hf_t1 = std::chrono::high_resolution_clock::now();
    /*** =========================== ***/
    /*** main iterative loop         ***/
    /*** =========================== ***/

    auto iter          = 0;
    auto rmsd          = 1.0;
    auto ediff         = 0.0;
    auto ehf           = 0.0;
    auto is_conv       = true;

    // int idiis                     = 0;
    std::vector<tamm::Tensor<TensorType>> diis_hist;
    std::vector<tamm::Tensor<TensorType>> fock_hist;

    Tensor<TensorType> ehf_tmp{tAO, tAO};
    Tensor<TensorType> ehf_tamm{};

    Tensor<TensorType> F1{tAO, tAO};
    Tensor<TensorType> F1tmp1{tAO, tAO};
    Tensor<TensorType> F1tmp{tAOt, tAOt}; //not allocated
    Tensor<TensorType>::allocate(ec, F1, F1tmp1, ehf_tmp, ehf_tamm);

    Tensor<TensorType> D_tamm{tAO, tAO};
    Tensor<TensorType> D_diff{tAO, tAO};
    Tensor<TensorType> D_last_tamm{tAO, tAO};
    Tensor<TensorType>::allocate(ec, D_tamm, D_diff, D_last_tamm);

    // FSm12,Sp12D,SpFS
    Tensor<TensorType> FD_tamm{tAO, tAO}; 
    Tensor<TensorType> FDS_tamm{tAO, tAO};
    Tensor<TensorType>::allocate(ec, FD_tamm, FDS_tamm);//,err_mat_tamm);

    hf_t2 = std::chrono::high_resolution_clock::now();
    hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
    //ec->pg().barrier();
    if(rank == 0 && debug) std::cout << "\nTime to setup tensors for iterative loop: " << hf_time << " secs\n";

    eigen_to_tamm_tensor(D_tamm,D);
    // Matrix err_mat = Matrix::Zero(N,N);

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

    F.setZero(N,N);
    // double precision = tol_int;
    const libint2::BasisSet& obs = shells;
    // assert(N==obs.nbf());
    Matrix G = Matrix::Zero(N,N);
    const auto do_schwarz_screen = SchwarzK.cols() != 0 && SchwarzK.rows() != 0;
    // auto fock_precision = precision;
    // engine precision controls primitive truncation, assume worst-case scenario
    // (all primitive combinations add up constructively)
    auto max_nprim = obs.max_nprim();
    auto max_nprim4 = max_nprim * max_nprim * max_nprim * max_nprim;
    auto shell2bf = obs.shell2bf();
    // const auto nshells = obs.size();

    Scheduler{*ec}
        (D_diff(mu,nu) = D_tamm(mu,nu)).execute();

    //DF basis
    IndexSpace dfCocc{range(0,ndocc)}; 
    TiledIndexSpace tdfCocc{dfCocc,tile_size};
    auto [dCocc_til] = tdfCocc.labels<1>("all");

    decltype(nao) ndf;
    IndexSpace dfAO; 
    std::vector<Tile> dfAO_tiles;
    std::vector<Tile> dfAO_opttiles;
    std::vector<size_t> df_shell_tile_map;
    TiledIndexSpace tdfAO, tdfAOt;
    TiledIndexLabel d_mu,d_nu,d_ku;
    TiledIndexLabel d_mup, d_nup, d_kup;

    if(do_density_fitting){

      ndf = dfbs.nbf();
      dfAO = IndexSpace{range(0, ndf)};
      std::tie(df_shell_tile_map, dfAO_tiles, dfAO_opttiles) = compute_AO_tiles(exc,dfbs);
      // if(rank==0 && debug) 
      // cout << "Number of dfAO tiles = " << dfAO_tiles.size() << endl;

      tdfAO=TiledIndexSpace{dfAO, dfAO_opttiles};
      tdfAOt=TiledIndexSpace{dfAO, dfAO_tiles};
      std::tie(d_mu, d_nu, d_ku) = tdfAO.labels<3>("all");
      std::tie(d_mup, d_nup, d_kup) = tdfAOt.labels<3>("all");
    }

    bool is_3c_init = false;
    Tensor<TensorType> xyK_tamm; //n,n,ndf
    Tensor<TensorType> C_occ_tamm; //n,nocc
    if(do_density_fitting) {
      xyK_tamm = Tensor<TensorType>{tAO, tAO, tdfAO}; //n,n,ndf
      C_occ_tamm = Tensor<TensorType>{tAO,tdfCocc}; //n,nocc
      Tensor<TensorType>::allocate(ec, xyK_tamm,C_occ_tamm);
    }
    //df basis

    do {
        // Scheduler sch{ec};
        const auto loop_start = std::chrono::high_resolution_clock::now();
        ++iter;

        // Save a copy of the energy and the density
        auto ehf_last = ehf;
        // auto D_last   = D;

        Scheduler{*ec}
           (F1tmp1() = 0)
           (D_last_tamm(mu,nu) = D_tamm(mu,nu)).execute();

        // build a new Fock matrix
        // F           = H;

        compute_2bf(*ec, obs, do_schwarz_screen, shell2bf, SchwarzK, 
                    G, D, F1tmp, F1tmp1, max_nprim4);

        std::tie(ehf,rmsd) = scf_iter_body<TensorType>(*ec, iter, ndocc, X, F, C, C_occ, D,
                        S1, F1, H1, F1tmp1,FD_tamm, FDS_tamm, D_tamm, D_last_tamm, D_diff,
                        ehf_tmp, ehf_tamm, diis_hist, fock_hist);

        // compute difference with last iteration
        ediff = ehf - ehf_last;

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

        // if(rank==0)
        //   writeC(C,filename,options_map);

        if(iter > maxiter) {                
            is_conv = false;
            break;
        }

    } while (((fabs(ediff) > conve) || (fabs(rmsd) > convd)));

    // ec->pg().barrier(); 
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

    if(do_density_fitting) Tensor<TensorType>::deallocate(xyK_tamm,C_occ_tamm);
    Tensor<TensorType>::deallocate(H1, S1, D_tamm, ehf_tmp, ehf_tamm); 
    Tensor<TensorType>::deallocate(F1tmp1, D_last_tamm, D_diff, FD_tamm, FDS_tamm);

    if(rank==0 && !restart) {
     cout << "writing orbitals to file... ";
     writeC(C,filename,options_map);
     cout << "done.\n";
    }

    Tensor<TensorType> C_tamm{tAO,tAO};
    Tensor<TensorType>::allocate(ec,C_tamm);
    eigen_to_tamm_tensor(C_tamm,C);
    ec->pg().barrier();

    //F1, C are not deallocated.
    return std::make_tuple(ndocc, nao, ehf + enuc, shells, shell_tile_map, C_tamm, F1, tAO, tAOt);
}



#endif // TAMM_TESTS_HF_TAMM_HPP_
