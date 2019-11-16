
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

#include <filesystem>
namespace fs = std::filesystem;

#define SCF_THROTTLE_RESOURCES 1

std::tuple<int, int, double, libint2::BasisSet, std::vector<size_t>, Tensor<double>, Tensor<double>, TiledIndexSpace, TiledIndexSpace, bool> 
    hartree_fock(ExecutionContext &exc, const string filename,std::vector<libint2::Atom> atoms, OptionsMap options_map) {

    using libint2::Atom;
    using libint2::Engine;
    using libint2::Operator;
    using libint2::Shell;
    using libint2::BasisSet;

    /*** =========================== ***/
    /*** initialize molecule         ***/
    /*** =========================== ***/

    scf_options = options_map.scf_options;

    std::string basis = scf_options.basis;
    int maxiter = scf_options.maxiter;
    double conve = scf_options.conve;
    double convd = scf_options.convd;
    // int max_hist = scf_options.diis_hist; 
    auto debug = scf_options.debug;
    auto restart = scf_options.restart;

    //TODO:adjust tol_int as needed
    // double tol_int = scf_options.tol_int;
    // tol_int = std::min(tol_int, 0.01 * conve);

    auto hf_t1 = std::chrono::high_resolution_clock::now();

    // const auto ndocc1 = ndocc; //TODO: uhf

    // initializes the Libint integrals library ... now ready to compute
    libint2::initialize(false);

    /*** =========================== ***/
    /*** create basis set            ***/
    /*** =========================== ***/

    // LIBINT_INSTALL_DIR/share/libint/2.4.0-beta.1/basis
    libint2::BasisSet shells(std::string(basis), atoms);
    // auto shells = make_sto3g_basis(atoms);
    const size_t N = nbasis(shells);
    size_t nao = N;
    auto rank = exc.pg().rank();
    auto nnodes = GA_Cluster_nnodes();

    #if SCF_THROTTLE_RESOURCES
      auto [hf_nnodes,ppn,hf_nranks] = get_hf_nranks(N);
      int ranks[hf_nranks];
      for (int i = 0; i < hf_nranks; i++) ranks[i] = i;    
      auto gcomm = exc.pg().comm();
      MPI_Group wgroup;
      MPI_Comm_group(gcomm,&wgroup);
      MPI_Group hfgroup;
      MPI_Group_incl(wgroup,hf_nranks,ranks,&hfgroup);
      MPI_Comm hf_comm;
      MPI_Comm_create(gcomm,hfgroup,&hf_comm);
    #endif

    
    if(rank == 0) {
      cout << "\nNumber of nodes, mpi ranks per node provided: " << nnodes << ", " << GA_Cluster_nprocs(0) << endl;
      #if SCF_THROTTLE_RESOURCES
        cout << "Number of nodes, mpi ranks per node used for SCF calculation: " << hf_nnodes << ", " << ppn << endl;
      #endif
      scf_options.print();
    }

    auto [ndocc, enuc]  = compute_NRE(exc, atoms);

    if(rank==0) std::cout << "#electrons = " << 2*ndocc << std::endl;
    
    // compute OBS non-negligible shell-pair list
    compute_shellpair_list(exc, shells);

    if(rank==0) cout << "\nNumber of basis functions: " << N << endl;

    //DENSITY FITTING
    const auto dfbasisname = scf_options.dfbasis;
    bool do_density_fitting = false;
    if(!dfbasisname.empty()) do_density_fitting = true;

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
                  << endl;
      }
      #endif
      
    }
    std::unique_ptr<DFFockEngine> dffockengine(
        do_density_fitting ? new DFFockEngine(shells, dfbs) : nullptr);


    tamm::Tile tile_size = scf_options.AO_tilesize; //TODO
    IndexSpace AO{range(0, N)};
    std::tie(shell_tile_map, AO_tiles, AO_opttiles) = compute_AO_tiles(exc, shells);
    tAO = {AO, AO_opttiles};
    tAOt = {AO, AO_tiles};
    std::tie(mu, nu, ku) = tAO.labels<3>("all");
    std::tie(mup, nup, kup) = tAOt.labels<3>("all");

    auto hf_t2 = std::chrono::high_resolution_clock::now();

    double hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
    if(rank == 0) std::cout << "\nTime for initial setup: " << hf_time << " secs" << endl;

    Matrix C, F;
    TensorType ehf           = 0.0;

    exc.pg().barrier();

    std::string scf_files_prefix = getfilename(filename) +
        "." + scf_options.basis;
    std::string scfstatusfile = scf_files_prefix + ".scfstatus";
    const bool scf_conv = restart && fs::exists(scfstatusfile);

    #if SCF_THROTTLE_RESOURCES
  if (rank < hf_nranks) {

      int hrank;
      EXPECTS(hf_comm != MPI_COMM_NULL);
      MPI_Comm_rank(hf_comm,&hrank);
      EXPECTS(rank==hrank);
      // cout << "rank,hrank = " << rank << "," << hrank << endl;

      ProcGroup pg{hf_comm};
      auto mgr = MemoryManagerGA::create_coll(pg);
      Distribution_NW distribution;
      RuntimeEngine re;
      ExecutionContext ec{pg, &distribution, mgr, &re};

    #else 
      ExecutionContext& ec = exc;
    #endif

    ProcGroup pg_l{MPI_COMM_SELF};
    auto mgr_l = MemoryManagerLocal::create_coll(pg_l);
    Distribution_NW distribution_l;
    RuntimeEngine re_l;
    ExecutionContext ec_l{pg_l, &distribution_l, mgr_l, &re_l};

    #ifdef SCALAPACK

      auto blacs_setup_st = std::chrono::high_resolution_clock::now();
      // Sanity checks
      int scalapack_nranks = 
        scf_options.scalapack_np_row *
        scf_options.scalapack_np_col;

      // XXX: This should be for hf_comm
      int world_size;
      MPI_Comm_size( ec.pg().comm(), &world_size );
      assert( world_size >= scalapack_nranks );

      if( not scalapack_nranks ) scalapack_nranks = world_size;
      std::vector<int> scalapack_ranks( scalapack_nranks );
      std::iota( scalapack_ranks.begin(), scalapack_ranks.end(), 0 );

      MPI_Group world_group, scalapack_group;
      MPI_Comm scalapack_comm;
      MPI_Comm_group( ec.pg().comm(), &world_group );
      MPI_Group_incl( world_group, scalapack_nranks, scalapack_ranks.data(), &scalapack_group );
      MPI_Comm_create( ec.pg().comm(), scalapack_group, &scalapack_comm );
      
      

      // Define a BLACS grid
      const CB_INT MB = scf_options.scalapack_nb; 
      const CB_INT NPR = scf_options.scalapack_np_row;
      const CB_INT NPC = scf_options.scalapack_np_col;
      std::unique_ptr<CXXBLACS::BlacsGrid> blacs_grid = 
        scalapack_comm == MPI_COMM_NULL ? nullptr :
        std::make_unique<CXXBLACS::BlacsGrid>( scalapack_comm, MB, MB, NPR, NPC );

      auto blacs_setup_en = std::chrono::high_resolution_clock::now();

      std::chrono::duration<double> blacs_time = blacs_setup_en - blacs_setup_st;
      
      if(rank == 0) std::cout << "\nTime for BLACS setup: " << blacs_time.count() << " secs\n";

      if(debug and blacs_grid) blacs_grid->printCoord( std::cout );
    #endif

    /*** =========================== ***/
    /*** compute 1-e integrals       ***/
    /*** =========================== ***/

    auto [H, S, H1, S1] = compute_hamiltonian<TensorType>(ec,atoms,shells,shell_tile_map,AO_tiles);

    //auto H_down = H; //TODO

    /*** =========================== ***/
    /*** build initial-guess density ***/
    /*** =========================== ***/

    Matrix X = compute_orthogonalizer(ec, S);

    // pre-compute data for Schwarz bounds
    auto SchwarzK = compute_schwarz_ints<>(shells);

    Matrix D;

    hf_t1 = std::chrono::high_resolution_clock::now();

    if (restart)
        scf_restart(ec, N, filename, ndocc, C, D);
    else   // SOAD as the guess density
      compute_initial_guess<TensorType>(ec, ndocc, atoms, shells, basis, X, H, C, C_occ, D);

    H.resize(0,0);
    hf_t2 = std::chrono::high_resolution_clock::now();
    hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

    ec.pg().barrier();
    if(rank == 0) std::cout << "Total Time to compute initial guess: " << hf_time << " secs" << endl;


    hf_t1 = std::chrono::high_resolution_clock::now();
    /*** =========================== ***/
    /*** main iterative loop         ***/
    /*** =========================== ***/

    auto iter          = 0;
    auto rmsd          = 1.0;
    auto ediff         = 0.0;
    // auto ehf           = 0.0;
    auto is_conv       = true;

    // int idiis                     = 0;
    std::vector<tamm::Tensor<TensorType>> diis_hist;
    std::vector<tamm::Tensor<TensorType>> fock_hist;

    Tensor<TensorType> ehf_tmp{tAO, tAO};
    Tensor<TensorType> ehf_tamm{};

    Tensor<TensorType> F1{tAO, tAO};
    Tensor<TensorType> F1tmp1{tAO, tAO};
    Tensor<TensorType> F1tmp{tAOt, tAOt}; //not allocated
    Tensor<TensorType>::allocate(&ec, F1, F1tmp1, ehf_tmp, ehf_tamm);

    Tensor<TensorType> D_tamm{tAO, tAO};
    Tensor<TensorType> D_diff{tAO, tAO};
    Tensor<TensorType> D_last_tamm{tAO, tAO};
    Tensor<TensorType>::allocate(&ec, D_tamm, D_diff, D_last_tamm);

    // FSm12,Sp12D,SpFS
    Tensor<TensorType> FD_tamm{tAO, tAO}; 
    Tensor<TensorType> FDS_tamm{tAO, tAO};
    Tensor<TensorType>::allocate(&ec, FD_tamm, FDS_tamm);//,err_mat_tamm);

    hf_t2 = std::chrono::high_resolution_clock::now();
    hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
    //ec.pg().barrier();
    if(rank == 0 && debug) std::cout << "\nTime to setup tensors for iterative loop: " << hf_time << " secs" << endl;

    eigen_to_tamm_tensor(D_tamm,D);
    // Matrix err_mat = Matrix::Zero(N,N);

    F = Matrix::Zero(N,N);
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

    //DF basis
    IndexSpace dfCocc{range(0,ndocc)}; 
    tdfCocc = {dfCocc,tile_size};
    std::tie(dCocc_til) = tdfCocc.labels<1>("all");


    if(do_density_fitting){

      ndf = dfbs.nbf();
      dfAO = IndexSpace{range(0, ndf)};
      std::tie(df_shell_tile_map, dfAO_tiles, dfAO_opttiles) = compute_AO_tiles(ec,dfbs);
      // if(rank==0 && debug) 
      // cout << "Number of dfAO tiles = " << dfAO_tiles.size() << endl;

      tdfAO=TiledIndexSpace{dfAO, dfAO_opttiles};
      tdfAOt=TiledIndexSpace{dfAO, dfAO_tiles};
      std::tie(d_mu, d_nu, d_ku) = tdfAO.labels<3>("all");
      std::tie(d_mup, d_nup, d_kup) = tdfAOt.labels<3>("all");
    }

    if(do_density_fitting) {
      xyK_tamm = Tensor<TensorType>{tAO, tAO, tdfAO}; //n,n,ndf
      C_occ_tamm = Tensor<TensorType>{tAO,tdfCocc}; //n,nocc
      Tensor<TensorType>::allocate(&ec, xyK_tamm, C_occ_tamm);
      // Tensor<TensorType>::allocate(&ec_l,C_occ_tamm);
    }
    //df basis

    if(rank == 0) {
        std::cout << "\n\n";
        std::cout << " Hartree-Fock iterations" << endl;
        std::cout << std::string(70, '-') << endl;
        std::string  sph = " Iter     Energy            E-Diff            RMSD            Time";
        if(scf_conv) sph = " Iter     Energy            E-Diff            Time";
        std::cout << sph << endl;
        std::cout << std::string(70, '-') << endl;
    }

    std::cout << std::fixed << std::setprecision(2);

    do {
        // Scheduler sch{ec};
        const auto loop_start = std::chrono::high_resolution_clock::now();
        ++iter;

        // Save a copy of the energy and the density
        auto ehf_last = ehf;
        // auto D_last   = D;

        Scheduler{ec}
           (F1tmp1() = 0)
           (D_last_tamm(mu,nu) = D_tamm(mu,nu)).execute();

        // build a new Fock matrix
        // F           = H;

        compute_2bf(ec, obs, do_schwarz_screen, shell2bf, SchwarzK, G, D,
                    F1tmp, F1tmp1, max_nprim4,shells,do_density_fitting);

        std::tie(ehf,rmsd) = scf_iter_body<TensorType>(ec, 
    #ifdef SCALAPACK
                        blacs_grid.get(),
    #endif
                        iter, ndocc, X, F, C, C_occ, D,
                        S1, F1, H1, F1tmp1,FD_tamm, FDS_tamm, D_tamm, D_last_tamm, D_diff,
                        ehf_tmp, ehf_tamm, diis_hist, fock_hist, scf_conv);

        // compute difference with last iteration
        ediff = ehf - ehf_last;

        const auto loop_stop = std::chrono::high_resolution_clock::now();
        const auto loop_time =
        std::chrono::duration_cast<std::chrono::duration<double>>((loop_stop - loop_start)).count();

        if(rank == 0) {
            std::cout << std::setw(5) << iter << "  " << std::setw(14);
            std::cout << std::fixed << std::setprecision(10) << ehf + enuc;
            std::cout << ' ' << std::setw(16)  << ediff;
            if(!scf_conv) std::cout << ' ' << std::setw(15)  << rmsd << ' ';
            std::cout << std::fixed << std::setprecision(2);
            std::cout << ' ' << std::setw(12)  << loop_time << ' ' << endl;
        }

        // if(rank==0)
        //   writeC(C,filename,options_map);

        if(iter > maxiter) {                
            is_conv = false;
            break;
        }

        if(scf_conv) break;

    } while ( (fabs(ediff) > conve) || (fabs(rmsd) > convd) );

    // ec.pg().barrier(); 
    if(rank == 0) {
        std::cout.precision(13);
        if (is_conv)
            cout << "\n** Hartree-Fock energy = " << ehf + enuc << endl;
        else {
            cout << endl << std::string(50, '*') << endl;
            cout << std::string(10, ' ') << 
                    "ERROR: Hartree-Fock calculation does not converge!!!" << endl;
            cout << std::string(50, '*') << endl;
        }        
    }

    for (auto x: diis_hist) Tensor<TensorType>::deallocate(x);
    for (auto x: fock_hist) Tensor<TensorType>::deallocate(x);

    if(do_density_fitting) Tensor<TensorType>::deallocate(xyK_tamm,C_occ_tamm);
    Tensor<TensorType>::deallocate(H1, S1, D_tamm, ehf_tmp, ehf_tamm); 
    Tensor<TensorType>::deallocate(F1tmp1, D_last_tamm, D_diff, FD_tamm, FDS_tamm);

    if(rank==0 && !scf_conv) {
     cout << "writing orbitals to file... ";
     writeC(C,filename,scf_files_prefix);
     cout << "done." << endl;
    }

    if(!is_conv) {
      ec.pg().barrier();
      nwx_terminate("Please check SCF input parameters");
    }
    else{
        if(rank==0 && !scf_conv){
          std::ofstream out(scfstatusfile, std::ios::out);
          if(!out) cerr << "Error opening file " << scfstatusfile << endl;
          out << 1 << std::endl;
          out.close();
        }    
    }

      if(rank == 0) tamm_to_eigen_tensor(F1,F);

      Tensor<TensorType>::deallocate(F1); //deallocate using ec

      #if SCF_THROTTLE_RESOURCES
      ec.flush_and_sync();
      MemoryManagerGA::destroy_coll(mgr);
      #endif

      ec_l.flush_and_sync();
      MemoryManagerLocal::destroy_coll(mgr_l);      

      #ifdef SCALAPACK

      // Free up created comms / groups
      MPI_Comm_free( &scalapack_comm );
      MPI_Group_free( &scalapack_group );
      MPI_Group_free( &world_group );

      #endif
    
    #if SCF_THROTTLE_RESOURCES

    } //end scaled down process group

      // MPI_Group_free(&wgroup);
      // MPI_Group_free(&hfgroup);
      // MPI_Comm_free(&hf_comm);
    #endif

    //C,F1 is not allocated for ranks > hf_nranks 
    exc.pg().barrier(); 

    // GA_Brdcst(&ehf,sizeof(TensorType),0);
    MPI_Bcast(&ehf,1,mpi_type<TensorType>(),0,exc.pg().comm());
    Tensor<TensorType> C_tamm{tAO,tAO};
    Tensor<TensorType> F_tamm{tAO,tAO};
    Tensor<TensorType>::allocate(&exc,C_tamm,F_tamm);
    if (rank == 0) {
      eigen_to_tamm_tensor(C_tamm,C);
      eigen_to_tamm_tensor(F_tamm,F);
    }

    exc.pg().barrier();

    //F, C are not deallocated.
    return std::make_tuple(ndocc, nao, ehf + enuc, shells, shell_tile_map, C_tamm, F_tamm, tAO, tAOt, scf_conv);
}



#endif // TAMM_TESTS_HF_TAMM_HPP_
