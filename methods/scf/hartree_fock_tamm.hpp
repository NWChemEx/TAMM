
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

std::tuple<SystemData, double, libint2::BasisSet, std::vector<size_t>, Tensor<double>, Tensor<double>, TiledIndexSpace, TiledIndexSpace, bool> 
    hartree_fock(ExecutionContext &exc, const string filename,std::vector<libint2::Atom> atoms, OptionsMap options_map) {

    using libint2::Atom;
    using libint2::Engine;
    using libint2::Operator;
    using libint2::Shell;
    using libint2::BasisSet;

    /*** =========================== ***/
    /*** initialize molecule         ***/
    /*** =========================== ***/

    SystemData sys_data{options_map, options_map.scf_options.scf_type};

    SCFOptions scf_options = sys_data.options_map.scf_options;

    std::string basis = scf_options.basis;
    int maxiter = scf_options.maxiter;
    double conve = scf_options.conve;
    double convd = scf_options.convd;
    bool debug = scf_options.debug;
    auto restart = scf_options.restart;
    bool is_spherical = (scf_options.sphcart == "spherical");
    auto iter          = 0;

    auto hf_t1 = std::chrono::high_resolution_clock::now();

    // initializes the Libint integrals library ... now ready to compute
    libint2::initialize(false);
    libint2::Shell::do_enforce_unit_normalization(false);

    /*** =========================== ***/
    /*** create basis set            ***/
    /*** =========================== ***/

    // LIBINT_INSTALL_DIR/share/libint/2.4.0-beta.1/basis
    libint2::BasisSet shells(std::string(basis), atoms);
    if(is_spherical) shells.set_pure(true);
    else shells.set_pure(false);  // use cartesian gaussians

    // auto shells = make_sto3g_basis(atoms);
    const size_t N = nbasis(shells);
    auto rank = exc.pg().rank();
    auto nnodes = GA_Cluster_nnodes();

    sys_data.nbf = N;
    sys_data.nbf_orig = N;

    const bool is_rhf = (sys_data.scf_type == sys_data.SCFType::rhf);
    const bool is_uhf = (sys_data.scf_type == sys_data.SCFType::uhf);
    // const bool is_rohf = (sys_data.scf_type == sys_data.SCFType::rohf);    

    #if SCF_THROTTLE_RESOURCES
      auto [hf_nnodes,ppn,hf_nranks] = get_hf_nranks(N);
      if (scf_options.nnodes > hf_nnodes) {
        hf_nnodes = scf_options.nnodes;
        hf_nranks = hf_nnodes * ppn;
      }
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

    auto [ndocc, enuc]  = compute_NRE(exc, atoms, sys_data.focc);

    const auto nelectrons = sys_data.focc * ndocc;
    sys_data.nelectrons = nelectrons;
    EXPECTS ( (nelectrons + scf_options.multiplicity - 1) % 2 == 0 );  

    sys_data.nelectrons_alpha = (nelectrons - scf_options.multiplicity + 1)/2; // this is ndocc
    sys_data.nelectrons_beta = nelectrons - sys_data.nelectrons_alpha;
    
    if(rank==0) {
      std::cout << std::endl << "Total number of electrons = " << nelectrons << std::endl;
      std::cout << std::endl << "Number of basis functions = " << N << std::endl;
      std::cout << std::endl << "Nuclear repulsion energy = " << std::setprecision(15) << enuc << std::endl << std::endl; 
    }

    // compute OBS non-negligible shell-pair list
    compute_shellpair_list(exc, shells);    

    //DENSITY FITTING
    const auto dfbasisname = scf_options.dfbasis;
    bool do_density_fitting = false;
    if(!dfbasisname.empty()) do_density_fitting = true;

    if (do_density_fitting) {
      dfbs = BasisSet(dfbasisname, atoms);
      if(is_spherical) dfbs.set_pure(true);
      else dfbs.set_pure(false);  // use cartesian gaussians

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
    std::tie(shell_tile_map, AO_tiles, AO_opttiles) = compute_AO_tiles(exc, sys_data, shells);
    tAO = {AO, AO_opttiles};
    tAOt = {AO, AO_tiles};
    std::tie(mu, nu, ku) = tAO.labels<3>("all");
    std::tie(mup, nup, kup) = tAOt.labels<3>("all");

    auto hf_t2 = std::chrono::high_resolution_clock::now();

    double hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
    if(rank == 0) std::cout << "\nTime for initial setup: " << hf_time << " secs" << endl;

    double ehf           = 0.0;

    exc.pg().barrier();

    EigenTensors etensors;

    std::string scf_files_prefix = getfilename(filename) +
        "." + scf_options.basis;
    // std::string scfstatusfile = scf_files_prefix + ".scfstatus";
    const bool scf_conv = restart && scf_options.noscf; 

    const bool molden_exists = !scf_options.moldenfile.empty();

    if(molden_exists) {
      size_t nmo = N-scf_options.n_lindep;
      if(is_uhf) {
        nmo = 2*(N-scf_options.n_lindep);
      }

      std::vector<TensorType> evl_sorted(nmo); //FIXME: not used/broadcasted
      etensors.C.setZero(N,nmo);
      int n_occ_alpha=0, n_occ_beta=0, n_vir_alpha=0, n_vir_beta=0;

      bool molden_file_valid=std::filesystem::exists(scf_options.moldenfile);
      if(rank == 0) {
        cout << "\nReading from molden file provided ..." << endl;
        if(molden_file_valid) {
          // const size_t n_lindep = scf_options.n_lindep;
          std::tie(n_occ_alpha,n_vir_alpha,n_occ_beta,n_vir_beta) = read_molden<TensorType>(scf_options,evl_sorted,etensors.C,atoms.size());
        }
      }

      std::vector<TensorType> Cbuf(N*nmo);
      TensorType *Cbufp = &Cbuf[0];
      if(rank==0) Eigen::Map<Matrix>(Cbufp,N,nmo) = etensors.C;
      MPI_Bcast(Cbufp,N*nmo,mpi_type<TensorType>(),0,exc.pg().comm());
      etensors.C = Eigen::Map<Matrix>(Cbufp,N,nmo);
      
      if(!molden_file_valid) nwx_terminate("ERROR: Cannot open moldenfile provided: " + scf_options.moldenfile);

    }

    #if SCF_THROTTLE_RESOURCES
  if (rank < hf_nranks) {

      int hrank;
      EXPECTS(hf_comm != MPI_COMM_NULL);
      MPI_Comm_rank(hf_comm,&hrank);
      EXPECTS(rank==hrank);
      // cout << "rank,hrank = " << rank << "," << hrank << endl;

      ProcGroup pg = ProcGroup::create_coll(hf_comm);
      auto mgr = MemoryManagerGA::create_coll(pg);
      Distribution_NW distribution;
      RuntimeEngine re;
      ExecutionContext ec{pg, &distribution, mgr, &re};

    #else 
      ExecutionContext& ec = exc;
    #endif

    ProcGroup pg_l = ProcGroup::create_coll(MPI_COMM_SELF);
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


    TAMMTensors ttensors;

    /*** =========================== ***/
    /*** compute 1-e integrals       ***/
    /*** =========================== ***/

    compute_hamiltonian<TensorType>(ec,atoms,shells,shell_tile_map,AO_tiles,ttensors,etensors);

    //auto H_down = H; //TODO

    /*** =========================== ***/
    /*** build initial-guess density ***/
    /*** =========================== ***/

    etensors.X = compute_orthogonalizer(ec, sys_data, etensors);
    sys_data.nbf = sys_data.nbf_orig - sys_data.n_lindep;

    // pre-compute data for Schwarz bounds
    auto SchwarzK = compute_schwarz_ints<>(shells);

    //D = D_total 
    //Matrix D_spin; //diff of alpha - beta
    //UHF
    if(is_uhf){
      etensors.F_beta = Matrix::Zero(N,N);
      etensors.D_beta = Matrix::Zero(N,N);
    }

    hf_t1 = std::chrono::high_resolution_clock::now();

    if (restart) {
        scf_restart(ec, sys_data, filename, etensors);
        etensors.X = etensors.C;
    }
    else   // SOAD as the guess density
      if(!molden_exists)
      compute_initial_guess<TensorType>(ec, sys_data, atoms, shells, basis, is_spherical, etensors);

    if(molden_exists && is_rhf) {
      //X = C; //doesnt work
      etensors.C_occ = etensors.C.leftCols(sys_data.nelectrons_alpha);
      etensors.D     = etensors.C_occ * etensors.C_occ.transpose();
    }

    // For now etensors.D_beta = etensors.D_spin = 0;
    //if(is_uhf) etensors.D_beta = etensors.D;
    
    // H.resize(0,0);
    hf_t2 = std::chrono::high_resolution_clock::now();
    hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

    ec.pg().barrier();
    if(rank == 0) std::cout << "Total Time to compute initial guess: " << hf_time << " secs" << endl;

    if(rank == 0 && scf_options.debug) cout << "debug #electrons = " << (etensors.D*etensors.S).trace() << endl;

    hf_t1 = std::chrono::high_resolution_clock::now();
    /*** =========================== ***/
    /*** main iterative loop         ***/
    /*** =========================== ***/

    double rmsd          = 1.0;
    double ediff         = 0.0;
    auto is_conv         = true;

    ttensors.ehf_tamm         = {};
    ttensors.ehf_tmp          = {tAO, tAO};
    
    ttensors.F1               = {tAO, tAO};
    ttensors.F1tmp1           = {tAO, tAO};
    ttensors.F1tmp            = {tAOt, tAOt}; //not allocated
    
    ttensors.D_tamm           = {tAO, tAO};
    ttensors.D_diff           = {tAO, tAO};
    ttensors.D_last_tamm      = {tAO, tAO};
    ttensors.FD_tamm          = {tAO, tAO}; 
    ttensors.FDS_tamm         = {tAO, tAO};

    if(is_uhf) {
      ttensors.F1_beta          = {tAO, tAO};
      ttensors.F1tmp1_beta      = {tAO, tAO};
      ttensors.D_beta_tamm      = {tAO, tAO};
      ttensors.D_last_beta_tamm = {tAO, tAO};    
      ttensors.FD_beta_tamm     = {tAO, tAO}; 
      ttensors.FDS_beta_tamm    = {tAO, tAO};    
    }
    
    Tensor<TensorType>::allocate(&ec, ttensors.F1, ttensors.F1tmp1, ttensors.ehf_tmp, ttensors.ehf_tamm);
    Tensor<TensorType>::allocate(&ec, ttensors.D_tamm, ttensors.D_diff, ttensors.D_last_tamm);
    Tensor<TensorType>::allocate(&ec, ttensors.FD_tamm, ttensors.FDS_tamm);
    if(is_uhf) Tensor<TensorType>::allocate(&ec,ttensors.F1_beta,ttensors.F1tmp1_beta,
               ttensors.D_beta_tamm,ttensors.D_last_beta_tamm,ttensors.FD_beta_tamm,ttensors.FDS_beta_tamm);

    hf_t2 = std::chrono::high_resolution_clock::now();
    hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
    //ec.pg().barrier();
    if(rank == 0 && debug) std::cout << "\nTime to setup tensors for iterative loop: " << hf_time << " secs" << endl;

    eigen_to_tamm_tensor(ttensors.D_tamm,etensors.D);

    etensors.F = Matrix::Zero(N,N);
    const libint2::BasisSet& obs = shells;
    etensors.G = Matrix::Zero(N,N);
    const auto do_schwarz_screen = SchwarzK.cols() != 0 && SchwarzK.rows() != 0;
    // auto fock_precision = precision;
    // engine precision controls primitive truncation, assume worst-case scenario
    // (all primitive combinations add up constructively)
    auto max_nprim = obs.max_nprim();
    auto max_nprim4 = max_nprim * max_nprim * max_nprim * max_nprim;
    auto shell2bf = obs.shell2bf();
    // const auto nshells = obs.size();

    //DF basis
    IndexSpace dfCocc{range(0,sys_data.nelectrons_alpha)}; 
    tdfCocc = {dfCocc,tile_size};
    std::tie(dCocc_til) = tdfCocc.labels<1>("all");

    if(do_density_fitting){

      ndf = dfbs.nbf();
      dfAO = IndexSpace{range(0, ndf)};
      std::tie(df_shell_tile_map, dfAO_tiles, dfAO_opttiles) = compute_AO_tiles(ec,sys_data,dfbs);
      // if(rank==0 && debug) 
      // cout << "Number of dfAO tiles = " << dfAO_tiles.size() << endl;

      tdfAO=TiledIndexSpace{dfAO, dfAO_opttiles};
      tdfAOt=TiledIndexSpace{dfAO, dfAO_tiles};
      std::tie(d_mu, d_nu, d_ku) = tdfAO.labels<3>("all");
      std::tie(d_mup, d_nup, d_kup) = tdfAOt.labels<3>("all");
    }

    if(do_density_fitting) {
      ttensors.xyK_tamm = Tensor<TensorType>{tAO, tAO, tdfAO}; //n,n,ndf
      ttensors.C_occ_tamm = Tensor<TensorType>{tAO,tdfCocc}; //n,nocc
      Tensor<TensorType>::allocate(&ec, ttensors.xyK_tamm, ttensors.C_occ_tamm);
      // Tensor<TensorType>::allocate(&ec_l,C_occ_tamm);
    }
    //df basis

    if(rank == 0) {
        std::cout << "\n\n";
        std::cout << " Hartree-Fock iterations" << endl;
        std::cout << std::string(65, '-') << endl;
        std::string  sph = " Iter     Energy            E-Diff        RMSD        Time(s)";
        if(scf_conv) sph = " Iter     Energy            E-Diff        Time(s)";
        std::cout << sph << endl;
        std::cout << std::string(65, '-') << endl;
    }

    std::cout << std::fixed << std::setprecision(2);

    Scheduler sch{ec};

    if(scf_options.restart) {
      sch(ttensors.F1tmp1() = 0).execute();
      //F1 = H1 + F1tmp1
      compute_2bf<TensorType>(ec, sys_data, obs, do_schwarz_screen, shell2bf, SchwarzK,
                  max_nprim4,shells, ttensors, etensors, do_density_fitting);          
      // ehf = D * (H1+F1);
      sch
      (ttensors.F1(mu, nu) = ttensors.H1(mu, nu))
      (ttensors.F1(mu, nu) += ttensors.F1tmp1(mu, nu))    
      (ttensors.ehf_tmp(mu,nu) = ttensors.H1(mu,nu))
      (ttensors.ehf_tmp(mu,nu) += ttensors.F1(mu,nu))
      (ttensors.ehf_tamm() = ttensors.D_tamm() * ttensors.ehf_tmp()).execute();

      ehf = 0.5*get_scalar(ttensors.ehf_tamm) + enuc; 
      if(rank==0) std::cout << std::setprecision(13) << "Total HF energy after restart: " << ehf << std::endl;
    }   


    do {
        const auto loop_start = std::chrono::high_resolution_clock::now();
        ++iter;

        // Save a copy of the energy and the density
        double ehf_last = ehf;

        sch
           (ttensors.F1tmp1() = 0)
           (ttensors.D_last_tamm(mu,nu) = ttensors.D_tamm(mu,nu)).execute();

        if(is_uhf) {
          sch
            (ttensors.F1tmp1_beta() = 0)
            (ttensors.D_last_beta_tamm(mu,nu) = ttensors.D_beta_tamm(mu,nu)).execute();           
        }

        // build a new Fock matrix
        compute_2bf<TensorType>(ec, sys_data, obs, do_schwarz_screen, shell2bf, SchwarzK,
                    max_nprim4,shells, ttensors, etensors, do_density_fitting);            

        if(is_uhf) {
            compute_2bf<TensorType>(ec, sys_data, obs, do_schwarz_screen, shell2bf, SchwarzK,
                        max_nprim4,shells, ttensors, etensors, do_density_fitting, true);
        }

        std::tie(ehf,rmsd) = scf_iter_body<TensorType>(ec, 
    #ifdef SCALAPACK
                        blacs_grid.get(),
    #endif 
                        iter, sys_data, ttensors, etensors, scf_conv);

        ehf += enuc;
        // compute difference with last iteration
        ediff = ehf - ehf_last;

        const auto loop_stop = std::chrono::high_resolution_clock::now();
        const auto loop_time =
        std::chrono::duration_cast<std::chrono::duration<double>>((loop_stop - loop_start)).count();

        if(rank == 0) {
            std::cout << std::setw(4) << iter << "  " << std::setw(10);
            std::cout << std::fixed << std::setprecision(10) << ehf;
            std::cout << std::scientific << std::setprecision(2);
            std::cout << ' ' << std::scientific << std::setw(12)  << ediff;
            if(!scf_conv) std::cout << ' ' << std::setw(12)  << rmsd << ' ';
            std::cout << ' ' << std::setw(10) << std::fixed << std::setprecision(1) << loop_time << ' ' << endl;
        }

        if(rank==0 && scf_options.writem % iter == 0)
          writeC(etensors.C,filename,scf_files_prefix);

        if(iter >= maxiter) {                
            is_conv = false;
            break;
        }

        if(scf_conv) break;

        if(debug) print_energies(ec, ttensors, debug);

    } while ( (fabs(ediff) > conve) || (fabs(rmsd) > convd) );

    // ec.pg().barrier(); 
    if(rank == 0) {
        std::cout.precision(13);
        if (is_conv)
            cout << "\n** Hartree-Fock energy = " << ehf << endl;
        else {
            cout << endl << std::string(50, '*') << endl;
            cout << std::string(10, ' ') << 
                    "ERROR: Hartree-Fock calculation does not converge!!!" << endl;
            cout << std::string(50, '*') << endl;
        }        
    }

    for (auto x: ttensors.diis_hist) Tensor<TensorType>::deallocate(x);
    for (auto x: ttensors.fock_hist) Tensor<TensorType>::deallocate(x);

    if(is_uhf){
      for (auto x: ttensors.diis_hist_beta) Tensor<TensorType>::deallocate(x);
      for (auto x: ttensors.fock_hist_beta) Tensor<TensorType>::deallocate(x);
    }

    if(rank == 0) 
      std::cout << "\nNuclear repulsion energy = " << std::setprecision(15) << enuc << endl;       
    print_energies(ec, ttensors);

    if(do_density_fitting) Tensor<TensorType>::deallocate(ttensors.xyK_tamm,ttensors.C_occ_tamm);
    Tensor<TensorType>::deallocate(ttensors.S1, ttensors.D_tamm, ttensors.ehf_tmp, ttensors.ehf_tamm); 
    Tensor<TensorType>::deallocate(ttensors.D_last_tamm, ttensors.D_diff, ttensors.FD_tamm, ttensors.FDS_tamm);
    Tensor<TensorType>::deallocate(ttensors.H1, ttensors.T1, ttensors.V1, ttensors.F1tmp1);

    if(is_uhf) Tensor<TensorType>::deallocate(ttensors.F1_beta,ttensors.F1tmp1_beta,
            ttensors.D_beta_tamm,ttensors.D_last_beta_tamm,ttensors.FD_beta_tamm,ttensors.FDS_beta_tamm);


    if(rank==0 && !scf_conv) {
     cout << "writing orbitals to file... ";
     writeC(etensors.C,filename,scf_files_prefix);
     cout << "done." << endl;
    }

    if(!is_conv) {
      ec.pg().barrier();
      nwx_terminate("Please check SCF input parameters");
    }
    // else{
    //     if(rank==0 && !scf_conv){
    //       std::ofstream out(scfstatusfile, std::ios::out);
    //       if(!out) cerr << "Error opening file " << scfstatusfile << endl;
    //       out << 1 << std::endl;
    //       out.close();
    //     }    
    // }

      if(rank == 0) tamm_to_eigen_tensor(ttensors.F1,etensors.F);

      Tensor<TensorType>::deallocate(ttensors.F1); //deallocate using ec

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

    //F, C are not deallocated.
    sys_data.n_occ_alpha = sys_data.nelectrons_alpha;
    sys_data.n_occ_beta  = sys_data.nelectrons_beta;
    sys_data.n_vir_alpha = sys_data.nbf_orig - sys_data.n_occ_alpha - sys_data.n_lindep;
    sys_data.n_vir_beta  = sys_data.nbf_orig - sys_data.n_occ_beta - sys_data.n_lindep;

    MPI_Bcast(&ehf,1,mpi_type<TensorType>(),0,exc.pg().comm());
    MPI_Bcast(&sys_data.nbf        ,1,mpi_type<int>(),0,exc.pg().comm());    
    MPI_Bcast(&sys_data.n_lindep   ,1,mpi_type<int>(),0,exc.pg().comm());
    MPI_Bcast(&sys_data.n_occ_alpha,1,mpi_type<int>(),0,exc.pg().comm());
    MPI_Bcast(&sys_data.n_vir_alpha,1,mpi_type<int>(),0,exc.pg().comm());
    MPI_Bcast(&sys_data.n_occ_beta ,1,mpi_type<int>(),0,exc.pg().comm());
    MPI_Bcast(&sys_data.n_vir_beta ,1,mpi_type<int>(),0,exc.pg().comm());

    sys_data.update();
    if(rank==0 && debug) sys_data.print();
    sys_data.input_molecule = getfilename(filename);
    sys_data.scf_iterations = iter; //not broadcasted, but fine since only rank 0 writes to json
    sys_data.scf_energy = ehf;
    if(rank==0) write_results(sys_data,"SCF");

    tamm::Tile ao_tile = scf_options.AO_tilesize;
    if(tile_size < N*0.05 && !scf_options.force_tilesize)
        ao_tile = static_cast<tamm::Tile>(std::ceil(N*0.05));

    IndexSpace AO_ortho{range(0, (size_t)(sys_data.nbf_orig-sys_data.n_lindep))};
    TiledIndexSpace tAO_ortho{AO_ortho,ao_tile};
        
    Tensor<TensorType> C_tamm{tAO,tAO_ortho};
    Tensor<TensorType> F_tamm{tAO,tAO};
    Tensor<TensorType>::allocate(&exc,C_tamm,F_tamm);
    if (rank == 0) {
      eigen_to_tamm_tensor(C_tamm,etensors.C);
      eigen_to_tamm_tensor(F_tamm,etensors.F);
    }

    exc.pg().barrier();

    return std::make_tuple(sys_data, ehf, shells, shell_tile_map, C_tamm, F_tamm, tAO, tAOt, scf_conv);
}



#endif // TAMM_TESTS_HF_TAMM_HPP_
