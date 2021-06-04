
#ifndef TAMM_METHODS_HF_TAMM_HPP_
#define TAMM_METHODS_HF_TAMM_HPP_

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

#include "hf_tamm_common.hpp"

#include <filesystem>
namespace fs = std::filesystem;

#define SCF_THROTTLE_RESOURCES 1

std::tuple<SystemData, double, libint2::BasisSet, std::vector<size_t>, 
    Tensor<double>, Tensor<double>, Tensor<double>, Tensor<double>, TiledIndexSpace, TiledIndexSpace, bool> 
    hartree_fock(ExecutionContext &exc, const string filename,
                 std::vector<libint2::Atom> atoms, OptionsMap options_map) {

    using libint2::Atom;
    using libint2::Engine;
    using libint2::Operator;
    using libint2::Shell;
    using libint2::BasisSet;

    /*** =========================== ***/
    /*** initialize molecule         ***/
    /*** =========================== ***/

    SystemData sys_data{options_map, options_map.scf_options.scf_type};

    SCFOptions  scf_options  = sys_data.options_map.scf_options;

    std::string basis        = scf_options.basis;
    int         charge       = scf_options.charge;
    int         multiplicity = scf_options.multiplicity;
    int         maxiter      = scf_options.maxiter;
    double      conve        = scf_options.conve;
    double      convd        = scf_options.convd;
    bool        debug        = scf_options.debug;
    bool        ediis        = scf_options.ediis;
    double      ediis_off    = scf_options.ediis_off;
    auto        restart      = scf_options.restart;
    bool        is_spherical = (scf_options.sphcart == "spherical");
    // bool        sad          = scf_options.sad;
    auto        iter         = 0;

    auto hf_t1 = std::chrono::high_resolution_clock::now();

    // initializes the Libint integrals library ... now ready to compute
    libint2::initialize(false);
    libint2::Shell::do_enforce_unit_normalization(false);
    auto rank      = exc.pg().rank();

    /*** =========================== ***/
    /*** create basis set            ***/
    /*** =========================== ***/

    std::string basis_set_file = std::string(DATADIR) + "/basis/" + basis + ".g94";
    
    int basis_file_exists = 0;
    if(rank == 0) basis_file_exists = std::filesystem::exists(basis_set_file);

    MPI_Bcast(&basis_file_exists        ,1,mpi_type<int>()       ,0,exc.pg().comm());  
    if (!basis_file_exists) tamm_terminate("basis set file " + basis_set_file + " does not exist");

    libint2::BasisSet shells(std::string(basis), atoms);
    if(is_spherical) shells.set_pure(true);
    else shells.set_pure(false);  // use cartesian gaussians

    // auto shells = make_sto3g_basis(atoms);
    const size_t N = nbasis(shells);
    auto nnodes    = GA_Cluster_nnodes();

    sys_data.nbf      = N;
    sys_data.nbf_orig = N;
    sys_data.ediis    = ediis;

    const bool is_rhf = (sys_data.scf_type == sys_data.SCFType::rhf);
    const bool is_uhf = (sys_data.scf_type == sys_data.SCFType::uhf);
    // const bool is_rohf = (sys_data.scf_type == sys_data.SCFType::rohf);    

    std::string out_fp = options_map.options.output_file_prefix+"."+scf_options.basis;
    std::string files_dir = out_fp+"_files/"+sys_data.options_map.scf_options.scf_type+"/scf";
    std::string files_prefix = /*out_fp;*/ files_dir+"/"+out_fp;
    if(!fs::exists(files_dir)) fs::create_directories(files_dir);

    #if SCF_THROTTLE_RESOURCES
      auto [t_nnodes,hf_nnodes,ppn,hf_nranks] = get_hf_nranks(N);
      if (scf_options.nnodes > t_nnodes) {
        const std::string errmsg = "ERROR: nnodes (" + std::to_string(scf_options.nnodes)
        + ") provided is greater than the number of nodes (" + std::to_string(t_nnodes) + ") available!";
        tamm_terminate(errmsg);
      }
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

      // int lranks[ppn];
      // for (int i = 0; i < ppn; i++) lranks[i] = i;
      // MPI_Group lgroup;
      // MPI_Comm_group(gcomm,&lgroup);
      // MPI_Group hf_lgroup;
      // MPI_Group_incl(lgroup,ppn,lranks,&hf_lgroup);
      // MPI_Comm hf_lcomm;
      // MPI_Comm_create(gcomm,hf_lgroup,&hf_lcomm);
    #endif

    if(rank == 0) {
      cout << std::endl << "Number of nodes, mpi ranks per node provided: " << nnodes << ", " << GA_Cluster_nprocs(0) << endl;
      #if SCF_THROTTLE_RESOURCES
        cout << "Number of nodes, mpi ranks per node used for SCF calculation: " << hf_nnodes << ", " << ppn << endl;
      #endif
      scf_options.print();
    }

    auto [ndocc, enuc]  = compute_NRE(exc, atoms, sys_data.focc);

    const auto nelectrons = sys_data.focc * ndocc - charge;
    sys_data.nelectrons = nelectrons;
    EXPECTS ( (nelectrons + scf_options.multiplicity - 1) % 2 == 0 );  

    sys_data.nelectrons_alpha = (nelectrons + scf_options.multiplicity - 1)/2; 
    sys_data.nelectrons_beta = nelectrons - sys_data.nelectrons_alpha;
    
    if(rank==0) {
      std::cout << std::endl << "Number of basis functions = " << N << std::endl;
      std::cout << std::endl << "Total number of electrons = " << nelectrons << std::endl;      
      std::cout <<              "  # of alpha electrons    = " << sys_data.nelectrons_alpha << std::endl;
      std::cout <<              "  # of beta electons      = " << sys_data.nelectrons_beta << std::endl;
      std::cout << std::endl << "Nuclear repulsion energy  = " << std::setprecision(15) << enuc << std::endl << std::endl; 
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

      ndf = dfbs.nbf();
      dfAO = IndexSpace{range(0, ndf)};
      std::tie(df_shell_tile_map, dfAO_tiles, dfAO_opttiles) = compute_AO_tiles(exc,sys_data,dfbs);
    
      tdfAO=TiledIndexSpace{dfAO, dfAO_opttiles};
      tdfAOt=TiledIndexSpace{dfAO, dfAO_tiles};
      
    }
    std::unique_ptr<DFFockEngine> dffockengine(
        do_density_fitting ? new DFFockEngine(shells, dfbs) : nullptr);

    tamm::Tile tile_size = scf_options.AO_tilesize; //TODO
    IndexSpace AO{range(0, N)};
    std::tie(shell_tile_map, AO_tiles, AO_opttiles) = compute_AO_tiles(exc, sys_data, shells);
    tAO  = {AO, AO_opttiles};
    tAOt = {AO, AO_tiles};
    std::tie(mu, nu, ku)    = tAO.labels<3>("all");
    std::tie(mup, nup, kup) = tAOt.labels<3>("all");

    // if(rank==0) cout << endl << "Set AO indexspace" << endl;
      
    auto hf_t2 = std::chrono::high_resolution_clock::now();
    double hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
    if(rank == 0) std::cout << std::endl << "Time for initial setup: " << hf_time << " secs" << endl;

    double ehf = 0.0; // initialize Hartree-Fock energy

    exc.pg().barrier();

    EigenTensors etensors;

    const bool scf_conv = restart && scf_options.noscf; 
    const int  max_hist = sys_data.options_map.scf_options.diis_hist; 
    const bool molden_exists = !scf_options.moldenfile.empty();

    if(molden_exists) {
      //TODO: WIP, not working yet
      size_t nmo = N-scf_options.n_lindep;
      if(is_uhf) {
        nmo = 2*(N-scf_options.n_lindep);
      }

      std::vector<TensorType> evl_sorted(nmo); //FIXME: not used/broadcasted
      etensors.C.setZero(N,nmo);
      int n_occ_alpha=0, n_occ_beta=0, n_vir_alpha=0, n_vir_beta=0;

      bool molden_file_valid=std::filesystem::exists(scf_options.moldenfile);
      if(rank == 0) {
        cout << endl << "Reading from molden file provided ..." << endl;
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
      
      if(!molden_file_valid) tamm_terminate("ERROR: Cannot open moldenfile provided: " + scf_options.moldenfile);

    }

    scf_restart_test(exc, sys_data, filename, restart, files_prefix);

    std::string movecsfile_alpha  = files_prefix + ".alpha.movecs";
    std::string densityfile_alpha = files_prefix + ".alpha.density";
    std::string movecsfile_beta  =  files_prefix + ".beta.movecs";       
    std::string densityfile_beta =  files_prefix + ".beta.density"; 

    #if SCF_THROTTLE_RESOURCES
    if (rank < hf_nranks) {
      int hrank;
      EXPECTS(hf_comm != MPI_COMM_NULL);
      MPI_Comm_rank(hf_comm,&hrank);
      EXPECTS(rank==hrank);

      ProcGroup pg = ProcGroup::create_coll(hf_comm);
      ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
      // ProcGroup pg_m = ProcGroup::create_coll(hf_lcomm);
      // ExecutionContext ec_m{pg_m, DistributionKind::nw, MemoryManagerKind::ga};
    #else 
      //TODO: Fix - create ec_m, when throttle is disabled
      ExecutionContext& ec = exc;
    #endif
      ProcGroup pg_l = ProcGroup::create_coll(MPI_COMM_SELF);
      ExecutionContext ec_l{pg_l, DistributionKind::nw, MemoryManagerKind::local};
    #ifdef USE_SCALAPACK

      auto blacs_setup_st = std::chrono::high_resolution_clock::now();
      // Sanity checks
      int scalapack_nranks = 
        scf_options.scalapack_np_row *
        scf_options.scalapack_np_col;

      // XXX: This should be for hf_comm
      int world_size;
      MPI_Comm_size( ec.pg().comm(), &world_size );

      // Default to square(ish) grid
      if( scalapack_nranks == 0 ) {
        int64_t npr = std::sqrt( world_size );
        int64_t npc = world_size / npr; 
        while( npr * npc != world_size ) {
          npr--;
          npc = world_size / npr;
        }
        scalapack_nranks = world_size;
        scf_options.scalapack_np_row = npr;
        scf_options.scalapack_np_col = npc;
      }

      assert( world_size >= scalapack_nranks );

      if( not scalapack_nranks ) scalapack_nranks = world_size;
      std::vector<int64_t> scalapack_ranks( scalapack_nranks );
      std::iota( scalapack_ranks.begin(), scalapack_ranks.end(), 0 );

#if 0
      MPI_Group world_group, scalapack_group;
      MPI_Comm scalapack_comm;
      MPI_Comm_group( ec.pg().comm(), &world_group );
      MPI_Group_incl( world_group, scalapack_nranks, scalapack_ranks.data(), &scalapack_group );
      MPI_Comm_create( ec.pg().comm(), scalapack_group, &scalapack_comm );
      
      std::unique_ptr<blacspp::Grid> blacs_grid = nullptr;
      std::unique_ptr<scalapackpp::BlockCyclicDist2D> blockcyclic_dist = nullptr;
      if( scalapack_comm != MPI_COMM_NULL ) {
        const auto NPR = scf_options.scalapack_np_row;
        const auto NPC = scf_options.scalapack_np_col;
        blacs_grid = std::make_unique<blacspp::Grid>( scalapack_comm, NPR, NPC );

        const auto MB = scf_options.scalapack_nb;
        blockcyclic_dist = std::make_unique<scalapackpp::BlockCyclicDist2D>( 
          *blacs_grid, MB, MB, 0, 0 );
      }
#else
      auto blacs_grid = std::make_unique<blacspp::Grid>( 
        ec.pg().comm(), scf_options.scalapack_np_row, scf_options.scalapack_np_col,
        scalapack_ranks.data(), scf_options.scalapack_np_row );
      auto blockcyclic_dist = std::make_unique<scalapackpp::BlockCyclicDist2D>( 
          *blacs_grid, scf_options.scalapack_nb, scf_options.scalapack_nb, 0, 0 );
#endif

      auto blacs_setup_en = std::chrono::high_resolution_clock::now();

      std::chrono::duration<double> blacs_time = blacs_setup_en - blacs_setup_st;
      
      if(rank == 0) std::cout << std::endl << "Time for BLACS setup: " << blacs_time.count() << " secs" << std::endl;

    #endif

      TAMMTensors ttensors;

      /*** =========================== ***/
      /*** compute 1-e integrals       ***/
      /*** =========================== ***/
      compute_hamiltonian<TensorType>(ec,atoms,shells,shell_tile_map,AO_tiles,ttensors,etensors);

      /*** =========================== ***/
      /*** build initial-guess density ***/
      /*** =========================== ***/

      std::string ortho_file = files_prefix + ".orthogonalizer";   
      int  ostatus = 1;
      if(N > 2000) {
        if(rank==0) ostatus = fs::exists(ortho_file);
        // if(ostatus == 0) tamm_terminate("Error reading orthogonalizer: [" + ortho_file + "]");
        MPI_Bcast(&ostatus        ,1,mpi_type<int>()       ,0,ec.pg().comm());
      }

      if(rank == 0) etensors.F       = Matrix::Zero(N,N);
      etensors.D       = Matrix::Zero(N,N);
      etensors.G       = Matrix::Zero(N,N);
      if(ostatus && N > 2000) {
        if(rank==0) {
          etensors.X = read_scf_mat<TensorType>(ortho_file);
          sys_data.n_lindep = sys_data.nbf_orig - etensors.X.cols();
        }
        MPI_Bcast(&sys_data.n_lindep        ,1,mpi_type<int>()       ,0,ec.pg().comm());
      }
      else 
      {
        etensors.X  = compute_orthogonalizer(ec, sys_data, ttensors);
        if(rank==0) write_scf_mat<TensorType>(etensors.X, ortho_file);
      }
      if(is_uhf) {
        if(rank == 0) etensors.F_beta  = Matrix::Zero(N,N);
        etensors.D_beta  = Matrix::Zero(N,N);
        etensors.G_beta  = Matrix::Zero(N,N);
        etensors.X_beta  = etensors.X;
      }

      sys_data.nbf = sys_data.nbf_orig - sys_data.n_lindep;

      // pre-compute data for Schwarz bounds
      if(rank==0) cout << "pre-compute data for Schwarz bounds" << endl;
      auto SchwarzK = compute_schwarz_ints<>(shells);

      hf_t1 = std::chrono::high_resolution_clock::now();

      ttensors.ehf_tamm    = Tensor<TensorType>{};
      ttensors.F1tmp       = {tAOt, tAOt}; //not allocated

      ttensors.ehf_tmp     = {tAO, tAO};
      ttensors.F1          = {tAO, tAO};
      ttensors.D_tamm      = {tAO, tAO};
      ttensors.D_diff      = {tAO, tAO};
      ttensors.D_last_tamm = {tAO, tAO};
      ttensors.F1tmp1      = {tAO, tAO};
      ttensors.FD_tamm     = {tAO, tAO}; 
      ttensors.FDS_tamm    = {tAO, tAO};

      if(is_uhf) {
        ttensors.ehf_beta_tmp     = {tAO, tAO};
        ttensors.F1_beta          = {tAO, tAO};
        ttensors.D_beta_tamm      = {tAO, tAO};
        ttensors.D_last_beta_tamm = {tAO, tAO};    
        ttensors.F1tmp1_beta      = {tAO, tAO};
        ttensors.FD_beta_tamm     = {tAO, tAO}; 
        ttensors.FDS_beta_tamm    = {tAO, tAO};    
      }
    
      Tensor<TensorType>::allocate(&ec, ttensors.F1     , 
                                        ttensors.D_tamm , ttensors.D_last_tamm, ttensors.D_diff  ,
                                        ttensors.F1tmp1 , ttensors.ehf_tmp    , ttensors.ehf_tamm,
                                        ttensors.FD_tamm, ttensors.FDS_tamm);
      if(is_uhf) 
        Tensor<TensorType>::allocate(&ec, ttensors.F1_beta     , 
                                          ttensors.D_beta_tamm , ttensors.D_last_beta_tamm, 
                                          ttensors.F1tmp1_beta , ttensors.ehf_beta_tmp    ,
                                          ttensors.FD_beta_tamm, ttensors.FDS_beta_tamm);

      const auto do_schwarz_screen = SchwarzK.cols() != 0 && SchwarzK.rows() != 0;
      // engine precision controls primitive truncation, assume worst-case scenario
      // (all primitive combinations add up constructively)
      const libint2::BasisSet& obs = shells;
      auto max_nprim  = obs.max_nprim();
      auto max_nprim4 = max_nprim * max_nprim * max_nprim * max_nprim;
      auto shell2bf   = obs.shell2bf();
      
      Scheduler sch{ec};

      if (restart) {
        scf_restart(ec, sys_data, filename, etensors, files_prefix);
        if(is_rhf) 
          etensors.X      = etensors.C;
        if(is_uhf) {
          etensors.X      = etensors.C;
          etensors.X_beta = etensors.C_beta;
        }
      }
      else {
        // FIXME:UNCOMMENT
        #if 0
        if(sad) {
          if(rank==0) cout << "SAD enabled" << endl;

          compute_sad_guess<TensorType>(ec, sys_data, atoms, shells, basis, 
                                       is_spherical, etensors, charge, multiplicity); 
          compute_2bf<TensorType>(ec, sys_data, obs, do_schwarz_screen, shell2bf, SchwarzK,
                                         max_nprim4,shells, ttensors, etensors, do_density_fitting);
          sch
            (ttensors.F1()  = ttensors.H1())
            (ttensors.F1() += ttensors.F1tmp1())
            .execute();
          tamm_to_eigen_tensor(ttensors.F1,etensors.F);
          Eigen::SelfAdjointEigenSolver<Matrix> eig_solver_guess_a(etensors.X.transpose() * etensors.F * etensors.X);
          auto C_alpha = etensors.X * eig_solver_guess_a.eigenvectors();
          auto C_occ_a = C_alpha.leftCols(sys_data.nelectrons_alpha);
          if(is_rhf) 
            etensors.D = 2.0 * C_occ_a * C_occ_a.transpose();
          if(is_uhf) {
            etensors.D = C_occ_a * C_occ_a.transpose();
            sch
              (ttensors.F1_beta()  = ttensors.H1())
              (ttensors.F1_beta() += ttensors.F1tmp1_beta())
              .execute();
            tamm_to_eigen_tensor(ttensors.F1_beta,etensors.F_beta);
            Eigen::SelfAdjointEigenSolver<Matrix> eig_solver_guess_b(etensors.X.transpose() * etensors.F_beta * etensors.X);
            auto C_beta  = etensors.X * eig_solver_guess_b.eigenvectors();
            auto C_occ_b = C_beta.leftCols(sys_data.nelectrons_beta);
            etensors.D_beta = C_occ_b * C_occ_b.transpose();
          }
        }
        else
        #endif
        {
          compute_initial_guess<TensorType>(ec, sys_data, atoms, shells, basis, is_spherical,
                                            etensors, ttensors, charge, multiplicity);

          if(rank == 0) {
            write_scf_mat<TensorType>(etensors.C, movecsfile_alpha);
            write_scf_mat<TensorType>(etensors.D, densityfile_alpha);
            if(is_uhf) {
              write_scf_mat<TensorType>(etensors.C_beta, movecsfile_beta);
              write_scf_mat<TensorType>(etensors.D_beta, densityfile_beta);
            }
          }
          ec.pg().barrier();
        }
        
      }

      hf_t2   = std::chrono::high_resolution_clock::now();
      hf_time = std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
      if(rank == 0) std::cout << "Total Time to compute initial guess: " << hf_time << " secs" << endl;

      hf_t1 = std::chrono::high_resolution_clock::now();
      /*** =========================== ***/
      /*** main iterative loop         ***/
      /*** =========================== ***/
      double rmsd          = 1.0;
      double ediff         = 0.0;
      bool   is_conv       = true;
      

      hf_t2   = std::chrono::high_resolution_clock::now();
      hf_time = std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
      if(rank == 0 && debug) std::cout << std::endl << "Time to setup tensors for iterative loop: " << hf_time << " secs" << endl;

      if(rank == 0) {
        eigen_to_tamm_tensor(ttensors.D_tamm,etensors.D);
        if(is_uhf) {
          eigen_to_tamm_tensor(ttensors.D_beta_tamm,etensors.D_beta);
        }
      }

      MPI_Bcast( etensors.D.data(), etensors.D.size(), MPI_DOUBLE, 0, ec.pg().comm() );
      if(is_uhf) MPI_Bcast( etensors.D_beta.data(), etensors.D_beta.size(), MPI_DOUBLE, 0, ec.pg().comm() );
      
      // FIXME: No S in etensors
      // if(rank == 0 && scf_options.debug) {
      //   if(is_rhf) 
      //     cout << "debug #electrons       = " << (etensors.D*etensors.S).trace()      << endl;
      //   if(is_uhf) {
      //     cout << "debug #alpha electrons = " << (etensors.D*etensors.S).trace()      << endl;
      //     cout << "debug #beta  electrons = " << (etensors.D_beta*etensors.S).trace() << endl;
      //   }
      // }
    
      //DF basis
      IndexSpace dfCocc{range(0,sys_data.nelectrons_alpha)}; 
      tdfCocc = {dfCocc,tile_size};
      std::tie(dCocc_til) = tdfCocc.labels<1>("all");

      if(do_density_fitting){
        std::tie(d_mu, d_nu, d_ku) = tdfAO.labels<3>("all");
        std::tie(d_mup, d_nup, d_kup) = tdfAOt.labels<3>("all");

        ttensors.Zxy_tamm = Tensor<TensorType>{tdfAO, tAO, tAO}; //ndf,n,n
        ttensors.xyK_tamm = Tensor<TensorType>{tAO, tAO, tdfAO}; //n,n,ndf
        ttensors.C_occ_tamm = Tensor<TensorType>{tAO,tdfCocc}; //n,nocc
        Tensor<TensorType>::allocate(&ec, ttensors.xyK_tamm, ttensors.C_occ_tamm,ttensors.Zxy_tamm);
      }//df basis

      if(rank == 0) {
          std::cout << std::endl << std::endl;
          std::cout << " SCF iterations" << endl;
          std::cout << std::string(65, '-') << endl;
          std::string  sph = " Iter     Energy            E-Diff        RMSD        Time(s)";
          if(scf_conv) sph = " Iter     Energy            E-Diff        Time(s)";
          std::cout << sph << endl;
          std::cout << std::string(65, '-') << endl;
      }

      std::cout << std::fixed << std::setprecision(2);

      if(restart) {
        
        sch
          (ttensors.F1tmp1() = 0)
          .execute();

        if(is_uhf) {
          sch
            (ttensors.F1tmp1_beta() = 0)
            .execute();
        }
        //F1 = H1 + F1tmp1
        compute_2bf<TensorType>(ec, sys_data, obs, do_schwarz_screen, shell2bf, SchwarzK,
                                max_nprim4, shells, ttensors, etensors, do_density_fitting);          
        // ehf = D * (H1+F1);
        if(is_rhf) {
          sch
            (ttensors.ehf_tmp(mu,nu)  = 2.0 * ttensors.H1(mu,nu))
            (ttensors.ehf_tmp(mu,nu) += 1.0 * ttensors.F1tmp1(mu,nu))
            (ttensors.ehf_tamm()      = 1.0 * ttensors.D_tamm() * ttensors.ehf_tmp())
            .execute();
        }
        if(is_uhf) {
          sch
            (ttensors.ehf_tmp(mu,nu)  = 2.0 * ttensors.H1(mu,nu))
            (ttensors.ehf_tmp(mu,nu) += 1.0 * ttensors.F1tmp1(mu,nu))
            (ttensors.ehf_tamm()      = 1.0 * ttensors.D_tamm() * ttensors.ehf_tmp())
            (ttensors.ehf_tmp(mu,nu)  = 2.0 * ttensors.H1(mu,nu))
            (ttensors.ehf_tmp(mu,nu) += 1.0 * ttensors.F1tmp1_beta(mu,nu))
            (ttensors.ehf_tamm()     += 1.0 * ttensors.D_beta_tamm() * ttensors.ehf_tmp())
            .execute();
        }

        ehf = 0.5*get_scalar(ttensors.ehf_tamm) + enuc; 
        if(rank==0) 
          std::cout << std::setprecision(18) << "Total HF energy after restart: " << ehf << std::endl;
      }   

      //SCF main loop
      do {
        const auto loop_start = std::chrono::high_resolution_clock::now();
        ++iter;

        // Save a copy of the energy and the density
        double ehf_last = ehf;

        sch
          (ttensors.F1tmp1() = 0)
          (ttensors.D_last_tamm(mu,nu) = ttensors.D_tamm(mu,nu))
          .execute();
        
        if(is_uhf) {
          sch
            (ttensors.F1tmp1_beta() = 0)
            (ttensors.D_last_beta_tamm(mu,nu) = ttensors.D_beta_tamm(mu,nu))
            .execute();
        }
            
        // auto D_tamm_nrm = norm(ttensors.D_tamm);
        // if(rank==0) cout << std::setprecision(18) << "norm of D_tamm: " << D_tamm_nrm << endl;

        // build a new Fock matrix
        compute_2bf<TensorType>(ec, sys_data, obs, do_schwarz_screen, shell2bf, SchwarzK,
                                max_nprim4,shells, ttensors, etensors, do_density_fitting);

        //E_Diis
        if(ediis) {
          Tensor<TensorType>  Dcopy{tAO,tAO};
          Tensor<TensorType>  Fcopy{tAO, tAO};
          Tensor<TensorType>  ehf_tamm_copy{};
          Tensor<TensorType>::allocate(&ec,Dcopy,Fcopy,ehf_tamm_copy);
          sch
            (Dcopy()  = ttensors.D_tamm())
            (Fcopy()  = ttensors.F1tmp1())
            (Fcopy() += ttensors.H1())
            (ehf_tamm_copy()  = 0.5 * Dcopy() * Fcopy())
            (ehf_tamm_copy() += 0.5 * Dcopy() * ttensors.H1())
            .execute();

          auto H_nrm = norm(ttensors.H1);
          auto F_nrm = norm(Fcopy);
          auto D_nrm = norm(Dcopy);
          if(rank==0 && debug) cout << "<ediis> norm of H,F,D: " << H_nrm << "," << F_nrm << "," << D_nrm << "," << get_scalar(ehf_tamm_copy) << endl;
          ttensors.D_hist.push_back(Dcopy);
          ttensors.fock_hist.push_back(Fcopy);
          ttensors.ehf_tamm_hist.push_back(ehf_tamm_copy);
          // if(rank==0) cout << "iter: " << iter << "," << (int)ttensors.D_hist.size() << "," << get_scalar(ehf_tamm_copy) << endl;
          energy_diis(ec, tAO, iter, max_hist, ttensors.D_tamm, ttensors.F1, ttensors.ehf_tamm,
                      ttensors.D_hist, ttensors.fock_hist, ttensors.ehf_tamm_hist);
        }

        std::tie(ehf,rmsd) = scf_iter_body<TensorType>(ec, 
        #ifdef USE_SCALAPACK
                        blacs_grid.get(),
                        blockcyclic_dist.get(),
        #endif 
                        iter, sys_data, ttensors, etensors, ediis, scf_conv);

        if(ediis && fabs(rmsd) < ediis_off) ediis = false;

        ehf += enuc;
        // compute difference with last iteration
        ediff = ehf - ehf_last;

        const auto loop_stop = std::chrono::high_resolution_clock::now();
        const auto loop_time =
        std::chrono::duration_cast<std::chrono::duration<double>>((loop_stop - loop_start)).count();

        if(rank == 0) {
          std::cout << std::setw(4) << iter << "  " << std::setw(10);
          if(debug) {
            std::cout << std::fixed   << std::setprecision(18) << ehf;
            std::cout << std::scientific << std::setprecision(18);
          }
          else {
            std::cout << std::fixed   << std::setprecision(10) << ehf;
            std::cout << std::scientific << std::setprecision(2);            
          }
          std::cout << ' ' << std::scientific << std::setw(12)  << ediff;
          if(!scf_conv) std::cout << ' ' << std::setw(12)  << rmsd << ' ';
          std::cout << ' ' << std::setw(10) << std::fixed << std::setprecision(1) << loop_time << ' ' << endl;

          sys_data.results["output"]["SCF"]["iter"][std::to_string(iter)]["data"] = { {"energy", ehf}, {"e_diff", ediff}, {"rmsd", rmsd} };
          sys_data.results["output"]["SCF"]["iter"][std::to_string(iter)]["profile"] = { {"total_time", loop_time} };

        }

        // if(rank==0) cout << "D at the end of iteration: " << endl << std::setprecision(6) << etensors.D << endl;
        if(rank==0 && scf_options.writem % iter == 0) {
          write_scf_mat<TensorType>(etensors.C, movecsfile_alpha);
          write_scf_mat<TensorType>(etensors.D, densityfile_alpha);
          if(is_uhf) {
            write_scf_mat<TensorType>(etensors.C_beta, movecsfile_beta);
            write_scf_mat<TensorType>(etensors.D_beta, densityfile_beta);
          }
        }

        if(iter >= maxiter) {                
          is_conv = false;
          break;
        }

        if(scf_conv) break;

        if(debug) print_energies(ec, ttensors, sys_data, debug);

      } while ( (fabs(ediff) > conve) || (fabs(rmsd) > convd) ); //SCF main loop

      if(rank == 0) {
        std::cout.precision(13);
        if (is_conv)
          cout << endl << "** Total SCF energy = " << ehf << endl;
        else {
          cout << endl << std::string(50, '*') << endl;
          cout << std::string(10, ' ') << 
                  "ERROR: SCF calculation does not converge!!!" << endl;
          cout << std::string(50, '*') << endl;
        }        
      }

      for (auto x: ttensors.ehf_tamm_hist)  Tensor<TensorType>::deallocate(x);
      
      for (auto x: ttensors.diis_hist)      Tensor<TensorType>::deallocate(x);
      for (auto x: ttensors.fock_hist)      Tensor<TensorType>::deallocate(x);
      for (auto x: ttensors.D_hist)         Tensor<TensorType>::deallocate(x);
      
      if(is_uhf){
        for (auto x: ttensors.diis_beta_hist)      Tensor<TensorType>::deallocate(x);
        for (auto x: ttensors.fock_beta_hist)      Tensor<TensorType>::deallocate(x);
        for (auto x: ttensors.D_beta_hist)         Tensor<TensorType>::deallocate(x);
      }

      if(rank == 0) 
        std::cout << std::endl << "Nuclear repulsion energy = " << std::setprecision(15) << enuc << endl;       
      print_energies(ec, ttensors, sys_data);
      
      if(rank==0 && !scf_conv) {
        cout << "writing orbitals and density to file... ";
        write_scf_mat<TensorType>(etensors.C     , movecsfile_alpha);
        write_scf_mat<TensorType>(etensors.D     , densityfile_alpha);
        if(is_uhf) {
          write_scf_mat<TensorType>(etensors.C_beta, movecsfile_beta);
          write_scf_mat<TensorType>(etensors.D_beta, densityfile_beta);
        }
        cout << "done." << endl;
      }

      if(!is_conv) {
        ec.pg().barrier();
        tamm_terminate("Please check SCF input parameters");
      }

      if(rank == 0) tamm_to_eigen_tensor(ttensors.F1,etensors.F);

      if(do_density_fitting) Tensor<TensorType>::deallocate(ttensors.xyK_tamm, ttensors.C_occ_tamm, ttensors.Zxy_tamm);

      Tensor<TensorType>::deallocate(ttensors.H1     , ttensors.S1      , ttensors.T1         , ttensors.V1,
                                     ttensors.F1tmp1 , ttensors.ehf_tmp , ttensors.ehf_tamm   , ttensors.F1,
                                     ttensors.D_tamm , ttensors.D_diff  , ttensors.D_last_tamm,
                                     ttensors.FD_tamm, ttensors.FDS_tamm);
      
      if(is_uhf) 
        Tensor<TensorType>::deallocate(ttensors.F1_beta     , 
                                       ttensors.D_beta_tamm , ttensors.D_last_beta_tamm,
                                       ttensors.F1tmp1_beta , ttensors.ehf_beta_tmp    ,
                                       ttensors.FD_beta_tamm, ttensors.FDS_beta_tamm   );

      #if SCF_THROTTLE_RESOURCES
      ec.flush_and_sync();
      #endif

      ec_l.flush_and_sync();

      #if 0
      #ifdef USE_SCALAPACK
      // Free up created comms / groups
      MPI_Comm_free( &scalapack_comm );
      MPI_Group_free( &scalapack_group );
      MPI_Group_free( &world_group );
      #endif
      #endif
    
    #if SCF_THROTTLE_RESOURCES

    } //end scaled down process group

    #endif

    //C,F1 is not allocated for ranks > hf_nranks 
    exc.pg().barrier(); 

    //F, C are not deallocated.
    sys_data.n_occ_alpha = sys_data.nelectrons_alpha;
    sys_data.n_occ_beta  = sys_data.nelectrons_beta;
    sys_data.n_vir_alpha = sys_data.nbf_orig - sys_data.n_occ_alpha - sys_data.n_lindep;
    sys_data.n_vir_beta  = sys_data.nbf_orig - sys_data.n_occ_beta - sys_data.n_lindep;

    MPI_Bcast(&ehf                 ,1,mpi_type<TensorType>(),0,exc.pg().comm());
    MPI_Bcast(&sys_data.nbf        ,1,mpi_type<int>()       ,0,exc.pg().comm());    
    MPI_Bcast(&sys_data.n_lindep   ,1,mpi_type<int>()       ,0,exc.pg().comm());
    MPI_Bcast(&sys_data.n_occ_alpha,1,mpi_type<int>()       ,0,exc.pg().comm());
    MPI_Bcast(&sys_data.n_vir_alpha,1,mpi_type<int>()       ,0,exc.pg().comm());
    MPI_Bcast(&sys_data.n_occ_beta ,1,mpi_type<int>()       ,0,exc.pg().comm());
    MPI_Bcast(&sys_data.n_vir_beta ,1,mpi_type<int>()       ,0,exc.pg().comm());

    sys_data.update();
    if(rank==0 && debug) sys_data.print();
    // sys_data.input_molecule = getfilename(filename);
    sys_data.scf_energy = ehf;
    // iter not broadcasted, but fine since only rank 0 writes to json
    if(rank == 0) {
      sys_data.results["output"]["SCF"]["final_energy"] = ehf;
      sys_data.results["output"]["SCF"]["n_iterations"] = iter;
    }

    tamm::Tile ao_tile = scf_options.AO_tilesize;
    if(tile_size < N*0.05 && !scf_options.force_tilesize)
        ao_tile = static_cast<tamm::Tile>(std::ceil(N*0.05));

    IndexSpace AO_ortho{range(0, (size_t)(sys_data.nbf_orig-sys_data.n_lindep))};
    TiledIndexSpace tAO_ortho{AO_ortho,ao_tile};
        
    Tensor<TensorType> C_alpha_tamm{tAO,tAO_ortho};
    Tensor<TensorType> C_beta_tamm{tAO,tAO_ortho};
    Tensor<TensorType> F_alpha_tamm{tAO,tAO};
    Tensor<TensorType> F_beta_tamm{tAO,tAO};

    Tensor<TensorType>::allocate(&exc,C_alpha_tamm,F_alpha_tamm);
    if(is_uhf)
          Tensor<TensorType>::allocate(&exc,C_beta_tamm,F_beta_tamm);

    if (rank == 0) {
      eigen_to_tamm_tensor(C_alpha_tamm,etensors.C);
      eigen_to_tamm_tensor(F_alpha_tamm,etensors.F);
      if(is_uhf) {
        eigen_to_tamm_tensor(C_beta_tamm ,etensors.C_beta);
        eigen_to_tamm_tensor(F_beta_tamm ,etensors.F_beta);
      }
    }

    exc.pg().barrier();

    return std::make_tuple(sys_data, ehf, shells, shell_tile_map, 
      C_alpha_tamm, F_alpha_tamm, C_beta_tamm, F_beta_tamm, tAO, tAOt, scf_conv);
}

#endif // TAMM_METHODS_HF_TAMM_HPP_

