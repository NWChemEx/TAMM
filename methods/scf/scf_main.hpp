
//*******************************************************
// Compute intial guess -> D
// for each iter:
// 1. 2 body fock procedure -> computes G (combined JK)
// 2. [EXC, VXC] = xc_integrator.eval_exc_vxc(D)
// 3. F = H + G
// 4. F += VXC
// 5. E = 0.5 * Tr((H+F) * D)
// 6. E += EXC
// 7. diagonalize F -> updates D
// 8. E += enuc, print E
//*******************************************************

#pragma once

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

//#include "scf_iter.hpp"
#include "scf_taskmap.hpp"

#include <filesystem>
namespace fs = std::filesystem;

#define SCF_THROTTLE_RESOURCES 1

std::tuple<SystemData, double, libint2::BasisSet, std::vector<size_t>, Tensor<TensorType>,
           Tensor<TensorType>, Tensor<TensorType>, Tensor<TensorType>, TiledIndexSpace, TiledIndexSpace, bool>
hartree_fock(ExecutionContext& exc, const string filename, OptionsMap options_map) {

  using libint2::Atom;
  using libint2::Engine;
  using libint2::Operator;
  using libint2::Shell;
  using libint2::BasisSet;

  /*** Setup options ***/

  SystemData sys_data{options_map, options_map.scf_options.scf_type};
  SCFOptions scf_options = sys_data.options_map.scf_options;

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
  bool        is_spherical = (scf_options.gaussian_type == "spherical");
  // bool        sad          = scf_options.sad;
  auto       iter          = 0;
  auto       rank          = exc.pg().rank();
  const bool molden_exists = !scf_options.moldenfile.empty();
  const int  restart_size  = scf_options.restart_size;

  const bool is_uhf = sys_data.is_unrestricted;
  const bool is_rhf = sys_data.is_restricted;
  const bool is_ks  = sys_data.is_ks;
  // const bool is_rohf = sys_data.is_restricted_os;
  if(is_ks && is_uhf) tamm_terminate("UKS-DFT is currently not supported!");

  bool molden_file_valid = false;
  if(molden_exists) {
    molden_file_valid = std::filesystem::exists(scf_options.moldenfile);
    if(!molden_file_valid)
      tamm_terminate("ERROR: moldenfile provided: " + scf_options.moldenfile + " does not exist");
    if(!is_spherical)
      std::cout << "WARNING: molden interface is not tested with gaussian_type:cartesian" << std::endl;
    if(!is_rhf)
      tamm_terminate("ERROR: molden restart is currently only supported for RHF calculations!");
  }

  auto hf_t1 = std::chrono::high_resolution_clock::now();

  // Initialize the Libint integrals library
  libint2::initialize(false);
  // libint2::Shell::do_enforce_unit_normalization(false);

  // Create the basis set
  std::string basis_set_file = std::string(DATADIR) + "/basis/" + basis + ".g94";

  int basis_file_exists = 0;
  if(rank == 0) basis_file_exists = std::filesystem::exists(basis_set_file);
  exc.pg().broadcast(&basis_file_exists, 0);

  if(!basis_file_exists)
    tamm_terminate("ERROR: basis set file " + basis_set_file + " does not exist");

  // If starting guess is from a molden file, read the geometry.
  if(molden_file_valid) read_geom_molden(sys_data, sys_data.options_map.options.atoms);

  auto atoms          = sys_data.options_map.options.atoms;
  auto atom_basis_map = sys_data.options_map.options.atom_basis_map;

  libint2::BasisSet shells;
  {
    std::vector<std::vector<libint2::Shell>> bset_vec(119);
    for (int i = 0; i < atoms.size(); i++) {
      const auto Z = atoms[i].atomic_number;
      std::string _basisname = basis;
      if(atom_basis_map.find( Z ) != atom_basis_map.end())
        _basisname = atom_basis_map[Z];
      else atom_basis_map[Z] = _basisname;
      libint2::BasisSet ashells(_basisname,{atoms[i]});
      bset_vec[Z] = ashells.shells();
      // shells.insert(shells.end(),ashells.begin(),ashells.end());
    }
    libint2::BasisSet bset(atoms,bset_vec);
    shells = std::move(bset);
  }

  if(is_spherical)
    shells.set_pure(true);
  else
    shells.set_pure(false); // use cartesian gaussians

  const size_t N      = shells.nbf();
  auto         nnodes = exc.num_nodes();

  sys_data.nbf      = N;
  sys_data.nbf_orig = N;
  sys_data.ediis    = ediis;

  std::string out_fp    = options_map.options.output_file_prefix + "." + scf_options.basis;
  std::string files_dir = out_fp + "_files/" + sys_data.options_map.scf_options.scf_type + "/scf";
  std::string files_prefix = /*out_fp;*/ files_dir + "/" + out_fp;
  if(!fs::exists(files_dir)) fs::create_directories(files_dir);

  // If using molden file, read the exponents and coefficients and renormalize shells.
  // This modifies the existing basisset object.
  if(molden_exists && molden_file_valid) {
    read_basis_molden(sys_data, shells);
    renormalize_libint_shells(sys_data, shells);
    if(is_spherical)
      shells.set_pure(true);
    else
      shells.set_pure(false); // use cartesian gaussians
  }

    #if SCF_THROTTLE_RESOURCES
      auto [t_nnodes,hf_nnodes,ppn,hf_nranks,sca_nnodes,sca_nranks] = get_hf_nranks(scf_options,N);

#if defined(USE_UPCXX)
      bool in_new_team = (rank < hf_nranks);
      upcxx::team* gcomm = exc.pg().team();
      upcxx::team* hf_comm = new upcxx::team(gcomm->split(
                  in_new_team ? 0 : upcxx::team::color_none, rank.value()));
#else
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

      #if defined(USE_SCALAPACK)
        int lranks[sca_nranks];
        for(int i = 0; i < sca_nranks; i++) lranks[i] = i;
        MPI_Group sca_group;
        MPI_Group_incl(wgroup, sca_nranks, lranks, &sca_group);
        MPI_Comm scacomm;
        MPI_Comm_create(gcomm, sca_group, &scacomm);
      #endif
    #endif

    if(rank == 0) {
#if defined(USE_UPCXX)
      cout << std::endl << "Number of nodes, mpi ranks per node provided: " << nnodes << ", " << (int)(gcomm->rank_n() / nnodes) << endl;
#else
      cout << std::endl << "Number of nodes, mpi ranks per node provided: " << nnodes << ", " << GA_Cluster_nprocs(0) << endl;
#endif

      #if SCF_THROTTLE_RESOURCES
        cout << "Number of nodes, mpi ranks per node used for SCF calculation: " << hf_nnodes << ", " << ppn << endl;
      #endif
      #if defined(USE_SCALAPACK)
        cout << "Number of nodes, mpi ranks per node, total ranks used for Scalapack: " << sca_nnodes
            << ", " << sca_nranks / sca_nnodes << ", " << sca_nranks << endl;      
      #endif
      scf_options.print();
    }

    SCFVars scf_vars; // init vars

    // Compute Nuclear repulsion energy.
    auto [ndocc, enuc]  = compute_NRE(exc, atoms, sys_data.focc);

    // Compute number of electrons.
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

    // Compute non-negligible shell-pair list
    compute_shellpair_list(exc, shells, scf_vars);

    // Setup tiled index spaces
    IndexSpace AO{range(0, N)};
    recompute_tilesize(sys_data.options_map.scf_options.AO_tilesize,N,sys_data.options_map.scf_options.force_tilesize,rank==0);
    std::tie(scf_vars.shell_tile_map, scf_vars.AO_tiles, scf_vars.AO_opttiles) = compute_AO_tiles(exc, sys_data, shells);
    scf_vars.tAO  = {AO, scf_vars.AO_opttiles};
    scf_vars.tAOt = {AO, scf_vars.AO_tiles};
    std::tie(scf_vars.mu, scf_vars.nu, scf_vars.ku)    = scf_vars.tAO.labels<3>("all");
    std::tie(scf_vars.mup, scf_vars.nup, scf_vars.kup) = scf_vars.tAOt.labels<3>("all");

    auto mu = scf_vars.mu, nu = scf_vars.nu, ku = scf_vars.ku;
    auto mup = scf_vars.mup, nup = scf_vars.nup, kup = scf_vars.kup;

    Scheduler schg{exc};
    // Fock matrices allocated on world group
    Tensor<TensorType> Fa_global{scf_vars.tAO, scf_vars.tAO};
    Tensor<TensorType> Fb_global{scf_vars.tAO, scf_vars.tAO};
    schg.allocate(Fa_global);
    if(is_uhf) schg.allocate(Fb_global);
    schg.execute();

    // If a fitting basis is provided, perform the necessary setup
    const auto dfbasisname = scf_options.dfbasis;
    bool is_3c_init = false;
    bool do_density_fitting = false;

    if(!dfbasisname.empty()) do_density_fitting = true;
    if (do_density_fitting) {
      scf_vars.dfbs = BasisSet(dfbasisname, atoms);
      if(is_spherical) scf_vars.dfbs.set_pure(true);
      else scf_vars.dfbs.set_pure(false);  // use cartesian gaussians

      if (rank==0) cout << "density-fitting basis set rank = " << scf_vars.dfbs.nbf() << endl;
      // compute DFBS non-negligible shell-pair list
      #if 0
      {
        //TODO: Doesn't work to screen - revisit
        std::tie(scf_vars.dfbs_shellpair_list, scf_vars.dfbs_shellpair_data) = compute_shellpairs(scf_vars.dfbs);
        size_t nsp = 0;
        for (auto& sp : scf_vars.dfbs_shellpair_list) {
          nsp += sp.second.size();
        }
        if(rank==0) std::cout << "# of {all,non-negligible} DFBS shell-pairs = {"
                  << scf_vars.dfbs.size() * (scf_vars.dfbs.size() + 1) / 2 << "," << nsp << "}"
                  << endl;
      }
      #endif

      sys_data.ndf = scf_vars.dfbs.nbf();
      scf_vars.dfAO = IndexSpace{range(0, sys_data.ndf)};
      recompute_tilesize(sys_data.options_map.scf_options.dfAO_tilesize,sys_data.ndf,sys_data.options_map.scf_options.force_tilesize,rank==0);
      std::tie(scf_vars.df_shell_tile_map, scf_vars.dfAO_tiles, scf_vars.dfAO_opttiles) = compute_AO_tiles(exc, sys_data, scf_vars.dfbs, true);
    
      scf_vars.tdfAO  = TiledIndexSpace{scf_vars.dfAO, scf_vars.dfAO_opttiles};
      scf_vars.tdfAOt = TiledIndexSpace{scf_vars.dfAO, scf_vars.dfAO_tiles};
      
    }
    std::unique_ptr<DFFockEngine> dffockengine(
        do_density_fitting ? new DFFockEngine(shells, scf_vars.dfbs) : nullptr);
    // End setup for fitting basis

    auto hf_t2 = std::chrono::high_resolution_clock::now();
    double hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
    if(rank == 0) std::cout << std::fixed << std::setprecision(2) << std::endl << "Time for initial setup: " << hf_time << " secs" << endl;

    double ehf = 0.0; // initialize Hartree-Fock energy

    exc.pg().barrier();

    EigenTensors etensors;

    const bool scf_conv = restart && scf_options.noscf; 
    const int  max_hist = sys_data.options_map.scf_options.diis_hist; 

    scf_restart_test(exc, sys_data, filename, restart, files_prefix);

    std::string movecsfile_alpha  = files_prefix + ".alpha.movecs";
    std::string densityfile_alpha = files_prefix + ".alpha.density";
    std::string movecsfile_beta  =  files_prefix + ".beta.movecs";       
    std::string densityfile_beta =  files_prefix + ".beta.density"; 

    #if SCF_THROTTLE_RESOURCES
    if (rank < hf_nranks) {
      ScalapackInfo scalapack_info;

#if defined(USE_UPCXX)
      ProcGroup pg = ProcGroup::create_coll(*hf_comm);
#else
      EXPECTS(hf_comm != MPI_COMM_NULL);
      ProcGroup pg = ProcGroup::create_coll(hf_comm);
#endif
      ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
    #else 
      //TODO: Fix - create ec_m, when throttle is disabled
      ExecutionContext& ec = exc;
    #endif
      // ProcGroup pg_l = ProcGroup::create_coll(MPI_COMM_SELF);
      // ExecutionContext ec_l{pg_l, DistributionKind::nw, MemoryManagerKind::local};
    #if defined(USE_SCALAPACK)
      #if defined(USE_UPCXX)
      abort(); // Not supported with UPC++
      #endif
      scalapack_info.comm = scacomm;
      if(scacomm != MPI_COMM_NULL) {

      auto blacs_setup_st = std::chrono::high_resolution_clock::now();
      // Sanity checks
      scalapack_info.npr = scf_options.scalapack_np_row;
      scalapack_info.npc = scf_options.scalapack_np_col;
      int scalapack_nranks = scalapack_info.npr * scalapack_info.npc;

      scalapack_info.pg = ProcGroup::create_coll(scacomm);
      scalapack_info.ec = ExecutionContext{scalapack_info.pg, DistributionKind::dense, MemoryManagerKind::ga};

      int sca_world_size = scalapack_info.pg.size().value();

      // Default to square(ish) grid
      if( scalapack_nranks == 0 ) {
        int64_t npr = std::sqrt( sca_world_size );
        int64_t npc = sca_world_size / npr; 
        while( npr * npc != sca_world_size ) {
          npr--;
          npc = sca_world_size / npr;
        }
        scalapack_nranks = sca_world_size;
        scalapack_info.npr = npr;
        scalapack_info.npc = npc;        
      }

      EXPECTS( sca_world_size >= scalapack_nranks );

      if( not scalapack_nranks ) scalapack_nranks = sca_world_size;
      std::vector<int64_t> scalapack_ranks( scalapack_nranks );
      std::iota( scalapack_ranks.begin(), scalapack_ranks.end(), 0 );
      scalapack_info.scalapack_nranks = scalapack_nranks;

      if(scalapack_info.pg.rank() == 0) {
        std::cout << "scalapack_nranks = " << scalapack_nranks << std::endl;
        std::cout << "scalapack_np_row = " << scalapack_info.npr << std::endl;
        std::cout << "scalapack_np_col = " << scalapack_info.npc << std::endl;
      }

      int& mb_ = scf_options.scalapack_nb;
      if(mb_ > N / scalapack_info.npr) {
        mb_                                           = N / scalapack_info.npr;
        sys_data.options_map.scf_options.scalapack_nb = mb_;
        if(scalapack_info.pg.rank() == 0)
          std::cout << "WARNING: Resetting scalapack block size (scalapack_nb) to: " << mb_
                    << std::endl;
      }

      scalapack_info.blacs_grid =
        std::make_unique<blacspp::Grid>(scalapack_info.pg.comm(), scalapack_info.npr, scalapack_info.npc,
                                        scalapack_ranks.data(), scalapack_info.npr);
      scalapack_info.blockcyclic_dist = std::make_unique<scalapackpp::BlockCyclicDist2D>(
        *scalapack_info.blacs_grid, mb_, mb_, 0, 0);

      auto blacs_setup_en = std::chrono::high_resolution_clock::now();

      std::chrono::duration<double> blacs_time = blacs_setup_en - blacs_setup_st;

      if(scalapack_info.pg.rank() == 0)
        std::cout << std::fixed << std::setprecision(2) << std::endl
                  << "Time for BLACS setup: " << blacs_time.count() << " secs" << std::endl;
      }
    #endif // USE_SCALAPACK

    #if defined(USE_GAUXC)
    /*** =========================== ***/
    /*** Setup GauXC types           ***/
    /*** =========================== ***/
    auto gauxc_mol   = gauxc_util::make_gauxc_molecule( atoms  );
    auto gauxc_basis = gauxc_util::make_gauxc_basis   ( shells );

    GauXC::MolGrid 
      gauxc_molgrid( GauXC::AtomicGridSizeDefault::UltraFineGrid, gauxc_mol );
    auto gauxc_molmeta = std::make_shared<GauXC::MolMeta>( gauxc_mol );
    auto gauxc_lb      = std::make_shared<GauXC::LoadBalancer>(ec.pg().comm(),
      gauxc_mol, gauxc_molgrid, gauxc_basis, gauxc_molmeta
    );

    std::string xc_string = scf_options.xc_type;
    // TODO: Refactor DFT code path when we eventually enable GauXC by default.
    // is_ks=false, so we setup, but do not run DFT. 
    if(xc_string.empty()) xc_string = "pbe0";
    std::transform( xc_string.begin(), xc_string.end(), xc_string.begin(), ::toupper );
    GauXC::functional_type gauxc_func( ExchCXX::Backend::builtin,
                                       ExchCXX::functional_map.value(xc_string),
                                       ExchCXX::Spin::Unpolarized );
    GauXC::XCIntegrator<Matrix> gauxc_integrator( GauXC::ExecutionSpace::Host,
      ec.pg().comm(), gauxc_func, gauxc_basis, gauxc_lb );

    // TODO
    const double xHF = is_ks ? gauxc_func.hyb_exx() : 1.;
    if (rank == 0) cout << "HF exch = " << xHF << endl;
    #else
    const double xHF = 1.;
    #endif

    ec.pg().barrier();

    
      Scheduler sch{ec};

      TAMMTensors ttensors;
      const TiledIndexSpace& tAO  = scf_vars.tAO;
      const TiledIndexSpace& tAOt = scf_vars.tAOt;

      /*** =========================== ***/
      /*** compute 1-e integrals       ***/
      /*** =========================== ***/
      compute_hamiltonian<TensorType>(ec,scf_vars,atoms,shells,ttensors,etensors);

      /*** =========================== ***/
      /*** build initial-guess density ***/
      /*** =========================== ***/

      std::string ortho_file = files_prefix + ".orthogonalizer";   
      std::string ortho_jfile = ortho_file+".json";

      if(N >= restart_size && fs::exists(ortho_file)) {
        if(rank==0) {
          cout << "Reading orthogonalizer from disk ..." << endl << endl;
          auto jX = json_from_file(ortho_jfile);
          auto Xdims = jX["ortho_dims"].get<std::vector<int>>();          
          sys_data.n_lindep = sys_data.nbf_orig - Xdims[1];
        }
        ec.pg().broadcast(&sys_data.n_lindep,0);
        sys_data.nbf = sys_data.nbf_orig - sys_data.n_lindep; // Compute Northo

        scf_vars.tAO_ortho = TiledIndexSpace{IndexSpace{range(0, (size_t)(sys_data.nbf))},
                                            sys_data.options_map.scf_options.AO_tilesize};

        #if defined(USE_SCALAPACK)
        {
          const tamm::Tile _mb= scf_options.scalapack_nb; //(scalapack_info.blockcyclic_dist)->mb();
          scf_vars.tN_bc      = TiledIndexSpace{IndexSpace{range(sys_data.nbf_orig)}, _mb};
          scf_vars.tNortho_bc = TiledIndexSpace{IndexSpace{range(sys_data.nbf)}, _mb};
          if(scacomm != MPI_COMM_NULL) {
            ttensors.X_alpha    = {scf_vars.tN_bc, scf_vars.tNortho_bc};
            ttensors.X_alpha.set_block_cyclic({scalapack_info.npr, scalapack_info.npc});
            Tensor<TensorType>::allocate(&scalapack_info.ec, ttensors.X_alpha);
            read_from_disk<TensorType>(ttensors.X_alpha,ortho_file);
            if(is_uhf) {
              ttensors.X_beta = {scf_vars.tN_bc, scf_vars.tNortho_bc};
              ttensors.X_beta.set_block_cyclic({scalapack_info.npr, scalapack_info.npc});
              Tensor<TensorType>::allocate(&scalapack_info.ec, ttensors.X_beta);
              Scheduler{scalapack_info.ec}(ttensors.X_beta() = ttensors.X_alpha()).execute();
              // read_from_disk<TensorType>(ttensors.X_beta,ortho_file);
            }
          }
        }
        #else
          ttensors.X_alpha = {scf_vars.tAO, scf_vars.tAO_ortho};
          if(is_uhf) ttensors.X_beta = {scf_vars.tAO, scf_vars.tAO_ortho};
          sch.allocate(ttensors.X_alpha).execute();
          if(is_uhf) sch.allocate(ttensors.X_beta).execute();
          read_from_disk<TensorType>(ttensors.X_alpha,ortho_file);
          if(is_uhf) read_from_disk<TensorType>(ttensors.X_beta,ortho_file);
        #endif
      }
      else 
      {
        compute_orthogonalizer(ec, sys_data, scf_vars, scalapack_info, ttensors);
        // sys_data.nbf = sys_data.nbf_orig - sys_data.n_lindep;
        if(rank == 0) {
          json jX;
          jX["ortho_dims"] = {sys_data.nbf_orig, sys_data.nbf};
          json_to_file(jX,ortho_jfile);
        }
        if(N >= restart_size) {
          #if defined(USE_SCALAPACK)
            if(scacomm != MPI_COMM_NULL) write_to_disk<TensorType>(ttensors.X_alpha, ortho_file);
          #else
            write_to_disk<TensorType>(ttensors.X_alpha, ortho_file);
          #endif
        }
      }

      #if defined(USE_SCALAPACK)
      if(scacomm != MPI_COMM_NULL) {
        ttensors.F_BC = {scf_vars.tN_bc, scf_vars.tN_bc};
        ttensors.F_BC.set_block_cyclic({scalapack_info.npr, scalapack_info.npc});
        Tensor<TensorType>::allocate(&scalapack_info.ec, ttensors.F_BC);
      }
      #endif


      etensors.D       = Matrix::Zero(N,N);
      etensors.G       = Matrix::Zero(N,N);
      if(is_uhf) {
        etensors.D_beta  = Matrix::Zero(N,N);
        etensors.G_beta  = Matrix::Zero(N,N);
      }

      // pre-compute data for Schwarz bounds
      std::string schwarz_matfile = files_prefix + ".schwarz";
      Matrix      SchwarzK;
      if(N >= restart_size && fs::exists(schwarz_matfile)) {
        if(rank == 0) cout << "Read Schwarz matrix from disk... " << endl;
        SchwarzK = read_scf_mat<TensorType>(schwarz_matfile);
      }
      else {
        if(rank == 0) cout << "pre-computing data for Schwarz bounds... " << endl;
        SchwarzK = compute_schwarz_ints<>(ec, scf_vars, shells);
        if(rank == 0) write_scf_mat<TensorType>(SchwarzK, schwarz_matfile);
      }

      hf_t1 = std::chrono::high_resolution_clock::now();

      ttensors.ehf_tamm    = Tensor<TensorType>{};
      ttensors.F_dummy     = {tAOt, tAOt}; //not allocated

      ttensors.ehf_tmp     = {tAO, tAO};
      ttensors.F_alpha     = {tAO, tAO};
      ttensors.D_tamm      = {tAO, tAO};
      ttensors.D_diff      = {tAO, tAO};
      ttensors.D_last_tamm = {tAO, tAO};
      ttensors.F_alpha_tmp = {tAO, tAO};
      ttensors.FD_tamm     = {tAO, tAO}; 
      ttensors.FDS_tamm    = {tAO, tAO};

      // XXX: Enable only for DFT
      ttensors.VXC = {tAO, tAO};

      if(is_uhf) {
        ttensors.ehf_beta_tmp     = {tAO, tAO};
        ttensors.F_beta           = {tAO, tAO};
        ttensors.D_beta_tamm      = {tAO, tAO};
        ttensors.D_last_beta_tamm = {tAO, tAO};    
        ttensors.F_beta_tmp       = {tAO, tAO};
        ttensors.FD_beta_tamm     = {tAO, tAO}; 
        ttensors.FDS_beta_tamm    = {tAO, tAO};    
      }

#if defined(USE_UPCXX_DISTARRAY)
      ec.set_memory_manager_cache(1);
#endif

      Tensor<TensorType>::allocate(&ec, ttensors.F_alpha     , 
                                        ttensors.D_tamm , ttensors.D_last_tamm, ttensors.D_diff  ,
                                        ttensors.F_alpha_tmp , ttensors.ehf_tmp    , ttensors.ehf_tamm,
                                        ttensors.FD_tamm, ttensors.FDS_tamm);
      if(is_uhf) 
        Tensor<TensorType>::allocate(&ec, ttensors.F_beta     , 
                                          ttensors.D_beta_tamm , ttensors.D_last_beta_tamm, 
                                          ttensors.F_beta_tmp , ttensors.ehf_beta_tmp    ,
                                          ttensors.FD_beta_tamm, ttensors.FDS_beta_tamm);


      // XXX: Only allocate for DFT
      if(is_ks) Tensor<TensorType>::allocate( &ec, ttensors.VXC );

      const auto do_schwarz_screen = SchwarzK.cols() != 0 && SchwarzK.rows() != 0;
      // engine precision controls primitive truncation, assume worst-case scenario
      // (all primitive combinations add up constructively)
      const libint2::BasisSet& obs = shells;
      auto max_nprim  = obs.max_nprim();
      auto max_nprim4 = max_nprim * max_nprim * max_nprim * max_nprim;
      auto shell2bf   = obs.shell2bf();

      if (restart) {
        scf_restart(ec, sys_data, filename, etensors, files_prefix);
        if(rank == 0){
          eigen_to_tamm_tensor(ttensors.X_alpha, etensors.C); // Xa = Ca;
          if(is_uhf) 
            eigen_to_tamm_tensor(ttensors.X_beta, etensors.C_beta); // Xb = Cb;
        }
      }
      else if(molden_exists) {
        auto N = sys_data.nbf_orig;
        auto Northo = sys_data.nbf;

        etensors.C.setZero(N,Northo);
        if(is_uhf) etensors.C_beta.setZero(N,Northo);

        if(rank == 0) {
          cout << endl << "Reading from molden file provided ..." << endl;
          if(molden_file_valid) {
            read_molden<TensorType>(sys_data,shells,etensors.C,etensors.C_beta);
          }
        }

        // compute density
        if( rank == 0 ) {
          if(is_rhf) {
            auto C_occ   = etensors.C.leftCols(sys_data.nelectrons_alpha);
            etensors.D = 2.0 * C_occ * C_occ.transpose();
          }
          else if(is_uhf) {
            auto C_occ   = etensors.C.leftCols(sys_data.nelectrons_alpha);
            etensors.D = C_occ * C_occ.transpose();
            C_occ   = etensors.C_beta.leftCols(sys_data.nelectrons_beta);
            etensors.D_beta  = C_occ * C_occ.transpose();
          }
        }

        if(rank == 0){
          eigen_to_tamm_tensor(ttensors.X_alpha, etensors.C); // Xa = Ca;
          if(is_uhf) 
            eigen_to_tamm_tensor(ttensors.X_beta, etensors.C_beta); // Xb = Cb;
        }

        ec.pg().barrier(); 
      }
      else {
        // TODO: WIP
        #if 0
        if(sad) {
          if(rank==0) cout << "SAD enabled" << endl;

          compute_sad_guess<TensorType>(ec, sys_data, atoms, shells, basis, 
                                       is_spherical, etensors, charge, multiplicity); 
          compute_2bf<TensorType>(ec, sys_data, scf_vars, obs, do_schwarz_screen, shell2bf, SchwarzK,
                                         max_nprim4,shells, ttensors, etensors, false, do_density_fitting);
          sch
            (ttensors.F_alpha()  = ttensors.H1())
            (ttensors.F_alpha() += ttensors.F_alpha_tmp())
            .execute();
          Matrix Fa_eig = tamm_to_eigen_matrix(ttensors.F_alpha);
          Matrix X_eig  = tamm_to_eigen_matrix(ttensors.X_alpha);
          Eigen::SelfAdjointEigenSolver<Matrix> eig_solver_guess_a(X_eig.transpose() * Fa_eig * X_eig);
          auto C_alpha = X_eig * eig_solver_guess_a.eigenvectors();
          auto C_occ_a = C_alpha.leftCols(sys_data.nelectrons_alpha);
          if(is_rhf) 
            etensors.D = 2.0 * C_occ_a * C_occ_a.transpose();
          if(is_uhf) {
            etensors.D = C_occ_a * C_occ_a.transpose();
            sch
              (ttensors.F_beta()  = ttensors.H1())
              (ttensors.F_beta() += ttensors.F_beta_tmp())
              .execute();
            Matrix Fb_eig = tamm_to_eigen_matrix(ttensors.F_beta);
            Eigen::SelfAdjointEigenSolver<Matrix> eig_solver_guess_b(X_eig.transpose() * Fb_eig * X_eig);
            auto C_beta  = X_eig * eig_solver_guess_b.eigenvectors();
            auto C_occ_b = C_beta.leftCols(sys_data.nelectrons_beta);
            etensors.D_beta = C_occ_b * C_occ_b.transpose();
          }
        }
        else
        #endif
        {
          auto [s1vec,s2vec,ntask_vec] = compute_initial_guess_taskinfo<TensorType>
                                    (ec, sys_data, scf_vars, atoms, shells, basis, is_spherical,
                                            etensors, ttensors, charge, multiplicity);

          auto [s1_all,s2_all,ntasks_all] = gather_task_vectors<TensorType>(ec,s1vec,s2vec,ntask_vec);

          int tmdim = 0;
          if(rank == 0)
          {
              Loads dummyLoads;
              /***generate load balanced task map***/
              readLoads(s1_all, s2_all, ntasks_all, dummyLoads);
              simpleLoadBal(dummyLoads,ec.pg().size().value());
              tmdim = std::max(dummyLoads.maxS1,dummyLoads.maxS2);
              etensors.taskmap.resize(tmdim+1,tmdim+1);
              for(int i=0;i<tmdim+1;i++) {
                for(int j=0;j<tmdim+1;j++) {
                  // value in this array is the rank that executes task i,j
                  // -1 indicates a task i,j that can be skipped
                  etensors.taskmap(i,j) = -1;
                }
              }
              createTaskMap(etensors.taskmap,dummyLoads);
          }

          ec.pg().broadcast(&tmdim,0);
          if(rank!=0) etensors.taskmap.resize(tmdim+1,tmdim+1);
          ec.pg().broadcast(etensors.taskmap.data(),etensors.taskmap.size(),0);

          compute_initial_guess<TensorType>(ec, scalapack_info,
                  sys_data, scf_vars, atoms, shells, basis, is_spherical,
                  etensors, ttensors, charge, multiplicity);

          etensors.taskmap.resize(0,0);
          if(rank == 0) {
            write_scf_mat<TensorType>(etensors.C, movecsfile_alpha);
            write_scf_mat<TensorType>(etensors.D, densityfile_alpha);
            if(is_uhf) {
              write_scf_mat<TensorType>(etensors.C_beta, movecsfile_beta);
              write_scf_mat<TensorType>(etensors.D_beta, densityfile_beta);
            }
          }
          ec.pg().barrier();
        } //initial guess
      }

      hf_t2   = std::chrono::high_resolution_clock::now();
      hf_time = std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
      if(rank == 0) std::cout << std::fixed << std::setprecision(2) << "Total Time to compute initial guess: " << hf_time << " secs" << endl;

      /*** =========================== ***/
      /*** main iterative loop         ***/
      /*** =========================== ***/
      double rmsd          = 1.0;
      double ediff         = 0.0;
      bool   is_conv       = true;

      if(rank == 0) {
        eigen_to_tamm_tensor(ttensors.D_tamm,etensors.D);
        if(is_uhf) {
          eigen_to_tamm_tensor(ttensors.D_beta_tamm,etensors.D_beta);
        }
      }

      ec.pg().broadcast(etensors.D.data(), etensors.D.size(),0);
      if(is_uhf) ec.pg().broadcast(etensors.D_beta.data(), etensors.D_beta.size(),0);
      
      if(rank == 0 && scf_options.debug && N < restart_size) {
        Matrix S(sys_data.nbf_orig,sys_data.nbf_orig);
        tamm_to_eigen_tensor(ttensors.S1,S);
        if(is_rhf) 
          cout << "debug #electrons       = " << (etensors.D*S).trace()      << endl;
        if(is_uhf) {
          cout << "debug #alpha electrons = " << (etensors.D*S).trace()      << endl;
          cout << "debug #beta  electrons = " << (etensors.D_beta*S).trace() << endl;
        }
      }
    
      // Setup tiled index spaces when a fitting basis is provided
      IndexSpace dfCocc{range(0,sys_data.nelectrons_alpha)}; 
      scf_vars.tdfCocc = {dfCocc,sys_data.options_map.scf_options.dfAO_tilesize};
      std::tie(scf_vars.dCocc_til) = scf_vars.tdfCocc.labels<1>("all");
      if(do_density_fitting){
        std::tie(scf_vars.d_mu, scf_vars.d_nu, scf_vars.d_ku)    = scf_vars.tdfAO.labels<3>("all");
        std::tie(scf_vars.d_mup, scf_vars.d_nup, scf_vars.d_kup) = scf_vars.tdfAOt.labels<3>("all");

        ttensors.Zxy_tamm = Tensor<TensorType>{scf_vars.tdfAO, tAO, tAO}; //ndf,n,n
        ttensors.xyK_tamm = Tensor<TensorType>{tAO, tAO, scf_vars.tdfAO}; //n,n,ndf
        ttensors.C_occ_tamm = Tensor<TensorType>{tAO,scf_vars.tdfCocc}; //n,nocc
        Tensor<TensorType>::allocate(&ec, ttensors.xyK_tamm, ttensors.C_occ_tamm,ttensors.Zxy_tamm);
      }

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

      /*** Generate task mapping ***/

      //Collect task info
      auto [s1vec,s2vec,ntask_vec] = compute_2bf_taskinfo<TensorType>
                     (ec, sys_data, scf_vars, obs, 
                      do_schwarz_screen, shell2bf, SchwarzK,
                      max_nprim4,shells, ttensors, etensors, do_density_fitting);

      auto [s1_all,s2_all,ntasks_all] = gather_task_vectors<TensorType>(ec,s1vec,s2vec,ntask_vec);

      int tmdim = 0;
      if(rank == 0)
      {
          Loads dummyLoads;
          /***generate load balanced task map***/
          readLoads(s1_all, s2_all, ntasks_all, dummyLoads);
          simpleLoadBal(dummyLoads,ec.pg().size().value());
          tmdim = std::max(dummyLoads.maxS1,dummyLoads.maxS2);
          etensors.taskmap.resize(tmdim+1,tmdim+1);
          // value in this array is the rank that executes task i,j
          // -1 indicates a task i,j that can be skipped
          etensors.taskmap.setConstant(-1);
          //cout<<"creating task map"<<endl;
          createTaskMap(etensors.taskmap,dummyLoads);
          //cout<<"task map creation completed"<<endl;

      }

      ec.pg().broadcast(&tmdim,0);
      if(rank!=0) etensors.taskmap.resize(tmdim+1,tmdim+1);
      ec.pg().broadcast(etensors.taskmap.data(),etensors.taskmap.size(),0);
      
      if(restart || molden_exists) {
        
        sch
          (ttensors.F_alpha_tmp() = 0)
          .execute();

        if(is_uhf) {
          sch
            (ttensors.F_beta_tmp() = 0)
            .execute();
        }
        //F1 = H1 + F_alpha_tmp
        compute_2bf<TensorType>(ec, sys_data, scf_vars, obs, do_schwarz_screen, shell2bf, SchwarzK,
                                max_nprim4, shells, ttensors, etensors, is_3c_init, do_density_fitting, xHF);        

        TensorType gauxc_exc = 0.;
        #if defined(USE_GAUXC)
        if(is_ks) {
          gauxc_exc = gauxc_util::compute_xcf<TensorType>( ec, ttensors, etensors, gauxc_integrator );
        }
        #endif

        // ehf = D * (H1+F1);
        if(is_rhf) {
          sch
            (ttensors.ehf_tmp(mu,nu)  = 2.0 * ttensors.H1(mu,nu))
            (ttensors.ehf_tmp(mu,nu) += 1.0 * ttensors.F_alpha_tmp(mu,nu))
            (ttensors.ehf_tamm()      = 1.0 * ttensors.D_tamm() * ttensors.ehf_tmp())
            .execute();
        }
        if(is_uhf) {
          sch
            (ttensors.ehf_tmp(mu,nu)  = 2.0 * ttensors.H1(mu,nu))
            (ttensors.ehf_tmp(mu,nu) += 1.0 * ttensors.F_alpha_tmp(mu,nu))
            (ttensors.ehf_tamm()      = 1.0 * ttensors.D_tamm() * ttensors.ehf_tmp())
            (ttensors.ehf_tmp(mu,nu)  = 2.0 * ttensors.H1(mu,nu))
            (ttensors.ehf_tmp(mu,nu) += 1.0 * ttensors.F_beta_tmp(mu,nu))
            (ttensors.ehf_tamm()     += 1.0 * ttensors.D_beta_tamm() * ttensors.ehf_tmp())
            .execute();
        }

        ehf = 0.5*get_scalar(ttensors.ehf_tamm) + enuc + gauxc_exc;
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
          (ttensors.F_alpha_tmp() = 0)
          (ttensors.D_last_tamm(mu,nu) = ttensors.D_tamm(mu,nu))
          .execute();
        
        if(is_uhf) {
          sch
            (ttensors.F_beta_tmp() = 0)
            (ttensors.D_last_beta_tamm(mu,nu) = ttensors.D_beta_tamm(mu,nu))
            .execute();
        }
            
        // auto D_tamm_nrm = norm(ttensors.D_tamm);
        // if(rank==0) cout << std::setprecision(18) << "norm of D_tamm: " << D_tamm_nrm << endl;

        // build a new Fock matrix
        compute_2bf<TensorType>(ec, sys_data, scf_vars, obs, do_schwarz_screen, shell2bf, SchwarzK,
                                max_nprim4, shells, ttensors, etensors, is_3c_init, do_density_fitting, xHF);

        //E_Diis
        if(ediis) {
          Tensor<TensorType>  Dcopy{tAO,tAO};
          Tensor<TensorType>  Fcopy{tAO, tAO};
          Tensor<TensorType>  ehf_tamm_copy{};
          Tensor<TensorType>::allocate(&ec,Dcopy,Fcopy,ehf_tamm_copy);
          sch
            (Dcopy()  = ttensors.D_tamm())
            (Fcopy()  = ttensors.F_alpha_tmp())
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
          energy_diis(ec, tAO, iter, max_hist, ttensors.D_tamm, ttensors.F_alpha, ttensors.ehf_tamm,
                      ttensors.D_hist, ttensors.fock_hist, ttensors.ehf_tamm_hist);
        }

        std::tie(ehf,rmsd) = scf_iter_body<TensorType>(ec, scalapack_info,
                        iter, sys_data, scf_vars, ttensors, etensors, ediis, 
                        #if defined(USE_GAUXC)
                        gauxc_integrator, 
                        #endif
                        scf_conv);

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

          sys_data.results["output"]["SCF"]["iter"][std::to_string(iter)] = { {"energy", ehf}, {"e_diff", ediff}, {"rmsd", rmsd} };
          sys_data.results["output"]["SCF"]["iter"][std::to_string(iter)]["performance"] = { {"total_time", loop_time} };

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

      if(is_ks) { //or rohf
        sch(ttensors.F_alpha_tmp() = 0).execute();
        if(is_uhf) sch(ttensors.F_beta_tmp() = 0).execute();

        // build a new Fock matrix
        compute_2bf<TensorType>(ec, sys_data, scf_vars, obs, do_schwarz_screen, shell2bf, SchwarzK,
                                max_nprim4, shells, ttensors, etensors, is_3c_init, do_density_fitting, 1.0);

        sch
          (ttensors.F_alpha()  = ttensors.H1())
          (ttensors.F_alpha() += ttensors.F_alpha_tmp())
          .execute();
        
        if(is_uhf) {
          sch
            (ttensors.F_beta()   = ttensors.H1())
            (ttensors.F_beta()  += ttensors.F_beta_tmp())
            .execute();
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

      // copy to fock matrices allocated on world group
      sch(Fa_global(mu,nu) = ttensors.F_alpha(mu,nu));
      if(is_uhf) sch(Fb_global(mu,nu) = ttensors.F_beta(mu,nu));
      sch.execute();

      if(do_density_fitting) Tensor<TensorType>::deallocate(ttensors.xyK_tamm, ttensors.C_occ_tamm, ttensors.Zxy_tamm);

      if(scf_options.print_mos.first)
        write_to_disk<TensorType>(ttensors.H1, files_prefix + ".hcore");

      Tensor<TensorType>::deallocate(ttensors.H1     , ttensors.S1      , ttensors.T1         , ttensors.V1,
                                     ttensors.F_alpha_tmp , ttensors.ehf_tmp , ttensors.ehf_tamm   , ttensors.F_alpha,
                                     ttensors.D_tamm , ttensors.D_diff  , ttensors.D_last_tamm,
                                     ttensors.FD_tamm, ttensors.FDS_tamm);
      
      if(is_uhf) 
        Tensor<TensorType>::deallocate(ttensors.F_beta,
                                       ttensors.D_beta_tamm , ttensors.D_last_beta_tamm,
                                       ttensors.F_beta_tmp , ttensors.ehf_beta_tmp    ,
                                       ttensors.FD_beta_tamm, ttensors.FDS_beta_tamm   );

      if(is_ks) {
        if(rank==0) etensors.VXC = tamm_to_eigen_matrix(ttensors.VXC);
        Tensor<TensorType>::deallocate(ttensors.VXC);
      }

      #if SCF_THROTTLE_RESOURCES
      ec.flush_and_sync();
      #endif


      #if defined(USE_SCALAPACK)
      #if defined(USE_UPCXX)
      abort(); // Not supported currently in UPC++
      #endif
      if(scalapack_info.comm != MPI_COMM_NULL) {
        Tensor<TensorType>::deallocate(ttensors.F_BC, ttensors.X_alpha);
        if(is_uhf) Tensor<TensorType>::deallocate(ttensors.X_beta);
        scalapack_info.ec.flush_and_sync();
      }
      // Free up created comms / groups
      // MPI_Comm_free( &scalapack_comm );
      // MPI_Group_free( &scalapack_group );
      // MPI_Group_free( &world_group );
      #else
        sch.deallocate(ttensors.X_alpha);
        if(is_uhf) sch.deallocate(ttensors.X_beta);
        sch.execute();
      #endif
    
    #if SCF_THROTTLE_RESOURCES

    } //end scaled down process group

    #if defined(USE_UPCXX)
        hf_comm->destroy();
    #endif

    #endif

    //C,F1 is not allocated for ranks > hf_nranks 
    exc.pg().barrier();

    //F, C are not deallocated.
    sys_data.n_occ_alpha = sys_data.nelectrons_alpha;
    sys_data.n_occ_beta  = sys_data.nelectrons_beta;
    sys_data.n_vir_alpha = sys_data.nbf_orig - sys_data.n_occ_alpha - sys_data.n_lindep;
    sys_data.n_vir_beta  = sys_data.nbf_orig - sys_data.n_occ_beta - sys_data.n_lindep;

    exc.pg().broadcast(&ehf                 ,0);
    exc.pg().broadcast(&sys_data.nbf        ,0);  
    exc.pg().broadcast(&sys_data.n_lindep   ,0);
    exc.pg().broadcast(&sys_data.n_occ_alpha,0);
    exc.pg().broadcast(&sys_data.n_vir_alpha,0);
    exc.pg().broadcast(&sys_data.n_occ_beta ,0);
    exc.pg().broadcast(&sys_data.n_vir_beta ,0);

    sys_data.update();
    if(rank==0 && debug) sys_data.print();
    // sys_data.input_molecule = getfilename(filename);
    sys_data.scf_energy = ehf;
    // iter not broadcasted, but fine since only rank 0 writes to json
    if(rank == 0) {
      sys_data.results["output"]["SCF"]["final_energy"] = ehf;
      sys_data.results["output"]["SCF"]["n_iterations"] = iter;
    }

    IndexSpace AO_ortho{range(0, (size_t)(sys_data.nbf_orig-sys_data.n_lindep))};
    TiledIndexSpace tAO_ortho{AO_ortho,sys_data.options_map.scf_options.AO_tilesize};
        
    Tensor<TensorType> C_alpha_tamm{scf_vars.tAO,tAO_ortho};
    Tensor<TensorType> C_beta_tamm{scf_vars.tAO,tAO_ortho};
    vxc_tamm = Tensor<TensorType>{scf_vars.tAO,scf_vars.tAO};

#if defined(USE_UPCXX_DISTARRAY)
    exc.set_memory_manager_cache(1);
#endif

    schg.allocate(C_alpha_tamm);
    if(is_uhf) schg.allocate(C_beta_tamm);
    if(is_ks) schg.allocate(vxc_tamm);
    schg.execute();
#if defined(USE_UPCXX_DISTARRAY)
    exc.set_memory_manager_cache(); // resets cache to pg.size().value();
#endif

    if (rank == 0) {
      eigen_to_tamm_tensor(C_alpha_tamm, etensors.C);
      if(is_uhf) eigen_to_tamm_tensor(C_beta_tamm, etensors.C_beta);
      if(is_ks) eigen_to_tamm_tensor(vxc_tamm, etensors.VXC);
    }

    exc.pg().barrier();

    return std::make_tuple(sys_data, ehf, shells, scf_vars.shell_tile_map, C_alpha_tamm,
                           Fa_global, C_beta_tamm, Fb_global, scf_vars.tAO,
                           scf_vars.tAOt, scf_conv);
}

