
#ifndef TAMM_METHODS_HF_TAMM_COMMON_HPP_
#define TAMM_METHODS_HF_TAMM_COMMON_HPP_

#include "scf_guess.hpp"

//TODO: UHF,ROHF,diis,3c,dft

template<typename TensorType>
void diis(ExecutionContext& ec, TiledIndexSpace& tAO, Tensor<TensorType> D, Tensor<TensorType> F, Tensor<TensorType> err_mat, 
          int iter, int max_hist, int ndiis, const int n_lindep,
          std::vector<Tensor<TensorType>>& diis_hist, std::vector<Tensor<TensorType>>& fock_hist);

template<typename TensorType>
void energy_diis(ExecutionContext& ec, TiledIndexSpace& tAO, int iter, int max_hist, 
                 Tensor<TensorType> D, Tensor<TensorType> F, std::vector<Tensor<TensorType>>& D_hist, 
                 std::vector<Tensor<TensorType>>& fock_hist, std::vector<Tensor<TensorType>>& ehf_tamm_hist);

template<typename TensorType>
void compute_1body_ints(ExecutionContext& ec, Tensor<TensorType>& tensor1e, 
      std::vector<libint2::Atom>& atoms, libint2::BasisSet& shells, libint2::Operator otype,
      std::vector<size_t>& shell_tile_map, std::vector<Tile>& AO_tiles);

std::tuple<int,int,int,int> get_hf_nranks(const size_t N) {

    // auto nranks = GA_Nnodes();
    auto nnodes = GA_Cluster_nnodes();
    auto ppn = GA_Cluster_nprocs(0);

    int hf_guessranks = std::ceil(0.3*N);
    int hf_nnodes = hf_guessranks/ppn;
    if(hf_guessranks%ppn>0 || hf_nnodes==0) hf_nnodes++;
    if(hf_nnodes > nnodes) hf_nnodes = nnodes;
    int hf_nranks = hf_nnodes * ppn;

    return std::make_tuple(nnodes,hf_nnodes,ppn,hf_nranks);
}

void compute_shellpair_list(const ExecutionContext& ec, const libint2::BasisSet& shells){
    
    auto rank = ec.pg().rank(); 

    // compute OBS non-negligible shell-pair list
    std::tie(obs_shellpair_list, obs_shellpair_data) = compute_shellpairs(shells);
    size_t nsp = 0;
    for (auto& sp : obs_shellpair_list) {
      nsp += sp.second.size();
    }
    if(rank==0) std::cout << "# of {all,non-negligible} shell-pairs = {"
              << shells.size() * (shells.size() + 1) / 2 << "," << nsp << "}"
              << endl;
}


std::tuple<int,double> compute_NRE(const ExecutionContext& ec, std::vector<libint2::Atom>& atoms, const int focc){

  auto rank = ec.pg().rank(); 
      //  std::cout << "Geometries in bohr units " << std::endl;
    //  for (auto i = 0; i < atoms.size(); ++i)
    //    std::cout << atoms[i].atomic_number << "  " << atoms[i].x<< "  " <<
    //    atoms[i].y<< "  " << atoms[i].z << endl;
    // count the number of electrons
    auto nelectron = 0;
    for(size_t i = 0; i < atoms.size(); ++i)
        nelectron += atoms[i].atomic_number;
    const auto ndocc = nelectron / focc;

    // compute the nuclear repulsion energy
    double enuc = 0.0;
    for(size_t i = 0; i < atoms.size(); i++)
        for(size_t j = i + 1; j < atoms.size(); j++) {
            double xij = atoms[i].x - atoms[j].x;
            double yij = atoms[i].y - atoms[j].y;
            double zij = atoms[i].z - atoms[j].z;
            double r2  = xij * xij + yij * yij + zij * zij;
            double r   = sqrt(r2);
            enuc += atoms[i].atomic_number * atoms[j].atomic_number / r;
        }

    return std::make_tuple(ndocc,enuc);

}

std::tuple<std::vector<size_t>, std::vector<Tile>, std::vector<Tile>> 
    compute_AO_tiles(const ExecutionContext& ec, const SystemData& sys_data, libint2::BasisSet& shells){

      tamm::Tile tile_size = sys_data.options_map.scf_options.AO_tilesize; 

      auto rank = ec.pg().rank();

      auto N = nbasis(shells);
  
    //heuristic to set tilesize to atleast 5% of nbf
    if(tile_size < N*0.05 && !sys_data.options_map.scf_options.force_tilesize) {
      tile_size = std::ceil(N*0.05);
      final_AO_tilesize = tile_size;
      if(rank == 0) cout << "***** Reset tilesize to nbf*5% = " << tile_size << endl;
    }
    
    std::vector<Tile> AO_tiles;
    for(auto s : shells) AO_tiles.push_back(s.size());
    if(rank==0) 
      cout << "Number of AO tiles = " << AO_tiles.size() << endl;

    tamm::Tile est_ts = 0;
    std::vector<Tile> AO_opttiles;
    std::vector<size_t> shell_tile_map;
    for(auto s=0U;s<shells.size();s++){
      est_ts += shells[s].size();
      if(est_ts>=tile_size) {
        AO_opttiles.push_back(est_ts);
        shell_tile_map.push_back(s); //shell id specifying tile boundary
        est_ts=0;
      }
    }
    if(est_ts>0){
      AO_opttiles.push_back(est_ts);
      shell_tile_map.push_back(shells.size()-1);
    }

    // std::vector<int> vtc(AO_tiles.size());
    // std::iota (std::begin(vtc), std::end(vtc), 0);
    // cout << "AO tile indexes = " << vtc;
    // cout << "orig AO tiles = " << AO_tiles;
    
    // cout << "print new opt AO tiles = " << AO_opttiles;
    // cout << "print shell-tile map = " << shell_tile_map;

    return std::make_tuple(shell_tile_map,AO_tiles,AO_opttiles);
}


Matrix compute_orthogonalizer(ExecutionContext& ec, SystemData& sys_data, TAMMTensors& ttensors) {
    
    auto hf_t1 = std::chrono::high_resolution_clock::now();
    auto rank = ec.pg().rank(); 

    // compute orthogonalizer X such that X.transpose() . S . X = I
    //TODO: Xinv not used
    Matrix X, Xinv;
    double XtX_condition_number;  // condition number of "re-conditioned"
                                  // overlap obtained as Xinv.transpose() . Xinv
    // one should think of columns of Xinv as the conditioned basis
    // Re: name ... cond # (Xinv.transpose() . Xinv) = cond # (X.transpose() . X)
    // by default assume can manage to compute with condition number of S <= 1/eps
    // this is probably too optimistic, but in well-behaved cases even 10^11 is OK
    std::tie(X, Xinv, XtX_condition_number) =
        conditioning_orthogonalizer(ec, sys_data, ttensors.S1);

    // TODO Redeclare TAMM S1 with new dims?
    auto hf_t2   = std::chrono::high_resolution_clock::now();
    auto hf_time = std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

    if(rank == 0) std::cout << "Time for computing orthogonalizer: " << hf_time << " secs" << endl << endl;

    return X;

}


template<typename TensorType> 
void compute_hamiltonian(
ExecutionContext& ec, std::vector<libint2::Atom>& atoms, libint2::BasisSet& shells,
      std::vector<size_t>& shell_tile_map, std::vector<Tile>& AO_tiles, TAMMTensors& ttensors, EigenTensors& etensors){

    using libint2::Operator;
    // const size_t N = nbasis(shells);
    auto rank = ec.pg().rank();

    ttensors.H1 = {tAO, tAO};
    ttensors.S1 = {tAO, tAO};
    ttensors.T1 = {tAO, tAO};
    ttensors.V1 = {tAO, tAO};
    Tensor<TensorType>::allocate(&ec, ttensors.H1, ttensors.S1, ttensors.T1, ttensors.V1);

    auto [mu, nu] = tAO.labels<2>("all");

    auto hf_t1 = std::chrono::high_resolution_clock::now();

    compute_1body_ints(ec,ttensors.S1,atoms,shells,Operator::overlap,shell_tile_map,AO_tiles);
    compute_1body_ints(ec,ttensors.T1,atoms,shells,Operator::kinetic,shell_tile_map,AO_tiles);
    compute_1body_ints(ec,ttensors.V1,atoms,shells,Operator::nuclear,shell_tile_map,AO_tiles);
    auto hf_t2   = std::chrono::high_resolution_clock::now();
    auto hf_time = std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
    if(rank == 0) std::cout << std::endl << "Time for computing 1-e integrals T, V, S: " << hf_time << " secs" << endl;

    // Core Hamiltonian = T + V
    Scheduler{ec}
      (ttensors.H1(mu, nu)  =  ttensors.T1(mu, nu))
      (ttensors.H1(mu, nu) +=  ttensors.V1(mu, nu)).execute();

    // tamm::scale_ip(ttensors.H1(),2.0);
}

void scf_restart_test(const ExecutionContext& ec, const SystemData& sys_data, const std::string& filename, 
                      bool restart, std::string files_prefix) {
    if(!restart) return;
    const auto rank    = ec.pg().rank();
    const bool is_uhf  = (sys_data.scf_type == sys_data.SCFType::uhf);

    int        rstatus = 1;

    std::string movecsfile_alpha  = files_prefix + ".alpha.movecs";
    std::string densityfile_alpha = files_prefix + ".alpha.density";
    std::string movecsfile_beta  = files_prefix + ".beta.movecs";       
    std::string densityfile_beta = files_prefix + ".beta.density";   
    bool status = false;

    if(rank==0) {
      status = fs::exists(movecsfile_alpha) && fs::exists(densityfile_alpha);
      if(is_uhf) 
        status = status && fs::exists(movecsfile_beta) && fs::exists(densityfile_beta);
    }
    rstatus = status;
    ec.pg().barrier();
    MPI_Bcast(&rstatus        ,1,mpi_type<int>()       ,0,ec.pg().comm());
    std::string fnf = movecsfile_alpha + "; " + densityfile_alpha;
    if(is_uhf) fnf = fnf + "; " + movecsfile_beta + "; " + densityfile_beta;    
    if(rstatus == 0) tamm_terminate("Error reading one or all of the files: [" + fnf + "]");
}

void scf_restart(const ExecutionContext& ec, const SystemData& sys_data, const std::string& filename, 
                EigenTensors& etensors, std::string files_prefix) {

    const auto rank    = ec.pg().rank();
    const auto N       = sys_data.nbf_orig;
    const auto Northo  = N - sys_data.n_lindep;
    const bool is_uhf  = (sys_data.scf_type == sys_data.SCFType::uhf);

    EXPECTS(Northo == sys_data.nbf);

    std::string movecsfile_alpha  = files_prefix + ".alpha.movecs";
    std::string densityfile_alpha = files_prefix + ".alpha.density";

    if(rank==0) {
      cout << "Reading movecs and density files ... ";
      etensors.C = read_scf_mat<TensorType>(movecsfile_alpha);
      etensors.D = read_scf_mat<TensorType>(densityfile_alpha);

      if(is_uhf) {
        std::string movecsfile_beta  = files_prefix + ".beta.movecs";       
        std::string densityfile_beta = files_prefix + ".beta.density"; 
        etensors.C_beta = read_scf_mat<TensorType>(movecsfile_beta);
        etensors.D_beta = read_scf_mat<TensorType>(densityfile_beta);     
      }
      cout << "done" << endl;
    }
    ec.pg().barrier();

    TensorType *Dbufp_a = etensors.D.data();    
    MPI_Bcast(Dbufp_a,N*N,mpi_type<TensorType>(),0,ec.pg().comm());

    if(is_uhf) {
      TensorType *Dbufp_b = etensors.D_beta.data();      
      MPI_Bcast(Dbufp_b,N*N,mpi_type<TensorType>(),0,ec.pg().comm());
    }    
    ec.pg().barrier();
}

template<typename TensorType>
double tt_trace(ExecutionContext& ec, Tensor<TensorType>& T1, Tensor<TensorType>& T2){
    Tensor<TensorType> tensor = {tAO, tAO};
    Tensor<TensorType>::allocate(&ec,tensor);
    Scheduler{ec} (tensor(mu,nu) = T1(mu,ku) * T2(ku,nu)).execute();
    double trace = tamm::trace(tensor); 
    Tensor<TensorType>::deallocate(tensor);
    return trace;
} 

void print_energies(ExecutionContext& ec, TAMMTensors& ttensors, const SystemData& sys_data, bool debug=false){

      const bool is_uhf = (sys_data.scf_type == sys_data.SCFType::uhf);
      const bool is_rhf = (sys_data.scf_type == sys_data.SCFType::rhf);
      
      double nelectrons = 0.0;
      double kinetic_1e = 0.0;
      double NE_1e      = 0.0;
      double energy_1e  = 0.0;
      double energy_2e  = 0.0;
      if(is_rhf) {
        nelectrons  =       tt_trace(ec,ttensors.D_tamm,ttensors.S1);
        kinetic_1e  =       tt_trace(ec,ttensors.D_tamm,ttensors.T1);
        NE_1e       =       tt_trace(ec,ttensors.D_tamm,ttensors.V1);
        energy_1e   =       tt_trace(ec,ttensors.D_tamm,ttensors.H1);
        energy_2e   = 0.5 * tt_trace(ec,ttensors.D_tamm,ttensors.F1tmp1);
      }
      if(is_uhf) {
        nelectrons  =       tt_trace(ec,ttensors.D_tamm,ttensors.S1);
        kinetic_1e  =       tt_trace(ec,ttensors.D_tamm,ttensors.T1);
        NE_1e       =       tt_trace(ec,ttensors.D_tamm,ttensors.V1);
        energy_1e   =       tt_trace(ec,ttensors.D_tamm,ttensors.H1);
        energy_2e   = 0.5 * tt_trace(ec,ttensors.D_tamm,ttensors.F1tmp1);
        nelectrons +=       tt_trace(ec,ttensors.D_beta_tamm,ttensors.S1);
        kinetic_1e +=       tt_trace(ec,ttensors.D_beta_tamm,ttensors.T1);
        NE_1e      +=       tt_trace(ec,ttensors.D_beta_tamm,ttensors.V1);
        energy_1e  +=       tt_trace(ec,ttensors.D_beta_tamm,ttensors.H1);
        energy_2e  += 0.5 * tt_trace(ec,ttensors.D_beta_tamm,ttensors.F1tmp1_beta);
      }

      if(ec.pg().rank() == 0){
        std::cout << "#electrons        = " << nelectrons << endl;
        std::cout << "1e energy kinetic = "<< std::setprecision(16) << kinetic_1e << endl;
        std::cout << "1e energy N-e     = " << NE_1e << endl;
        std::cout << "1e energy         = " << energy_1e << endl;
        std::cout << "2e energy         = " << energy_2e << std::endl;
      }
}

template<typename TensorType>
std::tuple<TensorType,TensorType> scf_iter_body(ExecutionContext& ec, 
#ifdef USE_SCALAPACK
      blacspp::Grid* blacs_grid,
      scalapackpp::BlockCyclicDist2D* blockcyclic_dist,
#endif
      const int& iter, const SystemData& sys_data,
      TAMMTensors& ttensors, EigenTensors& etensors, bool ediis, bool scf_restart=false){

      const bool is_uhf = (sys_data.scf_type == sys_data.SCFType::uhf);
      const bool is_rhf = (sys_data.scf_type == sys_data.SCFType::rhf);

      Tensor<TensorType>& H1                = ttensors.H1;
      Tensor<TensorType>& S1                = ttensors.S1;
      Tensor<TensorType>& D_diff            = ttensors.D_diff;
      Tensor<TensorType>& ehf_tmp           = ttensors.ehf_tmp;
      Tensor<TensorType>& ehf_tamm          = ttensors.ehf_tamm; 

      Matrix& X_a     = etensors.X;
      Matrix& F_alpha = etensors.F;
      Matrix& C_alpha = etensors.C; 
      Matrix& D_alpha = etensors.D;
      Matrix& C_occ   = etensors.C_occ; 
      Tensor<TensorType>& F1_alpha          = ttensors.F1;
      Tensor<TensorType>& F1tmp1_alpha      = ttensors.F1tmp1;
      Tensor<TensorType>& FD_alpha_tamm     = ttensors.FD_tamm;
      Tensor<TensorType>& FDS_alpha_tamm    = ttensors.FDS_tamm;
      Tensor<TensorType>& D_alpha_tamm      = ttensors.D_tamm;
      Tensor<TensorType>& D_last_alpha_tamm = ttensors.D_last_tamm;
      
      Matrix& X_b     = etensors.X_beta;
      Matrix& F_beta  = etensors.F_beta;
      Matrix& C_beta  = etensors.C_beta; 
      Matrix& D_beta  = etensors.D_beta;      
      Tensor<TensorType>& F1_beta           = ttensors.F1_beta;
      Tensor<TensorType>& F1tmp1_beta       = ttensors.F1tmp1_beta;
      Tensor<TensorType>& FD_beta_tamm      = ttensors.FD_beta_tamm;
      Tensor<TensorType>& FDS_beta_tamm     = ttensors.FDS_beta_tamm; 
      Tensor<TensorType>& D_beta_tamm       = ttensors.D_beta_tamm;
      Tensor<TensorType>& D_last_beta_tamm  = ttensors.D_last_beta_tamm;
      
      Scheduler sch{ec};

      const int64_t N = sys_data.nbf_orig;

      auto rank  = ec.pg().rank(); 
      auto debug = sys_data.options_map.scf_options.debug;
      auto [mu, nu, ku]  = tAO.labels<3>("all"); //TODO
      const int max_hist = sys_data.options_map.scf_options.diis_hist; 

      sch
        (F1_alpha()  = H1())
        (F1_alpha() += F1tmp1_alpha())
        .execute();
      
      if(is_uhf) {
        sch
          (F1_beta()   = H1())
          (F1_beta()  += F1tmp1_beta())
          .execute();
      }

      Tensor<TensorType> err_mat_alpha_tamm{tAO, tAO};
      Tensor<TensorType> err_mat_beta_tamm{tAO, tAO};
      Tensor<TensorType>::allocate(&ec, err_mat_alpha_tamm);
      if(is_uhf) Tensor<TensorType>::allocate(&ec, err_mat_beta_tamm);

      sch
        (FD_alpha_tamm(mu,nu)       = F1_alpha(mu,ku)      * D_last_alpha_tamm(ku,nu))
        (FDS_alpha_tamm(mu,nu)      = FD_alpha_tamm(mu,ku) * S1(ku,nu))
        (err_mat_alpha_tamm(mu,nu)  = FDS_alpha_tamm(mu,nu))
        (err_mat_alpha_tamm(mu,nu) -= FDS_alpha_tamm(nu,mu))
        .execute();
      
      if(is_uhf) {
        sch
          (FD_beta_tamm(mu,nu)        = F1_beta(mu,ku)       * D_last_beta_tamm(ku,nu))
          (FDS_beta_tamm(mu,nu)       = FD_beta_tamm(mu,ku)  * S1(ku,nu))    
          (err_mat_beta_tamm(mu,nu)   = FDS_beta_tamm(mu,nu))
          (err_mat_beta_tamm(mu,nu)  -= FDS_beta_tamm(nu,mu))
          .execute();
      }

      if(iter >= 1 && !ediis) {
        if(is_rhf) {
          ++idiis;
          diis(ec, tAO, D_alpha_tamm, F1_alpha, err_mat_alpha_tamm, iter, max_hist, idiis, sys_data.n_lindep,
            ttensors.diis_hist, ttensors.fock_hist);
        }
        if(is_uhf) {
          Tensor<TensorType> F1_T{tAO, tAO};
          Tensor<TensorType> F1_S{tAO, tAO};
          Tensor<TensorType> D_T{tAO, tAO};
          Tensor<TensorType> D_S{tAO, tAO};
          sch
            .allocate(F1_T,F1_S,D_T,D_S)
            (F1_T()  = F1_alpha())
            (F1_T() += F1_beta())
            (F1_S()  = F1_alpha())
            (F1_S() -= F1_beta())
            (D_T()   = D_alpha_tamm())
            (D_T()  += D_beta_tamm())
            (D_S()   = D_alpha_tamm())
            (D_S()  -= D_beta_tamm())
            .execute();
          diis(ec, tAO, D_T, F1_T, err_mat_alpha_tamm, iter, max_hist, idiis, sys_data.n_lindep,
            ttensors.diis_hist, ttensors.fock_hist);
          diis(ec, tAO, D_S, F1_S,  err_mat_beta_tamm,  iter, max_hist, idiis, sys_data.n_lindep,
            ttensors.diis_beta_hist, ttensors.fock_beta_hist);
          sch
            (F1_alpha()  = 0.5 * F1_T())
            (F1_alpha() += 0.5 * F1_S())
            (F1_beta()   = 0.5 * F1_T())
            (F1_beta()  -= 0.5 * F1_S())
            (D_alpha_tamm()  = 0.5 * D_T())
            (D_alpha_tamm() += 0.5 * D_S())
            (D_beta_tamm()   = 0.5 * D_T())
            (D_beta_tamm()  -= 0.5 * D_S())
            .deallocate(F1_T,F1_S,D_T,D_S)
            .execute();
        }
      }
      
      if( rank == 0 ) {
        tamm_to_eigen_tensor(F1_alpha,F_alpha);
        if(is_uhf) {
          tamm_to_eigen_tensor(F1_beta, F_beta);
        }
      }
      
      auto do_t1 = std::chrono::high_resolution_clock::now();

      if(!scf_restart){
        // solve F C = e S C by (conditioned) transformation to F' C' = e C',
        // where
        // F' = X.transpose() . F . X; the original C is obtained as C = X . C'
        #ifdef USE_SCALAPACK
          const auto& grid = *blacs_grid;
          const auto  mb   = blockcyclic_dist->mb();
          const auto Northo = sys_data.nbf;
          if( grid.ipr() >= 0 and grid.ipc() >= 0 ) {
            // std::cout << "IN SCALAPACK " << rank << std::endl; 
            // TODO: Optimize intermediates here
            scalapackpp::BlockCyclicMatrix<double> 
              Fa_sca  ( grid, N,      N,      mb, mb ),
              Xa_sca  ( grid, Northo, N,      mb, mb ), // Xa is row-major
              Fp_sca  ( grid, Northo, Northo, mb, mb ),
              Ca_sca  ( grid, Northo, Northo, mb, mb ),
              TMP1_sca( grid, N,      Northo, mb, mb ),
              TMP2_sca( grid, Northo, N,      mb, mb );

            // Scatter Fock / X alpha from root
            Fa_sca.scatter_to( N,      N, F_alpha.data(), N,      0, 0 ); 
            Xa_sca.scatter_to( Northo, N, X_a.data(),     Northo, 0, 0 );

            // Compute TMP = F * X -> F * X**T (b/c row-major)
            scalapackpp::pgemm( scalapackpp::Op::NoTrans, scalapackpp::Op::Trans,
                                1., Fa_sca, Xa_sca, 0., TMP1_sca );

            // Compute Fp = X**T * TMP -> X * TMP (b/c row-major)
            scalapackpp::pgemm( scalapackpp::Op::NoTrans, scalapackpp::Op::NoTrans,
                                1., Xa_sca, TMP1_sca, 0., Fp_sca );

            // Solve EVP
            std::vector<double> eps_a( Northo );
            scalapackpp::hereigd( scalapackpp::Job::Vec, scalapackpp::Uplo::Lower,
                                  Fp_sca, eps_a.data(), Ca_sca );

            // Backtransform TMP = X * Ca -> TMP**T = Ca**T * X
            scalapackpp::pgemm( scalapackpp::Op::Trans, scalapackpp::Op::NoTrans,
                                1., Ca_sca, Xa_sca, 0., TMP2_sca );

            // Gather results
            if( rank == 0 ) C_alpha.resize( N, Northo );
            TMP2_sca.gather_from( Northo, N, C_alpha.data(), Northo, 0, 0 );

            if(is_uhf) {

              // Scatter Fock / X beta from root
              Fa_sca.scatter_to( N,      N, F_beta.data(), N,      0, 0 ); 
              Xa_sca.scatter_to( Northo, N, X_b.data(),    Northo, 0, 0 );

              // Compute TMP = F * X -> F * X**T (b/c row-major)
              scalapackpp::pgemm( scalapackpp::Op::NoTrans, scalapackpp::Op::Trans,
                                  1., Fa_sca, Xa_sca, 0., TMP1_sca );

              // Compute Fp = X**T * TMP -> X * TMP (b/c row-major)
              scalapackpp::pgemm( scalapackpp::Op::NoTrans, scalapackpp::Op::NoTrans,
                                  1., Xa_sca, TMP1_sca, 0., Fp_sca );

              // Solve EVP
              std::vector<double> eps_a( Northo );
              scalapackpp::hereigd( scalapackpp::Job::Vec, scalapackpp::Uplo::Lower,
                                    Fp_sca, eps_a.data(), Ca_sca );

              // Backtransform TMP = X * Cb -> TMP**T = Cb**T * X
              scalapackpp::pgemm( scalapackpp::Op::Trans, scalapackpp::Op::NoTrans,
                                  1., Ca_sca, Xa_sca, 0., TMP2_sca );

              // Gather results
              if( rank == 0 ) C_beta.resize( N, Northo );
              TMP2_sca.gather_from( Northo, N, C_beta.data(), Northo, 0, 0 );
            
            }
          } // rank participates in ScaLAPACK call

        #elif defined(EIGEN_DIAG)

          if(is_rhf) {
            Eigen::SelfAdjointEigenSolver<Matrix> eig_solver_alpha(X_a.transpose() * F_alpha * X_a);
            C_alpha = X_a * eig_solver_alpha.eigenvectors();
          }
          if(is_uhf) {
            Eigen::SelfAdjointEigenSolver<Matrix> eig_solver_alpha(X_a.transpose() * F_alpha * X_a);
            C_alpha = X_a * eig_solver_alpha.eigenvectors();
            Eigen::SelfAdjointEigenSolver<Matrix> eig_solver_beta( X_b.transpose() * F_beta  * X_b);
            C_beta  = X_b * eig_solver_beta.eigenvectors();      
          }

        #else
      
          if(is_rhf) {
            const int64_t Northo_a = sys_data.nbf; //X_a.cols();
            if( rank == 0 ) {
              Matrix Fp = F_alpha; // XXX: Can F be destroyed?
              C_alpha.resize(N,Northo_a);
              linalg::blas::gemm( 'N', 'T', N, Northo_a, N,
                                  1., Fp.data(), N, X_a.data(), Northo_a, 
                                  0., C_alpha.data(), N );
              linalg::blas::gemm( 'N', 'N', Northo_a, Northo_a, N,
                                  1., X_a.data(), Northo_a, C_alpha.data(), N, 
                                  0., Fp.data(), Northo_a );
              std::vector<double> eps_a(Northo_a);
              linalg::lapack::syevd( 'V', 'L', Northo_a, Fp.data(), Northo_a, eps_a.data() );
              linalg::blas::gemm( 'T', 'N', Northo_a, N, Northo_a, 
                                  1., Fp.data(), Northo_a, X_a.data(), Northo_a, 
                                  0., C_alpha.data(), Northo_a );
            } 
            // else C_alpha.resize( N, Northo_a );

            // if( ec.pg().size() > 1 )
            //   MPI_Bcast( C_alpha.data(), C_alpha.size(), MPI_DOUBLE, 0, ec.pg().comm() );
          }

          if(is_uhf) {
            const int64_t Northo_a = sys_data.nbf; //X_a.cols();
            const int64_t Northo_b = sys_data.nbf; //X_b.cols();
            if( rank == 0 ) {
              //alpha
              Matrix Fp = F_alpha; 
              C_alpha.resize(N,Northo_a);
              linalg::blas::gemm( 'N', 'T', N, Northo_a, N,
                                  1., Fp.data(), N, X_a.data(), Northo_a, 
                                  0., C_alpha.data(), N );
              linalg::blas::gemm( 'N', 'N', Northo_a, Northo_a, N,
                                  1., X_a.data(), Northo_a, C_alpha.data(), N, 
                                  0., Fp.data(), Northo_a );
              std::vector<double> eps_a(Northo_a);
              linalg::lapack::syevd( 'V', 'L', Northo_a, Fp.data(), Northo_a, eps_a.data() );
              linalg::blas::gemm( 'T', 'N', Northo_a, N, Northo_a, 
                                  1., Fp.data(), Northo_a, X_a.data(), Northo_a, 
                                  0., C_alpha.data(), Northo_a );
              //beta
              Fp = F_beta; 
              C_beta.resize(N,Northo_b);
              linalg::blas::gemm( 'N', 'T', N, Northo_b, N,
                                  1., Fp.data(), N, X_b.data(), Northo_b, 
                                  0., C_beta.data(), N );
              linalg::blas::gemm( 'N', 'N', Northo_b, Northo_b, N,
                                  1., X_b.data(), Northo_b, C_beta.data(), N, 
                                  0., Fp.data(), Northo_b );
              std::vector<double> eps_b(Northo_b);
              linalg::lapack::syevd( 'V', 'L', Northo_b, Fp.data(), Northo_b, eps_b.data() );
              linalg::blas::gemm( 'T', 'N', Northo_b, N, Northo_b, 
                                  1., Fp.data(), Northo_b, X_b.data(), Northo_b, 
                                  0., C_beta.data(), Northo_b );
            } 
            // else {
            //   C_alpha.resize(N, Northo_a);
            //   C_beta.resize(N, Northo_b);
            // }
          
            // if( ec.pg().size() > 1 ) {
            //   MPI_Bcast( C_alpha.data(), C_alpha.size(), MPI_DOUBLE, 0, ec.pg().comm() );
            //   MPI_Bcast( C_beta.data(),  C_beta.size(),  MPI_DOUBLE, 0, ec.pg().comm() );
            // }
          }

        #endif

        // compute density
        if( rank == 0 ) {
          if(is_rhf) {
            C_occ   = C_alpha.leftCols(sys_data.nelectrons_alpha);
            D_alpha = 2.0 * C_occ * C_occ.transpose();
            X_a     = C_alpha;
          }
          if(is_uhf) {
            C_occ   = C_alpha.leftCols(sys_data.nelectrons_alpha);
            D_alpha = C_occ * C_occ.transpose();
            X_a     = C_alpha;
            C_occ   = C_beta.leftCols(sys_data.nelectrons_beta);
            D_beta  = C_occ * C_occ.transpose();
            X_b     = C_beta;
          }
        }

        auto do_t2 = std::chrono::high_resolution_clock::now();
        auto do_time =
        std::chrono::duration_cast<std::chrono::duration<double>>((do_t2 - do_t1)).count();

        if(rank == 0 && debug) std::cout << "eigen_solve:" << do_time << "s, " << std::endl; 
      }//end scf_restart 

      if(rank == 0) {
        eigen_to_tamm_tensor(D_alpha_tamm,D_alpha);
        if(is_uhf) {
          eigen_to_tamm_tensor(D_beta_tamm, D_beta);
        }
      }

      MPI_Bcast(D_alpha.data(), D_alpha.size(), MPI_DOUBLE, 0, ec.pg().comm() );
      if(is_uhf) MPI_Bcast(D_beta.data(), D_beta.size(), MPI_DOUBLE, 0, ec.pg().comm() );

      double ehf = 0.0;

      if(is_rhf) {
        sch
          (ehf_tmp(mu,nu)  = H1(mu,nu))
          (ehf_tmp(mu,nu) += F1_alpha(mu,nu))
          (ehf_tamm()      = 0.5 * D_last_alpha_tamm() * ehf_tmp())
          .execute();
      }

      if(is_uhf) {
        sch
          (ehf_tmp(mu,nu)  = H1(mu,nu))
          (ehf_tmp(mu,nu) += F1_alpha(mu,nu))
          (ehf_tamm()      = 0.5 * D_last_alpha_tamm() * ehf_tmp())
          (ehf_tmp(mu,nu)  = H1(mu,nu))
          (ehf_tmp(mu,nu) += F1_beta(mu,nu))
          (ehf_tamm()     += 0.5 * D_last_beta_tamm()  * ehf_tmp())
          .execute();
      }
       
      ehf = get_scalar(ehf_tamm);
      
      // if(ediis) {
      //   compute_2bf<TensorType>(ec, sys_data, obs, do_schwarz_screen, shell2bf, SchwarzK,
      //                           max_nprim4,shells, ttensors, etensors, do_density_fitting);
      //   Tensor<TensorType>  Dcopy{tAO,tAO};
      //   Tensor<TensorType>  Fcopy{tAO, tAO};
      //   Tensor<TensorType>  ehf_tamm_copy{};
      //   Tensor<TensorType>::allocate(&ec,Dcopy,Fcopy,ehf_tamm_copy);
      //   sch
      //     (Dcopy() = D_alpha_tamm())
      //     (Fcopy() = F1tmp1_alpha())
      //     (ehf_tmp(mu,nu) = F1tmp1_alpha(mu,nu))
      //     (ehf_tamm_copy()     = 0.5 * D_alpha_tamm() * ehf_tmp())
      //     // (ehf_tamm_copy() = ehf_tamm())
      //     .execute();
      //   ttensors.D_hist.push_back(Dcopy);
      //   ttensors.fock_hist.push_back(Fcopy);
      //   ttensors.ehf_tamm_hist.push_back(ehf_tamm_copy);
      //   if(rank==0) cout << "iter: " << iter << "," << (int)ttensors.D_hist.size() << endl;
      //   energy_diis(ec, tAO, iter, max_hist, D_alpha_tamm,
      //               ttensors.D_hist, ttensors.fock_hist, ttensors.ehf_tamm_hist);
      // }

      double rmsd = 0.0;
      sch
        (D_diff()  = D_alpha_tamm())
        (D_diff() -= D_last_alpha_tamm())
        .execute();
      rmsd = norm(D_diff)/(double)(1.0*N);

      if(is_uhf) {
        sch
          (D_diff()  = D_beta_tamm())
          (D_diff() -= D_last_beta_tamm())
          .execute();
        rmsd += norm(D_diff)/(double)(1.0*N);        
      }
        
      double alpha = sys_data.options_map.scf_options.alpha;
      if(rmsd < 1e-6) {
        switch_diis = true;
      } //rmsd check

      // D = alpha*D + (1.0-alpha)*D_last;
      if(is_rhf) {
        tamm::scale_ip(D_alpha_tamm(),alpha);
        sch(D_alpha_tamm() += (1.0-alpha)*D_last_alpha_tamm()).execute();
        tamm_to_eigen_tensor(D_alpha_tamm,D_alpha);
      }
      if(is_uhf) {
        tamm::scale_ip(D_alpha_tamm(),alpha);
        sch(D_alpha_tamm() += (1.0-alpha) * D_last_alpha_tamm()).execute();
        tamm_to_eigen_tensor(D_alpha_tamm,D_alpha);
        tamm::scale_ip(D_beta_tamm(),alpha);
        sch(D_beta_tamm()  += (1.0-alpha) * D_last_beta_tamm()).execute();
        tamm_to_eigen_tensor(D_beta_tamm, D_beta);
      }

      return std::make_tuple(ehf,rmsd);       
}

template<typename TensorType>
void compute_2bf_simple(ExecutionContext& ec, const SystemData& sys_data, const libint2::BasisSet& obs,
      const bool do_schwarz_screen, const std::vector<size_t>& shell2bf,
      const Matrix& SchwarzK, 
      const size_t& max_nprim4, libint2::BasisSet& shells,
      TAMMTensors& ttensors, EigenTensors& etensors, const bool do_density_fitting=false){

      using libint2::Operator;

      const bool is_uhf = (sys_data.scf_type == sys_data.SCFType::uhf);
      const bool is_rhf = (sys_data.scf_type == sys_data.SCFType::rhf);

      Matrix& G_a = etensors.G;
      Matrix& D_a = etensors.D; 
      Tensor<TensorType>& F1tmp    = ttensors.F1tmp;
      Tensor<TensorType>& F1tmp1_a = ttensors.F1tmp1;
      
      Matrix& G_b = etensors.G_beta;
      Matrix& D_b = etensors.D_beta; 
      Tensor<TensorType>& F1tmp1_b = ttensors.F1tmp1_beta;
      
      double fock_precision = std::min(sys_data.options_map.scf_options.tol_int, 1e-3 * sys_data.options_map.scf_options.conve);
      auto   rank           = ec.pg().rank();
      auto   N              = sys_data.nbf_orig; 
      auto   debug          = sys_data.options_map.scf_options.debug;

      Matrix D_shblk_norm =  compute_shellblock_norm(obs, D_a);  // matrix of infty-norms of shell blocks
      
      double engine_precision = fock_precision;

      // construct the 2-electron repulsion integrals engine pool
      using libint2::Engine;
      Engine engine(Operator::coulomb, obs.max_nprim(), obs.max_l(), 0);

      engine.set_precision(engine_precision);
      const auto& buf = engine.results();

      auto comp_2bf_lambda = [&](IndexVector blockid) {

          auto s1 = blockid[0];
          auto bf1_first = shell2bf[s1]; 
          auto n1 = obs[s1].size();
          auto sp12_iter = obs_shellpair_data.at(s1).begin();

          auto s2 = blockid[1];
          auto s2spl = obs_shellpair_list[s1];
          auto s2_itr = std::find(s2spl.begin(),s2spl.end(),s2);
          if(s2_itr == s2spl.end()) return;
          auto s2_pos = std::distance(s2spl.begin(),s2_itr);
          auto bf2_first = shell2bf[s2];
          auto n2 = obs[s2].size();

          std::advance(sp12_iter,s2_pos);
          const auto* sp12 = sp12_iter->get();
        
          const auto Dnorm12 = do_schwarz_screen ? D_shblk_norm(s1, s2) : 0.;

          for (decltype(s1) s3 = 0; s3 <= s1; ++s3) {
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

              const auto Dnorm1234 =
                  do_schwarz_screen
                      ? std::max(D_shblk_norm(s1, s4),
                        std::max(D_shblk_norm(s2, s4),
                        std::max(D_shblk_norm(s3, s4), Dnorm123)))
                      : 0.;

              if (do_schwarz_screen &&
                  Dnorm1234 * SchwarzK(s1, s2) * SchwarzK(s3, s4) < fock_precision)
                continue;

              auto bf4_first = shell2bf[s4];
              auto n4 = obs[s4].size();

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
              //    F(a,b) += 1/2 * (ab|cd) * D(c,d)
              //    F(c,d) += 1/2 * (ab|cd) * D(a,b)
              //    F(b,d) -= 1/8 * (ab|cd) * D(a,c)
              //    F(b,c) -= 1/8 * (ab|cd) * D(a,d)
              //    F(a,c) -= 1/8 * (ab|cd) * D(b,d)
              //    F(a,d) -= 1/8 * (ab|cd) * D(b,c)
              // 2) each permutationally-unique integral (shell set) must be
              // scaled by its degeneracy,
              //    i.e. the number of the integrals/sets equivalent to it
              // 3) the end result must be symmetrized
              for (decltype(n1) f1 = 0, f1234 = 0; f1 != n1; ++f1) {
                const auto bf1 = f1 + bf1_first;
                for (decltype(n2) f2 = 0; f2 != n2; ++f2) {
                  const auto bf2 = f2 + bf2_first;
                  for (decltype(n3) f3 = 0; f3 != n3; ++f3) {
                    const auto bf3 = f3 + bf3_first;
                    for (decltype(n4) f4 = 0; f4 != n4; ++f4, ++f1234) {
                      const auto bf4 = f4 + bf4_first;
                      const auto value = buf_1234[f1234];
                      const auto value_scal_by_deg = value * s1234_deg;
                      if(is_uhf) {
                        //alpha_part
                        G_a(bf1, bf2) += 0.5   * D_a(bf3, bf4) * value_scal_by_deg;
                        G_a(bf3, bf4) += 0.5   * D_a(bf1, bf2) * value_scal_by_deg;
                        G_a(bf1, bf2) += 0.5   * D_b(bf3, bf4) * value_scal_by_deg;
                        G_a(bf3, bf4) += 0.5   * D_b(bf1, bf2) * value_scal_by_deg;
                        G_a(bf1, bf3) -= 0.25  * D_a(bf2, bf4) * value_scal_by_deg;
                        G_a(bf2, bf4) -= 0.25  * D_a(bf1, bf3) * value_scal_by_deg;
                        G_a(bf1, bf4) -= 0.25  * D_a(bf2, bf3) * value_scal_by_deg;
                        G_a(bf2, bf3) -= 0.25  * D_a(bf1, bf4) * value_scal_by_deg;
                        //beta_part
                        G_b(bf1, bf2) += 0.5   * D_b(bf3, bf4) * value_scal_by_deg;
                        G_b(bf3, bf4) += 0.5   * D_b(bf1, bf2) * value_scal_by_deg;
                        G_b(bf1, bf2) += 0.5   * D_a(bf3, bf4) * value_scal_by_deg;
                        G_b(bf3, bf4) += 0.5   * D_a(bf1, bf2) * value_scal_by_deg;
                        G_b(bf1, bf3) -= 0.25  * D_b(bf2, bf4) * value_scal_by_deg;
                        G_b(bf2, bf4) -= 0.25  * D_b(bf1, bf3) * value_scal_by_deg;
                        G_b(bf1, bf4) -= 0.25  * D_b(bf2, bf3) * value_scal_by_deg;
                        G_b(bf2, bf3) -= 0.25  * D_b(bf1, bf4) * value_scal_by_deg;
                      }
                      if(is_rhf) {
                        G_a(bf1, bf2) += 0.5   * D_a(bf3, bf4) * value_scal_by_deg;
                        G_a(bf3, bf4) += 0.5   * D_a(bf1, bf2) * value_scal_by_deg;
                        G_a(bf1, bf3) -= 0.125 * D_a(bf2, bf4) * value_scal_by_deg;
                        G_a(bf2, bf4) -= 0.125 * D_a(bf1, bf3) * value_scal_by_deg;
                        G_a(bf1, bf4) -= 0.125 * D_a(bf2, bf3) * value_scal_by_deg;
                        G_a(bf2, bf3) -= 0.125 * D_a(bf1, bf4) * value_scal_by_deg;
                      }
                    }
                  }
                }
              }
            }
          }
      };
     
      G_a.setZero(N,N);
      if(is_uhf) G_b.setZero(N,N);
      block_for(ec, F1tmp(), comp_2bf_lambda);
      //symmetrize G
      Matrix Gt = 0.5*(G_a + G_a.transpose());
      G_a = Gt;
      if(is_uhf) {
        Gt = 0.5*(G_b + G_b.transpose());
        G_b = Gt;
      }
      Gt.resize(0,0);      
      eigen_to_tamm_tensor_acc(F1tmp1_a,G_a);
      if(is_uhf) eigen_to_tamm_tensor_acc(F1tmp1_b,G_b);
      
      ec.pg().barrier();

      // auto F1tmp1_nrm = norm(F1tmp1);
      // if(rank==0) cout << std::setprecision(18) << "in compute_2bf, norm of F1tmp1: " << F1tmp1_nrm << endl;
      
}

template<typename TensorType>
void compute_2bf(ExecutionContext& ec, const SystemData& sys_data, const libint2::BasisSet& obs,
      const bool do_schwarz_screen, const std::vector<size_t>& shell2bf,
      const Matrix& SchwarzK, 
      const size_t& max_nprim4, libint2::BasisSet& shells,
      TAMMTensors& ttensors, EigenTensors& etensors, const bool do_density_fitting=false){

      using libint2::Operator;

      const bool is_uhf = (sys_data.scf_type == sys_data.SCFType::uhf);
      const bool is_rhf = (sys_data.scf_type == sys_data.SCFType::rhf);

      Matrix& G      = etensors.G;
      Matrix& D      = etensors.D; 
      Matrix& G_beta = etensors.G_beta;
      Matrix& D_beta = etensors.D_beta;

      Tensor<TensorType>& F1tmp       = ttensors.F1tmp;
      Tensor<TensorType>& F1tmp1      = ttensors.F1tmp1;
      Tensor<TensorType>& F1tmp1_beta = ttensors.F1tmp1_beta;
      Tensor<TensorType>& Zxy_tamm    = ttensors.Zxy_tamm;
      Tensor<TensorType>& xyK_tamm    = ttensors.xyK_tamm;

      double fock_precision = std::min(sys_data.options_map.scf_options.tol_int, 1e-3 * sys_data.options_map.scf_options.conve);
      auto   rank           = ec.pg().rank();
      auto   N              = sys_data.nbf_orig; 
      auto   debug          = sys_data.options_map.scf_options.debug;

      auto do_t1 = std::chrono::high_resolution_clock::now();
      Matrix D_shblk_norm =  compute_shellblock_norm(obs, D);  // matrix of infty-norms of shell blocks
      
      //TODO: Revisit
      double engine_precision = fock_precision;
      
      if(rank == 0)
        assert(engine_precision > max_engine_precision &&
          "using precomputed shell pair data limits the max engine precision"
          " ... make max_engine_precision smaller and recompile");

      // construct the 2-electron repulsion integrals engine pool
      using libint2::Engine;
      Engine engine(Operator::coulomb, obs.max_nprim(), obs.max_l(), 0);

      engine.set_precision(engine_precision);
      const auto& buf = engine.results();

      #if 1
        auto comp_2bf_lambda = [&](IndexVector blockid) {

          auto s1 = blockid[0];
          auto bf1_first = shell2bf[s1]; 
          auto n1 = obs[s1].size();
          auto sp12_iter = obs_shellpair_data.at(s1).begin();

          auto s2 = blockid[1];
          auto s2spl = obs_shellpair_list[s1];
          auto s2_itr = std::find(s2spl.begin(),s2spl.end(),s2);
          if(s2_itr == s2spl.end()) return;
          auto s2_pos = std::distance(s2spl.begin(),s2_itr);
          auto bf2_first = shell2bf[s2];
          auto n2 = obs[s2].size();

          std::advance(sp12_iter,s2_pos);
          const auto* sp12 = sp12_iter->get();
        
          const auto Dnorm12 = do_schwarz_screen ? D_shblk_norm(s1, s2) : 0.;

          for (decltype(s1) s3 = 0; s3 <= s1; ++s3) {
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

              const auto Dnorm1234 =
                  do_schwarz_screen
                      ? std::max(D_shblk_norm(s1, s4),
                        std::max(D_shblk_norm(s2, s4),
                        std::max(D_shblk_norm(s3, s4), Dnorm123)))
                      : 0.;

              if (do_schwarz_screen &&
                  Dnorm1234 * SchwarzK(s1, s2) * SchwarzK(s3, s4) < fock_precision)
                continue;

              auto bf4_first = shell2bf[s4];
              auto n4 = obs[s4].size();

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
              //    F(a,b) += 1/2 * (ab|cd) * D(c,d)
              //    F(c,d) += 1/2 * (ab|cd) * D(a,b)
              //    F(b,d) -= 1/8 * (ab|cd) * D(a,c)
              //    F(b,c) -= 1/8 * (ab|cd) * D(a,d)
              //    F(a,c) -= 1/8 * (ab|cd) * D(b,d)
              //    F(a,d) -= 1/8 * (ab|cd) * D(b,c)
              // 2) each permutationally-unique integral (shell set) must be
              // scaled by its degeneracy,
              //    i.e. the number of the integrals/sets equivalent to it
              // 3) the end result must be symmetrized
              for (decltype(n1) f1 = 0, f1234 = 0; f1 != n1; ++f1) {
                const auto bf1 = f1 + bf1_first;
                for (decltype(n2) f2 = 0; f2 != n2; ++f2) {
                  const auto bf2 = f2 + bf2_first;
                  for (decltype(n3) f3 = 0; f3 != n3; ++f3) {
                    const auto bf3 = f3 + bf3_first;
                    for (decltype(n4) f4 = 0; f4 != n4; ++f4, ++f1234) {
                      const auto bf4 = f4 + bf4_first;
  
                      const auto value = buf_1234[f1234];
                      const auto value_scal_by_deg = value * s1234_deg;

                      if(is_uhf) {
                        //alpha_part
                        G(bf1, bf2)      += 0.5   * D(bf3, bf4) * value_scal_by_deg;
                        G(bf3, bf4)      += 0.5   * D(bf1, bf2) * value_scal_by_deg;
                        G(bf1, bf2)      += 0.5   * D_beta(bf3, bf4) * value_scal_by_deg;
                        G(bf3, bf4)      += 0.5   * D_beta(bf1, bf2) * value_scal_by_deg;
                        G(bf1, bf3)      -= 0.25  * D(bf2, bf4) * value_scal_by_deg;
                        G(bf2, bf4)      -= 0.25  * D(bf1, bf3) * value_scal_by_deg;
                        G(bf1, bf4)      -= 0.25  * D(bf2, bf3) * value_scal_by_deg;
                        G(bf2, bf3)      -= 0.25  * D(bf1, bf4) * value_scal_by_deg;
                        //beta_part
                        G_beta(bf1, bf2) += 0.5   * D_beta(bf3, bf4) * value_scal_by_deg;
                        G_beta(bf3, bf4) += 0.5   * D_beta(bf1, bf2) * value_scal_by_deg;
                        G_beta(bf1, bf2) += 0.5   * D(bf3, bf4) * value_scal_by_deg;
                        G_beta(bf3, bf4) += 0.5   * D(bf1, bf2) * value_scal_by_deg;
                        G_beta(bf1, bf3) -= 0.25  * D_beta(bf2, bf4) * value_scal_by_deg;
                        G_beta(bf2, bf4) -= 0.25  * D_beta(bf1, bf3) * value_scal_by_deg;
                        G_beta(bf1, bf4) -= 0.25  * D_beta(bf2, bf3) * value_scal_by_deg;
                        G_beta(bf2, bf3) -= 0.25  * D_beta(bf1, bf4) * value_scal_by_deg;
                      }
                      if(is_rhf) {
                        G(bf1, bf2)      += 0.5   * D(bf3, bf4) * value_scal_by_deg;
                        G(bf3, bf4)      += 0.5   * D(bf1, bf2) * value_scal_by_deg;
                        G(bf1, bf3)      -= 0.125 * D(bf2, bf4) * value_scal_by_deg;
                        G(bf2, bf4)      -= 0.125 * D(bf1, bf3) * value_scal_by_deg;
                        G(bf1, bf4)      -= 0.125 * D(bf2, bf3) * value_scal_by_deg;
                        G(bf2, bf3)      -= 0.125 * D(bf1, bf4) * value_scal_by_deg;
                      }
                    }
                  }
                }
              }
            }
          }
        };
      #endif

    decltype(do_t1) do_t2;
    double do_time;

      if(!do_density_fitting){
        G.setZero(N,N);
        if(is_uhf) G_beta.setZero(N,N);
        block_for(ec, F1tmp(), comp_2bf_lambda);
        //symmetrize G
        Matrix Gt = 0.5*(G + G.transpose());
        G = Gt;
        if(is_uhf) {
          Gt = 0.5*(G_beta + G_beta.transpose());
          G_beta = Gt;
        }
        Gt.resize(0,0);

        do_t2 = std::chrono::high_resolution_clock::now();
        do_time =
        std::chrono::duration_cast<std::chrono::duration<double>>((do_t2 - do_t1)).count();

        if(rank == 0 && debug) std::cout << "2BF:" << do_time << "s, ";
        
        eigen_to_tamm_tensor_acc(F1tmp1,G);
        if(is_uhf) eigen_to_tamm_tensor_acc(F1tmp1_beta,G_beta);

        // ec.pg().barrier();

      }
      else {
        #if 1
          // const auto n = obs.nbf();
          // const auto ndf = dfbs.nbf();

          using libint2::Operator;
          using libint2::BraKet;
          using libint2::Engine;

          // using first time? compute 3-center ints and transform to inv sqrt
          // representation
          if (!is_3c_init) {
            is_3c_init = true;
          // const auto nshells = obs.size();
          // const auto nshells_df = dfbs.size();  
            const auto& unitshell = libint2::Shell::unit();
          
            auto engine = libint2::Engine(libint2::Operator::coulomb,
                                        std::max(obs.max_nprim(), dfbs.max_nprim()),
                                        std::max(obs.max_l(), dfbs.max_l()), 0);
            engine.set(libint2::BraKet::xs_xx);

            auto shell2bf = obs.shell2bf();
            auto shell2bf_df = dfbs.shell2bf();
            const auto& results = engine.results();

            // Tensor<TensorType>::allocate(&ec, Zxy_tamm);

            #if 1
              //TODO: Screening?
              auto compute_2body_fock_dfC_lambda = [&](const IndexVector& blockid) {

                auto bi0 = blockid[0];
                auto bi1 = blockid[1];
                auto bi2 = blockid[2];

                const TAMM_SIZE size = Zxy_tamm.block_size(blockid);
                auto block_dims   = Zxy_tamm.block_dims(blockid);
                std::vector<TensorType> dbuf(size);

                auto bd1 = block_dims[1];
                auto bd2 = block_dims[2];

                auto s0range_end = df_shell_tile_map[bi0];
                decltype(s0range_end) s0range_start = 0l;

                if (bi0>0) s0range_start = df_shell_tile_map[bi0-1]+1;
              
                for (auto s0 = s0range_start; s0 <= s0range_end; ++s0) {
                // auto n0 = dfbs[s0].size();
                auto s1range_end = shell_tile_map[bi1];
                decltype(s1range_end) s1range_start = 0l;
                if (bi1>0) s1range_start = shell_tile_map[bi1-1]+1;
              
              for (auto s1 = s1range_start; s1 <= s1range_end; ++s1) {
                // auto n1 = shells[s1].size();

                auto s2range_end = shell_tile_map[bi2];
                decltype(s2range_end) s2range_start = 0l;
                if (bi2>0) s2range_start = shell_tile_map[bi2-1]+1;

                for (auto s2 = s2range_start; s2 <= s2range_end; ++s2) {
                  //// if (s2>s1) continue;
                  // auto n2 = shells[s2].size();
                  // auto n123 = n0*n1*n2;
                  // std::vector<TensorType> tbuf(n123);
                  engine.compute2<Operator::coulomb, BraKet::xs_xx, 0>(
                      dfbs[s0], unitshell, obs[s1], obs[s2]);
                  const auto* buf = results[0];
                  if (buf == nullptr) continue;     
                  // std::copy(buf, buf + n123, tbuf.begin());         
                  
                      tamm::Tile curshelloffset_i = 0U;
                      tamm::Tile curshelloffset_j = 0U;
                      tamm::Tile curshelloffset_k = 0U;
                      for(auto x=s1range_start;x<s1;x++) curshelloffset_i += AO_tiles[x];
                      for(auto x=s2range_start;x<s2;x++) curshelloffset_j += AO_tiles[x];
                      for(auto x=s0range_start;x<s0;x++) curshelloffset_k += dfAO_tiles[x];

                      size_t c = 0;
                      auto dimi =  curshelloffset_i + AO_tiles[s1];
                      auto dimj =  curshelloffset_j + AO_tiles[s2];
                      auto dimk =  curshelloffset_k + dfAO_tiles[s0];

                      for(auto k = curshelloffset_k; k < dimk; k++) 
                      for(auto i = curshelloffset_i; i < dimi; i++) 
                      for(auto j = curshelloffset_j; j < dimj; j++, c++) 
                          dbuf[(k*bd1+i)*bd2+j] = buf[c]; //tbuf[c]
                    } //s2
                  } //s1
                } //s0
                Zxy_tamm.put(blockid,dbuf);
              };

              do_t1 = std::chrono::high_resolution_clock::now();
              block_for(ec, Zxy_tamm(), compute_2body_fock_dfC_lambda);

              do_t2 = std::chrono::high_resolution_clock::now();
              do_time =
              std::chrono::duration_cast<std::chrono::duration<double>>((do_t2 - do_t1)).count();
              if(rank == 0 && debug) std::cout << "2BF-DFC:" << do_time << "s, ";
            #endif  
        
            Tensor<TensorType> K_tamm{tdfAO, tdfAO}; //ndf,ndf
            Tensor<TensorType>::allocate(&ec, K_tamm);

            /*** ============================== ***/
            /*** compute 2body-2index integrals ***/
            /*** ============================== ***/

            engine =
                Engine(libint2::Operator::coulomb, dfbs.max_nprim(), dfbs.max_l(), 0);
            engine.set(BraKet::xs_xs);
        
            const auto& buf2 = engine.results();

            auto compute_2body_2index_ints_lambda = [&](const IndexVector& blockid) {

              auto bi0 = blockid[0];
              auto bi1 = blockid[1];

              const TAMM_SIZE size = K_tamm.block_size(blockid);
              auto block_dims   = K_tamm.block_dims(blockid);
              std::vector<TensorType> dbuf(size);

              auto bd1 = block_dims[1];
              auto s1range_end = df_shell_tile_map[bi0];
              decltype(s1range_end) s1range_start = 0l;

              if (bi0>0) s1range_start = df_shell_tile_map[bi0-1]+1;
              
              for (auto s1 = s1range_start; s1 <= s1range_end; ++s1) {
                auto n1 = dfbs[s1].size();

                auto s2range_end = df_shell_tile_map[bi1];
                decltype(s2range_end) s2range_start = 0l;
                if (bi1>0) s2range_start = df_shell_tile_map[bi1-1]+1;

                for (auto s2 = s2range_start; s2 <= s2range_end; ++s2) {
                // if (s2>s1) continue;          
                // if(s2>s1){ TODO: screening doesnt work - revisit
                //   auto s2spl = dfbs_shellpair_list[s2];
                //   if(std::find(s2spl.begin(),s2spl.end(),s1) == s2spl.end()) continue;
                // }
                // else{
                //   auto s2spl = dfbs_shellpair_list[s1];
                //   if(std::find(s2spl.begin(),s2spl.end(),s2) == s2spl.end()) continue;
                // }

                  auto n2 = dfbs[s2].size();

                  std::vector<TensorType> tbuf(n1*n2);
                  engine.compute(dfbs[s1], dfbs[s2]);
                  if (buf2[0] == nullptr) continue;
                  Eigen::Map<const Matrix> buf_mat(buf2[0], n1, n2);
                  Eigen::Map<Matrix>(&tbuf[0],n1,n2) = buf_mat;

                  tamm::Tile curshelloffset_i = 0U;
                  tamm::Tile curshelloffset_j = 0U;
                  for(decltype(s1) x=s1range_start;x<s1;x++) curshelloffset_i += dfAO_tiles[x];
                  for(decltype(s2) x=s2range_start;x<s2;x++) curshelloffset_j += dfAO_tiles[x];

                  size_t c = 0;
                  auto dimi =  curshelloffset_i + dfAO_tiles[s1];
                  auto dimj =  curshelloffset_j + dfAO_tiles[s2];
                  for(auto i = curshelloffset_i; i < dimi; i++) 
                  for(auto j = curshelloffset_j; j < dimj; j++, c++) 
                          dbuf[i*bd1+j] = tbuf[c];

                //TODO: not needed if screening works
                // if (s1 != s2)  // if s1 >= s2, copy {s1,s2} to the corresponding {s2,s1}
                //              // block, note the transpose!
                // result.block(bf2, bf1, n2, n1) = buf_mat.transpose();                            
                }
              }
              K_tamm.put(blockid,dbuf);
            };

            block_for(ec, K_tamm(), compute_2body_2index_ints_lambda);

            Matrix V(ndf,ndf);
            V.setZero(); 
            tamm_to_eigen_tensor(K_tamm,V);
          
            #if 1
              auto ig1 = std::chrono::high_resolution_clock::now();

              std::vector<TensorType> eps(ndf);
              linalg::lapack::syevd( 'V', 'L', ndf, V.data(), ndf, eps.data() );

              Matrix Vp = V;

              for (size_t j=0; j<ndf; ++j) {
                double  tmp=1.0/sqrt(eps[j]);
                linalg::blas::scal( ndf, tmp, Vp.data() + j*ndf,   1 );
              }

              Matrix ke(ndf,ndf);
              cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, ndf, ndf, ndf,
                          1, Vp.data(), ndf, V.data(), ndf, 0, ke.data(), ndf);
              eigen_to_tamm_tensor(K_tamm,ke);

              // Scheduler{ec}
              // (k_tmp(d_mu,d_nu) = V_tamm(d_ku,d_mu) * V_tamm(d_ku,d_nu))
              // (K_tamm(d_mu,d_nu) = eps_tamm(d_mu,d_ku) * k_tmp(d_ku,d_nu)) .execute();

              //Tensor<TensorType>::deallocate(k_tmp, eps_tamm, V_tamm);

              auto ig2 = std::chrono::high_resolution_clock::now();
              auto igtime =
              std::chrono::duration_cast<std::chrono::duration<double>>((ig2 - ig1)).count();

              if(rank == 0 && debug) std::cout << "V^-1/2:" << igtime << "s, ";

            #endif


          // contract(1.0, Zxy, {1, 2, 3}, K, {1, 4}, 0.0, xyK, {2, 3, 4});
          // Tensor3D xyK = Zxy.contract(K,aidx_00); 
            Scheduler{ec}
            (xyK_tamm(mu,nu,d_nu) = Zxy_tamm(d_mu,mu,nu) * K_tamm(d_mu,d_nu)).execute();
            Tensor<TensorType>::deallocate(K_tamm); //release memory Zxy_tamm

          }  // if (!is_3c_init)
      
          Scheduler sch{ec};
          auto ig1 = std::chrono::high_resolution_clock::now();
          auto tig1 = ig1;   

          Tensor<TensorType> Jtmp_tamm{tdfAO}; //ndf
          Tensor<TensorType> xiK_tamm{tAO,tdfCocc,tdfAO}; //n, nocc, ndf
          Tensor<TensorType>& C_occ_tamm = ttensors.C_occ_tamm;
      
          eigen_to_tamm_tensor(C_occ_tamm,etensors.C_occ);

        auto ig2 = std::chrono::high_resolution_clock::now();
        auto igtime =
        std::chrono::duration_cast<std::chrono::duration<double>>((ig2 - ig1)).count();
        ec.pg().barrier();
        if(rank == 0 && debug) std::cout << " C_occ_tamm <- C_occ:" << igtime << "s, ";

        ig1 = std::chrono::high_resolution_clock::now();
        // contract(1.0, xyK, {1, 2, 3}, Co, {2, 4}, 0.0, xiK, {1, 4, 3});
        sch.allocate(xiK_tamm,Jtmp_tamm)
        (xiK_tamm(mu,dCocc_til,d_mu) = xyK_tamm(mu,nu,d_mu) * C_occ_tamm(nu, dCocc_til)).execute();

         ig2 = std::chrono::high_resolution_clock::now();
         igtime =
        std::chrono::duration_cast<std::chrono::duration<double>>((ig2 - ig1)).count();
        if(rank == 0 && debug) std::cout << " xiK_tamm:" << igtime << "s, ";

        ig1 = std::chrono::high_resolution_clock::now();
        // compute Coulomb
        // contract(1.0, xiK, {1, 2, 3}, Co, {1, 2}, 0.0, Jtmp, {3});
        // Jtmp = xiK.contract(Co,idx_0011); 
        sch(Jtmp_tamm(d_mu) = xiK_tamm(mu,dCocc_til,d_mu) * C_occ_tamm(mu,dCocc_til)).execute();

         ig2 = std::chrono::high_resolution_clock::now();
         igtime =
        std::chrono::duration_cast<std::chrono::duration<double>>((ig2 - ig1)).count();
        if(rank == 0 && debug) std::cout << " Jtmp_tamm:" << igtime << "s, ";

        ig1 = std::chrono::high_resolution_clock::now();
        // contract(1.0, xiK, {1, 2, 3}, xiK, {4, 2, 3}, 0.0, G, {1, 4});
        // Tensor2D K_ret = xiK.contract(xiK,idx_1122); 
        // xiK.resize(0, 0, 0);
        sch
        (F1tmp1(mu,ku) += -1.0 * xiK_tamm(mu,dCocc_til,d_mu) * xiK_tamm(ku,dCocc_til,d_mu))
        .deallocate(xiK_tamm).execute();

         ig2 = std::chrono::high_resolution_clock::now();
         igtime =
        std::chrono::duration_cast<std::chrono::duration<double>>((ig2 - ig1)).count();
        if(rank == 0 && debug) std::cout << " F1tmp1:" << igtime << "s, ";

        ig1 = std::chrono::high_resolution_clock::now();
        //contract(2.0, xyK, {1, 2, 3}, Jtmp, {3}, -1.0, G, {1, 2});
        // Tensor2D J_ret = xyK.contract(Jtmp,aidx_20);
        sch
        (F1tmp1(mu,nu) += 2.0 * xyK_tamm(mu,nu,d_mu) * Jtmp_tamm(d_mu))
        .deallocate(Jtmp_tamm).execute();
        // (F1tmp1(mu,nu) = 2.0 * J_ret_tamm(mu,nu))
        // (F1tmp1(mu,nu) += -1.0 * K_ret_tamm(mu,nu)).execute();

         ig2 = std::chrono::high_resolution_clock::now();
         igtime =        
        std::chrono::duration_cast<std::chrono::duration<double>>((ig2 - ig1)).count();
        if(rank == 0 && debug) std::cout << " F1tmp1:" << igtime << "s, ";
        
      auto tig2 = std::chrono::high_resolution_clock::now();
      auto tigtime =
      std::chrono::duration_cast<std::chrono::duration<double>>((tig2 - tig1)).count();  

      if(rank == 0 && debug) std::cout << "3c contractions:" << tigtime << "s, ";
        
        #endif
      } //end density fitting
}

template<typename TensorType>
void energy_diis(ExecutionContext& ec, TiledIndexSpace& tAO, int iter, int max_hist, 
                 Tensor<TensorType> D, Tensor<TensorType> F, Tensor<TensorType> ehf_tamm,
                 std::vector<Tensor<TensorType>>& D_hist, 
                 std::vector<Tensor<TensorType>>& fock_hist, std::vector<Tensor<TensorType>>& ehf_tamm_hist) {

  tamm::Scheduler sch{ec};

  auto rank = ec.pg().rank().value();

  // if(rank == 0) cout << "contructing pulay matrix" << endl;
  
  int64_t idim = std::min((int)D_hist.size(), max_hist);
  if(idim < 2) return;

  int64_t info = -1;
  int64_t N    = idim+1;
  std::vector<double> X(N, 0.);

  Tensor<TensorType> dhi_trace{};
  auto [mu, nu] = tAO.labels<2>("all");
  Tensor<TensorType>::allocate(&ec,dhi_trace); 

  // ----- Construct Pulay matrix -----
  Matrix A = Matrix::Zero(idim + 1, idim + 1);
  Tensor<TensorType> dFij{tAO, tAO};
  Tensor<TensorType> dDij{tAO, tAO};
  Tensor<TensorType>::allocate(&ec,dFij,dDij);

  for(int i = 0; i < idim; i++) {
    for(int j = i+1; j < idim; j++) {
      sch
        (dFij()       = fock_hist[i]())
        (dFij()      -= fock_hist[j]())
        (dDij()       = D_hist[i]())
        (dDij()      -= D_hist[j]())
        (dhi_trace()  = dFij(nu,mu) * dDij(mu,nu))
        .execute();
      double dhi = get_scalar(dhi_trace); 
      A(i+1,j+1) = dhi;
    }
  }
  Tensor<TensorType>::deallocate(dFij,dDij);

  for(int i = 1; i <= idim; i++) {
    for(int j = i+1; j <= idim; j++) { 
      A(j, i) = A(i, j); 
    }
  }

  for(int i = 1; i <= idim; i++) {
    A(i, 0) = -1.0;
    A(0, i) = -1.0;
  }

  // if(rank == 0) cout << std::setprecision(8) << "in ediis, A: " << endl << A << endl;

  while(info!=0) {

    N  = idim+1;
    std::vector<double> AC( N*(N+1)/2 );
    std::vector<double> AF( AC.size() );
    std::vector<double> B(N, 0.); B.front() = -1.0;
    // if(rank == 0) cout << std::setprecision(8) << "in ediis, B ini: " << endl << B << endl;
    for(int i = 1; i < N; i++) {
      double dhi = get_scalar(ehf_tamm_hist[i-1]);
      // if(rank==0) cout << "i,N,dhi: " << i << "," << N << "," << dhi << endl;
      B[i] = dhi;
    }
    X.resize(N);

    if(rank == 0) cout << std::setprecision(8) << "in ediis, B: " << endl << B << endl;

    int ac_i = 0;
    for(int i = 0; i <= idim; i++) {
      for(int j = i; j <= idim; j++) { 
        AC[ac_i] = A(j, i); ac_i++; 
      }
    }

    double RCOND;
    std::vector<int64_t> IPIV(N);
    std::vector<double>  BERR(1), FERR(1); // NRHS
    info = linalg::lapack::spsvx( 'N', 'L', N, 1, AC.data(), AF.data(), 
      IPIV.data(), B.data(), N, X.data(), N, RCOND, FERR.data(), BERR.data() );
      
    if(info!=0) {
      // if(rank==0) cout << "<E-DIIS> Singularity in Pulay matrix. Density and Fock difference matrices removed." << endl;
      fock_hist.erase(fock_hist.begin());
      D_hist.erase(D_hist.begin());
      idim--;
      if(idim==1) return;
      Matrix A_pl = A.block(2,2,idim,idim);
      Matrix A_new  = Matrix::Zero(idim + 1, idim + 1);

      for(int i = 1; i <= idim; i++) {
          A_new(i, 0) = -1.0;
          A_new(0, i) = -1.0;
      }      
      A_new.block(1,1,idim,idim) = A_pl;
      A=A_new;
    }

  } //while

  Tensor<TensorType>::deallocate(dhi_trace); 

  // if(rank == 0) cout << std::setprecision(8) << "in ediis, X: " << endl << X << endl;

  std::vector<double> X_final(N);
  //Reordering [0...N] -> [1...N,0] to match eigen's lu solve
  X_final.back() = X.front();
  std::copy ( X.begin()+1, X.end(), X_final.begin() );

  if(rank == 0) cout << std::setprecision(8) << "in ediis, X_reordered: " << endl << X_final << endl;

  sch
    (D() = 0)
    (F() = 0)
    (ehf_tamm() = 0)
    .execute();
  for(int j = 0; j < idim; j++) { 
    sch
      (D() += X_final[j] * D_hist[j]())
      (F() += X_final[j] * fock_hist[j]())
      (ehf_tamm() += X_final[j] * ehf_tamm_hist[j]())
      ; 
  }
  sch.execute();

  // if(rank == 0) cout << "end of ediis" << endl;

}

template<typename TensorType>
void diis(ExecutionContext& ec, TiledIndexSpace& tAO, Tensor<TensorType> D, Tensor<TensorType> F, 
          Tensor<TensorType> err_mat, int iter, int max_hist, int ndiis, const int n_lindep,
          std::vector<Tensor<TensorType>>& diis_hist, std::vector<Tensor<TensorType>>& fock_hist) {
  
  using Vector =
      Eigen::Matrix<TensorType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  tamm::Scheduler sch{ec};

  auto rank = ec.pg().rank().value();

  if(ndiis > max_hist) {
    auto maxe = 0;
    if(!switch_diis) {
      std::vector<TensorType> max_err(diis_hist.size());
      for (size_t i=0; i<diis_hist.size(); i++) {
        max_err[i] = tamm::norm(diis_hist[i]);
      }

      maxe = std::distance(max_err.begin(), 
             std::max_element(max_err.begin(),max_err.end()));
    }
    diis_hist.erase(diis_hist.begin()+maxe);
    fock_hist.erase(fock_hist.begin()+maxe);      
  }
  else{
    if(ndiis == (int)(max_hist/2) && n_lindep > 1) {
      diis_hist.clear();
      fock_hist.clear();
    }
  }

  Tensor<TensorType> Fcopy{tAO, tAO};
  Tensor<TensorType>::allocate(&ec,Fcopy);
  sch(Fcopy() = F()).execute();

  diis_hist.push_back(err_mat);
  fock_hist.push_back(Fcopy);

  Matrix A;
  Vector b;
  int  idim = std::min((int)diis_hist.size(), max_hist);
  int64_t info = -1;
  int64_t N = idim+1;
  std::vector<TensorType> X;

  // Tensor<TensorType> dhi_trans{tAO, tAO};
  Tensor<TensorType> dhi_trace{};
  auto [mu, nu] = tAO.labels<2>("all");
  Tensor<TensorType>::allocate(&ec,dhi_trace); //dhi_trans

  // ----- Construct Pulay matrix -----
  A       = Matrix::Zero(idim + 1, idim + 1);

  for(int i = 0; i < idim; i++) {
      for(int j = i; j < idim; j++) {
          //A(i, j) = (diis_hist[i].transpose() * diis_hist[j]).trace();
          sch(dhi_trace() = diis_hist[i](nu,mu) * diis_hist[j](mu,nu)).execute();
          TensorType dhi = get_scalar(dhi_trace); //dhi_trace.trace();
          A(i+1,j+1) = dhi;
      }
  }

  for(int i = 1; i <= idim; i++) {
      for(int j = i; j <= idim; j++) { A(j, i) = A(i, j); }
  }

  for(int i = 1; i <= idim; i++) {
      A(i, 0) = -1.0;
      A(0, i) = -1.0;
  }

  while(info!=0) {

    if(idim==1) return;

    N  = idim+1;
    std::vector<TensorType> AC( N*(N+1)/2 );
    std::vector<TensorType> AF( AC.size() );
    std::vector<TensorType> B(N, 0.); B.front() = -1.0;
    X.resize(N);

    int ac_i = 0;
    for(int i = 0; i <= idim; i++) {
        for(int j = i; j <= idim; j++) { AC[ac_i] = A(j, i); ac_i++; }
    }

    TensorType RCOND;
    std::vector<int64_t> IPIV(N);
    std::vector<TensorType>  BERR(1), FERR(1); // NRHS
    info = linalg::lapack::spsvx( 'N', 'L', N, 1, AC.data(), AF.data(), 
      IPIV.data(), B.data(), N, X.data(), N, RCOND, FERR.data(), BERR.data() );
      
    if(info!=0) {
      if(rank==0) cout << "<DIIS> Singularity in Pulay matrix detected." /*Error and Fock matrices removed." */ << endl;
      diis_hist.erase(diis_hist.begin());
      fock_hist.erase(fock_hist.begin());            
      idim--;
      if(idim==1) return;
      Matrix A_pl = A.block(2,2,idim,idim);
      Matrix A_new  = Matrix::Zero(idim + 1, idim + 1);

      for(int i = 1; i <= idim; i++) {
          A_new(i, 0) = -1.0;
          A_new(0, i) = -1.0;
      }      
      A_new.block(1,1,idim,idim) = A_pl;
      A=A_new;
    }

  } //while

  Tensor<TensorType>::deallocate(dhi_trace); //dhi_trans

  std::vector<TensorType> X_final(N);
  //Reordering [0...N] -> [1...N,0] to match eigen's lu solve
  X_final.back() = X.front();
  std::copy ( X.begin()+1, X.end(), X_final.begin() );

  //if(rank==0 && scf_options.debug) 
  // std::cout << "diis weights sum, vector: " << std::setprecision(5) << std::accumulate(X.begin(),X.end(),0.0d) << std::endl << X << std::endl << X_final << std::endl;

  sch(F() = 0).execute();
  for(int j = 0; j < idim; j++) { 
    sch(F() += X_final[j] * fock_hist[j]()); 
  }
  // sch(F() += 0.5 * D()); //level shift
  sch.execute();

}

#endif // TAMM_METHODS_HF_TAMM_COMMON_HPP_
