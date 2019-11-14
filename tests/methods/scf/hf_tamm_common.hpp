
#ifndef TAMM_TESTS_HF_TAMM_COMMON_HPP_
#define TAMM_TESTS_HF_TAMM_COMMON_HPP_

#include "hf_common.hpp"

//TODO: UHF,ROHF,diis,3c,dft

template<typename TensorType>
void diis(ExecutionContext& ec, TiledIndexSpace& tAO, Tensor<TensorType> F, Tensor<TensorType> err_mat, 
          int iter, int max_hist, int ndiis,
          std::vector<Tensor<TensorType>>& diis_hist, std::vector<Tensor<TensorType>>& fock_hist);

template<typename TensorType>
void compute_1body_ints(ExecutionContext& ec, Tensor<TensorType>& tensor1e, 
      std::vector<libint2::Atom>& atoms, libint2::BasisSet& shells, libint2::Operator otype,
      std::vector<size_t>& shell_tile_map, std::vector<Tile>& AO_tiles);

std::tuple<int,int,int> get_hf_nranks(const size_t N){

    // auto nranks = GA_Nnodes();
    auto nnodes = GA_Cluster_nnodes();
    auto ppn = GA_Cluster_nprocs(0);

    int hf_guessranks = std::ceil(0.15*N);
    int hf_nnodes = hf_guessranks/ppn;
    if(hf_guessranks%ppn>0 || hf_nnodes==0) hf_nnodes++;
    if(hf_nnodes > nnodes) hf_nnodes = nnodes;
    int hf_nranks = hf_nnodes * ppn;

    return std::make_tuple(hf_nnodes,ppn,hf_nranks);
}

template<typename T, int ndim>
void t2e_hf_helper(const ExecutionContext& ec, tamm::Tensor<T>& ttensor,Matrix& etensor,
                   const std::string& ustr = "") {

    const string pstr = "(" + ustr + ")";                     

    auto hf_t1 = std::chrono::high_resolution_clock::now();

    const auto rank = ec.pg().rank();
    const auto N = etensor.rows(); //TODO

    if(rank == 0)
      tamm_to_eigen_tensor(ttensor, etensor);
    ec.pg().barrier();
    std::vector<T> Hbufv(N*N);
    T *Hbuf = &Hbufv[0];//Hbufv.data();
    Eigen::Map<Matrix>(Hbuf,N,N) = etensor;  
    // GA_Brdcst(Hbuf,N*N*sizeof(T),0);
    MPI_Bcast(Hbuf,N*N,mpi_type<T>(),0,ec.pg().comm());
    etensor = Eigen::Map<Matrix>(Hbuf,N,N);
    Hbufv.clear(); Hbufv.shrink_to_fit();

    auto hf_t2 = std::chrono::high_resolution_clock::now();
    auto hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
   
    //ec.pg().barrier(); //TODO
    if(rank == 0) std::cout << "\nTime for tamm to eigen " << pstr << " : " << hf_time << " secs" << endl;
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


std::tuple<int,double> compute_NRE(const ExecutionContext& ec, std::vector<libint2::Atom>& atoms){

  auto rank = ec.pg().rank(); 
      //  std::cout << "Geometries in bohr units \n";
    //  for (auto i = 0; i < atoms.size(); ++i)
    //    std::cout << atoms[i].atomic_number << "  " << atoms[i].x<< "  " <<
    //    atoms[i].y<< "  " << atoms[i].z << endl;
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
    if(rank==0) cout << "\nNuclear repulsion energy = " << enuc << endl;

    return std::make_tuple(ndocc,enuc);

}


std::tuple<std::vector<size_t>, std::vector<Tile>, std::vector<Tile>> 
    compute_AO_tiles(const ExecutionContext& ec, libint2::BasisSet& shells){

      tamm::Tile tile_size = scf_options.AO_tilesize; 

      auto rank = ec.pg().rank();

      auto N = nbasis(shells);
  
    //heuristic to set tilesize to atleast 5% of nbf
    if(tile_size < N*0.05 && !scf_options.force_tilesize) {
      tile_size = std::ceil(N*0.05);
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


Matrix compute_orthogonalizer(const ExecutionContext& ec, const Matrix& S) {
    
    auto hf_t1 = std::chrono::high_resolution_clock::now();
    auto rank = ec.pg().rank(); 

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
    double S_condition_number_threshold = 1.0 / scf_options.tol_lindep;
        //1.0 / std::numeric_limits<double>::epsilon();
    std::tie(X, Xinv, XtX_condition_number) =
        conditioning_orthogonalizer(ec, S, S_condition_number_threshold);

    // TODO Redeclare TAMM S1 with new dims?
    auto hf_t2 = std::chrono::high_resolution_clock::now();
    auto hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

    //TODO: not used ?
    Xinv.resize(0,0);

    if(rank == 0) std::cout << "Time for computing orthogonalizer: " << hf_time << " secs\n" << endl;

    return X;

}


template<typename TensorType> //TODO return H, S copy ?
std::tuple<Matrix, Matrix, Tensor<TensorType>, Tensor<TensorType>> compute_hamiltonian(
ExecutionContext& ec, std::vector<libint2::Atom>& atoms, libint2::BasisSet& shells,
      std::vector<size_t>& shell_tile_map, std::vector<Tile>& AO_tiles){

    using libint2::Operator;
    const size_t N = nbasis(shells);
    auto rank = ec.pg().rank();

    Tensor<TensorType> H1{tAO, tAO};
    Tensor<TensorType> S1{tAO, tAO};
    Tensor<TensorType> T1{tAO, tAO};
    Tensor<TensorType> V1{tAO, tAO};
    Tensor<TensorType>::allocate(&ec, H1, S1, T1, V1);

    auto [mu, nu] = tAO.labels<2>("all");

    Matrix H = Matrix::Zero(N,N);
    Matrix S = Matrix::Zero(N,N);

    auto hf_t1 = std::chrono::high_resolution_clock::now();

    // Scheduler{ec}(S1(mu, nu) = SL1(mu, nu)).execute();
    compute_1body_ints(ec,S1,atoms,shells,Operator::overlap,shell_tile_map,AO_tiles);
    compute_1body_ints(ec,T1,atoms,shells,Operator::kinetic,shell_tile_map,AO_tiles);
    compute_1body_ints(ec,V1,atoms,shells,Operator::nuclear,shell_tile_map,AO_tiles);
    auto hf_t2 = std::chrono::high_resolution_clock::now();
    auto hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
    if(rank == 0) std::cout << "\nTime for computing 1-e integrals T, V, S: " << hf_time << " secs" << endl;

    hf_t1 = std::chrono::high_resolution_clock::now();
    // Core Hamiltonian = T + V
    Scheduler{ec}
      (H1(mu, nu) = T1(mu, nu))
      (H1(mu, nu) += V1(mu, nu)).execute();
    hf_t2 = std::chrono::high_resolution_clock::now();
    hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
    if(rank == 0) std::cout << "\nTime for computing Core Hamiltonian H=T+V: " << hf_time << " secs" << endl;

    t2e_hf_helper<TensorType,2>(ec, H1, H, "H1-H");
    t2e_hf_helper<TensorType,2>(ec, S1, S, "S1-S");

    Tensor<TensorType>::deallocate(T1, V1);

    return std::make_tuple(H,S,H1,S1);

}

void scf_restart(const ExecutionContext& ec, const size_t& N, const std::string& filename,
   const int& ndocc, Matrix& C, Matrix& D){

    const auto rank = ec.pg().rank();

    std::vector<TensorType> Dbuf(N*N);
    TensorType *Dbufp = &Dbuf[0];
    int rstatus = 0;
    std::string movecsfile = getfilename(filename) +
       "." + scf_options.basis + ".movecs";

    if(rank==0) 
    {
      cout << "Reading movecs from file... ";
      std::vector<TensorType> Cbuf(N*N);
      TensorType *Hbuf = &Cbuf[0];
      std::ifstream in(movecsfile, std::ios::in | std::ios::binary);
      if(in.is_open()) rstatus = 1;
      if(rstatus == 1){
        in.read((char *) Hbuf, sizeof(TensorType)*N*N);
        C = Eigen::Map<Matrix>(Hbuf,N,N);
        cout << "done" << endl;
        auto C_occ = C.leftCols(ndocc);
        D = C_occ * C_occ.transpose();
        Eigen::Map<Matrix>(Dbufp,N,N)=D;
      }
    }
    ec.pg().barrier();
    // GA_Brdcst(&rstatus,sizeof(int),0);
    MPI_Bcast(&rstatus,1,mpi_type<int>(),0,ec.pg().comm());
    if(rstatus == 0) nwx_terminate("Error reading " + movecsfile);
    // GA_Brdcst(Dbufp,N*N*sizeof(TensorType),0);
    MPI_Bcast(Dbufp,N*N,mpi_type<TensorType>(),0,ec.pg().comm());
    D = Eigen::Map<Matrix>(Dbufp,N,N);
    //Dbuf.clear();
    ec.pg().barrier();
}

template<typename TensorType>
std::tuple<TensorType,TensorType> scf_iter_body(ExecutionContext& ec, 
#ifdef SCALAPACK
      CXXBLACS::BlacsGrid* blacs_grid,
#endif
      const int& iter, const int& ndocc, const Matrix& X, Matrix& F, 
      Matrix& C, Matrix& C_occ, Matrix& D,  Tensor<TensorType>& S1,
      Tensor<TensorType>& F1,Tensor<TensorType>& H1,Tensor<TensorType>& F1tmp1, 
      Tensor<TensorType>& FD_tamm, Tensor<TensorType>& FDS_tamm, 
      Tensor<TensorType>& D_tamm, Tensor<TensorType>& D_last_tamm,
      Tensor<TensorType>& D_diff, Tensor<TensorType>& ehf_tmp, 
      Tensor<TensorType>& ehf_tamm,
      std::vector<tamm::Tensor<TensorType>>& diis_hist, 
      std::vector<tamm::Tensor<TensorType>>& fock_hist,
      bool scf_restart=false){

        Scheduler sch{ec};

        const int64_t N = F.rows();

        auto rank = ec.pg().rank(); 
        auto debug = scf_options.debug;
        auto [mu, nu, ku] = tAO.labels<3>("all"); //TODO
        int max_hist = scf_options.diis_hist; 

        auto do_t1 = std::chrono::high_resolution_clock::now();
        // F += Ftmp;
        sch(F1(mu, nu) = H1(mu, nu))
           (F1(mu, nu) += F1tmp1(mu, nu)).execute();

        auto do_t2 = std::chrono::high_resolution_clock::now();
        auto do_time =
        std::chrono::duration_cast<std::chrono::duration<double>>((do_t2 - do_t1)).count();

        if(rank == 0 && debug) std::cout << "F=H+2BF:" << do_time << "s, ";

        // tamm_to_eigen_tensor(F1,F);
        // Matrix FSm12 = F * Sm12;
        // Matrix Sp12D = Sp12 * D_last;
        // Matrix SpFS  = Sp12D * FSm12;

        // // Assemble: S^(-1/2)*F*D*S^(1/2) - S^(1/2)*D*F*S^(-1/2)
        // err_mat = SpFS.transpose() - SpFS;    

        // eigen_to_tamm_tensor(D_last_tamm,D_last);

        Tensor<TensorType> err_mat_tamm{tAO, tAO};
        Tensor<TensorType>::allocate(&ec, err_mat_tamm);

        do_t1 = std::chrono::high_resolution_clock::now();
        
        // Scheduler{*ec}(FD_tamm(mu,nu) = F1(mu,ku) * Sm12_tamm(ku,nu))
        // (Sp12D_tamm(mu,nu) = Sp12_tamm(mu,ku) * D_last_tamm(ku,nu))
        // (FDS_tamm(mu,nu)  = Sp12D_tamm(mu,ku) * FD_tamm(ku,nu))

        sch(FD_tamm(mu,nu) = F1(mu,ku) * D_last_tamm(ku,nu))
           (FDS_tamm(mu,nu)  = FD_tamm(mu,ku) * S1(ku,nu))
    
        //FDS-SDF
        (err_mat_tamm(mu,nu) = FDS_tamm(mu,nu))
        (err_mat_tamm(mu,nu) -= FDS_tamm(nu,mu)).execute();

        do_t2 = std::chrono::high_resolution_clock::now();
        do_time =
        std::chrono::duration_cast<std::chrono::duration<double>>((do_t2 - do_t1)).count();

        if(rank == 0 && debug) std::cout << "FDS-SDF:" << do_time << "s, ";        

        ec.pg().barrier();
        //tamm_to_eigen_tensor(err_mat_tamm,err_mat);
        //tamm_to_eigen_tensor(F1,F);

        do_t1 = std::chrono::high_resolution_clock::now();

        if(iter > 1) {
            ++idiis;
            diis(ec, tAO, F1, err_mat_tamm, iter, max_hist, idiis,
                diis_hist, fock_hist);
        }
    
        do_t2 = std::chrono::high_resolution_clock::now();
        do_time =
        std::chrono::duration_cast<std::chrono::duration<double>>((do_t2 - do_t1)).count();

        if(rank == 0 && debug) std::cout << "diis:" << do_time << "s, ";    
        tamm_to_eigen_tensor(F1,F);

        do_t1 = std::chrono::high_resolution_clock::now();
        if(!scf_restart){
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
        #ifdef SCALAPACK

        // Allocate space for C_ortho
        Matrix C_ortho( F.rows(), F.cols() );
        
        if( blacs_grid ){ // Scope temp data

        Matrix F_ortho = X.transpose() * F * X;

        auto [MLoc, NLoc] = blacs_grid->getLocalDims( N, N );
        std::vector< double > F_ortho_loc( MLoc*NLoc );
        std::vector< double > C_ortho_loc( MLoc*NLoc );
        std::vector< double > F_eig( F.rows() );

        auto DescF = blacs_grid->descInit( N, N, 0, 0, MLoc );

        // Scatter copy of F_ortho from root rank to all ranks
        // FIXME: should just grab local data from replicated 
        //   matrix
        blacs_grid->Scatter( N, N, F_ortho.data(), N, 
                            F_ortho_loc.data(), MLoc,
                            0, 0 );

        // Diagonalize
        auto PSYEV_INFO = CXXBLACS::PSYEV( 'V', 'U', N, 
                        F_ortho_loc.data(), 1, 1, DescF,
                        F_eig.data(),
                        C_ortho_loc.data(), 1, 1, DescF);

        if( PSYEV_INFO ) {
          std::runtime_error err("PSYEV Failed");
          throw err;
        }

        // Gather the eigenvectors to root process and replicate
        blacs_grid->Gather( N, N, C_ortho.data(), N, 
                          C_ortho_loc.data(), MLoc,
                          0,0 );

        } // ScaLAPACK Scope


        MPI_Bcast( C_ortho.data(), C_ortho.size(), MPI_DOUBLE,
                  0, ec.pg().comm() );
                  

//      C_ortho.transposeInPlace(); // Col -> Row Major
        



        // Backtransform C
        C = X * C_ortho.transpose();

        #elif defined(EIGEN_DIAG)

          Eigen::SelfAdjointEigenSolver<Matrix> eig_solver(X.transpose() * F *
                                                          X);
          //eps = eig_solver.eigenvalues();
          C = X * eig_solver.eigenvectors();
        
        #else
        
          const int64_t Northo = X.cols();
          assert( N == Northo );

          if( rank == 0 ) {
          Matrix Fp = F; // XXX: Can F be destroyed?
          // TODO: Check for linear dep case

          // Take into account row major
          // X -> X**T
          // Ft is [N,N]
          // X**T is [Northo, N]
          C.resize(N,N);
          linalg::blas::gemm( 'N', 'T', N, Northo, N,
                              1., Fp.data(), N, X.data(), Northo, 
                              0., C.data(), N );
          linalg::blas::gemm( 'N', 'N', Northo, Northo, N,
                              1., X.data(), Northo, C.data(), N, 
                              0., Fp.data(), Northo );

          std::vector<double> eps(Northo);
          linalg::lapack::syevd( 'V', 'L', Northo, Fp.data(), Northo, eps.data() );
          // Take into account row major
          // X -> X**T
          // C -> C**T = Ft**T * X**T
          // X**T is [Northo, N]
          // Fp is [Northo, Northo]
          // C**T is [Northo, N] 
          C.resize(N, Northo);
          linalg::blas::gemm( 'T', 'N', Northo, N, Northo, 1., Fp.data(), Northo, X.data(), Northo, 0., C.data(), Northo );

          } else C.resize( N, Northo );


          if( ec.pg().size() > 1 )
            MPI_Bcast( C.data(), C.size(), MPI_DOUBLE, 0, ec.pg().comm() );
      #endif

        // compute density, D = C(occ) . C(occ)T
        C_occ = C.leftCols(ndocc);
        D          = C_occ * C_occ.transpose();

        do_t2 = std::chrono::high_resolution_clock::now();
        do_time =
        std::chrono::duration_cast<std::chrono::duration<double>>((do_t2 - do_t1)).count();

        if(rank == 0 && debug) std::cout << "eigen_solve:" << do_time << "s, ";   
        }//end scf_restart 

        do_t1 = std::chrono::high_resolution_clock::now();

        eigen_to_tamm_tensor(D_tamm,D);
        // eigen_to_tamm_tensor(F1,F);

        do_t2 = std::chrono::high_resolution_clock::now();
        do_time =
        std::chrono::duration_cast<std::chrono::duration<double>>((do_t2 - do_t1)).count();

        if(rank == 0 && debug) std::cout << "E2T-D:" << do_time << "s, ";  

         do_t1 = std::chrono::high_resolution_clock::now();
        // compute HF energy 
        // e = D * (H+F);
        sch
           (ehf_tmp(mu,nu) = H1(mu,nu))
           (ehf_tmp(mu,nu) += F1(mu,nu))
           (ehf_tamm() = D_tamm() * ehf_tmp()).execute();

        auto ehf = get_scalar(ehf_tamm);

        // rmsd  = (D - D_last).norm();
        sch(D_diff() = D_tamm())
           (D_diff() -= D_last_tamm()).execute();
        auto rmsd = norm(D_diff);

        do_t2 = std::chrono::high_resolution_clock::now();
        do_time =
        std::chrono::duration_cast<std::chrono::duration<double>>((do_t2 - do_t1)).count();

        if(rank == 0 && debug) std::cout << "HF-Energy:" << do_time << "s" << endl;    

        return std::make_tuple(ehf,rmsd);       
}

template<typename TensorType>
void compute_2bf(ExecutionContext& ec, const libint2::BasisSet& obs,
      const bool do_schwarz_screen, const std::vector<size_t>& shell2bf,
      const Matrix& SchwarzK, Matrix& G, const Matrix& D, 
      Tensor<TensorType>& F1tmp,Tensor<TensorType>& F1tmp1, 
      const size_t& max_nprim4, libint2::BasisSet& shells,
      const bool do_density_fitting=false){

      using libint2::Operator;

        auto N = G.rows(); //TODO
        double fock_precision = std::min(scf_options.tol_int, 0.01 * scf_options.conve);
        auto rank = ec.pg().rank();
        auto debug = scf_options.debug;

        // const auto precision_F = std::min(
        // std::min(1e-3 / XtX_condition_number, 1e-7),
        // std::max(rmsd / 1e4, std::numeric_limits<double>::epsilon()));

        // Matrix Ftmp = compute_2body_fock(shells, D, tol_int, SchwarzK);
        // eigen_to_tamm_tensor(F1tmp, Ftmp);

        
        auto do_t1 = std::chrono::high_resolution_clock::now();

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

      // Matrix Gt = 0.5*(G + G.transpose());
      // G = Gt;
    };

  #endif

    
    decltype(do_t1) do_t2;
    double do_time;

    if(!do_density_fitting){
      G.setZero(N,N);
      block_for(ec, F1tmp(), comp_2bf_lambda);
      Matrix Gt = 0.5*(G + G.transpose());
      G = Gt;
      Gt.resize(0,0);
      do_t2 = std::chrono::high_resolution_clock::now();
      do_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((do_t2 - do_t1)).count();

      if(rank == 0 && debug) std::cout << "2BF:" << do_time << "s, ";
      // ec.pg().barrier();

      do_t1 = std::chrono::high_resolution_clock::now();
      eigen_to_tamm_tensor_acc(F1tmp1,G);
      do_t2 = std::chrono::high_resolution_clock::now();
      do_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((do_t2 - do_t1)).count();

      if(rank == 0 && debug) std::cout << "E2T-ACC-G-F1:" << do_time << "s, ";
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

          Tensor<TensorType> Zxy_tamm{tdfAO, tAO, tAO}; //ndf,n,n
          Tensor<TensorType>::allocate(&ec, Zxy_tamm);

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
              }  //s1
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
          // cout << V << endl;
          // V = V.sqrt().inverse();
          // eigen_to_tamm_tensor(K_tamm,V);

          auto ig1 = std::chrono::high_resolution_clock::now();

          std::vector<TensorType> eps(ndf);
          linalg::lapack::syevd( 'V', 'L', ndf, V.data(), ndf, eps.data() );

          // Tensor<TensorType> V_tamm{tdfAO, tdfAO}; //ndf,ndf
          // Tensor<TensorType> k_tmp{tdfAO, tdfAO}; 
          // Tensor<TensorType> eps_tamm{tdfAO, tdfAO};
          // Tensor<TensorType>::allocate(&ec, k_tmp, eps_tamm, V_tamm);
          // eigen_to_tamm_tensor(V_tamm,V);

          Matrix Vp = V;

          for (size_t j=0; j<ndf; ++j) {
          double  tmp=1.0/sqrt(eps[j]);
          linalg::blas::scal( ndf, tmp, Vp.data() + j*ndf,   1 );
          // for (size_t i=0; i<ndf; ++i) {
          //   Vp(j,i) = V(j,i) * tmp; 
          // }
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
          Tensor<TensorType>::deallocate(K_tamm,Zxy_tamm); //release memory

      }  // if (!is_3c_init)
      
        Scheduler sch{ec};
        auto ig1 = std::chrono::high_resolution_clock::now();
        auto tig1 = ig1;

        Tensor<TensorType> Jtmp_tamm{tdfAO}; //ndf
        Tensor<TensorType> xiK_tamm{tAO,tdfCocc,tdfAO}; //n, nocc, ndf
      
        eigen_to_tamm_tensor(C_occ_tamm,C_occ);

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
void compute_initial_guess(ExecutionContext& ec, const int& ndocc,
      const std::vector<libint2::Atom>& atoms, const libint2::BasisSet& shells,
      const std::string& basis, const Matrix& X, const Matrix& H, 
      Matrix& C, Matrix& C_occ, Matrix& D){

    const auto rank       = ec.pg().rank();
    const auto world_size = ec.pg().size();
    const auto N = nbasis(shells);
    
    auto D_minbs = compute_soad(atoms);  // compute guess in minimal basis
    libint2::BasisSet minbs("STO-3G", atoms);

    #ifndef NDEBUG
      std::tie(minbs_shellpair_list, minbs_shellpair_data) = compute_shellpairs(minbs);
    #endif

      if(rank == 0) std::cout << 
    "\nProjecting minimal basis SOAD onto basis set specified (" << basis << ")" << endl;
    
    auto ig1 = std::chrono::high_resolution_clock::now();

    auto Ft = H;

    // Ft += compute_2body_fock_general(
    //     shells, D_minbs, minbs, true /* SOAD_D_is_shelldiagonal */,
    //     std::numeric_limits<double>::epsilon()  // this is cheap, no reason
    //                                             // to be cheaper
    //     );

  double precision = std::numeric_limits<double>::epsilon();
  bool D_is_shelldiagonal = true;
  const libint2::BasisSet& obs = shells;
  const libint2::BasisSet& D_bs = minbs;
  // const Matrix& D = D_minbs; 

  Matrix G = Matrix::Zero(N,N);
  Tensor<TensorType> F1tmp{tAOt, tAOt}; //not allocated
  Tensor<TensorType> F1tmp1{tAO, tAO};
  Tensor<TensorType>::allocate(&ec, F1tmp1);

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
        // if(s2>s1) return;

        auto sp12_iter = obs_shellpair_data.at(s1).begin();
        auto s2spl = obs_shellpair_list[s1];
        auto s2_itr = std::find(s2spl.begin(),s2spl.end(),s2);
        if(s2_itr == s2spl.end()) return;
        auto s2_pos = std::distance(s2spl.begin(),s2_itr);

        std::advance(sp12_iter,s2_pos);
        const auto* sp12 = sp12_iter->get();

        auto bf2_first = shell2bf[s2];
        auto n2 = obs[s2].size();

        for (decltype(s1) s3 = 0; s3 < D_bs.size(); ++s3) {
          auto bf3_first = shell2bf_D[s3];
          auto n3 = D_bs[s3].size();

          auto s4_begin = D_is_shelldiagonal ? s3 : 0;
          auto s4_fence = D_is_shelldiagonal ? s3 + 1 : D_bs.size();

          #ifndef NDEBUG
            auto sp34_iter = minbs_shellpair_data.at(s3).begin();
          #endif

          for (decltype(s1) s4 = s4_begin; s4 != s4_fence; ++s4) {

            #ifndef NDEBUG
              auto s4spl = minbs_shellpair_list[s3];
              auto s4_itr = std::find(s4spl.begin(),s4spl.end(),s4);
              if(s4_itr == s4spl.end()) continue;
              auto s4_pos = std::distance(s4spl.begin(),s4_itr);

              std::advance(sp34_iter,s4_pos);
              const auto* sp34 = sp34_iter->get();
            #endif 

            auto bf4_first = shell2bf_D[s4];
            auto n4 = D_bs[s4].size();

            // compute the permutational degeneracy (i.e. # of equivalents) of
            // the given shell set
            auto s12_deg = (s1 == s2) ? 1.0 : 2.0;

            if (s3 >= s4) {
              auto s34_deg = (s3 == s4) ? 1.0 : 2.0;
              auto s1234_deg = s12_deg * s34_deg;
              // auto s1234_deg = s12_deg;
              #ifndef NDEBUG
                engine.compute2<Operator::coulomb, BraKet::xx_xx, 0>(
                  obs[s1], obs[s2], D_bs[s3], D_bs[s4],sp12,sp34);
              #else
                engine.compute2<Operator::coulomb, BraKet::xx_xx, 0>(
                  obs[s1], obs[s2], D_bs[s3], D_bs[s4],sp12); //,sp34);
              #endif

              const auto* buf_1234 = buf[0];
              if (buf_1234 != nullptr) {
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

            for (decltype(n1) f1 = 0, f1324 = 0; f1 != n1; ++f1) {
              const auto bf1 = f1 + bf1_first;
              for (decltype(n3) f3 = 0; f3 != n3; ++f3) {
                const auto bf3 = f3 + bf3_first;
                for (decltype(n2) f2 = 0; f2 != n2; ++f2) {
                  const auto bf2 = f2 + bf2_first;
                  for (decltype(n4) f4 = 0; f4 != n4; ++f4, ++f1324) {
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
    };

    block_for(ec, F1tmp(), compute_2body_fock_general_lambda);

    // symmetrize the result
    Matrix Gt = 0.5 * (G + G.transpose());
    G = Gt;
    Gt.resize(0,0);
    auto ig2 = std::chrono::high_resolution_clock::now();
    auto igtime =
      std::chrono::duration_cast<std::chrono::duration<double>>((ig2 - ig1)).count();

    if(rank == 0) std::cout << "\nInitial guess: Time to compute guess: " << igtime << " secs" << endl;

    ig1 = std::chrono::high_resolution_clock::now();

    Scheduler{ec}(F1tmp1() = 0).execute();
    eigen_to_tamm_tensor_acc(F1tmp1,G);
    ec.pg().barrier();

    if(rank == 0){
      tamm_to_eigen_tensor(F1tmp1,G);
      Ft += G;
    }
    G.resize(0,0);
    ec.pg().barrier();
    std::vector<TensorType> Fbufv(N*N);
    TensorType *Fbuf = &Fbufv[0]; //.data();
    Eigen::Map<Matrix>(Fbuf,N,N) = Ft;  
    
    // GA_Brdcst(Fbuf,N*N*sizeof(TensorType),0);
    MPI_Bcast(Fbuf,N*N,mpi_type<TensorType>(),0,ec.pg().comm());

    Ft = Eigen::Map<Matrix>(Fbuf,N,N);
    Fbufv.clear(); Fbufv.shrink_to_fit();
    Tensor<TensorType>::deallocate(F1tmp1);

    ig2 = std::chrono::high_resolution_clock::now();
    igtime =
      std::chrono::duration_cast<std::chrono::duration<double>>((ig2 - ig1)).count();
    if(rank == 0) std::cout << "Initial guess: time to compute, broadcast Ft: " << igtime << " secs" << endl;

    D_minbs.resize(0,0);

        ig1 = std::chrono::high_resolution_clock::now();
    // Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> gen_eig_solver(Ft, S);
    // auto eps = gen_eig_solver.eigenvalues();
    // C = gen_eig_solver.eigenvectors();

    // solve F C = e S C by (conditioned) transformation to F' C' = e C',
    // where
    // F' = X.transpose() . F . X; the original C is obtained as C = X . C'

    #ifdef SCALAPACK
      //TODO
      Eigen::SelfAdjointEigenSolver<Matrix> eig_solver(X.transpose() * Ft * X);
      C = X * eig_solver.eigenvectors();

    #elif defined(EIGEN_DIAG)
      Eigen::SelfAdjointEigenSolver<Matrix> eig_solver(X.transpose() * Ft * X);
      C = X * eig_solver.eigenvectors();
    
    #else
      const int64_t Northo = X.cols();
      //EXPECTS( N == Northo );

      if( rank == 0 ) {

      // TODO: Check for linear dep case

      // Take into account row major
      // X -> X**T
      // Ft is [N,N]
      // X**T is [Northo, N]
      C.resize(N,N);
      linalg::blas::gemm( 'N', 'T', N, Northo, N,
                          1., Ft.data(), N, X.data(), Northo, 
                          0., C.data(), N );
      linalg::blas::gemm( 'N', 'N', Northo, Northo, N,
                          1., X.data(), Northo, C.data(), N, 
                          0., Ft.data(), Northo );

      //Ft = X.transpose() * Ft * X;

      std::vector<double> eps(Northo);
      linalg::lapack::syevd( 'V', 'L', Northo, Ft.data(), Northo, eps.data() );
      // Take into account row major
      // X -> X**T
      // C -> C**T = Ft**T * X**T
      // X**T is [Northo, N]
      // Ft is [Northo, Northo]
      // C**T is [Northo, N] 
      C.resize(N, Northo);
      linalg::blas::gemm( 'T', 'N', Northo, N, Northo, 1., Ft.data(), Northo, X.data(), Northo, 0., C.data(), Northo );

      } else C.resize(N, Northo);

      if( world_size > 1 ) 
        MPI_Bcast( C.data(), C.size(), MPI_DOUBLE, 0, ec.pg().comm() );

    #endif


    // compute density, D = C(occ) . C(occ)T
    C_occ = C.leftCols(ndocc);
    D = C_occ * C_occ.transpose();

    ig2 = std::chrono::high_resolution_clock::now();
    igtime =
      std::chrono::duration_cast<std::chrono::duration<double>>((ig2 - ig1)).count();

    if(rank == 0) std::cout << "Initial guess: Time to compute density: " << igtime << " secs" << endl;

}

template<typename TensorType>
void diis(ExecutionContext& ec, TiledIndexSpace& tAO, tamm::Tensor<TensorType> F, 
          Tensor<TensorType> err_mat, int iter, int max_hist, int ndiis,
          std::vector<Tensor<TensorType>>& diis_hist, std::vector<Tensor<TensorType>>& fock_hist) {
  
  using Vector =
      Eigen::Matrix<TensorType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  tamm::Scheduler sch{ec};

  if(ndiis > max_hist) {
    std::vector<TensorType> max_err(diis_hist.size());
    for (size_t i=0; i<diis_hist.size(); i++) {
      max_err[i] = tamm::norm(diis_hist[i]);
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
    auto [mu, nu] = tAO.labels<2>("all");
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

}

template<typename TensorType>
void compute_1body_ints(ExecutionContext& ec, Tensor<TensorType>& tensor1e, 
      std::vector<libint2::Atom>& atoms, libint2::BasisSet& shells, libint2::Operator otype,
      std::vector<size_t>& shell_tile_map, std::vector<Tile>& AO_tiles) {

    using libint2::Atom;
    using libint2::Engine;
    using libint2::Operator;
    using libint2::Shell;
    using libint2::BasisSet;

        Engine engine(otype, max_nprim(shells), max_l(shells), 0);

        // engine.set(otype);

       if(otype == Operator::nuclear) {
        std::vector<std::pair<double, std::array<double, 3>>> q;
        for(const auto& atom : atoms) 
            q.push_back({static_cast<double>(atom.atomic_number),
                            {{atom.x, atom.y, atom.z}}});
        
        engine.set_params(q);
       }

        auto& buf = (engine.results());

    auto compute_1body_ints_lambda = [&](const IndexVector& blockid) {

        auto bi0 = blockid[0];
        auto bi1 = blockid[1];

        const TAMM_SIZE size = tensor1e.block_size(blockid);
        auto block_dims   = tensor1e.block_dims(blockid);
        std::vector<TensorType> dbuf(size);

        auto bd1 = block_dims[1];

        // cout << "blockid: [" << blockid[0] <<"," << blockid[1] << "], dims(0,1) = " <<
        //  block_dims[0] << ", " << block_dims[1] << endl;

        // auto s1 = blockid[0];
        auto s1range_end = shell_tile_map[bi0];
        decltype(s1range_end) s1range_start = 0l;
        if (bi0>0) s1range_start = shell_tile_map[bi0-1]+1;
        
        // cout << "s1-start,end = " << s1range_start << ", " << s1range_end << endl; 
        for (auto s1 = s1range_start; s1 <= s1range_end; ++s1) {
        // auto bf1 = shell2bf[s1]; //shell2bf[s1]; // first basis function in
        // this shell
        auto n1 = shells[s1].size();

        auto s2range_end = shell_tile_map[bi1];
        decltype(s2range_end) s2range_start = 0l;
        if (bi1>0) s2range_start = shell_tile_map[bi1-1]+1;

        // cout << "s2-start,end = " << s2range_start << ", " << s2range_end << endl; 

          // cout << "screend shell pair list = " << s2spl << endl;
          for (auto s2 = s2range_start; s2 <= s2range_end; ++s2) {
          // for (auto s2: obs_shellpair_list[s1]) {
          // auto s2 = blockid[1];
          // if (s2>s1) continue;
          
          if(s2>s1){
            auto s2spl = obs_shellpair_list[s2];
            if(std::find(s2spl.begin(),s2spl.end(),s1) == s2spl.end()) continue;
          }
          else{
            auto s2spl = obs_shellpair_list[s1];
            if(std::find(s2spl.begin(),s2spl.end(),s2) == s2spl.end()) continue;
          }

          // auto bf2 = shell2bf[s2];
          auto n2 = shells[s2].size();

          std::vector<TensorType> tbuf(n1*n2);
          // cout << "s1,s2,n1,n2 = "  << s1 << "," << s2 << 
          //       "," << n1 <<"," << n2 <<endl;

          // compute shell pair; return is the pointer to the buffer
          engine.compute(shells[s1], shells[s2]);
          if (buf[0] == nullptr) continue;          
          // "map" buffer to a const Eigen Matrix, and copy it to the
          // corresponding blocks of the result
          Eigen::Map<const Matrix> buf_mat(buf[0], n1, n2);
          Eigen::Map<Matrix>(&tbuf[0],n1,n2) = buf_mat;
          // tensor1e.put(blockid, tbuf);

          auto curshelloffset_i = 0U;
          auto curshelloffset_j = 0U;
          for(auto x=s1range_start;x<s1;x++) curshelloffset_i += AO_tiles[x];
          for(auto x=s2range_start;x<s2;x++) curshelloffset_j += AO_tiles[x];

          size_t c = 0;
          auto dimi =  curshelloffset_i + AO_tiles[s1];
          auto dimj =  curshelloffset_j + AO_tiles[s2];

          // cout << "curshelloffset_i,curshelloffset_j,dimi,dimj = "  << curshelloffset_i << "," << curshelloffset_j << 
          //       "," << dimi <<"," << dimj <<endl;

          for(size_t i = curshelloffset_i; i < dimi; i++) {
          for(size_t j = curshelloffset_j; j < dimj; j++, c++) {
                  dbuf[i*bd1+j] = tbuf[c];
                }
          }

          // if(s1!=s2){
          //     std::vector<TensorType> ttbuf(n1*n2);
          //     Eigen::Map<Matrix>(ttbuf.data(),n2,n1) = buf_mat.transpose();
          //     // Matrix buf_mat_trans = buf_mat.transpose();
          //     size_t c = 0;
          //     for(size_t j = curshelloffset_j; j < dimj; j++) {
          //       for(size_t i = curshelloffset_i; i < dimi; i++, c++) {
          //             dbuf[j*block_dims[0]+i] = ttbuf[c];
          //       }
          //     }
          // }
              // tensor1e.put({s2,s1}, ttbuf);
          }
        }
        tensor1e.put(blockid,dbuf);
    };

    block_for(ec, tensor1e(), compute_1body_ints_lambda);

}

#endif // TAMM_TESTS_HF_TAMM_COMMON_HPP_
