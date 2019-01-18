
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

#include "hf_common.hpp"

//TODO: UHF,ROHF,diis,3c,dft

void diis(ExecutionContext& ec, TiledIndexSpace& tAO, tamm::Tensor<TensorType> F, 
          Tensor<TensorType> E, int iter, int max_hist,
          int idiis, std::vector<Tensor<TensorType>>& diis_hist,
          std::vector<tamm::Tensor<TensorType>>& fock_hist);


std::tuple<int, int, double, libint2::BasisSet, std::vector<size_t>, Tensor<double>, Tensor<double>, TiledIndexSpace, TiledIndexSpace> 
    hartree_fock(ExecutionContext &exc, const string filename,std::vector<libint2::Atom> atoms, OptionsMap options_map) {
    // Perform the simple HF calculation (Ed) and 2,4-index transform to get the
    // inputs for CCSD
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

  tol_int = std::min(tol_int, 0.01 * conve);

    auto hf_t1 = std::chrono::high_resolution_clock::now();

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
    if(rank==0) cout << "\nNuclear repulsion energy = " << enuc << endl;

    // initializes the Libint integrals library ... now ready to compute
    libint2::initialize(false);

    /*** =========================== ***/
    /*** create basis set            ***/
    /*** =========================== ***/

    // LIBINT_INSTALL_DIR/share/libint/2.4.0-beta.1/basis
    libint2::BasisSet shells(std::string(basis), atoms);
    // auto shells = make_sto3g_basis(atoms);

    // compute OBS non-negligible shell-pair list
    {
      std::tie(obs_shellpair_list, obs_shellpair_data) = compute_shellpairs(shells);
      size_t nsp = 0;
      for (auto& sp : obs_shellpair_list) {
        nsp += sp.second.size();
      }
      if(rank==0) std::cout << "# of {all,non-negligible} shell-pairs = {"
                << shells.size() * (shells.size() + 1) / 2 << "," << nsp << "}"
                << std::endl;
    }

    const size_t N = nbasis(shells);
    size_t nao = N;

    if(rank==0) cout << "\nNumber of basis functions: " << N << endl;

    tamm::Tile tile_size = scf_options.AO_tilesize; 
  
    //heuristic to set tilesize to atleast 5% of nbf
    // if(tile_size < N*0.05) {
    //   tile_size = std::ceil(N*0.05);
    //   if(rank == 0) cout << "***** Reset tilesize to nbf*5% = " << tile_size << endl;
    // }
    
    IndexSpace AO{range(0, N)};
    std::vector<unsigned int> AO_tiles;
    for(auto s : shells) AO_tiles.push_back(s.size());
    if(rank==0) 
      cout << "Number of AO tiles = " << AO_tiles.size() << endl;

    tamm::Tile est_ts = 0;
    std::vector<tamm::Tile> AO_opttiles;
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

    /*** =========================== ***/
    /*** compute 1-e integrals       ***/
    /*** =========================== ***/

    Engine engine(Operator::overlap, max_nprim(shells), max_l(shells), 0);
    auto& buf = unconst_cast(engine.results());

    Tensor<TensorType> tensor1e;
    auto compute_1body_ints = [&](const IndexVector& blockid) {

        auto bi0 = blockid[0];
        auto bi1 = blockid[1];

        const TAMM_SIZE size = tensor1e.block_size(blockid);
        auto block_dims   = tensor1e.block_dims(blockid);
        std::vector<TensorType> dbuf(size);

        auto bd1 = block_dims[1];

        // cout << "blockid: [" << blockid[0] <<"," << blockid[1] << "], dims(0,1) = " <<
        //  block_dims[0] << ", " << block_dims[1] << endl;

        // auto s1 = blockid[0];
        auto s1range_start = 0l;
        auto s1range_end = shell_tile_map[bi0];
        if (bi0>0) s1range_start = shell_tile_map[bi0-1]+1;
        
        // cout << "s1-start,end = " << s1range_start << ", " << s1range_end << endl; 
        for (auto s1 = s1range_start; s1 <= s1range_end; ++s1) {
        // auto bf1 = shell2bf[s1]; //shell2bf[s1]; // first basis function in
        // this shell
          auto n1 = shells[s1].size();

        auto s2range_start = 0l;
        auto s2range_end = shell_tile_map[bi1];
        if (bi1>0) s2range_start = shell_tile_map[bi1-1]+1;

        // cout << "s2-start,end = " << s2range_start << ", " << s2range_end << endl; 

          // cout << "screend shell pair list = " << s2spl << endl;
          for (size_t s2 = s2range_start; s2 <= s2range_end; ++s2) {
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
          // "map" buffer to a const Eigen Matrix, and copy it to the
          // corresponding blocks of the result
          Eigen::Map<const Matrix> buf_mat(buf[0], n1, n2);
          Eigen::Map<Matrix>(tbuf.data(),n1,n2) = buf_mat;
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

    TiledIndexSpace tAO{AO, AO_opttiles};
    TiledIndexSpace tAOt{AO, AO_tiles};
    auto [mu, nu, ku] = tAO.labels<3>("all");
    auto [mup, nup, kup] = tAOt.labels<3>("all");

    Tensor<TensorType> H1{tAO, tAO};

    Tensor<TensorType> S1{tAO, tAO};
    Tensor<TensorType> T1{tAO, tAO};
    Tensor<TensorType> V1{tAO, tAO};
    Tensor<TensorType>::allocate(ec, H1, S1, T1, V1);

    Matrix H, S; 
    H.setZero(N, N);
    S.setZero(N, N);

    auto hf_t2 = std::chrono::high_resolution_clock::now();

    double hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
    if(rank == 0) std::cout << "\nTime for initial setup: " << hf_time << " secs\n";

    // auto T = compute_1body_ints(shells, Operator::kinetic);
    // auto S = compute_1body_ints(shells, Operator::overlap);
    // Matrix V = compute_1body_ints(shells, Operator::nuclear, atoms);
    // Matrix H = T;

    hf_t1 = std::chrono::high_resolution_clock::now();

    // Scheduler{ec}(S1(mu, nu) = SL1(mu, nu)).execute();
    tensor1e = S1;
    block_for(*ec, S1(), compute_1body_ints);

    engine.set(Operator::kinetic);
    buf = engine.results();
    
    tensor1e = T1;
    block_for(*ec, T1(), compute_1body_ints);

    engine.set(Operator::nuclear);
    std::vector<std::pair<double, std::array<double, 3>>> q;
    for(const auto& atom : atoms) {
        q.push_back({static_cast<double>(atom.atomic_number),
                        {{atom.x, atom.y, atom.z}}});
    }
    engine.set_params(q);
    buf = engine.results();
    tensor1e = V1;
    block_for(*ec, V1(), compute_1body_ints);
    
    hf_t2 = std::chrono::high_resolution_clock::now();
    hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
    if(rank == 0) std::cout << "\nTime for computing 1-e integrals T, V, S: " << hf_time << " secs\n";

    hf_t1 = std::chrono::high_resolution_clock::now();
    // Core Hamiltonian = T + V
    Scheduler{*ec}
      (H1(mu, nu) = T1(mu, nu))
      (H1(mu, nu) += V1(mu, nu)).execute();
    hf_t2 = std::chrono::high_resolution_clock::now();
    hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
    if(rank == 0) std::cout << "\nTime for computing Core Hamiltonian H=T+V: " << hf_time << " secs\n";

    hf_t1 = std::chrono::high_resolution_clock::now();

    if(rank == 0)
      tamm_to_eigen_tensor(H1, H);
    GA_Sync();
    std::vector<TensorType> Hbufv(N*N);
    TensorType *Hbuf = Hbufv.data();
    Eigen::Map<Matrix>(Hbuf,N,N) = H;  
    GA_Brdcst(Hbuf,N*N*sizeof(TensorType),0);
    H = Eigen::Map<Matrix>(Hbuf,N,N);
    Hbufv.clear();

    hf_t2 = std::chrono::high_resolution_clock::now();
    hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
   
    if(rank == 0) std::cout << "\nTime for tamm to eigen - H1-H: " << hf_time << " secs\n";

    hf_t1 = std::chrono::high_resolution_clock::now();
    if(rank == 0)
      tamm_to_eigen_tensor(S1, S);
    GA_Sync();
    std::vector<TensorType> Sbufv(N*N);
    TensorType *Sbuf = Sbufv.data();
    Eigen::Map<Matrix>(Sbuf,N,N) = S;  
    GA_Brdcst(Sbuf,N*N*sizeof(TensorType),0);
    S = Eigen::Map<Matrix>(Sbuf,N,N);
    Sbufv.clear();
    hf_t2 = std::chrono::high_resolution_clock::now();
    hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
    GA_Sync();
    if(rank == 0) std::cout << "\nTime for tamm to eigen - S1-S: " << hf_time << " secs\n";

    // eigen_to_tamm_tensor(H1, H);

    Tensor<TensorType>::deallocate(T1, V1);

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
    double S_condition_number_threshold = 1.0 / scf_options.tol_lindep;
        //1.0 / std::numeric_limits<double>::epsilon();
    std::tie(X, Xinv, XtX_condition_number) =
        conditioning_orthogonalizer(S, S_condition_number_threshold);

    // TODO Redeclare TAMM S1 with new dims?
    hf_t2 = std::chrono::high_resolution_clock::now();
    hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

    //TODO: not used ?
    Xinv.resize(0,0);

    if(rank == 0) std::cout << "Time for computing orthogonalizer: " << hf_time << " secs\n\n";

    // pre-compute data for Schwarz bounds
    auto SchwarzK = compute_schwarz_ints<>(shells);

    hf_t1 = std::chrono::high_resolution_clock::now();

  const auto use_hcore_guess = false;  // use core Hamiltonian eigenstates to guess density?
  // set to true to match the result of versions 0, 1, and 2 of the code
  // HOWEVER !!! even for medium-size molecules hcore will usually fail !!!
  // thus set to false to use Superposition-Of-Atomic-Densities (SOAD) guess
  Matrix D;
  Matrix C;
  Matrix F;

  if (use_hcore_guess) { // hcore guess

    // solve H C = e S C
    Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> gen_eig_solver(H, S);
    auto eps = gen_eig_solver.eigenvalues();
    C = gen_eig_solver.eigenvectors();

    // compute density, D = C(occ) . C(occ)T
    auto C_occ = C.leftCols(ndocc);
    D = C_occ * C_occ.transpose();

    F = H;

    // F += compute_2body_fock_general(
    //     shells, D, shells, true /* SOAD_D_is_shelldiagonal */,
    //     std::numeric_limits<double>::epsilon()  // this is cheap, no reason
    //                                             // to be cheaper
    //     );

    F +=    compute_2body_fock(shells, D, 1e-8, SchwarzK);

    Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> gen_eig_solver1(F, S);
    eps = gen_eig_solver1.eigenvalues();
    C = gen_eig_solver1.eigenvectors();
    // compute density, D = C(occ) . C(occ)T
    C_occ = C.leftCols(ndocc);
    D = C_occ * C_occ.transpose();

  } 
  
  else if (restart){
    std::vector<TensorType> Dbuf(N*N);
    TensorType *Dbufp = Dbuf.data();
    if(rank==0) 
    {
      cout << "Reading orbitals from file... ";
      std::string orbitalsfile = getfilename(filename) +
        "." + scf_options.basis + ".orbitals";

      std::vector<TensorType> Cbuf(N*N);
      TensorType *Hbuf = Cbuf.data();
      std::ifstream in(orbitalsfile, std::ios::in | std::ios::binary);
      in.read((char *) Hbuf, sizeof(TensorType)*N*N);
      C = Eigen::Map<Matrix>(Hbuf,N,N);

      if(rank==0) cout << "done\n";

      auto C_occ = C.leftCols(ndocc);
      D = C_occ * C_occ.transpose();
      Eigen::Map<Matrix>(Dbufp,N,N)=D;
    }
    GA_Brdcst(Dbufp,N*N*sizeof(TensorType),0);
    D = Eigen::Map<Matrix>(Dbufp,N,N);
    Dbuf.clear();
    GA_Sync();

  }
  
  else {  // SOAD as the guess density
    
    auto D_minbs = compute_soad(atoms);  // compute guess in minimal basis
    BasisSet minbs("STO-3G", atoms);

    //std::tie(minbs_shellpair_list, minbs_shellpair_data) = compute_shellpairs(minbs);

    if(rank == 0) std::cout << 
    "\nProjecting minimal basis SOAD onto basis set specified (" << basis << ")\n";
    
    auto ig1 = std::chrono::high_resolution_clock::now();

    auto Ft = H;

    // Ft += compute_2body_fock_general(
    //     shells, D_minbs, minbs, true /* SOAD_D_is_shelldiagonal */,
    //     std::numeric_limits<double>::epsilon()  // this is cheap, no reason
    //                                             // to be cheaper
    //     );

#if 1
  double precision = std::numeric_limits<double>::epsilon();
  bool D_is_shelldiagonal = true;
  const libint2::BasisSet& obs = shells;
  const libint2::BasisSet& D_bs = minbs;
  // const Matrix& D = D_minbs; 

  Matrix G = Matrix::Zero(N,N);
  Tensor<TensorType> F1tmp{tAOt, tAOt}; //not allocated
  Tensor<TensorType> F1tmp1{tAO, tAO};
  Tensor<TensorType>::allocate(ec, F1tmp1);

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

        for (auto s3 = 0; s3 < D_bs.size(); ++s3) {
          auto bf3_first = shell2bf_D[s3];
          auto n3 = D_bs[s3].size();

          auto s4_begin = D_is_shelldiagonal ? s3 : 0;
          auto s4_fence = D_is_shelldiagonal ? s3 + 1 : D_bs.size();

          //auto sp34_iter = minbs_shellpair_data.at(s3).begin();

          for (auto s4 = s4_begin; s4 != s4_fence; ++s4) {

            // auto s4spl = minbs_shellpair_list[s3];
            // auto s4_itr = std::find(s4spl.begin(),s4spl.end(),s4);
            // if(s4_itr == s4spl.end()) continue;
            // auto s4_pos = std::distance(s4spl.begin(),s4_itr);

            // std::advance(sp34_iter,s4_pos);
            // const auto* sp34 = sp34_iter->get();

            auto bf4_first = shell2bf_D[s4];
            auto n4 = D_bs[s4].size();

            // compute the permutational degeneracy (i.e. # of equivalents) of
            // the given shell set
            auto s12_deg = (s1 == s2) ? 1.0 : 2.0;

            if (s3 >= s4) {
              auto s34_deg = (s3 == s4) ? 1.0 : 2.0;
              auto s1234_deg = s12_deg * s34_deg;
              // auto s1234_deg = s12_deg;
              engine.compute2<Operator::coulomb, BraKet::xx_xx, 0>(
                  obs[s1], obs[s2], D_bs[s3], D_bs[s4],sp12); //,sp34);
              const auto* buf_1234 = buf[0];
              if (buf_1234 != nullptr) {
                for (auto f1 = 0, f1234 = 0; f1 != n1; ++f1) {
                  const auto bf1 = f1 + bf1_first;
                  for (auto f2 = 0; f2 != n2; ++f2) {
                    const auto bf2 = f2 + bf2_first;
                    for (auto f3 = 0; f3 != n3; ++f3) {
                      const auto bf3 = f3 + bf3_first;
                      for (auto f4 = 0; f4 != n4; ++f4, ++f1234) {
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

            for (auto f1 = 0, f1324 = 0; f1 != n1; ++f1) {
              const auto bf1 = f1 + bf1_first;
              for (auto f3 = 0; f3 != n3; ++f3) {
                const auto bf3 = f3 + bf3_first;
                for (auto f2 = 0; f2 != n2; ++f2) {
                  const auto bf2 = f2 + bf2_first;
                  for (auto f4 = 0; f4 != n4; ++f4, ++f1324) {
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

    block_for(*ec, F1tmp(), compute_2body_fock_general_lambda);
    // symmetrize the result and return
    Matrix Gt = 0.5 * (G + G.transpose());
    G = Gt;
    Gt.resize(0,0);
    auto ig2 = std::chrono::high_resolution_clock::now();
    auto igtime =
      std::chrono::duration_cast<std::chrono::duration<double>>((ig2 - ig1)).count();

    if(rank == 0) std::cout << "\nInitial guess: Time to compute guess: " << igtime << " secs\n";

    ig1 = std::chrono::high_resolution_clock::now();

    Scheduler{*ec}(F1tmp1() = 0).execute();
    eigen_to_tamm_tensor_acc(F1tmp1,G);
    GA_Sync();

    if(rank == 0){
      tamm_to_eigen_tensor(F1tmp1,G);
      Ft += G;
    }
    G.resize(0,0);
    GA_Sync();
    std::vector<TensorType> Fbufv(N*N);
    TensorType *Fbuf = Fbufv.data();
    Eigen::Map<Matrix>(Fbuf,N,N) = Ft;  
    GA_Brdcst(Fbuf,N*N*sizeof(TensorType),0);
    Ft = Eigen::Map<Matrix>(Fbuf,N,N);
    Fbufv.clear();
    Tensor<TensorType>::deallocate(F1tmp1);

    ig2 = std::chrono::high_resolution_clock::now();
    igtime =
      std::chrono::duration_cast<std::chrono::duration<double>>((ig2 - ig1)).count();
    if(rank == 0) std::cout << "Initial guess: time to compute, broadcast Ft: " << igtime << " secs\n";

#endif

    D_minbs.resize(0,0);

    ig1 = std::chrono::high_resolution_clock::now();
    // Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> gen_eig_solver(Ft, S);
    // auto eps = gen_eig_solver.eigenvalues();
    // C = gen_eig_solver.eigenvectors();

    // solve F C = e S C by (conditioned) transformation to F' C' = e C',
    // where
    // F' = X.transpose() . F . X; the original C is obtained as C = X . C'
    Eigen::SelfAdjointEigenSolver<Matrix> eig_solver(X.transpose() * Ft * X);
    C = X * eig_solver.eigenvectors();

    // compute density, D = C(occ) . C(occ)T
    auto C_occ = C.leftCols(ndocc);
    D = C_occ * C_occ.transpose();

    ig2 = std::chrono::high_resolution_clock::now();
    igtime =
      std::chrono::duration_cast<std::chrono::duration<double>>((ig2 - ig1)).count();

    if(rank == 0) std::cout << "Initial guess: Time to compute density: " << igtime << " secs\n";

  }

    H.resize(0,0);
    hf_t2 = std::chrono::high_resolution_clock::now();
    hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

    GA_Sync();
    if(rank == 0) std::cout << "Total Time to compute initial guess: " << hf_time << " secs\n";


    hf_t1 = std::chrono::high_resolution_clock::now();
    /*** =========================== ***/
    /*** main iterative loop         ***/
    /*** =========================== ***/

    // bool simple_convergence = false;
    // double alpha = 0.5;
    auto iter          = 0;
    auto rmsd          = 1.0;
    auto ediff         = 0.0;
    auto ehf           = 0.0;
    auto is_conv       = true;
    //  Matrix C;
    //  Matrix F;
    
    // double alpha = 0.75;
    // Matrix F_old;

    int idiis                     = 0;
    std::vector<tamm::Tensor<TensorType>> diis_hist;
    std::vector<tamm::Tensor<TensorType>> fock_hist;

    Tensor<TensorType> ehf_tmp{tAO, tAO};
    Tensor<TensorType> ehf_tamm{}, rmsd_tamm{};

    Tensor<TensorType> F1{tAO, tAO};
    Tensor<TensorType> F1tmp1{tAO, tAO};
    Tensor<TensorType> F1tmp{tAOt, tAOt}; //not allocated
    Tensor<TensorType>::allocate(ec, F1, F1tmp1, ehf_tmp, ehf_tamm, rmsd_tamm);

    //Tensor<TensorType> Sm12_tamm{tAO, tAO}; 
    //Tensor<TensorType> Sp12_tamm{tAO, tAO};
    Tensor<TensorType> D_tamm{tAO, tAO};
    Tensor<TensorType> D_diff{tAO, tAO};
    Tensor<TensorType> D_last_tamm{tAO, tAO};
    Tensor<TensorType>::allocate(ec, D_tamm, D_diff, D_last_tamm);

    // FSm12,Sp12D,SpFS
    Tensor<TensorType> FD_tamm{tAO, tAO}; 
    //Tensor<TensorType> Sp12D_tamm{tAO, tAO};
    Tensor<TensorType> FDS_tamm{tAO, tAO};
    // Tensor<TensorType> err_mat_tamm{tAO, tAO};

    Tensor<TensorType>::allocate(ec, FD_tamm, FDS_tamm);//,err_mat_tamm);

    hf_t2 = std::chrono::high_resolution_clock::now();
    hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
    //GA_Sync();
    if(rank == 0 && debug) std::cout << "\nTime to setup tensors for iterative loop: " << hf_time << " secs\n";

    eigen_to_tamm_tensor(D_tamm,D);
    // Matrix err_mat = Matrix::Zero(N,N);

    // hf_t1 = std::chrono::high_resolution_clock::now();
    // // S^1/2
    // Matrix Sp12 = Xinv; //S.sqrt();
    // S.resize(0,0);
    // // S^-1/2
    // Matrix Sm12 = X; //Sp12.inverse();
    // hf_t2 = std::chrono::high_resolution_clock::now();
    // hf_time =
    //   std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
    // if(rank == 0) std::cout << "\nTime taken to compute S^1/2, S^-1/2: " << hf_time << " secs\n";

    // eigen_to_tamm_tensor(Sm12_tamm,Sm12);
    // eigen_to_tamm_tensor(Sp12_tamm,Sp12);   
    // Sp12.resize(0,0);
    // Sm12.resize(0,0);

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
    double precision = tol_int;
    const libint2::BasisSet& obs = shells;
    // assert(N==obs.nbf());
    Matrix G = Matrix::Zero(N,N);
    const auto do_schwarz_screen = SchwarzK.cols() != 0 && SchwarzK.rows() != 0;
    auto fock_precision = precision;
    // engine precision controls primitive truncation, assume worst-case scenario
    // (all primitive combinations add up constructively)
    auto max_nprim = obs.max_nprim();
    auto max_nprim4 = max_nprim * max_nprim * max_nprim * max_nprim;
    auto shell2bf = obs.shell2bf();
    // const auto nshells = obs.size();

    Scheduler{*ec}
        (D_diff(mu,nu) = D_tamm(mu,nu)).execute();

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

        auto do_t1 = std::chrono::high_resolution_clock::now();
        
        // const auto precision_F = std::min(
        // std::min(1e-3 / XtX_condition_number, 1e-7),
        // std::max(rmsd / 1e4, std::numeric_limits<double>::epsilon()));

        // Matrix Ftmp = compute_2body_fock(shells, D, tol_int, SchwarzK);
        // eigen_to_tamm_tensor(F1tmp, Ftmp);

        G.setZero(N,N);
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

        for (auto s3 = 0; s3 <= s1; ++s3) {
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
            for (auto f1 = 0, f1234 = 0; f1 != n1; ++f1) {
              const auto bf1 = f1 + bf1_first;
              for (auto f2 = 0; f2 != n2; ++f2) {
                const auto bf2 = f2 + bf2_first;
                for (auto f3 = 0; f3 != n3; ++f3) {
                  const auto bf3 = f3 + bf3_first;
                  for (auto f4 = 0; f4 != n4; ++f4, ++f1234) {
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

    block_for(*ec, F1tmp(), comp_2bf_lambda);
    Matrix Gt = 0.5*(G + G.transpose());
    G = Gt;
    Gt.resize(0,0);
    auto do_t2 = std::chrono::high_resolution_clock::now();
    auto do_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((do_t2 - do_t1)).count();

    if(rank == 0 && debug) std::cout << "2BF:" << do_time << "s, ";
    // GA_Sync();

    do_t1 = std::chrono::high_resolution_clock::now();
    eigen_to_tamm_tensor_acc(F1tmp1,G);
    do_t2 = std::chrono::high_resolution_clock::now();
    do_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((do_t2 - do_t1)).count();

    if(rank == 0 && debug) std::cout << "E2T-ACC-G-F1:" << do_time << "s, ";
#endif

        do_t1 = std::chrono::high_resolution_clock::now();
        // F += Ftmp;
        Scheduler{*ec}(F1(mu, nu) = H1(mu, nu))
                     (F1(mu, nu) += F1tmp1(mu, nu)).execute();

        do_t2 = std::chrono::high_resolution_clock::now();
        do_time =
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
        Tensor<TensorType>::allocate(ec, err_mat_tamm);

        do_t1 = std::chrono::high_resolution_clock::now();
        
        // Scheduler{*ec}(FD_tamm(mu,nu) = F1(mu,ku) * Sm12_tamm(ku,nu))
        // (Sp12D_tamm(mu,nu) = Sp12_tamm(mu,ku) * D_last_tamm(ku,nu))
        // (FDS_tamm(mu,nu)  = Sp12D_tamm(mu,ku) * FD_tamm(ku,nu))

        Scheduler{*ec}(FD_tamm(mu,nu) = F1(mu,ku) * D_last_tamm(ku,nu))
        (FDS_tamm(mu,nu)  = FD_tamm(mu,ku) * S1(ku,nu))
    
        //FDS-SDF
        (err_mat_tamm(mu,nu) = FDS_tamm(mu,nu))
        (err_mat_tamm(mu,nu) -= FDS_tamm(nu,mu)).execute();

        do_t2 = std::chrono::high_resolution_clock::now();
        do_time =
        std::chrono::duration_cast<std::chrono::duration<double>>((do_t2 - do_t1)).count();

        if(rank == 0 && debug) std::cout << "FDS-SDF:" << do_time << "s, ";        

        GA_Sync();
        //tamm_to_eigen_tensor(err_mat_tamm,err_mat);
        //tamm_to_eigen_tensor(F1,F);

        do_t1 = std::chrono::high_resolution_clock::now();

        if(iter > 1) {
            ++idiis;
            diis(*ec, tAO, F1, err_mat_tamm, iter, max_hist, idiis,
                diis_hist, fock_hist);
        }
    
        do_t2 = std::chrono::high_resolution_clock::now();
        do_time =
        std::chrono::duration_cast<std::chrono::duration<double>>((do_t2 - do_t1)).count();

        if(rank == 0 && debug) std::cout << "diis:" << do_time << "s, ";    
        tamm_to_eigen_tensor(F1,F);

        do_t1 = std::chrono::high_resolution_clock::now();
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
        Eigen::SelfAdjointEigenSolver<Matrix> eig_solver(X.transpose() * F *
                                                        X);
        //eps = eig_solver.eigenvalues();
        C = X * eig_solver.eigenvectors();

        // compute density, D = C(occ) . C(occ)T
        auto C_occ = C.leftCols(ndocc);
        D          = C_occ * C_occ.transpose();

        do_t2 = std::chrono::high_resolution_clock::now();
        do_time =
        std::chrono::duration_cast<std::chrono::duration<double>>((do_t2 - do_t1)).count();

        if(rank == 0 && debug) std::cout << "eigen_solve:" << do_time << "s, ";    

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
        Scheduler{*ec}
           (ehf_tmp(mu,nu) = H1(mu,nu))
           (ehf_tmp(mu,nu) += F1(mu,nu))
           (ehf_tamm() = D_tamm() * ehf_tmp()).execute();

        ehf = get_scalar(ehf_tamm);

        // compute difference with last iteration
        ediff = ehf - ehf_last;
        // rmsd  = (D - D_last).norm();
        Scheduler{*ec}(D_diff()=D_tamm())
                      (D_diff()-=D_last_tamm()).execute();
        rmsd = norm(*ec, D_diff());

        do_t2 = std::chrono::high_resolution_clock::now();
        do_time =
        std::chrono::duration_cast<std::chrono::duration<double>>((do_t2 - do_t1)).count();

        if(rank == 0 && debug) std::cout << "HF-Energy:" << do_time << "s\n";    

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

    for (auto x: diis_hist) Tensor<TensorType>::deallocate(x);
    for (auto x: fock_hist) Tensor<TensorType>::deallocate(x);

    Tensor<TensorType>::deallocate(H1, S1, D_tamm, ehf_tmp, 
                    ehf_tamm, rmsd_tamm); //F1
    Tensor<TensorType>::deallocate(F1tmp1, //Sm12_tamm, Sp12_tamm,
    D_last_tamm, D_diff, FD_tamm, FDS_tamm);//Sp12D_tamm,err_mat_tamm);

    if(rank==0 && !restart) {
     cout << "writing orbitals to file... ";
     writeC(C,filename,options_map);
     cout << "done.\n";
    }

    Tensor<TensorType> C_tamm{tAO,tAO};
    Tensor<TensorType>::allocate(ec,C_tamm);
    eigen_to_tamm_tensor(C_tamm,C);
    GA_Sync();

    //F1, C are not deallocated.
    return std::make_tuple(ndocc, nao, ehf + enuc, shells, shell_tile_map, C_tamm, F1, tAO, tAOt);
}

void diis(ExecutionContext& ec, TiledIndexSpace& tAO, tamm::Tensor<TensorType> F, Tensor<TensorType> err_mat, int iter, int max_hist, int ndiis,
          std::vector<Tensor<TensorType>>& diis_hist, std::vector<tamm::Tensor<TensorType>>& fock_hist) {
    using Vector =
      Eigen::Matrix<TensorType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  tamm::Scheduler sch{ec};

  if(ndiis > max_hist) {
    std::vector<TensorType> max_err(diis_hist.size());
    for (auto i=0; i<diis_hist.size(); i++) {
      max_err[i] = tamm::norm(ec, diis_hist[i]());
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
    auto [mu, nu, ku] = tAO.labels<3>("all");
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

#endif // TAMM_TESTS_HF_TAMM_HPP_
