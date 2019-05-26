
#ifndef TAMM_CD_SVD_GA_HPP_
#define TAMM_CD_SVD_GA_HPP_

#include "HF/hartree_fock_tamm.hpp"
#include "tamm/eigen_utils.hpp"
#include "tamm/tamm.hpp"

#include CBLAS_HEADER
#include LAPACKE_HEADER

using namespace tamm;
using TensorType = double;
using TAMM_GA_SIZE = int64_t;

bool cd_debug = false;

// From integer type to integer type
template <typename from>
constexpr typename std::enable_if<std::is_integral<from>::value && std::is_integral<int64_t>::value, int64_t>::type
cd_ncast(const from& value)
{
    return static_cast<int64_t>(value & (static_cast<typename std::make_unsigned<from>::type>(-1)));
}

#if 0
  std::tuple<Index, Index, size_t,size_t> get_shell_ids
    (const std::vector<size_t> &shell_tile_map, const IndexVector& AO_tiles,
     IndexVector& maxblockid, size_t bfu, size_t bfv) {
    
    auto mi0 = maxblockid[0];
    auto mi1 = maxblockid[1];
    auto s1rs = 0l;
    auto s1re = shell_tile_map[mi0];
    if (mi0>0) s1rs = shell_tile_map[mi0-1]+1;
    auto s2rs = 0l;
    auto s2re = shell_tile_map[mi1];
    if (mi1>0) s2rs = shell_tile_map[mi1-1]+1;

    Index s1 = 0;
    Index s2 = 0;
    auto curshelloffset_i = 0U;
    auto curshelloffset_j = 0U;

    for (auto x1 = s1rs; x1 <= s1re; ++x1) {
          if(bfu >= curshelloffset_i && bfu < curshelloffset_i+AO_tiles[x1]){
            s1 = x1; break;
          }
          curshelloffset_i += AO_tiles[x1];
    }

    for (auto x1 = s2rs; x1 <= s2re; ++x1) {
          if(bfv >= curshelloffset_j && bfv < curshelloffset_j+AO_tiles[x1]){
            s2 = x1; break;
          }
          curshelloffset_j += AO_tiles[x1];
    }

    return std::make_tuple(s1,s2,curshelloffset_i,curshelloffset_j);
  }
#endif

  int64_t ac_fetch_add(int ga_ac, int64_t index, int64_t amount) {
    auto ret = NGA_Read_inc64(ga_ac, &index, amount);
    return ret;
  }

Tensor<TensorType> cd_svd_ga(ExecutionContext& ec, TiledIndexSpace& tMO, TiledIndexSpace& tAO,
  const TAMM_GA_SIZE ndocc, const TAMM_GA_SIZE nao, const TAMM_GA_SIZE freeze_core,
  const TAMM_GA_SIZE freeze_virtual, Tensor<TensorType> C_AO, Tensor<TensorType> F_AO,
  Tensor<TensorType> F_MO, TAMM_SIZE& chol_count, const TAMM_GA_SIZE max_cvecs, double diagtol,
  libint2::BasisSet& shells, std::vector<size_t>& shell_tile_map) {

    using libint2::Atom;
    using libint2::Shell;
    using libint2::Engine;
    using libint2::Operator;

    auto rank = ec.pg().rank();

    // auto iptilesize = tAO.input_tile_sizes()[0];

    std::vector<unsigned int> AO_tiles;
    for(auto s : shells) AO_tiles.push_back(s.size());

    // auto [mu, nu, ku] = tAO.labels<3>("all");
    // auto [pmo, rmo] = tMO.labels<2>("all");


    if(rank == 0){
      cout << "\n-----------------------------------------------------\n";
      cout << "Begin Cholesky Decomposition ... " << endl;
      cout << "\n#AOs, #electrons = " << nao << " , " << ndocc << endl;
    }

    auto ov_alpha_freeze = ndocc - freeze_core;
    auto ov_beta_freeze  = nao - ndocc - freeze_virtual;
    
    // const auto v2dim  = 2 * nao - 2 * freeze_core - 2 * freeze_virtual;
    // const int n_alpha = ov_alpha_freeze;
    // const int n_beta  = ov_beta_freeze;
    // auto ov_alpha     = ndocc;
    // TAMM_GA_SIZE ov_beta{nao - ov_alpha};

    // 2-index transform

    auto hf_t1 = std::chrono::high_resolution_clock::now();

    //TODO: CTiled_tamm not needed
    // Tensor<TensorType> CTiled_tamm{tAO,tMO};
    // Tensor<TensorType>::allocate(&ec, CTiled_tamm);

    auto N = 2 * nao - 2 * freeze_core - 2 * freeze_virtual;
    Matrix CTiled(nao, N);

    if(rank == 0){

      Matrix C(nao,nao);
      tamm_to_eigen_tensor(C_AO,C);
      // replicate horizontally
      Matrix C_2N(nao, 2 * nao);
      C_2N << C, C;
      C.resize(0,0);
      // cout << "\n\t C_2N Matrix:\n";
      // cout << C_2N << endl;

      Matrix C_noa = C_2N.block(0, freeze_core, nao, ov_alpha_freeze);
      //  cout << "\n\t C occupied alpha:\n";
      //  cout << C_noa << endl;

      Matrix C_nva = C_2N.block(0, ndocc, nao, ov_beta_freeze);
      //  cout << "\n\t C virtual alpha:\n";
      //  cout << C_nva << endl;

      Matrix C_nob = C_2N.block(0, nao + freeze_core, nao, ov_alpha_freeze);
      //  cout << "\n\t C occupied beta:\n";
      //  cout << C_nob << endl;

      Matrix C_nvb = C_2N.block(0, ndocc + nao, nao, ov_beta_freeze);
      //  cout << "\n\t C virtual beta:\n";
      //  cout << C_nvb << endl;

      // For now C_noa = C_nob and C_nva = C_nvb
      // Matrix CTiled(nao, 2 * nao - 2 * freeze_core - 2 * freeze_virtual);
      CTiled << C_noa, C_nob, C_nva, C_nvb;

      //  cout << "\n\t CTiled Matrix = [C_noa C_nob C_nva C_nvb]:\n";
      //  cout << CTiled << endl;

      Matrix F(nao,nao);
      tamm_to_eigen_tensor(F_AO,F);
      F = CTiled.transpose() * (F * CTiled); //F is resized to 2N*2N
      
      // eigen_to_tamm_tensor(CTiled_tamm,CTiled);
      eigen_to_tamm_tensor(F_MO,F);

      // cout << "F_MO dims = " << F.rows() << "," << F.cols()  << endl;
    }

    std::vector<TensorType> CTiledBuf(nao*N);
    TensorType *k_movecs_sorted = &CTiledBuf[0];
    Eigen::Map<Matrix>(k_movecs_sorted,nao,N) = CTiled;  
    GA_Brdcst(k_movecs_sorted,nao*N*sizeof(TensorType),0);
    CTiled.resize(0,0);

  auto hf_t2 = std::chrono::high_resolution_clock::now();
  auto hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
  if(rank == 0) std::cout << "\nTime taken for 2-index transform: " << hf_time << " secs\n";
  GA_Sync();
  
  hf_t1 = std::chrono::high_resolution_clock::now();

  //rank = iproc
  int64_t iproc = rank.value();
  int64_t ndim = 3;
  // int64_t pref = static_cast<int64_t>(max_cvecs);
  auto nbf = nao;
  tamm::Tile count = 0; //Step A. Initialize chol vector count

  int g_chol_mo = 0;
  int64_t cd_nranks = std::abs(std::log10(diagtol)) * nbf; // max cores
  auto nnodes = GA_Cluster_nnodes();
  auto ppn = GA_Cluster_nprocs(0);
  int cd_nnodes = cd_nranks/ppn;
  if(cd_nranks%ppn>0 || cd_nnodes==0) cd_nnodes++;
  if(cd_nnodes > nnodes) cd_nnodes = nnodes;
  cd_nranks = cd_nnodes * ppn;
  if(rank == 0)  cout << "Total # of mpi ranks used for Cholesky decomposition: " << cd_nranks 
       << "\n  --> Number of nodes, mpi ranks per node: " << cd_nnodes << ", " << ppn << endl;


  int64_t dimsmo[3];
  int64_t chnkmo[3];
  int nblockmo32[GA_MAX_DIM]; 
  int64_t nblockmo[GA_MAX_DIM]; 
  int64_t size_map;
  std::vector<int64_t> k_map;
  // auto eltype   = tensor_element_type<TensorType>();
  int ga_eltype = C_DBL; //TODO: MemoryManagerGA::to_ga_eltype(eltype);

  auto create_map = [&] (auto& dims, auto& nblock) {
    std::vector<int64_t> k_map(size_map);
    auto mi=0;
    for (auto count_dim=0;count_dim<2;count_dim++){
      auto size_blk = dims[count_dim]/nblock[count_dim];
      // cout << "cdim,size_blk,nb(cd) = " << count_dim << ", " << size_blk << "," << nblock[count_dim] << endl;
      for (auto i=0;i<nblock[count_dim];i++){
        k_map[mi] = size_blk*i;
        mi++;
      }
    }
    k_map[mi] = 0;
    return k_map;
  };

  const bool throttle_cd = GA_Nnodes() > cd_nranks;
  int ga_pg_default = GA_Pgroup_get_default();
  int ga_pg = ga_pg_default;

  if(iproc < cd_nranks) { //throttle  

    if(throttle_cd){
      int ranks[cd_nranks];
      for (int i = 0; i < cd_nranks; i++) ranks[i] = i;    
      // int ranks_world[cd_nranks];
      // MPI_Group_translate_ranks(cdgroup, cd_nranks, ranks, wgroup, ranks_world);
      // GA_Pgroup_set_default(GA_Pgroup_get_world());
      ga_pg = GA_Pgroup_create(ranks, cd_nranks);
      GA_Pgroup_set_default(ga_pg);
    }
  

  int64_t dims[3] = {nbf,nbf,max_cvecs};
  int64_t chnk[3] = {-1,-1,max_cvecs};
  int nblock32[GA_MAX_DIM]; 
  int64_t nblock[GA_MAX_DIM]; 

  int g_test = NGA_Create64(ga_eltype,ndim,dims,const_cast<char*>("CholVecTmp"),chnk);
  // int itype;
  // NGA_Inquire(g_test,&itype,&ndim,dims);
  NGA_Nblock(g_test,nblock32);
  NGA_Destroy(g_test);

  for(auto x=0;x<GA_MAX_DIM;x++) nblock[x] = nblock32[x];

  size_map = nblock[0]+nblock[1]+nblock[2];

  k_map = create_map(dims,nblock);

  // cout << "print k_map\n";
  // for (auto x=0;x<=mi;x++)
  // cout << k_map[x] << ",";
  // cout << endl;

  int g_chol = NGA_Create_irreg64(ga_eltype,3,dims,const_cast<char*>("CholX"),nblock,&k_map[0]);

  //util_eri_cholesky(rtdb,tol,g_chol,k,int_mb(k_map),nblock)
  GA_Zero(g_chol);

  //line 103-112
  // NGA_Inquire(g_chol,&itype,&ndim,dims);
  // cout << "dims = " << dims[0] << "," << dims[1] << "," << dims[2] << "\n";

  int64_t dims2[2] = {nbf,nbf};
  int64_t nblock2[2] = {nblock[0],nblock[1]};
  
  //TODO: Check k_map;
  int g_d = NGA_Create_irreg64(ga_eltype,2,dims2,const_cast<char*>("ERI Diag"), nblock2, &k_map[0]);
  int g_r = NGA_Create_irreg64(ga_eltype,2,dims2,const_cast<char*>("ERI Res"), nblock2, &k_map[0]);

  int64_t lo_b[GA_MAX_DIM]; // The lower limits of blocks of B
  int64_t lo_r[GA_MAX_DIM]; // The lower limits of blocks of R
  int64_t lo_d[GA_MAX_DIM]; // The lower limits of blocks of D
  int64_t hi_b[GA_MAX_DIM]; // The upper limits of blocks of B
  int64_t hi_r[GA_MAX_DIM]; // The upper limits of blocks of R
  int64_t hi_d[GA_MAX_DIM]; // The upper limits of blocks of D
  int64_t ld_b[GA_MAX_DIM]; // The leading dims of blocks of B
  int64_t ld_r[GA_MAX_DIM]; // The leading dims of blocks of R
  int64_t ld_d[GA_MAX_DIM]; // The leading dims of blocks of D

  // Distribution Check
  NGA_Distribution64(g_chol,iproc,lo_b,hi_b);
  NGA_Distribution64(g_d,iproc,lo_d,hi_d);
  NGA_Distribution64(g_r,iproc,lo_r,hi_r);

  // auto lo_hi_print = [] (auto& lo_b, auto& hi_b, string arr){
  //   cout << "lo_" << arr << ", hi_" << arr << " = ";
  //   cout << "[" << lo_b[0] << " " << lo_b[1] << "], [";
  //   cout << hi_b[0] << " " << hi_b[1] << "]\n";
  // };

  // if(cd_debug){
  //   lo_hi_print(lo_b,hi_b,"b");
  //   lo_hi_print(lo_d,hi_d,"d");
  //   lo_hi_print(lo_r,hi_r,"r");
  //   cout << "------debug0-----\n";
  // }
  

  auto shell2bf = map_shell_to_basis_function(shells);
  auto bf2shell = map_basis_function_to_shell(shells);

  // Step B. Compute the diagonal
  Engine engine(Operator::coulomb, max_nprim(shells), max_l(shells), 0);
  const auto &buf = engine.results();  

  for (size_t s1 = 0; s1 != shells.size(); ++s1) {
    auto bf1_first = shell2bf[s1]; // first basis function in this shell
    auto n1 = shells[s1].size();
    decltype(bf1_first) lo_d0 = lo_d[0];
    decltype(bf1_first) hi_d0 = hi_d[0];
    if(lo_d0 <= bf1_first && bf1_first <= hi_d0){

      for (size_t s2 = 0; s2 != shells.size(); ++s2) {
        auto bf2_first = shell2bf[s2];
        auto n2 = shells[s2].size();

        decltype(bf2_first) lo_d1 = lo_d[1];
        decltype(bf2_first) hi_d1 = hi_d[1];
        if(lo_d1 <= bf2_first && bf2_first <= hi_d1){

          //TODO: Screening
          engine.compute(shells[s1], shells[s2], shells[s1], shells[s2]);
          const auto *buf_1212 = buf[0];
          if (buf_1212 == nullptr)
            continue; // if all integrals screened out, skip to next quartet

          std::vector<TensorType> k_eri(n1*n2);
          for (decltype(n1) f1 = 0; f1 != n1; ++f1) {
            // const auto bf1 = f1 + bf1_first;
            for (decltype(n2) f2 = 0; f2 != n2; ++f2) {
              // const auto bf2 = f2 + bf2_first;
              auto f1212 = f1*n2*n1*n2 + f2*n1*n2 + f1*n2 + f2;
              k_eri[f1*n2+f2] = buf_1212[f1212];
              //// cout << f1212 << " " << s1 << s2 << "(" << bf1 << bf2 << "|" << bf1 << bf2 << ") = " << DiagInt(bf1, bf2) << endl;
            }
          }
          int64_t ibflo[2] = {cd_ncast<size_t>(bf1_first),cd_ncast<size_t>(bf2_first)};
          int64_t ibfhi[2] = {cd_ncast<size_t>(bf1_first+n1-1),cd_ncast<size_t>(bf2_first+n2-1)};
          int64_t ld[1] = {cd_ncast<size_t>(n2)};
          const void *from_buf = &k_eri[0];
          NGA_Put64(g_d,ibflo,ibfhi,const_cast<void*>(from_buf),ld);
        } //if s2
      } //s2
    } //#if s1
  } //s1

  // if(cd_debug) cout << "debug1\n";

// Step C. Find the coordinates of the maximum element of the diagonal.

  int64_t indx_d0[GA_MAX_DIM];
  TensorType val_d0;
  NGA_Select_elem64(g_d,const_cast<char*>("max"),&val_d0,indx_d0);

  // cout << "debug2\n";

  int64_t lo_x[GA_MAX_DIM]; // The lower limits of blocks
  int64_t hi_x[GA_MAX_DIM]; // The upper limits of blocks
  int64_t ld_x[GA_MAX_DIM]; // The leading dims of blocks

//     Step D. Start the while loop

  while(val_d0 > diagtol && count < max_cvecs){

      // cout << "iter, maxval, max_indx = " << count << ", " << val_d0 << 
      // ", [" << indx_d0[0] << "," << indx_d0[1] << "]\n";

    NGA_Zero(g_r);
    auto bfu = indx_d0[0];
    auto bfv = indx_d0[1];
    auto s1 = bf2shell[bfu];
    auto n1 = shells[s1].size();
    auto s2 = bf2shell[bfv];
    auto n2 = shells[s2].size();
    auto n12 = n1*n2;
    auto f1 = bfu - shell2bf[s1];
    auto f2 = bfv - shell2bf[s2];
    auto ind12 = f1*n2 + f2;

    // cout << "s1,s2 = " << s1 <<"," << s2 << endl;

    for (size_t s3 = 0; s3 != shells.size(); ++s3) {
      auto bf3_first = shell2bf[s3]; // first basis function in this shell
      auto n3 = shells[s3].size();

      decltype(bf3_first) lo_r0 = lo_r[0];
      decltype(bf3_first) hi_r0 = hi_r[0];
      if(lo_r0 <= bf3_first && bf3_first <= hi_r0){

        for (decltype(s3) s4 = 0; s4 != shells.size(); ++s4) {
          auto bf4_first = shell2bf[s4];
          auto n4 = shells[s4].size();

          decltype(bf4_first) lo_r1 = lo_r[1];
          decltype(bf4_first) hi_r1 = hi_r[1];
          if(lo_r1 <= bf4_first && bf4_first <= hi_r1){

            engine.compute(shells[s3], shells[s4], shells[s1], shells[s2]);
            const auto *buf_3412 = buf[0];
            if (buf_3412 == nullptr)
              continue; // if all integrals screened out, skip to next quartet

            std::vector<TensorType> k_eri(n3*n4);
            for (decltype(n3) f3 = 0; f3 != n3; ++f3) {
              // const auto bf3 = f3 + bf3_first;
              for (decltype(n4) f4 = 0; f4 != n4; ++f4) {
                // const auto bf4 = f4 + bf4_first;

                auto f3412 = f3*n4*n12 + f4*n12 + ind12;
                k_eri[f3*n4+f4] = buf_3412[f3412];
              }
            }
                
            const void *fbuf = &k_eri[0];
            //TODO
            int64_t ibflo[2] = {cd_ncast<size_t>(bf3_first),cd_ncast<size_t>(bf4_first)};
            int64_t ibfhi[2] = {cd_ncast<size_t>(bf3_first+n3-1),cd_ncast<size_t>(bf4_first+n4-1)};
            int64_t ld[1] = {cd_ncast<size_t>(n4)}; //n3                  
            // cout << "ld_x = " << ld[0] << endl;
            //lo_hi_print(ibflo,ibfhi,"ibf");
            NGA_Put64(g_r,ibflo,ibfhi,const_cast<void*>(fbuf),ld);
            } //if s4
          } //s4
      } //if s3
    } //s3
    NGA_Sync();

    // if(cd_debug) cout << "debug3\n";

   //  Step F. Update the residual
    lo_x[0] = indx_d0[0];
    lo_x[1] = indx_d0[1];
    lo_x[2] = 0;
    hi_x[0] = indx_d0[0];
    hi_x[1] = indx_d0[1];
    hi_x[2] = count; //count>0? count : 0;
    ld_x[0] = 1;
    ld_x[1] = hi_x[2]+1;

    TensorType *indx_b, *indx_d, *indx_r;
    std::vector<TensorType> k_elems(max_cvecs);
    TensorType* k_row = &k_elems[0];
    NGA_Get64(g_chol, lo_x, hi_x, k_row, ld_x);
    NGA_Access64(g_r, lo_r, hi_r, &indx_r, ld_r);
    NGA_Access64(g_chol, lo_b, hi_b, &indx_b, ld_b);

    // cout << "lo_r,hi_r = [" << lo_r[0] << "," << lo_r[1]
    // << "], [" << hi_r[0] << "," << hi_r[1] << "]\n";

    // cout << "lo_b,hi_b = [" << lo_b[0] << "," << lo_b[1] << "," << lo_b[2]
    // << "], [" << hi_b[0] << "," << hi_b[1] << "," << hi_b[2] << "]\n";

    // cout << "ld_r,ld_b = [" << ld_r[0] << "," << ld_r[1]
    // << "], [" << ld_b[0] << "," << ld_b[1] << "]\n";

    for(decltype(count) icount = 0;icount < count; icount++){
      for(int64_t i = 0; i<= hi_r[0] - lo_r[0]; i++) {
        for(int64_t j = 0; j <= hi_r[1] - lo_r[1]; j++) {
          indx_r[i*ld_r[0] + j] -= indx_b[icount+j*ld_b[1] + i*ld_b[1]*ld_b[0]]
              * k_row[icount];
        }
      }
    }

    // if(cd_debug) cout << "debug4: end update res\n";

    NGA_Release64(g_chol,lo_b,hi_b);
    NGA_Release_update64(g_r,lo_r,hi_r);

   // Step G. Compute the new Cholesky vector
   NGA_Access64(g_r,lo_r,hi_r,&indx_r,ld_r);
   NGA_Access64(g_chol,lo_b,hi_b,&indx_b,ld_b);

    //    cout << "ld_r,ld_b = [" << ld_r[0] << "," << ld_r[1]
    // << "], [" << ld_b[0] << "," << ld_b[1] << "]\n";

    for(auto i = 0; i <= hi_r[0] - lo_r[0]; i++) {
      for(auto j = 0; j <= hi_r[1] - lo_r[1]; j++) {
      auto tmp = indx_r[i*ld_r[0]+j]/sqrt(val_d0);
      indx_b[count+j*ld_b[1]+i*ld_b[1]*ld_b[0]] = tmp;
      }
    }

    NGA_Release_update64(g_chol,lo_b,hi_b);
    NGA_Release64(g_r,lo_r,hi_r);


    //Step H. Increment count
    count++;

    //Step I. Update the diagonal
    NGA_Access64(g_d,lo_d,hi_d,&indx_d,ld_d);
    NGA_Access64(g_chol,lo_b,hi_b,&indx_b,ld_b);

    // cout << "ld_d,ld_b = [" << ld_d[0] << "," << ld_d[1]
    // << "], [" << ld_b[0] << "," << ld_b[1] << "]\n";

    for(auto i = 0;i<= hi_d[0] - lo_d[0];i++) {
      for(auto j = 0; j<= hi_d[1] - lo_d[1];j++) {
        auto tmp = indx_b[count-1 + j*ld_b[1] + i*ld_b[1]*ld_b[0]];
        //cout << "tmp = " << tmp << endl;
        indx_d[i*ld_d[0]+j] -= tmp*tmp;
      }
    }

    NGA_Release64(g_chol,lo_b,hi_b);
    NGA_Release_update64(g_d,lo_d,hi_d);


  //Step J. Find the coordinates of the maximum element of the diagonal.
  NGA_Select_elem64(g_d,const_cast<char*>("max"),&val_d0,indx_d0);

  // cout << "-----------------------------------------\n";

  }

  if (iproc == 0) cout << "Number of cholesky vectors = " << count << endl;
  NGA_Destroy(g_r);
  NGA_Destroy(g_d);

  hf_t2 = std::chrono::high_resolution_clock::now();
  hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
  if(iproc == 0) std::cout << "\nTime taken for cholesky decomp: " << hf_time << " secs\n";

 
  dimsmo[0] = N; dimsmo[1] = N; dimsmo[2] = count;
  chnkmo[0] = -1; chnkmo[1] = -1; chnkmo[2] = count;
  // chnkmo = {-1,-1,count};

  int g_test2 = NGA_Create64(ga_eltype,3,dimsmo,const_cast<char*>("CholVecMOTmp"),chnkmo);
  NGA_Nblock(g_test2,nblockmo32);
  NGA_Destroy(g_test2);

  for(auto x=0;x<GA_MAX_DIM;x++) nblockmo[x] = nblockmo32[x];

  size_map = nblockmo[0]+nblockmo[1]+nblockmo[2];
  k_map = create_map(dimsmo,nblockmo);
  g_chol_mo = NGA_Create_irreg64(ga_eltype,3,dimsmo,const_cast<char*>("CholXMO"),nblockmo,&k_map[0]);
  GA_Zero(g_chol_mo);

  
  std::vector<TensorType> k_pj(N*nbf);
  std::vector<TensorType> k_pq(N*N);

  std::vector<TensorType> k_ij(nbf*nbf);
  std::vector<TensorType> k_eval_r(nbf);
  // auto lwork = 2*nbf*nbf+6*nbf+1;
  // std::vector<TensorType> k_work(lwork);
  // auto liwork = 5*nbf+3;
  // std::vector<TensorType> k_iwork(liwork);
  // int64_t g_num = 0;
  
  #define DO_SVD 0
  #if DO_SVD
    auto svdtol = 1e-8; //TODO same as diagtol ?
  #endif 

  hf_t1 = std::chrono::high_resolution_clock::now();
  double cvpr_time = 0;

  // AtomicCounter* ac = new AtomicCounterGA(ec.pg(), 1);
  // ac->allocate(0);
    char name[] = "atomic-counter";
    int64_t num_counters_ = 1;
    int64_t init_val = 0;
    int64_t size = num_counters_;
    int ga_ac = NGA_Create_config64(MT_C_LONGLONG, 1, &size, name, nullptr, ga_pg);
    EXPECTS(ga_ac != 0);  
    if(GA_Pgroup_nodeid(ga_pg) == 0) {
      int64_t lo[1] = {0};
      int64_t hi[1] = {num_counters_ - 1};
      int64_t ld = -1;
      long long buf[num_counters_];
      for(int i=0; i<num_counters_; i++) {
        buf[i] = init_val;
      }
      NGA_Put64(ga_ac, lo, hi, buf, &ld);
    }
    GA_Pgroup_sync(ga_pg);

  int64_t taskcount = 0;
  int64_t next = ac_fetch_add(ga_ac, 0, 1);

  for(decltype(count) kk=0;kk<count;kk++) {
    if(next == taskcount) {

      int64_t lo_ao[3] = {0,0,kk};
      int64_t hi_ao[3] = {nbf-1,nbf-1,kk};
      int64_t ld_ao[2] = {nbf,1};

      NGA_Get64(g_chol, lo_ao, hi_ao, &k_ij[0], ld_ao);

      // cout << "kk = " << kk << endl;

      #if DO_SVD
        //uplotri
        for(auto i=0;i<nbf;i++)
        for(auto j=i+1;j<nbf;j++)
          k_ij[i*nbf+j] = 0;

        //TODO
        LAPACKE_dsyevd(LAPACK_ROW_MAJOR,'V','L',(TAMM_LAPACK_INT)nbf,
          &k_ij[0],(TAMM_LAPACK_INT)nbf, &k_eval_r[0]);

        // print_array(&k_eval_r[0],nbf,"k_eval_r");
        // print_array(&k_ij[0],nbf,"k_ij");

        auto m = 0;
        for(auto i=0;i<nbf;i++){
          if(fabs(k_eval_r[i]) <= svdtol) continue;
          k_eval_r[m] = k_eval_r[i];
          //ma_copy
          for(auto x=0;x<nbf;x++)
            k_ij[m*nbf+x] = k_ij[i*nbf+x]; 
          m++;
        }

        // cout << "m=" << m << endl;
        
        std::vector<TensorType> k_ij_tmp(nbf*m);
          for(auto i=0;i<nbf;i++)
          for(auto j=0;j<m;j++)
            k_ij_tmp[i*m+j] = k_ij[j*nbf+i];

        

        g_num += m;
        std::vector<TensorType> k_pi(N*m);
        std::vector<TensorType> k_qj(N*m);
        cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,N,m,nbf,
                    1.0,k_movecs_sorted,N,&k_ij[0],nbf,0,&k_pi[0],N);

        //  print_array(&k_pi[0],m,"k_pi");

        // print_array_col(&k_pi[0],m,"k_pi_col");

        //ma_copy
        for(auto x=0;x<N*m;x++) k_qj[x] = k_pi[x]; 

        //ma_scale
        for(auto i=0;i<N;i++){
          auto sf = k_eval_r[i];
          for (auto j=0;j<m;j++)
            k_pi[i*m+j] *= sf;
        }

        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,N,N,m,
                1,&k_pi[0],N,&k_qj[0],N,0,&k_pq[0],N);

      #else

      //---------Two-Step-Contraction----
      auto cvpr_t1 = std::chrono::high_resolution_clock::now();
      //  cout << "contraction1\n";
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,N,nbf,nbf,
          1,k_movecs_sorted,N,&k_ij[0],nbf,0,&k_pj[0],nbf);
        
      // cout << "contraction2\n";
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,N,N,nbf,
          1,&k_pj[0],nbf,k_movecs_sorted,N,0,&k_pq[0],N);
      // cblas_dgemm(CblasRowMajor,CblasTrans,CblasTrans,nbf,N,N,
      //     1,&k_pj[0],N,k_movecs_sorted,N,0,&k_pq[0],N);

      auto cvpr_t2 = std::chrono::high_resolution_clock::now();
      cvpr_time +=
        std::chrono::duration_cast<std::chrono::duration<double>>((cvpr_t2 - cvpr_t1)).count();

      #endif

      int64_t lo_mo[3] = {0,0,kk};
      int64_t hi_mo[3] = {N-1,N-1,kk};
      int64_t ld_mo[2] = {N,1};        

      NGA_Put64(g_chol_mo,lo_mo,hi_mo,&k_pq[0],ld_mo);
      // next = ac->fetch_add(0, 1);      
      next = ac_fetch_add(ga_ac, 0, 1);
    } //next==taskcount
    taskcount++;
  }

  // ec.pg().barrier();
  // ac->deallocate();
  // delete ac;
  GA_Pgroup_sync(ga_pg);

  NGA_Destroy(g_chol);
  k_pj.clear(); k_pj.shrink_to_fit();
  k_pq.clear(); k_pq.shrink_to_fit();
  k_ij.clear(); k_ij.shrink_to_fit();
  k_eval_r.clear(); k_eval_r.shrink_to_fit();

  hf_t2 = std::chrono::high_resolution_clock::now();
  hf_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
  if(rank == 0) {
    std::cout << "\nTotal Time for constructing CholVpr: " << hf_time << " secs";
    std::cout << "\n --> Time for 2-step contraction: " << cvpr_time << " secs\n";
  }

  if(throttle_cd) GA_Pgroup_set_default(ga_pg_default);

  }//end throttle

  ec.pg().barrier();
  GA_Brdcst(&count, sizeof(int64_t), 0);

  hf_t1 = std::chrono::high_resolution_clock::now();

  // dimsmo = {N,N,count};
  // chnkmo = {-1,-1,count};
  dimsmo[0] = N; dimsmo[1] = N; dimsmo[2] = count;
  chnkmo[0] = -1; chnkmo[1] = -1; chnkmo[2] = count;
  
  int g_test_mo = NGA_Create64(ga_eltype,3,dimsmo,const_cast<char*>("CholVecMOTmp"),chnkmo);
  NGA_Nblock(g_test_mo,nblockmo32);
  NGA_Destroy(g_test_mo);

  for(auto x=0;x<GA_MAX_DIM;x++) nblockmo[x] = nblockmo32[x];

  size_map = nblockmo[0]+nblockmo[1]+nblockmo[2];
  k_map = create_map(dimsmo,nblockmo);
  int g_chol_mo_copy = NGA_Create_irreg64(ga_eltype,3,dimsmo,const_cast<char*>("CholXMOCopy"),nblockmo,&k_map[0]);
  GA_Zero(g_chol_mo_copy);

  if(iproc < cd_nranks) { //throttle  
    GA_Pgroup_set_default(ga_pg);
    GA_Copy(g_chol_mo,g_chol_mo_copy);
    NGA_Destroy(g_chol_mo);
    GA_Pgroup_sync(ga_pg);
    GA_Pgroup_set_default(ga_pg_default);
  }

  ec.pg().barrier();

  IndexSpace CIp{range(0, count)};
  TiledIndexSpace tCIp{CIp, count}; //TODO: replace count with iptilesize 
  // auto [cindexp] = tCIp.labels<1>("all");

  Tensor<TensorType> CholVpr_tamm{{tMO,tMO,tCIp},{SpinPosition::upper,SpinPosition::lower,SpinPosition::ignore}};
  Tensor<TensorType>::allocate(&ec, CholVpr_tamm);
  
  //convert g_chol_mo_copy to CholVpr_tamm
    auto lambdacv = [&](const IndexVector& bid){
        const IndexVector blockid =
        internal::translate_blockid(bid, CholVpr_tamm());

        auto block_dims   = CholVpr_tamm.block_dims(blockid);
        auto block_offset = CholVpr_tamm.block_offsets(blockid);

        const tamm::TAMM_SIZE dsize = CholVpr_tamm.block_size(blockid);

        int64_t lo[3] = {cd_ncast<size_t>(block_offset[0]), 
                         cd_ncast<size_t>(block_offset[1]), 
                         cd_ncast<size_t>(block_offset[2])};
        int64_t hi[3] = {cd_ncast<size_t>(block_offset[0] + block_dims[0]-1), 
                         cd_ncast<size_t>(block_offset[1] + block_dims[1]-1),
                         cd_ncast<size_t>(block_offset[2] + block_dims[2]-1)};
        int64_t ld[2] = {cd_ncast<size_t>(block_dims[1]),
                         cd_ncast<size_t>(block_dims[2])};

        std::vector<TensorType> sbuf(dsize);
        NGA_Get64(g_chol_mo_copy,lo,hi,&sbuf[0],ld);

        CholVpr_tamm.put(blockid, sbuf);
    };

    block_for(ec, CholVpr_tamm(), lambdacv);

  NGA_Destroy(g_chol_mo_copy);

  hf_t2 = std::chrono::high_resolution_clock::now();
  hf_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
  if(rank == 0) std::cout << "\nTime for ga_chol_mo -> CholVpr_tamm conversion: " << hf_time << " secs\n";


  #if 0
  
  Tensor<TensorType> CholVuv_opt{tAO, tAO, tCIp};
  Tensor<TensorType>::allocate(&ec, CholVuv_opt);

  hf_t1 = std::chrono::high_resolution_clock::now();

  // Contraction 1
  Tensor<TensorType> CholVpv_tamm{tMO,tAO,tCIp};
  Tensor<TensorType>::allocate(&ec, CholVpv_tamm);
  Scheduler{ec}(CholVpv_tamm(pmo,mu,cindexp) = CTiled_tamm(nu, pmo) * CholVuv_opt(nu, mu, cindexp)).execute();
  Tensor<TensorType>::deallocate(CholVuv_opt);

  hf_t2 = std::chrono::high_resolution_clock::now();
  hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
  if(rank == 0) std::cout << "\nTime taken for computing CholVpv: " << hf_time << " secs\n";

  //Contraction 2
  hf_t1 = std::chrono::high_resolution_clock::now();
  
  Tensor<TensorType> CholVpr_tamm{{tMO,tMO,tCIp},{SpinPosition::upper,SpinPosition::lower,SpinPosition::ignore}};
  Tensor<TensorType>::allocate(&ec, CholVpr_tamm);
  Scheduler{ec}(CholVpr_tamm(pmo,rmo,cindexp) += CTiled_tamm(mu, rmo) * CholVpv_tamm(pmo, mu,cindexp)).execute();
  Tensor<TensorType>::deallocate(CholVpv_tamm,CTiled_tamm);

  hf_t2 = std::chrono::high_resolution_clock::now();
  hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
  if(rank == 0) std::cout << "\nTime taken for computing CholVpr: " << hf_time << " secs\n";

  #endif

  chol_count = count;
  return CholVpr_tamm;
}

#endif //TAMM_CD_SVD_HPP_