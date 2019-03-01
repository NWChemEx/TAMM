
#ifndef TAMM_CD_SVD_GA_HPP_
#define TAMM_CD_SVD_GA_HPP_

#include "HF/hartree_fock_tamm.hpp"
#include "tamm/eigen_utils.hpp"
#include "tamm/tamm.hpp"

using namespace tamm;
using TensorType = double;

bool cd_debug = true;

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

Tensor<TensorType> cd_svd_ga(ExecutionContext& ec, TiledIndexSpace& tMO, TiledIndexSpace& tAO,
  const TAMM_SIZE ndocc, const TAMM_SIZE nao, const TAMM_SIZE freeze_core,
  const TAMM_SIZE freeze_virtual, Tensor<TensorType> C_AO, Tensor<TensorType> F_AO,
  Tensor<TensorType> F_MO, TAMM_SIZE& chol_count, const tamm::Tile max_cvecs, double diagtol,
  libint2::BasisSet& shells, std::vector<size_t>& shell_tile_map) {

    using libint2::Atom;
    using libint2::Shell;
    using libint2::Engine;
    using libint2::Operator;

    auto rank = ec.pg().rank();

    std::vector<unsigned int> AO_tiles;
    for(auto s : shells) AO_tiles.push_back(s.size());

    auto [mu, nu, ku] = tAO.labels<3>("all");
    auto [pmo, rmo] = tMO.labels<2>("all");


    if(rank == 0){
      cout << "\n-----------------------------------------------------\n";
      cout << "Begin Cholesky Decomposition ... " << endl;
      cout << "\n#AOs, #electrons = " << nao << " , " << ndocc << endl;
    }

    auto ov_alpha_freeze = ndocc - freeze_core;
    auto ov_beta_freeze  = nao - ndocc - freeze_virtual;
    
    const auto v2dim  = 2 * nao - 2 * freeze_core - 2 * freeze_virtual;
    const int n_alpha = ov_alpha_freeze;
    const int n_beta  = ov_beta_freeze;
    auto ov_alpha     = ndocc;
    TAMM_SIZE ov_beta{nao - ov_alpha};

    // 2-index transform

    auto hf_t1 = std::chrono::high_resolution_clock::now();

    Tensor<TensorType> CTiled_tamm{tAO,tMO};
    Tensor<TensorType>::allocate(&ec, CTiled_tamm);

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
      Matrix CTiled(nao, 2 * nao - 2 * freeze_core - 2 * freeze_virtual);
      CTiled << C_noa, C_nob, C_nva, C_nvb;

      //  cout << "\n\t CTiled Matrix = [C_noa C_nob C_nva C_nvb]:\n";
      //  cout << CTiled << endl;

      Matrix F(nao,nao);
      tamm_to_eigen_tensor(F_AO,F);
      F = CTiled.transpose() * (F * CTiled); //F is resized to 2N*2N
      
      eigen_to_tamm_tensor(CTiled_tamm,CTiled);
      eigen_to_tamm_tensor(F_MO,F);

      // cout << "F_MO dims = " << F.rows() << "," << F.cols()  << endl;
    }

  GA_Sync();

  //rank = iproc
  int iproc = rank.value();
  int ndim = 3;
  // auto pref = max_cvecs;
  auto nbf = nao;

  int dims[3] = {nbf,nbf,max_cvecs};
  int chnk[3] = {-1,-1,max_cvecs};
  int nblock[GA_MAX_DIM]; 

  auto eltype   = tensor_element_type<TensorType>();
  int ga_eltype = C_DBL; //TODO: MemoryManagerGA::to_ga_eltype(eltype);
  int g_test = NGA_Create(ga_eltype,ndim,dims,const_cast<char*>("CholVecTmp"),chnk);
  // int itype;
  // NGA_Inquire(g_test,&itype,&ndim,dims);
  NGA_Nblock(g_test,nblock);
  NGA_Destroy(g_test);

  int size_map = nblock[0]+nblock[1]+nblock[2];
  std::vector<int> k_map(size_map);

  // cout << "size_map = " << size_map << endl;

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

  // cout << "print k_map\n";
  // for (auto x=0;x<=mi;x++)
  // cout << k_map[x] << ",";
  // cout << endl;

  int g_chol = NGA_Create_irreg(ga_eltype,3,dims,const_cast<char*>("CholX"),nblock,&k_map[0]);

  //util_eri_cholesky(rtdb,tol,g_chol,k,int_mb(k_map),nblock)
  GA_Zero(g_chol);

  //line 103-112
  // NGA_Inquire(g_chol,&itype,&ndim,dims);
  // cout << "dims = " << dims[0] << "," << dims[1] << "," << dims[2] << "\n";

  int dims2[2] = {nbf,nbf};
  int nblock2[2] = {nblock[0],nblock[1]};
  
  //TODO: Check k_map;
  int g_d = NGA_Create_irreg(ga_eltype,2,dims2,const_cast<char*>("ERI Diag"), nblock2, &k_map[0]);
  int g_r = NGA_Create_irreg(ga_eltype,2,dims2,const_cast<char*>("ERI Res"), nblock2, &k_map[0]);

  int lo_b[GA_MAX_DIM]; // The lower limits of blocks of B
  int lo_r[GA_MAX_DIM]; // The lower limits of blocks of R
  int lo_d[GA_MAX_DIM]; // The lower limits of blocks of D
  int hi_b[GA_MAX_DIM]; // The upper limits of blocks of B
  int hi_r[GA_MAX_DIM]; // The upper limits of blocks of R
  int hi_d[GA_MAX_DIM]; // The upper limits of blocks of D
  int ld_b[GA_MAX_DIM]; // The leading dims of blocks of B
  int ld_r[GA_MAX_DIM]; // The leading dims of blocks of R
  int ld_d[GA_MAX_DIM]; // The leading dims of blocks of D

  // Distribution Check
  NGA_Distribution(g_chol,iproc,lo_b,hi_b);
  NGA_Distribution(g_d,iproc,lo_d,hi_d);
  NGA_Distribution(g_r,iproc,lo_r,hi_r);

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
  
  auto count = 0U; //Step A. Initialize chol vector count

  auto shell2bf = map_shell_to_basis_function(shells);
  auto bf2shell = map_basis_function_to_shell(shells);

  // Step B. Compute the diagonal
  Engine engine(Operator::coulomb, max_nprim(shells), max_l(shells), 0);
  const auto &buf = engine.results();  

  for (size_t s1 = 0; s1 != shells.size(); ++s1) {
    auto bf1_first = shell2bf[s1]; // first basis function in this shell
    auto n1 = shells[s1].size();
    if(lo_d[0] <= bf1_first && bf1_first <= hi_d[0]){

      for (size_t s2 = 0; s2 != shells.size(); ++s2) {
        auto bf2_first = shell2bf[s2];
        auto n2 = shells[s2].size();

         if(lo_d[1] <= bf2_first && bf2_first <= hi_d[1]){

          //TODO:Screening
          engine.compute(shells[s1], shells[s2], shells[s1], shells[s2]);
          const auto *buf_1212 = buf[0];
          if (buf_1212 == nullptr)
            continue; // if all integrals screened out, skip to next quartet

          std::vector<TensorType> k_eri(n1*n2);
          for (size_t f1 = 0; f1 != n1; ++f1) {
            const auto bf1 = f1 + bf1_first;
            for (size_t f2 = 0; f2 != n2; ++f2) {
              const auto bf2 = f2 + bf2_first;
              auto f1212 = f1*n2*n1*n2 + f2*n1*n2 + f1*n2 + f2;
              k_eri[f1*n2+f2] = buf_1212[f1212];
              // cout << f1212 << " " << s1 << s2 << "(" << bf1 << bf2 << "|" << bf1 << bf2 << ") = " << DiagInt(bf1, bf2) << endl;
            }
          }
          int ibflo[2] = {bf1_first,bf2_first};
          int ibfhi[2] = {bf1_first+n1-1,bf2_first+n2-1};
          int ld[1] = {n2};
          const void *from_buf = k_eri.data();
          NGA_Put(g_d,ibflo,ibfhi,const_cast<void*>(from_buf),ld);
        } //if s2
      } //s2
    } //#if s1
  } //s1

  // if(cd_debug) cout << "debug1\n";

// Step C. Find the coordinates of the maximum element of the diagonal.

  int indx_d0[GA_MAX_DIM];
  TensorType val_d0;
  NGA_Select_elem(g_d,const_cast<char*>("max"),&val_d0,indx_d0);

  // cout << "debug2\n";

  int lo_x[GA_MAX_DIM]; // The lower limits of blocks
  int hi_x[GA_MAX_DIM]; // The upper limits of blocks
  int ld_x[GA_MAX_DIM]; // The leading dims of blocks

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

      if(lo_r[0] <= bf3_first && bf3_first <= hi_r[0]){

        for (size_t s4 = 0; s4 != shells.size(); ++s4) {
          auto bf4_first = shell2bf[s4];
          auto n4 = shells[s4].size();

          if(lo_r[1] <= bf4_first && bf4_first <= hi_r[1]){

            engine.compute(shells[s3], shells[s4], shells[s1], shells[s2]);
            const auto *buf_3412 = buf[0];
            if (buf_3412 == nullptr)
              continue; // if all integrals screened out, skip to next quartet

            std::vector<TensorType> k_eri(n3*n4);
            for (size_t f3 = 0; f3 != n3; ++f3) {
              const auto bf3 = f3 + bf3_first;
              for (size_t f4 = 0; f4 != n4; ++f4) {
                const auto bf4 = f4 + bf4_first;

                auto f3412 = f3*n4*n12 + f4*n12 + ind12;
                k_eri[f3*n4+f4] = buf_3412[f3412];
              }
            }
                
            const void *fbuf = &k_eri[0];
            //TODO
            int ibflo[2] = {bf3_first,bf4_first};
            int ibfhi[2] = {bf3_first+n3-1,bf4_first+n4-1};
            int ld[1] = {n4}; //n3                  
            // cout << "ld_x = " << ld[0] << endl;
            //lo_hi_print(ibflo,ibfhi,"ibf");
            NGA_Put(g_r,ibflo,ibfhi,const_cast<void*>(fbuf),ld);
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
    NGA_Get(g_chol, lo_x, hi_x, k_row, ld_x);
    NGA_Access(g_r, lo_r, hi_r, &indx_r, ld_r);
    NGA_Access(g_chol, lo_b, hi_b, &indx_b, ld_b);

    // cout << "lo_r,hi_r = [" << lo_r[0] << "," << lo_r[1]
    // << "], [" << hi_r[0] << "," << hi_r[1] << "]\n";

    // cout << "lo_b,hi_b = [" << lo_b[0] << "," << lo_b[1] << "," << lo_b[2]
    // << "], [" << hi_b[0] << "," << hi_b[1] << "," << hi_b[2] << "]\n";

    // cout << "ld_r,ld_b = [" << ld_r[0] << "," << ld_r[1]
    // << "], [" << ld_b[0] << "," << ld_b[1] << "]\n";

    for(auto icount =0;icount < count; icount++){
      for(auto i = 0;i<= hi_r[0] - lo_r[0];i++) {
        for(auto j = 0; j<= hi_r[1] - lo_r[1];j++) {
          indx_r[i*ld_r[0] + j] -= indx_b[icount+j*ld_b[1] + i*ld_b[1]*ld_b[0]]
              * k_row[icount];
        }
      }
    }

    // if(cd_debug) cout << "debug4: end update res\n";

    NGA_Release(g_chol,lo_b,hi_b);
    NGA_Release_update(g_r,lo_r,hi_r);

   // Step G. Compute the new Cholesky vector
   NGA_Access(g_r,lo_r,hi_r,&indx_r,ld_r);
   NGA_Access(g_chol,lo_b,hi_b,&indx_b,ld_b);

    //    cout << "ld_r,ld_b = [" << ld_r[0] << "," << ld_r[1]
    // << "], [" << ld_b[0] << "," << ld_b[1] << "]\n";

    for(auto i = 0; i <= hi_r[0] - lo_r[0]; i++) {
      for(auto j = 0; j <= hi_r[1] - lo_r[1]; j++) {
      auto tmp = indx_r[i*ld_r[0]+j]/sqrt(val_d0);
      indx_b[count+j*ld_b[1]+i*ld_b[1]*ld_b[0]] = tmp;
      }
    }

    NGA_Release_update(g_chol,lo_b,hi_b);
    NGA_Release(g_r,lo_r,hi_r);


    //Step H. Increment count
    count++;

    //Step I. Update the diagonal
    NGA_Access(g_d,lo_d,hi_d,&indx_d,ld_d);
    NGA_Access(g_chol,lo_b,hi_b,&indx_b,ld_b);

    // cout << "ld_d,ld_b = [" << ld_d[0] << "," << ld_d[1]
    // << "], [" << ld_b[0] << "," << ld_b[1] << "]\n";

    for(auto i = 0;i<= hi_d[0] - lo_d[0];i++) {
      for(auto j = 0; j<= hi_d[1] - lo_d[1];j++) {
        auto tmp = indx_b[count-1 + j*ld_b[1] + i*ld_b[1]*ld_b[0]];
        //cout << "tmp = " << tmp << endl;
        indx_d[i*ld_d[0]+j] -= tmp*tmp;
      }
    }

    NGA_Release(g_chol,lo_b,hi_b);
    NGA_Release_update(g_d,lo_d,hi_d);


  //Step J. Find the coordinates of the maximum element of the diagonal.
  NGA_Select_elem(g_d,const_cast<char*>("max"),&val_d0,indx_d0);

  // cout << "-----------------------------------------\n";

  }

  if (iproc == 0) cout << "number of cholesky vectors = " << count << endl;
  NGA_Destroy(g_r);
  NGA_Destroy(g_d);
  
  #if 0

  auto hf_t2 = std::chrono::high_resolution_clock::now();

  double hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
  if(rank == 0) 
      cout << "\nTime for 2-index transform: " << hf_time << " secs\n";
    

  IndexSpace CI{range(0, max_cvecs)};
  TiledIndexSpace tCI{CI, 1};
  auto [cindex] = tCI.labels<1>("all");

  libint2::initialize(false);

  auto shell2bf = map_shell_to_basis_function(shells);

  hf_t1 = std::chrono::high_resolution_clock::now();
  /* 
  DiagInt stores the diagonal integrals, i.e. (uv|uv)'s
  ScrCol temporarily stores all (uv|rs)'s with fixed r and s
  */

  // Compute diagonals
  Engine engine(Operator::coulomb, max_nprim(shells), max_l(shells), 0);
  const auto &buf = engine.results();  

  Tensor<TensorType> DiagInt_tamm{tAO, tAO};
  Tensor<TensorType>::allocate(&ec, DiagInt_tamm);
  std::tie(obs_shellpair_list, obs_shellpair_data) = compute_shellpairs(shells);

  auto tensor1e = DiagInt_tamm;

    auto compute_diagonals = [&](const IndexVector& blockid) {
       
        auto bi0 = blockid[0];
        auto bi1 = blockid[1];

        const TAMM_SIZE size = tensor1e.block_size(blockid);
        auto block_dims   = tensor1e.block_dims(blockid);
        std::vector<TensorType> dbuf(size);

        auto bd1 = block_dims[1];

        auto s1range_start = 0l;
        auto s1range_end = shell_tile_map[bi0];
        if (bi0>0) s1range_start = shell_tile_map[bi0-1]+1;
        
        for (auto s1 = s1range_start; s1 <= s1range_end; ++s1) {
          auto n1 = shells[s1].size();

        auto s2range_start = 0l;
        auto s2range_end = shell_tile_map[bi1];
        if (bi1>0) s2range_start = shell_tile_map[bi1-1]+1;

          for (size_t s2 = s2range_start; s2 <= s2range_end; ++s2) {

          if(s2>s1){
            auto s2spl = obs_shellpair_list[s2];
            if(std::find(s2spl.begin(),s2spl.end(),s1) == s2spl.end()) continue;
          }
          else{
            auto s2spl = obs_shellpair_list[s1];
            if(std::find(s2spl.begin(),s2spl.end(),s2) == s2spl.end()) continue;
          }
          
          auto n2 = shells[s2].size();

          // compute shell pair; return is the pointer to the buffer
        engine.compute(shells[s1], shells[s2], shells[s1], shells[s2]);
          const auto *buf_1212 = buf[0];
          if (buf_1212 == nullptr) continue;
          
        //   std::vector<TensorType> tbuf(n1*n2);
        //   for (size_t f1 = 0; f1 != n1; ++f1) {
        //   for (size_t f2 = 0; f2 != n2; ++f2) {
        //     auto f1212 = f1*n2*n1*n2 + f2*n1*n2 + f1*n2 + f2;
        //     tbuf[f1*n2+f2] = buf_1212[f1212];
        //   }
        // }

          auto curshelloffset_i = 0U;
          auto curshelloffset_j = 0U;
          for(auto x=s1range_start;x<s1;x++) curshelloffset_i += AO_tiles[x];
          for(auto x=s2range_start;x<s2;x++) curshelloffset_j += AO_tiles[x];

          auto dimi =  curshelloffset_i + AO_tiles[s1];
          auto dimj =  curshelloffset_j + AO_tiles[s2];

          for(size_t i = curshelloffset_i; i < dimi; i++) {
          for(size_t j = curshelloffset_j; j < dimj; j++) {
                  auto f1 = i - curshelloffset_i;
                  auto f2 = j - curshelloffset_j;
                   auto f1212 = f1*n2*n1*n2 + f2*n1*n2 + f1*n2 + f2;
                  dbuf[i*bd1+j] = buf_1212[f1212];
                  
                }
          }
          }
        }
        tensor1e.put(blockid,dbuf);
  };
  block_for(ec, DiagInt_tamm(),compute_diagonals);

   hf_t2 = std::chrono::high_resolution_clock::now();
  hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
  if(rank == 0) std::cout << "\nTime taken for computing diagonals: " << hf_time << " secs\n";


  hf_t1 = std::chrono::high_resolution_clock::now();

  auto count = 0U;
  size_t bfu = 0; // basis function pair |uv) corresponding to 
  size_t bfv = 0; //  maximun diagonal.
  Index s1 = 0;
  size_t n1 = 0;
  Index s2 = 0;
  size_t n2 = 0;
  size_t n12 = 0;
  size_t f1 = 0;
  size_t f2 = 0;
  size_t ind12 = 0;
  
  Tensor<TensorType> CholVuv_tamm{tCI, tAO, tAO};
  Tensor<TensorType>::allocate(&ec, CholVuv_tamm);

  std::vector<Tensor<TensorType>> cvecs;

  int CholVuv_ga = CholVuv_tamm.ga_handle();
  int DiagInt_ga = DiagInt_tamm.ga_handle();

  TensorType max=0;
  std::vector<size_t> bfuv;
  IndexVector maxblockid;

  auto citer = 0;
  auto debug = true;

    // Find maximum in DiagInt 
  std::tie(max,maxblockid,bfuv) = max_element(ec, DiagInt_tamm());
  bfu = bfuv[0];
  bfv = bfuv[1];

  // tamm::Tile est_ts = 0;
  std::vector<tamm::Tile> AO_opttiles;
  // for(auto s=0U;s<shells.size();s++){
  //   est_ts += shells[s].size();
  //   if(est_ts>=30) {
  //     AO_opttiles.push_back(est_ts);
  //     est_ts=0;
  //   }
  // }
  // if(est_ts>0){
  //   AO_opttiles.push_back(est_ts);
  // }

      Tensor<TensorType> tensor = DiagInt_tamm;
    // Defined only for NxN tensors
    EXPECTS(tensor.num_modes() == 2);

    LabelLoopNest loop_nest{tensor().labels()};
    std::vector<TensorType> dest;

    for(const IndexVector& bid : loop_nest) {
        const IndexVector blockid =
          internal::translate_blockid(bid, tensor());

          const TAMM_SIZE size = tensor.block_size(blockid);
          AO_opttiles.push_back(size);
    }

  IndexSpace AO1D{range(0, nao*nao)};
  TiledIndexSpace tAO1D{AO1D, AO_opttiles};

  // Tensor<TensorType> DInt_GA{tAO1D};
  // Tensor<TensorType>::allocate(&ec,DInt_GA);
  // int DI_GA_handle = DInt_GA.ga_handle();
  // Tensor2D diagint_eigen = tamm_to_eigen_tensor<TensorType,2>(DiagInt_tamm);
  // eigen_to_tamm_tensor(DInt_GA, diagint_eigen);

  do {

    Tensor<TensorType> tmp1{tAO,tAO}; //not allocated
    Tensor<TensorType> tmp{tAO1D};
    Tensor<TensorType>::allocate(&ec,tmp);
    cvecs.push_back(tmp);

    int tmp_ga = tmp.ga_handle();
    cout << "tmp_ga = " << tmp_ga << endl;

    auto cd_t1 = std::chrono::high_resolution_clock::now();

    citer++;

    auto curshelloffset_i = 0U;
    auto curshelloffset_j = 0U;
    std::tie(s1,s2,curshelloffset_i,curshelloffset_j) 
      = get_shell_ids(shell_tile_map, AO_tiles, maxblockid, bfu, bfv);

    // Compute all (**|uv)'s for given shells
    // s1 = maxblockid[0]; //bf2shell(bfu);
    n1 = shells[s1].size();
    // s2 = maxblockid[1]; //bf2shell(bfv);
    n2 = shells[s2].size();
    n12 = n1*n2;
    f1 = bfu - curshelloffset_i; //shell2bf[s1];
    f2 = bfv - curshelloffset_j; //shell2bf[s2];
    ind12 = f1*n2 + f2;

        // cout << "new update cols --- s1,s2,n1,n2 = "  << s1 << "," << s2 << "," << n1 <<"," << n2 <<endl;

     IndexVector di = {maxblockid[0],maxblockid[1]};
  //   const tamm::TAMM_SIZE dec = CholVuv_tamm.block_size(di);
  //   std::vector<TensorType> cvuvbuf(dec);
  //   CholVuv_tamm.get(di, cvuvbuf);
     auto block_offsets_di   = DiagInt_tamm.block_offsets(di);

  std::vector<TensorType> delems(count);
  // //  auto depos = ind12*block_dims_vuv[0];

    // for (auto icount = 0U; icount != count; ++icount) {
    //     delems[icount] = cvuvbuf[(icount*block_dims_vuv[1]+bfu)*
    //         block_dims_vuv[2]+bfv];
    // }

    int64_t diag_ga_offset = block_offsets_di[0]*nao; 
    for(Index y=0; y<maxblockid[1];y++){
      diag_ga_offset += DiagInt_tamm.block_size({maxblockid[0],y});
    }
    diag_ga_offset += bfu*DiagInt_tamm.block_dims(di)[1]+bfv;

    int64_t lo = diag_ga_offset, hi = diag_ga_offset+1, ld = -1;
    cout << "lo,hi,diag_ga_offset = "  << lo << "," << hi <<"," << diag_ga_offset << endl;

    for (auto icount = 0U; icount != count; ++icount) {
        int prevc_ga = cvecs[icount].ga_handle();
        cout << "prevc_Ga = " << prevc_ga << endl;
        TensorType to_buf=0;
        cout << "inside lo,hi = "  << lo << "," << hi <<"," << endl;
        NGA_Get64(prevc_ga, &lo, &hi, (void*)(&to_buf), &ld);
        delems[icount] = to_buf;
    }

    if(citer==2||citer==3) cout << "delems = " << delems << endl;

  auto update_columns = [&](const IndexVector& blockid) {

        auto bi0 = blockid[0];
        auto bi1 = blockid[1];

        const TAMM_SIZE size = tmp1.block_size(blockid);
        auto block_dims   = tmp1.block_dims(blockid);
        auto block_offsets   = tmp1.block_offsets(blockid);
        // std::vector<TensorType> vuvbuf(size);
        std::vector<TensorType> dibuf(DiagInt_tamm.block_size({bi0,bi1}));

        auto bd0 = block_dims[0];
        auto bd1 = block_dims[1];
        EXPECTS(size==bd0*bd1);

        std::vector<TensorType> cbuf(bd0*bd1);

        // cout << "bi0,bi1,bd0,bd1,boffset,nao = " << bi0 << "," << bi1 << "," << bd0 << "," << bd1 << 
        // "," << block_offsets[0] << " , " << nao << endl;
        int64_t ga_shell_offset = block_offsets[0]*nao;
        for(Index y=0; y<bi1;y++){
          // cout << "bi0,bi1,size = " << y << "," << bi1 << "," << tmp1.block_size({bi0,y}) << endl;
          ga_shell_offset += tmp1.block_size({bi0,y});
        }

        // auto l_t1 = std::chrono::high_resolution_clock::now();

        // CholVuv_tamm.get(blockid, vuvbuf);
        DiagInt_tamm.get({bi0,bi1}, dibuf);

        // auto l_t2 = std::chrono::high_resolution_clock::now();
        // auto l_time =
        //     std::chrono::duration_cast<std::chrono::duration<double>>((l_t2 - l_t1)).count();
        // get_time+=l_time;

        auto s3range_start = 0l;
        auto s3range_end = shell_tile_map[bi0];
        if (bi0>0) s3range_start = shell_tile_map[bi0-1]+1;

        
        
        for (Index s3 = s3range_start; s3 <= s3range_end; ++s3) {
          auto n3 = shells[s3].size();

          auto s4range_start = 0l;
          auto s4range_end = shell_tile_map[bi1];
          if (bi1>0) s4range_start = shell_tile_map[bi1-1]+1;

        for (Index s4 = s4range_start; s4 <= s4range_end; ++s4) {

          if(s4>s3){
            auto s2spl = obs_shellpair_list[s4];
            if(std::find(s2spl.begin(),s2spl.end(),s3) == s2spl.end()) continue;
          }
          else{
            auto s2spl = obs_shellpair_list[s3];
            if(std::find(s2spl.begin(),s2spl.end(),s4) == s2spl.end()) continue;
          }
          
          auto n4 = shells[s4].size();

        // cout << "s1,s2,s3,s4 = [" << s1 << "," << s2 << "," << s3 << "," << s4 << "]\n";
          engine.compute(shells[s3], shells[s4], shells[s1], shells[s2]);
          const auto *buf_3412 = buf[0];
          if (buf_3412 == nullptr)
            continue; // if all integrals screened out, skip to next quartet

            auto curshelloffset_i = 0U;
            auto curshelloffset_j = 0U;
            for(auto x=s3range_start;x<s3;x++) curshelloffset_i += AO_tiles[x];
            for(auto x=s4range_start;x<s4;x++) curshelloffset_j += AO_tiles[x];

            auto dimi =  curshelloffset_i + AO_tiles[s3];
            auto dimj =  curshelloffset_j + AO_tiles[s4];

        

        for (size_t f3 = 0; f3 != n3; ++f3) {
          auto bf3 = shell2bf[s3]+f3;
          for (size_t f4 = 0; f4 != n4; ++f4) {
            auto bf4 = shell2bf[s4]+f4;

            auto f3412 = f3*n4*n12 + f4*n12 + ind12;
            auto x = buf_3412[f3412];
            int64_t prevgaso = ga_shell_offset + f3*n4+f4;

            int64_t lo = prevgaso, hi = prevgaso+1, ld = -1;
            TensorType to_buf = 0;

            for (auto icount = 0U; icount != count; ++icount) {
                int prevc_ga = cvecs[icount].ga_handle();
                NGA_Get64(prevc_ga, &lo, &hi, (void*)(&to_buf), &ld);
                x -= to_buf *delems[icount];
            //   x -= vuvbuf[(f3+curshelloffset_i)*bd1+f4+curshelloffset_j]*delems[icount];
            }
            auto vtmp = x/sqrt(max);
            auto boffset = (f3+curshelloffset_i)*bd1+f4+curshelloffset_j;
            cbuf[boffset] = vtmp;
            dibuf[boffset] -= vtmp*vtmp;
          }
        }


          // for(size_t i = curshelloffset_i; i < dimi; i++) {
          // for(size_t j = curshelloffset_j; j < dimj; j++) {
          //     auto f3 = i - curshelloffset_i;
          //     auto f4 = j - curshelloffset_j;
          //     auto f3412 = f3*n4*n12 + f4*n12 + ind12;
          //     auto x = buf_3412[f3412];
          //     for (auto icount = 0U; icount != count; ++icount) {
          //       x -= vuvbuf[(icount*bd0+f3+curshelloffset_i)*bd1+f4+curshelloffset_j]*delems[icount];
          //     }

          //     auto vtmp = x/sqrt(max);
          //     vuvbuf[(count*bd0+i)*bd1+j] = vtmp;
          //     dibuf[i*bd1+j] -= vtmp*vtmp;
          //   }
          // }
          
          } //s4
        } //s3

        // cout << "ga_shell_offset = " << ga_shell_offset << endl;



        int64_t lo = ga_shell_offset, hi = ga_shell_offset+size -1, ld = -1;
        // cout << "lo,hi = " << lo << "," << hi << endl;
        const void* from_buf = cbuf.data();
        NGA_Put64(tmp_ga, &lo, &hi, const_cast<void*>(from_buf), &ld);

        // l_t1 = std::chrono::high_resolution_clock::now();

        // CholVuv_tamm.put(blockid, vuvbuf);
        DiagInt_tamm.put({bi0,bi1}, dibuf);

        // l_t2 = std::chrono::high_resolution_clock::now();
        // l_time =
        //     std::chrono::duration_cast<std::chrono::duration<double>>((l_t2 - l_t1)).count();

        // put_time+=l_time;

  };

  block_for(ec, tmp1(), update_columns);
  // delems.clear();
  
  count += 1;

  std::tie(max,maxblockid,bfuv) = max_element(ec, DiagInt_tamm());
  bfu = bfuv[0];
  bfv = bfuv[1];

  auto cd_t2 = std::chrono::high_resolution_clock::now();
  auto cd_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((cd_t2 - cd_t1)).count();
  if(rank == 0 && debug) {
    std::cout << "----------Time taken for iter " << citer << ": " << cd_time << " secs---------\n";
  
  
    // std::cout << "Time taken for puts : " << put_time << " secs\n";
    // std::cout << "Time taken for gets: " << get_time << " secs\n";
  }

  if(count <= 2){
  print_tensor(tmp);
  print_tensor(DiagInt_tamm);
  cout << "max,bfu,bfv == " << max << "," << bfu << "," << bfv << endl;
  }


} while (max > diagtol && count < max_cvecs);  

  Tensor<TensorType>::deallocate(DiagInt_tamm);

  chol_count = count;

  if(rank == 0) {
    cout << "... End Cholesky Decomposition " << endl;
    cout << "\nNumber of cholesky vectors = " << chol_count << endl;
  }
 //end CD

  hf_t2 = std::chrono::high_resolution_clock::now();
  hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
  if(rank == 0) std::cout << "\nTime taken for Cholesky Decomposition: " << hf_time << " secs\n";


  #if 0
    // Start SVD

    std::vector<std::pair<Matrix,Eigen::RowVectorXd>> evs(count);
    // std::vector<Eigen::RowVectorXd> evec(count);

    Matrix testev = Matrix::Random(nao,nao);
    for (auto i=0U;i<count;i++) {
      Matrix Vuvi(nao, nao);
      for (auto x = 0U; x < nao; x++) {
      for (auto y = 0U; y<nao; y++)
        Vuvi(x, y) = CholVuv(x, y, i);
      }

      Eigen::SelfAdjointEigenSolver<Matrix> es(Vuvi);
      evs[i] = std::make_pair(es.eigenvectors(),es.eigenvalues());
      evec[i] = es.eigenvalues();
    }

    //End SVD
  #endif

  IndexSpace CIp{range(0, count)};
  TiledIndexSpace tCIp{CIp, count};
  auto [cindexp] = tCIp.labels<1>("all");

  Tensor<TensorType> CholVuv_opt{tCIp, tAO, tAO};
  Tensor<TensorType>::allocate(&ec, CholVuv_opt);

    auto lambdacv = [&](const IndexVector& bid){

          Tensor<TensorType> tensor = CholVuv_opt;
          const IndexVector blockid =
          internal::translate_blockid(bid, tensor());

          const tamm::TAMM_SIZE dsize = tensor.block_size(blockid);
          std::vector<TensorType> dbuf(dsize);

          IndexVector cvpriv = {0,blockid[1],blockid[2]};
          const tamm::TAMM_SIZE ssize = CholVuv_tamm.block_size(cvpriv);
          std::vector<TensorType> sbuf(ssize);

          CholVuv_tamm.get(cvpriv, sbuf);
              
          for(auto i = 0U; i < dsize; i++) 
              dbuf[i] = sbuf[i];

          tensor.put(blockid, dbuf);
    };

    block_for(ec, CholVuv_opt(), lambdacv);


  Tensor<TensorType>::deallocate(CholVuv_tamm);

  hf_t1 = std::chrono::high_resolution_clock::now();

  Tensor<TensorType> CholVpv_tamm{tCIp,tMO,tAO};
  Tensor<TensorType>::allocate(&ec, CholVpv_tamm);

  //CD: CholVpv(p, fv, icount) += CTiled(fu, p) * CholVuv(fu, fv, icount);
  Scheduler{ec}
  (CholVpv_tamm(cindexp,pmo,mu) = CTiled_tamm(nu, pmo) * CholVuv_opt(cindexp, nu, mu)).execute();
  
  hf_t2 = std::chrono::high_resolution_clock::now();
  hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
  if(rank == 0) std::cout << "\nTime taken for computing CholVpv: " << hf_time << " secs\n";

  Tensor<TensorType>::deallocate(CholVuv_opt);
  
    // From evs to Vpsigma
    // for (auto p = 0U; p < v2dim; p++) {
    //   for (auto fu = 0U; fu != nao; ++fu) {
    //     for (auto fs = 0U; fs != nao; ++fs) {
    //       for (auto icount = 0U; icount != count; ++icount) {
    //         CholVpv(p, fv, icount) += CTiled(fu, p) * CholVuv(fu, fv, icount);
    //         // SVD: Vpsigma(p, fs, icount) += CTiled(fu, p) * evs.at(icount).first(fu, fs);
    //       }
    //     }
    //   }
    // }

  hf_t1 = std::chrono::high_resolution_clock::now();

  Tensor<TensorType> CholVpr_tamm{{tCIp,tMO,tMO},{SpinPosition::ignore,SpinPosition::upper,SpinPosition::lower}};
  Tensor<TensorType>::allocate(&ec, CholVpr_tamm);
  Scheduler{ec}
  (CholVpr_tamm(cindexp,pmo,rmo) += CTiled_tamm(mu, rmo) * CholVpv_tamm(cindexp, pmo, mu)).execute();

    hf_t2 = std::chrono::high_resolution_clock::now();
    hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
    if(rank == 0) std::cout << "\nTime taken for computing CholVpr: " << hf_time << " secs\n";

    Tensor<TensorType>::deallocate(CholVpv_tamm,CTiled_tamm);//,CholVuv_tamm);

  //  IndexSpace CIp{range(0, count)};
  //   TiledIndexSpace tCIp{CIp, 1};
  //   auto [cindexp] = tCIp.labels<1>("all");
  
  // Tensor<TensorType> CholVpr_opt{{tMO,tMO,tCIp},{1,1}};
  // Tensor<TensorType>::allocate(&ec, CholVpr_opt);

  // auto lambdacv = [&](const IndexVector& bid){
  //     const IndexVector blockid =
  //     internal::translate_blockid(bid, CholVpr_opt());

  //     auto block_dims   = CholVpr_opt.block_dims(blockid);
  //     auto block_offset = CholVpr_opt.block_offsets(blockid);

  //     const tamm::TAMM_SIZE dsize = CholVpr_opt.block_size(blockid);
  //     std::vector<TensorType> dbuf(dsize);

  //     const tamm::TAMM_SIZE ssize = CholVpr_tamm.block_size({blockid[0],blockid[1],0});
  //     std::vector<TensorType> sbuf(ssize);

  //     CholVpr_tamm.get({blockid[0],blockid[1],0}, sbuf);
          
  //     TAMM_SIZE c = 0;
  //     for(auto i = block_offset[0]; i < block_offset[0] + block_dims[0];
  //         i++) {
  //         for(auto j = block_offset[1]; j < block_offset[1] + block_dims[1];
  //             j++) {
  //               for(auto k = block_offset[2]; k < block_offset[2] + block_dims[2];
  //             k++, c++) {
  //         dbuf[c] = sbuf[c];
  //             }
  //         }
  //     }
  //     CholVpr_opt.put(blockid, dbuf);
  // };

  // block_for(ec, CholVpr_opt(), lambdacv);

  // Tensor<TensorType>::deallocate(CholVpr_tamm);


    // // From evs to CholVpr
    // for (auto p = 0U; p < v2dim; p++) {
    //   for (auto r = 0U; r < v2dim; r++) {
    //     if (spin_t(p) != spin_t(r)) {
    //       continue;
    //     }

    //     for (auto fs = 0U; fs != nao; ++fs) {
    //       for (auto icount = 0U; icount != count; ++icount) {
    //         CholVpr(p, r, icount) += CTiled(fs, r) * CholVpv(p, fs, icount);
    //         // SVD: CholVpr(p, r, icount) += Vpsigma(p,fs,icount)*Vpsigma(r,fs,icount)*evs.at(icount).second(fs);
    //       }
    //     }
    //   }
    // }

  //Need to explicitly create an array that contains the permutation
  //Eigen::array<std::ptrdiff_t, 4> psqr_shuffle = {{0, 3, 2, 1}};

//  Eigen::Tensor<double, 4, Eigen::RowMajor> V_psqr = V_prqs.shuffle(psqr_shuffle);
  // Eigen::Tensor<double, 4, Eigen::RowMajor> V_pqrs = V_prqs - V_psqr;
  libint2::finalize(); // done with libint
#endif

  IndexSpace CIp{range(0, count)};
  TiledIndexSpace tCIp{CIp, count};
  auto [cindexp] = tCIp.labels<1>("all");
  Tensor<TensorType> CholVpr_tamm{{tCIp,tMO,tMO},{SpinPosition::ignore,SpinPosition::upper,SpinPosition::lower}};
  Tensor<TensorType>::allocate(&ec, CholVpr_tamm);


  NGA_Destroy(g_chol);
  return CholVpr_tamm;
}

#endif //TAMM_CD_SVD_HPP_