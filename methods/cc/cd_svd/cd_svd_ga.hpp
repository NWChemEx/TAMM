
#pragma once

#include "scf/scf_main.hpp"
#include "tamm/eigen_utils.hpp"
#include "tamm/tamm.hpp"
#include "ga_over_upcxx.hpp"
#include "common/json_data.hpp"

using namespace tamm;
using TAMM_GA_SIZE = int64_t;

bool cd_debug = false;

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

#if 0
  int64_t ac_fetch_add(int ga_ac, int64_t index, int64_t amount) {
    auto ret = NGA_Read_inc64(ga_ac, &index, amount);
    return ret;
  }
#endif

std::tuple<TiledIndexSpace,TAMM_SIZE> setupMOIS(SystemData sys_data, bool triples=false, int nactv=0) {

    TAMM_SIZE n_occ_alpha = sys_data.n_occ_alpha;
    TAMM_SIZE n_occ_beta = sys_data.n_occ_beta;

    Tile tce_tile = sys_data.options_map.ccsd_options.tilesize;
    bool balance_tiles = sys_data.options_map.ccsd_options.balance_tiles;
    if(!triples) {
      if ((tce_tile < static_cast<Tile>(sys_data.nbf/10) || tce_tile < 50 || tce_tile > 100) && !sys_data.options_map.ccsd_options.force_tilesize) {
        tce_tile = static_cast<Tile>(sys_data.nbf/10);
        if(tce_tile < 50) tce_tile = 50; //50 is the default tilesize for CCSD.
        if(tce_tile > 100) tce_tile = 100; //100 is the max tilesize for CCSD.
        if(upcxx::rank_me()==0) std::cout << std::endl << "Resetting CCSD tilesize to: " << tce_tile << std::endl;
      }
    }
    else {
      balance_tiles = false;
      tce_tile = sys_data.options_map.ccsd_options.ccsdt_tilesize;
    }

    TAMM_SIZE nmo = sys_data.nmo;
    TAMM_SIZE n_vir_alpha = sys_data.n_vir_alpha;
    TAMM_SIZE n_vir_beta = sys_data.n_vir_beta;
    TAMM_SIZE nocc = sys_data.nocc;

    const TAMM_SIZE total_orbitals = nmo;

    // Construction of tiled index space MO
    TAMM_SIZE virt_alpha_int = nactv;
    TAMM_SIZE virt_beta_int = virt_alpha_int;
    TAMM_SIZE virt_alpha_ext = n_vir_alpha - nactv;
    TAMM_SIZE virt_beta_ext = total_orbitals - (nocc+nactv+n_vir_alpha);    
    IndexSpace MO_IS{range(0, total_orbitals),
                    {
                     {"occ", {range(0, nocc)}},
                     {"occ_alpha", {range(0, n_occ_alpha)}},
                     {"occ_beta", {range(n_occ_alpha, nocc)}},
                     {"virt", {range(nocc, total_orbitals)}},
                     {"virt_alpha", {range(nocc,nocc+n_vir_alpha)}},
                     {"virt_beta", {range(nocc+n_vir_alpha, total_orbitals)}},
                     {"virt_alpha_int", {range(nocc,nocc+nactv)}},
                     {"virt_beta_int", {range(nocc+n_vir_alpha, nocc+nactv+n_vir_alpha)}},
                     {"virt_int", {range(nocc,nocc+nactv), range(nocc+n_vir_alpha, nocc+nactv+n_vir_alpha)}},
                     {"virt_alpha_ext", {range(nocc+nactv,nocc+n_vir_alpha)}},
                     {"virt_beta_ext", {range(nocc+nactv+n_vir_alpha, total_orbitals)}},
                     {"virt_ext", {range(nocc+nactv,nocc+n_vir_alpha), range(nocc+nactv+n_vir_alpha, total_orbitals)}},
                    },
                     {
                      {Spin{1}, {range(0, n_occ_alpha), range(nocc,nocc+n_vir_alpha)}},
                      {Spin{2}, {range(n_occ_alpha, nocc), range(nocc+n_vir_alpha, total_orbitals)}} 
                     }
                     };

    std::vector<Tile> mo_tiles;
    
    if(!balance_tiles) {
      tamm::Tile est_nt = n_occ_alpha/tce_tile;
      tamm::Tile last_tile = n_occ_alpha%tce_tile;
      for (tamm::Tile x=0;x<est_nt;x++)mo_tiles.push_back(tce_tile);
      if(last_tile>0) mo_tiles.push_back(last_tile);
      est_nt = n_occ_beta/tce_tile;
      last_tile = n_occ_beta%tce_tile;
      for (tamm::Tile x=0;x<est_nt;x++) mo_tiles.push_back(tce_tile);
      if(last_tile>0) mo_tiles.push_back(last_tile);
      // est_nt = n_vir_alpha/tce_tile;
      // last_tile = n_vir_alpha%tce_tile;
      // for (tamm::Tile x=0;x<est_nt;x++) mo_tiles.push_back(tce_tile);
      // if(last_tile>0) mo_tiles.push_back(last_tile);
      est_nt = virt_alpha_int/tce_tile;
      last_tile = virt_alpha_int%tce_tile;
      for (tamm::Tile x=0;x<est_nt;x++) mo_tiles.push_back(tce_tile);
      if(last_tile>0) mo_tiles.push_back(last_tile);
      est_nt = virt_alpha_ext/tce_tile;
      last_tile = virt_alpha_ext%tce_tile;
      for (tamm::Tile x=0;x<est_nt;x++) mo_tiles.push_back(tce_tile);
      if(last_tile>0) mo_tiles.push_back(last_tile);
      // est_nt = n_vir_beta/tce_tile;
      // last_tile = n_vir_beta%tce_tile;
      // for (tamm::Tile x=0;x<est_nt;x++) mo_tiles.push_back(tce_tile);
      // if(last_tile>0) mo_tiles.push_back(last_tile);
      est_nt = virt_beta_int/tce_tile;
      last_tile = virt_beta_int%tce_tile;
      for (tamm::Tile x=0;x<est_nt;x++) mo_tiles.push_back(tce_tile);
      if(last_tile>0) mo_tiles.push_back(last_tile);
      est_nt = virt_beta_ext/tce_tile;
      last_tile = virt_beta_ext%tce_tile;    
      for (tamm::Tile x=0;x<est_nt;x++) mo_tiles.push_back(tce_tile);
      if(last_tile>0) mo_tiles.push_back(last_tile);
    }
    else {
      tamm::Tile est_nt = static_cast<tamm::Tile>(std::ceil(1.0 * n_occ_alpha / tce_tile));
      for (tamm::Tile x=0;x<est_nt;x++) mo_tiles.push_back(n_occ_alpha / est_nt + (x<(n_occ_alpha % est_nt)));

      est_nt = static_cast<tamm::Tile>(std::ceil(1.0 * n_occ_beta / tce_tile));
      for (tamm::Tile x=0;x<est_nt;x++) mo_tiles.push_back(n_occ_beta / est_nt + (x<(n_occ_beta % est_nt)));

      // est_nt = static_cast<tamm::Tile>(std::ceil(1.0 * n_vir_alpha / tce_tile));
      // for (tamm::Tile x=0;x<est_nt;x++) mo_tiles.push_back(n_vir_alpha / est_nt + (x<(n_vir_alpha % est_nt)));

      est_nt = static_cast<tamm::Tile>(std::ceil(1.0 * virt_alpha_int / tce_tile));
      for (tamm::Tile x=0;x<est_nt;x++) mo_tiles.push_back(virt_alpha_int / est_nt + (x<(virt_alpha_int % est_nt)));

      est_nt = static_cast<tamm::Tile>(std::ceil(1.0 * virt_alpha_ext / tce_tile));
      for (tamm::Tile x=0;x<est_nt;x++) mo_tiles.push_back(virt_alpha_ext / est_nt + (x<(virt_alpha_ext % est_nt)));      

      // est_nt = static_cast<tamm::Tile>(std::ceil(1.0 * n_vir_beta / tce_tile));
      // for (tamm::Tile x=0;x<est_nt;x++) mo_tiles.push_back(n_vir_beta / est_nt + (x<(n_vir_beta % est_nt)));

      est_nt = static_cast<tamm::Tile>(std::ceil(1.0 * virt_beta_int / tce_tile));
      for (tamm::Tile x=0;x<est_nt;x++) mo_tiles.push_back(virt_beta_int / est_nt + (x<(virt_beta_int % est_nt)));

      est_nt = static_cast<tamm::Tile>(std::ceil(1.0 * virt_beta_ext / tce_tile));
      for (tamm::Tile x=0;x<est_nt;x++) mo_tiles.push_back(virt_beta_ext / est_nt + (x<(virt_beta_ext % est_nt)));      
    }

    TiledIndexSpace MO{MO_IS, mo_tiles}; //{ova,ova,ovb,ovb}};

    return std::make_tuple(MO,total_orbitals);
}

void update_sysdata(SystemData& sys_data, TiledIndexSpace& MO) {
  const bool do_freeze = sys_data.n_frozen_core > 0 || sys_data.n_frozen_virtual > 0;
  TAMM_SIZE total_orbitals = sys_data.nmo;
  if(do_freeze) {
    sys_data.nbf -= (sys_data.n_frozen_core + sys_data.n_frozen_virtual);
    sys_data.n_occ_alpha -= sys_data.n_frozen_core;
    sys_data.n_vir_alpha -= sys_data.n_frozen_virtual;
    sys_data.n_occ_beta -= sys_data.n_frozen_core;
    sys_data.n_vir_beta -= sys_data.n_frozen_virtual;    
    sys_data.update();
    std::tie(MO,total_orbitals) = setupMOIS(sys_data);
  }
}

Matrix reshape_mo_matrix(SystemData sys_data, Matrix& emat) {

  const int noa = sys_data.n_occ_alpha;
  const int nob = sys_data.n_occ_beta;
  const int nva = sys_data.n_vir_alpha;
  const int nvb = sys_data.n_vir_beta;
  const int nocc = sys_data.nocc;
  const int N_eff = sys_data.nmo;

  const int n_frozen_core    = sys_data.n_frozen_core;
  const int n_frozen_virtual = sys_data.n_frozen_virtual;

  Matrix cvec(N_eff,N_eff);
  const int block2_off = 2*n_frozen_core+noa;
  const int block3_off = 2*n_frozen_core+nocc+n_frozen_virtual;
  const int last_block_off = block3_off+n_frozen_virtual+nva;

  cvec.block(0,0,noa,noa)        = emat.block(n_frozen_core,n_frozen_core,noa,noa);
  cvec.block(0,noa,noa,nob)      = emat.block(n_frozen_core,block2_off,noa,nob);
  cvec.block(0,nocc,noa,nva)     = emat.block(n_frozen_core,block3_off,noa,nva);
  cvec.block(0,nocc+nva,noa,nvb) = emat.block(n_frozen_core,last_block_off,noa,nvb);

  cvec.block(noa,0,nob,noa)        = emat.block(block2_off,n_frozen_core,nob,noa);
  cvec.block(noa,noa,nob,nob)      = emat.block(block2_off,block2_off,nob,nob);
  cvec.block(noa,nocc,nob,nva)     = emat.block(block2_off,block3_off,nob,nva);
  cvec.block(noa,nocc+nva,nob,nvb) = emat.block(block2_off,last_block_off,nob,nvb);

  cvec.block(nocc,0,nva,noa)        = emat.block(block3_off,n_frozen_core,nva,noa);
  cvec.block(nocc,noa,nva,nob)      = emat.block(block3_off,block2_off,nva,nob);    
  cvec.block(nocc,nocc,nva,nva)     = emat.block(block3_off,block3_off,nva,nva);  
  cvec.block(nocc,nocc+nva,nva,nvb) = emat.block(block3_off,last_block_off,nva,nvb);  

  cvec.block(nocc+nva,0,nvb,noa)        = emat.block(last_block_off,n_frozen_core,nvb,noa);
  cvec.block(nocc+nva,noa,nvb,nob)      = emat.block(last_block_off,block2_off,nvb,nob);    
  cvec.block(nocc+nva,nocc,nvb,nva)     = emat.block(last_block_off,block3_off,nvb,nva);  
  cvec.block(nocc+nva,nocc+nva,nvb,nvb) = emat.block(last_block_off,last_block_off,nvb,nvb);  
  emat.resize(0,0);

  return cvec;
}

template <typename TensorType>
Tensor<TensorType> cd_svd_ga(SystemData& sys_data, ExecutionContext& ec, TiledIndexSpace& tMO, TiledIndexSpace& tAO,
  TAMM_SIZE& chol_count, const TAMM_GA_SIZE max_cvecs, libint2::BasisSet& shells, Tensor<TensorType>& lcao) {

  using libint2::Atom;
  using libint2::Shell;
  using libint2::Engine;
  using libint2::Operator;

  double             diagtol        = sys_data.options_map.cd_options.diagtol;
  const tamm::Tile   itile_size     = sys_data.options_map.ccsd_options.itilesize;
  // const TAMM_GA_SIZE northo         = sys_data.nbf;
  const TAMM_GA_SIZE nao            = sys_data.nbf_orig;

  auto rank = ec.pg().rank().value();

  TAMM_GA_SIZE N = tMO("all").max_num_indices();

  Matrix lcao_eig(nao,N);
  lcao_eig.setZero();
  tamm_to_eigen_tensor(lcao,lcao_eig);
  TensorType *k_movecs_sorted = lcao_eig.data();

  //
  // Cholesky decomposition
  //
  if(rank==0) {
    cout << "Begin Cholesky Decomposition ... " << endl;
  }
  auto hf_t1 = std::chrono::high_resolution_clock::now();

  // Step A. Initialization
  int64_t iproc = rank;
  int64_t ndim  = 3;
  auto    nbf   = nao;
  int64_t count = 0; //Initialize chol vector count

  #ifdef CD_SVD_THROTTLE
    int64_t cd_nranks = std::abs(std::log10(diagtol)) * nbf; // max cores
    auto nnodes = ec.pg().num_nodes();
    auto ppn = ec.pg().ppn();
    int cd_nnodes     = cd_nranks/ppn;
    if(cd_nranks%ppn>0 || cd_nnodes==0) 
      cd_nnodes++;
    if(cd_nnodes > nnodes) 
      cd_nnodes = nnodes;
    cd_nranks = cd_nnodes * ppn;
    if(rank == 0)  cout << "Total # of mpi ranks used for Cholesky decomposition: " << cd_nranks << 
                   endl << "  --> Number of nodes, mpi ranks per node: " << cd_nnodes << ", " << ppn << endl;
  #endif

    // fprintf(stderr, "Rank %d cd_nranks=%ld nnodes=%ld ppn=%ld cd_nnodes=%ld iproc=%ld\n",
    //         upcxx::rank_me(), cd_nranks, nnodes, ppn, cd_nnodes, iproc);

  int64_t size_map;
  std::vector<int64_t> k_map;
  int ga_eltype = C_DBL; 

  auto create_map = [&] (auto& dims, auto& nblock) {
    std::vector<int64_t> k_map(size_map);
    auto mi=0;
    for (auto count_dim=0;count_dim<2;count_dim++){
      auto size_blk = dims[count_dim]/nblock[count_dim];
      for (auto i=0;i<nblock[count_dim];i++){
        k_map[mi] = size_blk*i;
        mi++;
      }
    }
    k_map[mi] = 0;
    return k_map;
  };

  upcxx::team& team_ref = upcxx::world();
  upcxx::team* team = &team_ref;

  #ifdef CD_SVD_THROTTLE
  const bool throttle_cd = upcxx::world().rank_n() > cd_nranks;

  if(iproc < cd_nranks) { //throttle  

  if(throttle_cd){
      //upcxx::persona_scope master_scope(master_mtx,
      //        upcxx::master_persona());
      team = new upcxx::team(team->split((team->rank_me() < cd_nranks ? 0 : upcxx::team::color_none), 0));
    }).wait();
  #endif

  int64_t dims[3] = {nbf,nbf,max_cvecs};
  int64_t chnk[3] = {-1,-1,max_cvecs};
  int nblock32[GA_MAX_DIM]; 
  int64_t nblock[GA_MAX_DIM]; 

  ga_over_upcxx *g_chol = new ga_over_upcxx(3, dims, chnk, *team);
  g_chol->zero();

  int64_t dims2[3] = {nbf,nbf, 1};
  int64_t chnk2[3] = {-1, -1, 1};
  
  //TODO: Check k_map;
  ga_over_upcxx* g_d = new ga_over_upcxx(3, dims2, chnk2, *team);
  ga_over_upcxx* g_r = new ga_over_upcxx(3, dims2, chnk2, *team);
  g_d->zero();

  auto shell2bf = map_shell_to_basis_function(shells);
  auto bf2shell = map_basis_function_to_shell(shells);

  // Step B. Compute the diagonal
  Engine engine(Operator::coulomb, max_nprim(shells), max_l(shells), 0);
  const auto &buf = engine.results();  

  for (size_t s1 = 0; s1 != shells.size(); ++s1) {
    auto bf1_first = shell2bf[s1]; // first basis function in this shell
    auto n1 = shells[s1].size();

    for (size_t s2 = 0; s2 != shells.size(); ++s2) {
      auto bf2_first = shell2bf[s2];
      auto n2 = shells[s2].size();

      if (g_d->coord_is_local(bf1_first, bf2_first, 0)) {
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
          int64_t ibflo[3] = {cd_ncast<size_t>(bf1_first),cd_ncast<size_t>(bf2_first), 0};
          int64_t ibfhi[3] = {cd_ncast<size_t>(bf1_first+n1-1),cd_ncast<size_t>(bf2_first+n2-1), 0};
          int64_t ld[3] = {cd_ncast<size_t>(n1), cd_ncast<size_t>(n2), 1};
          g_d->put(ibflo[0], ibflo[1], ibflo[2], ibfhi[0], ibfhi[1], ibfhi[2],
                  &k_eri[0], ld);
      } //if s2
    } //s2
  } //s1

  // Step C. Find the coordinates of the maximum element of the diagonal.
  int64_t indx_d0[3];
  TensorType val_d0;
  g_d->maximum(val_d0, indx_d0[0], indx_d0[1], indx_d0[2]);


  // Step D. Start the while loop
  while(val_d0 > diagtol && count < max_cvecs){

    g_r->zero();
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

    for (size_t s3 = 0; s3 != shells.size(); ++s3) {
      auto bf3_first = shell2bf[s3]; // first basis function in this shell
      auto n3 = shells[s3].size();

      for (decltype(s3) s4 = 0; s4 != shells.size(); ++s4) {
        auto bf4_first = shell2bf[s4];
        auto n4 = shells[s4].size();

        if (g_r->coord_is_local(bf3_first, bf4_first, 0)) {

            engine.compute(shells[s3], shells[s4], shells[s1], shells[s2]);
            const auto *buf_3412 = buf[0];
            if (buf_3412 == nullptr)
              continue; // if all integrals screened out, skip to next quartet

            std::vector<TensorType> k_eri(n3*n4);
            for (decltype(n3) f3 = 0; f3 != n3; ++f3) {
              for (decltype(n4) f4 = 0; f4 != n4; ++f4) {
                auto f3412 = f3*n4*n12 + f4*n12 + ind12;
                k_eri[f3*n4+f4] = buf_3412[f3412];
              }
            }
                
            int64_t ibflo[3] = {cd_ncast<size_t>(bf3_first),cd_ncast<size_t>(bf4_first), 0};
            int64_t ibfhi[3] = {cd_ncast<size_t>(bf3_first+n3-1),cd_ncast<size_t>(bf4_first+n4-1), 0};
            int64_t ld[3] = {cd_ncast<size_t>(n3), cd_ncast<size_t>(n4), 1}; //n3                  
            g_r->put(ibflo[0], ibflo[1], ibflo[2], ibfhi[0], ibfhi[1], ibfhi[2],
                    &k_eri[0], ld);
          } //if s4
        } //s4
    } //s3
    // g_r->print();
   
    {
        //upcxx::persona_scope master_scope(master_mtx,
        //        upcxx::master_persona());
        upcxx::barrier(*team);
    }

    // Step F. Update the residual
    int64_t lo_x[3]; // The lower limits of blocks
    lo_x[0] = indx_d0[0];
    lo_x[1] = indx_d0[1];
    lo_x[2] = 0;
    int64_t hi_x[3]; // The upper limits of blocks
    hi_x[0] = indx_d0[0];
    hi_x[1] = indx_d0[1];
    hi_x[2] = count; //count>0? count : 0;
    int64_t ld_x[3]; // The leading dims of blocks
    ld_x[0] = 1;
    ld_x[1] = 1;
    ld_x[2] = hi_x[2]+1;

    std::vector<TensorType> k_elems(max_cvecs);
    TensorType* k_row = &k_elems[0];
    g_chol->get(lo_x[0], lo_x[1], lo_x[2], hi_x[0], hi_x[1], hi_x[2], k_row,
            ld_x);

    auto g_r_iter = g_r->local_chunks_begin();
    auto g_r_end = g_r->local_chunks_end();
    auto g_chol_iter = g_chol->local_chunks_begin();
    auto g_chol_end = g_chol->local_chunks_end();

    while (g_r_iter != g_r_end && g_chol_iter != g_chol_end) {
        ga_over_upcxx_chunk *g_r_chunk = *g_r_iter;
        ga_over_upcxx_chunk *g_chol_chunk = *g_chol_iter;
        assert(g_r_chunk->same_coord(g_chol_chunk) &&
                g_r_chunk->same_size_or_smaller(g_chol_chunk));

        ga_over_upcxx_chunk_view g_chol_view = g_chol_chunk->local_view();
        ga_over_upcxx_chunk_view g_r_view = g_r_chunk->local_view();

        for(int64_t icount = 0; icount < count; icount++){
          for (int64_t i = 0; i < g_r_view.get_chunk_size(0); i++) {
              for (int64_t j = 0; j < g_r_view.get_chunk_size(1); j++) {
                  g_r_view.subtract(i, j, 0,
                          g_chol_view.read(i, j, icount) * k_row[icount]);
            }
          }
        }

        g_r_iter++;
        g_chol_iter++;
    }
    assert(g_r_iter == g_r_end && g_chol_iter == g_chol_end);
    // g_r->print();

    g_r_iter = g_r->local_chunks_begin();
    g_r_end = g_r->local_chunks_end();
    g_chol_iter = g_chol->local_chunks_begin();
    g_chol_end = g_chol->local_chunks_end();

    while (g_r_iter != g_r_end && g_chol_iter != g_chol_end) {
        ga_over_upcxx_chunk *g_r_chunk = *g_r_iter;
        ga_over_upcxx_chunk *g_chol_chunk = *g_chol_iter;
        assert(g_r_chunk->same_coord(g_chol_chunk) && g_r_chunk->same_size_or_smaller(g_chol_chunk));

        ga_over_upcxx_chunk_view g_r_view = g_r_chunk->local_view();
        ga_over_upcxx_chunk_view g_chol_view = g_chol_chunk->local_view();

        for(auto i = 0; i < g_r_view.get_chunk_size(0); i++) {
          for(auto j = 0; j < g_r_view.get_chunk_size(1); j++) {
              auto tmp = g_r_view.read(i, j, 0) / sqrt(val_d0);
              g_chol_view.write(i, j, count, tmp);
          }
        }
        g_r_iter++;
        g_chol_iter++;
    }
    assert(g_r_iter == g_r_end && g_chol_iter == g_chol_end);
    // g_chol->print(count);

    //Step H. Increment count
    count++;

    //Step I. Update the diagonal
    auto g_d_iter = g_d->local_chunks_begin();
    auto g_d_end = g_d->local_chunks_end();
    g_chol_iter = g_chol->local_chunks_begin();
    g_chol_end = g_chol->local_chunks_end();

    while (g_d_iter != g_d_end && g_chol_iter != g_chol_end) {
        ga_over_upcxx_chunk *g_d_chunk = *g_d_iter;
        ga_over_upcxx_chunk *g_chol_chunk = *g_chol_iter;
        assert(g_d_chunk->same_coord(g_chol_chunk) && g_d_chunk->same_size_or_smaller(g_chol_chunk));

        ga_over_upcxx_chunk_view g_chol_view = g_chol_chunk->local_view();
        ga_over_upcxx_chunk_view g_d_view = g_d_chunk->local_view();

        for(auto i = 0; i< g_d_view.get_chunk_size(0); i++) {
          for(auto j = 0; j< g_d_view.get_chunk_size(1); j++) {
            auto tmp = g_chol_view.read(i, j, count-1);
            g_d_view.subtract(i, j, 0, tmp*tmp);
          }
        }
        g_d_iter++;
        g_chol_iter++;
    }
    assert(g_d_iter == g_d_end && g_chol_iter == g_chol_end);
    // g_d->print();

    //Step J. Find the coordinates of the maximum element of the diagonal.
    g_d->maximum(val_d0, indx_d0[0], indx_d0[1], indx_d0[2]);
  }

  if (iproc == 0) cout << "Number of cholesky vectors = " << count << endl;
  g_r->destroy();
  g_d->destroy();

  auto hf_t2 = std::chrono::high_resolution_clock::now();
  auto hf_time = std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
  if(iproc == 0) std::cout << std::endl << "Time taken for cholesky decomp: " << hf_time << " secs" << endl;
 
  update_sysdata(sys_data, tMO);

  TAMM_GA_SIZE N_eff = tMO("all").max_num_indices();
 
  int64_t dimsmo[3];
  int64_t chnkmo[3];
  dimsmo[0] = N; dimsmo[1] = N; dimsmo[2] = count;
  chnkmo[0] = -1; chnkmo[1] = -1; chnkmo[2] = count;

  ga_over_upcxx *g_chol_mo = new ga_over_upcxx(3, dimsmo, chnkmo, *team);
  g_chol_mo->zero();

  std::vector<TensorType> k_pj(N*nbf);
  std::vector<TensorType> k_pq(N*N);

  std::vector<TensorType> k_ij(nbf*nbf);
  std::vector<TensorType> k_eval_r(nbf);
  
  #define DO_SVD 0
  #if DO_SVD
    auto svdtol = 1e-8; //TODO same as diagtol ?
  #endif 

  hf_t1 = std::chrono::high_resolution_clock::now();
  double cvpr_time = 0;

  atomic_counter_over_upcxx ga_ac(*team);

  int64_t taskcount = 0;
  int64_t next = ga_ac.fetch_add(1);
  /*
   * Necessary for progress. The atomic counter is stored on rank 0, and rank 0
   * may just proceed straight in to the loop below preventing others from
   * making progress in their fetch-adds?
   */
  {
      //upcxx::persona_scope master_scope(master_mtx,
      //        upcxx::master_persona());
      upcxx::barrier(*team);
  }

  const bool do_freeze = (sys_data.n_frozen_core > 0 || sys_data.n_frozen_virtual > 0);

  for(decltype(count) kk=0;kk<count;kk++) {
    if(next == taskcount) {

      int64_t lo_ao[3] = {0,0,kk};
      int64_t hi_ao[3] = {nbf-1,nbf-1,kk};
      int64_t ld_ao[3] = {nbf,nbf, 1};

      g_chol->get(lo_ao[0], lo_ao[1], lo_ao[2], hi_ao[0], hi_ao[1], hi_ao[2],
              &k_ij[0], ld_ao);

      #if DO_SVD
        //uplotri
        for(auto i=0;i<nbf;i++)
          for(auto j=i+1;j<nbf;j++)
            k_ij[i*nbf+j] = 0;

        //TODO
        LAPACKE_dsyevd(LAPACK_ROW_MAJOR,'V','L',(BLA_LAPACK_INT)nbf,
          &k_ij[0],(BLA_LAPACK_INT)nbf, &k_eval_r[0]);

        auto m = 0;
        for(auto i=0;i<nbf;i++){
          if(fabs(k_eval_r[i]) <= svdtol) continue;
          k_eval_r[m] = k_eval_r[i];
          for(auto x=0;x<nbf;x++)
            k_ij[m*nbf+x] = k_ij[i*nbf+x]; 
          m++;
        }

        std::vector<TensorType> k_ij_tmp(nbf*m);
          for(auto i=0;i<nbf;i++)
            for(auto j=0;j<m;j++)
              k_ij_tmp[i*m+j] = k_ij[j*nbf+i];

        g_num += m;
        std::vector<TensorType> k_pi(N*m);
        std::vector<TensorType> k_qj(N*m);
         
        blas::gemm(blas::Layout::RowMajor,blas::Op::Trans,blas::Op::NoTrans,N,m,nbf,
                    1.0,k_movecs_sorted,N,&k_ij[0],nbf,0,&k_pi[0],N);

        for(auto x=0;x<N*m;x++) k_qj[x] = k_pi[x]; 

        for(auto i=0;i<N;i++){
          auto sf = k_eval_r[i];
          for (auto j=0;j<m;j++)
            k_pi[i*m+j] *= sf;
        }

        blas::gemm(blas::Layout::RowMajor,blas::Op::NoTrans,blas::Op::Trans,N,N,m,
                1,&k_pi[0],N,&k_qj[0],N,0,&k_pq[0],N);

      #else

        //---------Two-Step-Contraction----
        auto cvpr_t1 = std::chrono::high_resolution_clock::now();
        blas::gemm(blas::Layout::RowMajor,blas::Op::Trans,blas::Op::NoTrans,N,nbf,nbf,
                    1,k_movecs_sorted,N,&k_ij[0],nbf,0,&k_pj[0],nbf);
        
        blas::gemm(blas::Layout::RowMajor,blas::Op::NoTrans,blas::Op::NoTrans,N,N,nbf,
                    1,&k_pj[0],nbf,k_movecs_sorted,N,0,&k_pq[0],N);
      
        auto cvpr_t2 = std::chrono::high_resolution_clock::now();
        cvpr_time   += std::chrono::duration_cast<std::chrono::duration<double>>((cvpr_t2 - cvpr_t1)).count();

      #endif

      int64_t lo_mo[3] = {0,0,kk};
      int64_t hi_mo[3] = {N_eff-1,N_eff-1,kk};
      int64_t ld_mo[2] = {N_eff,1};

      if(do_freeze) {
        Matrix emat = Eigen::Map<Matrix>(k_pq.data(),N,N);
        k_pq.clear();

        Matrix cvec = reshape_mo_matrix(sys_data,emat);

        k_pq.resize(N_eff*N_eff); 
        Eigen::Map<Matrix>(k_pq.data(), N_eff, N_eff) = cvec;
        cvec.resize(0,0);
      }

      g_chol_mo->put(lo_mo[0], lo_mo[1], lo_mo[2], hi_mo[0], hi_mo[1], hi_mo[2],
              &k_pq[0], ld_mo);
      next = ga_ac.fetch_add(1);
    }
    taskcount++;

    upcxx::progress();
  }

  {
      //upcxx::persona_scope master_scope(master_mtx,
      //        upcxx::master_persona());
      upcxx::barrier(*team);
  }

  g_chol->destroy();
  k_pj.clear();
  k_pq.clear();
  k_ij.clear();
  k_eval_r.clear(); k_eval_r.shrink_to_fit();

  hf_t2   = std::chrono::high_resolution_clock::now();
  hf_time = std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
  if(rank == 0) {
    std::cout << "Total Time for constructing CholVpr: " << hf_time   << " secs" << endl;
    std::cout << "    --> Time for 2-step contraction:   " << cvpr_time << " secs" << endl;
  }

  #ifdef CD_SVD_THROTTLE
    team = &team_ref;

  }//end throttle
  #endif

  ec.pg().barrier();
  #ifdef CD_SVD_THROTTLE
  ec.pg().broadcast(&count, 1, 0);
  #endif

  hf_t1 = std::chrono::high_resolution_clock::now();

  #ifdef CD_SVD_THROTTLE

  dimsmo[0] =  N; dimsmo[1] =  N; dimsmo[2] = count;
  chnkmo[0] = -1; chnkmo[1] = -1; chnkmo[2] = count;
 
  ga_over_upcxx *g_chol_mo_copy = new ga_over_upcxx(3, dimsmo, chnkmo, *team);
  g_chol_mo_copy->zero();

  if(iproc < cd_nranks) { //throttle  
    g_chol_mo->copy(g_chol_mo_copy);
    g_chol_mo->destroy();
  }

  ec.pg().barrier();
  #else
    ga_over_upcxx *g_chol_mo_copy = g_chol_mo;
  #endif

  IndexSpace CIp{range(0, count)};
  TiledIndexSpace tCIp{CIp, static_cast<tamm::Tile>(itile_size)}; 
  
  Tensor<TensorType> CholVpr_tamm{{tMO,tMO,tCIp},{SpinPosition::upper,SpinPosition::lower,SpinPosition::ignore}};
  Tensor<TensorType>::allocate(&ec, CholVpr_tamm);
  
  //convert g_chol_mo_copy to CholVpr_tamm
  auto lambdacv = [&](const IndexVector& bid){
    const IndexVector blockid = internal::translate_blockid(bid, CholVpr_tamm());

    auto block_dims   = CholVpr_tamm.block_dims(blockid);
    auto block_offset = CholVpr_tamm.block_offsets(blockid);

    const tamm::TAMM_SIZE dsize = CholVpr_tamm.block_size(blockid);

    int64_t lo[3] = {cd_ncast<size_t>(block_offset[0]), 
                     cd_ncast<size_t>(block_offset[1]), 
                     cd_ncast<size_t>(block_offset[2])};
    int64_t hi[3] = {cd_ncast<size_t>(block_offset[0] + block_dims[0]-1), 
                     cd_ncast<size_t>(block_offset[1] + block_dims[1]-1),
                     cd_ncast<size_t>(block_offset[2] + block_dims[2]-1)};
    int64_t ld[3] = {cd_ncast<size_t>(block_dims[0]),
                     cd_ncast<size_t>(block_dims[1]),
                     cd_ncast<size_t>(block_dims[2])};

    upcxx::progress();
    std::vector<TensorType> sbuf(dsize);
    g_chol_mo_copy->get(lo[0], lo[1], lo[2], hi[0], hi[1], hi[2], &sbuf[0], ld);

    CholVpr_tamm.put(blockid, sbuf);
    upcxx::progress();
  };

  block_for(ec, CholVpr_tamm(), lambdacv);

  g_chol_mo_copy->destroy();

  hf_t2   = std::chrono::high_resolution_clock::now();
  hf_time = std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
  if(rank == 0) std::cout << std::endl << "Time for ga_chol_mo -> CholVpr_tamm conversion: " << hf_time << " secs" << endl;

  ga_ac.destroy();
  #if 0
  
    Tensor<TensorType> CholVuv_opt{tAO, tAO, tCIp};
    Tensor<TensorType>::allocate(&ec, CholVuv_opt);

    hf_t1 = std::chrono::high_resolution_clock::now();

    // Contraction 1
    Tensor<TensorType> CholVpv_tamm{tMO,tAO,tCIp};
    Tensor<TensorType>::allocate(&ec, CholVpv_tamm);
    Scheduler{ec}(CholVpv_tamm(pmo,mu,cindexp) = CTiled_tamm(nu, pmo) * CholVuv_opt(nu, mu, cindexp)).execute();
    Tensor<TensorType>::deallocate(CholVuv_opt);

    hf_t2   = std::chrono::high_resolution_clock::now();
    hf_time = std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
    if(rank == 0) std::cout << std::endl << "Time taken for computing CholVpv: " << hf_time << " secs" << std::endl;

    //Contraction 2
    hf_t1 = std::chrono::high_resolution_clock::now();
  
    Tensor<TensorType> CholVpr_tamm{{tMO,tMO,tCIp},{SpinPosition::upper,SpinPosition::lower,SpinPosition::ignore}};
    Scheduler{ec}
      .allocate(CholVpr_tamm)
      (CholVpr_tamm(pmo,rmo,cindexp) += CTiled_tamm(mu, rmo) * CholVpv_tamm(pmo, mu,cindexp))
      .deallocate(CholVpv_tamm)
      .execute();
  
    hf_t2   = std::chrono::high_resolution_clock::now();
    hf_time = std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
    if(rank == 0) std::cout << std::endl << "Time taken for computing CholVpr: " << hf_time << " secs" << std::endl;

  #endif

  chol_count = count;
  return CholVpr_tamm;
}

