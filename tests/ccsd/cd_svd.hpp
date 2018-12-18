
#ifndef TAMM_CD_SVD_HPP_
#define TAMM_CD_SVD_HPP_

#include "HF/hartree_fock_tamm.hpp"
#include "tamm/eigen_utils.hpp"
#include "tamm/tamm.hpp"

using namespace tamm;
using TensorType = double;

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

Tensor<TensorType> cd_svd(ExecutionContext& ec, TiledIndexSpace& tMO, TiledIndexSpace& tAO,
  const TAMM_SIZE ndocc, const TAMM_SIZE nao, const TAMM_SIZE freeze_core,
  const TAMM_SIZE freeze_virtual, Tensor<TensorType> C_AO, Tensor<TensorType> F_AO,
  Tensor<TensorType> F_MO, TAMM_SIZE& chol_count, const tamm::Tile max_cvecs, double diagtol,
  libint2::BasisSet& shells) {

    using libint2::Atom;
    using libint2::Shell;
    using libint2::Engine;
    using libint2::Operator;

    auto rank = ec.pg().rank();

    IndexSpace AO{range(0, nao)};
    std::vector<unsigned int> AO_tiles;
    for(auto s : shells) AO_tiles.push_back(s.size());

    tamm::Tile est_ts = 0;
    std::vector<size_t> shell_tile_map;
    for(auto s=0U;s<shells.size();s++){
      est_ts += shells[s].size();
      if(est_ts>=30) {
        shell_tile_map.push_back(s); //shell id specifying tile boundary
        est_ts=0;
      }
    }
    if(est_ts>0){
    shell_tile_map.push_back(shells.size()-1);
    }

    auto [mu, nu, ku] = tAO.labels<3>("all");


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

    // auto [mup, nup, kup] = tAOt.labels<3>("all");

    auto [pmo, rmo] = tMO.labels<2>("all");

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

  auto hf_t2 = std::chrono::high_resolution_clock::now();

  double hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
  if(rank == 0) 
      cout << "\nTime for 2-index transform: " << hf_time << " secs\n";
    

  IndexSpace CI{range(0, max_cvecs)};
  TiledIndexSpace tCI{CI, max_cvecs};
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
          
          std::vector<TensorType> tbuf(n1*n2);
          for (size_t f1 = 0; f1 != n1; ++f1) {
          for (size_t f2 = 0; f2 != n2; ++f2) {
            auto f1212 = f1*n2*n1*n2 + f2*n1*n2 + f1*n2 + f2;
            tbuf[f1*n2+f2] = buf_1212[f1212];
          }
        }

          auto curshelloffset_i = 0U;
          auto curshelloffset_j = 0U;
          for(auto x=s1range_start;x<s1;x++) curshelloffset_i += AO_tiles[x];
          for(auto x=s2range_start;x<s2;x++) curshelloffset_j += AO_tiles[x];

          size_t c = 0;
          auto dimi =  curshelloffset_i + AO_tiles[s1];
          auto dimj =  curshelloffset_j + AO_tiles[s2];

          for(size_t i = curshelloffset_i; i < dimi; i++) {
          for(size_t j = curshelloffset_j; j < dimj; j++, c++) {
                  dbuf[i*bd1+j] = tbuf[c];
                }
          }
          }
        }
        tensor1e.put(blockid,dbuf);
  };
  block_for(ec, DiagInt_tamm(),compute_diagonals);


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

  TensorType max=0;
  std::vector<size_t> bfuv;
  IndexVector maxblockid;

  auto citer = 0;
  auto debug = false;

  do {

    auto cd_t1 = std::chrono::high_resolution_clock::now();

    citer++;
    // Find maximum in DiagInt 
    std::tie(max,maxblockid,bfuv) = max_element(ec, DiagInt_tamm());
    bfu = bfuv[0];
    bfv = bfuv[1];

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

    IndexVector di = {0,maxblockid[0],maxblockid[1]};
    const tamm::TAMM_SIZE dec = CholVuv_tamm.block_size(di);
    std::vector<TensorType> cvuvbuf(dec);
    CholVuv_tamm.get(di, cvuvbuf);
    auto block_dims_vuv   = CholVuv_tamm.block_dims(di);

   std::vector<TensorType> delems(count);
  //  auto depos = ind12*block_dims_vuv[0];

    for (auto icount = 0U; icount != count; ++icount) {
        delems[icount] = cvuvbuf[(icount*block_dims_vuv[1]+bfu)*
            block_dims_vuv[2]+bfv];
    }

    auto tensor1e = CholVuv_tamm;

  auto update_columns = [&](const IndexVector& blockid) {

        auto bi0 = blockid[1];
        auto bi1 = blockid[2];

        const TAMM_SIZE size = tensor1e.block_size(blockid);
        auto block_dims   = tensor1e.block_dims(blockid);
        std::vector<TensorType> vuvbuf(size);
        std::vector<TensorType> dibuf(DiagInt_tamm.block_size({bi0,bi1}));

        auto bd0 = block_dims[1];
        auto bd1 = block_dims[2];

        CholVuv_tamm.get(blockid, vuvbuf);
        DiagInt_tamm.get({bi0,bi1}, dibuf);

        auto s3range_start = 0l;
        auto s3range_end = shell_tile_map[bi0];
        if (bi0>0) s3range_start = shell_tile_map[bi0-1]+1;
        
        for (Index s3 = s3range_start; s3 <= s3range_end; ++s3) {
          auto n3 = shells[s3].size();

        auto s4range_start = 0l;
        auto s4range_end = shell_tile_map[bi1];
        if (bi1>0) s4range_start = shell_tile_map[bi1-1]+1;

          for (Index s4 = s4range_start; s4 <= s4range_end; ++s4) {

          // if(s4>s3){
          //   auto s2spl = obs_shellpair_list[s4];
          //   if(std::find(s2spl.begin(),s2spl.end(),s3) == s2spl.end()) continue;
          // }
          // else{
          //   auto s2spl = obs_shellpair_list[s3];
          //   if(std::find(s2spl.begin(),s2spl.end(),s4) == s2spl.end()) continue;
          // }
          
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

        std::vector<TensorType> cbuf(n3*n4);

        for (size_t f3 = 0; f3 != n3; ++f3) {
          // const auto bf3 = f3 + bf3_first;
          for (size_t f4 = 0; f4 != n4; ++f4) {
            // const auto bf4 = f4 + bf4_first;

            auto f3412 = f3*n4*n12 + f4*n12 + ind12;
            auto x = buf_3412[f3412];
            for (auto icount = 0U; icount != count; ++icount) {
              // x -= CholVuv(bf3,bf4,icount)*delems[icount];
              //x -= tbuf[(f3*n4+f4)+icount]*delems[icount];
              x -= vuvbuf[(icount*bd0+f3+curshelloffset_i)*bd1+f4+curshelloffset_j]*delems[icount];
              // cout << "x, delems[" << icount << "] = " << x << ", " << delems[icount] << endl;
            }
            // CholVuv(bf3, bf4, count) = x/sqrt(max);
            auto vtmp = x/sqrt(max);
            cbuf[f3*n4+f4] = vtmp;
          }
        }

          size_t c = 0;
          for(size_t i = curshelloffset_i; i < dimi; i++) {
          for(size_t j = curshelloffset_j; j < dimj; j++, c++) {
                  auto dval = cbuf[c];
                  vuvbuf[(count*bd0+i)*bd1+j] = dval;
                  dibuf[i*bd1+j] -= dval*dval;
                }
          }
          
        } //s4
        } //s3

        CholVuv_tamm.put(blockid, vuvbuf);
        DiagInt_tamm.put({bi0,bi1}, dibuf);
  };

  block_for(ec, CholVuv_tamm(), update_columns);
  delems.clear();
  
  count += 1;

} while (max > diagtol && count <= max_cvecs);  
  Tensor<TensorType>::deallocate(DiagInt_tamm);

  chol_count = count;

  if(rank == 0) {
    cout << "... End Cholesky Decomposition " << endl;
    cout << "\nNumber of cholesky vectors = " << chol_count << endl;
  }
 //end CD

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

          auto block_dims   = tensor.block_dims(blockid);
          auto block_offset = tensor.block_offsets(blockid);

          const tamm::TAMM_SIZE dsize = tensor.block_size(blockid);
          std::vector<TensorType> dbuf(dsize);

          IndexVector cvpriv = {0,blockid[1],blockid[2]};
          const tamm::TAMM_SIZE ssize = CholVuv_tamm.block_size(cvpriv);
          std::vector<TensorType> sbuf(ssize);

          CholVuv_tamm.get(cvpriv, sbuf);
              
          TAMM_SIZE c = 0;
          for(auto i = block_offset[0]; i < block_offset[0] + block_dims[0];
              i++) {
              for(auto j = block_offset[1]; j < block_offset[1] + block_dims[1];j++) {
                for(auto k = block_offset[2]; k < block_offset[2] + block_dims[2];k++,c++) {
              dbuf[c] = sbuf[c];
              }
          }
          tensor.put(blockid, dbuf);
          }
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

  return CholVpr_tamm;
}

#endif //TAMM_CD_SVD_HPP_