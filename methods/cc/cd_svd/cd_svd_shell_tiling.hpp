
#pragma once

#include "scf/scf_main.hpp"
#include "tamm/eigen_utils.hpp"
#include "tamm/tamm.hpp"

using namespace tamm;
using TensorType = double;

Tensor<TensorType> cd_svd_ao(ExecutionContext& ec, TiledIndexSpace& tMO, TiledIndexSpace& tAOt,
  const TAMM_SIZE ndocc, const TAMM_SIZE nao, const TAMM_SIZE freeze_core,
  const TAMM_SIZE freeze_virtual, Tensor<TensorType> C_AO, Tensor<TensorType> F_AO,
  Tensor<TensorType> F_MO, TAMM_SIZE& chol_count, const tamm::Tile max_cvecs, double diagtol,
  libint2::BasisSet& shells) {

    using libint2::Atom;
    using libint2::Shell;
    using libint2::Engine;
    using libint2::Operator;

    auto rank = ec.pg().rank();


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

    auto [mup, nup, kup] = tAOt.labels<3>("all");

    auto [pmo, rmo] = tMO.labels<2>("all");

    // 2-index transform

    auto hf_t1 = std::chrono::high_resolution_clock::now();

    Tensor<TensorType> CTiled_tamm{tAOt,tMO};
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

      cout << "F_MO dims = " << F.rows() << "," << F.cols()  << endl;
    }

#ifdef USE_UPCXX
    upcxx::barrier();
#else
    GA_Sync();
#endif

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
  Tensor<TensorType> DiagInt_tamm{tAOt, tAOt};
  // Tensor<TensorType> DiagInt_tamm{tAO, tAO};
  Tensor<TensorType>::allocate(&ec, DiagInt_tamm);

  // Compute diagonals
  Engine engine(Operator::coulomb, max_nprim(shells), max_l(shells), 0);
  const auto &buf = engine.results();  

  auto compute_diagonals = [&](const IndexVector& blockid) {
    auto s1 = blockid[0];
      // auto bf1_first = shell2bf[s1]; // first basis function in this shell
      auto n1 = shells[s1].size();

      auto s2 = blockid[1];
        // auto bf2_first = shell2bf[s2];
        auto n2 = shells[s2].size();

        engine.compute(shells[s1], shells[s2], shells[s1], shells[s2]);
        const auto *buf_1212 = buf[0];
        if (buf_1212 == nullptr)
          return; // if all integrals screened out, skip to next quartet

        std::vector<TensorType> tbuf(n1*n2);

        for (size_t f1 = 0; f1 != n1; ++f1) {
          // const auto bf1 = f1 + bf1_first;
          for (size_t f2 = 0; f2 != n2; ++f2) {
            // const auto bf2 = f2 + bf2_first;
            auto f1212 = f1*n2*n1*n2 + f2*n1*n2 + f1*n2 + f2;
            tbuf[f1*n2+f2] = buf_1212[f1212];
            
          }
        }
        DiagInt_tamm.put(blockid,tbuf);
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
  
  Tensor<TensorType> CholVuv_tamm{tCI, tAOt, tAOt};
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

    // Compute all (**|uv)'s for given shells
    s1 = maxblockid[0]; //bf2shell(bfu);
    n1 = shells[s1].size();
    s2 = maxblockid[1]; //bf2shell(bfv);
    n2 = shells[s2].size();
    n12 = n1*n2;
    f1 = bfu - shell2bf[s1];
    f2 = bfv - shell2bf[s2];
    ind12 = f1*n2 + f2;

    auto init_t1 = std::chrono::high_resolution_clock::now();

    IndexVector di = {0,s1,s2};
    const tamm::TAMM_SIZE dec = CholVuv_tamm.block_size(di);
    std::vector<TensorType> vuvbuf(dec);
    CholVuv_tamm.get(di, vuvbuf);
    auto block_dims_vuv   = CholVuv_tamm.block_dims(di);

   std::vector<TensorType> delems(count);
   auto depos = ind12*block_dims_vuv[0];
   for (auto icount = 0U; icount != count; ++icount) 
      delems[icount] = vuvbuf[(icount*n1+f1)*n2+f2];
     //delems[icount] = CholVuv(bfu,bfv,icount);
    //  delems[icount] = vuvbuf[depos+icount];
  // if(count > 0) CholVuv_tamm.get({bfu,bfv,count},delems);


    auto init_t2 = std::chrono::high_resolution_clock::now();

    if(rank==0 && debug) {
      cout << "time for computing delems = " << 
      std::chrono::duration_cast<std::chrono::duration<double>>((init_t2 - init_t1)).count() << "secs" << endl;
    }

  auto update_columns = [&](const IndexVector& blockid) {
      auto s3 = blockid[1];
      // auto bf3_first = shell2bf[s3]; // first basis function in this shell
      auto n3 = shells[s3].size();

        auto s4 = blockid[2];
        // auto bf4_first = shell2bf[s4];
        auto n4 = shells[s4].size();

        std::vector<TensorType> tbuf(n3*n4*max_cvecs);
        std::vector<TensorType> dibuf(n3*n4);

        auto get_t1 = std::chrono::high_resolution_clock::now();

        CholVuv_tamm.get({blockid[0],s3,s4}, tbuf);

         auto get_t2 = std::chrono::high_resolution_clock::now();

        if(rank==0 && debug) {
          cout << "time for get chol-vuv = " << 
          std::chrono::duration_cast<std::chrono::duration<double>>((get_t2 - get_t1)).count() << "secs" << endl;
        }

        DiagInt_tamm.get({s3,s4}, dibuf);

        engine.compute(shells[s3], shells[s4], shells[s1], shells[s2]);
        const auto *buf_3412 = buf[0];
        if (buf_3412 == nullptr)
          return; // if all integrals screened out, skip to next quartet

        auto loop_t1 = std::chrono::high_resolution_clock::now();

        for (size_t f3 = 0; f3 != n3; ++f3) {
          // const auto bf3 = f3 + bf3_first;
          for (size_t f4 = 0; f4 != n4; ++f4) {
            // const auto bf4 = f4 + bf4_first;

            auto f3412 = f3*n4*n12 + f4*n12 + ind12;
            auto x = buf_3412[f3412];
            for (auto icount = 0U; icount != count; ++icount) {
              // x -= CholVuv(bf3,bf4,icount)*delems[icount];
              //x -= tbuf[(f3*n4+f4)+icount]*delems[icount];
              x -= tbuf[(icount*n3+f3)*n4+f4]*delems[icount];
            }
            // CholVuv(bf3, bf4, count) = x/sqrt(max);
            auto vtmp = x/sqrt(max);
            dibuf[f3*n4+f4] -=  vtmp*vtmp;
            tbuf[(count*n3+f3)*n4+f4] = vtmp;

            //cout << bf3 << " " << bf4 << " " << bfu << " " << bfv << " " <<  buf_3412[f3412] << endl << endl;
          }
        }

        auto loop_t2 = std::chrono::high_resolution_clock::now();

        if(rank==0 && debug) {
          cout << "loop time = " <<  
          std::chrono::duration_cast<std::chrono::duration<double>>((loop_t2 - loop_t1)).count() << "secs" << endl;
        }

        DiagInt_tamm.put({s3,s4}, dibuf);

        auto put_t1 = std::chrono::high_resolution_clock::now();

        CholVuv_tamm.put({blockid[0],s3,s4}, tbuf);

        auto put_t2 = std::chrono::high_resolution_clock::now();

        if(rank==0 && debug) {
          cout << "time for put chol-vuv = " << 
          std::chrono::duration_cast<std::chrono::duration<double>>((put_t2 - put_t1)).count() << "secs" << endl;
        }
  };

  block_for(ec, CholVuv_tamm(), update_columns);
  delems.clear();
  
  count += 1;

  auto cd_t2 = std::chrono::high_resolution_clock::now();

  if(rank==0 && debug) {
    cout << "time for CD iter " << citer << " = " << 
    std::chrono::duration_cast<std::chrono::duration<double>>((cd_t2 - cd_t1)).count() << "secs" << endl;
    cout << "----------------------------------------------------------\n";
  }

    //cout << "max: (" << bfu << bfv << "|" << bfu << bfv << ") = " << max << " " << sqrt(max) << endl;
    //cout << "shells: " << bf2shell(bfu) << " " << bf2shell(bfv) << endl;

} while (max > diagtol && count < max_cvecs);  
  Tensor<TensorType>::deallocate(DiagInt_tamm);

  chol_count = count;

  hf_t2 = std::chrono::high_resolution_clock::now();

  hf_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
  if(rank == 0) {
    cout << "... End Cholesky Decomposition" << endl;
    cout << "\nNumber of cholesky vectors = " << chol_count << endl;
    cout << "Time taken for Cholesky Decomposition: " << hf_time << " secs\n";
  }

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

  Tensor<TensorType> CholVuv_opt{tCIp, tAOt, tAOt};
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

  Tensor<TensorType> CholVpv_tamm{tCIp,tMO,tAOt};
  Tensor<TensorType>::allocate(&ec, CholVpv_tamm);

  //CD: CholVpv(p, fv, icount) += CTiled(fu, p) * CholVuv(fu, fv, icount);
  Scheduler{ec}
  (CholVpv_tamm(cindexp,pmo,mup) = CTiled_tamm(nup, pmo) * CholVuv_opt(cindexp, nup, mup)).execute();
  
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
  (CholVpr_tamm(cindexp,pmo,rmo) += CTiled_tamm(mup, rmo) * CholVpv_tamm(cindexp, pmo, mup)).execute();

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
