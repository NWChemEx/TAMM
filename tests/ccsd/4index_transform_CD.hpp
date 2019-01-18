
#ifndef TAMM_TESTS_IT_HPP_
#define TAMM_TESTS_IT_HPP_

#include "HF/hartree_fock_eigen.hpp"

void four_index_transform(
  const uint64_t ndocc, const uint64_t nao, const uint64_t freeze_core,
  const uint64_t freeze_virtual, const Matrix& C, Matrix& F,
  libint2::BasisSet& shells, Tensor3D& CholVpr, std::vector<Eigen::RowVectorXd> &evec,
   Tensor3D &Vpsigma) {

  std::vector<std::pair<Matrix,Eigen::RowVectorXd>> evs;
    using libint2::Atom;
    using libint2::Shell;
    using libint2::Engine;
    using libint2::Operator;

    auto rank = GA_Nodeid();

    // 2-index transform
    if(rank == 0) cout << "\n\n** Number of electrons: " << ndocc << endl;

    //  cout << "\n\t C Matrix:\n";
    //  cout << C << endl;
    //
    //  cout << "\n\t F_AO Matrix:\n";
    //  cout << F << endl;

    // std::cout << "C Cols = " << C_cols << std::endl;
    if(rank == 0) std::cout << "nao, ndocc = " << nao << " : " << ndocc << std::endl;

    auto ov_alpha_freeze = ndocc - freeze_core;
    auto ov_beta_freeze  = nao - ndocc - freeze_virtual;

    // replicate horizontally
    Matrix C_2N(nao, 2 * nao);
    C_2N << C, C;
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

    F = CTiled.transpose() * (F * CTiled);

    //  cout << "\n\t F_MO Matrix:\n";
    //  cout << F << endl;

    //Start 4-index transform
  //const auto n = nbasis(shells);

  //
//-Bo-starts-----Cholesky-decomposition---------------
//
if(rank == 0) {
  cout << "\n--------------------\n" << endl;
  cout << "Cholesky Decomposition" << endl;
  cout << "nao = " << nao << endl;
}

  /* 
  DiagInt stores the diagonal integrals, i.e. (uv|uv)'s
  ScrCol temporarily stores all (uv|rs)'s with fixed r and s
  */
  
  Eigen::Tensor<double, 3, Eigen::RowMajor> CholVuv(nao,nao,8*nao);
  Eigen::Tensor<double, 2, Eigen::RowMajor> DiagInt(nao,nao);
  Eigen::Tensor<double, 2, Eigen::RowMajor> ScrCol(nao,nao);
  Eigen::Tensor<int, 1, Eigen::RowMajor> bf2shell(nao);
  
  CholVuv.setZero();
  DiagInt.setZero();
  ScrCol.setZero();
  bf2shell.setZero();

  libint2::initialize(false);

  // Generate bf to shell map
  auto shell2bf = map_shell_to_basis_function(shells);
  for (size_t s1 = 0; s1 != shells.size(); ++s1) {
    auto bf1_first = shell2bf[s1]; // first basis function in this shell
    auto n1 = shells[s1].size();
    for (size_t f1 = 0; f1 != n1; ++f1) {
      const auto bf1 = f1 + bf1_first;
      bf2shell(bf1) = s1;
    }
  }

  // Compute diagonals
  Engine engine(Operator::coulomb, max_nprim(shells), max_l(shells), 0);
  const auto &buf = engine.results();  

  for (size_t s1 = 0; s1 != shells.size(); ++s1) {
    auto bf1_first = shell2bf[s1]; // first basis function in this shell
    auto n1 = shells[s1].size();

    for (size_t s2 = 0; s2 != shells.size(); ++s2) {
      auto bf2_first = shell2bf[s2];
      auto n2 = shells[s2].size();

      engine.compute(shells[s1], shells[s2], shells[s1], shells[s2]);
      const auto *buf_1212 = buf[0];
      if (buf_1212 == nullptr)
        continue; // if all integrals screened out, skip to next quartet

      for (size_t f1 = 0; f1 != n1; ++f1) {
        const auto bf1 = f1 + bf1_first;
        for (size_t f2 = 0; f2 != n2; ++f2) {
          const auto bf2 = f2 + bf2_first;
          auto f1212 = f1*n2*n1*n2 + f2*n1*n2 + f1*n2 + f2;
          DiagInt(bf1, bf2) = buf_1212[f1212];
          // cout << f1212 << " " << s1 << s2 << "(" << bf1 << bf2 << "|" << bf1 << bf2 << ") = " << DiagInt(bf1, bf2) << endl;
        }
      }
    }
  }

  auto max = 0.0;
  auto count = 0U;
  auto bfu = 0; // basis function pair |uv) corresponding to 
  auto bfv = 0; //  maximun diagonal.
  auto s1 = bf2shell(bfu);
  auto n1 = shells[s1].size();
  auto s2 = bf2shell(bfv);
  auto n2 = shells[s2].size();
  auto n12 = n1*n2;
  auto f1 = bfu - shell2bf[s1];
  auto f2 = bfv - shell2bf[s2];
  auto ind12 = f1*n2 + f2;

  auto diagtol = 1.0e-6; // tolerance for the max. diagonal
  do {

    if (count == 0) {
      // Find maximum in DiagInt 
      for (size_t indi = 0; indi != nao; ++indi) {
        for (size_t indj = 0; indj!= nao; ++indj) {
          //max = std::max(DiagInt(indi,indj), max);
          if (DiagInt(indi,indj) > max) {
            max = DiagInt(indi,indj);
            bfu = indi;
            bfv = indj;
          }
        }
      }
      //cout << "count = " << count << endl;
      // cout << "max: (" << bfu << bfv << "|" << bfu << bfv << ") = " << max << sqrt(max) << endl;
      // cout << "shells: " << bf2shell(bfu) << " " << bf2shell(bfv) << endl;
    }
    
    // Compute all (**|uv)'s for given shells
    s1 = bf2shell(bfu);
    n1 = shells[s1].size();
    s2 = bf2shell(bfv);
    n2 = shells[s2].size();
    n12 = n1*n2;
    f1 = bfu - shell2bf[s1];
    f2 = bfv - shell2bf[s2];
    ind12 = f1*n2 + f2;

    for (size_t s3 = 0; s3 != shells.size(); ++s3) {
      auto bf3_first = shell2bf[s3]; // first basis function in this shell
      auto n3 = shells[s3].size();

      for (size_t s4 = 0; s4 != shells.size(); ++s4) {
        auto bf4_first = shell2bf[s4];
        auto n4 = shells[s4].size();

        engine.compute(shells[s3], shells[s4], shells[s1], shells[s2]);
        const auto *buf_3412 = buf[0];
        if (buf_3412 == nullptr)
          continue; // if all integrals screened out, skip to next quartet

        for (size_t f3 = 0; f3 != n3; ++f3) {
          const auto bf3 = f3 + bf3_first;
          for (size_t f4 = 0; f4 != n4; ++f4) {
            const auto bf4 = f4 + bf4_first;

            auto f3412 = f3*n4*n12 + f4*n12 + ind12;
            ScrCol(bf3, bf4) = buf_3412[f3412];
            for (auto icount = 0U; icount != count; ++icount) {
              ScrCol(bf3, bf4) -= CholVuv(bf3,bf4,icount)*CholVuv(bfu,bfv,icount);
            }
            CholVuv(bf3, bf4, count) = ScrCol(bf3, bf4)/sqrt(max);
            //cout << bf3 << " " << bf4 << " " << bfu << " " << bfv << " " <<  buf_3412[f3412] << "\n" << endl;
          }
        }
      }
    }
  
    // Update diagonals
    for (size_t indi = 0; indi != nao; ++indi) {
      for (size_t indj = 0; indj!= nao; ++indj) {
        //cout << indi << " " << indj << " " << DiagInt(indi, indj) << endl;
        //cout << CholVuv(indi, indj, count) << endl;
        DiagInt(indi, indj) -= CholVuv(indi, indj, count)*CholVuv(indi, indj, count);
        //cout << indi << " " << indj << " " << DiagInt(indi, indj) << endl;
      }
    }
    count += 1;
    //cout << "count = " << count << endl;

    // Find maximum in DiagInt 
    max = 0.0;
    for (size_t indi = 0; indi != nao; ++indi) {
      for (size_t indj = 0; indj!= nao; ++indj) {
        //max = std::max(DiagInt(indi,indj), max);
        if (DiagInt(indi,indj) > max) {
          max = DiagInt(indi,indj);
          bfu = indi;
          bfv = indj;
        }
      }
    }
    //cout << "max: (" << bfu << bfv << "|" << bfu << bfv << ") = " << max << " " << sqrt(max) << endl;
    //cout << "shells: " << bf2shell(bfu) << " " << bf2shell(bfv) << endl;

  } while (max > diagtol && count <= (8*nao)); // At most 8*ao CholVec's. For vast majority cases, this is way
                                              //   more than enough. For very large basis, it can be increased.
  
  // cout << "# of Cholesky vectors: " << count << " max: " << max << "(" << diagtol << ")" << endl;
  /*
  // reproduce ao2eint and compare it with libint results -- for test only!
  Eigen::Tensor<double, 4, Eigen::RowMajor> CholV2(nao,nao,nao,nao);
  CholV2.setZero();
  for (auto indu = 0; indu != nao; ++indu) {
    for (auto indv = 0; indv != nao; ++indv) {
      for (auto indr = 0; indr != nao; ++indr) {
        for (auto inds = 0; inds != nao; ++inds) {
          for (auto indc = 0; indc != count; ++indc) {
            CholV2 (indu,indv,indr,inds) += CholVuv(indu,indv,indc)*CholVuv(indr,inds,indc);
          }
          cout << indu << " " << indv << " " << indr << " " << inds << " " << CholV2 (indu,indv,indr,inds) << "\n" << endl;
        }
      }
    }
  }
  */

  //
//CD-ends----------------------------
//

  // Start SVD

  // std::vector<std::pair<Matrix,Eigen::RowVectorXd>> evs(count);
  evs.resize(count);
  evec.resize(count);

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

  const auto v2dim =  2 * nao - 2 * freeze_core - 2 * freeze_virtual;
  Vpsigma = Tensor3D(v2dim,nao,count);
  CholVpr = Tensor3D(v2dim,v2dim,count);
  Vpsigma.setZero();
  CholVpr.setZero();

  const int n_alpha = ov_alpha_freeze;
  const int n_beta = ov_beta_freeze;
  // buf[0] points to the target shell set after every call  to engine.compute()
  // const auto &buf = engine.results();
  Matrix spin_t = Matrix::Zero(1, 2 * nao - 2 * freeze_core - 2 * freeze_virtual);
  Matrix spin_1 = Matrix::Ones(1,n_alpha);
  Matrix spin_2 = Matrix::Constant(1,n_alpha,2);
  Matrix spin_3 = Matrix::Constant(1,n_beta,1);
  Matrix spin_4 = Matrix::Constant(1,n_beta,2);
  //spin_t << 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2; - water
  spin_t.block(0,0,1,n_alpha) = spin_1;
  spin_t.block(0,n_alpha,1,n_alpha) = spin_2;
  spin_t.block(0,2*n_alpha,1, n_beta) = spin_3;
  spin_t.block(0,2*n_alpha+n_beta,1, n_beta) = spin_4;

      // From evs to Vpsigma
    for (auto p = 0U; p < v2dim; p++) {
      for (auto fu = 0U; fu != nao; ++fu) {
        for (auto fs = 0U; fs != nao; ++fs) {
          for (auto icount = 0U; icount != count; ++icount) {
            //CD: CholVpv(p, fv, icount) += CTiled(fu, p) * CholVuv(fu, fv, icount);
            Vpsigma(p, fs, icount) += CTiled(fu, p) * evs.at(icount).first(fu, fs);
          }
        }
      }
    }

    // From evs to CholVpr
    for (auto p = 0U; p < v2dim; p++) {
      for (auto r = 0U; r < v2dim; r++) {
        if (spin_t(p) != spin_t(r)) {
          continue;
        }

        for (auto fs = 0U; fs != nao; ++fs) {
          for (auto icount = 0U; icount != count; ++icount) {
            //CD: CholVpr(p, r, icount) += CTiled(fv, r) * CholVpv(p, fv, icount);
            CholVpr(p, r, icount) += Vpsigma(p,fs,icount)*Vpsigma(r,fs,icount)*evs.at(icount).second(fs);
          }
        }
      }
    }

  //Bo-ends----------------------------

  //Need to explicitly create an array that contains the permutation
  //Eigen::array<std::ptrdiff_t, 4> psqr_shuffle = {{0, 3, 2, 1}};

//  Eigen::Tensor<double, 4, Eigen::RowMajor> V_psqr = V_prqs.shuffle(psqr_shuffle);
  // Eigen::Tensor<double, 4, Eigen::RowMajor> V_pqrs = V_prqs - V_psqr;
  libint2::finalize(); // done with libint

}

#endif //TAMM_TESTS_IT_HPP_