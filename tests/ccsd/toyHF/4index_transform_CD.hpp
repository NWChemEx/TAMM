
#ifndef TAMM_TESTS_IT_HPP_
#define TAMM_TESTS_IT_HPP_

#include "hartree_fock.hpp"

std::tuple<Tensor4D> four_index_transform(
  const uint64_t ndocc, const uint64_t nao, const uint64_t freeze_core,
  const uint64_t freeze_virtual, const Matrix& C, Matrix& F,
  libint2::BasisSet& shells, Tensor3D& CholVpr) {

    using libint2::Atom;
    using libint2::Shell;
    using libint2::Engine;
    using libint2::Operator;

    // 2-index transform
    cout << "\n\n** Number of electrons: " << ndocc << endl;

    //  cout << "\n\t C Matrix:\n";
    //  cout << C << endl;
    //
    //  cout << "\n\t F_AO Matrix:\n";
    //  cout << F << endl;

    // std::cout << "C Cols = " << C_cols << std::endl;
    std::cout << "nao, ndocc = " << nao << " : " << ndocc << std::endl;

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
  cout << "\n--------------------\n" << endl;
  cout << "Cholesky Decomposition" << endl;
  cout << "nao = " << nao << endl;

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

  // Generate bf to shell map
  auto shell2bf = map_shell_to_basis_function(shells);
  for (auto s1 = 0; s1 != shells.size(); ++s1) {
    auto bf1_first = shell2bf[s1]; // first basis function in this shell
    auto n1 = shells[s1].size();
    for (auto f1 = 0; f1 != n1; ++f1) {
      const auto bf1 = f1 + bf1_first;
      bf2shell(bf1) = s1;
    }
  }

  // Compute diagonals
  Engine engine(Operator::coulomb, max_nprim(shells), max_l(shells), 0);
  const auto &buf = engine.results();  

  for (auto s1 = 0; s1 != shells.size(); ++s1) {
    auto bf1_first = shell2bf[s1]; // first basis function in this shell
    auto n1 = shells[s1].size();

    for (auto s2 = 0; s2 != shells.size(); ++s2) {
      auto bf2_first = shell2bf[s2];
      auto n2 = shells[s2].size();

      engine.compute(shells[s1], shells[s2], shells[s1], shells[s2]);
      const auto *buf_1212 = buf[0];
      if (buf_1212 == nullptr)
        continue; // if all integrals screened out, skip to next quartet

      for (auto f1 = 0; f1 != n1; ++f1) {
        const auto bf1 = f1 + bf1_first;
        for (auto f2 = 0; f2 != n2; ++f2) {
          const auto bf2 = f2 + bf2_first;
          auto f1212 = f1*n2*n1*n2 + f2*n1*n2 + f1*n2 + f2;
          DiagInt(bf1, bf2) = buf_1212[f1212];
          cout << f1212 << " " << s1 << s2 << "(" << bf1 << bf2 << "|" << bf1 << bf2 << ") = " << DiagInt(bf1, bf2) << endl;
        }
      }
    }
  }

  auto max = 0.0;
  auto count = 0;
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

  auto diagtol = 1.0e-12; // tolerance for the max. diagonal
  do {

    if (count == 0) {
      // Find maximum in DiagInt 
      for (auto indi = 0; indi != nao; ++indi) {
        for (auto indj = 0; indj!= nao; ++indj) {
          //max = std::max(DiagInt(indi,indj), max);
          if (DiagInt(indi,indj) > max) {
            max = DiagInt(indi,indj);
            bfu = indi;
            bfv = indj;
          }
        }
      }
      //cout << "count = " << count << endl;
      cout << "max: (" << bfu << bfv << "|" << bfu << bfv << ") = " << max << sqrt(max) << endl;
      cout << "shells: " << bf2shell(bfu) << " " << bf2shell(bfv) << endl;
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

    for (auto s3 = 0; s3 != shells.size(); ++s3) {
      auto bf3_first = shell2bf[s3]; // first basis function in this shell
      auto n3 = shells[s3].size();

      for (auto s4 = 0; s4 != shells.size(); ++s4) {
        auto bf4_first = shell2bf[s4];
        auto n4 = shells[s4].size();

        engine.compute(shells[s3], shells[s4], shells[s1], shells[s2]);
        const auto *buf_3412 = buf[0];
        if (buf_3412 == nullptr)
          continue; // if all integrals screened out, skip to next quartet

        for (auto f3 = 0; f3 != n3; ++f3) {
          const auto bf3 = f3 + bf3_first;
          for (auto f4 = 0; f4 != n4; ++f4) {
            const auto bf4 = f4 + bf4_first;

            auto f3412 = f3*n4*n12 + f4*n12 + ind12;
            ScrCol(bf3, bf4) = buf_3412[f3412];
            for (auto icount = 0; icount != count; ++icount) {
              ScrCol(bf3, bf4) -= CholVuv(bf3,bf4,icount)*CholVuv(bfu,bfv,icount);
            }
            CholVuv(bf3, bf4, count) = ScrCol(bf3, bf4)/sqrt(max);
            //cout << bf3 << " " << bf4 << " " << bfu << " " << bfv << " " <<  buf_3412[f3412] << "\n" << endl;
          }
        }
      }
    }
  
    // Update diagonals
    for (auto indi = 0; indi != nao; ++indi) {
      for (auto indj = 0; indj!= nao; ++indj) {
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
    for (auto indi = 0; indi != nao; ++indi) {
      for (auto indj = 0; indj!= nao; ++indj) {
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
  
  cout << "# of Cholesky vectors: " << count << " max: " << max << endl;
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

  std::vector<std::pair<Matrix,Eigen::RowVectorXd>> evs(count);

  Matrix testev = Matrix::Random(nao,nao);
  for (auto i=0;i<count;i++) {
    Matrix Vuvi(nao, nao);
    for (auto x = 0; x < nao; x++) {
    for (auto y = 0; y<nao; y++)
      Vuvi(x, y) = CholVuv(x, y, i);
    }

    Eigen::SelfAdjointEigenSolver<Matrix> es(Vuvi);
    evs[i] = std::make_pair(es.eigenvectors(),es.eigenvalues());
  }


  //End SVD

  const auto v2dim =  2 * nao - 2 * freeze_core - 2 * freeze_virtual;
  Eigen::Tensor<double, 4, Eigen::RowMajor> V2_unfused(v2dim,v2dim,v2dim,v2dim);
  V2_unfused.setZero();

  const bool unfused_4index = true;
  const bool fully_fused_4index = false;

  Eigen::Tensor<double, 3, Eigen::RowMajor> Vpsigma(v2dim,nao,count);
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

  //cout << "\n\t spin_t\n";
  //cout << spin_t << endl;

  // Transform CholVuv to CholVpq
  if (unfused_4index) {
    Eigen::Tensor<double, 4, Eigen::RowMajor> V2_FromCholV(v2dim,v2dim,v2dim,v2dim);
    V2_FromCholV.setZero();

    // From evs to Vpsigma
    for (auto p = 0; p < v2dim; p++) {
      for (auto fu = 0; fu != nao; ++fu) {
        for (auto fs = 0; fs != nao; ++fs) {
          for (auto icount = 0; icount != count; ++icount) {
            //CD: CholVpv(p, fv, icount) += CTiled(fu, p) * CholVuv(fu, fv, icount);
            Vpsigma(p, fs, icount) += CTiled(fu, p) * evs.at(icount).first(fu, fs);
          }
        }
      }
    }

    // From evs to CholVpr
    for (auto p = 0; p < v2dim; p++) {
      for (auto r = 0; r < v2dim; r++) {
        if (spin_t(p) != spin_t(r)) {
          continue;
        }

        for (auto fs = 0; fs != nao; ++fs) {
          for (auto icount = 0; icount != count; ++icount) {
            //CD: CholVpr(p, r, icount) += CTiled(fv, r) * CholVpv(p, fv, icount);
            CholVpr(p, r, icount) += Vpsigma(p,fs,icount)*Vpsigma(r,fs,icount)*evs.at(icount).second(fs);
          }
        }
      }
    }

    // Form (pr|qs)
    for (auto p = 0; p < v2dim; p++) {
      for (auto r = 0; r < v2dim; r++) {
        if (spin_t(p) != spin_t(r)) {
          continue;
        }

        for (auto q = 0; q < v2dim; q++) {
          for (auto s = 0; s < v2dim; s++) {
            if (spin_t(q) != spin_t(s)) {
              continue;
            }

            for (auto icount = 0; icount != count; ++icount) {
              V2_unfused(p, r, q, s) += CholVpr(p, r, icount) * CholVpr(q, s, icount);
              //V2_FromCholV(p, r, q, s) += CholVpr(p, r, icount) * CholVpr(q, s, icount);
            }
            //cout << p << " " << r << " " << q << " " << s << " " << V2_unfused(p, r, q, s) << "\n" << endl;
          }
        }
      }
    }
  }

  cout << "\n--------------------\n" << endl;


  //Bo-ends----------------------------

  //Start 4-index transform
  //const auto n = nbasis(shells);

#if 0
  Eigen::Tensor<double, 4, Eigen::RowMajor> V2_fully_fused(v2dim,v2dim,v2dim,v2dim);
  V2_fully_fused.setZero();

  //V_prqs.setConstant(0.0d);
  //cout << t << endl;

  //cout << "num_basis: " << n << endl;
  // construct the electron repulsion integrals engine
  Engine engine(Operator::coulomb, max_nprim(shells), max_l(shells), 0);

  auto shell2bf = map_shell_to_basis_function(shells);

  const int n_alpha = ov_alpha_freeze;
  const int n_beta = ov_beta_freeze;
  // buf[0] points to the target shell set after every call  to engine.compute()
  const auto &buf = engine.results();
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

  cout << "\n\t spin_t\n";
  cout << spin_t << endl;

  if(unfused_4index) {
    Eigen::Tensor<double, 4, Eigen::RowMajor> I0(v2dim, v2dim, v2dim, v2dim);
    Eigen::Tensor<double, 4, Eigen::RowMajor> I1(v2dim, v2dim, v2dim, v2dim);
    Eigen::Tensor<double, 4, Eigen::RowMajor> I2(v2dim, v2dim, v2dim, v2dim);
    Eigen::Tensor<double, 4, Eigen::RowMajor> I3(v2dim, v2dim, v2dim, v2dim);
    I0.setZero();
    I1.setZero();
    I2.setZero();
    I3.setZero();

    //I0(s1, s2, s2, s4) = integral_function()
//#pragma omp parallel default(none), shared(shells, I0, shell2bf,engine,buf)
  //  {
      //#pragma omp for schedule(guided)
      for (size_t s1 = 0; s1 != shells.size(); ++s1) {
        auto bf1_first = shell2bf[s1]; // first basis function in this shell
        auto n1 = shells[s1].size();

        for (size_t s2 = 0; s2 != shells.size(); ++s2) {
          auto bf2_first = shell2bf[s2];
          auto n2 = shells[s2].size();

          // loop over shell pairs of the density matrix, {s3,s4}
          // again symmetry is not used for simplicity
          for (size_t s3 = 0; s3 != shells.size(); ++s3) {
            auto bf3_first = shell2bf[s3];
            auto n3 = shells[s3].size();

            for (size_t s4 = 0; s4 != shells.size(); ++s4) {
              auto bf4_first = shell2bf[s4];
              auto n4 = shells[s4].size();

              // Coulomb contribution to the Fock matrix is from {s1,s2,s3,s4} integrals
              engine.compute(shells[s1], shells[s2], shells[s3], shells[s4]);
              const auto *buf_1234 = buf[0];
              if (buf_1234 == nullptr)
                continue; // if all integrals screened out, skip to next quartet

              for (size_t f1 = 0, f1234 = 0; f1 != n1; ++f1) {
                const auto bf1 = f1 + bf1_first;
                for (size_t f2 = 0; f2 != n2; ++f2) {
                  const auto bf2 = f2 + bf2_first;
                  for (size_t f3 = 0; f3 != n3; ++f3) {
                    const auto bf3 = f3 + bf3_first;
                    for (size_t f4 = 0; f4 != n4; ++f4, ++f1234) {
                      const auto bf4 = f4 + bf4_first;
                      I0(bf1, bf2, bf3, bf4) += buf_1234[f1234];
                    }
                  }
                }
              }
            }
          }
        }
      }
//    }//omp parallel

    //I1(p, bf2, bf3, bf4) += CTiled(bf1, p) * I0(bf1, bf2, bf3, bf4);
//#pragma omp parallel default(none), firstprivate(v2dim), shared(engine,shells, CTiled, I0, I1, shell2bf)
  //  {
//#pragma  omp for schedule(guided)

      for (size_t p = 0; p < v2dim; p++) {
        for (size_t s1 = 0; s1 != shells.size(); ++s1) {
          auto bf1_first = shell2bf[s1]; // first basis function in this shell
          auto n1 = shells[s1].size();

          for (size_t s2 = 0; s2 != shells.size(); ++s2) {
            auto bf2_first = shell2bf[s2];
            auto n2 = shells[s2].size();

            // loop over shell pairs of the density matrix, {s3,s4}
            // again symmetry is not used for simplicity
            for (size_t s3 = 0; s3 != shells.size(); ++s3) {
              auto bf3_first = shell2bf[s3];
              auto n3 = shells[s3].size();

              for (size_t s4 = 0; s4 != shells.size(); ++s4) {
                auto bf4_first = shell2bf[s4];
                auto n4 = shells[s4].size();

                for (size_t f1 = 0; f1 != n1; ++f1) {
                  const auto bf1 = f1 + bf1_first;
                  for (size_t f2 = 0; f2 != n2; ++f2) {
                    const auto bf2 = f2 + bf2_first;
                    for (size_t f3 = 0; f3 != n3; ++f3) {
                      const auto bf3 = f3 + bf3_first;
                      for (size_t f4 = 0; f4 != n4; ++f4) {
                        const auto bf4 = f4 + bf4_first;
                        I1(p, bf2, bf3, bf4) += CTiled(bf1, p) * I0(bf1, bf2, bf3, bf4);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
  //  }//omp parallel

    //I2(p, r, bf3, bf4) += CTiled(bf2, r) * I1(p, bf2, bf3, bf4);
//#pragma omp parallel default(none), firstprivate(v2dim), shared(shells, CTiled, I1, I2, spin_t, shell2bf)
 //   {
//#pragma  omp for schedule(guided)
      for (size_t p = 0; p < v2dim; p++) {
        for (size_t r = 0; r < v2dim; r++) {
          if (spin_t(p) != spin_t(r)) {
            continue;
          }
          for (size_t s2 = 0; s2 != shells.size(); ++s2) {
            auto bf2_first = shell2bf[s2];
            auto n2 = shells[s2].size();

            // loop over shell pairs of the density matrix, {s3,s4}
            // again symmetry is not used for simplicity
            for (size_t s3 = 0; s3 != shells.size(); ++s3) {
              auto bf3_first = shell2bf[s3];
              auto n3 = shells[s3].size();

              for (size_t s4 = 0; s4 != shells.size(); ++s4) {
                auto bf4_first = shell2bf[s4];
                auto n4 = shells[s4].size();

                for (size_t f2 = 0; f2 != n2; ++f2) {
                  const auto bf2 = f2 + bf2_first;
                  for (size_t f3 = 0; f3 != n3; ++f3) {
                    const auto bf3 = f3 + bf3_first;
                    for (size_t f4 = 0; f4 != n4; ++f4) {
                      const auto bf4 = f4 + bf4_first;
                      I2(p, r, bf3, bf4) += CTiled(bf2, r) * I1(p, bf2, bf3, bf4);
                    }
                  }
                }
              }
            }
          }
        }
      }
  //  }//omp parallel


    //I3(p, r, q, bf4) += CTiled(bf3, q) * I1(p, r, bf3, bf4);
//#pragma omp parallel default(none), firstprivate(v2dim), shared(shells, CTiled, I3, I2, spin_t, shell2bf)
//    {
//#pragma  omp for schedule(guided)
      for (size_t p = 0; p < v2dim; p++) {
        for (size_t r = 0; r < v2dim; r++) {
          if (spin_t(p) != spin_t(r)) {
            continue;
          }
          for (size_t q = 0; q < v2dim; q++) {
            // loop over shell pairs of the density matrix, {s3,s4}
            // again symmetry is not used for simplicity
            for (size_t s3 = 0; s3 != shells.size(); ++s3) {
              auto bf3_first = shell2bf[s3];
              auto n3 = shells[s3].size();

              for (size_t s4 = 0; s4 != shells.size(); ++s4) {
                auto bf4_first = shell2bf[s4];
                auto n4 = shells[s4].size();

                for (size_t f3 = 0; f3 != n3; ++f3) {
                  const auto bf3 = f3 + bf3_first;
                  for (size_t f4 = 0; f4 != n4; ++f4) {
                    const auto bf4 = f4 + bf4_first;
                    I3(p, r, q, bf4) += CTiled(bf3, q) * I2(p, r, bf3, bf4);
                  }
                }
              }
            }
          }
        }
      }
   // }//omp parallel


    //V(p, r, q, s) += CTiled(bf4, s) * I1(p, r, q, bf4);
//#pragma omp parallel default(none), firstprivate(v2dim), shared(shells, CTiled, I3, V2_unfused, spin_t, shell2bf)
 //   {
//#pragma  omp for schedule(guided)
      for (size_t p = 0; p < v2dim; p++) {
        for (size_t r = 0; r < v2dim; r++) {
          if (spin_t(p) != spin_t(r)) {
            continue;
          }
          for (size_t q = 0; q < v2dim; q++) {
            for (size_t s = 0; s < v2dim; s++) {
              if (spin_t(q) != spin_t(s)) {
                continue;
              }
              // loop over shell pairs of the density matrix, {s3,s4}
              // again symmetry is not used for simplicity
              for (size_t s4 = 0; s4 != shells.size(); ++s4) {
                auto bf4_first = shell2bf[s4];
                auto n4 = shells[s4].size();

                for (size_t f4 = 0; f4 != n4; ++f4) {
                  const size_t bf4 = f4 + bf4_first;
                  V2_unfused(p, r, q, s) += CTiled(bf4, s) * I3(p, r, q, bf4);
                }
              }
            }
          }
        }
      }
    }
 // }//omp parallel

  if(fully_fused_4index) {
    for (size_t p = 0; p < v2dim; p++) {
      for (size_t r = 0; r < v2dim; r++) {

        if (spin_t(p) == spin_t(r)) {

          for (size_t q = 0; q < v2dim; q++) {
            for (size_t s = 0; s < v2dim; s++) {

              if (spin_t(q) == spin_t(s)) {

                // loop over shell pairs of the Fock matrix, {s1,s2}
                // Fock matrix is symmetric, but skipping it here for simplicity (see compute_2body_fock)
                for (size_t s1 = 0; s1 != shells.size(); ++s1) {

                  auto bf1_first = shell2bf[s1]; // first basis function in this shell
                  auto n1 = shells[s1].size();

                  for (size_t s2 = 0; s2 != shells.size(); ++s2) {

                    auto bf2_first = shell2bf[s2];
                    auto n2 = shells[s2].size();

                    // loop over shell pairs of the density matrix, {s3,s4}
                    // again symmetry is not used for simplicity
                    for (size_t s3 = 0; s3 != shells.size(); ++s3) {

                      auto bf3_first = shell2bf[s3];
                      auto n3 = shells[s3].size();

                      for (size_t s4 = 0; s4 != shells.size(); ++s4) {

                        auto bf4_first = shell2bf[s4];
                        auto n4 = shells[s4].size();

                        // Coulomb contribution to the Fock matrix is from {s1,s2,s3,s4} integrals
                        engine.compute(shells[s1], shells[s2], shells[s3], shells[s4]);
                        const auto *buf_1234 = buf[0];
                        if (buf_1234 == nullptr)
                          continue; // if all integrals screened out, skip to next quartet

                        // we don't have an analog of Eigen for tensors (yet ... see github.com/BTAS/BTAS, under development)
                        // hence some manual labor here:
                        // 1) loop over every integral in the shell set (= nested loops over basis functions in each shell)
                        // and 2) add contribution from each integral
                        for (size_t f1 = 0, f1234 = 0; f1 != n1; ++f1) {
                          const auto bf1 = f1 + bf1_first;
                          for (size_t f2 = 0; f2 != n2; ++f2) {
                            const auto bf2 = f2 + bf2_first;
                            for (size_t f3 = 0; f3 != n3; ++f3) {
                              const auto bf3 = f3 + bf3_first;
                              for (size_t f4 = 0; f4 != n4; ++f4, ++f1234) {
                                const auto bf4 = f4 + bf4_first;
                                //V4i(p*2*n+r,q*2*n+s) += CTiled(bf1,p) * CTiled(bf2,r) * CTiled(bf3,q) * CTiled(bf4,s) * buf_1234[f1234];
                                V2_fully_fused(p, r, q, s) +=
                                  CTiled(bf1, p) * CTiled(bf2, r) * CTiled(bf3, q) * CTiled(bf4, s) * buf_1234[f1234];
                              }
                            }
                          }
                        } //f1,f2,f3,f4

                      }
                    }
                  }
                } //s1,s2,s3,s4

              } //if qs
            } //s
          } //q
        } //if pr
      } //r
    }  //p
  }

  //correctness check
  if(fully_fused_4index && unfused_4index) {
    bool error = false;
    for (size_t p = 0; p < v2dim; p++) {
      for (size_t r = 0; r < v2dim; r++) {
        for (size_t q = 0; q < v2dim; q++) {
          for (size_t s = 0; s < v2dim; s++) {
            const double threshold = 1e-12;
            if(std::abs(V2_fully_fused(p, r, q, s) - V2_unfused(p, r, q, s)) > threshold) {
              std::cout<<"4index error. "<<p<<" "<<r<<" "<<q<<" "<<s<<" : "
                       <<V2_fully_fused(p, r, q, s)
                       <<"   "
                       <<V2_unfused(p, r, q, s)
                       <<"\n";
              error = true;
            }
          }
        }
      }
    }
    if(error) {
      assert(0); //crash here to debug
    }
  }

#endif

  //Need to explicitly create an array that contains the permutation
  //Eigen::array<std::ptrdiff_t, 4> psqr_shuffle = {{0, 3, 2, 1}};

//  Eigen::Tensor<double, 4, Eigen::RowMajor> V_psqr = V_prqs.shuffle(psqr_shuffle);
  // Eigen::Tensor<double, 4, Eigen::RowMajor> V_pqrs = V_prqs - V_psqr;

  Eigen::Tensor<double, 4, Eigen::RowMajor> A2(v2dim,v2dim,v2dim,v2dim);

  //cout << "\n\t V_pqrs tensor\n";


  if(unfused_4index) {
//#pragma omp parallel default(none), firstprivate(v2dim), shared(A2,V2_unfused)
    //{
//#pragma  omp for schedule(guided)
      for (size_t p = 0; p < v2dim; p++) {
        for (size_t q = 0; q < v2dim; q++) {
          for (size_t r = 0; r < v2dim; r++) {
            for (size_t s = 0; s < v2dim; s++) {
              A2(p, q, r, s) = V2_unfused(p, r, q, s) - V2_unfused(p, s, q, r);
            }
          }
        }
      }
    //}
  } 
  #if 0
  else if(fully_fused_4index) {
    for (size_t p = 0; p < v2dim; p++) {
      for (size_t q = 0; q < v2dim; q++) {
        for (size_t r = 0; r < v2dim; r++) {
          for (size_t s = 0; s < v2dim; s++) {
            A2(p, q, r, s)= V2_fully_fused(p,r,q,s) - V2_fully_fused(p,s,q,r);
          }
        }
      }
    }
  } 
  #endif
  else {
    assert(0); //one of two options must be selected
  }

  // for (auto p = 0; p < 2 * n; p++) {
  //     for (auto q = 0; q < 2 * n; q++) {
  //       for (auto r = 0; r < 2 * n; r++) {
  //       for (auto s = 0; s < 2 * n; s++) {
  //         cout << A2(p, q, r, s) << "\t" << p << " " << q << " " << r << " " << s << endl;
  //       }
  //     }
  //   }
  // }

  libint2::finalize(); // done with libint

  //std::cout << "MAX COEFF A2 == " << A2.maximum() << std::endl;

  //return CCSD inputs
  return std::make_tuple(A2);

}

#endif //TAMM_TESTS_IT_HPP_