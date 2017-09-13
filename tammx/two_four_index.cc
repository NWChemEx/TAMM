#include "hartree_fock.h"

std::tuple<Tensor4D> two_four_index_transform(const int ndocc, const int noa, const Matrix &C, Matrix &F, libint2::BasisSet &shells){

  using libint2::Atom;
  using libint2::Shell;
  using libint2::Engine;
  using libint2::Operator;

  // 2-index transform
  const int num_electrons = ndocc;
  cout << "\n\n** Number of electrons: " << num_electrons << endl;

  cout << "\n\t C Matrix:\n";
  cout << C << endl;

  cout << "\n\t F_AO Matrix:\n";
  cout << F << endl;

  const int C_rows = C.rows();
  const int C_cols = C.cols();

  std::cout << "C Cols = " << C_cols << std::endl;
  std::cout << "noa, num_electrons = " << noa << " : " << num_electrons << std::endl;

  // replicate horizontally
  Matrix C_2N(C_rows, 2 * C_cols);
  C_2N << C, C;
  //cout << "\n\t C_2N Matrix:\n";
  //cout << C_2N << endl;

  const int b_rows = noa;
  Matrix C_noa = C_2N.block(0, 0,b_rows, num_electrons);
  cout << "\n\t C occupied alpha:\n";
  cout << C_noa << endl;

  //Matrix C_nva = C_2N.block<b_rows, b_rows - num_electrons>(0, num_electrons);
  Matrix C_nva = C_2N.block(0, num_electrons,b_rows, b_rows - num_electrons);
  cout << "\n\t C virtual alpha:\n";
  cout << C_nva << endl;

  //Matrix C_nob = C_2N.block<b_rows, num_electrons>(0, C_cols);
  Matrix C_nob = C_2N.block(0, C_cols,b_rows, num_electrons);
  cout << "\n\t C occupied beta:\n";
  cout << C_nob << endl;

//  Matrix C_nvb = C_2N.block<b_rows, b_rows - num_electrons>(0, num_electrons + C_cols);
  Matrix C_nvb = C_2N.block(0, num_electrons + C_cols,b_rows, b_rows - num_electrons);
  cout << "\n\t C virtual beta:\n";
  cout << C_nvb << endl;

  // For now C_noa = C_nob and C_nva = C_nvb
  Matrix CTiled(C_rows, 2 * C_cols);
  CTiled << C_noa, C_nob, C_nva, C_nvb;

  cout << "\n\t CTiled Matrix = [C_noa C_nob C_nva C_nvb]:\n";
  cout << CTiled << endl;

  F = CTiled.transpose() * (F * CTiled);

  cout << "\n\t F_MO Matrix:\n";
  cout << F << endl;

  //Start 4-index transform
  const auto n = nbasis(shells);
  Eigen::Tensor<double, 4, Eigen::RowMajor> V2_unfused(2 * n, 2 * n, 2 * n, 2 * n);
  V2_unfused.setZero();

  Eigen::Tensor<double, 4, Eigen::RowMajor> V2_fully_fused(2 * n, 2 * n, 2 * n, 2 * n);
  V2_fully_fused.setZero();

  const bool unfused_4index = true;
  const bool fully_fused_4index = false;

  //V_prqs.setConstant(0.0d);
  //cout << t << endl;

  //cout << "num_basis: " << n << endl;
  // construct the electron repulsion integrals engine
  Engine engine(Operator::coulomb, max_nprim(shells), max_l(shells), 0);

  auto shell2bf = map_shell_to_basis_function(shells);

  // buf[0] points to the target shell set after every call  to engine.compute()
  const auto &buf = engine.results();
  Matrix spin_t = Matrix::Zero(1, 2 * n);
  spin_t << 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2;

  cout << "\n\t spin_t\n";
  cout << spin_t << endl;


  if(unfused_4index)  {
    Eigen::Tensor<double, 4, Eigen::RowMajor> I0(2 * n, 2 * n, 2 * n, 2 * n);
    Eigen::Tensor<double, 4, Eigen::RowMajor> I1(2 * n, 2 * n, 2 * n, 2 * n);
    Eigen::Tensor<double, 4, Eigen::RowMajor> I2(2 * n, 2 * n, 2 * n, 2 * n);
    Eigen::Tensor<double, 4, Eigen::RowMajor> I3(2 * n, 2 * n, 2 * n, 2 * n);
    I0.setZero();
    I1.setZero();
    I2.setZero();
    I3.setZero();

    //I0(s1, s2, s2, s4) = integral_function()
    for (auto s1 = 0; s1 != shells.size(); ++s1) {
      auto bf1_first = shell2bf[s1]; // first basis function in this shell
      auto n1 = shells[s1].size();

      for (auto s2 = 0; s2 != shells.size(); ++s2) {
        auto bf2_first = shell2bf[s2];
        auto n2 = shells[s2].size();

        // loop over shell pairs of the density matrix, {s3,s4}
        // again symmetry is not used for simplicity
        for (auto s3 = 0; s3 != shells.size(); ++s3) {
          auto bf3_first = shell2bf[s3];
          auto n3 = shells[s3].size();

          for (auto s4 = 0; s4 != shells.size(); ++s4) {
            auto bf4_first = shell2bf[s4];
            auto n4 = shells[s4].size();

            // Coulomb contribution to the Fock matrix is from {s1,s2,s3,s4} integrals
            engine.compute(shells[s1], shells[s2], shells[s3], shells[s4]);
            const auto *buf_1234 = buf[0];
            if (buf_1234 == nullptr)
              continue; // if all integrals screened out, skip to next quartet

            for (auto f1 = 0, f1234 = 0; f1 != n1; ++f1) {
              const auto bf1 = f1 + bf1_first;
              for (auto f2 = 0; f2 != n2; ++f2) {
                const auto bf2 = f2 + bf2_first;
                for (auto f3 = 0; f3 != n3; ++f3) {
                  const auto bf3 = f3 + bf3_first;
                  for (auto f4 = 0; f4 != n4; ++f4, ++f1234) {
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

    //I1(p, bf2, bf3, bf4) += CTiled(bf1, p) * I0(bf1, bf2, bf3, bf4);
    for (auto p = 0; p < 2 * n; p++) {
      for (auto s1 = 0; s1 != shells.size(); ++s1) {
        auto bf1_first = shell2bf[s1]; // first basis function in this shell
        auto n1 = shells[s1].size();

        for (auto s2 = 0; s2 != shells.size(); ++s2) {
          auto bf2_first = shell2bf[s2];
          auto n2 = shells[s2].size();

          // loop over shell pairs of the density matrix, {s3,s4}
          // again symmetry is not used for simplicity
          for (auto s3 = 0; s3 != shells.size(); ++s3) {
            auto bf3_first = shell2bf[s3];
            auto n3 = shells[s3].size();

            for (auto s4 = 0; s4 != shells.size(); ++s4) {
              auto bf4_first = shell2bf[s4];
              auto n4 = shells[s4].size();

              for (auto f1 = 0; f1 != n1; ++f1) {
                const auto bf1 = f1 + bf1_first;
                for (auto f2 = 0; f2 != n2; ++f2) {
                  const auto bf2 = f2 + bf2_first;
                  for (auto f3 = 0; f3 != n3; ++f3) {
                    const auto bf3 = f3 + bf3_first;
                    for (auto f4 = 0; f4 != n4; ++f4) {
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

    //I2(p, r, bf3, bf4) += CTiled(bf2, r) * I1(p, bf2, bf3, bf4);
    for (auto p = 0; p < 2 * n; p++) {
      for (auto r = 0; r < 2 * n; r++) {
        if(spin_t(p)  != spin_t(r)) {
          continue;
        }
        for (auto s2 = 0; s2 != shells.size(); ++s2) {
          auto bf2_first = shell2bf[s2];
          auto n2 = shells[s2].size();

          // loop over shell pairs of the density matrix, {s3,s4}
          // again symmetry is not used for simplicity
          for (auto s3 = 0; s3 != shells.size(); ++s3) {
            auto bf3_first = shell2bf[s3];
            auto n3 = shells[s3].size();

            for (auto s4 = 0; s4 != shells.size(); ++s4) {
              auto bf4_first = shell2bf[s4];
              auto n4 = shells[s4].size();

              for (auto f2 = 0; f2 != n2; ++f2) {
                const auto bf2 = f2 + bf2_first;
                for (auto f3 = 0; f3 != n3; ++f3) {
                  const auto bf3 = f3 + bf3_first;
                  for (auto f4 = 0; f4 != n4; ++f4) {
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

    //I3(p, r, q, bf4) += CTiled(bf3, q) * I1(p, r, bf3, bf4);
    for (auto p = 0; p < 2 * n; p++) {
      for (auto r = 0; r < 2 * n; r++) {
        if(spin_t(p)  != spin_t(r)) {
          continue;
        }
        for (auto q = 0; q < 2 * n; q++) {
          // loop over shell pairs of the density matrix, {s3,s4}
          // again symmetry is not used for simplicity
          for (auto s3 = 0; s3 != shells.size(); ++s3) {
            auto bf3_first = shell2bf[s3];
            auto n3 = shells[s3].size();

            for (auto s4 = 0; s4 != shells.size(); ++s4) {
              auto bf4_first = shell2bf[s4];
              auto n4 = shells[s4].size();

              for (auto f3 = 0; f3 != n3; ++f3) {
                const auto bf3 = f3 + bf3_first;
                for (auto f4 = 0; f4 != n4; ++f4) {
                  const auto bf4 = f4 + bf4_first;
                  I3(p, r, q, bf4) += CTiled(bf3, q) * I2(p, r, bf3, bf4);
                }
              }
            }
          }
        }
      }
    }

    //V(p, r, q, s) += CTiled(bf4, s) * I1(p, r, q, bf4);
    for (auto p = 0; p < 2 * n; p++) {
      for (auto r = 0; r < 2 * n; r++) {
        if(spin_t(p)  != spin_t(r)) {
          continue;
        }
        for (auto q = 0; q < 2 * n; q++) {
          for (auto s = 0; s < 2 * n; s++) {
            if (spin_t(q) != spin_t(s)) {
              continue;
            }
            // loop over shell pairs of the density matrix, {s3,s4}
            // again symmetry is not used for simplicity
            for (auto s4 = 0; s4 != shells.size(); ++s4) {
              auto bf4_first = shell2bf[s4];
              auto n4 = shells[s4].size();

              for (auto f4 = 0; f4 != n4; ++f4) {
                const auto bf4 = f4 + bf4_first;
                V2_unfused(p, r, q, s) += CTiled(bf4, s) * I3(p, r, q, bf4);
              }
            }
          }
        }
      }
    }
  }
  if(fully_fused_4index) {
    for (auto p = 0; p < 2 * n; p++) {
      for (auto r = 0; r < 2 * n; r++) {

        if (spin_t(p) == spin_t(r)) {

          for (auto q = 0; q < 2 * n; q++) {
            for (auto s = 0; s < 2 * n; s++) {

              if (spin_t(q) == spin_t(s)) {

                // loop over shell pairs of the Fock matrix, {s1,s2}
                // Fock matrix is symmetric, but skipping it here for simplicity (see compute_2body_fock)
                for (auto s1 = 0; s1 != shells.size(); ++s1) {

                  auto bf1_first = shell2bf[s1]; // first basis function in this shell
                  auto n1 = shells[s1].size();

                  for (auto s2 = 0; s2 != shells.size(); ++s2) {

                    auto bf2_first = shell2bf[s2];
                    auto n2 = shells[s2].size();

                    // loop over shell pairs of the density matrix, {s3,s4}
                    // again symmetry is not used for simplicity
                    for (auto s3 = 0; s3 != shells.size(); ++s3) {

                      auto bf3_first = shell2bf[s3];
                      auto n3 = shells[s3].size();

                      for (auto s4 = 0; s4 != shells.size(); ++s4) {

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
                        for (auto f1 = 0, f1234 = 0; f1 != n1; ++f1) {
                          const auto bf1 = f1 + bf1_first;
                          for (auto f2 = 0; f2 != n2; ++f2) {
                            const auto bf2 = f2 + bf2_first;
                            for (auto f3 = 0; f3 != n3; ++f3) {
                              const auto bf3 = f3 + bf3_first;
                              for (auto f4 = 0; f4 != n4; ++f4, ++f1234) {
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
    for (auto p = 0; p < 2 * n; p++) {
      for (auto r = 0; r < 2 * n; r++) {
        for (auto q = 0; q < 2 * n; q++) {
          for (auto s = 0; s < 2 * n; s++) {
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

  //Need to explicitly create an array that contains the permutation
  //Eigen::array<std::ptrdiff_t, 4> psqr_shuffle = {{0, 3, 2, 1}};

//  Eigen::Tensor<double, 4, Eigen::RowMajor> V_psqr = V_prqs.shuffle(psqr_shuffle);
  // Eigen::Tensor<double, 4, Eigen::RowMajor> V_pqrs = V_prqs - V_psqr;

  Eigen::Tensor<double, 4, Eigen::RowMajor> A2(2 * n, 2 * n, 2 * n, 2 * n);

  //cout << "\n\t V_pqrs tensor\n";


  if(unfused_4index) {
    for (auto p = 0; p < 2 * n; p++) {
      for (auto q = 0; q < 2 * n; q++) {
        for (auto r = 0; r < 2 * n; r++) {
          for (auto s = 0; s < 2 * n; s++) {
            A2(p, q, r, s)= V2_unfused(p,r,q,s) - V2_unfused(p,s,q,r);
          }
        }
      }
    }
  } else if(fully_fused_4index) {
    for (auto p = 0; p < 2 * n; p++) {
      for (auto q = 0; q < 2 * n; q++) {
        for (auto r = 0; r < 2 * n; r++) {
          for (auto s = 0; s < 2 * n; s++) {
            A2(p, q, r, s)= V2_fully_fused(p,r,q,s) - V2_fully_fused(p,s,q,r);
          }
        }
      }
    }
  } else {
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

  //return CCSD inputs
  return std::make_tuple(A2);

}
