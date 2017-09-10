/*
 *  Copyright (C) 2004-2017 Edward F. Valeev
 *
 *  This file is part of Libint.
 *
 *  Libint is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  Libint is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Libint.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

//COMPILE: g++ -Ofast -std=c++14 -o hf hartree_fock.cc -I$LIBINT/include -I$EIGEN/include/eigen3 -L$LIBINT/lib -lint2

// standard C++ headers
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <tuple>
#include <functional>

// Eigen matrix algebra library
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/CXX11/Tensor>

// Libint Gaussian integrals library
#include <libint2.hpp>
#include <libint2/basis.h>

// #include "tammx/tammx.h"

//#define EIGEN_USE_BLAS

using std::string;
using std::cout;
using std::cerr;
using std::endl;


using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Tensor4D = Eigen::Tensor<double, 4, Eigen::RowMajor>;

// import dense, dynamically sized Matrix type from Eigen;
// this is a matrix with row-major storage (http://en.wikipedia.org/wiki/Row-major_order)
// to meet the layout of the integrals returned by the Libint integral library

size_t nbasis(const std::vector<libint2::Shell> &shells);

std::vector<size_t> map_shell_to_basis_function(const std::vector<libint2::Shell> &shells);

Matrix compute_soad(const std::vector<libint2::Atom> &atoms);

Matrix compute_1body_ints(const std::vector<libint2::Shell> &shells,
                          libint2::Operator t,
                          const std::vector<libint2::Atom> &atoms = std::vector<libint2::Atom>());

// simple-to-read, but inefficient Fock builder; computes ~16 times as many ints as possible
Matrix compute_2body_fock_simple(const std::vector<libint2::Shell> &shells,
                                 const Matrix &D);

// an efficient Fock builder; *integral-driven* hence computes permutationally-unique ints once
Matrix compute_2body_fock(const std::vector<libint2::Shell> &shells,
                          const Matrix &D);

std::tuple<Matrix, Tensor4D, double> hartree_fock(const string filename);


// int main(int argc, char *argv[]) {
//   const auto filename = (argc > 1) ? argv[1] : "h2o.xyz";
//
//   Matrix F;
//   Tensor4D V;
//   double hf_energy{0.0};
//
//   std::tie(F, V, hf_energy) = hartree_fock(filename);
//
//   // @todo CALL CCSD_DRIVER(F,V,hf_energy,...);
//
//   // std::vector<double> rawC(C.rows()*C.cols());
//   // cout << "----------------\n";
//   // Eigen::Map<Matrix>(rawC.data(),C.rows(),C.cols()) = C;
//
// }


size_t nbasis(const std::vector<libint2::Shell> &shells) {
  size_t n = 0;
  for (const auto &shell: shells)
    n += shell.size();
  return n;
}

size_t max_nprim(const std::vector<libint2::Shell> &shells) {
  size_t n = 0;
  for (auto shell: shells)
    n = std::max(shell.nprim(), n);
  return n;
}

int max_l(const std::vector<libint2::Shell> &shells) {
  int l = 0;
  for (auto shell: shells)
    for (auto c: shell.contr)
      l = std::max(c.l, l);
  return l;
}

std::vector<size_t> map_shell_to_basis_function(const std::vector<libint2::Shell> &shells) {
  std::vector<size_t> result;
  result.reserve(shells.size());

  size_t n = 0;
  for (auto shell: shells) {
    result.push_back(n);
    n += shell.size();
  }

  return result;
}


std::tuple<Matrix, Tensor4D, double> hartree_fock(const string filename) {

  // Perform the simple HF calculation (Ed) and 2,4-index transform to get the inputs for CCSD
  using libint2::Atom;
  using libint2::Shell;
  using libint2::Engine;
  using libint2::Operator;

  /*** =========================== ***/
  /*** initialize molecule         ***/
  /*** =========================== ***/

  // read geometry from a file; by default read from h2o.xyz, else take filename (.xyz) from the command line
  auto is = std::ifstream(filename);
  const std::vector<Atom> atoms = libint2::read_dotxyz(is);

  // count the number of electrons
  auto nelectron = 0;
  for (auto i = 0; i < atoms.size(); ++i)
    nelectron += atoms[i].atomic_number;
  const auto ndocc = nelectron / 2;

  // compute the nuclear repulsion energy
  auto enuc = 0.0;
  for (auto i = 0; i < atoms.size(); i++)
    for (auto j = i + 1; j < atoms.size(); j++) {
      auto xij = atoms[i].x - atoms[j].x;
      auto yij = atoms[i].y - atoms[j].y;
      auto zij = atoms[i].z - atoms[j].z;
      auto r2 = xij * xij + yij * yij + zij * zij;
      auto r = sqrt(r2);
      enuc += atoms[i].atomic_number * atoms[j].atomic_number / r;
    }
  cout << "\tNuclear repulsion energy = " << enuc << endl;



  // initializes the Libint integrals library ... now ready to compute
  libint2::initialize();

  /*** =========================== ***/
  /*** create basis set            ***/
  /*** =========================== ***/

  // LIBINT_INSTALL_DIR/share/libint/2.4.0-beta.1/basis
  libint2::BasisSet shells(std::string("sto-3g"), atoms);
  //auto shells = make_sto3g_basis(atoms);
  size_t nao = 0;
  for (auto s = 0; s < shells.size(); ++s)
    nao += shells[s].size();

  /*** =========================== ***/
  /*** compute 1-e integrals       ***/
  /*** =========================== ***/

  // compute overlap integrals
  auto S = compute_1body_ints(shells, Operator::overlap);
  cout << "\n\tOverlap Integrals:\n";
  cout << S << endl;

  // compute kinetic-energy integrals
  auto T = compute_1body_ints(shells, Operator::kinetic);
  cout << "\n\tKinetic-Energy Integrals:\n";
  cout << T << endl;

  // compute nuclear-attraction integrals
  Matrix V = compute_1body_ints(shells, Operator::nuclear, atoms);
  cout << "\n\tNuclear Attraction Integrals:\n";
  cout << V << endl;

  // Core Hamiltonian = T + V
  Matrix H = T + V;
  cout << "\n\tCore Hamiltonian:\n";
  cout << H << endl;

  // T and V no longer needed, free up the memory
  T.resize(0, 0);
  V.resize(0, 0);

  /*** =========================== ***/
  /*** build initial-guess density ***/
  /*** =========================== ***/

  const auto use_hcore_guess = false;  // use core Hamiltonian eigenstates to guess density?
  // set to true to match the result of versions 0, 1, and 2 of the code
  // HOWEVER !!! even for medium-size molecules hcore will usually fail !!!
  // thus set to false to use Superposition-Of-Atomic-Densities (SOAD) guess
  Matrix D;
  if (use_hcore_guess) { // hcore guess
    // solve H C = e S C
    Eigen::GeneralizedSelfAdjointEigenSolver <Matrix> gen_eig_solver(H, S);
    auto eps = gen_eig_solver.eigenvalues();
    auto C = gen_eig_solver.eigenvectors();
    cout << "\n\tInitial C Matrix:\n";
    cout << C << endl;

    // compute density, D = C(occ) . C(occ)T
    auto C_occ = C.leftCols(ndocc);
    D = C_occ * C_occ.transpose();
  } else {  // SOAD as the guess density, assumes STO-nG basis
    D = compute_soad(atoms);
  }

  cout << "\n\tInitial Density Matrix:\n";
  cout << D << endl;

  /*** =========================== ***/
  /*** main iterative loop         ***/
  /*** =========================== ***/

  const auto maxiter = 100;
  const auto conv = 1e-12;
  auto iter = 0;
  auto rmsd = 0.0;
  auto ediff = 0.0;
  auto ehf = 0.0;
  Matrix C;
  Matrix F;
  Matrix eps;

  do {
    const auto tstart = std::chrono::high_resolution_clock::now();
    ++iter;

    // Save a copy of the energy and the density
    auto ehf_last = ehf;
    auto D_last = D;

    // build a new Fock matrix
    //auto F = H;
    //F += compute_2body_fock_simple(shells, D);
    F = H;
    F += compute_2body_fock_simple(shells, D);

    if (iter == 1) {
      cout << "\n\tFock Matrix:\n";
      cout << F << endl;
    }

    // solve F C = e S C
    Eigen::GeneralizedSelfAdjointEigenSolver <Matrix> gen_eig_solver(F, S);
    //auto
    eps = gen_eig_solver.eigenvalues();
    C = gen_eig_solver.eigenvectors();
    //auto C1 = gen_eig_solver.eigenvectors();

    // compute density, D = C(occ) . C(occ)T
    auto C_occ = C.leftCols(ndocc);
    D = C_occ * C_occ.transpose();

    // compute HF energy
    ehf = 0.0;
    for (auto i = 0; i < nao; i++)
      for (auto j = 0; j < nao; j++)
        ehf += D(i, j) * (H(i, j) + F(i, j));

    // compute difference with last iteration
    ediff = ehf - ehf_last;
    rmsd = (D - D_last).norm();

    const auto tstop = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> time_elapsed = tstop - tstart;

    // if (iter == 1)
    //   std::cout <<
    //   "\n\n Iter        E(elec)              E(tot)               Delta(E)             RMS(D)         Time(s)\n";
    // printf(" %02d %20.12f %20.12f %20.12f %20.12f %10.5lf\n", iter, ehf, ehf + enuc,
    //        ediff, rmsd, time_elapsed.count());

  } while (((fabs(ediff) > conv) || (fabs(rmsd) > conv)) && (iter < maxiter));

  printf("\n** Hartree-Fock energy = %20.12f\n", ehf + enuc);

  cout << "\n** Eigen Values:\n";
  cout << eps << endl;


  // 2-index transform
  int num_electrons = ndocc;
  cout << "\n\n** Number of electrons: " << num_electrons << endl;

  cout << "\n\t C Matrix:\n";
  cout << C << endl;

  cout << "\n\t F_AO Matrix:\n";
  cout << F << endl;

  const int C_rows = C.rows();
  const int C_cols = C.cols();

  // replicate horizontally
  Matrix C_2N(C_rows, 2 * C_cols);
  C_2N << C, C;
  //cout << "\n\t C_2N Matrix:\n";
  //cout << C_2N << endl;

  const int b_rows = 7, nelectrons = 5;
  Matrix C_noa = C_2N.block<b_rows, nelectrons>(0, 0);
  cout << "\n\t C occupied alpha:\n";
  cout << C_noa << endl;

  Matrix C_nva = C_2N.block<b_rows, b_rows - nelectrons>(0, num_electrons);
  cout << "\n\t C virtual alpha:\n";
  cout << C_nva << endl;

  Matrix C_nob = C_2N.block<b_rows, nelectrons>(0, C_cols);
  cout << "\n\t C occupied beta:\n";
  cout << C_nob << endl;

  Matrix C_nvb = C_2N.block<b_rows, b_rows - nelectrons>(0, num_electrons + C_cols);
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
  return std::make_tuple(F, A2, (ehf + enuc));

}


// computes Superposition-Of-Atomic-Densities guess for the molecular density matrix
// in minimal basis; occupies subshells by smearing electrons evenly over the orbitals
Matrix compute_soad(const std::vector<libint2::Atom> &atoms) {

  // compute number of atomic orbitals
  size_t nao = 0;
  for (const auto &atom: atoms) {
    const auto Z = atom.atomic_number;
    if (Z == 1 || Z == 2) // H, He
      nao += 1;
    else if (Z <= 10) // Li - Ne
      nao += 5;
    else
      throw "SOAD with Z > 10 is not yet supported";
  }

  // compute the minimal basis density
  Matrix D = Matrix::Zero(nao, nao);
  size_t ao_offset = 0; // first AO of this atom
  for (const auto &atom: atoms) {
    const auto Z = atom.atomic_number;
    if (Z == 1 || Z == 2) { // H, He
      D(ao_offset, ao_offset) = Z; // all electrons go to the 1s
      ao_offset += 1;
    } else if (Z <= 10) {
      D(ao_offset, ao_offset) = 2; // 2 electrons go to the 1s
      D(ao_offset + 1, ao_offset + 1) = (Z == 3) ? 1 : 2; // Li? only 1 electron in 2s, else 2 electrons
      // smear the remaining electrons in 2p orbitals
      const double num_electrons_per_2p = (Z > 4) ? (double) (Z - 4) / 3 : 0;
      for (auto xyz = 0; xyz != 3; ++xyz)
        D(ao_offset + 2 + xyz, ao_offset + 2 + xyz) = num_electrons_per_2p;
      ao_offset += 5;
    }
  }

  return D * 0.5; // we use densities normalized to # of electrons/2
}

Matrix compute_1body_ints(const std::vector<libint2::Shell> &shells,
                          libint2::Operator obtype,
                          const std::vector<libint2::Atom> &atoms) {
  using libint2::Shell;
  using libint2::Engine;
  using libint2::Operator;

  const auto n = nbasis(shells);
  Matrix result(n, n);

  // construct the overlap integrals engine
  Engine engine(obtype, max_nprim(shells), max_l(shells), 0);
  // nuclear attraction ints engine needs to know where the charges sit ...
  // the nuclei are charges in this case; in QM/MM there will also be classical charges
  if (obtype == Operator::nuclear) {
    std::vector<std::pair<double, std::array<double, 3>>> q;
    for (const auto &atom : atoms) {
      q.push_back({static_cast<double>(atom.atomic_number), {{atom.x, atom.y, atom.z}}});
    }
    engine.set_params(q);
  }

  auto shell2bf = map_shell_to_basis_function(shells);

  // buf[0] points to the target shell set after every call  to engine.compute()
  const auto &buf = engine.results();

  // loop over unique shell pairs, {s1,s2} such that s1 >= s2
  // this is due to the permutational symmetry of the real integrals over Hermitian operators: (1|2) = (2|1)
  for (auto s1 = 0; s1 != shells.size(); ++s1) {

    auto bf1 = shell2bf[s1]; // first basis function in this shell
    auto n1 = shells[s1].size();

    for (auto s2 = 0; s2 <= s1; ++s2) {

      auto bf2 = shell2bf[s2];
      auto n2 = shells[s2].size();

      // compute shell pair; return is the pointer to the buffer
      engine.compute(shells[s1], shells[s2]);

      // "map" buffer to a const Eigen Matrix, and copy it to the corresponding blocks of the result
      Eigen::Map<const Matrix> buf_mat(buf[0], n1, n2);
      result.block(bf1, bf2, n1, n2) = buf_mat;
      if (s1 != s2) // if s1 >= s2, copy {s1,s2} to the corresponding {s2,s1} block, note the transpose!
        result.block(bf2, bf1, n2, n1) = buf_mat.transpose();

    }
  }

  return result;
}

Matrix compute_2body_fock_simple(const std::vector<libint2::Shell> &shells,
                                 const Matrix &D) {

  using libint2::Shell;
  using libint2::Engine;
  using libint2::Operator;

  const auto n = nbasis(shells);
  Matrix G = Matrix::Zero(n, n);

  // construct the electron repulsion integrals engine
  Engine engine(Operator::coulomb, max_nprim(shells), max_l(shells), 0);

  auto shell2bf = map_shell_to_basis_function(shells);

  // buf[0] points to the target shell set after every call  to engine.compute()
  const auto &buf = engine.results();

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
                  G(bf1, bf2) += D(bf3, bf4) * 2.0 * buf_1234[f1234];
                }
              }
            }
          }

          // exchange contribution to the Fock matrix is from {s1,s3,s2,s4} integrals
          engine.compute(shells[s1], shells[s3], shells[s2], shells[s4]);
          const auto *buf_1324 = buf[0];

          for (auto f1 = 0, f1324 = 0; f1 != n1; ++f1) {
            const auto bf1 = f1 + bf1_first;
            for (auto f3 = 0; f3 != n3; ++f3) {
              const auto bf3 = f3 + bf3_first;
              for (auto f2 = 0; f2 != n2; ++f2) {
                const auto bf2 = f2 + bf2_first;
                for (auto f4 = 0; f4 != n4; ++f4, ++f1324) {
                  const auto bf4 = f4 + bf4_first;
                  G(bf1, bf2) -= D(bf3, bf4) * buf_1324[f1324];
                }
              }
            }
          }

        }
      }
    }
  }

  return G;
}

Matrix compute_2body_fock(const std::vector<libint2::Shell> &shells,
                          const Matrix &D) {

  using libint2::Shell;
  using libint2::Engine;
  using libint2::Operator;

  std::chrono::duration<double> time_elapsed = std::chrono::duration<double>::zero();

  const auto n = nbasis(shells);
  Matrix G = Matrix::Zero(n, n);

  // construct the 2-electron repulsion integrals engine
  Engine engine(Operator::coulomb, max_nprim(shells), max_l(shells), 0);

  auto shell2bf = map_shell_to_basis_function(shells);

  const auto &buf = engine.results();

  // The problem with the simple Fock builder is that permutational symmetries of the Fock,
  // density, and two-electron integrals are not taken into account to reduce the cost.
  // To make the simple Fock builder efficient we must rearrange our computation.
  // The most expensive step in Fock matrix construction is the evaluation of 2-e integrals;
  // hence we must minimize the number of computed integrals by taking advantage of their permutational
  // symmetry. Due to the multiplicative and Hermitian nature of the Coulomb kernel (and realness
  // of the Gaussians) the permutational symmetry of the 2-e ints is given by the following relations:
  //
  // (12|34) = (21|34) = (12|43) = (21|43) = (34|12) = (43|12) = (34|21) = (43|21)
  //
  // (here we use chemists' notation for the integrals, i.e in (ab|cd) a and b correspond to
  // electron 1, and c and d -- to electron 2).
  //
  // It is easy to verify that the following set of nested loops produces a permutationally-unique
  // set of integrals:
  // foreach a = 0 .. n-1
  //   foreach b = 0 .. a
  //     foreach c = 0 .. a
  //       foreach d = 0 .. (a == c ? b : c)
  //         compute (ab|cd)
  //
  // The only complication is that we must compute integrals over shells. But it's not that complicated ...
  //
  // The real trick is figuring out to which matrix elements of the Fock matrix each permutationally-unique
  // (ab|cd) contributes. STOP READING and try to figure it out yourself. (to check your answer see below)

  // loop over permutationally-unique set of shells
  for (auto s1 = 0; s1 != shells.size(); ++s1) {

    auto bf1_first = shell2bf[s1]; // first basis function in this shell
    auto n1 = shells[s1].size();   // number of basis functions in this shell

    for (auto s2 = 0; s2 <= s1; ++s2) {

      auto bf2_first = shell2bf[s2];
      auto n2 = shells[s2].size();

      for (auto s3 = 0; s3 <= s1; ++s3) {

        auto bf3_first = shell2bf[s3];
        auto n3 = shells[s3].size();

        const auto s4_max = (s1 == s3) ? s2 : s3;
        for (auto s4 = 0; s4 <= s4_max; ++s4) {

          auto bf4_first = shell2bf[s4];
          auto n4 = shells[s4].size();

          // compute the permutational degeneracy (i.e. # of equivalents) of the given shell set
          auto s12_deg = (s1 == s2) ? 1.0 : 2.0;
          auto s34_deg = (s3 == s4) ? 1.0 : 2.0;
          auto s12_34_deg = (s1 == s3) ? (s2 == s4 ? 1.0 : 2.0) : 2.0;
          auto s1234_deg = s12_deg * s34_deg * s12_34_deg;

          const auto tstart = std::chrono::high_resolution_clock::now();

          engine.compute(shells[s1], shells[s2], shells[s3], shells[s4]);
          const auto *buf_1234 = buf[0];
          if (buf_1234 == nullptr)
            continue; // if all integrals screened out, skip to next quartet

          const auto tstop = std::chrono::high_resolution_clock::now();
          time_elapsed += tstop - tstart;

          // ANSWER
          // 1) each shell set of integrals contributes up to 6 shell sets of the Fock matrix:
          //    F(a,b) += (ab|cd) * D(c,d)
          //    F(c,d) += (ab|cd) * D(a,b)
          //    F(b,d) -= 1/4 * (ab|cd) * D(a,c)
          //    F(b,c) -= 1/4 * (ab|cd) * D(a,d)
          //    F(a,c) -= 1/4 * (ab|cd) * D(b,d)
          //    F(a,d) -= 1/4 * (ab|cd) * D(b,c)
          // 2) each permutationally-unique integral (shell set) must be scaled by its degeneracy,
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
    }
  }

  // symmetrize the result and return
  Matrix Gt = G.transpose();
  return 0.5 * (G + Gt);
}
