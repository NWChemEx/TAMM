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

std::tuple<int,double, libint2::BasisSet> hartree_fock(const string filename,Matrix &C, Matrix &F);

size_t max_nprim(const std::vector<libint2::Shell> &shells);

int max_l(const std::vector<libint2::Shell> &shells);

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
