#ifndef UTILITIES_H
#define UTILITIES_H
// Basic I/O
#include <iostream>
#include <fstream>
#include <iomanip>
extern std::ofstream logOFS; // logfile stream

using std::cout;
using std::endl;
using std::scientific;
using std::showpos;
using std::noshowpos;

// For 32-bit and 64-bit integers
// MPI and OpenMP
#include <omp.h>
#include <mpi.h>

// Eigen classes
#include <Eigen/Dense>
#include <Eigen/Eigenvalues> 
using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::MatrixXd;

//using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

// vector, numeric and other commonly used class 
#include <vector>
#include <numeric>
#include <set>
#include <map>
#include <stack>
#include <memory>

// using namespace std;
using std::min;
using std::max;

// for string I/O?
#include <stdlib.h>
#include <stdint.h>

// algorithm and other commonly used math classes
#include <algorithm>
#include <cmath>
#include <random>

// exception handling classes
#include <cassert>
#include <stdexcept>
#include <execinfo.h>

// input parsing parameters
#include "parseinput.hpp"

// utility functions
VectorXi sortinds(VectorXd xs);
void print_results(std::ofstream &resultsfile, int iter, 
                   VectorXd &evals, VectorXd &resnrms);

int MPI_Bcast_params(MPI_Comm comm, sislice_param *params);
int MPI_Bcast_scf_params(MPI_Comm comm, scf_param *params);
Matrix ReadAndBcastMatrix(MPI_Comm comm, char *filename);

#endif
