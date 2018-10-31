#ifndef UTILITIES_H
#define UTILITIES_H
// Basic I/O
#include <iostream>
#include <fstream>
#include <iomanip>
static std::ofstream logOFS; // logfile stream

using std::cout;
using std::endl;
using std::scientific;
using std::showpos;
using std::noshowpos;

// For 32-bit and 64-bit integers
// MPI and OpenMP
#include <omp.h>
#include <mpi.h>
#include "lapacke.h"

// Eigen classes
#include <Eigen/Dense>
using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::MatrixXd;

using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
//using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

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
#undef I

 #ifdef __cplusplus
 extern "C"
 {
 #endif

 //BLAS
 void dtrsm_(const char *side, const char *uplo,
                              const char *transa, const char *diag,
                              const int *m, const int *n, const double *alpha,
                              const double *a, const int *lda,
                              double *b, const int *ldb);

/* Routines for inverting matrices */
void dgetrf_(int *m, int *n, double *A, int *lda, int *ipiv, int *info);

/* Routines for computing the least-squares solution to Ax = b */
void dgelss_(int *m, int *n, int *nrhs, double *A, int *lda, double *b, int *ldb, 
	     double *s, double *rcond, int *rank, double *work, int *lwork, int *info);
void dgelsy_(int *m, int *n, int *nrhs, double *A, int *lda, double *b, int *ldb,
	     int *jpvt, double *rcond, int *rank, double *work, int *lwork, int *info);

void dgesv_(int *n, int *nrhs, double *A, int *lda, int *ipiv, double *b,
	    int *ldb, int *info);


/* Routine for computing the eigenvalues / eigenvectors of a matrix */
void dgeev_(char *jobvl, char *jobvr, int *n, double *A, int *lda, double *wr, double *wi,
	    double *vl, int *ldvl, double *vr, int *ldvr, double *work, int *lwork,
	    int *info);

/* Routine for singular value decomposition */
void dgesvd_(char *jobu, char *jobvt, int *m, int *n, double *A, int *lda, 
	     double *S, double *U, int *ldu, double *VT, int *ldvt,
	     double *work, int *lwork, int *info);

/* Routine for Cholesky decomposition */
void dpotrf_(char *uplo, int *n, double *A, int *lda, int *info);

/* Routine for QR factorization */
void dgeqrf_(int *m, int *n, double *A, int *lda, double *tau, double *work, 
	     int *lwork, int *info);

/* Routine for RQ factorization */
void dgerqf_(int *m, int *n, double *A, int *lda, double *tau, double *work, 
	     int *lwork, int *info);

  #ifdef __cplusplus
 }
 #endif

// utility functions
VectorXi sortinds(VectorXd xs);
void print_results(std::ofstream &resultsfile, int iter, 
                   VectorXd &evals, VectorXd &resnrms);

int MPI_Bcast_params(MPI_Comm comm, sislice_param *params);
int MPI_Bcast_scf_params(MPI_Comm comm, scf_param *params);
Matrix ReadAndBcastMatrix(MPI_Comm comm, char *filename);

#endif
