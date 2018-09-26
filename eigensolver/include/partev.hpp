#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues> 
#include <omp.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

VectorXd HSLanczos(const MatrixXd &H, const MatrixXd &S, const VectorXd &v0, MatrixXd &T, MatrixXd &V);
int partev(const MatrixXd &T, const VectorXd &f, VectorXd &shifts, int nev, int nevloc);
VectorXd getevbnd(const VectorXd &d, const MatrixXd &S, const double &beta);
