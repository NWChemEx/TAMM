extern "C" {
#include <stdio.h>
#include <stdlib.h>
}
#include <iostream>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::LDLT;
using Eigen::LLT;

MatrixXd cholQR(const MatrixXd S, const MatrixXd X0);

int main(int argc, char *argv[])
{
  int n, nnz;
  FILE *fp, *fpmat;
  char fmatname[100];

  if (argc < 2) {
     fprintf(stderr, "Missing input arguments! \n");
     fprintf(stderr, "runsisubit <input matrix> \n");
     exit(2);
  }

  sscanf(*(argv+1),"%s", fmatname);
  printf("input fname =  %s\n", fmatname);
  fpmat = fopen(fmatname,"r");
  fread(&n, sizeof(int), 1, fpmat);
  fread(&nnz, sizeof(int), 1, fpmat);
  fclose(fpmat);

  n = 5;
  MatrixXd S(n,n), HS(n,n), L(n,n); 
  VectorXd D(n);
  
  int nev = 2;
  MatrixXd X0 = MatrixXd::Random(n,nev);
  MatrixXd X(n,nev);
  
  double shift = 1.0;

//  MatrixXd H = MatrixXd::Random(n,n);
  MatrixXd H(n,n);
  H.setZero();

  H(0,0) = -0.0904118;
  H(1,0) = 1.09013;
  H(1,1) = 0.542847;
  H(2,0) = -1.23783;
  H(2,1) = -0.0796326;
  H(2,2) = -1.45107;
  H(3,0) = -0.171309;
  H(3,1) = -1.45721;
  H(3,2) = -0.174029;
  H(3,3) =  1.9957;
  H(4,0) =  0.930324;
  H(4,1) =  0.892162;
  H(4,2) = -0.461362;
  H(4,3) = -0.971423;
  H(4,4) =  0.550209;

  S = H + H.transpose();
  H = S;

  std::cout << "H = " << H << std::endl;

  S.setZero();
  S(0,0) = 1.0;
  S(1,1) = 2.0;
  S(2,2) = 3.0;
  S(3,3) = 4.0;
  S(4,4) = 5.0;

  HS = H - shift*S;

  LDLT<MatrixXd> ldlt(HS);

  ldlt.compute(HS);
  L = ldlt.matrixL();
  D = ldlt.vectorD();
  std::cout << L << std::endl;
  std::cout << D << std::endl;

  int maxiter = 10;

  for (int iter=0; iter<maxiter; iter++) {
     X  = S*X0;
     X0 = ldlt.solve(X);
     X  = cholQR(S,X0);
     X0 = X;
     std::cout << X0.transpose()*(S*X0) << std::endl;
  }
  std::cout << X0.transpose()*(H*X0) << std::endl;

}

MatrixXd cholQR(const MatrixXd S, const MatrixXd X0)
{
   int n     = X0.rows();
   int ncols = X0.cols();
   MatrixXd G(n,ncols);
   G = X0.transpose()*(S*X0);

   LLT<MatrixXd> cholfac(G);
   MatrixXd R = cholfac.matrixU();

   return X0*R.inverse();
}

