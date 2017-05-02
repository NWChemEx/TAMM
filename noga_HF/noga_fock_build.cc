

#include <iostream>
#include <vector>

//    p, q, r, s = N (O+V)
//    i, j, m, n = O
//    a, b, c, d, e, f = V
//    Q = 3 * (O+V)

const int O = 50;
const int V = 100;

constexpr int N = O + V;
const int pR = N;
const int qR = N;

constexpr int QR = 3 * N;
const int iR = O, jR = O, mR = O, nR = O;
const int aR = V, bR = V, cR = V, dR = V;

void noga_fock_build(double (*b_T)[N], double (*h_T)[N], double (*F_T)[N],
                     double (*X_OO)[O], double (*X_OV)[V], double (*X_VV)[V]) {
  for (int p = 0; p < pR; p++) {
    for (int q = 0; q < qR; q++) {
      F_T[p][q] += h_T[p][q];

      for (int i = 0; i < iR; i++) {
        F_T[p][q] += b_T[p][q] * b_T[i][i];
      }

      for (int i = 0; i < iR; i++) {
        for (int j = 0; j < jR; j++) {
          F_T[p][q] += b_T[p][q] * X_OO[i][j] * b_T[i][j];
        }
      }

      for (int i = 0; i < iR; i++) {
        for (int a = 0; a < aR; a++) {
          F_T[p][q] += 2 * b_T[p][q] * X_OV[i][a] * b_T[i][a];
        }
      }

      for (int a = 0; a < aR; a++) {
        for (int b = 0; b < bR; b++) {
          F_T[p][q] += b_T[p][q] * X_VV[a][b] * b_T[a][b];
        }
      }

      for (int i = 0; i < iR; i++) {
        F_T[p][q] -= b_T[p][i] * b_T[i][q];
      }

      for (int i = 0; i < iR; i++) {
        for (int j = 0; j < jR; j++) {
          F_T[p][q] -= b_T[p][i] * X_OO[i][j] * b_T[j][q];
        }
      }

      for (int i = 0; i < iR; i++) {
        for (int a = 0; a < aR; a++) {
          F_T[p][q] -= b_T[p][i] * X_OV[i][a] * b_T[a][q];
        }
      }

      for (int i = 0; i < iR; i++) {
        for (int a = 0; a < aR; a++) {
          F_T[p][q] -= b_T[i][q] * X_OV[i][a] * b_T[p][a];
        }
      }

      for (int a = 0; a < aR; a++) {
        for (int b = 0; b < bR; b++) {
          F_T[p][q] -= b_T[p][a] * X_VV[a][b] * b_T[b][q];
        }
      }
    }  // END OUTER Q
  }    // END OUTER P
}  // END FOCK BUILD

int main() {
  double(*b_T)[N] = new double[N][N];
  double(*h_T)[N] = new double[N][N];
  double(*F_T)[N] = new double[N][N];

  double(*X_OO)[O] = new double[O][O];
  double(*X_OV)[V] = new double[O][V];
  double(*X_VV)[V] = new double[V][V];

  noga_fock_build(b_T, h_T, F_T, X_OO, X_OV, X_VV);
  return 0;
}
