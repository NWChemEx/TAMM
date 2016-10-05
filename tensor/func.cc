#include "func.h"

namespace tamm {

double rtclock() {
  struct timezone Tzp;
  struct timeval Tp;
  int stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0) printf("rtclock error.\n");
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

int factorial(int n) {
  assert(n >= 0 && n <= 5);
  if (n <= 1) return 1;
  if (n == 2) return 2;
  if (n == 3) return 6;
  if (n == 4) return 12;
  if (n == 5) return 60;
}

// compute factor reduction for isuperp
double computeBeta(const std::vector<IndexName>& sum_ids,
                   const std::vector<size_t>& sum_vec) {
  std::vector<size_t> p_group, h_group;
  for (int i = 0; i < sum_ids.size(); i++) {
    assert(sum_ids[i] < IndexNum && sum_ids[i] >= 0);
    if (sum_ids[i] < pIndexNum && sum_ids[i] >= 0)
      p_group.push_back(sum_vec[i]);
    else
      h_group.push_back(sum_vec[i]);
  }
  double p_fact = factorial(p_group.size());
  double h_fact = factorial(h_group.size());
  std::sort(p_group.begin(), p_group.end());
  std::sort(h_group.begin(), h_group.end());
  int tsize = 1;
  for (int i = 1; i < p_group.size(); i++) {
    if (p_group[i] != p_group[i - 1]) {
      p_fact /= factorial(tsize);
      tsize = 0;
    }
    tsize += 1;
  }
  p_fact /= factorial(tsize);
  /*  std::cout << p_group << std::endl;
      std::cout << p_fact << std::endl;*/
  tsize = 1;
  for (int i = 1; i < h_group.size(); i++) {
    if (h_group[i] != h_group[i - 1]) {
      h_fact /= factorial(tsize);
      tsize = 0;
    }
    tsize += 1;
  }
  h_fact /= factorial(tsize);
  /*  std::cout << h_group << std::endl;
      std::cout << h_fact << std::endl;*/
  return p_fact * h_fact;
}

size_t compute_size(const std::vector<size_t>& ids) {
  size_t size = 1;
  Integer* int_mb = Variables::int_mb();
  size_t k_range = Variables::k_range() - 1;
  for (int i = 0; i < ids.size(); i++) size *= int_mb[k_range + ids[i]];
  return size;
}

IndexType getIndexType(const IndexName& name) {
  if (name >= 0 && name < pIndexNum)
    return pIndex;
  else
    return hIndex;
}

// used in ccsd_t.cc
int is_spin_restricted_le(const std::vector<size_t>& ids, const size_t& sval) {
  size_t lval = 0;
  Integer* int_mb = Variables::int_mb();
  size_t k_spin = Variables::k_spin() - 1;
  size_t restricted = Variables::restricted();
  for (int i = 0; i < ids.size(); i++) lval += int_mb[k_spin + ids[i]];
  return ((!restricted) || (lval <= sval));
}

double computeFactor(const std::vector<size_t>& ids) {
  double factor;
  if (Variables::restricted())
    factor = 2.0;
  else
    factor = 1.0;
  if ((ids[0] == ids[1]) && (ids[1] == ids[2]))
    factor /= 6.0;
  else if ((ids[0] == ids[1]) || (ids[1] == ids[2]))
    factor /= 2.0;
  if ((ids[3] == ids[4]) && (ids[4] == ids[5]))
    factor /= 6.0;
  else if ((ids[3] == ids[4]) || (ids[4] == ids[5]))
    factor /= 2.0;
  return factor;
}

// hard-coded
void computeEnergy(const std::vector<size_t>& rvec,
                   const std::vector<size_t>& ovec, double* energy1,
                   double* energy2, double* buf_single, double* buf_double,
                   const double& factor) {
  double* dbl_mb = Variables::dbl_mb();
  double denom, denom_p4, denom_p5, denom_p6, denom_h1, denom_h2, denom_h3;
  size_t rp4, rp5, rp6, rh1, rh2, rh3, op4, op5, op6, oh1, oh2, oh3;
  rp4 = rvec[0];
  op4 = ovec[0];
  rp5 = rvec[1];
  op5 = ovec[1];
  rp6 = rvec[2];
  op6 = ovec[2];
  rh1 = rvec[3];
  oh1 = ovec[3];
  rh2 = rvec[4];
  oh2 = ovec[4];
  rh3 = rvec[5];
  oh3 = ovec[5];
  int i = 0;
  for (int t_p4 = 1; t_p4 <= rp4; t_p4++) {
    denom_p4 = dbl_mb[op4 + t_p4];
    for (int t_p5 = 1; t_p5 <= rp5; t_p5++) {
      denom_p5 = dbl_mb[op5 + t_p5];
      for (int t_p6 = 1; t_p6 <= rp6; t_p6++) {
        denom_p6 = dbl_mb[op6 + t_p6];
        for (int t_h1 = 1; t_h1 <= rh1; t_h1++) {
          denom_h1 = dbl_mb[oh1 + t_h1];
          for (int t_h2 = 1; t_h2 <= rh2; t_h2++) {
            denom_h2 = dbl_mb[oh2 + t_h2];
            for (int t_h3 = 1; t_h3 <= rh3; t_h3++) {
              denom_h3 = dbl_mb[oh3 + t_h3];
              denom = 1.0 / ((denom_h1 + denom_h2 + denom_h3) -
                             (denom_p4 + denom_p5 + denom_p6));
              *energy1 += factor * denom * buf_double[i] * buf_double[i];
              *energy2 += factor * denom * buf_double[i] *
                          (buf_double[i] + buf_single[i]);
              i++;
            }
          }
        }
      }
    }
  }
}

};  // namespace tamm
