//------------------------------------------------------------------------------
// Copyright (C) 2016, Pacific Northwest National Laboratory
// This software is subject to copyright protection under the laws of the
// United States and other countries
//
// All rights in this computer software are reserved by the
// Pacific Northwest National Laboratory (PNNL)
// Operated by Battelle for the U.S. Department of Energy
//
//------------------------------------------------------------------------------
#include "tensor/variables.h"
#include <mpi.h>
#include <vector>

namespace tamm {

double Timer::total = 0.0;
double Timer::dg_time = 0.0;
double Timer::ah_time = 0.0;
double Timer::sa_time = 0.0;
double Timer::so_time = 0.0;
int Timer::dg_num = 0;
int Timer::ah_num = 0;
int Timer::sa_num = 0;
int Timer::so_num = 0;

int MPI_Timer::timer_num[40] = {0};
double MPI_Timer::timer_value[40] = {0.e0};
double MPI_Timer::timer_cum_value[40] = {0.e0};
char const *MPI_Timer::timer_text[] = {
    "1",  "2",  "3",  "4",  "5",  "6",  "7",  "8",  "9",  "10",
    "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
    "21", "22", "23", "24", "25", "26", "27", "28", "29", "30",
    "31", "32", "33", "34", "35", "36", "37", "38", "39", "40"};

F77Integer Variables::noab_ = 0;
F77Integer Variables::nvab_ = 0;
F77Integer Variables::noa_ = 0;
F77Integer Variables::nva_ = 0;
F77Integer Variables::k_range_ = 0;
F77Integer Variables::k_spin_ = 0;
F77Integer Variables::k_sym_ = 0;
F77Integer Variables::k_offset_ = 0;
F77Integer Variables::k_evl_sorted_ = 0;
F77Integer Variables::irrep_v_ = 0;
F77Integer Variables::irrep_t_ = 0;
F77Integer Variables::irrep_f_ = 0;
F77Integer Variables::irrep_x_ = 0;
F77Integer Variables::irrep_y_ = 0;
logical Variables::intorb_ = 0;
logical Variables::restricted_ = 0;
F77Integer *Variables::int_mb_;
double *Variables::dbl_mb_;
F77Integer Variables::k_alpha_ = 0;
F77Integer Variables::k_b2am_ = 0;
F77Integer Variables::d_v2orb_ = 0;
F77Integer Variables::k_v2_alpha_offset_ = 0;

void Variables::set_ov(F77Integer *o, F77Integer *v) {
  noab_ = *o;
  nvab_ = *v;
}
void Variables::set_ova(F77Integer *noa, F77Integer *nva) {
  noa_ = *noa;
  nva_ = *nva;
}
void Variables::set_idmb(F77Integer *int_mb_f, double *dbl_mb_f) {
  int_mb_ = int_mb_f - 1;
  dbl_mb_ = dbl_mb_f - 1;
}
void Variables::set_irrep(F77Integer *irrep_v, F77Integer *irrep_t,
                          F77Integer *irrep_f) {
  irrep_v_ = *irrep_v;
  irrep_t_ = *irrep_t;
  irrep_f_ = *irrep_f;
}
void Variables::set_irrep_x(F77Integer *irrep_x) { irrep_x_ = *irrep_x; }
void Variables::set_irrep_y(F77Integer *irrep_y) { irrep_y_ = *irrep_y; }
void Variables::set_k1(F77Integer *k_range, F77Integer *k_spin, F77Integer *k_sym) {
  k_range_ = *k_range;
  k_spin_ = *k_spin;
  k_sym_ = *k_sym;
}
void Variables::set_k2(F77Integer *k_offset, F77Integer *k_evl_sorted) {
  k_offset_ = *k_offset;
  k_evl_sorted_ = *k_evl_sorted;
}
void Variables::set_log(logical *intorb, logical *restricted) {
  intorb_ = *intorb;
  restricted_ = *restricted;
}
void Variables::set_k_alpha(F77Integer *k_alpha) { k_alpha_ = *k_alpha; }

void Variables::set_k_b2am(F77Integer *k_b2am) { k_b2am_ = *k_b2am; }

void Variables::set_d_v2orb(F77Integer *d_v2orb) { d_v2orb_ = *d_v2orb; }

void Variables::set_k_v2_alpha_offset(F77Integer *k_v2_alpha_offset) {
  k_v2_alpha_offset_ = *k_v2_alpha_offset;
}
extern "C" {

// called in ccsd_t.F
void set_k2_cxx_(F77Integer *k_offset, F77Integer *k_evl_sorted) {
  Variables::set_k2(k_offset, k_evl_sorted);
}

void set_irrep_x_cxx_(F77Integer *irrep_x) { Variables::set_irrep_x(irrep_x); }

void set_irrep_y_cxx_(F77Integer *irrep_y) { Variables::set_irrep_y(irrep_y); }

void set_ova_cxx_(F77Integer *noa, F77Integer *nva) { Variables::set_ova(noa, nva); }
void set_var_cxx_(F77Integer *noab, F77Integer *nvab, F77Integer *int_mb, double *dbl_mb,
                  F77Integer *k_range, F77Integer *k_spin, F77Integer *k_sym,
                  logical *intorb, logical *restricted, F77Integer *irrep_v,
                  F77Integer *irrep_t, F77Integer *irrep_f) {
  Dummy::construct();
  Table::construct();
  Variables::set_ov(noab, nvab);
  Variables::set_idmb(int_mb, dbl_mb);
  Variables::set_k1(k_range, k_spin, k_sym);
  Variables::set_log(intorb, restricted);
  Variables::set_irrep(irrep_v, irrep_t, irrep_f);
}

void set_k_alpha_cxx_(F77Integer *k_alpha) { Variables::set_k_alpha(k_alpha); }

void set_k_b2am_cxx_(F77Integer *k_b2am) { Variables::set_k_b2am(k_b2am); }
void set_d_v2orb_cxx_(F77Integer *d_v2orb) { Variables::set_d_v2orb(d_v2orb); }

void set_k_v2_alpha_offset_cxx_(F77Integer *k_v2_alpha_offset) {
  Variables::set_k_v2_alpha_offset(k_v2_alpha_offset);
}
}

std::vector<RangeType> Table::range_;
// std::vector<Integer> Table::value_;
void Table::construct() {
  range_.resize(IndexNum);
  // value_.resize(IndexNum);
  for (int i = 0; i < pIndexNum; i++) range_[i] = TV;
  for (int i = pIndexNum; i < IndexNum; i++) range_[i] = TO;
}

} /* namespace tamm */
