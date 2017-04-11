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
#ifndef TAMM_TENSOR_VARIABLES_H_
#define TAMM_TENSOR_VARIABLES_H_

#include <typesf2c.h>
#include <vector>

#include "tensor/common.h"
#include "tensor/define.h"
#include "tensor/dummy.h"
#include "tensor/gmem.h"

namespace tamm {

/**
 * Global variables from FORTRAN
 */
class Variables {
 public:
  static void set_ov(F77Integer *o, F77Integer *v);
  static void set_ova(F77Integer *noa, F77Integer *nva);
  static void set_idmb(F77Integer *int_mb, double *dbl_mb);
  static void set_irrep(F77Integer *irrep_v, F77Integer *irrep_t, F77Integer *irrep_f);
  static void set_irrep_x(F77Integer *irrep_x);
  static void set_irrep_y(F77Integer *irrep_y);
  static void set_k1(F77Integer *k_range, F77Integer *k_spin, F77Integer *k_sym);
  static void set_log(logical *intorb, logical *restricted);
  static void set_k2(F77Integer *k_offset, F77Integer *k_evl_sorted);
  static void set_k_alpha(F77Integer *k_alpha);
  static void set_k_b2am(F77Integer *k_b2am);
  static void set_d_v2orb(F77Integer *d_v2orb);
  static void set_k_v2_alpha_offset(F77Integer *k_v2_alpha_offset);

  /**
   * Set k_offset and k_evl_sorted, use in ccsd(t) to compute r and o value
   */
  static void set_k2_cxx_(F77Integer *k_offset, F77Integer *k_evl_sorted);

  static void set_irrep_x_cxx_(F77Integer *irrep_x);

  static void set_irrep_y_cxx_(F77Integer *irrep_y);

  /**
   * Set FORTRAN global parameters
   */
  static void set_var_cxx_(F77Integer *noab, F77Integer *nvab, F77Integer *int_mb,
                           double *dbl_mb, F77Integer *k_range, F77Integer *k_spin,
                           F77Integer *k_sym, logical *intorb, logical *restricted,
                           F77Integer *irrep_v, F77Integer *irrep_t,
                           F77Integer *irrep_f);

  static const F77Integer &noab() { return noab_; }
  static const F77Integer &nvab() { return nvab_; }
  static const F77Integer &noa() { return noa_; }
  static const F77Integer &nva() { return nva_; }
  static const F77Integer &k_range() { return k_range_; }
  static const F77Integer &k_spin() { return k_spin_; }
  static const F77Integer &k_sym() { return k_sym_; }
  static const F77Integer &k_offset() { return k_offset_; }
  static const F77Integer &k_evl_sorted() { return k_evl_sorted_; }
  static const F77Integer &irrep_v() { return irrep_v_; }
  static const F77Integer &irrep_t() { return irrep_t_; }
  static const F77Integer &irrep_f() { return irrep_f_; }
  static const F77Integer &irrep_x() { return irrep_x_; }
  static const F77Integer &irrep_y() { return irrep_y_; }
  static const F77Integer &intorb() { return intorb_; }
  static const logical &restricted() { return restricted_; }
  static F77Integer *int_mb() { return int_mb_; }
  static double *dbl_mb() { return dbl_mb_; }

  static F77Integer k_alpha() { return k_alpha_; }
  static F77Integer k_b2am() { return k_b2am_; }
  static F77Integer d_v2orb() { return d_v2orb_; }
  static F77Integer k_v2_alpha_offset() { return k_v2_alpha_offset_; }
  static size_t k_range(Tile t) { return int_mb()[k_range() - 1 + t]; }

 private:
  static F77Integer noab_, nvab_;
  static F77Integer noa_, nva_;
  static F77Integer *int_mb_;
  static double *dbl_mb_;
  static F77Integer k_range_, k_spin_, k_sym_;
  static F77Integer k_offset_, k_evl_sorted_;
  static F77Integer irrep_v_, irrep_t_, irrep_f_, irrep_x_, irrep_y_;
  static logical intorb_, restricted_;
  static F77Integer k_alpha_, k_b2am_, d_v2orb_;
  static F77Integer k_v2_alpha_offset_;
};

/**
 * Global table that stores the current value of the indices
 */
class Table {
 public:
  static void construct();
  /**
   * Return the range type of idx
   * @param[in] idx name of a index
   * @return Rangetype of input idx
   */
  static const RangeType &rangeOf(const IndexName &idx) { return range_[idx]; }

  /**
   * Return the value of whole table
   */
  // static std::vector<Integer>& value() { return value_; }
 private:
  static std::vector<RangeType> range_; /*< range of the indices */
  // static std::vector<Integer> value_; /*< value of the indices */
};

/**
 * Timer class, easy to store number of calling and execution time
 */
class Timer {
 public:
  static int dg_num;  // DGEMM
  static int ah_num;  // ADD_HASH_BLOCK
  static int sa_num;  // SORT_ACC
  static int so_num;  // SORT
  static double total;
  static double dg_time;
  static double ah_time;
  static double sa_time;
  static double so_time;
};

class MPI_Timer {
 public:
  static int timer_num[40];
  static double timer_value[40];
  static double timer_cum_value[40];
  static char const *timer_text[];
};

} /* namespace tamm */

#endif  // TAMM_TENSOR_VARIABLES_H_
