#ifndef __tce_variables_h__
#define __tce_variables_h__

#include "common.h"
#include "typesf2c.h"
#include "define.h"
#include "dummy.h"
#include "ga.h"

namespace ctce {

  /**
   * Global variables from FORTRAN
   */
  class Variables {
    private:
      static Integer noab_, nvab_;
      static Integer *int_mb_;
      static double *dbl_mb_;
      static Integer k_range_, k_spin_, k_sym_;
      static Integer k_offset_, k_evl_sorted_;
      static Integer irrep_v_, irrep_t_, irrep_f_, irrep_x_;
      static logical intorb_, restricted_;
    static Integer k_alpha_;
    public:
      static void set_ov(Integer *o, Integer *v);
      static void set_idmb(Integer *int_mb, double *dbl_mb);
      static void set_irrep(Integer *irrep_v, Integer *irrep_t, Integer *irrep_f);
      static void set_irrep_x(Integer *irrep_x);
      static void set_k1(Integer *k_range, Integer *k_spin, Integer *k_sym);
      static void set_log(logical *intorb, logical *restricted);
      static void set_k2(Integer *k_offset, Integer *k_evl_sorted);
    static void set_k_alpha(Integer *k_alpha);

      /** 
       * Set k_offset and k_evl_sorted, use in ccsd(t) to compute r and o value
       */
      static void set_k2_cxx_(Integer *k_offset, Integer *k_evl_sorted);

      static void set_irrep_x_cxx_(Integer *irrep_x);

      /** 
       * Set FORTRAN global parameters
       */
      static void set_var_cxx_(Integer* noab, Integer* nvab, 
          Integer* int_mb, double* dbl_mb,
          Integer* k_range, Integer *k_spin, Integer *k_sym, 
          logical *intorb, logical *restricted,
          Integer *irrep_v, Integer *irrep_t, Integer *irrep_f);

      static const Integer& noab() { return noab_; }
      static const Integer& nvab() { return nvab_; }
      static const Integer& k_range() { return k_range_; }
      static const Integer& k_spin() { return k_spin_; }
      static const Integer& k_sym() { return k_sym_; }
      static const Integer& k_offset() { return k_offset_; }
      static const Integer& k_evl_sorted() { return k_evl_sorted_; }
      static const Integer& irrep_v() { return irrep_v_; }
      static const Integer& irrep_t() { return irrep_t_; }
      static const Integer& irrep_f() { return irrep_f_; }
      static const Integer& irrep_x() { return irrep_x_; }
      static const Integer& intorb() { return intorb_; }
      static const logical& restricted() { return restricted_; }
      static Integer* int_mb() { return int_mb_; }
      static double* dbl_mb() { return dbl_mb_; }

    static Integer k_alpha() { return k_alpha_; }
    static size_t k_range(Tile t) {
      return int_mb()[k_range()-1+t];
    }
  };

  /**
   * Global table that stores the current value of the indices
   */
  class Table {
    private:
      static std::vector<RangeType> range_; /*< range of the indices */
      //static std::vector<Integer> value_; /*< value of the indices */
    public:
      static void construct();
      /**
       * Return the range type of idx
       * @param[in] idx name of a index
       * @return Rangetype of input idx
       */
      static const RangeType& rangeOf(const IndexName& idx) { return range_[idx]; }

      /**
       * Return the value of whole table
       */
      //static std::vector<Integer>& value() { return value_; }
  };

  /**
   * Timer class, easy to store number of calling and execution time
   */
  class Timer {
    public:
      static int dg_num; // DGEMM
      static int ah_num; // ADD_HASH_BLOCK
      static int sa_num; // SORT_ACC
      static int so_num; // SORT
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
      static char* timer_text[];
  };

} /* namespace ctce */

#endif

