#pragma once

#include "ccsd_t_common.hpp"

//
// created by tc_gen_definition()
constexpr unsigned int FUSION_SIZE_SLICE_1_H3 = 4;
constexpr unsigned int FUSION_SIZE_SLICE_1_H2 = 4;
constexpr unsigned int FUSION_SIZE_SLICE_1_H1 = 4;
constexpr unsigned int FUSION_SIZE_SLICE_1_P6 = 4;
constexpr unsigned int FUSION_SIZE_SLICE_1_P5 = 4;
constexpr unsigned int FUSION_SIZE_SLICE_1_P4 = 4;
constexpr unsigned int FUSION_SIZE_SLICE_1_H7 = 16;

constexpr unsigned int FUSION_SIZE_SLICE_2_H3 = 4;
constexpr unsigned int FUSION_SIZE_SLICE_2_H2 = 4;
constexpr unsigned int FUSION_SIZE_SLICE_2_H1 = 4;
constexpr unsigned int FUSION_SIZE_SLICE_2_P6 = 4;
constexpr unsigned int FUSION_SIZE_SLICE_2_P5 = 4;
constexpr unsigned int FUSION_SIZE_SLICE_2_P4 = 4;
constexpr unsigned int FUSION_SIZE_SLICE_2_H7 = 16;

constexpr unsigned int FUSION_SIZE_INT_UNIT = FUSION_SIZE_SLICE_1_H7;

constexpr unsigned int FUSION_SIZE_TB_1_X = FUSION_SIZE_SLICE_1_H3 * FUSION_SIZE_SLICE_1_H2;
constexpr unsigned int FUSION_SIZE_TB_1_Y = FUSION_SIZE_SLICE_1_P6 * FUSION_SIZE_SLICE_1_H1;
constexpr unsigned int FUSION_SIZE_REG_1_X = FUSION_SIZE_SLICE_1_P5;
constexpr unsigned int FUSION_SIZE_REG_1_Y = FUSION_SIZE_SLICE_1_P4;

constexpr unsigned int FUSION_SIZE_TB_2_X = FUSION_SIZE_SLICE_2_H3 * FUSION_SIZE_SLICE_2_H2;
constexpr unsigned int FUSION_SIZE_TB_2_Y = FUSION_SIZE_SLICE_2_P4 * FUSION_SIZE_SLICE_2_H1;
constexpr unsigned int FUSION_SIZE_REG_2_X = FUSION_SIZE_SLICE_2_P5;
constexpr unsigned int FUSION_SIZE_REG_2_Y = FUSION_SIZE_SLICE_2_P6;

template <typename T1, typename T2>
constexpr auto CEIL(T1 a, T2 b) {return (a + b - 1) / b; };

constexpr unsigned int NUM_D1_EQUATIONS = 9;
constexpr unsigned int NUM_D2_EQUATIONS = 9;
constexpr unsigned int NUM_D1_INDEX = 7;
constexpr unsigned int NUM_D2_INDEX = 7;

template<typename T>
using localAcc = sycl::accessor<T, 1, sycl::access_mode::read_write, sycl::target::local>;

// 64 KB = 65536 bytes = 16384 (int) = 8192 (size_t)
// 9 * 9 * noab = 81 * noab

//
// 	|constant memory| = sizeof(int) * {(6 + 9) + ((7 + 9) * MAX_NOAB) + ((7 + 9) * MAX_NVAB)}
// 										= 4 bytes * (15 + 16 * 20 + 16 * 80) = 8 bytes * (15 + 320 + 1280) = 1615 * 4
// bytes = 6460 bytes (6.30 KB)
//

constexpr unsigned short COL = 64;

template<typename T>
__attribute__((always_inline))
void revised_jk_ccsd_t_fully_fused_kernel(
  int size_noab, int size_nvab,
  // 	common
  int size_max_dim_s1_t1, int size_max_dim_s1_v2, int size_max_dim_d1_t2, int size_max_dim_d1_v2,
  int size_max_dim_d2_t2, int size_max_dim_d2_v2,
  //
  T* __restrict__ df_dev_d1_t2_all, T* __restrict__ df_dev_d1_v2_all,
  T* __restrict__ df_dev_d2_t2_all, T* __restrict__ df_dev_d2_v2_all,
  T* __restrict__ df_dev_s1_t1_all, T* __restrict__ df_dev_s1_v2_all,
  //  energies
  T* __restrict__ dev_evl_sorted_h1b, T* __restrict__ dev_evl_sorted_h2b,
  T* __restrict__ dev_evl_sorted_h3b, T* __restrict__ dev_evl_sorted_p4b,
  T* __restrict__ dev_evl_sorted_p5b, T* __restrict__ dev_evl_sorted_p6b,
  // 	not-fully reduced results
  T* __restrict__ reduced_energy,
  //  common
  int num_blks_h3b, int num_blks_h2b, int num_blks_h1b, int num_blks_p6b, int num_blks_p5b,
  int num_blks_p4b,
  //
  int base_size_h1b, int base_size_h2b, int base_size_h3b, int base_size_p4b, int base_size_p5b,
  int base_size_p6b, sycl::nd_item<2>& item, const int* __restrict__ const_df_s1_size,
  const int* __restrict__ const_df_s1_exec, const int* __restrict__ const_df_d1_size,
  const int* __restrict__ const_df_d1_exec, const int* __restrict__ const_df_d2_size,
  const int* __restrict__ const_df_d2_exec, sycl::local_ptr<T> sm_a, sycl::local_ptr<T> sm_b) {
  sycl::group thread_block = item.get_group();
  size_t      threadIdx_x  = item.get_local_id(1);
  size_t      threadIdx_y  = item.get_local_id(0);
  size_t      blockIdx_x   = item.get_group(1);

  short internal_upperbound = 0;
  short internal_offset = 0;

  // should support for non-full tiles
  int idx_h3 = threadIdx_x % FUSION_SIZE_SLICE_1_H3;
  int idx_h2 = threadIdx_x / FUSION_SIZE_SLICE_1_H3;
  int idx_p6 = threadIdx_y % FUSION_SIZE_SLICE_1_P6;
  int idx_h1 = threadIdx_y / FUSION_SIZE_SLICE_1_P6;

  int blk_idx_p4b =
    blockIdx_x / (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b * num_blks_p5b);
  int tmp_blkIdx =
    blockIdx_x % (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b * num_blks_p5b);
  int blk_idx_p5b = (tmp_blkIdx) / (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b);
  tmp_blkIdx      = (tmp_blkIdx) % (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b);
  int blk_idx_p6b = (tmp_blkIdx) / (num_blks_h3b * num_blks_h2b * num_blks_h1b);
  tmp_blkIdx      = (tmp_blkIdx) % (num_blks_h3b * num_blks_h2b * num_blks_h1b);
  int blk_idx_h1b = (tmp_blkIdx) / (num_blks_h3b * num_blks_h2b);
  tmp_blkIdx      = (tmp_blkIdx) % (num_blks_h3b * num_blks_h2b);
  int blk_idx_h2b = (tmp_blkIdx) / (num_blks_h3b);
  int blk_idx_h3b = blockIdx_x % (num_blks_h3b);

  int str_blk_idx_h3 = blk_idx_h3b * FUSION_SIZE_SLICE_1_H3;
  int str_blk_idx_h2 = blk_idx_h2b * FUSION_SIZE_SLICE_1_H2;
  int str_blk_idx_h1 = blk_idx_h1b * FUSION_SIZE_SLICE_1_H1;
  int str_blk_idx_p6 = blk_idx_p6b * FUSION_SIZE_SLICE_1_P6;
  int str_blk_idx_p5 = blk_idx_p5b * FUSION_SIZE_SLICE_1_P5;
  int str_blk_idx_p4 = blk_idx_p4b * FUSION_SIZE_SLICE_1_P4;

  //
  short rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;
  short energy_rng_h3, energy_rng_h2, energy_rng_h1, energy_rng_p6, energy_rng_p5, energy_rng_p4;
  energy_rng_h3 = ((base_size_h3b - str_blk_idx_h3) >= FUSION_SIZE_SLICE_1_H3) ? FUSION_SIZE_SLICE_1_H3 : (base_size_h3b % FUSION_SIZE_SLICE_1_H3);
  energy_rng_h2 = ((base_size_h2b - str_blk_idx_h2) >= FUSION_SIZE_SLICE_1_H2) ? FUSION_SIZE_SLICE_1_H2 : (base_size_h2b % FUSION_SIZE_SLICE_1_H2);
  energy_rng_h1 = ((base_size_h1b - str_blk_idx_h1) >= FUSION_SIZE_SLICE_1_H1) ? FUSION_SIZE_SLICE_1_H1 : (base_size_h1b % FUSION_SIZE_SLICE_1_H1);
  energy_rng_p6 = ((base_size_p6b - str_blk_idx_p6) >= FUSION_SIZE_SLICE_1_P6) ? FUSION_SIZE_SLICE_1_P6 : (base_size_p6b % FUSION_SIZE_SLICE_1_P6);
  energy_rng_p5 = ((base_size_p5b - str_blk_idx_p5) >= FUSION_SIZE_SLICE_1_P5) ? FUSION_SIZE_SLICE_1_P5 : (base_size_p5b % FUSION_SIZE_SLICE_1_P5);
  energy_rng_p4 = ((base_size_p4b - str_blk_idx_p4) >= FUSION_SIZE_SLICE_1_P4) ? FUSION_SIZE_SLICE_1_P4 : (base_size_p4b % FUSION_SIZE_SLICE_1_P4);

  //
  T temp_av{0};
  T temp_bv[4]        = {0};
  T reg_tile[4][4]    = {{0}};
  T reg_singles[4][4] = {{0}};

  int base_size_h7b, base_size_p7b;

  int energy_str_blk_idx_p4 = str_blk_idx_p4;
  int energy_str_blk_idx_p5 = str_blk_idx_p5;
  T   eval_h3               = dev_evl_sorted_h3b[str_blk_idx_h3 + idx_h3];
  T   eval_h2               = dev_evl_sorted_h2b[str_blk_idx_h2 + idx_h2];
  T   eval_p6               = dev_evl_sorted_p6b[str_blk_idx_p6 + idx_p6];
  T   eval_h1               = dev_evl_sorted_h1b[str_blk_idx_h1 + idx_h1];

  T partial_inner_factor = eval_h3 + eval_h2 + eval_h1 - eval_p6;

#pragma unroll 1
  for(int iter_noab = 0; iter_noab < size_noab; iter_noab++) {
    //
    int flag_d1_1 = const_df_d1_exec[0 + (iter_noab) *NUM_D1_EQUATIONS];
    int flag_d1_2 = const_df_d1_exec[1 + (iter_noab) *NUM_D1_EQUATIONS];
    int flag_d1_3 = const_df_d1_exec[2 + (iter_noab) *NUM_D1_EQUATIONS];

    //
    base_size_h1b = const_df_d1_size[0 + (iter_noab) *NUM_D1_INDEX];
    base_size_h2b = const_df_d1_size[1 + (iter_noab) *NUM_D1_INDEX];
    base_size_h3b = const_df_d1_size[2 + (iter_noab) *NUM_D1_INDEX];
    base_size_h7b = const_df_d1_size[3 + (iter_noab) *NUM_D1_INDEX];
    base_size_p4b = const_df_d1_size[4 + (iter_noab) *NUM_D1_INDEX];
    base_size_p5b = const_df_d1_size[5 + (iter_noab) *NUM_D1_INDEX];
    base_size_p6b = const_df_d1_size[6 + (iter_noab) *NUM_D1_INDEX];

    //
    num_blks_h1b = CEIL(base_size_h1b, FUSION_SIZE_SLICE_1_H1);
    num_blks_h2b = CEIL(base_size_h2b, FUSION_SIZE_SLICE_1_H2);
    num_blks_h3b = CEIL(base_size_h3b, FUSION_SIZE_SLICE_1_H3);
    num_blks_p4b = CEIL(base_size_p4b, FUSION_SIZE_SLICE_1_P4);
    num_blks_p5b = CEIL(base_size_p5b, FUSION_SIZE_SLICE_1_P5);
    num_blks_p6b = CEIL(base_size_p6b, FUSION_SIZE_SLICE_1_P6);

    // 	(2) blk_idx_h/p*b
    blk_idx_p4b =
      blockIdx_x / (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b * num_blks_p5b);
    tmp_blkIdx =
      blockIdx_x % (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b * num_blks_p5b);
    blk_idx_p5b = (tmp_blkIdx) / (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b);
    tmp_blkIdx  = (tmp_blkIdx) % (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b);
    blk_idx_p6b = (tmp_blkIdx) / (num_blks_h3b * num_blks_h2b * num_blks_h1b);
    tmp_blkIdx  = (tmp_blkIdx) % (num_blks_h3b * num_blks_h2b * num_blks_h1b);
    blk_idx_h1b = (tmp_blkIdx) / (num_blks_h3b * num_blks_h2b);
    tmp_blkIdx  = (tmp_blkIdx) % (num_blks_h3b * num_blks_h2b);
    blk_idx_h2b = (tmp_blkIdx) / (num_blks_h3b);
    blk_idx_h3b = blockIdx_x % (num_blks_h3b);

    // 	(3) str_blk_idx_h/p*
    str_blk_idx_h3 = blk_idx_h3b * FUSION_SIZE_SLICE_1_H3;
    str_blk_idx_h2 = blk_idx_h2b * FUSION_SIZE_SLICE_1_H2;
    str_blk_idx_h1 = blk_idx_h1b * FUSION_SIZE_SLICE_1_H1;
    str_blk_idx_p6 = blk_idx_p6b * FUSION_SIZE_SLICE_1_P6;
    str_blk_idx_p5 = blk_idx_p5b * FUSION_SIZE_SLICE_1_P5;
    str_blk_idx_p4 = blk_idx_p4b * FUSION_SIZE_SLICE_1_P4;

    // 	(4) rng_h/p*
    rng_h3 = ((base_size_h3b - str_blk_idx_h3) >= FUSION_SIZE_SLICE_1_H3) ? FUSION_SIZE_SLICE_1_H3 : (base_size_h3b % FUSION_SIZE_SLICE_1_H3);
    rng_h2 = ((base_size_h2b - str_blk_idx_h2) >= FUSION_SIZE_SLICE_1_H2) ? FUSION_SIZE_SLICE_1_H2 : (base_size_h2b % FUSION_SIZE_SLICE_1_H2);
    rng_h1 = ((base_size_h1b - str_blk_idx_h1) >= FUSION_SIZE_SLICE_1_H1) ? FUSION_SIZE_SLICE_1_H1 : (base_size_h1b % FUSION_SIZE_SLICE_1_H1);
    rng_p6 = ((base_size_p6b - str_blk_idx_p6) >= FUSION_SIZE_SLICE_1_P6) ? FUSION_SIZE_SLICE_1_P6 : (base_size_p6b % FUSION_SIZE_SLICE_1_P6);
    rng_p5 = ((base_size_p5b - str_blk_idx_p5) >= FUSION_SIZE_SLICE_1_P5) ? FUSION_SIZE_SLICE_1_P5 : (base_size_p5b % FUSION_SIZE_SLICE_1_P5);
    rng_p4 = ((base_size_p4b - str_blk_idx_p4) >= FUSION_SIZE_SLICE_1_P4) ? FUSION_SIZE_SLICE_1_P4 : (base_size_p4b % FUSION_SIZE_SLICE_1_P4);

    //  sd1_1
    if(flag_d1_1 >= 0) {
      //
      T* tmp_dev_d1_t2 = df_dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_1;
      T* tmp_dev_d1_v2 = df_dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_1;

      //
      internal_upperbound = 0;
#pragma unroll 1
      for(int l = 0; l < base_size_h7b; l += FUSION_SIZE_INT_UNIT) {
        // Part: Generalized Contraction Index (p7b)
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - base_size_h7b;
        if(internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of size_internal Indices: 1
        if(idx_p6 < rng_p4 && idx_h1 < rng_h1 &&
           threadIdx_x < FUSION_SIZE_INT_UNIT - internal_upperbound)
          for(short ll = 0; ll < rng_p5; ll++) {
            sm_a[threadIdx_y + ll * FUSION_SIZE_TB_2_Y + (threadIdx_x * COL)] =
              tmp_dev_d1_t2[(str_blk_idx_p4 + idx_p6 +
                             (str_blk_idx_p5 + ll + (str_blk_idx_h1 + idx_h1) * base_size_p5b) *
                               base_size_p4b) *
                              base_size_h7b +
                            (threadIdx_x + l)];
          }

        // Load Input Tensor to Shared Memory
        if(idx_h3 < rng_h3 && idx_h2 < rng_h2 &&
           threadIdx_y < FUSION_SIZE_INT_UNIT - internal_upperbound) {
          for(short ll = 0; ll < rng_p6; ll++) {
            sm_b[threadIdx_x + ll * FUSION_SIZE_TB_2_X + (threadIdx_y * COL)] =
              tmp_dev_d1_v2[str_blk_idx_h3 + idx_h3 +
                            (str_blk_idx_h2 + idx_h2 +
                             (str_blk_idx_p6 + ll + (threadIdx_y + l) * base_size_p6b) *
                               base_size_h2b) *
                              base_size_h3b];
          }
        }
        sycl::group_barrier(thread_block);

        // Cross-Product: -1
        // Part: Generalized Threads
        for(short ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++) {
          temp_bv[0] = sm_b[idx_h3 + idx_h2 * FUSION_SIZE_SLICE_2_H3 + 0  + (ll * COL)];
          temp_bv[1] = sm_b[idx_h3 + idx_h2 * FUSION_SIZE_SLICE_2_H3 + 16 + (ll * COL)];
          temp_bv[2] = sm_b[idx_h3 + idx_h2 * FUSION_SIZE_SLICE_2_H3 + 32 + (ll * COL)];
          temp_bv[3] = sm_b[idx_h3 + idx_h2 * FUSION_SIZE_SLICE_2_H3 + 48 + (ll * COL)];

          for(unsigned short xx = 0; xx < 4; xx++) {
            temp_av = sm_a[idx_p6 + idx_h1 * FUSION_SIZE_SLICE_1_P6 + (xx * 16) + (ll * COL)];

            reg_tile[0][xx] -= temp_av * temp_bv[0];
            reg_tile[1][xx] -= temp_av * temp_bv[1];
            reg_tile[2][xx] -= temp_av * temp_bv[2];
            reg_tile[3][xx] -= temp_av * temp_bv[3];
          }
        }
        sycl::group_barrier(thread_block);
      }
    }

    //  sd1_2
    if(flag_d1_2 >= 0) {
      //
      T* tmp_dev_d1_t2 = df_dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_2;
      T* tmp_dev_d1_v2 = df_dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_2;

      internal_upperbound = 0;
#pragma unroll 1
      for(int l = 0; l < base_size_h7b; l += FUSION_SIZE_INT_UNIT) {
        // Part: Generalized Contraction Index (p7b)
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - base_size_h7b;
        if(internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of size_internal Indices: 1
        if(idx_p6 < rng_p4 && idx_h1 < rng_h2 &&
           threadIdx_x < FUSION_SIZE_INT_UNIT - internal_upperbound) {
          for(short ll = 0; ll < rng_p5; ll++) {
            sm_a[threadIdx_y + ll * FUSION_SIZE_TB_2_Y + (threadIdx_x * COL)] =
              tmp_dev_d1_t2[(str_blk_idx_p4 + idx_p6 +
                             (str_blk_idx_p5 + ll + (str_blk_idx_h2 + idx_h1) * base_size_p5b) *
                               base_size_p4b) *
                              base_size_h7b +
                            (threadIdx_x + l)];
          }
        }

        // Load Input Tensor to Shared Memory
        if(idx_h3 < rng_h3 && idx_h2 < rng_h1 &&
           threadIdx_y < FUSION_SIZE_INT_UNIT - internal_upperbound)
          for(short ll = 0; ll < rng_p6; ll++) {
            sm_b[threadIdx_x + ll * FUSION_SIZE_TB_2_X + (threadIdx_y * COL)] =
              tmp_dev_d1_v2[str_blk_idx_h3 + idx_h3 +
                            (str_blk_idx_h1 + idx_h2 +
                             (str_blk_idx_p6 + ll + (threadIdx_y + l) * base_size_p6b) *
                               base_size_h1b) *
                              base_size_h3b];
          }
        sycl::group_barrier(thread_block);

        // Cross-Product: -1
        // Part: Generalized Threads
        for(short ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++) {
          temp_bv[0] = sm_b[idx_h3 + idx_h1 * FUSION_SIZE_SLICE_2_H3 + 0  + (ll * COL)];
          temp_bv[1] = sm_b[idx_h3 + idx_h1 * FUSION_SIZE_SLICE_2_H3 + 16 + (ll * COL)];
          temp_bv[2] = sm_b[idx_h3 + idx_h1 * FUSION_SIZE_SLICE_2_H3 + 32 + (ll * COL)];
          temp_bv[3] = sm_b[idx_h3 + idx_h1 * FUSION_SIZE_SLICE_2_H3 + 48 + (ll * COL)];

          for(unsigned short xx = 0; xx < 4; xx++) {
            temp_av = sm_a[idx_p6 + idx_h2 * FUSION_SIZE_SLICE_2_P4 + (xx * 16) + (ll * COL)];

            reg_tile[0][xx] += temp_av * temp_bv[0];
            reg_tile[1][xx] += temp_av * temp_bv[1];
            reg_tile[2][xx] += temp_av * temp_bv[2];
            reg_tile[3][xx] += temp_av * temp_bv[3];
          }
        }
        sycl::group_barrier(thread_block);
      }
    }

    //  sd1_3
    if(flag_d1_3 >= 0) {
      //
      T* tmp_dev_d1_t2 = df_dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_3;
      T* tmp_dev_d1_v2 = df_dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_3;

      internal_upperbound = 0;
#pragma unroll 1
      for(int l = 0; l < base_size_h7b; l += FUSION_SIZE_INT_UNIT) {
        // Part: Generalized Contraction Index (p7b)
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - base_size_h7b;
        if(internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of size_internal Indices: 1
        if(idx_p6 < rng_p4 && idx_h1 < rng_h3 &&
           threadIdx_x < FUSION_SIZE_INT_UNIT - internal_upperbound)
          for(short ll = 0; ll < rng_p5; ll++) {
            sm_a[threadIdx_y + ll * FUSION_SIZE_TB_2_Y + (threadIdx_x * COL)] =
              tmp_dev_d1_t2[(str_blk_idx_p4 + idx_p6 +
                             (str_blk_idx_p5 + ll + (str_blk_idx_h3 + idx_h1) * base_size_p5b) *
                               base_size_p4b) *
                              base_size_h7b +
                            (threadIdx_x + l)];
          }

        // Load Input Tensor to Shared Memory
        if(idx_h3 < rng_h2 && idx_h2 < rng_h1 &&
           threadIdx_y < FUSION_SIZE_INT_UNIT - internal_upperbound)
          for(short ll = 0; ll < rng_p6; ll++) {
            sm_b[threadIdx_x + ll * FUSION_SIZE_TB_2_X + (threadIdx_y * COL)] = tmp_dev_d1_v2[(
              str_blk_idx_h2 + idx_h3 +
              (str_blk_idx_h1 + idx_h2 +
               (str_blk_idx_p6 + ll + (threadIdx_y + l) * base_size_p6b) * base_size_h1b) *
                base_size_h2b)];
          }
        sycl::group_barrier(thread_block);

        // Cross-Product: -1
        // Part: Generalized Threads
        for(short ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++) {
          temp_bv[0] = sm_b[idx_h2 + idx_h1 * FUSION_SIZE_SLICE_2_H2 + 0  + (ll * COL)];
          temp_bv[1] = sm_b[idx_h2 + idx_h1 * FUSION_SIZE_SLICE_2_H2 + 16 + (ll * COL)];
          temp_bv[2] = sm_b[idx_h2 + idx_h1 * FUSION_SIZE_SLICE_2_H2 + 32 + (ll * COL)];
          temp_bv[3] = sm_b[idx_h2 + idx_h1 * FUSION_SIZE_SLICE_2_H2 + 48 + (ll * COL)];

          for(unsigned short xx = 0; xx < 4; xx++) {
            temp_av = sm_a[idx_p6 + idx_h3 * FUSION_SIZE_SLICE_2_P4 + (xx * 16) + (ll * COL)];

            reg_tile[0][xx] -= temp_av * temp_bv[0];
            reg_tile[1][xx] -= temp_av * temp_bv[1];
            reg_tile[2][xx] -= temp_av * temp_bv[2];
            reg_tile[3][xx] -= temp_av * temp_bv[3];
          }
        }
        sycl::group_barrier(thread_block);
      }
    }
  }

  //  d2-top: sd2_7, 8 and 9
#pragma unroll 1
  for(int iter_nvab = 0; iter_nvab < size_nvab; iter_nvab++) {
    //
    int flag_d2_7 = const_df_d2_exec[6 + (iter_nvab) *NUM_D2_EQUATIONS];
    int flag_d2_8 = const_df_d2_exec[7 + (iter_nvab) *NUM_D2_EQUATIONS];
    int flag_d2_9 = const_df_d2_exec[8 + (iter_nvab) *NUM_D2_EQUATIONS];

    //
    base_size_h1b = const_df_d2_size[0 + (iter_nvab) *NUM_D2_INDEX];
    base_size_h2b = const_df_d2_size[1 + (iter_nvab) *NUM_D2_INDEX];
    base_size_h3b = const_df_d2_size[2 + (iter_nvab) *NUM_D2_INDEX];
    base_size_p4b = const_df_d2_size[3 + (iter_nvab) *NUM_D2_INDEX];
    base_size_p5b = const_df_d2_size[4 + (iter_nvab) *NUM_D2_INDEX];
    base_size_p6b = const_df_d2_size[5 + (iter_nvab) *NUM_D2_INDEX];
    base_size_p7b = const_df_d2_size[6 + (iter_nvab) *NUM_D2_INDEX];

    //
    num_blks_h1b = CEIL(base_size_h1b, FUSION_SIZE_SLICE_1_H1);
    num_blks_h2b = CEIL(base_size_h2b, FUSION_SIZE_SLICE_1_H2);
    num_blks_h3b = CEIL(base_size_h3b, FUSION_SIZE_SLICE_1_H3);
    num_blks_p4b = CEIL(base_size_p4b, FUSION_SIZE_SLICE_1_P4);
    num_blks_p5b = CEIL(base_size_p5b, FUSION_SIZE_SLICE_1_P5);
    num_blks_p6b = CEIL(base_size_p6b, FUSION_SIZE_SLICE_1_P6);

    // 	(2) blk_idx_h/p*b
    blk_idx_p4b =
      blockIdx_x / (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b * num_blks_p5b);
    tmp_blkIdx =
      blockIdx_x % (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b * num_blks_p5b);
    blk_idx_p5b = (tmp_blkIdx) / (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b);
    tmp_blkIdx  = (tmp_blkIdx) % (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b);
    blk_idx_p6b = (tmp_blkIdx) / (num_blks_h3b * num_blks_h2b * num_blks_h1b);
    tmp_blkIdx  = (tmp_blkIdx) % (num_blks_h3b * num_blks_h2b * num_blks_h1b);
    blk_idx_h1b = (tmp_blkIdx) / (num_blks_h3b * num_blks_h2b);
    tmp_blkIdx  = (tmp_blkIdx) % (num_blks_h3b * num_blks_h2b);
    blk_idx_h2b = (tmp_blkIdx) / (num_blks_h3b);
    blk_idx_h3b = blockIdx_x % (num_blks_h3b);

    // 	(3) str_blk_idx_h/p*
    str_blk_idx_h3 = blk_idx_h3b * FUSION_SIZE_SLICE_1_H3;
    str_blk_idx_h2 = blk_idx_h2b * FUSION_SIZE_SLICE_1_H2;
    str_blk_idx_h1 = blk_idx_h1b * FUSION_SIZE_SLICE_1_H1;
    str_blk_idx_p6 = blk_idx_p6b * FUSION_SIZE_SLICE_1_P6;
    str_blk_idx_p5 = blk_idx_p5b * FUSION_SIZE_SLICE_1_P5;
    str_blk_idx_p4 = blk_idx_p4b * FUSION_SIZE_SLICE_1_P4;

    // 	(4) rng_h/p*
    rng_h3 = ((base_size_h3b - str_blk_idx_h3) >= FUSION_SIZE_SLICE_1_H3) ? FUSION_SIZE_SLICE_1_H3 : (base_size_h3b % FUSION_SIZE_SLICE_1_H3);
    rng_h2 = ((base_size_h2b - str_blk_idx_h2) >= FUSION_SIZE_SLICE_1_H2) ? FUSION_SIZE_SLICE_1_H2 : (base_size_h2b % FUSION_SIZE_SLICE_1_H2);
    rng_h1 = ((base_size_h1b - str_blk_idx_h1) >= FUSION_SIZE_SLICE_1_H1) ? FUSION_SIZE_SLICE_1_H1 : (base_size_h1b % FUSION_SIZE_SLICE_1_H1);
    rng_p6 = ((base_size_p6b - str_blk_idx_p6) >= FUSION_SIZE_SLICE_1_P6) ? FUSION_SIZE_SLICE_1_P6 : (base_size_p6b % FUSION_SIZE_SLICE_1_P6);
    rng_p5 = ((base_size_p5b - str_blk_idx_p5) >= FUSION_SIZE_SLICE_1_P5) ? FUSION_SIZE_SLICE_1_P5 : (base_size_p5b % FUSION_SIZE_SLICE_1_P5);
    rng_p4 = ((base_size_p4b - str_blk_idx_p4) >= FUSION_SIZE_SLICE_1_P4) ? FUSION_SIZE_SLICE_1_P4 : (base_size_p4b % FUSION_SIZE_SLICE_1_P4);

    //	sd2_7
    if(flag_d2_7 >= 0) {
      //
      T* tmp_dev_d2_t2_7 =
        df_dev_d2_t2_all +
        size_max_dim_d2_t2 * flag_d2_7; // const_list_d2_flags_offset[local_offset];
      T* tmp_dev_d2_v2_7 =
        df_dev_d2_v2_all +
        size_max_dim_d2_v2 * flag_d2_7; // const_list_d2_flags_offset[local_offset];

      //	sd2_7
      internal_upperbound = 0;
#pragma unroll 1
      for(int l = 0; l < base_size_p7b; l += FUSION_SIZE_INT_UNIT) {
        // Part: Generalized Contraction Index (p7b)
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - base_size_p7b;
        if(internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of size_internal Indices: 1
        if(idx_p6 < rng_h1 && idx_h1 < rng_h2 &&
           threadIdx_x < FUSION_SIZE_INT_UNIT - internal_upperbound)
          for(short ll = 0; ll < rng_p6; ll++) {
            sm_a[threadIdx_y + ll * FUSION_SIZE_TB_2_Y + (threadIdx_x * COL)] =
              tmp_dev_d2_t2_7[(blk_idx_p6b * FUSION_SIZE_SLICE_2_P6 + ll +
                               (str_blk_idx_h1 + idx_p6 +
                                (str_blk_idx_h2 + idx_h1) * base_size_h1b) *
                                 base_size_p6b) *
                                base_size_p7b +
                              (threadIdx_x + l)];
          }

        // Load Input Tensor to Shared Memory
        if(idx_p6 < rng_h3 && idx_h1 < rng_p4 &&
           threadIdx_x < FUSION_SIZE_INT_UNIT - internal_upperbound)
          for(short ll = 0; ll < rng_p5; ll++) {
            sm_b[threadIdx_y + ll * FUSION_SIZE_TB_2_Y + (threadIdx_x * COL)] =
              tmp_dev_d2_v2_7[(str_blk_idx_h3 + idx_p6 +
                               (str_blk_idx_p5 + ll + (str_blk_idx_p4 + idx_h1) * base_size_p5b) *
                                 base_size_h3b) *
                                base_size_p7b +
                              (threadIdx_x + l)];
          }
        sycl::group_barrier(thread_block);

        // Cross-Product: 16
        // Part: Generalized Threads
        for(short ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++) {
          temp_bv[0] = sm_a[idx_h1 + idx_h2 * FUSION_SIZE_SLICE_2_H1 + 0  + (ll * COL)];
          temp_bv[1] = sm_a[idx_h1 + idx_h2 * FUSION_SIZE_SLICE_2_H1 + 16 + (ll * COL)];
          temp_bv[2] = sm_a[idx_h1 + idx_h2 * FUSION_SIZE_SLICE_2_H1 + 32 + (ll * COL)];
          temp_bv[3] = sm_a[idx_h1 + idx_h2 * FUSION_SIZE_SLICE_2_H1 + 48 + (ll * COL)];

          for(unsigned short xx = 0; xx < 4; xx++) {
            temp_av = sm_b[idx_h3 + idx_p6 * FUSION_SIZE_SLICE_2_H3 + (xx * 16) + (ll * COL)];

            reg_tile[0][xx] -= temp_av * temp_bv[0];
            reg_tile[1][xx] -= temp_av * temp_bv[1];
            reg_tile[2][xx] -= temp_av * temp_bv[2];
            reg_tile[3][xx] -= temp_av * temp_bv[3];
          }
        }
        sycl::group_barrier(thread_block);
      }
    }

    // 	sd2_8
    if(flag_d2_8 >= 0) {
      //
      T* tmp_dev_d2_t2_8 =
        df_dev_d2_t2_all +
        size_max_dim_d2_t2 * flag_d2_8; // const_list_d2_flags_offset[local_offset];
      T* tmp_dev_d2_v2_8 =
        df_dev_d2_v2_all +
        size_max_dim_d2_v2 * flag_d2_8; // const_list_d2_flags_offset[local_offset];

      internal_upperbound = 0;
#pragma unroll 1
      for(int l = 0; l < base_size_p7b; l += FUSION_SIZE_INT_UNIT) {
        // Part: Generalized Contraction Index (p7b)
        // internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - base_size_p7b;
        if(internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of size_internal Indices: 1
        if(idx_p6 < rng_h2 && idx_h1 < rng_h3 &&
           threadIdx_x < FUSION_SIZE_INT_UNIT - internal_upperbound)
          for(short ll = 0; ll < rng_p6; ll++) {
            sm_a[threadIdx_y + ll * FUSION_SIZE_TB_2_Y + (threadIdx_x * COL)] =
              tmp_dev_d2_t2_8[(str_blk_idx_p6 + ll +
                               (str_blk_idx_h2 + idx_p6 +
                                (str_blk_idx_h3 + idx_h1) * base_size_h2b) *
                                 base_size_p6b) *
                                base_size_p7b +
                              (threadIdx_x + l)];
          }

        // Load Input Tensor to Shared Memory
        if(idx_p6 < rng_h1 && idx_h1 < rng_p4 &&
           threadIdx_x < FUSION_SIZE_INT_UNIT - internal_upperbound)
          for(short ll = 0; ll < rng_p5; ll++) {
            sm_b[threadIdx_y + ll * FUSION_SIZE_TB_2_Y + (threadIdx_x * COL)] =
              tmp_dev_d2_v2_8[(str_blk_idx_h1 + idx_p6 +
                               (str_blk_idx_p5 + ll + (str_blk_idx_p4 + idx_h1) * base_size_p5b) *
                                 base_size_h1b) *
                                base_size_p7b +
                              (threadIdx_x + l)];
          }
        sycl::group_barrier(thread_block);

        // Cross-Product: 16
        // Part: Generalized Threads
        for(short ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++) {
          temp_bv[0] = sm_a[idx_h2 + idx_h3 * FUSION_SIZE_SLICE_2_H2 + 0  + (ll * COL)];
          temp_bv[1] = sm_a[idx_h2 + idx_h3 * FUSION_SIZE_SLICE_2_H2 + 16 + (ll * COL)];
          temp_bv[2] = sm_a[idx_h2 + idx_h3 * FUSION_SIZE_SLICE_2_H2 + 32 + (ll * COL)];
          temp_bv[3] = sm_a[idx_h2 + idx_h3 * FUSION_SIZE_SLICE_2_H2 + 48 + (ll * COL)];

          for(unsigned short xx = 0; xx < 4; xx++) {
            temp_av = sm_b[idx_h1 + idx_p6 * FUSION_SIZE_SLICE_2_H1 + (xx * 16) + (ll * COL)];

            reg_tile[0][xx] -= temp_av * temp_bv[0];
            reg_tile[1][xx] -= temp_av * temp_bv[1];
            reg_tile[2][xx] -= temp_av * temp_bv[2];
            reg_tile[3][xx] -= temp_av * temp_bv[3];
          }
        }
        sycl::group_barrier(thread_block);
      }
    }

    // 	sd2_9
    if(flag_d2_9 >= 0) {
      //
      T* tmp_dev_d2_t2_9 =
        df_dev_d2_t2_all +
        size_max_dim_d2_t2 * flag_d2_9; // const_list_d2_flags_offset[local_offset];
      T* tmp_dev_d2_v2_9 =
        df_dev_d2_v2_all +
        size_max_dim_d2_v2 * flag_d2_9; // const_list_d2_flags_offset[local_offset];

      internal_upperbound = 0;
#pragma unroll 1
      for(int l = 0; l < base_size_p7b; l += FUSION_SIZE_INT_UNIT) {
        // Part: Generalized Contraction Index (p7b)
        // internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - base_size_p7b;
        if(internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of size_internal Indices: 1
        if(idx_p6 < rng_h1 && idx_h1 < rng_h3 &&
           threadIdx_x < FUSION_SIZE_INT_UNIT - internal_upperbound)
          for(short ll = 0; ll < rng_p6; ll++) {
            sm_a[threadIdx_y + ll * FUSION_SIZE_TB_2_Y + (threadIdx_x * COL)] =
              tmp_dev_d2_t2_9[(str_blk_idx_p6 + ll +
                               (str_blk_idx_h1 + idx_p6 +
                                (str_blk_idx_h3 + idx_h1) * base_size_h1b) *
                                 base_size_p6b) *
                                base_size_p7b +
                              (threadIdx_x + l)];
          }

        // Load Input Tensor to Shared Memory
        if(idx_p6 < rng_h2 && idx_h1 < rng_p4 &&
           threadIdx_x < FUSION_SIZE_INT_UNIT - internal_upperbound)
          for(short ll = 0; ll < rng_p5; ll++) {
            sm_b[threadIdx_y + ll * FUSION_SIZE_TB_2_Y + (threadIdx_x * COL)] =
              tmp_dev_d2_v2_9[(str_blk_idx_h2 + idx_p6 +
                               (str_blk_idx_p5 + ll + (str_blk_idx_p4 + idx_h1) * base_size_p5b) *
                                 base_size_h2b) *
                                base_size_p7b +
                              (threadIdx_x + l)];
          }
        sycl::group_barrier(thread_block);

        // Cross-Product: 16
        // Part: Generalized Threads
        for(short ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++) {
          temp_bv[0] = sm_a[idx_h1 + idx_h3 * FUSION_SIZE_SLICE_2_H1 + 0  + (ll * COL)];
          temp_bv[1] = sm_a[idx_h1 + idx_h3 * FUSION_SIZE_SLICE_2_H1 + 16 + (ll * COL)];
          temp_bv[2] = sm_a[idx_h1 + idx_h3 * FUSION_SIZE_SLICE_2_H1 + 32 + (ll * COL)];
          temp_bv[3] = sm_a[idx_h1 + idx_h3 * FUSION_SIZE_SLICE_2_H1 + 48 + (ll * COL)];

          for(unsigned short xx = 0; xx < 4; xx++) {
            temp_av = sm_b[idx_h2 + idx_p6 * FUSION_SIZE_SLICE_2_H2 + (xx * 16) + (ll * COL)];

            reg_tile[0][xx] += temp_av * temp_bv[0];
            reg_tile[1][xx] += temp_av * temp_bv[1];
            reg_tile[2][xx] += temp_av * temp_bv[2];
            reg_tile[3][xx] += temp_av * temp_bv[3];
          }
        }
        sycl::group_barrier(thread_block);
      }
    }
  }

  //
  //  Register Transpose (top - bottom)
  //
  {
    if(threadIdx_y < 4) // 0, 1, 2, 3
    {
      // sm_a[16][64] <-- (4 x 16) x (4 x 4) = (16 x 64)		  'y''x'
      sm_a[threadIdx_x      + (0 + threadIdx_y * 4) * COL] = reg_tile[0][0];
      sm_a[threadIdx_x      + (1 + threadIdx_y * 4) * COL] = reg_tile[1][0];
      sm_a[threadIdx_x      + (2 + threadIdx_y * 4) * COL] = reg_tile[2][0];
      sm_a[threadIdx_x      + (3 + threadIdx_y * 4) * COL] = reg_tile[3][0];

      sm_a[threadIdx_x + 16 + (0 + threadIdx_y * 4) * COL] = reg_tile[0][1];
      sm_a[threadIdx_x + 16 + (1 + threadIdx_y * 4) * COL] = reg_tile[1][1];
      sm_a[threadIdx_x + 16 + (2 + threadIdx_y * 4) * COL] = reg_tile[2][1];
      sm_a[threadIdx_x + 16 + (3 + threadIdx_y * 4) * COL] = reg_tile[3][1];

      sm_a[threadIdx_x + 32 + (0 + threadIdx_y * 4) * COL] = reg_tile[0][2];
      sm_a[threadIdx_x + 32 + (1 + threadIdx_y * 4) * COL] = reg_tile[1][2];
      sm_a[threadIdx_x + 32 + (2 + threadIdx_y * 4) * COL] = reg_tile[2][2];
      sm_a[threadIdx_x + 32 + (3 + threadIdx_y * 4) * COL] = reg_tile[3][2];

      sm_a[threadIdx_x + 48 + (0 + threadIdx_y * 4) * COL] = reg_tile[0][3];
      sm_a[threadIdx_x + 48 + (1 + threadIdx_y * 4) * COL] = reg_tile[1][3];
      sm_a[threadIdx_x + 48 + (2 + threadIdx_y * 4) * COL] = reg_tile[2][3];
      sm_a[threadIdx_x + 48 + (3 + threadIdx_y * 4) * COL] = reg_tile[3][3];
    }

    if(threadIdx_y >= 4 && threadIdx_y < 8) // 4, 5, 6, 7
    {
      sm_b[threadIdx_x      + (0 + (threadIdx_y - 4) * 4) * COL] = reg_tile[0][0];
      sm_b[threadIdx_x      + (1 + (threadIdx_y - 4) * 4) * COL] = reg_tile[1][0];
      sm_b[threadIdx_x      + (2 + (threadIdx_y - 4) * 4) * COL] = reg_tile[2][0];
      sm_b[threadIdx_x      + (3 + (threadIdx_y - 4) * 4) * COL] = reg_tile[3][0];

      sm_b[threadIdx_x + 16 + (0 + (threadIdx_y - 4) * 4) * COL] = reg_tile[0][1];
      sm_b[threadIdx_x + 16 + (1 + (threadIdx_y - 4) * 4) * COL] = reg_tile[1][1];
      sm_b[threadIdx_x + 16 + (2 + (threadIdx_y - 4) * 4) * COL] = reg_tile[2][1];
      sm_b[threadIdx_x + 16 + (3 + (threadIdx_y - 4) * 4) * COL] = reg_tile[3][1];

      sm_b[threadIdx_x + 32 + (0 + (threadIdx_y - 4) * 4) * COL] = reg_tile[0][2];
      sm_b[threadIdx_x + 32 + (1 + (threadIdx_y - 4) * 4) * COL] = reg_tile[1][2];
      sm_b[threadIdx_x + 32 + (2 + (threadIdx_y - 4) * 4) * COL] = reg_tile[2][2];
      sm_b[threadIdx_x + 32 + (3 + (threadIdx_y - 4) * 4) * COL] = reg_tile[3][2];

      sm_b[threadIdx_x + 48 + (0 + (threadIdx_y - 4) * 4) * COL] = reg_tile[0][3];
      sm_b[threadIdx_x + 48 + (1 + (threadIdx_y - 4) * 4) * COL] = reg_tile[1][3];
      sm_b[threadIdx_x + 48 + (2 + (threadIdx_y - 4) * 4) * COL] = reg_tile[2][3];
      sm_b[threadIdx_x + 48 + (3 + (threadIdx_y - 4) * 4) * COL] = reg_tile[3][3];
    }
    sycl::group_barrier(thread_block);

    if(threadIdx_y < 4) // 0, 1, 2, 3
    {
      reg_tile[0][0] = sm_a[threadIdx_x      + (threadIdx_y + 0) * COL];
      reg_tile[1][0] = sm_a[threadIdx_x      + (threadIdx_y + 4) * COL];
      reg_tile[2][0] = sm_a[threadIdx_x      + (threadIdx_y + 8) * COL];
      reg_tile[3][0] = sm_a[threadIdx_x      + (threadIdx_y + 12)* COL];

      reg_tile[0][1] = sm_a[threadIdx_x + 16 + (threadIdx_y + 0) * COL];
      reg_tile[1][1] = sm_a[threadIdx_x + 16 + (threadIdx_y + 4) * COL];
      reg_tile[2][1] = sm_a[threadIdx_x + 16 + (threadIdx_y + 8) * COL];
      reg_tile[3][1] = sm_a[threadIdx_x + 16 + (threadIdx_y + 12)* COL];

      reg_tile[0][2] = sm_a[threadIdx_x + 32 + (threadIdx_y + 0) * COL];
      reg_tile[1][2] = sm_a[threadIdx_x + 32 + (threadIdx_y + 4) * COL];
      reg_tile[2][2] = sm_a[threadIdx_x + 32 + (threadIdx_y + 8) * COL];
      reg_tile[3][2] = sm_a[threadIdx_x + 32 + (threadIdx_y + 12)* COL];

      reg_tile[0][3] = sm_a[threadIdx_x + 48 + (threadIdx_y + 0) * COL];
      reg_tile[1][3] = sm_a[threadIdx_x + 48 + (threadIdx_y + 4) * COL];
      reg_tile[2][3] = sm_a[threadIdx_x + 48 + (threadIdx_y + 8) * COL];
      reg_tile[3][3] = sm_a[threadIdx_x + 48 + (threadIdx_y + 12)* COL];
    }

    if(threadIdx_y >= 4 && threadIdx_y < 8) // 4, 5, 6, 7
    {
      reg_tile[0][0] = sm_b[threadIdx_x      + (threadIdx_y - 4 + 0) * COL];
      reg_tile[1][0] = sm_b[threadIdx_x      + (threadIdx_y - 4 + 4) * COL];
      reg_tile[2][0] = sm_b[threadIdx_x      + (threadIdx_y - 4 + 8) * COL];
      reg_tile[3][0] = sm_b[threadIdx_x      + (threadIdx_y - 4 + 12)* COL];

      reg_tile[0][1] = sm_b[threadIdx_x + 16 + (threadIdx_y - 4 + 0) * COL];
      reg_tile[1][1] = sm_b[threadIdx_x + 16 + (threadIdx_y - 4 + 4) * COL];
      reg_tile[2][1] = sm_b[threadIdx_x + 16 + (threadIdx_y - 4 + 8) * COL];
      reg_tile[3][1] = sm_b[threadIdx_x + 16 + (threadIdx_y - 4 + 12)* COL];

      reg_tile[0][2] = sm_b[threadIdx_x + 32 + (threadIdx_y - 4 + 0) * COL];
      reg_tile[1][2] = sm_b[threadIdx_x + 32 + (threadIdx_y - 4 + 4) * COL];
      reg_tile[2][2] = sm_b[threadIdx_x + 32 + (threadIdx_y - 4 + 8) * COL];
      reg_tile[3][2] = sm_b[threadIdx_x + 32 + (threadIdx_y - 4 + 12)* COL];

      reg_tile[0][3] = sm_b[threadIdx_x + 48 + (threadIdx_y - 4 + 0) * COL];
      reg_tile[1][3] = sm_b[threadIdx_x + 48 + (threadIdx_y - 4 + 4) * COL];
      reg_tile[2][3] = sm_b[threadIdx_x + 48 + (threadIdx_y - 4 + 8) * COL];
      reg_tile[3][3] = sm_b[threadIdx_x + 48 + (threadIdx_y - 4 + 12)* COL];
    }
    sycl::group_barrier(thread_block);

    if(threadIdx_y >= 8 && threadIdx_y < 12) // 8, 9, 10, 11
    {
      sm_a[threadIdx_x      + (0 + (threadIdx_y - 8) * 4) * COL] = reg_tile[0][0];
      sm_a[threadIdx_x      + (1 + (threadIdx_y - 8) * 4) * COL] = reg_tile[1][0];
      sm_a[threadIdx_x      + (2 + (threadIdx_y - 8) * 4) * COL] = reg_tile[2][0];
      sm_a[threadIdx_x      + (3 + (threadIdx_y - 8) * 4) * COL] = reg_tile[3][0];

      sm_a[threadIdx_x + 16 + (0 + (threadIdx_y - 8) * 4) * COL] = reg_tile[0][1];
      sm_a[threadIdx_x + 16 + (1 + (threadIdx_y - 8) * 4) * COL] = reg_tile[1][1];
      sm_a[threadIdx_x + 16 + (2 + (threadIdx_y - 8) * 4) * COL] = reg_tile[2][1];
      sm_a[threadIdx_x + 16 + (3 + (threadIdx_y - 8) * 4) * COL] = reg_tile[3][1];

      sm_a[threadIdx_x + 32 + (0 + (threadIdx_y - 8) * 4) * COL] = reg_tile[0][2];
      sm_a[threadIdx_x + 32 + (1 + (threadIdx_y - 8) * 4) * COL] = reg_tile[1][2];
      sm_a[threadIdx_x + 32 + (2 + (threadIdx_y - 8) * 4) * COL] = reg_tile[2][2];
      sm_a[threadIdx_x + 32 + (3 + (threadIdx_y - 8) * 4) * COL] = reg_tile[3][2];

      sm_a[threadIdx_x + 48 + (0 + (threadIdx_y - 8) * 4) * COL] = reg_tile[0][3];
      sm_a[threadIdx_x + 48 + (1 + (threadIdx_y - 8) * 4) * COL] = reg_tile[1][3];
      sm_a[threadIdx_x + 48 + (2 + (threadIdx_y - 8) * 4) * COL] = reg_tile[2][3];
      sm_a[threadIdx_x + 48 + (3 + (threadIdx_y - 8) * 4) * COL] = reg_tile[3][3];
    }

    if(threadIdx_y >= 12) // 12, 13, 14, 15
    {
      sm_b[threadIdx_x      + (0 + (threadIdx_y - 12) * 4) * COL] = reg_tile[0][0];
      sm_b[threadIdx_x      + (1 + (threadIdx_y - 12) * 4) * COL] = reg_tile[1][0];
      sm_b[threadIdx_x      + (2 + (threadIdx_y - 12) * 4) * COL] = reg_tile[2][0];
      sm_b[threadIdx_x      + (3 + (threadIdx_y - 12) * 4) * COL] = reg_tile[3][0];

      sm_b[threadIdx_x + 16 + (0 + (threadIdx_y - 12) * 4) * COL] = reg_tile[0][1];
      sm_b[threadIdx_x + 16 + (1 + (threadIdx_y - 12) * 4) * COL] = reg_tile[1][1];
      sm_b[threadIdx_x + 16 + (2 + (threadIdx_y - 12) * 4) * COL] = reg_tile[2][1];
      sm_b[threadIdx_x + 16 + (3 + (threadIdx_y - 12) * 4) * COL] = reg_tile[3][1];

      sm_b[threadIdx_x + 32 + (0 + (threadIdx_y - 12) * 4) * COL] = reg_tile[0][2];
      sm_b[threadIdx_x + 32 + (1 + (threadIdx_y - 12) * 4) * COL] = reg_tile[1][2];
      sm_b[threadIdx_x + 32 + (2 + (threadIdx_y - 12) * 4) * COL] = reg_tile[2][2];
      sm_b[threadIdx_x + 32 + (3 + (threadIdx_y - 12) * 4) * COL] = reg_tile[3][2];

      sm_b[threadIdx_x + 48 + (0 + (threadIdx_y - 12) * 4) * COL] = reg_tile[0][3];
      sm_b[threadIdx_x + 48 + (1 + (threadIdx_y - 12) * 4) * COL] = reg_tile[1][3];
      sm_b[threadIdx_x + 48 + (2 + (threadIdx_y - 12) * 4) * COL] = reg_tile[2][3];
      sm_b[threadIdx_x + 48 + (3 + (threadIdx_y - 12) * 4) * COL] = reg_tile[3][3];
    }
    sycl::group_barrier(thread_block);

    if(threadIdx_y >= 8 && threadIdx_y < 12) // 8, 9, 10, 11
    {
      reg_tile[0][0] = sm_a[threadIdx_x      + (threadIdx_y - 8 + 0) * COL];
      reg_tile[1][0] = sm_a[threadIdx_x      + (threadIdx_y - 8 + 4) * COL];
      reg_tile[2][0] = sm_a[threadIdx_x      + (threadIdx_y - 8 + 8) * COL];
      reg_tile[3][0] = sm_a[threadIdx_x      + (threadIdx_y - 8 + 12)* COL];

      reg_tile[0][1] = sm_a[threadIdx_x + 16 + (threadIdx_y - 8 + 0) * COL];
      reg_tile[1][1] = sm_a[threadIdx_x + 16 + (threadIdx_y - 8 + 4) * COL];
      reg_tile[2][1] = sm_a[threadIdx_x + 16 + (threadIdx_y - 8 + 8) * COL];
      reg_tile[3][1] = sm_a[threadIdx_x + 16 + (threadIdx_y - 8 + 12)* COL];

      reg_tile[0][2] = sm_a[threadIdx_x + 32 + (threadIdx_y - 8 + 0) * COL];
      reg_tile[1][2] = sm_a[threadIdx_x + 32 + (threadIdx_y - 8 + 4) * COL];
      reg_tile[2][2] = sm_a[threadIdx_x + 32 + (threadIdx_y - 8 + 8) * COL];
      reg_tile[3][2] = sm_a[threadIdx_x + 32 + (threadIdx_y - 8 + 12)* COL];

      reg_tile[0][3] = sm_a[threadIdx_x + 48 + (threadIdx_y - 8 + 0) * COL];
      reg_tile[1][3] = sm_a[threadIdx_x + 48 + (threadIdx_y - 8 + 4) * COL];
      reg_tile[2][3] = sm_a[threadIdx_x + 48 + (threadIdx_y - 8 + 8) * COL];
      reg_tile[3][3] = sm_a[threadIdx_x + 48 + (threadIdx_y - 8 + 12)* COL];
    }

    if(threadIdx_y >= 12) // 12, 13, 14, 15
    {
      reg_tile[0][0] = sm_b[threadIdx_x      + (threadIdx_y - 12 + 0) * COL];
      reg_tile[1][0] = sm_b[threadIdx_x      + (threadIdx_y - 12 + 4) * COL];
      reg_tile[2][0] = sm_b[threadIdx_x      + (threadIdx_y - 12 + 8) * COL];
      reg_tile[3][0] = sm_b[threadIdx_x      + (threadIdx_y - 12 + 12)* COL];

      reg_tile[0][1] = sm_b[threadIdx_x + 16 + (threadIdx_y - 12 + 0) * COL];
      reg_tile[1][1] = sm_b[threadIdx_x + 16 + (threadIdx_y - 12 + 4) * COL];
      reg_tile[2][1] = sm_b[threadIdx_x + 16 + (threadIdx_y - 12 + 8) * COL];
      reg_tile[3][1] = sm_b[threadIdx_x + 16 + (threadIdx_y - 12 + 12)* COL];

      reg_tile[0][2] = sm_b[threadIdx_x + 32 + (threadIdx_y - 12 + 0) * COL];
      reg_tile[1][2] = sm_b[threadIdx_x + 32 + (threadIdx_y - 12 + 4) * COL];
      reg_tile[2][2] = sm_b[threadIdx_x + 32 + (threadIdx_y - 12 + 8) * COL];
      reg_tile[3][2] = sm_b[threadIdx_x + 32 + (threadIdx_y - 12 + 12)* COL];

      reg_tile[0][3] = sm_b[threadIdx_x + 48 + (threadIdx_y - 12 + 0) * COL];
      reg_tile[1][3] = sm_b[threadIdx_x + 48 + (threadIdx_y - 12 + 4) * COL];
      reg_tile[2][3] = sm_b[threadIdx_x + 48 + (threadIdx_y - 12 + 8) * COL];
      reg_tile[3][3] = sm_b[threadIdx_x + 48 + (threadIdx_y - 12 + 12)* COL];
    }
    sycl::group_barrier(thread_block);
  } // 	End of Register Transpose

  //
  // 	based on "noab"
  //  d1-bottom: sd1_4, 5 , 6 , 7 , 8 and 9.
  //
#pragma unroll 1
  for(int iter_noab = 0; iter_noab < size_noab; iter_noab++) {
    // 	flags
    int flag_d1_4 = const_df_d1_exec[3 + (iter_noab) *NUM_D1_EQUATIONS];
    int flag_d1_5 = const_df_d1_exec[4 + (iter_noab) *NUM_D1_EQUATIONS];
    int flag_d1_6 = const_df_d1_exec[5 + (iter_noab) *NUM_D1_EQUATIONS];
    int flag_d1_7 = const_df_d1_exec[6 + (iter_noab) *NUM_D1_EQUATIONS];
    int flag_d1_8 = const_df_d1_exec[7 + (iter_noab) *NUM_D1_EQUATIONS];
    int flag_d1_9 = const_df_d1_exec[8 + (iter_noab) *NUM_D1_EQUATIONS];

    base_size_h1b = const_df_d1_size[0 + (iter_noab) *NUM_D1_INDEX];
    base_size_h2b = const_df_d1_size[1 + (iter_noab) *NUM_D1_INDEX];
    base_size_h3b = const_df_d1_size[2 + (iter_noab) *NUM_D1_INDEX];
    base_size_h7b = const_df_d1_size[3 + (iter_noab) *NUM_D1_INDEX];
    base_size_p4b = const_df_d1_size[4 + (iter_noab) *NUM_D1_INDEX];
    base_size_p5b = const_df_d1_size[5 + (iter_noab) *NUM_D1_INDEX];
    base_size_p6b = const_df_d1_size[6 + (iter_noab) *NUM_D1_INDEX];

    //
    num_blks_h1b = CEIL(base_size_h1b, FUSION_SIZE_SLICE_1_H1);
    num_blks_h2b = CEIL(base_size_h2b, FUSION_SIZE_SLICE_1_H2);
    num_blks_h3b = CEIL(base_size_h3b, FUSION_SIZE_SLICE_1_H3);
    num_blks_p4b = CEIL(base_size_p4b, FUSION_SIZE_SLICE_1_P4);
    num_blks_p5b = CEIL(base_size_p5b, FUSION_SIZE_SLICE_1_P5);
    num_blks_p6b = CEIL(base_size_p6b, FUSION_SIZE_SLICE_1_P6);

    // 	(2) blk_idx_h/p*b
    blk_idx_p4b =
      blockIdx_x / (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b * num_blks_p5b);
    tmp_blkIdx =
      blockIdx_x % (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b * num_blks_p5b);
    blk_idx_p5b = (tmp_blkIdx) / (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b);
    tmp_blkIdx  = (tmp_blkIdx) % (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b);
    blk_idx_p6b = (tmp_blkIdx) / (num_blks_h3b * num_blks_h2b * num_blks_h1b);
    tmp_blkIdx  = (tmp_blkIdx) % (num_blks_h3b * num_blks_h2b * num_blks_h1b);
    blk_idx_h1b = (tmp_blkIdx) / (num_blks_h3b * num_blks_h2b);
    tmp_blkIdx  = (tmp_blkIdx) % (num_blks_h3b * num_blks_h2b);
    blk_idx_h2b = (tmp_blkIdx) / (num_blks_h3b);
    blk_idx_h3b = blockIdx_x % (num_blks_h3b);

    // 	(3) str_blk_idx_h/p*
    str_blk_idx_h3 = blk_idx_h3b * FUSION_SIZE_SLICE_1_H3;
    str_blk_idx_h2 = blk_idx_h2b * FUSION_SIZE_SLICE_1_H2;
    str_blk_idx_h1 = blk_idx_h1b * FUSION_SIZE_SLICE_1_H1;
    str_blk_idx_p6 = blk_idx_p6b * FUSION_SIZE_SLICE_1_P6;
    str_blk_idx_p5 = blk_idx_p5b * FUSION_SIZE_SLICE_1_P5;
    str_blk_idx_p4 = blk_idx_p4b * FUSION_SIZE_SLICE_1_P4;

    // 	(4) rng_h/p*
    rng_h3 = ((base_size_h3b - str_blk_idx_h3) >= FUSION_SIZE_SLICE_1_H3) ? FUSION_SIZE_SLICE_1_H3 : (base_size_h3b % FUSION_SIZE_SLICE_1_H3);
    rng_h2 = ((base_size_h2b - str_blk_idx_h2) >= FUSION_SIZE_SLICE_1_H2) ? FUSION_SIZE_SLICE_1_H2 : (base_size_h2b % FUSION_SIZE_SLICE_1_H2);
    rng_h1 = ((base_size_h1b - str_blk_idx_h1) >= FUSION_SIZE_SLICE_1_H1) ? FUSION_SIZE_SLICE_1_H1 : (base_size_h1b % FUSION_SIZE_SLICE_1_H1);
    rng_p6 = ((base_size_p6b - str_blk_idx_p6) >= FUSION_SIZE_SLICE_1_P6) ? FUSION_SIZE_SLICE_1_P6 : (base_size_p6b % FUSION_SIZE_SLICE_1_P6);
    rng_p5 = ((base_size_p5b - str_blk_idx_p5) >= FUSION_SIZE_SLICE_1_P5) ? FUSION_SIZE_SLICE_1_P5 : (base_size_p5b % FUSION_SIZE_SLICE_1_P5);
    rng_p4 = ((base_size_p4b - str_blk_idx_p4) >= FUSION_SIZE_SLICE_1_P4) ? FUSION_SIZE_SLICE_1_P4 : (base_size_p4b % FUSION_SIZE_SLICE_1_P4);

    // 	sd1_4
    if(flag_d1_4 >= 0) {
      //
      T* tmp_dev_d1_t2_4 = df_dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_4;
      T* tmp_dev_d1_v2_4 = df_dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_4;

      internal_upperbound = 0;
#pragma unroll 1
      for(int l = 0; l < base_size_h7b; l += FUSION_SIZE_INT_UNIT) {
        // Part: Generalized Contraction Index (p7b)
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - base_size_h7b;
        if(internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of size_internal Indices: 1
        if(idx_p6 < rng_p6 && idx_h1 < rng_h1 &&
           threadIdx_x < FUSION_SIZE_INT_UNIT - internal_upperbound)
          for(short ll = 0; ll < rng_p5; ll++) {
            sm_a[threadIdx_y + ll * FUSION_SIZE_TB_1_Y + (threadIdx_x * COL)] =
              tmp_dev_d1_t2_4[(str_blk_idx_p5 + ll +
                               (str_blk_idx_p6 + idx_p6 +
                                (str_blk_idx_h1 + idx_h1) * base_size_p6b) *
                                 base_size_p5b) *
                                base_size_h7b +
                              (threadIdx_x + l)];
          }

        // Load Input Tensor to Shared Memory
        if(idx_h3 < rng_h3 && idx_h2 < rng_h2 &&
           threadIdx_y < FUSION_SIZE_INT_UNIT - internal_upperbound)
          for(short ll = 0; ll < rng_p4; ll++) {
            sm_b[threadIdx_x + ll * FUSION_SIZE_TB_1_X + (threadIdx_y * COL)] = tmp_dev_d1_v2_4[(
              str_blk_idx_h3 + idx_h3 +
              (str_blk_idx_h2 + idx_h2 +
               (str_blk_idx_p4 + ll + (threadIdx_y + l) * base_size_p4b) * base_size_h2b) *
                base_size_h3b)];
          }
        sycl::group_barrier(thread_block);

        // Cross-Product: -1
        // Part: Generalized Threads
        for(short ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++) {
          temp_bv[0] = sm_b[idx_h3 + idx_h2 * FUSION_SIZE_SLICE_1_H3 + 0  + (ll * COL)];
          temp_bv[1] = sm_b[idx_h3 + idx_h2 * FUSION_SIZE_SLICE_1_H3 + 16 + (ll * COL)];
          temp_bv[2] = sm_b[idx_h3 + idx_h2 * FUSION_SIZE_SLICE_1_H3 + 32 + (ll * COL)];
          temp_bv[3] = sm_b[idx_h3 + idx_h2 * FUSION_SIZE_SLICE_1_H3 + 48 + (ll * COL)];

          for(unsigned short xx = 0; xx < 4; xx++) {
            temp_av = sm_a[idx_p6 + idx_h1 * FUSION_SIZE_SLICE_1_P6 + (xx * 16) + (ll * COL)];

            reg_tile[0][xx] -= temp_av * temp_bv[0];
            reg_tile[1][xx] -= temp_av * temp_bv[1];
            reg_tile[2][xx] -= temp_av * temp_bv[2];
            reg_tile[3][xx] -= temp_av * temp_bv[3];
          }
        }
        sycl::group_barrier(thread_block);
      }
    }

    // 	sd1_5
    if(flag_d1_5 >= 0) {
      //
      T* tmp_dev_d1_t2_5 = df_dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_5;
      T* tmp_dev_d1_v2_5 = df_dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_5;

      internal_upperbound = 0;
#pragma unroll 1
      for(int l = 0; l < base_size_h7b; l += FUSION_SIZE_INT_UNIT) {
        // Part: Generalized Contraction Index (p7b)
        // internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - base_size_h7b;
        if(internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of Internal Indices: 1
        if(idx_p6 < rng_p6 && idx_h1 < rng_h2 &&
           threadIdx_x < FUSION_SIZE_INT_UNIT - internal_upperbound)
          for(short ll = 0; ll < rng_p5; ll++) {
            sm_a[threadIdx_y + ll * FUSION_SIZE_TB_1_Y + (threadIdx_x * COL)] =
              tmp_dev_d1_t2_5[(str_blk_idx_p5 + ll +
                               (str_blk_idx_p6 + idx_p6 +
                                (str_blk_idx_h2 + idx_h1) * base_size_p6b) *
                                 base_size_p5b) *
                                base_size_h7b +
                              (threadIdx_x + l)];
          }

        // Load Input Tensor to Shared Memory
        if(idx_h3 < rng_h3 && idx_h2 < rng_h1 &&
           threadIdx_y < FUSION_SIZE_INT_UNIT - internal_upperbound)
          for(short ll = 0; ll < rng_p4; ll++) {
            sm_b[threadIdx_x + ll * FUSION_SIZE_TB_1_X + (threadIdx_y * COL)] = tmp_dev_d1_v2_5[(
              str_blk_idx_h3 + idx_h3 +
              (str_blk_idx_h1 + idx_h2 +
               (str_blk_idx_p4 + ll + (threadIdx_y + l) * base_size_p4b) * base_size_h1b) *
                base_size_h3b)];
          }
        sycl::group_barrier(thread_block);

        // Cross-Product: -1
        // Part: Generalized Threads
        for(short ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++) {
          temp_bv[0] = sm_b[idx_h3 + idx_h1 * FUSION_SIZE_SLICE_1_H3 + 0  + (ll * COL)];
          temp_bv[1] = sm_b[idx_h3 + idx_h1 * FUSION_SIZE_SLICE_1_H3 + 16 + (ll * COL)];
          temp_bv[2] = sm_b[idx_h3 + idx_h1 * FUSION_SIZE_SLICE_1_H3 + 32 + (ll * COL)];
          temp_bv[3] = sm_b[idx_h3 + idx_h1 * FUSION_SIZE_SLICE_1_H3 + 48 + (ll * COL)];

          for(unsigned short xx = 0; xx < 4; xx++) {
            temp_av = sm_a[idx_p6 + idx_h2 * FUSION_SIZE_SLICE_1_P6 + (xx * 16) + (ll * COL)];

            reg_tile[0][xx] += temp_av * temp_bv[0];
            reg_tile[1][xx] += temp_av * temp_bv[1];
            reg_tile[2][xx] += temp_av * temp_bv[2];
            reg_tile[3][xx] += temp_av * temp_bv[3];
          }
        }
        sycl::group_barrier(thread_block);
      }
    }

    // 	sd1_6
    if(flag_d1_6 >= 0) {
      //
      T* tmp_dev_d1_t2_6 = df_dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_6;
      T* tmp_dev_d1_v2_6 = df_dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_6;

      internal_upperbound = 0;
#pragma unroll 1
      for(int l = 0; l < base_size_h7b; l += FUSION_SIZE_INT_UNIT) {
        // Part: Generalized Contraction Index (p7b)
        // internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - base_size_h7b;
        if(internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of Internal Indices: 1 //63, 21
        if(idx_p6 < rng_p6 && idx_h1 < rng_h3 &&
           threadIdx_x < FUSION_SIZE_INT_UNIT - internal_upperbound)
          for(short ll = 0; ll < rng_p5; ll++) {
            sm_a[threadIdx_y + ll * FUSION_SIZE_TB_1_Y + (threadIdx_x * COL)] =
              tmp_dev_d1_t2_6[(str_blk_idx_p5 + ll +
                               (str_blk_idx_p6 + idx_p6 +
                                (str_blk_idx_h3 + idx_h1) * base_size_p6b) *
                                 base_size_p5b) *
                                base_size_h7b +
                              (threadIdx_x + l)];
          }

        // Load Input Tensor to Shared Memory
        if(idx_h3 < rng_h2 && idx_h2 < rng_h1 &&
           threadIdx_y < FUSION_SIZE_INT_UNIT - internal_upperbound)
          for(short ll = 0; ll < rng_p4; ll++) {
            sm_b[threadIdx_x + ll * FUSION_SIZE_TB_1_X + (threadIdx_y * COL)] = tmp_dev_d1_v2_6[(
              str_blk_idx_h2 + idx_h3 +
              (str_blk_idx_h1 + idx_h2 +
               (str_blk_idx_p4 + ll + (threadIdx_y + l) * base_size_p4b) * base_size_h1b) *
                base_size_h2b)];
          }
        sycl::group_barrier(thread_block);

        // Cross-Product: -1
        // Part: Generalized Threads
        for(short ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++) {
          temp_bv[0] = sm_b[idx_h2 + idx_h1 * FUSION_SIZE_SLICE_1_H2 + 0  + (ll * COL)];
          temp_bv[1] = sm_b[idx_h2 + idx_h1 * FUSION_SIZE_SLICE_1_H2 + 16 + (ll * COL)];
          temp_bv[2] = sm_b[idx_h2 + idx_h1 * FUSION_SIZE_SLICE_1_H2 + 32 + (ll * COL)];
          temp_bv[3] = sm_b[idx_h2 + idx_h1 * FUSION_SIZE_SLICE_1_H2 + 48 + (ll * COL)];

          for(unsigned short xx = 0; xx < 4; xx++) {
            temp_av = sm_a[idx_p6 + idx_h3 * FUSION_SIZE_SLICE_1_P6 + (xx * 16) + (ll * COL)];

            reg_tile[0][xx] -= temp_av * temp_bv[0];
            reg_tile[1][xx] -= temp_av * temp_bv[1];
            reg_tile[2][xx] -= temp_av * temp_bv[2];
            reg_tile[3][xx] -= temp_av * temp_bv[3];
          }
        }
        sycl::group_barrier(thread_block);
      }
    }

    // 	sd1_7
    if(flag_d1_7 >= 0) {
      //
      T* tmp_dev_d1_t2_7 = df_dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_7;
      T* tmp_dev_d1_v2_7 = df_dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_7;

      internal_upperbound = 0;
#pragma unroll 1
      for(int l = 0; l < base_size_h7b; l += FUSION_SIZE_INT_UNIT) {
        // Part: Generalized Contraction Index (p7b)
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - base_size_h7b;
        if(internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of Internal Indices: 1
        if(idx_p6 < rng_p6 && idx_h1 < rng_h1 &&
           threadIdx_x < FUSION_SIZE_INT_UNIT - internal_upperbound)
          for(short ll = 0; ll < rng_p4; ll++) {
            sm_a[threadIdx_y + ll * FUSION_SIZE_TB_1_Y + (threadIdx_x * COL)] =
              tmp_dev_d1_t2_7[(str_blk_idx_p4 + ll +
                               (str_blk_idx_p6 + idx_p6 +
                                (str_blk_idx_h1 + idx_h1) * base_size_p6b) *
                                 base_size_p4b) *
                                base_size_h7b +
                              (threadIdx_x + l)];
          }

        // Load Input Tensor to Shared Memory
        if(idx_h3 < rng_h3 && idx_h2 < rng_h2 &&
           threadIdx_y < FUSION_SIZE_INT_UNIT - internal_upperbound)
          for(short ll = 0; ll < rng_p5; ll++) {
            sm_b[threadIdx_x + ll * FUSION_SIZE_TB_1_X + (threadIdx_y * COL)] = tmp_dev_d1_v2_7[(
              str_blk_idx_h3 + idx_h3 +
              (str_blk_idx_h2 + idx_h2 +
               (str_blk_idx_p5 + ll + (threadIdx_y + l) * base_size_p5b) * base_size_h2b) *
                base_size_h3b)];
          }
        sycl::group_barrier(thread_block);

        // Cross-Product: -1
        // Part: Generalized Threads
        for(short ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++) {
          temp_bv[0] = sm_a[idx_p6 + idx_h1 * FUSION_SIZE_SLICE_1_P6 + 0  + (ll * COL)];
          temp_bv[1] = sm_a[idx_p6 + idx_h1 * FUSION_SIZE_SLICE_1_P6 + 16 + (ll * COL)];
          temp_bv[2] = sm_a[idx_p6 + idx_h1 * FUSION_SIZE_SLICE_1_P6 + 32 + (ll * COL)];
          temp_bv[3] = sm_a[idx_p6 + idx_h1 * FUSION_SIZE_SLICE_1_P6 + 48 + (ll * COL)];

          for(unsigned short xx = 0; xx < 4; xx++) {
            temp_av = sm_b[idx_h3 + idx_h2 * FUSION_SIZE_SLICE_1_H3 + (xx * 16) + (ll * COL)];

            reg_tile[0][xx] += temp_av * temp_bv[0];
            reg_tile[1][xx] += temp_av * temp_bv[1];
            reg_tile[2][xx] += temp_av * temp_bv[2];
            reg_tile[3][xx] += temp_av * temp_bv[3];
          }
        }
        sycl::group_barrier(thread_block);
      }
    }

    // 	sd1_8
    if(flag_d1_8 >= 0) {
      //
      T* tmp_dev_d1_t2_8 = df_dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_8;
      T* tmp_dev_d1_v2_8 = df_dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_8;

      internal_upperbound = 0;
#pragma unroll 1
      for(int l = 0; l < base_size_h7b; l += FUSION_SIZE_INT_UNIT) {
        // Part: Generalized Contraction Index (p7b)
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - base_size_h7b;
        if(internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of Internal Indices: 1
        if(idx_p6 < rng_p6 && idx_h1 < rng_h2 &&
           threadIdx_x < FUSION_SIZE_INT_UNIT - internal_upperbound)
          for(short ll = 0; ll < rng_p4; ll++) {
            sm_a[threadIdx_y + ll * FUSION_SIZE_TB_1_Y + (threadIdx_x * COL)] =
              tmp_dev_d1_t2_8[(str_blk_idx_p4 + ll +
                               (str_blk_idx_p6 + idx_p6 +
                                (str_blk_idx_h2 + idx_h1) * base_size_p6b) *
                                 base_size_p4b) *
                                base_size_h7b +
                              (threadIdx_x + l)];
          }

        // Load Input Tensor to Shared Memory
        if(idx_h3 < rng_h3 && idx_h2 < rng_h1 &&
           threadIdx_y < FUSION_SIZE_INT_UNIT - internal_upperbound)
          for(short ll = 0; ll < rng_p5; ll++) {
            sm_b[threadIdx_x + ll * FUSION_SIZE_TB_1_X + (threadIdx_y * COL)] = tmp_dev_d1_v2_8[(
              str_blk_idx_h3 + idx_h3 +
              (str_blk_idx_h1 + idx_h2 +
               (str_blk_idx_p5 + ll + (threadIdx_y + l) * base_size_p5b) * base_size_h1b) *
                base_size_h3b)];
          }
        sycl::group_barrier(thread_block);

        // Cross-Product: -1
        // Part: Generalized Threads
        for(short ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++) {
          temp_bv[0] = sm_a[idx_p6 + idx_h2 * FUSION_SIZE_SLICE_1_P6 + 0  + (ll * COL)];
          temp_bv[1] = sm_a[idx_p6 + idx_h2 * FUSION_SIZE_SLICE_1_P6 + 16 + (ll * COL)];
          temp_bv[2] = sm_a[idx_p6 + idx_h2 * FUSION_SIZE_SLICE_1_P6 + 32 + (ll * COL)];
          temp_bv[3] = sm_a[idx_p6 + idx_h2 * FUSION_SIZE_SLICE_1_P6 + 48 + (ll * COL)];

          for(unsigned short xx = 0; xx < 4; xx++) {
            temp_av = sm_b[idx_h3 + idx_h1 * FUSION_SIZE_SLICE_1_H3 + (xx * 16) + (ll * COL)];

            reg_tile[0][xx] -= temp_av * temp_bv[0];
            reg_tile[1][xx] -= temp_av * temp_bv[1];
            reg_tile[2][xx] -= temp_av * temp_bv[2];
            reg_tile[3][xx] -= temp_av * temp_bv[3];
          }
        }
        sycl::group_barrier(thread_block);
      }
    }

    // 	sd1_9
    if(flag_d1_9 >= 0) {
      //
      T* tmp_dev_d1_t2_9 = df_dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_9;
      T* tmp_dev_d1_v2_9 = df_dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_9;

      internal_upperbound = 0;
#pragma unroll 1
      for(int l = 0; l < base_size_h7b; l += FUSION_SIZE_INT_UNIT) {
        // Part: Generalized Contraction Index (p7b)
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - base_size_h7b;
        if(internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of Internal Indices: 1
        if(idx_p6 < rng_p6 && idx_h1 < rng_h3 &&
           threadIdx_x < FUSION_SIZE_INT_UNIT - internal_upperbound)
          for(short ll = 0; ll < rng_p4; ll++) {
            sm_a[threadIdx_y + ll * FUSION_SIZE_TB_1_Y + (threadIdx_x * COL)] =
              tmp_dev_d1_t2_9[(str_blk_idx_p4 + ll +
                               (str_blk_idx_p6 + idx_p6 +
                                (str_blk_idx_h3 + idx_h1) * base_size_p6b) *
                                 base_size_p4b) *
                                base_size_h7b +
                              (threadIdx_x + l)];
          }

        // Load Input Tensor to Shared Memory
        if(idx_h3 < rng_h2 && idx_h2 < rng_h1 &&
           threadIdx_y < FUSION_SIZE_INT_UNIT - internal_upperbound)
          for(short ll = 0; ll < rng_p5; ll++) {
            sm_b[threadIdx_x + ll * FUSION_SIZE_TB_1_X + (threadIdx_y * COL)] = tmp_dev_d1_v2_9[(
              str_blk_idx_h2 + idx_h3 +
              (str_blk_idx_h1 + idx_h2 +
               (str_blk_idx_p5 + ll + (threadIdx_y + l) * base_size_p5b) * base_size_h1b) *
                base_size_h2b)];
          }
        sycl::group_barrier(thread_block);

        // Cross-Product: -1
        // Part: Generalized Threads
        for(short ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++) {
          temp_bv[0] = sm_a[idx_p6 + idx_h3 * FUSION_SIZE_SLICE_1_P6 + 0  + (ll * COL)];
          temp_bv[1] = sm_a[idx_p6 + idx_h3 * FUSION_SIZE_SLICE_1_P6 + 16 + (ll * COL)];
          temp_bv[2] = sm_a[idx_p6 + idx_h3 * FUSION_SIZE_SLICE_1_P6 + 32 + (ll * COL)];
          temp_bv[3] = sm_a[idx_p6 + idx_h3 * FUSION_SIZE_SLICE_1_P6 + 48 + (ll * COL)];

          for(unsigned short xx = 0; xx < 4; xx++) {
            temp_av = sm_b[idx_h2 + idx_h1 * FUSION_SIZE_SLICE_1_H2 + (xx * 16) + (ll * COL)];

            reg_tile[0][xx] += temp_av * temp_bv[0];
            reg_tile[1][xx] += temp_av * temp_bv[1];
            reg_tile[2][xx] += temp_av * temp_bv[2];
            reg_tile[3][xx] += temp_av * temp_bv[3];
          }
        }
        sycl::group_barrier(thread_block);
      }
    }
  }

  //  d2-bottom: sd2_1, 2, 3, 4, 5 and 6.
#pragma unroll 1
  for(int iter_nvab = 0; iter_nvab < size_nvab; iter_nvab++) {
    //
    int flag_d2_1 = const_df_d2_exec[0 + (iter_nvab) *NUM_D2_EQUATIONS];
    int flag_d2_2 = const_df_d2_exec[1 + (iter_nvab) *NUM_D2_EQUATIONS];
    int flag_d2_3 = const_df_d2_exec[2 + (iter_nvab) *NUM_D2_EQUATIONS];
    int flag_d2_4 = const_df_d2_exec[3 + (iter_nvab) *NUM_D2_EQUATIONS];
    int flag_d2_5 = const_df_d2_exec[4 + (iter_nvab) *NUM_D2_EQUATIONS];
    int flag_d2_6 = const_df_d2_exec[5 + (iter_nvab) *NUM_D2_EQUATIONS];

    //
    base_size_h1b = const_df_d2_size[0 + (iter_nvab) *NUM_D2_INDEX];
    base_size_h2b = const_df_d2_size[1 + (iter_nvab) *NUM_D2_INDEX];
    base_size_h3b = const_df_d2_size[2 + (iter_nvab) *NUM_D2_INDEX];
    base_size_p4b = const_df_d2_size[3 + (iter_nvab) *NUM_D2_INDEX];
    base_size_p5b = const_df_d2_size[4 + (iter_nvab) *NUM_D2_INDEX];
    base_size_p6b = const_df_d2_size[5 + (iter_nvab) *NUM_D2_INDEX];
    base_size_p7b = const_df_d2_size[6 + (iter_nvab) *NUM_D2_INDEX];

    //
    num_blks_h1b = CEIL(base_size_h1b, FUSION_SIZE_SLICE_1_H1);
    num_blks_h2b = CEIL(base_size_h2b, FUSION_SIZE_SLICE_1_H2);
    num_blks_h3b = CEIL(base_size_h3b, FUSION_SIZE_SLICE_1_H3);
    num_blks_p4b = CEIL(base_size_p4b, FUSION_SIZE_SLICE_1_P4);
    num_blks_p5b = CEIL(base_size_p5b, FUSION_SIZE_SLICE_1_P5);
    num_blks_p6b = CEIL(base_size_p6b, FUSION_SIZE_SLICE_1_P6);

    // 	(2) blk_idx_h/p*b
    blk_idx_p4b =
      blockIdx_x / (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b * num_blks_p5b);
    tmp_blkIdx =
      blockIdx_x % (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b * num_blks_p5b);
    blk_idx_p5b = (tmp_blkIdx) / (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b);
    tmp_blkIdx  = (tmp_blkIdx) % (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b);
    blk_idx_p6b = (tmp_blkIdx) / (num_blks_h3b * num_blks_h2b * num_blks_h1b);
    tmp_blkIdx  = (tmp_blkIdx) % (num_blks_h3b * num_blks_h2b * num_blks_h1b);
    blk_idx_h1b = (tmp_blkIdx) / (num_blks_h3b * num_blks_h2b);
    tmp_blkIdx  = (tmp_blkIdx) % (num_blks_h3b * num_blks_h2b);
    blk_idx_h2b = (tmp_blkIdx) / (num_blks_h3b);
    blk_idx_h3b = blockIdx_x % (num_blks_h3b);

    // 	(3) str_blk_idx_h/p*
    str_blk_idx_h3 = blk_idx_h3b * FUSION_SIZE_SLICE_1_H3;
    str_blk_idx_h2 = blk_idx_h2b * FUSION_SIZE_SLICE_1_H2;
    str_blk_idx_h1 = blk_idx_h1b * FUSION_SIZE_SLICE_1_H1;
    str_blk_idx_p6 = blk_idx_p6b * FUSION_SIZE_SLICE_1_P6;
    str_blk_idx_p5 = blk_idx_p5b * FUSION_SIZE_SLICE_1_P5;
    str_blk_idx_p4 = blk_idx_p4b * FUSION_SIZE_SLICE_1_P4;

    // 	(4) rng_h/p*
    rng_h3 = ((base_size_h3b - str_blk_idx_h3) >= FUSION_SIZE_SLICE_1_H3) ? FUSION_SIZE_SLICE_1_H3 : (base_size_h3b % FUSION_SIZE_SLICE_1_H3);
    rng_h2 = ((base_size_h2b - str_blk_idx_h2) >= FUSION_SIZE_SLICE_1_H2) ? FUSION_SIZE_SLICE_1_H2 : (base_size_h2b % FUSION_SIZE_SLICE_1_H2);
    rng_h1 = ((base_size_h1b - str_blk_idx_h1) >= FUSION_SIZE_SLICE_1_H1) ? FUSION_SIZE_SLICE_1_H1 : (base_size_h1b % FUSION_SIZE_SLICE_1_H1);
    rng_p6 = ((base_size_p6b - str_blk_idx_p6) >= FUSION_SIZE_SLICE_1_P6) ? FUSION_SIZE_SLICE_1_P6 : (base_size_p6b % FUSION_SIZE_SLICE_1_P6);
    rng_p5 = ((base_size_p5b - str_blk_idx_p5) >= FUSION_SIZE_SLICE_1_P5) ? FUSION_SIZE_SLICE_1_P5 : (base_size_p5b % FUSION_SIZE_SLICE_1_P5);
    rng_p4 = ((base_size_p4b - str_blk_idx_p4) >= FUSION_SIZE_SLICE_1_P4) ? FUSION_SIZE_SLICE_1_P4 : (base_size_p4b % FUSION_SIZE_SLICE_1_P4);

    //  sd2_1
    if(flag_d2_1 >= 0) {
      //
      T* tmp_dev_d2_t2_1 = df_dev_d2_t2_all + size_max_dim_d2_t2 * flag_d2_1;
      T* tmp_dev_d2_v2_1 = df_dev_d2_v2_all + size_max_dim_d2_v2 * flag_d2_1;

      internal_upperbound = 0;
#pragma unroll 1
      for(int l = 0; l < base_size_p7b; l += FUSION_SIZE_INT_UNIT) {
        // Part: Generalized Contraction Index (p7b)
        // internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - base_size_p7b;
        if(internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of size_internal Indices: 1
        if(idx_p6 < rng_h1 && idx_h1 < rng_h2 &&
           threadIdx_x < FUSION_SIZE_INT_UNIT - internal_upperbound)
          for(short ll = 0; ll < rng_p4; ll++) {
            sm_a[threadIdx_y + ll * FUSION_SIZE_TB_1_Y + (threadIdx_x * COL)] =
              tmp_dev_d2_t2_1[(str_blk_idx_p4 + ll +
                               (str_blk_idx_h1 + idx_p6 +
                                (str_blk_idx_h2 + idx_h1) * base_size_h1b) *
                                 base_size_p4b) *
                                base_size_p7b +
                              (threadIdx_x + l)];
          }

        // Load Input Tensor to Shared Memory
        if(idx_p6 < rng_h3 && idx_h1 < rng_p6 &&
           threadIdx_x < FUSION_SIZE_INT_UNIT - internal_upperbound)
          for(short ll = 0; ll < rng_p5; ll++) {
            sm_b[threadIdx_y + ll * FUSION_SIZE_TB_1_Y + (threadIdx_x * COL)] =
              tmp_dev_d2_v2_1[(str_blk_idx_h3 + idx_p6 +
                               (str_blk_idx_p6 + idx_h1 + (str_blk_idx_p5 + ll) * base_size_p6b) *
                                 base_size_h3b) *
                                base_size_p7b +
                              (threadIdx_x + l)];
          }
        sycl::group_barrier(thread_block);

        // Cross-Product: 16
        // Part: Generalized Threads
        for(short ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++) {
          temp_bv[0] = sm_a[idx_h1 + idx_h2 * FUSION_SIZE_SLICE_1_H1 + 0  + (ll * COL)];
          temp_bv[1] = sm_a[idx_h1 + idx_h2 * FUSION_SIZE_SLICE_1_H1 + 16 + (ll * COL)];
          temp_bv[2] = sm_a[idx_h1 + idx_h2 * FUSION_SIZE_SLICE_1_H1 + 32 + (ll * COL)];
          temp_bv[3] = sm_a[idx_h1 + idx_h2 * FUSION_SIZE_SLICE_1_H1 + 48 + (ll * COL)];

          for(unsigned short xx = 0; xx < 4; xx++) {
            temp_av = sm_b[idx_h3 + idx_p6 * FUSION_SIZE_SLICE_1_H3 + (xx * 16) + (ll * COL)];

            reg_tile[0][xx] -= temp_av * temp_bv[0];
            reg_tile[1][xx] -= temp_av * temp_bv[1];
            reg_tile[2][xx] -= temp_av * temp_bv[2];
            reg_tile[3][xx] -= temp_av * temp_bv[3];
          }
        }
        sycl::group_barrier(thread_block);
      }
    }

    // 	sd2_2
    if(flag_d2_2 >= 0) {
      //
      T* tmp_dev_d2_t2_2 = df_dev_d2_t2_all + size_max_dim_d2_t2 * flag_d2_2;
      T* tmp_dev_d2_v2_2 = df_dev_d2_v2_all + size_max_dim_d2_v2 * flag_d2_2;

      internal_upperbound = 0;
#pragma unroll 1
      for(int l = 0; l < base_size_p7b; l += FUSION_SIZE_INT_UNIT) {
        // Part: Generalized Contraction Index (p7b)
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - base_size_p7b;
        if(internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of size_internal Indices: 1
        if(idx_p6 < rng_h2 && idx_h1 < rng_h3 &&
           threadIdx_x < FUSION_SIZE_INT_UNIT - internal_upperbound)
          for(short ll = 0; ll < rng_p4; ll++) {
            sm_a[threadIdx_y + ll * FUSION_SIZE_TB_1_Y + (threadIdx_x * COL)] =
              tmp_dev_d2_t2_2[(str_blk_idx_p4 + ll +
                               (str_blk_idx_h2 + idx_p6 +
                                (str_blk_idx_h3 + idx_h1) * base_size_h2b) *
                                 base_size_p4b) *
                                base_size_p7b +
                              (threadIdx_x + l)];
          }

        // Load Input Tensor to Shared Memory
        if(idx_p6 < rng_h1 && idx_h1 < rng_p6 &&
           threadIdx_x < FUSION_SIZE_INT_UNIT - internal_upperbound)
          for(short ll = 0; ll < rng_p5; ll++) {
            sm_b[threadIdx_y + ll * FUSION_SIZE_TB_1_Y + (threadIdx_x * COL)] =
              tmp_dev_d2_v2_2[(str_blk_idx_h1 + idx_p6 +
                               (str_blk_idx_p6 + idx_h1 + (str_blk_idx_p5 + ll) * base_size_p6b) *
                                 base_size_h1b) *
                                base_size_p7b +
                              (threadIdx_x + l)];
          }
        sycl::group_barrier(thread_block);

        // Cross-Product: 16
        // Part: Generalized Threads
        for(short ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++) {
          temp_bv[0] = sm_a[idx_h2 + idx_h3 * FUSION_SIZE_SLICE_1_H2 + 0  + (ll * COL)];
          temp_bv[1] = sm_a[idx_h2 + idx_h3 * FUSION_SIZE_SLICE_1_H2 + 16 + (ll * COL)];
          temp_bv[2] = sm_a[idx_h2 + idx_h3 * FUSION_SIZE_SLICE_1_H2 + 32 + (ll * COL)];
          temp_bv[3] = sm_a[idx_h2 + idx_h3 * FUSION_SIZE_SLICE_1_H2 + 48 + (ll * COL)];

          for(unsigned short xx = 0; xx < 4; xx++) {
            temp_av = sm_b[idx_h1 + idx_p6 * FUSION_SIZE_SLICE_1_H1 + (xx * 16) + (ll * COL)];

            reg_tile[0][xx] -= temp_av * temp_bv[0];
            reg_tile[1][xx] -= temp_av * temp_bv[1];
            reg_tile[2][xx] -= temp_av * temp_bv[2];
            reg_tile[3][xx] -= temp_av * temp_bv[3];
          }
        }
        sycl::group_barrier(thread_block);
      }
    }

    // 	sd2_3
    if(flag_d2_3 >= 0) {
      //
      T* tmp_dev_d2_t2_3 = df_dev_d2_t2_all + size_max_dim_d2_t2 * flag_d2_3;
      T* tmp_dev_d2_v2_3 = df_dev_d2_v2_all + size_max_dim_d2_v2 * flag_d2_3;

      internal_upperbound = 0;
#pragma unroll 1
      for(int l = 0; l < base_size_p7b; l += FUSION_SIZE_INT_UNIT) {
        // Part: Generalized Contraction Index (p7b)
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - base_size_p7b;
        if(internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of size_internal Indices: 1
        if(idx_p6 < rng_h1 && idx_h1 < rng_h3 &&
           threadIdx_x < FUSION_SIZE_INT_UNIT - internal_upperbound)
          for(short ll = 0; ll < rng_p4; ll++) {
            sm_a[threadIdx_y + ll * FUSION_SIZE_TB_1_Y + (threadIdx_x * COL)] =
              tmp_dev_d2_t2_3[(str_blk_idx_p4 + ll +
                               (str_blk_idx_h1 + idx_p6 +
                                (str_blk_idx_h3 + idx_h1) * base_size_h1b) *
                                 base_size_p4b) *
                                base_size_p7b +
                              (threadIdx_x + l)];
          }

        // Load Input Tensor to Shared Memory
        if(idx_p6 < rng_h2 && idx_h1 < rng_p6 &&
           threadIdx_x < FUSION_SIZE_INT_UNIT - internal_upperbound)
          for(short ll = 0; ll < rng_p5; ll++) {
            sm_b[threadIdx_y + ll * FUSION_SIZE_TB_1_Y + (threadIdx_x * COL)] =
              tmp_dev_d2_v2_3[(str_blk_idx_h2 + idx_p6 +
                               (str_blk_idx_p6 + idx_h1 + (str_blk_idx_p5 + ll) * base_size_p6b) *
                                 base_size_h2b) *
                                base_size_p7b +
                              (threadIdx_x + l)];
          }
        sycl::group_barrier(thread_block);

        // Cross-Product: 16
        // Part: Generalized Threads
        for(short ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++) {
          temp_bv[0] = sm_a[idx_h1 + idx_h3 * FUSION_SIZE_SLICE_1_H1 + 0  + (ll * COL)];
          temp_bv[1] = sm_a[idx_h1 + idx_h3 * FUSION_SIZE_SLICE_1_H1 + 16 + (ll * COL)];
          temp_bv[2] = sm_a[idx_h1 + idx_h3 * FUSION_SIZE_SLICE_1_H1 + 32 + (ll * COL)];
          temp_bv[3] = sm_a[idx_h1 + idx_h3 * FUSION_SIZE_SLICE_1_H1 + 48 + (ll * COL)];

          for(unsigned short xx = 0; xx < 4; xx++) {
            temp_av = sm_b[idx_h2 + idx_p6 * FUSION_SIZE_SLICE_1_H2 + (xx * 16) + (ll * COL)];

            reg_tile[0][xx] += temp_av * temp_bv[0];
            reg_tile[1][xx] += temp_av * temp_bv[1];
            reg_tile[2][xx] += temp_av * temp_bv[2];
            reg_tile[3][xx] += temp_av * temp_bv[3];
          }
        }
        sycl::group_barrier(thread_block);
      }
    }

    // 	sd2_4
    if(flag_d2_4 >= 0) {
      //
      T* tmp_dev_d2_t2_4 = df_dev_d2_t2_all + size_max_dim_d2_t2 * flag_d2_4;
      T* tmp_dev_d2_v2_4 = df_dev_d2_v2_all + size_max_dim_d2_v2 * flag_d2_4;

      internal_upperbound = 0;
#pragma unroll 1
      for(int l = 0; l < base_size_p7b; l += FUSION_SIZE_INT_UNIT) {
        // Part: Generalized Contraction Index (p7b)
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - base_size_p7b;
        if(internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of size_internal Indices: 1
        if(idx_p6 < rng_h1 && idx_h1 < rng_h2 &&
           threadIdx_x < FUSION_SIZE_INT_UNIT - internal_upperbound)
          for(short ll = 0; ll < rng_p5; ll++) {
            sm_a[threadIdx_y + ll * FUSION_SIZE_TB_1_Y + (threadIdx_x * COL)] =
              tmp_dev_d2_t2_4[(str_blk_idx_p5 + ll +
                               (str_blk_idx_h1 + idx_p6 +
                                (str_blk_idx_h2 + idx_h1) * base_size_h1b) *
                                 base_size_p5b) *
                                base_size_p7b +
                              (threadIdx_x + l)];
          }

        // Load Input Tensor to Shared Memory
        if(idx_p6 < rng_h3 && idx_h1 < rng_p6 &&
           threadIdx_x < FUSION_SIZE_INT_UNIT - internal_upperbound)
          for(short ll = 0; ll < rng_p4; ll++) {
            sm_b[threadIdx_y + ll * FUSION_SIZE_TB_1_Y + (threadIdx_x * COL)] =
              tmp_dev_d2_v2_4[(str_blk_idx_h3 + idx_p6 +
                               (str_blk_idx_p6 + idx_h1 + (str_blk_idx_p4 + ll) * base_size_p6b) *
                                 base_size_h3b) *
                                base_size_p7b +
                              (threadIdx_x + l)];
          }
        sycl::group_barrier(thread_block);

        // Cross-Product: 16
        // Part: Generalized Threads
        for(short ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++) {
          temp_bv[0] = sm_b[idx_h3 + idx_p6 * FUSION_SIZE_SLICE_1_H1 + 0  + (ll * COL)];
          temp_bv[1] = sm_b[idx_h3 + idx_p6 * FUSION_SIZE_SLICE_1_H1 + 16 + (ll * COL)];
          temp_bv[2] = sm_b[idx_h3 + idx_p6 * FUSION_SIZE_SLICE_1_H1 + 32 + (ll * COL)];
          temp_bv[3] = sm_b[idx_h3 + idx_p6 * FUSION_SIZE_SLICE_1_H1 + 48 + (ll * COL)];

          for(unsigned short xx = 0; xx < 4; xx++) {
            temp_av = sm_a[idx_h1 + idx_h2 * FUSION_SIZE_SLICE_1_H1 + (xx * 16) + (ll * COL)];

            reg_tile[0][xx] += temp_av * temp_bv[0];
            reg_tile[1][xx] += temp_av * temp_bv[1];
            reg_tile[2][xx] += temp_av * temp_bv[2];
            reg_tile[3][xx] += temp_av * temp_bv[3];
          }
        }
        sycl::group_barrier(thread_block);
      }
    }

    // 	sd2_5
    if(flag_d2_5 >= 0) {
      //
      T* tmp_dev_d2_t2_5 = df_dev_d2_t2_all + size_max_dim_d2_t2 * flag_d2_5;
      T* tmp_dev_d2_v2_5 = df_dev_d2_v2_all + size_max_dim_d2_v2 * flag_d2_5;

      internal_upperbound = 0;
#pragma unroll 1
      for(int l = 0; l < base_size_p7b; l += FUSION_SIZE_INT_UNIT) {
        // Part: Generalized Contraction Index (p7b)
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - base_size_p7b;
        if(internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of size_internal Indices: 1
        if(idx_p6 < rng_h2 && idx_h1 < rng_h3 &&
           threadIdx_x < FUSION_SIZE_INT_UNIT - internal_upperbound)
          for(short ll = 0; ll < rng_p5; ll++) {
            sm_a[threadIdx_y + ll * FUSION_SIZE_TB_1_Y + (threadIdx_x * COL)] =
              tmp_dev_d2_t2_5[(str_blk_idx_p5 + ll +
                               (str_blk_idx_h2 + idx_p6 +
                                (str_blk_idx_h3 + idx_h1) * base_size_h2b) *
                                 base_size_p5b) *
                                base_size_p7b +
                              (threadIdx_x + l)];
          }

        // Load Input Tensor to Shared Memory
        if(idx_p6 < rng_h1 && idx_h1 < rng_p6 &&
           threadIdx_x < FUSION_SIZE_INT_UNIT - internal_upperbound)
          for(short ll = 0; ll < rng_p4; ll++) {
            sm_b[threadIdx_y + ll * FUSION_SIZE_TB_1_Y + (threadIdx_x * COL)] =
              tmp_dev_d2_v2_5[(str_blk_idx_h1 + idx_p6 +
                               (str_blk_idx_p6 + idx_h1 + (str_blk_idx_p4 + ll) * base_size_p6b) *
                                 base_size_h1b) *
                                base_size_p7b +
                              (threadIdx_x + l)];
          }
        sycl::group_barrier(thread_block);

        // Cross-Product: 16
        // Part: Generalized Threads
        for(short ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++) {
          temp_bv[0] = sm_b[idx_h1 + idx_p6 * FUSION_SIZE_SLICE_1_H1 + 0  + (ll * COL)];
          temp_bv[1] = sm_b[idx_h1 + idx_p6 * FUSION_SIZE_SLICE_1_H1 + 16 + (ll * COL)];
          temp_bv[2] = sm_b[idx_h1 + idx_p6 * FUSION_SIZE_SLICE_1_H1 + 32 + (ll * COL)];
          temp_bv[3] = sm_b[idx_h1 + idx_p6 * FUSION_SIZE_SLICE_1_H1 + 48 + (ll * COL)];

          for(unsigned short xx = 0; xx < 4; xx++) {
            temp_av = sm_a[idx_h2 + idx_h3 * FUSION_SIZE_SLICE_1_H2 + (xx * 16) + (ll * COL)];

            reg_tile[0][xx] += temp_av * temp_bv[0];
            reg_tile[1][xx] += temp_av * temp_bv[1];
            reg_tile[2][xx] += temp_av * temp_bv[2];
            reg_tile[3][xx] += temp_av * temp_bv[3];
          }
        }
        sycl::group_barrier(thread_block);
      }
    }

    // 	sd2_6
    if(flag_d2_6 >= 0) {
      //
      T* tmp_dev_d2_t2_6 = df_dev_d2_t2_all + size_max_dim_d2_t2 * flag_d2_6;
      T* tmp_dev_d2_v2_6 = df_dev_d2_v2_all + size_max_dim_d2_v2 * flag_d2_6;

      internal_upperbound = 0;
#pragma unroll 1
      for(int l = 0; l < base_size_p7b; l += FUSION_SIZE_INT_UNIT) {
        // Part: Generalized Contraction Index (p7b)
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - base_size_p7b;
        if(internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of size_internal Indices: 1
        if(idx_p6 < rng_h1 && idx_h1 < rng_h3 &&
           threadIdx_x < FUSION_SIZE_INT_UNIT - internal_upperbound)
          for(short ll = 0; ll < rng_p5; ll++) {
            sm_a[threadIdx_y + ll * FUSION_SIZE_TB_1_Y + (threadIdx_x * COL)] =
              tmp_dev_d2_t2_6[(blk_idx_p5b * FUSION_SIZE_SLICE_1_P6 + ll +
                               (str_blk_idx_h1 + idx_p6 +
                                (str_blk_idx_h3 + idx_h1) * base_size_h1b) *
                                 base_size_p5b) *
                                base_size_p7b +
                              (threadIdx_x + l)];
          }

        // Load Input Tensor to Shared Memory
        if(idx_p6 < rng_h2 && idx_h1 < rng_p6 &&
           threadIdx_x < FUSION_SIZE_INT_UNIT - internal_upperbound)
          for(short ll = 0; ll < rng_p4; ll++) {
            sm_b[threadIdx_y + ll * FUSION_SIZE_TB_1_Y + (threadIdx_x * COL)] =
              tmp_dev_d2_v2_6[(str_blk_idx_h2 + idx_p6 +
                               (str_blk_idx_p6 + idx_h1 + (str_blk_idx_p4 + ll) * base_size_p6b) *
                                 base_size_h2b) *
                                base_size_p7b +
                              (threadIdx_x + l)];
          }
        sycl::group_barrier(thread_block);

        // Cross-Product: 16
        // Part: Generalized Threads
        for(short ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++) {
          temp_bv[0] = sm_b[idx_h2 + idx_p6 * FUSION_SIZE_SLICE_1_H2 + 0  + (ll * COL)];
          temp_bv[1] = sm_b[idx_h2 + idx_p6 * FUSION_SIZE_SLICE_1_H2 + 16 + (ll * COL)];
          temp_bv[2] = sm_b[idx_h2 + idx_p6 * FUSION_SIZE_SLICE_1_H2 + 32 + (ll * COL)];
          temp_bv[3] = sm_b[idx_h2 + idx_p6 * FUSION_SIZE_SLICE_1_H2 + 48 + (ll * COL)];

          for(unsigned short xx = 0; xx < 4; xx++) {
            temp_av = sm_a[idx_h1 + idx_h3 * FUSION_SIZE_SLICE_1_H1 + (xx * 16) + (ll * COL)];

            reg_tile[0][xx] -= temp_av * temp_bv[0];
            reg_tile[1][xx] -= temp_av * temp_bv[1];
            reg_tile[2][xx] -= temp_av * temp_bv[2];
            reg_tile[3][xx] -= temp_av * temp_bv[3];
          }
        }
        sycl::group_barrier(thread_block);
      }
    }
  }

  //
  // 	>>> s1 <<<
  // 		- if
  //
  //  singles (s1)
  {
    base_size_h1b = const_df_s1_size[0];
    base_size_h2b = const_df_s1_size[1];
    base_size_h3b = const_df_s1_size[2];
    base_size_p4b = const_df_s1_size[3];
    base_size_p5b = const_df_s1_size[4];
    base_size_p6b = const_df_s1_size[5];

    num_blks_h1b = CEIL(base_size_h1b, FUSION_SIZE_SLICE_1_H1);
    num_blks_h2b = CEIL(base_size_h2b, FUSION_SIZE_SLICE_1_H2);
    num_blks_h3b = CEIL(base_size_h3b, FUSION_SIZE_SLICE_1_H3);
    num_blks_p4b = CEIL(base_size_p4b, FUSION_SIZE_SLICE_1_P4);
    num_blks_p5b = CEIL(base_size_p5b, FUSION_SIZE_SLICE_1_P5);
    num_blks_p6b = CEIL(base_size_p6b, FUSION_SIZE_SLICE_1_P6);

    // 	(2) blk_idx_h/p*b
    blk_idx_p4b =
      blockIdx_x / (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b * num_blks_p5b);
    tmp_blkIdx =
      blockIdx_x % (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b * num_blks_p5b);
    blk_idx_p5b = (tmp_blkIdx) / (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b);
    tmp_blkIdx  = (tmp_blkIdx) % (num_blks_h3b * num_blks_h2b * num_blks_h1b * num_blks_p6b);
    blk_idx_p6b = (tmp_blkIdx) / (num_blks_h3b * num_blks_h2b * num_blks_h1b);
    tmp_blkIdx  = (tmp_blkIdx) % (num_blks_h3b * num_blks_h2b * num_blks_h1b);
    blk_idx_h1b = (tmp_blkIdx) / (num_blks_h3b * num_blks_h2b);
    tmp_blkIdx  = (tmp_blkIdx) % (num_blks_h3b * num_blks_h2b);
    blk_idx_h2b = (tmp_blkIdx) / (num_blks_h3b);
    blk_idx_h3b = blockIdx_x % (num_blks_h3b);

    // 	(3) str_blk_idx_h/p*
    str_blk_idx_h3 = blk_idx_h3b * FUSION_SIZE_SLICE_1_H3;
    str_blk_idx_h2 = blk_idx_h2b * FUSION_SIZE_SLICE_1_H2;
    str_blk_idx_h1 = blk_idx_h1b * FUSION_SIZE_SLICE_1_H1;
    str_blk_idx_p6 = blk_idx_p6b * FUSION_SIZE_SLICE_1_P6;
    str_blk_idx_p5 = blk_idx_p5b * FUSION_SIZE_SLICE_1_P5;
    str_blk_idx_p4 = blk_idx_p4b * FUSION_SIZE_SLICE_1_P4;

    // 	(4) rng_h/p*
    rng_h3 = ((base_size_h3b - str_blk_idx_h3) >= FUSION_SIZE_SLICE_1_H3) ? FUSION_SIZE_SLICE_1_H3 : (base_size_h3b % FUSION_SIZE_SLICE_1_H3);
    rng_h2 = ((base_size_h2b - str_blk_idx_h2) >= FUSION_SIZE_SLICE_1_H2) ? FUSION_SIZE_SLICE_1_H2 : (base_size_h2b % FUSION_SIZE_SLICE_1_H2);
    rng_h1 = ((base_size_h1b - str_blk_idx_h1) >= FUSION_SIZE_SLICE_1_H1) ? FUSION_SIZE_SLICE_1_H1 : (base_size_h1b % FUSION_SIZE_SLICE_1_H1);
    rng_p6 = ((base_size_p6b - str_blk_idx_p6) >= FUSION_SIZE_SLICE_1_P6) ? FUSION_SIZE_SLICE_1_P6 : (base_size_p6b % FUSION_SIZE_SLICE_1_P6);
    rng_p5 = ((base_size_p5b - str_blk_idx_p5) >= FUSION_SIZE_SLICE_1_P5) ? FUSION_SIZE_SLICE_1_P5 : (base_size_p5b % FUSION_SIZE_SLICE_1_P5);
    rng_p4 = ((base_size_p4b - str_blk_idx_p4) >= FUSION_SIZE_SLICE_1_P4) ? FUSION_SIZE_SLICE_1_P4 : (base_size_p4b % FUSION_SIZE_SLICE_1_P4);

    // 	flags
    int flag_s1_1 = const_df_s1_exec[0];
    int flag_s1_2 = const_df_s1_exec[1];
    int flag_s1_3 = const_df_s1_exec[2];
    int flag_s1_4 = const_df_s1_exec[3];
    int flag_s1_5 = const_df_s1_exec[4];
    int flag_s1_6 = const_df_s1_exec[5];
    int flag_s1_7 = const_df_s1_exec[6];
    int flag_s1_8 = const_df_s1_exec[7];
    int flag_s1_9 = const_df_s1_exec[8];

    //                                        "x"         "x"
    //  >> s1_1:   t3[h3,h2,h1,p6,p5,p4] -= t2[p4,h1] * v2[h3,h2,p6,p5]
    //
    if(flag_s1_1 >= 0) // these if-conditions make 100 ms..
    {
      //
      T* tmp_dev_s1_t1_1 = df_dev_s1_t1_all + size_max_dim_s1_t1 * flag_s1_1;
      T* tmp_dev_s1_v2_1 = df_dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_1;

      if(idx_h3 < rng_p4 && idx_h2 < rng_h1 && idx_p6 == 0 && idx_h1 == 0)
        sm_a[idx_h3 + idx_h2 * FUSION_SIZE_SLICE_1_P4] =
          tmp_dev_s1_t1_1[str_blk_idx_p4 + idx_h3 + (str_blk_idx_h1 + idx_h2) * base_size_p4b];

      if(idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6 && idx_h1 < rng_p5)
        sm_b[idx_h3 + (idx_h2 + idx_p6 * 4) * 4 + (idx_h1 * COL)] =
          tmp_dev_s1_v2_1[blk_idx_h3b * 4 + idx_h3 +
                          (blk_idx_h2b * 4 + idx_h2 +
                           (blk_idx_p6b * 4 + idx_p6 + (blk_idx_p5b * 4 + idx_h1) * base_size_p6b) *
                             base_size_h2b) *
                            base_size_h3b];
      sycl::group_barrier(thread_block);

      //  "p4"
      temp_av = sm_a[0 + idx_h1 * 4];

      //  "p5"
      temp_bv[0] = sm_b[idx_h3 + (idx_h2 + idx_p6 * 4) * 4 + (0 * COL)];
      temp_bv[1] = sm_b[idx_h3 + (idx_h2 + idx_p6 * 4) * 4 + (1 * COL)];
      temp_bv[2] = sm_b[idx_h3 + (idx_h2 + idx_p6 * 4) * 4 + (2 * COL)];
      temp_bv[3] = sm_b[idx_h3 + (idx_h2 + idx_p6 * 4) * 4 + (3 * COL)];

      //  "p4 x p5"
      reg_singles[0][0] += temp_av * temp_bv[0];
      reg_singles[0][1] += temp_av * temp_bv[1];
      reg_singles[0][2] += temp_av * temp_bv[2];
      reg_singles[0][3] += temp_av * temp_bv[3];

      temp_av = sm_a[1 + idx_h1 * 4];

      reg_singles[1][0] += temp_av * temp_bv[0];
      reg_singles[1][1] += temp_av * temp_bv[1];
      reg_singles[1][2] += temp_av * temp_bv[2];
      reg_singles[1][3] += temp_av * temp_bv[3];

      temp_av = sm_a[2 + idx_h1 * 4];

      reg_singles[2][0] += temp_av * temp_bv[0];
      reg_singles[2][1] += temp_av * temp_bv[1];
      reg_singles[2][2] += temp_av * temp_bv[2];
      reg_singles[2][3] += temp_av * temp_bv[3];

      temp_av = sm_a[3 + idx_h1 * 4];

      reg_singles[3][0] += temp_av * temp_bv[0];
      reg_singles[3][1] += temp_av * temp_bv[1];
      reg_singles[3][2] += temp_av * temp_bv[2];
      reg_singles[3][3] += temp_av * temp_bv[3];
      sycl::group_barrier(thread_block);
    }

    //                                        "x1,x2"     "x1,x2,x3,y1"
    //  >> s1_2:   t3[h3,h2,h1,p6,p5,p4] -= t2[p4,h2] * v2[h3,h1,p6,p5] (h3,h2,p6), (h1)
    //
    if(flag_s1_2 >= 0) // these if-conditions make 100 ms..
    {
      //
      T* tmp_dev_s1_t1_2 = df_dev_s1_t1_all + size_max_dim_s1_t1 * flag_s1_2;
      T* tmp_dev_s1_v2_2 = df_dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_2;

      if(idx_h3 < rng_p4 && idx_h2 < rng_h2 && idx_p6 == 0 && idx_h1 == 0) {
        sm_a[idx_h3 + idx_h2 * FUSION_SIZE_SLICE_1_P4] =
          tmp_dev_s1_t1_2[str_blk_idx_p4 + idx_h3 + (str_blk_idx_h2 + idx_h2) * base_size_p4b];
      }

      if(idx_h3 < rng_h3 && idx_h2 < rng_h1 && idx_p6 < rng_p6 && idx_h1 < rng_p5) {
        sm_b[idx_h3 + (idx_h2 + idx_p6 * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H3 + (idx_h1 * COL)] =
          tmp_dev_s1_v2_2[str_blk_idx_h3 + idx_h3 +
                          (str_blk_idx_h1 + idx_h2 +
                           (str_blk_idx_p6 + idx_p6 + (str_blk_idx_p5 + idx_h1) * base_size_p6b) *
                             base_size_h1b) *
                            base_size_h3b];
      }
      sycl::group_barrier(thread_block);

      //  "p4"
      temp_av = sm_a[0 + idx_h2 * 4];

      //  "p5"
      temp_bv[0] = sm_b[idx_h3 + (idx_h1 + idx_p6 * 4) * 4 + (0 * COL)];
      temp_bv[1] = sm_b[idx_h3 + (idx_h1 + idx_p6 * 4) * 4 + (1 * COL)];
      temp_bv[2] = sm_b[idx_h3 + (idx_h1 + idx_p6 * 4) * 4 + (2 * COL)];
      temp_bv[3] = sm_b[idx_h3 + (idx_h1 + idx_p6 * 4) * 4 + (3 * COL)];

      //  "p4 x p5"
      reg_singles[0][0] -= temp_av * temp_bv[0];
      reg_singles[0][1] -= temp_av * temp_bv[1];
      reg_singles[0][2] -= temp_av * temp_bv[2];
      reg_singles[0][3] -= temp_av * temp_bv[3];

      temp_av = sm_a[1 + idx_h2 * 4];

      reg_singles[1][0] -= temp_av * temp_bv[0];
      reg_singles[1][1] -= temp_av * temp_bv[1];
      reg_singles[1][2] -= temp_av * temp_bv[2];
      reg_singles[1][3] -= temp_av * temp_bv[3];

      temp_av = sm_a[2 + idx_h2 * 4];

      reg_singles[2][0] -= temp_av * temp_bv[0];
      reg_singles[2][1] -= temp_av * temp_bv[1];
      reg_singles[2][2] -= temp_av * temp_bv[2];
      reg_singles[2][3] -= temp_av * temp_bv[3];

      temp_av = sm_a[3 + idx_h2 * 4];

      reg_singles[3][0] -= temp_av * temp_bv[0];
      reg_singles[3][1] -= temp_av * temp_bv[1];
      reg_singles[3][2] -= temp_av * temp_bv[2];
      reg_singles[3][3] -= temp_av * temp_bv[3];
      sycl::group_barrier(thread_block);
    }

    //
    //  >> s1_3:   t3[h3,h2,h1,p6,p5,p4] -= t2[p4,h1] * v2[h3,h2,p6,p5] >> t3[h3,h2,h1,p6,p5,p4] +=
    //  t2[p4,h3] * v2[h2,h1,p6,p5]
    //
    if(flag_s1_3 >= 0) // these if-conditions make 100 ms..
    {
      //
      T* tmp_dev_s1_t1_3 = df_dev_s1_t1_all + size_max_dim_s1_t1 * flag_s1_3;
      T* tmp_dev_s1_v2_3 = df_dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_3;

      if(idx_h3 < rng_p4 && idx_h2 < rng_h3 && idx_p6 == 0 && idx_h1 == 0) {
        sm_a[idx_h3 + idx_h2 * FUSION_SIZE_SLICE_1_P4] =
          tmp_dev_s1_t1_3[str_blk_idx_p4 + idx_h3 + (str_blk_idx_h3 + idx_h2) * base_size_p4b];
      }

      if(idx_h3 < rng_h2 && idx_h2 < rng_h1 && idx_p6 < rng_p6 && idx_h1 < rng_p5) {
        sm_b[idx_h3 + (idx_h2 + idx_p6 * 4) * 4 + (idx_h1 * COL)] =
          tmp_dev_s1_v2_3[blk_idx_h2b * 4 + idx_h3 +
                          (blk_idx_h1b * 4 + idx_h2 +
                           (blk_idx_p6b * 4 + idx_p6 + (blk_idx_p5b * 4 + idx_h1) * base_size_p6b) *
                             base_size_h1b) *
                            base_size_h2b];
      }
      sycl::group_barrier(thread_block);

      //  "p4"
      temp_av = sm_a[0 + idx_h3 * 4];

      //  "p5"
      temp_bv[0] = sm_b[idx_h2 + (idx_h1 + idx_p6 * 4) * 4 + (0 * COL)];
      temp_bv[1] = sm_b[idx_h2 + (idx_h1 + idx_p6 * 4) * 4 + (1 * COL)];
      temp_bv[2] = sm_b[idx_h2 + (idx_h1 + idx_p6 * 4) * 4 + (2 * COL)];
      temp_bv[3] = sm_b[idx_h2 + (idx_h1 + idx_p6 * 4) * 4 + (3 * COL)];

      //  "p4 x p5"
      reg_singles[0][0] += temp_av * temp_bv[0];
      reg_singles[0][1] += temp_av * temp_bv[1];
      reg_singles[0][2] += temp_av * temp_bv[2];
      reg_singles[0][3] += temp_av * temp_bv[3];

      temp_av = sm_a[1 + idx_h3 * 4];

      reg_singles[1][0] += temp_av * temp_bv[0];
      reg_singles[1][1] += temp_av * temp_bv[1];
      reg_singles[1][2] += temp_av * temp_bv[2];
      reg_singles[1][3] += temp_av * temp_bv[3];

      temp_av = sm_a[2 + idx_h3 * 4];

      reg_singles[2][0] += temp_av * temp_bv[0];
      reg_singles[2][1] += temp_av * temp_bv[1];
      reg_singles[2][2] += temp_av * temp_bv[2];
      reg_singles[2][3] += temp_av * temp_bv[3];

      temp_av = sm_a[3 + idx_h3 * 4];

      reg_singles[3][0] += temp_av * temp_bv[0];
      reg_singles[3][1] += temp_av * temp_bv[1];
      reg_singles[3][2] += temp_av * temp_bv[2];
      reg_singles[3][3] += temp_av * temp_bv[3];
      sycl::group_barrier(thread_block);
    }

    //
    //  >> s1_4:   t3[h3,h2,h1,p6,p5,p4] -= t2[p5,h1] * v2[h3,h2,p6,p4] (h3,h2,p6), (h1)
    //
    if(flag_s1_4 >= 0) // these if-conditions make 100 ms..
    {
      T* tmp_dev_s1_t1_4 = df_dev_s1_t1_all + size_max_dim_s1_t1 * flag_s1_4;
      T* tmp_dev_s1_v2_4 = df_dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_4;

      if(idx_h3 < rng_p5 && idx_h2 < rng_h1 && idx_p6 == 0 && idx_h1 == 0) {
        sm_a[idx_h3 + idx_h2 * FUSION_SIZE_SLICE_1_P5] =
          tmp_dev_s1_t1_4[str_blk_idx_p5 + idx_h3 + (str_blk_idx_h1 + idx_h2) * base_size_p5b];
      }

      if(idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6 && idx_h1 < rng_p4) {
        sm_b[idx_h3 + (idx_h2 + idx_p6 * 4) * 4 + (idx_h1 * COL)] =
          tmp_dev_s1_v2_4[str_blk_idx_h3 + idx_h3 +
                          (str_blk_idx_h2 + idx_h2 +
                           (str_blk_idx_p6 + idx_p6 + (str_blk_idx_p4 + idx_h1) * base_size_p6b) *
                             base_size_h2b) *
                            base_size_h3b];
      }
      sycl::group_barrier(thread_block);

      //  "p5"
      temp_av = sm_a[0 + idx_h1 * 4];

      //  "p4"
      temp_bv[0] = sm_b[idx_h3 + (idx_h2 + idx_p6 * 4) * 4 + (0 * COL)];
      temp_bv[1] = sm_b[idx_h3 + (idx_h2 + idx_p6 * 4) * 4 + (1 * COL)];
      temp_bv[2] = sm_b[idx_h3 + (idx_h2 + idx_p6 * 4) * 4 + (2 * COL)];
      temp_bv[3] = sm_b[idx_h3 + (idx_h2 + idx_p6 * 4) * 4 + (3 * COL)];

      //  "p4 x p5"
      reg_singles[0][0] -= temp_av * temp_bv[0];
      reg_singles[1][0] -= temp_av * temp_bv[1];
      reg_singles[2][0] -= temp_av * temp_bv[2];
      reg_singles[3][0] -= temp_av * temp_bv[3];

      temp_av = sm_a[1 + idx_h1 * 4];

      reg_singles[0][1] -= temp_av * temp_bv[0];
      reg_singles[1][1] -= temp_av * temp_bv[1];
      reg_singles[2][1] -= temp_av * temp_bv[2];
      reg_singles[3][1] -= temp_av * temp_bv[3];

      temp_av = sm_a[2 + idx_h1 * 4];

      reg_singles[0][2] -= temp_av * temp_bv[0];
      reg_singles[1][2] -= temp_av * temp_bv[1];
      reg_singles[2][2] -= temp_av * temp_bv[2];
      reg_singles[3][2] -= temp_av * temp_bv[3];

      temp_av = sm_a[3 + idx_h1 * 4];

      reg_singles[0][3] -= temp_av * temp_bv[0];
      reg_singles[1][3] -= temp_av * temp_bv[1];
      reg_singles[2][3] -= temp_av * temp_bv[2];
      reg_singles[3][3] -= temp_av * temp_bv[3];
      sycl::group_barrier(thread_block);
    }

    //
    //  >> s1_5:   t3[h3,h2,h1,p6,p5,p4] -= t2[p5,h2] * v2[h3,h1,p6,p4] (h3,h2,p6), (h1)
    //
    if(flag_s1_5 >= 0) // these if-conditions make 100 ms..
    {
      //
      T* tmp_dev_s1_t1_5 = df_dev_s1_t1_all + size_max_dim_s1_t1 * flag_s1_5;
      T* tmp_dev_s1_v2_5 = df_dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_5;

      if(idx_h3 < rng_p5 && idx_h2 < rng_h2 && idx_p6 == 0 && idx_h1 == 0) {
        sm_a[idx_h3 + idx_h2 * FUSION_SIZE_SLICE_1_P5] =
          tmp_dev_s1_t1_5[str_blk_idx_p5 + idx_h3 + (str_blk_idx_h2 + idx_h2) * base_size_p5b];
      }

      if(idx_h3 < rng_h3 && idx_h2 < rng_h1 && idx_p6 < rng_p6 && idx_h1 < rng_p4) {
        sm_b[idx_h3 + (idx_h2 + idx_p6 * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H3 + (idx_h1 * COL)] =
          tmp_dev_s1_v2_5[str_blk_idx_h3 + idx_h3 +
                          (str_blk_idx_h1 + idx_h2 +
                           (str_blk_idx_p6 + idx_p6 + (str_blk_idx_p4 + idx_h1) * base_size_p6b) *
                             base_size_h1b) *
                            base_size_h3b];
      }
      sycl::group_barrier(thread_block);

      //  "p5"
      temp_av = sm_a[0 + idx_h2 * 4];

      //  "p4"
      temp_bv[0] = sm_b[idx_h3 + (idx_h1 + idx_p6 * 4) * 4 + (0 * COL)];
      temp_bv[1] = sm_b[idx_h3 + (idx_h1 + idx_p6 * 4) * 4 + (1 * COL)];
      temp_bv[2] = sm_b[idx_h3 + (idx_h1 + idx_p6 * 4) * 4 + (2 * COL)];
      temp_bv[3] = sm_b[idx_h3 + (idx_h1 + idx_p6 * 4) * 4 + (3 * COL)];

      //  "p4 x p5"
      reg_singles[0][0] += temp_av * temp_bv[0];
      reg_singles[1][0] += temp_av * temp_bv[1];
      reg_singles[2][0] += temp_av * temp_bv[2];
      reg_singles[3][0] += temp_av * temp_bv[3];

      temp_av = sm_a[1 + idx_h2 * 4];

      reg_singles[0][1] += temp_av * temp_bv[0];
      reg_singles[1][1] += temp_av * temp_bv[1];
      reg_singles[2][1] += temp_av * temp_bv[2];
      reg_singles[3][1] += temp_av * temp_bv[3];

      temp_av = sm_a[2 + idx_h2 * 4];

      reg_singles[0][2] += temp_av * temp_bv[0];
      reg_singles[1][2] += temp_av * temp_bv[1];
      reg_singles[2][2] += temp_av * temp_bv[2];
      reg_singles[3][2] += temp_av * temp_bv[3];

      temp_av = sm_a[3 + idx_h2 * 4];

      reg_singles[0][3] += temp_av * temp_bv[0];
      reg_singles[1][3] += temp_av * temp_bv[1];
      reg_singles[2][3] += temp_av * temp_bv[2];
      reg_singles[3][3] += temp_av * temp_bv[3];
      sycl::group_barrier(thread_block);
    }

    //
    //  >> s1_6:   t3[h3,h2,h1,p6,p5,p4] -= t2[p5,h3] * v2[h2,h1,p6,p4] (h3,h2,p6), (h1)
    //
    if(flag_s1_6 >= 0) // these if-conditions make 100 ms..
    {
      //
      T* tmp_dev_s1_t1_6 = df_dev_s1_t1_all + size_max_dim_s1_t1 * flag_s1_6;
      T* tmp_dev_s1_v2_6 = df_dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_6;

      if(idx_h3 < rng_p5 && idx_h2 < rng_h3 && idx_p6 == 0 && idx_h1 == 0) {
        sm_a[idx_h3 + idx_h2 * FUSION_SIZE_SLICE_1_P5] =
          tmp_dev_s1_t1_6[str_blk_idx_p5 + idx_h3 + (str_blk_idx_h3 + idx_h2) * base_size_p5b];
      }

      if(idx_h3 < rng_h2 && idx_h2 < rng_h1 && idx_p6 < rng_p6 && idx_h1 < rng_p4) {
        sm_b[idx_h3 + (idx_h2 + idx_p6 * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H2 + (idx_h1 * COL)] =
          tmp_dev_s1_v2_6[str_blk_idx_h2 + idx_h3 +
                          (str_blk_idx_h1 + idx_h2 +
                           (str_blk_idx_p6 + idx_p6 + (str_blk_idx_p4 + idx_h1) * base_size_p6b) *
                             base_size_h1b) *
                            base_size_h2b];
      }
      sycl::group_barrier(thread_block);

      //  "p5"
      temp_av = sm_a[0 + idx_h3 * FUSION_SIZE_SLICE_1_P5];

      //  "p4"
      temp_bv[0] = sm_b[idx_h2 + (idx_h1 + idx_p6 * 4) * 4 + (0 * COL)];
      temp_bv[1] = sm_b[idx_h2 + (idx_h1 + idx_p6 * 4) * 4 + (1 * COL)];
      temp_bv[2] = sm_b[idx_h2 + (idx_h1 + idx_p6 * 4) * 4 + (2 * COL)];
      temp_bv[3] = sm_b[idx_h2 + (idx_h1 + idx_p6 * 4) * 4 + (3 * COL)];

      //  "p4 x p5"
      reg_singles[0][0] -= temp_av * temp_bv[0];
      reg_singles[1][0] -= temp_av * temp_bv[1];
      reg_singles[2][0] -= temp_av * temp_bv[2];
      reg_singles[3][0] -= temp_av * temp_bv[3];

      temp_av = sm_a[1 + idx_h3 * FUSION_SIZE_SLICE_1_P5];

      reg_singles[0][1] -= temp_av * temp_bv[0];
      reg_singles[1][1] -= temp_av * temp_bv[1];
      reg_singles[2][1] -= temp_av * temp_bv[2];
      reg_singles[3][1] -= temp_av * temp_bv[3];

      temp_av = sm_a[2 + idx_h3 * FUSION_SIZE_SLICE_1_P5];

      reg_singles[0][2] -= temp_av * temp_bv[0];
      reg_singles[1][2] -= temp_av * temp_bv[1];
      reg_singles[2][2] -= temp_av * temp_bv[2];
      reg_singles[3][2] -= temp_av * temp_bv[3];

      temp_av = sm_a[3 + idx_h3 * FUSION_SIZE_SLICE_1_P5];

      reg_singles[0][3] -= temp_av * temp_bv[0];
      reg_singles[1][3] -= temp_av * temp_bv[1];
      reg_singles[2][3] -= temp_av * temp_bv[2];
      reg_singles[3][3] -= temp_av * temp_bv[3];
      sycl::group_barrier(thread_block);
    }

    //
    //  >> s1_7:   t3[h3,h2,h1,p6,p5,p4] -= t2[p6,h1] * v2[h3,h2,p5,p4] (h3,h2,p6), (h1)
    //
    if(flag_s1_7 >= 0) // these if-conditions make 100 ms..
    {
      //
      T* tmp_dev_s1_t1_7 = df_dev_s1_t1_all + size_max_dim_s1_t1 * flag_s1_7;
      T* tmp_dev_s1_v2_7 = df_dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_7;

      if(idx_h3 < rng_p6 && idx_h2 < rng_h1 && idx_p6 == 0 && idx_h1 == 0) {
        sm_a[idx_h3 + idx_h2 * FUSION_SIZE_SLICE_1_P6] =
          tmp_dev_s1_t1_7[str_blk_idx_p6 + idx_h3 + (str_blk_idx_h1 + idx_h2) * base_size_p6b];
      }

      if(idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p5 && idx_h1 < rng_p4) {
        sm_b[idx_h3 + (idx_h2 + idx_p6 * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3 + (idx_h1 * COL)] =
          tmp_dev_s1_v2_7[str_blk_idx_h3 + idx_h3 +
                          (str_blk_idx_h2 + idx_h2 +
                           (str_blk_idx_p5 + idx_p6 + (str_blk_idx_p4 + idx_h1) * base_size_p5b) *
                             base_size_h2b) *
                            base_size_h3b];
      }
      sycl::group_barrier(thread_block);

      //  "p4" x "p5"
      double tmp_sm_a = sm_a[idx_p6 + idx_h1 * FUSION_SIZE_SLICE_1_P6];
      reg_singles[0][0] += tmp_sm_a * sm_b[idx_h3 + (idx_h2 + 0 * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3 + (0 * COL)];
      reg_singles[0][1] += tmp_sm_a * sm_b[idx_h3 + (idx_h2 + 1 * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3 + (0 * COL)];
      reg_singles[0][2] += tmp_sm_a * sm_b[idx_h3 + (idx_h2 + 2 * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3 + (0 * COL)];
      reg_singles[0][3] += tmp_sm_a * sm_b[idx_h3 + (idx_h2 + 3 * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3 + (0 * COL)];

      reg_singles[1][0] += tmp_sm_a * sm_b[idx_h3 + (idx_h2 + 0 * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3 + (1 * COL)];
      reg_singles[1][1] += tmp_sm_a * sm_b[idx_h3 + (idx_h2 + 1 * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3 + (1 * COL)];
      reg_singles[1][2] += tmp_sm_a * sm_b[idx_h3 + (idx_h2 + 2 * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3 + (1 * COL)];
      reg_singles[1][3] += tmp_sm_a * sm_b[idx_h3 + (idx_h2 + 3 * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3 + (1 * COL)];

      reg_singles[2][0] += tmp_sm_a * sm_b[idx_h3 + (idx_h2 + 0 * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3 + (2 * COL)];
      reg_singles[2][1] += tmp_sm_a * sm_b[idx_h3 + (idx_h2 + 1 * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3 + (2 * COL)];
      reg_singles[2][2] += tmp_sm_a * sm_b[idx_h3 + (idx_h2 + 2 * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3 + (2 * COL)];
      reg_singles[2][3] += tmp_sm_a * sm_b[idx_h3 + (idx_h2 + 3 * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3 + (2 * COL)];

      reg_singles[3][0] += tmp_sm_a * sm_b[idx_h3 + (idx_h2 + 0 * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3 + (3 * COL)];
      reg_singles[3][1] += tmp_sm_a * sm_b[idx_h3 + (idx_h2 + 1 * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3 + (3 * COL)];
      reg_singles[3][2] += tmp_sm_a * sm_b[idx_h3 + (idx_h2 + 2 * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3 + (3 * COL)];
      reg_singles[3][3] += tmp_sm_a * sm_b[idx_h3 + (idx_h2 + 3 * FUSION_SIZE_SLICE_1_H2) * FUSION_SIZE_SLICE_1_H3 + (3 * COL)];
      sycl::group_barrier(thread_block);
    }

    //
    //  >> s1_8:   t3[h3,h2,h1,p6,p5,p4] -= t2[p6,h2] * v2[h3,h1,p5,p4] (h3,h2,p6), (h1)
    //
    if(flag_s1_8 >= 0) // these if-conditions make 100 ms..
    {
      //
      T* tmp_dev_s1_t1_8 = df_dev_s1_t1_all + size_max_dim_s1_t1 * flag_s1_8;
      T* tmp_dev_s1_v2_8 = df_dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_8;

      if(idx_h3 < rng_p6 && idx_h2 < rng_h2 && idx_p6 == 0 && idx_h1 == 0) {
        sm_a[idx_h3 + idx_h2 * FUSION_SIZE_SLICE_1_P6] =
          tmp_dev_s1_t1_8[str_blk_idx_p6 + idx_h3 + (str_blk_idx_h2 + idx_h2) * base_size_p6b];
      }

      if(idx_h3 < rng_h3 && idx_h2 < rng_h1 && idx_p6 < rng_p5 && idx_h1 < rng_p4) {
        sm_b[idx_h3 + (idx_h2 + idx_p6 * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H3 + (idx_h1 * COL)] =
          tmp_dev_s1_v2_8[str_blk_idx_h3 + idx_h3 +
                          (str_blk_idx_h1 + idx_h2 +
                           (str_blk_idx_p5 + idx_p6 + (str_blk_idx_p4 + idx_h1) * base_size_p5b) *
                             base_size_h1b) *
                            base_size_h3b];
      }
      sycl::group_barrier(thread_block);

      //  "p4" x "p5"
      double tmp_sm_a = sm_a[idx_p6 + idx_h2 * FUSION_SIZE_SLICE_1_P6];
      reg_singles[0][0] -= tmp_sm_a * sm_b[idx_h3 + (idx_h1 + 0 * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H3 + (0 * COL)];
      reg_singles[0][1] -= tmp_sm_a * sm_b[idx_h3 + (idx_h1 + 1 * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H3 + (0 * COL)];
      reg_singles[0][2] -= tmp_sm_a * sm_b[idx_h3 + (idx_h1 + 2 * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H3 + (0 * COL)];
      reg_singles[0][3] -= tmp_sm_a * sm_b[idx_h3 + (idx_h1 + 3 * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H3 + (0 * COL)];

      reg_singles[1][0] -= tmp_sm_a * sm_b[idx_h3 + (idx_h1 + 0 * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H3 + (1 * COL)];
      reg_singles[1][1] -= tmp_sm_a * sm_b[idx_h3 + (idx_h1 + 1 * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H3 + (1 * COL)];
      reg_singles[1][2] -= tmp_sm_a * sm_b[idx_h3 + (idx_h1 + 2 * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H3 + (1 * COL)];
      reg_singles[1][3] -= tmp_sm_a * sm_b[idx_h3 + (idx_h1 + 3 * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H3 + (1 * COL)];

      reg_singles[2][0] -= tmp_sm_a * sm_b[idx_h3 + (idx_h1 + 0 * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H3 + (2 * COL)];
      reg_singles[2][1] -= tmp_sm_a * sm_b[idx_h3 + (idx_h1 + 1 * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H3 + (2 * COL)];
      reg_singles[2][2] -= tmp_sm_a * sm_b[idx_h3 + (idx_h1 + 2 * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H3 + (2 * COL)];
      reg_singles[2][3] -= tmp_sm_a * sm_b[idx_h3 + (idx_h1 + 3 * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H3 + (2 * COL)];

      reg_singles[3][0] -= tmp_sm_a * sm_b[idx_h3 + (idx_h1 + 0 * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H3 + (3 * COL)];
      reg_singles[3][1] -= tmp_sm_a * sm_b[idx_h3 + (idx_h1 + 1 * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H3 + (3 * COL)];
      reg_singles[3][2] -= tmp_sm_a * sm_b[idx_h3 + (idx_h1 + 2 * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H3 + (3 * COL)];
      reg_singles[3][3] -= tmp_sm_a * sm_b[idx_h3 + (idx_h1 + 3 * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H3 + (3 * COL)];
      sycl::group_barrier(thread_block);
    }

    //
    //  >> s1_9:   t3[h3,h2,h1,p6,p5,p4] -= t2[p6,h3] * v2[h2,h1,p5,p4] (h3,h2,p6), (h1)
    //
    if(flag_s1_9 >= 0) // these if-conditions make 100 ms..
    {
      //
      T* tmp_dev_s1_t1_9 = df_dev_s1_t1_all + size_max_dim_s1_t1 * flag_s1_9;
      T* tmp_dev_s1_v2_9 = df_dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_9;

      if(idx_h3 < rng_p6 && idx_h2 < rng_h3 && idx_p6 == 0 && idx_h1 == 0) {
        sm_a[idx_h3 + idx_h2 * FUSION_SIZE_SLICE_1_P6] =
          tmp_dev_s1_t1_9[str_blk_idx_p6 + idx_h3 + (str_blk_idx_h3 + idx_h2) * base_size_p6b];
      }

      if(idx_h3 < rng_h2 && idx_h2 < rng_h1 && idx_p6 < rng_p5 && idx_h1 < rng_p4) {
        sm_b[idx_h3 + (idx_h2 + idx_p6 * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H2 + (idx_h1 * COL)] =
          tmp_dev_s1_v2_9[str_blk_idx_h2 + idx_h3 +
                          (str_blk_idx_h1 + idx_h2 +
                           (str_blk_idx_p5 + idx_p6 + (str_blk_idx_p4 + idx_h1) * base_size_p5b) *
                             base_size_h1b) *
                            base_size_h2b];
      }
      sycl::group_barrier(thread_block);

      //  "p4" x "p5"
      double tmp_sm_a = sm_a[idx_p6 + idx_h3 * FUSION_SIZE_SLICE_1_P6];
      reg_singles[0][0] += tmp_sm_a * sm_b[idx_h2 + (idx_h1 + 0 * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H2];
      reg_singles[0][1] += tmp_sm_a * sm_b[idx_h2 + (idx_h1 + 1 * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H2];
      reg_singles[0][2] += tmp_sm_a * sm_b[idx_h2 + (idx_h1 + 2 * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H2];
      reg_singles[0][3] += tmp_sm_a * sm_b[idx_h2 + (idx_h1 + 3 * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H2];

      reg_singles[1][0] += tmp_sm_a * sm_b[idx_h2 + (idx_h1 + 0 * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H2 + (1 * COL)];
      reg_singles[1][1] += tmp_sm_a * sm_b[idx_h2 + (idx_h1 + 1 * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H2 + (1 * COL)];
      reg_singles[1][2] += tmp_sm_a * sm_b[idx_h2 + (idx_h1 + 2 * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H2 + (1 * COL)];
      reg_singles[1][3] += tmp_sm_a * sm_b[idx_h2 + (idx_h1 + 3 * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H2 + (1 * COL)];

      reg_singles[2][0] += tmp_sm_a * sm_b[idx_h2 + (idx_h1 + 0 * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H2 + (2 * COL)];
      reg_singles[2][1] += tmp_sm_a * sm_b[idx_h2 + (idx_h1 + 1 * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H2 + (2 * COL)];
      reg_singles[2][2] += tmp_sm_a * sm_b[idx_h2 + (idx_h1 + 2 * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H2 + (2 * COL)];
      reg_singles[2][3] += tmp_sm_a * sm_b[idx_h2 + (idx_h1 + 3 * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H2 + (2 * COL)];

      reg_singles[3][0] += tmp_sm_a * sm_b[idx_h2 + (idx_h1 + 0 * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H2 + (3 * COL)];
      reg_singles[3][1] += tmp_sm_a * sm_b[idx_h2 + (idx_h1 + 1 * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H2 + (3 * COL)];
      reg_singles[3][2] += tmp_sm_a * sm_b[idx_h2 + (idx_h1 + 2 * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H2 + (3 * COL)];
      reg_singles[3][3] += tmp_sm_a * sm_b[idx_h2 + (idx_h1 + 3 * FUSION_SIZE_SLICE_1_H1) * FUSION_SIZE_SLICE_1_H2 + (3 * COL)];
      sycl::group_barrier(thread_block);
    }
  }

  //  energies
  T energy_1 = 0.0;
  T energy_2 = 0.0;
  if(idx_h3 < energy_rng_h3 && idx_h2 < energy_rng_h2 && idx_p6 < energy_rng_p6 && idx_h1 < energy_rng_h1) {
    for(unsigned short j = 0; j < FUSION_SIZE_SLICE_1_P4; j++) {
      for(unsigned short i = 0; i < FUSION_SIZE_SLICE_1_P5; i++) {
        if(i < energy_rng_p5 && j < energy_rng_p4) {
          //
          T inner_factor = partial_inner_factor - dev_evl_sorted_p5b[i + (energy_str_blk_idx_p5)] -
                           dev_evl_sorted_p4b[j + (energy_str_blk_idx_p4)];
          //
          energy_1 += (reg_tile[j][i] * reg_tile[j][i]) / inner_factor;
          energy_2 += (reg_tile[j][i] * (reg_tile[j][i] + reg_singles[j][i])) / inner_factor;
        }
      }
    }
  }
  sycl::group_barrier(thread_block);

  //
  //  to partially reduce the energies--- E(4) and E(5)
  //  a warp: 32 -(1)-> 16 -(2)-> 8 -(3)-> 4 -(4)-> 2
  //
  // sm_a[16][64]
  // sm_b[16][64]
  sm_a[threadIdx_x + threadIdx_y * COL] = energy_1;
  sm_b[threadIdx_x + threadIdx_y * COL] = energy_2;
  sycl::group_barrier(thread_block);

  T final_energy_1 = 0.0;
  T final_energy_2 = 0.0;
  if(threadIdx_x == 0 && threadIdx_y == 0) {
    for(unsigned short j = 0; j < 16; j++) {
      for(unsigned short i = 0; i < 16; i++) {
        final_energy_1 += sm_a[i + j * COL];
        final_energy_2 += sm_b[i + j * COL];
      }
    }

    reduced_energy[blockIdx_x]                           = final_energy_1;
    reduced_energy[blockIdx_x + item.get_group_range(1)] = final_energy_2;
  }
}

template<typename T>
void fully_fused_ccsd_t_gpu(gpuStream_t& stream_id, size_t num_blocks, size_t base_size_h1b,
                            size_t base_size_h2b, size_t base_size_h3b, size_t base_size_p4b,
                            size_t base_size_p5b, size_t base_size_p6b,
                            //
                            T* df_dev_d1_t2_all, T* df_dev_d1_v2_all, T* df_dev_d2_t2_all,
                            T* df_dev_d2_v2_all, T* df_dev_s1_t1_all, T* df_dev_s1_v2_all,
                            //
                            size_t size_d1_t2_all, size_t size_d1_v2_all, size_t size_d2_t2_all,
                            size_t size_d2_v2_all, size_t size_s1_t1_all, size_t size_s1_v2_all,
                            //
                            int* host_d1_size, int* host_d1_exec, // used
                            int* host_d2_size, int* host_d2_exec, int* host_s1_size,
                            int* host_s1_exec,
                            //
                            size_t size_noab, size_t size_max_dim_d1_t2, size_t size_max_dim_d1_v2,
                            size_t size_nvab, size_t size_max_dim_d2_t2, size_t size_max_dim_d2_v2,
                            size_t size_max_dim_s1_t1, size_t size_max_dim_s1_v2,
                            //
                            T factor,
                            //
                            T* dev_evl_sorted_h1b, T* dev_evl_sorted_h2b, T* dev_evl_sorted_h3b,
                            T* dev_evl_sorted_p4b, T* dev_evl_sorted_p5b, T* dev_evl_sorted_p6b,
                            //
                            T* partial_energies, gpuEvent_t* done_compute, gpuEvent_t* done_copy) {
  // 	to call the fused kernel for singles, doubles and energies.
  // jk_ccsd_t_fully_fused_kernel_associative

  (*done_compute)[0] = stream_id.submit([&](sycl::handler& cgh) {
    // allocate local/shared memory
    sycl::range<1> sm_a_range(16 * 64);
    sycl::range<1> sm_b_range(16 * 64);
    localAcc<T>    sm_a_acc(sm_a_range, cgh);
    localAcc<T>    sm_b_acc(sm_b_range, cgh);

    // Depends on # of Fused Kernel
    sycl::range<2> gridsize(1, num_blocks);
    sycl::range<2> blocksize(FUSION_SIZE_TB_1_Y, FUSION_SIZE_TB_1_X);
    auto           global_range = gridsize * blocksize;

    cgh.depends_on(*done_copy);

    cgh.parallel_for(
      sycl::nd_range<2>(global_range, blocksize),
      [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(8)]] {
        revised_jk_ccsd_t_fully_fused_kernel(
          size_noab, size_nvab, size_max_dim_s1_t1, size_max_dim_s1_v2, size_max_dim_d1_t2,
          size_max_dim_d1_v2, size_max_dim_d2_t2, size_max_dim_d2_v2, df_dev_d1_t2_all,
          df_dev_d1_v2_all, df_dev_d2_t2_all, df_dev_d2_v2_all, df_dev_s1_t1_all, df_dev_s1_v2_all,
          dev_evl_sorted_h1b, dev_evl_sorted_h2b, dev_evl_sorted_h3b, dev_evl_sorted_p4b,
          dev_evl_sorted_p5b, dev_evl_sorted_p6b, partial_energies,
          CEIL(base_size_h3b, FUSION_SIZE_SLICE_1_H3), CEIL(base_size_h2b, FUSION_SIZE_SLICE_1_H2),
          CEIL(base_size_h1b, FUSION_SIZE_SLICE_1_H1), CEIL(base_size_p6b, FUSION_SIZE_SLICE_1_P6),
          CEIL(base_size_p5b, FUSION_SIZE_SLICE_1_P5), CEIL(base_size_p4b, FUSION_SIZE_SLICE_1_P4),
          base_size_h1b, base_size_h2b, base_size_h3b, base_size_p4b, base_size_p5b, base_size_p6b,
          item, host_s1_size, host_s1_exec, host_d1_size, host_d1_exec, host_d2_size, host_d2_exec,
          sm_a_acc.get_pointer(), sm_b_acc.get_pointer());
      });
  });
}
