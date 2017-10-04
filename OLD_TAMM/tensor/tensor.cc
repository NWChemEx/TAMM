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
#include "tensor/tensor.h"
#include <iostream>
#include "tensor/capi.h"
#include "tensor/expression.h"
#include "tensor/gmem.h"
#include <ga.h>

#include <cmath>

#include <random>
using std::vector;

namespace tamm {

bool Tensor::check_correctness(Tensor *F_tensor) {
  bool ret_val = true;
  int64_t me = gmem::rank();
  gmem::Handle F_g_a = F_tensor->ga();
  gmem::Handle C_g_a = ga();
  int64_t F_lo, F_hi, F_ld;
  int64_t C_lo, C_hi, C_ld;

  NGA_Distribution64(F_g_a, me, &F_lo, &F_hi);
  NGA_Distribution64(C_g_a, me, &C_lo, &C_hi);

  if (F_lo != C_lo || F_hi != C_hi) {
      std::cout << "In function " << __func__ << std::endl;
      std::cout << "; Buffer Dimension for C & F Tensors differ" << std::endl;
      return ret_val = false;
  }
  double *F_buf;
  double *C_buf;
  // use NGA_Access to access the buffer and initialize
  NGA_Access64(F_g_a, &F_lo, &F_hi, reinterpret_cast<void*>(&F_buf), &F_ld);
  NGA_Access64(C_g_a, &C_lo, &C_hi, reinterpret_cast<void*>(&C_buf), &C_ld);

  int deci_place = 11;
  for (int i = 0; i <= (F_hi - F_lo); ++i) {
	// std::cout << "F_buf[" << i << "]: " << F_buf[i] <<
	// 		" | C_buf[" << i << "]: " << C_buf[i] << std::endl;
  if( std::fabs(F_buf[i]- C_buf[i]) > pow(10,(-1.0 * deci_place))) {
      ret_val = false;
    }
  }

  NGA_Release64(F_g_a, &F_lo, &F_hi);
  NGA_Release64(C_g_a, &C_lo, &C_hi);

  return ret_val;
}

void Tensor::fill_random() {
  // std::cout << "File: " << __FILE__ <<"On Line: " << __LINE__ << std::endl;
  int64_t me = gmem::rank();
  gmem::Handle g_a = ga();
  // std::cout << "On Line: " << __LINE__ << "; ga_handle: " << g_a << std::endl;
  int64_t lo, hi, ld;
  // use NGA_Distribution to get the buffer
  NGA_Distribution64(g_a, me, &lo, &hi);
  // auto num_elements = hi - lo + 1;
  // std::cout << "On Line: " << __LINE__ << "; lo: " << lo << std::endl;
  // std::cout << "On Line: " << __LINE__ << "; hi: " << hi << std::endl;
  double *buf;

  NGA_Access_block_segment64(g_a, me, reinterpret_cast<void*>(&buf), &ld);
  // std::cout << "On Line: " << __LINE__ << "; ld: " << ld << std::endl;

  // use NGA_Access to access the buffer and initialize
  NGA_Access64(g_a, &lo, &hi, reinterpret_cast<void*>(&buf), &ld);

  double lower_fill_value = 0;
  double upper_fil_value = 10;
  std::uniform_real_distribution<double> uniform_distri(lower_fill_value,
		  upper_fil_value);
  std::default_random_engine rand;

  for (int i = 0; i <= (hi - lo); ++i) {
	// std::cout << "On Line: " << __LINE__ << "; i: " << i << std::endl;
	// std::cout << "On Line: " << __LINE__ << "; rand: " << uniform_distri(rand) << std::endl;
	buf[i] = uniform_distri(rand);
  }
  NGA_Release64(g_a, &lo, &hi);
}

void Tensor::fill_given(const double fill_val) {
  int64_t me = gmem::rank();
  gmem::Handle g_a = ga();
  int64_t lo, hi, ld;

  // use NGA_Distribution to get the buffer
  NGA_Distribution64(g_a, me, &lo, &hi);

  double *buf;
  NGA_Access_block_segment64(g_a, me, reinterpret_cast<void*>(&buf), &ld);

  // use NGA_Access to access the buffer and initialize
  NGA_Access64(g_a, &lo, &hi, reinterpret_cast<void*>(&buf), &ld);

  for (int i = 0; i <= (hi - lo); ++i) {
    buf[i] = fill_val;
  }
  NGA_Release64(g_a, &lo, &hi);
}

Tensor::Tensor(int n, int nupper, int irrep_val, RangeType rt[],
               DistType dist_type)
    : dim_(n),
      nupper_(nupper),
      irrep_(irrep_val),
      allocated_(false),
      attached_(false),
      dist_type_(dist_type), 
      offset_map_(nullptr),
      offset_index_(-1),
      offset_handle_(-1) {
  
  assert(n >= 0);
  assert(nupper_ >= 0 && nupper_ <= dim_);

  vector<IndexName> name(n);
  vector<int> ctr(RANGE_UB, 0);
  dim_type_ = dim_n;
  for (int i = 0; i < n; i++) {
    switch (rt[i]) {
      case TO:
        name[i] = (IndexName)(H1B + ctr[TO]);
        ctr[TO]++;
        dim_type_ = dim_ov;
        break;
      case TV:
        name[i] = (IndexName)(P1B + ctr[TV]);
        dim_type_ = dim_ov;
        ctr[TV]++;
      case TN: /*@FIXME @BUG should have a separate index name*/
        name[i] = (IndexName)(P1B + ctr[TV]);
        ctr[TV]++;
        break;
      default:
        assert(false);
    }
  }
  if (dim_ == 0) {
    dim_type_ = dim_ov;
  }
  ids_ = name2ids(*this, name);
}

void Tensor::gen_restricted(const std::vector<size_t> &value_,
                            std::vector<size_t> *pvalue_r) const {
  tamm_restricted(dim_, nupper_, value_, pvalue_r);
}

void Tensor::create() {
  const std::vector<IndexName> &name = id2name(ids());
  // const std::vector<int>& gp = ext_sym_group(*this,ids());
  const std::vector<int> &gp = ext_sym_group(*this, name);
  std::vector<std::vector<IndexName> > all;
  std::vector<IndexName> one;
  int prev = 0, curr = 0, n = dim_;

  assert(dim_type_ ==
         dim_ov);  // @FIXME @BUG: not implemented for other dime types

  for (int i = 0; i < n; i++) {
    curr = gp[i];
    if (curr != prev) {
      all.push_back(one);
      one.clear();
      prev = curr;
    }
    one.push_back(name[i]);
  }
  all.push_back(one);
  std::vector<triangular> vt(all.size());
  for (int i = 0; i < all.size(); i++) {
    triangular tr(all[i]);
    vt[i] = tr;
  }
  IterGroup<triangular> out_itr = IterGroup<triangular>(vt, TRIG);

  Fint length = 0;
  vector<size_t> out_vec;
  out_itr.reset();
  while (out_itr.next(&out_vec)) {
    if (is_spatial_nonzero(out_vec) && is_spin_nonzero(out_vec) &&
        is_spin_restricted_nonzero(out_vec)) {
      length++;
    }
  }

  offset_map_ = static_cast<Fint *>(malloc(sizeof(Fint) * 2 * length + 1));
  assert(offset_map_ != nullptr);
  assert(dim_type_ == dim_n || dim_type_ == dim_ov);
  size_t noab = Variables::noab();
  size_t nvab = Variables::nvab();
  Fint *int_mb = Variables::int_mb();

  offset_map_[0] = length;
  size_t addr = 0;
  size_t size = 0;
  out_itr.reset();
  while (out_itr.next(&out_vec)) {
    if (is_spatial_nonzero(out_vec) && is_spin_nonzero(out_vec) &&
        is_spin_restricted_nonzero(out_vec)) {
      size_t offset = 1, key = 0;
      if (dim_type_ == dim_n) {
        for (int i = n - 1; i >= 0; i--) {
          key += (out_vec[i] - 1) * offset;
          offset *= noab + nvab;
        }
      } else if (dim_type_ == dim_ov) {
        for (int i = n - 1; i >= 0; i--) {
          bool check = (Table::rangeOf(name[i]) == TO);
          if (check)
            key += (out_vec[i] - 1) * offset;
          else
            key += (out_vec[i] - noab - 1) * offset;  // TV
          offset *= (check) ? noab : nvab;
        }
      }

      addr += 1;
      offset_map_[addr] = key;
      offset_map_[length + addr] = size;
      size += compute_size(out_vec);
    }
  }

  if (size == 0) {
    size = 1;
  }
  {
    // int ndims = 2;
    // int dims[2] = {1, size};
    std::string gmc = "noname1";
    std::cout<<"----TAMM. ga create. size="<<size<<std::endl;
    ga_ = gmem::create(gmem::Double, size, gmc);
  }
  gmem::zero(ga_);
  offset_index_ = offset_map_ - int_mb;
  allocated_ = true;
}

void Tensor::attach(Fint fma_offset_index, Fint fma_handle, Fint array_handle) {
  Fint *int_mb = Variables::int_mb();
  ga_ = (gmem::Handle)array_handle;
  offset_index_ = fma_offset_index;
  offset_handle_ = fma_handle;
  offset_map_ = int_mb + fma_offset_index;
  attached_ = true;
}

void Tensor::destroy() {
  if (allocated_) {
    gmem::destroy(ga_);
    free(offset_map_);
    allocated_ = false;
  } else {
    assert(false);
  }
}

void Tensor::detach() {
  assert(attached_);
  attached_ = false;
}

void Tensor::get(const std::vector<size_t> &pvalue_r, double *buf,
                 size_t size) const {
  assert(allocated_ || attached_);
  gmem::Handle d_a = ga();
  size_t d_a_offset = offset_index();
  const std::vector<size_t> &is = pvalue_r;
  const std::vector<IndexName> &ns = id2name(ids_);
  Fint key = 0, offset = 1;
  size_t noab = Variables::noab();
  size_t nvab = Variables::nvab();
  Fint *int_mb = Variables::int_mb();
  int n = is.size();

  assert(offset_index() == d_a_offset);
  if (dim_type_ == dim_n) {
    for (int i = n - 1; i >= 0; i--) {
      key += (is[i] - 1) * offset;
      offset *= noab + nvab;
    }
  } else if (dim_type_ == dim_ov) {
    for (int i = n - 1; i >= 0; i--) {
      bool check = (Table::rangeOf(ns[i]) == TO);
      if (check)
        key += (is[i] - 1) * offset;
      else
        key += (is[i] - noab - 1) * offset;  // TV
      offset *= (check) ? noab : nvab;
    }
  } else {
    assert(false);
  }
  if (dist_type_ == dist_nwi) {
    assert(Variables::intorb() != 0);
    assert(dim_ == 4);
    cget_hash_block_i(d_a, buf, size, d_a_offset, key, is);
  } else if (dist_type_ == dist_nwma) {
    double *dbl_mb = Variables::dbl_mb();
    cget_hash_block_ma(d_a, buf, size, d_a_offset, key);
  } else if (dist_type_ == dist_nw) {
    cget_hash_block(d_a, buf, size, d_a_offset, key);
  } else {
    assert(false);
  }
}

void Tensor::add(const std::vector<size_t> &is, double *buf,
                 size_t size) const {
  assert(allocated_ || attached_);
  gmem::Handle d_c = ga();
  Fint *int_mb = Variables::int_mb();
  Fint *hash = &int_mb[offset_index()];
  size_t noab = Variables::noab();
  size_t nvab = Variables::nvab();
  const std::vector<IndexName> &ns = id2name(ids_);

  size_t key = 0, offset = 1;
  for (int i = is.size() - 1; i >= 0; i--) {
    bool check = (Table::rangeOf(ns[i]) == TO);
    if (check)
      key += (is[i] - 1) * offset;
    else
      key += (is[i] - noab - 1) * offset;  // TV
    offset *= (check) ? noab : nvab;
  }
  cadd_hash_block(d_c, buf, size, hash, key);
}

Tensor Tensor0_1(RangeType r1, DistType dt, int irrep) {
  RangeType rts[1] = {r1};
  return Tensor(1, 0, irrep, rts, dt);
}

Tensor Tensor2(RangeType r1, RangeType r2, DistType dt) {
  RangeType rts[2] = {r1, r2};
  return Tensor(2, 1, 0, rts, dt);
}

Tensor Tensor1_2(RangeType r1, RangeType r2, RangeType r3, DistType dt,
                 int irrep) {
  RangeType rts[3] = {r1, r2, r3};
  return Tensor(3, 1, irrep, rts, dt);
}

Tensor Tensor4(RangeType r1, RangeType r2, RangeType r3, RangeType r4,
               DistType dt) {
  RangeType rts[4] = {r1, r2, r3, r4};
  return Tensor(4, 2, 0, rts, dt);
}

} /*namespace tamm*/


