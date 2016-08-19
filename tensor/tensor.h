#ifndef __ctce_tensor_h__
#define __ctce_tensor_h__

#include "index.h"
#include "variables.h"
#include <vector>
#include <cassert>
#include <algorithm>

namespace ctce {

enum DistType { dist_nwi, dist_nwma, dist_nw};
enum DimType { dim_ov, dim_n };

class Tensor {
public:
  /**
   * Constructor
   */
  Tensor() {}
  /**
   * Destructor
   */
  ~Tensor() {}

  Tensor(int n, int nupper, int irrep_val, RangeType rt[], DistType dist_type);

  bool attached() const;
  bool allocated() const;
  int ga() const;
  size_t offset_index() const;
  int offset_handle() const;

  /**
   * Get the dimension of this tensor
   * @return dim as a int
   */
  int dim() const;

  int nupper() const;

  /**
   * Get the indices of the tensor
   * @return ids as a vector of Index
   */
  const std::vector<Index>& ids() const;

  /**
   * Get the corresponding irrep value from the global variables
   * @return irrep as integer
   */
  int irrep() const;

  int set_irrep(int irrep);
  int set_dist(DistType dist);

  void get(std::vector<size_t> &pvalue_r, double *buf, size_t size);

  void add(std::vector<size_t> &pvalue_r, double *buf, size_t size);

  bool is_spin_restricted_nonzero(const std::vector<size_t>& ids) const;

  bool is_spin_nonzero(const std::vector<size_t>& ids) const;

  bool is_spatial_nonzero(const std::vector<size_t> &ids) const;

  /**
   * Generate restricted value from value by calling tce_restricted2/4
   */
  void gen_restricted(const std::vector<size_t> &value,
                      std::vector<size_t> &pvalue_r);

  void create();
  void attach(Fint fma_offset_index, Fint fma_offset_handle, Fint array_handle);

  void destroy();
  void detach();


private:
  int dim_; /*< dimension of this tensor */
  std::vector<Index> ids_; /*< indices of the tensor, actual data */

  bool allocated_; /*true if this tensor were created using create()*/
  bool attached_;
  int ga_; /*underlying ga if this tensor was created*/
  Fint *offset_map_; /*offset map used as part of creation*/
  size_t offset_index_; /*index to offset map usable with int_mb */
  int offset_handle_; /*MA handle for the offset map when allocated in fortran*/
  int irrep_; /*irrep for spatial symmetry*/
  int nupper_; /* number of upper indices*/

  DistType dist_type_;
  DimType dim_type_;
}; // Tensor

/**
 * Inline implementations
 */

inline bool
Tensor::attached() const { return attached_; }

inline bool
Tensor::allocated() const { return allocated_; }

inline int
Tensor::ga() const { return ga_; }

inline size_t
Tensor::offset_index() const { return offset_index_; }

inline int
Tensor::offset_handle() const { return offset_handle_; }

inline int
Tensor::dim() const { return dim_; }

inline int
Tensor:: nupper() const { return nupper_; }

inline const std::vector<Index>&
Tensor::ids() const { return ids_; }

inline int
Tensor::irrep() const { return irrep_; }

inline int
Tensor::set_irrep(int irrep) { irrep_ = irrep; }

inline int
Tensor::set_dist(DistType dist) { dist_type_ = dist; }

inline bool
Tensor::is_spin_restricted_nonzero(const std::vector<size_t>& ids) const {
  int lval = dim_ - 2*nupper_;
  assert(lval >= 0 && lval <=1);
  assert(ids.size() == dim_);
  int dim_even = dim_ + (dim_%2);
  Fint *int_mb = Variables::int_mb();
  size_t k_spin = Variables::k_spin()-1;
  size_t restricted = Variables::restricted();
  for (int i=0; i<ids.size(); i++) lval += int_mb[k_spin+ids[i]];
  assert ((dim_%2==0) || (!restricted) || (lval != 2*dim_even));
  return ((!restricted) || (dim_==0) || (lval != 2*dim_even));
}

inline bool
Tensor::is_spin_nonzero(const std::vector<size_t>& ids) const {
  int lval=0, rval=0;
  Fint *int_mb = Variables::int_mb();
  Fint k_spin = Variables::k_spin()-1;
  for(int i=0; i<nupper_; i++) lval += int_mb[k_spin+ids[i]];
  for(int i=nupper_; i<dim_; i++) rval += int_mb[k_spin+ids[i]];
  return (rval - lval == dim_ - 2*nupper_);
}

inline bool
Tensor::is_spatial_nonzero(const std::vector<size_t> &ids) const {
  Fint lval=0;
  Fint *int_mb = Variables::int_mb();
  Fint k_sym = Variables::k_sym()-1;
  for(int i=0; i<ids.size(); i++) lval ^= int_mb[k_sym+ids[i]];
  return (lval == irrep_);
}

Tensor Tensor0_1(RangeType r1, DistType dt, int irrep);

Tensor Tensor2(RangeType r1, RangeType r2, DistType dt);

Tensor Tensor1_2(RangeType r1, RangeType r2, RangeType r3,
                 DistType dt, int irrep);

Tensor Tensor4(RangeType r1, RangeType r2, RangeType r3, RangeType r4,
               DistType dt);

} /* namespace ctce */

#endif /* __ctce_tensor_h__ */
