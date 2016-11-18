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
#ifndef TAMM_TENSOR_INDEX_H_
#define TAMM_TENSOR_INDEX_H_

#include "tensor/common.h"
// #include "typesf2c.h"
#include "tensor/define.h"

namespace tamm {

/**
 * @FIXME @BUG: ext_sym_group_ and probably this whole class needs to
 * be removed.
 */

class Index {
 public:
  /**
   * Constructor
   */
  Index() {}

  /**
   * Constructor. Assign data to this index.
   * @param[in] name Name of the index
   * @param[in] range Range of the index
   * @param[in] esg External symmetry group of the index
   */
  Index(const IndexName& name, const int& esg)
      : name_(name), value_(0), value_r_(0), ext_sym_group_(esg) {}

  /**
   * Destructor
   */
  ~Index() {}

  /**
   * Get the name of the index
   * @return name as a string
   */
  inline IndexName name() const { return name_; }

  /**
   * Get the value of the index
   * @return value as a Fint
   */
  inline Fint value() const { return value_; }

  /**
   * Get the value(restricted) of the index
   * @return value_r as an Fint
   */
  inline Fint value_r() const { return value_r_; }

  /**
   * Set the value of this index
   * @param[in] val value of this index
   */
  inline void setValue(const Fint& val) { value_ = val; }

  /**
   * Set the restricted value of this index
   * @param[in] val_r restricted value of this index
   */
  inline void setValueR(const Fint& val_r) { value_r_ = val_r; }

 private:
  /**
   * Get the external symmetry group id of the index
   * @return ext_sym_group as an int
   */
  inline int ext_sym_group() const { return ext_sym_group_; }

  IndexName name_;    /*< name of this index */
  Fint value_;        /*< value of this index */
  Fint value_r_;      /*< value(restricted) of this index */
  int ext_sym_group_;  //  < external symmetry group of this index, from lhs
                       //  of the expr(tC)

  friend bool compareExtSymGroup(const Index& lhs, const Index& rhs);
};

inline bool compareValue(const Index& lhs, const Index& rhs) {
  return (lhs.value() < rhs.value());
}
inline bool compareExtSymGroup(const Index& lhs, const Index& rhs) {
  return (lhs.ext_sym_group() < rhs.ext_sym_group());
}

} /* namespace tamm */

#endif  // TAMM_TENSOR_INDEX_H_
