// Copyright 2016 Pacific Northwest National Laboratory

#ifndef TAMMY_INDEX_SPACE_H_
#define TAMMY_INDEX_SPACE_H_

#include <algorithm>
#include <functional>
#include <vector>
#include <cassert>
#include <iostream>
#include <memory>

#include "tammy/boundvec.h"
#include "tammy/types.h"

namespace tammy {

class IndexRange;

class IndexSpace {
 public:
  virtual bool has_spatial() const {
    return false;
  }
  virtual Irrep spatial(BlockIndex) const {
    return Irrep{0};
  }
  virtual bool has_spin() const {
    return false;
  }
  virtual Spin spin(BlockIndex) const {
    return Spin{0};
  }
  virtual BlockIndex blo(IndexRange ir) const = 0;
  virtual BlockIndex bhi(IndexRange ir) const = 0;

  virtual IndexRange E() const = 0;
  virtual IndexRange N() const = 0;

  virtual size_t size(BlockIndex bid) const = 0;
  virtual Offset offset(BlockIndex bid) const = 0;

  bool is_identical_to(const IndexSpace& is) const {
    return this == &is;
  }

  //strict definition of compatbility
  bool is_compatible_with(const IndexSpace& is) const {
    return is_identical_to(is);
  }

  bool is_superset_of(const IndexRange& ir1, const IndexRange& ir2) const;
};  // IndexSpace


class IndexRange {
 public:

  IndexRange()
      : is_{nullptr} {}
  
  IndexRange(const IndexSpace& is,
             RangeValue rv)
      : is_{&is},
        rv_{rv} {}

  IndexRange(const IndexRange& ir)
      : IndexRange{*ir.is_, ir.rv_} {}

  IndexRange& operator = (const IndexRange& ir) {
    is_ = ir.is_;
    rv_ = ir.rv_;
    return *this;
  }

  const IndexSpace& is() const {
    return *is_;
  }

  const RangeValue rv() const {
    return rv_;
  }

  BlockIndex blo() const {
    return is_->blo(*this);
  }

  BlockIndex bhi() const {
    return is_->bhi(*this);
  }

  size_t size(BlockIndex bid) const {
    return is_->size(bid);
  }

  Offset offset(BlockIndex bid) const {
    return is_->offset(bid);
  }

  bool is_superset(const IndexRange& ir) const {
    return is_->is_superset_of(*this, ir);
  }
  
 private:
  const IndexSpace* is_;
  RangeValue rv_;
};  // Indexrange

inline bool
operator == (const IndexRange& lhs, const IndexRange& rhs) {
  return lhs.rv() == rhs.rv()
      && lhs.is().is_identical_to(rhs.is());
}

inline bool
operator <= (const IndexRange& lhs, const IndexRange& rhs) {
  return lhs.rv() <= rhs.rv()
      && &lhs.is() <= &rhs.is();
}


class MSO : public IndexSpace {
 public:
  bool has_spatial() const override {
    return true;
  }

  Irrep spatial(BlockIndex /*bi*/) const override {
    return Irrep{0};
  }

  virtual bool has_spin() const override {
    return true;
  }
  virtual Spin spin(BlockIndex) const override {
    return Spin{0};
  }

  BlockIndex blo(IndexRange rv) const override;
  BlockIndex bhi(IndexRange rv) const override;

  IndexRange N() const override;
  IndexRange E() const override;

  IndexRange O() const;
  IndexRange V() const;
  IndexRange Oa() const;
  IndexRange Va() const;
  IndexRange Ob() const;
  IndexRange Vb() const;
};  // MSO

class MO : public IndexSpace {
 public:
  bool has_spatial() const override {
    return true;
  }
  Irrep spatial(BlockIndex /*bi*/) const override {
    return Irrep{0};
  }
  virtual bool has_spin() const override {
    return true;
  }
  virtual Spin spin(BlockIndex) const override {
    return Spin{0};
  }

  BlockIndex blo(IndexRange rv) const override;
  BlockIndex bhi(IndexRange rv) const override;

  IndexRange N() const override;
  IndexRange E() const override;

  IndexRange alpha() const;
  IndexRange beta() const;
  IndexRange Oa() const;
  IndexRange Va() const;
  IndexRange Ob() const;
  IndexRange Vb() const;
};  // MO

class AO : public IndexSpace {
 public:
  BlockIndex blo(IndexRange rv) const override;
  BlockIndex bhi(IndexRange rv) const override;

  IndexRange N() const override;
  IndexRange E() const override;

  IndexRange O() const;
  IndexRange V() const;
};  // MO

class ASO : public IndexSpace {
 public:
  virtual bool has_spin() const override {
    return true;
  }
  virtual Spin spin(BlockIndex) const override {
    return Spin{0};
  }

  BlockIndex blo(IndexRange rv) const override;
  BlockIndex bhi(IndexRange rv) const override;

  IndexRange N() const override;
  IndexRange E() const override;

  IndexRange alpha() const;
  IndexRange beta() const;
  IndexRange Oa() const;
  IndexRange Va() const;
  IndexRange Ob() const;
  IndexRange Vb() const;
};  // ASO

class AUX : public IndexSpace {
 public:
  BlockIndex blo(IndexRange rv) const override;
  BlockIndex bhi(IndexRange rv) const override;

  IndexRange N() const override;
  IndexRange E() const override;
};  // AO


class IndexLabel {
 public:
  IndexLabel() = default;

  IndexLabel(int lbl, const IndexRange& ir)
      : ir_{ir},
        label_{lbl} { }

  const IndexRange& ir() const {
    return ir_;
  }

  int label() const {
    return label_;
  }
  
 private:
  IndexRange ir_;
  int label_;  
};

using IndexLabelVec = TensorVec<IndexLabel>;

inline bool
operator == (const IndexLabel& lhs, const IndexLabel& rhs) {
  return lhs.label() == rhs.label()
      && lhs.ir() == rhs.ir();
}

inline bool
operator != (const IndexLabel& lhs, const IndexLabel& rhs) {
  return !(lhs == rhs);
}

inline bool
operator <= (const IndexLabel& lhs, const IndexLabel& rhs) {
  return lhs.label() <= rhs.label()
      && lhs.ir() <= rhs.ir();
}

inline bool
operator < (const IndexLabel& lhs, const IndexLabel& rhs) {
  return (lhs <= rhs) && (lhs != rhs);
}


}  // namespace tammy


#endif  // TAMMY_INDEX_SPACE_H_

