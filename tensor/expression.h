#ifndef __ctce_expression_h__
#define __ctce_expression_h__

#include "typesf2c.h"
#include "index.h"
#include <vector>
#include <set>
#include <cassert>
#include <iostream>
#include "tensor.h"
#include "triangular.h"
#include "iterGroup.h"

namespace ctce {

/**
 * Assigment template. tC += coef * tA
 */
class Assignment {
public:
  /**
   * Constructor
   */
  Assignment();

  /**
   * Destructor
   */
  ~Assignment();

  /**
   * Constructor. Assign tC += coef * tA
   * @param[in] tC left hand side tensor
   * @param[in] tA right hand side tensor
   * @param[in] coef coefficient. most of time it is 1 or -1.
   */
  Assignment(Tensor& tC, Tensor& tA, double coef,
             const std::vector<IndexName>& cids,
             const std::vector<IndexName>& aids);

  Assignment(Tensor *tC, Tensor *tA, double coef,
             const std::vector<IndexName>& cids,
             const std::vector<IndexName>& aids);

  Assignment& operator = (const Assignment& as);
  /**
   * Get lhs tensor tC
   */
  Tensor& tC();
  const Tensor& tC() const;

  /**
   * Get rhs tensor tA
   */
  Tensor& tA();
  const Tensor& tA() const;

  const std::vector<IndexName> &cids() const;

  const std::vector<IndexName> &aids() const;

  /**
   * Get coefficient
   */
  double coef();

  /**
   * Get outer loop iterator
   */
  IterGroup<triangular>& out_itr();
  void execute(int sync_ga=0, int spos=0);

private:
  Tensor *tC_; /*< lhs tensor */
  Tensor *tA_; /*< rhs tensor */
  Tensor tC_bug, tA_bug; /*@FIXME: @BUG: to keep things working for now*/
  double coef_; /*< coefficient */
  IterGroup<triangular> out_itr_; /*< outer loop iterator */
  std::vector<IndexName> cids_;
  std::vector<IndexName> aids_;
  void init();
};

/**
 * Multiplication template. tC += coef * tA * tB;
 */
class Multiplication {
public: /* @BUG: @FIXME: should be made private*/
  std::vector<IndexName> a_mem_pos;
  std::vector<IndexName> b_mem_pos;
  std::vector<IndexName> c_mem_pos;
  std::vector<Index> a_ids;
  std::vector<Index> b_ids;
  std::vector<Index> c_ids;

public:

  /**
   * Constructor
   */
  Multiplication();

  /**
   * Destructor
   */
  ~Multiplication();

  /**
   * Constructor. Assign tC += coef * tA * tB.
   * @param[in] tC left hand side tensor
   * @param[in] tA right hand side tensor 1
   * @param[in] tB right hand side tensor 2
   * @param[in] coef coefficient
   */
  Multiplication(Tensor& tC1, Tensor& tA1, Tensor& tB1, double coef);

  Multiplication(Tensor *tC1, Tensor *tA1, Tensor *tB1, double coef);

  Multiplication(Tensor& tC1, const std::vector<IndexName> &c_ids1,
                 Tensor& tA1, const std::vector<IndexName> &a_ids1,
                 Tensor& tB1, const std::vector<IndexName> &b_ids1,
                 double coef);

  Multiplication(Tensor *tC1, const std::vector<IndexName> &c_ids1,
                 Tensor *tA1, const std::vector<IndexName> &a_ids1,
                 Tensor *tB1, const std::vector<IndexName> &b_ids1,
                 double coef);

  Multiplication& operator=(const Multiplication &m);

  /**
   * Get left hand side tensor tC
   */
  Tensor& tC();
  const Tensor& tC() const;

  /**
   * Get right hand side tensor tA
   */
  Tensor& tA();
  const Tensor& tA() const;

  /**
   * Get right hand side tensor tB
   */
  Tensor& tB();
  const Tensor& tB() const;

  /**
   * Get coefficient
   */
  const double coef();

  /**
   * Get summation indices
   */
  std::vector<IndexName>& sum_ids();

  /**
   * Get outer loop iterator
   */
  IterGroup<triangular>& out_itr();

  /**
   * Get summation indices iterator
   */
  IterGroup<triangular>& sum_itr();

  /**
   * Get copy iterator
   */
  IterGroup<CopyIter>& cp_itr();

  /**
   * Manually set out_itr with given symmetry group
   */
  void setOutItr(const std::vector<int>& gp);

  /**
   * Manually set sum_itr with given symmetry group
   */
  void setSumItr(const std::vector<int>& gp);

  /**
   * Manually set cp_itr with given copy group
   */
  void setCopyItr(const std::vector<int>& gp);
  void execute(int sync_ga=0, int spos=0);

private:
  Tensor *tC_; /*< left hand side tensor */
  Tensor *tA_; /*< right hand side tensor 1 */
  Tensor *tB_; /*< right hand side tensor 2 */
  Tensor tC_bug, tA_bug, tB_bug; /*@FIXME: @BUG: for now to keep everything working. to be removed when all implementations use this interface*/
  double coef_; /*< coefficient */

  std::vector<IndexName> sum_ids_; /*< summation indices of the contraction */
  IterGroup<triangular> out_itr_; /*< outer loop iterator */
  IterGroup<triangular> sum_itr_; /*< summation loop iterator */
  IterGroup<CopyIter> cp_itr_; /*< copy iterator, use to do tC add hash block */

  void genMemPos(); /*< generate memory position according to the indices order of the expression */
  void genTrigItr(IterGroup<triangular>& itr, const std::vector<int>& gp,
                  const std::vector<IndexName>& name); /*< generate triangular loops given IndexName and symmetry group */
  void genSumGroup(); /*< generate summation symmetry group and set sum_itr */
  void genCopyGroup(); /*< genertate copy group and set cp_itr */
  void genOutGroup(); /*< generate outer loop group and set out_itr */
};


inline std::vector<RangeType> id2range(const vector<IndexName> &ids) {
  std::vector<RangeType> retv(ids.size());
  for(int i=0; i<ids.size(); i++) {
    if(ids[i] < pIndexNum) {
      retv[i] = TO;
    }
    else {
      retv[i] = TV;
    }
  }
  return retv;
}

inline std::vector<int> ext_sym_group(const Tensor &tensor,
                                      const vector<IndexName> &ids) {
  int nupper = tensor.nupper();
  int ndim = tensor.dim();
  assert(ndim == ids.size());
  std::vector<RangeType> range_types = id2range(ids);
  int esgc=0;
  std::vector<int> retv(ndim);
  {
    std::vector<int> esg(RANGE_UB,-1);
    for(int i=0; i<nupper; i++) {
      if(esg[range_types[i]] != -1) {
        retv[i] = esg[range_types[i]];
      }
      else {
        retv[i] = esg[range_types[i]] = esgc++;
      }
    }
  }
  {
    std::vector<int> esg(RANGE_UB,-1);
    for(int i=nupper; i<ndim; i++) {
      if(esg[range_types[i]] != -1) {
        retv[i] = esg[range_types[i]];
      }
      else {
        retv[i] = esg[range_types[i]] = esgc++;
      }
    }
  }
  return retv;
}

inline std::vector<Index> name2ids(const Tensor &tensor,
                                   const vector<IndexName>& name) {
  int n = tensor.dim();
  assert(n == name.size());
  const std::vector<int> &esg = ext_sym_group(tensor, name);
  std::vector<Index> retv(n);

  for(int i=0; i<n; i++) {
    retv[i] = Index(name[i], esg[i]);
  }
  return retv;
}

inline bool is_permutation(const std::vector<IndexName>& ids) {
  std::set<IndexName> sids(ids.begin(), ids.end());
  return sids.size() == ids.size();
}

inline bool is_permutation(const std::vector<IndexName>& ids1,
                           const std::vector<IndexName>& ids2) {
  std::set<IndexName> sids1(ids1.begin(), ids1.end());
  std::set<IndexName> sids2(ids2.begin(), ids2.end());
  if(ids1.size() != sids1.size()) return false;
  if(ids2.size() != sids2.size()) return false;
  for (int i=0; i<ids1.size(); i++) {
    if(sids2.find(ids1[i]) == sids2.end())
      return false;
  }
  return true;
}

inline std::vector<IndexName> ivec(IndexName i1) {
  std::vector<IndexName> ret;
  ret.push_back(i1);
  return ret;
}

inline std::vector<IndexName> ivec(IndexName i1, IndexName i2) {
  std::vector<IndexName> ret = ivec(i1);
  ret.push_back(i2);
  return ret;
}

inline std::vector<IndexName> ivec(IndexName i1, IndexName i2, IndexName i3) {
  std::vector<IndexName> ret = ivec(i1,i2);
  ret.push_back(i3);
  return ret;
}

inline std::vector<IndexName> ivec(IndexName i1, IndexName i2, IndexName i3,
                                   IndexName i4) {
  std::vector<IndexName> ret = ivec(i1,i2,i3);
  ret.push_back(i4);
  return ret;
}

inline std::vector<size_t> sort_ids(const std::vector<IndexName> &name, const std::vector<IndexName> &mem_pos_) {
  assert(name.size() == mem_pos_.size());
  std::vector<size_t> sort_ids_(name.size());
  for (int i=0; i<name.size(); i++) {
    sort_ids_[i] = std::find(name.begin(), name.end(), mem_pos_[i]) - name.begin() + 1;
  }
  return sort_ids_;
}

inline std::vector<size_t> mult_perm(const std::vector<IndexName> &name, const std::vector<IndexName> &mem_pos_) {
  assert(name.size() == mem_pos_.size());
  vector<size_t> lperm(name.size());
  for (int i=0; i<name.size(); i++) {
    lperm[i] = std::find(mem_pos_.begin(), mem_pos_.end(), name[i]) - mem_pos_.begin() + 1;
  }
  return lperm;
}

inline std::vector<size_t> getMemPosVal(const std::vector<Index> &ids_,
                                        const std::vector<IndexName> &mem_pos_) {
  assert(ids_.size() == mem_pos_.size());
  const int n = ids_.size();
  std::vector<size_t> sort_ids_v_(n);
  std::vector<int> tab_(IndexNum, -1);
  for(int i=0; i<n; i++) {
    tab_[ids_[i].name()] = i;
  }
  for (int i=0; i<n; i++) {
    int pos = tab_[mem_pos_[i]];
    sort_ids_v_[i]=ids_[pos].value();
  }
  return sort_ids_v_;
}

inline void setValue(std::vector<Index> &ids_,
                     const std::vector<size_t>& val) {
  assert(ids_.size()==val.size());
  for (int i=0; i<ids_.size(); i++) {
    ids_[i].setValue(val[i]);
  }
}

/**
 * Set the restricted value of indices
 * @param val restricted value as a vector of Integer
 */
inline void setValueR(std::vector<Index> &ids_,
                      const std::vector<size_t>& val) {
  assert(ids_.size()==val.size());
  for (int i=0; i<ids_.size(); i++) {
    ids_[i].setValueR(val[i]);
  }
}

inline const std::vector<int> ext_sym_group(const std::vector<Index> &ids_) {
  int dim_ = ids_.size();
  std::vector<int> esg(dim_);
  for(int i=0; i<dim_; i++) {
    esg[i] = ids_[i].ext_sym_group();
  }
  return esg;
}

inline std::vector<IndexName> id2name(const std::vector<Index> &ids_) {
  int dim_ = ids_.size();
  std::vector<IndexName> n(dim_);
  for(int i=0; i<dim_; i++) {
    n[i] = ids_[i].name();
  }
  return n;
}

inline void id2name(const std::vector<Index>& ids_, std::vector<IndexName> &n)  {
  int dim_ = ids_.size();
  n.resize(dim_);
  for(int i=0; i<dim_; i++) {
    n[i] = ids_[i].name();
  }
}

inline int sortByValueThenExtSymGroup(const std::vector<Index> &ids_,
                                      std::vector<IndexName> &name,
                                      std::vector<size_t> &pvalue,
                                      std::vector<size_t> &pvalue_r) {
  std::vector<int> tab_(IndexNum, -1);
  for(int i=0; i<ids_.size(); i++) {
    tab_[ids_[i].name()] = i;
  }
  int n = ids_.size();
  std::vector<Index> _ids_ = ids_;
  std::sort(_ids_.begin(),_ids_.end(),compareValue);
  std::sort(_ids_.begin(),_ids_.end(),compareExtSymGroup);
  std::vector<int> pos1(n), pos2(n);
  for (int i=0; i<n; i++) {
    pos1[i] = i;
    pos2[i]=tab_[_ids_[i].name()];
  }
  int sign = countParitySign<int>(pos1,pos2);
  pvalue_r.resize(n);
  pvalue.resize(n);
  name.resize(n);
  for (int i=0; i<n; i++) {
    name[i] = _ids_[i].name();
    pvalue[i] = _ids_[i].value();
    pvalue_r[i] = _ids_[i].value_r();
  }
  return sign;
}

inline void orderIds(const std::vector<Index> & ids_,
                     const std::vector<size_t>& order,
                     std::vector<IndexName>& name,
                     std::vector<size_t>& value,
                     std::vector<size_t>& value_r) {
  int n = ids_.size();
  vector<Index> _ids_(ids_.size());
  for (int i=0; i<n; i++) {
    assert(order[i]>=0 && order[i]<n);
    _ids_[i]=ids_[order[i]];
  }
  name.resize(n);
  value.resize(n);
  value_r.resize(n);
  for (int i=0; i<n; i++) {
    name[i] = _ids_[i].name();
    value[i] = _ids_[i].value();
    value_r[i] = _ids_[i].value_r();
  }
  for(int i=0; i<n; i++) {
    assert(Table::rangeOf(name[i]) == Table::rangeOf(ids_[i].name()));
  }
}

Assignment::Assignment() {}

Assignment::~Assignment() {}

Assignment::Assignment(Tensor& tC, Tensor& tA, double coef,
                       const std::vector<IndexName>& cids,
                       const std::vector<IndexName>& aids)
  : tC_bug(tC), tA_bug(tA), coef_(coef), cids_(cids), aids_(aids) {
  tC_ = &tC_bug;
  tA_ = &tA_bug;
  init();
  assert(is_permutation(cids_));
  assert(is_permutation(aids_));
  assert(is_permutation(cids_, aids_));
}

Assignment::Assignment(Tensor *tC, Tensor *tA, double coef,
                       const std::vector<IndexName>& cids,
                       const std::vector<IndexName>& aids)
  : tC_(tC), tA_(tA), coef_(coef), cids_(cids), aids_(aids) {
  init();
  assert(is_permutation(cids_));
  assert(is_permutation(aids_));
  assert(is_permutation(cids_, aids_));
}

Assignment&
Assignment::operator = (const Assignment& as) {
  tC_bug = as.tC();
  tA_bug = as.tA();
  coef_ = as.coef_;
  out_itr_ = as.out_itr_;
  aids_ = as.aids_;
  cids_ = as.cids_;
  if(as.tC_ == &as.tC_bug) {
    tC_ = &tC_bug;
    tA_ = &tA_bug;
  }
  else {
    tC_ = as.tC_;
    tA_ = as.tA_;
  }
}

Tensor&
Assignment::tC() { return *tC_; }

const Tensor&
Assignment::tC() const { return *tC_; }

Tensor&
Assignment::tA() { return *tA_; }

const Tensor&
Assignment::tA() const { return *tA_; }

const std::vector<IndexName> &
Assignment::cids() const { return cids_; }

const std::vector<IndexName> &
Assignment::aids() const { return aids_; }

double
Assignment::coef() { return coef_; }

IterGroup<triangular>&
Assignment::out_itr() { return out_itr_; }

Multiplication::Multiplication() {}

Multiplication::~Multiplication() {}

Multiplication::Multiplication(Tensor& tC1, Tensor& tA1, Tensor& tB1, double coef)
  : tC_bug(tC1), tA_bug(tA1), tB_bug(tB1), coef_(coef) {
  tC_ = &tC_bug;
  tA_ = &tA_bug;
  tB_ = &tB_bug;
  c_ids = tC().ids();
  a_ids = tA().ids();
  b_ids = tB().ids();
  genMemPos();
  genSumGroup();
  genOutGroup();
}

Multiplication::Multiplication(Tensor *tC1, Tensor *tA1, Tensor *tB1, double coef)
  : tC_(tC1), tA_(tA1), tB_(tB1), coef_(coef) {
  c_ids = tC().ids();
  a_ids = tA().ids();
  b_ids = tB().ids();
  genMemPos();
  genSumGroup();
  genOutGroup();
}


Multiplication::Multiplication(Tensor& tC1, const std::vector<IndexName> &c_ids1,
                               Tensor& tA1, const std::vector<IndexName> &a_ids1,
                               Tensor& tB1, const std::vector<IndexName> &b_ids1,
                               double coef)
  : tC_bug(tC1), tA_bug(tA1), tB_bug(tB1), coef_(coef) {
  tC_ = &tC_bug;
  tA_ = &tA_bug;
  tB_ = &tB_bug;
  c_ids = name2ids(tC(), c_ids1);
  a_ids = name2ids(tA(), a_ids1);
  b_ids = name2ids(tB(), b_ids1);
  genMemPos();
  genSumGroup();
  genOutGroup();
}

Multiplication::Multiplication(Tensor *tC1, const std::vector<IndexName> &c_ids1,
                               Tensor *tA1, const std::vector<IndexName> &a_ids1,
                               Tensor *tB1, const std::vector<IndexName> &b_ids1,
                               double coef)
  : tC_(tC1), tA_(tA1), tB_(tB1), coef_(coef) {
  c_ids = name2ids(tC(), c_ids1);
  a_ids = name2ids(tA(), a_ids1);
  b_ids = name2ids(tB(), b_ids1);
  genMemPos();
  genSumGroup();
  genOutGroup();
}

Multiplication&
Multiplication::operator=(const Multiplication &m) {
  tC_bug = m.tC();
  tA_bug = m.tA();
  tB_bug = m.tB();
  coef_ = m.coef_;

  sum_ids_ = m.sum_ids_;
  out_itr_ = m.out_itr_;
  sum_itr_ = m.sum_itr_;
  cp_itr_  = m.cp_itr_ ;

  a_mem_pos = m.a_mem_pos;
  b_mem_pos = m.b_mem_pos;
  c_mem_pos = m.c_mem_pos;
  a_ids = m.a_ids;
  b_ids = m.b_ids;
  c_ids = m.c_ids;

  if(m.tC_ == &m.tC_bug) {
    tA_ = &tA_bug;
    tB_ = &tB_bug;
    tC_ = &tC_bug;
  }
  else {
    tA_ = m.tA_;
    tB_ = m.tB_;
    tC_ = m.tC_;
  }
}

Tensor&
Multiplication::tC() { return *tC_; }

const Tensor&
Multiplication::tC() const { return *tC_; }

Tensor&
Multiplication::tA() { return *tA_; }

const Tensor&
Multiplication::tA() const { return *tA_; }

Tensor&
Multiplication::tB() { return *tB_; }

const Tensor&
Multiplication::tB() const { return *tB_; }

const double
Multiplication::coef() { return coef_; }

std::vector<IndexName>&
Multiplication::sum_ids() { return sum_ids_; }

IterGroup<triangular>&
Multiplication::out_itr() { return out_itr_; }

IterGroup<triangular>&
Multiplication::sum_itr() { return sum_itr_; }


IterGroup<CopyIter>&
Multiplication::cp_itr() { return cp_itr_; }

} /* namespace ctce*/

#endif /* __ctce_expression_h__ */
