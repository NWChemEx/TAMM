#ifndef TAMMX_OPENTRY_H_
#define TAMMX_OPENTRY_H_

namespace tammx {
template<typename T>
struct SetOpEntry {
  LabeledTensor lhs;
  T value;
};

template<typename T>
struct AddOpEntry {
  LabeledTensor lhs;
  T alpha, beta;
  LabeledTensor rhs;
};

template<typename T>
struct MultOpEntry {
  LabeledTensor lhs;
  T alpha, beta;
  LabeledTensor rhs1, rhs2;
};

template<typename Func, unsigned ndim, unsigned nrhs>
struct MapOpEntry {
  LabeledTensor lhs;
  std::vector<LabeledTensor> rhss;
  Func func;
};
}  // mamespace tammx

#endif  // TAMMX_OPENTRY_H_

