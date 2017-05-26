#ifndef TAMMX_LABELEDBLOCK_H_
#define TAMMX_LABELEDBLOCK_H_

#include "tammx/types.h"

namespace tammx {

class Block;

struct LabeledBlock {
  Block *block;
  TensorLabel label_;

  template<typename T>
  void operator = (T value);

  void operator = (LabeledBlock rhs);

  template<typename T>
  void operator = (std::tuple<T, LabeledBlock> rhs);

  template<typename T>
  void operator = (std::tuple<T, LabeledBlock, LabeledBlock> rhs);
  
  void operator = (std::tuple<LabeledBlock, LabeledBlock> rhs);

  void operator += (LabeledBlock rhs);

  template<typename T>
  void operator += (std::tuple<T, LabeledBlock> rhs);

  template<typename T>
  void operator = (std::tuple<T, LabeledBlock, LabeledBlock> rhs);
  
  void operator = (std::tuple<LabeledBlock, LabeledBlock> rhs);  
};

} // namespace tammx

#endif  // TAMMX_LABELEDBLOCK_H_

