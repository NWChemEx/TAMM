#ifndef TAMMX_TENSOR_LABELS_H_
#define TAMMX_TENSOR_LABELS_H_

#include "tammx/types.h"
#include "tammx/tce.h"

namespace tammx {
namespace tensor_labels {

struct OLabel : public IndexLabel {
  OLabel(int n)
      : IndexLabel{n, RangeType{DimType::o}} {}
};

struct VLabel : public IndexLabel {
  VLabel(int n)
      : IndexLabel{n, RangeType{DimType::v}} {}
};

struct NLabel : public IndexLabel {
  NLabel(int n)
      : IndexLabel{n, RangeType{DimType::n}} {}
};

const OLabel h1{1}, h2{2}, h3{3}, h4{4}, h5{5}, h6{6}, h7{7}, h8{8}, h9{9}, h10{10}, h11{11};
const VLabel p1{1}, p2{2}, p3{3}, p4{4}, p5{5}, p6{6}, p7{7}, p8{8}, p9{9}, p10{10}, p11{11};

const OLabel i{0}, j{1};
const VLabel a{0}, b{1};

} // namespace tensor_labels
}  // namespace tammx

#endif // TAMMX_TENSOR_LABELS_H_
