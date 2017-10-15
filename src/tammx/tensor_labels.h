#ifndef TAMMX_TENSOR_LABELS_H_
#define TAMMX_TENSOR_LABELS_H_

#include "tammx/types.h"
#include "tammx/tce.h"

/**
 * @defgroup tensor_labels
 * @brief Convenience objects to label tensor indices
 */

namespace tammx {
namespace tensor_labels {

/**
 * @ingroup tensor_labels
 * Label for an occupied index
 */
struct OLabel : public IndexLabel {
  OLabel(int n)
      : IndexLabel{n, RangeType{DimType::o}} {}
};

/**
 * @ingroup tensor_labels
 * Label for an virtual index
 */
struct VLabel : public IndexLabel {
  VLabel(int n)
      : IndexLabel{n, RangeType{DimType::v}} {}
};

/**
 * @ingroup tensor_labels
 * Label for an N (occupied + virtual) index
 */
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
