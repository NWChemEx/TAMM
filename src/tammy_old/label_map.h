// Copyright 2016 Pacific Northwest National Laboratory

#ifndef TAMMY_UTIL_H_
#define TAMMY_UTIL_H_

#include <map>
#include "tammy/errors.h"
#include "tammy/types.h"
#include "tammy/boundvec.h"

namespace tammy {


/**
 * @ingroup utilities
 * @brief Efficient map from labels to templated values
 *
 * Keys are index labels. Values are of the templated type.
 * In principle, LabelMap<T> is equivalent to map<IndexLabel,T>.
 * But with finite index label space, LabelMap<> might be more
 * efficiently than a std::map.
 *
 * @tparam T type of each value in the map
 * @todo @bug Check if this works when a label already presented is
 * updated.
 * @todo Implement LabelMap uses a statically allocated array.
 */
template<typename T>
class LabelMap {
 public:
  /**
   * @brief Update the label map with a list of keys and corresponding values.
   * @param labels Vector of labels as keys
   * @param ids Vector of values, one per key in @p labels
   * @return Updated label map
   *
   * @pre labels.size() == ids.size()
   */
  LabelMap& update(const IndexLabelVec& labels,
                   const TensorVec<T>& ids) {
    // EXPECTS(labels.size() + labels_.size()  <= labels_.max_size());
    // labels_.insert_back(labels.begin(), labels.end());
    // ids_.insert_back(ids.begin(), ids.end());
    EXPECTS(labels.size() == ids.size());
    for(size_t i=0; i<labels.size(); i++) {
      lmap_[labels[i]] = ids[i];
    }
    return *this;
  }

  /**
   * Get the tensor vector of values corresponding to a tensor vector of index labels (as keys)
   * @param labels Vector of index labels
   * @return Vector of values corresponding to the vector of index labels
   */
  TensorVec<T> get_blockid(const IndexLabelVec& labels) const {
    TensorVec<T> ret;
    for(auto l: labels) {
      auto itr = lmap_.find(l);
      EXPECTS(itr != lmap_.end());
      ret.push_back(itr->second);
    }
    return ret;
    // TensorVec<T> ret(labels.size());
    // using size_type = IndexLabelVec::size_type;
    // for(size_type i=0; i<labels.size(); i++) {
    //   auto itr = std::find(begin(labels_), end(labels_), labels[i]);
    //   EXPECTS(itr != end(labels_));
    //   ret[i] = ids_[itr - begin(labels_)];
    // }
    // return ret;
  }

 private:
  // TensorRank rank_{};
  // IndexLabelVec labels_;
  // TensorVec<T> ids_;
  std::map<IndexLabel, T> lmap_;
};  // LabelMap


} //namespace tammy


#endif  // TAMMY_UTIL_H_

