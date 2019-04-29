#ifndef TAMM_UTILS_HPP_
#define TAMM_UTILS_HPP_

namespace tamm {

namespace internal {

inline void update_fillin_map(std::map<std::string, Label>& str_to_labels,
                              const std::vector<bool>& str_map,
                              const std::vector<std::string>& str_labels,
                              int initial_off) {
    const size_t sz = str_labels.size();
    for(size_t i = 0; i < sz; i++) {
        if(str_map[i]) { str_to_labels[str_labels[i]] = -initial_off - i - 1; }
    }
}

template<typename LabelTensorT>
inline void fillin_tensor_label_from_map(
  LabelTensorT& ltensor, const std::map<std::string, Label>& str_to_labels) {
    IndexLabelVec new_labels = ltensor.labels();
    const size_t sz          = ltensor.labels().size();
    for(size_t i = 0; i < sz; i++) {
        if(ltensor.str_map()[i]) {
            EXPECTS(str_to_labels.find(ltensor.str_labels()[i]) !=
                    str_to_labels.end());
            new_labels[i] = ltensor.tensor().tiled_index_spaces()[i].label(
              str_to_labels.find(ltensor.str_labels()[i])->second);
        }
    }
    ltensor.set_labels(new_labels);
}

/**
 * @ingroup perm
 * @brief Compute permutation to be performed to permute vector @p from to
 * vector @p to.
 * @param from Source vector for the permutation
 * @param to Target vector for the permutation
 * @pre @p from and @p to are permutations of each other
 * @pre from.size() == to.size()
 * @return Vector to permute @p from to @p to.
 * @post Return ret such that:
 * ensures 0<=i<from.size(): to[i] = from[ret[i]]
 */
template<typename T>
PermVector perm_compute(const std::vector<T>& from, const std::vector<T>& to) {
    PermVector layout;

    EXPECTS(from.size() == to.size());
    for(auto p : to) {
        auto itr = std::find(from.begin(), from.end(), p);
        EXPECTS(itr != from.end());
        layout.push_back(itr - from.begin());
    }
    return layout;
}

template<typename T>
bool are_permutations(const std::vector<T>& vec1, const std::vector<T>& vec2) {
    if(vec1.size() != vec2.size()) { return false; }
    std::vector<bool> taken(vec1.size(), false);
    for(size_t i = 0; i < vec1.size(); i++) {
        auto it = std::find(vec2.begin(), vec2.end(), vec1[i]);
        if(it == vec2.end()) { return false; }
        if(taken[std::distance(vec2.begin(), it)] == true) { return false; }
        taken[std::distance(vec2.begin(), it)] = true;
    }
    return true;
}

template<typename T>
std::vector<T> unique_entries(const std::vector<T>& input_vec) {
    std::vector<T> ret;

    for(const auto& val : input_vec) {
        auto it = std::find(ret.begin(), ret.end(), val);
        if(it == ret.end()) { ret.push_back(val); }
    }
    return ret;
}

template<typename T>
std::vector<size_t> perm_map_compute(const std::vector<T>& unique_vec,
                                     const std::vector<T>& vec_required) {
    std::vector<size_t> ret;
    for(const auto& val : vec_required) {
        auto it = std::find(unique_vec.begin(), unique_vec.end(), val);
        EXPECTS(it >= unique_vec.begin());
        EXPECTS(it != unique_vec.end());
        ret.push_back(it - unique_vec.begin());
    }
    return ret;
}

template<typename T, typename Integer>
std::vector<T> perm_map_apply(const std::vector<T>& input_vec,
                              const std::vector<Integer>& perm_map) {
    std::vector<T> ret;
    for(const auto& pm : perm_map) {
        EXPECTS(pm < input_vec.size());
        ret.push_back(input_vec[pm]);
    }
    return ret;
}

template<typename T, typename Integer>
void perm_map_apply(std::vector<T>& out_vec, const std::vector<T>& input_vec,
                    const std::vector<Integer>& perm_map) {
    out_vec.resize(perm_map.size());
    for(size_t i = 0; i < perm_map.size(); i++) {
        EXPECTS(perm_map[i] < input_vec.size());
        out_vec[i] = input_vec[perm_map[i]];
    }
}

inline IndexLabelVec sort_on_dependence(const IndexLabelVec& labels) {
    std::vector<TileLabelElement> primary_labels;
    std::vector<size_t> sorted_order;
    for(const auto& lbl : labels) {
        primary_labels.push_back(lbl.primary_label());
    }
    for(size_t i = 0; i < labels.size(); i++) {
        const auto& lbl = labels[i];
        for(const auto& slbl : lbl.secondary_labels()) {
            const auto it =
              std::find(primary_labels.begin(), primary_labels.end(), slbl);
            EXPECTS(it != primary_labels.end());
            const auto sit = std::find(sorted_order.begin(), sorted_order.end(),
                                       it - primary_labels.begin());
            if(sit == sorted_order.end()) {
                sorted_order.push_back(it - primary_labels.begin());
            }
        }
        const auto it = std::find(sorted_order.begin(), sorted_order.end(), i);
        if(it == sorted_order.end()) { sorted_order.push_back(i); }
    }
    IndexLabelVec ret;
    for(const auto& pos : sorted_order) { ret.push_back(labels[pos]); }
    return ret;
}

template<typename T>
bool cartesian_iteration(std::vector<T>& itr, const std::vector<T>& end) {
    EXPECTS(itr.size() == end.size());

    int i;
    for(i = -1 + itr.size(); i >= 0 && itr[i] + 1 == end[i]; i--) {
        itr[i] = T{0};
    }

    if(i >= 0) {
        ++itr[i];
        return true;
    }
    return false;
}

inline IndexVector indep_values(
  const IndexVector& blockid, const Index& idx,
  const std::map<size_t, std::vector<size_t>>& dep_map) {
    IndexVector ret{};
    if(dep_map.find(idx) != dep_map.end()) {
        for(const auto& dep_id : dep_map.at(idx)) {
            ret.push_back(blockid[dep_id]);
        }
    }
    return ret;
}

template<typename LabeledTensorT>
IndexVector translate_blockid(const IndexVector& blockid,
                              const LabeledTensorT& ltensor) {
    EXPECTS(blockid.size() == ltensor.labels().size());
    const auto& tensor  = ltensor.tensor();
    const auto& dep_map = tensor.dep_map();
    EXPECTS(blockid.size() == tensor.num_modes());
    IndexVector translate_blockid;
    for(size_t i = 0; i < blockid.size(); i++) {
        auto indep_vals = indep_values(blockid, i, dep_map);
        const auto& label_tis =
          ltensor.labels()[i].tiled_index_space()(indep_vals);
        const auto& tensor_tis = tensor.tiled_index_spaces()[i](indep_vals);
        Index val              = label_tis.translate(blockid[i], tensor_tis);
        translate_blockid.push_back(val);
    }
    return translate_blockid;
}

template<typename Iter>
inline std::string
join(Iter begin, Iter end, const std::string& sep) {
    std::ostringstream oss;
    if (begin != end) {
        oss << *begin;
        ++begin;
    }
    while (begin != end) {
        oss << sep << *begin;
        begin++;
    }
    return oss.str();
}

template<typename Container>
inline std::string
join(const Container& c, const std::string& sep) {
  return join(c.begin(), c.end(), sep);
}

// inline std::string
// talsh_mult_op_string(const IndexLabelVec& clabel,
//         const IndexLabelVec& alabel,
//         const IndexLabelVec& blabel)
inline std::string
talsh_mult_op_string(const std::vector<IntLabel>& clabel,
        const std::vector<IntLabel>& alabel,
        const std::vector<IntLabel>& blabel) {
  std::vector<char> c_label;
  std::vector<char> a_label;
  std::vector<char> b_label;
  const std::string sep = ",";

  char talsh_index_base = 'a';
  int label_count = 0;
  std::map<IntLabel, char> imap;
  for (auto &l : clabel) {
      imap[l] = talsh_index_base + label_count;
      label_count += 1;
  }
  for (auto &l : alabel) {
      if (imap.find(l) == imap.end()) {
          imap[l] = talsh_index_base + label_count;
          label_count += 1;
      }
  }

  for (auto &l : clabel) {
      c_label.push_back(imap[l]);
  }
  for (auto &l : alabel) {
      a_label.push_back(imap[l]);
  }
  for (auto &l : blabel) {
      b_label.push_back(imap[l]);
  }

  std::reverse(a_label.begin(),a_label.end());
  std::reverse(b_label.begin(),b_label.end());
  std::reverse(c_label.begin(),c_label.end());

  std::ostringstream oss;
  if(c_label.size() == 0) {
    /// inner-product leading to a scalar
    oss << "C()+="
            << "A(" <<join(a_label, ",") << ")"
            << "*B(" << join(b_label, ",") << ")";
  } else {
    /// normal mult operation
    oss << "C(" << join(c_label, ",") << ")+="
            << "A(" <<join(a_label, ",") << ")"
            << "*B(" << join(b_label, ",") << ")";
  }
  return oss.str();
}

inline std::string
talsh_add_op_string(const IndexLabelVec& clabel,
        const IndexLabelVec& alabel) {
  std::vector<char> c_label;
  std::vector<char> a_label;
  const std::string sep = ",";

  char talsh_index_base = 'a';
  int label_count = 0;
  std::map<TiledIndexLabel, char> imap;
  for (auto &l : clabel) {
      imap[l] = talsh_index_base + label_count;
      label_count += 1;
  }
  for (auto &l : alabel) {
      if (imap.find(l) == imap.end()) {
          imap[l] = talsh_index_base + label_count;
          label_count += 1;
      }
  }

  for (auto &l : clabel) {
      c_label.push_back(imap[l]);
  }
  for (auto &l : alabel) {
      a_label.push_back(imap[l]);
  }

  std::ostringstream oss;
  if(c_label.size() == 0) {
    /// scalar addition
    oss << "C()+="
            << "A(" <<join(a_label, ",") << ")"; 
  } else {
    /// normal add operation
    oss << "C(" << join(c_label, ",") << ")+="
            << "A(" <<join(a_label, ",") << ")"; 
  }
  return oss.str();
}

} // namespace internal

} // namespace tamm

#endif // TAMM_UTILS_HPP_
