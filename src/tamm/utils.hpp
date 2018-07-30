#if !defined(TAMM_UTILS_H_)
#define TAMM_UTILS_H_

#include "tamm/types.hpp"
#include "tamm/errors.hpp"
#include "tamm/tiled_index_space.hpp"

#include <vector>


namespace tamm {

namespace internal {

inline void update_fillin_map(std::map<std::string, Label>& str_to_labels,
                              const std::vector<bool>& str_map,
                              const std::vector<std::string>& str_labels,
                              int initial_off) {
    const size_t sz = str_labels.size();
    for(size_t i = 0; i < sz; i++) {
        if(str_map[i]) {
            str_to_labels[str_labels[i]] = -initial_off - i - 1;
        }
    }
}

template<typename LabelTensorT>
inline void fillin_tensor_label_from_map(
  LabelTensorT& ltensor, const std::map<std::string, Label>& str_to_labels) {
    IndexLabelVec new_labels = ltensor.labels();
    const size_t sz          = ltensor.labels().size();
    for(size_t i = 0; i < sz; i++) {
        if(ltensor.str_map()[i]) {
          EXPECTS(str_to_labels.find(ltensor.str_labels()[i]) != str_to_labels.end());
            new_labels[i] = ltensor.tensor().tiled_index_spaces()[i].label(
              str_to_labels.find(ltensor.str_labels()[i])->second);
        }
    }
    ltensor.set_labels(new_labels);
}

/**
 * @ingroup perm
 * @brief Compute permutation to be performed to permute vector @p from to vector @p to.
 * @param from Source vector for the permutation
 * @param to Target vector for the permutation
 * @pre @p from and @p to are permutations of each other
 * @pre from.size() == to.size()
 * @return Vector to permute @p from to @p to.
 * @post Return ret such that:
 * ensures 0<=i<from.size(): to[i] = from[ret[i]]
 */
template<typename T>
PermVector
perm_compute(const std::vector<T>& from, const std::vector<T>& to) {
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
    }
    return true;
}

template<typename T>
std::vector<T> unique_entries(const std::vector<T>& input_vec) {
    std::vector<T> ret;
#if 1
    for(const auto& val : input_vec) {
        auto it = std::find(ret.begin(), ret.end(), val);
        if(it == ret.end()) { ret.push_back(val); }
    }
#else
    ret = input_vec;
    std::sort(ret.begin(), ret.end());
    std::unique(ret.begin(), ret.end());
#endif
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
    for(size_t i=0; i<perm_map.size(); i++) {
        EXPECTS(perm_map[i] < input_vec.size());
        out_vec[i] = input_vec[perm_map[i]];
    }
}

inline IndexLabelVec sort_on_dependence(const IndexLabelVec& labels) {
    IndexLabelVec ret;
    for(const auto& lbl : labels) {
        for(const auto& dlbl : lbl.dep_labels()) {
            const auto it = std::find(ret.begin(), ret.end(), dlbl);
            if(it == ret.end()) { ret.push_back(dlbl); }
        }
        const auto it = std::find(ret.begin(), ret.end(), lbl);
        if(it == ret.end()) { ret.push_back(lbl); }
    }
    return ret;
}


template<typename T>
bool cartesian_iteration(std::vector<T>& itr, const std::vector<T>& end) {
    EXPECTS(itr.size() == end.size());
    // if(!std::lexicographical_compare(itr.begin(), itr.end(), end.begin(),
    //                                  end.end())) {
    //     return false;
    // }
    int i;
    for(i = -1 + itr.size(); i>=0 && itr[i]+1 == end[i]; i--) {
        itr[i] = T{0};        
    }
    // EXPECTS(itr.size() == 0 || i>=0);
    if(i>=0) {
        ++itr[i];
        return true;
    }
    return false;
}


} // namespace tamm::internal


} // namespace tamm

#endif // TAMM_UTILS_H_
