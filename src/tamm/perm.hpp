#pragma once

#include <vector>

#include "tamm/errors.hpp"
#include "tamm/types.hpp"

namespace tamm::internal {

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
  for(auto p: to) {
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
PermVector perm_map_compute(const std::vector<T>& unique_vec, const std::vector<T>& vec_required) {
  PermVector ret;
  for(const auto& val: vec_required) {
    auto it = std::find(unique_vec.begin(), unique_vec.end(), val);
    EXPECTS(it >= unique_vec.begin());
    EXPECTS(it != unique_vec.end());
    ret.push_back(it - unique_vec.begin());
  }
  return ret;
}

template<typename T, typename Integer>
std::vector<T> perm_map_apply(const std::vector<T>&       input_vec,
                              const std::vector<Integer>& perm_map) {
  std::vector<T> ret;
  for(const auto& pm: perm_map) {
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

template<typename T>
std::vector<T> unique_entries(const std::vector<T>& input_vec) {
  std::vector<T> ret;

  for(const auto& val: input_vec) {
    auto it = std::find(ret.begin(), ret.end(), val);
    if(it == ret.end()) { ret.push_back(val); }
  }
  return ret;
}

} // namespace tamm::internal
