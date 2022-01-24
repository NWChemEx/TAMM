#pragma once

#include <vector>

namespace tamm::internal {

template <typename T>
bool cartesian_iteration(std::vector<T>& itr, const std::vector<T>& end) {
  EXPECTS(itr.size() == end.size());

  int i;
  for (i = -1 + itr.size(); i >= 0 && itr[i] + 1 == end[i]; i--) {
    itr[i] = T{0};
  }

  if (i >= 0) {
    ++itr[i];
    return true;
  }
  return false;
}

}  // namespace tamm::internal


