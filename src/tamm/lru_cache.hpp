#ifndef TAMM_LRU_CACHE_HPP_
#define TAMM_LRU_CACHE_HPP_

#include <map>
#include <iostream>
#include <vector>
#include <algorithm>

namespace tamm {

template <typename KeyEl, typename Value>
class LRUCache {
 public:
 using Key = std::vector<KeyEl>;
  LRUCache(uint32_t max_size) : max_size_{max_size} { }

  void reset_clock() { cycle_ = 0; }

  uint32_t max_size() const { return max_size_; }

  std::pair<bool, Value&> log_access(const Key& key) {
    bool hit = false;
    if (max_size_ == 0) {
      clear(); // "hack" to disable cache
    }
    if (auto it = cache_.find(key); it != cache_.end()) {
      // hit
      hit = true;
      auto c2k_it = cycle_to_key_.find(it->second);
      auto dist = std::distance(c2k_it, cycle_to_key_.end());
      ++reuse_distance_histogram_[dist-1];
      cycle_to_key_.erase(c2k_it);
      it->second = cycle_;
      cycle_to_key_[cycle_] = key;
    } else {  // miss
      hit = false;
      ++reuse_distance_histogram_[max_size_];
      if (cache_.size() == max_size_ && max_size_ > 0) {   // eviction
        auto it = cycle_to_key_.begin();  // oldest entry
        cache_.erase(it->second);
        cycle_to_key_.erase(it);
      }
      cycle_to_key_[cycle_] = key;
      cache_[key] = cycle_;
    }
    ++cycle_;
    return {hit, cached_value_[key]};
  }

  Value& access(const Key& key) {
    EXPECTS(cached_value_.find(key) != cached_value_.end());
    return *cached_value_.find(key);
  }

  const Value& access(const Key& key) const {
    EXPECTS(cached_value_.find(key) != cached_value_.end());
    return *cached_value_.find(key);
  }

  void gather_stats(std::vector<uint32_t>& vec) {
    for (uint32_t i = 0; i <= max_size_; i++) {
      vec.push_back(reuse_distance_histogram_[i]);
    }
  }

  void clear() {
    cache_.clear();
    cycle_to_key_.clear();
    cached_value_.clear();
  }

 private:
  template <typename T>
  struct KeyComp {
    bool operator()(const std::vector<T>& v1, const std::vector<T>& v2) const {
      return std::lexicographical_compare(v1.begin(), v1.end(), v2.begin(),
                                          v2.end());
    }
  };
  uint32_t max_size_;
  uint32_t cycle_;
  std::map<Key, uint32_t, KeyComp<KeyEl>> cache_;
  std::map<uint32_t, Key> cycle_to_key_;
  std::map<Key, Value> cached_value_;
  std::map<uint32_t, uint32_t> reuse_distance_histogram_;
};  // class LRUCache
}  // namespace tamm

#endif  // TAMM_LRU_CACHE_HPP_
