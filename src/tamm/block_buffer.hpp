#ifndef TAMM_BLOCK_BUFFER_HPP_
#define TAMM_BLOCK_BUFFER_HPP_

#include "tamm/types.hpp"
#include "tamm/tensor.hpp"
#include <iterator>

namespace tamm {

class RuntimeEngine;

/**
 * @brief The class used to pass block buffers to user functions
 *
 * @tparam T 
 */
template<typename T>
class BlockBuffer {
public:
  BlockBuffer() = default;
  BlockBuffer(span<T> buf_span, IndexedTensor<T> indexedTensor, RuntimeEngine* re, bool allocated = false) 
    : buf_span{buf_span}, allocated{allocated}, indexedTensor{indexedTensor}, re{re} {}
  BlockBuffer(const BlockBuffer& block_buffer) 
    : indexedTensor{block_buffer.indexedTensor}, re{block_buffer.re}
    {
      if (allocated) delete[] buf_span.data();
      allocated = true;
      const auto size = block_buffer.buf_span.size();
      T* buffer = new T[size]; // that will need to be more complicated once we get device buffers
      std::copy(block_buffer.buf_span.begin(), block_buffer.buf_span.end(), buffer);
      buf_span = span{buffer, size};
    }
  BlockBuffer(BlockBuffer&& block_buffer) = default;
  BlockBuffer& operator=(const BlockBuffer& block_buffer) {
    indexedTensor = block_buffer.indexedTensor;
    re = block_buffer.indexedTensor;
    if (allocated) delete[] buf_span.data();
    allocated = true;
    const auto size = block_buffer.buf_span.size();
    T* buffer = new T[size]; // that will need to be more complicated once we get device buffers
    std::copy(block_buffer.buf_span.begin(), block_buffer.buf_span.end(), buffer);
    buf_span = span{buffer, size};
  }
  BlockBuffer(Tensor<T> tensor, IndexVector blockid) 
    : allocated{true}, indexedTensor{tensor, blockid} {
    const size_t size = tensor.block_size(blockid);
    T* buffer = new T[size];
    buf_span = span{buffer, size};
    tensor.get(blockid, buf_span);
  }
  ~BlockBuffer() {
    if (allocated) delete[] buf_span.data();
  }

  // Whatever else is necessary to make the type regular

  auto begin() { return buf_span.begin(); }
  auto begin() const { return buf_span.begin(); }
  auto end() { return buf_span.end(); }
  auto end() const { return buf_span.end(); }
  auto get_span() { return buf_span; }
  const auto get_span() const { return buf_span; }
  auto data() { return buf_span.data(); }
  const auto data() const { return buf_span.data(); }
  void release_put() { indexedTensor.put(buf_span); release(); }
  void release_put(Tensor<T> tensor, IndexVector blockid) { tensor.put(blockid, buf_span); release(); }
  void release_add() { indexedTensor.add(buf_span); release(); }
  void release_add(Tensor<T> tensor, IndexVector blockid) { tensor.add(blockid, buf_span); release(); }
  void release() { 
    if (allocated) {
      delete[] buf_span.data();
      allocated = false;
    }
  }
  std::vector<size_t> block_dims() { return indexedTensor.first.block_dims(indexedTensor.second); }
  template<typename V>
  BlockBuffer& operator=(const V val) {
    std::fill(begin(), end(), val);
    return *this;
  }
private:
  span<T> buf_span;
  bool allocated = false;
  IndexedTensor<T> indexedTensor;
  // re is a pointer to allow it to be uninitialized.
  RuntimeEngine* re;
};

template<typename T>
bool operator==(const BlockBuffer<T> lhs, const BlockBuffer<T> rhs) {
  return lhs.size = rhs.size &&
    std::equal(lhs.get_data(), lhs.get_data() + lhs.get_size(), rhs.get_data()) &&
    lhs.get_tensor() == rhs.get_tensor() &&
    lhs.get_block_id() == rhs.get_block_id();
}

template<typename T, typename Stream>
inline auto& operator<<(Stream& os, BlockBuffer<T> bf) {
    // std::copy(bf.begin(), bf.end(), std::ostream_iterator<T>(os, " "));
    for (auto it = bf.begin(); it != bf.end(); ++it) {
        os << *it << " ";
    }
    return os;
}

} // namespace tamm

#endif
