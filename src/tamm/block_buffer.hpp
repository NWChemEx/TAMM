#ifndef TAMM_BLOCK_BUFFER_HPP_
#define TAMM_BLOCK_BUFFER_HPP_

#include "tamm/types.hpp"

namespace tamm {

/**
 * @brief The class used to pass block buffers to user functions
 *
 * @tparam T 
 */
template<typename T>
class BlockBuffer {
public:
  BlockBuffer() = default;
  BlockBuffer(span<T> buf_span, Tensor<T> tensor, IndexVector id, bool allocated = false) 
    : buf_span{buf_span}, allocated{allocated}, tensor{tensor}, id{id} {}
  BlockBuffer(const BlockBuffer& block_buffer) 
    : id(block_buffer.id), tensor(block_buffer.tensor) 
    {
      if (allocated) delete[] buf_span.data();
      allocated = true;
      const auto size = block_buffer.buf_span.size();
      T* buffer = new T[size]; // that will need to be more complicated once we get device buffers
      std::copy(block_buffer.buf_span.begin(), block_buffer.buf_span.end(), buffer);
      buf_span = span{buffer, size};
    }
  BlockBuffer(BlockBuffer&& block_buffer) {
    buf_span = block_buffer.buf_span;
    id = block_buffer.id;
    tensor = block_buffer.tensor;
    allocated = block_buffer.allocated;
    block_buffer.allocated = false;
    block_buffer.buffer = nullptr;
  }
  BlockBuffer& operator=(const BlockBuffer& block_buffer) {
    id = block_buffer.id;
    tensor = block_buffer.tensor;
    if (allocated) delete[] buf_span.data();
    allocated = true;
    const auto size = block_buffer.buf_span.size();
    T* buffer = new T[size]; // that will need to be more complicated once we get device buffers
    std::copy(block_buffer.buf_span.begin(), block_buffer.buf_span.end(), buffer);
    buf_span = span{buffer, size};
  }
  ~BlockBuffer() {
    if (allocated) delete[] buf_span.data();
  }

  // Whatever else is necessary to make the type regular

  Tensor<T> get_tensor() const { return tensor; }
  T* get_data() { return buffer; }
  const T* get_data() const { return buffer; }
  BlockIdType get_block_id() const { return id; }
  size_t get_size() const { return size; }
  void put() { tensor.put(id, {buffer, size}); }
  void add() { tensor.add(id, {buffer, size}); }
private:
  span<T> buf_span;
  bool allocated = false;
  Tensor<T> tensor;
  BlockIdType id;
  RuntimeEngine& re;
};

template<typename T>
bool operator==(const BlockBuffer<T> lhs, const BlockBuffer<T> rhs) {
  return lhs.size = rhs.size &&
    std::equal(lhs.get_data(), lhs.get_data() + lhs.get_size(), rhs.get_data()) &&
    lhs.get_tensor() == rhs.get_tensor() &&
    lhs.get_block_id() == rhs.get_block_id();
}

} // namespace tamm

#endif