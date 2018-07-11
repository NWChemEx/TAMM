#include "tamm/tamm.hpp"

namespace tamm {

template<typename T>
template<class... Args>
LabeledTensor<T>
Tensor<T>::operator()(Args&&... rest) const {
    return LabeledTensor<T>{*this, std::forward<Args>(rest)...};
}



} // namespace tamm
