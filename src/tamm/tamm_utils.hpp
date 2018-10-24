#ifndef TAMM_TAMM_UTILS_HPP_
#define TAMM_TAMM_UTILS_HPP_

#include <vector>

namespace tamm {

template<typename T>
std::ostream& operator<<(std::ostream& os, std::vector<T>& vec) {
    os << "[";
    for(auto& x : vec) os << x << ",";
    os << "]\n";
    return os;
}

template<typename T>
void print_tensor(const Tensor<T>& t) {
    auto lt = t();
    for(auto it : t.loop_nest()) {
        auto blockid   = internal::translate_blockid(it, lt);
        TAMM_SIZE size = t.block_size(blockid);
        std::vector<T> buf(size);
        t.get(blockid, buf);
        std::cout << "block" << blockid;
        // if (buf[i]>0.0000000000001||buf[i]<-0.0000000000001)
        for(TAMM_SIZE i = 0; i < size; i++) std::cout << buf[i] << " ";
        std::cout << std::endl;
    }
}

template<typename T>
T get_scalar(Tensor<T>& tensor) {
    T scalar;
    EXPECTS(tensor.num_modes() == 0);
    tensor.get({}, {&scalar, 1});
    return scalar;
}

template<typename T, typename Func> 
void update_tensor(LabeledTensor<T> lt, Func lambda){
    LabelLoopNest loop_nest{lt.labels()};

    for(const auto& itval : loop_nest) {
        const IndexVector blockid = internal::translate_blockid(itval, lt);  
        size_t size               = lt.tensor().block_size(blockid);
        std::vector<T> buf(size);

        lt.tensor().get(blockid,buf);
        lambda(blockid, buf);
        lt.tensor().put(blockid, buf);
    }  
}

} // namespace tamm

#endif // TAMM_TAMM_UTILS_HPP_
