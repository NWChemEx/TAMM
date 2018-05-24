#ifndef TAMM_DAG_IMPL_HPP_
#define TAMM_DAG_IMPL_HPP_

#include "tamm/ops.hpp"
#include "tamm/types.hpp"

namespace tamm {

namespace internal {

class DAGBase {
public:
    virtual OpList ops()    = 0;
    virtual HashData hash() = 0;
    virtual ~DAGBase() {}
};

template<typename Func, typename... Args>
class DAGImpl : public DAGBase {
public:
    // Ctors
    DAGImpl() = default;

    DAGImpl(Func func, Args... args) :
      func_{func},
      args_{std::forward_as_tuple(args...)} {}

    // Copy/Move Ctors and Assignment Operators
    DAGImpl(DAGImpl&&)      = default;
    DAGImpl(const DAGImpl&) = default;
    DAGImpl& operator=(DAGImpl&&) = default;
    DAGImpl& operator=(const DAGImpl&) = default;

    DAGImpl& operator()(Args... args) {
        args_ = std::forward_as_tuple(args...);
        return *this;
    }

    DAGImpl& bind(Args... args) {
        args_ = std::forward_as_tuple(args...);
        return *this;
    }

    OpList ops() { return apply(func_, args_); }
    HashData hash() {
        HashData ret;

        return ret;
    }

protected:
    Func func_;
    std::tuple<Args...> args_;
}; // DAGImpl

} // namespace internal

using internal::DAGImpl;

template<typename Func, typename... Args>
static DAGImpl<Func, Args...> make_dag(Func func, Args... args) {
    return {func, args...};
}


} // namespace tamm

#endif // TAMM_DAG_IMPL_HPP_