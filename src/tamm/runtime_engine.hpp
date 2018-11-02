#ifndef TAMM_EXECUTION_ENGINE_HPP_
#define TAMM_EXECUTION_ENGINE_HPP_

#include "utility"
#include "tamm/types.hpp"
#include "tamm/tensor.hpp"

namespace tamm {

template<typename T>
using IndexedTensor = std::pair<tensor<T>, IndexVector>;

enum class AccessMode {
    KR, KW, KRW, AC
};

template<typename T> class Permission;

template<typename T>
class Permission<LabeledTensor<T>> {
    Permission(LabeledTensor<T> lt, AccessMode acc) : lt(lt), acc(acc) {}
private:
    LabeledTensor lt;
    AccessMode acc;
};

template<typename T>
class Permission<IndexedTensor<T>> {
    Permission(IndexedTensor<T> lt, AccessMode acc) : lt(lt) {}
    Permission(IndexedTensor<T>::typename first_type tensor, 
               IndexedTensor<T>::typename second_type index_vector,
               AccessMode acc) : lt(tensor, index_vector), acc(acc) {}
private:
    IndexedTensor lt;
    AccessMode acc;
};

template<typename T>
class ReadAccess : public Permission<T> {
    ReadAccess(T t) : Permission(T t, AccessMode::KR) {}
}

template<typename T>
class ReadWriteAccess : public Permission<T> {
    ReadWriteAccess(T t) : Permission(T t, AccessMode::KRW) {}
}

template<typename T>
class WriteAccess : public Permission<T> {
    WriteAccess(T t) : Permission(T t, AccessMode::KW) {}
}

template<typename T>
class AccumAccess : public Permission<T> {
    AccumAccess(T t) : Permission(T t, AccessMode::AC) {}
}

class RuntimeEngine {
public:
    RuntimeEngine() = default;
    RuntimeEngine(TaskEngine* taskengine);

    ~RuntimeEngine();
    void executeAllthreads(TaskEngine* taskengine);
    void executeAllthreads();

private:
};

} // namespace tamm

#endif // TAMM_EXECUTION_ENGINE_HPP_