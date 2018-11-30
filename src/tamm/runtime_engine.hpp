#ifndef TAMM_RUNTIME_ENGINE_HPP_
#define TAMM_RUNTIME_ENGINE_HPP_

#include "utility"
#include "tamm/types.hpp"
#include "tamm/tensor.hpp"
#include "tamm/block_buffer.hpp"

namespace tamm {

enum class AccessMode {
    KR, KW, KRW, AC
};

class PermissionBase {
    // Whatever polymorphic interface is needed, if any
    // Inherit from this class to mark a class as a permission
};

template<typename T> class Permission;

template<typename T>
class Permission<LabeledTensor<T>> : public PermissionBase {
public:
    Permission(LabeledTensor<T> lt, AccessMode acc) : lt(lt), acc(acc) {}
private:
    LabeledTensor<T> lt;
    AccessMode acc;
};

template<typename T>
class Permission<IndexedTensor<T>> : public PermissionBase {
public:
    Permission(IndexedTensor<T> lt, AccessMode acc) : lt(lt) {}
    Permission(typename IndexedTensor<T>::first_type tensor, 
               typename IndexedTensor<T>::second_type index_vector,
               AccessMode acc) : lt(tensor, index_vector), acc(acc) {}
private:
    IndexedTensor<T> lt;
    AccessMode acc;
};

template<typename T>
class ReadPermission : public Permission<T> {
public:
    ReadPermission(T t) : Permission<T>(t, AccessMode::KR) {}
};

template<typename T>
class ReadWritePermission : public Permission<T> {
public:
    ReadWritePermission(T t) : Permission<T>(t, AccessMode::KRW) {}
};

template<typename T>
class WritePermission : public Permission<T> {
public:
    WritePermission(T t) : Permission<T>(t, AccessMode::KW) {}
};

template<typename T>
class AccumPermission : public Permission<T> {
public:
    AccumPermission(T t) : Permission<T>(t, AccessMode::AC) {}
};

// TBD: We need a way to consolidate IndexedTensors with LabeledTensors to compute dependencies
//
// class Task {
// public:
//   virtual ~Task() = default;
//   Task(const Task&) = delete; 
//   Task& operator=(const Task&) = delete;
//   Task(Task&&) = delete;
//   Task& operator=(Task&&) = delete;
//   Task() = default;
//   size_t get_ndeps() {return ndeps_;}
//   void add_successor(Task *x) { successors_.push_back(x); }
//   void inc_ndeps(size_t nd=1) { ndeps_ += nd; }
//   void set_nremainingdeps(size_t nrd) { nremaining_deps_ = {nrd}; }
//   size_t get_nremainingdeps() { return nremaining_deps_; }
//   void dec_nremainingdeps(size_t d=1) { nremaining_deps_ -= d; }
//   std::vector<Task*> get_successors() { return successors_; }
//   virtual void execute() = 0;
//   virtual std::vector<BlockDescAccessPair> blocks() = 0;
// private:
//   //std::vector<Task*> predecessors_;
//   std::vector<Task*> successors_;
//   size_t ndeps_{};
//   size_t nremaining_deps_{};
// };

// template<typename Lambda, typename ...Args>
// class TaskImpl : public Task
// {
// public:
//   TaskImpl(TaskImpl&&) = default;
//   TaskImpl(const TaskImpl&) = default;
//   TaskImpl& operator=(TaskImpl&&) = default;
//   TaskImpl& operator=(const TaskImpl&) = default;
//   ~TaskImpl() override = default;

//   TaskImpl(Lambda lambda, Args ...args)
//     : lambda_{lambda}, args_(args...) {}

//   void execute() override { execute_helper(std::index_sequence_for<Args...>{}); }
//   template<typename LocalLambda>
//   void for_blocks(LocalLambda local_lambda) {
//     for_blocks_helper(local_lambda, std::index_sequence_for<Args...>{});
//   }
//   std::vector<BlockDescAccessPair> blocks() override {
//     std::vector<BlockDescAccessPair> block_access_pairs{};
//     for_blocks([&](auto access_wrapper){ 
//       block_access_pairs.push_back({access_wrapper.get_block().get_block_desc(), access_wrapper.get_mode()});
//       return;
//     });
//     return block_access_pairs;
//   }
// private:
//   Lambda lambda_;
//   std::tuple<Args...> args_;
//   template<size_t... Is>
//   void execute_helper(std::index_sequence<Is...>) {
//     lambda_(AccessVisitor{[](auto access_wrapper) { return access_wrapper.get_buffer(); } }(std::get<Is>(args_))...);
//   }
//   template<typename BlocksLambda, std::size_t... Is>
//   void for_blocks_helper(BlocksLambda blocks_lambda, std::index_sequence<Is...>) {
//     (AccessVisitor{[=](auto access_wrapper) { return blocks_lambda(access_wrapper); } }(std::get<Is>(args_)), ...);
//   }
// };

class RuntimeEngine {
public:
    RuntimeEngine() = default;

    ~RuntimeEngine();
    void executeAllthreads();

    template<typename T>
    BlockBuffer<T> temp_buf(Tensor<T> tensor, IndexVector blockid) {
        // TBD: figure out memory space: do we need GPU/CPU buffer?
        const size_t size = tensor.block_size(blockid);
        span<T> span(new T[size], size);
        return BlockBuffer<T>(span, IndexedTensor{tensor, blockid}, this, true);
    }

    // More buffer functions
    // e.g., buffer for accumulation
    //   * "forwarded" buffers with immediate effects? No need to explicitly write back.
    //   * Maybe special type for reference buffer.

    template<typename Lambda, typename ...Args>
    void submitTask(Lambda lambda, Args&&... args) {
        std::apply(lambda, std::tuple_cat(std::make_tuple(std::reference_wrapper(*this)), std::tuple_cat([&](){
                if constexpr (std::is_base_of_v<Args, PermissionBase>) {
                    return std::forward_as_tuple<Args>(args);
                } else {
                    return std::tuple{};
                }
            }()...
        )));
    }
    
private:
};

} // namespace tamm

#endif // TAMM_RUNTIME_ENGINE_HPP_