#ifndef TAMM_RUNTIME_ENGINE_HPP_
#define TAMM_RUNTIME_ENGINE_HPP_

#include "tamm/block_buffer.hpp"
#include "tamm/tensor.hpp"
#include "tamm/types.hpp"
#include "utility"

namespace tamm {

/**
 * @brief Mode values.
 * 
 */
enum class Mode { 
    PW,  /**< Write Permission */
    PR,  /**< Read Permission */
    PA,  /**< Accumulate Permission */
    AR,  /**< Read Access */
    AW,  /**< Write Access */
    ACW, /**< Cancellable Write Access */
    AA,  /**< Accumulate Access */
    AT   /**< Temporary Access */
};

class PermissionBase {
public:
    virtual Mode getMode() = 0;
};

template<typename T>
class Permission;

template<typename T>
class Permission<LabeledTensor<T>> : public PermissionBase {
public:
    Permission(LabeledTensor<T> lt, Mode mode) : lt(lt), mode(mode) {}

    Mode getMode() override { return mode; }
    
private:
    LabeledTensor<T> lt;
    Mode mode;
};

template<typename T>
class Permission<IndexedTensor<T>> : public PermissionBase {
public:
    Permission(IndexedTensor<T> lt, Mode mode) : lt(lt), mode(mode) {}
    Permission(typename IndexedTensor<T>::first_type tensor,
               typename IndexedTensor<T>::second_type index_vector,
               Mode mode) :
      lt(tensor, index_vector),
      mode(mode) {}

    Mode getMode() override { return mode; }

private:
    IndexedTensor<T> lt;
    Mode mode;
};

template<typename T>
class ReadPermission : public Permission<T> {
public:
    ReadPermission(T t) : Permission<T>(t, Mode::PR) {}
};

template<typename T>
class WritePermission : public Permission<T> {
public:
    WritePermission(T t) : Permission<T>(t, Mode::PW) {}
};

template<typename T>
class AccumPermission : public Permission<T> {
public:
    AccumPermission(T t) : Permission<T>(t, Mode::PA) {}
};

template<typename T>
class ReadAccess : public Permission<T> {
public:
    ReadAccess(T t) : Permission<T>(t, Mode::AR) {}
};

template<typename T>
class WriteAccess : public Permission<T> {
public:
    WriteAccess(T t) : Permission<T>(t, Mode::AW) {}
};

template<typename T>
class CancellableWriteAccess : public Permission<T> {
public:
    CancellableWriteAccess(T t) : Permission<T>(t, Mode::ACW) {}
};

template<typename T>
class AccumAccess : public Permission<T> {
public:
    AccumAccess(T t) : Permission<T>(t, Mode::AA) {}
};

template<typename T>
class TempAccess : public Permission<T> {
public:
    TempAccess(T t) : Permission<T>(t, Mode::AA) {}
};

// TBD: We need a way to consolidate IndexedTensors with LabeledTensors to
// compute dependencies
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

//   void execute() override {
//   execute_helper(std::index_sequence_for<Args...>{}); } template<typename
//   LocalLambda> void for_blocks(LocalLambda local_lambda) {
//     for_blocks_helper(local_lambda, std::index_sequence_for<Args...>{});
//   }
//   std::vector<BlockDescAccessPair> blocks() override {
//     std::vector<BlockDescAccessPair> block_access_pairs{};
//     for_blocks([&](auto access_wrapper){
//       block_access_pairs.push_back({access_wrapper.get_block().get_block_desc(),
//       access_wrapper.get_mode()}); return;
//     });
//     return block_access_pairs;
//   }
// private:
//   Lambda lambda_;
//   std::tuple<Args...> args_;
//   template<size_t... Is>
//   void execute_helper(std::index_sequence<Is...>) {
//     lambda_(AccessVisitor{[](auto access_wrapper) { return
//     access_wrapper.get_buffer(); } }(std::get<Is>(args_))...);
//   }
//   template<typename BlocksLambda, std::size_t... Is>
//   void for_blocks_helper(BlocksLambda blocks_lambda,
//   std::index_sequence<Is...>) {
//     (AccessVisitor{[=](auto access_wrapper) { return
//     blocks_lambda(access_wrapper); } }(std::get<Is>(args_)), ...);
//   }
// };

class RuntimeEngine {
public:
    class RuntimeContext {
    public:
        RuntimeContext(RuntimeEngine& re) : re(re) {}
        auto& runtimeEngine() { return re; }
        template<typename T>
        BlockBuffer<T> get_tmp_buffer(Tensor<T> tensor, IndexVector blockid) {
            // TBD: figure out memory space: do we need GPU/CPU buffer?
            const size_t size = tensor.block_size(blockid);
            span<T> span(new T[size], size);
            return BlockBuffer<T>(span, IndexedTensor{tensor, blockid}, &re,
                                  true);
        }

        template<typename T>
        BlockBuffer<T> get_buffer(Tensor<T> tensor, IndexVector blockid) {
            // TBD
        }

        template<typename Lambda, typename... Args>
        void submitTask(Lambda lambda, Args&&... args) {
            re.submitTask(lambda, std::forward<Args>(args)...);
        }

    private:
        RuntimeEngine& re;
    };

    RuntimeEngine() = default;

    ~RuntimeEngine() =  default;
    void executeAllthreads();

    // More buffer functions
    // e.g., buffer for accumulation
    //   * "forwarded" buffers with immediate effects? No need to explicitly
    //   write back.
    //   * Maybe special type for reference buffer.

    template<typename Lambda, typename... Args>
    void submitTask(Lambda lambda, Args&&... args) {
        std::apply(
          lambda,
          std::tuple_cat(
            std::make_tuple(RuntimeContext{*this}), std::tuple_cat([&]() {
                if constexpr(std::is_base_of_v<Args, PermissionBase>) {
                    return std::forward_as_tuple<Args>(args);
                } else {
                    return std::tuple{};
                }
            }()...)));
    }

private:
};

} // namespace tamm

#endif // TAMM_RUNTIME_ENGINE_HPP_