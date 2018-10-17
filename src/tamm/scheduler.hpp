#ifndef TAMM_SCHEDULER_HPP_
#define TAMM_SCHEDULER_HPP_

#include <set>

#include "tamm/dag_impl.hpp"
#include "tamm/execution_context.hpp"
#include "tamm/ops.hpp"
#include "tamm/tensor.hpp"

namespace tamm {

using internal::DAGImpl;
/**
 * @brief Scheduler to execute a list of operations.
 * @ingroup operations
 */
class Scheduler {
public:
    /**
     * @brief Allocation status of tensor.
     *
     * This is used to perform some correctness checks on the list of
     * operations.
     * @todo Can this be replaced by AllocationStatus
     */
    // enum class TensorStatus { invalid, allocated, deallocated, initialized };

    // @to-do: what is the default scheduler?
    Scheduler()                 = default;
    Scheduler(const Scheduler&) = default;
    Scheduler(Scheduler&&)      = default;
    Scheduler& operator=(const Scheduler&) = default;
    Scheduler& operator=(Scheduler&&) = default;

    Scheduler(ExecutionContext* ec) : ec_{ec} {}

    template<typename OpType>
    Scheduler& operator()(const OpType& op) {
        OpList t_ops = op.canonicalize();

        for(auto& op : t_ops) { ops_.push_back(op); }
        return (*this);
    }

    Scheduler& allocate() { return *this; }

    ExecutionContext* ec() { return ec_; }

    template<typename TensorType, typename... Args>
    Scheduler& allocate(TensorType tensor, Args&... tensors) {
        ops_.push_back(std::make_shared<AllocOp<TensorType>>(tensor, ec()));
        return allocate(tensors...);
    }

    Scheduler& deallocate() { return *this; }

    template<typename TensorType, typename... Args>
    Scheduler& deallocate(TensorType tensor, Args&... tensors) {
        ops_.push_back(std::make_shared<DeallocOp<TensorType>>(tensor));
        return deallocate(tensors...);
    }

    void execute() {
        // for(auto& op : ops_) { op->execute(ec()->pg()); }
        for(size_t i = start_idx_; i < ops_.size(); i++) {
            ops_[i]->execute(ec()->pg());
            start_idx_++;
        }
    }

    template<typename Func, typename... Args>
    static void execute(DAGImpl<Func, Args...> dag) {}

    ~Scheduler() {
        // delete ops
    }

    template<typename LabeledTensorType, typename Func>
    Scheduler& gop(LabeledTensorType lhs, Func func) {
        ops_.push_back(
          std::make_shared<ScanOp<LabeledTensorType, Func>>(lhs, func));
        return *this;
    }

    template<typename LabeledTensorType, typename Func, size_t N>
    Scheduler& gop(LabeledTensorType lhs, std::array<LabeledTensorType, N> rhs,
                   Func func, ResultMode mode = ResultMode::set) {
        ops_.push_back(std::make_shared<MapOp<LabeledTensorType, Func, N>>(
          lhs, func, rhs, mode));
        return *this;
    }

private:
    ExecutionContext* ec_;
    // void validate() {
    //     // 1. every tensor used by operarions should be listed in tensors_

    //     // 2. every tensor must be initialized (part of LHS) or be an
    //     // input (ilve_in) tensor before it is used

    //     // 3. every output tensor should be allocated and set (be LHS in
    //     // at least one operation or be a live_in tensor)

    //     // 4. every tensor must be allocated before it is used

    //     // 5. every non-output (not in live_out) tensor must be
    //     // deallocated
    // }
    std::vector<std::shared_ptr<Op>> ops_;
    size_t start_idx_ = 0;

}; // class Scheduler

} // namespace tamm

#endif // TAMM_SCHEDULER_HPP_
