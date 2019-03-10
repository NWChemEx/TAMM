#ifndef TAMM_SCHEDULER_HPP_
#define TAMM_SCHEDULER_HPP_

#include <set>

#include "tamm/dag_impl.hpp"
#include "tamm/execution_context.hpp"
#include "tamm/ops.hpp"
#include "tamm/tensor.hpp"
#include "ga-mpi.h"

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

    Scheduler(ExecutionContext& ec) : ec_{ec} {}

    template<typename OpType>
    Scheduler& operator()(const OpType& op) {
        OpList t_ops = op.canonicalize();

        for(auto& op : t_ops) { ops_.push_back(op); }
        return (*this);
    }

    Scheduler& allocate() { return *this; }

    ExecutionContext& ec() { return ec_; }

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

    template <typename T>
    bool has_intersect(const std::vector<T>& lhs,
                       const std::vector<T>& rhs) {
        
        for(const auto& l_item : lhs) {
            for(const auto& r_item : rhs) {
                if(l_item == r_item)
                    return true;
            }
        }
        return false;
    }
    bool has_dependence(const std::vector<TensorBase*>& R1,
                        const std::vector<TensorBase*>& W1,
                        const std::vector<TensorBase*>& R2,
                        const std::vector<TensorBase*>& W2) {
        
        return (has_intersect(R1, W2) || 
                has_intersect(W1, R2) ||
                has_intersect(W1, W2) );
    }

    bool has_dependence(const std::vector<TensorBase*>& R1,
                        const std::vector<TensorBase*>& W1,
                        const std::vector<TensorBase*>& A1,
                        const std::vector<TensorBase*>& R2,
                        const std::vector<TensorBase*>& W2,
                        const std::vector<TensorBase*>& A2) {
        return (has_intersect(R1, W2) ||
                has_intersect(W1, R2) ||
                has_intersect(W1, W2) ||
                has_intersect(R1, A2) ||
                has_intersect(W1, A2) ||
                has_intersect(R2, A1) ||
                has_intersect(W2, A1)
                );
    }

    std::vector<size_t> levelize(const std::vector<std::shared_ptr<Op>>& ops,
                                 size_t start_id,
                                 size_t end_id) {
        EXPECTS(start_id >= 0 && start_id <= ops.size());
        EXPECTS(end_id >= start_id && end_id <= ops.size());

        std::vector<size_t> groups;

        size_t group_start = start_id;
        std::vector<TensorBase*> group_reads, group_writes, group_accums;
        for(size_t i = start_id; i < end_id; i++) {
            std::vector<TensorBase*> reads, writes, accums;
            reads = std::vector<TensorBase*>(ops[i]->reads());
            if(auto wr = ops[i]->writes(); wr != nullptr) {
                writes = std::vector<TensorBase*>{wr};
            }
            if(auto ac = ops[i]->accumulates(); ac != nullptr) {
                accums = std::vector<TensorBase*>{ac};
            }

            if(ops[i]->is_memory_barrier() ||
               has_dependence(group_reads, group_writes, group_accums, reads,
                              writes, accums)) {
                groups.push_back(i - group_start);
                group_start = i;
                group_reads = reads;
                group_writes = writes;
                group_accums = accums;
            } else {
                group_reads.insert(group_reads.end(), reads.begin(), reads.end());
                group_writes.insert(group_writes.end(), writes.begin(), writes.end());
                group_accums.insert(group_accums.end(), accums.begin(), accums.end());
            }
        }

        if(group_start < end_id) {
            groups.push_back(end_id - group_start);
        }

        return groups;
    }

    void execute() {
        auto groups = levelize(ops_, start_idx_, ops_.size());
        // std::cerr << "Groups: [ ";
        // for(const auto& sz : groups) {
        //     std::cerr << sz << " ";
        // }
        // std::cerr << "]" << std::endl;

        // AtomicCounter* ac = new AtomicCounterGA(ec().pg(), ops_.size() - start_idx_);
        // ac->allocate(0);
        
        size_t off = start_idx_;
        for(size_t g : groups) {
            EXPECTS(g > 0);
            AtomicCounter* ac = new AtomicCounterGA(ec().pg(), g);
            ac->allocate(0);

            for(size_t i = off; i < off + g; i++, start_idx_++) {
                ec().set_ac(IndexedAC(ac, i - off));
                ops_[i]->execute(ec());
            }

            ec().set_ac(IndexedAC(nullptr, 0));
            ac->deallocate();
            delete ac;

            
            //memory fence. for now GA_Sync()
            // GA_Sync();
            //pg.barrier()
            ec().pg().barrier();
            off += g;
        }
        // ac->deallocate();
        // delete ac;
        // // for(auto& op : ops_) { op->execute(ec()->pg()); }
        // for(size_t i = start_idx_; i < ops_.size(); i++) {
        //     ops_[i]->execute(ec());
        //     start_idx_++;
        // }
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
    ExecutionContext& ec_;
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
