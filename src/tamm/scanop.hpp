#pragma once

#include <memory>
#include <vector>
#include <map>

#include "tamm/boundvec.hpp"
#include "tamm/errors.hpp"
#include "tamm/labeled_tensor.hpp"
#include "tamm/runtime_engine.hpp"
#include "tamm/tensor.hpp"
#include "tamm/types.hpp"
#include "tamm/utils.hpp"
#include "tamm/work.hpp"

namespace tamm {
template<typename LabeledTensorT, typename Func>
class ScanOp : public Op {
public:
    ScanOp(const LabeledTensorT& lhs, Func func) : lhs_{lhs}, func_{func} {
        fillin_labels();
    }

    OpList canonicalize() const override { return OpList{(*this)}; }

    OpType op_type() const override { return OpType::scan; }

    std::shared_ptr<Op> clone() const override {
        return std::shared_ptr<Op>(new ScanOp<LabeledTensorT, Func>{*this});
    }

    void execute(ExecutionContext& ec, ExecutionHW hw = ExecutionHW::CPU) override {
        using TensorElType = typename LabeledTensorT::element_type;
        // the iterator to generate the tasks
        const auto& tensor = lhs_.tensor();
        LabelLoopNest loop_nest{lhs_.labels()};
        // const IndexLabelVec& iter_labels =
        // internal::sort_on_dependence(lhs_.labels());
        // std::vector<IndexLoopBound> ilbs;
        // for(const auto& lbl : iter_labels) { ilbs.push_back({lbl}); }
        // IndexLoopNest loop_nest { ilbs };
        // const std::vector<size_t>& lhs_pm =
        //   internal::perm_map_compute(iter_labels, lhs_.labels());
        // auto loop_nest = lhs_.tensor().loop_nest();
        // function to compute one block
        auto lambda = [&](const IndexVector& blockid) {
            auto tensor = lhs_.tensor();
            EXPECTS(blockid.size() == lhs_.labels().size());
            EXPECTS(blockid.size() == tensor.num_modes());
            const auto& translated_blockid =
              internal::translate_blockid(blockid, lhs_);
            // const IndexVector& blockid =
            //   internal::perm_map_apply(itval, lhs_pm);
            const size_t size = tensor.block_size(translated_blockid);
            std::vector<TensorElType> buf(size);
            tensor.get(translated_blockid, buf);
            func_(tensor, translated_blockid, buf);
        };
        // ec->...(loop_nest, lambda);
        //@todo use a scheduler
        do_work(ec, loop_nest, lambda);
    }

    TensorBase* writes() const {
        return nullptr;
    }

    std::vector<TensorBase*> reads() const {
        return {lhs_.base_ptr()};
    }

    TensorBase* accumulates() const {
        return nullptr;
    }

    bool is_memory_barrier() const {
        return false;
    }

protected:
    void fillin_labels() {
        using internal::fillin_tensor_label_from_map;
        using internal::update_fillin_map;
        std::map<std::string, Label> str_to_labels;
        update_fillin_map(str_to_labels, lhs_.str_map(), lhs_.str_labels(), 0);
        fillin_tensor_label_from_map(lhs_, str_to_labels);
    }

    /**
     * @brief Check if the parameters form a valid operation. The parameters
     * form a valid operation if:
     *
     * 1. Every label depended on by another label (i.e., all 'd' such that
     * there exists label 'l(d)') is bound at least once
     *
     * 2. There are no conflicting dependent label specifications. That if
     * 'a(i)' is a label in either lta or ltc, there is no label 'a(j)' (i!=j)
     * in either lta or ltc.
     *
     * @pre lhs_.validate(), rhs1_.validate() and rhs2_.validate() have been
     *  invoked
     */
    void validate() {
        IndexLabelVec ilv{lhs_.labels()};

        for(size_t i = 0; i < ilv.size(); i++) {
            for(const auto& dl : ilv[i].secondary_labels()) {
                size_t j;
                for(j = 0; j < ilv.size(); j++) {
                    if(dl.tiled_index_space() == ilv[j].tiled_index_space() &&
                       dl.label() == ilv[j].label()) {
                        break;
                    }
                }
                EXPECTS(j < ilv.size());
            }
        }

        for(size_t i = 0; i < ilv.size(); i++) {
            const auto& ilbl = ilv[i];
            for(size_t j = i + 1; j < ilv.size(); j++) {
                const auto& jlbl = ilv[j];
                if(ilbl.tiled_index_space() == jlbl.tiled_index_space() &&
                   ilbl.label() == jlbl.label() && 
                   ilbl.label_str() == jlbl.label_str()) {
                    EXPECTS(ilbl == jlbl);
                }
            }
        }
    }

    LabeledTensorT lhs_;
    Func func_;
};
} // namespace tamm
