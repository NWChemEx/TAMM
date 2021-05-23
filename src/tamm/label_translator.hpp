#ifndef TAMM_LABEL_TRANSLATOR_H_
#define TAMM_LABEL_TRANSLATOR_H_

#include "tamm/tiled_index_space.hpp"
#include "tamm/utils.hpp"
namespace tamm::internal {

/**
 * @brief Class to efficient translate index vectors between spaces. Used for
 * slicing and dependent spaces.
 *
 * @todo: Support to comppose label translators
 *
 */
class LabelTranslator {
public:
    // inspect the labels to figure out where we have independent plan (aka
    // translate_blockid) or dependent plan (translate_if_possible). Setup data
    // structure appropriately.
    LabelTranslator(const IndexLabelVec& from, const IndexLabelVec& to) :
      from_{from}, to_{to} {
        EXPECTS(from.size() == to.size());
        if(from == to) {
            plan_ = Plan::identical;
        } else if(internal::is_dense_labels(from) &&
                  internal::is_dense_labels(to)) {
            if(is_dense_subspaces()) {
                plan_ = Plan::dense;
                IndexVector from_blockid(from.size(), 0);
                auto [to_blockid, is_valid] =
                  internal::translate_blockid_if_possible(from_blockid, from_,
                                                          to_);
                EXPECTS(is_valid);
                dense_plan_info_ = DensePlanInfo(to_blockid);
            } else {
                plan_ = Plan::independent;
            }
        } else {
            plan_              = Plan::general;
            general_plan_info_ = GeneralPlanInfo(from, to);
        }
    }

    /**
     * @brief Perform the index translation
     *
     * @tparam T Element type in index vector
     * @param from_blockid Index vector to translate from
     * @return std::pair<std::vector<T>,bool> Translated index vector. True if
     * translation was successful. False otherwise.
     */
    template<typename T>
    std::pair<std::vector<T>, bool> apply(const std::vector<T>& from_blockid) {
        EXPECTS(from_blockid.size() == from_.size());
        std::vector<T> to_blockid(from_blockid.size());
        bool is_valid = true; // set appropriately during translation
        if(plan_ == Plan::identical) {
            return {from_blockid, is_valid};
        } else if(plan_ == Plan::dense) {
            for(size_t i = 0; i < from_blockid.size(); i++) {
                to_blockid[i] = from_blockid[i] + dense_plan_info_.offsets_[i];
            }
        } else if(plan_ == Plan::independent) {
            for(size_t i = 0; i < from_blockid.size(); i++) {
                to_blockid[i] = from_[i].tiled_index_space().translate(
                  from_blockid[i], to_[i].tiled_index_space());
            }
        } else {
            // implementation based on translate if possible
            // used precomputed values in GeneralPlanInfo
            for(size_t i = 0; i < from_.size(); i++) {
                if(!from_[i].tiled_index_space().is_compatible_with(
                     to_[i].tiled_index_space())) {
                    return {IndexVector{}, false};
                }
            }
            const auto& compute_order = general_plan_info_.compute_order_;
            const auto& from_dep_map  = general_plan_info_.from_dep_map_;
            const auto& to_dep_map    = general_plan_info_.to_dep_map_;

            for(size_t i = 0; i < compute_order.size(); i++) {
                IndexVector from_indep_vec, to_indep_vec;
                const size_t cur_pos = compute_order[i];
                auto it              = from_dep_map.find(cur_pos);
                EXPECTS(it != from_dep_map.end());
                for(const auto& ipos : it->second) {
                    from_indep_vec.push_back(from_blockid[ipos]);
                }
                it = to_dep_map.find(cur_pos);
                EXPECTS(it != to_dep_map.end());
                for(const auto& ipos : it->second) {
                    to_indep_vec.push_back(to_blockid[ipos]);
                }

                EXPECTS(from_[cur_pos].tiled_index_space().is_compatible_with(
                  to_[cur_pos].tiled_index_space()));

                auto [to_id, valid] =
                  from_[cur_pos].tiled_index_space().translate_if_possible(
                    from_blockid[cur_pos], from_indep_vec,
                    to_[cur_pos].tiled_index_space(), to_indep_vec);

                if(!valid) {
                    return {to_blockid, false};
                } else {
                    to_blockid[compute_order[i]] = to_id;
                }
            }
        }
        return {to_blockid, is_valid};
    }

    LabelTranslator reverse() { return LabelTranslator{to_, from_}; }

    LabelTranslator invert() { return reverse(); }

private:
    bool is_dense_subspaces() {
        for(size_t i = 0; i < from_.size(); i++) {
            if(!from_[i].tiled_index_space().is_dense_subspace() ||
               !to_[i].tiled_index_space().is_dense_subspace()) {
                return false;
            }
        }
        return true;
    }

    enum class Plan { independent, general, identical, dense };
    struct GeneralPlanInfo {
        GeneralPlanInfo() = default;

        GeneralPlanInfo(const IndexLabelVec& from_labels,
                        const IndexLabelVec& to_labels) {
            from_dep_map_  = construct_dep_map(from_labels);
            to_dep_map_    = construct_dep_map(to_labels);
            compute_order_ = topological_sort(to_dep_map_);
        }
        // things that can be computed once, such as computing dependences and
        // topological sort
        std::map<size_t, std::vector<size_t>> from_dep_map_;
        std::map<size_t, std::vector<size_t>> to_dep_map_;
        std::vector<size_t> compute_order_;
    };
    ///@todo Currently, we don't allow translation to a superset TiledIndexSpace
    ///so using an IndexVector for offsets works, but we might need to update it
    ///once we have the support for translation to superset.
    struct DensePlanInfo {
        DensePlanInfo() = default;
        DensePlanInfo(const IndexVector& offsets) : offsets_{offsets} {}
        IndexVector offsets_;
    };

    IndexLabelVec from_;
    IndexLabelVec to_;
    Plan plan_;
    GeneralPlanInfo general_plan_info_;
    DensePlanInfo dense_plan_info_;
}; // class LabelTranslator

} // namespace tamm::internal

#endif // TAMM_LABEL_TRANSLATOR_H_
