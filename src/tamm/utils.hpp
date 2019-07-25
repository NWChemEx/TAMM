#ifndef TAMM_UTILS_HPP_
#define TAMM_UTILS_HPP_

namespace tamm {

namespace internal {

inline void update_fillin_map(std::map<std::string, Label>& str_to_labels,
                              const std::vector<bool>& str_map,
                              const std::vector<std::string>& str_labels,
                              int initial_off) {
    const size_t sz = str_labels.size();
    for(size_t i = 0; i < sz; i++) {
        if(str_map[i]) { str_to_labels[str_labels[i]] = -initial_off - i - 1; }
    }
}

template<typename LabelTensorT>
inline void fillin_tensor_label_from_map(
  LabelTensorT& ltensor, const std::map<std::string, Label>& str_to_labels) {
    IndexLabelVec new_labels = ltensor.labels();
    const size_t sz          = ltensor.labels().size();
    for(size_t i = 0; i < sz; i++) {
        if(ltensor.str_map()[i]) {
            EXPECTS(str_to_labels.find(ltensor.str_labels()[i]) !=
                    str_to_labels.end());
            new_labels[i] = ltensor.tensor().tiled_index_spaces()[i].label(
              str_to_labels.find(ltensor.str_labels()[i])->second);
        }
    }
    ltensor.set_labels(new_labels);
}

/**
 * @ingroup perm
 * @brief Compute permutation to be performed to permute vector @p from to
 * vector @p to.
 * @param from Source vector for the permutation
 * @param to Target vector for the permutation
 * @pre @p from and @p to are permutations of each other
 * @pre from.size() == to.size()
 * @return Vector to permute @p from to @p to.
 * @post Return ret such that:
 * ensures 0<=i<from.size(): to[i] = from[ret[i]]
 */
template<typename T>
PermVector perm_compute(const std::vector<T>& from, const std::vector<T>& to) {
    PermVector layout;

    EXPECTS(from.size() == to.size());
    for(auto p : to) {
        auto itr = std::find(from.begin(), from.end(), p);
        EXPECTS(itr != from.end());
        layout.push_back(itr - from.begin());
    }
    return layout;
}

template<typename T>
bool are_permutations(const std::vector<T>& vec1, const std::vector<T>& vec2) {
    if(vec1.size() != vec2.size()) { return false; }
    std::vector<bool> taken(vec1.size(), false);
    for(size_t i = 0; i < vec1.size(); i++) {
        auto it = std::find(vec2.begin(), vec2.end(), vec1[i]);
        if(it == vec2.end()) { return false; }
        if(taken[std::distance(vec2.begin(), it)] == true) { return false; }
        taken[std::distance(vec2.begin(), it)] = true;
    }
    return true;
}

template<typename T>
std::vector<T> unique_entries(const std::vector<T>& input_vec) {
    std::vector<T> ret;

    for(const auto& val : input_vec) {
        auto it = std::find(ret.begin(), ret.end(), val);
        if(it == ret.end()) { ret.push_back(val); }
    }
    return ret;
}

template<typename TiledIndexLabel>
std::vector<TiledIndexLabel> unique_entries_by_primary_label(const std::vector<TiledIndexLabel>& input_vec) {
    std::vector<TiledIndexLabel> ret;

    for(const auto& val : input_vec) {
        auto it = std::find_if(ret.begin(), ret.end(), [&](const auto& lbl) {
            return lbl.primary_label() == val.primary_label();
        });
        if(it == ret.end()) { ret.push_back(val); }
    }
    return ret;
}

template<typename T>
std::vector<size_t> perm_map_compute(const std::vector<T>& unique_vec,
                                     const std::vector<T>& vec_required) {
    std::vector<size_t> ret;
    for(const auto& val : vec_required) {
        auto it = std::find(unique_vec.begin(), unique_vec.end(), val);
        EXPECTS(it >= unique_vec.begin());
        EXPECTS(it != unique_vec.end());
        ret.push_back(it - unique_vec.begin());
    }
    return ret;
}

template<typename T>
std::vector<size_t> perm_map_compute_by_primary_label(
  const std::vector<T>& unique_vec, const std::vector<T>& vec_required) {
    std::vector<size_t> ret;
    for(const auto& val : vec_required) {
        auto it = std::find_if(
          unique_vec.begin(), unique_vec.end(), [&](const auto& lbl) {
              return val.primary_label() == lbl.primary_label();
          });
        EXPECTS(it >= unique_vec.begin());
        EXPECTS(it != unique_vec.end());
        ret.push_back(it - unique_vec.begin());
    }
    return ret;
}

template<typename T, typename Integer>
std::vector<T> perm_map_apply(const std::vector<T>& input_vec,
                              const std::vector<Integer>& perm_map) {
    std::vector<T> ret;
    for(const auto& pm : perm_map) {
        EXPECTS(pm < input_vec.size());
        ret.push_back(input_vec[pm]);
    }
    return ret;
}

template<typename T, typename Integer>
void perm_map_apply(std::vector<T>& out_vec, const std::vector<T>& input_vec,
                    const std::vector<Integer>& perm_map) {
    out_vec.resize(perm_map.size());
    for(size_t i = 0; i < perm_map.size(); i++) {
        EXPECTS(perm_map[i] < input_vec.size());
        out_vec[i] = input_vec[perm_map[i]];
    }
}

inline IndexLabelVec sort_on_dependence(const IndexLabelVec& labels) {
    std::vector<TileLabelElement> primary_labels;
    std::vector<size_t> sorted_order;
    for(const auto& lbl : labels) {
        primary_labels.push_back(lbl.primary_label());
    }
    for(size_t i = 0; i < labels.size(); i++) {
        const auto& lbl = labels[i];
        for(const auto& slbl : lbl.secondary_labels()) {
            const auto it =
              std::find(primary_labels.begin(), primary_labels.end(), slbl);
            EXPECTS(it != primary_labels.end());
            const auto sit = std::find(sorted_order.begin(), sorted_order.end(),
                                       it - primary_labels.begin());
            if(sit == sorted_order.end()) {
                sorted_order.push_back(it - primary_labels.begin());
            }
        }
        const auto it = std::find(sorted_order.begin(), sorted_order.end(), i);
        if(it == sorted_order.end()) { sorted_order.push_back(i); }
    }
    IndexLabelVec ret;
    for(const auto& pos : sorted_order) { ret.push_back(labels[pos]); }
    return ret;
}

inline std::tuple<IndexLabelVec, IndexVector>
extract_blockid_and_label(const IndexLabelVec& input_labels,
                          const IndexVector& input_blockid,
                          const IndexLabelVec& labels_to_match) {
    IndexLabelVec ret_labels;
    IndexVector ret_blockid;
    for (size_t i = 0; i < labels_to_match.size(); i++) {
        auto lbl = labels_to_match[i];
    // for(const auto& lbl : labels_to_match) {
        auto it = std::find_if(
          input_labels.begin(), input_labels.end(), [&](const auto& itlbl) {
              return lbl.primary_label() == itlbl.primary_label();
          });
        EXPECTS(it != input_labels.end());
        size_t pos = it - input_labels.begin();
        EXPECTS(pos < input_labels.size() && 0 <= pos);
        // EXPECTS(pos < input_blockid.size() && 0 <= pos);

        ret_labels.push_back(input_labels[pos]);
        ret_blockid.push_back(input_blockid[i]);
    }
    return {ret_labels, ret_blockid};
}

template<typename T>
bool cartesian_iteration(std::vector<T>& itr, const std::vector<T>& end) {
    EXPECTS(itr.size() == end.size());

    int i;
    for(i = -1 + itr.size(); i >= 0 && itr[i] + 1 == end[i]; i--) {
        itr[i] = T{0};
    }

    if(i >= 0) {
        ++itr[i];
        return true;
    }
    return false;
}

inline IndexVector indep_values(
  const IndexVector& blockid, const Index& idx,
  const std::map<size_t, std::vector<size_t>>& dep_map) {
    IndexVector ret{};
    if(dep_map.find(idx) != dep_map.end()) {
        for(const auto& dep_id : dep_map.at(idx)) {
            ret.push_back(blockid[dep_id]);
        }
    }
    return ret;
}

template<typename LabeledTensorT>
IndexVector translate_blockid(const IndexVector& blockid,
                              const LabeledTensorT& ltensor) {
    EXPECTS(blockid.size() == ltensor.labels().size());
    const auto& tensor  = ltensor.tensor();
    const auto& dep_map = tensor.dep_map();
    EXPECTS(blockid.size() == tensor.num_modes());
    IndexVector translate_blockid;
    for(size_t i = 0; i < blockid.size(); i++) {
        auto indep_vals = indep_values(blockid, i, dep_map);
        if(!indep_vals.empty()){
            auto l_dep_map = ltensor.labels()[i].tiled_index_space().tiled_dep_map();
            auto t_dep_map = tensor.tiled_index_spaces()[i].tiled_dep_map();
            // check if any one of them doesn't have the TIS for indep_values
            if(l_dep_map.find(indep_vals) == l_dep_map.end() ||
               t_dep_map.find(indep_vals) == t_dep_map.end())
                return IndexVector(blockid.size(), -1);
        }
        const auto& label_tis =
          ltensor.labels()[i].tiled_index_space()(indep_vals);
        const auto& tensor_tis = tensor.tiled_index_spaces()[i](indep_vals);
        Index val              = label_tis.translate(blockid[i], tensor_tis);
        translate_blockid.push_back(val);
    }
    return translate_blockid;
}

/**
 * @brief Construct a dependence map from a label vector. The returned dependence map returns the list of indices a given index depends on, by comparing the primary labels.
 * 
 * For example, the routine returns {0:[], 1:[0]} for (i,j(i)). All values in this map are empty when there are no dependent labels. When duplicates exist (e.g., (i,i,j(i))), one of them is arbitrarily picked. 
 */
inline std::map<size_t, std::vector<size_t>> construct_dep_map(
  const std::vector<TiledIndexLabel>& tile_labels) {
    std::map<size_t, std::vector<size_t>> dep_map;
    std::vector<TileLabelElement> primary_labels;
    for(const auto& lbl : tile_labels) {
        primary_labels.push_back(lbl.primary_label());
    }
    for(size_t i = 0; i < tile_labels.size(); i++) {
        std::vector<size_t> deps;
        for(auto& sec_lbl : tile_labels[i].secondary_labels()) {
            auto it =
              std::find(primary_labels.begin(), primary_labels.end(), sec_lbl);
            EXPECTS(it != primary_labels.end());
            deps.push_back(it - primary_labels.begin());
        }
        dep_map[i] = deps;
    }
    return dep_map;
}

template<typename T>
std::vector<T> topological_sort(const std::map<T,std::vector<T>>& dep_map) {
    size_t num_ids = dep_map.size();
    std::vector<T> order(num_ids);
    std::vector<bool> done(num_ids, false);
    size_t ctr=0;
    for(size_t i=0; i<num_ids; i++) {
        if(done[i]) continue;
        std::vector<size_t> stack{i};
        while(!stack.empty()) {
            for(auto id: dep_map.find(stack.back())->second) {
                EXPECTS(id != i);
                if(!done[id]) {
                    stack.push_back(id);
                    continue;
                }
            }
            order[stack.back()] = ctr++;
            done[stack.back()] = true;
            stack.pop_back();
        }
        EXPECTS(done[i]);
    }
    EXPECTS(ctr == num_ids);
    std::vector<T> new_order(num_ids);
    for (size_t i = 0; i < num_ids; i++) {
        new_order[order[i]] = i;
    }

    return new_order;
    // return order;
}

inline std::tuple<IndexVector, bool> translate_blockid_if_possible(
  const IndexVector& from_blockid,
  const IndexLabelVec& from_label,
  const IndexLabelVec& to_label) {
    EXPECTS(from_blockid.size() == from_label.size());
    EXPECTS(from_label.size() == to_label.size());
    if(from_label == to_label){
        auto to_blockid = from_blockid;
        return {to_blockid, true};
    }
    
    for(size_t i = 0; i < from_label.size(); i++) {
        if(!from_label[i].tiled_index_space().is_compatible_with(
             to_label[i].tiled_index_space())) {
            return {IndexVector{}, false};
        }
    }
    const std::map<size_t, std::vector<size_t>>& from_dep_map =
      construct_dep_map(from_label);
    const std::map<size_t, std::vector<size_t>>& to_dep_map = construct_dep_map(to_label);

    // for(auto& [key, value] : from_dep_map) {
    //     std::cout << "key - " << key << std::endl;
    //     for(auto& id : value) {
    //         std::cout << "value - " << id << std::endl;
    //     }
    // }

    // for(auto& [key, value] : to_dep_map) {
    //     std::cout << "key - " << key << std::endl;
    //     for(auto& id : value) {
    //         std::cout << "value - " << id << std::endl;
    //     }
    // }

    std::vector<size_t> compute_order = topological_sort(to_dep_map);
    // std::cout << "compute_order: ";
    // for(auto& i : compute_order) {
    //     std::cout << i << " ";
    // }
    // std::cout << std::endl;

    EXPECTS(compute_order.size() == from_blockid.size());
    IndexVector to_blockid(from_blockid.size(), -1);
    for(size_t i = 0; i < compute_order.size(); i++) {
        IndexVector from_indep_vec, to_indep_vec;
        const size_t cur_pos = compute_order[i];
        auto it = from_dep_map.find(cur_pos);
        EXPECTS(it != from_dep_map.end());
        for(const auto& ipos : it->second) {
            from_indep_vec.push_back(from_blockid[ipos]);
        }
        it = to_dep_map.find(cur_pos);
        EXPECTS(it != to_dep_map.end());
        for(const auto& ipos : it->second) {
            to_indep_vec.push_back(to_blockid[ipos]);
        }
        size_t to_id;
        bool valid;
        EXPECTS(from_label[cur_pos].tiled_index_space().is_compatible_with(
          to_label[cur_pos].tiled_index_space()));
        
        // if(!(from_blockid[cur_pos] < from_blockid.size())){
        //     std::cout << "from_blockid.size() = " << from_blockid.size() << std::endl;
        //     std::cout << "from_blockid[" << cur_pos << "] = " << from_blockid[cur_pos] << std::endl;
            
        // }
        // EXPECTS(from_blockid[cur_pos] < from_blockid.size());
        // std::cout << "cur_pos = " << cur_pos << std::endl;
        // std::cout << "from_blockid[cur_pos] = " << from_blockid[cur_pos] << std::endl;
        // std::cout << "from_label[cur_pos] = " << &from_label[cur_pos] << std::endl;
        // std::cout << "to_label[cur_pos] = " << &to_label[cur_pos] << std::endl;

        // std::cout << "from_indep_vec: ";
        // for(auto& i : from_indep_vec) {
        //     std::cout << i << " ";
        // }
        // std::cout << std::endl;

        // std::cout << "to_indep_vec: ";
        // for(auto& i : to_indep_vec) {
        //     std::cout << i << " ";
        // }
        // std::cout << std::endl;
        

        std::tie(to_id, valid) =
          from_label[cur_pos].tiled_index_space().translate_if_possible(
            from_blockid[cur_pos], from_indep_vec,
            to_label[cur_pos].tiled_index_space(), to_indep_vec);

       
        if(!valid) {
            return {to_blockid, false};
        } else {
            // std::cout << "to_blockid : " << i << " - " << compute_order[i] << std::endl;
            // std::cout << "to_id: " << to_id << std::endl;
            to_blockid[compute_order[i]] = to_id;
        }
    }
    return {to_blockid, true};
}

inline void update_labels(IndexLabelVec& labels) {
    EXPECTS(!labels.empty());
    auto dep_map            = construct_dep_map(labels);
    bool has_new_lbl        = false;
    bool have_other_dep_lbl = false;
    std::map<TiledIndexLabel, TiledIndexLabel> new_lbl_map;
    // construct new tis and lbls for dependent labels without secondary labels
    for(size_t i = 0; i < labels.size(); i++) {
        auto lbl     = labels[i];
        auto lbl_tis = lbl.tiled_index_space();
        
        if(lbl_tis.is_dependent() && lbl.secondary_labels().size() == 0) {
            if(new_lbl_map.find(lbl) == new_lbl_map.end()) {
               new_lbl_map[lbl] = lbl_tis.parent_tis().label();
            }
            labels[i]    = new_lbl_map[lbl];
            has_new_lbl  = true;
        } else if(lbl_tis.is_dependent() && lbl.secondary_labels().size() > 0) {
            have_other_dep_lbl = true;
        }
    }

    if(has_new_lbl && have_other_dep_lbl) {
        // Update dependent labels if a new label is created
        for(size_t i = 0; i < labels.size(); i++) {
            auto lbl            = labels[i];
            const auto& lbl_tis = lbl.tiled_index_space();
            if(lbl_tis.is_dependent()) {
                auto primary_label    = lbl.primary_label();
                auto secondary_labels = lbl.secondary_labels();
                EXPECTS(!secondary_labels.empty());
                EXPECTS(dep_map[i].size() == secondary_labels.size());
                auto sec_indices = dep_map[i];
                for(size_t j = 0; j < sec_indices.size(); j++) {
                    secondary_labels[j] =
                      labels[sec_indices[j]].primary_label();
                }
                labels[i] = TiledIndexLabel{primary_label, secondary_labels};
            }
        }
    }
}

inline void print_labels(const IndexLabelVec& labels) {
    for(auto& lbl : labels) {
        std::cout << "primary: " << lbl.primary_label().label() << " - secondary: [ ";
        for(const auto& l : lbl.secondary_labels()) {
            std::cout << l.label() << " ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "-------" << std::endl;
}

} // namespace internal

} // namespace tamm

#endif // TAMM_UTILS_HPP_
