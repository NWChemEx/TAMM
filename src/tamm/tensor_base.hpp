#ifndef TAMM_TENSOR_BASE_HPP_
#define TAMM_TENSOR_BASE_HPP_

#include "tamm/errors.hpp"
#include "tamm/index_loop_nest.hpp"

/**
 * @defgroup tensors
 */

namespace tamm {

/**
 * @ingroup tensors
 * @brief Base class for tensors.
 *
 * This class handles the indexing logic for tensors. Memory management is done
 * by subclasses. The class supports MO indices that are permutation symmetric
 * with anti-symmetry.
 *
 * @note In a spin-restricted tensor, a 𝛽𝛽|𝛽𝛽 block is mapped to its
 * corresponding to αα|αα block.
 *
 * @todo For now, we cannot handle tensors in which number of upper
 * and lower indices differ by more than one. This relates to
 * correctly handling spin symmetry.
 * @todo SymmGroup has a different connotation. Change name
 *
 */

class TensorBase {
public:
    // Ctors
    TensorBase() = default;

    /**
     * @brief Construct a new TensorBase object using a vector of
     * TiledIndexSpace objects for each mode of the tensor
     *
     * @param [in] block_indices vector of TiledIndexSpace objects for each mode
     * used to construct the tensor
     */
    TensorBase(const std::vector<TiledIndexSpace>& block_indices) :
      block_indices_{block_indices},
      num_modes_{block_indices.size()} {
        construct_dep_map();
    }

    /**
     * @brief Construct a new TensorBase object using a vector of
     * TiledIndexSpace objects for each mode of the tensor
     *
     * @param [in] lbls vector of tiled index labels used for extracting
     * corresponding TiledIndexSpace objects for each mode used to construct the
     * tensor
     */
    TensorBase(const std::vector<TiledIndexLabel>& lbls) :
      num_modes_{lbls.size()} {
        for(const auto& lbl : lbls) {
            block_indices_.push_back(lbl.tiled_index_space());
            tlabels_.push_back(lbl);
        }
        construct_dep_map();
    }

    /**
     * @brief Construct a new TensorBase object recursively with a set of
     * TiledIndexSpace objects followed by a lambda expression
     *
     * @tparam Ts variadic template for rest of the arguments
     * @param [in] tis TiledIndexSpace object used as a mode
     * @param [in] rest remaining part of the arguments
     */
    template<class... Ts>
    TensorBase(const TiledIndexSpace& tis, Ts... rest) : TensorBase{rest...} {
        block_indices_.insert(block_indices_.begin(), tis);
    }

    /**
     * @brief Construct a new TensorBase object from a single TiledIndexSpace
     * object and a lambda expression
     *
     * @tparam Func template for lambda expression
     * @param [in] tis TiledIndexSpace object used as the mode of the tensor
     * @param [in] func lambda expression
     */
    template<typename Func>
    TensorBase(const TiledIndexSpace& tis, const Func& func) {
        block_indices_.insert(block_indices_.begin(), tis);
    }

    // Dtor
    virtual ~TensorBase(){};

    /**
     * @brief Method for getting the number of modes of the tensor
     *
     * @returns a size_t for the number of nodes of the tensor
     */
    TensorRank num_modes() const { return num_modes_; };

    auto tindices() const { return block_indices_; }

    TAMM_SIZE block_size(const IndexVector& blockid) const {
        size_t ret = 1;
        EXPECTS(blockid.size() == num_modes());
        size_t rank = block_indices_.size();
        for(size_t i = 0; i < rank; i++) {
            IndexVector dep_idx_vals{};
            if(dep_map_.find(i) != dep_map_.end()) {
                for(const auto& pos : dep_map_.at(i)) {
                    dep_idx_vals.push_back(blockid[pos]);
                }
            }
            ret *= block_indices_[i](dep_idx_vals).tile_size(blockid[i]);
        }
        return ret;
    }

    std::vector<size_t> block_dims(const IndexVector& blockid) const {
        std::vector<size_t> ret;
        EXPECTS(blockid.size() == num_modes());
        size_t rank = block_indices_.size();
        for(size_t i = 0; i < rank; i++) {
            IndexVector dep_idx_vals{};
            if(dep_map_.find(i) != dep_map_.end()) {
                for(const auto& pos : dep_map_.at(i)) {
                    dep_idx_vals.push_back(blockid[pos]);
                }
            }
            ret.push_back(block_indices_[i](dep_idx_vals).tile_size(blockid[i]));
        }
        return ret;
    }

    /**
     * @brief Memory allocation method for the tensor object
     *
     */
    // virtual void allocate() = 0;

    /**
     * @brief Memory deallocation method for the tensor object
     *
     */
    // virtual void deallocate() = 0;

    IndexLoopNest loop_nest() const {
        // std::vector<IndexVector> lbloops, ubloops;
        // for(const auto& tis : block_indices_) {
        //     //iterator to indexvector - each index in vec points to begin of
        //     each tile in IS lbloops.push_back(tis.tindices());
        //     ubloops.push_back(tis.tindices());
        // }

        // //scalar??
        // // if(tisv.size() == 0){
        // //     lbloops.push_back({});
        // //     ubloops.push_back({});
        // // }

        std::vector<std::vector<size_t>> indep_indices(num_modes());
        for(const auto& kv : dep_map_) {
            Index key = kv.first;
            const auto& indices = kv.second;
            EXPECTS(key < indep_indices.size());
            EXPECTS(indep_indices[key].size() == 0);
            indep_indices[key] = indices;
        }

        // return IndexLoopNest{block_indices_,lbloops,ubloops,{}};
        return {block_indices_, {}, {}, indep_indices};
    }

    const std::vector<TiledIndexSpace>& tiled_index_spaces() const {
        return block_indices_;
    }

    const std::vector<TiledIndexLabel>& tlabels() const {
        return tlabels_;
    }

    const std::map<size_t, std::vector<size_t>>& dep_map() const { return dep_map_; }

    /// @todo The following methods could be refactored.
    size_t find_dep(const TiledIndexLabel& til) {
        size_t bis = tlabels_.size();
        for(size_t i = 0; i < bis; i++) {
            if(block_indices_[i].is_identical(til.tiled_index_space()) && til == tlabels_[i])
                return i;
        }
        return bis;
    }

    /// @todo refactor
    bool check_duplicates(){
        Index til = tlabels_.size();
        for(Index i1 = 0; i1 < til; i1++) {
            for(Index i2 = i1+1; i2 < til; i2++) {
                auto tl1 = tlabels_[i1];
                auto tl2 = tlabels_[i2];
                EXPECTS(!(tl1.tiled_index_space().is_identical(tl2.tiled_index_space())
                        && tl1 == tl2));
            }
        }
    }

    /// @todo refactor
    void construct_dep_map() {
        check_duplicates();
        size_t til = tlabels_.size();
        for(size_t i = 0; i < til; i++) {
            auto il  = tlabels_[i];
            auto tis = block_indices_[i];
            if(tis.is_dependent()) {
                /// @todo do we need this check here?
                EXPECTS(il.dep_labels().size() ==
                        il.tiled_index_space()
                          .index_space()
                          .num_key_tiled_index_spaces());
                for(auto& dep : il.dep_labels()) {
                    size_t pos = find_dep(dep);
                    EXPECTS(pos != til);
                    if(pos != til) {
                        dep_map_[i].push_back(pos);
                        // if(dep_map_.find(i) != dep_map_.end())
                        //     dep_map_[i].push_back(pos);
                        // else
                        //     dep_map_[i].push_back({pos});// = IndexVector{pos};
                    }
                }
                EXPECTS(dep_map_.find(i) != dep_map_.end());
            }
        }
    }

protected:
    std::vector<TiledIndexSpace> block_indices_;
    Spin spin_total_;
    bool has_spatial_symmetry_;
    bool has_spin_symmetry_;

    TensorRank num_modes_;
    /// When a tensor is constructed using Tiled Index Labels that correspond to
    /// tiled dependent index spaces.
    std::vector<TiledIndexLabel> tlabels_;

    /// Map that maintains position of dependent index space(s) for a given
    /// dependent index space.
    std::map<size_t, std::vector<size_t>> dep_map_;
    // std::vector<IndexPosition> ipmask_;
    // PermGroup perm_groups_;
    // Irrep irrep_;
    // std::vector<SpinMask> spin_mask_;
}; // TensorBase

inline bool operator<=(const TensorBase& lhs, const TensorBase& rhs) {
    return (lhs.tindices() <= rhs.tindices());
    //&& (lhs.nupper_indices() <= rhs.nupper_indices())
    //&& (lhs.irrep() < rhs.irrep())
    //&& (lhs.spin_restricted () < rhs.spin_restricted());
}

inline bool operator==(const TensorBase& lhs, const TensorBase& rhs) {
    return (lhs.tindices() == rhs.tindices());
    //&& (lhs.nupper_indices() == rhs.nupper_indices())
    //&& (lhs.irrep() < rhs.irrep())
    //&& (lhs.spin_restricted () < rhs.spin_restricted());
}

inline bool operator!=(const TensorBase& lhs, const TensorBase& rhs) {
    return !(lhs == rhs);
}

inline bool operator<(const TensorBase& lhs, const TensorBase& rhs) {
    return (lhs <= rhs) && (lhs != rhs);
}

} // namespace tamm

#endif // TAMM_TENSOR_BASE_HPP_
