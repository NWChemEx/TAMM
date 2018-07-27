#ifndef TAMM_TILED_INDEX_SPACE_HPP_
#define TAMM_TILED_INDEX_SPACE_HPP_

#include "tamm/index_space.hpp"

namespace tamm {

class TiledIndexLabel;

/**
 * @brief TiledIndexSpace class
 *
 */
class TiledIndexSpace {
public:
    TiledIndexSpace()                       = default;
    TiledIndexSpace(const TiledIndexSpace&) = default;
    TiledIndexSpace(TiledIndexSpace&&)      = default;
    TiledIndexSpace& operator=(TiledIndexSpace&&) = default;
    TiledIndexSpace& operator=(const TiledIndexSpace&) = default;
    ~TiledIndexSpace()                                 = default;

    /**
     * @brief Construct a new TiledIndexSpace from
     * a reference IndexSpace and a tile size
     *
     * @param [in] is reference IndexSpace
     * @param [in] input_tile_size tile size (default: 1)
     */
    TiledIndexSpace(const IndexSpace& is, Tile input_tile_size = 1) :
      tiled_info_{std::make_shared<TiledIndexSpace::TiledIndexSpaceInfo>(
        is, input_tile_size, std::vector<Tile>{})},
      root_tiled_info_{tiled_info_} {
        EXPECTS(input_tile_size > 0);

        // construct tiling for named subspaces
        tile_named_subspaces(is);
    }

    /**
     * @brief Construct a new TiledIndexSpace from a reference
     * IndexSpace and varying tile sizes
     *
     * @param [in] is
     * @param [in] input_tile_sizes
     */
    TiledIndexSpace(const IndexSpace& is,
                    const std::vector<Tile>& input_tile_sizes) :
      tiled_info_{std::make_shared<TiledIndexSpace::TiledIndexSpaceInfo>(
        is, 0, input_tile_sizes)},
      root_tiled_info_{tiled_info_} {
        for(const auto& in_tsize : input_tile_sizes) { EXPECTS(in_tsize > 0); }
    }

    /**
     * @brief Construct a new TiledIndexSpace object from
     * a sub-space of a reference TiledIndexSpace
     *
     * @param [in] t_is reference TiledIndexSpace
     * @param [in] range Range of the reference TiledIndexSpace
     */
    TiledIndexSpace(const TiledIndexSpace& t_is, const Range& range) :
      TiledIndexSpace{t_is, construct_index_vector(range)} {}

    /**
     * @brief Construct a new TiledIndexSpace object from
     * a sub-space of a reference TiledIndexSpace
     *
     * @param [in] t_is reference TiledIndexSpace
     * @param [in] indices set of indices of the reference TiledIndexSpace
     */
    TiledIndexSpace(const TiledIndexSpace& t_is, const IndexVector& indices) :
      root_tiled_info_{t_is.root_tiled_info_} {
        IndexVector new_indices, new_offsets;

        for(const auto& idx : indices) {
            new_indices.push_back(
              t_is.info_translate(idx, (*root_tiled_info_.lock())));
        }

        new_offsets.push_back(0);
        for(const auto& idx : indices) {
            new_offsets.push_back(new_offsets.back() +
                                  root_tiled_info_.lock()->tile_size(idx));
        }

        tiled_info_ = std::make_shared<TiledIndexSpaceInfo>(
          (*t_is.tiled_info_), new_offsets, new_indices);
    }

    /**
     * @brief Construct a new Tiled Index Space object from a tiled dependent
     *        index space
     *
     * @param [in] t_is parent tiled index space
     * @param [in] dep_map dependency map
     */
    TiledIndexSpace(const TiledIndexSpace& t_is,
                    const std::map<IndexVector, TiledIndexSpace>& dep_map) :
      tiled_info_{std::make_shared<TiledIndexSpace::TiledIndexSpaceInfo>(
        (*t_is.tiled_info_), dep_map)},
      root_tiled_info_{t_is.root_tiled_info_} {}

    // /**
    //  * @brief Construct a new TiledIndexSpace object from a reference
    //  * TiledIndexSpace and named subspace
    //  *
    //  * @param [in] t_is reference TiledIndexSpace
    //  * @param [in] id name string for the corresponding subspace
    //  * @param [in] size Tile size (default: 1)
    //  */
    // TiledIndexSpace(const TiledIndexSpace& t_is, const std::string& id,
    //                 Tile size = 1) :
    //   TiledIndexSpace((*t_is.tiled_info_).is_(id), size) {}

    /**
     * @brief Get a TiledIndexLabel for a specific subspace of the
     * TiledIndexSpace
     *
     * @param [in] id string name for the subspace
     * @param [in] lbl an integer value for associated Label
     * @returns a TiledIndexLabel associated with a TiledIndexSpace
     */
    TiledIndexLabel label(std::string id, Label lbl = Label{0}) const;

    TiledIndexLabel label(Label lbl = Label{0}) const;

    /**
     * @brief Construct a tuple of TiledIndexLabel given a count, subspace name
     * and a starting integer Label
     *
     * @tparam c_lbl count of labels
     * @param [in] id name string associated to the subspace
     * @param [in] start starting label value
     * @returns a tuple of TiledIndexLabel
     */
    template<std::size_t c_lbl>
    auto labels(std::string id, Label start = Label{0}) const {
        return labels_impl(id, start, std::make_index_sequence<c_lbl>{});
    }

    /**
     * @brief operator () overload for accessing a (sub)TiledIndexSpace with the
     * given subspace name string
     *
     * @param [in] id name string associated to the subspace
     * @returns a (sub)TiledIndexSpace associated with the subspace name string
     */
    TiledIndexSpace operator()(std::string id) const {
        if(id == "all") { return (*this); }

        return tiled_info_->tiled_named_subspaces_.at(id);
    }

    /**
     * @brief Operator overload for getting TiledIndexSpace from dependent
     * relation map
     *
     * @param [in] dep_idx_vec set of dependent index values
     * @returns TiledIndexSpace from the relation map
     */
    TiledIndexSpace operator()(const IndexVector& dep_idx_vec = {}) const {
        if(dep_idx_vec.empty()) { return (*this); }
        const auto& t_dep_map = tiled_info_->tiled_dep_map_;
        EXPECTS(t_dep_map.find(dep_idx_vec) != t_dep_map.end());
        return t_dep_map.at(dep_idx_vec);
    }

    /**
     * @brief Iterator accessor to the start of the reference IndexSpace
     *
     * @returns a const_iterator to an Index at the first element of the
     * IndexSpace
     */
    IndexIterator begin() const { return tiled_info_->simple_vec_.begin(); }

    /**
     * @brief Iterator accessor to the end of the reference IndexSpace
     *
     * @returns a const_iterator to an Index at the size-th element of the
     * IndexSpace
     */
    IndexIterator end() const { return tiled_info_->simple_vec_.end(); }

    /**
     * @brief Iterator accessor to the first Index element of a specific block
     *
     * @param [in] blck_ind Index of the block to get const_iterator
     * @returns a const_iterator to the first Index element of the specific
     * block
     */
    IndexIterator block_begin(Index blck_ind) const {
        EXPECTS(blck_ind <= size());
        return tiled_info_->is_.begin() + tiled_info_->tile_offsets_[blck_ind];
    }
    /**
     * @brief Iterator accessor to the last Index element of a specific block
     *
     * @param [in] blck_ind Index of the block to get const_iterator
     * @returns a const_iterator to the last Index element of the specific
     * block
     */
    IndexIterator block_end(Index blck_ind) const {
        EXPECTS(blck_ind <= size());
        return tiled_info_->is_.begin() +
               tiled_info_->tile_offsets_[blck_ind + 1];
    }

    /**
     * @brief Boolean method for checking if given TiledIndexSpace is identical
     * to this TiledIndexSpace
     *
     * @param [in] rhs reference TiledIndexSpace
     * @returns true if the Tile size and the reference IndexSpace is equal
     */
    bool is_identical(const TiledIndexSpace& rhs) const {
        //@todo return std::tie(tile_size_, is_) == std::tie(rhs.tile_size_,
        // rhs.is_);
        // return is_ == rhs.is_;
        return tiled_info_ == rhs.tiled_info_;
    }

    bool is_compatible_with(const TiledIndexSpace& tis) const {
        return (this->root_tiled_info_.lock() == tis.root_tiled_info_.lock());
    }

    /**
     * @brief Boolean method for checking if given TiledIndexSpace
     * is a subspace of this TiledIndexSpace
     *
     * @param [in] rhs reference TiledIndexSpace
     * @returns true if the Tile size and the reference IndexSpace is equal
     */
    bool is_less_than(const TiledIndexSpace& rhs) const {
        //@todo return (tile_size_ == rhs.tile_size_) &&
        //(is_.is_less_than(rhs.is_));
        // return is_.is_less_than(rhs.is_);
        return tiled_info_ < rhs.tiled_info_;
    }

    /**
     * @brief Accessor methods to Spin value associated with the input Index
     *
     * @param [in] idx input Index value
     * @returns associated Spin value for the input Index value
     */
    Spin spin(Index idx) const { return tiled_info_->is_.spin(idx); }

    /**
     * @brief Accessor methods to Spatial value associated with the input Index
     *
     * @param [in] idx input Index value
     * @returns associated Spatial value for the input Index value
     */
    Spatial spatial(Index idx) const { return tiled_info_->is_.spatial(idx); }

    /**
     * @brief Accessor method for the set of Ranges associated with a Spin value
     *
     * @param [in] spin input Spin value
     * @returns a vector of Ranges associated with the input Spin value
     */
    std::vector<Range> spin_ranges(Spin spin) const {
        return tiled_info_->is_.spin_ranges(spin);
    }

    /**
     * @brief Accessor method for the set of Ranges associated with a Spatial
     * value
     *
     * @param [in] spatial input Spatial value
     * @returns a vector of Ranges associated with the input Spatial value
     */
    std::vector<Range> spatial_ranges(Spatial spatial) const {
        return tiled_info_->is_.spatial_ranges(spatial);
    }

    /**
     * @brief Boolean method for checking if an IndexSpace has SpinAttribute
     *
     * @returns true if there is a SpinAttribute associated with the IndexSpace
     */
    bool has_spin() const { return tiled_info_->is_.has_spin(); }

    /**
     * @brief Boolean method for checking if an IndexSpace has SpatialAttribute
     *
     * @return true if there is a SpatialAttribute associated with the
     * IndexSpace
     */
    bool has_spatial() const { return tiled_info_->is_.has_spatial(); }

    /**
     * @brief Getter method for the reference IndexSpace
     *
     * @return IndexSpace reference
     */
    const IndexSpace& index_space() const { return tiled_info_->is_; }

    /**
     * @brief Get the number of tiled index blocks in TiledIndexSpace
     *
     * @return size of TiledIndexSpace
     */
    std::size_t size() const { return tiled_info_->tile_offsets_.size() - 1; }

    /**
     * @brief Get the max. number of tiled index blocks in TiledIndexSpace
     *
     * @return max size of TiledIndexSpace
     */
    std::size_t max_size() const { return tiled_info_->max_size(); }

    /**
     * @brief Get the tile size for the index blocks
     *
     * @return Tile size
     */
    std::size_t tile_size(Index i) const { return tiled_info_->tile_size(i); }

    /**
     * @brief Get the input tile size for tiled index space
     *
     * @returns input tile size
     */
    Tile input_tile_size() const { return tiled_info_->input_tile_size_; }

    /**
     * @brief Get the input tile size for tiled index space
     *
     * @returns input tile sizes
     */
    const std::vector<Tile>& input_tile_sizes() const {
        return tiled_info_->input_tile_sizes_;
    }

    /**
     * @brief Get tiled dependent spaces map
     *
     * @returns a map from dependent indicies to tiled index spaces
     */
    const std::map<IndexVector, TiledIndexSpace>& tiled_dep_map() const {
        return tiled_info_->tiled_dep_map_;
    }

    /**
     * @brief Accessor to tile offsets
     *
     * @return Tile offsets
     */
    const IndexVector& tile_offsets() const {
        return tiled_info_->tile_offsets_;
    }

    /**
     * @brief Accessor to tile offset with index id
     *
     * @param [in] id index for the tile offset
     * @returns offset for the corresponding index
     */
    const std::size_t tile_offset(size_t id) const {
        EXPECTS(id >= 0 && id < tiled_info_->simple_vec_.size());

        return tile_offsets()[id];
    }

    /**
     * @brief Translate id to another tiled index space
     *
     * @param [in] id index to be translated
     * @param [in] new_tis reference index space to translate to
     * @returns an index from the new_tis that corresponds to [in] id
     */
    std::size_t translate(size_t id, const TiledIndexSpace& new_tis) const {
        EXPECTS(id >= 0 && id < tiled_info_->simple_vec_.size());

        auto it = std::find(new_tis.begin(), new_tis.end(),
                            tiled_info_->simple_vec_[id]);
        EXPECTS(it != new_tis.end());

        return (*it);
    }

    /**
     * @brief Check if reference index space is a dependent index space
     *
     * @returns true if the reference index space is a dependent index space
     */
    const bool is_dependent() const { return tiled_info_->is_.is_dependent(); }

    /**
     * @brief Equality comparison operator
     *
     * @param [in] lhs Left-hand side
     * @param [in] rhs Right-hand side
     *
     * @return true if lhs == rhs
     */
    friend bool operator==(const TiledIndexSpace& lhs,
                           const TiledIndexSpace& rhs);

    /**
     * @brief Ordering comparison operator
     *
     * @param [in] lhs Left-hand side
     * @param [in] rhs Right-hand side
     *
     * @return true if lhs < rhs
     */
    friend bool operator<(const TiledIndexSpace& lhs,
                          const TiledIndexSpace& rhs);

    /**
     * @brief Inequality comparison operator
     *
     * @param [in] lhs Left-hand side
     * @param [in] rhs Right-hand side
     *
     * @return true if lhs != rhs
     */
    friend bool operator!=(const TiledIndexSpace& lhs,
                           const TiledIndexSpace& rhs);

    /**
     * @brief Ordering comparison operator
     *
     * @param [in] lhs Left-hand side
     * @param [in] rhs Right-hand side
     *
     * @return true if lhs > rhs
     */
    friend bool operator>(const TiledIndexSpace& lhs,
                          const TiledIndexSpace& rhs);

    /**
     * @brief Ordering comparison operator
     *
     * @param [in] lhs Left-hand side
     * @param [in] rhs Right-hand side
     *
     * @return true if lhs <= rhs
     */
    friend bool operator<=(const TiledIndexSpace& lhs,
                           const TiledIndexSpace& rhs);

    /**
     * @brief Ordering comparison operator
     *
     * @param [in] lhs Left-hand side
     * @param [in] rhs Right-hand side
     *
     * @return true if lhs >= rhs
     */
    friend bool operator>=(const TiledIndexSpace& lhs,
                           const TiledIndexSpace& rhs);

protected:
    struct TiledIndexSpaceInfo {
        /* data */
        IndexSpace is_;        /**< The index space being tiled*/
        Tile input_tile_size_; /**< User-specified tile size*/
        std::vector<Tile>
          input_tile_sizes_;       /**< User-specified multiple tile sizes*/
        IndexVector tile_offsets_; /**< Tile offsets */
        IndexVector simple_vec_;   /**< vector where at(i) = i*/
        std::map<IndexVector, TiledIndexSpace>
          tiled_dep_map_; /**< Tiled dependency map for dependent index spaces*/
        std::map<std::string, TiledIndexSpace>
          tiled_named_subspaces_; /**< Tiled named subspaces map string ids*/

        TiledIndexSpaceInfo(IndexSpace is, Tile input_tile_size,
                            const std::vector<Tile>& input_tile_sizes) :
          is_{is},
          input_tile_size_{input_tile_size},
          input_tile_sizes_{input_tile_sizes} {
            if(input_tile_sizes.size() > 0) {
                // construct indices with set of tile sizes
                tile_offsets_ = construct_tiled_indices(is, input_tile_sizes);
                // construct dependency according to tile sizes
                for(const auto& kv : is.map_tiled_index_spaces()) {
                    tiled_dep_map_.insert(
                      std::pair<IndexVector, TiledIndexSpace>{
                        kv.first,
                        TiledIndexSpace{kv.second, input_tile_sizes}});
                }
                // in case of multiple tile sizes no named spacing carried.
            } else {
                // construct indices with input tile size
                tile_offsets_ = construct_tiled_indices(is, input_tile_size);
                // construct dependency according to tile size
                for(const auto& kv : is.map_tiled_index_spaces()) {
                    tiled_dep_map_.insert(
                      std::pair<IndexVector, TiledIndexSpace>{
                        kv.first, TiledIndexSpace{kv.second, input_tile_size}});
                }
            }

            if(!is.is_dependent()) {
                for(Index i = 0; i < tile_offsets_.size() - 1; i++) {
                    simple_vec_.push_back(i);
                }
            }
        }

        TiledIndexSpaceInfo(const TiledIndexSpaceInfo& parent,
                            const IndexVector& offsets,
                            const IndexVector& indices) :
          is_{parent.is_},
          input_tile_size_{parent.input_tile_size_},
          input_tile_sizes_{parent.input_tile_sizes_},
          tile_offsets_{offsets},
          simple_vec_{indices} {}

        TiledIndexSpaceInfo(
          const TiledIndexSpaceInfo& parent,
          const std::map<IndexVector, TiledIndexSpace>& dep_map) :
          is_{parent.is_},
          input_tile_size_{parent.input_tile_size_},
          input_tile_sizes_{parent.input_tile_sizes_},
          tiled_dep_map_{dep_map} {}

        /**
         * @brief Construct starting and ending indices of each tile with
         * respect to the named subspaces
         *
         * @param [in] is reference IndexSpace
         * @param [in] size Tile size value
         * @returns a vector of indices corresponding to the start and end of
         * each tile
         */
        IndexVector construct_tiled_indices(const IndexSpace& is,
                                            Tile tile_size) {
            if(is.is_dependent()) { return {}; }

            if(is.size() == 0) { return {0}; }

            IndexVector boundries, ret;
            // Get lo and hi for each named subspace ranges
            for(const auto& kv : is.get_named_ranges()) {
                for(const auto& range : kv.second) {
                    boundries.push_back(range.lo());
                    boundries.push_back(range.hi());
                }
            }

            // Get SpinAttribute boundries
            if(is.has_spin()) {
                for(const auto& kv : is.get_spin().get_map()) {
                    for(const auto& range : kv.second) {
                        boundries.push_back(range.lo());
                        boundries.push_back(range.hi());
                    }
                }
            }
            // Get SpinAttribute boundries
            if(is.has_spatial()) {
                for(const auto& kv : is.get_spatial().get_map()) {
                    for(const auto& range : kv.second) {
                        boundries.push_back(range.lo());
                        boundries.push_back(range.hi());
                    }
                }
            }

            // If no boundry clean split with respect to tile size
            if(boundries.empty()) {
                // add starting indices
                for(size_t i = 0; i < is.size(); i += tile_size) {
                    ret.push_back(i);
                }
                // add size of IndexSpace for the last block
                ret.push_back(is.size());
            } else { // Remove duplicates
                std::sort(boundries.begin(), boundries.end());
                auto last = std::unique(boundries.begin(), boundries.end());
                boundries.erase(last, boundries.end());
                // Construct start indices for blocks according to boundries.
                std::size_t i = 0;
                std::size_t j = (i == boundries[0]) ? 1 : 0;

                while(i < is.size()) {
                    ret.push_back(i);
                    i = (i + tile_size >= boundries[j]) ? boundries[j++] :
                                                          (i + tile_size);
                }
                // add size of IndexSpace for the last block
                ret.push_back(is.size());
            }

            return ret;
        }

        IndexVector construct_tiled_indices(const IndexSpace& is,
                                            const std::vector<Tile>& tiles) {
            if(is.is_dependent()) { return {}; }

            if(is.size() == 0) { return {0}; }
            // Check if sizes match
            EXPECTS(is.size() == [&tiles]() {
                size_t ret = 0;
                for(const auto& var : tiles) { ret += var; }
                return ret;
            }());

            IndexVector ret, boundries;

            if(is.has_spin()) {
                auto spin_map = is.get_spin().get_map();
                for(const auto& kv : spin_map) {
                    for(const auto& range : kv.second) {
                        boundries.push_back(range.lo());
                        boundries.push_back(range.hi());
                    }
                }
            }

            if(is.has_spatial()) {
                auto spatial_map = is.get_spatial().get_map();
                for(const auto& kv : spatial_map) {
                    for(const auto& range : kv.second) {
                        boundries.push_back(range.lo());
                        boundries.push_back(range.hi());
                    }
                }
            }

            if(is.get_named_ranges().empty()) {
                for(const auto& kv : is.get_named_ranges()) {
                    for(const auto& range : kv.second) {
                        boundries.push_back(range.lo());
                        boundries.push_back(range.hi());
                    }
                }
            }

            // add starting indices
            size_t j = 0;
            for(size_t i = 0; i < is.size(); i += tiles[j++]) {
                ret.push_back(i);
            }
            // add size of IndexSpace for the last block
            ret.push_back(is.size());

            if(!(boundries.empty())) {
                std::sort(boundries.begin(), boundries.end());
                auto last = std::unique(boundries.begin(), boundries.end());
                boundries.erase(last, boundries.end());
                // check if there is any mismatch between boudries and generated
                // start indices
                for(auto& bound : boundries) {
                    EXPECTS(std::binary_search(ret.begin(), ret.end(), bound));
                }
            }

            return ret;
        }

        std::size_t tile_size(Index i) const {
            EXPECTS(i >= 0 && i < tile_offsets_.size());
            return tile_offsets_[i + 1] - tile_offsets_[i];
        }

        std::size_t max_size() const {
            std::size_t ret = 0;
            if(tiled_dep_map_.empty()) {
                return tile_offsets_.size();
            } else {
                for(const auto& kv : tiled_dep_map_) {
                    if(ret < kv.second.max_size()){
                        ret = kv.second.max_size();
                    }
                }
            }

            return ret;
        }
    };

    std::shared_ptr<TiledIndexSpaceInfo> tiled_info_;
    std::weak_ptr<TiledIndexSpaceInfo> root_tiled_info_;

    std::size_t info_translate(size_t id,
                               const TiledIndexSpaceInfo& new_info) const {
        EXPECTS(id >= 0 && id < tiled_info_->simple_vec_.size());

        auto it =
          std::find(new_info.simple_vec_.begin(), new_info.simple_vec_.end(),
                    tiled_info_->simple_vec_[id]);
        EXPECTS(it != new_info.simple_vec_.end());

        return (*it);
    }

    void set_root(const std::shared_ptr<TiledIndexSpaceInfo>& root) {
        root_tiled_info_ = root;
    }

    void set_tiled_info(
      const std::shared_ptr<TiledIndexSpaceInfo>& tiled_info) {
        tiled_info_ = tiled_info;
    }

    void tile_named_subspaces(const IndexSpace& is) {
        // construct tiled spaces for named subspaces
        for(const auto& str_subis : is.map_named_sub_index_spaces()) {
            auto named_is = str_subis.second;

            IndexVector indices;

            for(const auto& idx : named_is) {
                // find position in the parent index space
                size_t pos = is.find_pos(idx);
                // named subspace should always find it
                EXPECTS(pos >= 0);

                size_t tile_idx     = 0;
                const auto& offsets = tiled_info_->tile_offsets_;
                // find in which tiles in the parent it would be
                for(Index i = 0; pos >= offsets[i]; i++) { tile_idx = i; }
                indices.push_back(tile_idx);
            }
            // remove duplicates from the indices
            std::sort(indices.begin(), indices.end());
            auto last = std::unique(indices.begin(), indices.end());
            indices.erase(last, indices.end());

            IndexVector new_offsets;

            new_offsets.push_back(0);
            for(const auto& idx : indices) {
                new_offsets.push_back(new_offsets.back() +
                                      root_tiled_info_.lock()->tile_size(idx));
            }
            TiledIndexSpace tempTIS{};
            tempTIS.set_tiled_info(std::make_shared<TiledIndexSpaceInfo>(
              (*tiled_info_), new_offsets, indices));
            tempTIS.set_root(tiled_info_);

            tiled_info_->tiled_named_subspaces_.insert(
              {str_subis.first, tempTIS});
        }
    }

    template<std::size_t... Is>
    auto labels_impl(std::string id, Label start,
                     std::index_sequence<Is...>) const;

}; // class TiledIndexSpace

// Comparison operator implementations
inline bool operator==(const TiledIndexSpace& lhs, const TiledIndexSpace& rhs) {
    return lhs.is_identical(rhs);
}

inline bool operator<(const TiledIndexSpace& lhs, const TiledIndexSpace& rhs) {
    return lhs.is_less_than(rhs);
}

inline bool operator!=(const TiledIndexSpace& lhs, const TiledIndexSpace& rhs) {
    return !(lhs == rhs);
}

inline bool operator>(const TiledIndexSpace& lhs, const TiledIndexSpace& rhs) {
    return (rhs < lhs);
}

inline bool operator<=(const TiledIndexSpace& lhs, const TiledIndexSpace& rhs) {
    return (lhs < rhs) || (lhs == rhs);
}

inline bool operator>=(const TiledIndexSpace& lhs, const TiledIndexSpace& rhs) {
    return (rhs <= lhs);
}

/**
 * @brief Index label to index into tensors. The labels used by the user need to
 * be positive.
 *
 */
class TiledIndexLabel {
public:
    // Constructor
    TiledIndexLabel() = default;

    TiledIndexLabel(const TiledIndexSpace& t_is, Label lbl = 0,
                    const std::vector<TiledIndexLabel> dep_labels = {}) :
      tis_{t_is},
      label_{lbl},
      dep_labels_{dep_labels} {}

    TiledIndexLabel(const TiledIndexLabel& t_il,
                    const std::vector<TiledIndexLabel>& dep_labels) :
      tis_{t_il.tis_},
      label_{t_il.label_},
      dep_labels_{dep_labels} {
        EXPECTS(is_compatible_with(tis_));
    }

    // Copy Construtors
    TiledIndexLabel(const TiledIndexLabel&) = default;
    TiledIndexLabel& operator=(const TiledIndexLabel&) = default;

    TiledIndexLabel(TiledIndexLabel&&) = default;
    TiledIndexLabel& operator=(TiledIndexLabel&&) = default;

    // Destructor
    ~TiledIndexLabel() = default;

    TiledIndexLabel operator()() const { return {*this}; }

    template<typename... Args>
    TiledIndexLabel operator()(const TiledIndexLabel& il1, Args... rest) {
        std::vector<TiledIndexLabel> dep_ilv;
        unpack_dep(dep_ilv, il1, rest...);
        return {*this, dep_ilv};
    }

    TiledIndexLabel operator()(const std::vector<TiledIndexLabel>& dep_ilv) {
        return {*this, dep_ilv};
    }

    bool is_identical(const TiledIndexLabel& rhs) const {
        return (std::tie(label_, dep_labels_, tis_) ==
                std::tie(rhs.label_, rhs.dep_labels_, rhs.tis_));
    }

    bool is_less_than(const TiledIndexLabel& rhs) const {
        return (tis_ < rhs.tis_) ||
               ((tis_ == rhs.tis_) && (label_ < rhs.label_));
    }

    Label get_label() const { return label_; }

    /// @todo: this is never called from outside currently, should this be
    /// private and used internally?
    bool is_compatible_with(const TiledIndexSpace& tis) const {
        const auto& key_tiss = tis.index_space().key_tiled_index_spaces();
        EXPECTS(key_tiss.size() == dep_labels().size());
        for(size_t i = 0; i < dep_labels().size(); i++) {
            dep_labels()[i].tiled_index_space().is_compatible_with(key_tiss[i]);
        }
        return true;
    }

    const std::vector<TiledIndexLabel>& dep_labels() const {
        return dep_labels_;
    }

    std::pair<TiledIndexSpace, Label> primary_label() const {
        return {tis_, label_};
    }

    const TiledIndexSpace& tiled_index_space() const { return tis_; }

    // Comparison operators
    friend bool operator==(const TiledIndexLabel& lhs,
                           const TiledIndexLabel& rhs);
    friend bool operator<(const TiledIndexLabel& lhs,
                          const TiledIndexLabel& rhs);
    friend bool operator!=(const TiledIndexLabel& lhs,
                           const TiledIndexLabel& rhs);
    friend bool operator>(const TiledIndexLabel& lhs,
                          const TiledIndexLabel& rhs);
    friend bool operator<=(const TiledIndexLabel& lhs,
                           const TiledIndexLabel& rhs);
    friend bool operator>=(const TiledIndexLabel& lhs,
                           const TiledIndexLabel& rhs);

protected:
    TiledIndexSpace tis_;
    Label label_;
    std::vector<TiledIndexLabel> dep_labels_;

private:
    void unpack_dep(std::vector<TiledIndexLabel>& in_vec) {}

    template<typename... Args>
    void unpack_dep(std::vector<TiledIndexLabel>& in_vec,
                    const TiledIndexLabel& il1, Args... rest) {
        EXPECTS(!(*this).is_identical(il1));

        in_vec.push_back(il1);
        unpack_dep(in_vec, rest...);
    }
}; // class TiledIndexLabel

// Comparison operator implementations
inline bool operator==(const TiledIndexLabel& lhs, const TiledIndexLabel& rhs) {
    return lhs.is_identical(rhs);
}
inline bool operator<(const TiledIndexLabel& lhs, const TiledIndexLabel& rhs) {
    return lhs.is_less_than(rhs);
}

inline bool operator!=(const TiledIndexLabel& lhs, const TiledIndexLabel& rhs) {
    return !(lhs == rhs);
}

inline bool operator>(const TiledIndexLabel& lhs, const TiledIndexLabel& rhs) {
    return (rhs < lhs);
}

inline bool operator<=(const TiledIndexLabel& lhs, const TiledIndexLabel& rhs) {
    return (lhs < rhs) || (lhs == rhs);
}

inline bool operator>=(const TiledIndexLabel& lhs, const TiledIndexLabel& rhs) {
    return (rhs <= lhs);
}

/**
 * @brief Vector of tiled index labels
 */
using IndexLabelVec = std::vector<TiledIndexLabel>;

///////////////////////////////////////////////////////////

inline TiledIndexLabel TiledIndexSpace::label(std::string id, Label lbl) const {
    if(id == "all") return TiledIndexLabel{(*this), lbl};
    return TiledIndexLabel{(*this)(id), lbl};
}

template<std::size_t... Is>
auto TiledIndexSpace::labels_impl(std::string id, Label start,
                                  std::index_sequence<Is...>) const {
    return std::make_tuple(label(id, start + Is)...);
}

} // namespace tamm

#endif // TAMM_TILED_INDEX_SPACE_HPP_
