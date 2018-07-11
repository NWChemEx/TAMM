#ifndef TAMM_INDEX_SPACE_HPP_
#define TAMM_INDEX_SPACE_HPP_

#include "tamm/attribute.hpp"
#include "tamm/range.hpp"
#include "tamm/types.hpp"
#include <algorithm>
#include <iterator>
#include <map>
#include <memory>
#include <sstream>

namespace tamm {

class IndexSpaceInterface;

/**
 * @class IndexSpace
 * @brief Main IndexSpace class that clients will be actively
 *        using for constructing Tensors via TiledIndexSpaces.
 *
 * @todo Possibly use named parameters idiom for construction
 */
class IndexSpace {
    public:
    using Iterator = IndexIterator;
    // Constructors
    IndexSpace() = default;

    IndexSpace(const IndexSpace&) = default;
    IndexSpace(IndexSpace&&)      = default;
    ~IndexSpace()                 = default;
    IndexSpace& operator=(const IndexSpace&) = default;
    IndexSpace& operator=(IndexSpace&&) = default;

    /**
     * @brief Construct a new Index Space object using initializer list of
     * indices.
     *
     * @param [in] indices input Index values
     */
    IndexSpace(const std::initializer_list<Index>& indices) :
      IndexSpace{indices, {}, {}, {}} {}

    /**
     * @brief Construct a new Index Space object using a range, named subspaces
     * and attributes
     *
     * @param [in] range Range object that is being used to construct indices
     * @param [in] named_subspaces partition of IndexSpace into named subspaces
     * @param [in] spin map from Spin attribute values to corresponding Ranges
     * in IndexSpace
     * @param [in] spatial map from Spatial attribute values to corresponding
     * Ranges in IndexSpace
     */
    IndexSpace(const Range& range, const NameToRangeMap& named_subspaces = {},
               const AttributeToRangeMap<Spin>& spin       = {},
               const AttributeToRangeMap<Spatial>& spatial = {}) :
      IndexSpace{construct_index_vector(range), named_subspaces, spin,
                 spatial} {}

    /**
     * @brief Construct a new IndexSpace object using a vector of index values
     *
     * @param [in] indices input Index values
     * @param [in] named_subspaces partition of the IndexSpace into named
     * subspaces
     * @param [in] spin map from Spin attribute values to corresponding ranges
     * in the IndexSpace
     * @param [in] spatial map from Spatial attribute values to corresponding
     * ranges in the IndexSpace
     */
    IndexSpace(const IndexVector& indices,
               const NameToRangeMap& named_subspaces       = {},
               const AttributeToRangeMap<Spin>& spin       = {},
               const AttributeToRangeMap<Spatial>& spatial = {});

    /**
     * @brief Construct a new (Sub-)IndexSpace object by getting a range from
     * the reference index space
     *
     * Sub-space. no inherited named subspaces from the reference index space
     * To iterate over the ranges into which an attribute partitions this
     * index space, the parent space's attributes are accessed by using the
     * input Range.
     *
     * @param [in] is reference IndexSpace
     * @param [in] range range of indices get from the reference IndexSpace
     * @param [in] named_subspaces map from strings to (sub-)IndexSpace
     */
    IndexSpace(const IndexSpace& is, const Range& range,
               const NameToRangeMap& named_subspaces = {});
    /**
     * @brief Construct a new (Sub-)IndexSpace object by getting a set of
     * indices from the reference index space
     *
     * @param [in] is reference IndexSpace
     * @param [in] indices set of indices used for constructing the sub space
     * @param [in] named_subspaces map from strings to (sub-)IndexSpace object
     */
    IndexSpace(const IndexSpace& is, const IndexVector& indices,
               const NameToRangeMap& named_subspaces = {});

    /**
     * @brief Construct a new (Aggregated) Index Space object by aggregating
     * other index spaces
     *
     * Aggregate. named subspaces  and attributes from all spaces in
     * input IndexSpaces with a non-empty name/attributes are accessible through
     * the reference index spaces
     *
     * @todo we could have functions to get "named" subspaces by position.
     * Basically fn(i) returns spaces[i].
     *
     * @param [in] spaces vector of reference IndexSpaces that are being
     * aggregated
     * @param [in] names strings associated with each reference IndexSpace
     * @param [in] named_subspaces additional named subspaces by a map from
     * strings to vector of Ranges
     * @param [in] subspace_references additional named subspaces defined over
     * reference index spaces by a map from strings to ':' separated strings
     */
    IndexSpace(const std::vector<IndexSpace>& spaces,
               const std::vector<std::string>& names = {},
               const NameToRangeMap& named_subspaces = {},
               const std::map<std::string, std::vector<std::string>>&
                 subspace_references = {});

    /**
     * @brief Construct a new (Dependent) IndexSpace object using a vector of
     * dependent index spaces.
     *
     * Dependent: named subspaces and attributes for all dependent spaces in
     * input dependent IndexSpaces with a non-empty name/attributes are
     * accessible through the dependent index spaces
     *
     * @param [in] indep_spaces dependent IndexSpaces used for construction
     * @param [in] dep_space_relation relation between each set of indices on
     * dependent IndexSpaces
     */
    IndexSpace(const std::vector<IndexSpace>& indep_spaces,
               const std::map<IndexVector, IndexSpace>& dep_space_relation);
    /**
     * @brief Construct a new (Dependent) IndexSpace object using a vector of
     * dependent index spaces and a specific reference index space
     *
     * @param [in] indep_spaces dependent IndexSpaces used for construction
     * @param [in] ref_space reference IndexSpace
     * @param [in] dep_space_relation relation between each set of indices on
     * dependent IndexSpace
     */
    IndexSpace(const std::vector<IndexSpace>& indep_spaces,
               const IndexSpace& ref_space,
               const std::map<IndexVector, IndexSpace>& dep_space_relation);

    IndexSpace(const std::vector<IndexSpace>& indep_spaces,
               const std::map<Range, IndexSpace>& dep_space_relation);

    IndexSpace(const std::vector<IndexSpace>& indep_spaces,
               const IndexSpace& ref_space,
               const std::map<Range, IndexSpace>& dep_space_relation);
    /**
     * @brief Construct a new Index Space object by using a shared_ptr
     *
     * Used for constructing reference IndexSpace object from the
     * implementations
     *
     * @param [in] impl input shared_ptr to IndexSpaceInterface implementation
     */
    IndexSpace(const std::shared_ptr<IndexSpaceInterface> impl) : impl_{impl} {}

    // Index Accessors
    Index index(Index i, const IndexVector& indep_index = {});
    Index operator[](Index i) const;

    // Subspace Accessors
    IndexSpace operator()(const IndexVector& indep_index = {}) const;
    IndexSpace operator()(const std::string& named_subspace_id) const;

    // Iterators
    IndexIterator begin() const;
    IndexIterator end() const;

    // Size of this index space
    std::size_t size() const;

    // Maximum size of this index space for any dependent index
    std::size_t max_size() const;

    // Attribute Accessors
    Spin spin(Index idx) const;
    Spatial spatial(Index idx) const;

    std::vector<Range> spin_ranges(Spin spin) const;
    std::vector<Range> spatial_ranges(Spatial spatial) const;

    bool has_spin() const;
    bool has_spatial() const;

    const NameToRangeMap& get_named_ranges() const;

    IndexSpace root_index_space() const;

    bool is_identical(const IndexSpace& rhs) const {
        return impl_ == rhs.impl_;
    }

    bool is_less_than(const IndexSpace& rhs) const { return impl_ < rhs.impl_; }

    // @todo Re-visit later
    bool is_compatible(const IndexSpace& rhs) const {
        return is_identical(rhs);
    }

    bool is_identical_reference(const IndexSpace& rhs) const {
        return (*this).root_index_space().is_identical(rhs.root_index_space());
    }

    bool is_compatible_reference(const IndexSpace& rhs) const {
        return (*this).root_index_space().is_compatible(rhs.root_index_space());
    }

    SpinAttribute get_spin() const;
    SpatialAttribute get_spatial() const;

    // Comparison operators
    friend bool operator==(const IndexSpace& lhs, const IndexSpace& rhs);
    friend bool operator<(const IndexSpace& lhs, const IndexSpace& rhs);
    friend bool operator!=(const IndexSpace& lhs, const IndexSpace& rhs);
    friend bool operator>(const IndexSpace& lhs, const IndexSpace& rhs);
    friend bool operator<=(const IndexSpace& lhs, const IndexSpace& rhs);
    friend bool operator>=(const IndexSpace& lhs, const IndexSpace& rhs);

    protected:
    std::shared_ptr<IndexSpaceInterface> impl_;
}; // IndexSpace

class TiledIndexLabel;
/**
 * @brief
 *
 */
class TiledIndexSpace {
    public:
    // Ctors
    TiledIndexSpace() = default;

    /**
     * @brief Construct a new TiledIndexSpace object from a reference
     * IndexSpace and a tile size
     *
     * @param [in] is reference IndexSpace
     * @param [in] size tile size (default: 1)
     */
    TiledIndexSpace(const IndexSpace& is, Tile tile_size = 1) :
      is_{is},
      tile_size_{tile_size},
      tiled_indices_{construct_tiled_indices(is, tile_size)} {}

    /**
     * @brief Constructor with multiple tile sizes
     *
     * @todo Implement
     *
     * @param [in] is
     * @param [in] sizes
     */
    TiledIndexSpace(const IndexSpace& is, const std::vector<Tile>& sizes) :
      is_{is},
      sizes_{sizes},
      tiled_indices_{construct_tiled_indices(is, sizes)} {}

    /**
     * @brief Construct a new TiledIndexSpace object from a sub-space of a
     * reference TiledIndexSpace
     *
     * @param [in] t_is reference TiledIndexSpace
     * @param [in] range Range of the reference TiledIndexSpace
     * @param [in] size Tile size (default: 1)
     */
    TiledIndexSpace(const TiledIndexSpace& t_is, const Range& range,
                    Tile size = 1) :
      TiledIndexSpace(IndexSpace{t_is.is_, range}, size) {}

    /**
     * @brief Construct a new TiledIndexSpace object from a reference
     * TiledIndexSpace and named subspace
     *
     * @param [in] t_is reference TiledIndexSpace
     * @param [in] id name string for the corresponding subspace
     * @param [in] size Tile size (default: 1)
     */
    TiledIndexSpace(const TiledIndexSpace& t_is, const std::string& id,
                    Tile size = 1) :
      TiledIndexSpace(t_is.is_(id), size) {}

    // Copy Ctors
    TiledIndexSpace(const TiledIndexSpace&) = default;
    TiledIndexSpace& operator=(const TiledIndexSpace&) = default;

    // Dtor
    ~TiledIndexSpace() = default;

    /**
     * @brief Get a TiledIndexLabel for a specific subspace of the
     * TiledIndexSpace
     *
     * @param [in] id string name for the subspace
     * @param [in] lbl an integer value for associated Label
     * @returns a TiledIndexLabel associated with a TiledIndexSpace
     */
    TiledIndexLabel label(std::string id, Label lbl = Label{0}) const;

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
        return TiledIndexSpace((*this), id);
    }

    /**
     * @brief Iterator accessor to the start of the reference IndexSpace
     *
     * @returns a const_iterator to an Index at the first element of the
     * IndexSpace
     */
    IndexIterator begin() const { return tiled_indices_.begin(); }

    /**
     * @brief Iterator accessor to the end of the reference IndexSpace
     *
     * @returns a const_iterator to an Index at the size-th element of the
     * IndexSpace
     */
    IndexIterator end() const { return tiled_indices_.end() - 1; }

    /**
     * @brief Iterator accessor to the first Index element of a specific block
     *
     * @param [in] blck_ind Index of the block to get const_iterator
     * @returns a const_iterator to the first Index element of the specific
     * block
     */
    IndexIterator block_begin(Index blck_ind) const {
        EXPECTS(blck_ind <= size());
        return is_.begin() + tiled_indices_[blck_ind];
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
        return is_.begin() + tiled_indices_[blck_ind + 1];
    }

    /**
     * @brief Boolean method for checking if given TiledIndexSpace is identical
     * to this TiledIndexSpace
     *
     * @param [in] rhs reference TiledIndexSpace
     * @returns true if the Tile size and the reference IndexSpace is equal
     */
    bool is_identical(const TiledIndexSpace& rhs) const {
        return std::tie(tile_size_, is_) == std::tie(rhs.tile_size_, rhs.is_);
    }

    /**
     * @brief Boolean method for checking if given TiledIndexSpace is subspace
     * of this TiledIndexSpace
     *
     * @param [in] rhs reference TiledIndexSpace
     * @returns true if the Tile size and the reference IndexSpace is equal
     */
    bool is_less_than(const TiledIndexSpace& rhs) const {
        return (tile_size_ == rhs.tile_size_) && (is_.is_less_than(rhs.is_));
    }

    /**
     * @brief Accessor methods to Spin value associated with the input Index
     *
     * @param [in] idx input Index value
     * @returns associated Spin value for the input Index value
     */
    Spin spin(Index idx) const { return is_.spin(idx); }

    /**
     * @brief Accessor methods to Spatial value associated with the input Index
     *
     * @param [in] idx input Index value
     * @returns associated Spatial value for the input Index value
     */
    Spatial spatial(Index idx) const { return is_.spatial(idx); }

    /**
     * @brief Accessor method for the set of Ranges associated with a Spin value
     *
     * @param [in] spin input Spin value
     * @returns a vector of Ranges associated with the input Spin value
     */
    std::vector<Range> spin_ranges(Spin spin) const {
        return is_.spin_ranges(spin);
    }

    /**
     * @brief Accessor method for the set of Ranges associated with a Spatial
     * value
     *
     * @param [in] spatial input Spatial value
     * @returns a vector of Ranges associated with the input Spatial value
     */
    std::vector<Range> spatial_ranges(Spatial spatial) const {
        return is_.spatial_ranges(spatial);
    }

    /**
     * @brief Boolean method for checking if an IndexSpace has SpinAttribute
     *
     * @returns true if there is a SpinAttribute associated with the IndexSpace
     */
    bool has_spin() const { return is_.has_spin(); }

    /**
     * @brief Boolean method for checking if an IndexSpace has SpatialAttribute
     *
     * @returns true if there is a SpatialAttribute associated with the
     * IndexSpace
     */
    bool has_spatial() const { return is_.has_spatial(); }

    /**
     * @brief Getter method for the reference IndexSpace
     *
     * @returns IndexSpace reference
     */
    const IndexSpace& index_space() const { return is_; }

    /**
     * @brief Get the number of tiled index blocks in TiledIndexSpace
     *
     * @returns size of TiledIndexSpace
     */
    std::size_t size() const { return tiled_indices_.size() - 1; }

    /**
     * @brief Get the max. number of tiled index blocks in TiledIndexSpace
     *
     * @returns max size of TiledIndexSpace
     */
    std::size_t max_size() const { return tiled_indices_.size(); }

    /**
     * @brief Get the tile size for the index blocks
     *
     * @returns Tile size
     */
    Tile tile_size() const { return tile_size_; }

    IndexVector tindices() const { return tiled_indices_; }

    std::vector<Tile> tile_sizes() const { return sizes_; }

    // Comparison operators
    friend bool operator==(const TiledIndexSpace& lhs,
                           const TiledIndexSpace& rhs);
    friend bool operator<(const TiledIndexSpace& lhs,
                          const TiledIndexSpace& rhs);
    friend bool operator!=(const TiledIndexSpace& lhs,
                           const TiledIndexSpace& rhs);
    friend bool operator>(const TiledIndexSpace& lhs,
                          const TiledIndexSpace& rhs);
    friend bool operator<=(const TiledIndexSpace& lhs,
                           const TiledIndexSpace& rhs);
    friend bool operator>=(const TiledIndexSpace& lhs,
                           const TiledIndexSpace& rhs);

    protected:
    IndexSpace is_;
    Tile tile_size_;
    IndexVector tiled_indices_;
    std::vector<Tile> sizes_;

    /**
     * @brief Construct starting and ending indices of each tile with respect to
     * the named subspaces
     *
     * @param [in] is reference IndexSpace
     * @param [in] size Tile size value
     * @returns a vector of indices corresponding to the start and end of each
     * tile
     */
    IndexVector construct_tiled_indices(const IndexSpace& is, Tile tile_size) {
        if(is.size() == 0) { return {}; }

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
                i = (i + tile_size > boundries[j]) ? boundries[j++] :
                                                     (i + tile_size);
            }
            // add size of IndexSpace for the last block
            ret.push_back(is.size());
        }

        return ret;
    }

    IndexVector construct_tiled_indices(const IndexSpace& is,
                                        const std::vector<Tile>& tiles) {
        if(is.size() == 0) { return {}; }
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
        for(size_t i = 0; i < is.size(); i += tiles[j++]) { ret.push_back(i); }
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

    template<std::size_t... Is>
    auto labels_impl(std::string id, Label start,
                     std::index_sequence<Is...>) const;

}; // TiledIndexSpace

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
      dep_labels_{dep_labels} {}

    // Copy Construtors
    TiledIndexLabel(const TiledIndexLabel&) = default;
    TiledIndexLabel& operator=(const TiledIndexLabel&) = default;

    // Destructor
    ~TiledIndexLabel() = default;

    TiledIndexLabel operator()(TiledIndexLabel il1) const {
        return TiledIndexLabel{*this, {il1}};
    }
    TiledIndexLabel operator()() const { return {*this}; }

    TiledIndexLabel operator()(TiledIndexLabel il1, TiledIndexLabel il2) const {
        return TiledIndexLabel{*this, {il1, il2}};
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
}; // TiledIndexLabel

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

#endif // TAMM_INDEX_SPACE_HPP_
