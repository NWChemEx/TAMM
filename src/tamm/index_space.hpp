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
class TiledIndexSpace;

/**
 * @class IndexSpace
 * @brief Main IndexSpace class that clients will be actively
 *        using for constructing Tensors via TiledIndexSpaces.
 *
 * @todo Possibly use named parameters idiom for construction
 */
class IndexSpace {
public:
    /**
     * @brief Type alias for iterators from this index space
     */
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
    IndexSpace(const std::vector<TiledIndexSpace>& indep_spaces,
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
    IndexSpace(const std::vector<TiledIndexSpace>& indep_spaces,
               const IndexSpace& ref_space,
               const std::map<IndexVector, IndexSpace>& dep_space_relation);

    IndexSpace(const std::vector<TiledIndexSpace>& indep_spaces,
               const std::map<Range, IndexSpace>& dep_space_relation);

    IndexSpace(const std::vector<TiledIndexSpace>& indep_spaces,
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
    IndexSpace(const std::shared_ptr<IndexSpaceInterface>& impl) :
      impl_{impl} {}

    // Index Accessors
    Index index(Index i, const IndexVector& indep_index = {});
    Index operator[](Index i) const;

    // Subspace Accessors
    IndexSpace operator()(const IndexVector& indep_index = {}) const;
    IndexSpace operator()(const std::string& named_subspace_id) const;

    // Iterators
    IndexIterator begin() const;
    IndexIterator end() const;

    /**
     * @brief Size of this index space in terms of the number of indices in it
     *
     * @return index space size
     */
    std::size_t size() const;

    /**
     * @brief Maximum size of this index space for any value of indices this
     * index space depends on
     *
     * @return maximum size of the index space
     */
    std::size_t max_size() const;

    /**
     * @brief Spin attribute accessor for an index
     *
     * @param [in] idx Index value
     *
     * @return spin attribute
     */
    Spin spin(Index idx) const;

    /**
     * @brief Spatial attribute accessor for an index
     *
     * @param [in] idx Index value
     *
     * @return spatial attribute
     */
    Spatial spatial(Index idx) const;

    /**
     * @brief Access ranges of indices with a specific spin value
     *
     * @param [in] spin Spin value
     *
     * @return Ranges of indices with the given spin value
     */
    const std::vector<Range>& spin_ranges(Spin spin) const;

    /**
     * @brief Access ranges of indices with a specific spatial value
     *
     * @param [in] spatial Spatial  value
     *
     * @return Ranges of indices with the given spatial value
     */
    const std::vector<Range>& spatial_ranges(Spatial spatial) const;

    /**
     * @brief Does this index space have spin attribute
     *
     * @return true if this index space has spin attribute
     */
    bool has_spin() const;

    /**
     * @brief Does this index space have spatial attribute
     *
     * @return true if this index space has spatial attribute
     */
    bool has_spatial() const;

    /**
     * @brief Access the named ranges in this index space
     *
     * @return Map of named ranges
     */
    const NameToRangeMap& get_named_ranges() const;

    IndexSpace root_index_space() const;

    const std::vector<TiledIndexSpace>& key_tiled_index_spaces() const;

    /**
     * @brief Number of index spaces this index space depends on
     *
     * @return Number of index spaces this index space depends on
     */
    size_t num_key_tiled_index_spaces() const;

    const std::map<IndexVector, IndexSpace>& map_tiled_index_spaces() const;

    const std::map<std::string, IndexSpace>& map_named_sub_index_spaces() const;

    /**
     * @brief Are two index spaces identical
     *
     * @param [in] rhs Index space to be compared with
     *
     * @return true if this index space is idential to @param rhs
     */
    bool is_identical(const IndexSpace& rhs) const {
        return impl_ == rhs.impl_;
    }

    /**
     * @todo @bug Do we need this routine?
     */
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

    bool is_dependent() const { return (num_key_tiled_index_spaces() > 0); }

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
}; // class IndexSpace

} // namespace tamm

#endif // TAMM_INDEX_SPACE_HPP_
