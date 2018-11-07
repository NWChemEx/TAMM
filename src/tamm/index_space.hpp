/**
 * @file index_space.hpp
 * @author Erdal Mutlu (erdal.mutlu@pnnl.gov)
 * @brief User facing IndexSpace class used for constructing
 * IndexSpaces
 * @version 0.1
 * @date 2018-11-06
 *
 * @copyright Copyright (c) 2018, Pacific Northwest National Laboratory
 *
 */
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

/**
 * @defgroup index_space IndexSpace
 *
 * @file index_space
 * @brief user facing object for constructing IndexSpaces
 *
 */
namespace tamm {
/// Forward declaration for IndexSpaceInterface
class IndexSpaceInterface;
/// Forward declaration for TiledIndexSpace
class TiledIndexSpace;

/**
 * @ingroup index_space
 * @class IndexSpace
 * @brief Main IndexSpace class that clients will be actively
 *        using for constructing Tensors via TiledIndexSpaces.
 *
 * @todo Possibly use named parameters idiom for construction
 */
class IndexSpace {
public:
    /// Type alias for iterators from this IndexSpace
    using Iterator = IndexIterator;
    // Constructors
    IndexSpace() = default;

    IndexSpace(const IndexSpace&) = default;
    IndexSpace(IndexSpace&&)      = default;
    ~IndexSpace()                 = default;
    IndexSpace& operator=(const IndexSpace&) = default;
    IndexSpace& operator=(IndexSpace&&) = default;

    /**
     * @brief Construct a new IndexSpace object using initializer list of
     * indices.
     *
     * @param [in] indices input Index values
     */
    IndexSpace(const std::initializer_list<Index>& indices) :
      IndexSpace{indices, {}, {}, {}} {}

    /**
     * @brief Construct a new IndexSpace object using a range, named subspaces
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
     * the reference IndexSpace
     *
     * Sub-space. no inherited named subspaces from the reference IndexSpace
     * To iterate over the ranges into which an attribute partitions this
     * IndexSpace, the parent space's attributes are accessed by using the
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
     * indices from the reference IndexSpace
     *
     * @param [in] is reference IndexSpace
     * @param [in] indices set of indices used for constructing the sub space
     * @param [in] named_subspaces map from strings to (sub-)IndexSpace object
     */
    IndexSpace(const IndexSpace& is, const IndexVector& indices,
               const NameToRangeMap& named_subspaces = {});

    /**
     * @brief Construct a new (Aggregated) IndexSpace object by aggregating
     * other IndexSpaces
     *
     * Aggregate. named subspaces  and attributes from all spaces in
     * input IndexSpaces with a non-empty name/attributes are accessible through
     * the reference IndexSpaces
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
     * reference IndexSpaces by a map from strings to ':' separated strings
     */
    IndexSpace(const std::vector<IndexSpace>& spaces,
               const std::vector<std::string>& names = {},
               const NameToRangeMap& named_subspaces = {},
               const std::map<std::string, std::vector<std::string>>&
                 subspace_references = {});

    /**
     * @brief Construct a new (Dependent) IndexSpace object using a vector of
     * dependent IndexSpaces.
     *
     * Dependent: named subspaces and attributes for all dependent spaces in
     * input dependent IndexSpaces with a non-empty name/attributes are
     * accessible through the dependent IndexSpaces
     *
     * @param [in] indep_spaces dependent TiledIndexSpaces used for construction
     * @param [in] dep_space_relation relation between each set of indices on
     * dependent IndexSpaces
     */
    IndexSpace(const std::vector<TiledIndexSpace>& indep_spaces,
               const std::map<IndexVector, IndexSpace>& dep_space_relation);
    /**
     * @brief Construct a new (Dependent) IndexSpace object using a vector of
     * dependent IndexSpaces and a specific reference IndexSpace
     *
     * @param [in] indep_spaces dependent TiledIndexSpaces used for construction
     * @param [in] ref_space reference IndexSpace
     * @param [in] dep_space_relation relation between each set of indices on
     * dependent IndexSpace
     */
    IndexSpace(const std::vector<TiledIndexSpace>& indep_spaces,
               const IndexSpace& ref_space,
               const std::map<IndexVector, IndexSpace>& dep_space_relation);
    /**
     * @brief Construct a new (Dependent) IndexSpace object using a vector of
     * dependent IndexSpaces
     *
     * @param [in] indep_spaces dependent TiledIndexSpace used for construction
     * @param [in] dep_space_relation a map between a range of indices and a
     * dependent IndexSpace
     */
    IndexSpace(const std::vector<TiledIndexSpace>& indep_spaces,
               const std::map<Range, IndexSpace>& dep_space_relation);

    /**
     * @brief Construct a new (Dependent) IndexSpace object using a vector of
     * dependent IndexSpaces and a reference IndexSpace
     *
     * @param [in] indep_spaces dependent TiledIndexSpace used for construction
     * @param [in] ref_space reference IndexSpace
     * @param [in] dep_space_relation a map between a range of indices and a
     * dependent IndexSpace
     */
    IndexSpace(const std::vector<TiledIndexSpace>& indep_spaces,
               const IndexSpace& ref_space,
               const std::map<Range, IndexSpace>& dep_space_relation);
    /**
     * @brief Construct a new IndexSpace object by using a shared_ptr
     *
     * Used for constructing reference IndexSpace object from the
     * implementations
     *
     * @param [in] impl input shared_ptr to IndexSpaceInterface implementation
     */
    IndexSpace(const std::shared_ptr<IndexSpaceInterface>& impl) :
      impl_{impl} {}

    // Index Accessors
    /**
     * @brief Accessor method for accessing an index on an IndexSpace
     *
     * @param [in] i index to be accessed
     * @param [in] indep_index dependent indices if IndexSpace objects
     * is dependent on other IndexSpaces
     * @returns value for the corresponding index
     */
    Index index(Index i, const IndexVector& indep_index = {});

    /**
     * @brief Operator overload for accessing an index on an IndexSpace
     *
     *
     * @param [in] i index to be accessed
     * @returns value for the corresponding index
     *
     * @warning This method is not implemented for dependent IndexSpaces
     */
    Index operator[](Index i) const;

    // Subspace Accessors
    /**
     * @brief Operator overload for getting dependent IndexSpaces
     *
     * @param [in] indep_index index values for the dependent IndexSpace
     * @returns an IndexSpace that is dependent on for given indices
     * @warning This method will return itself if it is not a dependent
     * IndexSpace
     */
    IndexSpace operator()(const IndexVector& indep_index = {}) const;

    /**
     * @brief Operator overload for getting named sub IndexSpaces
     *
     * @param [in] named_subspace_id name of the sub IndexSpace
     * @returns an IndexSpace for the named sub space
     */
    IndexSpace operator()(const std::string& named_subspace_id) const;

    // Iterators
    /**
     * @brief Iterator overload for begin method
     *
     * @returns a std::vector<Index>::const_iterator pointing to the beginning
     * of indices
     */
    IndexIterator begin() const;

    /**
     * @brief Iterator overload for end method
     *
     * @returns a std::vector<Index>::const_iterator pointing to the end of
     * indices
     */
    IndexIterator end() const;

    /**
     * @brief Number of indices in this IndexSpace in terms of the number of
     * indices in it
     *
     * @returns number of indices in the IndexSpace
     */
    std::size_t num_indices() const;

    /**
     * @brief Maximum number of indices in this IndexSpace for any value of
     * indices this IndexSpace depends on
     *
     * @returns maximum number of indices in the IndexSpace
     */
    std::size_t max_num_indices() const;

    /**
     * @brief Spin attribute accessor for an index
     *
     * @param [in] idx Index value
     *
     * @returns spin attribute
     */
    Spin spin(Index idx) const;

    /**
     * @brief Spatial attribute accessor for an index
     *
     * @param [in] idx Index value
     *
     * @returns spatial attribute
     */
    Spatial spatial(Index idx) const;

    /**
     * @brief Access ranges of indices with a specific spin value
     *
     * @param [in] spin Spin value
     *
     * @returns Ranges of indices with the given spin value
     */
    const std::vector<Range>& spin_ranges(Spin spin) const;

    /**
     * @brief Access ranges of indices with a specific spatial value
     *
     * @param [in] spatial Spatial  value
     *
     * @returns Ranges of indices with the given spatial value
     */
    const std::vector<Range>& spatial_ranges(Spatial spatial) const;

    /**
     * @brief Does this IndexSpace have spin attribute
     *
     * @returns true if this IndexSpace has spin attribute
     */
    bool has_spin() const;

    /**
     * @brief Does this IndexSpace have spatial attribute
     *
     * @returns true if this IndexSpace has spatial attribute
     */
    bool has_spatial() const;

    /**
     * @brief Access the named ranges in this IndexSpace
     *
     * @returns Map of named ranges
     */
    const NameToRangeMap& get_named_ranges() const;

    /**
     * @brief Getter method for the root IndexSpace
     *
     * @returns the root IndexSpace object
     */
    IndexSpace root_index_space() const;

    /**
     * @brief Getter method for dependent IndexSpace
     *
     * @returns vector of TiledIndexSpace's that this IndexSpace depends on
     * @warning returns an empty vector if it is an independent IndexSpace
     */
    const std::vector<TiledIndexSpace>& key_tiled_index_spaces() const;

    /**
     * @brief Number of IndexSpaces this IndexSpace depends on
     *
     * @returns Number of IndexSpaces this IndexSpace depends on
     */
    size_t num_key_tiled_index_spaces() const;

    /**
     * @brief Getter method for the dependency map
     *
     * @returns a map between set of indices and dependent IndexSpace
     * @warning This method returns an empty map if it is an independent
     * IndexSpace
     */
    const std::map<IndexVector, IndexSpace>& map_tiled_index_spaces() const;

    /**
     * @brief Getter method for the named subspace map
     *
     * @returns a map between a string and an IndexSpace
     * @warning This method returns an empty map if there is no named subspace
     * associated with it
     */
    const std::map<std::string, IndexSpace>& map_named_sub_index_spaces() const;

    /**
     * @brief Checks if two IndexSpaces are identical using hash values
     *
     * @param [in] rhs IndexSpace to be compared with
     *
     * @returns true if this IndexSpace is identical to @param rhs
     */
    bool is_identical(const IndexSpace& rhs) const {
        return (hash_value_ == rhs.hash());
    }

    /**
     * @brief Checks if @param rhs is created before this IndexSpace
     *
     * @param [in] rhs IndexSpace to be compared
     * @returns true if @param rhs is constructed before this
     * @todo @bug Do we need this routine?
     * @warning Currently this check is implemented by comparing the addresses
     */
    bool is_less_than(const IndexSpace& rhs) const { return impl_ < rhs.impl_; }

    /**
     * @brief Checks if @param rhs is compatible with this IndexSpace
     *
     * @param [in] rhs IndexSpace to be compared
     * @returns true if two IndexSpaces are identical
     * @todo Update compatibility check
     * @warning Currently this check is using identical, it will be updated
     * later.
     */
    bool is_compatible(const IndexSpace& rhs) const {
        return is_identical(rhs);
    }

    /**
     * @brief Checks if @param rhs and this IndexSpace have the same root
     * IndexSpace
     *
     * @param [in] rhs IndexSpace to be compared
     * @returns true if two IndexSpaces have the same root IndexSpace
     */
    bool is_identical_reference(const IndexSpace& rhs) const {
        return (*this).root_index_space().is_identical(rhs.root_index_space());
    }

    /**
     * @brief Checks if @param rhs and this IndexSpace have compatible root
     * IndexSpaces
     *
     * @param [in] rhs IndexSpace sto be compared
     * @returns true if two IndexSpaces have compatible root IndexSpaces
     */
    bool is_compatible_reference(const IndexSpace& rhs) const {
        return (*this).root_index_space().is_compatible(rhs.root_index_space());
    }

    /**
     * @brief Checks if this is a dependent IndexSpace
     *
     * @returns true if this IndexSpace depends on some other IndexSpace
     */
    bool is_dependent() const { return (num_key_tiled_index_spaces() > 0); }

    /**
     * @brief Getter method for the SpinAttribute object associated with this
     * IndexSpace
     *
     * @returns SpinAttribute associated with this IndexSpace
     * @warning This method will return a default SpinAttribute if there is no
     * spin associated with this IndexSpace
     */
    SpinAttribute get_spin() const;

    /**
     * @brief Getter method for the SpatialAttribute object associated with this
     * IndexSpace
     *
     * @returns SpatialAttribute associated with this IndexSpace
     * @warning This method will return a default SpatialAttribute if there is
     * no spatial associated with this IndexSpace
     */
    SpatialAttribute get_spatial() const;

    /**
     * @brief Finds the position of Index @param idx
     *
     * @param [in] idx Index to be searched
     * @returns position of Index @param idx if it is among the indices
     * @warning This method will return -1 if Index being searched is not in the
     * IndexSpace
     */
    int find_pos(Index idx) const {
        int pos = 0;
        for(auto i = begin(); i != end(); i++, pos++) {
            if((*i) == idx) { return pos; }
        }
        return -1;
    }

    /**
     * @brief Getter method for the hash value of this IndexSpace
     *
     * @returns a hash value associated with this IndexSpace
     */
    size_t hash() const { return hash_value_; }

    // Comparison operators
    friend bool operator==(const IndexSpace& lhs, const IndexSpace& rhs);
    friend bool operator<(const IndexSpace& lhs, const IndexSpace& rhs);
    friend bool operator!=(const IndexSpace& lhs, const IndexSpace& rhs);
    friend bool operator>(const IndexSpace& lhs, const IndexSpace& rhs);
    friend bool operator<=(const IndexSpace& lhs, const IndexSpace& rhs);
    friend bool operator>=(const IndexSpace& lhs, const IndexSpace& rhs);

protected:
    std::shared_ptr<IndexSpaceInterface>
      impl_;            /**< shared pointer to the implementation */
    size_t hash_value_; /**< hash value associated with the IndexSpace */
};                      // class IndexSpace

} // namespace tamm

#endif // TAMM_INDEX_SPACE_HPP_
