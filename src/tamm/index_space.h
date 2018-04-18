#ifndef TAMM_INDEX_SPACE_H_
#define TAMM_INDEX_SPACE_H_

#include "tamm/types.h"
#include <algorithm>
#include <iterator>
#include <map>
#include <memory>
#include <sstream>

namespace tamm {

using Index         = uint32_t;
using IndexVector   = std::vector<Index>;
using IndexIterator = std::vector<Index>::const_iterator;
using Tile          = uint32_t;

/**
 * @brief Helper methods for Range based constructors.
 *        For now we are using a simple range representation
 *        We will use range constructs from Utilities repo.
 *
 * @todo Possibly replace with Range class in Utilities repo.
 */
class Range {
    public:
    // Default Ctors
    Range() = default;

    // Copy/Move Ctors and Assignment Operators
    Range(Range&&)      = default;
    Range(const Range&) = default;
    Range& operator=(Range&&) = default;
    Range& operator=(const Range&) = default;

    // Dtor
    ~Range() = default;

    // Ctors
    Range(Index lo, Index hi, Index step = 1) : lo_{lo}, hi_{hi}, step_{step} {}

    // Accessors
    Index lo() const { return lo_; }
    Index hi() const { return hi_; }
    Index step() const { return step_; }

    /**
     * @brief Method for checking if a given Index value
     *        is within a range or not
     *
     * @param [in] idx Index value being checked for
     * @returns true if range includes corresponding index idx
     */
    bool contains(Index idx) const {
        if(idx == lo_) { return true; }

        if(idx < lo_ && hi_ <= idx) { return false; }

        if(step_ > 1) { return ((idx - lo_) % step_ == 0); }

        return true;
    }

    /**
     * @brief Method for checking disjointness of two ranges
     *
     * @param [in] rhs input Range value for checking disjointness
     * @returns true if ranges are disjoint
     */
    bool is_disjoint_with(const Range& rhs) const {
        if(lo_ == rhs.lo()) { return false; }
        if(rhs.hi() <= lo_ && hi_ <= rhs.lo()) { return true; }

        if(step_ == rhs.step()) {
            return (lo_ % step_ != rhs.lo() % step_);
        } else { // if ranges overlap and steps are different
            Index inter_hi = hi_ <= rhs.hi() ? hi_ : rhs.hi();
            Index inter_lo = lo_ >= rhs.lo() ? lo_ : rhs.lo();

            Index startA =
              inter_lo + ((lo_ - inter_lo) % step_ + step_) % step_;
            Index startB =
              inter_lo +
              ((rhs.lo() - inter_lo) % rhs.step() + rhs.step()) % rhs.step();
            if(startA >= inter_hi || startB >= inter_hi) { return true; }
            Index offset = startB - startA;
            Index gcd, x, y;
            std::tie(gcd, x, y) = extended_gcd(step_, rhs.step());
            Index interval_     = step_ / gcd;
            Index interval_rhs  = rhs.step() / gcd;
            Index step          = interval_ * interval_rhs * gcd;
            if(offset % gcd != 0) {
                return true;
            } else {
                Index crt    = (offset * interval_ * (x % interval_rhs)) % step;
                Index filler = 0;
                Index gap    = offset - crt;
                filler      = gap % step == 0 ? gap : ((gap / step) + 1) * step;
                Index start = startA + crt + filler;
                return !(start < inter_hi && start >= inter_lo);
            }
        }
        return true;
    }

    protected:
    Index lo_;
    Index hi_;
    Index step_;

    private:
    /**
     * @brief Euclid's extended gcd for is_disjoint_with method
     *        - ax + by = gcd(a,b)
     *
     * @param [in] a first number for calculating gcd
     * @param [in] b second number for calculating gcd
     * @returns a tuple for gcd, x and y coefficients
     */
    std::tuple<int, int, int> extended_gcd(int a, int b) const {
        if(a == 0) { return std::make_tuple(b, 0, 1); }

        int gcd, x, y;
        // unpack tuple  returned by function into variables
        std::tie(gcd, x, y) = extended_gcd(b % a, a);

        return std::make_tuple(gcd, (y - (b / a) * x), x);
    }
}; // Range

/**
 * @brief Range constructor with low, high and step size
 *
 */
static inline Range range(Index lo, Index hi, Index step = 1) {
    return Range(lo, hi, step);
}

/**
 * @brief Range constructor by giving only a count
 *
 */
static inline Range range(Index count) { return range(Index{0}, count); }

/**
 * @brief Helper method for constructing IndexVector for a given Range
 *
 * @param [in] range a Range type argument
 * @returns an IndexVector for the corresponding range
 */
static inline IndexVector construct_index_vector(const Range& range) {
    IndexVector ret;
    for(Index i = range.lo(); i < range.hi(); i += range.step()) {
        ret.push_back(i);
    }

    return ret;
}

/**
 * @brief Type definition for the map between range names
 *        and corresponding set of ranges (e.g. "occ", "virt")
 *
 */
using NameToRangeMap = std::map<std::string, const std::vector<Range>>;

template<typename AttributeType>
using AttributeToRangeMap = std::map<AttributeType, std::vector<Range>>;

/**
 * @class Attribute
 * @brief Attribute definition which will be used for representing
 *        Spin, Spatial and any other attributes required.
 *
 * @tparam T is an Attribute type (e.g. Spin, Spatial)
 */
template<typename T>
class Attribute {
    public:
    /**
     * @brief Construct a new Attribute object using default implementation
     *
     */
    Attribute() = default;

    /**
     * @brief Construct a new Attribute object using the map from attribute
     * values to set of ranges
     *
     * @param [in] attr_map a map from attribute values to vector of Ranges
     */
    Attribute(const AttributeToRangeMap<T>& attr_map) : attr_map_{attr_map} {}

    Attribute(const Attribute&) = default;
    Attribute(Attribute&&)      = default;
    Attribute& operator=(const Attribute&) = default;
    Attribute& operator=(Attribute&&) = default;

    ~Attribute() = default;

    /**
     * @brief Given an Index, ind, user can query
     *        associated Attribute value
     *
     * @param [in] idx Index argument
     * @returns the associated Attribute value
     */
    T operator()(Index idx) const {
        for(const auto& kv : attr_map_) {
            for(const auto& range : kv.second) {
                if(range.contains(idx)) { return kv.first; }
            }
        }

        // return default attribute value if not found
        // assumption for default attribute -> {0}
        return T{0};
    }

    /**
     * @brief Accessor to the map from attribute values to set of ranges
     *
     * @param [in] att input Attribute value being searched
     * @returns Range vector for the corresponding attribute value
     *          (empty vector if it doesn't exist)
     */
    std::vector<Range> attribute_range(T att) const {
        return ((attr_map_.find(att) == attr_map_.end()) ?
                  std::vector<Range>{} :
                  attr_map_.at(att));
    }

    /**
     * @brief Getting the iterators for Index vectors
     *        for a given Attribute value (e.g. Spin{1})
     *
     * @param [in] val an Attribute value as an argument
     * @returns std::vector<Range>::const_iterator returns beginning
     * iterator for associated ranges for the attribute
     */
    std::vector<Range>::const_iterator attribute_begin(const T& val) const {
        return attr_map_[val].begin();
    }

    /**
     * @brief Getting the iterators for Index vectors
     *        for a given Attribute value (e.g. Spin{1})
     *
     * @param [in] val an Attribute value as an argument
     * @returns std::vector<Range>::const_iterator returns end
     * iterator for associated ranges for the attribute
     */
    std::vector<Range>::const_iterator attribute_end(const T& val) {
        return attr_map_[val].end();
    }

    /**
     * @brief Check if the attribute relations are empty
     *
     * @returns empty
     */
    bool empty() const { return attr_map_.find(T{0}) != attr_map_.end(); }

    protected:
    AttributeToRangeMap<T> attr_map_; //
};                                    // Attribute

using SpinAttribute    = Attribute<Spin>;
using SpatialAttribute = Attribute<Spatial>;

// forward decleration for IndexSpaceInterface
class IndexSpace;

/**
 * @class IndexSpaceInterface
 * @brief Base abstract implementation class as an interface
 *        for the different IndexSpace implementations
 */
class IndexSpaceInterface {
    public:
    /**
     * @brief Destroy the Index Space Interface object
     *
     * @todo specify (=default or =delete) the implicit functions
     */
    virtual ~IndexSpaceInterface() {}

    /**
     * @brief Accessor method for Index values in an IndexSpace
     *
     * @param [in] idx input Index value
     * @param [in] indep_index dependent Index values (mainly used for dependent
     * IndexSpace)
     * @returns an Index value from the IndexSpace for the corresponding input
     * Index
     */
    virtual Index index(Index idx,
                        const IndexVector& indep_index = {}) const = 0;

    /**
     * @brief
     *
     * @param [in] idx input Index value
     * @returns an Index value from the IndexSpace for the corresponding input
     * Index
     */
    virtual Index operator[](Index idx) const = 0;

    /**
     * @brief operator () for accessing IndexSpace objects associated with the
     * interface implementation
     *
     * @param [in] indep_index dependent Index values (mainly used for dependent
     * IndexSpaces)
     * @returns an IndexSpace object
     */
    virtual IndexSpace operator()(
      const IndexVector& indep_index = {}) const = 0;

    /**
     * @brief operator () overload with an input string value
     *
     * @param [in] named_subspace_id string value of the subspace name to be
     * accessed
     * @returns an IndexSpace corresponding to the subspace name
     */
    virtual IndexSpace operator()(
      const std::string& named_subspace_id) const = 0;

    /**
     * @brief Iterator accessor to the Index values associated with the
     * IndexSpace
     *
     * @returns const_iterator to the first element of the IndexVector
     */
    virtual IndexIterator begin() const = 0;
    /**
     * @brief Iterator accessor to the Index values associated with the
     * IndexSpace
     *
     * @returns const_iterator to the last element of the IndexVector
     */
    virtual IndexIterator end() const = 0;
    /**
     * @brief Returns the size of the IndexVector associated with the
     * IndexSpaceß
     *
     * @returns size of the IndexVector
     */
    virtual Index size() const = 0;

    /**
     * @brief Accessor methods to Spin value associated with the input Index
     *
     * @param [in] idx input Index value
     * @returns associated Spin value for the input Index value
     */
    virtual Spin spin(Index idx) const = 0;
    /**
     * @brief Accessor methods to Spatial value associated with the input Index
     *
     * @param [in] idx input Index value
     * @returns associated Spatial value for the input Index value
     */
    virtual Spatial spatial(Index idx) const = 0;

    /**
     * @brief Accessor method for the set of Ranges associated with a Spin value
     *
     * @param [in] spin input Spin value
     * @returns a vector of Ranges associated with the input Spin value
     */
    virtual std::vector<Range> spin_ranges(Spin spin) const = 0;

    /**
     * @brief Accessor method for the set of Ranges associated with a Spatial
     * value
     *
     * @param [in] spatial input Spatial value
     * @returns a vector of Ranges associated with the input Spatial value
     */
    virtual std::vector<Range> spatial_ranges(Spatial spatial) const = 0;

    /**
     * @brief Boolean method for checking if an IndexSpace has SpinAttribute
     *
     * @returns true if there is a SpinAttribute associated with the IndexSpace
     */
    virtual bool has_spin() const = 0;
    /**
     * @brief Boolean method for checking if an IndexSpace has SpatialAttribute
     *
     * @returns true if there is a SpatialAttribute associated with the
     * IndexSpace
     */
    virtual bool has_spatial() const = 0;

    protected:
    std::weak_ptr<IndexSpaceInterface> this_weak_ptr_;

    /**
     * @brief Helper methods for string manupulation, the main use
     *        is to split a string into vector of strings with
     *        respect to a deliminator

    * @param [in] str string to be split
    * @param [in] delim used char deliminator
    * @returns a vector of split strings
    */
    static std::vector<std::string> split(const std::string& str, char delim) {
        std::vector<std::string> elems;
        std::size_t start = 0, end = 0;
        while((end = str.find(delim, start)) != std::string::npos) {
            if(end != start) {
                elems.push_back(str.substr(start, end - start));
            }
            start = end + 1;
        }
        if(end != start) { elems.push_back(str.substr(start)); }
        return elems;
    }

    /**
     * @brief Helper method checking if a vector of data
     *        has any duplicates by:
     *          - sorting a copy of the vector
     *          - check for adjacent repeation
     *
     * @tparam ContainerType stl container type with iterator
     * (RandomAccessIterator) support
     * @param [in] data_vec input vector
     * @returns true returned if there are duplicates
     */
    template<typename ContainerType>
    static bool has_duplicate(const ContainerType& data_vec) {
        ContainerType temp_vec = data_vec;
        std::sort(temp_vec.begin(), temp_vec.end());

        return (std::adjacent_find(temp_vec.begin(), temp_vec.end()) ==
                temp_vec.end());
    }

    /**
     * @brief Check if the input attributes is valid:
     *        - no-overlap between attribute ranges
     *        - covers all indicies
     *
     * @tparam AttributeType an attribute type (e.g. Spin, Spatial)
     * @param indices set of indices to check against
     * @param att attribute map to check for validity
     * @return true if there is no overlap on the ranges and fully covers
     *              indices
     * @return false otherwise
     */
    template<typename AttributeType>
    bool is_valid_attribute(const IndexVector& indices,
                            const AttributeToRangeMap<AttributeType>& att) {
        // Construct index vector from input attribute ranges
        IndexVector att_indices = {};
        for(const auto& kv : att) {
            for(const auto& range : kv.second) {
                for(const auto& index : construct_index_vector(range)) {
                    att_indices.push_back(indices[index]);
                }
            }
        }
        // Check no overlap on the ranges
        std::sort(att_indices.begin(), att_indices.end());
        EXPECTS(has_duplicate<IndexVector>(att_indices));

        // Check for full coverage of the indices
        EXPECTS(indices.size() == att_indices.size());
        // copy indicies
        IndexVector temp_indices(indices);
        // sort temporary index vector for equality check
        std::sort(temp_indices.begin(), temp_indices.end());
        EXPECTS(std::equal(temp_indices.begin(), temp_indices.end(),
                           att_indices.begin()));

        return true;
    }

    private:
    /**
     * @brief Set the weak ptr object for IndexSpaceInterface
     *
     * @param [in] weak_ptr std::weak_ptr to IndexSpaceInterface
     */
    void set_weak_ptr(std::weak_ptr<IndexSpaceInterface> weak_ptr) {
        this_weak_ptr_ = weak_ptr;
    }

    friend class IndexSpace;
}; // IndexSpaceInterface

// Forward class declarations for different types
class RangeIndexSpaceImpl;
class SubSpaceImpl;
class AggregateSpaceImpl;
class DependentIndexSpaceImpl;

/**
 * @class IndexSpace
 * @brief Main IndexSpace class that clients will be actively
 *        using for constructing Tensors via TiledIndexSpaces.
 *
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
     * @param [in] spatial map from Spatial attributeß values to corresponding
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
    Index size() const;

    // Attribute Accessors
    Spin spin(Index idx) const;
    Spatial spatial(Index idx) const;

    std::vector<Range> spin_ranges(Spin spin) const;
    std::vector<Range> spatial_ranges(Spatial spatial) const;

    bool has_spin() const;
    bool has_spatial() const;

    bool is_identical(const IndexSpace& rhs) const {
        return impl_ == rhs.impl_;
    }

    bool is_less_than(const IndexSpace& rhs) const { return impl_ < rhs.impl_; }

    // @todo Re-visit later
    bool is_compatible(const IndexSpace& rhs) const {
        return is_identical(rhs);
    }

    protected:
    std::shared_ptr<IndexSpaceInterface> impl_;
}; // IndexSpace

/**
 * @brief IndexSpace implementation for range based
 *        IndexSpace construction.
 *
 */
class RangeIndexSpaceImpl : public IndexSpaceInterface {
    public:
    // @todo do we need a default constructor?
    // RangeIndexSpaceImpl() = default;

    /**
     * @brief Construct a new RangeIndexSpaceImpl object
     *
     * @todo optimization - constructing map<string, IndexSpace> can be delayed
     * until a specific subspace is requested
     *
     * @param [in] indices vector of Index values
     * @param [in] named_ranges a map from string value to a set of associated
     * ranges
     * @param [in] spin a map for Spin values to set of associated ranges
     * @param [in] spatial a map from Spatial values to set of associated ranges
     */
    RangeIndexSpaceImpl(const IndexVector& indices,
                        const NameToRangeMap& named_ranges,
                        const AttributeToRangeMap<Spin>& spin,
                        const AttributeToRangeMap<Spatial>& spatial) :
      indices_{indices},
      named_ranges_{named_ranges},
      named_subspaces_{construct_subspaces(named_ranges)},
      spin_{construct_spin(spin)},
      spatial_{construct_spatial(spatial)} {
        EXPECTS(has_duplicate<IndexVector>(indices_));
    }

    // @todo do we need these copy/move constructor/operators?
    RangeIndexSpaceImpl(RangeIndexSpaceImpl&&)      = default;
    RangeIndexSpaceImpl(const RangeIndexSpaceImpl&) = default;
    RangeIndexSpaceImpl& operator=(RangeIndexSpaceImpl&&) = default;
    RangeIndexSpaceImpl& operator=(const RangeIndexSpaceImpl&) = default;
    ~RangeIndexSpaceImpl()                                     = default;

    // Index Accessors
    Index index(Index i, const IndexVector& indep_index = {}) const override {
        return indices_[i];
    }
    Index operator[](Index i) const override { return indices_[i]; }

    // Subspace Accessors
    IndexSpace operator()(const IndexVector& indep_index = {}) const override {
        return IndexSpace{this_weak_ptr_.lock()};
    }

    IndexSpace operator()(const std::string& named_subspace_id) const override {
        return named_subspaces_.at(named_subspace_id);
    }

    // Iterators
    IndexIterator begin() const override { return indices_.begin(); }
    IndexIterator end() const override { return indices_.end(); }

    // Size of this index space
    Index size() const override { return indices_.size(); }

    // Attribute Accessors
    Spin spin(Index idx) const override { return spin_(idx); }
    Spatial spatial(Index idx) const override { return spatial_(idx); }

    std::vector<Range> spin_ranges(Spin spin) const override {
        return spin_.attribute_range(spin);
    }
    std::vector<Range> spatial_ranges(Spatial spatial) const override {
        return spatial_.attribute_range(spatial);
    }

    bool has_spin() const override { return spin_.empty(); }
    bool has_spatial() const override { return spatial_.empty(); }

    protected:
    IndexVector indices_;
    NameToRangeMap named_ranges_;
    std::map<std::string, IndexSpace> named_subspaces_;
    SpinAttribute spin_;
    SpatialAttribute spatial_;

    /**
     * @brief Helper method for generating the map between string values to
     * IndexSpaces. Mainly used for constructing the subspaces.
     *
     * @param [in] in_map NameToRangeMap argument holding string to Range vector
     * map
     * @return std::map<std::string, IndexSpace> returns the map from
     *                                           strings to subspaces
     */
    std::map<std::string, IndexSpace> construct_subspaces(
      const NameToRangeMap& in_map) {
        std::map<std::string, IndexSpace> ret;
        for(auto& kv : in_map) {
            std::string name         = kv.first;
            IndexVector temp_indices = {};
            for(auto& range : kv.second) {
                for(auto& i : construct_index_vector(range)) {
                    temp_indices.push_back(indices_[i]);
                }
            }
            ret.insert({name, IndexSpace{temp_indices}});
        }

        return ret;
    }

    /**
     * @brief Helper method for constructing  and validating the attributes for
     * IndexSpace
     *
     * @param [in] spin Spin attribute to Range map that is used for
     *             constructing Spin attribute
     * @returns a SpinAttribute constructed using input map
     */
    SpinAttribute construct_spin(const AttributeToRangeMap<Spin>& spin) {
        // return default spin value (Spin{0}) for the whole range
        if(spin.empty()) {
            return SpinAttribute(
              AttributeToRangeMap<Spin>{{Spin{0}, {range(indices_.size())}}});
        }

        // Check validity of the input attribute
        EXPECTS(is_valid_attribute<Spin>(indices_, spin));

        return SpinAttribute{spin};
    }

    /**
     * @brief Helper method for constructing  and validating the attributes for
     * IndexSpace
     *
     * @param [in] spatial a Spatial value to Range map that is used for
     *             constructing Spin attribute
     * @return [in] SpinAttribute returns a Spin attribute
     */
    SpatialAttribute construct_spatial(
      const AttributeToRangeMap<Spatial>& spatial) {
        // return default spatial value (Spatial{0}) for the whole range
        if(spatial.empty()) {
            return SpatialAttribute(AttributeToRangeMap<Spatial>{
              {Spatial{0}, {range(indices_.size())}}});
        }

        // Check validity of the input attribute
        EXPECTS(is_valid_attribute<Spatial>(indices_, spatial));

        return SpatialAttribute{spatial};
    }
}; // RangeIndexSpaceImpl

/**
 * @brief IndexSpace implementation for subspace based
 *        IndexSpace construction.
 *
 */
class SubSpaceImpl : public IndexSpaceInterface {
    public:
    // @todo do we need a default constructor?
    // SubSpaceImpl() = default;

    /**
     * @brief Construct a new SubSpaceImpl object
     *
     * @todo optimization - constructing map<string, IndexSpace> can be delayed
     * until a specific subspace is requested
     *
     * @param [in] is
     * @param [in] range
     * @param [in] named_ranges
     */
    SubSpaceImpl(const IndexSpace& is, const Range& range,
                 const NameToRangeMap& named_ranges) :
      ref_space_{is},
      ref_range_{range},
      indices_{construct_indices(is, range)},
      named_ranges_{named_ranges},
      named_subspaces_{construct_subspaces(named_ranges)} {}

    // @todo do we need these copy/move constructor/operators
    SubSpaceImpl(SubSpaceImpl&&)      = default;
    SubSpaceImpl(const SubSpaceImpl&) = default;
    SubSpaceImpl& operator=(SubSpaceImpl&&) = default;
    SubSpaceImpl& operator=(const SubSpaceImpl&) = default;
    ~SubSpaceImpl()                              = default;

    // Index Accessors
    Index index(Index i, const IndexVector& indep_index = {}) const override {
        return indices_[i];
    }
    Index operator[](Index i) const override { return indices_[i]; }

    // Subspace Accessors
    IndexSpace operator()(const IndexVector& indep_index = {}) const override {
        return IndexSpace{this_weak_ptr_.lock()};
    }

    IndexSpace operator()(const std::string& named_subspace_id) const override {
        return named_subspaces_.at(named_subspace_id);
    }

    // Iterators
    IndexIterator begin() const override { return indices_.begin(); }
    IndexIterator end() const override { return indices_.end(); }

    // Size of this index space
    Index size() const override { return indices_.size(); }

    // Attribute Accessors
    Spin spin(Index idx) const override { return ref_space_.spin(idx); }
    Spatial spatial(Index idx) const override {
        return ref_space_.spatial(idx);
    }

    std::vector<Range> spin_ranges(Spin spin) const override {
        return ref_space_.spin_ranges(spin);
    }
    std::vector<Range> spatial_ranges(Spatial spatial) const override {
        return ref_space_.spatial_ranges(spatial);
    }

    bool has_spin() const override { return ref_space_.has_spin(); }
    bool has_spatial() const override { return ref_space_.has_spatial(); }

    protected:
    IndexSpace ref_space_;
    Range ref_range_;
    IndexVector indices_;
    NameToRangeMap named_ranges_;
    std::map<std::string, IndexSpace> named_subspaces_;

    /**
     * @brief Helper method for constructing the new set of
     *        indicies from the reference IndexSpace
     *
     * @param ref_space reference IndexSpace argument
     * @param range     Range argument for generating the subspace
     * @return IndexVector returns a vector of Indicies
     */
    IndexVector construct_indices(const IndexSpace& ref_space,
                                  const Range& range) {
        IndexVector ret = {};
        for(const auto& i : construct_index_vector(range)) {
            ret.push_back(ref_space[i]);
        }

        return ret;
    }

    /**
     * @brief Helper method for generating the map between
     *        string values to IndexSpaces. Mainly used for
     *        constructing the subspaces.
     *
     * @param in_map NameToRangeMap argument holding string to Range map
     * @return std::map<std::string, IndexSpace> returns the map from
     *                                           strings to subspaces
     */
    std::map<std::string, IndexSpace> construct_subspaces(
      const NameToRangeMap& in_map) {
        std::map<std::string, IndexSpace> ret;
        for(auto& kv : in_map) {
            std::string name    = kv.first;
            IndexVector indices = {};
            for(auto& range : kv.second) {
                for(auto& i : construct_index_vector(range)) {
                    indices.push_back(indices_[i]);
                }
            }
            ret.insert({name, IndexSpace{indices}});
        }

        return ret;
    }
}; // SubSpaceImpl

/**
 * @brief IndexSpace implementation for aggregation
 *        based IndexSpace construction.
 *
 */
class AggregateSpaceImpl : public IndexSpaceInterface {
    public:
    // @todo do we need a default constructor?
    // AggregateSpaceImpl() = default;

    /**
     * @brief Construct a new Aggregate Space Impl object
     *
     * @param [in] spaces reference IndexSpace objects for aggregating
     * @param [in] names string names associated with each reference IndexSpace
     * @param [in] named_ranges additional string names to Range vector map
     * @param [in] subspace_references named subspace relations using reference
     * IndexSpace named subspaces
     */
    AggregateSpaceImpl(const std::vector<IndexSpace>& spaces,
                       const std::vector<std::string>& names,
                       const NameToRangeMap& named_ranges,
                       const std::map<std::string, std::vector<std::string>>&
                         subspace_references) :
      ref_spaces_(spaces),
      indices_{construct_indices(spaces)},
      named_ranges_{named_ranges},
      named_subspaces_{construct_subspaces(named_ranges)} {
        EXPECTS(has_duplicate<IndexVector>(indices_));
        if(names.size() > 0) { add_ref_names(spaces, names); }
        if(subspace_references.size() > 0) {
            add_subspace_references(subspace_references);
        }
    }

    // @todo do we need these constructor/operators
    AggregateSpaceImpl(AggregateSpaceImpl&&)      = default;
    AggregateSpaceImpl(const AggregateSpaceImpl&) = default;
    AggregateSpaceImpl& operator=(AggregateSpaceImpl&&) = default;
    AggregateSpaceImpl& operator=(const AggregateSpaceImpl&) = default;
    ~AggregateSpaceImpl()                                    = default;

    // Index Accessors
    Index index(Index i, const IndexVector& indep_index = {}) const override {
        return indices_[i];
    }
    Index operator[](Index i) const override { return indices_[i]; }

    // Subspace Accessors
    IndexSpace operator()(const IndexVector& indep_index = {}) const override {
        return IndexSpace{this_weak_ptr_.lock()};
    }

    IndexSpace operator()(const std::string& named_subspace_id) const override {
        return named_subspaces_.at(named_subspace_id);
    }

    // Iterators
    IndexIterator begin() const override { return indices_.begin(); }
    IndexIterator end() const override { return indices_.end(); }

    // Size of this index space
    Index size() const override { return indices_.size(); }

    // @todo what should these return? Currently, it returns the
    // first reference space's spin and spatial attributes.
    // Attribute Accessors
    Spin spin(Index idx) const override {
        NOT_ALLOWED();
        return Spin{0};
    }
    Spatial spatial(Index idx) const override {
        NOT_ALLOWED();
        return Spatial{0};
    }

    std::vector<Range> spin_ranges(Spin spin) const override {
        NOT_ALLOWED();
        return {};
    }
    std::vector<Range> spatial_ranges(Spatial spatial) const override {
        NOT_ALLOWED();
        return {};
    }

    bool has_spin() const override {
        for(const auto& space : ref_spaces_) {
            if(space.has_spin() == false) { return false; }
        }
        return true;
    }
    bool has_spatial() const override {
        for(const auto& space : ref_spaces_) {
            if(space.has_spatial() == false) { return false; }
        }
        return true;
    }

    protected:
    std::vector<IndexSpace> ref_spaces_;
    IndexVector indices_;
    NameToRangeMap named_ranges_;
    std::map<std::string, IndexSpace> named_subspaces_;

    /**
     * @brief Add subspaces reference names foreach aggregated
     *        IndexSpace
     *
     * @param [in] ref_spaces a vector of reference IndexSpaces
     * @param [in] ref_names  a vector of associated names for each
     *                   reference IndexSpace
     */
    void add_ref_names(const std::vector<IndexSpace>& ref_spaces,
                       const std::vector<std::string>& ref_names) {
        EXPECTS(ref_spaces.size() == ref_names.size());
        size_t i = 0;
        for(const auto& space : ref_spaces) {
            named_subspaces_.insert({ref_names[i], space});
            i++;
        }
    }

    /**
     * @brief Add extra references for subspace names
     *        associated with the reference subspaces.
     *
     * @param [in] subspace_references a map from subspace names
     *                            to reference subspace names
     */
    void add_subspace_references(
      const std::map<std::string, std::vector<std::string>>&
        subspace_references) {
        for(const auto& kv : subspace_references) {
            IndexVector temp_indices = {};

            std::string key = kv.first;
            for(const auto& ref_str : kv.second) {
                std::vector<std::string> ref_names = split(ref_str, ':');
                IndexSpace temp_is = named_subspaces_.at(ref_names[0]);

                for(size_t i = 1; i < ref_names.size(); i++) {
                    temp_is = temp_is(ref_names[i]);
                }

                temp_indices.insert(temp_indices.end(), temp_is.begin(),
                                    temp_is.end());
            }
            named_subspaces_.insert({key, IndexSpace{temp_indices}});
        }
    }

    /**
     * @brief Construct set of indicies from the aggregated
     *        IndexSpaces
     *
     * @param [in] spaces vector of IndexSpaces
     * @returns a vector of Index values
     */
    IndexVector construct_indices(const std::vector<IndexSpace>& spaces) {
        IndexVector ret = {};
        for(const auto& space : spaces) {
            ret.insert(ret.end(), space.begin(), space.end());
        }

        return ret;
    }

    /**
     * @brief Helper method for generating the map between
     *        string values to IndexSpaces. Mainly used for
     *        constructing the subspaces.
     *
     * @param [in] in_map NameToRangeMap argument holding string to Range map
     * @returns the map from strings to subspaces
     */
    std::map<std::string, IndexSpace> construct_subspaces(
      const NameToRangeMap& in_map) {
        std::map<std::string, IndexSpace> ret;
        for(auto& kv : in_map) {
            std::string name    = kv.first;
            IndexVector indices = {};
            for(auto& range : kv.second) {
                for(auto& i : construct_index_vector(range)) {
                    // std::cout << "index: "<< i << " value: "<< indices_[i] <<
                    // std::endl;
                    indices.push_back(indices_[i]);
                }
            }
            ret.insert({name, IndexSpace{indices}});
        }

        return ret;
    }

}; // AggregateSpaceImpl

/**
 * @brief IndexSpace implementation for constructing
 *        dependent IndexSpaces
 *
 */
class DependentIndexSpaceImpl : public IndexSpaceInterface {
    public:
    // @todo do we need a default constructor?
    // DependentIndexSpaceImpl() = default;

    /**
     * @brief Construct a new Dependent Index Space Impl object
     *
     * @param [in] indep_spaces a vector of dependent IndexSpace objects
     * @param [in] dep_space_relation a relation map between IndexVectors to
     * IndexSpaces
     */
    DependentIndexSpaceImpl(
      const std::vector<IndexSpace>& indep_spaces,
      const std::map<IndexVector, IndexSpace>& dep_space_relation) :
      dep_spaces_{indep_spaces},
      dep_space_relation_{dep_space_relation} {}

    /**
     * @brief Construct a new Dependent Index Space Impl object
     *
     * @param [in] indep_spaces a vector of dependent IndexSpace objects
     * @param [in] ref_space a reference IndexSpace
     * @param [in] dep_space_relation a relation map between IndexVectors to
     * IndexSpaces
     */
    DependentIndexSpaceImpl(
      const std::vector<IndexSpace>& indep_spaces, const IndexSpace& ref_space,
      const std::map<IndexVector, IndexSpace>& dep_space_relation) :
      dep_spaces_{indep_spaces},
      dep_space_relation_{dep_space_relation} {}

    // @todo do we need these constructor/operators
    DependentIndexSpaceImpl(DependentIndexSpaceImpl&&)      = default;
    DependentIndexSpaceImpl(const DependentIndexSpaceImpl&) = default;
    DependentIndexSpaceImpl& operator=(DependentIndexSpaceImpl&&) = default;
    DependentIndexSpaceImpl& operator=(const DependentIndexSpaceImpl&) =
      default;
    ~DependentIndexSpaceImpl() = default;

    // Index Accessors
    /**
     * @brief Given an Index and a IndexVector return
     *        corresponding Index from the dependent IndexSpace
     *
     * @param i an Index argument
     * @param indep_index a vector of Index
     * @return Index an Index value from the dependent IndexSpace
     */
    Index index(Index i, const IndexVector& indep_index = {}) const override {
        return dep_space_relation_.at(indep_index)[i];
    }

    // @todo what should ve returned?
    Index operator[](Index i) const override {
        NOT_ALLOWED();
        return Index{0};
    }

    // Subspace Accessors
    IndexSpace operator()(const IndexVector& indep_index = {}) const override {
        return dep_space_relation_.at(indep_index);
    }

    // @todo What should this return, currently returning itself
    IndexSpace operator()(const std::string& named_subspace_id) const override {
        return IndexSpace{this_weak_ptr_.lock()};
    }

    // Iterators
    // @todo Error on call
    IndexIterator begin() const override {
        NOT_ALLOWED();
        return IndexIterator();
    }
    IndexIterator end() const override {
        NOT_ALLOWED();
        return IndexIterator();
    }

    // @todo What should this return?
    Index size() const override {
        NOT_ALLOWED();
        return Index{0};
    }

    // Attribute Accessors
    Spin spin(Index idx) const override {
        NOT_ALLOWED();
        return Spin{0};
    }
    Spatial spatial(Index idx) const override {
        NOT_ALLOWED();
        return Spatial{0};
    }

    std::vector<Range> spin_ranges(Spin spin) const override {
        NOT_ALLOWED();
        return {};
    }
    std::vector<Range> spatial_ranges(Spatial spatial) const override {
        NOT_ALLOWED();
        return {};
    }

    bool has_spin() const override {
        NOT_ALLOWED();
        return false;
    }
    bool has_spatial() const override {
        NOT_ALLOWED();
        return false;
    }

    protected:
    std::vector<IndexSpace> dep_spaces_;
    IndexSpace ref_space_;
    std::map<IndexVector, IndexSpace> dep_space_relation_;
}; // DependentIndexSpaceImpl

////////////////////////////////////////////////////////////////
// IndexSpace Method Implementations
// Ctors
IndexSpace::IndexSpace(const IndexVector& indices,
                       const NameToRangeMap& named_subspaces,
                       const AttributeToRangeMap<Spin>& spin,
                       const AttributeToRangeMap<Spatial>& spatial) :
  impl_{std::make_shared<RangeIndexSpaceImpl>(indices, named_subspaces, spin,
                                              spatial)

  } {
    impl_->set_weak_ptr(impl_);
}

IndexSpace::IndexSpace(const IndexSpace& is, const Range& range,
                       const NameToRangeMap& named_subspaces) :
  impl_{std::make_shared<SubSpaceImpl>(is, range, named_subspaces)} {
    impl_->set_weak_ptr(impl_);
}

IndexSpace::IndexSpace(
  const std::vector<IndexSpace>& spaces, const std::vector<std::string>& names,
  const NameToRangeMap& named_subspaces,
  const std::map<std::string, std::vector<std::string>>& subspace_references) :
  impl_{std::make_shared<AggregateSpaceImpl>(spaces, names, named_subspaces,
                                             subspace_references)} {
    impl_->set_weak_ptr(impl_);
}

IndexSpace::IndexSpace(const std::vector<IndexSpace>& indep_spaces,
                       const std::map<Range, IndexSpace>& dep_space_relation) {
    std::map<IndexVector, IndexSpace> ret;
    for(const auto& kv : dep_space_relation) {
        ret.insert({construct_index_vector(kv.first), kv.second});
    }

    impl_ = std::make_shared<DependentIndexSpaceImpl>(indep_spaces, ret);
    impl_->set_weak_ptr(impl_);
}

IndexSpace::IndexSpace(
  const std::vector<IndexSpace>& indep_spaces,
  const std::map<IndexVector, IndexSpace>& dep_space_relation) :
  impl_{std::make_shared<DependentIndexSpaceImpl>(indep_spaces,
                                                  dep_space_relation)} {
    impl_->set_weak_ptr(impl_);
}

IndexSpace::IndexSpace(
  const std::vector<IndexSpace>& indep_spaces, const IndexSpace& ref_space,
  const std::map<IndexVector, IndexSpace>& dep_space_relation) :
  impl_{std::make_shared<DependentIndexSpaceImpl>(indep_spaces, ref_space,
                                                  dep_space_relation)} {
    impl_->set_weak_ptr(impl_);
}

IndexSpace::IndexSpace(const std::vector<IndexSpace>& indep_spaces,
                       const IndexSpace& ref_space,
                       const std::map<Range, IndexSpace>& dep_space_relation) {
    std::map<IndexVector, IndexSpace> ret;
    for(const auto& kv : dep_space_relation) {
        ret.insert({construct_index_vector(kv.first), kv.second});
    }

    impl_ =
      std::make_shared<DependentIndexSpaceImpl>(indep_spaces, ref_space, ret);
    impl_->set_weak_ptr(impl_);
}

// Index Accessors
Index IndexSpace::index(Index i, const IndexVector& indep_index) {
    return impl_->index(i, indep_index);
}
Index IndexSpace::operator[](Index i) const { return impl_->operator[](i); }

// Subspace Accessors
IndexSpace IndexSpace::operator()(const IndexVector& indep_index) const {
    return impl_->operator()(indep_index);
}
IndexSpace IndexSpace::operator()(const std::string& named_subspace_id) const {
    if(named_subspace_id == "all") { return (*this); }
    return impl_->operator()(named_subspace_id);
}

// Iterators
IndexIterator IndexSpace::begin() const { return impl_->begin(); }
IndexIterator IndexSpace::end() const { return impl_->end(); }

// Size of this index space
Index IndexSpace::size() const { return impl_->size(); }

// Attribute Accessors
Spin IndexSpace::spin(Index idx) const { return impl_->spin(idx); }
Spatial IndexSpace::spatial(Index idx) const { return impl_->spatial(idx); }

std::vector<Range> IndexSpace::spin_ranges(Spin spin) const {
    return impl_->spin_ranges(spin);
}
std::vector<Range> IndexSpace::spatial_ranges(Spatial spatial) const {
    return impl_->spatial_ranges(spatial);
}

bool IndexSpace::has_spin() const { return impl_->has_spin(); }
bool IndexSpace::has_spatial() const { return impl_->has_spatial(); }

// Comparison operator implementations
inline bool operator==(const IndexSpace& lhs, const IndexSpace& rhs) {
    return lhs.is_identical(rhs);
}

inline bool operator<(const IndexSpace& lhs, const IndexSpace& rhs) {
    return lhs.is_less_than(rhs);
}

inline bool operator!=(const IndexSpace& lhs, const IndexSpace& rhs) {
    return !(lhs == rhs);
}

inline bool operator>(const IndexSpace& lhs, const IndexSpace& rhs) {
    return !(lhs < rhs) && (lhs != rhs);
}

inline bool operator<=(const IndexSpace& lhs, const IndexSpace& rhs) {
    return (lhs < rhs) || (lhs == rhs);
}

inline bool operator>=(const IndexSpace& lhs, const IndexSpace& rhs) {
    return (lhs > rhs) || (lhs == rhs);
}

////////////////////////////////////////////////////////////////////

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
    TiledIndexSpace(const IndexSpace& is, Tile size = 1) :
      is_{is},
      size_{size} {}

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
    TiledIndexLabel label(std::string id, Label lbl) const;

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
    auto labels(std::string id, Label start = 0) const {
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
    IndexIterator begin() const { return is_.begin(); }

    /**
     * @brief Iterator accessor to the end of the reference IndexSpace
     *
     * @returns a const_iterator to an Index at the size-th element of the
     * IndexSpace
     */
    IndexIterator end() const { return begin() + size_; }

    /**
     * @brief Iterator accessor to the first Index element of a specific block
     *
     * @param [in] blck_ind Index of the block to get const_iterator
     * @returns a const_iterator to the first Index element of the specific
     * block
     */
    IndexIterator block_begin(Index blck_ind) const {
        return is_.begin() + (size_ * blck_ind);
    }
    /**
     * @brief Iterator accessor to the last Index element of a specific block
     *
     * @param [in] blck_ind Index of the block to get const_iterator
     * @returns a const_iterator to the last Index element of the specific
     * block
     */
    IndexIterator block_end(Index blck_ind) const {
        return block_begin(blck_ind) + size_;
    }

    /**
     * @brief Boolean method for checking if given TiledIndexSpace is identical
     * to this TiledIndexSpace
     *
     * @param [in] rhs reference TiledIndexSpace
     * @returns true if the Tile size and the reference IndexSpace is equal
     */
    bool is_identical(const TiledIndexSpace& rhs) const {
        return std::tie(size_, is_) == std::tie(rhs.size_, rhs.is_);
    }

    /**
     * @brief Boolean method for checking if given TiledIndexSpace is subspace
     * of this TiledIndexSpace
     *
     * @param [in] rhs reference TiledIndexSpace
     * @returns true if the Tile size and the reference IndexSpace is equal
     */
    bool is_less_than(const TiledIndexSpace& rhs) const {
        return (size_ == rhs.size_) && (is_ < rhs.is_);
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

    protected:
    IndexSpace is_;
    Tile size_;

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

///////////////////////////////////////////////////////////

inline TiledIndexLabel TiledIndexSpace::label(std::string id, Label lbl) const {
    if(id == "all")
        return TiledIndexLabel{(*this),lbl};
    return TiledIndexLabel{(*this)(id), lbl};
}

template<std::size_t... Is>
auto TiledIndexSpace::labels_impl(std::string id, Label start,
                                  std::index_sequence<Is...>) const {
    return std::make_tuple(label(id, start + Is)...);
}

} // namespace tamm

#endif // INDEX_SPACE_H_ƒ