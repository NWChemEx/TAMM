#ifndef INDEX_SPACE_SKETCH_H_
#define INDEX_SPACE_SKETCH_H_

#include "types.h"
#include <iterator>
#include <map>
#include <memory>

namespace tammy {
/**
 * \brief Type definitios for Index, IndexVector and IndexIterators
 */
using Index         = uint32_t;
using IndexVector   = std::vector<Index>;
using IndexIterator = std::vector<Index>::const_iterator;
using Tile          = uint32_t;

/**
 * \brief Helper methods for Range based constructors.
 *        For now we are using a simple range representation
 *        We will use range constructs from Utilities repo.
 */
struct Range {
    Index lo_;
    Index hi_;
    Index step_;

    Range(Index lo, Index hi, Index step = 1) : lo_{lo}, hi_{hi}, step_{step} {}
};

static bool compare_range(const Range& lhs, const Range& rhs) {
    return lhs.lo_ < rhs.lo_;
}

static Range range(Index lo, Index hi, Index step = 1) {
    return Range(lo, hi, step);
}

static Range range(Index end) { return range(Index{0}, end); }

/**
 * \brief Helper method for constructing IndexVector for a given Range
 *
 * \param range a Range type argument
 * \return IndexVector the Index vector for the corresponding range
 */
static IndexVector construct_index_vector(const Range& range) {
    IndexVector ret = {};
    for(Index i = range.lo_; i < range.hi_; i += range.step_) {
        ret.push_back(i);
    }

    return ret;
}

/**
 * \brief Type definition for the map between range names
 *        and corresponding set of ranges (e.g. "occ", "virt")
 *
 */
using NameToRangeMap = std::map<std::string, const std::vector<Range>>;

template<typename AttributeType>
using AttributeToRangeMap = std::map<AttributeType, std::vector<Range>>;

// TODO: Finalize attribute type
/**
 * \brief Attribute definition which will be used for representing
 *        Spin, Spatial and any other attributes required.
 *
 * \tparam T is an Attribute type (e.g. Spin, Spatial)
 */
template<typename T>
class Attribute {
    public:
    Attribute() = default;
    Attribute(const AttributeToRangeMap<T> attr_map) : attr_map_{attr_map} {}

    Attribute(const Attribute&) = default;
    Attribute& operator=(const Attribute&) = default;
    ~Attribute()                           = default;

    /**
     * \brief Given an Index, ind, user can query
     *        associated Attribute value
     *
     * \param ind Index argument
     * \return T  associated Attribute value
     */
    T operator()(Index ind) {
        for(const auto& kv : attr_map_) {
            for(const auto& vec : kv.second) {
                if(std::find(vec.begin(), vec.end(), ind) != vec.end()) {
                    return kv.first;
                }
            }
        }

        return T{0};
    }

    /**
     * \brief Getting the iterators for Index vectors
     *        for a given Attribute value (e.g. Spin{1})
     *
     * \param val an Attribute value as an argument
     * \return std::vector<IndexVector>::const_iterator returns an iterator for
     * set of Index vectors
     */
    std::vector<Range>::const_iterator begin(const T& val) {
        return attr_map_[val].begin();
    }
    std::vector<Range>::const_iterator end(const T& val) {
        return attr_map_[val].end();
    }

    protected:
    AttributeToRangeMap<T> attr_map_;
}; // Attribute

/**
 * \brief Type definitions for Spin and Spatial attributes
 *
 */
using SpinAttribute    = Attribute<Spin>;
using SpatialAttribute = Attribute<Spatial>;

class IndexSpace;

/**
 * \brief Base abstract implementation class as an interface
 *        for the different IndexSpace implementations
 *
 */
class IndexSpaceImpl {
    public:
    virtual ~IndexSpaceImpl(){};

    // Index Accessors (e.g. MO[Index{10}])
    virtual Index point(Index i, const IndexVector& indep_index = {}) const = 0;
    virtual Index operator[](Index i) const                                 = 0;

    // Subspace Accesors (e.g. MO("occ"))
    virtual IndexSpace operator()(
      const IndexVector& indep_index = {}) const = 0;
    virtual IndexSpace operator()(
      const std::string& named_subspace_id) const = 0;

    // Iterators
    virtual IndexIterator begin() const = 0;
    virtual IndexIterator end() const   = 0;
    // Size of this index space
    virtual Index size() const = 0;

    // Attribute Accesors
    virtual SpinAttribute get_spin() const       = 0;
    virtual SpatialAttribute get_spatial() const = 0;

    protected:
    std::weak_ptr<IndexSpaceImpl> this_weak_ptr_;

    private:
    void set_weak_ptr(std::weak_ptr<IndexSpaceImpl> weak_ptr) {
        this_weak_ptr_ = weak_ptr;
    }
    friend class IndexSpace;
}; // IndexSpaceImpl

/**
 * \brief Forward class declarations for different types
 *        of IndexSpace implementations
 */
class RangeIndexSpaceImpl;
class SubSpaceImpl;
class AggregateSpaceImpl;
class DependentIndexSpaceImpl;

/**
 * \brief Main IndexSpace class that users will be using
 *        Implemented using PIMPL idiom.
 */
class IndexSpace {
    public:
    // Constructors
    IndexSpace() = default;

    IndexSpace(const IndexSpace&) = default;
    IndexSpace(IndexSpace&&)      = default;
    ~IndexSpace()                 = default;
    IndexSpace& operator=(const IndexSpace&) = default;
    IndexSpace& operator=(IndexSpace&&) = default;

    // creating named sub-space groups
    // "inheriting" names sub-spaces
    // same as above for attributes

    // Initializer-list / vector based. no inherited named subspaces
    IndexSpace(const std::initializer_list<Index>& indices) :
      IndexSpace{indices, {}, {}, {}} {}

    IndexSpace(const Range& range, const NameToRangeMap& named_subspaces = {},
               const AttributeToRangeMap<Spin>& spin       = {},
               const AttributeToRangeMap<Spatial>& spatial = {}) :
      IndexSpace{construct_index_vector(range), named_subspaces, spin,
                 spatial} {}

    IndexSpace(const IndexVector& indices,
               const NameToRangeMap& named_subspaces       = {},
               const AttributeToRangeMap<Spin>& spin       = {},
               const AttributeToRangeMap<Spatial>& spatial = {}) :
      impl_{std::dynamic_pointer_cast<IndexSpaceImpl>(
        std::make_shared<RangeIndexSpaceImpl>(indices, named_subspaces, spin,
                                              spatial))} {
        impl_->set_weak_ptr(impl_);
    }

    // Sub-space. no inherited named subspaces
    // all attributes in \param{is} are inherited.
    // to iterate over the ranges into which an attribute partitions this index
    // space, the parent space's attributes need to "chopped" to match the
    // \param{range}.
    // TODO: we also need string based named subspaces. e.g. "alpha" =
    // "occ_alpha, virt_alpha".
    IndexSpace(const IndexSpace& is, const Range& range,
               const NameToRangeMap& named_subspaces = {}) :
      impl_{std::dynamic_pointer_cast<IndexSpaceImpl>(
        std::make_shared<SubSpaceImpl>(is, range, named_subspaces))} {
        impl_->set_weak_ptr(impl_);
    }

    // Aggregate. named subspaces from all space in \param{spaces} with a
    // non-empty name are "inherited" any attributes in all sp in \param{spaces}
    // is inherited. If any of the aggregated spaces does not have an attribute,
    // that attribute is not inherited.
    // TODO: we could have functions to get "named" subspaces by position.
    // Basically fn(i) returns spaces[i]. IndexSpace(const
    IndexSpace(const std::vector<IndexSpace>& spaces,
               const std::vector<std::string>& names = {},
               const NameToRangeMap& named_subspaces = {},
               const std::map<std::string, std::vector<std::string>>&
                 subspace_references = {}) :
      impl_{std::dynamic_pointer_cast<IndexSpaceImpl>(
        std::make_shared<AggregateSpaceImpl>(spaces, names, named_subspaces,
                                             subspace_references))} {
        impl_->set_weak_ptr(impl_);
    }

    // Dependent : what about attributes here
    // named subspaces in dep_space_relation are "inherited" by default. Note
    // that the index spaces in dep_space_relation might have no relation with
    // one another. Attributes in dep_space_relation are inherited by default.
    IndexSpace(const std::vector<IndexSpace>& indep_spaces,
               const std::map<IndexVector, IndexSpace>& dep_space_relation) :
      impl_{std::dynamic_pointer_cast<IndexSpaceImpl>(
        std::make_shared<DependentIndexSpaceImpl>(indep_spaces,
                                                  dep_space_relation))} {
        impl_->set_weak_ptr(impl_);
    }

    // Dependent subspace : what about attributes here
    // named subspaces in dep_space_relation are "inherited" by default. Note
    // that the index spaces in dep_space_relation are all subspaces of
    // ref_space.
    // all attributes in \param{ref_space} are "inherited". see also the
    // sub-space constructor comments.
    IndexSpace(const std::vector<IndexSpace>& indep_spaces,
               const IndexSpace& ref_space,
               const std::map<IndexVector, IndexSpace>& dep_space_relation) :
      impl_{std::dynamic_pointer_cast<IndexSpaceImpl>(
        std::make_shared<DependentIndexSpaceImpl>(indep_spaces, ref_space,
                                                  dep_space_relation))} {
        impl_->set_weak_ptr(impl_);
    }

    // constructor to wrap a shared_ptr
    IndexSpace(const std::shared_ptr<IndexSpaceImpl> impl) : impl_{impl} {}

    // Index Accessors
    Index point(Index i, const IndexVector& indep_index = {});
    Index operator[](Index i) const;

    // Subspace Accessors
    IndexSpace operator()(const IndexVector& indep_index = {}) const;
    IndexSpace operator()(const std::string& named_subspace_id) const;

    // Iterators
    IndexIterator begin() const;
    IndexIterator end() const;

    // Size of this index space
    Index size() const;

    // Attribute Accesors
    SpinAttribute get_spin() const;
    SpatialAttribute get_spatial() const;

    bool is_identical(const IndexSpace& rhs) const {
        return impl_ == rhs.impl_;
    }

    bool is_less_than(const IndexSpace& rhs) const { return impl_ < rhs.impl_; }

    // TODO: implement
    bool is_compatible(const IndexSpace& rhs) const {
        // NOT_IMPLEMENTED();
        return true;
    }

    protected:
    std::shared_ptr<IndexSpaceImpl> impl_;
}; // IndexSpace

/**
 * \brief IndexSpace implementation for range based
 *        IndexSpace construction.
 *
 */
class RangeIndexSpaceImpl : public IndexSpaceImpl {
    public:
    // TODO: do we need a default constructor?
    // RangeIndexSpaceImpl() = default;

    // Range-based. no inherited named subspaces
    // TODO: optimization - constructing map<string, IndexSpace> can be delayed
    // until a specific subspace is requested
    RangeIndexSpaceImpl(const IndexVector& indices,
                        const NameToRangeMap& named_ranges,
                        const AttributeToRangeMap<Spin>& spin,
                        const AttributeToRangeMap<Spatial>& spatial) :
      indices_{indices},
      named_ranges_{named_ranges},
      named_subspaces_{construct_subspaces(named_ranges)},
      spin_{construct_spin(spin)},
      spatial_{construct_spatial(spatial)} {}

    // TODO : do we need these copy/move constructor/operators?
    RangeIndexSpaceImpl(RangeIndexSpaceImpl&&)      = default;
    RangeIndexSpaceImpl(const RangeIndexSpaceImpl&) = default;
    RangeIndexSpaceImpl& operator=(RangeIndexSpaceImpl&&) = default;
    RangeIndexSpaceImpl& operator=(const RangeIndexSpaceImpl&) = default;
    ~RangeIndexSpaceImpl()                                     = default;

    // Index Accesors
    Index point(Index i, const IndexVector& indep_index = {}) const {
        return indices_[i];
    }
    Index operator[](Index i) const { return indices_[i]; }

    // Subspace Accesors
    IndexSpace operator()(const IndexVector& indep_index = {}) const {
        return IndexSpace{this_weak_ptr_.lock()};
    }

    IndexSpace operator()(const std::string& named_subspace_id) const {
        return named_subspaces_.at(named_subspace_id);
    }

    // Iterators
    IndexIterator begin() const { return indices_.begin(); }
    IndexIterator end() const { return indices_.end(); }

    // Size of this index space
    Index size() const { return indices_.size(); }

    // Attribute Accesors
    SpinAttribute get_spin() const { return spin_; }
    SpatialAttribute get_spatial() const { return spatial_; }

    protected:
    IndexVector indices_;
    NameToRangeMap named_ranges_;
    std::map<std::string, IndexSpace> named_subspaces_;
    SpinAttribute spin_;
    SpatialAttribute spatial_;

    /**
     * \brief Helper method for generating the map between
     *        string values to IndexSpaces. Mainly used for
     *        constructing the subspaces.
     *
     * \param in_map NameToRangeMap argument holding string to Range map
     * \return std::map<std::string, IndexSpace> returns the map from
     *                                           strings to subspaces
     */
    std::map<std::string, IndexSpace> construct_subspaces(
      const NameToRangeMap& in_map) {
        std::map<std::string, IndexSpace> ret;
        for(auto& kv : in_map) {
            std::string name      = kv.first;
            IndexVector tempRange = {};
            for(auto& range : kv.second) {
                for(auto& i : construct_index_vector(range)) {
                    tempRange.push_back(indices_[i]);
                }
            }
            ret.insert({name, IndexSpace{tempRange}});
        }

        return ret;
    }

    /**
     * \brief Helper method for constructing the attributes for
     *        new subspace by "chopping" the attributes from the
     *        reference IndexSpace
     *
     * \param is reference IndexSpace argument
     * \param range Range argument
     * \return Attribute an "chopped" version of the reference Attribute
     */
    SpinAttribute construct_spin(const AttributeToRangeMap<Spin>& spin) {
        // return default spin value (Spin{0}) for the whole range
        if(spin.empty()) {
            return SpinAttribute({{Spin{0}, {range(indices_.size())}}});
        }

        // TODO: validation -> no overlap, full space coverage

        return SpinAttribute{spin};
    }
    // TODO: Implement depending on Attribute structure
    SpatialAttribute construct_spatial(
      const AttributeToRangeMap<Spatial>& spatial) {
        // return default spatial value (Spatial{0}) for the whole range
        if(spatial.empty()) {
            return SpatialAttribute({{Spatial{0}, {range(indices_.size())}}});
        }
        // TODO: validation -> no overlap, full space coverage

        return SpatialAttribute{spatial};
    }
}; // RangeIndexSpaceImpl

/**
 * \brief IndexSpace implementation for subspace based
 *        IndexSpace construction.
 *
 */
class SubSpaceImpl : public IndexSpaceImpl {
    public:
    // TODO: do we need a default constructor?
    // SubSpaceImpl() = default;

    // Sub-space construction
    // TODO: optimization - constructing map<string, IndexSpace> can be delayed
    // until a specific subspace is requested
    SubSpaceImpl(const IndexSpace& is, const Range& range,
                 const NameToRangeMap& named_ranges) :
      ref_space_{is},
      indices_{construct_indices(is, range)},
      named_ranges_{named_ranges},
      named_subspaces_{construct_subspaces(named_ranges)},
      spin_{construct_spin(is, range)},
      spatial_{construct_spatial(is, range)} {}

    // TODO : do we need these copy/move constructor/operators
    SubSpaceImpl(SubSpaceImpl&&)      = default;
    SubSpaceImpl(const SubSpaceImpl&) = default;
    SubSpaceImpl& operator=(SubSpaceImpl&&) = default;
    SubSpaceImpl& operator=(const SubSpaceImpl&) = default;
    ~SubSpaceImpl()                              = default;

    // Index Accesors
    Index point(Index i, const IndexVector& indep_index = {}) const {
        return indices_[i];
    }
    Index operator[](Index i) const { return indices_[i]; }

    // Subspace Accesors
    IndexSpace operator()(const IndexVector& indep_index = {}) const {
        return IndexSpace{this_weak_ptr_.lock()};
    }

    IndexSpace operator()(const std::string& named_subspace_id) const {
        return named_subspaces_.at(named_subspace_id);
    }

    // Iterators
    IndexIterator begin() const { return indices_.begin(); }
    IndexIterator end() const { return indices_.end(); }

    // Size of this index space
    Index size() const { return indices_.size(); }

    // Attribute Accesors
    SpinAttribute get_spin() const { return spin_; }
    SpatialAttribute get_spatial() const { return spatial_; }

    protected:
    IndexSpace ref_space_;
    IndexVector indices_;
    NameToRangeMap named_ranges_;
    std::map<std::string, IndexSpace> named_subspaces_;
    SpinAttribute spin_;
    SpatialAttribute spatial_;

    /**
     * \brief Helper method for constructing the new set of
     *        indicies from the reference IndexSpace
     *
     * \param ref_space reference IndexSpace argument
     * \param range     Range argument for generating the subspace
     * \return IndexVector returns a vector of Indicies
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
     * \brief Helper method for generating the map between
     *        string values to IndexSpaces. Mainly used for
     *        constructing the subspaces.
     *
     * \param in_map NameToRangeMap argument holding string to Range map
     * \return std::map<std::string, IndexSpace> returns the map from
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

    // TODO: Implement depending on Attribute structure
    /**
     * \brief Helper method for constructing the attributes for
     *        new subspace by "chopping" the attributes from the
     *        reference IndexSpace
     *
     * \param is reference IndexSpace argument
     * \param range Range argument
     * \return Attribute an "chopped" version of the reference Attribute
     */
    SpinAttribute construct_spin(const IndexSpace& is, const Range& range) {
        // NOT_IMPLEMENTED();
        return SpinAttribute{};
    }
    // TODO: Implement depending on Attribute structure
    SpatialAttribute construct_spatial(const IndexSpace& is,
                                       const Range& range) {
        // NOT_IMPLEMENTED();
        return SpatialAttribute{};
    }
}; // SubSpaceImpl

/**
 * \brief IndexSpace implementation for aggregation
 *        based IndexSpace construction.
 *
 */
class AggregateSpaceImpl : public IndexSpaceImpl {
    public:
    // TODO: do we need a default constructor?
    // AggregateSpaceImpl() = default;

    // IndexSpace aggregation construction
    // TODO: optimization - constructing map<string, IndexSpace> can be delayed
    // until a specific subspace is requested
    AggregateSpaceImpl(const std::vector<IndexSpace>& spaces,
                       const std::vector<std::string>& names,
                       const NameToRangeMap& named_ranges,
                       const std::map<std::string, std::vector<std::string>>&
                         subspace_references) :
      ref_spaces_{spaces},
      indices_{construct_indices(spaces)},
      named_ranges_{named_ranges},
      named_subspaces_{construct_subspaces(named_ranges)},
      spin_{construct_spin(spaces)},
      spatial_{construct_spatial(spaces)} {
        if(names.size() > 0) { add_ref_names(spaces, names); }
        if(subspace_references.size() > 0) {
            add_subspace_references(subspace_references);
        }
    }

    // TODO : do we need these constructor/operators
    AggregateSpaceImpl(AggregateSpaceImpl&&)      = default;
    AggregateSpaceImpl(const AggregateSpaceImpl&) = default;
    AggregateSpaceImpl& operator=(AggregateSpaceImpl&&) = default;
    AggregateSpaceImpl& operator=(const AggregateSpaceImpl&) = default;
    ~AggregateSpaceImpl()                                    = default;

    // Index Accessors
    Index point(Index i, const IndexVector& indep_index = {}) const {
        return indices_[i];
    }
    Index operator[](Index i) const { return indices_[i]; }

    // Subspace Accessors
    IndexSpace operator()(const IndexVector& indep_index = {}) const {
        return IndexSpace{this_weak_ptr_.lock()};
    }

    IndexSpace operator()(const std::string& named_subspace_id) const {
        return named_subspaces_.at(named_subspace_id);
    }

    // Iterators
    IndexIterator begin() const { return indices_.begin(); }
    IndexIterator end() const { return indices_.end(); }

    // Size of this index space
    Index size() const { return indices_.size(); }

    // Attribute Accesors
    SpinAttribute get_spin() const { return spin_; }
    SpatialAttribute get_spatial() const { return spatial_; }

    protected:
    std::vector<IndexSpace> ref_spaces_;
    IndexVector indices_;
    NameToRangeMap named_ranges_;
    std::map<std::string, IndexSpace> named_subspaces_;
    SpinAttribute spin_;
    SpatialAttribute spatial_;

    /**
     * \brief Add subspaces reference names foreach aggregated
     *        IndexSpace
     *
     * \param ref_spaces a vector of reference IndexSpaces
     * \param ref_names  a vector of associated names for each
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

    // TODO: Implement based on the discussion
    /**
     * \brief Add extra references for subspace names
     *        associated with the reference subspaces.
     *
     * \param subspace_references a map from subspace names
     *                            to reference subspace names
     */
    void add_subspace_references(
      const std::map<std::string, std::vector<std::string>>&
        subspace_references) { /*NOT_IMPLEMENTED();*/
    }

    /**
     * \brief Construct set of indicies from the aggregated
     *        IndexSpaces
     *
     * \param spaces vector of IndexSpaces
     * \return IndexVector returns a vector Index objects
     */
    IndexVector construct_indices(const std::vector<IndexSpace>& spaces) {
        IndexVector ret = {};
        for(const auto& space : spaces) {
            ret.insert(ret.end(), space.begin(), space.end());
        }

        return ret;
    }

    /**
     * \brief Helper method for generating the map between
     *        string values to IndexSpaces. Mainly used for
     *        constructing the subspaces.
     *
     * \param in_map NameToRangeMap argument holding string to Range map
     * \return std::map<std::string, IndexSpace> returns the map from
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

    // TODO: Implement depending on Attribute structure
    /**
     * \brief Helper method for constructing the attributes for
     *        new aggregated space by combining the attributes
     *
     * \param spaces vector of IndexSpace argument
     * \return Attribute aggregated Attributes.
     */
    SpinAttribute construct_spin(const std::vector<IndexSpace>& spaces) {
        // NOT_IMPLEMENTED();
        return SpinAttribute{};
    }
    // TODO: Implement depending on Attribute structure
    SpatialAttribute construct_spatial(const std::vector<IndexSpace>& spaces) {
        // NOT_IMPLEMENTED();
        return SpatialAttribute{};
    }

}; // AggregateSpaceImpl

/**
 * \brief IndexSpace implementation for constructing
 *        dependent IndexSpaces
 *
 */
class DependentIndexSpaceImpl : public IndexSpaceImpl {
    public:
    // TODO: do we need a default constructor?
    // DependentIndexSpaceImpl() = default;

    // IndexSpace constructor for dependent spaces
    DependentIndexSpaceImpl(
      const std::vector<IndexSpace>& indep_spaces,
      const std::map<IndexVector, IndexSpace>& dep_space_relation) :
      dep_spaces_{indep_spaces},
      dep_space_relation_{dep_space_relation} {}

    DependentIndexSpaceImpl(
      const std::vector<IndexSpace>& indep_spaces, const IndexSpace& ref_space,
      const std::map<IndexVector, IndexSpace>& dep_space_relation) :
      dep_spaces_{indep_spaces},
      dep_space_relation_{dep_space_relation} {}

    // TODO : do we need these constructor/operators
    DependentIndexSpaceImpl(DependentIndexSpaceImpl&&)      = default;
    DependentIndexSpaceImpl(const DependentIndexSpaceImpl&) = default;
    DependentIndexSpaceImpl& operator=(DependentIndexSpaceImpl&&) = default;
    DependentIndexSpaceImpl& operator=(const DependentIndexSpaceImpl&) =
      default;
    ~DependentIndexSpaceImpl() = default;

    // Index Accessors
    /**
     * \brief Given an Index and a IndexVector return
     *        corresponding Index from the dependent IndexSpace
     *
     * \param i an Index argument
     * \param indep_index a vector of Index
     * \return Index an Index value from the dependent IndexSpace
     */
    Index point(Index i, const IndexVector& indep_index = {}) const {
        return dep_space_relation_.at(indep_index)[i];
    }

    // TODO: what should ve returned?
    Index operator[](Index i) const { NOT_IMPLEMENTED(); }

    // Subspace Accessors
    IndexSpace operator()(const IndexVector& indep_index = {}) const {
        return dep_space_relation_.at(indep_index);
    }

    // TODO: What should this return?
    IndexSpace operator()(const std::string& named_subspace_id) const {
        return IndexSpace{this_weak_ptr_.lock()};
    }

    // Iterators
    // TODO: Error on call
    IndexIterator begin() const { NOT_IMPLEMENTED(); }
    IndexIterator end() const { NOT_IMPLEMENTED(); }

    // TODO: What should this return?
    Index size() const { NOT_IMPLEMENTED(); }

    // Attribute Accesors
    SpinAttribute get_spin() const { NOT_IMPLEMENTED(); }
    SpatialAttribute get_spatial() const { NOT_IMPLEMENTED(); }

    protected:
    std::vector<IndexSpace> dep_spaces_;
    IndexSpace ref_space_;
    std::map<IndexVector, IndexSpace> dep_space_relation_;
}; // DependentIndexSpaceImpl

//////////////////////////////////////////////////////////////////////////////////
// IndexSpace Method Implementations
// Index Accessors
Index IndexSpace::point(Index i, const IndexVector& indep_index) {
    return impl_->point(i, indep_index);
}
Index IndexSpace::operator[](Index i) const { return impl_->operator[](i); }

// Subspace Accessors
IndexSpace IndexSpace::operator()(const IndexVector& indep_index) const {
    return impl_->operator()(indep_index);
}
IndexSpace IndexSpace::operator()(const std::string& named_subspace_id) const {
    return impl_->operator()(named_subspace_id);
}

// Iterators
IndexIterator IndexSpace::begin() const { return impl_->begin(); }
IndexIterator IndexSpace::end() const { return impl_->end(); }

// Size of this index space
Index IndexSpace::size() const { return impl_->size(); }

// Attribute Accesors
SpinAttribute IndexSpace::get_spin() const { return impl_->get_spin(); }
SpatialAttribute IndexSpace::get_spatial() const {
    return impl_->get_spatial();
}

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

class TiledIndexSpace {
    public:
    // Ctors
    TiledIndexSpace() = default;

    // IndexSpace based
    TiledIndexSpace(const IndexSpace& is, Tile size = 1) :
      is_{is},
      size_{size} {}

    // Sub-space
    TiledIndexSpace(const TiledIndexSpace& t_is, const Range& range,
                    Tile size = 1) :
      TiledIndexSpace(IndexSpace{t_is.is_, range}, size) {}
    TiledIndexSpace(const TiledIndexSpace& t_is, const std::string& id,
                    Tile size = 1) :
      TiledIndexSpace(t_is.is_(id), size) {}

    // Copy Ctors
    TiledIndexSpace(const TiledIndexSpace&) = default;
    TiledIndexSpace& operator=(const TiledIndexSpace&) = default;

    // Dtor
    ~TiledIndexSpace() = default;

    // Get labels for subspace
    TiledIndexLabel labels(std::string id, Label lbl) const;

    // TODO: Implement with variadic templates
    template<unsigned c_lbl>
    auto range_labels(std::string id, Label start = 0);

    TiledIndexSpace operator()(std::string id) const {
        return TiledIndexSpace((*this), id);
    }

    // Iterators
    IndexIterator begin() { NOT_IMPLEMENTED(); }
    IndexIterator end() { NOT_IMPLEMENTED(); }

    private:
    IndexSpace is_;
    Tile size_;

}; // TiledIndexSpace

// TODO:  Implement based on IndexSpace comparison operators
inline bool operator==(const TiledIndexSpace& lhs, const TiledIndexSpace& rhs);
inline bool operator<(const TiledIndexSpace& lhs, const TiledIndexSpace& rhs);

inline bool operator!=(const TiledIndexSpace& lhs, const TiledIndexSpace& rhs) {
    return !(lhs == rhs);
}

inline bool operator>(const TiledIndexSpace& lhs, const TiledIndexSpace& rhs) {
    return !(lhs < rhs) && (lhs != rhs);
}

inline bool operator<=(const TiledIndexSpace& lhs, const TiledIndexSpace& rhs) {
    return (lhs < rhs) || (lhs == rhs);
}

inline bool operator>=(const TiledIndexSpace& lhs, const TiledIndexSpace& rhs) {
    return (lhs > rhs) || (lhs == rhs);
}

class TiledIndexLabel {
    public:
    // Constructor
    TiledIndexLabel() = default;

    TiledIndexLabel(TiledIndexSpace t_is, Label lbl,
                    const std::vector<TiledIndexLabel> dep_labels = {}) :
      tis_{t_is},
      label_{lbl},
      dep_labels_{dep_labels} {}

    TiledIndexLabel(TiledIndexLabel t_il,
                    std::vector<TiledIndexLabel> dep_labels) :
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
    TiledIndexLabel operator()() const {
        return {*this};
    }

    TiledIndexLabel operator()(TiledIndexLabel il1, TiledIndexLabel il2) const {
        return TiledIndexLabel{*this, {il1,il2}};
    }

    private:
    TiledIndexSpace tis_;
    Label label_;
    std::vector<TiledIndexLabel> dep_labels_;
}; // TiledIndexLabel

// TODO: Implement based on TiledIndexSpace comparison operators
inline bool operator==(const TiledIndexLabel& lhs, const TiledIndexLabel& rhs);
inline bool operator<(const TiledIndexLabel& lhs, const TiledIndexLabel& rhs);

inline bool operator!=(const TiledIndexLabel& lhs, const TiledIndexLabel& rhs) {
    return !(lhs == rhs);
}

inline bool operator>(const TiledIndexLabel& lhs, const TiledIndexLabel& rhs) {
    return !(lhs < rhs) && (lhs != rhs);
}

inline bool operator<=(const TiledIndexLabel& lhs, const TiledIndexLabel& rhs) {
    return (lhs < rhs) || (lhs == rhs);
}

inline bool operator>=(const TiledIndexLabel& lhs, const TiledIndexLabel& rhs) {
    return (lhs > rhs) || (lhs == rhs);
}

///////////////////////////////////////////////////////////

inline TiledIndexLabel TiledIndexSpace::labels(std::string id,
                                               Label lbl) const {
    return TiledIndexLabel{(*this)(id), lbl};
}

} // namespace tammy

#endif // INDEX_SPACE_SKETCH_H_