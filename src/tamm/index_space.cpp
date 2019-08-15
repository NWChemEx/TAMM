#include "tamm/index_space_interface.hpp"
#include "tamm/tiled_index_space.hpp"

namespace tamm {
// IndexSpace Method Implementations
// Ctors
IndexSpace::IndexSpace(const IndexVector& indices,
                       const NameToRangeMap& named_subspaces,
                       const AttributeToRangeMap<Spin>& spin,
                       const AttributeToRangeMap<Spatial>& spatial) :
  impl_{std::make_shared<RangeIndexSpaceImpl>(indices, named_subspaces, spin,
                                              spatial)

  } {
    hash_value_ = impl_->hash();
    impl_->set_weak_ptr(impl_);
}

IndexSpace::IndexSpace(const IndexSpace& is, const Range& range,
                       const NameToRangeMap& named_subspaces) :
  impl_{std::make_shared<SubSpaceImpl>(is, range, named_subspaces)} {
    hash_value_ = impl_->hash();    
    impl_->set_weak_ptr(impl_);
}

IndexSpace::IndexSpace(const IndexSpace& is, const IndexVector& indices,
                       const NameToRangeMap& named_subspaces) :
  impl_{std::make_shared<SubSpaceImpl>(is, indices, named_subspaces)} {
    hash_value_ = impl_->hash();    
    impl_->set_weak_ptr(impl_);
}

IndexSpace::IndexSpace(
  const std::vector<IndexSpace>& spaces, const std::vector<std::string>& names,
  const NameToRangeMap& named_subspaces,
  const std::map<std::string, std::vector<std::string>>& subspace_references) :
  impl_{std::make_shared<AggregateSpaceImpl>(spaces, names, named_subspaces,
                                             subspace_references)} {
    hash_value_ = impl_->hash();    
    impl_->set_weak_ptr(impl_);
}

IndexSpace::IndexSpace(const std::vector<TiledIndexSpace>& indep_spaces,
                       const std::map<Range, IndexSpace>& dep_space_relation) {
    std::map<IndexVector, IndexSpace> ret;
    for(const auto& kv : dep_space_relation) {
        ret.insert({construct_index_vector(kv.first), kv.second});
    }

    impl_ = std::make_shared<DependentIndexSpaceImpl>(indep_spaces, ret);
    hash_value_ = impl_->hash();
    impl_->set_weak_ptr(impl_);
}

IndexSpace::IndexSpace(
  const std::vector<TiledIndexSpace>& indep_spaces,
  const std::map<IndexVector, IndexSpace>& dep_space_relation) :
  impl_{std::make_shared<DependentIndexSpaceImpl>(indep_spaces,
                                                  dep_space_relation)} {
    hash_value_ = impl_->hash();    
    impl_->set_weak_ptr(impl_);
}

IndexSpace::IndexSpace(
  const std::vector<TiledIndexSpace>& indep_spaces, const IndexSpace& ref_space,
  const std::map<IndexVector, IndexSpace>& dep_space_relation) :
  impl_{std::make_shared<DependentIndexSpaceImpl>(indep_spaces, ref_space,
                                                  dep_space_relation)} {
    hash_value_ = impl_->hash();    
    impl_->set_weak_ptr(impl_);
}

IndexSpace::IndexSpace(const std::vector<TiledIndexSpace>& indep_spaces,
                       const IndexSpace& ref_space,
                       const std::map<Range, IndexSpace>& dep_space_relation) {
    std::map<IndexVector, IndexSpace> ret;
    for(const auto& kv : dep_space_relation) {
        ret.insert({construct_index_vector(kv.first), kv.second});
    }

    impl_ =
      std::make_shared<DependentIndexSpaceImpl>(indep_spaces, ref_space, ret);
    hash_value_ = impl_->hash();    
    impl_->set_weak_ptr(impl_);
}

const std::vector<TiledIndexSpace>& IndexSpace::key_tiled_index_spaces() const {
    return impl_->key_tiled_index_spaces();
}

size_t IndexSpace::num_key_tiled_index_spaces() const {
    return impl_->num_key_tiled_index_spaces();
}

const std::map<IndexVector, IndexSpace>& IndexSpace::map_tiled_index_spaces()
  const {
    return impl_->map_tiled_index_spaces();
}

const std::map<std::string, IndexSpace>&
IndexSpace::map_named_sub_index_spaces() const {
    return impl_->map_named_sub_index_spaces();
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
std::size_t IndexSpace::num_indices() const { return impl_->num_indices(); }

// Maximum size of this index space for any dependent index
std::size_t IndexSpace::max_num_indices() const {
    return impl_->max_num_indices();
}

// Attribute Accessors
Spin IndexSpace::spin(Index idx) const { return impl_->spin(idx); }
Spatial IndexSpace::spatial(Index idx) const { return impl_->spatial(idx); }

const std::vector<Range>& IndexSpace::spin_ranges(Spin spin) const {
    return impl_->spin_ranges(spin);
}

const std::vector<Range>& IndexSpace::spatial_ranges(Spatial spatial) const {
    return impl_->spatial_ranges(spatial);
}

bool IndexSpace::has_spin() const { return impl_->has_spin(); }

bool IndexSpace::has_spatial() const { return impl_->has_spatial(); }

SpinAttribute IndexSpace::get_spin() const { return impl_->get_spin(); }

SpatialAttribute IndexSpace::get_spatial() const {
    return impl_->get_spatial();
}

const NameToRangeMap& IndexSpace::get_named_ranges() const {
    return impl_->get_named_ranges();
}

IndexSpace IndexSpace::root_index_space() const {
    return impl_->root_index_space();
}

// Comparison operator implementations
bool operator==(const IndexSpace& lhs, const IndexSpace& rhs) {
    return lhs.is_identical(rhs);
}

bool operator<(const IndexSpace& lhs, const IndexSpace& rhs) {
    return lhs.is_less_than(rhs);
}

bool operator!=(const IndexSpace& lhs, const IndexSpace& rhs) {
    return !(lhs == rhs);
}

bool operator>(const IndexSpace& lhs, const IndexSpace& rhs) {
    return !(lhs < rhs) && (lhs != rhs);
}

bool operator<=(const IndexSpace& lhs, const IndexSpace& rhs) {
    return (lhs < rhs) || (lhs == rhs);
}

bool operator>=(const IndexSpace& lhs, const IndexSpace& rhs) {
    return (lhs > rhs) || (lhs == rhs);
}

} // namespace tamm
