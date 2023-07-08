#pragma once

#include "tamm/index_space.hpp"
#include "tamm/symbol.hpp"
#include <set>

namespace tamm {

class TiledIndexLabel;

/**
 * @brief TiledIndexSpace class
 *
 */
class TiledIndexSpace {
public:
  TiledIndexSpace()                                  = default;
  TiledIndexSpace(const TiledIndexSpace&)            = default;
  TiledIndexSpace(TiledIndexSpace&&)                 = default;
  TiledIndexSpace& operator=(TiledIndexSpace&&)      = default;
  TiledIndexSpace& operator=(const TiledIndexSpace&) = default;
  ~TiledIndexSpace()                                 = default;

  /**
   * @brief Construct a new TiledIndexSpace from
   * a reference IndexSpace and a tile size
   *
   * @param [in] is reference IndexSpace
   * @param [in] input_tile_size tile size (default: 1)
   */
  TiledIndexSpace(const IndexSpace& is, Tile input_tile_size = 1):
    tiled_info_{std::make_shared<TiledIndexSpace::TiledIndexSpaceInfo>(is, input_tile_size,
                                                                       std::vector<Tile>{})},
    root_tiled_info_{tiled_info_},
    parent_tis_{nullptr} {
    EXPECTS(input_tile_size > 0);
    compute_hash();
    // construct tiling for named subspaces
    tile_named_subspaces(is);
  }

  /**
   * @brief Construct a new TiledIndexSpace from a reference
   * IndexSpace and varying tile sizes
   *
   * @param [in] is reference IndexSpace
   * @param [in] input_tile_sizes list of tile sizes
   */
  TiledIndexSpace(const IndexSpace& is, const std::vector<Tile>& input_tile_sizes):
    tiled_info_{std::make_shared<TiledIndexSpace::TiledIndexSpaceInfo>(is, 0, input_tile_sizes)},
    root_tiled_info_{tiled_info_},
    parent_tis_{nullptr} {
    for(const auto& in_tsize: input_tile_sizes) { EXPECTS(in_tsize > 0); }
    compute_hash();
    // construct tiling for named subspaces
    tile_named_subspaces(is);
  }

  /**
   * @brief Construct a new TiledIndexSpace object from a dependent IndexSpace
   * with different list of tile sizes for each dependency
   *
   * @param [in] is reference dependent IndexSpace
   * @param [in] dep_tile_sizes map from dependent index id to tile sizes
   */
  TiledIndexSpace(const IndexSpace&                               is,
                  const std::map<IndexVector, std::vector<Tile>>& dep_tile_sizes):
    tiled_info_{std::make_shared<TiledIndexSpace::TiledIndexSpaceInfo>(is, dep_tile_sizes)},
    root_tiled_info_{tiled_info_},
    parent_tis_{nullptr} {
    for(const auto& [idx_vec, tile_sizes]: dep_tile_sizes) { EXPECTS(tile_sizes.size() > 0); }
    compute_hash();
    // construct tiling for named subspaces
    // tile_named_subspaces(is);
  }

  /**
   * @brief Construct a new sub TiledIndexSpace object from
   * a sub-space of a reference TiledIndexSpace
   *
   * @param [in] t_is reference TiledIndexSpace
   * @param [in] range Range of the reference TiledIndexSpace
   */
  TiledIndexSpace(const TiledIndexSpace& t_is, const Range& range):
    TiledIndexSpace{t_is, construct_index_vector(range)} {
    is_dense_subspace_ = (range.step() == 1);
  }

  /**
   * @brief Construct a new sub TiledIndexSpace object from
   * a sub-space of a reference TiledIndexSpace
   *
   * @param [in] t_is reference TiledIndexSpace
   * @param [in] indices set of indices of the reference TiledIndexSpace
   */
  TiledIndexSpace(const TiledIndexSpace& t_is, const IndexVector& indices):
    root_tiled_info_{t_is.root_tiled_info_}, parent_tis_{std::make_shared<TiledIndexSpace>(t_is)} {
    EXPECTS(parent_tis_ != nullptr);
    IndexVector new_indices, new_offsets;

    IndexVector is_indices;
    auto        root_tis = parent_tis_;
    while(root_tis->parent_tis_ != nullptr) { root_tis = root_tis->parent_tis_; }

    new_offsets.push_back(0);
    for(const auto& idx: indices) {
      auto new_idx = t_is.info_translate(idx, (*root_tiled_info_.lock()));
      new_indices.push_back(new_idx);
      new_offsets.push_back(new_offsets.back() + root_tiled_info_.lock()->tile_size(new_idx));

      for(auto i = root_tis->block_begin(new_idx); i != root_tis->block_end(new_idx); i++) {
        is_indices.push_back((*i));
      }
    }

    IndexSpace sub_is{root_tis->index_space(), is_indices};

    tiled_info_ = std::make_shared<TiledIndexSpaceInfo>((*t_is.root_tiled_info_.lock()), sub_is,
                                                        new_offsets, new_indices);
    compute_hash();
  }

  /**
   * @brief Construct a new Tiled Index Space object from a tiled dependent
   *        index space
   *
   * @param [in] t_is parent tiled index space
   * @param [in] dep_map dependency map
   */
  TiledIndexSpace(const TiledIndexSpace& t_is, const TiledIndexSpaceVec& dep_vec,
                  const std::map<IndexVector, TiledIndexSpace>& dep_map):
    tiled_info_{std::make_shared<TiledIndexSpace::TiledIndexSpaceInfo>((*t_is.tiled_info_), dep_vec,
                                                                       dep_map)},
    root_tiled_info_{t_is.root_tiled_info_},
    parent_tis_{std::make_shared<TiledIndexSpace>(t_is)} {
    // validate dependency map
    std::vector<size_t> tis_sizes;
    for(const auto& tis: dep_vec) { tis_sizes.push_back(tis.num_tiles()); }

    for(const auto& [key, value]: dep_map) {
      if(value == TiledIndexSpace{IndexSpace{IndexVector{}}}) continue;
      EXPECTS(key.size() == dep_vec.size());
      EXPECTS(t_is.is_compatible_with(value));
      for(size_t i = 0; i < key.size(); i++) { EXPECTS(key[i] < tis_sizes[i]); }
      EXPECTS(value.num_tiles() == 0 || value.is_subset_of(t_is));
    }

    compute_hash();
  }

  /**
   * @brief Get a TiledIndexLabel for a specific subspace of the
   * TiledIndexSpace
   *
   * @param [in] id string name for the subspace
   * @param [in] lbl an integer value for associated Label
   * @returns a TiledIndexLabel associated with a TiledIndexSpace
   */
  TiledIndexLabel label(std::string id, Label lbl = make_label()) const;

  TiledIndexLabel label(Label lbl = make_label()) const;

  TiledIndexLabel string_label(std::string lbl_str) const;

  /**
   * @brief Construct a tuple of TiledIndexLabel given a count, subspace name
   * and a starting integer Label
   *
   * @tparam c_lbl count of labels
   * @param [in] id name string associated to the subspace
   * @param [in] start starting label value
   * @returns a tuple of TiledIndexLabel
   */
  template<size_t c_lbl>
  auto labels(std::string id = "all", Label start = make_label()) const {
    for(size_t i = 0; i < c_lbl - 1; i++) { auto temp = make_label(); }
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
    if(tiled_info_->tiled_named_subspaces_.find(id) == tiled_info_->tiled_named_subspaces_.end()) {
      std::cerr << "Named sub-space " + id + " doesn't exist!" << std::endl;
    }
    EXPECTS_STR(tiled_info_->tiled_named_subspaces_.find(id) !=
                  tiled_info_->tiled_named_subspaces_.end(),
                "Named sub-space doesn't exist!");

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
    // if(t_dep_map.find(dep_idx_vec) == t_dep_map.end()){
    //     return TiledIndexSpace{IndexSpace{{}}};
    // }

    return t_dep_map.at(dep_idx_vec);
  }

  /**
   * @brief Function for getting TiledIndexSpace from the dependent
   * relation map
   *
   * @param [in] dep_idx_vec set of dependent index values
   * @returns TiledIndexSpace from the relation map and true if one exists. An empty relation and
   * false otherwise
   */
  std::tuple<TiledIndexSpace, bool>
  lookup_dependent_space(const IndexVector& dep_idx_vec = {}) const {
    if(dep_idx_vec.empty()) { return {(*this), true}; }
    const auto& t_dep_map = tiled_info_->tiled_dep_map_;
    if(t_dep_map.find(dep_idx_vec) == t_dep_map.end()) {
      return {TiledIndexSpace{IndexSpace{IndexVector{}}}, false};
    }
    else if(t_dep_map.at(dep_idx_vec) == TiledIndexSpace{IndexSpace{IndexVector{}}}) {
      return {t_dep_map.at(dep_idx_vec), false};
    }

    return {t_dep_map.at(dep_idx_vec), true};
  }

  /**
   * @brief Iterator accessor to the start of the reference IndexSpace
   *
   * @returns a const_iterator to an Index at the first element of the
   * IndexSpace
   */
  IndexIterator begin() const {
    EXPECTS(tiled_info_ != nullptr);
    return tiled_info_->simple_vec_.begin();
  }

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
    EXPECTS(blck_ind <= num_tiles());
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
    EXPECTS(blck_ind <= num_tiles());
    return tiled_info_->is_.begin() + tiled_info_->tile_offsets_[blck_ind + 1];
  }

  /**
   * @brief Boolean method for checking if given TiledIndexSpace is identical
   * to this TiledIndexSpace
   *
   * @param [in] rhs reference TiledIndexSpace
   * @returns true if the tiled_info_ pointer is the same for both
   * TiledIndexSpaces
   */
  bool is_identical(const TiledIndexSpace& rhs) const { return (hash_value_ == rhs.hash()); }

  /**
   * @brief Boolean method for checking if this TiledIndexSpace is a subset of
   * input TiledIndexSpace
   *
   * @param [in] tis reference TiledIndexSpace
   * @returns true if this is a subset of input TiledIndexSpace
   */
  bool is_subset_of(const TiledIndexSpace& tis) const {
    if(this->is_identical(tis)) { return true; }
    else if(this->is_dependent()) {
      if(tis.is_dependent()) { return (this->parent_tis_->is_subset_of((*tis.parent_tis_))); }
      else { return (this->parent_tis_->is_subset_of(tis)); }
    }
    else if(this->parent_tis_ != nullptr) { return (this->parent_tis_->is_subset_of(tis)); }

    return false;
  }

  /**
   * @brief Boolean method for checking if given TiledIndexSpace is compatible
   * to this TiledIndexSpace
   *
   * @param [in] tis reference TiledIndexSpace
   * @returns true if the root_tiled_info_ is the same for both
   * TiledIndexSpaces
   */
  bool is_compatible_with(const TiledIndexSpace& tis) const {
    // return is_subset_of(tis);
    return root_tis() == tis.root_tis();
  }

  /**
   * @brief Boolean method for checking if given TiledIndexSpace
   * is a subspace of this TiledIndexSpace
   *
   * @param [in] rhs reference TiledIndexSpace
   * @returns true if the TiledIndexInfo object of rhs is constructed later
   * then this
   */
  bool is_less_than(const TiledIndexSpace& rhs) const { return (hash_value_ < rhs.hash()); }

  /**
   * @brief Accessor methods to Spin value associated with the input Index
   *
   * @param [in] idx input Index value
   * @returns associated Spin value for the input Index value
   */
  Spin spin(size_t idx) const {
    size_t translated_idx = info_translate(idx, (*root_tiled_info_.lock()));
    return root_tiled_info_.lock()->spin_value(translated_idx);
  }

  /**
   * @brief Accessor methods to Spatial value associated with the input Index
   *
   * @todo: fix once we have spatial
   *
   * @param [in] idx input Index value
   * @returns associated Spatial value for the input Index value
   */
  Spatial spatial(size_t idx) const { return tiled_info_->is_.spatial(idx); }

  /**
   * @brief Accessor method for the set of Ranges associated with a Spin value
   *
   * @param [in] spin input Spin value
   * @returns a vector of Ranges associated with the input Spin value
   */
  std::vector<Range> spin_ranges(Spin spin) const { return tiled_info_->is_.spin_ranges(spin); }

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
   * @return number of tiles in the TiledIndexSpace
   */
  size_t num_tiles() const {
    if(is_dependent()) { NOT_ALLOWED(); }

    return tiled_info_->tile_offsets_.size() - 1;
  }

  /**
   * @brief Get the reference indices for the reference indices to the root
   * TiledIndexSpace
   *
   * @returns vector of indices
   */
  const IndexVector& ref_indices() const { return tiled_info_->ref_indices_; }
  /**
   * @brief Get the maximum number of tiled index blocks in TiledIndexSpace
   *
   * @return maximum number of tiles in the TiledIndexSpace
   */
  size_t max_num_tiles() const { return tiled_info_->max_num_tiles(); }

  /**
   * @brief Get the maximum number of indices in TiledIndexSpace
   *
   * @return maximum number of indices in the TiledIndexSpace
   */
  // size_t max_num_indices() const { return index_space().num_indices(); }
  size_t max_num_indices() const { return tiled_info_->max_num_indices(); }

  /**
   * @brief Get the maximum tile size of TiledIndexSpace
   *
   * @return maximum tile size in the TiledIndexSpace
   */
  size_t max_tile_size() const { return tiled_info_->max_tile_size(); }

  /**
   * @brief Get the tile size for the index blocks
   *
   * @return Tile size
   */
  size_t tile_size(Index i) const { return tiled_info_->tile_size(i); }

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
  const std::vector<Tile>& input_tile_sizes() const { return tiled_info_->input_tile_sizes_; }

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
  const IndexVector& tile_offsets() const { return tiled_info_->tile_offsets_; }

  /**
   * @brief Accessor to tile offset with index id
   *
   * @param [in] id index for the tile offset
   * @returns offset for the corresponding index
   */
  const size_t tile_offset(size_t id) const {
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
  size_t translate(size_t id, const TiledIndexSpace& new_tis) const {
    EXPECTS(!is_dependent());
    EXPECTS(!new_tis.is_dependent());
    EXPECTS(is_compatible_with(new_tis));
    EXPECTS(id >= 0 && id < tiled_info_->ref_indices_.size());
    if(new_tis == (*this)) { return id; }

    auto new_ref_indices = new_tis.tiled_info_->ref_indices_;
    EXPECTS(new_ref_indices.size() == new_tis.tiled_info_->simple_vec_.size());

    auto it =
      std::find(new_ref_indices.begin(), new_ref_indices.end(), tiled_info_->ref_indices_[id]);
    EXPECTS(it != new_ref_indices.end());

    return (it - new_ref_indices.begin());
  }

  std::tuple<size_t, bool> translate_if_possible(size_t id, const TiledIndexSpace& new_tis) const {
    EXPECTS(!is_dependent());
    EXPECTS(!new_tis.is_dependent());
    EXPECTS(is_compatible_with(new_tis));
    EXPECTS(id >= 0 && id < tiled_info_->ref_indices_.size());
    if(new_tis == (*this)) { return {id, true}; }

    auto new_ref_indices = new_tis.tiled_info_->ref_indices_;
    auto it =
      std::find(new_ref_indices.begin(), new_ref_indices.end(), tiled_info_->ref_indices_[id]);
    if(it != new_ref_indices.end()) { return {(it - new_ref_indices.begin()), true}; }
    else { return {-1, false}; }
  }

  /**
   * @brief Translate a tile id from this index space to the new tiled index space.
   *
   * @param [in] id index to be translated
   * @param [in] indep_space_vec indices for the spaces this tiled index space depends on
   * @param [in] target_tis Tiled index space to translate to
   * @param [in] target_indep_space_vec indices for the space the id in the new tiled index space
   * depends on
   *
   * @returns The translated id and true if the given index exists in @p target_tis with independent
   * indices @p target_indep_space_vec
   */
  std::tuple<size_t, bool> translate_if_possible(size_t id, const IndexVector& indep_space_vec,
                                                 const TiledIndexSpace& target_tis,
                                                 const IndexVector& target_indep_space_vec) const {
    EXPECTS(is_compatible_with(target_tis));
    EXPECTS(num_key_tiled_index_spaces() == indep_space_vec.size());
    EXPECTS(target_tis.num_key_tiled_index_spaces() == target_indep_space_vec.size());
    if(target_tis == (*this) && indep_space_vec == target_indep_space_vec) { return {id, true}; }
    TiledIndexSpace from_tis, to_tis;
    bool            from_exists, to_exists;
    std::tie(from_tis, from_exists) = lookup_dependent_space(indep_space_vec);
    std::tie(to_tis, to_exists)     = target_tis.lookup_dependent_space(target_indep_space_vec);
    if(!from_exists || !to_exists) { return {-1, false}; }
    EXPECTS(from_tis.is_compatible_with(to_tis));
    return from_tis.translate_if_possible(id, to_tis);
  }

  /**
   * @brief Check if reference index space is a dependent index space
   *
   * @returns true if the reference index space is a dependent index space
   */
  const bool is_dependent() const {
    // return tiled_info_->is_.is_dependent();
    if(tiled_info_ == nullptr) return false;

    return !tiled_info_->tiled_dep_map_.empty();
  }

  /**
   * @brief Gets the hash value of TiledIndexSpace object
   *
   * @returns a hash value
   */
  size_t hash() const { return hash_value_; }

  /**
   * @brief Gets number of dependent TiledIndexSpace
   *
   * @returns size of the dependent TiledIndexSpace vector
   */
  size_t num_key_tiled_index_spaces() const {
    if(tiled_info_->is_.is_dependent()) { return tiled_info_->is_.num_key_tiled_index_spaces(); }
    return tiled_info_->dep_vec_.size();
  }

  /**
   * @brief
   *
   * @param [in] rhs
   * @returns
   */
  TiledIndexSpace intersect_tis(const TiledIndexSpace& rhs) const {
    EXPECTS(is_dependent() == rhs.is_dependent());

    if(is_dependent()) { return intersect_dep(rhs); }

    if(num_tiles() == 0) { return (*this); }
    else if(rhs.num_tiles() == 0) { return rhs; }

    EXPECTS(root_tiled_info_.lock() == rhs.root_tiled_info_.lock());

    if(is_dependent()) {
      EXPECTS(rhs.is_dependent());

      return intersect_dep(rhs);
    }

    auto lhs_depth = tis_depth();
    auto rhs_depth = rhs.tis_depth();

    auto common_tis = common_ancestor(rhs);

    // if common ancestor is empty return it
    if(common_tis.num_tiles() == 0) { return common_tis; }

    if(common_tis == (*this) || common_tis == rhs) {
      if(lhs_depth > rhs_depth) { return (*this); }
      else { return rhs; }
    }
    const auto& lhs_ref  = this->tiled_info_->ref_indices_;
    const auto& rhs_ref  = rhs.tiled_info_->ref_indices_;
    size_t      tot_size = lhs_ref.size() + rhs_ref.size();
    IndexVector new_indices(tot_size);

    auto it = std::set_intersection(lhs_ref.begin(), lhs_ref.end(), rhs_ref.begin(), rhs_ref.end(),
                                    new_indices.begin());

    new_indices.resize(it - new_indices.begin());

    TiledIndexSpace root = (*this);
    while(root.parent_tis_) { root = *root.parent_tis_; }

    IndexVector translated_indices;
    for(const auto& id: new_indices) {
      translated_indices.push_back(root.translate(id, common_tis));
    }

    return TiledIndexSpace{common_tis, translated_indices};
  }

  TiledIndexSpace compose_tis(const TiledIndexSpace& rhs) const {
    EXPECTS(is_dependent() && rhs.is_dependent());

    const auto& lhs_dep_map = tiled_dep_map();
    const auto& rhs_dep_map = rhs.tiled_dep_map();

    auto it =
      std::find(rhs.tiled_info_->dep_vec_.begin(), rhs.tiled_info_->dep_vec_.end(), (*parent_tis_));
    EXPECTS(it != rhs.tiled_info_->dep_vec_.end());

    size_t dep_id = std::distance(rhs.tiled_info_->dep_vec_.begin(), it);

    std::map<IndexVector, std::set<Index>> idx_vec_map;
    for(const auto& [rhs_key, rhs_val]: rhs_dep_map) {
      auto dep_val = rhs_key[dep_id];

      for(const auto& [lhs_key, lhs_val]: lhs_dep_map) {
        const auto& ref_indices = lhs_val.tiled_info_->ref_indices_;
        auto        ref_it      = std::find(ref_indices.begin(), ref_indices.end(), dep_val);
        if(ref_it != ref_indices.end()) {
          auto        swap_it = rhs_key.begin() + dep_id;
          IndexVector new_key(rhs_key.begin(), swap_it);
          new_key.insert(new_key.end(), lhs_key.begin(), lhs_key.end());
          new_key.insert(new_key.end(), swap_it + 1, rhs_key.end());
          idx_vec_map[new_key].insert(rhs_val.tiled_info_->ref_indices_.begin(),
                                      rhs_val.tiled_info_->ref_indices_.end());
        }
      }
    }

    std::map<IndexVector, TiledIndexSpace> new_dep;

    for(const auto& [key, val]: idx_vec_map) {
      new_dep[key] = TiledIndexSpace{rhs.root_tis(), IndexVector(val.begin(), val.end())};
    }

    TiledIndexSpaceVec new_dep_vec(rhs.tiled_info_->dep_vec_.begin(), it);
    new_dep_vec.insert(new_dep_vec.end(), tiled_info_->dep_vec_.begin(),
                       tiled_info_->dep_vec_.end());
    new_dep_vec.insert(new_dep_vec.end(), it + 1, rhs.tiled_info_->dep_vec_.end());

    return TiledIndexSpace{rhs.root_tis(), new_dep_vec, new_dep};
  }

  TiledIndexSpace invert_tis() const {
    EXPECTS(is_dependent() && num_key_tiled_index_spaces() == 1);

    std::map<IndexVector, TiledIndexSpace> new_dep;
    const auto&                            lhs_dep = tiled_info_->tiled_dep_map_;
    auto                                   ref_tis = tiled_info_->dep_vec_[0];
    std::map<Index, IndexVector>           new_map;

    for(const auto& [key, dep_tis]: lhs_dep) {
      for(const auto& idx: dep_tis.tiled_info_->ref_indices_) { new_map[idx].emplace_back(key[0]); }
    }

    for(const auto& [idx, idx_vec]: new_map) {
      TiledIndexSpace val_tis{ref_tis, idx_vec};
      new_dep[IndexVector{idx}] = val_tis;
    }

    return TiledIndexSpace{ref_tis, {(*parent_tis_)}, new_dep};
  }

  TiledIndexSpace project_tis(const TiledIndexSpace& rhs) const {
    EXPECTS(is_dependent());
    EXPECTS(!rhs.is_dependent());

    auto tis_it = std::find(tiled_info_->dep_vec_.begin(), tiled_info_->dep_vec_.end(), rhs);
    EXPECTS(tis_it != tiled_info_->dep_vec_.end());

    size_t p_idx = std::distance(tiled_info_->dep_vec_.begin(), tis_it);

    const auto& lhs_dep_map = tiled_dep_map();

    TiledIndexSpace res;

    if(num_key_tiled_index_spaces() > 1) {
      TiledIndexSpaceVec new_dep_vec(tiled_info_->dep_vec_);
      new_dep_vec.erase(new_dep_vec.begin() + p_idx);

      std::map<IndexVector, std::set<Index>> idx_vec_map;
      for(const auto& [lhs_key, lhs_val]: lhs_dep_map) {
        IndexVector new_vec(lhs_key);
        new_vec.erase(new_vec.begin() + p_idx);

        idx_vec_map[new_vec].insert(lhs_val.tiled_info_->ref_indices_.begin(),
                                    lhs_val.tiled_info_->ref_indices_.end());
      }

      std::map<IndexVector, TiledIndexSpace> new_dep;
      for(const auto& [key, val]: idx_vec_map) {
        new_dep[key] = TiledIndexSpace{root_tis(), IndexVector(val.begin(), val.end())};
      }
      res = TiledIndexSpace{root_tis(), new_dep_vec, new_dep};
    }
    else {
      std::set<Index> all_idx;
      for(const auto& [lhs_key, lhs_val]: lhs_dep_map) {
        all_idx.insert(lhs_val.tiled_info_->ref_indices_.begin(),
                       lhs_val.tiled_info_->ref_indices_.end());
      }

      res = TiledIndexSpace{root_tis(), IndexVector(all_idx.begin(), all_idx.end())};
    }

    return res;
  }

  TiledIndexSpace union_tis(const TiledIndexSpace& rhs) const {
    EXPECTS(root_tiled_info_.lock() == rhs.root_tiled_info_.lock());

    TiledIndexSpace res;
    if(is_dependent() && rhs.is_dependent()) {
      EXPECTS(tiled_info_->dep_vec_ == rhs.tiled_info_->dep_vec_);
      const auto& lhs_dep_map = tiled_dep_map();
      const auto& rhs_dep_map = rhs.tiled_dep_map();

      std::map<IndexVector, std::set<Index>> idx_vec_map;
      for(const auto& [rhs_key, rhs_val]: rhs_dep_map) {
        idx_vec_map[rhs_key].insert(rhs_val.tiled_info_->ref_indices_.begin(),
                                    rhs_val.tiled_info_->ref_indices_.end());
      }

      for(const auto& [lhs_key, lhs_val]: lhs_dep_map) {
        idx_vec_map[lhs_key].insert(lhs_val.tiled_info_->ref_indices_.begin(),
                                    lhs_val.tiled_info_->ref_indices_.end());
      }

      std::map<IndexVector, TiledIndexSpace> new_dep;
      for(const auto& [key, val]: idx_vec_map) {
        new_dep[key] = TiledIndexSpace{root_tis(), IndexVector(val.begin(), val.end())};
      }

      res = TiledIndexSpace{root_tis(), tiled_info_->dep_vec_, new_dep};
    }
    else {
      std::set<Index> new_refs;
      auto            lhs_ref = ref_indices();
      auto            rhs_ref = rhs.ref_indices();
      new_refs.insert(lhs_ref.begin(), lhs_ref.end());
      new_refs.insert(rhs_ref.begin(), rhs_ref.end());

      res = TiledIndexSpace{root_tis(), IndexVector(new_refs.begin(), new_refs.end())};
    }

    return res;
  }

  /**
   * @brief Gets the depth of TiledIndexSpace hierarchy
   *
   * @returns size of the depth
   */
  size_t tis_depth() const {
    size_t res = 0;
    auto   tmp = parent_tis_;
    while(tmp) {
      tmp = tmp->parent_tis_;
      res++;
    }
    return res;
  }

  /**
   * @brief Finds the common ancestor TiledIndexSpace for input and current
   * TiledIndexSpace object
   *
   * @param [in] tis input TiledIndexSpace object
   * @returns closest common ancestor TiledIndexSpace
   */
  TiledIndexSpace common_ancestor(const TiledIndexSpace& tis) const {
    auto lhs_depth = tis_depth();
    auto rhs_depth = tis.tis_depth();

    if(lhs_depth == rhs_depth) {
      if((*this) != tis) {
        if(lhs_depth == 0 && rhs_depth == 0) { return TiledIndexSpace{IndexSpace{IndexVector{}}}; }
        else { return parent_tis_->common_ancestor((*tis.parent_tis_)); }
      }
      else { return (*this); }
    }

    if(lhs_depth > rhs_depth) { return parent_tis_->common_ancestor(tis); }
    else { return common_ancestor((*tis.parent_tis_)); }
  }

  bool empty() const { return (tiled_info_->simple_vec_.size() == 1); }

  TiledIndexSpace root_tis() const {
    if(!parent_tis_) { return (*this); }

    auto tmp = parent_tis_;
    while(tmp->parent_tis_) { tmp = tmp->parent_tis_; }

    return (*tmp);
  }

  TiledIndexSpace parent_tis() const { return (*parent_tis_.get()); }

  bool is_dense_subspace() const { return is_dense_subspace_; }

  /**
   * @brief Equality comparison operator
   *
   * @param [in] lhs Left-hand side
   * @param [in] rhs Right-hand side
   *
   * @return true if lhs == rhs
   */
  friend bool operator==(const TiledIndexSpace& lhs, const TiledIndexSpace& rhs);

  /**
   * @brief Ordering comparison operator
   *
   * @param [in] lhs Left-hand side
   * @param [in] rhs Right-hand side
   *
   * @return true if lhs < rhs
   */
  friend bool operator<(const TiledIndexSpace& lhs, const TiledIndexSpace& rhs);

  /**
   * @brief Inequality comparison operator
   *
   * @param [in] lhs Left-hand side
   * @param [in] rhs Right-hand side
   *
   * @return true if lhs != rhs
   */
  friend bool operator!=(const TiledIndexSpace& lhs, const TiledIndexSpace& rhs);

  /**
   * @brief Ordering comparison operator
   *
   * @param [in] lhs Left-hand side
   * @param [in] rhs Right-hand side
   *
   * @return true if lhs > rhs
   */
  friend bool operator>(const TiledIndexSpace& lhs, const TiledIndexSpace& rhs);

  /**
   * @brief Ordering comparison operator
   *
   * @param [in] lhs Left-hand side
   * @param [in] rhs Right-hand side
   *
   * @return true if lhs <= rhs
   */
  friend bool operator<=(const TiledIndexSpace& lhs, const TiledIndexSpace& rhs);

  /**
   * @brief Ordering comparison operator
   *
   * @param [in] lhs Left-hand side
   * @param [in] rhs Right-hand side
   *
   * @return true if lhs >= rhs
   */
  friend bool operator>=(const TiledIndexSpace& lhs, const TiledIndexSpace& rhs);

protected:
  /**
   * @brief Internal struct for representing a TiledIndexSpace details. Mainly
   * used for comparing TiledIndexSpace between eachother for compatibility
   * checks. This also behaves as PIMPL to ease the copy of TiledIndexSpaces.
   *
   */
  struct TiledIndexSpaceInfo {
    /* data */
    IndexSpace        is_;               /**< The index space being tiled*/
    Tile              input_tile_size_;  /**< User-specified tile size*/
    std::vector<Tile> input_tile_sizes_; /**< User-specified multiple tile sizes*/
    std::map<IndexVector, std::vector<Tile>> dep_tile_sizes_;
    IndexVector                              tile_offsets_; /**< Tile offsets */
    IndexVector                              ref_indices_;  /**< Reference indices to root */
    IndexVector                              simple_vec_;   /**< vector where at(i) = i*/
    size_t             max_num_tiles_;   /**< Maximum number of tiles in this tiled space */
    size_t             max_tile_size_;   /**< Maximum tile size */
    size_t             max_num_indices_; /**< Maximum number of indices */
    TiledIndexSpaceVec dep_vec_;         /**< vector of TiledIndexSpaces that are
                                            key for the dependency map */
    std::map<IndexVector, TiledIndexSpace>
      tiled_dep_map_; /**< Tiled dependency map for dependent index spaces*/
    std::map<std::string, TiledIndexSpace>
      tiled_named_subspaces_; /**< Tiled named subspaces map string ids*/

    /**
     * @brief Construct a new TiledIndexSpaceInfo object from an IndexSpace
     * and input tile size(s). The size can be a single size or a set of
     * tile sizes. Note that, set of tiles sizes are required to tile the
     * underlying IndexSpace completely.
     *
     * @param [in] is reference IndexSpace to be tiled
     * @param [in] input_tile_size input single tile size
     * @param [in] input_tile_sizes input set of tile sizes
     */
    TiledIndexSpaceInfo(IndexSpace is, Tile input_tile_size,
                        const std::vector<Tile>& input_tile_sizes):
      is_{is},
      input_tile_size_{input_tile_size},
      input_tile_sizes_{input_tile_sizes},
      max_num_indices_{is.max_num_indices()} {
      if(input_tile_sizes.size() > 0) {
        auto max_item = std::max_element(input_tile_sizes.begin(), input_tile_sizes.end());

        max_tile_size_ = *max_item;
        // construct indices with set of tile sizes
        tile_offsets_ = construct_tiled_indices(is, input_tile_sizes);
        // construct dependency according to tile sizes
        for(const auto& kv: is.map_tiled_index_spaces()) {
          tiled_dep_map_.insert(std::pair<IndexVector, TiledIndexSpace>{
            kv.first, TiledIndexSpace{kv.second, input_tile_sizes}});
        }
        // in case of multiple tile sizes no named spacing carried.
      }
      else {
        // set maximum tile size to single input tile size
        max_tile_size_ = input_tile_size;
        // construct indices with input tile size
        tile_offsets_ = construct_tiled_indices(is, input_tile_size);
        // construct dependency according to tile size
        for(const auto& kv: is.map_tiled_index_spaces()) {
          tiled_dep_map_.insert(std::pair<IndexVector, TiledIndexSpace>{
            kv.first, TiledIndexSpace{kv.second, input_tile_size}});
        }
      }

      if(!is.is_dependent()) {
        for(Index i = 0; i < tile_offsets_.size() - 1; i++) {
          simple_vec_.push_back(i);
          ref_indices_.push_back(i);
        }
      }
      compute_max_num_tiles();
      validate();
    }

    /**
     * @brief Construct a new TiledIndexSpaceInfo object for dependent
     * IndexSpaces with custom tile sizes for each dependency
     *
     * @param [in] is input dependent IndexSpace
     * @param [in] dep_tile_sizes map from set of indices to vector of tile
     * sizes for custom tiling
     */
    TiledIndexSpaceInfo(IndexSpace is, std::map<IndexVector, std::vector<Tile>> dep_tile_sizes):
      is_{is},
      input_tile_size_{0},
      input_tile_sizes_{{}},
      dep_tile_sizes_{dep_tile_sizes},
      max_num_indices_{is.max_num_indices()} {
      EXPECTS(is.is_dependent());
      // construct dependency according to tile size
      for(const auto& kv: is.map_tiled_index_spaces()) {
        EXPECTS(dep_tile_sizes.find(kv.first) != dep_tile_sizes.end());

        tiled_dep_map_.insert(std::pair<IndexVector, TiledIndexSpace>{
          kv.first, TiledIndexSpace{kv.second, dep_tile_sizes[kv.first]}});
      }

      max_tile_size_ = 0;
      for(const auto& [idx, tile_sizes]: dep_tile_sizes) {
        auto it = std::max_element(tile_sizes.begin(), tile_sizes.end());
        if(max_tile_size_ < *it) { max_tile_size_ = *it; }
      }

      compute_max_num_tiles();
      validate();
    }

    /**
     * @brief Construct a new sub-TiledIndexSpaceInfo object from a root
     * TiledIndexInfo object along with a set of offsets and indices
     * corresponding to the root object
     *
     * @param [in] root TiledIndexSpaceInfo object
     * @param [in] offsets input offsets from the root
     * @param [in] indices input indices from the root
     */
    TiledIndexSpaceInfo(const TiledIndexSpaceInfo& root, const IndexVector& offsets,
                        const IndexVector& indices):
      is_{root.is_},
      input_tile_size_{root.input_tile_size_},
      input_tile_sizes_{root.input_tile_sizes_},
      tile_offsets_{offsets},
      ref_indices_{indices} {
      for(Index i = 0; i < tile_offsets_.size() - 1; i++) { simple_vec_.push_back(i); }

      max_num_indices_ = 0;
      if(input_tile_sizes_.size() > 0) {
        max_tile_size_ = 0;
        for(const auto& idx: indices) {
          max_num_indices_ += input_tile_sizes_[idx];
          if(max_tile_size_ < input_tile_sizes_[idx]) { max_tile_size_ = input_tile_sizes_[idx]; }
        }
      }
      else {
        max_tile_size_   = root.max_tile_size_;
        max_num_indices_ = indices.size() * max_tile_size_;
      }

      compute_max_num_tiles();
      validate();
    }

    /**
     * @brief Construct a new sub-TiledIndexSpaceInfo object from a root
     * TiledIndexInfo object along with a root IndexSpace, a set of offsets
     * and indices corresponding to the root object
     *
     * @param [in] root TiledIndexSpaceInfo object
     * @param [in] offsets input offsets from the root
     * @param [in] indices input indices from the root
     */
    TiledIndexSpaceInfo(const TiledIndexSpaceInfo& root, const IndexSpace& is,
                        const IndexVector& offsets, const IndexVector& indices):
      is_{is},
      input_tile_size_{root.input_tile_size_},
      input_tile_sizes_{root.input_tile_sizes_},
      tile_offsets_{offsets},
      ref_indices_{indices} {
      for(Index i = 0; i < tile_offsets_.size() - 1; i++) { simple_vec_.push_back(i); }

      max_num_indices_ = 0;
      if(input_tile_sizes_.size() > 0) {
        max_tile_size_ = 0;
        for(const auto& idx: indices) {
          max_num_indices_ += input_tile_sizes_[idx];
          if(max_tile_size_ < input_tile_sizes_[idx]) { max_tile_size_ = input_tile_sizes_[idx]; }
        }
      }
      else {
        max_tile_size_   = root.max_tile_size_;
        max_num_indices_ = indices.size() * max_tile_size_;
      }

      compute_max_num_tiles();
      validate();
    }

    /**
     * @brief Construct a new TiledIndexSpaceInfo object from a root
     * (dependent) TiledIndexSpaceInfo object with a new set of relations
     * for the dependency relation.
     *
     * @param [in] root TiledIndexSpaceInfo object
     * @param [in] dep_map dependency relation between indices of the
     * dependent TiledIndexSpace and corresponding TiledIndexSpaces
     */
    TiledIndexSpaceInfo(const TiledIndexSpaceInfo& root, const TiledIndexSpaceVec& dep_vec,
                        const std::map<IndexVector, TiledIndexSpace>& dep_map):
      is_{root.is_},
      input_tile_size_{root.input_tile_size_},
      input_tile_sizes_{root.input_tile_sizes_},
      dep_vec_{dep_vec},
      tiled_dep_map_{dep_map} {
      // Check if the new dependency relation is sub set of root
      // dependency relation
      // const auto& root_dep = root.tiled_dep_map_;
      // for(const auto& dep_kv : dep_map) {
      //     const auto& key     = dep_kv.first;
      //     const auto& dep_tis = dep_kv.second;
      //     EXPECTS(root_dep.find(key) != root_dep.end());
      //     EXPECTS(dep_tis.is_compatible_with(root_dep.at(key)));
      // }
      std::vector<IndexVector> result;
      IndexVector              acc;
      combinations_tis(result, acc, dep_vec, 0);
      TiledIndexSpace empty_tis{IndexSpace{IndexVector{}}};

      for(const auto& iv: result) {
        if(tiled_dep_map_.find(iv) == tiled_dep_map_.end()) { tiled_dep_map_[iv] = empty_tis; }
      }

      max_tile_size_   = 0;
      max_num_indices_ = 0;
      for(const auto& [idx, tis]: dep_map) {
        if(max_tile_size_ < tis.max_tile_size()) {
          max_tile_size_   = tis.max_tile_size();
          max_num_indices_ = tis.max_num_indices();
        }
      }

      compute_max_num_tiles();
      validate();
    }

    /**
     * @brief Construct starting and ending indices of each tile with
     * respect to input tile size.
     *
     * @param [in] is reference IndexSpace
     * @param [in] size Tile size value
     * @returns a vector of indices corresponding to the start and end of
     * each tile
     */
    IndexVector construct_tiled_indices(const IndexSpace& is, Tile tile_size) {
      if(is.is_dependent()) { return {}; }

      if(is.num_indices() == 0) { return {0}; }

      IndexVector boundries, ret;
      // Get lo and hi for each named subspace ranges
      for(const auto& kv: is.get_named_ranges()) {
        for(const auto& range: kv.second) {
          boundries.push_back(range.lo());
          boundries.push_back(range.hi());
        }
      }

      // Get SpinAttribute boundries
      if(is.has_spin()) {
        auto spin_map = is.get_spin().get_map();
        for(const auto& kv: spin_map) {
          for(const auto& range: kv.second) {
            boundries.push_back(range.lo());
            boundries.push_back(range.hi());
          }
        }
      }
      // Get SpinAttribute boundries
      if(is.has_spatial()) {
        auto spatial_map = is.get_spatial().get_map();
        for(const auto& kv: spatial_map) {
          for(const auto& range: kv.second) {
            boundries.push_back(range.lo());
            boundries.push_back(range.hi());
          }
        }
      }

      // If no boundry clean split with respect to tile size
      if(boundries.empty()) {
        // add starting indices
        for(size_t i = 0; i < is.num_indices(); i += tile_size) { ret.push_back(i); }
        // add size of IndexSpace for the last block
        ret.push_back(is.num_indices());
      }
      else { // Remove duplicates
        std::sort(boundries.begin(), boundries.end());
        auto last = std::unique(boundries.begin(), boundries.end());
        boundries.erase(last, boundries.end());
        // Construct start indices for blocks according to boundries.
        size_t i = 0;
        size_t j = (i == boundries[0]) ? 1 : 0;

        while(i < is.num_indices()) {
          ret.push_back(i);
          i = (i + tile_size >= boundries[j]) ? boundries[j++] : (i + tile_size);
        }
        // add size of IndexSpace for the last block
        ret.push_back(is.num_indices());
      }

      return ret;
    }

    /**
     * @brief Construct starting and ending indices of each tile with
     * respect to input tile sizes
     *
     * @param [in] is reference IndexSpace
     * @param [in] sizes set of input Tile sizes
     * @returns a vector of indices corresponding to the start and end of
     * each tile
     */
    IndexVector construct_tiled_indices(const IndexSpace& is, const std::vector<Tile>& tiles) {
      if(is.is_dependent()) { return {}; }

      if(is.num_indices() == 0) { return {0}; }
      // Check if sizes match
      EXPECTS(is.num_indices() == [&tiles]() {
        size_t ret = 0;
        for(const auto& var: tiles) { ret += var; }
        return ret;
      }());

      IndexVector ret, boundries;

      if(is.has_spin()) {
        auto spin_map = is.get_spin().get_map();
        for(const auto& kv: spin_map) {
          for(const auto& range: kv.second) {
            boundries.push_back(range.lo());
            boundries.push_back(range.hi());
          }
        }
      }

      if(is.has_spatial()) {
        auto spatial_map = is.get_spatial().get_map();
        for(const auto& kv: spatial_map) {
          for(const auto& range: kv.second) {
            boundries.push_back(range.lo());
            boundries.push_back(range.hi());
          }
        }
      }

      for(const auto& kv: is.get_named_ranges()) {
        for(const auto& range: kv.second) {
          boundries.push_back(range.lo());
          boundries.push_back(range.hi());
        }
      }

      // add starting indices
      size_t j = 0;
      for(size_t i = 0; i < is.num_indices(); i += tiles[j++]) { ret.push_back(i); }
      // add size of IndexSpace for the last block
      ret.push_back(is.num_indices());

      if(!(boundries.empty())) {
        std::sort(boundries.begin(), boundries.end());
        auto last = std::unique(boundries.begin(), boundries.end());
        boundries.erase(last, boundries.end());
        // check if there is any mismatch between boudries and generated
        // start indices
        for(auto& bound: boundries) { EXPECTS(std::binary_search(ret.begin(), ret.end(), bound)); }
      }

      return ret;
    }

    /**
     * @brief Accessor for getting the size of a specific tile in the
     * TiledIndexSpaceInfo object
     *
     * @param [in] idx input index
     * @returns the size of the tile at the corresponding index
     */
    size_t tile_size(Index idx) const {
      if(!(idx >= 0 && idx < tile_offsets_.size() - 1)) std::cerr << "idx: " << idx << std::endl;
      EXPECTS(idx >= 0 && idx < tile_offsets_.size() - 1);
      return tile_offsets_[idx + 1] - tile_offsets_[idx];
    }

    /**
     * @brief Gets the maximum number of tiles in the TiledIndexSpaceInfo
     * object. In case of independent TiledIndexSpace it returns the number
     * of tiles, otherwise returns the maximum size of the TiledIndexSpaces
     * in the dependency relation
     *
     * @returns the maximum number of tiles in the TiledIndexSpaceInfo
     */
    size_t max_num_tiles() const { return max_num_tiles_; }

    /**
     * @brief Gets the maximum number of indices in the TiledIndexSpaceInfo
     * object.
     *
     * @returns the maximum number of indices in the TiledIndexSpaceInfo
     */
    size_t max_num_indices() const { return max_num_indices_; }

    /**
     * @brief Gets the maximum tile size in TiledIndexSpaceInfo object. In
     * case of independent TiledIndexSpace returns the single tile size, or
     * the maximum of the tile size vector. If it is dependent TiledIndexSpace
     * it will return the maximum tile size from the dependency relation
     *
     * @returns the maximum tile size in the TiledIndexSpaceInfo
     */
    size_t max_tile_size() const { return max_tile_size_; }

    /**
     * @brief Helper method for computing the maximum number of tiles
     * (mainly used for dependent TiledIndexSpace)
     *
     */
    void compute_max_num_tiles() {
      if(tiled_dep_map_.empty()) { max_num_tiles_ = tile_offsets_.size() - 1; }
      else {
        max_num_tiles_ = 0;
        for(const auto& kv: tiled_dep_map_) {
          if(max_num_tiles_ < kv.second.max_num_tiles()) {
            max_num_tiles_ = kv.second.max_num_tiles();
          }
        }
      }
    }

    /**
     * @brief Gets the spin value for an index in an IndexSpace object
     *
     * @param [in] id index value to be accessed
     * @returns corresponding Spin value
     */
    Spin spin_value(size_t id) const { return is_.spin(tile_offsets_[id]); }

    /**
     * @brief Validates the construction of TiledIndexSpaceInfo object
     *
     */
    void validate() {
      // Post-condition
      EXPECTS(simple_vec_.size() == ref_indices_.size());
      if(tiled_dep_map_.empty()) { EXPECTS(simple_vec_.size() + 1 == tile_offsets_.size()); }
    }

    void combinations_tis(std::vector<IndexVector>& res, const IndexVector& accum,
                          const TiledIndexSpaceVec& tis_vec, size_t i) {
      if(i == tis_vec.size()) { res.push_back(accum); }
      else {
        auto tis = tis_vec[i];

        for(const auto& tile_id: tis) {
          IndexVector tmp{accum};
          tmp.push_back(tile_id);
          combinations_tis(res, tmp, tis_vec, i + 1);
        }
      }
    }

    /**
     * @brief Computes the hash value for TiledIndexSpaceInfo object
     *
     * @returns a hash value
     */
    size_t hash() {
      // get hash of IndexSpace as seed
      size_t result = is_.hash();

      // combine hash with tile size(s)
      internal::hash_combine(result, input_tile_size_);
      // if there are mutliple tiles
      if(input_tile_sizes_.size() > 0) {
        // combine hash with number of tile sizes
        internal::hash_combine(result, input_tile_sizes_.size());
        // combine hash with each tile size
        for(const auto& tile: input_tile_sizes_) { internal::hash_combine(result, tile); }
      }

      // if it is a dependent TiledIndexSpace
      if(!tiled_dep_map_.empty()) {
        for(const auto& iv_is: tiled_dep_map_) {
          const auto& iv = iv_is.first;
          const auto& is = iv_is.second;
          // combine hash with size of index vector
          internal::hash_combine(result, iv.size());
          // combine hash with each key element
          for(const auto& idx: iv) { internal::hash_combine(result, idx); }
          // combine hash with dependent index space hash
          internal::hash_combine(result, is.hash());
        }
      }

      /// @to-do add dep_tile_sizes stuff

      return result;
    }
  }; // struct TiledIndexSpaceInfo

  std::shared_ptr<TiledIndexSpaceInfo>
    tiled_info_; /**< Shared pointer to the TiledIndexSpaceInfo object*/
  std::weak_ptr<TiledIndexSpaceInfo> root_tiled_info_; /**< Weak pointer to the root
                                                          TiledIndexSpaceInfo object*/
  std::shared_ptr<TiledIndexSpace> parent_tis_;        /**< Shared pointer to the parent
                                                        TiledIndexSpace object*/
  size_t hash_value_;
  bool   is_dense_subspace_ = false;

  /**
   * @brief Return the corresponding tile position of an index for a give
   * TiledIndexSpaceInfo object
   *
   * @param [in] id index position to be found on input TiledIndexSpaceInfo
   * object
   * @param [in] new_info reference input TiledIndexSpaceInfo object
   * @returns The tile index of the corresponding from the reference
   * TiledIndexSpaceInfo object
   */
  size_t info_translate(size_t id, const TiledIndexSpaceInfo& new_info) const {
    EXPECTS(id >= 0 && id < tiled_info_->ref_indices_.size());
    EXPECTS(new_info.ref_indices_.size() == new_info.simple_vec_.size());

    auto it = std::find(new_info.ref_indices_.begin(), new_info.ref_indices_.end(),
                        tiled_info_->ref_indices_[id]);
    EXPECTS(it != new_info.ref_indices_.end());

    return (it - new_info.ref_indices_.begin());
  }

  /**
   * @brief Set the root TiledIndexSpaceInfo object
   *
   * @param [in] root a shared pointer to a TiledIndexSpaceInfo object
   */
  void set_root(const std::shared_ptr<TiledIndexSpaceInfo>& root) { root_tiled_info_ = root; }

  /**
   * @brief Set the parent TiledIndexSpace object
   *
   * @param [in] parent a shared pointer to a TiledIndexSpace object
   */
  void set_parent(const std::shared_ptr<TiledIndexSpace>& parent) { parent_tis_ = parent; }

  /**
   * @brief Set the shared pointer to TiledIndexSpaceInfo object
   *
   * @param [in] tiled_info shared pointer to TiledIndexSpaceInfo object
   */
  void set_tiled_info(const std::shared_ptr<TiledIndexSpaceInfo>& tiled_info) {
    tiled_info_ = tiled_info;
  }

  /**
   * @brief Sets hash value of TiledIndexSpace object using TiledIndexInfo
   *
   */
  void compute_hash() { hash_value_ = tiled_info_->hash(); }

  /**
   * @brief Method for tiling all the named subspaces in an IndexSpace
   *
   * @param [in] is input IndexSpace
   */
  void tile_named_subspaces(const IndexSpace& is) {
    // construct tiled spaces for named subspaces
    for(const auto& str_subis: is.map_named_sub_index_spaces()) {
      auto named_is = str_subis.second;

      IndexVector indices;

      for(const auto& idx: named_is) {
        // find position in the root index space
        size_t pos = is.find_pos(idx);
        // named subspace should always find it
        EXPECTS(pos >= 0);

        size_t      tile_idx = 0;
        const auto& offsets  = tiled_info_->tile_offsets_;
        // find in which tiles in the root it would be
        for(Index i = 0; pos >= offsets[i]; i++) { tile_idx = i; }
        indices.push_back(tile_idx);
      }
      // remove duplicates from the indices
      std::sort(indices.begin(), indices.end());
      auto last = std::unique(indices.begin(), indices.end());
      indices.erase(last, indices.end());

      IndexVector new_offsets;

      new_offsets.push_back(0);
      for(const auto& idx: indices) {
        new_offsets.push_back(new_offsets.back() + root_tiled_info_.lock()->tile_size(idx));
      }
      TiledIndexSpace tempTIS{};
      tempTIS.set_tiled_info(std::make_shared<TiledIndexSpaceInfo>((*root_tiled_info_.lock()),
                                                                   named_is, new_offsets, indices));
      tempTIS.set_root(tiled_info_);
      tempTIS.set_parent(std::make_shared<TiledIndexSpace>(*this));
      tempTIS.compute_hash();

      tiled_info_->tiled_named_subspaces_.insert({str_subis.first, tempTIS});
    }
  }

  /**
   * @brief Helper method for intersecting dependent TiledIndexSpaces
   *
   * @param [in] rhs input dependent TiledIndexSpace
   * @returns an intersection TiledIndexSpace of the input and this
   * TiledIndexSpace
   */
  TiledIndexSpace intersect_dep(const TiledIndexSpace& rhs) const {
    EXPECTS(is_dependent() == rhs.is_dependent());
    EXPECTS(tiled_info_->dep_vec_ == rhs.tiled_info_->dep_vec_);

    auto common_tis = parent_tis_->common_ancestor(rhs.root_tis());

    std::map<IndexVector, TiledIndexSpace> new_dep;
    const auto&                            lhs_dep = tiled_info_->tiled_dep_map_;
    const auto&                            rhs_dep = rhs.tiled_info_->tiled_dep_map_;

    for(const auto& [key, dep_tis]: lhs_dep) {
      new_dep.insert(new_dep.begin(), {key, dep_tis.intersect_tis(rhs_dep.at(key))});
    }

    return TiledIndexSpace{common_tis, tiled_info_->dep_vec_, new_dep};
  }

  template<size_t... Is>
  auto labels_impl(std::string id, Label start, std::index_sequence<Is...>) const;

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

class TileLabelElement: public Symbol {
public:
  TileLabelElement()                                   = default;
  TileLabelElement(const TileLabelElement&)            = default;
  TileLabelElement(TileLabelElement&&)                 = default;
  ~TileLabelElement()                                  = default;
  TileLabelElement& operator=(const TileLabelElement&) = default;
  TileLabelElement& operator=(TileLabelElement&&)      = default;

  TileLabelElement(const TiledIndexSpace& tis, Label label = 0, std::string label_str = ""):
    tis_{tis}, label_{label}, label_str_{label_str} {}

  const TiledIndexSpace& tiled_index_space() const { return tis_; }

  Label label() const { return label_; }

  std::string label_str() const { return label_str_; }

  bool is_compatible_with(const TiledIndexSpace& tis) const { return tis_.is_compatible_with(tis); }

private:
  TiledIndexSpace tis_;
  Label           label_;
  std::string     label_str_;
}; // class TileLabelElement

// Comparison operator implementations
inline bool operator==(const TileLabelElement& lhs, const TileLabelElement& rhs) {
  bool has_same_label = false;
  if(lhs.label_str() != "" && rhs.label_str() != "") {
    has_same_label = lhs.label_str() == rhs.label_str();
  }
  else { has_same_label = lhs.label() == rhs.label(); }
  return lhs.tiled_index_space() == rhs.tiled_index_space() && has_same_label;
}

inline bool operator<(const TileLabelElement& lhs, const TileLabelElement& rhs) {
  bool is_less_then = false;
  if(lhs.label_str() != "" && rhs.label_str() != "") {
    is_less_then = lhs.label_str() < rhs.label_str();
  }
  else { is_less_then = lhs.label() < rhs.label(); }

  return lhs.tiled_index_space() < rhs.tiled_index_space() ||
         (lhs.tiled_index_space() == rhs.tiled_index_space() && is_less_then);
}

inline bool operator!=(const TileLabelElement& lhs, const TileLabelElement& rhs) {
  return !(lhs == rhs);
}

inline bool operator>(const TileLabelElement& lhs, const TileLabelElement& rhs) {
  return (rhs < lhs);
}

inline bool operator<=(const TileLabelElement& lhs, const TileLabelElement& rhs) {
  return (lhs < rhs) || (lhs == rhs);
}

inline bool operator>=(const TileLabelElement& lhs, const TileLabelElement& rhs) {
  return (rhs <= lhs);
}

/**
 * @brief Index label to index into tensors. The labels used by the user need to
 * be positive.
 *
 */
class TiledIndexLabel: public SymbolInterface {
public:
  // Constructor
  TiledIndexLabel()                       = default;
  TiledIndexLabel(const TiledIndexLabel&) = default;
  TiledIndexLabel(TiledIndexLabel&&)      = default;
  ~TiledIndexLabel()                      = default;

  TiledIndexLabel& operator=(const TiledIndexLabel&) = default;
  TiledIndexLabel& operator=(TiledIndexLabel&&)      = default;

  /**
   * @brief Construct a new TiledIndexLabel object from a reference
   * TileLabelElement object and a label
   *
   * @param [in] t_is reference TileLabelElement object
   * @param [in] lbl input label (default: 0, negative values used internally)
   * @param [in] dep_labels set of dependent TiledIndexLabels (default: empty
   * set)
   */
  TiledIndexLabel(const TiledIndexSpace& tis, Label lbl = 0,
                  const std::vector<TileLabelElement>& secondary_labels = {}):
    TiledIndexLabel{TileLabelElement{tis, lbl}, secondary_labels} {
    // no-op
  }

  TiledIndexLabel(const TiledIndexSpace& tis, Label lbl,
                  const std::vector<TiledIndexLabel>& secondary_labels):
    TiledIndexLabel{TileLabelElement{tis, lbl}, secondary_labels} {
    // no-op
  }

  TiledIndexLabel(const TiledIndexSpace& tis, const std::string& lbl_str):
    TiledIndexLabel{TileLabelElement{tis, -1, lbl_str}, std::vector<TileLabelElement>{}} {
    // no-op
  }

  TiledIndexLabel(const TiledIndexSpace& tis, const std::vector<TiledIndexLabel>& secondary_labels):
    TiledIndexLabel{TileLabelElement{tis, make_label()}, secondary_labels} {
    // no-op
  }

  TiledIndexLabel(const TileLabelElement&              primary_label,
                  const std::vector<TileLabelElement>& secondary_labels = {}):
    primary_label_{primary_label}, secondary_labels_{secondary_labels} {
    validate();
  }

  TiledIndexLabel(const TileLabelElement&             primary_label,
                  const std::vector<TiledIndexLabel>& secondary_labels = {}):
    primary_label_{primary_label} {
    for(const auto& lbl: secondary_labels) { secondary_labels_.push_back(lbl.primary_label()); }
    validate();
  }

  /**
   * @brief Construct a new TiledIndexLabel object from another one with input
   * dependent labels
   *
   * @param [in] t_il reference TiledIndexLabel object
   * @param [in] dep_labels set of dependent TiledIndexLabels
   */
  TiledIndexLabel(const TiledIndexLabel&               til,
                  const std::vector<TileLabelElement>& secondary_labels):
    primary_label_{til.primary_label()}, secondary_labels_{secondary_labels} {
    validate();
  }

  TiledIndexLabel(const TiledIndexLabel& til, const std::vector<TiledIndexLabel>& secondary_labels):
    primary_label_{til.primary_label()} {
    for(const auto& lbl: secondary_labels) { secondary_labels_.push_back(lbl.primary_label()); }
    validate();
  }

  void* get_symbol_ptr() const { return primary_label_.get_symbol_ptr(); }

  /**
   * @brief Operator overload for () to construct dependent TiledIndexLabel
   * objects from the input TiledIndexLabels
   *
   * @returns a new TiledIndexLabel from this.
   */
  const TiledIndexLabel& operator()() const { return (*this); }

  /**
   * @brief Operator overload for () to construct dependent TiledIndexLabel
   * objects from the input TiledIndexLabels
   *
   * @tparam Args variadic template for multiple TiledIndexLabel object
   * @param [in] il1 input TiledIndexLabel object
   * @param [in] rest variadic template for rest of the arguments
   * @returns a new TiledIndexLabel object with corresponding dependent
   * TiledIndexLabels
   */
  template<typename... Args>
  TiledIndexLabel operator()(const TiledIndexLabel& il1, Args... rest) {
    // EXPECTS(this->tiled_index_space().is_dependent());
    if(!this->tiled_index_space().is_dependent()) return {*this};
    std::vector<TileLabelElement> secondary_labels;
    unpack(secondary_labels, il1, rest...);
    return {*this, secondary_labels};
  }

  /// @todo: Implement
  template<typename... Args>
  TiledIndexLabel operator()(Index id, Args... rest) {
    IndexVector idx_vec;
    idx_vec.push_back(id);
    internal::unfold_vec(idx_vec, rest...);
    return tiled_index_space()(idx_vec).label();
  }

  // template <typename... Args>
  // void unfold_indices(IndexVector& vec, Args&&... args) {
  //     (vec.push_back(std::forward<Args>(args)), ...);
  // }

  /**
   * @brief Operator overload for () to construct dependent TiledIndexLabel
   * objects from the input TiledIndexLabels
   *
   * @param [in] dep_ilv
   * @returns
   */
  TiledIndexLabel operator()(const std::vector<TileLabelElement>& secondary_labels) {
    EXPECTS(this->tiled_index_space().is_dependent());
    return {*this, secondary_labels};
  }

  Label label() const { return primary_label_.label(); }

  std::string label_str() const { return primary_label_.label_str(); }

  /// @todo: this is never called from outside currently, should this be
  /// private and used internally?
  bool is_compatible_with(const TiledIndexSpace& tis) const {
    return tiled_index_space().is_compatible_with(tis);
  }

  /**
   * @brief Accessor method to dependent labels
   *
   * @returns a set of dependent TiledIndexLabels
   */
  const std::vector<TileLabelElement>& secondary_labels() const { return secondary_labels_; }

  /**
   * @brief The primary label pair that is composed of the reference
   * TiledIndexLabel and the label value
   *
   * @returns a pair of TiledIndexSpace object and Label value for the
   * TiledIndexLabel
   */
  const TileLabelElement& primary_label() const { return primary_label_; }

  /**
   * @brief Accessor to the reference TiledIndexSpace
   *
   * @returns the reference TiledIndexSpace object
   */
  const TiledIndexSpace& tiled_index_space() const { return primary_label_.tiled_index_space(); }

  bool is_dependent() const { return tiled_index_space().is_dependent(); }

  void set_spin_pos(SpinPosition spin_pos) {
    spin_pos_ = spin_pos;
    has_spin_ = true;
  }

  SpinPosition spin_pos() const { return spin_pos_; }

  // Comparison operators
  friend bool operator==(const TiledIndexLabel& lhs, const TiledIndexLabel& rhs);
  friend bool operator<(const TiledIndexLabel& lhs, const TiledIndexLabel& rhs);
  friend bool operator!=(const TiledIndexLabel& lhs, const TiledIndexLabel& rhs);
  friend bool operator>(const TiledIndexLabel& lhs, const TiledIndexLabel& rhs);
  friend bool operator<=(const TiledIndexLabel& lhs, const TiledIndexLabel& rhs);
  friend bool operator>=(const TiledIndexLabel& lhs, const TiledIndexLabel& rhs);

protected:
  // TiledIndexSpace* tis_;
  // Label label_;
  TileLabelElement              primary_label_;
  std::vector<TileLabelElement> secondary_labels_;
  // std::vector<TiledIndexLabel> dep_labels_;
  bool         has_spin_ = false;
  SpinPosition spin_pos_ = SpinPosition::ignore;
  /**
   * @brief Validates a TiledIndexLabel object with regard to its reference
   * TiledIndexSpace and dependent labels
   *
   */
  void validate() {
    // const auto& tis = tiled_index_space();
    // if(tis.is_dependent()) {
    //     const auto& sec_tis = tis.index_space().key_tiled_index_spaces();
    //     auto num_sec_tis    = sec_tis.size();
    //     auto num_sec_lbl    = secondary_labels_.size();
    //     EXPECTS((num_sec_lbl == 0) || (num_sec_lbl == num_sec_tis));
    //     for(size_t i = 0; i < num_sec_lbl; i++) {
    //         EXPECTS(secondary_labels_[i].is_compatible_with(sec_tis[i]));
    //     }
    // } else {
    //     EXPECTS(secondary_labels_.empty());
    // }
  }

private:
  void unpack(std::vector<TileLabelElement>& in_vec) {}

  template<typename... Args>
  void unpack(std::vector<TileLabelElement>& in_vec, const TileLabelElement& il1) {
    in_vec.push_back(il1);
  }

  template<typename... Args>
  void unpack(std::vector<TileLabelElement>& in_vec, const TiledIndexLabel& il1) {
    in_vec.push_back(il1.primary_label());
  }

  template<typename... Args>
  void unpack(std::vector<TileLabelElement>& in_vec, const TileLabelElement& il1, Args... rest) {
    in_vec.push_back(il1);
    unpack(in_vec, std::forward<Args>(rest)...);
  }

  template<typename... Args>
  void unpack(std::vector<TileLabelElement>& in_vec, const TiledIndexLabel& il1, Args... rest) {
    in_vec.push_back(il1.primary_label());
    unpack(in_vec, std::forward<Args>(rest)...);
  }
}; // class TiledIndexLabel

// Comparison operator implementations
inline bool operator==(const TiledIndexLabel& lhs, const TiledIndexLabel& rhs) {
  return lhs.primary_label() == rhs.primary_label() &&
         lhs.secondary_labels() == rhs.secondary_labels();
}

inline bool operator<(const TiledIndexLabel& lhs, const TiledIndexLabel& rhs) {
  return (lhs.primary_label() < rhs.primary_label()) ||
         (lhs.primary_label() == rhs.primary_label() &&
          std::lexicographical_compare(lhs.secondary_labels().begin(), lhs.secondary_labels().end(),
                                       rhs.secondary_labels().begin(),
                                       rhs.secondary_labels().end()));
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
  if(id == "all") return TiledIndexLabel{*this, lbl};
  return TiledIndexLabel{(*this)(id), lbl};
}

inline TiledIndexLabel TiledIndexSpace::label(Label lbl) const {
  return TiledIndexLabel{*this, lbl};
}

inline TiledIndexLabel TiledIndexSpace::string_label(std::string lbl_str) const {
  return TiledIndexLabel{*this, lbl_str};
}

template<size_t... Is>
auto TiledIndexSpace::labels_impl(std::string id, Label start, std::index_sequence<Is...>) const {
  return std::make_tuple(label(id, start + Is)...);
}

} // namespace tamm
