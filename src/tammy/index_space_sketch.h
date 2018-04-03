#ifndef INDEX_SPACE_SKETCH_H_
#define INDEX_SPACE_SKETCH_H_

#include <memory>
#include <iterator>
#include <map>
#include "types.h"

namespace tammy
{
  using Index = uint32_t;

  template<class T>
  using Range = std::vector<T>;
  template<class T>
  using RangeIterator = typename std::vector<T>::const_iterator;

  struct RangeData{
    int lo_;
    int hi_;
    int step_;

    RangeData(int lo, int hi, int step = 1) : lo_{lo}, hi_{hi}, step_{step} {}
  };

  static Range<Index> range(Index begin, Index end, int step = 1) {
    Range<Index> ret; 
    for(int i = begin; i < end; i += step) {
      ret.push_back(i);
    }
    return ret;
  } 

  static Range<Index> range(Index end) {
    return range(Index{0}, end);
  }

  static RangeData range(int lo, int high, int step = 1) {
    return RangeData(lo, high, step);
  }

  static std::vector<Index> ConstructIndexVector(const RangeData& range) {
    std::vector<Index> ret = {};
    for (int i = range.lo_; i < range.hi_; i+= range.step_) {
      ret.push_back(i);
    }

    return ret;
  }

  using SubspaceMap = std::map<std::string, const std::vector<RangeData>>;

  template<typename T>
  class Attribute {
   public:
    Attribute() = default;

    Attribute(const Attribute &) = default;
    Attribute &operator=(const Attribute &) = default;
    ~Attribute() = default;

    T operator () (Index ind) {
      NOT_IMPLEMENTED();
    }
    std::vector<Range<Index>>::const_iterator begin() {
      NOT_IMPLEMENTED();
    }
    std::vector<Range<Index>>::const_iterator end() {
      NOT_IMPLEMENTED();
    }

   protected:
    std::map<T, std::vector<Range<Index>>> attr_map_;
  }; // Attribute

  using SpinAttribute = Attribute<Spin>;
  using SpatialAttribute = Attribute<Spatial>;

  class IndexSpaceImpl;
  class RangeIndexSpaceImpl;
  class SubSpaceImpl;
  class UnionSpaceImpl;
  class DependentIndexSpaceImpl;
  
  class IndexSpace {
   public:
    // Constructors
    IndexSpace() = default;

    IndexSpace(const IndexSpace&) = default;
    IndexSpace(IndexSpace&&) = default;
    ~IndexSpace() = default;
    IndexSpace& operator = (const IndexSpace&) = default;
    IndexSpace& operator = (IndexSpace&&) = default;

    // creating named sub-space groups
    // "inheriting" names sub-spaces
    // same as above for attributes

    // Initializer-list / vector based. no inherited named subspaces
    IndexSpace(const std::initializer_list<Index>& indices)
        : IndexSpace{indices, {}, SpinAttribute{}, SpatialAttribute{}} {}

    IndexSpace(const std::vector<Index>& indices,
               const SubspaceMap& named_subspaces = {},
               const SpinAttribute& spin = SpinAttribute{},
               const SpatialAttribute& spatial = SpatialAttribute{})
        : impl_{std::dynamic_pointer_cast<IndexSpaceImpl>(std::make_shared<RangeIndexSpaceImpl>(indices, named_subspaces, spin, spatial))} {}

    // Sub-space. no inherited named subspaces
    // all attributes in \param{is} are inherited.
    // to iterate over the ranges into which an attribute partitions this index space,
    // the parent space's attributes need to "chopped" to match the \param{range}.
    // TODO: we also need string based named subspaces. e.g. "alpha" = "occ_alpha, virt_alpha".
    IndexSpace(const IndexSpace& is, 
               const RangeData& range,
               const SubspaceMap& named_subspaces = {})
        : impl_{std::dynamic_pointer_cast<IndexSpaceImpl>(std::make_shared<SubSpaceImpl>(is, range, named_subspaces))} {}

    // Aggregate. named subspaces from all space in \param{spaces} with a non-empty name are "inherited"
    // any attributes in all sp in \param{spaces} is inherited. If any of the aggregated spaces does
    // not have an attribute, that attribute is not inherited.
    // TODO: we could have functions to get "named" subspaces by position. Basically fn(i) returns spaces[i].
    // IndexSpace(const std::initializer_list<IndexSpace>& spaces)
    //     : impl_{std::dynamic_pointer_cast<IndexSpaceImpl>(std::make_shared<UnionSpaceImpl>(spaces))} {}
    IndexSpace(const std::vector<IndexSpace>& spaces,
               const std::vector<std::string>& names = {},
               const SubspaceMap& named_subspaces = {},
               const std::map<std::string, std::vector<std::string>>& subspace_references = {})
        : impl_{std::dynamic_pointer_cast<IndexSpaceImpl>(std::make_shared<UnionSpaceImpl>(spaces, names, named_subspaces, subspace_references))} {}
    
    IndexSpace(const std::vector<IndexSpace>& spaces,
               const std::vector<std::string>& names = {},
               const std::map<std::string, std::vector<std::string>>& subspace_references = {},
               const SubspaceMap& named_subspaces = {})
        : impl_{std::dynamic_pointer_cast<IndexSpaceImpl>(std::make_shared<UnionSpaceImpl>(spaces, names, named_subspaces, subspace_references))} {}
               
      
    // Dependent : what about attributes here
    // named subspaces in dep_space_relation are "inherited" by default. Note that the index spaces in
    // dep_space_relation might have no relation with one another.
    // Attributes in dep_space_relation are inherited by default.
    IndexSpace(const std::vector<IndexSpace>& indep_spaces,
               const std::map<std::vector<Index>, IndexSpace>& dep_space_relation)
        : impl_{std::dynamic_pointer_cast<IndexSpaceImpl>(std::make_shared<DependentIndexSpaceImpl>(indep_spaces, dep_space_relation))} {}
               

    // Dependent subspace : what about attributes here
    // named subspaces in dep_space_relation are "inherited" by default. Note that the index spaces in
    // dep_space_relation are all subspaces of ref_space.
    //all attributes in \param{ref_space} are "inherited". see also the sub-space constructor comments.
    IndexSpace(const std::vector<IndexSpace>& indep_spaces,
               const IndexSpace& ref_space,
               const std::map<std::vector<Index>, IndexSpace>& dep_space_relation);

    // Accessors
    const Index& point(Index i, const std::vector<Index>& indep_index = {});
    const Index& operator[](Index i) const;

    // IndexSpace accessors
    const IndexSpace& operator()(const std::vector<Index>& indep_index = {}) const;
    const IndexSpace& operator()(const std::string& named_subspace_id) const;

    // Iterators
    RangeIterator<Index> begin() const;
    RangeIterator<Index> end() const;

    //Size of this index space
    Index size() const;

   protected:
    std::shared_ptr<IndexSpaceImpl> impl_;
  }; // IndexSpace

  class IndexSpaceImpl {
   public:
    virtual const Index& point(Index i, const std::vector<Index>& indep_index = {}) const = 0;
    virtual const Index& operator[](Index i) const = 0;

    virtual const IndexSpace& operator()(const std::vector<Index>& indep_index = {}) const = 0;
    virtual const IndexSpace& operator() (const std::string& named_subspace_id) const = 0;

    // Iterators
    virtual RangeIterator<Index> begin() const = 0;
    virtual RangeIterator<Index> end() const = 0;

    virtual Index size() const = 0;
    
  }; // IndexSpaceImpl

  class RangeIndexSpaceImpl : public  IndexSpaceImpl {
   public:
    // TODO: do we need a default constructor?
    // RangeIndexSpaceImpl() = default;

    // TODO : do we need these constructor/operators?
    RangeIndexSpaceImpl(RangeIndexSpaceImpl &&) = default;
    RangeIndexSpaceImpl(const RangeIndexSpaceImpl &) = default;
    RangeIndexSpaceImpl &operator=(RangeIndexSpaceImpl &&) = default;
    RangeIndexSpaceImpl &operator=(const RangeIndexSpaceImpl &) = default;
    ~RangeIndexSpaceImpl() = default;

    
    // Range-based. no inherited named subspaces
    // TODO: optimization - constructing map<string, IndexSpace> can be delayed 
    // until a specific subspace is requested
    RangeIndexSpaceImpl(const std::vector<Index>& indices,
                        const SubspaceMap& named_ranges,
                        const SpinAttribute& spin,
                        const SpatialAttribute& spatial)
            : indices_{indices}, named_ranges_{named_ranges}, named_subspaces_{construct_subspaces(named_ranges)}, spin_{spin}, spatial_{spatial} {}

     // Element Accesors
    const Index& point(Index i, const std::vector<Index>& indep_index = {}) const{
      return indices_[i];
    }
    const Index& operator[](Index i) const {
      return indices_[i];
    }

    // TODO: This should return corresponding IndexSpace object for (*this)
    const IndexSpace& operator()(const std::vector<Index>& indep_index = {}) const {
      NOT_IMPLEMENTED();
    }
    
    const IndexSpace& operator()(const std::string& named_subspace_id) const {
      return named_subspaces_.at(named_subspace_id);
    }

    // Iterators
    RangeIterator<Index> begin() const {
      return indices_.begin();
    }
    RangeIterator<Index> end() const {
      return indices_.end();
    }

    Index size() const {
      return indices_.size();
    }

   protected:
    std::vector<Index> indices_;
    SubspaceMap named_ranges_;
    std::map<std::string, IndexSpace> named_subspaces_;
    SpinAttribute spin_;
    SpatialAttribute spatial_;

   private:
    std::map<std::string, IndexSpace> construct_subspaces(const SubspaceMap& in_map){
      std::map<std::string, IndexSpace> ret;
      for(auto& kv: in_map){
        std::string name = kv.first;
        Range<Index> tempRange = {};
        for(auto& range: kv.second){
          for (auto& i: ConstructIndexVector(range))
          {
           tempRange.push_back(indices_[i]);
          }
        }
        ret.insert({name, IndexSpace{tempRange}});
      }
      
      return ret;
    }
  }; // RangeIndexSpaceImpl

  class SubSpaceImpl : public IndexSpaceImpl {
   public:
    //TODO: do we need a default constructor?
    // SubSpaceImpl() = default;

    // Sub-space construction
    // TODO: optimization - constructing map<string, IndexSpace> can be delayed 
    // until a specific subspace is requested
    SubSpaceImpl(const IndexSpace& is, 
                 const RangeData& range,
                 const SubspaceMap& named_ranges) 
        : ref_space_{is}, indices_{ConstructIndexVector(range)},
          named_ranges_{named_ranges}, 
          named_subspaces_{construct_subspaces(named_ranges)},
          spin_{construct_spin(is, range)},
          spatial_{construct_spatial(is, range)} {}

    // TODO : do we need these constructor/operators
    SubSpaceImpl(SubSpaceImpl &&) = default;
    SubSpaceImpl(const SubSpaceImpl &) = default;
    SubSpaceImpl &operator=(SubSpaceImpl &&) = default;
    SubSpaceImpl &operator=(const SubSpaceImpl &) = default;
    ~SubSpaceImpl() = default;

    // IndexSpaceImpl Methods
    const Index& point(Index i, const std::vector<Index>& indep_index = {}) const {
      return indices_[i];
    }
    const Index& operator[](Index i) const {
      return indices_[i];
    }


    // TODO: This should return corresponding IndexSpace object for (*this)
    const IndexSpace& operator()(const std::vector<Index>& indep_index = {}) const {
      NOT_IMPLEMENTED();
    }
    
    const IndexSpace& operator()(const std::string& named_subspace_id) const {
      return named_subspaces_.at(named_subspace_id);
    }

    // Iterators
    RangeIterator<Index> begin() const {
      return indices_.begin();
    }
    RangeIterator<Index> end() const {
      return indices_.end();
    }

    Index size() const {
      return indices_.size();
    }
  
   protected:
    IndexSpace ref_space_;
    std::vector<Index> indices_;
    SubspaceMap named_ranges_;
    std::map<std::string, IndexSpace> named_subspaces_;
    SpinAttribute spin_;
    SpatialAttribute spatial_;

   private:
    std::map<std::string, IndexSpace> construct_subspaces(const SubspaceMap& in_map){
      std::map<std::string, IndexSpace> ret;
      for(auto& kv: in_map){
        std::string name = kv.first;
        std::vector<Index> indices = {};
        for(auto& range: kv.second){
          for (auto& i: ConstructIndexVector(range))
          {
           indices.push_back(indices_[i]);
          }
        }
        ret.insert({name, IndexSpace{indices}});
      }
      
      return ret;
    }
    //TODO: Implement depending on Attribute structure
    SpinAttribute construct_spin(const IndexSpace& is, 
                                 const RangeData& range) {
      NOT_IMPLEMENTED();
    }
    //TODO: Implement depending on Attribute structure
    SpatialAttribute construct_spatial(const IndexSpace& is, 
                                       const RangeData& range) {
      NOT_IMPLEMENTED();
    }
  }; // SubSpaceImpl

  class UnionSpaceImpl : public IndexSpaceImpl {
   public:
    // UnionSpaceImpl() = default;

 
    // TODO : do we need these constructor/operators
    UnionSpaceImpl(UnionSpaceImpl &&) = default;
    UnionSpaceImpl(const UnionSpaceImpl &) = default;
    UnionSpaceImpl& operator=(UnionSpaceImpl &&) = default;
    UnionSpaceImpl& operator=(const UnionSpaceImpl &) = default;
    ~UnionSpaceImpl() = default;

    // IndexSpace aggregation construction
    // TODO: optimization - constructing map<string, IndexSpace> can be delayed 
    // until a specific subspace is requested
    UnionSpaceImpl(const std::vector<IndexSpace>& spaces,
                   const std::vector<std::string>& names,
                   const SubspaceMap& named_ranges,
                   const std::map<std::string, std::vector<std::string>>& subspace_references)
           : ref_spaces_{spaces},
             indices_{construct_indices(spaces,names)},
             named_ranges_{named_ranges},
             named_subspaces_{construct_subspaces(named_ranges)},
             spin_{construct_spin(spaces,names)},
             spatial_{construct_spatial(spaces,names)} {
               add_ref_names(spaces,names);
             }

     UnionSpaceImpl(const std::initializer_list<IndexSpace>& spaces)
          : UnionSpaceImpl(spaces, {}, {}, {}) {}
                   
    // IndexSpaceImpl Methods
    const Index& point(Index i, const std::vector<Index>& indep_index = {}) const {
      return indices_[i];
    }
    const Index& operator[](Index i) const {
      return indices_[i];
    }

    // TODO: This should return corresponding IndexSpace object for (*this)
    const IndexSpace& operator()(const std::vector<Index>& indep_index = {}) const {
      NOT_IMPLEMENTED();
    }

    const IndexSpace& operator()(const std::string& named_subspace_id) const {
      return named_subspaces_.at(named_subspace_id);
    }

    // Iterators
    RangeIterator<Index> begin() const {
      return indices_.begin();
    }
    RangeIterator<Index> end() const {
      return indices_.end();
    }

    Index size() const {
      return indices_.size();
    }

   protected:
    std::vector<IndexSpace> ref_spaces_;

    std::vector<Index> indices_;
    SubspaceMap named_ranges_;
    std::map<std::string, IndexSpace> named_subspaces_;
    SpinAttribute spin_;
    SpatialAttribute spatial_;

   private:
    void add_ref_names(const std::vector<IndexSpace>& ref_spaces,
                       const std::vector<std::string>& ref_names) {
      EXPECTS(ref_spaces.size() == ref_names.size());
      size_t i = 0;
      for(const auto& space: ref_spaces) {
        named_subspaces_.insert({ref_names[i], space});
        i++;
      }
    }
    std::vector<Index> construct_indices(const std::vector<IndexSpace>& spaces,
                                         const std::vector<std::string>& names) {

      std::vector<Index> ret = {};
      for(const auto& space : spaces) {
       ret.insert(ret.end(), space.begin(), space.end());
      }
  
      return ret;
    }

    std::map<std::string, IndexSpace> construct_subspaces(const SubspaceMap& in_map){
      std::map<std::string, IndexSpace> ret;
      for(auto& kv: in_map){
        std::string name = kv.first;
        std::vector<Index> indices = {};
        for(auto& range: kv.second){
          for (auto& i: ConstructIndexVector(range))
          {
           indices.push_back(indices_[i]);
          }
        }
        ret.insert({name, IndexSpace{indices}});
      }
      
      return ret;
    }

    //TODO: Implement depending on Attribute structure                             
    SpinAttribute construct_spin(const std::vector<IndexSpace>& spaces,
                                 const std::vector<std::string>& names) {
      NOT_IMPLEMENTED();
    }
    //TODO: Implement depending on Attribute structure
    SpatialAttribute construct_spatial(const std::vector<IndexSpace>& spaces,
                                       const std::vector<std::string>& names) {
      NOT_IMPLEMENTED();
    }

  }; // UnionSpaceImpl

  class DependentIndexSpaceImpl : public IndexSpaceImpl {
   public:
    DependentIndexSpaceImpl() = default;

    DependentIndexSpaceImpl(const std::vector<IndexSpace>& indep_spaces,
                            const std::map<std::vector<Index>, IndexSpace>& dep_space_relation)
            : dep_spaces_{indep_spaces}, dep_space_relation_{dep_space_relation} {}
    
    DependentIndexSpaceImpl(DependentIndexSpaceImpl &&) = default;
    DependentIndexSpaceImpl(const DependentIndexSpaceImpl &) = default;
    DependentIndexSpaceImpl &operator=(DependentIndexSpaceImpl &&) = default;
    DependentIndexSpaceImpl &operator=(const DependentIndexSpaceImpl &) = default;
    ~DependentIndexSpaceImpl() = default;

    // IndexSpaceImpl Methods
    const Index& point(Index i, const std::vector<Index>& indep_index = {}) const {
      return dep_space_relation_.at(indep_index)[i];
    }

    // TODO: what should ve returned?
    const Index& operator[](Index i) const {
      NOT_IMPLEMENTED();
    }

    const IndexSpace& operator()(const std::vector<Index>& indep_index = {}) const {
      return dep_space_relation_.at(indep_index);
    }

    // TODO: What should this return?
    const IndexSpace& operator()(const std::string& named_subspace_id) const {
      NOT_IMPLEMENTED(); 
    }

    // Iterators
    // TODO: Error on call
    RangeIterator<Index> begin() const {
      NOT_IMPLEMENTED(); 
    }
    RangeIterator<Index> end() const {
      NOT_IMPLEMENTED(); 
    }

    // TODO: What should this return?
    Index size() const {
      NOT_IMPLEMENTED(); 
    }
  
   private:
    std::vector<IndexSpace> dep_spaces_;
    std::map<std::vector<Index>, IndexSpace> dep_space_relation_;
  }; // DependentIndexSpaceImpl

  //////////////////////////////////////////////////////////////////////////////////
  // IndexSpace Method Implementations
  const Index& IndexSpace::point(Index i, const std::vector<Index>& indep_index) {
    return impl_->point(i, indep_index);
  }
  const Index& IndexSpace::operator[](Index i) const {
    return impl_->operator[](i);
  }

  const IndexSpace& IndexSpace::operator()(const std::vector<Index>& indep_index) const {
    return impl_->operator()(indep_index);
  }
  const IndexSpace& IndexSpace::operator()(const std::string& named_subspace_id) const {
    return impl_->operator()(named_subspace_id);
  }

  // Iterators
  RangeIterator<Index> IndexSpace::begin() const{
    return impl_->begin();
  }
  RangeIterator<Index> IndexSpace::end() const{
    return impl_->end();
  }

  //Size of this index space
  Index IndexSpace::size() const {
    return impl_->size();
  }


} // tammy

#endif // INDEX_SPACE_SKETCH_H_