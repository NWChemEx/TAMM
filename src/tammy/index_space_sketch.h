#ifndef INDEX_SPACE_N_H_
#define INDEX_SPACE_N_H_

#include <memory>
#include <iterator>
#include "types.h"

namespace tammy
{
#if 0
  template <typename v_type>
  class range_impl {
   public:
    class iterator : 
        public std::iterator<std::random_access_iterator_tag, v_type> {

          public:
            iterator();
            iterator(int start);

            iterator(const iterator& it);
            iterator& operator=(const iterator& rhs);


            v_type& operator*();
            v_type* operator->();
            
            iterator& operator++();
            iterator operator++(int);

            iterator& operator--();
            iterator operator--(int);

            bool operator==(const iterator& rhs);
            bool operator!=(const iterator& rhs);

            iterator operator+(int s);
            iterator operator-(int s);
            friend iterator operator+(const int& lhs, const iterator& rhs);
            friend iterator operator-(const int& lhs, const iterator& rhs);

           
            bool operator<(const iterator& rhs);
            bool operator>(const iterator& rhs);
            bool operator<=(const iterator& rhs);
            bool operator>=(const iterator& rhs);

            iterator operator+=(int s);
            iterator operator-=(int s);

            v_type& operator[](const int& rhs);
          
          private:
            v_type current_val_;
            int current_step_;
            range_impl& parent_;
        };

    // Constructors
    range_impl() = default;
    range_impl(v_type end);
    range_impl(v_type begin, v_type end, v_type step = 1);

    // Copy Constructors
    range_impl(const range_impl &) = default;
    range_impl &operator=(const range_impl &) = default;

    // Destructor
    ~range_impl() = default;

    v_type operator[](int s);
    const v_type operator[](int s) const;
    
    int size() const;

    iterator begin();
    iterator end();
  
   private:
    v_type rbegin_;
    v_type rend_;
    v_type step_;
    int step_end_; 
  }; // range_impl
#endif

  using Index = int32_t;
  using Range = std::vector<Index>;
  using RangeIterator = std::vector<Index>::const_iterator;
  using Point = int32_t;
  using Attribute = Spin;
  using Tile = int32_t;

  static Range range(Index begin, Index end, int step = 1) {
    Range ret; 
    for(int i = begin; i < end; i += step) {
      ret.push_back(i);
    }
    return ret;
  } 

  static Range range(Index end) {
    return range(Index{0}, end);
  }

  class IndexSpace {
   public:
    // Constructors
    IndexSpace() = default;

    // Initializer-list based - Inde
    IndexSpace(std::initializer_list<Index> indices);

    // Initializer-list based - Aggregation
    IndexSpace(std::initializer_list<IndexSpace> spaces);
    
    // Count-based
    IndexSpace(Index end);

    // Range-based
    IndexSpace(Range r);
  
    
    // Sub-space
    IndexSpace(const IndexSpace& is, Range r = {} , std::string id = "");

    // With attributes
    IndexSpace(const IndexSpace& is, Attribute att);

    // With identifier string
    IndexSpace(const IndexSpace& is, std::string i_str, ...);

    // Copy Constructors
    IndexSpace(const IndexSpace &) = default;
    IndexSpace &operator=(const IndexSpace &) = default;

    // Destructor
    ~IndexSpace() = default;

    // Element Accesors
    Point point(Index i);

    Point& operator[](Index i);
    const Point& operator[](Index i) const;

    RangeIterator begin();
    RangeIterator end();

    Range& range(std::string id);

   private:
    Range points_;
    std::vector<IndexSpace> spaces_;
    std::vector<std::string> names_;
    std::vector<Attribute> block_attributes_;

  }; // IndexSpace

  inline bool operator==(const IndexSpace& lhs, const IndexSpace& rhs);

  inline bool operator<(const IndexSpace& lhs, const IndexSpace& rhs);

  inline bool operator!=(const IndexSpace& lhs, const IndexSpace& rhs);

  inline bool operator>(const IndexSpace& lhs, const IndexSpace& rhs);

  inline bool operator<=(const IndexSpace& lhs, const IndexSpace& rhs);

  inline bool operator>=(const IndexSpace& lhs, const IndexSpace& rhs);

  ///////////////////////////////////////////////////////////////////////
  class IndexSpacePimpl {
   public:
    IndexSpacePimpl() = default;

    IndexSpacePimpl(const IndexSpacePimpl &) = default;
    IndexSpacePimpl &operator=(const IndexSpacePimpl &) = default;
    ~IndexSpacePimpl() = default;

    template<typename... Args> 
    static IndexSpacePimpl create(Args&&... args);
  
   protected:

    std::shared_ptr<IndexSpace> p_impl_;
    
  }; // IndexSpace
  ///////////////////////////////////////////////////////////////////////
  class TiledIndexLabel;

  class TiledIndexSpace {
   public:
    // Ctors
    TiledIndexSpace() = default;

    // IndexSpace based
    TiledIndexSpace(const IndexSpace& is, Tile size = 1);

    // Sub-space
    TiledIndexSpace(const TiledIndexSpace& tis, Range r);
    TiledIndexSpace(const TiledIndexSpace& tis, Index begin);
    TiledIndexSpace(const TiledIndexSpace& tis, std::string id);

    // Copy Ctors
    TiledIndexSpace(const TiledIndexSpace &) = default;
    TiledIndexSpace &operator=(const TiledIndexSpace &) = default;

    // Dtor
    ~TiledIndexSpace() = default;

    TiledIndexLabel labels(std::string id, Label lbl);

    template<unsigned c_lbl> 
    auto range_labels(std::string id);

    const TiledIndexSpace range(std::string id);

    RangeIterator begin();
    RangeIterator end();

   private:
   std::shared_ptr<IndexSpace> is_;
   Tile size_;
    
  }; // TiledIndexSpace

  class DependentIndexSpace {
   public:
    DependentIndexSpace() = default;
    DependentIndexSpace(std::initializer_list<TiledIndexSpace> spaces);


    DependentIndexSpace(const DependentIndexSpace &) = default;
    DependentIndexSpace &operator=(const DependentIndexSpace &) = default;
    ~DependentIndexSpace() = default;

    int num_indep_indices() const; 
  
   private:
    std::shared_ptr<IndexSpace> ref_is_;
    std::vector<std::shared_ptr<TiledIndexSpace>> dep_iss_;
  }; // DependentIndexSpace


  class DependentIndexLabel;

  class TiledIndexLabel {
   public:
    // Constructor
    TiledIndexLabel() = default;

    // Copy Construtors
    TiledIndexLabel(const TiledIndexLabel &) = default;
    TiledIndexLabel &operator=(const TiledIndexLabel &) = default;

    // Destructor
    ~TiledIndexLabel() = default;

    DependentIndexLabel operator()(TiledIndexLabel il1) const;
    DependentIndexLabel operator()() const;
    DependentIndexLabel operator()(TiledIndexLabel il1, TiledIndexLabel il2) const;
  
   private:
    std::shared_ptr<TiledIndexSpace> tis_;
    Label label_; 
  }; // TiledIndexLabel

  class DependentIndexLabel {
   public:
    DependentIndexLabel() = default;


    DependentIndexLabel(const DependentIndexLabel &) = default;
    DependentIndexLabel &operator=(const DependentIndexLabel &) = default;
    ~DependentIndexLabel() = default;
  
   private:
    TiledIndexLabel til_;
    std::vector<TiledIndexLabel> indep_labels_;
  }; // DependentIndexLabel
} // tammy

#endif // INDEX_SPACE_N_H_