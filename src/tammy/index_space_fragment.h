// Copyright (C) 2018, Pacific Northwest National Laboratory

#ifndef INDEX_SPACE_FRAGMENT_H_
#define INDEX_SPACE_FRAGMENT_H_

#include <memory>
#include "types.h"
#include "strong_num_indexed_vector.h"

namespace tammy
{

// enum class RangeName { occ, virt, all, none };

// RangeName RangeNameStr(std::string name){
//   if(name == "occ") 
//     return RangeName::occ;
//   if(name == "virt") 
//     return RangeName::virt;
//   if(name == "all") 
//     return RangeName::all;

//   return RangeName::none;
// }

using Index = BlockIndex;
using IndexIterator = std::vector<Index>::const_iterator;

class IndexSpaceFragment {
 public:
  // Constructors
  IndexSpaceFragment() = default;
  IndexSpaceFragment(BlockIndex lo , BlockIndex hi, std::vector<Index> points = {});
  IndexSpaceFragment(uint64_t lo, uint64_t hi, std::vector<Index> points = {});

  // Copy Constructor/Operators
  IndexSpaceFragment(const IndexSpaceFragment &) = default;
  IndexSpaceFragment &operator=(const IndexSpaceFragment &) = default;

  // Move Constructor/Operator
  IndexSpaceFragment(IndexSpaceFragment &&) = default;
  IndexSpaceFragment &operator=(IndexSpaceFragment &&) = default;

  // Destructors
  ~IndexSpaceFragment() = default;

  Index& point(size_t i);
  const Index& point(size_t i) const;

  Index& operator ()(size_t i);
  const Index& operator ()(size_t i) const;

  IndexIterator begin();
  IndexIterator end();

  Size size() const;

  const BlockIndex& lo() const;
  const BlockIndex& hi() const;

  const Offset offset() const;

  bool has_spin() const;
  Spin spin() const;

  bool has_spatial() const;
  Irrep spatial() const;

  bool is_subset(const IndexSpaceFragment& isf) const;
  bool is_disjoint(const IndexSpaceFragment& isf) const;


 private:
  std::vector<Index> points_;

  BlockIndex lo_;
  BlockIndex hi_;

  Spin spin_;
  Irrep spatial_;
}; // IndexSpaceFragment

class IndexSpaceX {
 public: 
  IndexSpaceX() = default;
  IndexSpaceX(std::initializer_list<IndexSpaceFragment> fragments) : fragments_(fragments) {}

  IndexSpaceX(const IndexSpaceX &) = default;
  IndexSpaceX &operator=(const IndexSpaceX &) = default;
  ~IndexSpaceX() = default;

  IndexSpaceFragment& fragment(size_t i) const;

  Index& operator ()(size_t i) const;
  
  IndexIterator begin();
  IndexIterator end();

  bool is_subset(const IndexSpaceX& is) const;
  
 private:
  std::vector<IndexSpaceFragment> fragments_;
}; // IndexSpaceX

// struct TileInfo
// {
//   size_t size;
// }; // TileInfo

class TiledIndexRange;
class TiledIndexLabel;

class TiledIndexSpace {
 public:
  TiledIndexSpace() = default;
  TiledIndexSpace(IndexSpaceX is, size_t tile_size);

  TiledIndexSpace(const TiledIndexSpace &) = default;
  TiledIndexSpace &operator=(const TiledIndexSpace &) = default;
  ~TiledIndexSpace() = default;

  const TiledIndexRange& range(std::string r_str) const;

  template <typename... LabelArgs>
  auto range_labels(std::string r_str, LabelArgs... labels);

  bool is_identical_to(const TiledIndexSpace& tis) const;
  bool is_compatible_with(const TiledIndexSpace& tis) const;

 private:
  IndexSpaceX is_;
  size_t tile_size_; // default: 1
};

class TiledIndexRange {
 public:
  TiledIndexRange() = default;
  TiledIndexRange(std::shared_ptr<TiledIndexSpace> is, RangeValue rv);

  TiledIndexRange(const TiledIndexRange &) = default;
  TiledIndexRange &operator=(const TiledIndexRange &) = default;
  ~TiledIndexRange() = default;

  const std::shared_ptr<TiledIndexSpace> is() const;
  RangeValue rv() const;

  TiledIndexLabel labels(Label label) const;
  template<typename... LabelArgs>
  auto labels(Label label, LabelArgs... labels) const;

 private:
  const std::shared_ptr<TiledIndexSpace> is_;
  RangeValue rv_;
}; // TiledIndexSpace

class TiledIndexLabel {
 public:
  TiledIndexLabel() = default;
  TiledIndexLabel(TiledIndexRange ir, Label label);

  TiledIndexLabel(const TiledIndexLabel &) = default;
  TiledIndexLabel &operator=(const TiledIndexLabel &) = default;
  ~TiledIndexLabel() = default;

  const TiledIndexRange& ir() const;

  Label label() const;

 private:
  TiledIndexRange ir_;
  Label label_;
}; // TiledIndexLabel


}

#endif // INDEX_SPACE_FRAGMENT_H_