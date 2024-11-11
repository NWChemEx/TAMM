#pragma once

#include "tamm/tensor.hpp"

namespace tamm {

// template<typename T>
// class LabeledTensor;
using Char2TisMap = std::unordered_map<char, std::string>;

struct BlockSparseInfo {
  // Input members
  TiledIndexSpaceVec       full_tis_vec;
  std::vector<std::string> allowed_blocks;
  Char2TisMap              char_to_sub_tis;
  std::vector<std::string> disallowed_blocks;
  NonZeroCheck             is_non_zero = [](const IndexVector&) -> bool { return true; };

  // Generated members
  std::vector<TiledIndexSpaceVec> allowed_tis_vecs;
  std::vector<TiledIndexSpaceVec> disallowed_tis_vecs;

  BlockSparseInfo()                                  = default;
  BlockSparseInfo(BlockSparseInfo&&)                 = default;
  BlockSparseInfo(const BlockSparseInfo&)            = default;
  BlockSparseInfo& operator=(BlockSparseInfo&&)      = default;
  BlockSparseInfo& operator=(const BlockSparseInfo&) = default;
  ~BlockSparseInfo()                                 = default;

  BlockSparseInfo(TiledIndexSpaceVec tis_vec, std::vector<std::string> allowed_strs,
                  Char2TisMap char_to_sub_str, std::vector<std::string> disallowed_strs = {}):
    full_tis_vec(tis_vec),
    allowed_blocks(allowed_strs),
    char_to_sub_tis(char_to_sub_str),
    disallowed_blocks(disallowed_strs) {
    for(size_t i = 0; i < allowed_blocks.size(); i++) {
      auto               block_str = allowed_blocks[i];
      TiledIndexSpaceVec tis_vec;
      for(size_t j = 0; j < block_str.size(); j++) {
        auto lbl_char = block_str[j];
        tis_vec.push_back(full_tis_vec[j](char_to_sub_tis[lbl_char]));
      }
      allowed_tis_vecs.push_back(tis_vec);
    }

    for(size_t i = 0; i < disallowed_blocks.size(); i++) {
      auto               block_str = disallowed_blocks[i];
      TiledIndexSpaceVec tis_vec;
      for(size_t j = 0; j < block_str.size(); j++) {
        auto lbl_char = block_str[j];
        tis_vec.push_back(full_tis_vec[j](char_to_sub_tis[lbl_char]));
      }
      disallowed_tis_vecs.push_back(tis_vec);
    }
  }
};

/// @brief Creates a local copy of the distributed tensor
/// @tparam T Data type for the tensor being made local
template<typename T>
class BlockSparseTensor: public Tensor<T> { // move to another hpp
public:
  BlockSparseTensor()                                    = default;
  BlockSparseTensor(BlockSparseTensor&&)                 = default;
  BlockSparseTensor(const BlockSparseTensor&)            = default;
  BlockSparseTensor& operator=(BlockSparseTensor&&)      = default;
  BlockSparseTensor& operator=(const BlockSparseTensor&) = default;
  ~BlockSparseTensor()                                   = default;

  /// @brief
  /// @param tis_vec
  /// @param sparse_info
  BlockSparseTensor(TiledIndexSpaceVec tis_vec, BlockSparseInfo sparse_info):
    Tensor<T>(tis_vec, construct_is_non_zero_check(tis_vec, sparse_info)) {}

  /// @brief s
  /// @param tis_vec
  /// @param sparse_info
  /// @return
  NonZeroCheck construct_is_non_zero_check(const TiledIndexSpaceVec& tis_vec,
                                           BlockSparseInfo           sparse_info) const {
    auto is_within_tis = [](size_t block_offset, size_t tis_lo, size_t tis_hi) -> bool {
      return (block_offset >= tis_lo && block_offset < tis_hi);
    };

    auto is_in_allowed_blocks = [tis_vec, allowed_tis_vecs = sparse_info.allowed_tis_vecs,
                                 is_within_tis](const IndexVector& blockid) -> bool {
      std::vector<size_t> blockid_offsets;
      for(size_t i = 0; i < blockid.size(); i++) {
        blockid_offsets.push_back(tis_vec[i].tile_offset(blockid[i]));
      }

      for(size_t i = 0; i < allowed_tis_vecs.size(); i++) {
        auto curr_tis_vec = allowed_tis_vecs[i];
        for(size_t j = 0; j < blockid_offsets.size(); j++) {
          if(!is_within_tis(blockid_offsets[j], curr_tis_vec[j].tile_offsets().front(),
                            curr_tis_vec[j].tile_offsets().back())) {
            return false;
          }
        }
      }

      return true;
    };

    auto is_in_disallowed_blocks = [tis_vec, disallowed_tis_vecs = sparse_info.disallowed_tis_vecs,
                                    is_within_tis](const IndexVector& blockid) -> bool {
      std::vector<size_t> blockid_offsets;
      for(size_t i = 0; i < blockid.size(); i++) {
        blockid_offsets.push_back(tis_vec[i].tile_offset(blockid[i]));
      }

      for(size_t i = 0; i < disallowed_tis_vecs.size(); i++) {
        std::cerr << __FUNCTION__ << " " << __LINE__ << std::endl;

        auto curr_tis_vec = disallowed_tis_vecs[i];
        for(size_t j = 0; j < blockid_offsets.size(); j++) {
          if(!is_within_tis(blockid_offsets[j], curr_tis_vec[j].tile_offsets().front(),
                            curr_tis_vec[j].tile_offsets().back())) {
            return false;
          }
        }
      }

      return true;
    };

    auto non_zero_check =
      [is_in_allowed_blocks, is_in_disallowed_blocks, is_non_zero = sparse_info.is_non_zero,
       allowed_tis_vecs    = sparse_info.allowed_tis_vecs,
       disallowed_tis_vecs = sparse_info.disallowed_tis_vecs](const IndexVector& blockid) -> bool {
      if(allowed_tis_vecs.size() > 0) {
        return is_in_allowed_blocks(blockid) && is_non_zero(blockid);
      }
      else if(disallowed_tis_vecs.size() > 0) {
        return (!is_in_disallowed_blocks(blockid) && is_non_zero(blockid));
      }
      else { return is_non_zero(blockid); }
    };
    return non_zero_check;
  }
};

} // namespace tamm
