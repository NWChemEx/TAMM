#pragma once

#include "tamm/tensor_base.hpp"

namespace tamm {

using Char2TISMap = std::unordered_map<char, std::string>;

/**
 * @brief Block sparse description using (dis)allowed blocks and/or a non-zero check function
 *
 */
struct BlockSparseInfo {
  // Input members
  TiledIndexSpaceVec       full_tis_vec;      /**< list of TiledIndexSpaces for referencing */
  std::vector<std::string> allowed_blocks;    /**< list of allowed blocks using string indices*/
  Char2TISMap              char_to_sub_tis;   /**< map of indices to sub-TIS names*/
  std::vector<std::string> disallowed_blocks; /**< list of disallowed blocks using string indices*/
  NonZeroCheck             is_non_zero = [](const IndexVector&) -> bool {
    return true;
  }; /**< is_non_zero check function */

  // Generated members
  std::vector<TiledIndexSpaceVec>
    allowed_tis_vecs; /**< list of allowed TIS vectors generated from allowed blocks */
  std::vector<TiledIndexSpaceVec>
    disallowed_tis_vecs; /**< list of dis-allowed TIS vectors generated from allowed blocks */

  BlockSparseInfo()                                  = default;
  BlockSparseInfo(BlockSparseInfo&&)                 = default;
  BlockSparseInfo(const BlockSparseInfo&)            = default;
  BlockSparseInfo& operator=(BlockSparseInfo&&)      = default;
  BlockSparseInfo& operator=(const BlockSparseInfo&) = default;
  ~BlockSparseInfo()                                 = default;

  /**
   * @brief Construct a new BlockSparseInfo object
   *
   * @param tis_vec list of TiledIndexSpaces for the reference
   * @param allowed_strs list of allowed string indices
   * @param char_to_sub_str map for char to string for sub-TIS
   * @param disallowed_strs list of disallowed string indices
   * @param non_zero_check non zero check function
   */
  BlockSparseInfo(
    TiledIndexSpaceVec tis_vec, std::vector<std::string> allowed_strs, Char2TISMap char_to_sub_str,
    std::vector<std::string> disallowed_strs = {},
    NonZeroCheck             non_zero_check  = [](const IndexVector&) -> bool { return true; }):
    full_tis_vec(tis_vec),
    allowed_blocks(allowed_strs),
    char_to_sub_tis(char_to_sub_str),
    disallowed_blocks(disallowed_strs),
    is_non_zero(non_zero_check) {
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

  /**
   * @brief Construct a new Block Sparse Info object
   *
   * @param tis_vec list of TiledIndexSpaces for the reference
   * @param non_zero_check non zero check function
   */
  BlockSparseInfo(TiledIndexSpaceVec tis_vec, NonZeroCheck non_zero_check):
    BlockSparseInfo(tis_vec, {}, {}, {}, non_zero_check) {}

  /**
   * @brief Internal function constructing an is_non_zero function to be used to construct a lambda
   * function based block sparse info
   *
   * @return NonZeroCheck function
   */
  NonZeroCheck construct_is_non_zero_check() const {
    auto is_in_allowed_blocks = [full_tis_vec = this->full_tis_vec,
                                 allowed_tis_vecs =
                                   this->allowed_tis_vecs](const IndexVector& blockid) -> bool {
      std::vector<size_t> ref_indices;
      for(size_t i = 0; i < blockid.size(); i++) {
        ref_indices.push_back(full_tis_vec[i].ref_indices()[blockid[i]]);
      }

      for(size_t i = 0; i < allowed_tis_vecs.size(); i++) {
        auto curr_tis_vec  = allowed_tis_vecs[i];
        bool is_disallowed = false;
        for(size_t j = 0; j < ref_indices.size(); j++) {
          auto allowed_ref_indices = curr_tis_vec[j].ref_indices();

          if(std::find(allowed_ref_indices.begin(), allowed_ref_indices.end(), ref_indices[j]) ==
             allowed_ref_indices.end()) {
            is_disallowed = true;
            break;
          }
        }
        if(!is_disallowed) { return true; }
      }

      return false;
    };

    auto is_in_disallowed_blocks =
      [full_tis_vec        = this->full_tis_vec,
       disallowed_tis_vecs = this->disallowed_tis_vecs](const IndexVector& blockid) -> bool {
      std::vector<size_t> ref_indices;
      for(size_t i = 0; i < blockid.size(); i++) {
        ref_indices.push_back(full_tis_vec[i].ref_indices()[blockid[i]]);
      }

      for(size_t i = 0; i < disallowed_tis_vecs.size(); i++) {
        auto curr_tis_vec  = disallowed_tis_vecs[i];
        bool is_disallowed = false;
        for(size_t j = 0; j < ref_indices.size(); j++) {
          auto allowed_ref_indices = curr_tis_vec[j].ref_indices();

          if(std::find(allowed_ref_indices.begin(), allowed_ref_indices.end(), ref_indices[j]) ==
             allowed_ref_indices.end()) {
            is_disallowed = true;
            break;
          }
        }
        if(!is_disallowed) { return true; }
      }

      return false;
    };

    auto allowed_tis_vecs_size    = allowed_tis_vecs.size();
    auto disallowed_tis_vecs_size = disallowed_tis_vecs.size();
    auto non_zero_check           = [is_in_allowed_blocks, is_in_disallowed_blocks,
                           is_non_zero = this->is_non_zero, allowed_tis_vecs_size,
                           disallowed_tis_vecs_size](const IndexVector& blockid) -> bool {
      if(allowed_tis_vecs_size > 0) {
        return is_in_allowed_blocks(blockid) && is_non_zero(blockid);
      }
      else if(disallowed_tis_vecs_size > 0) {
        return (!is_in_disallowed_blocks(blockid) && is_non_zero(blockid));
      }
      else { return is_non_zero(blockid); }
    };
    return non_zero_check;
  }
};
} // namespace tamm
