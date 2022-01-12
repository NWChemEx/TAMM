#ifndef TAMM_DISTRIBUTION_HPP_
#define TAMM_DISTRIBUTION_HPP_

#include "ga/ga.h"
#include <map>
#include <memory>
#include <tuple>
#include <type_traits>
#include <iostream>
#include <random>

#include "tamm/tensor_base.hpp"
#include "tamm/utils.hpp"
#include "tamm/types.hpp"
#include "tamm/proc_group.hpp"

template<typename T>
std::ostream& operator <<(std::ostream& os, const std::vector<T>& vec) {
  os<<"[";
  for(const auto& v : vec) {
    os<<v<<", ";
  }
  os<<"]";
  return os;
}

template <typename T>
std::string vec_to_string(const std::vector<T>& vec) {
  std::string result = "[";
  for (const auto& v : vec) {
    result +=  std::to_string(v)+", ";
  }
  result += "]";
  return result;
}

template <typename X, typename T>
std::string vec_to_string(const std::vector<tamm::StrongNum<X,T>>& vec) {
  std::string result = "[";
  for (const auto& v : vec) {
    result += std::to_string(v.value()) + ", ";
  }
  result += "]";
  return result;
}

namespace tamm {

/**
 * @brief Base class representing Distribution of tensor structure
 *
 */
class Distribution {
public:
    /*enum class Kind {
      invalid,
      nw,
      dense
    };*/
    /**
     * @brief Destroy the Distribution object
     *
     */
    virtual ~Distribution() {}
    /**
     * @brief Locate the tensor data for a given block id
     *
     * @param [in] blockid identifier for the block to be located
     * @returns a pair of process id and offset for the location of the block
     */
    virtual std::pair<Proc, Offset> locate(const IndexVector& blockid) const = 0;

    /**
     * @brief Get the buffer size for a process
     *
     * @param [in] proc process id to get the buffer size
     * @returns size of the buffer
     */
    virtual Size buf_size(Proc proc) const = 0;

    /**
     * @brief Clones a Distribution with a given tensor structure and process id
     *
     * @param [in] tensor_structure TensorBase object
     * @param [in] nproc number of processes
     * @returns A pointer to the Distribution object constructed with the inputs
     */
    virtual Distribution* clone(const TensorBase* tensor_structure,
                                Proc nproc) const = 0;

    /**
     * @brief Maximum size (in number of elements) to be allocated on any rank
     * for this tensor distribution
     *
     * @return Maximum buffer size on any rank
     */
    virtual Size max_proc_buf_size() const = 0;

    /**
     * @brief Maximum size of any block in this distribution
     *
     * @return Maximum size of any block (in number of elements)
     */
    virtual Size max_block_size() const = 0;
    
    /**
     * @brief Computes the hash value for the given distribution
     * 
     * @returns hash value of type size_t
     */
    virtual size_t compute_hash() const = 0;

    /**
     * @brief Gets the total tensor size distributed over the proc grid
     *
     * @returns total size of type Size (aka. uint64_t)
     */
    virtual Size total_size() const = 0;

    /**
     * @brief Return the distribution type
     * 
     * @return DistributionType 
     */
    DistributionKind kind() const {
      return kind_;
    }

    const TensorBase* get_tensor_base() const {
      return tensor_structure_;
    }

    Proc get_dist_proc() const {
      return nproc_;
    }

    size_t hash() const { return hash_; }

    void set_hash(size_t hash) { hash_ = hash; }

    void set_proc_grid(std::vector<Proc> pg) { proc_grid_ = pg; }

    std::vector<Proc> proc_grid() const { return proc_grid_; }

    /**
     * @brief Construct a new Distribution object using a TensorBase object and
     * number of processes
     *
     * @param [in] tensor_structure TensorBase object
     * @param [in] nproc number of processes
     */
    Distribution(const TensorBase* tensor_structure, Proc nproc,
                 DistributionKind kind)
        : tensor_structure_{tensor_structure}, nproc_{nproc}, kind_{kind} {}

    /**
     * @brief Equality comparison operator
     *
     * @param [in] lhs Left-hand side
     * @param [in] rhs Right-hand side
     *
     * @return true if lhs == rhs
     */
    friend bool operator==(const Distribution& lhs, const Distribution& rhs);

    /**
     * @brief Inequality comparison operator
     *
     * @param [in] lhs Left-hand side
     * @param [in] rhs Right-hand side
     *
     * @return true if lhs != rhs
     */
    friend bool operator!=(const Distribution& lhs, const Distribution& rhs);

protected:
    const TensorBase* tensor_structure_; /**< TensorBase object for the
                                            corresponding Tensor structure */
    Proc nproc_;                         /**< Number of processes */
    
    DistributionKind kind_; /**< Distribution kind */

    size_t hash_;

    std::vector<Proc> proc_grid_; /**< Processor grid */

}; // class Distribution

// Comparison operator implementations
inline bool operator==(const Distribution& lhs, const Distribution& rhs) {
    return lhs.hash() == rhs.hash();
}

inline bool operator!=(const Distribution& lhs, const Distribution& rhs) {
    return !(lhs == rhs);
}

/**
 * @brief Implementation of the Distribution object for NWChem
 *
 */
class Distribution_NW : public Distribution {
public:
    using HashValue = size_t;
    using Key       = HashValue;

    Distribution_NW(const TensorBase* tensor_structure = nullptr,
                    Proc nproc                         = Proc{1}) :
      Distribution{tensor_structure, nproc, DistributionKind::nw} {
        EXPECTS(nproc > 0);
        if(tensor_structure == nullptr) { return; }

        compute_key_offsets();
        max_block_size_ = 0;
        if (!tensor_structure->is_dense()) {
          for (const auto &blockid : tensor_structure_->loop_nest()) {
            if (tensor_structure_->is_non_zero(blockid)) {
              Size sz = tensor_structure_->block_size(blockid);
              max_block_size_ = std::max(max_block_size_, sz);
              hash_.push_back({compute_key(blockid), sz});
            }
          }
        } else {
          auto loop_nest = tensor_structure_->loop_nest();

          loop_nest.iterate([&](IndexVector blockid) {
            if (tensor_structure_->is_non_zero(blockid)) {
              Size sz = tensor_structure_->block_size(blockid);
              max_block_size_ = std::max(max_block_size_, sz);
              hash_.push_back({compute_key(blockid), sz});
            }
          });
        }
        EXPECTS(hash_.size() > 0);

        std::sort(hash_.begin(), hash_.end(),
                  [](const KeyOffsetPair& lhs, const KeyOffsetPair& rhs) {
                      return lhs.key_ < rhs.key_;
                  });

        Offset offset = 0;
        for(size_t i = 0; i < hash_.size(); i++) {
            auto sz          = hash_[i].offset_;
            hash_[i].offset_ = offset;
            offset += sz;
        }
        EXPECTS(offset > 0);
        total_size_ = offset;

        Offset per_proc_size = std::max(offset / nproc.value(), Offset{1});
        auto itr             = hash_.begin();
        auto itr_last        = hash_.end();
        for(int i = 0; i < nproc.value(); i++) {
            if(itr != itr_last) {
                proc_offsets_.push_back(Offset{itr->offset_});
            } else {
                proc_offsets_.push_back(Offset{total_size_});
            }
            itr =
              std::lower_bound(itr, itr_last, (Offset{i} + 1) * per_proc_size,
                               [](const KeyOffsetPair& hv, const Offset& v) {
                                   return hv.offset_ < v;
                               });
        }
        EXPECTS(proc_offsets_.size() == static_cast<uint64_t>(nproc.value()));
        proc_offsets_.push_back(total_size_);
        init_max_proc_buf_size();
        set_hash(compute_hash());
    }

    Distribution* clone(const TensorBase* tensor_structure, Proc nproc) const {
        /** @warning
         *  totalview LD on following statement
         *  back traced to tamm::Tensor<double>::alloc shared_ptr_base.h
         *  backtraced to ccsd_driver<double> execution_context.h
         *  back traced to main
         */
        return new Distribution_NW(tensor_structure, nproc);
    }

    std::pair<Proc, Offset> locate(const IndexVector& blockid) const {
        auto key = compute_key(blockid);
        auto itr = std::lower_bound(
          hash_.begin(), hash_.end(), KeyOffsetPair{key, 0},
          [](const KeyOffsetPair& lhs, const KeyOffsetPair& rhs) {
              return lhs.key_ < rhs.key_;
          });
        EXPECTS(itr != hash_.end());
        EXPECTS(key == itr->key_);
        auto ioffset = itr->offset_;
        auto pptr    = std::upper_bound(std::begin(proc_offsets_),
                                     std::end(proc_offsets_), Offset{ioffset});
        EXPECTS(pptr != std::begin(proc_offsets_));
        auto proc = Proc{pptr - std::begin(proc_offsets_)};
        proc -= 1;
        auto offset = Offset{ioffset - proc_offsets_[proc.value()].value()};
        return {proc, offset};
    }

    Size buf_size(Proc proc) const {
        EXPECTS(proc >= 0);
        EXPECTS(proc < nproc_);
        EXPECTS(proc_offsets_.size() > static_cast<uint64_t>(proc.value()) + 1);
        return proc_offsets_[proc.value() + 1] - proc_offsets_[proc.value()];
    }

    Size max_proc_buf_size() const override {
      return max_proc_buf_size_;
    }

    Size max_block_size() const override { return max_block_size_; }

    Size total_size() const override { return total_size_; }

    /**
     * @brief Initialize distribution for maximum buffer size on rnay rank
     */
    void init_max_proc_buf_size() {
      max_proc_buf_size_ = 0;
      for (size_t i = 0; i + 1 < proc_offsets_.size(); i++) {
        max_proc_buf_size_ =
            std::max(max_proc_buf_size_,
                     Size{(proc_offsets_[i + 1] - proc_offsets_[i]).value()});
      }
    }

    size_t compute_hash() const override {
      size_t result = static_cast<size_t>(kind());
      internal::hash_combine(result, get_dist_proc().value());
      internal::hash_combine(result, max_proc_buf_size_.value());
      internal::hash_combine(result, max_block_size_.value());
      internal::hash_combine(result, total_size_.value());

      return result;
    }

private:
    /**
     * @brief Struct for pair of HashValue and Offset
     *
     */
    struct KeyOffsetPair {
        HashValue key_;
        Offset offset_;
    };

    /**
     * @brief Computes offset for each key value
     *
     */
    void compute_key_offsets() {
        const auto& tis_list = tensor_structure_->tindices();
        int rank             = tis_list.size();
        key_offsets_.resize(rank);
        if(rank > 0) { key_offsets_[rank - 1] = 1; }
        for(int i = rank - 2; i >= 0; i--) {
            key_offsets_[i] =
              key_offsets_[i + 1] * tis_list[i + 1].max_num_tiles();
        }
    }

    /**
     * @brief Computes the key value for a given block id
     *
     * @param [in] blockid identifier for the tensor block
     * @returns a key value
     */
    Key compute_key(const IndexVector& blockid) const {
        Key key{0};
        auto rank = tensor_structure_->tindices().size();
        for(size_t i = 0; i < rank; i++) {
            key += blockid[i] * key_offsets_[i].value();
        }
        return key;
    }

    Size max_proc_buf_size_;           /**< Max buffer size on any rank */
    Size max_block_size_;              /**< Max size of any block */
    Offset total_size_;                /**< Total size of the distribution */
    std::vector<KeyOffsetPair> hash_;  /**< Vector of key and offset pairs  */
    std::vector<Offset> proc_offsets_; /**< Vector of offsets for each process */
    std::vector<Offset> key_offsets_;  /**< Vector of offsets for each key value */

}; // class Distribution_NW


/**
 *  @brief A simple round-robin distribution that allocates equal-sized blocks
 * (or the largest block) in a round-robin fashion. This distribution
 * over-allocates and ignores sparsity.
 *
 */
class Distribution_SimpleRoundRobin : public Distribution {
 public:
  using Key = int64_t;

  Distribution_SimpleRoundRobin(const TensorBase* tensor_structure = nullptr,
                                Proc nproc = Proc{1})
      : Distribution{tensor_structure, nproc, DistributionKind::simple_round_robin} {
    EXPECTS(nproc > 0);
    if (tensor_structure == nullptr) {
      return;
    }

    compute_key_offsets();
    // compute number of tiles
    total_num_blocks_ = 1;
    EXPECTS(tensor_structure != nullptr);
    for (const auto& tis : tensor_structure->tiled_index_spaces()) {
      total_num_blocks_ *= tis.max_num_tiles();
    }
    // compute max block size
    max_block_size_ = 1;
    for (const auto& tis : tensor_structure->tiled_index_spaces()) {
      max_block_size_ *= tis.max_tile_size();
    }
    // compute max buffer size on any proc
    max_proc_buf_size_ = ((total_num_blocks_ / nproc_.value()) +
                          (total_num_blocks_.value() % nproc_.value() ? 1 : 0)) *
                         max_block_size_;
    set_hash(compute_hash());
    
    start_proc_ = compute_start_proc(nproc_);
    step_proc_ = std::max(Proc{nproc_.value() / total_num_blocks_.value()}, Proc{1});
  }

  Distribution* clone(const TensorBase* tensor_structure, Proc nproc) const {
    /** @warning
     *  totalview LD on following statement
     *  back traced to tamm::Tensor<double>::alloc shared_ptr_base.h
     *  backtraced to ccsd_driver<double> execution_context.h
     *  back traced to main
     */
    return new Distribution_SimpleRoundRobin(tensor_structure, nproc);
  }

  std::pair<Proc, Offset> locate(const IndexVector& blockid) const {
    auto key = compute_key(blockid);
    EXPECTS(key >= 0 && key < total_num_blocks_);
    // return {key % nproc_.value(), (key / nproc_.value()) * max_block_size_.value()};
    Proc proc =
        (key * step_proc_.value() + start_proc_.value()) % nproc_.value();
    EXPECTS(step_proc_ == 1 || total_num_blocks_.value() <= nproc_.value());
    Offset offset = (step_proc_ != Proc{1}
                         ? Offset{0}
                         : (key / nproc_.value()) * max_block_size_.value());
    return {proc, offset};
  }

  Size buf_size(Proc proc) const {
    EXPECTS(proc >= 0);
    EXPECTS(proc < nproc_);
    return max_proc_buf_size_;
  }

  Size max_proc_buf_size() const override { return max_proc_buf_size_; }

  Size max_block_size() const override { return max_block_size_; }

  /**
   * @brief Total size of the distribution across all ranks
   *
   * @return Size
   */
  Size total_size() const override {
    return nproc_.value() * max_proc_buf_size_;
  }

  size_t compute_hash() const override {
      size_t result = static_cast<size_t>(kind());
      internal::hash_combine(result, total_num_blocks_.value());
      internal::hash_combine(result, max_proc_buf_size_.value());
      internal::hash_combine(result, max_block_size_.value());
      internal::hash_combine(result, total_size().value());

      return result;
    }

 private:
  /**
   * @brief Computes offset for each key value
   *
   */
  void compute_key_offsets() {
    const auto& tis_list = tensor_structure_->tindices();
    int rank = tis_list.size();
    key_offsets_.resize(rank);
    if (rank > 0) {
      key_offsets_[rank - 1] = 1;
    }
    for (int i = rank - 2; i >= 0; i--) {
      key_offsets_[i] = key_offsets_[i + 1] * tis_list[i + 1].max_num_tiles();
    }
  }

  /**
   * @brief Computes the key value for a given block id
   *
   * @param [in] blockid identifier for the tensor block
   * @returns a key value
   */
  Key compute_key(const IndexVector& blockid) const {
    Key key{0};
    const auto rank = blockid.size();
    for (size_t i = 0; i < rank; i++) {
      key += blockid[i] * key_offsets_[i].value();
    }
    return key;
  }

  /**
   * @brief Randomly determine the proc to hold the 0-th block. All ranks should
   * agree on this start proc. For now, we set it arbitrary to 0.
   *
   * @param nproc Number of procs in the distribution, assumed to be < MAX_PROC
   * (currently 100,000).
   * @return Proc A proc in the range [0, nproc)
   */
  static Proc compute_start_proc(Proc nproc) {
    return Proc{0};
    static std::random_device dev;
    static std::mt19937 rng(dev());
    static const int MAX_PROC = 100000;
    static std::uniform_int_distribution<std::mt19937::result_type> dist(
        0, MAX_PROC);
    return Proc{dist(rng)%nproc.value()};
  }

  Offset total_num_blocks_;         /**< Total number of blocks */
  Size max_proc_buf_size_;          /**< Max buffer size on any rank */
  Size max_block_size_;             /**< Max size of any block */
  std::vector<Offset> key_offsets_; /**< Vector of offsets for each key value */
  Proc start_proc_;                 /**< Proc with 0-th block */
  Proc step_proc_;                  /**< Step size in distributing blocks */
};                                  // class Distribution_SimpleRoundRobin

/**
 * @brief Dense distribution logic for dense multidimensional tensors.
 * Currently, it supports blocked distribution of tensora 's blocks.
 *
 */
class Distribution_Dense : public Distribution {
 public:
  /**
   * @brief Constructor
   *
   * @param tensor_structure Tensor to be distributed
   * @param nproc Number of ranks to be distributed into
   */
  Distribution_Dense(const TensorBase* tensor_structure = nullptr,
                     Proc nproc = Proc{1}, ProcGrid pg = {})
      : Distribution{tensor_structure, nproc, DistributionKind::dense} {
    EXPECTS(nproc > 0);
    if (tensor_structure == nullptr) {
      return;
    }
    EXPECTS(is_valid(tensor_structure, nproc));
    nproc_ = nproc;
    ndim_ = tensor_structure->num_modes();
    
    if(pg.empty()) {
      if(tensor_structure_->num_modes() == 2) {
        proc_grid_ = grid_factor(tensor_structure, nproc);
      }
      else {
        //TODO: Fix proc grid for nD tensors?
        proc_grid_ = compute_proc_grid(tensor_structure, nproc);
      }
    }
    else proc_grid_ = pg;
    set_proc_grid(proc_grid_);
    max_proc_with_data_ = 1;
    for (const auto& p : proc_grid_) {
      max_proc_with_data_ *= p;
    }
    tiss_ = tensor_structure->tiled_index_spaces();
    init_index_partition(tensor_structure);

    set_hash(compute_hash());
  }

  /**
   * @brief Clone this distribution, but for a different configuration. The
   * clone's internal data will match the new configuration desired, but will
   * retain the distribution strategy.
   *
   * @param tensor_structure Tensor to be distributed
   * @param nproc Number of ranks to distribute into
   * @return Distribution* Cloned distrbution object for @param tensor_structure
   * and @param nproc
   */
  Distribution* clone(const TensorBase* tensor_structure, Proc nproc) const {
    /** @warning
     *  totalview LD on following statement
     *  back traced to tamm::Tensor<double>::alloc shared_ptr_base.h
     *  backtraced to ccsd_driver<double> execution_context.h
     *  back traced to main
     */
    return new Distribution_Dense(tensor_structure, nproc, proc_grid_);
  }

  /**
   * @brief Total size of buffer (in number of elements) on the given processor
   *
   * @param proc
   * @return Size
   */
  Size buf_size(Proc proc) const override {
    EXPECTS(proc >= 0);
    EXPECTS(proc < nproc_);
    if (proc >= max_proc_with_data_) {
      return {0};
    }
    const std::vector<Proc>& grid_rank = proc_rank_to_grid_rank(proc);
    const std::vector<Offset>& pbuf_extents = proc_buf_extents(grid_rank);
    Size result = 1;
    for (const auto& e : pbuf_extents) {
      result *= e;
    }
    return result;
  }

  Size max_proc_buf_size() const override { return max_proc_buf_size_; }

  Size max_block_size() const override { return max_block_size_; }

  Size total_size() const override { 
    Size result{0};
    for(int i = 0; i < nproc_.value(); i++) { result += buf_size(i); }
    return result;
  }

  // template <typename Func>
  // void iterate(const std::vector<Proc>& grid_rank, Func&& func,
  //              std::vector<Index>& itr) const {
  //   EXPECTS(grid_rank.size() == ndim_);
  //   for (int i = 0; i < ndim_; i++) {
  //     EXPECTS(grid_rank[i] >= 0 && grid_rank[i] < proc_grid_[i]);
  //   }
  //   std::vector<Range> ranges;
  //   for (int i = 0; i < ndim_; i++) {
  //     ranges.push_back(
  //         Range{index_part_offsets_[i][grid_rank[i].value()],
  //               index_part_offsets_[i][1+grid_rank[i].value()], 1});
  //   }
  //   itr.clear();
  //   itr.resize(ndim_);
  //   loop_nest_exec(ranges, func, itr);
  // }

  std::vector<Range> proc_tile_extents(const std::vector<Proc>& grid_rank) const {
    std::vector<Range> ranges(ndim_);
    for (int i = 0; i < ndim_; i++) {
      ranges[i] = Range{index_part_offsets_[i][grid_rank[i].value()],
                        index_part_offsets_[i][1 + grid_rank[i].value()], 1};
    }
    return ranges;
  }

    /**
   * @brief Compute the position of a given @param proc in the process grid.
   * This call is only valid for ranks that have data allocated to them. If the
   * process grid excludes @param proc, this call should not be called.
   *
   * @param proc rank to be located
   * @return std::vector<Proc> Grid rank for @param proc
   *
   * @pre proc>=0 && proc < max_proc_with_data()
   */
  std::vector<Proc> proc_rank_to_grid_rank(Proc proc) const {
    EXPECTS(proc >= 0 && proc < max_proc_with_data_);
    std::vector<Proc> result(ndim_);
    Proc rest = proc;
    for (int i = ndim_ - 1; i >= 0; i--) {
      result[i] = rest % proc_grid_[i];
      rest /= proc_grid_[i];
    }
    if (ndim_ > 0) {
      result[0] = rest;
    }
    return result;
  }

   /**
   * @brief Size (in number of elements) of given block
   *
   * @param blockid Index of queried block
   * @return Size Number of elements in this block
   *
   * @pre blockid.size() == ndim_
   * @pre forall i: blockid[i] >= 0 && blockid[i] < num_tiles_[i]
   */
  Size block_size(const IndexVector& blockid) const {
    EXPECTS(blockid.size() == static_cast<uint64_t>(ndim_));
    Size result = 1;
    for (int i = 0; i < ndim_; i++) {
      result *= tiss_[i].tile_size(blockid[i]);
    }
    return result;
  }

  //private:
  /**
   * @brief Compute an effective processor grid for the given number of ranks.
   * Note that not all ranks provided might be used.
   *
   * @param tensor_structure Tensor for which the grid is to be computed
   * @param nproc Number of ranks available
   * @return std::vector<Proc> The processor grid
   *
   * @post Product of the grid size along all dimensions is less than @param
   * nproc
   */
  static std::vector<Proc> compute_proc_grid(const TensorBase* tensor_structure,
                                             Proc nproc) {
    std::vector<Proc> proc_grid;
    EXPECTS(tensor_structure != nullptr);
    int ndim = tensor_structure->num_modes();
    const int64_t dim1_size = (tensor_structure->tlabels())[0].tiled_index_space().max_num_indices();

    proc_grid = std::vector<Proc>(ndim, 1);
    if (ndim > 0) {
      proc_grid[0] = nproc;
      if(nproc > dim1_size) proc_grid[0] = dim1_size; 
    }
    return proc_grid;
  }

  /**
   * Factor p processors into 2D processor grid of dimensions px, py
   */
  static std::vector<Proc> grid_factor(const TensorBase* tensor_structure, Proc nproc) {
    int               ndim = tensor_structure->num_modes();
    std::vector<Proc> proc_grid(ndim, 1);
    const int64_t     dim1_size =
      (tensor_structure->tlabels())[0].tiled_index_space().max_num_indices();
    const int64_t dim2_size =
      (tensor_structure->tlabels())[1].tiled_index_space().max_num_indices();

    int       i, j, p = nproc.value();
    int       px, py;
    int *     idx = &px, *idy = &py;
    const int MAX_FACTOR = 512;
    int       ip, ifac, pmax, prime[MAX_FACTOR];
    int       fac[MAX_FACTOR];
    int       ix, iy, ichk;

    i = 1;
    /**
     *   factor p completely
     *   first, find all prime numbers, besides 1, less than or equal to
     *   the square root of p
     */
    ip   = (int) (sqrt((double) p)) + 1;
    pmax = 0;
    for(i = 2; i <= ip; i++) {
      ichk = 1;
      for(j = 0; j < pmax; j++) {
        if(i % prime[j] == 0) {
          ichk = 0;
          break;
        }
      }
      if(ichk) {
        pmax = pmax + 1;
        if(pmax > MAX_FACTOR) printf("Overflow in grid_factor\n");
        prime[pmax - 1] = i;
      }
    }
    /**
     *   find all prime factors of p
     */
    ip   = p;
    ifac = 0;
    for(i = 0; i < pmax; i++) {
      while(ip % prime[i] == 0) {
        ifac          = ifac + 1;
        fac[ifac - 1] = prime[i];
        ip            = ip / prime[i];
      }
    }
    /**
     *  p is prime
     */
    if(ifac == 0) {
      ifac++;
      fac[0] = p;
    }
    /**
     *    find two factors of p of approximately the
     *    same size
     */
    *idx = 1;
    *idy = 1;
    for(i = ifac - 1; i >= 0; i--) {
      ix = *idx;
      iy = *idy;
      if(ix <= iy) { *idx = fac[i] * (*idx); }
      else {
        *idy = fac[i] * (*idy);
      }
    }

    proc_grid[0] = *idx;
    proc_grid[1] = *idy;

    if(proc_grid[0] > dim1_size) proc_grid[0] = dim1_size;
    if(proc_grid[1] > dim2_size) proc_grid[1] = dim2_size;

    return proc_grid;
  }

  /**
   * @brief Initialize the tensor structure (complete object construction)
   *
   * @param tensor_structure Tensoe structure to be used in extracting
   * iformation for initialization
   */
  void init_index_partition(const TensorBase* tensor_structure) {
    int ndim = tensor_structure->num_modes();
    const std::vector<TiledIndexLabel>& tlabels = tensor_structure->tlabels();

    EXPECTS(proc_grid_.size() == static_cast<uint64_t>(ndim));
    for (int i = 0; i < ndim; i++) {
      const TiledIndexSpace& tis = tlabels[i].tiled_index_space();
      num_tiles_.push_back(tis.num_tiles());
      extents_.push_back(tis.index_space().num_indices());

      Offset off = (extents_[i].value() + proc_grid_[i].value() - 1) /
                   proc_grid_[i].value();
      const IndexVector& tile_offsets = tis.tile_offsets();
      index_part_offsets_.push_back({});
      index_part_offsets_.back().push_back(0);
      part_offsets_.push_back({});
      part_offsets_.back().push_back(0);
      auto pptr = tile_offsets.begin();
      for (int p = 0; p < proc_grid_[i] - 1; p++) {
        pptr = std::lower_bound(pptr, tile_offsets.end(), off * (p + 1));
        auto tile_part = Index(pptr - tile_offsets.begin());
        if(pptr != tile_offsets.end()) {
          part_offsets_.back().push_back(tile_offsets[tile_part]);
          index_part_offsets_.back().push_back(tile_part);
        } else {
          part_offsets_.back().push_back(extents_[i]);
          index_part_offsets_.back().push_back(num_tiles_[i]);
        }
      }
      index_part_offsets_.back().push_back(num_tiles_[i]);
      part_offsets_.back().push_back(extents_[i]);
    }
    max_proc_buf_size_ = 1;
    for (int i = 0; i < ndim_; i++) {
      Size dim = 0;
      for (size_t j = 0; j + 1 < part_offsets_[i].size(); j++) {
        dim = std::max(dim, part_offsets_[i][j + 1] - part_offsets_[i][j]);
      }
      max_proc_buf_size_ *= dim;
    }
    max_block_size_ = 1;
    for (int i = 0; i < ndim_; i++) {
      size_t dim = 0;
      for (size_t j = 0; j < tiss_[i].num_tiles(); j++) {
        dim = std::max(dim, tiss_[i].tile_size(j));
      }
      max_block_size_ *= dim;
    }
  }

  /**
   * @brief Check if the given arguments are suited to construct a valid object.
   *
   * @param tensor_structure Tensor to be distributed
   * @param proc Number of rank to be distributed into
   * @return true if these arguments are valid
   * @return false otherwise
   */
  static bool is_valid(const TensorBase* tensor_structure, Proc proc) {
    EXPECTS(proc > 0);
    // check that this is a dense tensor with no dependent spaces
    // EXPECTS(!tensor_structure->has_spin());
    // EXPECTS(!tensor_structure->has_spatial());
    const std::vector<TiledIndexLabel>& tlabels = tensor_structure->tlabels();
    for (const auto& tl : tlabels) {
      EXPECTS(!tl.is_dependent());
    }
    return true;
  }


  /**
   * @brief Convert grid rank of a process to the process rank.
   *
   * @param grid_rank Grid rank of the process
   * @return Proc linearized rank for the given @param grid_rank
   *
   * @pre grid_rank.size() == ndim_
   */
  Proc grid_rank_to_proc_rank(const std::vector<Proc>& grid_rank) const {
    Proc result = 0;
    EXPECTS(static_cast<uint64_t>(ndim_) == grid_rank.size());
    if (ndim_ > 0) {
      result += grid_rank[0];
    }
    for (int i = 1; i < ndim_; i++) {
      result = result * proc_grid_[i] + grid_rank[i];
    }
    return result;
  }

  /**
   * @brief Extent of buffer along dimension @param dim for rank @param
   * grid_rank on the process grid
   *
   * @param dim Dimension being considered
   * @param grid_rank Rank of process in the process grid along dimension @param
   * dim
   * @return Offset Buffer extent along dimension @param dim
   *
   * @pre dim>=0 && dim < ndim_
   * @pre grid_rank>=0 && grid_rank < proc_grid_[dim]
   */
  Offset proc_buf_extent_for_grid_rank(int dim, Proc grid_rank) const {
    EXPECTS(dim >= 0 && dim < ndim_);
    EXPECTS(grid_rank >= 0 && grid_rank < proc_grid_[dim]);
    return part_offsets_[dim][grid_rank.value() + 1] -
           part_offsets_[dim][grid_rank.value()];
  }

  /**
   * @brief Extent of buffer along dimension each dimension for rank with a
   * given position in the process grid
   *
   * @param grid_rank Position of rank in the process grid
   * @return std::vector<Offset> Extent along all dimensions
   *
   * @pre grid_rank.size() == ndim_
   * @pre forall i: grid_rank[i]>=0 && grid_rank[i]<proc_grid_[i]
   * @post return_value.size() == ndim_
   */
  std::vector<Offset> proc_buf_extents(
      const std::vector<Proc>& grid_rank) const {
    EXPECTS(grid_rank.size() == static_cast<uint64_t>(ndim_));
    for (int i = 0; i < ndim_; i++) {
      EXPECTS(grid_rank[0] >= 0 && grid_rank[i] < proc_grid_[i]);
    }
    std::vector<Offset> result(ndim_, 0);
    for (int i = 0; i < ndim_; i++) {
      result[i] = proc_buf_extent_for_grid_rank(i, grid_rank[i]);
    }
    return result;
  }

  /**
   * @brief Offset of given block among all blocks allocated at a given rank
   * with given grid position. Note that this is logical offset among all blocks
   * allocated to the given rank.
   *
   * @param blockid Index of block being queried
   * @param grid_rank Grid position of given rank
   * @return std::vector<Offset> Offset along each dimenion in local process's
   * logical allocation
   *
   * @pre blockid.size() == ndim_
   * @pre blockid is allocated to grid_rank
   */
  std::vector<Offset> proc_buf_block_offset_grid(
      const IndexVector& blockid, const std::vector<Proc>& grid_rank) const {
    EXPECTS(blockid.size() == static_cast<uint64_t>(ndim_));
    std::vector<Offset> result(ndim_);
    for (int i = 0; i < ndim_; i++) {
      result[i] = Offset{tiss_[i].tile_offset(blockid[i])} -
                  part_offsets_[i][grid_rank[i].value()];
    }
    return result;
  }

  /**
   * @brief Linearized of given block
   *
   * @param blockid Index of block to be located
   * @return Offset Linearized offset of block with id @param blockid
   *
   * @pre blockid.size() == ndim_
   * @pre forall i: blockid[i] >= 0 && blockid[i] < num_tiles_[i]
   */
  Offset block_offset_within_proc(const IndexVector& blockid) const {
    const std::vector<Size>& bdims = block_dims(blockid);
    const std::vector<Proc>& grid_rank = block_owner_grid_rank(blockid);
    const std::vector<Offset>& pbuf_extents = proc_buf_extents(grid_rank);
    const std::vector<Offset>& pbuf_block_offset =
        proc_buf_block_offset_grid(blockid, grid_rank);
    Offset cur_contrib = 1;
    for (int i = 0; i < ndim_; i++) {
      cur_contrib *= pbuf_extents[i];
    }
    EXPECTS(cur_contrib > 0);
    Offset result = 0;
    for (int i = 0; i < ndim_; i++) {
      result += cur_contrib / pbuf_extents[i] * pbuf_block_offset[i];
      cur_contrib = cur_contrib / pbuf_extents[i] * bdims[i];
    }
    return result;
  }

  /**
   * @brief Compute the owner grid rank for given block
   *
   * @param blockid Index of block to be located
   * @return std::vector<Proc> Grid rank of process that owns his block
   */
  std::vector<Proc> block_owner_grid_rank(const IndexVector& blockid) const {
    EXPECTS(blockid.size() == static_cast<uint64_t>(ndim_));
    std::vector<Proc> grid_rank;
    for (int i = 0; i < ndim_; i++) {
      EXPECTS(blockid[i] >= 0 && blockid[i] < num_tiles_[i]);
      auto pptr = std::upper_bound(index_part_offsets_[i].begin(),
                                   index_part_offsets_[i].end(), blockid[i]);
      EXPECTS(pptr != index_part_offsets_[i].begin());
      grid_rank.push_back(Proc(pptr - index_part_offsets_[i].begin() - 1));
    }
    return grid_rank;
  }

  /**
   * @brief Compute rank of process owning this block
   *
   * @param blockid Block index to be located
   * @return Proc Rank that owns this block
   *
   * @pre blockid.size() == ndim_
   * @pre forall i: blockid[i] >= 0 && blockid[i] < num_tiles_[i]
   */
  Proc block_owner(const IndexVector& blockid) const {
    return grid_rank_to_proc_rank(block_owner_grid_rank(blockid));
  }

  /**
   * @brief Determine the given block's location
   *
   * @param blockid Index of block to be located
   * @return std::pair<Proc, Offset> Rank of process owning @param blockid and
   * linearized offset in this process
   *
   * @pre blockid.size() == ndim_
   * @pre forall i: blockid[i] >= 0 && blockid[i] < num_tiles_[i]
   */
  std::pair<Proc, Offset> locate(const IndexVector& blockid) const {
    return {block_owner(blockid), block_offset_within_proc(blockid)};
  }

  /**
   * @brief Dimensions of a given block.
   *
   * @param blockid Index of block
   * @return std::vector<Size> Dimensions of block with id @param blockid
   *
   * @pre blockid.size() == ndim_
   * @pre forall i: blockid[i] >= 0 && blockid[i] < num_tiles_[i]
   */
  std::vector<Size> block_dims(const IndexVector& blockid) const {
    EXPECTS(blockid.size() == static_cast<uint64_t>(ndim_));
    std::vector<Size> bdims(ndim_);
    for (size_t i = 0; i < bdims.size(); i++) {
      bdims[i] = tiss_[i].tile_size(blockid[i]);
    }
    return bdims;
  }

  size_t compute_hash() const override {
      size_t result = static_cast<size_t>(kind());
      internal::hash_combine(result, get_dist_proc().value());
      internal::hash_combine(result, tiss_.size());

      for(const auto& tis : tiss_) {
          internal::hash_combine(result, tis.hash());
      }

      internal::hash_combine(result, max_proc_with_data_.value());
      internal::hash_combine(result, max_proc_buf_size_.value());
      internal::hash_combine(result, max_block_size_.value());

      return result;
  }

  Proc nproc_; /**< Number of ranks */
  std::vector<TiledIndexSpace>
      tiss_; /**< TiledIndexSpace associated with each dimension */
  int ndim_; /**< Number of dimensions in underlying tensor */
  // std::vector<Proc> proc_grid_;  /**< Processor grid */
  Proc max_proc_with_data_;      /**< Max ranks with any data */
  Size max_proc_buf_size_;       /**< Max buffer size on any rank */
  Size max_block_size_;          /**< Max size of a single block */
  std::vector<Index> num_tiles_; /**< Number of tiles along each dimension */
  std::vector<Offset> extents_;  /**< Number of elements along each dimention */
  std::vector<std::vector<Index>>
      index_part_offsets_; /**< Offset (in tile index) partitioned among ranks
                              along each dimension*/
  std::vector<std::vector<Offset>>
      part_offsets_; /**< Offset (in elements) partitioned among ranks along
                        each dimension */
};                   // class Distribution_Dense

class ViewDistribution : public Distribution {
  public:
  using Func = std::function<IndexVector(const IndexVector&)>;
  // Ctors
  ViewDistribution(const Distribution* ref_dist, Func map_func) :
    Distribution(nullptr, ref_dist->get_dist_proc(),
                 DistributionKind::view),
    ref_dist_{ref_dist},
    map_func_{map_func} {}

  // Dtor
  ~ViewDistribution() = default;

  std::pair<Proc, Offset> locate(const IndexVector& blockid) const override {
    const auto& idx_vec = map_func_(blockid);
    return ref_dist_->locate(idx_vec);
  }

    Size buf_size(Proc proc) const override {
      return ref_dist_->buf_size(proc);
    }

    Distribution* clone(const TensorBase* tensor_structure, Proc nproc) const {
        NOT_ALLOWED();
    }

    Size max_proc_buf_size() const override {
      return ref_dist_->max_proc_buf_size();
    }

    Size max_block_size() const override { return ref_dist_->max_block_size(); }

    Size total_size() const override {
      return ref_dist_->total_size();
    }

    size_t compute_hash() const {
      return ref_dist_->compute_hash();
    }
  
  protected:
  const Distribution* ref_dist_;
  Func map_func_;

}; // class ViewDistribution

class UnitTileDistribution: public Distribution {
public:

  // Ctors
  UnitTileDistribution(const TensorBase* tensor_structure, const Distribution* ref_dist):
    Distribution(tensor_structure, ref_dist->get_dist_proc(), DistributionKind::view), ref_dist_{
                                                                                         ref_dist} {
    initialize_tile_information();
  }

  // Dtor
  ~UnitTileDistribution() = default;

  std::pair<Proc, Offset> locate(const IndexVector& blockid) const override {
    const auto& opt_blockid = translate_blockid(blockid);
    auto [proc, offset]     = ref_dist_->locate(opt_blockid);

    Offset translated_offset = translate_offset(blockid, offset);

    return {proc, translated_offset};
  }

  Size buf_size(Proc proc) const override { return ref_dist_->buf_size(proc); }

  Distribution* clone(const TensorBase* tensor_structure, Proc nproc) const { NOT_ALLOWED(); }

  Size max_proc_buf_size() const override { return ref_dist_->max_proc_buf_size(); }

  Size max_block_size() const override { return ref_dist_->max_block_size(); }

  Size total_size() const override { return ref_dist_->total_size(); }

  size_t compute_hash() const { return ref_dist_->compute_hash(); }

  IndexVector translate_blockid(const IndexVector& blockid) const {
    IndexVector translated_blockid = blockid;

    for(size_t i = 0; i < unit_tiled_dims_.size(); i++) {
      translated_blockid[unit_tiled_dims_[i]] = blockid[i] / opt_tile_sizes_[unit_tiled_dims_[i]];
    }

    return translated_blockid;
  }

  Offset translate_offset(const IndexVector& blockid, Offset offset) const {
    auto unit_tiled_tensor = get_tensor_base();
    size_t num_modes = unit_tiled_tensor->num_modes();
    auto unit_dims = unit_tiled_tensor->block_dims(blockid);

    std::vector<Offset> dim_offsets(num_modes);
    if(num_modes > 0) { dim_offsets[num_modes - 1] = 1; }
    for(int i = num_modes - 2; i >= 0; i--) {
      dim_offsets[i] = dim_offsets[i + 1] * unit_dims[i + 1];
    }
    
    size_t local_offset = 0;

    for (size_t i = 0; i < unit_tiled_dims_.size(); i++) {
      local_offset += (blockid[i] % opt_tile_sizes_[unit_tiled_dims_[i]]) * dim_offsets[i].value();
    }
    
    local_offset += offset.value();

    return Offset{local_offset};
  }

  void initialize_tile_information() {
    auto unit_tiled_tensor = get_tensor_base();
    auto opt_tiled_tensor  = ref_dist_->get_tensor_base();

    auto unit_tis_list = unit_tiled_tensor->tiled_index_spaces();
    for(size_t i = 0; i < unit_tis_list.size(); i++) {
      auto tis = unit_tis_list[i];
      if(tis.num_tiles() == tis.index_space().num_indices()) { unit_tiled_dims_.push_back(i); }
    }
    // There should be at least one unit tiled dim
    EXPECTS(unit_tiled_dims_.size() > 0);
    auto check_cond = [&]() -> bool {
      for(size_t i = 0; i < unit_tiled_dims_.size(); i++) {
        if(unit_tiled_dims_[i] != i) return false;
      }
      return true;
    };
    // The unit tiled dims only allowed to be on the leftmost dims
    EXPECTS(check_cond());

    auto opt_tis_list = opt_tiled_tensor->tiled_index_spaces();

    // EXPECTS(opt_tis_list.size() > unit_tiled_dims_.size());

    for(size_t i = 0; i < unit_tiled_dims_.size(); i++) {
      if (opt_tis_list[unit_tiled_dims_[i]].input_tile_sizes().empty())
        opt_tile_sizes_.push_back(opt_tis_list[unit_tiled_dims_[i]].input_tile_size());
      else 
        opt_tile_sizes_.push_back(opt_tis_list[unit_tiled_dims_[i]].input_tile_sizes()[unit_tiled_dims_[i]]);

    }
  }

protected:
  const Distribution* ref_dist_;
  std::vector<size_t>    unit_tiled_dims_;
  std::vector<Tile>      opt_tile_sizes_;


}; // class UnitTileDistribution

//#if 0
template <int N, typename BodyFunc, typename InitFunc, typename UpdateFunc,
          typename... Args>
class LoopY;

template <typename BodyFunc, typename InitFunc, typename UpdateFunc,
          typename... Args>
class LoopY<1, BodyFunc, InitFunc, UpdateFunc, Args...> {
 public:
  LoopY(BodyFunc bfunc, InitFunc ifunc, UpdateFunc ufunc, Index* itr,
        const Range* range, Args&&... args)
      : bfunc_{bfunc},
        ifunc_{ifunc},
        ufunc_{ufunc},
        itr_{itr},
        range_{range},
        args_{args...} {}

  void operator()() {
    for (itr_[0] = range_[0].lo(), (void)std::apply(ifunc_, args_);
         itr_[0] < range_[0].hi(); itr_[0] += range_[0].step()) {
      (void)std::apply(ufunc_, args_);
      bfunc_();
    }
  }
  BodyFunc bfunc_;
  InitFunc ifunc_;
  UpdateFunc ufunc_;
  Index* itr_;
  const Range* range_;
  std::tuple<Args...> args_;
};

//@todo range need not be pointers
//@bug Do these work with 0-d loops
template <typename BodyFunc, typename InitFunc, typename UpdateFunc,
          typename... Args>
class LoopY<2, BodyFunc, InitFunc, UpdateFunc, Args...> {
 public:
  LoopY(BodyFunc bfunc, InitFunc ifunc, UpdateFunc ufunc, Index* itr,
        const Range* range, Args&&... args)
      : bfunc_{bfunc},
        ifunc_{ifunc},
        ufunc_{ufunc},
        itr_{itr},
        range_{range},
        args_{args...},
        args_1_{(args + 1)...} {}

  void operator()() {
    for (itr_[0] = range_[0].lo(), (void)std::apply(ifunc_, args_);
         itr_[0] < range_[0].hi(); itr_[0] += range_[0].step()) {
      (void)std::apply(ufunc_, args_);
      for (itr_[1] = range_[1].lo(), (void)std::apply(ifunc_, args_1_);
           itr_[1] < range_[1].hi(); itr_[1] += range_[1].step()) {
        (void)std::apply(ufunc_, args_1_);
        bfunc_();
      }
    }
  }
  BodyFunc bfunc_;
  InitFunc ifunc_;
  UpdateFunc ufunc_;
  const Range* range_;
  Index* itr_;
  std::tuple<Args...> args_;
  std::tuple<Args...> args_1_;
};

template <typename BodyFunc, typename InitFunc, typename UpdateFunc,
          typename... Args>
class LoopY<3, BodyFunc, InitFunc, UpdateFunc, Args...> {
 public:
  LoopY(BodyFunc bfunc, InitFunc ifunc, UpdateFunc ufunc, Index* itr,
        const Range* range, Args&&... args)
      : bfunc_{bfunc},
        ifunc_{ifunc},
        ufunc_{ufunc},
        itr_{itr},
        range_{range},
        args_{args...},
        args_1_{(args + 1)...},
        args_2_{(args + 2)...} {}

  void operator()() {
    for (itr_[0] = range_[0].lo(), (void)std::apply(ifunc_, args_);
         itr_[0] < range_[0].hi(); itr_[0] += range_[0].step()) {
      (void)std::apply(ufunc_, args_);
      for (itr_[1] = range_[1].lo(), (void)std::apply(ifunc_, args_1_);
           itr_[1] < range_[1].hi(); itr_[1] += range_[1].step()) {
        (void)std::apply(ufunc_, args_1_);
        for (itr_[2] = range_[2].lo(), (void)std::apply(ifunc_, args_2_);
             itr_[2] < range_[2].hi(); itr_[2] += range_[2].step()) {
          (void)std::apply(ufunc_, args_2_);
          bfunc_();
        }
      }
    }
  }
  BodyFunc bfunc_;
  InitFunc ifunc_;
  UpdateFunc ufunc_;
  const Range* range_;
  Index* itr_;
  std::tuple<Args...> args_;
  std::tuple<Args...> args_1_;
  std::tuple<Args...> args_2_;
};

template <typename BodyFunc, typename InitFunc, typename UpdateFunc,
          typename... Args>
class LoopY<4, BodyFunc, InitFunc, UpdateFunc, Args...> {
 public:
  LoopY(BodyFunc bfunc, InitFunc ifunc, UpdateFunc ufunc, Index* itr,
        const Range* range, Args&&... args)
      : bfunc_{bfunc},
        ifunc_{ifunc},
        ufunc_{ufunc},
        itr_{itr},
        range_{range},
        args_{args...},
        args_1_{(args + 1)...},
        args_2_{(args + 2)...},
        args_3_{(args + 3)...} {}

  void operator()() {
    for (itr_[0] = range_[0].lo(), (void)std::apply(ifunc_, args_);
         itr_[0] < range_[0].hi(); itr_[0] += range_[0].step()) {
      (void)std::apply(ufunc_, args_);
      for (itr_[1] = range_[1].lo(), (void)std::apply(ifunc_, args_1_);
           itr_[1] < range_[1].hi(); itr_[1] += range_[1].step()) {
        (void)std::apply(ufunc_, args_1_);
        for (itr_[2] = range_[2].lo(), (void)std::apply(ifunc_, args_2_);
             itr_[2] < range_[2].hi(); itr_[2] += range_[2].step()) {
          (void)std::apply(ufunc_, args_2_);
          for (itr_[3] = range_[3].lo(), (void)std::apply(ifunc_, args_3_);
               itr_[3] < range_[3].hi(); itr_[3] += range_[3].step()) {
            (void)std::apply(ufunc_, args_3_);
            bfunc_();
          }
        }
      }
    }
  }
  BodyFunc bfunc_;
  InitFunc ifunc_;
  UpdateFunc ufunc_;
  const Range* range_;
  Index* itr_;
  std::tuple<Args...> args_;
  std::tuple<Args...> args_1_;
  std::tuple<Args...> args_2_;
  std::tuple<Args...> args_3_;
};

template <typename BodyFunc, typename InitFunc, typename UpdateFunc,
          typename... Args>
auto loop_nest_y_1(BodyFunc&& bfunc, InitFunc&& ifunc, UpdateFunc&& ufunc,
                   Index* itr, const Range* ranges, Args&&... args) {
  return LoopY<1, BodyFunc, InitFunc, UpdateFunc, Args...>(
      std::forward<BodyFunc>(bfunc), std::forward<InitFunc>(ifunc),
      std::forward<UpdateFunc>(ufunc), itr, ranges,
      std::forward<Args>(args)...);
}

template <typename BodyFunc, typename InitFunc, typename UpdateFunc,
          typename... Args>
auto loop_nest_y_2(BodyFunc&& bfunc, InitFunc&& ifunc, UpdateFunc&& ufunc,
                   Index* itr, const Range* ranges, Args&&... args) {
  return LoopY<2, BodyFunc, InitFunc, UpdateFunc, Args...>(
      std::forward<BodyFunc>(bfunc), std::forward<InitFunc>(ifunc),
      std::forward<UpdateFunc>(ufunc), itr, ranges,
      std::forward<Args>(args)...);
}

template <typename BodyFunc, typename InitFunc, typename UpdateFunc,
          typename... Args>
auto loop_nest_y_3(BodyFunc&& bfunc, InitFunc&& ifunc, UpdateFunc&& ufunc,
                   Index* itr, const Range* ranges, Args&&... args) {
  return LoopY<3, BodyFunc, InitFunc, UpdateFunc, Args...>(
      std::forward<BodyFunc>(bfunc), std::forward<InitFunc>(ifunc),
      std::forward<UpdateFunc>(ufunc), itr, ranges,
      std::forward<Args>(args)...);
}

template <typename BodyFunc, typename InitFunc, typename UpdateFunc,
          typename... Args>
auto loop_nest_y_4(BodyFunc&& bfunc, InitFunc&& ifunc, UpdateFunc&& ufunc,
                   Index* itr, const Range* ranges, Args&&... args) {
  return LoopY<4, BodyFunc, InitFunc, UpdateFunc, Args...>(
      std::forward<BodyFunc>(bfunc), std::forward<InitFunc>(ifunc),
      std::forward<UpdateFunc>(ufunc), itr, ranges,
      std::forward<Args>(args)...);
}

template <typename Func, typename... Args>
auto loop_nest_y_1(Func&& func, Index* itr, const Range* ranges,
                   Args&&... args) {
  auto noop = []() {};
  return loop_nest_y_1(std::forward<Func>(func), noop, noop, itr, ranges,
                       std::forward<Args>(args)...);
}

template <typename Func, typename... Args>
auto loop_nest_y_2(Func&& func, Index* itr, const Range* ranges,
                   Args&&... args) {
  auto noop = []() {};
  return loop_nest_y_2(std::forward<Func>(func), noop, noop, itr, ranges,
                       std::forward<Args>(args)...);
}

template <typename Func, typename... Args>
auto loop_nest_y_3(Func&& func, Index* itr, const Range* ranges,
                   Args&&... args) {
  auto noop = []() {};
  return loop_nest_y_3(std::forward<Func>(func), noop, noop, itr, ranges,
                       std::forward<Args>(args)...);
}

template <typename Func, typename... Args>
auto loop_nest_y_4(Func&& func, Index* itr, const Range* ranges,
                   Args&&... args) {
  auto noop = []() {};
  return loop_nest_y_4(std::forward<Func>(func), noop, noop, itr, ranges,
                       std::forward<Args>(args)...);
}

template <typename BodyFunc, typename InitFunc, typename UpdateFunc, typename... Args>
void loop_nest_exec(BodyFunc&& bfunc, InitFunc&& ifunc, UpdateFunc&& ufunc, std::vector<Index>& itr,
                    const std::vector<Range>& ranges, Args&&... args) {
  EXPECTS(itr.size() == ranges.size());
  int N = itr.size();
  if (N == 1) {
    loop_nest_y_1(std::forward<BodyFunc>(bfunc), std::forward<InitFunc>(ifunc),
                  std::forward<UpdateFunc>(ufunc), &itr[0], &ranges[0],
                  std::forward<Args>(args)...)();
  } else if (N == 2) {
    loop_nest_y_2(std::forward<BodyFunc>(bfunc), std::forward<InitFunc>(ifunc),
                  std::forward<UpdateFunc>(ufunc), &itr[0], &ranges[0],
                  std::forward<Args>(args)...)();
  } else if (N == 3) {
    loop_nest_y_3(std::forward<BodyFunc>(bfunc), std::forward<InitFunc>(ifunc),
                  std::forward<UpdateFunc>(ufunc), &itr[0], &ranges[0],
                  std::forward<Args>(args)...)();
  } else if (N == 4) {
    loop_nest_y_4(std::forward<BodyFunc>(bfunc), std::forward<InitFunc>(ifunc),
                  std::forward<UpdateFunc>(ufunc), &itr[0], &ranges[0],
                  std::forward<Args>(args)...)();
  } else {
    NOT_IMPLEMENTED();
  }
}


template <typename BodyFunc, typename... Args>
void loop_nest_exec(BodyFunc&& bfunc, std::vector<Index>& itr,
                    const std::vector<Range>& ranges, Args&&... args) {
  auto noop = []() {};
  loop_nest_exec(std::forward<BodyFunc>(bfunc), noop, noop, itr, ranges,
                 std::forward<Args>(args)...);
}

template <typename BodyFunc, typename InitFunc, typename UpdateFunc,
          typename... Args>
void loop_nest_exec(BodyFunc&& bfunc, InitFunc&& ifunc, UpdateFunc&& ufunc,
                    const IndexLabelVec& ilv, std::vector<Index>& itr,
                    Args&&... args) {
  EXPECTS(internal::is_dense_labels(ilv));  // no dependent labels
  std::vector<Range> ranges;
  for (const TiledIndexLabel& lbl : ilv) {
    ranges.push_back({0, static_cast<tamm::Index>(lbl.tiled_index_space().num_tiles()), 1});
  }
  loop_nest_exec(std::forward<BodyFunc>(bfunc), std::forward<InitFunc>(ifunc),
                 std::forward<UpdateFunc>(ufunc), itr, ranges,
                 std::forward<Args>(args)...);
}

template <typename Func, typename... Args>
void loop_nest_exec(Func&& func, const IndexLabelVec& ilv,
                    std::vector<Index>& itr, Args&&... args) {
  auto noop = []() {};
  loop_nest_exec(func, noop, noop, ilv, itr, std::forward<Args>(args)...);
}

template <typename BodyFunc, typename InitFunc, typename UpdateFunc,
          typename LabeledTensorT, typename... Args>
void loop_nest_exec(BodyFunc&& bfunc, InitFunc&& ifunc, UpdateFunc&& ufunc,
                    const LabeledTensorT& lt, std::vector<Index>& itr,
                    Args&&... args) {
  EXPECTS(!internal::is_slicing(lt));  // dense and no slicing
  loop_nest_exec(std::forward<BodyFunc>(bfunc), std::forward<InitFunc>(ifunc),
                 std::forward<UpdateFunc>(ufunc), lt.labels(), itr,
                 std::forward<Args>(args)...);
}

template <typename Func, typename LabeledTensorT, typename... Args>
void loop_nest_exec(Func&& func, const LabeledTensorT& lt,
                    std::vector<Index>& itr, Args&&... args) {
  auto noop = [](){};
  loop_nest_exec(func, noop, noop, lt, itr, std::forward<Args>(args)...);
}

/*
  [DONE] @todo Implement lhs.tensor().is_dense_distribution()

  [DONE] @todo Allocate buf once (max buf size). Also put needs to works with
  larger buffer sizes

  [DONE] @todo Tensor access local buf directly instead of put/acc

  @todo An efficient put path (only cheap EXPECTS checks) and do put

  [DONE?] @todo Efficient way of determine offset of blockid in proc local buf

  @todo support dense (aka Cartesian) slicing
*/
// template <typename LabeledTensorT, typename T>
// void set_op_execute(const ProcGroup& pg, const LabeledTensorT& lhs, T value,
//                     bool is_assign) {
//   EXPECTS(!is_slicing(lhs));  // dense and no slicing in labels
//   EXPECTS(lhs.tensor().distribution().kind() == DistributionKind::dense); //tensor uses Distribution_Dense
//   //EXPECTS(
//   //    lhs.tensor().execution_context()->pg() ==
//   //        pg);  // tensor allocation proc grid same as op execution proc grid
//   const Distribution_Dense& dd =
//       static_cast<const Distribution_Dense&>(lhs.tensor().distribution());
//   Proc rank = pg.rank();
//   const std::vector<Proc>& grid_rank = dd.proc_rank_to_grid_rank(rank);  
//   const std::vector<Range>& ranges = dd.proc_tile_extents(grid_rank);
//   std::vector<Index> blockid(ranges.size());
//   using LT_eltype = typename LabeledTensorT::element_type;
//   std::vector<LT_eltype> buf(dd.max_block_size(), LT_eltype{value});
//   LT_eltype* lbuf = lhs.tensor().access_local_buf();
//   Offset off = 0;
//   if (is_assign) {
//     loop_nest_exec(
//         ranges,
//         [&]() {
// #if 1
//           auto sz = dd.block_size(blockid);
//           std::copy(buf, buf + sz, &lbuf[off]);
//           off += sz;
// #elif 0
//     // lhs.tensor().put(blockid, buf);
// #else
//           Proc proc;
//           Offset off;
//           std::tie(proc, off) = dd.locate(blockid);
//           std::copy(buf, buf + dd.block_size(blockid), &lbuf[off]);
// #endif
//         },
//         blockid);
//   } else {
//     loop_nest_exec(
//         ranges,
//         [&]() {
// #if 1
//           auto sz = dd.block_size(blockid);
//           for (size_t i = 0; i < dd.block_size(blockid); i++) {
//             lbuf[i] += buf[i];
//           }
//           off += sz;
// #elif 0
//           lhs.tensor().add(blockid, buf);
// #else
//           Proc proc;
//           Offset off;
//           std::tie(proc, off) = dd.locate(blockid);
//           for (size_t i = 0; i < dd.block_size(blockid); i++) {
//             lbuf[off + i] += buf[i];
//           }
// #endif
//         },
//         blockid);
//   }
// }

} // namespace tamm

#endif // TAMM_DISTRIBUTION_HPP_
