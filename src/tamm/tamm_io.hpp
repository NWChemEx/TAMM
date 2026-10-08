#pragma once

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

#if defined(USE_UPCXX)
#include <upcxx/upcxx.hpp>
#endif

#if defined(USE_HDF5)
#include <hdf5.h>
#endif

#include <iomanip>

#include "eigen_includes.hpp"
#include "ga_over_upcxx.hpp"

// #define IO_ISIRREG 1

namespace tamm {

void tamm_terminate(std::string msg);

/**
 * @brief Overload of << operator for printing Tensor blocks
 *
 * @tparam T template type for Tensor element type
 * @param [in] os output stream
 * @param [in] vec vector to be printed
 * @returns the reference to input output stream with vector elements printed
 * out
 */
template<typename T>
std::ostream& operator<<(std::ostream& os, std::vector<T>& vec) {
  os << "[";
  for(auto& x: vec) os << x << ",";
  os << "]"; // << std::endl;
  return os;
}

// From integer type to integer type
template<typename from>
constexpr typename std::enable_if<std::is_integral<from>::value && std::is_integral<int64_t>::value,
                                  int64_t>::type
cd_ncast(const from& value) {
  return static_cast<int64_t>(value & (static_cast<typename std::make_unsigned<from>::type>(-1)));
}

/**
 * @brief convert tamm tensor to N-D GA
 *
 * @tparam TensorType the type of the elements in the tensor
 * @param ec ExecutionContext
 * @param tensor tamm tensor handle
 * @return GA handle
 */
template<typename TensorType>
#if defined(USE_UPCXX)
ga_over_upcxx<TensorType>* tamm_to_ga(ExecutionContext& ec, Tensor<TensorType>& tensor)
#else
int tamm_to_ga(ExecutionContext& ec, Tensor<TensorType>& tensor)
#endif
{
  int                  ndims = tensor.num_modes();
  std::vector<int64_t> dims(ndims, 1), chnks(ndims, -1);
  auto                 tis = tensor.tiled_index_spaces();

  for(int i = 0; i < ndims; ++i) { dims[i] = tis[i].index_space().num_indices(); }

#if defined(USE_UPCXX)
  if(ndims > 4) {
    fprintf(stderr, "Invalid ndims=%d, only support up to 4\n", ndims);
    abort();
  }

  ga_over_upcxx<TensorType>* ga_tens =
    new ga_over_upcxx<TensorType>(ndims, dims.data(), chnks.data(), upcxx::world());
#else
  int ga_pg_default = GA_Pgroup_get_default();
  GA_Pgroup_set_default(ec.pg().ga_pg());

  auto ga_eltype = to_ga_eltype(tensor_element_type<TensorType>());
  int  ga_tens   = NGA_Create64(ga_eltype, ndims, &dims[0], const_cast<char*>("iotemp"), &chnks[0]);
  GA_Pgroup_set_default(ga_pg_default);
#endif

  // convert tamm tensor to GA
  auto tamm_ga_lambda = [&](const IndexVector& bid) {
    const IndexVector blockid = internal::translate_blockid(bid, tensor());

    auto block_dims   = tensor.block_dims(blockid);
    auto block_offset = tensor.block_offsets(blockid);

    const tamm::TAMM_SIZE dsize = tensor.block_size(blockid);

#if defined(USE_UPCXX)
    std::vector<int64_t> lo(4, 0), hi(4, 0);
    std::vector<int64_t> ld(4, 1);
#else
    std::vector<int64_t> lo(ndims), hi(ndims);
    std::vector<int64_t> ld(ndims - 1);
#endif

    for(size_t i = 0; i < ndims; i++) lo[i] = cd_ncast<size_t>(block_offset[i]);
    for(size_t i = 0; i < ndims; i++) hi[i] = cd_ncast<size_t>(block_offset[i] + block_dims[i] - 1);

#if defined(USE_UPCXX)
    for(size_t i = 0; i < ndims; i++) ld[i] = cd_ncast<size_t>(block_dims[i]);
#else
    for(size_t i = 1; i < ndims; i++) ld[i - 1] = cd_ncast<size_t>(block_dims[i]);
#endif

    std::vector<TensorType> sbuf(dsize);
    tensor.get(blockid, sbuf);

#if defined(USE_UPCXX)
    ga_tens->put(lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3], sbuf.data(), ld.data());
#else
    NGA_Put64(ga_tens, &lo[0], &hi[0], &sbuf[0], &ld[0]);
#endif
  };

  block_for(ec, tensor(), tamm_ga_lambda);

  return ga_tens;
}

#if defined(USE_HDF5)
template<typename T>
hid_t get_hdf5_dt() {
  using std::is_same_v;

  if constexpr(is_same_v<int, T>) return H5T_NATIVE_INT;
  if constexpr(is_same_v<int64_t, T>) return H5T_NATIVE_LLONG;
  else if constexpr(is_same_v<float, T>) return H5T_NATIVE_FLOAT;
  else if constexpr(is_same_v<double, T>) return H5T_NATIVE_DOUBLE;
  else if constexpr(is_same_v<std::complex<float>, T>) {
    struct complex_t {
      float re; /*real part*/
      float im; /*imaginary part*/
    };

    hid_t complex_id = H5Tcreate(H5T_COMPOUND, sizeof(complex_t));
    H5Tinsert(complex_id, "real", HOFFSET(complex_t, re), H5T_NATIVE_FLOAT);
    H5Tinsert(complex_id, "imaginary", HOFFSET(complex_t, im), H5T_NATIVE_FLOAT);
    return complex_id;
  }
  else if constexpr(is_same_v<std::complex<double>, T>) {
    struct complex_t {
      double re; /*real part*/
      double im; /*imaginary part*/
    };

    hid_t complex_id = H5Tcreate(H5T_COMPOUND, sizeof(complex_t));
    H5Tinsert(complex_id, "real", HOFFSET(complex_t, re), H5T_NATIVE_DOUBLE);
    H5Tinsert(complex_id, "imaginary", HOFFSET(complex_t, im), H5T_NATIVE_DOUBLE);
    return complex_id;
  }
}

namespace internal {

/// A tensor file is written into this staging file and then committed (renamed) to its final
/// name, so a tensor file at its final name is always complete.
inline std::string staging_filename(const std::string& filename) { return filename + ".tmp"; }

/// Turns off HDF5's automatic error-stack printing for its lifetime; failures are detected and
/// reported through H5Check instead.
class H5ErrorSilencer {
public:
  H5ErrorSilencer() {
    H5Eget_auto2(H5E_DEFAULT, &func_, &data_);
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
  }
  ~H5ErrorSilencer() { H5Eset_auto2(H5E_DEFAULT, func_, data_); }

private:
  H5E_auto2_t func_ = nullptr;
  void*       data_ = nullptr;
};

/// The description of the innermost entry of the current HDF5 error stack, i.e. where the error
/// originated, e.g. "MPI_File_open failed: MPI error string is 'MPI_ERR_BAD_FILE: bad file'".
inline std::string h5_error_cause() {
  std::string cause;
  H5Ewalk2(
    H5E_DEFAULT, H5E_WALK_UPWARD,
    [](unsigned n, const H5E_error2_t* err, void* data) -> herr_t {
      if(n == 0 && err->desc) *static_cast<std::string*>(data) = err->desc;
      return 1; // innermost entry only
    },
    &cause);
  return cause;
}

/// Records whether any HDF5 call on this rank failed while accessing one tensor file. The first
/// failure is reported with the file, the failing call and its cause, as a WARNING for writes
/// (not fatal) or an ERROR for reads (fatal). A collective call fails on every rank of the group
/// accessing the file, so its failure is reported only by that group's root.
struct H5Check {
  std::string filename;
  const char* severity; // "WARNING" or "ERROR"
  bool        is_root;  // root of the group accessing the file
  bool        failed = false;

  template<typename R>
  R operator()(R ret, const char* call, bool collective = true) {
    if(ret < 0 && !failed) {
      failed = true;
      if(is_root || !collective) {
        std::cerr << "[TAMM " << severity << "] " << call << " failed for " << filename << ": "
                  << h5_error_cause() << std::endl;
      }
    }
    return ret;
  }

  /// Records a failure that is not an HDF5 error and that every rank detects alike (e.g. a
  /// mismatched tensor description); reported once, by the root.
  void fail(const std::string& reason) {
    if(failed) return;
    failed = true;
    if(is_root)
      std::cerr << "[TAMM " << severity << "] " << filename << ": " << reason << std::endl;
  }
};

/// Time each rank spends in the phases of writing or reading one tensor file. When profiling, the
/// maximum and average over the ranks doing the I/O are printed, with the number of blocks each
/// rank handled; a large gap between maximum and average points to load imbalance.
class IOPhases {
public:
  using clock = std::chrono::steady_clock;

  /// Adds the time since start to phase name (phases are reported in the order first added).
  void add(const std::string& name, clock::time_point start) {
    const double secs = std::chrono::duration<double>(clock::now() - start).count();
    for(auto& [phase, total]: phases_)
      if(phase == name) {
        total += secs;
        return;
      }
    phases_.emplace_back(name, secs);
  }

  /// Adds n to the per-rank count name (e.g. blocks handled), reported as its min - max over ranks.
  void count(const std::string& name, int64_t n = 1) {
    for(auto& [counter, total]: counts_)
      if(counter == name) {
        total += n;
        return;
      }
    counts_.emplace_back(name, n);
  }

  /// Collective over pg: prints the breakdown on pg's rank 0.
  void report(ProcGroup pg, const std::string& title) const {
    const int           n = static_cast<int>(phases_.size());
    std::vector<double> secs(n), max_secs(n), sum_secs(n);
    for(int i = 0; i < n; i++) secs[i] = phases_[i].second;
    pg.allreduce(secs.data(), max_secs.data(), n, ReduceOp::max);
    pg.allreduce(secs.data(), sum_secs.data(), n, ReduceOp::sum);
    const int            nc = static_cast<int>(counts_.size());
    std::vector<int64_t> counts(nc), min_counts(nc), max_counts(nc);
    for(int i = 0; i < nc; i++) counts[i] = counts_[i].second;
    pg.allreduce(counts.data(), min_counts.data(), nc, ReduceOp::min);
    pg.allreduce(counts.data(), max_counts.data(), nc, ReduceOp::max);
    if(pg.rank() != 0) return;

    const int ranks = pg.size().value();
    std::cout << title << " breakdown over " << ranks << " ranks (max / avg seconds):" << std::endl;
    for(int i = 0; i < n; i++)
      std::cout << "  " << std::left << std::setw(24) << phases_[i].first << std::right
                << std::fixed << std::setprecision(2) << std::setw(10) << max_secs[i] << " / "
                << sum_secs[i] / ranks << std::endl;
    for(int i = 0; i < nc; i++)
      std::cout << "  " << std::left << std::setw(24) << counts_[i].first + " per rank"
                << std::right << std::setw(10) << min_counts[i] << " - " << max_counts[i]
                << std::endl;
  }

private:
  std::vector<std::pair<std::string, double>>  phases_;
  std::vector<std::pair<std::string, int64_t>> counts_;
};

/// Reads a positive integer from environment variable name; 0 if it is not set. Reported once, on
/// world rank 0, so a run's output shows that the default was overridden.
inline int64_t io_env_override(const char* name) {
  const char* raw = std::getenv(name);
  if(raw == nullptr) return 0;
  char*           end = nullptr;
  const long long val = std::strtoll(raw, &end, 10);
  if(end == raw || *end != '\0' || val <= 0)
    tamm_terminate(std::string("[TAMM ERROR] ") + name + " must be a positive integer; got \"" +
                   raw + "\"");
  // if(ProcGroup::world_rank().value() == 0)
  //   std::cout << "[TAMM] " << name << " = " << val << std::endl;
  return val;
}

/// Lustre striping of a new tensor file.
struct Striping {
  int64_t count;      // number of OSTs
  int64_t size_bytes; // stripe size
};

/// Striping for a tensor file of tensor_bytes: 4 MiB stripes and one OST per 4 GiB of data, at
/// least 1 and at most 64. TAMM_IO_STRIPE_COUNT and TAMM_IO_STRIPE_SIZE (in MiB) override them.
inline Striping tensor_file_striping(int64_t tensor_bytes) {
  static const int64_t count_override = io_env_override("TAMM_IO_STRIPE_COUNT");
  static const int64_t size_override  = io_env_override("TAMM_IO_STRIPE_SIZE");
  constexpr int64_t    MiB            = int64_t{1} << 20;
  constexpr int64_t    GiB            = int64_t{1} << 30;

  Striping striping;
  striping.count      = count_override > 0
                          ? count_override
                          : std::clamp<int64_t>((tensor_bytes + 4 * GiB - 1) / (4 * GiB), 1, 64);
  striping.size_bytes = (size_override > 0 ? size_override : 4) * MiB;
  return striping;
}

/// Sets striping hints for creating a tensor file through MPI-IO (honored on Lustre, ignored by
/// other filesystems), and aligns HDF5 objects of at least one stripe to stripe boundaries.
inline void set_striping(MPI_Info info, hid_t fapl, const Striping& striping) {
  MPI_Info_set(info, "striping_factor", std::to_string(striping.count).c_str());
  MPI_Info_set(info, "striping_unit", std::to_string(striping.size_bytes).c_str());
  H5Pset_alignment(fapl, striping.size_bytes, striping.size_bytes);
}

/// Version of the tensor description stored with each tensor file.
constexpr int64_t tensor_file_format = 1;

/// The tensor description stored with a tensor file: everything that determines where each block
/// of the tensor is in the file.
struct TensorDescription {
  std::vector<std::vector<int64_t>> tile_sizes;       // per mode, the size of each tile
  int64_t                           nonzero_blocks{}; // number of non-zero blocks
  uint64_t                          nonzero_hash{};   // FNV-1a of the non-zero block ids
};

/// Describes how tensor is laid out in a tensor file. The non-zero blocks are hashed in file
/// order with a fixed hash function, so the description is the same on every machine.
template<typename TensorType>
TensorDescription describe_tensor(Tensor<TensorType> tensor) {
  TensorDescription desc;
  for(const auto& tis: tensor.tiled_index_spaces()) {
    std::vector<int64_t> sizes;
    for(size_t t = 0; t < tis.num_tiles(); t++) sizes.push_back(tis.tile_size(t));
    desc.tile_sizes.push_back(sizes);
  }

  uint64_t      hash = 14695981039346656037ull; // FNV-1a 64-bit offset basis
  LabelLoopNest loop_nest{tensor().labels()};
  for(const IndexVector& blockid: loop_nest) {
    if(!tensor.is_non_zero(blockid)) continue;
    desc.nonzero_blocks++;
    for(auto id: blockid) {
      const uint64_t v = static_cast<uint64_t>(id);
      for(int byte = 0; byte < 8; byte++) {
        hash ^= (v >> (8 * byte)) & 0xff;
        hash *= 1099511628211ull; // FNV-1a 64-bit prime
      }
    }
  }
  desc.nonzero_hash = hash;
  return desc;
}

inline void write_attribute(hid_t dataset, const std::string& name, hid_t type, const void* data,
                            hsize_t count, H5Check& h5) {
  if(h5.failed) return;
  const hid_t space = count == 0 ? H5Screate(H5S_NULL) : H5Screate_simple(1, &count, nullptr);
  const hid_t attr =
    h5(H5Acreate2(dataset, name.c_str(), type, space, H5P_DEFAULT, H5P_DEFAULT), "H5Acreate");
  if(attr >= 0 && count > 0) h5(H5Awrite(attr, type, data), "H5Awrite");
  if(attr >= 0) H5Aclose(attr);
  H5Sclose(space);
}

/// Stores the tensor description as attributes of the tensor file's dataset.
inline void write_description(hid_t dataset, const TensorDescription& desc, H5Check& h5) {
  write_attribute(dataset, "tamm_format", H5T_NATIVE_INT64, &tensor_file_format, 1, h5);
  for(size_t m = 0; m < desc.tile_sizes.size(); m++)
    write_attribute(dataset, "tile_sizes_" + std::to_string(m), H5T_NATIVE_INT64,
                    desc.tile_sizes[m].data(), desc.tile_sizes[m].size(), h5);
  write_attribute(dataset, "nonzero_blocks", H5T_NATIVE_INT64, &desc.nonzero_blocks, 1, h5);
  write_attribute(dataset, "nonzero_hash", H5T_NATIVE_UINT64, &desc.nonzero_hash, 1, h5);
}

/// Reads an integer attribute (a scalar or a 1-D array) of the tensor file's dataset.
template<typename T>
std::vector<T> read_attribute(hid_t dataset, const std::string& name, hid_t type, H5Check& h5) {
  if(h5.failed) return {};
  const hid_t attr = h5(H5Aopen(dataset, name.c_str(), H5P_DEFAULT), "H5Aopen");
  if(attr < 0) return {};
  const hid_t    space = H5Aget_space(attr);
  const hssize_t count = H5Sget_simple_extent_npoints(space);
  std::vector<T> values(count > 0 ? count : 0);
  if(!values.empty()) h5(H5Aread(attr, type, values.data()), "H5Aread");
  H5Sclose(space);
  H5Aclose(attr);
  return values;
}

inline std::string join(const std::vector<int64_t>& values) {
  std::string out;
  for(auto v: values) out += (out.empty() ? "" : ",") + std::to_string(v);
  return out;
}

/// Checks that the tensor file's dataset holds elements of type hdf5_dt laid out as described by
/// desc; a missing or different description is a failure.
inline void check_description(hid_t dataset, hid_t hdf5_dt, const TensorDescription& desc,
                              H5Check& h5) {
  if(h5.failed) return;
  const hid_t file_dt = h5(H5Dget_type(dataset), "H5Dget_type");
  if(file_dt < 0) return;
  const bool same_type = H5Tequal(file_dt, hdf5_dt) > 0;
  H5Tclose(file_dt);
  if(!same_type) return h5.fail("Element type differs from the tensor's");

  const auto format = read_attribute<int64_t>(dataset, "tamm_format", H5T_NATIVE_INT64, h5);
  if(h5.failed) return;
  if(format.size() != 1 || format[0] != tensor_file_format)
    return h5.fail("Unsupported tensor file format");

  for(size_t m = 0; m < desc.tile_sizes.size(); m++) {
    const auto file_tile_sizes =
      read_attribute<int64_t>(dataset, "tile_sizes_" + std::to_string(m), H5T_NATIVE_INT64, h5);
    if(h5.failed) return;
    const auto& tensor_tile_sizes = desc.tile_sizes[m];
    if(file_tile_sizes != tensor_tile_sizes) {
      const int64_t file_dim_length =
        std::accumulate(file_tile_sizes.begin(), file_tile_sizes.end(), int64_t{0});
      const int64_t tensor_dim_length =
        std::accumulate(tensor_tile_sizes.begin(), tensor_tile_sizes.end(), int64_t{0});
      const std::string dim_label = "dimension " + std::to_string(m);
      if(file_dim_length != tensor_dim_length)
        return h5.fail("Dimension " + std::to_string(m) + " has length " +
                       std::to_string(file_dim_length) + " in the file but " +
                       std::to_string(tensor_dim_length) + " in the tensor");
      auto tiles = [](size_t count) {
        return std::to_string(count) + (count == 1 ? " tile" : " tiles");
      };
      if(file_tile_sizes.size() != tensor_tile_sizes.size())
        return h5.fail("Tiling differs in " + dim_label + ": file has " +
                       tiles(file_tile_sizes.size()) + ", tensor has " +
                       tiles(tensor_tile_sizes.size()));
      return h5.fail("Tiling differs in " + dim_label + ": same number of tiles (" +
                     std::to_string(tensor_tile_sizes.size()) + ") but different tile sizes");
    }
  }
  // A file with more modes than the tensor has a tile_sizes attribute past the tensor's last mode.
  if(H5Aexists(dataset, ("tile_sizes_" + std::to_string(desc.tile_sizes.size())).c_str()) > 0)
    return h5.fail("File has more dimensions than the tensor's " +
                   std::to_string(desc.tile_sizes.size()));

  const auto blocks = read_attribute<int64_t>(dataset, "nonzero_blocks", H5T_NATIVE_INT64, h5);
  const auto hash   = read_attribute<uint64_t>(dataset, "nonzero_hash", H5T_NATIVE_UINT64, h5);
  if(h5.failed) return;
  if(blocks.size() != 1 || hash.size() != 1 || blocks[0] != desc.nonzero_blocks ||
     hash[0] != desc.nonzero_hash)
    return h5.fail("Non-zero block structure differs (file: " +
                   (blocks.empty() ? std::string("?") : std::to_string(blocks[0])) +
                   " blocks, tensor: " + std::to_string(desc.nonzero_blocks) + " blocks)");
}

/// Commits a fully written staging file by renaming it over its tensor file, or (when commit is
/// false) discards it so the previous tensor file is kept. Returns whether the tensor file was
/// replaced. A failed rename is reported but never fatal.
inline bool finish_staging_file(const std::string& filename, bool commit) {
  namespace fs              = std::filesystem;
  const std::string staging = staging_filename(filename);
  std::error_code   ec;
  if(commit) {
    fs::rename(staging, filename, ec);
    if(!ec) return true;
    std::cerr << "[TAMM WARNING] rename " << staging << " -> " << filename
              << " failed: " << ec.message() << std::endl;
  }
  if(fs::is_regular_file(staging, ec)) fs::remove(staging, ec);
  return false;
}

/// Formats a titled list of files, one file per line; empty if there are no files.
inline std::string file_list(const std::string& title, const std::vector<std::string>& files) {
  if(files.empty()) return "";
  std::string list = "\n  " + title + ":";
  for(const auto& f: files) list += "\n    " + f;
  return list;
}

/// Describes what an uncommitted group write left in place: which tensor files still hold their
/// previous version, and which never had one.
inline std::string kept_versions(const std::vector<std::string>& filenames) {
  std::vector<std::string> kept, none;
  for(const auto& f: filenames) (std::filesystem::exists(f) ? kept : none).push_back(f);
  return file_list("previous version kept", kept) + file_list("no previous version", none);
}

/// Whether each block of tensor is stored contiguously on one rank, so that the rank owning a
/// block can write and read it straight from its local memory. Dense (N-D Global Array), view and
/// lambda tensors are not.
template<typename TensorType>
bool is_block_distributed(const Tensor<TensorType>& tensor) {
  using Kind      = TensorBase::TensorKind;
  const auto kind = tensor.kind();
  return kind == Kind::normal || kind == Kind::spin || kind == Kind::block_sparse;
}

/// A non-zero block of a tensor and where it is in the tensor file.
struct FileBlock {
  IndexVector id;
  hsize_t     file_offset;  // in elements
  hsize_t     size;         // in elements
  size_t      local_offset; // in elements, into this rank's local buffer (local blocks only)
};

/// How the ranks writing or reading a tensor file split its blocks. A block owned by one of these
/// ranks is local: its owner writes or reads it straight from or into its local memory. All other
/// blocks (owned by ranks not doing the I/O, or of a tensor whose blocks are not each stored on one
/// rank) are shared: handed out among the I/O ranks and moved with get/put.
struct BlockPlan {
  std::vector<FileBlock> local;
  std::vector<FileBlock> shared; // the same list on every I/O rank
};

/// Plans the blocks of tensor for the ranks of io_pg. Blocks are laid out in the file one after
/// another in loop-nest order, skipping zero blocks. Collective over io_pg.
template<typename TensorType>
BlockPlan plan_blocks(Tensor<TensorType> tensor, ProcGroup io_pg, bool use_local) {
  // Which ranks of the tensor's process group are doing the I/O.
  std::vector<char> is_io_rank;
  int               my_rank = -1;
  if(use_local) {
    ProcGroup tensor_pg = tensor.execution_context()->pg();
    my_rank             = tensor_pg.rank().value();
    std::vector<int> io_ranks(io_pg.size().value());
    io_pg.allgather(&my_rank, io_ranks.data());
    is_io_rank.assign(tensor_pg.size().value(), 0);
    for(int r: io_ranks) is_io_rank[r] = 1;
  }

  BlockPlan     plan;
  hsize_t       file_offset = 0;
  auto          ltensor     = tensor();
  LabelLoopNest loop_nest{ltensor.labels()};
  for(const IndexVector& bid: loop_nest) {
    const IndexVector blockid = translate_blockid(bid, ltensor);
    if(!tensor.is_non_zero(blockid)) continue;
    const hsize_t size = tensor.block_size(blockid);
    if(use_local) {
      auto [proc, local_offset] = tensor.distribution().locate(blockid);
      if(!is_io_rank[proc.value()]) plan.shared.push_back({blockid, file_offset, size, 0});
      else if(proc.value() == my_rank)
        plan.local.push_back(
          {blockid, file_offset, size, static_cast<size_t>(local_offset.value())});
    }
    else plan.shared.push_back({blockid, file_offset, size, 0});
    file_offset += size;
  }
  return plan;
}

/// Calls func for each block in blocks (the same list on every rank of pg), handing the blocks out
/// dynamically among the ranks of pg. Collective over pg.
template<typename Func>
void for_each_shared(ProcGroup pg, const std::vector<FileBlock>& blocks, Func&& func) {
  if(blocks.empty()) return;
  AtomicCounterGA counter(pg, 1);
  counter.allocate(0);
  for(int64_t next = counter.fetch_add(0, 1); next < static_cast<int64_t>(blocks.size());
      next         = counter.fetch_add(0, 1))
    func(blocks[next]);
  counter.deallocate();
  pg.barrier();
}

/// Selects block in the tensor file's dataspace and returns a matching memory dataspace.
inline hid_t select_block(hid_t file_space, const FileBlock& block) {
  H5Sselect_hyperslab(file_space, H5S_SELECT_SET, &block.file_offset, nullptr, &block.size,
                      nullptr);
  return H5Screate_simple(1, &block.size, nullptr);
}

/// Writes the blocks of tensor planned for this rank: local blocks from local memory, then its
/// share of the shared blocks, fetched with fetch(block, buffer). Collective over io_pg.
template<typename TensorType, typename Fetch>
void write_blocks(ProcGroup io_pg, Tensor<TensorType> tensor, const BlockPlan& plan, hid_t dataset,
                  hid_t file_space, hid_t xfer, hid_t dt, H5Check& h5, IOPhases& phases,
                  Fetch&& fetch) {
  using io_clock = IOPhases::clock;
  auto write     = [&](const FileBlock& block, const TensorType* data) {
    if(h5.failed) return;
    const hid_t mem_space = select_block(file_space, block);
    h5(H5Dwrite(dataset, dt, mem_space, file_space, xfer, data), "H5Dwrite", false);
    H5Sclose(mem_space);
  };

  auto phase_t = io_clock::now();
  if(!plan.local.empty()) {
    const TensorType* local = tensor.access_local_buf();
    for(const auto& block: plan.local) write(block, local + block.local_offset);
  }
  phases.add("local block writes", phase_t);
  phases.count("local blocks", plan.local.size());

  phase_t = io_clock::now();
  std::vector<TensorType> buffer;
  for_each_shared(io_pg, plan.shared, [&](const FileBlock& block) {
    buffer.resize(block.size);
    auto block_t = io_clock::now();
    fetch(block, buffer.data());
    phases.add("gather shared (get)", block_t);
    block_t = io_clock::now();
    write(block, buffer.data());
    phases.add("shared block writes", block_t);
    phases.count("shared blocks");
  });
  if(!plan.shared.empty()) phases.add("shared blocks (total)", phase_t);
}

/// Reads the blocks of tensor planned for this rank: local blocks straight into local memory, then
/// its share of the shared blocks, stored with store(block, buffer). Collective over io_pg.
template<typename TensorType, typename Store>
void read_blocks(ProcGroup io_pg, Tensor<TensorType> tensor, const BlockPlan& plan, hid_t dataset,
                 hid_t file_space, hid_t xfer, hid_t dt, H5Check& h5, IOPhases& phases,
                 Store&& store) {
  using io_clock = IOPhases::clock;
  auto read      = [&](const FileBlock& block, TensorType* data) {
    if(h5.failed) return false;
    const hid_t mem_space = select_block(file_space, block);
    const bool ok = h5(H5Dread(dataset, dt, mem_space, file_space, xfer, data), "H5Dread", false) >=
                    0;
    H5Sclose(mem_space);
    return ok;
  };

  auto phase_t = io_clock::now();
  if(!plan.local.empty()) {
    TensorType* local = tensor.access_local_buf();
    for(const auto& block: plan.local) read(block, local + block.local_offset);
  }
  phases.add("local block reads", phase_t);
  phases.count("local blocks", plan.local.size());

  phase_t = io_clock::now();
  std::vector<TensorType> buffer;
  for_each_shared(io_pg, plan.shared, [&](const FileBlock& block) {
    buffer.resize(block.size);
    auto       block_t = io_clock::now();
    const bool ok      = read(block, buffer.data());
    phases.add("shared block reads", block_t);
    block_t = io_clock::now();
    if(ok) store(block, buffer.data());
    phases.add("scatter shared (put)", block_t);
    phases.count("shared blocks");
  });
  if(!plan.shared.empty()) phases.add("shared blocks (total)", phase_t);
}

/// The lo/hi/ld arguments of a Global Array patch access for block of tensor, in the N-D Global
/// Array that holds the whole tensor.
template<typename TensorType>
void ga_block_patch(Tensor<TensorType> tensor, const IndexVector& blockid, std::vector<int64_t>& lo,
                    std::vector<int64_t>& hi, std::vector<int64_t>& ld) {
  const auto   block_dims   = tensor.block_dims(blockid);
  const auto   block_offset = tensor.block_offsets(blockid);
  const size_t ndims        = block_dims.size();
  lo.resize(ndims);
  hi.resize(ndims);
  ld.resize(ndims - 1);
  for(size_t i = 0; i < ndims; i++) lo[i] = cd_ncast<size_t>(block_offset[i]);
  for(size_t i = 0; i < ndims; i++) hi[i] = cd_ncast<size_t>(block_offset[i] + block_dims[i] - 1);
  for(size_t i = 1; i < ndims; i++) ld[i - 1] = cd_ncast<size_t>(block_dims[i]);
}

/// Number of elements of the tensor file's dataset: the product of the tensor's dimensions.
template<typename TensorType>
hsize_t tensor_file_elements(const Tensor<TensorType>& tensor) {
  hsize_t elements = 1;
  for(const auto& tis: tensor.tiled_index_spaces()) elements *= tis.index_space().num_indices();
  return elements;
}

/// GiB of tensor data per node used to size the process group writing or reading a tensor file:
/// TAMM_IO_GIB_PER_NODE, default 5.
inline int64_t io_gib_per_node() {
  static const int64_t gib = io_env_override("TAMM_IO_GIB_PER_NODE");
  return gib > 0 ? gib : 5;
}

/// Whether each tensor file is handled by all of the calling process group, one file after another:
/// TAMM_IO_GROUPS=all. The default, TAMM_IO_GROUPS=size, sizes the I/O groups from the data.
inline bool io_groups_all() {
  static const bool all = [] {
    const char* raw = std::getenv("TAMM_IO_GROUPS");
    if(raw == nullptr || *raw == '\0' || std::string(raw) == "size") return false;
    if(std::string(raw) == "all") return true;
    tamm_terminate(std::string("[TAMM ERROR] TAMM_IO_GROUPS must be \"size\" or \"all\"; got \"") +
                   raw + "\"");
    return false;
  }();
  return all;
}

/// How the nodes of the calling process group are split into I/O groups, each writing or reading
/// tensor files. Groups are made of whole nodes, taken consecutively from the first node.
struct IOAllocation {
  std::vector<int>    group_nodes; // number of nodes of each I/O group
  std::vector<size_t> order;       // tensor indices, largest first
  bool
    dynamic; // tensors are handed out to free groups in order; otherwise group j handles order[j]
};

/// Splits nnodes among tensor files of the given sizes. Each tensor ideally gets one node per
/// io_gib_per_node() GiB. If all ideal groups fit, every tensor gets its ideal group and all are
/// handled at once; if not, and there are no more tensors than nodes, the groups are scaled down in
/// proportion to the tensor sizes (at least one node each); with more tensors than nodes, every
/// node is a group and tensors are handed out largest first. With TAMM_IO_GROUPS=all, all nodes
/// form one group that handles the tensors one after another, largest first.
inline IOAllocation allocate_io_groups(const std::vector<int64_t>& bytes, int nnodes) {
  const int    ntensors = static_cast<int>(bytes.size());
  IOAllocation alloc;
  alloc.order.resize(ntensors);
  std::iota(alloc.order.begin(), alloc.order.end(), size_t{0});
  std::stable_sort(alloc.order.begin(), alloc.order.end(),
                   [&](size_t a, size_t b) { return bytes[a] > bytes[b]; });

  if(io_groups_all()) {
    alloc.group_nodes = {nnodes};
    alloc.dynamic     = true;
    return alloc;
  }

  if(ntensors > nnodes) {
    alloc.group_nodes.assign(nnodes, 1);
    alloc.dynamic = true;
    return alloc;
  }

  const int64_t    gib = int64_t{1} << 30;
  std::vector<int> ideal(ntensors);
  int64_t          total = 0;
  for(int j = 0; j < ntensors; j++) {
    const int64_t b = bytes[alloc.order[j]];
    ideal[j]        = static_cast<int>(std::clamp<int64_t>(
      (b + io_gib_per_node() * gib - 1) / (io_gib_per_node() * gib), 1, nnodes));
    total += ideal[j];
  }
  alloc.dynamic = false;
  if(total <= nnodes) {
    alloc.group_nodes = ideal;
    return alloc;
  }

  // Scale down in proportion to size, at least one node each, then use any nodes left over.
  std::vector<int> nodes(ntensors);
  int              used = 0;
  for(int j = 0; j < ntensors; j++) {
    nodes[j] = std::max(1, static_cast<int>(static_cast<int64_t>(ideal[j]) * nnodes / total));
    used += nodes[j];
  }
  for(int j = 0; used > nnodes; j = (j + 1) % ntensors) // too many after the one-node minimum
    if(nodes[j] > 1) {
      nodes[j]--;
      used--;
    }
  for(int j = 0; used < nnodes; j = (j + 1) % ntensors) // nodes left over go to the largest first
    if(nodes[j] < ideal[j]) {
      nodes[j]++;
      used++;
    }
    else if(std::equal(nodes.begin(), nodes.end(), ideal.begin())) break;
  alloc.group_nodes = nodes;
  return alloc;
}

/// Writes tensor to the staging file of filename. Collective over io_pg, the ranks writing this
/// file. Returns whether any of them failed (the same on every rank of io_pg).
template<typename TensorType>
bool write_file(ProcGroup io_pg, Tensor<TensorType> tensor, const std::string& filename,
                bool profile) {
  using io_clock = IOPhases::clock;
  IOPhases phases;

  H5ErrorSilencer   h5_silencer;
  H5Check           h5{filename, "WARNING", io_pg.rank() == 0};
  const std::string staging       = staging_filename(filename);
  const hid_t       hdf5_dt       = get_hdf5_dt<TensorType>();
  const hsize_t     file_elements = tensor_file_elements(tensor);

  auto     phase_t = io_clock::now();
  MPI_Info info;
  MPI_Info_create(&info);
  const hid_t fapl     = H5Pcreate(H5P_FILE_ACCESS);
  const auto  striping = tensor_file_striping(file_elements * sizeof(TensorType));
  set_striping(info, fapl, striping);
  H5Pset_fapl_mpio(fapl, io_pg.comm(), info);
  const hid_t file = h5(H5Fcreate(staging.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl), "H5Fcreate");
  H5Pclose(fapl);
  MPI_Info_free(&info);

  const hid_t file_space = H5Screate_simple(1, &file_elements, nullptr);
  const hid_t dataset =
    h5(H5Dcreate(file, "tensor", hdf5_dt, file_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
       "H5Dcreate");
  write_description(dataset, describe_tensor(tensor), h5);
  const hid_t xfer = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(xfer, H5FD_MPIO_INDEPENDENT);
  phases.add("create file", phase_t);

  phase_t         = io_clock::now();
  const auto plan = plan_blocks(tensor, io_pg, is_block_distributed(tensor));
  phases.add("block plan", phase_t);
  write_blocks(io_pg, tensor, plan, dataset, file_space, xfer, hdf5_dt, h5, phases,
               [&](const FileBlock& block, TensorType* buffer) {
                 tensor.get(block.id, span<TensorType>{buffer, block.size});
               });

  phase_t = io_clock::now();
  H5Pclose(xfer);
  H5Dclose(dataset);
  H5Sclose(file_space);
  h5(H5Fclose(file), "H5Fclose");
  phases.add("close file", phase_t);

  if(profile) {
    if(io_pg.rank() == 0)
      std::cout << std::endl
                << filename << ": " << std::fixed << std::setprecision(2)
                << file_elements * sizeof(TensorType) / (1024 * 1024 * 1024.0) << " GiB, "
                << io_pg.size().value() << " ranks, striping " << striping.count << " OSTs x "
                << (striping.size_bytes >> 20) << " MiB" << std::endl;
    phases.report(io_pg, "write_to_disk " + filename);
  }
  const int failed = h5.failed ? 1 : 0;
  return io_pg.allreduce(&failed, ReduceOp::max) != 0;
}

/// Reads filename into tensor. Collective over io_pg, the ranks reading this file. Returns
/// whether any of them failed (the same on every rank of io_pg).
template<typename TensorType>
bool read_file(ProcGroup io_pg, Tensor<TensorType> tensor, const std::string& filename,
               bool profile) {
  using io_clock = IOPhases::clock;
  IOPhases phases;

  H5ErrorSilencer h5_silencer;
  H5Check         h5{filename, "ERROR", io_pg.rank() == 0};
  const hid_t     hdf5_dt = get_hdf5_dt<TensorType>();

  auto     phase_t = io_clock::now();
  MPI_Info info;
  MPI_Info_create(&info);
  const hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(fapl, io_pg.comm(), info);
  const hid_t file = h5(H5Fopen(filename.c_str(), H5F_ACC_RDONLY, fapl), "H5Fopen");
  H5Pclose(fapl);
  MPI_Info_free(&info);

  const hid_t dataset = h5(H5Dopen(file, "tensor", H5P_DEFAULT), "H5Dopen");
  check_description(dataset, hdf5_dt, describe_tensor(tensor), h5);
  const hid_t file_space = H5Dget_space(dataset);
  const hid_t xfer       = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(xfer, H5FD_MPIO_INDEPENDENT);
  phases.add("open file + check", phase_t);

  phase_t         = io_clock::now();
  const auto plan = plan_blocks(tensor, io_pg, is_block_distributed(tensor));
  phases.add("block plan", phase_t);
  read_blocks(io_pg, tensor, plan, dataset, file_space, xfer, hdf5_dt, h5, phases,
              [&](const FileBlock& block, TensorType* buffer) {
                tensor.put(block.id, span<TensorType>{buffer, block.size});
              });

  phase_t = io_clock::now();
  H5Pclose(xfer);
  H5Sclose(file_space);
  H5Dclose(dataset);
  h5(H5Fclose(file), "H5Fclose");
  phases.add("close file", phase_t);

  if(profile) {
    if(io_pg.rank() == 0)
      std::cout << std::endl << filename << ": " << io_pg.size().value() << " ranks" << std::endl;
    phases.report(io_pg, "read_from_disk " + filename);
  }
  const int failed = h5.failed ? 1 : 0;
  return io_pg.allreduce(&failed, ReduceOp::max) != 0;
}

/// Runs io(io_pg, i) for every tensor file i, on I/O groups of whole nodes of pg allocated by
/// allocate_io_groups. Collective over pg. Returns, for every file, whether its I/O failed (the
/// same on every rank of pg), and marks the files whose I/O group root this rank is.
template<typename IO>
std::vector<int> run_io_groups(ExecutionContext& ec, const std::vector<int64_t>& bytes,
                               bool profile, const std::string& what, std::vector<char>& root_of,
                               IO&& io) {
  ProcGroup    pg       = ec.pg();
  const int    rank     = pg.rank().value();
  const int    nranks   = pg.size().value();
  const int    ppn      = std::max(1, ec.ppn());
  const int    nnodes   = (nranks + ppn - 1) / ppn;
  const size_t nfiles   = bytes.size();
  const auto   alloc    = allocate_io_groups(bytes, nnodes);
  const int    my_node  = rank / ppn;
  int          my_group = -1;
  for(int j = 0, first = 0; j < static_cast<int>(alloc.group_nodes.size()); j++) {
    if(my_node >= first && my_node < first + alloc.group_nodes[j]) my_group = j;
    first += alloc.group_nodes[j];
  }

  if(profile && rank == 0) {
    std::cout << what << ": " << nfiles << " tensor file(s), " << alloc.group_nodes.size()
              << " I/O group(s) of nodes";
    for(auto n: alloc.group_nodes) std::cout << " " << n;
    std::cout << " (" << ppn << " ranks per node), "
              << (alloc.dynamic ? "files handed out largest first" : "one file per group")
              << (io_groups_all() ? " (TAMM_IO_GROUPS=all)" : "") << std::endl;
  }

  std::vector<int> failed(nfiles, 0);
  root_of.assign(nfiles, 0);
  AtomicCounterGA* counter = nullptr;
  if(alloc.dynamic) {
    counter = new AtomicCounterGA(pg, 1);
    counter->allocate(0);
  }

  MPI_Comm io_comm;
  MPI_Comm_split(pg.comm(), my_group >= 0 ? my_group : MPI_UNDEFINED, rank, &io_comm);
  if(io_comm != MPI_COMM_NULL) {
    ProcGroup io_pg  = ProcGroup::create_coll(io_comm);
    auto      handle = [&](size_t i) {
      if(io(io_pg, i)) failed[i] = 1;
      if(io_pg.rank() == 0) root_of[i] = 1;
    };
    if(!alloc.dynamic) handle(alloc.order[my_group]);
    else
      while(true) {
        int64_t next = 0;
        if(io_pg.rank() == 0) next = counter->fetch_add(0, 1);
        io_pg.broadcast(&next, 0);
        if(next >= static_cast<int64_t>(nfiles)) break;
        handle(alloc.order[next]);
      }
    io_pg.destroy_coll();
    MPI_Comm_free(&io_comm);
  }
  if(counter) {
    counter->deallocate();
    delete counter;
  }

  std::vector<int> any_failed(nfiles, 0);
  pg.allreduce(failed.data(), any_failed.data(), static_cast<int>(nfiles), ReduceOp::max);
  return any_failed;
}

} // namespace internal
#endif

/**
 * @brief Writes tensors to tensor files using HDF5, in parallel.
 *
 * Collective over ec, the process group doing the I/O; the tensors may be allocated on ec or on a
 * larger group. ec's nodes are split into I/O groups (see internal::allocate_io_groups), each
 * writing one or more files; within a group, every rank writes the blocks it owns from its local
 * memory, and blocks owned by other ranks are fetched. The files are committed all-or-nothing: if
 * any file fails to write, none is replaced. A failed write is reported but not fatal.
 *
 * @param ec        process group doing the I/O
 * @param tensors   tensors to write
 * @param filenames tensor file of each tensor
 * @param profile   print the I/O groups and a per-file timing breakdown
 */
template<typename TensorType>
void write_to_disk(ExecutionContext& ec, std::vector<Tensor<TensorType>> tensors,
                   std::vector<std::string> filenames, bool profile = false) {
  EXPECTS(tensors.size() == filenames.size());
#if !defined(USE_HDF5)
  tamm_terminate("HDF5 is not enabled. Please rebuild TAMM with HDF5 support");
#else
  if(tensors.empty()) return;
  const auto io_t1 = std::chrono::steady_clock::now();
  ProcGroup  pg    = ec.pg();
  const int  rank  = pg.rank().value();

  std::vector<int64_t> bytes(tensors.size());
  for(size_t i = 0; i < tensors.size(); i++)
    bytes[i] = internal::tensor_file_elements(tensors[i]) * sizeof(TensorType);

  std::vector<char> root_of;
  const auto        failed = internal::run_io_groups(
    ec, bytes, profile, "write_to_disk", root_of, [&](ProcGroup io_pg, size_t i) {
      return internal::write_file(io_pg, tensors[i], filenames[i], profile);
    });

  // Commit all files only if none failed, so the files always come from the same call.
  std::vector<std::string> failed_files;
  for(size_t i = 0; i < tensors.size(); i++)
    if(failed[i]) failed_files.push_back(filenames[i]);
  const bool commit = failed_files.empty();
  for(size_t i = 0; i < tensors.size(); i++)
    if(root_of[i]) internal::finish_staging_file(filenames[i], commit);
  if(!commit && rank == 0) {
    if(tensors.size() == 1)
      std::cerr << "[TAMM WARNING] write_to_disk: " << filenames[0] << " not committed; "
                << (std::filesystem::exists(filenames[0]) ? "previous version kept"
                                                          : "no previous version exists")
                << std::endl;
    else
      std::cerr << "[TAMM WARNING] write_to_disk: no tensor file in the group was committed"
                << internal::file_list("failed to write", failed_files)
                << internal::kept_versions(filenames) << std::endl;
  }
  pg.barrier();

  if(profile && rank == 0)
    std::cout << "Time for writing " << tensors.size() << " tensor file(s) to disk: "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - io_t1).count()
              << " secs" << std::endl;
#endif
}

/// Writes one tensor to a tensor file; see the overload for a list of tensors.
template<typename TensorType>
void write_to_disk(ExecutionContext& ec, Tensor<TensorType> tensor, const std::string& filename,
                   bool profile = false) {
  write_to_disk(ec, std::vector<Tensor<TensorType>>{tensor}, std::vector<std::string>{filename},
                profile);
}

/**
 * @brief convert N-D GA to a tamm tensor
 *
 * @tparam TensorType the type of the elements in the tensor
 * @param ec ExecutionContext
 * @param tensor tamm tensor handle
 * @param ga_tens GA handle
 */
template<typename TensorType>
void ga_to_tamm(ExecutionContext& ec, Tensor<TensorType>& tensor,
#if defined(USE_UPCXX)
                ga_over_upcxx<TensorType>* ga_tens)
#else
                int ga_tens)
#endif
{

  size_t ndims = tensor.num_modes();

  // convert ga to tamm tensor
  auto ga_tamm_lambda = [&](const IndexVector& bid) {
    const IndexVector blockid = internal::translate_blockid(bid, tensor());

    auto block_dims   = tensor.block_dims(blockid);
    auto block_offset = tensor.block_offsets(blockid);

    const tamm::TAMM_SIZE dsize = tensor.block_size(blockid);

#if defined(USE_UPCXX)
    std::vector<int64_t> lo(4, 0), hi(4, 0);
    std::vector<int64_t> ld(4, 1);
#else
    std::vector<int64_t> lo(ndims), hi(ndims);
    std::vector<int64_t> ld(ndims - 1);
#endif

    for(size_t i = 0; i < ndims; i++) lo[i] = cd_ncast<size_t>(block_offset[i]);
    for(size_t i = 0; i < ndims; i++) hi[i] = cd_ncast<size_t>(block_offset[i] + block_dims[i] - 1);

#if defined(USE_UPCXX)
    for(size_t i = 0; i < ndims; i++) ld[i] = cd_ncast<size_t>(block_dims[i]);
#else
    for(size_t i = 1; i < ndims; i++) ld[i - 1] = cd_ncast<size_t>(block_dims[i]);
#endif

    std::vector<TensorType> sbuf(dsize);
#if defined(USE_UPCXX)
    ga_tens->get(lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3], sbuf.data(), ld.data());
#else
    NGA_Get64(ga_tens, &lo[0], &hi[0], &sbuf[0], &ld[0]);
#endif

    tensor.put(blockid, sbuf);
  };

  block_for(ec, tensor(), ga_tamm_lambda);
}

/**
 * @brief Reads tensors from tensor files using HDF5, in parallel.
 *
 * Collective over ec, the process group doing the I/O; the tensors may be allocated on ec or on a
 * larger group, and on a different number of ranks than when they were written. Each tensor
 * file's tensor description must match the tensor read into. A failed read terminates the
 * program, reporting every tensor file that could not be read.
 *
 * @param ec        process group doing the I/O
 * @param tensors   tensors to read into
 * @param filenames tensor file of each tensor
 * @param profile   print the I/O groups and a per-file timing breakdown
 */
template<typename TensorType>
void read_from_disk(ExecutionContext& ec, std::vector<Tensor<TensorType>> tensors,
                    std::vector<std::string> filenames, bool profile = false) {
  EXPECTS(tensors.size() == filenames.size());
#if !defined(USE_HDF5)
  tamm_terminate("HDF5 is not enabled. Please rebuild TAMM with HDF5 support");
#else
  if(tensors.empty()) return;
  const auto io_t1 = std::chrono::steady_clock::now();
  ProcGroup  pg    = ec.pg();
  const int  rank  = pg.rank().value();

  std::vector<int64_t> bytes(tensors.size());
  for(size_t i = 0; i < tensors.size(); i++)
    bytes[i] = internal::tensor_file_elements(tensors[i]) * sizeof(TensorType);

  std::vector<char> root_of;
  const auto        failed = internal::run_io_groups(
    ec, bytes, profile, "read_from_disk", root_of, [&](ProcGroup io_pg, size_t i) {
      return internal::read_file(io_pg, tensors[i], filenames[i], profile);
    });

  std::vector<std::string> failed_files;
  for(size_t i = 0; i < tensors.size(); i++)
    if(failed[i]) failed_files.push_back(filenames[i]);
  if(failed_files.size() == 1 && tensors.size() == 1)
    tamm_terminate("read_from_disk: failed to read tensor file: " + failed_files[0]);
  if(!failed_files.empty())
    tamm_terminate("read_from_disk: one or more tensor files could not be read" +
                   internal::file_list("failed to read", failed_files) + "\n");

  if(profile && rank == 0)
    std::cout << "Time for reading " << tensors.size() << " tensor file(s) from disk: "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - io_t1).count()
              << " secs" << std::endl;
#endif
}

/// Reads one tensor from a tensor file; see the overload for a list of tensors.
template<typename TensorType>
void read_from_disk(ExecutionContext& ec, Tensor<TensorType> tensor, const std::string& filename,
                    bool profile = false) {
  read_from_disk(ec, std::vector<Tensor<TensorType>>{tensor}, std::vector<std::string>{filename},
                 profile);
}

template<typename T>
void write_to_disk_hdf5(
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> eigen_tensor,
  std::string filename, bool write1D = false) {
#if !defined(USE_HDF5)
  tamm_terminate("HDF5 is not enabled. Please rebuild TAMM with HDF5 support");
#else
  std::string outputfile = filename + ".data";
  hid_t       file_id    = H5Fcreate(outputfile.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  T*    buf = eigen_tensor.data();
  hid_t dataspace_id;

  std::vector<hsize_t> dims(2);
  dims[0]      = eigen_tensor.rows();
  dims[1]      = eigen_tensor.cols();
  int rank     = 2;
  dataspace_id = H5Screate_simple(rank, dims.data(), NULL);

  hid_t dataset_id = H5Dcreate(file_id, "data", get_hdf5_dt<T>(), dataspace_id, H5P_DEFAULT,
                               H5P_DEFAULT, H5P_DEFAULT);

  /* herr_t status = */ H5Dwrite(dataset_id, get_hdf5_dt<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);

  /* Create and write attribute information - reduced dims */
  std::vector<int> reduced_dims{static_cast<int>(dims[0]), static_cast<int>(dims[1])};
  hsize_t          attr_size      = reduced_dims.size();
  auto             attr_dataspace = H5Screate_simple(1, &attr_size, NULL);
  auto attr_dataset = H5Dcreate(file_id, "rdims", H5T_NATIVE_INT, attr_dataspace, H5P_DEFAULT,
                                H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(attr_dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, reduced_dims.data());
  H5Dclose(attr_dataset);
  H5Sclose(attr_dataspace);

  H5Dclose(dataset_id);
  H5Sclose(dataspace_id);
  H5Fclose(file_id);
#endif
}

template<typename T, int N>
void write_to_disk_hdf5(Eigen::Tensor<T, N, Eigen::RowMajor> eigen_tensor, std::string filename,
                        bool write1D = false) {
#if !defined(USE_HDF5)
  tamm_terminate("HDF5 is not enabled. Please rebuild TAMM with HDF5 support");
#else
  std::string outputfile = filename + ".data";
  hid_t       file_id    = H5Fcreate(outputfile.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  T* buf = eigen_tensor.data();

  hid_t dataspace_id;
  auto  dims = eigen_tensor.dimensions();

  if(write1D) {
    hsize_t total_size = 1;
    for(int i = 0; i < N; i++) { total_size *= dims[i]; }
    dataspace_id = H5Screate_simple(1, &total_size, NULL);
  }
  else {
    std::vector<hsize_t> hdims;
    for(int i = 0; i < N; i++) { hdims.push_back(dims[i]); }
    int rank     = eigen_tensor.NumDimensions;
    dataspace_id = H5Screate_simple(rank, hdims.data(), NULL);
  }

  hid_t dataset_id = H5Dcreate(file_id, "data", get_hdf5_dt<T>(), dataspace_id, H5P_DEFAULT,
                               H5P_DEFAULT, H5P_DEFAULT);

  /* herr_t status = */ H5Dwrite(dataset_id, get_hdf5_dt<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);

  /* Create and write attribute information - reduced dims */
  std::vector<int> reduced_dims{static_cast<int>(dims[0]), static_cast<int>(dims[1])};
  hsize_t          attr_size      = reduced_dims.size();
  auto             attr_dataspace = H5Screate_simple(1, &attr_size, NULL);
  auto attr_dataset = H5Dcreate(file_id, "rdims", H5T_NATIVE_INT, attr_dataspace, H5P_DEFAULT,
                                H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(attr_dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, reduced_dims.data());
  H5Dclose(attr_dataset);
  H5Sclose(attr_dataspace);

  H5Dclose(dataset_id);
  H5Sclose(dataspace_id);
  H5Fclose(file_id);
#endif
}

template<typename T, int N>
void write_to_disk_hdf5(Tensor<T> tensor, std::string filename, bool write1D = false) {
#if !defined(USE_HDF5)
  tamm::tamm_terminate("HDF5 is not enabled. Please rebuild TAMM with HDF5 support");
#else
  std::string outputfile = filename + ".data";
  hid_t       file_id    = H5Fcreate(outputfile.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  // Eigen::Tensor<T, N, Eigen::RowMajor> eigen_tensor = tamm_to_eigen_tensor<T,N>(tensor);
  std::array<Eigen::Index, N> dims;
  const auto&                 tindices = tensor.tiled_index_spaces();
  for(int i = 0; i < N; i++) { dims[i] = tindices[i].max_num_indices(); }
  Eigen::Tensor<T, N, Eigen::RowMajor> eigen_tensor;
  eigen_tensor = eigen_tensor.reshape(dims);
  eigen_tensor.setZero();

  tamm_to_eigen_tensor(tensor, eigen_tensor);
  T* buf = eigen_tensor.data();

  hid_t dataspace_id;

  if(write1D) {
    hsize_t total_size = 1;
    for(int i = 0; i < N; i++) { total_size *= dims[i]; }
    dataspace_id = H5Screate_simple(1, &total_size, NULL);
  }
  else {
    std::vector<hsize_t> hdims;
    for(int i = 0; i < N; i++) { hdims.push_back(dims[i]); }
    int rank     = eigen_tensor.NumDimensions;
    dataspace_id = H5Screate_simple(rank, hdims.data(), NULL);
  }

  hid_t dataset_id = H5Dcreate(file_id, "data", get_hdf5_dt<T>(), dataspace_id, H5P_DEFAULT,
                               H5P_DEFAULT, H5P_DEFAULT);

  /* herr_t status = */ H5Dwrite(dataset_id, get_hdf5_dt<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);

  /* Create and write attribute information - reduced dims */
  std::vector<int> reduced_dims{static_cast<int>(dims[0]), static_cast<int>(dims[1])};
  hsize_t          attr_size      = reduced_dims.size();
  auto             attr_dataspace = H5Screate_simple(1, &attr_size, NULL);
  auto attr_dataset = H5Dcreate(file_id, "rdims", H5T_NATIVE_INT, attr_dataspace, H5P_DEFAULT,
                                H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(attr_dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, reduced_dims.data());
  H5Dclose(attr_dataset);
  H5Sclose(attr_dataspace);

  H5Dclose(dataset_id);
  H5Sclose(dataspace_id);
  H5Fclose(file_id);
#endif
}

} // namespace tamm
