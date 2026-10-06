#pragma once

#include <chrono>
#include <filesystem>
#include <fstream>
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
#define TU_SG true
#define TU_SG_IO true

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

template<typename TensorType>
std::tuple<int, int, int> get_agg_info(ExecutionContext& gec, const int nranks,
                                       Tensor<TensorType> tensor, const int nagg_hint) {
  long double nelements = 1;
  // Heuristic: Use 1 agg for every 14 GiB
  const long double ne_mb = 131072 * 14.0;
  const int         ndims = tensor.num_modes();
  for(auto i = 0; i < ndims; i++)
    nelements *= tensor.tiled_index_spaces()[i].index_space().num_indices();
  // nelements = tensor.size();
  int nagg = (nelements / (ne_mb * 1024)) + 1;
#if defined(USE_UPCXX)
  const int nnodes = upcxx::local_team().rank_n();
#else
  // TODO: gec.nnodes() fails with sub-groups ?
  const int nnodes = GA_Cluster_nnodes();
#endif
  const int ppn         = gec.ppn();
  const int avail_nodes = std::min(nranks / ppn + 1, nnodes);

  if(nagg > avail_nodes) nagg = avail_nodes;
  if(nagg_hint > 0) nagg = nagg_hint;

  int subranks = nagg * ppn;
  if(subranks > nranks) subranks = nranks;

  return std::make_tuple(nagg, ppn, subranks);
}

template<typename TensorType>
std::tuple<int, int, int> get_subgroup_info(ExecutionContext& gec, Tensor<TensorType> tensor,
                                            int nagg_hint = 0) {
  int nranks = gec.pg().size().value();

  auto [nagg, ppn, subranks] = get_agg_info(gec, nranks, tensor, nagg_hint);

  return std::make_tuple(nagg, ppn, subranks);
}

#if !defined(USE_UPCXX)
static inline void subcomm_from_subranks(ExecutionContext& gec, int subranks, MPI_Comm& subcomm) {
  MPI_Group group; //, world_group;
  auto      comm = gec.pg().comm();
  MPI_Comm_group(comm, &group);
  int ranks[subranks]; //,ranks_world[subranks];
  for(int i = 0; i < subranks; i++) ranks[i] = i;
  MPI_Group tamm_subgroup;
  MPI_Group_incl(group, subranks, ranks, &tamm_subgroup);
  MPI_Comm_create(comm, tamm_subgroup, &subcomm);
  MPI_Group_free(&group);
  MPI_Group_free(&tamm_subgroup);
}
#endif

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

/// Commits the staging file of a tensor file written collectively on pg. The ranks agree on
/// whether all of them wrote their part; if so, the pg root renames the staging file over the
/// tensor file, otherwise it removes the staging file and the previous tensor file is kept.
/// A failed commit is reported but never fatal.
inline void commit_staging_file(ProcGroup pg, bool local_failed, const std::string& filename) {
  const int failed     = local_failed ? 1 : 0;
  const int any_failed = pg.allreduce(&failed, ReduceOp::max);
  if(pg.rank() != 0) return;

  if(!finish_staging_file(filename, !any_failed))
    std::cerr << "[TAMM WARNING] write_to_disk: " << filename << " not committed; "
              << (std::filesystem::exists(filename) ? "previous version kept"
                                                    : "no previous version exists")
              << std::endl;
}

} // namespace internal
#endif

/**
 * @brief write tensor to disk using HDF5
 *
 * @tparam TensorType the type of the elements in the tensor
 * @param tensor to write to disk
 * @param filename to write to disk
 */
template<typename TensorType>
void write_to_disk(Tensor<TensorType> tensor, const std::string& filename, bool tammio = true,
                   bool profile = false, int nagg_hint = 0) {
#if !defined(USE_HDF5)
  tamm_terminate("HDF5 is not enabled. Please rebuild TAMM with HDF5 support");
#else

  ExecutionContext& gec   = get_ec(tensor());
  auto              io_t1 = std::chrono::high_resolution_clock::now();
  int               rank  = gec.pg().rank().value();

#ifdef TU_SG_IO
  auto [nagg, ppn, subranks] = get_subgroup_info(gec, tensor, nagg_hint);
#if defined(USE_UPCXX)
  upcxx::team* io_comm = new upcxx::team(
    gec.pg().comm()->split(gec.pg().rank() < subranks ? 0 : upcxx::team::color_none, 0));
#else
  MPI_Comm io_comm;
  subcomm_from_subranks(gec, subranks, io_comm);
#endif
#else
  auto [nagg, ppn, subranks] = get_agg_info(gec, gec.pg().size().value(), tensor, nagg_hint);
#endif

#if !defined(USE_UPCXX)
  size_t            ndims = tensor.num_modes();
  const std::string nppn  = std::to_string(nagg) + "n," + std::to_string(ppn) + "ppn";

  int ga_tens = tensor.ga_handle();
  if(!tammio) ga_tens = tamm_to_ga(gec, tensor);

  std::vector<int64_t> tensor_dims(ndims, 1);
  const auto           tis = tensor.tiled_index_spaces();
  for(size_t i = 0; i < ndims; i++) { tensor_dims[i] = tis[i].index_space().num_indices(); }
  // NGA_Inquire64(ga_tens, &itype, &ndim, tensor_dims);
  int64_t tensor_size = std::accumulate(tensor_dims.begin(), tensor_dims.end(), (int64_t) 1,
                                        std::multiplies<int64_t>());

  if(rank == 0 && profile)
    std::cout << "tensor size: " << std::fixed << std::setprecision(2)
              << (tensor_size * 8.0) / (1024 * 1024 * 1024.0)
              << "GiB, write to disk using: " << nppn << std::endl;

  hid_t hdf5_dt = get_hdf5_dt<TensorType>();

#ifdef TU_SG_IO
  if(rank < subranks) {
    ProcGroup        pg = ProcGroup::create_coll(io_comm);
    ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
#else
  ExecutionContext& ec = gec;
#endif
    auto          ltensor = tensor();
    LabelLoopNest loop_nest{ltensor.labels()};

    internal::H5ErrorSilencer h5_silencer;
    internal::H5Check         h5{filename, "WARNING", ec.pg().rank() == 0};
    const std::string         staging = internal::staging_filename(filename);

    int ierr;
    // MPI_File fh;
    MPI_Info info;
    // MPI_Status status;
    hsize_t file_offset;
    MPI_Info_create(&info);
    MPI_Info_set(info, "cb_nodes", std::to_string(nagg).c_str());
    // MPI_File_open(ec.pg().comm(), filename.c_str(), MPI_MODE_CREATE|MPI_MODE_WRONLY,
    //             info, &fh);

    /* set the file access template for parallel IO access */
    auto acc_template = H5Pcreate(H5P_FILE_ACCESS);
    // ierr = H5Pset_sieve_buf_size(acc_template, 262144);
    // ierr = H5Pset_alignment(acc_template, 524288, 262144);
    // ierr = MPI_Info_set(info, "access_style", "write_once");
    // ierr = MPI_Info_set(info, "collective_buffering", "true");
    // ierr = MPI_Info_set(info, "cb_block_size", "1048576");
    // ierr = MPI_Info_set(info, "cb_buffer_size", "4194304");

    /* tell the HDF5 library that we want to use MPI-IO to do the writing */
    ierr = H5Pset_fapl_mpio(acc_template, ec.pg().comm(), info);
    auto file_identifier =
      h5(H5Fcreate(staging.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, acc_template), "H5Fcreate");

    /* release the file access template */
    ierr = H5Pclose(acc_template);
    ierr = MPI_Info_free(&info);

    int     tensor_rank = 1;
    hsize_t dimens_1d   = tensor_size;
    auto    dataspace   = H5Screate_simple(tensor_rank, &dimens_1d, NULL);
    /* create a dataset collectively */
    auto dataset = h5(H5Dcreate(file_identifier, "tensor", hdf5_dt, dataspace, H5P_DEFAULT,
                                H5P_DEFAULT, H5P_DEFAULT),
                      "H5Dcreate");
    internal::write_description(dataset, internal::describe_tensor(tensor), h5);
    /* create a file dataspace independently */
    auto file_dataspace = H5Dget_space(dataset);

    /* Create and write additional metadata */
    // std::vector<int> attr_dims{11,29,42};
    // hsize_t attr_size = attr_dims.size();
    // auto attr_dataspace = H5Screate_simple(1, &attr_size, NULL);
    // auto attr_dataset = H5Dcreate(file_identifier, "attr", H5T_NATIVE_INT, attr_dataspace,
    // H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); H5Dwrite(attr_dataset, H5T_NATIVE_INT, H5S_ALL,
    // H5S_ALL, H5P_DEFAULT, attr_dims.data()); H5Dclose(attr_dataset); H5Sclose(attr_dataspace);

    hid_t xfer_plist;
    /* set up the collective transfer properties list */
    xfer_plist = H5Pcreate(H5P_DATASET_XFER);
    /*auto ret = */ H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_INDEPENDENT);

    if(/*is_irreg &&*/ tammio) {
      auto lambda = [&](const IndexVector& bid) {
        const IndexVector blockid = internal::translate_blockid(bid, ltensor);

        if(h5.failed) return;
        file_offset = 0;
        for(const IndexVector& pbid: loop_nest) {
          bool is_zero = !tensor.is_non_zero(pbid);
          if(pbid == blockid) {
            if(is_zero) return;
            break;
          }
          if(is_zero) continue;
          file_offset += tensor.block_size(pbid);
        }

        // const tamm::TAMM_SIZE
        hsize_t                 dsize = tensor.block_size(blockid);
        std::vector<TensorType> dbuf(dsize);
        tensor.get(blockid, dbuf);

        // std::cout << "WRITE: rank, file_offset, size = " << rank << "," << file_offset << ", " <<
        // dsize << std::endl;

        hsize_t stride = 1;
        herr_t  ret    = H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, &file_offset, &stride,
                                             &dsize, NULL); // stride=NULL?

        // /* create a memory dataspace independently */
        auto mem_dataspace = H5Screate_simple(tensor_rank, &dsize, NULL);

        // /* write data independently */
        ret = h5(H5Dwrite(dataset, hdf5_dt, mem_dataspace, file_dataspace, xfer_plist, dbuf.data()),
                 "H5Dwrite", false);

        H5Sclose(mem_dataspace);
      };

      block_for(ec, ltensor, lambda);
    }
    else {
      // N-D GA
      auto ga_write_lambda = [&](const IndexVector& bid) {
        const IndexVector blockid = internal::translate_blockid(bid, ltensor);

        if(h5.failed) return;
        file_offset = 0;
        for(const IndexVector& pbid: loop_nest) {
          bool is_zero = !tensor.is_non_zero(pbid);
          if(pbid == blockid) {
            if(is_zero) return;
            break;
          }
          if(is_zero) continue;
          file_offset += tensor.block_size(pbid);
        }

        // file_offset = file_offset*sizeof(TensorType);

        auto block_dims   = tensor.block_dims(blockid);
        auto block_offset = tensor.block_offsets(blockid);

        hsize_t dsize = tensor.block_size(blockid);

        std::vector<int64_t> lo(ndims), hi(ndims), ld(ndims - 1);

        for(size_t i = 0; i < ndims; i++) lo[i] = cd_ncast<size_t>(block_offset[i]);
        for(size_t i = 0; i < ndims; i++)
          hi[i] = cd_ncast<size_t>(block_offset[i] + block_dims[i] - 1);
        for(size_t i = 1; i < ndims; i++) ld[i - 1] = cd_ncast<size_t>(block_dims[i]);

        std::vector<TensorType> sbuf(dsize);
        NGA_Get64(ga_tens, &lo[0], &hi[0], &sbuf[0], &ld[0]);
        // MPI_File_write_at(fh,file_offset,reinterpret_cast<void*>(&sbuf[0]),
        //     static_cast<int>(dsize),mpi_type<TensorType>(),&status);

        hsize_t stride = 1;
        herr_t  ret    = H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, &file_offset, &stride,
                                             &dsize, NULL); // stride=NULL?

        // /* create a memory dataspace independently */
        auto mem_dataspace = H5Screate_simple(tensor_rank, &dsize, NULL);

        // /* write data independently */
        ret = h5(H5Dwrite(dataset, hdf5_dt, mem_dataspace, file_dataspace, xfer_plist, sbuf.data()),
                 "H5Dwrite", false);

        H5Sclose(mem_dataspace);
      };

      block_for(ec, ltensor, ga_write_lambda);
    }

    H5Sclose(file_dataspace);
    // H5Sclose(mem_dataspace);
    H5Pclose(xfer_plist);

    H5Dclose(dataset);
    H5Sclose(dataspace);
    h5(H5Fclose(file_identifier), "H5Fclose");

    internal::commit_staging_file(ec.pg(), h5.failed, filename);

#ifdef TU_SG_IO
    ec.flush_and_sync();
    // MemoryManagerGA::destroy_coll(mgr);
    MPI_Comm_free(&io_comm);
    pg.destroy_coll();
  }
#endif

  gec.pg().barrier();
  if(!tammio) NGA_Destroy(ga_tens);
  auto io_t2 = std::chrono::high_resolution_clock::now();

  double io_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((io_t2 - io_t1)).count();
  if(rank == 0 && profile)
    std::cout << "Time for writing " << filename << " to disk (" << nppn << "): " << io_time
              << " secs" << std::endl;
#endif
#endif
}

template<typename TensorType>
void write_to_disk(Tensor<TensorType> tensor, const std::string& filename, bool profile,
                   int nagg_hint = 0) {
  write_to_disk(tensor, filename, true, profile, nagg_hint);
}

/**
 * @brief Write batch of tensors to disk using HDF5.
 *        Uses process groups for concurrent writes.
 * @tparam TensorType the type of the elements in the tensor
 * @param tensor to write to disk
 * @param filename to write to disk
 */
template<typename TensorType>
void write_to_disk_group(ExecutionContext& gec, std::vector<Tensor<TensorType>> tensors,
                         std::vector<std::string> filenames, bool profile = false,
                         int nagg_hint = 0) {
  EXPECTS(tensors.size() == filenames.size());

#if !defined(USE_HDF5)
  tamm_terminate("HDF5 is not enabled. Please rebuild TAMM with HDF5 support");
#else
#if !defined(USE_UPCXX)
  auto io_t1 = std::chrono::high_resolution_clock::now();

  hid_t hdf5_dt = get_hdf5_dt<TensorType>();

  const int  world_rank = gec.pg().rank().value();
  const auto world_size = gec.pg().size().value();
  auto       world_comm = gec.pg().comm();

  int nranks        = world_size;
  int color         = -1;
  int prev_subranks = 0;

  std::vector<int> rankspertensor;
  if(nagg_hint > 0) nagg_hint = nagg_hint / tensors.size();
  for(size_t i = 0; i < tensors.size(); i++) {
    auto [nagg, ppn, subranks] = get_agg_info(gec, gec.pg().size().value(), tensors[i], nagg_hint);
    rankspertensor.push_back(subranks);
    if(world_rank >= prev_subranks && world_rank < (subranks + prev_subranks)) color = i;
    nranks -= subranks;
    if(nranks <= 0) break;
    prev_subranks += subranks;
  }
  if(color == -1) color = MPI_UNDEFINED;

  if(world_rank == 0 && profile) {
    std::cout << "Number of tensors to be written, process groups, sizes: " << tensors.size() << ","
              << rankspertensor.size() << ", " << rankspertensor << std::endl;
  }

  MPI_Comm io_comm;
  MPI_Comm_split(world_comm, color, world_rank, &io_comm);

  AtomicCounter* ac = new AtomicCounterGA(gec.pg(), 1);
  ac->allocate(0);
  int64_t taskcount = 0;
  int64_t next      = -1;
  // int total_pi_pg = 0;

  // The group is committed all-or-nothing: per tensor file, whether any rank failed writing it,
  // and whether this rank is the root of the subgroup that wrote it (and so commits it).
  std::vector<int>  write_failed(tensors.size(), 0);
  std::vector<bool> commits_here(tensors.size(), false);

  if(io_comm != MPI_COMM_NULL) {
    ProcGroup        pg = ProcGroup::create_coll(io_comm);
    ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};

    int root_ppi = -1;
    MPI_Comm_rank(ec.pg().comm(), &root_ppi);

    // int pg_id = rank/subranks;
    if(root_ppi == 0) next = ac->fetch_add(0, 1);
    ec.pg().broadcast(&next, 0);

    for(size_t i = 0; i < tensors.size(); i++) {
      if(next == taskcount) {
        Tensor<TensorType> tensor   = tensors[i];
        auto               filename = filenames[i];

        auto io_t1 = std::chrono::high_resolution_clock::now();

        size_t ndims = tensor.num_modes();
        // const std::string nppn = std::to_string(nagg) + "n," + std::to_string(ppn) + "ppn";
        // if(root_ppi == 0 && profile)
        //   std::cout << "write " << filename << " to disk using: " << ec.pg().size().value() <<
        //   " ranks" << std::endl;

        std::vector<int64_t> tensor_dims(ndims, 1);
        const auto           tis = tensor.tiled_index_spaces();
        for(size_t i = 0; i < ndims; i++) { tensor_dims[i] = tis[i].index_space().num_indices(); }

        int64_t tensor_size = std::accumulate(tensor_dims.begin(), tensor_dims.end(), (int64_t) 1,
                                              std::multiplies<int64_t>());
        auto    ltensor     = tensor();
        LabelLoopNest loop_nest{ltensor.labels()};

        internal::H5ErrorSilencer h5_silencer;
        internal::H5Check         h5{filename, "WARNING", root_ppi == 0};
        const std::string         staging = internal::staging_filename(filename);

        int ierr;
        // MPI_File fh;
        MPI_Info info;
        // MPI_Status status;
        hsize_t file_offset;
        MPI_Info_create(&info);
        // MPI_Info_set(info,"cb_nodes",std::to_string(nagg).c_str());
        // MPI_File_open(ec.pg().comm(), filename.c_str(), MPI_MODE_CREATE|MPI_MODE_WRONLY,
        //             info, &fh);

        /* set the file access template for parallel IO access */
        auto acc_template = H5Pcreate(H5P_FILE_ACCESS);
        // ierr = H5Pset_sieve_buf_size(acc_template, 262144);
        // ierr = H5Pset_alignment(acc_template, 524288, 262144);
        // ierr = MPI_Info_set(info, "access_style", "write_once");
        // ierr = MPI_Info_set(info, "collective_buffering", "true");
        // ierr = MPI_Info_set(info, "cb_block_size", "1048576");
        // ierr = MPI_Info_set(info, "cb_buffer_size", "4194304");

        /* tell the HDF5 library that we want to use MPI-IO to do the writing */
        ierr = H5Pset_fapl_mpio(acc_template, ec.pg().comm(), info);
        auto file_identifier =
          h5(H5Fcreate(staging.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, acc_template), "H5Fcreate");

        /* release the file access template */
        ierr = H5Pclose(acc_template);
        ierr = MPI_Info_free(&info);

        int     tensor_rank = 1;
        hsize_t dimens_1d   = tensor_size;
        auto    dataspace   = H5Screate_simple(tensor_rank, &dimens_1d, NULL);
        /* create a dataset collectively */
        auto dataset = h5(H5Dcreate(file_identifier, "tensor", hdf5_dt, dataspace, H5P_DEFAULT,
                                    H5P_DEFAULT, H5P_DEFAULT),
                          "H5Dcreate");
        internal::write_description(dataset, internal::describe_tensor(tensor), h5);
        /* create a file dataspace independently */
        auto file_dataspace = H5Dget_space(dataset);

        /* Create and write additional metadata */
        // std::vector<int> attr_dims{11,29,42};
        // hsize_t attr_size = attr_dims.size();
        // auto attr_dataspace = H5Screate_simple(1, &attr_size, NULL);
        // auto attr_dataset = H5Dcreate(file_identifier, "attr", H5T_NATIVE_INT, attr_dataspace,
        // H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); H5Dwrite(attr_dataset, H5T_NATIVE_INT, H5S_ALL,
        // H5S_ALL, H5P_DEFAULT, attr_dims.data()); H5Dclose(attr_dataset);
        // H5Sclose(attr_dataspace);

        hid_t xfer_plist;
        /* set up the collective transfer properties list */
        xfer_plist = H5Pcreate(H5P_DATASET_XFER);
        /*auto ret = */ H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_INDEPENDENT);

        auto lambda = [&](const IndexVector& bid) {
          const IndexVector blockid = internal::translate_blockid(bid, ltensor);

          if(h5.failed) return;
          file_offset = 0;
          for(const IndexVector& pbid: loop_nest) {
            bool is_zero = !tensor.is_non_zero(pbid);
            if(pbid == blockid) {
              if(is_zero) return;
              break;
            }
            if(is_zero) continue;
            file_offset += tensor.block_size(pbid);
          }

          hsize_t                 dsize = tensor.block_size(blockid);
          std::vector<TensorType> dbuf(dsize);
          tensor.get(blockid, dbuf);

          hsize_t stride = 1;
          herr_t  ret = H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, &file_offset, &stride,
                                            &dsize, NULL); // stride=NULL?

          // /* create a memory dataspace independently */
          auto mem_dataspace = H5Screate_simple(tensor_rank, &dsize, NULL);

          // /* write data independently */
          ret =
            h5(H5Dwrite(dataset, hdf5_dt, mem_dataspace, file_dataspace, xfer_plist, dbuf.data()),
               "H5Dwrite", false);

          H5Sclose(mem_dataspace);
        };

        block_for(ec, ltensor, lambda);

        H5Sclose(file_dataspace);
        // H5Sclose(mem_dataspace);
        H5Pclose(xfer_plist);

        H5Dclose(dataset);
        H5Sclose(dataspace);
        h5(H5Fclose(file_identifier), "H5Fclose");

        if(h5.failed) write_failed[i] = 1;
        if(root_ppi == 0) commits_here[i] = true;

        auto io_t2 = std::chrono::high_resolution_clock::now();

        double io_time =
          std::chrono::duration_cast<std::chrono::duration<double>>((io_t2 - io_t1)).count();
        if(root_ppi == 0 && profile)
          std::cout << "Time for writing " << filename << " to disk (" << ec.pg().size().value()
                    << "): " << io_time << " secs" << std::endl;

        if(root_ppi == 0) next = ac->fetch_add(0, 1);
        ec.pg().broadcast(&next, 0);

      } // next==taskcount

      if(root_ppi == 0) taskcount++;
      ec.pg().broadcast(&taskcount, 0);

    } // loop over tensors

    ec.flush_and_sync();
    MPI_Comm_free(&io_comm);
    // MemoryManagerGA::destroy_coll(mgr);
    pg.destroy_coll();
  } // io_comm != MPI_COMM_NULL

  ac->deallocate();
  delete ac;

  // All ranks agree on which tensor files failed. Only if none did is the group committed, so
  // the tensor files of a group always come from the same write_to_disk_group call.
  std::vector<int> any_write_failed(tensors.size(), 0);
  gec.pg().allreduce(write_failed.data(), any_write_failed.data(),
                     static_cast<int>(write_failed.size()), ReduceOp::max);
  std::vector<std::string> failed_files;
  for(size_t i = 0; i < tensors.size(); i++) {
    if(any_write_failed[i]) failed_files.push_back(filenames[i]);
  }
  const bool commit = failed_files.empty();
  for(size_t i = 0; i < tensors.size(); i++) {
    if(commits_here[i]) internal::finish_staging_file(filenames[i], commit);
  }
  if(!commit && world_rank == 0)
    std::cerr << "[TAMM WARNING] write_to_disk_group: no tensor file in the group was committed"
              << internal::file_list("failed to write", failed_files)
              << internal::kept_versions(filenames) << std::endl;
  gec.pg().barrier();

  auto io_t2 = std::chrono::high_resolution_clock::now();

  double io_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((io_t2 - io_t1)).count();
  if(world_rank == 0 && profile)
    std::cout << "Total Time for writing tensors" << " to disk: " << io_time << " secs"
              << std::endl;
#endif
#endif
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
 * @brief read tensor from disk using HDF5
 *
 * @tparam TensorType the type of the elements in the tensor
 * @param tensor to read into
 * @param filename to read from disk
 */
template<typename TensorType>
void read_from_disk(Tensor<TensorType> tensor, const std::string& filename, bool tammio = true,
                    Tensor<TensorType> wtensor = {}, bool profile = false, int nagg_hint = 0) {
#if !defined(USE_HDF5)
  tamm_terminate("HDF5 is not enabled. Please rebuild TAMM with HDF5 support");
#else
#if !defined(USE_UPCXX)
  ExecutionContext& gec   = get_ec(tensor());
  auto              io_t1 = std::chrono::high_resolution_clock::now();
  int               rank  = gec.pg().rank().value();
#ifdef TU_SG_IO
  auto [nagg, ppn, subranks] = get_subgroup_info(gec, tensor, nagg_hint);
  MPI_Comm io_comm;
  subcomm_from_subranks(gec, subranks, io_comm);
#else
  auto [nagg, ppn, subranks] = get_agg_info(gec, gec.pg().size().value(), tensor, nagg_hint);
#endif

  const std::string nppn = std::to_string(nagg) + "n," + std::to_string(ppn) + "ppn";
  if(rank == 0 && profile) std::cout << "read from disk using: " << nppn << std::endl;

  int ga_tens = tensor.ga_handle();
  if(!tammio) {
    auto tis_dims = tensor.tiled_index_spaces();

    int                  ndims = tensor.num_modes();
    std::vector<int64_t> dims;
    std::vector<int64_t> chnks(ndims, -1);
    for(auto tis: tis_dims) dims.push_back(tis.index_space().num_indices());

    ga_tens = NGA_Create64(to_ga_eltype(tensor_element_type<TensorType>()), ndims, &dims[0],
                           const_cast<char*>("iotemp"), &chnks[0]);
  }

  hid_t hdf5_dt = get_hdf5_dt<TensorType>();

  auto tensor_back = tensor;
  int  read_failed = 0;

#ifdef TU_SG_IO
  if(io_comm != MPI_COMM_NULL) {
    ProcGroup        pg = ProcGroup::create_coll(io_comm);
    ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
#else
  ExecutionContext& ec = gec;
#endif

    if(wtensor.num_modes() > 0) tensor = wtensor;

    auto          ltensor = tensor();
    LabelLoopNest loop_nest{ltensor.labels()};

    internal::H5ErrorSilencer h5_silencer;
    internal::H5Check         h5{filename, "ERROR", ec.pg().rank() == 0};

    int ierr;
    // MPI_File fh;
    MPI_Info info;
    // MPI_Status status;
    hsize_t file_offset;
    MPI_Info_create(&info);
    // MPI_Info_set(info,"romio_cb_read", "enable");
    // MPI_Info_set(info,"striping_unit","4194304");
    MPI_Info_set(info, "cb_nodes", std::to_string(nagg).c_str());

    // MPI_File_open(ec.pg().comm(), filename.c_str(), MPI_MODE_RDONLY,
    //                 info, &fh);

    /* set the file access template for parallel IO access */
    auto acc_template = H5Pcreate(H5P_FILE_ACCESS);

    /* tell the HDF5 library that we want to use MPI-IO to do the reading */
    ierr                 = H5Pset_fapl_mpio(acc_template, ec.pg().comm(), info);
    auto file_identifier = h5(H5Fopen(filename.c_str(), H5F_ACC_RDONLY, acc_template), "H5Fopen");

    /* release the file access template */
    ierr = H5Pclose(acc_template);
    ierr = MPI_Info_free(&info);

    int tensor_rank = 1;
    // hsize_t dimens_1d = tensor_size;
    /* create a dataset collectively */
    auto dataset = h5(H5Dopen(file_identifier, "tensor", H5P_DEFAULT), "H5Dopen");
    internal::check_description(dataset, hdf5_dt, internal::describe_tensor(tensor), h5);
    /* create a file dataspace independently */
    auto file_dataspace = H5Dget_space(dataset);

    /* Read additional metadata */
    // std::vector<int> attr_dims(3);
    // auto attr_dataset = H5Dopen(file_identifier, "attr",  H5P_DEFAULT);
    // H5Dread(attr_dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, attr_dims.data());
    // H5Dclose(attr_dataset);

    hid_t xfer_plist;
    /* set up the collective transfer properties list */
    xfer_plist = H5Pcreate(H5P_DATASET_XFER);
    /*auto ret = */ H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_INDEPENDENT);

    if(/*is_irreg &&*/ tammio) {
      auto lambda = [&](const IndexVector& bid) {
        const IndexVector blockid = internal::translate_blockid(bid, ltensor);

        if(h5.failed) return;
        file_offset = 0;
        for(const IndexVector& pbid: loop_nest) {
          bool is_zero = !tensor.is_non_zero(pbid);
          if(pbid == blockid) {
            if(is_zero) return;
            break;
          }
          if(is_zero) continue;
          file_offset += tensor.block_size(pbid);
        }

        // file_offset = file_offset*sizeof(TensorType);

        hsize_t                 dsize = tensor.block_size(blockid);
        std::vector<TensorType> dbuf(dsize);

        // std::cout << "READ: rank, file_offset, size = " << rank << "," << file_offset << ", " <<
        // dsize << std::endl;

        hsize_t stride = 1;
        herr_t  ret    = H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, &file_offset, &stride,
                                             &dsize, NULL); // stride=NULL?

        // /* create a memory dataspace independently */
        auto mem_dataspace = H5Screate_simple(tensor_rank, &dsize, NULL);

        // MPI_File_read_at(fh,file_offset,reinterpret_cast<void*>(&dbuf[0]),
        //                 static_cast<int>(dsize),mpi_type<TensorType>(),&status);

        // /* read data independently */
        ret = h5(H5Dread(dataset, hdf5_dt, mem_dataspace, file_dataspace, xfer_plist, dbuf.data()),
                 "H5Dread", false);

        if(ret >= 0) tensor.put(blockid, dbuf);

        H5Sclose(mem_dataspace);
      };

      block_for(ec, ltensor, lambda);
    }
    else {
      auto ga_read_lambda = [&](const IndexVector& bid) {
        const IndexVector blockid = internal::translate_blockid(bid, tensor());

        if(h5.failed) return;
        file_offset = 0;
        for(const IndexVector& pbid: loop_nest) {
          bool is_zero = !tensor.is_non_zero(pbid);
          if(pbid == blockid) {
            if(is_zero) return;
            break;
          }
          if(is_zero) continue;
          file_offset += tensor.block_size(pbid);
        }

        // file_offset = file_offset*sizeof(TensorType);

        auto block_dims   = tensor.block_dims(blockid);
        auto block_offset = tensor.block_offsets(blockid);

        hsize_t dsize = tensor.block_size(blockid);

        size_t               ndims = block_dims.size();
        std::vector<int64_t> lo(ndims), hi(ndims), ld(ndims - 1);

        for(size_t i = 0; i < ndims; i++) lo[i] = cd_ncast<size_t>(block_offset[i]);
        for(size_t i = 0; i < ndims; i++)
          hi[i] = cd_ncast<size_t>(block_offset[i] + block_dims[i] - 1);
        for(size_t i = 1; i < ndims; i++) ld[i - 1] = cd_ncast<size_t>(block_dims[i]);

        std::vector<TensorType> sbuf(dsize);

        // MPI_File_read_at(fh,file_offset,reinterpret_cast<void*>(&sbuf[0]),
        //             static_cast<int>(dsize),mpi_type<TensorType>(),&status);
        hsize_t stride = 1;
        herr_t  ret    = H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, &file_offset, &stride,
                                             &dsize, NULL); // stride=NULL?

        // /* create a memory dataspace independently */
        auto mem_dataspace = H5Screate_simple(tensor_rank, &dsize, NULL);

        // MPI_File_read_at(fh,file_offset,reinterpret_cast<void*>(&dbuf[0]),
        //                 static_cast<int>(dsize),mpi_type<TensorType>(),&status);

        // /* read data independently */
        ret = h5(H5Dread(dataset, hdf5_dt, mem_dataspace, file_dataspace, xfer_plist, sbuf.data()),
                 "H5Dread", false);

        if(ret >= 0) NGA_Put64(ga_tens, &lo[0], &hi[0], &sbuf[0], &ld[0]);
      };

      block_for(ec, tensor(), ga_read_lambda);
    }

    H5Sclose(file_dataspace);
    // H5Sclose(mem_dataspace);
    H5Pclose(xfer_plist);

    H5Dclose(dataset);
    h5(H5Fclose(file_identifier), "H5Fclose");
    read_failed = h5.failed ? 1 : 0;

#ifdef TU_SG_IO
    ec.flush_and_sync();
    // MemoryManagerGA::destroy_coll(mgr);
    MPI_Comm_free(&io_comm);
    pg.destroy_coll();
    // MPI_File_close(&fh);
  }
#endif

  tensor = tensor_back;

  // All ranks agree on the outcome (this also replaces the barrier); a failed read is fatal.
  const int any_read_failed = gec.pg().allreduce(&read_failed, ReduceOp::max);
  if(any_read_failed) tamm_terminate("read_from_disk: failed to read tensor file: " + filename);

  if(!tammio) {
    ga_to_tamm(gec, tensor, ga_tens);
    NGA_Destroy(ga_tens);
  }

  auto io_t2 = std::chrono::high_resolution_clock::now();

  double io_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((io_t2 - io_t1)).count();
  if(rank == 0 && profile)
    std::cout << "Time for reading " << filename << " from disk (" << nppn << "): " << io_time
              << " secs" << std::endl;
#endif
#endif
}

template<typename TensorType>
void read_from_disk(Tensor<TensorType> tensor, const std::string& filename, bool profile,
                    int nagg_hint = 0) {
  read_from_disk(tensor, filename, true, {}, profile, nagg_hint);
}

/**
 * @brief Read batch of tensors from disk using HDF5.
 *        Uses process groups for concurrent reads.
 * @tparam TensorType the type of the elements in the tensor
 * @param tensor to read into
 * @param filename to read from disk
 */
template<typename TensorType>
void read_from_disk_group(ExecutionContext& gec, std::vector<Tensor<TensorType>> tensors,
                          std::vector<std::string>        filenames,
                          std::vector<Tensor<TensorType>> wtensors = {}, bool profile = false,
                          int nagg_hint = 0) {
#if !defined(USE_HDF5)
  tamm_terminate("HDF5 is not enabled. Please rebuild TAMM with HDF5 support");
#else
  EXPECTS(tensors.size() == filenames.size());
#if !defined(USE_UPCXX)
  auto io_t1 = std::chrono::high_resolution_clock::now();

  hid_t hdf5_dt = get_hdf5_dt<TensorType>();

  const int  world_rank = gec.pg().rank().value();
  const auto world_size = gec.pg().size().value();
  auto       world_comm = gec.pg().comm();

  int nranks        = world_size;
  int color         = -1;
  int prev_subranks = 0;

  std::vector<int> rankspertensor;
  if(nagg_hint > 0) nagg_hint = nagg_hint / tensors.size();
  for(size_t i = 0; i < tensors.size(); i++) {
    auto [nagg, ppn, subranks] = get_agg_info(gec, gec.pg().size().value(), tensors[i], nagg_hint);
    rankspertensor.push_back(subranks);
    if(world_rank >= prev_subranks && world_rank < (subranks + prev_subranks)) color = i;
    nranks -= subranks;
    if(nranks <= 0) break;
    prev_subranks += subranks;
  }
  if(color == -1) color = MPI_UNDEFINED;

  if(world_rank == 0 && profile) {
    std::cout << "Number of tensors to be read, process groups, sizes: " << tensors.size() << ","
              << rankspertensor.size() << ", " << rankspertensor << std::endl;
  }

  MPI_Comm io_comm;
  MPI_Comm_split(world_comm, color, world_rank, &io_comm);

  AtomicCounter* ac = new AtomicCounterGA(gec.pg(), 1);
  ac->allocate(0);
  int64_t taskcount = 0;
  int64_t next      = -1;
  // int total_pi_pg = 0;

  std::vector<int> read_failed(tensors.size(), 0);

  if(io_comm != MPI_COMM_NULL) {
    ProcGroup        pg = ProcGroup::create_coll(io_comm);
    ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};

    int root_ppi = -1;
    MPI_Comm_rank(ec.pg().comm(), &root_ppi);

    // int pg_id = rank/subranks;
    if(root_ppi == 0) next = ac->fetch_add(0, 1);
    ec.pg().broadcast(&next, 0);

    bool is_wt = wtensors.empty();
    for(size_t i = 0; i < tensors.size(); i++) {
      if(next == taskcount) {
        auto io_t1 = std::chrono::high_resolution_clock::now();

        Tensor<TensorType> tensor   = tensors[i];
        auto               filename = filenames[i];

        // auto tensor_back = tensor;

        if(!is_wt) {
          if(wtensors[i].num_modes() > 0) tensor = wtensors[i];
        }

        auto          ltensor = tensor();
        LabelLoopNest loop_nest{ltensor.labels()};

        internal::H5ErrorSilencer h5_silencer;
        internal::H5Check         h5{filename, "ERROR", root_ppi == 0};

        int ierr;
        // MPI_File fh;
        MPI_Info info;
        // MPI_Status status;
        hsize_t file_offset;
        MPI_Info_create(&info);
        // MPI_Info_set(info,"romio_cb_read", "enable");
        // MPI_Info_set(info,"striping_unit","4194304");
        // MPI_Info_set(info,"cb_nodes",std::to_string(nagg).c_str());

        // MPI_File_open(ec.pg().comm(), filename.c_str(), MPI_MODE_RDONLY,
        //                 info, &fh);

        /* set the file access template for parallel IO access */
        auto acc_template = H5Pcreate(H5P_FILE_ACCESS);

        /* tell the HDF5 library that we want to use MPI-IO to do the reading */
        ierr = H5Pset_fapl_mpio(acc_template, ec.pg().comm(), info);
        auto file_identifier =
          h5(H5Fopen(filename.c_str(), H5F_ACC_RDONLY, acc_template), "H5Fopen");

        /* release the file access template */
        ierr = H5Pclose(acc_template);
        ierr = MPI_Info_free(&info);

        int tensor_rank = 1;
        // hsize_t dimens_1d = tensor_size;
        /* create a dataset collectively */
        auto dataset = h5(H5Dopen(file_identifier, "tensor", H5P_DEFAULT), "H5Dopen");
        internal::check_description(dataset, hdf5_dt, internal::describe_tensor(tensor), h5);
        /* create a file dataspace independently */
        auto file_dataspace = H5Dget_space(dataset);

        /* Read additional metadata */
        // std::vector<int> attr_dims(3);
        // auto attr_dataset = H5Dopen(file_identifier, "attr",  H5P_DEFAULT);
        // H5Dread(attr_dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, attr_dims.data());
        // H5Dclose(attr_dataset);

        hid_t xfer_plist;
        /* set up the collective transfer properties list */
        xfer_plist = H5Pcreate(H5P_DATASET_XFER);
        /*auto ret = */ H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_INDEPENDENT);

        auto lambda = [&](const IndexVector& bid) {
          const IndexVector blockid = internal::translate_blockid(bid, ltensor);

          if(h5.failed) return;
          file_offset = 0;
          for(const IndexVector& pbid: loop_nest) {
            bool is_zero = !tensor.is_non_zero(pbid);
            if(pbid == blockid) {
              if(is_zero) return;
              break;
            }
            if(is_zero) continue;
            file_offset += tensor.block_size(pbid);
          }

          // file_offset = file_offset*sizeof(TensorType);

          hsize_t                 dsize = tensor.block_size(blockid);
          std::vector<TensorType> dbuf(dsize);

          // std::cout << "READ: rank, file_offset, size = " << rank << "," << file_offset << ", "
          // << dsize << std::endl;

          hsize_t stride = 1;
          herr_t  ret = H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, &file_offset, &stride,
                                            &dsize, NULL); // stride=NULL?

          // /* create a memory dataspace independently */
          auto mem_dataspace = H5Screate_simple(tensor_rank, &dsize, NULL);

          // MPI_File_read_at(fh,file_offset,reinterpret_cast<void*>(&dbuf[0]),
          //                 static_cast<int>(dsize),mpi_type<TensorType>(),&status);

          // /* read data independently */
          ret =
            h5(H5Dread(dataset, hdf5_dt, mem_dataspace, file_dataspace, xfer_plist, dbuf.data()),
               "H5Dread", false);

          if(ret >= 0) tensor.put(blockid, dbuf);

          H5Sclose(mem_dataspace);
        };

        block_for(ec, ltensor, lambda);

        H5Sclose(file_dataspace);
        // H5Sclose(mem_dataspace);
        H5Pclose(xfer_plist);

        H5Dclose(dataset);
        h5(H5Fclose(file_identifier), "H5Fclose");
        if(h5.failed) read_failed[i] = 1;

        // tensor = tensor_back;

        auto io_t2 = std::chrono::high_resolution_clock::now();

        double io_time =
          std::chrono::duration_cast<std::chrono::duration<double>>((io_t2 - io_t1)).count();
        if(root_ppi == 0 && profile)
          std::cout << "Time for reading " << filename << " from disk (" << ec.pg().size().value()
                    << "): " << io_time << " secs" << std::endl;

        if(root_ppi == 0) next = ac->fetch_add(0, 1);
        ec.pg().broadcast(&next, 0);

      } // next==taskcount

      if(root_ppi == 0) taskcount++;
      ec.pg().broadcast(&taskcount, 0);

    } // loop over tensors

    ec.flush_and_sync();
    MPI_Comm_free(&io_comm);
    // MemoryManagerGA::destroy_coll(mgr);
    pg.destroy_coll();
  } // iocomm!=MPI_COMM_NULL

  ac->deallocate();
  delete ac;

  // All ranks agree on which tensor files failed (this also replaces the barrier); any failed
  // read is fatal and every failed file is reported together.
  std::vector<int> any_read_failed(tensors.size(), 0);
  gec.pg().allreduce(read_failed.data(), any_read_failed.data(),
                     static_cast<int>(read_failed.size()), ReduceOp::max);
  std::vector<std::string> failed_files;
  for(size_t i = 0; i < tensors.size(); i++) {
    if(any_read_failed[i]) failed_files.push_back(filenames[i]);
  }
  if(!failed_files.empty())
    tamm_terminate("read_from_disk_group: one or more tensor files could not be read" +
                   internal::file_list("failed to read", failed_files) + "\n");

  auto io_t2 = std::chrono::high_resolution_clock::now();

  double io_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((io_t2 - io_t1)).count();
  if(world_rank == 0 && profile)
    std::cout << "Total Time for reading tensors" << " from disk: " << io_time << " secs"
              << std::endl;
#endif
#endif
}

template<typename TensorType>
void read_from_disk_group(ExecutionContext& gec, std::vector<Tensor<TensorType>> tensors,
                          std::vector<std::string> filenames, bool profile, int nagg_hint = 0) {
  read_from_disk_group(gec, tensors, filenames, {}, profile, nagg_hint);
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
