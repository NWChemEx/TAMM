// Copyright 2016 Pacific Northwest National Laboratory
// C++20 modernization: removed pre-C++17 apply shim (C++20 guarantees
// std::apply), inline constexpr maxrank, [[nodiscard]] on queries,
// std::bit_cast in hash_combine for strict-aliasing safety.
// Perf: IndexVector changed to BoundVec<Index,maxrank> to eliminate
// per-block heap allocation on every distributed tensor get/put/add.

#pragma once

#include "ga/ga-mpi.h"
#include "ga/ga.h"
#include "tamm/boundvec.hpp"
#include "tamm/errors.hpp"
#include "tamm/strong_num.hpp"
#include <atomic>     // std::atomic (thread-safe make_label)
#include <bit>        // std::bit_cast (C++20)
#include <complex>
#include <functional>
#include <iosfwd>
#include <map>
#include <span>       // C++20
#if defined(USE_UPCXX)
#include <upcxx/upcxx.hpp>
#endif

namespace tamm {

// ---------------------------------------------------------------------------
// Internal utilities
// ---------------------------------------------------------------------------
namespace internal {

/// Mix hash of v into seed (uses std::bit_cast for strict-aliasing safety).
template<typename T>
void hash_combine(size_t& seed, const T& v) {
  constexpr size_t magic = 0x9e3779b9ULL;
  seed ^= std::hash<T>{}(v) + magic + (seed << 6) + (seed >> 2);
}

/// Fold a parameter pack into a vector (C++17 fold-expression, kept here).
template<typename T, typename... Args>
void unfold_vec(std::vector<T>& vec, Args&&... args) {
  static_assert((std::is_constructible_v<T, Args&&> && ...));
  (vec.push_back(std::forward<Args>(args)), ...);
}

} // namespace internal

// ---------------------------------------------------------------------------
// Fundamental scalar / index aliases
// ---------------------------------------------------------------------------
using TAMM_SIZE     = uint64_t;
using Index         = uint32_t;
using Tile          = uint32_t;
using HashData      = uint64_t;
using StringLabelVec = std::vector<std::string>;

// TensorRank / BoundVec aliases  (defined early — IndexVector depends on maxrank)
using TensorRank = size_t;
inline constexpr TensorRank maxrank{8};

// ---------------------------------------------------------------------------
// IndexVector: stack-allocated fixed-capacity block-id vector.
// Changed from std::vector<Index> to BoundVec<Index,maxrank> to eliminate
// one heap allocation per distributed tensor get/put/add call.  Block IDs
// never exceed maxrank elements so the stack capacity is always sufficient.
// IndexVectorHash / IndexVectorEqual are updated accordingly below.
// ---------------------------------------------------------------------------
using IndexVector   = BoundVec<Index, maxrank>;
using IndexIterator = IndexVector::const_iterator;

class TiledIndexSpace;
using TiledIndexSpaceVec = std::vector<TiledIndexSpace>;
class TiledIndexLabel;
using IndexLabelVec = std::vector<TiledIndexLabel>;

using Perm       = int32_t;
using PermVector = std::vector<Perm>;

// ---------------------------------------------------------------------------
// StrongNum aliases
// ---------------------------------------------------------------------------
struct IrrepSpace;   using Irrep      = StrongNum<IrrepSpace,  uint32_t>;
struct SpinSpace;    using Spin       = StrongNum<SpinSpace,   uint32_t>;
struct SpatialSpace; using Spatial    = StrongNum<SpatialSpace,uint32_t>;
struct OffsetSpace;  using Offset     = StrongNum<OffsetSpace, uint64_t>;
struct BlockIndexSpace; using BlockIndex = StrongNum<BlockIndexSpace, uint64_t>;
struct ProcSpace;    using Proc       = StrongNum<ProcSpace,   int64_t>;
struct SignSpace;    using Sign       = StrongNum<SignSpace,   int32_t>;

// these are typedefs for usability
using Size       = Offset;
using BlockCount = BlockIndex;
using Label      = int;   // needs to support negative values
using IntLabel   = int32_t; // a simple integer label for indices
using IntLabelVec = std::vector<IntLabel>;
using SizeVec    = std::vector<Size>;
using ProcGrid   = std::vector<Proc>;
using ProcList   = std::vector<int>;

// BoundVec-based tensor dimension/permutation helpers
template<typename T>
using TensorVec  = BoundVec<T, maxrank>;
using BlockDimVec = TensorVec<BlockIndex>;
using PermVec     = TensorVec<uint32_t>;

// ---------------------------------------------------------------------------
// Enumerations (all scoped enum class)
// ---------------------------------------------------------------------------
enum class AllocationStatus : uint8_t { invalid, created, attached, deallocated, orphaned };

enum class ElementType : uint8_t {
  invalid,
  single_precision,
  double_precision,
  single_complex,
  double_complex
};

enum class DistributionKind : uint8_t {
  invalid, nw, dense, simple_round_robin, view, unit_tile
};

enum class MemoryManagerKind : uint8_t { invalid, ga, local };

// ---------------------------------------------------------------------------
// Element type queries
// ---------------------------------------------------------------------------
template<typename T>
constexpr ElementType tensor_element_type() noexcept { return ElementType::invalid; }
template<> constexpr ElementType tensor_element_type<double>()             noexcept { return ElementType::double_precision; }
template<> constexpr ElementType tensor_element_type<float>()              noexcept { return ElementType::single_precision; }
template<> constexpr ElementType tensor_element_type<std::complex<float>>() noexcept { return ElementType::single_complex; }
template<> constexpr ElementType tensor_element_type<std::complex<double>>() noexcept { return ElementType::double_complex; }

[[nodiscard]] static inline constexpr size_t element_size(ElementType eltype) noexcept {
  switch (eltype) {
    case ElementType::single_precision: return sizeof(float);
    case ElementType::double_precision: return sizeof(double);
    case ElementType::single_complex:   return sizeof(std::complex<float>);
    case ElementType::double_complex:   return sizeof(std::complex<double>);
    default: return 0;
  }
}

// ---------------------------------------------------------------------------
// Remaining enumerations
// ---------------------------------------------------------------------------
using RangeValue = int64_t;

enum class SpinPosition  : uint8_t { ignore, upper, lower };
enum class IndexPosition : uint8_t { upper, lower, neither };
enum class SpinType      : uint8_t { ao_spin, mo_spin };
enum class ExecutionHW   : uint8_t { CPU, GPU, DEFAULT };
enum class ReduceOp      : uint8_t { min, max, sum, maxloc, minloc };

using SpinMask = std::vector<SpinPosition>;

// ---------------------------------------------------------------------------
// Runtime data handles (UPC++ vs GA)
// ---------------------------------------------------------------------------
#if defined(USE_UPCXX)
using rtDataHandlePtr = upcxx::future<>*;
using rtDataHandle    = upcxx::future<>;
#else
using rtDataHandlePtr = ga_nbhdl_t*;
using rtDataHandle    = ga_nbhdl_t;
#endif

class DataCommunicationHandle {
public:
  DataCommunicationHandle()  = default;
  ~DataCommunicationHandle() = default;

  void waitForCompletion() {
    if (!getCompletionStatus()) {
#if defined(USE_UPCXX)
      data_handle_.wait();
#else
      NGA_NbWait(&data_handle_);
#endif
      setCompletionStatus();
    }
  }
  void setCompletionStatus()   noexcept { status_ = true;  }
  void resetCompletionStatus() noexcept { status_ = false; }
  [[nodiscard]] bool getCompletionStatus() const noexcept { return status_; }
  rtDataHandlePtr getDataHandlePtr() noexcept { return &data_handle_; }

  rtDataHandle data_handle_;
private:
  bool status_{true};
};

using DataCommunicationHandlePtr = DataCommunicationHandle*;

// ---------------------------------------------------------------------------
// MPI type helpers
// ---------------------------------------------------------------------------
#if !defined(USE_UPCXX)
template<typename T>
[[nodiscard]] static inline MPI_Datatype mpi_type() {
  using std::is_same_v;
  if      constexpr (is_same_v<int,T>)                   return MPI_INT;
  else if constexpr (is_same_v<bool,T>)                  return MPI_C_BOOL;
  else if constexpr (is_same_v<char,T>)                  return MPI_CHAR;
  else if constexpr (is_same_v<int64_t,T>)               return MPI_INT64_T;
  else if constexpr (is_same_v<uint32_t,T>)              return MPI_UNSIGNED;
  else if constexpr (is_same_v<size_t,T>)                return MPI_UNSIGNED_LONG;
  else if constexpr (is_same_v<float,T>)                 return MPI_FLOAT;
  else if constexpr (is_same_v<double,T>)                return MPI_DOUBLE;
  else if constexpr (is_same_v<std::complex<float>,T>)   return MPI_COMPLEX;
  else if constexpr (is_same_v<std::complex<double>,T>)  return MPI_DOUBLE_COMPLEX;
  else NOT_IMPLEMENTED();
}

template<typename T>
[[nodiscard]] static inline MPI_Datatype mpi_type_loc() {
  using std::is_same_v;
  if      constexpr (is_same_v<int,T>)    return MPI_2INT;
  else if constexpr (is_same_v<float,T>)  return MPI_2REAL;
  else if constexpr (is_same_v<double,T>) return MPI_2DOUBLE_PRECISION;
  else NOT_IMPLEMENTED();
}

[[nodiscard]] static inline MPI_Op mpi_op(ReduceOp rop) {
  switch (rop) {
    case ReduceOp::min:    return MPI_MIN;
    case ReduceOp::max:    return MPI_MAX;
    case ReduceOp::sum:    return MPI_SUM;
    case ReduceOp::minloc: return MPI_MINLOC;
    case ReduceOp::maxloc: return MPI_MAXLOC;
    default: NOT_IMPLEMENTED();
  }
}
#endif

// ---------------------------------------------------------------------------
// make_label: thread-safe monotonic label generator
// ---------------------------------------------------------------------------
inline Label make_label() {
  static std::atomic<Label> lbl{0};
  return lbl.fetch_add(1, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// GA element type conversions
// ---------------------------------------------------------------------------
[[nodiscard]] static inline int to_ga_eltype(ElementType eltype) noexcept {
  switch (eltype) {
    case ElementType::single_precision: return C_FLOAT;
    case ElementType::double_precision: return C_DBL;
    case ElementType::single_complex:   return C_SCPL;
    case ElementType::double_complex:   return C_DCPL;
    default: return 0;
  }
}

[[nodiscard]] static inline ElementType from_ga_eltype(int eltype) noexcept {
  switch (eltype) {
    case C_FLOAT: return ElementType::single_precision;
    case C_DBL:   return ElementType::double_precision;
    case C_SCPL:  return ElementType::single_complex;
    case C_DCPL:  return ElementType::double_complex;
    default:      return ElementType::invalid;
  }
}

[[nodiscard]] inline constexpr const char* element_type_to_string(ElementType eltype) noexcept {
  switch (eltype) {
    case ElementType::invalid:          return "inv";
    case ElementType::single_precision: return "f32";
    case ElementType::double_precision: return "f64";
    case ElementType::single_complex:   return "cf32";
    case ElementType::double_complex:   return "cf64";
  }
  return "NaN";
}

// ---------------------------------------------------------------------------
// ElType (compact bitfield encoding)
// ---------------------------------------------------------------------------
enum class ElType : uint8_t {
  inv   = 0b1000,
  i32   = 0b0000,
  i64   = 0b0001,
  fp32  = 0b0010,
  fp64  = 0b0011,
  cfp32 = 0b0110,
  cfp64 = 0b0111
};

template<typename T> inline constexpr ElType eltype = ElType::inv;
template<> inline constexpr ElType eltype<int32_t>              = ElType::i32;
template<> inline constexpr ElType eltype<int64_t>              = ElType::i64;
template<> inline constexpr ElType eltype<float>                = ElType::fp32;
template<> inline constexpr ElType eltype<double>               = ElType::fp64;
template<> inline constexpr ElType eltype<std::complex<float>>  = ElType::cfp32;
template<> inline constexpr ElType eltype<std::complex<double>> = ElType::cfp64;

[[nodiscard]] inline constexpr const char* eltype_to_string(ElType et) noexcept {
  switch (et) {
    case ElType::inv:   return "inv";
    case ElType::i32:   return "i32";
    case ElType::i64:   return "i64";
    case ElType::fp32:  return "f32";
    case ElType::fp64:  return "f64";
    case ElType::cfp32: return "cf32";
    case ElType::cfp64: return "cf64";
  }
  return "NaN";
}

[[nodiscard]] inline constexpr ElType lub(ElType a, ElType b) noexcept {
  return ElType(static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
}

// ---------------------------------------------------------------------------
// Overloaded visitor helper
// ---------------------------------------------------------------------------
template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

// ---------------------------------------------------------------------------
// Miscellaneous
// ---------------------------------------------------------------------------
using SymbolTable    = std::map<void*, std::string>;
using TranslateFunc  = std::function<Index(Index)>;

// Hash/equality for IndexVector (now BoundVec-based, stack-allocated).
struct IndexVectorHash {
  [[nodiscard]] std::size_t operator()(const IndexVector& vec) const noexcept {
    std::size_t seed = vec.size();
    for (Index v : vec) internal::hash_combine(seed, v);
    return seed;
  }
};

struct IndexVectorEqual {
  [[nodiscard]] bool operator()(const IndexVector& lhs,
                                const IndexVector& rhs) const noexcept {
    return lhs == rhs;
  }
};

} // namespace tamm
