// Copyright 2016 Pacific Northwest National Laboratory

#ifndef TAMM_TYPES_HPP_
#define TAMM_TYPES_HPP_

#include "tamm/boundvec.hpp"
#include "tamm/errors.hpp"
#include "tamm/strong_num.hpp"
#include <complex>
#include <iosfwd>

//#include <mpi.h>

namespace tamm {

// Free functions
#if __cplusplus < 201703L
namespace internal {
template<class F, class Tuple, std::size_t... I>
constexpr decltype(auto) apply_impl(F&& f, Tuple&& t,
                                    std::index_sequence<I...>) {
    return f(std::get<I>(std::forward<Tuple>(t))...);
}
} // namespace detail

template<class F, class Tuple>
constexpr decltype(auto) apply(F&& f, Tuple&& t) {
    return internal::apply_impl(
      std::forward<F>(f), std::forward<Tuple>(t),
      std::make_index_sequence<std::tuple_size<std::decay_t<Tuple>>::value>{});
}
#endif
// End Free functions


// IndexSpace related type definitions
using Index         = uint32_t;
using IndexVector   = std::vector<Index>;
using IndexIterator = std::vector<Index>::const_iterator;
using Tile          = uint32_t;
// DAG related Hash
using HashData      = uint64_t;

//////////////////////////////////
struct IrrepSpace;
using Irrep = StrongNum<IrrepSpace, uint32_t>;
struct SpinSpace;
using Spin = StrongNum<SpinSpace, uint32_t>;
struct SpatialSpace;
using Spatial    = StrongNum<SpatialSpace, uint32_t>;
using TensorRank = size_t;
struct OffsetSpace;
using Offset = StrongNum<OffsetSpace, uint64_t>;
struct BlockIndexSpace;
using BlockIndex = StrongNum<BlockIndexSpace, uint64_t>;
struct ProcSpace;
using Proc = StrongNum<ProcSpace, int64_t>;
struct SignSpace;
using Sign = StrongNum<SignSpace, int32_t>;

// these are typedefs for usability
using Size       = Offset;
using BlockCount = BlockIndex;
using Label      = int;

enum class AllocationStatus { invalid, created, attached };

enum class ElementType {
    invalid,
    single_precision,
    double_precision,
    single_complex,
    double_complex
};

template<typename T>
constexpr ElementType tensor_element_type() {
    return ElementType::invalid;
}

template<>
constexpr ElementType tensor_element_type<double>() {
    return ElementType::double_precision;
}

template<>
constexpr ElementType tensor_element_type<float>() {
    return ElementType::single_precision;
}

template<>
constexpr ElementType tensor_element_type<std::complex<float>>() {
    return ElementType::single_complex;
}

template<>
constexpr ElementType tensor_element_type<std::complex<double>>() {
    return ElementType::double_complex;
}

static inline constexpr size_t element_size(ElementType eltype) {
    size_t ret = 0;
    switch(eltype) {
        case ElementType::single_precision: ret = sizeof(float); break;
        case ElementType::double_precision: ret = sizeof(double); break;
        case ElementType::single_complex:
            ret = sizeof(std::complex<float>);
            break;
        case ElementType::double_complex:
            ret = sizeof(std::complex<double>);
            break;
        default: UNREACHABLE();
    }
    return ret;
}

// enum class DimType {
//   o  = 0b0011,
//   v  = 0b1100,
//   oa = 0b0001,
//   ob = 0b0010,
//   va = 0b0100,
//   vb = 0b1000,
//   n  = 0b1111,
//   c  = 0b0000,
//   inv = 0b11111,
// };

// inline bool
// is_dim_subset(DimType superset, DimType subset) {
//   auto sup = static_cast<unsigned>(superset);
//   auto sub = static_cast<unsigned>(subset);
//   return (sup & sub) == sub;
// }

// inline std::string
// to_string(DimType dt) {
//   switch(dt) {
//     case DimType::o:
//       return "o";
//       break;
//     case DimType::v:
//       return "v";
//       break;
//     case DimType::oa:
//       return "oa";
//       break;
//     case DimType::ob:
//       return "ob";
//       break;
//     case DimType::va:
//       return "va";
//       break;
//     case DimType::vb:
//       return "vb";
//       break;
//     case DimType::n:
//       return "n";
//       break;
//     default:
//       UNREACHABLE();
//   }
// }

// inline std::ostream&
// operator << (std::ostream& os, DimType dt) {
//   os << to_string(dt);
//   return os;
// }

const TensorRank maxrank{8};

template<typename T>
using TensorVec = BoundVec<T, maxrank>;

using BlockDimVec = TensorVec<BlockIndex>;
// using DimTypeVec = TensorVec<DimType>;
using PermVec = TensorVec<uint32_t>;

///////////

///////////////////

using RangeValue = int64_t;

// enum class IndexSpaceType { mo, mso, ao, aso, aux };
enum class SpinMask { ignore, upper, lower };
enum class IndexPosition { upper, lower, neither };

//////////////////

namespace SpinType {
const Spin alpha{1};
const Spin beta{2};
}; // namespace SpinType

// /**
//  * @brief Signature of a tensor addition operation in NWChem TCE
//  */
// typedef void add_fn(MPI_Fint *ta, MPI_Fint *offseta,
//                     MPI_Fint *tc, MPI_Fint *offsetc);

// /**
//  * @brief Signature of a tensor addition operation in NWChem TCE
//  */
// typedef void mult_fn(MPI_Fint *ta, MPI_Fint *offseta,
//                      MPI_Fint *tb, MPI_Fint *offsetb,
//                      MPI_Fint *tc, MPI_Fint *offsetc);

} // namespace tamm

#endif // TAMM_TYPES_HPP_
