// Copyright 2016 Pacific Northwest National Laboratory
// C++20 modernization: concepts, operator<=>, [[nodiscard]], requires-clauses.

#pragma once

#include <cassert>
#include <compare>    // std::strong_ordering, operator<=>
#include <concepts>   // std::integral, std::floating_point
#include <functional> // std::hash
#include <iosfwd>
#include <limits>
#include <vector>

namespace tamm {

/**
 * @todo Check the narrow cast implementation in:
 *  http://stackoverflow.com/questions/17860657/well-defined-narrowing-cast
 */

/**
 * @todo Add debug mode checks for overflow and underflow
 */

// ---------------------------------------------------------------------------
// Concept: StrongNumeric
// Constrains what underlying types StrongNum may wrap.
// ---------------------------------------------------------------------------
template<typename T>
concept StrongNumeric = std::integral<T> || std::floating_point<T>;

// ---------------------------------------------------------------------------
// checked_cast: narrow cast with optional debug assertion.
// ---------------------------------------------------------------------------

/**
 * @brief Return input value in the desired target type after possibly checking
 * for overflow/underflow.
 *
 * @tparam Target Desired output type
 * @tparam Source Input type
 * @param s Value being type cast
 * @return Value after type casting
 */
template<StrongNumeric Target, StrongNumeric Source>
requires(!std::is_same_v<Target, Source>) constexpr Target checked_cast(Source s) noexcept(false) {
  auto r = static_cast<Target>(s);
#if defined(TAMM_DEBUG_STRONGNUM)
  assert(static_cast<Source>(r) == s && "checked_cast: narrowing lost data");
#endif
  return r;
}

/**
 * @brief Trivial checked cast when casting to the same type
 *
 * @tparam T Value type
 * @param s input value
 * @return same input value
 */
template<StrongNumeric T>
constexpr T checked_cast(T s) noexcept {
  return s;
}

/**
 * @brief Strongly typed wrapper for a numeric type.
 *
 * This class provides a strongly typed alias that cannot be implicitly
 * converted to another type. To define a new StrongNum wrapper StrongType to,
 * say, int, we do the following:
 *
 * @code
 * class StrongIntSpace;
 * using StrongInt = StrongNum<StrongIntSpace, int>;
 * @endcode
 *
 * Checked casts are to be used to convert between types and possibly check the
 * conversions in debug mode.
 *
 * C++20 changes vs. original:
 *  - StrongNumeric concept replaces enable_if arithmetic checks.
 *  - 12 comparison operators replaced by operator<=> (spaceship).
 *  - [[nodiscard]] applied to value() and all binary operators.
 *  - requires-clause replaces enable_if on constructor.
 *
 * @tparam Space Unique type name
 * @tparam T Numeric type being wrapped
 */
template<typename Space, StrongNumeric T>
struct StrongNum {
  /**
   * @brief Type of wrapper number
   *
   */
  using value_type = T;

  /**
   * @brief Alias for this StrongNum type
   *
   */
  using NumType = StrongNum<Space, T>;

  // ---- Lifecycle --------------------------------------------------------
  StrongNum()                            = default;
  StrongNum(const StrongNum&)            = default;
  StrongNum& operator=(const StrongNum&) = default;
  ~StrongNum()                           = default;

  /// Implicit construction from any compatible arithmetic type.
  ///
  /// NOTE: this is intentionally NOT explicit.  TAMM relies pervasively on
  /// implicit conversions such as `Proc p = GA_Nodeid();`, `Offset off = 0;`,
  /// and `Size sz = block_size(...)`.  Making it explicit breaks hundreds of
  /// call sites, so the original (implicit) behaviour is preserved here.
  template<StrongNumeric T2>
  requires std::is_convertible_v<T2, T>
  constexpr StrongNum(T2 v1) noexcept: v{checked_cast<T>(v1)} {}

  // ---- Assignment from raw arithmetic -----------------------------------
  template<StrongNumeric T2>
  NumType& operator=(T2 t) noexcept {
    v = checked_cast<T>(t);
    return *this;
  }

  // ---- Compound assignment (NumType operand) ----------------------------
  NumType& operator+=(NumType d) noexcept {
    v += d.v;
    return *this;
  }
  NumType& operator-=(NumType d) noexcept {
    v -= d.v;
    return *this;
  }
  NumType& operator*=(NumType d) noexcept {
    v *= d.v;
    return *this;
  }
  NumType& operator/=(NumType d) noexcept {
    v /= d.v;
    return *this;
  }
  NumType& operator%=(NumType d) noexcept {
    v %= d.v;
    return *this;
  }
  NumType& operator^=(NumType d) noexcept {
    v ^= d.v;
    return *this;
  }

  // ---- Compound assignment (raw arithmetic operand) ---------------------
  template<StrongNumeric T2>
  NumType& operator+=(T2 t) noexcept {
    v += checked_cast<T>(t);
    return *this;
  }
  template<StrongNumeric T2>
  NumType& operator-=(T2 t) noexcept {
    v -= checked_cast<T>(t);
    return *this;
  }
  template<StrongNumeric T2>
  NumType& operator*=(T2 t) noexcept {
    v *= checked_cast<T>(t);
    return *this;
  }
  template<StrongNumeric T2>
  NumType& operator/=(T2 t) noexcept {
    v /= checked_cast<T>(t);
    return *this;
  }
  template<StrongNumeric T2>
  NumType& operator^=(T2 t) noexcept {
    v ^= checked_cast<T>(t);
    return *this;
  }

  // ---- Increment / decrement -------------------------------------------
  NumType& operator++() noexcept {
    v += 1;
    return *this;
  }
  NumType operator++(int) noexcept {
    NumType ret{*this};
    v += 1;
    return ret;
  }
  NumType& operator--() noexcept {
    v -= 1;
    return *this;
  }
  NumType operator--(int) noexcept {
    NumType ret{*this};
    v -= 1;
    return ret;
  }

  // ---- Binary arithmetic (NumType operands) ----------------------------
  [[nodiscard]] NumType operator+(NumType d) const noexcept { return NumType{v + d.v}; }
  [[nodiscard]] NumType operator-(NumType d) const noexcept { return NumType{v - d.v}; }
  [[nodiscard]] NumType operator*(NumType d) const noexcept { return NumType{v * d.v}; }
  [[nodiscard]] NumType operator/(NumType d) const noexcept { return NumType{v / d.v}; }
  [[nodiscard]] NumType operator%(NumType d) const noexcept { return NumType{v % d.v}; }

  // ---- Binary arithmetic (raw arithmetic operands) ---------------------
  template<StrongNumeric T2>
  [[nodiscard]] NumType operator+(T2 t) const noexcept {
    return NumType{v + checked_cast<T>(t)};
  }
  template<StrongNumeric T2>
  [[nodiscard]] NumType operator-(T2 t) const noexcept {
    return NumType{v - checked_cast<T>(t)};
  }
  template<StrongNumeric T2>
  [[nodiscard]] NumType operator*(T2 t) const noexcept {
    return NumType{v * checked_cast<T>(t)};
  }
  template<StrongNumeric T2>
  [[nodiscard]] NumType operator/(T2 t) const noexcept {
    return NumType{v / checked_cast<T>(t)};
  }
  template<StrongNumeric T2>
  [[nodiscard]] NumType operator%(T2 t) const noexcept {
    return NumType{v % checked_cast<T>(t)};
  }

  // ---- Comparison (same-type) -------------------------------------------
  // Defaulted operator== AND operator<=>.  operator<=> alone does NOT
  // synthesize ==/!=, so both are needed; == synthesizes != and <=>
  // synthesizes < <= > >=.
  [[nodiscard]] bool operator==(const NumType& d) const noexcept  = default;
  [[nodiscard]] auto operator<=>(const NumType& d) const noexcept = default;

  // Heterogeneous comparisons against raw arithmetic (non-defaulted).
  template<StrongNumeric T2>
  [[nodiscard]] bool operator==(T2 t) const noexcept {
    return v == checked_cast<T>(t);
  }
  template<StrongNumeric T2>
  [[nodiscard]] auto operator<=>(T2 t) const noexcept {
    return v <=> checked_cast<T>(t);
  }

  // ---- Value accessors --------------------------------------------------
  [[nodiscard]] T value() const noexcept { return v; }
  T&              value() noexcept { return v; }

private:
  T v{}; /**< Value wrapped by this object */
};

// ---------------------------------------------------------------------------
// strongnum_cast: cross-space conversion
// ---------------------------------------------------------------------------
template<typename Space1, StrongNumeric T1, typename Space2, StrongNumeric T2>
[[nodiscard]] inline StrongNum<Space1, T1> strongnum_cast(StrongNum<Space2, T2> v2) noexcept {
  return StrongNum<Space1, T1>{checked_cast<T1>(v2.value())};
}

// ---------------------------------------------------------------------------
// Non-member operators: raw-lhs op StrongNum-rhs
// (Needed because the member operators only cover StrongNum-lhs.)
// ---------------------------------------------------------------------------
template<typename Space, StrongNumeric Type, StrongNumeric T>
[[nodiscard]] inline bool operator==(T val, StrongNum<Space, Type> sn) noexcept {
  return sn == val;
}
template<typename Space, StrongNumeric Type, StrongNumeric T>
[[nodiscard]] inline auto operator<=>(T val, StrongNum<Space, Type> sn) noexcept {
  return checked_cast<Type>(val) <=> sn.value();
}

template<typename Space, StrongNumeric Type, StrongNumeric T>
[[nodiscard]] inline StrongNum<Space, Type> operator-(T v, StrongNum<Space, Type> sn) noexcept {
  return StrongNum<Space, Type>{checked_cast<Type>(v) - sn.value()};
}
template<typename Space, StrongNumeric Type, StrongNumeric T>
[[nodiscard]] inline StrongNum<Space, Type> operator+(T v, StrongNum<Space, Type> sn) noexcept {
  return StrongNum<Space, Type>{checked_cast<Type>(v) + sn.value()};
}
template<typename Space, StrongNumeric Type, StrongNumeric T>
[[nodiscard]] inline StrongNum<Space, Type> operator*(T v, StrongNum<Space, Type> sn) noexcept {
  return StrongNum<Space, Type>{checked_cast<Type>(v) * sn.value()};
}
template<typename Space, StrongNumeric Type, StrongNumeric T>
[[nodiscard]] inline StrongNum<Space, Type> operator/(T v, StrongNum<Space, Type> sn) noexcept {
  return StrongNum<Space, Type>{checked_cast<Type>(v) / sn.value()};
}
template<typename Space, StrongNumeric Type, StrongNumeric T>
[[nodiscard]] inline StrongNum<Space, Type> operator%(T v, StrongNum<Space, Type> sn) noexcept {
  return StrongNum<Space, Type>{checked_cast<Type>(v) % sn.value()};
}

// ---------------------------------------------------------------------------
// I/O
// ---------------------------------------------------------------------------
template<typename S, StrongNumeric T>
inline std::ostream& operator<<(std::ostream& os, StrongNum<S, T> s) {
  return os << s.value();
}
template<typename S, StrongNumeric T>
inline std::istream& operator>>(std::istream& is, StrongNum<S, T>& s) {
  is >> s.value();
  return is;
}

} // namespace tamm

// ---------------------------------------------------------------------------
// std::hash specialization so StrongNum can be used in unordered containers.
// ---------------------------------------------------------------------------
namespace std {
template<typename Space, tamm::StrongNumeric T>
struct hash<tamm::StrongNum<Space, T>> {
  [[nodiscard]] size_t operator()(tamm::StrongNum<Space, T> s) const noexcept {
    return std::hash<T>{}(s.value());
  }
};
} // namespace std
