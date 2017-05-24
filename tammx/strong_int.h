// Copyright 2016 Pacific Northwest National Laboratory

#ifndef TAMMX_STRONGINT_H__
#define TAMMX_STRONGINT_H__

#include <limits>
#include <iostream>

namespace tammx {


/**
 * @todo Check the narrow cast implementation in:
 *  http://stackoverflow.com/questions/17860657/well-defined-narrowing-cast
 */

#define DEBUG

// template<class T, T v>
// struct integral_constant {
//     static constexpr T value = v;
//     typedef T value_type;
//     typedef integral_constant type; // using injected-class-name
//     constexpr operator value_type() const noexcept { return value; }
//     constexpr value_type operator()() const noexcept { return value; } //since c++14
// };

// template <bool B>
// using bool_constant = integral_constant<bool, B>;

// template<class B>
// struct negation : bool_constant<!bool(B::value)> { };

template<typename Target, typename Source,
         typename = std::enable_if_t<std::is_integral<Source>::value>,
         typename = std::enable_if_t<std::is_integral<Target>::value>>
         // typename = std::enable_if<negation<std::is_same<Target, Source>::value>>>
Target checked_cast(const Source& s) {
  using slimits = std::numeric_limits<Source>;
  using tlimits = std::numeric_limits<Target>;
#if defined(DEBUG)
  if (slimits::max() <= tlimits::max() &&
      slimits::min() >= tlimits::min()) {
    return static_cast<Target>(s);
  } else {
    auto r = static_cast<Target>(s);
    assert(static_cast<Source>(r) == s);
    return r;
  }
#else
  return Target{static_cast<Target>(s)};
#endif
}

// template<typename Source,
//          typename = std::enable_if_t<std::is_integral<Source>::value>>
// Source checked_cast(const Source& s) {
//   return s;
// }



template<typename Space, typename T>
struct StrongInt {
  using value_type = T;
  using IntType =  StrongInt<Space, T>;

  StrongInt() = default;
  StrongInt(const StrongInt<Space, T>&) = default;
  StrongInt& operator=(const StrongInt&) = default;
  ~StrongInt() = default;

  template<typename T2>
  explicit StrongInt(const T2 v1): v(checked_cast<T>(v1)) {}

  // StrongInt<Space, T>& operator += (const T v1) {
  //   v += v1;
  //   return *this;
  // }
  //T operator() () const { return v; }
  //T& operator() () { return v; }

  IntType& operator=(const T& t)        { v = t; return *this; }

  IntType& operator+=(const T& t)       { v += t; return *this; }
  IntType& operator+=(const IntType& d) { v += d.v; return *this; }
  IntType& operator-=(const T& t)       { v -= t; return *this; }
  IntType& operator-=(const IntType& d) { v -= d.v; return *this; }
  IntType& operator*=(const T& t)       { v *= t; return *this; }
  IntType& operator*=(const IntType& d) { v *= d.v; return *this; }
  IntType& operator/=(const T& t)       { v /= t; return *this; }
  IntType& operator/=(const IntType& d) { v /= d.v; return *this; }
  IntType& operator^=(const IntType& d) { v ^= d.v; return *this; }

  IntType& operator++()       { v += 1; return *this; }
  IntType  operator++(int)       { v += 1; return *this; }
  IntType& operator--()       { v -= 1; return *this; }

  IntType operator+(const IntType& d) const { return IntType(v+d.v); }
  IntType operator-(const IntType& d) const { return IntType(v-d.v); }
  IntType operator*(const IntType& d) const { return IntType(v*d.v); }
  IntType operator/(const IntType& d) const { return IntType(v/d.v); }
  IntType operator%(const IntType& d) const { return IntType(v%d.v); }

  IntType operator+(const T& t)       const { return IntType(v+t); }
  IntType operator-(const T& t)       const { return IntType(v-t); }
  IntType operator*(const T& t)       const { return IntType(v*t); }
  IntType operator/(const T& t)       const { return IntType(v/t); }
  IntType operator%(const T& t)       const { return IntType(v%t); }

  bool operator==(const IntType& d) const { return v == d.v; }
  bool operator!=(const IntType& d) const { return v != d.v; }
  bool operator>=(const IntType& d) const { return v >= d.v; }
  bool operator<=(const IntType& d) const { return v <= d.v; }
  bool operator> (const IntType& d) const { return v >  d.v; }
  bool operator< (const IntType& d) const { return v <  d.v; }

  bool operator==(const T& t) const { return v == t; }
  bool operator!=(const T& t) const { return v != t; }
  bool operator>=(const T& t) const { return v >= t; }
  bool operator<=(const T& t) const { return v <= t; }
  bool operator> (const T& t) const { return v >  t; }
  bool operator< (const T& t) const { return v <  t; }

  T value() const { return v; }
  T& value() { return v; }
 private:
  T v;
};

// template<typename Target, typename Source,
//          typename = std::enable_if_t<std::is_integral<Source>::value>,
//          typename = std::enable_if_t<std::is_class<Target>::value>>
// Target strongint_cast(const Source& s) {
//   return Target{strongint_cast<Target::value_type>(s)};
// }

// template<typename Target, typename Source,
//          typename = std::enable_if_t<std::is_class<Source>::value>,
//          typename = std::enable_if_t<std::is_class<Target>::value>>
// Target strongint_cast(const Source& s) {
//   return Target{strongint_cast<Target::value_type>(s.value)};
// }


// template<typename S1, typename T1, typename S2, typename T2>
// StrongInt<S2, T2> strongint_cast(const StrongInt<S1, T1>& s) {
//   return StrongInt<S2, T2>{strongint_cast<T2>(s.value())};
// }

// template<typename S2, typename T2>
// template<typename T1, typename X = StrongInt<S2,T2>>
// StrongInt<S2, T2> strongint_cast(const T1& s) {
//   return StrongInt<S2, T2>{strongint_cast<T2>(s)};
// }

template<typename Space, typename Int, typename Int2>
StrongInt<Space,Int> operator * (Int2 value, StrongInt<Space,Int> sint) {
  return StrongInt<Space,Int>{checked_cast<Int>(sint.value() * value)};
}


template<typename S, typename T>
std::ostream& operator<<(std::ostream& os, const StrongInt<S, T>& s) {
  return os << s.value();
}

template<typename S, typename T>
std::istream& operator>>(std::istream& is, StrongInt<S, T>& s) {
  is >> s.value();
  return is;
}

}  // namespace tammx

#endif  // TAMMX_STRONGINT_H__

