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
   

template<typename Space, typename T>
struct StrongInt {
  using value_type = T;
  using IntType =  StrongInt<Space, T>;

  StrongInt() = default;
  StrongInt(const StrongInt<Space, T>&) = default;
  StrongInt& operator=(const StrongInt&) = default;
  ~StrongInt() = default;

  explicit StrongInt(const T v1): v(v1) {}
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

#define DEBUG

template<typename Target, typename Source>
Target strongint_cast(const Source& s) {
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

template<typename S1, typename T1, typename S2, typename T2>
StrongInt<S2, T2> strongint_cast(const StrongInt<S1, T1>& s) {
  return StrongInt<S2, T2>{strongint_cast<T2>(s.value())};
}

template<typename Space, typename Int, typename Int2>
StrongInt<Space,Int> operator * (Int2 value, StrongInt<Space,Int> sint) {
  return StrongInt<Space,Int>{strongint_cast<Int>(sint.value() * value)};
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

