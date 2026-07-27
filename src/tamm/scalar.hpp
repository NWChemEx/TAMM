#pragma once

#include <complex>
#include <iomanip>
#include <iostream>
#include <type_traits>
#include <variant>

#include "tamm/types.hpp"

namespace tamm {

namespace detail {
// Local, dependency-free complex trait (avoids pulling in the heavyweight
// tamm/utils.hpp just for internal::is_complex_v).
template<typename T>
struct is_complex: std::false_type {};
template<typename T>
struct is_complex<std::complex<T>>: std::true_type {};
template<typename T>
inline constexpr bool is_complex_v = is_complex<T>::value;
} // namespace detail

/**
 * @brief Scalar object for representing scalar values from different data types
 *
 */
class Scalar {
public:
  using ElementType =
    std::variant<int, int64_t, float, double, std::complex<float>, std::complex<double>>;
  Scalar() = default;
  template<typename T, std::enable_if_t<std::is_arithmetic<T>::value, int> = 0>
  Scalar(T value): value_{value} {}

  Scalar(ElType eltype) {
    switch(eltype) {
      case ElType::inv: value_ = 1.0; break;
      case ElType::i32: value_ = 1; break;
      case ElType::i64: value_ = int64_t{1}; break;
      case ElType::fp32: value_ = 1.0f; break;
      case ElType::fp64: value_ = 1.0; break;
      case ElType::cfp32: value_ = std::complex<float>(1.0, 0.0); break;
      case ElType::cfp64: value_ = std::complex<double>(1.0, 0.0); break;
    }
  }

  template<typename T, std::enable_if_t<std::is_arithmetic<T>::value, int> = 0>

  Scalar(const std::complex<T>& value): value_{value} {}
  Scalar(const Scalar&)  = default;
  Scalar(Scalar&& other) = default;

  Scalar& operator=(Scalar other) noexcept {
    using std::swap;
    swap(*this, other);
    return *this;
  }

  friend void swap(Scalar& first, Scalar& second) noexcept {
    using std::swap;
    // swap(static_cast<Op&>(first), static_cast<Op&>(second));
    swap(first.value_, second.value_);
  }

  ElType eltype() const {
    return std::visit(
      overloaded{
        [&](int32_t e) { return ElType::i32; }, [&](int64_t e) { return ElType::i64; },
        [&](float e) { return ElType::fp32; }, [&](double e) { return ElType::fp64; },
        [&](std::complex<float> e) { return ElType::cfp32; },
        [&](std::complex<double> e) { return ElType::cfp64; },
        //[&](auto e) { return ElType::fp32; /*erroeneous case*/ }
      },
      value_);
  }

  std::string to_string() const {
    using namespace std::string_literals;
    auto fn = [](auto e) {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(2) << e;
      return oss.str();
    };
    std::string str = std::visit(
      overloaded{[&](auto e) { return fn(e); },
                 [&](std::complex<float> e) { return fn(e.real()) + "+" + fn(e.imag()) + "i"; },
                 [&](std::complex<double> e) { return fn(e.real()) + "+" + fn(e.imag()) + "i"; }},
      value_);
    return str;
  }

  ElementType value() const { return value_; }

  /**
   * @brief Extract the stored value converted to the requested element type T.
   *
   * Visits the underlying variant and converts the active alternative to T.
   * Real target types drop the imaginary part of a stored complex value;
   * complex target types are constructed from real alternatives.  This is the
   * type-safe replacement for `static_cast<T>(scalar)`, which is ill-formed
   * because Scalar holds a std::variant with no implicit numeric conversion.
   */
  template<typename T>
  [[nodiscard]] T get() const {
    return std::visit(
      [](auto v) -> T {
        using V = std::decay_t<decltype(v)>;
        if constexpr(detail::is_complex_v<T>) {
          if constexpr(detail::is_complex_v<V>)
            return T{static_cast<typename T::value_type>(v.real()),
                     static_cast<typename T::value_type>(v.imag())};
          else return T{static_cast<typename T::value_type>(v)};
        }
        else {
          if constexpr(detail::is_complex_v<V>) return static_cast<T>(v.real());
          else return static_cast<T>(v);
        }
      },
      value_);
  }

private:
  ElementType value_;
};

/**
 * @brief Complex Type operator overloads for different data types
 *
 */
///////////////// Multiplication Operator Overloads for std::complex
////////////////////

inline std::complex<double> operator*(const std::complex<double>& c1,
                                      const std::complex<float>&  c2) {
  return {c1.real() * c2.real() - c1.imag() * c2.imag(),
          c1.real() * c2.imag() + c1.imag() * c2.real()};
}

inline std::complex<double> operator*(const std::complex<float>&  c1,
                                      const std::complex<double>& c2) {
  return {c1.real() * c2.real() - c1.imag() * c2.imag(),
          c1.real() * c2.imag() + c1.imag() * c2.real()};
}

template<typename T, std::enable_if_t<std::is_integral<T>::value, int> = 0>
inline std::complex<double> operator*(const std::complex<double>& c1, T v) {
  return {c1.real() * v, c1.imag() * v};
}

template<typename T, std::enable_if_t<std::is_integral<T>::value, int> = 0>
inline std::complex<double> operator*(T v, const std::complex<double>& c1) {
  return {c1.real() * v, c1.imag() * v};
}

inline std::complex<double> operator*(const std::complex<double>& c1, float v) {
  return {c1.real() * v, c1.imag() * v};
}

inline std::complex<double> operator*(float v, const std::complex<double>& c1) {
  return {c1.real() * v, c1.imag() * v};
}

template<typename T, std::enable_if_t<std::is_integral<T>::value, int> = 0>
inline std::complex<float> operator*(const std::complex<float>& c1, T v) {
  return {c1.real() * v, c1.imag() * v};
}

template<typename T, std::enable_if_t<std::is_integral<T>::value, int> = 0>
inline std::complex<float> operator*(T v, const std::complex<float>& c1) {
  return {c1.real() * v, c1.imag() * v};
}

inline std::complex<double> operator*(const std::complex<float>& c1, double v) {
  return {c1.real() * v, c1.imag() * v};
}

inline std::complex<double> operator*(double v, const std::complex<float>& c1) {
  return {c1.real() * v, c1.imag() * v};
}

///////////////// Addition Operator Overloads for std::complex /////////////////

inline std::complex<double> operator+(const std::complex<double>& c1,
                                      const std::complex<float>&  c2) {
  return {c1.real() + c2.real(), c1.imag() + c2.imag()};
}

inline std::complex<double> operator+(const std::complex<float>&  c1,
                                      const std::complex<double>& c2) {
  return {c1.real() + c2.real(), c1.imag() + c2.imag()};
}

template<typename T, std::enable_if_t<std::is_integral<T>::value, int> = 0>
inline std::complex<double> operator+(const std::complex<double>& c1, T v) {
  return {c1.real() + v, c1.imag()};
}

template<typename T, std::enable_if_t<std::is_integral<T>::value, int> = 0>
inline std::complex<double> operator+(T v, const std::complex<double>& c1) {
  return {c1.real() + v, c1.imag()};
}

inline std::complex<double> operator+(const std::complex<double>& c1, float v) {
  return {c1.real() + v, c1.imag()};
}

inline std::complex<double> operator+(float v, const std::complex<double>& c1) {
  return {c1.real() + v, c1.imag()};
}

template<typename T, std::enable_if_t<std::is_integral<T>::value, int> = 0>
inline std::complex<float> operator+(const std::complex<float>& c1, T v) {
  return {c1.real() + v, c1.imag()};
}

template<typename T, std::enable_if_t<std::is_integral<T>::value, int> = 0>
inline std::complex<float> operator+(T v, const std::complex<float>& c1) {
  return {c1.real() + v, c1.imag()};
}

inline std::complex<double> operator+(const std::complex<float>& c1, double v) {
  return {c1.real() + v, c1.imag()};
}

inline std::complex<double> operator+(double v, const std::complex<float>& c1) {
  return {c1.real() + v, c1.imag()};
}

///////////////// Subtraction Operator Overloads for std::complex
////////////////////

inline std::complex<double> operator-(const std::complex<double>& c1,
                                      const std::complex<float>&  c2) {
  return {c1.real() - c2.real(), c1.imag() - c2.imag()};
}

inline std::complex<double> operator-(const std::complex<float>&  c1,
                                      const std::complex<double>& c2) {
  return {c1.real() - c2.real(), c1.imag() - c2.imag()};
}

template<typename T, std::enable_if_t<std::is_integral<T>::value, int> = 0>
inline std::complex<double> operator-(const std::complex<double>& c1, T v) {
  return {c1.real() - v, c1.imag()};
}

template<typename T, std::enable_if_t<std::is_integral<T>::value, int> = 0>
inline std::complex<double> operator-(T v, const std::complex<double>& c1) {
  return {c1.real() - v, c1.imag()};
}

inline std::complex<double> operator-(const std::complex<double>& c1, float v) {
  return {c1.real() - v, c1.imag()};
}

inline std::complex<double> operator-(float v, const std::complex<double>& c1) {
  return {c1.real() - v, c1.imag()};
}

template<typename T, std::enable_if_t<std::is_integral<T>::value, int> = 0>
inline std::complex<float> operator-(const std::complex<float>& c1, T v) {
  return {c1.real() - v, c1.imag()};
}

template<typename T, std::enable_if_t<std::is_integral<T>::value, int> = 0>
inline std::complex<float> operator-(T v, const std::complex<float>& c1) {
  return {c1.real() - v, c1.imag()};
}

inline std::complex<double> operator-(const std::complex<float>& c1, double v) {
  return {c1.real() - v, c1.imag()};
}

inline std::complex<double> operator-(double v, const std::complex<float>& c1) {
  return {c1.real() - v, c1.imag()};
}

///////////////// Division Operator Overloads for std::complex /////////////////

inline std::complex<double> operator/(const std::complex<double>& c1,
                                      const std::complex<float>&  c2) {
  return c1 * std::conj(c2) / static_cast<double>(std::norm(c2));
}

inline std::complex<double> operator/(const std::complex<float>&  c1,
                                      const std::complex<double>& c2) {
  return std::conj(c2) * c1 / std::norm(c2);
}

template<typename T, std::enable_if_t<std::is_integral<T>::value, int> = 0>
inline std::complex<double> operator/(const std::complex<double>& c1, T v) {
  return {c1.real() / v, c1.imag() / v};
}

template<typename T, std::enable_if_t<std::is_integral<T>::value, int> = 0>
inline std::complex<double> operator/(T v, const std::complex<double>& c1) {
  return {c1.real() / v, c1.imag() / v};
}

inline std::complex<double> operator/(const std::complex<double>& c1, float v) {
  return {c1.real() / v, c1.imag() / v};
}

inline std::complex<double> operator/(float v, const std::complex<double>& c1) {
  return {c1.real() / v, c1.imag() / v};
}

template<typename T, std::enable_if_t<std::is_integral<T>::value, int> = 0>
inline std::complex<float> operator/(const std::complex<float>& c1, T v) {
  return {c1.real() / v, c1.imag() / v};
}

template<typename T, std::enable_if_t<std::is_integral<T>::value, int> = 0>
inline std::complex<float> operator/(T v, const std::complex<float>& c1) {
  return {c1.real() / v, c1.imag() / v};
}

inline std::complex<double> operator/(const std::complex<float>& c1, double v) {
  return {c1.real() / v, c1.imag() / v};
}

inline std::complex<double> operator/(double v, const std::complex<float>& c1) {
  return {c1.real() / v, c1.imag() / v};
}

///////////////////////////////////////////////////////////////////////////

/**
 * @brief Operator overloads for Scalar class
 *
 */

inline std::ostream& operator<<(std::ostream& os, Scalar v) {
  std::visit([&](auto e) { os << e; }, v.value());
  return os;
}

inline Scalar operator*(const Scalar& v1, const Scalar& v2) {
  return std::visit(overloaded{[&](const auto& e1) {
                      return std::visit(overloaded{[&](const auto& e2) { return Scalar{e1 * e2}; }},
                                        v2.value());
                    }},
                    v1.value());
}

template<typename T, std::enable_if_t<std::is_arithmetic<T>::value, int> = 0>
inline Scalar operator*(const Scalar& v1, T v) {
  return std::visit(overloaded{[&](const auto& e1) { return Scalar{e1 * v}; }}, v1.value());
}

template<typename T, std::enable_if_t<std::is_arithmetic<T>::value, int> = 0>
inline Scalar operator*(T v, const Scalar& v1) {
  return std::visit(overloaded{[&](const auto& e1) { return Scalar{e1 * v}; }}, v1.value());
}

inline Scalar operator+(const Scalar& v1, const Scalar& v2) {
  return std::visit(overloaded{[&](const auto& e1) {
                      return std::visit(overloaded{[&](const auto& e2) { return Scalar{e1 + e2}; }},
                                        v2.value());
                    }},
                    v1.value());
}

inline Scalar operator-(const Scalar& v1, const Scalar& v2) {
  return std::visit(overloaded{[&](const auto& e1) {
                      return std::visit(overloaded{[&](const auto& e2) { return Scalar{e1 - e2}; }},
                                        v2.value());
                    }},
                    v1.value());
}

template<typename T, std::enable_if_t<std::is_arithmetic<T>::value, int> = 0>
inline Scalar operator+(const Scalar& v1, T v) {
  return std::visit(overloaded{[&](const auto& e1) { return Scalar{e1 + v}; }}, v1.value());
}

template<typename T, std::enable_if_t<std::is_arithmetic<T>::value, int> = 0>
inline Scalar operator+(T v, const Scalar& v1) {
  return std::visit(overloaded{[&](const auto& e1) { return Scalar{e1 + v}; }}, v1.value());
}

template<typename T, std::enable_if_t<std::is_arithmetic<T>::value, int> = 0>
inline Scalar operator-(const Scalar& v1, T v) {
  return std::visit(overloaded{[&](const auto& e1) { return Scalar{e1 - v}; }}, v1.value());
}

template<typename T, std::enable_if_t<std::is_arithmetic<T>::value, int> = 0>
inline Scalar operator-(T v, const Scalar& v1) {
  return std::visit(overloaded{[&](const auto& e1) { return Scalar{e1 - v}; }}, v1.value());
}

inline Scalar operator/(const Scalar& v1, const Scalar& v2) {
  return std::visit(overloaded{[&](const auto& e1) {
                      return std::visit(overloaded{[&](const auto& e2) { return Scalar{e1 / e2}; }},
                                        v2.value());
                    }},
                    v1.value());
}

template<typename T, std::enable_if_t<std::is_arithmetic<T>::value, int> = 0>
inline Scalar operator/(const Scalar& v1, T v) {
  return std::visit(overloaded{[&](const auto& e1) { return Scalar{e1 / v}; }}, v1.value());
}

template<typename T, std::enable_if_t<std::is_arithmetic<T>::value, int> = 0>
inline Scalar operator/(T v, const Scalar& v1) {
  return std::visit(overloaded{[&](const auto& e1) { return Scalar{v / e1}; }}, v1.value());
}

} // namespace tamm
