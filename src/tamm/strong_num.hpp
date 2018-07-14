// Copyright 2016 Pacific Northwest National Laboratory

#ifndef TAMM_STRONGNUM_HPP_
#define TAMM_STRONGNUM_HPP_

#include <functional>
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

#define DEBUG_STRONGNUM

/**
 * @brief Return input value in the desired target type after possibly checking
 * for overflow/underflow.
 *
 * @tparam Target Desired output type
 * @tparam Source Input type
 * @param s Value being type cast
 * @return Value after type casting
 */
template<typename Target, typename Source,
         typename = std::enable_if_t<std::is_arithmetic<Source>::value>,
         typename = std::enable_if_t<std::is_arithmetic<Target>::value>,
         typename = std::enable_if_t<!std::is_same<Target, Source>::value>>
constexpr Target checked_cast(Source s) {
#if defined(DEBUG_STRONGNUM)
    auto r = static_cast<Target>(s);
    assert(static_cast<Source>(r) == s);
    return r;
#else
    return static_cast<Target>(s);
#endif
}

/**
 * @brief Trivial checked cast when casting to the same type
 *
 * @tparam T Value type
 * @param s input value
 * @return same input value
 */
template<typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value>>
constexpr T checked_cast(T s) {
    return s;
}

// template <typename Target, typename Source> Target strongnum_cast(Source s) {
//   return checked_cast<Target>(s);
// }

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
 * @tparam Space Unique type name
 * @tparam T Numeric typed being wrapper
 */
template<typename Space, typename T>
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

    StrongNum()                           = default;
    StrongNum(const StrongNum<Space, T>&) = default;
    StrongNum& operator=(const StrongNum<Space, T>&) = default;
    ~StrongNum()                                     = default;

    template<typename T2,
             typename = std::enable_if_t<std::is_arithmetic<T2>::value>,
             typename = std::enable_if_t<std::is_convertible<T2, T>::value>>
    StrongNum(const T2 v1) : v{checked_cast<T>(v1)} {}

    template<typename T2,
             typename = std::enable_if_t<std::is_arithmetic<T2>::value>>
    NumType& operator=(T2 t) {
        v = checked_cast<T>(t);
        return *this;
    }

    template<typename T2,
             typename = std::enable_if_t<std::is_arithmetic<T2>::value>>
    NumType& operator+=(T2 t) {
        v += checked_cast<T>(t);
        return *this;
    }

    NumType& operator+=(NumType d) {
        v += d.v;
        return *this;
    }

    template<typename T2,
             typename = std::enable_if_t<std::is_arithmetic<T2>::value>>
    NumType& operator-=(T2 t) {
        v -= checked_cast<T>(t);
        return *this;
    }

    NumType& operator-=(NumType d) {
        v -= d.v;
        return *this;
    }

    template<typename T2,
             typename = std::enable_if_t<std::is_arithmetic<T2>::value>>
    NumType& operator*=(T2 t) {
        v *= checked_cast<T>(t);
        return *this;
    }

    NumType& operator*=(NumType d) {
        v *= d.v;
        return *this;
    }

    template<typename T2,
             typename = std::enable_if_t<std::is_arithmetic<T2>::value>>
    NumType& operator/=(T2 t) {
        v /= checked_cast<T>(t);
        return *this;
    }

    NumType& operator/=(NumType d) {
        v /= d.v;
        return *this;
    }

    template<typename T2,
             typename = std::enable_if_t<std::is_arithmetic<T2>::value>>
    NumType& operator^=(T2 t) {
        v ^= checked_cast<T>(t);
        return *this;
    }

    NumType& operator^=(NumType d) {
        v ^= d.v;
        return *this;
    }

    NumType& operator++() {
        v += 1;
        return *this;
    }

    NumType operator++(int) {
        NumType ret{*this};
        v += 1;
        return ret;
    }

    NumType& operator--() {
        v -= 1;
        return *this;
    }

    NumType operator--(int) {
        NumType ret{*this};
        v -= 1;
        return ret;
    }

    NumType operator+(NumType d) const { return v + d.v; }
    NumType operator-(NumType d) const { return v - d.v; }
    NumType operator*(NumType d) const { return v * d.v; }
    NumType operator/(NumType d) const { return v / d.v; }
    NumType operator%(NumType d) const { return v % d.v; }

    template<typename T2,
             typename = std::enable_if_t<std::is_arithmetic<T2>::value>>
    NumType operator+(T2 t) const {
        return v + checked_cast<T>(t);
    }

    template<typename T2,
             typename = std::enable_if_t<std::is_arithmetic<T2>::value>>
    NumType operator-(T2 t) const {
        return v - checked_cast<T>(t);
    }

    template<typename T2,
             typename = std::enable_if_t<std::is_arithmetic<T2>::value>>
    NumType operator*(T2 t) const {
        return v * checked_cast<T>(t);
    }

    template<typename T2,
             typename = std::enable_if_t<std::is_arithmetic<T2>::value>>
    NumType operator/(T2 t) const {
        return v / checked_cast<T>(t);
    }

    template<typename T2,
             typename = std::enable_if_t<std::is_arithmetic<T2>::value>>
    NumType operator%(T2 t) const {
        return v % checked_cast<T>(t);
    }

    bool operator==(NumType d) const { return v == d.v; }
    bool operator!=(NumType d) const { return v != d.v; }
    bool operator>=(NumType d) const { return v >= d.v; }
    bool operator<=(NumType d) const { return v <= d.v; }
    bool operator>(NumType d) const { return v > d.v; }
    bool operator<(NumType d) const { return v < d.v; }

    template<typename T2>
    bool operator==(T2 t) const {
        return v == checked_cast<T>(t);
    }

    template<typename T2>
    bool operator!=(T2 t) const {
        return v != checked_cast<T>(t);
    }

    template<typename T2>
    bool operator>=(T2 t) const {
        return v >= checked_cast<T>(t);
    }

    template<typename T2>
    bool operator<=(T2 t) const {
        return v <= checked_cast<T>(t);
    }

    template<typename T2>
    bool operator>(T2 t) const {
        return v > checked_cast<T>(t);
    }

    template<typename T2>
    bool operator<(T2 t) const {
        return v < checked_cast<T>(t);
    }

    T value() const { return v; }
    T& value() { return v; }

    //   template <typename T1> T1 value() const { return checked_cast<T1>(v); }

private:
    T v; /**< Value wrapped by this object */
};

template<typename Space1, typename T1, typename Space2, typename T2>
inline StrongNum<Space1, T1> strongnum_cast(StrongNum<Space2, T2> v2) {
    return {checked_cast<T1>(v2.value())};
}

template<typename Space, typename Type, typename T>
inline bool operator==(T val, StrongNum<Space, Type> snum) {
    return snum == val;
}

template<typename Space, typename Type, typename T>
inline bool operator!=(T val, StrongNum<Space, Type> snum) {
    return snum != val;
}

template<typename Space, typename Type, typename T>
inline bool operator>=(T val, StrongNum<Space, Type> snum) {
    return snum <= val;
}

template<typename Space, typename Type, typename T>
inline bool operator<=(T val, StrongNum<Space, Type> snum) {
    return snum >= val;
}

template<typename Space, typename Type, typename T>
inline bool operator>(T val, StrongNum<Space, Type> snum) {
    return snum < val;
}

template<typename Space, typename Type, typename T>
inline bool operator<(T val, StrongNum<Space, Type> snum) {
    return snum > val;
}

template<typename Space, typename Type, typename T>
StrongNum<Space, Type> operator-(T value, StrongNum<Space, Type> snum) {
    return checked_cast<Type>(value) - snum;
}

template<typename Space, typename Type, typename T>
StrongNum<Space, Type> operator*(T value, StrongNum<Space, Type> snum) {
    return checked_cast<Type>(value) * snum;
}

template<typename Space, typename Type, typename T>
StrongNum<Space, Type> operator/(T value, StrongNum<Space, Type> snum) {
    return checked_cast<Type>(value) / snum;
}

template<typename Space, typename Type, typename T>
StrongNum<Space, Type> operator+(T value, StrongNum<Space, Type> snum) {
    return checked_cast<Type>(value) + snum;
}

template<typename Space, typename Type, typename T>
StrongNum<Space, Type> operator%(T value, StrongNum<Space, Type> snum) {
    return checked_cast<Type>(value) % snum;
}

template<typename S, typename T>
inline std::ostream& operator<<(std::ostream& os, StrongNum<S, T> s) {
    return os << s.value();
}

template<typename S, typename T>
inline std::istream& operator>>(std::istream& is, StrongNum<S, T>& s) {
    is >> s.value();
    return is;
}

} // namespace tamm

#endif // TAMM_STRONGNUM_HPP_
