#ifndef TAMM_RANGE_HPP_
#define TAMM_RANGE_HPP_

#include "tamm/types.hpp"
#include <algorithm>
#include <map>
#include <tuple>

namespace tamm {

/**
 * @brief Helper methods for Range based constructors. 
 * A range is represented by a triplet [lo,hi,step] and includes all elements 
 * i in lo, lo+step, lo+2*step, ..., hi, but excluding ih.
 *        For now we are using a simple range representation
 *        We will use range constructs from Utilities repo.
 *
 * @todo Possibly replace with Range class in Utilities repo.
 */
class Range {
public:
    /**
     *  @brief Default constructor
     */
    Range() = default;

    // Copy/Move Ctors and Assignment Operators
    Range(Range&&)      = default;
    Range(const Range&) = default;
    Range& operator=(Range&&) = default;
    Range& operator=(const Range&) = default;

    /**
     * @brief Default destructor
     */
    ~Range() = default;

    /**
     * @brief Range constructor
     * 
     * @param [in] lo Low end of the range
     * @param [in] hi High end of the range
     * @param [in] step Step size
     * 
     * @pre @param step > 0 
     * @pre @param lo <= @param hi
     * 
     * @todo Make ranges work with negative step sizes
     */
    Range(Index lo, Index hi, Index step = 1) 
        : lo_{lo}, hi_{hi}, step_{step} {
            EXPECTS(step > 0);
            EXPECTS(lo_ <= hi_);
        }

    /**
     * @brief Accessor for low end of range
     * 
     * @return Low end of range
     */
    constexpr Index lo() const { return lo_; }

    /**
     * @brief Accessor for high end of range
     * 
     * @return High end of range
     */
    constexpr Index hi() const { return hi_; }

    /**
     * @brief Accessor for range's step size
     * 
     * @return Step size
     */
    constexpr Index step() const { return step_; }

    /**
     * @brief Method for checking if a given index value
     *        is in this range
     *
     * @param [in] idx Index value being checked for
     * @return true if range includes index @param idx
     */
    constexpr bool contains(Index idx) const {
        return idx >= lo_ && idx < hi_ && (idx-lo_)%step_ == 0;
        // if(idx == lo_) { return true; }

        // if(idx < lo_ && hi_ <= idx) { return false; }

        // if(step_ > 1) { return ((idx - lo_) % step_ == 0); }

        // return true;
    }

    /**
     * @brief Method for checking disjointness of two ranges
     *
     * @param [in] rhs input range for checking disjointedness
     * 
     * @return true if ranges are disjoint
     * 
     * @todo Recheck this algorithm and cite a reference
     */
    bool is_disjoint_with(const Range& rhs) const {
        if(lo_ == rhs.lo()) { return false; }
        if(rhs.hi() <= lo_ && hi_ <= rhs.lo()) { return true; }

        if(step_ == rhs.step()) {
            return (lo_ % step_ != rhs.lo() % step_);
        } else { // if ranges overlap and steps are different
            Index inter_hi = hi_ <= rhs.hi() ? hi_ : rhs.hi();
            Index inter_lo = lo_ >= rhs.lo() ? lo_ : rhs.lo();

            Index startA =
              inter_lo + ((lo_ - inter_lo) % step_ + step_) % step_;
            Index startB =
              inter_lo +
              ((rhs.lo() - inter_lo) % rhs.step() + rhs.step()) % rhs.step();
            if(startA >= inter_hi || startB >= inter_hi) { return true; }
            Index offset = startB - startA;
            Index gcd, x, y;
            std::tie(gcd, x, y) = extended_gcd(step_, rhs.step());
            Index interval_     = step_ / gcd;
            Index interval_rhs  = rhs.step() / gcd;
            Index step          = interval_ * interval_rhs * gcd;
            if(offset % gcd != 0) {
                return true;
            } else {
                Index crt    = (offset * interval_ * (x % interval_rhs)) % step;
                Index filler = 0;
                Index gap    = offset - crt;
                filler      = gap % step == 0 ? gap : ((gap / step) + 1) * step;
                Index start = startA + crt + filler;
                return !(start < inter_hi && start >= inter_lo);
            }
        }
        return true;
    }

protected:
    Index lo_; /**< Low end of range */
    Index hi_; /**< High end of range */
    Index step_; /**< step size for the range */

private:
    /**
     * @brief Euclid's extended gcd for is_disjoint_with method
     *        - ax + by = gcd(a,b)
     *
     * @todo Move to a separate header/folder for utility functions like this
     *
     * @param [in] a first number for calculating gcd
     * @param [in] b second number for calculating gcd
     * 
     * @return a tuple for gcd, x and y coefficients
     */
    std::tuple<int, int, int> extended_gcd(int a, int b) const {
        if(a == 0) { return std::make_tuple(b, 0, 1); }

        int gcd, x, y;
        // unpack tuple  returned by function into variables
        std::tie(gcd, x, y) = extended_gcd(b % a, a);

        return std::make_tuple(gcd, (y - (b / a) * x), x);
    }
}; // class Range

/**
 * @brief Range constructor with low, high and step size
 *
 * @param [in] lo Low end of the range
 * @param [in] hi High end of the range
 * @param [in] step Step size
 * 
 * @return Constructed Range object
 */
static inline Range range(Index lo, Index hi, Index step = 1) {
    return Range{lo, hi, step};
}

/**
 * @brief Construct a range object from 0 to a given number
 * 
 * @param [in] count Upper bound on range
 * 
 * @return range with low 0, high @param count, and step 1
 *
 */
static inline Range range(Index count) { return range(Index{0}, count); }

/**
 * @brief Helper method for constructing IndexVector for a given Range
 *
 * @param [in] range a Range type argument
 * @returns an IndexVector for the corresponding range
 */
static inline IndexVector construct_index_vector(const Range& range) {
    IndexVector ret;
    for(Index i = range.lo(); i < range.hi(); i += range.step()) {
        ret.push_back(i);
    }

    return ret;
}

/**
 * @brief Type definition for the map between range names
 *        and corresponding set of ranges (e.g. "occ", "virt")
 *
 */
using NameToRangeMap = std::map<std::string, const std::vector<Range>>;

template<typename AttributeType>
using AttributeToRangeMap = std::map<AttributeType, std::vector<Range>>;

} // namespace tamm

#endif // TAMM_RANGE_HPP_