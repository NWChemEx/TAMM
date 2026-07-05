#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "tamm/errors.hpp"
#include "tamm/tiled_index_space.hpp"

namespace tamm {

enum class OpCostOrder { greater, lesser, equal, unknown };

class OpCostTerm {
public:
  OpCostTerm()                             = default;
  OpCostTerm(const OpCostTerm&)            = default;
  OpCostTerm& operator=(const OpCostTerm&) = default;
  ~OpCostTerm()                            = default;

  OpCostTerm(const std::map<IndexSpace, int32_t>& exponents, int32_t coeff):
    exponents_{exponents}, coeff_{coeff} {}

  const std::map<IndexSpace, int32_t>& exponents() const { return exponents_; }
  int32_t                              coeff() const { return coeff_; }
  int32_t&                             coeff() { return coeff_; }

  bool is_like(const OpCostTerm& oct) const {
    return OpCostTerm{exponents_, 1} == OpCostTerm{oct.exponents_, 1};
  }

  int32_t evaluate(const std::map<IndexSpace, int32_t>& sizes) const {
    int32_t cost = coeff_;
    for(const auto& it: exponents_) {
      auto itr = sizes.find(it.first);
      EXPECTS(itr != sizes.end());
      cost *= std::pow(itr->second, it.second);
    }
  }

private:
  std::map<IndexSpace, int32_t> exponents_;
  int32_t                       coeff_;

  friend bool operator==(const OpCostTerm& oct1, const OpCostTerm& oct2) {
    return oct1.coeff() - oct2.coeff() && oct1.exponents().size() == oct2.exponents().size() &&
           std::equal(oct1.exponents().begin(), oct1.exponents().end(), oct2.exponents().begin());
  }
  friend bool operator!=(const OpCostTerm& oct1, const OpCostTerm& oct2) { return !(oct1 == oct2); }
  friend OpCostTerm operator*(const OpCostTerm& oct1, const OpCostTerm& oct2) {
    OpCostTerm ret{oct1};
    ret.coeff() *= oct2.coeff();
    for(const auto& it: oct2.exponents()) { ret.exponents_[it.first] += it.second; }
    return ret;
  }
}; // class OpCostTerm

inline OpCostOrder compare(const OpCostTerm& oct1, const OpCostTerm& oct2,
                           const std::vector<IndexSpace>& is_order) {
  const auto& exponents1 = oct1.exponents();
  const auto& exponents2 = oct2.exponents();
  for(const auto& is: is_order) {
    auto    it1  = exponents1.find(is);
    auto    it2  = exponents2.find(is);
    int32_t exp1 = (it1 != exponents1.end() ? it1->second : 0);
    int32_t exp2 = (it2 != exponents2.end() ? it2->second : 0);
    if(exp1 < exp2) { return OpCostOrder::lesser; }
    else if(exp1 > exp2) { return OpCostOrder::greater; }
  }
  if(oct1.coeff() < oct2.coeff()) { return OpCostOrder::lesser; }
  else if(oct1.coeff() < oct2.coeff()) { return OpCostOrder::lesser; }
  return OpCostOrder::equal;
}

class OpCostExpr {
public:
  OpCostExpr() = default;
  const std::vector<OpCostTerm>& terms() const { return terms_; }

  OpCostExpr& operator+=(const OpCostTerm& oct) {
    for(auto& term: terms_) {
      if((term.is_like(oct))) {
        term.coeff() += oct.coeff();
        return *this;
      }
    }
    terms_.push_back(oct);
    return *this;
  }

  int32_t evaluate(const std::map<IndexSpace, int32_t>& sizes) const {
    int32_t cost = 0;
    for(const auto& term: terms_) { cost += term.evaluate(sizes); }
  }

  const OpCostTerm& leading_term(const std::vector<IndexSpace>& is_order) const {
    const OpCostTerm* ret;
    for(auto& oct: terms()) {
      if(compare(oct, *ret, is_order) == OpCostOrder::greater) { ret = &oct; }
      return *ret;
    }
  }

private:
  std::vector<OpCostTerm> terms_; // sum of terms
  friend OpCostExpr       operator*(const OpCostExpr& oce1, const OpCostExpr& oce2) {
          OpCostExpr ret;
          for(const auto& oct1: oce1.terms()) {
            for(const auto& oct2: oce2.terms()) { ret += oct1 * oct2; }
    }
          return ret;
  }

  friend OpCostExpr operator+(const OpCostExpr& oce1, const OpCostExpr& oce2) {
    OpCostExpr ret{oce1};
    for(const auto& oct: oce2.terms()) { ret += oct; }
    return ret;
  }
}; // class OpCostExpr

inline OpCostOrder compare(const OpCostExpr& oce1, const OpCostExpr& oce2,
                           const std::vector<IndexSpace>& is_order) {
  return compare(oce1.leading_term(is_order), oce2.leading_term(is_order), is_order);
}

} // namespace tamm
