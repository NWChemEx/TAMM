// Unit tests for tamm::BoundVec (boundvec.hpp).
//
// Previously uncovered.  Guards the restored/refactored API:
//  - (count, value) constructor
//  - initializer-list and iterator-pair constructors
//  - insert_back(value), insert_back(first,last), insert_back(count,value)
//  - reverse iterators (rbegin/rend)
//  - data(), cbegin/cend
//  - push_back / pop_back / resize / clear / front / back / operator[]
//  - equality
// Behavior is cross-checked against std::vector where meaningful.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <tamm/boundvec.hpp>

#include <numeric>
#include <vector>

using namespace tamm;

namespace {
constexpr std::size_t CAP = 8;
using BV                  = BoundVec<int, CAP>;
} // namespace

TEST_CASE("BoundVec: default construction is empty") {
  BV v;
  CHECK(v.size() == 0);
  CHECK(v.empty());
  CHECK(BV::max_size() == CAP);
}

TEST_CASE("BoundVec: (count, value) constructor") {
  BV v(3, 7);
  CHECK(v.size() == 3);
  for(auto x: v) CHECK(x == 7);

  // default value form
  BV z(4);
  CHECK(z.size() == 4);
  for(auto x: z) CHECK(x == 0);
}

TEST_CASE("BoundVec: initializer-list constructor") {
  BV v{1, 2, 3, 4};
  CHECK(v.size() == 4);
  CHECK(v[0] == 1);
  CHECK(v[3] == 4);
}

TEST_CASE("BoundVec: iterator-pair constructor (disambiguated from count,value)") {
  std::vector<int> src{10, 20, 30};
  BV               v(src.begin(), src.end());
  CHECK(v.size() == 3);
  CHECK(v[0] == 10);
  CHECK(v[2] == 30);
}

TEST_CASE("BoundVec: push_back / pop_back / back / front") {
  BV v;
  v.push_back(1);
  v.push_back(2);
  v.push_back(3);
  CHECK(v.size() == 3);
  CHECK(v.front() == 1);
  CHECK(v.back() == 3);
  v.pop_back();
  CHECK(v.size() == 2);
  CHECK(v.back() == 2);
}

TEST_CASE("BoundVec: resize grows (value-init) and shrinks") {
  BV v{1, 2, 3};
  v.resize(5);
  CHECK(v.size() == 5);
  CHECK(v[3] == 0);
  CHECK(v[4] == 0);
  v.resize(2);
  CHECK(v.size() == 2);
  CHECK(v[1] == 2);
}

TEST_CASE("BoundVec: clear") {
  BV v{1, 2, 3};
  v.clear();
  CHECK(v.empty());
  CHECK(v.size() == 0);
}

TEST_CASE("BoundVec: insert_back single value returns iterator") {
  BV   v{1, 2};
  auto it = v.insert_back(3);
  CHECK(*it == 3);
  CHECK(v.size() == 3);
  CHECK(v.back() == 3);
}

TEST_CASE("BoundVec: insert_back(first,last) range append") {
  BV               v{1};
  std::vector<int> more{2, 3, 4};
  v.insert_back(more.begin(), more.end());
  CHECK(v.size() == 4);
  CHECK(v[0] == 1);
  CHECK(v[1] == 2);
  CHECK(v[3] == 4);
}

TEST_CASE("BoundVec: insert_back(count,value) append") {
  BV v{9};
  v.insert_back(std::size_t(3), 5);
  CHECK(v.size() == 4);
  CHECK(v[0] == 9);
  CHECK(v[1] == 5);
  CHECK(v[2] == 5);
  CHECK(v[3] == 5);
}

TEST_CASE("BoundVec: reverse iterators") {
  BV               v{1, 2, 3, 4};
  std::vector<int> reversed(v.rbegin(), v.rend());
  CHECK(reversed.size() == 4);
  CHECK(reversed[0] == 4);
  CHECK(reversed[1] == 3);
  CHECK(reversed[2] == 2);
  CHECK(reversed[3] == 1);
}

TEST_CASE("BoundVec: data() points at contiguous storage") {
  BV   v{5, 6, 7};
  int* p = v.data();
  CHECK(p[0] == 5);
  CHECK(p[1] == 6);
  CHECK(p[2] == 7);
  p[1] = 60; // mutable
  CHECK(v[1] == 60);

  const BV&  cv = v;
  const int* cp = cv.data();
  CHECK(cp[0] == 5);
}

TEST_CASE("BoundVec: cbegin/cend and range-for") {
  BV  v{2, 4, 6};
  int sum = std::accumulate(v.cbegin(), v.cend(), 0);
  CHECK(sum == 12);
}

TEST_CASE("BoundVec: equality") {
  BV a{1, 2, 3};
  BV b{1, 2, 3};
  BV c{1, 2, 4};
  BV d{1, 2};
  CHECK(a == b);
  CHECK_FALSE(a == c);
  CHECK_FALSE(a == d);
}

TEST_CASE("BoundVec: fill up to capacity") {
  BV v;
  for(std::size_t i = 0; i < CAP; ++i) v.push_back(static_cast<int>(i));
  CHECK(v.size() == CAP);
  CHECK(v.back() == static_cast<int>(CAP - 1));
}
