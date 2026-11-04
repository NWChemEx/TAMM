// Unit tests for TAMM's core value types: BlockSpan, BoundVec and StrongNum.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <tamm/block_span.hpp>
#include <tamm/boundvec.hpp>
#include <tamm/strong_num.hpp>
#include <tamm/types.hpp> // for the concrete aliases Spin/Proc/Offset

#include <numeric>
#include <unordered_map>
#include <vector>

using namespace tamm;

namespace {
constexpr std::size_t CAP = 8;
using BV                  = BoundVec<int, CAP>;

struct ASpace;
using AInt = StrongNum<ASpace, int>;
struct BSpace;
using BLong = StrongNum<BSpace, long>;
struct USpace;
using UInt = StrongNum<USpace, unsigned>;
} // namespace

// tamm::BlockSpan (block_span.hpp):
//  - flat operator[] maps to row-major offset
//  - num_elements(), rank(), extent(d), block_dims(), block_dims_span()
//  - flat_span(), data()/buf() equivalence
//  - make_block_span factory helpers
//  - is_valid()

TEST_CASE("BlockSpan: default is invalid") {
  BlockSpan<double> bs;
  CHECK_FALSE(bs.is_valid());
  CHECK(bs.num_elements() == 0);
}

TEST_CASE("BlockSpan: construction from pointer + dims (vector)") {
  std::vector<double> buf(2 * 3 * 4);
  std::iota(buf.begin(), buf.end(), 0.0);
  std::vector<size_t> dims{2, 3, 4};

  BlockSpan<double> bs{buf.data(), dims};
  CHECK(bs.is_valid());
  CHECK(bs.rank() == 3);
  CHECK(bs.num_elements() == 24);
  CHECK(bs.extent(0) == 2);
  CHECK(bs.extent(1) == 3);
  CHECK(bs.extent(2) == 4);
  CHECK(bs.buf() == buf.data());
  CHECK(bs.data() == buf.data());
}

TEST_CASE("BlockSpan: block_dims() and block_dims_span() report extents") {
  std::vector<float>  buf(6);
  std::vector<size_t> dims{2, 3};
  BlockSpan<float>    bs{buf.data(), dims};

  const auto& bd = bs.block_dims();
  REQUIRE(bd.size() == 2);
  CHECK(bd[0] == 2);
  CHECK(bd[1] == 3);

  auto sp = bs.block_dims_span();
  REQUIRE(sp.size() == 2);
  CHECK(sp[0] == 2);
  CHECK(sp[1] == 3);
}

TEST_CASE("BlockSpan: flat operator[] is row-major over the buffer") {
  // 2x3x4 row-major: element (i,j,k) lives at i*12 + j*4 + k.
  std::vector<int> buf(2 * 3 * 4);
  std::iota(buf.begin(), buf.end(), 0);
  std::vector<size_t> dims{2, 3, 4};
  BlockSpan<int>      bs{buf.data(), dims};

  for(size_t i = 0; i < 2; i++)
    for(size_t j = 0; j < 3; j++)
      for(size_t k = 0; k < 4; k++) {
        const size_t flat = i * 12 + j * 4 + k;
        CHECK(bs[flat] == static_cast<int>(flat));
      }

  // mutation through operator[]
  bs[5] = 999;
  CHECK(buf[5] == 999);

  // const access
  const BlockSpan<int>& cbs = bs;
  CHECK(cbs[5] == 999);
}

TEST_CASE("BlockSpan: flat_span covers the whole block") {
  std::vector<double> buf(12, 1.5);
  std::vector<size_t> dims{3, 4};
  BlockSpan<double>   bs{buf.data(), dims};

  auto sp = bs.flat_span();
  REQUIRE(sp.size() == 12);
  double sum = std::accumulate(sp.begin(), sp.end(), 0.0);
  CHECK(sum == doctest::Approx(18.0));
}

TEST_CASE("BlockSpan: make_block_span factory (initializer list + range)") {
  std::vector<double> buf(6, 0.0);

  auto bs1 = make_block_span<double>(buf.data(), {2, 3});
  CHECK(bs1.rank() == 2);
  CHECK(bs1.num_elements() == 6);

  std::vector<size_t> dims{3, 2};
  auto                bs2 = make_block_span<double>(buf.data(), dims);
  CHECK(bs2.rank() == 2);
  CHECK(bs2.num_elements() == 6);
}

TEST_CASE("BlockSpan: rank-1 and scalar-like (rank-1 size 1)") {
  std::vector<double> buf{42.0};
  std::vector<size_t> dims{1};
  BlockSpan<double>   bs{buf.data(), dims};
  CHECK(bs.rank() == 1);
  CHECK(bs.num_elements() == 1);
  CHECK(bs[0] == 42.0);
}

TEST_CASE("BlockSpan: copy preserves view") {
  std::vector<int>    buf{1, 2, 3, 4};
  std::vector<size_t> dims{2, 2};
  BlockSpan<int>      a{buf.data(), dims};
  BlockSpan<int>      b = a;
  CHECK(b.buf() == a.buf());
  CHECK(b.num_elements() == 4);
  CHECK(b[3] == 4);
}

// tamm::BoundVec (boundvec.hpp):
//  - (count, value) constructor
//  - initializer-list and iterator-pair constructors
//  - insert_back(value), insert_back(first,last), insert_back(count,value)
//  - reverse iterators (rbegin/rend)
//  - data(), cbegin/cend
//  - push_back / pop_back / resize / clear / front / back / operator[]
//  - equality
// Behavior is cross-checked against std::vector where meaningful.

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

// tamm::StrongNum (strong_num.hpp):
//  - implicit construction from arithmetic types (NOT explicit)
//  - defaulted operator== plus operator<=> (so ==, !=, <, <=, >, >= all work)
//  - heterogeneous comparisons against raw arithmetic
//  - arithmetic operators (member and non-member raw-lhs forms)
//  - std::hash usability (unordered_map key)

TEST_CASE("StrongNum: implicit construction and value()") {
  // Implicit conversion from arithmetic must compile (the refactor kept this).
  AInt a = 5; // copy-init from int
  AInt b{7};  // direct-list-init
  CHECK(a.value() == 5);
  CHECK(b.value() == 7);

  // assignment from raw arithmetic
  a = 9;
  CHECK(a.value() == 9);

  // widening/narrowing across underlying types
  BLong c = 40;
  CHECK(c.value() == 40L);
}

TEST_CASE("StrongNum: same-type comparisons (==,!=,<,<=,>,>=)") {
  AInt a{3}, b{3}, c{5};
  CHECK(a == b);
  CHECK_FALSE(a == c);
  CHECK(a != c);
  CHECK_FALSE(a != b);
  CHECK(a < c);
  CHECK(a <= b);
  CHECK(c > a);
  CHECK(c >= a);
  CHECK_FALSE(c < a);
}

TEST_CASE("StrongNum: heterogeneous comparison against raw arithmetic") {
  AInt a{4};
  CHECK(a == 4);
  CHECK(4 == a);
  CHECK(a != 5);
  CHECK(a < 10);
  CHECK(a > 1);
  CHECK(a <= 4);
  CHECK(a >= 4);
}

TEST_CASE("StrongNum: arithmetic (same-type operands)") {
  AInt a{10}, b{3};
  CHECK((a + b).value() == 13);
  CHECK((a - b).value() == 7);
  CHECK((a * b).value() == 30);
  CHECK((a / b).value() == 3);
  CHECK((a % b).value() == 1);
}

TEST_CASE("StrongNum: arithmetic with raw arithmetic operands") {
  AInt a{10};
  CHECK((a + 5).value() == 15);
  CHECK((a - 4).value() == 6);
  CHECK((a * 3).value() == 30);
  CHECK((a / 2).value() == 5);
}

TEST_CASE("StrongNum: non-member raw-lhs arithmetic") {
  BLong a{10};
  CHECK((100 - a).value() == 90L);
  CHECK((5 + a).value() == 15L);
  CHECK((4 * a).value() == 40L);
  CHECK((100 / a).value() == 10L);
}

TEST_CASE("StrongNum: compound assignment") {
  AInt a{10};
  a += AInt{5};
  CHECK(a.value() == 15);
  a -= AInt{3};
  CHECK(a.value() == 12);
  a *= AInt{2};
  CHECK(a.value() == 24);
  a /= AInt{4};
  CHECK(a.value() == 6);
  a += 4; // raw operand
  CHECK(a.value() == 10);
}

TEST_CASE("StrongNum: increment / decrement") {
  AInt a{5};
  CHECK((++a).value() == 6);
  CHECK((a++).value() == 6);
  CHECK(a.value() == 7);
  CHECK((--a).value() == 6);
  CHECK((a--).value() == 6);
  CHECK(a.value() == 5);
}

TEST_CASE("StrongNum: std::hash and use as unordered_map key") {
  std::unordered_map<AInt, std::string> m;
  m[AInt{1}]  = "one";
  m[AInt{2}]  = "two";
  m[AInt{42}] = "answer";
  CHECK(m.at(AInt{1}) == "one");
  CHECK(m.at(AInt{2}) == "two");
  CHECK(m.at(AInt{42}) == "answer");
  CHECK(m.find(AInt{7}) == m.end());
  // equal keys hash equal
  CHECK(std::hash<AInt>{}(AInt{5}) == std::hash<AInt>{}(AInt{5}));
}

TEST_CASE("StrongNum: works with the concrete TAMM aliases") {
  Spin   s1{0}, s2{1};
  Proc   p{3};
  Offset off{100};
  CHECK(s1 != s2);
  CHECK((s1 + s2).value() == 1u);
  CHECK(p.value() == 3);
  CHECK((off + Offset{5}).value() == 105u);
  // implicit-from-arithmetic on the aliases (used widely in TAMM)
  Proc   pz = 0;
  Offset o0 = 0;
  CHECK(pz.value() == 0);
  CHECK(o0.value() == 0u);
}

TEST_CASE("StrongNum: strongnum_cast across spaces") {
  AInt a{7};
  auto b = strongnum_cast<BSpace, long>(a);
  CHECK(b.value() == 7L);
}
