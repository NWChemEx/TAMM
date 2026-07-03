// Unit tests for tamm::StrongNum (strong_num.hpp).
//
// Previously uncovered.  Guards the C++20 refactor of StrongNum:
//  - implicit construction from arithmetic types (NOT explicit)
//  - defaulted operator== plus operator<=> (so ==, !=, <, <=, >, >= all work)
//  - heterogeneous comparisons against raw arithmetic
//  - arithmetic operators (member and non-member raw-lhs forms)
//  - std::hash usability (unordered_map key)

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <tamm/strong_num.hpp>
#include <tamm/types.hpp> // for the concrete aliases Spin/Proc/Offset

#include <unordered_map>
#include <vector>

using namespace tamm;

namespace {
struct ASpace;
using AInt = StrongNum<ASpace, int>;
struct BSpace;
using BLong = StrongNum<BSpace, long>;
struct USpace;
using UInt = StrongNum<USpace, unsigned>;
} // namespace

TEST_CASE("StrongNum: implicit construction and value()") {
  // Implicit conversion from arithmetic must compile (the refactor kept this).
  AInt a = 5;      // copy-init from int
  AInt b{7};       // direct-list-init
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
  m[AInt{1}] = "one";
  m[AInt{2}] = "two";
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
  Proc   pz  = 0;
  Offset o0  = 0;
  CHECK(pz.value() == 0);
  CHECK(o0.value() == 0u);
}

TEST_CASE("StrongNum: strongnum_cast across spaces") {
  AInt a{7};
  auto b = strongnum_cast<BSpace, long>(a);
  CHECK(b.value() == 7L);
}
