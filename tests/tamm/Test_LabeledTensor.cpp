#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include <tamm/tamm.hpp>
#include <iostream>

using namespace tamm;


TEST_CASE("Zero-dimensional tensor") {
  bool failed = false;
  //IndexSpace is {range(10)};
  //TiledIndexSpace tis{is};
  Tensor<double> T1{};
  try {
    auto lt = T1();
  } catch (...) {
    failed = true;
  }
  REQUIRE(!failed);
}

TEST_CASE("Zero-dimensional tensor wrong str count") {
  bool failed = false;
  //IndexSpace is {range(10)};
  //TiledIndexSpace tis{is};
  Tensor<double> T1{};
  try {
    LabeledTensor<double> lt{T1("a")};
  } catch (...) {
    failed = true;
  }
  REQUIRE(failed);
}

TEST_CASE("Zero-dimensional tensor wrong label count") {
  bool failed = false;
  IndexSpace is {range(10)};
  TiledIndexSpace tis{is};
  Tensor<double> T1{};
  try {
    LabeledTensor<double> lt{T1(tis)};
  } catch (...) {
    failed = true;
  }
  REQUIRE(failed);
}

TEST_CASE("One-dimensional tensor") {
  bool failed = false;
  IndexSpace is {range(10)};
  TiledIndexSpace tis{is};
  Tensor<double> T1{tis};
  try {
    auto lt = T1();
  } catch (...) {
    failed = true;
  }
  REQUIRE(!failed);
}

TEST_CASE("One-dimensional tensor with correct string count") {
  bool failed = false;
  IndexSpace is {range(10)};
  TiledIndexSpace tis{is};
  Tensor<double> T1{tis};
  try {
    auto lt = T1("i");
  } catch (...) {
    failed = true;
  }
  REQUIRE(!failed);
}

TEST_CASE("One-dimensional tensor with correct label count") {
  bool failed = false;
  IndexSpace is {range(10)};
  TiledIndexSpace tis{is};
  Tensor<double> T1{tis};
  try {
    auto lt = T1(tis.label());
  } catch (...) {
    failed = true;
  }
  REQUIRE(!failed);
}

TEST_CASE("One-dimensional tensor with wrong string count") {
  bool failed = false;
  IndexSpace is {range(10)};
  TiledIndexSpace tis{is};
  Tensor<double> T1{tis};
  try {
    auto lt = T1("i", "i");
  } catch (...) {
    failed = true;
  }
  REQUIRE(failed);
}

TEST_CASE("One-dimensional tensor with wrong label count") {
  bool failed = false;
  IndexSpace is {range(10)};
  TiledIndexSpace tis{is};
  Tensor<double> T1{tis};
  try {
    auto lt = T1(tis.label(0), tis.label(1));
  } catch (...) {
    failed = true;
  }
  REQUIRE(failed);
}

TEST_CASE("Two-dimensional tensor") {
  bool failed = false;
  IndexSpace is {range(10)};
  TiledIndexSpace tis{is};
  Tensor<double> T1{tis, tis};
  try {
    auto lt = T1();
  } catch (...) {
    failed = true;
  }
  REQUIRE(!failed);
}

TEST_CASE("Two-dimensional tensor with correct string count") {
  bool failed = false;
  IndexSpace is {range(10)};
  TiledIndexSpace tis{is};
  Tensor<double> T1{tis, tis};
  try {
    auto lt = T1("i", "j");
  } catch (...) {
    failed = true;
  }
  REQUIRE(!failed);
}

TEST_CASE("Two-dimensional tensor with correct label count") {
  bool failed = false;
  IndexSpace is {range(10)};
  TiledIndexSpace tis{is};
  Tensor<double> T1{tis, tis};
  try {
    auto lt = T1(tis.label(0), tis.label(1));
  } catch (...) {
    failed = true;
  }
  REQUIRE(!failed);
}

TEST_CASE("Two-dimensional tensor with smaller string count") {
  bool failed = false;
  IndexSpace is {range(10)};
  TiledIndexSpace tis{is};
  Tensor<double> T1{tis, tis};
  try {
    auto lt = T1("i");
  } catch (...) {
    failed = true;
  }
  REQUIRE(failed);
}

TEST_CASE("Two-dimensional tensor with smaller label count") {
  bool failed = false;
  IndexSpace is {range(10)};
  TiledIndexSpace tis{is};
  Tensor<double> T1{tis, tis};
  try {
    auto lt = T1(tis.label(0));
  } catch (...) {
    failed = true;
  }
  REQUIRE(failed);
}

TEST_CASE("Two-dimensional tensor with larger string count") {
  bool failed = false;
  IndexSpace is {range(10)};
  TiledIndexSpace tis{is};
  Tensor<double> T1{tis, tis};
  try {
    auto lt = T1("i", "j", "k");
  } catch (...) {
    failed = true;
  }
  REQUIRE(failed);
}

TEST_CASE("Two-dimensional tensor with larger label count") {
  bool failed = false;
  IndexSpace is {range(10)};
  TiledIndexSpace tis{is};
  Tensor<double> T1{tis, tis};
  try {
    auto lt = T1(tis.label(0), tis.label(1), tis.label(2));
  } catch (...) {
    failed = true;
  }
  REQUIRE(failed);
}

TEST_CASE("Two-dimensional dependent index tensor") {
  bool failed = false;
  IndexSpace is {range(1)};
  TiledIndexSpace tis{is};
  IndexSpace is2{range(5)};
  std::map<IndexVector, IndexSpace> dep_space_relation;
  dep_space_relation[{0}] = is2;
  IndexSpace dis{{tis}, dep_space_relation};
  TiledIndexSpace tdis{dis};

  TiledIndexLabel i, j, a, b;
  std::tie(i,j) = tis.labels<2>("all");
  std::tie(a,b) = tdis.labels<2>("all");

  //some tensor tests. @todo should be in Test_Tensor.cpp
  try {
    Tensor<double> T1{tdis, tdis};
  } catch (...) {
    failed = true;
  }
  REQUIRE(failed);
  failed = false;

  try {
    Tensor<double> T1{a(i), j};
  } catch (...) {
    failed = true;
  }
  REQUIRE(failed);
  failed = false;

  try {
    Tensor<double> T1{i(a), j};
  } catch (...) {
    failed = true;
  }
  REQUIRE(!failed);
  failed = false;


  //labeled tensor tests follow
  Tensor<double> T1({a(i), i});
  try {
    auto lt = T1();
  } catch (...) {
    failed = true;
  }
  REQUIRE(!failed);
  failed = false;

  try {
    auto lt = T1(b(j), j);
  } catch (...) {
    failed = true;
  }
  REQUIRE(!failed);
  failed = false;

  try {
    auto lt = T1("x");
  } catch (...) {
    failed = true;
  }
  REQUIRE(failed);
  failed = false;

  try {
    auto lt = T1(a);
  } catch (...) {
    failed = true;
  }
  REQUIRE(failed);
  failed = false;

  try {
    auto lt = T1(i);
  } catch (...) {
    failed = true;
  }
  REQUIRE(failed);
  failed = false;

  try {
    auto lt = T1("x", "y");
  } catch (...) {
    failed = true;
  }
  REQUIRE(!failed);
  failed = false;

  try {
    auto lt = T1(a, i);
  } catch (...) {
    failed = true;
  }
  REQUIRE(!failed);
  failed = false;

  try {
    auto lt = T1(i, a(i));
  } catch (...) {
    failed = true;
  }
  REQUIRE(failed);
  failed = false;

  try {
    auto lt = T1(a(i));
  } catch (...) {
    failed = true;
  }
  REQUIRE(failed);
  failed = false;

  try {
    auto lt = T1(i);
  } catch (...) {
    failed = true;
  }
  REQUIRE(failed);
  failed = false;

  try {
    auto lt = T1(i(a));
  } catch (...) {
    failed = true;
  }
  REQUIRE(failed);
  failed = false;

  try {
    auto lt = T1(a(a));
  } catch (...) {
    failed = true;
  }
  REQUIRE(failed);
  failed = false;

  try {
    auto lt = T1(i(i));
  } catch (...) {
    failed = true;
  }
  REQUIRE(failed);
  failed = false;
}

